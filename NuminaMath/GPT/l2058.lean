import Mathlib

namespace NUMINAMATH_GPT_neg_exists_x_sq_lt_one_eqv_forall_x_real_x_leq_neg_one_or_x_geq_one_l2058_205845

theorem neg_exists_x_sq_lt_one_eqv_forall_x_real_x_leq_neg_one_or_x_geq_one :
  ¬(∃ x : ℝ, x^2 < 1) ↔ ∀ x : ℝ, x ≤ -1 ∨ x ≥ 1 := 
by 
  sorry

end NUMINAMATH_GPT_neg_exists_x_sq_lt_one_eqv_forall_x_real_x_leq_neg_one_or_x_geq_one_l2058_205845


namespace NUMINAMATH_GPT_warriors_truth_tellers_l2058_205835

/-- There are 33 warriors. Each warrior is either a truth-teller or a liar, 
    with only one favorite weapon: a sword, a spear, an axe, or a bow. 
    They were asked four questions, and the number of "Yes" answers to the 
    questions are 13, 15, 20, and 27 respectively. Prove that the number of 
    warriors who always tell the truth is 12. -/
theorem warriors_truth_tellers
  (warriors : ℕ) (truth_tellers : ℕ)
  (yes_to_sword : ℕ) (yes_to_spear : ℕ)
  (yes_to_axe : ℕ) (yes_to_bow : ℕ)
  (h1 : warriors = 33)
  (h2 : yes_to_sword = 13)
  (h3 : yes_to_spear = 15)
  (h4 : yes_to_axe = 20)
  (h5 : yes_to_bow = 27)
  (h6 : yes_to_sword + yes_to_spear + yes_to_axe + yes_to_bow = 75) :
  truth_tellers = 12 := by
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_warriors_truth_tellers_l2058_205835


namespace NUMINAMATH_GPT_volume_of_cube_l2058_205895

theorem volume_of_cube (d : ℝ) (h : d = 5 * Real.sqrt 3) : ∃ (V : ℝ), V = 125 := by
  sorry

end NUMINAMATH_GPT_volume_of_cube_l2058_205895


namespace NUMINAMATH_GPT_b_share_l2058_205853

-- Definitions based on the conditions
def salary (a b c d : ℕ) : Prop :=
  ∃ x : ℕ, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x ∧ d = 6 * x

def condition (d c : ℕ) : Prop :=
  d = c + 700

-- Proof problem based on the correct answer
theorem b_share (a b c d : ℕ) (x : ℕ) (salary_cond : salary a b c d) (cond : condition d c) :
  b = 1050 := by
  sorry

end NUMINAMATH_GPT_b_share_l2058_205853


namespace NUMINAMATH_GPT_range_of_a_l2058_205849

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x + Real.log x - (x^2 / (x - Real.log x))

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0) ↔
  1 < a ∧ a < (Real.exp 1) / (Real.exp 1 - 1) - 1 / Real.exp 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2058_205849


namespace NUMINAMATH_GPT_never_attains_95_l2058_205894

def dihedral_angle_condition (α β : ℝ) : Prop :=
  0 < α ∧ 0 < β ∧ α + β < 90

theorem never_attains_95 (α β : ℝ) (h : dihedral_angle_condition α β) :
  α + β ≠ 95 :=
by
  sorry

end NUMINAMATH_GPT_never_attains_95_l2058_205894


namespace NUMINAMATH_GPT_Chrysler_Building_floors_l2058_205882

variable (C L : ℕ)

theorem Chrysler_Building_floors :
  (C = L + 11) → (C + L = 35) → (C = 23) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_Chrysler_Building_floors_l2058_205882


namespace NUMINAMATH_GPT_factor_square_difference_l2058_205802

theorem factor_square_difference (t : ℝ) : t^2 - 121 = (t - 11) * (t + 11) := 
  sorry

end NUMINAMATH_GPT_factor_square_difference_l2058_205802


namespace NUMINAMATH_GPT_total_texts_sent_is_97_l2058_205862

def textsSentOnMondayAllison := 5
def textsSentOnMondayBrittney := 5
def textsSentOnMondayCarol := 5

def textsSentOnTuesdayAllison := 15
def textsSentOnTuesdayBrittney := 10
def textsSentOnTuesdayCarol := 12

def textsSentOnWednesdayAllison := 20
def textsSentOnWednesdayBrittney := 18
def textsSentOnWednesdayCarol := 7

def totalTextsAllison := textsSentOnMondayAllison + textsSentOnTuesdayAllison + textsSentOnWednesdayAllison
def totalTextsBrittney := textsSentOnMondayBrittney + textsSentOnTuesdayBrittney + textsSentOnWednesdayBrittney
def totalTextsCarol := textsSentOnMondayCarol + textsSentOnTuesdayCarol + textsSentOnWednesdayCarol

def totalTextsAllThree := totalTextsAllison + totalTextsBrittney + totalTextsCarol

theorem total_texts_sent_is_97 : totalTextsAllThree = 97 := by
  sorry

end NUMINAMATH_GPT_total_texts_sent_is_97_l2058_205862


namespace NUMINAMATH_GPT_stream_current_rate_l2058_205879

theorem stream_current_rate (r c : ℝ) (h1 : 20 / (r + c) + 6 = 20 / (r - c)) (h2 : 20 / (3 * r + c) + 1.5 = 20 / (3 * r - c)) 
  : c = 3 :=
  sorry

end NUMINAMATH_GPT_stream_current_rate_l2058_205879


namespace NUMINAMATH_GPT_intercept_x_parallel_lines_l2058_205855

theorem intercept_x_parallel_lines (m : ℝ) 
    (line_l : ∀ x y : ℝ, y + m * (x + 1) = 0) 
    (parallel : ∀ x y : ℝ, y * m - (2 * m + 1) * x = 1) : 
    ∃ x : ℝ, x + 1 = -1 :=
by
  sorry

end NUMINAMATH_GPT_intercept_x_parallel_lines_l2058_205855


namespace NUMINAMATH_GPT_distinct_remainders_l2058_205834

theorem distinct_remainders (p : ℕ) (a : Fin p → ℤ) (hp : Nat.Prime p) :
  ∃ k : ℤ, (Finset.univ.image (fun i : Fin p => (a i + i * k) % p)).card ≥ ⌈(p / 2 : ℚ)⌉ :=
sorry

end NUMINAMATH_GPT_distinct_remainders_l2058_205834


namespace NUMINAMATH_GPT_servings_in_box_l2058_205876

def totalCereal : ℕ := 18
def servingSize : ℕ := 2

theorem servings_in_box : totalCereal / servingSize = 9 := by
  sorry

end NUMINAMATH_GPT_servings_in_box_l2058_205876


namespace NUMINAMATH_GPT_triangle_problem_l2058_205839

/-- 
Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively, 
if b = 2 and 2*b*cos B = a*cos C + c*cos A,
prove that B = π/3 and find the maximum area of ΔABC.
-/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (h1 : b = 2) (h2 : 2 * b * Real.cos B = a * Real.cos C + c * Real.cos A) :
  B = Real.pi / 3 ∧
  (∃ (max_area : ℝ), max_area = Real.sqrt 3 ∧ max_area = (1/2) * a * c * Real.sin B) :=
by
  sorry

end NUMINAMATH_GPT_triangle_problem_l2058_205839


namespace NUMINAMATH_GPT_shadow_length_to_time_l2058_205857

theorem shadow_length_to_time (shadow_length_inches : ℕ) (stretch_rate_feet_per_hour : ℕ) (inches_per_foot : ℕ) 
                              (shadow_start_time : ℕ) :
  shadow_length_inches = 360 → stretch_rate_feet_per_hour = 5 → inches_per_foot = 12 → shadow_start_time = 0 →
  (shadow_length_inches / inches_per_foot) / stretch_rate_feet_per_hour = 6 := by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_shadow_length_to_time_l2058_205857


namespace NUMINAMATH_GPT_find_roots_of_equation_l2058_205836

theorem find_roots_of_equation
  (a b c d x : ℝ)
  (h1 : a + d = 2015)
  (h2 : b + c = 2015)
  (h3 : a ≠ c)
  (h4 : (x - a) * (x - b) = (x - c) * (x - d)) :
  x = 1007.5 :=
by
  sorry

end NUMINAMATH_GPT_find_roots_of_equation_l2058_205836


namespace NUMINAMATH_GPT_positive_integers_a_2014_b_l2058_205851

theorem positive_integers_a_2014_b (a : ℕ) :
  (∃! b : ℕ, 2 ≤ a / b ∧ a / b ≤ 5) → a = 6710 ∨ a = 6712 ∨ a = 6713 :=
by
  sorry

end NUMINAMATH_GPT_positive_integers_a_2014_b_l2058_205851


namespace NUMINAMATH_GPT_math_problem_l2058_205892

theorem math_problem :
  2.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 5000 := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l2058_205892


namespace NUMINAMATH_GPT_farmer_trees_l2058_205803

theorem farmer_trees (x n m : ℕ) 
  (h1 : x + 20 = n^2) 
  (h2 : x - 39 = m^2) : 
  x = 880 := 
by sorry

end NUMINAMATH_GPT_farmer_trees_l2058_205803


namespace NUMINAMATH_GPT_Tim_has_7_times_more_l2058_205823

-- Define the number of Dan's violet balloons
def Dan_violet_balloons : ℕ := 29

-- Define the number of Tim's violet balloons
def Tim_violet_balloons : ℕ := 203

-- Prove that the ratio of Tim's balloons to Dan's balloons is 7
theorem Tim_has_7_times_more (h : Tim_violet_balloons = 7 * Dan_violet_balloons) : 
  Tim_violet_balloons = 7 * Dan_violet_balloons := 
by {
  sorry
}

end NUMINAMATH_GPT_Tim_has_7_times_more_l2058_205823


namespace NUMINAMATH_GPT_value_range_of_f_l2058_205807

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then 2 * x - x^2
  else if -4 ≤ x ∧ x < 0 then x^2 + 6 * x
  else 0

theorem value_range_of_f : Set.range f = {y : ℝ | -9 ≤ y ∧ y ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_value_range_of_f_l2058_205807


namespace NUMINAMATH_GPT_num_ways_distribute_balls_l2058_205886

-- Definition of the combinatorial problem
def indistinguishableBallsIntoBoxes : ℕ := 11

-- Main theorem statement
theorem num_ways_distribute_balls : indistinguishableBallsIntoBoxes = 11 := by
  sorry

end NUMINAMATH_GPT_num_ways_distribute_balls_l2058_205886


namespace NUMINAMATH_GPT_smallest_lcm_l2058_205859

theorem smallest_lcm (a b : ℕ) (h₁ : 1000 ≤ a ∧ a < 10000) (h₂ : 1000 ≤ b ∧ b < 10000) (h₃ : Nat.gcd a b = 5) : 
  Nat.lcm a b = 201000 :=
sorry

end NUMINAMATH_GPT_smallest_lcm_l2058_205859


namespace NUMINAMATH_GPT_part_1_part_2_l2058_205800

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 ≤ 0}

theorem part_1 (m : ℝ) : (A ∩ B m = {x | 0 ≤ x ∧ x ≤ 3}) → (m = 2) := 
by
  sorry

theorem part_2 (m : ℝ) : (A ⊆ (Set.univ \ B m)) → (m > 5 ∨ m < -3) := 
by
  sorry

end NUMINAMATH_GPT_part_1_part_2_l2058_205800


namespace NUMINAMATH_GPT_triangle_area_l2058_205832

theorem triangle_area (r : ℝ) (a b c : ℝ) (h : a^2 + b^2 = c^2) (hc : c = 2 * r) (r_val : r = 5) (ratio : a / b = 3 / 4 ∧ b / c = 4 / 5) :
  (1 / 2) * a * b = 24 :=
by
  -- We assume statements are given
  sorry

end NUMINAMATH_GPT_triangle_area_l2058_205832


namespace NUMINAMATH_GPT_area_perimeter_quadratic_l2058_205826

theorem area_perimeter_quadratic (a x y : ℝ) (h1 : x = 4 * a) (h2 : y = a^2) : y = (x / 4)^2 :=
by sorry

end NUMINAMATH_GPT_area_perimeter_quadratic_l2058_205826


namespace NUMINAMATH_GPT_average_score_l2058_205883

variable (K M : ℕ) (E : ℕ)

theorem average_score (h1 : (K + M) / 2 = 86) (h2 : E = 98) :
  (K + M + E) / 3 = 90 :=
by
  sorry

end NUMINAMATH_GPT_average_score_l2058_205883


namespace NUMINAMATH_GPT_part1_part2_l2058_205829

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

-- Statement for part (1)
theorem part1 (m : ℝ) : (m > -2) → (∀ x : ℝ, m + f x > 0) :=
sorry

-- Statement for part (2)
theorem part2 (m : ℝ) : (m > 2) ↔ (∀ x : ℝ, m - f x > 0) :=
sorry

end NUMINAMATH_GPT_part1_part2_l2058_205829


namespace NUMINAMATH_GPT_seq_a_2014_l2058_205804

theorem seq_a_2014 {a : ℕ → ℕ}
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → n * a (n + 1) = (n + 1) * a n) :
  a 2014 = 2014 :=
sorry

end NUMINAMATH_GPT_seq_a_2014_l2058_205804


namespace NUMINAMATH_GPT_find_analytical_expression_of_f_l2058_205885

-- Define the function f satisfying the condition
def f (x : ℝ) : ℝ := sorry

-- Lean 4 theorem statement
theorem find_analytical_expression_of_f :
  (∀ x : ℝ, f (x + 1) = x^2 + 2*x + 2) → (∀ x : ℝ, f x = x^2 + 1) :=
by
  -- The initial f definition and theorem statement are created
  -- The proof is omitted since the focus is on translating the problem
  sorry

end NUMINAMATH_GPT_find_analytical_expression_of_f_l2058_205885


namespace NUMINAMATH_GPT_volume_is_correct_l2058_205852

def condition1 (x y z : ℝ) : Prop := abs (x + 2 * y + 3 * z) + abs (x + 2 * y - 3 * z) ≤ 18
def condition2 (x y z : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0
def region (x y z : ℝ) : Prop := condition1 x y z ∧ condition2 x y z

noncomputable def volume_of_region : ℝ :=
  60.75 -- the result obtained from the calculation steps

theorem volume_is_correct : ∀ (x y z : ℝ), region x y z → volume_of_region = 60.75 :=
by
  sorry

end NUMINAMATH_GPT_volume_is_correct_l2058_205852


namespace NUMINAMATH_GPT_calories_burned_l2058_205881

/-- 
  The football coach makes his players run up and down the bleachers 60 times. 
  Each time they run up and down, they encounter 45 stairs. 
  The first half of the staircase has 20 stairs and every stair burns 3 calories, 
  while the second half has 25 stairs burning 4 calories each. 
  Prove that each player burns 9600 calories during this exercise.
--/
theorem calories_burned (n_stairs_first_half : ℕ) (calories_first_half : ℕ) 
  (n_stairs_second_half : ℕ) (calories_second_half : ℕ) (n_trips : ℕ) 
  (total_calories : ℕ) :
  n_stairs_first_half = 20 → calories_first_half = 3 → 
  n_stairs_second_half = 25 → calories_second_half = 4 → 
  n_trips = 60 → total_calories = 
  (n_stairs_first_half * calories_first_half + n_stairs_second_half * calories_second_half) * n_trips →
  total_calories = 9600 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_calories_burned_l2058_205881


namespace NUMINAMATH_GPT_div_neg_forty_five_l2058_205873

theorem div_neg_forty_five : (-40 / 5) = -8 :=
by
  sorry

end NUMINAMATH_GPT_div_neg_forty_five_l2058_205873


namespace NUMINAMATH_GPT_maria_scored_33_points_l2058_205828

-- Defining constants and parameters
def num_shots := 40
def equal_distribution : ℕ := num_shots / 3 -- each type of shot

-- Given success rates
def success_rate_three_point : ℚ := 0.25
def success_rate_two_point : ℚ := 0.50
def success_rate_free_throw : ℚ := 0.80

-- Defining the points per successful shot
def points_per_successful_three_point_shot : ℕ := 3
def points_per_successful_two_point_shot : ℕ := 2
def points_per_successful_free_throw_shot : ℕ := 1

-- Calculating total points scored
def total_points_scored :=
  (success_rate_three_point * points_per_successful_three_point_shot * equal_distribution) +
  (success_rate_two_point * points_per_successful_two_point_shot * equal_distribution) +
  (success_rate_free_throw * points_per_successful_free_throw_shot * equal_distribution)

theorem maria_scored_33_points :
  total_points_scored = 33 := 
sorry

end NUMINAMATH_GPT_maria_scored_33_points_l2058_205828


namespace NUMINAMATH_GPT_correct_expression_l2058_205858

theorem correct_expression (a b : ℝ) : (a - b) * (b + a) = a^2 - b^2 :=
by
  sorry

end NUMINAMATH_GPT_correct_expression_l2058_205858


namespace NUMINAMATH_GPT_price_of_movie_ticket_l2058_205824

theorem price_of_movie_ticket
  (M F : ℝ)
  (h1 : 8 * M = 2 * F)
  (h2 : 8 * M + 5 * F = 840) :
  M = 30 :=
by
  sorry

end NUMINAMATH_GPT_price_of_movie_ticket_l2058_205824


namespace NUMINAMATH_GPT_identical_dice_probability_l2058_205814

def num_ways_to_paint_die : ℕ := 3^6

def total_ways_to_paint_dice (n : ℕ) : ℕ := (num_ways_to_paint_die ^ n)

def count_identical_ways : ℕ := 1 + 324 + 8100

def probability_identical_dice : ℚ :=
  (count_identical_ways : ℚ) / (total_ways_to_paint_dice 2 : ℚ)

theorem identical_dice_probability : probability_identical_dice = 8425 / 531441 := by
  sorry

end NUMINAMATH_GPT_identical_dice_probability_l2058_205814


namespace NUMINAMATH_GPT_counting_unit_of_0_75_l2058_205877

def decimal_places (n : ℝ) : ℕ := 
  by sorry  -- Assume this function correctly calculates the number of decimal places of n

def counting_unit (n : ℝ) : ℝ :=
  by sorry  -- Assume this function correctly determines the counting unit based on decimal places

theorem counting_unit_of_0_75 : counting_unit 0.75 = 0.01 :=
  by sorry


end NUMINAMATH_GPT_counting_unit_of_0_75_l2058_205877


namespace NUMINAMATH_GPT_housing_price_growth_l2058_205833

theorem housing_price_growth (x : ℝ) (h₁ : (5500 : ℝ) > 0) (h₂ : (7000 : ℝ) > 0) :
  5500 * (1 + x) ^ 2 = 7000 := 
sorry

end NUMINAMATH_GPT_housing_price_growth_l2058_205833


namespace NUMINAMATH_GPT_simplify_expression_l2058_205812

variable (x y : ℝ)
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)

theorem simplify_expression : (x⁻¹ - y) ^ 2 = (1 / x ^ 2 - 2 * y / x + y ^ 2) :=
  sorry

end NUMINAMATH_GPT_simplify_expression_l2058_205812


namespace NUMINAMATH_GPT_least_positive_integer_k_l2058_205816

noncomputable def least_k (a : ℝ) (n : ℕ) : ℝ :=
  (1 : ℝ) / ((n + 1 : ℝ) ^ 3)

theorem least_positive_integer_k :
  ∃ k : ℕ , (∀ a : ℝ, ∀ n : ℕ,
  (0 ≤ a ∧ a ≤ 1) → (a^k * (1 - a)^n < least_k a n)) ∧
  (∀ k' : ℕ, k' < 4 → ¬(∀ a : ℝ, ∀ n : ℕ, (0 ≤ a ∧ a ≤ 1) → (a^k' * (1 - a)^n < least_k a n))) :=
sorry

end NUMINAMATH_GPT_least_positive_integer_k_l2058_205816


namespace NUMINAMATH_GPT_date_behind_D_correct_l2058_205865

noncomputable def date_behind_B : ℕ := sorry
noncomputable def date_behind_E : ℕ := date_behind_B + 2
noncomputable def date_behind_F : ℕ := date_behind_B + 15
noncomputable def date_behind_D : ℕ := sorry

theorem date_behind_D_correct :
  date_behind_B + date_behind_D = date_behind_E + date_behind_F := sorry

end NUMINAMATH_GPT_date_behind_D_correct_l2058_205865


namespace NUMINAMATH_GPT_maximum_area_of_flower_bed_l2058_205868

-- Definitions based on conditions
def length_of_flower_bed : ℝ := 150
def total_fencing : ℝ := 450

-- Question reframed as a proof statement
theorem maximum_area_of_flower_bed :
  ∀ (w : ℝ), 2 * w + length_of_flower_bed = total_fencing → (length_of_flower_bed * w = 22500) :=
by
  intro w h
  sorry

end NUMINAMATH_GPT_maximum_area_of_flower_bed_l2058_205868


namespace NUMINAMATH_GPT_last_digit_1993_2002_plus_1995_2002_l2058_205897

theorem last_digit_1993_2002_plus_1995_2002 :
  (1993 ^ 2002 + 1995 ^ 2002) % 10 = 4 :=
by sorry

end NUMINAMATH_GPT_last_digit_1993_2002_plus_1995_2002_l2058_205897


namespace NUMINAMATH_GPT_mark_bananas_equals_mike_matt_fruits_l2058_205825

theorem mark_bananas_equals_mike_matt_fruits :
  (∃ (bananas_mike matt_apples mark_bananas : ℕ),
    bananas_mike = 3 ∧
    matt_apples = 2 * bananas_mike ∧
    mark_bananas = 18 - (bananas_mike + matt_apples) ∧
    mark_bananas = (bananas_mike + matt_apples)) :=
sorry

end NUMINAMATH_GPT_mark_bananas_equals_mike_matt_fruits_l2058_205825


namespace NUMINAMATH_GPT_planting_area_l2058_205878

variable (x : ℝ)

def garden_length := x + 2
def garden_width := 4
def path_width := 1

def effective_garden_length := garden_length x - 2 * path_width
def effective_garden_width := garden_width - 2 * path_width

theorem planting_area : effective_garden_length x * effective_garden_width = 2 * x := by
  simp [garden_length, garden_width, path_width, effective_garden_length, effective_garden_width]
  sorry

end NUMINAMATH_GPT_planting_area_l2058_205878


namespace NUMINAMATH_GPT_evaluate_expression_l2058_205820

theorem evaluate_expression :
  - (18 / 3 * 8 - 70 + 5 * 7) = -13 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2058_205820


namespace NUMINAMATH_GPT_parallel_x_axis_implies_conditions_l2058_205867

variable (a b : ℝ)

theorem parallel_x_axis_implies_conditions (h1 : (5, a) ≠ (b, -2)) (h2 : (5, -2) = (5, a)) : a = -2 ∧ b ≠ 5 :=
sorry

end NUMINAMATH_GPT_parallel_x_axis_implies_conditions_l2058_205867


namespace NUMINAMATH_GPT_no_integer_solutions_l2058_205872

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ), x^1988 + y^1988 + z^1988 = 7^1990 := by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l2058_205872


namespace NUMINAMATH_GPT_exists_multiple_of_n_with_ones_l2058_205893

theorem exists_multiple_of_n_with_ones (n : ℤ) (hn1 : n ≥ 1) (hn2 : Int.gcd n 10 = 1) :
  ∃ k : ℕ, n ∣ (10^k - 1) / 9 :=
by sorry

end NUMINAMATH_GPT_exists_multiple_of_n_with_ones_l2058_205893


namespace NUMINAMATH_GPT_john_bought_slurpees_l2058_205822

noncomputable def slurpees_bought (total_money paid change slurpee_cost : ℕ) : ℕ :=
  (paid - change) / slurpee_cost

theorem john_bought_slurpees :
  let total_money := 20
  let slurpee_cost := 2
  let change := 8
  slurpees_bought total_money total_money change slurpee_cost = 6 :=
by
  sorry

end NUMINAMATH_GPT_john_bought_slurpees_l2058_205822


namespace NUMINAMATH_GPT_divisible_12_or_36_l2058_205841

theorem divisible_12_or_36 (x : ℕ) (n : ℕ) (h1 : Nat.Prime x) (h2 : 3 < x) (h3 : x = 3 * n + 1 ∨ x = 3 * n - 1) :
  12 ∣ (x^6 - x^3 - x^2 + x) ∨ 36 ∣ (x^6 - x^3 - x^2 + x) := 
by
  sorry

end NUMINAMATH_GPT_divisible_12_or_36_l2058_205841


namespace NUMINAMATH_GPT_exists_function_f_l2058_205801

-- Define the problem statement
theorem exists_function_f :
  ∃ (f : ℝ → ℝ), ∀ x : ℝ, f (abs (x + 1)) = x^2 + 2 * x :=
sorry

end NUMINAMATH_GPT_exists_function_f_l2058_205801


namespace NUMINAMATH_GPT_adam_has_10_apples_l2058_205815

theorem adam_has_10_apples
  (Jackie_has_2_apples : ∀ Jackie_apples, Jackie_apples = 2)
  (Adam_has_8_more_apples : ∀ Adam_apples Jackie_apples, Adam_apples = Jackie_apples + 8)
  : ∀ Adam_apples, Adam_apples = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_adam_has_10_apples_l2058_205815


namespace NUMINAMATH_GPT_max_y_difference_eq_l2058_205899

theorem max_y_difference_eq (x y p q : ℤ) (hp : 0 < p) (hq : 0 < q)
  (h : x * y = p * x + q * y) : y - x = (p - 1) * (q + 1) :=
sorry

end NUMINAMATH_GPT_max_y_difference_eq_l2058_205899


namespace NUMINAMATH_GPT_integer_pairs_satisfy_equation_l2058_205813

theorem integer_pairs_satisfy_equation :
  ∀ (a b : ℤ), b + 1 ≠ 0 → b + 2 ≠ 0 → a + b + 1 ≠ 0 →
    ( (a + 2)/(b + 1) + (a + 1)/(b + 2) = 1 + 6/(a + b + 1) ↔ 
      (a = 1 ∧ b = 0) ∨ (∃ t : ℤ, t ≠ 0 ∧ t ≠ -1 ∧ a = -3 - t ∧ b = t) ) :=
by
  intros a b h1 h2 h3
  sorry

end NUMINAMATH_GPT_integer_pairs_satisfy_equation_l2058_205813


namespace NUMINAMATH_GPT_tenth_number_in_sixteenth_group_is_257_l2058_205842

-- Define the general term of the sequence a_n = 2n - 3.
def a_n (n : ℕ) : ℕ := 2 * n - 3

-- Define the first number of the n-th group.
def first_number_of_group (n : ℕ) : ℕ := n^2 - n - 1

-- Define the m-th number in the n-th group.
def group_n_m (n m : ℕ) : ℕ := first_number_of_group n + (m - 1) * 2

theorem tenth_number_in_sixteenth_group_is_257 : group_n_m 16 10 = 257 := by
  sorry

end NUMINAMATH_GPT_tenth_number_in_sixteenth_group_is_257_l2058_205842


namespace NUMINAMATH_GPT_vertical_asymptote_at_x_4_l2058_205811

def P (x : ℝ) : ℝ := x^2 + 2 * x + 8
def Q (x : ℝ) : ℝ := x^2 - 8 * x + 16

theorem vertical_asymptote_at_x_4 : ∃ x : ℝ, Q x = 0 ∧ P x ≠ 0 ∧ x = 4 :=
by
  use 4
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_vertical_asymptote_at_x_4_l2058_205811


namespace NUMINAMATH_GPT_evaluate_M_l2058_205874

noncomputable def M : ℝ := 
  (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 7 + 2) - Real.sqrt (5 - 2 * Real.sqrt 6)

theorem evaluate_M : M = (1 + Real.sqrt 3 + Real.sqrt 5 + 3 * Real.sqrt 2) / 3 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_M_l2058_205874


namespace NUMINAMATH_GPT_time_to_pass_platform_l2058_205831

-- Definitions
def train_length : ℕ := 1400
def platform_length : ℕ := 700
def time_to_cross_tree : ℕ := 100
def train_speed : ℕ := train_length / time_to_cross_tree
def total_distance : ℕ := train_length + platform_length

-- Prove that the time to pass the platform is 150 seconds
theorem time_to_pass_platform : total_distance / train_speed = 150 :=
by
  sorry

end NUMINAMATH_GPT_time_to_pass_platform_l2058_205831


namespace NUMINAMATH_GPT_gas_volume_at_31_degrees_l2058_205847

theorem gas_volume_at_31_degrees :
  (∀ T V : ℕ, (T = 45 → V = 30) ∧ (∀ k, T = 45 - 2 * k → V = 30 - 3 * k)) →
  ∃ V, (T = 31) ∧ (V = 9) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_gas_volume_at_31_degrees_l2058_205847


namespace NUMINAMATH_GPT_unique_primes_solution_l2058_205884

theorem unique_primes_solution (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) : 
    p + q^2 = r^4 ↔ (p = 7 ∧ q = 3 ∧ r = 2) := 
by
  sorry

end NUMINAMATH_GPT_unique_primes_solution_l2058_205884


namespace NUMINAMATH_GPT_intersection_of_domains_l2058_205863

def A_domain : Set ℝ := { x : ℝ | 4 - x^2 ≥ 0 }
def B_domain : Set ℝ := { x : ℝ | 1 - x > 0 }

theorem intersection_of_domains :
  (A_domain ∩ B_domain) = { x : ℝ | -2 ≤ x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_domains_l2058_205863


namespace NUMINAMATH_GPT_x_minus_y_eq_11_l2058_205818

theorem x_minus_y_eq_11 (x y : ℝ) (h : |x - 6| + |y + 5| = 0) : x - y = 11 := by
  sorry

end NUMINAMATH_GPT_x_minus_y_eq_11_l2058_205818


namespace NUMINAMATH_GPT_time_for_A_to_complete_work_l2058_205866

-- Defining the work rates and the condition
def workRateA (a : ℕ) : ℚ := 1 / a
def workRateB : ℚ := 1 / 12
def workRateC : ℚ := 1 / 24
def combinedWorkRate (a : ℕ) : ℚ := workRateA a + workRateB + workRateC
def togetherWorkRate : ℚ := 1 / 4

-- Stating the theorem
theorem time_for_A_to_complete_work : 
  ∃ (a : ℕ), combinedWorkRate a = togetherWorkRate ∧ a = 8 :=
by
  sorry

end NUMINAMATH_GPT_time_for_A_to_complete_work_l2058_205866


namespace NUMINAMATH_GPT_average_age_increase_l2058_205843

theorem average_age_increase 
  (n : Nat) 
  (a : ℕ) 
  (b : ℕ) 
  (total_students : Nat)
  (avg_age_9 : ℕ) 
  (tenth_age : ℕ) 
  (original_total_age : Nat)
  (new_total_age : Nat)
  (new_avg_age : ℕ)
  (age_increase : ℕ) 
  (h1 : n = 9) 
  (h2 : avg_age_9 = 8) 
  (h3 : tenth_age = 28)
  (h4 : total_students = 10)
  (h5 : original_total_age = n * avg_age_9) 
  (h6 : new_total_age = original_total_age + tenth_age)
  (h7 : new_avg_age = new_total_age / total_students)
  (h8 : age_increase = new_avg_age - avg_age_9) :
  age_increase = 2 := 
by 
  sorry

end NUMINAMATH_GPT_average_age_increase_l2058_205843


namespace NUMINAMATH_GPT_each_girl_brought_2_cups_l2058_205888

-- Here we define all the given conditions
def total_students : ℕ := 30
def num_boys : ℕ := 10
def cups_per_boy : ℕ := 5
def total_cups : ℕ := 90

-- Define the conditions as Lean definitions
def num_girls : ℕ := total_students - num_boys -- From condition 3
def cups_by_boys : ℕ := num_boys * cups_per_boy
def cups_by_girls : ℕ := total_cups - cups_by_boys

-- Define the final question and expected answer
def cups_per_girl : ℕ := cups_by_girls / num_girls

-- Final problem statement to prove
theorem each_girl_brought_2_cups :
  cups_per_girl = 2 :=
by
  have h1 : num_girls = 20 := by sorry
  have h2 : cups_by_boys = 50 := by sorry
  have h3 : cups_by_girls = 40 := by sorry
  have h4 : cups_per_girl = 2 := by sorry
  exact h4

end NUMINAMATH_GPT_each_girl_brought_2_cups_l2058_205888


namespace NUMINAMATH_GPT_brian_tape_needed_l2058_205890

-- Definitions of conditions
def tape_needed_for_box (short_side: ℕ) (long_side: ℕ) : ℕ := 
  2 * short_side + long_side

def total_tape_needed (num_short_long_boxes: ℕ) (short_side: ℕ) (long_side: ℕ) (num_square_boxes: ℕ) (side: ℕ) : ℕ := 
  (num_short_long_boxes * tape_needed_for_box short_side long_side) + (num_square_boxes * 3 * side)

-- Theorem statement
theorem brian_tape_needed : total_tape_needed 5 15 30 2 40 = 540 := 
by 
  sorry

end NUMINAMATH_GPT_brian_tape_needed_l2058_205890


namespace NUMINAMATH_GPT_fraction_numerator_greater_than_denominator_l2058_205870

theorem fraction_numerator_greater_than_denominator (x : ℝ) :
  (4 * x + 2 > 8 - 3 * x) ↔ (6 / 7 < x ∧ x ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_fraction_numerator_greater_than_denominator_l2058_205870


namespace NUMINAMATH_GPT_problem_inequality_l2058_205846

noncomputable def A (x : ℝ) := (x - 3) ^ 2
noncomputable def B (x : ℝ) := (x - 2) * (x - 4)

theorem problem_inequality (x : ℝ) : A x > B x :=
  by
    sorry

end NUMINAMATH_GPT_problem_inequality_l2058_205846


namespace NUMINAMATH_GPT_intersection_eq_l2058_205887

-- Define sets A and B
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {0, 1, 2}

-- The theorem to be proved
theorem intersection_eq : A ∩ B = {2} := 
by sorry

end NUMINAMATH_GPT_intersection_eq_l2058_205887


namespace NUMINAMATH_GPT_prime_quadruples_unique_l2058_205810

noncomputable def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → (m = 1 ∨ m = n)

theorem prime_quadruples_unique (p q r n : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) (hn : n > 0)
  (h_eq : p^2 = q^2 + r^n) :
  (p, q, r, n) = (3, 2, 5, 1) ∨ (p, q, r, n) = (5, 3, 2, 4) :=
by
  sorry

end NUMINAMATH_GPT_prime_quadruples_unique_l2058_205810


namespace NUMINAMATH_GPT_number_of_pencils_l2058_205844

theorem number_of_pencils (P : ℕ) (h : ∃ (n : ℕ), n * 4 = P) : ∃ k, 4 * k = P :=
  by
  sorry

end NUMINAMATH_GPT_number_of_pencils_l2058_205844


namespace NUMINAMATH_GPT_exists_multiple_sum_divides_l2058_205830

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_multiple_sum_divides {n : ℕ} (hn : n > 0) :
  ∃ (n_ast : ℕ), n ∣ n_ast ∧ sum_of_digits n_ast ∣ n_ast :=
by
  sorry

end NUMINAMATH_GPT_exists_multiple_sum_divides_l2058_205830


namespace NUMINAMATH_GPT_ivan_total_money_l2058_205871

-- Define values of the coins
def penny_value : ℝ := 0.01
def dime_value : ℝ := 0.1
def nickel_value : ℝ := 0.05
def quarter_value : ℝ := 0.25

-- Define number of each type of coin in each piggy bank
def first_piggybank_pennies := 100
def first_piggybank_dimes := 50
def first_piggybank_nickels := 20
def first_piggybank_quarters := 10

def second_piggybank_pennies := 150
def second_piggybank_dimes := 30
def second_piggybank_nickels := 40
def second_piggybank_quarters := 15

def third_piggybank_pennies := 200
def third_piggybank_dimes := 60
def third_piggybank_nickels := 10
def third_piggybank_quarters := 20

-- Calculate the total value of each piggy bank
def first_piggybank_value : ℝ :=
  (first_piggybank_pennies * penny_value) +
  (first_piggybank_dimes * dime_value) +
  (first_piggybank_nickels * nickel_value) +
  (first_piggybank_quarters * quarter_value)

def second_piggybank_value : ℝ :=
  (second_piggybank_pennies * penny_value) +
  (second_piggybank_dimes * dime_value) +
  (second_piggybank_nickels * nickel_value) +
  (second_piggybank_quarters * quarter_value)

def third_piggybank_value : ℝ :=
  (third_piggybank_pennies * penny_value) +
  (third_piggybank_dimes * dime_value) +
  (third_piggybank_nickels * nickel_value) +
  (third_piggybank_quarters * quarter_value)

-- Calculate the total amount of money Ivan has
def total_value : ℝ :=
  first_piggybank_value + second_piggybank_value + third_piggybank_value

-- The theorem to prove
theorem ivan_total_money :
  total_value = 33.25 :=
by
  sorry

end NUMINAMATH_GPT_ivan_total_money_l2058_205871


namespace NUMINAMATH_GPT_find_fg3_l2058_205808

def f (x : ℝ) : ℝ := 2 * x - 5
def g (x : ℝ) : ℝ := x^2 + 1

theorem find_fg3 : f (g 3) = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_fg3_l2058_205808


namespace NUMINAMATH_GPT_A_formula_l2058_205861

noncomputable def A (i : ℕ) (A₀ θ : ℝ) : ℝ :=
match i with
| 0     => A₀
| (i+1) => (A i A₀ θ * Real.cos θ + Real.sin θ) / (-A i A₀ θ * Real.sin θ + Real.cos θ)

theorem A_formula (A₀ θ : ℝ) (n : ℕ) :
  A n A₀ θ = (A₀ * Real.cos (n * θ) + Real.sin (n * θ)) / (-A₀ * Real.sin (n * θ) + Real.cos (n * θ)) :=
by
  sorry

end NUMINAMATH_GPT_A_formula_l2058_205861


namespace NUMINAMATH_GPT_sin_alpha_of_point_P_l2058_205891

theorem sin_alpha_of_point_P (α : ℝ) 
  (h1 : ∃ P : ℝ × ℝ, P = (Real.cos (π / 3), 1) ∧ P = (Real.cos α, Real.sin α) ) :
  Real.sin α = (2 * Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_GPT_sin_alpha_of_point_P_l2058_205891


namespace NUMINAMATH_GPT_calculate_expression_value_l2058_205860

theorem calculate_expression_value :
  (23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2) = 288 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_value_l2058_205860


namespace NUMINAMATH_GPT_find_number_l2058_205856

theorem find_number (x : ℝ) :
  10 * x - 10 = 50 ↔ x = 6 := by
  sorry

end NUMINAMATH_GPT_find_number_l2058_205856


namespace NUMINAMATH_GPT_round_robin_10_person_tournament_l2058_205809

noncomputable def num_matches (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem round_robin_10_person_tournament :
  num_matches 10 = 45 :=
by
  sorry

end NUMINAMATH_GPT_round_robin_10_person_tournament_l2058_205809


namespace NUMINAMATH_GPT_total_handshakes_l2058_205875

theorem total_handshakes :
  let gremlins := 20
  let imps := 20
  let sprites := 10
  let handshakes_gremlins := gremlins * (gremlins - 1) / 2
  let handshakes_gremlins_imps := gremlins * imps
  let handshakes_imps_sprites := imps * sprites
  handshakes_gremlins + handshakes_gremlins_imps + handshakes_imps_sprites = 790 :=
by
  sorry

end NUMINAMATH_GPT_total_handshakes_l2058_205875


namespace NUMINAMATH_GPT_divisibility_check_l2058_205898

variable (d : ℕ) (h1 : d % 2 = 1) (h2 : d % 5 ≠ 0)
variable (δ : ℕ) (h3 : ∃ m : ℕ, 10 * δ + 1 = m * d)
variable (N : ℕ)

def last_digit (N : ℕ) : ℕ := N % 10
def remove_last_digit (N : ℕ) : ℕ := N / 10

theorem divisibility_check (h4 : ∃ N' u : ℕ, N = 10 * N' + u ∧ N = N' * 10 + u ∧ N' = remove_last_digit N ∧ u = last_digit N)
  (N' : ℕ) (u : ℕ) (N1 : ℕ) (h5 : N1 = N' - δ * u) :
  d ∣ N1 → d ∣ N := by
  sorry

end NUMINAMATH_GPT_divisibility_check_l2058_205898


namespace NUMINAMATH_GPT_complex_problem_proof_l2058_205880

open Complex

noncomputable def z : ℂ := (1 - I)^2 + 1 + 3 * I

theorem complex_problem_proof : z = 1 + I ∧ abs (z - 2 * I) = Real.sqrt 2 ∧ (∀ a b : ℝ, (z^2 + a + b = 1 - I) → (a = -3 ∧ b = 4)) := 
by
  have h1 : z = (1 - I)^2 + 1 + 3 * I := rfl
  have h2 : z = 1 + I := sorry
  have h3 : abs (z - 2 * I) = Real.sqrt 2 := sorry
  have h4 : (∀ a b : ℝ, (z^2 + a + b = 1 - I) → (a = -3 ∧ b = 4)) := sorry
  exact ⟨h2, h3, h4⟩

end NUMINAMATH_GPT_complex_problem_proof_l2058_205880


namespace NUMINAMATH_GPT_find_A_l2058_205889

def heartsuit (A B : ℤ) : ℤ := 4 * A + A * B + 3 * B + 6

theorem find_A (A : ℤ) : heartsuit A 3 = 75 ↔ A = 60 / 7 := sorry

end NUMINAMATH_GPT_find_A_l2058_205889


namespace NUMINAMATH_GPT_vector_on_line_l2058_205869

theorem vector_on_line (t : ℝ) (x y : ℝ) : 
  (x = 3 * t + 1) → (y = 2 * t + 3) → 
  ∃ t, (∃ x y, (x = 3 * t + 1) ∧ (y = 2 * t + 3) ∧ (x = 23 / 2) ∧ (y = 10)) :=
  by
  sorry

end NUMINAMATH_GPT_vector_on_line_l2058_205869


namespace NUMINAMATH_GPT_polynomial_at_neg_one_eq_neg_two_l2058_205837

-- Define the polynomial f(x)
def polynomial (x : ℝ) : ℝ := 1 + x + 2 * x^2 + 3 * x^3 + 4 * x^4 + 5 * x^5

-- Define Horner's method process
def horner_method (x : ℝ) : ℝ :=
  let a5 := 5
  let a4 := 4
  let a3 := 3
  let a2 := 2
  let a1 := 1
  let a  := 1
  let u4 := a5 * x + a4
  let u3 := u4 * x + a3
  let u2 := u3 * x + a2
  let u1 := u2 * x + a1
  let u0 := u1 * x + a
  u0

-- Prove that the polynomial evaluated using Horner's method at x := -1 is equal to -2
theorem polynomial_at_neg_one_eq_neg_two : horner_method (-1) = -2 := by
  sorry

end NUMINAMATH_GPT_polynomial_at_neg_one_eq_neg_two_l2058_205837


namespace NUMINAMATH_GPT_trapezoid_area_l2058_205805

theorem trapezoid_area (h : ℝ) : 
  let base1 := 3 * h 
  let base2 := 4 * h 
  let average_base := (base1 + base2) / 2 
  let area := average_base * h 
  area = (7 * h^2) / 2 := 
by
  sorry

end NUMINAMATH_GPT_trapezoid_area_l2058_205805


namespace NUMINAMATH_GPT_cos_sin_identity_l2058_205827

theorem cos_sin_identity : 
  (Real.cos (75 * Real.pi / 180) + Real.sin (75 * Real.pi / 180)) * 
  (Real.cos (75 * Real.pi / 180) - Real.sin (75 * Real.pi / 180)) = -Real.sqrt 3 / 2 := 
  sorry

end NUMINAMATH_GPT_cos_sin_identity_l2058_205827


namespace NUMINAMATH_GPT_circle_condition_l2058_205840

theorem circle_condition (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - 2*m*x - 2*m*y + 2*m^2 + m - 1 = 0) →
  m < 1 :=
sorry

end NUMINAMATH_GPT_circle_condition_l2058_205840


namespace NUMINAMATH_GPT_ratio_blue_yellow_l2058_205819

theorem ratio_blue_yellow (total_butterflies blue_butterflies black_butterflies : ℕ)
  (h_total : total_butterflies = 19)
  (h_blue : blue_butterflies = 6)
  (h_black : black_butterflies = 10) :
  (blue_butterflies : ℚ) / (total_butterflies - blue_butterflies - black_butterflies : ℚ) = 2 / 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_ratio_blue_yellow_l2058_205819


namespace NUMINAMATH_GPT_least_positive_integer_l2058_205854

theorem least_positive_integer (N : ℕ) :
  (N % 11 = 10) ∧
  (N % 12 = 11) ∧
  (N % 13 = 12) ∧
  (N % 14 = 13) ∧
  (N % 15 = 14) ∧
  (N % 16 = 15) →
  N = 720719 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_l2058_205854


namespace NUMINAMATH_GPT_minimum_cups_needed_l2058_205896

theorem minimum_cups_needed (container_capacity cup_capacity : ℕ) (h1 : container_capacity = 980) (h2 : cup_capacity = 80) : 
  Nat.ceil (container_capacity / cup_capacity : ℚ) = 13 :=
by
  sorry

end NUMINAMATH_GPT_minimum_cups_needed_l2058_205896


namespace NUMINAMATH_GPT_concert_revenue_l2058_205864

-- Defining the conditions
def ticket_price_adult : ℕ := 26
def ticket_price_child : ℕ := ticket_price_adult / 2
def attendees_adults : ℕ := 183
def attendees_children : ℕ := 28

-- Defining the total revenue calculation based on the conditions
def total_revenue : ℕ :=
  attendees_adults * ticket_price_adult +
  attendees_children * ticket_price_child

-- The theorem to prove the total revenue
theorem concert_revenue : total_revenue = 5122 := by
  sorry

end NUMINAMATH_GPT_concert_revenue_l2058_205864


namespace NUMINAMATH_GPT_population_2002_l2058_205838

-- Predicate P for the population of rabbits in a given year
def P : ℕ → ℝ := sorry

-- Given conditions
axiom cond1 : ∃ k : ℝ, P 2003 - P 2001 = k * P 2002
axiom cond2 : ∃ k : ℝ, P 2002 - P 2000 = k * P 2001
axiom condP2000 : P 2000 = 50
axiom condP2001 : P 2001 = 80
axiom condP2003 : P 2003 = 186

-- The statement we need to prove
theorem population_2002 : P 2002 = 120 :=
by
  sorry

end NUMINAMATH_GPT_population_2002_l2058_205838


namespace NUMINAMATH_GPT_sum_interest_l2058_205821

noncomputable def simple_interest (P : ℝ) (R : ℝ) := P * R * 3 / 100

theorem sum_interest (P R : ℝ) (h : simple_interest P (R + 1) - simple_interest P R = 75) : P = 2500 :=
by
  sorry

end NUMINAMATH_GPT_sum_interest_l2058_205821


namespace NUMINAMATH_GPT_evaluate_expression_l2058_205817

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2058_205817


namespace NUMINAMATH_GPT_find_a4_b4_c4_l2058_205848

theorem find_a4_b4_c4 (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 5) (h3 : a^3 + b^3 + c^3 = 15) : 
    a^4 + b^4 + c^4 = 35 := 
by 
  sorry

end NUMINAMATH_GPT_find_a4_b4_c4_l2058_205848


namespace NUMINAMATH_GPT_movie_tickets_l2058_205850

theorem movie_tickets (r h : ℕ) (h1 : r = 25) (h2 : h = 3 * r + 18) : h = 93 :=
by
  sorry

end NUMINAMATH_GPT_movie_tickets_l2058_205850


namespace NUMINAMATH_GPT_man_speed_with_stream_l2058_205806

variable (V_m V_as : ℝ)
variable (V_s V_ws : ℝ)

theorem man_speed_with_stream
  (cond1 : V_m = 5)
  (cond2 : V_as = 8)
  (cond3 : V_as = V_m - V_s)
  (cond4 : V_ws = V_m + V_s) :
  V_ws = 8 := 
by
  sorry

end NUMINAMATH_GPT_man_speed_with_stream_l2058_205806
