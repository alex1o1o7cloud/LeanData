import Mathlib

namespace NUMINAMATH_GPT_xy_plus_one_is_perfect_square_l397_39738

theorem xy_plus_one_is_perfect_square (x y : ℕ) (h : 1 / (x : ℝ) + 1 / (y : ℝ) = 1 / (x + 2 : ℝ) + 1 / (y - 2 : ℝ)) :
  ∃ k : ℕ, xy + 1 = k^2 :=
by
  sorry

end NUMINAMATH_GPT_xy_plus_one_is_perfect_square_l397_39738


namespace NUMINAMATH_GPT_original_wire_length_l397_39709

theorem original_wire_length 
(L : ℝ) 
(h1 : L / 2 - 3 / 2 > 0) 
(h2 : L / 2 - 3 > 0) 
(h3 : L / 4 - 11.5 > 0)
(h4 : L / 4 - 6.5 = 7) : 
L = 54 := 
sorry

end NUMINAMATH_GPT_original_wire_length_l397_39709


namespace NUMINAMATH_GPT_sin_over_cos_inequality_l397_39706

-- Define the main theorem and condition
theorem sin_over_cos_inequality (t : ℝ) (h₁ : 0 < t) (h₂ : t ≤ Real.pi / 2) : 
  (Real.sin t / t)^3 > Real.cos t := 
sorry

end NUMINAMATH_GPT_sin_over_cos_inequality_l397_39706


namespace NUMINAMATH_GPT_cost_of_article_l397_39712

theorem cost_of_article (C : ℝ) (H1 : 350 - C = G + 0.05 * G) (H2 : 345 - C = G) : C = 245 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_article_l397_39712


namespace NUMINAMATH_GPT_inverse_function_point_l397_39701

noncomputable def f (a : ℝ) (x : ℝ) := a^(x + 1)

theorem inverse_function_point (a : ℝ) (h_pos : 0 < a) (h_annoylem : f a (-1) = 1) :
  ∃ g : ℝ → ℝ, (∀ y, f a (g y) = y ∧ g (f a y) = y) ∧ g 1 = -1 :=
by
  sorry

end NUMINAMATH_GPT_inverse_function_point_l397_39701


namespace NUMINAMATH_GPT_cos_180_eq_neg_one_l397_39721

theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 := by
  sorry

end NUMINAMATH_GPT_cos_180_eq_neg_one_l397_39721


namespace NUMINAMATH_GPT_value_of_f_neg_a_l397_39759

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = -2 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_f_neg_a_l397_39759


namespace NUMINAMATH_GPT_cookie_contest_l397_39752

theorem cookie_contest (A B : ℚ) (hA : A = 5/6) (hB : B = 2/3) :
  A - B = 1/6 :=
by 
  sorry

end NUMINAMATH_GPT_cookie_contest_l397_39752


namespace NUMINAMATH_GPT_tomatoes_picked_second_week_l397_39776

-- Define the constants
def initial_tomatoes : Nat := 100
def fraction_picked_first_week : Nat := 1 / 4
def remaining_tomatoes : Nat := 15

-- Theorem to prove the number of tomatoes Jane picked in the second week
theorem tomatoes_picked_second_week (x : Nat) :
  let T := initial_tomatoes
  let p := fraction_picked_first_week
  let r := remaining_tomatoes
  let first_week_pick := T * p
  let remaining_after_first := T - first_week_pick
  let total_picked := remaining_after_first - r
  let second_week_pick := total_picked / 3
  second_week_pick = 20 := 
sorry

end NUMINAMATH_GPT_tomatoes_picked_second_week_l397_39776


namespace NUMINAMATH_GPT_edge_c_eq_3_or_5_l397_39773

noncomputable def a := 7
noncomputable def b := 8
noncomputable def A := Real.pi / 3

theorem edge_c_eq_3_or_5 (c : ℝ) (h : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) : c = 3 ∨ c = 5 :=
by
  sorry

end NUMINAMATH_GPT_edge_c_eq_3_or_5_l397_39773


namespace NUMINAMATH_GPT_width_of_rectangle_l397_39748

-- Define the problem constants and parameters
variable (L W : ℝ)

-- State the main theorem about the width
theorem width_of_rectangle (h₁ : L * W = 50) (h₂ : L + W = 15) : W = 5 :=
sorry

end NUMINAMATH_GPT_width_of_rectangle_l397_39748


namespace NUMINAMATH_GPT_value_of_a_l397_39735

theorem value_of_a 
  (a : ℝ) 
  (h : 0.005 * a = 0.85) : 
  a = 170 :=
sorry

end NUMINAMATH_GPT_value_of_a_l397_39735


namespace NUMINAMATH_GPT_radius_of_cookie_l397_39734

theorem radius_of_cookie (x y : ℝ) : 
  (x^2 + y^2 + x - 5 * y = 10) → 
  ∃ r, (r = Real.sqrt (33 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_radius_of_cookie_l397_39734


namespace NUMINAMATH_GPT_total_students_experimental_primary_school_l397_39767

theorem total_students_experimental_primary_school : 
  ∃ (n : ℕ), 
  n = (21 + 11) * 28 ∧ 
  n = 896 := 
by {
  -- Since the proof is not required, we use "sorry"
  sorry
}

end NUMINAMATH_GPT_total_students_experimental_primary_school_l397_39767


namespace NUMINAMATH_GPT_percent_shaded_area_of_rectangle_l397_39781

theorem percent_shaded_area_of_rectangle
  (side_length : ℝ)
  (length_rectangle : ℝ)
  (width_rectangle : ℝ)
  (overlap_length : ℝ)
  (h1 : side_length = 12)
  (h2 : length_rectangle = 20)
  (h3 : width_rectangle = 12)
  (h4 : overlap_length = 4)
  : (overlap_length * width_rectangle) / (length_rectangle * width_rectangle) * 100 = 20 :=
  sorry

end NUMINAMATH_GPT_percent_shaded_area_of_rectangle_l397_39781


namespace NUMINAMATH_GPT_number_of_marbles_in_Ellen_box_l397_39700

-- Defining the conditions given in the problem
def Dan_box_volume : ℕ := 216
def Ellen_side_multiplier : ℕ := 3
def marble_size_consistent_between_boxes : Prop := True -- Placeholder for the consistency condition

-- Main theorem statement
theorem number_of_marbles_in_Ellen_box :
  ∃ number_of_marbles_in_Ellen_box : ℕ,
  (∀ s : ℕ, s^3 = Dan_box_volume → (Ellen_side_multiplier * s)^3 / s^3 = 27 → 
  number_of_marbles_in_Ellen_box = 27 * Dan_box_volume) :=
by
  sorry

end NUMINAMATH_GPT_number_of_marbles_in_Ellen_box_l397_39700


namespace NUMINAMATH_GPT_geometric_sequence_k_value_l397_39754

theorem geometric_sequence_k_value (a : ℕ → ℝ) (S : ℕ → ℝ) (a1_pos : 0 < a 1)
  (geometric_seq : ∀ n, a (n + 2) = a n * (a 3 / a 1)) (h_a1 : a 1 = 1)
  (h_a3 : a 3 = 4) (h_Sk : S k = 63) :
  k = 6 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_k_value_l397_39754


namespace NUMINAMATH_GPT_fraction_value_l397_39780

variable (x y : ℚ)

theorem fraction_value (h₁ : x = 4 / 6) (h₂ : y = 8 / 12) : 
  (6 * x + 8 * y) / (48 * x * y) = 7 / 16 :=
by
  sorry

end NUMINAMATH_GPT_fraction_value_l397_39780


namespace NUMINAMATH_GPT_compare_flavors_l397_39737

def flavor_ratings_A := [7, 9, 8, 6, 10]
def flavor_ratings_B := [5, 6, 10, 10, 9]

def mean (l : List ℕ) : ℚ := l.sum / l.length

def variance (l : List ℕ) : ℚ :=
  let μ := mean l
  (l.map (λ x => (x - μ) ^ 2)).sum / l.length

theorem compare_flavors : 
  mean flavor_ratings_A = mean flavor_ratings_B ∧ variance flavor_ratings_A < variance flavor_ratings_B := by
  sorry

end NUMINAMATH_GPT_compare_flavors_l397_39737


namespace NUMINAMATH_GPT_samara_oil_spent_l397_39761

theorem samara_oil_spent (O : ℕ) (A_total : ℕ) (S_tires : ℕ) (S_detailing : ℕ) (diff : ℕ) (S_total : ℕ) :
  A_total = 2457 →
  S_tires = 467 →
  S_detailing = 79 →
  diff = 1886 →
  S_total = O + S_tires + S_detailing →
  A_total = S_total + diff →
  O = 25 :=
by
  sorry

end NUMINAMATH_GPT_samara_oil_spent_l397_39761


namespace NUMINAMATH_GPT_hexagon_perimeter_l397_39750

theorem hexagon_perimeter (side_length : ℕ) (num_sides : ℕ) (h1 : side_length = 5) (h2 : num_sides = 6) : 
  num_sides * side_length = 30 := by
  sorry

end NUMINAMATH_GPT_hexagon_perimeter_l397_39750


namespace NUMINAMATH_GPT_points_below_line_l397_39711

theorem points_below_line (d q x1 x2 y1 y2 : ℝ) 
  (h1 : 2 = 1 + 3 * d)
  (h2 : x1 = 1 + d)
  (h3 : x2 = x1 + d)
  (h4 : 2 = q ^ 3)
  (h5 : y1 = q)
  (h6 : y2 = q ^ 2) :
  x1 > y1 ∧ x2 > y2 :=
by {
  sorry
}

end NUMINAMATH_GPT_points_below_line_l397_39711


namespace NUMINAMATH_GPT_ellipse_properties_l397_39742

theorem ellipse_properties (h k a b : ℝ)
  (h_eq : h = 1)
  (k_eq : k = -3)
  (a_eq : a = 7)
  (b_eq : b = 4) :
  h + k + a + b = 9 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_properties_l397_39742


namespace NUMINAMATH_GPT_upper_bound_for_k_squared_l397_39747

theorem upper_bound_for_k_squared :
  (∃ (k : ℤ), k^2 > 121 ∧ ∀ m : ℤ, (m^2 > 121 ∧ m^2 < 323 → m = k + 1)) →
  (k ≤ 17) → (18^2 > 323) := 
by 
  sorry

end NUMINAMATH_GPT_upper_bound_for_k_squared_l397_39747


namespace NUMINAMATH_GPT_shift_down_two_units_l397_39769

theorem shift_down_two_units (x : ℝ) : 
  (y = 2 * x) → (y - 2 = 2 * x - 2) := by
sorry

end NUMINAMATH_GPT_shift_down_two_units_l397_39769


namespace NUMINAMATH_GPT_marissa_initial_ribbon_l397_39784

theorem marissa_initial_ribbon (ribbon_per_box : ℝ) (number_of_boxes : ℝ) (ribbon_left : ℝ) : 
  (ribbon_per_box = 0.7) → (number_of_boxes = 5) → (ribbon_left = 1) → 
  (ribbon_per_box * number_of_boxes + ribbon_left = 4.5) :=
  by
    intros
    sorry

end NUMINAMATH_GPT_marissa_initial_ribbon_l397_39784


namespace NUMINAMATH_GPT_batsman_average_after_11th_inning_l397_39705

variable (x : ℝ) -- The average before the 11th inning
variable (new_average : ℝ) -- The average after the 11th inning
variable (total_runs : ℝ) -- Total runs scored after 11 innings

-- Given conditions
def condition1 := total_runs = 11 * (x + 5)
def condition2 := total_runs = 10 * x + 110

theorem batsman_average_after_11th_inning : 
  ∀ (x : ℝ), 
    (x = 55) → (x + 5 = 60) :=
by
  intros
  sorry

end NUMINAMATH_GPT_batsman_average_after_11th_inning_l397_39705


namespace NUMINAMATH_GPT_total_bricks_used_l397_39727

def numberOfCoursesPerWall := 6
def bricksPerCourse := 10
def numberOfWalls := 4
def incompleteCourses := 2

theorem total_bricks_used :
  (numberOfCoursesPerWall * bricksPerCourse * (numberOfWalls - 1)) + ((numberOfCoursesPerWall - incompleteCourses) * bricksPerCourse) = 220 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_bricks_used_l397_39727


namespace NUMINAMATH_GPT_tammy_speed_second_day_l397_39713

theorem tammy_speed_second_day:
  ∃ (v t: ℝ), 
    t + (t - 2) = 14 ∧
    v * t + (v + 0.5) * (t - 2) = 52 ∧
    (v + 0.5) = 4 := sorry

end NUMINAMATH_GPT_tammy_speed_second_day_l397_39713


namespace NUMINAMATH_GPT_expr_simplification_l397_39702

noncomputable def simplify_sqrt_expr : ℝ :=
  Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 27

theorem expr_simplification : simplify_sqrt_expr = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_expr_simplification_l397_39702


namespace NUMINAMATH_GPT_automobile_travel_distance_l397_39785

theorem automobile_travel_distance
  (a r : ℝ) : 
  let feet_per_yard := 3
  let seconds_per_minute := 60
  let travel_feet := a / 4
  let travel_seconds := 2 * r
  let rate_yards_per_second := (travel_feet / travel_seconds) / feet_per_yard
  let total_seconds := 10 * seconds_per_minute
  let total_yards := rate_yards_per_second * total_seconds
  total_yards = 25 * a / r := by
  sorry

end NUMINAMATH_GPT_automobile_travel_distance_l397_39785


namespace NUMINAMATH_GPT_sum_opposite_abs_val_eq_neg_nine_l397_39765

theorem sum_opposite_abs_val_eq_neg_nine (a b : ℤ) (h1 : a = -15) (h2 : b = 6) : a + b = -9 := 
by
  -- conditions given
  rw [h1, h2]
  -- skip the proof
  sorry

end NUMINAMATH_GPT_sum_opposite_abs_val_eq_neg_nine_l397_39765


namespace NUMINAMATH_GPT_complement_of_M_l397_39755

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x ≥ 1}

theorem complement_of_M :
  (U \ M) = {x | x < 1} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_M_l397_39755


namespace NUMINAMATH_GPT_icosahedron_path_count_l397_39782

-- Definitions from the conditions
def vertices := 12
def edges := 30
def top_adjacent := 5
def bottom_adjacent := 5

-- Define the total paths calculation based on the given structural conditions
theorem icosahedron_path_count (v e ta ba : ℕ) (hv : v = 12) (he : e = 30) (hta : ta = 5) (hba : ba = 5) : 
  (ta * (ta - 1) * (ba - 1)) * 2 = 810 :=
by
-- Insert calculation logic here if needed or detailed structure definitions
  sorry

end NUMINAMATH_GPT_icosahedron_path_count_l397_39782


namespace NUMINAMATH_GPT_axis_of_symmetry_l397_39710

noncomputable def f (x : ℝ) := x^2 - 2 * x + Real.cos (x - 1)

theorem axis_of_symmetry :
  ∀ x : ℝ, f (1 + x) = f (1 - x) :=
by 
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_l397_39710


namespace NUMINAMATH_GPT_min_Sn_value_l397_39725

noncomputable def a (n : ℕ) (d : ℤ) : ℤ := -11 + (n - 1) * d

def Sn (n : ℕ) (d : ℤ) : ℤ := n * -11 + n * (n - 1) * d / 2

theorem min_Sn_value {d : ℤ} (h5_6 : a 5 d + a 6 d = -4) : 
  ∃ n, Sn n d = (n - 6)^2 - 36 ∧ n = 6 :=
by
  sorry

end NUMINAMATH_GPT_min_Sn_value_l397_39725


namespace NUMINAMATH_GPT_total_money_divided_l397_39791

theorem total_money_divided (A B C T : ℝ) 
    (h1 : A = (2/5) * (B + C)) 
    (h2 : B = (1/5) * (A + C)) 
    (h3 : A = 600) :
    T = A + B + C →
    T = 2100 :=
by 
  sorry

end NUMINAMATH_GPT_total_money_divided_l397_39791


namespace NUMINAMATH_GPT_vacation_cost_division_l397_39775

theorem vacation_cost_division 
  (total_cost : ℝ) 
  (initial_people : ℝ) 
  (initial_cost_per_person : ℝ) 
  (cost_difference : ℝ) 
  (new_cost_per_person : ℝ) 
  (new_people : ℝ) 
  (h1 : total_cost = 1000) 
  (h2 : initial_people = 4) 
  (h3 : initial_cost_per_person = total_cost / initial_people) 
  (h4 : initial_cost_per_person = 250) 
  (h5 : cost_difference = 50) 
  (h6 : new_cost_per_person = initial_cost_per_person - cost_difference) 
  (h7 : new_cost_per_person = 200) 
  (h8 : total_cost / new_people = new_cost_per_person) :
  new_people = 5 := 
sorry

end NUMINAMATH_GPT_vacation_cost_division_l397_39775


namespace NUMINAMATH_GPT_chocolates_problem_l397_39704

theorem chocolates_problem (C S : ℝ) (n : ℕ) 
  (h1 : 24 * C = n * S)
  (h2 : (S - C) / C = 0.5) : 
  n = 16 :=
by 
  sorry

end NUMINAMATH_GPT_chocolates_problem_l397_39704


namespace NUMINAMATH_GPT_no_integer_solutions_l397_39794

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), x^3 + 21 * y^2 + 5 = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_no_integer_solutions_l397_39794


namespace NUMINAMATH_GPT_find_two_digit_numbers_l397_39716

theorem find_two_digit_numbers (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h : 2 * (a + b) = a * b) : 
  10 * a + b = 63 ∨ 10 * a + b = 44 ∨ 10 * a + b = 36 :=
by sorry

end NUMINAMATH_GPT_find_two_digit_numbers_l397_39716


namespace NUMINAMATH_GPT_original_number_is_0_02_l397_39733

theorem original_number_is_0_02 (x : ℝ) (h : 10000 * x = 4 / x) : x = 0.02 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_0_02_l397_39733


namespace NUMINAMATH_GPT_none_of_these_l397_39720

theorem none_of_these (a T : ℝ) : 
  ¬(∀ (x y : ℝ), 4 * T * x + 2 * a^2 * y + 4 * a * T = 0) ∧ 
  ¬(∀ (x y : ℝ), 4 * T * x - 2 * a^2 * y + 4 * a * T = 0) ∧ 
  ¬(∀ (x y : ℝ), 4 * T * x + 2 * a^2 * y - 4 * a * T = 0) ∧ 
  ¬(∀ (x y : ℝ), 4 * T * x - 2 * a^2 * y - 4 * a * T = 0) :=
sorry

end NUMINAMATH_GPT_none_of_these_l397_39720


namespace NUMINAMATH_GPT_solve_trig_equation_l397_39797

theorem solve_trig_equation (x : ℝ) : 
  (∃ (k : ℤ), x = (Real.pi / 16) * (4 * k + 1)) ↔ 2 * (Real.cos (4 * x) - Real.sin x * Real.cos (3 * x)) = Real.sin (4 * x) + Real.sin (2 * x) :=
by
  -- The full proof detail goes here.
  sorry

end NUMINAMATH_GPT_solve_trig_equation_l397_39797


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_l397_39788

theorem line_passes_through_fixed_point (k : ℝ) : (k * 2 - 1 + 1 - 2 * k = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_l397_39788


namespace NUMINAMATH_GPT_sum_of_dice_less_than_10_probability_l397_39764

/-
  Given:
  - A fair die with faces labeled 1, 2, 3, 4, 5, 6.
  - The die is rolled twice.

  Prove that the probability that the sum of the face values is less than 10 is 5/6.
-/

noncomputable def probability_sum_less_than_10 : ℚ :=
  let total_outcomes := 36
  let favorable_outcomes := 30
  favorable_outcomes / total_outcomes

theorem sum_of_dice_less_than_10_probability :
  probability_sum_less_than_10 = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_dice_less_than_10_probability_l397_39764


namespace NUMINAMATH_GPT_num_bags_of_cookies_l397_39770

theorem num_bags_of_cookies (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 703) (h2 : cookies_per_bag = 19) : total_cookies / cookies_per_bag = 37 :=
by
  sorry

end NUMINAMATH_GPT_num_bags_of_cookies_l397_39770


namespace NUMINAMATH_GPT_commission_percentage_l397_39722

theorem commission_percentage (fixed_salary second_base_salary sales_amount earning: ℝ) (commission: ℝ) 
  (h1 : fixed_salary = 1800)
  (h2 : second_base_salary = 1600)
  (h3 : sales_amount = 5000)
  (h4 : earning = 1800) :
  fixed_salary = second_base_salary + (sales_amount * commission) → 
  commission * 100 = 4 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_commission_percentage_l397_39722


namespace NUMINAMATH_GPT_BD_is_diameter_of_circle_l397_39799

variables {A B C D X Y : Type} [MetricSpace A] [MetricSpace B] 
  [MetricSpace C] [MetricSpace D] [MetricSpace X] [MetricSpace Y]

-- Assume these four points lie on a circle with certain ordering
variables (circ : Circle A B C D)

-- Given conditions
variables (h1 : circ.AB < circ.AD)
variables (h2 : circ.BC > circ.CD)

-- Points X and Y are where angle bisectors meet the circle again
variables (h3 : circ.bisects_angle_BAD_at X)
variables (h4 : circ.bisects_angle_BCD_at Y)

-- Hexagon sides with four equal lengths
variables (hex_equal : circ.hexagon_sides_equal_length A B X C D Y)

-- Prove that BD is a diameter
theorem BD_is_diameter_of_circle : circ.is_diameter BD := 
by
  sorry

end NUMINAMATH_GPT_BD_is_diameter_of_circle_l397_39799


namespace NUMINAMATH_GPT_admittedApplicants_l397_39714

-- Definitions for the conditions in the problem
def totalApplicants : ℕ := 70
def task1Applicants : ℕ := 35
def task2Applicants : ℕ := 48
def task3Applicants : ℕ := 64
def task4Applicants : ℕ := 63

-- The proof statement
theorem admittedApplicants : 
  ∀ (totalApplicants task3Applicants task4Applicants : ℕ),
  totalApplicants = 70 →
  task3Applicants = 64 →
  task4Applicants = 63 →
  ∃ (interApplicants : ℕ), interApplicants = 57 :=
by
  intros totalApplicants task3Applicants task4Applicants
  intros h_totalApps h_task3Apps h_task4Apps
  sorry

end NUMINAMATH_GPT_admittedApplicants_l397_39714


namespace NUMINAMATH_GPT_determinant_of_A_l397_39703

-- Define the 2x2 matrix A
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![7, -2], ![-3, 6]]

-- The statement to be proved
theorem determinant_of_A : Matrix.det A = 36 := 
  by sorry

end NUMINAMATH_GPT_determinant_of_A_l397_39703


namespace NUMINAMATH_GPT_day_crew_fraction_l397_39749

theorem day_crew_fraction (D W : ℕ) (h1 : ∀ n, n = D / 4) (h2 : ∀ w, w = 4 * W / 5) :
  (D * W) / ((D * W) + ((D / 4) * (4 * W / 5))) = 5 / 6 :=
by 
  sorry

end NUMINAMATH_GPT_day_crew_fraction_l397_39749


namespace NUMINAMATH_GPT_complement_union_A_B_complement_A_intersection_B_l397_39743

open Set

-- Definitions of A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- Proving the complement of A ∪ B
theorem complement_union_A_B : (A ∪ B)ᶜ = {x : ℝ | x ≤ 2 ∨ 10 ≤ x} :=
by sorry

-- Proving the intersection of the complement of A with B
theorem complement_A_intersection_B : (Aᶜ ∩ B) = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} :=
by sorry

end NUMINAMATH_GPT_complement_union_A_B_complement_A_intersection_B_l397_39743


namespace NUMINAMATH_GPT_solve_for_k_l397_39760

theorem solve_for_k (k : ℝ) (h₁ : ∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) (h₂ : k ≠ 0) : k = 6 :=
sorry

end NUMINAMATH_GPT_solve_for_k_l397_39760


namespace NUMINAMATH_GPT_find_real_solutions_l397_39736

noncomputable def polynomial_expression (x : ℝ) : ℝ := (x - 2)^2 * (x - 4) * (x - 1)

theorem find_real_solutions :
  ∀ (x : ℝ), (x ≠ 3) ∧ (x ≠ 5) ∧ (polynomial_expression x = 1) ↔ (x = 1 ∨ x = (3 + Real.sqrt 3) / 2 ∨ x = (3 - Real.sqrt 3) / 2) := sorry

end NUMINAMATH_GPT_find_real_solutions_l397_39736


namespace NUMINAMATH_GPT_brenda_age_l397_39741

variable (A B J : ℕ)

theorem brenda_age :
  (A = 3 * B) →
  (J = B + 6) →
  (A = J) →
  (B = 3) :=
by
  intros h1 h2 h3
  -- condition: A = 3 * B
  -- condition: J = B + 6
  -- condition: A = J
  -- prove B = 3
  sorry

end NUMINAMATH_GPT_brenda_age_l397_39741


namespace NUMINAMATH_GPT_micah_water_l397_39798

theorem micah_water (x : ℝ) (h1 : 3 * x + x = 6) : x = 1.5 :=
sorry

end NUMINAMATH_GPT_micah_water_l397_39798


namespace NUMINAMATH_GPT_escher_probability_l397_39758

def num_arrangements (n : ℕ) : ℕ := Nat.factorial n

def favorable_arrangements (total_art : ℕ) (escher_prints : ℕ) : ℕ :=
  num_arrangements (total_art - escher_prints + 1) * num_arrangements escher_prints

def total_arrangements (total_art : ℕ) : ℕ :=
  num_arrangements total_art

def prob_all_escher_consecutive (total_art : ℕ) (escher_prints : ℕ) : ℚ :=
  favorable_arrangements total_art escher_prints / total_arrangements total_art

theorem escher_probability :
  prob_all_escher_consecutive 12 4 = 1/55 :=
by
  sorry

end NUMINAMATH_GPT_escher_probability_l397_39758


namespace NUMINAMATH_GPT_starting_number_is_33_l397_39739

theorem starting_number_is_33 (n : ℕ)
  (h1 : ∀ k, (33 + k * 11 ≤ 79) → (k < 5))
  (h2 : ∀ k, (k < 5) → (33 + k * 11 ≤ 79)) :
  n = 33 :=
sorry

end NUMINAMATH_GPT_starting_number_is_33_l397_39739


namespace NUMINAMATH_GPT_percent_of_a_is_b_percent_of_d_is_c_percent_of_d_is_e_l397_39746

variables (a b c d e : ℝ)

-- Conditions
def condition1 : Prop := c = 0.25 * a
def condition2 : Prop := c = 0.50 * b
def condition3 : Prop := d = 0.40 * a
def condition4 : Prop := d = 0.20 * b
def condition5 : Prop := e = 0.35 * d
def condition6 : Prop := e = 0.15 * c

-- Proof Problem Statements
theorem percent_of_a_is_b (h1 : condition1 a c) (h2 : condition2 c b) : b = 0.5 * a := sorry

theorem percent_of_d_is_c (h1 : condition1 a c) (h3 : condition3 a d) : c = 0.625 * d := sorry

theorem percent_of_d_is_e (h5 : condition5 e d) : e = 0.35 * d := sorry

end NUMINAMATH_GPT_percent_of_a_is_b_percent_of_d_is_c_percent_of_d_is_e_l397_39746


namespace NUMINAMATH_GPT_batsman_average_after_17th_inning_l397_39762

-- Definitions for the conditions
def runs_scored_in_17th_inning : ℝ := 95
def increase_in_average : ℝ := 2.5

-- Lean statement encapsulating the problem
theorem batsman_average_after_17th_inning (A : ℝ) (h : 16 * A + runs_scored_in_17th_inning = 17 * (A + increase_in_average)) :
  A + increase_in_average = 55 := 
sorry

end NUMINAMATH_GPT_batsman_average_after_17th_inning_l397_39762


namespace NUMINAMATH_GPT_sum_of_digits_7_pow_11_l397_39772

/-
  Let's define the problem with the given conditions and expected answer.
  We need to prove that the sum of the tens digit and the ones digit of 7^11 is 7.
-/

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_7_pow_11 : tens_digit (7 ^ 11) + ones_digit (7 ^ 11) = 7 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_7_pow_11_l397_39772


namespace NUMINAMATH_GPT_coterminal_angle_equivalence_l397_39718

theorem coterminal_angle_equivalence (k : ℤ) : ∃ n : ℤ, -463 % 360 = (k * 360 + 257) % 360 :=
by
  sorry

end NUMINAMATH_GPT_coterminal_angle_equivalence_l397_39718


namespace NUMINAMATH_GPT_total_distance_traveled_l397_39719

/-- The total distance traveled by Mr. and Mrs. Hugo over three days. -/
theorem total_distance_traveled :
  let first_day := 200
  let second_day := (3/4 : ℚ) * first_day
  let third_day := (1/2 : ℚ) * (first_day + second_day)
  first_day + second_day + third_day = 525 := by
  let first_day := 200
  let second_day := (3/4 : ℚ) * first_day
  let third_day := (1/2 : ℚ) * (first_day + second_day)
  have h1 : first_day + second_day + third_day = 525 := by
    sorry
  exact h1

end NUMINAMATH_GPT_total_distance_traveled_l397_39719


namespace NUMINAMATH_GPT_curve_tangents_intersection_l397_39778

theorem curve_tangents_intersection (a : ℝ) :
  (∃ x₀ y₀, y₀ = Real.exp x₀ ∧ y₀ = (x₀ + a)^2 ∧ Real.exp x₀ = 2 * (x₀ + a)) → a = 2 - Real.log 4 :=
by
  sorry

end NUMINAMATH_GPT_curve_tangents_intersection_l397_39778


namespace NUMINAMATH_GPT_find_k_l397_39744

theorem find_k (k : ℝ) : (∃ b : ℝ, (x : ℝ) → x^2 - 60 * x + k = (x + b)^2) → k = 900 :=
by 
  sorry

end NUMINAMATH_GPT_find_k_l397_39744


namespace NUMINAMATH_GPT_widget_difference_l397_39730

variable (w t : ℕ)

def monday_widgets (w t : ℕ) : ℕ := w * t
def tuesday_widgets (w t : ℕ) : ℕ := (w + 5) * (t - 3)

theorem widget_difference (h : w = 3 * t) :
  monday_widgets w t - tuesday_widgets w t = 4 * t + 15 :=
by
  sorry

end NUMINAMATH_GPT_widget_difference_l397_39730


namespace NUMINAMATH_GPT_paintings_after_30_days_l397_39723

theorem paintings_after_30_days (paintings_per_day : ℕ) (initial_paintings : ℕ) (days : ℕ)
    (h1 : paintings_per_day = 2)
    (h2 : initial_paintings = 20)
    (h3 : days = 30) :
    initial_paintings + paintings_per_day * days = 80 := by
  sorry

end NUMINAMATH_GPT_paintings_after_30_days_l397_39723


namespace NUMINAMATH_GPT_no_matching_formula_l397_39732

def xy_pairs : List (ℕ × ℕ) := [(1, 5), (2, 15), (3, 35), (4, 69), (5, 119)]

def formula_a (x : ℕ) : ℕ := x^3 + x^2 + x + 2
def formula_b (x : ℕ) : ℕ := 3 * x^2 + 2 * x + 1
def formula_c (x : ℕ) : ℕ := 2 * x^3 - x + 4
def formula_d (x : ℕ) : ℕ := 3 * x^3 + 2 * x^2 + x + 1

theorem no_matching_formula :
  ¬ (∀ (pair : ℕ × ℕ), pair ∈ xy_pairs → pair.snd = formula_a pair.fst) ∧
  ¬ (∀ (pair : ℕ × ℕ), pair ∈ xy_pairs → pair.snd = formula_b pair.fst) ∧
  ¬ (∀ (pair : ℕ × ℕ), pair ∈ xy_pairs → pair.snd = formula_c pair.fst) ∧
  ¬ (∀ (pair : ℕ × ℕ), pair ∈ xy_pairs → pair.snd = formula_d pair.fst) :=
by
  sorry

end NUMINAMATH_GPT_no_matching_formula_l397_39732


namespace NUMINAMATH_GPT_mean_of_sequence_starting_at_3_l397_39707

def arithmetic_sequence (start : ℕ) (n : ℕ) : List ℕ :=
List.range n |>.map (λ i => start + i)

def arithmetic_mean (seq : List ℕ) : ℚ := (seq.sum : ℚ) / seq.length

theorem mean_of_sequence_starting_at_3 : 
  ∀ (seq : List ℕ),
  seq = arithmetic_sequence 3 60 → 
  arithmetic_mean seq = 32.5 := 
by
  intros seq h
  rw [h]
  sorry

end NUMINAMATH_GPT_mean_of_sequence_starting_at_3_l397_39707


namespace NUMINAMATH_GPT_length_of_second_train_l397_39729

theorem length_of_second_train
  (length_first_train : ℝ)
  (speed_first_train : ℝ)
  (speed_second_train : ℝ)
  (time_to_cross : ℝ)
  (h1 : length_first_train = 270)
  (h2 : speed_first_train = 120)
  (h3 : speed_second_train = 80)
  (h4 : time_to_cross = 9) :
  ∃ length_second_train : ℝ, length_second_train = 229.95 :=
by
  sorry

end NUMINAMATH_GPT_length_of_second_train_l397_39729


namespace NUMINAMATH_GPT_mark_donates_cans_of_soup_l397_39731

theorem mark_donates_cans_of_soup:
  let n_shelters := 6
  let p_per_shelter := 30
  let c_per_person := 10
  let total_people := n_shelters * p_per_shelter
  let total_cans := total_people * c_per_person
  total_cans = 1800 :=
by sorry

end NUMINAMATH_GPT_mark_donates_cans_of_soup_l397_39731


namespace NUMINAMATH_GPT_cylinder_cut_is_cylinder_l397_39768

-- Define what it means to be a cylinder
structure Cylinder (r h : ℝ) : Prop :=
(r_pos : r > 0)
(h_pos : h > 0)

-- Define the condition of cutting a cylinder with two parallel planes
def cut_by_parallel_planes (c : Cylinder r h) (d : ℝ) : Prop :=
d > 0 ∧ d < h

-- Prove that the part between the parallel planes is still a cylinder
theorem cylinder_cut_is_cylinder (r h d : ℝ) (c : Cylinder r h) (H : cut_by_parallel_planes c d) :
  ∃ r' h', Cylinder r' h' :=
sorry

end NUMINAMATH_GPT_cylinder_cut_is_cylinder_l397_39768


namespace NUMINAMATH_GPT_geometric_seq_arith_condition_half_l397_39766

variables {a : ℕ → ℝ} {q : ℝ}

-- Conditions from the problem
def geometric_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q
def positive_terms (a : ℕ → ℝ) := ∀ n, a n > 0
def arithmetic_condition (a : ℕ → ℝ) (q : ℝ) := 
  a 1 = q * a 0 ∧ (1/2 : ℝ) * a 2 = a 1 + 2 * a 0

-- The statement to be proven
theorem geometric_seq_arith_condition_half (a : ℕ → ℝ) (q : ℝ) :
  geometric_seq a q →
  positive_terms a →
  arithmetic_condition a q →
  q = 2 →
  (a 2 + a 3) / (a 3 + a 4) = 1 / 2 :=
by
  intros h1 h2 h3 hq
  sorry

end NUMINAMATH_GPT_geometric_seq_arith_condition_half_l397_39766


namespace NUMINAMATH_GPT_circle_center_l397_39715

theorem circle_center (x y : ℝ) (h : x^2 + 8*x + y^2 - 4*y = 16) : (x, y) = (-4, 2) :=
by 
  sorry

end NUMINAMATH_GPT_circle_center_l397_39715


namespace NUMINAMATH_GPT_num_both_sports_l397_39783

def num_people := 310
def num_tennis := 138
def num_baseball := 255
def num_no_sport := 11

theorem num_both_sports : (num_tennis + num_baseball - (num_people - num_no_sport)) = 94 :=
by 
-- leave the proof out for now
sorry

end NUMINAMATH_GPT_num_both_sports_l397_39783


namespace NUMINAMATH_GPT_wand_cost_l397_39796

-- Conditions based on the problem
def initialWands := 3
def salePrice (x : ℝ) := x + 5
def totalCollected := 130
def soldWands := 2

-- Proof statement
theorem wand_cost (x : ℝ) : 
  2 * salePrice x = totalCollected → x = 60 := 
by 
  sorry

end NUMINAMATH_GPT_wand_cost_l397_39796


namespace NUMINAMATH_GPT_initial_money_l397_39771

/-- Given the following conditions:
  (1) June buys 4 maths books at $20 each.
  (2) June buys 6 more science books than maths books at $10 each.
  (3) June buys twice as many art books as maths books at $20 each.
  (4) June spends $160 on music books.
  Prove that June had initially $500 for buying school supplies. -/
theorem initial_money (maths_books : ℕ) (science_books : ℕ) (art_books : ℕ) (music_books_cost : ℕ)
  (h_math_books : maths_books = 4) (price_per_math_book : ℕ) (price_per_science_book : ℕ) 
  (price_per_art_book : ℕ) (price_per_music_books_cost : ℕ) (h_maths_price : price_per_math_book = 20)
  (h_science_books : science_books = maths_books + 6) (h_science_price : price_per_science_book = 10)
  (h_art_books : art_books = 2 * maths_books) (h_art_price : price_per_art_book = 20)
  (h_music_books_cost : music_books_cost = 160) :
  4 * 20 + (4 + 6) * 10 + (2 * 4) * 20 + 160 = 500 :=
by sorry

end NUMINAMATH_GPT_initial_money_l397_39771


namespace NUMINAMATH_GPT_inequality_holds_l397_39795

theorem inequality_holds : ∀ (n : ℕ), (n - 1)^(n + 1) * (n + 1)^(n - 1) < n^(2 * n) :=
by sorry

end NUMINAMATH_GPT_inequality_holds_l397_39795


namespace NUMINAMATH_GPT_annika_hike_distance_l397_39728

-- Define the conditions as definitions
def hiking_rate : ℝ := 10  -- rate of 10 minutes per kilometer
def total_minutes : ℝ := 35 -- total available time in minutes
def total_distance_east : ℝ := 3 -- total distance hiked east

-- Define the statement to prove
theorem annika_hike_distance : ∃ (x : ℝ), (x / hiking_rate) + ((total_distance_east - x) / hiking_rate) = (total_minutes - 30) / hiking_rate :=
by
  sorry

end NUMINAMATH_GPT_annika_hike_distance_l397_39728


namespace NUMINAMATH_GPT_frustum_slant_height_l397_39763

-- The setup: we are given specific conditions for a frustum resulting from cutting a cone
variable {r : ℝ} -- represents the radius of the upper base of the frustum
variable {h : ℝ} -- represents the slant height of the frustum
variable {h_removed : ℝ} -- represents the slant height of the removed cone

-- The given conditions
def upper_base_radius : ℝ := r
def lower_base_radius : ℝ := 4 * r
def slant_height_removed_cone : ℝ := 3

-- The proportion derived from similar triangles
def proportion (h r : ℝ) := (h / (4 * r)) = ((h + 3) / (5 * r))

-- The main statement: proving the slant height of the frustum is 9 cm
theorem frustum_slant_height (r : ℝ) (h : ℝ) (hr : proportion h r) : h = 9 :=
sorry

end NUMINAMATH_GPT_frustum_slant_height_l397_39763


namespace NUMINAMATH_GPT_largest_possible_perimeter_l397_39756

theorem largest_possible_perimeter (x : ℕ) (h1 : 1 < x) (h2 : x < 15) : 
  (7 + 8 + x) ≤ 29 := 
sorry

end NUMINAMATH_GPT_largest_possible_perimeter_l397_39756


namespace NUMINAMATH_GPT_distinct_balls_boxes_l397_39724

def count_distinct_distributions (balls : ℕ) (boxes : ℕ) : ℕ :=
  if balls = 7 ∧ boxes = 3 then 8 else 0

theorem distinct_balls_boxes :
  count_distinct_distributions 7 3 = 8 :=
by sorry

end NUMINAMATH_GPT_distinct_balls_boxes_l397_39724


namespace NUMINAMATH_GPT_highest_power_of_2_dividing_15_pow_4_minus_9_pow_4_l397_39793

theorem highest_power_of_2_dividing_15_pow_4_minus_9_pow_4 :
  (∃ k, 15^4 - 9^4 = 2^k * m ∧ ¬ ∃ m', m = 2 * m') ∧ (k = 5) :=
by
  sorry

end NUMINAMATH_GPT_highest_power_of_2_dividing_15_pow_4_minus_9_pow_4_l397_39793


namespace NUMINAMATH_GPT_smallest_number_of_coins_l397_39789

theorem smallest_number_of_coins :
  ∃ (n : ℕ), (∀ (a : ℕ), 5 ≤ a ∧ a < 100 → 
    ∃ (c : ℕ → ℕ), (a = 5 * c 0 + 10 * c 1 + 25 * c 2) ∧ 
    (c 0 + c 1 + c 2 = n)) ∧ n = 9 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_coins_l397_39789


namespace NUMINAMATH_GPT_square_of_binomial_b_value_l397_39753

theorem square_of_binomial_b_value (b : ℤ) (h : ∃ c : ℤ, 16 * (x : ℤ) * x + 40 * x + b = (4 * x + c) ^ 2) : b = 25 :=
sorry

end NUMINAMATH_GPT_square_of_binomial_b_value_l397_39753


namespace NUMINAMATH_GPT_vertex_of_quadratic_l397_39787

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := -3 * x^2 - 6 * x + 5

-- State the theorem for vertex coordinates
theorem vertex_of_quadratic :
  (∀ x : ℝ, quadratic_function (- (-6) / (2 * -3)) = quadratic_function 1)
  → (1, quadratic_function 1) = (1, 8) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_vertex_of_quadratic_l397_39787


namespace NUMINAMATH_GPT_consecutive_probability_l397_39786

-- Define the total number of ways to choose 2 episodes out of 6
def total_combinations : ℕ := Nat.choose 6 2

-- Define the number of ways to choose consecutive episodes
def consecutive_combinations : ℕ := 5

-- Define the probability of choosing consecutive episodes
def probability_of_consecutive : ℚ := consecutive_combinations / total_combinations

-- Theorem stating that the calculated probability should equal 1/3
theorem consecutive_probability : probability_of_consecutive = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_probability_l397_39786


namespace NUMINAMATH_GPT_age_of_15th_student_l397_39717

theorem age_of_15th_student 
  (avg_age_all : ℕ → ℕ → ℕ)
  (avg_age : avg_age_all 15 15 = 15)
  (avg_age_4 : avg_age_all 4 14 = 14)
  (avg_age_10 : avg_age_all 10 16 = 16) : 
  ∃ age15 : ℕ, age15 = 9 := 
by
  sorry

end NUMINAMATH_GPT_age_of_15th_student_l397_39717


namespace NUMINAMATH_GPT_geometric_sum_a4_a6_l397_39792

-- Definitions based on the conditions
def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sum_a4_a6 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h_pos : ∀ n, a n > 0) 
(h_cond : a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100) : a 4 + a 6 = 10 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sum_a4_a6_l397_39792


namespace NUMINAMATH_GPT_x_y_ge_two_l397_39745

open Real

theorem x_y_ge_two (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + x * y = 3) : 
  x + y ≥ 2 ∧ (x + y = 2 → x = 1 ∧ y = 1) :=
by {
 sorry
}

end NUMINAMATH_GPT_x_y_ge_two_l397_39745


namespace NUMINAMATH_GPT_p_sufficient_condition_neg_q_l397_39779

variables (p q : Prop)

theorem p_sufficient_condition_neg_q (hnecsuff_q : ¬p → q) (hnecsuff_p : ¬q → p) : (p → ¬q) :=
by
  sorry

end NUMINAMATH_GPT_p_sufficient_condition_neg_q_l397_39779


namespace NUMINAMATH_GPT_number_of_10_yuan_coins_is_1_l397_39774

theorem number_of_10_yuan_coins_is_1
  (n : ℕ) -- number of coins
  (v : ℕ) -- total value of coins
  (c1 c5 c10 c50 : ℕ) -- number of 1, 5, 10, and 50 yuan coins
  (h1 : n = 9) -- there are nine coins in total
  (h2 : v = 177) -- the total value of these coins is 177 yuan
  (h3 : c1 ≥ 1 ∧ c5 ≥ 1 ∧ c10 ≥ 1 ∧ c50 ≥ 1) -- at least one coin of each denomination
  (h4 : c1 + c5 + c10 + c50 = n) -- sum of all coins number is n
  (h5 : c1 * 1 + c5 * 5 + c10 * 10 + c50 * 50 = v) -- total value of all coins is v
  : c10 = 1 := 
sorry

end NUMINAMATH_GPT_number_of_10_yuan_coins_is_1_l397_39774


namespace NUMINAMATH_GPT_number_divisors_l397_39777

theorem number_divisors (p : ℕ) (h : p = 2^56 - 1) : ∃ x y : ℕ, 95 ≤ x ∧ x ≤ 105 ∧ 95 ≤ y ∧ y ≤ 105 ∧ p % x = 0 ∧ p % y = 0 ∧ x = 101 ∧ y = 127 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_divisors_l397_39777


namespace NUMINAMATH_GPT_sum_remainder_mod_9_l397_39751

theorem sum_remainder_mod_9 : 
  (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 9 = 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_remainder_mod_9_l397_39751


namespace NUMINAMATH_GPT_geometric_sequence_a_eq_one_l397_39726

theorem geometric_sequence_a_eq_one (a : ℝ) 
  (h₁ : ∃ (r : ℝ), a = 1 / (1 - r) ∧ r = a - 1/2 ∧ r ≠ 0) : 
  a = 1 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_a_eq_one_l397_39726


namespace NUMINAMATH_GPT_race_distance_l397_39740

-- Definitions for the conditions
def A_time : ℕ := 20
def B_time : ℕ := 25
def A_beats_B_by : ℕ := 14

-- Definition of the function to calculate whether the total distance D is correct
def total_distance : ℕ := 56

-- The theorem statement without proof
theorem race_distance (D : ℕ) (A_time B_time A_beats_B_by : ℕ)
  (hA : A_time = 20)
  (hB : B_time = 25)
  (hAB : A_beats_B_by = 14)
  (h_eq : (D / A_time) * B_time = D + A_beats_B_by) : 
  D = total_distance :=
sorry

end NUMINAMATH_GPT_race_distance_l397_39740


namespace NUMINAMATH_GPT_frustum_volume_l397_39790

noncomputable def volume_of_frustum (V₁ V₂ : ℝ) : ℝ :=
  V₁ - V₂

theorem frustum_volume : 
  let base_edge_original := 15
  let height_original := 10
  let base_edge_smaller := 9
  let height_smaller := 6
  let base_area_original := base_edge_original ^ 2
  let base_area_smaller := base_edge_smaller ^ 2
  let V_original := (1 / 3 : ℝ) * base_area_original * height_original
  let V_smaller := (1 / 3 : ℝ) * base_area_smaller * height_smaller
  volume_of_frustum V_original V_smaller = 588 := 
by
  sorry

end NUMINAMATH_GPT_frustum_volume_l397_39790


namespace NUMINAMATH_GPT_min_value_a4b3c2_l397_39757

theorem min_value_a4b3c2 {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 1/a + 1/b + 1/c = 9) :
  a ^ 4 * b ^ 3 * c ^ 2 ≥ 1 / 5184 := 
sorry

end NUMINAMATH_GPT_min_value_a4b3c2_l397_39757


namespace NUMINAMATH_GPT_lottery_win_amount_l397_39708

theorem lottery_win_amount (total_tax : ℝ) (federal_tax_rate : ℝ) (local_tax_rate : ℝ) (tax_paid : ℝ) :
  total_tax = tax_paid →
  federal_tax_rate = 0.25 →
  local_tax_rate = 0.15 →
  tax_paid = 18000 →
  ∃ x : ℝ, x = 49655 :=
by
  intros h1 h2 h3 h4
  use (tax_paid / (federal_tax_rate + local_tax_rate * (1 - federal_tax_rate))), by
    norm_num at h1 h2 h3 h4
    sorry

end NUMINAMATH_GPT_lottery_win_amount_l397_39708
