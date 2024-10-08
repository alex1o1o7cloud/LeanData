import Mathlib

namespace number_of_solutions_sine_quadratic_l93_93935

theorem number_of_solutions_sine_quadratic :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 * Real.pi → 3 * (Real.sin x) ^ 2 - 5 * (Real.sin x) + 2 = 0 →
  ∃ a b c, x = a ∨ x = b ∨ x = c ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c :=
sorry

end number_of_solutions_sine_quadratic_l93_93935


namespace solve_for_A_l93_93002

def clubsuit (A B : ℤ) : ℤ := 3 * A + 2 * B + 7

theorem solve_for_A (A : ℤ) : (clubsuit A 6 = 70) -> (A = 17) :=
by
  sorry

end solve_for_A_l93_93002


namespace minimum_value_fraction_1_x_plus_1_y_l93_93260

theorem minimum_value_fraction_1_x_plus_1_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) :
  1 / x + 1 / y = 1 :=
sorry

end minimum_value_fraction_1_x_plus_1_y_l93_93260


namespace transformed_A_coordinates_l93_93722

open Real

def rotate_90_clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.snd, -p.fst)

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.fst, p.snd)

def A : ℝ × ℝ := (-3, 2)

theorem transformed_A_coordinates :
  reflect_over_y_axis (rotate_90_clockwise A) = (-2, 3) :=
by
  sorry

end transformed_A_coordinates_l93_93722


namespace determine_p_l93_93824

theorem determine_p (m : ℕ) (p : ℕ) (h1: m = 34) 
  (h2: (1 : ℝ)^ (m + 1) / 5^ (m + 1) * 1^18 / 4^18 = 1 / (2 * 10^ p)) : 
  p = 35 := by sorry

end determine_p_l93_93824


namespace largest_unpayable_soldo_l93_93255

theorem largest_unpayable_soldo : ∃ N : ℕ, N ≤ 50 ∧ (∀ a b : ℕ, a * 5 + b * 6 ≠ N) ∧ (∀ M : ℕ, (M ≤ 50 ∧ ∀ a b : ℕ, a * 5 + b * 6 ≠ M) → M ≤ 19) :=
by
  sorry

end largest_unpayable_soldo_l93_93255


namespace number_of_students_to_bring_donuts_l93_93594

theorem number_of_students_to_bring_donuts (students_brownies students_cookies students_donuts : ℕ) :
  (students_brownies * 12 * 2) + (students_cookies * 24 * 2) + (students_donuts * 12 * 2) = 2040 →
  students_brownies = 30 →
  students_cookies = 20 →
  students_donuts = 15 :=
by
  -- Proof skipped
  sorry

end number_of_students_to_bring_donuts_l93_93594


namespace jessica_withdrawal_l93_93514

/-- Jessica withdrew some money from her bank account, causing her account balance to decrease by 2/5.
    She then deposited an amount equal to 1/4 of the remaining balance. The final balance in her bank account is $750.
    Prove that Jessica initially withdrew $400. -/
theorem jessica_withdrawal (X W : ℝ) 
  (initial_eq : W = (2 / 5) * X)
  (remaining_eq : X * (3 / 5) + (1 / 4) * (X * (3 / 5)) = 750) :
  W = 400 := 
sorry

end jessica_withdrawal_l93_93514


namespace quadratic_inequality_solution_set_l93_93160

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - x - 3 ≥ 0} = {x : ℝ | x ≤ -1 ∨ x ≥ 3 / 2} :=
sorry

end quadratic_inequality_solution_set_l93_93160


namespace parity_of_expression_l93_93522

theorem parity_of_expression (e m : ℕ) (he : (∃ k : ℕ, e = 2 * k)) : Odd (e ^ 2 + 3 ^ m) :=
  sorry

end parity_of_expression_l93_93522


namespace ratio_of_perimeters_l93_93312

-- Define lengths of the rectangular patch
def length_rect : ℝ := 400
def width_rect : ℝ := 300

-- Define the length of the side of the square patch
def side_square : ℝ := 700

-- Define the perimeters of both patches
def P_square : ℝ := 4 * side_square
def P_rectangle : ℝ := 2 * (length_rect + width_rect)

-- Theorem stating the ratio of the perimeters
theorem ratio_of_perimeters : P_square / P_rectangle = 2 :=
by sorry

end ratio_of_perimeters_l93_93312


namespace parallelogram_sides_l93_93677

theorem parallelogram_sides (a b : ℕ): 
  (a = 3 * b) ∧ (2 * a + 2 * b = 24) → (a = 9) ∧ (b = 3) :=
by
  sorry

end parallelogram_sides_l93_93677


namespace a_minus_b_eq_three_l93_93007

theorem a_minus_b_eq_three (a b : ℝ) (h : (a+bi) * i = 1 + 2 * i) : a - b = 3 :=
by
  sorry

end a_minus_b_eq_three_l93_93007


namespace gym_monthly_revenue_l93_93738

-- Defining the conditions
def charge_per_session : ℕ := 18
def sessions_per_month : ℕ := 2
def number_of_members : ℕ := 300

-- Defining the question as a theorem statement
theorem gym_monthly_revenue : 
  (number_of_members * (charge_per_session * sessions_per_month)) = 10800 := 
by 
  -- Skip the proof, verifying the statement only
  sorry

end gym_monthly_revenue_l93_93738


namespace min_height_regular_quadrilateral_pyramid_l93_93301

theorem min_height_regular_quadrilateral_pyramid (r : ℝ) (a : ℝ) (h : 2 * r < a / 2) : 
  ∃ x : ℝ, (0 < x) ∧ (∃ V : ℝ, ∀ x' : ℝ, V = (a^2 * x) / 3 ∧ (∀ x' ≠ x, V < (a^2 * x') / 3)) ∧ x = (r * (5 + Real.sqrt 17)) / 2 :=
sorry

end min_height_regular_quadrilateral_pyramid_l93_93301


namespace muffin_half_as_expensive_as_banana_l93_93110

-- Define Susie's expenditure in terms of muffin cost (m) and banana cost (b)
def susie_expenditure (m b : ℝ) : ℝ := 5 * m + 2 * b

-- Define Calvin's expenditure as three times Susie's expenditure
def calvin_expenditure_via_susie (m b : ℝ) : ℝ := 3 * (susie_expenditure m b)

-- Define Calvin's direct expenditure on muffins and bananas
def calvin_direct_expenditure (m b : ℝ) : ℝ := 3 * m + 12 * b

-- Formulate the theorem stating the relationship between muffin and banana costs
theorem muffin_half_as_expensive_as_banana (m b : ℝ) 
  (h₁ : susie_expenditure m b = 5 * m + 2 * b)
  (h₂ : calvin_expenditure_via_susie m b = calvin_direct_expenditure m b) : 
  m = (1/2) * b := 
by {
  -- These conditions automatically fulfill the given problem requirements.
  sorry
}

end muffin_half_as_expensive_as_banana_l93_93110


namespace sin_1320_eq_neg_sqrt_3_div_2_l93_93709

theorem sin_1320_eq_neg_sqrt_3_div_2 : Real.sin (1320 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_1320_eq_neg_sqrt_3_div_2_l93_93709


namespace additional_sugar_is_correct_l93_93409

def sugar_needed : ℝ := 450
def sugar_in_house : ℝ := 287
def sugar_in_basement_kg : ℝ := 50
def kg_to_lbs : ℝ := 2.20462

def sugar_in_basement : ℝ := sugar_in_basement_kg * kg_to_lbs
def total_sugar : ℝ := sugar_in_house + sugar_in_basement
def additional_sugar_needed : ℝ := sugar_needed - total_sugar

theorem additional_sugar_is_correct : additional_sugar_needed = 52.769 := by
  sorry

end additional_sugar_is_correct_l93_93409


namespace evie_l93_93185

variable (Evie_current_age : ℕ) 

theorem evie's_age_in_one_year
  (h : Evie_current_age + 4 = 3 * (Evie_current_age - 2)) : 
  Evie_current_age + 1 = 6 :=
by
  sorry

end evie_l93_93185


namespace geometry_biology_overlap_diff_l93_93202

theorem geometry_biology_overlap_diff :
  ∀ (total_students geometry_students biology_students : ℕ),
  total_students = 232 →
  geometry_students = 144 →
  biology_students = 119 →
  (max geometry_students biology_students - max 0 (geometry_students + biology_students - total_students)) = 88 :=
by
  intros total_students geometry_students biology_students
  sorry

end geometry_biology_overlap_diff_l93_93202


namespace consecutive_sums_permutations_iff_odd_l93_93979

theorem consecutive_sums_permutations_iff_odd (n : ℕ) (h : n ≥ 2) :
  (∃ (a b : Fin n → ℕ), (∀ i, 1 ≤ a i ∧ a i ≤ n) ∧ (∀ i, 1 ≤ b i ∧ b i ≤ n) ∧
    ∃ N, ∀ i, a i + b i = N + i) ↔ (Odd n) :=
by
  sorry

end consecutive_sums_permutations_iff_odd_l93_93979


namespace student_avg_always_greater_l93_93034

theorem student_avg_always_greater (x y z : ℝ) (h1 : x < y) (h2 : y < z) : 
  ( ( (x + y) / 2 + z) / 2 ) > ( (x + y + z) / 3 ) :=
by
  sorry

end student_avg_always_greater_l93_93034


namespace trapezium_distance_l93_93848

variable (a b h : ℝ)

theorem trapezium_distance (h_pos : 0 < h) (a_pos : 0 < a) (b_pos : 0 < b)
  (area_eq : 270 = 1/2 * (a + b) * h) (a_eq : a = 20) (b_eq : b = 16) : h = 15 :=
by {
  sorry
}

end trapezium_distance_l93_93848


namespace expand_product_l93_93376

theorem expand_product : ∀ (x : ℝ), (x + 2) * (x^2 - 4 * x + 1) = x^3 - 2 * x^2 - 7 * x + 2 :=
by 
  intro x
  sorry

end expand_product_l93_93376


namespace fraction_area_above_line_l93_93759

-- Define the problem conditions
def point1 : ℝ × ℝ := (4, 1)
def point2 : ℝ × ℝ := (9, 5)
def vertex1 : ℝ × ℝ := (4, 0)
def vertex2 : ℝ × ℝ := (9, 0)
def vertex3 : ℝ × ℝ := (9, 5)
def vertex4 : ℝ × ℝ := (4, 5)

-- Define the theorem statement
theorem fraction_area_above_line :
  let area_square := 25
  let area_below_line := 2.5
  let area_above_line := area_square - area_below_line
  area_above_line / area_square = 9 / 10 :=
by
  sorry -- Proof omitted

end fraction_area_above_line_l93_93759


namespace milk_butterfat_problem_l93_93079

variable (x : ℝ)

def butterfat_10_percent (x : ℝ) := 0.10 * x
def butterfat_35_percent_in_8_gallons : ℝ := 0.35 * 8
def total_milk (x : ℝ) := x + 8
def total_butterfat (x : ℝ) := 0.20 * (x + 8)

theorem milk_butterfat_problem 
    (h : butterfat_10_percent x + butterfat_35_percent_in_8_gallons = total_butterfat x) : x = 12 :=
by
  sorry

end milk_butterfat_problem_l93_93079


namespace point_in_fourth_quadrant_l93_93204

def in_fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant :
  in_fourth_quadrant (1, -2) ∧
  ¬ in_fourth_quadrant (2, 1) ∧
  ¬ in_fourth_quadrant (-2, 1) ∧
  ¬ in_fourth_quadrant (-1, -3) :=
by
  sorry

end point_in_fourth_quadrant_l93_93204


namespace average_temperature_problem_l93_93335

variable {T W Th F : ℝ}

theorem average_temperature_problem (h1 : (W + Th + 44) / 3 = 34) (h2 : T = 38) : 
  (T + W + Th) / 3 = 32 := by
  sorry

end average_temperature_problem_l93_93335


namespace f_at_4_l93_93620

-- Define the conditions on the function f
variable (f : ℝ → ℝ)
variable (h_domain : true) -- All ℝ → ℝ functions have ℝ as their domain.

-- f is an odd function
axiom h_odd : ∀ x : ℝ, f (-x) = -f x

-- Given functional equation
axiom h_eqn : ∀ x : ℝ, f (2 * x - 3) - 2 * f (3 * x - 10) + f (x - 3) = 28 - 6 * x 

-- The goal is to determine the value of f(4), which should be 8.
theorem f_at_4 : f 4 = 8 :=
sorry

end f_at_4_l93_93620


namespace polynomial_roots_l93_93451

theorem polynomial_roots : ∀ x : ℝ, (x^3 - 4*x^2 - x + 4) * (x - 3) * (x + 2) = 0 ↔ 
  (x = -2 ∨ x = -1 ∨ x = 1 ∨ x = 3 ∨ x = 4) :=
by 
  sorry

end polynomial_roots_l93_93451


namespace minimum_numbers_to_form_triangle_l93_93779

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem minimum_numbers_to_form_triangle :
  ∀ (S : Finset ℕ), (S ⊆ Finset.range 1001) →
    16 ≤ S.card →
    ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ {a, b, c} ⊆ S ∧ is_triangle a b c :=
by
  sorry

end minimum_numbers_to_form_triangle_l93_93779


namespace proportional_function_quadrants_l93_93764

theorem proportional_function_quadrants (k : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y > 0 ∧ y = k * x) ∧ (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ y = k * x) → k < 0 :=
by
  sorry

end proportional_function_quadrants_l93_93764


namespace red_and_purple_probability_l93_93741

def total_balls : ℕ := 120
def white_balls : ℕ := 30
def green_balls : ℕ := 25
def yellow_balls : ℕ := 24
def red_balls : ℕ := 20
def blue_balls : ℕ := 10
def purple_balls : ℕ := 5
def orange_balls : ℕ := 4
def gray_balls : ℕ := 2

def probability_red_purple : ℚ := 5 / 357

theorem red_and_purple_probability :
  ((red_balls / total_balls) * (purple_balls / (total_balls - 1)) +
  (purple_balls / total_balls) * (red_balls / (total_balls - 1))) = probability_red_purple :=
by
  sorry

end red_and_purple_probability_l93_93741


namespace bottles_difference_l93_93929

noncomputable def Donald_drinks_bottles (P: ℕ): ℕ := 2 * P + 3
noncomputable def Paul_drinks_bottles: ℕ := 3
noncomputable def actual_Donald_bottles: ℕ := 9

theorem bottles_difference:
  actual_Donald_bottles - 2 * Paul_drinks_bottles = 3 :=
by 
  sorry

end bottles_difference_l93_93929


namespace m_range_satisfies_inequality_l93_93100

open Real

noncomputable def f (x : ℝ) : ℝ := -2 * x + sin x

theorem m_range_satisfies_inequality :
  ∀ (m : ℝ), f (2 * m ^ 2 - m + π - 1) ≥ -2 * π ↔ -1 / 2 ≤ m ∧ m ≤ 1 := 
by
  sorry

end m_range_satisfies_inequality_l93_93100


namespace distance_B_to_center_l93_93850

/-- Definitions for the geometrical scenario -/
structure NotchedCircleGeom where
  radius : ℝ
  A_pos : ℝ × ℝ
  B_pos : ℝ × ℝ
  C_pos : ℝ × ℝ
  AB_len : ℝ
  BC_len : ℝ
  angle_ABC_right : Prop
  
  -- Conditions derived from problem statement
  radius_eq_sqrt72 : radius = Real.sqrt 72
  AB_len_eq_8 : AB_len = 8
  BC_len_eq_3 : BC_len = 3
  angle_ABC_right_angle : angle_ABC_right
  
/-- Problem statement -/
theorem distance_B_to_center (geom : NotchedCircleGeom) :
  let x := geom.B_pos.1
  let y := geom.B_pos.2
  x^2 + y^2 = 50 :=
sorry

end distance_B_to_center_l93_93850


namespace jason_cutting_grass_time_l93_93921

-- Conditions
def time_to_cut_one_lawn : ℕ := 30 -- in minutes
def lawns_cut_each_day : ℕ := 8
def days : ℕ := 2
def minutes_in_an_hour : ℕ := 60

-- Proof that the number of hours Jason spends cutting grass over the weekend is 8
theorem jason_cutting_grass_time:
  ((lawns_cut_each_day * days) * time_to_cut_one_lawn) / minutes_in_an_hour = 8 :=
by
  sorry

end jason_cutting_grass_time_l93_93921


namespace andrew_start_age_l93_93567

-- Define the conditions
def annual_donation : ℕ := 7
def current_age : ℕ := 29
def total_donation : ℕ := 133

-- The theorem to prove
theorem andrew_start_age : (total_donation / annual_donation) = (current_age - 10) :=
by
  sorry

end andrew_start_age_l93_93567


namespace students_in_class_l93_93933

theorem students_in_class (S : ℕ) (h1 : S / 3 + 2 * S / 5 + 12 = S) : S = 45 :=
sorry

end students_in_class_l93_93933


namespace simplify_expression_l93_93444

theorem simplify_expression (w x : ℝ) :
  3 * w + 6 * w + 9 * w + 12 * w + 15 * w - 2 * x - 4 * x - 6 * x - 8 * x - 10 * x + 24 = 
  45 * w - 30 * x + 24 :=
by sorry

end simplify_expression_l93_93444


namespace rachel_math_homework_pages_l93_93983

theorem rachel_math_homework_pages (M : ℕ) 
  (h1 : 23 = M + (M + 3)) : M = 10 :=
by {
  sorry
}

end rachel_math_homework_pages_l93_93983


namespace circle_placement_in_rectangle_l93_93557

theorem circle_placement_in_rectangle
  (L W : ℝ) (n : ℕ) (side_length diameter : ℝ)
  (h_dim : L = 20) (w_dim : W = 25)
  (h_squares : n = 120) (h_side_length : side_length = 1)
  (h_diameter : diameter = 1) :
  ∃ (x y : ℝ) (circle_radius : ℝ), 
    circle_radius = diameter / 2 ∧
    0 ≤ x ∧ x + diameter / 2 ≤ L ∧ 
    0 ≤ y ∧ y + diameter / 2 ≤ W ∧ 
    ∀ (i : ℕ) (hx : i < n) (sx sy : ℝ),
      0 ≤ sx ∧ sx + side_length ≤ L ∧
      0 ≤ sy ∧ sy + side_length ≤ W ∧
      dist (x, y) (sx + side_length / 2, sy + side_length / 2) ≥ diameter / 2 := 
sorry

end circle_placement_in_rectangle_l93_93557


namespace relationship_between_a_and_b_l93_93140

open Real

theorem relationship_between_a_and_b
  (a b x : ℝ)
  (h1 : a ≠ 1)
  (h2 : b ≠ 1)
  (h3 : 4 * (log x / log a)^3 + 5 * (log x / log b)^3 = 7 * (log x)^3) :
  b = a ^ (3 / 5)^(1 / 3) := 
sorry

end relationship_between_a_and_b_l93_93140


namespace sin_minus_pi_over_3_eq_neg_four_fifths_l93_93359

theorem sin_minus_pi_over_3_eq_neg_four_fifths
  (α : ℝ)
  (h : Real.cos (α + π / 6) = 4 / 5) :
  Real.sin (α - π / 3) = - (4 / 5) :=
by
  sorry

end sin_minus_pi_over_3_eq_neg_four_fifths_l93_93359


namespace find_value_of_a2004_b2004_l93_93088

-- Given Definitions and Conditions
def a : ℝ := sorry
def b : ℝ := sorry
def A : Set ℝ := {a, a^2, a * b}
def B : Set ℝ := {1, a, b}

-- The theorem statement
theorem find_value_of_a2004_b2004 (h : A = B) : a ^ 2004 + b ^ 2004 = 1 :=
sorry

end find_value_of_a2004_b2004_l93_93088


namespace Marla_laps_per_hour_l93_93794

theorem Marla_laps_per_hour (M : ℝ) :
  (0.8 * M = 0.8 * 5 + 4) → M = 10 :=
by
  sorry

end Marla_laps_per_hour_l93_93794


namespace hyperbola_eccentricity_l93_93004

theorem hyperbola_eccentricity :
  ∀ (a b c : ℝ), a = 3 → b = 4 → c = Real.sqrt (a^2 + b^2) → c / a = 5 / 3 :=
by
  intros a b c ha hb h_eq
  sorry

end hyperbola_eccentricity_l93_93004


namespace find_smallest_number_l93_93634

theorem find_smallest_number (x y z : ℝ) 
  (h1 : x + y + z = 150) 
  (h2 : y = 3 * x + 10) 
  (h3 : z = x^2 - 5) 
  : x = 10.21 :=
sorry

end find_smallest_number_l93_93634


namespace dollar_eval_l93_93658

def dollar (a b : ℝ) : ℝ := (a^2 - b^2)^2

theorem dollar_eval (x : ℝ) : dollar (x^3 + x) (x - x^3) = 16 * x^8 :=
by
  sorry

end dollar_eval_l93_93658


namespace solve_eqn_l93_93832

theorem solve_eqn (x : ℚ) (h1 : x ≠ 4) (h2 : x ≠ 6) :
  (x + 11) / (x - 4) = (x - 1) / (x + 6) → x = -31 / 11 :=
by
sorry

end solve_eqn_l93_93832


namespace k_polygonal_intersects_fermat_l93_93401

theorem k_polygonal_intersects_fermat (k : ℕ) (n m : ℕ) (h1: k > 2) 
  (h2 : ∃ n m, (k - 2) * n * (n - 1) / 2 + n = 2 ^ (2 ^ m) + 1) : 
  k = 3 ∨ k = 5 :=
  sorry

end k_polygonal_intersects_fermat_l93_93401


namespace calculate_f_value_l93_93014

def f (x y : ℚ) : ℚ := x - y * ⌈x / y⌉

theorem calculate_f_value :
  f (1/3) (-3/7) = -2/21 := by
  sorry

end calculate_f_value_l93_93014


namespace geometric_sequence_increasing_iff_l93_93939

variable {a : ℕ → ℝ} {q : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

theorem geometric_sequence_increasing_iff 
  (ha : is_geometric_sequence a q) 
  (h : a 0 < a 1 ∧ a 1 < a 2) : 
  is_increasing_sequence a ↔ (a 0 < a 1 ∧ a 1 < a 2) := 
sorry

end geometric_sequence_increasing_iff_l93_93939


namespace apple_ratio_simplest_form_l93_93698

theorem apple_ratio_simplest_form (sarah_apples brother_apples cousin_apples : ℕ) 
  (h1 : sarah_apples = 630)
  (h2 : brother_apples = 270)
  (h3 : cousin_apples = 540)
  (gcd_simplified : Nat.gcd (Nat.gcd sarah_apples brother_apples) cousin_apples = 90) :
  (sarah_apples / 90, brother_apples / 90, cousin_apples / 90) = (7, 3, 6) := 
by
  sorry

end apple_ratio_simplest_form_l93_93698


namespace g_18_equals_324_l93_93392

def is_strictly_increasing (g : ℕ → ℕ) :=
  ∀ n : ℕ, n > 0 → g (n + 1) > g n

def multiplicative (g : ℕ → ℕ) :=
  ∀ m n : ℕ, m > 0 → n > 0 → g (m * n) = g m * g n

def m_n_condition (g : ℕ → ℕ) :=
  ∀ m n : ℕ, m ≠ n → m > 0 → n > 0 → m ^ n = n ^ m → (g m = n ∨ g n = m)

noncomputable def g : ℕ → ℕ := sorry

theorem g_18_equals_324 :
  is_strictly_increasing g →
  multiplicative g →
  m_n_condition g →
  g 18 = 324 :=
sorry

end g_18_equals_324_l93_93392


namespace exists_x_gt_zero_negation_l93_93343

theorem exists_x_gt_zero_negation :
  (∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
  sorry  -- Proof goes here

end exists_x_gt_zero_negation_l93_93343


namespace arithmetic_sequence_sum_l93_93005

theorem arithmetic_sequence_sum (x y : ℕ)
  (h₁ : ∃ d, 9 = 3 + d)  -- Common difference exists, d = 6
  (h₂ : ∃ n, 15 = 3 + n * 6)  -- Arithmetic sequence term verification
  (h₃ : y = 33 - 6)
  (h₄ : x = 27 - 6) : x + y = 48 :=
sorry

end arithmetic_sequence_sum_l93_93005


namespace unique_triplets_l93_93770

theorem unique_triplets (a b c : ℝ) :
  (∀ x y z : ℝ, |a * x + b * y + c * z| + |b * x + c * y + a * z| + 
               |c * x + a * y + b * z| = |x| + |y| + |z|) ↔ 
  ((a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = 0 ∧ b = 0 ∧ c = -1)) :=
sorry

end unique_triplets_l93_93770


namespace max_n_for_factorization_l93_93415

theorem max_n_for_factorization (A B n : ℤ) (AB_cond : A * B = 48) (n_cond : n = 5 * B + A) :
  n ≤ 241 :=
by
  sorry

end max_n_for_factorization_l93_93415


namespace smallest_value_at_x_5_l93_93325

-- Define the variable x
def x : ℕ := 5

-- Define each expression
def exprA := 8 / x
def exprB := 8 / (x + 2)
def exprC := 8 / (x - 2)
def exprD := x / 8
def exprE := (x + 2) / 8

-- The goal is to prove that exprD yields the smallest value
theorem smallest_value_at_x_5 : exprD = min (min (min exprA exprB) (min exprC exprE)) :=
sorry

end smallest_value_at_x_5_l93_93325


namespace additional_time_to_empty_tank_l93_93815

-- Definitions based on conditions
def tankCapacity : ℕ := 3200  -- litres
def outletTimeAlone : ℕ := 5  -- hours
def inletRate : ℕ := 4  -- litres/min

-- Calculate rates
def outletRate : ℕ := tankCapacity / outletTimeAlone  -- litres/hour
def inletRatePerHour : ℕ := inletRate * 60  -- Convert litres/min to litres/hour

-- Calculate effective_rate when both pipes open
def effectiveRate : ℕ := outletRate - inletRatePerHour  -- litres/hour

-- Calculate times
def timeWithInletOpen : ℕ := tankCapacity / effectiveRate  -- hours
def additionalTime : ℕ := timeWithInletOpen - outletTimeAlone  -- hours

-- Proof statement
theorem additional_time_to_empty_tank : additionalTime = 3 := by
  -- It's clear from calculation above, we just add sorry for now to skip the proof
  sorry

end additional_time_to_empty_tank_l93_93815


namespace range_of_independent_variable_l93_93674

theorem range_of_independent_variable (x : ℝ) :
  (x + 2 >= 0) → (x - 1 ≠ 0) → (x ≥ -2 ∧ x ≠ 1) :=
by
  intros h₁ h₂
  sorry

end range_of_independent_variable_l93_93674


namespace breadth_of_rectangular_plot_l93_93839

theorem breadth_of_rectangular_plot
  (b l : ℕ)
  (h1 : l = 3 * b)
  (h2 : l * b = 2028) :
  b = 26 :=
sorry

end breadth_of_rectangular_plot_l93_93839


namespace runners_meet_time_l93_93582

theorem runners_meet_time :
  let time_runner_1 := 2
  let time_runner_2 := 4
  let time_runner_3 := 11 / 2
  Nat.lcm time_runner_1 (Nat.lcm time_runner_2 (Nat.lcm (11) 2)) = 44 := by
  sorry

end runners_meet_time_l93_93582


namespace minimal_erasure_l93_93390

noncomputable def min_factors_to_erase : ℕ :=
  2016

theorem minimal_erasure:
  ∀ (f g : ℝ → ℝ), 
    (∀ x, f x = g x) → 
    (∃ f' g' : ℝ → ℝ, (∀ x, f x ≠ g x) ∧ 
      ((∃ s : Finset ℕ, s.card = min_factors_to_erase ∧ (∀ i ∈ s, f' x = (x - i) * f x)) ∧ 
      (∃ t : Finset ℕ, t.card = min_factors_to_erase ∧ (∀ i ∈ t, g' x = (x - i) * g x)))) :=
by
  sorry

end minimal_erasure_l93_93390


namespace villager4_truth_teller_l93_93727

def villager1_statement (liars : Finset ℕ) : Prop := liars = {0, 1, 2, 3}
def villager2_statement (liars : Finset ℕ) : Prop := liars.card = 1
def villager3_statement (liars : Finset ℕ) : Prop := liars.card = 2
def villager4_statement (liars : Finset ℕ) : Prop := 3 ∉ liars

theorem villager4_truth_teller (liars : Finset ℕ) :
  ¬ villager1_statement liars ∧
  ¬ villager2_statement liars ∧
  ¬ villager3_statement liars ∧
  villager4_statement liars ↔
  liars = {0, 1, 2} :=
by
  sorry

end villager4_truth_teller_l93_93727


namespace xyz_neg_of_ineq_l93_93560

variables {x y z : ℝ}

theorem xyz_neg_of_ineq
  (h1 : 2 * x - y < 0)
  (h2 : 3 * y - 2 * z < 0)
  (h3 : 4 * z - 3 * x < 0) :
  x < 0 ∧ y < 0 ∧ z < 0 :=
sorry

end xyz_neg_of_ineq_l93_93560


namespace problem_f_f2_equals_16_l93_93287

noncomputable def f (x : ℝ) : ℝ :=
if h : x < 3 then x^2 else 2^x

theorem problem_f_f2_equals_16 : f (f 2) = 16 :=
by
  sorry

end problem_f_f2_equals_16_l93_93287


namespace f_at_7_l93_93487

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_property : ∀ x, f (x + 4) = f x
axiom specific_interval_definition : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem f_at_7 : f 7 = -2 := 
  by sorry

end f_at_7_l93_93487


namespace smallest_rel_prime_greater_than_one_l93_93016

theorem smallest_rel_prime_greater_than_one (n : ℕ) (h : n > 1) (h0: ∀ (m : ℕ), m > 1 ∧ Nat.gcd m 2100 = 1 → 11 ≤ m):
  Nat.gcd n 2100 = 1 → n = 11 :=
by
  -- Proof skipped
  sorry

end smallest_rel_prime_greater_than_one_l93_93016


namespace negation_of_conditional_l93_93155

-- Define the propositions
def P (x : ℝ) : Prop := x > 2015
def Q (x : ℝ) : Prop := x > 0

-- Negate the propositions
def notP (x : ℝ) : Prop := x <= 2015
def notQ (x : ℝ) : Prop := x <= 0

-- Theorem: Negation of the conditional statement
theorem negation_of_conditional (x : ℝ) : ¬ (P x → Q x) ↔ (notP x → notQ x) :=
by
  sorry

end negation_of_conditional_l93_93155


namespace find_biology_marks_l93_93904

variables (e m p c b : ℕ)
variable (a : ℝ)

def david_marks_in_biology : Prop :=
  e = 72 ∧
  m = 45 ∧
  p = 72 ∧
  c = 77 ∧
  a = 68.2 ∧
  (e + m + p + c + b) / 5 = a

theorem find_biology_marks (h : david_marks_in_biology e m p c b a) : b = 75 :=
sorry

end find_biology_marks_l93_93904


namespace chocolate_difference_l93_93037

theorem chocolate_difference :
  let nick_chocolates := 10
  let alix_chocolates := 3 * nick_chocolates - 5
  alix_chocolates - nick_chocolates = 15 :=
by
  sorry

end chocolate_difference_l93_93037


namespace min_max_calculation_l93_93988

theorem min_max_calculation
  (p q r s : ℝ)
  (h1 : p + q + r + s = 8)
  (h2 : p^2 + q^2 + r^2 + s^2 = 20) :
  -32 ≤ 5 * (p^3 + q^3 + r^3 + s^3) - (p^4 + q^4 + r^4 + s^4) ∧
  5 * (p^3 + q^3 + r^3 + s^3) - (p^4 + q^4 + r^4 + s^4) ≤ 12 :=
sorry

end min_max_calculation_l93_93988


namespace trains_at_starting_positions_after_2016_minutes_l93_93992

-- Definitions corresponding to conditions
def round_trip_minutes (line: String) : Nat :=
  if line = "red" then 14
  else if line = "blue" then 16
  else if line = "green" then 18
  else 0

def is_multiple_of (n m : Nat) : Prop :=
  n % m = 0

-- Formalize the statement to be proven
theorem trains_at_starting_positions_after_2016_minutes :
  ∀ (line: String), 
  line = "red" ∨ line = "blue" ∨ line = "green" →
  is_multiple_of 2016 (round_trip_minutes line) :=
by
  intro line h
  cases h with
  | inl red =>
    sorry
  | inr hb =>
    cases hb with
    | inl blue =>
      sorry
    | inr green =>
      sorry

end trains_at_starting_positions_after_2016_minutes_l93_93992


namespace children_distribution_l93_93922

theorem children_distribution (a b c d N : ℕ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d)
  (h4 : a + b + c + d < 18) 
  (h5 : a * b * c * d = N) : 
  N = 120 ∧ a = 5 ∧ b = 4 ∧ c = 3 ∧ d = 2 := 
by 
  sorry

end children_distribution_l93_93922


namespace polynomial_identity_l93_93990

theorem polynomial_identity (x : ℝ) :
  (x - 2)^5 + 5 * (x - 2)^4 + 10 * (x - 2)^3 + 10 * (x - 2)^2 + 5 * (x - 2) + 1 = (x - 1)^5 := 
by 
  sorry

end polynomial_identity_l93_93990


namespace no_integer_x_square_l93_93249

theorem no_integer_x_square (x : ℤ) : 
  ∀ n : ℤ, x^5 + 5 * x^4 + 10 * x^3 + 10 * x^2 + 5 * x + 1 ≠ n^2 :=
by sorry

end no_integer_x_square_l93_93249


namespace jasmine_max_stickers_l93_93676

-- Given conditions and data
def sticker_cost : ℝ := 0.75
def jasmine_budget : ℝ := 10.0

-- Proof statement
theorem jasmine_max_stickers : ∃ n : ℕ, (n : ℝ) * sticker_cost ≤ jasmine_budget ∧ (∀ m : ℕ, (m > n) → (m : ℝ) * sticker_cost > jasmine_budget) :=
sorry

end jasmine_max_stickers_l93_93676


namespace range_of_a_l93_93845

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - a * x

theorem range_of_a (a : ℝ) :
  (∀ (x1 x2 : ℝ), 0 < x1 ∧ 0 < x2 ∧ x2 > x1 → (f x1 a / x2 - f x2 a / x1 < 0)) ↔ a ≤ Real.exp 1 / 2 := sorry

end range_of_a_l93_93845


namespace price_increase_for_1620_profit_maximizing_profit_l93_93300

-- To state the problem, we need to define some variables and the associated conditions.

def cost_price : ℝ := 13
def initial_selling_price : ℝ := 20
def initial_monthly_sales : ℝ := 200
def decrease_in_sales_per_yuan : ℝ := 10
def profit_condition (x : ℝ) : ℝ := (initial_selling_price + x - cost_price) * (initial_monthly_sales - decrease_in_sales_per_yuan * x)
def profit_function (x : ℝ) : ℝ := -(10 * x ^ 2) + (130 * x) + 140

-- Part (1): Prove the price increase x such that the profit is 1620 yuan
theorem price_increase_for_1620_profit :
  ∃ (x : ℝ), profit_condition x = 1620 ∧ (x = 2 ∨ x = 11) :=
sorry

-- Part (2): Prove that the selling price that maximizes profit is 26.5 yuan and max profit is 1822.5 yuan
theorem maximizing_profit :
  ∃ (x : ℝ), (x = 13 / 2) ∧ profit_function (13 / 2) = 3645 / 2 :=
sorry

end price_increase_for_1620_profit_maximizing_profit_l93_93300


namespace measure_of_angle_C_l93_93831

theorem measure_of_angle_C (A B : ℝ) (h1 : A + B = 180) (h2 : A = 5 * B) : A = 150 := by
  sorry

end measure_of_angle_C_l93_93831


namespace reciprocal_of_neg_2023_l93_93203

theorem reciprocal_of_neg_2023 : (1 : ℝ) / (-2023) = -(1 / 2023) :=
by 
  sorry

end reciprocal_of_neg_2023_l93_93203


namespace math_problem_l93_93065

open Function

noncomputable def rotate_90_ccw (p : ℝ × ℝ) (c : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  let (h, k) := c
  (h - (y - k), k + (x - h))

noncomputable def reflect_over_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (y, x)

theorem math_problem (a b : ℝ) :
  reflect_over_y_eq_x (rotate_90_ccw (a, b) (2, 3)) = (4, -5) → b - a = -5 :=
by
  intros h
  sorry

end math_problem_l93_93065


namespace no_perfect_square_solution_l93_93107

theorem no_perfect_square_solution (n : ℕ) (x : ℕ) (hx : x < 10^n) :
  ¬ (∀ y, 0 ≤ y ∧ y ≤ 9 → ∃ z : ℤ, ∃ k : ℤ, 10^(n+1) * z + 10 * x + y = k^2) :=
sorry

end no_perfect_square_solution_l93_93107


namespace correct_calculation_result_l93_93766

theorem correct_calculation_result :
  (∃ x : ℤ, 14 * x = 70) → (5 - 6 = -1) :=
by
  sorry

end correct_calculation_result_l93_93766


namespace average_children_l93_93081

theorem average_children (total_families : ℕ) (avg_children_all : ℕ) 
  (childless_families : ℕ) (total_children : ℕ) (families_with_children : ℕ) : 
  total_families = 15 →
  avg_children_all = 3 →
  childless_families = 3 →
  total_children = total_families * avg_children_all →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℚ) = 3.8 :=
by
  intros
  sorry

end average_children_l93_93081


namespace solve_inequalities_l93_93313

theorem solve_inequalities :
  (∀ x : ℝ, x^2 + 3 * x - 10 ≥ 0 ↔ (x ≤ -5 ∨ x ≥ 2)) ∧
  (∀ x : ℝ, x^2 - 3 * x - 2 ≤ 0 ↔ (3 - Real.sqrt 17) / 2 ≤ x ∧ x ≤ (3 + Real.sqrt 17) / 2) :=
by
  sorry

end solve_inequalities_l93_93313


namespace boxes_of_apples_l93_93470

theorem boxes_of_apples (apples_per_crate crates_delivered rotten_apples apples_per_box : ℕ) 
       (h1 : apples_per_crate = 42) 
       (h2 : crates_delivered = 12) 
       (h3 : rotten_apples = 4) 
       (h4 : apples_per_box = 10) : 
       crates_delivered * apples_per_crate - rotten_apples = 500 ∧
       (crates_delivered * apples_per_crate - rotten_apples) / apples_per_box = 50 := by
  sorry

end boxes_of_apples_l93_93470


namespace Cathy_and_Chris_worked_months_l93_93043

theorem Cathy_and_Chris_worked_months (Cathy_hours : ℕ) (weekly_hours : ℕ) (weeks_in_month : ℕ) (extra_weekly_hours : ℕ) (weeks_for_Chris_sick : ℕ) : 
  Cathy_hours = 180 →
  weekly_hours = 20 →
  weeks_in_month = 4 →
  extra_weekly_hours = weekly_hours →
  weeks_for_Chris_sick = 1 →
  (Cathy_hours - extra_weekly_hours * weeks_for_Chris_sick) / weekly_hours / weeks_in_month = (2 : ℕ) :=
by
  intros hCathy_hours hweekly_hours hweeks_in_month hextra_weekly_hours hweeks_for_Chris_sick
  rw [hCathy_hours, hweekly_hours, hweeks_in_month, hextra_weekly_hours, hweeks_for_Chris_sick]
  norm_num
  sorry

end Cathy_and_Chris_worked_months_l93_93043


namespace chess_group_players_l93_93315

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 91) : n = 14 :=
sorry

end chess_group_players_l93_93315


namespace chandler_bike_purchase_l93_93785

theorem chandler_bike_purchase : 
    ∀ (x : ℕ), (200 + 20 * x = 800) → (x = 30) :=
by
  intros x h
  sorry

end chandler_bike_purchase_l93_93785


namespace largest_possible_n_l93_93346

theorem largest_possible_n :
  ∃ (m n : ℕ), (0 < m) ∧ (0 < n) ∧ (m + n = 10) ∧ (n = 9) :=
by
  sorry

end largest_possible_n_l93_93346


namespace trapezoidal_field_base_count_l93_93243

theorem trapezoidal_field_base_count
  (A : ℕ) (h : ℕ) (b1 b2 : ℕ)
  (hdiv8 : ∃ m n : ℕ, b1 = 8 * m ∧ b2 = 8 * n)
  (area_eq : A = (h * (b1 + b2)) / 2)
  (A_val : A = 1400)
  (h_val : h = 50) :
  (∃ pair1 pair2 pair3, (pair1 + pair2 + pair3 = (b1 + b2))) :=
by
  sorry

end trapezoidal_field_base_count_l93_93243


namespace teal_bakery_pumpkin_pie_l93_93154

theorem teal_bakery_pumpkin_pie (P : ℕ) 
    (pumpkin_price_per_slice : ℕ := 5)
    (custard_price_per_slice : ℕ := 6)
    (pumpkin_pies_sold : ℕ := 4)
    (custard_pies_sold : ℕ := 5)
    (custard_pieces_per_pie : ℕ := 6)
    (total_revenue : ℕ := 340) :
    4 * P * pumpkin_price_per_slice + custard_pies_sold * custard_pieces_per_pie * custard_price_per_slice = total_revenue → P = 8 := 
by
  sorry

end teal_bakery_pumpkin_pie_l93_93154


namespace complete_the_square_l93_93728

theorem complete_the_square (x : ℝ) : (x^2 + 2 * x - 1 = 0) -> ((x + 1)^2 = 2) :=
by
  intro h
  sorry

end complete_the_square_l93_93728


namespace average_eq_instantaneous_velocity_at_t_eq_3_l93_93174

theorem average_eq_instantaneous_velocity_at_t_eq_3
  (S : ℝ → ℝ) (hS : ∀ t, S t = 24 * t - 3 * t^2) :
  (1 / 6) * (S 6 - S 0) = 24 - 6 * 3 :=
by 
  sorry

end average_eq_instantaneous_velocity_at_t_eq_3_l93_93174


namespace total_money_received_l93_93756

-- Define the conditions
def total_puppies : ℕ := 20
def fraction_sold : ℚ := 3 / 4
def price_per_puppy : ℕ := 200

-- Define the statement to prove
theorem total_money_received : fraction_sold * total_puppies * price_per_puppy = 3000 := by
  sorry

end total_money_received_l93_93756


namespace number_of_paths_l93_93151

theorem number_of_paths (n : ℕ) (h1 : n > 3) : 
  (2 * (8 * n^3 - 48 * n^2 + 88 * n - 48) + (4 * n^2 - 12 * n + 8) + (2 * n - 2)) = 16 * n^3 - 92 * n^2 + 166 * n - 90 :=
by
  sorry

end number_of_paths_l93_93151


namespace spinner_prob_C_l93_93127

theorem spinner_prob_C (P_A P_B P_C : ℚ) (h_A : P_A = 1/3) (h_B : P_B = 5/12) (h_total : P_A + P_B + P_C = 1) : 
  P_C = 1/4 := 
sorry

end spinner_prob_C_l93_93127


namespace largest_x_satisfies_condition_l93_93603

theorem largest_x_satisfies_condition :
  ∃ x : ℝ, (⌊x⌋ / x = 7 / 8) ∧ (∀ y : ℝ, (⌊y⌋ / y = 7 / 8) → y ≤ 48 / 7) :=
sorry

end largest_x_satisfies_condition_l93_93603


namespace second_supply_cost_is_24_l93_93384

-- Definitions based on the given problem conditions
def cost_first_supply : ℕ := 13
def last_year_remaining : ℕ := 6
def this_year_budget : ℕ := 50
def remaining_budget : ℕ := 19

-- Sum of last year's remaining budget and this year's budget
def total_budget : ℕ := last_year_remaining + this_year_budget

-- Total amount spent on school supplies
def total_spent : ℕ := total_budget - remaining_budget

-- Cost of second school supply
def cost_second_supply : ℕ := total_spent - cost_first_supply

-- The theorem to prove
theorem second_supply_cost_is_24 : cost_second_supply = 24 := by
  sorry

end second_supply_cost_is_24_l93_93384


namespace ratio_of_supply_to_demand_l93_93668

theorem ratio_of_supply_to_demand (supply demand : ℕ)
  (hs : supply = 1800000)
  (hd : demand = 2400000) :
  supply / (Nat.gcd supply demand) = 3 ∧ demand / (Nat.gcd supply demand) = 4 :=
by
  sorry

end ratio_of_supply_to_demand_l93_93668


namespace focus_of_parabola_l93_93651

theorem focus_of_parabola (x y : ℝ) : 
  (∃ x y : ℝ, x^2 = -2 * y) → (0, -1/2) = (0, -1/2) :=
sorry

end focus_of_parabola_l93_93651


namespace empty_vessel_percentage_l93_93363

theorem empty_vessel_percentage
  (P : ℝ) -- weight of the paint that completely fills the vessel
  (E : ℝ) -- weight of the empty vessel
  (h1 : 0.5 * (E + P) = E + 0.42857142857142855 * P)
  (h2 : 0.07142857142857145 * P = 0.5 * E):
  (E / (E + P) * 100) = 12.5 :=
by
  sorry

end empty_vessel_percentage_l93_93363


namespace profit_percentage_l93_93663

theorem profit_percentage (SP CP : ℝ) (h₁ : SP = 300) (h₂ : CP = 250) : ((SP - CP) / CP) * 100 = 20 := by
  sorry

end profit_percentage_l93_93663


namespace suff_not_nec_l93_93706

variables (a b : ℝ)
def P := (a = 1) ∧ (b = 1)
def Q := (a + b = 2)

theorem suff_not_nec : P a b → Q a b ∧ ¬ (Q a b → P a b) :=
by
  sorry

end suff_not_nec_l93_93706


namespace max_value_sqrt_add_l93_93345

noncomputable def sqrt_add (a b : ℝ) : ℝ := Real.sqrt (a + 1) + Real.sqrt (b + 3)

theorem max_value_sqrt_add (a b : ℝ) (h : 0 < a) (h' : 0 < b) (hab : a + b = 5) :
  sqrt_add a b ≤ 3 * Real.sqrt 2 :=
by
  sorry

end max_value_sqrt_add_l93_93345


namespace find_n_l93_93868

theorem find_n : ∃ n : ℕ, n < 2006 ∧ ∀ m : ℕ, 2006 * n = m * (2006 + n) ↔ n = 1475 := by
  sorry

end find_n_l93_93868


namespace infinite_natural_solutions_l93_93032

theorem infinite_natural_solutions : ∀ n : ℕ, ∃ x y z : ℕ, (x + y + z)^2 + 2 * (x + y + z) = 5 * (x * y + y * z + z * x) :=
by
  sorry

end infinite_natural_solutions_l93_93032


namespace trig_identity_cos2theta_tan_minus_pi_over_4_l93_93357

variable (θ : ℝ)

-- Given condition
def tan_theta_is_2 : Prop := Real.tan θ = 2

-- Proof problem 1: Prove that cos(2θ) = -3/5
def cos2theta (θ : ℝ) (h : tan_theta_is_2 θ) : Prop :=
  Real.cos (2 * θ) = -3 / 5

-- Proof problem 2: Prove that tan(θ - π/4) = 1/3
def tan_theta_minus_pi_over_4 (θ : ℝ) (h : tan_theta_is_2 θ) : Prop :=
  Real.tan (θ - Real.pi / 4) = 1 / 3

-- Main theorem statement
theorem trig_identity_cos2theta_tan_minus_pi_over_4 
  (θ : ℝ) (h : tan_theta_is_2 θ) :
  cos2theta θ h ∧ tan_theta_minus_pi_over_4 θ h :=
sorry

end trig_identity_cos2theta_tan_minus_pi_over_4_l93_93357


namespace hypotenuse_length_triangle_l93_93024

theorem hypotenuse_length_triangle (a b c : ℝ) (h1 : a + b + c = 40) (h2 : (1/2) * a * b = 30) 
  (h3 : a = b) : c = 2 * Real.sqrt 30 :=
by
  sorry

end hypotenuse_length_triangle_l93_93024


namespace complex_division_product_l93_93129

theorem complex_division_product
  (i : ℂ)
  (h_exp: i * i = -1)
  (a b : ℝ)
  (h_div: (1 + 7 * i) / (2 - i) = a + b * i)
  : a * b = -3 := 
sorry

end complex_division_product_l93_93129


namespace gymnastics_average_people_per_team_l93_93717

def average_people_per_team (boys girls teams : ℕ) : ℕ :=
  (boys + girls) / teams

theorem gymnastics_average_people_per_team:
  average_people_per_team 83 77 4 = 40 :=
by
  sorry

end gymnastics_average_people_per_team_l93_93717


namespace direct_proportion_function_l93_93017

theorem direct_proportion_function (m : ℝ) : 
  (m^2 + 2 * m ≠ 0) ∧ (m^2 - 3 = 1) → m = 2 :=
by {
  sorry
}

end direct_proportion_function_l93_93017


namespace no_zero_sum_of_vectors_l93_93681

-- Definitions and conditions for the problem
variable {n : ℕ} (odd_n : n % 2 = 1) -- n is odd, representing the number of sides of the polygon

-- The statement of the proof problem
theorem no_zero_sum_of_vectors (odd_n : n % 2 = 1) : false :=
by
  sorry

end no_zero_sum_of_vectors_l93_93681


namespace pieces_given_l93_93252

def pieces_initially := 38
def pieces_now := 54

theorem pieces_given : pieces_now - pieces_initially = 16 := by
  sorry

end pieces_given_l93_93252


namespace ninth_term_l93_93336

variable (a d : ℤ)
variable (h1 : a + 2 * d = 20)
variable (h2 : a + 5 * d = 26)

theorem ninth_term (a d : ℤ) (h1 : a + 2 * d = 20) (h2 : a + 5 * d = 26) : a + 8 * d = 32 :=
sorry

end ninth_term_l93_93336


namespace regular_price_of_shirt_is_50_l93_93837

-- Define all relevant conditions and given prices.
variables (P : ℝ) (shirt_price_discounted : ℝ) (total_paid : ℝ) (number_of_shirts : ℝ)

-- Define the conditions as hypotheses
def conditions :=
  (shirt_price_discounted = 0.80 * P) ∧
  (total_paid = 240) ∧
  (number_of_shirts = 6) ∧
  (total_paid = number_of_shirts * shirt_price_discounted)

-- State the theorem to prove that the regular price of the shirt is $50.
theorem regular_price_of_shirt_is_50 (h : conditions P shirt_price_discounted total_paid number_of_shirts) :
  P = 50 := 
sorry

end regular_price_of_shirt_is_50_l93_93837


namespace geometric_sequence_first_term_l93_93775

theorem geometric_sequence_first_term (a r : ℝ)
    (h1 : a * r^2 = 3)
    (h2 : a * r^4 = 27) :
    a = 1 / 3 := by
    sorry

end geometric_sequence_first_term_l93_93775


namespace quoted_value_of_stock_l93_93375

theorem quoted_value_of_stock (D Y Q : ℝ) (h1 : D = 8) (h2 : Y = 10) (h3 : Y = (D / Q) * 100) : Q = 80 :=
by 
  -- Insert proof here
  sorry

end quoted_value_of_stock_l93_93375


namespace average_age_of_coaches_l93_93828

theorem average_age_of_coaches 
  (total_members : ℕ) (average_age_members : ℕ)
  (num_girls : ℕ) (average_age_girls : ℕ)
  (num_boys : ℕ) (average_age_boys : ℕ)
  (num_coaches : ℕ) :
  total_members = 30 →
  average_age_members = 20 →
  num_girls = 10 →
  average_age_girls = 18 →
  num_boys = 15 →
  average_age_boys = 19 →
  num_coaches = 5 →
  (600 - (num_girls * average_age_girls) - (num_boys * average_age_boys)) / num_coaches = 27 :=
by
  intros
  sorry

end average_age_of_coaches_l93_93828


namespace comparison_of_abc_l93_93648

noncomputable def a : ℝ := 24 / 7
noncomputable def b : ℝ := Real.log 7
noncomputable def c : ℝ := Real.log (7 / Real.exp 1) / Real.log 3 + 1

theorem comparison_of_abc :
  (a = 24 / 7) →
  (b * Real.exp b = 7 * Real.log 7) →
  (3 ^ (c - 1) = 7 / Real.exp 1) →
  a > b ∧ b > c :=
by
  intros ha hb hc
  sorry

end comparison_of_abc_l93_93648


namespace exists_duplicate_in_grid_of_differences_bounded_l93_93056

theorem exists_duplicate_in_grid_of_differences_bounded :
  ∀ (f : ℕ × ℕ → ℤ), 
  (∀ i j, i < 10 → j < 10 → (i + 1 < 10 → (abs (f (i, j) - f (i + 1, j)) ≤ 5)) 
                             ∧ (j + 1 < 10 → (abs (f (i, j) - f (i, j + 1)) ≤ 5))) → 
  ∃ x y : ℕ × ℕ, x ≠ y ∧ f x = f y :=
by
  intros
  sorry -- Proof goes here

end exists_duplicate_in_grid_of_differences_bounded_l93_93056


namespace probability_of_collinear_dots_l93_93621

theorem probability_of_collinear_dots (dots : ℕ) (rows : ℕ) (columns : ℕ) (choose : ℕ → ℕ → ℕ) :
  dots = 20 ∧ rows = 5 ∧ columns = 4 ∧ choose 20 4 = 4845 → 
  (∃ sets_of_collinear_dots : ℕ, sets_of_collinear_dots = 20 ∧ 
   ∃ probability : ℚ,  probability = 4 / 969) :=
by
  sorry

end probability_of_collinear_dots_l93_93621


namespace correct_statement_l93_93289

open Set

variable (U : Set ℕ) (M : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5}) (hM : U \ M = {1, 3})

theorem correct_statement : 2 ∈ M :=
by
  sorry

end correct_statement_l93_93289


namespace meaningful_sqrt_condition_l93_93830

theorem meaningful_sqrt_condition (x : ℝ) : (2 * x - 1 ≥ 0) ↔ (x ≥ 1 / 2) :=
by
  sorry

end meaningful_sqrt_condition_l93_93830


namespace animal_shelter_l93_93408

theorem animal_shelter : ∃ D C : ℕ, (D = 75) ∧ (D / C = 15 / 7) ∧ (D / (C + 20) = 15 / 11) :=
by
  sorry

end animal_shelter_l93_93408


namespace locus_of_centers_l93_93606

theorem locus_of_centers (a b : ℝ) :
  (∃ r : ℝ, a^2 + b^2 = (r + 2)^2 ∧ (a - 3)^2 + b^2 = (9 - r)^2) →
  12 * a^2 + 169 * b^2 - 36 * a - 1584 = 0 :=
by
  sorry

end locus_of_centers_l93_93606


namespace symmetric_points_ab_value_l93_93270

theorem symmetric_points_ab_value
  (a b : ℤ)
  (h₁ : a + 2 = -4)
  (h₂ : 2 = b) :
  a * b = -12 :=
by
  sorry

end symmetric_points_ab_value_l93_93270


namespace ratio_of_group_average_l93_93410

theorem ratio_of_group_average
  (d l e : ℕ)
  (avg_group_age : ℕ := 45) 
  (avg_doctors_age : ℕ := 40) 
  (avg_lawyers_age : ℕ := 55) 
  (avg_engineers_age : ℕ := 35)
  (h : (40 * d + 55 * l + 35 * e) / (d + l + e) = avg_group_age)
  : d = 2 * l - e ∧ l = 2 * e :=
sorry

end ratio_of_group_average_l93_93410


namespace find_abc_l93_93261

theorem find_abc (a b c : ℚ) 
  (h1 : a + b + c = 24)
  (h2 : a + 2 * b = 2 * c)
  (h3 : a = b / 2) : 
  a = 16 / 3 ∧ b = 32 / 3 ∧ c = 8 := 
by 
  sorry

end find_abc_l93_93261


namespace chord_length_y_eq_x_plus_one_meets_circle_l93_93309

noncomputable def chord_length (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

theorem chord_length_y_eq_x_plus_one_meets_circle 
  (A B : ℝ × ℝ) 
  (hA : A.2 = A.1 + 1) 
  (hB : B.2 = B.1 + 1) 
  (hA_on_circle : A.1^2 + A.2^2 + 2 * A.2 - 3 = 0)
  (hB_on_circle : B.1^2 + B.2^2 + 2 * B.2 - 3 = 0) :
  chord_length A B = 2 * Real.sqrt 2 := 
sorry

end chord_length_y_eq_x_plus_one_meets_circle_l93_93309


namespace value_of_expression_when_x_is_neg2_l93_93726

theorem value_of_expression_when_x_is_neg2 : 
  ∀ (x : ℤ), x = -2 → (3 * x + 4) ^ 2 = 4 :=
by
  sorry

end value_of_expression_when_x_is_neg2_l93_93726


namespace abs_neg_five_l93_93669

theorem abs_neg_five : abs (-5) = 5 :=
by
  sorry

end abs_neg_five_l93_93669


namespace find_number_subtracted_l93_93565

theorem find_number_subtracted (x : ℕ) (h : 88 - x = 54) : x = 34 := by
  sorry

end find_number_subtracted_l93_93565


namespace bouquets_needed_to_earn_1000_l93_93286

theorem bouquets_needed_to_earn_1000 :
  ∀ (cost_per_bouquet sell_price_bouquet: ℕ) (roses_per_bouquet_bought roses_per_bouquet_sold target_profit: ℕ),
    cost_per_bouquet = 20 →
    sell_price_bouquet = 20 →
    roses_per_bouquet_bought = 7 →
    roses_per_bouquet_sold = 5 →
    target_profit = 1000 →
    (target_profit / (sell_price_bouquet * roses_per_bouquet_sold / roses_per_bouquet_bought - cost_per_bouquet) * roses_per_bouquet_bought = 125) :=
by
  intros cost_per_bouquet sell_price_bouquet roses_per_bouquet_bought roses_per_bouquet_sold target_profit 
    h_cost_per_bouquet h_sell_price_bouquet h_roses_per_bouquet_bought h_roses_per_bouquet_sold h_target_profit
  sorry

end bouquets_needed_to_earn_1000_l93_93286


namespace wicket_keeper_age_difference_l93_93711

def cricket_team_average_age : Nat := 24
def total_members : Nat := 11
def remaining_members : Nat := 9
def age_difference : Nat := 1

theorem wicket_keeper_age_difference :
  let total_age := cricket_team_average_age * total_members
  let remaining_average_age := cricket_team_average_age - age_difference
  let remaining_total_age := remaining_average_age * remaining_members
  let combined_age := total_age - remaining_total_age
  let average_age := cricket_team_average_age
  let wicket_keeper_age := combined_age - average_age
  wicket_keeper_age - average_age = 9 := 
by
  sorry

end wicket_keeper_age_difference_l93_93711


namespace min_value_PF_PA_l93_93169

open Classical

noncomputable section

def parabola_eq (x y : ℝ) : Prop := y^2 = 16 * x

def point_A : ℝ × ℝ := (1, 2)

def focus_F : ℝ × ℝ := (4, 0)  -- Focus of the given parabola y^2 = 16x

def distance (P1 P2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

def PF_PA (P : ℝ × ℝ) : ℝ :=
  distance P focus_F + distance P point_A

theorem min_value_PF_PA :
  ∃ P : ℝ × ℝ, parabola_eq P.1 P.2 ∧ PF_PA P = 5 :=
sorry

end min_value_PF_PA_l93_93169


namespace no_infinite_subdivision_exists_l93_93742

theorem no_infinite_subdivision_exists : ¬ ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  (∀ n : ℕ,
    ∃ (ai bi : ℝ), ai > bi ∧ bi > 0 ∧ ai * bi = a * b ∧
    (ai / bi = a / b ∨ bi / ai = a / b)) :=
sorry

end no_infinite_subdivision_exists_l93_93742


namespace hannah_late_times_l93_93952

variable (hourly_rate : ℝ)
variable (hours_worked : ℝ)
variable (dock_per_late : ℝ)
variable (actual_pay : ℝ)

theorem hannah_late_times (h1 : hourly_rate = 30)
                          (h2 : hours_worked = 18)
                          (h3 : dock_per_late = 5)
                          (h4 : actual_pay = 525) :
  ((hourly_rate * hours_worked - actual_pay) / dock_per_late) = 3 := 
by
  sorry

end hannah_late_times_l93_93952


namespace infinitely_many_squares_of_form_l93_93919

theorem infinitely_many_squares_of_form (k : ℕ) (hk : 0 < k) : 
  ∃ (n : ℕ), ∀ m : ℕ, ∃ n' > n, 2 * k * n' - 7 = m^2 :=
sorry

end infinitely_many_squares_of_form_l93_93919


namespace modulus_product_eq_sqrt_5_l93_93578

open Complex

-- Define the given complex number.
def z : ℂ := 2 + I

-- Declare the product with I.
def z_product := z * I

-- State the theorem that the modulus of the product is sqrt(5).
theorem modulus_product_eq_sqrt_5 : abs z_product = Real.sqrt 5 := 
sorry

end modulus_product_eq_sqrt_5_l93_93578


namespace distance_relation_possible_l93_93931

-- Define a structure representing points in 2D space
structure Point where
  x : ℤ
  y : ℤ

-- Define the artificial geometry distance function (Euclidean distance)
def varrho (p1 p2 : Point) : ℝ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2).sqrt

-- Define the non-collinearity condition for points A, B, and C
def non_collinear (A B C : Point) : Prop :=
  ¬(A.x = B.x ∧ B.x = C.x) ∧ ¬(A.y = B.y ∧ B.y = C.y)

theorem distance_relation_possible :
  ∃ (A B C : Point), non_collinear A B C ∧ varrho A C ^ 2 + varrho B C ^ 2 = varrho A B ^ 2 :=
by
  sorry

end distance_relation_possible_l93_93931


namespace min_pencils_to_ensure_18_l93_93277

theorem min_pencils_to_ensure_18 :
  ∀ (total red green yellow blue brown black : ℕ),
  total = 120 → red = 35 → green = 23 → yellow = 14 → blue = 26 → brown = 11 → black = 11 →
  ∃ (n : ℕ), n = 88 ∧
  (∀ (picked_pencils : ℕ → ℕ), (
    (picked_pencils 0 + picked_pencils 1 + picked_pencils 2 + picked_pencils 3 + picked_pencils 4 + picked_pencils 5 = n) →
    (picked_pencils 0 ≤ red) → (picked_pencils 1 ≤ green) → (picked_pencils 2 ≤ yellow) →
    (picked_pencils 3 ≤ blue) → (picked_pencils 4 ≤ brown) → (picked_pencils 5 ≤ black) →
    (picked_pencils 0 ≥ 18 ∨ picked_pencils 1 ≥ 18 ∨ picked_pencils 2 ≥ 18 ∨ picked_pencils 3 ≥ 18 ∨ picked_pencils 4 ≥ 18 ∨ picked_pencils 5 ≥ 18)
  )) := 
sorry

end min_pencils_to_ensure_18_l93_93277


namespace calculate_expression_l93_93045

theorem calculate_expression : |1 - Real.sqrt 3| - (Real.sqrt 3 - 1)^0 = Real.sqrt 3 - 2 := by
  sorry

end calculate_expression_l93_93045


namespace same_terminal_side_angles_l93_93094

theorem same_terminal_side_angles (k : ℤ) :
  ∃ (k1 k2 : ℤ), k1 * 360 - 1560 = -120 ∧ k2 * 360 - 1560 = 240 :=
by
  -- Conditions and property definitions can be added here if needed
  sorry

end same_terminal_side_angles_l93_93094


namespace set_complement_intersection_l93_93461

theorem set_complement_intersection
  (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)
  (hU : U = {0, 1, 2, 3, 4})
  (hM : M = {0, 1, 2})
  (hN : N = {2, 3}) :
  ((U \ M) ∩ N) = {3} :=
  by sorry

end set_complement_intersection_l93_93461


namespace find_triplets_l93_93367

theorem find_triplets (m n k : ℕ) (pos_m : 0 < m) (pos_n : 0 < n) (pos_k : 0 < k) : 
  (k^m ∣ m^n - 1) ∧ (k^n ∣ n^m - 1) ↔ (k = 1) ∨ (m = 1 ∧ n = 1) :=
by
  sorry

end find_triplets_l93_93367


namespace janet_clarinet_hours_l93_93008

theorem janet_clarinet_hours 
  (C : ℕ)  -- number of clarinet lessons hours per week
  (clarinet_cost_per_hour : ℕ := 40)
  (piano_cost_per_hour : ℕ := 28)
  (hours_of_piano_per_week : ℕ := 5)
  (annual_extra_piano_cost : ℕ := 1040) :
  52 * (piano_cost_per_hour * hours_of_piano_per_week - clarinet_cost_per_hour * C) = annual_extra_piano_cost → 
  C = 3 :=
by
  sorry

end janet_clarinet_hours_l93_93008


namespace greatest_four_digit_multiple_of_17_l93_93550

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_multiple_of (n d : ℕ) : Prop :=
  ∃ k : ℕ, n = k * d

theorem greatest_four_digit_multiple_of_17 : ∃ n, is_four_digit n ∧ is_multiple_of n 17 ∧
  ∀ m, is_four_digit m → is_multiple_of m 17 → m ≤ n :=
  by
  existsi 9996
  sorry

end greatest_four_digit_multiple_of_17_l93_93550


namespace race_head_start_l93_93915

/-- A's speed is 22/19 times that of B. If A and B run a race, A should give B a head start of (3 / 22) of the race length so the race ends in a dead heat. -/
theorem race_head_start {Va Vb L H : ℝ} (hVa : Va = (22 / 19) * Vb) (hL_Va : L / Va = (L - H) / Vb) : 
  H = (3 / 22) * L :=
by
  sorry

end race_head_start_l93_93915


namespace middle_part_proportional_l93_93862

theorem middle_part_proportional (x : ℚ) (s : ℚ) (h : s = 120) 
    (proportional : (2 * x) + (1/2 * x) + (1/4 * x) = s) : 
    (1/2 * x) = 240/11 := 
by
  sorry

end middle_part_proportional_l93_93862


namespace break_even_production_volume_l93_93109

theorem break_even_production_volume
  (Q : ℕ) 
  (ATC : ℕ → ℚ)
  (P : ℚ)
  (h1 : ∀ Q, ATC Q = 100 + 100000 / Q)
  (h2 : P = 300) :
  ATC 500 = P :=
by
  sorry

end break_even_production_volume_l93_93109


namespace tooth_extraction_cost_l93_93510

variable (c f b e : ℕ)

-- Conditions
def cost_cleaning := c = 70
def cost_filling := f = 120
def bill := b = 5 * f

-- Proof Problem
theorem tooth_extraction_cost (h_cleaning : cost_cleaning c) (h_filling : cost_filling f) (h_bill : bill b f) :
  e = b - (c + 2 * f) :=
sorry

end tooth_extraction_cost_l93_93510


namespace jacqueline_candy_multiple_l93_93475

theorem jacqueline_candy_multiple :
  let fred_candy := 12
  let uncle_bob_candy := fred_candy + 6
  let total_candy := fred_candy + uncle_bob_candy
  let jackie_candy := 120 / 0.40
  (jackie_candy / total_candy = 10) :=
by
  let fred_candy := 12
  let uncle_bob_candy := fred_candy + 6
  let total_candy := fred_candy + uncle_bob_candy
  let jackie_candy := 120 / 0.40
  show _ = _
  sorry

end jacqueline_candy_multiple_l93_93475


namespace find_unknown_number_l93_93033

theorem find_unknown_number : 
  ∃ x : ℚ, (x * 7) / (10 * 17) = 10000 ∧ x = 1700000 / 7 :=
by
  sorry

end find_unknown_number_l93_93033


namespace profit_percentage_B_l93_93421

theorem profit_percentage_B (cost_price_A : ℝ) (sell_price_C : ℝ) 
  (profit_A_percent : ℝ) (profit_B_percent : ℝ) 
  (cost_price_A_eq : cost_price_A = 148) 
  (sell_price_C_eq : sell_price_C = 222) 
  (profit_A_percent_eq : profit_A_percent = 0.2) :
  profit_B_percent = 0.25 := 
by
  have cost_price_B := cost_price_A * (1 + profit_A_percent)
  have profit_B := sell_price_C - cost_price_B
  have profit_B_percent := (profit_B / cost_price_B) * 100 
  sorry

end profit_percentage_B_l93_93421


namespace factorization_identity_l93_93551

variable (a b : ℝ)

theorem factorization_identity : 3 * a^2 + 6 * a * b = 3 * a * (a + 2 * b) := by
  sorry

end factorization_identity_l93_93551


namespace positive_expression_with_b_l93_93423

-- Defining the conditions and final statement
open Real

theorem positive_expression_with_b (a : ℝ) : (a + 2) * (a + 5) * (a + 8) * (a + 11) + 82 > 0 := 
sorry

end positive_expression_with_b_l93_93423


namespace min_value_expression_l93_93736

theorem min_value_expression : ∃ x y : ℝ, (x = 2 ∧ y = -3/2) ∧ ∀ a b : ℝ, 2 * a^2 + 2 * b^2 - 8 * a + 6 * b + 28 ≥ 10.5 :=
sorry

end min_value_expression_l93_93736


namespace parallelogram_height_l93_93734

theorem parallelogram_height (A B H : ℝ) (hA : A = 462) (hB : B = 22) (hArea : A = B * H) : H = 21 :=
by
  sorry

end parallelogram_height_l93_93734


namespace tile_in_center_l93_93426

-- Define the coloring pattern of the grid
inductive Color
| A | B | C

-- Predicates for grid, tile placement, and colors
def Grid := Fin 5 × Fin 5

def is_1x3_tile (t : Grid × Grid × Grid) : Prop :=
  -- Ensure each tuple t represents three cells that form a $1 \times 3$ tile
  sorry

def is_tiling (g : Grid → Option Color) : Prop :=
  -- Ensure the entire grid is correctly tiled with the given tiles and within the coloring pattern
  sorry

def center : Grid := (Fin.mk 2 (by decide), Fin.mk 2 (by decide))

-- The theorem statement
theorem tile_in_center (g : Grid → Option Color) : is_tiling g → 
  (∃! tile : Grid, g tile = some Color.B) :=
sorry

end tile_in_center_l93_93426


namespace ball_count_difference_l93_93355

open Nat

theorem ball_count_difference :
  (total_balls = 145) →
  (soccer_balls = 20) →
  (basketballs > soccer_balls) →
  (tennis_balls = 2 * soccer_balls) →
  (baseballs = soccer_balls + 10) →
  (volleyballs = 30) →
  let accounted_balls := soccer_balls + tennis_balls + baseballs + volleyballs
  let basketballs := total_balls - accounted_balls
  (basketballs - soccer_balls = 5) :=
by
  intros
  let tennis_balls := 2 * soccer_balls
  let baseballs := soccer_balls + 10
  let accounted_balls := soccer_balls + tennis_balls + baseballs + volleyballs
  let basketballs := total_balls - accounted_balls
  exact sorry

end ball_count_difference_l93_93355


namespace total_miles_ran_l93_93084

theorem total_miles_ran (miles_monday miles_wednesday miles_friday : ℕ)
  (h1 : miles_monday = 3)
  (h2 : miles_wednesday = 2)
  (h3 : miles_friday = 7) :
  miles_monday + miles_wednesday + miles_friday = 12 := 
by
  sorry

end total_miles_ran_l93_93084


namespace domain_sqrt_sin_cos_l93_93285

open Real

theorem domain_sqrt_sin_cos (k : ℤ) :
  {x : ℝ | ∃ k : ℤ, (2 * k * π + π / 4 ≤ x) ∧ (x ≤ 2 * k * π + 5 * π / 4)} = 
  {x : ℝ | sin x - cos x ≥ 0} :=
sorry

end domain_sqrt_sin_cos_l93_93285


namespace no_such_six_tuples_exist_l93_93011

theorem no_such_six_tuples_exist :
  ∀ (a b c x y z : ℕ),
    1 ≤ c → c ≤ b → b ≤ a →
    1 ≤ z → z ≤ y → y ≤ x →
    2 * a + b + 4 * c = 4 * x * y * z →
    2 * x + y + 4 * z = 4 * a * b * c →
    False :=
by
  intros a b c x y z h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end no_such_six_tuples_exist_l93_93011


namespace cobbler_works_fri_hours_l93_93877

-- Conditions
def mending_rate : ℕ := 3  -- Pairs of shoes per hour
def mon_to_thu_days : ℕ := 4
def hours_per_day : ℕ := 8
def weekly_mended_pairs : ℕ := 105

-- Translate the conditions
def hours_mended_mon_to_thu : ℕ := mon_to_thu_days * hours_per_day
def pairs_mended_mon_to_thu : ℕ := mending_rate * hours_mended_mon_to_thu
def pairs_mended_fri : ℕ := weekly_mended_pairs - pairs_mended_mon_to_thu

-- Theorem statement to prove the desired question
theorem cobbler_works_fri_hours : (pairs_mended_fri / mending_rate) = 3 := by
  sorry

end cobbler_works_fri_hours_l93_93877


namespace linda_original_savings_l93_93070

theorem linda_original_savings (S : ℝ) (f : ℝ) (a : ℝ) (t : ℝ) 
  (h1 : f = 7 / 13 * S) (h2 : a = 3 / 13 * S) 
  (h3 : t = S - f - a) (h4 : t = 180) (h5 : a = 360) : 
  S = 1560 :=
by 
  sorry

end linda_original_savings_l93_93070


namespace find_value_at_frac_one_third_l93_93896

theorem find_value_at_frac_one_third
  (f : ℝ → ℝ) 
  (a : ℝ)
  (h₁ : ∀ x, f x = x ^ a)
  (h₂ : f 2 = 1 / 4) :
  f (1 / 3) = 9 := 
  sorry

end find_value_at_frac_one_third_l93_93896


namespace tiles_walked_on_l93_93946

/-- 
A park has a rectangular shape with a width of 13 feet and a length of 19 feet.
Square-shaped tiles of dimension 1 foot by 1 foot cover the entire area.
The gardener walks in a straight line from one corner of the rectangle to the opposite corner.
One specific tile in the path is not to be stepped on. 
Prove that the number of tiles the gardener walks on is 30.
-/
theorem tiles_walked_on (width length gcd_width_length tiles_to_avoid : ℕ)
  (h_width : width = 13)
  (h_length : length = 19)
  (h_gcd : gcd width length = 1)
  (h_tiles_to_avoid : tiles_to_avoid = 1) : 
  (width + length - gcd_width_length - tiles_to_avoid = 30) := 
by
  sorry

end tiles_walked_on_l93_93946


namespace contradiction_proof_l93_93816

theorem contradiction_proof (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) : ¬ (¬ (a > 0) ∨ ¬ (b > 0) ∨ ¬ (c > 0)) → false :=
by sorry

end contradiction_proof_l93_93816


namespace event_B_C_mutually_exclusive_l93_93012

-- Define the events based on the given conditions
def EventA (products : Type) (is_defective : products → Prop) (x y : products) : Prop :=
  ¬is_defective x ∧ ¬is_defective y

def EventB (products : Type) (is_defective : products → Prop) (x y : products) : Prop :=
  is_defective x ∧ is_defective y

def EventC (products : Type) (is_defective : products → Prop) (x y : products) : Prop :=
  ¬(is_defective x ∧ is_defective y)

-- Prove that Event B and Event C are mutually exclusive
theorem event_B_C_mutually_exclusive (products : Type) (is_defective : products → Prop) (x y : products) :
  (EventB products is_defective x y) → ¬(EventC products is_defective x y) :=
sorry

end event_B_C_mutually_exclusive_l93_93012


namespace ratio_a_f_l93_93318

theorem ratio_a_f (a b c d e f : ℕ)
  (h1 : a / b = 1 / 4)
  (h2 : b / c = 13 / 9)
  (h3 : c / d = 5 / 13)
  (h4 : d / e = 2 / 3)
  (h5 : e / f = 7 / 5) :
  a / f = 7 / 6 := by
  sorry

end ratio_a_f_l93_93318


namespace custom_dollar_five_neg3_l93_93680

-- Define the custom operation
def custom_dollar (a b : Int) : Int :=
  a * (b - 1) + a * b

-- State the theorem
theorem custom_dollar_five_neg3 : custom_dollar 5 (-3) = -35 := by
  sorry

end custom_dollar_five_neg3_l93_93680


namespace num_exclusive_multiples_4_6_less_151_l93_93852

def numMultiplesExclusive (n : ℕ) (a b : ℕ) : ℕ :=
  let lcm_ab := Nat.lcm a b
  (n-1) / a - (n-1) / lcm_ab + (n-1) / b - (n-1) / lcm_ab

theorem num_exclusive_multiples_4_6_less_151 : 
  numMultiplesExclusive 151 4 6 = 38 := 
by 
  sorry

end num_exclusive_multiples_4_6_less_151_l93_93852


namespace combined_work_rate_l93_93322

-- Define the context and the key variables
variable (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b)

-- State the theorem corresponding to the proof problem
theorem combined_work_rate (h_a : a ≠ 0) (h_b : b ≠ 0) : 
  1/a + 1/b = (a * b) / (a + b) * (1/a * 1/b) :=
sorry

end combined_work_rate_l93_93322


namespace num_pos_multiples_of_six_is_150_l93_93699

theorem num_pos_multiples_of_six_is_150 : 
  ∃ (n : ℕ), (∀ k, (n = 150) ↔ (102 + (k - 1) * 6 = 996 ∧ 102 ≤ 6 * k ∧ 6 * k ≤ 996)) :=
sorry

end num_pos_multiples_of_six_is_150_l93_93699


namespace normal_level_shortage_l93_93039

variable (T : ℝ) (normal_capacity : ℝ) (end_of_month_reservoir : ℝ)
variable (h1 : end_of_month_reservoir = 6)
variable (h2 : end_of_month_reservoir = 2 * normal_capacity)
variable (h3 : end_of_month_reservoir = 0.60 * T)

theorem normal_level_shortage :
  normal_capacity = 7 :=
by
  sorry

end normal_level_shortage_l93_93039


namespace total_legs_camden_dogs_l93_93628

variable (c r j : ℕ) -- c: Camden's dogs, r: Rico's dogs, j: Justin's dogs

theorem total_legs_camden_dogs :
  (r = j + 10) ∧ (j = 14) ∧ (c = (3 * r) / 4) → 4 * c = 72 :=
by
  sorry

end total_legs_camden_dogs_l93_93628


namespace probability_two_units_of_origin_l93_93188

def square_vertices (x_min x_max y_min y_max : ℝ) :=
  { p : ℝ × ℝ // x_min ≤ p.1 ∧ p.1 ≤ x_max ∧ y_min ≤ p.2 ∧ p.2 ≤ y_max }

def within_radius (r : ℝ) (origin : ℝ × ℝ) (p : ℝ × ℝ) :=
  (p.1 - origin.1)^2 + (p.2 - origin.2)^2 ≤ r^2

noncomputable def probability_within_radius (x_min x_max y_min y_max r : ℝ) : ℝ :=
  let square_area := (x_max - x_min) * (y_max - y_min)
  let circle_area := r^2 * Real.pi
  circle_area / square_area

theorem probability_two_units_of_origin :
  probability_within_radius (-3) 3 (-3) 3 2 = Real.pi / 9 :=
by
  sorry

end probability_two_units_of_origin_l93_93188


namespace max_trains_final_count_l93_93757

-- Define the conditions
def trains_per_birthdays : Nat := 1
def trains_per_christmas : Nat := 2
def trains_per_easter : Nat := 3
def years : Nat := 7

-- Function to calculate total trains after 7 years
def total_trains_after_years (trains_per_years : Nat) (num_years : Nat) : Nat :=
  trains_per_years * num_years

-- Calculate inputs
def trains_per_year : Nat := trains_per_birthdays + trains_per_christmas + trains_per_easter
def total_initial_trains : Nat := total_trains_after_years trains_per_year years

-- Bonus and final steps
def bonus_trains_from_cousins (initial_trains : Nat) : Nat := initial_trains / 2
def final_total_trains (initial_trains : Nat) (bonus_trains : Nat) : Nat :=
  let after_bonus := initial_trains + bonus_trains
  let additional_from_parents := after_bonus * 3
  after_bonus + additional_from_parents

-- Main theorem
theorem max_trains_final_count : final_total_trains total_initial_trains (bonus_trains_from_cousins total_initial_trains) = 252 := by
  sorry

end max_trains_final_count_l93_93757


namespace find_difference_l93_93158

theorem find_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 := 
by
  sorry

end find_difference_l93_93158


namespace renovation_project_total_l93_93622

def sand : ℝ := 0.17
def dirt : ℝ := 0.33
def cement : ℝ := 0.17

theorem renovation_project_total : sand + dirt + cement = 0.67 := 
by
  sorry

end renovation_project_total_l93_93622


namespace twenty_kopeck_greater_than_ten_kopeck_l93_93513

-- Definitions of the conditions
variables (x y z : ℕ)
axiom total_coins : x + y + z = 30 
axiom total_value : 10 * x + 15 * y + 20 * z = 500 

-- The proof statement
theorem twenty_kopeck_greater_than_ten_kopeck : z > x :=
sorry

end twenty_kopeck_greater_than_ten_kopeck_l93_93513


namespace highest_probability_white_ball_l93_93150

theorem highest_probability_white_ball :
  let red_balls := 2
  let black_balls := 3
  let white_balls := 4
  let total_balls := red_balls + black_balls + white_balls
  let prob_red := red_balls / total_balls
  let prob_black := black_balls / total_balls
  let prob_white := white_balls / total_balls
  prob_white > prob_black ∧ prob_black > prob_red :=
by
  sorry

end highest_probability_white_ball_l93_93150


namespace value_of_a_range_of_m_l93_93484

def f (x a : ℝ) : ℝ := abs (x - a)

-- Given the following conditions
axiom cond1 (x : ℝ) (a : ℝ) : f x a = abs (x - a)
axiom cond2 (x : ℝ) (a : ℝ) : (f x a >= 3) ↔ (x <= 1 ∨ x >= 5)

-- Prove that a = 2
theorem value_of_a (a : ℝ) : (∀ x : ℝ, (f x a >= 3) ↔ (x <= 1 ∨ x >= 5)) → a = 2 := by
  sorry

-- Additional condition for m
axiom cond3 (x : ℝ) (a : ℝ) (m : ℝ) : ∀ x : ℝ, f x a + f (x + 4) a >= m

-- Prove that m ≤ 4
theorem range_of_m (a : ℝ) (m : ℝ) : (∀ x : ℝ, f x a + f (x + 4) a >= m) → a = 2 → m ≤ 4 := by
  sorry

end value_of_a_range_of_m_l93_93484


namespace employee_y_payment_l93_93748

variable (x y : ℝ)

def total_payment (x y : ℝ) : ℝ := x + y
def x_payment (y : ℝ) : ℝ := 1.20 * y

theorem employee_y_payment : (total_payment x y = 638) ∧ (x = x_payment y) → y = 290 :=
by
  sorry

end employee_y_payment_l93_93748


namespace second_number_is_40_l93_93350

-- Defining the problem
theorem second_number_is_40
  (a b c : ℚ)
  (h1 : a + b + c = 120)
  (h2 : a = (3/4 : ℚ) * b)
  (h3 : c = (5/4 : ℚ) * b) :
  b = 40 :=
sorry

end second_number_is_40_l93_93350


namespace solve_system_of_equations_l93_93211

theorem solve_system_of_equations : ∃ (x y : ℝ), (2 * x - y = 3) ∧ (3 * x + 2 * y = 8) ∧ (x = 2) ∧ (y = 1) := by
  sorry

end solve_system_of_equations_l93_93211


namespace nth_odd_positive_integer_is_199_l93_93930

def nth_odd_positive_integer (n : ℕ) : ℕ :=
  2 * n - 1

theorem nth_odd_positive_integer_is_199 :
  nth_odd_positive_integer 100 = 199 :=
by
  sorry

end nth_odd_positive_integer_is_199_l93_93930


namespace difference_place_values_l93_93656

def place_value (digit : Char) (position : String) : Real :=
  match digit, position with
  | '1', "hundreds" => 100
  | '1', "tenths" => 0.1
  | _, _ => 0 -- for any other cases (not required in this problem)

theorem difference_place_values :
  (place_value '1' "hundreds" - place_value '1' "tenths" = 99.9) :=
by
  sorry

end difference_place_values_l93_93656


namespace silver_coin_value_l93_93124

--- Definitions from the conditions
def total_value_hoard (value_silver : ℕ) := 100 * 3 * value_silver + 60 * value_silver + 33

--- Statement of the theorem to prove
theorem silver_coin_value (x : ℕ) (h : total_value_hoard x = 2913) : x = 8 :=
by {
  sorry
}

end silver_coin_value_l93_93124


namespace num_valid_pairs_l93_93194

theorem num_valid_pairs : ∃ (n : ℕ), n = 8 ∧ (∀ (a b : ℕ), 0 < a ∧ 0 < b ∧ a + b ≤ 150 ∧ ((a + 1 / b) / (1 / a + b) = 17) ↔ (a = 17 * b) ∧ b ≤ 8) :=
by
  sorry

end num_valid_pairs_l93_93194


namespace find_a_l93_93855

variable {a : ℝ}

def A : Set ℝ := {2, 4}
def B (a : ℝ) : Set ℝ := {a, a^2 + 3}

theorem find_a (h : A ∩ (B a) = {2}) : a = 2 :=
by
  sorry

end find_a_l93_93855


namespace maxOccursAt2_l93_93142

def f (x : ℝ) : ℝ := -x^2 + 4 * x - 3

theorem maxOccursAt2 {m : ℝ} :
  (∀ x, 0 ≤ x ∧ x ≤ m → f x ≤ f m) ∧ 0 ≤ m ∧ m ≤ 2 → (0 < m ∧ m ≤ 2) :=
sorry

end maxOccursAt2_l93_93142


namespace ball_picking_problem_proof_l93_93503

-- Define the conditions
def red_balls : ℕ := 8
def white_balls : ℕ := 7

-- Define the questions
def num_ways_to_pick_one_ball : ℕ :=
  red_balls + white_balls

def num_ways_to_pick_two_different_color_balls : ℕ :=
  red_balls * white_balls

-- Define the correct answers
def correct_answer_to_pick_one_ball : ℕ := 15
def correct_answer_to_pick_two_different_color_balls : ℕ := 56

-- State the theorem to be proved
theorem ball_picking_problem_proof :
  (num_ways_to_pick_one_ball = correct_answer_to_pick_one_ball) ∧
  (num_ways_to_pick_two_different_color_balls = correct_answer_to_pick_two_different_color_balls) :=
by
  sorry

end ball_picking_problem_proof_l93_93503


namespace find_x_squared_plus_y_squared_l93_93655

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : (x - y)^2 = 49) (h2 : x * y = -8) : x^2 + y^2 = 33 := 
by 
  sorry

end find_x_squared_plus_y_squared_l93_93655


namespace eval_expression_at_values_l93_93042

theorem eval_expression_at_values : 
  ∀ x y : ℕ, x = 3 ∧ y = 4 → 
  5 * (x^(y+1)) + 6 * (y^(x+1)) + 2 * x * y = 2775 :=
by
  intros x y hxy
  cases hxy
  sorry

end eval_expression_at_values_l93_93042


namespace distance_from_edge_to_bottom_l93_93233

theorem distance_from_edge_to_bottom (d x : ℕ) 
  (h1 : 63 + d + 20 = 10 + d + x) : x = 73 := by
  -- This is where the proof would go
  sorry

end distance_from_edge_to_bottom_l93_93233


namespace number_is_0_point_5_l93_93665

theorem number_is_0_point_5 (x : ℝ) (h : x = 1/6 + 0.33333333333333337) : x = 0.5 := 
by
  -- The actual proof would go here.
  sorry

end number_is_0_point_5_l93_93665


namespace triangle_area_is_4_l93_93022

-- Define the lines
def line1 (x : ℝ) : ℝ := 4
def line2 (x : ℝ) : ℝ := 2 + x
def line3 (x : ℝ) : ℝ := 2 - x

-- Define intersection points
def intersection1 : ℝ × ℝ := (2, 4)
def intersection2 : ℝ × ℝ := (-2, 4)
def intersection3 : ℝ × ℝ := (0, 2)

-- Function to calculate the area of a triangle using its vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

-- Statement of the proof problem
theorem triangle_area_is_4 :
  ∀ A B C : ℝ × ℝ, A = intersection1 → B = intersection2 → C = intersection3 →
  triangle_area A B C = 4 := by
  sorry

end triangle_area_is_4_l93_93022


namespace alloy_mixture_l93_93223

theorem alloy_mixture (x y : ℝ) 
  (h1 : x + y = 1000)
  (h2 : 0.25 * x + 0.50 * y = 450) : 
  x = 200 ∧ y = 800 :=
by
  -- Proof will follow here
  sorry

end alloy_mixture_l93_93223


namespace extreme_value_proof_l93_93274

noncomputable def extreme_value (x y : ℝ) := 4 * x + 3 * y 

theorem extreme_value_proof 
  (x y : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (h : x + y = 5 * x * y) : 
  extreme_value x y = 3 :=
sorry

end extreme_value_proof_l93_93274


namespace final_amounts_calculation_l93_93104

noncomputable def article_A_original_cost : ℚ := 200
noncomputable def article_B_original_cost : ℚ := 300
noncomputable def article_C_original_cost : ℚ := 400
noncomputable def exchange_rate_euro_to_usd : ℚ := 1.10
noncomputable def exchange_rate_gbp_to_usd : ℚ := 1.30
noncomputable def discount_A : ℚ := 0.50
noncomputable def discount_B : ℚ := 0.30
noncomputable def discount_C : ℚ := 0.40
noncomputable def sales_tax_rate : ℚ := 0.05
noncomputable def reward_points : ℚ := 100
noncomputable def reward_point_value : ℚ := 0.05

theorem final_amounts_calculation :
  let discounted_A := article_A_original_cost * discount_A
  let final_A := (article_A_original_cost - discounted_A) * exchange_rate_euro_to_usd
  let discounted_B := article_B_original_cost * discount_B
  let final_B := (article_B_original_cost - discounted_B) * exchange_rate_gbp_to_usd
  let discounted_C := article_C_original_cost * discount_C
  let final_C := article_C_original_cost - discounted_C
  let total_discounted_cost_usd := final_A + final_B + final_C
  let sales_tax := total_discounted_cost_usd * sales_tax_rate
  let reward := reward_points * reward_point_value
  let final_amount_usd := total_discounted_cost_usd + sales_tax - reward
  let final_amount_euro := final_amount_usd / exchange_rate_euro_to_usd
  final_amount_usd = 649.15 ∧ final_amount_euro = 590.14 :=
by
  sorry

end final_amounts_calculation_l93_93104


namespace triangle_isosceles_if_equal_bisectors_l93_93836

theorem triangle_isosceles_if_equal_bisectors
  (A B C : ℝ)
  (a b c l_a l_b : ℝ)
  (ha : l_a = l_b)
  (h1 : l_a = 2 * b * c * Real.cos (A / 2) / (b + c))
  (h2 : l_b = 2 * a * c * Real.cos (B / 2) / (a + c)) :
  a = b :=
by
  sorry

end triangle_isosceles_if_equal_bisectors_l93_93836


namespace harriet_return_speed_l93_93920

/-- Harriet's trip details: 
  - speed from A-ville to B-town is 100 km/h
  - the entire trip took 5 hours
  - time to drive from A-ville to B-town is 180 minutes (3 hours) 
  Prove the speed while driving back to A-ville is 150 km/h
--/
theorem harriet_return_speed:
  ∀ (t₁ t₂ : ℝ),
  (t₁ = 3) ∧ 
  (100 * t₁ = d) ∧ 
  (t₁ + t₂ = 5) ∧ 
  (t₂ = 2) →
  (d / t₂ = 150) :=
by
  intros t₁ t₂ h
  sorry

end harriet_return_speed_l93_93920


namespace parabola_whose_directrix_is_tangent_to_circle_l93_93048

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 9

noncomputable def is_tangent (line_eq : ℝ → ℝ → Prop) (circle_eq : ℝ → ℝ → Prop) : Prop := 
  ∃ p : ℝ × ℝ, (line_eq p.1 p.2) ∧ (circle_eq p.1 p.2) ∧ 
  (∀ q : ℝ × ℝ, (circle_eq q.1 q.2) → (line_eq q.1 q.2) → q = p)

-- Definitions of parabolas
noncomputable def parabola_A_directrix (x y : ℝ) : Prop := y = 2

noncomputable def parabola_B_directrix (x y : ℝ) : Prop := x = 2

noncomputable def parabola_C_directrix (x y : ℝ) : Prop := x = -4

noncomputable def parabola_D_directrix (x y : ℝ) : Prop := y = -1

-- The final statement to prove
theorem parabola_whose_directrix_is_tangent_to_circle :
  is_tangent parabola_D_directrix circle_eq ∧ ¬ is_tangent parabola_A_directrix circle_eq ∧ 
  ¬ is_tangent parabola_B_directrix circle_eq ∧ ¬ is_tangent parabola_C_directrix circle_eq :=
sorry

end parabola_whose_directrix_is_tangent_to_circle_l93_93048


namespace part_one_part_two_l93_93492

-- Define the function f
def f (x : ℝ) : ℝ := abs (x - 2) + abs (x + 1)

-- Define the inequality condition
def inequality_condition (x : ℝ) : Prop := f x ≥ 4 - x

-- Problem set (I)
theorem part_one (x : ℝ) : inequality_condition x ↔ (x ≤ -3 ∨ x ≥ 1) :=
sorry

-- Define range conditions for a and b
def range_condition (a b : ℝ) : Prop := a ≥ 3 ∧ b ≥ 3

-- Problem set (II)
theorem part_two (a b : ℝ) (h : range_condition a b) : 2 * (a + b) < a * b + 4 :=
sorry

end part_one_part_two_l93_93492


namespace intersection_M_N_eq_M_l93_93548

-- Definitions of M and N
def M : Set ℝ := { x : ℝ | x^2 - x < 0 }
def N : Set ℝ := { x : ℝ | abs x < 2 }

-- Proof statement
theorem intersection_M_N_eq_M : M ∩ N = M := 
  sorry

end intersection_M_N_eq_M_l93_93548


namespace quadrilateral_area_l93_93319

theorem quadrilateral_area (a b x : ℝ)
  (h1: ∀ (y z : ℝ), y^2 + z^2 = a^2 ∧ (x + y)^2 + (x + z)^2 = b^2)
  (hx_perp: ∀ (p q : ℝ), x * q = 0 ∧ x * p = 0) :
  S = (1 / 4) * |b^2 - a^2| :=
by
  sorry

end quadrilateral_area_l93_93319


namespace find_numbers_l93_93407

theorem find_numbers (x y a : ℕ) (h1 : x = 6 * y - a) (h2 : x + y = 38) : 7 * x = 228 - a → y = 38 - x :=
by
  sorry

end find_numbers_l93_93407


namespace area_of_sector_l93_93162

theorem area_of_sector (r : ℝ) (theta : ℝ) (h_r : r = 6) (h_theta : theta = 60) : (θ / 360 * π * r^2 = 6 * π) :=
by sorry

end area_of_sector_l93_93162


namespace ratio_of_investments_l93_93526

theorem ratio_of_investments (I B_profit total_profit : ℝ) (x : ℝ)
  (h1 : B_profit = 4000) (h2 : total_profit = 28000) (h3 : I * (2 * B_profit / 4000 - 1) = total_profit - B_profit) :
  x = 3 :=
by
  sorry

end ratio_of_investments_l93_93526


namespace negation_of_existence_l93_93027

theorem negation_of_existence :
  ¬ (∃ x : ℝ, 0 < x ∧ x^3 - x + 1 > 0) ↔ ∀ x : ℝ, 0 < x → x^3 - x + 1 ≤ 0 :=
by sorry

end negation_of_existence_l93_93027


namespace find_expression_value_l93_93686

theorem find_expression_value (x y : ℝ) (h : x / (2 * y) = 3 / 2) : (7 * x + 8 * y) / (x - 2 * y) = 29 := by
  sorry

end find_expression_value_l93_93686


namespace jenny_research_time_l93_93414

noncomputable def time_spent_on_research (total_hours : ℕ) (proposal_hours : ℕ) (report_hours : ℕ) : ℕ :=
  total_hours - proposal_hours - report_hours

theorem jenny_research_time : time_spent_on_research 20 2 8 = 10 := by
  sorry

end jenny_research_time_l93_93414


namespace cesaro_sum_51_term_sequence_l93_93256

noncomputable def cesaro_sum (B : List ℝ) : ℝ :=
  let T := List.scanl (· + ·) 0 B
  T.drop 1 |>.sum / B.length

theorem cesaro_sum_51_term_sequence (B : List ℝ) (h_length : B.length = 49)
  (h_cesaro_sum_49 : cesaro_sum B = 500) :
  cesaro_sum (B ++ [0, 0]) = 1441.18 :=
by
  sorry

end cesaro_sum_51_term_sequence_l93_93256


namespace remainder_div_1234_567_89_1011_mod_12_l93_93356

theorem remainder_div_1234_567_89_1011_mod_12 :
  (1234^567 + 89^1011) % 12 = 9 := 
sorry

end remainder_div_1234_567_89_1011_mod_12_l93_93356


namespace evaluate_expression_l93_93055

theorem evaluate_expression : (3 : ℚ) / (1 - (2 : ℚ) / 5) = 5 := sorry

end evaluate_expression_l93_93055


namespace number_of_perfect_square_divisors_of_450_l93_93496

theorem number_of_perfect_square_divisors_of_450 : 
    let p := 450;
    let factors := [(3, 2), (5, 2), (2, 1)];
    ∃ n, (n = 4 ∧ 
          ∀ (d : ℕ), d ∣ p → 
                     (∃ (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ 
                              (a = 0) ∧ (b = 0 ∨ b = 2) ∧ (c = 0 ∨ c = 2) → 
                              a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) :=
    sorry

end number_of_perfect_square_divisors_of_450_l93_93496


namespace net_change_in_onions_l93_93020

-- Definitions for the given conditions
def onions_added_by_sara : ℝ := 4.5
def onions_taken_by_sally : ℝ := 5.25
def onions_added_by_fred : ℝ := 9.75

-- Statement of the problem to be proved
theorem net_change_in_onions : 
  onions_added_by_sara - onions_taken_by_sally + onions_added_by_fred = 9 := 
by
  sorry -- hint that proof is required

end net_change_in_onions_l93_93020


namespace isosceles_triangle_l93_93614

theorem isosceles_triangle 
  {a b : ℝ} {α β : ℝ} 
  (h : a / (Real.cos α) = b / (Real.cos β)) : 
  a = b :=
sorry

end isosceles_triangle_l93_93614


namespace shaded_area_correct_l93_93465

noncomputable def grid_width : ℕ := 15
noncomputable def grid_height : ℕ := 5
noncomputable def triangle_base : ℕ := 15
noncomputable def triangle_height : ℕ := 3
noncomputable def total_area : ℝ := (grid_width * grid_height : ℝ)
noncomputable def triangle_area : ℝ := (1 / 2) * triangle_base * triangle_height
noncomputable def shaded_area : ℝ := total_area - triangle_area

theorem shaded_area_correct : shaded_area = 52.5 := 
by sorry

end shaded_area_correct_l93_93465


namespace mean_equality_l93_93112

theorem mean_equality (z : ℝ) :
  (8 + 15 + 24) / 3 = (16 + z) / 2 → z = 15.34 :=
by
  intro h
  sorry

end mean_equality_l93_93112


namespace genevieve_drinks_pints_l93_93282

theorem genevieve_drinks_pints (total_gallons : ℝ) (thermoses : ℕ) 
  (gallons_to_pints : ℝ) (genevieve_thermoses : ℕ) 
  (h1 : total_gallons = 4.5) (h2 : thermoses = 18) 
  (h3 : gallons_to_pints = 8) (h4 : genevieve_thermoses = 3) : 
  (total_gallons * gallons_to_pints / thermoses) * genevieve_thermoses = 6 := 
by
  admit

end genevieve_drinks_pints_l93_93282


namespace find_initial_population_l93_93518

theorem find_initial_population
  (birth_rate : ℕ)
  (death_rate : ℕ)
  (net_growth_rate_percent : ℝ)
  (net_growth_rate_per_person : ℕ)
  (h1 : birth_rate = 32)
  (h2 : death_rate = 11)
  (h3 : net_growth_rate_percent = 2.1)
  (h4 : net_growth_rate_per_person = birth_rate - death_rate)
  (h5 : (net_growth_rate_per_person : ℝ) / 100 = net_growth_rate_percent / 100) :
  P = 1000 :=
by
  sorry

end find_initial_population_l93_93518


namespace number_of_buyers_l93_93125

theorem number_of_buyers 
  (today yesterday day_before : ℕ) 
  (h1 : today = yesterday + 40) 
  (h2 : yesterday = day_before / 2) 
  (h3 : day_before + yesterday + today = 140) : 
  day_before = 67 :=
by
  -- skip the proof
  sorry

end number_of_buyers_l93_93125


namespace gcd_polynomial_primes_l93_93183

theorem gcd_polynomial_primes (a : ℤ) (k : ℤ) (ha : a = 2 * 947 * k) : 
  Int.gcd (3 * a^2 + 47 * a + 101) (a + 19) = 1 :=
by
  sorry

end gcd_polynomial_primes_l93_93183


namespace no_way_to_write_as_sum_l93_93208

def can_be_written_as_sum (S : ℕ → ℕ) (n : ℕ) (k : ℕ) : Prop :=
  n + k - 1 + (n - 1) * (k - 1) / 2 = 528 ∧ n > 0 ∧ 2 ∣ n ∧ k > 1

theorem no_way_to_write_as_sum : 
  ∀ (S : ℕ → ℕ) (n k : ℕ), can_be_written_as_sum S n k →
    0 = 0 :=
by
  -- Problem states that there are 0 valid ways to write 528 as the sum
  -- of an increasing sequence of two or more consecutive positive integers
  sorry

end no_way_to_write_as_sum_l93_93208


namespace solve_by_completing_square_l93_93281

theorem solve_by_completing_square (x: ℝ) (h: x^2 + 4 * x - 3 = 0) : (x + 2)^2 = 7 := 
by 
  sorry

end solve_by_completing_square_l93_93281


namespace value_of_c_l93_93533

theorem value_of_c
    (x y c : ℝ)
    (h1 : 3 * x - 5 * y = 5)
    (h2 : x / (x + y) = c)
    (h3 : x - y = 2.999999999999999) :
    c = 0.7142857142857142 :=
by
    sorry

end value_of_c_l93_93533


namespace solve_system_of_equations_l93_93957

theorem solve_system_of_equations:
  ∃ (x y : ℚ), 3 * x + 4 * y = 16 ∧ 5 * x - 6 * y = 33 ∧ x = 6 ∧ y = -1/2 :=
by
  sorry

end solve_system_of_equations_l93_93957


namespace sum_of_interior_angles_hexagon_l93_93530

theorem sum_of_interior_angles_hexagon : 
  ∀ (n : ℕ), n = 6 → (n - 2) * 180 = 720 :=
by
  sorry

end sum_of_interior_angles_hexagon_l93_93530


namespace negation_proposition_l93_93364

theorem negation_proposition :
  ¬(∀ x : ℝ, x^2 > x) ↔ ∃ x : ℝ, x^2 ≤ x :=
sorry

end negation_proposition_l93_93364


namespace checkered_rectangle_minimal_area_checkered_rectangle_possible_perimeters_l93_93216

-- Define the conditions
def is_checkered_rectangle (S : ℕ) : Prop :=
  (∃ (a b : ℕ), a * b = S) ∧
  (∀ x y k l : ℕ, x * 13 + y * 1 = S) ∧
  (S % 39 = 0)

-- Define that S is minimal satisfying the conditions
def minimal_area_checkered_rectangle (S : ℕ) : Prop :=
  is_checkered_rectangle S ∧
  (∀ (S' : ℕ), S' < S → ¬ is_checkered_rectangle S')

-- Prove that S = 78 is the minimal area
theorem checkered_rectangle_minimal_area : minimal_area_checkered_rectangle 78 :=
  sorry

-- Define the condition for possible perimeters
def possible_perimeters (S : ℕ) (p : ℕ) : Prop :=
  (∀ (a b : ℕ), a * b = S → 2 * (a + b) = p)

-- Prove the possible perimeters for area 78
theorem checkered_rectangle_possible_perimeters :
  ∀ p, p = 38 ∨ p = 58 ∨ p = 82 ↔ possible_perimeters 78 p :=
  sorry

end checkered_rectangle_minimal_area_checkered_rectangle_possible_perimeters_l93_93216


namespace midpoint_product_zero_l93_93956

theorem midpoint_product_zero (x y : ℝ) :
  let A := (2, 6)
  let B := (x, y)
  let C := (4, 3)
  (C = ((2 + x) / 2, (6 + y) / 2)) → (x * y = 0) := by
  intros
  sorry

end midpoint_product_zero_l93_93956


namespace age_of_twin_brothers_l93_93928

theorem age_of_twin_brothers (x : Nat) : (x + 1) * (x + 1) = x * x + 11 ↔ x = 5 :=
by
  sorry  -- Proof omitted.

end age_of_twin_brothers_l93_93928


namespace solve_quadratic_l93_93258

def quadratic_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem solve_quadratic : (quadratic_eq (-2) 1 3 (-1)) ∧ (quadratic_eq (-2) 1 3 (3/2)) :=
by
  sorry

end solve_quadratic_l93_93258


namespace final_number_of_cards_l93_93180

def initial_cards : ℕ := 26
def cards_given_to_mary : ℕ := 18
def cards_found_in_box : ℕ := 40
def cards_given_to_john : ℕ := 12
def cards_purchased_at_fleamarket : ℕ := 25

theorem final_number_of_cards :
  (initial_cards - cards_given_to_mary) + (cards_found_in_box - cards_given_to_john) + cards_purchased_at_fleamarket = 61 :=
by sorry

end final_number_of_cards_l93_93180


namespace range_f1_l93_93251
open Function

theorem range_f1 (a : ℝ) : (∀ x y : ℝ, x ∈ Set.Ici (-1) → y ∈ Set.Ici (-1) → x ≤ y → (x^2 + 2*a*x + 3) ≤ (y^2 + 2*a*y + 3)) →
  6 ≤ (1^2 + 2*a*1 + 3) :=
by
  intro h
  sorry

end range_f1_l93_93251


namespace part1_equation_solution_part2_inequality_solution_l93_93209

theorem part1_equation_solution (x : ℝ) (h : x / (x - 1) = (x - 1) / (2 * (x - 1))) : 
  x = -1 :=
sorry

theorem part2_inequality_solution (x : ℝ) (h₁ : 5 * x - 1 > 3 * x - 4) (h₂ : - (1 / 3) * x ≤ 2 / 3 - x) : 
  -3 / 2 < x ∧ x ≤ 1 :=
sorry

end part1_equation_solution_part2_inequality_solution_l93_93209


namespace fill_tank_with_leak_l93_93910

theorem fill_tank_with_leak (A L : ℝ) (h1 : A = 1 / 6) (h2 : L = 1 / 18) : (1 / (A - L)) = 9 :=
by
  sorry

end fill_tank_with_leak_l93_93910


namespace GPA_of_rest_of_classroom_l93_93895

variable (n : ℕ) (x : ℝ)
variable (H1 : ∀ n, n > 0)
variable (H2 : (15 * n + 2 * n * x) / (3 * n) = 17)

theorem GPA_of_rest_of_classroom (n : ℕ) (H1 : ∀ n, n > 0) (H2 : (15 * n + 2 * n * x) / (3 * n) = 17) : x = 18 := by
  sorry

end GPA_of_rest_of_classroom_l93_93895


namespace number_of_outcomes_l93_93949

-- Define the conditions
def students : Nat := 4
def events : Nat := 3

-- Define the problem: number of possible outcomes for the champions
theorem number_of_outcomes : students ^ events = 64 :=
by sorry

end number_of_outcomes_l93_93949


namespace K_time_9_hours_l93_93395

theorem K_time_9_hours
  (x : ℝ) -- x is the speed of K
  (hx : 45 / x = 9) -- K's time for 45 miles is 9 hours
  (y : ℝ) -- y is the speed of M
  (h₁ : x = y + 0.5) -- K travels 0.5 mph faster than M
  (h₂ : 45 / y - 45 / x = 3 / 4) -- K takes 3/4 hour less than M
  : 45 / x = 9 :=
by
  sorry

end K_time_9_hours_l93_93395


namespace baseball_league_games_l93_93569

theorem baseball_league_games (n m : ℕ) (h : 3 * n + 4 * m = 76) (h1 : n > 2 * m) (h2 : m > 4) : n = 16 :=
by 
  sorry

end baseball_league_games_l93_93569


namespace kishore_savings_l93_93061

noncomputable def rent := 5000
noncomputable def milk := 1500
noncomputable def groceries := 4500
noncomputable def education := 2500
noncomputable def petrol := 2000
noncomputable def miscellaneous := 700
noncomputable def total_expenses := rent + milk + groceries + education + petrol + miscellaneous
noncomputable def salary : ℝ := total_expenses / 0.9 -- given that savings is 10% of salary

theorem kishore_savings : (salary * 0.1) = 1800 :=
by
  sorry

end kishore_savings_l93_93061


namespace smallest_of_powers_l93_93943

theorem smallest_of_powers :
  (2:ℤ)^(55) < (3:ℤ)^(44) ∧ (2:ℤ)^(55) < (5:ℤ)^(33) ∧ (2:ℤ)^(55) < (6:ℤ)^(22) := by
  sorry

end smallest_of_powers_l93_93943


namespace solution_set_system_of_inequalities_l93_93479

theorem solution_set_system_of_inequalities :
  { x : ℝ | (2 - x) * (2 * x + 4) ≥ 0 ∧ -3 * x^2 + 2 * x + 1 < 0 } = 
  { x : ℝ | -2 ≤ x ∧ x < -1/3 ∨ 1 < x ∧ x ≤ 2 } := 
by
  sorry

end solution_set_system_of_inequalities_l93_93479


namespace other_root_of_quadratic_l93_93431

theorem other_root_of_quadratic (m t : ℝ) : (∀ (x : ℝ),
    (3 * x^2 - m * x - 3 = 0) → 
    (x = 1)) → 
    (1 * t = -1) := 
sorry

end other_root_of_quadratic_l93_93431


namespace age_difference_l93_93691

theorem age_difference (M S : ℕ) (h1 : S = 16) (h2 : M + 2 = 2 * (S + 2)) : M - S = 18 :=
by
  sorry

end age_difference_l93_93691


namespace min_abs_diff_x1_x2_l93_93702

theorem min_abs_diff_x1_x2 (x1 x2 : ℝ) (f : ℝ → ℝ) (Hf : ∀ x, f x = Real.sin (π * x))
  (Hbounds : ∀ x, f x1 ≤ f x ∧ f x ≤ f x2) : |x1 - x2| = 1 := 
by
  sorry

end min_abs_diff_x1_x2_l93_93702


namespace ordered_triples_count_l93_93063

noncomputable def count_valid_triples (n : ℕ) :=
  ∃ x y z : ℕ, ∃ k : ℕ, x * y * z = k ∧ k = 5 ∧ lcm x y = 48 ∧ lcm x z = 450 ∧ lcm y z = 600

theorem ordered_triples_count : count_valid_triples 5 := by
  sorry

end ordered_triples_count_l93_93063


namespace common_ratio_of_arithmetic_sequence_l93_93808

theorem common_ratio_of_arithmetic_sequence (S_odd S_even : ℤ) (q : ℤ) 
  (h1 : S_odd + S_even = -240) (h2 : S_odd - S_even = 80) 
  (h3 : q = S_even / S_odd) : q = 2 := 
  sorry

end common_ratio_of_arithmetic_sequence_l93_93808


namespace son_l93_93974

theorem son's_present_age
  (S F : ℤ)
  (h1 : F = S + 45)
  (h2 : F + 10 = 4 * (S + 10))
  (h3 : S + 15 = 2 * S) :
  S = 15 :=
by
  sorry

end son_l93_93974


namespace soccer_team_probability_l93_93729

theorem soccer_team_probability :
  let total_players := 12
  let forwards := 6
  let defenders := 6
  let total_ways := Nat.choose total_players 2
  let defender_ways := Nat.choose defenders 2
  ∃ p : ℚ, p = defender_ways / total_ways ∧ p = 5 / 22 :=
sorry

end soccer_team_probability_l93_93729


namespace fraction_equality_l93_93316

theorem fraction_equality
  (a b : ℝ)
  (x : ℝ)
  (h1 : x = (a^2) / (b^2))
  (h2 : a ≠ b)
  (h3 : b ≠ 0) :
  (a^2 + b^2) / (a^2 - b^2) = (x + 1) / (x - 1) :=
by
  sorry

end fraction_equality_l93_93316


namespace expression_value_l93_93098

theorem expression_value : 7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 := 
by 
  -- proof goes here 
  sorry

end expression_value_l93_93098


namespace segment_area_formula_l93_93546
noncomputable def area_of_segment (r a : ℝ) : ℝ :=
  r^2 * Real.arcsin (a / (2 * r)) - (a / 4) * Real.sqrt (4 * r^2 - a^2)

theorem segment_area_formula (r a : ℝ) : area_of_segment r a =
  r^2 * Real.arcsin (a / (2 * r)) - (a / 4) * Real.sqrt (4 * r^2 - a^2) :=
sorry

end segment_area_formula_l93_93546


namespace sum_of_digits_of_9ab_l93_93780

noncomputable def a : ℕ := 10^2023 - 1
noncomputable def b : ℕ := 2*(10^2023 - 1) / 3

def digitSum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem sum_of_digits_of_9ab :
  digitSum (9 * a * b) = 20235 :=
by
  sorry

end sum_of_digits_of_9ab_l93_93780


namespace geometric_series_sum_l93_93521

open Real

theorem geometric_series_sum :
  let a1 := (5 / 4 : ℝ)
  let r := (5 / 4 : ℝ)
  let n := (12 : ℕ)
  let S := a1 * (1 - r^n) / (1 - r)
  S = -716637955 / 16777216 :=
by
  let a1 := (5 / 4 : ℝ)
  let r := (5 / 4 : ℝ)
  let n := (12 : ℕ)
  let S := a1 * (1 - r^n) / (1 - r)
  have h : S = -716637955 / 16777216 := sorry
  exact h

end geometric_series_sum_l93_93521


namespace cubic_repeated_root_b_eq_100_l93_93825

theorem cubic_repeated_root_b_eq_100 (b : ℝ) (h1 : b ≠ 0)
  (h2 : ∃ x : ℝ, (b * x^3 + 15 * x^2 + 9 * x + 2 = 0) ∧ 
                 (3 * b * x^2 + 30 * x + 9 = 0)) :
  b = 100 :=
sorry

end cubic_repeated_root_b_eq_100_l93_93825


namespace alice_stops_in_quarter_D_l93_93822

-- Definitions and conditions
def indoor_track_circumference : ℕ := 40
def starting_point_S : ℕ := 0
def run_distance : ℕ := 1600

-- Desired theorem statement
theorem alice_stops_in_quarter_D :
  (run_distance % indoor_track_circumference = 0) → 
  (0 ≤ (run_distance % indoor_track_circumference) ∧ 
   (run_distance % indoor_track_circumference) < indoor_track_circumference) → 
  true := by
  sorry

end alice_stops_in_quarter_D_l93_93822


namespace find_k_l93_93792

theorem find_k (k : ℝ) (x : ℝ) :
  x^2 + k * x + 1 = 0 ∧ x^2 - x - k = 0 → k = 2 := 
sorry

end find_k_l93_93792


namespace circle_chord_intersect_zero_l93_93980

noncomputable def circle_product (r : ℝ) : ℝ :=
  let O := (0, 0)
  let A := (r, 0)
  let B := (-r, 0)
  let C := (0, r)
  let D := (0, -r)
  let P := (0, 0)
  (dist A P) * (dist P B)

theorem circle_chord_intersect_zero (r : ℝ) :
  let A := (r, 0)
  let B := (-r, 0)
  let C := (0, r)
  let D := (0, -r)
  let P := (0, 0)
  (dist A P) * (dist P B) = 0 :=
by sorry

end circle_chord_intersect_zero_l93_93980


namespace probability_sum_l93_93793

noncomputable def P : ℕ → ℝ := sorry

theorem probability_sum (n : ℕ) (h : n ≥ 7) :
  P n = (1/6) * (P (n-1) + P (n-2) + P (n-3) + P (n-4) + P (n-5) + P (n-6)) :=
sorry

end probability_sum_l93_93793


namespace find_f_l93_93889

def f (x : ℝ) : ℝ := 3 * x + 2

theorem find_f (x : ℝ) : f x = 3 * x + 2 :=
  sorry

end find_f_l93_93889


namespace probability_heads_exactly_8_in_10_l93_93111

def fair_coin_probability (n k : ℕ) : ℚ := (Nat.choose n k : ℚ) / (2 ^ n)

theorem probability_heads_exactly_8_in_10 :
  fair_coin_probability 10 8 = 45 / 1024 :=
by 
  sorry

end probability_heads_exactly_8_in_10_l93_93111


namespace determine_d_and_vertex_l93_93406

-- Definition of the quadratic equation
def g (x d : ℝ) : ℝ := 3 * x^2 + 12 * x + d

-- The proof problem
theorem determine_d_and_vertex (d : ℝ) :
  (∃ x : ℝ, g x d = 0 ∧ ∀ y : ℝ, g y d ≥ g x d) ↔ (d = 12 ∧ ∀ x : ℝ, 3 > 0 ∧ (g x d ≥ g 0 d)) := 
by 
  sorry

end determine_d_and_vertex_l93_93406


namespace sum_of_digits_l93_93297

theorem sum_of_digits (x : ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (h : 10 * x + 6 * x = 16) : x + 6 * x = 7 :=
by
  -- The proof is omitted
  sorry

end sum_of_digits_l93_93297


namespace alpha_plus_beta_eq_118_l93_93958

theorem alpha_plus_beta_eq_118 (α β : ℝ) (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 96 * x + 2209) / (x^2 + 63 * x - 3969)) : α + β = 118 :=
by
  sorry

end alpha_plus_beta_eq_118_l93_93958


namespace range_of_a_l93_93044

theorem range_of_a (a : ℝ) (h₀ : a > 0) : (∃ x : ℝ, |x - 5| + |x - 1| < a) ↔ a > 4 :=
sorry

end range_of_a_l93_93044


namespace problem_statement_l93_93146

variable (f : ℝ → ℝ)
variable (h1 : ∀ x, 8 < x → f (x) > f (x + 1))
variable (h2 : ∀ x, f (x + 8) = f (-x + 8))

theorem problem_statement : f 7 > f 10 := by
  sorry

end problem_statement_l93_93146


namespace largest_number_l93_93195

theorem largest_number (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ = 0.9791) 
  (h₂ : x₂ = 0.97019)
  (h₃ : x₃ = 0.97909)
  (h₄ : x₄ = 0.971)
  (h₅ : x₅ = 0.97109)
  : max x₁ (max x₂ (max x₃ (max x₄ x₅))) = 0.9791 :=
  sorry

end largest_number_l93_93195


namespace ribbon_length_difference_l93_93577

theorem ribbon_length_difference (S : ℝ) : 
  let Seojun_ribbon := S 
  let Siwon_ribbon := S + 8.8 
  let Seojun_new := Seojun_ribbon - 4.3
  let Siwon_new := Siwon_ribbon + 4.3 
  Siwon_new - Seojun_new = 17.4 :=
by
  -- Definition of original ribbon lengths
  let Seojun_ribbon := S
  let Siwon_ribbon := S + 8.8
  -- Seojun cuts and gives 4.3 meters to Siwon
  let Seojun_new := Seojun_ribbon - 4.3
  let Siwon_new := Siwon_ribbon + 4.3
  -- Compute the difference
  have h1 : Siwon_new - Seojun_new = (S + 8.8 + 4.3) - (S - 4.3) := by sorry
  -- Prove the final answer
  have h2 : Siwon_new - Seojun_new = 17.4 := by sorry

  exact h2

end ribbon_length_difference_l93_93577


namespace find_a_b_monotonicity_l93_93295

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (x^2 + a * x + b) / x

theorem find_a_b (a b : ℝ) (h_odd : ∀ x ≠ 0, f (-x) a b = -f x a b) (h_eq : f 1 a b = f 4 a b) :
  a = 0 ∧ b = 4 := by sorry

theorem monotonicity (f : ℝ → ℝ) (h : ∀ x ≠ 0, f x = x + 4 / x) :
  (∀ x1 x2, 0 < x1 ∧ x1 ≤ 2 ∧ x1 < x2 ∧ x2 ≤ 2 → f x1 > f x2) ∧
  (∀ x1 x2, 2 < x1 ∧ x1 < x2 → f x1 < f x2) := by sorry

end find_a_b_monotonicity_l93_93295


namespace employee_pays_216_l93_93745

def retail_price (wholesale_cost : ℝ) (markup_percentage : ℝ) : ℝ :=
    wholesale_cost + markup_percentage * wholesale_cost

def employee_payment (retail_price : ℝ) (discount_percentage : ℝ) : ℝ :=
    retail_price - discount_percentage * retail_price

theorem employee_pays_216 (wholesale_cost : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) :
    wholesale_cost = 200 ∧ markup_percentage = 0.20 ∧ discount_percentage = 0.10 →
    employee_payment (retail_price wholesale_cost markup_percentage) discount_percentage = 216 :=
by
  intro h
  rcases h with ⟨h_wholesale, h_markup, h_discount⟩
  rw [h_wholesale, h_markup, h_discount]
  -- Now we have to prove the final statement: employee_payment (retail_price 200 0.20) 0.10 = 216
  -- This follows directly by computation, so we leave it as a sorry for now
  sorry

end employee_pays_216_l93_93745


namespace total_payment_correct_l93_93671

-- Define the prices of different apples.
def price_small_apple : ℝ := 1.5
def price_medium_apple : ℝ := 2.0
def price_big_apple : ℝ := 3.0

-- Define the quantities of apples bought by Donny.
def quantity_small_apples : ℕ := 6
def quantity_medium_apples : ℕ := 6
def quantity_big_apples : ℕ := 8

-- Define the conditions.
def discount_medium_apples_threshold : ℕ := 5
def discount_medium_apples_rate : ℝ := 0.20
def tax_rate : ℝ := 0.10
def big_apple_special_offer_count : ℕ := 3
def big_apple_special_offer_discount_rate : ℝ := 0.50

-- Step function to calculate discount and total cost.
noncomputable def total_cost : ℝ :=
  let cost_small := quantity_small_apples * price_small_apple
  let cost_medium := quantity_medium_apples * price_medium_apple
  let discount_medium := if quantity_medium_apples > discount_medium_apples_threshold 
                         then cost_medium * discount_medium_apples_rate else 0
  let cost_medium_after_discount := cost_medium - discount_medium
  let cost_big := quantity_big_apples * price_big_apple
  let discount_big := (quantity_big_apples / big_apple_special_offer_count) * 
                       (price_big_apple * big_apple_special_offer_discount_rate)
  let cost_big_after_discount := cost_big - discount_big
  let total_cost_before_tax := cost_small + cost_medium_after_discount + cost_big_after_discount
  let tax := total_cost_before_tax * tax_rate
  total_cost_before_tax + tax

-- Define the expected total payment.
def expected_total_payment : ℝ := 43.56

-- The theorem statement: Prove that total_cost equals the expected total payment.
theorem total_payment_correct : total_cost = expected_total_payment := sorry

end total_payment_correct_l93_93671


namespace find_trapezoid_bases_l93_93725

-- Define the conditions of the isosceles trapezoid
variables {AD BC : ℝ}
variables (h1 : ∀ (A B C D : ℝ), is_isosceles_trapezoid A B C D ∧ intersects_at_right_angle A B C D)
variables (h2 : ∀ {A B C D : ℝ}, trapezoid_area A B C D = 12)
variables (h3 : ∀ {A B C D : ℝ}, trapezoid_height A B C D = 2)

-- Prove the bases AD and BC are 8 and 4 respectively under the given conditions
theorem find_trapezoid_bases (AD BC : ℝ) : 
  AD = 8 ∧ BC = 4 :=
  sorry

end find_trapezoid_bases_l93_93725


namespace cricket_average_increase_l93_93139

theorem cricket_average_increase
    (A : ℝ) -- average score after 18 innings
    (score19 : ℝ) -- runs scored in 19th inning
    (new_average : ℝ) -- new average after 19 innings
    (score19_def : score19 = 97)
    (new_average_def :  new_average = 25)
    (total_runs_def : 19 * new_average = 18 * A + 97) : 
    new_average - (18 * A + score19) / 19 = 4 := 
by
  sorry

end cricket_average_increase_l93_93139


namespace sum_of_first_eight_terms_l93_93834

theorem sum_of_first_eight_terms (a : ℝ) (r : ℝ) 
  (h1 : r = 2) (h2 : a * (1 + 2 + 4 + 8) = 1) :
  a * (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 17 :=
by
  -- sorry is used to skip the proof
  sorry

end sum_of_first_eight_terms_l93_93834


namespace complementary_angles_not_obtuse_l93_93311

-- Define the concept of complementary angles.
def is_complementary (a b : ℝ) : Prop :=
  a + b = 90

-- Define that neither angle should be obtuse.
def not_obtuse (a b : ℝ) : Prop :=
  a < 90 ∧ b < 90

-- Proof problem statement
theorem complementary_angles_not_obtuse (a b : ℝ) (ha : a < 90) (hb : b < 90) (h_comp : is_complementary a b) : 
  not_obtuse a b :=
by
  sorry

end complementary_angles_not_obtuse_l93_93311


namespace arrangement_count_example_l93_93411

theorem arrangement_count_example 
  (teachers : Finset String) 
  (students : Finset String) 
  (locations : Finset String) 
  (h_teachers : teachers.card = 2) 
  (h_students : students.card = 4) 
  (h_locations : locations.card = 2)
  : ∃ n : ℕ, n = 12 := 
sorry

end arrangement_count_example_l93_93411


namespace average_growth_rate_inequality_l93_93165

theorem average_growth_rate_inequality (p q x : ℝ) (h₁ : (1+x)^2 = (1+p)*(1+q)) (h₂ : p ≠ q) :
  x < (p + q) / 2 :=
sorry

end average_growth_rate_inequality_l93_93165


namespace f_2011_equals_1_l93_93172

-- Define odd function property
def is_odd_function (f : ℤ → ℤ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Define function with period property
def has_period_3 (f : ℤ → ℤ) : Prop :=
  ∀ x, f (x + 3) = f (x)

-- Define main problem statement
theorem f_2011_equals_1 
  (f : ℤ → ℤ)
  (h1 : is_odd_function f)
  (h2 : has_period_3 f)
  (h3 : f (-1) = -1) 
  : f 2011 = 1 :=
sorry

end f_2011_equals_1_l93_93172


namespace p_iff_q_l93_93291

def f (x a : ℝ) := x * (x - a) * (x - 2)

def p (a : ℝ) := 0 < a ∧ a < 2

def q (a : ℝ) := 
  let f' x := 3 * x^2 - 2 * (a + 2) * x + 2 * a
  f' a < 0

theorem p_iff_q (a : ℝ) : (p a) ↔ (q a) := by
  sorry

end p_iff_q_l93_93291


namespace at_least_one_not_less_than_two_l93_93385

theorem at_least_one_not_less_than_two (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x + 1/y) ≥ 2 ∨ (y + 1/z) ≥ 2 ∨ (z + 1/x) ≥ 2 :=
by
  sorry

end at_least_one_not_less_than_two_l93_93385


namespace general_term_a_n_l93_93951

open BigOperators

variable {a : ℕ → ℝ}  -- The sequence a_n
variable {S : ℕ → ℝ}  -- The sequence sum S_n

-- Define the sum of the first n terms:
def seq_sum (a : ℕ → ℝ) (n : ℕ) := ∑ k in Finset.range (n + 1), a k

theorem general_term_a_n (h : ∀ n : ℕ, S n = 2 ^ n - 1) (n : ℕ) : a n = 2 ^ (n - 1) :=
by
  sorry

end general_term_a_n_l93_93951


namespace zero_of_f_l93_93609

noncomputable def f (x : ℝ) : ℝ := 2^x - 4

theorem zero_of_f : f 2 = 0 :=
by
  sorry

end zero_of_f_l93_93609


namespace robin_albums_l93_93739

theorem robin_albums (phone_pics : ℕ) (camera_pics : ℕ) (pics_per_album : ℕ) (total_pics : ℕ) (albums_created : ℕ)
  (h1 : phone_pics = 35)
  (h2 : camera_pics = 5)
  (h3 : pics_per_album = 8)
  (h4 : total_pics = phone_pics + camera_pics)
  (h5 : albums_created = total_pics / pics_per_album) : albums_created = 5 := 
sorry

end robin_albums_l93_93739


namespace james_income_ratio_l93_93626

theorem james_income_ratio
  (January_earnings : ℕ := 4000)
  (Total_earnings : ℕ := 18000)
  (Earnings_difference : ℕ := 2000) :
  ∃ (February_earnings : ℕ), 
    (January_earnings + February_earnings + (February_earnings - Earnings_difference) = Total_earnings) ∧
    (February_earnings / January_earnings = 2) := by
  sorry

end james_income_ratio_l93_93626


namespace sequence_divisible_by_11_l93_93284

theorem sequence_divisible_by_11 {a : ℕ → ℕ} (h1 : a 1 = 1) (h2 : a 2 = 3)
    (h_rec : ∀ n : ℕ, a (n + 2) = (n + 3) * a (n + 1) - (n + 2) * a n) :
    (a 4 % 11 = 0) ∧ (a 8 % 11 = 0) ∧ (a 10 % 11 = 0) ∧ (∀ n, n ≥ 11 → a n % 11 = 0) :=
by
  sorry

end sequence_divisible_by_11_l93_93284


namespace sector_area_l93_93046

theorem sector_area (r l : ℝ) (h_r : r = 2) (h_l : l = 3) : (1/2) * l * r = 3 :=
by
  rw [h_r, h_l]
  norm_num

end sector_area_l93_93046


namespace inverse_variation_z_x_square_l93_93288

theorem inverse_variation_z_x_square (x z : ℝ) (K : ℝ) 
  (h₀ : z * x^2 = K) 
  (h₁ : x = 3 ∧ z = 2)
  (h₂ : z = 8) :
  x = 3 / 2 := 
by 
  sorry

end inverse_variation_z_x_square_l93_93288


namespace boxes_containing_pans_l93_93378

def num_boxes : Nat := 26
def num_teacups_per_box : Nat := 20
def num_cups_broken_per_box : Nat := 2
def teacups_left : Nat := 180

def num_teacup_boxes (num_boxes : Nat) (num_teacups_per_box : Nat) (num_cups_broken_per_box : Nat) (teacups_left : Nat) : Nat :=
  teacups_left / (num_teacups_per_box - num_cups_broken_per_box)

def num_remaining_boxes (num_boxes : Nat) (num_teacup_boxes : Nat) : Nat :=
  num_boxes - num_teacup_boxes

def num_pans_boxes (num_remaining_boxes : Nat) : Nat :=
  num_remaining_boxes / 2

theorem boxes_containing_pans : ∀ (num_boxes num_teacups_per_box num_cups_broken_per_box teacups_left : Nat),
  num_boxes = 26 →
  num_teacups_per_box = 20 →
  num_cups_broken_per_box = 2 →
  teacups_left = 180 →
  num_pans_boxes (num_remaining_boxes num_boxes (num_teacup_boxes num_boxes num_teacups_per_box num_cups_broken_per_box teacups_left)) = 8 :=
by
  intros
  sorry

end boxes_containing_pans_l93_93378


namespace inequality_false_implies_range_of_a_l93_93633

theorem inequality_false_implies_range_of_a (a : ℝ) : 
  (∀ t : ℝ, t^2 - 2 * t - a ≥ 0) ↔ a ≤ -1 :=
by
  sorry

end inequality_false_implies_range_of_a_l93_93633


namespace abs_nested_expression_l93_93558

theorem abs_nested_expression (x : ℝ) (h : x = 2023) : 
  abs (abs (abs x - x) - abs x) - x = 0 :=
by
  subst h
  sorry

end abs_nested_expression_l93_93558


namespace sara_staircase_steps_l93_93416

-- Define the problem statement and conditions
theorem sara_staircase_steps (n : ℕ) :
  (3 * n * (n + 1) / 2 = 270) → n = 12 := 
by
  intro h
  sorry

end sara_staircase_steps_l93_93416


namespace average_age_of_5_students_l93_93159

theorem average_age_of_5_students
  (avg_age_20_students : ℕ → ℕ → ℕ → ℕ)
  (total_age_20 : avg_age_20_students 20 20 0 = 400)
  (total_age_9 : 9 * 16 = 144)
  (age_20th_student : ℕ := 186) :
  avg_age_20_students 5 ((400 - 144 - 186) / 5) 5 = 14 :=
by
  sorry

end average_age_of_5_students_l93_93159


namespace find_scooters_l93_93941

variables (b t s : ℕ)

theorem find_scooters (h1 : b + t + s = 13) (h2 : 2 * b + 3 * t + 2 * s = 30) : s = 9 :=
sorry

end find_scooters_l93_93941


namespace primitive_root_set_equality_l93_93618

theorem primitive_root_set_equality 
  {p : ℕ} (hp : Nat.Prime p) (hodd: p % 2 = 1) (g : ℕ) (hg : g ^ (p - 1) % p = 1) :
  (∀ k, 1 ≤ k ∧ k ≤ (p - 1) / 2 → ∃ m, 1 ≤ m ∧ m ≤ (p - 1) / 2 ∧ (k^2 + 1) % p = g ^ m % p) ↔ p = 3 :=
by sorry

end primitive_root_set_equality_l93_93618


namespace point_below_parabola_l93_93644

theorem point_below_parabola (a b c : ℝ) (h : 2 < a + b + c) : 
  2 < c + b + a :=
by
  sorry

end point_below_parabola_l93_93644


namespace number_of_pupils_in_class_l93_93236

theorem number_of_pupils_in_class
(U V : ℕ) (increase : ℕ) (avg_increase : ℕ) (n : ℕ) 
(h1 : U = 85) (h2 : V = 45) (h3 : increase = U - V) (h4 : avg_increase = 1 / 2) (h5 : increase / avg_increase = n) :
n = 80 := by
sorry

end number_of_pupils_in_class_l93_93236


namespace geometric_sequence_b_mn_theorem_l93_93679

noncomputable def geometric_sequence_b_mn (b : ℕ → ℝ) (c d : ℝ) (m n : ℕ) 
  (h_b : ∀ (k : ℕ), b k > 0)
  (h_seq : ∃ q : ℝ, (q ≠ 0) ∧ ∀ k : ℕ, b k = b 1 * q ^ (k - 1))
  (h_m : b m = c)
  (h_n : b n = d)
  (h_cond : n - m ≥ 2) 
  (h_nm_pos : m > 0 ∧ n > 0): Prop :=
  b (m + n) = (d ^ n / c ^ m) ^ (1 / (n - m))

-- We skip the proof using sorry.
theorem geometric_sequence_b_mn_theorem 
  (b : ℕ → ℝ) (c d : ℝ) (m n : ℕ)
  (h_b : ∀ (k : ℕ), b k > 0)
  (h_seq : ∃ q : ℝ, (q ≠ 0) ∧ ∀ k : ℕ, b k = b 1 * q ^ (k - 1))
  (h_m : b m = c)
  (h_n : b n = d)
  (h_cond : n - m ≥ 2)
  (h_nm_pos : m > 0 ∧ n > 0) : 
  b (m + n) = (d ^ n / c ^ m) ^ (1 / (n - m)) :=
sorry

end geometric_sequence_b_mn_theorem_l93_93679


namespace Lee_payment_total_l93_93718

theorem Lee_payment_total 
  (ticket_price : ℝ := 10.00)
  (booking_fee : ℝ := 1.50)
  (youngest_discount : ℝ := 0.40)
  (oldest_discount : ℝ := 0.30)
  (middle_discount : ℝ := 0.20)
  (youngest_tickets : ℕ := 3)
  (oldest_tickets : ℕ := 3)
  (middle_tickets : ℕ := 4) :
  (youngest_tickets * (ticket_price * (1 - youngest_discount)) + 
   oldest_tickets * (ticket_price * (1 - oldest_discount)) + 
   middle_tickets * (ticket_price * (1 - middle_discount)) + 
   (youngest_tickets + oldest_tickets + middle_tickets) * booking_fee) = 86.00 :=
by 
  sorry

end Lee_payment_total_l93_93718


namespace greatest_y_value_l93_93512

theorem greatest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -6) : y ≤ 3 :=
sorry

end greatest_y_value_l93_93512


namespace figure_Z_has_largest_shaded_area_l93_93635

noncomputable def shaded_area_X :=
  let rectangle_area := 4 * 2
  let circle_area := Real.pi * (1)^2
  rectangle_area - circle_area

noncomputable def shaded_area_Y :=
  let rectangle_area := 4 * 2
  let semicircle_area := (1 / 2) * Real.pi * (1)^2
  rectangle_area - semicircle_area

noncomputable def shaded_area_Z :=
  let outer_square_area := 4^2
  let inner_square_area := 2^2
  outer_square_area - inner_square_area

theorem figure_Z_has_largest_shaded_area :
  shaded_area_Z > shaded_area_X ∧ shaded_area_Z > shaded_area_Y :=
by
  sorry

end figure_Z_has_largest_shaded_area_l93_93635


namespace divide_composite_products_l93_93926

theorem divide_composite_products :
  let first_three := [4, 6, 8]
  let next_three := [9, 10, 12]
  let prod_first_three := first_three.prod
  let prod_next_three := next_three.prod
  (prod_first_three : ℚ) / prod_next_three = 8 / 45 :=
by
  sorry

end divide_composite_products_l93_93926


namespace sum_first_four_terms_eq_12_l93_93323

noncomputable def a : ℕ → ℤ := sorry -- An arithmetic sequence aₙ

-- Given conditions
axiom h1 : a 2 = 4
axiom h2 : a 1 + a 5 = 4 * a 3 - 4

theorem sum_first_four_terms_eq_12 : (a 1 + a 2 + a 3 + a 4) = 12 := 
by {
  sorry
}

end sum_first_four_terms_eq_12_l93_93323


namespace unique_two_digit_factors_l93_93789

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def factors (n : ℕ) (a b : ℕ) : Prop := a * b = n

theorem unique_two_digit_factors : 
  ∃! (a b : ℕ), is_two_digit a ∧ is_two_digit b ∧ factors 1950 a b :=
by sorry

end unique_two_digit_factors_l93_93789


namespace elysse_bags_per_trip_l93_93304

-- Definitions from the problem conditions
def total_bags : ℕ := 30
def total_trips : ℕ := 5
def bags_per_trip : ℕ := total_bags / total_trips

def carries_same_amount (elysse_bags brother_bags : ℕ) : Prop := elysse_bags = brother_bags

-- Statement to prove
theorem elysse_bags_per_trip :
  ∀ (elysse_bags brother_bags : ℕ), 
  bags_per_trip = elysse_bags + brother_bags → 
  carries_same_amount elysse_bags brother_bags → 
  elysse_bags = 3 := 
by 
  intros elysse_bags brother_bags h1 h2
  sorry

end elysse_bags_per_trip_l93_93304


namespace three_gt_sqrt_seven_l93_93264

theorem three_gt_sqrt_seven : 3 > Real.sqrt 7 := sorry

end three_gt_sqrt_seven_l93_93264


namespace boys_of_other_communities_l93_93927

axiom total_boys : ℕ
axiom muslim_percentage : ℝ
axiom hindu_percentage : ℝ
axiom sikh_percentage : ℝ

noncomputable def other_boy_count (total_boys : ℕ) 
                                   (muslim_percentage : ℝ) 
                                   (hindu_percentage : ℝ) 
                                   (sikh_percentage : ℝ) : ℝ :=
  let total_percentage := muslim_percentage + hindu_percentage + sikh_percentage
  let other_percentage := 1 - total_percentage
  other_percentage * total_boys

theorem boys_of_other_communities : 
    other_boy_count 850 0.44 0.32 0.10 = 119 :=
  by 
    sorry

end boys_of_other_communities_l93_93927


namespace total_savings_l93_93907

-- Define the given conditions
def number_of_tires : ℕ := 4
def sale_price : ℕ := 75
def original_price : ℕ := 84

-- State the proof problem
theorem total_savings : (original_price - sale_price) * number_of_tires = 36 :=
by
  -- Proof omitted
  sorry

end total_savings_l93_93907


namespace smallest_prime_perf_sqr_minus_eight_l93_93982

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def is_perf_sqr_minus_eight (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 8

theorem smallest_prime_perf_sqr_minus_eight :
  ∃ (n : ℕ), is_prime n ∧ is_perf_sqr_minus_eight n ∧ (∀ m : ℕ, is_prime m ∧ is_perf_sqr_minus_eight m → n ≤ m) :=
sorry

end smallest_prime_perf_sqr_minus_eight_l93_93982


namespace M_is_set_of_positive_rationals_le_one_l93_93637

def M : Set ℚ := {x | 0 < x ∧ x ≤ 1}

axiom contains_one (M : Set ℚ) : 1 ∈ M

axiom closed_under_operations (M : Set ℚ) :
  ∀ x ∈ M, (1 / (1 + x) ∈ M) ∧ (x / (1 + x) ∈ M)

theorem M_is_set_of_positive_rationals_le_one :
  M = {x | 0 < x ∧ x ≤ 1} :=
sorry

end M_is_set_of_positive_rationals_le_one_l93_93637


namespace overlapping_area_l93_93573

def area_of_overlap (g1 g2 : Grid) : ℝ :=
  -- Dummy implementation to ensure code compiles
  6.0

structure Grid :=
  (size : ℝ) (arrow_direction : Direction)

inductive Direction
| North
| West

theorem overlapping_area (g1 g2 : Grid) 
  (h1 : g1.size = 4) 
  (h2 : g2.size = 4) 
  (d1 : g1.arrow_direction = Direction.North) 
  (d2 : g2.arrow_direction = Direction.West) 
  : area_of_overlap g1 g2 = 6 :=
by
  sorry

end overlapping_area_l93_93573


namespace amount_paid_is_51_l93_93273

def original_price : ℕ := 204
def discount_fraction : ℚ := 0.75
def paid_fraction : ℚ := 1 - discount_fraction

theorem amount_paid_is_51 : paid_fraction * original_price = 51 := by
  sorry

end amount_paid_is_51_l93_93273


namespace son_time_to_complete_work_l93_93881

noncomputable def man_work_rate : ℚ := 1 / 6
noncomputable def combined_work_rate : ℚ := 1 / 3

theorem son_time_to_complete_work :
  (1 / (combined_work_rate - man_work_rate)) = 6 := by
  sorry

end son_time_to_complete_work_l93_93881


namespace tom_total_amount_l93_93307

-- Definitions of the initial conditions
def initial_amount : ℕ := 74
def amount_earned : ℕ := 86

-- Main statement to prove
theorem tom_total_amount : initial_amount + amount_earned = 160 := 
by
  -- sorry added to skip the proof
  sorry

end tom_total_amount_l93_93307


namespace smallest_percentage_increase_l93_93996

theorem smallest_percentage_increase :
  let n2005 := 75
  let n2006 := 85
  let n2007 := 88
  let n2008 := 94
  let n2009 := 96
  let n2010 := 102
  let perc_increase (a b : ℕ) := ((b - a) : ℚ) / a * 100
  perc_increase n2008 n2009 < perc_increase n2006 n2007 ∧
  perc_increase n2008 n2009 < perc_increase n2007 n2008 ∧
  perc_increase n2008 n2009 < perc_increase n2009 n2010 ∧
  perc_increase n2008 n2009 < perc_increase n2005 n2006
:= sorry

end smallest_percentage_increase_l93_93996


namespace min_value_3x_plus_4y_l93_93675

theorem min_value_3x_plus_4y (x y : ℝ) (h_pos : 0 < x ∧ 0 < y) (h_eq : x + 3*y = 5*x*y) :
  ∃ (c : ℝ), (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 3 * y = 5 * x * y → 3 * x + 4 * y ≥ c) ∧ c = 5 :=
sorry

end min_value_3x_plus_4y_l93_93675


namespace smallest_positive_period_max_min_values_l93_93469

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.sin x)
noncomputable def f (x : ℝ) : ℝ := (vector_a x).1 * (vector_b x).1 + (vector_a x).2 * (vector_b x).2 - 1 / 2

-- Theorem 1: Smallest positive period of the function f(x)
theorem smallest_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi :=
  sorry

-- Theorem 2: Maximum and minimum values of the function f(x) on [0, π/2]
theorem max_min_values : 
  ∀ x ∈ Set.Icc 0 (Real.pi / 2),
    f x ≤ 1 ∧ f x ≥ -1 / 2 ∧ (∃ (x_max : ℝ), x_max ∈ Set.Icc 0 (Real.pi / 2) ∧ f x_max = 1) ∧
    (∃ (x_min : ℝ), x_min ∈ Set.Icc 0 (Real.pi / 2) ∧ f x_min = -1 / 2) :=
  sorry

end smallest_positive_period_max_min_values_l93_93469


namespace quadratic_unique_solution_k_neg_l93_93028

theorem quadratic_unique_solution_k_neg (k : ℝ) :
  (∃ x : ℝ, 9 * x^2 + k * x + 36 = 0 ∧ ∀ y : ℝ, 9 * y^2 + k * y + 36 = 0 → y = x) →
  k = -36 :=
by
  sorry

end quadratic_unique_solution_k_neg_l93_93028


namespace num_int_values_x_l93_93498

theorem num_int_values_x (x : ℕ) :
  (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) → ∃ n : ℕ, n = 3 :=
by
  sorry

end num_int_values_x_l93_93498


namespace sum_of_three_smallest_positive_solutions_l93_93362

theorem sum_of_three_smallest_positive_solutions :
  let sol1 := 2
  let sol2 := 8 / 3
  let sol3 := 7 / 2
  sol1 + sol2 + sol3 = 8 + 1 / 6 :=
by
  sorry

end sum_of_three_smallest_positive_solutions_l93_93362


namespace calc_area_of_quadrilateral_l93_93632

-- Define the terms and conditions using Lean definitions
noncomputable def triangle_areas : ℕ × ℕ × ℕ := (6, 9, 15)

-- State the theorem
theorem calc_area_of_quadrilateral (a b c d : ℕ) (area1 area2 area3 : ℕ):
  area1 = 6 →
  area2 = 9 →
  area3 = 15 →
  a + b + c + d = area1 + area2 + area3 →
  d = 65 :=
  sorry

end calc_area_of_quadrilateral_l93_93632


namespace product_of_five_consecutive_integers_not_perfect_square_l93_93072

theorem product_of_five_consecutive_integers_not_perfect_square (n : ℕ) : 
  ¬ ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) = k * k :=
by {
  sorry
}

end product_of_five_consecutive_integers_not_perfect_square_l93_93072


namespace distance_between_A_and_B_l93_93035

def average_speed : ℝ := 50  -- Speed in miles per hour

def travel_time : ℝ := 15.8  -- Time in hours

noncomputable def total_distance : ℝ := average_speed * travel_time  -- Distance in miles

theorem distance_between_A_and_B :
  total_distance = 790 :=
by
  sorry

end distance_between_A_and_B_l93_93035


namespace simplify_expression_l93_93134

variable (y : ℝ)

theorem simplify_expression : 3 * y + 4 * y^2 - 2 - (7 - 3 * y - 4 * y^2) = 8 * y^2 + 6 * y - 9 := 
  by
  sorry

end simplify_expression_l93_93134


namespace greatest_difference_areas_l93_93331

theorem greatest_difference_areas (l w l' w' : ℕ) (h₁ : 2*l + 2*w = 120) (h₂ : 2*l' + 2*w' = 120) : 
  l * w ≤ 900 ∧ (l = 30 → w = 30) ∧ l' * w' ≤ 900 ∧ (l' = 30 → w' = 30)  → 
  ∃ (A₁ A₂ : ℕ), (A₁ = l * w ∧ A₂ = l' * w') ∧ (841 = l * w - l' * w') := 
sorry

end greatest_difference_areas_l93_93331


namespace red_fraction_is_three_fifths_l93_93847

noncomputable def fraction_of_red_marbles (x : ℕ) : ℚ := 
  let blue_marbles := (2 / 3 : ℚ) * x
  let red_marbles := x - blue_marbles
  let new_red_marbles := 3 * red_marbles
  let new_total_marbles := blue_marbles + new_red_marbles
  new_red_marbles / new_total_marbles

theorem red_fraction_is_three_fifths (x : ℕ) (hx : x ≠ 0) : fraction_of_red_marbles x = 3 / 5 :=
by {
  sorry
}

end red_fraction_is_three_fifths_l93_93847


namespace problem_a_b_sum_l93_93482

-- Define the operation
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- Given conditions
variable (a b : ℝ)

-- Theorem statement: Prove that a + b = 4
theorem problem_a_b_sum :
  (∀ x, ((2 < x) ∧ (x < 3)) ↔ ((x - a) * (x - b - 1) < 0)) → a + b = 4 :=
by
  sorry

end problem_a_b_sum_l93_93482


namespace sandwiches_consumption_difference_l93_93707

theorem sandwiches_consumption_difference :
  let monday_lunch := 3
  let monday_dinner := 2 * monday_lunch
  let monday_total := monday_lunch + monday_dinner

  let tuesday_lunch := 4
  let tuesday_dinner := tuesday_lunch / 2
  let tuesday_total := tuesday_lunch + tuesday_dinner

  let wednesday_lunch := 2 * tuesday_lunch
  let wednesday_dinner := 3 * tuesday_lunch
  let wednesday_total := wednesday_lunch + wednesday_dinner

  let combined_monday_tuesday := monday_total + tuesday_total

  combined_monday_tuesday - wednesday_total = -5 :=
by
  sorry

end sandwiches_consumption_difference_l93_93707


namespace probability_three_heads_l93_93871

noncomputable def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def probability (n : ℕ) (k : ℕ) : ℚ :=
  (binom n k) / (2 ^ n)

theorem probability_three_heads : probability 12 3 = 55 / 1024 := 
by
  sorry

end probability_three_heads_l93_93871


namespace number_of_two_element_subsets_l93_93800

def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem number_of_two_element_subsets (S : Type*) [Fintype S] 
  (h : binomial_coeff (Fintype.card S) 7 = 36) :
  binomial_coeff (Fintype.card S) 2 = 36 :=
by
  sorry

end number_of_two_element_subsets_l93_93800


namespace y_intercept_tangent_line_l93_93377

/-- Three circles have radii 3, 2, and 1 respectively. The first circle has center at (3,0), 
the second at (7,0), and the third at (11,0). A line is tangent to all three circles 
at points in the first quadrant. Prove the y-intercept of this line is 36.
-/
theorem y_intercept_tangent_line
  (r1 r2 r3 : ℝ) (h1 : r1 = 3) (h2 : r2 = 2) (h3 : r3 = 1)
  (c1 c2 c3 : ℝ × ℝ) (hc1 : c1 = (3, 0)) (hc2 : c2 = (7, 0)) (hc3 : c3 = (11, 0)) :
  ∃ y_intercept : ℝ, y_intercept = 36 :=
sorry

end y_intercept_tangent_line_l93_93377


namespace number_of_negative_x_values_l93_93506

theorem number_of_negative_x_values : 
  (∃ (n : ℕ), ∀ (x : ℤ), x = n^2 - 196 ∧ x < 0) ∧ (n ≤ 13) :=
by 
  -- To formalize our problem we need quantifiers, inequalities and integer properties.
  sorry

end number_of_negative_x_values_l93_93506


namespace hyperbola_eccentricity_l93_93805

open Real

theorem hyperbola_eccentricity (a b c : ℝ) (ha : 0 < a) (hb : 0 < b)
    (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
    (h_right_focus : ∀ x y, x = c ∧ y = 0)
    (h_circle : ∀ x y, (x - c)^2 + y^2 = 4 * a^2)
    (h_tangent : ∀ x y, x = c ∧ y = 0 → (x^2 + y^2 = a^2 + b^2))
    : ∃ e : ℝ, e = sqrt 5 := by sorry

end hyperbola_eccentricity_l93_93805


namespace total_weight_of_shells_l93_93370

noncomputable def initial_weight : ℝ := 5.25
noncomputable def weight_large_shell_g : ℝ := 700
noncomputable def grams_per_pound : ℝ := 453.592
noncomputable def additional_weight : ℝ := 4.5

/-
We need to prove:
5.25 pounds (initial weight) + (700 grams * (1 pound / 453.592 grams)) (weight of large shell in pounds) + 4.5 pounds (additional weight) = 11.293235835 pounds
-/
theorem total_weight_of_shells :
  initial_weight + (weight_large_shell_g / grams_per_pound) + additional_weight = 11.293235835 := by
    -- Proof will be inserted here
    sorry

end total_weight_of_shells_l93_93370


namespace equal_rental_costs_l93_93057

variable {x : ℝ}

def SunshineCarRentalsCost (x : ℝ) : ℝ := 17.99 + 0.18 * x
def CityRentalsCost (x : ℝ) : ℝ := 18.95 + 0.16 * x

theorem equal_rental_costs (x : ℝ) : SunshineCarRentalsCost x = CityRentalsCost x ↔ x = 48 :=
by
  sorry

end equal_rental_costs_l93_93057


namespace amount_B_l93_93502

noncomputable def A : ℝ := sorry -- Definition of A
noncomputable def B : ℝ := sorry -- Definition of B

-- Conditions
def condition1 : Prop := A + B = 100
def condition2 : Prop := (3 / 10) * A = (1 / 5) * B

-- Statement to prove
theorem amount_B : condition1 ∧ condition2 → B = 60 :=
by
  intros
  sorry

end amount_B_l93_93502


namespace find_divisor_l93_93975

theorem find_divisor (x : ℕ) (h : 144 = (x * 13) + 1) : x = 11 := by
  sorry

end find_divisor_l93_93975


namespace water_depth_is_60_l93_93579

def Ron_height : ℕ := 12
def depth_of_water (h_R : ℕ) : ℕ := 5 * h_R

theorem water_depth_is_60 : depth_of_water Ron_height = 60 :=
by
  sorry

end water_depth_is_60_l93_93579


namespace possible_values_of_N_l93_93272

theorem possible_values_of_N (N : ℕ) (h1 : N ≥ 8 + 1)
  (h2 : ∀ (i : ℕ), (i < N → (i ≥ 0 ∧ i < 1/3 * (N-1)) → false) ) 
  (h3 : ∀ (i : ℕ), (i < N → (i ≥ 1/3 * (N-1) ∨ i < 1/3 * (N-1)) → true)) :
  23 ≤ N ∧ N ≤ 25 :=
by
  sorry

end possible_values_of_N_l93_93272


namespace scientific_notation_correct_l93_93093

-- Define the given number
def given_number : ℕ := 138000

-- Define the scientific notation expression
def scientific_notation : ℝ := 1.38 * 10^5

-- The proof goal: Prove that 138,000 expressed in scientific notation is 1.38 * 10^5
theorem scientific_notation_correct : (given_number : ℝ) = scientific_notation := by
  -- Sorry is used to skip the proof
  sorry

end scientific_notation_correct_l93_93093


namespace product_of_two_numbers_l93_93229

theorem product_of_two_numbers 
  (x y : ℝ) 
  (h₁ : x - y = 8) 
  (h₂ : x^2 + y^2 = 160) 
  : x * y = 48 := 
sorry

end product_of_two_numbers_l93_93229


namespace distinct_real_solutions_exist_l93_93226

theorem distinct_real_solutions_exist (a : ℝ) (h : a > 3 / 4) : 
  ∃ (x y : ℝ), x ≠ y ∧ x = a - y^2 ∧ y = a - x^2 := 
sorry

end distinct_real_solutions_exist_l93_93226


namespace June_sweets_count_l93_93515

variable (A M J : ℕ)

-- condition: May has three-quarters of the number of sweets that June has
def May_sweets := M = (3/4) * J

-- condition: April has two-thirds of the number of sweets that May has
def April_sweets := A = (2/3) * M

-- condition: April, May, and June have 90 sweets between them
def Total_sweets := A + M + J = 90

-- proof problem: How many sweets does June have?
theorem June_sweets_count : 
  May_sweets M J ∧ April_sweets A M ∧ Total_sweets A M J → J = 40 :=
by
  sorry

end June_sweets_count_l93_93515


namespace vartan_recreation_l93_93199

noncomputable def vartan_recreation_percent (W : ℝ) (P : ℝ) : Prop := 
  let W_this_week := 0.9 * W
  let recreation_last_week := (P / 100) * W
  let recreation_this_week := 0.3 * W_this_week
  recreation_this_week = 1.8 * recreation_last_week

theorem vartan_recreation (W : ℝ) : ∀ P : ℝ, vartan_recreation_percent W P → P = 15 := 
by
  intro P h
  unfold vartan_recreation_percent at h
  sorry

end vartan_recreation_l93_93199


namespace total_amount_shared_l93_93374

theorem total_amount_shared (X_share Y_share Z_share total_amount : ℝ) 
                            (h1 : Y_share = 0.45 * X_share) 
                            (h2 : Z_share = 0.50 * X_share) 
                            (h3 : Y_share = 45) : 
                            total_amount = X_share + Y_share + Z_share := 
by 
  -- Sorry to skip the proof
  sorry

end total_amount_shared_l93_93374


namespace sum_of_numbers_with_six_zeros_and_56_divisors_l93_93710

theorem sum_of_numbers_with_six_zeros_and_56_divisors :
  ∃ N1 N2 : ℕ, (N1 % 10^6 = 0) ∧ (N2 % 10^6 = 0) ∧ (N1_divisors = 56) ∧ (N2_divisors = 56) ∧ (N1 + N2 = 7000000) :=
by
  sorry

end sum_of_numbers_with_six_zeros_and_56_divisors_l93_93710


namespace lines_intersect_l93_93840

theorem lines_intersect (a b : ℝ) (h1 : 2 = (1/3) * 1 + a) (h2 : 1 = (1/2) * 2 + b) : a + b = 5 / 3 := 
by {
  -- Skipping the proof itself
  sorry
}

end lines_intersect_l93_93840


namespace find_2a_plus_b_l93_93719

theorem find_2a_plus_b (a b : ℝ) (h1 : 0 < a ∧ a < π / 2) (h2 : 0 < b ∧ b < π / 2)
  (h3 : 5 * (Real.sin a)^2 + 3 * (Real.sin b)^2 = 2)
  (h4 : 4 * Real.sin (2 * a) + 3 * Real.sin (2 * b) = 3) :
  2 * a + b = π / 2 :=
sorry

end find_2a_plus_b_l93_93719


namespace decompose_series_l93_93382

-- Define the 11-arithmetic Fibonacci sequence using the given series
def Φ₁₁₀ (n : ℕ) : ℕ :=
  if n % 11 = 0 then 0 else
  if n % 11 = 1 then 1 else
  if n % 11 = 2 then 1 else
  if n % 11 = 3 then 2 else
  if n % 11 = 4 then 3 else
  if n % 11 = 5 then 5 else
  if n % 11 = 6 then 8 else
  if n % 11 = 7 then 2 else
  if n % 11 = 8 then 10 else
  if n % 11 = 9 then 1 else
  0

-- Define the two geometric progressions
def G₁ (n : ℕ) : ℤ := 3 * (8 ^ n)
def G₂ (n : ℕ) : ℤ := 8 * (4 ^ n)

-- The decomposed sequence
def decomposedSequence (n : ℕ) : ℤ := G₁ n + G₂ n

-- The theorem to prove the decomposition
theorem decompose_series : ∀ n : ℕ, Φ₁₁₀ n = decomposedSequence n := by
  sorry

end decompose_series_l93_93382


namespace combined_savings_after_four_weeks_l93_93960

-- Definitions based on problem conditions
def hourly_wage : ℕ := 10
def daily_hours : ℕ := 10
def days_per_week : ℕ := 5
def weeks : ℕ := 4

def robby_saving_ratio : ℚ := 2/5
def jaylene_saving_ratio : ℚ := 3/5
def miranda_saving_ratio : ℚ := 1/2

-- Definitions derived from the conditions
def daily_earnings : ℕ := hourly_wage * daily_hours
def total_working_days : ℕ := days_per_week * weeks
def monthly_earnings : ℕ := daily_earnings * total_working_days

def robby_savings : ℚ := robby_saving_ratio * monthly_earnings
def jaylene_savings : ℚ := jaylene_saving_ratio * monthly_earnings
def miranda_savings : ℚ := miranda_saving_ratio * monthly_earnings

def total_savings : ℚ := robby_savings + jaylene_savings + miranda_savings

-- The main theorem to prove
theorem combined_savings_after_four_weeks :
  total_savings = 3000 := by sorry

end combined_savings_after_four_weeks_l93_93960


namespace ratio_of_line_cutting_median_lines_l93_93219

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem ratio_of_line_cutting_median_lines (A B C P Q : ℝ × ℝ) 
    (hA : A = (1, 0)) (hB : B = (0, 1)) (hC : C = (0, 0)) 
    (h_mid_AB : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) 
    (h_mid_BC : Q = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) 
    (h_ratio : (Real.sqrt (P.1^2 + P.2^2) / Real.sqrt (Q.1^2 + Q.2^2)) = (Real.sqrt (Q.1^2 + Q.2^2) / Real.sqrt (P.1^2 + P.2^2))) :
  (P.1 / Q.1) = golden_ratio :=
by 
  sorry

end ratio_of_line_cutting_median_lines_l93_93219


namespace line_through_point_parallel_to_y_axis_eq_x_eq_neg1_l93_93135

-- Define the point (M) and the properties of the line
def point_M : ℝ × ℝ := (-1, 3)

def parallel_to_y_axis (line : ℝ × ℝ → Prop) : Prop :=
  ∃ b : ℝ, ∀ y : ℝ, line (b, y)

-- Statement we need to prove
theorem line_through_point_parallel_to_y_axis_eq_x_eq_neg1 :
  (∃ line : ℝ × ℝ → Prop, line point_M ∧ parallel_to_y_axis line) → ∀ p : ℝ × ℝ, (p.1 = -1 ↔ (∃ line : ℝ × ℝ → Prop, line p ∧ line point_M ∧ parallel_to_y_axis line)) :=
by
  sorry

end line_through_point_parallel_to_y_axis_eq_x_eq_neg1_l93_93135


namespace curve_crosses_itself_and_point_of_crossing_l93_93023

-- Define the function for x and y
def x (t : ℝ) : ℝ := t^2 + 1
def y (t : ℝ) : ℝ := t^4 - 9 * t^2 + 6

-- Definition of the curve crossing itself and the point of crossing
theorem curve_crosses_itself_and_point_of_crossing :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ x t₁ = x t₂ ∧ y t₁ = y t₂ ∧ (x t₁ = 10 ∧ y t₁ = 6) :=
by
  sorry

end curve_crosses_itself_and_point_of_crossing_l93_93023


namespace supermarket_spent_more_than_collected_l93_93400

-- Given conditions
def initial_amount : ℕ := 53
def collected_amount : ℕ := 91
def amount_left : ℕ := 14

-- Finding the total amount before shopping and amount spent in supermarket
def total_amount : ℕ := initial_amount + collected_amount
def spent_amount : ℕ := total_amount - amount_left

-- Prove that the difference between spent amount and collected amount is 39
theorem supermarket_spent_more_than_collected : (spent_amount - collected_amount) = 39 := by
  -- The proof will go here
  sorry

end supermarket_spent_more_than_collected_l93_93400


namespace eight_faucets_fill_time_in_seconds_l93_93215

open Nat

-- Definitions under the conditions
def four_faucets_rate (gallons : ℕ) (minutes : ℕ) : ℕ := gallons / minutes

def one_faucet_rate (four_faucets_rate : ℕ) : ℕ := four_faucets_rate / 4

def eight_faucets_rate (one_faucet_rate : ℕ) : ℕ := one_faucet_rate * 8

def time_to_fill (rate : ℕ) (gallons : ℕ) : ℕ := gallons / rate

-- Main theorem to prove 
theorem eight_faucets_fill_time_in_seconds (gallons_tub : ℕ) (four_faucets_time : ℕ) :
    let four_faucets_rate := four_faucets_rate 200 8
    let one_faucet_rate := one_faucet_rate four_faucets_rate
    let rate_eight_faucets := eight_faucets_rate one_faucet_rate
    let time_fill := time_to_fill rate_eight_faucets 50
    gallons_tub = 50 ∧ four_faucets_time = 8 ∧ rate_eight_faucets = 50 -> time_fill * 60 = 60 :=
by
    intros
    sorry

end eight_faucets_fill_time_in_seconds_l93_93215


namespace fraction_simplify_l93_93601

theorem fraction_simplify (x : ℝ) (hx : x ≠ 1) (hx_ne_1 : x ≠ -1) :
  (x^2 - 1) / (x^2 - 2 * x + 1) = (x + 1) / (x - 1) :=
by
  sorry

end fraction_simplify_l93_93601


namespace water_addition_to_achieve_concentration_l93_93598

theorem water_addition_to_achieve_concentration :
  ∀ (w1 w2 : ℝ), 
  (60 * 0.25 = 15) →              -- initial amount of acid
  (15 / (60 + w1) = 0.15) →       -- first dilution to 15%
  (15 / (100 + w2) = 0.10) →      -- second dilution to 10%
  w1 + w2 = 90 :=                 -- total water added to achieve final concentration
by
  intros w1 w2 h_initial h_first h_second
  sorry

end water_addition_to_achieve_concentration_l93_93598


namespace number_add_thrice_number_eq_twenty_l93_93713

theorem number_add_thrice_number_eq_twenty (x : ℝ) (h : x + 3 * x = 20) : x = 5 :=
sorry

end number_add_thrice_number_eq_twenty_l93_93713


namespace value_of_k_l93_93328

theorem value_of_k (x y k : ℝ) (h1 : 4 * x - 3 * y = k) (h2 : 2 * x + 3 * y = 5) (h3 : x = y) : k = 1 :=
sorry

end value_of_k_l93_93328


namespace range_of_m_l93_93791

noncomputable def f (x m : ℝ) := x^2 - 2 * m * x + 4

def P (m : ℝ) : Prop := ∀ x, 2 ≤ x → f x m ≥ f (2 : ℝ) m
def Q (m : ℝ) : Prop := ∀ x, 4 * x^2 + 4 * (m - 2) * x + 1 > 0

theorem range_of_m (m : ℝ) : (P m ∨ Q m) ∧ ¬(P m ∧ Q m) ↔ m ≤ 1 ∨ (2 < m ∧ m < 3) := sorry

end range_of_m_l93_93791


namespace forming_n_and_m_l93_93432

def is_created_by_inserting_digit (n: ℕ) (base: ℕ): Prop :=
  ∃ d1 d2 d3 d: ℕ, n = d1 * 1000 + d * 100 + d2 * 10 + d3 ∧ base = d1 * 100 + d2 * 10 + d3

theorem forming_n_and_m (a b: ℕ) (base: ℕ) (sum: ℕ) 
  (h1: is_created_by_inserting_digit a base)
  (h2: is_created_by_inserting_digit b base) 
  (h3: a + b = sum):
  (a = 2195 ∧ b = 2165) 
  ∨ (a = 2185 ∧ b = 2175) 
  ∨ (a = 2215 ∧ b = 2145) 
  ∨ (a = 2165 ∧ b = 2195) 
  ∨ (a = 2175 ∧ b = 2185) 
  ∨ (a = 2145 ∧ b = 2215) := 
sorry

end forming_n_and_m_l93_93432


namespace fraction_equation_solution_l93_93758

theorem fraction_equation_solution (a : ℤ) (hpos : a > 0) (h : (a : ℝ) / (a + 50) = 0.870) : a = 335 :=
by {
  sorry
}

end fraction_equation_solution_l93_93758


namespace count_squares_with_center_55_25_l93_93913

noncomputable def number_of_squares_with_natural_number_coordinates : ℕ :=
  600

theorem count_squares_with_center_55_25 :
  ∀ (x y : ℕ), (x = 55) ∧ (y = 25) → number_of_squares_with_natural_number_coordinates = 600 :=
by
  intros x y h
  cases h
  sorry

end count_squares_with_center_55_25_l93_93913


namespace least_n_divisibility_condition_l93_93909

theorem least_n_divisibility_condition :
  ∃ n : ℕ, 0 < n ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (k ∣ (n^2 - n + 1) ↔ (n = 5 ∧ k = 3)) := 
sorry

end least_n_divisibility_condition_l93_93909


namespace find_initial_avg_height_l93_93148

noncomputable def initially_calculated_avg_height (A : ℚ) (boys : ℕ) (wrong_height right_height : ℚ) (actual_avg_height : ℚ) :=
  boys = 35 ∧
  wrong_height = 166 ∧
  right_height = 106 ∧
  actual_avg_height = 182 ∧
  35 * A - (wrong_height - right_height) = 35 * actual_avg_height

theorem find_initial_avg_height : ∃ A : ℚ, initially_calculated_avg_height A 35 166 106 182 ∧ A = 183.71 :=
by
  sorry

end find_initial_avg_height_l93_93148


namespace bill_spots_l93_93073

theorem bill_spots (b p : ℕ) (h1 : b + p = 59) (h2 : b = 2 * p - 1) : b = 39 := by
  sorry

end bill_spots_l93_93073


namespace proof_y_times_1_minus_g_eq_1_l93_93678
noncomputable def y : ℝ := (3 + Real.sqrt 8) ^ 100
noncomputable def m : ℤ := Int.floor y
noncomputable def g : ℝ := y - m

theorem proof_y_times_1_minus_g_eq_1 :
  y * (1 - g) = 1 := 
sorry

end proof_y_times_1_minus_g_eq_1_l93_93678


namespace orthocenter_of_triangle_ABC_l93_93123

def point : Type := ℝ × ℝ × ℝ

def A : point := (2, 3, 4)
def B : point := (6, 4, 2)
def C : point := (4, 5, 6)

def orthocenter (A B C : point) : point := sorry -- We'll skip the function implementation here

theorem orthocenter_of_triangle_ABC :
  orthocenter A B C = (13/7, 41/14, 55/7) :=
sorry

end orthocenter_of_triangle_ABC_l93_93123


namespace unique_solution_l93_93372

theorem unique_solution (p : ℕ) (a b n : ℕ) : 
  p.Prime → 2^a + p^b = n^(p-1) → (p, a, b, n) = (3, 0, 1, 2) ∨ (p = 2) :=
by {
  sorry
}

end unique_solution_l93_93372


namespace geometric_seq_increasing_l93_93167

theorem geometric_seq_increasing (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) → 
  (a 1 > a 0) = (∃ a1, (a1 > 0 ∧ q > 1) ∨ (a1 < 0 ∧ 0 < q ∧ q < 1)) :=
sorry

end geometric_seq_increasing_l93_93167


namespace total_number_of_boys_in_class_is_40_l93_93317

theorem total_number_of_boys_in_class_is_40 
  (n : ℕ) (h : 27 - 7 = n / 2):
  n = 40 :=
by
  sorry

end total_number_of_boys_in_class_is_40_l93_93317


namespace company_hired_22_additional_males_l93_93332

theorem company_hired_22_additional_males
  (E M : ℕ) 
  (initial_percentage_female : ℝ)
  (final_total_employees : ℕ)
  (final_percentage_female : ℝ)
  (initial_female_count : initial_percentage_female * E = 0.6 * E)
  (final_employee_count : E + M = 264) 
  (final_female_count : initial_percentage_female * E = final_percentage_female * (E + M)) :
  M = 22 := 
by
  sorry

end company_hired_22_additional_males_l93_93332


namespace find_acute_angles_of_alex_triangle_l93_93464

theorem find_acute_angles_of_alex_triangle (α : ℝ) (h1 : α > 0) (h2 : α < 90) :
  let condition1 := «Alex drew a geometric picture by tracing his plastic right triangle four times»
  let condition2 := «Each time aligning the shorter leg with the hypotenuse and matching the vertex of the acute angle with the vertex of the right angle»
  let condition3 := «The "closing" fifth triangle was isosceles»
  α = 90 / 11 :=
sorry

end find_acute_angles_of_alex_triangle_l93_93464


namespace jack_marathon_time_l93_93205

noncomputable def marathon_distance : ℝ := 42
noncomputable def jill_time : ℝ := 4.2
noncomputable def speed_ratio : ℝ := 0.7636363636363637

noncomputable def jill_speed : ℝ := marathon_distance / jill_time
noncomputable def jack_speed : ℝ := speed_ratio * jill_speed
noncomputable def jack_time : ℝ := marathon_distance / jack_speed

theorem jack_marathon_time : jack_time = 5.5 := sorry

end jack_marathon_time_l93_93205


namespace probability_of_six_being_largest_l93_93646

noncomputable def probability_six_is_largest : ℚ := sorry

theorem probability_of_six_being_largest (cards : Finset ℕ) (selected_cards : Finset ℕ) :
  cards = {1, 2, 3, 4, 5, 6, 7} →
  selected_cards ⊆ cards →
  selected_cards.card = 4 →
  (probability_six_is_largest = 2 / 7) := sorry

end probability_of_six_being_largest_l93_93646


namespace min_value_l93_93101

-- Definition of the conditions
def positive (a : ℝ) : Prop := a > 0

theorem min_value (a : ℝ) (h : positive a) : 
  ∃ m : ℝ, (m = 2 * Real.sqrt 6) ∧ (∀ x : ℝ, positive x → (3 / (2 * x) + 4 * x) ≥ m) :=
sorry

end min_value_l93_93101


namespace simplify_and_evaluate_l93_93969

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sin (Real.pi / 6)) :
  (1 - 2 / (x - 1)) / ((x - 3) / (x^2 - 1)) = 3 / 2 :=
by
  -- simplify and evaluate the expression given the condition on x
  sorry

end simplify_and_evaluate_l93_93969


namespace sum_of_solutions_l93_93298

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 2)^2 = 81) (h2 : (x2 - 2)^2 = 81) :
  x1 + x2 = 4 := by
  sorry

end sum_of_solutions_l93_93298


namespace banana_count_l93_93275

-- Variables representing the number of bananas, oranges, and apples
variables (B O A : ℕ)

-- Conditions translated from the problem statement
def conditions : Prop :=
  (O = 2 * B) ∧
  (A = 2 * O) ∧
  (B + O + A = 35)

-- Theorem to prove the number of bananas is 5 given the conditions
theorem banana_count (B O A : ℕ) (h : conditions B O A) : B = 5 :=
sorry

end banana_count_l93_93275


namespace minimum_ab_condition_l93_93985

open Int

theorem minimum_ab_condition 
  (a b : ℕ) 
  (h_pos : 0 < a ∧ 0 < b)
  (h_div7_ab_sum : ab * (a + b) % 7 ≠ 0) 
  (h_div7_expansion : ((a + b) ^ 7 - a ^ 7 - b ^ 7) % 7 = 0) : 
  ab = 18 :=
sorry

end minimum_ab_condition_l93_93985


namespace real_solution_exists_l93_93026

theorem real_solution_exists (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 3) :
  (x^3 - 4*x^2) / (x^2 - 5*x + 6) - x = 9 → x = 9/2 :=
by sorry

end real_solution_exists_l93_93026


namespace find_a_l93_93132

theorem find_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : x^2 - 2*a*x - 8*(a^2) < 0) (h3 : x2 - x1 = 15) : a = 5 / 2 :=
by
  -- Sorry is used to skip the actual proof.
  sorry

end find_a_l93_93132


namespace machine_produces_480_cans_in_8_hours_l93_93797

def cans_produced_in_interval : ℕ := 30
def interval_duration_minutes : ℕ := 30
def hours_worked : ℕ := 8
def minutes_in_hour : ℕ := 60

theorem machine_produces_480_cans_in_8_hours :
  (hours_worked * (minutes_in_hour / interval_duration_minutes) * cans_produced_in_interval) = 480 := by
  sorry

end machine_produces_480_cans_in_8_hours_l93_93797


namespace half_of_one_point_zero_one_l93_93612

theorem half_of_one_point_zero_one : (1.01 / 2) = 0.505 := 
by
  sorry

end half_of_one_point_zero_one_l93_93612


namespace sum_of_reciprocals_of_transformed_roots_l93_93654

theorem sum_of_reciprocals_of_transformed_roots :
  ∀ (a b c : ℂ), (a^3 - a + 1 = 0) → (b^3 - b + 1 = 0) → (c^3 - c + 1 = 0) → 
  (a ≠ b ∧ b ≠ c ∧ c ≠ a) →
  (1/(a+1) + 1/(b+1) + 1/(c+1) = -2) :=
by
  intros a b c ha hb hc habc
  sorry

end sum_of_reciprocals_of_transformed_roots_l93_93654


namespace cost_of_jeans_l93_93106

theorem cost_of_jeans 
  (price_socks : ℕ)
  (price_tshirt : ℕ)
  (price_jeans : ℕ)
  (h1 : price_socks = 5)
  (h2 : price_tshirt = price_socks + 10)
  (h3 : price_jeans = 2 * price_tshirt) :
  price_jeans = 30 :=
  by
    -- Sorry skips the proof, complies with the instructions
    sorry

end cost_of_jeans_l93_93106


namespace miranda_monthly_savings_l93_93972

noncomputable def total_cost := 260
noncomputable def sister_contribution := 50
noncomputable def months := 3

theorem miranda_monthly_savings : 
  (total_cost - sister_contribution) / months = 70 := 
by
  sorry

end miranda_monthly_savings_l93_93972


namespace rest_of_customers_bought_20_l93_93870

/-
Let's define the number of melons sold by the stand, number of customers who bought one and three melons, and total number of melons bought by these customers.
-/

def total_melons_sold : ℕ := 46
def customers_bought_one : ℕ := 17
def customers_bought_three : ℕ := 3

def melons_bought_by_those_bought_one := customers_bought_one * 1
def melons_bought_by_those_bought_three := customers_bought_three * 3

def remaining_melons := total_melons_sold - (melons_bought_by_those_bought_one + melons_bought_by_those_bought_three)

-- Now we state the theorem that the number of melons bought by the rest of the customers is 20 
theorem rest_of_customers_bought_20 :
  remaining_melons = 20 :=
by
  -- Skip the proof with 'sorry'
  sorry

end rest_of_customers_bought_20_l93_93870


namespace yellow_tint_percent_l93_93500

theorem yellow_tint_percent (total_volume: ℕ) (initial_yellow_percent: ℚ) (yellow_added: ℕ) (answer: ℚ) 
  (h_initial_total: total_volume = 20) 
  (h_initial_yellow: initial_yellow_percent = 0.50) 
  (h_yellow_added: yellow_added = 6) 
  (h_answer: answer = 61.5): 
  (yellow_added + initial_yellow_percent * total_volume) / (total_volume + yellow_added) * 100 = answer := 
by 
  sorry

end yellow_tint_percent_l93_93500


namespace exists_bijection_l93_93947

-- Define the non-negative integers set
def N_0 := {n : ℕ // n ≥ 0}

-- Translation of the equivalent proof statement into Lean
theorem exists_bijection (f : N_0 → N_0) :
  (∀ m n : N_0, f ⟨3 * m.val * n.val + m.val + n.val, sorry⟩ = 
   ⟨4 * (f m).val * (f n).val + (f m).val + (f n).val, sorry⟩) :=
sorry

end exists_bijection_l93_93947


namespace tables_count_is_correct_l93_93186

-- Definitions based on conditions
def invited_people : ℕ := 18
def people_didnt_show_up : ℕ := 12
def people_per_table : ℕ := 3

-- Calculation based on definitions
def people_attended : ℕ := invited_people - people_didnt_show_up
def tables_needed : ℕ := people_attended / people_per_table

-- The main theorem statement
theorem tables_count_is_correct : tables_needed = 2 := by
  unfold tables_needed
  unfold people_attended
  unfold invited_people
  unfold people_didnt_show_up
  unfold people_per_table
  sorry

end tables_count_is_correct_l93_93186


namespace john_blue_pens_l93_93865

variables (R B Bl : ℕ)

axiom total_pens : R + B + Bl = 31
axiom black_more_red : B = R + 5
axiom blue_twice_black : Bl = 2 * B

theorem john_blue_pens : Bl = 18 :=
by
  apply sorry

end john_blue_pens_l93_93865


namespace cookies_per_person_l93_93445

/-- Brenda's mother made cookies for 5 people. She prepared 35 cookies, 
    and each of them had the same number of cookies. 
    We aim to prove that each person had 7 cookies. --/
theorem cookies_per_person (total_cookies : ℕ) (number_of_people : ℕ) 
  (h1 : total_cookies = 35) (h2 : number_of_people = 5) : total_cookies / number_of_people = 7 := 
by
  sorry

end cookies_per_person_l93_93445


namespace cylinder_radius_l93_93299

theorem cylinder_radius
  (r₁ r₂ : ℝ)
  (rounds₁ rounds₂ : ℕ)
  (H₁ : r₁ = 14)
  (H₂ : rounds₁ = 70)
  (H₃ : rounds₂ = 49)
  (L₁ : rounds₁ * 2 * Real.pi * r₁ = rounds₂ * 2 * Real.pi * r₂) :
  r₂ = 20 := 
sorry

end cylinder_radius_l93_93299


namespace math_proof_problem_l93_93761

theorem math_proof_problem (a b : ℝ) (h1 : 64 = 8^2) (h2 : 16 = 8^2) :
  8^15 / (64^7) * 16 = 512 :=
by
  sorry

end math_proof_problem_l93_93761


namespace fg_of_3_l93_93667

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 + 2
def g (x : ℝ) : ℝ := 3 * x + 4

-- Theorem statement to prove f(g(3)) = 2199
theorem fg_of_3 : f (g 3) = 2199 :=
by
  sorry

end fg_of_3_l93_93667


namespace inequality_proof_l93_93587

theorem inequality_proof (a b c : ℝ) (h : a + b + c = 3) : 
  (1 / (5 * a^2 - 4 * a + 11)) + (1 / (5 * b^2 - 4 * b + 11)) + (1 / (5 * c^2 - 4 * c + 11)) ≤ 1 / 4 := 
by
  -- proof steps will be here
  sorry

end inequality_proof_l93_93587


namespace relationship_y1_y2_l93_93156

variables {x1 x2 : ℝ}

noncomputable def f (x : ℝ) : ℝ := -3 * x ^ 2 + 6 * x - 5

theorem relationship_y1_y2 (hx1 : 0 ≤ x1) (hx1_lt : x1 < 1) (hx2 : 2 ≤ x2) (hx2_lt : x2 < 3) :
  f x1 ≥ f x2 :=
sorry

end relationship_y1_y2_l93_93156


namespace sum_of_decimals_l93_93266

theorem sum_of_decimals : (5.76 + 4.29 = 10.05) :=
by
  sorry

end sum_of_decimals_l93_93266


namespace multiple_of_age_is_3_l93_93542

def current_age : ℕ := 9
def age_six_years_ago : ℕ := 3
def age_multiple (current : ℕ) (previous : ℕ) : ℕ := current / previous

theorem multiple_of_age_is_3 : age_multiple current_age age_six_years_ago = 3 :=
by
  sorry

end multiple_of_age_is_3_l93_93542


namespace geometric_sequence_S4_l93_93000

/-
In the geometric sequence {a_n}, S_2 = 7, S_6 = 91. Prove that S_4 = 28.
-/

theorem geometric_sequence_S4 (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n * q)
  (h_sum : ∀ n, S n = a 1 * (1 - q^n) / (1 - q))
  (h_S2 : S 2 = 7) 
  (h_S6 : S 6 = 91) :
  S 4 = 28 := 
sorry

end geometric_sequence_S4_l93_93000


namespace david_marks_in_physics_l93_93970

theorem david_marks_in_physics 
  (english_marks mathematics_marks chemistry_marks biology_marks : ℕ)
  (num_subjects : ℕ)
  (average_marks : ℕ)
  (h1 : english_marks = 81)
  (h2 : mathematics_marks = 65)
  (h3 : chemistry_marks = 67)
  (h4 : biology_marks = 85)
  (h5 : num_subjects = 5)
  (h6 : average_marks = 76) :
  ∃ physics_marks : ℕ, physics_marks = 82 :=
by
  sorry

end david_marks_in_physics_l93_93970


namespace smallest_q_p_difference_l93_93387

theorem smallest_q_p_difference :
  ∃ (p q : ℕ), 
  (3 : ℚ) / 5 < p / q ∧ p / q < (5 : ℚ) / 8 ∧
  ∀ (r : ℕ), (3 : ℚ) / 5 < r / q ∧ r / q < (5 : ℚ) / 8 → p = r ∧ q = 13 →
  q - p = 5 :=
by {
  -- proof goes here
  sorry
}

end smallest_q_p_difference_l93_93387


namespace decreased_cost_l93_93938

theorem decreased_cost (original_cost : ℝ) (decrease_percentage : ℝ) (h1 : original_cost = 200) (h2 : decrease_percentage = 0.50) : 
  (original_cost - original_cost * decrease_percentage) = 100 :=
by
  -- This is the proof placeholder
  sorry

end decreased_cost_l93_93938


namespace repeating_decimal_sum_l93_93796

-- Definitions based on conditions
def x := 5 / 9  -- We derived this from 0.5 repeating as a fraction
def y := 7 / 99  -- Similarly, derived from 0.07 repeating as a fraction

-- Proposition to prove
theorem repeating_decimal_sum : x + y = 62 / 99 := by
  sorry

end repeating_decimal_sum_l93_93796


namespace number_of_participants_2005_l93_93191

variable (participants : ℕ → ℕ)
variable (n : ℕ)

-- Conditions
def initial_participants := participants 2001 = 1000
def increase_till_2003 := ∀ n, 2001 ≤ n ∧ n ≤ 2003 → participants (n + 1) = 2 * participants n
def increase_from_2004 := ∀ n, n ≥ 2004 → participants (n + 1) = 2 * participants n + 500

-- Proof problem
theorem number_of_participants_2005 :
    initial_participants participants →
    increase_till_2003 participants →
    increase_from_2004 participants →
    participants 2005 = 17500 :=
by sorry

end number_of_participants_2005_l93_93191


namespace money_distribution_l93_93583

theorem money_distribution (Maggie_share : ℝ) (fraction_Maggie : ℝ) (total_sum : ℝ) :
  Maggie_share = 7500 →
  fraction_Maggie = (1/8) →
  total_sum = Maggie_share / fraction_Maggie →
  total_sum = 60000 :=
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  linarith

end money_distribution_l93_93583


namespace average_speed_round_trip_l93_93627

theorem average_speed_round_trip
  (n : ℕ)
  (distance_km : ℝ := n / 1000)
  (pace_west_min_per_km : ℝ := 2)
  (speed_east_kmh : ℝ := 3)
  (wait_time_hr : ℝ := 30 / 60) :
  (2 * distance_km) / 
  ((pace_west_min_per_km * distance_km / 60) + wait_time_hr + (distance_km / speed_east_kmh)) = 
  60 * n / (11 * n + 150000) := by
  sorry

end average_speed_round_trip_l93_93627


namespace greatest_divisor_of_product_of_four_consecutive_integers_l93_93488

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l93_93488


namespace find_integer_n_l93_93361

theorem find_integer_n :
  ∃ n : ℤ, 
    50 ≤ n ∧ n ≤ 120 ∧ (n % 5 = 0) ∧ (n % 6 = 3) ∧ (n % 7 = 4) ∧ n = 165 :=
by
  sorry

end find_integer_n_l93_93361


namespace value_of_b_prod_l93_93049

-- Conditions
def a (n : ℕ) : ℕ := 2 * n - 1

def b (n : ℕ) : ℕ := 2 ^ (n - 1)

-- The goal is to prove that b_{a_1} * b_{a_3} * b_{a_5} = 4096
theorem value_of_b_prod : b (a 1) * b (a 3) * b (a 5) = 4096 := by
  sorry

end value_of_b_prod_l93_93049


namespace middle_number_between_52_and_certain_number_l93_93157

theorem middle_number_between_52_and_certain_number :
  ∃ n, n > 52 ∧ (∀ k, 52 ≤ k ∧ k ≤ n → ∃ l, k = 52 + l) ∧ (n = 52 + 16) :=
sorry

end middle_number_between_52_and_certain_number_l93_93157


namespace solve_for_r_l93_93525

theorem solve_for_r (r s : ℚ) (h : (2 * (r - 45)) / 3 = (3 * s - 2 * r) / 4) (s_val : s = 20) :
  r = 270 / 7 :=
by
  sorry

end solve_for_r_l93_93525


namespace score_difference_l93_93218

noncomputable def mean_score (scores pcts : List ℕ) : ℚ := 
  (List.zipWith (· * ·) scores pcts).sum / 100

def median_score (scores pcts : List ℕ) : ℚ := 75

theorem score_difference :
  let scores := [60, 75, 85, 95]
  let pcts := [20, 50, 15, 15]
  abs (median_score scores pcts - mean_score scores pcts) = 1.5 := by
  sorry

end score_difference_l93_93218


namespace Sunzi_problem_correctness_l93_93254

theorem Sunzi_problem_correctness (x y : ℕ) :
  3 * (x - 2) = 2 * x + 9 ∧ (y / 3) + 2 = (y - 9) / 2 :=
by
  sorry

end Sunzi_problem_correctness_l93_93254


namespace Nadia_distance_is_18_l93_93480

-- Variables and conditions
variables (x : ℕ)

-- Definitions based on conditions
def Hannah_walked (x : ℕ) : ℕ := x
def Nadia_walked (x : ℕ) : ℕ := 2 * x
def total_distance (x : ℕ) : ℕ := Hannah_walked x + Nadia_walked x

-- The proof statement
theorem Nadia_distance_is_18 (h : total_distance x = 27) : Nadia_walked x = 18 :=
by
  sorry

end Nadia_distance_is_18_l93_93480


namespace wall_area_in_square_meters_l93_93790

variable {W H : ℤ} -- We treat W and H as integers referring to centimeters

theorem wall_area_in_square_meters 
  (h₁ : W / 30 = 8) 
  (h₂ : H / 30 = 5) : 
  (W / 100) * (H / 100) = 360 / 100 :=
by 
  sorry

end wall_area_in_square_meters_l93_93790


namespace jay_more_points_than_tobee_l93_93818

-- Declare variables.
variables (x J S : ℕ)

-- Given conditions
def Tobee_points := 4
def Jay_points := Tobee_points + x -- Jay_score is 4 + x
def Sean_points := (Tobee_points + Jay_points) - 2 -- Sean_score is 4 + Jay - 2

-- The total score condition
def total_score_condition := Tobee_points + Jay_points + Sean_points = 26

-- The main statement to be proven
theorem jay_more_points_than_tobee (h : total_score_condition) : J - Tobee_points = 6 :=
sorry

end jay_more_points_than_tobee_l93_93818


namespace find_n_arithmetic_sequence_l93_93059

-- Given conditions
def a₁ : ℕ := 20
def aₙ : ℕ := 54
def Sₙ : ℕ := 999

-- Arithmetic sequence sum formula and proof statement of n = 27
theorem find_n_arithmetic_sequence
  (a₁ : ℕ)
  (aₙ : ℕ)
  (Sₙ : ℕ)
  (h₁ : a₁ = 20)
  (h₂ : aₙ = 54)
  (h₃ : Sₙ = 999) : ∃ n : ℕ, n = 27 := 
by
  sorry

end find_n_arithmetic_sequence_l93_93059


namespace range_of_a_l93_93932

noncomputable def f (x a : ℝ) := Real.log x + 1 / 2 * x^2 + a * x

theorem range_of_a
  (a : ℝ)
  (h : ∃ x : ℝ, x > 0 ∧ (1/x + x + a = 3)) :
  a ≤ 1 :=
by
  sorry

end range_of_a_l93_93932


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l93_93280

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.45
  let a := 9 -- GCD of 45 and 99
  let numerator := 5
  let denominator := 11
  numerator + denominator = 16 :=
by { 
  sorry 
}

end sum_of_numerator_and_denominator_of_repeating_decimal_l93_93280


namespace scientific_notation_14000000_l93_93763

theorem scientific_notation_14000000 :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 14000000 = a * 10 ^ n ∧ a = 1.4 ∧ n = 7 :=
by
  sorry

end scientific_notation_14000000_l93_93763


namespace infinite_series_value_l93_93809

noncomputable def sum_infinite_series : ℝ := ∑' n : ℕ, if n > 0 then 1 / (n * (n + 3)) else 0

theorem infinite_series_value :
  sum_infinite_series = 11 / 18 :=
sorry

end infinite_series_value_l93_93809


namespace proof_problem_l93_93826

def RealSets (A B : Set ℝ) : Set ℝ :=
let complementA := {x | -2 < x ∧ x < 3}
let unionAB := complementA ∪ B
unionAB

theorem proof_problem :
  let A := {x : ℝ | (x + 2) * (x - 3) ≥ 0}
  let B := {x : ℝ | x > 1}
  let complementA := {x : ℝ | -2 < x ∧ x < 3}
  let unionAB := complementA ∪ B
  unionAB = {x : ℝ | x > -2} :=
by
  sorry

end proof_problem_l93_93826


namespace piecewise_function_identity_l93_93062

theorem piecewise_function_identity (x : ℝ) : 
  (3 * x + abs (5 * x - 10)) = if x < 2 then -2 * x + 10 else 8 * x - 10 := by
  sorry

end piecewise_function_identity_l93_93062


namespace math_proof_problems_l93_93071

open Real

noncomputable def problem1 (α : ℝ) : Prop :=
  (sin (π - α) - 2 * sin (π / 2 + α) = 0) → (sin α * cos α + sin α ^ 2 = 6 / 5)

noncomputable def problem2 (α β : ℝ) : Prop :=
  (tan (α + β) = -1) → (tan α = 2) → (tan β = 3)

-- Example of how to state these problems as a theorem
theorem math_proof_problems (α β : ℝ) : problem1 α ∧ problem2 α β := by
  sorry

end math_proof_problems_l93_93071


namespace sin_pi_minus_alpha_cos_2pi_minus_alpha_sin_minus_cos_l93_93600

-- Problem 1: Given that tan(α) = 3, prove that sin(π - α) * cos(2π - α) = 3 / 10.
theorem sin_pi_minus_alpha_cos_2pi_minus_alpha (α : ℝ) (h : Real.tan α = 3) : 
  Real.sin (Real.pi - α) * Real.cos (2 * Real.pi - α) = 3 / 10 :=
by
  sorry

-- Problem 2: Given that sin(α) * cos(α) = 1/4 and 0 < α < π/4, prove that sin(α) - cos(α) = - sqrt(2) / 2.
theorem sin_minus_cos (α : ℝ) (h₁ : Real.sin α * Real.cos α = 1 / 4) (h₂ : 0 < α) (h₃ : α < Real.pi / 4) :
  Real.sin α - Real.cos α = - (Real.sqrt 2) / 2 :=
by
  sorry

end sin_pi_minus_alpha_cos_2pi_minus_alpha_sin_minus_cos_l93_93600


namespace seven_thirteenths_of_3940_percent_25000_l93_93182

noncomputable def seven_thirteenths (x : ℝ) : ℝ := (7 / 13) * x

noncomputable def percent (part whole : ℝ) : ℝ := (part / whole) * 100

theorem seven_thirteenths_of_3940_percent_25000 :
  percent (seven_thirteenths 3940) 25000 = 8.484 :=
by
  sorry

end seven_thirteenths_of_3940_percent_25000_l93_93182


namespace necessary_but_not_sufficient_condition_l93_93175

variable {M N P : Set α}

theorem necessary_but_not_sufficient_condition (h : M ∩ P = N ∩ P) : 
  (M = N) → (M ∩ P = N ∩ P) :=
sorry

end necessary_but_not_sufficient_condition_l93_93175


namespace maximize_profit_marginal_profit_monotonic_decreasing_l93_93688

-- Definition of revenue function R
def R (x : ℕ) : ℤ := 3700 * x + 45 * x^2 - 10 * x^3

-- Definition of cost function C
def C (x : ℕ) : ℤ := 460 * x + 500

-- Definition of profit function p
def p (x : ℕ) : ℤ := R x - C x

-- Lemma for the solution
theorem maximize_profit (x : ℕ) (h1 : 1 ≤ x ∧ x ≤ 20) : 
  p x = -10 * x^3 + 45 * x^2 + 3240 * x - 500 ∧ 
  (∀ y, 1 ≤ y ∧ y ≤ 20 → p y ≤ p 12) :=
by
  sorry

-- Definition of marginal profit function Mp
def Mp (x : ℕ) : ℤ := p (x + 1) - p x

-- Lemma showing Mp is monotonically decreasing
theorem marginal_profit_monotonic_decreasing (x : ℕ) (h2 : 1 ≤ x ∧ x ≤ 19) : 
  Mp x = -30 * x^2 + 60 * x + 3275 ∧ 
  ∀ y, 1 ≤ y ∧ y ≤ 19 → (Mp y ≥ Mp (y + 1)) :=
by
  sorry

end maximize_profit_marginal_profit_monotonic_decreasing_l93_93688


namespace balance_squares_circles_l93_93589

theorem balance_squares_circles (x y z : ℕ) (h1 : 5 * x + 2 * y = 21 * z) (h2 : 2 * x = y + 3 * z) : 
  3 * y = 9 * z :=
by 
  sorry

end balance_squares_circles_l93_93589


namespace integer_points_between_A_B_l93_93114

/-- 
Prove that the number of integer coordinate points strictly between 
A(2, 3) and B(50, 80) on the line passing through A and B is c.
-/
theorem integer_points_between_A_B 
  (A B : ℤ × ℤ) (hA : A = (2, 3)) (hB : B = (50, 80)) 
  (c : ℕ) :
  ∃ (n : ℕ), n = c ∧ ∀ (x y : ℤ), (A.1 < x ∧ x < B.1) → (A.2 < y ∧ y < B.2) → 
              (y = ((A.2 - B.2) / (A.1 - B.1) * x + 3 - (A.2 - B.2) / (A.1 - B.1) * 2)) :=
by {
  sorry
}

end integer_points_between_A_B_l93_93114


namespace find_x_value_l93_93556

theorem find_x_value (x : ℝ) (h : (7 / (x - 2) + x / (2 - x) = 4)) : x = 3 :=
sorry

end find_x_value_l93_93556


namespace rectangle_ratio_l93_93029

/-- Conditions:
1. There are three identical squares and two rectangles forming a large square.
2. Each rectangle shares one side with a square and another side with the edge of the large square.
3. The side length of each square is 1 unit.
4. The total side length of the large square is 5 units.
Question:
What is the ratio of the length to the width of one of the rectangles? --/

theorem rectangle_ratio (sq_len : ℝ) (large_sq_len : ℝ) (side_ratio : ℝ) :
  sq_len = 1 ∧ large_sq_len = 5 ∧ 
  (∀ (rect_len rect_wid : ℝ), 3 * sq_len + 2 * rect_len = large_sq_len ∧ side_ratio = rect_len / rect_wid) →
  side_ratio = 1 / 2 :=
by
  sorry

end rectangle_ratio_l93_93029


namespace num_perpendicular_line_plane_pairs_in_cube_l93_93507

-- Definitions based on the problem conditions

def is_perpendicular_line_plane_pair (l : line) (p : plane) : Prop :=
  -- Assume an implementation that defines when a line is perpendicular to a plane
  sorry

-- Define a cube structure with its vertices, edges, and faces
structure Cube :=
  (vertices : Finset Point)
  (edges : Finset (Point × Point))
  (faces : Finset (Finset Point))

-- Make assumptions about cube properties
variable (cube : Cube)

-- Define the property of counting perpendicular line-plane pairs
def count_perpendicular_line_plane_pairs (c : Cube) : Nat :=
  -- Assume an implementation that counts the number of such pairs in the cube
  sorry

-- The theorem to prove
theorem num_perpendicular_line_plane_pairs_in_cube (c : Cube) :
  count_perpendicular_line_plane_pairs c = 36 :=
  sorry

end num_perpendicular_line_plane_pairs_in_cube_l93_93507


namespace minimum_value_inverse_sum_l93_93905

variables {m n : ℝ}

theorem minimum_value_inverse_sum 
  (hm : m > 0) 
  (hn : n > 0) 
  (hline : ∀ x y : ℝ, m * x + n * y + 2 = 0 → (x + 3)^2 + (y + 1)^2 = 1)
  (hchord : ∀ x1 y1 x2 y2 : ℝ, m * x1 + n * y1 + 2 = 0 ∧ m * x2 + n * y2 + 2 = 0 → 
    (x1 - x2)^2 + (y1 - y2)^2 = 4) : 
  ∃ m n : ℝ, 3 * m + n = 2 ∧ m > 0 ∧ n > 0 ∧ 
    (∀ m' n' : ℝ, 3 * m' + n' = 2 → m' > 0 → n' > 0 → 
      (1 / m' + 3 / n' ≥ 6)) :=
sorry

end minimum_value_inverse_sum_l93_93905


namespace zeta_1_8_add_zeta_2_8_add_zeta_3_8_l93_93333

noncomputable def compute_s8 (s : ℕ → ℂ) : ℂ :=
  s 8

theorem zeta_1_8_add_zeta_2_8_add_zeta_3_8 {ζ : ℕ → ℂ} 
  (h1 : ζ 1 + ζ 2 + ζ 3 = 2)
  (h2 : ζ 1^2 + ζ 2^2 + ζ 3^2 = 6)
  (h3 : ζ 1^3 + ζ 2^3 + ζ 3^3 = 18)
  (rec : ∀ n, ζ (n + 3) = 2 * ζ (n + 2) + ζ (n + 1) - (4 / 3) * ζ n)
  (s0 : ζ 0 = 3)
  (s1 : ζ 1 = 2)
  (s2 : ζ 2 = 6)
  (s3 : ζ 3 = 18)
  : ζ 8 = compute_s8 ζ := 
sorry

end zeta_1_8_add_zeta_2_8_add_zeta_3_8_l93_93333


namespace cookie_ratio_l93_93602

theorem cookie_ratio (K : ℕ) (h1 : K / 2 + K + 24 = 33) : 24 / K = 4 :=
by {
  sorry
}

end cookie_ratio_l93_93602


namespace tan_alpha_minus_pi_over_4_l93_93119

theorem tan_alpha_minus_pi_over_4 (α : ℝ) (h1 : 0 < α ∧ α < π)
  (h2 : Real.sin α = 3 / 5) : Real.tan (α - π / 4) = -1 / 7 ∨ Real.tan (α - π / 4) = -7 := 
sorry

end tan_alpha_minus_pi_over_4_l93_93119


namespace sweatshirt_cost_l93_93712

/--
Hannah bought 3 sweatshirts and 2 T-shirts.
Each T-shirt cost $10.
Hannah spent $65 in total.
Prove that the cost of each sweatshirt is $15.
-/
theorem sweatshirt_cost (S : ℝ) (h1 : 3 * S + 2 * 10 = 65) : S = 15 :=
by
  sorry

end sweatshirt_cost_l93_93712


namespace min_value_of_fraction_l93_93245

theorem min_value_of_fraction (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (4 / (a + 2) + 1 / (b + 1)) = 9 / 4 :=
sorry

end min_value_of_fraction_l93_93245


namespace total_cost_of_tickets_l93_93227

theorem total_cost_of_tickets (num_family_members num_adult_tickets num_children_tickets : ℕ)
    (cost_adult_ticket cost_children_ticket total_cost : ℝ) 
    (h1 : num_family_members = 7) 
    (h2 : cost_adult_ticket = 21) 
    (h3 : cost_children_ticket = 14) 
    (h4 : num_adult_tickets = 4) 
    (h5 : num_children_tickets = num_family_members - num_adult_tickets) 
    (h6 : total_cost = num_adult_tickets * cost_adult_ticket + num_children_tickets * cost_children_ticket) :
    total_cost = 126 :=
by
  sorry

end total_cost_of_tickets_l93_93227


namespace value_of_a_plus_b_l93_93225

def f (x : ℝ) (a b : ℝ) := x^3 + (a - 1) * x^2 + a * x + b

theorem value_of_a_plus_b (a b : ℝ) :
  (∀ x : ℝ, f (-x) a b = -f x a b) → a + b = 1 :=
by
  sorry

end value_of_a_plus_b_l93_93225


namespace people_joined_group_l93_93572

theorem people_joined_group (x y : ℕ) (h1 : 1430 = 22 * x) (h2 : 1430 = 13 * (x + y)) : y = 45 := 
by 
  -- This is just the statement, so we add sorry to skip the proof
  sorry

end people_joined_group_l93_93572


namespace calculate_expression_l93_93867

theorem calculate_expression : 5^3 + 5^3 + 5^3 + 5^3 = 625 :=
  sorry

end calculate_expression_l93_93867


namespace simplify_expression_l93_93130

variable (x : ℝ)

theorem simplify_expression :
  (2 * x * (4 * x^2 - 3) - 4 * (x^2 - 3 * x + 6)) = (8 * x^3 - 4 * x^2 + 6 * x - 24) := 
by 
  sorry

end simplify_expression_l93_93130


namespace daily_evaporation_l93_93624

theorem daily_evaporation (initial_water: ℝ) (days: ℝ) (evap_percentage: ℝ) : 
  initial_water = 10 → days = 50 → evap_percentage = 2 →
  (initial_water * evap_percentage / 100) / days = 0.04 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end daily_evaporation_l93_93624


namespace solve_for_k_l93_93108

theorem solve_for_k (x k : ℝ) (h : k ≠ 0) 
(h_eq : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 7)) : k = 7 :=
by
  -- Proof would go here
  sorry

end solve_for_k_l93_93108


namespace find_n_value_l93_93222

theorem find_n_value :
  ∃ m n : ℝ, (4 * x^2 + 8 * x - 448 = 0 → (x + m)^2 = n) ∧ n = 113 :=
by
  sorry

end find_n_value_l93_93222


namespace family_members_count_l93_93235

variable (F : ℕ) -- Number of other family members

def annual_cost_per_person : ℕ := 4000 + 12 * 1000
def john_total_cost_for_family (F : ℕ) : ℕ := (F + 1) * annual_cost_per_person / 2

theorem family_members_count :
  john_total_cost_for_family F = 32000 → F = 3 := by
  sorry

end family_members_count_l93_93235


namespace average_additional_minutes_per_day_l93_93303

def daily_differences : List ℤ := [20, 5, -5, 0, 15, -10, 10]

theorem average_additional_minutes_per_day :
  (List.sum daily_differences / daily_differences.length) = 5 := by
  sorry

end average_additional_minutes_per_day_l93_93303


namespace bushels_needed_l93_93080

theorem bushels_needed (cows sheep chickens : ℕ) (cows_eat sheep_eat chickens_eat : ℕ) :
  cows = 4 → cows_eat = 2 →
  sheep = 3 → sheep_eat = 2 →
  chickens = 7 → chickens_eat = 3 →
  4 * 2 + 3 * 2 + 7 * 3 = 35 := 
by
  intros hc hec hs hes hch hech
  sorry

end bushels_needed_l93_93080


namespace product_modulo_seven_l93_93013

theorem product_modulo_seven (a b c d : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3)
(h3 : c % 7 = 4) (h4 : d % 7 = 5) : (a * b * c * d) % 7 = 1 := 
sorry

end product_modulo_seven_l93_93013


namespace zachary_needs_more_money_l93_93003

def cost_of_football : ℝ := 3.75
def cost_of_shorts : ℝ := 2.40
def cost_of_shoes : ℝ := 11.85
def zachary_money : ℝ := 10.00
def total_cost : ℝ := cost_of_football + cost_of_shorts + cost_of_shoes
def amount_needed : ℝ := total_cost - zachary_money

theorem zachary_needs_more_money : amount_needed = 7.00 := by
  sorry

end zachary_needs_more_money_l93_93003


namespace contradiction_in_triangle_l93_93562

theorem contradiction_in_triangle (A B C : ℝ) (hA : A > 60) (hB : B > 60) (hC : C > 60) (sum_angles : A + B + C = 180) : false :=
by
  sorry

end contradiction_in_triangle_l93_93562


namespace prime_difference_condition_l93_93163

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_difference_condition :
  ∃ (x y : ℕ), is_prime x ∧ is_prime y ∧ 4 < x ∧ x < 18 ∧ 4 < y ∧ y < 18 ∧ x ≠ y ∧ (x * y - (x + y)) = 119 :=
by
  sorry

end prime_difference_condition_l93_93163


namespace calculate_expression_l93_93405

theorem calculate_expression : 2 * (-3)^3 - 4 * (-3) + 15 = -27 := 
by
  sorry

end calculate_expression_l93_93405


namespace repeated_1991_mod_13_l93_93917

theorem repeated_1991_mod_13 (k : ℕ) : 
  ((10^4 - 9) * (1991 * (10^(4*k) - 1)) / 9) % 13 = 8 :=
by
  sorry

end repeated_1991_mod_13_l93_93917


namespace find_f6_l93_93394

variable {R : Type} [LinearOrderedField R]

def f : R → R := sorry

theorem find_f6 (h1 : ∀ x y : R, f (x - y) = f x * f y) (h2 : ∀ x : R, f x ≠ 0) : f 6 = 1 :=
sorry

end find_f6_l93_93394


namespace davi_minimum_spending_l93_93769

-- Define the cost of a single bottle
def singleBottleCost : ℝ := 2.80

-- Define the cost of a box of six bottles
def boxCost : ℝ := 15.00

-- Define the number of bottles Davi needs to buy
def totalBottles : ℕ := 22

-- Calculate the minimum amount Davi will spend
def minimumCost : ℝ := 45.00 + 11.20 

-- The theorem to prove
theorem davi_minimum_spending :
  ∃ minCost : ℝ, minCost = 56.20 ∧ minCost = 3 * boxCost + 4 * singleBottleCost := 
by
  use 56.20
  sorry

end davi_minimum_spending_l93_93769


namespace simplify_trig_expression_l93_93425

open Real

theorem simplify_trig_expression (theta : ℝ) (h : 0 < theta ∧ theta < π / 4) :
  sqrt (1 - 2 * sin (π + theta) * sin (3 * π / 2 - theta)) = cos theta - sin theta :=
sorry

end simplify_trig_expression_l93_93425


namespace ferry_time_increases_l93_93838

noncomputable def ferryRoundTrip (S V x : ℝ) : ℝ :=
  (S / (V + x)) + (S / (V - x))

theorem ferry_time_increases (S V x : ℝ) (h_V_pos : 0 < V) (h_x_lt_V : x < V) :
  ferryRoundTrip S V (x + 1) > ferryRoundTrip S V x :=
by
  sorry

end ferry_time_increases_l93_93838


namespace tree_placement_impossible_l93_93267

theorem tree_placement_impossible
  (length width : ℝ) (h_length : length = 4) (h_width : width = 1) :
  ¬ (∃ (t1 t2 t3 : ℝ × ℝ), 
       dist t1 t2 ≥ 2.5 ∧ 
       dist t2 t3 ≥ 2.5 ∧ 
       dist t1 t3 ≥ 2.5 ∧ 
       t1.1 ≥ 0 ∧ t1.1 ≤ length ∧ t1.2 ≥ 0 ∧ t1.2 ≤ width ∧ 
       t2.1 ≥ 0 ∧ t2.1 ≤ length ∧ t2.2 ≥ 0 ∧ t2.2 ≤ width ∧ 
       t3.1 ≥ 0 ∧ t3.1 ≤ length ∧ t3.2 ≥ 0 ∧ t3.2 ≤ width) := 
by {
  sorry
}

end tree_placement_impossible_l93_93267


namespace max_b_value_l93_93977

theorem max_b_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a * b = (2 * a - b) / (2 * a + 3 * b)) : b ≤ 1 / 3 :=
  sorry

end max_b_value_l93_93977


namespace maximize_garden_area_l93_93520

def optimal_dimensions_area : Prop :=
  let l := 100
  let w := 60
  let area := 6000
  (2 * l) + (2 * w) = 320 ∧ l >= 100 ∧ (l * w) = area

theorem maximize_garden_area : optimal_dimensions_area := by
  sorry

end maximize_garden_area_l93_93520


namespace a_2016_is_1_l93_93544

noncomputable def seq_a (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * b n

theorem a_2016_is_1 (a b : ℕ → ℝ)
  (h1 : a 1 = 1)
  (hb : seq_a a b)
  (h3 : b 1008 = 1) :
  a 2016 = 1 :=
sorry

end a_2016_is_1_l93_93544


namespace trigonometric_identity_l93_93786

open Real

-- Lean 4 statement
theorem trigonometric_identity (α β γ x : ℝ) :
  (sin (x - β) * sin (x - γ) / (sin (α - β) * sin (α - γ))) +
  (sin (x - γ) * sin (x - α) / (sin (β - γ) * sin (β - α))) +
  (sin (x - α) * sin (x - β) / (sin (γ - α) * sin (γ - β))) = 1 := 
sorry

end trigonometric_identity_l93_93786


namespace sum_seq_equals_2_pow_n_minus_1_l93_93434

-- Define the sequences a_n and b_n with given conditions
def a (n : ℕ) : ℕ := if n = 0 then 2 else if n = 1 then 4 else sorry
def b (n : ℕ) : ℕ := if n = 0 then 2 else if n = 1 then 4 else sorry

-- Relation for a_n: 2a_{n+1} = a_n + a_{n+2}
axiom a_relation (n : ℕ) : 2 * a (n + 1) = a n + a (n + 2)

-- Inequalities for b_n
axiom b_inequality_1 (n : ℕ) : b (n + 1) - b n < 2^n + 1 / 2
axiom b_inequality_2 (n : ℕ) : b (n + 2) - b n > 3 * 2^n - 1

-- Note that b_n ∈ ℤ is implied by the definition being in ℕ

-- Prove that the sum of the first n terms of the sequence { n * b_n / a_n }
theorem sum_seq_equals_2_pow_n_minus_1 (n : ℕ) : 
  (Finset.range n).sum (λ k => k * b k / a k) = 2^n - 1 := 
sorry

end sum_seq_equals_2_pow_n_minus_1_l93_93434


namespace heather_oranges_l93_93967

theorem heather_oranges (initial_oranges additional_oranges : ℝ) (h1 : initial_oranges = 60.5) (h2 : additional_oranges = 35.8) :
  initial_oranges + additional_oranges = 96.3 :=
by
  -- sorry is used to indicate the proof is omitted
  sorry

end heather_oranges_l93_93967


namespace integer_between_sqrt3_add1_and_sqrt11_l93_93944

theorem integer_between_sqrt3_add1_and_sqrt11 :
  (∀ x, (1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2) ∧ (3 < Real.sqrt 11 ∧ Real.sqrt 11 < 4) → (2 < Real.sqrt 3 + 1 ∧ Real.sqrt 3 + 1 < 3) ∧ (3 < Real.sqrt 11 ∧ Real.sqrt 11 < 4) ∧ x = 3) :=
by
  sorry

end integer_between_sqrt3_add1_and_sqrt11_l93_93944


namespace inequality_smallest_integer_solution_l93_93078

theorem inequality_smallest_integer_solution (x : ℤ) :
    (9 * x + 8) / 6 - x / 3 ≥ -1 → x ≥ -2 := sorry

end inequality_smallest_integer_solution_l93_93078


namespace total_coins_l93_93428

-- Defining the conditions
def stack1 : Nat := 4
def stack2 : Nat := 8

-- Statement of the proof problem
theorem total_coins : stack1 + stack2 = 12 :=
by
  sorry

end total_coins_l93_93428


namespace cos_neg_75_eq_l93_93833

noncomputable def cos_75_degrees : Real := (Real.sqrt 6 - Real.sqrt 2) / 4

theorem cos_neg_75_eq : Real.cos (-(75 * Real.pi / 180)) = cos_75_degrees := by
  sorry

end cos_neg_75_eq_l93_93833


namespace quadratic_expression_factorization_l93_93994

theorem quadratic_expression_factorization :
  ∃ c d : ℕ, (c > d) ∧ (x^2 - 18*x + 72 = (x - c) * (x - d)) ∧ (4*d - c = 12) := 
by
  sorry

end quadratic_expression_factorization_l93_93994


namespace birds_flew_up_count_l93_93517

def initial_birds : ℕ := 29
def final_birds : ℕ := 42

theorem birds_flew_up_count : final_birds - initial_birds = 13 :=
by sorry

end birds_flew_up_count_l93_93517


namespace range_of_m_l93_93436

theorem range_of_m (f g : ℝ → ℝ) (m : ℝ) :
  (∀ x1 : ℝ, 0 < x1 ∧ x1 < 3 / 2 → ∃ x2 : ℝ, 0 < x2 ∧ x2 < 3 / 2 ∧ f x1 > g x2) →
  (∀ x : ℝ, f x = -x + x * Real.log x + m) →
  (∀ x : ℝ, g x = -3 * Real.exp x / (3 + 4 * x ^ 2)) →
  m > 1 - 3 / 4 * Real.sqrt (Real.exp 1) :=
by
  sorry

end range_of_m_l93_93436


namespace seashells_given_l93_93041

theorem seashells_given (original_seashells : ℕ) (current_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : original_seashells = 35) 
  (h2 : current_seashells = 17) 
  (h3 : given_seashells = original_seashells - current_seashells) : 
  given_seashells = 18 := 
by 
  sorry

end seashells_given_l93_93041


namespace long_sleeve_shirts_l93_93210

variable (short_sleeve long_sleeve : Nat)
variable (total_shirts washed_shirts : Nat)
variable (not_washed_shirts : Nat)

-- Given conditions
axiom h1 : short_sleeve = 9
axiom h2 : total_shirts = 29
axiom h3 : not_washed_shirts = 1
axiom h4 : washed_shirts = total_shirts - not_washed_shirts

-- The question to be proved
theorem long_sleeve_shirts : long_sleeve = washed_shirts - short_sleeve := by
  sorry

end long_sleeve_shirts_l93_93210


namespace boat_speed_in_still_water_l93_93443

variable (x : ℝ) -- Speed of the boat in still water
variable (r : ℝ) -- Rate of the stream
variable (d : ℝ) -- Distance covered downstream
variable (t : ℝ) -- Time taken downstream

theorem boat_speed_in_still_water (h_rate : r = 5) (h_distance : d = 168) (h_time : t = 8) :
  x = 16 :=
by
  -- Substitute conditions into the equation.
  -- Calculate the effective speed downstream.
  -- Solve x from the resulting equation.
  sorry

end boat_speed_in_still_water_l93_93443


namespace fraction_difference_l93_93672

variable (a b : ℝ)

theorem fraction_difference (h : 1/a - 1/b = 1/(a + b)) : 
  1/a^2 - 1/b^2 = 1/(a * b) := 
  sorry

end fraction_difference_l93_93672


namespace kimmie_earnings_l93_93968

theorem kimmie_earnings (K : ℚ) (h : (1/2 : ℚ) * K + (1/3 : ℚ) * K = 375) : K = 450 := 
by
  sorry

end kimmie_earnings_l93_93968


namespace impossible_to_transport_50_stones_l93_93890

def arithmetic_sequence (a d n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a + i * d)

def can_transport (weights : List ℕ) (k : ℕ) (max_weight : ℕ) : Prop :=
  ∃ partition : List (List ℕ), partition.length = k ∧
    (∀ part ∈ partition, (part.sum ≤ max_weight))

theorem impossible_to_transport_50_stones :
  ¬ can_transport (arithmetic_sequence 370 2 50) 7 3000 :=
by
  sorry

end impossible_to_transport_50_stones_l93_93890


namespace jack_flyers_count_l93_93198

-- Definitions based on the given conditions
def total_flyers : ℕ := 1236
def rose_flyers : ℕ := 320
def flyers_left : ℕ := 796

-- Statement to prove
theorem jack_flyers_count : total_flyers - (rose_flyers + flyers_left) = 120 := by
  sorry

end jack_flyers_count_l93_93198


namespace track_width_l93_93740

theorem track_width (r_1 r_2 : ℝ) (h1 : r_2 = 20) (h2 : 2 * Real.pi * r_1 - 2 * Real.pi * r_2 = 20 * Real.pi) : r_1 - r_2 = 10 :=
sorry

end track_width_l93_93740


namespace solve_for_x_l93_93402

theorem solve_for_x 
  (a b c d x y z w : ℝ) 
  (H1 : x + y + z + w = 360)
  (H2 : a = x + y / 2) 
  (H3 : b = y + z / 2) 
  (H4 : c = z + w / 2) 
  (H5 : d = w + x / 2) : 
  x = (16 / 15) * (a - b / 2 + c / 4 - d / 8) :=
sorry


end solve_for_x_l93_93402


namespace counterexample_disproves_statement_l93_93429

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem counterexample_disproves_statement :
  ∃ n : ℕ, ¬ is_prime n ∧ is_prime (n + 3) :=
  by
    use 8
    -- Proof that 8 is not prime
    -- Proof that 11 (8 + 3) is prime
    sorry

end counterexample_disproves_statement_l93_93429


namespace find_number_l93_93087

theorem find_number (x : ℝ) (h : 7 * x + 21.28 = 50.68) : x = 4.2 :=
sorry

end find_number_l93_93087


namespace digit_for_divisibility_by_45_l93_93468

theorem digit_for_divisibility_by_45 (n : ℕ) (h₀ : n < 10)
  (h₁ : 5 ∣ (5 + 10 * (7 + 4 * (1 + 5 * (8 + n))))) 
  (h₂ : 9 ∣ (5 + 7 + 4 + n + 5 + 8)) : 
  n = 7 :=
by { sorry }

end digit_for_divisibility_by_45_l93_93468


namespace min_perimeter_l93_93025

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the coordinates of the right focus, point on the hyperbola, and point M
def right_focus (F : ℝ × ℝ) : Prop := F = (2, 0)
def point_on_left_branch (P : ℝ × ℝ) : Prop := P.1 < 0 ∧ hyperbola P.1 P.2
def point_M (M : ℝ × ℝ) : Prop := M = (0, 2)

-- Perimeter of ΔPFM
noncomputable def perimeter (P F M : ℝ × ℝ) : ℝ :=
  let PF := (P.1 - F.1)^2 + (P.2 - F.2)^2
  let PM := (P.1 - M.1)^2 + (P.2 - M.2)^2
  let MF := (M.1 - F.1)^2 + (M.2 - F.2)^2
  PF.sqrt + PM.sqrt + MF.sqrt

-- Theorem statement
theorem min_perimeter (P F M : ℝ × ℝ) 
  (hF : right_focus F)
  (hP : point_on_left_branch P)
  (hM : point_M M) :
  ∃ P, perimeter P F M = 2 + 4 * Real.sqrt 2 :=
sorry

end min_perimeter_l93_93025


namespace min_value_x_y_l93_93232

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1 / x + 4 / y + 8) : 
  x + y ≥ 9 :=
sorry

end min_value_x_y_l93_93232


namespace larger_inscribed_angle_corresponds_to_larger_chord_l93_93217

theorem larger_inscribed_angle_corresponds_to_larger_chord
  (R : ℝ) (α β : ℝ) (hα : α < 90) (hβ : β < 90) (h : α < β)
  (BC LM : ℝ) (hBC : BC = 2 * R * Real.sin α) (hLM : LM = 2 * R * Real.sin β) :
  BC < LM :=
sorry

end larger_inscribed_angle_corresponds_to_larger_chord_l93_93217


namespace number_of_red_balls_l93_93880

-- Initial conditions
def num_black_balls : ℕ := 7
def num_white_balls : ℕ := 5
def freq_red_ball : ℝ := 0.4

-- Proving the number of red balls
theorem number_of_red_balls (total_balls : ℕ) (num_red_balls : ℕ) :
  total_balls = num_black_balls + num_white_balls + num_red_balls ∧
  (num_red_balls : ℝ) / total_balls = freq_red_ball →
  num_red_balls = 8 :=
by
  sorry

end number_of_red_balls_l93_93880


namespace math_problem_l93_93897

noncomputable def parametric_equation_line (x y t : ℝ) : Prop :=
  x = 1 + (1/2) * t ∧ y = -5 + (Real.sqrt 3 / 2) * t

noncomputable def polar_equation_circle (ρ θ : ℝ) : Prop :=
  ρ = 8 * Real.sin θ

noncomputable def line_disjoint_circle (sqrt3 x y d : ℝ) : Prop :=
  sqrt3 = Real.sqrt 3 ∧ x = 0 ∧ y = 4 ∧ d = (9 + sqrt3) / 2 ∧ d > 4

theorem math_problem 
  (t θ x y ρ sqrt3 d : ℝ) :
  parametric_equation_line x y t ∧
  polar_equation_circle ρ θ ∧
  line_disjoint_circle sqrt3 x y d :=
by
  sorry

end math_problem_l93_93897


namespace remainder_of_division_l93_93876

theorem remainder_of_division :
  ∀ (L S R : ℕ), 
  L = 1575 → 
  L - S = 1365 → 
  S * 7 + R = L → 
  R = 105 :=
by
  intros L S R h1 h2 h3
  sorry

end remainder_of_division_l93_93876


namespace value_of_m_l93_93566

theorem value_of_m
  (m : ℤ)
  (h1 : ∃ p : ℕ → ℝ, p 4 = 1/3 ∧ p 1 = -(m + 4) ∧ p 0 = -11 ∧ (∀ (n : ℕ), (n ≠ 4 ∧ n ≠ 1 ∧ n ≠ 0) → p n = 0) ∧ 1 ≤ p 4 + p 1 + p 0) :
  m = 4 :=
  sorry

end value_of_m_l93_93566


namespace part1_part2_l93_93262

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * Real.log x - a * x ^ 2 + 1

theorem part1 (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ (0 < a ∧ a < 1) :=
sorry

theorem part2 (a : ℝ) :
  (∃ α β m : ℝ, 1 ≤ α ∧ α ≤ 4 ∧ 1 ≤ β ∧ β ≤ 4 ∧ β - α = 1 ∧ f α a = m ∧ f β a = m) ↔ 
  (Real.log 4 / 3 * (2 / 7) ≤ a ∧ a ≤ Real.log 2 * (2 / 3)) :=
sorry

end part1_part2_l93_93262


namespace smallest_t_for_circle_l93_93399

theorem smallest_t_for_circle (t : ℝ) :
  (∀ r θ, 0 ≤ θ ∧ θ ≤ t → r = Real.sin θ) → t ≥ π :=
by sorry

end smallest_t_for_circle_l93_93399


namespace regression_prediction_l93_93683

-- Define the linear regression model as a function
def linear_regression (x : ℝ) : ℝ :=
  7.19 * x + 73.93

-- State that using this model, the predicted height at age 10 is approximately 145.83
theorem regression_prediction :
  abs (linear_regression 10 - 145.83) < 0.01 :=
by 
  sorry

end regression_prediction_l93_93683


namespace cos_pi_over_4_minus_alpha_l93_93306

theorem cos_pi_over_4_minus_alpha (α : ℝ) (h : Real.sin (Real.pi / 4 + α) = 2 / 3) :
  Real.cos (Real.pi / 4 - α) = 2 / 3 := 
by
  sorry

end cos_pi_over_4_minus_alpha_l93_93306


namespace range_of_m_hyperbola_l93_93616

noncomputable def is_conic_hyperbola (expr : ℝ → ℝ → ℝ) : Prop :=
  ∃ f : ℝ, ∀ x y, expr x y = ((x - 2 * y + 3)^2 - f * (x^2 + y^2 + 2 * y + 1))

theorem range_of_m_hyperbola (m : ℝ) :
  is_conic_hyperbola (fun x y => m * (x^2 + y^2 + 2 * y + 1) - (x - 2 * y + 3)^2) → 5 < m :=
sorry

end range_of_m_hyperbola_l93_93616


namespace divides_trans_l93_93196

theorem divides_trans (m n : ℤ) (h : n ∣ m * (n + 1)) : n ∣ m :=
by
  sorry

end divides_trans_l93_93196


namespace triangle_perimeter_l93_93998

theorem triangle_perimeter (P₁ P₂ P₃ : ℝ) (hP₁ : P₁ = 12) (hP₂ : P₂ = 14) (hP₃ : P₃ = 16) : 
  P₁ + P₂ + P₃ = 42 := by
  sorry

end triangle_perimeter_l93_93998


namespace least_number_to_subtract_l93_93884

theorem least_number_to_subtract (n : ℕ) (h : n = 13294) : ∃ k : ℕ, n - 1 = k * 97 :=
by
  sorry

end least_number_to_subtract_l93_93884


namespace value_of_last_installment_l93_93171

noncomputable def total_amount_paid_without_processing_fee : ℝ :=
  36 * 2300

noncomputable def total_interest_paid : ℝ :=
  total_amount_paid_without_processing_fee - 35000

noncomputable def last_installment_value : ℝ :=
  2300 + 1000

theorem value_of_last_installment :
  last_installment_value = 3300 :=
  by
    sorry

end value_of_last_installment_l93_93171


namespace center_of_conic_l93_93911

-- Define the conic equation
def conic_equation (p q r α β γ : ℝ) : Prop :=
  p * α * β + q * α * γ + r * β * γ = 0

-- Define the barycentric coordinates of the center
def center_coordinates (p q r : ℝ) : ℝ × ℝ × ℝ :=
  (r * (p + q - r), q * (p + r - q), p * (r + q - p))

-- Theorem to prove that the barycentric coordinates of the center are as expected
theorem center_of_conic (p q r α β γ : ℝ) (h : conic_equation p q r α β γ) :
  center_coordinates p q r = (r * (p + q - r), q * (p + r - q), p * (r + q - p)) := 
sorry

end center_of_conic_l93_93911


namespace inequality_holds_for_all_l93_93783

theorem inequality_holds_for_all (m n : ℕ) (m_pos : 0 < m) (n_pos : 0 < n) :
  (∀ α β : ℝ, ⌊(m + n) * α⌋ + ⌊(m + n) * β⌋ ≥ ⌊m * α⌋ + ⌊m * β⌋ + ⌊n * (α + β)⌋) → m = n :=
by sorry

end inequality_holds_for_all_l93_93783


namespace cube_path_count_l93_93605

noncomputable def numberOfWaysToMoveOnCube : Nat :=
  20

theorem cube_path_count :
  ∀ (cube : Type) (top bottom side1 side2 side3 side4 : cube),
    (∀ (p : cube → cube → Prop), 
      (p top side1 ∨ p top side2 ∨ p top side3 ∨ p top side4) ∧ 
      (p side1 bottom ∨ p side2 bottom ∨ p side3 bottom ∨ p side4 bottom)) →
    numberOfWaysToMoveOnCube = 20 :=
by
  intros
  sorry

end cube_path_count_l93_93605


namespace num_monomials_degree_7_l93_93504

theorem num_monomials_degree_7 : 
  ∃ (count : Nat), 
    (∀ (a b c : ℕ), a + b + c = 7 → (1 : ℕ) = 1) ∧ 
    count = 15 := 
sorry

end num_monomials_degree_7_l93_93504


namespace at_least_one_prob_better_option_l93_93516

-- Definitions based on the conditions in a)

def player_A_prelim := 1 / 2
def player_B_prelim := 1 / 3
def player_C_prelim := 1 / 2

def final_round := 1 / 3

def prelim_prob_A := player_A_prelim * final_round
def prelim_prob_B := player_B_prelim * final_round
def prelim_prob_C := player_C_prelim * final_round

def prob_none := (1 - prelim_prob_A) * (1 - prelim_prob_B) * (1 - prelim_prob_C)

def prob_at_least_one := 1 - prob_none

-- Question 1 statement

theorem at_least_one_prob :
  prob_at_least_one = 31 / 81 :=
sorry

-- Definitions based on the reward options in the conditions

def option_1_lottery_prob := 1 / 3
def option_1_reward := 600
def option_1_expected_value := 600 * 3 * (1 / 3)

def option_2_prelim_reward := 100
def option_2_final_reward := 400

-- Expected values calculation for Option 2

def option_2_expected_value :=
  (300 * (1 / 6) + 600 * (5 / 12) + 900 * (1 / 3) + 1200 * (1 / 12))

-- Question 2 statement

theorem better_option :
  option_1_expected_value < option_2_expected_value :=
sorry

end at_least_one_prob_better_option_l93_93516


namespace y_directly_proportional_x_l93_93750

-- Definition for direct proportionality
def directly_proportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ y = k * x

-- Theorem stating the relationship between y and x given the condition
theorem y_directly_proportional_x (x y : ℝ) (h : directly_proportional x y) :
  ∃ k : ℝ, k ≠ 0 ∧ y = k * x :=
by
  sorry

end y_directly_proportional_x_l93_93750


namespace minimum_value_l93_93754

noncomputable def min_value_b_plus_4_over_a (a : ℝ) (b : ℝ) :=
  b + 4 / a

theorem minimum_value (a : ℝ) (b : ℝ) (h₁ : a > 0) 
  (h₂ : ∀ x : ℝ, x > 0 → (a * x - 2) * (x^2 + b * x - 5) ≥ 0) :
  min_value_b_plus_4_over_a a b = 2 * Real.sqrt 5 :=
sorry

end minimum_value_l93_93754


namespace leftmost_square_side_length_l93_93696

open Real

/-- Given the side lengths of three squares, 
    where the middle square's side length is 17 cm longer than the leftmost square,
    the rightmost square's side length is 6 cm shorter than the middle square,
    and the sum of the side lengths of all three squares is 52 cm,
    prove that the side length of the leftmost square is 8 cm. -/
theorem leftmost_square_side_length
  (x : ℝ)
  (h1 : ∀ m : ℝ, m = x + 17)
  (h2 : ∀ r : ℝ, r = x + 11)
  (h3 : x + (x + 17) + (x + 11) = 52) :
  x = 8 := by
  sorry

end leftmost_square_side_length_l93_93696


namespace correctly_calculated_value_l93_93197

theorem correctly_calculated_value (x : ℕ) (h : 5 * x = 40) : 2 * x = 16 := 
by {
  sorry
}

end correctly_calculated_value_l93_93197


namespace smallest_positive_integer_with_18_divisors_l93_93176

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ (∀ d : ℕ, d ∣ n → 0 < d → d ≠ n → (∀ m : ℕ, m ∣ d → m = 1) → n = 180) :=
sorry

end smallest_positive_integer_with_18_divisors_l93_93176


namespace latest_start_time_is_correct_l93_93292

noncomputable def doughComingToRoomTemp : ℕ := 1  -- 1 hour
noncomputable def shapingDough : ℕ := 15         -- 15 minutes
noncomputable def proofingDough : ℕ := 2         -- 2 hours
noncomputable def bakingBread : ℕ := 30          -- 30 minutes
noncomputable def coolingBread : ℕ := 15         -- 15 minutes
noncomputable def bakeryOpeningTime : ℕ := 6     -- 6:00 am

-- Total preparation time in minutes
noncomputable def totalPreparationTimeInMinutes : ℕ :=
  (doughComingToRoomTemp * 60) + shapingDough + (proofingDough * 60) + bakingBread + coolingBread

-- Total preparation time in hours
noncomputable def totalPreparationTimeInHours : ℕ :=
  totalPreparationTimeInMinutes / 60

-- Latest time the baker can start working
noncomputable def latestTimeBakerCanStart : ℕ :=
  if (bakeryOpeningTime - totalPreparationTimeInHours) < 0 then 24 + (bakeryOpeningTime - totalPreparationTimeInHours)
  else bakeryOpeningTime - totalPreparationTimeInHours

theorem latest_start_time_is_correct : latestTimeBakerCanStart = 2 := by
  sorry

end latest_start_time_is_correct_l93_93292


namespace second_number_is_30_l93_93984

-- Definitions from the conditions
def second_number (x : ℕ) := x
def first_number (x : ℕ) := 2 * x
def third_number (x : ℕ) := (2 * x) / 3
def sum_of_numbers (x : ℕ) := first_number x + second_number x + third_number x

-- Lean statement
theorem second_number_is_30 (x : ℕ) (h1 : sum_of_numbers x = 110) : x = 30 :=
by
  sorry

end second_number_is_30_l93_93984


namespace jason_total_spent_l93_93473

def cost_of_flute : ℝ := 142.46
def cost_of_music_tool : ℝ := 8.89
def cost_of_song_book : ℝ := 7.00

def total_spent (flute_cost music_tool_cost song_book_cost : ℝ) : ℝ :=
  flute_cost + music_tool_cost + song_book_cost

theorem jason_total_spent :
  total_spent cost_of_flute cost_of_music_tool cost_of_song_book = 158.35 :=
by
  -- Proof omitted
  sorry

end jason_total_spent_l93_93473


namespace highlighter_difference_l93_93976

theorem highlighter_difference :
  ∃ (P : ℕ), 7 + P + (P + 5) = 40 ∧ P - 7 = 7 :=
by
  sorry

end highlighter_difference_l93_93976


namespace find_number_l93_93657

theorem find_number (x : ℝ) (h : x - (3/5) * x = 50) : x = 125 := by
  sorry

end find_number_l93_93657


namespace sqrt_fraction_value_l93_93597

theorem sqrt_fraction_value (a b c d : Nat) (h : a = 2 ∧ b = 0 ∧ c = 2 ∧ d = 3) : 
  Real.sqrt (2023 / (a + b + c + d)) = 17 := by
  sorry

end sqrt_fraction_value_l93_93597


namespace find_diagonal_length_l93_93089

theorem find_diagonal_length (d : ℝ) (offset1 offset2 : ℝ) (area : ℝ)
  (h1 : offset1 = 9)
  (h2 : offset2 = 6)
  (h3 : area = 300) :
  (1/2) * d * (offset1 + offset2) = area → d = 40 :=
by
  -- placeholder for proof
  sorry

end find_diagonal_length_l93_93089


namespace slope_angle_AB_l93_93206

noncomputable def A : ℝ × ℝ := (0, 1)
noncomputable def B : ℝ × ℝ := (1, 0)

theorem slope_angle_AB :
  let θ := Real.arctan (↑(B.2 - A.2) / ↑(B.1 - A.1))
  θ = 3 * Real.pi / 4 := 
by
  -- Proof goes here
  sorry

end slope_angle_AB_l93_93206


namespace kyle_speed_l93_93812

theorem kyle_speed (S : ℝ) (joseph_speed : ℝ) (joseph_time : ℝ) (kyle_time : ℝ) (H1 : joseph_speed = 50) (H2 : joseph_time = 2.5) (H3 : kyle_time = 2) (H4 : joseph_speed * joseph_time = kyle_time * S + 1) : S = 62 :=
by
  sorry

end kyle_speed_l93_93812


namespace original_perimeter_not_necessarily_multiple_of_four_l93_93772

/-
Define the conditions given in the problem:
1. A rectangle is divided into several smaller rectangles.
2. The perimeter of each of these smaller rectangles is a multiple of 4.
-/
structure Rectangle where
  length : ℕ
  width : ℕ

def perimeter (r : Rectangle) : ℕ :=
  2 * (r.length + r.width)

def is_multiple_of_four (n : ℕ) : Prop :=
  n % 4 = 0

def smaller_rectangles (rs : List Rectangle) : Prop :=
  ∀ r ∈ rs, is_multiple_of_four (perimeter r)

-- Define the main statement to be proved
theorem original_perimeter_not_necessarily_multiple_of_four (original : Rectangle) (rs : List Rectangle)
  (h1 : smaller_rectangles rs) (h2 : ∀ r ∈ rs, r.length * r.width = original.length * original.width) :
  ¬ is_multiple_of_four (perimeter original) :=
by
  sorry

end original_perimeter_not_necessarily_multiple_of_four_l93_93772


namespace total_buyers_l93_93999

-- Definitions based on conditions
def C : ℕ := 50
def M : ℕ := 40
def B : ℕ := 19
def pN : ℝ := 0.29  -- Probability that a random buyer purchases neither

-- The theorem statement
theorem total_buyers :
  ∃ T : ℝ, (T = (C + M - B) + pN * T) ∧ T = 100 :=
by
  sorry

end total_buyers_l93_93999


namespace cost_of_45_lilies_l93_93153

-- Definitions of the given conditions
def cost_per_lily := 30 / 18
def lilies_18_bouquet_cost := 30
def number_of_lilies_in_bouquet := 45

-- Theorem stating the mathematical proof problem
theorem cost_of_45_lilies : cost_per_lily * number_of_lilies_in_bouquet = 75 := by
  -- The proof is omitted
  sorry

end cost_of_45_lilies_l93_93153


namespace find_number_l93_93961

theorem find_number (x : ℕ) (h : (x + 720) / 125 = 7392 / 462) : x = 1280 :=
sorry

end find_number_l93_93961


namespace simplify_expression_l93_93660

-- Define the fractions involved
def frac1 : ℚ := 1 / 2
def frac2 : ℚ := 1 / 3
def frac3 : ℚ := 1 / 5
def frac4 : ℚ := 1 / 7

-- Define the expression to be simplified
def expr : ℚ := (frac1 - frac2 + frac3) / (frac2 - frac1 + frac4)

-- The goal is to show that the expression simplifies to -77 / 5
theorem simplify_expression : expr = -77 / 5 := by
  sorry

end simplify_expression_l93_93660


namespace sum_of_first_8_terms_l93_93798

theorem sum_of_first_8_terms (a : ℝ) (h : 15 * a = 1) : 
  (a + 2 * a + 4 * a + 8 * a + 16 * a + 32 * a + 64 * a + 128 * a) = 17 :=
by
  sorry

end sum_of_first_8_terms_l93_93798


namespace number_of_unit_squares_in_50th_ring_l93_93955

def nth_ring_unit_squares (n : ℕ) : ℕ :=
  8 * n

-- Statement to prove
theorem number_of_unit_squares_in_50th_ring : nth_ring_unit_squares 50 = 400 :=
by
  -- Proof steps (skip with sorry)
  sorry

end number_of_unit_squares_in_50th_ring_l93_93955


namespace tens_digit_of_72_pow_25_l93_93403

theorem tens_digit_of_72_pow_25 : (72^25 % 100) / 10 = 3 := 
by
  sorry

end tens_digit_of_72_pow_25_l93_93403


namespace minimum_value_f_l93_93259

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / 2 + 2 / (Real.sin x)

theorem minimum_value_f (x : ℝ) (h : 0 < x ∧ x ≤ Real.pi / 2) :
  ∃ y, (∀ z, 0 < z ∧ z ≤ Real.pi / 2 → f z ≥ y) ∧ y = 5 / 2 :=
sorry

end minimum_value_f_l93_93259


namespace max_identifiable_cards_2013_l93_93875

-- Define the number of cards
def num_cards : ℕ := 2013

-- Define the function that determines the maximum t for which the numbers can be found
def max_identifiable_cards (cards : ℕ) (select : ℕ) : ℕ :=
  if (cards = 2013) ∧ (select = 10) then 1986 else 0

-- The theorem to prove the property
theorem max_identifiable_cards_2013 :
  max_identifiable_cards 2013 10 = 1986 :=
sorry

end max_identifiable_cards_2013_l93_93875


namespace cupcake_packages_l93_93714

theorem cupcake_packages (total_cupcakes eaten_cupcakes cupcakes_per_package number_of_packages : ℕ) 
  (h1 : total_cupcakes = 18)
  (h2 : eaten_cupcakes = 8)
  (h3 : cupcakes_per_package = 2)
  (h4 : number_of_packages = (total_cupcakes - eaten_cupcakes) / cupcakes_per_package) :
  number_of_packages = 5 :=
by
  -- The proof goes here, we'll use sorry to indicate it's not needed for now.
  sorry

end cupcake_packages_l93_93714


namespace a3_eq_5_l93_93661

variable {a_n : ℕ → Real} (S : ℕ → Real)
variable (a1 d : Real)

-- Define arithmetic sequence
def is_arithmetic_sequence (a_n : ℕ → Real) (a1 d : Real) : Prop :=
  ∀ n : ℕ, n > 0 → a_n n = a1 + (n - 1) * d

-- Define sum of first n terms
def sum_of_arithmetic (S : ℕ → Real) (a_n : ℕ → Real) : Prop :=
  ∀ n : ℕ, S n = n / 2 * (a_n 1 + a_n n)

-- Given conditions: S_5 = 25
def S_5_eq_25 (S : ℕ → Real) : Prop :=
  S 5 = 25

-- Goal: prove a_3 = 5
theorem a3_eq_5 (h_arith : is_arithmetic_sequence a_n a1 d)
                (h_sum : sum_of_arithmetic S a_n)
                (h_S5 : S_5_eq_25 S) : a_n 3 = 5 :=
  sorry

end a3_eq_5_l93_93661


namespace evaluate_expression_at_2_l93_93541

noncomputable def replace_and_evaluate (x : ℝ) : ℝ :=
  (3 * x - 2) / (-x + 6)

theorem evaluate_expression_at_2 :
  replace_and_evaluate 2 = -2 :=
by
  -- evaluation and computation would go here, skipped with sorry
  sorry

end evaluate_expression_at_2_l93_93541


namespace helium_balloon_buoyancy_l93_93829

variable (m m₁ Mₐ M_b : ℝ)
variable (h₁ : m₁ = 10)
variable (h₂ : Mₐ = 4)
variable (h₃ : M_b = 29)

theorem helium_balloon_buoyancy :
  m = (m₁ * Mₐ) / (M_b - Mₐ) :=
by
  sorry

end helium_balloon_buoyancy_l93_93829


namespace difference_is_four_l93_93694

open Nat

-- Assume we have a 5x5x5 cube
def cube_side_length : ℕ := 5
def total_unit_cubes : ℕ := cube_side_length ^ 3

-- Define the two configurations
def painted_cubes_config1 : ℕ := 65  -- Two opposite faces and one additional face
def painted_cubes_config2 : ℕ := 61  -- Three adjacent faces

-- The difference in the number of unit cubes with at least one painted face
def painted_difference : ℕ := painted_cubes_config1 - painted_cubes_config2

theorem difference_is_four :
    painted_difference = 4 := by
  sorry

end difference_is_four_l93_93694


namespace product_div_sum_eq_5_quotient_integer_condition_next_consecutive_set_l93_93449

theorem product_div_sum_eq_5 (x : ℤ) (h : (x^3 - x) / (3 * x) = 5) : x = 4 := by
  sorry

theorem quotient_integer_condition (x : ℤ) : ((∃ k : ℤ, x = 3 * k + 1) ∨ (∃ k : ℤ, x = 3 * k - 1)) ↔ ∃ q : ℤ, (x^3 - x) / (3 * x) = q := by
  sorry

theorem next_consecutive_set (x : ℤ) (h : x = 4) : x - 1 = 3 ∧ x = 4 ∧ x + 1 = 5 := by
  sorry

end product_div_sum_eq_5_quotient_integer_condition_next_consecutive_set_l93_93449


namespace vertex_of_parabola_l93_93064

theorem vertex_of_parabola : ∀ x y : ℝ, y = 2 * (x - 1) ^ 2 + 2 → (1, 2) = (1, 2) :=
by
  sorry

end vertex_of_parabola_l93_93064


namespace chips_in_bag_l93_93540

theorem chips_in_bag :
  let initial_chips := 5
  let additional_chips := 5
  let daily_chips := 10
  let total_days := 10
  let first_day_chips := initial_chips + additional_chips
  let remaining_days := total_days - 1
  (first_day_chips + remaining_days * daily_chips) = 100 :=
by
  sorry

end chips_in_bag_l93_93540


namespace velocity_at_specific_time_acceleration_at_specific_time_acceleration_proportional_to_displacement_l93_93851

noncomputable def x (A ω t : ℝ) : ℝ := A * Real.sin (ω * t)
noncomputable def v (A ω t : ℝ) : ℝ := deriv (x A ω) t
noncomputable def α (A ω t : ℝ) : ℝ := deriv (v A ω) t

theorem velocity_at_specific_time (A ω : ℝ) : 
  v A ω (2 * Real.pi / ω) = A * ω := 
sorry

theorem acceleration_at_specific_time (A ω : ℝ) :
  α A ω (2 * Real.pi / ω) = 0 :=
sorry

theorem acceleration_proportional_to_displacement (A ω t : ℝ) :
  α A ω t = -ω^2 * x A ω t :=
sorry

end velocity_at_specific_time_acceleration_at_specific_time_acceleration_proportional_to_displacement_l93_93851


namespace probability_exactly_two_even_dice_l93_93296

theorem probability_exactly_two_even_dice :
  let p_even := 1 / 2
  let p_not_even := 1 / 2
  let number_of_ways := 3
  let probability_each_way := (p_even * p_even * p_not_even)
  3 * probability_each_way = 3 / 8 :=
by
  sorry

end probability_exactly_two_even_dice_l93_93296


namespace min_value_of_expression_l93_93559

theorem min_value_of_expression (a : ℝ) (h₀ : a > 0)
  (x₁ x₂ : ℝ)
  (h₁ : x₁ + x₂ = 4 * a)
  (h₂ : x₁ * x₂ = a * a) :
  x₁ + x₂ + a / (x₁ * x₂) = 4 :=
sorry

end min_value_of_expression_l93_93559


namespace one_third_12x_plus_5_l93_93083

-- Define x as a real number
variable (x : ℝ)

-- Define the hypothesis
def h := 12 * x + 5

-- State the theorem
theorem one_third_12x_plus_5 : (1 / 3) * (12 * x + 5) = 4 * x + 5 / 3 :=
  by 
    sorry -- Proof is omitted

end one_third_12x_plus_5_l93_93083


namespace shares_difference_l93_93685

theorem shares_difference (x : ℝ) (hp : ℝ) (hq : ℝ) (hr : ℝ)
  (hx : hp = 3 * x) (hqx : hq = 7 * x) (hrx : hr = 12 * x) 
  (hqr_diff : hr - hq = 3500) : (hq - hp = 2800) :=
by
  -- The proof would be done here, but the problem statement requires only the theorem statement
  sorry

end shares_difference_l93_93685


namespace total_respondents_l93_93888

theorem total_respondents (X Y : ℕ) 
  (hX : X = 60) 
  (hRatio : 3 * Y = X) : 
  X + Y = 80 := 
by
  sorry

end total_respondents_l93_93888


namespace ratio_eq_one_l93_93787

theorem ratio_eq_one (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a * b ≠ 0) : (a / 3) / (b / 2) = 1 :=
sorry

end ratio_eq_one_l93_93787


namespace fraction_of_two_bedroom_l93_93213

theorem fraction_of_two_bedroom {x : ℝ} 
    (h1 : 0.17 + x = 0.5) : x = 0.33 :=
by
  sorry

end fraction_of_two_bedroom_l93_93213


namespace children_tickets_l93_93499

-- Definition of the problem
variables (A C t : ℕ) (h_eq_people : A + C = t) (h_eq_money : 9 * A + 5 * C = 190)

-- The main statement we need to prove
theorem children_tickets (h_t : t = 30) : C = 20 :=
by {
  -- Proof will go here eventually
  sorry
}

end children_tickets_l93_93499


namespace book_transaction_difference_l93_93647

def number_of_books : ℕ := 15
def cost_per_book : ℕ := 11
def selling_price_per_book : ℕ := 25

theorem book_transaction_difference :
  number_of_books * selling_price_per_book - number_of_books * cost_per_book = 210 :=
by
  sorry

end book_transaction_difference_l93_93647


namespace fraction_value_l93_93221

theorem fraction_value : (2 * 0.24) / (20 * 2.4) = 0.01 := by
  sorry

end fraction_value_l93_93221


namespace length_SR_l93_93803

theorem length_SR (cos_S : ℝ) (SP : ℝ) (SR : ℝ) (h1 : cos_S = 0.5) (h2 : SP = 10) (h3 : cos_S = SP / SR) : SR = 20 := by
  sorry

end length_SR_l93_93803


namespace no_equalities_l93_93389

def f1 (x : ℤ) : ℤ := x * (x - 2007)
def f2 (x : ℤ) : ℤ := (x - 1) * (x - 2006)
def f1004 (x : ℤ) : ℤ := (x - 1003) * (x - 1004)

theorem no_equalities (x : ℤ) (h : 0 ≤ x ∧ x ≤ 2007) :
  ¬(f1 x = f2 x ∨ f1 x = f1004 x ∨ f2 x = f1004 x) :=
by
  sorry

end no_equalities_l93_93389


namespace angles_does_not_exist_l93_93141

theorem angles_does_not_exist (a1 a2 a3 : ℝ) 
  (h1 : a1 + a2 = 90) 
  (h2 : a2 + a3 = 180) 
  (h3 : a3 = 18) : False :=
by
  sorry

end angles_does_not_exist_l93_93141


namespace probability_either_boy_A_or_girl_B_correct_probability_B_correct_conditional_probability_A_given_B_correct_l93_93749

-- Define the total number of ways to choose 3 leaders from 6 students
def total_ways : ℕ := Nat.choose 6 3

-- Calculate the number of ways in which boy A or girl B is chosen
def boy_A_chosen_ways : ℕ := Nat.choose 4 2 + 4 * 2
def girl_B_chosen_ways : ℕ := Nat.choose 4 1 + Nat.choose 4 2
def either_boy_A_or_girl_B_chosen_ways : ℕ := boy_A_chosen_ways + girl_B_chosen_ways

-- Calculate the probability that either boy A or girl B is chosen
def probability_either_boy_A_or_girl_B : ℚ := either_boy_A_or_girl_B_chosen_ways / total_ways

-- Calculate the probability that girl B is chosen
def girl_B_total_ways : ℕ := Nat.choose 5 2
def probability_B : ℚ := girl_B_total_ways / total_ways

-- Calculate the probability that both boy A and girl B are chosen
def both_A_and_B_chosen_ways : ℕ := Nat.choose 4 1
def probability_AB : ℚ := both_A_and_B_chosen_ways / total_ways

-- Calculate the conditional probability P(A|B) given P(B)
def conditional_probability_A_given_B : ℚ := probability_AB / probability_B

-- Theorem statements
theorem probability_either_boy_A_or_girl_B_correct : probability_either_boy_A_or_girl_B = (4 / 5) := sorry
theorem probability_B_correct : probability_B = (1 / 2) := sorry
theorem conditional_probability_A_given_B_correct : conditional_probability_A_given_B = (2 / 5) := sorry

end probability_either_boy_A_or_girl_B_correct_probability_B_correct_conditional_probability_A_given_B_correct_l93_93749


namespace sum_of_angles_around_point_l93_93224

theorem sum_of_angles_around_point (x : ℝ) (h : 6 * x + 3 * x + 4 * x + x + 2 * x = 360) : x = 22.5 :=
by
  sorry

end sum_of_angles_around_point_l93_93224


namespace inequality_proof_l93_93178

variable {α β γ : ℝ}

theorem inequality_proof (h1 : β * γ ≠ 0) (h2 : (1 - γ^2) / (β * γ) ≥ 0) :
  10 * (α^2 + β^2 + γ^2 - β * γ^2) ≥ 2 * α * β + 5 * α * γ :=
sorry

end inequality_proof_l93_93178


namespace sin_A_sin_C_eq_3_over_4_triangle_is_equilateral_l93_93334

variable {α : Type*}

-- Part 1
theorem sin_A_sin_C_eq_3_over_4
  (A B C : Real)
  (a b c : Real)
  (h1 : b ^ 2 = a * c)
  (h2 : (Real.cos (A - C)) + (Real.cos B) = 3 / 2) :
  Real.sin A * Real.sin C = 3 / 4 :=
sorry

-- Part 2
theorem triangle_is_equilateral
  (A B C : Real)
  (a b c : Real)
  (h1 : b ^ 2 = a * c)
  (h2 : (Real.cos (A - C)) + (Real.cos B) = 3 / 2) :
  A = B ∧ B = C :=
sorry

end sin_A_sin_C_eq_3_over_4_triangle_is_equilateral_l93_93334


namespace fraction_sum_l93_93327

theorem fraction_sum : (3 / 4 : ℚ) + (6 / 9 : ℚ) = 17 / 12 := 
by 
  -- Sorry placeholder to indicate proof is not provided.
  sorry

end fraction_sum_l93_93327


namespace total_amount_is_4000_l93_93348

-- Define the amount put at a 3% interest rate
def amount_at_3_percent : ℝ := 2800

-- Define the total annual interest from both investments
def total_annual_interest : ℝ := 144

-- Define the interest rate for the amount put at 3% and 5%
def interest_rate_3_percent : ℝ := 0.03
def interest_rate_5_percent : ℝ := 0.05

-- Define the total amount to be proved
def total_amount_divided (T : ℝ) : Prop :=
  interest_rate_3_percent * amount_at_3_percent + 
  interest_rate_5_percent * (T - amount_at_3_percent) = total_annual_interest

-- The theorem that states the total amount divided is Rs. 4000
theorem total_amount_is_4000 : ∃ T : ℝ, total_amount_divided T ∧ T = 4000 :=
by
  use 4000
  unfold total_amount_divided
  simp
  sorry

end total_amount_is_4000_l93_93348


namespace measure_Z_is_19_6_l93_93959

def measure_angle_X : ℝ := 72
def measure_Y (measure_Z : ℝ) : ℝ := 4 * measure_Z + 10
def angle_sum_condition (measure_Z : ℝ) : Prop :=
  measure_angle_X + (measure_Y measure_Z) + measure_Z = 180

theorem measure_Z_is_19_6 :
  ∃ measure_Z : ℝ, measure_Z = 19.6 ∧ angle_sum_condition measure_Z :=
by
  sorry

end measure_Z_is_19_6_l93_93959


namespace extreme_points_exactly_one_zero_in_positive_interval_l93_93177

noncomputable def f (x a : ℝ) : ℝ := (x - 1) * Real.exp x - (1 / 3) * a * x^3

theorem extreme_points (a : ℝ) (h : a > Real.exp 1) :
  ∃ (x1 x2 x3 : ℝ), (0 < x1) ∧ (x1 < x2) ∧ (x2 < x3) ∧ (deriv (f x) = 0) := sorry

theorem exactly_one_zero_in_positive_interval (a : ℝ) (h : a > Real.exp 1) :
  ∃! x : ℝ, (0 < x) ∧ (f x a = 0) := sorry

end extreme_points_exactly_one_zero_in_positive_interval_l93_93177


namespace problem_a_problem_b_l93_93743

-- Problem (a): Prove that (1 + 1/x)(1 + 1/y) ≥ 9 given x > 0, y > 0, and x + y = 1
theorem problem_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : 
  (1 + 1 / x) * (1 + 1 / y) ≥ 9 := sorry

-- Problem (b): Prove that 0 < u + v - uv < 1 given 0 < u < 1 and 0 < v < 1
theorem problem_b (u v : ℝ) (hu : 0 < u) (hu1 : u < 1) (hv : 0 < v) (hv1 : v < 1) : 
  0 < u + v - u * v ∧ u + v - u * v < 1 := sorry

end problem_a_problem_b_l93_93743


namespace problem_solution_l93_93563

theorem problem_solution (x y : ℝ) (h₁ : x + Real.cos y = 2010) (h₂ : x + 2010 * Real.sin y = 2011) (h₃ : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 2011 + Real.pi := 
sorry

end problem_solution_l93_93563


namespace function_range_l93_93116

theorem function_range (f : ℝ → ℝ) (s : Set ℝ) (h : s = Set.Ico (-5 : ℝ) 2) (h_f : ∀ x ∈ s, f x = 3 * x - 1) :
  Set.image f s = Set.Ico (-16 : ℝ) 5 :=
sorry

end function_range_l93_93116


namespace max_min_sum_zero_l93_93599

def cubic_function (x : ℝ) : ℝ :=
  x^3 - 3 * x

def first_derivative (x : ℝ) : ℝ :=
  3 * x^2 - 3

theorem max_min_sum_zero :
  let m := cubic_function (-1);
  let n := cubic_function 1;
  m + n = 0 :=
by
  sorry

end max_min_sum_zero_l93_93599


namespace find_interval_l93_93776

theorem find_interval (x : ℝ) : (x > 3/4 ∧ x < 4/5) ↔ (5 * x + 1 > 3 ∧ 5 * x + 1 < 5 ∧ 4 * x > 3 ∧ 4 * x < 5) :=
by
  sorry

end find_interval_l93_93776


namespace remainder_zero_division_l93_93883

theorem remainder_zero_division :
  ∀ x : ℂ, (x^2 - x + 1 = 0) →
    ((x^5 + x^4 - x^3 - x^2 + 1) * (x^3 - 1)) % (x^2 - x + 1) = 0 :=
by sorry

end remainder_zero_division_l93_93883


namespace proof_valid_set_exists_l93_93082

noncomputable def valid_set_exists : Prop :=
∃ (s : Finset ℕ), s.card = 10 ∧ 
(∀ (a b : ℕ), a ∈ s → b ∈ s → a ≠ b → a ≠ b) ∧ 
(∃ (t1 : Finset ℕ), t1 ⊆ s ∧ t1.card = 3 ∧ ∀ n ∈ t1, 5 ∣ n) ∧
(∃ (t2 : Finset ℕ), t2 ⊆ s ∧ t2.card = 4 ∧ ∀ n ∈ t2, 4 ∣ n) ∧
s.sum id < 75

theorem proof_valid_set_exists : valid_set_exists :=
sorry

end proof_valid_set_exists_l93_93082


namespace three_tenths_of_number_l93_93440

theorem three_tenths_of_number (N : ℝ) (h : (1/3) * (1/4) * N = 15) : (3/10) * N = 54 :=
sorry

end three_tenths_of_number_l93_93440


namespace abs_sum_bound_l93_93247

theorem abs_sum_bound (x : ℝ) (a : ℝ) (h : |x - 4| + |x - 3| < a) (ha : 0 < a) : 1 < a :=
by
  sorry

end abs_sum_bound_l93_93247


namespace find_x_values_l93_93966

def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem find_x_values :
  {x : ℝ | f (f x) = f x} = {0, 4, 5, -1} :=
by
  sorry

end find_x_values_l93_93966


namespace fruit_eating_problem_l93_93882

theorem fruit_eating_problem (a₀ p₀ o₀ : ℕ) (h₀ : a₀ = 5) (h₁ : p₀ = 8) (h₂ : o₀ = 11) :
  ¬ ∃ (d : ℕ), (a₀ - d) = (p₀ - d) ∧ (p₀ - d) = (o₀ - d) ∧ ∀ k, k ≤ d → ((a₀ - k) + (p₀ - k) + (o₀ - k) = 24 - 2 * k ∧ a₀ - k ≥ 0 ∧ p₀ - k ≥ 0 ∧ o₀ - k ≥ 0) :=
by
  sorry

end fruit_eating_problem_l93_93882


namespace learning_machine_price_reduction_l93_93529

theorem learning_machine_price_reduction (x : ℝ) (h1 : 2000 * (1 - x) * (1 - x) = 1280) : 2000 * (1 - x)^2 = 1280 :=
by
  sorry

end learning_machine_price_reduction_l93_93529


namespace prove_divisibility_l93_93588

-- Definitions for natural numbers m, n, k
variables (m n k : ℕ)

-- Conditions stating divisibility
def div1 := m^n ∣ n^m
def div2 := n^k ∣ k^n

-- The final theorem to prove
theorem prove_divisibility (hmn : div1 m n) (hnk : div2 n k) : m^k ∣ k^m :=
sorry

end prove_divisibility_l93_93588


namespace scientist_birth_day_is_wednesday_l93_93735

noncomputable def calculate_birth_day : String :=
  let years := 150
  let leap_years := 36
  let regular_years := years - leap_years
  let total_days_backward := regular_years + 2 * leap_years -- days to move back
  let days_mod := total_days_backward % 7
  let day_of_birth := (5 + 7 - days_mod) % 7 -- 5 is for backward days from Monday
  match day_of_birth with
  | 0 => "Monday"
  | 1 => "Sunday"
  | 2 => "Saturday"
  | 3 => "Friday"
  | 4 => "Thursday"
  | 5 => "Wednesday"
  | 6 => "Tuesday"
  | _ => "Error"

theorem scientist_birth_day_is_wednesday :
  calculate_birth_day = "Wednesday" :=
  by
    sorry

end scientist_birth_day_is_wednesday_l93_93735


namespace product_of_three_consecutive_cubes_divisible_by_504_l93_93593

theorem product_of_three_consecutive_cubes_divisible_by_504 (a : ℤ) : 
  ∃ k : ℤ, (a^3 - 1) * a^3 * (a^3 + 1) = 504 * k :=
by
  -- Proof omitted
  sorry

end product_of_three_consecutive_cubes_divisible_by_504_l93_93593


namespace sum_of_first_12_terms_l93_93437

noncomputable def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

def Sn (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2

theorem sum_of_first_12_terms (a d : ℤ) (h1 : a + d * 4 = 3 * (a + d * 2))
                             (h2 : a + d * 9 = 14) : Sn a d 12 = 84 := 
by
  sorry

end sum_of_first_12_terms_l93_93437


namespace hyperbola_asymptotes_l93_93716

theorem hyperbola_asymptotes (x y : ℝ) : 
  (x^2 - (y^2 / 4) = 1) ↔ (y = 2 * x ∨ y = -2 * x) := by
  sorry

end hyperbola_asymptotes_l93_93716


namespace geometric_sequence_common_ratio_l93_93873

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_cond : (a 0 * (1 + q + q^2)) / (a 0 * q^2) = 3) : q = 1 :=
by
  sorry

end geometric_sequence_common_ratio_l93_93873


namespace greatest_two_digit_number_l93_93623

theorem greatest_two_digit_number (x y : ℕ) (h1 : x < y) (h2 : x * y = 12) : 10 * x + y = 34 :=
sorry

end greatest_two_digit_number_l93_93623


namespace football_goals_l93_93991

variable (A : ℚ) (G : ℚ)

theorem football_goals (A G : ℚ) 
    (h1 : G = 14 * A)
    (h2 : G + 3 = (A + 0.08) * 15) :
    G = 25.2 :=
by
  -- Proof here
  sorry

end football_goals_l93_93991


namespace inverse_function_less_than_zero_l93_93173

theorem inverse_function_less_than_zero (x : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = 2^x + 1) (h₂ : ∀ y, f (f⁻¹ y) = y) (h₃ : ∀ y, f⁻¹ (f y) = y) :
  {x | f⁻¹ x < 0} = {x | 1 < x ∧ x < 2} :=
by
  sorry

end inverse_function_less_than_zero_l93_93173


namespace afternoon_snack_calories_l93_93684

def ellen_daily_calories : ℕ := 2200
def breakfast_calories : ℕ := 353
def lunch_calories : ℕ := 885
def dinner_remaining_calories : ℕ := 832

theorem afternoon_snack_calories :
  ellen_daily_calories - (breakfast_calories + lunch_calories + dinner_remaining_calories) = 130 :=
by sorry

end afternoon_snack_calories_l93_93684


namespace total_value_of_bills_l93_93810

theorem total_value_of_bills 
  (total_bills : Nat := 12) 
  (num_5_dollar_bills : Nat := 4) 
  (num_10_dollar_bills : Nat := 8)
  (value_5_dollar_bill : Nat := 5)
  (value_10_dollar_bill : Nat := 10) :
  (num_5_dollar_bills * value_5_dollar_bill + num_10_dollar_bills * value_10_dollar_bill = 100) :=
by
  sorry

end total_value_of_bills_l93_93810


namespace proposition_C_l93_93853

theorem proposition_C (a b : ℝ) : a^3 > b^3 → a > b :=
sorry

end proposition_C_l93_93853


namespace part_a_part_b_l93_93537

-- Part (a): Number of ways to distribute 20 identical balls into 6 boxes so that no box is empty
theorem part_a:
  ∃ (n : ℕ), n = Nat.choose 19 5 :=
sorry

-- Part (b): Number of ways to distribute 20 identical balls into 6 boxes if some boxes can be empty
theorem part_b:
  ∃ (n : ℕ), n = Nat.choose 25 5 :=
sorry

end part_a_part_b_l93_93537


namespace multiplication_pattern_correct_l93_93918

theorem multiplication_pattern_correct :
  (1 * 9 + 2 = 11) ∧
  (12 * 9 + 3 = 111) ∧
  (123 * 9 + 4 = 1111) ∧
  (1234 * 9 + 5 = 11111) ∧
  (12345 * 9 + 6 = 111111) →
  123456 * 9 + 7 = 1111111 :=
by
  sorry

end multiplication_pattern_correct_l93_93918


namespace eleven_pow_2023_mod_eight_l93_93693

theorem eleven_pow_2023_mod_eight (h11 : 11 % 8 = 3) (h3 : 3^2 % 8 = 1) : 11^2023 % 8 = 3 :=
by
  sorry

end eleven_pow_2023_mod_eight_l93_93693


namespace quadratic_one_root_greater_than_two_other_less_than_two_l93_93523

theorem quadratic_one_root_greater_than_two_other_less_than_two (m : ℝ) :
  (∀ x y : ℝ, x^2 + (2 * m - 3) * x + m - 150 = 0 ∧ x > 2 ∧ y < 2) →
  m > 5 :=
by
  sorry

end quadratic_one_root_greater_than_two_other_less_than_two_l93_93523


namespace find_real_roots_l93_93649

theorem find_real_roots : 
  {x : ℝ | x^9 + (9 / 8) * x^6 + (27 / 64) * x^3 - x + (219 / 512) = 0} =
  {1 / 2, (-1 + Real.sqrt 13) / 4, (-1 - Real.sqrt 13) / 4} :=
by
  sorry

end find_real_roots_l93_93649


namespace three_digit_sum_27_l93_93555

theorem three_digit_sum_27 {a b c : ℕ} (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) :
  a + b + c = 27 → (a, b, c) = (9, 9, 9) :=
by
  sorry

end three_digit_sum_27_l93_93555


namespace vector_dot_product_l93_93666

open Real

variables (a b : ℝ × ℝ)

def condition1 : Prop := (a.1 + b.1 = 1 ∧ a.2 + b.2 = -3)
def condition2 : Prop := (a.1 - b.1 = 3 ∧ a.2 - b.2 = 7)
def dot_product : ℝ := a.1 * b.1 + a.2 * b.2

theorem vector_dot_product :
  condition1 a b ∧ condition2 a b → dot_product a b = -12 := by
  sorry

end vector_dot_product_l93_93666


namespace zero_of_f_l93_93643

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x + 1)

theorem zero_of_f :
  ∃ x : ℝ, f x = 0 ↔ x = 1 :=
by
  sorry

end zero_of_f_l93_93643


namespace problem_statement_l93_93483

theorem problem_statement (h: 2994 * 14.5 = 175) : 29.94 * 1.45 = 1.75 := 
by {
  sorry
}

end problem_statement_l93_93483


namespace floor_T_value_l93_93305

noncomputable def floor_T : ℝ := 
  let p := (0 : ℝ)
  let q := (0 : ℝ)
  let r := (0 : ℝ)
  let s := (0 : ℝ)
  p + q + r + s

theorem floor_T_value (p q r s : ℝ) (hpq: p^2 + q^2 = 2500) (hrs: r^2 + s^2 = 2500) (hpr: p * r = 1200) (hqs: q * s = 1200) (hpos: p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0) :
  ∃ T : ℝ, T = p + q + r + s ∧ ⌊T⌋ = 140 := 
  by
  sorry

end floor_T_value_l93_93305


namespace ratio_AB_CD_lengths_AB_CD_l93_93545

-- Given conditions as definitions
def ABD_triangle (A B D : Point) : Prop := true  -- In quadrilateral ABCD, a diagonal BD is drawn
def BCD_triangle (B C D : Point) : Prop := true  -- Circles are inscribed in triangles ABD and BCD
def Line_through_B_center_AM_M (A B D M : Point) (AM MD : ℚ) : Prop :=
  (AM = 8/5) ∧ (MD = 12/5)
def Line_through_D_center_BN_N (B C D N : Point) (BN NC : ℚ) : Prop :=
  (BN = 30/11) ∧ (NC = 25/11)

-- Mathematically equivalent proof problems
theorem ratio_AB_CD (A B C D M N : Point) (AM MD BN NC : ℚ) :
  ABD_triangle A B D → 
  BCD_triangle B C D →
  Line_through_B_center_AM_M A B D M AM MD → 
  Line_through_D_center_BN_N B C D N BN NC →
  AB / CD = 4 / 5 :=
by
  sorry

theorem lengths_AB_CD (A B C D M N : Point) (AM MD BN NC : ℚ) :
  ABD_triangle A B D → 
  BCD_triangle B C D →
  Line_through_B_center_AM_M A B D M AM MD → 
  Line_through_D_center_BN_N B C D N BN NC →
  AB + CD = 9 ∧
  AB - CD = -1 :=
by 
  sorry

end ratio_AB_CD_lengths_AB_CD_l93_93545


namespace number_of_bars_in_box_l93_93166

variable (x : ℕ)
variable (cost_per_bar : ℕ := 6)
variable (remaining_bars : ℕ := 6)
variable (total_money_made : ℕ := 42)

theorem number_of_bars_in_box :
  cost_per_bar * (x - remaining_bars) = total_money_made → x = 13 :=
by
  intro h
  sorry

end number_of_bars_in_box_l93_93166


namespace marks_in_biology_l93_93723

theorem marks_in_biology (marks_english : ℕ) (marks_math : ℕ) (marks_physics : ℕ) (marks_chemistry : ℕ) (average_marks : ℕ) :
  marks_english = 73 → marks_math = 69 → marks_physics = 92 → marks_chemistry = 64 → average_marks = 76 →
  (380 - (marks_english + marks_math + marks_physics + marks_chemistry)) = 82 :=
by
  intros
  sorry

end marks_in_biology_l93_93723


namespace angle_sum_l93_93053

theorem angle_sum {A B D F G : Type} 
  (angle_A : ℝ) 
  (angle_AFG : ℝ) 
  (angle_AGF : ℝ) 
  (angle_BFD : ℝ)
  (H1 : angle_A = 30)
  (H2 : angle_AFG = angle_AGF)
  (H3 : angle_BFD = 105)
  (H4 : angle_AFG + angle_BFD = 180) 
  : angle_B + angle_D = 75 := 
by 
  sorry

end angle_sum_l93_93053


namespace willam_farm_tax_l93_93842

theorem willam_farm_tax
  (T : ℝ)
  (h1 : 0.4 * T * (3840 / (0.4 * T)) = 3840)
  (h2 : 0 < T) :
  0.3125 * T * (3840 / (0.4 * T)) = 3000 := by
  sorry

end willam_farm_tax_l93_93842


namespace plane_through_points_l93_93386

def point := (ℝ × ℝ × ℝ)

def plane_equation (A B C D : ℤ) (x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

theorem plane_through_points : 
  ∃ (A B C D : ℤ), (A > 0) ∧ (Int.gcd (Int.gcd (Int.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1) ∧
  plane_equation A B C D 2 (-3) 5 ∧
  plane_equation A B C D (-1) (-3) 7 ∧
  plane_equation A B C D (-4) (-5) 6 ∧
  (A = 2) ∧ (B = -9) ∧ (C = 3) ∧ (D = -46) :=
sorry

end plane_through_points_l93_93386


namespace intersection_of_A_B_find_a_b_l93_93973

-- Lean 4 definitions based on the given conditions
def setA (x : ℝ) : Prop := 4 - x^2 > 0
def setB (x : ℝ) (y : ℝ) : Prop := y = Real.log (-x^2 + 2*x + 3) ∧ -x^2 + 2*x + 3 > 0

-- Prove the intersection of sets A and B
theorem intersection_of_A_B :
  {x : ℝ | setA x} ∩ {x : ℝ | ∃ y : ℝ, setB x y} = {x : ℝ | -2 < x ∧ x < 1} :=
by
  sorry

-- On the roots of the quadratic equation and solution interval of inequality
theorem find_a_b (a b : ℝ) :
  (∀ x : ℝ, 2 * x^2 + a * x + b < 0 ↔ -3 < x ∧ x < 1) →
  a = 4 ∧ b = -6 :=
by
  sorry

end intersection_of_A_B_find_a_b_l93_93973


namespace triangle_perimeter_l93_93778

theorem triangle_perimeter (a b : ℝ) (x : ℝ) 
  (h₁ : a = 3) 
  (h₂ : b = 5) 
  (h₃ : x ^ 2 - 5 * x + 6 = 0)
  (h₄ : 2 < x ∧ x < 8) : a + b + x = 11 :=
by sorry

end triangle_perimeter_l93_93778


namespace ted_alex_age_ratio_l93_93771

theorem ted_alex_age_ratio (t a : ℕ) 
  (h1 : t - 3 = 4 * (a - 3))
  (h2 : t - 5 = 5 * (a - 5)) : 
  ∃ x : ℕ, (t + x) / (a + x) = 3 ∧ x = 1 :=
by
  sorry

end ted_alex_age_ratio_l93_93771


namespace area_of_shaded_rectangle_l93_93234

theorem area_of_shaded_rectangle (w₁ h₁ w₂ h₂: ℝ) 
  (hw₁: w₁ * h₁ = 6)
  (hw₂: w₂ * h₁ = 15)
  (hw₃: w₂ * h₂ = 25) :
  w₁ * h₂ = 10 :=
by
  sorry

end area_of_shaded_rectangle_l93_93234


namespace sum_of_squares_of_rates_l93_93330

theorem sum_of_squares_of_rates :
  ∃ (b j s : ℕ), 3 * b + j + 5 * s = 89 ∧ 4 * b + 3 * j + 2 * s = 106 ∧ b^2 + j^2 + s^2 = 821 := 
by
  sorry

end sum_of_squares_of_rates_l93_93330


namespace find_function_f_l93_93856

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_function_f (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y →
    f (f y / f x + 1) = f (x + y / x + 1) - f x) →
  ∀ x : ℝ, 0 < x → f x = a * x :=
  by sorry

end find_function_f_l93_93856


namespace balloon_permutations_l93_93396

theorem balloon_permutations : 
  let n : ℕ := 7
  let k1 : ℕ := 2
  let k2 : ℕ := 2
  ∃ distinct_arrangements : ℕ, 
  distinct_arrangements = n.factorial / (k1.factorial * k2.factorial) 
  ∧ distinct_arrangements = 1260 :=
by
  sorry

end balloon_permutations_l93_93396


namespace lines_perpendicular_l93_93412

-- Define the lines l1 and l2
def line1 (m x y : ℝ) := m * x + y - 1 = 0
def line2 (m x y : ℝ) := x + (m - 1) * y + 2 = 0

-- State the problem: Find the value of m such that the lines l1 and l2 are perpendicular.
theorem lines_perpendicular (m : ℝ) (h₁ : line1 m x y) (h₂ : line2 m x y) : m = 1/2 := 
sorry

end lines_perpendicular_l93_93412


namespace right_triangle_legs_sum_l93_93009

theorem right_triangle_legs_sum
  (x : ℕ)
  (h_even : Even x)
  (h_eq : x^2 + (x + 2)^2 = 34^2) :
  x + (x + 2) = 50 := 
by
  sorry

end right_triangle_legs_sum_l93_93009


namespace transaction_loss_l93_93302

theorem transaction_loss :
  let house_sale_price := 10000
  let store_sale_price := 15000
  let house_loss_percentage := 0.25
  let store_gain_percentage := 0.25
  let h := house_sale_price / (1 - house_loss_percentage)
  let s := store_sale_price / (1 + store_gain_percentage)
  let total_cost_price := h + s
  let total_selling_price := house_sale_price + store_sale_price
  let difference := total_selling_price - total_cost_price
  difference = -1000 / 3 :=
by
  sorry

end transaction_loss_l93_93302


namespace determine_ts_l93_93874

theorem determine_ts :
  ∃ t s : ℝ, 
  (⟨3, 1⟩ : ℝ × ℝ) + t • (⟨4, -6⟩) = (⟨0, 2⟩ : ℝ × ℝ) + s • (⟨-3, 5⟩) :=
by
  use 6, -9
  sorry

end determine_ts_l93_93874


namespace rational_solution_exists_l93_93755

theorem rational_solution_exists :
  ∃ (a b : ℚ), (a + b) / a + a / (a + b) = b :=
by
  sorry

end rational_solution_exists_l93_93755


namespace unique_handshakes_462_l93_93463

theorem unique_handshakes_462 : 
  ∀ (twins triplets : Type) (twin_set : ℕ) (triplet_set : ℕ) (handshakes_among_twins handshakes_among_triplets cross_handshakes_twins cross_handshakes_triplets : ℕ),
  twin_set = 12 ∧
  triplet_set = 4 ∧
  handshakes_among_twins = (24 * 22) / 2 ∧
  handshakes_among_triplets = (12 * 9) / 2 ∧
  cross_handshakes_twins = 24 * (12 / 3) ∧
  cross_handshakes_triplets = 12 * (24 / 3 * 2) →
  (handshakes_among_twins + handshakes_among_triplets + (cross_handshakes_twins + cross_handshakes_triplets) / 2) = 462 := 
by
  sorry

end unique_handshakes_462_l93_93463


namespace parallel_lines_m_l93_93953

theorem parallel_lines_m (m : ℝ) :
  (∀ (x y : ℝ), 2 * x + (m + 1) * y + 4 = 0) ∧ (∀ (x y : ℝ), m * x + 3 * y - 2 = 0) →
  (m = -3 ∨ m = 2) :=
by
  sorry

end parallel_lines_m_l93_93953


namespace weighted_average_correct_l93_93993

noncomputable def weightedAverage := 
  (5 * (3/5 : ℝ) + 3 * (4/9 : ℝ) + 8 * 0.45 + 4 * 0.067) / (5 + 3 + 8 + 4)

theorem weighted_average_correct :
  weightedAverage = 0.41 :=
by
  sorry

end weighted_average_correct_l93_93993


namespace albums_not_in_both_l93_93472

-- Definitions representing the problem conditions
def andrew_albums : ℕ := 23
def common_albums : ℕ := 11
def john_unique_albums : ℕ := 8

-- Proof statement (not the actual proof)
theorem albums_not_in_both : 
  (andrew_albums - common_albums) + john_unique_albums = 20 :=
by
  sorry

end albums_not_in_both_l93_93472


namespace initial_fliers_l93_93446

theorem initial_fliers (F : ℕ) (morning_sent afternoon_sent remaining : ℕ) :
  morning_sent = F / 5 → 
  afternoon_sent = (F - morning_sent) / 4 → 
  remaining = F - morning_sent - afternoon_sent → 
  remaining = 1800 → 
  F = 3000 := 
by 
  sorry

end initial_fliers_l93_93446


namespace sufficient_but_not_necessary_condition_l93_93784

variables (x y : ℝ)

theorem sufficient_but_not_necessary_condition :
  ((x - 1) ^ 2 + (y - 2) ^ 2 = 0) → ((x - 1) * (y - 2) = 0) ∧ (¬ ((x - 1) * (y-2) = 0 → (x - 1)^2 + (y - 2)^2 = 0)) :=
by 
  sorry

end sufficient_but_not_necessary_condition_l93_93784


namespace symmetric_point_x_axis_l93_93493

theorem symmetric_point_x_axis (P Q : ℝ × ℝ) (hP : P = (-1, 2)) (hQ : Q = (P.1, -P.2)) : Q = (-1, -2) :=
sorry

end symmetric_point_x_axis_l93_93493


namespace solve_for_k_l93_93115

theorem solve_for_k (a k : ℝ) (h : a ^ 10 / (a ^ k) ^ 4 = a ^ 2) : k = 2 :=
by
  sorry

end solve_for_k_l93_93115


namespace average_production_n_days_l93_93466

theorem average_production_n_days (n : ℕ) (P : ℕ) 
  (hP : P = 80 * n)
  (h_new_avg : (P + 220) / (n + 1) = 95) : 
  n = 8 := 
by
  sorry -- Proof of the theorem

end average_production_n_days_l93_93466


namespace minimum_value_of_a_l93_93149

theorem minimum_value_of_a (a b : ℕ) (h₁ : b - a = 2013) 
(h₂ : ∃ x : ℕ, x^2 - a * x + b = 0) : a = 93 :=
sorry

end minimum_value_of_a_l93_93149


namespace ways_A_not_head_is_600_l93_93912

-- Definitions for the problem conditions
def num_people : ℕ := 6
def valid_positions_for_A : ℕ := 5
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- The total number of ways person A can be placed in any position except the first
def num_ways_A_not_head : ℕ := valid_positions_for_A * factorial (num_people - 1)

-- The theorem to prove
theorem ways_A_not_head_is_600 : num_ways_A_not_head = 600 := by
  sorry

end ways_A_not_head_is_600_l93_93912


namespace cycling_sequences_reappear_after_28_cycles_l93_93902

/-- Cycling pattern of letters and digits. Letter cycle length is 7; digit cycle length is 4.
Prove that the LCM of 7 and 4 is 28, which is the first line on which both sequences will reappear -/
theorem cycling_sequences_reappear_after_28_cycles 
  (letters_cycle_length : ℕ) (digits_cycle_length : ℕ) 
  (h_letters : letters_cycle_length = 7) 
  (h_digits : digits_cycle_length = 4) 
  : Nat.lcm letters_cycle_length digits_cycle_length = 28 :=
by
  rw [h_letters, h_digits]
  sorry

end cycling_sequences_reappear_after_28_cycles_l93_93902


namespace length_of_faster_train_is_380_meters_l93_93924

-- Defining the conditions
def speed_faster_train_kmph := 144
def speed_slower_train_kmph := 72
def time_seconds := 19

-- Conversion factor
def kmph_to_mps (speed : Nat) : Nat := speed * 1000 / 3600

-- Relative speed in m/s
def relative_speed_mps : Nat := kmph_to_mps (speed_faster_train_kmph - speed_slower_train_kmph)

-- Problem statement: Prove that the length of the faster train is 380 meters
theorem length_of_faster_train_is_380_meters :
  relative_speed_mps * time_seconds = 380 :=
sorry

end length_of_faster_train_is_380_meters_l93_93924


namespace minimize_product_l93_93352

theorem minimize_product
    (a b c : ℕ) 
    (h_positive: a > 0 ∧ b > 0 ∧ c > 0)
    (h_eq: 10 * a^2 - 3 * a * b + 7 * c^2 = 0) : 
    (gcd a b) * (gcd b c) * (gcd c a) = 3 :=
sorry

end minimize_product_l93_93352


namespace track_champion_races_l93_93491

theorem track_champion_races (total_sprinters : ℕ) (lanes : ℕ) (eliminations_per_race : ℕ)
  (h1 : total_sprinters = 216) (h2 : lanes = 6) (h3 : eliminations_per_race = 5) : 
  (total_sprinters - 1) / eliminations_per_race = 43 :=
by
  -- We acknowledge that a proof is needed here. Placeholder for now.
  sorry

end track_champion_races_l93_93491


namespace chessboard_accessible_squares_l93_93782

def is_accessible (board_size : ℕ) (central_exclusion_count : ℕ) (total_squares central_inaccessible : ℕ) : Prop :=
  total_squares = board_size * board_size ∧
  central_inaccessible = central_exclusion_count + 1 + 14 + 14 ∧
  board_size = 15 ∧
  total_squares - central_inaccessible = 196

theorem chessboard_accessible_squares :
  is_accessible 15 29 225 29 :=
by {
  sorry
}

end chessboard_accessible_squares_l93_93782


namespace chess_tournament_third_place_wins_l93_93636

theorem chess_tournament_third_place_wins :
  ∀ (points : Fin 8 → ℕ)
  (total_games : ℕ)
  (total_points : ℕ),
  (total_games = 28) →
  (∀ i j : Fin 8, i ≠ j → points i ≠ points j) →
  ((points 1) = (points 4 + points 5 + points 6 + points 7)) →
  (points 2 > points 4) →
  ∃ (games_won : Fin 8 → Fin 8 → Prop),
  (games_won 2 4) :=
by
  sorry

end chess_tournament_third_place_wins_l93_93636


namespace odd_and_increasing_f1_odd_and_increasing_f2_l93_93564

-- Define the functions
def f1 (x : ℝ) : ℝ := x * |x|
def f2 (x : ℝ) : ℝ := x^3

-- Define the odd function property
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

-- Define the increasing function property
def is_increasing (f : ℝ → ℝ) : Prop := ∀ ⦃x1 x2 : ℝ⦄, x1 < x2 → f x1 < f x2

-- Lean statement to prove
theorem odd_and_increasing_f1 : is_odd f1 ∧ is_increasing f1 := by
  sorry

theorem odd_and_increasing_f2 : is_odd f2 ∧ is_increasing f2 := by
  sorry

end odd_and_increasing_f1_odd_and_increasing_f2_l93_93564


namespace solve_for_y_l93_93136

theorem solve_for_y (x y : ℝ) (h1 : x * y = 9) (h2 : x / y = 36) (h3 : x > 0) (h4 : y > 0) : y = 1/2 := 
by
  sorry

end solve_for_y_l93_93136


namespace total_coins_correct_l93_93835

-- Define basic parameters
def stacks_pennies : Nat := 3
def coins_per_penny_stack : Nat := 10
def stacks_nickels : Nat := 5
def coins_per_nickel_stack : Nat := 8
def stacks_dimes : Nat := 7
def coins_per_dime_stack : Nat := 4

-- Calculate total coins for each type
def total_pennies : Nat := stacks_pennies * coins_per_penny_stack
def total_nickels : Nat := stacks_nickels * coins_per_nickel_stack
def total_dimes : Nat := stacks_dimes * coins_per_dime_stack

-- Calculate total number of coins
def total_coins : Nat := total_pennies + total_nickels + total_dimes

-- Proof statement
theorem total_coins_correct : total_coins = 98 := by
  -- Proof steps go here (omitted)
  sorry

end total_coins_correct_l93_93835


namespace probability_exactly_two_sunny_days_l93_93179

-- Define the conditions
def rain_probability : ℝ := 0.8
def sun_probability : ℝ := 1 - rain_probability
def days : ℕ := 5
def sunny_days : ℕ := 2
def rainy_days : ℕ := days - sunny_days

-- Define the combinatorial and probability calculations
def comb (n k : ℕ) : ℕ := Nat.choose n k
def probability_sunny_days : ℝ := comb days sunny_days * (sun_probability ^ sunny_days) * (rain_probability ^ rainy_days)

theorem probability_exactly_two_sunny_days : probability_sunny_days = 51 / 250 := by
  sorry

end probability_exactly_two_sunny_days_l93_93179


namespace perimeter_of_polygon_l93_93762

-- Define the dimensions of the strips and their arrangement
def strip_width : ℕ := 4
def strip_length : ℕ := 16
def num_vertical_strips : ℕ := 2
def num_horizontal_strips : ℕ := 2

-- State the problem condition and the expected perimeter
theorem perimeter_of_polygon : 
  let vertical_perimeter := num_vertical_strips * strip_length
  let horizontal_perimeter := num_horizontal_strips * strip_length
  let corner_segments_perimeter := (num_vertical_strips + num_horizontal_strips) * strip_width
  vertical_perimeter + horizontal_perimeter + corner_segments_perimeter = 80 :=
by
  sorry

end perimeter_of_polygon_l93_93762


namespace max_diameters_l93_93934

theorem max_diameters (n : ℕ) (points : Finset (ℝ × ℝ)) (h : n ≥ 3) (hn : points.card = n)
  (d : ℝ) (h_d_max : ∀ {p q : ℝ × ℝ}, p ∈ points → q ∈ points → dist p q ≤ d) :
  ∃ m : ℕ, m ≤ n ∧ (∀ {p q : ℝ × ℝ}, p ∈ points → q ∈ points → dist p q = d → m ≤ n) := 
sorry

end max_diameters_l93_93934


namespace regular_polygon_sides_l93_93901

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (k : ℕ), (k : ℕ) * 18 = 360) : n = 20 :=
by
  -- Proof body here
  sorry

end regular_polygon_sides_l93_93901


namespace divides_six_ab_l93_93021

theorem divides_six_ab 
  (a b n : ℕ) 
  (hb : b < 10) 
  (hn : n > 3) 
  (h_eq : 2^n = 10 * a + b) : 
  6 ∣ (a * b) :=
sorry

end divides_six_ab_l93_93021


namespace ratio_of_money_given_l93_93752

theorem ratio_of_money_given
  (T : ℕ) (W : ℕ) (Th : ℕ) (m : ℕ)
  (h1 : T = 8) 
  (h2 : W = m * T) 
  (h3 : Th = W + 9)
  (h4 : Th = T + 41) : 
  W / T = 5 := 
sorry

end ratio_of_money_given_l93_93752


namespace smallest_even_sum_equals_200_l93_93641

theorem smallest_even_sum_equals_200 :
  ∃ (x : ℤ), (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 200) ∧ (x = 36) :=
by
  sorry

end smallest_even_sum_equals_200_l93_93641


namespace solution_l93_93161

namespace Proof

open Set

def proof_problem : Prop :=
  let U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {3, 4, 5, 6}
  A ∩ (U \ B) = {1, 2}

theorem solution : proof_problem := by
  -- The pre-defined proof_problem must be shown here
  -- Proof: sorry
  sorry

end Proof

end solution_l93_93161


namespace homothety_maps_C_to_E_l93_93366

-- Defining Points and Circles
variable {Point Circle : Type}
variable [Inhabited Point] -- assuming Point type is inhabited

-- Definitions for points H, K_A, I_A, K_B, I_B, K_C, I_C
variables (H K_A I_A K_B I_B K_C I_C : Point)

-- Define midpoints
def is_midpoint (A B M : Point) : Prop := sorry -- In a real proof, you would define midpoint in terms of coordinates

-- Define homothety function
def homothety (center : Point) (ratio : ℝ) (P : Point) : Point := sorry -- In a real proof, you would define the homothety transformation

-- Defining Circles
variables (C E : Circle)

-- Define circumcircle of a triangle
def is_circumcircle (a b c : Point) (circle : Circle) : Prop := sorry

-- Statements from conditions
axiom midpointA : is_midpoint H K_A I_A
axiom midpointB : is_midpoint H K_B I_B
axiom midpointC : is_midpoint H K_C I_C

axiom circumcircle_C : is_circumcircle K_A K_B K_C C
axiom circumcircle_E : is_circumcircle I_A I_B I_C E

-- Lean theorem stating the proof problem
theorem homothety_maps_C_to_E :
  ∀ (H K_A I_A K_B I_B K_C I_C : Point) (C E : Circle),
  (is_midpoint H K_A I_A) →
  (is_midpoint H K_B I_B) →
  (is_midpoint H K_C I_C) →
  (is_circumcircle K_A K_B K_C C) →
  (is_circumcircle I_A I_B I_C E) →
  (homothety H 0.5 K_A = I_A ) →
  (homothety H 0.5 K_B = I_B ) →
  (homothety H 0.5 K_C = I_C ) →
  C = E :=
by intro; sorry

end homothety_maps_C_to_E_l93_93366


namespace minimum_value_of_f_l93_93128

noncomputable def f (x : ℝ) : ℝ := (x^2 + 9) / Real.sqrt (x^2 + 5)

theorem minimum_value_of_f : ∀ x : ℝ, f x ≥ 6 :=
by 
  sorry

end minimum_value_of_f_l93_93128


namespace non_zero_number_is_nine_l93_93347

theorem non_zero_number_is_nine {x : ℝ} (h1 : (x + x^2) / 2 = 5 * x) (h2 : x ≠ 0) : x = 9 :=
by
  sorry

end non_zero_number_is_nine_l93_93347


namespace expression_value_eq_3084_l93_93420

theorem expression_value_eq_3084 (x : ℤ) (hx : x = -3007) :
  (abs (abs (Real.sqrt (abs x - x) - x) - x) - Real.sqrt (abs (x - x^2)) = 3084) :=
by
  sorry

end expression_value_eq_3084_l93_93420


namespace am_hm_inequality_l93_93294

theorem am_hm_inequality (a1 a2 a3 : ℝ) (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) (h_sum : a1 + a2 + a3 = 1) : 
  (1 / a1) + (1 / a2) + (1 / a3) ≥ 9 :=
by
  sorry

end am_hm_inequality_l93_93294


namespace monotonically_increasing_interval_l93_93485

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x) / Real.log (1/2)

theorem monotonically_increasing_interval :
  ∀ x, x ∈ Set.Iio (0 : ℝ) → StrictMono f :=
by
  sorry

end monotonically_increasing_interval_l93_93485


namespace Jungkook_has_bigger_number_l93_93948

theorem Jungkook_has_bigger_number : (3 + 6) > 4 :=
by {
  sorry
}

end Jungkook_has_bigger_number_l93_93948


namespace arithmetic_sequence_a17_l93_93898

theorem arithmetic_sequence_a17 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : S 13 = 78)
  (h2 : a 7 + a 12 = 10)
  (h_sum : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 1 + (a 2 - a 1) / (2 - 1)))
  (h_term : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1) / (2 - 1)) :
  a 17 = 2 :=
by
  sorry

end arithmetic_sequence_a17_l93_93898


namespace simplify_expression_l93_93380

theorem simplify_expression (x y : ℝ) : 7 * x + 8 * y - 3 * x + 4 * y + 10 = 4 * x + 12 * y + 10 :=
by
  sorry

end simplify_expression_l93_93380


namespace max_value_of_f_l93_93532

noncomputable def f (x : ℝ) : ℝ := (1/5) * Real.sin (x + Real.pi/3) + Real.cos (x - Real.pi/6)

theorem max_value_of_f : ∀ x : ℝ, f x ≤ 6/5 := by
  sorry

end max_value_of_f_l93_93532


namespace train_speed_first_part_l93_93068

theorem train_speed_first_part (x v : ℝ) (h1 : 0 < x) (h2 : 0 < v) 
  (h_avg_speed : (3 * x) / (x / v + 2 * x / 20) = 22.5) : v = 30 :=
sorry

end train_speed_first_part_l93_93068


namespace calculation_result_l93_93085

theorem calculation_result : 3 * 11 + 3 * 12 + 3 * 15 + 11 = 125 := 
by
  sorry

end calculation_result_l93_93085


namespace factorize_2mn_cube_arithmetic_calculation_l93_93591

-- Problem 1: Factorization problem
theorem factorize_2mn_cube (m n : ℝ) : 
  2 * m^3 * n - 8 * m * n^3 = 2 * m * n * (m + 2 * n) * (m - 2 * n) :=
by sorry

-- Problem 2: Arithmetic calculation problem
theorem arithmetic_calculation : 
  |1 - Real.sqrt 3| + 3 * Real.tan (Real.pi / 6) - ((Real.pi - 3)^0) + (-1/3)⁻¹ = 2 * Real.sqrt 3 - 5 :=
by sorry

end factorize_2mn_cube_arithmetic_calculation_l93_93591


namespace time_to_pass_bridge_l93_93404

noncomputable def train_length : Real := 357
noncomputable def speed_km_per_hour : Real := 42
noncomputable def bridge_length : Real := 137

noncomputable def speed_m_per_s : Real := speed_km_per_hour * (1000 / 3600)

noncomputable def total_distance : Real := train_length + bridge_length

noncomputable def time_to_pass : Real := total_distance / speed_m_per_s

theorem time_to_pass_bridge : abs (time_to_pass - 42.33) < 0.01 :=
sorry

end time_to_pass_bridge_l93_93404


namespace vector_addition_correct_l93_93640

variables (a b : ℝ × ℝ)
def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (-1, 2)

theorem vector_addition_correct : vector_a + vector_b = (1, 5) :=
by
  -- Assume a and b are vectors in 2D space
  have a := vector_a
  have b := vector_b
  -- By definition of vector addition
  sorry

end vector_addition_correct_l93_93640


namespace fred_games_last_year_proof_l93_93324

def fred_games_last_year (this_year: ℕ) (diff: ℕ) : ℕ := this_year + diff

theorem fred_games_last_year_proof : 
  ∀ (this_year: ℕ) (diff: ℕ),
  this_year = 25 → 
  diff = 11 →
  fred_games_last_year this_year diff = 36 := 
by 
  intros this_year diff h_this_year h_diff
  rw [h_this_year, h_diff]
  sorry

end fred_games_last_year_proof_l93_93324


namespace find_x_l93_93986

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : ⌈x⌉ * x = 220) : x = 14.67 :=
sorry

end find_x_l93_93986


namespace cost_of_greenhouses_possible_renovation_plans_l93_93060

noncomputable def cost_renovation (x y : ℕ) : Prop :=
  (2 * x = y + 6) ∧ (x + 2 * y = 48)

theorem cost_of_greenhouses : ∃ x y, cost_renovation x y ∧ x = 12 ∧ y = 18 :=
by {
  sorry
}

noncomputable def renovation_plan (m : ℕ) : Prop :=
  (5 * m + 3 * (8 - m) ≤ 35) ∧ (12 * m + 18 * (8 - m) ≤ 128)

theorem possible_renovation_plans : ∃ m, renovation_plan m ∧ (m = 3 ∨ m = 4 ∨ m = 5) :=
by {
  sorry
}

end cost_of_greenhouses_possible_renovation_plans_l93_93060


namespace octagon_diag_20_algebraic_expr_positive_l93_93575

def octagon_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem octagon_diag_20 : octagon_diagonals 8 = 20 := by
  -- Formula for diagonals is used here
  sorry

theorem algebraic_expr_positive (x : ℝ) : 2 * x^2 - 2 * x + 1 > 0 := by
  -- Complete the square to show it's always positive
  sorry

end octagon_diag_20_algebraic_expr_positive_l93_93575


namespace most_stable_performance_l93_93077

-- Given variances for the four people
def S_A_var : ℝ := 0.56
def S_B_var : ℝ := 0.60
def S_C_var : ℝ := 0.50
def S_D_var : ℝ := 0.45

-- We need to prove that the variance for D is the smallest
theorem most_stable_performance :
  S_D_var < S_C_var ∧ S_D_var < S_A_var ∧ S_D_var < S_B_var :=
by
  sorry

end most_stable_performance_l93_93077


namespace quadratic_inequality_solution_l93_93893

theorem quadratic_inequality_solution (x : ℝ) :
  3 * x^2 - 2 * x - 8 ≤ 0 ↔ -4/3 ≤ x ∧ x ≤ 2 :=
sorry

end quadratic_inequality_solution_l93_93893


namespace range_of_a_max_value_of_z_l93_93193

variable (a b : ℝ)

-- Definition of the assumptions
def condition1 := (2 * a + b = 9)
def condition2 := (|9 - b| + |a| < 3)
def condition3 := (a > 0)
def condition4 := (b > 0)
def z := a^2 * b

-- Statement for problem (i)
theorem range_of_a (h1 : condition1 a b) (h2 : condition2 a b) : -1 < a ∧ a < 1 := sorry

-- Statement for problem (ii)
theorem max_value_of_z (h1 : condition1 a b) (h2 : condition3 a) (h3 : condition4 b) : 
  z a b = 27 := sorry

end range_of_a_max_value_of_z_l93_93193


namespace chocolate_chip_cookies_count_l93_93844

theorem chocolate_chip_cookies_count (h1 : 5 / 2 = 20 / (x : ℕ)) : x = 8 := 
by
  sorry -- Proof to be implemented

end chocolate_chip_cookies_count_l93_93844


namespace youngest_child_age_l93_93807

theorem youngest_child_age (x : ℕ) 
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 50) : x = 4 := 
by 
  sorry

end youngest_child_age_l93_93807


namespace arith_seq_s14_gt_0_l93_93673

variable {S : ℕ → ℝ} -- S_n is the sum of the first n terms of an arithmetic sequence
variable {a : ℕ → ℝ} -- a_n is the nth term of the arithmetic sequence
variable {d : ℝ} -- d is the common difference of the arithmetic sequence

-- Conditions
variable (a_7_lt_0 : a 7 < 0)
variable (a_5_plus_a_10_gt_0 : a 5 + a 10 > 0)

-- Assertion
theorem arith_seq_s14_gt_0 (a_7_lt_0 : a 7 < 0) (a_5_plus_a_10_gt_0 : a 5 + a 10 > 0) : S 14 > 0 := by
  sorry

end arith_seq_s14_gt_0_l93_93673


namespace complex_combination_l93_93090

open Complex

def a : ℂ := 2 - I
def b : ℂ := -1 + I

theorem complex_combination : 2 * a + 3 * b = 1 + I :=
by
  -- Proof goes here
  sorry

end complex_combination_l93_93090


namespace two_circles_common_tangents_l93_93358

theorem two_circles_common_tangents (r : ℝ) (h_r : 0 < r) :
  ¬ ∃ (n : ℕ), n = 2 ∧
  (∀ (config : ℕ), 
    (config = 0 → n = 4) ∨
    (config = 1 → n = 0) ∨
    (config = 2 → n = 3) ∨
    (config = 3 → n = 1)) :=
by
  sorry

end two_circles_common_tangents_l93_93358


namespace intersection_P_Q_l93_93642

def P : Set ℤ := {-4, -2, 0, 2, 4}
def Q : Set ℤ := {x : ℤ | -1 < x ∧ x < 3}

theorem intersection_P_Q : P ∩ Q = {0, 2} := by
  sorry

end intersection_P_Q_l93_93642


namespace total_people_in_line_l93_93581

theorem total_people_in_line (n_front n_behind : ℕ) (hfront : n_front = 11) (hbehind : n_behind = 12) : n_front + n_behind + 1 = 24 := by
  sorry

end total_people_in_line_l93_93581


namespace inequality_holds_for_all_x_l93_93841

variable (p : ℝ)
variable (x : ℝ)

theorem inequality_holds_for_all_x (h : -3 < p ∧ p < 6) : 
  -9 < (3*x^2 + p*x - 6) / (x^2 - x + 1) ∧ (3*x^2 + p*x - 6) / (x^2 - x + 1) < 6 := by
  sorry

end inequality_holds_for_all_x_l93_93841


namespace cloud_height_l93_93456

/--
Given:
- α : ℝ (elevation angle from the top of a tower)
- β : ℝ (depression angle seen in the lake)
- m : ℝ (height of the tower)
Prove:
- The height of the cloud hovering above the observer (h - m) is given by
 2 * m * cos β * sin α / sin (β - α)
-/
theorem cloud_height (α β m : ℝ) :
  (∃ h : ℝ, h - m = 2 * m * Real.cos β * Real.sin α / Real.sin (β - α)) :=
by
  sorry

end cloud_height_l93_93456


namespace limit_hours_overtime_l93_93133

theorem limit_hours_overtime (R O : ℝ) (earnings total_hours : ℕ) (L : ℕ) 
    (hR : R = 16)
    (hO : O = R + 0.75 * R)
    (h_earnings : earnings = 864)
    (h_total_hours : total_hours = 48)
    (calc_earnings : earnings = L * R + (total_hours - L) * O) :
    L = 40 := by
  sorry

end limit_hours_overtime_l93_93133


namespace students_preferring_windows_is_correct_l93_93457

-- Define the total number of students surveyed
def total_students : ℕ := 210

-- Define the number of students preferring Mac
def students_preferring_mac : ℕ := 60

-- Define the number of students preferring both Mac and Windows equally
def students_preferring_both : ℕ := students_preferring_mac / 3

-- Define the number of students with no preference
def students_no_preference : ℕ := 90

-- Calculate the total number of students with a preference
def students_with_preference : ℕ := total_students - students_no_preference

-- Calculate the number of students preferring Windows
def students_preferring_windows : ℕ := students_with_preference - (students_preferring_mac + students_preferring_both)

-- State the theorem to prove that the number of students preferring Windows is 40
theorem students_preferring_windows_is_correct : students_preferring_windows = 40 :=
by
  -- calculations based on definitions
  unfold students_preferring_windows students_with_preference students_preferring_mac students_preferring_both students_no_preference total_students
  sorry

end students_preferring_windows_is_correct_l93_93457


namespace felipe_total_time_l93_93885

-- Given definitions
def combined_time_without_breaks := 126
def combined_time_with_breaks := 150
def felipe_break := 6
def emilio_break := 2 * felipe_break
def carlos_break := emilio_break / 2

theorem felipe_total_time (F E C : ℕ) 
(h1 : F = E / 2) 
(h2 : C = F + E)
(h3 : (F + E + C) = combined_time_without_breaks)
(h4 : (F + felipe_break) + (E + emilio_break) + (C + carlos_break) = combined_time_with_breaks) : 
F + felipe_break = 27 := 
sorry

end felipe_total_time_l93_93885


namespace Kenny_running_to_basketball_ratio_l93_93152

theorem Kenny_running_to_basketball_ratio (basketball_hours trumpet_hours running_hours : ℕ) 
    (h1 : basketball_hours = 10)
    (h2 : trumpet_hours = 2 * running_hours)
    (h3 : trumpet_hours = 40) :
    running_hours = 20 ∧ basketball_hours = 10 ∧ (running_hours / basketball_hours = 2) :=
by
  sorry

end Kenny_running_to_basketball_ratio_l93_93152


namespace ram_leela_money_next_week_l93_93369

theorem ram_leela_money_next_week (x : ℕ)
  (initial_money : ℕ := 100)
  (total_money_after_52_weeks : ℕ := 1478)
  (sum_of_series : ℕ := 1378) :
  let n := 52
  let a1 := x
  let an := x + 51
  let S := (n / 2) * (a1 + an)
  initial_money + S = total_money_after_52_weeks → x = 1 :=
by
  sorry

end ram_leela_money_next_week_l93_93369


namespace negation_of_exists_x_squared_gt_one_l93_93441

-- Negation of the proposition
theorem negation_of_exists_x_squared_gt_one :
  ¬ (∃ x : ℝ, x^2 > 1) ↔ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 :=
by
  sorry

end negation_of_exists_x_squared_gt_one_l93_93441


namespace rational_ordering_l93_93916

theorem rational_ordering :
  (-3:ℚ)^2 < -1/3 ∧ (-1/3 < ((-3):ℚ)^2 ∧ ((-3:ℚ)^2 = |((-3:ℚ))^2|)) := 
by 
  sorry

end rational_ordering_l93_93916


namespace middle_digit_base8_l93_93200

theorem middle_digit_base8 (M : ℕ) (e : ℕ) (d f : Fin 8) 
  (M_base8 : M = 64 * d + 8 * e + f)
  (M_base10 : M = 100 * f + 10 * e + d) :
  e = 6 :=
by sorry

end middle_digit_base8_l93_93200


namespace sum_of_xyz_l93_93238

theorem sum_of_xyz (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 18) (hxz : x * z = 3) (hyz : y * z = 6) : x + y + z = 10 := 
sorry

end sum_of_xyz_l93_93238


namespace simplify_fraction_l93_93438

theorem simplify_fraction
  (a b c : ℝ)
  (h : 2 * a - 3 * c - 4 - b ≠ 0)
  : (6 * a ^ 2 - 2 * b ^ 2 + 6 * c ^ 2 + a * b - 13 * a * c - 4 * b * c - 18 * a - 5 * b + 17 * c + 12) /
    (4 * a ^ 2 - b ^ 2 + 9 * c ^ 2 - 12 * a * c - 16 * a + 24 * c + 16) =
    (3 * a - 2 * c - 3 + 2 * b) / (2 * a - 3 * c - 4 + b) :=
  sorry

end simplify_fraction_l93_93438


namespace find_a_if_even_function_l93_93765

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2 * a^2 - a) * x + 1

theorem find_a_if_even_function (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 1 / 2 := by
  sorry

end find_a_if_even_function_l93_93765


namespace student_second_subject_percentage_l93_93925

theorem student_second_subject_percentage (x : ℝ) (h : (50 + x + 90) / 3 = 70) : x = 70 :=
by { sorry }

end student_second_subject_percentage_l93_93925


namespace problem_statement_l93_93568

-- Define the functions
def f (x : ℤ) : ℤ := x^2
def g (x : ℤ) : ℤ := 2 * x - 5

-- Define the main theorem statement
theorem problem_statement : f (g (-2)) = 81 := by
  sorry

end problem_statement_l93_93568


namespace tetrahedron_cross_section_area_l93_93511

theorem tetrahedron_cross_section_area (a : ℝ) : 
  ∃ (S : ℝ), 
    let AB := a; 
    let AC := a;
    let AD := a;
    S = (3 * a^2) / 8 
    := sorry

end tetrahedron_cross_section_area_l93_93511


namespace company_percentage_increase_l93_93447

theorem company_percentage_increase (employees_jan employees_dec : ℝ) (P_increase : ℝ) 
  (h_jan : employees_jan = 391.304347826087)
  (h_dec : employees_dec = 450)
  (h_P : P_increase = 15) : 
  (employees_dec - employees_jan) / employees_jan * 100 = P_increase :=
by 
  sorry

end company_percentage_increase_l93_93447


namespace root_interval_l93_93788

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 4 * x - 3

theorem root_interval (x0 : ℝ) (h : f x0 = 0): x0 ∈ Set.Ioo (1 / 4 : ℝ) (1 / 2 : ℝ) :=
by
  sorry

end root_interval_l93_93788


namespace lily_milk_left_l93_93590

theorem lily_milk_left : 
  let initial_milk := 5 
  let given_to_james := 18 / 7
  ∃ r : ℚ, r = 2 + 3/7 ∧ (initial_milk - given_to_james) = r :=
by
  sorry

end lily_milk_left_l93_93590


namespace hashN_of_25_l93_93181

def hashN (N : ℝ) : ℝ := 0.6 * N + 2

theorem hashN_of_25 : hashN (hashN (hashN (hashN 25))) = 7.592 :=
by
  sorry

end hashN_of_25_l93_93181


namespace seven_segments_impossible_l93_93894

theorem seven_segments_impossible :
  ¬(∃(segments : Fin 7 → Set (Fin 7)), (∀i, ∃ (S : Finset (Fin 7)), S.card = 3 ∧ ∀ j ∈ S, i ≠ j ∧ segments i j) ∧ (∀ i j, i ≠ j → segments i j → segments j i)) :=
sorry

end seven_segments_impossible_l93_93894


namespace total_population_l93_93799

theorem total_population (P : ℝ) : 0.96 * P = 23040 → P = 24000 :=
by
  sorry

end total_population_l93_93799


namespace omega_range_for_monotonically_decreasing_l93_93164

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

theorem omega_range_for_monotonically_decreasing
  (ω : ℝ)
  (hω : ω > 0)
  (h_decreasing : ∀ x ∈ Set.Ioo (Real.pi / 2) Real.pi, f ω x < f ω (x + 1e-6)) :
  1/2 ≤ ω ∧ ω ≤ 5/4 :=
by
  sorry

end omega_range_for_monotonically_decreasing_l93_93164


namespace finitely_many_n_divisors_in_A_l93_93768

-- Lean 4 statement
theorem finitely_many_n_divisors_in_A (A : Finset ℕ) (a : ℕ) (hA : ∀ p ∈ A, Nat.Prime p) (ha : a ≥ 2) :
  ∃ N : ℕ, ∀ n : ℕ, n > N → ∃ p : ℕ, p ∣ a^n - 1 ∧ p ∉ A := by
  sorry

end finitely_many_n_divisors_in_A_l93_93768


namespace find_m_n_l93_93721

noncomputable def A : Set ℝ := {3, 5}
def B (m n : ℝ) : Set ℝ := {x | x^2 + m * x + n = 0}

theorem find_m_n (m n : ℝ) (h_union : A ∪ B m n = A) (h_inter : A ∩ B m n = {5}) :
  m = -10 ∧ n = 25 :=
by
  sorry

end find_m_n_l93_93721


namespace system_solution_equation_solution_l93_93981

-- Proof problem for the first system of equations
theorem system_solution (x y : ℝ) : 
  (2 * x + 3 * y = 8) ∧ (3 * x - 5 * y = -7) → (x = 1 ∧ y = 2) :=
by sorry

-- Proof problem for the second equation
theorem equation_solution (x : ℝ) : 
  ((x - 2) / (x + 2) - 12 / (x^2 - 4) = 1) → (x = -1) :=
by sorry

end system_solution_equation_solution_l93_93981


namespace directrix_of_parabola_l93_93689

noncomputable def parabola_directrix (x : ℝ) : ℝ := 4 * x^2 + 4 * x + 1

theorem directrix_of_parabola :
  ∃ (y : ℝ) (x : ℝ), parabola_directrix x = y ∧ y = 4 * (x + 1/2)^2 + 3/4 ∧ y - 1/16 = 11/16 :=
by
  sorry

end directrix_of_parabola_l93_93689


namespace orange_beads_in_necklace_l93_93201

theorem orange_beads_in_necklace (O : ℕ) : 
    (∀ g w o : ℕ, g = 9 ∧ w = 6 ∧ ∃ t : ℕ, t = 45 ∧ 5 * (g + w + O) = 5 * (9 + 6 + O) ∧ 
    ∃ n : ℕ, n = 5 ∧ n * (45) =
    n * (5 * O)) → O = 9 :=
by
  sorry

end orange_beads_in_necklace_l93_93201


namespace blue_eyes_count_l93_93143

theorem blue_eyes_count (total_students students_both students_neither : ℕ)
  (ratio_blond_to_blue : ℕ → ℕ)
  (h_total : total_students = 40)
  (h_ratio : ratio_blond_to_blue 3 = 2)
  (h_both : students_both = 8)
  (h_neither : students_neither = 5) :
  ∃ y : ℕ, y = 18 :=
by
  sorry

end blue_eyes_count_l93_93143


namespace find_integer_N_l93_93659

theorem find_integer_N : ∃ N : ℤ, (N ^ 2 ≡ N [ZMOD 10000]) ∧ (N - 2 ≡ 0 [ZMOD 7]) :=
by
  sorry

end find_integer_N_l93_93659


namespace beetle_speed_l93_93524

theorem beetle_speed
  (distance_ant : ℝ )
  (time_minutes : ℝ)
  (distance_beetle : ℝ) 
  (distance_percent_less : ℝ)
  (time_hours : ℝ)
  (beetle_speed_kmh : ℝ)
  (h1 : distance_ant = 600)
  (h2 : time_minutes = 10)
  (h3 : time_hours = time_minutes / 60)
  (h4 : distance_percent_less = 0.25)
  (h5 : distance_beetle = distance_ant * (1 - distance_percent_less))
  (h6 : beetle_speed_kmh = distance_beetle / time_hours) : 
  beetle_speed_kmh = 2.7 :=
by 
  sorry

end beetle_speed_l93_93524


namespace polygon_sides_l93_93207

theorem polygon_sides (n : ℕ) 
  (h : (n - 2) * 180 = 2 * 360) : n = 6 :=
sorry

end polygon_sides_l93_93207


namespace find_f_of_3_l93_93241

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (y : ℝ) (h : y > 0) : f ((4 * y + 1) / (y + 1)) = 1 / y

theorem find_f_of_3 : f 3 = 0.5 :=
by
  have y := 2.0
  sorry

end find_f_of_3_l93_93241


namespace original_card_count_l93_93413

theorem original_card_count
  (r b : ℕ)
  (initial_prob_red : (r : ℚ) / (r + b) = 2 / 5)
  (prob_red_after_adding_black : (r : ℚ) / (r + (b + 6)) = 1 / 3) :
  r + b = 30 := sorry

end original_card_count_l93_93413


namespace eval_expr_l93_93638

theorem eval_expr (b c : ℕ) (hb : b = 2) (hc : c = 5) : b^3 * b^4 * c^2 = 3200 :=
by {
  -- the proof is omitted
  sorry
}

end eval_expr_l93_93638


namespace maximize_wz_xy_zx_l93_93349

-- Variables definition
variables {w x y z : ℝ}

-- Main statement
theorem maximize_wz_xy_zx (h_sum : w + x + y + z = 200) (h_nonneg : w ≥ 0 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) :
  (w * z + x * y + z * x) ≤ 7500 :=
sorry

end maximize_wz_xy_zx_l93_93349


namespace distance_between_cities_A_B_l93_93310

-- Define the problem parameters
def train_1_speed : ℝ := 60 -- km/hr
def train_2_speed : ℝ := 75 -- km/hr
def start_time_train_1 : ℝ := 8 -- 8 a.m.
def start_time_train_2 : ℝ := 9 -- 9 a.m.
def meeting_time : ℝ := 12 -- 12 p.m.

-- Define the times each train travels
def hours_train_1_travelled := meeting_time - start_time_train_1
def hours_train_2_travelled := meeting_time - start_time_train_2

-- Calculate the distances covered by each train
def distance_train_1_cover := train_1_speed * hours_train_1_travelled
def distance_train_2_cover := train_2_speed * hours_train_2_travelled

-- Define the total distance between cities A and B
def distance_AB := distance_train_1_cover + distance_train_2_cover

-- The theorem to be proved
theorem distance_between_cities_A_B : distance_AB = 465 := 
  by
    -- placeholder for the proof
    sorry

end distance_between_cities_A_B_l93_93310


namespace min_value_of_y_l93_93474

theorem min_value_of_y (x : ℝ) (hx : x > 0) : (∃ y, y = x + 4 / x^2 ∧ ∀ z, z = x + 4 / x^2 → z ≥ 3) :=
sorry

end min_value_of_y_l93_93474


namespace consecutive_numbers_square_sum_l93_93574

theorem consecutive_numbers_square_sum (n : ℕ) (a b : ℕ) (h1 : 2 * n + 1 = 144169^2)
  (h2 : a = 72084) (h3 : b = a + 1) : a^2 + b^2 = n + 1 :=
by
  sorry

end consecutive_numbers_square_sum_l93_93574


namespace length_LM_in_triangle_l93_93341

theorem length_LM_in_triangle 
  (A B C K L M : Type*) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace K] [MetricSpace L] [MetricSpace M]
  (angle_A: Real) (angle_B: Real) (angle_C: Real)
  (AK: Real) (BL: Real) (MC: Real) (KL: Real) (KM: Real)
  (H1: angle_A = 90) (H2: angle_B = 30) (H3: angle_C = 60) 
  (H4: AK = 4) (H5: BL = 31) (H6: MC = 3) 
  (H7: KL = KM) : 
  (LM = 20) :=
sorry

end length_LM_in_triangle_l93_93341


namespace factorize_expression_l93_93242

theorem factorize_expression (a b : ℝ) :
  4 * a^3 * b - a * b = a * b * (2 * a + 1) * (2 * a - 1) :=
by
  sorry

end factorize_expression_l93_93242


namespace parallel_lines_iff_m_eq_neg2_l93_93720

theorem parallel_lines_iff_m_eq_neg2 (m : ℝ) :
  (∀ x y : ℝ, 2 * x + m * y - 2 * m + 4 = 0 → m * x + 2 * y - m + 2 = 0 ↔ m = -2) :=
sorry

end parallel_lines_iff_m_eq_neg2_l93_93720


namespace students_like_all_three_l93_93554

variables (N : ℕ) (r : ℚ) (j : ℚ) (o : ℕ) (n : ℕ)

-- Number of students in the class
def num_students := N = 40

-- Fraction of students who like Rock
def fraction_rock := r = 1/4

-- Fraction of students who like Jazz
def fraction_jazz := j = 1/5

-- Number of students who like other genres
def num_other_genres := o = 8

-- Number of students who do not like any of the three genres
def num_no_genres := n = 6

---- Proof theorem
theorem students_like_all_three
  (h1 : num_students N)
  (h2 : fraction_rock r)
  (h3 : fraction_jazz j)
  (h4 : num_other_genres o)
  (h5 : num_no_genres n) :
  ∃ z : ℕ, z = 2 := 
sorry

end students_like_all_three_l93_93554


namespace truncated_cone_radius_l93_93531

theorem truncated_cone_radius (R: ℝ) (l: ℝ) (h: 0 < l)
  (h1 : ∃ (r: ℝ), r = (R + 5) / 2 ∧ (5 + r) = (1 / 2) * (R + r))
  : R = 25 :=
sorry

end truncated_cone_radius_l93_93531


namespace hyperbola_asymptotes_iff_l93_93283

def hyperbola_asymptotes_orthogonal (a b c d e f : ℝ) : Prop :=
  a + c = 0

theorem hyperbola_asymptotes_iff (a b c d e f : ℝ) :
  (∃ x y : ℝ, a * x^2 + 2 * b * x * y + c * y^2 + d * x + e * y + f = 0) →
  hyperbola_asymptotes_orthogonal a b c d e f ↔ a + c = 0 :=
by sorry

end hyperbola_asymptotes_iff_l93_93283


namespace divisor_and_remainder_correct_l93_93536

theorem divisor_and_remainder_correct:
  ∃ d r : ℕ, d ≠ 0 ∧ 1270 = 74 * d + r ∧ r = 12 ∧ d = 17 :=
by
  sorry

end divisor_and_remainder_correct_l93_93536


namespace donna_soda_crates_l93_93692

def soda_crates (bridge_limit : ℕ) (truck_empty : ℕ) (crate_weight : ℕ) (dryer_weight : ℕ) (num_dryers : ℕ) (truck_loaded : ℕ) (produce_ratio : ℕ) : ℕ :=
  sorry

theorem donna_soda_crates :
  soda_crates 20000 12000 50 3000 3 24000 2 = 20 :=
sorry

end donna_soda_crates_l93_93692


namespace snow_globes_in_box_l93_93095

theorem snow_globes_in_box (S : ℕ) 
  (h1 : ∀ (box_decorations : ℕ), box_decorations = 4 + 1 + S)
  (h2 : ∀ (num_boxes : ℕ), num_boxes = 12)
  (h3 : ∀ (total_decorations : ℕ), total_decorations = 120) :
  S = 5 :=
by
  sorry

end snow_globes_in_box_l93_93095


namespace cube_edge_factor_l93_93879

theorem cube_edge_factor (e f : ℝ) (h₁ : e > 0) (h₂ : (f * e) ^ 3 = 8 * e ^ 3) : f = 2 :=
by
  sorry

end cube_edge_factor_l93_93879


namespace total_strawberry_weight_l93_93419

def MarcosStrawberries : ℕ := 3
def DadsStrawberries : ℕ := 17

theorem total_strawberry_weight : MarcosStrawberries + DadsStrawberries = 20 := by
  sorry

end total_strawberry_weight_l93_93419


namespace correct_calculation_l93_93519

theorem correct_calculation (x : ℕ) (h : x + 10 = 21) : x * 10 = 110 :=
by
  sorry

end correct_calculation_l93_93519


namespace partial_fraction_sum_zero_l93_93866

theorem partial_fraction_sum_zero
    (A B C D E : ℝ)
    (h : ∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -5 →
    1 = A * (x + 1) * (x + 2) * (x + 3) * (x + 5) +
        B * x * (x + 2) * (x + 3) * (x + 5) +
        C * x * (x + 1) * (x + 3) * (x + 5) +
        D * x * (x + 1) * (x + 2) * (x + 5) +
        E * x * (x + 1) * (x + 2) * (x + 3)) :
    A + B + C + D + E = 0 := by
    sorry

end partial_fraction_sum_zero_l93_93866


namespace calorie_limit_l93_93501

variable (breakfastCalories lunchCalories dinnerCalories extraCalories : ℕ)
variable (plannedCalories : ℕ)

-- Given conditions
axiom breakfast_calories : breakfastCalories = 400
axiom lunch_calories : lunchCalories = 900
axiom dinner_calories : dinnerCalories = 1100
axiom extra_calories : extraCalories = 600

-- To Prove
theorem calorie_limit (h : plannedCalories = (breakfastCalories + lunchCalories + dinnerCalories - extraCalories)) :
  plannedCalories = 1800 := by sorry

end calorie_limit_l93_93501


namespace length_of_first_train_is_correct_l93_93937

noncomputable def length_of_first_train 
  (speed_first_train_kmph : ℝ)
  (length_second_train_m : ℝ)
  (speed_second_train_kmph : ℝ)
  (time_crossing_s : ℝ) : ℝ :=
  let speed_first_train_mps := (speed_first_train_kmph * 1000) / 3600
  let speed_second_train_mps := (speed_second_train_kmph * 1000) / 3600
  let relative_speed_mps := speed_first_train_mps + speed_second_train_mps
  let total_distance_m := relative_speed_mps * time_crossing_s
  total_distance_m - length_second_train_m

theorem length_of_first_train_is_correct :
  length_of_first_train 50 112 82 6 = 108.02 :=
by
  sorry

end length_of_first_train_is_correct_l93_93937


namespace even_parts_impossible_odd_parts_possible_l93_93604

theorem even_parts_impossible (n m : ℕ) (h₁ : n = 1) (h₂ : ∀ k, m = n + 2 * k) : n + 2 * m ≠ 100 := by
  -- Proof omitted
  sorry

theorem odd_parts_possible (n m : ℕ) (h₁ : n = 1) (h₂ : ∀ k, m = n + 2 * k) : ∃ k, n + 2 * k = 2017 := by
  -- Proof omitted
  sorry

end even_parts_impossible_odd_parts_possible_l93_93604


namespace number_of_two_digit_integers_l93_93019

def digits : Finset ℕ := {2, 4, 6, 7, 8}

theorem number_of_two_digit_integers : 
  (digits.card * (digits.card - 1)) = 20 := 
by
  sorry

end number_of_two_digit_integers_l93_93019


namespace arithmetic_sequence_formula_geometric_sequence_sum_formula_l93_93462

noncomputable def arithmetic_sequence_a_n (n : ℕ) : ℤ :=
  sorry

noncomputable def geometric_sequence_T_n (n : ℕ) : ℤ :=
  sorry

theorem arithmetic_sequence_formula :
  (∃ a₃ : ℤ, a₃ = 5) ∧ (∃ S₃ : ℤ, S₃ = 9) →
  -- Suppose we have an arithmetic sequence $a_n$
  (∀ n : ℕ, n ≥ 1 → arithmetic_sequence_a_n n = 2 * n - 1) := 
sorry

theorem geometric_sequence_sum_formula :
  (∃ q : ℤ, q > 0 ∧ q = 3) ∧ (∃ b₃ : ℤ, b₃ = 9) ∧ (∃ T₃ : ℤ, T₃ = 13) →
  -- Suppose we have a geometric sequence $b_n$ where $b_3 = a_5$
  (∀ n : ℕ, n ≥ 1 → geometric_sequence_T_n n = (3 ^ n - 1) / 2) := 
sorry

end arithmetic_sequence_formula_geometric_sequence_sum_formula_l93_93462


namespace sufficient_not_necessary_condition_x_plus_a_div_x_geq_2_l93_93613

open Real

theorem sufficient_not_necessary_condition_x_plus_a_div_x_geq_2 (x a : ℝ)
  (h₁ : x > 0) :
  (∀ x > 0, x + a / x ≥ 2) → (a = 1) :=
sorry

end sufficient_not_necessary_condition_x_plus_a_div_x_geq_2_l93_93613


namespace range_of_c_over_a_l93_93611

theorem range_of_c_over_a (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + 2 * b + c = 0) :
    -3 < c / a ∧ c / a < -(1 / 3) := 
sorry

end range_of_c_over_a_l93_93611


namespace tony_bought_10_play_doughs_l93_93596

noncomputable def num_play_doughs 
    (lego_cost : ℕ) 
    (sword_cost : ℕ) 
    (play_dough_cost : ℕ) 
    (bought_legos : ℕ) 
    (bought_swords : ℕ) 
    (total_paid : ℕ) : ℕ :=
  let lego_total := lego_cost * bought_legos
  let sword_total := sword_cost * bought_swords
  let total_play_dough_cost := total_paid - (lego_total + sword_total)
  total_play_dough_cost / play_dough_cost

theorem tony_bought_10_play_doughs : 
  num_play_doughs 250 120 35 3 7 1940 = 10 := 
sorry

end tony_bought_10_play_doughs_l93_93596


namespace sequence_a19_l93_93495

theorem sequence_a19 :
  ∃ (a : ℕ → ℝ), a 3 = 2 ∧ a 7 = 1 ∧
    (∃ d : ℝ, ∀ n m : ℕ, (1 / (a n + 1) - 1 / (a m + 1)) / (n - m) = d) →
    a 19 = 0 :=
by sorry

end sequence_a19_l93_93495


namespace acute_angle_sum_l93_93900

theorem acute_angle_sum (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
    (h1 : Real.sin α = (2 * Real.sqrt 5) / 5) (h2 : Real.sin β = (3 * Real.sqrt 10) / 10) :
    α + β = 3 * Real.pi / 4 :=
sorry

end acute_angle_sum_l93_93900


namespace find_all_n_l93_93950

theorem find_all_n (n : ℕ) : 
  (∀ k : ℤ, ∃ a : ℤ, (a^3 + a - k) % n = 0) ↔ (∃ j : ℕ, n = 3^j) :=
by 
  -- proof goes here
  sorry

end find_all_n_l93_93950


namespace alpha_beta_working_together_time_l93_93145

theorem alpha_beta_working_together_time
  (A B C : ℝ)
  (h : ℝ)
  (hA : A = B + 5)
  (work_together_A : A > 0)
  (work_together_B : B > 0)
  (work_together_C : C > 0)
  (combined_work : 1/A + 1/B + 1/C = 1/(A - 6))
  (combined_work2 : 1/A + 1/B + 1/C = 1/(B - 1))
  (time_gamma : 1/A + 1/B + 1/C = 2/C) :
  h = 4/3 :=
sorry

end alpha_beta_working_together_time_l93_93145


namespace other_number_of_given_conditions_l93_93097

theorem other_number_of_given_conditions 
  (a b : ℕ) 
  (h_lcm : Nat.lcm a b = 4620) 
  (h_gcd : Nat.gcd a b = 21) 
  (h_a : a = 210) : 
  b = 462 := 
sorry

end other_number_of_given_conditions_l93_93097


namespace sean_less_points_than_combined_l93_93290

def tobee_points : ℕ := 4
def jay_points : ℕ := tobee_points + 6
def combined_points_tobee_jay : ℕ := tobee_points + jay_points
def total_team_points : ℕ := 26
def sean_points : ℕ := total_team_points - combined_points_tobee_jay

theorem sean_less_points_than_combined : (combined_points_tobee_jay - sean_points) = 2 := by
  sorry

end sean_less_points_than_combined_l93_93290


namespace least_add_to_divisible_by_17_l93_93631

/-- Given that the remainder when 433124 is divided by 17 is 2,
    prove that the least number that must be added to 433124 to make 
    it divisible by 17 is 15. -/
theorem least_add_to_divisible_by_17: 
  (433124 % 17 = 2) → 
  (∃ n, n ≥ 0 ∧ (433124 + n) % 17 = 0 ∧ n = 15) := 
by
  sorry

end least_add_to_divisible_by_17_l93_93631


namespace cab_speed_fraction_l93_93857

theorem cab_speed_fraction (S R : ℝ) (h1 : S * 40 = R * 48) : (R / S) = (5 / 6) :=
sorry

end cab_speed_fraction_l93_93857


namespace matrix_not_invertible_l93_93878

noncomputable def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem matrix_not_invertible (x : ℝ) :
  determinant (2*x + 1) 9 (4 - x) 10 = 0 ↔ x = 26/29 := by
  sorry

end matrix_not_invertible_l93_93878


namespace average_test_score_before_dropping_l93_93997

theorem average_test_score_before_dropping (A B C : ℝ) :
  (A + B + C) / 3 = 40 → (A + B + C + 20) / 4 = 35 :=
by
  intros h
  sorry

end average_test_score_before_dropping_l93_93997


namespace problem_statement_l93_93052

noncomputable def equation_of_altitude (A B C: (ℝ × ℝ)): (ℝ × ℝ × ℝ) :=
by
  sorry

theorem problem_statement :
  let A := (-1, 4)
  let B := (-2, -1)
  let C := (2, 3)
  equation_of_altitude A B C = (1, 1, -3) ∧
  |1 / 2 * (4 - (-1)) * 4| = 8 :=
by
  sorry

end problem_statement_l93_93052


namespace intercepts_l93_93704

def line_equation (x y : ℝ) : Prop :=
  5 * x + 3 * y - 15 = 0

theorem intercepts (a b : ℝ) : line_equation a 0 ∧ line_equation 0 b → (a = 3 ∧ b = 5) :=
  sorry

end intercepts_l93_93704


namespace car_interval_length_l93_93231

theorem car_interval_length (S1 T : ℝ) (interval_length : ℝ) 
  (h1 : S1 = 39) 
  (h2 : (fun (n : ℕ) => S1 - 3 * n) 4 = 27)
  (h3 : 3.6 = 27 * T) 
  (h4 : interval_length = T * 60) :
  interval_length = 8 :=
by
  sorry

end car_interval_length_l93_93231


namespace evaluate_expression_l93_93804

theorem evaluate_expression (x : Real) (hx : x = -52.7) : 
  ⌈(⌊|x|⌋ + ⌈|x|⌉)⌉ = 105 := by
  sorry

end evaluate_expression_l93_93804


namespace games_bought_l93_93144

/-- 
Given:
1. Geoffrey received €20 from his grandmother.
2. Geoffrey received €25 from his aunt.
3. Geoffrey received €30 from his uncle.
4. Geoffrey now has €125 in his wallet.
5. Geoffrey has €20 left after buying games.
6. Each game costs €35.

Prove that Geoffrey bought 3 games.
-/
theorem games_bought 
  (grandmother_money aunt_money uncle_money total_money left_money game_cost spent_money games_bought : ℤ)
  (h1 : grandmother_money = 20)
  (h2 : aunt_money = 25)
  (h3 : uncle_money = 30)
  (h4 : total_money = 125)
  (h5 : left_money = 20)
  (h6 : game_cost = 35)
  (h7 : spent_money = total_money - left_money)
  (h8 : games_bought = spent_money / game_cost) :
  games_bought = 3 := 
sorry

end games_bought_l93_93144


namespace simplify_expression_l93_93724

theorem simplify_expression (x : ℝ) : 
  x - 2 * (1 + x) + 3 * (1 - x) - 4 * (1 + 2 * x) = -12 * x - 3 := 
by 
  -- Proof goes here
  sorry

end simplify_expression_l93_93724


namespace max_min_diff_eq_l93_93963

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 2*x + 2) - Real.sqrt (x^2 - 3*x + 3)

theorem max_min_diff_eq : 
  (∀ x : ℝ, ∃ max min : ℝ, max = Real.sqrt (8 - Real.sqrt 3) ∧ min = -Real.sqrt (8 - Real.sqrt 3) ∧ 
  (max - min = 2 * Real.sqrt (8 - Real.sqrt 3))) :=
sorry

end max_min_diff_eq_l93_93963


namespace intersection_A_B_l93_93442

def set_A (x : ℝ) : Prop := x^2 - 4 * x - 5 < 0
def set_B (x : ℝ) : Prop := 2 < x ∧ x < 4

theorem intersection_A_B (x : ℝ) :
  (set_A x ∧ set_B x) ↔ 2 < x ∧ x < 4 :=
by sorry

end intersection_A_B_l93_93442


namespace factorize_polynomial_l93_93811

theorem factorize_polynomial (a x : ℝ) : 
  (x^3 - 3*x^2 + (a + 2)*x - 2*a) = (x^2 - x + a)*(x - 2) :=
by
  sorry

end factorize_polynomial_l93_93811


namespace speed_difference_is_36_l93_93069

open Real

noncomputable def alex_speed : ℝ := 8 / (40 / 60)
noncomputable def jordan_speed : ℝ := 12 / (15 / 60)
noncomputable def speed_difference : ℝ := jordan_speed - alex_speed

theorem speed_difference_is_36 : speed_difference = 36 := by
  have hs1 : alex_speed = 8 / (40 / 60) := rfl
  have hs2 : jordan_speed = 12 / (15 / 60) := rfl
  have hd : speed_difference = jordan_speed - alex_speed := rfl
  rw [hs1, hs2] at hd
  simp [alex_speed, jordan_speed, speed_difference] at hd
  sorry

end speed_difference_is_36_l93_93069


namespace average_of_all_digits_l93_93388

theorem average_of_all_digits (d : List ℕ) (h_len : d.length = 9)
  (h1 : (d.take 4).sum = 32)
  (h2 : (d.drop 4).sum = 130) : 
  (d.sum / d.length : ℚ) = 18 := 
by
  sorry

end average_of_all_digits_l93_93388


namespace range_of_m_for_false_proposition_l93_93746

theorem range_of_m_for_false_proposition :
  ¬ (∃ x : ℝ, x^2 - m * x - m ≤ 0) → m ∈ Set.Ioo (-4 : ℝ) 0 :=
sorry

end range_of_m_for_false_proposition_l93_93746


namespace square_area_l93_93936

/- Given: 
    1. The area of the isosceles right triangle ΔAEF is 1 cm².
    2. The area of the rectangle EFGH is 10 cm².
- To prove: 
    The area of the square ABCD is 24.5 cm².
-/

theorem square_area
  (h1 : ∃ a : ℝ, (0 < a) ∧ (a * a / 2 = 1))  -- Area of isosceles right triangle ΔAEF is 1 cm²
  (h2 : ∃ w l : ℝ, (w = 2) ∧ (l * w = 10))  -- Area of rectangle EFGH is 10 cm²
  : ∃ s : ℝ, (s * s = 24.5) := -- Area of the square ABCD is 24.5 cm²
sorry

end square_area_l93_93936


namespace book_prices_l93_93806

theorem book_prices (x : ℝ) (y : ℝ) (h1 : y = 2.5 * x) (h2 : 800 / x - 800 / y = 24) : (x = 20 ∧ y = 50) :=
by
  sorry

end book_prices_l93_93806


namespace square_of_1027_l93_93398

theorem square_of_1027 :
  1027 * 1027 = 1054729 :=
by
  sorry

end square_of_1027_l93_93398


namespace sum_of_x_y_l93_93320

theorem sum_of_x_y (x y : ℚ) (h1 : 1/x + 1/y = 5) (h2 : 1/x - 1/y = -9) : x + y = -5/14 := 
by
  sorry

end sum_of_x_y_l93_93320


namespace distinct_cyclic_quadrilaterals_perimeter_36_l93_93454

noncomputable def count_distinct_cyclic_quadrilaterals : Nat :=
  1026

theorem distinct_cyclic_quadrilaterals_perimeter_36 :
  (∃ (a b c d : ℕ), a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ a + b + c + d = 36 ∧ a < b + c + d) → count_distinct_cyclic_quadrilaterals = 1026 :=
by
  rintro ⟨a, b, c, d, hab, hbc, hcd, hsum, hlut⟩
  sorry

end distinct_cyclic_quadrilaterals_perimeter_36_l93_93454


namespace hospital_cost_minimization_l93_93608

theorem hospital_cost_minimization :
  ∃ (x y : ℕ), (5 * x + 6 * y = 50) ∧ (10 * x + 20 * y = 140) ∧ (2 * x + 3 * y = 23) :=
by
  sorry

end hospital_cost_minimization_l93_93608


namespace interval_monotonically_decreasing_l93_93460

noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 2 * x + 3)

theorem interval_monotonically_decreasing :
  ∀ x y : ℝ, 1 < x → x < 3 → 1 < y → y < 3 → x < y → f y < f x := 
by sorry

end interval_monotonically_decreasing_l93_93460


namespace root_quadratic_l93_93040

theorem root_quadratic (m : ℝ) (h : m^2 - 2*m - 1 = 0) : m^2 + 1/m^2 = 6 :=
sorry

end root_quadratic_l93_93040


namespace smallest_number_l93_93138

theorem smallest_number (a b c d e: ℕ) (h1: a = 5) (h2: b = 8) (h3: c = 1) (h4: d = 2) (h5: e = 6) :
  min (min (min (min a b) c) d) e = 1 :=
by
  -- Proof skipped using sorry
  sorry

end smallest_number_l93_93138


namespace change_factor_w_l93_93695

theorem change_factor_w (w d z F_w : Real)
  (h_q : ∀ w d z, q = 5 * w / (4 * d * z^2))
  (h1 : d' = 2 * d)
  (h2 : z' = 3 * z)
  (h3 : F_q = 0.2222222222222222)
  : F_w = 4 :=
by
  sorry

end change_factor_w_l93_93695


namespace unique_a_exists_iff_n_eq_two_l93_93781

theorem unique_a_exists_iff_n_eq_two (n : ℕ) (h1 : 1 < n) : 
  (∃ a : ℕ, 0 < a ∧ a ≤ n! ∧ n! ∣ a^n + 1 ∧ ∀ b : ℕ, (0 < b ∧ b ≤ n! ∧ n! ∣ b^n + 1) → b = a) ↔ n = 2 := 
by {
  sorry
}

end unique_a_exists_iff_n_eq_two_l93_93781


namespace find_F_16_l93_93576

noncomputable def F : ℝ → ℝ := sorry

lemma F_condition_1 : ∀ x, (x + 4) ≠ 0 ∧ (x + 2) ≠ 0 → (F (4 * x) / F (x + 4) = 16 - (64 * x + 64) / (x^2 + 6 * x + 8)) := sorry

lemma F_condition_2 : F 8 = 33 := sorry

theorem find_F_16 : F 16 = 136 :=
by
  have h1 := F_condition_1
  have h2 := F_condition_2
  sorry

end find_F_16_l93_93576


namespace theater_ticket_difference_l93_93253

theorem theater_ticket_difference
  (O B : ℕ)
  (h1 : O + B = 355)
  (h2 : 12 * O + 8 * B = 3320) :
  B - O = 115 :=
sorry

end theater_ticket_difference_l93_93253


namespace probability_of_smallest_section_l93_93250

-- Define the probabilities for the largest and next largest sections
def P_largest : ℚ := 1 / 2
def P_next_largest : ℚ := 1 / 3

-- Define the total probability constraint
def total_probability (P_smallest : ℚ) : Prop :=
  P_largest + P_next_largest + P_smallest = 1

-- State the theorem to be proved
theorem probability_of_smallest_section : 
  ∃ P_smallest : ℚ, total_probability P_smallest ∧ P_smallest = 1 / 6 := 
by
  sorry

end probability_of_smallest_section_l93_93250


namespace simple_interest_rate_l93_93015

/-- Prove that given Principal (P) = 750, Amount (A) = 900, and Time (T) = 5 years,
    the rate (R) such that the Simple Interest formula holds is 4 percent. -/
theorem simple_interest_rate :
  ∀ (P A T : ℕ) (R : ℕ),
    P = 750 → 
    A = 900 → 
    T = 5 → 
    A = P + (P * R * T / 100) →
    R = 4 :=
by
  intros P A T R hP hA hT h_si
  sorry

end simple_interest_rate_l93_93015


namespace bicycles_purchased_on_Friday_l93_93365

theorem bicycles_purchased_on_Friday (F : ℕ) : (F - 10) - 4 + 2 = 3 → F = 15 := by
  intro h
  sorry

end bicycles_purchased_on_Friday_l93_93365


namespace segment_radius_with_inscribed_equilateral_triangle_l93_93864

theorem segment_radius_with_inscribed_equilateral_triangle (α h : ℝ) : 
  ∃ x : ℝ, x = (h / (Real.sin (α / 2))^2) * (Real.cos (α / 2) + Real.sqrt (1 + (1 / 3) * (Real.sin (α / 2))^2)) :=
sorry

end segment_radius_with_inscribed_equilateral_triangle_l93_93864


namespace incorrect_statement_B_l93_93494

theorem incorrect_statement_B (A B C : ℝ) (hAB : A * B < 0) (hBC : B * C < 0) : ¬ ∀ (x y : ℝ), x * y + A * x + B * y + C = 0 → (x < 0 ∧ y < 0) :=
by
  sorry

end incorrect_statement_B_l93_93494


namespace fraction_remains_unchanged_l93_93846

theorem fraction_remains_unchanged (x y : ℝ) : 
  (3 * (2 * x)) / (2 * (2 * y)) = (3 * x) / (2 * y) :=
by {
  sorry
}

end fraction_remains_unchanged_l93_93846


namespace simplify_expression_l93_93006

theorem simplify_expression (x y z : ℝ) (h1 : x = 3) (h2 : y = 2) (h3 : z = 4) :
  (12 * x^2 * y^3 * z) / (4 * x * y * z^2) = 9 :=
by
  sorry

end simplify_expression_l93_93006


namespace train_stops_15_min_per_hour_l93_93137

/-
Without stoppages, a train travels a certain distance with an average speed of 80 km/h,
and with stoppages, it covers the same distance with an average speed of 60 km/h.
Prove that the train stops for 15 minutes per hour.
-/
theorem train_stops_15_min_per_hour (D : ℝ) (h1 : 0 < D) :
  let T_no_stop := D / 80
  let T_stop := D / 60
  let T_lost := T_stop - T_no_stop
  let mins_per_hour := T_lost * 60
  mins_per_hour = 15 := by
  sorry

end train_stops_15_min_per_hour_l93_93137


namespace whisky_replacement_l93_93471

variable (x : ℝ) -- Original quantity of whisky in the jar
variable (y : ℝ) -- Quantity of whisky replaced

-- Condition: A jar full of whisky contains 40% alcohol
-- Condition: After replacement, the percentage of alcohol is 24%
theorem whisky_replacement (h : 0 < x) : 
  0.40 * x - 0.40 * y + 0.19 * y = 0.24 * x → y = (16 / 21) * x :=
by
  intro h_eq
  -- Sorry for the proof
  sorry

end whisky_replacement_l93_93471


namespace grandpa_tomatoes_before_vacation_l93_93067

theorem grandpa_tomatoes_before_vacation 
  (tomatoes_after_vacation : ℕ) 
  (growth_factor : ℕ) 
  (actual_number : ℕ) 
  (h1 : growth_factor = 100) 
  (h2 : tomatoes_after_vacation = 3564) 
  (h3 : actual_number = tomatoes_after_vacation / growth_factor) : 
  actual_number = 36 := 
by
  -- Here would be the step-by-step proof, but we use sorry to skip it
  sorry

end grandpa_tomatoes_before_vacation_l93_93067


namespace Keenan_essay_length_l93_93903

-- Given conditions
def words_per_hour_first_two_hours : ℕ := 400
def first_two_hours : ℕ := 2
def words_per_hour_later : ℕ := 200
def later_hours : ℕ := 2

-- Total words written in 4 hours
def total_words : ℕ := words_per_hour_first_two_hours * first_two_hours + words_per_hour_later * later_hours

-- Theorem statement
theorem Keenan_essay_length : total_words = 1200 := by
  sorry

end Keenan_essay_length_l93_93903


namespace percentage_increase_l93_93553

theorem percentage_increase (x y P : ℚ)
  (h1 : x = 0.9 * y)
  (h2 : x = 123.75)
  (h3 : y = 125 + 1.25 * P) : 
  P = 10 := 
by 
  sorry

end percentage_increase_l93_93553


namespace boxes_needed_l93_93650

def initial_games : ℕ := 76
def games_sold : ℕ := 46
def games_per_box : ℕ := 5

theorem boxes_needed : (initial_games - games_sold) / games_per_box = 6 := by
  sorry

end boxes_needed_l93_93650


namespace sum_ages_is_13_l93_93580

-- Define the variables for the ages
variables (a b c : ℕ)

-- Define the conditions given in the problem
def conditions : Prop :=
  a * b * c = 72 ∧ a < b ∧ c < b

-- State the theorem to be proved
theorem sum_ages_is_13 (h : conditions a b c) : a + b + c = 13 :=
sorry

end sum_ages_is_13_l93_93580


namespace rectangle_perimeter_l93_93490

theorem rectangle_perimeter (s : ℕ) (h : 4 * s = 160) : 2 * (s + s / 4) = 100 :=
by
  sorry

end rectangle_perimeter_l93_93490


namespace evaluate_expression_l93_93121

theorem evaluate_expression : - (16 / 4 * 7 + 25 - 2 * 7) = -39 :=
by sorry

end evaluate_expression_l93_93121


namespace intersection_M_N_l93_93351

def M : Set ℝ := { x : ℝ | x + 1 ≥ 0 }
def N : Set ℝ := { x : ℝ | x^2 < 4 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | -1 ≤ x ∧ x < 2 } :=
sorry

end intersection_M_N_l93_93351


namespace triangle_formation_ways_l93_93508

-- Given conditions
def parallel_tracks : Prop := true -- The tracks are parallel, implicit condition not affecting calculation
def first_track_checkpoints := 6
def second_track_checkpoints := 10

-- The proof problem
theorem triangle_formation_ways : 
  (first_track_checkpoints * Nat.choose second_track_checkpoints 2) = 270 := by
  sorry

end triangle_formation_ways_l93_93508


namespace correct_average_weight_l93_93539

theorem correct_average_weight 
  (n : ℕ) 
  (w_avg : ℝ) 
  (W_init : ℝ)
  (d1 : ℝ)
  (d2 : ℝ)
  (d3 : ℝ)
  (W_adj : ℝ)
  (w_corr : ℝ)
  (h1 : n = 30)
  (h2 : w_avg = 58.4)
  (h3 : W_init = n * w_avg)
  (h4 : d1 = 62 - 56)
  (h5 : d2 = 59 - 65)
  (h6 : d3 = 54 - 50)
  (h7 : W_adj = W_init + d1 + d2 + d3)
  (h8 : w_corr = W_adj / n) :
  w_corr = 58.5 := 
sorry

end correct_average_weight_l93_93539


namespace probability_event_A_probability_event_B_probability_event_C_l93_93265

-- Define the total number of basic events for three dice
def total_basic_events : ℕ := 6 * 6 * 6

-- Define events and their associated basic events
def event_A_basic_events : ℕ := 2 * 3 * 3
def event_B_basic_events : ℕ := 2 * 3 * 6
def event_C_basic_events : ℕ := 6 * 6 * 3

-- Define probabilities for each event
def P_A : ℚ := event_A_basic_events / total_basic_events
def P_B : ℚ := event_B_basic_events / total_basic_events
def P_C : ℚ := event_C_basic_events / total_basic_events

-- Statement to be proven
theorem probability_event_A : P_A = 1 / 12 := by
  sorry

theorem probability_event_B : P_B = 1 / 6 := by
  sorry

theorem probability_event_C : P_C = 1 / 2 := by
  sorry

end probability_event_A_probability_event_B_probability_event_C_l93_93265


namespace necessary_condition_of_and_is_or_l93_93058

variable (p q : Prop)

theorem necessary_condition_of_and_is_or (hpq : p ∧ q) : p ∨ q :=
by {
    sorry
}

end necessary_condition_of_and_is_or_l93_93058


namespace gcd_45_75_l93_93945

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l93_93945


namespace number_is_100_l93_93549

theorem number_is_100 (n : ℕ) 
  (hquot : n / 11 = 9) 
  (hrem : n % 11 = 1) : 
  n = 100 := 
by 
  sorry

end number_is_100_l93_93549


namespace mass_percentage_of_N_in_NH4Br_l93_93050

theorem mass_percentage_of_N_in_NH4Br :
  let molar_mass_N := 14.01
  let molar_mass_H := 1.01
  let molar_mass_Br := 79.90
  let molar_mass_NH4Br := (1 * molar_mass_N) + (4 * molar_mass_H) + (1 * molar_mass_Br)
  let mass_percentage_N := (molar_mass_N / molar_mass_NH4Br) * 100
  mass_percentage_N = 14.30 :=
by
  sorry

end mass_percentage_of_N_in_NH4Br_l93_93050


namespace printer_cost_comparison_l93_93271

-- Definitions based on the given conditions
def in_store_price : ℝ := 150.00
def discount_rate : ℝ := 0.10
def installment_payment : ℝ := 28.00
def number_of_installments : ℕ := 5
def shipping_handling_charge : ℝ := 12.50

-- Discounted in-store price calculation
def discounted_in_store_price : ℝ := in_store_price * (1 - discount_rate)

-- Total cost from the television advertiser
def tv_advertiser_total_cost : ℝ := (number_of_installments * installment_payment) + shipping_handling_charge

-- Proof statement
theorem printer_cost_comparison :
  discounted_in_store_price - tv_advertiser_total_cost = -17.50 :=
by
  sorry

end printer_cost_comparison_l93_93271


namespace minimum_f_value_l93_93105

noncomputable def f (x : ℝ) : ℝ :=
   Real.sqrt (2 * x ^ 2 - 4 * x + 4) + 
   Real.sqrt (2 * x ^ 2 - 16 * x + (Real.log x / Real.log 2) ^ 2 - 2 * x * (Real.log x / Real.log 2) + 
              2 * (Real.log x / Real.log 2) + 50)

theorem minimum_f_value : ∀ x : ℝ, x > 0 → f x ≥ 7 ∧ f 2 = 7 :=
by
  sorry

end minimum_f_value_l93_93105


namespace evaluate_expression_l93_93131

variable (a : ℤ) (x : ℤ)

theorem evaluate_expression (h : x = a + 9) : x - a + 5 = 14 :=
by
  sorry

end evaluate_expression_l93_93131


namespace distinct_meals_l93_93314

def num_entrees : ℕ := 4
def num_drinks : ℕ := 2
def num_desserts : ℕ := 2

theorem distinct_meals : num_entrees * num_drinks * num_desserts = 16 := by
  sorry

end distinct_meals_l93_93314


namespace kindergarten_library_models_l93_93338

theorem kindergarten_library_models
  (paid : ℕ)
  (reduced_price : ℕ)
  (models_total_gt_5 : ℕ)
  (bought : ℕ) 
  (condition : paid = 570 ∧ reduced_price = 95 ∧ models_total_gt_5 > 5 ∧ bought = 3 * (2 : ℕ)) :
  exists x : ℕ, bought / 3 = x ∧ x = 2 :=
by
  sorry

end kindergarten_library_models_l93_93338


namespace Q_over_P_l93_93767

theorem Q_over_P (P Q : ℚ)
  (h : ∀ (x : ℝ), x ≠ 0 ∧ x ≠ 3 ∧ x ≠ -3 → 
    (P / (x + 3) + Q / (x^2 - 3*x) = (x^2 - x + 8) / (x^3 + x^2 - 9*x))) :
  Q / P = 8 / 3 :=
by
  sorry

end Q_over_P_l93_93767


namespace find_prime_n_l93_93478

theorem find_prime_n (n k m : ℤ) (h1 : n - 6 = k ^ 2) (h2 : n + 10 = m ^ 2) (h3 : m ^ 2 - k ^ 2 = 16) (h4 : Nat.Prime (Int.natAbs n)) : n = 71 := by
  sorry

end find_prime_n_l93_93478


namespace polynomial_value_l93_93645

theorem polynomial_value (a b : ℝ) (h₁ : a * b = 7) (h₂ : a + b = 2) : a^2 * b + a * b^2 - 20 = -6 :=
by {
  sorry
}

end polynomial_value_l93_93645


namespace Brittany_second_test_grade_is_83_l93_93505

theorem Brittany_second_test_grade_is_83
  (first_test_score : ℝ) (first_test_weight : ℝ) 
  (second_test_weight : ℝ) (final_weighted_average : ℝ) : 
  first_test_score = 78 → 
  first_test_weight = 0.40 →
  second_test_weight = 0.60 →
  final_weighted_average = 81 →
  ∃ G : ℝ, 0.40 * first_test_score + 0.60 * G = final_weighted_average ∧ G = 83 :=
by
  sorry

end Brittany_second_test_grade_is_83_l93_93505


namespace quadratic_has_distinct_real_roots_l93_93995

-- Definitions of the coefficients
def a : ℝ := 1
def b : ℝ := -1
def c : ℝ := -2

-- Definition of the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The theorem stating the quadratic equation has two distinct real roots
theorem quadratic_has_distinct_real_roots :
  discriminant a b c > 0 :=
by
  -- Coefficients specific to the problem
  unfold a b c
  -- Calculate the discriminant
  unfold discriminant
  -- Substitute the values and compute
  sorry -- Skipping the actual proof as per instructions

end quadratic_has_distinct_real_roots_l93_93995


namespace problem_statement_l93_93326

theorem problem_statement (a b c : ℝ) (h_nonneg_a : a ≥ 0) (h_nonneg_b : b ≥ 0) (h_nonneg_c : c ≥ 0) (h_condition : a * b + b * c + c * a = 1 / 3) :
  1 / (a^2 - b * c + 1) + 1 / (b^2 - c * a + 1) + 1 / (c^2 - a * b + 1) ≤ 3 :=
by
  sorry

end problem_statement_l93_93326


namespace pure_imaginary_a_l93_93497

theorem pure_imaginary_a (a : ℝ) :
  (a^2 - 4 = 0) ∧ (a - 2 ≠ 0) ↔ a = -2 :=
by
  sorry

end pure_imaginary_a_l93_93497


namespace find_f_of_3_l93_93329

noncomputable def f : ℝ → ℝ :=
  sorry

theorem find_f_of_3 (h : ∀ x : ℝ, x ≠ 0 → f x - 3 * f (1 / x) = 3 ^ x) :
  f 3 = (-27 + 3 * (3 ^ (1 / 3))) / 8 :=
sorry

end find_f_of_3_l93_93329


namespace number_replacement_l93_93731

theorem number_replacement :
  ∃ x : ℝ, ( (x / (1 / 2) * x) / (x * (1 / 2) / x) = 25 ) ↔ x = 2.5 :=
by 
  sorry

end number_replacement_l93_93731


namespace find_n_l93_93617

theorem find_n (n : ℕ) (h1 : 0 ≤ n ∧ n ≤ 360) (h2 : Real.cos (n * Real.pi / 180) = Real.cos (340 * Real.pi / 180)) : 
  n = 20 ∨ n = 340 := 
by
  sorry

end find_n_l93_93617


namespace part1_part2_l93_93126

open Classical

theorem part1 (x : ℝ) (a : ℝ) (b : ℝ) :
  (a = 1) ∧ (b = 2) ∧ (∀ x, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :=
by
  sorry

theorem part2 (x y k : ℝ) (a b : ℝ) :
  a = 1 ∧ b = 2 ∧ (x > 0) ∧ (y > 0) ∧ (1 / x + 2 / y = 1) ∧ (2 * x + y ≥ k^2 + k + 2) → -3 ≤ k ∧ k ≤ 2 :=
by
  sorry

end part1_part2_l93_93126


namespace girls_to_boys_ratio_l93_93795

theorem girls_to_boys_ratio (g b : ℕ) (h1 : g = b + 5) (h2 : g + b = 35) : g / b = 4 / 3 :=
by
  sorry

end girls_to_boys_ratio_l93_93795


namespace bakery_problem_l93_93892

theorem bakery_problem :
  let chocolate_chip := 154
  let oatmeal_raisin := 86
  let sugar := 52
  let capacity := 16
  let needed_chocolate_chip := capacity - (chocolate_chip % capacity)
  let needed_oatmeal_raisin := capacity - (oatmeal_raisin % capacity)
  let needed_sugar := capacity - (sugar % capacity)
  (needed_chocolate_chip = 6) ∧ (needed_oatmeal_raisin = 10) ∧ (needed_sugar = 12) :=
by
  sorry

end bakery_problem_l93_93892


namespace opposite_of_reciprocal_negative_one_third_l93_93368

theorem opposite_of_reciprocal_negative_one_third : -(1 / (-1 / 3)) = 3 := by
  sorry

end opposite_of_reciprocal_negative_one_third_l93_93368


namespace total_notes_in_week_l93_93509

-- Define the conditions for day hours ring pattern
def day_notes (hour : ℕ) (minute : ℕ) : ℕ :=
  if minute = 15 then 2
  else if minute = 30 then 4
  else if minute = 45 then 6
  else if minute = 0 then 
    8 + (if hour % 2 = 0 then hour else hour / 2)
  else 0

-- Define the conditions for night hours ring pattern
def night_notes (hour : ℕ) (minute : ℕ) : ℕ :=
  if minute = 15 then 3
  else if minute = 30 then 5
  else if minute = 45 then 7
  else if minute = 0 then 
    9 + (if hour % 2 = 1 then hour else hour / 2)
  else 0

-- Define total notes over day period
def total_day_notes : ℕ := 
  (day_notes 6 0 + day_notes 7 0 + day_notes 8 0 + day_notes 9 0 + day_notes 10 0 + day_notes 11 0
 + day_notes 12 0 + day_notes 1 0 + day_notes 2 0 + day_notes 3 0 + day_notes 4 0 + day_notes 5 0)
 +
 (2 * 12 + 4 * 12 + 6 * 12)

-- Define total notes over night period
def total_night_notes : ℕ := 
  (night_notes 6 0 + night_notes 7 0 + night_notes 8 0 + night_notes 9 0 + night_notes 10 0 + night_notes 11 0
 + night_notes 12 0 + night_notes 1 0 + night_notes 2 0 + night_notes 3 0 + night_notes 4 0 + night_notes 5 0)
 +
 (3 * 12 + 5 * 12 + 7 * 12)

-- Define the total number of notes the clock will ring in a full week
def total_week_notes : ℕ :=
  7 * (total_day_notes + total_night_notes)

theorem total_notes_in_week : 
  total_week_notes = 3297 := 
  by 
  sorry

end total_notes_in_week_l93_93509


namespace determine_b_l93_93076

theorem determine_b (b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Iio 2 ∪ Set.Ioi 6 → -x^2 + b * x - 7 < 0) ∧ 
  (∀ x : ℝ, ¬(x ∈ Set.Iio 2 ∪ Set.Ioi 6) → ¬(-x^2 + b * x - 7 < 0)) → 
  b = 8 :=
sorry

end determine_b_l93_93076


namespace remainder_proof_l93_93455

theorem remainder_proof (x y u v : ℕ) (h1 : x = u * y + v) (h2 : 0 ≤ v ∧ v < y) : 
  (x + y * u^2 + 3 * v) % y = 4 * v % y :=
by
  sorry

end remainder_proof_l93_93455


namespace length_of_first_train_l93_93906

theorem length_of_first_train 
  (speed_first_train_kmph : ℝ) 
  (speed_second_train_kmph : ℝ) 
  (crossing_time_s : ℝ) 
  (length_second_train_m : ℝ) 
  (hspeed_first : speed_first_train_kmph = 120) 
  (hspeed_second : speed_second_train_kmph = 80) 
  (htime : crossing_time_s = 9) 
  (hlength_second : length_second_train_m = 320.04) :
  ∃ (length_first_train_m : ℝ), abs (length_first_train_m - 180) < 0.1 :=
by
  sorry

end length_of_first_train_l93_93906


namespace total_charge_correct_l93_93653

def boxwoodTrimCost (numBoxwoods : Nat) (trimCost : Nat) : Nat :=
  numBoxwoods * trimCost

def boxwoodShapeCost (numBoxwoods : Nat) (shapeCost : Nat) : Nat :=
  numBoxwoods * shapeCost

theorem total_charge_correct :
  let numBoxwoodsTrimmed := 30
  let trimCost := 5
  let numBoxwoodsShaped := 4
  let shapeCost := 15
  let totalTrimCost := boxwoodTrimCost numBoxwoodsTrimmed trimCost
  let totalShapeCost := boxwoodShapeCost numBoxwoodsShaped shapeCost
  let totalCharge := totalTrimCost + totalShapeCost
  totalCharge = 210 :=
by sorry

end total_charge_correct_l93_93653


namespace number_of_welders_left_l93_93417

-- Define the constants and variables
def welders_total : ℕ := 36
def days_to_complete : ℕ := 5
def rate : ℝ := 1  -- Assume the rate per welder is 1 for simplicity
def total_work : ℝ := welders_total * days_to_complete * rate

def days_after_first : ℕ := 6
def work_done_in_first_day : ℝ := welders_total * 1 * rate
def remaining_work : ℝ := total_work - work_done_in_first_day

-- Define the theorem to solve for the number of welders x that started to work on another project
theorem number_of_welders_left (x : ℕ) : (welders_total - x) * days_after_first * rate = remaining_work → x = 12 := by
  intros h
  sorry

end number_of_welders_left_l93_93417


namespace train_waiting_probability_l93_93263

-- Conditions
def trains_per_hour : ℕ := 1
def total_minutes : ℕ := 60
def wait_time : ℕ := 10

-- Proposition
theorem train_waiting_probability : 
  (wait_time : ℝ) / (total_minutes / trains_per_hour) = 1 / 6 :=
by
  -- Here we assume the proof proceeds correctly
  sorry

end train_waiting_probability_l93_93263


namespace billy_soda_distribution_l93_93760

theorem billy_soda_distribution (sisters : ℕ) (brothers : ℕ) (total_sodas : ℕ) (total_siblings : ℕ)
  (h1 : total_sodas = 12)
  (h2 : sisters = 2)
  (h3 : brothers = 2 * sisters)
  (h4 : total_siblings = sisters + brothers) :
  total_sodas / total_siblings = 2 :=
by
  sorry

end billy_soda_distribution_l93_93760


namespace smallest_b_factors_l93_93687

theorem smallest_b_factors (p q b : ℤ) (hpq : p * q = 1764) (hb : b = p + q) (hposp : p > 0) (hposq : q > 0) :
  b = 84 :=
by
  sorry

end smallest_b_factors_l93_93687


namespace bag_contains_fifteen_balls_l93_93978

theorem bag_contains_fifteen_balls 
  (r b : ℕ) 
  (h1 : r + b = 15) 
  (h2 : (r * (r - 1)) / 210 = 1 / 21) 
  : r = 4 := 
sorry

end bag_contains_fifteen_balls_l93_93978


namespace permutations_of_six_digit_number_l93_93422

/-- 
Theorem: The number of distinct permutations of the digits 1, 1, 3, 3, 3, 8 
to form six-digit positive integers is 60. 
-/
theorem permutations_of_six_digit_number : 
  (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 3)) = 60 := 
by 
  sorry

end permutations_of_six_digit_number_l93_93422


namespace peg_arrangement_l93_93308

theorem peg_arrangement :
  let Y := 5
  let R := 4
  let G := 3
  let B := 2
  let O := 1
  (Y! * R! * G! * B! * O!) = 34560 :=
by
  sorry

end peg_arrangement_l93_93308


namespace triangle_largest_angle_l93_93212

theorem triangle_largest_angle (x : ℝ) (AB : ℝ) (AC : ℝ) (BC : ℝ) (h1 : AB = x + 5) 
                               (h2 : AC = 2 * x + 3) (h3 : BC = x + 10)
                               (h_angle_A_largest : BC > AB ∧ BC > AC)
                               (triangle_inequality_1 : AB + AC > BC)
                               (triangle_inequality_2 : AB + BC > AC)
                               (triangle_inequality_3 : AC + BC > AB) :
  1 < x ∧ x < 7 ∧ 6 = 6 := 
by {
  sorry
}

end triangle_largest_angle_l93_93212


namespace problem1_problem2_problem3_l93_93031

-- Definition of the polynomial expansion
def poly (x : ℝ) := (1 - 2*x)^7

-- Definitions capturing the conditions directly
def a_0 := 1
def sum_a_1_to_a_7 := -2
def sum_a_1_3_5_7 := -1094
def sum_abs_a_0_to_a_7 := 2187

-- Lean statements for the proof problems
theorem problem1 (x : ℝ) (a : Fin 8 → ℝ) (h : poly x = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = sum_a_1_to_a_7 :=
sorry

theorem problem2 (x : ℝ) (a : Fin 8 → ℝ) (h : poly x = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7) :
  a 1 + a 3 + a 5 + a 7 = sum_a_1_3_5_7 :=
sorry

theorem problem3 (x : ℝ) (a : Fin 8 → ℝ) (h : poly x = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7) :
  abs (a 0) + abs (a 1) + abs (a 2) + abs (a 3) + abs (a 4) + abs (a 5) + abs (a 6) + abs (a 7) = sum_abs_a_0_to_a_7 :=
sorry

end problem1_problem2_problem3_l93_93031


namespace triangle_height_dist_inequality_l93_93820

variable {T : Type} [MetricSpace T] 

theorem triangle_height_dist_inequality {h_a h_b h_c l_a l_b l_c : ℝ} (h_a_pos : 0 < h_a) (h_b_pos : 0 < h_b) (h_c_pos : 0 < h_c) 
  (l_a_pos : 0 < l_a) (l_b_pos : 0 < l_b) (l_c_pos : 0 < l_c) :
  h_a / l_a + h_b / l_b + h_c / l_c >= 9 :=
sorry

end triangle_height_dist_inequality_l93_93820


namespace range_of_k_l93_93715

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, (x - 1) / (x - 2) = k / (x - 2) + 2 ∧ x ≥ 0 ∧ x ≠ 2) ↔ (k ≤ 3 ∧ k ≠ 1) :=
by
  sorry

end range_of_k_l93_93715


namespace time_taken_by_abc_l93_93625

-- Define the work rates for a, b, and c
def work_rate_a_b : ℚ := 1 / 15
def work_rate_c : ℚ := 1 / 41.25

-- Define the combined work rate for a, b, and c
def combined_work_rate : ℚ := work_rate_a_b + work_rate_c

-- Define the reciprocal of the combined work rate, which is the time taken
def time_taken : ℚ := 1 / combined_work_rate

-- Prove that the time taken by a, b, and c together is 11 days
theorem time_taken_by_abc : time_taken = 11 := by
  -- Substitute the values to compute the result
  sorry

end time_taken_by_abc_l93_93625


namespace permutations_containing_substring_l93_93340

open Nat

/-- Prove that the number of permutations of the string "000011112222" that contain the substring "2020" is equal to 3575. -/
theorem permutations_containing_substring :
  let total_permutations := factorial 8 / (factorial 2 * factorial 4 * factorial 2)
  let num_positions := 9
  let non_overlap_count := total_permutations * num_positions
  let overlap_subtract := 7 * (factorial 6 / (factorial 1 * factorial 4 * factorial 1))
  let add_back := 5 * (factorial 4 / factorial 4)
  non_overlap_count - overlap_subtract + add_back = 3575 := 
by
  let total_permutations := factorial 8 / (factorial 2 * factorial 4 * factorial 2)
  let num_positions := 9
  let non_overlap_count := total_permutations * num_positions
  let overlap_subtract := 7 * (factorial 6 / (factorial 1 * factorial 4 * factorial 1))
  let add_back := 5 * (factorial 4 / factorial 4)
  have h: non_overlap_count - overlap_subtract + add_back = 3575 := by sorry
  exact h

end permutations_containing_substring_l93_93340


namespace min_value_m_plus_n_l93_93257

theorem min_value_m_plus_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : 45 * m = n^3) : m + n = 90 :=
sorry

end min_value_m_plus_n_l93_93257


namespace sandwiches_prepared_l93_93268

variable (S : ℕ)
variable (H1 : S > 0)
variable (H2 : ∃ r : ℕ, r = S / 4)
variable (H3 : ∃ b : ℕ, b = (3 * S / 4) / 6)
variable (H4 : ∃ c : ℕ, c = 2 * b)
variable (H5 : ∃ x : ℕ, 5 * x = 5)
variable (H6 : 3 * S / 8 - 5 = 4)

theorem sandwiches_prepared : S = 24 :=
by
  sorry

end sandwiches_prepared_l93_93268


namespace calculate_rows_l93_93538

-- Definitions based on conditions
def totalPecanPies : ℕ := 16
def totalApplePies : ℕ := 14
def piesPerRow : ℕ := 5

-- The goal is to prove the total rows of pies
theorem calculate_rows : (totalPecanPies + totalApplePies) / piesPerRow = 6 := by
  sorry

end calculate_rows_l93_93538


namespace max_distance_circle_to_point_A_l93_93753

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 2

noncomputable def point_A : ℝ × ℝ := (-1, 3)

noncomputable def max_distance (d : ℝ) : Prop :=
  ∃ x y, circle_eq x y ∧ d = Real.sqrt ((2 + 1)^2 + (0 - 3)^2) + Real.sqrt 2 

theorem max_distance_circle_to_point_A : max_distance (4 * Real.sqrt 2) :=
sorry

end max_distance_circle_to_point_A_l93_93753


namespace total_number_of_workers_l93_93730

-- Definitions based on the given conditions
def avg_salary_total : ℝ := 8000
def avg_salary_technicians : ℝ := 12000
def avg_salary_non_technicians : ℝ := 6000
def num_technicians : ℕ := 7

-- Problem statement in Lean
theorem total_number_of_workers
    (W : ℕ) (N : ℕ)
    (h1 : W * avg_salary_total = num_technicians * avg_salary_technicians + N * avg_salary_non_technicians)
    (h2 : W = num_technicians + N) :
    W = 21 :=
sorry

end total_number_of_workers_l93_93730


namespace inequalities_quadrants_l93_93586

theorem inequalities_quadrants :
  (∀ x y : ℝ, y > 2 * x → y > 4 - x → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0)) := sorry

end inequalities_quadrants_l93_93586


namespace line_and_circle_separate_l93_93886

theorem line_and_circle_separate
  (θ : ℝ) (hθ : ¬ ∃ k : ℤ, θ = k * Real.pi) :
  ¬ ∃ (x y : ℝ), (x^2 + y^2 = 1 / 2) ∧ (x * Real.cos θ + y - 1 = 0) :=
by
  sorry

end line_and_circle_separate_l93_93886


namespace Tony_science_degree_years_l93_93371

theorem Tony_science_degree_years (X : ℕ) (Total : ℕ)
  (h1 : Total = 14)
  (h2 : Total = X + 2 * X + 2) :
  X = 4 :=
by
  sorry

end Tony_science_degree_years_l93_93371


namespace initial_number_of_men_l93_93872

theorem initial_number_of_men
  (M : ℕ) (A : ℕ)
  (h1 : ∀ A_new : ℕ, A_new = A + 4)
  (h2 : ∀ total_age_increase : ℕ, total_age_increase = (2 * 52) - (36 + 32))
  (h3 : ∀ sum_age_men : ℕ, sum_age_men = M * A)
  (h4 : ∀ new_sum_age_men : ℕ, new_sum_age_men = sum_age_men + ((2 * 52) - (36 + 32))) :
  M = 9 := 
by
  -- Proof skipped
  sorry

end initial_number_of_men_l93_93872


namespace general_term_l93_93940

noncomputable def seq (n : ℕ) : ℤ :=
  if n = 0 then 0 else
  if n = 1 then -1 else
  if n % 2 = 0 then (2 * 2 ^ (n / 2 - 1) - 1) / 3 else 
  (-2)^(n - n / 2) / 3 - 1

-- Conditions
def condition1 : Prop := seq 1 = -1
def condition2 : Prop := seq 2 > seq 1
def condition3 (n : ℕ) : Prop := |seq (n + 1) - seq n| = 2^n
def condition4 : Prop := ∀ m, seq (2*m + 1) > seq (2*m - 1)
def condition5 : Prop := ∀ m, seq (2*m) < seq (2*m + 2)

-- The theorem stating the general term of the sequence
theorem general_term (n : ℕ) :
  condition1 →
  condition2 →
  (∀ n, condition3 n) →
  condition4 →
  condition5 →
  seq n = ( (-2)^n - 1) / 3 :=
by
  sorry

end general_term_l93_93940


namespace simplify_expression_l93_93220

theorem simplify_expression (a : ℝ) (h : a / 2 - 2 / a = 3) : 
  (a^8 - 256) / (16 * a^4) * (2 * a) / (a^2 + 4) = 33 :=
by
  sorry

end simplify_expression_l93_93220


namespace min_value_expr_l93_93705

theorem min_value_expr (a d : ℝ) (b c : ℝ) (h_a : 0 ≤ a) (h_d : 0 ≤ d) (h_b : 0 < b) (h_c : 0 < c) (h : b + c ≥ a + d) :
  (b / (c + d) + c / (a + b)) ≥ (Real.sqrt 2 - 1 / 2) :=
sorry

end min_value_expr_l93_93705


namespace total_students_in_both_classrooms_l93_93592

theorem total_students_in_both_classrooms
  (x y : ℕ)
  (hx1 : 80 * x - 250 = 90 * (x - 5))
  (hy1 : 85 * y - 480 = 95 * (y - 8)) :
  x + y = 48 := 
sorry

end total_students_in_both_classrooms_l93_93592


namespace tom_purchases_l93_93391

def total_cost_before_discount (price_per_box : ℝ) (num_boxes : ℕ) : ℝ :=
  price_per_box * num_boxes

def discount (total_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  total_cost * discount_rate

def total_cost_after_discount (total_cost : ℝ) (discount_amount : ℝ) : ℝ :=
  total_cost - discount_amount

def remaining_boxes (total_boxes : ℕ) (given_boxes : ℕ) : ℕ :=
  total_boxes - given_boxes

def total_pieces (num_boxes : ℕ) (pieces_per_box : ℕ) : ℕ :=
  num_boxes * pieces_per_box

theorem tom_purchases
  (price_per_box : ℝ) (num_boxes : ℕ) (discount_rate : ℝ) (given_boxes : ℕ) (pieces_per_box : ℕ) :
  (price_per_box = 4) →
  (num_boxes = 12) →
  (discount_rate = 0.15) →
  (given_boxes = 7) →
  (pieces_per_box = 6) →
  total_cost_after_discount (total_cost_before_discount price_per_box num_boxes) 
                             (discount (total_cost_before_discount price_per_box num_boxes) discount_rate)
  = 40.80 ∧
  total_pieces (remaining_boxes num_boxes given_boxes) pieces_per_box
  = 30 :=
by
  intros
  sorry

end tom_purchases_l93_93391


namespace no_such_xy_between_988_and_1991_l93_93477

theorem no_such_xy_between_988_and_1991 :
  ¬ ∃ (x y : ℕ), 988 ≤ x ∧ x < y ∧ y ≤ 1991 ∧ 
  (∃ a b : ℕ, xy = x * y ∧ (xy + x = a^2 ∧ xy + y = b^2)) :=
by
  sorry

end no_such_xy_between_988_and_1991_l93_93477


namespace product_of_c_values_l93_93964

theorem product_of_c_values :
  ∃ (c1 c2 : ℕ), (c1 > 0 ∧ c2 > 0) ∧
  (∃ (x1 x2 : ℚ), (7 * x1^2 + 15 * x1 + c1 = 0) ∧ (7 * x2^2 + 15 * x2 + c2 = 0)) ∧
  (c1 * c2 = 16) :=
sorry

end product_of_c_values_l93_93964


namespace sum_ratio_l93_93293

variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}
variable {d : ℕ}

axiom arithmetic_sequence : ∀ n, a_n n = a_n 1 + (n - 1) * d
axiom sum_of_first_n_terms : ∀ n, S_n n = n * (a_n 1 + a_n n) / 2
axiom condition_a4 : a_n 4 = 2 * (a_n 2 + a_n 3)
axiom non_zero_difference : d ≠ 0

theorem sum_ratio : S_n 7 / S_n 4 = 7 / 4 := 
by
  sorry

end sum_ratio_l93_93293


namespace inequality_am_gm_l93_93701

theorem inequality_am_gm (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) (h : a^2 + b^2 + c^2 = 12) :
  1/(a-1) + 1/(b-1) + 1/(c-1) ≥ 3 := 
by
  sorry

end inequality_am_gm_l93_93701


namespace ratio_of_segments_l93_93344

variable (F S T : ℕ)

theorem ratio_of_segments : T = 10 → F = 2 * (S + T) → F + S + T = 90 → (T / S = 1 / 2) :=
by
  intros hT hF hSum
  sorry

end ratio_of_segments_l93_93344


namespace cos_tan_quadrant_l93_93430

theorem cos_tan_quadrant (α : ℝ) 
  (hcos : Real.cos α < 0) 
  (htan : Real.tan α > 0) : 
  (2 * π / 2 < α ∧ α < π) :=
by
  sorry

end cos_tan_quadrant_l93_93430


namespace middle_managers_sample_count_l93_93751

def employees_total : ℕ := 1000
def managers_middle_total : ℕ := 150
def sample_total : ℕ := 200

theorem middle_managers_sample_count :
  sample_total * managers_middle_total / employees_total = 30 := by
  sorry

end middle_managers_sample_count_l93_93751


namespace sqrt_range_l93_93607

theorem sqrt_range (x : ℝ) : (1 - x ≥ 0) ↔ (x ≤ 1) := sorry

end sqrt_range_l93_93607


namespace angle_B_l93_93269

-- Define the conditions
variables {A B C : ℝ} (a b c : ℝ)
variable (h : a^2 + c^2 = b^2 + ac)

-- State the theorem
theorem angle_B (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  B = π / 3 :=
sorry

end angle_B_l93_93269


namespace inequality_f_n_l93_93670

theorem inequality_f_n {f : ℕ → ℕ} {k : ℕ} (strict_mono_f : ∀ {a b : ℕ}, a < b → f a < f b)
  (h_f : ∀ n : ℕ, f (f n) = k * n) : ∀ n : ℕ, 
  (2 * k * n) / (k + 1) ≤ f n ∧ f n ≤ ((k + 1) * n) / 2 :=
by
  sorry

end inequality_f_n_l93_93670


namespace time_ratio_school_home_l93_93629

open Real

noncomputable def time_ratio (y x : ℝ) : ℝ :=
  let time_school := (y / (3 * x)) + (2 * y / (2 * x)) + (y / (4 * x))
  let time_home := (y / (4 * x)) + (2 * y / (2 * x)) + (y / (3 * x))
  time_school / time_home

theorem time_ratio_school_home (y x : ℝ) (hy : y ≠ 0) (hx : x ≠ 0) : time_ratio y x = 19 / 16 :=
  sorry

end time_ratio_school_home_l93_93629


namespace piggy_bank_total_l93_93448

def amount_added_in_january: ℕ := 19
def amount_added_in_february: ℕ := 19
def amount_added_in_march: ℕ := 8

theorem piggy_bank_total:
  amount_added_in_january + amount_added_in_february + amount_added_in_march = 46 := by
  sorry

end piggy_bank_total_l93_93448


namespace smallest_positive_period_f_intervals_monotonically_increasing_f_l93_93244

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) * (Real.sin x + Real.cos x)

-- 1. Proving the smallest positive period is π
theorem smallest_positive_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = Real.pi := 
sorry

-- 2. Proving the intervals where the function is monotonically increasing
theorem intervals_monotonically_increasing_f : 
  ∀ k : ℤ, ∀ x : ℝ, x ∈ Set.Icc (k * Real.pi - (3 * Real.pi / 8)) (k * Real.pi + (Real.pi / 8)) → 
    0 < deriv f x :=
sorry

end smallest_positive_period_f_intervals_monotonically_increasing_f_l93_93244


namespace ratio_chloe_to_max_l93_93700

/-- Chloe’s wins and Max’s wins -/
def chloe_wins : ℕ := 24
def max_wins : ℕ := 9

/-- The ratio of Chloe's wins to Max's wins is 8:3 -/
theorem ratio_chloe_to_max : (chloe_wins / Nat.gcd chloe_wins max_wins) = 8 ∧ (max_wins / Nat.gcd chloe_wins max_wins) = 3 := by
  sorry

end ratio_chloe_to_max_l93_93700


namespace square_field_area_l93_93433

theorem square_field_area (speed time perimeter : ℕ) (h1 : speed = 20) (h2 : time = 4) (h3 : perimeter = speed * time) :
  ∃ s : ℕ, perimeter = 4 * s ∧ s * s = 400 :=
by
  -- All conditions and definitions are stated, proof is skipped using sorry
  sorry

end square_field_area_l93_93433


namespace crayons_per_child_l93_93854

theorem crayons_per_child (children : ℕ) (total_crayons : ℕ) (h1 : children = 18) (h2 : total_crayons = 216) : 
    total_crayons / children = 12 := 
by
  sorry

end crayons_per_child_l93_93854


namespace part1_part2_l93_93099

namespace Problem

open Real

def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2*x - 8 > 0)

theorem part1 (h : p 1 x ∧ q x) : 2 < x ∧ x < 3:= 
sorry

theorem part2 (hpq : ∀ x, ¬ p a x → ¬ q x) : 
   1 < a ∧ a ≤ 2 := 
sorry

end Problem

end part1_part2_l93_93099


namespace milk_remaining_l93_93737

def initial_whole_milk := 15
def initial_low_fat_milk := 12
def initial_almond_milk := 8

def jason_buys := 5
def jason_promotion := 2 -- every 2 bottles he gets 1 free

def harry_buys_low_fat := 4
def harry_gets_free_low_fat := 1
def harry_buys_almond := 2

theorem milk_remaining : 
  (initial_whole_milk - jason_buys = 10) ∧ 
  (initial_low_fat_milk - (harry_buys_low_fat + harry_gets_free_low_fat) = 7) ∧ 
  (initial_almond_milk - harry_buys_almond = 6) :=
by
  sorry

end milk_remaining_l93_93737


namespace milburg_population_l93_93535

-- Define the number of grown-ups and children in Milburg
def grown_ups : ℕ := 5256
def children : ℕ := 2987

-- The total population is defined as the sum of grown-ups and children
def total_population : ℕ := grown_ups + children

-- Goal: Prove that the total population in Milburg is 8243
theorem milburg_population : total_population = 8243 := 
by {
  -- the proof should be here, but we use sorry to skip it
  sorry
}

end milburg_population_l93_93535


namespace remove_parentheses_correct_l93_93246

variable {a b c : ℝ}

theorem remove_parentheses_correct :
  -(a - b) = -a + b :=
by sorry

end remove_parentheses_correct_l93_93246


namespace melissa_solves_equation_l93_93817

theorem melissa_solves_equation : 
  ∃ b c : ℤ, (∀ x : ℝ, x^2 - 6 * x + 9 = 0 ↔ (x + b)^2 = c) ∧ b + c = -3 :=
by
  sorry

end melissa_solves_equation_l93_93817


namespace Lewis_found_20_items_l93_93379

-- Define the number of items Tanya found
def Tanya_items : ℕ := 4

-- Define the number of items Samantha found
def Samantha_items : ℕ := 4 * Tanya_items

-- Define the number of items Lewis found
def Lewis_items : ℕ := Samantha_items + 4

-- Theorem to prove the number of items Lewis found
theorem Lewis_found_20_items : Lewis_items = 20 := by
  sorry

end Lewis_found_20_items_l93_93379


namespace value_of_a_minus_2b_l93_93863

theorem value_of_a_minus_2b 
  (a b : ℚ) 
  (h : ∀ y : ℚ, y > 0 → y ≠ 2 → y ≠ -3 → (a / (y-2) + b / (y+3) = (2 * y + 5) / ((y-2)*(y+3)))) 
  : a - 2 * b = 7 / 5 :=
sorry

end value_of_a_minus_2b_l93_93863


namespace total_games_in_single_elimination_tournament_l93_93821

def single_elimination_tournament_games (teams : ℕ) : ℕ :=
teams - 1

theorem total_games_in_single_elimination_tournament :
  single_elimination_tournament_games 23 = 22 :=
by
  sorry

end total_games_in_single_elimination_tournament_l93_93821


namespace table_area_l93_93354

/-- Given the combined area of three table runners is 224 square inches, 
     overlapping the runners to cover 80% of a table results in exactly 24 square inches being covered by 
     two layers, and the area covered by three layers is 30 square inches,
     prove that the area of the table is 175 square inches. -/
theorem table_area (A : ℝ) (S T H : ℝ) (h1 : S + 2 * T + 3 * H = 224)
   (h2 : 0.80 * A = S + T + H) (h3 : T = 24) (h4 : H = 30) : A = 175 := 
sorry

end table_area_l93_93354


namespace james_bought_400_fish_l93_93393

theorem james_bought_400_fish
  (F : ℝ)
  (h1 : 0.80 * F = 320)
  (h2 : F / 0.80 = 400) :
  F = 400 :=
by
  sorry

end james_bought_400_fish_l93_93393


namespace distance_AC_l93_93690

theorem distance_AC (south_dist : ℕ) (west_dist : ℕ) (north_dist : ℕ) (east_dist : ℕ) :
  south_dist = 50 → west_dist = 70 → north_dist = 30 → east_dist = 40 →
  Real.sqrt ((south_dist - north_dist)^2 + (west_dist - east_dist)^2) = 36.06 :=
by
  intros h_south h_west h_north h_east
  rw [h_south, h_west, h_north, h_east]
  simp
  norm_num
  sorry

end distance_AC_l93_93690


namespace Malou_score_third_quiz_l93_93038

-- Defining the conditions as Lean definitions
def score1 : ℕ := 91
def score2 : ℕ := 92
def average : ℕ := 91
def num_quizzes : ℕ := 3

-- Proving that score3 equals 90
theorem Malou_score_third_quiz :
  ∃ score3 : ℕ, (score1 + score2 + score3) / num_quizzes = average ∧ score3 = 90 :=
by
  use (90 : ℕ)
  sorry

end Malou_score_third_quiz_l93_93038


namespace remainder_when_divided_by_5_l93_93989

theorem remainder_when_divided_by_5 
  (n : ℕ) 
  (h : n % 10 = 7) : 
  n % 5 = 2 := 
by 
  sorry

end remainder_when_divided_by_5_l93_93989


namespace increase_in_average_weight_l93_93147

theorem increase_in_average_weight 
    (A : ℝ) 
    (weight_left : ℝ)
    (weight_new : ℝ)
    (h_weight_left : weight_left = 67)
    (h_weight_new : weight_new = 87) : 
    ((8 * A - weight_left + weight_new) / 8 - A) = 2.5 := 
by
  sorry

end increase_in_average_weight_l93_93147


namespace value_of_y_l93_93697

theorem value_of_y (x y : ℝ) (h₁ : 1.5 * x = 0.75 * y) (h₂ : x = 20) : y = 40 :=
sorry

end value_of_y_l93_93697


namespace cone_height_l93_93662

theorem cone_height (R : ℝ) (r h l : ℝ)
  (volume_sphere : ∀ R,  V_sphere = (4 / 3) * π * R^3)
  (volume_cone : ∀ r h,  V_cone = (1 / 3) * π * r^2 * h)
  (lateral_surface_area : ∀ r l, A_lateral = π * r * l)
  (area_base : ∀ r, A_base = π * r^2)
  (vol_eq : (1/3) * π * r^2 * h = (4/3) * π * R^3)
  (lat_eq : π * r * l = 3 * π * r^2) 
  (pyth_rel : l^2 = r^2 + h^2) :
  h = 4 * R * Real.sqrt 2 := 
sorry

end cone_height_l93_93662


namespace intersect_at_2d_l93_93113

def g (x : ℝ) (c : ℝ) : ℝ := 4 * x + c

theorem intersect_at_2d (c d : ℤ) (h₁ : d = 8 + c) (h₂ : 2 = g d c) : d = 2 :=
by
  sorry

end intersect_at_2d_l93_93113


namespace number_of_students_l93_93360

-- Definitions based on conditions
def candy_bar_cost : ℝ := 2
def chips_cost : ℝ := 0.5
def total_cost_per_student : ℝ := candy_bar_cost + 2 * chips_cost
def total_amount : ℝ := 15

-- Statement to prove
theorem number_of_students : (total_amount / total_cost_per_student) = 5 :=
by
  sorry

end number_of_students_l93_93360


namespace abs_reciprocal_inequality_l93_93489

theorem abs_reciprocal_inequality (a b : ℝ) (h : 1 / |a| < 1 / |b|) : |a| > |b| :=
sorry

end abs_reciprocal_inequality_l93_93489


namespace b_minus_a_eq_two_l93_93075

theorem b_minus_a_eq_two (a b : ℤ) (h1 : b = 7) (h2 : a * b = 2 * (a + b) + 11) : b - a = 2 :=
by
  sorry

end b_minus_a_eq_two_l93_93075


namespace find_m_range_l93_93827

theorem find_m_range (m : ℝ) : 
  (∃ x : ℤ, 2 * (x : ℝ) - 1 ≤ 5 ∧ x - 1 ≥ m ∧ x ≤ 3) ∧ 
  (∃ y : ℤ, 2 * (y : ℝ) - 1 ≤ 5 ∧ y - 1 ≥ m ∧ y ≤ 3 ∧ x ≠ y) → 
  -1 < m ∧ m ≤ 0 := by
  sorry

end find_m_range_l93_93827


namespace find_number_l93_93091

theorem find_number (N : ℕ) (h : N / 7 = 12 ∧ N % 7 = 5) : N = 89 := 
by
  sorry

end find_number_l93_93091


namespace simplify_sqrt_sum_l93_93036

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sqrt_sum_l93_93036


namespace cubic_difference_l93_93418

theorem cubic_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 50) : a^3 - b^3 = 353.5 := by
  sorry

end cubic_difference_l93_93418


namespace frog_climbing_time_l93_93802

-- Defining the conditions as Lean definitions
def well_depth : ℕ := 12
def climb_distance : ℕ := 3
def slip_distance : ℕ := 1
def climb_time : ℚ := 1 -- time in minutes for the frog to climb 3 meters
def slip_time : ℚ := climb_time / 3
def total_time_per_cycle : ℚ := climb_time + slip_time
def total_climbed_at_817 : ℕ := well_depth - 3 -- 3 meters from the top means it climbed 9 meters

-- The equivalent proof statement in Lean:
theorem frog_climbing_time : 
  ∃ (T : ℚ), T = 22 ∧ 
    (well_depth = 9 + 3) ∧
    (∀ (cycles : ℕ), cycles = 4 → 
         total_time_per_cycle * cycles + 2 = T) :=
by 
  sorry

end frog_climbing_time_l93_93802


namespace linear_system_solution_l93_93230

/-- Given a system of three linear equations:
      x + y + z = 1
      a x + b y + c z = h
      a² x + b² y + c² z = h²
    Prove that the solution x, y, z is given by:
    x = (h - b)(h - c) / (a - b)(a - c)
    y = (h - a)(h - c) / (b - a)(b - c)
    z = (h - a)(h - b) / (c - a)(c - b) -/
theorem linear_system_solution (a b c h : ℝ) (x y z : ℝ) :
  x + y + z = 1 →
  a * x + b * y + c * z = h →
  a^2 * x + b^2 * y + c^2 * z = h^2 →
  x = (h - b) * (h - c) / ((a - b) * (a - c)) ∧
  y = (h - a) * (h - c) / ((b - a) * (b - c)) ∧
  z = (h - a) * (h - b) / ((c - a) * (c - b)) :=
by
  intros
  sorry

end linear_system_solution_l93_93230


namespace number_of_distinct_triangle_areas_l93_93187

noncomputable def distinct_triangle_area_counts : ℕ :=
sorry  -- Placeholder for the proof to derive the correct answer

theorem number_of_distinct_triangle_areas
  (G H I J K L : ℝ × ℝ)
  (h₁ : G.2 = H.2)
  (h₂ : G.2 = I.2)
  (h₃ : G.2 = J.2)
  (h₄ : H.2 = I.2)
  (h₅ : H.2 = J.2)
  (h₆ : I.2 = J.2)
  (h₇ : dist G H = 2)
  (h₈ : dist H I = 2)
  (h₉ : dist I J = 2)
  (h₁₀ : K.2 = L.2 - 2)  -- Assuming constant perpendicular distance between parallel lines
  (h₁₁ : dist K L = 2) : 
  distinct_triangle_area_counts = 3 :=
sorry  -- Placeholder for the proof

end number_of_distinct_triangle_areas_l93_93187


namespace neither_necessary_nor_sufficient_l93_93547

theorem neither_necessary_nor_sufficient (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) :
  ¬(∀ a b, (a > b → (1 / a < 1 / b)) ∧ ((1 / a < 1 / b) → a > b)) := sorry

end neither_necessary_nor_sufficient_l93_93547


namespace problem1_solution_problem2_solution_problem3_solution_l93_93458

noncomputable def problem1 : Real :=
  3 * Real.sqrt 3 + Real.sqrt 8 - Real.sqrt 2 + Real.sqrt 27

theorem problem1_solution : problem1 = 6 * Real.sqrt 3 + Real.sqrt 2 := by
  sorry

noncomputable def problem2 : Real :=
  (1/2) * (Real.sqrt 3 + Real.sqrt 5) - (3/4) * (Real.sqrt 5 - Real.sqrt 12)

theorem problem2_solution : problem2 = 2 * Real.sqrt 3 - (1/4) * Real.sqrt 5 := by
  sorry

noncomputable def problem3 : Real :=
  (2 * Real.sqrt 5 + Real.sqrt 6) * (2 * Real.sqrt 5 - Real.sqrt 6) - (Real.sqrt 5 - Real.sqrt 6) ^ 2

theorem problem3_solution : problem3 = 3 + 2 * Real.sqrt 30 := by
  sorry

end problem1_solution_problem2_solution_problem3_solution_l93_93458


namespace jasmine_percentage_new_solution_l93_93652

-- Define the initial conditions
def initial_volume : ℝ := 80
def initial_jasmine_percent : ℝ := 0.10
def added_jasmine : ℝ := 5
def added_water : ℝ := 15

-- Define the correct answer
theorem jasmine_percentage_new_solution :
  let initial_jasmine := initial_jasmine_percent * initial_volume
  let new_jasmine := initial_jasmine + added_jasmine
  let total_new_volume := initial_volume + added_jasmine + added_water
  (new_jasmine / total_new_volume) * 100 = 13 := 
by 
  sorry

end jasmine_percentage_new_solution_l93_93652


namespace diane_coins_in_third_piggy_bank_l93_93189

theorem diane_coins_in_third_piggy_bank :
  ∀ n1 n2 n4 n5 n6 : ℕ, n1 = 72 → n2 = 81 → n4 = 99 → n5 = 108 → n6 = 117 → (n4 - (n4 - 9)) = 90 :=
by
  -- sorry is needed to avoid an incomplete proof, as only the statement is required.
  sorry

end diane_coins_in_third_piggy_bank_l93_93189


namespace find_integer_l93_93353

theorem find_integer (N : ℤ) (hN : N^2 + N = 12) (h_pos : 0 < N) : N = 3 :=
sorry

end find_integer_l93_93353


namespace age_sum_is_27_l93_93279

noncomputable def a : ℕ := 12
noncomputable def b : ℕ := 10
noncomputable def c : ℕ := 5

theorem age_sum_is_27
  (h1: a = b + 2)
  (h2: b = 2 * c)
  (h3: b = 10) :
  a + b + c = 27 :=
  sorry

end age_sum_is_27_l93_93279


namespace geom_seq_ratio_l93_93814
noncomputable section

theorem geom_seq_ratio (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h₁ : 0 < a_1)
  (h₂ : 0 < a_2)
  (h₃ : 0 < a_3)
  (h₄ : 0 < a_4)
  (h₅ : 0 < a_5)
  (h_seq : a_2 = a_1 * 2)
  (h_seq2 : a_3 = a_1 * 2^2)
  (h_seq3 : a_4 = a_1 * 2^3)
  (h_seq4 : a_5 = a_1 * 2^4)
  (h_ratio : a_4 / a_1 = 8) :
  (a_1 + a_2) * a_4 / ((a_1 + a_3) * a_5) = 3 / 10 := 
by
  sorry

end geom_seq_ratio_l93_93814


namespace triangle_max_area_in_quarter_ellipse_l93_93476

theorem triangle_max_area_in_quarter_ellipse (a b c : ℝ) (h : c^2 = a^2 - b^2) :
  ∃ (T_max : ℝ), T_max = b / 2 :=
by sorry

end triangle_max_area_in_quarter_ellipse_l93_93476


namespace max_roses_l93_93774

theorem max_roses (budget : ℝ) (indiv_price : ℝ) (dozen_1_price : ℝ) (dozen_2_price : ℝ) (dozen_5_price : ℝ) (hundred_price : ℝ) 
  (budget_eq : budget = 1000) (indiv_price_eq : indiv_price = 5.30) (dozen_1_price_eq : dozen_1_price = 36) 
  (dozen_2_price_eq : dozen_2_price = 50) (dozen_5_price_eq : dozen_5_price = 110) (hundred_price_eq : hundred_price = 180) : 
  ∃ max_roses : ℕ, max_roses = 548 :=
by
  sorry

end max_roses_l93_93774


namespace problem_X_plus_Y_l93_93528

def num_five_digit_even_numbers : Nat := 45000
def num_five_digit_multiples_of_7 : Nat := 12857
def X := num_five_digit_even_numbers
def Y := num_five_digit_multiples_of_7

theorem problem_X_plus_Y : X + Y = 57857 :=
by
  sorry

end problem_X_plus_Y_l93_93528


namespace simplify_neg_expression_l93_93122

variable (a b c : ℝ)

theorem simplify_neg_expression : 
  - (a - (b - c)) = -a + b - c :=
sorry

end simplify_neg_expression_l93_93122


namespace probability_same_number_l93_93092

theorem probability_same_number (n k : ℕ) (h₁ : n = 8) (h₂ : k = 6) : 
  (∃ m : ℝ, 0 ≤ m ∧ m ≤ 1 ∧ m = 1) := by
  sorry

end probability_same_number_l93_93092


namespace area_of_region_l93_93733

theorem area_of_region (x y : ℝ) :
  x ≤ 2 * y ∧ y ≤ 2 * x ∧ x + y ≤ 60 →
  ∃ (A : ℝ), A = 600 :=
by
  sorry

end area_of_region_l93_93733


namespace distinct_nonzero_digits_sum_l93_93427

theorem distinct_nonzero_digits_sum (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0) 
  (h7 : 100*a + 10*b + c + 100*a + 10*c + b + 100*b + 10*a + c + 100*b + 10*c + a + 100*c + 10*a + b + 100*c + 10*b + a = 1776) : 
  (a = 1 ∧ b = 2 ∧ c = 5) ∨ (a = 1 ∧ b = 3 ∧ c = 4) ∨ (a = 1 ∧ b = 4 ∧ c = 3) ∨ (a = 1 ∧ b = 5 ∧ c = 2) ∨ (a = 2 ∧ b = 1 ∧ c = 5) ∨
  (a = 2 ∧ b = 5 ∧ c = 1) ∨ (a = 3 ∧ b = 1 ∧ c = 4) ∨ (a = 3 ∧ b = 4 ∧ c = 1) ∨ (a = 4 ∧ b = 1 ∧ c = 3) ∨ (a = 4 ∧ b = 3 ∧ c = 1) ∨
  (a = 5 ∧ b = 1 ∧ c = 2) ∨ (a = 5 ∧ b = 2 ∧ c = 1) :=
sorry

end distinct_nonzero_digits_sum_l93_93427


namespace negative_remainder_l93_93534

theorem negative_remainder (a : ℤ) (h : a % 1999 = 1) : (-a) % 1999 = 1998 :=
by
  sorry

end negative_remainder_l93_93534


namespace number_of_principals_in_oxford_high_school_l93_93891

-- Define the conditions
def numberOfTeachers : ℕ := 48
def numberOfClasses : ℕ := 15
def studentsPerClass : ℕ := 20
def totalStudents : ℕ := numberOfClasses * studentsPerClass
def totalPeople : ℕ := 349
def numberOfPrincipals : ℕ := totalPeople - (numberOfTeachers + totalStudents)

-- Proposition: Prove the number of principals in Oxford High School
theorem number_of_principals_in_oxford_high_school :
  numberOfPrincipals = 1 := by sorry

end number_of_principals_in_oxford_high_school_l93_93891


namespace find_k_l93_93823

variables (m n k : ℤ)  -- Declaring m, n, k as integer variables.

theorem find_k (h1 : m = 2 * n + 5) (h2 : m + 2 = 2 * (n + k) + 5) : k = 1 :=
by
  sorry

end find_k_l93_93823


namespace candidate_function_is_odd_and_increasing_l93_93321

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_increasing_function (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

def candidate_function (x : ℝ) : ℝ := x * |x|

theorem candidate_function_is_odd_and_increasing :
  is_odd_function candidate_function ∧ is_increasing_function candidate_function :=
by
  sorry

end candidate_function_is_odd_and_increasing_l93_93321


namespace parade_team_people_count_min_l93_93439

theorem parade_team_people_count_min (n : ℕ) :
  n ≥ 1000 ∧ n % 5 = 0 ∧ n % 4 = 3 ∧ n % 3 = 2 ∧ n % 2 = 1 → n = 1045 :=
by
  sorry

end parade_team_people_count_min_l93_93439


namespace cost_of_three_stamps_is_correct_l93_93908

-- Define the cost of one stamp
def cost_of_one_stamp : ℝ := 0.34

-- Define the number of stamps
def number_of_stamps : ℕ := 3

-- Define the expected total cost for three stamps
def expected_cost : ℝ := 1.02

-- Prove that the cost of three stamps is equal to the expected cost
theorem cost_of_three_stamps_is_correct : cost_of_one_stamp * number_of_stamps = expected_cost :=
by
  sorry

end cost_of_three_stamps_is_correct_l93_93908


namespace solve_inequality_l93_93239

theorem solve_inequality (x : ℝ) : (1 / (x + 2) + 4 / (x + 8) ≤ 3 / 4) ↔ ((-8 < x ∧ x ≤ -4) ∨ (-4 ≤ x ∧ x ≤ 4 / 3)) ∧ x ≠ -2 ∧ x ≠ -8 :=
by
  sorry

end solve_inequality_l93_93239


namespace no_base6_digit_d_divisible_by_7_l93_93339

theorem no_base6_digit_d_divisible_by_7 : 
∀ d : ℕ, (d < 6) → ¬ (654 + 42 * d) % 7 = 0 :=
by
  intro d h
  -- Proof is omitted as requested
  sorry

end no_base6_digit_d_divisible_by_7_l93_93339


namespace find_D_coordinates_l93_93639

theorem find_D_coordinates:
  ∀ (A B C : (ℝ × ℝ)), 
  A = (-2, 5) ∧ C = (3, 7) ∧ B = (-3, 0) →
  ∃ D : (ℝ × ℝ), D = (2, 2) :=
by
  sorry

end find_D_coordinates_l93_93639


namespace milkman_profit_percentage_l93_93899

noncomputable def profit_percentage (x : ℝ) : ℝ :=
  let cp_per_litre := x
  let sp_per_litre := 2 * x
  let mixture_litres := 8
  let milk_litres := 6
  let cost_price := milk_litres * cp_per_litre
  let selling_price := mixture_litres * sp_per_litre
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage

theorem milkman_profit_percentage (x : ℝ) 
  (h : x > 0) : 
  profit_percentage x = 166.67 :=
by
  sorry

end milkman_profit_percentage_l93_93899


namespace maximize_profit_at_14_yuan_and_720_l93_93170

def initial_cost : ℝ := 8
def initial_price : ℝ := 10
def initial_units_sold : ℝ := 200
def decrease_units_per_half_yuan_increase : ℝ := 10
def increase_price_per_step : ℝ := 0.5

noncomputable def profit (x : ℝ) : ℝ := 
  let selling_price := initial_price + increase_price_per_step * x
  let units_sold := initial_units_sold - decrease_units_per_half_yuan_increase * x
  (selling_price - initial_cost) * units_sold

theorem maximize_profit_at_14_yuan_and_720 :
  profit 8 = 720 ∧ (initial_price + increase_price_per_step * 8 = 14) :=
by
  sorry

end maximize_profit_at_14_yuan_and_720_l93_93170


namespace count_ones_digits_of_numbers_divisible_by_4_and_3_l93_93682

theorem count_ones_digits_of_numbers_divisible_by_4_and_3 :
  let eligible_numbers := { n : ℕ | n < 100 ∧ n % 4 = 0 ∧ n % 3 = 0 }
  ∃ (digits : Finset ℕ), 
    (∀ n ∈ eligible_numbers, n % 10 ∈ digits) ∧
    digits.card = 5 :=
by
  sorry

end count_ones_digits_of_numbers_divisible_by_4_and_3_l93_93682


namespace smallest_integer_representation_l93_93860

theorem smallest_integer_representation :
  ∃ (A B C : ℕ), 0 ≤ A ∧ A < 5 ∧ 0 ≤ B ∧ B < 7 ∧ 0 ≤ C ∧ C < 4 ∧ 6 * A = 8 * B ∧ 6 * A = 5 * C ∧ 8 * B = 5 * C ∧ (6 * A) = 24 :=
  sorry

end smallest_integer_representation_l93_93860


namespace centroid_distance_l93_93747

-- Define the given conditions and final goal
theorem centroid_distance (a b c p q r : ℝ) 
  (ha : a ≠ 0)  (hb : b ≠ 0)  (hc : c ≠ 0)
  (centroid : p = a / 3 ∧ q = b / 3 ∧ r = c / 3) 
  (plane_distance : (1 / (1 / a^2 + 1 / b^2 + 1 / c^2).sqrt) = 2) :
  (1 / p^2 + 1 / q^2 + 1 / r^2) = 2.25 := 
by 
  -- Start proof here
  sorry

end centroid_distance_l93_93747


namespace percentage_increase_l93_93228

variable (E : ℝ) (P : ℝ)
variable (h1 : 1.36 * E = 495)
variable (h2 : (1 + P) * E = 454.96)

theorem percentage_increase :
  P = 0.25 :=
by
  sorry

end percentage_increase_l93_93228


namespace find_average_speed_l93_93777

noncomputable def average_speed (distance1 distance2 : ℝ) (time1 time2 : ℝ) : ℝ := 
  (distance1 + distance2) / (time1 + time2)

theorem find_average_speed :
  average_speed 1000 1000 10 4 = 142.86 := by
  sorry

end find_average_speed_l93_93777


namespace find_g_720_l93_93467

noncomputable def g (n : ℕ) : ℕ := sorry

axiom g_multiplicative : ∀ (x y : ℕ), g (x * y) = g x + g y
axiom g_8 : g 8 = 12
axiom g_12 : g 12 = 16

theorem find_g_720 : g 720 = 44 := by sorry

end find_g_720_l93_93467


namespace unique_two_digit_integer_s_l93_93383

-- We define s to satisfy the two given conditions.
theorem unique_two_digit_integer_s (s : ℕ) (h1 : 13 * s % 100 = 52) (h2 : 1 ≤ s) (h3 : s ≤ 99) : s = 4 :=
sorry

end unique_two_digit_integer_s_l93_93383


namespace solve_equation_l93_93342

theorem solve_equation (x : ℝ) : x^2 = 5 * x → x = 0 ∨ x = 5 := 
by
  sorry

end solve_equation_l93_93342


namespace transfer_people_eq_l93_93459

theorem transfer_people_eq : ∃ x : ℕ, 22 + x = 2 * (26 - x) := 
by 
  -- hypothesis and equation statement
  sorry

end transfer_people_eq_l93_93459


namespace crayons_lost_or_given_away_l93_93030

theorem crayons_lost_or_given_away (given_away lost : ℕ) (H_given_away : given_away = 213) (H_lost : lost = 16) :
  given_away + lost = 229 :=
by
  sorry

end crayons_lost_or_given_away_l93_93030


namespace total_potatoes_l93_93703

theorem total_potatoes (monday_to_friday_potatoes : ℕ) (double_potatoes : ℕ) 
(lunch_potatoes_mon_fri : ℕ) (lunch_potatoes_weekend : ℕ)
(dinner_potatoes_mon_fri : ℕ) (dinner_potatoes_weekend : ℕ)
(h1 : monday_to_friday_potatoes = 5)
(h2 : double_potatoes = 10)
(h3 : lunch_potatoes_mon_fri = 25)
(h4 : lunch_potatoes_weekend = 20)
(h5 : dinner_potatoes_mon_fri = 40)
(h6 : dinner_potatoes_weekend = 26)
  : monday_to_friday_potatoes * 5 + double_potatoes * 2 + dinner_potatoes_mon_fri * 5 + (double_potatoes + 3) * 2 = 111 := 
sorry

end total_potatoes_l93_93703


namespace min_value_at_x_zero_l93_93192

noncomputable def f (x : ℝ) := Real.sqrt (x^2 + (x + 1)^2) + Real.sqrt (x^2 + (x - 1)^2)

theorem min_value_at_x_zero : ∀ x : ℝ, f x ≥ f 0 := by
  sorry

end min_value_at_x_zero_l93_93192


namespace geom_seq_42_l93_93074

variable {α : Type*} [Field α] [CharZero α]

noncomputable def a_n (n : ℕ) (a1 q : α) : α := a1 * q ^ n

theorem geom_seq_42 (a1 q : α) (h1 : a1 = 3) (h2 : a1 * (1 + q^2 + q^4) = 21) :
  a1 * (q^2 + q^4 + q^6) = 42 := 
by
  sorry

end geom_seq_42_l93_93074


namespace january_salary_l93_93278

variable (J F M A My : ℕ)

axiom average_salary_1 : (J + F + M + A) / 4 = 8000
axiom average_salary_2 : (F + M + A + My) / 4 = 8400
axiom may_salary : My = 6500

theorem january_salary : J = 4900 :=
by
  /- To be filled with the proof steps applying the given conditions -/
  sorry

end january_salary_l93_93278


namespace range_of_a_l93_93585

theorem range_of_a {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h1 : x + y + 4 = 2 * x * y) (h2 : ∀ (x y : ℝ), x^2 + 2*x*y + y^2 - a*x - a*y + 1 ≥ 0) :
  a ≤ 17/4 := sorry

end range_of_a_l93_93585


namespace football_points_difference_l93_93435

theorem football_points_difference :
  let points_per_touchdown := 7
  let brayden_gavin_touchdowns := 7
  let cole_freddy_touchdowns := 9
  let brayden_gavin_points := brayden_gavin_touchdowns * points_per_touchdown
  let cole_freddy_points := cole_freddy_touchdowns * points_per_touchdown
  cole_freddy_points - brayden_gavin_points = 14 :=
by sorry

end football_points_difference_l93_93435


namespace girls_attending_event_l93_93118

theorem girls_attending_event (total_students girls_attending boys_attending : ℕ) 
    (h1 : total_students = 1500) 
    (h2 : girls_attending = 3 / 5 * girls) 
    (h3 : boys_attending = 2 / 3 * (total_students - girls)) 
    (h4 : girls_attending + boys_attending = 900) : 
    girls_attending = 900 := 
by 
    sorry

end girls_attending_event_l93_93118


namespace sandy_age_l93_93117

variable (S M N : ℕ)

theorem sandy_age (h1 : M = S + 20)
                  (h2 : (S : ℚ) / M = 7 / 9)
                  (h3 : S + M + N = 120)
                  (h4 : N - M = (S - M) / 2) :
                  S = 70 := 
sorry

end sandy_age_l93_93117


namespace tank_capacity_l93_93819

theorem tank_capacity (x : ℝ) (h : 0.24 * x = 120) : x = 500 := 
sorry

end tank_capacity_l93_93819


namespace calc1_calc2_calc3_calc4_l93_93240

-- Problem 1
theorem calc1 : (-2: ℝ) ^ 2 - (7 - Real.pi) ^ 0 - (1 / 3) ^ (-1: ℝ) = 0 := by
  sorry

-- Problem 2
variable (m : ℝ)
theorem calc2 : 2 * m ^ 3 * 3 * m - (2 * m ^ 2) ^ 2 + m ^ 6 / m ^ 2 = 3 * m ^ 4 := by
  sorry

-- Problem 3
variable (a : ℝ)
theorem calc3 : (a + 1) ^ 2 + (a + 1) * (a - 2) = 2 * a ^ 2 + a - 1 := by
  sorry

-- Problem 4
variables (x y : ℝ)
theorem calc4 : (x + y - 1) * (x - y - 1) = x ^ 2 - 2 * x + 1 - y ^ 2 := by
  sorry

end calc1_calc2_calc3_calc4_l93_93240


namespace max_length_MN_l93_93965

theorem max_length_MN (p : ℝ) (h a b c r : ℝ)
  (h_perimeter : a + b + c = 2 * p)
  (h_tangent : r = (a * h) / (2 * p))
  (h_parallel : ∀ h r : ℝ, ∃ k : ℝ, MN = k * (1 - 2 * r / h)) :
  ∀ k : ℝ, MN = (p / 4) :=
sorry

end max_length_MN_l93_93965


namespace cone_volume_calc_l93_93450

noncomputable def cone_volume (diameter slant_height: ℝ) : ℝ :=
  let r := diameter / 2
  let h := Real.sqrt (slant_height^2 - r^2)
  (1 / 3) * Real.pi * r^2 * h

theorem cone_volume_calc :
  cone_volume 12 10 = 96 * Real.pi :=
by
  sorry

end cone_volume_calc_l93_93450


namespace vermont_clicked_ads_l93_93543

theorem vermont_clicked_ads :
  let ads1 := 12
  let ads2 := 2 * ads1
  let ads3 := ads2 + 24
  let ads4 := 3 * ads2 / 4
  let total_ads := ads1 + ads2 + ads3 + ads4
  let ads_clicked := 2 * total_ads / 3
  ads_clicked = 68 := by
  let ads1 := 12
  let ads2 := 2 * ads1
  let ads3 := ads2 + 24
  let ads4 := 3 * ads2 / 4
  let total_ads := ads1 + ads2 + ads3 + ads4
  let ads_clicked := 2 * total_ads / 3
  have h1 : ads_clicked = 68 := by sorry
  exact h1

end vermont_clicked_ads_l93_93543


namespace solve_system_of_equations_l93_93859

variable (a x y z : ℝ)

theorem solve_system_of_equations (h1 : x^2 + y^2 - 2 * z^2 = 2 * a^2)
                                  (h2 : x + y + 2 * z = 4 * (a^2 + 1))
                                  (h3 : z^2 - x * y = a^2) :
                                  (x = a^2 + a + 1 ∧ y = a^2 - a + 1 ∧ z = a^2 + 1) ∨
                                  (x = a^2 - a + 1 ∧ y = a^2 + a + 1 ∧ z = a^2 + 1) :=
by
  sorry

end solve_system_of_equations_l93_93859


namespace find_center_of_ellipse_l93_93381

-- Defining the equation of the ellipse
def ellipse (x y : ℝ) : Prop := 2*x^2 + 2*x*y + y^2 + 2*x + 2*y - 4 = 0

-- The coordinates of the center
def center_of_ellipse : ℝ × ℝ := (0, -1)

-- The theorem asserting the center of the ellipse
theorem find_center_of_ellipse (x y : ℝ) (h : ellipse x y) : (x, y) = center_of_ellipse :=
sorry

end find_center_of_ellipse_l93_93381


namespace rectangle_area_percentage_increase_l93_93010

theorem rectangle_area_percentage_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  let A := l * w
  let len_inc := 1.3 * l
  let wid_inc := 1.15 * w
  let A_new := len_inc * wid_inc
  let percentage_increase := ((A_new - A) / A) * 100
  percentage_increase = 49.5 :=
by
  sorry

end rectangle_area_percentage_increase_l93_93010


namespace expression_eval_l93_93054

theorem expression_eval : (3^2 - 3) - (5^2 - 5) * 2 + (6^2 - 6) = -4 :=
by sorry

end expression_eval_l93_93054


namespace max_area_of_garden_l93_93664

theorem max_area_of_garden (L : ℝ) (hL : 0 ≤ L) :
  ∃ x y : ℝ, x + 2 * y = L ∧ x ≥ 0 ∧ y ≥ 0 ∧ x * y = L^2 / 8 :=
by
  sorry

end max_area_of_garden_l93_93664


namespace first_day_is_sunday_l93_93861

-- Define the days of the week
inductive Day
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

open Day

-- Function to determine the day of the week for a given day number
def day_of_month (n : ℕ) (start_day : Day) : Day :=
  match n % 7 with
  | 0 => start_day
  | 1 => match start_day with
          | Sunday    => Monday
          | Monday    => Tuesday
          | Tuesday   => Wednesday
          | Wednesday => Thursday
          | Thursday  => Friday
          | Friday    => Saturday
          | Saturday  => Sunday
  | 2 => match start_day with
          | Sunday    => Tuesday
          | Monday    => Wednesday
          | Tuesday   => Thursday
          | Wednesday => Friday
          | Thursday  => Saturday
          | Friday    => Sunday
          | Saturday  => Monday
-- ... and so on for the rest of the days of the week.
  | _ => start_day -- Assuming the pattern continues accordingly.

-- Prove that the first day of the month is a Sunday given that the 18th day of the month is a Wednesday.
theorem first_day_is_sunday (h : day_of_month 18 Wednesday = Wednesday) : day_of_month 1 Wednesday = Sunday :=
  sorry

end first_day_is_sunday_l93_93861


namespace common_factor_of_polynomial_l93_93047

variables (x y m n : ℝ)

theorem common_factor_of_polynomial :
  ∃ (k : ℝ), (2 * (m - n)) = k ∧ (4 * x * (m - n) + 2 * y * (m - n)^2) = k * (2 * x * (m - n)) :=
sorry

end common_factor_of_polynomial_l93_93047


namespace mod_remainder_l93_93914

theorem mod_remainder (a b c x: ℤ):
    a = 9 → b = 5 → c = 3 → x = 7 →
    (a^6 + b^7 + c^8) % x = 4 :=
by
  intros
  sorry

end mod_remainder_l93_93914


namespace tire_circumference_l93_93869

theorem tire_circumference (rpm : ℕ) (speed_kmh : ℕ) (C : ℝ) 
  (h1 : rpm = 400) 
  (h2 : speed_kmh = 144) 
  (h3 : (speed_kmh * 1000 / 60) = (rpm * C)) : 
  C = 6 :=
by
  sorry

end tire_circumference_l93_93869


namespace victor_weekly_earnings_l93_93486

def wage_per_hour : ℕ := 12
def hours_monday : ℕ := 5
def hours_tuesday : ℕ := 6
def hours_wednesday : ℕ := 7
def hours_thursday : ℕ := 4
def hours_friday : ℕ := 8

def earnings_monday := hours_monday * wage_per_hour
def earnings_tuesday := hours_tuesday * wage_per_hour
def earnings_wednesday := hours_wednesday * wage_per_hour
def earnings_thursday := hours_thursday * wage_per_hour
def earnings_friday := hours_friday * wage_per_hour

def total_earnings := earnings_monday + earnings_tuesday + earnings_wednesday + earnings_thursday + earnings_friday

theorem victor_weekly_earnings : total_earnings = 360 := by
  sorry

end victor_weekly_earnings_l93_93486


namespace total_workers_l93_93954

-- Definitions for the conditions in the problem
def avg_salary_all : ℝ := 8000
def num_technicians : ℕ := 7
def avg_salary_technicians : ℝ := 18000
def avg_salary_non_technicians : ℝ := 6000

-- Main theorem stating the total number of workers
theorem total_workers (W : ℕ) :
  (7 * avg_salary_technicians + (W - 7) * avg_salary_non_technicians = W * avg_salary_all) → W = 42 :=
by
  sorry

end total_workers_l93_93954


namespace problem_scores_ordering_l93_93858

variable {J K L R : ℕ}

theorem problem_scores_ordering (h1 : J > K) (h2 : J > L) (h3 : J > R)
                                (h4 : L > min K R) (h5 : R > min K L)
                                (h6 : (J ≠ K) ∧ (J ≠ L) ∧ (J ≠ R) ∧ (K ≠ L) ∧ (K ≠ R) ∧ (L ≠ R)) :
                                K < L ∧ L < R :=
sorry

end problem_scores_ordering_l93_93858


namespace y_coord_range_of_M_l93_93813

theorem y_coord_range_of_M :
  ∀ (M : ℝ × ℝ), ((M.1 + 1)^2 + M.2^2 = 2) → 
  ((M.1 - 2)^2 + M.2^2 + M.1^2 + M.2^2 ≤ 10) →
  - (Real.sqrt 7) / 2 ≤ M.2 ∧ M.2 ≤ (Real.sqrt 7) / 2 := 
by 
  sorry

end y_coord_range_of_M_l93_93813


namespace charge_difference_percentage_l93_93584

-- Given definitions
variables (G R P : ℝ)
def hotelR := 1.80 * G
def hotelP := 0.90 * G

-- Theorem statement
theorem charge_difference_percentage (G : ℝ) (hR : R = 1.80 * G) (hP : P = 0.90 * G) :
  (R - P) / R * 100 = 50 :=
by sorry

end charge_difference_percentage_l93_93584


namespace max_value_of_quadratic_l93_93619

theorem max_value_of_quadratic (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) : 
  ∃ y, y = x * (1 - 2 * x) ∧ y ≤ 1 / 8 ∧ (y = 1 / 8 ↔ x = 1 / 4) :=
by sorry

end max_value_of_quadratic_l93_93619


namespace Keith_picked_6_apples_l93_93373

def m : ℝ := 7.0
def n : ℝ := 3.0
def t : ℝ := 10.0

noncomputable def r_m := m - n
noncomputable def k := t - r_m

-- Theorem Statement confirming Keith picked 6.0 apples
theorem Keith_picked_6_apples : k = 6.0 := by
  sorry

end Keith_picked_6_apples_l93_93373


namespace find_third_number_l93_93962

theorem find_third_number (A B C : ℝ) (h1 : (A + B + C) / 3 = 48) (h2 : (A + B) / 2 = 56) : C = 32 :=
by sorry

end find_third_number_l93_93962


namespace equation_of_line_through_P_l93_93801

theorem equation_of_line_through_P (P : (ℝ × ℝ)) (A B : (ℝ × ℝ))
  (hP : P = (1, 3))
  (hMidpoint : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hA : A.2 = 0)
  (hB : B.1 = 0) :
  ∃ c : ℝ, 3 * c + 1 = 3 ∧ (3 * A.1 / c + A.2 / 6 = 1) ∧ (3 * B.1 / c + B.2 / 6 = 1) := sorry

end equation_of_line_through_P_l93_93801


namespace angle_BDC_is_15_degrees_l93_93018

theorem angle_BDC_is_15_degrees (A B C D : Type) (AB AC AD CD : ℝ) (angle_BAC : ℝ) :
  AB = AC → AC = AD → CD = 2 * AC → angle_BAC = 30 →
  ∃ angle_BDC, angle_BDC = 15 := 
by
  sorry

end angle_BDC_is_15_degrees_l93_93018


namespace shelves_used_l93_93923

-- Definitions from conditions
def initial_bears : ℕ := 6
def shipment_bears : ℕ := 18
def bears_per_shelf : ℕ := 6

-- Theorem statement
theorem shelves_used : (initial_bears + shipment_bears) / bears_per_shelf = 4 := by
  sorry

end shelves_used_l93_93923


namespace equal_intercepts_on_both_axes_l93_93773

theorem equal_intercepts_on_both_axes (m : ℝ) :
  (5 - 2 * m ≠ 0) ∧
  (- (5 - 2 * m) / (m^2 - 2 * m - 3) = - (5 - 2 * m) / (2 * m^2 + m - 1)) ↔ m = -2 :=
by sorry

end equal_intercepts_on_both_axes_l93_93773


namespace max_gold_coins_l93_93527

theorem max_gold_coins : ∃ n : ℕ, (∃ k : ℕ, n = 7 * k + 2) ∧ 50 < n ∧ n < 150 ∧ n = 149 :=
by
  sorry

end max_gold_coins_l93_93527


namespace hyperbola_condition_l93_93744

theorem hyperbola_condition (m : ℝ) : (∀ x y : ℝ, x^2 + m * y^2 = 1 → m < 0 ↔ x ≠ 0 ∧ y ≠ 0) :=
by
  sorry

end hyperbola_condition_l93_93744


namespace averageSpeed_l93_93190

-- Define the total distance driven by Jane
def totalDistance : ℕ := 200

-- Define the total time duration from 6 a.m. to 11 a.m.
def totalTime : ℕ := 5

-- Theorem stating that the average speed is 40 miles per hour
theorem averageSpeed (h1 : totalDistance = 200) (h2 : totalTime = 5) : totalDistance / totalTime = 40 := 
by
  sorry

end averageSpeed_l93_93190


namespace sqrt_inequality_sum_of_squares_geq_sum_of_products_l93_93237

theorem sqrt_inequality : (Real.sqrt 6) + (Real.sqrt 10) > (2 * Real.sqrt 3) + 2 := by
  sorry

theorem sum_of_squares_geq_sum_of_products (a b c : ℝ) : 
    a^2 + b^2 + c^2 ≥ a * b + b * c + a * c := by
  sorry

end sqrt_inequality_sum_of_squares_geq_sum_of_products_l93_93237


namespace carrie_mom_money_l93_93561

theorem carrie_mom_money :
  ∀ (sweater_cost t_shirt_cost shoes_cost left_money total_money : ℕ),
  sweater_cost = 24 →
  t_shirt_cost = 6 →
  shoes_cost = 11 →
  left_money = 50 →
  total_money = sweater_cost + t_shirt_cost + shoes_cost + left_money →
  total_money = 91 :=
sorry

end carrie_mom_money_l93_93561


namespace brenda_age_l93_93570

theorem brenda_age
  (A B J : ℕ)
  (h1 : A = 4 * B)
  (h2 : J = B + 9)
  (h3 : A = J)
  : B = 3 :=
by 
  sorry

end brenda_age_l93_93570


namespace amount_on_table_A_l93_93184

-- Definitions based on conditions
variables (A B C : ℝ)
variables (h1 : B = 2 * C)
variables (h2 : C = A + 20)
variables (h3 : A + B + C = 220)

-- Theorem statement
theorem amount_on_table_A : A = 40 :=
by
  -- This is expected to be filled in with the proof steps, but we skip it with 'sorry'
  sorry

end amount_on_table_A_l93_93184


namespace percentage_female_officers_on_duty_l93_93102

theorem percentage_female_officers_on_duty:
  ∀ (total_on_duty female_on_duty total_female_officers : ℕ),
    total_on_duty = 160 →
    female_on_duty = total_on_duty / 2 →
    total_female_officers = 500 →
    female_on_duty / total_female_officers * 100 = 16 :=
by
  intros total_on_duty female_on_duty total_female_officers h1 h2 h3
  -- Ensure types are correct
  change total_on_duty = 160 at h1
  change female_on_duty = total_on_duty / 2 at h2
  change total_female_officers = 500 at h3
  sorry

end percentage_female_officers_on_duty_l93_93102


namespace total_investment_is_correct_l93_93849

-- Define principal, rate, and number of years
def principal : ℝ := 8000
def rate : ℝ := 0.04
def years : ℕ := 10

-- Define the formula for compound interest
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- State the theorem
theorem total_investment_is_correct :
  compound_interest principal rate years = 11842 :=
by
  sorry

end total_investment_is_correct_l93_93849


namespace blue_hat_cost_l93_93732

theorem blue_hat_cost :
  ∀ (total_hats green_hats total_price green_hat_price blue_hat_price) 
  (B : ℕ),
  total_hats = 85 →
  green_hats = 30 →
  total_price = 540 →
  green_hat_price = 7 →
  blue_hat_price = B →
  (30 * 7) + (55 * B) = 540 →
  B = 6 := sorry

end blue_hat_cost_l93_93732


namespace randy_trip_distance_l93_93595

theorem randy_trip_distance (x : ℝ) (h1 : x = x / 4 + 30 + x / 10 + (x - (x / 4 + 30 + x / 10))) :
  x = 60 :=
by {
  sorry -- Placeholder for the actual proof
}

end randy_trip_distance_l93_93595


namespace calculate_expression_l93_93001

-- Define the numerator and denominator
def numerator := 11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2 + 1
def denominator := 2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10

-- Prove the expression equals 1
theorem calculate_expression : (numerator / denominator) = 1 := by
  sorry

end calculate_expression_l93_93001


namespace find_tan_angle_F2_F1_B_l93_93610

-- Definitions for the points and chord lengths
def F1 : Type := ℝ × ℝ
def F2 : Type := ℝ × ℝ
def A : Type := ℝ × ℝ
def B : Type := ℝ × ℝ

-- Given distances
def F1A : ℝ := 3
def AB : ℝ := 4
def BF1 : ℝ := 5

-- The angle we want to find the tangent of
def angle_F2_F1_B (F1 F2 A B : Type) : ℝ := sorry -- Placeholder for angle calculation

-- The main theorem to prove
theorem find_tan_angle_F2_F1_B (F1 F2 A B : Type) (F1A_dist : F1A = 3) (AB_dist : AB = 4) (BF1_dist : BF1 = 5) :
  angle_F2_F1_B F1 F2 A B = 1 / 7 :=
sorry

end find_tan_angle_F2_F1_B_l93_93610


namespace intersecting_line_circle_condition_l93_93168

theorem intersecting_line_circle_condition {a b : ℝ} (h : ∃ x y : ℝ, x^2 + y^2 = 1 ∧ x / a + y / b = 1) :
  (1 / a ^ 2) + (1 / b ^ 2) ≥ 1 :=
sorry

end intersecting_line_circle_condition_l93_93168


namespace shortest_side_of_triangle_l93_93843

noncomputable def triangle_shortest_side_length (a b r : ℝ) (shortest : ℝ) : Prop :=
a = 8 ∧ b = 6 ∧ r = 4 ∧ shortest = 12

theorem shortest_side_of_triangle 
  (a b r shortest : ℝ) 
  (h : triangle_shortest_side_length a b r shortest) : shortest = 12 :=
sorry

end shortest_side_of_triangle_l93_93843


namespace problem_π_digit_sequence_l93_93887

def f (n : ℕ) : ℕ :=
  match n with
  | 1  => 1
  | 2  => 4
  | 3  => 1
  | 4  => 5
  | 5  => 9
  | 6  => 2
  | 7  => 6
  | 8  => 5
  | 9  => 3
  | 10 => 5
  | _  => 0  -- for simplicity we define other cases arbitrarily

theorem problem_π_digit_sequence :
  ∃ n : ℕ, n > 0 ∧ f (f (f (f (f 10)))) = 1 := by
  sorry

end problem_π_digit_sequence_l93_93887


namespace polygon_diagonals_l93_93481

-- Definitions of the conditions
def sum_of_angles (n : ℕ) : ℝ := (n - 2) * 180 + 360

def num_diagonals (n : ℕ) : ℤ := n * (n - 3) / 2

-- Theorem statement
theorem polygon_diagonals (n : ℕ) (h : sum_of_angles n = 2160) : num_diagonals n = 54 :=
sorry

end polygon_diagonals_l93_93481


namespace solve_inequality_l93_93397

theorem solve_inequality (x : ℝ) : 
  3 * (2 * x - 1) - 2 * (x + 1) ≤ 1 → x ≤ 3 / 2 :=
by
  sorry

end solve_inequality_l93_93397


namespace factorial_trailing_digits_l93_93276

theorem factorial_trailing_digits (n : ℕ) :
  ¬ ∃ k : ℕ, (n! / 10^k) % 10000 = 1976 ∧ k > 0 := 
sorry

end factorial_trailing_digits_l93_93276


namespace apples_sold_fresh_l93_93453

-- Definitions per problem conditions
def total_production : Float := 8.0
def initial_percentage_mixed : Float := 0.30
def percentage_increase_per_million : Float := 0.05
def percentage_for_apple_juice : Float := 0.60
def percentage_sold_fresh : Float := 0.40

-- We need to prove that given the conditions, the amount of apples sold fresh is 2.24 million tons
theorem apples_sold_fresh :
  ( (total_production - (initial_percentage_mixed * total_production)) * percentage_sold_fresh = 2.24 ) :=
by
  sorry

end apples_sold_fresh_l93_93453


namespace smallest_integer_value_of_x_satisfying_eq_l93_93708

theorem smallest_integer_value_of_x_satisfying_eq (x : ℤ) (h : |x^2 - 5*x + 6| = 14) : 
  ∃ y : ℤ, (y = -1) ∧ ∀ z : ℤ, (|z^2 - 5*z + 6| = 14) → (y ≤ z) :=
sorry

end smallest_integer_value_of_x_satisfying_eq_l93_93708


namespace average_value_of_T_l93_93987

noncomputable def expected_value_T : ℕ := 22

theorem average_value_of_T (boys girls : ℕ) (boy_pair girl_pair : Prop) (T : ℕ) :
  boys = 9 → girls = 15 →
  boy_pair ∧ girl_pair →
  T = expected_value_T :=
by
  intros h_boys h_girls h_pairs
  sorry

end average_value_of_T_l93_93987


namespace jill_salary_l93_93630

-- Defining the conditions
variables (S : ℝ) -- Jill's net monthly salary
variables (discretionary_income : ℝ) -- One fifth of her net monthly salary
variables (vacation_fund : ℝ) -- 30% of discretionary income into a vacation fund
variables (savings : ℝ) -- 20% of discretionary income into savings
variables (eating_out_socializing : ℝ) -- 35% of discretionary income on eating out and socializing
variables (leftover : ℝ) -- The remaining amount, which is $99

-- Given Conditions
-- One fifth of her net monthly salary left as discretionary income
def one_fifth_of_salary : Prop := discretionary_income = (1/5) * S

-- 30% into a vacation fund
def vacation_allocation : Prop := vacation_fund = 0.30 * discretionary_income

-- 20% into savings
def savings_allocation : Prop := savings = 0.20 * discretionary_income

-- 35% on eating out and socializing
def socializing_allocation : Prop := eating_out_socializing = 0.35 * discretionary_income

-- This leaves her with $99
def leftover_amount : Prop := leftover = 99

-- Eqution considering all conditions results her leftover being $99
def income_allocation : Prop := 
  vacation_fund + savings + eating_out_socializing + leftover = discretionary_income

-- The main proof goal: given all the conditions, Jill's net monthly salary is $3300
theorem jill_salary : 
  one_fifth_of_salary S discretionary_income → 
  vacation_allocation discretionary_income vacation_fund → 
  savings_allocation discretionary_income savings → 
  socializing_allocation discretionary_income eating_out_socializing → 
  leftover_amount leftover → 
  income_allocation discretionary_income vacation_fund savings eating_out_socializing leftover → 
  S = 3300 := by sorry

end jill_salary_l93_93630


namespace vanessa_video_files_initial_l93_93120

theorem vanessa_video_files_initial (m v r d t : ℕ) (h1 : m = 13) (h2 : r = 33) (h3 : d = 10) (h4 : t = r + d) (h5 : t = m + v) : v = 30 :=
by
  sorry

end vanessa_video_files_initial_l93_93120


namespace value_of_expression_l93_93571

theorem value_of_expression (x : ℝ) (h : |x| = x + 2) : 19 * x ^ 99 + 3 * x + 27 = 5 :=
by
  have h1: x ≥ -2 := sorry
  have h2: x = -1 := sorry
  sorry

end value_of_expression_l93_93571


namespace ladder_base_distance_l93_93942

noncomputable def length_of_ladder : ℝ := 8.5
noncomputable def height_on_wall : ℝ := 7.5

theorem ladder_base_distance (x : ℝ) (h : x ^ 2 + height_on_wall ^ 2 = length_of_ladder ^ 2) :
  x = 4 :=
by sorry

end ladder_base_distance_l93_93942


namespace required_run_rate_l93_93086

/-
In the first 10 overs of a cricket game, the run rate was 3.5. 
What should be the run rate in the remaining 40 overs to reach the target of 320 runs?
-/

def run_rate_in_10_overs : ℝ := 3.5
def overs_played : ℕ := 10
def target_runs : ℕ := 320 
def remaining_overs : ℕ := 40

theorem required_run_rate : 
  (target_runs - (run_rate_in_10_overs * overs_played)) / remaining_overs = 7.125 := by 
sorry

end required_run_rate_l93_93086


namespace coordinates_with_respect_to_origin_l93_93971

theorem coordinates_with_respect_to_origin :
  ∀ (point : ℝ × ℝ), point = (3, -2) → point = (3, -2) := by
  intro point h
  exact h

end coordinates_with_respect_to_origin_l93_93971


namespace discount_is_15_point_5_percent_l93_93424

noncomputable def wholesale_cost (W : ℝ) := W
noncomputable def retail_price (W : ℝ) := 1.5384615384615385 * W
noncomputable def selling_price (W : ℝ) := 1.3 * W
noncomputable def discount_percentage (W : ℝ) := 
  let D := retail_price W - selling_price W
  (D / retail_price W) * 100

theorem discount_is_15_point_5_percent (W : ℝ) (hW : W > 0) : 
  discount_percentage W = 15.5 := 
by 
  sorry

end discount_is_15_point_5_percent_l93_93424


namespace larry_correct_evaluation_l93_93051

theorem larry_correct_evaluation (a b c d e : ℝ) 
(Ha : a = 5) (Hb : b = 3) (Hc : c = 6) (Hd : d = 4) :
a - b + c + d - e = a - (b - (c + (d - e))) → e = 0 :=
by
  -- Not providing the actual proof
  sorry

end larry_correct_evaluation_l93_93051


namespace smaller_balloon_radius_is_correct_l93_93452

-- Condition: original balloon radius
def original_balloon_radius : ℝ := 2

-- Condition: number of smaller balloons
def num_smaller_balloons : ℕ := 64

-- Question (to be proved): Radius of each smaller balloon
theorem smaller_balloon_radius_is_correct :
  ∃ r : ℝ, (4/3) * Real.pi * (original_balloon_radius^3) = num_smaller_balloons * (4/3) * Real.pi * (r^3) ∧ r = 1/2 := 
by {
  sorry
}

end smaller_balloon_radius_is_correct_l93_93452


namespace auntie_em_parking_probability_l93_93103

theorem auntie_em_parking_probability :
  let total_spaces := 20
  let cars := 15
  let empty_spaces := total_spaces - cars
  let possible_configurations := Nat.choose total_spaces cars
  let unfavourable_configurations := Nat.choose (empty_spaces - 8 + 5) (empty_spaces - 8)
  let favourable_probability := 1 - ((unfavourable_configurations : ℚ) / (possible_configurations : ℚ))
  (favourable_probability = 1839 / 1938) :=
by
  -- sorry to skip the actual proof
  sorry

end auntie_em_parking_probability_l93_93103


namespace probability_two_dice_sum_gt_8_l93_93248

def num_ways_to_get_sum_at_most_8 := 
  1 + 2 + 3 + 4 + 5 + 6 + 5

def total_outcomes := 36

def probability_sum_greater_than_8 : ℚ := 1 - (num_ways_to_get_sum_at_most_8 / total_outcomes)

theorem probability_two_dice_sum_gt_8 :
  probability_sum_greater_than_8 = 5 / 18 :=
by
  sorry

end probability_two_dice_sum_gt_8_l93_93248


namespace linear_equation_solution_l93_93096

theorem linear_equation_solution (a b : ℤ) (x y : ℤ) (h1 : x = 2) (h2 : y = -1) (h3 : a * x + b * y = -1) : 
  1 + 2 * a - b = 0 :=
by
  sorry

end linear_equation_solution_l93_93096


namespace tank_fill_rate_l93_93214

theorem tank_fill_rate
  (length width depth : ℝ)
  (time_to_fill : ℝ)
  (h_length : length = 10)
  (h_width : width = 6)
  (h_depth : depth = 5)
  (h_time : time_to_fill = 60) : 
  (length * width * depth) / time_to_fill = 5 :=
by
  -- Proof would go here
  sorry

end tank_fill_rate_l93_93214


namespace inequality_proof_equality_condition_l93_93337

theorem inequality_proof (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ( (3 * a * b * c / (a * b + a * c + b * c)) ^ (a^2 + b^2 + c^2) ) ≥ (a ^ (b * c) * b ^ (a * c) * c ^ (a * b)) := 
sorry

theorem equality_condition (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ( (3 * a * b * c / (a * b + a * c + b * c)) ^ (a^2 + b^2 + c^2) ) = (a ^ (b * c) * b ^ (a * c) * c ^ (a * b)) ↔ a = b ∧ b = c := 
sorry

end inequality_proof_equality_condition_l93_93337


namespace area_of_rhombus_l93_93615

-- Given values for the diagonals of a rhombus.
def d1 : ℝ := 14
def d2 : ℝ := 24

-- The target statement we want to prove.
theorem area_of_rhombus : (d1 * d2) / 2 = 168 := by
  sorry

end area_of_rhombus_l93_93615


namespace estimate_diff_and_prod_l93_93066

variable {x y : ℝ}
variable (hx : x > y) (hy : y > 0)

theorem estimate_diff_and_prod :
  (1.1*x) - (y - 2) = (x - y) + 0.1 * x + 2 ∧ (1.1 * x) * (y - 2) = 1.1 * (x * y) - 2.2 * x :=
by 
  sorry -- Proof details go here

end estimate_diff_and_prod_l93_93066


namespace part1_part2_part3_l93_93552

-- Define the sequence and conditions
variable {a : ℕ → ℕ}
axiom sequence_def (n : ℕ) : a n = max (a (n + 1)) (a (n + 2)) - min (a (n + 1)) (a (n + 2))

-- Part (1)
axiom a1_def : a 1 = 1
axiom a2_def : a 2 = 2
theorem part1 : a 4 = 1 ∨ a 4 = 3 ∨ a 4 = 5 :=
  sorry

-- Part (2)
axiom has_max (M : ℕ) : ∀ n, a n ≤ M
theorem part2 : ∃ n, a n = 0 :=
  sorry

-- Part (3)
axiom positive_seq : ∀ n, a n > 0
theorem part3 : ¬∃ M : ℝ, ∀ n, a n ≤ M :=
  sorry

end part1_part2_part3_l93_93552
