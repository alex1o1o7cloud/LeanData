import Mathlib

namespace positive_difference_of_squares_l1_105

theorem positive_difference_of_squares {x y : ℕ} (hx : x > y) (hxy_sum : x + y = 70) (hxy_diff : x - y = 20) :
  x^2 - y^2 = 1400 :=
by
  sorry

end positive_difference_of_squares_l1_105


namespace original_cone_volume_l1_174

theorem original_cone_volume
  (H R h r : ℝ)
  (Vcylinder : ℝ) (Vfrustum : ℝ)
  (cylinder_volume : Vcylinder = π * r^2 * h)
  (frustum_volume : Vfrustum = (1 / 3) * π * (R^2 + R * r + r^2) * (H - h))
  (Vcylinder_value : Vcylinder = 9)
  (Vfrustum_value : Vfrustum = 63) :
  (1 / 3) * π * R^2 * H = 64 :=
by
  sorry

end original_cone_volume_l1_174


namespace expenses_neg_of_income_pos_l1_133

theorem expenses_neg_of_income_pos :
  ∀ (income expense : Int), income = 5 → expense = -income → expense = -5 :=
by
  intros income expense h_income h_expense
  rw [h_income] at h_expense
  exact h_expense

end expenses_neg_of_income_pos_l1_133


namespace determine_a_range_f_l1_141

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - (2 / (2 ^ x + 1))

theorem determine_a (a : ℝ) : (∀ x : ℝ, f a (-x) = -f a x) -> a = 1 :=
by
  sorry

theorem range_f (x : ℝ) : (∀ x : ℝ, f 1 (-x) = -f 1 x) -> -1 < f 1 x ∧ f 1 x < 1 :=
by
  sorry

end determine_a_range_f_l1_141


namespace Ian_kept_1_rose_l1_117

def initial_roses : ℕ := 20
def roses_given_to_mother : ℕ := 6
def roses_given_to_grandmother : ℕ := 9
def roses_given_to_sister : ℕ := 4
def total_roses_given : ℕ := roses_given_to_mother + roses_given_to_grandmother + roses_given_to_sister
def roses_kept (initial: ℕ) (given: ℕ) : ℕ := initial - given

theorem Ian_kept_1_rose :
  roses_kept initial_roses total_roses_given = 1 :=
by
  sorry

end Ian_kept_1_rose_l1_117


namespace money_left_after_purchases_is_correct_l1_196

noncomputable def initial_amount : ℝ := 12.50
noncomputable def cost_pencil : ℝ := 1.25
noncomputable def cost_notebook : ℝ := 3.45
noncomputable def cost_pens : ℝ := 4.80

noncomputable def total_cost : ℝ := cost_pencil + cost_notebook + cost_pens
noncomputable def money_left : ℝ := initial_amount - total_cost

theorem money_left_after_purchases_is_correct : money_left = 3.00 :=
by
  -- proof goes here, skipping with sorry for now
  sorry

end money_left_after_purchases_is_correct_l1_196


namespace find_a_extreme_value_at_2_l1_143

noncomputable def f (x : ℝ) (a : ℝ) := (2 / 3) * x^3 + a * x^2

theorem find_a_extreme_value_at_2 (a : ℝ) :
  (∀ x : ℝ, x ≠ 2 -> 0 = 2 * x^2 + 2 * a * x) ->
  (2 * 2^2 + 2 * a * 2 = 0) ->
  a = -2 :=
by {
  sorry
}

end find_a_extreme_value_at_2_l1_143


namespace laura_total_miles_per_week_l1_160

def round_trip_school : ℕ := 20
def round_trip_supermarket : ℕ := 40
def round_trip_gym : ℕ := 10
def round_trip_friends_house : ℕ := 24

def school_trips_per_week : ℕ := 5
def supermarket_trips_per_week : ℕ := 2
def gym_trips_per_week : ℕ := 3
def friends_house_trips_per_week : ℕ := 1

def total_miles_driven_per_week :=
  round_trip_school * school_trips_per_week +
  round_trip_supermarket * supermarket_trips_per_week +
  round_trip_gym * gym_trips_per_week +
  round_trip_friends_house * friends_house_trips_per_week

theorem laura_total_miles_per_week : total_miles_driven_per_week = 234 :=
by
  sorry

end laura_total_miles_per_week_l1_160


namespace hyperbola_line_intersection_l1_119

theorem hyperbola_line_intersection
  (A B m : ℝ) (hA : A ≠ 0) (hB : B ≠ 0) (hm : m ≠ 0) :
  ∃ x y : ℝ, A^2 * x^2 - B^2 * y^2 = 1 ∧ Ax - By = m ∧ Bx + Ay ≠ 0 :=
by
  sorry

end hyperbola_line_intersection_l1_119


namespace rectangle_perimeter_of_triangle_area_l1_161

theorem rectangle_perimeter_of_triangle_area
  (h_right : ∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ a = 9 ∧ b = 12 ∧ c = 15)
  (rect_length : ℕ) 
  (rect_area_eq_triangle_area : ∃ (area : ℕ), area = 1/2 * 9 * 12 ∧ area = rect_length * rect_width ) 
  : ∃ (perimeter : ℕ), perimeter = 2 * (6 + rect_width) ∧ perimeter = 30 :=
sorry

end rectangle_perimeter_of_triangle_area_l1_161


namespace age_difference_l1_148

theorem age_difference (x : ℕ) 
  (h_ratio : 4 * x + 3 * x + 7 * x = 126)
  (h_halima : 4 * x = 36)
  (h_beckham : 3 * x = 27) :
  4 * x - 3 * x = 9 :=
by sorry

end age_difference_l1_148


namespace dacid_weighted_average_l1_149

theorem dacid_weighted_average :
  let english := 96
  let mathematics := 95
  let physics := 82
  let chemistry := 87
  let biology := 92
  let weight_english := 0.20
  let weight_mathematics := 0.25
  let weight_physics := 0.15
  let weight_chemistry := 0.25
  let weight_biology := 0.15
  (english * weight_english) + (mathematics * weight_mathematics) +
  (physics * weight_physics) + (chemistry * weight_chemistry) +
  (biology * weight_biology) = 90.8 :=
by
  sorry

end dacid_weighted_average_l1_149


namespace div_of_floats_l1_124

theorem div_of_floats : (0.2 : ℝ) / (0.005 : ℝ) = 40 := 
by
  sorry

end div_of_floats_l1_124


namespace five_coins_all_heads_or_tails_l1_146

theorem five_coins_all_heads_or_tails : 
  (1 / 2) ^ 5 + (1 / 2) ^ 5 = 1 / 16 := 
by 
  sorry

end five_coins_all_heads_or_tails_l1_146


namespace weight_first_watermelon_l1_123

-- We define the total weight and the weight of the second watermelon
def total_weight := 14.02
def second_watermelon := 4.11

-- We need to prove that the weight of the first watermelon is 9.91 pounds
theorem weight_first_watermelon : total_weight - second_watermelon = 9.91 := by
  -- Insert mathematical steps here (omitted in this case)
  sorry

end weight_first_watermelon_l1_123


namespace sum_of_six_terms_arithmetic_sequence_l1_132

theorem sum_of_six_terms_arithmetic_sequence (S : ℕ → ℕ)
    (h1 : S 2 = 2)
    (h2 : S 4 = 10) :
    S 6 = 42 :=
by
  sorry

end sum_of_six_terms_arithmetic_sequence_l1_132


namespace throwers_count_l1_182

variable (totalPlayers : ℕ) (rightHandedPlayers : ℕ) (nonThrowerLeftHandedFraction nonThrowerRightHandedFraction : ℚ)

theorem throwers_count
  (h1 : totalPlayers = 70)
  (h2 : rightHandedPlayers = 64)
  (h3 : nonThrowerLeftHandedFraction = 1 / 3)
  (h4 : nonThrowerRightHandedFraction = 2 / 3)
  (h5 : nonThrowerLeftHandedFraction + nonThrowerRightHandedFraction = 1) : 
  ∃ T : ℕ, T = 52 := by
  sorry

end throwers_count_l1_182


namespace Democrats_in_House_l1_187

-- Let D be the number of Democrats.
-- Let R be the number of Republicans.
-- Given conditions.

def Democrats (D R : ℕ) : Prop := 
  D + R = 434 ∧ R = D + 30

theorem Democrats_in_House : ∃ D, ∃ R, Democrats D R ∧ D = 202 :=
by
  -- skip the proof
  sorry

end Democrats_in_House_l1_187


namespace perimeter_ratio_l1_108

def original_paper : ℕ × ℕ := (12, 8)
def folded_paper : ℕ × ℕ := (original_paper.1, original_paper.2 / 2)
def small_rectangle : ℕ × ℕ := (folded_paper.1 / 2, folded_paper.2)

def perimeter (rect : ℕ × ℕ) : ℕ :=
  2 * (rect.1 + rect.2)

theorem perimeter_ratio :
  perimeter small_rectangle = 1 / 2 * perimeter original_paper :=
by
  sorry

end perimeter_ratio_l1_108


namespace wire_length_before_cut_l1_139

-- Defining the conditions
def wire_cut (L S : ℕ) : Prop :=
  S = 20 ∧ S = (2 / 5 : ℚ) * L

-- The statement we need to prove
theorem wire_length_before_cut (L S : ℕ) (h : wire_cut L S) : (L + S) = 70 := 
by 
  sorry

end wire_length_before_cut_l1_139


namespace martha_makes_40_cookies_martha_needs_7_5_cups_l1_153

theorem martha_makes_40_cookies :
  (24 / 3) * 5 = 40 :=
by
  sorry

theorem martha_needs_7_5_cups :
  60 / (24 / 3) = 7.5 :=
by
  sorry

end martha_makes_40_cookies_martha_needs_7_5_cups_l1_153


namespace smallest_three_digit_multiple_of_17_l1_115

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l1_115


namespace triangles_pentagons_difference_l1_162

theorem triangles_pentagons_difference :
  ∃ x y : ℕ, 
  (x + y = 50) ∧ (3 * x + 5 * y = 170) ∧ (x - y = 30) :=
sorry

end triangles_pentagons_difference_l1_162


namespace percent_gain_on_transaction_l1_114

theorem percent_gain_on_transaction :
  ∀ (x : ℝ), (850 : ℝ) * x + (50 : ℝ) * (1.10 * ((850 : ℝ) * x / 800)) = 850 * x * (1 + 0.06875) := 
by
  intro x
  sorry

end percent_gain_on_transaction_l1_114


namespace mangoes_count_l1_112

noncomputable def total_fruits : ℕ := 58
noncomputable def pears : ℕ := 10
noncomputable def pawpaws : ℕ := 12
noncomputable def lemons : ℕ := 9
noncomputable def kiwi : ℕ := 9

theorem mangoes_count (mangoes : ℕ) : 
  (pears + pawpaws + lemons + kiwi + mangoes = total_fruits) → 
  mangoes = 18 :=
by
  sorry

end mangoes_count_l1_112


namespace years_to_rise_to_chief_l1_118

-- Definitions based on the conditions
def ageWhenRetired : ℕ := 46
def ageWhenJoined : ℕ := 18
def additionalYearsAsMasterChief : ℕ := 10
def multiplierForChiefToMasterChief : ℚ := 1.25

-- Total years spent in the military
def totalYearsInMilitary : ℕ := ageWhenRetired - ageWhenJoined

-- Given conditions and correct answer
theorem years_to_rise_to_chief (x : ℚ) (h : totalYearsInMilitary = x + multiplierForChiefToMasterChief * x + additionalYearsAsMasterChief) :
  x = 8 := by
  sorry

end years_to_rise_to_chief_l1_118


namespace root_integer_l1_130

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

def is_root (x_0 : ℝ) : Prop := f x_0 = 0

theorem root_integer (x_0 : ℝ) (h : is_root x_0) : Int.floor x_0 = 2 := by
  sorry

end root_integer_l1_130


namespace adam_spent_on_ferris_wheel_l1_156

theorem adam_spent_on_ferris_wheel (t_initial t_left t_price : ℕ) (h1 : t_initial = 13)
  (h2 : t_left = 4) (h3 : t_price = 9) : t_initial - t_left = 9 ∧ (t_initial - t_left) * t_price = 81 := 
by
  sorry

end adam_spent_on_ferris_wheel_l1_156


namespace area_expression_l1_142

noncomputable def overlapping_area (m : ℝ) (h1 : 0 < m) (h2 : m < 4 * Real.sqrt 2) : ℝ :=
if h : m ≤ 2 * Real.sqrt 2 then
  6 - Real.sqrt 2 * m
else
  (1 / 4) * m^2 - 2 * Real.sqrt 2 * m + 8

theorem area_expression (m : ℝ) (h1 : 0 < m) (h2 : m < 4 * Real.sqrt 2) :
  let y := overlapping_area m h1 h2
  (if h : m ≤ 2 * Real.sqrt 2 then y = 6 - Real.sqrt 2 * m
   else y = (1 / 4) * m^2 - 2 * Real.sqrt 2 * m + 8) := 
sorry

end area_expression_l1_142


namespace number_of_students_in_first_class_l1_166

theorem number_of_students_in_first_class 
  (x : ℕ) -- number of students in the first class
  (avg_first_class : ℝ := 50) 
  (num_second_class : ℕ := 50)
  (avg_second_class : ℝ := 60)
  (avg_all_students : ℝ := 56.25)
  (total_avg_eqn : (avg_first_class * x + avg_second_class * num_second_class) / (x + num_second_class) = avg_all_students) : 
  x = 30 :=
by sorry

end number_of_students_in_first_class_l1_166


namespace find_y_l1_175

theorem find_y (x y : ℝ) (h : x = 180) (h1 : 0.25 * x = 0.10 * y - 5) : y = 500 :=
by sorry

end find_y_l1_175


namespace pascal_triangle_ratios_l1_157
open Nat

theorem pascal_triangle_ratios :
  ∃ n r : ℕ, 
  (choose n r) * 4 = (choose n (r + 1)) * 3 ∧ 
  (choose n (r + 1)) * 3 = (choose n (r + 2)) * 4 ∧ 
  n = 34 :=
by
  sorry

end pascal_triangle_ratios_l1_157


namespace conditional_probability_of_A_given_target_hit_l1_111

theorem conditional_probability_of_A_given_target_hit :
  (3 / 5 : ℚ) * ( ( 4 / 5 + 1 / 5) ) = (15 / 23 : ℚ) :=
  sorry

end conditional_probability_of_A_given_target_hit_l1_111


namespace woman_works_finish_days_l1_145

theorem woman_works_finish_days (M W : ℝ) 
  (hm_work : ∀ n : ℝ, n * M = 1 / 100)
  (hw_work : ∀ men women : ℝ, (10 * M + 15 * women) * 6 = 1) :
  W = 1 / 225 :=
by
  have man_work := hm_work 1
  have woman_work := hw_work 10 W
  sorry

end woman_works_finish_days_l1_145


namespace sum_zero_quotient_l1_190

   theorem sum_zero_quotient (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h_sum_zero : x + y + z = 0) :
     (xy + yz + zx) / (x^2 + y^2 + z^2) = -1 / 2 :=
   by
     sorry
   
end sum_zero_quotient_l1_190


namespace BurjKhalifaHeight_l1_150

def SearsTowerHeight : ℕ := 527
def AdditionalHeight : ℕ := 303

theorem BurjKhalifaHeight : (SearsTowerHeight + AdditionalHeight) = 830 :=
by
  sorry

end BurjKhalifaHeight_l1_150


namespace total_material_weight_l1_169

def gravel_weight : ℝ := 5.91
def sand_weight : ℝ := 8.11

theorem total_material_weight : gravel_weight + sand_weight = 14.02 := by
  sorry

end total_material_weight_l1_169


namespace fraction_paint_left_after_third_day_l1_109

noncomputable def original_paint : ℝ := 2
noncomputable def paint_after_first_day : ℝ := original_paint - (1 / 2 * original_paint)
noncomputable def paint_after_second_day : ℝ := paint_after_first_day - (1 / 4 * paint_after_first_day)
noncomputable def paint_after_third_day : ℝ := paint_after_second_day - (1 / 2 * paint_after_second_day)

theorem fraction_paint_left_after_third_day :
  paint_after_third_day / original_ppaint = 3 / 8 :=
sorry

end fraction_paint_left_after_third_day_l1_109


namespace arithmetic_sequence_fifth_term_l1_120

noncomputable def fifth_term (x y : ℝ) : ℝ :=
  let a1 := x^2 + y^2
  let a2 := x^2 - y^2
  let a3 := x^2 * y^2
  let a4 := x^2 / y^2
  let d := -2 * y^2
  a4 + d

theorem arithmetic_sequence_fifth_term (x y : ℝ) (hy : y ≠ 0) (hx2 : x ^ 2 = 3 * y ^ 2 / (y ^ 2 - 1)) :
  fifth_term x y = 3 / (y ^ 2 - 1) - 2 * y ^ 2 :=
by
  sorry

end arithmetic_sequence_fifth_term_l1_120


namespace measure_of_angle_D_l1_192

def angle_A := 95 -- Defined in step b)
def angle_B := angle_A
def angle_C := angle_A
def angle_D := angle_A + 50
def angle_E := angle_D
def angle_F := angle_D

theorem measure_of_angle_D (x : ℕ) (y : ℕ) :
  (angle_A = x) ∧ (angle_D = y) ∧ (y = x + 50) ∧ (3 * x + 3 * y = 720) → y = 145 :=
by
  intros
  sorry

end measure_of_angle_D_l1_192


namespace John_has_22_quarters_l1_158

variable (q d n : ℕ)

-- Conditions
axiom cond1 : d = q + 3
axiom cond2 : n = q - 6
axiom cond3 : q + d + n = 63

theorem John_has_22_quarters : q = 22 := by
  sorry

end John_has_22_quarters_l1_158


namespace remaining_paint_fraction_l1_168

def initial_paint : ℚ := 1

def paint_day_1 : ℚ := initial_paint - (1/2) * initial_paint
def paint_day_2 : ℚ := paint_day_1 - (1/4) * paint_day_1
def paint_day_3 : ℚ := paint_day_2 - (1/3) * paint_day_2

theorem remaining_paint_fraction : paint_day_3 = 1/4 :=
by
  sorry

end remaining_paint_fraction_l1_168


namespace line_passes_vertex_parabola_l1_136

theorem line_passes_vertex_parabola :
  ∃ (b₁ b₂ : ℚ), (b₁ ≠ b₂) ∧ (∀ b, (b = b₁ ∨ b = b₂) → 
    (∃ x y, y = x + b ∧ y = x^2 + 4 * b^2 ∧ x = 0 ∧ y = 4 * b^2)) :=
by 
  sorry

end line_passes_vertex_parabola_l1_136


namespace distinct_placements_of_two_pieces_l1_176

-- Definitions of the conditions
def grid_size : ℕ := 3
def cell_count : ℕ := grid_size * grid_size
def pieces_count : ℕ := 2

-- The theorem statement
theorem distinct_placements_of_two_pieces : 
  (number_of_distinct_placements : ℕ) = 10 := by
  -- Proof goes here with calculations and accounting for symmetry
  sorry

end distinct_placements_of_two_pieces_l1_176


namespace inversely_proportional_ratios_l1_164

theorem inversely_proportional_ratios (x y x₁ x₂ y₁ y₂ : ℝ) (hx_inv : ∀ x y, x * y = 1)
  (hx_ratio : x₁ / x₂ = 3 / 5) :
  y₁ / y₂ = 5 / 3 :=
sorry

end inversely_proportional_ratios_l1_164


namespace adult_ticket_cost_l1_185

variable (A : ℝ)

theorem adult_ticket_cost :
  (20 * 6) + (12 * A) = 216 → A = 8 :=
by
  intro h
  sorry

end adult_ticket_cost_l1_185


namespace line_not_in_fourth_quadrant_l1_129

-- Let the line be defined as y = 3x + 2
def line_eq (x : ℝ) : ℝ := 3 * x + 2

-- The Fourth quadrant is defined by x > 0 and y < 0
def in_fourth_quadrant (x : ℝ) (y : ℝ) : Prop := x > 0 ∧ y < 0

-- Prove that the line does not intersect the Fourth quadrant
theorem line_not_in_fourth_quadrant : ¬ (∃ x : ℝ, in_fourth_quadrant x (line_eq x)) :=
by
  -- Proof goes here (abstracted)
  sorry

end line_not_in_fourth_quadrant_l1_129


namespace sample_std_dev_range_same_l1_128

noncomputable def sample_std_dev (data : List ℝ) : ℝ := sorry
noncomputable def sample_range (data : List ℝ) : ℝ := sorry

theorem sample_std_dev_range_same (n : ℕ) (c : ℝ) (Hc : c ≠ 0) (x : Fin n → ℝ) :
  sample_std_dev (List.ofFn (λ i => x i)) = sample_std_dev (List.ofFn (λ i => x i + c)) ∧
  sample_range (List.ofFn (λ i => x i)) = sample_range (List.ofFn (λ i => x i + c)) :=
by
  sorry

end sample_std_dev_range_same_l1_128


namespace range_of_x_when_y_lt_0_l1_165

variable (a b c n m : ℝ)

-- The definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions given in the problem
axiom value_at_neg1 : quadratic_function a b c (-1) = 4
axiom value_at_0 : quadratic_function a b c 0 = 0
axiom value_at_1 : quadratic_function a b c 1 = n
axiom value_at_2 : quadratic_function a b c 2 = m
axiom value_at_3 : quadratic_function a b c 3 = 4

-- Proof statement
theorem range_of_x_when_y_lt_0 : ∀ (x : ℝ), quadratic_function a b c x < 0 ↔ 0 < x ∧ x < 2 :=
sorry

end range_of_x_when_y_lt_0_l1_165


namespace ratio_xy_half_l1_189

noncomputable def common_ratio_k (x y z : ℝ) (h : (x + 4) / 2 = (y + 9) / (z - 3) ∧ (y + 9) / (z - 3) = (x + 5) / (z - 5)) : ℝ := sorry

theorem ratio_xy_half (x y z k : ℝ) (h : (x + 4) / 2 = (y + 9) / (z - 3) ∧ (y + 9) / (z - 3) = (x + 5) / (z - 5)) :
  ∃ k, (x + 4) = 2 * k ∧ (y + 9) = k * (z - 3) ∧ (x + 5) = k * (z - 5) → (x / y) = 1 / 2 :=
sorry

end ratio_xy_half_l1_189


namespace solve_equation_l1_154

theorem solve_equation :
  ∀ x : ℝ, (3 * x^2 / (x - 2) - (3 * x + 4) / 2 + (5 - 9 * x) / (x - 2) + 2 = 0) →
    (x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6) :=
by
  intro x h
  -- the proof would go here
  sorry

end solve_equation_l1_154


namespace sixth_power_of_sqrt_l1_195

variable (x : ℝ)
axiom h1 : x = Real.sqrt (2 + Real.sqrt 2)

theorem sixth_power_of_sqrt : x^6 = 16 + 10 * Real.sqrt 2 :=
by {
    sorry
}

end sixth_power_of_sqrt_l1_195


namespace largest_A_l1_180

theorem largest_A (A B C : ℕ) (h1 : A = 7 * B + C) (h2 : B = C) : A ≤ 48 :=
  sorry

end largest_A_l1_180


namespace brad_money_l1_193

noncomputable def money_problem : Prop :=
  ∃ (B J D : ℝ), 
    J = 2 * B ∧
    J = (3/4) * D ∧
    B + J + D = 68 ∧
    B = 12

theorem brad_money : money_problem :=
by {
  -- Insert proof steps here if necessary
  sorry
}

end brad_money_l1_193


namespace necessary_and_sufficient_condition_l1_100

def line1 (x y : ℝ) : Prop := x - y - 1 = 0
def line2 (x y a : ℝ) : Prop := x + a * y - 2 = 0

def p (a : ℝ) : Prop := ∀ x y : ℝ, line1 x y → line2 x y a
def q (a : ℝ) : Prop := a = -1

theorem necessary_and_sufficient_condition (a : ℝ) : (p a) ↔ (q a) :=
by
  sorry

end necessary_and_sufficient_condition_l1_100


namespace ral_age_is_26_l1_110

def ral_current_age (suri_age : ℕ) (ral_age : ℕ) : Prop :=
  ral_age = 2 * suri_age

theorem ral_age_is_26 (suri_current_age : ℕ) (ral_current_age : ℕ) (h1 : suri_current_age + 3 = 16) (h2 : ral_age = 2 * suri_age) : ral_current_age = 26 := 
by
  sorry

end ral_age_is_26_l1_110


namespace celia_receives_correct_amount_of_aranha_l1_170

def borboleta_to_tubarao (b : Int) : Int := 3 * b
def tubarao_to_periquito (t : Int) : Int := 2 * t
def periquito_to_aranha (p : Int) : Int := 3 * p
def macaco_to_aranha (m : Int) : Int := 4 * m
def cobra_to_periquito (c : Int) : Int := 3 * c

def celia_stickers_to_aranha (borboleta tubarao cobra periquito macaco : Int) : Int :=
  let borboleta_to_aranha := periquito_to_aranha (tubarao_to_periquito (borboleta_to_tubarao borboleta))
  let tubarao_to_aranha := periquito_to_aranha (tubarao_to_periquito tubarao)
  let cobra_to_aranha := periquito_to_aranha (cobra_to_periquito cobra)
  let periquito_to_aranha := periquito_to_aranha periquito
  let macaco_to_aranha := macaco_to_aranha macaco
  borboleta_to_aranha + tubarao_to_aranha + cobra_to_aranha + periquito_to_aranha + macaco_to_aranha

theorem celia_receives_correct_amount_of_aranha : 
  celia_stickers_to_aranha 4 5 3 6 6 = 171 := 
by
  simp only [celia_stickers_to_aranha, borboleta_to_tubarao, tubarao_to_periquito, periquito_to_aranha, cobra_to_periquito, macaco_to_aranha]
  -- Here we need to perform the arithmetic steps to verify the sum
  sorry -- This is the placeholder for the actual proof

end celia_receives_correct_amount_of_aranha_l1_170


namespace perfect_square_iff_divisibility_l1_167

theorem perfect_square_iff_divisibility (A : ℕ) :
  (∃ d : ℕ, A = d^2) ↔ ∀ n : ℕ, n > 0 → ∃ j : ℕ, 1 ≤ j ∧ j ≤ n ∧ n ∣ (A + j)^2 - A :=
sorry

end perfect_square_iff_divisibility_l1_167


namespace jet_flight_distance_l1_137

-- Setting up the hypotheses and the statement
theorem jet_flight_distance (v d : ℕ) (h1 : d = 4 * (v + 50)) (h2 : d = 5 * (v - 50)) : d = 2000 :=
sorry

end jet_flight_distance_l1_137


namespace simplify_and_evaluate_l1_173

theorem simplify_and_evaluate (a : ℚ) (h : a = 3) :
  (1 - (a - 2) / (a^2 - 4)) / ((a^2 + a) / (a^2 + 4*a + 4)) = 5 / 3 :=
by
  sorry

end simplify_and_evaluate_l1_173


namespace division_of_decimals_l1_155

theorem division_of_decimals : 0.18 / 0.003 = 60 :=
by
  sorry

end division_of_decimals_l1_155


namespace solve_inequality_part1_solve_inequality_part2_l1_122

-- Define the first part of the problem
theorem solve_inequality_part1 (a : ℝ) (x : ℝ) :
  (x^2 - a * x - 2 * a^2 < 0) ↔ 
    (a = 0 ∧ false) ∨ 
    (a > 0 ∧ -a < x ∧ x < 2 * a) ∨ 
    (a < 0 ∧ 2 * a < x ∧ x < -a) := 
sorry

-- Define the second part of the problem
theorem solve_inequality_part2 (a b : ℝ) (x : ℝ) 
  (h : { x | x^2 - a * x - b < 0 } = { x | -1 < x ∧ x < 2 }) :
  { x | a * x^2 + x - b > 0 } = { x | x < -2 } ∪ { x | 1 < x } :=
sorry

end solve_inequality_part1_solve_inequality_part2_l1_122


namespace number_of_sides_of_polygon_l1_121

theorem number_of_sides_of_polygon (n : ℕ) :
  ((n - 2) * 180 = 3 * 360) → n = 8 := 
by
  sorry

end number_of_sides_of_polygon_l1_121


namespace inequality_solution_l1_147

theorem inequality_solution (x : ℝ) 
  (hx1 : x ≠ 1) 
  (hx2 : x ≠ 2) 
  (hx3 : x ≠ 3) 
  (hx4 : x ≠ 4) :
  (1 / (x - 1) - 3 / (x - 2) + 5 / (x - 3) - 1 / (x - 4) < 1 / 24) ↔ (x ∈ Set.Ioo (-7 : ℝ) 1 ∪ Set.Ioo 3 4) := 
sorry

end inequality_solution_l1_147


namespace text_messages_December_l1_103

-- Definitions of the number of text messages sent each month
def text_messages_November := 1
def text_messages_January := 4
def text_messages_February := 8
def doubling_pattern (a b : ℕ) : Prop := b = 2 * a

-- Prove that Jared sent 2 text messages in December
theorem text_messages_December : ∃ x : ℕ, 
  doubling_pattern text_messages_November x ∧ 
  doubling_pattern x text_messages_January ∧ 
  doubling_pattern text_messages_January text_messages_February ∧ 
  x = 2 :=
by
  sorry

end text_messages_December_l1_103


namespace toothpick_grid_l1_178

theorem toothpick_grid (l w : ℕ) (h_l : l = 45) (h_w : w = 25) :
  let effective_vertical_lines := l + 1 - (l + 1) / 5
  let effective_horizontal_lines := w + 1 - (w + 1) / 5
  let vertical_toothpicks := effective_vertical_lines * w
  let horizontal_toothpicks := effective_horizontal_lines * l
  let total_toothpicks := vertical_toothpicks + horizontal_toothpicks
  total_toothpicks = 1722 :=
by {
  sorry
}

end toothpick_grid_l1_178


namespace average_percentage_of_first_20_percent_l1_138

theorem average_percentage_of_first_20_percent (X : ℝ) 
  (h1 : 0.20 * X + 0.50 * 60 + 0.30 * 40 = 58) : 
  X = 80 :=
sorry

end average_percentage_of_first_20_percent_l1_138


namespace total_team_cost_l1_134

-- Define the costs of individual items and the number of players
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.2
def socks_cost : ℝ := 6.8
def num_players : ℕ := 16

-- Define the total cost for equipment for one player
def player_cost : ℝ :=
  jersey_cost + shorts_cost + socks_cost

-- The main theorem stating the total cost for all players
theorem total_team_cost : num_players * player_cost = 752 := by
  sorry

end total_team_cost_l1_134


namespace width_of_lawn_is_30_m_l1_106

-- Define the conditions
def lawn_length : ℕ := 70
def lawn_width : ℕ := 30
def road_width : ℕ := 5
def gravel_rate_per_sqm : ℕ := 4
def gravel_cost : ℕ := 1900

-- Mathematically equivalent proof problem statement
theorem width_of_lawn_is_30_m 
  (H1 : lawn_length = 70)
  (H2 : road_width = 5)
  (H3 : gravel_rate_per_sqm = 4)
  (H4 : gravel_cost = 1900)
  (H5 : 2*road_width*5 + (lawn_length - road_width) * 5 * gravel_rate_per_sqm = gravel_cost) :
  lawn_width = 30 := 
sorry

end width_of_lawn_is_30_m_l1_106


namespace smaller_consecutive_number_divisibility_l1_184

theorem smaller_consecutive_number_divisibility :
  ∃ (m : ℕ), (m < m + 1) ∧ (1 ≤ m ∧ m ≤ 200) ∧ (1 ≤ m + 1 ∧ m + 1 ≤ 200) ∧
              (∀ n, (1 ≤ n ∧ n ≤ 200 ∧ n ≠ m ∧ n ≠ m + 1) → ∃ k, chosen_num = k * n) ∧
              (128 = m) :=
sorry

end smaller_consecutive_number_divisibility_l1_184


namespace midpoint_in_polar_coordinates_l1_116

theorem midpoint_in_polar_coordinates :
  let A := (9, Real.pi / 3)
  let B := (9, 2 * Real.pi / 3)
  let mid := (Real.sqrt (3) * 9 / 2, Real.pi / 2)
  (mid = (Real.sqrt (3) * 9 / 2, Real.pi / 2)) :=
by 
  sorry

end midpoint_in_polar_coordinates_l1_116


namespace value_of_expression_l1_113

theorem value_of_expression (a b m n x : ℝ) 
    (hab : a * b = 1) 
    (hmn : m + n = 0) 
    (hxsq : x^2 = 1) : 
    2022 * (m + n) + 2018 * x^2 - 2019 * (a * b) = -1 := 
by 
    sorry

end value_of_expression_l1_113


namespace isabella_initial_hair_length_l1_188

theorem isabella_initial_hair_length
  (final_length : ℕ)
  (growth_over_year : ℕ)
  (initial_length : ℕ)
  (h_final : final_length = 24)
  (h_growth : growth_over_year = 6)
  (h_initial : initial_length = 18) :
  initial_length + growth_over_year = final_length := 
by 
  sorry

end isabella_initial_hair_length_l1_188


namespace length_of_X_l1_140

theorem length_of_X
  {X : ℝ}
  (h1 : 2 + 2 + X = 4 + X)
  (h2 : 3 + 4 + 1 = 8)
  (h3 : ∃ y : ℝ, y * (4 + X) = 29) : 
  X = 4 := sorry

end length_of_X_l1_140


namespace negation_propositional_logic_l1_126

theorem negation_propositional_logic :
  ¬ (∀ x : ℝ, x^2 + x + 1 < 0) ↔ ∃ x : ℝ, x^2 + x + 1 ≥ 0 :=
by sorry

end negation_propositional_logic_l1_126


namespace convert_degrees_to_radians_l1_199

theorem convert_degrees_to_radians (deg : ℝ) (deg_eq : deg = -300) : 
  deg * (π / 180) = - (5 * π) / 3 := 
by
  rw [deg_eq]
  sorry

end convert_degrees_to_radians_l1_199


namespace solve_x4_minus_16_eq_0_l1_198

open Complex  -- Open the complex number notation

theorem solve_x4_minus_16_eq_0 :
  {x : ℂ | x^4 = 16} = {2, -2, 2 * Complex.I, -2 * Complex.I} :=
by sorry

end solve_x4_minus_16_eq_0_l1_198


namespace exists_polyhedron_with_given_vertices_and_edges_l1_186

theorem exists_polyhedron_with_given_vertices_and_edges :
  ∃ (V : Finset (String)) (E : Finset (Finset (String))),
    V = { "A", "B", "C", "D", "E", "F", "G", "H" } ∧
    E = { { "A", "B" }, { "A", "C" }, { "A", "H" }, { "B", "C" },
          { "B", "D" }, { "C", "D" }, { "D", "E" }, { "E", "F" },
          { "E", "G" }, { "F", "G" }, { "F", "H" }, { "G", "H" } } ∧
    (V.card : ℤ) - (E.card : ℤ) + 6 = 2 :=
by
  sorry

end exists_polyhedron_with_given_vertices_and_edges_l1_186


namespace soccer_ball_diameter_l1_102

theorem soccer_ball_diameter 
  (h : ℝ)
  (s : ℝ)
  (d : ℝ)
  (h_eq : h = 1.25)
  (s_eq : s = 1)
  (d_eq : d = 0.23) : 2 * (d * h / (s - h)) = 0.46 :=
by
  sorry

end soccer_ball_diameter_l1_102


namespace reduced_bucket_fraction_l1_127

theorem reduced_bucket_fraction (C : ℝ) (F : ℝ) (h : 25 * F * C = 10 * C) : F = 2 / 5 :=
by sorry

end reduced_bucket_fraction_l1_127


namespace determinant_transformation_l1_135

theorem determinant_transformation (p q r s : ℝ) (h : p * s - q * r = -3) :
    p * (5 * r + 2 * s) - r * (5 * p + 2 * q) = -6 := by
  sorry

end determinant_transformation_l1_135


namespace years_taught_third_grade_l1_194

def total_years : ℕ := 26
def years_taught_second_grade : ℕ := 8

theorem years_taught_third_grade :
  total_years - years_taught_second_grade = 18 :=
by {
  -- Subtract the years taught second grade from the total years
  -- Exact the result
  sorry
}

end years_taught_third_grade_l1_194


namespace layers_removed_l1_183

theorem layers_removed (n : ℕ) (original_volume remaining_volume side_length : ℕ) :
  original_volume = side_length^3 →
  remaining_volume = (side_length - 2 * n)^3 →
  original_volume = 1000 →
  remaining_volume = 512 →
  side_length = 10 →
  n = 1 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end layers_removed_l1_183


namespace problem_abcd_eq_14400_l1_125

theorem problem_abcd_eq_14400
 (a b c d : ℝ)
 (h1 : a^2 + b^2 + c^2 + d^2 = 762)
 (h2 : a * b + c * d = 260)
 (h3 : a * c + b * d = 365)
 (h4 : a * d + b * c = 244) :
 a * b * c * d = 14400 := 
sorry

end problem_abcd_eq_14400_l1_125


namespace train_speed_is_45_kmph_l1_131

noncomputable def speed_of_train_kmph (train_length bridge_length total_time : ℕ) : ℕ :=
  let total_distance := train_length + bridge_length
  let speed_mps := total_distance / total_time
  let speed_kmph := speed_mps * 36 / 10
  speed_kmph

theorem train_speed_is_45_kmph :
  speed_of_train_kmph 150 225 30 = 45 :=
  sorry

end train_speed_is_45_kmph_l1_131


namespace h_inverse_correct_l1_171

noncomputable def f (x : ℝ) := 4 * x + 7
noncomputable def g (x : ℝ) := 3 * x - 2
noncomputable def h (x : ℝ) := f (g x)
noncomputable def h_inv (y : ℝ) := (y + 1) / 12

theorem h_inverse_correct : ∀ x : ℝ, h_inv (h x) = x :=
by
  intro x
  sorry

end h_inverse_correct_l1_171


namespace floor_div_eq_floor_div_l1_151

theorem floor_div_eq_floor_div
  (a : ℝ) (n : ℤ) (ha_pos : 0 < a) :
  (⌊⌊a⌋ / n⌋ : ℤ) = ⌊a / n⌋ := 
sorry

end floor_div_eq_floor_div_l1_151


namespace infinity_gcd_binom_l1_163

theorem infinity_gcd_binom {k l : ℕ} : ∃ᶠ m in at_top, m ≥ k ∧ Nat.gcd (Nat.choose m k) l = 1 :=
sorry

end infinity_gcd_binom_l1_163


namespace stratified_sample_correct_l1_101

variable (popA popB popC : ℕ) (totalSample : ℕ)

def stratified_sample (popA popB popC totalSample : ℕ) : ℕ × ℕ × ℕ :=
  let totalChickens := popA + popB + popC
  let sampledA := (popA * totalSample) / totalChickens
  let sampledB := (popB * totalSample) / totalChickens
  let sampledC := (popC * totalSample) / totalChickens
  (sampledA, sampledB, sampledC)

theorem stratified_sample_correct
  (hA : popA = 12000) (hB : popB = 8000) (hC : popC = 4000) (hSample : totalSample = 120) :
  stratified_sample popA popB popC totalSample = (60, 40, 20) :=
by
  sorry

end stratified_sample_correct_l1_101


namespace sphere_surface_area_l1_144

theorem sphere_surface_area
  (a : ℝ)
  (expansion : (1 - 2 * 1 : ℝ)^6 = a)
  (a_value : a = 1) :
  4 * Real.pi * ((Real.sqrt (2^2 + 3^2 + a^2) / 2)^2) = 14 * Real.pi :=
by
  sorry

end sphere_surface_area_l1_144


namespace heavy_equipment_pay_l1_179

theorem heavy_equipment_pay
  (total_workers : ℕ)
  (total_payroll : ℕ)
  (laborers : ℕ)
  (laborer_pay : ℕ)
  (heavy_operator_pay : ℕ)
  (h1 : total_workers = 35)
  (h2 : total_payroll = 3950)
  (h3 : laborers = 19)
  (h4 : laborer_pay = 90)
  (h5 : (total_workers - laborers) * heavy_operator_pay + laborers * laborer_pay = total_payroll) :
  heavy_operator_pay = 140 :=
by
  sorry

end heavy_equipment_pay_l1_179


namespace proof_problem_l1_177

theorem proof_problem (p q r : ℝ) 
  (h1 : p + q = 20)
  (h2 : p * q = 144) 
  (h3 : q + r = 52) 
  (h4 : 4 * (r + p) = r * p) : 
  r - p = 32 := 
sorry

end proof_problem_l1_177


namespace area_inside_S_outside_R_l1_181

theorem area_inside_S_outside_R (area_R area_S : ℝ) (h1: area_R = 1 + 3 * Real.sqrt 3) (h2: area_S = 6 * Real.sqrt 3) :
  area_S - area_R = 1 :=
by {
   sorry
}

end area_inside_S_outside_R_l1_181


namespace required_blue_balls_to_remove_l1_107

-- Define the constants according to conditions
def total_balls : ℕ := 120
def red_balls : ℕ := 54
def initial_blue_balls : ℕ := total_balls - red_balls
def desired_percentage_red : ℚ := 0.75 -- ℚ is the type for rational numbers

-- Lean theorem statement
theorem required_blue_balls_to_remove (x : ℕ) : 
    (red_balls:ℚ) / (total_balls - x : ℚ) = desired_percentage_red → x = 48 :=
by
  sorry

end required_blue_balls_to_remove_l1_107


namespace ratio_division_l1_104

theorem ratio_division
  (A B C : ℕ)
  (h : (A : ℚ) / B = 3 / 2 ∧ (B : ℚ) / C = 1 / 3) :
  (5 * A + 3 * B) / (5 * C - 2 * A) = 7 / 8 :=
by
  sorry

end ratio_division_l1_104


namespace new_area_shortening_other_side_l1_172

-- Define the dimensions of the original card
def original_length : ℕ := 5
def original_width : ℕ := 7

-- Define the shortened length and the resulting area after shortening one side by 2 inches
def shortened_length_1 := original_length - 2
def new_area_1 : ℕ := shortened_length_1 * original_width
def condition_1 : Prop := new_area_1 = 21

-- Prove that shortening the width by 2 inches results in an area of 25 square inches
theorem new_area_shortening_other_side : condition_1 → (original_length * (original_width - 2) = 25) :=
by
  intro h
  sorry

end new_area_shortening_other_side_l1_172


namespace find_a_l1_152

theorem find_a (A B : Real) (b a : Real) (hA : A = 45) (hB : B = 60) (hb : b = Real.sqrt 3) : 
  a = Real.sqrt 2 :=
sorry

end find_a_l1_152


namespace solve_eq_l1_191

theorem solve_eq (a b : ℕ) : a * a = b * (b + 7) ↔ (a, b) = (0, 0) ∨ (a, b) = (12, 9) :=
by
  sorry

end solve_eq_l1_191


namespace range_of_values_for_a_l1_159

theorem range_of_values_for_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 + a * x - 1 < 0) ↔ (-4 < a ∧ a ≤ 0) :=
by
  sorry

end range_of_values_for_a_l1_159


namespace probability_A_or_B_complement_l1_197

-- Define the sample space for rolling a die
def sample_space : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define Event A: the outcome is an even number not greater than 4
def event_A : Finset ℕ := {2, 4}

-- Define Event B: the outcome is less than 6
def event_B : Finset ℕ := {1, 2, 3, 4, 5}

-- Define the complement of Event B
def event_B_complement : Finset ℕ := {6}

-- Mutually exclusive property of events A and B_complement
axiom mutually_exclusive (A B_complement: Finset ℕ) : A ∩ B_complement = ∅

-- Define the probability function
def probability (events: Finset ℕ) : ℚ := (events.card : ℚ) / (sample_space.card : ℚ)

-- Theorem stating the probability of event (A + B_complement)
theorem probability_A_or_B_complement : probability (event_A ∪ event_B_complement) = 1 / 2 :=
by 
  sorry

end probability_A_or_B_complement_l1_197
