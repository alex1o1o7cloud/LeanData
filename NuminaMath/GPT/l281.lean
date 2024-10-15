import Mathlib

namespace NUMINAMATH_GPT_polynomial_identity_l281_28125

theorem polynomial_identity (P : ℝ → ℝ) :
  (∀ x, (x - 1) * P (x + 1) - (x + 2) * P x = 0) ↔ ∃ a : ℝ, ∀ x, P x = a * (x^3 - x) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_identity_l281_28125


namespace NUMINAMATH_GPT_willie_bananas_l281_28185

variable (W : ℝ) 

theorem willie_bananas (h1 : 35.0 - 14.0 = 21.0) (h2: W + 35.0 = 83.0) : 
  W = 48.0 :=
by
  sorry

end NUMINAMATH_GPT_willie_bananas_l281_28185


namespace NUMINAMATH_GPT_height_of_original_triangle_l281_28163

variable (a b c : ℝ)

theorem height_of_original_triangle (a b c : ℝ) : 
  ∃ h : ℝ, h = a + b + c :=
  sorry

end NUMINAMATH_GPT_height_of_original_triangle_l281_28163


namespace NUMINAMATH_GPT_angle_B_of_triangle_l281_28195

theorem angle_B_of_triangle {A B C a b c : ℝ} (h1 : b^2 = a * c) (h2 : Real.sin A + Real.sin C = 2 * Real.sin B) : 
  B = Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_angle_B_of_triangle_l281_28195


namespace NUMINAMATH_GPT_beads_per_bracelet_l281_28119

def beads_bella_has : Nat := 36
def beads_bella_needs : Nat := 12
def total_bracelets : Nat := 6

theorem beads_per_bracelet : (beads_bella_has + beads_bella_needs) / total_bracelets = 8 :=
by
  sorry

end NUMINAMATH_GPT_beads_per_bracelet_l281_28119


namespace NUMINAMATH_GPT_curve_equation_l281_28170

noncomputable def curve_passing_condition (x y : ℝ) : Prop :=
  (∃ (f : ℝ → ℝ), f 2 = 3 ∧ ∀ (t : ℝ), (f t) * t = 6 ∧ ((t ≠ 0 ∧ f t ≠ 0) → (t, f t) = (x, y)))

theorem curve_equation (x y : ℝ) (h1 : curve_passing_condition x y) : x * y = 6 :=
  sorry

end NUMINAMATH_GPT_curve_equation_l281_28170


namespace NUMINAMATH_GPT_number_of_sick_animals_l281_28137

def total_animals := 26 + 40 + 34  -- Total number of animals at Stacy's farm
def sick_fraction := 1 / 2  -- Half of all animals get sick

-- Defining sick animals for each type
def sick_chickens := 26 * sick_fraction
def sick_piglets := 40 * sick_fraction
def sick_goats := 34 * sick_fraction

-- The main theorem to prove
theorem number_of_sick_animals :
  sick_chickens + sick_piglets + sick_goats = 50 :=
by
  -- Skeleton of the proof that is to be completed later
  sorry

end NUMINAMATH_GPT_number_of_sick_animals_l281_28137


namespace NUMINAMATH_GPT_track_length_l281_28128

theorem track_length (L : ℝ)
  (h_brenda_first_meeting : ∃ (brenda_run1: ℝ), brenda_run1 = 100)
  (h_sally_first_meeting : ∃ (sally_run1: ℝ), sally_run1 = L/2 - 100)
  (h_brenda_second_meeting : ∃ (brenda_run2: ℝ), brenda_run2 = L - 100)
  (h_sally_second_meeting : ∃ (sally_run2: ℝ), sally_run2 = sally_run1 + 100)
  (h_meeting_total : brenda_run2 + sally_run2 = L) :
  L = 200 :=
by
  sorry

end NUMINAMATH_GPT_track_length_l281_28128


namespace NUMINAMATH_GPT_thirteen_consecutive_nat_power_l281_28126

def consecutive_sum_power (N : ℕ) : ℕ :=
  (N - 6) + (N - 5) + (N - 4) + (N - 3) + (N - 2) + (N - 1) +
  N + (N + 1) + (N + 2) + (N + 3) + (N + 4) + (N + 5) + (N + 6)

theorem thirteen_consecutive_nat_power (N : ℕ) (n : ℕ) :
  N = 13^2020 →
  n = 2021 →
  consecutive_sum_power N = 13^n := by
  sorry

end NUMINAMATH_GPT_thirteen_consecutive_nat_power_l281_28126


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l281_28168

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_deriv : ∀ x ≠ 0, f'' x + f x / x > 0)

noncomputable def a : ℝ := (1 / Real.exp 1) * f (1 / Real.exp 1)
noncomputable def b : ℝ := -Real.exp 1 * f (-Real.exp 1)
noncomputable def c : ℝ := f 1

theorem relationship_among_a_b_c :
  a < c ∧ c < b :=
by
  -- sorry to skip the proof steps
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l281_28168


namespace NUMINAMATH_GPT_consecutive_integers_sum_l281_28160

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end NUMINAMATH_GPT_consecutive_integers_sum_l281_28160


namespace NUMINAMATH_GPT_factorize_expression_l281_28194

theorem factorize_expression (x y : ℝ) : x^2 * y + 2 * x * y + y = y * (x + 1)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l281_28194


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l281_28136

theorem sufficient_not_necessary_condition (x : ℝ) (h1 : 0 < x) (h2 : x < 2) : (0 < x ∧ x < 2) → (x^2 - x - 2 < 0) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l281_28136


namespace NUMINAMATH_GPT_calculate_R_cubed_plus_R_squared_plus_R_l281_28120

theorem calculate_R_cubed_plus_R_squared_plus_R (R : ℕ) (hR : R > 0)
  (h1 : ∃ q : ℚ, q = (R / (2 * R + 2)) * ((R - 1) / (2 * R + 1)))
  (h2 : (R / (2 * R + 2)) * ((R + 2) / (2 * R + 1)) + ((R + 2) / (2 * R + 2)) * (R / (2 * R + 1)) = 3 * q) :
  R^3 + R^2 + R = 399 :=
by
  sorry

end NUMINAMATH_GPT_calculate_R_cubed_plus_R_squared_plus_R_l281_28120


namespace NUMINAMATH_GPT_bert_money_problem_l281_28186

-- Define the conditions as hypotheses
theorem bert_money_problem
  (n : ℝ)
  (h1 : n > 0)  -- Since he can't have negative or zero dollars initially
  (h2 : (1/2) * ((3/4) * n - 9) = 15) :
  n = 52 :=
sorry

end NUMINAMATH_GPT_bert_money_problem_l281_28186


namespace NUMINAMATH_GPT_value_of_expression_l281_28164

theorem value_of_expression (m : ℝ) (h : 2 * m ^ 2 - 3 * m - 1 = 0) : 4 * m ^ 2 - 6 * m = 2 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l281_28164


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l281_28196

open Real

theorem quadratic_inequality_solution :
    ∀ x : ℝ, -8 * x^2 + 6 * x - 1 < 0 ↔ 0.25 < x ∧ x < 0.5 :=
by sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l281_28196


namespace NUMINAMATH_GPT_bottles_left_on_shelf_l281_28150

theorem bottles_left_on_shelf (initial_bottles : ℕ) (jason_buys : ℕ) (harry_buys : ℕ) (total_buys : ℕ) (remaining_bottles : ℕ)
  (h1 : initial_bottles = 35)
  (h2 : jason_buys = 5)
  (h3 : harry_buys = 6)
  (h4 : total_buys = jason_buys + harry_buys)
  (h5 : remaining_bottles = initial_bottles - total_buys)
  : remaining_bottles = 24 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_bottles_left_on_shelf_l281_28150


namespace NUMINAMATH_GPT_miaCompletedAdditionalTasksOn6Days_l281_28151

def numDaysCompletingAdditionalTasks (n m : ℕ) : Prop :=
  n + m = 15 ∧ 4 * n + 7 * m = 78

theorem miaCompletedAdditionalTasksOn6Days (n m : ℕ): numDaysCompletingAdditionalTasks n m -> m = 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_miaCompletedAdditionalTasksOn6Days_l281_28151


namespace NUMINAMATH_GPT_karlsson_candies_28_l281_28184

def karlsson_max_candies (n : ℕ) : ℕ := (n * (n - 1)) / 2

theorem karlsson_candies_28 : karlsson_max_candies 28 = 378 := by
  sorry

end NUMINAMATH_GPT_karlsson_candies_28_l281_28184


namespace NUMINAMATH_GPT_inequality_proof_l281_28178

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ( (2 * a + b + c) ^ 2 / (2 * a ^ 2 + (b + c) ^ 2) +
    (2 * b + a + c) ^ 2 / (2 * b ^ 2 + (c + a) ^ 2) +
    (2 * c + a + b) ^ 2 / (2 * c ^ 2 + (a + b) ^ 2)
  ) ≤ 8 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l281_28178


namespace NUMINAMATH_GPT_inclination_angle_of_line_l281_28115

-- Definitions and conditions
def line_equation (x y : ℝ) : Prop := x - y + 3 = 0

-- Theorem statement
theorem inclination_angle_of_line (x y : ℝ) (h : line_equation x y) : angle = 45 := by
  sorry

end NUMINAMATH_GPT_inclination_angle_of_line_l281_28115


namespace NUMINAMATH_GPT_quadratic_has_one_solution_l281_28182

theorem quadratic_has_one_solution (n : ℤ) : 
  (n ^ 2 - 64 = 0) ↔ (n = 8 ∨ n = -8) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_has_one_solution_l281_28182


namespace NUMINAMATH_GPT_symmetric_curve_wrt_line_l281_28165

theorem symmetric_curve_wrt_line {f : ℝ → ℝ → ℝ} :
  (∀ x y : ℝ, f x y = 0 → f (y + 3) (x - 3) = 0) := by
  sorry

end NUMINAMATH_GPT_symmetric_curve_wrt_line_l281_28165


namespace NUMINAMATH_GPT_largest_ball_radius_l281_28148

def torus_inner_radius : ℝ := 2
def torus_outer_radius : ℝ := 4
def circle_center : ℝ × ℝ × ℝ := (3, 0, 1)
def circle_radius : ℝ := 1

theorem largest_ball_radius : ∃ r : ℝ, r = 9 / 4 ∧
  (∃ (sphere_center : ℝ × ℝ × ℝ) (torus_center : ℝ × ℝ × ℝ),
  (sphere_center = (0, 0, r)) ∧
  (torus_center = (3, 0, 1)) ∧
  (dist (0, 0, r) (3, 0, 1) = r + 1)) := sorry

end NUMINAMATH_GPT_largest_ball_radius_l281_28148


namespace NUMINAMATH_GPT_probability_greater_than_4_l281_28105

-- Given conditions
def die_faces : ℕ := 6
def favorable_outcomes : Finset ℕ := {5, 6}

-- Probability calculation
def probability (total : ℕ) (favorable : Finset ℕ) : ℚ :=
  favorable.card / total

theorem probability_greater_than_4 :
  probability die_faces favorable_outcomes = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_greater_than_4_l281_28105


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l281_28131

theorem quadratic_no_real_roots : ∀ (a b c : ℝ), a ≠ 0 → Δ = (b*b - 4*a*c) → x^2 + 3 = 0 → Δ < 0 := by
  sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l281_28131


namespace NUMINAMATH_GPT_father_ate_8_brownies_l281_28172

noncomputable def brownies_initial := 24
noncomputable def brownies_mooney_ate := 4
noncomputable def brownies_after_mooney := brownies_initial - brownies_mooney_ate
noncomputable def brownies_mother_made_next_day := 24
noncomputable def brownies_total_expected := brownies_after_mooney + brownies_mother_made_next_day
noncomputable def brownies_actual_on_counter := 36

theorem father_ate_8_brownies :
  brownies_total_expected - brownies_actual_on_counter = 8 :=
by
  sorry

end NUMINAMATH_GPT_father_ate_8_brownies_l281_28172


namespace NUMINAMATH_GPT_pears_total_correct_l281_28122

noncomputable def pickedPearsTotal (sara_picked tim_picked : Nat) : Nat :=
  sara_picked + tim_picked

theorem pears_total_correct :
    pickedPearsTotal 6 5 = 11 :=
  by
    sorry

end NUMINAMATH_GPT_pears_total_correct_l281_28122


namespace NUMINAMATH_GPT_only_one_true_l281_28155

def statement_dong (xi: Prop) := ¬ xi
def statement_xi (nan: Prop) := ¬ nan
def statement_nan (dong: Prop) := ¬ dong
def statement_bei (nan: Prop) := ¬ (statement_nan nan) 

-- Define the main proof problem assuming all statements
theorem only_one_true : (statement_dong xi → false ∧ statement_xi nan → false ∧ statement_nan dong → true ∧ statement_bei nan → false) 
                        ∨ (statement_dong xi → false ∧ statement_xi nan → true ∧ statement_nan dong → false ∧ statement_bei nan → false) 
                        ∨ (statement_dong xi → true ∧ statement_xi nan → false ∧ statement_nan dong → false ∧ statement_bei nan → true) 
                        ∨ (statement_dong xi → false ∧ statement_xi nan → false ∧ statement_nan dong → false ∧ statement_bei nan → true) 
                        ∧ (statement_nan (statement_dong xi)) = true :=
sorry

end NUMINAMATH_GPT_only_one_true_l281_28155


namespace NUMINAMATH_GPT_P1_coordinates_l281_28114

-- Define initial point coordinates
def P : (ℝ × ℝ) := (0, 3)

-- Define the transformation functions
def move_left (p : ℝ × ℝ) (units : ℝ) : (ℝ × ℝ) := (p.1 - units, p.2)
def move_up (p : ℝ × ℝ) (units : ℝ) : (ℝ × ℝ) := (p.1, p.2 + units)

-- Calculate the coordinates of point P1
def P1 : (ℝ × ℝ) := move_up (move_left P 2) 1

-- Statement to prove
theorem P1_coordinates : P1 = (-2, 4) := by
  sorry

end NUMINAMATH_GPT_P1_coordinates_l281_28114


namespace NUMINAMATH_GPT_bacteria_colony_growth_l281_28147

theorem bacteria_colony_growth : 
  ∃ (n : ℕ), n = 4 ∧ 5 * 3 ^ n > 200 ∧ (∀ (m : ℕ), 5 * 3 ^ m > 200 → m ≥ n) :=
by
  sorry

end NUMINAMATH_GPT_bacteria_colony_growth_l281_28147


namespace NUMINAMATH_GPT_base_conversion_least_sum_l281_28173

theorem base_conversion_least_sum (a b : ℕ) (h : 3 * a + 5 = 5 * b + 3) : a + b = 10 :=
sorry

end NUMINAMATH_GPT_base_conversion_least_sum_l281_28173


namespace NUMINAMATH_GPT_option_C_is_nonnegative_rational_l281_28107

def isNonNegativeRational (x : ℚ) : Prop :=
  x ≥ 0

theorem option_C_is_nonnegative_rational :
  isNonNegativeRational (-( - (4^2 : ℚ))) :=
by
  sorry

end NUMINAMATH_GPT_option_C_is_nonnegative_rational_l281_28107


namespace NUMINAMATH_GPT_average_physics_chemistry_l281_28102

theorem average_physics_chemistry (P C M : ℕ) 
  (h1 : (P + C + M) / 3 = 80)
  (h2 : (P + M) / 2 = 90)
  (h3 : P = 80) :
  (P + C) / 2 = 70 := 
sorry

end NUMINAMATH_GPT_average_physics_chemistry_l281_28102


namespace NUMINAMATH_GPT_inequality_proof_l281_28135

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z > 0) :
  (x^2 * y / (y + z) + y^2 * z / (z + x) + z^2 * x / (x + y) ≥ 1 / 2 * (x^2 + y^2 + z^2)) :=
by sorry

end NUMINAMATH_GPT_inequality_proof_l281_28135


namespace NUMINAMATH_GPT_ratio_of_girls_who_like_pink_l281_28132

theorem ratio_of_girls_who_like_pink 
  (total_students : ℕ) (answered_green : ℕ) (answered_yellow : ℕ) (total_girls : ℕ) (answered_yellow_students : ℕ)
  (portion_girls_pink : ℕ) 
  (h1 : total_students = 30)
  (h2 : answered_green = total_students / 2)
  (h3 : total_girls = 18)
  (h4 : answered_yellow_students = 9)
  (answered_pink := total_students - answered_green - answered_yellow_students)
  (ratio_pink : ℚ := answered_pink / total_girls) : 
  ratio_pink = 1 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_of_girls_who_like_pink_l281_28132


namespace NUMINAMATH_GPT_elberta_has_22_dollars_l281_28161

theorem elberta_has_22_dollars (granny_smith : ℝ) (anjou : ℝ) (elberta : ℝ) 
  (h1 : granny_smith = 75) 
  (h2 : anjou = granny_smith / 4)
  (h3 : elberta = anjou + 3) : 
  elberta = 22 := 
by
  sorry

end NUMINAMATH_GPT_elberta_has_22_dollars_l281_28161


namespace NUMINAMATH_GPT_f_one_value_l281_28111

def f (x a: ℝ) : ℝ := x^2 + a*x - 3*a - 9

theorem f_one_value (a : ℝ) (h : ∀ x, f x a ≥ 0) : f 1 a = 4 :=
by
  sorry

end NUMINAMATH_GPT_f_one_value_l281_28111


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l281_28117

def represents_ellipse (k : ℝ) (x y : ℝ) :=
    1 < k ∧ k < 5 ∧ k ≠ 3

theorem necessary_but_not_sufficient_condition (k : ℝ) (x y : ℝ):
    (1 < k ∧ k < 5) → (represents_ellipse k x y) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l281_28117


namespace NUMINAMATH_GPT_average_score_l281_28157

theorem average_score (classA_students classB_students : ℕ)
  (avg_score_classA avg_score_classB : ℕ)
  (h_classA : classA_students = 40)
  (h_classB : classB_students = 50)
  (h_avg_classA : avg_score_classA = 90)
  (h_avg_classB : avg_score_classB = 81) :
  (classA_students * avg_score_classA + classB_students * avg_score_classB) / 
  (classA_students + classB_students) = 85 := 
  by sorry

end NUMINAMATH_GPT_average_score_l281_28157


namespace NUMINAMATH_GPT_avg_two_expressions_l281_28162

theorem avg_two_expressions (a : ℝ) (h : ((2 * a + 16) + (3 * a - 8)) / 2 = 84) : a = 32 := sorry

end NUMINAMATH_GPT_avg_two_expressions_l281_28162


namespace NUMINAMATH_GPT_box_width_l281_28103

theorem box_width (W S : ℕ) (h1 : 30 * W * 12 = 80 * S^3) (h2 : S ∣ 30 ∧ S ∣ 12) : W = 48 :=
by
  sorry

end NUMINAMATH_GPT_box_width_l281_28103


namespace NUMINAMATH_GPT_ratio_of_times_gina_chooses_to_her_sister_l281_28176

theorem ratio_of_times_gina_chooses_to_her_sister (sister_shows : ℕ) (minutes_per_show : ℕ) (gina_minutes : ℕ) (ratio : ℕ × ℕ) :
  sister_shows = 24 →
  minutes_per_show = 50 →
  gina_minutes = 900 →
  ratio = (900 / Nat.gcd 900 1200, 1200 / Nat.gcd 900 1200) →
  ratio = (3, 4) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_ratio_of_times_gina_chooses_to_her_sister_l281_28176


namespace NUMINAMATH_GPT_probability_two_white_balls_same_color_l281_28124

theorem probability_two_white_balls_same_color :
  let num_white := 3
  let num_black := 2
  let total_combinations_white := num_white.choose 2
  let total_combinations_black := num_black.choose 2
  let total_combinations_same_color := total_combinations_white + total_combinations_black
  (total_combinations_white + total_combinations_black > 0) →
  (total_combinations_white / total_combinations_same_color) = (3 / 4) :=
by
  let num_white := 3
  let num_black := 2
  let total_combinations_white := num_white.choose 2
  let total_combinations_black := num_black.choose 2
  let total_combinations_same_color := total_combinations_white + total_combinations_black
  intro h
  sorry

end NUMINAMATH_GPT_probability_two_white_balls_same_color_l281_28124


namespace NUMINAMATH_GPT_total_balloons_l281_28187

theorem total_balloons (Gold Silver Black Total : Nat) (h1 : Gold = 141)
  (h2 : Silver = 2 * Gold) (h3 : Black = 150) (h4 : Total = Gold + Silver + Black) :
  Total = 573 := 
by
  sorry

end NUMINAMATH_GPT_total_balloons_l281_28187


namespace NUMINAMATH_GPT_contrapositive_proposition_l281_28143

theorem contrapositive_proposition (x : ℝ) : 
  (x^2 = 1 → (x = 1 ∨ x = -1)) ↔ ((x ≠ 1 ∧ x ≠ -1) → x^2 ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_proposition_l281_28143


namespace NUMINAMATH_GPT_old_edition_pages_l281_28145

-- Define the conditions
variables (new_edition : ℕ) (old_edition : ℕ)

-- The conditions given in the problem
axiom new_edition_pages : new_edition = 450
axiom pages_relationship : new_edition = 2 * old_edition - 230

-- Goal: Prove that the old edition Geometry book had 340 pages
theorem old_edition_pages : old_edition = 340 :=
by sorry

end NUMINAMATH_GPT_old_edition_pages_l281_28145


namespace NUMINAMATH_GPT_clever_question_l281_28174

-- Define the conditions as predicates
def inhabitants_truthful (city : String) : Prop := 
  city = "Mars-Polis"

def inhabitants_lying (city : String) : Prop := 
  city = "Mars-City"

def responses (question : String) (city : String) : String :=
  if question = "Are we in Mars-City?" then
    if city = "Mars-City" then "No" else "Yes"
  else if question = "Do you live here?" then
    if city = "Mars-City" then "No" else "Yes"
  else "Unknown"

-- Define the main theorem
theorem clever_question (city : String) (initial_response : String) :
  (inhabitants_truthful city ∨ inhabitants_lying city) →
  responses "Are we in Mars-City?" city = initial_response →
  responses "Do you live here?" city = "Yes" ∨ responses "Do you live here?" city = "No" :=
by
  sorry

end NUMINAMATH_GPT_clever_question_l281_28174


namespace NUMINAMATH_GPT_num_passenger_cars_l281_28175

noncomputable def passengerCars (p c : ℕ) : Prop :=
  c = p / 2 + 3 ∧ p + c = 69

theorem num_passenger_cars (p c : ℕ) (h : passengerCars p c) : p = 44 :=
by
  unfold passengerCars at h
  cases h
  sorry

end NUMINAMATH_GPT_num_passenger_cars_l281_28175


namespace NUMINAMATH_GPT_find_first_year_l281_28109

-- Define sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) % 10 + (n / 100) % 10 + (n / 10) % 10 + n % 10

-- Define the conditions
def after_2020 (n : ℕ) : Prop := n > 2020
def sum_of_digits_eq (n required_sum : ℕ) : Prop := sum_of_digits n = required_sum

noncomputable def first_year_after_2020_with_digit_sum_15 : ℕ :=
  2049

-- The statement to be proved
theorem find_first_year : 
  ∃ y : ℕ, after_2020 y ∧ sum_of_digits_eq y 15 ∧ y = first_year_after_2020_with_digit_sum_15 :=
by
  sorry

end NUMINAMATH_GPT_find_first_year_l281_28109


namespace NUMINAMATH_GPT_percent_problem_l281_28144

theorem percent_problem (x y z w : ℝ) 
  (h1 : x = 1.20 * y) 
  (h2 : y = 0.40 * z) 
  (h3 : z = 0.70 * w) : 
  x = 0.336 * w :=
sorry

end NUMINAMATH_GPT_percent_problem_l281_28144


namespace NUMINAMATH_GPT_Jamie_liquid_limit_l281_28192

theorem Jamie_liquid_limit :
  let milk_ounces := 8
  let grape_juice_ounces := 16
  let water_bottle_limit := 8
  let already_consumed := milk_ounces + grape_juice_ounces
  let max_before_bathroom := already_consumed + water_bottle_limit
  max_before_bathroom = 32 :=
by
  sorry

end NUMINAMATH_GPT_Jamie_liquid_limit_l281_28192


namespace NUMINAMATH_GPT_rect_length_is_20_l281_28123

-- Define the conditions
def rect_length_four_times_width (l w : ℝ) : Prop := l = 4 * w
def rect_area_100 (l w : ℝ) : Prop := l * w = 100

-- The main theorem to prove
theorem rect_length_is_20 {l w : ℝ} (h1 : rect_length_four_times_width l w) (h2 : rect_area_100 l w) : l = 20 := by
  sorry

end NUMINAMATH_GPT_rect_length_is_20_l281_28123


namespace NUMINAMATH_GPT_increase_by_fraction_l281_28133

theorem increase_by_fraction (original_value : ℕ) (fraction : ℚ) : original_value = 120 → fraction = 5/6 → original_value + original_value * fraction = 220 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_increase_by_fraction_l281_28133


namespace NUMINAMATH_GPT_work_completion_time_l281_28134

theorem work_completion_time
  (W : ℝ) -- Total work
  (p_rate : ℝ := W / 40) -- p's work rate
  (q_rate : ℝ := W / 24) -- q's work rate
  (work_done_by_p_alone : ℝ := 8 * p_rate) -- Work done by p in first 8 days
  (remaining_work : ℝ := W - work_done_by_p_alone) -- Remaining work after 8 days
  (combined_rate : ℝ := p_rate + q_rate) -- Combined work rate of p and q
  (time_to_complete_remaining_work : ℝ := remaining_work / combined_rate) -- Time to complete remaining work
  : (8 + time_to_complete_remaining_work) = 20 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_time_l281_28134


namespace NUMINAMATH_GPT_sasha_train_problem_l281_28118

def wagon_number (W : ℕ) (S : ℕ) : Prop :=
  -- Conditions
  (1 ≤ W ∧ W ≤ 9) ∧          -- Wagon number is a single-digit number
  (S < W) ∧                  -- Seat number is less than the wagon number
  ( (W = 1 ∧ S ≠ 1) ∨ 
    (W = 2 ∧ S = 1)
  ) -- Monday is the 1st or 2nd day of the month and corresponding seat constraints

theorem sasha_train_problem :
  ∃ (W S : ℕ), wagon_number W S ∧ W = 2 ∧ S = 1 :=
by
  sorry

end NUMINAMATH_GPT_sasha_train_problem_l281_28118


namespace NUMINAMATH_GPT_Aiyanna_has_more_cookies_l281_28193

def Alyssa_cookies : ℕ := 129
def Aiyanna_cookies : ℕ := 140

theorem Aiyanna_has_more_cookies :
  Aiyanna_cookies - Alyssa_cookies = 11 := by
  sorry

end NUMINAMATH_GPT_Aiyanna_has_more_cookies_l281_28193


namespace NUMINAMATH_GPT_speed_of_current_l281_28198

-- Definitions for the conditions
variables (m c : ℝ)

-- Condition 1: man's speed with the current
def speed_with_current := m + c = 16

-- Condition 2: man's speed against the current
def speed_against_current := m - c = 9.6

-- The goal is to prove c = 3.2 given the conditions
theorem speed_of_current (h1 : speed_with_current m c) 
                         (h2 : speed_against_current m c) :
  c = 3.2 := 
sorry

end NUMINAMATH_GPT_speed_of_current_l281_28198


namespace NUMINAMATH_GPT_max_convex_quadrilaterals_l281_28169

-- Define the points on the plane and the conditions
variable (A : Fin 7 → (ℝ × ℝ))

-- Hypothesis that any 3 given points are not collinear
def not_collinear (P Q R : (ℝ × ℝ)) : Prop :=
  (Q.1 - P.1) * (R.2 - P.2) ≠ (Q.2 - P.2) * (R.1 - P.1)

-- Hypothesis that the convex hull of all points is \triangle A1 A2 A3
def convex_hull_triangle (A : Fin 7 → (ℝ × ℝ)) : Prop :=
  ∀ (i j k : Fin 7), i ≠ j → j ≠ k → i ≠ k → not_collinear (A i) (A j) (A k)

-- The theorem to be proven
theorem max_convex_quadrilaterals :
  convex_hull_triangle A →
  (∀ i j k : Fin 7, i ≠ j → j ≠ k → i ≠ k → not_collinear (A i) (A j) (A k)) →
  ∃ n, n = 17 := 
by
  sorry

end NUMINAMATH_GPT_max_convex_quadrilaterals_l281_28169


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l281_28154

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
    (h_arith_seq : ∀ n, a (n + 1) = a n + d)
    (h_a5 : a 5 = 3)
    (h_a6 : a 6 = -2) :
  a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = -49 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l281_28154


namespace NUMINAMATH_GPT_square_area_from_diagonal_l281_28191

theorem square_area_from_diagonal (d : ℝ) (h : d = 28) : (∃ A : ℝ, A = 392) :=
by
  sorry

end NUMINAMATH_GPT_square_area_from_diagonal_l281_28191


namespace NUMINAMATH_GPT_bobby_additional_candy_l281_28121

variable (initial_candy additional_candy chocolate total_candy : ℕ)
variable (bobby_initial_candy : initial_candy = 38)
variable (bobby_ate_chocolate : chocolate = 16)
variable (bobby_more_candy : initial_candy + additional_candy = 58 + chocolate)

theorem bobby_additional_candy :
  additional_candy = 36 :=
by {
  sorry
}

end NUMINAMATH_GPT_bobby_additional_candy_l281_28121


namespace NUMINAMATH_GPT_son_l281_28189

theorem son's_age (S F : ℕ) (h1: F = S + 27) (h2: F + 2 = 2 * (S + 2)) : S = 25 := by
  sorry

end NUMINAMATH_GPT_son_l281_28189


namespace NUMINAMATH_GPT_find_num_chickens_l281_28146

-- Definitions based on problem conditions
def num_dogs : ℕ := 2
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2
def total_legs_seen : ℕ := 12

-- Proof problem: Prove the number of chickens Mrs. Hilt saw
theorem find_num_chickens (C : ℕ) (h1 : num_dogs * legs_per_dog + C * legs_per_chicken = total_legs_seen) : C = 2 := 
sorry

end NUMINAMATH_GPT_find_num_chickens_l281_28146


namespace NUMINAMATH_GPT_value_of_def_ef_l281_28138

theorem value_of_def_ef
  (a b c d e f : ℝ)
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 500)
  (h4 : (a * f) / (c * d) = 1)
  : d * e * f = 250 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_def_ef_l281_28138


namespace NUMINAMATH_GPT_solve_for_x_l281_28108

theorem solve_for_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0) → x = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l281_28108


namespace NUMINAMATH_GPT_special_op_equality_l281_28100

def special_op (x y : ℕ) : ℕ := x * y - x - 2 * y

theorem special_op_equality : (special_op 7 4) - (special_op 4 7) = 3 := by
  sorry

end NUMINAMATH_GPT_special_op_equality_l281_28100


namespace NUMINAMATH_GPT_triangle_longest_side_l281_28101

theorem triangle_longest_side 
  (x : ℝ)
  (h1 : 7 + (x + 4) + (2 * x + 1) = 36) :
  2 * x + 1 = 17 := by
  sorry

end NUMINAMATH_GPT_triangle_longest_side_l281_28101


namespace NUMINAMATH_GPT_last_two_digits_7_pow_2017_l281_28149

noncomputable def last_two_digits_of_pow :=
  ∀ n : ℕ, ∃ (d : ℕ), d < 100 ∧ 7^n % 100 = d

theorem last_two_digits_7_pow_2017 : ∃ (d : ℕ), d = 7 ∧ 7^2017 % 100 = d :=
by
  sorry

end NUMINAMATH_GPT_last_two_digits_7_pow_2017_l281_28149


namespace NUMINAMATH_GPT_find_extrema_of_A_l281_28113

theorem find_extrema_of_A (x y : ℝ) (h : x^2 + y^2 = 4) : 2 ≤ x^2 + x * y + y^2 ∧ x^2 + x * y + y^2 ≤ 6 :=
by 
  sorry

end NUMINAMATH_GPT_find_extrema_of_A_l281_28113


namespace NUMINAMATH_GPT_range_of_a_l281_28153

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 4 → (x^2 + 2 * (a - 1) * x + 2) ≥ (y^2 + 2 * (a - 1) * y + 2)) →
  a ≤ -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l281_28153


namespace NUMINAMATH_GPT_find_T_l281_28142

variable {n : ℕ}
variable {a b : ℕ → ℕ}
variable {S T : ℕ → ℕ}

-- Conditions
axiom h1 : ∀ n, b n - a n = 2^n + 1
axiom h2 : ∀ n, S n + T n = 2^(n + 1) + n^2 - 2

-- Goal
theorem find_T (n : ℕ) (a b S T : ℕ → ℕ)
  (h1 : ∀ n, b n - a n = 2^n + 1)
  (h2 : ∀ n, S n + T n = 2^(n + 1) + n^2 - 2) :
  T n = 2^(n + 1) + n * (n + 1) / 2 - 5 := sorry

end NUMINAMATH_GPT_find_T_l281_28142


namespace NUMINAMATH_GPT_function_zeros_range_l281_28129

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then (1 / 2)^x + 2 / x else x * Real.log x - a

theorem function_zeros_range (a : ℝ) :
  (∀ x : ℝ, f x a = 0 → x < 0) ∧ (∀ x : ℝ, f x a = 0 → x > 0 → (a > -1 / Real.exp 1 ∧ a < 0)) ↔
  (a > -1 / Real.exp 1 ∧ a < 0) :=
sorry

end NUMINAMATH_GPT_function_zeros_range_l281_28129


namespace NUMINAMATH_GPT_binomial_10_3_l281_28110

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end NUMINAMATH_GPT_binomial_10_3_l281_28110


namespace NUMINAMATH_GPT_repeating_decimal_as_fraction_l281_28177

/-- Define x as the repeating decimal 7.182182... -/
def x : ℚ := 
  7 + 182 / 999

/-- Define y as the fraction 7175/999 -/
def y : ℚ := 
  7175 / 999

/-- Theorem stating that the repeating decimal 7.182182... is equal to the fraction 7175/999 -/
theorem repeating_decimal_as_fraction : x = y :=
sorry

end NUMINAMATH_GPT_repeating_decimal_as_fraction_l281_28177


namespace NUMINAMATH_GPT_measure_time_with_hourglasses_l281_28179

def hourglass7 : ℕ := 7
def hourglass11 : ℕ := 11
def target_time : ℕ := 15

theorem measure_time_with_hourglasses :
  ∃ (time_elapsed : ℕ), time_elapsed = target_time :=
by
  use 15
  sorry

end NUMINAMATH_GPT_measure_time_with_hourglasses_l281_28179


namespace NUMINAMATH_GPT_problem_l281_28106

def operation (a b : ℤ) (h : a ≠ 0) : ℤ := (b - a) ^ 2 / a ^ 2

theorem problem : 
  operation (-1) (operation 1 (-1) (by decide)) (by decide) = 25 := 
by
  sorry

end NUMINAMATH_GPT_problem_l281_28106


namespace NUMINAMATH_GPT_ray_steps_problem_l281_28156

theorem ray_steps_problem : ∃ n, n > 15 ∧ n % 3 = 2 ∧ n % 7 = 1 ∧ n % 4 = 3 ∧ n = 71 :=
by
  sorry

end NUMINAMATH_GPT_ray_steps_problem_l281_28156


namespace NUMINAMATH_GPT_hour_hand_degrees_per_hour_l281_28159

-- Definitions based on the conditions
def number_of_rotations_in_6_days : ℕ := 12
def degrees_per_rotation : ℕ := 360
def hours_in_6_days : ℕ := 6 * 24

-- Statement to prove
theorem hour_hand_degrees_per_hour :
  (number_of_rotations_in_6_days * degrees_per_rotation) / hours_in_6_days = 30 :=
by sorry

end NUMINAMATH_GPT_hour_hand_degrees_per_hour_l281_28159


namespace NUMINAMATH_GPT_minimum_tangent_length_4_l281_28104

noncomputable def minimum_tangent_length (a b : ℝ) : ℝ :=
  Real.sqrt ((b + 4)^2 + (b - 2)^2 - 2)

theorem minimum_tangent_length_4 :
  ∀ (a b : ℝ), (x^2 + y^2 + 2 * x - 4 * y + 3 = 0) ∧ (x = a ∧ y = b) ∧ (2*a*x + b*y + 6 = 0) → 
    minimum_tangent_length a b = 4 :=
by
  sorry

end NUMINAMATH_GPT_minimum_tangent_length_4_l281_28104


namespace NUMINAMATH_GPT_subtract_3a_result_l281_28141

theorem subtract_3a_result (a : ℝ) : 
  (9 * a^2 - 3 * a + 8) + 3 * a = 9 * a^2 + 8 := 
sorry

end NUMINAMATH_GPT_subtract_3a_result_l281_28141


namespace NUMINAMATH_GPT_minimum_dimes_to_afford_sneakers_l281_28181

-- Define constants and conditions using Lean
def sneaker_cost : ℝ := 45.35
def ten_dollar_bills_count : ℕ := 3
def quarter_count : ℕ := 4
def dime_value : ℝ := 0.1
def quarter_value : ℝ := 0.25
def ten_dollar_bill_value : ℝ := 10.0

-- Define a function to calculate the total amount based on the number of dimes
def total_amount (dimes : ℕ) : ℝ :=
  (ten_dollar_bills_count * ten_dollar_bill_value) +
  (quarter_count * quarter_value) +
  (dimes * dime_value)

-- The main theorem to be proven
theorem minimum_dimes_to_afford_sneakers (n : ℕ) : total_amount n ≥ sneaker_cost ↔ n ≥ 144 :=
by
  sorry

end NUMINAMATH_GPT_minimum_dimes_to_afford_sneakers_l281_28181


namespace NUMINAMATH_GPT_quadratic_solution_set_R_l281_28130

theorem quadratic_solution_set_R (a b c : ℝ) (h1 : a ≠ 0) (h2 : a < 0) (h3 : b^2 - 4 * a * c < 0) : 
  ∀ x : ℝ, a * x^2 + b * x + c < 0 :=
by sorry

end NUMINAMATH_GPT_quadratic_solution_set_R_l281_28130


namespace NUMINAMATH_GPT_smallest_integer_with_remainders_l281_28180

theorem smallest_integer_with_remainders :
  ∃ n : ℕ, 
  n > 0 ∧ 
  n % 2 = 1 ∧ 
  n % 3 = 2 ∧ 
  n % 4 = 3 ∧ 
  n % 10 = 9 ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 2 = 1 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 10 = 9 → n ≤ m) := 
sorry

end NUMINAMATH_GPT_smallest_integer_with_remainders_l281_28180


namespace NUMINAMATH_GPT_probability_same_heads_l281_28112

noncomputable def probability_heads_after_flips (p : ℚ) (n : ℕ) : ℚ :=
  (1 - p)^(n-1) * p

theorem probability_same_heads (p : ℚ) (n : ℕ) : p = 1/3 → 
  ∑' n : ℕ, (probability_heads_after_flips p n)^4 = 1/65 := 
sorry

end NUMINAMATH_GPT_probability_same_heads_l281_28112


namespace NUMINAMATH_GPT_expression_value_l281_28188

theorem expression_value (m n a b x : ℤ) (h1 : m = -n) (h2 : a * b = 1) (h3 : |x| = 3) :
  x = 3 ∨ x = -3 → (x = 3 → x^3 - (1 + m + n - a * b) * x^2010 + (m + n) * x^2007 + (-a * b)^2009 = 26) ∧
                  (x = -3 → x^3 - (1 + m + n - a * b) * x^2010 + (m + n) * x^2007 + (-a * b)^2009 = -28) := by
  sorry

end NUMINAMATH_GPT_expression_value_l281_28188


namespace NUMINAMATH_GPT_proof_base_5_conversion_and_addition_l281_28139

-- Define the given numbers in decimal (base 10)
def n₁ := 45
def n₂ := 25

-- Base 5 conversion function and proofs of correctness
def to_base_5 (n : ℕ) : ℕ := sorry
def from_base_5 (n : ℕ) : ℕ := sorry

-- Converted values to base 5
def a₅ : ℕ := to_base_5 n₁
def b₅ : ℕ := to_base_5 n₂

-- Sum in base 5
def c₅ : ℕ := a₅ + b₅  -- addition in base 5

-- Convert the final sum back to decimal base 10
def d₁₀ : ℕ := from_base_5 c₅

theorem proof_base_5_conversion_and_addition :
  d₁₀ = 65 ∧ to_base_5 65 = 230 :=
by sorry

end NUMINAMATH_GPT_proof_base_5_conversion_and_addition_l281_28139


namespace NUMINAMATH_GPT_age_contradiction_l281_28127

-- Given the age ratios and future age of Sandy
def current_ages (x : ℕ) : ℕ × ℕ × ℕ := (4 * x, 3 * x, 5 * x)
def sandy_age_after_6_years (age_sandy_current : ℕ) : ℕ := age_sandy_current + 6

-- Given conditions
def ratio_condition (x : ℕ) (age_sandy age_molly age_danny : ℕ) : Prop :=
  current_ages x = (age_sandy, age_molly, age_danny)

def sandy_age_condition (age_sandy_current : ℕ) : Prop :=
  sandy_age_after_6_years age_sandy_current = 30

def age_sum_condition (age_molly age_danny : ℕ) : Prop :=
  age_molly + age_danny = (age_molly + 4) + (age_danny + 4)

-- Main theorem
theorem age_contradiction : ∃ x age_sandy age_molly age_danny, 
  ratio_condition x age_sandy age_molly age_danny ∧
  sandy_age_condition age_sandy ∧
  (¬ age_sum_condition age_molly age_danny) := 
by
  -- Omitting the proof; the focus is on setting up the statement only
  sorry

end NUMINAMATH_GPT_age_contradiction_l281_28127


namespace NUMINAMATH_GPT_y_intercept_of_line_l281_28158

theorem y_intercept_of_line (x y : ℝ) (h : 3 * x - 2 * y + 7 = 0) (hx : x = 0) : y = 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l281_28158


namespace NUMINAMATH_GPT_all_roots_are_nth_roots_of_unity_l281_28167

noncomputable def smallest_positive_integer_n : ℕ :=
  5
  
theorem all_roots_are_nth_roots_of_unity :
  (∀ z : ℂ, (z^4 + z^3 + z^2 + z + 1 = 0) → z^(smallest_positive_integer_n) = 1) :=
  by
    sorry

end NUMINAMATH_GPT_all_roots_are_nth_roots_of_unity_l281_28167


namespace NUMINAMATH_GPT_base_7_to_base_10_equiv_l281_28183

theorem base_7_to_base_10_equiv : 
  ∀ (d2 d1 d0 : ℕ), 
      d2 = 3 → d1 = 4 → d0 = 6 → 
      (d2 * 7^2 + d1 * 7^1 + d0 * 7^0) = 181 := 
by 
  sorry

end NUMINAMATH_GPT_base_7_to_base_10_equiv_l281_28183


namespace NUMINAMATH_GPT_selling_price_for_loss_l281_28199

noncomputable def cp : ℝ := 640
def sp1 : ℝ := 768
def sp2 : ℝ := 448
def sp_profitable_sale : ℝ := 832

theorem selling_price_for_loss :
  sp_profitable_sale - cp = cp - sp2 :=
by
  sorry

end NUMINAMATH_GPT_selling_price_for_loss_l281_28199


namespace NUMINAMATH_GPT_summer_camp_activity_l281_28190

theorem summer_camp_activity :
  ∃ (a b c d e f : ℕ), 
  a + b + c + d + 3 * e + 4 * f = 12 ∧ 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧
  f = 1 := by
  sorry

end NUMINAMATH_GPT_summer_camp_activity_l281_28190


namespace NUMINAMATH_GPT_arithmetic_sequence_iff_condition_l281_28140

-- Definitions: A sequence and the condition
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_iff_condition (a : ℕ → ℝ) :
  is_arithmetic_sequence a ↔ (∀ n : ℕ, 2 * a (n + 1) = a n + a (n + 2)) :=
by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_iff_condition_l281_28140


namespace NUMINAMATH_GPT_correct_combined_monthly_rate_of_profit_l281_28116

structure Book :=
  (cost_price : ℕ)
  (selling_price : ℕ)
  (months_held : ℕ)

def profit (b : Book) : ℕ :=
  b.selling_price - b.cost_price

def monthly_rate_of_profit (b : Book) : ℕ :=
  if b.months_held = 0 then profit b else profit b / b.months_held

def combined_monthly_rate_of_profit (b1 b2 b3 : Book) : ℕ :=
  monthly_rate_of_profit b1 + monthly_rate_of_profit b2 + monthly_rate_of_profit b3

theorem correct_combined_monthly_rate_of_profit :
  combined_monthly_rate_of_profit
    {cost_price := 50, selling_price := 90, months_held := 1}
    {cost_price := 120, selling_price := 150, months_held := 2}
    {cost_price := 75, selling_price := 110, months_held := 0} 
    = 90 := 
by
  sorry

end NUMINAMATH_GPT_correct_combined_monthly_rate_of_profit_l281_28116


namespace NUMINAMATH_GPT_solve_system_of_equations_l281_28166

theorem solve_system_of_equations
  (x y : ℚ)
  (h1 : 5 * x - 3 * y = -7)
  (h2 : 4 * x + 6 * y = 34) :
  x = 10 / 7 ∧ y = 33 / 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l281_28166


namespace NUMINAMATH_GPT_range_of_k_l281_28152

theorem range_of_k (k : ℝ) (a : ℕ+ → ℝ) (h : ∀ n : ℕ+, a n = 2 * (n:ℕ)^2 + k * (n:ℕ)) 
  (increasing : ∀ n : ℕ+, a n < a (n + 1)) : 
  k > -6 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_k_l281_28152


namespace NUMINAMATH_GPT_combinations_of_coins_l281_28171

theorem combinations_of_coins (p n d : ℕ) (h₁ : p ≥ 0) (h₂ : n ≥ 0) (h₃ : d ≥ 0) 
  (value_eq : p + 5 * n + 10 * d = 25) : 
  ∃! c : ℕ, c = 12 :=
sorry

end NUMINAMATH_GPT_combinations_of_coins_l281_28171


namespace NUMINAMATH_GPT_problem_solution_l281_28197

theorem problem_solution :
  ∀ x y : ℝ, 9 * y^2 + 6 * x * y + x + 12 = 0 → (x ≤ -3 ∨ x ≥ 4) :=
  sorry

end NUMINAMATH_GPT_problem_solution_l281_28197
