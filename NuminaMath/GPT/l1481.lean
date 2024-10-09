import Mathlib

namespace no_real_roots_poly_l1481_148147

theorem no_real_roots_poly (a b c : ℝ) (h : |a| + |b| + |c| ≤ Real.sqrt 2) :
  ∀ x : ℝ, x^4 + a*x^3 + b*x^2 + c*x + 1 > 0 := 
  sorry

end no_real_roots_poly_l1481_148147


namespace simplify_and_evaluate_l1481_148167

theorem simplify_and_evaluate :
  let a := 1
  let b := 2
  (a - b) ^ 2 - a * (a - b) + (a + b) * (a - b) = -1 := by
  sorry

end simplify_and_evaluate_l1481_148167


namespace arithmetic_progression_ratio_l1481_148174

variable {α : Type*} [LinearOrder α] [Field α]

theorem arithmetic_progression_ratio (a d : α) (h : 15 * a + 105 * d = 3 * (8 * a + 28 * d)) : a / d = 7 / 3 := 
by sorry

end arithmetic_progression_ratio_l1481_148174


namespace log_suff_nec_l1481_148192

theorem log_suff_nec (a b : ℝ) (ha : a > 0) (hb : b > 0) : ¬ ((a > b) ↔ (Real.log b / Real.log a < 1)) := 
sorry

end log_suff_nec_l1481_148192


namespace smallest_x_plus_y_l1481_148151

theorem smallest_x_plus_y (x y : ℕ) (h1 : x ≥ 1) (h2 : y ≥ 1) (h3 : x^2 - 29 * y^2 = 1) : x + y = 11621 := 
sorry

end smallest_x_plus_y_l1481_148151


namespace solution_is_13_l1481_148198

def marbles_in_jars : Prop :=
  let jar1 := (5, 3, 1)  -- (red, blue, green)
  let jar2 := (1, 5, 3)  -- (red, blue, green)
  let jar3 := (3, 1, 5)  -- (red, blue, green)
  let total_ways := 125 + 15 + 15 + 3 + 27 + 15
  let favorable_ways := 125
  let probability := favorable_ways / total_ways
  let simplified_probability := 5 / 8
  let m := 5
  let n := 8
  m + n = 13

theorem solution_is_13 : marbles_in_jars :=
by {
  sorry
}

end solution_is_13_l1481_148198


namespace prove_f_10_l1481_148182

variable (f : ℝ → ℝ)

-- Conditions from the problem
def condition : Prop := ∀ x : ℝ, f (3 ^ x) = x

-- Statement of the problem
theorem prove_f_10 (h : condition f) : f 10 = Real.log 10 / Real.log 3 :=
by
  sorry

end prove_f_10_l1481_148182


namespace common_difference_of_consecutive_multiples_l1481_148171

/-- The sides of a rectangular prism are consecutive multiples of a certain number n. The base area is 450.
    Prove that the common difference between the consecutive multiples is 15. -/
theorem common_difference_of_consecutive_multiples (n d : ℕ) (h₁ : n * (n + d) = 450) : d = 15 :=
sorry

end common_difference_of_consecutive_multiples_l1481_148171


namespace total_rainfall_recorded_l1481_148166

-- Define the conditions based on the rainfall amounts for each day
def rainfall_monday : ℝ := 0.16666666666666666
def rainfall_tuesday : ℝ := 0.4166666666666667
def rainfall_wednesday : ℝ := 0.08333333333333333

-- State the theorem: the total rainfall recorded over the three days is 0.6666666666666667 cm.
theorem total_rainfall_recorded :
  (rainfall_monday + rainfall_tuesday + rainfall_wednesday) = 0.6666666666666667 := by
  sorry

end total_rainfall_recorded_l1481_148166


namespace pet_store_dogs_count_l1481_148124

def initial_dogs : ℕ := 2
def sunday_received_dogs : ℕ := 5
def sunday_sold_dogs : ℕ := 2
def monday_received_dogs : ℕ := 3
def monday_returned_dogs : ℕ := 1
def tuesday_received_dogs : ℕ := 4
def tuesday_sold_dogs : ℕ := 3

theorem pet_store_dogs_count :
  initial_dogs 
  + sunday_received_dogs - sunday_sold_dogs
  + monday_received_dogs + monday_returned_dogs
  + tuesday_received_dogs - tuesday_sold_dogs = 10 := 
sorry

end pet_store_dogs_count_l1481_148124


namespace theater_rows_25_l1481_148137

theorem theater_rows_25 (n : ℕ) (x : ℕ) (k : ℕ) (h : n = 1000) (h1 : k > 16) (h2 : (2 * x + k) * (k + 1) = 2000) : (k + 1) = 25 :=
by
  -- The proof goes here, which we omit for the problem statement.
  sorry

end theater_rows_25_l1481_148137


namespace cost_to_feed_turtles_l1481_148165

-- Define the conditions
def ounces_per_half_pound : ℝ := 1 
def total_weight_turtles : ℝ := 30
def food_per_half_pound : ℝ := 0.5
def ounces_per_jar : ℝ := 15
def cost_per_jar : ℝ := 2

-- Define the statement to prove
theorem cost_to_feed_turtles : (total_weight_turtles / food_per_half_pound) / ounces_per_jar * cost_per_jar = 8 := by
  sorry

end cost_to_feed_turtles_l1481_148165


namespace num_divisible_by_both_digits_l1481_148152

theorem num_divisible_by_both_digits : 
  ∃ n, n = 14 ∧ ∀ (d : ℕ), (d ≥ 10 ∧ d < 100) → 
      (∀ a b, (d = 10 * a + b) → d % a = 0 ∧ d % b = 0 → (a = b ∨ a * 2 = b ∨ a * 5 = b)) :=
sorry

end num_divisible_by_both_digits_l1481_148152


namespace sequence_general_term_l1481_148154

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, n ≥ 1 → a (n + 1) = a n + 2) : ∀ n, a n = 2 * n - 1 :=
by
  sorry

end sequence_general_term_l1481_148154


namespace find_north_speed_l1481_148132

-- Define the variables and conditions
variables (v : ℝ)  -- the speed of the cyclist going towards the north
def south_speed : ℝ := 25  -- the speed of the cyclist going towards the south is 25 km/h
def time_taken : ℝ := 1.4285714285714286  -- time taken to be 50 km apart
def distance_apart : ℝ := 50  -- distance apart after given time

-- Define the hypothesis based on the conditions
def relative_speed (v : ℝ) : ℝ := v + south_speed
def distance_formula (v : ℝ) : Prop :=
  distance_apart = relative_speed v * time_taken

-- The statement to prove
theorem find_north_speed : distance_formula v → v = 10 :=
  sorry

end find_north_speed_l1481_148132


namespace notable_features_points_l1481_148121

namespace Points3D

def is_first_octant (x y z : ℝ) : Prop := (x > 0) ∧ (y > 0) ∧ (z > 0)
def is_second_octant (x y z : ℝ) : Prop := (x < 0) ∧ (y > 0) ∧ (z > 0)
def is_eighth_octant (x y z : ℝ) : Prop := (x > 0) ∧ (y < 0) ∧ (z < 0)
def lies_in_YOZ_plane (x y z : ℝ) : Prop := (x = 0) ∧ (y ≠ 0) ∧ (z ≠ 0)
def lies_on_OY_axis (x y z : ℝ) : Prop := (x = 0) ∧ (y ≠ 0) ∧ (z = 0)
def is_origin (x y z : ℝ) : Prop := (x = 0) ∧ (y = 0) ∧ (z = 0)

theorem notable_features_points :
  is_first_octant 3 2 6 ∧
  is_second_octant (-2) 3 1 ∧
  is_eighth_octant 1 (-4) (-2) ∧
  is_eighth_octant 1 (-2) (-1) ∧
  lies_in_YOZ_plane 0 4 1 ∧
  lies_on_OY_axis 0 2 0 ∧
  is_origin 0 0 0 :=
by
  sorry

end Points3D

end notable_features_points_l1481_148121


namespace exam_question_correct_count_l1481_148107

theorem exam_question_correct_count (C W : ℕ) (h1 : C + W = 60) (h2 : 4 * C - W = 110) : C = 34 :=
by
  sorry

end exam_question_correct_count_l1481_148107


namespace correct_equation_after_moving_digit_l1481_148112

theorem correct_equation_after_moving_digit :
  (101 - 102 = 1) →
  101 - 10^2 = 1 :=
by
  intro h
  sorry

end correct_equation_after_moving_digit_l1481_148112


namespace sinA_value_find_b_c_l1481_148190

-- Define the conditions
def triangle (A B C : Type) (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

variable {A B C : Type} (a b c : ℝ)
variable {S_triangle_ABC : ℝ}
variable {cosB : ℝ}

-- Given conditions
axiom cosB_val : cosB = 3 / 5
axiom a_val : a = 2

-- Problem 1: Prove sinA = 2/5 given additional condition b = 4
axiom b_val : b = 4

theorem sinA_value (h_triangle : triangle A B C a b c) (h_cosB : cosB = 3/5) (h_a : a = 2) (h_b : b = 4) : 
  ∃ sinA : ℝ, sinA = 2 / 5 :=
sorry

-- Problem 2: Prove b = sqrt(17) and c = 5 given the area
axiom area_val : S_triangle_ABC = 4

theorem find_b_c (h_triangle : triangle A B C a b c) (h_cosB : cosB = 3/5) (h_a : a = 2) (h_area : S_triangle_ABC = 4) : 
  ∃ b c : ℝ, b = Real.sqrt 17 ∧ c = 5 :=
sorry

end sinA_value_find_b_c_l1481_148190


namespace dice_sum_four_l1481_148194

def possible_outcomes (x : Nat) : Set (Nat × Nat) :=
  { (d1, d2) | d1 + d2 = x ∧ 1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 }

theorem dice_sum_four :
  possible_outcomes 4 = {(3, 1), (1, 3), (2, 2)} :=
by
  sorry -- We acknowledge that this outline is equivalent to the provided math problem.

end dice_sum_four_l1481_148194


namespace total_flowering_bulbs_count_l1481_148159

-- Definitions for the problem conditions
def crocus_cost : ℝ := 0.35
def daffodil_cost : ℝ := 0.65
def total_budget : ℝ := 29.15
def crocus_count : ℕ := 22

-- Theorem stating the total number of bulbs that can be bought
theorem total_flowering_bulbs_count : 
  ∃ daffodil_count : ℕ, (crocus_count + daffodil_count = 55) ∧ (total_budget = crocus_cost * crocus_count + daffodil_count * daffodil_cost) :=
  sorry

end total_flowering_bulbs_count_l1481_148159


namespace determine_a_square_binomial_l1481_148129

theorem determine_a_square_binomial (a : ℝ) :
  (∃ r s : ℝ, ∀ x : ℝ, ax^2 + 24*x + 9 = (r*x + s)^2) → a = 16 :=
by
  sorry

end determine_a_square_binomial_l1481_148129


namespace true_prices_for_pie_and_mead_l1481_148125

-- Definitions for true prices
variable (k m : ℕ)

-- Definitions for conditions
def honest_pravdoslav (k m : ℕ) : Prop :=
  4*k = 3*(m + 2) ∧ 4*(m+2) = 3*k + 14

theorem true_prices_for_pie_and_mead (k m : ℕ) (h : honest_pravdoslav k m) : k = 6 ∧ m = 6 := sorry

end true_prices_for_pie_and_mead_l1481_148125


namespace horner_method_evaluation_l1481_148135

def f (x : ℝ) := 0.5 * x^5 + 4 * x^4 + 0 * x^3 - 3 * x^2 + x - 1

theorem horner_method_evaluation : f 3 = 1 :=
by
  -- Placeholder for the proof
  sorry

end horner_method_evaluation_l1481_148135


namespace total_actions_135_l1481_148161

theorem total_actions_135
  (y : ℕ) -- represents the total number of actions
  (h1 : y ≥ 10) -- since there are at least 10 initial comments
  (h2 : ∀ (likes dislikes : ℕ), likes + dislikes = y - 10) -- total votes exclude neutral comments
  (score_eq : ∀ (likes dislikes : ℕ), 70 * dislikes = 30 * likes)
  (score_50 : ∀ (likes dislikes : ℕ), 50 = likes - dislikes) :
  y = 135 :=
by {
  sorry
}

end total_actions_135_l1481_148161


namespace find_values_l1481_148113

theorem find_values (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x - 4 = 21 * (1 / x)) 
  (h2 : x + y^2 = 45) : 
  x = 7 ∧ y = Real.sqrt 38 :=
by
  sorry

end find_values_l1481_148113


namespace parallel_vectors_y_value_l1481_148146

theorem parallel_vectors_y_value (y : ℝ) :
  let a := (2, 3)
  let b := (4, y)
  ∃ y : ℝ, (2 : ℝ) / 4 = 3 / y → y = 6 :=
sorry

end parallel_vectors_y_value_l1481_148146


namespace sign_of_c_l1481_148128

theorem sign_of_c (a b c : ℝ) (h1 : (a * b / c) < 0) (h2 : (a * b) < 0) : c > 0 :=
sorry

end sign_of_c_l1481_148128


namespace find_line_through_and_perpendicular_l1481_148141

def point (x y : ℝ) := (x, y)

def passes_through (P : ℝ × ℝ) (a b c : ℝ) :=
  a * P.1 + b * P.2 + c = 0

def is_perpendicular (a1 b1 a2 b2 : ℝ) :=
  a1 * a2 + b1 * b2 = 0

theorem find_line_through_and_perpendicular :
  ∃ c : ℝ, passes_through (1, -1) 1 1 c ∧ is_perpendicular 1 (-1) 1 1 → 
  c = 0 :=
by
  sorry

end find_line_through_and_perpendicular_l1481_148141


namespace evaluate_f_difference_l1481_148127

def f (x : ℝ) : ℝ := x^6 - 2 * x^4 + 7 * x

theorem evaluate_f_difference :
  f 3 - f (-3) = 42 := by
  sorry

end evaluate_f_difference_l1481_148127


namespace simplify_tangent_expression_l1481_148105

theorem simplify_tangent_expression :
  (1 + Real.tan (Real.pi / 18)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 :=
by
  sorry

end simplify_tangent_expression_l1481_148105


namespace find_missing_dimension_of_carton_l1481_148122

-- Definition of given dimensions and conditions
def carton_length : ℕ := 25
def carton_width : ℕ := 48
def soap_length : ℕ := 8
def soap_width : ℕ := 6
def soap_height : ℕ := 5
def max_soap_boxes : ℕ := 300
def soap_volume : ℕ := soap_length * soap_width * soap_height
def total_carton_volume : ℕ := max_soap_boxes * soap_volume

-- The main statement to prove
theorem find_missing_dimension_of_carton (h : ℕ) (volume_eq : carton_length * carton_width * h = total_carton_volume) : h = 60 :=
sorry

end find_missing_dimension_of_carton_l1481_148122


namespace min_value_is_four_l1481_148119

noncomputable def min_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) : ℝ :=
  (x + y) / (x * y * z)

theorem min_value_is_four (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) :
  min_value x y z h1 h2 h3 h4 = 4 :=
sorry

end min_value_is_four_l1481_148119


namespace decreasing_by_25_l1481_148106

theorem decreasing_by_25 (n : ℕ) (k : ℕ) (y : ℕ) (hy : 0 ≤ y ∧ y < 10^k) : 
  (n = 6 * 10^k + y → n / 10 = y / 25) → (∃ m, n = 625 * 10^m) := 
sorry

end decreasing_by_25_l1481_148106


namespace not_divisible_by_1955_l1481_148170

theorem not_divisible_by_1955 (n : ℤ) : ¬ ∃ k : ℤ, (n^2 + n + 1) = 1955 * k :=
by
  sorry

end not_divisible_by_1955_l1481_148170


namespace team_formation_problem_l1481_148109

def num_team_formation_schemes : Nat :=
  let comb (n k : Nat) : Nat := Nat.choose n k
  (comb 5 1 * comb 4 2) + (comb 5 2 * comb 4 1)

theorem team_formation_problem :
  num_team_formation_schemes = 70 :=
sorry

end team_formation_problem_l1481_148109


namespace nonnegative_integer_solution_count_l1481_148180

theorem nonnegative_integer_solution_count :
  ∃ n : ℕ, (∀ x : ℕ, x^2 + 6 * x = 0 → x = 0) ∧ n = 1 :=
by
  sorry

end nonnegative_integer_solution_count_l1481_148180


namespace farm_horses_cows_l1481_148111

variables (H C : ℕ)

theorem farm_horses_cows (H C : ℕ) (h1 : H = 6 * C) (h2 : (H - 15) = 3 * (C + 15)) : (H - 15) - (C + 15) = 70 :=
by {
  sorry
}

end farm_horses_cows_l1481_148111


namespace phung_more_than_chiu_l1481_148153

theorem phung_more_than_chiu
  (C P H : ℕ)
  (h1 : C = 56)
  (h2 : H = P + 5)
  (h3 : C + P + H = 205) :
  P - C = 16 :=
by
  sorry

end phung_more_than_chiu_l1481_148153


namespace sum_of_x_and_y_l1481_148195

theorem sum_of_x_and_y 
  (x y : ℝ)
  (h : ((x + 1) + (y-1)) / 2 = 10) : x + y = 20 :=
sorry

end sum_of_x_and_y_l1481_148195


namespace petya_max_votes_difference_l1481_148181

theorem petya_max_votes_difference :
  ∃ (P1 P2 V1 V2 : ℕ), 
    P1 = V1 + 9 ∧ 
    V2 = P2 + 9 ∧ 
    P1 + P2 + V1 + V2 = 27 ∧ 
    P1 + P2 > V1 + V2 ∧ 
    (P1 + P2) - (V1 + V2) = 9 := 
by
  sorry

end petya_max_votes_difference_l1481_148181


namespace distinct_positive_integers_solution_l1481_148176

theorem distinct_positive_integers_solution (x y : ℕ) (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h : 1 / x + 1 / y = 2 / 7) : (x = 4 ∧ y = 28) ∨ (x = 28 ∧ y = 4) :=
by
  sorry -- proof to be filled in.

end distinct_positive_integers_solution_l1481_148176


namespace largest_crate_dimension_l1481_148177

def largest_dimension_of_crate : ℝ := 10

theorem largest_crate_dimension (length width : ℝ) (r : ℝ) (h : ℝ) 
  (h_length : length = 5) (h_width : width = 8) (h_radius : r = 5) (h_height : h >= 10) :
  h = largest_dimension_of_crate :=
by 
  sorry

end largest_crate_dimension_l1481_148177


namespace max_value_sequence_l1481_148156

theorem max_value_sequence (a : ℕ → ℝ)
  (h1 : ∀ n : ℕ, a (n + 1) = (-1 : ℝ)^n * n - a n)
  (h2 : a 10 = a 1) :
  ∃ n, a n * a (n + 1) = 33 / 4 :=
sorry

end max_value_sequence_l1481_148156


namespace parallel_lines_slope_l1481_148187

-- Define the given conditions
def line1_slope (x : ℝ) : ℝ := 6
def line2_slope (c : ℝ) (x : ℝ) : ℝ := 3 * c

-- State the proof problem
theorem parallel_lines_slope (c : ℝ) : 
  (∀ x : ℝ, line1_slope x = line2_slope c x) → c = 2 :=
by
  intro h
  -- Intro provides a human-readable variable and corresponding proof obligation
  -- The remainder of the proof would follow here, but instead,
  -- we use "sorry" to indicate an incomplete proof
  sorry

end parallel_lines_slope_l1481_148187


namespace roots_of_polynomial_in_range_l1481_148160

theorem roots_of_polynomial_in_range (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 < -1 ∧ x2 > 1 ∧ x1 * x2 = m^2 - 2 ∧ (x1 + x2) = -(m - 1)) 
  -> 0 < m ∧ m < 1 :=
by
  sorry

end roots_of_polynomial_in_range_l1481_148160


namespace social_media_phone_ratio_l1481_148191

/-- 
Given that Jonathan spends 8 hours on his phone daily and 28 hours on social media in a week, 
prove that the ratio of the time spent on social media to the total time spent on his phone daily is \( 1 : 2 \).
-/
theorem social_media_phone_ratio (daily_phone_hours : ℕ) (weekly_social_media_hours : ℕ) 
  (h1 : daily_phone_hours = 8) (h2 : weekly_social_media_hours = 28) :
  (weekly_social_media_hours / 7) / daily_phone_hours = 1 / 2 := 
by
  sorry

end social_media_phone_ratio_l1481_148191


namespace intersection_M_N_l1481_148175

def I : Set ℤ := {0, -1, -2, -3, -4}
def M : Set ℤ := {0, -1, -2}
def N : Set ℤ := {0, -3, -4}

theorem intersection_M_N : M ∩ N = {0} := 
by 
  sorry

end intersection_M_N_l1481_148175


namespace eval_expression_l1481_148118

theorem eval_expression : 5 + 4 - 3 + 2 - 1 = 7 :=
by
  -- Mathematically, this statement holds by basic arithmetic operations.
  sorry

end eval_expression_l1481_148118


namespace min_value_of_f_l1481_148110

def f (x : ℝ) : ℝ := x^2 - 4 * x + 4

theorem min_value_of_f : ∀ x : ℝ, f x ≥ 0 ∧ f 2 = 0 :=
  by sorry

end min_value_of_f_l1481_148110


namespace find_m_value_l1481_148148

theorem find_m_value (x y m : ℝ) 
  (h1 : 2 * x + y = 5) 
  (h2 : x - 2 * y = m)
  (h3 : 2 * x - 3 * y = 1) : 
  m = 0 := 
sorry

end find_m_value_l1481_148148


namespace max_X_leq_ratio_XY_l1481_148103

theorem max_X_leq_ratio_XY (x y z u : ℕ) (h1 : x + y = z + u) (h2 : 2 * x * y = z * u) (h3 : x ≥ y) : 
  ∃ m, m = 3 + 2 * Real.sqrt 2 ∧ ∀ (x y z u : ℕ), (x + y = z + u) → (2 * x *y = z * u) → (x ≥ y) → m ≤ x / y :=
sorry

end max_X_leq_ratio_XY_l1481_148103


namespace calculate_expression_l1481_148140

noncomputable def expr1 : ℝ := (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3)
noncomputable def expr2 : ℝ := (2 * Real.sqrt 2 - 1) ^ 2
noncomputable def combined_expr : ℝ := expr1 + expr2

-- We need to prove the main statement
theorem calculate_expression : combined_expr = 8 - 4 * Real.sqrt 2 :=
by
  sorry

end calculate_expression_l1481_148140


namespace painted_cells_l1481_148120

open Int

theorem painted_cells : ∀ (m n : ℕ), (m = 20210) → (n = 1505) →
  let sub_rectangles := 215
  let cells_per_diagonal := 100
  let total_cells := sub_rectangles * cells_per_diagonal
  let total_painted_cells := 2 * total_cells
  let overlap_cells := sub_rectangles
  let unique_painted_cells := total_painted_cells - overlap_cells
  unique_painted_cells = 42785 := sorry

end painted_cells_l1481_148120


namespace intersection_complement_l1481_148144

open Set

variable (U A B : Set ℕ)

-- Definitions based on conditions given in the problem
def universal_set : Set ℕ := {1, 2, 3, 4, 5}
def set_A : Set ℕ := {2, 4}
def set_B : Set ℕ := {4, 5}

-- Proof statement
theorem intersection_complement :
  A = set_A → 
  B = set_B → 
  U = universal_set → 
  (A ∩ (U \ B)) = {2} := 
by
  intros hA hB hU
  sorry

end intersection_complement_l1481_148144


namespace polynomial_factor_c_zero_l1481_148115

theorem polynomial_factor_c_zero (c q : ℝ) :
    ∃ q : ℝ, (3*q + 6 = 0 ∧ c = 6*q + 12) ↔ c = 0 :=
by
  sorry

end polynomial_factor_c_zero_l1481_148115


namespace unique_solution_f_eq_x_l1481_148162

theorem unique_solution_f_eq_x (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^2 + y + f y) = 2 * y + f x ^ 2) :
  ∀ x : ℝ, f x = x :=
sorry

end unique_solution_f_eq_x_l1481_148162


namespace union_sets_eq_l1481_148199

-- Definitions of the sets M and N according to the conditions.
def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {x | Real.log x ≤ 0}

-- The theorem we want to prove
theorem union_sets_eq :
  (M ∪ N) = Set.Icc 0 1 :=
by
  sorry

end union_sets_eq_l1481_148199


namespace tina_brownies_per_meal_l1481_148164

-- Define the given conditions
def total_brownies : ℕ := 24
def days : ℕ := 5
def meals_per_day : ℕ := 2
def brownies_by_husband_per_day : ℕ := 1
def total_brownies_shared_with_guests : ℕ := 4
def total_brownies_left : ℕ := 5

-- Conjecture: How many brownies did Tina have with each meal
theorem tina_brownies_per_meal :
  (total_brownies 
  - (brownies_by_husband_per_day * days) 
  - total_brownies_shared_with_guests 
  - total_brownies_left)
  / (days * meals_per_day) = 1 :=
by
  sorry

end tina_brownies_per_meal_l1481_148164


namespace equidistant_xaxis_point_l1481_148158

theorem equidistant_xaxis_point {x : ℝ} :
  (∃ x : ℝ, ∀ A B : ℝ × ℝ, A = (-3, 0) ∧ B = (2, 5) →
    ∀ P : ℝ × ℝ, P = (x, 0) →
      (dist A P = dist B P) → x = 2) := sorry

end equidistant_xaxis_point_l1481_148158


namespace sequence_expression_l1481_148169

theorem sequence_expression (a : ℕ → ℚ)
  (h1 : a 1 = 2 / 3)
  (h2 : ∀ n : ℕ, a (n + 1) = (n / (n + 1)) * a n) :
  ∀ n : ℕ, a n = 2 / (3 * n) :=
sorry

end sequence_expression_l1481_148169


namespace intersection_M_N_l1481_148157

-- Define the sets M and N
def M : Set ℝ := { x : ℝ | x^2 - x - 6 < 0 }
def N : Set ℝ := { x : ℝ | 1 < x }

-- State the problem in terms of Lean definitions and theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end intersection_M_N_l1481_148157


namespace inequality_l1481_148126

-- Given three distinct positive real numbers a, b, c
variables {a b c : ℝ}

-- Assume a, b, and c are distinct and positive
axiom distinct_positive (h: a ≠ b ∧ b ≠ c ∧ a ≠ c) (ha: a > 0) (hb: b > 0) (hc: c > 0) : 
  ∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a > 0 ∧ b > 0 ∧ c > 0

-- The inequality to be proven
theorem inequality (h: a ≠ b ∧ b ≠ c ∧ a ≠ c) (ha: a > 0) (hb: b > 0) (hc: c > 0) :
  (a / b) + (b / c) > (a / c) + (c / a) := 
sorry

end inequality_l1481_148126


namespace value_of_N_l1481_148134

theorem value_of_N (a b c N : ℚ) 
  (h1 : a + b + c = 120)
  (h2 : a + 8 = N)
  (h3 : 8 * b = N)
  (h4 : c / 8 = N) :
  N = 960 / 73 :=
by
  sorry

end value_of_N_l1481_148134


namespace solve_fraction_identity_l1481_148172

theorem solve_fraction_identity (x : ℝ) (hx : (x + 5) / (x - 3) = 4) : x = 17 / 3 :=
by
  sorry

end solve_fraction_identity_l1481_148172


namespace fixed_point_l1481_148185

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2*x - 1)

theorem fixed_point (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) : f a (1/2) = 1 :=
by
  sorry

end fixed_point_l1481_148185


namespace triangle_area_0_0_0_5_7_12_l1481_148101

theorem triangle_area_0_0_0_5_7_12 : 
    let base := 5
    let height := 7
    let area := (1 / 2) * base * height
    area = 17.5 := 
by
    sorry

end triangle_area_0_0_0_5_7_12_l1481_148101


namespace distance_interval_l1481_148189

-- Define the conditions based on the false statements:
variable (d : ℝ)

def false_by_alice : Prop := d < 8
def false_by_bob : Prop := d > 7
def false_by_charlie : Prop := d ≠ 6

theorem distance_interval (h_alice : false_by_alice d) (h_bob : false_by_bob d) (h_charlie : false_by_charlie d) :
  7 < d ∧ d < 8 :=
by
  sorry

end distance_interval_l1481_148189


namespace bread_cost_each_is_3_l1481_148178

-- Define the given conditions
def initial_amount : ℕ := 86
def bread_quantity : ℕ := 3
def orange_juice_quantity : ℕ := 3
def orange_juice_cost_each : ℕ := 6
def remaining_amount : ℕ := 59

-- Define the variable for bread cost
variable (B : ℕ)

-- Lean 4 statement to prove the cost of each loaf of bread
theorem bread_cost_each_is_3 :
  initial_amount - remaining_amount = (bread_quantity * B + orange_juice_quantity * orange_juice_cost_each) →
  B = 3 :=
by
  sorry

end bread_cost_each_is_3_l1481_148178


namespace correct_option_is_A_l1481_148145

def a (n : ℕ) : ℤ :=
  match n with
  | 1 => -3
  | 2 => 7
  | _ => 0  -- This is just a placeholder for other values

def optionA (n : ℕ) : ℤ := (-1)^n * (4*n - 1)
def optionB (n : ℕ) : ℤ := (-1)^n * (4*n + 1)
def optionC (n : ℕ) : ℤ := 4*n - 7
def optionD (n : ℕ) : ℤ := (-1)^(n + 1) * (4*n - 1)

theorem correct_option_is_A :
  (a 1 = -3) ∧ (a 2 = 7) ∧
  (optionA 1 = -3 ∧ optionA 2 = 7) ∧
  ¬(optionB 1 = -3 ∧ optionB 2 = 7) ∧
  ¬(optionC 1 = -3 ∧ optionC 2 = 7) ∧
  ¬(optionD 1 = -3 ∧ optionD 2 = 7) :=
by
  sorry

end correct_option_is_A_l1481_148145


namespace find_multiple_l1481_148149

variable (P W : ℕ)
variable (h1 : ∀ P W, P * 16 * (W / (P * 16)) = W)
variable (h2 : ∀ P W m, (m * P) * 4 * (W / (16 * P)) = W / 2)

theorem find_multiple (P W : ℕ) (h1 : ∀ P W, P * 16 * (W / (P * 16)) = W)
                      (h2 : ∀ P W m, (m * P) * 4 * (W / (16 * P)) = W / 2) : m = 2 :=
by
  sorry

end find_multiple_l1481_148149


namespace max_val_4ab_sqrt3_12bc_l1481_148117

theorem max_val_4ab_sqrt3_12bc {a b c : ℝ} (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a^2 + b^2 + c^2 = 3) :
  4 * a * b * Real.sqrt 3 + 12 * b * c ≤ Real.sqrt 39 :=
sorry

end max_val_4ab_sqrt3_12bc_l1481_148117


namespace no_solution_for_x_l1481_148143

open Real

theorem no_solution_for_x (m : ℝ) : ¬ ∃ x : ℝ, (sin (3 * x) * cos (↑60 - x) + 1) / (sin (↑60 - 7 * x) - cos (↑30 + x) + m) = 0 :=
by
  sorry

end no_solution_for_x_l1481_148143


namespace orvin_max_balloons_l1481_148102

variable (C : ℕ) (P : ℕ)

noncomputable def max_balloons (C P : ℕ) : ℕ :=
  let pair_cost := P + P / 2  -- Cost for two balloons
  let pairs := C / pair_cost  -- Maximum number of pairs
  pairs * 2 + (if C % pair_cost >= P then 1 else 0) -- Total balloons considering the leftover money

theorem orvin_max_balloons (hC : C = 120) (hP : P = 3) : max_balloons C P = 53 :=
by
  sorry

end orvin_max_balloons_l1481_148102


namespace percentage_difference_between_maximum_and_minimum_changes_is_40_l1481_148114

-- Definitions of initial and final survey conditions
def initialYesPercentage : ℝ := 0.40
def initialNoPercentage : ℝ := 0.60
def finalYesPercentage : ℝ := 0.80
def finalNoPercentage : ℝ := 0.20
def absenteePercentage : ℝ := 0.10

-- Main theorem stating the problem
theorem percentage_difference_between_maximum_and_minimum_changes_is_40 :
  let attendeesPercentage := 1 - absenteePercentage
  let adjustedFinalYesPercentage := finalYesPercentage / attendeesPercentage
  let minChange := adjustedFinalYesPercentage - initialYesPercentage
  let maxChange := initialYesPercentage + minChange
  maxChange - minChange = 0.40 :=
by
  -- Proof is omitted
  sorry

end percentage_difference_between_maximum_and_minimum_changes_is_40_l1481_148114


namespace blue_paint_quantity_l1481_148142

-- Conditions
def paint_ratio (r b y w : ℕ) : Prop := r = 2 * w / 4 ∧ b = 3 * w / 4 ∧ y = 1 * w / 4 ∧ w = 4 * (r + b + y + w) / 10

-- Given
def quart_white_paint : ℕ := 16

-- Prove that Victor should use 12 quarts of blue paint
theorem blue_paint_quantity (r b y w : ℕ) (h : paint_ratio r b y w) (hw : w = quart_white_paint) : 
  b = 12 := by
  sorry

end blue_paint_quantity_l1481_148142


namespace cyclist_rate_l1481_148138

theorem cyclist_rate 
  (rate_hiker : ℝ := 4)
  (wait_time_1 : ℝ := 5 / 60)
  (wait_time_2 : ℝ := 10.000000000000002 / 60)
  (hiker_distance : ℝ := rate_hiker * wait_time_2)
  (cyclist_distance : ℝ := hiker_distance)
  (cyclist_rate := cyclist_distance / wait_time_1) :
  cyclist_rate = 8 := by 
sorry

end cyclist_rate_l1481_148138


namespace cost_of_apples_and_bananas_l1481_148108

variable (a b : ℝ) -- Assume a and b are real numbers.

theorem cost_of_apples_and_bananas (a b : ℝ) : 
  (3 * a + 2 * b) = 3 * a + 2 * b :=
by 
  sorry -- Proof placeholder

end cost_of_apples_and_bananas_l1481_148108


namespace boys_count_eq_792_l1481_148104

-- Definitions of conditions
variables (B G : ℤ)

-- Total number of students is 1443
axiom total_students : B + G = 1443

-- Number of girls is 141 fewer than the number of boys
axiom girls_fewer_than_boys : G = B - 141

-- Proof statement to show that the number of boys (B) is 792
theorem boys_count_eq_792 (B G : ℤ)
  (h1 : B + G = 1443)
  (h2 : G = B - 141) : B = 792 :=
by
  sorry

end boys_count_eq_792_l1481_148104


namespace only_n_divides_2_to_n_minus_1_l1481_148173

theorem only_n_divides_2_to_n_minus_1 (n : ℕ) (h1 : n > 0) : n ∣ (2^n - 1) ↔ n = 1 :=
by
  sorry

end only_n_divides_2_to_n_minus_1_l1481_148173


namespace dima_can_find_heavy_ball_l1481_148150

noncomputable def find_heavy_ball
  (balls : Fin 9) -- 9 balls, indexed from 0 to 8 representing the balls 1 to 9
  (heavy : Fin 9) -- One of the balls is heavier
  (weigh : Fin 9 → Fin 9 → Ordering) -- A function that compares two groups of balls and gives an Ordering: .lt, .eq, or .gt
  (predetermined_sets : List (Fin 9 × Fin 9)) -- A list of tuples representing balls on each side for the two weighings
  (valid_sets : predetermined_sets.length ≤ 2) : Prop := -- Not more than two weighings
  ∃ idx : Fin 9, idx = heavy -- Need to prove that we can always find the heavier ball

theorem dima_can_find_heavy_ball :
  ∀ (balls : Fin 9) (heavy : Fin 9)
    (weigh : Fin 9 → Fin 9 → Ordering)
    (predetermined_sets : List (Fin 9 × Fin 9))
    (valid_sets : predetermined_sets.length ≤ 2),
  find_heavy_ball balls heavy weigh predetermined_sets valid_sets :=
sorry -- Proof is omitted

end dima_can_find_heavy_ball_l1481_148150


namespace complement_set_l1481_148139

open Set

variable (U : Set ℝ) (M : Set ℝ)

theorem complement_set :
  U = univ ∧ M = {x | x^2 - 2 * x ≤ 0} → (U \ M) = {x | x < 0 ∨ x > 2} :=
by
  intros
  sorry

end complement_set_l1481_148139


namespace halfway_between_frac_l1481_148179

theorem halfway_between_frac : (1 / 7 + 1 / 9) / 2 = 8 / 63 := by
  sorry

end halfway_between_frac_l1481_148179


namespace quadratic_inequality_iff_abs_a_le_two_l1481_148183

-- Definitions from the condition
variable (a : ℝ)
def quadratic_expr (x : ℝ) : ℝ := x^2 + a * x + 1

-- Statement of the problem as a Lean 4 statement
theorem quadratic_inequality_iff_abs_a_le_two :
  (∀ x : ℝ, quadratic_expr a x ≥ 0) ↔ (|a| ≤ 2) := sorry

end quadratic_inequality_iff_abs_a_le_two_l1481_148183


namespace relay_race_total_time_l1481_148193

-- Definitions based on the problem conditions
def athlete1_time : ℕ := 55
def athlete2_time : ℕ := athlete1_time + 10
def athlete3_time : ℕ := athlete2_time - 15
def athlete4_time : ℕ := athlete1_time - 25

-- Problem statement
theorem relay_race_total_time : 
  athlete1_time + athlete2_time + athlete3_time + athlete4_time = 200 := 
by 
  sorry

end relay_race_total_time_l1481_148193


namespace distinct_names_impossible_l1481_148116

-- Define the alphabet
inductive Letter
| a | u | o | e

-- Simplified form of words in the Mumbo-Jumbo language
def simplified_form : List Letter → List Letter
| [] => []
| (Letter.e :: xs) => simplified_form xs
| (Letter.a :: Letter.a :: Letter.a :: Letter.a :: xs) => simplified_form (Letter.a :: Letter.a :: xs)
| (Letter.o :: Letter.o :: Letter.o :: Letter.o :: xs) => simplified_form xs
| (Letter.a :: Letter.a :: Letter.a :: Letter.u :: xs) => simplified_form (Letter.u :: xs)
| (x :: xs) => x :: simplified_form xs

-- Number of possible names
def num_possible_names : ℕ := 343

-- Number of tribe members
def num_tribe_members : ℕ := 400

theorem distinct_names_impossible : num_possible_names < num_tribe_members :=
by
  -- Skipping the proof with 'sorry'
  sorry

end distinct_names_impossible_l1481_148116


namespace cubics_inequality_l1481_148196

theorem cubics_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) : a^3 + b^3 ≥ a^2 * b + a * b^2 :=
by
  sorry

end cubics_inequality_l1481_148196


namespace total_legs_correct_l1481_148131

variable (a b : ℕ)

def total_legs (a b : ℕ) : ℕ := 2 * a + 4 * b

theorem total_legs_correct (a b : ℕ) : total_legs a b = 2 * a + 4 * b :=
by sorry

end total_legs_correct_l1481_148131


namespace focus_of_parabola_l1481_148155

theorem focus_of_parabola (x y : ℝ) (h : x^2 = -y) : (0, -1/4) = (0, -1/4) :=
by sorry

end focus_of_parabola_l1481_148155


namespace total_candy_count_l1481_148123

def numberOfRedCandies : ℕ := 145
def numberOfBlueCandies : ℕ := 3264
def totalNumberOfCandies : ℕ := numberOfRedCandies + numberOfBlueCandies

theorem total_candy_count :
  totalNumberOfCandies = 3409 :=
by
  unfold totalNumberOfCandies
  unfold numberOfRedCandies
  unfold numberOfBlueCandies
  sorry

end total_candy_count_l1481_148123


namespace volume_of_cone_l1481_148188

theorem volume_of_cone (d : ℝ) (h : ℝ) (r : ℝ) : 
  d = 10 ∧ h = 0.6 * d ∧ r = d / 2 → (1 / 3) * π * r^2 * h = 50 * π :=
by
  intro h1
  rcases h1 with ⟨h_d, h_h, h_r⟩
  sorry

end volume_of_cone_l1481_148188


namespace volume_displacement_square_l1481_148186

-- Define the given conditions
def radius_cylinder := 5
def height_cylinder := 12
def side_length_cube := 10

theorem volume_displacement_square :
  let r := radius_cylinder
  let h := height_cylinder
  let s := side_length_cube
  let cube_diagonal := s * Real.sqrt 3
  let w := (125 * Real.sqrt 6) / 8
  w^2 = 1464.0625 :=
by
  sorry

end volume_displacement_square_l1481_148186


namespace average_disk_space_per_minute_l1481_148168

theorem average_disk_space_per_minute 
  (days : ℕ := 15) 
  (disk_space : ℕ := 36000) 
  (minutes_per_day : ℕ := 1440) 
  (total_minutes := days * minutes_per_day) 
  (average_space_per_minute := disk_space / total_minutes) :
  average_space_per_minute = 2 :=
sorry

end average_disk_space_per_minute_l1481_148168


namespace license_plate_combinations_l1481_148133

theorem license_plate_combinations : 
  let letters := 26 
  let letters_and_digits := 36 
  let middle_character_choices := 2
  3 * letters * letters_and_digits * middle_character_choices = 1872 :=
by
  sorry

end license_plate_combinations_l1481_148133


namespace prove_inequality_l1481_148197

noncomputable def inequality_holds (x y : ℝ) : Prop :=
  x^3 * (y + 1) + y^3 * (x + 1) ≥ x^2 * (y + y^2) + y^2 * (x + x^2)

theorem prove_inequality (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) : inequality_holds x y :=
  sorry

end prove_inequality_l1481_148197


namespace complete_the_square_l1481_148100

theorem complete_the_square :
  ∀ x : ℝ, (x^2 - 2 * x - 2 = 0) → ((x - 1)^2 = 3) :=
by
  intros x h
  sorry

end complete_the_square_l1481_148100


namespace find_X_l1481_148163

theorem find_X (X : ℝ) (h : (X + 200 / 90) * 90 = 18200) : X = 18000 :=
sorry

end find_X_l1481_148163


namespace eval_abs_a_plus_b_l1481_148136

theorem eval_abs_a_plus_b (a b : ℤ) (x : ℤ) 
(h : (7 * x - a) ^ 2 = 49 * x ^ 2 - b * x + 9) : |a + b| = 45 :=
sorry

end eval_abs_a_plus_b_l1481_148136


namespace solution_set_of_inequality_l1481_148130

theorem solution_set_of_inequality (x : ℝ) : (1 / 2 < x ∧ x < 1) ↔ (x / (2 * x - 1) > 1) :=
by { sorry }

end solution_set_of_inequality_l1481_148130


namespace original_price_of_cycle_l1481_148184

noncomputable def original_price_given_gain (SP : ℝ) (gain : ℝ) : ℝ :=
  SP / (1 + gain)

theorem original_price_of_cycle (SP : ℝ) (HSP : SP = 1350) (Hgain : gain = 0.5) : 
  original_price_given_gain SP gain = 900 := 
by
  sorry

end original_price_of_cycle_l1481_148184
