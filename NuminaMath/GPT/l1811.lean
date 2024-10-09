import Mathlib

namespace circle_condition_l1811_181148

-- Define the center of the circle
def center := ((-3 + 27) / 2, (0 + 0) / 2)

-- Define the radius of the circle
def radius := 15

-- Define the circle's equation
def circle_eq (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Define the final Lean 4 statement
theorem circle_condition (x : ℝ) : circle_eq x 12 → (x = 21 ∨ x = 3) :=
  by
  intro h
  -- Proof goes here
  sorry

end circle_condition_l1811_181148


namespace find_extrema_of_S_l1811_181112

theorem find_extrema_of_S (x y z : ℚ) (h1 : 3 * x + 2 * y + z = 5) (h2 : x + y - z = 2) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  2 ≤ 2 * x + y - z ∧ 2 * x + y - z ≤ 3 :=
by
  sorry

end find_extrema_of_S_l1811_181112


namespace star_polygon_net_of_pyramid_l1811_181117

theorem star_polygon_net_of_pyramid (R r : ℝ) (h : R > r) : R > 2 * r :=
by
  sorry

end star_polygon_net_of_pyramid_l1811_181117


namespace sqrt_five_eq_l1811_181123

theorem sqrt_five_eq (m n a b c d : ℤ)
  (h : m + n * Real.sqrt 5 = (a + b * Real.sqrt 5) * (c + d * Real.sqrt 5)) :
  m - n * Real.sqrt 5 = (a - b * Real.sqrt 5) * (c - d * Real.sqrt 5) := by
  sorry

end sqrt_five_eq_l1811_181123


namespace smallest_repeating_block_of_5_over_13_l1811_181163

theorem smallest_repeating_block_of_5_over_13 : 
  ∃ n, n = 6 ∧ (∃ m, (5 / 13 : ℚ) = (m/(10^6) : ℚ) ) := 
sorry

end smallest_repeating_block_of_5_over_13_l1811_181163


namespace problem1_problem2_l1811_181121

-- Problem 1
theorem problem1 {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / a) + (1 / b) + (1 / c) ≥ (1 / (Real.sqrt (a * b))) + (1 / (Real.sqrt (b * c))) + (1 / (Real.sqrt (a * c))) :=
sorry

-- Problem 2
theorem problem2 {x y : ℝ} :
  Real.sin x + Real.sin y ≤ 1 + Real.sin x * Real.sin y :=
sorry

end problem1_problem2_l1811_181121


namespace hyperbola_equation_l1811_181110

noncomputable def hyperbola (a b : ℝ) := ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1

def parabola_focus_same_as_hyperbola_focus (c : ℝ) : Prop :=
  ∃ x y : ℝ, y^2 = 4 * (10:ℝ).sqrt * x ∧ (c, 0) = ((10:ℝ).sqrt, 0)

def hyperbola_eccentricity (c a : ℝ) := (c / a) = (10:ℝ).sqrt / 3

theorem hyperbola_equation :
  ∃ a b : ℝ, (hyperbola a b) ∧
  (parabola_focus_same_as_hyperbola_focus ((10:ℝ).sqrt)) ∧
  (hyperbola_eccentricity ((10:ℝ).sqrt) a) ∧
  ((a = 3) ∧ (b = 1)) :=
sorry

end hyperbola_equation_l1811_181110


namespace quadratic_inverse_condition_l1811_181198

theorem quadratic_inverse_condition : 
  (∀ x₁ x₂ : ℝ, (x₁ ≥ 2 ∧ x₂ ≥ 2 ∧ x₁ ≠ x₂) → (x₁^2 - 4*x₁ + 5 ≠ x₂^2 - 4*x₂ + 5)) :=
sorry

end quadratic_inverse_condition_l1811_181198


namespace student_correct_answers_l1811_181161

variable (C I : ℕ) -- Define C and I as natural numbers
variable (score totalQuestions : ℕ) -- Define score and totalQuestions as natural numbers

-- Define the conditions
def grading_system (C I score : ℕ) : Prop := C - 2 * I = score
def total_questions (C I totalQuestions : ℕ) : Prop := C + I = totalQuestions

-- The theorem statement to prove
theorem student_correct_answers :
  (grading_system C I 76) ∧ (total_questions C I 100) → C = 92 := by
  sorry -- Proof to be filled in

end student_correct_answers_l1811_181161


namespace min_value_of_ab_l1811_181122

theorem min_value_of_ab {a b : ℝ} (ha : a > 0) (hb : b > 0)
    (h : 1 / a + 1 / b = 1) : a + b ≥ 4 :=
sorry

end min_value_of_ab_l1811_181122


namespace weight_of_triangular_piece_l1811_181195

noncomputable def density_factor (weight : ℝ) (area : ℝ) : ℝ :=
  weight / area

noncomputable def square_weight (side_length : ℝ) (weight : ℝ) : ℝ := weight

noncomputable def triangle_area (side_length : ℝ) : ℝ :=
  (side_length ^ 2 * Real.sqrt 3) / 4

theorem weight_of_triangular_piece :
  let side_square := 4
  let weight_square := 16
  let side_triangle := 6
  let area_square := side_square ^ 2
  let area_triangle := triangle_area side_triangle
  let density_square := density_factor weight_square area_square
  let weight_triangle := area_triangle * density_square
  abs weight_triangle - 15.59 < 0.01 :=
by
  sorry

end weight_of_triangular_piece_l1811_181195


namespace remainder_of_sum_of_squares_mod_l1811_181180

-- Define the function to compute the sum of squares of the first n natural numbers
def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

-- Define the specific sum for the first 15 natural numbers
def S : ℕ := sum_of_squares 15

-- State the theorem
theorem remainder_of_sum_of_squares_mod (n : ℕ) (h : n = 15) : 
  S % 13 = 5 := by
  sorry

end remainder_of_sum_of_squares_mod_l1811_181180


namespace tangent_line_eq_range_f_l1811_181145

-- Given the function f(x) = 2x^3 - 9x^2 + 12x
def f(x : ℝ) : ℝ := 2 * x^3 - 9 * x^2 + 12 * x

-- (1) Prove that the equation of the tangent line to y = f(x) at (0, f(0)) is y = 12x
theorem tangent_line_eq : ∀ x, x = 0 → f x = 0 → (∃ m, m = 12 ∧ (∀ y, y = 12 * x)) :=
by
  sorry

-- (2) Prove that the range of f(x) on the interval [0, 3] is [0, 9]
theorem range_f : Set.Icc 0 9 = Set.image f (Set.Icc (0 : ℝ) 3) :=
by
  sorry

end tangent_line_eq_range_f_l1811_181145


namespace triangles_same_base_height_have_equal_areas_l1811_181181

theorem triangles_same_base_height_have_equal_areas 
  (b1 h1 b2 h2 : ℝ) 
  (A1 A2 : ℝ) 
  (h1_nonneg : 0 ≤ h1) 
  (h2_nonneg : 0 ≤ h2) 
  (A1_eq : A1 = b1 * h1 / 2) 
  (A2_eq : A2 = b2 * h2 / 2) :
  (A1 = A2 ↔ b1 * h1 = b2 * h2) ∧ (b1 = b2 ∧ h1 = h2 → A1 = A2) :=
by {
  sorry
}

end triangles_same_base_height_have_equal_areas_l1811_181181


namespace flight_time_NY_to_CT_l1811_181138

def travelTime (start_time_NY : ℕ) (end_time_CT : ℕ) (layover_Johannesburg : ℕ) : ℕ :=
  end_time_CT - start_time_NY + layover_Johannesburg

theorem flight_time_NY_to_CT :
  let start_time_NY := 0 -- 12:00 a.m. Tuesday as 0 hours from midnight in ET
  let end_time_CT := 10  -- 10:00 a.m. Tuesday as 10 hours from midnight in ET
  let layover_Johannesburg := 4
  travelTime start_time_NY end_time_CT layover_Johannesburg = 10 :=
by
  sorry

end flight_time_NY_to_CT_l1811_181138


namespace arithmetic_sequence_sum_l1811_181150

theorem arithmetic_sequence_sum (x y : ℕ) (h₀: ∃ (n : ℕ), x = 3 + n * 4) (h₁: ∃ (m : ℕ), y = 3 + m * 4) (h₂: y = 31 - 4) (h₃: x = y - 4) : x + y = 50 := by
  sorry

end arithmetic_sequence_sum_l1811_181150


namespace geese_left_park_l1811_181170

noncomputable def initial_ducks : ℕ := 25
noncomputable def initial_geese (ducks : ℕ) : ℕ := 2 * ducks - 10
noncomputable def final_ducks (ducks_added : ℕ) (ducks : ℕ) : ℕ := ducks + ducks_added
noncomputable def geese_after_leaving (geese_before : ℕ) (geese_left : ℕ) : ℕ := geese_before - geese_left

theorem geese_left_park
    (ducks : ℕ)
    (ducks_added : ℕ)
    (initial_geese : ℕ := 2 * ducks - 10)
    (final_ducks : ℕ := ducks + ducks_added)
    (geese_left : ℕ)
    (geese_remaining : ℕ := initial_geese - geese_left) :
    geese_remaining = final_ducks + 1 → geese_left = 10 := by
  sorry

end geese_left_park_l1811_181170


namespace perp_bisector_eq_l1811_181154

/-- The circles x^2+y^2=4 and x^2+y^2-4x+6y=0 intersect at points A and B. 
Find the equation of the perpendicular bisector of line segment AB. -/

theorem perp_bisector_eq : 
  let C1 := (0, 0)
  let C2 := (2, -3)
  ∃ (a b c : ℝ), a = 3 ∧ b = 2 ∧ c = 0 ∧ ∀ (x y : ℝ), a * x + b * y + c = 0 := 
by
  sorry

end perp_bisector_eq_l1811_181154


namespace root_abs_sum_l1811_181133

-- Definitions and conditions
variable (p q r n : ℤ)
variable (h_root : (x^3 - 2018 * x + n).coeffs[0] = 0)  -- This needs coefficient definition (simplified for clarity)
variable (h_vieta1 : p + q + r = 0)
variable (h_vieta2 : p * q + q * r + r * p = -2018)

theorem root_abs_sum :
  |p| + |q| + |r| = 100 :=
sorry

end root_abs_sum_l1811_181133


namespace match_proverbs_l1811_181157

-- Define each condition as a Lean definition
def condition1 : Prop :=
"As cold comes and heat goes, the four seasons change" = "Things are developing"

def condition2 : Prop :=
"Thousands of flowers arranged, just waiting for the first thunder" = 
"Decisively seize the opportunity to promote qualitative change"

def condition3 : Prop :=
"Despite the intention to plant flowers, they don't bloom; unintentionally planting willows, they grow into shade" = 
"The unity of contradictions"

def condition4 : Prop :=
"There will be times when the strong winds break the waves, and we will sail across the sea with clouds" = 
"The future is bright"

-- The theorem we need to prove, using the condition definitions
theorem match_proverbs : condition2 ∧ condition4 :=
sorry

end match_proverbs_l1811_181157


namespace max_principals_in_10_years_l1811_181165

theorem max_principals_in_10_years (p : ℕ) (is_principal_term : p = 4) : 
  ∃ n : ℕ, n = 4 ∧ ∀ k : ℕ, (k = 10 → n ≤ 4) :=
by
  sorry

end max_principals_in_10_years_l1811_181165


namespace geometric_series_sum_l1811_181142

theorem geometric_series_sum : 
  (3 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7 + 3^8 + 3^9 + 3^10) = 88572 := 
by 
  sorry

end geometric_series_sum_l1811_181142


namespace set_intersection_complement_l1811_181149

open Set

variable (A B U : Set ℕ)

theorem set_intersection_complement (A B : Set ℕ) (U : Set ℕ) (hU : U = {1, 2, 3, 4})
  (h1 : compl (A ∪ B) = {4}) (h2 : B = {1, 2}) :
  A ∩ compl B = {3} :=
by
  sorry

end set_intersection_complement_l1811_181149


namespace option_d_not_true_l1811_181146

variable (a b : ℝ)
variable (h1 : 0 < a)
variable (h2 : 0 < b)

theorem option_d_not_true : (2 * a * b) / (a + b) ≤ Real.sqrt (a * b) := sorry

end option_d_not_true_l1811_181146


namespace cistern_length_l1811_181162

theorem cistern_length (L : ℝ) (H : 0 < L) :
    (∃ (w d A : ℝ), w = 14 ∧ d = 1.25 ∧ A = 233 ∧ A = L * w + 2 * L * d + 2 * w * d) →
    L = 12 :=
by
  sorry

end cistern_length_l1811_181162


namespace tony_rope_length_l1811_181158

-- Definition of the initial rope lengths and the number of ropes
def ropes : List ℝ := [8, 20, 2, 2, 2, 7]

-- Definition of the length lost per knot
def loss_per_knot : ℝ := 1.2

-- The expected total length of the rope after tying them all together
def total_length_after_knots : ℝ := 35

theorem tony_rope_length : 
  (ropes.sum - (ropes.length - 1) * loss_per_knot) = total_length_after_knots := 
by
  -- define sorry to skip the proof
  sorry

end tony_rope_length_l1811_181158


namespace parallelogram_base_length_l1811_181130

variable (base height : ℝ)
variable (Area : ℝ)

theorem parallelogram_base_length (h₁ : Area = 162) (h₂ : height = 2 * base) (h₃ : Area = base * height) : base = 9 := 
by
  sorry

end parallelogram_base_length_l1811_181130


namespace find_m_value_l1811_181113

theorem find_m_value (a m : ℤ) (h : a ≠ 1) (hx : ∀ x y : ℤ, (x > 0) ∧ (y > 0) ∧ (a - 1) * x^2 - m * x + a = 0 ∧ (a - 1) * y^2 - m * y + a = 0) : m = 3 :=
sorry

end find_m_value_l1811_181113


namespace largest_integer_l1811_181105

def bin_op (n : ℤ) : ℤ := n - 5 * n

theorem largest_integer (n : ℤ) (h : 0 < n) (h' : bin_op n < 18) : n = 4 := sorry

end largest_integer_l1811_181105


namespace smallest_positive_z_l1811_181199

theorem smallest_positive_z (x z : ℝ) (hx : Real.sin x = 1) (hz : Real.sin (x + z) = -1/2) : z = 2 * Real.pi / 3 :=
by
  sorry

end smallest_positive_z_l1811_181199


namespace share_money_3_people_l1811_181175

theorem share_money_3_people (total_money : ℝ) (amount_per_person : ℝ) (h1 : total_money = 3.75) (h2 : amount_per_person = 1.25) : 
  total_money / amount_per_person = 3 := by
  sorry

end share_money_3_people_l1811_181175


namespace louise_needs_eight_boxes_l1811_181169

-- Define the given conditions
def red_pencils : ℕ := 20
def blue_pencils : ℕ := 2 * red_pencils
def yellow_pencils : ℕ := 40
def green_pencils : ℕ := red_pencils + blue_pencils
def pencils_per_box : ℕ := 20

-- Define the functions to calculate the required number of boxes for each color
def boxes_needed (pencils : ℕ) : ℕ := (pencils + pencils_per_box - 1) / pencils_per_box

-- Calculate the total number of boxes needed by summing the boxes for each color
def total_boxes_needed : ℕ := boxes_needed red_pencils + boxes_needed blue_pencils + boxes_needed yellow_pencils + boxes_needed green_pencils

-- The proof problem statement
theorem louise_needs_eight_boxes : total_boxes_needed = 8 :=
by
  sorry

end louise_needs_eight_boxes_l1811_181169


namespace percentage_of_Y_salary_l1811_181155

variable (X Y : ℝ)
variable (total_salary Y_salary : ℝ)
variable (P : ℝ)

theorem percentage_of_Y_salary :
  total_salary = 638 ∧ Y_salary = 290 ∧ X = (P / 100) * Y_salary → P = 120 := by
  sorry

end percentage_of_Y_salary_l1811_181155


namespace empty_solution_set_range_l1811_181103

theorem empty_solution_set_range (m : ℝ) : 
  (¬ ∃ x : ℝ, (m * x^2 + 2 * m * x + 1) < 0) ↔ (m = 0 ∨ (0 < m ∧ m ≤ 1)) :=
by sorry

end empty_solution_set_range_l1811_181103


namespace nine_pow_1000_mod_13_l1811_181100

theorem nine_pow_1000_mod_13 :
  (9^1000) % 13 = 9 :=
by
  have h1 : 9^1 % 13 = 9 := by sorry
  have h2 : 9^2 % 13 = 3 := by sorry
  have h3 : 9^3 % 13 = 1 := by sorry
  have cycle : ∀ n, 9^(3 * n + 1) % 13 = 9 := by sorry
  exact (cycle 333)

end nine_pow_1000_mod_13_l1811_181100


namespace union_P_Q_l1811_181187

-- Definition of sets P and Q
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 4 }
def Q : Set ℝ := { x | -3 < x ∧ x < 3 }

-- Statement to prove
theorem union_P_Q :
  P ∪ Q = { x : ℝ | -3 < x ∧ x ≤ 4 } :=
sorry

end union_P_Q_l1811_181187


namespace man_speed_with_stream_l1811_181129

-- Define the man's rate in still water
def man_rate_in_still_water : ℝ := 6

-- Define the man's rate against the stream
def man_rate_against_stream (stream_speed : ℝ) : ℝ :=
  man_rate_in_still_water - stream_speed

-- The given condition that the man's rate against the stream is 10 km/h
def man_rate_against_condition : Prop := ∃ (stream_speed : ℝ), man_rate_against_stream stream_speed = 10

-- We aim to prove that the man's speed with the stream is 10 km/h
theorem man_speed_with_stream (stream_speed : ℝ) (h : man_rate_against_stream stream_speed = 10) :
  man_rate_in_still_water + stream_speed = 10 := by
  sorry

end man_speed_with_stream_l1811_181129


namespace marbles_count_l1811_181156

def num_violet_marbles := 64

def num_red_marbles := 14

def total_marbles (violet : Nat) (red : Nat) : Nat :=
  violet + red

theorem marbles_count :
  total_marbles num_violet_marbles num_red_marbles = 78 := by
  sorry

end marbles_count_l1811_181156


namespace total_height_of_buildings_l1811_181141

noncomputable def tallest_building := 100
noncomputable def second_tallest_building := tallest_building / 2
noncomputable def third_tallest_building := second_tallest_building / 2
noncomputable def fourth_tallest_building := third_tallest_building / 5

theorem total_height_of_buildings : 
  (tallest_building + second_tallest_building + third_tallest_building + fourth_tallest_building) = 180 := by
  sorry

end total_height_of_buildings_l1811_181141


namespace sin_cos_105_l1811_181127

theorem sin_cos_105 (h1 : ∀ x : ℝ, Real.sin x * Real.cos x = 1 / 2 * Real.sin (2 * x))
                    (h2 : ∀ x : ℝ, Real.sin (180 * Real.pi / 180 + x) = - Real.sin x)
                    (h3 : Real.sin (30 * Real.pi / 180) = 1 / 2) :
  Real.sin (105 * Real.pi / 180) * Real.cos (105 * Real.pi / 180) = - 1 / 4 :=
by
  sorry

end sin_cos_105_l1811_181127


namespace F_final_coordinates_l1811_181139

-- Define the original coordinates of point F
def F : ℝ × ℝ := (5, 2)

-- Reflection over the y-axis changes the sign of the x-coordinate
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Reflection over the line y = x involves swapping x and y coordinates
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- The combined transformation: reflect over the y-axis, then reflect over y = x
def F_final : ℝ × ℝ := reflect_y_eq_x (reflect_y_axis F)

-- The proof statement
theorem F_final_coordinates : F_final = (2, -5) :=
by
  -- Proof goes here
  sorry

end F_final_coordinates_l1811_181139


namespace compare_game_A_and_C_l1811_181192

-- Probability definitions for coin toss
def p_heads := 2/3
def p_tails := 1/3

-- Probability of winning Game A
def prob_win_A := (p_heads^3) + (p_tails^3)

-- Probability of winning Game C
def prob_win_C := (p_heads^3 + p_tails^3)^2

-- Theorem statement to compare chances of winning Game A to Game C
theorem compare_game_A_and_C : prob_win_A - prob_win_C = 2/9 := by sorry

end compare_game_A_and_C_l1811_181192


namespace find_natural_numbers_eq_36_sum_of_digits_l1811_181107

-- Define the sum of digits function
def sum_of_digits (x : ℕ) : ℕ := 
  if x = 0 then 0
  else sum_of_digits (x / 10) + (x % 10)

-- Lean theorem statement proving the given problem
theorem find_natural_numbers_eq_36_sum_of_digits :
  {x : ℕ | x = 36 * (sum_of_digits x)} = {324, 648} :=
sorry

end find_natural_numbers_eq_36_sum_of_digits_l1811_181107


namespace exists_natural_n_l1811_181135

theorem exists_natural_n (a b : ℕ) (h1 : b ≥ 2) (h2 : Nat.gcd a b = 1) : ∃ n : ℕ, (n * a) % b = 1 :=
by
  sorry

end exists_natural_n_l1811_181135


namespace danny_steve_ratio_l1811_181173

theorem danny_steve_ratio :
  ∀ (D S : ℝ),
  D = 29 →
  2 * (S / 2 - D / 2) = 29 →
  D / S = 1 / 2 :=
by
  intros D S hD h_eq
  sorry

end danny_steve_ratio_l1811_181173


namespace intersection_M_N_l1811_181153

variable (x : ℝ)

def M := {x : ℝ | -2 < x ∧ x < 2}
def N := {x : ℝ | x > 1}

theorem intersection_M_N :
  {x | x ∈ M ∧ x ∈ N} = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end intersection_M_N_l1811_181153


namespace no_real_solution_for_x_l1811_181166

theorem no_real_solution_for_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 8) (h2 : y + 1 / x = 7 / 20) : false :=
by sorry

end no_real_solution_for_x_l1811_181166


namespace inequality_solution_set_l1811_181119

theorem inequality_solution_set :
  { x : ℝ | 1 < x ∧ x < 2 } = { x : ℝ | (x - 2) / (1 - x) > 0 } :=
by sorry

end inequality_solution_set_l1811_181119


namespace sum_zero_l1811_181115

variable {a b c d : ℝ}

-- Pairwise distinct real numbers
axiom h1 : a ≠ b
axiom h2 : a ≠ c
axiom h3 : a ≠ d
axiom h4 : b ≠ c
axiom h5 : b ≠ d
axiom h6 : c ≠ d

-- Given condition
axiom h : (a^2 + b^2 - 1) * (a + b) = (b^2 + c^2 - 1) * (b + c) ∧ 
          (b^2 + c^2 - 1) * (b + c) = (c^2 + d^2 - 1) * (c + d)

theorem sum_zero : a + b + c + d = 0 :=
sorry

end sum_zero_l1811_181115


namespace initial_average_weight_l1811_181178

theorem initial_average_weight (A : ℝ) (weight7th : ℝ) (new_avg_weight : ℝ) (initial_num : ℝ) (total_num : ℝ) 
  (h_weight7th : weight7th = 97) (h_new_avg_weight : new_avg_weight = 151) (h_initial_num : initial_num = 6) (h_total_num : total_num = 7) :
  initial_num * A + weight7th = total_num * new_avg_weight → A = 160 := 
by 
  intros h
  sorry

end initial_average_weight_l1811_181178


namespace doris_hourly_wage_l1811_181191

-- Defining the conditions from the problem
def money_needed : ℕ := 1200
def weekday_hours_per_day : ℕ := 3
def saturday_hours_per_day : ℕ := 5
def weeks_needed : ℕ := 3
def weekdays_per_week : ℕ := 5
def saturdays_per_week : ℕ := 1

-- Calculating total hours worked by Doris in 3 weeks
def total_hours (w_hours: ℕ) (s_hours: ℕ) 
    (w_days : ℕ) (s_days : ℕ) (weeks : ℕ) : ℕ := 
    (w_days * w_hours + s_days * s_hours) * weeks

-- Defining the weekly work hours
def weekly_hours := total_hours weekday_hours_per_day saturday_hours_per_day weekdays_per_week saturdays_per_week 1

-- Result of hours worked in 3 weeks
def hours_worked_in_3_weeks := weekly_hours * weeks_needed

-- Define the proof task
theorem doris_hourly_wage : 
  (money_needed : ℕ) / (hours_worked_in_3_weeks : ℕ) = 20 := by 
  sorry

end doris_hourly_wage_l1811_181191


namespace task2_probability_l1811_181176

variable (P_task1_on_time P_task2_on_time : ℝ)

theorem task2_probability 
  (h1 : P_task1_on_time = 5 / 8)
  (h2 : (P_task1_on_time * (1 - P_task2_on_time)) = 0.25) :
  P_task2_on_time = 3 / 5 := by
  sorry

end task2_probability_l1811_181176


namespace chord_length_of_circle_and_line_intersection_l1811_181160

theorem chord_length_of_circle_and_line_intersection :
  ∀ (x y : ℝ), (x - 2 * y = 3) → ((x - 2)^2 + (y + 3)^2 = 9) → ∃ chord_length : ℝ, (chord_length = 4) :=
by
  intros x y hx hy
  sorry

end chord_length_of_circle_and_line_intersection_l1811_181160


namespace intersection_M_N_l1811_181177

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l1811_181177


namespace two_digit_remainder_one_when_divided_by_4_and_17_l1811_181194

-- Given the conditions
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def yields_remainder (n d r : ℕ) : Prop := n % d = r

-- Define the main problem that checks if there is only one such number
theorem two_digit_remainder_one_when_divided_by_4_and_17 :
  ∃! n : ℕ, is_two_digit n ∧ yields_remainder n 4 1 ∧ yields_remainder n 17 1 :=
sorry

end two_digit_remainder_one_when_divided_by_4_and_17_l1811_181194


namespace farmer_land_l1811_181136

variable (T : ℝ) -- Total land owned by the farmer

def is_cleared (T : ℝ) : ℝ := 0.90 * T
def cleared_barley (T : ℝ) : ℝ := 0.80 * is_cleared T
def cleared_potato (T : ℝ) : ℝ := 0.10 * is_cleared T
def cleared_tomato : ℝ := 90
def cleared_land (T : ℝ) : ℝ := cleared_barley T + cleared_potato T + cleared_tomato

theorem farmer_land (T : ℝ) (h : cleared_land T = is_cleared T) : T = 1000 := sorry

end farmer_land_l1811_181136


namespace scientific_notation_l1811_181144

theorem scientific_notation : (0.000000005 : ℝ) = 5 * 10^(-9 : ℤ) := 
by
  sorry

end scientific_notation_l1811_181144


namespace students_neither_cs_nor_elec_l1811_181193

theorem students_neither_cs_nor_elec
  (total_students : ℕ)
  (cs_students : ℕ)
  (elec_students : ℕ)
  (both_cs_and_elec : ℕ)
  (h_total : total_students = 150)
  (h_cs : cs_students = 90)
  (h_elec : elec_students = 60)
  (h_both : both_cs_and_elec = 20) :
  (total_students - (cs_students + elec_students - both_cs_and_elec) = 20) :=
by
  sorry

end students_neither_cs_nor_elec_l1811_181193


namespace find_f_value_l1811_181101

-- Condition 1: f is an odd function
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Condition 2: f has a period of 5
def is_periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x

-- Condition 3: f(-3) = -4
def f_value_at_neg3 (f : ℝ → ℝ) := f (-3) = -4

-- Condition 4: cos(α) = 1 / 2
def cos_alpha_value (α : ℝ) := Real.cos α = 1 / 2

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def α : ℝ := sorry

theorem find_f_value (h_odd : is_odd_function f)
                     (h_periodic : is_periodic f 5)
                     (h_f_neg3 : f_value_at_neg3 f)
                     (h_cos_alpha : cos_alpha_value α) :
  f (4 * Real.cos (2 * α)) = 4 := 
sorry

end find_f_value_l1811_181101


namespace bianca_total_bags_l1811_181182

theorem bianca_total_bags (bags_recycled_points : ℕ) (bags_not_recycled : ℕ) (total_points : ℕ) (total_bags : ℕ) 
  (h1 : bags_recycled_points = 5) 
  (h2 : bags_not_recycled = 8) 
  (h3 : total_points = 45) 
  (recycled_bags := total_points / bags_recycled_points) :
  total_bags = recycled_bags + bags_not_recycled := 
by 
  sorry

end bianca_total_bags_l1811_181182


namespace corrected_average_l1811_181189

theorem corrected_average (incorrect_avg : ℕ) (correct_val incorrect_val number_of_values : ℕ) (avg := 17) (n := 10) (inc := 26) (cor := 56) :
  incorrect_avg = 17 →
  number_of_values = 10 →
  correct_val = 56 →
  incorrect_val = 26 →
  correct_avg = (incorrect_avg * number_of_values + (correct_val - incorrect_val)) / number_of_values →
  correct_avg = 20 := by
  sorry

end corrected_average_l1811_181189


namespace man_age_twice_son_age_in_n_years_l1811_181152

theorem man_age_twice_son_age_in_n_years
  (S M Y : ℤ)
  (h1 : S = 26)
  (h2 : M = S + 28)
  (h3 : M + Y = 2 * (S + Y)) :
  Y = 2 :=
by
  sorry

end man_age_twice_son_age_in_n_years_l1811_181152


namespace unique_positive_b_solution_exists_l1811_181108

theorem unique_positive_b_solution_exists (c : ℝ) (k : ℝ) :
  (∃b : ℝ, b > 0 ∧ ∀x : ℝ, x^2 + (b + 1/b) * x + c = 0 → x = 0) ∧
  (∀b : ℝ, b^4 + (2 - 4 * c) * b^2 + k = 0) → c = 1 :=
by
  sorry

end unique_positive_b_solution_exists_l1811_181108


namespace sqrt_expr_is_integer_l1811_181131

theorem sqrt_expr_is_integer (x : ℤ) (n : ℤ) (h : n^2 = x^2 - x + 1) : x = 0 ∨ x = 1 := by
  sorry

end sqrt_expr_is_integer_l1811_181131


namespace spent_on_veggies_l1811_181111

noncomputable def total_amount : ℕ := 167
noncomputable def spent_on_meat : ℕ := 17
noncomputable def spent_on_chicken : ℕ := 22
noncomputable def spent_on_eggs : ℕ := 5
noncomputable def spent_on_dog_food : ℕ := 45
noncomputable def amount_left : ℕ := 35

theorem spent_on_veggies : 
  total_amount - (spent_on_meat + spent_on_chicken + spent_on_eggs + spent_on_dog_food + amount_left) = 43 := 
by 
  sorry

end spent_on_veggies_l1811_181111


namespace seeds_per_flowerbed_l1811_181106

theorem seeds_per_flowerbed (total_seeds flowerbeds : ℕ) (h1 : total_seeds = 32) (h2 : flowerbeds = 8) :
  total_seeds / flowerbeds = 4 :=
by {
  sorry
}

end seeds_per_flowerbed_l1811_181106


namespace problem_proof_l1811_181114

theorem problem_proof (x y : ℝ) (h : x / (2 * y) = 3 / 2) : (7 * x + 2 * y) / (x - 2 * y) = 23 :=
by sorry

end problem_proof_l1811_181114


namespace part1_part2_l1811_181128

variable {a b c : ℝ}

theorem part1 (hpos : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a + b + c = 1) : 
  a * b + b * c + a * c ≤ 1 / 3 := 
sorry 

theorem part2 (hpos : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a + b + c = 1) : 
  a^2 / b + b^2 / c + c^2 / a ≥ 1 := 
sorry

end part1_part2_l1811_181128


namespace ratio_boys_to_girls_l1811_181185

theorem ratio_boys_to_girls (total_students girls : ℕ) (h1 : total_students = 455) (h2 : girls = 175) :
  let boys := total_students - girls
  (boys : ℕ) / Nat.gcd boys girls = 8 / 1 ∧ (girls : ℕ) / Nat.gcd boys girls = 5 / 1 :=
by
  sorry

end ratio_boys_to_girls_l1811_181185


namespace tom_spent_on_videogames_l1811_181151

theorem tom_spent_on_videogames (batman_game superman_game : ℝ) 
  (h1 : batman_game = 13.60) 
  (h2 : superman_game = 5.06) : 
  batman_game + superman_game = 18.66 :=
by 
  sorry

end tom_spent_on_videogames_l1811_181151


namespace chili_pepper_cost_l1811_181188

theorem chili_pepper_cost :
  ∃ x : ℝ, 
    (3 * 2.50 + 4 * 1.50 + 5 * x = 18) ∧ 
    x = 0.90 :=
by
  use 0.90
  sorry

end chili_pepper_cost_l1811_181188


namespace vector_perpendicular_l1811_181120

open Real

theorem vector_perpendicular (t : ℝ) (a b : ℝ × ℝ) (h_a : a = (1, 2)) (h_b : b = (4, 3)) :
  a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0 ↔ t = -2 := by
  sorry

end vector_perpendicular_l1811_181120


namespace ordered_pair_arith_progression_l1811_181132

/-- 
Suppose (a, b) is an ordered pair of integers such that the three numbers a, b, and ab 
form an arithmetic progression, in that order. Prove the sum of all possible values of a is 8.
-/
theorem ordered_pair_arith_progression (a b : ℤ) (h : ∃ (a b : ℤ), (b - a = ab - b)) : 
  (a = 0 ∨ a = 1 ∨ a = 3 ∨ a = 4) → a + (if a = 0 then 1 else 0) + 
  (if a = 1 then 1 else 0) + (if a = 3 then 3 else 0) + (if a = 4 then 4 else 0) = 8 :=
by
  sorry

end ordered_pair_arith_progression_l1811_181132


namespace quadrilateral_iff_segments_lt_half_l1811_181186

theorem quadrilateral_iff_segments_lt_half (a b c d : ℝ) (h₁ : a + b + c + d = 1) (h₂ : a ≤ b) (h₃ : b ≤ c) (h₄ : c ≤ d) : 
    (a + b > d) ∧ (a + c > d) ∧ (a + b + c > d) ∧ (b + c > d) ↔ a < 1/2 ∧ b < 1/2 ∧ c < 1/2 ∧ d < 1/2 :=
by
  sorry

end quadrilateral_iff_segments_lt_half_l1811_181186


namespace locus_of_points_equidistant_from_axes_l1811_181140

-- Define the notion of being equidistant from the x-axis and the y-axis
def is_equidistant_from_axes (P : (ℝ × ℝ)) : Prop :=
  abs P.1 = abs P.2

-- The proof problem: given a moving point, the locus equation when P is equidistant from both axes
theorem locus_of_points_equidistant_from_axes (x y : ℝ) :
  is_equidistant_from_axes (x, y) → abs x - abs y = 0 :=
by
  intros h
  exact sorry

end locus_of_points_equidistant_from_axes_l1811_181140


namespace Albaszu_machine_productivity_l1811_181197

theorem Albaszu_machine_productivity (x : ℝ) 
  (h1 : 1.5 * x = 25) : x = 16 := 
by 
  sorry

end Albaszu_machine_productivity_l1811_181197


namespace det_example_1_simplified_form_det_at_4_l1811_181137

-- Definition for second-order determinant
def second_order_determinant (a b c d : ℤ) : ℤ :=
  a * d - b * c

-- Part (1)
theorem det_example_1 :
  second_order_determinant 3 (-2) 4 (-3) = -1 :=
by
  sorry

-- Part (2) simplified determinant
def simplified_det (x : ℤ) : ℤ :=
  second_order_determinant (2 * x - 3) (x + 2) 2 4

-- Proving simplified determinant form
theorem simplified_form :
  ∀ x : ℤ, simplified_det x = 6 * x - 16 :=
by
  sorry

-- Proving specific case when x = 4
theorem det_at_4 :
  simplified_det 4 = 8 :=
by 
  sorry

end det_example_1_simplified_form_det_at_4_l1811_181137


namespace customer_buys_two_pens_l1811_181174

def num_pens (total_pens non_defective_pens : Nat) (prob : ℚ) : Nat :=
  sorry

theorem customer_buys_two_pens :
  num_pens 16 13 0.65 = 2 :=
sorry

end customer_buys_two_pens_l1811_181174


namespace infinite_sqrt_solution_l1811_181172

noncomputable def infinite_sqrt (x : ℝ) : ℝ := Real.sqrt (20 + x)

theorem infinite_sqrt_solution : 
  ∃ x : ℝ, infinite_sqrt x = x ∧ x ≥ 0 ∧ x = 5 :=
by
  sorry

end infinite_sqrt_solution_l1811_181172


namespace inequality_solution_range_l1811_181190

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x ^ 2 - 8 * x - 4 - a > 0) → a < -4 :=
by
  sorry

end inequality_solution_range_l1811_181190


namespace tan_mul_tan_l1811_181168

variables {α β : ℝ}

theorem tan_mul_tan (h : 3 * Real.cos (2 * α + β) + 5 * Real.cos β = 0) : 
  Real.tan (α + β) * Real.tan α = -4 :=
sorry

end tan_mul_tan_l1811_181168


namespace equality_of_ha_l1811_181104

theorem equality_of_ha 
  {p a b α β γ : ℝ} 
  (h1 : h_a = (2 * (p - a) * Real.cos (β / 2) * Real.cos (γ / 2)) / Real.cos (α / 2))
  (h2 : h_a = (2 * (p - b) * Real.sin (β / 2) * Real.cos (γ / 2)) / Real.sin (α / 2)) : 
  (2 * (p - a) * Real.cos (β / 2) * Real.cos (γ / 2)) / Real.cos (α / 2) = 
  (2 * (p - b) * Real.sin (β / 2) * Real.cos (γ / 2)) / Real.sin (α / 2) :=
by sorry

end equality_of_ha_l1811_181104


namespace inequality_abc_l1811_181167

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b))) ≥ 3 / 2 := 
sorry

end inequality_abc_l1811_181167


namespace total_jumps_l1811_181147

theorem total_jumps (hattie_1 : ℕ) (lorelei_1 : ℕ) (hattie_2 : ℕ) (lorelei_2 : ℕ) (hattie_3 : ℕ) (lorelei_3 : ℕ) :
  hattie_1 = 180 →
  lorelei_1 = 3 / 4 * hattie_1 →
  hattie_2 = 2 / 3 * hattie_1 →
  lorelei_2 = hattie_2 + 50 →
  hattie_3 = hattie_2 + 1 / 3 * hattie_2 →
  lorelei_3 = 4 / 5 * lorelei_1 →
  hattie_1 + hattie_2 + hattie_3 + lorelei_1 + lorelei_2 + lorelei_3 = 873 :=
by
  intros h1 l1 h2 l2 h3 l3
  sorry

end total_jumps_l1811_181147


namespace lcm_135_468_l1811_181164

theorem lcm_135_468 : Nat.lcm 135 468 = 7020 := by
  sorry

end lcm_135_468_l1811_181164


namespace find_Y_l1811_181125

theorem find_Y (Y : ℕ) 
  (h_top : 2 + 1 + Y + 3 = 6 + Y)
  (h_bottom : 4 + 3 + 1 + 5 = 13)
  (h_equal : 6 + Y = 13) : 
  Y = 7 := 
by
  sorry

end find_Y_l1811_181125


namespace least_possible_value_of_smallest_integer_l1811_181159

theorem least_possible_value_of_smallest_integer :
  ∀ (A B C D : ℕ), A < B → B < C → C < D → (A + B + C + D) / 4 = 70 → D = 90 → A ≥ 13 :=
by
  intros A B C D h₁ h₂ h₃ h₄ h₅
  sorry

end least_possible_value_of_smallest_integer_l1811_181159


namespace ratio_depth_to_height_l1811_181143

noncomputable def height_ron : ℝ := 12
noncomputable def depth_water : ℝ := 60

theorem ratio_depth_to_height : depth_water / height_ron = 5 := by
  sorry

end ratio_depth_to_height_l1811_181143


namespace Dexter_card_count_l1811_181134

theorem Dexter_card_count : 
  let basketball_boxes := 9
  let cards_per_basketball_box := 15
  let football_boxes := basketball_boxes - 3
  let cards_per_football_box := 20
  let basketball_cards := basketball_boxes * cards_per_basketball_box
  let football_cards := football_boxes * cards_per_football_box
  let total_cards := basketball_cards + football_cards
  total_cards = 255 :=
sorry

end Dexter_card_count_l1811_181134


namespace value_of_x4_plus_inv_x4_l1811_181179

theorem value_of_x4_plus_inv_x4 (x : ℝ) (h : x^2 + 1 / x^2 = 6) : x^4 + 1 / x^4 = 34 := 
by
  sorry

end value_of_x4_plus_inv_x4_l1811_181179


namespace ratio_of_triangle_areas_l1811_181102

-- Define the given conditions
variables (m n x a : ℝ) (S T1 T2 : ℝ)

-- Conditions
def area_of_square : Prop := S = x^2
def area_of_triangle_1 : Prop := T1 = m * x^2
def length_relation : Prop := x = n * a

-- The proof goal
theorem ratio_of_triangle_areas (h1 : area_of_square S x) 
                                (h2 : area_of_triangle_1 T1 m x)
                                (h3 : length_relation x n a) : 
                                T2 / S = m / n^2 := 
sorry

end ratio_of_triangle_areas_l1811_181102


namespace painted_rooms_l1811_181109

def total_rooms : ℕ := 12
def hours_per_room : ℕ := 7
def remaining_hours : ℕ := 49

theorem painted_rooms : total_rooms - (remaining_hours / hours_per_room) = 5 := by
  sorry

end painted_rooms_l1811_181109


namespace geometric_triangle_condition_right_geometric_triangle_condition_l1811_181126

-- Definitions for the geometric progression
def geometric_sequence (a b c q : ℝ) : Prop :=
  b = a * q ∧ c = a * q^2

-- Conditions for forming a triangle
def forms_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Conditions for forming a right triangle using Pythagorean theorem
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem geometric_triangle_condition (a q : ℝ) (h1 : 1 ≤ q) (h2 : q < (1 + Real.sqrt 5) / 2) :
  ∃ (b c : ℝ), geometric_sequence a b c q ∧ forms_triangle a b c := 
sorry

theorem right_geometric_triangle_condition (a q : ℝ) :
  q = Real.sqrt ((1 + Real.sqrt 5) / 2) →
  ∃ (b c : ℝ), geometric_sequence a b c q ∧ right_triangle a b c :=
sorry

end geometric_triangle_condition_right_geometric_triangle_condition_l1811_181126


namespace marie_erasers_l1811_181124

theorem marie_erasers (initial_erasers : ℕ) (lost_erasers : ℕ) (final_erasers : ℕ) 
  (h1 : initial_erasers = 95) (h2 : lost_erasers = 42) : final_erasers = 53 :=
by
  sorry

end marie_erasers_l1811_181124


namespace rectangle_ratio_expression_value_l1811_181116

theorem rectangle_ratio_expression_value (l w : ℝ) (S : ℝ) (h1 : l / w = (2 * (l + w)) / (2 * l)) (h2 : S = w / l) :
  S ^ (S ^ (S^2 + 1/S) + 1/S) + 1/S = Real.sqrt 5 :=
by
  sorry

end rectangle_ratio_expression_value_l1811_181116


namespace courtyard_length_eq_40_l1811_181118

/-- Defining the dimensions of a paving stone -/
def stone_length : ℝ := 4
def stone_width : ℝ := 2

/-- Defining the width of the courtyard -/
def courtyard_width : ℝ := 20

/-- Number of paving stones used -/
def num_stones : ℝ := 100

/-- Area covered by one paving stone -/
def stone_area : ℝ := stone_length * stone_width

/-- Total area covered by the paving stones -/
def total_area : ℝ := num_stones * stone_area

/-- The main statement to be proved -/
theorem courtyard_length_eq_40 (h1 : total_area = num_stones * stone_area)
(h2 : total_area = 800)
(h3 : courtyard_width = 20) : total_area / courtyard_width = 40 :=
by sorry

end courtyard_length_eq_40_l1811_181118


namespace problem_r_minus_s_l1811_181196

theorem problem_r_minus_s (r s : ℝ) (h1 : r ≠ s) (h2 : ∀ x : ℝ, (6 * x - 18) / (x ^ 2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) (h3 : r > s) : r - s = 3 :=
by
  sorry

end problem_r_minus_s_l1811_181196


namespace line_through_two_points_line_with_intercept_sum_l1811_181183

theorem line_through_two_points (a b x1 y1 x2 y2: ℝ) : 
  (x1 = 2) → (y1 = 1) → (x2 = 0) → (y2 = -3) → (2 * x - y - 3 = 0) :=
by
                
  sorry

theorem line_with_intercept_sum (a b : ℝ) (x y : ℝ) :
  (x = 0) → (y = 5) → (a + b = 2) → (b = 5) → (5 * x - 3 * y + 15 = 0) :=
by
  sorry

end line_through_two_points_line_with_intercept_sum_l1811_181183


namespace correct_conclusion_l1811_181184

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ :=
  x^3 - 6*x^2 + 9*x - a*b*c

-- The statement to be proven, without providing the actual proof.
theorem correct_conclusion 
  (a b c : ℝ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : f a a b c = 0) 
  (h4 : f b a b c = 0) 
  (h5 : f c a b c = 0) :
  f 0 a b c * f 1 a b c < 0 ∧ f 0 a b c * f 3 a b c > 0 :=
sorry

end correct_conclusion_l1811_181184


namespace find_costs_of_accessories_max_type_a_accessories_l1811_181171

theorem find_costs_of_accessories (x y : ℕ) 
  (h1 : x + 3 * y = 530) 
  (h2 : 3 * x + 2 * y = 890) : 
  x = 230 ∧ y = 100 := 
by 
  sorry

theorem max_type_a_accessories (m n : ℕ) 
  (m_n_sum : m + n = 30) 
  (cost_constraint : 230 * m + 100 * n ≤ 4180) : 
  m ≤ 9 := 
by 
  sorry

end find_costs_of_accessories_max_type_a_accessories_l1811_181171
