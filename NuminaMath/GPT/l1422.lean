import Mathlib

namespace NUMINAMATH_GPT_pig_duck_ratio_l1422_142292

theorem pig_duck_ratio (G C D P : ℕ)
(h₁ : G = 66)
(h₂ : C = 2 * G)
(h₃ : D = (G + C) / 2)
(h₄ : P = G - 33) :
  P / D = 1 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_pig_duck_ratio_l1422_142292


namespace NUMINAMATH_GPT_can_place_circles_l1422_142243

theorem can_place_circles (r: ℝ) (h: r = 2008) :
  ∃ (n: ℕ), (n > 4016) ∧ ((n: ℝ) / 2 > r) :=
by 
  sorry

end NUMINAMATH_GPT_can_place_circles_l1422_142243


namespace NUMINAMATH_GPT_arithmetic_sequence_5th_term_l1422_142232

theorem arithmetic_sequence_5th_term :
  let a1 := 3
  let d := 4
  a1 + 4 * (5 - 1) = 19 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_5th_term_l1422_142232


namespace NUMINAMATH_GPT_brick_height_calc_l1422_142225

theorem brick_height_calc 
  (length_wall : ℝ) (height_wall : ℝ) (width_wall : ℝ) 
  (num_bricks : ℕ) 
  (length_brick : ℝ) (width_brick : ℝ) 
  (H : ℝ) 
  (volume_wall : ℝ) 
  (volume_brick : ℝ)
  (condition1 : length_wall = 800) 
  (condition2 : height_wall = 600) 
  (condition3 : width_wall = 22.5)
  (condition4 : num_bricks = 3200) 
  (condition5 : length_brick = 50) 
  (condition6 : width_brick = 11.25) 
  (condition7 : volume_wall = length_wall * height_wall * width_wall) 
  (condition8 : volume_brick = length_brick * width_brick * H) 
  (condition9 : num_bricks * volume_brick = volume_wall) 
  : H = 6 := 
by
  sorry

end NUMINAMATH_GPT_brick_height_calc_l1422_142225


namespace NUMINAMATH_GPT_paint_coverage_l1422_142216

-- Define the conditions
def cost_per_gallon : ℝ := 45
def total_area : ℝ := 1600
def number_of_coats : ℝ := 2
def total_contribution : ℝ := 180 + 180

-- Define the target statement to prove
theorem paint_coverage (H : total_contribution = 360) : 
  let cost_per_gallon := 45 
  let number_of_gallons := total_contribution / cost_per_gallon
  let total_coverage := total_area * number_of_coats
  let coverage_per_gallon := total_coverage / number_of_gallons
  coverage_per_gallon = 400 :=
by
  sorry

end NUMINAMATH_GPT_paint_coverage_l1422_142216


namespace NUMINAMATH_GPT_students_just_passed_l1422_142239

theorem students_just_passed (total_students : ℕ) (first_division : ℕ) (second_division : ℕ) (just_passed : ℕ)
  (h1 : total_students = 300)
  (h2 : first_division = 26 * total_students / 100)
  (h3 : second_division = 54 * total_students / 100)
  (h4 : just_passed = total_students - (first_division + second_division)) :
  just_passed = 60 :=
sorry

end NUMINAMATH_GPT_students_just_passed_l1422_142239


namespace NUMINAMATH_GPT_sum_arithmetic_series_l1422_142269

theorem sum_arithmetic_series :
  let a := -42
  let d := 2
  let l := 0
  let n := (l - a) / d + 1
  let S := n * (a + l) / 2
  S = -462 := by
sorry

end NUMINAMATH_GPT_sum_arithmetic_series_l1422_142269


namespace NUMINAMATH_GPT_first_number_in_proportion_l1422_142263

variable (x y : ℝ)

theorem first_number_in_proportion
  (h1 : x = 0.9)
  (h2 : y / x = 5 / 6) : 
  y = 0.75 := 
  by 
    sorry

end NUMINAMATH_GPT_first_number_in_proportion_l1422_142263


namespace NUMINAMATH_GPT_fraction_of_menu_vegan_soy_free_l1422_142202

def num_vegan_dishes : Nat := 6
def fraction_menu_vegan : ℚ := 1 / 4
def num_vegan_dishes_with_soy : Nat := 4

def num_vegan_soy_free_dishes : Nat := num_vegan_dishes - num_vegan_dishes_with_soy
def fraction_vegan_soy_free : ℚ := num_vegan_soy_free_dishes / num_vegan_dishes
def fraction_menu_vegan_soy_free : ℚ := fraction_vegan_soy_free * fraction_menu_vegan

theorem fraction_of_menu_vegan_soy_free :
  fraction_menu_vegan_soy_free = 1 / 12 := by
  sorry

end NUMINAMATH_GPT_fraction_of_menu_vegan_soy_free_l1422_142202


namespace NUMINAMATH_GPT_correct_answer_l1422_142215

def g (x : ℤ) : ℤ := x^3
def f (x : ℤ) : ℤ := 3*x - 2

theorem correct_answer : f (g 3) = 79 := by
  sorry

end NUMINAMATH_GPT_correct_answer_l1422_142215


namespace NUMINAMATH_GPT_curve_transformation_l1422_142204

-- Define the scaling transformation
def scaling_transform (x y : ℝ) : ℝ × ℝ :=
  (5 * x, 3 * y)

-- Define the transformed curve
def transformed_curve (x' y' : ℝ) : Prop :=
  2 * x' ^ 2 + 8 * y' ^ 2 = 1

-- Define the curve C's equation after scaling
def curve_C (x y : ℝ) : Prop :=
  50 * x ^ 2 + 72 * y ^ 2 = 1

-- Statement of the proof problem
theorem curve_transformation (x y : ℝ) (h : transformed_curve (5 * x) (3 * y)) : curve_C x y :=
by {
  -- The actual proof would be filled in here
  sorry
}

end NUMINAMATH_GPT_curve_transformation_l1422_142204


namespace NUMINAMATH_GPT_largest_integer_m_l1422_142296

theorem largest_integer_m (m n : ℕ) (h1 : ∀ n ≤ m, (2 * n + 1) / (3 * n + 8) < (Real.sqrt 5 - 1) / 2) 
(h2 : ∀ n ≤ m, (Real.sqrt 5 - 1) / 2 < (n + 7) / (2 * n + 1)) : 
  m = 27 :=
sorry

end NUMINAMATH_GPT_largest_integer_m_l1422_142296


namespace NUMINAMATH_GPT_find_x_l1422_142264

open Real

theorem find_x 
  (x y : ℝ) 
  (hx_pos : 0 < x)
  (hy_pos : 0 < y) 
  (h_eq : 7 * x^2 + 21 * x * y = 2 * x^3 + 3 * x^2 * y) 
  : x = 7 := 
sorry

end NUMINAMATH_GPT_find_x_l1422_142264


namespace NUMINAMATH_GPT_total_number_of_squares_l1422_142259

variable (x y : ℕ) -- Variables for the number of 10 cm and 20 cm squares

theorem total_number_of_squares
  (h1 : 100 * x + 400 * y = 2500) -- Condition for area
  (h2 : 40 * x + 80 * y = 280)    -- Condition for cutting length
  : (x + y = 16) :=
sorry

end NUMINAMATH_GPT_total_number_of_squares_l1422_142259


namespace NUMINAMATH_GPT_jan_uses_24_gallons_for_plates_and_clothes_l1422_142237

theorem jan_uses_24_gallons_for_plates_and_clothes :
  (65 - (2 * 7 + (2 * 7 - 11))) / 2 = 24 :=
by sorry

end NUMINAMATH_GPT_jan_uses_24_gallons_for_plates_and_clothes_l1422_142237


namespace NUMINAMATH_GPT_two_digit_even_multiple_of_7_l1422_142238

def all_digits_product_square (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  (d1 * d2) > 0 ∧ ∃ k, d1 * d2 = k * k

theorem two_digit_even_multiple_of_7 (n : ℕ) :
  10 ≤ n ∧ n < 100 ∧ n % 2 = 0 ∧ n % 7 = 0 ∧ all_digits_product_square n ↔ n = 14 ∨ n = 28 ∨ n = 70 :=
by sorry

end NUMINAMATH_GPT_two_digit_even_multiple_of_7_l1422_142238


namespace NUMINAMATH_GPT_seahawks_touchdowns_l1422_142213

theorem seahawks_touchdowns (total_points : ℕ) (points_per_touchdown : ℕ) (points_per_field_goal : ℕ) (field_goals : ℕ) (touchdowns : ℕ) :
  total_points = 37 →
  points_per_touchdown = 7 →
  points_per_field_goal = 3 →
  field_goals = 3 →
  total_points = (touchdowns * points_per_touchdown) + (field_goals * points_per_field_goal) →
  touchdowns = 4 :=
by
  intros h_total_points h_points_per_touchdown h_points_per_field_goal h_field_goals h_equation
  sorry

end NUMINAMATH_GPT_seahawks_touchdowns_l1422_142213


namespace NUMINAMATH_GPT_circle_properties_l1422_142286

noncomputable def circle_center_and_radius (x y : ℝ) : ℝ × ℝ × ℝ :=
  let eq1 := x^2 - 4 * y - 18
  let eq2 := -y^2 + 6 * x + 26
  let lhs := x^2 - 6 * x + y^2 - 4 * y
  let rhs := 44
  let center_x := 3
  let center_y := 2
  let radius := Real.sqrt 57
  let target := 5 + radius
  (center_x, center_y, target)

theorem circle_properties
  (x y : ℝ) :
  let (a, b, r) := circle_center_and_radius x y 
  a + b + r = 5 + Real.sqrt 57 :=
by
  sorry

end NUMINAMATH_GPT_circle_properties_l1422_142286


namespace NUMINAMATH_GPT_infinite_geometric_sum_l1422_142290

noncomputable def geometric_sequence (n : ℕ) : ℝ := 3 * (-1 / 2)^(n - 1)

theorem infinite_geometric_sum :
  ∑' n, geometric_sequence n = 2 :=
sorry

end NUMINAMATH_GPT_infinite_geometric_sum_l1422_142290


namespace NUMINAMATH_GPT_ratio_of_teaspoons_to_knives_is_2_to_1_l1422_142299

-- Define initial conditions based on the problem
def initial_knives : ℕ := 24
def initial_teaspoons (T : ℕ) : Prop := 
  initial_knives + T + (1 / 3 : ℚ) * initial_knives + (2 / 3 : ℚ) * T = 112

-- Define the ratio to be proved
def ratio_teaspoons_to_knives (T : ℕ) : Prop :=
  initial_teaspoons T ∧ T = 48 ∧ 48 / initial_knives = 2

theorem ratio_of_teaspoons_to_knives_is_2_to_1 : ∃ T, ratio_teaspoons_to_knives T :=
by
  -- Proof would follow here
  sorry

end NUMINAMATH_GPT_ratio_of_teaspoons_to_knives_is_2_to_1_l1422_142299


namespace NUMINAMATH_GPT_smallest_percentage_owning_90_percent_money_l1422_142283

theorem smallest_percentage_owning_90_percent_money
  (P M : ℝ)
  (h1 : 0.2 * P = 0.8 * M) :
  (∃ x : ℝ, x = 0.6 * P ∧ 0.9 * M <= (0.2 * P + (x - 0.2 * P))) :=
sorry

end NUMINAMATH_GPT_smallest_percentage_owning_90_percent_money_l1422_142283


namespace NUMINAMATH_GPT_shorter_piece_length_l1422_142271

theorem shorter_piece_length (x : ℕ) (h1 : 177 = x + 2*x) : x = 59 :=
by sorry

end NUMINAMATH_GPT_shorter_piece_length_l1422_142271


namespace NUMINAMATH_GPT_ratio_proof_l1422_142261

noncomputable def total_capacity : ℝ := 10 -- million gallons
noncomputable def amount_end_month : ℝ := 6 -- million gallons
noncomputable def normal_level : ℝ := total_capacity - 5 -- million gallons

theorem ratio_proof (h1 : amount_end_month = 0.6 * total_capacity)
                    (h2 : normal_level = total_capacity - 5) :
  (amount_end_month / normal_level) = 1.2 :=
by sorry

end NUMINAMATH_GPT_ratio_proof_l1422_142261


namespace NUMINAMATH_GPT_haley_tv_total_hours_l1422_142200

theorem haley_tv_total_hours (h_sat : Nat) (h_sun : Nat) (H_sat : h_sat = 6) (H_sun : h_sun = 3) :
  h_sat + h_sun = 9 := by
  sorry

end NUMINAMATH_GPT_haley_tv_total_hours_l1422_142200


namespace NUMINAMATH_GPT_integer_multiple_of_ten_l1422_142277

theorem integer_multiple_of_ten (x : ℤ) :
  10 * x = 30 ↔ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_integer_multiple_of_ten_l1422_142277


namespace NUMINAMATH_GPT_number_of_birds_l1422_142219

-- Conditions
def geese : ℕ := 58
def ducks : ℕ := 37

-- Proof problem statement
theorem number_of_birds : geese + ducks = 95 :=
by
  -- The actual proof is to be provided
  sorry

end NUMINAMATH_GPT_number_of_birds_l1422_142219


namespace NUMINAMATH_GPT_number_of_toys_bought_l1422_142212

def toy_cost (T : ℕ) : ℕ := 10 * T
def card_cost : ℕ := 2 * 5
def shirt_cost : ℕ := 5 * 6
def total_cost (T : ℕ) : ℕ := toy_cost T + card_cost + shirt_cost

theorem number_of_toys_bought (T : ℕ) : total_cost T = 70 → T = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_number_of_toys_bought_l1422_142212


namespace NUMINAMATH_GPT_problem_1_problem_2_l1422_142230

variable (a : ℝ) (x : ℝ)

theorem problem_1 (h : a ≠ 1) : (a^2 / (a - 1)) - (a / (a - 1)) = a := 
sorry

theorem problem_2 (h : x ≠ -1) : (x^2 / (x + 1)) - x + 1 = 1 / (x + 1) := 
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1422_142230


namespace NUMINAMATH_GPT_sequence_term_2023_l1422_142201

theorem sequence_term_2023 (a : ℕ → ℚ) (h₁ : a 1 = 2) 
  (h₂ : ∀ n, 1 / a n - 1 / a (n + 1) - 1 / (a n * a (n + 1)) = 1) : 
  a 2023 = -1 / 2 := 
sorry

end NUMINAMATH_GPT_sequence_term_2023_l1422_142201


namespace NUMINAMATH_GPT_study_time_l1422_142253

theorem study_time (n_mcq n_fitb : ℕ) (t_mcq t_fitb : ℕ) (total_minutes_per_hour : ℕ) 
  (h1 : n_mcq = 30) (h2 : n_fitb = 30) (h3 : t_mcq = 15) (h4 : t_fitb = 25) (h5 : total_minutes_per_hour = 60) : 
  n_mcq * t_mcq + n_fitb * t_fitb = 20 * total_minutes_per_hour := 
by 
  -- This is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_study_time_l1422_142253


namespace NUMINAMATH_GPT_tan_identity_l1422_142223

theorem tan_identity (A B : ℝ) (hA : A = 30) (hB : B = 30) :
  (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = (4 + 2 * Real.sqrt 3)/3 := by
  sorry

end NUMINAMATH_GPT_tan_identity_l1422_142223


namespace NUMINAMATH_GPT_prove_a_range_l1422_142257

noncomputable def f (x : ℝ) : ℝ := 1 / (2 ^ x + 2)

theorem prove_a_range (a : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 3 → f x + f (a - 2 * x) ≤ 1 / 2) → 5 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_prove_a_range_l1422_142257


namespace NUMINAMATH_GPT_mean_of_xyz_l1422_142251

theorem mean_of_xyz (mean7 : ℕ) (mean10 : ℕ) (x y z : ℕ) (h1 : mean7 = 40) (h2 : mean10 = 50) : (x + y + z) / 3 = 220 / 3 :=
by
  have sum7 := 7 * mean7
  have sum10 := 10 * mean10
  have sum_xyz := sum10 - sum7
  have mean_xyz := sum_xyz / 3
  sorry

end NUMINAMATH_GPT_mean_of_xyz_l1422_142251


namespace NUMINAMATH_GPT_percentage_increased_is_correct_l1422_142252

-- Define the initial and final numbers
def initial_number : Nat := 150
def final_number : Nat := 210

-- Define the function to compute the percentage increase
def percentage_increase (initial final : Nat) : Float :=
  ((final - initial).toFloat / initial.toFloat) * 100.0

-- The theorem we need to prove
theorem percentage_increased_is_correct :
  percentage_increase initial_number final_number = 40 := 
by
  simp [percentage_increase, initial_number, final_number]
  sorry

end NUMINAMATH_GPT_percentage_increased_is_correct_l1422_142252


namespace NUMINAMATH_GPT_find_b_l1422_142248

noncomputable def a (c : ℚ) : ℚ := 10 * c - 10
noncomputable def b (c : ℚ) : ℚ := 10 * c + 10
noncomputable def c_val := (200 : ℚ) / 21

theorem find_b : 
  let a := a c_val
  let b := b c_val
  let c := c_val
  a + b + c = 200 ∧ 
  a + 10 = b - 10 ∧ 
  a + 10 = 10 * c → 
  b = 2210 / 21 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_b_l1422_142248


namespace NUMINAMATH_GPT_problem1_problem2_l1422_142293

variable (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)

theorem problem1 : 
  (a * b + a + b + 1) * (a * b + a * c + b * c + c ^ 2) ≥ 16 * a * b * c := 
by sorry

theorem problem2 : 
  (b + c - a) / a + (c + a - b) / b + (a + b - c) / c ≥ 3 := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1422_142293


namespace NUMINAMATH_GPT_train_pass_time_l1422_142289

-- Definitions based on the conditions
def train_length : ℕ := 360   -- Length of the train in meters
def platform_length : ℕ := 190 -- Length of the platform in meters
def speed_kmh : ℕ := 45       -- Speed of the train in km/h
def speed_ms : ℚ := speed_kmh * (1000 / 3600) -- Speed of the train in m/s

-- Total distance to be covered
def total_distance : ℕ := train_length + platform_length 

-- Time taken to pass the platform
def time_to_pass_platform : ℚ := total_distance / speed_ms

-- Proof that the time taken is 44 seconds
theorem train_pass_time : time_to_pass_platform = 44 := 
by 
  -- this is where the detailed proof would go
  sorry  

end NUMINAMATH_GPT_train_pass_time_l1422_142289


namespace NUMINAMATH_GPT_solve_for_x_l1422_142249

theorem solve_for_x (x : ℝ) (y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3/8 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_solve_for_x_l1422_142249


namespace NUMINAMATH_GPT_abc_positive_l1422_142234

theorem abc_positive (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) (h3 : abc > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_abc_positive_l1422_142234


namespace NUMINAMATH_GPT_min_value_xy_l1422_142285

theorem min_value_xy (x y : ℕ) (h : 0 < x ∧ 0 < y) (cond : (1 : ℚ) / x + (1 : ℚ) /(3 * y) = 1 / 6) : 
  xy = 192 :=
sorry

end NUMINAMATH_GPT_min_value_xy_l1422_142285


namespace NUMINAMATH_GPT_area_of_ABC_l1422_142236

def point : Type := ℝ × ℝ

def area_of_triangle (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_ABC : area_of_triangle (0, 0) (1, 0) (0, 1) = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_area_of_ABC_l1422_142236


namespace NUMINAMATH_GPT_max_value_of_f_l1422_142291

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem max_value_of_f : ∃ max, max ∈ Set.image f (Set.Icc (-1 : ℝ) 1) ∧ max = Real.exp 1 - 1 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l1422_142291


namespace NUMINAMATH_GPT_equal_if_fraction_is_positive_integer_l1422_142262

theorem equal_if_fraction_is_positive_integer
  (a b : ℕ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (K : ℝ := Real.sqrt ((a^2 + b^2:ℕ)/2))
  (A : ℝ := (a + b:ℕ)/2)
  (h_int_pos : ∃ (n : ℕ), n > 0 ∧ K / A = n) :
  a = b := sorry

end NUMINAMATH_GPT_equal_if_fraction_is_positive_integer_l1422_142262


namespace NUMINAMATH_GPT_solve_x_for_equation_l1422_142256

theorem solve_x_for_equation :
  ∃ (x : ℚ), 3 * x - 5 = abs (-20 + 6) ∧ x = 19 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_x_for_equation_l1422_142256


namespace NUMINAMATH_GPT_total_chapters_read_l1422_142272

def books_read : ℕ := 12
def chapters_per_book : ℕ := 32

theorem total_chapters_read : books_read * chapters_per_book = 384 :=
by
  sorry

end NUMINAMATH_GPT_total_chapters_read_l1422_142272


namespace NUMINAMATH_GPT_cone_section_volume_ratio_l1422_142278

theorem cone_section_volume_ratio :
  ∀ (r h : ℝ), (h > 0 ∧ r > 0) →
  let V1 := ((75 / 3) * π * r^2 * h - (64 / 3) * π * r^2 * h)
  let V2 := ((64 / 3) * π * r^2 * h - (27 / 3) * π * r^2 * h)
  V2 / V1 = 37 / 11 :=
by
  intros r h h_pos
  sorry

end NUMINAMATH_GPT_cone_section_volume_ratio_l1422_142278


namespace NUMINAMATH_GPT_sum_of_perpendiculars_l1422_142250

-- define the points on the rectangle
variables {A B C D P S R Q F : Type}

-- define rectangle ABCD and points P, S, R, Q, F
def is_rectangle (A B C D : Type) : Prop := sorry -- conditions for ABCD to be a rectangle
def point_on_segment (P A B: Type) : Prop := sorry -- P is a point on segment AB
def perpendicular (X Y Z : Type) : Prop := sorry -- definition for perpendicular between two segments
def length (X Y : Type) : ℝ := sorry -- definition for the length of a segment

-- Given conditions
axiom rect : is_rectangle A B C D
axiom p_on_ab : point_on_segment P A B
axiom ps_perp_bd : perpendicular P S D
axiom pr_perp_ac : perpendicular P R C
axiom af_perp_bd : perpendicular A F D
axiom pq_perp_af : perpendicular P Q F

-- Prove that PR + PS = AF
theorem sum_of_perpendiculars :
  length P R + length P S = length A F :=
sorry

end NUMINAMATH_GPT_sum_of_perpendiculars_l1422_142250


namespace NUMINAMATH_GPT_number_of_dogs_with_both_tags_and_collars_l1422_142280

-- Defining the problem
def total_dogs : ℕ := 80
def dogs_with_tags : ℕ := 45
def dogs_with_collars : ℕ := 40
def dogs_with_neither : ℕ := 1

-- Statement: Prove the number of dogs with both tags and collars
theorem number_of_dogs_with_both_tags_and_collars : 
  (dogs_with_tags + dogs_with_collars - total_dogs + dogs_with_neither) = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_dogs_with_both_tags_and_collars_l1422_142280


namespace NUMINAMATH_GPT_cost_of_four_dozen_apples_l1422_142270

-- Define the given conditions and problem
def half_dozen_cost : ℚ := 4.80 -- cost of half a dozen apples
def full_dozen_cost : ℚ := half_dozen_cost / 0.5
def four_dozen_cost : ℚ := 4 * full_dozen_cost

-- Statement of the theorem to prove
theorem cost_of_four_dozen_apples : four_dozen_cost = 38.40 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_four_dozen_apples_l1422_142270


namespace NUMINAMATH_GPT_number_of_true_propositions_l1422_142240

theorem number_of_true_propositions : 
  let original_p := ∀ (a : ℝ), a > -1 → a > -2
  let converse_p := ∀ (a : ℝ), a > -2 → a > -1
  let inverse_p := ∀ (a : ℝ), a ≤ -1 → a ≤ -2
  let contrapositive_p := ∀ (a : ℝ), a ≤ -2 → a ≤ -1
  (original_p ∧ contrapositive_p ∧ ¬converse_p ∧ ¬inverse_p) → (2 = 2) :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_true_propositions_l1422_142240


namespace NUMINAMATH_GPT_negation_proof_l1422_142233

-- Definitions based on conditions
def atMostTwoSolutions (solutions : ℕ) : Prop := solutions ≤ 2
def atLeastThreeSolutions (solutions : ℕ) : Prop := solutions ≥ 3

-- Statement of the theorem
theorem negation_proof (solutions : ℕ) : atMostTwoSolutions solutions ↔ ¬ atLeastThreeSolutions solutions :=
by
  sorry

end NUMINAMATH_GPT_negation_proof_l1422_142233


namespace NUMINAMATH_GPT_salary_increase_correct_l1422_142273

noncomputable def old_average_salary : ℕ := 1500
noncomputable def number_of_employees : ℕ := 24
noncomputable def manager_salary : ℕ := 11500
noncomputable def new_total_salary := (number_of_employees * old_average_salary) + manager_salary
noncomputable def new_number_of_people := number_of_employees + 1
noncomputable def new_average_salary := new_total_salary / new_number_of_people
noncomputable def salary_increase := new_average_salary - old_average_salary

theorem salary_increase_correct : salary_increase = 400 := by
sorry

end NUMINAMATH_GPT_salary_increase_correct_l1422_142273


namespace NUMINAMATH_GPT_angle_B_pi_div_3_triangle_perimeter_l1422_142226

-- Problem 1: Prove that B = π / 3 given the condition.
theorem angle_B_pi_div_3 (A B C : ℝ) (hTriangle : A + B + C = Real.pi) 
  (hCos : Real.cos B = Real.cos ((A + C) / 2)) : 
  B = Real.pi / 3 :=
sorry

-- Problem 2: Prove the perimeter given the conditions.
theorem triangle_perimeter (a b c : ℝ) (m : ℝ) 
  (altitude : ℝ) 
  (hSides : 8 * a = 3 * c) 
  (hAltitude : altitude = 12 * Real.sqrt 3 / 7) 
  (hAngleB : ∃ B, B = Real.pi / 3) :
  a + b + c = 18 := 
sorry

end NUMINAMATH_GPT_angle_B_pi_div_3_triangle_perimeter_l1422_142226


namespace NUMINAMATH_GPT_coloring_points_l1422_142287

theorem coloring_points
  (A : ℤ × ℤ) (B : ℤ × ℤ) (C : ℤ × ℤ)
  (hA : A.fst % 2 = 1 ∧ A.snd % 2 = 1)
  (hB : (B.fst % 2 = 1 ∧ B.snd % 2 = 0) ∨ (B.fst % 2 = 0 ∧ B.snd % 2 = 1))
  (hC : C.fst % 2 = 0 ∧ C.snd % 2 = 0) :
  ∃ D : ℤ × ℤ,
    (D.fst % 2 = 1 ∧ D.snd % 2 = 0) ∨ (D.fst % 2 = 0 ∧ D.snd % 2 = 1) ∧
    (A.fst + C.fst = B.fst + D.fst) ∧
    (A.snd + C.snd = B.snd + D.snd) := 
sorry

end NUMINAMATH_GPT_coloring_points_l1422_142287


namespace NUMINAMATH_GPT_probability_both_counterfeit_given_one_counterfeit_l1422_142258

-- Conditions
def total_bills := 20
def counterfeit_bills := 5
def selected_bills := 2
def at_least_one_counterfeit := true

-- Definition of events
def eventA := "both selected bills are counterfeit"
def eventB := "at least one of the selected bills is counterfeit"

-- The theorem to prove
theorem probability_both_counterfeit_given_one_counterfeit : 
  at_least_one_counterfeit →
  ( (counterfeit_bills * (counterfeit_bills - 1)) / (total_bills * (total_bills - 1)) ) / 
    ( (counterfeit_bills * (counterfeit_bills - 1) + counterfeit_bills * (total_bills - counterfeit_bills)) / (total_bills * (total_bills - 1)) ) = 2/17 :=
by
  sorry

end NUMINAMATH_GPT_probability_both_counterfeit_given_one_counterfeit_l1422_142258


namespace NUMINAMATH_GPT_find_integer_pairs_l1422_142255

-- Define the plane and lines properties
def horizontal_lines (h : ℕ) : Prop := h > 0
def non_horizontal_lines (s : ℕ) : Prop := s > 0
def non_parallel (s : ℕ) : Prop := s > 0
def no_three_intersect (total_lines : ℕ) : Prop := total_lines > 0

-- Function to calculate regions from the given formula
def calculate_regions (h s : ℕ) : ℕ :=
  h * (s + 1) + 1 + (s * (s + 1)) / 2

-- Prove that the given (h, s) pairs divide the plane into 1992 regions
theorem find_integer_pairs :
  (horizontal_lines 995 ∧ non_horizontal_lines 1 ∧ non_parallel 1 ∧ no_three_intersect (995 + 1) ∧ calculate_regions 995 1 = 1992)
  ∨ (horizontal_lines 176 ∧ non_horizontal_lines 10 ∧ non_parallel 10 ∧ no_three_intersect (176 + 10) ∧ calculate_regions 176 10 = 1992)
  ∨ (horizontal_lines 80 ∧ non_horizontal_lines 21 ∧ non_parallel 21 ∧ no_three_intersect (80 + 21) ∧ calculate_regions 80 21 = 1992) :=
by
  -- Include individual cases to verify correctness of regions calculation
  sorry

end NUMINAMATH_GPT_find_integer_pairs_l1422_142255


namespace NUMINAMATH_GPT_insurance_payment_yearly_l1422_142297

noncomputable def quarterly_payment : ℝ := 378
noncomputable def quarters_per_year : ℕ := 12 / 3
noncomputable def annual_payment : ℝ := quarterly_payment * quarters_per_year

theorem insurance_payment_yearly : annual_payment = 1512 := by
  sorry

end NUMINAMATH_GPT_insurance_payment_yearly_l1422_142297


namespace NUMINAMATH_GPT_greatest_value_x_l1422_142242

theorem greatest_value_x (x : ℕ) (h : lcm (lcm x 12) 18 = 108) : x ≤ 108 := sorry

end NUMINAMATH_GPT_greatest_value_x_l1422_142242


namespace NUMINAMATH_GPT_average_marks_l1422_142229

theorem average_marks :
  let a1 := 76
  let a2 := 65
  let a3 := 82
  let a4 := 67
  let a5 := 75
  let n := 5
  let total_marks := a1 + a2 + a3 + a4 + a5
  let avg_marks := total_marks / n
  avg_marks = 73 :=
by
  sorry

end NUMINAMATH_GPT_average_marks_l1422_142229


namespace NUMINAMATH_GPT_Jon_regular_bottle_size_is_16oz_l1422_142281

noncomputable def Jon_bottle_size (x : ℝ) : Prop :=
  let daily_intake := 4 * x + 2 * 1.25 * x
  let weekly_intake := 7 * daily_intake
  weekly_intake = 728

theorem Jon_regular_bottle_size_is_16oz : ∃ x : ℝ, Jon_bottle_size x ∧ x = 16 :=
by
  use 16
  sorry

end NUMINAMATH_GPT_Jon_regular_bottle_size_is_16oz_l1422_142281


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1422_142211

theorem arithmetic_sequence_sum :
  ∃ (a l d n : ℕ), a = 71 ∧ l = 109 ∧ d = 2 ∧ n = ((l - a) / d) + 1 ∧ 
    (3 * (n * (a + l) / 2) = 5400) := sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1422_142211


namespace NUMINAMATH_GPT_price_of_other_stock_l1422_142241

theorem price_of_other_stock (total_shares : ℕ) (total_spent : ℝ) (share_1_quantity : ℕ) (share_1_price : ℝ) :
  total_shares = 450 ∧ total_spent = 1950 ∧ share_1_quantity = 400 ∧ share_1_price = 3 →
  (750 / 50 = 15) :=
by sorry

end NUMINAMATH_GPT_price_of_other_stock_l1422_142241


namespace NUMINAMATH_GPT_triangle_inscribed_in_semicircle_l1422_142205

variables {R : ℝ} (P Q R' : ℝ) (PR QR : ℝ)
variables (hR : 0 < R) (h_pq_diameter: P = -R ∧ Q = R)
variables (h_pr_square_qr_square : PR^2 + QR^2 = 4 * R^2)
variables (t := PR + QR)

theorem triangle_inscribed_in_semicircle (h_pos_pr : 0 < PR) (h_pos_qr : 0 < QR) : 
  t^2 ≤ 8 * R^2 :=
sorry

end NUMINAMATH_GPT_triangle_inscribed_in_semicircle_l1422_142205


namespace NUMINAMATH_GPT_selection_options_l1422_142221

theorem selection_options (group1 : Fin 5) (group2 : Fin 4) : (group1.1 + group2.1 + 1 = 9) :=
sorry

end NUMINAMATH_GPT_selection_options_l1422_142221


namespace NUMINAMATH_GPT_ceil_mul_eq_225_l1422_142276

theorem ceil_mul_eq_225 {x : ℝ} (h₁ : ⌈x⌉ * x = 225) (h₂ : x > 0) : x = 15 :=
sorry

end NUMINAMATH_GPT_ceil_mul_eq_225_l1422_142276


namespace NUMINAMATH_GPT_sum_of_a_b_l1422_142218

theorem sum_of_a_b (a b : ℝ) (h1 : ∀ x : ℝ, (a * (b * x + a) + b = x))
  (h2 : ∀ y : ℝ, (b * (a * y + b) + a = y)) : a + b = -2 := 
sorry

end NUMINAMATH_GPT_sum_of_a_b_l1422_142218


namespace NUMINAMATH_GPT_factor_polynomial_l1422_142268

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l1422_142268


namespace NUMINAMATH_GPT_intersection_eq_l1422_142222

def setA : Set ℝ := { x | abs (x - 3) < 2 }
def setB : Set ℝ := { x | (x - 4) / x ≥ 0 }

theorem intersection_eq : setA ∩ setB = { x | 4 ≤ x ∧ x < 5 } :=
by 
  sorry

end NUMINAMATH_GPT_intersection_eq_l1422_142222


namespace NUMINAMATH_GPT_gravitational_equal_forces_point_l1422_142294

variable (d M m : ℝ) (hM : 0 < M) (hm : 0 < m) (hd : 0 < d)

theorem gravitational_equal_forces_point :
  ∃ x : ℝ, (0 < x ∧ x < d) ∧ x = d / (1 + Real.sqrt (m / M)) :=
by
  sorry

end NUMINAMATH_GPT_gravitational_equal_forces_point_l1422_142294


namespace NUMINAMATH_GPT_find_xyz_l1422_142266

-- Let a, b, c, x, y, z be nonzero complex numbers
variables (a b c x y z : ℂ)
-- Given conditions
variables (h1 : a = (b + c) / (x - 2))
variables (h2 : b = (a + c) / (y - 2))
variables (h3 : c = (a + b) / (z - 2))
variables (h4 : x * y + x * z + y * z = 5)
variables (h5 : x + y + z = 3)

-- Prove that xyz = 5
theorem find_xyz : x * y * z = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_xyz_l1422_142266


namespace NUMINAMATH_GPT_domain_of_transformed_function_l1422_142275

theorem domain_of_transformed_function (f : ℝ → ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 2 → True) :
  ∀ x, -1 ≤ x ∧ x ≤ 1 → True :=
sorry

end NUMINAMATH_GPT_domain_of_transformed_function_l1422_142275


namespace NUMINAMATH_GPT_initial_number_correct_l1422_142260

def initial_number_problem : Prop :=
  ∃ (x : ℝ), x + 3889 - 47.80600000000004 = 3854.002 ∧
            x = 12.808000000000158

theorem initial_number_correct : initial_number_problem :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_initial_number_correct_l1422_142260


namespace NUMINAMATH_GPT_part1_part2_l1422_142246

open Real

def f (x : ℝ) : ℝ := |x - 5| - |x - 2|

theorem part1 (m : ℝ) : (∃ x : ℝ, f x ≤ m) ↔ m ≥ -3 := 
sorry

theorem part2 : {x : ℝ | x^2 - 8 * x + 15 + f x ≤ 0} = {x : ℝ | 5 - sqrt 3 ≤ x ∧ x ≤ 6} :=
sorry

end NUMINAMATH_GPT_part1_part2_l1422_142246


namespace NUMINAMATH_GPT_pen_cost_price_l1422_142288

-- Define the variables and assumptions
variable (x : ℝ)

-- Given conditions
def profit_one_pen (x : ℝ) := 10 - x
def profit_three_pens (x : ℝ) := 20 - 3 * x

-- Statement to prove
theorem pen_cost_price : profit_one_pen x = profit_three_pens x → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_pen_cost_price_l1422_142288


namespace NUMINAMATH_GPT_pens_multiple_91_l1422_142267

theorem pens_multiple_91 (S : ℕ) (P : ℕ) (total_pencils : ℕ) 
  (h1 : S = 91) (h2 : total_pencils = 910) (h3 : total_pencils % S = 0) :
  ∃ (x : ℕ), P = S * x :=
by 
  sorry

end NUMINAMATH_GPT_pens_multiple_91_l1422_142267


namespace NUMINAMATH_GPT_total_points_first_four_games_l1422_142284

-- Define the scores for the first three games
def score1 : ℕ := 10
def score2 : ℕ := 14
def score3 : ℕ := 6

-- Define the score for the fourth game as the average of the first three games
def score4 : ℕ := (score1 + score2 + score3) / 3

-- Define the total points scored in the first four games
def total_points : ℕ := score1 + score2 + score3 + score4

-- State the theorem to prove
theorem total_points_first_four_games : total_points = 40 :=
  sorry

end NUMINAMATH_GPT_total_points_first_four_games_l1422_142284


namespace NUMINAMATH_GPT_gold_coins_l1422_142244

theorem gold_coins (c n : ℕ) 
  (h₁ : n = 8 * (c - 1))
  (h₂ : n = 5 * c + 4) :
  n = 24 :=
by
  sorry

end NUMINAMATH_GPT_gold_coins_l1422_142244


namespace NUMINAMATH_GPT_profit_percent_l1422_142247

-- Definitions for the given conditions
variables (P C : ℝ)
-- Condition given: selling at (2/3) of P results in a loss of 5%, i.e., (2/3) * P = 0.95 * C
def condition : Prop := (2 / 3) * P = 0.95 * C

-- Theorem statement: Given the condition, the profit percent when selling at price P is 42.5%
theorem profit_percent (h : condition P C) : ((P - C) / C) * 100 = 42.5 :=
sorry

end NUMINAMATH_GPT_profit_percent_l1422_142247


namespace NUMINAMATH_GPT_krakozyabr_count_l1422_142265

variable (n H W T : ℕ)
variable (h1 : H = 5 * n) -- 20% of the 'krakozyabrs' with horns also have wings
variable (h2 : W = 4 * n) -- 25% of the 'krakozyabrs' with wings also have horns
variable (h3 : T = H + W - n) -- Total number of 'krakozyabrs' using inclusion-exclusion
variable (h4 : 25 < T)
variable (h5 : T < 35)

theorem krakozyabr_count : T = 32 := by
  sorry

end NUMINAMATH_GPT_krakozyabr_count_l1422_142265


namespace NUMINAMATH_GPT_largest_possible_b_l1422_142217

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c ≤ b) (h3 : b < a) (h4 : a * b * c = 360) : b = 10 :=
sorry

end NUMINAMATH_GPT_largest_possible_b_l1422_142217


namespace NUMINAMATH_GPT_arithmetic_series_sum_l1422_142208

theorem arithmetic_series_sum (n P q S₃n : ℕ) (h₁ : 2 * S₃n = 3 * P - q) : S₃n = 3 * P - q :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_series_sum_l1422_142208


namespace NUMINAMATH_GPT_greatest_value_of_n_l1422_142279

theorem greatest_value_of_n (n : ℤ) (h : 101 * n ^ 2 ≤ 3600) : n ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_greatest_value_of_n_l1422_142279


namespace NUMINAMATH_GPT_distinct_positive_integer_roots_l1422_142227

theorem distinct_positive_integer_roots (m a b : ℤ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) (h4 : a + b = -m) (h5 : a * b = -m + 1) : m = -5 := 
by
  sorry

end NUMINAMATH_GPT_distinct_positive_integer_roots_l1422_142227


namespace NUMINAMATH_GPT_traveler_meets_truck_at_15_48_l1422_142282

noncomputable def timeTravelerMeetsTruck : ℝ := 15 + 48 / 60

theorem traveler_meets_truck_at_15_48 {S Vp Vm Vg : ℝ}
  (h_travel_covered : Vp = S / 4)
  (h_motorcyclist_catch : 1 = (S / 4) / (Vm - Vp))
  (h_motorcyclist_meet_truck : 1.5 = S / (Vm + Vg)) :
  (S / 4 + (12 / 5) * (Vg + Vp)) / (12 / 5) = timeTravelerMeetsTruck := sorry

end NUMINAMATH_GPT_traveler_meets_truck_at_15_48_l1422_142282


namespace NUMINAMATH_GPT_sum_of_minimums_is_zero_l1422_142228

noncomputable def P : Polynomial ℝ := sorry
noncomputable def Q : Polynomial ℝ := sorry

-- Conditions: P(Q(x)) has zeros at -5, -3, -1, 1
lemma zeroes_PQ : 
  P.eval (Q.eval (-5)) = 0 ∧ 
  P.eval (Q.eval (-3)) = 0 ∧ 
  P.eval (Q.eval (-1)) = 0 ∧ 
  P.eval (Q.eval (1)) = 0 := 
  sorry

-- Conditions: Q(P(x)) has zeros at -7, -5, -1, 3
lemma zeroes_QP : 
  Q.eval (P.eval (-7)) = 0 ∧ 
  Q.eval (P.eval (-5)) = 0 ∧ 
  Q.eval (P.eval (-1)) = 0 ∧ 
  Q.eval (P.eval (3)) = 0 := 
  sorry

-- Definition to find the minimum value of a polynomial
noncomputable def min_value (P : Polynomial ℝ) : ℝ := sorry

-- Main theorem
theorem sum_of_minimums_is_zero :
  min_value P + min_value Q = 0 := 
  sorry

end NUMINAMATH_GPT_sum_of_minimums_is_zero_l1422_142228


namespace NUMINAMATH_GPT_hyperbola_range_k_l1422_142235

theorem hyperbola_range_k (k : ℝ) : 
  (1 < k ∧ k < 3) ↔ (∃ x y : ℝ, (3 - k > 0) ∧ (k - 1 > 0) ∧ (x * x) / (3 - k) - (y * y) / (k - 1) = 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_hyperbola_range_k_l1422_142235


namespace NUMINAMATH_GPT_equation_solution_system_solution_l1422_142298

theorem equation_solution (x : ℚ) :
  (3 * x + 1) / 5 = 1 - (4 * x + 3) / 2 ↔ x = -7 / 26 :=
by sorry

theorem system_solution (x y : ℚ) :
  (3 * x - 4 * y = 14) ∧ (5 * x + 4 * y = 2) ↔
  (x = 2) ∧ (y = -2) :=
by sorry

end NUMINAMATH_GPT_equation_solution_system_solution_l1422_142298


namespace NUMINAMATH_GPT_find_prob_p_l1422_142206

variable (p : ℚ)

theorem find_prob_p (h : 15 * p^4 * (1 - p)^2 = 500 / 2187) : p = 3 / 7 := 
  sorry

end NUMINAMATH_GPT_find_prob_p_l1422_142206


namespace NUMINAMATH_GPT_trapezoid_base_difference_is_10_l1422_142274

noncomputable def trapezoid_base_difference (AD BC AB : ℝ) (angle_BAD angle_ADC : ℝ) : ℝ :=
if angle_BAD = 60 ∧ angle_ADC = 30 ∧ AB = 5 then AD - BC else 0

theorem trapezoid_base_difference_is_10 (AD BC : ℝ) (angle_BAD angle_ADC : ℝ) (h_BAD : angle_BAD = 60)
(h_ADC : angle_ADC = 30) (h_AB : AB = 5) : trapezoid_base_difference AD BC AB angle_BAD angle_ADC = 10 :=
sorry

end NUMINAMATH_GPT_trapezoid_base_difference_is_10_l1422_142274


namespace NUMINAMATH_GPT_sum_of_consecutive_negative_integers_with_product_3080_l1422_142203

theorem sum_of_consecutive_negative_integers_with_product_3080 :
  ∃ (n : ℤ), n < 0 ∧ (n * (n + 1) = 3080) ∧ (n + (n + 1) = -111) :=
sorry

end NUMINAMATH_GPT_sum_of_consecutive_negative_integers_with_product_3080_l1422_142203


namespace NUMINAMATH_GPT_diamond_value_l1422_142295

def diamond (a b : ℕ) : ℚ := 1 / (a : ℚ) + 2 / (b : ℚ)

theorem diamond_value : ∀ (a b : ℕ), a + b = 10 ∧ a * b = 24 → diamond a b = 2 / 3 := by
  intros a b h
  sorry

end NUMINAMATH_GPT_diamond_value_l1422_142295


namespace NUMINAMATH_GPT_Alyssa_number_of_quarters_l1422_142224

def value_penny : ℝ := 0.01
def value_quarter : ℝ := 0.25
def num_pennies : ℕ := 7
def total_money : ℝ := 3.07

def num_quarters (q : ℕ) : Prop :=
  total_money - (num_pennies * value_penny) = q * value_quarter

theorem Alyssa_number_of_quarters : ∃ q : ℕ, num_quarters q ∧ q = 12 :=
by
  sorry

end NUMINAMATH_GPT_Alyssa_number_of_quarters_l1422_142224


namespace NUMINAMATH_GPT_junior_score_is_90_l1422_142207

theorem junior_score_is_90 {n : ℕ} (hn : n > 0)
    (j : ℕ := n / 5) (s : ℕ := 4 * n / 5)
    (overall_avg : ℝ := 86)
    (senior_avg : ℝ := 85)
    (junior_score : ℝ)
    (h1 : 20 * j = n)
    (h2 : 80 * s = n * 4)
    (h3 : overall_avg * n = 86 * n)
    (h4 : senior_avg * s = 85 * s)
    (h5 : j * junior_score = overall_avg * n - senior_avg * s) :
    junior_score = 90 :=
by
  sorry

end NUMINAMATH_GPT_junior_score_is_90_l1422_142207


namespace NUMINAMATH_GPT_total_charge_rush_hour_trip_l1422_142245

def initial_fee : ℝ := 2.35
def non_rush_hour_cost_per_two_fifths_mile : ℝ := 0.35
def rush_hour_cost_increase_percentage : ℝ := 0.20
def traffic_delay_cost_per_mile : ℝ := 1.50
def distance_travelled : ℝ := 3.6

theorem total_charge_rush_hour_trip (initial_fee : ℝ) 
  (non_rush_hour_cost_per_two_fifths_mile : ℝ) 
  (rush_hour_cost_increase_percentage : ℝ)
  (traffic_delay_cost_per_mile : ℝ)
  (distance_travelled : ℝ) : 
  initial_fee = 2.35 → 
  non_rush_hour_cost_per_two_fifths_mile = 0.35 →
  rush_hour_cost_increase_percentage = 0.20 →
  traffic_delay_cost_per_mile = 1.50 →
  distance_travelled = 3.6 →
  (initial_fee + ((5/2) * (non_rush_hour_cost_per_two_fifths_mile * (1 + rush_hour_cost_increase_percentage))) * distance_travelled + (traffic_delay_cost_per_mile * distance_travelled)) = 11.53 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_total_charge_rush_hour_trip_l1422_142245


namespace NUMINAMATH_GPT_smallest_possible_fourth_number_l1422_142254

theorem smallest_possible_fourth_number 
  (a b : ℕ) 
  (h1 : 21 + 34 + 65 = 120)
  (h2 : 1 * (21 + 34 + 65 + 10 * a + b) = 4 * (2 + 1 + 3 + 4 + 6 + 5 + a + b)) :
  10 * a + b = 12 := 
sorry

end NUMINAMATH_GPT_smallest_possible_fourth_number_l1422_142254


namespace NUMINAMATH_GPT_volume_ratio_of_spheres_l1422_142214

theorem volume_ratio_of_spheres
  (r1 r2 r3 : ℝ)
  (A1 A2 A3 : ℝ)
  (V1 V2 V3 : ℝ)
  (hA : A1 / A2 = 1 / 4 ∧ A2 / A3 = 4 / 9)
  (hSurfaceArea : A1 = 4 * π * r1^2 ∧ A2 = 4 * π * r2^2 ∧ A3 = 4 * π * r3^2)
  (hVolume : V1 = (4 / 3) * π * r1^3 ∧ V2 = (4 / 3) * π * r2^3 ∧ V3 = (4 / 3) * π * r3^3) :
  V1 / V2 = 1 / 8 ∧ V2 / V3 = 8 / 27 := by
  sorry

end NUMINAMATH_GPT_volume_ratio_of_spheres_l1422_142214


namespace NUMINAMATH_GPT_max_distance_with_optimal_swapping_l1422_142210

-- Define the conditions
def front_tire_lifetime : ℕ := 24000
def rear_tire_lifetime : ℕ := 36000

-- Prove that the maximum distance the car can travel given optimal tire swapping is 48,000 km
theorem max_distance_with_optimal_swapping : 
    ∃ x : ℕ, x < 24000 ∧ x < 36000 ∧ (x + min (24000 - x) (36000 - x) = 48000) :=
by {
  sorry
}

end NUMINAMATH_GPT_max_distance_with_optimal_swapping_l1422_142210


namespace NUMINAMATH_GPT_total_length_of_river_is_80_l1422_142231

-- Definitions based on problem conditions
def straight_part_length := 20
def crooked_part_length := 3 * straight_part_length
def total_length_of_river := straight_part_length + crooked_part_length

-- Theorem stating that the total length of the river is 80 miles
theorem total_length_of_river_is_80 :
  total_length_of_river = 80 := by
    -- The proof is omitted
    sorry

end NUMINAMATH_GPT_total_length_of_river_is_80_l1422_142231


namespace NUMINAMATH_GPT_min_trials_correct_l1422_142209

noncomputable def minimum_trials (α p : ℝ) (hα : 0 < α ∧ α < 1) (hp : 0 < p ∧ p < 1) : ℕ :=
  Nat.floor ((Real.log (1 - α)) / (Real.log (1 - p))) + 1

-- The theorem to prove the correctness of minimum_trials
theorem min_trials_correct (α p : ℝ) (hα : 0 < α ∧ α < 1) (hp : 0 < p ∧ p < 1) :
  ∃ n : ℕ, minimum_trials α p hα hp = n ∧ (1 - (1 - p)^n ≥ α) :=
by
  sorry

end NUMINAMATH_GPT_min_trials_correct_l1422_142209


namespace NUMINAMATH_GPT_determine_a_l1422_142220

theorem determine_a (a p q : ℚ) (h1 : p^2 = a) (h2 : 2 * p * q = 28) (h3 : q^2 = 9) : a = 196 / 9 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_l1422_142220
