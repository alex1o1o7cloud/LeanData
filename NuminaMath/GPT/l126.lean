import Mathlib

namespace amare_additional_fabric_needed_l126_126587

-- Defining the conditions
def yards_per_dress : ℝ := 5.5
def num_dresses : ℝ := 4
def initial_fabric_feet : ℝ := 7
def yard_to_feet : ℝ := 3

-- The theorem to prove
theorem amare_additional_fabric_needed : 
  (yards_per_dress * num_dresses * yard_to_feet) - initial_fabric_feet = 59 := 
by
  sorry

end amare_additional_fabric_needed_l126_126587


namespace jerry_original_butterflies_l126_126243

/-- Define the number of butterflies Jerry originally had -/
def original_butterflies (let_go : ℕ) (now_has : ℕ) : ℕ := let_go + now_has

/-- Given conditions -/
def let_go : ℕ := 11
def now_has : ℕ := 82

/-- Theorem to prove the number of butterflies Jerry originally had -/
theorem jerry_original_butterflies : original_butterflies let_go now_has = 93 :=
by
  sorry

end jerry_original_butterflies_l126_126243


namespace bricks_required_to_pave_courtyard_l126_126339

theorem bricks_required_to_pave_courtyard :
  let courtyard_length_m := 24
  let courtyard_width_m := 14
  let brick_length_cm := 25
  let brick_width_cm := 15
  let courtyard_area_m2 := courtyard_length_m * courtyard_width_m
  let courtyard_area_cm2 := courtyard_area_m2 * 10000
  let brick_area_cm2 := brick_length_cm * brick_width_cm
  let num_bricks := courtyard_area_cm2 / brick_area_cm2
  num_bricks = 8960 := by
  {
    -- Additional context not needed for theorem statement, mock proof omitted
    sorry
  }

end bricks_required_to_pave_courtyard_l126_126339


namespace candies_total_l126_126042

-- Defining the given conditions
def LindaCandies : ℕ := 34
def ChloeCandies : ℕ := 28
def TotalCandies : ℕ := LindaCandies + ChloeCandies

-- Proving the total number of candies
theorem candies_total : TotalCandies = 62 :=
  by
    sorry

end candies_total_l126_126042


namespace sarah_total_height_in_cm_l126_126330

def sarah_height_in_inches : ℝ := 54
def book_thickness_in_inches : ℝ := 2
def conversion_factor : ℝ := 2.54

def total_height_in_inches : ℝ := sarah_height_in_inches + book_thickness_in_inches
def total_height_in_cm : ℝ := total_height_in_inches * conversion_factor

theorem sarah_total_height_in_cm : total_height_in_cm = 142.2 :=
by
  -- Skip the proof for now
  sorry

end sarah_total_height_in_cm_l126_126330


namespace least_value_of_sum_l126_126491

theorem least_value_of_sum (x y z : ℤ) 
  (h_cond : (x - 10) * (y - 5) * (z - 2) = 1000) : x + y + z ≥ 56 :=
sorry

end least_value_of_sum_l126_126491


namespace total_turnips_l126_126311

-- Conditions
def turnips_keith : ℕ := 6
def turnips_alyssa : ℕ := 9

-- Statement to be proved
theorem total_turnips : turnips_keith + turnips_alyssa = 15 := by
  -- Proof is not required for this prompt, so we use sorry
  sorry

end total_turnips_l126_126311


namespace trajectory_midpoint_l126_126285

theorem trajectory_midpoint (P M D : ℝ × ℝ) (hP : P.1 ^ 2 + P.2 ^ 2 = 16) (hD : D = (P.1, 0)) (hM : M = ((P.1 + D.1)/2, (P.2 + D.2)/2)) :
  (M.1 ^ 2) / 4 + (M.2 ^ 2) / 16 = 1 :=
by
  sorry

end trajectory_midpoint_l126_126285


namespace cuboid_to_cube_surface_area_l126_126038

variable (h w l : ℝ)
variable (volume_decreases : 64 = w^3 - w^2 * h)

theorem cuboid_to_cube_surface_area 
  (h w l : ℝ) 
  (cube_condition : w = l ∧ h = w + 4)
  (volume_condition : w^2 * h - w^3 = 64) : 
  (6 * w^2 = 96) :=
by
  sorry

end cuboid_to_cube_surface_area_l126_126038


namespace walking_speed_l126_126410

theorem walking_speed (d : ℝ) (w_speed r_speed : ℝ) (w_time r_time : ℝ)
    (h1 : d = r_speed * r_time)
    (h2 : r_speed = 24)
    (h3 : r_time = 1)
    (h4 : w_time = 3) :
    w_speed = 8 :=
by
  sorry

end walking_speed_l126_126410


namespace base4_to_base10_conversion_l126_126188

theorem base4_to_base10_conversion : 
  2 * 4^4 + 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0 = 582 :=
by 
  sorry

end base4_to_base10_conversion_l126_126188


namespace slope_of_perpendicular_line_l126_126342

theorem slope_of_perpendicular_line (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  ∃ m : ℝ, a * x - b * y = c → m = - (b / a) :=
by
  -- Here we state the definition and conditions provided in the problem
  -- And indicate what we want to prove (that the slope is -b/a in this case)
  sorry

end slope_of_perpendicular_line_l126_126342


namespace units_digit_of_m3_plus_2m_l126_126219

def m : ℕ := 2021^2 + 2^2021

theorem units_digit_of_m3_plus_2m : (m^3 + 2^m) % 10 = 5 := by
  sorry

end units_digit_of_m3_plus_2m_l126_126219


namespace find_a_l126_126430

theorem find_a
  (f : ℝ → ℝ)
  (h₁ : ∀ x, f x = 3 * Real.sin (2 * x - Real.pi / 3))
  (a : ℝ)
  (h₂ : 0 < a)
  (h₃ : a < Real.pi / 2)
  (h₄ : ∀ x, f (x + a) = f (-x + a)) :
  a = 5 * Real.pi / 12 :=
sorry

end find_a_l126_126430


namespace average_age_of_two_new_men_l126_126159

theorem average_age_of_two_new_men :
  ∀ (A N : ℕ), 
    (∀ n : ℕ, n = 12) → 
    (N = 21 + 23 + 12) → 
    (A = N / 2) → 
    A = 28 :=
by
  intros A N twelve men_replace_eq_avg men_avg_eq
  sorry

end average_age_of_two_new_men_l126_126159


namespace minimize_cylinder_surface_area_l126_126051

noncomputable def cylinder_surface_area (r h : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

noncomputable def cylinder_volume (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem minimize_cylinder_surface_area :
  ∃ r h : ℝ, cylinder_volume r h = 16 * Real.pi ∧
  (∀ r' h', cylinder_volume r' h' = 16 * Real.pi → cylinder_surface_area r h ≤ cylinder_surface_area r' h') ∧ r = 2 := by
  sorry

end minimize_cylinder_surface_area_l126_126051


namespace percent_of_y_l126_126113

theorem percent_of_y (y : ℝ) : 0.30 * (0.80 * y) = 0.24 * y :=
by sorry

end percent_of_y_l126_126113


namespace point_reflection_y_l126_126829

def coordinates_with_respect_to_y_axis (x y : ℝ) : ℝ × ℝ :=
  (-x, y)

theorem point_reflection_y (x y : ℝ) (h : (x, y) = (-2, 3)) : coordinates_with_respect_to_y_axis x y = (2, 3) := by
  sorry

end point_reflection_y_l126_126829


namespace sum_and_product_of_roots_l126_126318

theorem sum_and_product_of_roots (m n : ℝ) (h1 : (m / 3) = 9) (h2 : (n / 3) = 20) : m + n = 87 :=
by
  sorry

end sum_and_product_of_roots_l126_126318


namespace binomial_distribution_parameters_l126_126713

noncomputable def E (n : ℕ) (p : ℝ) : ℝ := n * p
noncomputable def D (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem binomial_distribution_parameters (n : ℕ) (p : ℝ) 
  (h1 : E n p = 2.4) (h2 : D n p = 1.44) : 
  n = 6 ∧ p = 0.4 :=
by
  sorry

end binomial_distribution_parameters_l126_126713


namespace painter_total_rooms_l126_126222

theorem painter_total_rooms (hours_per_room : ℕ) (rooms_already_painted : ℕ) (additional_painting_hours : ℕ) 
  (h1 : hours_per_room = 8) (h2 : rooms_already_painted = 8) (h3 : additional_painting_hours = 16) : 
  rooms_already_painted + (additional_painting_hours / hours_per_room) = 10 := by
  sorry

end painter_total_rooms_l126_126222


namespace gcd_lcm_product_l126_126873

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 24) (h2 : b = 45) : (Int.gcd a b * Nat.lcm a b) = 1080 := by
  rw [h1, h2]
  sorry

end gcd_lcm_product_l126_126873


namespace problem1_problem2a_problem2b_l126_126840

-- Problem 1: Deriving y in terms of x
theorem problem1 (x y : ℕ) (h1 : 30 * x + 10 * y = 2000) : y = 200 - 3 * x :=
by sorry

-- Problem 2(a): Minimum ingredient B for at least 220 yuan profit with a=3
theorem problem2a (x y a w : ℕ) (h1 : a = 3) 
  (h2 : 3 * x + 2 * y ≥ 220) (h3 : y = 200 - 3 * x) 
  (h4 : w = 15 * x + 20 * y) : w = 1300 :=
by sorry

-- Problem 2(b): Profit per portion of dessert A for 450 yuan profit with 3100 grams of B
theorem problem2b (x : ℕ) (a : ℕ) (B : ℕ) 
  (h1 : B = 3100) (h2 : 15 * x + 20 * (200 - 3 * x) ≤ B) 
  (h3 : a * x + 2 * (200 - 3 * x) = 450) 
  (h4 : x ≥ 20) : a = 8 :=
by sorry

end problem1_problem2a_problem2b_l126_126840


namespace cube_volume_given_face_area_l126_126684

theorem cube_volume_given_face_area (s : ℝ) (h : s^2 = 36) : s^3 = 216 := by
  sorry

end cube_volume_given_face_area_l126_126684


namespace sum_of_arithmetic_series_l126_126819

def a₁ : ℕ := 9
def d : ℕ := 4
def n : ℕ := 50

noncomputable def nth_term (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d
noncomputable def sum_arithmetic_series (a₁ d n : ℕ) : ℕ := n / 2 * (a₁ + nth_term a₁ d n)

theorem sum_of_arithmetic_series :
  sum_arithmetic_series a₁ d n = 5350 :=
by
  sorry

end sum_of_arithmetic_series_l126_126819


namespace find_x_range_l126_126823

theorem find_x_range : 
  {x : ℝ | (2 / (x + 2) + 4 / (x + 8) ≤ 3 / 4)} = 
  {x : ℝ | (-4 < x ∧ x ≤ -2) ∨ (4 ≤ x)} := by
  sorry

end find_x_range_l126_126823


namespace determine_radii_l126_126837

-- Definitions based on conditions from a)
variable (S1 S2 S3 S4 : Type) -- Centers of the circles
variable (dist_S2_S4 : ℝ) (dist_S1_S2 : ℝ) (dist_S2_S3 : ℝ) (dist_S3_S4 : ℝ)
variable (r1 r2 r3 r4 : ℝ) -- Radii of circles k1, k2, k3, and k4
variable (rhombus : Prop) -- Quadrilateral S1S2S3S4 is a rhombus

-- Given conditions
axiom C1 : ∀ t : S1, r1 = 5
axiom C2 : dist_S2_S4 = 24
axiom C3 : rhombus

-- Equivalency to be proven
theorem determine_radii : 
  r2 = 12 ∧ r4 = 12 ∧ r1 = 5 ∧ r3 = 5 :=
sorry

end determine_radii_l126_126837


namespace six_digit_number_count_correct_l126_126175

-- Defining the 6-digit number formation problem
def count_six_digit_numbers_with_conditions : Nat := 1560

-- Problem statement
theorem six_digit_number_count_correct :
  count_six_digit_numbers_with_conditions = 1560 :=
sorry

end six_digit_number_count_correct_l126_126175


namespace find_A_in_triangle_l126_126379

theorem find_A_in_triangle
  (a b : ℝ) (B A : ℝ)
  (h₀ : a = Real.sqrt 3)
  (h₁ : b = Real.sqrt 2)
  (h₂ : B = Real.pi / 4)
  (h₃ : a / Real.sin A = b / Real.sin B) :
  A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end find_A_in_triangle_l126_126379


namespace series_value_is_correct_l126_126619

noncomputable def check_series_value : ℚ :=
  let p : ℚ := 1859 / 84
  let q : ℚ := -1024 / 63
  let r : ℚ := 512 / 63
  let m : ℕ := 3907
  let n : ℕ := 84
  100 * m + n

theorem series_value_is_correct : check_series_value = 390784 := 
by 
  sorry

end series_value_is_correct_l126_126619


namespace factorial_sum_power_of_two_l126_126522

theorem factorial_sum_power_of_two (a b c n : ℕ) (h : a ≤ b ∧ b ≤ c) :
  a! + b! + c! = 2^n →
  (a = 1 ∧ b = 1 ∧ c = 2) ∨
  (a = 1 ∧ b = 1 ∧ c = 3) ∨
  (a = 2 ∧ b = 3 ∧ c = 4) ∨
  (a = 2 ∧ b = 3 ∧ c = 5) :=
by
  sorry

end factorial_sum_power_of_two_l126_126522


namespace find_b_l126_126747

-- Define the slopes of the two lines derived from the given conditions
noncomputable def slope1 := -2 / 3
noncomputable def slope2 (b : ℚ) := -b / 3

-- Lean 4 statement to prove that for the lines to be perpendicular, b must be -9/2
theorem find_b (b : ℚ) (h_perpendicular: slope1 * slope2 b = -1) : b = -9 / 2 := by
  sorry

end find_b_l126_126747


namespace earl_stuff_rate_l126_126365

variable (E L : ℕ)

-- Conditions
def ellen_rate : Prop := L = (2 * E) / 3
def combined_rate : Prop := E + L = 60

-- Main statement
theorem earl_stuff_rate (h1 : ellen_rate E L) (h2 : combined_rate E L) : E = 36 := by
  sorry

end earl_stuff_rate_l126_126365


namespace sara_steps_l126_126795

theorem sara_steps (n : ℕ) (h : n^2 ≤ 210) : n = 14 :=
sorry

end sara_steps_l126_126795


namespace min_xy_eq_nine_l126_126556

theorem min_xy_eq_nine (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y = x + y + 3) : x * y = 9 :=
sorry

end min_xy_eq_nine_l126_126556


namespace number_division_equals_value_l126_126078

theorem number_division_equals_value (x : ℝ) (h : x / 0.144 = 14.4 / 0.0144) : x = 144 :=
by
  sorry

end number_division_equals_value_l126_126078


namespace overall_percentage_l126_126076

theorem overall_percentage (s1 s2 s3 : ℝ) (h1 : s1 = 60) (h2 : s2 = 80) (h3 : s3 = 85) :
  (s1 + s2 + s3) / 3 = 75 := by
  sorry

end overall_percentage_l126_126076


namespace erased_number_is_one_or_twenty_l126_126565

theorem erased_number_is_one_or_twenty (x : ℕ) (h₁ : 1 ≤ x ∧ x ≤ 20)
  (h₂ : (210 - x) % 19 = 0) : x = 1 ∨ x = 20 :=
  by sorry

end erased_number_is_one_or_twenty_l126_126565


namespace height_of_building_l126_126321

def flagpole_height : ℝ := 18
def flagpole_shadow_length : ℝ := 45

def building_shadow_length : ℝ := 65
def building_height : ℝ := 26

theorem height_of_building
  (hflagpole : flagpole_height / flagpole_shadow_length = building_height / building_shadow_length) :
  building_height = 26 :=
sorry

end height_of_building_l126_126321


namespace product_of_sums_of_four_squares_is_sum_of_four_squares_l126_126611

theorem product_of_sums_of_four_squares_is_sum_of_four_squares (x1 x2 x3 x4 y1 y2 y3 y4 : ℤ) :
  let a := x1^2 + x2^2 + x3^2 + x4^2
  let b := y1^2 + y2^2 + y3^2 + y4^2
  let z1 := x1 * y1 + x2 * y2 + x3 * y3 + x4 * y4
  let z2 := x1 * y2 - x2 * y1 + x3 * y4 - x4 * y3
  let z3 := x1 * y3 - x3 * y1 + x4 * y2 - x2 * y4
  let z4 := x1 * y4 - x4 * y1 + x2 * y3 - x3 * y2
  a * b = z1^2 + z2^2 + z3^2 + z4^2 :=
by
  sorry

end product_of_sums_of_four_squares_is_sum_of_four_squares_l126_126611


namespace fred_more_than_daniel_l126_126101

-- Definitions and conditions from the given problem.
def total_stickers : ℕ := 750
def andrew_kept : ℕ := 130
def daniel_received : ℕ := 250
def fred_received : ℕ := total_stickers - andrew_kept - daniel_received

-- The proof problem statement.
theorem fred_more_than_daniel : fred_received - daniel_received = 120 := by 
  sorry

end fred_more_than_daniel_l126_126101


namespace solve_inequality_l126_126890

theorem solve_inequality (x : ℝ) :
  (x^2 - 4 * x - 12) / (x - 3) < 0 ↔ (-2 < x ∧ x < 3) ∨ (3 < x ∧ x < 6) := by
  sorry

end solve_inequality_l126_126890


namespace find_real_num_l126_126987

noncomputable def com_num (a : ℝ) : ℂ := (a + 3 * Complex.I) / (1 + 2 * Complex.I)

theorem find_real_num (a : ℝ) : (∃ b : ℝ, com_num a = b * Complex.I) → a = -6 :=
by
  sorry

end find_real_num_l126_126987


namespace tangent_line_at_point_l126_126920

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then (Real.exp (-(x - 1)) - x) else (Real.exp (x - 1) + x)

theorem tangent_line_at_point (f_even : ∀ x : ℝ, f x = f (-x)) :
    ∀ (x y : ℝ), x = 1 → y = 2 → (∃ m b : ℝ, y = m * x + b ∧ m = 2 ∧ b = 0) := by
  sorry

end tangent_line_at_point_l126_126920


namespace max_ab_is_5_l126_126924

noncomputable def max_ab : ℝ :=
  sorry

theorem max_ab_is_5 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h : a / 4 + b / 5 = 1) : max_ab = 5 :=
  sorry

end max_ab_is_5_l126_126924


namespace fred_gave_balloons_to_sandy_l126_126612

-- Define the number of balloons Fred originally had
def original_balloons : ℕ := 709

-- Define the number of balloons Fred has now
def current_balloons : ℕ := 488

-- Define the number of balloons Fred gave to Sandy
def balloons_given := original_balloons - current_balloons

-- Theorem: The number of balloons given to Sandy is 221
theorem fred_gave_balloons_to_sandy : balloons_given = 221 :=
by
  sorry

end fred_gave_balloons_to_sandy_l126_126612


namespace second_number_deduction_l126_126252

theorem second_number_deduction
  (x : ℝ)
  (h1 : (10 * 16 = 10 * x + (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9)))
  (h2 : 2.5 + (x+1 - y) + 6.5 + 8.5 + 10.5 + 12.5 + 14.5 + 16.5 + 18.5 + 20.5 = 115)
  : y = 8 :=
by
  -- This is where the proof would go, but we'll leave it as 'sorry' for now.
  sorry

end second_number_deduction_l126_126252


namespace minimum_correct_answers_l126_126798

/-
There are a total of 20 questions. Answering correctly scores 10 points, while answering incorrectly or not answering deducts 5 points. 
To pass, one must score no less than 80 points. Xiao Ming passed the selection. Prove that the minimum number of questions Xiao Ming 
must have answered correctly is no less than 12.
-/

theorem minimum_correct_answers (total_questions correct_points incorrect_points pass_score : ℕ)
  (h1 : total_questions = 20)
  (h2 : correct_points = 10)
  (h3 : incorrect_points = 5)
  (h4 : pass_score = 80)
  (h_passed : ∃ x : ℕ, x ≤ total_questions ∧ (correct_points * x - incorrect_points * (total_questions - x)) ≥ pass_score) :
  ∃ x : ℕ, x ≥ 12 ∧ (correct_points * x - incorrect_points * (total_questions - x)) ≥ pass_score := 
sorry

end minimum_correct_answers_l126_126798


namespace div_inside_parentheses_l126_126641

theorem div_inside_parentheses :
  100 / (6 / 2) = 100 / 3 :=
by
  sorry

end div_inside_parentheses_l126_126641


namespace arithmetic_sequence_difference_l126_126039

theorem arithmetic_sequence_difference (a b c : ℤ) (d : ℤ)
  (h1 : 9 - 1 = 4 * d)
  (h2 : c - a = 2 * d) :
  c - a = 4 := by sorry

end arithmetic_sequence_difference_l126_126039


namespace div_by_133_l126_126771

theorem div_by_133 (n : ℕ) : 133 ∣ 11^(n+2) + 12^(2*n+1) :=
by sorry

end div_by_133_l126_126771


namespace geometric_sequence_sum_range_l126_126232

theorem geometric_sequence_sum_range {a : ℕ → ℝ}
  (h4_8: a 4 * a 8 = 9) :
  a 3 + a 9 ∈ Set.Iic (-6) ∪ Set.Ici 6 :=
sorry

end geometric_sequence_sum_range_l126_126232


namespace microorganism_half_filled_time_l126_126463

theorem microorganism_half_filled_time :
  (∀ x, 2^x = 2^9 ↔ x = 9) :=
by
  sorry

end microorganism_half_filled_time_l126_126463


namespace positive_integers_ab_divides_asq_bsq_implies_a_eq_b_l126_126547

theorem positive_integers_ab_divides_asq_bsq_implies_a_eq_b
  (a b : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hdiv : a * b ∣ a^2 + b^2) : a = b := by
  sorry

end positive_integers_ab_divides_asq_bsq_implies_a_eq_b_l126_126547


namespace wharf_length_l126_126371

-- Define the constants
def avg_speed := 2 -- average speed in m/s
def travel_time := 16 -- travel time in seconds

-- Define the formula to calculate length of the wharf
def length_of_wharf := 2 * avg_speed * travel_time

-- The goal is to prove that length_of_wharf equals 64
theorem wharf_length : length_of_wharf = 64 :=
by
  -- Proof would be here
  sorry

end wharf_length_l126_126371


namespace prove_inequality_l126_126351

noncomputable def valid_x (x : ℝ) : Prop :=
  x ≠ 0 ∧ x ≠ (1-Real.sqrt 5)/2 ∧ x ≠ (1+Real.sqrt 5)/2

noncomputable def valid_intervals (x : ℝ) : Prop :=
  (x ≥ -1 ∧ x < (1 - Real.sqrt 5) / 2) ∨
  ((1 - Real.sqrt 5) / 2 < x ∧ x < 0) ∨
  (0 < x ∧ x < (1 + Real.sqrt 5) / 2) ∨
  (x > (1 + Real.sqrt 5) / 2)

theorem prove_inequality (x : ℝ) (hx : valid_x x) :
  (x^2 + x^3 - x^4) / (x + x^2 - x^3) ≥ -1 ↔ valid_intervals x := by
  sorry

end prove_inequality_l126_126351


namespace trig_identity_l126_126888

variable {α : Real}

theorem trig_identity (h : Real.tan α = 3) : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 := 
by
  sorry

end trig_identity_l126_126888


namespace find_k_l126_126348

theorem find_k (Z K : ℤ) (h1 : 2000 < Z) (h2 : Z < 3000) (h3 : K > 1) (h4 : Z = K * K^2) (h5 : ∃ n : ℤ, n^3 = Z) : K = 13 :=
by
-- Solution omitted
sorry

end find_k_l126_126348


namespace contrapositive_proposition_l126_126944

theorem contrapositive_proposition (x a b : ℝ) : (x < 2 * a * b) → (x < a^2 + b^2) :=
sorry

end contrapositive_proposition_l126_126944


namespace no_students_unable_to_partner_l126_126618

def students_males_females :=
  let males_6th_class1 : Nat := 17
  let females_6th_class1 : Nat := 13
  let males_6th_class2 : Nat := 14
  let females_6th_class2 : Nat := 18
  let males_6th_class3 : Nat := 15
  let females_6th_class3 : Nat := 17
  let males_7th_class : Nat := 22
  let females_7th_class : Nat := 20

  let total_males := males_6th_class1 + males_6th_class2 + males_6th_class3 + males_7th_class
  let total_females := females_6th_class1 + females_6th_class2 + females_6th_class3 + females_7th_class

  total_males == total_females

theorem no_students_unable_to_partner : students_males_females = true := by
  -- Skipping the proof
  sorry

end no_students_unable_to_partner_l126_126618


namespace intercepts_of_line_l126_126625

theorem intercepts_of_line (x y : ℝ) (h_eq : 4 * x + 7 * y = 28) :
  (∃ y, (x = 0 ∧ y = 4) ∧ ∃ x, (y = 0 ∧ x = 7)) :=
by
  sorry

end intercepts_of_line_l126_126625


namespace c_geq_one_l126_126239

theorem c_geq_one {a b : ℕ} {c : ℝ} (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (h : (a + 1) / (b + c) = b / a) : 1 ≤ c :=
  sorry

end c_geq_one_l126_126239


namespace eesha_late_by_15_minutes_l126_126584

theorem eesha_late_by_15_minutes 
  (T usual_time : ℕ) (delay : ℕ) (slower_factor : ℚ) (T' : ℕ) 
  (usual_time_eq : usual_time = 60)
  (delay_eq : delay = 30)
  (slower_factor_eq : slower_factor = 0.75)
  (new_time_eq : T' = unusual_time * slower_factor) 
  (T'' : ℕ) (total_time_eq: T'' = T' + delay)
  (time_taken : ℕ) (time_diff_eq : time_taken = T'' - usual_time) :
  time_taken = 15 :=
by
  -- Proof construction
  sorry

end eesha_late_by_15_minutes_l126_126584


namespace batsman_average_after_12th_innings_l126_126108

theorem batsman_average_after_12th_innings 
  (A : ℕ) 
  (h1 : 75 = (A + 12)) 
  (h2 : 11 * A + 75 = 12 * (A + 1)) :
  (A + 1) = 64 :=
by 
  sorry

end batsman_average_after_12th_innings_l126_126108


namespace number_of_distinct_triangles_l126_126417

-- Definition of the grid
def grid_points : List (ℕ × ℕ) := 
  [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1)]

-- Definition involving combination logic
def binomial (n k : ℕ) : ℕ := n.choose k

-- Count all possible combinations of 3 points
def total_combinations : ℕ := binomial 8 3

-- Count the degenerate cases (collinear points) in the grid
def degenerate_cases : ℕ := 2 * binomial 4 3

-- The required value of distinct triangles
def distinct_triangles : ℕ := total_combinations - degenerate_cases

theorem number_of_distinct_triangles :
  distinct_triangles = 48 :=
by
  sorry

end number_of_distinct_triangles_l126_126417


namespace correct_operation_l126_126878

theorem correct_operation (a : ℝ) (h : a ≠ 0) : a * a⁻¹ = 1 :=
by
  sorry

end correct_operation_l126_126878


namespace perfect_square_quotient_l126_126838

theorem perfect_square_quotient (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h : (a * b + 1) ∣ (a * a + b * b)) : 
  ∃ k : ℕ, (a * a + b * b) = (a * b + 1) * (k * k) := 
sorry

end perfect_square_quotient_l126_126838


namespace correct_sentence_is_D_l126_126948

-- Define the sentences as strings
def sentence_A : String :=
  "Between any two adjacent integers on the number line, an infinite number of fractions can be inserted to fill the gaps on the number line; mathematicians once thought that with this approach, the entire number line was finally filled."

def sentence_B : String :=
  "With zero as the center, all integers are arranged from right to left at equal distances, and then connected with a horizontal line; this is what we call the 'number line'."

def sentence_C : String :=
  "The vast collection of books in the Beijing Library contains an enormous amount of information, but it is still finite, whereas the number pi contains infinite information, which is awe-inspiring."

def sentence_D : String :=
  "Pi is fundamentally the exact ratio of a circle's circumference to its diameter, but the infinite sequence it produces has the greatest uncertainty; we cannot help but be amazed and shaken by the marvel and mystery of nature."

-- Define the problem statement
theorem correct_sentence_is_D :
  sentence_D ≠ "" := by
  sorry

end correct_sentence_is_D_l126_126948


namespace cyclist_arrives_first_l126_126728

-- Definitions based on given conditions
def speed_cyclist (v : ℕ) := v
def speed_motorist (v : ℕ) := 5 * v

def distance_total (d : ℕ) := d
def distance_half (d : ℕ) := d / 2

def time_motorist_first_half (d v : ℕ) : ℕ := distance_half d / speed_motorist v

def remaining_distance_cyclist (d v : ℕ) := d - v * time_motorist_first_half d v

def speed_motorist_walking (v : ℕ) := v / 2

def time_motorist_second_half (d v : ℕ) := distance_half d / speed_motorist_walking v
def time_cyclist_remaining (d v : ℕ) : ℕ := remaining_distance_cyclist d v / speed_cyclist v

-- Comparison to prove cyclist arrives first
theorem cyclist_arrives_first (d v : ℕ) (hv : 0 < v) (hd : 0 < d) :
  time_cyclist_remaining d v < time_motorist_second_half d v :=
by sorry

end cyclist_arrives_first_l126_126728


namespace carbon_paper_count_l126_126036

theorem carbon_paper_count (x : ℕ) (sheets : ℕ) (copies : ℕ) (h1 : sheets = 3) (h2 : copies = 2) :
  x = 1 :=
sorry

end carbon_paper_count_l126_126036


namespace simplify_expression_l126_126723

theorem simplify_expression (t : ℝ) (t_ne_zero : t ≠ 0) : (t^5 * t^3) / t^4 = t^4 := 
by
  sorry

end simplify_expression_l126_126723


namespace include_both_male_and_female_l126_126789

noncomputable def probability_includes_both_genders (total_students male_students female_students selected_students : ℕ) : ℚ :=
  let total_ways := Nat.choose total_students selected_students
  let all_female_ways := Nat.choose female_students selected_students
  (total_ways - all_female_ways) / total_ways

theorem include_both_male_and_female :
  probability_includes_both_genders 6 2 4 4 = 14 / 15 := 
by
  sorry

end include_both_male_and_female_l126_126789


namespace clips_and_earnings_l126_126647

variable (x y z : ℝ)
variable (h_y : y = x / 2)
variable (totalClips : ℝ := 48 * x + y)
variable (avgEarning : ℝ := z / totalClips)

theorem clips_and_earnings :
  totalClips = 97 * x / 2 ∧ avgEarning = 2 * z / (97 * x) :=
by
  sorry

end clips_and_earnings_l126_126647


namespace isabella_euros_l126_126544

theorem isabella_euros (d : ℝ) : 
  (5 / 8) * d - 80 = 2 * d → d = 58 :=
by
  sorry

end isabella_euros_l126_126544


namespace adjacent_girl_pairs_l126_126152

variable (boyCount girlCount : ℕ) 
variable (adjacentBoyPairs adjacentGirlPairs: ℕ)

theorem adjacent_girl_pairs
  (h1 : boyCount = 10)
  (h2 : girlCount = 15)
  (h3 : adjacentBoyPairs = 5) :
  adjacentGirlPairs = 10 :=
sorry

end adjacent_girl_pairs_l126_126152


namespace equations_not_equivalent_l126_126469

theorem equations_not_equivalent :
  (∀ x, (2 * (x - 10) / (x^2 - 13 * x + 30) = 1 ↔ x = 5)) ∧ 
  (∃ x, x ≠ 5 ∧ (x^2 - 15 * x + 50 = 0)) :=
sorry

end equations_not_equivalent_l126_126469


namespace probability_at_least_one_succeeds_l126_126905

variable (p1 p2 : ℝ)

theorem probability_at_least_one_succeeds : 
  0 ≤ p1 ∧ p1 ≤ 1 → 0 ≤ p2 ∧ p2 ≤ 1 → (1 - (1 - p1) * (1 - p2)) = 1 - (1 - p1) * (1 - p2) :=
by 
  intro h1 h2
  sorry

end probability_at_least_one_succeeds_l126_126905


namespace find_coordinates_of_D_l126_126534

theorem find_coordinates_of_D
  (A B C D : ℝ × ℝ)
  (hA : A = (-1, 2))
  (hB : B = (0, 0))
  (hC : C = (1, 7))
  (hParallelogram : ∃ u v, u * (B - A) + v * (C - D) = (0, 0) ∧ u * (C - D) + v * (B - A) = (0, 0)) :
  D = (0, 9) :=
sorry

end find_coordinates_of_D_l126_126534


namespace integer_solution_count_l126_126858

theorem integer_solution_count (x : ℤ) : (12 * x - 1) * (6 * x - 1) * (4 * x - 1) * (3 * x - 1) = 330 ↔ x = 1 :=
by
  sorry

end integer_solution_count_l126_126858


namespace factor_theorem_l126_126964

theorem factor_theorem (t : ℝ) : (5 * t^2 + 15 * t - 20 = 0) ↔ (t = 1 ∨ t = -4) :=
by
  sorry

end factor_theorem_l126_126964


namespace smallest_positive_a_integer_root_l126_126411

theorem smallest_positive_a_integer_root :
  ∀ x a : ℚ, (exists x : ℚ, (x > 0) ∧ (a > 0) ∧ 
    (
      ((x - a) / 2 + (x - 2 * a) / 3) / ((x + 4 * a) / 5 - (x + 3 * a) / 4) =
      ((x - 3 * a) / 4 + (x - 4 * a) / 5) / ((x + 2 * a) / 3 - (x + a) / 2)
    )
  ) → a = 419 / 421 :=
by sorry

end smallest_positive_a_integer_root_l126_126411


namespace ticket_price_difference_l126_126530

noncomputable def price_difference (adult_price total_cost : ℕ) (num_adults num_children : ℕ) (child_price : ℕ) : ℕ :=
  adult_price - child_price

theorem ticket_price_difference :
  ∀ (adult_price total_cost num_adults num_children child_price : ℕ),
  adult_price = 19 →
  total_cost = 77 →
  num_adults = 2 →
  num_children = 3 →
  num_adults * adult_price + num_children * child_price = total_cost →
  price_difference adult_price total_cost num_adults num_children child_price = 6 :=
by
  intros
  simp [price_difference]
  sorry

end ticket_price_difference_l126_126530


namespace no_solution_iff_a_leq_8_l126_126887

theorem no_solution_iff_a_leq_8 (a : ℝ) :
  (¬ ∃ x : ℝ, |x - 5| + |x + 3| < a) ↔ a ≤ 8 := 
sorry

end no_solution_iff_a_leq_8_l126_126887


namespace fraction_of_peaches_l126_126328

-- Define the number of peaches each person has
def Benjy_peaches : ℕ := 5
def Martine_peaches : ℕ := 16
def Gabrielle_peaches : ℕ := 15

-- Condition that Martine has 6 more than twice Benjy's peaches
def Martine_cond : Prop := Martine_peaches = 2 * Benjy_peaches + 6

-- The goal is to prove the fraction of Gabrielle's peaches that Benjy has
theorem fraction_of_peaches :
  Martine_cond → (Benjy_peaches : ℚ) / (Gabrielle_peaches : ℚ) = 1 / 3 :=
by
  -- Assuming the condition holds
  intro h
  rw [Martine_cond] at h
  -- Use the condition directly, since Martine_cond implies Benjy_peaches = 5
  exact sorry

end fraction_of_peaches_l126_126328


namespace solve_for_x_l126_126839

theorem solve_for_x (x : ℝ) (h : 7 - 2 * x = -3) : x = 5 := by
  sorry

end solve_for_x_l126_126839


namespace solve_eq1_solve_eq2_l126_126841

-- Define the first proof problem
theorem solve_eq1 (x : ℝ) : 2 * x - 3 = 3 * (x + 1) → x = -6 :=
by
  sorry

-- Define the second proof problem
theorem solve_eq2 (x : ℝ) : (1 / 2) * x - (9 * x - 2) / 6 - 2 = 0 → x = -5 / 3 :=
by
  sorry

end solve_eq1_solve_eq2_l126_126841


namespace symmetric_point_line_l126_126326

theorem symmetric_point_line (a b : ℝ) :
  (∀ (x y : ℝ), (y - 2) / (x - 1) = -2 → (x + 1)/2 + 2 * (y + 2)/2 - 10 = 0) →
  a = 3 ∧ b = 6 := by
  intro h
  sorry

end symmetric_point_line_l126_126326


namespace xy_fraction_l126_126041

theorem xy_fraction (x y : ℚ) (h1 : 1 / x + 1 / y = 4) (h2 : 1 / x - 1 / y = -6) :
  x * y = -1 / 5 := 
by sorry

end xy_fraction_l126_126041


namespace quadratic_function_value_2_l126_126826

variables (a b : ℝ)
def f (x : ℝ) : ℝ := x^2 + a * x + b

theorem quadratic_function_value_2 :
  f a b 2 = 3 :=
by
  -- Definitions and assumptions to be used
  sorry

end quadratic_function_value_2_l126_126826


namespace find_unknown_rate_of_two_blankets_l126_126067

-- Definitions of conditions based on the problem statement
def purchased_blankets_at_100 : Nat := 3
def price_per_blanket_at_100 : Nat := 100
def total_cost_at_100 := purchased_blankets_at_100 * price_per_blanket_at_100

def purchased_blankets_at_150 : Nat := 3
def price_per_blanket_at_150 : Nat := 150
def total_cost_at_150 := purchased_blankets_at_150 * price_per_blanket_at_150

def purchased_blankets_at_x : Nat := 2
def blankets_total : Nat := 8
def average_price : Nat := 150
def total_cost := blankets_total * average_price

-- The proof statement
theorem find_unknown_rate_of_two_blankets (x : Nat) 
  (h : purchased_blankets_at_100 * price_per_blanket_at_100 + 
       purchased_blankets_at_150 * price_per_blanket_at_150 + 
       purchased_blankets_at_x * x = total_cost) : x = 225 :=
by sorry

end find_unknown_rate_of_two_blankets_l126_126067


namespace arithmetic_mean_of_geometric_sequence_l126_126915

theorem arithmetic_mean_of_geometric_sequence (a r : ℕ) (h_a : a = 4) (h_r : r = 3) :
    ((a) + (a * r) + (a * r^2)) / 3 = (52 / 3) :=
by
  sorry

end arithmetic_mean_of_geometric_sequence_l126_126915


namespace cost_of_1500_pieces_of_gum_in_dollars_l126_126230

theorem cost_of_1500_pieces_of_gum_in_dollars :
  (2 * 1500 * (1 - 0.10) / 100) = 27 := sorry

end cost_of_1500_pieces_of_gum_in_dollars_l126_126230


namespace variation_relationship_l126_126670

theorem variation_relationship (k j : ℝ) (y z x : ℝ) (h1 : x = k * y^3) (h2 : y = j * z^(1/5)) :
  ∃ m : ℝ, x = m * z^(3/5) :=
by
  sorry

end variation_relationship_l126_126670


namespace time_to_send_data_in_minutes_l126_126138

def blocks := 100
def chunks_per_block := 256
def transmission_rate := 100 -- chunks per second
def seconds_per_minute := 60

theorem time_to_send_data_in_minutes :
    (blocks * chunks_per_block) / transmission_rate / seconds_per_minute = 4 := by
  sorry

end time_to_send_data_in_minutes_l126_126138


namespace solve_for_n_l126_126249

variable (n : ℚ)

theorem solve_for_n (h : 22 + Real.sqrt (-4 + 18 * n) = 24) : n = 4 / 9 := by
  sorry

end solve_for_n_l126_126249


namespace perpendicular_lines_l126_126344

theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, (a * x - y + 2 * a = 0) → ((2 * a - 1) * x + a * y + a = 0) -> 
  (a ≠ 0 → ∃ k : ℝ, k = (a * ((1 - 2 * a) / a)) ∧ k = -1) -> a * ((1 - 2 * a) / a) = -1) →
  a = 0 ∨ a = 1 := by sorry

end perpendicular_lines_l126_126344


namespace circle_condition_l126_126368

theorem circle_condition (m : ℝ): (∃ x y : ℝ, (x^2 + y^2 - 2*x - 4*y + m = 0)) ↔ (m < 5) :=
by
  sorry

end circle_condition_l126_126368


namespace defect_free_product_probability_is_correct_l126_126553

noncomputable def defect_free_probability : ℝ :=
  let p1 := 0.2
  let p2 := 0.3
  let p3 := 0.5
  let d1 := 0.95
  let d2 := 0.90
  let d3 := 0.80
  p1 * d1 + p2 * d2 + p3 * d3

theorem defect_free_product_probability_is_correct :
  defect_free_probability = 0.86 :=
by
  sorry

end defect_free_product_probability_is_correct_l126_126553


namespace chickens_at_stacy_farm_l126_126155
-- Importing the necessary library

-- Defining the provided conditions and correct answer in Lean 4.
theorem chickens_at_stacy_farm (C : ℕ) (piglets : ℕ) (goats : ℕ) : 
  piglets = 40 → 
  goats = 34 → 
  (C + piglets + goats) = 2 * 50 → 
  C = 26 :=
by
  intros h_piglets h_goats h_animals
  sorry

end chickens_at_stacy_farm_l126_126155


namespace hockey_season_games_l126_126043

theorem hockey_season_games (n_teams : ℕ) (n_faces : ℕ) (h1 : n_teams = 18) (h2 : n_faces = 10) :
  let total_games := (n_teams * (n_teams - 1) / 2) * n_faces
  total_games = 1530 :=
by
  sorry

end hockey_season_games_l126_126043


namespace discriminant_of_quadratic_5x2_minus_2x_minus_7_l126_126031

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b ^ 2 - 4 * a * c

theorem discriminant_of_quadratic_5x2_minus_2x_minus_7 :
  quadratic_discriminant 5 (-2) (-7) = 144 :=
by
  sorry

end discriminant_of_quadratic_5x2_minus_2x_minus_7_l126_126031


namespace train_crossing_time_l126_126764

def train_length := 140
def train_speed_kmph := 45
def bridge_length := 235
def speed_to_mps (kmph : ℕ) : ℕ := (kmph * 1000) / 3600
def total_distance := train_length + bridge_length
def train_speed := speed_to_mps train_speed_kmph
def time_to_cross := total_distance / train_speed

theorem train_crossing_time : time_to_cross = 30 := by
  sorry

end train_crossing_time_l126_126764


namespace Raja_and_Ram_together_l126_126070

def RajaDays : ℕ := 12
def RamDays : ℕ := 6

theorem Raja_and_Ram_together (W : ℕ) : 
  let RajaRate := W / RajaDays
  let RamRate := W / RamDays
  let CombinedRate := RajaRate + RamRate 
  let DaysTogether := W / CombinedRate 
  DaysTogether = 4 := 
by
  sorry

end Raja_and_Ram_together_l126_126070


namespace steve_oranges_count_l126_126096

variable (Brian_oranges Marcie_oranges Shawn_oranges Steve_oranges : ℝ)

def oranges_conditions : Prop :=
  (Marcie_oranges = 12) ∧
  (Brian_oranges = Marcie_oranges) ∧
  (Shawn_oranges = 1.075 * (Brian_oranges + Marcie_oranges)) ∧
  (Steve_oranges = 3 * (Marcie_oranges + Brian_oranges + Shawn_oranges))

theorem steve_oranges_count (h : oranges_conditions Brian_oranges Marcie_oranges Shawn_oranges Steve_oranges) :
  Steve_oranges = 149.4 :=
sorry

end steve_oranges_count_l126_126096


namespace Phil_quarters_l126_126941

theorem Phil_quarters (initial_amount : ℝ)
  (pizza : ℝ) (soda : ℝ) (jeans : ℝ) (book : ℝ) (gum : ℝ) (ticket : ℝ)
  (quarter_value : ℝ) (spent := pizza + soda + jeans + book + gum + ticket)
  (remaining := initial_amount - spent)
  (quarters := remaining / quarter_value) :
  initial_amount = 40 ∧ pizza = 2.75 ∧ soda = 1.50 ∧ jeans = 11.50 ∧
  book = 6.25 ∧ gum = 1.75 ∧ ticket = 8.50 ∧ quarter_value = 0.25 →
  quarters = 31 :=
by
  intros
  sorry

end Phil_quarters_l126_126941


namespace problem_statement_l126_126856

-- Define the basic problem setup
def defect_rate (p : ℝ) := p = 0.01
def sample_size (n : ℕ) := n = 200

-- Define the binomial distribution
noncomputable def binomial_expectation (n : ℕ) (p : ℝ) := n * p
noncomputable def binomial_variance (n : ℕ) (p : ℝ) := n * p * (1 - p)

-- The actual statement that we will prove
theorem problem_statement (p : ℝ) (n : ℕ) (X : ℕ → ℕ) 
  (h_defect_rate : defect_rate p) 
  (h_sample_size : sample_size n) 
  (h_distribution : ∀ k, X k = (n.choose k) * (p ^ k) * ((1 - p) ^ (n - k))) 
  : binomial_expectation n p = 2 ∧ binomial_variance n p = 1.98 :=
by
  sorry

end problem_statement_l126_126856


namespace find_extrema_of_f_l126_126092

noncomputable def f (x : ℝ) := x^2 - 4 * x - 2

theorem find_extrema_of_f : 
  (∀ x, (1 ≤ x ∧ x ≤ 4) → f x ≤ -2) ∧ 
  (∃ x, (1 ≤ x ∧ x ≤ 4 ∧ f x = -6)) :=
by sorry

end find_extrema_of_f_l126_126092


namespace tax_amount_is_correct_l126_126182

def camera_cost : ℝ := 200.00
def tax_rate : ℝ := 0.15

theorem tax_amount_is_correct :
  (camera_cost * tax_rate) = 30.00 :=
sorry

end tax_amount_is_correct_l126_126182


namespace percentage_is_36_point_4_l126_126446

def part : ℝ := 318.65
def whole : ℝ := 875.3

theorem percentage_is_36_point_4 : (part / whole) * 100 = 36.4 := 
by sorry

end percentage_is_36_point_4_l126_126446


namespace complex_numbers_satisfying_conditions_l126_126389

theorem complex_numbers_satisfying_conditions (x y z : ℂ) 
  (h1 : x + y + z = 3) 
  (h2 : x^2 + y^2 + z^2 = 3) 
  (h3 : x^3 + y^3 + z^3 = 3) : x = 1 ∧ y = 1 ∧ z = 1 := 
by sorry

end complex_numbers_satisfying_conditions_l126_126389


namespace problem1_l126_126637

theorem problem1 {a b c : ℝ} (h : a + b + c = 2) : a^2 + b^2 + c^2 + 2 * a * b * c < 2 :=
sorry

end problem1_l126_126637


namespace unique_x_intersect_l126_126208

theorem unique_x_intersect (m : ℝ) (h : ∀ x : ℝ, (m - 4) * x^2 - 2 * m * x - m - 6 = 0 → ∀ y : ℝ, (m - 4) * y^2 - 2 * m * y - m - 6 = 0 → x = y) :
  m = -4 ∨ m = 3 ∨ m = 4 :=
sorry

end unique_x_intersect_l126_126208


namespace percentage_class_takes_lunch_l126_126852

theorem percentage_class_takes_lunch (total_students boys girls : ℕ)
  (h_total: total_students = 100)
  (h_ratio: boys = 6 * total_students / (6 + 4))
  (h_girls: girls = 4 * total_students / (6 + 4))
  (boys_lunch_ratio : ℝ)
  (girls_lunch_ratio : ℝ)
  (h_boys_lunch_ratio : boys_lunch_ratio = 0.60)
  (h_girls_lunch_ratio : girls_lunch_ratio = 0.40):
  ((boys_lunch_ratio * boys + girls_lunch_ratio * girls) / total_students) * 100 = 52 :=
by
  sorry

end percentage_class_takes_lunch_l126_126852


namespace area_of_region_l126_126115

theorem area_of_region (x y : ℝ) : (x^2 + y^2 + 6 * x - 8 * y = 1) → (π * 26) = 26 * π :=
by
  intro h
  sorry

end area_of_region_l126_126115


namespace river_width_l126_126966

theorem river_width 
  (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) 
  (h_depth : depth = 2) 
  (h_flow_rate: flow_rate_kmph = 3) 
  (h_volume : volume_per_minute = 4500) : 
  the_width_of_the_river = 45 :=
by
  sorry 

end river_width_l126_126966


namespace real_estate_commission_l126_126264

theorem real_estate_commission (commission_rate commission selling_price : ℝ) 
  (h1 : commission_rate = 0.06) 
  (h2 : commission = 8880) : 
  selling_price = 148000 :=
by
  sorry

end real_estate_commission_l126_126264


namespace perfect_square_of_expression_l126_126114

theorem perfect_square_of_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) : 
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m^2 := 
by 
  sorry

end perfect_square_of_expression_l126_126114


namespace initially_calculated_average_height_l126_126466

/-- Suppose the average height of 20 students was initially calculated incorrectly. Later, it was found that one student's height 
was incorrectly recorded as 151 cm instead of 136 cm. Given the actual average height of the students is 174.25 cm, prove that the 
initially calculated average height was 173.5 cm. -/
theorem initially_calculated_average_height
  (initial_avg actual_avg : ℝ)
  (num_students : ℕ)
  (incorrect_height correct_height : ℝ)
  (h_avg : actual_avg = 174.25)
  (h_students : num_students = 20)
  (h_incorrect : incorrect_height = 151)
  (h_correct : correct_height = 136)
  (h_total_actual : num_students * actual_avg = num_students * initial_avg + incorrect_height - correct_height) :
  initial_avg = 173.5 :=
by
  sorry

end initially_calculated_average_height_l126_126466


namespace megan_folders_l126_126220

def filesOnComputer : Nat := 93
def deletedFiles : Nat := 21
def filesPerFolder : Nat := 8

theorem megan_folders:
  let remainingFiles := filesOnComputer - deletedFiles
  (remainingFiles / filesPerFolder) = 9 := by
    sorry

end megan_folders_l126_126220


namespace conic_is_pair_of_lines_l126_126387

-- Define the specific conic section equation
def conic_eq (x y : ℝ) : Prop := 9 * x^2 - 36 * y^2 = 0

-- State the theorem
theorem conic_is_pair_of_lines : ∀ x y : ℝ, conic_eq x y ↔ (x = 2 * y ∨ x = -2 * y) :=
by
  -- Sorry is placed to denote that proof steps are omitted in this statement
  sorry

end conic_is_pair_of_lines_l126_126387


namespace problem1_problem2_l126_126638

variables {a b c : ℝ}

-- (1) Prove that a + b + c = 4 given the conditions
theorem problem1 (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_min : ∀ x, abs (x + a) + abs (x - b) + c ≥ 4) : a + b + c = 4 := 
sorry

-- (2) Prove that the minimum value of (1/4)a^2 + (1/9)b^2 + c^2 is 8/7 given the conditions and that a + b + c = 4
theorem problem2 (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 4) : (1/4) * a^2 + (1/9) * b^2 + c^2 ≥ 8 / 7 := 
sorry

end problem1_problem2_l126_126638


namespace sum_of_solutions_sum_of_possible_values_l126_126642

theorem sum_of_solutions (y : ℝ) (h : y^2 = 81) : y = 9 ∨ y = -9 :=
sorry

theorem sum_of_possible_values (y : ℝ) (h : y^2 = 81) : (∀ x, x = 9 ∨ x = -9 → x = 9 ∨ x = -9 → x = 9 + (-9)) :=
by
  have y_sol : y = 9 ∨ y = -9 := sum_of_solutions y h
  sorry

end sum_of_solutions_sum_of_possible_values_l126_126642


namespace customers_total_l126_126312

theorem customers_total 
  (initial : ℝ) 
  (added_lunch_rush : ℝ) 
  (added_after_lunch_rush : ℝ) :
  initial = 29.0 →
  added_lunch_rush = 20.0 →
  added_after_lunch_rush = 34.0 →
  initial + added_lunch_rush + added_after_lunch_rush = 83.0 :=
by
  intros h1 h2 h3
  sorry

end customers_total_l126_126312


namespace polynomial_has_real_root_l126_126471

theorem polynomial_has_real_root (b : ℝ) : ∃ x : ℝ, (x^4 + b * x^3 + 2 * x^2 + b * x - 2 = 0) := sorry

end polynomial_has_real_root_l126_126471


namespace allen_mother_age_l126_126650

variable (A M : ℕ)

theorem allen_mother_age (h1 : A = M - 25) (h2 : (A + 3) + (M + 3) = 41) : M = 30 :=
by
  sorry

end allen_mother_age_l126_126650


namespace segment_area_l126_126231

theorem segment_area (d : ℝ) (θ : ℝ) (r := d / 2)
  (A_triangle := (1 / 2) * r^2 * Real.sin (θ * Real.pi / 180))
  (A_sector := (θ / 360) * Real.pi * r^2) :
  θ = 60 →
  d = 10 →
  A_sector - A_triangle = (100 * Real.pi - 75 * Real.sqrt 3) / 24 :=
by
  sorry

end segment_area_l126_126231


namespace lower_limit_brother_opinion_l126_126633

variables (w B : ℝ)

-- Conditions
-- Arun's weight is between 61 and 72 kg
def arun_cond := 61 < w ∧ w < 72
-- Arun's brother's opinion: greater than B, less than 70
def brother_cond := B < w ∧ w < 70
-- Arun's mother's view: not greater than 64
def mother_cond :=  w ≤ 64

-- Given the average
def avg_weight := 63

theorem lower_limit_brother_opinion (h_arun : arun_cond w) (h_brother: brother_cond w B) (h_mother: mother_cond w) (h_avg: avg_weight = (B + 64)/2) : 
  B = 62 :=
sorry

end lower_limit_brother_opinion_l126_126633


namespace cookies_per_kid_l126_126020

theorem cookies_per_kid (total_calories_per_lunch : ℕ) (burger_calories : ℕ) (carrot_calories_per_stick : ℕ) (num_carrot_sticks : ℕ) (cookie_calories : ℕ) (num_cookies : ℕ) : 
  total_calories_per_lunch = 750 →
  burger_calories = 400 →
  carrot_calories_per_stick = 20 →
  num_carrot_sticks = 5 →
  cookie_calories = 50 →
  num_cookies = (total_calories_per_lunch - (burger_calories + num_carrot_sticks * carrot_calories_per_stick)) / cookie_calories →
  num_cookies = 5 :=
by
  sorry

end cookies_per_kid_l126_126020


namespace simplify_and_evaluate_l126_126613

theorem simplify_and_evaluate (x : ℝ) (h : x = 3) : 
  ( ( (x^2 - 4 * x + 4) / (x^2 - 4) ) / ( (x-2) / (x^2 + 2*x) ) ) + 3 = 6 :=
by
  sorry

end simplify_and_evaluate_l126_126613


namespace log_inequality_l126_126397

theorem log_inequality (n : ℕ) (h1 : n > 1) : 
  (1 : ℝ) / (n : ℝ) > Real.log ((n + 1 : ℝ) / n) ∧ 
  Real.log ((n + 1 : ℝ) / n) > (1 : ℝ) / (n + 1) := 
by
  sorry

end log_inequality_l126_126397


namespace mean_of_five_numbers_is_correct_l126_126077

-- Define the sum of the five numbers
def sum_of_five_numbers : ℚ := 3 / 4

-- Define the number of numbers
def number_of_numbers : ℚ := 5

-- Define the mean
def mean_of_five_numbers := sum_of_five_numbers / number_of_numbers

-- State the theorem
theorem mean_of_five_numbers_is_correct : mean_of_five_numbers = 3 / 20 :=
by
  -- The proof is omitted, use sorry to indicate this.
  sorry

end mean_of_five_numbers_is_correct_l126_126077


namespace smallest_positive_x_for_maximum_l126_126369

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 4) + Real.cos (x / 9)

theorem smallest_positive_x_for_maximum (x : ℝ) :
  (∀ k m : ℤ, x = 360 * (1 + k) ∧ x = 3600 * m ∧ 0 < x → x = 3600) :=
by
  sorry

end smallest_positive_x_for_maximum_l126_126369


namespace range_of_a_l126_126630

theorem range_of_a {a : ℝ} (h : ∃ x : ℝ, (a+2)/(x+1) = 1 ∧ x ≤ 0) :
  a ≤ -1 ∧ a ≠ -2 := 
sorry

end range_of_a_l126_126630


namespace find_common_difference_l126_126738

section
variables (a1 a7 a8 a9 S5 S6 : ℚ) (d : ℚ)

/-- Given an arithmetic sequence with the sum of the first n terms S_n,
    if S_5 = a_8 + 5 and S_6 = a_7 + a_9 - 5, we need to find the common difference d. -/
theorem find_common_difference
  (h1 : S5 = a8 + 5)
  (h2 : S6 = a7 + a9 - 5)
  (h3 : S5 = 5 / 2 * (2 * a1 + 4 * d))
  (h4 : S6 = 6 / 2 * (2 * a1 + 5 * d))
  (h5 : a8 = a1 + 7 * d)
  (h6 : a7 = a1 + 6 * d)
  (h7 : a9 = a1 + 8 * d):
  d = -55 / 19 :=
by
  sorry
end

end find_common_difference_l126_126738


namespace find_n_l126_126218

theorem find_n : ∃ n : ℤ, 3^3 - 5 = 2^5 + n ∧ n = -10 :=
by
  sorry

end find_n_l126_126218


namespace total_books_l126_126356

def sam_books : ℕ := 110
def joan_books : ℕ := 102

theorem total_books : sam_books + joan_books = 212 := by
  sorry

end total_books_l126_126356


namespace symmetry_probability_l126_126459

-- Define the setting of the problem
def grid_points : ℕ := 121
def grid_size : ℕ := 11
def center_point : (ℕ × ℕ) := (6, 6)
def total_points : ℕ := grid_points - 1
def symmetric_lines : ℕ := 4
def points_per_line : ℕ := 10
def total_symmetric_points : ℕ := symmetric_lines * points_per_line
def probability : ℚ := total_symmetric_points / total_points

-- Theorem statement
theorem symmetry_probability 
  (hp: grid_points = 121) 
  (hs: grid_size = 11) 
  (hc: center_point = (6, 6))
  (htp: total_points = 120)
  (hsl: symmetric_lines = 4)
  (hpl: points_per_line = 10)
  (htsp: total_symmetric_points = 40)
  (hp: probability = 1 / 3) : 
  probability = 1 / 3 :=
by 
  sorry

end symmetry_probability_l126_126459


namespace min_value_expr_l126_126927

theorem min_value_expr (a b : ℝ) (h : a * b > 0) : (a^4 + 4 * b^4 + 1) / (a * b) ≥ 4 := 
sorry

end min_value_expr_l126_126927


namespace remainder_of_number_divided_by_39_l126_126720

theorem remainder_of_number_divided_by_39 
  (N : ℤ) 
  (k m : ℤ) 
  (h₁ : N % 195 = 79) 
  (h₂ : N % 273 = 109) : 
  N % 39 = 1 :=
by 
  sorry

end remainder_of_number_divided_by_39_l126_126720


namespace monotone_decreasing_interval_3_l126_126130

variable {f : ℝ → ℝ}

theorem monotone_decreasing_interval_3 
  (h1 : ∀ x, f (x + 3) = f (x - 3))
  (h2 : ∀ x, f (x + 3) = f (-x + 3))
  (h3 : ∀ ⦃x y⦄, 0 < x → x < 3 → 0 < y → y < 3 → x < y → f y < f x) :
  f 3.5 < f (-4.5) ∧ f (-4.5) < f 12.5 :=
sorry

end monotone_decreasing_interval_3_l126_126130


namespace smallest_x_for_quadratic_l126_126513

theorem smallest_x_for_quadratic :
  ∃ x, 8 * x^2 - 38 * x + 35 = 0 ∧ (∀ y, 8 * y^2 - 38 * y + 35 = 0 → x ≤ y) ∧ x = 1.25 :=
by
  sorry

end smallest_x_for_quadratic_l126_126513


namespace absolute_value_and_power_sum_l126_126425

theorem absolute_value_and_power_sum :
  |(-4 : ℤ)| + (3 - Real.pi)^0 = 5 := by
  sorry

end absolute_value_and_power_sum_l126_126425


namespace pipe_B_time_l126_126199

theorem pipe_B_time (C : ℝ) (T : ℝ) 
    (h1 : 2 / 3 * C + C / 3 = C)
    (h2 : C / 36 + C / (3 * T) = C / 14.4) 
    (h3 : T > 0) : 
    T = 8 := 
sorry

end pipe_B_time_l126_126199


namespace number_of_planes_l126_126307

theorem number_of_planes (total_wings: ℕ) (wings_per_plane: ℕ) 
  (h1: total_wings = 50) (h2: wings_per_plane = 2) : 
  total_wings / wings_per_plane = 25 := by 
  sorry

end number_of_planes_l126_126307


namespace ana_salary_after_changes_l126_126486

-- Definitions based on conditions in part (a)
def initial_salary : ℝ := 2000
def raise_factor : ℝ := 1.20
def cut_factor : ℝ := 0.80

-- Statement of the proof problem
theorem ana_salary_after_changes : 
  (initial_salary * raise_factor * cut_factor) = 1920 :=
by
  sorry

end ana_salary_after_changes_l126_126486


namespace prob_at_least_one_hit_correct_prob_exactly_two_hits_correct_l126_126477

-- Define the probabilities that A, B, and C hit the target
def prob_A := 0.7
def prob_B := 0.6
def prob_C := 0.5

-- Define the probabilities that A, B, and C miss the target
def miss_A := 1 - prob_A
def miss_B := 1 - prob_B
def miss_C := 1 - prob_C

-- Probability that no one hits the target
def prob_no_hits := miss_A * miss_B * miss_C

-- Probability that at least one person hits the target
def prob_at_least_one_hit := 1 - prob_no_hits

-- Probabilities for the cases where exactly two people hit the target:
def prob_A_B_hits := prob_A * prob_B * miss_C
def prob_A_C_hits := prob_A * miss_B * prob_C
def prob_B_C_hits := miss_A * prob_B * prob_C

-- Probability that exactly two people hit the target
def prob_exactly_two_hits := prob_A_B_hits + prob_A_C_hits + prob_B_C_hits

-- Theorem statement to prove the probabilities match given conditions
theorem prob_at_least_one_hit_correct : prob_at_least_one_hit = 0.94 := by
  sorry

theorem prob_exactly_two_hits_correct : prob_exactly_two_hits = 0.44 := by
  sorry

end prob_at_least_one_hit_correct_prob_exactly_two_hits_correct_l126_126477


namespace problem1_problem2_l126_126412

-- Proof problem 1
theorem problem1 (x : ℝ) : (x - 1)^2 + x * (3 - x) = x + 1 := sorry

-- Proof problem 2
theorem problem2 (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ -2) : (a - 2) / (a - 1) / (a + 1 - 3 / (a - 1)) = 1 / (a + 2) := sorry

end problem1_problem2_l126_126412


namespace length_of_first_train_l126_126300

noncomputable def first_train_length 
  (speed_first_train_km_h : ℕ) 
  (speed_second_train_km_h : ℕ) 
  (length_second_train_m : ℕ) 
  (time_seconds : ℝ) 
  (relative_speed_m_s : ℝ) : ℝ :=
  let relative_speed_mps := (speed_first_train_km_h + speed_second_train_km_h) * (5 / 18)
  let distance_covered := relative_speed_mps * time_seconds
  let length_first_train := distance_covered - length_second_train_m
  length_first_train

theorem length_of_first_train : 
  first_train_length 40 50 165 11.039116870650348 25 = 110.9779217662587 :=
by 
  rw [first_train_length]
  sorry

end length_of_first_train_l126_126300


namespace product_of_three_numbers_is_correct_l126_126261

noncomputable def sum_three_numbers_product (x y z n : ℚ) : Prop :=
  x + y + z = 200 ∧
  8 * x = y - 12 ∧
  8 * x = z + 12 ∧
  (x * y * z = 502147200 / 4913)

theorem product_of_three_numbers_is_correct :
  ∃ (x y z n : ℚ), sum_three_numbers_product x y z n :=
by
  sorry

end product_of_three_numbers_is_correct_l126_126261


namespace geometric_progression_fourth_term_eq_one_l126_126812

theorem geometric_progression_fourth_term_eq_one :
  let a₁ := (2:ℝ)^(1/4)
  let a₂ := (2:ℝ)^(1/6)
  let a₃ := (2:ℝ)^(1/12)
  let r := a₂ / a₁
  let a₄ := a₃ * r
  a₄ = 1 := by
  sorry

end geometric_progression_fourth_term_eq_one_l126_126812


namespace bowlfuls_per_box_l126_126806

def clusters_per_spoonful : ℕ := 4
def spoonfuls_per_bowl : ℕ := 25
def clusters_per_box : ℕ := 500

theorem bowlfuls_per_box : clusters_per_box / (clusters_per_spoonful * spoonfuls_per_bowl) = 5 :=
by
  sorry

end bowlfuls_per_box_l126_126806


namespace recipe_required_ingredients_l126_126546

-- Define the number of cups required for each ingredient in the recipe
def sugar_cups : Nat := 11
def flour_cups : Nat := 8
def cocoa_cups : Nat := 5

-- Define the cups of flour and cocoa already added
def flour_already_added : Nat := 3
def cocoa_already_added : Nat := 2

-- Define the cups of flour and cocoa that still need to be added
def flour_needed_to_add : Nat := 6
def cocoa_needed_to_add : Nat := 3

-- Sum the total amount of flour and cocoa powder based on already added and still needed amounts
def total_flour: Nat := flour_already_added + flour_needed_to_add
def total_cocoa: Nat := cocoa_already_added + cocoa_needed_to_add

-- Total ingredients calculation according to the problem's conditions
def total_ingredients : Nat := sugar_cups + total_flour + total_cocoa

-- The theorem to be proved
theorem recipe_required_ingredients : total_ingredients = 24 := by
  sorry

end recipe_required_ingredients_l126_126546


namespace range_of_a_l126_126223

variable {x a : ℝ}

theorem range_of_a (hx : 1 ≤ x ∧ x ≤ 2) (h : 2 * x > a - x^2) : a < 8 :=
by sorry

end range_of_a_l126_126223


namespace smallest_twice_perfect_square_three_times_perfect_cube_l126_126027

theorem smallest_twice_perfect_square_three_times_perfect_cube :
  ∃ n : ℕ, (∃ k : ℕ, n = 2 * k^2) ∧ (∃ m : ℕ, n = 3 * m^3) ∧ n = 648 :=
by
  sorry

end smallest_twice_perfect_square_three_times_perfect_cube_l126_126027


namespace min_value_z_l126_126190

theorem min_value_z : ∃ (min_z : ℝ), min_z = 24.1 ∧ 
  ∀ (x y : ℝ), (3 * x ^ 2 + 4 * y ^ 2 + 8 * x - 6 * y + 30) ≥ min_z :=
sorry

end min_value_z_l126_126190


namespace intersection_of_M_and_N_l126_126558

-- Definitions of the sets
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x : ℤ | -2 < x ∧ x < 2}

-- The theorem to prove
theorem intersection_of_M_and_N : M ∩ N = {-1, 0, 1} := by
  sorry

end intersection_of_M_and_N_l126_126558


namespace find_x_plus_3y_l126_126324

variables {α : Type*} {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (x y : ℝ)
variables (OA OB OC OD OE : V)

-- Defining the conditions
def condition1 := OA = (1/2) • OB + x • OC + y • OD
def condition2 := OB = 2 • x • OC + (1/3) • OD + y • OE

-- Writing the theorem statement
theorem find_x_plus_3y (h1 : condition1 x y OA OB OC OD) (h2 : condition2 x y OB OC OD OE) : 
  x + 3 * y = 7 / 6 := 
sorry

end find_x_plus_3y_l126_126324


namespace sum_consecutive_triangular_sum_triangular_2020_l126_126542

-- Define triangular numbers
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

-- The theorem to be proved
theorem sum_consecutive_triangular (n : ℕ) : triangular n + triangular (n + 1) = (n + 1)^2 :=
by 
  sorry

-- Applying the theorem for the specific case of n = 2020
theorem sum_triangular_2020 : triangular 2020 + triangular 2021 = 2021^2 :=
by 
  exact sum_consecutive_triangular 2020

end sum_consecutive_triangular_sum_triangular_2020_l126_126542


namespace infinite_geometric_series_common_ratio_l126_126606

theorem infinite_geometric_series_common_ratio
  (a S : ℝ)
  (h₁ : a = 500)
  (h₂ : S = 4000)
  (h₃ : S = a / (1 - (r : ℝ))) :
  r = 7 / 8 :=
by
  sorry

end infinite_geometric_series_common_ratio_l126_126606


namespace number_of_toothpicks_l126_126202

def num_horizontal_toothpicks(lines width : Nat) : Nat := lines * width
def num_vertical_toothpicks(lines height : Nat) : Nat := lines * height

theorem number_of_toothpicks (high wide : Nat) (missing : Nat) 
  (h_high : high = 15) (h_wide : wide = 15) (h_missing : missing = 1) : 
  num_horizontal_toothpicks (high + 1) wide + num_vertical_toothpicks (wide + 1) high - missing = 479 := by
  sorry

end number_of_toothpicks_l126_126202


namespace zero_point_interval_l126_126664

noncomputable def f (x : ℝ) : ℝ := -x^3 - 3 * x + 5

theorem zero_point_interval: 
  ∃ x₀ : ℝ, f x₀ = 0 → 1 < x₀ ∧ x₀ < 2 :=
sorry

end zero_point_interval_l126_126664


namespace handshake_count_l126_126265

-- Define the conditions
def num_companies : ℕ := 5
def reps_per_company : ℕ := 4
def total_people : ℕ := num_companies * reps_per_company
def handshakes_per_person : ℕ := total_people - 1 - (reps_per_company - 1)

-- Define the theorem to prove
theorem handshake_count : 
  total_people * (total_people - 1 - (reps_per_company - 1)) / 2 = 160 := 
by
  -- Here should be the proof, but it is omitted
  sorry

end handshake_count_l126_126265


namespace necessary_and_sufficient_l126_126403

theorem necessary_and_sufficient (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ¬ ((a > 0 ∧ b > 0 → ab < (a + b) / 2 ^ 2) 
  ∧ (ab < (a + b) / 2 ^ 2 → a > 0 ∧ b > 0)) := 
sorry

end necessary_and_sufficient_l126_126403


namespace find_b_l126_126594

-- Define the constants and assumptions
variables {a b c d : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d)

-- The function completes 5 periods between 0 and 2π
def completes_5_periods (b : ℝ) : Prop :=
  (2 * Real.pi) / b = (2 * Real.pi) / 5

theorem find_b (h : completes_5_periods b) : b = 5 :=
sorry

end find_b_l126_126594


namespace zeros_in_Q_l126_126846

def R_k (k : ℕ) : ℤ := (7^k - 1) / 6

def Q : ℤ := (7^30 - 1) / (7^6 - 1)

def count_zeros (n : ℤ) : ℕ := sorry

theorem zeros_in_Q : count_zeros Q = 470588 :=
by sorry

end zeros_in_Q_l126_126846


namespace reeya_third_subject_score_l126_126802

theorem reeya_third_subject_score (s1 s2 s3 s4 : ℝ) (average : ℝ) (num_subjects : ℝ) (total_score : ℝ) :
    s1 = 65 → s2 = 67 → s4 = 95 → average = 76.6 → num_subjects = 4 → total_score = 306.4 →
    (s1 + s2 + s3 + s4) / num_subjects = average → s3 = 79.4 :=
by
  intros h1 h2 h4 h_average h_num_subjects h_total_score h_avg_eq
  -- Proof steps can be added here
  sorry

end reeya_third_subject_score_l126_126802


namespace monica_cookies_left_l126_126335

theorem monica_cookies_left 
  (father_cookies : ℕ) 
  (mother_cookies : ℕ) 
  (brother_cookies : ℕ) 
  (sister_cookies : ℕ) 
  (aunt_cookies : ℕ) 
  (cousin_cookies : ℕ) 
  (total_cookies : ℕ)
  (father_cookies_eq : father_cookies = 12)
  (mother_cookies_eq : mother_cookies = father_cookies / 2)
  (brother_cookies_eq : brother_cookies = mother_cookies + 2)
  (sister_cookies_eq : sister_cookies = brother_cookies * 3)
  (aunt_cookies_eq : aunt_cookies = father_cookies * 2)
  (cousin_cookies_eq : cousin_cookies = aunt_cookies - 5)
  (total_cookies_eq : total_cookies = 120) : 
  total_cookies - (father_cookies + mother_cookies + brother_cookies + sister_cookies + aunt_cookies + cousin_cookies) = 27 :=
by
  sorry

end monica_cookies_left_l126_126335


namespace one_minus_repeating_decimal_l126_126535

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ := x

theorem one_minus_repeating_decimal:
  ∀ (x : ℚ), x = 1/3 → 1 - x = 2/3 :=
by
  sorry

end one_minus_repeating_decimal_l126_126535


namespace angle_WYZ_correct_l126_126721

-- Define the angles as constants
def angle_XYZ : ℝ := 36
def angle_XYW : ℝ := 15

-- Theorem statement asserting the solution
theorem angle_WYZ_correct :
  (angle_XYZ - angle_XYW = 21) := 
by
  -- This is where the proof would go, but we use 'sorry' as instructed
  sorry

end angle_WYZ_correct_l126_126721


namespace height_of_flagpole_l126_126507

-- Define the given conditions
variables (h : ℝ) -- height of the flagpole
variables (s_f : ℝ) (s_b : ℝ) (h_b : ℝ) -- s_f: shadow length of flagpole, s_b: shadow length of building, h_b: height of building

-- Problem conditions
def flagpole_shadow := (s_f = 45)
def building_shadow := (s_b = 50)
def building_height := (h_b = 20)

-- Mathematically equivalent statement
theorem height_of_flagpole
  (h_f : ℝ) (hsf : flagpole_shadow s_f) (hsb : building_shadow s_b) (hhb : building_height h_b)
  (similar_conditions : h / s_f = h_b / s_b) :
  h_f = 18 :=
by
  sorry

end height_of_flagpole_l126_126507


namespace trains_cross_in_12_seconds_l126_126569

noncomputable def length := 120 -- Length of each train in meters
noncomputable def time_train1 := 10 -- Time taken by the first train to cross the post in seconds
noncomputable def time_train2 := 15 -- Time taken by the second train to cross the post in seconds

noncomputable def speed_train1 := length / time_train1 -- Speed of the first train in m/s
noncomputable def speed_train2 := length / time_train2 -- Speed of the second train in m/s

noncomputable def relative_speed := speed_train1 + speed_train2 -- Relative speed when traveling in opposite directions in m/s
noncomputable def total_length := 2 * length -- Total distance covered when crossing each other

noncomputable def crossing_time := total_length / relative_speed -- Time to cross each other in seconds

theorem trains_cross_in_12_seconds : crossing_time = 12 := by
  sorry

end trains_cross_in_12_seconds_l126_126569


namespace part1_part2_l126_126322

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := 
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) := 
sorry

end part1_part2_l126_126322


namespace length_of_rectangular_sheet_l126_126817

/-- The length of each rectangular sheet is 10 cm given that:
    1. Two identical rectangular sheets each have an area of 48 square centimeters,
    2. The covered area when overlapping the sheets is 72 square centimeters,
    3. The diagonal BD of the overlapping quadrilateral ABCD is 6 centimeters. -/
theorem length_of_rectangular_sheet :
  ∀ (length width : ℝ),
    width * length = 48 ∧
    2 * 48 - 72 = width * 6 ∧
    width * 6 = 24 →
    length = 10 :=
sorry

end length_of_rectangular_sheet_l126_126817


namespace quadratic_range_l126_126255

theorem quadratic_range (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + 1 < 0) → (-1 ≤ a ∧ a ≤ 1) :=
by
  sorry

end quadratic_range_l126_126255


namespace mr_arevalo_change_l126_126799

-- Definitions for the costs of the food items
def cost_smoky_salmon : ℤ := 40
def cost_black_burger : ℤ := 15
def cost_chicken_katsu : ℤ := 25

-- Definitions for the service charge and tip percentages
def service_charge_percent : ℝ := 0.10
def tip_percent : ℝ := 0.05

-- Definition for the amount Mr. Arevalo pays
def amount_paid : ℤ := 100

-- Calculation for total food cost
def total_food_cost : ℤ := cost_smoky_salmon + cost_black_burger + cost_chicken_katsu

-- Calculation for service charge
def service_charge : ℝ := service_charge_percent * total_food_cost

-- Calculation for tip
def tip : ℝ := tip_percent * total_food_cost

-- Calculation for the final bill amount
def final_bill_amount : ℝ := total_food_cost + service_charge + tip

-- Calculation for the change
def change : ℝ := amount_paid - final_bill_amount

-- Proof statement
theorem mr_arevalo_change : change = 8 := by
  sorry

end mr_arevalo_change_l126_126799


namespace fraction_eaten_on_third_day_l126_126146

theorem fraction_eaten_on_third_day
  (total_pieces : ℕ)
  (first_day_fraction : ℚ)
  (second_day_fraction : ℚ)
  (remaining_after_third_day : ℕ)
  (initial_pieces : total_pieces = 200)
  (first_day_eaten : first_day_fraction = 1/4)
  (second_day_eaten : second_day_fraction = 2/5)
  (remaining_bread_after_third_day : remaining_after_third_day = 45) :
  (1 : ℚ) / 2 = 1/2 := sorry

end fraction_eaten_on_third_day_l126_126146


namespace janet_needs_9_dog_collars_l126_126928

variable (D : ℕ)

theorem janet_needs_9_dog_collars (h1 : ∀ d : ℕ, d = 18)
  (h2 : ∀ c : ℕ, c = 10)
  (h3 : (18 * D) + (3 * 10) = 192) :
  D = 9 :=
by
  sorry

end janet_needs_9_dog_collars_l126_126928


namespace intermission_length_l126_126614

def concert_duration : ℕ := 80
def song_duration_total : ℕ := 70

theorem intermission_length : 
  concert_duration - song_duration_total = 10 :=
by
  -- conditions are already defined above
  sorry

end intermission_length_l126_126614


namespace number_of_girls_l126_126294

theorem number_of_girls (sections : ℕ) (boys_per_section : ℕ) (total_boys : ℕ) (total_sections : ℕ) (boys_sections girls : ℕ) :
  total_boys = 408 → 
  total_sections = 27 → 
  total_boys / total_sections = boys_per_section → 
  boys_sections = total_boys / boys_per_section → 
  total_sections - boys_sections = girls / boys_per_section → 
  girls = 324 :=
by sorry

end number_of_girls_l126_126294


namespace compute_k_plus_m_l126_126064

theorem compute_k_plus_m :
  ∃ k m : ℝ, 
    (∀ (x y z : ℝ), x^3 - 9 * x^2 + k * x - m = 0 -> x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 9 ∧ 
    (x = 1 ∨ y = 1 ∨ z = 1) ∧ (x = 3 ∨ y = 3 ∨ z = 3) ∧ (x = 5 ∨ y = 5 ∨ z = 5)) →
    k + m = 38 :=
by
  sorry

end compute_k_plus_m_l126_126064


namespace marble_catch_up_time_l126_126148

theorem marble_catch_up_time 
    (a b c : ℝ) 
    (L : ℝ)
    (h1 : a - b = L / 50)
    (h2 : a - c = L / 40) 
    : (110 * (c - b)) / (c - b) = 110 := 
by 
    sorry

end marble_catch_up_time_l126_126148


namespace electronics_weight_l126_126003

variable (B C E : ℝ)
variable (h1 : B / (B * (4 / 7) - 8) = 2 * (B / (B * (4 / 7))))
variable (h2 : C = B * (4 / 7))
variable (h3 : E = B * (3 / 7))

theorem electronics_weight : E = 12 := by
  sorry

end electronics_weight_l126_126003


namespace circle_and_line_properties_l126_126247

-- Define the circle C with center on the positive x-axis and passing through the origin
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line l: y = kx + 2
def line_l (k x y : ℝ) : Prop := y = k * x + 2

-- Statement: the circle and line setup
theorem circle_and_line_properties (k : ℝ) : 
  ∀ (x y : ℝ), 
  circle_C x y → 
  ∃ (x1 y1 x2 y2 : ℝ), 
  line_l k x1 y1 ∧ 
  line_l k x2 y2 ∧ 
  circle_C x1 y1 ∧ 
  circle_C x2 y2 ∧ 
  (x1 ≠ x2 ∧ y1 ≠ y2) → 
  k < -3/4 ∧
  ( (y1 / x1) + (y2 / x2) = 1 ) :=
by
  sorry

end circle_and_line_properties_l126_126247


namespace range_of_mu_l126_126090

theorem range_of_mu (a b μ : ℝ) (ha : 0 < a) (hb : 0 < b) (hμ : 0 < μ) (h : 1 / a + 9 / b = 1) : μ ≤ 16 :=
by
  sorry

end range_of_mu_l126_126090


namespace divisor_in_first_division_l126_126018

theorem divisor_in_first_division
  (N : ℕ)
  (D : ℕ)
  (Q : ℕ)
  (h1 : N = 8 * D)
  (h2 : N % 5 = 4) :
  D = 3 := 
sorry

end divisor_in_first_division_l126_126018


namespace arcsin_one_half_eq_pi_six_l126_126870

theorem arcsin_one_half_eq_pi_six :
  Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l126_126870


namespace trains_cross_each_other_in_5_76_seconds_l126_126710

noncomputable def trains_crossing_time (l1 l2 v1_kmh v2_kmh : ℕ) : ℚ :=
  let v1 := (v1_kmh : ℚ) * 5 / 18  -- convert speed from km/h to m/s
  let v2 := (v2_kmh : ℚ) * 5 / 18  -- convert speed from km/h to m/s
  let total_distance := (l1 : ℚ) + (l2 : ℚ)
  let relative_velocity := v1 + v2
  total_distance / relative_velocity

theorem trains_cross_each_other_in_5_76_seconds :
  trains_crossing_time 100 60 60 40 = 160 / 27.78 := by
  sorry

end trains_cross_each_other_in_5_76_seconds_l126_126710


namespace smallest_k_for_ten_ruble_heads_up_l126_126095

-- Conditions
def num_total_coins : ℕ := 30
def num_ten_ruble_coins : ℕ := 23
def num_five_ruble_coins : ℕ := 7
def num_heads_up : ℕ := 20
def num_tails_up : ℕ := 10

-- Prove the smallest k such that any k coins chosen include at least one ten-ruble coin heads-up.
theorem smallest_k_for_ten_ruble_heads_up (k : ℕ) :
  (∀ (coins : Finset ℕ), coins.card = k → (∃ (coin : ℕ) (h : coin ∈ coins), coin < num_ten_ruble_coins ∧ coin < num_heads_up)) →
  k = 18 :=
sorry

end smallest_k_for_ten_ruble_heads_up_l126_126095


namespace numBoysInClassroom_l126_126234

-- Definitions based on the problem conditions
def numGirls : ℕ := 10
def girlsToBoysRatio : ℝ := 0.5

-- The statement to prove
theorem numBoysInClassroom : ∃ B : ℕ, girlsToBoysRatio * B = numGirls ∧ B = 20 :=
by
  -- Proof goes here
  sorry

end numBoysInClassroom_l126_126234


namespace determinant_2x2_l126_126068

theorem determinant_2x2 (a b c d : ℝ) 
  (h : Matrix.det (Matrix.of ![![1, a, b], ![2, c, d], ![3, 0, 0]]) = 6) : 
  Matrix.det (Matrix.of ![![a, b], ![c, d]]) = 2 :=
by
  sorry

end determinant_2x2_l126_126068


namespace peter_needs_5000_for_vacation_l126_126272

variable (currentSavings : ℕ) (monthlySaving : ℕ) (months : ℕ)

-- Conditions
def peterSavings := currentSavings
def monthlySavings := monthlySaving
def savingDuration := months

-- Goal
def vacationFundsRequired (currentSavings monthlySaving months : ℕ) : ℕ :=
  currentSavings + (monthlySaving * months)

theorem peter_needs_5000_for_vacation
  (h1 : currentSavings = 2900)
  (h2 : monthlySaving = 700)
  (h3 : months = 3) :
  vacationFundsRequired currentSavings monthlySaving months = 5000 := by
  sorry

end peter_needs_5000_for_vacation_l126_126272


namespace projectile_reaches_50_first_at_0point5_l126_126561

noncomputable def height_at_time (t : ℝ) : ℝ := -16 * t^2 + 100 * t

theorem projectile_reaches_50_first_at_0point5 :
  ∃ t : ℝ, (height_at_time t = 50) ∧ (t = 0.5) :=
sorry

end projectile_reaches_50_first_at_0point5_l126_126561


namespace tom_finishes_in_four_hours_l126_126994

noncomputable def maryMowingRate := 1 / 3
noncomputable def tomMowingRate := 1 / 6
noncomputable def timeMaryMows := 1
noncomputable def remainingLawn := 1 - (timeMaryMows * maryMowingRate)

theorem tom_finishes_in_four_hours :
  remainingLawn / tomMowingRate = 4 :=
by sorry

end tom_finishes_in_four_hours_l126_126994


namespace min_prime_factors_of_expression_l126_126570

theorem min_prime_factors_of_expression (m n : ℕ) : 
  ∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ p1 ≠ p2 ∧ p1 ∣ (m * (n + 9) * (m + 2 * n^2 + 3)) ∧ p2 ∣ (m * (n + 9) * (m + 2 * n^2 + 3)) := 
sorry

end min_prime_factors_of_expression_l126_126570


namespace total_wholesale_cost_is_correct_l126_126677

-- Given values
def retail_price_pants : ℝ := 36
def markup_pants : ℝ := 0.8

def retail_price_shirt : ℝ := 45
def markup_shirt : ℝ := 0.6

def retail_price_jacket : ℝ := 120
def markup_jacket : ℝ := 0.5

noncomputable def wholesale_cost_pants : ℝ := retail_price_pants / (1 + markup_pants)
noncomputable def wholesale_cost_shirt : ℝ := retail_price_shirt / (1 + markup_shirt)
noncomputable def wholesale_cost_jacket : ℝ := retail_price_jacket / (1 + markup_jacket)

noncomputable def total_wholesale_cost : ℝ :=
  wholesale_cost_pants + wholesale_cost_shirt + wholesale_cost_jacket

theorem total_wholesale_cost_is_correct :
  total_wholesale_cost = 128.125 := by
  sorry

end total_wholesale_cost_is_correct_l126_126677


namespace maple_logs_correct_l126_126945

/-- Each pine tree makes 80 logs. -/
def pine_logs := 80

/-- Each walnut tree makes 100 logs. -/
def walnut_logs := 100

/-- Jerry cuts up 8 pine trees. -/
def pine_trees := 8

/-- Jerry cuts up 3 maple trees. -/
def maple_trees := 3

/-- Jerry cuts up 4 walnut trees. -/
def walnut_trees := 4

/-- The total number of logs is 1220. -/
def total_logs := 1220

/-- The number of logs each maple tree makes. -/
def maple_logs := 60

theorem maple_logs_correct :
  (pine_trees * pine_logs) + (maple_trees * maple_logs) + (walnut_trees * walnut_logs) = total_logs :=
by
  -- (8 * 80) + (3 * 60) + (4 * 100) = 1220
  sorry

end maple_logs_correct_l126_126945


namespace cubes_sum_correct_l126_126055

noncomputable def max_cubes : ℕ := 11
noncomputable def min_cubes : ℕ := 9

theorem cubes_sum_correct : max_cubes + min_cubes = 20 :=
by
  unfold max_cubes min_cubes
  sorry

end cubes_sum_correct_l126_126055


namespace range_of_a_l126_126688

theorem range_of_a (a : ℝ) : (∀ x : ℝ, abs (x + 2) - abs (x - 1) ≥ a^3 - 4 * a^2 - 3) → a ≤ 4 :=
sorry

end range_of_a_l126_126688


namespace units_digit_of_sum_of_cubes_l126_126396

theorem units_digit_of_sum_of_cubes : 
  (24^3 + 42^3) % 10 = 2 := by
sorry

end units_digit_of_sum_of_cubes_l126_126396


namespace system_of_equations_solution_l126_126085

theorem system_of_equations_solution (x y : ℤ) 
  (h1 : x^2 + x * y + y^2 = 37) 
  (h2 : x^4 + x^2 * y^2 + y^4 = 481) : 
  (x = 3 ∧ y = 4) ∨ (x = 4 ∧ y = 3) ∨ (x = -3 ∧ y = -4) ∨ (x = -4 ∧ y = -3) := 
by sorry

end system_of_equations_solution_l126_126085


namespace fraction_sum_l126_126801

theorem fraction_sum (n : ℕ) (a : ℚ) (sum_fraction : a = 1/12) (number_of_fractions : n = 450) : 
  ∀ (f : ℚ), (n * f = a) → (f = 1/5400) :=
by
  intros f H
  sorry

end fraction_sum_l126_126801


namespace max_pies_without_ingredients_l126_126711

def total_pies : ℕ := 30
def blueberry_pies : ℕ := total_pies / 3
def raspberry_pies : ℕ := (3 * total_pies) / 5
def blackberry_pies : ℕ := (5 * total_pies) / 6
def walnut_pies : ℕ := total_pies / 10

theorem max_pies_without_ingredients : 
  (total_pies - blackberry_pies) = 5 :=
by 
  -- We only require the proof part.
  sorry

end max_pies_without_ingredients_l126_126711


namespace blue_balls_needed_l126_126157

theorem blue_balls_needed 
  (G B Y W : ℝ)
  (h1 : G = 2 * B)
  (h2 : Y = (8 / 3) * B)
  (h3 : W = (4 / 3) * B) :
  5 * G + 3 * Y + 4 * W = (70 / 3) * B :=
by
  sorry

end blue_balls_needed_l126_126157


namespace second_box_capacity_l126_126241

-- Given conditions
def height1 := 4 -- height of the first box in cm
def width1 := 2 -- width of the first box in cm
def length1 := 6 -- length of the first box in cm
def clay_capacity1 := 48 -- weight capacity of the first box in grams

def height2 := 3 * height1 -- height of the second box in cm
def width2 := 2 * width1 -- width of the second box in cm
def length2 := length1 -- length of the second box in cm

-- Hypothesis: weight capacity increases quadratically with height
def quadratic_relationship (h1 h2 : ℕ) (capacity1 : ℕ) : ℕ :=
  (h2 / h1) * (h2 / h1) * capacity1

-- The proof problem
theorem second_box_capacity :
  quadratic_relationship height1 height2 clay_capacity1 = 432 :=
by
  -- proof omitted
  sorry

end second_box_capacity_l126_126241


namespace position_of_2019_in_splits_l126_126900

def sum_of_consecutive_odds (n : ℕ) : ℕ :=
  n^2 - (n - 1)

theorem position_of_2019_in_splits : ∃ n : ℕ, sum_of_consecutive_odds n = 2019 ∧ n = 45 :=
by
  sorry

end position_of_2019_in_splits_l126_126900


namespace line_eq_of_midpoint_and_hyperbola_l126_126730

theorem line_eq_of_midpoint_and_hyperbola (x1 y1 x2 y2 : ℝ) (h1 : 9 * (8 : ℝ)^2 - 16 * (3 : ℝ)^2 = 144)
    (h2 : x1 + x2 = 16) (h3 : y1 + y2 = 6) (h4 : 9 * x1^2 - 16 * y1^2 = 144) (h5 : 9 * x2^2 - 16 * y2^2 = 144) :
    3 * (8 : ℝ) - 2 * (3 : ℝ) - 18 = 0 :=
by
  -- The proof steps would go here
  sorry

end line_eq_of_midpoint_and_hyperbola_l126_126730


namespace money_left_l126_126001

theorem money_left 
  (salary : ℝ)
  (spent_on_food : ℝ)
  (spent_on_rent : ℝ)
  (spent_on_clothes : ℝ)
  (total_spent : ℝ)
  (money_left : ℝ)
  (h_salary : salary = 170000)
  (h_food : spent_on_food = salary * (1 / 5))
  (h_rent : spent_on_rent = salary * (1 / 10))
  (h_clothes : spent_on_clothes = salary * (3 / 5))
  (h_total_spent : total_spent = spent_on_food + spent_on_rent + spent_on_clothes)
  (h_money_left : money_left = salary - total_spent) :
  money_left = 17000 :=
by
  sorry

end money_left_l126_126001


namespace smaller_solution_of_quadratic_l126_126362

theorem smaller_solution_of_quadratic :
  (∃ x y : ℝ, x ≠ y ∧ (x^2 - 13 * x + 36 = 0) ∧ (y^2 - 13 * y + 36 = 0) ∧ min x y = 4) :=
sorry

end smaller_solution_of_quadratic_l126_126362


namespace percentage_increase_after_decrease_and_increase_l126_126030

theorem percentage_increase_after_decrease_and_increase 
  (P : ℝ) 
  (h : 0.8 * P + (x / 100) * (0.8 * P) = 1.16 * P) : 
  x = 45 :=
by
  sorry

end percentage_increase_after_decrease_and_increase_l126_126030


namespace geometric_sequence_sum_l126_126842

-- Let {a_n} be a geometric sequence such that S_2 = 7 and S_6 = 91. Prove that S_4 = 28

-- Define the sum of the first n terms of a geometric sequence
noncomputable def S (n : ℕ) (a1 r : ℝ) : ℝ := a1 * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum (a1 r : ℝ) (h1 : S 2 a1 r = 7) (h2 : S 6 a1 r = 91) :
  S 4 a1 r = 28 := 
by 
  sorry

end geometric_sequence_sum_l126_126842


namespace original_rice_amount_l126_126790

theorem original_rice_amount (r : ℚ) (x y : ℚ)
  (h1 : r = 3/5)
  (h2 : x + y = 10)
  (h3 : x + r * y = 7) : 
  x + y = 10 ∧ x + 3/5 * y = 7 := 
by
  sorry

end original_rice_amount_l126_126790


namespace inequality_proof_l126_126123

theorem inequality_proof (x a : ℝ) (h1 : x > a) (h2 : a > 0) : x^2 > ax ∧ ax > a^2 :=
by
  sorry

end inequality_proof_l126_126123


namespace dots_not_visible_on_3_dice_l126_126939

theorem dots_not_visible_on_3_dice :
  let total_dots := 18 * 21 / 6
  let visible_dots := 1 + 2 + 2 + 3 + 5 + 4 + 5 + 6
  let hidden_dots := total_dots - visible_dots
  hidden_dots = 35 := 
by 
  let total_dots := 18 * 21 / 6
  let visible_dots := 1 + 2 + 2 + 3 + 5 + 4 + 5 + 6
  let hidden_dots := total_dots - visible_dots
  show total_dots - visible_dots = 35
  sorry

end dots_not_visible_on_3_dice_l126_126939


namespace horse_goat_sheep_consumption_l126_126349

theorem horse_goat_sheep_consumption :
  (1 / (1 / (1 : ℝ) + 1 / 2 + 1 / 3)) = 6 / 11 :=
by
  sorry

end horse_goat_sheep_consumption_l126_126349


namespace misha_total_shots_l126_126901

theorem misha_total_shots (x y : ℕ) 
  (h1 : 18 * x + 5 * y = 99) 
  (h2 : 2 * x + y = 15) 
  (h3 : (15 / 0.9375 : ℝ) = 16) : 
  (¬(x = 0) ∧ ¬(y = 24)) ->
  16 = 16 :=
by
  sorry

end misha_total_shots_l126_126901


namespace sum_largest_and_second_smallest_l126_126659

-- Define the list of numbers
def numbers : List ℕ := [10, 11, 12, 13, 14]

-- Define a predicate to get the largest number
def is_largest (n : ℕ) : Prop := ∀ x ∈ numbers, x ≤ n

-- Define a predicate to get the second smallest number
def is_second_smallest (n : ℕ) : Prop :=
  ∃ a b, (a ∈ numbers ∧ b ∈ numbers ∧ a < b ∧ b < n ∧ ∀ x ∈ numbers, (x < a ∨ x > b))

-- The main goal: To prove that the sum of the largest number and the second smallest number is 25
theorem sum_largest_and_second_smallest : 
  ∃ l s, is_largest l ∧ is_second_smallest s ∧ l + s = 25 := 
sorry

end sum_largest_and_second_smallest_l126_126659


namespace compute_d1e1_d2e2_d3e3_l126_126947

-- Given polynomials and conditions
variables {R : Type*} [CommRing R]

noncomputable def P (x : R) : R :=
  x^7 - x^6 + x^4 - x^3 + x^2 - x + 1

noncomputable def Q (x : R) (d1 d2 d3 e1 e2 e3 : R) : R :=
  (x^2 + d1 * x + e1) * (x^2 + d2 * x + e2) * (x^2 + d3 * x + e3)

-- Given conditions
theorem compute_d1e1_d2e2_d3e3 
  (d1 d2 d3 e1 e2 e3 : R)
  (h : ∀ x : R, P x = Q x d1 d2 d3 e1 e2 e3) : 
  d1 * e1 + d2 * e2 + d3 * e3 = -1 :=
by
  sorry

end compute_d1e1_d2e2_d3e3_l126_126947


namespace snack_eaters_left_after_second_newcomers_l126_126906

theorem snack_eaters_left_after_second_newcomers
  (initial_snackers : ℕ)
  (new_outsiders_1 : ℕ)
  (half_left_1 : ℕ)
  (new_outsiders_2 : ℕ)
  (final_snackers : ℕ)
  (H1 : initial_snackers = 100)
  (H2 : new_outsiders_1 = 20)
  (H3 : half_left_1 = (initial_snackers + new_outsiders_1) / 2)
  (H4 : new_outsiders_2 = 10)
  (H5 : final_snackers = 20)
  : (initial_snackers + new_outsiders_1 - half_left_1 + new_outsiders_2 - (initial_snackers + new_outsiders_1 - half_left_1 + new_outsiders_2 - final_snackers * 2)) = 30 :=
by 
  sorry

end snack_eaters_left_after_second_newcomers_l126_126906


namespace rectangular_solid_surface_area_l126_126295

theorem rectangular_solid_surface_area (a b c : ℕ) (h₁ : Prime a ∨ ∃ p : ℕ, Prime p ∧ a = p + (p + 1))
                                         (h₂ : Prime b ∨ ∃ q : ℕ, Prime q ∧ b = q + (q + 1))
                                         (h₃ : Prime c ∨ ∃ r : ℕ, Prime r ∧ c = r + (r + 1))
                                         (h₄ : a * b * c = 399) :
  2 * (a * b + b * c + c * a) = 422 := 
sorry

end rectangular_solid_surface_area_l126_126295


namespace proof_problem_l126_126110

def x := 3
def y := 4

theorem proof_problem : 3 * x - 2 * y = 1 := by
  -- We will rely on these definitions and properties of arithmetic to show the result.
  -- The necessary proof steps would follow here, but are skipped for now.
  sorry

end proof_problem_l126_126110


namespace domain_of_f_l126_126617

def function_domain (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∀ x, x ∈ domain ↔ ∃ y, f y = x

noncomputable def f (x : ℝ) : ℝ :=
  (x + 6) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_of_f :
  function_domain f ((Set.Iio 2) ∪ (Set.Ioi 3)) :=
by
  sorry

end domain_of_f_l126_126617


namespace avg_two_ab_l126_126269

-- Defining the weights and conditions
variables (A B C : ℕ)

-- The conditions provided in the problem
def avg_three (A B C : ℕ) := (A + B + C) / 3 = 45
def avg_two_bc (B C : ℕ) := (B + C) / 2 = 43
def weight_b (B : ℕ) := B = 35

-- The target proof statement
theorem avg_two_ab (A B C : ℕ) (h1 : avg_three A B C) (h2 : avg_two_bc B C) (h3 : weight_b B) : (A + B) / 2 = 42 := 
sorry

end avg_two_ab_l126_126269


namespace least_positive_divisible_by_primes_l126_126690

theorem least_positive_divisible_by_primes : 
  let p1 := 2 
  let p2 := 3 
  let p3 := 5 
  let p4 := 7
  ∃ n : ℕ, n > 0 ∧ (n % p1 = 0) ∧ (n % p2 = 0) ∧ (n % p3 = 0) ∧ (n % p4 = 0) ∧ 
  (∀ m : ℕ, m > 0 → (m % p1 = 0) ∧ (m % p2 = 0) ∧ (m % p3 = 0) ∧ (m % p4 = 0) → m ≥ n) ∧ n = 210 := 
by {
  sorry
}

end least_positive_divisible_by_primes_l126_126690


namespace vegetable_options_l126_126965

open Nat

theorem vegetable_options (V : ℕ) : 
  3 * V + 6 = 57 → V = 5 :=
by
  intro h
  sorry

end vegetable_options_l126_126965


namespace eval_expression_l126_126292

theorem eval_expression :
  let x := 2
  let y := -3
  let z := 1
  x^2 + y^2 - z^2 + 2 * x * y + 3 * z = 0 := by
sorry

end eval_expression_l126_126292


namespace gigi_has_15_jellybeans_l126_126560

variable (G : ℕ) -- G is the number of jellybeans Gigi has
variable (R : ℕ) -- R is the number of jellybeans Rory has
variable (L : ℕ) -- L is the number of jellybeans Lorelai has eaten

-- Conditions
def condition1 := R = G + 30
def condition2 := L = 3 * (G + R)
def condition3 := L = 180

-- Proof statement
theorem gigi_has_15_jellybeans (G R L : ℕ) (h1 : condition1 G R) (h2 : condition2 G R L) (h3 : condition3 L) : G = 15 := by
  sorry

end gigi_has_15_jellybeans_l126_126560


namespace find_y_l126_126582

theorem find_y (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : x = 4) : y = 3 := 
by sorry

end find_y_l126_126582


namespace find_c_l126_126708

theorem find_c (b c : ℝ) (h : (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c * x + 12)) : c = 7 :=
sorry

end find_c_l126_126708


namespace sum_of_x_coordinates_l126_126857

def line1 (x : ℝ) : ℝ := -3 * x - 5
def line2 (x : ℝ) : ℝ := 2 * x - 3

def has_x_intersect (line : ℝ → ℝ) (y : ℝ) : Prop := ∃ x : ℝ, line x = y

theorem sum_of_x_coordinates :
  (∃ x1 x2 : ℝ, line1 x1 = 2.2 ∧ line2 x2 = 2.2 ∧ x1 + x2 = 0.2) :=
  sorry

end sum_of_x_coordinates_l126_126857


namespace find_difference_l126_126884

variable (d : ℕ) (A B : ℕ)
open Nat

theorem find_difference (hd : d > 7)
  (hAB : d * A + B + d * A + A = d * d + 7 * d + 4)  (hA_gt_B : A > B):
  A - B = 3 :=
sorry

end find_difference_l126_126884


namespace lisa_walks_distance_per_minute_l126_126282

-- Variables and conditions
variable (d : ℤ) -- distance that Lisa walks each minute (what we're solving for)
variable (daily_distance : ℤ) -- distance that Lisa walks each hour
variable (total_distance_in_two_days : ℤ := 1200) -- total distance in two days
variable (hours_per_day : ℤ := 1) -- one hour per day

-- Given conditions
axiom walks_for_an_hour_each_day : ∀ (d: ℤ), daily_distance = d * 60
axiom walks_1200_meters_in_two_days : ∀ (d: ℤ), total_distance_in_two_days = 2 * daily_distance

-- The theorem we want to prove
theorem lisa_walks_distance_per_minute : (d = 10) :=
by
  -- TODO: complete the proof
  sorry

end lisa_walks_distance_per_minute_l126_126282


namespace problem_inequality_l126_126904

theorem problem_inequality 
  (a b c d : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : a ≤ b) (h6 : b ≤ c) (h7 : c ≤ d) 
  (h8 : a + b + c + d ≥ 1) : 
  a^2 + 3*b^2 + 5*c^2 + 7*d^2 ≥ 1 := 
sorry

end problem_inequality_l126_126904


namespace lowest_selling_price_l126_126768

/-- Define the variables and constants -/
def production_cost_per_component := 80
def shipping_cost_per_component := 7
def fixed_costs_per_month := 16500
def components_per_month := 150

/-- Define the total variable cost -/
def total_variable_cost (production_cost_per_component shipping_cost_per_component : ℕ) (components_per_month : ℕ) :=
  (production_cost_per_component + shipping_cost_per_component) * components_per_month

/-- Define the total cost -/
def total_cost (variable_cost fixed_costs_per_month : ℕ) :=
  variable_cost + fixed_costs_per_month

/-- Define the lowest price per component -/
def lowest_price_per_component (total_cost components_per_month : ℕ) :=
  total_cost / components_per_month

/-- The main theorem to prove the lowest selling price required to cover all costs -/
theorem lowest_selling_price (production_cost shipping_cost fixed_costs components : ℕ)
  (h1 : production_cost = 80)
  (h2 : shipping_cost = 7)
  (h3 : fixed_costs = 16500)
  (h4 : components = 150) :
  lowest_price_per_component (total_cost (total_variable_cost production_cost shipping_cost components) fixed_costs) components = 197 :=
by
  sorry

end lowest_selling_price_l126_126768


namespace picked_clovers_when_one_four_found_l126_126880

-- Definition of conditions
def total_leaves : ℕ := 100
def leaves_three_leaved_clover : ℕ := 3
def leaves_four_leaved_clover : ℕ := 4
def one_four_leaved_clover : ℕ := 1

-- Proof Statement
theorem picked_clovers_when_one_four_found (three_leaved_count : ℕ) :
  (total_leaves - leaves_four_leaved_clover) / leaves_three_leaved_clover = three_leaved_count → 
  three_leaved_count = 32 :=
by
  sorry

end picked_clovers_when_one_four_found_l126_126880


namespace pinky_pig_apples_l126_126543

variable (P : ℕ)

theorem pinky_pig_apples (h : P + 73 = 109) : P = 36 := sorry

end pinky_pig_apples_l126_126543


namespace smallest_positive_period_monotonic_decreasing_interval_l126_126536

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) ^ 2 + 2 * Real.sin x * Real.cos x

theorem smallest_positive_period (T : ℝ) :
  (∀ x, f (x + T) = f x) ∧ T > 0 → T = Real.pi :=
by
  sorry

theorem monotonic_decreasing_interval :
  (∀ x, x ∈ Set.Icc (3 * Real.pi / 8) (7 * Real.pi / 8) → ∃ k : ℤ, 
     f (x + k * π) = f x ∧ f (x + k * π) ≤ f (x + (k + 1) * π)) :=
by
  sorry

end smallest_positive_period_monotonic_decreasing_interval_l126_126536


namespace parallel_lines_when_m_is_neg7_l126_126270

-- Given two lines l1 and l2 defined as:
def l1 (m : ℤ) (x y : ℤ) := (3 + m) * x + 4 * y = 5 - 3 * m
def l2 (m : ℤ) (x y : ℤ) := 2 * x + (5 + m) * y = 8

-- The proof problem to show that l1 is parallel to l2 when m = -7
theorem parallel_lines_when_m_is_neg7 :
  ∃ m : ℤ, (∀ x y : ℤ, l1 m x y → l2 m x y) → m = -7 := 
sorry

end parallel_lines_when_m_is_neg7_l126_126270


namespace deepak_present_age_l126_126698

theorem deepak_present_age (R D : ℕ) (h1 : R =  4 * D / 3) (h2 : R + 10 = 26) : D = 12 :=
by
  sorry

end deepak_present_age_l126_126698


namespace earnings_per_widget_l126_126308

theorem earnings_per_widget (W_h : ℝ) (H_w : ℕ) (W_t : ℕ) (E_w : ℝ) (E : ℝ) :
  W_h = 12.50 ∧ H_w = 40 ∧ W_t = 1000 ∧ E_w = 660 →
  E = 0.16 :=
by
  sorry

end earnings_per_widget_l126_126308


namespace proof_statement_B_proof_statement_D_proof_statement_E_l126_126372

def statement_B (x : ℝ) : Prop := x^2 = 0 → x = 0

def statement_D (x : ℝ) : Prop := x^2 < 2 * x → x > 0

def statement_E (x : ℝ) : Prop := x > 2 → x^2 > x

theorem proof_statement_B (x : ℝ) : statement_B x := sorry

theorem proof_statement_D (x : ℝ) : statement_D x := sorry

theorem proof_statement_E (x : ℝ) : statement_E x := sorry

end proof_statement_B_proof_statement_D_proof_statement_E_l126_126372


namespace problem1_problem2_min_value_l126_126062

theorem problem1 (x : ℝ) : |x + 1| + |x - 2| ≥ 3 := sorry

theorem problem2 (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : 
  x^2 + y^2 + z^2 ≥ 1 / 14 := sorry

theorem min_value (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) :
  ∃ x y z, x^2 + y^2 + z^2 = 1 / 14 := sorry

end problem1_problem2_min_value_l126_126062


namespace minimum_n_for_all_columns_l126_126859

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Function to check if a given number covers all columns from 0 to 9
def covers_all_columns (n : ℕ) : Bool :=
  let columns := (List.range n).map (λ i => triangular_number i % 10)
  List.range 10 |>.all (λ c => c ∈ columns)

theorem minimum_n_for_all_columns : ∃ n, covers_all_columns n ∧ triangular_number n = 253 :=
by 
  sorry

end minimum_n_for_all_columns_l126_126859


namespace minimum_value_of_f_range_of_t_l126_126240

noncomputable def f (x : ℝ) : ℝ := x + 9 / (x - 3)

theorem minimum_value_of_f :
  (∃ x > 3, f x = 9) :=
by
  sorry

theorem range_of_t (t : ℝ) :
  (∀ x > 3, f x ≥ t / (t + 1) + 7) ↔ (t ≤ -2 ∨ t > -1) :=
by
  sorry

end minimum_value_of_f_range_of_t_l126_126240


namespace hexagon_side_length_l126_126400

-- Define the conditions for the side length of a hexagon where the area equals the perimeter
theorem hexagon_side_length (s : ℝ) (h1 : (3 * Real.sqrt 3 / 2) * s^2 = 6 * s) :
  s = 4 * Real.sqrt 3 / 3 :=
sorry

end hexagon_side_length_l126_126400


namespace transformed_function_correct_l126_126237

-- Given function
def f (x : ℝ) : ℝ := 2 * x + 1

-- Main theorem to be proven
theorem transformed_function_correct (x : ℝ) (h : 2 ≤ x ∧ x ≤ 4) : 
  f (x - 1) = 2 * x - 1 :=
by {
  sorry
}

end transformed_function_correct_l126_126237


namespace jake_peaches_l126_126482

noncomputable def steven_peaches : ℕ := 15
noncomputable def jake_fewer : ℕ := 7

theorem jake_peaches : steven_peaches - jake_fewer = 8 :=
by
  sorry

end jake_peaches_l126_126482


namespace paperclips_exceed_target_in_days_l126_126193

def initial_paperclips := 3
def ratio := 2
def target_paperclips := 200

theorem paperclips_exceed_target_in_days :
  ∃ k : ℕ, initial_paperclips * ratio ^ k > target_paperclips ∧ k = 8 :=
by {
  sorry
}

end paperclips_exceed_target_in_days_l126_126193


namespace multiplication_identity_l126_126191

theorem multiplication_identity : 32519 * 9999 = 324857481 := by
  sorry

end multiplication_identity_l126_126191


namespace root_of_equation_l126_126126

theorem root_of_equation :
  ∀ x : ℝ, (x - 3)^2 = x - 3 ↔ x = 3 ∨ x = 4 :=
by
  sorry

end root_of_equation_l126_126126


namespace average_after_17th_inning_l126_126204

-- Define the conditions.
variable (A : ℚ) -- The initial average after 16 innings

-- Define the score in the 17th inning and the increment in the average.
def runs_in_17th_inning : ℚ := 87
def increment_in_average : ℚ := 3

-- Define the equation derived from the conditions.
theorem average_after_17th_inning :
  (16 * A + runs_in_17th_inning) / 17 = A + increment_in_average →
  A + increment_in_average = 39 :=
sorry

end average_after_17th_inning_l126_126204


namespace find_m_l126_126161

theorem find_m (m n : ℤ) (h : 21 * (m + n) + 21 = 21 * (-m + n) + 21) : m = 0 :=
sorry

end find_m_l126_126161


namespace remainder_division_l126_126910

-- Define the polynomial f(x) = x^51 + 51
def f (x : ℤ) : ℤ := x^51 + 51

-- State the theorem to be proven
theorem remainder_division : f (-1) = 50 :=
by
  -- proof goes here
  sorry

end remainder_division_l126_126910


namespace percentage_exceed_l126_126589

theorem percentage_exceed (x y : ℝ) (h : y = x + (0.25 * x)) : (y - x) / x * 100 = 25 :=
by
  sorry

end percentage_exceed_l126_126589


namespace trapezoid_area_l126_126375

theorem trapezoid_area 
  (h : ℝ) (BM CM : ℝ) 
  (height_cond : h = 12) 
  (BM_cond : BM = 15) 
  (CM_cond : CM = 13) 
  (angle_bisectors_intersect : ∃ M : ℝ, (BM^2 - h^2) = 9^2 ∧ (CM^2 - h^2) = 5^2) : 
  ∃ (S : ℝ), S = 260.4 :=
by
  -- Skipping the proof part by using sorry
  sorry

end trapezoid_area_l126_126375


namespace angle_sum_is_180_l126_126808

theorem angle_sum_is_180 (A B C : ℝ) (h_triangle : (A + B + C) = 180) (h_sum : A + B = 90) : C = 90 :=
by
  -- Proof placeholder
  sorry

end angle_sum_is_180_l126_126808


namespace mrs_hilt_remaining_cents_l126_126897

-- Define the initial amount of money Mrs. Hilt had
def initial_cents : ℕ := 43

-- Define the cost of the pencil
def pencil_cost : ℕ := 20

-- Define the cost of the candy
def candy_cost : ℕ := 5

-- Define the remaining money Mrs. Hilt has after the purchases
def remaining_cents : ℕ := initial_cents - (pencil_cost + candy_cost)

-- Theorem statement to prove that the remaining amount is 18 cents
theorem mrs_hilt_remaining_cents : remaining_cents = 18 := by
  -- Proof omitted
  sorry

end mrs_hilt_remaining_cents_l126_126897


namespace tom_dimes_now_l126_126315

-- Define the initial number of dimes and the number of dimes given by dad
def initial_dimes : ℕ := 15
def dimes_given_by_dad : ℕ := 33

-- Define the final count of dimes Tom has now
def final_dimes (initial_dimes dimes_given_by_dad : ℕ) : ℕ :=
  initial_dimes + dimes_given_by_dad

-- The main theorem to prove "how many dimes Tom has now"
theorem tom_dimes_now : initial_dimes + dimes_given_by_dad = 48 :=
by
  -- The proof can be skipped using sorry
  sorry

end tom_dimes_now_l126_126315


namespace highway_length_l126_126979

theorem highway_length 
  (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) 
  (h_speed1 : speed1 = 14)
  (h_speed2 : speed2 = 16)
  (h_time : time = 1.5) : 
  speed1 * time + speed2 * time = 45 := 
sorry

end highway_length_l126_126979


namespace christmas_bonus_remainder_l126_126913

theorem christmas_bonus_remainder (B P R : ℕ) (hP : P = 8 * B + 5) (hR : (4 * P) % 8 = R) : R = 4 :=
by
  sorry

end christmas_bonus_remainder_l126_126913


namespace james_weekly_expenses_l126_126813

noncomputable def utility_cost (rent: ℝ):  ℝ := 0.2 * rent
noncomputable def weekly_hours_open (hours_per_day: ℕ) (days_per_week: ℕ): ℕ := hours_per_day * days_per_week
noncomputable def employee_weekly_wages (wage_per_hour: ℝ) (weekly_hours: ℕ): ℝ := wage_per_hour * weekly_hours
noncomputable def total_employee_wages (employees: ℕ) (weekly_wages: ℝ): ℝ := employees * weekly_wages
noncomputable def total_weekly_expenses (rent: ℝ) (utilities: ℝ) (employee_wages: ℝ): ℝ := rent + utilities + employee_wages

theorem james_weekly_expenses : 
  let rent := 1200
  let utility_percentage := 0.2
  let hours_per_day := 16
  let days_per_week := 5
  let employees := 2
  let wage_per_hour := 12.5
  let weekly_hours := weekly_hours_open hours_per_day days_per_week
  let utilities := utility_cost rent
  let employee_wages_per_week := employee_weekly_wages wage_per_hour weekly_hours
  let total_employee_wages_per_week := total_employee_wages employees employee_wages_per_week
  total_weekly_expenses rent utilities total_employee_wages_per_week = 3440 := 
by
  sorry

end james_weekly_expenses_l126_126813


namespace rug_area_calculation_l126_126697

theorem rug_area_calculation (length_floor width_floor strip_width : ℕ)
  (h_length : length_floor = 10)
  (h_width : width_floor = 8)
  (h_strip : strip_width = 2) :
  (length_floor - 2 * strip_width) * (width_floor - 2 * strip_width) = 24 := by
  sorry

end rug_area_calculation_l126_126697


namespace polynomial_inequality_solution_l126_126271

theorem polynomial_inequality_solution (x : ℝ) :
  x^4 + x^3 - 10 * x^2 + 25 * x > 0 ↔ x > 0 :=
sorry

end polynomial_inequality_solution_l126_126271


namespace least_sugar_l126_126024

theorem least_sugar (f s : ℚ) (h1 : f ≥ 10 + 3 * s / 4) (h2 : f ≤ 3 * s) :
  s ≥ 40 / 9 :=
  sorry

end least_sugar_l126_126024


namespace Bettina_card_value_l126_126800

theorem Bettina_card_value (x : ℝ) (h₀ : 0 < x) (h₁ : x < π / 2) (h₂ : Real.tan x ≠ 1) (h₃ : Real.sin x ≠ Real.cos x) :
  ∀ {a b c : ℝ}, (a = Real.sin x ∨ a = Real.cos x ∨ a = Real.tan x) →
                  (b = Real.sin x ∨ b = Real.cos x ∨ b = Real.tan x) →
                  (c = Real.sin x ∨ c = Real.cos x ∨ c = Real.tan x) →
                  a ≠ b → b ≠ c → a ≠ c →
                  (b = Real.cos x) → b = Real.sqrt 3 / 2 := 
  sorry

end Bettina_card_value_l126_126800


namespace square_side_length_l126_126334

theorem square_side_length (x : ℝ) (h : 4 * x = 8 * Real.pi) : x = 6.28 := 
by {
  -- proof will go here
  sorry
}

end square_side_length_l126_126334


namespace company_production_average_l126_126047

theorem company_production_average (n : ℕ) 
  (h1 : (50 * n) / n = 50) 
  (h2 : (50 * n + 105) / (n + 1) = 55) :
  n = 10 :=
sorry

end company_production_average_l126_126047


namespace find_possible_values_of_y_l126_126592

noncomputable def solve_y (x : ℝ) : ℝ :=
  ((x - 3) ^ 2 * (x + 4)) / (2 * x - 4)

theorem find_possible_values_of_y (x : ℝ) 
  (h : x ^ 2 + 9 * (x / (x - 3)) ^ 2 = 90) : 
  solve_y x = 0 ∨ solve_y x = 105.23 := 
sorry

end find_possible_values_of_y_l126_126592


namespace part1_part2_l126_126907

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l126_126907


namespace compare_constants_l126_126198

noncomputable def a := 1 / Real.exp 1
noncomputable def b := Real.log 2 / 2
noncomputable def c := Real.log 3 / 3

theorem compare_constants : b < c ∧ c < a := by
  sorry

end compare_constants_l126_126198


namespace percentage_of_black_marbles_l126_126217

variable (T : ℝ) -- Total number of marbles
variable (C : ℝ) -- Number of clear marbles
variable (B : ℝ) -- Number of black marbles
variable (O : ℝ) -- Number of other colored marbles

-- Conditions
def condition1 := C = 0.40 * T
def condition2 := O = (2 / 5) * T
def condition3 := C + B + O = T

-- Proof statement
theorem percentage_of_black_marbles :
  C = 0.40 * T → O = (2 / 5) * T → C + B + O = T → B = 0.20 * T :=
by
  intros hC hO hTotal
  -- Intermediate steps would go here, but we use sorry to skip the proof.
  sorry

end percentage_of_black_marbles_l126_126217


namespace avg_of_other_two_l126_126209

-- Definitions and conditions from the problem
def avg (l : List ℕ) : ℕ := l.sum / l.length

variables {A B C D E : ℕ}
variables (h_avg_five : avg [A, B, C, D, E] = 20)
variables (h_sum_three : A + B + C = 48)
variables (h_twice : A = 2 * B)

-- Theorem to prove
theorem avg_of_other_two (A B C D E : ℕ) 
  (h_avg_five : avg [A, B, C, D, E] = 20)
  (h_sum_three : A + B + C = 48)
  (h_twice : A = 2 * B) :
  avg [D, E] = 26 := 
  sorry

end avg_of_other_two_l126_126209


namespace range_of_m_l126_126978

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (2 * x - m) / (x - 3) - 1 = x / (3 - x)) →
  m > 3 ∧ m ≠ 9 :=
by
  sorry

end range_of_m_l126_126978


namespace find_values_of_expression_l126_126717

theorem find_values_of_expression (a b : ℝ) 
  (h : (2 * a) / (a + b) + b / (a - b) = 2) : 
  (∃ x : ℝ, x = (3 * a - b) / (a + 5 * b) ∧ (x = 3 ∨ x = 1)) :=
by 
  sorry

end find_values_of_expression_l126_126717


namespace ten_years_less_average_age_l126_126931

-- Defining the conditions formally
def lukeAge : ℕ := 20
def mrBernardAgeInEightYears : ℕ := 3 * lukeAge

-- Lean statement to prove the problem
theorem ten_years_less_average_age : 
  mrBernardAgeInEightYears - 8 = 52 → (lukeAge + (mrBernardAgeInEightYears - 8)) / 2 - 10 = 26 := 
by
  intros h
  sorry

end ten_years_less_average_age_l126_126931


namespace sphere_surface_area_l126_126154

theorem sphere_surface_area (R h : ℝ) (R_pos : 0 < R) (h_pos : 0 < h) :
  ∃ A : ℝ, A = 2 * Real.pi * R * h := 
sorry

end sphere_surface_area_l126_126154


namespace even_function_phi_l126_126399

noncomputable def phi := (3 * Real.pi) / 2

theorem even_function_phi (phi_val : Real) (hphi : 0 ≤ phi_val ∧ phi_val ≤ 2 * Real.pi) :
  (∀ x, Real.sin ((x + phi) / 3) = Real.sin ((-x + phi) / 3)) ↔ phi_val = phi := by
  sorry

end even_function_phi_l126_126399


namespace mike_went_to_last_year_l126_126028

def this_year_games : ℕ := 15
def games_missed_this_year : ℕ := 41
def total_games_attended : ℕ := 54
def last_year_games : ℕ := total_games_attended - this_year_games

theorem mike_went_to_last_year :
  last_year_games = 39 :=
  by sorry

end mike_went_to_last_year_l126_126028


namespace unclaimed_candy_fraction_l126_126864

-- Definitions for the shares taken by each person.
def al_share (x : ℕ) : ℚ := 3 / 7 * x
def bert_share (x : ℕ) : ℚ := 2 / 7 * (x - al_share x)
def carl_share (x : ℕ) : ℚ := 1 / 7 * ((x - al_share x) - bert_share x)
def dana_share (x : ℕ) : ℚ := 1 / 7 * (((x - al_share x) - bert_share x) - carl_share x)

-- The amount of candy that goes unclaimed.
def remaining_candy (x : ℕ) : ℚ := x - (al_share x + bert_share x + carl_share x + dana_share x)

-- The theorem we want to prove.
theorem unclaimed_candy_fraction (x : ℕ) : remaining_candy x / x = 584 / 2401 :=
by
  sorry

end unclaimed_candy_fraction_l126_126864


namespace find_m_l126_126983

-- Define the given vectors and the parallel condition
def vectors_parallel (m : ℝ) : Prop :=
  let a := (1, m)
  let b := (3, 1)
  a.1 * b.2 = a.2 * b.1

-- Statement to be proved
theorem find_m (m : ℝ) : vectors_parallel m → m = 1 / 3 :=
by
  sorry

end find_m_l126_126983


namespace each_child_receive_amount_l126_126668

def husband_weekly_contribution : ℕ := 335
def wife_weekly_contribution : ℕ := 225
def weeks_in_month : ℕ := 4
def months : ℕ := 6
def children : ℕ := 4

noncomputable def total_weekly_contribution : ℕ := husband_weekly_contribution + wife_weekly_contribution
noncomputable def total_savings : ℕ := total_weekly_contribution * (weeks_in_month * months)
noncomputable def half_savings : ℕ := total_savings / 2
noncomputable def amount_per_child : ℕ := half_savings / children

theorem each_child_receive_amount :
  amount_per_child = 1680 :=
by
  sorry

end each_child_receive_amount_l126_126668


namespace smallest_n_divisible_by_2016_smallest_n_divisible_by_2016_pow_10_l126_126912

-- Problem (a): Smallest n such that n! is divisible by 2016
theorem smallest_n_divisible_by_2016 : ∃ (n : ℕ), n = 8 ∧ 2016 ∣ n.factorial :=
by
  sorry

-- Problem (b): Smallest n such that n! is divisible by 2016^10
theorem smallest_n_divisible_by_2016_pow_10 : ∃ (n : ℕ), n = 63 ∧ 2016^10 ∣ n.factorial :=
by
  sorry

end smallest_n_divisible_by_2016_smallest_n_divisible_by_2016_pow_10_l126_126912


namespace incorrect_conclusion_symmetry_l126_126140

/-- Given the function f(x) = sin(1/5 * x + 13/6 * π), we define another function g(x) as the
translated function of f rightward by 10/3 * π units. We need to show that the graph of g(x)
is not symmetrical about the line x = π/4. -/
theorem incorrect_conclusion_symmetry (f g : ℝ → ℝ)
  (h₁ : ∀ x, f x = Real.sin (1/5 * x + 13/6 * Real.pi))
  (h₂ : ∀ x, g x = f (x - 10/3 * Real.pi)) :
  ¬ (∀ x, g (2 * (Real.pi / 4) - x) = g x) :=
sorry

end incorrect_conclusion_symmetry_l126_126140


namespace moles_of_water_used_l126_126902

-- Define the balanced chemical equation's molar ratios
def balanced_reaction (Li3N_moles : ℕ) (H2O_moles : ℕ) (LiOH_moles : ℕ) (NH3_moles : ℕ) : Prop :=
  Li3N_moles = 1 ∧ H2O_moles = 3 ∧ LiOH_moles = 3 ∧ NH3_moles = 1

-- Given 1 mole of lithium nitride and 3 moles of lithium hydroxide produced, 
-- prove that 3 moles of water were used.
theorem moles_of_water_used (Li3N_moles : ℕ) (LiOH_moles : ℕ) (H2O_moles : ℕ) :
  Li3N_moles = 1 → LiOH_moles = 3 → H2O_moles = 3 :=
by
  intros h1 h2
  sorry

end moles_of_water_used_l126_126902


namespace minimize_AC_plus_BC_l126_126976

noncomputable def minimize_distance (k : ℝ) : Prop :=
  let A := (5, 5)
  let B := (2, 1)
  let C := (0, k)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let AC := dist A C
  let BC := dist B C
  ∀ k', dist (0, k') A + dist (0, k') B ≥ AC + BC

theorem minimize_AC_plus_BC : minimize_distance (15 / 7) :=
sorry

end minimize_AC_plus_BC_l126_126976


namespace range_of_m_l126_126456

theorem range_of_m (m : ℝ) (x : ℝ) 
  (h1 : ∀ x : ℝ, -1 < x ∧ x < 4 → x > 2 * m^2 - 3)
  (h2 : ¬ (∀ x : ℝ, x > 2 * m^2 - 3 → -1 < x ∧ x < 4))
  :
  -1 ≤ m ∧ m ≤ 1 :=
sorry

end range_of_m_l126_126456


namespace sufficient_and_necessary_condition_l126_126999

variable {a_n : ℕ → ℝ}

-- Defining the geometric sequence and the given conditions
def is_geometric_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a_n (n + 1) = a_n n * r

def is_increasing_sequence (a_n : ℕ → ℝ) : Prop :=
  ∀ n, a_n n < a_n (n + 1)

def condition (a_n : ℕ → ℝ) : Prop := a_n 0 < a_n 1 ∧ a_n 1 < a_n 2

-- The proof statement
theorem sufficient_and_necessary_condition (a_n : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a_n) :
  condition a_n ↔ is_increasing_sequence a_n :=
sorry

end sufficient_and_necessary_condition_l126_126999


namespace distance_traveled_downstream_l126_126972

noncomputable def speed_boat : ℝ := 20  -- Speed of the boat in still water in km/hr
noncomputable def rate_current : ℝ := 5  -- Rate of current in km/hr
noncomputable def time_minutes : ℝ := 24  -- Time traveled downstream in minutes
noncomputable def time_hours : ℝ := time_minutes / 60  -- Convert time to hours
noncomputable def effective_speed_downstream : ℝ := speed_boat + rate_current  -- Effective speed downstream

theorem distance_traveled_downstream :
  effective_speed_downstream * time_hours = 10 := by {
  sorry
}

end distance_traveled_downstream_l126_126972


namespace find_first_number_l126_126050

theorem find_first_number (x : ℝ) : (10 + 70 + 28) / 3 = 36 →
  (x + 40 + 60) / 3 = 40 →
  x = 20 := 
by
  intros h_avg_old h_avg_new
  sorry

end find_first_number_l126_126050


namespace percentage_of_a_is_4b_l126_126432

variable (a b : ℝ)

theorem percentage_of_a_is_4b (h : a = 1.2 * b) : 4 * b = (10 / 3) * a := 
by 
    sorry

end percentage_of_a_is_4b_l126_126432


namespace students_on_playground_l126_126097

theorem students_on_playground (rows_left : ℕ) (rows_right : ℕ) (rows_front : ℕ) (rows_back : ℕ) (h1 : rows_left = 12) (h2 : rows_right = 11) (h3 : rows_front = 18) (h4 : rows_back = 8) :
    (rows_left + rows_right - 1) * (rows_front + rows_back - 1) = 550 := 
by
  sorry

end students_on_playground_l126_126097


namespace cubs_more_home_runs_l126_126496

-- Define the conditions for the Chicago Cubs
def cubs_home_runs_third_inning : Nat := 2
def cubs_home_runs_fifth_inning : Nat := 1
def cubs_home_runs_eighth_inning : Nat := 2

-- Define the conditions for the Cardinals
def cardinals_home_runs_second_inning : Nat := 1
def cardinals_home_runs_fifth_inning : Nat := 1

-- Total home runs scored by each team
def total_cubs_home_runs : Nat :=
  cubs_home_runs_third_inning + cubs_home_runs_fifth_inning + cubs_home_runs_eighth_inning

def total_cardinals_home_runs : Nat :=
  cardinals_home_runs_second_inning + cardinals_home_runs_fifth_inning

-- The statement to prove
theorem cubs_more_home_runs : total_cubs_home_runs - total_cardinals_home_runs = 3 := by
  sorry

end cubs_more_home_runs_l126_126496


namespace isosceles_triangle_perimeter_l126_126545

theorem isosceles_triangle_perimeter (a b : ℕ) (h_eq : a = 5 ∨ a = 9) (h_side : b = 9 ∨ b = 5) (h_neq : a ≠ b) : 
  (a + a + b = 19 ∨ a + a + b = 23) :=
by
  sorry

end isosceles_triangle_perimeter_l126_126545


namespace correct_equation_l126_126877

theorem correct_equation (x y a b : ℝ) :
  ¬ (-(x - 6) = -x - 6) ∧
  ¬ (-y^2 - y^2 = 0) ∧
  ¬ (9 * a^2 * b - 9 * a * b^2 = 0) ∧
  (-9 * y^2 + 16 * y^2 = 7 * y^2) :=
by
  sorry

end correct_equation_l126_126877


namespace initial_amount_simple_interest_l126_126725

theorem initial_amount_simple_interest 
  (A : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (hA : A = 1125)
  (hR : R = 0.10)
  (hT : T = 5) :
  A = P * (1 + R * T) → P = 750 := 
by
  sorry

end initial_amount_simple_interest_l126_126725


namespace original_price_of_article_l126_126776

theorem original_price_of_article 
  (S : ℝ) (gain_percent : ℝ) (P : ℝ)
  (h1 : S = 25)
  (h2 : gain_percent = 1.5)
  (h3 : S = P + P * gain_percent) : 
  P = 10 :=
by 
  sorry

end original_price_of_article_l126_126776


namespace product_not_perfect_square_l126_126657

theorem product_not_perfect_square :
  ¬ ∃ n : ℕ, n^2 = (2021^1004) * (6^3) :=
by
  sorry

end product_not_perfect_square_l126_126657


namespace test_tube_full_with_two_amoebas_l126_126772

-- Definition: Each amoeba doubles in number every minute.
def amoeba_doubling (initial : Nat) (minutes : Nat) : Nat :=
  initial * 2 ^ minutes

-- Condition: Starting with one amoeba, the test tube is filled in 60 minutes.
def time_to_fill_one_amoeba := 60

-- Theorem: If two amoebas are placed in the test tube, it takes 59 minutes to fill.
theorem test_tube_full_with_two_amoebas : amoeba_doubling 2 59 = amoeba_doubling 1 time_to_fill_one_amoeba :=
by sorry

end test_tube_full_with_two_amoebas_l126_126772


namespace derivative_f_eq_l126_126557

noncomputable def f (x : ℝ) : ℝ :=
  (7^x * (3 * Real.sin (3 * x) + Real.cos (3 * x) * Real.log 7)) / (9 + Real.log 7 ^ 2)

theorem derivative_f_eq :
  ∀ x : ℝ, deriv f x = 7^x * Real.cos (3 * x) :=
by
  intro x
  sorry

end derivative_f_eq_l126_126557


namespace calories_for_breakfast_l126_126075

theorem calories_for_breakfast :
  let cake_calories := 110
  let chips_calories := 310
  let coke_calories := 215
  let lunch_calories := 780
  let daily_limit := 2500
  let remaining_calories := 525
  let total_dinner_snacks := cake_calories + chips_calories + coke_calories
  let total_lunch_dinner := total_dinner_snacks + lunch_calories
  let total_consumed := daily_limit - remaining_calories
  total_consumed - total_lunch_dinner = 560 := by
  sorry

end calories_for_breakfast_l126_126075


namespace find_unknown_number_l126_126116

theorem find_unknown_number (x : ℝ) : 
  (1000 * 7) / (x * 17) = 10000 → x = 24.285714285714286 := by
  sorry

end find_unknown_number_l126_126116


namespace intersection_A_complement_B_eq_minus_three_to_zero_l126_126714

-- Define the set A
def A : Set ℝ := { x : ℝ | x^2 + x - 6 ≤ 0 }

-- Define the set B
def B : Set ℝ := { y : ℝ | ∃ x : ℝ, y = Real.sqrt x ∧ 0 ≤ x ∧ x ≤ 4 }

-- Define the complement of B
def C_RB : Set ℝ := { y : ℝ | ¬ (y ∈ B) }

-- The proof problem
theorem intersection_A_complement_B_eq_minus_three_to_zero :
  (A ∩ C_RB) = { x : ℝ | -3 ≤ x ∧ x < 0 } :=
by
  sorry

end intersection_A_complement_B_eq_minus_three_to_zero_l126_126714


namespace impossible_to_maintain_Gini_l126_126879

variables (X Y G0 Y' Z : ℝ)
variables (G1 : ℝ)

-- Conditions
axiom initial_Gini : G0 = 0.1
axiom proportion_poor : X = 0.5
axiom income_poor_initial : Y = 0.4
axiom income_poor_half : Y' = 0.2
axiom population_split : ∀ a b c : ℝ, (a + b + c = 1) ∧ (a = b ∧ b = c)
axiom Gini_constant : G1 = G0

-- Equation system representation final value post situation
axiom Gini_post_reform : 
  G1 = (1 / 2 - ((1 / 6) * 0.2 + (1 / 6) * (0.2 + Z) + (1 / 6) * (1 - 0.2 - Z))) / (1 / 2)

-- Proof problem: to prove inconsistency or inability to maintain Gini coefficient given the conditions
theorem impossible_to_maintain_Gini : false :=
sorry

end impossible_to_maintain_Gini_l126_126879


namespace count_rectangles_with_perimeter_twenty_two_l126_126868

theorem count_rectangles_with_perimeter_twenty_two : 
  (∃! (n : ℕ), n = 11) :=
by
  sorry

end count_rectangles_with_perimeter_twenty_two_l126_126868


namespace find_circle_center_l126_126661

noncomputable def circle_center_lemma (a b : ℝ) : Prop :=
  -- Condition: Circle passes through (1, 0)
  (a - 1)^2 + b^2 = (a - 1)^2 + (b - 0)^2 ∧
  -- Condition: Circle is tangent to the parabola y = x^2 at (1, 1)
  (a - 1)^2 + (b - 1)^2 = 0

theorem find_circle_center : ∃ a b : ℝ, circle_center_lemma a b ∧ a = 1 ∧ b = 1 :=
by
  sorry

end find_circle_center_l126_126661


namespace sphere_radius_equals_4_l126_126444

noncomputable def radius_of_sphere
  (sun_parallel : true)
  (meter_stick_height : ℝ)
  (meter_stick_shadow : ℝ)
  (sphere_shadow_distance : ℝ) : ℝ :=
if h : meter_stick_height / meter_stick_shadow = sphere_shadow_distance / 16 then
  4
else
  sorry

theorem sphere_radius_equals_4 
  (sun_parallel : true = true)
  (meter_stick_height : ℝ := 1)
  (meter_stick_shadow : ℝ := 4)
  (sphere_shadow_distance : ℝ := 16) : 
  radius_of_sphere sun_parallel meter_stick_height meter_stick_shadow sphere_shadow_distance = 4 :=
by
  simp [radius_of_sphere]
  sorry

end sphere_radius_equals_4_l126_126444


namespace problem_l126_126060

theorem problem (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ q) : ¬ p ∧ q :=
by
  -- proof goes here
  sorry

end problem_l126_126060


namespace min_value_fraction_l126_126304

variable (x y : ℝ)

theorem min_value_fraction (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  ∃ (m : ℝ), (∀ z, (z = (1/x) + (9/y)) → z ≥ 16) ∧ ((1/x) + (9/y) = m) :=
sorry

end min_value_fraction_l126_126304


namespace sum_of_ages_l126_126648

theorem sum_of_ages 
  (a1 a2 a3 : ℕ) 
  (h1 : a1 ≠ a2) 
  (h2 : a1 ≠ a3) 
  (h3 : a2 ≠ a3) 
  (h4 : 1 ≤ a1 ∧ a1 ≤ 9) 
  (h5 : 1 ≤ a2 ∧ a2 ≤ 9) 
  (h6 : 1 ≤ a3 ∧ a3 ≤ 9) 
  (h7 : a1 * a2 = 18) 
  (h8 : a3 * min a1 a2 = 28) : 
  a1 + a2 + a3 = 18 := 
sorry

end sum_of_ages_l126_126648


namespace weekly_allowance_l126_126280

theorem weekly_allowance (A : ℝ) (h1 : A / 2 + 6 = 11) : A = 10 := 
by 
  sorry

end weekly_allowance_l126_126280


namespace largest_possible_square_area_l126_126016

def rectangle_length : ℕ := 9
def rectangle_width : ℕ := 6
def largest_square_side : ℕ := rectangle_width
def largest_square_area : ℕ := largest_square_side * largest_square_side

theorem largest_possible_square_area :
  largest_square_area = 36 := by
    sorry

end largest_possible_square_area_l126_126016


namespace each_persons_contribution_l126_126866

def total_cost : ℝ := 67
def coupon : ℝ := 4
def num_people : ℝ := 3

theorem each_persons_contribution :
  (total_cost - coupon) / num_people = 21 := by
  sorry

end each_persons_contribution_l126_126866


namespace num_female_managers_l126_126166

-- Definitions based on the conditions
def total_employees : ℕ := 250
def female_employees : ℕ := 90
def total_managers : ℕ := 40
def male_associates : ℕ := 160

-- Proof statement that computes the number of female managers
theorem num_female_managers : 
  (total_managers - (total_employees - female_employees - male_associates)) = 40 := 
by 
  sorry

end num_female_managers_l126_126166


namespace money_left_after_deductions_l126_126696

-- Define the weekly income
def weekly_income : ℕ := 500

-- Define the tax deduction as 10% of the weekly income
def tax : ℕ := (10 * weekly_income) / 100

-- Define the weekly water bill
def water_bill : ℕ := 55

-- Define the tithe as 10% of the weekly income
def tithe : ℕ := (10 * weekly_income) / 100

-- Define the total deductions
def total_deductions : ℕ := tax + water_bill + tithe

-- Define the money left
def money_left : ℕ := weekly_income - total_deductions

-- The statement to prove
theorem money_left_after_deductions : money_left = 345 := by
  sorry

end money_left_after_deductions_l126_126696


namespace company_A_profit_l126_126796

-- Define the conditions
def total_profit (x : ℝ) : ℝ := x
def company_B_share (x : ℝ) : Prop := 0.4 * x = 60000
def company_A_percentage : ℝ := 0.6

-- Define the statement to be proved
theorem company_A_profit (x : ℝ) (h : company_B_share x) : 0.6 * x = 90000 := sorry

end company_A_profit_l126_126796


namespace sum_of_exponents_of_1985_eq_40_l126_126997

theorem sum_of_exponents_of_1985_eq_40 :
  ∃ (e₀ e₁ e₂ e₃ e₄ e₅ : ℕ), 1985 = 2^e₀ + 2^e₁ + 2^e₂ + 2^e₃ + 2^e₄ + 2^e₅ 
  ∧ e₀ ≠ e₁ ∧ e₀ ≠ e₂ ∧ e₀ ≠ e₃ ∧ e₀ ≠ e₄ ∧ e₀ ≠ e₅
  ∧ e₁ ≠ e₂ ∧ e₁ ≠ e₃ ∧ e₁ ≠ e₄ ∧ e₁ ≠ e₅
  ∧ e₂ ≠ e₃ ∧ e₂ ≠ e₄ ∧ e₂ ≠ e₅
  ∧ e₃ ≠ e₄ ∧ e₃ ≠ e₅
  ∧ e₄ ≠ e₅
  ∧ e₀ + e₁ + e₂ + e₃ + e₄ + e₅ = 40 := 
by
  sorry

end sum_of_exponents_of_1985_eq_40_l126_126997


namespace parabola_equation_with_left_focus_l126_126709

theorem parabola_equation_with_left_focus (x y : ℝ) :
  (∀ x y : ℝ, (x^2)/25 + (y^2)/9 = 1 → (y^2 = -16 * x)) :=
by
  sorry

end parabola_equation_with_left_focus_l126_126709


namespace money_made_arkansas_game_is_8722_l126_126949

def price_per_tshirt : ℕ := 98
def tshirts_sold_arkansas_game : ℕ := 89
def total_money_made_arkansas_game (price_per_tshirt tshirts_sold_arkansas_game : ℕ) : ℕ :=
  price_per_tshirt * tshirts_sold_arkansas_game

theorem money_made_arkansas_game_is_8722 :
  total_money_made_arkansas_game price_per_tshirt tshirts_sold_arkansas_game = 8722 :=
by
  sorry

end money_made_arkansas_game_is_8722_l126_126949


namespace find_pairs_l126_126779

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 1 + 5^a = 6^b → (a, b) = (1, 1) := by
  sorry

end find_pairs_l126_126779


namespace line_intersects_y_axis_at_origin_l126_126381

theorem line_intersects_y_axis_at_origin 
  (x₁ y₁ x₂ y₂ : ℤ) 
  (h₁ : (x₁, y₁) = (3, 9)) 
  (h₂ : (x₂, y₂) = (-7, -21)) 
  : 
  ∃ y : ℤ, (0, y) = (0, 0) := by
  sorry

end line_intersects_y_axis_at_origin_l126_126381


namespace Ludwig_daily_salary_l126_126260

theorem Ludwig_daily_salary 
(D : ℝ)
(h_weekly_earnings : 4 * D + (3 / 2) * D = 55) :
D = 10 := 
by
  sorry

end Ludwig_daily_salary_l126_126260


namespace radius_of_circle_l126_126357

variables (O P A B : Type) [MetricSpace O] [MetricSpace P] [MetricSpace A] [MetricSpace B]
variables (circle_radius : ℝ) (PA PB OP : ℝ)

theorem radius_of_circle
  (h1 : PA * PB = 24)
  (h2 : OP = 5)
  (circle_radius : ℝ)
  : circle_radius = 7 :=
by sorry

end radius_of_circle_l126_126357


namespace katie_speed_l126_126479

theorem katie_speed (eugene_speed : ℝ)
  (brianna_ratio : ℝ)
  (katie_ratio : ℝ)
  (h1 : eugene_speed = 4)
  (h2 : brianna_ratio = 2 / 3)
  (h3 : katie_ratio = 7 / 5) :
  katie_ratio * (brianna_ratio * eugene_speed) = 56 / 15 := 
by
  sorry

end katie_speed_l126_126479


namespace average_weight_of_remaining_boys_l126_126682

theorem average_weight_of_remaining_boys (avg_weight_16: ℝ) (avg_weight_total: ℝ) (weight_16: ℝ) (total_boys: ℝ) (avg_weight_8: ℝ) : 
  (avg_weight_16 = 50.25) → (avg_weight_total = 48.55) → (weight_16 = 16 * avg_weight_16) → (total_boys = 24) → 
  (total_weight = total_boys * avg_weight_total) → (weight_16 + 8 * avg_weight_8 = total_weight) → avg_weight_8 = 45.15 :=
by
  intros h_avg_weight_16 h_avg_weight_total h_weight_16 h_total_boys h_total_weight h_equation
  sorry

end average_weight_of_remaining_boys_l126_126682


namespace bubble_gum_cost_l126_126508

-- Define the conditions
def total_cost : ℕ := 2448
def number_of_pieces : ℕ := 136

-- Main theorem to state that each piece of bubble gum costs 18 cents
theorem bubble_gum_cost : total_cost / number_of_pieces = 18 :=
by
  sorry

end bubble_gum_cost_l126_126508


namespace average_annual_reduction_10_percent_l126_126974

theorem average_annual_reduction_10_percent :
  ∀ x : ℝ, (1 - x) ^ 2 = 1 - 0.19 → x = 0.1 :=
by
  intros x h
  -- Proof to be filled in
  sorry

end average_annual_reduction_10_percent_l126_126974


namespace bus_probability_l126_126441

/-- A bus arrives randomly between 3:00 and 4:00, waits for 15 minutes, and then leaves. 
Sarah also arrives randomly between 3:00 and 4:00. Prove the probability that the bus 
will be there when Sarah arrives is 4275/7200. -/
theorem bus_probability : (4275 : ℚ) / 7200 = (4275 / 7200) :=
by 
  sorry

end bus_probability_l126_126441


namespace count_possible_x_values_l126_126156

theorem count_possible_x_values (x y : ℕ) (H : (x + 2) * (y + 2) - x * y = x * y) :
  (∃! x, ∃ y, (x - 2) * (y - 2) = 8) :=
by {
  sorry
}

end count_possible_x_values_l126_126156


namespace michael_initial_fish_l126_126969

-- Define the conditions
def benGave : ℝ := 18.0
def totalFish : ℝ := 67

-- Define the statement to be proved
theorem michael_initial_fish :
  (totalFish - benGave) = 49 := by
  sorry

end michael_initial_fish_l126_126969


namespace positive_root_of_equation_l126_126416

theorem positive_root_of_equation :
  ∃ a b : ℤ, (a + b * Real.sqrt 3)^3 - 5 * (a + b * Real.sqrt 3)^2 + 2 * (a + b * Real.sqrt 3) - Real.sqrt 3 = 0 ∧
    a + b * Real.sqrt 3 > 0 ∧
    (a + b * Real.sqrt 3) = 3 + Real.sqrt 3 := 
by
  sorry

end positive_root_of_equation_l126_126416


namespace polynomial_coeffs_l126_126049

theorem polynomial_coeffs :
  ( ∃ (a1 a2 a3 a4 a5 : ℕ), (∀ (x : ℝ), (x + 1) ^ 3 * (x + 2) ^ 2 = x^5 + a1 * x^4 + a2 * x^3 + a3 * x^2 + a4 * x + a5) ∧ a4 = 16 ∧ a5 = 4) := 
by
  sorry

end polynomial_coeffs_l126_126049


namespace initial_books_in_bin_l126_126059

theorem initial_books_in_bin
  (x : ℝ)
  (h : x + 33.0 + 2.0 = 76) :
  x = 41.0 :=
by 
  -- Proof goes here
  sorry

end initial_books_in_bin_l126_126059


namespace gifts_wrapped_with_third_roll_l126_126002

def num_rolls : ℕ := 3
def num_gifts : ℕ := 12
def first_roll_gifts : ℕ := 3
def second_roll_gifts : ℕ := 5

theorem gifts_wrapped_with_third_roll : 
  first_roll_gifts + second_roll_gifts < num_gifts → 
  num_gifts - (first_roll_gifts + second_roll_gifts) = 4 := 
by
  intros h
  sorry

end gifts_wrapped_with_third_roll_l126_126002


namespace pentagon_square_ratio_l126_126509

theorem pentagon_square_ratio (s p : ℕ) (h1 : 4 * s = 20) (h2 : 5 * p = 20) :
  p / s = 4 / 5 :=
by
  sorry

end pentagon_square_ratio_l126_126509


namespace joanne_trip_l126_126773

theorem joanne_trip (a b c x : ℕ) (h1 : 1 ≤ a) (h2 : a + b + c = 9) (h3 : 100 * c + 10 * a + b - (100 * a + 10 * b + c) = 60 * x) : 
  a^2 + b^2 + c^2 = 51 :=
by
  sorry

end joanne_trip_l126_126773


namespace intersection_unique_element_l126_126346

noncomputable def A := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
noncomputable def B (r : ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

theorem intersection_unique_element (r : ℝ) (hr : r > 0) :
  (∃! p : ℝ × ℝ, p ∈ A ∧ p ∈ B r) → (r = 3 ∨ r = 7) :=
sorry

end intersection_unique_element_l126_126346


namespace largest_divisor_of_n4_sub_4n2_is_4_l126_126539

theorem largest_divisor_of_n4_sub_4n2_is_4 (n : ℤ) : 4 ∣ (n^4 - 4 * n^2) :=
sorry

end largest_divisor_of_n4_sub_4n2_is_4_l126_126539


namespace jerome_time_6_hours_l126_126783

theorem jerome_time_6_hours (T: ℝ) (s_J: ℝ) (t_N: ℝ) (s_N: ℝ)
  (h1: s_J = 4) 
  (h2: t_N = 3) 
  (h3: s_N = 8): T = 6 :=
by
  -- Given s_J = 4, t_N = 3, and s_N = 8,
  -- we need to prove that T = 6.
  sorry

end jerome_time_6_hours_l126_126783


namespace find_compounding_frequency_l126_126439

-- Lean statement defining the problem conditions and the correct answer

theorem find_compounding_frequency (P A : ℝ) (r t : ℝ) (hP : P = 12000) (hA : A = 13230) 
(hri : r = 0.10) (ht : t = 1) 
: ∃ (n : ℕ), A = P * (1 + r / n) ^ (n * t) ∧ n = 2 := 
by
  -- Definitions from the conditions
  have hP := hP
  have hA := hA
  have hr := hri
  have ht := ht
  
  -- Substitute known values
  use 2
  -- Show that the statement holds with n = 2
  sorry

end find_compounding_frequency_l126_126439


namespace no_convex_quad_with_given_areas_l126_126386

theorem no_convex_quad_with_given_areas :
  ¬ ∃ (A B C D M : Type) 
    (T_MAB T_MBC T_MDA T_MDC : ℕ) 
    (H1 : T_MAB = 1) 
    (H2 : T_MBC = 2)
    (H3 : T_MDA = 3) 
    (H4 : T_MDC = 4),
    true :=
by {
  sorry
}

end no_convex_quad_with_given_areas_l126_126386


namespace isosceles_triangle_base_length_l126_126314

theorem isosceles_triangle_base_length
  (a b c: ℕ) 
  (h_iso: a = b ∨ a = c ∨ b = c)
  (h_perimeter: a + b + c = 21)
  (h_side: a = 5 ∨ b = 5 ∨ c = 5) :
  c = 5 :=
by
  sorry

end isosceles_triangle_base_length_l126_126314


namespace length_of_MN_l126_126160

theorem length_of_MN (b : ℝ) (h_focus : ∃ b : ℝ, (3/2, b).1 > 0 ∧ (3/2, b).2 * (3/2, b).2 = 6 * (3 / 2)) : 
  |2 * b| = 6 :=
by sorry

end length_of_MN_l126_126160


namespace ratio_of_sum_of_terms_l126_126831

theorem ratio_of_sum_of_terms (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 5 / a 3 = 5 / 9) : S 9 / S 5 = 1 := 
  sorry

end ratio_of_sum_of_terms_l126_126831


namespace evaluate_expression_l126_126515

theorem evaluate_expression : 5000 * 5000^3000 = 5000^3001 := 
by sorry

end evaluate_expression_l126_126515


namespace max_abs_z_l126_126267

open Complex

theorem max_abs_z (z : ℂ) (h : abs (z + I) + abs (z - I) = 2) : abs z ≤ 1 :=
sorry

end max_abs_z_l126_126267


namespace hotel_charge_l126_126125

variable (R G P : ℝ)

theorem hotel_charge (h1 : P = 0.60 * R) (h2 : P = 0.90 * G) : (R - G) / G = 0.50 :=
by
  sorry

end hotel_charge_l126_126125


namespace circle_point_outside_range_l126_126521

theorem circle_point_outside_range (m : ℝ) :
  ¬ (1 + 1 + 4 * m - 2 * 1 + 5 * m = 0) → 
  (m > 1 ∨ (0 < m ∧ m < 1 / 4)) := 
sorry

end circle_point_outside_range_l126_126521


namespace cylinder_to_sphere_volume_ratio_l126_126531

theorem cylinder_to_sphere_volume_ratio:
  ∀ (a r : ℝ), (a^2 = π * r^2) → (a^3)/( (4/3) * π * r^3) = 3/2 :=
by
  intros a r h
  sorry

end cylinder_to_sphere_volume_ratio_l126_126531


namespace petya_finishes_earlier_than_masha_l126_126359

variable (t_P t_M t_K : ℕ)

-- Given conditions
def condition1 := t_K = 2 * t_P
def condition2 := t_P + 12 = t_K
def condition3 := t_M = 3 * t_P

-- The proof goal: Petya finishes 24 seconds earlier than Masha
theorem petya_finishes_earlier_than_masha
    (h1 : condition1 t_P t_K)
    (h2 : condition2 t_P t_K)
    (h3 : condition3 t_P t_M) :
    t_M - t_P = 24 := by
  sorry

end petya_finishes_earlier_than_masha_l126_126359


namespace probability_blue_given_glass_l126_126505

-- Defining the various conditions given in the problem
def total_red_balls : ℕ := 5
def total_blue_balls : ℕ := 11
def red_glass_balls : ℕ := 2
def red_wooden_balls : ℕ := 3
def blue_glass_balls : ℕ := 4
def blue_wooden_balls : ℕ := 7
def total_balls : ℕ := total_red_balls + total_blue_balls
def total_glass_balls : ℕ := red_glass_balls + blue_glass_balls

-- The mathematically equivalent proof problem statement.
theorem probability_blue_given_glass :
  (blue_glass_balls : ℚ) / (total_glass_balls : ℚ) = 2 / 3 := by
sorry

end probability_blue_given_glass_l126_126505


namespace steven_card_count_l126_126310

theorem steven_card_count (num_groups : ℕ) (cards_per_group : ℕ) (h_groups : num_groups = 5) (h_cards : cards_per_group = 6) : num_groups * cards_per_group = 30 := by
  sorry

end steven_card_count_l126_126310


namespace omega_in_abc_l126_126227

variables {R : Type*}
variables [LinearOrderedField R]
variables {a b c ω x y z : R} 

theorem omega_in_abc 
  (distinct_abc : a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ω ≠ a ∧ ω ≠ b ∧ ω ≠ c)
  (h1 : x + y + z = 1)
  (h2 : a^2 * x + b^2 * y + c^2 * z = ω^2)
  (h3 : a^3 * x + b^3 * y + c^3 * z = ω^3)
  (h4 : a^4 * x + b^4 * y + c^4 * z = ω^4):
  ω = a ∨ ω = b ∨ ω = c :=
sorry

end omega_in_abc_l126_126227


namespace probability_of_a_plus_b_gt_5_l126_126706

noncomputable def all_events : Finset (ℕ × ℕ) := 
  { (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4) }

noncomputable def successful_events : Finset (ℕ × ℕ) :=
  { (2, 4), (3, 3), (3, 4) }

theorem probability_of_a_plus_b_gt_5 : 
  (successful_events.card : ℚ) / (all_events.card : ℚ) = 1 / 3 := by
  sorry

end probability_of_a_plus_b_gt_5_l126_126706


namespace circle_center_and_sum_l126_126139

/-- Given the equation of a circle x^2 + y^2 - 6x + 14y = -28,
    prove that the coordinates (h, k) of the center of the circle are (3, -7)
    and compute h + k. -/
theorem circle_center_and_sum (x y : ℝ) :
  (∃ h k, (x^2 + y^2 - 6*x + 14*y = -28) ∧ (h = 3) ∧ (k = -7) ∧ (h + k = -4)) :=
by {
  sorry
}

end circle_center_and_sum_l126_126139


namespace domain_of_f_l126_126238

def denominator (x : ℝ) : ℝ := x^2 - 4 * x + 3

def is_defined (x : ℝ) : Prop := denominator x ≠ 0

theorem domain_of_f :
  {x : ℝ // is_defined x} = {x : ℝ | x < 1} ∪ {x : ℝ | 1 < x ∧ x < 3} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_f_l126_126238


namespace second_integer_value_l126_126961

-- Definitions of conditions directly from a)
def consecutive_integers (a b c : ℤ) : Prop :=
  b = a + 1 ∧ c = b + 1

def sum_of_first_and_third (a c : ℤ) (sum : ℤ) : Prop :=
  a + c = sum

-- Translated proof problem
theorem second_integer_value (n: ℤ) (h1: consecutive_integers (n - 1) n (n + 1))
  (h2: sum_of_first_and_third (n - 1) (n + 1) 118) : 
  n = 59 :=
by
  sorry

end second_integer_value_l126_126961


namespace compound_interest_principal_l126_126498

noncomputable def compound_interest (P R T : ℝ) : ℝ :=
  P * (Real.exp (T * Real.log (1 + R / 100)) - 1)

noncomputable def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

theorem compound_interest_principal :
  let P_SI := 2800.0000000000027
  let R_SI := 5
  let T_SI := 3
  let P_CI := 4000
  let R_CI := 10
  let T_CI := 2
  let SI := simple_interest P_SI R_SI T_SI
  let CI := 2 * SI
  CI = compound_interest P_CI R_CI T_CI → P_CI = 4000 :=
by
  intros
  sorry

end compound_interest_principal_l126_126498


namespace cotangent_positives_among_sequence_l126_126843

def cotangent_positive_count (n : ℕ) : ℕ :=
  if n ≤ 2019 then
    let count := (n / 4) * 3 + if n % 4 ≠ 0 then (3 + 1 - max 0 ((n % 4) - 1)) else 0
    count
  else 0

theorem cotangent_positives_among_sequence :
  cotangent_positive_count 2019 = 1515 := sorry

end cotangent_positives_among_sequence_l126_126843


namespace rajans_position_l126_126273

theorem rajans_position
    (total_boys : ℕ)
    (vinay_position_from_right : ℕ)
    (boys_between_rajan_and_vinay : ℕ)
    (total_boys_eq : total_boys = 24)
    (vinay_position_from_right_eq : vinay_position_from_right = 10)
    (boys_between_eq : boys_between_rajan_and_vinay = 8) :
    ∃ R : ℕ, R = 6 :=
by
  sorry

end rajans_position_l126_126273


namespace equal_number_of_coins_l126_126804

theorem equal_number_of_coins (x : ℕ) (hx : 1 * x + 5 * x + 10 * x + 25 * x + 100 * x = 305) : x = 2 :=
sorry

end equal_number_of_coins_l126_126804


namespace power_function_analysis_l126_126086

theorem power_function_analysis (f : ℝ → ℝ) (α : ℝ) (h : ∀ x > 0, f x = x ^ α) (h_f : f 9 = 3) :
  (∀ x ≥ 0, f x = x ^ (1 / 2)) ∧
  (∀ x ≥ 4, f x ≥ 2) ∧
  (∀ x1 x2 : ℝ, x2 > x1 ∧ x1 > 0 → (f (x1) + f (x2)) / 2 < f ((x1 + x2) / 2)) :=
by
  -- Solution steps would go here
  sorry

end power_function_analysis_l126_126086


namespace grid_midpoint_exists_l126_126452

theorem grid_midpoint_exists (points : Fin 5 → ℤ × ℤ) :
  ∃ i j : Fin 5, i ≠ j ∧ (points i).fst % 2 = (points j).fst % 2 ∧ (points i).snd % 2 = (points j).snd % 2 :=
by 
  sorry

end grid_midpoint_exists_l126_126452


namespace average_age_of_first_and_fifth_fastest_dogs_l126_126393

-- Definitions based on the conditions
def first_dog_age := 10
def second_dog_age := first_dog_age - 2
def third_dog_age := second_dog_age + 4
def fourth_dog_age := third_dog_age / 2
def fifth_dog_age := fourth_dog_age + 20

-- Statement to prove
theorem average_age_of_first_and_fifth_fastest_dogs : 
  (first_dog_age + fifth_dog_age) / 2 = 18 := by
  -- Add your proof here
  sorry

end average_age_of_first_and_fifth_fastest_dogs_l126_126393


namespace impossible_partition_10x10_square_l126_126405

theorem impossible_partition_10x10_square :
  ¬ ∃ (x y : ℝ), (x - y = 1) ∧ (x * y = 1) ∧ (∃ (n m : ℕ), 10 = n * x + m * y ∧ n + m = 100) :=
by
  sorry

end impossible_partition_10x10_square_l126_126405


namespace area_of_triangle_POF_l126_126853

noncomputable def origin : (ℝ × ℝ) := (0, 0)
noncomputable def focus : (ℝ × ℝ) := (Real.sqrt 2, 0)

noncomputable def parabola (x y : ℝ) : Prop :=
  y ^ 2 = 4 * Real.sqrt 2 * x

noncomputable def point_on_parabola (x y : ℝ) : Prop :=
  parabola x y

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

noncomputable def PF_eq_4sqrt2 (x y : ℝ) : Prop :=
  distance x y (Real.sqrt 2) 0 = 4 * Real.sqrt 2

theorem area_of_triangle_POF (x y : ℝ) 
  (h1: point_on_parabola x y)
  (h2: PF_eq_4sqrt2 x y) :
   1 / 2 * distance 0 0 (Real.sqrt 2) 0 * |y| = 2 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_POF_l126_126853


namespace group_A_can_form_triangle_l126_126599

def can_form_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem group_A_can_form_triangle : can_form_triangle 9 6 13 :=
by
  sorry

end group_A_can_form_triangle_l126_126599


namespace simple_interest_rate_l126_126811

theorem simple_interest_rate (P : ℝ) (T : ℝ) (A : ℝ) (R : ℝ) (h : A = 3 * P) (h1 : T = 12) (h2 : A - P = (P * R * T) / 100) :
  R = 16.67 :=
by sorry

end simple_interest_rate_l126_126811


namespace calculate_f_at_2_l126_126595

def f (x : ℝ) : ℝ := 15 * x ^ 5 - 24 * x ^ 4 + 33 * x ^ 3 - 42 * x ^ 2 + 51 * x

theorem calculate_f_at_2 : f 2 = 294 := by
  sorry

end calculate_f_at_2_l126_126595


namespace items_in_descending_order_l126_126284

-- Assume we have four real numbers representing the weights of the items.
variables (C S B K : ℝ)

-- The conditions given in the problem.
axiom h1 : S > B
axiom h2 : C + B > S + K
axiom h3 : K + C = S + B

-- Define a predicate to check if the weights are in descending order.
def DescendingOrder (C S B K : ℝ) : Prop :=
  C > S ∧ S > B ∧ B > K

-- The theorem to prove the descending order of weights.
theorem items_in_descending_order : DescendingOrder C S B K :=
sorry

end items_in_descending_order_l126_126284


namespace greatest_value_x_l126_126448

theorem greatest_value_x (x: ℤ) : 
  (∃ k: ℤ, (x^2 - 5 * x + 14) = k * (x - 4)) → x ≤ 14 :=
sorry

end greatest_value_x_l126_126448


namespace play_children_count_l126_126380

theorem play_children_count (cost_adult_ticket cost_children_ticket total_receipts total_attendance adult_count children_count : ℕ) :
  cost_adult_ticket = 25 →
  cost_children_ticket = 15 →
  total_receipts = 7200 →
  total_attendance = 400 →
  adult_count = 280 →
  25 * adult_count + 15 * children_count = total_receipts →
  adult_count + children_count = total_attendance →
  children_count = 120 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end play_children_count_l126_126380


namespace monotonically_increasing_interval_l126_126483

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.cos (2 * x + φ)

noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := Real.cos ((2 / 3) * x - (5 * Real.pi / 12))

theorem monotonically_increasing_interval 
  (φ : ℝ) (h1 : -Real.pi / 2 < φ) (h2 : φ < 0) 
  (h3 : 2 * (Real.pi / 8) + φ = Real.pi / 4) : 
  ∀ x : ℝ, (-(Real.pi / 2) ≤ x) ∧ (x ≤ Real.pi / 2) ↔ ∃ k : ℤ, x ∈ [(-7 * Real.pi / 8 + 3 * k * Real.pi), (5 * Real.pi / 8 + 3 * k * Real.pi)] :=
sorry

end monotonically_increasing_interval_l126_126483


namespace correct_investment_allocation_l126_126413

noncomputable def investment_division (x : ℤ) : Prop :=
  let s := 2000
  let w := 500
  let rogers_investment := 2500
  let total_initial_capital := (5 / 2 : ℚ) * x
  let new_total_capital := total_initial_capital + rogers_investment
  let equal_share := new_total_capital / 3
  s + w = rogers_investment ∧ 
  (3 / 2 : ℚ) * x + s = equal_share ∧ 
  x + w = equal_share

theorem correct_investment_allocation (x : ℤ) (hx : 3 * x % 2 = 0) :
  x > 0 ∧ investment_division x :=
by
  sorry

end correct_investment_allocation_l126_126413


namespace salted_duck_eggs_min_cost_l126_126631

-- Define the system of equations and their solutions
def salted_duck_eggs_pricing (a b : ℕ) : Prop :=
  (9 * a + 6 * b = 390) ∧ (5 * a + 8 * b = 310)

-- Total number of boxes and constraints
def total_boxes_conditions (x y : ℕ) : Prop :=
  (x + y = 30) ∧ (x ≥ y + 5) ∧ (x ≤ 2 * y)

-- Minimize cost function given prices and constraints
def minimum_cost (x y a b : ℕ) : Prop :=
  (salted_duck_eggs_pricing a b) ∧
  (total_boxes_conditions x y) ∧
  (a = 30) ∧ (b = 20) ∧
  (10 * x + 600 = 780)

-- Statement to prove
theorem salted_duck_eggs_min_cost : ∃ x y : ℕ, minimum_cost x y 30 20 :=
by
  sorry

end salted_duck_eggs_min_cost_l126_126631


namespace find_m_l126_126424

theorem find_m (x p q m : ℝ) 
    (h1 : 4 * p^2 + 9 * q^2 = 2) 
    (h2 : (1/2) * x + 3 * p * q = 1) 
    (h3 : ∀ x, x^2 + 2 * m * x - 3 * m + 1 ≥ 1) :
    m = -3 ∨ m = 1 :=
sorry

end find_m_l126_126424


namespace max_of_three_diff_pos_int_with_mean_7_l126_126105

theorem max_of_three_diff_pos_int_with_mean_7 (a b c : ℕ) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_mean : (a + b + c) / 3 = 7) :
  max a (max b c) = 18 := 
sorry

end max_of_three_diff_pos_int_with_mean_7_l126_126105


namespace divides_expression_l126_126567

theorem divides_expression (y : ℕ) (h : y ≠ 0) : (y - 1) ∣ (y ^ (y ^ 2) - 2 * y ^ (y + 1) + 1) :=
sorry

end divides_expression_l126_126567


namespace simplify_expression_correct_l126_126061

noncomputable def simplify_expr (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :=
  ((a / b) * ((b - (4 * (a^6) / b^3)) ^ (1 / 3))
    - a^2 * ((b / a^6 - (4 / b^3)) ^ (1 / 3))
    + (2 / (a * b)) * ((a^3 * b^4 - 4 * a^9) ^ (1 / 3))) /
    ((b^2 - 2 * a^3) ^ (1 / 3) / b^2)

theorem simplify_expression_correct (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  simplify_expr a b ha hb = (a + b) * ((b^2 + 2 * a^3) ^ (1 / 3)) :=
sorry

end simplify_expression_correct_l126_126061


namespace car_with_highest_avg_speed_l126_126898

-- Conditions
def distance_A : ℕ := 715
def time_A : ℕ := 11
def distance_B : ℕ := 820
def time_B : ℕ := 12
def distance_C : ℕ := 950
def time_C : ℕ := 14

-- Average Speeds
def avg_speed_A : ℚ := distance_A / time_A
def avg_speed_B : ℚ := distance_B / time_B
def avg_speed_C : ℚ := distance_C / time_C

-- Theorem
theorem car_with_highest_avg_speed : avg_speed_B > avg_speed_A ∧ avg_speed_B > avg_speed_C :=
by
  sorry

end car_with_highest_avg_speed_l126_126898


namespace total_pencils_correct_l126_126615

def Mitchell_pencils := 30
def Antonio_pencils := Mitchell_pencils - 6
def total_pencils := Antonio_pencils + Mitchell_pencils

theorem total_pencils_correct : total_pencils = 54 := by
  sorry

end total_pencils_correct_l126_126615


namespace solve_for_x_l126_126099

theorem solve_for_x (x : ℝ) (h : x ≠ 0) (h_eq : (8 * x) ^ 16 = (32 * x) ^ 8) : x = 1 / 2 :=
by
  sorry

end solve_for_x_l126_126099


namespace red_or_black_prob_red_black_or_white_prob_l126_126519

-- Defining the probabilities
def prob_red : ℚ := 5 / 12
def prob_black : ℚ := 4 / 12
def prob_white : ℚ := 2 / 12
def prob_green : ℚ := 1 / 12

-- Question 1: Probability of drawing a red or black ball
theorem red_or_black_prob : prob_red + prob_black = 3 / 4 :=
by sorry

-- Question 2: Probability of drawing a red, black, or white ball
theorem red_black_or_white_prob : prob_red + prob_black + prob_white = 11 / 12 :=
by sorry

end red_or_black_prob_red_black_or_white_prob_l126_126519


namespace find_second_sum_l126_126246

theorem find_second_sum (S : ℤ) (x : ℤ) (h_S : S = 2678)
  (h_eq_interest : x * 3 * 8 = (S - x) * 5 * 3) : (S - x) = 1648 :=
by {
  sorry
}

end find_second_sum_l126_126246


namespace jerome_contact_list_l126_126656

def classmates := 20
def out_of_school_friends := classmates / 2
def family_members := 3
def total_contacts := classmates + out_of_school_friends + family_members

theorem jerome_contact_list : total_contacts = 33 := by
  sorry

end jerome_contact_list_l126_126656


namespace Jamie_owns_2_Maine_Coons_l126_126111

-- Definitions based on conditions
variables (Jamie_MaineCoons Gordon_MaineCoons Hawkeye_MaineCoons Jamie_Persians Gordon_Persians Hawkeye_Persians : ℕ)

-- Conditions
axiom Jamie_owns_4_Persians : Jamie_Persians = 4
axiom Gordon_owns_half_as_many_Persians_as_Jamie : Gordon_Persians = Jamie_Persians / 2
axiom Gordon_owns_one_more_Maine_Coon_than_Jamie : Gordon_MaineCoons = Jamie_MaineCoons + 1
axiom Hawkeye_owns_one_less_Maine_Coon_than_Gordon : Hawkeye_MaineCoons = Gordon_MaineCoons - 1
axiom Hawkeye_owns_no_Persian_cats : Hawkeye_Persians = 0
axiom total_number_of_cats_is_13 : Jamie_Persians + Jamie_MaineCoons + Gordon_Persians + Gordon_MaineCoons + Hawkeye_Persians + Hawkeye_MaineCoons = 13

-- Theorem statement
theorem Jamie_owns_2_Maine_Coons : Jamie_MaineCoons = 2 :=
by {
  -- Ideally, you would provide the proof here, stepping through algebraically as shown in the solution,
  -- but we are skipping the proof as specified in the instructions.
  sorry
}

end Jamie_owns_2_Maine_Coons_l126_126111


namespace original_number_is_3_l126_126701

theorem original_number_is_3 
  (A B C D E : ℝ) 
  (h1 : (A + B + C + D + E) / 5 = 8) 
  (h2 : (8 + B + C + D + E) / 5 = 9): 
  A = 3 :=
sorry

end original_number_is_3_l126_126701


namespace range_of_m_l126_126275

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (2 * x + m) / (x - 2) + (x - 1) / (2 - x) = 3) ↔ (m > -7 ∧ m ≠ -3) :=
by
  sorry

end range_of_m_l126_126275


namespace area_of_triangle_ABC_l126_126040

theorem area_of_triangle_ABC 
  (ABCD_is_trapezoid : ∀ {a b c d : ℝ}, a + d = b + c)
  (area_ABCD : ∀ {a b : ℝ}, a * b = 24)
  (CD_three_times_AB : ∀ {a : ℝ}, a * 3 = 24) :
  ∃ (area_ABC : ℝ), area_ABC = 6 :=
by 
  sorry

end area_of_triangle_ABC_l126_126040


namespace pyarelal_loss_l126_126451

theorem pyarelal_loss (P : ℝ) (total_loss : ℝ) (ashok_ratio pyarelal_ratio : ℝ)
  (h1 : ashok_ratio = 1/9) (h2 : pyarelal_ratio = 1)
  (h3 : total_loss = 2000) : (pyarelal_ratio / (ashok_ratio + pyarelal_ratio)) * total_loss = 1800 :=
by
  sorry

end pyarelal_loss_l126_126451


namespace problem_relation_l126_126415

-- Definitions indicating relationships.
def related₁ : Prop := ∀ (s : ℝ), (s ≥ 0) → (∃ a p : ℝ, a = s^2 ∧ p = 4 * s)
def related₂ : Prop := ∀ (d t : ℝ), (t > 0) → (∃ v : ℝ, d = v * t)
def related₃ : Prop := ∃ (h w : ℝ) (f : ℝ → ℝ), w = f h
def related₄ : Prop := ∀ (h : ℝ) (v : ℝ), False

-- The theorem stating that A, B, and C are related.
theorem problem_relation : 
  related₁ ∧ related₂ ∧ related₃ ∧ ¬ related₄ :=
by sorry

end problem_relation_l126_126415


namespace overall_average_speed_is_six_l126_126960

-- Definitions of the conditions
def cycling_time := 45 / 60 -- hours
def cycling_speed := 12 -- mph
def stopping_time := 15 / 60 -- hours
def walking_time := 75 / 60 -- hours
def walking_speed := 3 -- mph

-- Problem statement: Proving that the overall average speed is 6 mph
theorem overall_average_speed_is_six : 
  (cycling_speed * cycling_time + walking_speed * walking_time) /
  (cycling_time + walking_time + stopping_time) = 6 :=
by
  sorry

end overall_average_speed_is_six_l126_126960


namespace find_a_l126_126259

noncomputable def f (x a : ℝ) : ℝ := 4 * x ^ 2 - 4 * a * x + a ^ 2 - 2 * a + 2

theorem find_a (a : ℝ) : 
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧  ∀ y : ℝ, 0 ≤ y ∧ y ≤ 2 → f y a ≤ f x a) ∧ f 0 a = 3 ∧ f 2 a = 3 → 
  a = 5 - Real.sqrt 10 ∨ a = 1 + Real.sqrt 2 := 
sorry

end find_a_l126_126259


namespace chord_line_equation_l126_126470

/-- 
  Given the parabola y^2 = 4x and a chord AB 
  that exactly bisects at point P(1,1), prove 
  that the equation of the line on which chord AB lies is 2x - y - 1 = 0.
-/
theorem chord_line_equation (x y : ℝ) 
  (hx : y^2 = 4 * x)
  (bisect : ∃ A B : ℝ × ℝ, 
             (A.1^2 = 4 * A.2) ∧ (B.1^2 = 4 * B.2) ∧
             (A.1 + B.1 = 2 * 1) ∧ (A.2 + B.2 = 2 * 1)) :
  2 * x - y - 1 = 0 := sorry

end chord_line_equation_l126_126470


namespace find_m_l126_126860

theorem find_m (m : ℝ) (h1 : (∀ x : ℝ, (x^2 - m) * (x + m) = x^3 + m * (x^2 - x - 12))) (h2 : m ≠ 0) : m = 12 :=
by
  sorry

end find_m_l126_126860


namespace total_population_l126_126427

theorem total_population (b g t : ℕ) (h₁ : b = 6 * g) (h₂ : g = 5 * t) :
  b + g + t = 36 * t :=
by
  sorry

end total_population_l126_126427


namespace smallest_four_digit_equiv_8_mod_9_l126_126908

theorem smallest_four_digit_equiv_8_mod_9 :
  ∃ n : ℕ, n % 9 = 8 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 9 = 8 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m :=
sorry

end smallest_four_digit_equiv_8_mod_9_l126_126908


namespace problem_statement_l126_126409

-- Define the arithmetic sequence and the conditions
noncomputable def a : ℕ → ℝ := sorry
axiom a_arith_seq : ∃ d : ℝ, ∀ n m : ℕ, a (n + m) = a n + m • d
axiom condition : a 4 + a 10 + a 16 = 30

-- State the theorem
theorem problem_statement : a 18 - 2 * a 14 = -10 :=
sorry

end problem_statement_l126_126409


namespace proof_OPQ_Constant_l126_126882

open Complex

def OPQ_Constant :=
  ∀ (z1 z2 : ℂ) (θ : ℝ), abs z1 = 5 ∧
    (z1^2 - z1 * z2 * Real.sin θ + z2^2 = 0) →
      abs z2 = 5

theorem proof_OPQ_Constant : OPQ_Constant :=
by
  sorry

end proof_OPQ_Constant_l126_126882


namespace igor_reach_top_time_l126_126591

-- Define the conditions
def cabins_numbered_consecutively := (1, 99)
def igor_initial_cabin := 42
def first_aligned_cabin := 13
def second_aligned_cabin := 12
def alignment_time := 15
def total_cabins := 99
def expected_time := 17 * 60 + 15

-- State the problem as a theorem
theorem igor_reach_top_time :
  ∃ t, t = expected_time ∧
  -- Assume the cabins are numbered consecutively
  cabins_numbered_consecutively = (1, total_cabins) ∧
  -- Igor starts in cabin #42
  igor_initial_cabin = 42 ∧
  -- Cabin #42 first aligns with cabin #13, then aligns with cabin #12, 15 seconds later
  first_aligned_cabin = 13 ∧
  second_aligned_cabin = 12 ∧
  alignment_time = 15 :=
sorry

end igor_reach_top_time_l126_126591


namespace cow_problem_l126_126438

noncomputable def problem_statement : Prop :=
  ∃ (F M : ℕ), F + M = 300 ∧
               (∃ S H : ℕ, S = 1/2 * F ∧ H = 1/2 * M ∧ S = H + 50) ∧
               F = 2 * M

theorem cow_problem : problem_statement :=
sorry

end cow_problem_l126_126438


namespace stratified_sampling_grade10_sampled_count_l126_126573

def total_students : ℕ := 2000
def grade10_students : ℕ := 600
def grade11_students : ℕ := 680
def grade12_students : ℕ := 720
def total_sampled_students : ℕ := 50

theorem stratified_sampling_grade10_sampled_count :
  15 = (total_sampled_students * grade10_students / total_students) :=
by sorry

end stratified_sampling_grade10_sampled_count_l126_126573


namespace intersection_point_a_l126_126935

-- Definitions for the given conditions 
def f (x : ℤ) (b : ℤ) : ℤ := 3 * x + b
def f_inv (x : ℤ) (b : ℤ) : ℤ := (x - b) / 3 -- Considering that f is invertible for integer b

-- The problem statement
theorem intersection_point_a (a b : ℤ) (h1 : a = f (-3) b) (h2 : a = f_inv (-3)) (h3 : f (-3) b = -3):
  a = -3 := sorry

end intersection_point_a_l126_126935


namespace polynomial_evaluation_l126_126165

theorem polynomial_evaluation 
  (x : ℝ) (h : x^2 - 3*x - 10 = 0 ∧ x > 0) :
  x^4 - 3*x^3 - 4*x^2 + 12*x + 9 = 219 :=
sorry

end polynomial_evaluation_l126_126165


namespace steven_has_15_more_peaches_than_jill_l126_126551

-- Definitions based on conditions
def peaches_jill : ℕ := 12
def peaches_jake : ℕ := peaches_jill - 1
def peaches_steven : ℕ := peaches_jake + 16

-- The proof problem
theorem steven_has_15_more_peaches_than_jill : peaches_steven - peaches_jill = 15 := by
  sorry

end steven_has_15_more_peaches_than_jill_l126_126551


namespace original_flour_quantity_l126_126102

-- Definitions based on conditions
def flour_called (x : ℝ) : Prop := 
  -- total flour Mary uses is x + extra 2 cups, which equals to 9 cups.
  x + 2 = 9

-- The proof statement we need to show
theorem original_flour_quantity : ∃ x : ℝ, flour_called x ∧ x = 7 := 
  sorry

end original_flour_quantity_l126_126102


namespace probability_not_snowing_l126_126429

theorem probability_not_snowing (P_snowing : ℚ) (h : P_snowing = 2/7) :
  (1 - P_snowing) = 5/7 :=
sorry

end probability_not_snowing_l126_126429


namespace children_neither_happy_nor_sad_l126_126644

theorem children_neither_happy_nor_sad (total_children happy_children sad_children : ℕ)
  (total_boys total_girls happy_boys sad_girls boys_neither_happy_nor_sad : ℕ)
  (h₀ : total_children = 60)
  (h₁ : happy_children = 30)
  (h₂ : sad_children = 10)
  (h₃ : total_boys = 19)
  (h₄ : total_girls = 41)
  (h₅ : happy_boys = 6)
  (h₆ : sad_girls = 4)
  (h₇ : boys_neither_happy_nor_sad = 7) :
  total_children - happy_children - sad_children = 20 :=
by
  sorry

end children_neither_happy_nor_sad_l126_126644


namespace factorize_x_squared_minus_25_l126_126554

theorem factorize_x_squared_minus_25 : ∀ (x : ℝ), (x^2 - 25) = (x + 5) * (x - 5) :=
by
  intros x
  sorry

end factorize_x_squared_minus_25_l126_126554


namespace total_strength_college_l126_126015

-- Defining the conditions
def C : ℕ := 500
def B : ℕ := 600
def Both : ℕ := 220

-- Declaring the theorem
theorem total_strength_college : (C + B - Both) = 880 :=
by
  -- The proof is not required, put sorry
  sorry

end total_strength_college_l126_126015


namespace evaluate_expression_l126_126518

theorem evaluate_expression (x : ℝ) : (1 - x^2) * (1 + x^4) = 1 - x^2 + x^4 - x^6 :=
by
  sorry

end evaluate_expression_l126_126518


namespace nancy_deleted_files_correct_l126_126080

-- Variables and conditions
def nancy_original_files : Nat := 43
def files_per_folder : Nat := 6
def number_of_folders : Nat := 2

-- Definition of the number of files that were deleted
def nancy_files_deleted : Nat :=
  nancy_original_files - (files_per_folder * number_of_folders)

-- Theorem to prove
theorem nancy_deleted_files_correct :
  nancy_files_deleted = 31 :=
by
  sorry

end nancy_deleted_files_correct_l126_126080


namespace probability_two_points_one_unit_apart_l126_126093

def twelve_points_probability : ℚ := 2 / 11

/-- Twelve points are spaced around at intervals of one unit around a \(3 \times 3\) square.
    Two of the 12 points are chosen at random.
    Prove that the probability that the two points are one unit apart is \(\frac{2}{11}\). -/
theorem probability_two_points_one_unit_apart :
  let total_points := 12
  let total_combinations := (total_points * (total_points - 1)) / 2
  let favorable_pairs := 12
  (favorable_pairs : ℚ) / total_combinations = twelve_points_probability := by
  sorry

end probability_two_points_one_unit_apart_l126_126093


namespace divisible_by_56_l126_126700

theorem divisible_by_56 (n : ℕ) (h1 : ∃ k, 3 * n + 1 = k * k) (h2 : ∃ m, 4 * n + 1 = m * m) : 56 ∣ n := 
sorry

end divisible_by_56_l126_126700


namespace number_of_real_roots_l126_126377

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem number_of_real_roots (a : ℝ) :
    ((|a| < (2 * Real.sqrt 3) / 9) → (∃ x₁ x₂ x₃ : ℝ, f x₁ = a ∧ f x₂ = a ∧ f x₃ = a ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)) ∧
    ((|a| > (2 * Real.sqrt 3) / 9) → (∃ x : ℝ, f x = a ∧ ∀ y : ℝ, f y = a → y = x)) ∧
    ((|a| = (2 * Real.sqrt 3) / 9) → (∃ x₁ x₂ : ℝ, f x₁ = a ∧ f x₂ = a ∧ x₁ ≠ x₂ ∧ ∀ y : ℝ, (f y = a → (y = x₁ ∨ y = x₂)) ∧ (x₁ = x₂ ∨ ∀ z : ℝ, (f z = a → z = x₁ ∨ z = x₂)))) := sorry

end number_of_real_roots_l126_126377


namespace mary_flour_total_l126_126632

-- Definitions for conditions
def initial_flour : ℝ := 7.0
def extra_flour : ℝ := 2.0
def total_flour (x y : ℝ) : ℝ := x + y

-- The statement we want to prove
theorem mary_flour_total : total_flour initial_flour extra_flour = 9.0 := 
by sorry

end mary_flour_total_l126_126632


namespace multiple_of_four_l126_126422

open BigOperators

theorem multiple_of_four (n : ℕ) (x y z : Fin n → ℤ)
  (hx : ∀ i, x i = 1 ∨ x i = -1)
  (hy : ∀ i, y i = 1 ∨ y i = -1)
  (hz : ∀ i, z i = 1 ∨ z i = -1)
  (hxy : ∑ i, x i * y i = 0)
  (hxz : ∑ i, x i * z i = 0)
  (hyz : ∑ i, y i * z i = 0) :
  (n % 4 = 0) :=
sorry

end multiple_of_four_l126_126422


namespace tire_usage_l126_126742

theorem tire_usage (total_distance : ℕ) (num_tires : ℕ) (active_tires : ℕ) 
  (h1 : total_distance = 45000) 
  (h2 : num_tires = 5) 
  (h3 : active_tires = 4) 
  (equal_usage : (total_distance * active_tires) / num_tires = 36000) : 
  (∀ tire, tire < num_tires → used_miles_per_tire = 36000) := 
by
  sorry

end tire_usage_l126_126742


namespace convex_pentadecagon_diagonals_l126_126229

def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem convex_pentadecagon_diagonals :
  number_of_diagonals 15 = 90 :=
by sorry

end convex_pentadecagon_diagonals_l126_126229


namespace masking_tape_needed_l126_126457

def wall1_width : ℝ := 4
def wall1_count : ℕ := 2
def wall2_width : ℝ := 6
def wall2_count : ℕ := 2
def door_width : ℝ := 2
def door_count : ℕ := 1
def window_width : ℝ := 1.5
def window_count : ℕ := 2

def total_width_of_walls : ℝ := (wall1_count * wall1_width) + (wall2_count * wall2_width)
def total_width_of_door_and_windows : ℝ := (door_count * door_width) + (window_count * window_width)

theorem masking_tape_needed : total_width_of_walls - total_width_of_door_and_windows = 15 := by
  sorry

end masking_tape_needed_l126_126457


namespace integer_solutions_count_l126_126797

theorem integer_solutions_count :
  let cond1 (x : ℤ) := -4 * x ≥ 2 * x + 9
  let cond2 (x : ℤ) := -3 * x ≤ 15
  let cond3 (x : ℤ) := -5 * x ≥ x + 22
  ∃ s : Finset ℤ, 
    (∀ x ∈ s, cond1 x ∧ cond2 x ∧ cond3 x) ∧
    (∀ x, cond1 x ∧ cond2 x ∧ cond3 x → x ∈ s) ∧
    s.card = 2 :=
sorry

end integer_solutions_count_l126_126797


namespace shift_line_one_unit_left_l126_126083

theorem shift_line_one_unit_left : ∀ (x y : ℝ), (y = x) → (y - 1 = (x + 1) - 1) :=
by
  intros x y h
  sorry

end shift_line_one_unit_left_l126_126083


namespace resulting_chemical_percentage_l126_126736

theorem resulting_chemical_percentage 
  (init_solution_pct : ℝ) (replacement_frac : ℝ) (replacing_solution_pct : ℝ) (resulting_solution_pct : ℝ) : 
  init_solution_pct = 0.85 →
  replacement_frac = 0.8181818181818182 →
  replacing_solution_pct = 0.30 →
  resulting_solution_pct = 0.40 :=
by
  intros h1 h2 h3
  sorry

end resulting_chemical_percentage_l126_126736


namespace boat_speed_greater_than_current_l126_126171

theorem boat_speed_greater_than_current (U V : ℝ) (hU_gt_V : U > V)
  (h_equation : 1 / (U - V) - 1 / (U + V) + 1 / (2 * V + 1) = 1) :
  U - V = 1 :=
sorry

end boat_speed_greater_than_current_l126_126171


namespace sum_of_squares_of_chords_in_sphere_l126_126820

-- Defining variables
variables (R PO : ℝ)

-- Define the problem statement
theorem sum_of_squares_of_chords_in_sphere
  (chord_lengths_squared : ℝ)
  (H_chord_lengths_squared : chord_lengths_squared = 3 * R^2 - 2 * PO^2) :
  chord_lengths_squared = 3 * R^2 - 2 * PO^2 :=
by
  sorry -- proof is omitted

end sum_of_squares_of_chords_in_sphere_l126_126820


namespace range_of_a_l126_126737

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x_0 : ℝ, x_0^2 + (a - 1) * x_0 + 1 ≤ 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end range_of_a_l126_126737


namespace bus_ride_cost_l126_126454

variable (cost_bus cost_train : ℝ)

-- Condition 1: cost_train = cost_bus + 2.35
#check (cost_train = cost_bus + 2.35)

-- Condition 2: cost_bus + cost_train = 9.85
#check (cost_bus + cost_train = 9.85)

theorem bus_ride_cost :
  (∃ (cost_bus cost_train : ℝ),
    cost_train = cost_bus + 2.35 ∧
    cost_bus + cost_train = 9.85) →
  cost_bus = 3.75 :=
sorry

end bus_ride_cost_l126_126454


namespace integer_solutions_to_system_l126_126836

theorem integer_solutions_to_system (x y z : ℤ) (h1 : x + y + z = 2) (h2 : x^3 + y^3 + z^3 = -10) :
  (x = 3 ∧ y = 3 ∧ z = -4) ∨
  (x = 3 ∧ y = -4 ∧ z = 3) ∨
  (x = -4 ∧ y = 3 ∧ z = 3) :=
sorry

end integer_solutions_to_system_l126_126836


namespace minimum_sum_of_dimensions_l126_126290

theorem minimum_sum_of_dimensions {a b c : ℕ} (h1 : a * b * c = 2310) (h2 : a > 0) (h3 : b > 0) (h4 : c > 0) :
  a + b + c ≥ 42 := 
sorry

end minimum_sum_of_dimensions_l126_126290


namespace imag_part_z_is_3_l126_126520

namespace ComplexMultiplication

-- Define the imaginary unit i
def i := Complex.I

-- Define the complex number z
def z := (1 + 2 * i) * (2 - i)

-- Define the imaginary part of a complex number
def imag_part (z : ℂ) : ℂ := Complex.im z

-- Statement to prove: The imaginary part of z = 3
theorem imag_part_z_is_3 : imag_part z = 3 := by
  sorry

end ComplexMultiplication

end imag_part_z_is_3_l126_126520


namespace suraj_next_innings_runs_l126_126473

variable (A R : ℕ)

def suraj_average_eq (A : ℕ) : Prop :=
  A + 8 = 128

def total_runs_eq (A R : ℕ) : Prop :=
  9 * A + R = 10 * 128

theorem suraj_next_innings_runs :
  ∃ A : ℕ, suraj_average_eq A ∧ ∃ R : ℕ, total_runs_eq A R ∧ R = 200 := 
by
  sorry

end suraj_next_innings_runs_l126_126473


namespace bacteria_growth_rate_l126_126044

-- Define the existence of the growth rate and the initial amount of bacteria
variable (B : ℕ → ℝ) (B0 : ℝ) (r : ℝ)

-- State the conditions from the problem
axiom bacteria_growth_model : ∀ t : ℕ, B t = B0 * r ^ t
axiom day_30_full : B 30 = B0 * r ^ 30
axiom day_26_sixteenth : B 26 = (1 / 16) * B 30

-- Theorem stating that the growth rate r of the bacteria each day is 2
theorem bacteria_growth_rate : r = 2 := by
  sorry

end bacteria_growth_rate_l126_126044


namespace sugar_percentage_first_solution_l126_126266

theorem sugar_percentage_first_solution 
  (x : ℝ) (h1 : 0 < x ∧ x < 100) 
  (h2 : 17 = 3 / 4 * x + 1 / 4 * 38) : 
  x = 10 :=
sorry

end sugar_percentage_first_solution_l126_126266


namespace largest_garden_is_candace_and_difference_is_100_l126_126186

-- Define the dimensions of the gardens
def area_alice : Nat := 30 * 50
def area_bob : Nat := 35 * 45
def area_candace : Nat := 40 * 40

-- The proof goal
theorem largest_garden_is_candace_and_difference_is_100 :
  area_candace > area_alice ∧ area_candace > area_bob ∧ area_candace - area_alice = 100 := by
    sorry

end largest_garden_is_candace_and_difference_is_100_l126_126186


namespace part1_part2_l126_126597

-- Definition of the operation '※'
def operation (a b : ℝ) : ℝ := a^2 - b^2

-- Part 1: Proving 2※(-4) = -12
theorem part1 : operation 2 (-4) = -12 := 
by
  sorry

-- Part 2: Proving the solutions to the equation (x + 5)※3 = 0 are x = -8 and x = -2
theorem part2 : (∃ x : ℝ, operation (x + 5) 3 = 0) ↔ (x = -8 ∨ x = -2) := 
by
  sorry

end part1_part2_l126_126597


namespace min_a2_plus_b2_quartic_eq_l126_126345

theorem min_a2_plus_b2_quartic_eq (a b : ℝ) (x : ℝ) 
  (h : x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : 
  a^2 + b^2 ≥ 4/5 := 
sorry

end min_a2_plus_b2_quartic_eq_l126_126345


namespace sum_of_roots_l126_126735

theorem sum_of_roots (a b : Real) (h : (x - 7)^2 = 16):
  a + b = 14 :=
sorry

end sum_of_roots_l126_126735


namespace smallest_quotient_is_1_9_l126_126366

def is_two_digit_number (n : ℕ) : Prop :=
  10 <= n ∧ n <= 99

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  let x := n / 10
  let y := n % 10
  x + y

noncomputable def quotient (n : ℕ) : ℚ :=
  n / (sum_of_digits n)

theorem smallest_quotient_is_1_9 :
  ∃ n, is_two_digit_number n ∧ (∃ x y, n = 10 * x + y ∧ x ≠ y ∧ 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9) ∧ quotient n = 1.9 := 
sorry

end smallest_quotient_is_1_9_l126_126366


namespace range_of_a_l126_126739

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + 1 / 2 ≤ 0) ↔ (-1 < a ∧ a < 3) := 
sorry

end range_of_a_l126_126739


namespace common_ratio_is_two_l126_126835

-- Given a geometric sequence with specific terms
variable (a : ℕ → ℝ) (q : ℝ)

-- Conditions: all terms are positive, a_2 = 3, a_6 = 48
axiom pos_terms : ∀ n, a n > 0
axiom a2_eq : a 2 = 3
axiom a6_eq : a 6 = 48

-- Question: Prove the common ratio q is 2
theorem common_ratio_is_two :
  (∀ n, a n = a 1 * q ^ (n - 1)) → q = 2 :=
by
  sorry

end common_ratio_is_two_l126_126835


namespace longer_part_length_l126_126562

-- Conditions
def total_length : ℕ := 180
def diff_length : ℕ := 32

-- Hypothesis for the shorter part of the wire
def shorter_part (x : ℕ) : Prop :=
  x + (x + diff_length) = total_length

-- The goal is to find the longer part's length
theorem longer_part_length (x : ℕ) (h : shorter_part x) : x + diff_length = 106 := by
  sorry

end longer_part_length_l126_126562


namespace parallel_lines_m_values_l126_126572

theorem parallel_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, (3 + m) * x + 4 * y = 5) ∧ (2 * x + (5 + m) * y = 8) → (m = -1 ∨ m = -7) :=
by
  sorry

end parallel_lines_m_values_l126_126572


namespace number_of_a_l126_126665

theorem number_of_a (h : ∃ a : ℝ, ∃! x : ℝ, |x^2 + 2 * a * x + 3 * a| ≤ 2) : 
  ∃! a : ℝ, ∃! x : ℝ, |x^2 + 2 * a * x + 3 * a| ≤ 2 :=
sorry

end number_of_a_l126_126665


namespace initial_profit_percentage_l126_126676

theorem initial_profit_percentage
  (CP : ℝ)
  (h1 : CP = 2400)
  (h2 : ∀ SP : ℝ, 15 / 100 * CP = 120 + SP) :
  ∃ P : ℝ, (P / 100) * CP = 10 :=
by
  sorry

end initial_profit_percentage_l126_126676


namespace original_number_is_seven_l126_126103

theorem original_number_is_seven (x : ℕ) (h : 3 * x - 5 = 16) : x = 7 := by
sorry

end original_number_is_seven_l126_126103


namespace emily_disproved_jacob_by_turnover_5_and_7_l126_126370

def is_vowel (c : Char) : Prop :=
  c = 'A'

def is_consonant (c : Char) : Prop :=
  ¬ is_vowel c

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

def card_A_is_vowel : Prop := is_vowel 'A'
def card_1_is_odd : Prop := ¬ is_even 1 ∧ ¬ is_prime 1
def card_8_is_even : Prop := is_even 8 ∧ ¬ is_prime 8
def card_R_is_consonant : Prop := is_consonant 'R'
def card_S_is_consonant : Prop := is_consonant 'S'
def card_5_conditions : Prop := ¬ is_even 5 ∧ is_prime 5
def card_7_conditions : Prop := ¬ is_even 7 ∧ is_prime 7

theorem emily_disproved_jacob_by_turnover_5_and_7 :
  card_5_conditions ∧ card_7_conditions →
  (∃ (c : Char), (is_prime 5 ∧ is_consonant c)) ∨
  (∃ (c : Char), (is_prime 7 ∧ is_consonant c)) :=
by sorry

end emily_disproved_jacob_by_turnover_5_and_7_l126_126370


namespace copper_tin_alloy_weight_l126_126810

theorem copper_tin_alloy_weight :
  let c1 := (4/5 : ℝ) * 10 -- Copper in the first alloy
  let t1 := (1/5 : ℝ) * 10 -- Tin in the first alloy
  let c2 := (1/4 : ℝ) * 16 -- Copper in the second alloy
  let t2 := (3/4 : ℝ) * 16 -- Tin in the second alloy
  let x := ((3 * 14 - 24) / 2 : ℝ) -- Pure copper added
  let total_copper := c1 + c2 + x
  let total_tin := t1 + t2
  total_copper + total_tin = 35 := 
by
  sorry

end copper_tin_alloy_weight_l126_126810


namespace B_pow_99_identity_l126_126207

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem B_pow_99_identity : (B ^ 99) = 1 := by
  sorry

end B_pow_99_identity_l126_126207


namespace alice_total_distance_correct_l126_126151

noncomputable def alice_daily_morning_distance : ℕ := 10

noncomputable def alice_daily_afternoon_distance : ℕ := 12

noncomputable def alice_daily_distance : ℕ :=
  alice_daily_morning_distance + alice_daily_afternoon_distance

noncomputable def alice_weekly_distance : ℕ :=
  5 * alice_daily_distance

theorem alice_total_distance_correct :
  alice_weekly_distance = 110 :=
by
  unfold alice_weekly_distance alice_daily_distance alice_daily_morning_distance alice_daily_afternoon_distance
  norm_num

end alice_total_distance_correct_l126_126151


namespace regular_polygon_sides_l126_126196

theorem regular_polygon_sides (h : ∀ n : ℕ, 140 * n = 180 * (n - 2)) : n = 9 :=
sorry

end regular_polygon_sides_l126_126196


namespace FGH_supermarkets_US_l126_126672

/-- There are 60 supermarkets in the FGH chain,
all of them are either in the US or Canada,
there are 14 more FGH supermarkets in the US than in Canada.
Prove that there are 37 FGH supermarkets in the US. -/
theorem FGH_supermarkets_US (C U : ℕ) (h1 : C + U = 60) (h2 : U = C + 14) : U = 37 := by
  sorry

end FGH_supermarkets_US_l126_126672


namespace set_intersection_complement_l126_126660

open Set

noncomputable def A : Set ℝ := { x | abs (x - 1) > 2 }
noncomputable def B : Set ℝ := { x | x^2 - 6 * x + 8 < 0 }
noncomputable def notA : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }
noncomputable def targetSet : Set ℝ := { x | 2 < x ∧ x ≤ 3 }

theorem set_intersection_complement :
  (notA ∩ B) = targetSet :=
  by
  sorry

end set_intersection_complement_l126_126660


namespace mod_residue_17_l126_126792

theorem mod_residue_17 : (513 + 3 * 68 + 9 * 289 + 2 * 34 - 10) % 17 = 7 := by
  -- We first compute the modulo 17 residue of each term given in the problem:
  -- 513 == 0 % 17
  -- 68 == 0 % 17
  -- 289 == 0 % 17
  -- 34 == 0 % 17
  -- -10 == 7 % 17
  sorry

end mod_residue_17_l126_126792


namespace solve_for_x_l126_126855

theorem solve_for_x :
  ∃ x : ℕ, (12 ^ 3) * (6 ^ x) / 432 = 144 ∧ x = 2 := by
  sorry

end solve_for_x_l126_126855


namespace johns_average_speed_remaining_duration_l126_126504

noncomputable def average_speed_remaining_duration : ℝ :=
  let total_distance := 150
  let total_time := 3
  let first_hour_speed := 45
  let stop_time := 0.5
  let next_45_minutes_speed := 50
  let next_45_minutes_time := 0.75
  let driving_time := total_time - stop_time
  let distance_first_hour := first_hour_speed * 1
  let distance_next_45_minutes := next_45_minutes_speed * next_45_minutes_time
  let remaining_distance := total_distance - distance_first_hour - distance_next_45_minutes
  let remaining_time := driving_time - (1 + next_45_minutes_time)
  remaining_distance / remaining_time

theorem johns_average_speed_remaining_duration : average_speed_remaining_duration = 90 := by
  sorry

end johns_average_speed_remaining_duration_l126_126504


namespace percent_students_two_novels_l126_126578

theorem percent_students_two_novels :
  let total_students := 240
  let students_three_or_more := (1/6 : ℚ) * total_students
  let students_one := (5/12 : ℚ) * total_students
  let students_none := 16
  let students_two := total_students - students_three_or_more - students_one - students_none
  (students_two / total_students) * 100 = 35 := 
by
  sorry

end percent_students_two_novels_l126_126578


namespace find_first_month_sales_l126_126861

noncomputable def avg_sales (sales_1 sales_2 sales_3 sales_4 sales_5 sales_6 : ℕ) : ℕ :=
(sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / 6

theorem find_first_month_sales :
  let sales_2 := 6927
  let sales_3 := 6855
  let sales_4 := 7230
  let sales_5 := 6562
  let sales_6 := 5091
  let avg_sales_needed := 6500
  ∃ sales_1, avg_sales sales_1 sales_2 sales_3 sales_4 sales_5 sales_6 = avg_sales_needed := 
by
  sorry

end find_first_month_sales_l126_126861


namespace num_integer_solutions_l126_126433

def circle_center := (3, 3)
def circle_radius := 10

theorem num_integer_solutions :
  (∃ f : ℕ, f = 15) :=
sorry

end num_integer_solutions_l126_126433


namespace probability_yellow_or_blue_twice_l126_126082

theorem probability_yellow_or_blue_twice :
  let total_faces := 12
  let yellow_faces := 4
  let blue_faces := 2
  let probability_yellow_or_blue := (yellow_faces / total_faces) + (blue_faces / total_faces)
  (probability_yellow_or_blue * probability_yellow_or_blue) = 1 / 4 := 
by
  sorry

end probability_yellow_or_blue_twice_l126_126082


namespace sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l126_126828

-- Definition of conditions
variables {a b c d : ℝ} 

-- First proof statement
theorem sum_of_fifth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := 
sorry

-- Second proof statement
theorem cannot_conclude_sum_of_fourth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬(a^4 + b^4 = c^4 + d^4) := 
sorry

end sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l126_126828


namespace number_of_candidates_l126_126583

theorem number_of_candidates (n : ℕ) (h : n * (n - 1) = 132) : n = 12 :=
by
  sorry

end number_of_candidates_l126_126583


namespace percentage_sold_correct_l126_126178

variables 
  (initial_cost : ℝ) 
  (tripled_value : ℝ) 
  (selling_price : ℝ) 
  (percentage_sold : ℝ)

def game_sold_percentage (initial_cost tripled_value selling_price percentage_sold : ℝ) :=
  tripled_value = initial_cost * 3 ∧ 
  selling_price = 240 ∧ 
  initial_cost = 200 ∧ 
  percentage_sold = (selling_price / tripled_value) * 100

theorem percentage_sold_correct : game_sold_percentage 200 (200 * 3) 240 40 :=
  by simp [game_sold_percentage]; sorry

end percentage_sold_correct_l126_126178


namespace simplify_expression_l126_126627

theorem simplify_expression (x : ℝ) (h : x^2 + 2 * x - 6 = 0) : 
  ((x - 1) / (x - 3) - (x + 1) / x) / ((x^2 + 3 * x) / (x^2 - 6 * x + 9)) = -1/2 := 
by
  sorry

end simplify_expression_l126_126627


namespace fraction_order_l126_126278

theorem fraction_order :
  (21:ℚ) / 17 < (23:ℚ) / 18 ∧ (23:ℚ) / 18 < (25:ℚ) / 19 :=
by
  sorry

end fraction_order_l126_126278


namespace geometric_sequence_product_l126_126109

theorem geometric_sequence_product (a b : ℝ) (h : 2 * b = a * 16) : a * b = 32 :=
sorry

end geometric_sequence_product_l126_126109


namespace focus_coordinates_of_parabola_l126_126975

def parabola_focus_coordinates (x y : ℝ) : Prop :=
  x^2 + y = 0 ∧ (0, -1/4) = (0, y)

theorem focus_coordinates_of_parabola (x y : ℝ) :
  parabola_focus_coordinates x y →
  (0, y) = (0, -1/4) := by
  sorry

end focus_coordinates_of_parabola_l126_126975


namespace neg_p_true_l126_126461

theorem neg_p_true :
  (∃ x : ℝ, x^2 ≤ 0) :=
sorry

end neg_p_true_l126_126461


namespace relationship_of_exponents_l126_126874

theorem relationship_of_exponents (m p r s : ℝ) (u v w t : ℝ) (h1 : m^u = r) (h2 : p^v = r) (h3 : p^w = s) (h4 : m^t = s) : u * v = w * t :=
by
  sorry

end relationship_of_exponents_l126_126874


namespace unique_vector_a_l126_126293

-- Defining the vectors
def vector_a (x y : ℝ) : ℝ × ℝ := (x, y)
def vector_b (x y : ℝ) : ℝ × ℝ := (x^2, y^2)
def vector_c : ℝ × ℝ := (1, 1)
def vector_d : ℝ × ℝ := (2, 2)

-- Dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The Lean statement to prove
theorem unique_vector_a (x y : ℝ) 
  (h1 : dot_product (vector_a x y) vector_c = 1)
  (h2 : dot_product (vector_b x y) vector_d = 1) : 
  vector_a x y = vector_a (1/2) (1/2) :=
by {
  sorry 
}

end unique_vector_a_l126_126293


namespace entrance_exam_proof_l126_126669

-- Define the conditions
variables (x y : ℕ)
variables (h1 : x + y = 70)
variables (h2 : 3 * x - y = 38)

-- The proof goal
theorem entrance_exam_proof : x = 27 :=
by
  -- The actual proof steps are omitted here
  sorry

end entrance_exam_proof_l126_126669


namespace solve_system_l126_126453

theorem solve_system : ∃ s t : ℝ, (11 * s + 7 * t = 240) ∧ (s = 1 / 2 * t + 3) ∧ (t = 414 / 25) :=
by
  sorry

end solve_system_l126_126453


namespace number_increase_when_reversed_l126_126628

theorem number_increase_when_reversed :
  let n := 253
  let reversed_n := 352
  reversed_n - n = 99 :=
by
  let n := 253
  let reversed_n := 352
  sorry

end number_increase_when_reversed_l126_126628


namespace find_original_faculty_count_l126_126766

variable (F : ℝ)
variable (final_count : ℝ := 195)
variable (first_year_reduction : ℝ := 0.075)
variable (second_year_increase : ℝ := 0.125)
variable (third_year_reduction : ℝ := 0.0325)
variable (fourth_year_increase : ℝ := 0.098)
variable (fifth_year_reduction : ℝ := 0.1465)

theorem find_original_faculty_count (h : F * (1 - first_year_reduction)
                                        * (1 + second_year_increase)
                                        * (1 - third_year_reduction)
                                        * (1 + fourth_year_increase)
                                        * (1 - fifth_year_reduction) = final_count) :
  F = 244 :=
by sorry

end find_original_faculty_count_l126_126766


namespace lunch_special_cost_l126_126475

theorem lunch_special_cost (total_bill : ℕ) (num_people : ℕ) (cost_per_lunch_special : ℕ)
  (h1 : total_bill = 24) 
  (h2 : num_people = 3) 
  (h3 : cost_per_lunch_special = total_bill / num_people) : 
  cost_per_lunch_special = 8 := 
by
  sorry

end lunch_special_cost_l126_126475


namespace bullets_shot_per_person_l126_126201

-- Definitions based on conditions
def num_people : ℕ := 5
def initial_bullets_per_person : ℕ := 25
def total_remaining_bullets : ℕ := 25

-- Statement to prove
theorem bullets_shot_per_person (x : ℕ) :
  (initial_bullets_per_person * num_people - num_people * x) = total_remaining_bullets → x = 20 :=
by
  sorry

end bullets_shot_per_person_l126_126201


namespace factor_expression_zero_l126_126337

theorem factor_expression_zero (a b c : ℝ) (h : a + b + c ≠ 0) :
  (a^3 - b^3)^2 + (b^3 - c^3)^2 + (c^3 - a^3)^2 = 0 :=
sorry

end factor_expression_zero_l126_126337


namespace jim_reads_less_hours_l126_126506

-- Conditions
def initial_speed : ℕ := 40 -- pages per hour
def initial_pages_per_week : ℕ := 600 -- pages
def speed_increase_factor : ℚ := 1.5
def new_pages_per_week : ℕ := 660 -- pages

-- Calculations based on conditions
def initial_hours_per_week : ℚ := initial_pages_per_week / initial_speed
def new_speed : ℚ := initial_speed * speed_increase_factor
def new_hours_per_week : ℚ := new_pages_per_week / new_speed

-- Theorem Statement
theorem jim_reads_less_hours :
  initial_hours_per_week - new_hours_per_week = 4 :=
  sorry

end jim_reads_less_hours_l126_126506


namespace odd_function_max_to_min_l126_126991

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem odd_function_max_to_min (a b : ℝ) (f : ℝ → ℝ)
  (hodd : is_odd_function f)
  (hmax : ∃ x : ℝ, x > 0 ∧ (a * f x + b * x + 1) = 2) :
  ∃ y : ℝ, y < 0 ∧ (a * f y + b * y + 1) = 0 :=
sorry

end odd_function_max_to_min_l126_126991


namespace max_oranges_to_teachers_l126_126437

theorem max_oranges_to_teachers {n r : ℕ} (h1 : n % 8 = r) (h2 : r < 8) : r = 7 :=
sorry

end max_oranges_to_teachers_l126_126437


namespace find_5y_45_sevenths_l126_126501

theorem find_5y_45_sevenths (x y : ℝ) 
(h1 : 3 * x + 4 * y = 0) 
(h2 : x = y + 3) : 
5 * y = -45 / 7 :=
by
  sorry

end find_5y_45_sevenths_l126_126501


namespace rose_bushes_in_park_l126_126989

theorem rose_bushes_in_park (current_bushes : ℕ) (newly_planted : ℕ) (h1 : current_bushes = 2) (h2 : newly_planted = 4) : current_bushes + newly_planted = 6 :=
by
  sorry

end rose_bushes_in_park_l126_126989


namespace find_units_digit_of_n_l126_126512

-- Define the problem conditions
def units_digit (a : ℕ) : ℕ := a % 10

theorem find_units_digit_of_n (m n : ℕ) (h1 : units_digit m = 3) (h2 : units_digit (m * n) = 6) (h3 : units_digit (14^8) = 6) :
  units_digit n = 2 :=
  sorry

end find_units_digit_of_n_l126_126512


namespace car_speed_second_hour_l126_126917

theorem car_speed_second_hour
  (S : ℕ)
  (first_hour_speed : ℕ := 98)
  (avg_speed : ℕ := 79)
  (total_time : ℕ := 2)
  (h_avg_speed : avg_speed = (first_hour_speed + S) / total_time) :
  S = 60 :=
by
  -- Proof steps omitted
  sorry

end car_speed_second_hour_l126_126917


namespace evaluate_polynomial_at_3_l126_126639

def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 - 3*x + 2

theorem evaluate_polynomial_at_3 : f 3 = 2 :=
by
  sorry

end evaluate_polynomial_at_3_l126_126639


namespace opposite_sign_pairs_l126_126814

def opposite_sign (a b : ℤ) : Prop := (a < 0 ∧ b > 0) ∨ (a > 0 ∧ b < 0)

theorem opposite_sign_pairs :
  ¬opposite_sign (-(-1)) 1 ∧
  ¬opposite_sign ((-1)^2) 1 ∧
  ¬opposite_sign (|(-1)|) 1 ∧
  opposite_sign (-1) 1 :=
by {
  sorry
}

end opposite_sign_pairs_l126_126814


namespace angle_hyperbola_l126_126455

theorem angle_hyperbola (a b : ℝ) (e : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (hyperbola_eq : ∀ (x y : ℝ), ((x^2)/(a^2) - (y^2)/(b^2) = 1)) 
  (eccentricity_eq : e = 2 + Real.sqrt 6 - Real.sqrt 3 - Real.sqrt 2) :
  ∃ α : ℝ, α = 15 :=
by
  sorry

end angle_hyperbola_l126_126455


namespace inequality_sum_l126_126770

theorem inequality_sum 
  (a1 a2 a3 b1 b2 b3 : ℝ)
  (h1 : a1 ≥ a2)
  (h2 : a2 ≥ a3)
  (h3 : a3 > 0)
  (h4 : b1 ≥ b2)
  (h5 : b2 ≥ b3)
  (h6 : b3 > 0)
  (h7 : a1 * a2 * a3 = b1 * b2 * b3)
  (h8 : a1 - a3 ≤ b1 - b3) :
  a1 + a2 + a3 ≤ 2 * (b1 + b2 + b3) := 
sorry

end inequality_sum_l126_126770


namespace trapezoid_area_l126_126347

theorem trapezoid_area (A B : ℝ) (n : ℕ) (hA : A = 36) (hB : B = 4) (hn : n = 6) :
    (A - B) / n = 5.33 := 
by 
  -- Given conditions and the goal
  sorry

end trapezoid_area_l126_126347


namespace line_intersects_circle_l126_126645

theorem line_intersects_circle : 
  ∀ (x y : ℝ), 
  (2 * x + y = 0) ∧ (x^2 + y^2 + 2 * x - 4 * y - 4 = 0) ↔
    ∃ (x0 y0 : ℝ), (2 * x0 + y0 = 0) ∧ ((x0 + 1)^2 + (y0 - 2)^2 = 9) :=
by
  sorry

end line_intersects_circle_l126_126645


namespace calculate_expression_l126_126922

theorem calculate_expression : 2^3 * 2^3 + 2^3 = 72 := by
  sorry

end calculate_expression_l126_126922


namespace neg_prop_l126_126419

-- Definition of the proposition to be negated
def prop (x : ℝ) : Prop := x^2 + 2 * x + 5 = 0

-- Negation of the proposition
theorem neg_prop : ¬ (∃ x : ℝ, prop x) ↔ ∀ x : ℝ, ¬ prop x :=
by
  sorry

end neg_prop_l126_126419


namespace imo_1990_q31_l126_126540

def A (n : ℕ) : ℕ := sorry -- definition of A(n)
def B (n : ℕ) : ℕ := sorry -- definition of B(n)
def f (n : ℕ) : ℕ := if B n = 1 then 1 else -- largest prime factor of B(n)
  sorry -- logic to find the largest prime factor of B(n)

theorem imo_1990_q31 :
  ∃ (M : ℕ), (∀ n : ℕ, f n ≤ M) ∧ (∀ N, (∀ n, f n ≤ N) → M ≤ N) ∧ M = 1999 :=
by sorry

end imo_1990_q31_l126_126540


namespace koi_fish_in_pond_l126_126026

theorem koi_fish_in_pond:
  ∃ k : ℕ, 2 * k - 14 = 64 ∧ k = 39 := sorry

end koi_fish_in_pond_l126_126026


namespace largest_divisor_of_seven_consecutive_odd_numbers_l126_126133

theorem largest_divisor_of_seven_consecutive_odd_numbers (n : ℕ) (h : Even n) (h_pos : n > 0) :
  ∃ d, d = 45 ∧ ∀ k, k ∣ ((n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13)) → k ≤ 45 :=
sorry

end largest_divisor_of_seven_consecutive_odd_numbers_l126_126133


namespace problem_composite_for_n_geq_9_l126_126443

theorem problem_composite_for_n_geq_9 (n : ℤ) (h : n ≥ 9) : ∃ k m : ℤ, (2 ≤ k ∧ 2 ≤ m ∧ n + 7 = k * m) :=
by
  sorry

end problem_composite_for_n_geq_9_l126_126443


namespace scientific_notation_350_million_l126_126767

theorem scientific_notation_350_million : 350000000 = 3.5 * 10^8 := 
  sorry

end scientific_notation_350_million_l126_126767


namespace find_number_l126_126609

theorem find_number (x : ℝ) (h : x / 4 + 15 = 4 * x - 15) : x = 8 :=
sorry

end find_number_l126_126609


namespace xy_yx_eq_zy_yz_eq_xz_zx_l126_126476

theorem xy_yx_eq_zy_yz_eq_xz_zx 
  (x y z : ℝ) 
  (h : x * (y + z - x) / x = y * (z + x - y) / y ∧ y * (z + x - y) / y = z * (x + y - z) / z): 
  x ^ y * y ^ x = z ^ y * y ^ z ∧ z ^ y * y ^ z = x ^ z * z ^ x :=
by
  sorry

end xy_yx_eq_zy_yz_eq_xz_zx_l126_126476


namespace angle_GDA_is_135_l126_126926

-- Definitions for the geometric entities and conditions mentioned
structure Triangle :=
  (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ)

structure Square :=
  (angle : ℝ := 90)

def BCD : Triangle :=
  { angle_A := 45, angle_B := 45, angle_C := 90 }

def ABCD : Square :=
  {}

def DEFG : Square :=
  {}

-- The proof problem stated in Lean 4
theorem angle_GDA_is_135 :
  ∃ θ : ℝ, θ = 135 ∧ 
  (∀ (BCD : Triangle), BCD.angle_C = 90 ∧ BCD.angle_A = 45 ∧ BCD.angle_B = 45) ∧ 
  (∀ (Square : Square), Square.angle = 90) → 
  θ = 135 :=
by
  sorry

end angle_GDA_is_135_l126_126926


namespace daily_harvest_l126_126537

theorem daily_harvest (sacks_per_section : ℕ) (num_sections : ℕ) 
  (h1 : sacks_per_section = 45) (h2 : num_sections = 8) : 
  sacks_per_section * num_sections = 360 :=
by
  sorry

end daily_harvest_l126_126537


namespace anoop_joined_after_6_months_l126_126414

/- Conditions -/
def arjun_investment : ℕ := 20000
def arjun_months : ℕ := 12
def anoop_investment : ℕ := 40000

/- Main theorem -/
theorem anoop_joined_after_6_months (x : ℕ) (h : arjun_investment * arjun_months = anoop_investment * (arjun_months - x)) : 
  x = 6 :=
sorry

end anoop_joined_after_6_months_l126_126414


namespace totalFriendsAreFour_l126_126524

-- Define the friends
def friends := ["Mary", "Sam", "Keith", "Alyssa"]

-- Define the number of friends
def numberOfFriends (f : List String) : ℕ := f.length

-- Claim that the number of friends is 4
theorem totalFriendsAreFour : numberOfFriends friends = 4 :=
by
  -- Skip proof
  sorry

end totalFriendsAreFour_l126_126524


namespace ratio_of_areas_eq_nine_sixteenth_l126_126936

-- Definitions based on conditions
def side_length_C : ℝ := 45
def side_length_D : ℝ := 60
def area (s : ℝ) : ℝ := s * s

-- Theorem stating the desired proof problem
theorem ratio_of_areas_eq_nine_sixteenth :
  (area side_length_C) / (area side_length_D) = 9 / 16 :=
by
  sorry

end ratio_of_areas_eq_nine_sixteenth_l126_126936


namespace no_k_satisfying_condition_l126_126079

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_k_satisfying_condition :
  ∀ k : ℕ, (∃ p q : ℕ, p ≠ q ∧ is_prime p ∧ is_prime q ∧ k = p * q ∧ p + q = 71) → false :=
by
  sorry

end no_k_satisfying_condition_l126_126079


namespace gum_candy_ratio_l126_126336

theorem gum_candy_ratio
  (g c : ℝ)  -- let g be the cost of a stick of gum and c be the cost of a candy bar.
  (hc : c = 1.5)  -- the cost of each candy bar is $1.5
  (h_total_cost : 2 * g + 3 * c = 6)  -- total cost of 2 sticks of gum and 3 candy bars is $6
  : g / c = 1 / 2 := -- the ratio of the cost of gum to candy is 1:2
sorry

end gum_candy_ratio_l126_126336


namespace find_b_l126_126004

-- Define the conditions of the equations
def condition_1 (x y a : ℝ) : Prop := x * Real.cos a + y * Real.sin a + 3 ≤ 0
def condition_2 (x y b : ℝ) : Prop := x^2 + y^2 + 8 * x - 4 * y - b^2 + 6 * b + 11 = 0

-- Define the proof problem
theorem find_b (b : ℝ) :
  (∀ a x y, condition_1 x y a → condition_2 x y b) →
  b ∈ Set.Iic (-2 * Real.sqrt 5) ∪ Set.Ici (6 + 2 * Real.sqrt 5) :=
by
  sorry

end find_b_l126_126004


namespace number_of_other_values_l126_126731

def orig_value : ℕ := 2 ^ (2 ^ (2 ^ 2))

def other_values : Finset ℕ :=
  {2 ^ (2 ^ (2 ^ 2)), 2 ^ ((2 ^ 2) ^ 2), ((2 ^ 2) ^ 2) ^ 2, (2 ^ (2 ^ 2)) ^ 2, (2 ^ 2) ^ (2 ^ 2)}

theorem number_of_other_values :
  other_values.erase orig_value = {256} :=
by
  sorry

end number_of_other_values_l126_126731


namespace induction_step_l126_126995

theorem induction_step
  (x y : ℝ)
  (k : ℕ)
  (base : ∀ n, ∃ m, (n = 2 * m - 1) → (x^n + y^n) = (x + y) * m) :
  (x^(2 * k + 1) + y^(2 * k + 1)) = (x + y) * (k + 1) :=
by
  sorry

end induction_step_l126_126995


namespace combination_problem_l126_126388

noncomputable def combination (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.choose n k

theorem combination_problem (x : ℕ) (h : combination 25 (2 * x) = combination 25 (x + 4)) : x = 4 ∨ x = 7 :=
by {
  sorry
}

end combination_problem_l126_126388


namespace translated_line_value_m_l126_126849

theorem translated_line_value_m :
  (∀ x y : ℝ, (y = x → y = x + 3) → y = 2 + 3 → ∃ m : ℝ, y = m) :=
by sorry

end translated_line_value_m_l126_126849


namespace Austin_work_hours_on_Wednesdays_l126_126679

variable {W : ℕ}

theorem Austin_work_hours_on_Wednesdays
  (h1 : 5 * 2 + 5 * W + 5 * 3 = 25 + 5 * W)
  (h2 : 6 * (25 + 5 * W) = 180)
  : W = 1 := by
  sorry

end Austin_work_hours_on_Wednesdays_l126_126679


namespace olympiad_scores_l126_126743

theorem olympiad_scores (scores : Fin 20 → ℕ) 
  (uniqueScores : ∀ i j, i ≠ j → scores i ≠ scores j)
  (less_than_sum_of_others : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i, scores i > 18 := 
by sorry

end olympiad_scores_l126_126743


namespace monomial_properties_l126_126426

theorem monomial_properties (a b : ℕ) (h : a = 2 ∧ b = 1) : 
  (2 * a ^ 2 * b = 2 * (a ^ 2) * b) ∧ (2 = 2) ∧ ((2 + 1) = 3) :=
by
  sorry

end monomial_properties_l126_126426


namespace volume_of_water_overflow_l126_126956

-- Definitions based on given conditions
def mass_of_ice : ℝ := 50
def density_of_fresh_ice : ℝ := 0.9
def density_of_salt_ice : ℝ := 0.95
def density_of_fresh_water : ℝ := 1
def density_of_salt_water : ℝ := 1.03

-- Theorem statement corresponding to the problem
theorem volume_of_water_overflow
  (m : ℝ := mass_of_ice) 
  (rho_n : ℝ := density_of_fresh_ice) 
  (rho_c : ℝ := density_of_salt_ice) 
  (rho_fw : ℝ := density_of_fresh_water) 
  (rho_sw : ℝ := density_of_salt_water) :
  ∃ (ΔV : ℝ), ΔV = 2.63 :=
by
  sorry

end volume_of_water_overflow_l126_126956


namespace solve_equation_l126_126338

theorem solve_equation (x : ℝ) (floor : ℝ → ℤ) 
  (h_floor : ∀ y, floor y ≤ y ∧ y < floor y + 1) :
  (floor (20 * x + 23) = 20 + 23 * x) ↔ 
  (∃ n : ℤ, 20 ≤ n ∧ n ≤ 43 ∧ x = (n - 23) / 20) := 
by
  sorry

end solve_equation_l126_126338


namespace joan_mortgage_payment_l126_126774

noncomputable def geometric_series_sum (a r : ℕ) (n : ℕ) : ℕ :=
  a * (1 - r^n) / (1 - r)

theorem joan_mortgage_payment : 
  ∃ n : ℕ, geometric_series_sum 100 3 n = 109300 ∧ n = 7 :=
by
  sorry

end joan_mortgage_payment_l126_126774


namespace min_value_expr_l126_126211

theorem min_value_expr (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + 1 / (2 * y))^2 + (y + 1 / (2 * x))^2 ≥ 4 :=
sorry

end min_value_expr_l126_126211


namespace valid_integer_values_of_x_l126_126035

theorem valid_integer_values_of_x (x : ℤ) 
  (h1 : 3 < x) (h2 : x < 10)
  (h3 : 5 < x) (h4 : x < 18)
  (h5 : -2 < x) (h6 : x < 9)
  (h7 : 0 < x) (h8 : x < 8) 
  (h9 : x + 1 < 9) : x = 6 ∨ x = 7 :=
by
  sorry

end valid_integer_values_of_x_l126_126035


namespace max_value_part1_l126_126689

theorem max_value_part1 (a : ℝ) (h : a < 3 / 2) : 2 * a + 4 / (2 * a - 3) + 3 ≤ 2 :=
sorry

end max_value_part1_l126_126689


namespace sequence_general_formula_l126_126748

theorem sequence_general_formula (a : ℕ → ℕ) 
  (h₁ : a 1 = 2)
  (h₂ : ∀ n, a (n + 1) = 2 * a n - 1) :
  ∀ n, a n = 1 + 2^(n - 1) := 
sorry

end sequence_general_formula_l126_126748


namespace eval_expression_l126_126918

theorem eval_expression : 5 - 7 * (8 - 12 / 3^2) * 6 = -275 := by
  sorry

end eval_expression_l126_126918


namespace katherine_has_5_bananas_l126_126262

/-- Katherine has 4 apples -/
def apples : ℕ := 4

/-- Katherine has 3 times as many pears as apples -/
def pears : ℕ := 3 * apples

/-- Katherine has a total of 21 pieces of fruit (apples + pears + bananas) -/
def total_fruit : ℕ := 21

/-- Define the number of bananas Katherine has -/
def bananas : ℕ := total_fruit - (apples + pears)

/-- Prove that Katherine has 5 bananas -/
theorem katherine_has_5_bananas : bananas = 5 := by
  sorry

end katherine_has_5_bananas_l126_126262


namespace same_function_absolute_value_l126_126383

theorem same_function_absolute_value :
  (∀ (x : ℝ), |x| = if x > 0 then x else -x) :=
by
  intro x
  split_ifs with h
  · exact abs_of_pos h
  · exact abs_of_nonpos (le_of_not_gt h)

end same_function_absolute_value_l126_126383


namespace cheesecakes_sold_l126_126765

theorem cheesecakes_sold
  (initial_display : Nat)
  (initial_fridge : Nat)
  (left_to_sell : Nat)
  (total_cheesecakes := initial_display + initial_fridge)
  (total_after_sales : Nat) :
  initial_display = 10 →
  initial_fridge = 15 →
  left_to_sell = 18 →
  total_after_sales = total_cheesecakes - left_to_sell →
  total_after_sales = 7 := sorry

end cheesecakes_sold_l126_126765


namespace sum_of_diagonals_l126_126971

-- Definitions of the given lengths
def AB := 5
def CD := 5
def BC := 12
def DE := 12
def AE := 18

-- Variables for the diagonal lengths
variables (AC BD CE : ℚ)

-- The Lean 4 theorem statement
theorem sum_of_diagonals (hAC : AC = 723 / 44) (hBD : BD = 44 / 3) (hCE : CE = 351 / 22) :
  AC + BD + CE = 6211 / 132 :=
by
  sorry

end sum_of_diagonals_l126_126971


namespace gcd_of_gy_and_y_l126_126598

theorem gcd_of_gy_and_y (y : ℕ) (h : ∃ k : ℕ, y = k * 3456) :
  gcd ((5 * y + 4) * (9 * y + 1) * (12 * y + 6) * (3 * y + 9)) y = 216 :=
by {
  sorry
}

end gcd_of_gy_and_y_l126_126598


namespace fill_bathtub_with_drain_open_l126_126892

theorem fill_bathtub_with_drain_open :
  let fill_rate := 1 / 10
  let drain_rate := 1 / 12
  let net_fill_rate := fill_rate - drain_rate
  fill_rate = 1 / 10 ∧ drain_rate = 1 / 12 → 1 / net_fill_rate = 60 :=
by
  intros
  sorry

end fill_bathtub_with_drain_open_l126_126892


namespace exists_natural_numbers_gt_10pow10_divisible_by_self_plus_2012_l126_126938

theorem exists_natural_numbers_gt_10pow10_divisible_by_self_plus_2012 :
  ∃ (a b c : ℕ), 
    a > 10^10 ∧ b > 10^10 ∧ c > 10^10 ∧ 
    a ∣ (a * b * c + 2012) ∧ b ∣ (a * b * c + 2012) ∧ c ∣ (a * b * c + 2012) :=
by
  sorry

end exists_natural_numbers_gt_10pow10_divisible_by_self_plus_2012_l126_126938


namespace evaluate_expression_l126_126465

theorem evaluate_expression (x y z : ℤ) (h1 : x = -2) (h2 : y = -4) (h3 : z = 3) :
  (5 * (x - y)^2 - x * z^2) / (z - y) = 38 / 7 := by
  sorry

end evaluate_expression_l126_126465


namespace candy_distribution_l126_126555

theorem candy_distribution (A B C : ℕ) (x y : ℕ)
  (h1 : A > 2 * B)
  (h2 : B > 3 * C)
  (h3 : A + B + C = 200) :
  (A = 121) ∧ (C = 19) :=
  sorry

end candy_distribution_l126_126555


namespace range_of_a_l126_126071

theorem range_of_a (a : ℝ) : ((1 - a) ^ 2 + (1 + a) ^ 2 < 4) → (-1 < a ∧ a < 1) :=
by
  sorry

end range_of_a_l126_126071


namespace w_z_ratio_l126_126634

theorem w_z_ratio (w z : ℝ) (h : (1/w + 1/z) / (1/w - 1/z) = 2023) : (w + z) / (w - z) = -2023 :=
by sorry

end w_z_ratio_l126_126634


namespace initial_customers_l126_126089

theorem initial_customers (x : ℝ) : (x - 8 + 4 = 9) → x = 13 :=
by
  sorry

end initial_customers_l126_126089


namespace original_price_l126_126585

theorem original_price (P : ℝ) (h₁ : P - 0.30 * P = 0.70 * P) (h₂ : P - 0.20 * P = 0.80 * P) (h₃ : 0.70 * P + 0.80 * P = 50) :
  P = 100 / 3 :=
by
  -- Proof skipped
  sorry

end original_price_l126_126585


namespace fraction_of_females_l126_126781

def local_soccer_league_female_fraction : Prop :=
  ∃ (males_last_year females_last_year : ℕ),
    males_last_year = 30 ∧
    (1.10 * males_last_year : ℝ) = 33 ∧
    (males_last_year + females_last_year : ℝ) * 1.15 = 52 ∧
    (females_last_year : ℝ) * 1.25 = 19 ∧
    (33 + 19 = 52)

theorem fraction_of_females
  : local_soccer_league_female_fraction → 
    ∃ (females fraction : ℝ),
    females = 19 ∧ 
    fraction = 19 / 52 :=
by
  sorry

end fraction_of_females_l126_126781


namespace fraction_value_l126_126192

theorem fraction_value (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : x + 2 * y = 8) : (x + y) / 3 = 5 / 3 :=
by sorry

end fraction_value_l126_126192


namespace friendP_walks_23_km_l126_126600

noncomputable def friendP_distance (v : ℝ) : ℝ :=
  let trail_length := 43
  let speedP := 1.15 * v
  let speedQ := v
  let dQ := trail_length - 23
  let timeP := 23 / speedP
  let timeQ := dQ / speedQ
  if timeP = timeQ then 23 else 0  -- Ensuring that both reach at the same time.

theorem friendP_walks_23_km (v : ℝ) : 
  friendP_distance v = 23 :=
by
  sorry

end friendP_walks_23_km_l126_126600


namespace kilos_of_bananas_l126_126663

-- Define the conditions
def initial_money := 500
def remaining_money := 426
def cost_per_kilo_potato := 2
def cost_per_kilo_tomato := 3
def cost_per_kilo_cucumber := 4
def cost_per_kilo_banana := 5
def kilos_potato := 6
def kilos_tomato := 9
def kilos_cucumber := 5

-- Total cost of potatoes, tomatoes, and cucumbers
def total_cost_vegetables : ℕ := 
  (kilos_potato * cost_per_kilo_potato) +
  (kilos_tomato * cost_per_kilo_tomato) +
  (kilos_cucumber * cost_per_kilo_cucumber)

-- Money spent on bananas
def money_spent_on_bananas : ℕ := initial_money - remaining_money - total_cost_vegetables

-- The proof problem statement
theorem kilos_of_bananas : money_spent_on_bananas / cost_per_kilo_banana = 14 :=
by
  -- The sorry is a placeholder for the proof
  sorry

end kilos_of_bananas_l126_126663


namespace sector_area_l126_126398

theorem sector_area (r l : ℝ) (h1 : l + 2 * r = 8) (h2 : l = 2 * r) : 
  (1 / 2) * l * r = 4 := 
by sorry

end sector_area_l126_126398


namespace convert_deg_to_rad1_convert_deg_to_rad2_convert_deg_to_rad3_convert_rad_to_deg1_convert_rad_to_deg2_convert_rad_to_deg3_l126_126170

theorem convert_deg_to_rad1 : 780 * (Real.pi / 180) = (13 * Real.pi) / 3 := sorry
theorem convert_deg_to_rad2 : -1560 * (Real.pi / 180) = -(26 * Real.pi) / 3 := sorry
theorem convert_deg_to_rad3 : 67.5 * (Real.pi / 180) = (3 * Real.pi) / 8 := sorry
theorem convert_rad_to_deg1 : -(10 * Real.pi / 3) * (180 / Real.pi) = -600 := sorry
theorem convert_rad_to_deg2 : (Real.pi / 12) * (180 / Real.pi) = 15 := sorry
theorem convert_rad_to_deg3 : (7 * Real.pi / 4) * (180 / Real.pi) = 315 := sorry

end convert_deg_to_rad1_convert_deg_to_rad2_convert_deg_to_rad3_convert_rad_to_deg1_convert_rad_to_deg2_convert_rad_to_deg3_l126_126170


namespace teaching_arrangements_l126_126693

-- Define the conditions
structure Conditions :=
  (teach_A : ℕ)
  (teach_B : ℕ)
  (teach_C : ℕ)
  (teach_D : ℕ)
  (max_teach_AB : ∀ t, t = teach_A ∨ t = teach_B → t ≤ 2)
  (max_teach_CD : ∀ t, t = teach_C ∨ t = teach_D → t ≤ 1)
  (total_periods : ℕ)
  (teachers_per_period : ℕ)

-- Constants and assumptions
def problem_conditions : Conditions := {
  teach_A := 2,
  teach_B := 2,
  teach_C := 1,
  teach_D := 1,
  max_teach_AB := by sorry,
  max_teach_CD := by sorry,
  total_periods := 2,
  teachers_per_period := 2
}

-- Define the proof goal
theorem teaching_arrangements (c : Conditions) :
  c = problem_conditions → ∃ arrangements, arrangements = 19 :=
by
  sorry

end teaching_arrangements_l126_126693


namespace evaluate_product_roots_of_unity_l126_126580

theorem evaluate_product_roots_of_unity :
  let w := Complex.exp (2 * Real.pi * Complex.I / 13)
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) *
  (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) =
  (3^12 + 3^11 + 3^10 + 3^9 + 3^8 + 3^7 + 3^6 + 3^5 + 3^4 + 3^3 + 3^2 + 3 + 1) :=
by
  sorry

end evaluate_product_roots_of_unity_l126_126580


namespace ratio_of_segments_l126_126526

theorem ratio_of_segments (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
sorry

end ratio_of_segments_l126_126526


namespace total_time_to_fill_tank_l126_126602

-- Definitions as per conditions
def tank_fill_time_for_one_tap (total_time : ℕ) : Prop :=
  total_time = 16

def number_of_taps_for_second_half (num_taps : ℕ) : Prop :=
  num_taps = 4

-- Theorem statement to prove the total time taken to fill the tank
theorem total_time_to_fill_tank : ∀ (time_one_tap time_total : ℕ),
  tank_fill_time_for_one_tap time_one_tap →
  number_of_taps_for_second_half 4 →
  time_total = 10 :=
by
  intros time_one_tap time_total h1 h2
  -- Proof needed here
  sorry

end total_time_to_fill_tank_l126_126602


namespace negation_of_P_l126_126406

def P : Prop := ∃ x_0 : ℝ, x_0^2 + 2 * x_0 + 2 ≤ 0

theorem negation_of_P : ¬ P ↔ ∀ x : ℝ, x^2 + 2 * x + 2 > 0 :=
by sorry

end negation_of_P_l126_126406


namespace theta_value_l126_126658

theorem theta_value (theta : ℝ) (h1 : 0 ≤ theta ∧ theta ≤ 90)
    (h2 : Real.cos 60 = Real.cos 45 * Real.cos theta) : theta = 45 :=
  sorry

end theta_value_l126_126658


namespace find_number_eq_l126_126673

theorem find_number_eq : ∃ x : ℚ, (35 / 100) * x = (25 / 100) * 40 ∧ x = 200 / 7 :=
by
  sorry

end find_number_eq_l126_126673


namespace smallest_largest_number_in_list_l126_126865

theorem smallest_largest_number_in_list :
  ∃ (a b c d e : ℕ), (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧ 
  (a + b + c + d + e = 50) ∧ (e - a = 20) ∧ 
  (c = 6) ∧ (b = 6) ∧ 
  (e = 20) :=
by
  sorry

end smallest_largest_number_in_list_l126_126865


namespace fred_found_28_more_seashells_l126_126988

theorem fred_found_28_more_seashells (tom_seashells : ℕ) (fred_seashells : ℕ) (h_tom : tom_seashells = 15) (h_fred : fred_seashells = 43) : 
  fred_seashells - tom_seashells = 28 := 
by 
  sorry

end fred_found_28_more_seashells_l126_126988


namespace option_D_correct_l126_126233

theorem option_D_correct (a b : ℝ) (h : a > b) : 3 * a > 3 * b :=
sorry

end option_D_correct_l126_126233


namespace monotonicity_condition_l126_126757

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x

theorem monotonicity_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo 1 2, f a x ≥ f a 1) ↔ a ∈ Set.Ici 2 :=
by
  sorry

end monotonicity_condition_l126_126757


namespace equivalent_problem_l126_126550

variable (x y : ℝ)
variable (hx_ne_zero : x ≠ 0)
variable (hy_ne_zero : y ≠ 0)
variable (h : (3 * x + y) / (x - 3 * y) = -2)

theorem equivalent_problem : (x + 3 * y) / (3 * x - y) = 2 :=
by
  sorry

end equivalent_problem_l126_126550


namespace restaurant_meal_cost_l126_126959

/--
Each adult meal costs $8 and kids eat free. 
If there is a group of 11 people, out of which 2 are kids, 
prove that the total cost for the group to eat is $72.
-/
theorem restaurant_meal_cost (cost_per_adult : ℕ) (group_size : ℕ) (kids : ℕ) 
  (all_free_kids : ℕ → Prop) (total_cost : ℕ)  
  (h1 : cost_per_adult = 8) 
  (h2 : group_size = 11) 
  (h3 : kids = 2) 
  (h4 : all_free_kids kids) 
  (h5 : total_cost = (group_size - kids) * cost_per_adult) : 
  total_cost = 72 := 
by 
  sorry

end restaurant_meal_cost_l126_126959


namespace count_five_digit_progressive_numbers_find_110th_five_digit_progressive_number_l126_126525

def is_progressive_number (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 : ℕ), 1 ≤ d1 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ d4 < d5 ∧ d5 ≤ 9 ∧
                          n = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5

theorem count_five_digit_progressive_numbers : ∃ n, n = 126 :=
by
  sorry

theorem find_110th_five_digit_progressive_number : ∃ n, n = 34579 :=
by
  sorry

end count_five_digit_progressive_numbers_find_110th_five_digit_progressive_number_l126_126525


namespace max_height_of_projectile_l126_126327

def projectile_height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

theorem max_height_of_projectile : 
  ∃ t : ℝ, projectile_height t = 161 :=
sorry

end max_height_of_projectile_l126_126327


namespace abs_neg_2023_l126_126268

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l126_126268


namespace carrots_chloe_l126_126474

theorem carrots_chloe (c_i c_t c_p : ℕ) (H1 : c_i = 48) (H2 : c_t = 45) (H3 : c_p = 42) : 
  c_i - c_t + c_p = 45 := by
  sorry

end carrots_chloe_l126_126474


namespace marble_244_is_white_l126_126803

noncomputable def color_of_marble (n : ℕ) : String :=
  let cycle := ["white", "white", "white", "white", "gray", "gray", "gray", "gray", "gray", "black", "black", "black"]
  cycle.get! (n % 12)

theorem marble_244_is_white : color_of_marble 244 = "white" :=
by
  sorry

end marble_244_is_white_l126_126803


namespace least_number_subtraction_l126_126566

theorem least_number_subtraction (n : ℕ) (h₀ : n = 3830) (k : ℕ) (h₁ : k = 5) : (n - k) % 15 = 0 :=
by {
  sorry
}

end least_number_subtraction_l126_126566


namespace discount_percentage_l126_126213

theorem discount_percentage 
    (original_price : ℝ) 
    (total_paid : ℝ) 
    (sales_tax_rate : ℝ) 
    (sale_price_before_tax : ℝ) 
    (discount_amount : ℝ) 
    (discount_percentage : ℝ) :
    original_price = 200 → total_paid = 165 → sales_tax_rate = 0.10 →
    total_paid = sale_price_before_tax * (1 + sales_tax_rate) →
    sale_price_before_tax = original_price - discount_amount →
    discount_percentage = (discount_amount / original_price) * 100 →
    discount_percentage = 25 :=
by
  intros h_original h_total h_tax h_eq1 h_eq2 h_eq3
  sorry

end discount_percentage_l126_126213


namespace rich_walked_distance_l126_126045

def total_distance_walked (d1 d2 : ℕ) := 
  d1 + d2 + 2 * (d1 + d2) + (d1 + d2 + 2 * (d1 + d2)) / 2

def distance_to_intersection (d1 d2 : ℕ) := 
  2 * (d1 + d2)

def distance_to_end_route (d1 d2 : ℕ) := 
  (d1 + d2 + distance_to_intersection d1 d2) / 2

def total_distance_one_way (d1 d2 : ℕ) := 
  (d1 + d2) + (distance_to_intersection d1 d2) + (distance_to_end_route d1 d2)

theorem rich_walked_distance
  (d1 : ℕ := 20)
  (d2 : ℕ := 200) :
  2 * total_distance_one_way d1 d2 = 1980 :=
by
  simp [total_distance_one_way, distance_to_intersection, distance_to_end_route, total_distance_walked]
  sorry

end rich_walked_distance_l126_126045


namespace distinct_arrangements_of_pebbles_in_octagon_l126_126320

noncomputable def number_of_distinct_arrangements : ℕ :=
  (Nat.factorial 8) / 16

theorem distinct_arrangements_of_pebbles_in_octagon : 
  number_of_distinct_arrangements = 2520 :=
by
  sorry

end distinct_arrangements_of_pebbles_in_octagon_l126_126320


namespace minimum_value_abs_sum_l126_126226

theorem minimum_value_abs_sum (α β γ : ℝ) (h1 : α + β + γ = 2) (h2 : α * β * γ = 4) : 
  |α| + |β| + |γ| ≥ 6 :=
by
  sorry

end minimum_value_abs_sum_l126_126226


namespace smallest_k_l126_126620

theorem smallest_k (m n k : ℤ) (h : 221 * m + 247 * n + 323 * k = 2001) (hk : k > 100) : 
∃ k', k' = 111 ∧ k' > 100 :=
by
  sorry

end smallest_k_l126_126620


namespace vertical_asymptote_at_9_over_4_l126_126883

def vertical_asymptote (y : ℝ → ℝ) (x : ℝ) : Prop :=
  (∀ ε > 0, ∃ δ > 0, ∀ x', x' ≠ x → abs (x' - x) < δ → abs (y x') > ε)

noncomputable def function_y (x : ℝ) : ℝ :=
  (2 * x + 3) / (4 * x - 9)

theorem vertical_asymptote_at_9_over_4 :
  vertical_asymptote function_y (9 / 4) :=
sorry

end vertical_asymptote_at_9_over_4_l126_126883


namespace total_water_output_l126_126187

theorem total_water_output (flow_rate: ℚ) (time_duration: ℕ) (total_water: ℚ) :
  flow_rate = 2 + 2 / 3 → time_duration = 9 → total_water = 24 →
  flow_rate * time_duration = total_water :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_water_output_l126_126187


namespace train_speed_l126_126844

theorem train_speed (v t : ℝ) (h1 : 16 * t + v * t = 444) (h2 : v * t = 16 * t + 60) : v = 21 := 
sorry

end train_speed_l126_126844


namespace inequality_solution_set_l126_126493

theorem inequality_solution_set (a b c : ℝ)
  (h1 : a < 0)
  (h2 : b = -a)
  (h3 : c = -2 * a) :
  ∀ x : ℝ, (c * x^2 + b * x + a > 0) ↔ (x < -1 ∨ x > 1 / 2) :=
by
  sorry

end inequality_solution_set_l126_126493


namespace domain_g_l126_126577

noncomputable def g (x : ℝ) : ℝ := (x - 3) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_g : {x : ℝ | g x = (x - 3) / Real.sqrt (x^2 - 5 * x + 6)} = 
  {x : ℝ | x < 2 ∨ x > 3} :=
by
  sorry

end domain_g_l126_126577


namespace root_expression_eq_l126_126065

theorem root_expression_eq (p q α β γ δ : ℝ) 
  (h1 : ∀ x, (x - α) * (x - β) = x^2 + p * x + 2)
  (h2 : ∀ x, (x - γ) * (x - δ) = x^2 + q * x + 2) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = 4 + 2 * (p^2 - q^2) := 
sorry

end root_expression_eq_l126_126065


namespace acute_angle_proof_l126_126805

theorem acute_angle_proof
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (h : Real.cos (α + β) = Real.sin (α - β)) : α = π / 4 :=
  sorry

end acute_angle_proof_l126_126805


namespace cos_four_alpha_sub_9pi_over_2_l126_126702

open Real

theorem cos_four_alpha_sub_9pi_over_2 (α : ℝ) 
  (cond : 4.53 * (1 + cos (2 * α - 2 * π) + cos (4 * α + 2 * π) - cos (6 * α - π)) /
                  (cos (2 * π - 2 * α) + 2 * cos (2 * α + π) ^ 2 - 1) = 2 * cos (2 * α)) :
  cos (4 * α - 9 * π / 2) = cos (4 * α - π / 2) :=
by sorry

end cos_four_alpha_sub_9pi_over_2_l126_126702


namespace graveling_cost_is_correct_l126_126893

noncomputable def cost_of_graveling (lawn_length : ℕ) (lawn_breadth : ℕ) 
(road_width : ℕ) (cost_per_sq_m : ℕ) : ℕ :=
  let area_road_parallel_to_length := road_width * lawn_breadth
  let area_road_parallel_to_breadth := road_width * lawn_length
  let area_overlap := road_width * road_width
  let total_area := area_road_parallel_to_length + area_road_parallel_to_breadth - area_overlap
  total_area * cost_per_sq_m

theorem graveling_cost_is_correct : cost_of_graveling 90 60 10 3 = 4200 := by
  sorry

end graveling_cost_is_correct_l126_126893


namespace shipping_cost_correct_l126_126194

noncomputable def shipping_cost (W : ℝ) : ℕ := 7 + 5 * (⌈W⌉₊ - 1)

theorem shipping_cost_correct (W : ℝ) : shipping_cost W = 5 * ⌈W⌉₊ + 2 :=
by
  sorry

end shipping_cost_correct_l126_126194


namespace total_weight_CaBr2_l126_126376

-- Definitions derived from conditions
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_Br : ℝ := 79.904
def mol_weight_CaBr2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_Br
def moles_CaBr2 : ℝ := 4

-- Theorem statement based on the problem and correct answer
theorem total_weight_CaBr2 : moles_CaBr2 * mol_weight_CaBr2 = 799.552 :=
by
  -- Prove the theorem step-by-step
  -- substitute the definition of mol_weight_CaBr2
  -- show lhs = rhs
  sorry

end total_weight_CaBr2_l126_126376


namespace abs_neg_six_l126_126253

theorem abs_neg_six : abs (-6) = 6 :=
sorry

end abs_neg_six_l126_126253


namespace halfway_between_fractions_l126_126692

-- Definitions used in the conditions
def one_eighth := (1 : ℚ) / 8
def three_tenths := (3 : ℚ) / 10

-- The mathematical assertion to prove
theorem halfway_between_fractions : (one_eighth + three_tenths) / 2 = 17 / 80 := by
  sorry

end halfway_between_fractions_l126_126692


namespace greatest_prime_factor_294_l126_126943

theorem greatest_prime_factor_294 : ∃ p, Nat.Prime p ∧ p ∣ 294 ∧ ∀ q, Nat.Prime q ∧ q ∣ 294 → q ≤ p := 
by
  let prime_factors := [2, 3, 7]
  have h1 : 294 = 2 * 3 * 7 * 7 := by
    -- Proof of factorization should be inserted here
    sorry

  have h2 : ∀ p, p ∣ 294 → p = 2 ∨ p = 3 ∨ p = 7 := by
    -- Proof of prime factor correctness should be inserted here
    sorry

  use 7
  -- Prove 7 is the greatest prime factor here
  sorry

end greatest_prime_factor_294_l126_126943


namespace probability_of_last_two_marbles_one_green_one_red_l126_126074

theorem probability_of_last_two_marbles_one_green_one_red : 
    let total_marbles := 10
    let blue := 4
    let white := 3
    let red := 2
    let green := 1
    let total_ways := Nat.choose total_marbles 8
    let favorable_ways := Nat.choose (total_marbles - red - green) 6
    total_ways = 45 ∧ favorable_ways = 28 →
    (favorable_ways : ℚ) / total_ways = 28 / 45 :=
by
    intros total_marbles blue white red green total_ways favorable_ways h
    sorry

end probability_of_last_two_marbles_one_green_one_red_l126_126074


namespace area_PQR_l126_126283

-- Define the coordinates of the points
def P : ℝ × ℝ := (-3, 4)
def Q : ℝ × ℝ := (4, 9)
def R : ℝ × ℝ := (5, -3)

-- Function to calculate the area of a triangle given three points
def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * (abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)))

-- Statement to prove the area of triangle PQR is 44.5
theorem area_PQR : area_of_triangle P Q R = 44.5 := sorry

end area_PQR_l126_126283


namespace final_retail_price_l126_126460

theorem final_retail_price (wholesale_price markup_percentage discount_percentage desired_profit_percentage : ℝ)
  (h_wholesale : wholesale_price = 90)
  (h_markup : markup_percentage = 1)
  (h_discount : discount_percentage = 0.2)
  (h_desired_profit : desired_profit_percentage = 0.6) :
  let initial_retail_price := wholesale_price + (wholesale_price * markup_percentage)
  let discount_amount := initial_retail_price * discount_percentage
  let final_retail_price := initial_retail_price - discount_amount
  final_retail_price = 144 ∧ final_retail_price = wholesale_price + (wholesale_price * desired_profit_percentage) := by
 sorry

end final_retail_price_l126_126460


namespace add_to_fraction_eq_l126_126299

theorem add_to_fraction_eq (n : ℤ) (h : (4 + n) / (7 + n) = 3 / 4) : n = 5 :=
by sorry

end add_to_fraction_eq_l126_126299


namespace brianna_initial_marbles_l126_126135

-- Defining the variables and constants
def initial_marbles : Nat := 24
def marbles_lost : Nat := 4
def marbles_given : Nat := 2 * marbles_lost
def marbles_ate : Nat := marbles_lost / 2
def marbles_remaining : Nat := 10

-- The main statement to prove
theorem brianna_initial_marbles :
  marbles_remaining + marbles_ate + marbles_given + marbles_lost = initial_marbles :=
by
  sorry

end brianna_initial_marbles_l126_126135


namespace april_plant_arrangement_l126_126309

theorem april_plant_arrangement :
    let nBasil := 5
    let nTomato := 4
    let nPairs := nTomato / 2
    let nUnits := nBasil + nPairs
    let totalWays := (Nat.factorial nUnits) * (Nat.factorial nPairs) * (Nat.factorial (nPairs - 1))
    totalWays = 20160 := by
{
  let nBasil := 5
  let nTomato := 4
  let nPairs := nTomato / 2
  let nUnits := nBasil + nPairs
  let totalWays := (Nat.factorial nUnits) * (Nat.factorial nPairs) * (Nat.factorial (nPairs - 1))
  sorry
}

end april_plant_arrangement_l126_126309


namespace Rikki_earnings_l126_126173

theorem Rikki_earnings
  (price_per_word : ℝ := 0.01)
  (words_per_5_minutes : ℕ := 25)
  (total_minutes : ℕ := 120)
  (earning : ℝ := 6)
  : price_per_word * (words_per_5_minutes * (total_minutes / 5)) = earning := by
  sorry

end Rikki_earnings_l126_126173


namespace solution_set_transformation_l126_126741

noncomputable def solution_set_of_first_inequality (a b : ℝ) : Set ℝ :=
  {x | a * x^2 - 5 * x + b > 0}

noncomputable def solution_set_of_second_inequality (a b : ℝ) : Set ℝ :=
  {x | b * x^2 - 5 * x + a > 0}

theorem solution_set_transformation (a b : ℝ)
  (h : solution_set_of_first_inequality a b = {x | -3 < x ∧ x < 2}) :
  solution_set_of_second_inequality a b = {x | x < -3 ∨ x > 2} :=
by
  sorry

end solution_set_transformation_l126_126741


namespace train_travel_distance_l126_126559

theorem train_travel_distance (speed time: ℕ) (h1: speed = 85) (h2: time = 4) : speed * time = 340 :=
by
-- Given: speed = 85 km/hr and time = 4 hr
-- To prove: speed * time = 340
-- Since speed = 85 and time = 4, then 85 * 4 = 340
sorry

end train_travel_distance_l126_126559


namespace trench_dig_time_l126_126378

theorem trench_dig_time (a b c d : ℝ) (h1 : a + b + c + d = 1/6)
  (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
  (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4) :
  a + b + c = 1 / 6 := sorry

end trench_dig_time_l126_126378


namespace maximize_value_l126_126937

noncomputable def maximum_value (x y : ℝ) : ℝ :=
  3 * x - 2 * y

theorem maximize_value (x y : ℝ) (h : x^2 + y^2 + x * y = 1) : maximum_value x y ≤ 5 :=
sorry

end maximize_value_l126_126937


namespace abs_inequality_solution_set_l126_126896

theorem abs_inequality_solution_set (x : ℝ) :
  |x| + |x - 1| < 2 ↔ - (1 / 2) < x ∧ x < (3 / 2) :=
by
  sorry

end abs_inequality_solution_set_l126_126896


namespace square_side_length_l126_126891

theorem square_side_length (P : ℝ) (s : ℝ) (h1 : P = 36) (h2 : P = 4 * s) : s = 9 := 
by sorry

end square_side_length_l126_126891


namespace color_divisors_with_conditions_l126_126478

/-- Define the primes, product of the first 100 primes, and set S -/
def first_100_primes : List Nat := sorry -- Assume we have the list of first 100 primes
def product_of_first_100_primes : Nat := first_100_primes.foldr (· * ·) 1
def S := {d : Nat | d > 1 ∧ ∃ m, product_of_first_100_primes = m * d}

/-- Statement of the problem in Lean 4 -/
theorem color_divisors_with_conditions :
  (∃ (k : Nat), (∀ (coloring : S → Fin k), 
    (∀ s1 s2 s3 : S, (s1 * s2 * s3 = product_of_first_100_primes) → (coloring s1 = coloring s2 ∨ coloring s1 = coloring s3 ∨ coloring s2 = coloring s3)) ∧
    (∀ c : Fin k, ∃ s : S, coloring s = c))) ↔ k = 100 := 
by
  sorry

end color_divisors_with_conditions_l126_126478


namespace find_range_of_x_l126_126489

noncomputable def f (x : ℝ) : ℝ := 2^(1 + x^2) - 1 / (1 + x^2)

theorem find_range_of_x (x : ℝ) :
  (f (2 * x) > f (x - 3)) ↔ (x < -3 ∨ x > 1) :=
sorry

end find_range_of_x_l126_126489


namespace main_theorem_l126_126574

noncomputable def circle_center : Prop :=
  ∃ x y : ℝ, 2*x - y - 7 = 0 ∧ y = -3 ∧ x = 2

noncomputable def circle_equation : Prop :=
  (∀ (x y : ℝ), (x - 2)^2 + (y + 3)^2 = 5)

noncomputable def tangent_condition (k : ℝ) : Prop :=
  (3 + 3*k)^2 / (1 + k^2) = 5

noncomputable def symmetric_circle_center : Prop :=
  ∃ x y : ℝ, x = -22/5 ∧ y = 1/5

noncomputable def symmetric_circle_equation : Prop :=
  (∀ (x y : ℝ), (x + 22/5)^2 + (y - 1/5)^2 = 5)

theorem main_theorem : circle_center → circle_equation ∧ (∃ k : ℝ, tangent_condition k) ∧ symmetric_circle_center → symmetric_circle_equation :=
  by sorry

end main_theorem_l126_126574


namespace intersection_M_N_l126_126724

def M : Set ℝ := { x | x < 2017 }
def N : Set ℝ := { x | 0 < x ∧ x < 1 }

theorem intersection_M_N : M ∩ N = { x | 0 < x ∧ x < 1 } := 
by 
  sorry

end intersection_M_N_l126_126724


namespace sum_of_reciprocal_transformed_roots_l126_126447

-- Define the polynomial f
def f (x : ℝ) : ℝ := 15 * x^3 - 35 * x^2 + 20 * x - 2

-- Define the condition that the roots are distinct real numbers between 0 and 1
def is_root (f : ℝ → ℝ) (x : ℝ) : Prop := f x = 0
def roots_between_0_and_1 (a b c : ℝ) : Prop := 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  0 < a ∧ a < 1 ∧ 
  0 < b ∧ b < 1 ∧ 
  0 < c ∧ c < 1 ∧
  is_root f a ∧ is_root f b ∧ is_root f c

-- The theorem representing the proof problem
theorem sum_of_reciprocal_transformed_roots (a b c : ℝ) 
  (h : roots_between_0_and_1 a b c) :
  (1/(1-a)) + (1/(1-b)) + (1/(1-c)) = 2/3 :=
by
  sorry

end sum_of_reciprocal_transformed_roots_l126_126447


namespace day_care_center_toddlers_l126_126932

theorem day_care_center_toddlers (I T : ℕ) (h_ratio1 : 7 * I = 3 * T) (h_ratio2 : 7 * (I + 12) = 5 * T) :
  T = 42 :=
by
  sorry

end day_care_center_toddlers_l126_126932


namespace subset_if_a_neg_third_set_of_real_numbers_for_A_union_B_eq_A_l126_126149

def A : Set ℝ := {x | x ^ 2 - 8 * x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

theorem subset_if_a_neg_third (a : ℝ) (h : a = -1/3) : B a ⊆ A := by
  sorry

theorem set_of_real_numbers_for_A_union_B_eq_A : {a : ℝ | A ∪ B a = A} = {0, -1/3, -1/5} := by
  sorry

end subset_if_a_neg_third_set_of_real_numbers_for_A_union_B_eq_A_l126_126149


namespace dot_product_eq_neg29_l126_126236

-- Given definitions and conditions
variables (a b : ℝ × ℝ)

-- Theorem to prove the dot product condition.
theorem dot_product_eq_neg29 (h1 : a + b = (2, -4)) (h2 : 3 • a - b = (-10, 16)) :
  a.1 * b.1 + a.2 * b.2 = -29 :=
sorry

end dot_product_eq_neg29_l126_126236


namespace inequality_solution_l126_126291

theorem inequality_solution 
  (x : ℝ) 
  (h1 : (x + 3) / 2 ≤ x + 2) 
  (h2 : 2 * (x + 4) > 4 * x + 2) : 
  -1 ≤ x ∧ x < 3 := sorry

end inequality_solution_l126_126291


namespace tan_half_theta_l126_126488

theorem tan_half_theta (θ : ℝ) (h1 : Real.sin θ = -3 / 5) (h2 : 3 * Real.pi < θ ∧ θ < 7 / 2 * Real.pi) :
  Real.tan (θ / 2) = -3 :=
sorry

end tan_half_theta_l126_126488


namespace orthogonal_circles_l126_126733

theorem orthogonal_circles (R1 R2 d : ℝ) :
  (d^2 = R1^2 + R2^2) ↔ (d^2 = R1^2 + R2^2) :=
by sorry

end orthogonal_circles_l126_126733


namespace root_ratio_equiv_l126_126909

theorem root_ratio_equiv :
  (81 ^ (1 / 3)) / (81 ^ (1 / 4)) = 81 ^ (1 / 12) :=
by
  sorry

end root_ratio_equiv_l126_126909


namespace dealer_gross_profit_l126_126354

noncomputable def computeGrossProfit (purchasePrice initialMarkupRate discountRate salesTaxRate: ℝ) : ℝ :=
  let initialSellingPrice := purchasePrice / (1 - initialMarkupRate)
  let discount := discountRate * initialSellingPrice
  let discountedPrice := initialSellingPrice - discount
  let salesTax := salesTaxRate * discountedPrice
  let finalSellingPrice := discountedPrice + salesTax
  finalSellingPrice - purchasePrice - discount

theorem dealer_gross_profit 
  (purchasePrice : ℝ)
  (initialMarkupRate : ℝ)
  (discountRate : ℝ)
  (salesTaxRate : ℝ) 
  (grossProfit : ℝ) :
  purchasePrice = 150 →
  initialMarkupRate = 0.25 →
  discountRate = 0.10 →
  salesTaxRate = 0.05 →
  grossProfit = 19 →
  computeGrossProfit purchasePrice initialMarkupRate discountRate salesTaxRate = grossProfit :=
  by
    intros hp hm hd hs hg
    rw [hp, hm, hd, hs, hg]
    rw [computeGrossProfit]
    sorry

end dealer_gross_profit_l126_126354


namespace refrigerator_cost_is_15000_l126_126934

theorem refrigerator_cost_is_15000 (R : ℝ) 
  (phone_cost : ℝ := 8000)
  (phone_profit : ℝ := 0.10) 
  (fridge_loss : ℝ := 0.03) 
  (overall_profit : ℝ := 350) :
  (0.97 * R + phone_cost * (1 + phone_profit) = (R + phone_cost) + overall_profit) →
  (R = 15000) :=
by
  sorry

end refrigerator_cost_is_15000_l126_126934


namespace pairs_of_boys_girls_l126_126903

theorem pairs_of_boys_girls (a_g b_g a_b b_b : ℕ) 
  (h1 : a_b = 3 * a_g)
  (h2 : b_b = 4 * b_g) :
  ∃ c : ℕ, b_b = 7 * b_g :=
sorry

end pairs_of_boys_girls_l126_126903


namespace point_on_x_axis_m_eq_2_l126_126494

theorem point_on_x_axis_m_eq_2 (m : ℝ) (h : (m + 5, m - 2).2 = 0) : m = 2 :=
sorry

end point_on_x_axis_m_eq_2_l126_126494


namespace Billy_Reads_3_Books_l126_126462

theorem Billy_Reads_3_Books 
    (weekend_days : ℕ) 
    (hours_per_day : ℕ) 
    (reading_percentage : ℕ) 
    (pages_per_hour : ℕ) 
    (pages_per_book : ℕ) : 
    (weekend_days = 2) ∧ 
    (hours_per_day = 8) ∧ 
    (reading_percentage = 25) ∧ 
    (pages_per_hour = 60) ∧ 
    (pages_per_book = 80) → 
    ((weekend_days * hours_per_day * reading_percentage / 100 * pages_per_hour) / pages_per_book = 3) :=
by
  intros
  sorry

end Billy_Reads_3_Books_l126_126462


namespace longest_side_of_garden_l126_126254

theorem longest_side_of_garden (l w : ℝ) (h1 : 2 * l + 2 * w = 225) (h2 : l * w = 8 * 225) :
  l = 93.175 ∨ w = 93.175 :=
by
  sorry

end longest_side_of_garden_l126_126254


namespace prime_factorization_2020_prime_factorization_2021_l126_126384

theorem prime_factorization_2020 : 2020 = 2^2 * 5 * 101 := by
  sorry

theorem prime_factorization_2021 : 2021 = 43 * 47 := by
  sorry

end prime_factorization_2020_prime_factorization_2021_l126_126384


namespace sum_product_smallest_number_l126_126834

theorem sum_product_smallest_number (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 80) : min x y = 8 :=
  sorry

end sum_product_smallest_number_l126_126834


namespace determine_z_l126_126235

theorem determine_z (i z : ℂ) (hi : i^2 = -1) (h : i * z = 2 * z + 1) : 
  z = - (2/5 : ℂ) - (1/5 : ℂ) * i := by
  sorry

end determine_z_l126_126235


namespace Jane_age_proof_l126_126786

theorem Jane_age_proof (D J : ℕ) (h1 : D + 6 = (J + 6) / 2) (h2 : D + 14 = 25) : J = 28 :=
by
  sorry

end Jane_age_proof_l126_126786


namespace surprise_shop_daily_revenue_l126_126895

def closed_days_per_year : ℕ := 3
def years_active : ℕ := 6
def total_revenue_lost : ℚ := 90000

def total_closed_days : ℕ :=
  closed_days_per_year * years_active

def daily_revenue : ℚ :=
  total_revenue_lost / total_closed_days

theorem surprise_shop_daily_revenue :
  daily_revenue = 5000 := by
  sorry

end surprise_shop_daily_revenue_l126_126895


namespace fraction_product_equals_64_l126_126289

theorem fraction_product_equals_64 : 
  (1 / 4) * (8 / 1) * (1 / 32) * (64 / 1) * (1 / 128) * (256 / 1) * (1 / 512) * (1024 / 1) * (1 / 2048) * (4096 / 1) * (1 / 8192) * (16384 / 1) = 64 :=
by
  sorry

end fraction_product_equals_64_l126_126289


namespace battery_life_remaining_l126_126640

variables (full_battery_life : ℕ) (used_fraction : ℚ) (exam_duration : ℕ) (remaining_battery : ℕ)

def brody_calculator_conditions :=
  full_battery_life = 60 ∧
  used_fraction = 3 / 4 ∧
  exam_duration = 2

theorem battery_life_remaining
  (h : brody_calculator_conditions full_battery_life used_fraction exam_duration) :
  remaining_battery = 13 :=
by 
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end battery_life_remaining_l126_126640


namespace arithmetic_geom_sequence_a2_l126_126350

theorem arithmetic_geom_sequence_a2 :
  ∀ (a : ℕ → ℤ), 
    (∀ n, a (n+1) = a n + 2) →  -- Arithmetic sequence with common difference of 2
    a 1 * a 4 = a 3 ^ 2 →  -- Geometric sequence property for a_1, a_3, a_4
    a 2 = -6 :=             -- The value of a_2
by
  intros a h_arith h_geom
  sorry

end arithmetic_geom_sequence_a2_l126_126350


namespace length_of_marquita_garden_l126_126137

variable (length_marquita_garden : ℕ)

def total_area_mancino_gardens : ℕ := 3 * (16 * 5)
def total_gardens_area : ℕ := 304
def total_area_marquita_gardens : ℕ := total_gardens_area - total_area_mancino_gardens
def area_one_marquita_garden : ℕ := total_area_marquita_gardens / 2

theorem length_of_marquita_garden :
  (4 * length_marquita_garden = area_one_marquita_garden) →
  length_marquita_garden = 8 := by
  sorry

end length_of_marquita_garden_l126_126137


namespace football_game_spectators_l126_126538

theorem football_game_spectators (total_wristbands wristbands_per_person : ℕ)
  (h1 : total_wristbands = 250) (h2 : wristbands_per_person = 2) : 
  total_wristbands / wristbands_per_person = 125 :=
by
  sorry

end football_game_spectators_l126_126538


namespace avg_rate_of_change_eq_l126_126680

variable (Δx : ℝ)

def function_y (x : ℝ) : ℝ := x^2 + 1

theorem avg_rate_of_change_eq : (function_y (1 + Δx) - function_y 1) / Δx = 2 + Δx :=
by
  sorry

end avg_rate_of_change_eq_l126_126680


namespace meet_days_l126_126749

-- Definition of conditions
def person_a_days : ℕ := 5
def person_b_days : ℕ := 7
def person_b_early_departure : ℕ := 2

-- Definition of the number of days after A's start that they meet
variable {x : ℕ}

-- Statement to be proven
theorem meet_days (x : ℕ) : (x + 2 : ℚ) / person_b_days + x / person_a_days = 1 := sorry

end meet_days_l126_126749


namespace intersecting_lines_b_plus_m_l126_126395

theorem intersecting_lines_b_plus_m :
  ∃ (m b : ℚ), (∀ x y : ℚ, y = m * x + 5 → y = 4 * x + b → (x, y) = (8, 14)) →
               b + m = -63 / 4 :=
by
  sorry

end intersecting_lines_b_plus_m_l126_126395


namespace solve_for_x_l126_126675

theorem solve_for_x :
  ∀ x : ℤ, (35 - (23 - (15 - x)) = (12 * 2) / 1 / 2) → x = -21 :=
by
  intro x
  sorry

end solve_for_x_l126_126675


namespace factorize_expr_l126_126830

theorem factorize_expr (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) :=
sorry

end factorize_expr_l126_126830


namespace series_pattern_l126_126718

theorem series_pattern :
    (3 / (1 * 2) * (1 / 2) + 4 / (2 * 3) * (1 / 2^2) + 5 / (3 * 4) * (1 / 2^3) + 6 / (4 * 5) * (1 / 2^4) + 7 / (5 * 6) * (1 / 2^5)) 
    = (1 - 1 / (6 * 2^5)) :=
  sorry

end series_pattern_l126_126718


namespace ivan_ivanovich_increase_l126_126458

variable (p v s i : ℝ)
variable (k : ℝ)

-- Conditions
def initial_shares_sum := p + v + s + i = 1
def petya_doubles := 2 * p + v + s + i = 1.3
def vanya_doubles := p + 2 * v + s + i = 1.4
def sergey_triples := p + v + 3 * s + i = 1.2

-- Target statement to be proved
theorem ivan_ivanovich_increase (hp : p = 0.3) (hv : v = 0.4) (hs : s = 0.1)
  (hi : i = 0.2) (k : ℝ) : k * i > 0.75 → k > 3.75 :=
sorry

end ivan_ivanovich_increase_l126_126458


namespace inequality_three_variables_l126_126033

theorem inequality_three_variables (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 1) : 
  (1/x) + (1/y) + (1/z) ≥ 9 := 
by 
  sorry

end inequality_three_variables_l126_126033


namespace Giovanni_burgers_l126_126624

theorem Giovanni_burgers : 
  let toppings := 10
  let patty_choices := 4
  let topping_combinations := 2 ^ toppings
  let total_combinations := patty_choices * topping_combinations
  total_combinations = 4096 :=
by
  sorry

end Giovanni_burgers_l126_126624


namespace largest_pos_int_divisible_l126_126184

theorem largest_pos_int_divisible (n : ℕ) (h1 : n > 0) (h2 : n + 11 ∣ n^3 + 101) : n = 1098 :=
sorry

end largest_pos_int_divisible_l126_126184


namespace remainder_M_mod_32_l126_126678

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end remainder_M_mod_32_l126_126678


namespace triangle_perimeter_l126_126571

-- Definitions of the geometric problem conditions
def inscribed_circle_tangent (A B C P : Type) : Prop := sorry
def radius_of_inscribed_circle (r : ℕ) : Prop := r = 24
def segment_lengths (AP PB : ℕ) : Prop := AP = 25 ∧ PB = 29

-- Main theorem to prove the perimeter of the triangle ABC
theorem triangle_perimeter (A B C P : Type) (r AP PB : ℕ)
  (H1 : inscribed_circle_tangent A B C P)
  (H2 : radius_of_inscribed_circle r)
  (H3 : segment_lengths AP PB) :
  2 * (54 + 208.72) = 525.44 :=
  sorry

end triangle_perimeter_l126_126571


namespace correct_calculation_l126_126581

theorem correct_calculation (x : ℕ) (h1 : 21 * x = 63) : x + 40 = 43 :=
by
  -- proof steps would go here, but we skip them with 'sorry'
  sorry

end correct_calculation_l126_126581


namespace y_expression_l126_126755

theorem y_expression (x y : ℝ) (h : 4 * x + y = 9) : y = 9 - 4 * x := 
by
  sorry

end y_expression_l126_126755


namespace proof_problem_l126_126013

def p : Prop := ∃ x : ℝ, x^2 - x + 1 ≥ 0
def q : Prop := ∀ (a b : ℝ), (a^2 < b^2) → (a < b)

theorem proof_problem (h₁ : p) (h₂ : ¬ q) : p ∧ ¬ q := by
  exact ⟨h₁, h₂⟩

end proof_problem_l126_126013


namespace solve_xy_l126_126449

theorem solve_xy (x y : ℕ) :
  (x^2 + (x + y)^2 = (x + 9)^2) ↔ (x = 0 ∧ y = 9) ∨ (x = 8 ∧ y = 7) ∨ (x = 20 ∧ y = 1) :=
by
  sorry

end solve_xy_l126_126449


namespace find_m_and_domain_parity_of_F_range_of_x_for_F_positive_l126_126225

noncomputable def f (a m x : ℝ) := Real.log (x + m) / Real.log a
noncomputable def g (a x : ℝ) := Real.log (1 - x) / Real.log a
noncomputable def F (a m x : ℝ) := f a m x - g a x

theorem find_m_and_domain (a : ℝ) (m : ℝ) (h : F a m 0 = 0) : m = 1 ∧ ∀ x, -1 < x ∧ x < 1 :=
sorry

theorem parity_of_F (a : ℝ) (m : ℝ) (h : m = 1) : ∀ x, F a m (-x) = -F a m x :=
sorry

theorem range_of_x_for_F_positive (a : ℝ) (m : ℝ) (h : m = 1) :
  (a > 1 → ∀ x, 0 < x ∧ x < 1 → F a m x > 0) ∧ (0 < a ∧ a < 1 → ∀ x, -1 < x ∧ x < 0 → F a m x > 0) :=
sorry

end find_m_and_domain_parity_of_F_range_of_x_for_F_positive_l126_126225


namespace Gretchen_weekend_profit_l126_126143

theorem Gretchen_weekend_profit :
  let saturday_revenue := 24 * 25
  let sunday_revenue := 16 * 15
  let total_revenue := saturday_revenue + sunday_revenue
  let park_fee := 5 * 6 * 2
  let art_supplies_cost := 8 * 2
  let total_expenses := park_fee + art_supplies_cost
  let profit := total_revenue - total_expenses
  profit = 764 :=
by
  sorry

end Gretchen_weekend_profit_l126_126143


namespace fraction_of_y_l126_126982

theorem fraction_of_y (w x y : ℝ) (h1 : wx = y) 
  (h2 : (w + x) / 2 = 0.5) : 
  (2 / w + 2 / x = 2 / y) := 
by
  sorry

end fraction_of_y_l126_126982


namespace sheets_of_paper_in_each_box_l126_126763

theorem sheets_of_paper_in_each_box (S E : ℕ) 
  (h1 : S - E = 70) 
  (h2 : 4 * (E - 20) = S) : 
  S = 120 := 
by 
  sorry

end sheets_of_paper_in_each_box_l126_126763


namespace max_sin_product_proof_l126_126981

noncomputable def max_sin_product : ℝ :=
  let A := (-8, 0)
  let B := (8, 0)
  let C (t : ℝ) := (t, 6)
  let AB : ℝ := 16
  let AC (t : ℝ) := Real.sqrt ((t + 8)^2 + 36)
  let BC (t : ℝ) := Real.sqrt ((t - 8)^2 + 36)
  let area : ℝ := 48
  let sin_ACB (t : ℝ) := 96 / Real.sqrt (((t + 8)^2 + 36) * ((t - 8)^2 + 36))
  let sin_CAB_CBA : ℝ := 3 / 8
  sin_CAB_CBA

theorem max_sin_product_proof : ∀ t : ℝ, max_sin_product = 3 / 8 :=
by
  sorry

end max_sin_product_proof_l126_126981


namespace four_digit_numbers_proof_l126_126353

noncomputable def four_digit_numbers_total : ℕ := 9000
noncomputable def two_digit_numbers_total : ℕ := 90
noncomputable def max_distinct_products : ℕ := 4095
noncomputable def cannot_be_expressed_as_product : ℕ := four_digit_numbers_total - max_distinct_products

theorem four_digit_numbers_proof :
  cannot_be_expressed_as_product = 4905 :=
by
  sorry

end four_digit_numbers_proof_l126_126353


namespace factor_3a3_minus_6a2_plus_3a_factor_a2_minus_b2_x_minus_y_factor_16a_plus_b_sq_minus_9a_minus_b_sq_l126_126750

-- First factorization problem
theorem factor_3a3_minus_6a2_plus_3a (a : ℝ) : 
  3 * a ^ 3 - 6 * a ^ 2 + 3 * a = 3 * a * (a - 1) ^ 2 :=
by sorry

-- Second factorization problem
theorem factor_a2_minus_b2_x_minus_y (a b x y : ℝ) : 
  a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a - b) * (a + b) :=
by sorry

-- Third factorization problem
theorem factor_16a_plus_b_sq_minus_9a_minus_b_sq (a b : ℝ) : 
  16 * (a + b) ^ 2 - 9 * (a - b) ^ 2 = (a + 7 * b) * (7 * a + b) :=
by sorry

end factor_3a3_minus_6a2_plus_3a_factor_a2_minus_b2_x_minus_y_factor_16a_plus_b_sq_minus_9a_minus_b_sq_l126_126750


namespace total_avg_donation_per_person_l126_126863

-- Definition of variables and conditions
variables (avgA avgB : ℝ) (numA numB : ℕ)
variables (h1 : avgB = avgA - 100)
variables (h2 : 2 * numA * avgA = 4 * numB * (avgA - 100))
variables (h3 : numA = numB / 4)

-- Lean 4 statement to prove the total average donation per person is 120
theorem total_avg_donation_per_person (h1 :  avgB = avgA - 100)
    (h2 : 2 * numA * avgA = 4 * numB * (avgA - 100))
    (h3 : numA = numB / 4) : 
    ( (numA * avgA + numB * avgB) / (numA + numB) ) = 120 :=
sorry

end total_avg_donation_per_person_l126_126863


namespace average_of_first_45_results_l126_126973

theorem average_of_first_45_results
  (A : ℝ)
  (h1 : (45 + 25 : ℝ) = 70)
  (h2 : (25 : ℝ) * 45 = 1125)
  (h3 : (70 : ℝ) * 32.142857142857146 = 2250)
  (h4 : ∀ x y z : ℝ, 45 * x + y = z → x = 25) :
  A = 25 :=
by
  sorry

end average_of_first_45_results_l126_126973


namespace ladybugs_without_spots_l126_126361

-- Defining the conditions given in the problem
def total_ladybugs : ℕ := 67082
def ladybugs_with_spots : ℕ := 12170

-- Proving the number of ladybugs without spots
theorem ladybugs_without_spots : total_ladybugs - ladybugs_with_spots = 54912 := by
  sorry

end ladybugs_without_spots_l126_126361


namespace mary_should_drink_6_glasses_l126_126023

-- Definitions based on conditions
def daily_water_goal_liters : ℚ := 1.5
def glass_capacity_ml : ℚ := 250
def liter_to_milliliters : ℚ := 1000

-- Conversion from liters to milliliters
def daily_water_goal_milliliters : ℚ := daily_water_goal_liters * liter_to_milliliters

-- Proof problem to show Mary needs 6 glasses per day
theorem mary_should_drink_6_glasses :
  daily_water_goal_milliliters / glass_capacity_ml = 6 := by
  sorry

end mary_should_drink_6_glasses_l126_126023


namespace derivative_y_over_x_l126_126876

noncomputable def x (t : ℝ) : ℝ := (t^2 * Real.log t) / (1 - t^2) + Real.log (Real.sqrt (1 - t^2))
noncomputable def y (t : ℝ) : ℝ := (t / Real.sqrt (1 - t^2)) * Real.arcsin t + Real.log (Real.sqrt (1 - t^2))

theorem derivative_y_over_x (t : ℝ) (ht : t ≠ 0) (h1 : t ≠ 1) (hneg1 : t ≠ -1) : 
  (deriv y t) / (deriv x t) = (Real.arcsin t * Real.sqrt (1 - t^2)) / (2 * t * Real.log t) :=
by
  sorry

end derivative_y_over_x_l126_126876


namespace does_not_uniquely_determine_equilateral_l126_126651

def equilateral_triangle (a b c : ℕ) : Prop :=
a = b ∧ b = c

def right_triangle (a b c : ℕ) : Prop :=
a^2 + b^2 = c^2

def isosceles_triangle (a b c : ℕ) : Prop :=
a = b ∨ b = c ∨ a = c

def scalene_triangle (a b c : ℕ) : Prop :=
a ≠ b ∧ b ≠ c ∧ a ≠ c

def circumscribed_circle_radius (a b c r : ℕ) : Prop :=
r = a * b * c / (4 * (a * b * c))

def angle_condition (α β γ : ℕ) (t : ℕ → ℕ → ℕ → Prop) : Prop :=
∃ (a b c : ℕ), t a b c ∧ α + β + γ = 180

theorem does_not_uniquely_determine_equilateral :
  ¬ ∃ (α β : ℕ), equilateral_triangle α β β ∧ α + β = 120 :=
sorry

end does_not_uniquely_determine_equilateral_l126_126651


namespace base_conversion_addition_l126_126532

theorem base_conversion_addition :
  (214 % 8 / 32 % 5 + 343 % 9 / 133 % 4) = 9134 / 527 :=
by sorry

end base_conversion_addition_l126_126532


namespace ellipse_foci_distance_l126_126487

theorem ellipse_foci_distance :
  (∀ x y : ℝ, x^2 / 56 + y^2 / 14 = 8) →
  ∃ d : ℝ, d = 8 * Real.sqrt 21 :=
by
  sorry

end ellipse_foci_distance_l126_126487


namespace proposition_B_proposition_D_l126_126996

open Real

variable (a b : ℝ)

theorem proposition_B (h : a^2 ≠ b^2) : a ≠ b := 
sorry

theorem proposition_D (h : a > abs b) : a^2 > b^2 :=
sorry

end proposition_B_proposition_D_l126_126996


namespace find_factor_l126_126390

theorem find_factor {n f : ℝ} (h1 : n = 10) (h2 : f * (2 * n + 8) = 84) : f = 3 :=
by
  sorry

end find_factor_l126_126390


namespace inequality_AM_GM_l126_126527

theorem inequality_AM_GM (a b t : ℝ) (h₁ : 1 < a) (h₂ : 1 < b) (h₃ : 0 < t) : 
  (a^2 / (b^t - 1) + b^(2 * t) / (a^t - 1)) ≥ 8 :=
by
  sorry

end inequality_AM_GM_l126_126527


namespace fraction_sum_lt_one_l126_126355

theorem fraction_sum_lt_one (n : ℕ) (h_pos : n > 0) : 
  (1 / 2 + 1 / 3 + 1 / 10 + 1 / n < 1) ↔ (n > 15) :=
sorry

end fraction_sum_lt_one_l126_126355


namespace sqrt_2_plus_x_nonnegative_l126_126134

theorem sqrt_2_plus_x_nonnegative (x : ℝ) : (2 + x ≥ 0) → (x ≥ -2) :=
by
  sorry

end sqrt_2_plus_x_nonnegative_l126_126134


namespace soap_bubble_thickness_scientific_notation_l126_126012

theorem soap_bubble_thickness_scientific_notation :
  (0.0007 * 0.001) = 7 * 10^(-7) := by
sorry

end soap_bubble_thickness_scientific_notation_l126_126012


namespace emily_can_see_emerson_l126_126933

theorem emily_can_see_emerson : 
  ∀ (emily_speed emerson_speed : ℝ) 
    (initial_distance final_distance : ℝ), 
  emily_speed = 15 → 
  emerson_speed = 9 → 
  initial_distance = 1 → 
  final_distance = 1 →
  (initial_distance / (emily_speed - emerson_speed) + final_distance / (emily_speed - emerson_speed)) * 60 = 20 :=
by
  intros emily_speed emerson_speed initial_distance final_distance
  sorry

end emily_can_see_emerson_l126_126933


namespace sum_of_a_and_b_l126_126408

theorem sum_of_a_and_b (a b : ℕ) (h1 : a > 0) (h2 : b > 1) (h3 : a^b < 500) (h_max : ∀ (a' b' : ℕ), a' > 0 → b' > 1 → a'^b' < 500 → a'^b' ≤ a^b) : a + b = 24 :=
sorry

end sum_of_a_and_b_l126_126408


namespace curve_C1_parametric_equiv_curve_C2_general_equiv_curve_C3_rectangular_equiv_max_distance_C2_to_C3_l126_126815

-- Definitions of the curves
def curve_C1 (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 1
def curve_C2_parametric (theta : ℝ) (x y : ℝ) : Prop := (x = 4 * Real.cos theta) ∧ (y = 3 * Real.sin theta)
def curve_C3_polar (rho theta : ℝ) : Prop := rho * (Real.cos theta - 2 * Real.sin theta) = 7

-- Proving the mathematical equivalence:
theorem curve_C1_parametric_equiv (t : ℝ) : ∃ x y, curve_C1 x y ∧ (x = 3 + Real.cos t) ∧ (y = 2 + Real.sin t) :=
by sorry

theorem curve_C2_general_equiv (x y : ℝ) : (∃ theta, curve_C2_parametric theta x y) ↔ (x^2 / 16 + y^2 / 9 = 1) :=
by sorry

theorem curve_C3_rectangular_equiv (x y : ℝ) : (∃ rho theta, x = rho * Real.cos theta ∧ y = rho * Real.sin theta ∧ curve_C3_polar rho theta) ↔ (x - 2 * y - 7 = 0) :=
by sorry

theorem max_distance_C2_to_C3 : ∃ (d : ℝ), d = (2 * Real.sqrt 65 + 7 * Real.sqrt 5) / 5 :=
by sorry

end curve_C1_parametric_equiv_curve_C2_general_equiv_curve_C3_rectangular_equiv_max_distance_C2_to_C3_l126_126815


namespace find_larger_number_l126_126385

variable (x y : ℝ)
axiom h1 : x + y = 27
axiom h2 : x - y = 5

theorem find_larger_number : x = 16 :=
by
  sorry

end find_larger_number_l126_126385


namespace fifteenth_number_in_base_5_l126_126516

theorem fifteenth_number_in_base_5 :
  ∃ n : ℕ, n = 15 ∧ (n : ℕ) = 3 * 5^1 + 0 * 5^0 :=
by
  sorry

end fifteenth_number_in_base_5_l126_126516


namespace problem_projection_eq_l126_126671

variable (m n : ℝ × ℝ)
variable (m_val : m = (1, 2))
variable (n_val : n = (2, 3))

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def projection (u v : ℝ × ℝ) : ℝ :=
  (dot_product u v) / (magnitude v)

theorem problem_projection_eq : projection m n = (8 * Real.sqrt 13) / 13 :=
by
  rw [m_val, n_val]
  sorry

end problem_projection_eq_l126_126671


namespace find_r_l126_126699

theorem find_r (r : ℝ) (h₁ : 0 < r) (h₂ : ∀ x y : ℝ, (x - y = r → x^2 + y^2 = r → False)) : r = 2 :=
sorry

end find_r_l126_126699


namespace price_of_eraser_l126_126358

variables (x y : ℝ)

theorem price_of_eraser : 
  (3 * x + 5 * y = 10.6) ∧ (4 * x + 4 * y = 12) → x = 2.2 :=
by
  sorry

end price_of_eraser_l126_126358


namespace ticTacToeConfigCorrect_l126_126695

def ticTacToeConfigCount (board : Fin 3 → Fin 3 → Option Char) : Nat := 
  sorry -- this function will count the configurations according to the game rules

theorem ticTacToeConfigCorrect (board : Fin 3 → Fin 3 → Option Char) :
  ticTacToeConfigCount board = 438 := 
  sorry

end ticTacToeConfigCorrect_l126_126695


namespace vector_a_properties_l126_126394

-- Definitions of the points in space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Definition of vector subtraction to find the vector between two points
def vector_sub (p1 p2 : Point3D) : Point3D :=
  { x := p2.x - p1.x, y := p2.y - p1.y, z := p2.z - p1.z }

-- Definition of dot product for vectors
def dot_product (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

-- Definition of vector magnitude squared for vectors
def magnitude_squared (v : Point3D) : ℝ :=
  v.x * v.x + v.y * v.y + v.z * v.z

-- Main theorem statement
theorem vector_a_properties :
  let A := {x := 0, y := 2, z := 3}
  let B := {x := -2, y := 1, z := 6}
  let C := {x := 1, y := -1, z := 5}
  let AB := vector_sub A B
  let AC := vector_sub A C
  ∀ (a : Point3D), 
    (magnitude_squared a = 3) → 
    (dot_product a AB = 0) → 
    (dot_product a AC = 0) → 
    (a = {x := 1, y := 1, z := 1} ∨ a = {x := -1, y := -1, z := -1}) := 
by
  intros A B C AB AC a ha_magnitude ha_perpendicular_AB ha_perpendicular_AC
  sorry

end vector_a_properties_l126_126394


namespace f_of_10_is_20_l126_126758

theorem f_of_10_is_20 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (3 * x + 1) = x^2 + 3 * x + 2) : f 10 = 20 :=
  sorry

end f_of_10_is_20_l126_126758


namespace systematic_sampling_distance_l126_126391

-- Conditions
def total_students : ℕ := 1200
def sample_size : ℕ := 30

-- Problem: Compute sampling distance
def sampling_distance (n : ℕ) (m : ℕ) : ℕ := n / m

-- The formal proof statement
theorem systematic_sampling_distance :
  sampling_distance total_students sample_size = 40 := by
  sorry

end systematic_sampling_distance_l126_126391


namespace part1_extreme_value_at_2_part2_increasing_function_l126_126195

noncomputable def f (a x : ℝ) := a * x - a / x - 2 * Real.log x

theorem part1_extreme_value_at_2 (a : ℝ) :
  (∃ x : ℝ, x = 2 ∧ ∀ y : ℝ, f a x ≥ f a y) → a = 4 / 5 ∧ f a 1/2 = 2 * Real.log 2 - 6 / 5 := by
  sorry

theorem part2_increasing_function (a : ℝ) :
  (∀ x : ℝ, 0 < x → deriv (f a) x ≥ 0) → a ≥ 1 := by
  sorry

end part1_extreme_value_at_2_part2_increasing_function_l126_126195


namespace ribbon_tape_remaining_l126_126911

theorem ribbon_tape_remaining 
  (initial_length used_for_ribbon used_for_gift : ℝ)
  (h_initial: initial_length = 1.6)
  (h_ribbon: used_for_ribbon = 0.8)
  (h_gift: used_for_gift = 0.3) : 
  initial_length - used_for_ribbon - used_for_gift = 0.5 :=
by 
  sorry

end ribbon_tape_remaining_l126_126911


namespace find_x_2y_3z_l126_126490

theorem find_x_2y_3z (x y z : ℕ) (hx : Prime x) (hy : Prime y) (hz : Prime z)
  (h1 : x ≤ y) (h2 : y ≤ z) (h3 : x + y + z = 12) (h4 : x * y + y * z + z * x = 41) :
  x + 2 * y + 3 * z = 29 :=
by
  sorry

end find_x_2y_3z_l126_126490


namespace number_of_girls_more_than_boys_l126_126100

theorem number_of_girls_more_than_boys
    (total_students : ℕ)
    (number_of_boys : ℕ)
    (h1 : total_students = 485)
    (h2 : number_of_boys = 208) :
    total_students - number_of_boys - number_of_boys = 69 :=
by
    sorry

end number_of_girls_more_than_boys_l126_126100


namespace gcd_m_n_l126_126761

def m : ℕ := 333333
def n : ℕ := 888888888

theorem gcd_m_n : Nat.gcd m n = 3 := by
  sorry

end gcd_m_n_l126_126761


namespace original_number_increased_by_40_percent_l126_126850

theorem original_number_increased_by_40_percent (x : ℝ) (h : 1.40 * x = 700) : x = 500 :=
by
  sorry

end original_number_increased_by_40_percent_l126_126850


namespace number_is_nine_l126_126744

theorem number_is_nine (x : ℤ) (h : 3 * (2 * x + 9) = 81) : x = 9 :=
by
  sorry

end number_is_nine_l126_126744


namespace repeated_number_divisibility_l126_126094

theorem repeated_number_divisibility (x : ℕ) (h : 1000 ≤ x ∧ x < 10000) :
  73 ∣ (10001 * x) ∧ 137 ∣ (10001 * x) :=
sorry

end repeated_number_divisibility_l126_126094


namespace f_satisfies_equation_l126_126418

noncomputable def f (x : ℝ) : ℝ := (20 / 3) * x * (Real.sqrt (1 - x^2))

theorem f_satisfies_equation (f : ℝ → ℝ) :
  (∀ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), 2 * f (Real.sin x * -1) + 3 * f (Real.sin x) = 4 * Real.sin x * Real.cos x) →
  (∀ x ∈ Set.Icc (-Real.sqrt 2 / 2) (Real.sqrt 2 / 2), f x = (20 / 3) * x * (Real.sqrt (1 - x^2))) :=
by
  intro h
  sorry

end f_satisfies_equation_l126_126418


namespace promotional_event_probabilities_l126_126629

def P_A := 1 / 1000
def P_B := 1 / 100
def P_C := 1 / 20
def P_A_B_C := P_A + P_B + P_C
def P_A_B := P_A + P_B
def P_complement_A_B := 1 - P_A_B

theorem promotional_event_probabilities :
  P_A = 1 / 1000 ∧
  P_B = 1 / 100 ∧
  P_C = 1 / 20 ∧
  P_A_B_C = 61 / 1000 ∧
  P_complement_A_B = 989 / 1000 :=
by
  sorry

end promotional_event_probabilities_l126_126629


namespace r_p_q_sum_l126_126019

theorem r_p_q_sum (t p q r : ℕ) (h1 : (1 + Real.sin t) * (1 + Real.cos t) = 9 / 4)
    (h2 : (1 - Real.sin t) * (1 - Real.cos t) = p / q - Real.sqrt r)
    (h3 : r > 0) (h4 : p > 0) (h5 : q > 0)
    (h6 : Nat.gcd p q = 1) : r + p + q = 5 := 
sorry

end r_p_q_sum_l126_126019


namespace condition_for_equation_l126_126752

theorem condition_for_equation (a b c d : ℝ) 
  (h : (a^2 + b) / (b + c^2) = (c^2 + d) / (d + a^2)) : 
  a = c ∨ a^2 + d + 2 * b = 0 :=
by
  sorry

end condition_for_equation_l126_126752


namespace union_eq_set_l126_126484

noncomputable def M : Set ℤ := {x | |x| < 2}
noncomputable def N : Set ℤ := {-2, -1, 0}

theorem union_eq_set : M ∪ N = {-2, -1, 0, 1} := by
  sorry

end union_eq_set_l126_126484


namespace library_books_count_l126_126719

def students_per_day : List ℕ := [4, 5, 6, 9]
def books_per_student : ℕ := 5
def total_books_given (students : List ℕ) (books_per_student : ℕ) : ℕ :=
  students.foldl (λ acc n => acc + n * books_per_student) 0

theorem library_books_count :
  total_books_given students_per_day books_per_student = 120 :=
by
  sorry

end library_books_count_l126_126719


namespace quadratic_always_positive_l126_126206

theorem quadratic_always_positive (k : ℝ) :
  ∀ x : ℝ, x^2 - (k - 4) * x + k - 7 > 0 :=
sorry

end quadratic_always_positive_l126_126206


namespace sum_infinite_series_l126_126053

theorem sum_infinite_series : (∑' n : ℕ, (n + 1) / 8^(n + 1)) = 8 / 49 := sorry

end sum_infinite_series_l126_126053


namespace candy_problem_l126_126871

theorem candy_problem
  (x y m : ℤ)
  (hx : x ≥ 0)
  (hy : y ≥ 0)
  (hxy : x + y = 176)
  (hcond : x - m * (y - 16) = 47)
  (hm : m > 1) :
  x ≥ 131 := 
sorry

end candy_problem_l126_126871


namespace arcsin_neg_one_half_l126_126998

theorem arcsin_neg_one_half : Real.arcsin (-1 / 2) = -Real.pi / 6 :=
by
  sorry

end arcsin_neg_one_half_l126_126998


namespace single_elimination_games_l126_126245

theorem single_elimination_games (n : ℕ) (h : n = 512) : ∃ g : ℕ, g = 511 := by
  sorry

end single_elimination_games_l126_126245


namespace solve_polynomial_l126_126992

theorem solve_polynomial (z : ℂ) :
    z^5 - 5 * z^3 + 6 * z = 0 ↔ 
    z = 0 ∨ z = -Real.sqrt 2 ∨ z = Real.sqrt 2 ∨ z = -Real.sqrt 3 ∨ z = Real.sqrt 3 := 
by 
  sorry

end solve_polynomial_l126_126992


namespace exists_infinite_bisecting_circles_l126_126691

-- Define circle and bisecting condition
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def bisects (B C : Circle) : Prop :=
  let chord_len := (B.center.1 - C.center.1)^2 + (B.center.2 - C.center.2)^2 + C.radius^2
  (B.center.1 - C.center.1)^2 + (B.center.2 - C.center.2)^2 + C.radius^2 = B.radius^2

-- Define the theorem statement
theorem exists_infinite_bisecting_circles (C1 C2 : Circle) (h : C1.center ≠ C2.center) :
  ∃ (B : Circle), bisects B C1 ∧ bisects B C2 ∧
  ∀ (b_center : ℝ × ℝ), (∃ (B : Circle), bisects B C1 ∧ bisects B C2 ∧ B.center = b_center) ↔
  2 * (C2.center.1 - C1.center.1) * b_center.1 + 2 * (C2.center.2 - C1.center.2) * b_center.2 =
  (C2.center.1^2 - C1.center.1^2) + (C2.center.2^2 - C1.center.2^2) + (C2.radius^2 - C1.radius^2) := 
sorry

end exists_infinite_bisecting_circles_l126_126691


namespace jaya_amitabh_number_of_digits_l126_126756

-- Definitions
def is_two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def digit_sum (n1 n2 : ℕ) : ℕ :=
  let (d1, d2) := (n1 % 10, n1 / 10)
  let (d3, d4) := (n2 % 10, n2 / 10)
  d1 + d2 + d3 + d4
def append_ages (j a : ℕ) : ℕ := 1000 * (j / 10) + 100 * (j % 10) + 10 * (a / 10) + (a % 10)
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Main theorem
theorem jaya_amitabh_number_of_digits 
  (j a : ℕ) 
  (hj : is_two_digit_number j)
  (ha : is_two_digit_number a)
  (h_sum : digit_sum j a = 7)
  (h_square : is_perfect_square (append_ages j a)) : 
  ∃ n : ℕ, String.length (toString (append_ages j a)) = 4 :=
by
  sorry

end jaya_amitabh_number_of_digits_l126_126756


namespace dogs_in_pet_shop_l126_126421

variable (D C B x : ℕ)

theorem dogs_in_pet_shop 
  (h1 : D = 7 * x) 
  (h2 : B = 8 * x)
  (h3 : D + B = 330) : 
  D = 154 :=
by
  sorry

end dogs_in_pet_shop_l126_126421


namespace fraction_of_juniors_l126_126216

theorem fraction_of_juniors (J S : ℕ) (h1 : 0 < J) (h2 : 0 < S) (h3 : J = (4 / 3) * S) :
  (J : ℚ) / (J + S) = 4 / 7 :=
by
  sorry

end fraction_of_juniors_l126_126216


namespace intersection_of_M_and_N_l126_126373

open Set

def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | x^2 - 25 < 0}
def I : Set ℝ := {x | 2 ≤ x ∧ x < 5}

theorem intersection_of_M_and_N : M ∩ N = I := by
  sorry

end intersection_of_M_and_N_l126_126373


namespace alice_distance_from_start_l126_126694

theorem alice_distance_from_start :
  let hexagon_side := 3
  let distance_walked := 10
  let final_distance := 3 * Real.sqrt 3 / 2
  final_distance =
    let a := (0, 0)
    let b := (3, 0)
    let c := (4.5, 3 * Real.sqrt 3 / 2)
    let d := (1.5, 3 * Real.sqrt 3 / 2)
    let e := (0, 3 * Real.sqrt 3 / 2)
    dist a e := sorry

end alice_distance_from_start_l126_126694


namespace min_max_abs_poly_eq_zero_l126_126401

theorem min_max_abs_poly_eq_zero :
  ∃ y : ℝ, (∀ x : ℝ, 0 ≤ x → x ≤ 1 → |x^2 - x^3 * y| ≤ 0) :=
sorry

end min_max_abs_poly_eq_zero_l126_126401


namespace pizza_pasta_cost_difference_l126_126032

variable (x y z : ℝ)
variable (A1 : 2 * x + 3 * y + 4 * z = 53)
variable (A2 : 5 * x + 6 * y + 7 * z = 107)

theorem pizza_pasta_cost_difference :
  x - z = 1 :=
by
  sorry

end pizza_pasta_cost_difference_l126_126032


namespace sufficient_condition_for_min_value_not_necessary_condition_for_min_value_l126_126590

noncomputable def f (x b : ℝ) : ℝ := x^2 + b*x

theorem sufficient_condition_for_min_value (b : ℝ) : b < 0 → ∀ x, min (f (f x b) b) = min (f x b) :=
sorry

theorem not_necessary_condition_for_min_value (b : ℝ) : (b < 0) ∧ (∀ x, min (f (f x b) b) = min (f x b)) → b ≤ 0 ∨ b ≥ 2 := 
sorry

end sufficient_condition_for_min_value_not_necessary_condition_for_min_value_l126_126590


namespace ordering_l126_126952

noncomputable def a : ℝ := 1 / (Real.exp 0.6)
noncomputable def b : ℝ := 0.4
noncomputable def c : ℝ := Real.log 1.4 / 1.4

theorem ordering : a > b ∧ b > c :=
by
  have ha : a = 1 / (Real.exp 0.6) := rfl
  have hb : b = 0.4 := rfl
  have hc : c = Real.log 1.4 / 1.4 := rfl
  sorry

end ordering_l126_126952


namespace marbles_left_l126_126256

def initial_marbles : ℝ := 150
def lost_marbles : ℝ := 58.5
def given_away_marbles : ℝ := 37.2
def found_marbles : ℝ := 10.8

theorem marbles_left :
  initial_marbles - lost_marbles - given_away_marbles + found_marbles = 65.1 :=
by 
  sorry

end marbles_left_l126_126256


namespace expectation_is_four_thirds_l126_126244

-- Define the probability function
def P_ξ (k : ℕ) : ℚ :=
  if k = 0 then (1/2)^2 * (2/3)
  else if k = 1 then (1/2) * (1/2) * (2/3) + (1/2) * (1/2) * (2/3) + (1/2) * (1/2) * (1/3)
  else if k = 2 then (1/2) * (1/2) * (2/3) + (1/2) * (1/2) * (1/3) + (1/2) * (1/2) * (1/3)
  else if k = 3 then (1/2) * (1/2) * (1/3)
  else 0

-- Define the expected value function
def E_ξ : ℚ :=
  0 * P_ξ 0 + 1 * P_ξ 1 + 2 * P_ξ 2 + 3 * P_ξ 3

-- Formal statement of the problem
theorem expectation_is_four_thirds : E_ξ = 4 / 3 :=
  sorry

end expectation_is_four_thirds_l126_126244


namespace not_solvable_det_three_times_l126_126472

theorem not_solvable_det_three_times (a b c d : ℝ) (h : a * d - b * c = 5) :
  ¬∃ (x : ℝ), (3 * a + 1) * (3 * d + 1) - (3 * b + 1) * (3 * c + 1) = x :=
by {
  -- This is where the proof would go, but the problem states that it's not solvable with the given information.
  sorry
}

end not_solvable_det_three_times_l126_126472


namespace no_solution_exists_l126_126793

theorem no_solution_exists : 
  ¬(∃ x y : ℝ, 2 * x - 3 * y = 7 ∧ 4 * x - 6 * y = 20) :=
by
  sorry

end no_solution_exists_l126_126793


namespace abe_age_sum_l126_126705

theorem abe_age_sum (x : ℕ) : 25 + (25 - x) = 29 ↔ x = 21 :=
by sorry

end abe_age_sum_l126_126705


namespace total_votes_is_240_l126_126492

-- Defining the problem conditions
variables (total_votes : ℕ) (likes : ℕ) (dislikes : ℕ)
def score : ℤ := likes - dislikes
def percentage_likes : ℚ := 3 / 4
def percentage_dislikes : ℚ := 1 / 4

-- Stating the given conditions
axiom h1 : total_votes = likes + dislikes
axiom h2 : (likes : ℤ) = (percentage_likes * total_votes)
axiom h3 : (dislikes : ℤ) = (percentage_dislikes * total_votes)
axiom h4 : score = 120

-- The statement to prove
theorem total_votes_is_240 : total_votes = 240 :=
by
  sorry

end total_votes_is_240_l126_126492


namespace min_value_xy_l126_126950

theorem min_value_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y + 6 = x * y) : x * y ≥ 18 := 
sorry

end min_value_xy_l126_126950


namespace compare_magnitude_p2_for_n1_compare_magnitude_p2_for_n2_compare_magnitude_p2_for_n_ge_3_compare_magnitude_p_eq_n_for_all_n_l126_126200

def a_n (p n : ℕ) : ℕ := (2 * n + 1) ^ p
def b_n (p n : ℕ) : ℕ := (2 * n) ^ p + (2 * n - 1) ^ p

theorem compare_magnitude_p2_for_n1 :
  b_n 2 1 < a_n 2 1 := sorry

theorem compare_magnitude_p2_for_n2 :
  b_n 2 2 = a_n 2 2 := sorry

theorem compare_magnitude_p2_for_n_ge_3 (n : ℕ) (hn : n ≥ 3) :
  b_n 2 n > a_n 2 n := sorry

theorem compare_magnitude_p_eq_n_for_all_n (n : ℕ) :
  a_n n n ≥ b_n n n := sorry

end compare_magnitude_p2_for_n1_compare_magnitude_p2_for_n2_compare_magnitude_p2_for_n_ge_3_compare_magnitude_p_eq_n_for_all_n_l126_126200


namespace rachel_remaining_pictures_l126_126548

theorem rachel_remaining_pictures 
  (p1 p2 p_colored : ℕ)
  (h1 : p1 = 23)
  (h2 : p2 = 32)
  (h3 : p_colored = 44) :
  (p1 + p2 - p_colored = 11) :=
by
  sorry

end rachel_remaining_pictures_l126_126548


namespace eval_expression_l126_126316

theorem eval_expression :
  2^0 + 9^5 / 9^3 = 82 :=
by
  have h1 : 2^0 = 1 := by sorry
  have h2 : 9^5 / 9^3 = 9^(5-3) := by sorry
  have h3 : 9^(5-3) = 9^2 := by sorry
  have h4 : 9^2 = 81 := by sorry
  sorry

end eval_expression_l126_126316


namespace dog_food_vs_cat_food_l126_126185

-- Define the quantities of dog food and cat food
def dog_food : ℕ := 600
def cat_food : ℕ := 327

-- Define the problem as a statement asserting the required difference
theorem dog_food_vs_cat_food : dog_food - cat_food = 273 := by
  sorry

end dog_food_vs_cat_food_l126_126185


namespace second_player_wins_optimal_play_l126_126715

def players_take_turns : Prop := sorry
def win_condition (box_count : ℕ) : Prop := box_count = 21

theorem second_player_wins_optimal_play (boxes : Fin 11 → ℕ)
    (h_turns : players_take_turns)
    (h_win : ∀ i : Fin 11, win_condition (boxes i)) : 
    ∃ P : ℕ, P = 2 :=
sorry

end second_player_wins_optimal_play_l126_126715


namespace vector_calculation_l126_126363

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (1, -1)

def vector_operation (a b : ℝ × ℝ) : ℝ × ℝ :=
(3 * a.1 - 2 * b.1, 3 * a.2 - 2 * b.2)

theorem vector_calculation : vector_operation vector_a vector_b = (1, 5) :=
by sorry

end vector_calculation_l126_126363


namespace pair_of_operations_equal_l126_126497

theorem pair_of_operations_equal :
  (-3) ^ 3 = -(3 ^ 3) ∧
  (¬((-2) ^ 4 = -(2 ^ 4))) ∧
  (¬((3 / 2) ^ 2 = (2 / 3) ^ 2)) ∧
  (¬(2 ^ 3 = 3 ^ 2)) :=
by 
  sorry

end pair_of_operations_equal_l126_126497


namespace total_amount_spent_l126_126502

noncomputable def food_price : ℝ := 160
noncomputable def sales_tax_rate : ℝ := 0.10
noncomputable def tip_rate : ℝ := 0.20

theorem total_amount_spent :
  let sales_tax := sales_tax_rate * food_price
  let total_before_tip := food_price + sales_tax
  let tip := tip_rate * total_before_tip
  let total_amount := total_before_tip + tip
  total_amount = 211.20 :=
by
  -- include the proof logic here if necessary
  sorry

end total_amount_spent_l126_126502


namespace contractor_absent_days_proof_l126_126833

def contractor_absent_days (x y : ℝ) : Prop :=
  x + y = 30 ∧ 25 * x - 7.5 * y = 425

theorem contractor_absent_days_proof : ∃ (y : ℝ), contractor_absent_days x y ∧ y = 10 :=
by
  sorry

end contractor_absent_days_proof_l126_126833


namespace angle_problem_l126_126686

-- Definitions for degrees and minutes
structure Angle where
  degrees : ℕ
  minutes : ℕ

-- Adding two angles
def add_angles (a1 a2 : Angle) : Angle :=
  let total_minutes := a1.minutes + a2.minutes
  let extra_degrees := total_minutes / 60
  { degrees := a1.degrees + a2.degrees + extra_degrees,
    minutes := total_minutes % 60 }

-- Subtracting two angles
def sub_angles (a1 a2 : Angle) : Angle :=
  let total_minutes := if a1.minutes < a2.minutes then a1.minutes + 60 else a1.minutes
  let extra_deg := if a1.minutes < a2.minutes then 1 else 0
  { degrees := a1.degrees - a2.degrees - extra_deg,
    minutes := total_minutes - a2.minutes }

-- Multiplying an angle by a constant
def mul_angle (a : Angle) (k : ℕ) : Angle :=
  let total_minutes := a.minutes * k
  let extra_degrees := total_minutes / 60
  { degrees := a.degrees * k + extra_degrees,
    minutes := total_minutes % 60 }

-- Given angles
def angle1 : Angle := { degrees := 24, minutes := 31}
def angle2 : Angle := { degrees := 62, minutes := 10}

-- Prove the problem statement
theorem angle_problem : sub_angles (mul_angle angle1 4) angle2 = { degrees := 35, minutes := 54} :=
  sorry

end angle_problem_l126_126686


namespace no_real_solutions_l126_126851

theorem no_real_solutions :
  ∀ x : ℝ, (2 * x - 6) ^ 2 + 4 ≠ -(x - 3) :=
by
  intro x
  sorry

end no_real_solutions_l126_126851


namespace coffee_shop_brewed_cups_in_week_l126_126122

theorem coffee_shop_brewed_cups_in_week 
    (weekday_rate : ℕ) (weekend_rate : ℕ)
    (weekday_hours : ℕ) (saturday_hours : ℕ) (sunday_hours : ℕ)
    (num_weekdays : ℕ) (num_saturdays : ℕ) (num_sundays : ℕ)
    (h1 : weekday_rate = 10)
    (h2 : weekend_rate = 15)
    (h3 : weekday_hours = 5)
    (h4 : saturday_hours = 6)
    (h5 : sunday_hours = 4)
    (h6 : num_weekdays = 5)
    (h7 : num_saturdays = 1)
    (h8 : num_sundays = 1) :
    (weekday_rate * weekday_hours * num_weekdays) + 
    (weekend_rate * saturday_hours * num_saturdays) + 
    (weekend_rate * sunday_hours * num_sundays) = 400 := 
by
  sorry

end coffee_shop_brewed_cups_in_week_l126_126122


namespace exists_natural_numbers_solving_equation_l126_126168

theorem exists_natural_numbers_solving_equation :
  ∃ (x y z : ℕ), 28 * x + 30 * y + 31 * z = 365 := by
  sorry

end exists_natural_numbers_solving_equation_l126_126168


namespace inlet_pipe_rate_l126_126297

-- Conditions definitions
def tank_capacity : ℕ := 4320
def leak_empty_time : ℕ := 6
def full_empty_time_with_inlet : ℕ := 8

-- Question translated into a theorem
theorem inlet_pipe_rate : 
  (tank_capacity / leak_empty_time) = 720 →
  (tank_capacity / full_empty_time_with_inlet) = 540 →
  ∀ R : ℕ, 
    R - 720 = 540 →
    (R / 60) = 21 :=
by
  intros h_leak h_net R h_R
  sorry

end inlet_pipe_rate_l126_126297


namespace maximum_figures_per_shelf_l126_126052

theorem maximum_figures_per_shelf
  (figures_shelf_1 : ℕ)
  (figures_shelf_2 : ℕ)
  (figures_shelf_3 : ℕ)
  (additional_shelves : ℕ)
  (max_figures_per_shelf : ℕ)
  (total_figures : ℕ)
  (total_shelves : ℕ)
  (H1 : figures_shelf_1 = 9)
  (H2 : figures_shelf_2 = 14)
  (H3 : figures_shelf_3 = 7)
  (H4 : additional_shelves = 2)
  (H5 : max_figures_per_shelf = 11)
  (H6 : total_figures = figures_shelf_1 + figures_shelf_2 + figures_shelf_3)
  (H7 : total_shelves = 3 + additional_shelves)
  (H8 : ∃ d, d ∈ ({x : ℕ | x ∣ total_figures} ∩ {y : ℕ | y ≤ max_figures_per_shelf}))
  : ∃ d, d ∈ ({x : ℕ | x ∣ total_figures} ∩ {y : ℕ | y ≤ max_figures_per_shelf}) ∧ d = 6 := sorry

end maximum_figures_per_shelf_l126_126052


namespace arithmetic_sequence_third_eighth_term_sum_l126_126958

variable {α : Type*} [AddCommGroup α] [Module ℚ α]

def arith_sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

theorem arithmetic_sequence_third_eighth_term_sum {a : ℕ → ℚ} {S : ℕ → ℚ} 
  (h_seq: ∀ n, a n = a 1 + (n - 1) * d)
  (h_sum: arith_sequence_sum a S) 
  (h_S10 : S 10 = 4) : 
  a 3 + a 8 = 4 / 5 :=
by
  sorry

end arithmetic_sequence_third_eighth_term_sum_l126_126958


namespace number_of_boys_is_90_l126_126404

-- Define the conditions
variables (B G : ℕ)
axiom sum_condition : B + G = 150
axiom percentage_condition : G = (B / 150) * 100

-- State the theorem
theorem number_of_boys_is_90 : B = 90 :=
by
  -- We can skip the proof for now using sorry
  sorry

end number_of_boys_is_90_l126_126404


namespace real_part_of_diff_times_i_l126_126084

open Complex

def z1 : ℂ := (4 : ℂ) + (29 : ℂ) * I
def z2 : ℂ := (6 : ℂ) + (9 : ℂ) * I

theorem real_part_of_diff_times_i :
  re ((z1 - z2) * I) = -20 := 
sorry

end real_part_of_diff_times_i_l126_126084


namespace number_of_exercise_books_l126_126872

theorem number_of_exercise_books (pencils pens exercise_books : ℕ) (h_ratio : (14 * pens = 4 * pencils) ∧ (14 * exercise_books = 3 * pencils)) (h_pencils : pencils = 140) : exercise_books = 30 :=
by
  sorry

end number_of_exercise_books_l126_126872


namespace solve_for_x_l126_126894

noncomputable def equation (x : ℝ) := (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2

theorem solve_for_x (h : ∀ x, x ≠ 3) : equation (-7 / 6) :=
by
  sorry

end solve_for_x_l126_126894


namespace range_of_m_l126_126141

-- Defining the quadratic function with the given condition
def quadratic (m : ℝ) (x : ℝ) : ℝ := (m-1)*x^2 + (m-1)*x + 2

-- Stating the problem
theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, quadratic m x > 0) ↔ 1 ≤ m ∧ m < 9 :=
by
  sorry

end range_of_m_l126_126141


namespace volume_increased_by_3_l126_126822

theorem volume_increased_by_3 {l w h : ℝ}
  (h1 : l * w * h = 5000)
  (h2 : l * w + w * h + l * h = 925)
  (h3 : l + w + h = 60) :
  (l + 3) * (w + 3) * (h + 3) = 8342 := 
by
  sorry

end volume_increased_by_3_l126_126822


namespace sin_ratio_equal_one_or_neg_one_l126_126277

theorem sin_ratio_equal_one_or_neg_one
  (a b : Real)
  (h1 : Real.cos (a + b) = 1/4)
  (h2 : Real.cos (a - b) = 3/4) :
  (Real.sin a) / (Real.sin b) = 1 ∨ (Real.sin a) / (Real.sin b) = -1 :=
sorry

end sin_ratio_equal_one_or_neg_one_l126_126277


namespace solve_for_x_l126_126306

-- Define the necessary condition
def problem_statement (x : ℚ) : Prop :=
  x / 4 - x - 3 / 6 = 1

-- Prove that if the condition holds, then x = -14/9
theorem solve_for_x (x : ℚ) (h : problem_statement x) : x = -14 / 9 :=
by
  sorry

end solve_for_x_l126_126306


namespace janice_overtime_shifts_l126_126120

theorem janice_overtime_shifts (x : ℕ) (h1 : 5 * 30 + 15 * x = 195) : x = 3 :=
by
  -- leaving the proof unfinished, as asked
  sorry

end janice_overtime_shifts_l126_126120


namespace find_sum_l126_126128

theorem find_sum (I r1 r2 r3 r4 r5: ℝ) (t1 t2 t3 t4 t5 : ℝ) (P: ℝ) 
  (hI: I = 6016.75)
  (hr1: r1 = 0.06) (hr2: r2 = 0.075) (hr3: r3 = 0.08) (hr4: r4 = 0.085) (hr5: r5 = 0.09)
  (ht: ∀ i, (i = t1 ∨ i = t2 ∨ i = t3 ∨ i = t4 ∨ i = t5) → i = 1): 
  I = P * (r1 * t1 + r2 * t2 + r3 * t3 + r4 * t4 + r5 * t5) → P = 15430 :=
by
  sorry

end find_sum_l126_126128


namespace enclosed_area_abs_x_abs_3y_eq_12_l126_126374

theorem enclosed_area_abs_x_abs_3y_eq_12 : 
  let f (x y : ℝ) := |x| + |3 * y|
  ∃ (A : ℝ), ∀ (x y : ℝ), f x y = 12 → A = 96 := 
sorry

end enclosed_area_abs_x_abs_3y_eq_12_l126_126374


namespace fred_total_earnings_l126_126954

def fred_earnings (earnings_per_hour hours_worked : ℝ) : ℝ := earnings_per_hour * hours_worked

theorem fred_total_earnings :
  fred_earnings 12.5 8 = 100 := by
sorry

end fred_total_earnings_l126_126954


namespace present_value_l126_126716

theorem present_value (BD TD PV : ℝ) (hBD : BD = 42) (hTD : TD = 36)
  (h : BD = TD + (TD^2 / PV)) : PV = 216 :=
sorry

end present_value_l126_126716


namespace jean_spots_l126_126646

theorem jean_spots (total_spots upper_torso_spots back_hindspots sides_spots : ℕ)
  (h1 : upper_torso_spots = 30)
  (h2 : total_spots = 2 * upper_torso_spots)
  (h3 : back_hindspots = total_spots / 3)
  (h4 : sides_spots = total_spots - upper_torso_spots - back_hindspots) :
  sides_spots = 10 :=
by
  sorry

end jean_spots_l126_126646


namespace words_per_page_large_font_l126_126623

theorem words_per_page_large_font
    (total_words : ℕ)
    (large_font_pages : ℕ)
    (small_font_pages : ℕ)
    (small_font_words_per_page : ℕ)
    (total_pages : ℕ)
    (words_in_large_font : ℕ) :
    total_words = 48000 →
    total_pages = 21 →
    large_font_pages = 4 →
    small_font_words_per_page = 2400 →
    words_in_large_font = total_words - (small_font_pages * small_font_words_per_page) →
    small_font_pages = total_pages - large_font_pages →
    (words_in_large_font = large_font_pages * 1800) :=
by 
    sorry

end words_per_page_large_font_l126_126623


namespace convert_speed_l126_126780

-- Definitions based on the given condition
def kmh_to_mps (kmh : ℝ) : ℝ := kmh * 0.277778

-- Theorem statement
theorem convert_speed : kmh_to_mps 84 = 23.33 :=
by
  -- Proof omitted
  sorry

end convert_speed_l126_126780


namespace train_length_l126_126054

noncomputable def length_of_each_train : ℝ :=
  let speed_faster_train_km_per_hr := 46
  let speed_slower_train_km_per_hr := 36
  let relative_speed_km_per_hr := speed_faster_train_km_per_hr - speed_slower_train_km_per_hr
  let relative_speed_m_per_s := (relative_speed_km_per_hr * 1000) / 3600
  let time_s := 54
  let distance_m := relative_speed_m_per_s * time_s
  distance_m / 2

theorem train_length : length_of_each_train = 75 := by
  sorry

end train_length_l126_126054


namespace temperature_problem_product_of_possible_N_l126_126984

theorem temperature_problem (M L : ℤ) (N : ℤ) :
  (M = L + N) →
  (M - 8 = L + N - 8) →
  (L + 4 = L + 4) →
  (|((L + N - 8) - (L + 4))| = 3) →
  N = 15 ∨ N = 9 :=
by sorry

theorem product_of_possible_N :
  (∀ M L : ℤ, ∀ N : ℤ,
    (M = L + N) →
    (M - 8 = L + N - 8) →
    (L + 4 = L + 4) →
    (|((L + N - 8) - (L + 4))| = 3) →
    N = 15 ∨ N = 9) →
    15 * 9 = 135 :=
by sorry

end temperature_problem_product_of_possible_N_l126_126984


namespace line_parabola_one_intersection_not_tangent_l126_126011

theorem line_parabola_one_intersection_not_tangent {A B C D : ℝ} (h: ∀ x : ℝ, ((A * x ^ 2 + B * x + C) = D) → False) :
  ¬ ∃ x : ℝ, (A * x ^ 2 + B * x + C) = D ∧ 2 * x * A + B = 0 := sorry

end line_parabola_one_intersection_not_tangent_l126_126011


namespace percentage_profit_l126_126734

variable (total_crates : ℕ)
variable (total_cost : ℕ)
variable (lost_crates : ℕ)
variable (sell_price_per_crate : ℕ)

theorem percentage_profit (h1 : total_crates = 10) (h2 : total_cost = 160)
  (h3 : lost_crates = 2) (h4 : sell_price_per_crate = 25) :
  (8 * sell_price_per_crate - total_cost) * 100 / total_cost = 25 :=
by
  -- Definitions and steps to prove this can be added here.
  sorry

end percentage_profit_l126_126734


namespace sum_geometric_seq_l126_126885

theorem sum_geometric_seq (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 1)
  (h2 : 4 * a 2 = 4 * a 1 + a 3)
  (h3 : ∀ n, S n = a 1 * (1 - q ^ (n + 1)) / (1 - q)) :
  S 3 = 15 :=
by
  sorry

end sum_geometric_seq_l126_126885


namespace cos_pi_plus_2alpha_l126_126258

-- Define the main theorem using the given condition and the result to be proven
theorem cos_pi_plus_2alpha (α : ℝ) (h : Real.sin (π / 2 - α) = 1 / 3) : Real.cos (π + 2 * α) = 7 / 9 :=
sorry

end cos_pi_plus_2alpha_l126_126258


namespace min_expression_min_expression_achieve_l126_126367

theorem min_expression (x : ℝ) (hx : 0 < x) : 
  (x^2 + 8 * x + 64 / x^3) ≥ 28 :=
sorry

theorem min_expression_achieve (x : ℝ) (hx : x = 2): 
  (x^2 + 8 * x + 64 / x^3) = 28 :=
sorry

end min_expression_min_expression_achieve_l126_126367


namespace basketball_card_price_l126_126517

variable (x : ℝ)

def total_cost_basketball_cards (x : ℝ) : ℝ := 2 * x
def total_cost_baseball_cards : ℝ := 5 * 4
def total_spent : ℝ := 50 - 24

theorem basketball_card_price :
  total_cost_basketball_cards x + total_cost_baseball_cards = total_spent ↔ x = 3 := by
  sorry

end basketball_card_price_l126_126517


namespace intersection_with_xz_plane_l126_126118

-- Initial points on the line
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def point1 : Point3D := ⟨2, -1, 3⟩
def point2 : Point3D := ⟨6, -4, 7⟩

-- Definition of the line parametrization
def param_line (t : ℝ) : Point3D :=
  ⟨ point1.x + t * (point2.x - point1.x)
  , point1.y + t * (point2.y - point1.y)
  , point1.z + t * (point2.z - point1.z) ⟩

-- Prove that the line intersects the xz-plane at the expected point
theorem intersection_with_xz_plane :
  ∃ t : ℝ, param_line t = ⟨ 2/3, 0, 5/3 ⟩ :=
sorry

end intersection_with_xz_plane_l126_126118


namespace third_person_profit_share_l126_126681

noncomputable def investment_first : ℤ := 9000
noncomputable def investment_second : ℤ := investment_first + 2000
noncomputable def investment_third : ℤ := investment_second - 3000
noncomputable def investment_fourth : ℤ := 2 * investment_third
noncomputable def investment_fifth : ℤ := investment_fourth + 4000
noncomputable def total_investment : ℤ := investment_first + investment_second + investment_third + investment_fourth + investment_fifth

noncomputable def total_profit : ℤ := 25000
noncomputable def third_person_share : ℚ := (investment_third : ℚ) / (total_investment : ℚ) * (total_profit : ℚ)

theorem third_person_profit_share :
  third_person_share = 3076.92 := sorry

end third_person_profit_share_l126_126681


namespace no_five_coprime_two_digit_composites_l126_126616

/-- 
  Prove that there do not exist five two-digit composite 
  numbers such that each pair of them is coprime, under 
  the conditions that each composite number must be made 
  up of the primes 2, 3, 5, and 7.
-/
theorem no_five_coprime_two_digit_composites :
  ¬∃ (a b c d e : ℕ),
    10 ≤ a ∧ a < 100 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ a → p ∣ a) ∧
    10 ≤ b ∧ b < 100 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ b → p ∣ b) ∧
    10 ≤ c ∧ c < 100 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ c → p ∣ c) ∧
    10 ≤ d ∧ d < 100 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ d → p ∣ d) ∧
    10 ≤ e ∧ e < 100 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ e → p ∣ e) ∧
    ∀ (x y : ℕ), (x ∈ [a, b, c, d, e] ∧ y ∈ [a, b, c, d, e] ∧ x ≠ y) → Nat.gcd x y = 1 :=
by
  sorry

end no_five_coprime_two_digit_composites_l126_126616


namespace max_value_inequality_l126_126098

theorem max_value_inequality : 
  ∀ (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (n : ℕ) (m : ℝ),
  (∀ n, S_n n = (n * a_n 1 + (1 / 2) * n * (n - 1) * d) ∧
  (∀ n, a_n n ^ 2 + (S_n n ^ 2 / n ^ 2) >= m * (a_n 1) ^ 2)) → 
  m ≤ 1 / 5 := 
sorry

end max_value_inequality_l126_126098


namespace equivalent_proof_l126_126468

theorem equivalent_proof :
  let a := 4
  let b := Real.sqrt 17 - a
  b^2020 * (a + Real.sqrt 17)^2021 = Real.sqrt 17 + 4 :=
by
  let a := 4
  let b := Real.sqrt 17 - a
  sorry

end equivalent_proof_l126_126468


namespace lending_period_C_l126_126889

theorem lending_period_C (P_B P_C : ℝ) (R : ℝ) (T_B I_total : ℝ) (T_C_months : ℝ) :
  P_B = 5000 ∧ P_C = 3000 ∧ R = 0.10 ∧ T_B = 2 ∧ I_total = 2200 ∧ 
  T_C_months = (2 / 3) * 12 → T_C_months = 8 := by
  intros h
  sorry

end lending_period_C_l126_126889


namespace no_such_function_l126_126967

noncomputable def no_such_function_exists : Prop :=
  ¬∃ f : ℝ → ℝ, 
    (∀ y : ℝ, ∃ x : ℝ, f x = y) ∧ 
    (∀ x : ℝ, f (f x) = (x - 1) * f x + 2)

-- Here's the theorem statement to be proved
theorem no_such_function : no_such_function_exists :=
sorry

end no_such_function_l126_126967


namespace n_four_plus_n_squared_plus_one_not_prime_l126_126176

theorem n_four_plus_n_squared_plus_one_not_prime (n : ℤ) (h : n ≥ 2) : ¬ Prime (n^4 + n^2 + 1) :=
sorry

end n_four_plus_n_squared_plus_one_not_prime_l126_126176


namespace base_5_to_base_10_l126_126787

theorem base_5_to_base_10 : 
  let n : ℕ := 1 * 5^3 + 2 * 5^2 + 3 * 5^1 + 4 * 5^0
  n = 194 :=
by 
  sorry

end base_5_to_base_10_l126_126787


namespace sum_of_ages_in_three_years_l126_126087

theorem sum_of_ages_in_three_years (H : ℕ) (J : ℕ) (SumAges : ℕ) 
  (h1 : J = 3 * H) 
  (h2 : H = 15) 
  (h3 : SumAges = (H + 3) + (J + 3)) : 
  SumAges = 66 :=
by
  sorry

end sum_of_ages_in_three_years_l126_126087


namespace preimage_of_3_2_eq_l126_126654

noncomputable def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * p.2, p.1 + p.2)

theorem preimage_of_3_2_eq (x y : ℝ) :
  f (x, y) = (-3, 2) ↔ (x = 3 ∧ y = -1) ∨ (x = -1 ∧ y = 3) :=
by
  sorry

end preimage_of_3_2_eq_l126_126654


namespace statement_1_statement_2_statement_3_all_statements_correct_l126_126281

-- Define the function f and the axioms/conditions given in the problem
def f : ℕ → ℕ → ℕ := sorry

-- Conditions
axiom f_initial : f 1 1 = 1
axiom f_nat : ∀ m n : ℕ, m > 0 → n > 0 → f m n > 0
axiom f_condition_1 : ∀ m n : ℕ, m > 0 → n > 0 → f m (n + 1) = f m n + 2
axiom f_condition_2 : ∀ m : ℕ, m > 0 → f (m + 1) 1 = 2 * f m 1

-- Statements to be proved
theorem statement_1 : f 1 5 = 9 := sorry
theorem statement_2 : f 5 1 = 16 := sorry
theorem statement_3 : f 5 6 = 26 := sorry

theorem all_statements_correct : (f 1 5 = 9) ∧ (f 5 1 = 16) ∧ (f 5 6 = 26) := by
  exact ⟨statement_1, statement_2, statement_3⟩

end statement_1_statement_2_statement_3_all_statements_correct_l126_126281


namespace buses_more_than_vans_l126_126010

-- Definitions based on conditions
def vans : Float := 6.0
def buses : Float := 8.0
def people_per_van : Float := 6.0
def people_per_bus : Float := 18.0

-- Calculate total people in vans and buses
def total_people_vans : Float := vans * people_per_van
def total_people_buses : Float := buses * people_per_bus

-- Prove the difference
theorem buses_more_than_vans : total_people_buses - total_people_vans = 108.0 :=
by
  sorry

end buses_more_than_vans_l126_126010


namespace min_value_expr_l126_126963

theorem min_value_expr (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ k : ℝ, k = 6 ∧ (∃ a b c : ℝ,
                  0 < a ∧
                  0 < b ∧
                  0 < c ∧
                  (k = (a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a)) :=
sorry

end min_value_expr_l126_126963


namespace length_de_l126_126511

theorem length_de (a b c d e : ℝ) (ab bc cd de ac ae : ℝ)
  (H1 : ab = 5)
  (H2 : bc = 2 * cd)
  (H3 : ac = ab + bc)
  (H4 : ac = 11)
  (H5 : ae = ab + bc + cd + de)
  (H6 : ae = 18) :
  de = 4 :=
by {
  sorry
}

-- Explanation:
-- a, b, c, d, e are points on a straight line
-- ab, bc, cd, de, ac, ae are lengths of segments between these points
-- H1: ab = 5
-- H2: bc = 2 * cd
-- H3: ac = ab + bc
-- H4: ac = 11
-- H5: ae = ab + bc + cd + de
-- H6: ae = 18
-- Prove that de = 4

end length_de_l126_126511


namespace ratio_sharks_to_pelicans_l126_126745

-- Define the conditions given in the problem
def original_pelican_count {P : ℕ} (h : (2/3 : ℚ) * P = 20) : Prop :=
  P = 30

-- Define the final ratio we want to prove
def shark_to_pelican_ratio (sharks pelicans : ℕ) : ℚ :=
  sharks / pelicans

theorem ratio_sharks_to_pelicans
  (P : ℕ) (h : (2/3 : ℚ) * P = 20) (number_sharks : ℕ) (number_pelicans : ℕ)
  (H_sharks : number_sharks = 60) (H_pelicans : number_pelicans = P)
  (H_original_pelicans : original_pelican_count h) :
  shark_to_pelican_ratio number_sharks number_pelicans = 2 :=
by
  -- proof skipped
  sorry

end ratio_sharks_to_pelicans_l126_126745


namespace false_statement_is_D_l126_126785

def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

def is_scalene_triangle (a b c : ℝ) : Prop :=
  (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ a)

def is_right_isosceles_triangle (a b c : ℝ) : Prop :=
  is_right_triangle a b c ∧ is_isosceles_triangle a b c

-- Statements derived from conditions
def statement_A : Prop := ∀ (a b c : ℝ), is_isosceles_triangle a b c → a = b ∨ b = c ∨ c = a
def statement_B : Prop := ∀ (a b c : ℝ), is_right_triangle a b c → a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2
def statement_C : Prop := ∀ (a b c : ℝ), is_scalene_triangle a b c → a ≠ b ∧ b ≠ c ∧ c ≠ a
def statement_D : Prop := ∀ (a b c : ℝ), is_right_triangle a b c → is_isosceles_triangle a b c
def statement_E : Prop := ∀ (a b c : ℝ), is_right_isosceles_triangle a b c → ∃ (θ : ℝ), θ ≠ 90 ∧ θ = 45

-- Main theorem to be proved
theorem false_statement_is_D : statement_D = false :=
by
  sorry

end false_statement_is_D_l126_126785


namespace domino_covering_l126_126791

theorem domino_covering (m n : ℕ) (m_eq : (m, n) ∈ [(5, 5), (4, 6), (3, 7), (5, 6), (3, 8)]) :
  (m * n % 2 = 1) ↔ (m = 5 ∧ n = 5) ∨ (m = 3 ∧ n = 7) :=
by
  sorry

end domino_covering_l126_126791


namespace tree_age_difference_l126_126586

theorem tree_age_difference
  (groups_rings : ℕ)
  (rings_per_group : ℕ)
  (first_tree_groups : ℕ)
  (second_tree_groups : ℕ)
  (rings_per_year : ℕ)
  (h_rg : rings_per_group = 6)
  (h_ftg : first_tree_groups = 70)
  (h_stg : second_tree_groups = 40)
  (h_rpy : rings_per_year = 1) :
  ((first_tree_groups * rings_per_group) - (second_tree_groups * rings_per_group)) = 180 := 
by
  sorry

end tree_age_difference_l126_126586


namespace range_of_a_l126_126605

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (h₁ : ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ ≥ f x₂)
                    (h₂ : -2 ≤ a + 1 ∧ a + 1 ≤ 4)
                    (h₃ : -2 ≤ 2 * a ∧ 2 * a ≤ 4)
                    (h₄ : f (a + 1) > f (2 * a)) : 1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l126_126605


namespace smallest_period_sum_l126_126431

noncomputable def smallest_positive_period (f : ℝ → ℝ) (g : ℝ → ℝ): ℝ → ℝ :=
λ x => f x + g x

theorem smallest_period_sum
  (f g : ℝ → ℝ)
  (m n : ℕ)
  (hf : ∀ x, f (x + m) = f x)
  (hg : ∀ x, g (x + n) = g x)
  (hm : m > 1)
  (hn : n > 1)
  (hgcd : Nat.gcd m n = 1)
  : ∃ T, T > 0 ∧ (∀ x, smallest_positive_period f g (x + T) = smallest_positive_period f g x) ∧ T = m * n := by
  sorry

end smallest_period_sum_l126_126431


namespace find_subtracted_value_l126_126722

theorem find_subtracted_value (N V : ℤ) (hN : N = 12) (h : 4 * N - 3 = 9 * (N - V)) : V = 7 := 
by
  sorry

end find_subtracted_value_l126_126722


namespace anna_stamp_count_correct_l126_126485

-- Defining the initial counts of stamps
def anna_initial := 37
def alison_initial := 28
def jeff_initial := 31

-- Defining the operations
def alison_gives_half_to_anna := alison_initial / 2
def anna_after_receiving_from_alison := anna_initial + alison_gives_half_to_anna
def anna_final := anna_after_receiving_from_alison - 2 + 1

-- Formalizing the proof problem
theorem anna_stamp_count_correct : anna_final = 50 := by
  -- proof omitted
  sorry

end anna_stamp_count_correct_l126_126485


namespace investment_calculation_l126_126210

theorem investment_calculation
  (face_value : ℝ)
  (market_price : ℝ)
  (rate_of_dividend : ℝ)
  (annual_income : ℝ)
  (h1 : face_value = 10)
  (h2 : market_price = 8.25)
  (h3 : rate_of_dividend = 12)
  (h4 : annual_income = 648) :
  ∃ investment : ℝ, investment = 4455 :=
by
  sorry

end investment_calculation_l126_126210


namespace remainder_six_n_mod_four_l126_126242

theorem remainder_six_n_mod_four (n : ℤ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := 
by sorry

end remainder_six_n_mod_four_l126_126242


namespace average_side_lengths_of_squares_l126_126288

theorem average_side_lengths_of_squares:
  let a₁ := 25
  let a₂ := 36
  let a₃ := 64

  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃

  (s₁ + s₂ + s₃) / 3 = 19 / 3 :=
by 
  sorry

end average_side_lengths_of_squares_l126_126288


namespace probability_top_card_king_l126_126636

theorem probability_top_card_king :
  let total_cards := 52
  let total_kings := 4
  let probability := total_kings / total_cards
  probability = 1 / 13 :=
by
  -- sorry to skip the proof
  sorry

end probability_top_card_king_l126_126636


namespace gcd_315_2016_l126_126732

def a : ℕ := 315
def b : ℕ := 2016

theorem gcd_315_2016 : Nat.gcd a b = 63 := 
by 
  sorry

end gcd_315_2016_l126_126732


namespace edric_monthly_salary_l126_126228

theorem edric_monthly_salary 
  (hours_per_day : ℝ)
  (days_per_week : ℝ)
  (weeks_per_month : ℝ)
  (hourly_rate : ℝ) :
  hours_per_day = 8 ∧ days_per_week = 6 ∧ weeks_per_month = 4.33 ∧ hourly_rate = 3 →
  (hours_per_day * days_per_week * weeks_per_month * hourly_rate) = 623.52 :=
by
  intros h
  sorry

end edric_monthly_salary_l126_126228


namespace simplify_expression_l126_126325

theorem simplify_expression (x : ℝ) :
  x * (4 * x^3 - 3 * x + 2) - 6 * (2 * x^3 + x^2 - 3 * x + 4) = 4 * x^4 - 12 * x^3 - 9 * x^2 + 20 * x - 24 :=
by sorry

end simplify_expression_l126_126325


namespace greatest_prime_factor_3_8_plus_6_7_l126_126317

theorem greatest_prime_factor_3_8_plus_6_7 : ∃ p, p = 131 ∧ Prime p ∧ ∀ q, Prime q ∧ q ∣ (3^8 + 6^7) → q ≤ 131 :=
by
  sorry


end greatest_prime_factor_3_8_plus_6_7_l126_126317


namespace perp_line_parallel_plane_perp_line_l126_126816

variable {Line : Type} {Plane : Type}
variable (a b : Line) (α β : Plane)
variable (parallel : Line → Plane → Prop) (perpendicular : Line → Plane → Prop) (parallel_lines : Line → Line → Prop)

-- Conditions
variable (non_coincident_lines : ¬(a = b))
variable (non_coincident_planes : ¬(α = β))
variable (a_perp_α : perpendicular a α)
variable (b_par_α : parallel b α)

-- Prove
theorem perp_line_parallel_plane_perp_line :
  perpendicular a α ∧ parallel b α → parallel_lines a b :=
sorry

end perp_line_parallel_plane_perp_line_l126_126816


namespace repeating_decimal_to_fraction_l126_126674

theorem repeating_decimal_to_fraction : 
  (∃ (x : ℚ), x = 7 + 3 / 9) → 7 + 3 / 9 = 22 / 3 :=
by
  intros h
  sorry

end repeating_decimal_to_fraction_l126_126674


namespace train_speed_is_72_kmph_l126_126189

-- Define the given conditions in Lean
def crossesMan (L V : ℝ) : Prop := L = 19 * V
def crossesPlatform (L V : ℝ) : Prop := L + 220 = 30 * V

-- The main theorem which states that the speed of the train is 72 km/h under given conditions
theorem train_speed_is_72_kmph (L V : ℝ) (h1 : crossesMan L V) (h2 : crossesPlatform L V) :
  (V * 3.6) = 72 := by
  -- We will provide a full proof here later
  sorry

end train_speed_is_72_kmph_l126_126189


namespace two_digit_number_representation_l126_126286

theorem two_digit_number_representation (m n : ℕ) (hm : m < 10) (hn : n < 10) : 10 * n + m = m + 10 * n :=
by sorry

end two_digit_number_representation_l126_126286


namespace matrix_diagonal_neg5_l126_126144

variable (M : Matrix (Fin 3) (Fin 3) ℝ)

theorem matrix_diagonal_neg5 
    (h : ∀ v : Fin 3 → ℝ, (M.mulVec v) = -5 • v) : 
    M = !![-5, 0, 0; 0, -5, 0; 0, 0, -5] :=
by
  sorry

end matrix_diagonal_neg5_l126_126144


namespace infinite_geometric_series_sum_l126_126784

theorem infinite_geometric_series_sum (p q : ℝ)
  (h : (∑' n : ℕ, p / q ^ (n + 1)) = 5) :
  (∑' n : ℕ, p / (p^2 + q) ^ (n + 1)) = 5 * (q - 1) / (25 * q^2 - 50 * q + 26) :=
sorry

end infinite_geometric_series_sum_l126_126784


namespace common_ratio_of_geometric_progression_l126_126916

theorem common_ratio_of_geometric_progression (a1 q : ℝ) (S3 : ℝ) (a2 : ℝ)
  (h1 : S3 = a1 * (1 + q + q^2))
  (h2 : a2 = a1 * q)
  (h3 : a2 + S3 = 0) :
  q = -1 := 
  sorry

end common_ratio_of_geometric_progression_l126_126916


namespace absolute_value_condition_l126_126169

theorem absolute_value_condition (x : ℝ) (h : |x| = 32) : x = 32 ∨ x = -32 :=
sorry

end absolute_value_condition_l126_126169


namespace expand_product_l126_126106

theorem expand_product (x : ℝ) : (3 * x + 4) * (2 * x + 6) = 6 * x^2 + 26 * x + 24 := 
by 
  sorry

end expand_product_l126_126106


namespace cleaning_time_with_doubled_an_speed_l126_126778

def A := 1 / 12  -- Anne's cleaning rate (houses per hour)
def B := 1 / 6   -- Bruce's cleaning rate (houses per hour)

theorem cleaning_time_with_doubled_an_speed :
  (A * 2 + B) * 3 = 1 := by
  -- Proof omitted
  sorry

end cleaning_time_with_doubled_an_speed_l126_126778


namespace correct_answer_l126_126942

def A : Set ℝ := { x | x^2 + 2 * x - 3 > 0 }
def B : Set ℝ := { -1, 0, 1, 2 }

theorem correct_answer : A ∩ B = { 2 } :=
  sorry

end correct_answer_l126_126942


namespace cubic_meter_to_cubic_centimeters_l126_126088

theorem cubic_meter_to_cubic_centimeters : 
  (1 : ℝ)^3 = (100 : ℝ)^3 * (1 : ℝ)^0 := 
by 
  sorry

end cubic_meter_to_cubic_centimeters_l126_126088


namespace unique_solution_of_system_l126_126136

theorem unique_solution_of_system :
  ∃! (x y z : ℝ), x + y = 2 ∧ xy - z^2 = 1 ∧ x = 1 ∧ y = 1 ∧ z = 0 := by
  sorry

end unique_solution_of_system_l126_126136


namespace greatest_two_digit_multiple_of_17_l126_126142

theorem greatest_two_digit_multiple_of_17 : ∃ m : ℕ, (10 ≤ m) ∧ (m ≤ 99) ∧ (17 ∣ m) ∧ (∀ n : ℕ, (10 ≤ n) ∧ (n ≤ 99) ∧ (17 ∣ n) → n ≤ m) ∧ m = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l126_126142


namespace number_count_two_digit_property_l126_126970

open Nat

theorem number_count_two_digit_property : 
  (∃ (n : Finset ℕ), (∀ (x : ℕ), x ∈ n ↔ ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 11 * a + 2 * b ≡ 7 [MOD 10] ∧ x = 10 * a + b) ∧ n.card = 5) :=
by
  sorry

end number_count_two_digit_property_l126_126970


namespace no_person_has_fewer_than_6_cards_l126_126794

-- Definition of the problem and conditions
def cards := 60
def people := 10
def cards_per_person := cards / people

-- Lean statement of the proof problem
theorem no_person_has_fewer_than_6_cards
  (cards_dealt : cards = 60)
  (people_count : people = 10)
  (even_distribution : cards % people = 0) :
  ∀ person, person < people → cards_per_person = 6 ∧ person < people → person = 0 := 
by 
  sorry

end no_person_has_fewer_than_6_cards_l126_126794


namespace Edmund_earns_64_dollars_l126_126009

-- Conditions
def chores_per_week : Nat := 12
def pay_per_extra_chore : Nat := 2
def chores_per_day : Nat := 4
def weeks : Nat := 2
def days_per_week : Nat := 7

-- Goal
theorem Edmund_earns_64_dollars :
  let total_chores_without_extra := chores_per_week * weeks
  let total_chores_with_extra := chores_per_day * (days_per_week * weeks)
  let extra_chores := total_chores_with_extra - total_chores_without_extra
  let earnings := pay_per_extra_chore * extra_chores
  earnings = 64 :=
by
  sorry

end Edmund_earns_64_dollars_l126_126009


namespace max_grapes_discarded_l126_126955

theorem max_grapes_discarded (n : ℕ) : 
  ∃ k : ℕ, k ∣ n → 7 * k + 6 = n → ∃ m, m = 6 := by
  sorry

end max_grapes_discarded_l126_126955


namespace batsman_average_after_17th_inning_l126_126131

theorem batsman_average_after_17th_inning
  (A : ℝ) -- average before 17th inning
  (h1 : (16 * A + 50) / 17 = A + 2) : 
  (A + 2) = 18 :=
by
  -- Proof goes here
  sorry

end batsman_average_after_17th_inning_l126_126131


namespace work_completion_days_l126_126523

theorem work_completion_days (d : ℝ) : (1 / 15 + 1 / d = 1 / 11.25) → d = 45 := sorry

end work_completion_days_l126_126523


namespace P_iff_Q_l126_126287

def P (x : ℝ) := x > 1 ∨ x < -1
def Q (x : ℝ) := |x + 1| + |x - 1| > 2

theorem P_iff_Q : ∀ x, P x ↔ Q x :=
by
  intros x
  sorry

end P_iff_Q_l126_126287


namespace sqrt_meaningful_range_l126_126341

theorem sqrt_meaningful_range (x : ℝ) (h : 0 ≤ x - 1) : 1 ≤ x :=
by
sorry

end sqrt_meaningful_range_l126_126341


namespace cos_neg_570_eq_neg_sqrt3_div_2_l126_126875

theorem cos_neg_570_eq_neg_sqrt3_div_2 :
  Real.cos (-(570 : ℝ) * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_neg_570_eq_neg_sqrt3_div_2_l126_126875


namespace value_of_expression_l126_126946

theorem value_of_expression :
  (10^2 - 10) / 9 = 10 :=
by
  sorry

end value_of_expression_l126_126946


namespace problem1_problem2_l126_126703

-- Problem 1
theorem problem1 : (1/4 / 1/5) - 1/4 = 1 := 
by 
  sorry

-- Problem 2
theorem problem2 : ∃ x : ℚ, x + 1/2 * x = 12/5 ∧ x = 4 :=
by
  sorry

end problem1_problem2_l126_126703


namespace question1_question2_l126_126428

-- Define the sets A and B as given in the problem
def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}
def B : Set ℝ := {x | x < -2 ∨ x > 5}

-- Lean statement for (1)
theorem question1 (m : ℝ) : 
  (A m ⊆ B) ↔ (m < 2 ∨ m > 4) :=
by
  sorry

-- Lean statement for (2)
theorem question2 (m : ℝ) : 
  (A m ∩ B = ∅) ↔ (m ≤ 3) :=
by
  sorry

end question1_question2_l126_126428


namespace min_club_members_l126_126302

theorem min_club_members (n : ℕ) :
  (∀ k : ℕ, k = 8 ∨ k = 9 ∨ k = 11 → n % k = 0) ∧ (n ≥ 300) → n = 792 :=
sorry

end min_club_members_l126_126302


namespace simple_interest_time_period_l126_126181

variable (SI P R T : ℝ)

theorem simple_interest_time_period (h₁ : SI = 4016.25) (h₂ : P = 8925) (h₃ : R = 9) :
  (P * R * T) / 100 = SI ↔ T = 5 := by
  sorry

end simple_interest_time_period_l126_126181


namespace time_spent_on_aerobics_l126_126069

theorem time_spent_on_aerobics (A W : ℝ) 
  (h1 : A + W = 250) 
  (h2 : A / W = 3 / 2) : 
  A = 150 := 
sorry

end time_spent_on_aerobics_l126_126069


namespace tim_income_percentage_less_l126_126854

theorem tim_income_percentage_less (M T J : ℝ)
  (h₁ : M = 1.60 * T)
  (h₂ : M = 0.96 * J) :
  100 - (T / J) * 100 = 40 :=
by sorry

end tim_income_percentage_less_l126_126854


namespace filling_time_calculation_l126_126164

namespace TankerFilling

-- Define the filling rates
def fill_rate_A : ℚ := 1 / 60
def fill_rate_B : ℚ := 1 / 40
def combined_fill_rate : ℚ := fill_rate_A + fill_rate_B

-- Define the time variable
variable (T : ℚ)

-- State the theorem to be proved
theorem filling_time_calculation
  (h_fill_rate_A : fill_rate_A = 1 / 60)
  (h_fill_rate_B : fill_rate_B = 1 / 40)
  (h_combined_fill_rate : combined_fill_rate = 1 / 24) :
  (fill_rate_B * (T / 2) + combined_fill_rate * (T / 2)) = 1 → T = 30 :=
by
  intros h
  -- Proof will go here
  sorry

end TankerFilling

end filling_time_calculation_l126_126164


namespace triangle_is_right_angled_l126_126655

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point :=
  { x := Q.x - P.x, y := Q.y - P.y }

def dot_product (u v : Point) : ℝ :=
  u.x * v.x + u.y * v.y

def is_right_angle_triangle (A B C : Point) : Prop :=
  let AB := vector A B
  let BC := vector B C
  dot_product AB BC = 0

theorem triangle_is_right_angled :
  let A := { x := 2, y := 5 }
  let B := { x := 5, y := 2 }
  let C := { x := 10, y := 7 }
  is_right_angle_triangle A B C :=
by
  sorry

end triangle_is_right_angled_l126_126655


namespace additional_employees_hired_l126_126174

-- Conditions
def initial_employees : ℕ := 500
def hourly_wage : ℕ := 12
def daily_hours : ℕ := 10
def weekly_days : ℕ := 5
def weekly_hours := daily_hours * weekly_days
def monthly_weeks : ℕ := 4
def monthly_hours_per_employee := weekly_hours * monthly_weeks
def wage_per_employee_per_month := monthly_hours_per_employee * hourly_wage

-- Given new payroll
def new_monthly_payroll : ℕ := 1680000

-- Calculate the initial payroll
def initial_monthly_payroll := initial_employees * wage_per_employee_per_month

-- Statement of the proof problem
theorem additional_employees_hired :
  (new_monthly_payroll - initial_monthly_payroll) / wage_per_employee_per_month = 200 :=
by
  sorry

end additional_employees_hired_l126_126174


namespace sum_of_factors_of_30_multiplied_by_2_equals_144_l126_126777

-- We define the factors of 30
def factors_of_30 : List ℕ := [1, 2, 3, 5, 6, 10, 15, 30]

-- We define the function to multiply each factor by 2 and sum them
def sum_factors_multiplied_by_2 (factors : List ℕ) : ℕ :=
  factors.foldl (λ acc x => acc + 2 * x) 0

-- The final statement to be proven
theorem sum_of_factors_of_30_multiplied_by_2_equals_144 :
  sum_factors_multiplied_by_2 factors_of_30 = 144 :=
by sorry

end sum_of_factors_of_30_multiplied_by_2_equals_144_l126_126777


namespace glycerin_percentage_proof_l126_126622

-- Conditions given in problem
def original_percentage : ℝ := 0.90
def original_volume : ℝ := 4
def added_volume : ℝ := 0.8

-- Total glycerin in original solution
def glycerin_amount : ℝ := original_percentage * original_volume

-- Total volume after adding water
def new_volume : ℝ := original_volume + added_volume

-- Desired percentage proof statement
theorem glycerin_percentage_proof : 
  (glycerin_amount / new_volume) * 100 = 75 := 
by
  sorry

end glycerin_percentage_proof_l126_126622


namespace evaluate_expression_l126_126867

noncomputable def lg (x : ℝ) : ℝ := Real.log x

theorem evaluate_expression :
  lg 5 * lg 50 - lg 2 * lg 20 - lg 625 = -2 :=
by
  sorry

end evaluate_expression_l126_126867


namespace four_digit_integers_with_repeated_digits_l126_126568

noncomputable def count_four_digit_integers_with_repeated_digits : ℕ := sorry

theorem four_digit_integers_with_repeated_digits : 
  count_four_digit_integers_with_repeated_digits = 1984 :=
sorry

end four_digit_integers_with_repeated_digits_l126_126568


namespace melissa_bananas_l126_126588

theorem melissa_bananas (a b : ℕ) (h1 : a = 88) (h2 : b = 4) : a - b = 84 :=
by
  sorry

end melissa_bananas_l126_126588


namespace ratio_p_r_l126_126021

     variables (p q r s : ℚ)

     -- Given conditions
     def ratio_p_q := p / q = 3 / 5
     def ratio_r_s := r / s = 5 / 4
     def ratio_s_q := s / q = 1 / 3

     -- Statement to be proved
     theorem ratio_p_r 
       (h1 : ratio_p_q p q)
       (h2 : ratio_r_s r s) 
       (h3 : ratio_s_q s q) : 
       p / r = 36 / 25 :=
     sorry
     
end ratio_p_r_l126_126021


namespace exp_add_l126_126957

theorem exp_add (z w : Complex) : Complex.exp z * Complex.exp w = Complex.exp (z + w) := 
by 
  sorry

end exp_add_l126_126957


namespace brittany_age_when_returning_l126_126263

def rebecca_age : ℕ := 25
def age_difference : ℕ := 3
def vacation_duration : ℕ := 4

theorem brittany_age_when_returning : (rebecca_age + age_difference + vacation_duration) = 32 := by
  sorry

end brittany_age_when_returning_l126_126263


namespace ptolemys_theorem_l126_126057

-- Definition of the variables describing the lengths of the sides and diagonals
variables {a b c d m n : ℝ}

-- We declare that they belong to a cyclic quadrilateral
def cyclic_quadrilateral (a b c d m n : ℝ) : Prop :=
∃ (A B C D : ℝ), 
  A + C = 180 ∧ 
  B + D = 180 ∧ 
  m = (A * C) ∧ 
  n = (B * D) ∧ 
  a = (A * B) ∧ 
  b = (B * C) ∧ 
  c = (C * D) ∧ 
  d = (D * A)

-- The theorem statement in Lean form
theorem ptolemys_theorem (h : cyclic_quadrilateral a b c d m n) : m * n = a * c + b * d :=
sorry

end ptolemys_theorem_l126_126057


namespace class_size_l126_126726

theorem class_size (n : ℕ) (h₁ : 60 - n > 0) (h₂ : (60 - n) / 2 = n) : n = 20 :=
by
  sorry

end class_size_l126_126726


namespace intercept_form_l126_126968

theorem intercept_form (x y : ℝ) : 2 * x - 3 * y - 4 = 0 ↔ x / 2 + y / (-4/3) = 1 := sorry

end intercept_form_l126_126968


namespace sum_arithmetic_sequence_l126_126495

theorem sum_arithmetic_sequence (m : ℕ) (S : ℕ → ℕ) 
  (h1 : S m = 30) 
  (h2 : S (3 * m) = 90) : 
  S (2 * m) = 60 := 
sorry

end sum_arithmetic_sequence_l126_126495


namespace total_plates_used_l126_126760

-- Definitions from the conditions
def number_of_people := 6
def meals_per_day_per_person := 3
def plates_per_meal_per_person := 2
def number_of_days := 4

-- Statement of the theorem
theorem total_plates_used : number_of_people * meals_per_day_per_person * plates_per_meal_per_person * number_of_days = 144 := 
by
  sorry

end total_plates_used_l126_126760


namespace inverse_proportion_relationship_l126_126119

theorem inverse_proportion_relationship (k : ℝ) (y1 y2 y3 : ℝ) :
  y1 = (k^2 + 1) / -1 →
  y2 = (k^2 + 1) / 1 →
  y3 = (k^2 + 1) / 2 →
  y1 < y3 ∧ y3 < y2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end inverse_proportion_relationship_l126_126119


namespace average_age_increase_l126_126549

theorem average_age_increase (n : ℕ) (m : ℕ) (a b : ℝ) (h1 : n = 19) (h2 : m = 20) (h3 : a = 20) (h4 : b = 40) :
  ((n * a + b) / (n + 1)) - a = 1 :=
by
  -- Proof omitted
  sorry

end average_age_increase_l126_126549


namespace mapping_sum_l126_126081

theorem mapping_sum (f : ℝ × ℝ → ℝ × ℝ) (a b : ℝ)
(h1 : ∀ x y, f (x, y) = (x, x + y))
(h2 : (a, b) = f (1, 3)) :
  a + b = 5 :=
sorry

end mapping_sum_l126_126081


namespace second_student_catches_up_l126_126666

open Nat

-- Definitions for the problems
def distance_first_student (n : ℕ) : ℕ := 7 * n
def distance_second_student (n : ℕ) : ℕ := n * (n + 1) / 2

-- Theorem statement indicating the second student catches up with the first at n = 13
theorem second_student_catches_up : ∃ n, (distance_first_student n = distance_second_student n) ∧ n = 13 := 
by 
  sorry

end second_student_catches_up_l126_126666


namespace find_digit_A_l126_126467

def sum_of_digits_divisible_by_3 (A : ℕ) : Prop :=
  (2 + A + 3) % 3 = 0

theorem find_digit_A (A : ℕ) (hA : sum_of_digits_divisible_by_3 A) : A = 1 ∨ A = 4 :=
  sorry

end find_digit_A_l126_126467


namespace distance_between_planes_is_zero_l126_126503

def plane1 (x y z : ℝ) : Prop := x - 2 * y + 2 * z = 9
def plane2 (x y z : ℝ) : Prop := 2 * x - 4 * y + 4 * z = 18

theorem distance_between_planes_is_zero :
  (∀ x y z : ℝ, plane1 x y z ↔ plane2 x y z) → 0 = 0 :=
by
  sorry

end distance_between_planes_is_zero_l126_126503


namespace ezekiel_shoes_l126_126150

theorem ezekiel_shoes (pairs : ℕ) (shoes_per_pair : ℕ) (bought_pairs : pairs = 3) (pair_contains : shoes_per_pair = 2) : pairs * shoes_per_pair = 6 := by
  sorry

end ezekiel_shoes_l126_126150


namespace savings_fraction_l126_126529

variable (P : ℝ) 
variable (S : ℝ)
variable (E : ℝ)
variable (T : ℝ)

theorem savings_fraction :
  (12 * P * S) = 2 * P * (1 - S) → S = 1 / 7 :=
by
  intro h
  sorry

end savings_fraction_l126_126529


namespace david_savings_l126_126754

def lawn_rate_monday : ℕ := 14
def lawn_rate_wednesday : ℕ := 18
def lawn_rate_friday : ℕ := 20
def hours_per_day : ℕ := 2
def weekly_earnings : ℕ := (lawn_rate_monday * hours_per_day) + (lawn_rate_wednesday * hours_per_day) + (lawn_rate_friday * hours_per_day)

def tax_rate : ℚ := 0.10
def tax_paid (earnings : ℚ) : ℚ := earnings * tax_rate

def shoe_price : ℚ := 75
def discount : ℚ := 0.15
def discounted_shoe_price : ℚ := shoe_price * (1 - discount)

def money_remaining (earnings : ℚ) (tax : ℚ) (shoes : ℚ) : ℚ := earnings - tax - shoes

def gift_rate : ℚ := 1 / 3
def money_given_to_mom (remaining : ℚ) : ℚ := remaining * gift_rate

def final_savings (remaining : ℚ) (gift : ℚ) : ℚ := remaining - gift

theorem david_savings : 
  final_savings (money_remaining weekly_earnings (tax_paid weekly_earnings) discounted_shoe_price) 
                (money_given_to_mom (money_remaining weekly_earnings (tax_paid weekly_earnings) discounted_shoe_price)) 
  = 19.90 :=
by
  -- The proof goes here
  sorry

end david_savings_l126_126754


namespace train_speed_l126_126402

theorem train_speed (train_length platform_length total_time : ℕ) 
  (h_train_length : train_length = 150) 
  (h_platform_length : platform_length = 250) 
  (h_total_time : total_time = 8) : 
  (train_length + platform_length) / total_time = 50 := 
by
  -- Proof goes here
  -- Given: train_length = 150, platform_length = 250, total_time = 8
  -- We need to prove: (train_length + platform_length) / total_time = 50
  -- So we calculate
  --  (150 + 250)/8 = 400/8 = 50
  sorry

end train_speed_l126_126402


namespace min_value_n_constant_term_l126_126177

-- Define the problem statement
theorem min_value_n_constant_term (n r : ℕ) (h : 2 * n = 5 * r) : n = 5 :=
by sorry

end min_value_n_constant_term_l126_126177


namespace certain_number_is_65_l126_126923

-- Define the conditions
variables (N : ℕ)
axiom condition1 : N < 81
axiom condition2 : ∀ k : ℕ, k ≤ 15 → N + k < 81
axiom last_consecutive : N + 15 = 80

-- Prove the theorem
theorem certain_number_is_65 (h1 : N < 81) (h2 : ∀ k : ℕ, k ≤ 15 → N + k < 81) (h3 : N + 15 = 80) : N = 65 :=
sorry

end certain_number_is_65_l126_126923


namespace peter_invested_for_3_years_l126_126014

-- Definitions of parameters
def P : ℝ := 650
def APeter : ℝ := 815
def ADavid : ℝ := 870
def tDavid : ℝ := 4

-- Simple interest formula for Peter
def simple_interest_peter (r : ℝ) (t : ℝ) : Prop :=
  APeter = P + P * r * t

-- Simple interest formula for David
def simple_interest_david (r : ℝ) : Prop :=
  ADavid = P + P * r * tDavid

-- The main theorem to find out how many years Peter invested his money
theorem peter_invested_for_3_years : ∃ t : ℝ, (∃ r : ℝ, simple_interest_peter r t ∧ simple_interest_david r) ∧ t = 3 :=
by
  sorry

end peter_invested_for_3_years_l126_126014


namespace inequality_range_l126_126753

theorem inequality_range (a : ℝ) : (-1 < a ∧ a ≤ 0) → ∀ x : ℝ, a * x^2 + 2 * a * x - (a + 2) < 0 :=
by
  intro ha
  sorry

end inequality_range_l126_126753


namespace eval_six_times_f_l126_126818

def f (x : Int) : Int :=
  if x % 2 == 0 then
    x / 2
  else
    5 * x + 1

theorem eval_six_times_f : f (f (f (f (f (f 7))))) = 116 := 
by
  -- Skipping proof body (since it's not required)
  sorry

end eval_six_times_f_l126_126818


namespace total_animals_l126_126248

-- Definitions of the initial conditions
def initial_beavers := 20
def initial_chipmunks := 40
def doubled_beavers := 2 * initial_beavers
def decreased_chipmunks := initial_chipmunks - 10

theorem total_animals (initial_beavers initial_chipmunks doubled_beavers decreased_chipmunks : ℕ)
    (h1 : doubled_beavers = 2 * initial_beavers)
    (h2 : decreased_chipmunks = initial_chipmunks - 10) :
    (initial_beavers + initial_chipmunks) + (doubled_beavers + decreased_chipmunks) = 130 :=
by 
  sorry

end total_animals_l126_126248


namespace min_binary_questions_to_determine_number_l126_126352

theorem min_binary_questions_to_determine_number (x : ℕ) (h : 10 ≤ x ∧ x ≤ 19) : 
  ∃ (n : ℕ), n = 3 := 
sorry

end min_binary_questions_to_determine_number_l126_126352


namespace coins_in_bag_l126_126251

theorem coins_in_bag (x : ℕ) (h : x + x / 2 + x / 4 = 105) : x = 60 :=
by
  sorry

end coins_in_bag_l126_126251


namespace students_in_two_courses_l126_126652

def total_students := 400
def num_math_modelling := 169
def num_chinese_literacy := 158
def num_international_perspective := 145
def num_all_three := 30
def num_none := 20

theorem students_in_two_courses : 
  ∃ x y z, 
    (num_math_modelling + num_chinese_literacy + num_international_perspective - (x + y + z) + num_all_three + num_none = total_students) ∧
    (x + y + z = 32) := 
  by
  sorry

end students_in_two_courses_l126_126652


namespace painting_methods_correct_l126_126740

def num_painting_methods : Nat := 72

theorem painting_methods_correct :
  let vertices : Fin 4 := by sorry -- Ensures there are four vertices
  let edges : Fin 4 := by sorry -- Ensures each edge has different colored endpoints
  let available_colors : Fin 4 := by sorry -- Ensures there are four available colors
  num_painting_methods = 72 :=
sorry

end painting_methods_correct_l126_126740


namespace solve_equation_l126_126563

theorem solve_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -6) :
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) ↔ x = -4 ∨ x = -2 :=
by
  sorry

end solve_equation_l126_126563


namespace sam_dads_dimes_l126_126203

theorem sam_dads_dimes (original_dimes new_dimes given_dimes : ℕ) 
  (h1 : original_dimes = 9)
  (h2 : new_dimes = 16)
  (h3 : new_dimes = original_dimes + given_dimes) : 
  given_dimes = 7 := 
by 
  sorry

end sam_dads_dimes_l126_126203


namespace smallest_z_in_arithmetic_and_geometric_progression_l126_126442

theorem smallest_z_in_arithmetic_and_geometric_progression :
  ∃ x y z : ℤ, x < y ∧ y < z ∧ (2 * y = x + z) ∧ (z^2 = x * y) ∧ z = -2 :=
by
  sorry

end smallest_z_in_arithmetic_and_geometric_progression_l126_126442


namespace num_real_a_with_int_roots_l126_126037

theorem num_real_a_with_int_roots :
  (∃ n : ℕ, n = 15 ∧ ∀ a : ℝ, (∃ r s : ℤ, (r + s = -a) ∧ (r * s = 12 * a) → true)) :=
sorry

end num_real_a_with_int_roots_l126_126037


namespace angle_value_is_140_l126_126727

-- Definitions of conditions
def angle_on_straight_line_degrees (x y : ℝ) : Prop := x + y = 180

-- Main statement in Lean
theorem angle_value_is_140 (x : ℝ) (h₁ : angle_on_straight_line_degrees 40 x) : x = 140 :=
by
  -- Proof is omitted (not required as per instructions)
  sorry

end angle_value_is_140_l126_126727


namespace marcy_total_people_served_l126_126046

noncomputable def total_people_served_lip_gloss
  (tubs_lip_gloss : ℕ) (tubes_per_tub_lip_gloss : ℕ) (people_per_tube_lip_gloss : ℕ) : ℕ :=
  tubs_lip_gloss * tubes_per_tub_lip_gloss * people_per_tube_lip_gloss

noncomputable def total_people_served_mascara
  (tubs_mascara : ℕ) (tubes_per_tub_mascara : ℕ) (people_per_tube_mascara : ℕ) : ℕ :=
  tubs_mascara * tubes_per_tub_mascara * people_per_tube_mascara

theorem marcy_total_people_served :
  ∀ (tubs_lip_gloss tubs_mascara : ℕ) 
    (tubes_per_tub_lip_gloss tubes_per_tub_mascara 
     people_per_tube_lip_gloss people_per_tube_mascara : ℕ),
    tubs_lip_gloss = 6 → 
    tubes_per_tub_lip_gloss = 2 → 
    people_per_tube_lip_gloss = 3 → 
    tubs_mascara = 4 → 
    tubes_per_tub_mascara = 3 → 
    people_per_tube_mascara = 5 → 
    total_people_served_lip_gloss tubs_lip_gloss 
                                 tubes_per_tub_lip_gloss 
                                 people_per_tube_lip_gloss = 36 :=
by
  intros tubs_lip_gloss tubs_mascara 
         tubes_per_tub_lip_gloss tubes_per_tub_mascara 
         people_per_tube_lip_gloss people_per_tube_mascara
         h_tubs_lip_gloss h_tubes_per_tub_lip_gloss h_people_per_tube_lip_gloss
         h_tubs_mascara h_tubes_per_tub_mascara h_people_per_tube_mascara
  rw [h_tubs_lip_gloss, h_tubes_per_tub_lip_gloss, h_people_per_tube_lip_gloss]
  exact rfl


end marcy_total_people_served_l126_126046


namespace lengths_C_can_form_triangle_l126_126643

-- Definition of sets of lengths
def lengths_A := (3, 6, 9)
def lengths_B := (3, 5, 9)
def lengths_C := (4, 6, 9)
def lengths_D := (2, 6, 4)

-- Triangle condition for a given set of lengths
def can_form_triangle (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Proof problem statement 
theorem lengths_C_can_form_triangle : can_form_triangle 4 6 9 :=
by
  sorry

end lengths_C_can_form_triangle_l126_126643


namespace sufficient_but_not_necessary_l126_126925

theorem sufficient_but_not_necessary (x : ℝ) (h : x > 0): (x = 1 → x > 0) ∧ ¬(x > 0 → x = 1) :=
by
  sorry

end sufficient_but_not_necessary_l126_126925


namespace range_2a_minus_b_and_a_div_b_range_3x_minus_y_l126_126564

-- Proof for finding the range of 2a - b and a / b
theorem range_2a_minus_b_and_a_div_b (a b : ℝ) (h_a : 12 < a ∧ a < 60) (h_b : 15 < b ∧ b < 36) : 
  -12 < 2 * a - b ∧ 2 * a - b < 105 ∧ 1 / 3 < a / b ∧ a / b < 4 :=
by
  sorry

-- Proof for finding the range of 3x - y
theorem range_3x_minus_y (x y : ℝ) (h_xy_diff : -1 / 2 < x - y ∧ x - y < 1 / 2) (h_xy_sum : 0 < x + y ∧ x + y < 1) : 
  -1 < 3 * x - y ∧ 3 * x - y < 2 :=
by
  sorry

end range_2a_minus_b_and_a_div_b_range_3x_minus_y_l126_126564


namespace closely_related_interval_unique_l126_126158

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x + 4
noncomputable def g (x : ℝ) : ℝ := 2 * x - 3

def closely_related (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

theorem closely_related_interval_unique :
  closely_related f g 2 3 :=
sorry

end closely_related_interval_unique_l126_126158


namespace fabric_amount_for_each_dress_l126_126626

def number_of_dresses (total_hours : ℕ) (hours_per_dress : ℕ) : ℕ :=
  total_hours / hours_per_dress 

def fabric_per_dress (total_fabric : ℕ) (number_of_dresses : ℕ) : ℕ :=
  total_fabric / number_of_dresses

theorem fabric_amount_for_each_dress (total_fabric : ℕ) (hours_per_dress : ℕ) (total_hours : ℕ) :
  total_fabric = 56 ∧ hours_per_dress = 3 ∧ total_hours = 42 →
  fabric_per_dress total_fabric (number_of_dresses total_hours hours_per_dress) = 4 :=
by
  sorry

end fabric_amount_for_each_dress_l126_126626


namespace all_statements_correct_l126_126104

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem all_statements_correct (b : ℝ) (h1 : b > 0) (h2 : b ≠ 1) :
  (f b b = 1) ∧
  (f b 1 = 0) ∧
  (¬(0 ∈ Set.range (f b))) ∧
  (∀ x, 0 < x ∧ x < b → f b x < 1) ∧
  (∀ x, x > b → f b x > 1) := by
  unfold f
  sorry

end all_statements_correct_l126_126104


namespace sqrt_eight_simplify_l126_126153

theorem sqrt_eight_simplify : Real.sqrt 8 = 2 * Real.sqrt 2 := by
  sorry

end sqrt_eight_simplify_l126_126153


namespace integer_expression_l126_126025

theorem integer_expression (m : ℤ) : ∃ k : ℤ, k = (m / 3) + (m^2 / 2) + (m^3 / 6) :=
sorry

end integer_expression_l126_126025


namespace width_of_bottom_trapezium_l126_126333

theorem width_of_bottom_trapezium (top_width : ℝ) (area : ℝ) (depth : ℝ) (bottom_width : ℝ) 
  (h_top_width : top_width = 10)
  (h_area : area = 640)
  (h_depth : depth = 80) :
  bottom_width = 6 :=
by
  -- Problem description: calculating the width of the bottom of the trapezium given the conditions.
  sorry

end width_of_bottom_trapezium_l126_126333


namespace pictures_remaining_l126_126250

-- Define the initial number of pictures taken at the zoo and museum
def zoo_pictures : Nat := 50
def museum_pictures : Nat := 8
-- Define the number of pictures deleted
def deleted_pictures : Nat := 38

-- Define the total number of pictures taken initially and remaining after deletion
def total_pictures : Nat := zoo_pictures + museum_pictures
def remaining_pictures : Nat := total_pictures - deleted_pictures

theorem pictures_remaining : remaining_pictures = 20 := 
by 
  -- This theorem states that, given the conditions, the remaining pictures count must be 20
  sorry

end pictures_remaining_l126_126250


namespace numberOfFlowerbeds_l126_126048

def totalSeeds : ℕ := 32
def seedsPerFlowerbed : ℕ := 4

theorem numberOfFlowerbeds : totalSeeds / seedsPerFlowerbed = 8 :=
by
  sorry

end numberOfFlowerbeds_l126_126048


namespace actual_distance_traveled_l126_126951

theorem actual_distance_traveled :
  ∀ (t : ℝ) (d1 d2 : ℝ),
  d1 = 15 * t →
  d2 = 30 * t →
  d2 = d1 + 45 →
  d1 = 45 := by
  intro t d1 d2 h1 h2 h3
  sorry

end actual_distance_traveled_l126_126951


namespace problem_statement_l126_126653

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (S : ℕ → ℝ)

-- Conditions
def increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def condition1 := a 1 = 1
def condition2 := (a 3 + a 4) / (a 1 + a 2) = 4
def increasing := q > 0

-- Definition of S_n
def sum_geom (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)

theorem problem_statement (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (h_geom : increasing_geometric_sequence a q) 
  (h_condition1 : condition1 a) 
  (h_condition2 : condition2 a) 
  (h_increasing : increasing q)
  (h_sum_geom : sum_geom a q S) : 
  S 5 = 31 :=
sorry

end problem_statement_l126_126653


namespace right_triangle_leg_square_l126_126301

theorem right_triangle_leg_square (a b c : ℝ) 
  (h1 : c = a + 2) 
  (h2 : a^2 + b^2 = c^2) : b^2 = 4 * a + 4 := 
by
  sorry

end right_triangle_leg_square_l126_126301


namespace seated_people_count_l126_126576

theorem seated_people_count (n : ℕ) :
  (∀ (i : ℕ), i > 0 → i ≤ n) ∧
  (∀ (k : ℕ), k > 0 → k ≤ n → ∃ (p q : ℕ), 
         p = 31 ∧ q = 7 ∧ (p < n) ∧ (q < n) ∧
         p + 16 + 1 = q ∨ 
         p = 31 ∧ q = 14 ∧ (p < n) ∧ (q < n) ∧ 
         p - (n - q) + 1 = 16) → 
  n = 41 := 
by 
  sorry

end seated_people_count_l126_126576


namespace optionA_optionC_optionD_l126_126608

noncomputable def f (x : ℝ) := (3 : ℝ) ^ x / (1 + (3 : ℝ) ^ x)

theorem optionA : ∀ x : ℝ, f (-x) + f x = 1 := by
  sorry

theorem optionC : ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ (y > 0 ∧ y < 1) := by
  sorry

theorem optionD : ∀ x : ℝ, f (2 * x - 3) + f (x - 3) > 1 ↔ x > 2 := by
  sorry

end optionA_optionC_optionD_l126_126608


namespace corveus_lack_of_sleep_l126_126303

def daily_sleep_actual : ℕ := 4
def daily_sleep_recommended : ℕ := 6
def days_in_week : ℕ := 7

theorem corveus_lack_of_sleep : (daily_sleep_recommended - daily_sleep_actual) * days_in_week = 14 := 
by 
  sorry

end corveus_lack_of_sleep_l126_126303


namespace bowling_ball_surface_area_l126_126986

theorem bowling_ball_surface_area (diameter : ℝ) (h : diameter = 9) :
    let r := diameter / 2
    let surface_area := 4 * Real.pi * r^2
    surface_area = 81 * Real.pi := by
  sorry

end bowling_ball_surface_area_l126_126986


namespace initial_people_in_gym_l126_126420

variables (W A S : ℕ)

theorem initial_people_in_gym (h1 : (W - 3 + 2 - 3 + 4 - 2 + 1 = W + 1))
                              (h2 : (A + 2 - 1 + 3 - 3 + 1 = A + 2))
                              (h3 : (S + 1 - 2 + 1 + 3 - 2 + 2 = S + 3))
                              (final_total : (W + 1) + (A + 2) + (S + 3) + 2 = 30) :
  W + A + S = 22 :=
by 
  sorry

end initial_people_in_gym_l126_126420


namespace pants_and_coat_cost_l126_126579

noncomputable def pants_shirt_costs : ℕ := 100
noncomputable def coat_cost_times_shirt : ℕ := 5
noncomputable def coat_cost : ℕ := 180

theorem pants_and_coat_cost (p s c : ℕ) 
  (h1 : p + s = pants_shirt_costs)
  (h2 : c = coat_cost_times_shirt * s)
  (h3 : c = coat_cost) :
  p + c = 244 :=
by
  sorry

end pants_and_coat_cost_l126_126579


namespace calc_eq_neg_ten_thirds_l126_126183

theorem calc_eq_neg_ten_thirds :
  (7 / 4 - 7 / 8 - 7 / 12) / (-7 / 8) + (-7 / 8) / (7 / 4 - 7 / 8 - 7 / 12) = -10 / 3 := by 
sorry

end calc_eq_neg_ten_thirds_l126_126183


namespace combin_sum_l126_126107

def combin (n m : ℕ) : ℕ := Nat.factorial n / (Nat.factorial m * Nat.factorial (n - m))

theorem combin_sum (n : ℕ) (h₁ : n = 99) : combin n 2 + combin n 3 = 161700 := by
  sorry

end combin_sum_l126_126107


namespace tom_current_yellow_tickets_l126_126180

-- Definitions based on conditions provided
def yellow_to_red (y : ℕ) : ℕ := y * 10
def red_to_blue (r : ℕ) : ℕ := r * 10
def yellow_to_blue (y : ℕ) : ℕ := (yellow_to_red y) * 10

def tom_red_tickets : ℕ := 3
def tom_blue_tickets : ℕ := 7

def tom_total_blue_tickets : ℕ := (red_to_blue tom_red_tickets) + tom_blue_tickets
def tom_needed_blue_tickets : ℕ := 163

-- Proving that Tom currently has 2 yellow tickets
theorem tom_current_yellow_tickets : (tom_total_blue_tickets + tom_needed_blue_tickets) / yellow_to_blue 1 = 2 :=
by
  sorry

end tom_current_yellow_tickets_l126_126180


namespace charlie_share_l126_126824

theorem charlie_share (A B C : ℕ) 
  (h1 : (A - 10) * 18 = (B - 20) * 11)
  (h2 : (A - 10) * 24 = (C - 15) * 11)
  (h3 : A + B + C = 1105) : 
  C = 495 := 
by
  sorry

end charlie_share_l126_126824


namespace multiplication_expansion_l126_126274

theorem multiplication_expansion (y : ℤ) :
  (y^4 + 9 * y^2 + 81) * (y^2 - 9) = y^6 - 729 :=
by
  sorry

end multiplication_expansion_l126_126274


namespace paolo_coconuts_l126_126667

theorem paolo_coconuts
  (P : ℕ)
  (dante_coconuts : ℕ := 3 * P)
  (dante_sold : ℕ := 10)
  (dante_left : ℕ := 32)
  (h : dante_left + dante_sold = dante_coconuts) : P = 14 :=
by {
  sorry
}

end paolo_coconuts_l126_126667


namespace conditional_probability_age_30_40_female_l126_126464

noncomputable def total_people : ℕ := 350
noncomputable def total_females : ℕ := 180
noncomputable def females_30_40 : ℕ := 50

theorem conditional_probability_age_30_40_female :
  (females_30_40 : ℚ) / total_females = 5 / 18 :=
by
  sorry

end conditional_probability_age_30_40_female_l126_126464


namespace simplify_expression_l126_126707

theorem simplify_expression : ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := 
by 
  sorry

end simplify_expression_l126_126707


namespace remainder_2503_div_28_l126_126541

theorem remainder_2503_div_28 : 2503 % 28 = 11 := 
by
  -- The proof goes here
  sorry

end remainder_2503_div_28_l126_126541


namespace parallel_trans_l126_126510

variables {Line : Type} (a b c : Line)

-- Define parallel relation
def parallel (x y : Line) : Prop := sorry -- Replace 'sorry' with the actual definition

-- The main theorem
theorem parallel_trans (h1 : parallel a c) (h2 : parallel b c) : parallel a b :=
sorry

end parallel_trans_l126_126510


namespace problem1_problem2_l126_126762

-- Definitions and Lean statement for Problem 1
noncomputable def curve1 (x : ℝ) : ℝ := x / (2 * x - 1)
def point1 : ℝ × ℝ := (1, 1)
noncomputable def tangent_line1 (x y : ℝ) : Prop := x + y - 2 = 0

theorem problem1 : tangent_line1 (point1.fst) (curve1 (point1.fst)) :=
sorry -- proof goes here

-- Definitions and Lean statement for Problem 2
def parabola (x : ℝ) : ℝ := x^2
def point2 : ℝ × ℝ := (2, 3)
noncomputable def tangent_line2a (x y : ℝ) : Prop := 2 * x - y - 1 = 0
noncomputable def tangent_line2b (x y : ℝ) : Prop := 6 * x - y - 9 = 0

theorem problem2 : (tangent_line2a point2.fst point2.snd ∨ tangent_line2b point2.fst point2.snd) :=
sorry -- proof goes here

end problem1_problem2_l126_126762


namespace hostel_food_duration_l126_126197

noncomputable def food_last_days (total_food_units daily_consumption_new: ℝ) : ℝ :=
  total_food_units / daily_consumption_new

theorem hostel_food_duration:
  let x : ℝ := 1 -- assuming x is a positive real number
  let men_initial := 100
  let women_initial := 100
  let children_initial := 50
  let total_days := 40
  let consumption_man := 3 * x
  let consumption_woman := 2 * x
  let consumption_child := 1 * x
  let food_sufficient_for := 250
  let total_food_units := 550 * x * 40
  let men_leave := 30
  let women_leave := 20
  let children_leave := 10
  let men_new := men_initial - men_leave
  let women_new := women_initial - women_leave
  let children_new := children_initial - children_leave
  let daily_consumption_new := 210 * x + 160 * x + 40 * x 
  (food_last_days total_food_units daily_consumption_new) = 22000 / 410 := 
by
  sorry

end hostel_food_duration_l126_126197


namespace star_operation_result_l126_126533

def set_minus (A B : Set ℝ) : Set ℝ := {x : ℝ | x ∈ A ∧ x ∉ B}

def set_star (A B : Set ℝ) : Set ℝ :=
  set_minus A B ∪ set_minus B A

def A : Set ℝ := { y : ℝ | y ≥ 0 }
def B : Set ℝ := { x : ℝ | -3 ≤ x ∧ x ≤ 3 }

theorem star_operation_result :
  set_star A B = {x : ℝ | (-3 ≤ x ∧ x < 0) ∨ (x > 3)} :=
  sorry

end star_operation_result_l126_126533


namespace compare_star_values_l126_126809

def star (A B : ℤ) : ℤ := A * B - A / B

theorem compare_star_values : star 6 (-3) < star 4 (-4) := by
  sorry

end compare_star_values_l126_126809


namespace min_y_value_l126_126305

theorem min_y_value :
  ∃ c : ℝ, ∀ x : ℝ, (5 * x^2 + 20 * x + 25) >= c ∧ (∀ x : ℝ, (5 * x^2 + 20 * x + 25 = c) → x = -2) ∧ c = 5 :=
by
  sorry

end min_y_value_l126_126305


namespace wardrobe_single_discount_l126_126072

theorem wardrobe_single_discount :
  let p : ℝ := 50
  let d1 : ℝ := 0.30
  let d2 : ℝ := 0.20
  let final_price := p * (1 - d1) * (1 - d2)
  let equivalent_discount := 1 - (final_price / p)
  equivalent_discount = 0.44 :=
by
  let p : ℝ := 50
  let d1 : ℝ := 0.30
  let d2 : ℝ := 0.20
  let final_price := p * (1 - d1) * (1 - d2)
  let equivalent_discount := 1 - (final_price / p)
  show equivalent_discount = 0.44
  sorry

end wardrobe_single_discount_l126_126072


namespace chemistry_more_than_physics_l126_126445

noncomputable def M : ℕ := sorry
noncomputable def P : ℕ := sorry
noncomputable def C : ℕ := sorry
noncomputable def x : ℕ := sorry

theorem chemistry_more_than_physics :
  M + P = 20 ∧ C = P + x ∧ (M + C) / 2 = 20 → x = 20 :=
by
  sorry

end chemistry_more_than_physics_l126_126445


namespace sum_of_three_numbers_l126_126127

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 241)
  (h2 : a * b + b * c + c * a = 100) : 
  a + b + c = 21 := 
by
  sorry

end sum_of_three_numbers_l126_126127


namespace geometric_sequence_11th_term_l126_126848

theorem geometric_sequence_11th_term (a r : ℕ) :
    a * r^4 = 3 →
    a * r^7 = 24 →
    a * r^10 = 192 := by
    sorry

end geometric_sequence_11th_term_l126_126848


namespace fourth_square_area_l126_126029

theorem fourth_square_area (AB BC CD AD AC : ℝ) (h1 : AB^2 = 25) (h2 : BC^2 = 49) (h3 : CD^2 = 64) (h4 : AC^2 = AB^2 + BC^2)
  (h5 : AD^2 = AC^2 + CD^2) : AD^2 = 138 :=
by
  sorry

end fourth_square_area_l126_126029


namespace ratio_a_b_l126_126440

theorem ratio_a_b (a b c : ℝ) (h1 : a * (-1) ^ 2 + b * (-1) + c = 1) (h2 : a * 3 ^ 2 + b * 3 + c = 1) : 
  a / b = -2 :=
by 
  sorry

end ratio_a_b_l126_126440


namespace find_c_l126_126364

theorem find_c (x y c : ℝ) (h : x = 5 * y) (h2 : 7 * x + 4 * y = 13 * c) : c = 3 * y :=
by
  sorry

end find_c_l126_126364


namespace mascot_sales_growth_rate_equation_l126_126759

-- Define the conditions
def march_sales : ℝ := 100000
def may_sales : ℝ := 115000
def growth_rate (x : ℝ) : Prop := x > 0

-- Define the equation to be proven
theorem mascot_sales_growth_rate_equation (x : ℝ) (h : growth_rate x) :
    10 * (1 + x) ^ 2 = 11.5 :=
sorry

end mascot_sales_growth_rate_equation_l126_126759


namespace gerald_pfennigs_left_l126_126953

theorem gerald_pfennigs_left (cost_of_pie : ℕ) (farthings_initial : ℕ) (farthings_per_pfennig : ℕ) :
  cost_of_pie = 2 → farthings_initial = 54 → farthings_per_pfennig = 6 → 
  (farthings_initial / farthings_per_pfennig) - cost_of_pie = 7 :=
by
  intros h1 h2 h3
  sorry

end gerald_pfennigs_left_l126_126953


namespace number_subsets_property_p_l126_126017

def has_property_p (a b : ℕ) : Prop := 17 ∣ (a + b)

noncomputable def num_subsets_with_property_p : ℕ :=
  -- sorry, put computation result here using the steps above but skipping actual computation for brevity
  3928

theorem number_subsets_property_p :
  num_subsets_with_property_p = 3928 := sorry

end number_subsets_property_p_l126_126017


namespace focus_of_parabola_l126_126319

theorem focus_of_parabola (a : ℝ) (h : ℝ) (k : ℝ) (x y : ℝ) :
  (∀ x, y = a * (x - h) ^ 2 + k) →
  a = -2 ∧ h = 0 ∧ k = 4 →
  (0, y - (1 / (4 * a))) = (0, 31 / 8) := by
  sorry

end focus_of_parabola_l126_126319


namespace unique_linear_eq_sol_l126_126807

theorem unique_linear_eq_sol (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ∃ (a b c : ℤ), (∀ x y : ℕ, (a * x + b * y = c ↔ x = m ∧ y = n)) :=
by
  sorry

end unique_linear_eq_sol_l126_126807


namespace fg_evaluation_l126_126782

def f (x : ℝ) : ℝ := 4 * x - 3
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem fg_evaluation : f (g 3) = 97 := by
  sorry

end fg_evaluation_l126_126782


namespace find_larger_number_l126_126914

theorem find_larger_number (a b : ℕ) (h_diff : a - b = 3) (h_sum_squares : a^2 + b^2 = 117) (h_pos : 0 < a ∧ 0 < b) : a = 9 :=
by
  sorry

end find_larger_number_l126_126914


namespace pen_defect_probability_l126_126313

theorem pen_defect_probability :
  ∀ (n m : ℕ) (k : ℚ), n = 12 → m = 4 → k = 2 → 
  (8 / 12) * (7 / 11) = 141 / 330 := 
by
  intros n m k h1 h2 h3
  sorry

end pen_defect_probability_l126_126313


namespace BowlingAlleyTotalPeople_l126_126005

/--
There are 31 groups of people at the bowling alley.
Each group has about 6 people.
Prove that the total number of people at the bowling alley is 186.
-/
theorem BowlingAlleyTotalPeople : 
  let groups := 31
  let people_per_group := 6
  groups * people_per_group = 186 :=
by
  sorry

end BowlingAlleyTotalPeople_l126_126005


namespace arithmetic_sequence_15th_term_is_171_l126_126610

theorem arithmetic_sequence_15th_term_is_171 :
  ∀ (a d : ℕ), a = 3 → d = 15 - a → a + 14 * d = 171 :=
by
  intros a d h_a h_d
  rw [h_a, h_d]
  -- The proof would follow with the arithmetic calculation to determine the 15th term
  sorry

end arithmetic_sequence_15th_term_is_171_l126_126610


namespace handbag_monday_price_l126_126179

theorem handbag_monday_price (initial_price : ℝ) (primary_discount : ℝ) (additional_discount : ℝ)
(h_initial_price : initial_price = 250)
(h_primary_discount : primary_discount = 0.4)
(h_additional_discount : additional_discount = 0.1) :
(initial_price - initial_price * primary_discount) - ((initial_price - initial_price * primary_discount) * additional_discount) = 135 := by
  sorry

end handbag_monday_price_l126_126179


namespace whatsapp_messages_total_l126_126500

-- Define conditions
def messages_monday : ℕ := 300
def messages_tuesday : ℕ := 200
def messages_wednesday : ℕ := messages_tuesday + 300
def messages_thursday : ℕ := 2 * messages_wednesday
def messages_friday : ℕ := messages_thursday + (20 * messages_thursday) / 100
def messages_saturday : ℕ := messages_friday - (10 * messages_friday) / 100

-- Theorem statement to be proved
theorem whatsapp_messages_total :
  messages_monday + messages_tuesday + messages_wednesday + messages_thursday + messages_friday + messages_saturday = 4280 :=
by 
  sorry

end whatsapp_messages_total_l126_126500


namespace sqrt_14_bounds_l126_126224

theorem sqrt_14_bounds : 3 < Real.sqrt 14 ∧ Real.sqrt 14 < 4 := by
  sorry

end sqrt_14_bounds_l126_126224


namespace probability_C_and_D_l126_126221

theorem probability_C_and_D (P_A P_B : ℚ) (H1 : P_A = 1/4) (H2 : P_B = 1/3) :
  P_C + P_D = 5/12 :=
by
  sorry

end probability_C_and_D_l126_126221


namespace decorations_cost_l126_126124

def tablecloth_cost : ℕ := 20 * 25
def place_setting_cost : ℕ := 20 * 4 * 10
def rose_cost : ℕ := 20 * 10 * 5
def lily_cost : ℕ := 20 * 15 * 4

theorem decorations_cost :
  tablecloth_cost + place_setting_cost + rose_cost + lily_cost = 3500 :=
by sorry

end decorations_cost_l126_126124


namespace initial_contribution_amount_l126_126167

variable (x : ℕ)
variable (workers : ℕ := 1200)
variable (total_with_extra_contribution: ℕ := 360000)
variable (extra_contribution_each: ℕ := 50)

theorem initial_contribution_amount :
  (workers * x = total_with_extra_contribution - workers * extra_contribution_each) →
  workers * x = 300000 :=
by
  intro h
  sorry

end initial_contribution_amount_l126_126167


namespace half_ears_kernels_l126_126635

theorem half_ears_kernels (stalks ears_per_stalk total_kernels : ℕ) (X : ℕ)
  (half_ears : ℕ := stalks * ears_per_stalk / 2)
  (total_ears : ℕ := stalks * ears_per_stalk)
  (condition_e1 : stalks = 108)
  (condition_e2 : ears_per_stalk = 4)
  (condition_e3 : total_kernels = 237600)
  (condition_kernel_sum : total_kernels = 216 * X + 216 * (X + 100)) :
  X = 500 := by
  have condition_eq : 432 * X + 21600 = 237600 := by sorry
  have X_value : X = 216000 / 432 := by sorry
  have X_result : X = 500 := by sorry
  exact X_result

end half_ears_kernels_l126_126635


namespace pen_rubber_length_difference_l126_126063

theorem pen_rubber_length_difference (P R : ℕ) 
    (h1 : P = R + 3)
    (h2 : P = 12 - 2) 
    (h3 : R + P + 12 = 29) : 
    P - R = 3 :=
  sorry

end pen_rubber_length_difference_l126_126063


namespace scientific_notation_l126_126332

theorem scientific_notation : (20160 : ℝ) = 2.016 * 10^4 := 
  sorry

end scientific_notation_l126_126332


namespace rate_per_kg_of_grapes_l126_126862

theorem rate_per_kg_of_grapes : 
  ∀ (rate_per_kg_grapes : ℕ), 
    (10 * rate_per_kg_grapes + 9 * 55 = 1195) → 
    rate_per_kg_grapes = 70 := 
by
  intros rate_per_kg_grapes h
  sorry

end rate_per_kg_of_grapes_l126_126862


namespace triangle_inequality_l126_126214

theorem triangle_inequality
  (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : c > 0)
  (h5 : a + b > c)
  (h6 : a + c > b)
  (h7 : b + c > a) :
  a^2 + b^2 + c^2 < 2 * (1 - a * b * c) :=
sorry

end triangle_inequality_l126_126214


namespace relationships_with_correlation_l126_126257

-- Definitions for each of the relationships as conditions
def person_age_wealth := true -- placeholder definition 
def curve_points_coordinates := true -- placeholder definition
def apple_production_climate := true -- placeholder definition
def tree_diameter_height := true -- placeholder definition
def student_school := true -- placeholder definition

-- Statement to prove which relationships involve correlation
theorem relationships_with_correlation :
  person_age_wealth ∧ apple_production_climate ∧ tree_diameter_height :=
by
  sorry

end relationships_with_correlation_l126_126257


namespace intersection_of_A_and_B_l126_126980

def A : Set ℝ := {x | x^2 - x = 0}
def B : Set ℝ := {y | y^2 + y = 0}

theorem intersection_of_A_and_B : A ∩ B = {0} :=
by
  sorry

end intersection_of_A_and_B_l126_126980


namespace product_of_consecutive_numbers_l126_126006

theorem product_of_consecutive_numbers (n : ℕ) (k : ℕ) (h₁: n * (n + 1) * (n + 2) = 210) (h₂: n + (n + 1) = 11) : k = 3 :=
by
  sorry

end product_of_consecutive_numbers_l126_126006


namespace students_left_is_31_l126_126712

-- Define the conditions based on the problem statement
def total_students : ℕ := 124
def checked_out_early : ℕ := 93

-- Define the theorem that states the problem we want to prove
theorem students_left_is_31 :
  total_students - checked_out_early = 31 :=
by
  -- Proof would go here
  sorry

end students_left_is_31_l126_126712


namespace minimum_gloves_needed_l126_126480

-- Definitions based on conditions:
def participants : Nat := 43
def gloves_per_participant : Nat := 2

-- Problem statement proving the minimum number of gloves needed
theorem minimum_gloves_needed : participants * gloves_per_participant = 86 := by
  -- sorry allows us to omit the proof, focusing only on the formal statement
  sorry

end minimum_gloves_needed_l126_126480


namespace max_car_passing_400_l126_126147

noncomputable def max_cars_passing (speed : ℕ) (car_length : ℤ) (hour : ℕ) : ℕ :=
  20000 * speed / (5 * (speed + 1))

theorem max_car_passing_400 :
  max_cars_passing 20 5 1 / 10 = 400 := by
  sorry

end max_car_passing_400_l126_126147


namespace find_correct_r_l126_126869

noncomputable def ellipse_tangent_circle_intersection : Prop :=
  ∃ (E F : ℝ × ℝ) (r : ℝ), E ∈ { p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1 } ∧
                             F ∈ { p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1 } ∧ 
                             (E ≠ F) ∧
                             ((E.1 - 2)^2 + (E.2 - 3/2)^2 = r^2) ∧
                             ((F.1 - 2)^2 + (F.2 - 3/2)^2 = r^2) ∧
                             r = (Real.sqrt 37) / 37

theorem find_correct_r : ellipse_tangent_circle_intersection :=
sorry

end find_correct_r_l126_126869


namespace probability_red_or_blue_is_713_l126_126172

-- Definition of area ratios
def area_ratio_red : ℕ := 6
def area_ratio_yellow : ℕ := 2
def area_ratio_blue : ℕ := 1
def area_ratio_black : ℕ := 4

-- Total area ratio
def total_area_ratio := area_ratio_red + area_ratio_yellow + area_ratio_blue + area_ratio_black

-- Probability of stopping on either red or blue
def probability_red_or_blue := (area_ratio_red + area_ratio_blue) / total_area_ratio

-- Theorem stating the probability is 7/13
theorem probability_red_or_blue_is_713 : probability_red_or_blue = 7 / 13 :=
by
  unfold probability_red_or_blue total_area_ratio area_ratio_red area_ratio_blue
  simp
  sorry

end probability_red_or_blue_is_713_l126_126172


namespace range_of_k_l126_126212

theorem range_of_k (k : ℝ) : (4 < k ∧ k < 9 ∧ k ≠ 13 / 2) ↔ (k ∈ Set.Ioo 4 (13 / 2) ∪ Set.Ioo (13 / 2) 9) :=
by
  sorry

end range_of_k_l126_126212


namespace value_of_ak_l126_126296

noncomputable def Sn (n : ℕ) : ℤ := n^2 - 9 * n
noncomputable def a (n : ℕ) : ℤ := Sn n - Sn (n - 1)

theorem value_of_ak (k : ℕ) (hk : 5 < a k ∧ a k < 8) : a k = 6 := by
  sorry

end value_of_ak_l126_126296


namespace cost_price_l126_126921

theorem cost_price (SP MP CP : ℝ) (discount_rate : ℝ) 
  (h1 : MP = CP * 1.15)
  (h2 : SP = MP * (1 - discount_rate))
  (h3 : SP = 459)
  (h4 : discount_rate = 0.2608695652173913) : CP = 540 :=
by
  -- We use the hints given as conditions to derive the statement
  sorry

end cost_price_l126_126921


namespace training_days_l126_126769

def total_minutes : ℕ := 5 * 60
def minutes_per_day : ℕ := 10 + 20

theorem training_days :
  total_minutes / minutes_per_day = 10 :=
by
  sorry

end training_days_l126_126769


namespace cookies_difference_l126_126704

theorem cookies_difference :
  let bags := 9
  let boxes := 8
  let cookies_per_bag := 7
  let cookies_per_box := 12
  8 * 12 - 9 * 7 = 33 := 
by
  sorry

end cookies_difference_l126_126704


namespace check_double_root_statements_l126_126434

-- Condition Definitions
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ r : ℝ, a * r^2 + b * r + c = 0 ∧ a * (2 * r)^2 + b * (2 * r) + c = 0

-- Statement ①
def statement_1 : Prop := ¬is_double_root_equation 1 2 (-8)

-- Statement ②
def statement_2 : Prop := is_double_root_equation 1 (-3) 2

-- Statement ③
def statement_3 (m n : ℝ) : Prop := 
  (∃ r : ℝ, (r - 2) * (m * r + n) = 0 ∧ (m * (2 * r) + n = 0) ∧ r = 2) → 4 * m^2 + 5 * m * n + n^2 = 0

-- Statement ④
def statement_4 (p q : ℝ) : Prop := 
  (p * q = 2 → is_double_root_equation p 3 q)

-- Main proof problem statement
theorem check_double_root_statements (m n p q : ℝ) : 
  statement_1 ∧ statement_2 ∧ statement_3 m n ∧ statement_4 p q :=
by
  sorry

end check_double_root_statements_l126_126434


namespace rain_probability_in_two_locations_l126_126575

noncomputable def probability_no_rain_A : ℝ := 0.3
noncomputable def probability_no_rain_B : ℝ := 0.4

-- The probability of raining at a location is 1 - the probability of no rain at that location
noncomputable def probability_rain_A : ℝ := 1 - probability_no_rain_A
noncomputable def probability_rain_B : ℝ := 1 - probability_no_rain_B

-- The rain status in location A and location B are independent
theorem rain_probability_in_two_locations :
  probability_rain_A * probability_rain_B = 0.42 := by
  sorry

end rain_probability_in_two_locations_l126_126575


namespace john_not_stronger_than_ivan_l126_126552

-- Define strength relations
axiom stronger (a b : Type) : Prop

variable (whiskey liqueur vodka beer : Type)

axiom whiskey_stronger_than_vodka : stronger whiskey vodka
axiom liqueur_stronger_than_beer : stronger liqueur beer

-- Define types for cocktails and their strengths
variable (John_cocktail Ivan_cocktail : Type)

axiom John_mixed_whiskey_liqueur : John_cocktail
axiom Ivan_mixed_vodka_beer : Ivan_cocktail

-- Prove that it can't be asserted that John's cocktail is stronger
theorem john_not_stronger_than_ivan :
  ¬ (stronger John_cocktail Ivan_cocktail) :=
sorry

end john_not_stronger_than_ivan_l126_126552


namespace miguel_paint_area_l126_126929

def wall_height := 10
def wall_length := 15
def window_side := 3

theorem miguel_paint_area :
  (wall_height * wall_length) - (window_side * window_side) = 141 := 
by
  sorry

end miguel_paint_area_l126_126929


namespace raven_current_age_l126_126621

variable (R P : ℕ) -- Raven's current age, Phoebe's current age
variable (h₁ : P = 10) -- Phoebe is currently 10 years old
variable (h₂ : R + 5 = 4 * (P + 5)) -- In 5 years, Raven will be 4 times as old as Phoebe

theorem raven_current_age : R = 55 := 
by
  -- h2: R + 5 = 4 * (P + 5)
  -- h1: P = 10
  sorry

end raven_current_age_l126_126621


namespace eval_inverse_l126_126132

variable (g : ℕ → ℕ)
variable (g_inv : ℕ → ℕ)
variable (h₁ : g 4 = 6)
variable (h₂ : g 7 = 2)
variable (h₃ : g 3 = 7)
variable (h_inv₁ : g_inv 6 = 4)
variable (h_inv₂ : g_inv 7 = 3)

theorem eval_inverse (g : ℕ → ℕ)
(g_inv : ℕ → ℕ)
(h₁ : g 4 = 6)
(h₂ : g 7 = 2)
(h₃ : g 3 = 7)
(h_inv₁ : g_inv 6 = 4)
(h_inv₂ : g_inv 7 = 3) :
g_inv (g_inv 7 + g_inv 6) = 3 := by
  sorry

end eval_inverse_l126_126132


namespace relationship_of_f_values_l126_126788

noncomputable def f : ℝ → ℝ := sorry  -- placeholder for the actual function 

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f (-x + 2)

def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop := a < b → f a < f b

theorem relationship_of_f_values (h1 : is_increasing f 0 2) (h2 : is_even f) :
  f (5/2) > f 1 ∧ f 1 > f (7/2) :=
sorry -- proof goes here

end relationship_of_f_values_l126_126788


namespace sale_first_month_l126_126034

-- Declaration of all constant sales amounts in rupees
def sale_second_month : ℕ := 6927
def sale_third_month : ℕ := 6855
def sale_fourth_month : ℕ := 7230
def sale_fifth_month : ℕ := 6562
def sale_sixth_month : ℕ := 6791
def average_required : ℕ := 6800
def months : ℕ := 6

-- Total sales computed from the average sale requirement
def total_sales_needed : ℕ := months * average_required

-- The sum of sales for the second to sixth months
def total_sales_last_five_months := sale_second_month + sale_third_month + sale_fourth_month + sale_fifth_month + sale_sixth_month

-- Prove the sales in the first month given the conditions
theorem sale_first_month :
  total_sales_needed - total_sales_last_five_months = 6435 :=
by
  sorry

end sale_first_month_l126_126034


namespace sum_first_n_terms_of_geometric_seq_l126_126683

variable {α : Type*} [LinearOrderedField α] (a r : α) (n : ℕ)

def geometric_sequence (a r : α) (n : ℕ) : α :=
  a * r ^ (n - 1)

def sum_geometric_sequence (a r : α) (n : ℕ) : α :=
  a * (1 - r ^ n) / (1 - r)

theorem sum_first_n_terms_of_geometric_seq (h₁ : a * r + a * r^3 = 20) 
    (h₂ : a * r^2 + a * r^4 = 40) :
  sum_geometric_sequence a r n = 2^(n + 1) - 2 := 
sorry

end sum_first_n_terms_of_geometric_seq_l126_126683


namespace mapping_has_output_l126_126022

variable (M N : Type) (f : M → N)

theorem mapping_has_output (x : M) : ∃ y : N, f x = y :=
by
  sorry

end mapping_has_output_l126_126022


namespace construction_work_rate_l126_126279

theorem construction_work_rate (C : ℝ) 
  (h1 : ∀ t1 : ℝ, t1 = 10 → t1 * 8 = 80)
  (h2 : ∀ t2 : ℝ, t2 = 15 → t2 * C + 80 ≥ 300)
  (h3 : ∀ t : ℝ, t = 25 → ∀ t1 t2 : ℝ, t = t1 + t2 → t1 = 10 → t2 = 15)
  : C = 14.67 :=
by
  sorry

end construction_work_rate_l126_126279


namespace triangle_ABC_right_angled_l126_126687

variable {α : Type*} [LinearOrderedField α]

variables (a b c : α)
variables (A B C : ℝ)

theorem triangle_ABC_right_angled
  (h1 : b^2 = c^2 + a^2 - c * a)
  (h2 : Real.sin A = 2 * Real.sin C)
  (h3 : Real.cos B = 1 / 2) :
  B = (Real.pi / 2) := by
  sorry

end triangle_ABC_right_angled_l126_126687


namespace concert_parking_fee_l126_126117

theorem concert_parking_fee :
  let ticket_cost := 50 
  let processing_fee_percentage := 0.15 
  let entrance_fee_per_person := 5 
  let total_cost_concert := 135
  let num_people := 2 

  let total_ticket_cost := ticket_cost * num_people
  let processing_fee := total_ticket_cost * processing_fee_percentage
  let total_ticktet_cost_with_fee := total_ticket_cost + processing_fee
  let total_entrance_fee := entrance_fee_per_person * num_people
  let total_cost_without_parking := total_ticktet_cost_with_fee + total_entrance_fee
  total_cost_concert - total_cost_without_parking = 10 := by 
  sorry

end concert_parking_fee_l126_126117


namespace corrected_mean_l126_126845

open Real

theorem corrected_mean (n : ℕ) (mu_incorrect : ℝ)
                      (x1 y1 x2 y2 x3 y3 : ℝ)
                      (h1 : mu_incorrect = 41)
                      (h2 : n = 50)
                      (h3 : x1 = 48 ∧ y1 = 23)
                      (h4 : x2 = 36 ∧ y2 = 42)
                      (h5 : x3 = 55 ∧ y3 = 28) :
                      ((mu_incorrect * n + (x1 - y1) + (x2 - y2) + (x3 - y3)) / n = 41.92) :=
by
  sorry

end corrected_mean_l126_126845


namespace highest_y_coordinate_l126_126729

theorem highest_y_coordinate (x y : ℝ) (h : (x^2 / 49 + (y-3)^2 / 25 = 0)) : y = 3 :=
by
  sorry

end highest_y_coordinate_l126_126729


namespace inequality_holds_l126_126276

variable {a b c r : ℝ}
variable (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)

/-- 
To prove that the inequality r (ab + bc + ca) + (3 - r) (1/a + 1/b + 1/c) ≥ 9 
is true for all r satisfying 0 < r < 3 and for arbitrary positive reals a, b, c. 
-/
theorem inequality_holds (h : 0 < r ∧ r < 3) : 
  r * (a * b + b * c + c * a) + (3 - r) * (1 / a + 1 / b + 1 / c) ≥ 9 := by
  sorry

end inequality_holds_l126_126276


namespace find_three_leaf_clovers_l126_126596

-- Define the conditions
def total_leaves : Nat := 1000

-- Define the statement
theorem find_three_leaf_clovers (n : Nat) (h : 3 * n + 4 = total_leaves) : n = 332 :=
  sorry

end find_three_leaf_clovers_l126_126596


namespace a_minus_b_l126_126930

theorem a_minus_b (a b : ℚ) :
  (∀ x y, (x = 3 → y = 7) ∨ (x = 10 → y = 19) → y = a * x + b) →
  a - b = -(1/7) :=
by
  sorry

end a_minus_b_l126_126930


namespace find_p_l126_126323

theorem find_p (p : ℕ) (hp : Nat.Prime p) (hp2 : Nat.Prime (5 * p^2 - 2)) : p = 3 :=
sorry

end find_p_l126_126323


namespace probability_adjacent_vertices_decagon_l126_126993

noncomputable def probability_adjacent_vertices : ℚ :=
  let num_vertices := 10
  let adjacent := 2
  let remaining := num_vertices - 1 -- 9
  adjacent / remaining

theorem probability_adjacent_vertices_decagon :
  probability_adjacent_vertices = 2 / 9 :=
by
  unfold probability_adjacent_vertices
  simp
  sorry

end probability_adjacent_vertices_decagon_l126_126993


namespace necessary_but_not_sufficient_l126_126977

def p (x : ℝ) : Prop := x < 1
def q (x : ℝ) : Prop := x^2 + x - 2 < 0

theorem necessary_but_not_sufficient (x : ℝ):
  (p x → q x) ∧ (q x → p x) → False ∧ (q x → p x) :=
sorry

end necessary_but_not_sufficient_l126_126977


namespace value_of_m_div_x_l126_126340

variable (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ratio : a / b = 4 / 5)

def x := a + 0.25 * a
def m := b - 0.40 * b

theorem value_of_m_div_x (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ratio : a / b = 4 / 5) :
    m / x = 3 / 5 :=
by
  sorry

end value_of_m_div_x_l126_126340


namespace linlin_speed_l126_126601

theorem linlin_speed (distance time : ℕ) (q_speed linlin_speed : ℕ)
  (h1 : distance = 3290)
  (h2 : time = 7)
  (h3 : q_speed = 70)
  (h4 : distance = (q_speed + linlin_speed) * time) : linlin_speed = 400 :=
by sorry

end linlin_speed_l126_126601


namespace apples_bought_l126_126058

theorem apples_bought (x : ℕ) 
  (h1 : x ≠ 0)  -- x must be a positive integer
  (h2 : 2 * (x/3) = 2 * x / 3 + 2 - 6) : x = 24 := 
  by sorry

end apples_bought_l126_126058


namespace least_possible_n_l126_126821

noncomputable def d (n : ℕ) := 105 * n - 90

theorem least_possible_n :
  ∀ n : ℕ, d n > 0 → (45 - (d n + 90) / n = 150) → n ≥ 2 :=
by
  sorry

end least_possible_n_l126_126821


namespace color_fig_l126_126886

noncomputable def total_colorings (dots : Finset (Fin 9)) (colors : Finset (Fin 4))
  (adj : dots → dots → Prop)
  (diag : dots → dots → Prop) : Nat :=
  -- coloring left triangle
  let left_triangle := 4 * 3 * 2;
  -- coloring middle triangle considering diagonal restrictions
  let middle_triangle := 3 * 2;
  -- coloring right triangle considering same restrictions
  let right_triangle := 3 * 2;
  left_triangle * middle_triangle * middle_triangle

theorem color_fig (dots : Finset (Fin 9)) (colors : Finset (Fin 4))
  (adj : dots → dots → Prop)
  (diag : dots → dots → Prop) :
  total_colorings dots colors adj diag = 864 :=
by
  sorry

end color_fig_l126_126886


namespace initial_outlay_is_10000_l126_126825

theorem initial_outlay_is_10000 
  (I : ℝ)
  (manufacturing_cost_per_set : ℝ := 20)
  (selling_price_per_set : ℝ := 50)
  (num_sets : ℝ := 500)
  (profit : ℝ := 5000) :
  profit = (selling_price_per_set * num_sets) - (I + manufacturing_cost_per_set * num_sets) → I = 10000 :=
by
  intro h
  sorry

end initial_outlay_is_10000_l126_126825


namespace length_of_first_video_l126_126163

theorem length_of_first_video
  (total_time : ℕ)
  (second_video_time : ℕ)
  (last_two_videos_time : ℕ)
  (first_video_time : ℕ)
  (total_seconds : total_time = 510)
  (second_seconds : second_video_time = 4 * 60 + 30)
  (last_videos_seconds : last_two_videos_time = 60 + 60)
  (total_watch_time : total_time = second_video_time + last_two_videos_time + first_video_time) :
  first_video_time = 120 :=
by
  sorry

end length_of_first_video_l126_126163


namespace second_occurrence_at_55_l126_126514

/-- On the highway, starting from 3 kilometers, there is a speed limit sign every 4 kilometers,
and starting from 10 kilometers, there is a speed monitoring device every 9 kilometers.
The first time both types of facilities are encountered simultaneously is at 19 kilometers.
The second time both types of facilities are encountered simultaneously is at 55 kilometers. -/
theorem second_occurrence_at_55 :
  ∀ (k : ℕ), (∃ n m : ℕ, 3 + 4 * n = k ∧ 10 + 9 * m = k ∧ 19 + 36 = k) := sorry

end second_occurrence_at_55_l126_126514


namespace original_water_depth_in_larger_vase_l126_126331

-- Definitions based on the conditions
noncomputable def largerVaseDiameter := 20 -- in cm
noncomputable def smallerVaseDiameter := 10 -- in cm
noncomputable def smallerVaseHeight := 16 -- in cm

-- Proving the original depth of the water in the larger vase
theorem original_water_depth_in_larger_vase :
  ∃ depth : ℝ, depth = 14 :=
by
  sorry

end original_water_depth_in_larger_vase_l126_126331


namespace fraction_same_ratio_l126_126329

theorem fraction_same_ratio (x : ℚ) : 
  (x / (2 / 5)) = (3 / 7) / (6 / 5) ↔ x = 1 / 7 :=
by
  sorry

end fraction_same_ratio_l126_126329


namespace eval_expression_l126_126847

theorem eval_expression : 
  (520 * 0.43 / 0.26 - 217 * (2 + 3/7)) - (31.5 / (12 + 3/5) + 114 * (2 + 1/3) + (61 + 1/2)) = 0.5 := 
by
  sorry

end eval_expression_l126_126847


namespace kaleb_balance_l126_126450

theorem kaleb_balance (springEarnings : ℕ) (summerEarnings : ℕ) (suppliesCost : ℕ) (totalBalance : ℕ)
  (h1 : springEarnings = 4)
  (h2 : summerEarnings = 50)
  (h3 : suppliesCost = 4)
  (h4 : totalBalance = (springEarnings + summerEarnings) - suppliesCost) : totalBalance = 50 := by
  sorry

end kaleb_balance_l126_126450


namespace fractions_equiv_conditions_l126_126162

theorem fractions_equiv_conditions (x y z : ℝ) (h₁ : 2 * x - z ≠ 0) (h₂ : z ≠ 0) : 
  ((2 * x + y) / (2 * x - z) = y / -z) ↔ (y = -z) :=
by
  sorry

end fractions_equiv_conditions_l126_126162


namespace q_investment_l126_126423

theorem q_investment (p_investment : ℕ) (ratio_pq : ℕ × ℕ) (profit_ratio : ℕ × ℕ) (hp : p_investment = 12000) (hpr : ratio_pq = (3, 5)) : 
  (∃ q_investment, q_investment = 20000) :=
  sorry

end q_investment_l126_126423


namespace simplify_expression_l126_126091

theorem simplify_expression (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h : a^4 + b^4 = a^2 + b^2) :
  (a / b + b / a - 1 / (a * b)) = 3 :=
  sorry

end simplify_expression_l126_126091


namespace yunkyung_work_per_day_l126_126603

theorem yunkyung_work_per_day (T : ℝ) (h : T > 0) (H : T / 3 = 1) : T / 3 = 1/3 := 
by sorry

end yunkyung_work_per_day_l126_126603


namespace least_positive_integer_lemma_l126_126481

theorem least_positive_integer_lemma :
  ∃ x : ℕ, x > 0 ∧ x + 7237 ≡ 5017 [MOD 12] ∧ (∀ y : ℕ, y > 0 ∧ y + 7237 ≡ 5017 [MOD 12] → x ≤ y) :=
by
  sorry

end least_positive_integer_lemma_l126_126481


namespace total_sum_lent_l126_126593

theorem total_sum_lent (x : ℝ) (second_part : ℝ) (total_sum : ℝ) 
  (h1 : second_part = 1640) 
  (h2 : (x * 8 * 0.03) = (second_part * 3 * 0.05)) :
  total_sum = x + second_part → total_sum = 2665 := by
  sorry

end total_sum_lent_l126_126593


namespace range_of_x_l126_126685

noncomputable def problem_statement (x : ℝ) : Prop :=
  ∀ m : ℝ, abs m ≤ 2 → m * x^2 - 2 * x - m + 1 < 0 

theorem range_of_x (x : ℝ) :
  problem_statement x → ( ( -1 + Real.sqrt 7) / 2 < x ∧ x < ( 1 + Real.sqrt 3) / 2) :=
by
  intros h
  sorry

end range_of_x_l126_126685


namespace gcd_lcm_252_l126_126649

theorem gcd_lcm_252 {a b : ℕ} (h : Nat.gcd a b * Nat.lcm a b = 252) :
  ∃ S : Finset ℕ, S.card = 8 ∧ ∀ d ∈ S, d = Nat.gcd a b :=
by sorry

end gcd_lcm_252_l126_126649


namespace P_subset_Q_l126_126073

def P (x : ℝ) := abs x < 2
def Q (x : ℝ) := x < 2

theorem P_subset_Q : ∀ x : ℝ, P x → Q x := by
  sorry

end P_subset_Q_l126_126073


namespace sushil_marks_ratio_l126_126215

theorem sushil_marks_ratio
  (E M Science : ℕ)
  (h1 : E + M + Science = 170)
  (h2 : E = M / 4)
  (h3 : Science = 17) :
  E = 31 :=
by
  sorry

end sushil_marks_ratio_l126_126215


namespace no_real_solution_for_x_l126_126962

theorem no_real_solution_for_x
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1/y = 5) (h2 : y + 1/x = 1/3) :
  false :=
by
  sorry

end no_real_solution_for_x_l126_126962


namespace tayzia_tip_l126_126298

theorem tayzia_tip (haircut_women : ℕ) (haircut_children : ℕ) (num_women : ℕ) (num_children : ℕ) (tip_percentage : ℕ) :
  ((num_women * haircut_women + num_children * haircut_children) * tip_percentage / 100) = 24 :=
by
  -- Given conditions
  let haircut_women := 48
  let haircut_children := 36
  let num_women := 1
  let num_children := 2
  let tip_percentage := 20
  -- Perform the calculations as shown in the solution steps
  sorry

end tayzia_tip_l126_126298


namespace math_proof_problem_l126_126832

namespace Proofs

-- Definition of the arithmetic sequence {a_n}
def arithmetic_seq (a : ℕ → ℤ) : Prop := 
  ∀ m n, a n = a m + (n - m) * (a (m + 1) - a m)

-- Conditions for the arithmetic sequence
def a_conditions (a : ℕ → ℤ) : Prop := 
  a 3 = -6 ∧ a 6 = 0

-- Definition of the geometric sequence {b_n}
def geometric_seq (b : ℕ → ℤ) : Prop := 
  ∃ q, ∀ n, b (n + 1) = q * b n

-- Conditions for the geometric sequence
def b_conditions (b a : ℕ → ℤ) : Prop := 
  b 1 = -8 ∧ b 2 = a 1 + a 2 + a 3

-- The general formula for {a_n}
def a_formula (a : ℕ → ℤ) :=
  ∀ n, a n = 2 * n - 12

-- The sum formula of the first n terms of {b_n}
def S_n_formula (b : ℕ → ℤ) (S_n : ℕ → ℤ) :=
  ∀ n, S_n n = 4 * (1 - 3^n)

-- The main theorem combining all
theorem math_proof_problem (a b : ℕ → ℤ) (S_n : ℕ → ℤ) :
  arithmetic_seq a →
  a_conditions a →
  geometric_seq b →
  b_conditions b a →
  (a_formula a ∧ S_n_formula b S_n) :=
by 
  sorry

end Proofs

end math_proof_problem_l126_126832


namespace probability_of_odd_number_l126_126775

theorem probability_of_odd_number (wedge1 wedge2 wedge3 wedge4 wedge5 : ℝ)
  (h_wedge1_split : wedge1/3 = wedge2) 
  (h_wedge2_twice_wedge1 : wedge2 = 2 * (wedge1/3))
  (h_wedge3 : wedge3 = 1/4)
  (h_wedge5 : wedge5 = 1/4)
  (h_total : wedge1/3 + wedge2 + wedge3 + wedge4 + wedge5 = 1) :
  wedge1/3 + wedge3 + wedge5 = 7 / 12 :=
by
  sorry

end probability_of_odd_number_l126_126775


namespace calculate_expression_l126_126112

theorem calculate_expression : ((-1 + 2) * 3 + 2^2 / (-4)) = 2 :=
by
  sorry

end calculate_expression_l126_126112


namespace sum_of_roots_l126_126121

theorem sum_of_roots (g : ℝ → ℝ) 
  (h_symmetry : ∀ x : ℝ, g (3 + x) = g (3 - x))
  (h_roots : ∃ s1 s2 s3 s4 : ℝ, 
               g s1 = 0 ∧ 
               g s2 = 0 ∧ 
               g s3 = 0 ∧ 
               g s4 = 0 ∧ 
               s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ 
               s2 ≠ s3 ∧ s2 ≠ s4 ∧ s3 ≠ s4) :
  s1 + s2 + s3 + s4 = 12 :=
by 
  sorry

end sum_of_roots_l126_126121


namespace largest_whole_number_satisfying_inequality_l126_126528

theorem largest_whole_number_satisfying_inequality : ∃ n : ℤ, (1 / 3 + n / 7 < 1) ∧ (∀ m : ℤ, (1 / 3 + m / 7 < 1) → m ≤ n) ∧ n = 4 :=
sorry

end largest_whole_number_satisfying_inequality_l126_126528


namespace tetrahedron_edge_assignment_possible_l126_126066

theorem tetrahedron_edge_assignment_possible 
(s S a b : ℝ) 
(hs : s ≥ 0) (hS : S ≥ 0) (ha : a ≥ 0) (hb : b ≥ 0) :
  ∃ (e₁ e₂ e₃ e₄ e₅ e₆ : ℝ),
    e₁ ≥ 0 ∧ e₂ ≥ 0 ∧ e₃ ≥ 0 ∧ e₄ ≥ 0 ∧ e₅ ≥ 0 ∧ e₆ ≥ 0 ∧
    (e₁ + e₂ + e₃ = s) ∧ (e₁ + e₄ + e₅ = S) ∧
    (e₂ + e₄ + e₆ = a) ∧ (e₃ + e₅ + e₆ = b) := by
  sorry

end tetrahedron_edge_assignment_possible_l126_126066


namespace mr_bird_speed_to_be_on_time_l126_126145

theorem mr_bird_speed_to_be_on_time 
  (d : ℝ) 
  (t : ℝ)
  (h1 : d = 40 * (t + 1/20))
  (h2 : d = 60 * (t - 1/20)) :
  (d / t) = 48 :=
by
  sorry

end mr_bird_speed_to_be_on_time_l126_126145


namespace MH_greater_than_MK_l126_126604

-- Defining the conditions: BH perpendicular to HK and BH = 2
def BH := 2

-- Defining the conditions: CK perpendicular to HK and CK = 5
def CK := 5

-- M is the midpoint of BC, which implicitly means MB = MC in length
def M_midpoint_BC (MB MC : ℝ) :=
  MB = MC

theorem MH_greater_than_MK (MB MC MH MK : ℝ) 
  (hM_midpoint : M_midpoint_BC MB MC)
  (hMH : MH^2 + BH^2 = MB^2)
  (hMK : MK^2 + CK^2 = MC^2) :
  MH > MK :=
by
  sorry

end MH_greater_than_MK_l126_126604


namespace intersection_M_N_l126_126000

-- Define the set M and N
def M : Set ℝ := { x | x^2 ≤ 1 }
def N : Set ℝ := {-2, 0, 1}

-- Theorem stating that the intersection of M and N is {0, 1}
theorem intersection_M_N : M ∩ N = {0, 1} :=
by
  sorry

end intersection_M_N_l126_126000


namespace birthday_friends_count_l126_126985

theorem birthday_friends_count (n : ℕ) 
    (h1 : ∃ T, T = 12 * (n + 2)) 
    (h2 : ∃ T', T' = 16 * n) 
    (h3 : (∃ T, T = 12 * (n + 2)) → ∃ T', T' = 16 * n) : 
    n = 6 := 
by
    sorry

end birthday_friends_count_l126_126985


namespace total_cost_of_soup_l126_126746

theorem total_cost_of_soup 
  (pounds_beef : ℕ) (pounds_veg : ℕ) (cost_veg_per_pound : ℕ) (beef_price_multiplier : ℕ)
  (h1 : pounds_beef = 4)
  (h2 : pounds_veg = 6)
  (h3 : cost_veg_per_pound = 2)
  (h4 : beef_price_multiplier = 3):
  (pounds_veg * cost_veg_per_pound + pounds_beef * (cost_veg_per_pound * beef_price_multiplier)) = 36 :=
by
  sorry

end total_cost_of_soup_l126_126746


namespace ellen_painted_roses_l126_126129

theorem ellen_painted_roses :
  ∀ (r : ℕ),
    (5 * 17 + 7 * r + 3 * 6 + 2 * 20 = 213) → (r = 10) :=
by
  intros r h
  sorry

end ellen_painted_roses_l126_126129


namespace find_last_number_2_l126_126436

theorem find_last_number_2 (A B C D : ℤ) 
  (h1 : A + B + C = 18)
  (h2 : B + C + D = 9)
  (h3 : A + D = 13) : 
  D = 2 := 
sorry

end find_last_number_2_l126_126436


namespace find_a_l126_126435

variable (a : ℕ) (N : ℕ)
variable (h1 : Nat.gcd (2 * a + 1) (2 * a + 2) = 1) 
variable (h2 : Nat.gcd (2 * a + 1) (2 * a + 3) = 1)
variable (h3 : Nat.gcd (2 * a + 2) (2 * a + 3) = 2)
variable (hN : N = Nat.lcm (2 * a + 1) (Nat.lcm (2 * a + 2) (2 * a + 3)))
variable (hDiv : (2 * a + 4) ∣ N)

theorem find_a (h_pos : a > 0) : a = 1 :=
by
  -- Lean proof code will go here
  sorry

end find_a_l126_126435


namespace division_theorem_l126_126007

variable (x : ℤ)

def dividend := 8 * x ^ 4 + 7 * x ^ 3 + 3 * x ^ 2 - 5 * x - 8
def divisor := x - 1
def quotient := 8 * x ^ 3 + 15 * x ^ 2 + 18 * x + 13
def remainder := 5

theorem division_theorem : dividend x = divisor x * quotient x + remainder := by
  sorry

end division_theorem_l126_126007


namespace xy_square_sum_l126_126205

variable (x y : ℝ)

theorem xy_square_sum : (y + 6 = (x - 3)^2) →
                        (x + 6 = (y - 3)^2) →
                        (x ≠ y) →
                        x^2 + y^2 = 43 :=
by
  intros h₁ h₂ h₃
  sorry

end xy_square_sum_l126_126205


namespace quadratic_expression_evaluation_l126_126990

theorem quadratic_expression_evaluation (x y : ℝ) (h1 : 3 * x + y = 10) (h2 : x + 3 * y = 14) :
  10 * x^2 + 12 * x * y + 10 * y^2 = 296 :=
by
  -- Proof goes here
  sorry

end quadratic_expression_evaluation_l126_126990


namespace find_number_l126_126056

theorem find_number (x : ℝ) (h : x / 0.04 = 25) : x = 1 := 
by 
  -- the steps for solving this will be provided here
  sorry

end find_number_l126_126056


namespace probability_Q_within_2_of_origin_eq_pi_div_9_l126_126407

noncomputable def probability_within_circle (π : ℝ) : ℝ :=
  let area_of_square := (2 * 3)^2
  let area_of_circle := π * 2^2
  area_of_circle / area_of_square

theorem probability_Q_within_2_of_origin_eq_pi_div_9 :
  probability_within_circle Real.pi = Real.pi / 9 :=
by
  sorry

end probability_Q_within_2_of_origin_eq_pi_div_9_l126_126407


namespace eighth_odd_multiple_of_5_is_75_l126_126008

theorem eighth_odd_multiple_of_5_is_75 : ∃ n : ℕ, (n > 0 ∧ n % 2 = 1 ∧ n % 5 = 0 ∧ ∃ k : ℕ, k = 8 ∧ n = 10 * k - 5) :=
  sorry

end eighth_odd_multiple_of_5_is_75_l126_126008


namespace button_remainders_l126_126343

theorem button_remainders 
  (a : ℤ)
  (h1 : a % 2 = 1)
  (h2 : a % 3 = 1)
  (h3 : a % 4 = 3)
  (h4 : a % 5 = 3) :
  a % 12 = 7 := 
sorry

end button_remainders_l126_126343


namespace min_value_f_l126_126940

open Real

noncomputable def f (x : ℝ) : ℝ := (1 / x) + (4 / (1 - 2 * x))

theorem min_value_f : ∃ (x : ℝ), (0 < x ∧ x < 1 / 2) ∧ f x = 6 + 4 * sqrt 2 := by
  sorry

end min_value_f_l126_126940


namespace option_c_not_equivalent_l126_126919

theorem option_c_not_equivalent :
  ¬ (785 * 10^(-9) = 7.845 * 10^(-6)) :=
by
  sorry

end option_c_not_equivalent_l126_126919


namespace factorize_x9_minus_512_l126_126607

theorem factorize_x9_minus_512 : 
  ∀ (x : ℝ), x^9 - 512 = (x - 2) * (x^2 + 2 * x + 4) * (x^6 + 8 * x^3 + 64) := by
  intro x
  sorry

end factorize_x9_minus_512_l126_126607


namespace abs_two_minus_sqrt_five_l126_126382

noncomputable def sqrt_5 : ℝ := Real.sqrt 5

theorem abs_two_minus_sqrt_five : |2 - sqrt_5| = sqrt_5 - 2 := by
  sorry

end abs_two_minus_sqrt_five_l126_126382


namespace find_angle_A_find_AB_l126_126881

theorem find_angle_A (A B C : ℝ) (h1 : 2 * Real.sin B * Real.cos A = Real.sin (A + C)) (h2 : A + B + C = Real.pi) :
  A = Real.pi / 3 := by
  sorry

theorem find_AB (A B C : ℝ) (AB BC AC : ℝ) (h1 : 2 * Real.sin B * Real.cos A = Real.sin (A + C))
  (h2 : BC = 2) (h3 : 1 / 2 * AB * AC * Real.sin (Real.pi / 3) = Real.sqrt 3)
  (h4 : A = Real.pi / 3) :
  AB = 2 := by
  sorry

end find_angle_A_find_AB_l126_126881


namespace extra_apples_correct_l126_126662

def num_red_apples : ℕ := 6
def num_green_apples : ℕ := 15
def num_students : ℕ := 5
def num_apples_ordered : ℕ := num_red_apples + num_green_apples
def num_apples_taken : ℕ := num_students
def num_extra_apples : ℕ := num_apples_ordered - num_apples_taken

theorem extra_apples_correct : num_extra_apples = 16 := by
  sorry

end extra_apples_correct_l126_126662


namespace find_x_value_l126_126360

theorem find_x_value (x : ℝ) (h1 : 0 < x) (h2 : x < 180) :
  (Real.tan (150 - x * Real.pi / 180) = 
   (Real.sin (150 * Real.pi / 180) - Real.sin (x * Real.pi / 180)) /
   (Real.cos (150 * Real.pi / 180) - Real.cos (x * Real.pi / 180))) → 
  x = 110 := 
by 
  sorry

end find_x_value_l126_126360


namespace football_starting_lineup_count_l126_126899

variable (n_team_members n_offensive_linemen : ℕ)
variable (H_team_members : 12 = n_team_members)
variable (H_offensive_linemen : 5 = n_offensive_linemen)

theorem football_starting_lineup_count :
  n_team_members = 12 → n_offensive_linemen = 5 →
  (n_offensive_linemen * (n_team_members - 1) * (n_team_members - 2) * ((n_team_members - 3) * (n_team_members - 4) / 2)) = 19800 := 
by
  intros
  sorry

end football_starting_lineup_count_l126_126899


namespace min_sum_xy_l126_126827

theorem min_sum_xy (x y : ℕ) (hx : x ≠ y) (pos_x : 0 < x) (pos_y : 0 < y)
  (h : (1 : ℚ) / x + 1 / y = 1 / 12) : x + y = 49 :=
sorry

end min_sum_xy_l126_126827


namespace first_term_geometric_sequence_l126_126392

theorem first_term_geometric_sequence (a r : ℕ) (h₁ : a * r^5 = 32) (h₂ : r = 2) : a = 1 := by
  sorry

end first_term_geometric_sequence_l126_126392


namespace parabola_equation_line_AB_fixed_point_min_area_AMBN_l126_126751

-- Prove that the equation of the parabola is y^2 = 4x given the focus (1,0) for y^2 = 2px
theorem parabola_equation (p : ℝ) (h : p > 0) (foc : (1, 0) = (1, 2*p*1/4)):
  (∀ x y: ℝ, y^2 = 4*x ↔ y^2 = 2*p*x) := sorry

-- Prove that line AB passes through fixed point T(2,0) given conditions
theorem line_AB_fixed_point (A B : ℝ × ℝ) (hA : A.2^2 = 4*A.1) 
    (hB : B.2^2 = 4*B.1) (h : A.1*B.1 + A.2*B.2 = -4) :
  ∃ T : ℝ × ℝ, T = (2, 0) := sorry

-- Prove that minimum value of area Quadrilateral AMBN is 48
theorem min_area_AMBN (T : ℝ × ℝ) (A B M N : ℝ × ℝ)
    (hT : T = (2, 0)) (hA : A.2^2 = 4*A.1) (hB : B.2^2 = 4*B.1)
    (hM : M.2^2 = 4*M.1) (hN : N.2^2 = 4*N.1)
    (line_AB : A.1 * B.1 + A.2 * B.2 = -4) :
  ∀ (m : ℝ), T.2 = -(1/m)*T.1 + 2 → 
  ((1+m^2) * (1+1/m^2)) * ((m^2 + 2) * (1/m^2 + 2)) = 256 → 
  8 * 48 = 48 := sorry

end parabola_equation_line_AB_fixed_point_min_area_AMBN_l126_126751


namespace giants_need_to_win_more_games_l126_126499

/-- The Giants baseball team is trying to make their league playoff.
They have played 20 games and won 12 of them. To make the playoffs, they need to win 2/3 of 
their games over the season. If there are 10 games left, how many do they have to win to
make the playoffs? 
-/
theorem giants_need_to_win_more_games (played won needed_won total remaining required_wins additional_wins : ℕ)
    (h1 : played = 20)
    (h2 : won = 12)
    (h3 : remaining = 10)
    (h4 : total = played + remaining)
    (h5 : total = 30)
    (h6 : required_wins = 2 * total / 3)
    (h7 : additional_wins = required_wins - won) :
    additional_wins = 8 := 
    by
      -- sorry should be used if the proof steps were required.
sorry

end giants_need_to_win_more_games_l126_126499
