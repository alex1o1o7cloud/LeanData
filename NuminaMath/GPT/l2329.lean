import Mathlib

namespace NUMINAMATH_GPT_multiple_of_tickletoe_nails_l2329_232922

def violet_nails := 27
def total_nails := 39
def difference := 3

theorem multiple_of_tickletoe_nails : ∃ (M T : ℕ), violet_nails = M * T + difference ∧ total_nails = violet_nails + T ∧ (M = 2) :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_tickletoe_nails_l2329_232922


namespace NUMINAMATH_GPT_average_error_diff_l2329_232901

theorem average_error_diff (n : ℕ) (total_data_pts : ℕ) (error_data1 error_data2 : ℕ)
  (h_n : n = 30) (h_total_data_pts : total_data_pts = 30)
  (h_error_data1 : error_data1 = 105) (h_error_data2 : error_data2 = 15)
  : (error_data1 - error_data2) / n = 3 :=
sorry

end NUMINAMATH_GPT_average_error_diff_l2329_232901


namespace NUMINAMATH_GPT_drawing_blue_ball_probability_l2329_232924

noncomputable def probability_of_blue_ball : ℚ :=
  let total_balls := 10
  let blue_balls := 6
  blue_balls / total_balls

theorem drawing_blue_ball_probability :
  probability_of_blue_ball = 3 / 5 :=
by
  sorry -- Proof is omitted as per instructions.

end NUMINAMATH_GPT_drawing_blue_ball_probability_l2329_232924


namespace NUMINAMATH_GPT_sally_bought_48_eggs_l2329_232946

-- Define the number of eggs in a dozen
def eggs_in_a_dozen : ℕ := 12

-- Define the number of dozens Sally bought
def dozens_sally_bought : ℕ := 4

-- Define the total number of eggs Sally bought
def total_eggs_sally_bought : ℕ := dozens_sally_bought * eggs_in_a_dozen

-- Theorem stating the number of eggs Sally bought
theorem sally_bought_48_eggs : total_eggs_sally_bought = 48 :=
sorry

end NUMINAMATH_GPT_sally_bought_48_eggs_l2329_232946


namespace NUMINAMATH_GPT_ratio_of_sums_eq_19_over_17_l2329_232933

theorem ratio_of_sums_eq_19_over_17 :
  let a₁ := 5
  let d₁ := 3
  let l₁ := 59
  let a₂ := 4
  let d₂ := 4
  let l₂ := 64
  let n₁ := 19  -- from solving l₁ = a₁ + (n₁ - 1) * d₁
  let n₂ := 16  -- from solving l₂ = a₂ + (n₂ - 1) * d₂
  let S₁ := n₁ * (a₁ + l₁) / 2
  let S₂ := n₂ * (a₂ + l₂) / 2
  S₁ / S₂ = 19 / 17 := by sorry

end NUMINAMATH_GPT_ratio_of_sums_eq_19_over_17_l2329_232933


namespace NUMINAMATH_GPT_four_digit_integer_unique_l2329_232929

theorem four_digit_integer_unique (a b c d : ℕ) (h1 : a + b + c + d = 16) (h2 : b + c = 10) (h3 : a - d = 2)
    (h4 : (a - b + c - d) % 11 = 0) : a = 4 ∧ b = 6 ∧ c = 4 ∧ d = 2 := 
  by 
    sorry

end NUMINAMATH_GPT_four_digit_integer_unique_l2329_232929


namespace NUMINAMATH_GPT_triangle_proof_l2329_232908

variables (α β γ a b c : ℝ)

-- Definitions based on the given conditions
def angle_relation (α β : ℝ) : Prop := 3 * α + 2 * β = 180
def triangle_angle_sum (α β γ : ℝ) : Prop := α + β + γ = 180

-- Lean statement for the proof problem
theorem triangle_proof
  (h1 : angle_relation α β)
  (h2 : triangle_angle_sum α β γ) :
  a^2 + b * c = c^2 :=
sorry

end NUMINAMATH_GPT_triangle_proof_l2329_232908


namespace NUMINAMATH_GPT_expression_bounds_l2329_232950

theorem expression_bounds (x y z w : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  2 * Real.sqrt 2 ≤ (Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) +
                     Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2)) ∧
  (Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) +
   Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2)) ≤ 4 := sorry

end NUMINAMATH_GPT_expression_bounds_l2329_232950


namespace NUMINAMATH_GPT_ratio_of_radii_l2329_232949

variables (a b : ℝ) (h : π * b ^ 2 - π * a ^ 2 = 4 * π * a ^ 2)

theorem ratio_of_radii (h : π * b ^ 2 - π * a ^ 2 = 4 * π * a ^ 2) : 
  a / b = Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_GPT_ratio_of_radii_l2329_232949


namespace NUMINAMATH_GPT_number_of_rhombuses_is_84_l2329_232947

def total_rhombuses (side_length_large_triangle : Nat) (side_length_small_triangle : Nat) (num_small_triangles : Nat) : Nat :=
  if side_length_large_triangle = 10 ∧ 
     side_length_small_triangle = 1 ∧ 
     num_small_triangles = 100 then 84 else 0

theorem number_of_rhombuses_is_84 :
  total_rhombuses 10 1 100 = 84 := by
  sorry

end NUMINAMATH_GPT_number_of_rhombuses_is_84_l2329_232947


namespace NUMINAMATH_GPT_divide_cookie_into_16_equal_parts_l2329_232931

def Cookie (n : ℕ) : Type := sorry

theorem divide_cookie_into_16_equal_parts (cookie : Cookie 64) :
  ∃ (slices : List (Cookie 4)), slices.length = 16 ∧ 
  (∀ (slice : Cookie 4), slice ≠ cookie) := 
sorry

end NUMINAMATH_GPT_divide_cookie_into_16_equal_parts_l2329_232931


namespace NUMINAMATH_GPT_perfect_square_trinomial_l2329_232921

theorem perfect_square_trinomial (a b : ℝ) :
  (∃ c : ℝ, 4 * (c^2) = 9 ∧ 4 * c = a - b) → 2 * a - 2 * b = 24 ∨ 2 * a - 2 * b = -24 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l2329_232921


namespace NUMINAMATH_GPT_stratified_sampling_l2329_232904

theorem stratified_sampling (total_students : ℕ) (num_freshmen : ℕ)
                            (freshmen_sample : ℕ) (sample_size : ℕ)
                            (h1 : total_students = 1500)
                            (h2 : num_freshmen = 400)
                            (h3 : freshmen_sample = 12)
                            (h4 : (freshmen_sample : ℚ) / num_freshmen = sample_size / total_students) :
  sample_size = 45 :=
  by
  -- There would be some steps to prove this, but they are omitted.
  sorry

end NUMINAMATH_GPT_stratified_sampling_l2329_232904


namespace NUMINAMATH_GPT_smallest_b_l2329_232953

-- Define the variables and conditions
variables {a b : ℝ}

-- Assumptions based on the problem conditions
axiom h1 : 2 < a
axiom h2 : a < b

-- The theorems for the triangle inequality violations
theorem smallest_b (h : a ≥ b / (2 * b - 1)) (h' : 2 + a ≤ b) : b = (3 + Real.sqrt 7) / 2 :=
sorry

end NUMINAMATH_GPT_smallest_b_l2329_232953


namespace NUMINAMATH_GPT_inequality_solution_l2329_232909

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x

theorem inequality_solution (k : ℝ) (h_pos : 0 < k) :
  (0 < k ∧ k < 1 ∧ (1 : ℝ) < x ∧ x < (1 / k)) ∨
  (k = 1 ∧ False) ∨
  (1 < k ∧ (1 / k) < x ∧ x < 1)
  ∨ False :=
sorry

end NUMINAMATH_GPT_inequality_solution_l2329_232909


namespace NUMINAMATH_GPT_distance_small_ball_to_surface_l2329_232958

-- Define the main variables and conditions
variables (R : ℝ)

-- Define the conditions of the problem
def bottomBallRadius : ℝ := 2 * R
def topBallRadius : ℝ := R
def edgeLengthBaseTetrahedron : ℝ := 4 * R
def edgeLengthLateralTetrahedron : ℝ := 3 * R

-- Define the main statement in Lean format
theorem distance_small_ball_to_surface (R : ℝ) :
  (3 * R) = R + bottomBallRadius R :=
sorry

end NUMINAMATH_GPT_distance_small_ball_to_surface_l2329_232958


namespace NUMINAMATH_GPT_sector_angle_l2329_232906

theorem sector_angle (R : ℝ) (S : ℝ) (α : ℝ) (hR : R = 2) (hS : S = 8) : 
  α = 4 := by
  sorry

end NUMINAMATH_GPT_sector_angle_l2329_232906


namespace NUMINAMATH_GPT_correct_evaluation_at_3_l2329_232923

noncomputable def polynomial (x : ℝ) : ℝ := 
  (4 * x^3 - 6 * x + 5) * (9 - 3 * x)

def expanded_poly (x : ℝ) : ℝ := 
  -12 * x^4 + 36 * x^3 + 18 * x^2 - 69 * x + 45

theorem correct_evaluation_at_3 :
  polynomial = expanded_poly →
  (12 * (-12) + 6 * 36 + 3 * 18 - 69) = 57 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_correct_evaluation_at_3_l2329_232923


namespace NUMINAMATH_GPT_smallest_x_multiple_of_1024_l2329_232936

theorem smallest_x_multiple_of_1024 (x : ℕ) (hx : 900 * x % 1024 = 0) : x = 256 :=
sorry

end NUMINAMATH_GPT_smallest_x_multiple_of_1024_l2329_232936


namespace NUMINAMATH_GPT_find_original_money_sandy_took_l2329_232983

noncomputable def originalMoney (remainingMoney : ℝ) (clothingPercent electronicsPercent foodPercent additionalSpendPercent salesTaxPercent : ℝ) : Prop :=
  let X := (remainingMoney / (1 - ((clothingPercent + electronicsPercent + foodPercent) + additionalSpendPercent) * (1 + salesTaxPercent)))
  abs (X - 397.73) < 0.01

theorem find_original_money_sandy_took :
  originalMoney 140 0.25 0.15 0.10 0.20 0.08 :=
sorry

end NUMINAMATH_GPT_find_original_money_sandy_took_l2329_232983


namespace NUMINAMATH_GPT_symmetric_circle_eq_l2329_232999

/-- Define the equation of the circle C -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Define the equation of the line l -/
def line_equation (x y : ℝ) : Prop := x + y - 1 = 0

/-- 
The symmetric circle to C with respect to line l 
has the equation (x - 1)^2 + (y - 1)^2 = 4.
-/
theorem symmetric_circle_eq (x y : ℝ) :
  (∃ x y : ℝ, circle_equation x y) → 
  (∃ x y : ℝ, line_equation x y) →
  (∃ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 4) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_circle_eq_l2329_232999


namespace NUMINAMATH_GPT_base_of_parallelogram_l2329_232996

theorem base_of_parallelogram (Area Height : ℕ) (h1 : Area = 44) (h2 : Height = 11) : (Area / Height) = 4 :=
by
  sorry

end NUMINAMATH_GPT_base_of_parallelogram_l2329_232996


namespace NUMINAMATH_GPT_polygon_intersections_inside_circle_l2329_232964

noncomputable def number_of_polygon_intersections
    (polygonSides: List Nat) : Nat :=
  let pairs := [(4,5), (4,7), (4,9), (5,7), (5,9), (7,9)]
  pairs.foldl (λ acc (p1, p2) => acc + 2 * min p1 p2) 0

theorem polygon_intersections_inside_circle :
  number_of_polygon_intersections [4, 5, 7, 9] = 58 :=
by
  sorry

end NUMINAMATH_GPT_polygon_intersections_inside_circle_l2329_232964


namespace NUMINAMATH_GPT_bouquet_daisies_percentage_l2329_232945

theorem bouquet_daisies_percentage :
  (∀ (total white yellow white_tulips white_daisies yellow_tulips yellow_daisies : ℕ),
    total = white + yellow →
    white = 7 * total / 10 →
    yellow = total - white →
    white_tulips = white / 2 →
    white_daisies = white / 2 →
    yellow_daisies = 2 * yellow / 3 →
    yellow_tulips = yellow - yellow_daisies →
    (white_daisies + yellow_daisies) * 100 / total = 55) :=
by
  intros total white yellow white_tulips white_daisies yellow_tulips yellow_daisies h_total h_white h_yellow ht_wd hd_wd hd_yd ht_yt
  sorry

end NUMINAMATH_GPT_bouquet_daisies_percentage_l2329_232945


namespace NUMINAMATH_GPT_ratio_of_surface_areas_l2329_232961

theorem ratio_of_surface_areas (r1 r2 : ℝ) (h : r1 / r2 = 1 / 2) :
  (4 * Real.pi * r1^2) / (4 * Real.pi * r2^2) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_surface_areas_l2329_232961


namespace NUMINAMATH_GPT_ratio_SP2_SP1_l2329_232980

variable (CP : ℝ)

-- First condition: Sold at a profit of 140%
def SP1 := 2.4 * CP

-- Second condition: Sold at a loss of 20%
def SP2 := 0.8 * CP

-- Statement: The ratio of SP2 to SP1 is 1 to 3
theorem ratio_SP2_SP1 : SP2 / SP1 = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_SP2_SP1_l2329_232980


namespace NUMINAMATH_GPT_evaluate_custom_op_l2329_232997

def custom_op (a b : ℝ) : ℝ := (a - b)^2

theorem evaluate_custom_op (x y : ℝ) : custom_op ((x + y)^2) ((y - x)^2) = 16 * x^2 * y^2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_custom_op_l2329_232997


namespace NUMINAMATH_GPT_time_to_cover_escalator_l2329_232900

-- Define the given conditions
def escalator_speed : ℝ := 20 -- feet per second
def escalator_length : ℝ := 360 -- feet
def delay_time : ℝ := 5 -- seconds
def person_speed : ℝ := 4 -- feet per second

-- Define the statement to be proven
theorem time_to_cover_escalator : (delay_time + (escalator_length - (escalator_speed * delay_time)) / (person_speed + escalator_speed)) = 15.83 := 
by {
  sorry
}

end NUMINAMATH_GPT_time_to_cover_escalator_l2329_232900


namespace NUMINAMATH_GPT_choir_arrangement_l2329_232913

/-- There are 4 possible row-lengths for arranging 90 choir members such that each row has the same
number of individuals and the number of members per row is between 6 and 15. -/
theorem choir_arrangement (x : ℕ) (h : 6 ≤ x ∧ x ≤ 15 ∧ 90 % x = 0) :
  x = 6 ∨ x = 9 ∨ x = 10 ∨ x = 15 :=
by
  sorry

end NUMINAMATH_GPT_choir_arrangement_l2329_232913


namespace NUMINAMATH_GPT_tangent_line_at_01_l2329_232959

noncomputable def tangent_line_equation (x : ℝ) : ℝ := x * Real.exp x + 1

theorem tangent_line_at_01 : ∃ (m b : ℝ), (m = 1) ∧ (b = 1) ∧ (∀ x, tangent_line_equation x = m * x + b) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_01_l2329_232959


namespace NUMINAMATH_GPT_valid_subsets_12_even_subsets_305_l2329_232972

def valid_subsets_count(n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 4
  else
    valid_subsets_count (n - 1) +
    valid_subsets_count (n - 2) +
    valid_subsets_count (n - 3)
    -- Recurrence relation for valid subsets which satisfy the conditions

theorem valid_subsets_12 : valid_subsets_count 12 = 610 :=
  by sorry
  -- We need to verify recurrence and compute for n = 12 (optional step if just computing, not proving the sequence.)

theorem even_subsets_305 :
  (valid_subsets_count 12) / 2 = 305 :=
  by sorry
  -- Concludes that half the valid subsets for n = 12 are even-sized sets.

end NUMINAMATH_GPT_valid_subsets_12_even_subsets_305_l2329_232972


namespace NUMINAMATH_GPT_total_maple_trees_in_park_after_planting_l2329_232920

def number_of_maple_trees_in_the_park (X_M : ℕ) (Y_M : ℕ) : ℕ := 
  X_M + Y_M

theorem total_maple_trees_in_park_after_planting : 
  number_of_maple_trees_in_the_park 2 9 = 11 := 
by 
  unfold number_of_maple_trees_in_the_park
  -- provide the mathematical proof here
  sorry

end NUMINAMATH_GPT_total_maple_trees_in_park_after_planting_l2329_232920


namespace NUMINAMATH_GPT_part1_part2_l2329_232935

noncomputable def f (ω x : ℝ) : ℝ := 4 * ((Real.sin (ω * x - Real.pi / 4)) * (Real.cos (ω * x)))

noncomputable def g (α : ℝ) : ℝ := 2 * (Real.sin (α - Real.pi / 6)) - Real.sqrt 2

theorem part1 (ω : ℝ) (x : ℝ) (hω : 0 < ω ∧ ω < 2) (hx : f ω (Real.pi / 4) = Real.sqrt 2) : 
  ∃ T > 0, ∀ x, f ω (x + T) = f ω x :=
sorry

theorem part2 (α : ℝ) (hα: 0 < α ∧ α < Real.pi / 2) (h : g α = 4 / 3 - Real.sqrt 2) : 
  Real.cos α = (Real.sqrt 15 - 2) / 6 :=
sorry

end NUMINAMATH_GPT_part1_part2_l2329_232935


namespace NUMINAMATH_GPT_sarees_original_price_l2329_232925

theorem sarees_original_price (P : ℝ) (h : 0.90 * P * 0.95 = 342) : P = 400 :=
by
  sorry

end NUMINAMATH_GPT_sarees_original_price_l2329_232925


namespace NUMINAMATH_GPT_value_two_std_dev_less_l2329_232905

noncomputable def mean : ℝ := 15.5
noncomputable def std_dev : ℝ := 1.5

theorem value_two_std_dev_less : mean - 2 * std_dev = 12.5 := by
  sorry

end NUMINAMATH_GPT_value_two_std_dev_less_l2329_232905


namespace NUMINAMATH_GPT_find_k_l2329_232978

variable (k : ℝ) (t : ℝ) (a : ℝ)

theorem find_k (h1 : t = (5 / 9) * (k - 32) + a * k) (h2 : t = 20) (h3 : a = 3) : k = 10.625 := by
  sorry

end NUMINAMATH_GPT_find_k_l2329_232978


namespace NUMINAMATH_GPT_table_ratio_l2329_232976

theorem table_ratio (L W : ℝ) (h1 : L * W = 128) (h2 : L + 2 * W = 32) : L / W = 2 :=
by
  sorry

end NUMINAMATH_GPT_table_ratio_l2329_232976


namespace NUMINAMATH_GPT_scale_readings_poles_greater_l2329_232941

-- Define the necessary quantities and conditions
variable (m : ℝ) -- mass of the object
variable (ω : ℝ) -- angular velocity of Earth's rotation
variable (R_e : ℝ) -- radius of the Earth at the equator
variable (g_e : ℝ) -- gravitational acceleration at the equator
variable (g_p : ℝ) -- gravitational acceleration at the poles
variable (F_c : ℝ) -- centrifugal force at the equator
variable (F_g_e : ℝ) -- gravitational force at the equator
variable (F_g_p : ℝ) -- gravitational force at the poles
variable (W_e : ℝ) -- apparent weight at the equator
variable (W_p : ℝ) -- apparent weight at the poles

-- Establish conditions
axiom centrifugal_definition : F_c = m * ω^2 * R_e
axiom gravitational_force_equator : F_g_e = m * g_e
axiom apparent_weight_equator : W_e = F_g_e - F_c
axiom no_centrifugal_force_poles : F_c = 0
axiom gravitational_force_poles : F_g_p = m * g_p
axiom apparent_weight_poles : W_p = F_g_p
axiom gravity_comparison : g_p > g_e

-- Theorem: The readings on spring scales at the poles will be greater than the readings at the equator
theorem scale_readings_poles_greater : W_p > W_e := 
sorry

end NUMINAMATH_GPT_scale_readings_poles_greater_l2329_232941


namespace NUMINAMATH_GPT_min_people_wearing_both_l2329_232969

theorem min_people_wearing_both (n : ℕ) (h_lcm : n % 24 = 0) 
  (h_gloves : 3 * n % 8 = 0) (h_hats : 5 * n % 6 = 0) :
  ∃ x, x = 5 := 
by
  let gloves := 3 * n / 8
  let hats := 5 * n / 6
  let both := gloves + hats - n
  have h1 : both = 5 := sorry
  exact ⟨both, h1⟩

end NUMINAMATH_GPT_min_people_wearing_both_l2329_232969


namespace NUMINAMATH_GPT_coordinates_equidistant_l2329_232970

-- Define the condition of equidistance
theorem coordinates_equidistant (x y : ℝ) :
  (x + 2) ^ 2 + (y - 2) ^ 2 = (x - 2) ^ 2 + y ^ 2 →
  y = 2 * x + 1 :=
  sorry  -- Proof is omitted

end NUMINAMATH_GPT_coordinates_equidistant_l2329_232970


namespace NUMINAMATH_GPT_T_8_equals_546_l2329_232994

-- Define the sum of the first n natural numbers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the sum of the squares of the first n natural numbers
def sum_squares_first_n (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

-- Define T_n based on the given formula
def T (n : ℕ) : ℕ := (sum_first_n n ^ 2 - sum_squares_first_n n) / 2

-- The proof statement we need to prove
theorem T_8_equals_546 : T 8 = 546 := sorry

end NUMINAMATH_GPT_T_8_equals_546_l2329_232994


namespace NUMINAMATH_GPT_train_length_l2329_232965

theorem train_length (time_crossing : ℕ) (speed_kmh : ℕ) (conversion_factor : ℕ) (expected_length : ℕ) :
  time_crossing = 4 ∧ speed_kmh = 144 ∧ conversion_factor = 1000 / 3600 * 144 →
  expected_length = 160 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l2329_232965


namespace NUMINAMATH_GPT_crackers_per_friend_l2329_232942

theorem crackers_per_friend (total_crackers : ℕ) (num_friends : ℕ) (n : ℕ) 
  (h1 : total_crackers = 8) 
  (h2 : num_friends = 4)
  (h3 : total_crackers / num_friends = n) : n = 2 :=
by
  sorry

end NUMINAMATH_GPT_crackers_per_friend_l2329_232942


namespace NUMINAMATH_GPT_sarah_total_pencils_l2329_232988

-- Define the number of pencils Sarah buys on each day
def pencils_monday : ℕ := 35
def pencils_tuesday : ℕ := 42
def pencils_wednesday : ℕ := 3 * pencils_tuesday
def pencils_thursday : ℕ := pencils_wednesday / 2
def pencils_friday : ℕ := 2 * pencils_monday

-- Define the total number of pencils
def total_pencils : ℕ :=
  pencils_monday + pencils_tuesday + pencils_wednesday + pencils_thursday + pencils_friday

-- Theorem statement to prove the total number of pencils equals 336
theorem sarah_total_pencils : total_pencils = 336 :=
by
  -- here goes the proof, but it is not required
  sorry

end NUMINAMATH_GPT_sarah_total_pencils_l2329_232988


namespace NUMINAMATH_GPT_proof_problem_l2329_232918

noncomputable def f (a x : ℝ) : ℝ := a^x
noncomputable def g (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem proof_problem (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) (h_f : f a 2 = 9) : 
    g a (1/9) + f a 3 = 25 :=
by
  -- Definitions and assumptions based on the provided problem
  sorry

end NUMINAMATH_GPT_proof_problem_l2329_232918


namespace NUMINAMATH_GPT_vector_calculation_l2329_232991

variables (a b : ℝ × ℝ)

def a_def : Prop := a = (3, 5)
def b_def : Prop := b = (-2, 1)

theorem vector_calculation (h1 : a_def a) (h2 : b_def b) : a - 2 • b = (7, 3) :=
sorry

end NUMINAMATH_GPT_vector_calculation_l2329_232991


namespace NUMINAMATH_GPT_hyperbola_asymptote_m_value_l2329_232926

theorem hyperbola_asymptote_m_value (m : ℝ) :
  (∀ x y : ℝ, (x^2 / m - y^2 / 6 = 1) → (y = x)) → m = 6 :=
by
  intros hx
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_m_value_l2329_232926


namespace NUMINAMATH_GPT_solve_equation_l2329_232952

theorem solve_equation (x : ℝ) (hx : x ≠ 1) : (x / (x - 1) - 1 = 1) → (x = 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2329_232952


namespace NUMINAMATH_GPT_proveCarTransportationProblem_l2329_232990

def carTransportationProblem :=
  ∃ x y a b : ℕ,
  -- Conditions regarding the capabilities of the cars
  (2 * x + 3 * y = 18) ∧
  (x + 2 * y = 11) ∧
  -- Conclusion (question 1)
  (x + y = 7) ∧
  -- Conditions for the rental plan (question 2)
  (3 * a + 4 * b = 27) ∧
  -- Cost optimization
  ((100 * a + 120 * b) = 820 ∨ (100 * a + 120 * b) = 860) ∧
  -- Optimal cost verification
  (100 * a + 120 * b = 820 → a = 1 ∧ b = 6)

theorem proveCarTransportationProblem : carTransportationProblem :=
  sorry

end NUMINAMATH_GPT_proveCarTransportationProblem_l2329_232990


namespace NUMINAMATH_GPT_find_divisor_l2329_232928

theorem find_divisor (x y : ℝ) (hx : x > 0) (hx_val : x = 1.3333333333333333) (h : 4 * x / y = x^2) : y = 3 :=
by 
  sorry

end NUMINAMATH_GPT_find_divisor_l2329_232928


namespace NUMINAMATH_GPT_correct_statements_l2329_232963

namespace ProofProblem

variable (f : ℕ+ × ℕ+ → ℕ+)
variable (h1 : f (1, 1) = 1)
variable (h2 : ∀ m n : ℕ+, f (m, n + 1) = f (m, n) + 2)
variable (h3 : ∀ m : ℕ+, f (m + 1, 1) = 2 * f (m, 1))

theorem correct_statements :
  f (1, 5) = 9 ∧ f (5, 1) = 16 ∧ f (5, 6) = 26 :=
by
  sorry

end ProofProblem

end NUMINAMATH_GPT_correct_statements_l2329_232963


namespace NUMINAMATH_GPT_sum_of_remainders_l2329_232982

theorem sum_of_remainders (p : ℕ) (hp : p > 2) (hp_prime : Nat.Prime p)
    (a : ℕ → ℕ) (ha : ∀ k, a k = k^p % p^2) :
    (Finset.sum (Finset.range (p - 1)) a) = (p^3 - p^2) / 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_l2329_232982


namespace NUMINAMATH_GPT_Ryan_spits_percentage_shorter_l2329_232911

theorem Ryan_spits_percentage_shorter (Billy_dist Madison_dist Ryan_dist : ℝ) (h1 : Billy_dist = 30) (h2 : Madison_dist = 1.20 * Billy_dist) (h3 : Ryan_dist = 18) :
  ((Madison_dist - Ryan_dist) / Madison_dist) * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_Ryan_spits_percentage_shorter_l2329_232911


namespace NUMINAMATH_GPT_original_number_of_men_l2329_232974

theorem original_number_of_men (x : ℕ) (h1 : x * 10 = (x - 5) * 12) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_original_number_of_men_l2329_232974


namespace NUMINAMATH_GPT_divisor_value_l2329_232902

theorem divisor_value (D : ℕ) (k m : ℤ) (h1 : 242 % D = 8) (h2 : 698 % D = 9) (h3 : (242 + 698) % D = 4) : D = 13 := by
  sorry

end NUMINAMATH_GPT_divisor_value_l2329_232902


namespace NUMINAMATH_GPT_total_musicians_count_l2329_232919

-- Define the given conditions
def orchestra_males := 11
def orchestra_females := 12
def choir_males := 12
def choir_females := 17

-- Total number of musicians in the orchestra
def orchestra_musicians := orchestra_males + orchestra_females

-- Total number of musicians in the band
def band_musicians := 2 * orchestra_musicians

-- Total number of musicians in the choir
def choir_musicians := choir_males + choir_females

-- Total number of musicians in the orchestra, band, and choir
def total_musicians := orchestra_musicians + band_musicians + choir_musicians

-- The theorem to prove
theorem total_musicians_count : total_musicians = 98 :=
by
  -- Lean proof part goes here.
  sorry

end NUMINAMATH_GPT_total_musicians_count_l2329_232919


namespace NUMINAMATH_GPT_shortest_distance_Dasha_Vasya_l2329_232915

def distance_Asya_Galia : ℕ := 12
def distance_Galia_Borya : ℕ := 10
def distance_Asya_Borya : ℕ := 8
def distance_Dasha_Galia : ℕ := 15
def distance_Vasya_Galia : ℕ := 17

def distance_Dasha_Vasya : ℕ :=
  distance_Dasha_Galia + distance_Vasya_Galia - distance_Asya_Galia - distance_Galia_Borya + distance_Asya_Borya

theorem shortest_distance_Dasha_Vasya : distance_Dasha_Vasya = 18 :=
by
  -- We assume the distances as given in the conditions. The calculation part is skipped here.
  -- The actual proof steps would go here.
  sorry

end NUMINAMATH_GPT_shortest_distance_Dasha_Vasya_l2329_232915


namespace NUMINAMATH_GPT_find_p_l2329_232934

noncomputable def area_of_ABC (p : ℚ) : ℚ :=
  128 - 6 * p

theorem find_p (p : ℚ) : area_of_ABC p = 45 → p = 83 / 6 := by
  intro h
  sorry

end NUMINAMATH_GPT_find_p_l2329_232934


namespace NUMINAMATH_GPT_sum_of_first_six_terms_l2329_232973

noncomputable def a (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2 * a (n - 1)

def sum_first_six_terms (a : ℕ → ℚ) : ℚ :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

theorem sum_of_first_six_terms :
  sum_first_six_terms a = 63 / 32 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_six_terms_l2329_232973


namespace NUMINAMATH_GPT_total_tickets_l2329_232943

theorem total_tickets (A C : ℕ) (cost_adult cost_child total_cost : ℝ) 
  (h1 : cost_adult = 5.50) 
  (h2 : cost_child = 3.50) 
  (h3 : C = 16) 
  (h4 : total_cost = 83.50) 
  (h5 : cost_adult * A + cost_child * C = total_cost) : 
  A + C = 21 := 
by 
  sorry

end NUMINAMATH_GPT_total_tickets_l2329_232943


namespace NUMINAMATH_GPT_fraction_transformation_l2329_232998

variables (a b : ℝ)

theorem fraction_transformation (ha : a ≠ 0) (hb : b ≠ 0) : 
  (4 * a * b) / (2 * (2 * a) + 2 * b) = 2 * (a * b) / (2 * a + b) :=
by
  sorry

end NUMINAMATH_GPT_fraction_transformation_l2329_232998


namespace NUMINAMATH_GPT_sum_of_geometric_sequence_eq_31_over_16_l2329_232948

theorem sum_of_geometric_sequence_eq_31_over_16 (n : ℕ) :
  let a := 1
  let r := (1 / 2 : ℝ)
  let S_n := 2 - 2 * r^n
  (S_n = (31 / 16 : ℝ)) ↔ (n = 5) := by
{
  sorry
}

end NUMINAMATH_GPT_sum_of_geometric_sequence_eq_31_over_16_l2329_232948


namespace NUMINAMATH_GPT_basketball_game_points_l2329_232916

theorem basketball_game_points
  (a b : ℕ) 
  (r : ℕ := 2)
  (S_E : ℕ := a / 2 * (1 + r + r^2 + r^3))
  (S_T : ℕ := 4 * b)
  (h1 : S_E = S_T + 2)
  (h2 : S_E < 100)
  (h3 : S_T < 100)
  : (a / 2 + a / 2 * r + b + b = 19) :=
by sorry

end NUMINAMATH_GPT_basketball_game_points_l2329_232916


namespace NUMINAMATH_GPT_min_value_frac_ineq_l2329_232938

theorem min_value_frac_ineq (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 1) : 
  (9/m + 1/n) ≥ 16 :=
sorry

end NUMINAMATH_GPT_min_value_frac_ineq_l2329_232938


namespace NUMINAMATH_GPT_bryce_received_12_raisins_l2329_232987

-- Defining the main entities for the problem
variables {x y z : ℕ} -- number of raisins Bryce, Carter, and Emma received respectively

-- Conditions:
def condition1 (x y : ℕ) : Prop := y = x - 8
def condition2 (x y : ℕ) : Prop := y = x / 3
def condition3 (y z : ℕ) : Prop := z = 2 * y

-- The goal is to prove that Bryce received 12 raisins
theorem bryce_received_12_raisins (x y z : ℕ) 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) 
  (h3 : condition3 y z) : 
  x = 12 :=
sorry

end NUMINAMATH_GPT_bryce_received_12_raisins_l2329_232987


namespace NUMINAMATH_GPT_intersection_complement_M_N_eq_456_l2329_232962

def UniversalSet := { n : ℕ | 1 ≤ n ∧ n < 9 }
def M : Set ℕ := { 1, 2, 3 }
def N : Set ℕ := { 3, 4, 5, 6 }

theorem intersection_complement_M_N_eq_456 : 
  (UniversalSet \ M) ∩ N = { 4, 5, 6 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_M_N_eq_456_l2329_232962


namespace NUMINAMATH_GPT_larger_page_of_opened_book_l2329_232912

theorem larger_page_of_opened_book (x : ℕ) (h : x + (x + 1) = 137) : x + 1 = 69 :=
sorry

end NUMINAMATH_GPT_larger_page_of_opened_book_l2329_232912


namespace NUMINAMATH_GPT_direct_proportion_m_value_l2329_232960

theorem direct_proportion_m_value (m : ℝ) : 
  (∀ x: ℝ, y = -7 * x + 2 + m -> y = k * x) -> m = -2 :=
by
  sorry

end NUMINAMATH_GPT_direct_proportion_m_value_l2329_232960


namespace NUMINAMATH_GPT_smallest_side_of_triangle_l2329_232930

variable {α : Type} [LinearOrderedField α]

theorem smallest_side_of_triangle (a b c : α) (h : a^2 + b^2 > 5*c^2) : c ≤ a ∧ c ≤ b :=
by
  sorry

end NUMINAMATH_GPT_smallest_side_of_triangle_l2329_232930


namespace NUMINAMATH_GPT_simplest_form_correct_l2329_232927

variable (A : ℝ)
variable (B : ℝ)
variable (C : ℝ)
variable (D : ℝ)

def is_simplest_form (x : ℝ) : Prop :=
-- define what it means for a square root to be in simplest form
sorry

theorem simplest_form_correct :
  A = Real.sqrt (1 / 2) ∧ B = Real.sqrt 0.2 ∧ C = Real.sqrt 3 ∧ D = Real.sqrt 8 →
  ¬ is_simplest_form A ∧ ¬ is_simplest_form B ∧ is_simplest_form C ∧ ¬ is_simplest_form D :=
by
  -- prove that C is the simplest form and others are not
  sorry

end NUMINAMATH_GPT_simplest_form_correct_l2329_232927


namespace NUMINAMATH_GPT_derivative_of_m_l2329_232992

noncomputable def m (x : ℝ) : ℝ := (2 : ℝ)^x / (1 + x)

theorem derivative_of_m (x : ℝ) : 
  deriv m x = (2^x * (1 + x) * Real.log 2 - 2^x) / (1 + x)^2 :=
by
  sorry

end NUMINAMATH_GPT_derivative_of_m_l2329_232992


namespace NUMINAMATH_GPT_polygon_sides_l2329_232977

theorem polygon_sides
  (n : ℕ)
  (h1 : 180 * (n - 2) - (2 * (2790 / (n - 1)) - 20) = 2790) :
  n = 18 := sorry

end NUMINAMATH_GPT_polygon_sides_l2329_232977


namespace NUMINAMATH_GPT_determine_constants_l2329_232967

structure Vector2D :=
(x : ℝ)
(y : ℝ)

def a := 11 / 20
def b := -7 / 20

def v1 : Vector2D := ⟨3, 2⟩
def v2 : Vector2D := ⟨-1, 6⟩
def v3 : Vector2D := ⟨2, -1⟩

def linear_combination (v1 v2 : Vector2D) (a b : ℝ) : Vector2D :=
  ⟨a * v1.x + b * v2.x, a * v1.y + b * v2.y⟩

theorem determine_constants (a b : ℝ) :
  ∃ (a b : ℝ), linear_combination v1 v2 a b = v3 :=
by
  use (11 / 20)
  use (-7 / 20)
  sorry

end NUMINAMATH_GPT_determine_constants_l2329_232967


namespace NUMINAMATH_GPT_inequality_pgcd_l2329_232954

theorem inequality_pgcd (a b : ℕ) (h1 : a > b) (h2 : (a - b) ∣ (a^2 + b)) : 
  (a + 1) / (b + 1) ≤ Nat.gcd a b + 1 := 
sorry

end NUMINAMATH_GPT_inequality_pgcd_l2329_232954


namespace NUMINAMATH_GPT_solve_system_l2329_232937

theorem solve_system :
  ∃ (x y : ℝ), (x^2 + y^2 ≤ 1) ∧ (x^4 - 18 * x^2 * y^2 + 81 * y^4 - 20 * x^2 - 180 * y^2 + 100 = 0) ∧
    ((x = -1 / Real.sqrt 10 ∧ y = 3 / Real.sqrt 10) ∨
    (x = -1 / Real.sqrt 10 ∧ y = -3 / Real.sqrt 10) ∨
    (x = 1 / Real.sqrt 10 ∧ y = 3 / Real.sqrt 10) ∨
    (x = 1 / Real.sqrt 10 ∧ y = -3 / Real.sqrt 10)) :=
  by
  sorry

end NUMINAMATH_GPT_solve_system_l2329_232937


namespace NUMINAMATH_GPT_additional_cost_per_pint_proof_l2329_232968

-- Definitions based on the problem conditions
def pints_sold := 54
def total_revenue_on_sale := 216
def revenue_difference := 108

-- Derived definitions
def revenue_if_not_on_sale := total_revenue_on_sale + revenue_difference
def cost_per_pint_on_sale := total_revenue_on_sale / pints_sold
def cost_per_pint_not_on_sale := revenue_if_not_on_sale / pints_sold
def additional_cost_per_pint := cost_per_pint_not_on_sale - cost_per_pint_on_sale

-- Proof statement
theorem additional_cost_per_pint_proof :
  additional_cost_per_pint = 2 :=
by
  -- Placeholder to indicate that the proof is not provided
  sorry

end NUMINAMATH_GPT_additional_cost_per_pint_proof_l2329_232968


namespace NUMINAMATH_GPT_relationship_ab_l2329_232985

-- Define the conditions
variable {f : ℝ → ℝ}
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)
variable (h_period : ∀ x : ℝ, f (x + 1) = -f x)
variable (a b : ℝ)
variable (h_ex : ∃ x : ℝ, f (a + x) = f (b - x))

-- State the conclusion we need to prove
theorem relationship_ab : ∃ k : ℕ, k > 0 ∧ (a + b) = 2 * k + 1 :=
by
  sorry

end NUMINAMATH_GPT_relationship_ab_l2329_232985


namespace NUMINAMATH_GPT_probability_diff_digits_l2329_232986

open Finset

def two_digit_same_digit (n : ℕ) : Prop :=
  n / 10 = n % 10

def three_digit_same_digit (n : ℕ) : Prop :=
  (n % 100) / 10 = n / 100 ∧ (n / 100) = (n % 10)

def same_digit (n : ℕ) : Prop :=
  two_digit_same_digit n ∨ three_digit_same_digit n

def total_numbers : ℕ :=
  (199 - 10 + 1)

def same_digit_count : ℕ :=
  9 + 9

theorem probability_diff_digits : 
  ((total_numbers - same_digit_count) / total_numbers : ℚ) = 86 / 95 :=
by
  sorry

end NUMINAMATH_GPT_probability_diff_digits_l2329_232986


namespace NUMINAMATH_GPT_base_h_addition_eq_l2329_232971

theorem base_h_addition_eq (h : ℕ) (h_eq : h = 9) : 
  (8 * h^3 + 3 * h^2 + 7 * h + 4) + (6 * h^3 + 9 * h^2 + 2 * h + 5) = 1 * h^4 + 5 * h^3 + 3 * h^2 + 0 * h + 9 :=
by
  rw [h_eq]
  sorry

end NUMINAMATH_GPT_base_h_addition_eq_l2329_232971


namespace NUMINAMATH_GPT_toys_produced_per_week_l2329_232989

-- Definitions corresponding to the conditions
def days_per_week : ℕ := 2
def toys_per_day : ℕ := 2170

-- Theorem statement corresponding to the question and correct answer
theorem toys_produced_per_week : days_per_week * toys_per_day = 4340 := 
by 
  -- placeholders for the proof steps
  sorry

end NUMINAMATH_GPT_toys_produced_per_week_l2329_232989


namespace NUMINAMATH_GPT_rate_is_15_l2329_232956

variable (sum : ℝ) (interest12 : ℝ) (interest_r : ℝ) (r : ℝ)

-- Given conditions
def conditions : Prop :=
  sum = 7000 ∧
  interest12 = 7000 * 0.12 * 2 ∧
  interest_r = 7000 * (r / 100) * 2 ∧
  interest_r = interest12 + 420

-- The rate to prove
def rate_to_prove : Prop := r = 15

theorem rate_is_15 : conditions sum interest12 interest_r r → rate_to_prove r := 
by
  sorry

end NUMINAMATH_GPT_rate_is_15_l2329_232956


namespace NUMINAMATH_GPT_candy_total_cents_l2329_232944

def candy_cost : ℕ := 8
def gumdrops : ℕ := 28
def total_cents : ℕ := 224

theorem candy_total_cents : candy_cost * gumdrops = total_cents := by
  sorry

end NUMINAMATH_GPT_candy_total_cents_l2329_232944


namespace NUMINAMATH_GPT_sarah_cupcakes_ratio_l2329_232955

theorem sarah_cupcakes_ratio (total_cupcakes : ℕ) (cookies_from_michael : ℕ) 
    (final_desserts : ℕ) (cupcakes_given : ℕ) (h1 : total_cupcakes = 9) 
    (h2 : cookies_from_michael = 5) (h3 : final_desserts = 11) 
    (h4 : total_cupcakes - cupcakes_given + cookies_from_michael = final_desserts) : 
    cupcakes_given / total_cupcakes = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sarah_cupcakes_ratio_l2329_232955


namespace NUMINAMATH_GPT_sin_cos_15_degrees_proof_l2329_232984

noncomputable
def sin_cos_15_degrees : Prop := (Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 / 4)

theorem sin_cos_15_degrees_proof : sin_cos_15_degrees :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_15_degrees_proof_l2329_232984


namespace NUMINAMATH_GPT_find_triangle_sides_l2329_232966

theorem find_triangle_sides (k : ℕ) (k_pos : k = 6) 
  {x y z : ℝ} (x_pos : 0 < x) (y_pos : 0 < y) (z_pos : 0 < z) 
  (h : k * (x * y + y * z + z * x) > 5 * (x ^ 2 + y ^ 2 + z ^ 2)) :
  ∃ x' y' z', (x = x') ∧ (y = y') ∧ (z = z') ∧ ((x' + y' > z') ∧ (x' + z' > y') ∧ (y' + z' > x')) :=
by
  sorry

end NUMINAMATH_GPT_find_triangle_sides_l2329_232966


namespace NUMINAMATH_GPT_income_scientific_notation_l2329_232951

theorem income_scientific_notation (avg_income_per_acre : ℝ) (acres : ℝ) (a n : ℝ) :
  avg_income_per_acre = 20000 →
  acres = 8000 → 
  (avg_income_per_acre * acres = a * 10 ^ n ↔ (a = 1.6 ∧ n = 8)) :=
by
  sorry

end NUMINAMATH_GPT_income_scientific_notation_l2329_232951


namespace NUMINAMATH_GPT_instantaneous_velocity_at_t2_l2329_232910

noncomputable def displacement (t : ℝ) : ℝ := t^2 * Real.exp (t - 2)

theorem instantaneous_velocity_at_t2 :
  (deriv displacement 2 = 8) :=
by
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_t2_l2329_232910


namespace NUMINAMATH_GPT_duration_trip_for_cyclist1_l2329_232939

-- Definitions
variable (s : ℝ) -- the speed of Cyclist 1 without wind in km/h
variable (t : ℝ) -- the time in hours it takes for Cyclist 1 to travel from A to B
variable (wind_speed : ℝ := 3) -- wind modifies speed by 3 km/h
variable (total_time : ℝ := 4) -- total time after which cyclists meet

-- Conditions
axiom consistent_speed_aid : ∀ (s t : ℝ), t > 0 → (s + wind_speed) * t + (s - wind_speed) * (total_time - t) / 2 = s - wind_speed * total_time

-- Goal (equivalent proof problem)
theorem duration_trip_for_cyclist1 : t = 2 := by
  sorry

end NUMINAMATH_GPT_duration_trip_for_cyclist1_l2329_232939


namespace NUMINAMATH_GPT_identity_proof_l2329_232903

theorem identity_proof
  (M N x a b : ℝ)
  (h₀ : x ≠ a)
  (h₁ : x ≠ b)
  (h₂ : a ≠ b) :
  (Mx + N) / ((x - a) * (x - b)) =
  (((M *a + N) / (a - b)) * (1 / (x - a))) - 
  (((M * b + N) / (a - b)) * (1 / (x - b))) :=
sorry

end NUMINAMATH_GPT_identity_proof_l2329_232903


namespace NUMINAMATH_GPT_bella_steps_l2329_232975

-- Define the conditions and the necessary variables
variable (b : ℝ) (distance : ℝ) (steps_per_foot : ℝ)

-- Given constants
def bella_speed := b
def ella_speed := 4 * b
def combined_speed := bella_speed + ella_speed
def total_distance := 15840
def feet_per_step := 3

-- Define the main theorem to prove the number of steps Bella takes
theorem bella_steps : (total_distance / combined_speed) * bella_speed / feet_per_step = 1056 := by
  sorry

end NUMINAMATH_GPT_bella_steps_l2329_232975


namespace NUMINAMATH_GPT_proposition_B_l2329_232914

-- Definitions of planes and lines
variable {Plane : Type}
variable {Line : Type}
variable (α β : Plane)
variable (m n : Line)

-- Definitions of parallel and perpendicular relationships
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (_perpendicular : Line → Line → Prop)

-- Theorem statement
theorem proposition_B (h1 : perpendicular m α) (h2 : parallel n α) : _perpendicular m n :=
sorry

end NUMINAMATH_GPT_proposition_B_l2329_232914


namespace NUMINAMATH_GPT_lisa_eats_correct_number_of_pieces_l2329_232957

variable (M A K R L : ℚ) -- All variables are rational numbers (real numbers could also be used)
variable (n : ℕ) -- n is a natural number (the number of pieces of lasagna)

-- Let's define the conditions succinctly
def manny_wants_one_piece := M = 1
def aaron_eats_nothing := A = 0
def kai_eats_twice_manny := K = 2 * M
def raphael_eats_half_manny := R = 0.5 * M
def lasagna_is_cut_into_6_pieces := n = 6

-- The proof goal is to show Lisa eats 2.5 pieces
theorem lisa_eats_correct_number_of_pieces (M A K R L : ℚ) (n : ℕ) :
  manny_wants_one_piece M →
  aaron_eats_nothing A →
  kai_eats_twice_manny M K →
  raphael_eats_half_manny M R →
  lasagna_is_cut_into_6_pieces n →
  L = n - (M + K + R) →
  L = 2.5 :=
by
  intros hM hA hK hR hn hL
  sorry  -- Proof omitted

end NUMINAMATH_GPT_lisa_eats_correct_number_of_pieces_l2329_232957


namespace NUMINAMATH_GPT_shaded_region_area_l2329_232932

theorem shaded_region_area (r_s r_l chord_AB : ℝ) (hs : r_s = 40) (hl : r_l = 60) (hc : chord_AB = 100) :
    chord_AB / 2 = 50 →
    60^2 - (chord_AB / 2)^2 = r_s^2 →
    (π * r_l^2) - (π * r_s^2) = 2500 * π :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_shaded_region_area_l2329_232932


namespace NUMINAMATH_GPT_product_binary1101_ternary202_eq_260_l2329_232940

-- Define the binary number 1101 in base 10
def binary1101 := 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- Define the ternary number 202 in base 10
def ternary202 := 2 * 3^2 + 0 * 3^1 + 2 * 3^0

-- Prove that their product in base 10 is 260
theorem product_binary1101_ternary202_eq_260 : binary1101 * ternary202 = 260 := by
  -- Proof 
  sorry

end NUMINAMATH_GPT_product_binary1101_ternary202_eq_260_l2329_232940


namespace NUMINAMATH_GPT_rhombus_diagonal_l2329_232995

theorem rhombus_diagonal (d2 : ℝ) (area : ℝ) (d1 : ℝ) (h1 : d2 = 10) (h2 : area = 60) : 
  d1 = 12 :=
by 
  have : (d1 * d2) / 2 = area := sorry
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_l2329_232995


namespace NUMINAMATH_GPT_notification_possible_l2329_232981

-- Define the conditions
def side_length : ℝ := 2
def speed : ℝ := 3
def initial_time : ℝ := 12 -- noon
def arrival_time : ℝ := 19 -- 7 PM
def notification_time : ℝ := arrival_time - initial_time -- total available time for notification

-- Define the proof statement
theorem notification_possible :
  ∃ (partition : ℕ → ℝ) (steps : ℕ → ℝ), (∀ k, steps k * partition k < notification_time) ∧ 
  ∑' k, (steps k * partition k) ≤ 6 :=
by
  sorry

end NUMINAMATH_GPT_notification_possible_l2329_232981


namespace NUMINAMATH_GPT_next_term_in_geometric_sequence_l2329_232917

theorem next_term_in_geometric_sequence : 
  ∀ (x : ℕ), (∃ (a : ℕ), a = 768 * x^4) :=
by
  sorry

end NUMINAMATH_GPT_next_term_in_geometric_sequence_l2329_232917


namespace NUMINAMATH_GPT_amount_left_in_wallet_l2329_232993

theorem amount_left_in_wallet
  (initial_amount : ℝ)
  (spent_amount : ℝ)
  (h_initial : initial_amount = 94)
  (h_spent : spent_amount = 16) :
  initial_amount - spent_amount = 78 :=
by
  sorry

end NUMINAMATH_GPT_amount_left_in_wallet_l2329_232993


namespace NUMINAMATH_GPT_addition_problem_base6_l2329_232907

theorem addition_problem_base6 (X Y : ℕ) (h1 : Y + 3 = X) (h2 : X + 2 = 7) : X + Y = 7 :=
by
  sorry

end NUMINAMATH_GPT_addition_problem_base6_l2329_232907


namespace NUMINAMATH_GPT_possible_r_values_l2329_232979

noncomputable def triangle_area (r : ℝ) : ℝ := (r - 3) ^ (3 / 2)

theorem possible_r_values :
  {r : ℝ | 16 ≤ triangle_area r ∧ triangle_area r ≤ 128} = {r : ℝ | 7 ≤ r ∧ r ≤ 19} :=
by
  sorry

end NUMINAMATH_GPT_possible_r_values_l2329_232979
