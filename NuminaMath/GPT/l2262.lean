import Mathlib

namespace NUMINAMATH_GPT_reciprocal_of_2016_is_1_div_2016_l2262_226203

theorem reciprocal_of_2016_is_1_div_2016 : (2016 * (1 / 2016) = 1) :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_2016_is_1_div_2016_l2262_226203


namespace NUMINAMATH_GPT_translation_proof_l2262_226288

-- Define the points and the translation process
def point_A : ℝ × ℝ := (-1, 0)
def point_B : ℝ × ℝ := (1, 2)
def point_C : ℝ × ℝ := (1, -2)

-- Translation from point A to point C
def translation_vector : ℝ × ℝ :=
  (point_C.1 - point_A.1, point_C.2 - point_A.2)

-- Define point D using the translation vector applied to point B
def point_D : ℝ × ℝ :=
  (point_B.1 + translation_vector.1, point_B.2 + translation_vector.2)

-- Statement to prove point D has the expected coordinates
theorem translation_proof : 
  point_D = (3, 0) :=
by 
  -- The exact proof is omitted, presented here for completion
  sorry

end NUMINAMATH_GPT_translation_proof_l2262_226288


namespace NUMINAMATH_GPT_sum_of_a_for_one_solution_l2262_226255

theorem sum_of_a_for_one_solution (a : ℝ) :
  (∀ x : ℝ, 3 * x^2 + (a + 15) * x + 18 = 0 ↔ (a + 15) ^ 2 - 4 * 3 * 18 = 0) →
  a = -15 + 6 * Real.sqrt 6 ∨ a = -15 - 6 * Real.sqrt 6 → a + (-15 + 6 * Real.sqrt 6) + (-15 - 6 * Real.sqrt 6) = -30 :=
by
  intros h1 h2
  have hsum : (-15 + 6 * Real.sqrt 6) + (-15 - 6 * Real.sqrt 6) = -30 := by linarith [Real.sqrt 6]
  sorry

end NUMINAMATH_GPT_sum_of_a_for_one_solution_l2262_226255


namespace NUMINAMATH_GPT_solve_equation_l2262_226271

theorem solve_equation : 
  ∀ x : ℝ, 
  (((15 * x - x^2) / (x + 2)) * (x + (15 - x) / (x + 2)) = 54) → (x = 9 ∨ x = -1) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2262_226271


namespace NUMINAMATH_GPT_calculate_total_marks_l2262_226297

theorem calculate_total_marks 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (marks_per_correct : ℕ) 
  (marks_per_wrong : ℤ) 
  (total_attempted : total_questions = 60) 
  (correct_attempted : correct_answers = 44)
  (marks_per_correct_is_4 : marks_per_correct = 4)
  (marks_per_wrong_is_neg1 : marks_per_wrong = -1) : 
  total_questions * marks_per_correct - (total_questions - correct_answers) * (abs marks_per_wrong) = 160 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_total_marks_l2262_226297


namespace NUMINAMATH_GPT_polynomial_root_recip_squares_l2262_226202

theorem polynomial_root_recip_squares (a b c : ℝ) 
  (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 11) (h3 : a * b * c = 6):
  1 / a^2 + 1 / b^2 + 1 / c^2 = 49 / 36 :=
sorry

end NUMINAMATH_GPT_polynomial_root_recip_squares_l2262_226202


namespace NUMINAMATH_GPT_value_of_x_plus_y_squared_l2262_226217

variable (x y : ℝ)

def condition1 : Prop := x * (x + y) = 40
def condition2 : Prop := y * (x + y) = 90
def condition3 : Prop := x - y = 5

theorem value_of_x_plus_y_squared (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 x y) : (x + y) ^ 2 = 130 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_plus_y_squared_l2262_226217


namespace NUMINAMATH_GPT_swimming_speed_in_still_water_l2262_226287

/-- The speed (in km/h) of a man swimming in still water given the speed of the water current
    and the time taken to swim a certain distance against the current. -/
theorem swimming_speed_in_still_water (v : ℝ) (speed_water : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed_water = 12) (h2 : time = 5) (h3 : distance = 40)
  (h4 : time = distance / (v - speed_water)) : v = 20 :=
by
  sorry

end NUMINAMATH_GPT_swimming_speed_in_still_water_l2262_226287


namespace NUMINAMATH_GPT_least_number_remainder_l2262_226290

theorem least_number_remainder (n : ℕ) (h1 : n % 34 = 4) (h2 : n % 5 = 4) : n = 174 :=
  sorry

end NUMINAMATH_GPT_least_number_remainder_l2262_226290


namespace NUMINAMATH_GPT_min_airlines_needed_l2262_226243

theorem min_airlines_needed 
  (towns : Finset ℕ) 
  (h_towns : towns.card = 21)
  (flights : Π (a : Finset ℕ), a.card = 5 → Finset (Finset ℕ))
  (h_flight : ∀ {a : Finset ℕ} (ha : a.card = 5), (flights a ha).card = 10):
  ∃ (n : ℕ), n = 21 :=
sorry

end NUMINAMATH_GPT_min_airlines_needed_l2262_226243


namespace NUMINAMATH_GPT_total_cost_eq_898_80_l2262_226289

theorem total_cost_eq_898_80 (M R F : ℝ)
  (h1 : 10 * M = 24 * R)
  (h2 : 6 * F = 2 * R)
  (h3 : F = 21) :
  4 * M + 3 * R + 5 * F = 898.80 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_eq_898_80_l2262_226289


namespace NUMINAMATH_GPT_intersecting_circles_range_l2262_226229

theorem intersecting_circles_range {k : ℝ} (a b : ℝ) :
  (-36 : ℝ) ≤ k ∧ k ≤ 104 →
  (∃ (x y : ℝ), (x^2 + y^2 - 4 - 12 * x + 6 * y) = 0 ∧ (x^2 + y^2 = k + 4 * x + 12 * y)) →
  b - a = (140 : ℝ) :=
by
  intro hk hab
  sorry

end NUMINAMATH_GPT_intersecting_circles_range_l2262_226229


namespace NUMINAMATH_GPT_binom_12_3_equal_220_l2262_226252

theorem binom_12_3_equal_220 : Nat.choose 12 3 = 220 := by sorry

end NUMINAMATH_GPT_binom_12_3_equal_220_l2262_226252


namespace NUMINAMATH_GPT_necessary_condition_for_acute_angle_l2262_226248

-- Defining vectors a and b
def vec_a (x : ℝ) : ℝ × ℝ := (x - 3, 2)
def vec_b : ℝ × ℝ := (1, 1)

-- Condition for the dot product to be positive
def dot_product_positive (x : ℝ) : Prop :=
  let (ax1, ax2) := vec_a x
  let (bx1, bx2) := vec_b
  ax1 * bx1 + ax2 * bx2 > 0

-- Statement for necessary condition
theorem necessary_condition_for_acute_angle (x : ℝ) :
  (dot_product_positive x) → (1 < x) :=
sorry

end NUMINAMATH_GPT_necessary_condition_for_acute_angle_l2262_226248


namespace NUMINAMATH_GPT_add_neg_two_eq_zero_l2262_226273

theorem add_neg_two_eq_zero :
  (-2) + 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_add_neg_two_eq_zero_l2262_226273


namespace NUMINAMATH_GPT_larry_daily_dog_time_l2262_226291

-- Definitions from the conditions
def half_hour_in_minutes : ℕ := 30
def twice_a_day (minutes : ℕ) : ℕ := 2 * minutes
def one_fifth_hour_in_minutes : ℕ := 60 / 5

-- Hypothesis resulting from the conditions
def time_walking_and_playing : ℕ := twice_a_day half_hour_in_minutes
def time_feeding : ℕ := one_fifth_hour_in_minutes

-- The theorem to prove
theorem larry_daily_dog_time : time_walking_and_playing + time_feeding = 72 := by
  show time_walking_and_playing + time_feeding = 72
  sorry

end NUMINAMATH_GPT_larry_daily_dog_time_l2262_226291


namespace NUMINAMATH_GPT_max_value_frac_l2262_226216

theorem max_value_frac (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
    ∃ (c : ℝ), c = 1/4 ∧ (∀ (x y z : ℝ), (0 < x) → (0 < y) → (0 < z) → (xyz * (x + y + z)) / ((x + z)^2 * (z + y)^2) ≤ c) := 
by
  sorry

end NUMINAMATH_GPT_max_value_frac_l2262_226216


namespace NUMINAMATH_GPT_value_of_a_plus_b_l2262_226268

variables (a b : ℝ)

theorem value_of_a_plus_b (h1 : a + 4 * b = 33) (h2 : 6 * a + 3 * b = 51) : a + b = 12 := 
by
  sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l2262_226268


namespace NUMINAMATH_GPT_reggie_loses_by_21_points_l2262_226241

-- Define the points for each type of shot.
def layup_points := 1
def free_throw_points := 2
def three_pointer_points := 3
def half_court_points := 5

-- Define Reggie's shot counts.
def reggie_layups := 4
def reggie_free_throws := 3
def reggie_three_pointers := 2
def reggie_half_court_shots := 1

-- Define Reggie's brother's shot counts.
def brother_layups := 3
def brother_free_throws := 2
def brother_three_pointers := 5
def brother_half_court_shots := 4

-- Calculate Reggie's total points.
def reggie_total_points :=
  reggie_layups * layup_points +
  reggie_free_throws * free_throw_points +
  reggie_three_pointers * three_pointer_points +
  reggie_half_court_shots * half_court_points

-- Calculate Reggie's brother's total points.
def brother_total_points :=
  brother_layups * layup_points +
  brother_free_throws * free_throw_points +
  brother_three_pointers * three_pointer_points +
  brother_half_court_shots * half_court_points

-- Calculate the difference in points.
def point_difference := brother_total_points - reggie_total_points

-- Prove that the difference in points Reggie lost by is 21.
theorem reggie_loses_by_21_points : point_difference = 21 := by
  sorry

end NUMINAMATH_GPT_reggie_loses_by_21_points_l2262_226241


namespace NUMINAMATH_GPT_green_or_yellow_probability_l2262_226200

-- Given the number of marbles of each color
def green_marbles : ℕ := 4
def yellow_marbles : ℕ := 3
def white_marbles : ℕ := 6

-- The total number of marbles
def total_marbles : ℕ := green_marbles + yellow_marbles + white_marbles

-- The number of favorable outcomes (green or yellow marbles)
def favorable_marbles : ℕ := green_marbles + yellow_marbles

-- The probability of drawing a green or yellow marble as a fraction
def probability_of_green_or_yellow : Rat := favorable_marbles / total_marbles

theorem green_or_yellow_probability :
  probability_of_green_or_yellow = 7 / 13 :=
by
  sorry

end NUMINAMATH_GPT_green_or_yellow_probability_l2262_226200


namespace NUMINAMATH_GPT_arithmetic_sequence_tenth_term_l2262_226245

theorem arithmetic_sequence_tenth_term (a d : ℤ)
  (h1 : a + 2 * d = 5)
  (h2 : a + 6 * d = 13) :
  a + 9 * d = 19 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_tenth_term_l2262_226245


namespace NUMINAMATH_GPT_range_of_a_l2262_226213

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x - 2| ≤ a) ↔ a ≥ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2262_226213


namespace NUMINAMATH_GPT_original_number_l2262_226292

theorem original_number (x : ℝ) (h : 20 = 0.4 * (x - 5)) : x = 55 :=
sorry

end NUMINAMATH_GPT_original_number_l2262_226292


namespace NUMINAMATH_GPT_inequality_always_holds_l2262_226222

theorem inequality_always_holds (x b : ℝ) (h : ∀ x : ℝ, x^2 + b * x + b > 0) : 0 < b ∧ b < 4 :=
sorry

end NUMINAMATH_GPT_inequality_always_holds_l2262_226222


namespace NUMINAMATH_GPT_simplify_fraction_l2262_226262

theorem simplify_fraction (x : ℚ) : 
  (↑(x + 2) / 4 + ↑(3 - 4 * x) / 3 : ℚ) = ((-13 * x + 18) / 12 : ℚ) :=
by 
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2262_226262


namespace NUMINAMATH_GPT_find_divisor_l2262_226226

theorem find_divisor (x : ℝ) (h : 740 / x - 175 = 10) : x = 4 := by
  sorry

end NUMINAMATH_GPT_find_divisor_l2262_226226


namespace NUMINAMATH_GPT_aluminum_carbonate_weight_l2262_226219

-- Define the atomic weights
def Al : ℝ := 26.98
def C : ℝ := 12.01
def O : ℝ := 16.00

-- Define the molecular weight of aluminum carbonate
def molecularWeightAl2CO3 : ℝ := (2 * Al) + (3 * C) + (9 * O)

-- Define the number of moles
def moles : ℝ := 5

-- Calculate the total weight of 5 moles of aluminum carbonate
def totalWeight : ℝ := moles * molecularWeightAl2CO3

-- Statement to prove
theorem aluminum_carbonate_weight : totalWeight = 1169.95 :=
by {
  sorry
}

end NUMINAMATH_GPT_aluminum_carbonate_weight_l2262_226219


namespace NUMINAMATH_GPT_statement_B_not_true_l2262_226298

def diamondsuit (x y : ℝ) : ℝ := 2 * |(x - y)| + 1

theorem statement_B_not_true : ¬ (∀ x y : ℝ, 3 * diamondsuit x y = 3 * diamondsuit (2 * x) (2 * y)) :=
sorry

end NUMINAMATH_GPT_statement_B_not_true_l2262_226298


namespace NUMINAMATH_GPT_inequality_proof_l2262_226257

theorem inequality_proof (x1 x2 y1 y2 z1 z2 : ℝ) 
  (hx1 : x1 > 0) (hx2 : x2 > 0) (hy1 : y1 > 0) (hy2 : y2 > 0)
  (hx1y1_pos : x1 * y1 - z1^2 > 0) (hx2y2_pos : x2 * y2 - z2^2 > 0) :
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2)^2) ≤ 
    1 / (x1 * y1 - z1^2) + 1 / (x2 * y2 - z2^2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2262_226257


namespace NUMINAMATH_GPT_people_came_in_first_hour_l2262_226239
-- Import the entirety of the necessary library

-- Lean 4 statement for the given problem
theorem people_came_in_first_hour (X : ℕ) (net_change_first_hour : ℕ) (net_change_second_hour : ℕ) (people_after_2_hours : ℕ) : 
    (net_change_first_hour = X - 27) → 
    (net_change_second_hour = 18 - 9) →
    (people_after_2_hours = 76) → 
    (X - 27 + 9 = 76) → 
    X = 94 :=
by 
    intros h1 h2 h3 h4 
    sorry -- Proof is not required by instructions

end NUMINAMATH_GPT_people_came_in_first_hour_l2262_226239


namespace NUMINAMATH_GPT_relationship_between_x_and_y_l2262_226214

variables (x y : ℝ)

theorem relationship_between_x_and_y (h1 : x + y > 2 * x) (h2 : x - y < 2 * y) : y > x := 
sorry

end NUMINAMATH_GPT_relationship_between_x_and_y_l2262_226214


namespace NUMINAMATH_GPT_find_a_l2262_226249

def possible_scores : List ℕ := [103, 104, 105, 106, 107, 108, 109, 110]

def is_possible_score (a : ℕ) (n : ℕ) : Prop :=
  ∃ (k8 k0 ka : ℕ), k8 * 8 + ka * a + k0 * 0 = n

def is_impossible_score (a : ℕ) (n : ℕ) : Prop :=
  ¬ is_possible_score a n

theorem find_a : ∀ (a : ℕ), a ≠ 0 → a ≠ 8 →
  (∀ n ∈ possible_scores, is_possible_score a n) →
  is_impossible_score a 83 →
  a = 13 := by
  intros a ha1 ha2 hpossible himpossible
  sorry

end NUMINAMATH_GPT_find_a_l2262_226249


namespace NUMINAMATH_GPT_juan_marbles_eq_64_l2262_226251

def connie_marbles : ℕ := 39
def juan_extra_marbles : ℕ := 25

theorem juan_marbles_eq_64 : (connie_marbles + juan_extra_marbles) = 64 :=
by
  -- definition and conditions handled above
  sorry

end NUMINAMATH_GPT_juan_marbles_eq_64_l2262_226251


namespace NUMINAMATH_GPT_max_value_log_div_x_l2262_226208

noncomputable def func (x : ℝ) := (Real.log x) / x

theorem max_value_log_div_x : ∃ x > 0, func x = 1 / Real.exp 1 ∧ 
(∀ t > 0, t ≠ x → func t ≤ func x) :=
sorry

end NUMINAMATH_GPT_max_value_log_div_x_l2262_226208


namespace NUMINAMATH_GPT_condition_on_a_l2262_226206

theorem condition_on_a (a : ℝ) : 
  (∀ x : ℝ, (5 * x - 3 < 3 * x + 5) → (x < a)) ↔ (a ≥ 4) :=
by
  sorry

end NUMINAMATH_GPT_condition_on_a_l2262_226206


namespace NUMINAMATH_GPT_solve_for_y_l2262_226205

theorem solve_for_y (y : ℚ) (h : 1/3 + 1/y = 7/9) : y = 9/4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l2262_226205


namespace NUMINAMATH_GPT_avg_seven_consecutive_integers_l2262_226230

variable (c d : ℕ)
variable (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7)

theorem avg_seven_consecutive_integers (c d : ℕ) (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7) :
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7) = c + 6 :=
sorry

end NUMINAMATH_GPT_avg_seven_consecutive_integers_l2262_226230


namespace NUMINAMATH_GPT_sum_areas_of_tangent_circles_l2262_226275

theorem sum_areas_of_tangent_circles : 
  ∃ r s t : ℝ, 
    (r + s = 6) ∧ 
    (r + t = 8) ∧ 
    (s + t = 10) ∧ 
    (π * (r^2 + s^2 + t^2) = 36 * π) :=
by
  sorry

end NUMINAMATH_GPT_sum_areas_of_tangent_circles_l2262_226275


namespace NUMINAMATH_GPT_boxes_given_to_mom_l2262_226282

theorem boxes_given_to_mom 
  (sophie_boxes : ℕ) 
  (donuts_per_box : ℕ) 
  (donuts_to_sister : ℕ) 
  (donuts_left_for_her : ℕ) 
  (H1 : sophie_boxes = 4) 
  (H2 : donuts_per_box = 12) 
  (H3 : donuts_to_sister = 6) 
  (H4 : donuts_left_for_her = 30)
  : sophie_boxes * donuts_per_box - donuts_to_sister - donuts_left_for_her = donuts_per_box := 
by
  sorry

end NUMINAMATH_GPT_boxes_given_to_mom_l2262_226282


namespace NUMINAMATH_GPT_g_50_zero_l2262_226215

noncomputable def g : ℕ → ℝ → ℝ
| 0, x     => x + |x - 50| - |x + 50|
| (n+1), x => |g n x| - 2

theorem g_50_zero :
  ∃! x : ℝ, g 50 x = 0 :=
sorry

end NUMINAMATH_GPT_g_50_zero_l2262_226215


namespace NUMINAMATH_GPT_isosceles_triangle_three_times_ce_l2262_226220

/-!
# Problem statement
In the isosceles triangle \( ABC \) with \( \overline{AC} = \overline{BC} \), 
\( D \) is the foot of the altitude through \( C \) and \( M \) is 
the midpoint of segment \( CD \). The line \( BM \) intersects \( AC \) 
at \( E \). Prove that \( AC \) is three times as long as \( CE \).
-/

-- Definition of isosceles triangle and related points
variables {A B C D E M : Type} 

-- Assume necessary conditions
variables (triangle_isosceles : A = B)
variables (D_foot : true) -- Placeholder, replace with proper definition if needed
variables (M_midpoint : true) -- Placeholder, replace with proper definition if needed
variables (BM_intersects_AC : true) -- Placeholder, replace with proper definition if needed

-- Main statement to prove
theorem isosceles_triangle_three_times_ce (h1 : A = B)
    (h2 : true) (h3 : true) (h4 : true) : 
    AC = 3 * CE :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_three_times_ce_l2262_226220


namespace NUMINAMATH_GPT_square_area_multiplier_l2262_226223

theorem square_area_multiplier 
  (perimeter_square : ℝ) (length_rectangle : ℝ) (width_rectangle : ℝ)
  (perimeter_square_eq : perimeter_square = 800) 
  (length_rectangle_eq : length_rectangle = 125) 
  (width_rectangle_eq : width_rectangle = 64)
  : (perimeter_square / 4) ^ 2 / (length_rectangle * width_rectangle) = 5 := 
by
  sorry

end NUMINAMATH_GPT_square_area_multiplier_l2262_226223


namespace NUMINAMATH_GPT_proof_stops_with_two_pizzas_l2262_226236

/-- The number of stops with orders of two pizzas. -/
def stops_with_two_pizzas : ℕ := 2

theorem proof_stops_with_two_pizzas
  (total_pizzas : ℕ)
  (single_stops : ℕ)
  (two_pizza_stops : ℕ)
  (average_time : ℕ)
  (total_time : ℕ)
  (h1 : total_pizzas = 12)
  (h2 : two_pizza_stops * 2 + single_stops = total_pizzas)
  (h3 : total_time = 40)
  (h4 : average_time = 4)
  (h5 : two_pizza_stops + single_stops = total_time / average_time) :
  two_pizza_stops = stops_with_two_pizzas := 
sorry

end NUMINAMATH_GPT_proof_stops_with_two_pizzas_l2262_226236


namespace NUMINAMATH_GPT_sequence_formula_l2262_226228

-- Define the properties of the sequence
axiom seq_prop_1 (a : ℕ → ℝ) (m n : ℕ) (h : m > n) : a (m - n) = a m - a n

axiom seq_increasing (a : ℕ → ℝ) : ∀ n m : ℕ, n < m → a n < a m

-- Formulate the theorem to prove the general sequence formula
theorem sequence_formula (a : ℕ → ℝ) (h1 : ∀ m n : ℕ, m > n → a (m - n) = a m - a n)
    (h2 : ∀ n m : ℕ, n < m → a n < a m) :
    ∃ k > 0, ∀ n, a n = k * n :=
sorry

end NUMINAMATH_GPT_sequence_formula_l2262_226228


namespace NUMINAMATH_GPT_teacher_age_l2262_226269

theorem teacher_age (avg_age_students : ℕ) (num_students : ℕ) (avg_age_with_teacher : ℕ) (num_total : ℕ) 
  (h1 : avg_age_students = 14) (h2 : num_students = 50) (h3 : avg_age_with_teacher = 15) (h4 : num_total = 51) :
  ∃ (teacher_age : ℕ), teacher_age = 65 :=
by sorry

end NUMINAMATH_GPT_teacher_age_l2262_226269


namespace NUMINAMATH_GPT_inequality_has_real_solution_l2262_226279

variable {f : ℝ → ℝ}

theorem inequality_has_real_solution (h : ∃ x : ℝ, f x > 0) : 
    (∃ x : ℝ, f x > 0) :=
by
  sorry

end NUMINAMATH_GPT_inequality_has_real_solution_l2262_226279


namespace NUMINAMATH_GPT_triangle_area_l2262_226266

theorem triangle_area : 
  let p1 := (3, 2)
  let p2 := (3, -4)
  let p3 := (12, 2)
  let height := |2 - (-4)|
  let base := |12 - 3|
  let area := (1 / 2) * base * height
  area = 27 := sorry

end NUMINAMATH_GPT_triangle_area_l2262_226266


namespace NUMINAMATH_GPT_joe_out_of_money_after_one_month_worst_case_l2262_226209

-- Define the initial amount Joe has
def initial_amount : ℝ := 240

-- Define Joe's monthly subscription cost
def subscription_cost : ℝ := 15

-- Define the range of prices for buying games
def min_game_cost : ℝ := 40
def max_game_cost : ℝ := 60

-- Define the range of prices for selling games
def min_resale_price : ℝ := 20
def max_resale_price : ℝ := 40

-- Define the maximum number of games Joe can purchase per month
def max_games_per_month : ℕ := 3

-- Prove that Joe will be out of money after 1 month in the worst-case scenario
theorem joe_out_of_money_after_one_month_worst_case :
  initial_amount - 
  (max_games_per_month * max_game_cost - max_games_per_month * min_resale_price + subscription_cost) < 0 :=
by
  sorry

end NUMINAMATH_GPT_joe_out_of_money_after_one_month_worst_case_l2262_226209


namespace NUMINAMATH_GPT_instantaneous_velocity_at_3_l2262_226227

-- Define the position function s(t)
def s (t : ℝ) : ℝ := 1 - t + t^2

-- The main statement we need to prove
theorem instantaneous_velocity_at_3 : (deriv s 3) = 5 :=
by 
  -- The theorem requires a proof which we mark as sorry for now.
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_3_l2262_226227


namespace NUMINAMATH_GPT_stratified_sampling_by_edu_stage_is_reasonable_l2262_226256

variable (visionConditions : String → Type) -- visionConditions for different sampling methods
variable (primaryVision : Type) -- vision condition for primary school
variable (juniorVision : Type) -- vision condition for junior high school
variable (seniorVision : Type) -- vision condition for senior high school
variable (insignificantDiffGender : Prop) -- insignificant differences between boys and girls

-- Given conditions
variable (sigDiffEduStage : Prop) -- significant differences between educational stages

-- Stating the theorem
theorem stratified_sampling_by_edu_stage_is_reasonable (h1 : sigDiffEduStage) (h2 : insignificantDiffGender) : 
  visionConditions "Stratified_sampling_by_educational_stage" = visionConditions C :=
sorry

end NUMINAMATH_GPT_stratified_sampling_by_edu_stage_is_reasonable_l2262_226256


namespace NUMINAMATH_GPT_min_value_inequality_l2262_226247

open Real

theorem min_value_inequality (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 9) :
  ( (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ) ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_value_inequality_l2262_226247


namespace NUMINAMATH_GPT_polynomial_identity_l2262_226272

theorem polynomial_identity
  (x : ℝ)
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ)
  (h : (2*x + 1)^6 = a_0*x^6 + a_1*x^5 + a_2*x^4 + a_3*x^3 + a_4*x^2 + a_5*x + a_6) :
  (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 729)
  ∧ (a_1 + a_3 + a_5 = 364)
  ∧ (a_2 + a_4 = 300) := sorry

end NUMINAMATH_GPT_polynomial_identity_l2262_226272


namespace NUMINAMATH_GPT_mismatching_socks_l2262_226201

theorem mismatching_socks (total_socks : ℕ) (pairs : ℕ) (socks_per_pair : ℕ) 
  (h1 : total_socks = 25) (h2 : pairs = 4) (h3 : socks_per_pair = 2) : 
  total_socks - (socks_per_pair * pairs) = 17 :=
by
  sorry

end NUMINAMATH_GPT_mismatching_socks_l2262_226201


namespace NUMINAMATH_GPT_problem_solution_l2262_226281

noncomputable def S : ℝ :=
  1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 1 / (Real.sqrt 14 - Real.sqrt 13) - 1 / (Real.sqrt 13 - 3)

theorem problem_solution : S = (13 / 4) + (3 / 4) * Real.sqrt 13 :=
by sorry

end NUMINAMATH_GPT_problem_solution_l2262_226281


namespace NUMINAMATH_GPT_digits_difference_l2262_226234

theorem digits_difference (X Y : ℕ) (h : 10 * X + Y - (10 * Y + X) = 90) : X - Y = 10 :=
by
  sorry

end NUMINAMATH_GPT_digits_difference_l2262_226234


namespace NUMINAMATH_GPT_water_needed_l2262_226258

-- Definitions as per conditions
def heavy_wash : ℕ := 20
def regular_wash : ℕ := 10
def light_wash : ℕ := 2
def extra_light_wash (bleach : ℕ) : ℕ := bleach * light_wash

def num_heavy_washes : ℕ := 2
def num_regular_washes : ℕ := 3
def num_light_washes : ℕ := 1
def num_bleached_loads : ℕ := 2

-- Function to calculate total water usage
def total_water_used : ℕ :=
  (num_heavy_washes * heavy_wash) +
  (num_regular_washes * regular_wash) +
  (num_light_washes * light_wash) + 
  (extra_light_wash num_bleached_loads)

-- Theorem to be proved
theorem water_needed : total_water_used = 76 := by
  sorry

end NUMINAMATH_GPT_water_needed_l2262_226258


namespace NUMINAMATH_GPT_cannot_be_written_as_square_l2262_226285

theorem cannot_be_written_as_square (A B : ℤ) : 
  99999 + 111111 * Real.sqrt 3 ≠ (A + B * Real.sqrt 3) ^ 2 :=
by
  -- Here we would provide the actual mathematical proof
  sorry

end NUMINAMATH_GPT_cannot_be_written_as_square_l2262_226285


namespace NUMINAMATH_GPT_intersection_of_sets_l2262_226233

noncomputable def set_A := {x : ℝ | Real.log x ≥ 0}
noncomputable def set_B := {x : ℝ | x^2 < 9}

theorem intersection_of_sets :
  set_A ∩ set_B = {x : ℝ | 1 ≤ x ∧ x < 3} :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_of_sets_l2262_226233


namespace NUMINAMATH_GPT_cost_equation_l2262_226274

variables (x y z : ℝ)

theorem cost_equation (h1 : 2 * x + y + 3 * z = 24) (h2 : 3 * x + 4 * y + 2 * z = 36) : x + y + z = 12 := by
  -- proof steps would go here, but are omitted as per instruction
  sorry

end NUMINAMATH_GPT_cost_equation_l2262_226274


namespace NUMINAMATH_GPT_simplify_expression_l2262_226259

theorem simplify_expression :
  (49 * 91^3 + 338 * 343^2) / (66^3 - 176 * 121) / (39^3 * 7^5 / 1331000) = 125 / 13 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2262_226259


namespace NUMINAMATH_GPT_spider_has_eight_legs_l2262_226244

-- Define the number of legs a human has
def human_legs : ℕ := 2

-- Define the number of legs for a spider, based on the given condition
def spider_legs : ℕ := 2 * (2 * human_legs)

-- The theorem to be proven, that the spider has 8 legs
theorem spider_has_eight_legs : spider_legs = 8 :=
by
  sorry

end NUMINAMATH_GPT_spider_has_eight_legs_l2262_226244


namespace NUMINAMATH_GPT_possible_values_of_c_l2262_226207

theorem possible_values_of_c (a b c : ℕ) (n : ℕ) (h₀ : a ≠ 0) (h₁ : n = 729 * a + 81 * b + 36 + c) (h₂ : ∃ k, n = k^3) :
  c = 1 ∨ c = 8 :=
sorry

end NUMINAMATH_GPT_possible_values_of_c_l2262_226207


namespace NUMINAMATH_GPT_smallest_possible_difference_l2262_226286

theorem smallest_possible_difference :
  ∃ (x y z : ℕ), 
    x + y + z = 1801 ∧ x < y ∧ y ≤ z ∧ x + y > z ∧ y + z > x ∧ z + x > y ∧ (y - x = 1) := 
by
  sorry

end NUMINAMATH_GPT_smallest_possible_difference_l2262_226286


namespace NUMINAMATH_GPT_emails_received_in_afternoon_l2262_226299

theorem emails_received_in_afternoon (A : ℕ) 
  (h1 : 4 + (A - 3) = 9) : 
  A = 8 :=
by
  sorry

end NUMINAMATH_GPT_emails_received_in_afternoon_l2262_226299


namespace NUMINAMATH_GPT_expand_expression_l2262_226231

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l2262_226231


namespace NUMINAMATH_GPT_system_solution_l2262_226295

noncomputable def x1 : ℝ := 55 / Real.sqrt 91
noncomputable def y1 : ℝ := 18 / Real.sqrt 91
noncomputable def x2 : ℝ := -55 / Real.sqrt 91
noncomputable def y2 : ℝ := -18 / Real.sqrt 91

theorem system_solution (x y : ℝ) (h1 : x^2 = 4 * y^2 + 19) (h2 : x * y + 2 * y^2 = 18) :
  (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) :=
sorry

end NUMINAMATH_GPT_system_solution_l2262_226295


namespace NUMINAMATH_GPT_find_grazing_months_l2262_226237

def oxen_months_A := 10 * 7
def oxen_months_B := 12 * 5
def total_rent := 175
def rent_C := 45

def proportion_equation (x : ℕ) : Prop :=
  45 / 175 = (15 * x) / (oxen_months_A + oxen_months_B + 15 * x)

theorem find_grazing_months (x : ℕ) (h : proportion_equation x) : x = 3 :=
by
  -- We will need to involve some calculations leading to x = 3
  sorry

end NUMINAMATH_GPT_find_grazing_months_l2262_226237


namespace NUMINAMATH_GPT_secretary_worked_longest_l2262_226293

theorem secretary_worked_longest
  (h1 : ∀ (x : ℕ), 3 * x + 5 * x + 7 * x + 11 * x = 2080)
  (h2 : ∀ (a b c d : ℕ), a = 3 * x ∧ b = 5 * x ∧ c = 7 * x ∧ d = 11 * x → d = 11 * x):
  ∃ y : ℕ, y = 880 :=
by
  sorry

end NUMINAMATH_GPT_secretary_worked_longest_l2262_226293


namespace NUMINAMATH_GPT_fill_tank_with_reduced_bucket_capacity_l2262_226204

theorem fill_tank_with_reduced_bucket_capacity (C : ℝ) :
    let original_buckets := 200
    let original_capacity := C
    let new_capacity := (4 / 5) * original_capacity
    let new_buckets := 250
    (original_buckets * original_capacity) = ((new_buckets) * new_capacity) :=
by
    sorry

end NUMINAMATH_GPT_fill_tank_with_reduced_bucket_capacity_l2262_226204


namespace NUMINAMATH_GPT_amount_received_by_Sam_l2262_226263

noncomputable def final_amount (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem amount_received_by_Sam 
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ)
  (hP : P = 12000) (hr : r = 0.10) (hn : n = 2) (ht : t = 1) :
  final_amount P r n t = 12607.50 :=
by
  sorry

end NUMINAMATH_GPT_amount_received_by_Sam_l2262_226263


namespace NUMINAMATH_GPT_roger_total_miles_l2262_226250

def morning_miles : ℕ := 2
def evening_multiplicative_factor : ℕ := 5
def evening_miles := evening_multiplicative_factor * morning_miles
def third_session_subtract : ℕ := 1
def third_session_miles := (2 * morning_miles) - third_session_subtract
def total_miles := morning_miles + evening_miles + third_session_miles

theorem roger_total_miles : total_miles = 15 := by
  sorry

end NUMINAMATH_GPT_roger_total_miles_l2262_226250


namespace NUMINAMATH_GPT_simplify_expression_l2262_226253

theorem simplify_expression (x y z : ℝ) : (x + (y + z)) - ((x + z) + y) = 0 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2262_226253


namespace NUMINAMATH_GPT_math_proof_problem_l2262_226284

noncomputable def f (x : ℝ) := Real.log (Real.sin x) * Real.log (Real.cos x)

def domain (k : ℤ) : Set ℝ := { x | 2 * k * Real.pi < x ∧ x < 2 * k * Real.pi + Real.pi / 2 }

def is_even_shifted : Prop :=
  ∀ x, f (x + Real.pi / 4) = f (- (x + Real.pi / 4))

def has_unique_maximum : Prop :=
  ∃! x, 0 < x ∧ x < Real.pi / 2 ∧ ∀ y, 0 < y ∧ y < Real.pi / 2 → f y ≤ f x

theorem math_proof_problem (k : ℤ) :
  (∀ x, x ∈ domain k → f x ∈ domain k) ∧
  ¬ (∀ x, f (-x) = f x) ∧
  is_even_shifted ∧
  has_unique_maximum :=
by
  sorry

end NUMINAMATH_GPT_math_proof_problem_l2262_226284


namespace NUMINAMATH_GPT_complete_square_l2262_226296

theorem complete_square {x : ℝ} (h : x^2 + 10 * x - 3 = 0) : (x + 5)^2 = 28 :=
sorry

end NUMINAMATH_GPT_complete_square_l2262_226296


namespace NUMINAMATH_GPT_intersection_of_sets_l2262_226212

-- Defining set M
def M : Set ℝ := { x | x^2 + x - 2 < 0 }

-- Defining set N
def N : Set ℝ := { x | 0 < x ∧ x ≤ 2 }

-- Theorem stating the solution
theorem intersection_of_sets : M ∩ N = { x : ℝ | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l2262_226212


namespace NUMINAMATH_GPT_triple_divisor_sum_6_l2262_226276

-- Summarize the definition of the divisor sum function excluding the number itself
def divisorSumExcluding (n : ℕ) : ℕ :=
  (Finset.filter (λ x => x ≠ n) (Finset.range (n + 1))).sum id

-- This is the main statement that we need to prove
theorem triple_divisor_sum_6 : divisorSumExcluding (divisorSumExcluding (divisorSumExcluding 6)) = 6 := 
by sorry

end NUMINAMATH_GPT_triple_divisor_sum_6_l2262_226276


namespace NUMINAMATH_GPT_find_sum_of_A_and_B_l2262_226221

theorem find_sum_of_A_and_B :
  ∃ (A B : ℕ), A < 10 ∧ B < 10 ∧ B = A - 2 ∧ A = 5 + 3 ∧ A + B = 14 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_of_A_and_B_l2262_226221


namespace NUMINAMATH_GPT_units_digit_of_a_l2262_226238

theorem units_digit_of_a (a : ℕ) (ha : (∃ b : ℕ, 1 ≤ b ∧ b ≤ 9 ∧ (a*a / 10^1) % 10 = b)) : 
  ((a % 10 = 4) ∨ (a % 10 = 6)) :=
sorry

end NUMINAMATH_GPT_units_digit_of_a_l2262_226238


namespace NUMINAMATH_GPT_find_ticket_price_l2262_226210

theorem find_ticket_price
  (P : ℝ) -- The original price of each ticket
  (h1 : 10 * 0.6 * P + 20 * 0.85 * P + 26 * P = 980) :
  P = 20 :=
sorry

end NUMINAMATH_GPT_find_ticket_price_l2262_226210


namespace NUMINAMATH_GPT_find_a_and_b_l2262_226246

noncomputable def find_ab (a b : ℝ) : Prop :=
  (3 - 2 * a + b = 0) ∧
  (27 + 6 * a + b = 0)

theorem find_a_and_b :
  ∃ (a b : ℝ), (find_ab a b) ∧ (a = -3) ∧ (b = -9) :=
by
  sorry

end NUMINAMATH_GPT_find_a_and_b_l2262_226246


namespace NUMINAMATH_GPT_sheela_monthly_income_eq_l2262_226211

-- Defining the conditions
def sheela_deposit : ℝ := 4500
def percentage_of_income : ℝ := 0.28

-- Define Sheela's monthly income as I
variable (I : ℝ)

-- The theorem to prove
theorem sheela_monthly_income_eq : (percentage_of_income * I = sheela_deposit) → (I = 16071.43) :=
by
  sorry

end NUMINAMATH_GPT_sheela_monthly_income_eq_l2262_226211


namespace NUMINAMATH_GPT_line_ellipse_tangent_l2262_226267

theorem line_ellipse_tangent (m : ℝ) (h : ∃ x y : ℝ, y = 2 * m * x + 2 ∧ 2 * x^2 + 8 * y^2 = 8) :
  m^2 = 3 / 16 :=
sorry

end NUMINAMATH_GPT_line_ellipse_tangent_l2262_226267


namespace NUMINAMATH_GPT_evaluate_expression_l2262_226240

theorem evaluate_expression : (1.2^3 - (0.9^3 / 1.2^2) + 1.08 + 0.9^2 = 3.11175) :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_evaluate_expression_l2262_226240


namespace NUMINAMATH_GPT_laura_owes_amount_l2262_226225

def principal : ℝ := 35
def rate : ℝ := 0.05
def time : ℝ := 1
def interest (P R T : ℝ) := P * R * T
def totalAmountOwed (P I : ℝ) := P + I

theorem laura_owes_amount : totalAmountOwed principal (interest principal rate time) = 36.75 :=
by
  sorry

end NUMINAMATH_GPT_laura_owes_amount_l2262_226225


namespace NUMINAMATH_GPT_machine_work_rate_l2262_226218

theorem machine_work_rate (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -6 ∧ x ≠ -1) : 
  (1 / (x + 6) + 1 / (x + 1) + 1 / (2 * x) = 1 / x) → x = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_machine_work_rate_l2262_226218


namespace NUMINAMATH_GPT_no_quadratic_polynomials_f_g_l2262_226260

theorem no_quadratic_polynomials_f_g (f g : ℝ → ℝ) 
  (hf : ∃ a b c, ∀ x, f x = a * x^2 + b * x + c)
  (hg : ∃ d e h, ∀ x, g x = d * x^2 + e * x + h) : 
  ¬ (∀ x, f (g x) = x^4 - 3 * x^3 + 3 * x^2 - x) :=
by
  sorry

end NUMINAMATH_GPT_no_quadratic_polynomials_f_g_l2262_226260


namespace NUMINAMATH_GPT_center_of_tangent_circle_lies_on_hyperbola_l2262_226283

open Real

def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4

def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 6*y + 24 = 0

noncomputable def locus_of_center : Set (ℝ × ℝ) :=
  {P | ∃ (r : ℝ), ∀ (x1 y1 x2 y2 : ℝ), circle1 x1 y1 ∧ circle2 x2 y2 → 
    dist P (x1, y1) = r + 2 ∧ dist P (x2, y2) = r + 1}

theorem center_of_tangent_circle_lies_on_hyperbola :
  ∀ P : ℝ × ℝ, P ∈ locus_of_center → ∃ (a b : ℝ) (F1 F2 : ℝ × ℝ), ∀ Q : ℝ × ℝ,
    dist Q F1 - dist Q F2 = 1 ∧ 
    dist F1 F2 = 5 ∧
    P ∈ {Q | dist Q F1 - dist Q F2 = 1} :=
sorry

end NUMINAMATH_GPT_center_of_tangent_circle_lies_on_hyperbola_l2262_226283


namespace NUMINAMATH_GPT_common_ratio_l2262_226280

theorem common_ratio (a1 a2 a3 : ℚ) (S3 q : ℚ)
  (h1 : a3 = 3 / 2)
  (h2 : S3 = 9 / 2)
  (h3 : a1 + a2 + a3 = S3)
  (h4 : a1 = a3 / q^2)
  (h5 : a2 = a3 / q):
  q = 1 ∨ q = -1/2 :=
by sorry

end NUMINAMATH_GPT_common_ratio_l2262_226280


namespace NUMINAMATH_GPT_find_bags_l2262_226294

theorem find_bags (x : ℕ) : 10 + x + 7 = 20 → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_bags_l2262_226294


namespace NUMINAMATH_GPT_lisa_need_add_pure_juice_l2262_226232

theorem lisa_need_add_pure_juice
  (x : ℝ) 
  (total_volume : ℝ := 2)
  (initial_pure_juice_fraction : ℝ := 0.10)
  (desired_pure_juice_fraction : ℝ := 0.25) 
  (added_pure_juice : ℝ := x) 
  (initial_pure_juice_amount : ℝ := total_volume * initial_pure_juice_fraction)
  (final_pure_juice_amount : ℝ := initial_pure_juice_amount + added_pure_juice)
  (final_volume : ℝ := total_volume + added_pure_juice) :
  (final_pure_juice_amount / final_volume) = desired_pure_juice_fraction → x = 0.4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_lisa_need_add_pure_juice_l2262_226232


namespace NUMINAMATH_GPT_greg_age_is_16_l2262_226277

-- Define the ages of Cindy, Jan, Marcia, and Greg based on the conditions
def cindy_age : ℕ := 5
def jan_age : ℕ := cindy_age + 2
def marcia_age : ℕ := 2 * jan_age
def greg_age : ℕ := marcia_age + 2

-- Theorem statement: Prove that Greg's age is 16
theorem greg_age_is_16 : greg_age = 16 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_greg_age_is_16_l2262_226277


namespace NUMINAMATH_GPT_total_amount_l2262_226270

-- Conditions as given definitions
def ratio_a : Nat := 2
def ratio_b : Nat := 3
def ratio_c : Nat := 4
def share_b : Nat := 1500

-- The final statement
theorem total_amount (parts_b := 3) (one_part := share_b / parts_b) :
  (2 * one_part) + (3 * one_part) + (4 * one_part) = 4500 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_l2262_226270


namespace NUMINAMATH_GPT_sum_of_first_4_terms_l2262_226242

noncomputable def geom_sum (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem sum_of_first_4_terms (a r : ℝ) 
  (h1 : a * (1 + r + r^2) = 13) (h2 : a * (1 + r + r^2 + r^3 + r^4) = 121) : 
  a * (1 + r + r^2 + r^3) = 27.857 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_4_terms_l2262_226242


namespace NUMINAMATH_GPT_triangle_area_l2262_226254

theorem triangle_area 
  (DE EL EF : ℝ)
  (hDE : DE = 14)
  (hEL : EL = 9)
  (hEF : EF = 17)
  (DL : ℝ)
  (hDL : DE^2 = DL^2 + EL^2)
  (hDL_val : DL = Real.sqrt 115):
  (1/2) * EF * DL = 17 * Real.sqrt 115 / 2 :=
by
  -- Sorry, as the proof is not required.
  sorry

end NUMINAMATH_GPT_triangle_area_l2262_226254


namespace NUMINAMATH_GPT_average_salary_of_all_workers_l2262_226264

-- Definitions of conditions
def T : ℕ := 7
def total_workers : ℕ := 56
def W : ℕ := total_workers - T
def A_T : ℕ := 12000
def A_W : ℕ := 6000

-- Definition of total salary and average salary
def total_salary : ℕ := (T * A_T) + (W * A_W)

theorem average_salary_of_all_workers : total_salary / total_workers = 6750 := 
  by sorry

end NUMINAMATH_GPT_average_salary_of_all_workers_l2262_226264


namespace NUMINAMATH_GPT_side_length_range_l2262_226278

-- Define the inscribed circle diameter condition
def inscribed_circle_diameter (d : ℝ) (cir_diameter : ℝ) := cir_diameter = 1

-- Define inscribed square side condition
def inscribed_square_side (d side : ℝ) :=
  ∃ (triangle_ABC : Type) (AB AC BC : triangle_ABC → ℝ), 
    side = d ∧
    side < 1

-- Define the main theorem: The side length of the inscribed square lies within given bounds
theorem side_length_range (d : ℝ) :
  inscribed_circle_diameter d 1 → inscribed_square_side d d → (4/5) ≤ d ∧ d < 1 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_side_length_range_l2262_226278


namespace NUMINAMATH_GPT_exists_n_not_perfect_square_l2262_226235

theorem exists_n_not_perfect_square (a b : ℤ) (h1 : a > 1) (h2 : b > 1) (h3 : a ≠ b) : 
  ∃ (n : ℕ), (n > 0) ∧ ¬∃ (k : ℤ), (a^n - 1) * (b^n - 1) = k^2 :=
by sorry

end NUMINAMATH_GPT_exists_n_not_perfect_square_l2262_226235


namespace NUMINAMATH_GPT_original_fish_count_l2262_226261

def initial_fish_count (fish_taken_out : ℕ) (current_fish : ℕ) : ℕ :=
  fish_taken_out + current_fish

theorem original_fish_count :
  initial_fish_count 16 3 = 19 :=
by
  sorry

end NUMINAMATH_GPT_original_fish_count_l2262_226261


namespace NUMINAMATH_GPT_distribute_candy_bars_l2262_226265

theorem distribute_candy_bars (candies bags : ℕ) (h1 : candies = 15) (h2 : bags = 5) :
  candies / bags = 3 :=
by
  sorry

end NUMINAMATH_GPT_distribute_candy_bars_l2262_226265


namespace NUMINAMATH_GPT_sphere_radius_eq_3_l2262_226224

theorem sphere_radius_eq_3 (r : ℝ) (h : (4/3) * Real.pi * r^3 = 4 * Real.pi * r^2) : r = 3 :=
sorry

end NUMINAMATH_GPT_sphere_radius_eq_3_l2262_226224
