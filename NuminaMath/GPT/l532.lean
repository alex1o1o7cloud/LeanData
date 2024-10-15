import Mathlib

namespace NUMINAMATH_GPT_range_of_inverse_proportion_l532_53253

theorem range_of_inverse_proportion (x : ℝ) (h : 3 < x) :
    -1 < -3 / x ∧ -3 / x < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_inverse_proportion_l532_53253


namespace NUMINAMATH_GPT_batch_of_pizza_dough_makes_three_pizzas_l532_53283

theorem batch_of_pizza_dough_makes_three_pizzas
  (pizza_dough_time : ℕ)
  (baking_time : ℕ)
  (total_time_minutes : ℕ)
  (oven_capacity : ℕ)
  (total_pizzas : ℕ) 
  (number_of_batches : ℕ)
  (one_batch_pizzas : ℕ) :
  pizza_dough_time = 30 →
  baking_time = 30 →
  total_time_minutes = 300 →
  oven_capacity = 2 →
  total_pizzas = 12 →
  total_time_minutes = total_pizzas / oven_capacity * baking_time + number_of_batches * pizza_dough_time →
  number_of_batches = total_time_minutes / 30 →
  one_batch_pizzas = total_pizzas / number_of_batches →
  one_batch_pizzas = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_batch_of_pizza_dough_makes_three_pizzas_l532_53283


namespace NUMINAMATH_GPT_pitbull_chihuahua_weight_ratio_l532_53225

theorem pitbull_chihuahua_weight_ratio
  (C P G : ℕ)
  (h1 : G = 307)
  (h2 : G = 3 * P + 10)
  (h3 : C + P + G = 439) :
  P / C = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_pitbull_chihuahua_weight_ratio_l532_53225


namespace NUMINAMATH_GPT_log_base_property_l532_53294

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x / log a

theorem log_base_property
  (a : ℝ)
  (ha_pos : a > 0)
  (ha_ne_one : a ≠ 1)
  (hf9 : f a 9 = 2) :
  f a (3^a) = 3 :=
by
  sorry

end NUMINAMATH_GPT_log_base_property_l532_53294


namespace NUMINAMATH_GPT_chimney_bricks_l532_53298

variable (h : ℕ)

/-- Brenda would take 8 hours to build a chimney alone. 
    Brandon would take 12 hours to build it alone. 
    When they work together, their efficiency is diminished by 15 bricks per hour due to their chatting. 
    If they complete the chimney in 6 hours when working together, then the total number of bricks in the chimney is 360. -/
theorem chimney_bricks
  (h : ℕ)
  (Brenda_rate : ℕ)
  (Brandon_rate : ℕ)
  (effective_rate : ℕ)
  (completion_time : ℕ)
  (h_eq : Brenda_rate = h / 8)
  (h_eq_alt : Brandon_rate = h / 12)
  (effective_rate_eq : effective_rate = (Brenda_rate + Brandon_rate) - 15)
  (completion_eq : 6 * effective_rate = h) :
  h = 360 := by 
  sorry

end NUMINAMATH_GPT_chimney_bricks_l532_53298


namespace NUMINAMATH_GPT_geom_sequence_a7_l532_53287

theorem geom_sequence_a7 (a : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n : ℕ, a (n+1) = a n * r) 
  (h_a1 : a 1 = 8) 
  (h_a4_eq : a 4 = a 3 * a 5) : 
  a 7 = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_geom_sequence_a7_l532_53287


namespace NUMINAMATH_GPT_find_n_l532_53227

theorem find_n (n : ℤ) : 
  50 < n ∧ n < 120 ∧ (n % 8 = 0) ∧ (n % 7 = 3) ∧ (n % 9 = 3) → n = 192 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l532_53227


namespace NUMINAMATH_GPT_terminal_side_equiv_l532_53218

theorem terminal_side_equiv (θ : ℝ) (hθ : θ = 23 * π / 3) : 
  ∃ k : ℤ, θ = 2 * π * k + 5 * π / 3 := by
  sorry

end NUMINAMATH_GPT_terminal_side_equiv_l532_53218


namespace NUMINAMATH_GPT_root_intervals_l532_53250

noncomputable def f (a b c x : ℝ) : ℝ := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem root_intervals (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ r1 r2 : ℝ, (a < r1 ∧ r1 < b ∧ f a b c r1 = 0) ∧ (b < r2 ∧ r2 < c ∧ f a b c r2 = 0) :=
sorry

end NUMINAMATH_GPT_root_intervals_l532_53250


namespace NUMINAMATH_GPT_exponentiation_problem_l532_53209

theorem exponentiation_problem (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : 2^a * 2^b = 8) : (2^a)^b = 4 := 
sorry

end NUMINAMATH_GPT_exponentiation_problem_l532_53209


namespace NUMINAMATH_GPT_sort_mail_together_time_l532_53243

-- Definitions of work rates
def mail_handler_work_rate : ℚ := 1 / 3
def assistant_work_rate : ℚ := 1 / 6

-- Definition to calculate combined work time
def combined_time (rate1 rate2 : ℚ) : ℚ := 1 / (rate1 + rate2)

-- Statement to prove
theorem sort_mail_together_time :
  combined_time mail_handler_work_rate assistant_work_rate = 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sort_mail_together_time_l532_53243


namespace NUMINAMATH_GPT_inequality_abc_l532_53205

theorem inequality_abc (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1/a) + (1/b) ≥ 4/(a + b) :=
by
  sorry

end NUMINAMATH_GPT_inequality_abc_l532_53205


namespace NUMINAMATH_GPT_weight_loss_percentage_l532_53231

theorem weight_loss_percentage 
  (weight_before weight_after : ℝ) 
  (h_before : weight_before = 800) 
  (h_after : weight_after = 640) : 
  (weight_before - weight_after) / weight_before * 100 = 20 := 
by
  sorry

end NUMINAMATH_GPT_weight_loss_percentage_l532_53231


namespace NUMINAMATH_GPT_calculate_ab_plus_cd_l532_53235

theorem calculate_ab_plus_cd (a b c d : ℝ) 
  (h1 : a + b + c = 5)
  (h2 : a + b + d = -1)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = 12) :
  a * b + c * d = 27 :=
by
  sorry -- Proof to be filled in.

end NUMINAMATH_GPT_calculate_ab_plus_cd_l532_53235


namespace NUMINAMATH_GPT_integer_solutions_inequality_system_l532_53239

theorem integer_solutions_inequality_system :
  {x : ℤ | 2 * (x - 1) ≤ x + 3 ∧ (x + 1) / 3 < x - 1} = {3, 4, 5} :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_inequality_system_l532_53239


namespace NUMINAMATH_GPT_hyperbola_center_l532_53207

theorem hyperbola_center (x y : ℝ) :
    9 * x^2 - 18 * x - 16 * y^2 + 64 * y - 143 = 0 →
    (x, y) = (1, 2) :=
sorry

end NUMINAMATH_GPT_hyperbola_center_l532_53207


namespace NUMINAMATH_GPT_find_f_2017_l532_53245

theorem find_f_2017 (f : ℕ → ℕ) (H1 : ∀ x y : ℕ, f (x * y + 1) = f x * f y - f y - x + 2) (H2 : f 0 = 1) : f 2017 = 2018 :=
sorry

end NUMINAMATH_GPT_find_f_2017_l532_53245


namespace NUMINAMATH_GPT_triangle_inequality_equality_condition_l532_53296

theorem triangle_inequality (a b c S : ℝ)
  (h_tri : a^2 + b^2 + c^2 = 4 * S * Real.sqrt 3)
  (h_area : S = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))):
  a^2 + b^2 + c^2 ≥ 4 * S * Real.sqrt 3 :=
sorry

theorem equality_condition (a b c S : ℝ)
  (h_tri : a^2 + b^2 + c^2 = 4 * S * Real.sqrt 3)
  (h_area : S = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))):
  (a = b) ∧ (b = c) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_equality_condition_l532_53296


namespace NUMINAMATH_GPT_math_problem_l532_53295

def is_polynomial (expr : String) : Prop := sorry
def is_monomial (expr : String) : Prop := sorry
def is_cubic (expr : String) : Prop := sorry
def is_quintic (expr : String) : Prop := sorry
def correct_option_C : String := "C"

theorem math_problem :
  ¬ is_polynomial "8 - 2 / z" ∧
  ¬ (is_monomial "-x^2yz" ∧ is_cubic "-x^2yz") ∧
  is_polynomial "x^2 - 3xy^2 + 2x^2y^3 - 1" ∧
  is_quintic "x^2 - 3xy^2 + 2x^2y^3 - 1" ∧
  ¬ is_monomial "5b / x" →
  correct_option_C = "C" := sorry

end NUMINAMATH_GPT_math_problem_l532_53295


namespace NUMINAMATH_GPT_sum_local_values_l532_53223

theorem sum_local_values :
  let local_value_2 := 2000
  let local_value_3 := 300
  let local_value_4 := 40
  let local_value_5 := 5
  local_value_2 + local_value_3 + local_value_4 + local_value_5 = 2345 :=
by
  sorry

end NUMINAMATH_GPT_sum_local_values_l532_53223


namespace NUMINAMATH_GPT_sequence_is_arithmetic_l532_53272

theorem sequence_is_arithmetic 
  (a_n : ℕ → ℤ) 
  (h : ∀ n : ℕ, a_n n = n + 1) 
  : ∀ n : ℕ, a_n (n + 1) - a_n n = 1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_is_arithmetic_l532_53272


namespace NUMINAMATH_GPT_find_a_range_l532_53216

theorem find_a_range (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ 2 * x^2 - 2 * x + a - 3 = 0) ∧ 
  (∃ y : ℝ, y > 0 ∧ y ≠ x ∧ 2 * y^2 - 2 * y + a - 3 = 0) 
  ↔ 3 < a ∧ a < 7 / 2 := 
sorry

end NUMINAMATH_GPT_find_a_range_l532_53216


namespace NUMINAMATH_GPT_multiply_fractions_l532_53259

theorem multiply_fractions :
  (1 / 3 : ℚ) * (3 / 5) * (5 / 6) = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_multiply_fractions_l532_53259


namespace NUMINAMATH_GPT_cost_of_cookies_l532_53233

theorem cost_of_cookies (diane_has : ℕ) (needs_more : ℕ) (cost : ℕ) :
  diane_has = 27 → needs_more = 38 → cost = 65 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_cookies_l532_53233


namespace NUMINAMATH_GPT_arithmetic_sequence_nth_term_l532_53255

theorem arithmetic_sequence_nth_term (a₁ a₂ a₃ : ℤ) (x : ℤ) (n : ℕ)
  (h₁ : a₁ = 3 * x - 4)
  (h₂ : a₂ = 6 * x - 14)
  (h₃ : a₃ = 4 * x + 3)
  (h₄ : ∀ k : ℕ, a₁ + (k - 1) * ((a₂ - a₁) + (a₃ - a₂) / 2) = 3012) :
  n = 247 :=
by {
  -- Proof to be provided
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_nth_term_l532_53255


namespace NUMINAMATH_GPT_largest_packet_size_gcd_l532_53279

theorem largest_packet_size_gcd:
    ∀ (n1 n2 : ℕ), n1 = 36 → n2 = 60 → Nat.gcd n1 n2 = 12 :=
by
  intros n1 n2 h1 h2
  -- Sorry is added because the proof is not required as per the instructions
  sorry

end NUMINAMATH_GPT_largest_packet_size_gcd_l532_53279


namespace NUMINAMATH_GPT_blue_marble_difference_l532_53229

theorem blue_marble_difference :
  ∃ a b : ℕ, (10 * a = 10 * b) ∧ (3 * a + b = 80) ∧ (7 * a - 9 * b = 40) := by
  sorry

end NUMINAMATH_GPT_blue_marble_difference_l532_53229


namespace NUMINAMATH_GPT_find_x_l532_53210

theorem find_x (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
(h_eq : 7 * x^3 + 14 * x^2 * y = x^4 + 2 * x^3 * y) :
  x = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l532_53210


namespace NUMINAMATH_GPT_box_volume_correct_l532_53238

-- Define the dimensions of the obelisk
def obelisk_height : ℕ := 15
def base_length : ℕ := 8
def base_width : ℕ := 10

-- Define the dimension and volume goal for the cube-shaped box
def box_side_length : ℕ := obelisk_height
def box_volume : ℕ := box_side_length ^ 3

-- The proof goal
theorem box_volume_correct : box_volume = 3375 := 
by sorry

end NUMINAMATH_GPT_box_volume_correct_l532_53238


namespace NUMINAMATH_GPT_daniel_dolls_l532_53204

theorem daniel_dolls (normal_price discount_price: ℕ) 
  (normal_dolls: ℕ) 
  (saved_money: ℕ := normal_dolls * normal_price):
  normal_price = 4 →
  normal_dolls = 15 →
  discount_price = 3 →
  saved_money = normal_dolls * normal_price →
  saved_money / discount_price = 20 :=
by
  sorry

end NUMINAMATH_GPT_daniel_dolls_l532_53204


namespace NUMINAMATH_GPT_odd_function_periodic_example_l532_53230

theorem odd_function_periodic_example (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_period : ∀ x, f (x + 2) = -f x) 
  (h_segment : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x) :
  f (10 * Real.sqrt 3) = 36 - 20 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_odd_function_periodic_example_l532_53230


namespace NUMINAMATH_GPT_find_a_l532_53286

theorem find_a (x1 x2 x3 x4 x5 x6 x7 : ℝ)
  (h1 : x1 = 180)
  (h2 : x2 = 182)
  (h3 : x3 = 173)
  (h4 : x4 = 175)
  (h6 : x6 = 178)
  (h7 : x7 = 176)
  (h_avg : (x1 + x2 + x3 + x4 + x5 + x6 + x7) / 7 = 178) : x5 = 182 := by
  sorry

end NUMINAMATH_GPT_find_a_l532_53286


namespace NUMINAMATH_GPT_m_condition_sufficient_not_necessary_l532_53240

-- Define the function f(x) and its properties
def f (m : ℝ) (x : ℝ) : ℝ := abs (x * (m * x + 2))

-- Define the condition for the function being increasing on (0, ∞)
def is_increasing_on_positives (m : ℝ) :=
  ∀ x y : ℝ, 0 < x → x < y → f m x < f m y

-- Prove that if m > 0, then the function is increasing on (0, ∞)
lemma m_gt_0_sufficient (m : ℝ) (h : 0 < m) : is_increasing_on_positives m :=
sorry

-- Show that the condition is indeed sufficient but not necessary
theorem m_condition_sufficient_not_necessary :
  ∀ m : ℝ, (0 < m → is_increasing_on_positives m) ∧ (is_increasing_on_positives m → 0 < m) :=
sorry

end NUMINAMATH_GPT_m_condition_sufficient_not_necessary_l532_53240


namespace NUMINAMATH_GPT_calculate_expression_l532_53220

theorem calculate_expression : 287 * 287 + 269 * 269 - (2 * 287 * 269) = 324 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l532_53220


namespace NUMINAMATH_GPT_find_m_l532_53258

theorem find_m (a0 a1 a2 a3 a4 a5 a6 m : ℝ) 
  (h1 : (1 + m) ^ 6 = a0 + a1 + a2 + a3 + a4 + a5 + a6) 
  (h2 : a0 + a1 + a2 + a3 + a4 + a5 + a6 = 64) :
  m = 1 ∨ m = -3 := 
  sorry

end NUMINAMATH_GPT_find_m_l532_53258


namespace NUMINAMATH_GPT_ratio_proof_l532_53222

noncomputable def ratio_of_segment_lengths (a b : ℝ) (points : Finset (ℝ × ℝ)) : Prop :=
  points.card = 5 ∧
  ∃ (dists : Finset ℝ), 
    dists = {a, a, a, a, a, b, 3 * a} ∧
    ∀ (p1 p2 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p1 ≠ p2 →
      (dist p1 p2 ∈ dists)

theorem ratio_proof (a b : ℝ) (points : Finset (ℝ × ℝ)) (h : ratio_of_segment_lengths a b points) : 
  b / a = 2.8 :=
sorry

end NUMINAMATH_GPT_ratio_proof_l532_53222


namespace NUMINAMATH_GPT_find_a_l532_53292

theorem find_a (a : ℝ) (h₁ : a > 1) (h₂ : (∀ x : ℝ, a^3 = 8)) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l532_53292


namespace NUMINAMATH_GPT_cos_double_angle_l532_53297

theorem cos_double_angle (α : ℝ) (h : Real.cos α = -Real.sqrt 3 / 2) : Real.cos (2 * α) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l532_53297


namespace NUMINAMATH_GPT_episodes_lost_per_season_l532_53214

theorem episodes_lost_per_season (s1 s2 : ℕ) (e : ℕ) (remaining : ℕ) (total_seasons : ℕ) (total_episodes_before : ℕ) (total_episodes_lost : ℕ)
  (h1 : s1 = 12) (h2 : s2 = 14) (h3 : e = 16) (h4 : remaining = 364) 
  (h5 : total_seasons = s1 + s2) (h6 : total_episodes_before = s1 * e + s2 * e) 
  (h7 : total_episodes_lost = total_episodes_before - remaining) :
  total_episodes_lost / total_seasons = 2 := by
  sorry

end NUMINAMATH_GPT_episodes_lost_per_season_l532_53214


namespace NUMINAMATH_GPT_tangent_line_intersects_y_axis_at_10_l532_53269

-- Define the curve y = x^2 + 11
def curve (x : ℝ) : ℝ := x^2 + 11

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 2 * x

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (1, 12)

-- Define the tangent line at point_of_tangency
def tangent_line (x : ℝ) : ℝ :=
  let slope := curve_derivative point_of_tangency.1
  let y_intercept := point_of_tangency.2 - slope * point_of_tangency.1
  slope * x + y_intercept

-- Theorem stating the y-coordinate of the intersection of the tangent line with the y-axis
theorem tangent_line_intersects_y_axis_at_10 :
  tangent_line 0 = 10 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_intersects_y_axis_at_10_l532_53269


namespace NUMINAMATH_GPT_edith_books_total_l532_53219

-- Define the conditions
def novels := 80
def writing_books := novels * 2

-- Theorem statement
theorem edith_books_total : novels + writing_books = 240 :=
by sorry

end NUMINAMATH_GPT_edith_books_total_l532_53219


namespace NUMINAMATH_GPT_find_x_l532_53285

def Hiram_age := 40
def Allyson_age := 28
def Twice_Allyson_age := 2 * Allyson_age
def Four_less_than_twice_Allyson_age := Twice_Allyson_age - 4

theorem find_x (x : ℤ) : Hiram_age + x = Four_less_than_twice_Allyson_age → x = 12 := 
by
  intros h -- introducing the assumption 
  sorry

end NUMINAMATH_GPT_find_x_l532_53285


namespace NUMINAMATH_GPT_common_difference_divisible_by_p_l532_53278

variable (a : ℕ → ℕ) (p : ℕ)

-- Define that the sequence a is an arithmetic progression with common difference d
def is_arithmetic_progression (d : ℕ) : Prop :=
  ∀ i : ℕ, a (i + 1) = a i + d

-- Define that the sequence a is strictly increasing
def is_increasing_arithmetic_progression : Prop :=
  ∀ i j : ℕ, i < j → a i < a j

-- Define that all elements a_i are prime numbers
def all_primes : Prop :=
  ∀ i : ℕ, Nat.Prime (a i)

-- Define that the first element of the sequence is greater than p
def first_element_greater_than_p : Prop :=
  a 1 > p

-- Combining all conditions
def conditions (d : ℕ) : Prop :=
  is_arithmetic_progression a d ∧ is_increasing_arithmetic_progression a ∧ all_primes a ∧ first_element_greater_than_p a p ∧ Nat.Prime p

-- Statement to prove: common difference is divisible by p
theorem common_difference_divisible_by_p (d : ℕ) (h : conditions a p d) : p ∣ d :=
sorry

end NUMINAMATH_GPT_common_difference_divisible_by_p_l532_53278


namespace NUMINAMATH_GPT_edward_spent_13_l532_53282

-- Define the initial amount of money Edward had
def initial_amount : ℕ := 19
-- Define the current amount of money Edward has now
def current_amount : ℕ := 6
-- Define the amount of money Edward spent
def amount_spent : ℕ := initial_amount - current_amount

-- The proof we need to show
theorem edward_spent_13 : amount_spent = 13 := by
  -- The proof goes here.
  sorry

end NUMINAMATH_GPT_edward_spent_13_l532_53282


namespace NUMINAMATH_GPT_image_center_coordinates_l532_53281

-- Define the point reflecting across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define the point translation by adding some units to the y-coordinate
def translate_y (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + dy)

-- Define the initial point and translation
def initial_point : ℝ × ℝ := (3, -4)
def translation_units : ℝ := 5

-- Prove the final coordinates of the image of the center of circle Q
theorem image_center_coordinates : translate_y (reflect_x initial_point) translation_units = (3, 9) :=
  sorry

end NUMINAMATH_GPT_image_center_coordinates_l532_53281


namespace NUMINAMATH_GPT_positive_integer_solutions_of_inequality_l532_53251

theorem positive_integer_solutions_of_inequality :
  {x : ℕ | 2 * (x - 1) < 7 - x ∧ x > 0} = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_solutions_of_inequality_l532_53251


namespace NUMINAMATH_GPT_jane_last_day_vases_l532_53236

theorem jane_last_day_vases (vases_per_day : ℕ) (total_vases : ℕ) (days : ℕ) (day_arrange_total: days = 17) (vases_per_day_is_25 : vases_per_day = 25) (total_vases_is_378 : total_vases = 378) :
  (vases_per_day * (days - 1) >= total_vases) → (total_vases - vases_per_day * (days - 1)) = 0 :=
by
  intros h
  -- adding this line below to match condition ": (total_vases - vases_per_day * (days - 1)) = 0"
  sorry

end NUMINAMATH_GPT_jane_last_day_vases_l532_53236


namespace NUMINAMATH_GPT_correct_calculated_value_l532_53270

theorem correct_calculated_value (x : ℝ) (h : 3 * x - 5 = 103) : x / 3 - 5 = 7 := 
by 
  sorry

end NUMINAMATH_GPT_correct_calculated_value_l532_53270


namespace NUMINAMATH_GPT_right_triangle_area_l532_53242

theorem right_triangle_area (x : ℝ) (h : 3 * x + 4 * x = 10) : 
  (1 / 2) * (3 * x) * (4 * x) = 24 :=
sorry

end NUMINAMATH_GPT_right_triangle_area_l532_53242


namespace NUMINAMATH_GPT_find_a_range_l532_53262

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then -(x - 1) ^ 2 else (3 - a) * x + 4 * a

theorem find_a_range (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → (f x₁ a - f x₂ a) / (x₁ - x₂) > 0) ↔ (-1 ≤ a ∧ a < 3) :=
sorry

end NUMINAMATH_GPT_find_a_range_l532_53262


namespace NUMINAMATH_GPT_kolya_win_l532_53291

theorem kolya_win : ∀ stones : ℕ, stones = 100 → (∃ strategy : (ℕ → ℕ × ℕ), ∀ opponent_strategy : (ℕ → ℕ × ℕ), true → true) :=
by
  sorry

end NUMINAMATH_GPT_kolya_win_l532_53291


namespace NUMINAMATH_GPT_max_partitioned_test_plots_is_78_l532_53273

def field_length : ℕ := 52
def field_width : ℕ := 24
def total_fence : ℕ := 1994
def gcd_field_dimensions : ℕ := Nat.gcd field_length field_width

-- Since gcd_field_dimensions divides both 52 and 24 and gcd_field_dimensions = 4
def possible_side_lengths : List ℕ := [1, 2, 4]

noncomputable def max_square_plots : ℕ :=
  let max_plots (a : ℕ) : ℕ := (field_length / a) * (field_width / a)
  let valid_fence (a : ℕ) : Bool :=
    let vertical_fence := (field_length / a - 1) * field_width
    let horizontal_fence := (field_width / a - 1) * field_length
    vertical_fence + horizontal_fence ≤ total_fence
  let valid_lengths := possible_side_lengths.filter valid_fence
  valid_lengths.map max_plots |>.maximum? |>.getD 0

theorem max_partitioned_test_plots_is_78 : max_square_plots = 78 := by
  sorry

end NUMINAMATH_GPT_max_partitioned_test_plots_is_78_l532_53273


namespace NUMINAMATH_GPT_solve_fractional_equation_l532_53280

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 1) :
  (x^2 - x + 2) / (x - 1) = x + 3 ↔ x = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l532_53280


namespace NUMINAMATH_GPT_jordans_score_l532_53212

theorem jordans_score 
  (N : ℕ) 
  (first_19_avg : ℚ) 
  (total_avg : ℚ)
  (total_score_19 : ℚ) 
  (total_score_20 : ℚ) 
  (jordan_score : ℚ) 
  (h1 : N = 19)
  (h2 : first_19_avg = 74)
  (h3 : total_avg = 76)
  (h4 : total_score_19 = N * first_19_avg)
  (h5 : total_score_20 = (N + 1) * total_avg)
  (h6 : jordan_score = total_score_20 - total_score_19) :
  jordan_score = 114 :=
by {
  -- the proof will be filled in, but for now we use sorry
  sorry
}

end NUMINAMATH_GPT_jordans_score_l532_53212


namespace NUMINAMATH_GPT_parameter_a_values_l532_53277

theorem parameter_a_values (a : ℝ) :
  (∃ x y : ℝ, |x + y + 8| + |x - y + 8| = 16 ∧ ((|x| - 8)^2 + (|y| - 15)^2 = a) ∧
    (∀ x₁ y₁ x₂ y₂ : ℝ, |x₁ + y₁ + 8| + |x₁ - y₁ + 8| = 16 →
      (|x₁| - 8)^2 + (|y₁| - 15)^2 = a →
      |x₂ + y₂ + 8| + |x₂ - y₂ + 8| = 16 →
      (|x₂| - 8)^2 + (|y₂| - 15)^2 = a →
      (x₁, y₁) = (x₂, y₂) ∨ (x₁, y₁) = (y₂, x₂))) ↔ a = 49 ∨ a = 289 :=
by sorry

end NUMINAMATH_GPT_parameter_a_values_l532_53277


namespace NUMINAMATH_GPT_bananas_needed_to_make_yogurts_l532_53234

theorem bananas_needed_to_make_yogurts 
    (slices_per_yogurt : ℕ) 
    (slices_per_banana: ℕ) 
    (number_of_yogurts: ℕ) 
    (total_needed_slices: ℕ) 
    (bananas_needed: ℕ) 
    (h1: slices_per_yogurt = 8)
    (h2: slices_per_banana = 10)
    (h3: number_of_yogurts = 5)
    (h4: total_needed_slices = number_of_yogurts * slices_per_yogurt)
    (h5: bananas_needed = total_needed_slices / slices_per_banana): 
    bananas_needed = 4 := 
by
    sorry

end NUMINAMATH_GPT_bananas_needed_to_make_yogurts_l532_53234


namespace NUMINAMATH_GPT_replaced_solution_percentage_l532_53268

theorem replaced_solution_percentage (y x z w : ℝ) 
  (h1 : x = 0.5)
  (h2 : y = 80)
  (h3 : z = 0.5 * y)
  (h4 : w = 50) 
  :
  (40 + 0.5 * x) = 50 → x = 20 :=
by
  sorry

end NUMINAMATH_GPT_replaced_solution_percentage_l532_53268


namespace NUMINAMATH_GPT_remainder_product_div_6_l532_53202

theorem remainder_product_div_6 :
  (3 * 7 * 13 * 17 * 23 * 27 * 33 * 37 * 43 * 47 * 53 * 57 * 63 * 67 * 73 * 77 * 83 * 87 * 93 * 97 
   * 103 * 107 * 113 * 117 * 123 * 127 * 133 * 137 * 143 * 147 * 153 * 157 * 163 * 167 * 173 
   * 177 * 183 * 187 * 193 * 197) % 6 = 3 := 
by 
  -- basic info about modulo arithmetic and properties of sequences
  sorry

end NUMINAMATH_GPT_remainder_product_div_6_l532_53202


namespace NUMINAMATH_GPT_Cauchy_solution_on_X_l532_53224

section CauchyEquation

variable (f : ℝ → ℝ)

def is_morphism (f : ℝ → ℝ) := ∀ x y : ℝ, f (x + y) = f x + f y

theorem Cauchy_solution_on_X :
  (∀ a b : ℤ, ∀ c d : ℤ, a + b * Real.sqrt 2 = c + d * Real.sqrt 2 → a = c ∧ b = d) →
  is_morphism f →
  ∃ x y : ℝ, ∀ a b : ℤ,
    f (a + b * Real.sqrt 2) = a * x + b * y :=
by
  intros h1 h2
  let x := f 1
  let y := f (Real.sqrt 2)
  exists x, y
  intros a b
  sorry

end CauchyEquation

end NUMINAMATH_GPT_Cauchy_solution_on_X_l532_53224


namespace NUMINAMATH_GPT_unique_triplet_l532_53288

theorem unique_triplet (a b p : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hb : 0 < b) :
  (1 / (p : ℚ) = 1 / (a^2 : ℚ) + 1 / (b^2 : ℚ)) → (a = 2 ∧ b = 2 ∧ p = 2) :=
by
  sorry

end NUMINAMATH_GPT_unique_triplet_l532_53288


namespace NUMINAMATH_GPT_bounds_of_F_and_G_l532_53275

noncomputable def F (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def G (a b c x : ℝ) : ℝ := c * x^2 + b * x + a

theorem bounds_of_F_and_G {a b c : ℝ}
  (hF0 : |F a b c 0| ≤ 1)
  (hF1 : |F a b c 1| ≤ 1)
  (hFm1 : |F a b c (-1)| ≤ 1) :
  (∀ x, |x| ≤ 1 → |F a b c x| ≤ 5/4) ∧
  (∀ x, |x| ≤ 1 → |G a b c x| ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_bounds_of_F_and_G_l532_53275


namespace NUMINAMATH_GPT_translated_upwards_2_units_l532_53215

theorem translated_upwards_2_units (x : ℝ) : (x + 2 > 0) → (x > -2) :=
by 
  intros h
  exact sorry

end NUMINAMATH_GPT_translated_upwards_2_units_l532_53215


namespace NUMINAMATH_GPT_scientific_notation_of_750000_l532_53221

theorem scientific_notation_of_750000 : 750000 = 7.5 * 10^5 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_750000_l532_53221


namespace NUMINAMATH_GPT_eval_expression_at_minus_3_l532_53266

theorem eval_expression_at_minus_3 :
  (5 + 2 * x * (x + 2) - 4^2) / (x - 4 + x^2) = -5 / 2 :=
by
  let x := -3
  sorry

end NUMINAMATH_GPT_eval_expression_at_minus_3_l532_53266


namespace NUMINAMATH_GPT_odd_phone_calls_are_even_l532_53247

theorem odd_phone_calls_are_even (n : ℕ) : Even (2 * n) :=
by
  sorry

end NUMINAMATH_GPT_odd_phone_calls_are_even_l532_53247


namespace NUMINAMATH_GPT_point_reflection_correct_l532_53276

def point_reflection_y_axis (x y z : ℝ) : ℝ × ℝ × ℝ :=
  (-x, y, -z)

theorem point_reflection_correct :
  point_reflection_y_axis (-3) 5 2 = (3, 5, -2) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_point_reflection_correct_l532_53276


namespace NUMINAMATH_GPT_floor_add_frac_eq_154_l532_53264

theorem floor_add_frac_eq_154 (r : ℝ) (h : ⌊r⌋ + r = 15.4) : r = 7.4 := 
sorry

end NUMINAMATH_GPT_floor_add_frac_eq_154_l532_53264


namespace NUMINAMATH_GPT_midpoint_sum_l532_53293

theorem midpoint_sum (x y : ℝ) (h1 : (x + 0) / 2 = 2) (h2 : (y + 9) / 2 = 4) : x + y = 3 := by
  sorry

end NUMINAMATH_GPT_midpoint_sum_l532_53293


namespace NUMINAMATH_GPT_find_d1_over_d2_l532_53244

variables {k c1 c2 d1 d2 : ℝ}
variables (c1_nonzero : c1 ≠ 0) (c2_nonzero : c2 ≠ 0) 
variables (d1_nonzero : d1 ≠ 0) (d2_nonzero : d2 ≠ 0)
variables (h1 : c1 * d1 = k) (h2 : c2 * d2 = k)
variables (h3 : c1 / c2 = 3 / 4)

theorem find_d1_over_d2 : d1 / d2 = 4 / 3 :=
sorry

end NUMINAMATH_GPT_find_d1_over_d2_l532_53244


namespace NUMINAMATH_GPT_height_difference_l532_53265

theorem height_difference :
  let janet_height := 3.6666666666666665
  let sister_height := 2.3333333333333335
  janet_height - sister_height = 1.333333333333333 :=
by
  sorry

end NUMINAMATH_GPT_height_difference_l532_53265


namespace NUMINAMATH_GPT_min_value_f_l532_53267

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 1/2 * Real.cos (2 * x) - 1

theorem min_value_f : ∃ x : ℝ, f x = -5/2 := sorry

end NUMINAMATH_GPT_min_value_f_l532_53267


namespace NUMINAMATH_GPT_arithmetic_seq_formula_sum_first_n_terms_l532_53256

/-- Define the given arithmetic sequence an -/
def arithmetic_seq (a1 d : ℤ) : ℕ → ℤ
| 0       => a1
| (n + 1) => arithmetic_seq a1 d n + d

variable {a3 a7 : ℤ}
variable (a3_eq : arithmetic_seq 1 2 2 = 5)
variable (a7_eq : arithmetic_seq 1 2 6 = 13)

/-- Define the sequence bn -/
def b_seq (n : ℕ) : ℚ :=
  1 / ((2 * n + 1) * (arithmetic_seq 1 2 n))

/-- Define the sum of the first n terms of the sequence bn -/
def sum_b_seq : ℕ → ℚ
| 0       => 0
| (n + 1) => sum_b_seq n + b_seq (n + 1)
          
theorem arithmetic_seq_formula:
  ∀ (n : ℕ), arithmetic_seq 1 2 n = 2 * n - 1 :=
by
  intros
  sorry

theorem sum_first_n_terms:
  ∀ (n : ℕ), sum_b_seq n = n / (2 * n + 1) :=
by
  intros
  sorry

end NUMINAMATH_GPT_arithmetic_seq_formula_sum_first_n_terms_l532_53256


namespace NUMINAMATH_GPT_probability_intersection_l532_53261

variables (A B : Type → Prop)

-- Assuming we have a measure space (probability) P
variables {P : Type → Prop}

-- Given probabilities
def p_A := 0.65
def p_B := 0.55
def p_Ac_Bc := 0.20

-- The theorem to be proven
theorem probability_intersection :
  (p_A + p_B - (1 - p_Ac_Bc) = 0.40) :=
by
  sorry

end NUMINAMATH_GPT_probability_intersection_l532_53261


namespace NUMINAMATH_GPT_equal_cubes_l532_53263

theorem equal_cubes (a : ℤ) : -(a ^ 3) = (-a) ^ 3 :=
by
  sorry

end NUMINAMATH_GPT_equal_cubes_l532_53263


namespace NUMINAMATH_GPT_hakimi_age_is_40_l532_53257

variable (H : ℕ)
variable (Jared_age : ℕ) (Molly_age : ℕ := 30)
variable (total_age : ℕ := 120)

theorem hakimi_age_is_40 (h1 : Jared_age = H + 10) (h2 : H + Jared_age + Molly_age = total_age) : H = 40 :=
by
  sorry

end NUMINAMATH_GPT_hakimi_age_is_40_l532_53257


namespace NUMINAMATH_GPT_complex_multiply_cis_l532_53201

open Complex

theorem complex_multiply_cis :
  (4 * (cos (25 * Real.pi / 180) + sin (25 * Real.pi / 180) * I)) *
  (-3 * (cos (48 * Real.pi / 180) + sin (48 * Real.pi / 180) * I)) =
  12 * (cos (253 * Real.pi / 180) + sin (253 * Real.pi / 180) * I) :=
sorry

end NUMINAMATH_GPT_complex_multiply_cis_l532_53201


namespace NUMINAMATH_GPT_value_of_a_l532_53254

theorem value_of_a (a : ℕ) (h : ∀ x, ((a - 2) * x > a - 2) ↔ (x < 1)) : a = 0 ∨ a = 1 := by
  sorry

end NUMINAMATH_GPT_value_of_a_l532_53254


namespace NUMINAMATH_GPT_solve_for_y_l532_53248

theorem solve_for_y (y : ℝ) (h : 3 * y ^ (1 / 4) - 5 * (y / y ^ (3 / 4)) = 2 + y ^ (1 / 4)) : y = 16 / 81 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l532_53248


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l532_53217

noncomputable def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h_seq : arithmetic_sequence a)
  (h_a5 : a 5 = 3)
  (h_a7 : a 7 = 7) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l532_53217


namespace NUMINAMATH_GPT_intersection_M_N_l532_53228

open Set

def M := {x : ℝ | x^2 - 2 * x - 3 ≤ 0}
def N := {x : ℝ | 0 < x}
def intersection := {x : ℝ | 0 < x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = intersection := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l532_53228


namespace NUMINAMATH_GPT_find_number_l532_53226

theorem find_number (x : ℝ) 
  (h : 0.4 * x + (0.3 * 0.2) = 0.26) : x = 0.5 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l532_53226


namespace NUMINAMATH_GPT_max_abs_sum_of_squares_eq_2_sqrt_2_l532_53237

theorem max_abs_sum_of_squares_eq_2_sqrt_2 (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_max_abs_sum_of_squares_eq_2_sqrt_2_l532_53237


namespace NUMINAMATH_GPT_books_added_l532_53252

theorem books_added (initial_books sold_books current_books added_books : ℕ)
  (h1 : initial_books = 4)
  (h2 : sold_books = 3)
  (h3 : current_books = 11)
  (h4 : added_books = current_books - (initial_books - sold_books)) :
  added_books = 10 :=
by
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end NUMINAMATH_GPT_books_added_l532_53252


namespace NUMINAMATH_GPT_sum_of_factors_36_eq_91_l532_53260

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end NUMINAMATH_GPT_sum_of_factors_36_eq_91_l532_53260


namespace NUMINAMATH_GPT_second_worker_time_l532_53203

theorem second_worker_time 
  (first_worker_rate : ℝ)
  (combined_rate : ℝ)
  (x : ℝ)
  (h1 : first_worker_rate = 1 / 6)
  (h2 : combined_rate = 1 / 2.4) :
  (1 / 6) + (1 / x) = combined_rate → x = 4 := 
by 
  intros h
  sorry

end NUMINAMATH_GPT_second_worker_time_l532_53203


namespace NUMINAMATH_GPT_isosceles_triangle_base_angle_l532_53284

-- Define the problem and the given conditions
theorem isosceles_triangle_base_angle (A B C : ℝ)
(h_triangle : A + B + C = 180)
(h_isosceles : (A = B ∨ B = C ∨ C = A))
(h_ratio : (A = B / 2 ∨ B = C / 2 ∨ C = A / 2)) :
(A = 45 ∨ A = 72) ∨ (B = 45 ∨ B = 72) ∨ (C = 45 ∨ C = 72) :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_base_angle_l532_53284


namespace NUMINAMATH_GPT_number_of_integers_l532_53271

theorem number_of_integers (n : ℤ) : 
  (16 < n^2) → (n^2 < 121) → n = -10 ∨ n = -9 ∨ n = -8 ∨ n = -7 ∨ n = -6 ∨ n = -5 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 10 := 
by
  sorry

end NUMINAMATH_GPT_number_of_integers_l532_53271


namespace NUMINAMATH_GPT_cubic_ineq_solution_l532_53213

theorem cubic_ineq_solution (x : ℝ) :
  (4 < x ∧ x < 4 + 2 * Real.sqrt 3) ∨ (x > 4 + 2 * Real.sqrt 3) → (x^3 - 12 * x^2 + 44 * x - 16 > 0) :=
by
  sorry

end NUMINAMATH_GPT_cubic_ineq_solution_l532_53213


namespace NUMINAMATH_GPT_min_value_quadratic_l532_53211

theorem min_value_quadratic (x : ℝ) : ∃ x, x = -7 ∧ (x^2 + 14 * x + 24 = -25) := sorry

end NUMINAMATH_GPT_min_value_quadratic_l532_53211


namespace NUMINAMATH_GPT_trains_cross_time_l532_53200

theorem trains_cross_time (length1 length2 : ℕ) (time1 time2 : ℕ) 
  (speed1 speed2 relative_speed total_length : ℚ) 
  (h1 : length1 = 120) (h2 : length2 = 150) 
  (h3 : time1 = 10) (h4 : time2 = 15) 
  (h5 : speed1 = length1 / time1) (h6 : speed2 = length2 / time2) 
  (h7 : relative_speed = speed1 - speed2) 
  (h8 : total_length = length1 + length2) : 
  (total_length / relative_speed = 135) := 
by sorry

end NUMINAMATH_GPT_trains_cross_time_l532_53200


namespace NUMINAMATH_GPT_least_expensive_trip_is_1627_44_l532_53290

noncomputable def least_expensive_trip_cost : ℝ :=
  let distance_DE := 4500
  let distance_DF := 4000
  let distance_EF := Real.sqrt (distance_DE ^ 2 - distance_DF ^ 2)
  let cost_bus (distance : ℝ) : ℝ := distance * 0.20
  let cost_plane (distance : ℝ) : ℝ := distance * 0.12 + 120
  let cost_DE := min (cost_bus distance_DE) (cost_plane distance_DE)
  let cost_EF := min (cost_bus distance_EF) (cost_plane distance_EF)
  let cost_DF := min (cost_bus distance_DF) (cost_plane distance_DF)
  cost_DE + cost_EF + cost_DF

theorem least_expensive_trip_is_1627_44 :
  least_expensive_trip_cost = 1627.44 := sorry

end NUMINAMATH_GPT_least_expensive_trip_is_1627_44_l532_53290


namespace NUMINAMATH_GPT_commute_distance_l532_53232

theorem commute_distance (D : ℝ)
  (h1 : ∀ t : ℝ, t > 0 → t = D / 45)
  (h2 : ∀ t : ℝ, t > 0 → t = D / 30)
  (h3 : D / 45 + D / 30 = 1) :
  D = 18 :=
by
  sorry

end NUMINAMATH_GPT_commute_distance_l532_53232


namespace NUMINAMATH_GPT_sum_of_altitudes_less_than_sum_of_sides_l532_53274

theorem sum_of_altitudes_less_than_sum_of_sides 
  (a b c h_a h_b h_c K : ℝ) 
  (triangle_area : K = (1/2) * a * h_a)
  (h_a_def : h_a = 2 * K / a) 
  (h_b_def : h_b = 2 * K / b)
  (h_c_def : h_c = 2 * K / c) : 
  h_a + h_b + h_c < a + b + c := by
  sorry

end NUMINAMATH_GPT_sum_of_altitudes_less_than_sum_of_sides_l532_53274


namespace NUMINAMATH_GPT_fraction_ratio_l532_53289

theorem fraction_ratio
  (m n p q r : ℚ)
  (h1 : m / n = 20)
  (h2 : p / n = 4)
  (h3 : p / q = 1 / 5)
  (h4 : m / r = 10) :
  r / q = 1 / 10 :=
by
  sorry

end NUMINAMATH_GPT_fraction_ratio_l532_53289


namespace NUMINAMATH_GPT_steve_popsicle_sticks_l532_53206

theorem steve_popsicle_sticks (S Sid Sam : ℕ) (h1 : Sid = 2 * S) (h2 : Sam = 3 * Sid) (h3 : S + Sid + Sam = 108) : S = 12 :=
by
  sorry

end NUMINAMATH_GPT_steve_popsicle_sticks_l532_53206


namespace NUMINAMATH_GPT_a_range_l532_53249

theorem a_range (x y z a : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 1)
  (h_eq : a / (x * y * z) = (1 / x) + (1 / y) + (1 / z) - 2) :
  0 < a ∧ a ≤ 7 / 27 :=
sorry

end NUMINAMATH_GPT_a_range_l532_53249


namespace NUMINAMATH_GPT_quadruplet_babies_l532_53246

variable (a b c : ℕ)
variable (h1 : b = 3 * c)
variable (h2 : a = 5 * b)
variable (h3 : 2 * a + 3 * b + 4 * c = 1500)

theorem quadruplet_babies : 4 * c = 136 := by
  sorry

end NUMINAMATH_GPT_quadruplet_babies_l532_53246


namespace NUMINAMATH_GPT_find_x_plus_y_l532_53299

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2008) (h2 : x + 2008 * Real.cos y = 2007) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_plus_y_l532_53299


namespace NUMINAMATH_GPT_simplify_expression_l532_53241

theorem simplify_expression :
  (3^4 + 3^2) / (3^3 - 3) = 15 / 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_simplify_expression_l532_53241


namespace NUMINAMATH_GPT_coin_flip_sequences_count_l532_53208

theorem coin_flip_sequences_count : 
  let total_flips := 10;
  let heads_fixed := 2;
  (2 : ℕ) ^ (total_flips - heads_fixed) = 256 := 
by 
  sorry

end NUMINAMATH_GPT_coin_flip_sequences_count_l532_53208
