import Mathlib

namespace NUMINAMATH_GPT_even_odd_difference_l1462_146232

def even_sum_n (n : ℕ) : ℕ := (n * (n + 1))
def odd_sum_n (n : ℕ) : ℕ := n * n

theorem even_odd_difference : even_sum_n 100 - odd_sum_n 100 = 100 := by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_even_odd_difference_l1462_146232


namespace NUMINAMATH_GPT_tan_105_eq_neg2_sub_sqrt3_l1462_146285

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_tan_105_eq_neg2_sub_sqrt3_l1462_146285


namespace NUMINAMATH_GPT_tangent_line_correct_l1462_146222

-- Define the curve y = x^3 - 1
def curve (x : ℝ) : ℝ := x^3 - 1

-- Define the derivative of the curve
def derivative_curve (x : ℝ) : ℝ := 3 * x^2

-- Define the point of tangency
def tangent_point : ℝ × ℝ := (1, curve 1)

-- Define the tangent line equation at x = 1
def tangent_line (x : ℝ) : ℝ := 3 * x - 3

-- The formal statement to be proven
theorem tangent_line_correct :
  ∀ x : ℝ, curve x = x^3 - 1 ∧ derivative_curve x = 3 * x^2 ∧ tangent_point = (1, 0) → 
    tangent_line 1 = 3 * 1 - 3 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_correct_l1462_146222


namespace NUMINAMATH_GPT_plane_speed_east_l1462_146220

def plane_travel_problem (v : ℕ) : Prop :=
  let time : ℕ := 35 / 10 
  let distance_east := v * time
  let distance_west := 275 * time
  let total_distance := distance_east + distance_west
  total_distance = 2100

theorem plane_speed_east : ∃ v : ℕ, plane_travel_problem v ∧ v = 325 :=
sorry

end NUMINAMATH_GPT_plane_speed_east_l1462_146220


namespace NUMINAMATH_GPT_relationship_coefficients_l1462_146204

-- Definitions based directly on the conditions
def has_extrema (a b c : ℝ) : Prop := b^2 - 3 * a * c > 0
def passes_through_origin (x1 x2 y1 y2 : ℝ) : Prop := x1 * y2 = x2 * y1

-- Main statement proving the relationship among the coefficients
theorem relationship_coefficients (a b c d : ℝ) (x1 x2 y1 y2 : ℝ)
  (h_extrema : has_extrema a b c)
  (h_line : passes_through_origin x1 x2 y1 y2)
  (hx1 : x1 ≠ 0) (hx2 : x2 ≠ 0)
  (h_y1 : y1 = a * x1^3 + b * x1^2 + c * x1 + d)
  (h_y2 : y2 = a * x2^3 + b * x2^2 + c * x2 + d) :
  9 * a * d = b * c :=
sorry

end NUMINAMATH_GPT_relationship_coefficients_l1462_146204


namespace NUMINAMATH_GPT_intersect_point_l1462_146289

noncomputable def f (x : ℤ) (b : ℤ) : ℤ := 5 * x + b
noncomputable def f_inv (x : ℤ) (b : ℤ) : ℤ := (x - b) / 5

theorem intersect_point (a b : ℤ) (h_intersections : (f (-3) b = a ∧ f a b = -3)) : a = -3 :=
by
  sorry

end NUMINAMATH_GPT_intersect_point_l1462_146289


namespace NUMINAMATH_GPT_megan_initial_markers_l1462_146212

theorem megan_initial_markers (gave : ℕ) (total : ℕ) (initial : ℕ) 
  (h1 : gave = 109) 
  (h2 : total = 326) 
  (h3 : initial + gave = total) : 
  initial = 217 := 
by 
  sorry

end NUMINAMATH_GPT_megan_initial_markers_l1462_146212


namespace NUMINAMATH_GPT_sum_quotient_product_diff_l1462_146202

theorem sum_quotient_product_diff (x y : ℚ) (h₁ : x + y = 6) (h₂ : x / y = 6) : 
  (x * y) - (x - y) = 6 / 49 :=
  sorry

end NUMINAMATH_GPT_sum_quotient_product_diff_l1462_146202


namespace NUMINAMATH_GPT_profit_ratio_a_to_b_l1462_146278

noncomputable def capital_a : ℕ := 3500
noncomputable def time_a : ℕ := 12
noncomputable def capital_b : ℕ := 10500
noncomputable def time_b : ℕ := 6

noncomputable def capital_months (capital : ℕ) (time : ℕ) : ℕ :=
  capital * time

noncomputable def capital_months_a : ℕ :=
  capital_months capital_a time_a

noncomputable def capital_months_b : ℕ :=
  capital_months capital_b time_b

theorem profit_ratio_a_to_b : (capital_months_a / Nat.gcd capital_months_a capital_months_b) =
                             2 ∧
                             (capital_months_b / Nat.gcd capital_months_a capital_months_b) =
                             3 := 
by
  sorry

end NUMINAMATH_GPT_profit_ratio_a_to_b_l1462_146278


namespace NUMINAMATH_GPT_Jackie_exercise_hours_l1462_146219

variable (work_hours : ℕ) (sleep_hours : ℕ) (free_time_hours : ℕ) (total_hours_in_day : ℕ)
variable (time_for_exercise : ℕ)

noncomputable def prove_hours_exercising (work_hours sleep_hours free_time_hours total_hours_in_day : ℕ) : Prop :=
  work_hours = 8 ∧
  sleep_hours = 8 ∧
  free_time_hours = 5 ∧
  total_hours_in_day = 24 → 
  time_for_exercise = total_hours_in_day - (work_hours + sleep_hours + free_time_hours)

theorem Jackie_exercise_hours :
  prove_hours_exercising 8 8 5 24 3 :=
by
  -- Proof is omitted as per instruction
  sorry

end NUMINAMATH_GPT_Jackie_exercise_hours_l1462_146219


namespace NUMINAMATH_GPT_circles_max_ab_l1462_146256

theorem circles_max_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ (x y : ℝ), (x + a)^2 + (y - 2)^2 = 1 ∧ (x - b)^2 + (y - 2)^2 = 4) →
  a + b = 3 →
  ab ≤ 9 / 4 := 
  by
  sorry

end NUMINAMATH_GPT_circles_max_ab_l1462_146256


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1462_146238

variable {a b : ℝ}
variable (h1 : a > 0) (h2 : b > 0)
variable (h3 : (a : ℝ) / (b : ℝ) = 3)

theorem hyperbola_eccentricity (h1 : a > 0) (h2 : b > 0) (h3 : b / a = 1 / 3) : 
  (Real.sqrt ((a ^ 2 + b ^ 2) / (a ^ 2))) = Real.sqrt 10 := by sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1462_146238


namespace NUMINAMATH_GPT_equation_solution_l1462_146223

open Real

theorem equation_solution (x : ℝ) : 
  (x = 4 ∨ x = -1 → 3 * (2 * x - 5) ≠ (2 * x - 5) ^ 2) ∧
  (3 * (2 * x - 5) = (2 * x - 5) ^ 2 → x = 5 / 2 ∨ x = 4) :=
by
  sorry

end NUMINAMATH_GPT_equation_solution_l1462_146223


namespace NUMINAMATH_GPT_minimum_guests_l1462_146241

theorem minimum_guests (x : ℕ) : (120 + 18 * x > 250 + 15 * x) → (x ≥ 44) := by
  intro h
  sorry

end NUMINAMATH_GPT_minimum_guests_l1462_146241


namespace NUMINAMATH_GPT_line_passing_quadrants_l1462_146299

theorem line_passing_quadrants (a k : ℝ) (a_nonzero : a ≠ 0)
  (x1 x2 y1 y2 : ℝ) (hx1 : y1 = a * x1^2 - a) (hx2 : y2 = a * x2^2 - a)
  (hx1_y1 : y1 = k * x1) (hx2_y2 : y2 = k * x2) 
  (sum_x : x1 + x2 < 0) : 
  ∃ (q1 q4 : (ℝ × ℝ)), 
  (q1.1 > 0 ∧ q1.2 > 0 ∧ q1.2 = a * q1.1 + k) ∧ (q4.1 > 0 ∧ q4.2 < 0 ∧ q4.2 = a * q4.1 + k) := 
sorry

end NUMINAMATH_GPT_line_passing_quadrants_l1462_146299


namespace NUMINAMATH_GPT_probability_neither_red_nor_purple_l1462_146298

section Probability

def total_balls : ℕ := 60
def red_balls : ℕ := 15
def purple_balls : ℕ := 3
def total_red_or_purple_balls : ℕ := red_balls + purple_balls
def non_red_or_purple_balls : ℕ := total_balls - total_red_or_purple_balls

theorem probability_neither_red_nor_purple :
  (non_red_or_purple_balls : ℚ) / (total_balls : ℚ) = 7 / 10 :=
by
  sorry

end Probability

end NUMINAMATH_GPT_probability_neither_red_nor_purple_l1462_146298


namespace NUMINAMATH_GPT_cuboid_height_l1462_146270

-- Define the base area and volume of the cuboid
def base_area : ℝ := 50
def volume : ℝ := 2000

-- Prove that the height is 40 cm given the base area and volume
theorem cuboid_height : volume / base_area = 40 := by
  sorry

end NUMINAMATH_GPT_cuboid_height_l1462_146270


namespace NUMINAMATH_GPT_average_grade_of_male_students_l1462_146272

theorem average_grade_of_male_students (M : ℝ) (H1 : (90 : ℝ) = (8 + 32 : ℝ) / 40) 
(H2 : (92 : ℝ) = 32 / 40) :
  M = 82 := 
sorry

end NUMINAMATH_GPT_average_grade_of_male_students_l1462_146272


namespace NUMINAMATH_GPT_wire_length_after_two_bends_is_three_l1462_146297

-- Let's define the initial length and the property of bending the wire.
def initial_length : ℕ := 12

def half_length (length : ℕ) : ℕ :=
  length / 2

-- Define the final length after two bends.
def final_length_after_two_bends : ℕ :=
  half_length (half_length initial_length)

-- The theorem stating that the final length is 3 cm after two bends.
theorem wire_length_after_two_bends_is_three :
  final_length_after_two_bends = 3 :=
by
  -- The proof can be added later.
  sorry

end NUMINAMATH_GPT_wire_length_after_two_bends_is_three_l1462_146297


namespace NUMINAMATH_GPT_days_to_complete_l1462_146266

variable {m n : ℕ}

theorem days_to_complete (h : ∀ (m n : ℕ), (m + n) * m = 1) : 
  ∀ (n m : ℕ), (m * (m + n)) / n = m * (m + n) / n :=
by
  sorry

end NUMINAMATH_GPT_days_to_complete_l1462_146266


namespace NUMINAMATH_GPT_general_term_of_sequence_l1462_146211

theorem general_term_of_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) 
    (h1 : ∀ n, S n = 2 * a n - 1) 
    (a₁ : a 1 = 1) :
  ∀ n, a n = 2^(n - 1) := 
sorry

end NUMINAMATH_GPT_general_term_of_sequence_l1462_146211


namespace NUMINAMATH_GPT_work_days_together_l1462_146263

theorem work_days_together (A_rate B_rate : ℚ) (h1 : A_rate = 1 / 12) (h2 : B_rate = 5 / 36) : 
  1 / (A_rate + B_rate) = 4.5 := by
  sorry

end NUMINAMATH_GPT_work_days_together_l1462_146263


namespace NUMINAMATH_GPT_max_possible_scores_l1462_146215

theorem max_possible_scores (num_questions : ℕ) (points_correct : ℤ) (points_incorrect : ℤ) (points_unanswered : ℤ) :
  num_questions = 10 →
  points_correct = 4 →
  points_incorrect = -1 →
  points_unanswered = 0 →
  ∃ n, n = 45 :=
by
  sorry

end NUMINAMATH_GPT_max_possible_scores_l1462_146215


namespace NUMINAMATH_GPT_units_digit_sum_base8_l1462_146249

theorem units_digit_sum_base8 : 
  ∀ (x y : ℕ), (x = 64 ∧ y = 34 ∧ (x % 8 = 4) ∧ (y % 8 = 4) → (x + y) % 8 = 0) :=
by
  sorry

end NUMINAMATH_GPT_units_digit_sum_base8_l1462_146249


namespace NUMINAMATH_GPT_angle_y_is_80_l1462_146260

def parallel (m n : ℝ) : Prop := sorry

def angle_at_base (θ : ℝ) := θ = 40
def right_angle (θ : ℝ) := θ = 90
def exterior_angle (θ1 θ2 : ℝ) := θ1 + θ2 = 180

theorem angle_y_is_80 (m n : ℝ) (θ1 θ2 θ3 θ_ext : ℝ) :
  parallel m n →
  angle_at_base θ1 →
  right_angle θ2 →
  angle_at_base θ3 →
  exterior_angle θ_ext θ3 →
  θ_ext = 80 := by
  sorry

end NUMINAMATH_GPT_angle_y_is_80_l1462_146260


namespace NUMINAMATH_GPT_inequality_x4_y4_z2_l1462_146250

theorem inequality_x4_y4_z2 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    x^4 + y^4 + z^2 ≥  xyz * 8^(1/2) :=
  sorry

end NUMINAMATH_GPT_inequality_x4_y4_z2_l1462_146250


namespace NUMINAMATH_GPT_train_pass_time_l1462_146261

-- Definitions based on conditions
def train_length : Float := 250
def pole_time : Float := 10
def platform_length : Float := 1250
def incline_angle : Float := 5 -- degrees
def speed_reduction_factor : Float := 0.75

-- The statement to be proved
theorem train_pass_time :
  let original_speed := train_length / pole_time
  let incline_speed := original_speed * speed_reduction_factor
  let total_distance := train_length + platform_length
  let time_to_pass_platform := total_distance / incline_speed
  time_to_pass_platform = 80 := by
  simp [train_length, pole_time, platform_length, incline_angle, speed_reduction_factor]
  sorry

end NUMINAMATH_GPT_train_pass_time_l1462_146261


namespace NUMINAMATH_GPT_largest_divisor_69_86_l1462_146210

theorem largest_divisor_69_86 (n : ℕ) (h₁ : 69 % n = 5) (h₂ : 86 % n = 6) : n = 16 := by
  sorry

end NUMINAMATH_GPT_largest_divisor_69_86_l1462_146210


namespace NUMINAMATH_GPT_high_probability_event_is_C_l1462_146239

-- Define the probabilities of events A, B, and C
def prob_A : ℝ := 0.5
def prob_B : ℝ := 0.1
def prob_C : ℝ := 0.9

-- Statement asserting Event C has the high possibility of occurring
theorem high_probability_event_is_C : prob_C > prob_A ∧ prob_C > prob_B :=
by
  sorry

end NUMINAMATH_GPT_high_probability_event_is_C_l1462_146239


namespace NUMINAMATH_GPT_find_other_side_length_l1462_146206

variable (total_shingles : ℕ)
variable (shingles_per_sqft : ℕ)
variable (num_roofs : ℕ)
variable (side_length : ℕ)

theorem find_other_side_length
  (h1 : total_shingles = 38400)
  (h2 : shingles_per_sqft = 8)
  (h3 : num_roofs = 3)
  (h4 : side_length = 20)
  : (total_shingles / shingles_per_sqft / num_roofs / 2) / side_length = 40 :=
by
  sorry

end NUMINAMATH_GPT_find_other_side_length_l1462_146206


namespace NUMINAMATH_GPT_valentino_farm_total_birds_l1462_146248

-- The definitions/conditions from the problem statement
def chickens := 200
def ducks := 2 * chickens
def turkeys := 3 * ducks

-- The theorem to prove the total number of birds
theorem valentino_farm_total_birds : 
  chickens + ducks + turkeys = 1800 :=
by
  -- Proof is not required, so we use 'sorry'
  sorry

end NUMINAMATH_GPT_valentino_farm_total_birds_l1462_146248


namespace NUMINAMATH_GPT_sequence_general_formula_l1462_146273

theorem sequence_general_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) 
  (hS : ∀ n, S n = 3 / 2 * a n - 3) : 
  (∀ n, a n = 2 * 3 ^ n) :=
by 
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l1462_146273


namespace NUMINAMATH_GPT_greatest_integer_solution_l1462_146226

theorem greatest_integer_solution (n : ℤ) (h : n^2 - 13 * n + 40 ≤ 0) : n ≤ 8 :=
sorry

end NUMINAMATH_GPT_greatest_integer_solution_l1462_146226


namespace NUMINAMATH_GPT_positive_intervals_of_product_l1462_146231

theorem positive_intervals_of_product (x : ℝ) : 
  ((x + 2) * (x - 3) > 0) ↔ (x < -2 ∨ x > 3) := 
sorry

end NUMINAMATH_GPT_positive_intervals_of_product_l1462_146231


namespace NUMINAMATH_GPT_function_positivity_range_l1462_146218

theorem function_positivity_range (m x : ℝ): 
  (∀ x, (2 * x^2 + (4 - m) * x + 4 - m > 0) ∨ (m * x > 0)) ↔ m < 4 :=
sorry

end NUMINAMATH_GPT_function_positivity_range_l1462_146218


namespace NUMINAMATH_GPT_real_and_equal_roots_condition_l1462_146213

theorem real_and_equal_roots_condition (k : ℝ) : 
  ∀ k : ℝ, (∃ (x : ℝ), 3 * x^2 + 6 * k * x + 9 = 0) ↔ (k = Real.sqrt 3 ∨ k = -Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_real_and_equal_roots_condition_l1462_146213


namespace NUMINAMATH_GPT_find_sides_of_rectangle_l1462_146258

-- Define the conditions
def isRectangle (w l : ℝ) : Prop :=
  l = 3 * w ∧ 2 * l + 2 * w = l * w

-- Main theorem statement
theorem find_sides_of_rectangle (w l : ℝ) :
  isRectangle w l → w = 8 / 3 ∧ l = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_sides_of_rectangle_l1462_146258


namespace NUMINAMATH_GPT_expression_nonnegative_l1462_146201

theorem expression_nonnegative (x : ℝ) : 
  0 ≤ x → x < 3 → 0 ≤ (x - 12 * x^2 + 36 * x^3) / (9 - x^3) :=
  sorry

end NUMINAMATH_GPT_expression_nonnegative_l1462_146201


namespace NUMINAMATH_GPT_triangle_count_with_perimeter_11_l1462_146284

theorem triangle_count_with_perimeter_11 :
  ∃ (s : Finset (ℕ × ℕ × ℕ)), s.card = 5 ∧ ∀ (a b c : ℕ), (a, b, c) ∈ s ->
    a ≤ b ∧ b ≤ c ∧ a + b + c = 11 ∧ a + b > c :=
sorry

end NUMINAMATH_GPT_triangle_count_with_perimeter_11_l1462_146284


namespace NUMINAMATH_GPT_part1_l1462_146262

def setA (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x - a^2 - 2*a < 0}
def setB (a : ℝ) : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 3^x - 2*a ∧ x ≤ 2}

theorem part1 (a : ℝ) (h : a = 3) : setA 3 ∪ setB 3 = Set.Ioo (-6) 5 :=
by
  sorry

end NUMINAMATH_GPT_part1_l1462_146262


namespace NUMINAMATH_GPT_rice_mixture_ratio_l1462_146290

theorem rice_mixture_ratio (x y : ℝ) (h1 : 7 * x + 8.75 * y = 7.50 * (x + y)) : x / y = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_rice_mixture_ratio_l1462_146290


namespace NUMINAMATH_GPT_area_of_shaded_triangle_l1462_146296

-- Definitions of the conditions
def AC := 4
def BC := 3
def BD := 10
def CD := BD - BC

-- Statement of the proof problem
theorem area_of_shaded_triangle :
  (1 / 2 * CD * AC = 14) := by
  sorry

end NUMINAMATH_GPT_area_of_shaded_triangle_l1462_146296


namespace NUMINAMATH_GPT_inequality_l1462_146288

theorem inequality (a b c d e p q : ℝ) 
  (h0 : 0 < p ∧ p ≤ a ∧ p ≤ b ∧ p ≤ c ∧ p ≤ d ∧ p ≤ e)
  (h1 : a ≤ q ∧ b ≤ q ∧ c ≤ q ∧ d ≤ q ∧ e ≤ q) :
  (a + b + c + d + e) * ((1 / a) + (1 / b) + (1 / c) + (1 / d) + (1 / e)) 
  ≤ 25 + 6 * (Real.sqrt (p / q) - Real.sqrt (q / p))^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_l1462_146288


namespace NUMINAMATH_GPT_spring_extension_l1462_146264

theorem spring_extension (A1 A2 : ℝ) (x1 x2 : ℝ) (hA1 : A1 = 29.43) (hx1 : x1 = 0.05) (hA2 : A2 = 9.81) : x2 = 0.029 :=
by 
  sorry

end NUMINAMATH_GPT_spring_extension_l1462_146264


namespace NUMINAMATH_GPT_convert_to_base5_l1462_146268

theorem convert_to_base5 : ∀ n : ℕ, n = 1729 → Nat.digits 5 n = [2, 3, 4, 0, 4] :=
by
  intros n hn
  rw [hn]
  -- proof steps can be filled in here
  sorry

end NUMINAMATH_GPT_convert_to_base5_l1462_146268


namespace NUMINAMATH_GPT_smallest_number_is_51_l1462_146292

-- Definitions based on conditions
def conditions (x y : ℕ) : Prop :=
  (x + y = 2014) ∧ (∃ n a : ℕ, (x = 100 * n + a) ∧ (a < 100) ∧ (3 * n = y + 6))

-- The proof problem statement that needs to be proven
theorem smallest_number_is_51 :
  ∃ x y : ℕ, conditions x y ∧ min x y = 51 := 
sorry

end NUMINAMATH_GPT_smallest_number_is_51_l1462_146292


namespace NUMINAMATH_GPT_elimination_method_equation_y_l1462_146275

theorem elimination_method_equation_y (x y : ℝ)
    (h1 : 5 * x - 3 * y = -5)
    (h2 : 5 * x + 4 * y = -1) :
    7 * y = 4 :=
by
  -- Adding the required conditions as hypotheses and skipping the proof.
  sorry

end NUMINAMATH_GPT_elimination_method_equation_y_l1462_146275


namespace NUMINAMATH_GPT_range_of_a_l1462_146294

def A := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) := {x : ℝ | x^2 + 2*(a+1)*x + (a^2 -1) = 0}

theorem range_of_a (a : ℝ) :
  (A ∩ B a = B a) → (a = 1 ∨ a ≤ -1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1462_146294


namespace NUMINAMATH_GPT_find_P_x_l1462_146277

noncomputable def P (x : ℝ) : ℝ :=
  (-17 / 3) * x^3 + (68 / 3) * x^2 - (31 / 3) * x - 18

variable (a b c : ℝ)

axiom h1 : a^3 - 4 * a^2 + 2 * a + 3 = 0
axiom h2 : b^3 - 4 * b^2 + 2 * b + 3 = 0
axiom h3 : c^3 - 4 * c^2 + 2 * c + 3 = 0

axiom h4 : P a = b + c
axiom h5 : P b = a + c
axiom h6 : P c = a + b
axiom h7 : a + b + c = 4
axiom h8 : P 4 = -20

theorem find_P_x :
  P x = (-17 / 3) * x^3 + (68 / 3) * x^2 - (31 / 3) * x - 18 := sorry

end NUMINAMATH_GPT_find_P_x_l1462_146277


namespace NUMINAMATH_GPT_calculate_error_percentage_l1462_146203

theorem calculate_error_percentage (x : ℝ) (hx : x > 0) (x_eq_9 : x = 9) :
  (abs ((x * (x - 8)) / (8 * x)) * 100) = 12.5 := by
  sorry

end NUMINAMATH_GPT_calculate_error_percentage_l1462_146203


namespace NUMINAMATH_GPT_could_be_simple_random_sampling_l1462_146234

-- Conditions
def boys : Nat := 20
def girls : Nat := 30
def total_students : Nat := boys + girls
def sample_size : Nat := 10
def boys_in_sample : Nat := 4
def girls_in_sample : Nat := 6

-- Theorem Statement
theorem could_be_simple_random_sampling :
  boys = 20 ∧ girls = 30 ∧ sample_size = 10 ∧ boys_in_sample = 4 ∧ girls_in_sample = 6 →
  (∃ (sample_method : String), sample_method = "simple random sampling"):=
by 
  sorry

end NUMINAMATH_GPT_could_be_simple_random_sampling_l1462_146234


namespace NUMINAMATH_GPT_ratio_of_areas_l1462_146282

-- Definitions of the perimeters for each region
def perimeter_I : ℕ := 16
def perimeter_II : ℕ := 36
def perimeter_IV : ℕ := 48

-- Define the side lengths based on the given perimeters
def side_length (P : ℕ) : ℕ := P / 4

-- Calculate the areas from the side lengths
def area (s : ℕ) : ℕ := s * s

-- Now we state the theorem
theorem ratio_of_areas : 
  (area (side_length perimeter_II)) / (area (side_length perimeter_IV)) = 9 / 16 := 
by sorry

end NUMINAMATH_GPT_ratio_of_areas_l1462_146282


namespace NUMINAMATH_GPT_value_of_a_pow_sum_l1462_146280

variable {a : ℝ}
variable {m n : ℕ}

theorem value_of_a_pow_sum (h1 : a^m = 5) (h2 : a^n = 3) : a^(m + n) = 15 := by
  sorry

end NUMINAMATH_GPT_value_of_a_pow_sum_l1462_146280


namespace NUMINAMATH_GPT_symmetric_probability_l1462_146295

-- Definitions based on the problem conditions
def total_points : ℕ := 121
def central_point : ℕ × ℕ := (6, 6)
def remaining_points : ℕ := total_points - 1
def symmetric_points : ℕ := 40

-- Predicate for the probability that line PQ is a line of symmetry
def is_symmetrical_line (p q : (ℕ × ℕ)) : Prop := 
  (q.fst = 11 - p.fst ∧ q.snd = p.snd) ∨
  (q.fst = p.fst ∧ q.snd = 11 - p.snd) ∨
  (q.fst + q.snd = 12) ∨ 
  (q.fst - q.snd = 0)

-- The theorem stating the probability is 1/3
theorem symmetric_probability :
  ∃ (total_points : ℕ) (remaining_points : ℕ) (symmetric_points : ℕ),
    total_points = 121 ∧
    remaining_points = total_points - 1 ∧
    symmetric_points = 40 ∧
    (symmetric_points : ℚ) / (remaining_points : ℚ) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_probability_l1462_146295


namespace NUMINAMATH_GPT_consecutive_even_product_l1462_146217

theorem consecutive_even_product (x : ℤ) (h : x * (x + 2) = 224) : x * (x + 2) = 224 := by
  sorry

end NUMINAMATH_GPT_consecutive_even_product_l1462_146217


namespace NUMINAMATH_GPT_simplify_expression_l1462_146276

theorem simplify_expression (m : ℝ) (h : m ≠ 0) : (4 * m^2 - 2 * m) / (2 * m) = 2 * m - 1 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1462_146276


namespace NUMINAMATH_GPT_how_many_unanswered_l1462_146216

theorem how_many_unanswered (c w u : ℕ) (h1 : 25 + 5 * c - 2 * w = 95)
                            (h2 : 6 * c + u = 110) (h3 : c + w + u = 30) : u = 10 :=
by
  sorry

end NUMINAMATH_GPT_how_many_unanswered_l1462_146216


namespace NUMINAMATH_GPT_no_real_number_pairs_satisfy_equation_l1462_146209

theorem no_real_number_pairs_satisfy_equation :
  ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 →
  ¬ (1 / a + 1 / b = 1 / (2 * a + 3 * b)) :=
by
  intros a b ha hb
  sorry

end NUMINAMATH_GPT_no_real_number_pairs_satisfy_equation_l1462_146209


namespace NUMINAMATH_GPT_polynomial_sum_of_squares_l1462_146214

theorem polynomial_sum_of_squares (P : Polynomial ℝ) 
  (hP : ∀ x : ℝ, 0 ≤ P.eval x) : 
  ∃ (f g : Polynomial ℝ), P = f * f + g * g := 
sorry

end NUMINAMATH_GPT_polynomial_sum_of_squares_l1462_146214


namespace NUMINAMATH_GPT_centroid_inverse_square_sum_l1462_146208

theorem centroid_inverse_square_sum
  (α β γ p q r : ℝ)
  (h1 : 1/α^2 + 1/β^2 + 1/γ^2 = 1)
  (hp : p = α / 3)
  (hq : q = β / 3)
  (hr : r = γ / 3) :
  (1/p^2 + 1/q^2 + 1/r^2 = 9) :=
sorry

end NUMINAMATH_GPT_centroid_inverse_square_sum_l1462_146208


namespace NUMINAMATH_GPT_inequality1_inequality2_l1462_146207

theorem inequality1 (x : ℝ) : x ≠ 2 → (x + 1)/(x - 2) ≥ 3 → 2 < x ∧ x ≤ 7/2 :=
sorry

theorem inequality2 (x a : ℝ) : 
  (x^2 - a * x - 2 * a^2 ≤ 0) → 
  (a = 0 → x = 0) ∧ 
  (a > 0 → -a ≤ x ∧ x ≤ 2 * a) ∧ 
  (a < 0 → 2 * a ≤ x ∧ x ≤ -a) :=
sorry

end NUMINAMATH_GPT_inequality1_inequality2_l1462_146207


namespace NUMINAMATH_GPT_Christine_distance_went_l1462_146235

-- Definitions from conditions
def Speed : ℝ := 20 -- miles per hour
def Time : ℝ := 4  -- hours

-- Statement of the problem
def Distance_went : ℝ := Speed * Time

-- The theorem we need to prove
theorem Christine_distance_went : Distance_went = 80 :=
by
  sorry

end NUMINAMATH_GPT_Christine_distance_went_l1462_146235


namespace NUMINAMATH_GPT_mark_weekly_reading_time_l1462_146244

-- Define the conditions
def hours_per_day : ℕ := 2
def days_per_week : ℕ := 7
def additional_hours : ℕ := 4

-- State the main theorem to prove
theorem mark_weekly_reading_time : (hours_per_day * days_per_week) + additional_hours = 18 := 
by
  -- The proof steps are omitted as per instructions
  sorry

end NUMINAMATH_GPT_mark_weekly_reading_time_l1462_146244


namespace NUMINAMATH_GPT_bn_six_eight_product_l1462_146227

noncomputable def sequence_an (n : ℕ) : ℝ := sorry  -- given that an is an arithmetic sequence and an ≠ 0
noncomputable def sequence_bn (n : ℕ) : ℝ := sorry  -- given that bn is a geometric sequence

theorem bn_six_eight_product :
  (∀ n : ℕ, sequence_an n ≠ 0) →
  2 * sequence_an 3 - sequence_an 7 ^ 2 + 2 * sequence_an 11 = 0 →
  sequence_bn 7 = sequence_an 7 →
  sequence_bn 6 * sequence_bn 8 = 16 :=
sorry

end NUMINAMATH_GPT_bn_six_eight_product_l1462_146227


namespace NUMINAMATH_GPT_polynomial_root_abs_sum_eq_80_l1462_146269

theorem polynomial_root_abs_sum_eq_80 (a b c : ℤ) (m : ℤ) 
  (h1 : a + b + c = 0) 
  (h2 : ab + bc + ac = -2023) 
  (h3 : ∃ m, ∀ x : ℤ, x^3 - 2023 * x + m = (x - a) * (x - b) * (x - c)) : 
  |a| + |b| + |c| = 80 := 
by {
  sorry
}

end NUMINAMATH_GPT_polynomial_root_abs_sum_eq_80_l1462_146269


namespace NUMINAMATH_GPT_triangle_largest_angle_l1462_146255

theorem triangle_largest_angle (k : ℕ) 
  (h1 : 3 * k + 4 * k + 5 * k = 180)
  (h2 : ∃ k, 3 * k + 4 * k + 5 * k = 180) :
  5 * k = 75 :=
sorry

end NUMINAMATH_GPT_triangle_largest_angle_l1462_146255


namespace NUMINAMATH_GPT_jane_wins_game_l1462_146247

noncomputable def jane_win_probability : ℚ :=
  1/3 / (1 - (2/3 * 1/3 * 2/3))

theorem jane_wins_game :
  jane_win_probability = 9/23 :=
by
  -- detailed proof steps would be filled in here
  sorry

end NUMINAMATH_GPT_jane_wins_game_l1462_146247


namespace NUMINAMATH_GPT_axis_of_symmetry_l1462_146225

-- Define the given parabolic function
def parabola (x : ℝ) : ℝ := (2 - x) * x

-- Define the axis of symmetry property for the given parabola
theorem axis_of_symmetry : ∀ x : ℝ, ((2 - x) * x) = -((x - 1)^2) + 1 → (∃ x_sym : ℝ, x_sym = 1) :=
by
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_l1462_146225


namespace NUMINAMATH_GPT_given_problem_l1462_146283

theorem given_problem :
  3^3 + 4^3 + 5^3 = 6^3 :=
by sorry

end NUMINAMATH_GPT_given_problem_l1462_146283


namespace NUMINAMATH_GPT_distance_from_home_to_school_l1462_146259

variable (t : ℕ) (D : ℕ)

-- conditions
def condition1 := 60 * (t - 10) = D
def condition2 := 50 * (t + 4) = D

-- the mathematical equivalent proof problem: proving the distance is 4200 given conditions
theorem distance_from_home_to_school :
  (∃ t, condition1 t 4200 ∧ condition2 t 4200) :=
  sorry

end NUMINAMATH_GPT_distance_from_home_to_school_l1462_146259


namespace NUMINAMATH_GPT_dealership_sales_l1462_146279

theorem dealership_sales (sports_cars sedans suvs : ℕ) (h_sc : sports_cars = 35)
  (h_ratio_sedans : 5 * sedans = 8 * sports_cars) 
  (h_ratio_suvs : 5 * suvs = 3 * sports_cars) : 
  sedans = 56 ∧ suvs = 21 := by
  sorry

#print dealership_sales

end NUMINAMATH_GPT_dealership_sales_l1462_146279


namespace NUMINAMATH_GPT_first_year_with_sum_of_digits_10_after_2200_l1462_146291

/-- Prove that the first year after 2200 in which the sum of the digits equals 10 is 2224. -/
theorem first_year_with_sum_of_digits_10_after_2200 :
  ∃ y, y > 2200 ∧ (List.sum (y.digits 10) = 10) ∧ 
       ∀ z, (2200 < z ∧ z < y) → (List.sum (z.digits 10) ≠ 10) :=
sorry

end NUMINAMATH_GPT_first_year_with_sum_of_digits_10_after_2200_l1462_146291


namespace NUMINAMATH_GPT_john_bought_three_sodas_l1462_146224

-- Define the conditions

def cost_per_soda := 2
def total_money_paid := 20
def change_received := 14

-- Definition indicating the number of sodas bought
def num_sodas_bought := (total_money_paid - change_received) / cost_per_soda

-- Question: Prove that John bought 3 sodas given these conditions
theorem john_bought_three_sodas : num_sodas_bought = 3 := by
  -- Proof: This is an example of how you may structure the proof
  sorry

end NUMINAMATH_GPT_john_bought_three_sodas_l1462_146224


namespace NUMINAMATH_GPT_xy_value_l1462_146257

theorem xy_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + 3 / x = y + 3 / y) (hxy : x ≠ y) : x * y = 3 :=
sorry

end NUMINAMATH_GPT_xy_value_l1462_146257


namespace NUMINAMATH_GPT_popsicle_sticks_left_l1462_146242

/-- Danielle has $10 for supplies. She buys one set of molds for $3, 
a pack of 100 popsicle sticks for $1. Each bottle of juice makes 20 popsicles and costs $2.
Prove that the number of popsicle sticks Danielle will be left with after making as many popsicles as she can is 40. -/
theorem popsicle_sticks_left (initial_money : ℕ)
    (mold_cost : ℕ) (sticks_cost : ℕ) (initial_sticks : ℕ)
    (juice_cost : ℕ) (popsicles_per_bottle : ℕ)
    (final_sticks : ℕ) :
    initial_money = 10 →
    mold_cost = 3 → 
    sticks_cost = 1 → 
    initial_sticks = 100 →
    juice_cost = 2 →
    popsicles_per_bottle = 20 →
    final_sticks = initial_sticks - (popsicles_per_bottle * (initial_money - mold_cost - sticks_cost) / juice_cost) →
    final_sticks = 40 :=
by
  intros h_initial_money h_mold_cost h_sticks_cost h_initial_sticks h_juice_cost h_popsicles_per_bottle h_final_sticks
  rw [h_initial_money, h_mold_cost, h_sticks_cost, h_initial_sticks, h_juice_cost, h_popsicles_per_bottle] at h_final_sticks
  norm_num at h_final_sticks
  exact h_final_sticks

end NUMINAMATH_GPT_popsicle_sticks_left_l1462_146242


namespace NUMINAMATH_GPT_one_divides_the_other_l1462_146293

theorem one_divides_the_other (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
(h : Nat.lcm (x+2) (y+2) - Nat.lcm (x+1) (y+1) = Nat.lcm (x+1) (y+1) - Nat.lcm x y) :
  ∃ m n : ℕ, (x = m * y) ∨ (y = n * x) :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_one_divides_the_other_l1462_146293


namespace NUMINAMATH_GPT_units_digit_of_n_cubed_minus_n_squared_l1462_146287

-- Define n for the purpose of the problem
def n : ℕ := 9867

-- Prove that the units digit of n^3 - n^2 is 4
theorem units_digit_of_n_cubed_minus_n_squared : ∃ d : ℕ, d = (n^3 - n^2) % 10 ∧ d = 4 := by
  sorry

end NUMINAMATH_GPT_units_digit_of_n_cubed_minus_n_squared_l1462_146287


namespace NUMINAMATH_GPT_quadratic_real_roots_condition_l1462_146233

theorem quadratic_real_roots_condition (a b c : ℝ) (q : b^2 - 4 * a * c ≥ 0) (h : a ≠ 0) : 
  (b^2 - 4 * a * c ≥ 0 ∧ a ≠ 0) ↔ ((∃ x1 x2 : ℝ, a * x1 ^ 2 + b * x1 + c = 0 ∧ a * x2 ^ 2 + b * x2 + c = 0) ∨ (∃ x : ℝ, a * x ^ 2 + b * x + c = 0)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_condition_l1462_146233


namespace NUMINAMATH_GPT_faster_runner_l1462_146252

-- Define the speeds of A and B
variables (v_A v_B : ℝ)
-- A's speed as a multiple of B's speed
variables (k : ℝ)

-- A's and B's distances in the race
variables (d_A d_B : ℝ)
-- Distance of the race
variables (distance : ℝ)
-- Head start given to B
variables (head_start : ℝ)

-- The theorem to prove that the factor k is 4 given the conditions
theorem faster_runner (k : ℝ) (v_A v_B : ℝ) (d_A d_B distance head_start : ℝ) :
  v_A = k * v_B ∧ d_B = distance - head_start ∧ d_A = distance ∧ (d_A / v_A) = (d_B / v_B) → k = 4 :=
by
  sorry

end NUMINAMATH_GPT_faster_runner_l1462_146252


namespace NUMINAMATH_GPT_molecular_weight_of_one_mole_l1462_146221

theorem molecular_weight_of_one_mole (total_weight : ℝ) (number_of_moles : ℕ) 
    (h : total_weight = 204) (n : number_of_moles = 3) : 
    (total_weight / number_of_moles) = 68 :=
by
  have h_weight : total_weight = 204 := h
  have h_moles : number_of_moles = 3 := n
  rw [h_weight, h_moles]
  norm_num

end NUMINAMATH_GPT_molecular_weight_of_one_mole_l1462_146221


namespace NUMINAMATH_GPT__l1462_146230

noncomputable def charlesPictures : Prop :=
  ∀ (bought : ℕ) (drew_today : ℕ) (drew_yesterday_after_work : ℕ) (left : ℕ),
    (bought = 20) →
    (drew_today = 6) →
    (drew_yesterday_after_work = 6) →
    (left = 2) →
    (bought - left - drew_today - drew_yesterday_after_work = 6)

-- We can use this statement "charlesPictures" to represent the theorem to be proved in Lean 4.

end NUMINAMATH_GPT__l1462_146230


namespace NUMINAMATH_GPT_domain_of_f_l1462_146265

noncomputable def f (x : ℝ) : ℝ := 1 / ⌊x^2 - 6 * x + 10⌋

theorem domain_of_f : {x : ℝ | ∀ y, f y ≠ 0 → x ≠ 3} = {x : ℝ | x < 3 ∨ x > 3} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1462_146265


namespace NUMINAMATH_GPT_david_produces_8_more_widgets_l1462_146205

variable (w t : ℝ)

def widgets_monday (w t : ℝ) : ℝ :=
  w * t

def widgets_tuesday (w t : ℝ) : ℝ :=
  (w + 4) * (t - 2)

theorem david_produces_8_more_widgets (h : w = 2 * t) : 
  widgets_monday w t - widgets_tuesday w t = 8 :=
by
  sorry

end NUMINAMATH_GPT_david_produces_8_more_widgets_l1462_146205


namespace NUMINAMATH_GPT_problem1_problem2_l1462_146251

-- Define sets A and B
def A (a b : ℝ) : Set ℝ := { x | a - b < x ∧ x < a + b }
def B : Set ℝ := { x | x < -1 ∨ x > 5 }

-- First problem: prove the range of a
theorem problem1 (a : ℝ) (h : A a 1 ⊆ B) : a ≤ -2 ∨ a ≥ 6 := by
  sorry

-- Second problem: prove the range of b
theorem problem2 (b : ℝ) (h : A 1 b ∩ B = ∅) : b ≤ 2 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1462_146251


namespace NUMINAMATH_GPT_three_pow_m_plus_2n_l1462_146271

theorem three_pow_m_plus_2n (m n : ℕ) (h1 : 3^m = 5) (h2 : 9^n = 10) : 3^(m + 2 * n) = 50 :=
by
  sorry

end NUMINAMATH_GPT_three_pow_m_plus_2n_l1462_146271


namespace NUMINAMATH_GPT_grasshoppers_cannot_return_to_initial_positions_l1462_146243

theorem grasshoppers_cannot_return_to_initial_positions :
  (∀ (a b c : ℕ), a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 → a + b + c ≠ 1985) :=
by
  sorry

end NUMINAMATH_GPT_grasshoppers_cannot_return_to_initial_positions_l1462_146243


namespace NUMINAMATH_GPT_students_in_both_math_and_chem_l1462_146229

theorem students_in_both_math_and_chem (students total math physics chem math_physics physics_chem : ℕ) :
  total = 36 →
  students ≤ 2 →
  math = 26 →
  physics = 15 →
  chem = 13 →
  math_physics = 6 →
  physics_chem = 4 →
  math + physics + chem - math_physics - physics_chem - students = total →
  students = 8 := by
  intros h_total h_students h_math h_physics h_chem h_math_physics h_physics_chem h_equation
  sorry

end NUMINAMATH_GPT_students_in_both_math_and_chem_l1462_146229


namespace NUMINAMATH_GPT_marian_baked_cookies_l1462_146254

theorem marian_baked_cookies :
  let cookies_per_tray := 12
  let trays_used := 23
  trays_used * cookies_per_tray = 276 :=
by
  sorry

end NUMINAMATH_GPT_marian_baked_cookies_l1462_146254


namespace NUMINAMATH_GPT_mike_taller_than_mark_l1462_146246

def feet_to_inches (feet : ℕ) : ℕ := 12 * feet

def mark_height_feet := 5
def mark_height_inches := 3
def mike_height_feet := 6
def mike_height_inches := 1

def mark_total_height := feet_to_inches mark_height_feet + mark_height_inches
def mike_total_height := feet_to_inches mike_height_feet + mike_height_inches

theorem mike_taller_than_mark : mike_total_height - mark_total_height = 10 :=
by
  sorry

end NUMINAMATH_GPT_mike_taller_than_mark_l1462_146246


namespace NUMINAMATH_GPT_evaluate_expression_l1462_146200

theorem evaluate_expression : (-3)^7 / 3^5 + 2^5 - 7^2 = -26 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1462_146200


namespace NUMINAMATH_GPT_gcd_polynomial_eq_one_l1462_146253

theorem gcd_polynomial_eq_one (b : ℤ) (hb : Even b) (hmb : 431 ∣ b) : 
  Int.gcd (8 * b^2 + 63 * b + 143) (4 * b + 17) = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_polynomial_eq_one_l1462_146253


namespace NUMINAMATH_GPT_Ravi_probability_l1462_146237

-- Conditions from the problem
def P_Ram : ℚ := 4 / 7
def P_BothSelected : ℚ := 0.11428571428571428

-- Statement to prove
theorem Ravi_probability :
  ∃ P_Ravi : ℚ, P_Rami = 0.2 ∧ P_Ram * P_Ravi = P_BothSelected := by
  sorry

end NUMINAMATH_GPT_Ravi_probability_l1462_146237


namespace NUMINAMATH_GPT_min_value_of_sum_of_squares_l1462_146267

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : x^2 + y^2 + z^2 ≥ 4.8 :=
sorry

end NUMINAMATH_GPT_min_value_of_sum_of_squares_l1462_146267


namespace NUMINAMATH_GPT_total_pieces_of_junk_mail_l1462_146245

-- Definition of the problem based on given conditions
def pieces_per_house : ℕ := 4
def number_of_blocks : ℕ := 16
def houses_per_block : ℕ := 17

-- Statement of the theorem to prove the total number of pieces of junk mail
theorem total_pieces_of_junk_mail :
  (houses_per_block * pieces_per_house * number_of_blocks) = 1088 :=
by
  sorry

end NUMINAMATH_GPT_total_pieces_of_junk_mail_l1462_146245


namespace NUMINAMATH_GPT_number_of_eggs_left_l1462_146240

theorem number_of_eggs_left (initial_eggs : ℕ) (eggs_eaten_morning : ℕ) (eggs_eaten_afternoon : ℕ) (eggs_left : ℕ) :
    initial_eggs = 20 → eggs_eaten_morning = 4 → eggs_eaten_afternoon = 3 → eggs_left = initial_eggs - (eggs_eaten_morning + eggs_eaten_afternoon) → eggs_left = 13 :=
by
  intros h_initial h_morning h_afternoon h_calc
  rw [h_initial, h_morning, h_afternoon] at h_calc
  norm_num at h_calc
  exact h_calc

end NUMINAMATH_GPT_number_of_eggs_left_l1462_146240


namespace NUMINAMATH_GPT_bicycles_wheels_l1462_146274

theorem bicycles_wheels (b : ℕ) (h1 : 3 * b + 4 * 3 + 7 * 1 = 25) : b = 2 :=
sorry

end NUMINAMATH_GPT_bicycles_wheels_l1462_146274


namespace NUMINAMATH_GPT_triangle_min_area_l1462_146286

theorem triangle_min_area :
  ∃ (p q : ℤ), (p, q).fst = 3 ∧ (p, q).snd = 3 ∧ 1/2 * |18 * p - 30 * q| = 3 := 
sorry

end NUMINAMATH_GPT_triangle_min_area_l1462_146286


namespace NUMINAMATH_GPT_total_sample_size_is_72_l1462_146228

-- Definitions based on the given conditions:
def production_A : ℕ := 600
def production_B : ℕ := 1200
def production_C : ℕ := 1800
def total_production : ℕ := production_A + production_B + production_C
def sampled_B : ℕ := 2

-- Main theorem to prove the sample size:
theorem total_sample_size_is_72 : 
  ∃ (n : ℕ), 
    (∃ s_A s_B s_C, 
      s_A = (production_A * sampled_B * total_production) / production_B^2 ∧ 
      s_B = sampled_B ∧ 
      s_C = (production_C * sampled_B * total_production) / production_B^2 ∧
      n = s_A + s_B + s_C) ∧ 
  (n = 72) :=
sorry

end NUMINAMATH_GPT_total_sample_size_is_72_l1462_146228


namespace NUMINAMATH_GPT_man_to_son_age_ratio_l1462_146281

-- Definitions based on conditions
variable (son_age : ℕ) (man_age : ℕ)
variable (h1 : man_age = son_age + 18) -- The man is 18 years older than his son
variable (h2 : 2 * (son_age + 2) = man_age + 2) -- In two years, the man's age will be a multiple of the son's age
variable (h3 : son_age = 16) -- The present age of the son is 16

-- Theorem statement to prove the desired ratio
theorem man_to_son_age_ratio (son_age man_age : ℕ) (h1 : man_age = son_age + 18) (h2 : 2 * (son_age + 2) = man_age + 2) (h3 : son_age = 16) :
  (man_age + 2) / (son_age + 2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_man_to_son_age_ratio_l1462_146281


namespace NUMINAMATH_GPT_weaving_increase_is_sixteen_over_twentynine_l1462_146236

-- Conditions for the problem as definitions
def first_day_weaving := 5
def total_days := 30
def total_weaving := 390

-- The arithmetic series sum formula for 30 days
def sum_arithmetic_series (a d : ℚ) (n : ℕ) := n * a + (n * (n-1) / 2) * d

-- The question is to prove the increase in chi per day is 16/29
theorem weaving_increase_is_sixteen_over_twentynine
  (d : ℚ)
  (h : sum_arithmetic_series first_day_weaving d total_days = total_weaving) :
  d = 16 / 29 :=
sorry

end NUMINAMATH_GPT_weaving_increase_is_sixteen_over_twentynine_l1462_146236
