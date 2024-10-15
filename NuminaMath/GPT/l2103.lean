import Mathlib

namespace NUMINAMATH_GPT_eldora_boxes_paper_clips_l2103_210320

theorem eldora_boxes_paper_clips (x y : ℝ)
  (h1 : 1.85 * x + 7 * y = 55.40)
  (h2 : 1.85 * 12 + 10 * y = 61.70)
  (h3 : 1.85 = 1.85) : -- Given && Asserting the constant price of one box

  x = 15 :=
by
  sorry

end NUMINAMATH_GPT_eldora_boxes_paper_clips_l2103_210320


namespace NUMINAMATH_GPT_carpet_rate_proof_l2103_210326

noncomputable def carpet_rate (breadth_first : ℝ) (length_ratio : ℝ) (cost_second : ℝ) : ℝ :=
  let length_first := length_ratio * breadth_first
  let area_first := length_first * breadth_first
  let length_second := length_first * 1.4
  let breadth_second := breadth_first * 1.25
  let area_second := length_second * breadth_second 
  cost_second / area_second

theorem carpet_rate_proof : carpet_rate 6 1.44 4082.4 = 45 :=
by
  -- Here we provide the goal and state what needs to be proven.
  sorry

end NUMINAMATH_GPT_carpet_rate_proof_l2103_210326


namespace NUMINAMATH_GPT_simplify_expression_calculate_expression_l2103_210331

-- Problem 1
theorem simplify_expression (x : ℝ) : 
  (x + 1) * (x + 1) - x * (x + 1) = x + 1 := by
  sorry

-- Problem 2
theorem calculate_expression : 
  (-1 : ℝ) ^ 2023 + 2 ^ (-2 : ℝ) + 4 * (Real.cos (Real.pi / 6))^2 = 9 / 4 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_calculate_expression_l2103_210331


namespace NUMINAMATH_GPT_find_value_of_4_minus_2a_l2103_210354

theorem find_value_of_4_minus_2a (a b : ℚ) (h1 : 4 + 2 * a = 5 - b) (h2 : 5 + b = 9 + 3 * a) : 4 - 2 * a = 26 / 5 := 
by
  sorry

end NUMINAMATH_GPT_find_value_of_4_minus_2a_l2103_210354


namespace NUMINAMATH_GPT_num_ways_books_distribution_l2103_210382

-- Given conditions
def num_copies_type1 : ℕ := 8
def num_copies_type2 : ℕ := 4
def min_books_in_library_type1 : ℕ := 1
def max_books_in_library_type1 : ℕ := 7
def min_books_in_library_type2 : ℕ := 1
def max_books_in_library_type2 : ℕ := 3

-- The proof problem statement
theorem num_ways_books_distribution : 
  (max_books_in_library_type1 - min_books_in_library_type1 + 1) * 
  (max_books_in_library_type2 - min_books_in_library_type2 + 1) = 21 := by
    sorry

end NUMINAMATH_GPT_num_ways_books_distribution_l2103_210382


namespace NUMINAMATH_GPT_problem_statement_l2103_210345

theorem problem_statement (A B : ℝ) (hA : A = 10 * π / 180) (hB : B = 35 * π / 180) :
  (1 + Real.tan A) * (1 + Real.sin B) = 
  1 + Real.tan A + (Real.sqrt 2 / 2) * (Real.cos A - Real.sin A) + Real.tan A * (Real.sqrt 2 / 2) * (Real.cos A - Real.sin A) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2103_210345


namespace NUMINAMATH_GPT_sum_of_dimensions_l2103_210384

noncomputable def rectangular_prism_dimensions (A B C : ℝ) : Prop :=
  (A * B = 30) ∧ (A * C = 40) ∧ (B * C = 60)

theorem sum_of_dimensions (A B C : ℝ) (h : rectangular_prism_dimensions A B C) : A + B + C = 9 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_dimensions_l2103_210384


namespace NUMINAMATH_GPT_scarves_per_box_l2103_210309

theorem scarves_per_box (S : ℕ) 
  (boxes : ℕ := 8) 
  (mittens_per_box : ℕ := 6) 
  (total_clothing : ℕ := 80) 
  (total_mittens : ℕ := boxes * mittens_per_box) 
  (total_scarves : ℕ := total_clothing - total_mittens) 
  (scarves_per_box : ℕ := total_scarves / boxes) 
  : scarves_per_box = 4 := 
by 
  sorry

end NUMINAMATH_GPT_scarves_per_box_l2103_210309


namespace NUMINAMATH_GPT_seeds_germinated_percentage_l2103_210306

theorem seeds_germinated_percentage (n1 n2 : ℕ) (p1 p2 : ℝ) (h1 : n1 = 300) (h2 : n2 = 200) (h3 : p1 = 0.25) (h4 : p2 = 0.30) :
  ( (n1 * p1 + n2 * p2) / (n1 + n2) ) * 100 = 27 :=
by
  sorry

end NUMINAMATH_GPT_seeds_germinated_percentage_l2103_210306


namespace NUMINAMATH_GPT_total_fish_l2103_210318

-- Defining the number of fish each person has, based on the conditions.
def billy_fish : ℕ := 10
def tony_fish : ℕ := 3 * billy_fish
def sarah_fish : ℕ := tony_fish + 5
def bobby_fish : ℕ := 2 * sarah_fish

-- The theorem stating the total number of fish.
theorem total_fish : billy_fish + tony_fish + sarah_fish + bobby_fish = 145 := by
  sorry

end NUMINAMATH_GPT_total_fish_l2103_210318


namespace NUMINAMATH_GPT_estimate_students_in_range_l2103_210347

noncomputable def n_students := 3000
noncomputable def score_range_low := 70
noncomputable def score_range_high := 80
noncomputable def est_students_in_range := 408

theorem estimate_students_in_range : ∀ (n : ℕ) (k : ℕ), n = n_students →
  k = est_students_in_range →
  normal_distribution :=
sorry

end NUMINAMATH_GPT_estimate_students_in_range_l2103_210347


namespace NUMINAMATH_GPT_students_taking_art_l2103_210392

theorem students_taking_art :
  ∀ (total_students music_students both_students neither_students : ℕ),
  total_students = 500 →
  music_students = 30 →
  both_students = 10 →
  neither_students = 460 →
  music_students + both_students + neither_students = total_students →
  ((total_students - neither_students) - (music_students - both_students) + both_students = 20) :=
by
  intros total_students music_students both_students neither_students 
  intro h_total h_music h_both h_neither h_sum 
  sorry

end NUMINAMATH_GPT_students_taking_art_l2103_210392


namespace NUMINAMATH_GPT_polynomial_multiplication_l2103_210348

theorem polynomial_multiplication (x a : ℝ) : (x - a) * (x^2 + a * x + a^2) = x^3 - a^3 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_multiplication_l2103_210348


namespace NUMINAMATH_GPT_additional_laps_needed_l2103_210335

-- Definitions of problem conditions
def total_required_distance : ℕ := 2400
def lap_length : ℕ := 150
def madison_laps : ℕ := 6
def gigi_laps : ℕ := 6

-- Target statement to prove the number of additional laps needed
theorem additional_laps_needed : (total_required_distance - (madison_laps + gigi_laps) * lap_length) / lap_length = 4 := by
  sorry

end NUMINAMATH_GPT_additional_laps_needed_l2103_210335


namespace NUMINAMATH_GPT_max_value_of_g_l2103_210372

noncomputable def f1 (x : ℝ) : ℝ := 3 * x + 3
noncomputable def f2 (x : ℝ) : ℝ := (1/3) * x + 2
noncomputable def f3 (x : ℝ) : ℝ := -x + 8

noncomputable def g (x : ℝ) : ℝ := min (min (f1 x) (f2 x)) (f3 x)

theorem max_value_of_g : ∃ x : ℝ, g x = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_g_l2103_210372


namespace NUMINAMATH_GPT_triangle_area_is_14_l2103_210346

def vector : Type := (ℝ × ℝ)
def a : vector := (4, -1)
def b : vector := (2 * 2, 2 * 3)

noncomputable def parallelogram_area (u v : vector) : ℝ :=
  let (ux, uy) := u
  let (vx, vy) := v
  abs (ux * vy - uy * vx)

noncomputable def triangle_area (u v : vector) : ℝ :=
  (parallelogram_area u v) / 2

theorem triangle_area_is_14 : triangle_area a b = 14 :=
by
  unfold a b triangle_area parallelogram_area
  sorry

end NUMINAMATH_GPT_triangle_area_is_14_l2103_210346


namespace NUMINAMATH_GPT_joan_games_last_year_l2103_210361

theorem joan_games_last_year (games_this_year : ℕ) (total_games : ℕ) (games_last_year : ℕ) 
  (h1 : games_this_year = 4) 
  (h2 : total_games = 9) 
  (h3 : total_games = games_this_year + games_last_year) : 
  games_last_year = 5 := 
by
  sorry

end NUMINAMATH_GPT_joan_games_last_year_l2103_210361


namespace NUMINAMATH_GPT_tan_half_angle_l2103_210308

theorem tan_half_angle (p q : ℝ) (h_cos : Real.cos p + Real.cos q = 3 / 5) (h_sin : Real.sin p + Real.sin q = 1 / 5) : Real.tan ((p + q) / 2) = 1 / 3 :=
sorry

end NUMINAMATH_GPT_tan_half_angle_l2103_210308


namespace NUMINAMATH_GPT_future_ages_equation_l2103_210378

-- Defining the ages of Joe and James with given conditions
def joe_current_age : ℕ := 22
def james_current_age : ℕ := 12

-- Defining the condition that Joe is 10 years older than James
lemma joe_older_than_james : joe_current_age = james_current_age + 10 := by
  unfold joe_current_age james_current_age
  simp

-- Defining the future age condition equation and the target years y.
theorem future_ages_equation (y : ℕ) :
  2 * (joe_current_age + y) = 3 * (james_current_age + y) → y = 8 := by
  unfold joe_current_age james_current_age
  intro h
  linarith

end NUMINAMATH_GPT_future_ages_equation_l2103_210378


namespace NUMINAMATH_GPT_evaluate_expression_l2103_210325

theorem evaluate_expression : 12 * ((1/3 : ℚ) + (1/4) + (1/6))⁻¹ = 16 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2103_210325


namespace NUMINAMATH_GPT_ticket_ratio_proof_l2103_210315

-- Define the initial number of tickets Tate has.
def initial_tate_tickets : ℕ := 32

-- Define the additional tickets Tate buys.
def additional_tickets : ℕ := 2

-- Define the total tickets they have together.
def combined_tickets : ℕ := 51

-- Calculate Tate's total number of tickets after buying more tickets.
def total_tate_tickets := initial_tate_tickets + additional_tickets

-- Define the number of tickets Peyton has.
def peyton_tickets := combined_tickets - total_tate_tickets

-- Define the ratio of Peyton's tickets to Tate's tickets.
def tickets_ratio := peyton_tickets / total_tate_tickets

theorem ticket_ratio_proof : tickets_ratio = 1 / 2 :=
by
  unfold tickets_ratio peyton_tickets total_tate_tickets initial_tate_tickets additional_tickets
  norm_num
  sorry

end NUMINAMATH_GPT_ticket_ratio_proof_l2103_210315


namespace NUMINAMATH_GPT_six_digit_phone_number_count_l2103_210383

def six_digit_to_seven_digit_count (six_digit : ℕ) (h : 100000 ≤ six_digit ∧ six_digit < 1000000) : ℕ :=
  let num_positions := 7
  let num_digits := 10
  num_positions * num_digits

theorem six_digit_phone_number_count (six_digit : ℕ) (h : 100000 ≤ six_digit ∧ six_digit < 1000000) :
  six_digit_to_seven_digit_count six_digit h = 70 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_six_digit_phone_number_count_l2103_210383


namespace NUMINAMATH_GPT_g_of_5_l2103_210338

theorem g_of_5 (g : ℝ → ℝ) (h : ∀ x ≠ 0, 4 * g x - 3 * g (1 / x) = 2 * x) :
  g 5 = 402 / 70 := 
sorry

end NUMINAMATH_GPT_g_of_5_l2103_210338


namespace NUMINAMATH_GPT_tangent_line_equation_l2103_210381

theorem tangent_line_equation 
  (A : ℝ × ℝ)
  (hA : A = (-1, 2))
  (parabola : ℝ → ℝ)
  (h_parabola : ∀ x, parabola x = 2 * x ^ 2) 
  (tangent : ℝ × ℝ → ℝ)
  (h_tangent : ∀ P, tangent P = -4 * P.1 + 4 * (-1) + 2) : 
  tangent A = 4 * (-1) + 2 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_equation_l2103_210381


namespace NUMINAMATH_GPT_factor_complete_polynomial_l2103_210368

theorem factor_complete_polynomial :
  5 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 =
  (5 * x^2 + 94 * x + 385) * (x^2 - 20 * x + 77) :=
sorry

end NUMINAMATH_GPT_factor_complete_polynomial_l2103_210368


namespace NUMINAMATH_GPT_calculate_seedlings_l2103_210314

-- Define conditions
def condition_1 (x n : ℕ) : Prop :=
  x = 5 * n + 6

def condition_2 (x m : ℕ) : Prop :=
  x = 6 * m - 9

-- Define the main theorem based on these conditions
theorem calculate_seedlings (x : ℕ) : (∃ n, condition_1 x n) ∧ (∃ m, condition_2 x m) → x = 81 :=
by {
  sorry
}

end NUMINAMATH_GPT_calculate_seedlings_l2103_210314


namespace NUMINAMATH_GPT_simplify_fraction_l2103_210328

theorem simplify_fraction : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1/2 :=
by sorry

end NUMINAMATH_GPT_simplify_fraction_l2103_210328


namespace NUMINAMATH_GPT_final_bicycle_price_l2103_210319

-- Define conditions 
def original_price : ℝ := 200
def first_discount : ℝ := 0.20
def second_discount : ℝ := 0.25
def price_after_first_discount := original_price * (1 - first_discount)
def final_price := price_after_first_discount * (1 - second_discount)

-- Define the Lean statement to be proven
theorem final_bicycle_price :
  final_price = 120 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_final_bicycle_price_l2103_210319


namespace NUMINAMATH_GPT_positivity_of_fraction_l2103_210373

theorem positivity_of_fraction
  (a b c d x1 x2 x3 x4 : ℝ)
  (h_neg_a : a < 0)
  (h_neg_b : b < 0)
  (h_neg_c : c < 0)
  (h_neg_d : d < 0)
  (h_abs : |x1 - a| + |x2 + b| + |x3 - c| + |x4 + d| = 0) :
  (x1 * x2 / (x3 * x4) > 0) := by
  sorry

end NUMINAMATH_GPT_positivity_of_fraction_l2103_210373


namespace NUMINAMATH_GPT_equilateral_triangle_l2103_210323

namespace TriangleEquilateral

-- Define the structure of a triangle and given conditions
structure Triangle :=
  (A B C : ℝ)  -- vertices
  (angleA : ℝ) -- angle at vertex A
  (sideBC : ℝ) -- length of side BC
  (perimeter : ℝ)  -- perimeter of the triangle

-- Define the proof problem
theorem equilateral_triangle (T : Triangle) (h1 : T.angleA = 60)
  (h2 : T.sideBC = T.perimeter / 3) : 
  T.A = T.B ∧ T.B = T.C ∧ T.A = T.C ∧ T.A = T.B ∧ T.B = T.C ∧ T.A = T.C :=
  sorry

end TriangleEquilateral

end NUMINAMATH_GPT_equilateral_triangle_l2103_210323


namespace NUMINAMATH_GPT_girl_buys_roses_l2103_210358

theorem girl_buys_roses 
  (x y : ℤ)
  (h1 : y = 1)
  (h2 : x > 0)
  (h3 : (200 : ℤ) / (x + 10) < (100 : ℤ) / x)
  (h4 : (80 : ℤ) / 12 = ((100 : ℤ) / x) - ((200 : ℤ) / (x + 10))) :
  x = 5 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_girl_buys_roses_l2103_210358


namespace NUMINAMATH_GPT_parameter_a_solution_exists_l2103_210352

theorem parameter_a_solution_exists (a : ℝ) : 
  (a < -2 / 3 ∨ a > 0) → ∃ b x y : ℝ, 
  x = 6 / a - abs (y - a) ∧ x^2 + y^2 + b^2 + 63 = 2 * (b * y - 8 * x) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_parameter_a_solution_exists_l2103_210352


namespace NUMINAMATH_GPT_kim_probability_same_color_l2103_210353

noncomputable def probability_same_color (total_shoes : ℕ) (pairs_of_shoes : ℕ) : ℚ :=
  let total_selections := (total_shoes * (total_shoes - 1)) / 2
  let successful_selections := pairs_of_shoes
  successful_selections / total_selections

theorem kim_probability_same_color :
  probability_same_color 10 5 = 1 / 9 :=
by
  unfold probability_same_color
  have h_total : (10 * 9) / 2 = 45 := by norm_num
  have h_success : 5 = 5 := by norm_num
  rw [h_total, h_success]
  norm_num
  done

end NUMINAMATH_GPT_kim_probability_same_color_l2103_210353


namespace NUMINAMATH_GPT_jia_jia_clover_count_l2103_210371

theorem jia_jia_clover_count : ∃ x : ℕ, 3 * x + 4 = 100 ∧ x = 32 := by
  sorry

end NUMINAMATH_GPT_jia_jia_clover_count_l2103_210371


namespace NUMINAMATH_GPT_slices_of_pizza_left_l2103_210337

theorem slices_of_pizza_left (initial_slices: ℕ) 
  (breakfast_slices: ℕ) (lunch_slices: ℕ) (snack_slices: ℕ) (dinner_slices: ℕ) :
  initial_slices = 15 →
  breakfast_slices = 4 →
  lunch_slices = 2 →
  snack_slices = 2 →
  dinner_slices = 5 →
  (initial_slices - breakfast_slices - lunch_slices - snack_slices - dinner_slices) = 2 :=
by
  intros
  repeat { sorry }

end NUMINAMATH_GPT_slices_of_pizza_left_l2103_210337


namespace NUMINAMATH_GPT_fraction_B_A_C_l2103_210334

theorem fraction_B_A_C (A B C : ℕ) (x : ℚ) 
  (h1 : A = (1 / 3) * (B + C)) 
  (h2 : A = B + 10) 
  (h3 : A + B + C = 360) : 
  x = 2 / 7 ∧ B = x * (A + C) :=
by
  sorry -- The proof steps can be filled in

end NUMINAMATH_GPT_fraction_B_A_C_l2103_210334


namespace NUMINAMATH_GPT_net_pay_rate_per_hour_l2103_210355

-- Defining the given conditions
def travel_hours : ℕ := 3
def speed_mph : ℕ := 50
def fuel_efficiency : ℕ := 25 -- miles per gallon
def pay_rate_per_mile : ℚ := 0.60 -- dollars per mile
def gas_cost_per_gallon : ℚ := 2.50 -- dollars per gallon

-- Define the statement we want to prove
theorem net_pay_rate_per_hour : 
  (travel_hours * speed_mph * pay_rate_per_mile - 
  (travel_hours * speed_mph / fuel_efficiency) * gas_cost_per_gallon) / 
  travel_hours = 25 :=
by
  repeat {sorry}

end NUMINAMATH_GPT_net_pay_rate_per_hour_l2103_210355


namespace NUMINAMATH_GPT_solve_equation_l2103_210387

theorem solve_equation (x : ℝ) : 
  (x - 1) / 2 - (2 * x + 3) / 3 = 1 ↔ 3 * (x - 1) - 2 * (2 * x + 3) = 6 := 
sorry

end NUMINAMATH_GPT_solve_equation_l2103_210387


namespace NUMINAMATH_GPT_gain_percent_correct_l2103_210332

variable (CP SP Gain : ℝ)
variable (H₁ : CP = 900)
variable (H₂ : SP = 1125)
variable (H₃ : Gain = SP - CP)

theorem gain_percent_correct : (Gain / CP) * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_gain_percent_correct_l2103_210332


namespace NUMINAMATH_GPT_max_ab_l2103_210344

theorem max_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : ab ≤ 1 / 4 :=
sorry

end NUMINAMATH_GPT_max_ab_l2103_210344


namespace NUMINAMATH_GPT_tiles_per_row_l2103_210313

theorem tiles_per_row (area : ℝ) (tile_length : ℝ) (h1 : area = 256) (h2 : tile_length = 2/3) : 
  (16 * 12) / (8) = 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_tiles_per_row_l2103_210313


namespace NUMINAMATH_GPT_min_adj_white_pairs_l2103_210367

theorem min_adj_white_pairs (black_cells : Finset (Fin 64)) (h_black_count : black_cells.card = 20) : 
  ∃ rem_white_pairs, rem_white_pairs = 34 := 
sorry

end NUMINAMATH_GPT_min_adj_white_pairs_l2103_210367


namespace NUMINAMATH_GPT_dan_must_exceed_speed_to_arrive_before_cara_l2103_210394

noncomputable def minimum_speed_for_dan (distance : ℕ) (cara_speed : ℕ) (dan_delay : ℕ) : ℕ :=
  (distance / (distance / cara_speed - dan_delay)) + 1

theorem dan_must_exceed_speed_to_arrive_before_cara
  (distance : ℕ) (cara_speed : ℕ) (dan_delay : ℕ) :
  distance = 180 →
  cara_speed = 30 →
  dan_delay = 1 →
  minimum_speed_for_dan distance cara_speed dan_delay > 36 :=
by
  sorry

end NUMINAMATH_GPT_dan_must_exceed_speed_to_arrive_before_cara_l2103_210394


namespace NUMINAMATH_GPT_smallest_two_digit_palindrome_l2103_210330

def is_palindrome {α : Type} [DecidableEq α] (xs : List α) : Prop :=
  xs = xs.reverse

-- A number is a two-digit palindrome in base 5 if it has the form ab5 where a and b are digits 0-4
def two_digit_palindrome_base5 (n : ℕ) : Prop :=
  ∃ a b : ℕ, a < 5 ∧ b < 5 ∧ a ≠ 0 ∧ n = a * 5 + b ∧ is_palindrome [a, b]

-- A number is a three-digit palindrome in base 2 if it has the form abc2 where a = c and b can vary (0-1)
def three_digit_palindrome_base2 (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a < 2 ∧ b < 2 ∧ c < 2 ∧ a = c ∧ n = a * 4 + b * 2 + c ∧ is_palindrome [a, b, c]

theorem smallest_two_digit_palindrome :
  ∃ n, two_digit_palindrome_base5 n ∧ three_digit_palindrome_base2 n ∧
       (∀ m, two_digit_palindrome_base5 m ∧ three_digit_palindrome_base2 m → n ≤ m) :=
sorry

end NUMINAMATH_GPT_smallest_two_digit_palindrome_l2103_210330


namespace NUMINAMATH_GPT_greatest_divisor_of_976543_and_897623_l2103_210396

theorem greatest_divisor_of_976543_and_897623 :
  Nat.gcd (976543 - 7) (897623 - 11) = 4 := by
  sorry

end NUMINAMATH_GPT_greatest_divisor_of_976543_and_897623_l2103_210396


namespace NUMINAMATH_GPT_probability_is_3888_over_7533_l2103_210375

noncomputable def probability_odd_sum_given_even_product : ℚ := 
  let total_outcomes := 6^5
  let all_odd_outcomes := 3^5
  let at_least_one_even_outcomes := total_outcomes - all_odd_outcomes
  let favorable_outcomes := 5 * 3^4 + 10 * 3^4 + 3^5
  favorable_outcomes / at_least_one_even_outcomes

theorem probability_is_3888_over_7533 :
  probability_odd_sum_given_even_product = 3888 / 7533 := 
sorry

end NUMINAMATH_GPT_probability_is_3888_over_7533_l2103_210375


namespace NUMINAMATH_GPT_slices_per_large_pizza_l2103_210362

structure PizzaData where
  total_pizzas : Nat
  small_pizzas : Nat
  medium_pizzas : Nat
  slices_per_small : Nat
  slices_per_medium : Nat
  total_slices : Nat

def large_slices (data : PizzaData) : Nat := (data.total_slices - (data.small_pizzas * data.slices_per_small + data.medium_pizzas * data.slices_per_medium)) / (data.total_pizzas - data.small_pizzas - data.medium_pizzas)

def PizzaSlicingConditions := {data : PizzaData // 
  data.total_pizzas = 15 ∧
  data.small_pizzas = 4 ∧
  data.medium_pizzas = 5 ∧
  data.slices_per_small = 6 ∧
  data.slices_per_medium = 8 ∧
  data.total_slices = 136}

theorem slices_per_large_pizza (data : PizzaSlicingConditions) : large_slices data.val = 12 :=
by
  sorry

end NUMINAMATH_GPT_slices_per_large_pizza_l2103_210362


namespace NUMINAMATH_GPT_prime_odd_sum_l2103_210301

theorem prime_odd_sum (x y : ℕ) (h_prime : Prime x) (h_odd : y % 2 = 1) (h_eq : x^2 + y = 2005) : x + y = 2003 :=
by
  sorry

end NUMINAMATH_GPT_prime_odd_sum_l2103_210301


namespace NUMINAMATH_GPT_ways_to_divide_day_l2103_210380

theorem ways_to_divide_day (n m : ℕ) (h : n * m = 86400) : 
  (∃ k : ℕ, k = 96) :=
  sorry

end NUMINAMATH_GPT_ways_to_divide_day_l2103_210380


namespace NUMINAMATH_GPT_blanch_breakfast_slices_l2103_210329

-- Define the initial number of slices
def initial_slices : ℕ := 15

-- Define the slices eaten at different times
def lunch_slices : ℕ := 2
def snack_slices : ℕ := 2
def dinner_slices : ℕ := 5

-- Define the number of slices left
def slices_left : ℕ := 2

-- Calculate the total slices eaten during lunch, snack, and dinner
def total_eaten_ex_breakfast : ℕ := lunch_slices + snack_slices + dinner_slices

-- Define the slices eaten during breakfast
def breakfast_slices : ℕ := initial_slices - total_eaten_ex_breakfast - slices_left

-- The theorem to prove
theorem blanch_breakfast_slices : breakfast_slices = 4 := by
  sorry

end NUMINAMATH_GPT_blanch_breakfast_slices_l2103_210329


namespace NUMINAMATH_GPT_length_of_interval_l2103_210340

theorem length_of_interval (a b : ℝ) (h : 10 = (b - a) / 2) : b - a = 20 :=
by 
  sorry

end NUMINAMATH_GPT_length_of_interval_l2103_210340


namespace NUMINAMATH_GPT_find_a8_l2103_210324

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
def geom_sequence (a : ℕ → ℝ) (a1 q : ℝ) :=
  ∀ n, a n = a1 * q ^ n

def sum_geom_sequence (S a : ℕ → ℝ) (a1 q : ℝ) :=
  ∀ n, S n = a1 * (1 - q ^ (n + 1)) / (1 - q)

def arithmetic_sequence (S : ℕ → ℝ) :=
  S 9 = S 3 + S 6

def sum_a2_a5 (a : ℕ → ℝ) :=
  a 2 + a 5 = 4

theorem find_a8 (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 q : ℝ)
  (hgeom_seq : geom_sequence a a1 q)
  (hsum_geom_seq : sum_geom_sequence S a a1 q)
  (harith_seq : arithmetic_sequence S)
  (hsum_a2_a5 : sum_a2_a5 a) :
  a 8 = 2 :=
sorry

end NUMINAMATH_GPT_find_a8_l2103_210324


namespace NUMINAMATH_GPT_val_4_at_6_l2103_210307

def at_op (a b : ℤ) : ℤ := 2 * a - 4 * b

theorem val_4_at_6 : at_op 4 6 = -16 := by
  sorry

end NUMINAMATH_GPT_val_4_at_6_l2103_210307


namespace NUMINAMATH_GPT_star_is_addition_l2103_210393

theorem star_is_addition (star : ℝ → ℝ → ℝ) 
  (H : ∀ a b c : ℝ, star (star a b) c = a + b + c) : 
  ∀ a b : ℝ, star a b = a + b :=
by
  sorry

end NUMINAMATH_GPT_star_is_addition_l2103_210393


namespace NUMINAMATH_GPT_sum_of_triangles_l2103_210322

def triangle (a b c : ℤ) : ℤ := a + b - c

theorem sum_of_triangles : triangle 1 3 4 + triangle 2 5 6 = 1 := by
  sorry

end NUMINAMATH_GPT_sum_of_triangles_l2103_210322


namespace NUMINAMATH_GPT_set_expression_l2103_210386

def is_natural_number (x : ℚ) : Prop :=
  ∃ n : ℕ, x = n

theorem set_expression :
  {x : ℕ | is_natural_number (6 / (5 - x) : ℚ)} = {2, 3, 4} :=
sorry

end NUMINAMATH_GPT_set_expression_l2103_210386


namespace NUMINAMATH_GPT_dark_more_than_light_l2103_210363

-- Define the board size
def board_size : ℕ := 9

-- Define the number of dark squares in odd rows
def dark_in_odd_row : ℕ := 5

-- Define the number of light squares in odd rows
def light_in_odd_row : ℕ := 4

-- Define the number of dark squares in even rows
def dark_in_even_row : ℕ := 4

-- Define the number of light squares in even rows
def light_in_even_row : ℕ := 5

-- Calculate the total number of dark squares
def total_dark_squares : ℕ := (dark_in_odd_row * ((board_size + 1) / 2)) + (dark_in_even_row * (board_size / 2))

-- Calculate the total number of light squares
def total_light_squares : ℕ := (light_in_odd_row * ((board_size + 1) / 2)) + (light_in_even_row * (board_size / 2))

-- Define the main theorem
theorem dark_more_than_light : total_dark_squares - total_light_squares = 1 := by
  sorry

end NUMINAMATH_GPT_dark_more_than_light_l2103_210363


namespace NUMINAMATH_GPT_find_k_l2103_210349

theorem find_k (k : ℝ) : 
  (1 : ℝ)^2 + k * 1 - 3 = 0 → k = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_k_l2103_210349


namespace NUMINAMATH_GPT_bob_hair_length_l2103_210391

theorem bob_hair_length (h_0 : ℝ) (r : ℝ) (t : ℝ) (months_per_year : ℝ) (h : ℝ) :
  h_0 = 6 ∧ r = 0.5 ∧ t = 5 ∧ months_per_year = 12 → h = h_0 + r * months_per_year * t :=
sorry

end NUMINAMATH_GPT_bob_hair_length_l2103_210391


namespace NUMINAMATH_GPT_determine_A_l2103_210304

noncomputable def is_single_digit (n : ℕ) : Prop := n < 10

theorem determine_A (A B C : ℕ) (hABC : 3 * (100 * A + 10 * B + C) = 888)
  (hA_single_digit : is_single_digit A) (hB_single_digit : is_single_digit B) (hC_single_digit : is_single_digit C)
  (h_different : A ≠ B ∧ B ≠ C ∧ A ≠ C) : A = 2 := 
  sorry

end NUMINAMATH_GPT_determine_A_l2103_210304


namespace NUMINAMATH_GPT_system_of_equations_solution_l2103_210303

theorem system_of_equations_solution
  (x y z : ℤ)
  (h1 : x + y + z = 12)
  (h2 : 8 * x + 5 * y + 3 * z = 60) :
  (x = 0 ∧ y = 12 ∧ z = 0) ∨
  (x = 2 ∧ y = 7 ∧ z = 3) ∨
  (x = 4 ∧ y = 2 ∧ z = 6) :=
sorry

end NUMINAMATH_GPT_system_of_equations_solution_l2103_210303


namespace NUMINAMATH_GPT_fred_total_cards_l2103_210379

theorem fred_total_cards 
  (initial_cards : ℕ := 26) 
  (cards_given_to_mary : ℕ := 18) 
  (unopened_box_cards : ℕ := 40) : 
  initial_cards - cards_given_to_mary + unopened_box_cards = 48 := 
by 
  sorry

end NUMINAMATH_GPT_fred_total_cards_l2103_210379


namespace NUMINAMATH_GPT_point_outside_circle_l2103_210357

theorem point_outside_circle (a b : ℝ) (h : ∃ x y : ℝ, (a * x + b * y = 1) ∧ (x^2 + y^2 = 1)) :
  a^2 + b^2 > 1 := by
  sorry

end NUMINAMATH_GPT_point_outside_circle_l2103_210357


namespace NUMINAMATH_GPT_solve_fraction_eq_zero_l2103_210399

theorem solve_fraction_eq_zero (x : ℝ) (h₁ : 3 - x = 0) (h₂ : 4 + 2 * x ≠ 0) : x = 3 :=
by sorry

end NUMINAMATH_GPT_solve_fraction_eq_zero_l2103_210399


namespace NUMINAMATH_GPT_amoeba_reproduction_time_l2103_210341

/--
An amoeba reproduces by fission, splitting itself into two separate amoebae. 
It takes 8 days for one amoeba to divide into 16 amoebae. 

Prove that it takes 2 days for an amoeba to reproduce.
-/
theorem amoeba_reproduction_time (day_per_cycle : ℕ) (n_cycles : ℕ) 
  (h1 : n_cycles * day_per_cycle = 8)
  (h2 : 2^n_cycles = 16) : 
  day_per_cycle = 2 :=
by
  sorry

end NUMINAMATH_GPT_amoeba_reproduction_time_l2103_210341


namespace NUMINAMATH_GPT_quadratic_function_min_value_l2103_210312

noncomputable def f (a h k : ℝ) (x : ℝ) : ℝ :=
  a * (x - h) ^ 2 + k

theorem quadratic_function_min_value :
  ∀ (f : ℝ → ℝ) (n : ℕ),
  (f n = 13) ∧ (f (n + 1) = 13) ∧ (f (n + 2) = 35) →
  (∃ k, k = 2) :=
  sorry

end NUMINAMATH_GPT_quadratic_function_min_value_l2103_210312


namespace NUMINAMATH_GPT_root_of_quadratic_eq_l2103_210376

theorem root_of_quadratic_eq (a b : ℝ) (h : a + b - 3 = 0) : a + b = 3 :=
sorry

end NUMINAMATH_GPT_root_of_quadratic_eq_l2103_210376


namespace NUMINAMATH_GPT_am_gm_inequality_even_sum_l2103_210374

theorem am_gm_inequality_even_sum (a b : ℤ) (ha : 0 < a) (hb : 0 < b) (h_even : (a + b) % 2 = 0) :
  (a + b : ℚ) / 2 ≥ Real.sqrt (a * b) :=
sorry

end NUMINAMATH_GPT_am_gm_inequality_even_sum_l2103_210374


namespace NUMINAMATH_GPT_one_greater_one_smaller_l2103_210327

theorem one_greater_one_smaller (a b : ℝ) (h : ( (1 + a * b) / (a + b) )^2 < 1) :
  (a > 1 ∧ -1 < b ∧ b < 1) ∨ (b > 1 ∧ -1 < a ∧ a < 1) ∨ (a < -1 ∧ -1 < b ∧ b < 1) ∨ (b < -1 ∧ -1 < a ∧ a < 1) :=
by
  sorry

end NUMINAMATH_GPT_one_greater_one_smaller_l2103_210327


namespace NUMINAMATH_GPT_sufficient_condition_for_product_l2103_210321

-- Given conditions
def intersects_parabola_at_two_points (x1 y1 x2 y2 : ℝ) : Prop :=
  y1^2 = 4 * x1 ∧ y2^2 = 4 * x2 ∧ x1 ≠ x2

def line_through_focus (x y : ℝ) (k : ℝ) : Prop :=
  y = k * (x - 1)

noncomputable def line_l (k : ℝ) (x : ℝ) : ℝ :=
  k * (x - 1)

-- The theorem to prove
theorem sufficient_condition_for_product 
  (x1 y1 x2 y2 k : ℝ)
  (h1 : intersects_parabola_at_two_points x1 y1 x2 y2)
  (h2 : line_through_focus x1 y1 k)
  (h3 : line_through_focus x2 y2 k) :
  x1 * x2 = 1 :=
sorry

end NUMINAMATH_GPT_sufficient_condition_for_product_l2103_210321


namespace NUMINAMATH_GPT_average_marks_second_class_l2103_210385

variable (average_marks_first_class : ℝ) (students_first_class : ℕ)
variable (students_second_class : ℕ) (combined_average_marks : ℝ)

theorem average_marks_second_class (H1 : average_marks_first_class = 60)
  (H2 : students_first_class = 55) (H3 : students_second_class = 48)
  (H4 : combined_average_marks = 59.067961165048544) :
  48 * 57.92 = 103 * 59.067961165048544 - 3300 := by
  sorry

end NUMINAMATH_GPT_average_marks_second_class_l2103_210385


namespace NUMINAMATH_GPT_problem1_problem2_l2103_210366

-- Problem 1 Statement
theorem problem1 : (3 * Real.sqrt 48 - 2 * Real.sqrt 27) / Real.sqrt 3 = 6 :=
by sorry

-- Problem 2 Statement
theorem problem2 : 
  (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) - Real.sqrt ((-3)^2) + 1 / (2 - Real.sqrt 5) = -3 - Real.sqrt 5 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l2103_210366


namespace NUMINAMATH_GPT_additional_interest_due_to_higher_rate_l2103_210356

def principal : ℝ := 2500
def rate1 : ℝ := 0.18
def rate2 : ℝ := 0.12
def time : ℝ := 2

def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

theorem additional_interest_due_to_higher_rate :
  simple_interest principal rate1 time - simple_interest principal rate2 time = 300 :=
by
  sorry

end NUMINAMATH_GPT_additional_interest_due_to_higher_rate_l2103_210356


namespace NUMINAMATH_GPT_train_length_490_l2103_210342

noncomputable def train_length (speed_kmh : ℕ) (time_sec : ℕ) : ℕ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_sec

theorem train_length_490 :
  train_length 63 28 = 490 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_train_length_490_l2103_210342


namespace NUMINAMATH_GPT_problem_l2103_210398

def f (x : ℝ) : ℝ := x^3 + 2 * x

theorem problem : f 5 + f (-5) = 0 := by
  sorry

end NUMINAMATH_GPT_problem_l2103_210398


namespace NUMINAMATH_GPT_typhoon_probabilities_l2103_210336

-- Defining the conditions
def probAtLeastOneHit : ℝ := 0.36

-- Defining the events and probabilities
def probOfHit (p : ℝ) := p
def probBothHit (p : ℝ) := p^2

def probAtLeastOne (p : ℝ) : ℝ := p^2 + 2 * p * (1 - p)

-- Defining the variable X as the number of cities hit by the typhoon
def P_X_0 (p : ℝ) : ℝ := (1 - p)^2
def P_X_1 (p : ℝ) : ℝ := 2 * p * (1 - p)
def E_X (p : ℝ) : ℝ := 2 * p

-- Main theorem
theorem typhoon_probabilities :
  ∀ (p : ℝ),
    probAtLeastOne p = probAtLeastOneHit → 
    p = 0.2 ∧ P_X_0 p = 0.64 ∧ P_X_1 p = 0.32 ∧ E_X p = 0.4 :=
by
  intros p h
  sorry

end NUMINAMATH_GPT_typhoon_probabilities_l2103_210336


namespace NUMINAMATH_GPT_age_ratio_l2103_210388

noncomputable def rahul_present_age (future_age : ℕ) (years_passed : ℕ) : ℕ := future_age - years_passed

theorem age_ratio (future_rahul_age : ℕ) (years_passed : ℕ) (deepak_age : ℕ) :
  future_rahul_age = 26 →
  years_passed = 6 →
  deepak_age = 15 →
  rahul_present_age future_rahul_age years_passed / deepak_age = 4 / 3 :=
by
  intros
  have h1 : rahul_present_age 26 6 = 20 := rfl
  sorry

end NUMINAMATH_GPT_age_ratio_l2103_210388


namespace NUMINAMATH_GPT_find_missing_number_l2103_210365

theorem find_missing_number (x : ℚ) (h : (476 + 424) * 2 - x * 476 * 424 = 2704) : 
  x = -1 / 223 :=
by
  sorry

end NUMINAMATH_GPT_find_missing_number_l2103_210365


namespace NUMINAMATH_GPT_sum_of_midpoints_x_coordinates_l2103_210317

theorem sum_of_midpoints_x_coordinates (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := by
  sorry

end NUMINAMATH_GPT_sum_of_midpoints_x_coordinates_l2103_210317


namespace NUMINAMATH_GPT_problem_f_2016_eq_l2103_210370

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin x + b * x^3 + 4
noncomputable def f' (a b : ℝ) (x : ℝ) : ℝ := a * Real.cos x + 3 * b * x^2

theorem problem_f_2016_eq :
  ∀ (a b : ℝ),
  f a b 2016 + f a b (-2016) + f' a b 2017 - f' a b (-2017) = 8 + 2 * b * 2016^3 :=
by
  intro a b
  sorry

end NUMINAMATH_GPT_problem_f_2016_eq_l2103_210370


namespace NUMINAMATH_GPT_D_72_eq_81_l2103_210333

-- Definition of the function for the number of decompositions
def D (n : Nat) : Nat :=
  -- D(n) would ideally be implemented here as per the given conditions
  sorry

-- Prime factorization of 72
def prime_factorization_72 : List Nat :=
  [2, 2, 2, 3, 3]

-- Statement to prove
theorem D_72_eq_81 : D 72 = 81 :=
by
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_D_72_eq_81_l2103_210333


namespace NUMINAMATH_GPT_find_angle4_l2103_210339

theorem find_angle4
  (angle1 angle2 angle3 angle4 : ℝ)
  (h1 : angle1 = 70)
  (h2 : angle2 = 110)
  (h3 : angle3 = 40)
  (h4 : angle2 + angle3 + angle4 = 180) :
  angle4 = 30 := 
  sorry

end NUMINAMATH_GPT_find_angle4_l2103_210339


namespace NUMINAMATH_GPT_factor_polynomial_l2103_210364

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end NUMINAMATH_GPT_factor_polynomial_l2103_210364


namespace NUMINAMATH_GPT_unique_polynomial_P_l2103_210316

open Polynomial

/-- The only polynomial P with real coefficients such that
    xP(y/x) + yP(x/y) = x + y for all nonzero real numbers x and y 
    is P(x) = x. --/
theorem unique_polynomial_P (P : ℝ[X]) (hP : ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x * P.eval (y / x) + y * P.eval (x / y) = x + y) :
P = Polynomial.C 1 * X :=
by sorry

end NUMINAMATH_GPT_unique_polynomial_P_l2103_210316


namespace NUMINAMATH_GPT_minimum_value_x2_minus_x1_range_of_a_l2103_210395

noncomputable def f (x : ℝ) := Real.sin x + Real.exp x
noncomputable def g (x : ℝ) (a : ℝ) := a * x
noncomputable def F (x : ℝ) (a : ℝ) := f x - g x a

-- Question (I)
theorem minimum_value_x2_minus_x1 : ∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ 0 ≤ x₂ ∧ a = 1 / 3 ∧ f x₁ = g x₂ a → x₂ - x₁ = 3 := 
sorry

-- Question (II)
theorem range_of_a (a : ℝ) : (∀ x ≥ 0, F x a ≥ F (-x) a) ↔ a ≤ 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_x2_minus_x1_range_of_a_l2103_210395


namespace NUMINAMATH_GPT_n_divisible_by_40_l2103_210311

theorem n_divisible_by_40 {n : ℕ} (h_pos : 0 < n)
  (h1 : ∃ k1 : ℕ, 2 * n + 1 = k1 * k1)
  (h2 : ∃ k2 : ℕ, 3 * n + 1 = k2 * k2) :
  ∃ k : ℕ, n = 40 * k := 
sorry

end NUMINAMATH_GPT_n_divisible_by_40_l2103_210311


namespace NUMINAMATH_GPT_expand_expression_l2103_210359

theorem expand_expression (y : ℚ) : 5 * (4 * y^3 - 3 * y^2 + 2 * y - 6) = 20 * y^3 - 15 * y^2 + 10 * y - 30 := by
  sorry

end NUMINAMATH_GPT_expand_expression_l2103_210359


namespace NUMINAMATH_GPT_limit_of_power_seq_l2103_210390

-- Define the problem and its conditions
theorem limit_of_power_seq (a : ℝ) (h : 0 < a ∨ 1 < a) :
  (0 < a ∧ a < 1 → ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, a^n < ε) ∧ 
  (1 < a → ∀ N > 0, ∃ n : ℕ, a^n > N) :=
by
  sorry

end NUMINAMATH_GPT_limit_of_power_seq_l2103_210390


namespace NUMINAMATH_GPT_orchid_bushes_planted_l2103_210302

theorem orchid_bushes_planted (b1 b2 : ℕ) (h1 : b1 = 22) (h2 : b2 = 35) : b2 - b1 = 13 :=
by 
  sorry

end NUMINAMATH_GPT_orchid_bushes_planted_l2103_210302


namespace NUMINAMATH_GPT_sum_of_sequence_l2103_210350

noncomputable def sequence_sum (n : ℕ) : ℤ :=
  6 * 2^n - (n + 6)

theorem sum_of_sequence (a S : ℕ → ℤ) (n : ℕ) :
  a 1 = 5 →
  (∀ n : ℕ, 1 ≤ n → S (n + 1) = 2 * S n + n + 5) →
  S n = sequence_sum n :=
by sorry

end NUMINAMATH_GPT_sum_of_sequence_l2103_210350


namespace NUMINAMATH_GPT_triangle_side_eq_nine_l2103_210397

theorem triangle_side_eq_nine (a b c : ℕ) 
  (h_tri_ineq : a + b > c ∧ a + c > b ∧ b + c > a)
  (h_sqrt_eq : (Nat.sqrt (a - 9)) + (b - 2)^2 = 0)
  (h_c_odd : c % 2 = 1) :
  c = 9 :=
sorry

end NUMINAMATH_GPT_triangle_side_eq_nine_l2103_210397


namespace NUMINAMATH_GPT_longer_side_length_l2103_210300

theorem longer_side_length (x y : ℝ) (h1 : 2 * x + 2 * y = 60) (h2 : x * y = 221) : max x y = 17 :=
by
  sorry

end NUMINAMATH_GPT_longer_side_length_l2103_210300


namespace NUMINAMATH_GPT_does_not_pass_through_second_quadrant_l2103_210343

def line_eq (x y : ℝ) : Prop := x - y - 1 = 0

theorem does_not_pass_through_second_quadrant :
  ¬ ∃ (x y : ℝ), line_eq x y ∧ x < 0 ∧ y > 0 :=
sorry

end NUMINAMATH_GPT_does_not_pass_through_second_quadrant_l2103_210343


namespace NUMINAMATH_GPT_line_tangent_to_parabola_l2103_210351

theorem line_tangent_to_parabola (c : ℝ) : (∀ (x y : ℝ), 2 * x - y + c = 0 ∧ x^2 = 4 * y) → c = -4 := by
  sorry

end NUMINAMATH_GPT_line_tangent_to_parabola_l2103_210351


namespace NUMINAMATH_GPT_solution_exists_unique_n_l2103_210389

theorem solution_exists_unique_n (n : ℕ) : 
  (∀ m : ℕ, (10 * m > 120) ∨ ∃ k1 k2 k3 : ℕ, 10 * k1 + n * k2 + (n + 1) * k3 = 120) = false → 
  n = 16 := by sorry

end NUMINAMATH_GPT_solution_exists_unique_n_l2103_210389


namespace NUMINAMATH_GPT_graph_transform_l2103_210360

-- Define the quadratic function y1 as y = -2x^2 + 4x + 1
def y1 (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 1

-- Define the quadratic function y2 as y = -2x^2
def y2 (x : ℝ) : ℝ := -2 * x^2

-- Define the transformation function for moving 1 unit to the left and 3 units down
def transform (y : ℝ → ℝ) (x : ℝ) : ℝ := y (x + 1) - 3

-- Statement to prove
theorem graph_transform : ∀ x : ℝ, transform y1 x = y2 x :=
by
  intros x
  sorry

end NUMINAMATH_GPT_graph_transform_l2103_210360


namespace NUMINAMATH_GPT_least_four_digit_palindrome_divisible_by_11_l2103_210305

theorem least_four_digit_palindrome_divisible_by_11 : 
  ∃ (A B : ℕ), (A ≠ 0 ∧ A < 10 ∧ B < 10 ∧ 1000 * A + 100 * B + 10 * B + A = 1111 ∧ (2 * A - 2 * B) % 11 = 0) := 
by
  sorry

end NUMINAMATH_GPT_least_four_digit_palindrome_divisible_by_11_l2103_210305


namespace NUMINAMATH_GPT_distance_focus_directrix_l2103_210377

theorem distance_focus_directrix (p : ℝ) (x_1 : ℝ) (h1 : 0 < p) (h2 : x_1^2 = 2 * p)
  (h3 : 1 + p / 2 = 3) : p = 4 :=
by
  sorry

end NUMINAMATH_GPT_distance_focus_directrix_l2103_210377


namespace NUMINAMATH_GPT_red_star_team_wins_l2103_210369

theorem red_star_team_wins (x y : ℕ) (h1 : x + y = 9) (h2 : 3 * x + y = 23) : x = 7 := by
  sorry

end NUMINAMATH_GPT_red_star_team_wins_l2103_210369


namespace NUMINAMATH_GPT_husband_and_wife_age_l2103_210310

theorem husband_and_wife_age (x y : ℕ) (h1 : 11 * x = 2 * (22 * y - 11 * x)) (h2 : 11 * x ≠ 0) (h3 : 11 * y ≠ 0) (h4 : 11 * (x + y) ≤ 99) : 
  x = 4 ∧ y = 3 :=
by
  sorry

end NUMINAMATH_GPT_husband_and_wife_age_l2103_210310
