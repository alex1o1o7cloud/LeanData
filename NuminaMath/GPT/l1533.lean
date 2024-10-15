import Mathlib

namespace NUMINAMATH_GPT_triangle_max_area_in_quarter_ellipse_l1533_153346

theorem triangle_max_area_in_quarter_ellipse (a b c : ℝ) (h : c^2 = a^2 - b^2) :
  ∃ (T_max : ℝ), T_max = b / 2 :=
by sorry

end NUMINAMATH_GPT_triangle_max_area_in_quarter_ellipse_l1533_153346


namespace NUMINAMATH_GPT_rectangle_perimeter_l1533_153373

theorem rectangle_perimeter (s : ℕ) (h : 4 * s = 160) : 2 * (s + s / 4) = 100 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1533_153373


namespace NUMINAMATH_GPT_polynomial_factorization_l1533_153309

theorem polynomial_factorization :
  5 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 = 5 * x^4 + 180 * x^3 + 1431 * x^2 + 4900 * x + 5159 :=
by sorry

end NUMINAMATH_GPT_polynomial_factorization_l1533_153309


namespace NUMINAMATH_GPT_min_value_of_ab_l1533_153312

theorem min_value_of_ab {a b : ℝ} (ha : a > 0) (hb : b > 0) 
  (h_eq : ∀ (x y : ℝ), (x / a + y / b = 1) → (x^2 + y^2 = 1)) : a * b = 2 :=
by sorry

end NUMINAMATH_GPT_min_value_of_ab_l1533_153312


namespace NUMINAMATH_GPT_track_champion_races_l1533_153344

theorem track_champion_races (total_sprinters : ℕ) (lanes : ℕ) (eliminations_per_race : ℕ)
  (h1 : total_sprinters = 216) (h2 : lanes = 6) (h3 : eliminations_per_race = 5) : 
  (total_sprinters - 1) / eliminations_per_race = 43 :=
by
  -- We acknowledge that a proof is needed here. Placeholder for now.
  sorry

end NUMINAMATH_GPT_track_champion_races_l1533_153344


namespace NUMINAMATH_GPT_max_value_of_f_l1533_153399

noncomputable def f (x : ℝ) : ℝ := (1/5) * Real.sin (x + Real.pi/3) + Real.cos (x - Real.pi/6)

theorem max_value_of_f : ∀ x : ℝ, f x ≤ 6/5 := by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l1533_153399


namespace NUMINAMATH_GPT_tooth_extraction_cost_l1533_153352

variable (c f b e : ℕ)

-- Conditions
def cost_cleaning := c = 70
def cost_filling := f = 120
def bill := b = 5 * f

-- Proof Problem
theorem tooth_extraction_cost (h_cleaning : cost_cleaning c) (h_filling : cost_filling f) (h_bill : bill b f) :
  e = b - (c + 2 * f) :=
sorry

end NUMINAMATH_GPT_tooth_extraction_cost_l1533_153352


namespace NUMINAMATH_GPT_num_perpendicular_line_plane_pairs_in_cube_l1533_153358

-- Definitions based on the problem conditions

def is_perpendicular_line_plane_pair (l : line) (p : plane) : Prop :=
  -- Assume an implementation that defines when a line is perpendicular to a plane
  sorry

-- Define a cube structure with its vertices, edges, and faces
structure Cube :=
  (vertices : Finset Point)
  (edges : Finset (Point × Point))
  (faces : Finset (Finset Point))

-- Make assumptions about cube properties
variable (cube : Cube)

-- Define the property of counting perpendicular line-plane pairs
def count_perpendicular_line_plane_pairs (c : Cube) : Nat :=
  -- Assume an implementation that counts the number of such pairs in the cube
  sorry

-- The theorem to prove
theorem num_perpendicular_line_plane_pairs_in_cube (c : Cube) :
  count_perpendicular_line_plane_pairs c = 36 :=
  sorry

end NUMINAMATH_GPT_num_perpendicular_line_plane_pairs_in_cube_l1533_153358


namespace NUMINAMATH_GPT_jason_total_spent_l1533_153349

def cost_of_flute : ℝ := 142.46
def cost_of_music_tool : ℝ := 8.89
def cost_of_song_book : ℝ := 7.00

def total_spent (flute_cost music_tool_cost song_book_cost : ℝ) : ℝ :=
  flute_cost + music_tool_cost + song_book_cost

theorem jason_total_spent :
  total_spent cost_of_flute cost_of_music_tool cost_of_song_book = 158.35 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_jason_total_spent_l1533_153349


namespace NUMINAMATH_GPT_monotonically_increasing_interval_l1533_153359

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x) / Real.log (1/2)

theorem monotonically_increasing_interval :
  ∀ x, x ∈ Set.Iio (0 : ℝ) → StrictMono f :=
by
  sorry

end NUMINAMATH_GPT_monotonically_increasing_interval_l1533_153359


namespace NUMINAMATH_GPT_number_of_parrots_l1533_153300

noncomputable def daily_consumption_parakeet : ℕ := 2
noncomputable def daily_consumption_parrot : ℕ := 14
noncomputable def daily_consumption_finch : ℕ := 1  -- Each finch eats half of what a parakeet eats

noncomputable def num_parakeets : ℕ := 3
noncomputable def num_finches : ℕ := 4
noncomputable def required_birdseed : ℕ := 266
noncomputable def days_in_week : ℕ := 7

theorem number_of_parrots (num_parrots : ℕ) : 
  daily_consumption_parakeet * num_parakeets * days_in_week +
  daily_consumption_finch * num_finches * days_in_week + 
  daily_consumption_parrot * num_parrots * days_in_week = required_birdseed → num_parrots = 2 :=
by 
  -- The proof is omitted as per the instructions
  sorry

end NUMINAMATH_GPT_number_of_parrots_l1533_153300


namespace NUMINAMATH_GPT_value_of_a_range_of_m_l1533_153394

def f (x a : ℝ) : ℝ := abs (x - a)

-- Given the following conditions
axiom cond1 (x : ℝ) (a : ℝ) : f x a = abs (x - a)
axiom cond2 (x : ℝ) (a : ℝ) : (f x a >= 3) ↔ (x <= 1 ∨ x >= 5)

-- Prove that a = 2
theorem value_of_a (a : ℝ) : (∀ x : ℝ, (f x a >= 3) ↔ (x <= 1 ∨ x >= 5)) → a = 2 := by
  sorry

-- Additional condition for m
axiom cond3 (x : ℝ) (a : ℝ) (m : ℝ) : ∀ x : ℝ, f x a + f (x + 4) a >= m

-- Prove that m ≤ 4
theorem range_of_m (a : ℝ) (m : ℝ) : (∀ x : ℝ, f x a + f (x + 4) a >= m) → a = 2 → m ≤ 4 := by
  sorry

end NUMINAMATH_GPT_value_of_a_range_of_m_l1533_153394


namespace NUMINAMATH_GPT_find_prime_n_l1533_153375

theorem find_prime_n (n k m : ℤ) (h1 : n - 6 = k ^ 2) (h2 : n + 10 = m ^ 2) (h3 : m ^ 2 - k ^ 2 = 16) (h4 : Nat.Prime (Int.natAbs n)) : n = 71 := by
  sorry

end NUMINAMATH_GPT_find_prime_n_l1533_153375


namespace NUMINAMATH_GPT_total_notes_in_week_l1533_153351

-- Define the conditions for day hours ring pattern
def day_notes (hour : ℕ) (minute : ℕ) : ℕ :=
  if minute = 15 then 2
  else if minute = 30 then 4
  else if minute = 45 then 6
  else if minute = 0 then 
    8 + (if hour % 2 = 0 then hour else hour / 2)
  else 0

-- Define the conditions for night hours ring pattern
def night_notes (hour : ℕ) (minute : ℕ) : ℕ :=
  if minute = 15 then 3
  else if minute = 30 then 5
  else if minute = 45 then 7
  else if minute = 0 then 
    9 + (if hour % 2 = 1 then hour else hour / 2)
  else 0

-- Define total notes over day period
def total_day_notes : ℕ := 
  (day_notes 6 0 + day_notes 7 0 + day_notes 8 0 + day_notes 9 0 + day_notes 10 0 + day_notes 11 0
 + day_notes 12 0 + day_notes 1 0 + day_notes 2 0 + day_notes 3 0 + day_notes 4 0 + day_notes 5 0)
 +
 (2 * 12 + 4 * 12 + 6 * 12)

-- Define total notes over night period
def total_night_notes : ℕ := 
  (night_notes 6 0 + night_notes 7 0 + night_notes 8 0 + night_notes 9 0 + night_notes 10 0 + night_notes 11 0
 + night_notes 12 0 + night_notes 1 0 + night_notes 2 0 + night_notes 3 0 + night_notes 4 0 + night_notes 5 0)
 +
 (3 * 12 + 5 * 12 + 7 * 12)

-- Define the total number of notes the clock will ring in a full week
def total_week_notes : ℕ :=
  7 * (total_day_notes + total_night_notes)

theorem total_notes_in_week : 
  total_week_notes = 3297 := 
  by 
  sorry

end NUMINAMATH_GPT_total_notes_in_week_l1533_153351


namespace NUMINAMATH_GPT_find_numbers_l1533_153322

-- Definitions for the conditions
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_even_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 98 ∧ n % 2 = 0
def difference_is_three (x y : ℕ) : Prop := x - y = 3

-- Statement of the proof problem
theorem find_numbers (x y : ℕ) (h1 : is_three_digit x) (h2 : is_even_two_digit y) (h3 : difference_is_three x y) :
  x = 101 ∧ y = 98 :=
sorry

end NUMINAMATH_GPT_find_numbers_l1533_153322


namespace NUMINAMATH_GPT_mayo_bottle_count_l1533_153310

-- Define the given ratio and the number of ketchup bottles
def ratio_ketchup : ℕ := 3
def ratio_mustard : ℕ := 3
def ratio_mayo : ℕ := 2
def num_ketchup_bottles : ℕ := 6

-- Define the proof problem: The number of mayo bottles
theorem mayo_bottle_count :
  (num_ketchup_bottles / ratio_ketchup) * ratio_mayo = 4 :=
by sorry

end NUMINAMATH_GPT_mayo_bottle_count_l1533_153310


namespace NUMINAMATH_GPT_find_g_720_l1533_153389

noncomputable def g (n : ℕ) : ℕ := sorry

axiom g_multiplicative : ∀ (x y : ℕ), g (x * y) = g x + g y
axiom g_8 : g 8 = 12
axiom g_12 : g 12 = 16

theorem find_g_720 : g 720 = 44 := by sorry

end NUMINAMATH_GPT_find_g_720_l1533_153389


namespace NUMINAMATH_GPT_shaded_area_correct_l1533_153366

noncomputable def grid_width : ℕ := 15
noncomputable def grid_height : ℕ := 5
noncomputable def triangle_base : ℕ := 15
noncomputable def triangle_height : ℕ := 3
noncomputable def total_area : ℝ := (grid_width * grid_height : ℝ)
noncomputable def triangle_area : ℝ := (1 / 2) * triangle_base * triangle_height
noncomputable def shaded_area : ℝ := total_area - triangle_area

theorem shaded_area_correct : shaded_area = 52.5 := 
by sorry

end NUMINAMATH_GPT_shaded_area_correct_l1533_153366


namespace NUMINAMATH_GPT_sequence_a19_l1533_153379

theorem sequence_a19 :
  ∃ (a : ℕ → ℝ), a 3 = 2 ∧ a 7 = 1 ∧
    (∃ d : ℝ, ∀ n m : ℕ, (1 / (a n + 1) - 1 / (a m + 1)) / (n - m) = d) →
    a 19 = 0 :=
by sorry

end NUMINAMATH_GPT_sequence_a19_l1533_153379


namespace NUMINAMATH_GPT_calculate_DA_l1533_153318

open Real

-- Definitions based on conditions
def AU := 90
def AN := 180
def UB := 270
def AB := AU + UB
def ratio := 3 / 4

-- Statement of the problem in Lean 
theorem calculate_DA :
  ∃ (p q : ℕ), (q ≠ 0) ∧ (∀ p' q' : ℕ, ¬ (q = p'^2 * q')) ∧ DA = p * sqrt q ∧ p + q = result :=
  sorry

end NUMINAMATH_GPT_calculate_DA_l1533_153318


namespace NUMINAMATH_GPT_seventeenth_replacement_month_l1533_153330

def months_after_january (n : Nat) : Nat :=
  n % 12

theorem seventeenth_replacement_month :
  months_after_january (7 * 16) = 4 :=
by
  sorry

end NUMINAMATH_GPT_seventeenth_replacement_month_l1533_153330


namespace NUMINAMATH_GPT_quilt_block_shading_fraction_l1533_153328

theorem quilt_block_shading_fraction :
  (fraction_shaded : ℚ) → 
  (quilt_block_size : ℕ) → 
  (fully_shaded_squares : ℕ) → 
  (half_shaded_squares : ℕ) → 
  quilt_block_size = 16 →
  fully_shaded_squares = 6 →
  half_shaded_squares = 4 →
  fraction_shaded = 1/2 :=
by 
  sorry

end NUMINAMATH_GPT_quilt_block_shading_fraction_l1533_153328


namespace NUMINAMATH_GPT_victor_weekly_earnings_l1533_153360

def wage_per_hour : ℕ := 12
def hours_monday : ℕ := 5
def hours_tuesday : ℕ := 6
def hours_wednesday : ℕ := 7
def hours_thursday : ℕ := 4
def hours_friday : ℕ := 8

def earnings_monday := hours_monday * wage_per_hour
def earnings_tuesday := hours_tuesday * wage_per_hour
def earnings_wednesday := hours_wednesday * wage_per_hour
def earnings_thursday := hours_thursday * wage_per_hour
def earnings_friday := hours_friday * wage_per_hour

def total_earnings := earnings_monday + earnings_tuesday + earnings_wednesday + earnings_thursday + earnings_friday

theorem victor_weekly_earnings : total_earnings = 360 := by
  sorry

end NUMINAMATH_GPT_victor_weekly_earnings_l1533_153360


namespace NUMINAMATH_GPT_unique_handshakes_462_l1533_153356

theorem unique_handshakes_462 : 
  ∀ (twins triplets : Type) (twin_set : ℕ) (triplet_set : ℕ) (handshakes_among_twins handshakes_among_triplets cross_handshakes_twins cross_handshakes_triplets : ℕ),
  twin_set = 12 ∧
  triplet_set = 4 ∧
  handshakes_among_twins = (24 * 22) / 2 ∧
  handshakes_among_triplets = (12 * 9) / 2 ∧
  cross_handshakes_twins = 24 * (12 / 3) ∧
  cross_handshakes_triplets = 12 * (24 / 3 * 2) →
  (handshakes_among_twins + handshakes_among_triplets + (cross_handshakes_twins + cross_handshakes_triplets) / 2) = 462 := 
by
  sorry

end NUMINAMATH_GPT_unique_handshakes_462_l1533_153356


namespace NUMINAMATH_GPT_arithmetic_sequence_formula_geometric_sequence_sum_formula_l1533_153355

noncomputable def arithmetic_sequence_a_n (n : ℕ) : ℤ :=
  sorry

noncomputable def geometric_sequence_T_n (n : ℕ) : ℤ :=
  sorry

theorem arithmetic_sequence_formula :
  (∃ a₃ : ℤ, a₃ = 5) ∧ (∃ S₃ : ℤ, S₃ = 9) →
  -- Suppose we have an arithmetic sequence $a_n$
  (∀ n : ℕ, n ≥ 1 → arithmetic_sequence_a_n n = 2 * n - 1) := 
sorry

theorem geometric_sequence_sum_formula :
  (∃ q : ℤ, q > 0 ∧ q = 3) ∧ (∃ b₃ : ℤ, b₃ = 9) ∧ (∃ T₃ : ℤ, T₃ = 13) →
  -- Suppose we have a geometric sequence $b_n$ where $b_3 = a_5$
  (∀ n : ℕ, n ≥ 1 → geometric_sequence_T_n n = (3 ^ n - 1) / 2) := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_formula_geometric_sequence_sum_formula_l1533_153355


namespace NUMINAMATH_GPT_min_value_of_y_l1533_153370

theorem min_value_of_y (x : ℝ) (hx : x > 0) : (∃ y, y = x + 4 / x^2 ∧ ∀ z, z = x + 4 / x^2 → z ≥ 3) :=
sorry

end NUMINAMATH_GPT_min_value_of_y_l1533_153370


namespace NUMINAMATH_GPT_height_of_tree_in_kilmer_park_l1533_153337

-- Define the initial conditions
def initial_height_ft := 52
def growth_per_year_ft := 5
def years := 8
def ft_to_inch := 12

-- Define the expected result in inches
def expected_height_inch := 1104

-- State the problem as a theorem
theorem height_of_tree_in_kilmer_park :
  (initial_height_ft + growth_per_year_ft * years) * ft_to_inch = expected_height_inch :=
by
  sorry

end NUMINAMATH_GPT_height_of_tree_in_kilmer_park_l1533_153337


namespace NUMINAMATH_GPT_incorrect_statement_B_l1533_153378

theorem incorrect_statement_B (A B C : ℝ) (hAB : A * B < 0) (hBC : B * C < 0) : ¬ ∀ (x y : ℝ), x * y + A * x + B * y + C = 0 → (x < 0 ∧ y < 0) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_B_l1533_153378


namespace NUMINAMATH_GPT_jakes_digging_time_l1533_153332

theorem jakes_digging_time
  (J : ℕ)
  (Paul_work_rate : ℚ := 1/24)
  (Hari_work_rate : ℚ := 1/48)
  (Combined_work_rate : ℚ := 1/8)
  (Combined_work_eq : 1 / J + Paul_work_rate + Hari_work_rate = Combined_work_rate) :
  J = 16 := sorry

end NUMINAMATH_GPT_jakes_digging_time_l1533_153332


namespace NUMINAMATH_GPT_jacqueline_candy_multiple_l1533_153369

theorem jacqueline_candy_multiple :
  let fred_candy := 12
  let uncle_bob_candy := fred_candy + 6
  let total_candy := fred_candy + uncle_bob_candy
  let jackie_candy := 120 / 0.40
  (jackie_candy / total_candy = 10) :=
by
  let fred_candy := 12
  let uncle_bob_candy := fred_candy + 6
  let total_candy := fred_candy + uncle_bob_candy
  let jackie_candy := 120 / 0.40
  show _ = _
  sorry

end NUMINAMATH_GPT_jacqueline_candy_multiple_l1533_153369


namespace NUMINAMATH_GPT_children_tickets_l1533_153388

-- Definition of the problem
variables (A C t : ℕ) (h_eq_people : A + C = t) (h_eq_money : 9 * A + 5 * C = 190)

-- The main statement we need to prove
theorem children_tickets (h_t : t = 30) : C = 20 :=
by {
  -- Proof will go here eventually
  sorry
}

end NUMINAMATH_GPT_children_tickets_l1533_153388


namespace NUMINAMATH_GPT_solution_set_of_floor_equation_l1533_153307

theorem solution_set_of_floor_equation (x : ℝ) : 
  (⌊⌊2 * x⌋ - 1/2⌋ = ⌊x + 3⌋) ↔ (3.5 ≤ x ∧ x < 4.5) :=
by sorry

end NUMINAMATH_GPT_solution_set_of_floor_equation_l1533_153307


namespace NUMINAMATH_GPT_Brittany_second_test_grade_is_83_l1533_153381

theorem Brittany_second_test_grade_is_83
  (first_test_score : ℝ) (first_test_weight : ℝ) 
  (second_test_weight : ℝ) (final_weighted_average : ℝ) : 
  first_test_score = 78 → 
  first_test_weight = 0.40 →
  second_test_weight = 0.60 →
  final_weighted_average = 81 →
  ∃ G : ℝ, 0.40 * first_test_score + 0.60 * G = final_weighted_average ∧ G = 83 :=
by
  sorry

end NUMINAMATH_GPT_Brittany_second_test_grade_is_83_l1533_153381


namespace NUMINAMATH_GPT_birds_flew_up_count_l1533_153361

def initial_birds : ℕ := 29
def final_birds : ℕ := 42

theorem birds_flew_up_count : final_birds - initial_birds = 13 :=
by sorry

end NUMINAMATH_GPT_birds_flew_up_count_l1533_153361


namespace NUMINAMATH_GPT_jessica_withdrawal_l1533_153368

/-- Jessica withdrew some money from her bank account, causing her account balance to decrease by 2/5.
    She then deposited an amount equal to 1/4 of the remaining balance. The final balance in her bank account is $750.
    Prove that Jessica initially withdrew $400. -/
theorem jessica_withdrawal (X W : ℝ) 
  (initial_eq : W = (2 / 5) * X)
  (remaining_eq : X * (3 / 5) + (1 / 4) * (X * (3 / 5)) = 750) :
  W = 400 := 
sorry

end NUMINAMATH_GPT_jessica_withdrawal_l1533_153368


namespace NUMINAMATH_GPT_amount_B_l1533_153390

noncomputable def A : ℝ := sorry -- Definition of A
noncomputable def B : ℝ := sorry -- Definition of B

-- Conditions
def condition1 : Prop := A + B = 100
def condition2 : Prop := (3 / 10) * A = (1 / 5) * B

-- Statement to prove
theorem amount_B : condition1 ∧ condition2 → B = 60 :=
by
  intros
  sorry

end NUMINAMATH_GPT_amount_B_l1533_153390


namespace NUMINAMATH_GPT_max_gold_coins_l1533_153396

theorem max_gold_coins : ∃ n : ℕ, (∃ k : ℕ, n = 7 * k + 2) ∧ 50 < n ∧ n < 150 ∧ n = 149 :=
by
  sorry

end NUMINAMATH_GPT_max_gold_coins_l1533_153396


namespace NUMINAMATH_GPT_vector_evaluation_l1533_153331

-- Define the vectors
def v1 : ℝ × ℝ := (3, -2)
def v2 : ℝ × ℝ := (2, -6)
def v3 : ℝ × ℝ := (0, 3)
def scalar : ℝ := 5
def expected_result : ℝ × ℝ := (-7, 31)

-- Statement to be proved
theorem vector_evaluation : v1 - scalar • v2 + v3 = expected_result :=
by
  sorry

end NUMINAMATH_GPT_vector_evaluation_l1533_153331


namespace NUMINAMATH_GPT_max_value_f_l1533_153311

open Real

/-- Determine the maximum value of the function f(x) = 1 / (1 - x * (1 - x)). -/
theorem max_value_f (x : ℝ) : 
  ∃ y, y = (1 / (1 - x * (1 - x))) ∧ y ≤ 4/3 ∧ ∀ z, z = (1 / (1 - x * (1 - x))) → z ≤ 4/3 :=
by
  sorry

end NUMINAMATH_GPT_max_value_f_l1533_153311


namespace NUMINAMATH_GPT_problem_statement_l1533_153387

theorem problem_statement (h: 2994 * 14.5 = 175) : 29.94 * 1.45 = 1.75 := 
by {
  sorry
}

end NUMINAMATH_GPT_problem_statement_l1533_153387


namespace NUMINAMATH_GPT_pure_imaginary_a_l1533_153384

theorem pure_imaginary_a (a : ℝ) :
  (a^2 - 4 = 0) ∧ (a - 2 ≠ 0) ↔ a = -2 :=
by
  sorry

end NUMINAMATH_GPT_pure_imaginary_a_l1533_153384


namespace NUMINAMATH_GPT_final_answer_after_subtracting_l1533_153303

theorem final_answer_after_subtracting (n : ℕ) (h : n = 990) : (n / 9) - 100 = 10 :=
by
  sorry

end NUMINAMATH_GPT_final_answer_after_subtracting_l1533_153303


namespace NUMINAMATH_GPT_find_a_minus_b_l1533_153314

theorem find_a_minus_b (a b : ℤ) 
  (h1 : 3015 * a + 3019 * b = 3023) 
  (h2 : 3017 * a + 3021 * b = 3025) : 
  a - b = -3 := 
sorry

end NUMINAMATH_GPT_find_a_minus_b_l1533_153314


namespace NUMINAMATH_GPT_number_of_perfect_square_divisors_of_450_l1533_153377

theorem number_of_perfect_square_divisors_of_450 : 
    let p := 450;
    let factors := [(3, 2), (5, 2), (2, 1)];
    ∃ n, (n = 4 ∧ 
          ∀ (d : ℕ), d ∣ p → 
                     (∃ (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ 
                              (a = 0) ∧ (b = 0 ∨ b = 2) ∧ (c = 0 ∨ c = 2) → 
                              a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) :=
    sorry

end NUMINAMATH_GPT_number_of_perfect_square_divisors_of_450_l1533_153377


namespace NUMINAMATH_GPT_find_acute_angles_of_alex_triangle_l1533_153365

theorem find_acute_angles_of_alex_triangle (α : ℝ) (h1 : α > 0) (h2 : α < 90) :
  let condition1 := «Alex drew a geometric picture by tracing his plastic right triangle four times»
  let condition2 := «Each time aligning the shorter leg with the hypotenuse and matching the vertex of the acute angle with the vertex of the right angle»
  let condition3 := «The "closing" fifth triangle was isosceles»
  α = 90 / 11 :=
sorry

end NUMINAMATH_GPT_find_acute_angles_of_alex_triangle_l1533_153365


namespace NUMINAMATH_GPT_Maria_trip_time_l1533_153305

/-- 
Given:
- Maria drove 80 miles on a freeway.
- Maria drove 20 miles on a rural road.
- Her speed on the rural road was half of her speed on the freeway.
- Maria spent 40 minutes driving on the rural road.

Prove that Maria's entire trip took 120 minutes.
-/ 
theorem Maria_trip_time
  (distance_freeway : ℕ)
  (distance_rural : ℕ)
  (rural_speed_ratio : ℕ → ℕ)
  (time_rural_minutes : ℕ) 
  (time_freeway : ℕ)
  (total_time : ℕ) 
  (speed_rural : ℕ)
  (speed_freeway : ℕ) 
  :
  distance_freeway = 80 ∧
  distance_rural = 20 ∧ 
  rural_speed_ratio (speed_freeway) = speed_rural ∧ 
  time_rural_minutes = 40 ∧
  time_rural_minutes = 20 / speed_rural ∧
  speed_freeway = 2 * speed_rural ∧
  time_freeway = distance_freeway / speed_freeway ∧
  total_time = time_rural_minutes + time_freeway → 
  total_time = 120 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Maria_trip_time_l1533_153305


namespace NUMINAMATH_GPT_modulus_product_eq_sqrt_5_l1533_153398

open Complex

-- Define the given complex number.
def z : ℂ := 2 + I

-- Declare the product with I.
def z_product := z * I

-- State the theorem that the modulus of the product is sqrt(5).
theorem modulus_product_eq_sqrt_5 : abs z_product = Real.sqrt 5 := 
sorry

end NUMINAMATH_GPT_modulus_product_eq_sqrt_5_l1533_153398


namespace NUMINAMATH_GPT_smallest_positive_period_max_min_values_l1533_153357

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.sin x)
noncomputable def f (x : ℝ) : ℝ := (vector_a x).1 * (vector_b x).1 + (vector_a x).2 * (vector_b x).2 - 1 / 2

-- Theorem 1: Smallest positive period of the function f(x)
theorem smallest_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi :=
  sorry

-- Theorem 2: Maximum and minimum values of the function f(x) on [0, π/2]
theorem max_min_values : 
  ∀ x ∈ Set.Icc 0 (Real.pi / 2),
    f x ≤ 1 ∧ f x ≥ -1 / 2 ∧ (∃ (x_max : ℝ), x_max ∈ Set.Icc 0 (Real.pi / 2) ∧ f x_max = 1) ∧
    (∃ (x_min : ℝ), x_min ∈ Set.Icc 0 (Real.pi / 2) ∧ f x_min = -1 / 2) :=
  sorry

end NUMINAMATH_GPT_smallest_positive_period_max_min_values_l1533_153357


namespace NUMINAMATH_GPT_ellipse_triangle_perimeter_l1533_153306

-- Definitions based on conditions
def is_ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 2) = 1

-- Triangle perimeter calculation
def triangle_perimeter (a c : ℝ) : ℝ := 2 * a + 2 * c

-- Main theorem statement
theorem ellipse_triangle_perimeter :
  let a := 2
  let b2 := 2
  let c := Real.sqrt (a ^ 2 - b2)
  ∀ (P : ℝ × ℝ), (is_ellipse P.1 P.2) → triangle_perimeter a c = 4 + 2 * Real.sqrt 2 :=
by
  intros P hP
  -- Here, we would normally provide the proof.
  sorry

end NUMINAMATH_GPT_ellipse_triangle_perimeter_l1533_153306


namespace NUMINAMATH_GPT_trigonometric_expression_equals_one_l1533_153315

theorem trigonometric_expression_equals_one :
  let cos30 := Real.sqrt 3 / 2
  let sin60 := Real.sqrt 3 / 2
  let sin30 := 1 / 2
  let cos60 := 1 / 2

  (1 - 1 / cos30) * (1 + 1 / sin60) *
  (1 - 1 / sin30) * (1 + 1 / cos60) = 1 :=
by
  let cos30 := Real.sqrt 3 / 2
  let sin60 := Real.sqrt 3 / 2
  let sin30 := 1 / 2
  let cos60 := 1 / 2
  sorry

end NUMINAMATH_GPT_trigonometric_expression_equals_one_l1533_153315


namespace NUMINAMATH_GPT_line_equation_min_intercepts_l1533_153336

theorem line_equation_min_intercepts (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : 1 / a + 4 / b = 1) : 2 * 1 + 4 - 6 = 0 ↔ (a = 3 ∧ b = 6) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_min_intercepts_l1533_153336


namespace NUMINAMATH_GPT_greatest_divisor_of_product_of_four_consecutive_integers_l1533_153383

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end NUMINAMATH_GPT_greatest_divisor_of_product_of_four_consecutive_integers_l1533_153383


namespace NUMINAMATH_GPT_twenty_kopeck_greater_than_ten_kopeck_l1533_153343

-- Definitions of the conditions
variables (x y z : ℕ)
axiom total_coins : x + y + z = 30 
axiom total_value : 10 * x + 15 * y + 20 * z = 500 

-- The proof statement
theorem twenty_kopeck_greater_than_ten_kopeck : z > x :=
sorry

end NUMINAMATH_GPT_twenty_kopeck_greater_than_ten_kopeck_l1533_153343


namespace NUMINAMATH_GPT_math_more_than_reading_homework_l1533_153334

-- Definitions based on given conditions
def M : Nat := 9  -- Math homework pages
def R : Nat := 2  -- Reading homework pages

theorem math_more_than_reading_homework :
  M - R = 7 :=
by
  -- Proof would go here, showing that 9 - 2 indeed equals 7
  sorry

end NUMINAMATH_GPT_math_more_than_reading_homework_l1533_153334


namespace NUMINAMATH_GPT_Nadia_distance_is_18_l1533_153393

-- Variables and conditions
variables (x : ℕ)

-- Definitions based on conditions
def Hannah_walked (x : ℕ) : ℕ := x
def Nadia_walked (x : ℕ) : ℕ := 2 * x
def total_distance (x : ℕ) : ℕ := Hannah_walked x + Nadia_walked x

-- The proof statement
theorem Nadia_distance_is_18 (h : total_distance x = 27) : Nadia_walked x = 18 :=
by
  sorry

end NUMINAMATH_GPT_Nadia_distance_is_18_l1533_153393


namespace NUMINAMATH_GPT_whisky_replacement_l1533_153350

variable (x : ℝ) -- Original quantity of whisky in the jar
variable (y : ℝ) -- Quantity of whisky replaced

-- Condition: A jar full of whisky contains 40% alcohol
-- Condition: After replacement, the percentage of alcohol is 24%
theorem whisky_replacement (h : 0 < x) : 
  0.40 * x - 0.40 * y + 0.19 * y = 0.24 * x → y = (16 / 21) * x :=
by
  intro h_eq
  -- Sorry for the proof
  sorry

end NUMINAMATH_GPT_whisky_replacement_l1533_153350


namespace NUMINAMATH_GPT_geometric_sequence_sum_ratio_l1533_153335

theorem geometric_sequence_sum_ratio 
  (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a (n+1) = a 0 * q ^ n)
  (h2 : ∀ n, S n = (a 0 * (q ^ n - 1)) / (q - 1))
  (h3 : 6 * a 3 = a 0 * q ^ 5 - a 0 * q ^ 4) :
  S 4 / S 2 = 10 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_ratio_l1533_153335


namespace NUMINAMATH_GPT_digit_for_divisibility_by_45_l1533_153362

theorem digit_for_divisibility_by_45 (n : ℕ) (h₀ : n < 10)
  (h₁ : 5 ∣ (5 + 10 * (7 + 4 * (1 + 5 * (8 + n))))) 
  (h₂ : 9 ∣ (5 + 7 + 4 + n + 5 + 8)) : 
  n = 7 :=
by { sorry }

end NUMINAMATH_GPT_digit_for_divisibility_by_45_l1533_153362


namespace NUMINAMATH_GPT_valid_parameterizations_l1533_153313

theorem valid_parameterizations :
  (∀ t : ℝ, ∃ x y : ℝ, (x = 0 + 4 * t) ∧ (y = -4 + 8 * t) ∧ (y = 2 * x - 4)) ∧
  (∀ t : ℝ, ∃ x y : ℝ, (x = 3 + 1 * t) ∧ (y = 2 + 2 * t) ∧ (y = 2 * x - 4)) ∧
  (∀ t : ℝ, ∃ x y : ℝ, (x = -1 + 2 * t) ∧ (y = -6 + 4 * t) ∧ (y = 2 * x - 4)) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_valid_parameterizations_l1533_153313


namespace NUMINAMATH_GPT_min_value_a_plus_3b_l1533_153316

theorem min_value_a_plus_3b (a b : ℝ) (h_positive : 0 < a ∧ 0 < b)
  (h_condition : (1 / (a + 3) + 1 / (b + 3) = 1 / 4)) :
  a + 3 * b ≥ 4 + 8 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_min_value_a_plus_3b_l1533_153316


namespace NUMINAMATH_GPT_f_at_7_l1533_153382

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_property : ∀ x, f (x + 4) = f x
axiom specific_interval_definition : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem f_at_7 : f 7 = -2 := 
  by sorry

end NUMINAMATH_GPT_f_at_7_l1533_153382


namespace NUMINAMATH_GPT_polygon_diagonals_l1533_153340

-- Definitions of the conditions
def sum_of_angles (n : ℕ) : ℝ := (n - 2) * 180 + 360

def num_diagonals (n : ℕ) : ℤ := n * (n - 3) / 2

-- Theorem statement
theorem polygon_diagonals (n : ℕ) (h : sum_of_angles n = 2160) : num_diagonals n = 54 :=
sorry

end NUMINAMATH_GPT_polygon_diagonals_l1533_153340


namespace NUMINAMATH_GPT_problem_a_b_sum_l1533_153341

-- Define the operation
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- Given conditions
variable (a b : ℝ)

-- Theorem statement: Prove that a + b = 4
theorem problem_a_b_sum :
  (∀ x, ((2 < x) ∧ (x < 3)) ↔ ((x - a) * (x - b - 1) < 0)) → a + b = 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_a_b_sum_l1533_153341


namespace NUMINAMATH_GPT_abs_reciprocal_inequality_l1533_153372

theorem abs_reciprocal_inequality (a b : ℝ) (h : 1 / |a| < 1 / |b|) : |a| > |b| :=
sorry

end NUMINAMATH_GPT_abs_reciprocal_inequality_l1533_153372


namespace NUMINAMATH_GPT_sale_price_is_207_l1533_153323

-- Definitions for the conditions given
def price_at_store_P : ℝ := 200
def regular_price_at_store_Q (price_P : ℝ) : ℝ := price_P * 1.15
def sale_price_at_store_Q (regular_price_Q : ℝ) : ℝ := regular_price_Q * 0.90

-- Goal: Prove the sale price of the bicycle at Store Q is 207
theorem sale_price_is_207 : sale_price_at_store_Q (regular_price_at_store_Q price_at_store_P) = 207 :=
by
  sorry

end NUMINAMATH_GPT_sale_price_is_207_l1533_153323


namespace NUMINAMATH_GPT_intersect_sets_l1533_153329

def set_M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def set_N : Set ℝ := {x | abs x < 2}

theorem intersect_sets :
  (set_M ∩ set_N) = {x | -1 ≤ x ∧ x < 2} :=
sorry

end NUMINAMATH_GPT_intersect_sets_l1533_153329


namespace NUMINAMATH_GPT_ribbon_length_difference_l1533_153397

theorem ribbon_length_difference (S : ℝ) : 
  let Seojun_ribbon := S 
  let Siwon_ribbon := S + 8.8 
  let Seojun_new := Seojun_ribbon - 4.3
  let Siwon_new := Siwon_ribbon + 4.3 
  Siwon_new - Seojun_new = 17.4 :=
by
  -- Definition of original ribbon lengths
  let Seojun_ribbon := S
  let Siwon_ribbon := S + 8.8
  -- Seojun cuts and gives 4.3 meters to Siwon
  let Seojun_new := Seojun_ribbon - 4.3
  let Siwon_new := Siwon_ribbon + 4.3
  -- Compute the difference
  have h1 : Siwon_new - Seojun_new = (S + 8.8 + 4.3) - (S - 4.3) := by sorry
  -- Prove the final answer
  have h2 : Siwon_new - Seojun_new = 17.4 := by sorry

  exact h2

end NUMINAMATH_GPT_ribbon_length_difference_l1533_153397


namespace NUMINAMATH_GPT_no_such_xy_between_988_and_1991_l1533_153374

theorem no_such_xy_between_988_and_1991 :
  ¬ ∃ (x y : ℕ), 988 ≤ x ∧ x < y ∧ y ≤ 1991 ∧ 
  (∃ a b : ℕ, xy = x * y ∧ (xy + x = a^2 ∧ xy + y = b^2)) :=
by
  sorry

end NUMINAMATH_GPT_no_such_xy_between_988_and_1991_l1533_153374


namespace NUMINAMATH_GPT_number_of_negative_x_values_l1533_153386

theorem number_of_negative_x_values : 
  (∃ (n : ℕ), ∀ (x : ℤ), x = n^2 - 196 ∧ x < 0) ∧ (n ≤ 13) :=
by 
  -- To formalize our problem we need quantifiers, inequalities and integer properties.
  sorry

end NUMINAMATH_GPT_number_of_negative_x_values_l1533_153386


namespace NUMINAMATH_GPT_parallel_vectors_x_eq_one_l1533_153317

/-- Given vectors a = (2x + 1, 3) and b = (2 - x, 1), prove that if they 
are parallel, then x = 1. -/
theorem parallel_vectors_x_eq_one (x : ℝ) :
  (∃ k : ℝ, (2 * x + 1) = k * (2 - x) ∧ 3 = k * 1) → x = 1 :=
by 
  sorry

end NUMINAMATH_GPT_parallel_vectors_x_eq_one_l1533_153317


namespace NUMINAMATH_GPT_part_one_part_two_l1533_153345

-- Define the function f
def f (x : ℝ) : ℝ := abs (x - 2) + abs (x + 1)

-- Define the inequality condition
def inequality_condition (x : ℝ) : Prop := f x ≥ 4 - x

-- Problem set (I)
theorem part_one (x : ℝ) : inequality_condition x ↔ (x ≤ -3 ∨ x ≥ 1) :=
sorry

-- Define range conditions for a and b
def range_condition (a b : ℝ) : Prop := a ≥ 3 ∧ b ≥ 3

-- Problem set (II)
theorem part_two (a b : ℝ) (h : range_condition a b) : 2 * (a + b) < a * b + 4 :=
sorry

end NUMINAMATH_GPT_part_one_part_two_l1533_153345


namespace NUMINAMATH_GPT_find_initial_population_l1533_153385

theorem find_initial_population
  (birth_rate : ℕ)
  (death_rate : ℕ)
  (net_growth_rate_percent : ℝ)
  (net_growth_rate_per_person : ℕ)
  (h1 : birth_rate = 32)
  (h2 : death_rate = 11)
  (h3 : net_growth_rate_percent = 2.1)
  (h4 : net_growth_rate_per_person = birth_rate - death_rate)
  (h5 : (net_growth_rate_per_person : ℝ) / 100 = net_growth_rate_percent / 100) :
  P = 1000 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_population_l1533_153385


namespace NUMINAMATH_GPT_find_number_subtracted_l1533_153395

theorem find_number_subtracted (x : ℕ) (h : 88 - x = 54) : x = 34 := by
  sorry

end NUMINAMATH_GPT_find_number_subtracted_l1533_153395


namespace NUMINAMATH_GPT_at_least_one_prob_better_option_l1533_153353

-- Definitions based on the conditions in a)

def player_A_prelim := 1 / 2
def player_B_prelim := 1 / 3
def player_C_prelim := 1 / 2

def final_round := 1 / 3

def prelim_prob_A := player_A_prelim * final_round
def prelim_prob_B := player_B_prelim * final_round
def prelim_prob_C := player_C_prelim * final_round

def prob_none := (1 - prelim_prob_A) * (1 - prelim_prob_B) * (1 - prelim_prob_C)

def prob_at_least_one := 1 - prob_none

-- Question 1 statement

theorem at_least_one_prob :
  prob_at_least_one = 31 / 81 :=
sorry

-- Definitions based on the reward options in the conditions

def option_1_lottery_prob := 1 / 3
def option_1_reward := 600
def option_1_expected_value := 600 * 3 * (1 / 3)

def option_2_prelim_reward := 100
def option_2_final_reward := 400

-- Expected values calculation for Option 2

def option_2_expected_value :=
  (300 * (1 / 6) + 600 * (5 / 12) + 900 * (1 / 3) + 1200 * (1 / 12))

-- Question 2 statement

theorem better_option :
  option_1_expected_value < option_2_expected_value :=
sorry

end NUMINAMATH_GPT_at_least_one_prob_better_option_l1533_153353


namespace NUMINAMATH_GPT_variable_cost_per_book_l1533_153325

theorem variable_cost_per_book
  (F : ℝ) (S : ℝ) (N : ℕ) (V : ℝ)
  (fixed_cost : F = 56430) 
  (selling_price_per_book : S = 21.75) 
  (num_books : N = 4180) 
  (production_eq_sales : S * N = F + V * N) :
  V = 8.25 :=
by sorry

end NUMINAMATH_GPT_variable_cost_per_book_l1533_153325


namespace NUMINAMATH_GPT_words_to_score_A_l1533_153321

-- Define the total number of words
def total_words : ℕ := 600

-- Define the target percentage
def target_percentage : ℚ := 90 / 100

-- Define the minimum number of words to learn
def min_words_to_learn : ℕ := 540

-- Define the condition for scoring at least 90%
def meets_requirement (learned_words : ℕ) : Prop :=
  learned_words / total_words ≥ target_percentage

-- The goal is to prove that learning 540 words meets the requirement
theorem words_to_score_A : meets_requirement min_words_to_learn :=
by
  sorry

end NUMINAMATH_GPT_words_to_score_A_l1533_153321


namespace NUMINAMATH_GPT_num_int_values_x_l1533_153339

theorem num_int_values_x (x : ℕ) :
  (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) → ∃ n : ℕ, n = 3 :=
by
  sorry

end NUMINAMATH_GPT_num_int_values_x_l1533_153339


namespace NUMINAMATH_GPT_calculate_expression_value_l1533_153324

theorem calculate_expression_value :
  5 * 7 + 6 * 9 + 13 * 2 + 4 * 6 = 139 :=
by
  -- proof can be added here
  sorry

end NUMINAMATH_GPT_calculate_expression_value_l1533_153324


namespace NUMINAMATH_GPT_triangle_formation_ways_l1533_153338

-- Given conditions
def parallel_tracks : Prop := true -- The tracks are parallel, implicit condition not affecting calculation
def first_track_checkpoints := 6
def second_track_checkpoints := 10

-- The proof problem
theorem triangle_formation_ways : 
  (first_track_checkpoints * Nat.choose second_track_checkpoints 2) = 270 := by
  sorry

end NUMINAMATH_GPT_triangle_formation_ways_l1533_153338


namespace NUMINAMATH_GPT_boxes_of_apples_l1533_153376

theorem boxes_of_apples (apples_per_crate crates_delivered rotten_apples apples_per_box : ℕ) 
       (h1 : apples_per_crate = 42) 
       (h2 : crates_delivered = 12) 
       (h3 : rotten_apples = 4) 
       (h4 : apples_per_box = 10) : 
       crates_delivered * apples_per_crate - rotten_apples = 500 ∧
       (crates_delivered * apples_per_crate - rotten_apples) / apples_per_box = 50 := by
  sorry

end NUMINAMATH_GPT_boxes_of_apples_l1533_153376


namespace NUMINAMATH_GPT_albums_not_in_both_l1533_153380

-- Definitions representing the problem conditions
def andrew_albums : ℕ := 23
def common_albums : ℕ := 11
def john_unique_albums : ℕ := 8

-- Proof statement (not the actual proof)
theorem albums_not_in_both : 
  (andrew_albums - common_albums) + john_unique_albums = 20 :=
by
  sorry

end NUMINAMATH_GPT_albums_not_in_both_l1533_153380


namespace NUMINAMATH_GPT_num_monomials_degree_7_l1533_153371

theorem num_monomials_degree_7 : 
  ∃ (count : Nat), 
    (∀ (a b c : ℕ), a + b + c = 7 → (1 : ℕ) = 1) ∧ 
    count = 15 := 
sorry

end NUMINAMATH_GPT_num_monomials_degree_7_l1533_153371


namespace NUMINAMATH_GPT_minimize_distance_l1533_153333

theorem minimize_distance
  (a b c d : ℝ)
  (h1 : (a + 3 * Real.log a) / b = 1)
  (h2 : (d - 3) / (2 * c) = 1) :
  (a - c)^2 + (b - d)^2 = (9 / 5) * (Real.log (Real.exp 1 / 3))^2 :=
by sorry

end NUMINAMATH_GPT_minimize_distance_l1533_153333


namespace NUMINAMATH_GPT_solution_set_system_of_inequalities_l1533_153392

theorem solution_set_system_of_inequalities :
  { x : ℝ | (2 - x) * (2 * x + 4) ≥ 0 ∧ -3 * x^2 + 2 * x + 1 < 0 } = 
  { x : ℝ | -2 ≤ x ∧ x < -1/3 ∨ 1 < x ∧ x ≤ 2 } := 
by
  sorry

end NUMINAMATH_GPT_solution_set_system_of_inequalities_l1533_153392


namespace NUMINAMATH_GPT_ways_to_divide_week_l1533_153302

def week_seconds : ℕ := 604800

theorem ways_to_divide_week (n m : ℕ) (h1 : 0 < n) (h2 : 0 < m) (h3 : week_seconds = n * m) :
  (∃ (pairs : ℕ), pairs = 336) :=
sorry

end NUMINAMATH_GPT_ways_to_divide_week_l1533_153302


namespace NUMINAMATH_GPT_pies_with_no_ingredients_l1533_153301

theorem pies_with_no_ingredients (total_pies : ℕ)
  (pies_with_chocolate : ℕ)
  (pies_with_blueberries : ℕ)
  (pies_with_vanilla : ℕ)
  (pies_with_almonds : ℕ)
  (H_total : total_pies = 60)
  (H_chocolate : pies_with_chocolate = 1 / 3 * total_pies)
  (H_blueberries : pies_with_blueberries = 3 / 4 * total_pies)
  (H_vanilla : pies_with_vanilla = 2 / 5 * total_pies)
  (H_almonds : pies_with_almonds = 1 / 10 * total_pies) :
  ∃ (pies_without_ingredients : ℕ), pies_without_ingredients = 15 :=
by
  sorry

end NUMINAMATH_GPT_pies_with_no_ingredients_l1533_153301


namespace NUMINAMATH_GPT_tetrahedron_cross_section_area_l1533_153342

theorem tetrahedron_cross_section_area (a : ℝ) : 
  ∃ (S : ℝ), 
    let AB := a; 
    let AC := a;
    let AD := a;
    S = (3 * a^2) / 8 
    := sorry

end NUMINAMATH_GPT_tetrahedron_cross_section_area_l1533_153342


namespace NUMINAMATH_GPT_ball_picking_problem_proof_l1533_153391

-- Define the conditions
def red_balls : ℕ := 8
def white_balls : ℕ := 7

-- Define the questions
def num_ways_to_pick_one_ball : ℕ :=
  red_balls + white_balls

def num_ways_to_pick_two_different_color_balls : ℕ :=
  red_balls * white_balls

-- Define the correct answers
def correct_answer_to_pick_one_ball : ℕ := 15
def correct_answer_to_pick_two_different_color_balls : ℕ := 56

-- State the theorem to be proved
theorem ball_picking_problem_proof :
  (num_ways_to_pick_one_ball = correct_answer_to_pick_one_ball) ∧
  (num_ways_to_pick_two_different_color_balls = correct_answer_to_pick_two_different_color_balls) :=
by
  sorry

end NUMINAMATH_GPT_ball_picking_problem_proof_l1533_153391


namespace NUMINAMATH_GPT_average_production_n_days_l1533_153367

theorem average_production_n_days (n : ℕ) (P : ℕ) 
  (hP : P = 80 * n)
  (h_new_avg : (P + 220) / (n + 1) = 95) : 
  n = 8 := 
by
  sorry -- Proof of the theorem

end NUMINAMATH_GPT_average_production_n_days_l1533_153367


namespace NUMINAMATH_GPT_triangle_area_proof_l1533_153319

-- Conditions
variables (P r : ℝ) (semi_perimeter : ℝ)
-- The perimeter of the triangle is 40 cm
def perimeter_condition : Prop := P = 40
-- The inradius of the triangle is 2.5 cm
def inradius_condition : Prop := r = 2.5
-- The semi-perimeter is half of the perimeter
def semi_perimeter_def : Prop := semi_perimeter = P / 2

-- The area of the triangle
def area_of_triangle : ℝ := r * semi_perimeter

-- Proof Problem
theorem triangle_area_proof (hP : perimeter_condition P) (hr : inradius_condition r) (hsemi : semi_perimeter_def P semi_perimeter) :
  area_of_triangle r semi_perimeter = 50 :=
  sorry

end NUMINAMATH_GPT_triangle_area_proof_l1533_153319


namespace NUMINAMATH_GPT_symmetric_point_x_axis_l1533_153347

theorem symmetric_point_x_axis (P Q : ℝ × ℝ) (hP : P = (-1, 2)) (hQ : Q = (P.1, -P.2)) : Q = (-1, -2) :=
sorry

end NUMINAMATH_GPT_symmetric_point_x_axis_l1533_153347


namespace NUMINAMATH_GPT_base5_number_l1533_153308

/-- A base-5 number only contains the digits 0, 1, 2, 3, and 4.
    Given the number 21340, we need to prove that it could possibly be a base-5 number. -/
theorem base5_number (n : ℕ) (h : n = 21340) : 
  ∀ d ∈ [2, 1, 3, 4, 0], d < 5 :=
by sorry

end NUMINAMATH_GPT_base5_number_l1533_153308


namespace NUMINAMATH_GPT_yellow_tint_percent_l1533_153363

theorem yellow_tint_percent (total_volume: ℕ) (initial_yellow_percent: ℚ) (yellow_added: ℕ) (answer: ℚ) 
  (h_initial_total: total_volume = 20) 
  (h_initial_yellow: initial_yellow_percent = 0.50) 
  (h_yellow_added: yellow_added = 6) 
  (h_answer: answer = 61.5): 
  (yellow_added + initial_yellow_percent * total_volume) / (total_volume + yellow_added) * 100 = answer := 
by 
  sorry

end NUMINAMATH_GPT_yellow_tint_percent_l1533_153363


namespace NUMINAMATH_GPT_evaluate_expression_l1533_153326

theorem evaluate_expression (b : ℕ) (h : b = 5) : b^3 * b^4 * 2 = 156250 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1533_153326


namespace NUMINAMATH_GPT_calorie_limit_l1533_153364

variable (breakfastCalories lunchCalories dinnerCalories extraCalories : ℕ)
variable (plannedCalories : ℕ)

-- Given conditions
axiom breakfast_calories : breakfastCalories = 400
axiom lunch_calories : lunchCalories = 900
axiom dinner_calories : dinnerCalories = 1100
axiom extra_calories : extraCalories = 600

-- To Prove
theorem calorie_limit (h : plannedCalories = (breakfastCalories + lunchCalories + dinnerCalories - extraCalories)) :
  plannedCalories = 1800 := by sorry

end NUMINAMATH_GPT_calorie_limit_l1533_153364


namespace NUMINAMATH_GPT_greatest_y_value_l1533_153354

theorem greatest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -6) : y ≤ 3 :=
sorry

end NUMINAMATH_GPT_greatest_y_value_l1533_153354


namespace NUMINAMATH_GPT_speed_of_man_upstream_l1533_153304

-- Conditions stated as definitions 
def V_m : ℝ := 33 -- Speed of the man in still water
def V_downstream : ℝ := 40 -- Speed of the man rowing downstream

-- Required proof problem
theorem speed_of_man_upstream : V_m - (V_downstream - V_m) = 26 := 
by
  -- the following sorry is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_speed_of_man_upstream_l1533_153304


namespace NUMINAMATH_GPT_proj_b_l1533_153320

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let (ux, uy) := u
  let (vx, vy) := v
  let factor := (ux * vx + uy * vy) / (vx * vx + vy * vy)
  (factor * vx, factor * vy)

theorem proj_b (a b v : ℝ × ℝ) (h_ortho : a.1 * b.1 + a.2 * b.2 = 0)
  (h_proj_a : proj v a = (1, 2)) : proj v b = (3, -4) :=
by
  sorry

end NUMINAMATH_GPT_proj_b_l1533_153320


namespace NUMINAMATH_GPT_required_earnings_correct_l1533_153327

-- Definitions of the given conditions
def retail_price : ℝ := 600
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05
def amount_saved : ℝ := 120
def amount_given_by_mother : ℝ := 250
def additional_costs : ℝ := 50

-- Required amount Maria must earn
def required_earnings : ℝ := 247

-- Lean 4 theorem statement
theorem required_earnings_correct :
  let discount_amount := discount_rate * retail_price
  let discounted_price := retail_price - discount_amount
  let sales_tax_amount := sales_tax_rate * discounted_price
  let total_bike_cost := discounted_price + sales_tax_amount
  let total_cost := total_bike_cost + additional_costs
  let total_have := amount_saved + amount_given_by_mother
  required_earnings = total_cost - total_have :=
by
  sorry

end NUMINAMATH_GPT_required_earnings_correct_l1533_153327


namespace NUMINAMATH_GPT_June_sweets_count_l1533_153348

variable (A M J : ℕ)

-- condition: May has three-quarters of the number of sweets that June has
def May_sweets := M = (3/4) * J

-- condition: April has two-thirds of the number of sweets that May has
def April_sweets := A = (2/3) * M

-- condition: April, May, and June have 90 sweets between them
def Total_sweets := A + M + J = 90

-- proof problem: How many sweets does June have?
theorem June_sweets_count : 
  May_sweets M J ∧ April_sweets A M ∧ Total_sweets A M J → J = 40 :=
by
  sorry

end NUMINAMATH_GPT_June_sweets_count_l1533_153348
