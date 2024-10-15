import Mathlib

namespace NUMINAMATH_GPT_sequence_expression_l773_77342

theorem sequence_expression (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) - 2 * a n = 2^n) :
  ∀ n, a n = n * 2^(n - 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_expression_l773_77342


namespace NUMINAMATH_GPT_base6_to_base10_l773_77305

theorem base6_to_base10 (c d : ℕ) (h1 : 524 = 2 * (10 * c + d)) (hc : c < 10) (hd : d < 10) :
  (c * d : ℚ) / 12 = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_base6_to_base10_l773_77305


namespace NUMINAMATH_GPT_no_perfect_square_E_l773_77382

noncomputable def E (x : ℝ) : ℤ :=
  round x

theorem no_perfect_square_E (n : ℕ) (h : n > 0) : ¬ (∃ k : ℕ, E (n + Real.sqrt n) = k * k) :=
  sorry

end NUMINAMATH_GPT_no_perfect_square_E_l773_77382


namespace NUMINAMATH_GPT_rooster_weight_l773_77328

variable (W : ℝ)  -- The weight of the first rooster

theorem rooster_weight (h1 : 0.50 * W + 0.50 * 40 = 35) : W = 30 :=
by
  sorry

end NUMINAMATH_GPT_rooster_weight_l773_77328


namespace NUMINAMATH_GPT_minimum_value_correct_l773_77317

noncomputable def minimum_value (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_eq : x^2 + y^2 + z^2 = 1) : ℝ :=
  (z + 1)^2 / (2 * x * y * z)

theorem minimum_value_correct {x y z : ℝ}
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_eq : x^2 + y^2 + z^2 = 1) :
  minimum_value x y z h_pos h_eq = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_correct_l773_77317


namespace NUMINAMATH_GPT_first_number_is_twenty_l773_77340

theorem first_number_is_twenty (x : ℕ) : 
  (x + 40 + 60) / 3 = ((10 + 70 + 16) / 3) + 8 → x = 20 := 
by 
  sorry

end NUMINAMATH_GPT_first_number_is_twenty_l773_77340


namespace NUMINAMATH_GPT_school_enrollment_l773_77347

theorem school_enrollment
  (X Y : ℝ)
  (h1 : X + Y = 4000)
  (h2 : 1.07 * X > X)
  (h3 : 1.03 * Y > Y)
  (h4 : 0.07 * X - 0.03 * Y = 40) :
  Y = 2400 :=
by
  -- problem reduction
  sorry

end NUMINAMATH_GPT_school_enrollment_l773_77347


namespace NUMINAMATH_GPT_distance_between_trains_l773_77310

theorem distance_between_trains
  (v1 v2 : ℕ) (d_diff : ℕ)
  (h_v1 : v1 = 50) (h_v2 : v2 = 60) (h_d_diff : d_diff = 100) :
  ∃ d, d = 1100 :=
by
  sorry

-- Explanation:
-- v1 is the speed of the first train.
-- v2 is the speed of the second train.
-- d_diff is the difference in the distances traveled by the two trains at the time of meeting.
-- h_v1 states that the speed of the first train is 50 kmph.
-- h_v2 states that the speed of the second train is 60 kmph.
-- h_d_diff states that the second train travels 100 km more than the first train.
-- The existential statement asserts that there exists a distance d such that d equals 1100 km.

end NUMINAMATH_GPT_distance_between_trains_l773_77310


namespace NUMINAMATH_GPT_b_20_value_l773_77309

-- Definitions based on conditions
def a (n : ℕ) : ℕ := 2 * n - 1

def b (n : ℕ) : ℕ := a n  -- Given that \( b_n = a_n \)

-- The theorem stating that \( b_{20} = 39 \)
theorem b_20_value : b 20 = 39 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_b_20_value_l773_77309


namespace NUMINAMATH_GPT_product_base9_conversion_l773_77373

noncomputable def base_9_to_base_10 (n : ℕ) : ℕ :=
match n with
| 237 => 2 * 9^2 + 3 * 9^1 + 7
| 17 => 9 + 7
| _ => 0

noncomputable def base_10_to_base_9 (n : ℕ) : ℕ :=
match n with
-- Step of conversion from example: 3136 => 4*9^3 + 2*9^2 + 6*9^1 + 4*9^0
| 3136 => 4 * 1000 + 2 * 100 + 6 * 10 + 4 -- representing 4264 in base 9
| _ => 0

theorem product_base9_conversion :
  base_10_to_base_9 ((base_9_to_base_10 237) * (base_9_to_base_10 17)) = 4264 := by
  sorry

end NUMINAMATH_GPT_product_base9_conversion_l773_77373


namespace NUMINAMATH_GPT_total_distance_traveled_eq_l773_77323

-- Define the conditions as speeds and times for each segment of Jeff's trip.
def speed1 : ℝ := 80
def time1 : ℝ := 6
def speed2 : ℝ := 60
def time2 : ℝ := 4
def speed3 : ℝ := 40
def time3 : ℝ := 2

-- Define the distance function given speed and time.
def distance (speed time : ℝ) : ℝ := speed * time

-- Calculate the individual distances for each segment.
def distance1 : ℝ := distance speed1 time1
def distance2 : ℝ := distance speed2 time2
def distance3 : ℝ := distance speed3 time3

-- State the proof problem to show that the total distance is 800 miles.
theorem total_distance_traveled_eq : distance1 + distance2 + distance3 = 800 :=
by
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_total_distance_traveled_eq_l773_77323


namespace NUMINAMATH_GPT_find_n_l773_77379

theorem find_n 
  (molecular_weight : ℕ)
  (atomic_weight_Al : ℕ)
  (weight_OH : ℕ)
  (n : ℕ) 
  (h₀ : molecular_weight = 78)
  (h₁ : atomic_weight_Al = 27) 
  (h₂ : weight_OH = 17)
  (h₃ : molecular_weight = atomic_weight_Al + n * weight_OH) : 
  n = 3 := 
by 
  -- the proof is omitted
  sorry

end NUMINAMATH_GPT_find_n_l773_77379


namespace NUMINAMATH_GPT_power_of_exponents_l773_77336

theorem power_of_exponents (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 := by
  sorry

end NUMINAMATH_GPT_power_of_exponents_l773_77336


namespace NUMINAMATH_GPT_fully_simplify_expression_l773_77388

theorem fully_simplify_expression :
  (3 + 4 + 5 + 6) / 2 + (3 * 6 + 9) / 3 = 18 :=
by
  sorry

end NUMINAMATH_GPT_fully_simplify_expression_l773_77388


namespace NUMINAMATH_GPT_triangle_angle_l773_77326

-- Definitions of the conditions and theorem
variables {a b c : ℝ}
variables {A B C : ℝ}

theorem triangle_angle (h : b^2 + c^2 - a^2 = bc)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hA : 0 < A) (hA_max : A < π) :
  A = π / 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_l773_77326


namespace NUMINAMATH_GPT_fourth_ball_black_probability_l773_77330

noncomputable def prob_fourth_is_black : Prop :=
  let total_balls := 8
  let black_balls := 4
  let prob_black := black_balls / total_balls
  prob_black = 1 / 2

theorem fourth_ball_black_probability :
  prob_fourth_is_black :=
sorry

end NUMINAMATH_GPT_fourth_ball_black_probability_l773_77330


namespace NUMINAMATH_GPT_original_number_is_24_l773_77370

theorem original_number_is_24 (N : ℕ) 
  (h1 : (N + 1) % 25 = 0)
  (h2 : 1 = 1) : N = 24 := 
sorry

end NUMINAMATH_GPT_original_number_is_24_l773_77370


namespace NUMINAMATH_GPT_multiply_exponents_l773_77376

theorem multiply_exponents (a : ℝ) : 2 * a^3 * 3 * a^2 = 6 * a^5 := by
  sorry

end NUMINAMATH_GPT_multiply_exponents_l773_77376


namespace NUMINAMATH_GPT_value_of_k_l773_77369

theorem value_of_k (x y : ℝ) (t : ℝ) (k : ℝ) : 
  (x + t * y + 8 = 0) ∧ (5 * x - t * y + 4 = 0) ∧ (3 * x - k * y + 1 = 0) → k = 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_k_l773_77369


namespace NUMINAMATH_GPT_fraction_of_odd_products_is_0_25_l773_77301

noncomputable def fraction_of_odd_products : ℝ :=
  let odd_products := 8 * 8
  let total_products := 16 * 16
  (odd_products / total_products : ℝ)

theorem fraction_of_odd_products_is_0_25 :
  fraction_of_odd_products = 0.25 :=
by sorry

end NUMINAMATH_GPT_fraction_of_odd_products_is_0_25_l773_77301


namespace NUMINAMATH_GPT_max_abs_sum_on_ellipse_l773_77378

theorem max_abs_sum_on_ellipse :
  ∀ (x y : ℝ), 4 * x^2 + y^2 = 4 -> |x| + |y| ≤ (3 * Real.sqrt 2) / Real.sqrt 5 :=
by
  intro x y h
  sorry

end NUMINAMATH_GPT_max_abs_sum_on_ellipse_l773_77378


namespace NUMINAMATH_GPT_first_group_correct_l773_77353

/-- Define the total members in the choir --/
def total_members : ℕ := 70

/-- Define members in the second group --/
def second_group_members : ℕ := 30

/-- Define members in the third group --/
def third_group_members : ℕ := 15

/-- Define the number of members in the first group by subtracting second and third groups members from total members --/
def first_group_members : ℕ := total_members - (second_group_members + third_group_members)

/-- Prove that the first group has 25 members --/
theorem first_group_correct : first_group_members = 25 := by
  -- insert the proof steps here
  sorry

end NUMINAMATH_GPT_first_group_correct_l773_77353


namespace NUMINAMATH_GPT_tangent_lines_to_circle_through_point_l773_77399

noncomputable def circle_center : ℝ × ℝ := (1, 2)
noncomputable def circle_radius : ℝ := 2
noncomputable def point_P : ℝ × ℝ := (-1, 5)

theorem tangent_lines_to_circle_through_point :
  ∃ m c : ℝ, (∀ x y : ℝ, (x - 1) ^ 2 + (y - 2) ^ 2 = 4 → (m * x + y + c = 0 → (y = -m * x - c))) ∧
  (m = 5/12 ∧ c = -55/12) ∨ (m = 0 ∧ ∀ x : ℝ, x = -1) :=
sorry

end NUMINAMATH_GPT_tangent_lines_to_circle_through_point_l773_77399


namespace NUMINAMATH_GPT_g_g_g_g_15_eq_3_l773_77365

def g (x : ℕ) : ℕ :=
if x % 3 = 0 then x / 3 else 5 * x + 2

theorem g_g_g_g_15_eq_3 : g (g (g (g 15))) = 3 := 
by
  sorry

end NUMINAMATH_GPT_g_g_g_g_15_eq_3_l773_77365


namespace NUMINAMATH_GPT_proof_problem_l773_77387

-- Definitions of arithmetic and geometric sequences
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + n * d

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (b1 r : ℝ), ∀ n, b n = b1 * r^n

-- Lean statement of the problem
theorem proof_problem 
  (a b : ℕ → ℝ)
  (h_a_arithmetic : is_arithmetic_sequence a)
  (h_b_geometric : is_geometric_sequence b)
  (h_condition : a 1 - (a 7)^2 + a 13 = 0)
  (h_b7_a7 : b 7 = a 7) :
  b 3 * b 11 = 4 :=
sorry

end NUMINAMATH_GPT_proof_problem_l773_77387


namespace NUMINAMATH_GPT_initial_white_cookies_l773_77368

theorem initial_white_cookies (B W : ℕ) 
  (h1 : B = W + 50)
  (h2 : (1 / 2 : ℚ) * B + (1 / 4 : ℚ) * W = 85) :
  W = 80 :=
by
  sorry

end NUMINAMATH_GPT_initial_white_cookies_l773_77368


namespace NUMINAMATH_GPT_ratio_of_areas_of_concentric_circles_l773_77343

theorem ratio_of_areas_of_concentric_circles 
  (C1 C2 : ℝ) (r1 r2 : ℝ)
  (h1 : r1 * C1 = 2 * π * r1)
  (h2 : r2 * C2 = 2 * π * r2)
  (h_c1 : 60 / 360 * C1 = 48 / 360 * C2) :
  (π * r1^2) / (π * r2^2) = 16 / 25 := by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_concentric_circles_l773_77343


namespace NUMINAMATH_GPT_greatest_possible_large_chips_l773_77397

theorem greatest_possible_large_chips (s l : ℕ) (even_prime : ℕ) (h1 : s + l = 100) (h2 : s = l + even_prime) (h3 : even_prime = 2) : l = 49 :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_large_chips_l773_77397


namespace NUMINAMATH_GPT_directrix_of_parabola_l773_77334

theorem directrix_of_parabola (a b c : ℝ) (h_eqn : ∀ x, b = -4 * x^2 + c) : 
  b = 5 → c = 0 → (∃ y, y = 81 / 16) :=
by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l773_77334


namespace NUMINAMATH_GPT_rows_seating_nine_people_l773_77359

theorem rows_seating_nine_people (x y : ℕ) (h : 9 * x + 7 * y = 74) : x = 2 :=
by sorry

end NUMINAMATH_GPT_rows_seating_nine_people_l773_77359


namespace NUMINAMATH_GPT_middle_card_is_five_l773_77304

theorem middle_card_is_five 
    (a b c : ℕ) 
    (h1 : a ≠ b ∧ a ≠ c ∧ b ≠ c) 
    (h2 : a + b + c = 16)
    (h3 : a < b ∧ b < c)
    (casey : ¬(∃ y z, y ≠ z ∧ y + z + a = 16 ∧ a < y ∧ y < z))
    (tracy : ¬(∃ x y, x ≠ y ∧ x + y + c = 16 ∧ x < y ∧ y < c))
    (stacy : ¬(∃ x z, x ≠ z ∧ x + z + b = 16 ∧ x < b ∧ b < z)) 
    : b = 5 :=
sorry

end NUMINAMATH_GPT_middle_card_is_five_l773_77304


namespace NUMINAMATH_GPT_area_triangle_DEF_l773_77398

theorem area_triangle_DEF 
  (DE EL EF : ℝ) (H1 : DE = 15) (H2 : EL = 12) (H3 : EF = 20) 
  (DL : ℝ) (H4 : DE^2 = EL^2 + DL^2) (H5 : DL * EF = DL * 20) :
  1/2 * EF * DL = 90 :=
by
  -- Use the assumptions and conditions to state the theorem.
  sorry

end NUMINAMATH_GPT_area_triangle_DEF_l773_77398


namespace NUMINAMATH_GPT_max_intersection_l773_77352

open Finset

def n (S : Finset α) : ℕ := (2 : ℕ) ^ S.card

theorem max_intersection (A B C : Finset ℕ)
  (h1 : A.card = 2016)
  (h2 : B.card = 2016)
  (h3 : n A + n B + n C = n (A ∪ B ∪ C)) :
  (A ∩ B ∩ C).card ≤ 2015 :=
sorry

end NUMINAMATH_GPT_max_intersection_l773_77352


namespace NUMINAMATH_GPT_arithmetic_mean_of_first_40_consecutive_integers_l773_77372

-- Define the arithmetic sequence with the given conditions
def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of the given arithmetic sequence
def arithmetic_sum (a₁ d n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- Define the arithmetic mean of the first n terms of the given arithmetic sequence
def arithmetic_mean (a₁ d n : ℕ) : ℚ :=
  (arithmetic_sum a₁ d n : ℚ) / n

-- The arithmetic sequence starts at 5, has a common difference of 1, and has 40 terms
theorem arithmetic_mean_of_first_40_consecutive_integers :
  arithmetic_mean 5 1 40 = 24.5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_first_40_consecutive_integers_l773_77372


namespace NUMINAMATH_GPT_english_speaking_students_l773_77355

theorem english_speaking_students (T H B E : ℕ) (hT : T = 40) (hH : H = 30) (hB : B = 10) (h_inclusion_exclusion : T = H + E - B) : E = 20 :=
by
  sorry

end NUMINAMATH_GPT_english_speaking_students_l773_77355


namespace NUMINAMATH_GPT_fraction_of_product_l773_77329

theorem fraction_of_product (c d: ℕ) 
  (h1: 5 * 64 + 4 * 8 + 3 = 355)
  (h2: 2 * (10 * c + d) = 355)
  (h3: c < 10)
  (h4: d < 10):
  (c * d : ℚ) / 12 = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_product_l773_77329


namespace NUMINAMATH_GPT_partA_l773_77389

theorem partA (n : ℕ) : 
  1 < (n + 1 / 2) * Real.log (1 + 1 / n) ∧ (n + 1 / 2) * Real.log (1 + 1 / n) < 1 + 1 / (12 * n * (n + 1)) := 
sorry

end NUMINAMATH_GPT_partA_l773_77389


namespace NUMINAMATH_GPT_mary_cards_left_l773_77390

noncomputable def mary_initial_cards : ℝ := 18.0
noncomputable def cards_to_fred : ℝ := 26.0
noncomputable def cards_bought : ℝ := 40.0
noncomputable def mary_final_cards : ℝ := 32.0

theorem mary_cards_left :
  (mary_initial_cards + cards_bought) - cards_to_fred = mary_final_cards := 
by 
  sorry

end NUMINAMATH_GPT_mary_cards_left_l773_77390


namespace NUMINAMATH_GPT_ellipse_foci_coordinates_l773_77315

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ), (x^2 / 64 + y^2 / 100 = 1) → (x = 0 ∧ (y = 6 ∨ y = -6)) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_foci_coordinates_l773_77315


namespace NUMINAMATH_GPT_maxwell_distance_when_meeting_l773_77316

theorem maxwell_distance_when_meeting 
  (distance_between_homes : ℕ)
  (maxwell_speed : ℕ) 
  (brad_speed : ℕ) 
  (total_distance : ℕ) 
  (h : distance_between_homes = 36) 
  (h1 : maxwell_speed = 2)
  (h2 : brad_speed = 4) 
  (h3 : 6 * (total_distance / 6) = distance_between_homes) :
  total_distance = 12 :=
sorry

end NUMINAMATH_GPT_maxwell_distance_when_meeting_l773_77316


namespace NUMINAMATH_GPT_remainder_when_2x_divided_by_7_l773_77319

theorem remainder_when_2x_divided_by_7 (x y r : ℤ) (h1 : x = 10 * y + 3)
    (h2 : 2 * x = 7 * (3 * y) + r) (h3 : 11 * y - x = 2) : r = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_when_2x_divided_by_7_l773_77319


namespace NUMINAMATH_GPT_quotient_correct_l773_77391

noncomputable def find_quotient (z : ℚ) : ℚ :=
  let dividend := (5 * z ^ 5 - 3 * z ^ 4 + 6 * z ^ 3 - 8 * z ^ 2 + 9 * z - 4)
  let divisor := (4 * z ^ 2 + 5 * z + 3)
  let quotient := ((5 / 4) * z ^ 3 - (47 / 16) * z ^ 2 + (257 / 64) * z - (1547 / 256))
  quotient

theorem quotient_correct (z : ℚ) :
  find_quotient z = ((5 / 4) * z ^ 3 - (47 / 16) * z ^ 2 + (257 / 64) * z - (1547 / 256)) :=
by
  sorry

end NUMINAMATH_GPT_quotient_correct_l773_77391


namespace NUMINAMATH_GPT_find_a_l773_77351

def f (x : ℝ) : ℝ := -x^2 - 2 * x + 3

theorem find_a : ∃ a : ℝ, (a > -1) ∧ (a < 2) ∧ (∀ x : ℝ, a ≤ x ∧ x ≤ 2 → f x ≤ f a) ∧ f a = 15 / 4 :=
by
  exists -1 / 2
  sorry

end NUMINAMATH_GPT_find_a_l773_77351


namespace NUMINAMATH_GPT_solve_for_x_l773_77380

theorem solve_for_x (x y : ℝ) (h1 : 3 * x + y = 75) (h2 : 2 * (3 * x + y) - y = 138) : x = 21 :=
  sorry

end NUMINAMATH_GPT_solve_for_x_l773_77380


namespace NUMINAMATH_GPT_area_of_shaded_region_l773_77348

theorem area_of_shaded_region 
  (ABCD : Type) 
  (BC : ℝ)
  (height : ℝ)
  (BE : ℝ)
  (CF : ℝ)
  (BC_length : BC = 12)
  (height_length : height = 10)
  (BE_length : BE = 5)
  (CF_length : CF = 3) :
  (BC * height - (1 / 2 * BE * height) - (1 / 2 * CF * height)) = 80 :=
by
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l773_77348


namespace NUMINAMATH_GPT_emilia_blueberries_l773_77344

def cartons_needed : Nat := 42
def cartons_strawberries : Nat := 2
def cartons_bought : Nat := 33

def cartons_blueberries (needed : Nat) (strawberries : Nat) (bought : Nat) : Nat :=
  needed - (strawberries + bought)

theorem emilia_blueberries : cartons_blueberries cartons_needed cartons_strawberries cartons_bought = 7 :=
by
  sorry

end NUMINAMATH_GPT_emilia_blueberries_l773_77344


namespace NUMINAMATH_GPT_fixed_point_linear_l773_77349

-- Define the linear function y = kx + k + 2
def linear_function (k x : ℝ) : ℝ := k * x + k + 2

-- Prove that the point (-1, 2) lies on the graph of the function for any k
theorem fixed_point_linear (k : ℝ) : linear_function k (-1) = 2 := by
  sorry

end NUMINAMATH_GPT_fixed_point_linear_l773_77349


namespace NUMINAMATH_GPT_rem_product_eq_l773_77333

theorem rem_product_eq 
  (P Q R k : ℤ) 
  (hk : k > 0) 
  (hPQ : P * Q = R) : 
  ((P % k) * (Q % k)) % k = R % k :=
by
  sorry

end NUMINAMATH_GPT_rem_product_eq_l773_77333


namespace NUMINAMATH_GPT_distance_from_y_axis_l773_77367

theorem distance_from_y_axis (dx dy : ℝ) (h1 : dx = 8) (h2 : dx = (1/2) * dy) : dy = 16 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_y_axis_l773_77367


namespace NUMINAMATH_GPT_remainder_when_divided_by_11_l773_77366

theorem remainder_when_divided_by_11 (N : ℕ)
  (h₁ : N = 5 * 5 + 0) :
  N % 11 = 3 := 
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_11_l773_77366


namespace NUMINAMATH_GPT_solution_for_b_l773_77313

theorem solution_for_b (x y b : ℚ) (h1 : 4 * x + 3 * y = b) (h2 : 3 * x + 4 * y = 3 * b) (hx : x = 3) : b = -21 / 5 := by
  sorry

end NUMINAMATH_GPT_solution_for_b_l773_77313


namespace NUMINAMATH_GPT_relationship_between_abc_l773_77345

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

theorem relationship_between_abc (h1 : 2^a = Real.log (1/a) / Real.log 2)
                                 (h2 : Real.log b / Real.log 2 = 2)
                                 (h3 : c = Real.log 2 + Real.log 3 - Real.log 7) :
  b > a ∧ a > c :=
sorry

end NUMINAMATH_GPT_relationship_between_abc_l773_77345


namespace NUMINAMATH_GPT_find_two_angles_of_scalene_obtuse_triangle_l773_77300

def is_scalene (a b c : ℝ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c
def is_obtuse (a : ℝ) : Prop := a > 90
def is_triangle (a b c : ℝ) : Prop := a + b + c = 180

theorem find_two_angles_of_scalene_obtuse_triangle
  (a b c : ℝ)
  (ha : is_obtuse a) (h_scalene : is_scalene a b c) 
  (h_sum : is_triangle a b c) 
  (ha_val : a = 108)
  (h_half : b = 2 * c) :
  b = 48 ∧ c = 24 :=
by
  sorry

end NUMINAMATH_GPT_find_two_angles_of_scalene_obtuse_triangle_l773_77300


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l773_77346

variable (S : ℕ → ℝ)
variable (a_n : ℕ → ℝ)

theorem arithmetic_sequence_sum (h₁ : S 5 = 8) (h₂ : S 10 = 20) : S 15 = 36 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l773_77346


namespace NUMINAMATH_GPT_polygon_triangle_division_l773_77324

theorem polygon_triangle_division (n k : ℕ) (h₁ : n ≥ 3) (h₂ : k ≥ 1):
  k ≥ n - 2 :=
sorry

end NUMINAMATH_GPT_polygon_triangle_division_l773_77324


namespace NUMINAMATH_GPT_weight_of_B_l773_77306

-- Definitions for the weights
variables (A B C : ℝ)

-- Conditions from the problem
def avg_ABC : Prop := (A + B + C) / 3 = 45
def avg_AB : Prop := (A + B) / 2 = 40
def avg_BC : Prop := (B + C) / 2 = 43

-- The theorem to prove the weight of B
theorem weight_of_B (h1 : avg_ABC A B C) (h2 : avg_AB A B) (h3 : avg_BC B C) : B = 31 :=
sorry

end NUMINAMATH_GPT_weight_of_B_l773_77306


namespace NUMINAMATH_GPT_wilson_total_cost_l773_77384

noncomputable def total_cost_wilson_pays : ℝ :=
let hamburger_price : ℝ := 5
let cola_price : ℝ := 2
let fries_price : ℝ := 3
let sundae_price : ℝ := 4
let nugget_price : ℝ := 1.5
let salad_price : ℝ := 6.25
let hamburger_count : ℕ := 2
let cola_count : ℕ := 3
let nugget_count : ℕ := 4

let total_before_discounts := (hamburger_count * hamburger_price) +
                              (cola_count * cola_price) +
                              fries_price +
                              sundae_price +
                              (nugget_count * nugget_price) +
                              salad_price

let free_nugget_discount := 1 * nugget_price
let total_after_promotion := total_before_discounts - free_nugget_discount
let coupon_discount := 4
let total_after_coupon := total_after_promotion - coupon_discount
let loyalty_discount := 0.10 * total_after_coupon
let total_after_loyalty := total_after_coupon - loyalty_discount

total_after_loyalty

theorem wilson_total_cost : total_cost_wilson_pays = 26.77 := 
by
  sorry

end NUMINAMATH_GPT_wilson_total_cost_l773_77384


namespace NUMINAMATH_GPT_range_of_function_l773_77331

theorem range_of_function :
  ∀ x, -1 ≤ Real.sin x ∧ Real.sin x ≤ 1 → -3 ≤ 2 * Real.sin x - 1 ∧ 2 * Real.sin x - 1 ≤ 1 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_range_of_function_l773_77331


namespace NUMINAMATH_GPT_find_smallest_n_l773_77339

-- defining the geometric sequence and its sum for the given conditions
def a_n (n : ℕ) := 3 * (4 ^ n)

def S_n (n : ℕ) := (a_n n - 1) / (4 - 1) -- simplification step

-- statement of the problem: finding the smallest natural number n such that S_n > 3000
theorem find_smallest_n :
  ∃ n : ℕ, S_n n > 3000 ∧ ∀ m : ℕ, m < n → S_n m ≤ 3000 := by
  sorry

end NUMINAMATH_GPT_find_smallest_n_l773_77339


namespace NUMINAMATH_GPT_rain_difference_l773_77341

variable (R : ℝ) -- Amount of rain in the second hour
variable (r1 : ℝ) -- Amount of rain in the first hour

-- Conditions
axiom h1 : r1 = 5
axiom h2 : R + r1 = 22

-- Theorem to prove
theorem rain_difference (R r1 : ℝ) (h1 : r1 = 5) (h2 : R + r1 = 22) : R - 2 * r1 = 7 := by
  sorry

end NUMINAMATH_GPT_rain_difference_l773_77341


namespace NUMINAMATH_GPT_max_cut_strings_preserving_net_l773_77320

-- Define the conditions of the problem
def volleyball_net_width : ℕ := 50
def volleyball_net_height : ℕ := 600

-- The vertices count is calculated as (width + 1) * (height + 1)
def vertices_count : ℕ := (volleyball_net_width + 1) * (volleyball_net_height + 1)

-- The total edges count is the sum of vertical and horizontal edges
def total_edges_count : ℕ := volleyball_net_width * (volleyball_net_height + 1) + (volleyball_net_width + 1) * volleyball_net_height

-- The edges needed to keep the graph connected (number of vertices - 1)
def edges_in_tree : ℕ := vertices_count - 1

-- The maximum removable edges (total edges - edges needed in tree)
def max_removable_edges : ℕ := total_edges_count - edges_in_tree

-- Define the theorem to prove
theorem max_cut_strings_preserving_net : max_removable_edges = 30000 := by
  sorry

end NUMINAMATH_GPT_max_cut_strings_preserving_net_l773_77320


namespace NUMINAMATH_GPT_jackson_final_grade_l773_77385

def jackson_hours_playing_video_games : ℕ := 9

def ratio_study_to_play : ℚ := 1 / 3

def time_spent_studying (hours_playing : ℕ) (ratio : ℚ) : ℚ := hours_playing * ratio

def points_per_hour_studying : ℕ := 15

def jackson_grade (time_studied : ℚ) (points_per_hour : ℕ) : ℚ := time_studied * points_per_hour

theorem jackson_final_grade :
  jackson_grade
    (time_spent_studying jackson_hours_playing_video_games ratio_study_to_play)
    points_per_hour_studying = 45 :=
by
  sorry

end NUMINAMATH_GPT_jackson_final_grade_l773_77385


namespace NUMINAMATH_GPT_quadratic_equation_general_form_l773_77314

theorem quadratic_equation_general_form :
  ∀ (x : ℝ), 3 * x^2 + 1 = 7 * x ↔ 3 * x^2 - 7 * x + 1 = 0 :=
by
  intro x
  constructor
  · intro h
    sorry
  · intro h
    sorry

end NUMINAMATH_GPT_quadratic_equation_general_form_l773_77314


namespace NUMINAMATH_GPT_max_students_l773_77377

-- Defining the problem's conditions
def cost_bus_rental : ℕ := 100
def max_capacity_students : ℕ := 25
def cost_per_student : ℕ := 10
def teacher_admission_cost : ℕ := 0
def total_budget : ℕ := 350

-- The Lean proof problem
theorem max_students (bus_cost : ℕ) (student_capacity : ℕ) (student_cost : ℕ) (teacher_cost : ℕ) (budget : ℕ) :
  bus_cost = cost_bus_rental → 
  student_capacity = max_capacity_students →
  student_cost = cost_per_student →
  teacher_cost = teacher_admission_cost →
  budget = total_budget →
  (student_capacity ≤ (budget - bus_cost) / student_cost) → 
  ∃ n : ℕ, n = student_capacity ∧ n ≤ (budget - bus_cost) / student_cost :=
by
  intros
  sorry

end NUMINAMATH_GPT_max_students_l773_77377


namespace NUMINAMATH_GPT_minimum_phi_l773_77383

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + (Real.pi / 4))

theorem minimum_phi (φ : ℝ) (hφ : φ > 0) :
  (∃ k : ℤ, φ = (3/8) * Real.pi - (k * Real.pi / 2)) → φ = (3/8) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_minimum_phi_l773_77383


namespace NUMINAMATH_GPT_train_length_l773_77302

noncomputable def train_speed_kmph : ℝ := 72
noncomputable def train_speed_mps : ℝ := 20
noncomputable def crossing_time : ℝ := 20
noncomputable def platform_length : ℝ := 220.032
noncomputable def total_distance : ℝ := train_speed_mps * crossing_time

theorem train_length :
  total_distance - platform_length = 179.968 := by
  sorry

end NUMINAMATH_GPT_train_length_l773_77302


namespace NUMINAMATH_GPT_ones_divisible_by_d_l773_77338

theorem ones_divisible_by_d (d : ℕ) (h1 : ¬ (2 ∣ d)) (h2 : ¬ (5 ∣ d))  : 
  ∃ n, (∃ k : ℕ, n = 10^k - 1) ∧ n % d = 0 := 
sorry

end NUMINAMATH_GPT_ones_divisible_by_d_l773_77338


namespace NUMINAMATH_GPT_problem_statement_l773_77322

noncomputable def S (k : ℕ) : ℚ := sorry

theorem problem_statement (k : ℕ) (a_k : ℚ) :
  S (k - 1) < 10 → S k > 10 → a_k = 6 / 7 :=
sorry

end NUMINAMATH_GPT_problem_statement_l773_77322


namespace NUMINAMATH_GPT_Jan_height_is_42_l773_77394

-- Given conditions
def Cary_height : ℕ := 72
def Bill_height : ℕ := Cary_height / 2
def Jan_height : ℕ := Bill_height + 6

-- Statement to prove
theorem Jan_height_is_42 : Jan_height = 42 := by
  sorry

end NUMINAMATH_GPT_Jan_height_is_42_l773_77394


namespace NUMINAMATH_GPT_michael_peach_pies_l773_77393

/--
Michael ran a bakeshop and had to fill an order for some peach pies, 4 apple pies and 3 blueberry pies.
Each pie recipe called for 3 pounds of fruit each. At the market, produce was on sale for $1.00 per pound for both blueberries and apples.
The peaches each cost $2.00 per pound. Michael spent $51 at the market buying the fruit for his pie order.
Prove that Michael had to make 5 peach pies.
-/
theorem michael_peach_pies :
  let apple_pies := 4
  let blueberry_pies := 3
  let peach_pie_cost_per_pound := 2
  let apple_blueberry_cost_per_pound := 1
  let pounds_per_pie := 3
  let total_spent := 51
  (total_spent - ((apple_pies + blueberry_pies) * pounds_per_pie * apple_blueberry_cost_per_pound)) 
  / (pounds_per_pie * peach_pie_cost_per_pound) = 5 :=
by
  let apple_pies := 4
  let blueberry_pies := 3
  let peach_pie_cost_per_pound := 2
  let apple_blueberry_cost_per_pound := 1
  let pounds_per_pie := 3
  let total_spent := 51
  have H1 : (total_spent - ((apple_pies + blueberry_pies) * pounds_per_pie * apple_blueberry_cost_per_pound)) 
             / (pounds_per_pie * peach_pie_cost_per_pound) = 5 := sorry
  exact H1

end NUMINAMATH_GPT_michael_peach_pies_l773_77393


namespace NUMINAMATH_GPT_nate_age_when_ember_is_14_l773_77303

theorem nate_age_when_ember_is_14 (nate_age : ℕ) (ember_age : ℕ) 
  (h1 : ember_age = nate_age / 2) (h2 : nate_age = 14) :
  ∃ (years_later : ℕ), ember_age + years_later = 14 ∧ nate_age + years_later = 21 :=
by
  -- sorry to skip the proof, adhering to the instructions
  sorry

end NUMINAMATH_GPT_nate_age_when_ember_is_14_l773_77303


namespace NUMINAMATH_GPT_focaccia_cost_l773_77395

theorem focaccia_cost :
  let almond_croissant := 4.50
  let salami_cheese_croissant := 4.50
  let plain_croissant := 3.00
  let latte := 2.50
  let total_spent := 21.00
  let known_costs := almond_croissant + salami_cheese_croissant + plain_croissant + 2 * latte
  let focaccia_cost := total_spent - known_costs
  focaccia_cost = 4.00 := 
by
  sorry

end NUMINAMATH_GPT_focaccia_cost_l773_77395


namespace NUMINAMATH_GPT_part1_part2_l773_77392

-- Definition for f(x)
def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

-- The first proof problem: Solve the inequality f(x) > 0
theorem part1 {x : ℝ} : f x > 0 ↔ x > 1 ∨ x < -5 :=
sorry

-- The second proof problem: Finding the range of m
theorem part2 {m : ℝ} : (∀ x, f x + 3 * |x - 4| ≥ m) → m ≤ 9 :=
sorry

end NUMINAMATH_GPT_part1_part2_l773_77392


namespace NUMINAMATH_GPT_Rahul_batting_average_l773_77396

theorem Rahul_batting_average 
  (A : ℕ) (current_matches : ℕ := 12) (new_matches : ℕ := 13) (scored_today : ℕ := 78) (new_average : ℕ := 54)
  (h1 : (A * current_matches + scored_today) = new_average * new_matches) : A = 52 := 
by
  sorry

end NUMINAMATH_GPT_Rahul_batting_average_l773_77396


namespace NUMINAMATH_GPT_evaluate_g_at_6_l773_77354

def g (x : ℝ) : ℝ := 3 * x^4 - 20 * x^3 + 30 * x^2 - 35 * x - 75

theorem evaluate_g_at_6 : g 6 = 363 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_evaluate_g_at_6_l773_77354


namespace NUMINAMATH_GPT_triangle_inequality_l773_77318

theorem triangle_inequality 
  (a b c : ℝ)
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) : 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := 
sorry

end NUMINAMATH_GPT_triangle_inequality_l773_77318


namespace NUMINAMATH_GPT_square_root_of_9_l773_77361

theorem square_root_of_9 : {x : ℝ // x^2 = 9} = {x : ℝ // x = 3 ∨ x = -3} :=
by
  sorry

end NUMINAMATH_GPT_square_root_of_9_l773_77361


namespace NUMINAMATH_GPT_cos_value_l773_77386

theorem cos_value (α : ℝ) (h : Real.sin (α + π / 6) = 1 / 3) : Real.cos (π / 3 - α) = 1 / 3 :=
  sorry

end NUMINAMATH_GPT_cos_value_l773_77386


namespace NUMINAMATH_GPT_fraction_even_odd_phonenumbers_l773_77356

-- Define a predicate for valid phone numbers
def isValidPhoneNumber (n : Nat) : Prop :=
  1000000 ≤ n ∧ n < 10000000 ∧ (n / 1000000 ≠ 0) ∧ (n / 1000000 ≠ 1)

-- Calculate the total number of valid phone numbers
def totalValidPhoneNumbers : Nat :=
  4 * 10^6

-- Calculate the number of valid phone numbers that begin with an even digit and end with an odd digit
def validEvenOddPhoneNumbers : Nat :=
  4 * (10^5) * 5

-- Determine the fraction of such phone numbers (valid ones and valid even-odd ones)
theorem fraction_even_odd_phonenumbers : 
  (validEvenOddPhoneNumbers) / (totalValidPhoneNumbers) = 1 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_even_odd_phonenumbers_l773_77356


namespace NUMINAMATH_GPT_no_alpha_exists_l773_77350

theorem no_alpha_exists (α : ℝ) (hα1 : 0 < α) (hα2 : α < 1) :
  ¬(∃ (a : ℕ → ℝ), (∀ n : ℕ, 0 < a n) ∧ (∀ n : ℕ, 1 + a (n+1) ≤ a n + (α / n.succ) * a n)) :=
by
  sorry

end NUMINAMATH_GPT_no_alpha_exists_l773_77350


namespace NUMINAMATH_GPT_polygon_side_count_l773_77358

theorem polygon_side_count (n : ℕ) 
    (h : (n - 2) * 180 + 1350 - (n - 2) * 180 = 1350) : n = 9 :=
by
  sorry

end NUMINAMATH_GPT_polygon_side_count_l773_77358


namespace NUMINAMATH_GPT_polynomial_evaluation_l773_77357

theorem polynomial_evaluation :
  (5 * 3^3 - 3 * 3^2 + 7 * 3 - 2 = 127) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_l773_77357


namespace NUMINAMATH_GPT_a5_value_l773_77375

-- Definitions
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n, a (n + 1) = q * a n

def positive_terms (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0

def product_condition (a : ℕ → ℝ) : Prop :=
  ∀ n, a n * a (n + 1) = 2^(2 * n + 1)

-- Theorem statement
theorem a5_value (a : ℕ → ℝ) (h_geo : geometric_sequence a) (h_pos : positive_terms a) (h_prod : product_condition a) : a 5 = 32 :=
sorry

end NUMINAMATH_GPT_a5_value_l773_77375


namespace NUMINAMATH_GPT_div_by_7_l773_77337

theorem div_by_7 (n : ℕ) (h : n ≥ 1) : 7 ∣ (8^n + 6) :=
sorry

end NUMINAMATH_GPT_div_by_7_l773_77337


namespace NUMINAMATH_GPT_find_k_l773_77327

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V) (k : ℝ)

-- Conditions
def not_collinear (a b : V) : Prop := ¬ ∃ (m : ℝ), b = m • a
def collinear (u v : V) : Prop := ∃ (m : ℝ), u = m • v

theorem find_k (h1 : not_collinear a b) (h2 : collinear (2 • a + k • b) (a - b)) : k = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l773_77327


namespace NUMINAMATH_GPT_gcf_75_135_l773_77381

theorem gcf_75_135 : Nat.gcd 75 135 = 15 :=
  by sorry

end NUMINAMATH_GPT_gcf_75_135_l773_77381


namespace NUMINAMATH_GPT_shaded_area_correct_l773_77335

-- Given definitions
def square_side_length : ℝ := 1
def grid_rows : ℕ := 3
def grid_columns : ℕ := 9

def triangle1_area : ℝ := 3
def triangle2_area : ℝ := 1
def triangle3_area : ℝ := 3
def triangle4_area : ℝ := 3

def total_grid_area := (grid_rows * grid_columns : ℕ) * square_side_length^2
def total_unshaded_area := triangle1_area + triangle2_area + triangle3_area + triangle4_area

-- Problem statement
theorem shaded_area_correct :
  total_grid_area - total_unshaded_area = 17 := 
by
  sorry

end NUMINAMATH_GPT_shaded_area_correct_l773_77335


namespace NUMINAMATH_GPT_find_divisor_l773_77332

theorem find_divisor (d : ℕ) (H1 : 199 = d * 11 + 1) : d = 18 := 
sorry

end NUMINAMATH_GPT_find_divisor_l773_77332


namespace NUMINAMATH_GPT_part1_part2_part3_part4_l773_77308

-- Part 1: Prove that 1/42 is equal to 1/6 - 1/7
theorem part1 : (1/42 : ℚ) = (1/6 : ℚ) - (1/7 : ℚ) := sorry

-- Part 2: Prove that 1/240 is equal to 1/15 - 1/16
theorem part2 : (1/240 : ℚ) = (1/15 : ℚ) - (1/16 : ℚ) := sorry

-- Part 3: Prove the general rule for all natural numbers m
theorem part3 (m : ℕ) (hm : m > 0) : (1 / (m * (m + 1)) : ℚ) = (1 / m : ℚ) - (1 / (m + 1) : ℚ) := sorry

-- Part 4: Prove the given expression evaluates to 0 for any x
theorem part4 (x : ℚ) (h1 : x ≠ 1) (h2 : x ≠ 2) (h3 : x ≠ 3) : 
  (1 / ((x - 2) * (x - 3)) : ℚ) - (2 / ((x - 1) * (x - 3)) : ℚ) + (1 / ((x - 1) * (x - 2)) : ℚ) = 0 := sorry

end NUMINAMATH_GPT_part1_part2_part3_part4_l773_77308


namespace NUMINAMATH_GPT_range_of_a_l773_77307

noncomputable def A (x : ℝ) : Prop := x^2 - x ≤ 0
noncomputable def B (x : ℝ) (a : ℝ) : Prop := 2^(1 - x) + a ≤ 0

theorem range_of_a (a : ℝ) : (∀ x, A x → B x a) → a ≤ -2 := by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_range_of_a_l773_77307


namespace NUMINAMATH_GPT_perfect_square_condition_l773_77360

theorem perfect_square_condition (x y : ℕ) :
  ∃ k : ℕ, (x + y)^2 + 3*x + y + 1 = k^2 ↔ x = y := 
by 
  sorry

end NUMINAMATH_GPT_perfect_square_condition_l773_77360


namespace NUMINAMATH_GPT_equality_of_floor_squares_l773_77321

theorem equality_of_floor_squares (n : ℕ) (hn : 0 < n) :
  (⌊Real.sqrt n + Real.sqrt (n + 1)⌋ : ℤ) = ⌊Real.sqrt (4 * n + 1)⌋ ∧
  (⌊Real.sqrt (4 * n + 1)⌋ : ℤ) = ⌊Real.sqrt (4 * n + 2)⌋ ∧
  (⌊Real.sqrt (4 * n + 2)⌋ : ℤ) = ⌊Real.sqrt (4 * n + 3)⌋ :=
by
  sorry

end NUMINAMATH_GPT_equality_of_floor_squares_l773_77321


namespace NUMINAMATH_GPT_product_simplification_l773_77312

theorem product_simplification :
  (10 * (1 / 5) * (1 / 2) * 4 / 2 : ℝ) = 2 :=
by
  sorry

end NUMINAMATH_GPT_product_simplification_l773_77312


namespace NUMINAMATH_GPT_diff_eq_40_l773_77325

theorem diff_eq_40 (x y : ℤ) (h1 : x + y = 24) (h2 : x = 32) : x - y = 40 := by
  sorry

end NUMINAMATH_GPT_diff_eq_40_l773_77325


namespace NUMINAMATH_GPT_find_sum_of_squares_of_roots_l773_77371

theorem find_sum_of_squares_of_roots:
  ∀ (a b c d : ℝ), (a^2 * b^2 * c^2 * d^2 - 15 * a * b * c * d + 56 = 0) → 
  a^2 + b^2 + c^2 + d^2 = 30 := by
  intros a b c d h
  sorry

end NUMINAMATH_GPT_find_sum_of_squares_of_roots_l773_77371


namespace NUMINAMATH_GPT_jane_brown_sheets_l773_77363

theorem jane_brown_sheets :
  ∀ (total_sheets yellow_sheets brown_sheets : ℕ),
    total_sheets = 55 →
    yellow_sheets = 27 →
    brown_sheets = total_sheets - yellow_sheets →
    brown_sheets = 28 := 
by
  intros total_sheets yellow_sheets brown_sheets ht hy hb
  rw [ht, hy] at hb
  simp at hb
  exact hb

end NUMINAMATH_GPT_jane_brown_sheets_l773_77363


namespace NUMINAMATH_GPT_sum_of_digits_l773_77374

theorem sum_of_digits (a b : ℕ) (h1 : 4 * 100 + a * 10 + 3 + 984 = 1 * 1000 + 3 * 100 + b * 10 + 7)
  (h2 : (1 + b) - (3 + 7) % 11 = 0) : a + b = 10 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_l773_77374


namespace NUMINAMATH_GPT_consecutive_odd_integers_l773_77364

theorem consecutive_odd_integers (n : ℕ) (h1 : n > 0) (h2 : (1 : ℚ) / n * ((n : ℚ) * 154) = 154) : n = 10 :=
sorry

end NUMINAMATH_GPT_consecutive_odd_integers_l773_77364


namespace NUMINAMATH_GPT_find_days_l773_77311

variables (a d e k m : ℕ) (y : ℕ)

-- Assumptions based on the problem
def workers_efficiency_condition : Prop := 
  (a * e * (d * k) / (a * e)) = d

-- Conclusion we aim to prove
def target_days_condition : Prop :=
  y = (a * a) / (d * k * m)

theorem find_days (h : workers_efficiency_condition a d e k) : target_days_condition a d k m y :=
  sorry

end NUMINAMATH_GPT_find_days_l773_77311


namespace NUMINAMATH_GPT_jim_saving_amount_l773_77362

theorem jim_saving_amount
    (sara_initial_savings : ℕ)
    (sara_weekly_savings : ℕ)
    (jim_weekly_savings : ℕ)
    (weeks_elapsed : ℕ)
    (sara_total_savings : ℕ := sara_initial_savings + weeks_elapsed * sara_weekly_savings)
    (jim_total_savings : ℕ := weeks_elapsed * jim_weekly_savings)
    (savings_equal: sara_total_savings = jim_total_savings)
    (sara_initial_savings_value : sara_initial_savings = 4100)
    (sara_weekly_savings_value : sara_weekly_savings = 10)
    (weeks_elapsed_value : weeks_elapsed = 820) :
    jim_weekly_savings = 15 := 
by
  sorry

end NUMINAMATH_GPT_jim_saving_amount_l773_77362
