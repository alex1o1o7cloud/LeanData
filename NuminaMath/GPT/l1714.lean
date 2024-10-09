import Mathlib

namespace evaluate_polynomial_at_3_l1714_171477

def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 - 3*x + 2

theorem evaluate_polynomial_at_3 : f 3 = 2 :=
by
  sorry

end evaluate_polynomial_at_3_l1714_171477


namespace find_angle_C_find_perimeter_l1714_171452

-- Definitions related to the triangle problem
variables {A B C : ℝ}
variables {a b c : ℝ} -- sides opposite to A, B, C

-- Condition: (2a - b) * cos C = c * cos B
def condition_1 (a b c C B : ℝ) : Prop := (2 * a - b) * Real.cos C = c * Real.cos B

-- Given C in radians (part 1: find angle C)
theorem find_angle_C 
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : condition_1 a b c C B) 
  (H1 : 0 < C) (H2 : C < Real.pi) :
  C = Real.pi / 3 := 
sorry

-- More conditions for part 2
variables (area : ℝ) -- given area of triangle
def condition_2 (a b C area : ℝ) : Prop := 0.5 * a * b * Real.sin C = area

-- Given c = 2 and area = sqrt(3) (part 2: find perimeter)
theorem find_perimeter 
  (A B C : ℝ) (a b : ℝ) (c : ℝ) (area : ℝ) 
  (h2 : condition_2 a b C area) 
  (Hc : c = 2) (Harea : area = Real.sqrt 3) :
  a + b + c = 6 := 
sorry

end find_angle_C_find_perimeter_l1714_171452


namespace intermission_length_l1714_171474

def concert_duration : ℕ := 80
def song_duration_total : ℕ := 70

theorem intermission_length : 
  concert_duration - song_duration_total = 10 :=
by
  -- conditions are already defined above
  sorry

end intermission_length_l1714_171474


namespace total_orchestra_l1714_171418

def percussion_section : ℕ := 4
def brass_section : ℕ := 13
def strings_section : ℕ := 18
def woodwinds_section : ℕ := 10
def keyboards_and_harp_section : ℕ := 3
def maestro : ℕ := 1

theorem total_orchestra (p b s w k m : ℕ) 
  (h_p : p = percussion_section)
  (h_b : b = brass_section)
  (h_s : s = strings_section)
  (h_w : w = woodwinds_section)
  (h_k : k = keyboards_and_harp_section)
  (h_m : m = maestro) :
  p + b + s + w + k + m = 49 := by 
  rw [h_p, h_b, h_s, h_w, h_k, h_m]
  unfold percussion_section brass_section strings_section woodwinds_section keyboards_and_harp_section maestro
  norm_num

end total_orchestra_l1714_171418


namespace Meghan_scored_20_marks_less_than_Jose_l1714_171447

theorem Meghan_scored_20_marks_less_than_Jose
  (M J A : ℕ)
  (h1 : J = A + 40)
  (h2 : M + J + A = 210)
  (h3 : J = 100 - 10) :
  J - M = 20 :=
by
  -- Skipping the proof
  sorry

end Meghan_scored_20_marks_less_than_Jose_l1714_171447


namespace geometric_sequence_arithmetic_median_l1714_171459

theorem geometric_sequence_arithmetic_median 
  (a : ℕ → ℝ) 
  (hpos : ∀ n, 0 < a n) 
  (h_arith : 2 * a 1 + a 2 = 2 * a 3) :
  (a 2017 + a 2016) / (a 2015 + a 2014) = 4 :=
sorry

end geometric_sequence_arithmetic_median_l1714_171459


namespace product_of_sums_of_four_squares_is_sum_of_four_squares_l1714_171480

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

end product_of_sums_of_four_squares_is_sum_of_four_squares_l1714_171480


namespace find_x_l1714_171448

theorem find_x : ∃ x : ℝ, (1 / 3 * ((2 * x + 5) + (8 * x + 3) + (3 * x + 8)) = 5 * x - 10) ∧ x = 23 :=
by
  sorry

end find_x_l1714_171448


namespace students_taking_history_but_not_statistics_l1714_171430

theorem students_taking_history_but_not_statistics :
  ∀ (total_students history_students statistics_students history_or_statistics_both : ℕ),
    total_students = 90 →
    history_students = 36 →
    statistics_students = 32 →
    history_or_statistics_both = 57 →
    history_students - (history_students + statistics_students - history_or_statistics_both) = 25 :=
by intros; sorry

end students_taking_history_but_not_statistics_l1714_171430


namespace ferris_wheel_seat_calculation_l1714_171429

theorem ferris_wheel_seat_calculation (n k : ℕ) (h1 : n = 4) (h2 : k = 2) : n / k = 2 := 
by
  sorry

end ferris_wheel_seat_calculation_l1714_171429


namespace arithmetic_sequence_15th_term_is_171_l1714_171479

theorem arithmetic_sequence_15th_term_is_171 :
  ∀ (a d : ℕ), a = 3 → d = 15 - a → a + 14 * d = 171 :=
by
  intros a d h_a h_d
  rw [h_a, h_d]
  -- The proof would follow with the arithmetic calculation to determine the 15th term
  sorry

end arithmetic_sequence_15th_term_is_171_l1714_171479


namespace smallest_nine_ten_eleven_consecutive_sum_l1714_171422

theorem smallest_nine_ten_eleven_consecutive_sum :
  ∃ n : ℕ, n > 0 ∧ (n % 9 = 0) ∧ (n % 10 = 5) ∧ (n % 11 = 0) ∧ n = 495 :=
by {
  sorry
}

end smallest_nine_ten_eleven_consecutive_sum_l1714_171422


namespace product_not_perfect_square_l1714_171475

theorem product_not_perfect_square :
  ¬ ∃ n : ℕ, n^2 = (2021^1004) * (6^3) :=
by
  sorry

end product_not_perfect_square_l1714_171475


namespace problem1_problem2_l1714_171476

variables {a b c : ℝ}

-- (1) Prove that a + b + c = 4 given the conditions
theorem problem1 (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_min : ∀ x, abs (x + a) + abs (x - b) + c ≥ 4) : a + b + c = 4 := 
sorry

-- (2) Prove that the minimum value of (1/4)a^2 + (1/9)b^2 + c^2 is 8/7 given the conditions and that a + b + c = 4
theorem problem2 (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 4) : (1/4) * a^2 + (1/9) * b^2 + c^2 ≥ 8 / 7 := 
sorry

end problem1_problem2_l1714_171476


namespace range_of_a_l1714_171490

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (h₁ : ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ ≥ f x₂)
                    (h₂ : -2 ≤ a + 1 ∧ a + 1 ≤ 4)
                    (h₃ : -2 ≤ 2 * a ∧ 2 * a ≤ 4)
                    (h₄ : f (a + 1) > f (2 * a)) : 1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l1714_171490


namespace simplify_and_evaluate_l1714_171496

theorem simplify_and_evaluate (x : ℝ) (h : x = 3) : 
  ( ( (x^2 - 4 * x + 4) / (x^2 - 4) ) / ( (x-2) / (x^2 + 2*x) ) ) + 3 = 6 :=
by
  sorry

end simplify_and_evaluate_l1714_171496


namespace larger_triangle_side_length_l1714_171404

theorem larger_triangle_side_length
    (A1 A2 : ℕ) (k : ℤ)
    (h1 : A1 - A2 = 32)
    (h2 : A1 = k^2 * A2)
    (h3 : A2 = 4 ∨ A2 = 8 ∨ A2 = 16)
    (h4 : ((4 : ℤ) * k = 12)) :
    (4 * k) = 12 :=
by sorry

end larger_triangle_side_length_l1714_171404


namespace image_of_center_l1714_171435

def original_center : ℤ × ℤ := (3, -4)

def reflect_x (p : ℤ × ℤ) : ℤ × ℤ := (p.1, -p.2)
def reflect_y (p : ℤ × ℤ) : ℤ × ℤ := (-p.1, p.2)
def translate_down (p : ℤ × ℤ) (d : ℤ) : ℤ × ℤ := (p.1, p.2 - d)

theorem image_of_center :
  (translate_down (reflect_y (reflect_x original_center)) 10) = (-3, -6) :=
by
  sorry

end image_of_center_l1714_171435


namespace intercepts_of_line_l1714_171489

theorem intercepts_of_line (x y : ℝ) (h_eq : 4 * x + 7 * y = 28) :
  (∃ y, (x = 0 ∧ y = 4) ∧ ∃ x, (y = 0 ∧ x = 7)) :=
by
  sorry

end intercepts_of_line_l1714_171489


namespace children_neither_happy_nor_sad_l1714_171485

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

end children_neither_happy_nor_sad_l1714_171485


namespace find_three_digit_number_l1714_171451

theorem find_three_digit_number (a b c : ℕ) (h1 : a + b + c = 16)
    (h2 : 100 * b + 10 * a + c = 100 * a + 10 * b + c - 360)
    (h3 : 100 * a + 10 * c + b = 100 * a + 10 * b + c + 54) :
    100 * a + 10 * b + c = 628 :=
by
  sorry

end find_three_digit_number_l1714_171451


namespace sum_of_prime_factors_eq_28_l1714_171458

-- Define 2310 as a constant
def n : ℕ := 2310

-- Define the prime factors of 2310
def prime_factors : List ℕ := [2, 3, 5, 7, 11]

-- The sum of the prime factors
def sum_prime_factors : ℕ := prime_factors.sum

-- State the theorem
theorem sum_of_prime_factors_eq_28 : sum_prime_factors = 28 :=
by 
  sorry

end sum_of_prime_factors_eq_28_l1714_171458


namespace domain_of_f_l1714_171493

def function_domain (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∀ x, x ∈ domain ↔ ∃ y, f y = x

noncomputable def f (x : ℝ) : ℝ :=
  (x + 6) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_of_f :
  function_domain f ((Set.Iio 2) ∪ (Set.Ioi 3)) :=
by
  sorry

end domain_of_f_l1714_171493


namespace number_increase_when_reversed_l1714_171470

theorem number_increase_when_reversed :
  let n := 253
  let reversed_n := 352
  reversed_n - n = 99 :=
by
  let n := 253
  let reversed_n := 352
  sorry

end number_increase_when_reversed_l1714_171470


namespace tan_half_angle_l1714_171449

-- Definition for the given angle in the third quadrant with a given sine value
def angle_in_third_quadrant_and_sin (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h_sin : Real.sin α = -24 / 25) : Prop :=
  True

-- The main theorem to prove the given condition implies the result
theorem tan_half_angle (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h_sin : Real.sin α = -24 / 25) :
  Real.tan (α / 2) = -4 / 3 :=
by
  sorry

end tan_half_angle_l1714_171449


namespace matthews_annual_income_l1714_171464

noncomputable def annual_income (q : ℝ) (I : ℝ) (T : ℝ) : Prop :=
  T = 0.01 * q * 50000 + 0.01 * (q + 3) * (I - 50000) ∧
  T = 0.01 * (q + 0.5) * I → I = 60000

-- Statement of the math proof
theorem matthews_annual_income (q : ℝ) (T : ℝ) :
  ∃ I : ℝ, I = 60000 ∧ annual_income q I T :=
sorry

end matthews_annual_income_l1714_171464


namespace intersection_of_A_and_B_l1714_171408

namespace ProofProblem

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {1, 2, 4, 5}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 5} := by
  sorry

end ProofProblem

end intersection_of_A_and_B_l1714_171408


namespace proof_inequality_l1714_171402

variable {a b c : ℝ}

theorem proof_inequality (h : a * b < 0) : a^2 + b^2 + c^2 > 2 * a * b + 2 * b * c + 2 * c * a := by
  sorry

end proof_inequality_l1714_171402


namespace just_passed_students_l1714_171462

theorem just_passed_students (total_students : ℕ) 
  (math_first_division_perc : ℕ) 
  (math_second_division_perc : ℕ)
  (eng_first_division_perc : ℕ)
  (eng_second_division_perc : ℕ)
  (sci_first_division_perc : ℕ)
  (sci_second_division_perc : ℕ) 
  (math_just_passed : ℕ)
  (eng_just_passed : ℕ)
  (sci_just_passed : ℕ) :
  total_students = 500 →
  math_first_division_perc = 35 →
  math_second_division_perc = 48 →
  eng_first_division_perc = 25 →
  eng_second_division_perc = 60 →
  sci_first_division_perc = 40 →
  sci_second_division_perc = 45 →
  math_just_passed = (100 - (math_first_division_perc + math_second_division_perc)) * total_students / 100 →
  eng_just_passed = (100 - (eng_first_division_perc + eng_second_division_perc)) * total_students / 100 →
  sci_just_passed = (100 - (sci_first_division_perc + sci_second_division_perc)) * total_students / 100 →
  math_just_passed = 85 ∧ eng_just_passed = 75 ∧ sci_just_passed = 75 :=
by
  intros ht hf1 hf2 he1 he2 hs1 hs2 hjm hje hjs
  sorry

end just_passed_students_l1714_171462


namespace dave_initial_apps_l1714_171415

theorem dave_initial_apps (x : ℕ) (h1 : x - 18 = 5) : x = 23 :=
by {
  -- This is where the proof would go 
  sorry -- The proof is omitted as per instructions
}

end dave_initial_apps_l1714_171415


namespace does_not_uniquely_determine_equilateral_l1714_171472

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

end does_not_uniquely_determine_equilateral_l1714_171472


namespace transform_M_eq_l1714_171439

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![0, 1/3], ![1, -2/3]]

def M : Fin 2 → ℚ :=
  ![-1, 1]

theorem transform_M_eq :
  A⁻¹.mulVec M = ![-1, -3] :=
by
  sorry

end transform_M_eq_l1714_171439


namespace Da_Yan_sequence_20th_term_l1714_171423

noncomputable def Da_Yan_sequence_term (n: ℕ) : ℕ :=
  if n % 2 = 0 then
    (n^2) / 2
  else
    (n^2 - 1) / 2

theorem Da_Yan_sequence_20th_term : Da_Yan_sequence_term 20 = 200 :=
by
  sorry

end Da_Yan_sequence_20th_term_l1714_171423


namespace jean_spots_l1714_171483

theorem jean_spots (total_spots upper_torso_spots back_hindspots sides_spots : ℕ)
  (h1 : upper_torso_spots = 30)
  (h2 : total_spots = 2 * upper_torso_spots)
  (h3 : back_hindspots = total_spots / 3)
  (h4 : sides_spots = total_spots - upper_torso_spots - back_hindspots) :
  sides_spots = 10 :=
by
  sorry

end jean_spots_l1714_171483


namespace inverse_exists_l1714_171419

noncomputable def f (x : ℝ) : ℝ := 7 * x^3 - 2 * x^2 + 5 * x - 9

theorem inverse_exists :
  ∃ x : ℝ, 7 * x^3 - 2 * x^2 + 5 * x - 5.5 = 0 :=
sorry

end inverse_exists_l1714_171419


namespace prime_pairs_divisibility_l1714_171410

theorem prime_pairs_divisibility (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (p * q) ∣ (p ^ p + q ^ q + 1) ↔ (p = 2 ∧ q = 5) ∨ (p = 5 ∧ q = 2) :=
by
  sorry

end prime_pairs_divisibility_l1714_171410


namespace valerie_money_left_l1714_171438

theorem valerie_money_left
  (small_bulb_cost : ℕ)
  (large_bulb_cost : ℕ)
  (num_small_bulbs : ℕ)
  (num_large_bulbs : ℕ)
  (initial_money : ℕ) :
  small_bulb_cost = 8 →
  large_bulb_cost = 12 →
  num_small_bulbs = 3 →
  num_large_bulbs = 1 →
  initial_money = 60 →
  initial_money - (num_small_bulbs * small_bulb_cost + num_large_bulbs * large_bulb_cost) = 24 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end valerie_money_left_l1714_171438


namespace average_of_list_l1714_171416

theorem average_of_list (n : ℕ) (h : (2 + 9 + 4 + n + 2 * n) / 5 = 6) : n = 5 := 
by
  sorry

end average_of_list_l1714_171416


namespace factorize_expression_l1714_171461

theorem factorize_expression (x y : ℂ) : (x * y^2 - x = x * (y + 1) * (y - 1)) :=
sorry

end factorize_expression_l1714_171461


namespace find_ordered_pair_l1714_171445

noncomputable def ordered_pair (c d : ℝ) := c = 1 ∧ d = -2

theorem find_ordered_pair (c d : ℝ) (h1 : c ≠ 0) (h2 : d ≠ 0) (h3 : ∀ x : ℝ, x^2 + c * x + d = 0 → (x = c ∨ x = d)) : ordered_pair c d :=
by
  sorry

end find_ordered_pair_l1714_171445


namespace fred_gave_balloons_to_sandy_l1714_171495

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

end fred_gave_balloons_to_sandy_l1714_171495


namespace bricks_in_top_half_l1714_171465

theorem bricks_in_top_half (total_rows bottom_rows top_rows bricks_per_bottom_row total_bricks bricks_per_top_row: ℕ) 
  (h_total_rows : total_rows = 10)
  (h_bottom_rows : bottom_rows = 5)
  (h_top_rows : top_rows = 5)
  (h_bricks_per_bottom_row : bricks_per_bottom_row = 12)
  (h_total_bricks : total_bricks = 100)
  (h_bricks_per_top_row : bricks_per_top_row = (total_bricks - bottom_rows * bricks_per_bottom_row) / top_rows) : 
  bricks_per_top_row = 8 := 
by 
  sorry

end bricks_in_top_half_l1714_171465


namespace collinear_points_x_value_l1714_171436

theorem collinear_points_x_value
  (x : ℝ)
  (h : ∃ m : ℝ, m = (1 - (-4)) / (-1 - 2) ∧ m = (-9 - (-4)) / (x - 2)) :
  x = 5 :=
by
  sorry

end collinear_points_x_value_l1714_171436


namespace helen_owes_more_l1714_171442

noncomputable def future_value (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def future_value_semiannually : ℝ :=
  future_value 8000 0.10 2 3

noncomputable def future_value_annually : ℝ :=
  8000 * (1 + 0.10) ^ 3

noncomputable def difference : ℝ :=
  future_value_semiannually - future_value_annually

theorem helen_owes_more : abs (difference - 72.80) < 0.01 :=
by
  sorry

end helen_owes_more_l1714_171442


namespace linlin_speed_l1714_171487

theorem linlin_speed (distance time : ℕ) (q_speed linlin_speed : ℕ)
  (h1 : distance = 3290)
  (h2 : time = 7)
  (h3 : q_speed = 70)
  (h4 : distance = (q_speed + linlin_speed) * time) : linlin_speed = 400 :=
by sorry

end linlin_speed_l1714_171487


namespace hockey_league_total_games_l1714_171413

theorem hockey_league_total_games 
  (divisions : ℕ)
  (teams_per_division : ℕ)
  (intra_division_games : ℕ)
  (inter_division_games : ℕ) :
  divisions = 2 →
  teams_per_division = 6 →
  intra_division_games = 4 →
  inter_division_games = 2 →
  (divisions * ((teams_per_division * (teams_per_division - 1)) / 2) * intra_division_games) + 
  ((divisions / 2) * (divisions / 2) * teams_per_division * teams_per_division * inter_division_games) = 192 :=
by
  intros h_div h_teams h_intra h_inter
  sorry

end hockey_league_total_games_l1714_171413


namespace greater_number_is_twelve_l1714_171443

theorem greater_number_is_twelve (x : ℕ) (a b : ℕ) 
  (h1 : a = 3 * x) 
  (h2 : b = 4 * x) 
  (h3 : a + b = 21) : 
  max a b = 12 :=
by 
  sorry

end greater_number_is_twelve_l1714_171443


namespace sum_of_fractions_l1714_171424

theorem sum_of_fractions : (1/2 + 1/2 + 1/3 + 1/3 + 1/3) = 2 :=
by
  -- Proof goes here
  sorry

end sum_of_fractions_l1714_171424


namespace jessica_allowance_l1714_171426

theorem jessica_allowance (A : ℝ) (h1 : A / 2 + 6 = 11) : A = 10 := by
  sorry

end jessica_allowance_l1714_171426


namespace words_per_page_large_font_l1714_171481

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

end words_per_page_large_font_l1714_171481


namespace heat_required_l1714_171453

theorem heat_required (m : ℝ) (c₀ : ℝ) (alpha : ℝ) (t₁ t₂ : ℝ) :
  m = 2 ∧ c₀ = 150 ∧ alpha = 0.05 ∧ t₁ = 20 ∧ t₂ = 100 →
  let Δt := t₂ - t₁
  let c_avg := (c₀ * (1 + alpha * t₁) + c₀ * (1 + alpha * t₂)) / 2
  let Q := c_avg * m * Δt
  Q = 96000 := by
  sorry

end heat_required_l1714_171453


namespace find_number_l1714_171482

theorem find_number (x : ℝ) (h : x / 4 + 15 = 4 * x - 15) : x = 8 :=
sorry

end find_number_l1714_171482


namespace workers_time_to_complete_job_l1714_171441

theorem workers_time_to_complete_job (D E Z H k : ℝ) (h1 : 1 / D + 1 / E + 1 / Z + 1 / H = 1 / (D - 8))
  (h2 : 1 / D + 1 / E + 1 / Z + 1 / H = 1 / (E - 2))
  (h3 : 1 / D + 1 / E + 1 / Z + 1 / H = 3 / Z) :
  E = 10 → Z = 3 * (E - 2) → k = 120 / 19 :=
by
  intros hE hZ
  sorry

end workers_time_to_complete_job_l1714_171441


namespace chord_length_range_l1714_171400

open Real

def chord_length_ge (t : ℝ) : Prop :=
  let r := sqrt 8
  let l := (4 * sqrt 2) / 3
  let d := abs t / sqrt 2
  let s := l / 2
  s ≤ sqrt (r^2 - d^2)

theorem chord_length_range (t : ℝ) : chord_length_ge t ↔ -((8 * sqrt 2) / 3) ≤ t ∧ t ≤ (8 * sqrt 2) / 3 :=
by
  sorry

end chord_length_range_l1714_171400


namespace solve_abs_ineq_l1714_171420

theorem solve_abs_ineq (x : ℝ) : |(8 - x) / 4| < 3 ↔ 4 < x ∧ x < 20 := by
  sorry

end solve_abs_ineq_l1714_171420


namespace relationship_between_x_and_y_l1714_171421

theorem relationship_between_x_and_y (m x y : ℝ) (h1 : x = 3 - m) (h2 : y = 2 * m + 1) : 2 * x + y = 7 :=
sorry

end relationship_between_x_and_y_l1714_171421


namespace minimum_square_area_l1714_171414

-- Definitions of the given conditions
structure Rectangle where
  width : ℕ
  height : ℕ

def rect1 : Rectangle := { width := 2, height := 4 }
def rect2 : Rectangle := { width := 3, height := 5 }
def circle_diameter : ℕ := 3

-- Statement of the theorem
theorem minimum_square_area :
  ∃ sq_side : ℕ, 
    (sq_side ≥ 5 ∧ sq_side ≥ 7) ∧ 
    sq_side * sq_side = 49 := 
by
  use 7
  have h1 : 7 ≥ 5 := by norm_num
  have h2 : 7 ≥ 7 := by norm_num
  have h3 : 7 * 7 = 49 := by norm_num
  exact ⟨⟨h1, h2⟩, h3⟩

end minimum_square_area_l1714_171414


namespace lineD_is_parallel_to_line1_l1714_171403

-- Define the lines
def line1 (x y : ℝ) := x - 2 * y + 1 = 0
def lineA (x y : ℝ) := 2 * x - y + 1 = 0
def lineB (x y : ℝ) := 2 * x - 4 * y + 2 = 0
def lineC (x y : ℝ) := 2 * x + 4 * y + 1 = 0
def lineD (x y : ℝ) := 2 * x - 4 * y + 1 = 0

-- Define a function to check parallelism between lines
def are_parallel (f g : ℝ → ℝ → Prop) :=
  ∀ x y : ℝ, (f x y → g x y) ∨ (g x y → f x y)

-- Prove that lineD is parallel to line1
theorem lineD_is_parallel_to_line1 : are_parallel line1 lineD :=
by
  sorry

end lineD_is_parallel_to_line1_l1714_171403


namespace mary_flour_total_l1714_171499

-- Definitions for conditions
def initial_flour : ℝ := 7.0
def extra_flour : ℝ := 2.0
def total_flour (x y : ℝ) : ℝ := x + y

-- The statement we want to prove
theorem mary_flour_total : total_flour initial_flour extra_flour = 9.0 := 
by sorry

end mary_flour_total_l1714_171499


namespace preimage_of_3_2_eq_l1714_171484

noncomputable def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * p.2, p.1 + p.2)

theorem preimage_of_3_2_eq (x y : ℝ) :
  f (x, y) = (-3, 2) ↔ (x = 3 ∧ y = -1) ∨ (x = -1 ∧ y = 3) :=
by
  sorry

end preimage_of_3_2_eq_l1714_171484


namespace MH_greater_than_MK_l1714_171488

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

end MH_greater_than_MK_l1714_171488


namespace exists_infinite_solutions_l1714_171460

theorem exists_infinite_solutions :
  ∃ (x y z : ℤ), (∀ k : ℤ, x = 2 * k ∧ y = 999 - 2 * k ^ 2 ∧ z = 998 - 2 * k ^ 2) ∧ (x ^ 2 + y ^ 2 - z ^ 2 = 1997) :=
by 
  -- The proof should go here
  sorry

end exists_infinite_solutions_l1714_171460


namespace double_angle_cosine_calculation_l1714_171444

theorem double_angle_cosine_calculation :
    2 * (Real.cos (Real.pi / 12))^2 - 1 = Real.cos (Real.pi / 6) := 
by
    sorry

end double_angle_cosine_calculation_l1714_171444


namespace no_five_coprime_two_digit_composites_l1714_171492

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

end no_five_coprime_two_digit_composites_l1714_171492


namespace simplify_expression_l1714_171469

theorem simplify_expression (x : ℝ) (h : x^2 + 2 * x - 6 = 0) : 
  ((x - 1) / (x - 3) - (x + 1) / x) / ((x^2 + 3 * x) / (x^2 - 6 * x + 9)) = -1/2 := 
by
  sorry

end simplify_expression_l1714_171469


namespace percent_errors_l1714_171425

theorem percent_errors (S : ℝ) (hS : S > 0) (Sm : ℝ) (hSm : Sm = 1.25 * S) :
  let P := 4 * S
  let Pm := 4 * Sm
  let A := S^2
  let Am := Sm^2
  let D := S * Real.sqrt 2
  let Dm := Sm * Real.sqrt 2
  let E_P := ((Pm - P) / P) * 100
  let E_A := ((Am - A) / A) * 100
  let E_D := ((Dm - D) / D) * 100
  E_P = 25 ∧ E_A = 56.25 ∧ E_D = 25 :=
by
  sorry

end percent_errors_l1714_171425


namespace complete_the_square_l1714_171446

theorem complete_the_square (x : ℝ) (h : x^2 - 4 * x + 3 = 0) : (x - 2)^2 = 1 :=
sorry

end complete_the_square_l1714_171446


namespace extra_apples_correct_l1714_171494

def num_red_apples : ℕ := 6
def num_green_apples : ℕ := 15
def num_students : ℕ := 5
def num_apples_ordered : ℕ := num_red_apples + num_green_apples
def num_apples_taken : ℕ := num_students
def num_extra_apples : ℕ := num_apples_ordered - num_apples_taken

theorem extra_apples_correct : num_extra_apples = 16 := by
  sorry

end extra_apples_correct_l1714_171494


namespace geometric_sequence_a4_a5_sum_l1714_171427

theorem geometric_sequence_a4_a5_sum :
  (∀ n : ℕ, a_n > 0) → (a_3 = 3) → (a_6 = (1 / 9)) → 
  (a_4 + a_5 = (4 / 3)) :=
by
  sorry

end geometric_sequence_a4_a5_sum_l1714_171427


namespace problem_pf_qf_geq_f_pq_l1714_171450

variable {R : Type*} [LinearOrderedField R]

theorem problem_pf_qf_geq_f_pq (f : R → R) (a b p q x y : R) (hpq : p + q = 1) :
  (∀ x y, p * f x + q * f y ≥ f (p * x + q * y)) ↔ (0 ≤ p ∧ p ≤ 1) := 
by
  sorry

end problem_pf_qf_geq_f_pq_l1714_171450


namespace half_ears_kernels_l1714_171497

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

end half_ears_kernels_l1714_171497


namespace dogs_Carly_worked_on_l1714_171463

-- Define the parameters for the problem
def total_nails := 164
def three_legged_dogs := 3
def three_nail_paw_dogs := 2
def extra_nail_paw_dog := 1
def regular_dog_nails := 16
def three_legged_nails := (regular_dog_nails - 4)
def three_nail_paw_nails := (regular_dog_nails - 1)
def extra_nail_paw_nails := (regular_dog_nails + 1)

-- Lean statement to prove the number of dogs Carly worked on today
theorem dogs_Carly_worked_on :
  (3 * three_legged_nails) + (2 * three_nail_paw_nails) + extra_nail_paw_nails 
  = 83 → ((total_nails - 83) / regular_dog_nails ≠ 0) → 5 + 3 + 2 + 1 = 11 :=
by sorry

end dogs_Carly_worked_on_l1714_171463


namespace distinct_complex_roots_A_eq_neg7_l1714_171401

theorem distinct_complex_roots_A_eq_neg7 (x₁ x₂ : ℂ) (A : ℝ) (hx1: x₁ ≠ x₂)
  (h1 : x₁ * (x₁ + 1) = A)
  (h2 : x₂ * (x₂ + 1) = A)
  (h3 : x₁^4 + 3 * x₁^3 + 5 * x₁ = x₂^4 + 3 * x₂^3 + 5 * x₂) : A = -7 := 
sorry

end distinct_complex_roots_A_eq_neg7_l1714_171401


namespace optionA_optionC_optionD_l1714_171471

noncomputable def f (x : ℝ) := (3 : ℝ) ^ x / (1 + (3 : ℝ) ^ x)

theorem optionA : ∀ x : ℝ, f (-x) + f x = 1 := by
  sorry

theorem optionC : ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ (y > 0 ∧ y < 1) := by
  sorry

theorem optionD : ∀ x : ℝ, f (2 * x - 3) + f (x - 3) > 1 ↔ x > 2 := by
  sorry

end optionA_optionC_optionD_l1714_171471


namespace evening_campers_l1714_171434

theorem evening_campers (morning_campers afternoon_campers total_campers : ℕ) (h_morning : morning_campers = 36) (h_afternoon : afternoon_campers = 13) (h_total : total_campers = 98) :
  total_campers - (morning_campers + afternoon_campers) = 49 :=
by
  sorry

end evening_campers_l1714_171434


namespace calculate_f8_f4_l1714_171432

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 5) = f x
axiom f_at_1 : f 1 = 1
axiom f_at_2 : f 2 = 3

theorem calculate_f8_f4 : f 8 - f 4 = -2 := by
  sorry

end calculate_f8_f4_l1714_171432


namespace battery_life_remaining_l1714_171498

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

end battery_life_remaining_l1714_171498


namespace projectile_reaches_100_feet_l1714_171412

theorem projectile_reaches_100_feet :
  ∃ (t : ℝ), t > 0 ∧ (-16 * t ^ 2 + 80 * t = 100) ∧ (t = 2.5) := by
sorry

end projectile_reaches_100_feet_l1714_171412


namespace least_three_digit_12_heavy_number_l1714_171433

def is_12_heavy (n : ℕ) : Prop :=
  n % 12 > 8

def three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem least_three_digit_12_heavy_number :
  ∃ n, three_digit n ∧ is_12_heavy n ∧ ∀ m, three_digit m ∧ is_12_heavy m → n ≤ m :=
  Exists.intro 105 (by
    sorry)

end least_three_digit_12_heavy_number_l1714_171433


namespace line_intersects_circle_l1714_171491

theorem line_intersects_circle : 
  ∀ (x y : ℝ), 
  (2 * x + y = 0) ∧ (x^2 + y^2 + 2 * x - 4 * y - 4 = 0) ↔
    ∃ (x0 y0 : ℝ), (2 * x0 + y0 = 0) ∧ ((x0 + 1)^2 + (y0 - 2)^2 = 9) :=
by
  sorry

end line_intersects_circle_l1714_171491


namespace katya_sold_glasses_l1714_171437

-- Definitions based on the conditions specified in the problem
def ricky_sales : ℕ := 9

def tina_sales (K : ℕ) : ℕ := 2 * (K + ricky_sales)

def katya_sales_eq (K : ℕ) : Prop := tina_sales K = K + 26

-- Lean statement to prove Katya sold 8 glasses of lemonade
theorem katya_sold_glasses : ∃ (K : ℕ), katya_sales_eq K ∧ K = 8 :=
by
  sorry

end katya_sold_glasses_l1714_171437


namespace ways_to_distribute_balls_in_boxes_l1714_171406

theorem ways_to_distribute_balls_in_boxes :
  ∃ (num_ways : ℕ), num_ways = 4 ^ 5 := sorry

end ways_to_distribute_balls_in_boxes_l1714_171406


namespace min_value_condition_l1714_171466

theorem min_value_condition
  (a b c d e f g h : ℝ)
  (h1 : a * b * c * d = 16)
  (h2 : e * f * g * h = 36) :
  ∃ x : ℝ, x = (ae)^2 + (bf)^2 + (cg)^2 + (dh)^2 ∧ x ≥ 576 := sorry

end min_value_condition_l1714_171466


namespace sum_of_squares_eq_zero_iff_all_zero_l1714_171417

theorem sum_of_squares_eq_zero_iff_all_zero (a b c : ℝ) :
  a^2 + b^2 + c^2 = 0 ↔ a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end sum_of_squares_eq_zero_iff_all_zero_l1714_171417


namespace largest_sphere_radius_on_torus_l1714_171411

theorem largest_sphere_radius_on_torus
  (inner_radius outer_radius : ℝ)
  (torus_center : ℝ × ℝ × ℝ)
  (circle_radius : ℝ)
  (sphere_radius : ℝ)
  (sphere_center : ℝ × ℝ × ℝ) :
  inner_radius = 3 →
  outer_radius = 5 →
  torus_center = (4, 0, 1) →
  circle_radius = 1 →
  sphere_center = (0, 0, sphere_radius) →
  sphere_radius = 4 :=
by
  intros h_inner_radius h_outer_radius h_torus_center h_circle_radius h_sphere_center
  sorry

end largest_sphere_radius_on_torus_l1714_171411


namespace students_in_two_courses_l1714_171473

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

end students_in_two_courses_l1714_171473


namespace time_to_cross_bridge_l1714_171431

def length_of_train : ℕ := 250
def length_of_bridge : ℕ := 150
def speed_in_kmhr : ℕ := 72
def speed_in_ms : ℕ := (speed_in_kmhr * 1000) / 3600

theorem time_to_cross_bridge : 
  (length_of_train + length_of_bridge) / speed_in_ms = 20 :=
by
  have total_distance := length_of_train + length_of_bridge
  have speed := speed_in_ms
  sorry

end time_to_cross_bridge_l1714_171431


namespace total_estate_value_l1714_171407

theorem total_estate_value 
  (estate : ℝ)
  (daughter_share son_share wife_share brother_share nanny_share : ℝ)
  (h1 : daughter_share + son_share = (3/5) * estate)
  (h2 : daughter_share = 5 * son_share / 2)
  (h3 : wife_share = 3 * son_share)
  (h4 : brother_share = daughter_share)
  (h5 : nanny_share = 400) :
  estate = 825 := by
  sorry

end total_estate_value_l1714_171407


namespace salted_duck_eggs_min_cost_l1714_171467

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

end salted_duck_eggs_min_cost_l1714_171467


namespace jerome_contact_list_l1714_171468

def classmates := 20
def out_of_school_friends := classmates / 2
def family_members := 3
def total_contacts := classmates + out_of_school_friends + family_members

theorem jerome_contact_list : total_contacts = 33 := by
  sorry

end jerome_contact_list_l1714_171468


namespace f_has_four_distinct_real_roots_l1714_171440

noncomputable def f (x d : ℝ) := x ^ 2 + 4 * x + d

theorem f_has_four_distinct_real_roots (d : ℝ) (h : d = 2) :
  ∃ r1 r2 r3 r4 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r3 ≠ r4 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r4 ∧ 
  f (f r1 d) = 0 ∧ f (f r2 d) = 0 ∧ f (f r3 d) = 0 ∧ f (f r4 d) = 0 :=
by
  sorry

end f_has_four_distinct_real_roots_l1714_171440


namespace probability_no_adjacent_green_hats_l1714_171409

-- Step d): Rewrite the math proof problem in a Lean 4 statement.

theorem probability_no_adjacent_green_hats (total_children green_hats : ℕ)
  (hc : total_children = 9) (hg : green_hats = 3) :
  (∃ (p : ℚ), p = 5 / 14) :=
sorry

end probability_no_adjacent_green_hats_l1714_171409


namespace ratio_of_speeds_l1714_171454

theorem ratio_of_speeds (va vb L : ℝ) (h1 : 0 < L) (h2 : 0 < va) (h3 : 0 < vb)
  (h4 : ∀ t : ℝ, t = L / va ↔ t = (L - 0.09523809523809523 * L) / vb) :
  va / vb = 21 / 19 :=
by
  sorry

end ratio_of_speeds_l1714_171454


namespace longest_side_of_enclosure_l1714_171428

theorem longest_side_of_enclosure (l w : ℝ) (hlw : 2*l + 2*w = 240) (harea : l*w = 2880) : max l w = 72 := 
by {
  sorry
}

end longest_side_of_enclosure_l1714_171428


namespace total_boys_slide_l1714_171456

theorem total_boys_slide (initial_boys additional_boys : ℕ) (h1 : initial_boys = 22) (h2 : additional_boys = 13) :
  initial_boys + additional_boys = 35 :=
by
  sorry

end total_boys_slide_l1714_171456


namespace condition_A_condition_B_condition_C_condition_D_correct_answer_l1714_171457

theorem condition_A : ∀ x : ℝ, x^2 + 2 * x - 1 ≠ x * (x + 2) - 1 := sorry

theorem condition_B : ∀ a b : ℝ, (a + b)^2 = a^2 + 2 * a * b + b^2 := sorry

theorem condition_C : ∀ x y : ℝ, x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y) := sorry

theorem condition_D : ∀ a b : ℝ, a^2 - a * b - a ≠ a * (a - b) := sorry

theorem correct_answer : ∀ x y : ℝ, (x^2 - 4 * y^2) = (x + 2 * y) * (x - 2 * y) := 
  by 
    exact condition_C

end condition_A_condition_B_condition_C_condition_D_correct_answer_l1714_171457


namespace ellipse_eq_and_line_eq_l1714_171455

theorem ellipse_eq_and_line_eq
  (e : ℝ) (a b c xC yC: ℝ)
  (h_e : e = (Real.sqrt 3 / 2))
  (h_a : a = 2)
  (h_c : c = Real.sqrt 3)
  (h_b : b = Real.sqrt (a^2 - c^2))
  (h_ellipse : ∀ x y, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 4 + y^2 = 1))
  (h_C_on_G : xC^2 / 4 + yC^2 = 1)
  (h_diameter_condition : ∀ (B : ℝ × ℝ), B = (0, 1) →
    ((2 * xC - yC + 1 = 0) →
    (xC = 0 ∧ yC = 1) ∨ (xC = -16 / 17 ∧ yC = -15 / 17)))
  : (∀ x y, (y = 2*x + 1) ↔ (x + 2*y - 2 = 0 ∨ 3*x - 10*y - 6 = 0)) :=
by
  sorry

end ellipse_eq_and_line_eq_l1714_171455


namespace friendP_walks_23_km_l1714_171486

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

end friendP_walks_23_km_l1714_171486


namespace find_circle_center_l1714_171478

noncomputable def circle_center_lemma (a b : ℝ) : Prop :=
  -- Condition: Circle passes through (1, 0)
  (a - 1)^2 + b^2 = (a - 1)^2 + (b - 0)^2 ∧
  -- Condition: Circle is tangent to the parabola y = x^2 at (1, 1)
  (a - 1)^2 + (b - 1)^2 = 0

theorem find_circle_center : ∃ a b : ℝ, circle_center_lemma a b ∧ a = 1 ∧ b = 1 :=
by
  sorry

end find_circle_center_l1714_171478


namespace trapezium_area_l1714_171405

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 17) : 
  (1 / 2 * (a + b) * h) = 323 :=
by
  have ha' : a = 20 := ha
  have hb' : b = 18 := hb
  have hh' : h = 17 := hh
  rw [ha', hb', hh']
  sorry

end trapezium_area_l1714_171405
