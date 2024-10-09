import Mathlib

namespace coupon_probability_l2024_202406

-- We will define our conditions
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Now we state our problem
theorem coupon_probability :
  ∀ (C6_6 C11_3 C17_9 : ℕ),
  C6_6 = combination 6 6 →
  C11_3 = combination 11 3 →
  C17_9 = combination 17 9 →
  (C6_6 * C11_3) / C17_9 = 3 / 442 :=
by
  intros C6_6 C11_3 C17_9 hC6_6 hC11_3 hC17_9
  rw [hC6_6, hC11_3, hC17_9]
  sorry

end coupon_probability_l2024_202406


namespace probability_of_70th_percentile_is_25_over_56_l2024_202428

-- Define the weights of the students
def weights : List ℕ := [90, 100, 110, 120, 140, 150, 150, 160]

-- Define the number of students to select
def n_selected_students : ℕ := 3

-- Define the percentile value
def percentile_value : ℕ := 70

-- Define the corresponding weight for the 70th percentile
def percentile_weight : ℕ := 150

-- Define the combination function
noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability calculation
noncomputable def probability_70th_percentile : ℚ :=
  let total_ways := C 8 3
  let favorable_ways := (C 2 2) * (C 5 1) + (C 2 1) * (C 5 2)
  favorable_ways / total_ways

-- Define the theorem to prove the probability
theorem probability_of_70th_percentile_is_25_over_56 :
  probability_70th_percentile = 25 / 56 := by
  sorry

end probability_of_70th_percentile_is_25_over_56_l2024_202428


namespace process_can_continue_indefinitely_l2024_202470

noncomputable def P (x : ℝ) : ℝ := x^3 - x^2 - x - 1

-- Assume the existence of t > 1 such that P(t) = 0
axiom exists_t : ∃ t : ℝ, t > 1 ∧ P t = 0

def triangle_inequality_fails (a b c : ℝ) : Prop :=
  ¬(a + b > c ∧ b + c > a ∧ c + a > b)

def shorten (a b : ℝ) : ℝ := a + b

def can_continue_indefinitely (a b c : ℝ) : Prop :=
  ∀ t, t > 0 → ∀ a b c, triangle_inequality_fails a b c → 
  (triangle_inequality_fails (shorten b c - shorten a b) b c ∧
   triangle_inequality_fails a (shorten a c - shorten b c) c ∧
   triangle_inequality_fails a b (shorten a b - shorten b c))

theorem process_can_continue_indefinitely (a b c : ℝ) (h : triangle_inequality_fails a b c) :
  can_continue_indefinitely a b c :=
sorry

end process_can_continue_indefinitely_l2024_202470


namespace no_solution_l2024_202490

theorem no_solution (x : ℝ) : ¬ (3 * x^2 + 9 * x ≤ -12) :=
sorry

end no_solution_l2024_202490


namespace smallest_palindrome_not_five_digit_l2024_202444

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let s := n.toDigits 10
  s = s.reverse

theorem smallest_palindrome_not_five_digit (n : ℕ) :
  (∃ n, is_palindrome n ∧ 100 ≤ n ∧ n < 1000 ∧ ¬is_palindrome (102 * n)) → n = 101 := by
  sorry

end smallest_palindrome_not_five_digit_l2024_202444


namespace ultratown_run_difference_l2024_202438

/-- In Ultratown, the streets are all 25 feet wide, 
and the blocks they enclose are rectangular with lengths of 500 feet and widths of 300 feet. 
Hannah runs around the block on the longer 500-foot side of the street, 
while Harry runs on the opposite, outward side of the street. 
Prove that Harry runs 200 more feet than Hannah does for every lap around the block.
-/ 
theorem ultratown_run_difference :
  let street_width : ℕ := 25
  let inner_length : ℕ := 500
  let inner_width : ℕ := 300
  let outer_length := inner_length + 2 * street_width
  let outer_width := inner_width + 2 * street_width
  let inner_perimeter := 2 * (inner_length + inner_width)
  let outer_perimeter := 2 * (outer_length + outer_width)
  (outer_perimeter - inner_perimeter) = 200 :=
by
  sorry

end ultratown_run_difference_l2024_202438


namespace total_questions_l2024_202407

theorem total_questions (S C I : ℕ) (h1 : S = 73) (h2 : C = 91) (h3 : S = C - 2 * I) : C + I = 100 :=
sorry

end total_questions_l2024_202407


namespace power_multiplication_eq_neg4_l2024_202441

theorem power_multiplication_eq_neg4 :
  (-0.25) ^ 11 * (-4) ^ 12 = -4 := 
  sorry

end power_multiplication_eq_neg4_l2024_202441


namespace rectangle_area_relation_l2024_202472

theorem rectangle_area_relation (x y : ℝ) (h : x * y = 4) (hx : x > 0) : y = 4 / x := 
sorry

end rectangle_area_relation_l2024_202472


namespace first_stack_height_is_seven_l2024_202423

-- Definitions of the conditions
def first_stack (h : ℕ) := h
def second_stack (h : ℕ) := h + 5
def third_stack (h : ℕ) := h + 12

-- Conditions on the blocks falling down
def blocks_fell_first_stack (h : ℕ) := h
def blocks_fell_second_stack (h : ℕ) := (h + 5) - 2
def blocks_fell_third_stack (h : ℕ) := (h + 12) - 3

-- Total blocks fell down
def total_blocks_fell (h : ℕ) := blocks_fell_first_stack h + blocks_fell_second_stack h + blocks_fell_third_stack h

-- Lean statement to prove the height of the first stack
theorem first_stack_height_is_seven (h : ℕ) (h_eq : total_blocks_fell h = 33) : h = 7 :=
by sorry

-- Testing the conditions hold for the solution h = 7
#eval total_blocks_fell 7 -- Expected: 33

end first_stack_height_is_seven_l2024_202423


namespace arithmetic_sequence_n_value_l2024_202465

theorem arithmetic_sequence_n_value (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 3) :
  a 672 = 2014 :=
sorry

end arithmetic_sequence_n_value_l2024_202465


namespace slope_of_line_inclination_angle_l2024_202462

theorem slope_of_line_inclination_angle 
  (k : ℝ) (θ : ℝ)
  (hθ1 : 30 * (π / 180) < θ)
  (hθ2 : θ < 90 * (π / 180)) :
  k = Real.tan θ → k > Real.tan (30 * (π / 180)) :=
by
  intro h
  sorry

end slope_of_line_inclination_angle_l2024_202462


namespace line_intersects_curve_equal_segments_l2024_202467

theorem line_intersects_curve_equal_segments (k m : ℝ)
  (A B C : ℝ × ℝ)
  (hA_curve : A.2 = A.1^3 - 6 * A.1^2 + 13 * A.1 - 8)
  (hB_curve : B.2 = B.1^3 - 6 * B.1^2 + 13 * B.1 - 8)
  (hC_curve : C.2 = C.1^3 - 6 * C.1^2 + 13 * C.1 - 8)
  (h_lineA : A.2 = k * A.1 + m)
  (h_lineB : B.2 = k * B.1 + m)
  (h_lineC : C.2 = k * C.1 + m)
  (h_midpoint : 2 * B.1 = A.1 + C.1 ∧ 2 * B.2 = A.2 + C.2)
  : 2 * k + m = 2 :=
sorry

end line_intersects_curve_equal_segments_l2024_202467


namespace positive_integer_solutions_of_inequality_l2024_202427

theorem positive_integer_solutions_of_inequality : 
  {x : ℕ | 3 * x - 1 ≤ 2 * x + 3} = {1, 2, 3, 4} :=
by
  sorry

end positive_integer_solutions_of_inequality_l2024_202427


namespace exists_positive_integer_pow_not_integer_l2024_202466

theorem exists_positive_integer_pow_not_integer
  (α β : ℝ)
  (hαβ : α ≠ β)
  (h_non_int : ¬(↑⌊α⌋ = α ∧ ↑⌊β⌋ = β)) :
  ∃ n : ℕ, 0 < n ∧ ¬∃ k : ℤ, α^n - β^n = k :=
by
  sorry

end exists_positive_integer_pow_not_integer_l2024_202466


namespace boxes_left_l2024_202453

theorem boxes_left (boxes_saturday boxes_sunday apples_per_box apples_sold : ℕ)
  (h_saturday : boxes_saturday = 50)
  (h_sunday : boxes_sunday = 25)
  (h_apples_per_box : apples_per_box = 10)
  (h_apples_sold : apples_sold = 720) :
  ((boxes_saturday + boxes_sunday) * apples_per_box - apples_sold) / apples_per_box = 3 :=
by
  sorry

end boxes_left_l2024_202453


namespace john_paid_after_tax_l2024_202411

-- Definitions based on problem conditions
def original_cost : ℝ := 200
def tax_rate : ℝ := 0.15

-- Definition of the tax amount
def tax_amount : ℝ := tax_rate * original_cost

-- Definition of the total amount paid
def total_amount_paid : ℝ := original_cost + tax_amount

-- Theorem statement for the proof
theorem john_paid_after_tax : total_amount_paid = 230 := by
  sorry

end john_paid_after_tax_l2024_202411


namespace ratio_steel_to_tin_l2024_202433

def mass_copper (C : ℝ) := C = 90
def total_weight (S C T : ℝ) := 20 * S + 20 * C + 20 * T = 5100
def mass_steel (S C : ℝ) := S = C + 20

theorem ratio_steel_to_tin (S T C : ℝ)
  (hC : mass_copper C)
  (hTW : total_weight S C T)
  (hS : mass_steel S C) :
  S / T = 2 :=
by
  sorry

end ratio_steel_to_tin_l2024_202433


namespace calc_f_g_3_minus_g_f_3_l2024_202455

def f (x : ℝ) : ℝ := 2 * x + 5
def g (x : ℝ) : ℝ := x^2 + 2

theorem calc_f_g_3_minus_g_f_3 :
  (f (g 3) - g (f 3)) = -96 :=
by
  sorry

end calc_f_g_3_minus_g_f_3_l2024_202455


namespace side_length_estimate_l2024_202422

theorem side_length_estimate (x : ℝ) (h : x^2 = 15) : 3 < x ∧ x < 4 :=
sorry

end side_length_estimate_l2024_202422


namespace emily_small_gardens_l2024_202469

theorem emily_small_gardens 
  (total_seeds : ℕ)
  (seeds_in_big_garden : ℕ)
  (seeds_per_small_garden : ℕ)
  (remaining_seeds := total_seeds - seeds_in_big_garden)
  (number_of_small_gardens := remaining_seeds / seeds_per_small_garden) :
  total_seeds = 41 → seeds_in_big_garden = 29 → seeds_per_small_garden = 4 → number_of_small_gardens = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end emily_small_gardens_l2024_202469


namespace pump1_half_drain_time_l2024_202410

-- Definitions and Conditions
def time_to_drain_half_pump1 (t : ℝ) : Prop :=
  ∃ rate1 rate2 : ℝ, 
    rate1 = 1 / (2 * t) ∧
    rate2 = 1 / 1.25 ∧
    rate1 + rate2 = 2

-- Equivalent Proof Problem
theorem pump1_half_drain_time (t : ℝ) : time_to_drain_half_pump1 t → t = 5 / 12 := sorry

end pump1_half_drain_time_l2024_202410


namespace range_of_m_l2024_202408

theorem range_of_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) (hineq : 4 / (x + 1) + 1 / y < m^2 + (3 / 2) * m) :
  m < -3 ∨ m > 3 / 2 :=
by sorry

end range_of_m_l2024_202408


namespace find_t_l2024_202424

theorem find_t (t : ℤ) :
  ((t + 1) * (3 * t - 3)) = ((3 * t - 5) * (t + 2) + 2) → 
  t = 5 :=
by
  intros
  sorry

end find_t_l2024_202424


namespace squirrel_travel_distance_l2024_202405

def squirrel_distance (height : ℕ) (circumference : ℕ) (rise_per_circuit : ℕ) : ℕ :=
  let circuits := height / rise_per_circuit
  let horizontal_distance := circuits * circumference
  Nat.sqrt (height * height + horizontal_distance * horizontal_distance)

theorem squirrel_travel_distance :
  (squirrel_distance 16 3 4) = 20 := by
  sorry

end squirrel_travel_distance_l2024_202405


namespace number_of_crocodiles_l2024_202474

theorem number_of_crocodiles
  (f : ℕ) -- number of frogs
  (c : ℕ) -- number of crocodiles
  (total_eyes : ℕ) -- total number of eyes
  (frog_eyes : ℕ) -- number of eyes per frog
  (croc_eyes : ℕ) -- number of eyes per crocodile
  (h_f : f = 20) -- condition: there are 20 frogs
  (h_total_eyes : total_eyes = 52) -- condition: total number of eyes is 52
  (h_frog_eyes : frog_eyes = 2) -- condition: each frog has 2 eyes
  (h_croc_eyes : croc_eyes = 2) -- condition: each crocodile has 2 eyes
  :
  c = 6 := -- proof goal: number of crocodiles is 6
by
  sorry

end number_of_crocodiles_l2024_202474


namespace tip_percentage_l2024_202463

variable (L : ℝ) (T : ℝ)
 
theorem tip_percentage (h : L = 60.50) (h1 : T = 72.6) :
  ((T - L) / L) * 100 = 20 :=
by
  sorry

end tip_percentage_l2024_202463


namespace houses_without_garage_nor_pool_l2024_202446

def total_houses : ℕ := 85
def houses_with_garage : ℕ := 50
def houses_with_pool : ℕ := 40
def houses_with_both : ℕ := 35
def neither_garage_nor_pool : ℕ := 30

theorem houses_without_garage_nor_pool :
  total_houses - (houses_with_garage + houses_with_pool - houses_with_both) = neither_garage_nor_pool :=
by
  sorry

end houses_without_garage_nor_pool_l2024_202446


namespace value_of_x_l2024_202487

theorem value_of_x (y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 := by
  sorry

end value_of_x_l2024_202487


namespace sum_of_midpoints_of_triangle_l2024_202464

theorem sum_of_midpoints_of_triangle (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_of_triangle_l2024_202464


namespace find_D_l2024_202495

-- Definitions
def divides (a b : ℕ) : Prop := ∃ k, b = a * k
def remainder (a b r : ℕ) : Prop := ∃ k, a = b * k + r

-- Problem Statement
theorem find_D {N D : ℕ} (h1 : remainder N D 75) (h2 : remainder N 37 1) : 
  D = 112 :=
by
  sorry

end find_D_l2024_202495


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l2024_202415

def is_composite (n : Nat) : Prop := n > 1 ∧ ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem exists_nine_consecutive_composites :
  ∃ (a : Nat), (a ≥ 1 ∧ a + 8 ≤ 500) ∧ ∀ i ∈ (List.range 9), is_composite (a + i) :=
sorry

theorem exists_eleven_consecutive_composites :
  ∃ (a : Nat), (a ≥ 1 ∧ a + 10 ≤ 500) ∧ ∀ i ∈ (List.range 11), is_composite (a + i) :=
sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l2024_202415


namespace distinct_midpoints_at_least_2n_minus_3_l2024_202429

open Set

theorem distinct_midpoints_at_least_2n_minus_3 
  (n : ℕ) 
  (points : Finset (ℝ × ℝ)) 
  (h_points_card : points.card = n) :
  ∃ (midpoints : Finset (ℝ × ℝ)), 
    midpoints.card ≥ 2 * n - 3 := 
sorry

end distinct_midpoints_at_least_2n_minus_3_l2024_202429


namespace present_population_l2024_202425

theorem present_population (P : ℝ)
  (h1 : P + 0.10 * P = 242) :
  P = 220 := 
sorry

end present_population_l2024_202425


namespace factorization_correct_l2024_202493

theorem factorization_correct (x : ℝ) : x^2 - 6*x + 9 = (x - 3)^2 :=
by
  sorry

end factorization_correct_l2024_202493


namespace hilltop_high_students_l2024_202432

theorem hilltop_high_students : 
  ∀ (n_sophomore n_freshman n_junior : ℕ), 
  (n_sophomore : ℚ) / n_freshman = 7 / 4 ∧ (n_junior : ℚ) / n_sophomore = 6 / 7 → 
  n_sophomore + n_freshman + n_junior = 17 :=
by
  sorry

end hilltop_high_students_l2024_202432


namespace max_writers_and_editors_l2024_202496

theorem max_writers_and_editors (total people writers editors y x : ℕ) 
  (h1 : total = 110) 
  (h2 : writers = 45) 
  (h3 : editors = 38 + y) 
  (h4 : y > 0) 
  (h5 : 45 + editors + 2 * x = 110) : 
  x = 13 := 
sorry

end max_writers_and_editors_l2024_202496


namespace solution_set_inequality_l2024_202483

theorem solution_set_inequality (x : ℝ) (h : x - 3 / x > 2) :
    -1 < x ∧ x < 0 ∨ x > 3 :=
  sorry

end solution_set_inequality_l2024_202483


namespace regular_polygon_sides_l2024_202484

theorem regular_polygon_sides (perimeter side_length : ℝ) (h1 : perimeter = 180) (h2 : side_length = 15) :
  perimeter / side_length = 12 :=
by sorry

end regular_polygon_sides_l2024_202484


namespace polynomial_divisibility_l2024_202409

theorem polynomial_divisibility (a b c : ℝ) :
  (∀ x, (x-1)^3 ∣ x^4 + a * x^2 + b * x + c) ↔ (a = -6 ∧ b = 8 ∧ c = -3) :=
by
  sorry

end polynomial_divisibility_l2024_202409


namespace johns_old_cards_l2024_202460

def cards_per_page : ℕ := 3
def new_cards : ℕ := 8
def total_pages : ℕ := 8

def total_cards := total_pages * cards_per_page
def old_cards := total_cards - new_cards

theorem johns_old_cards :
  old_cards = 16 :=
by
  -- Note: No specific solution steps needed here, just stating the theorem
  sorry

end johns_old_cards_l2024_202460


namespace P_div_by_Q_iff_l2024_202430

def P (x : ℂ) (n : ℕ) : ℂ := x^(4*n) + x^(3*n) + x^(2*n) + x^n + 1
def Q (x : ℂ) : ℂ := x^4 + x^3 + x^2 + x + 1

theorem P_div_by_Q_iff (n : ℕ) : (Q x ∣ P x n) ↔ ¬(5 ∣ n) := sorry

end P_div_by_Q_iff_l2024_202430


namespace choir_final_score_l2024_202497

theorem choir_final_score (content_score sing_score spirit_score : ℕ)
  (content_weight sing_weight spirit_weight : ℝ)
  (h_content : content_weight = 0.30) 
  (h_sing : sing_weight = 0.50) 
  (h_spirit : spirit_weight = 0.20) 
  (h_content_score : content_score = 90)
  (h_sing_score : sing_score = 94)
  (h_spirit_score : spirit_score = 95) :
  content_weight * content_score + sing_weight * sing_score + spirit_weight * spirit_score = 93 := by
  sorry

end choir_final_score_l2024_202497


namespace num_of_loads_l2024_202434

theorem num_of_loads (n : ℕ) (h1 : 7 * n = 42) : n = 6 :=
by
  sorry

end num_of_loads_l2024_202434


namespace complex_sum_equals_one_l2024_202448

noncomputable def main (x : ℂ) (h1 : x^7 = 1) (h2 : x ≠ 1) : ℂ :=
  (x^2 / (x - 1)) + (x^4 / (x^2 - 1)) + (x^6 / (x^3 - 1))

theorem complex_sum_equals_one (x : ℂ) (h1 : x^7 = 1) (h2 : x ≠ 1) : main x h1 h2 = 1 := by
  sorry

end complex_sum_equals_one_l2024_202448


namespace inequality_problem_l2024_202431

theorem inequality_problem
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ (3 / 2) :=
sorry

end inequality_problem_l2024_202431


namespace smallest_x_value_l2024_202418

theorem smallest_x_value {x : ℝ} (h : abs (x + 4) = 15) : x = -19 :=
sorry

end smallest_x_value_l2024_202418


namespace vegetables_in_one_serving_l2024_202440

theorem vegetables_in_one_serving
  (V : ℝ)
  (H1 : ∀ servings : ℝ, servings > 0 → servings * (V + 2.5) = 28)
  (H_pints_to_cups : 14 * 2 = 28) :
  V = 1 :=
by
  -- proof steps would go here
  sorry

end vegetables_in_one_serving_l2024_202440


namespace dice_probability_l2024_202449

/-- A standard six-sided die -/
inductive Die : Type
| one | two | three | four | five | six

open Die

/-- Calculates the probability that after re-rolling four dice, at least four out of the six total dice show the same number,
given that initially six dice are rolled and there is no three-of-a-kind, and there is a pair of dice showing the same number
which are then set aside before re-rolling the remaining four dice. -/
theorem dice_probability (h1 : ∀ (d1 d2 d3 d4 d5 d6 : Die), 
  ¬ (d1 = d2 ∧ d2 = d3 ∨ d1 = d2 ∧ d2 = d4 ∨ d1 = d2 ∧ d2 = d5 ∨
     d1 = d2 ∧ d2 = d6 ∨ d1 = d3 ∧ d3 = d4 ∨ d1 = d3 ∧ d3 = d5 ∨
     d1 = d3 ∧ d3 = d6 ∨ d1 = d4 ∧ d4 = d5 ∨ d1 = d4 ∧ d4 = d6 ∨
     d1 = d5 ∧ d5 = d6 ∨ d2 = d3 ∧ d3 = d4 ∨ d2 = d3 ∧ d3 = d5 ∨
     d2 = d3 ∧ d3 = d6 ∨ d2 = d4 ∧ d4 = d5 ∨ d2 = d4 ∧ d4 = d6 ∨
     d2 = d5 ∧ d5 = d6 ∨ d3 = d4 ∧ d4 = d5 ∨ d3 = d4 ∧ d4 = d6 ∨ d3 = d5 ∧ d5 = d6 ∨ d4 = d5 ∧ d5 = d6))
    (h2 : ∃ (d1 d2 : Die) (d3 d4 d5 d6 : Die), d1 = d2 ∧ d3 ≠ d1 ∧ d4 ≠ d1 ∧ d5 ≠ d1 ∧ d6 ≠ d1): 
    ℚ := 
11 / 81

end dice_probability_l2024_202449


namespace solve_inequality_l2024_202416

theorem solve_inequality :
  { x : ℝ | (9 * x^2 + 27 * x - 64) / ((3 * x - 4) * (x + 5) * (x - 1)) < 4 } = 
    { x : ℝ | -5 < x ∧ x < -17 / 3 } ∪ { x : ℝ | 1 < x ∧ x < 4 } :=
by
  sorry

end solve_inequality_l2024_202416


namespace first_thrilling_thursday_after_start_l2024_202439

theorem first_thrilling_thursday_after_start (start_date : ℕ) (school_start_month : ℕ) (school_start_day_of_week : ℤ) (month_length : ℕ → ℕ) (day_of_week_on_first_of_month : ℕ → ℤ) : 
    school_start_month = 9 ∧ school_start_day_of_week = 2 ∧ start_date = 12 ∧ month_length 9 = 30 ∧ day_of_week_on_first_of_month 10 = 0 → 
    ∃ day_of_thursday : ℕ, day_of_thursday = 26 :=
by
  sorry

end first_thrilling_thursday_after_start_l2024_202439


namespace regular_polygon_sides_l2024_202454

theorem regular_polygon_sides (θ : ℝ) (h : θ = 20) : 360 / θ = 18 := by
  sorry

end regular_polygon_sides_l2024_202454


namespace Daria_money_l2024_202476

theorem Daria_money (num_tickets : ℕ) (price_per_ticket : ℕ) (amount_needed : ℕ) (h1 : num_tickets = 4) (h2 : price_per_ticket = 90) (h3 : amount_needed = 171) : 
  (num_tickets * price_per_ticket) - amount_needed = 189 := 
by 
  sorry

end Daria_money_l2024_202476


namespace amount_subtracted_is_30_l2024_202421

-- Definitions based on conditions
def N : ℕ := 200
def subtracted_amount (A : ℕ) : Prop := 0.40 * (N : ℝ) - (A : ℝ) = 50

-- The theorem statement
theorem amount_subtracted_is_30 : subtracted_amount 30 :=
by 
  -- proof will be completed here
  sorry

end amount_subtracted_is_30_l2024_202421


namespace anne_total_bottle_caps_l2024_202414

def initial_bottle_caps_anne : ℕ := 10
def found_bottle_caps_anne : ℕ := 5

theorem anne_total_bottle_caps : initial_bottle_caps_anne + found_bottle_caps_anne = 15 := 
by
  sorry

end anne_total_bottle_caps_l2024_202414


namespace shaded_area_of_overlap_l2024_202480

structure Rectangle where
  width : ℕ
  height : ℕ

structure Parallelogram where
  base : ℕ
  height : ℕ

def area_of_rectangle (r : Rectangle) : ℕ :=
  r.width * r.height

def area_of_parallelogram (p : Parallelogram) : ℕ :=
  p.base * p.height

def overlapping_area_square (side : ℕ) : ℕ :=
  side * side

theorem shaded_area_of_overlap 
  (r : Rectangle)
  (p : Parallelogram)
  (overlapping_side : ℕ)
  (h1 : r.width = 4)
  (h2 : r.height = 12)
  (h3 : p.base = 10)
  (h4 : p.height = 4)
  (h5 : overlapping_side = 4) :
  area_of_rectangle r + area_of_parallelogram p - overlapping_area_square overlapping_side = 72 :=
by
  sorry

end shaded_area_of_overlap_l2024_202480


namespace sector_area_correct_l2024_202400

noncomputable def sector_area (θ r : ℝ) : ℝ :=
  (θ / (2 * Real.pi)) * (Real.pi * r^2)

theorem sector_area_correct : 
  sector_area (Real.pi / 3) 3 = (3 / 2) * Real.pi :=
by
  sorry

end sector_area_correct_l2024_202400


namespace opposite_of_neg_2_l2024_202471

noncomputable def opposite (a : ℤ) : ℤ := 
  a * (-1)

theorem opposite_of_neg_2 : opposite (-2) = 2 := by
  -- definition of opposite
  unfold opposite
  -- calculation using the definition
  rfl

end opposite_of_neg_2_l2024_202471


namespace probability_at_least_one_six_is_11_div_36_l2024_202450

noncomputable def probability_at_least_one_six : ℚ :=
  let total_outcomes := 36
  let no_six_outcomes := 25
  let favorable_outcomes := total_outcomes - no_six_outcomes
  favorable_outcomes / total_outcomes
  
theorem probability_at_least_one_six_is_11_div_36 : 
  probability_at_least_one_six = 11 / 36 :=
by
  sorry

end probability_at_least_one_six_is_11_div_36_l2024_202450


namespace exist_non_zero_function_iff_sum_zero_l2024_202498

theorem exist_non_zero_function_iff_sum_zero (a b c : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x y z : ℝ, a * f (x * y + f z) + b * f (y * z + f x) + c * f (z * x + f y) = 0) ∧ ¬ (∀ x : ℝ, f x = 0)) ↔ (a + b + c = 0) :=
by {
  sorry
}

end exist_non_zero_function_iff_sum_zero_l2024_202498


namespace simplify_expression_l2024_202481

theorem simplify_expression (y : ℝ) : y - 3 * (2 + y) + 4 * (2 - y) - 5 * (2 + 3 * y) = -21 * y - 8 :=
sorry

end simplify_expression_l2024_202481


namespace seq_general_term_l2024_202491

noncomputable def seq (n : ℕ) : ℚ :=
  if n = 0 then 1/2
  else if n = 1 then 1/2
  else seq (n - 1) * 3 / (seq (n - 1) + 3)

theorem seq_general_term : ∀ n : ℕ, seq (n + 1) = 3 / (n + 6) :=
by
  intro n
  induction n with
  | zero => sorry
  | succ k ih => sorry

end seq_general_term_l2024_202491


namespace log_216_eq_3_log_2_add_3_log_3_l2024_202445

theorem log_216_eq_3_log_2_add_3_log_3 (log : ℝ → ℝ) (h1 : ∀ x y, log (x * y) = log x + log y)
  (h2 : ∀ x n, log (x^n) = n * log x) :
  log 216 = 3 * log 2 + 3 * log 3 :=
by
  sorry

end log_216_eq_3_log_2_add_3_log_3_l2024_202445


namespace parallel_perpendicular_implies_l2024_202443

variables {Line : Type} {Plane : Type}
variables (m n : Line) (α β : Plane)

-- Conditions
axiom distinct_lines : m ≠ n
axiom distinct_planes : α ≠ β

-- Parallel and Perpendicular relationships
axiom parallel : Line → Plane → Prop
axiom perpendicular : Line → Plane → Prop

-- Given conditions
axiom parallel_mn : parallel m n
axiom perpendicular_mα : perpendicular m α

-- Proof statement
theorem parallel_perpendicular_implies (h1 : parallel m n) (h2 : perpendicular m α) : perpendicular n α :=
sorry

end parallel_perpendicular_implies_l2024_202443


namespace expression_value_l2024_202475

theorem expression_value (x y z : ℕ) (hx : x = 5) (hy : y = 4) (hz : z = 3) :
  ( (1 / (y : ℚ)) + (1 / (z : ℚ))) / (1 / (x : ℚ)) = 35 / 12 := by
  sorry

end expression_value_l2024_202475


namespace sqrt_ceil_eq_sqrt_sqrt_l2024_202492

theorem sqrt_ceil_eq_sqrt_sqrt (a : ℝ) (h : a > 1) : 
  (Int.floor (Real.sqrt (Int.floor (Real.sqrt a)))) = (Int.floor (Real.sqrt (Real.sqrt a))) :=
sorry

end sqrt_ceil_eq_sqrt_sqrt_l2024_202492


namespace contrapositive_of_not_p_implies_q_l2024_202488

variable (p q : Prop)

theorem contrapositive_of_not_p_implies_q :
  (¬p → q) → (¬q → p) := by
  sorry

end contrapositive_of_not_p_implies_q_l2024_202488


namespace sum_of_reciprocals_l2024_202473

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 5 * x * y) : 
  (1/x) + (1/y) = 5 :=
by
  sorry

end sum_of_reciprocals_l2024_202473


namespace M_gt_N_l2024_202482

-- Define M and N
def M (x y : ℝ) : ℝ := x^2 + y^2 + 1
def N (x y : ℝ) : ℝ := 2 * (x + y - 1)

-- State the theorem to prove M > N given the conditions
theorem M_gt_N (x y : ℝ) : M x y > N x y := by
  sorry

end M_gt_N_l2024_202482


namespace julian_needs_more_legos_l2024_202426

-- Definitions based on the conditions
def legos_julian_has := 400
def legos_per_airplane := 240
def number_of_airplanes := 2

-- Calculate the total number of legos required for two airplane models
def total_legos_needed := legos_per_airplane * number_of_airplanes

-- Calculate the number of additional legos Julian needs
def additional_legos_needed := total_legos_needed - legos_julian_has

-- Statement that needs to be proven
theorem julian_needs_more_legos : additional_legos_needed = 80 := by
  sorry

end julian_needs_more_legos_l2024_202426


namespace weight_triangle_correct_weight_l2024_202457

noncomputable def area_square (side : ℝ) : ℝ := side ^ 2

noncomputable def area_triangle (side : ℝ) : ℝ := (side ^ 2 * Real.sqrt 3) / 4

noncomputable def weight (area : ℝ) (density : ℝ) := area * density

noncomputable def weight_equilateral_triangle (weight_square : ℝ) (side_square : ℝ) (side_triangle : ℝ) : ℝ :=
  let area_s := area_square side_square
  let area_t := area_triangle side_triangle
  let density := weight_square / area_s
  weight area_t density

theorem weight_triangle_correct_weight :
  weight_equilateral_triangle 8 4 6 = 9 * Real.sqrt 3 / 2 := by sorry

end weight_triangle_correct_weight_l2024_202457


namespace length_of_arc_correct_l2024_202447

open Real

noncomputable def length_of_arc (r θ : ℝ) := θ * r

theorem length_of_arc_correct (A r θ : ℝ) (hA : A = (θ / (2 * π)) * (π * r^2)) (hr : r = 5) (hA_val : A = 13.75) :
  length_of_arc r θ = 5.5 :=
by
  -- Proof steps are omitted
  sorry

end length_of_arc_correct_l2024_202447


namespace jars_of_pickled_mangoes_l2024_202403

def total_mangoes := 54
def ratio_ripe := 1/3
def ratio_unripe := 2/3
def kept_unripe_mangoes := 16
def mangoes_per_jar := 4

theorem jars_of_pickled_mangoes : 
  (total_mangoes * ratio_unripe - kept_unripe_mangoes) / mangoes_per_jar = 5 :=
by
  sorry

end jars_of_pickled_mangoes_l2024_202403


namespace total_amount_paid_l2024_202468

/-- The owner's markup percentage and the cost price are given. 
We need to find out the total amount paid by the customer, which is equivalent to proving the total cost. -/
theorem total_amount_paid (markup_percentage : ℝ) (cost_price : ℝ) (markup : ℝ) (total_paid : ℝ) 
    (h1 : markup_percentage = 0.24) 
    (h2 : cost_price = 6425) 
    (h3 : markup = markup_percentage * cost_price) 
    (h4 : total_paid = cost_price + markup) : 
    total_paid = 7967 := 
sorry

end total_amount_paid_l2024_202468


namespace all_real_K_have_real_roots_l2024_202419

noncomputable def quadratic_discriminant (K : ℝ) : ℝ :=
  let a := K ^ 3
  let b := -(4 * K ^ 3 + 1)
  let c := 3 * K ^ 3
  b ^ 2 - 4 * a * c

theorem all_real_K_have_real_roots : ∀ K : ℝ, quadratic_discriminant K ≥ 0 :=
by
  sorry

end all_real_K_have_real_roots_l2024_202419


namespace students_received_B_l2024_202435

theorem students_received_B (charles_ratio : ℚ) (dawsons_class : ℕ) 
  (h_charles_ratio : charles_ratio = 3 / 5) (h_dawsons_class : dawsons_class = 30) : 
  ∃ y : ℕ, (charles_ratio = y / dawsons_class) ∧ y = 18 := 
by 
  sorry

end students_received_B_l2024_202435


namespace sum_a_for_exactly_one_solution_l2024_202485

theorem sum_a_for_exactly_one_solution :
  (∀ a : ℝ, ∃ x : ℝ, 3 * x^2 + (a + 6) * x + 7 = 0) →
  ((-6 + 2 * Real.sqrt 21) + (-6 - 2 * Real.sqrt 21) = -12) :=
by
  sorry

end sum_a_for_exactly_one_solution_l2024_202485


namespace car_pass_time_l2024_202494

theorem car_pass_time (length : ℝ) (speed_kmph : ℝ) (speed_mps : ℝ) (time : ℝ) :
  length = 10 → 
  speed_kmph = 36 → 
  speed_mps = speed_kmph * (1000 / 3600) → 
  time = length / speed_mps → 
  time = 1 :=
by
  intros h_length h_speed_kmph h_speed_conversion h_time_calculation
  -- Here we would normally construct the proof
  sorry

end car_pass_time_l2024_202494


namespace min_distance_sum_l2024_202477

theorem min_distance_sum
  (A B C D E P : ℝ)
  (h_collinear : B = A + 2 ∧ C = B + 2 ∧ D = C + 3 ∧ E = D + 4)
  (h_bisector : P = (A + E) / 2) :
  (A - P)^2 + (B - P)^2 + (C - P)^2 + (D - P)^2 + (E - P)^2 = 77.25 :=
by
  sorry

end min_distance_sum_l2024_202477


namespace entree_cost_difference_l2024_202489

theorem entree_cost_difference 
  (total_cost : ℕ)
  (entree_cost : ℕ)
  (dessert_cost : ℕ)
  (h1 : total_cost = 23)
  (h2 : entree_cost = 14)
  (h3 : total_cost = entree_cost + dessert_cost) :
  entree_cost - dessert_cost = 5 :=
by
  sorry

end entree_cost_difference_l2024_202489


namespace calculation_correct_l2024_202412

theorem calculation_correct :
  (-1 : ℝ)^51 + (2 : ℝ)^(4^2 + 5^2 - 7^2) = -(127 / 128) := 
by
  sorry

end calculation_correct_l2024_202412


namespace simplify_expression_l2024_202452

theorem simplify_expression :
  (64^(1/3) - 216^(1/3) = -2) :=
by
  have h1 : 64 = 4^3 := by norm_num
  have h2 : 216 = 6^3 := by norm_num
  sorry

end simplify_expression_l2024_202452


namespace min_satisfies_condition_only_for_x_eq_1_div_4_l2024_202451

theorem min_satisfies_condition_only_for_x_eq_1_div_4 (x : ℝ) (hx_nonneg : 0 ≤ x) :
  (min (Real.sqrt x) (min (x^2) x) = 1/16) ↔ (x = 1/4) :=
by sorry

end min_satisfies_condition_only_for_x_eq_1_div_4_l2024_202451


namespace right_triangle_acute_angle_l2024_202461

theorem right_triangle_acute_angle (a b : ℝ) (h1 : a + b = 90) (h2 : a = 55) : b = 35 := 
by sorry

end right_triangle_acute_angle_l2024_202461


namespace author_hardcover_percentage_l2024_202413

variable {TotalPaperCopies : Nat}
variable {PricePerPaperCopy : ℝ}
variable {TotalHardcoverCopies : Nat}
variable {PricePerHardcoverCopy : ℝ}
variable {PaperPercentage : ℝ}
variable {TotalEarnings : ℝ}

theorem author_hardcover_percentage (TotalPaperCopies : Nat)
  (PricePerPaperCopy : ℝ) (TotalHardcoverCopies : Nat)
  (PricePerHardcoverCopy : ℝ) (PaperPercentage TotalEarnings : ℝ)
  (h1 : TotalPaperCopies = 32000) (h2 : PricePerPaperCopy = 0.20)
  (h3 : TotalHardcoverCopies = 15000) (h4 : PricePerHardcoverCopy = 0.40)
  (h5 : PaperPercentage = 0.06) (h6 : TotalEarnings = 1104) :
  (720 / (15000 * 0.40) * 100) = 12 := by
  sorry

end author_hardcover_percentage_l2024_202413


namespace cuboid_volume_l2024_202404

theorem cuboid_volume (a b c : ℝ) (h1 : a * b = 2) (h2 : b * c = 6) (h3 : a * c = 9) : a * b * c = 6 := by
  sorry

end cuboid_volume_l2024_202404


namespace janna_sleep_hours_l2024_202417

-- Define the sleep hours from Monday to Sunday with the specified conditions
def sleep_hours_monday : ℕ := 7
def sleep_hours_tuesday : ℕ := 7 + 1 / 2
def sleep_hours_wednesday : ℕ := 7
def sleep_hours_thursday : ℕ := 7 + 1 / 2
def sleep_hours_friday : ℕ := 7 + 1
def sleep_hours_saturday : ℕ := 8
def sleep_hours_sunday : ℕ := 8

-- Calculate the total sleep hours in a week
noncomputable def total_sleep_hours : ℕ :=
  sleep_hours_monday +
  sleep_hours_tuesday +
  sleep_hours_wednesday +
  sleep_hours_thursday +
  sleep_hours_friday +
  sleep_hours_saturday +
  sleep_hours_sunday

-- The statement we want to prove
theorem janna_sleep_hours : total_sleep_hours = 53 := by
  sorry

end janna_sleep_hours_l2024_202417


namespace sum_of_rel_prime_ints_l2024_202456

theorem sum_of_rel_prime_ints (a b : ℕ) (h1 : a < 15) (h2 : b < 15) (h3 : a * b + a + b = 71)
    (h4 : Nat.gcd a b = 1) : a + b = 16 := by
  sorry

end sum_of_rel_prime_ints_l2024_202456


namespace x_cubed_gt_y_squared_l2024_202437

theorem x_cubed_gt_y_squared (x y : ℝ) (h1 : x^5 > y^4) (h2 : y^5 > x^4) : x^3 > y^2 := by
  sorry

end x_cubed_gt_y_squared_l2024_202437


namespace area_of_square_efgh_proof_l2024_202458

noncomputable def area_of_square_efgh : ℝ :=
  let original_square_side_length := 3
  let radius_of_circles := (3 * Real.sqrt 2) / 2
  let efgh_side_length := original_square_side_length + 2 * radius_of_circles 
  efgh_side_length ^ 2

theorem area_of_square_efgh_proof :
  area_of_square_efgh = 27 + 18 * Real.sqrt 2 :=
by
  sorry

end area_of_square_efgh_proof_l2024_202458


namespace election_valid_votes_l2024_202420

variable (V : ℕ)
variable (invalid_pct : ℝ)
variable (exceed_pct : ℝ)
variable (total_votes : ℕ)
variable (invalid_votes : ℝ)
variable (valid_votes : ℕ)
variable (A_votes : ℕ)
variable (B_votes : ℕ)

theorem election_valid_votes :
  V = 9720 →
  invalid_pct = 0.20 →
  exceed_pct = 0.15 →
  total_votes = V →
  invalid_votes = invalid_pct * V →
  valid_votes = total_votes - invalid_votes →
  A_votes = B_votes + exceed_pct * total_votes →
  A_votes + B_votes = valid_votes →
  B_votes = 3159 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end election_valid_votes_l2024_202420


namespace quadratic_equation_value_l2024_202442

theorem quadratic_equation_value (a : ℝ) (h₁ : a^2 - 2 = 2) (h₂ : a ≠ 2) : a = -2 :=
by
  sorry

end quadratic_equation_value_l2024_202442


namespace minimum_value_expression_l2024_202478

open Real

theorem minimum_value_expression : ∃ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019 = 2018 := 
sorry

end minimum_value_expression_l2024_202478


namespace ellipse_centroid_locus_l2024_202401

noncomputable def ellipse_equation (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1
noncomputable def centroid_locus (x y : ℝ) : Prop := (9 * x^2) / 4 + 3 * y^2 = 1 ∧ y ≠ 0

theorem ellipse_centroid_locus (x y : ℝ) (h : ellipse_equation x y) : centroid_locus (x / 3) (y / 3) :=
  sorry

end ellipse_centroid_locus_l2024_202401


namespace problem_statement_l2024_202479

theorem problem_statement (m : ℝ) (h_m : 0 ≤ m ∧ m ≤ 1) (x : ℝ) :
    (m * x^2 - 2 * x - m ≥ 2) ↔ (x ≤ -1) := sorry

end problem_statement_l2024_202479


namespace find_C_D_l2024_202499

theorem find_C_D (x C D : ℚ) 
  (h : 7 * x - 5 ≠ 0) -- Added condition to avoid zero denominator
  (hx : x^2 - 8 * x - 48 = (x - 12) * (x + 4))
  (h_eq : 7 * x - 5 = C * (x + 4) + D * (x - 12))
  (h_c : C = 79 / 16)
  (h_d : D = 33 / 16)
: 7 * x - 5 = 79 / 16 * (x + 4) + 33 / 16 * (x - 12) :=
by sorry

end find_C_D_l2024_202499


namespace sum_of_squares_s_comp_r_l2024_202436

def r (x : ℝ) : ℝ := x^2 - 4
def s (x : ℝ) : ℝ := -|x + 1|
def s_comp_r (x : ℝ) : ℝ := s (r x)

theorem sum_of_squares_s_comp_r :
  (s_comp_r (-4))^2 + (s_comp_r (-3))^2 + (s_comp_r (-2))^2 + (s_comp_r (-1))^2 +
  (s_comp_r 0)^2 + (s_comp_r 1)^2 + (s_comp_r 2)^2 + (s_comp_r 3)^2 + (s_comp_r 4)^2 = 429 :=
by
  sorry

end sum_of_squares_s_comp_r_l2024_202436


namespace cricket_player_average_l2024_202486

theorem cricket_player_average (A : ℝ) (h1 : 10 * A + 84 = 11 * (A + 4)) : A = 40 :=
by
  sorry

end cricket_player_average_l2024_202486


namespace calc_expr_l2024_202459

theorem calc_expr :
  (2 * Real.sqrt 2 - 1) ^ 2 + (1 + Real.sqrt 3) * (1 - Real.sqrt 3) = 7 - 4 * Real.sqrt 2 :=
by
  sorry

end calc_expr_l2024_202459


namespace workers_to_build_cars_l2024_202402

theorem workers_to_build_cars (W : ℕ) (hW : W > 0) : 
  (∃ D : ℝ, D = 63 / W) :=
by
  sorry

end workers_to_build_cars_l2024_202402
