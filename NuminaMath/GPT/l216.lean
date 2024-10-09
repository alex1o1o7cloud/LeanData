import Mathlib

namespace sequence_nth_term_l216_21690

theorem sequence_nth_term (a : ℕ → ℚ) (h : a 1 = 3 / 2 ∧ a 2 = 1 ∧ a 3 = 5 / 8 ∧ a 4 = 3 / 8) :
  ∀ n : ℕ, a n = (n^2 - 11*n + 34) / 16 := by
  sorry

end sequence_nth_term_l216_21690


namespace equal_roots_quadratic_l216_21651

theorem equal_roots_quadratic (k : ℝ) : (∃ (x : ℝ), x*(x + 2) + k = 0 ∧ ∀ y z, (y, z) = (x, x)) → k = 1 :=
sorry

end equal_roots_quadratic_l216_21651


namespace min_value_fraction_l216_21673

theorem min_value_fraction (a b : ℝ) (n : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_ab_sum : a + b = 2) : 
  (1 / (1 + a^n) + 1 / (1 + b^n)) = 1 :=
sorry

end min_value_fraction_l216_21673


namespace kelly_baking_powder_l216_21617

variable (current_supply : ℝ) (additional_supply : ℝ)

theorem kelly_baking_powder (h1 : current_supply = 0.3)
                            (h2 : additional_supply = 0.1) :
                            current_supply + additional_supply = 0.4 := 
by
  sorry

end kelly_baking_powder_l216_21617


namespace distance_A_to_B_is_64_yards_l216_21631

theorem distance_A_to_B_is_64_yards :
  let south1 := 50
  let west := 80
  let north := 20
  let east := 30
  let south2 := 10
  let net_south := south1 + south2 - north
  let net_west := west - east
  let distance := Real.sqrt ((net_south ^ 2) + (net_west ^ 2))
  distance = 64 :=
  by
  let south1 := 50
  let west := 80
  let north := 20
  let east := 30
  let south2 := 10
  let net_south := south1 + south2 - north
  let net_west := west - east
  let distance := Real.sqrt ((net_south ^ 2) + (net_west ^ 2))
  sorry

end distance_A_to_B_is_64_yards_l216_21631


namespace rolling_green_probability_l216_21623

/-- A cube with 5 green faces and 1 yellow face. -/
structure ColoredCube :=
  (green_faces : ℕ)
  (yellow_face : ℕ)
  (total_faces : ℕ)

def example_cube : ColoredCube :=
  { green_faces := 5, yellow_face := 1, total_faces := 6 }

/-- The probability of rolling a green face on a given cube. -/
def probability_of_rolling_green (cube : ColoredCube) : ℚ :=
  cube.green_faces / cube.total_faces

theorem rolling_green_probability :
  probability_of_rolling_green example_cube = 5 / 6 :=
by simp [probability_of_rolling_green, example_cube]

end rolling_green_probability_l216_21623


namespace find_a_l216_21674

theorem find_a (a b d : ℤ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 := by
  sorry

end find_a_l216_21674


namespace range_of_z_l216_21629

theorem range_of_z (a b : ℝ) (h1 : 2 < a) (h2 : a < 3) (h3 : -2 < b) (h4 : b < -1) :
  5 < 2 * a - b ∧ 2 * a - b < 8 :=
by
  sorry

end range_of_z_l216_21629


namespace friends_meet_probability_l216_21672

noncomputable def probability_of_meeting :=
  let duration_total := 60 -- Total duration from 14:00 to 15:00 in minutes
  let duration_meeting := 30 -- Duration they can meet from 14:00 to 14:30 in minutes
  duration_meeting / duration_total

theorem friends_meet_probability : probability_of_meeting = 1 / 2 := by
  sorry

end friends_meet_probability_l216_21672


namespace proof_p_and_q_true_l216_21634

open Real

def p : Prop := ∃ x : ℝ, x - 2 > log x
def q : Prop := ∀ x : ℝ, exp x > x

theorem proof_p_and_q_true : p ∧ q :=
by
  -- Assume you have already proven that p and q are true separately
  sorry

end proof_p_and_q_true_l216_21634


namespace red_other_side_probability_is_one_l216_21650

/-- Definitions from the problem conditions --/
def total_cards : ℕ := 10
def green_both_sides : ℕ := 5
def green_red_sides : ℕ := 2
def red_both_sides : ℕ := 3
def red_faces : ℕ := 6 -- 3 cards × 2 sides each

/-- The theorem proves the probability is 1 that the other side is red given that one side seen is red --/
theorem red_other_side_probability_is_one
  (h_total_cards : total_cards = 10)
  (h_green_both : green_both_sides = 5)
  (h_green_red : green_red_sides = 2)
  (h_red_both : red_both_sides = 3)
  (h_red_faces : red_faces = 6) :
  1 = (red_faces / red_faces) :=
by
  -- Write the proof steps here
  sorry

end red_other_side_probability_is_one_l216_21650


namespace radius_of_inscribed_circle_is_integer_l216_21657

theorem radius_of_inscribed_circle_is_integer 
  (a b c : ℤ) 
  (h_pythagorean : c^2 = a^2 + b^2) 
  : ∃ r : ℤ, r = (a + b - c) / 2 :=
by
  sorry

end radius_of_inscribed_circle_is_integer_l216_21657


namespace solve_expression_hundreds_digit_l216_21666

def factorial (n : ℕ) : ℕ :=
  Nat.factorial n

def div_mod (a b m : ℕ) : ℕ :=
  (a / b) % m

def hundreds_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

theorem solve_expression_hundreds_digit :
  hundreds_digit (div_mod (factorial 17) 5 1000 - div_mod (factorial 10) 2 1000) = 8 :=
by
  sorry

end solve_expression_hundreds_digit_l216_21666


namespace joan_apples_l216_21620

def initial_apples : ℕ := 43
def additional_apples : ℕ := 27
def total_apples (initial additional: ℕ) := initial + additional

theorem joan_apples : total_apples initial_apples additional_apples = 70 := by
  sorry

end joan_apples_l216_21620


namespace clock_rings_in_a_day_l216_21683

theorem clock_rings_in_a_day (intervals : ℕ) (hours_in_a_day : ℕ) (time_between_rings : ℕ) : 
  intervals = hours_in_a_day / time_between_rings + 1 → intervals = 7 :=
sorry

end clock_rings_in_a_day_l216_21683


namespace log_sum_identity_l216_21671

-- Prove that: lg 8 + 3 * lg 5 = 3

noncomputable def common_logarithm (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_sum_identity : 
    common_logarithm 8 + 3 * common_logarithm 5 = 3 := 
by
  sorry

end log_sum_identity_l216_21671


namespace optimal_play_probability_Reimu_l216_21609

noncomputable def probability_Reimu_wins : ℚ :=
  5 / 16

theorem optimal_play_probability_Reimu :
  probability_Reimu_wins = 5 / 16 := 
by
  sorry

end optimal_play_probability_Reimu_l216_21609


namespace two_digit_integers_remainder_3_count_l216_21693

theorem two_digit_integers_remainder_3_count :
  ∃ (n : ℕ) (a b : ℕ), a = 10 ∧ b = 99 ∧ (∀ k : ℕ, (a ≤ k ∧ k ≤ b) ↔ (∃ n : ℕ, k = 7 * n + 3 ∧ n ≥ 1 ∧ n ≤ 13)) :=
by
  sorry

end two_digit_integers_remainder_3_count_l216_21693


namespace sum_of_digits_of_fraction_repeating_decimal_l216_21679

theorem sum_of_digits_of_fraction_repeating_decimal :
  (exists (c d : ℕ), (4 / 13 : ℚ) = c * 0.1 + d * 0.01 ∧ (c + d) = 3) :=
sorry

end sum_of_digits_of_fraction_repeating_decimal_l216_21679


namespace smallest_b_to_the_a_l216_21608

theorem smallest_b_to_the_a (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = 2^2023) : b^a = 1 :=
by
  -- Proof steps go here
  sorry

end smallest_b_to_the_a_l216_21608


namespace simplify_expression_l216_21698

theorem simplify_expression (p : ℝ) : 
  (2 * (3 * p + 4) - 5 * p * 2)^2 + (6 - 2 / 2) * (9 * p - 12) = 16 * p^2 - 19 * p + 4 := 
by 
  sorry

end simplify_expression_l216_21698


namespace cos_A_minus_B_l216_21637

theorem cos_A_minus_B (A B : Real) 
  (h1 : Real.sin A + Real.sin B = -1) 
  (h2 : Real.cos A + Real.cos B = 1/2) :
  Real.cos (A - B) = -3/8 :=
by
  sorry

end cos_A_minus_B_l216_21637


namespace find_first_offset_l216_21652

theorem find_first_offset (x : ℝ) : 
  let area := 180
  let diagonal := 24
  let offset2 := 6
  (area = (diagonal * (x + offset2)) / 2) -> x = 9 :=
sorry

end find_first_offset_l216_21652


namespace max_ratio_of_two_digit_numbers_with_mean_55_l216_21643

theorem max_ratio_of_two_digit_numbers_with_mean_55 (x y : ℕ) (h1 : 10 ≤ x) (h2 : x ≤ 99) (h3 : 10 ≤ y) (h4 : y ≤ 99) (h5 : (x + y) / 2 = 55) : x / y ≤ 9 :=
sorry

end max_ratio_of_two_digit_numbers_with_mean_55_l216_21643


namespace correct_quadratic_graph_l216_21656

theorem correct_quadratic_graph (a b c : ℝ) (ha : a > 0) (hb : b < 0) (hc : c < 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (-b / (2 * a) > 0) ∧ (c < 0) :=
by
  sorry

end correct_quadratic_graph_l216_21656


namespace discounted_price_correct_l216_21605

noncomputable def discounted_price (original_price : ℝ) (discount : ℝ) : ℝ :=
  original_price - (discount / 100 * original_price)

theorem discounted_price_correct :
  discounted_price 800 30 = 560 :=
by
  -- Correctness of the discounted price calculation
  sorry

end discounted_price_correct_l216_21605


namespace Q_proper_subset_P_l216_21618

open Set

def P : Set ℝ := { x | x ≥ 1 }
def Q : Set ℝ := { 2, 3 }

theorem Q_proper_subset_P : Q ⊂ P :=
by
  sorry

end Q_proper_subset_P_l216_21618


namespace cylinder_height_l216_21684

theorem cylinder_height
  (V : ℝ → ℝ → ℝ) 
  (π : ℝ)
  (r h : ℝ)
  (vol_increase_height : ℝ)
  (vol_increase_radius : ℝ)
  (h_increase : ℝ)
  (r_increase : ℝ)
  (original_radius : ℝ) :
  V r h = π * r^2 * h → 
  vol_increase_height = π * r^2 * h_increase →
  vol_increase_radius = π * ((r + r_increase)^2 - r^2) * h →
  r = original_radius →
  vol_increase_height = 72 * π →
  vol_increase_radius = 72 * π →
  original_radius = 3 →
  r_increase = 2 →
  h_increase = 2 →
  h = 4.5 :=
by
  sorry

end cylinder_height_l216_21684


namespace jim_catches_up_to_cara_l216_21659

noncomputable def time_to_catch_up (jim_speed: ℝ) (cara_speed: ℝ) (initial_time: ℝ) (stretch_time: ℝ) : ℝ :=
  let initial_distance_jim := jim_speed * initial_time
  let initial_distance_cara := cara_speed * initial_time
  let added_distance_cara := cara_speed * stretch_time
  let distance_gap := added_distance_cara
  let relative_speed := jim_speed - cara_speed
  distance_gap / relative_speed

theorem jim_catches_up_to_cara :
  time_to_catch_up 6 5 (30/60) (18/60) * 60 = 90 :=
by
  sorry

end jim_catches_up_to_cara_l216_21659


namespace correct_calculation_l216_21602

theorem correct_calculation (a b : ℝ) : (-3 * a^3 * b)^2 = 9 * a^6 * b^2 :=
by sorry

end correct_calculation_l216_21602


namespace harry_total_payment_in_silvers_l216_21668

-- Definitions for the conditions
def spellbook_gold_cost : ℕ := 5
def spellbook_count : ℕ := 5
def potion_kit_silver_cost : ℕ := 20
def potion_kit_count : ℕ := 3
def owl_gold_cost : ℕ := 28
def silver_per_gold : ℕ := 9

-- Translate the total cost to silver
noncomputable def total_cost_in_silvers : ℕ :=
  spellbook_count * spellbook_gold_cost * silver_per_gold + 
  potion_kit_count * potion_kit_silver_cost + 
  owl_gold_cost * silver_per_gold

-- State the theorem
theorem harry_total_payment_in_silvers : total_cost_in_silvers = 537 :=
by
  unfold total_cost_in_silvers
  sorry

end harry_total_payment_in_silvers_l216_21668


namespace final_result_l216_21626

def a : ℕ := 2548
def b : ℕ := 364
def hcd := Nat.gcd a b
def result := hcd + 8 - 12

theorem final_result : result = 360 := by
  sorry

end final_result_l216_21626


namespace num_ints_between_sqrt2_and_sqrt32_l216_21664

theorem num_ints_between_sqrt2_and_sqrt32 : 
  ∃ n : ℕ, n = 4 ∧ 
  (∀ k : ℤ, (2 ≤ k) ∧ (k ≤ 5)) :=
by
  sorry

end num_ints_between_sqrt2_and_sqrt32_l216_21664


namespace odd_periodic_function_value_l216_21692

theorem odd_periodic_function_value
  (f : ℝ → ℝ)
  (odd_f : ∀ x, f (-x) = - f x)
  (periodic_f : ∀ x, f (x + 3) = f x)
  (bounded_f : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x) :
  f 8.5 = -1 :=
sorry

end odd_periodic_function_value_l216_21692


namespace ewan_sequence_has_113_l216_21606

def sequence_term (n : ℕ) : ℤ := 11 * n - 8

theorem ewan_sequence_has_113 : ∃ n : ℕ, sequence_term n = 113 := by
  sorry

end ewan_sequence_has_113_l216_21606


namespace length_of_opposite_leg_l216_21653

noncomputable def hypotenuse_length : Real := 18

noncomputable def angle_deg : Real := 30

theorem length_of_opposite_leg (h : Real) (angle : Real) (condition1 : h = hypotenuse_length) (condition2 : angle = angle_deg) : 
 ∃ x : Real, 2 * x = h ∧ angle = 30 → x = 9 := 
by
  sorry

end length_of_opposite_leg_l216_21653


namespace find_second_dimension_l216_21682

variable (l h w : ℕ)
variable (cost_per_sqft total_cost : ℕ)
variable (surface_area : ℕ)

def insulation_problem_conditions (l : ℕ) (h : ℕ) (cost_per_sqft : ℕ) (total_cost : ℕ) (w : ℕ) (surface_area : ℕ) : Prop :=
  l = 4 ∧ h = 3 ∧ cost_per_sqft = 20 ∧ total_cost = 1880 ∧ surface_area = (2 * l * w + 2 * l * h + 2 * w * h)

theorem find_second_dimension (l h w : ℕ) (cost_per_sqft total_cost surface_area : ℕ) :
  insulation_problem_conditions l h cost_per_sqft total_cost w surface_area →
  surface_area = 94 →
  w = 5 :=
by
  intros
  simp [insulation_problem_conditions] at *
  sorry

end find_second_dimension_l216_21682


namespace mark_age_l216_21628

-- Definitions based on the conditions in the problem
variables (M J P : ℕ)  -- Current ages of Mark, John, and their parents respectively

-- Condition definitions
def condition1 : Prop := J = M - 10
def condition2 : Prop := P = 5 * J
def condition3 : Prop := P - 22 = M

-- The theorem to prove the correct answer
theorem mark_age : condition1 M J ∧ condition2 J P ∧ condition3 P M → M = 18 := by
  sorry

end mark_age_l216_21628


namespace circle_passes_through_fixed_point_l216_21647

theorem circle_passes_through_fixed_point (a : ℝ) (ha : a ≠ 1) : 
  ∃ P : ℝ × ℝ, P = (1, 1) ∧ ∀ (x y : ℝ), (x^2 + y^2 - 2*a*x + 2*(a-2)*y + 2 = 0) → (x, y) = P :=
sorry

end circle_passes_through_fixed_point_l216_21647


namespace no_such_quadratics_l216_21686

theorem no_such_quadratics :
  ¬ ∃ (a b c : ℤ), ∃ (x1 x2 x3 x4 : ℤ),
    (a * x1 * x2 = c ∧ a * (x1 + x2) = -b) ∧
    ((a + 1) * x3 * x4 = c + 1 ∧ (a + 1) * (x3 + x4) = -(b + 1)) :=
sorry

end no_such_quadratics_l216_21686


namespace weight_of_one_fan_l216_21676

theorem weight_of_one_fan
  (total_weight_with_fans : ℝ)
  (num_fans : ℕ)
  (empty_box_weight : ℝ)
  (h1 : total_weight_with_fans = 11.14)
  (h2 : num_fans = 14)
  (h3 : empty_box_weight = 0.5) :
  (total_weight_with_fans - empty_box_weight) / num_fans = 0.76 :=
by
  simp [h1, h2, h3]
  sorry

end weight_of_one_fan_l216_21676


namespace find_number_l216_21611

theorem find_number (x : ℤ) :
  45 - (x - (37 - (15 - 18))) = 57 → x = 28 :=
by
  sorry

end find_number_l216_21611


namespace min_forget_all_three_l216_21695

theorem min_forget_all_three (total_students students_forgot_gloves students_forgot_scarves students_forgot_hats : ℕ) (h_total : total_students = 60) (h_gloves : students_forgot_gloves = 55) (h_scarves : students_forgot_scarves = 52) (h_hats : students_forgot_hats = 50) :
  ∃ min_students_forget_three, min_students_forget_three = total_students - (total_students - students_forgot_gloves + total_students - students_forgot_scarves + total_students - students_forgot_hats) :=
by
  use 37
  sorry

end min_forget_all_three_l216_21695


namespace ducks_drinking_l216_21619

theorem ducks_drinking (total_d : ℕ) (drank_before : ℕ) (drank_after : ℕ) :
  total_d = 20 → drank_before = 11 → drank_after = total_d - (drank_before + 1) → drank_after = 8 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end ducks_drinking_l216_21619


namespace B_pow_97_l216_21612

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 1, 0],
  ![0, 0, 1],
  ![1, 0, 0]
]

theorem B_pow_97 : B ^ 97 = B := by
  sorry

end B_pow_97_l216_21612


namespace sum_of_coordinates_is_17_over_3_l216_21636

theorem sum_of_coordinates_is_17_over_3
  (f : ℝ → ℝ)
  (h1 : 5 = 3 * f 2) :
  (5 / 3 + 4) = 17 / 3 :=
by
  have h2 : f 2 = 5 / 3 := by
    linarith
  have h3 : f⁻¹ (5 / 3) = 2 := by
    sorry -- we do not know more properties of f to conclude this proof step
  have h4 : 2 * f⁻¹ (5 / 3) = 4 := by
    sorry -- similarly, assume for now the desired property
  exact sorry -- finally putting everything together

end sum_of_coordinates_is_17_over_3_l216_21636


namespace arithmetic_sequence_term_difference_l216_21661

theorem arithmetic_sequence_term_difference :
  let a : ℕ := 3
  let d : ℕ := 6
  let t1 := a + 1499 * d
  let t2 := a + 1503 * d
  t2 - t1 = 24 :=
    by
    sorry

end arithmetic_sequence_term_difference_l216_21661


namespace combined_selling_price_l216_21646

theorem combined_selling_price :
  let cost_price_A := 180
  let profit_percent_A := 0.15
  let cost_price_B := 220
  let profit_percent_B := 0.20
  let cost_price_C := 130
  let profit_percent_C := 0.25
  let selling_price_A := cost_price_A * (1 + profit_percent_A)
  let selling_price_B := cost_price_B * (1 + profit_percent_B)
  let selling_price_C := cost_price_C * (1 + profit_percent_C)
  selling_price_A + selling_price_B + selling_price_C = 633.50 := by
  sorry

end combined_selling_price_l216_21646


namespace portrait_in_silver_box_l216_21645

theorem portrait_in_silver_box
  (gold_box : Prop)
  (silver_box : Prop)
  (lead_box : Prop)
  (p : Prop) (q : Prop) (r : Prop)
  (h1 : p ↔ gold_box)
  (h2 : q ↔ ¬silver_box)
  (h3 : r ↔ ¬gold_box)
  (h4 : (p ∨ q ∨ r) ∧ ¬(p ∧ q) ∧ ¬(q ∧ r) ∧ ¬(r ∧ p)) :
  silver_box :=
sorry

end portrait_in_silver_box_l216_21645


namespace evaluate_expression_l216_21649

theorem evaluate_expression : (1 / (1 - 1 / (3 + 1 / 4))) = (13 / 9) :=
by
  sorry

end evaluate_expression_l216_21649


namespace adding_sugar_increases_sweetness_l216_21697

theorem adding_sugar_increases_sweetness 
  (a b m : ℝ) (hb : b > a) (ha : a > 0) (hm : m > 0) : 
  (a / b) < (a + m) / (b + m) := 
by
  sorry

end adding_sugar_increases_sweetness_l216_21697


namespace coat_price_reduction_l216_21641

theorem coat_price_reduction (original_price reduction_amount : ℝ) 
  (h1 : original_price = 500) (h2 : reduction_amount = 400) :
  (reduction_amount / original_price) * 100 = 80 :=
by {
  sorry -- This is where the proof would go
}

end coat_price_reduction_l216_21641


namespace triangular_faces_area_of_pyramid_l216_21639

noncomputable def total_area_of_triangular_faces (base : ℝ) (lateral : ℝ) : ℝ :=
  let h := Real.sqrt (lateral ^ 2 - (base / 2) ^ 2)
  let area_one_triangle := (1 / 2) * base * h
  4 * area_one_triangle

theorem triangular_faces_area_of_pyramid :
  total_area_of_triangular_faces 8 10 = 32 * Real.sqrt 21 := by
  sorry

end triangular_faces_area_of_pyramid_l216_21639


namespace negation_of_P_l216_21688

-- Define the proposition P
def P (x : ℝ) : Prop := x^2 = 1 → x = 1

-- Define the negation of the proposition P
def neg_P (x : ℝ) : Prop := x^2 ≠ 1 → x ≠ 1

theorem negation_of_P (x : ℝ) : ¬P x ↔ neg_P x := by
  sorry

end negation_of_P_l216_21688


namespace find_distance_l216_21648

-- Definitions based on given conditions
def speed : ℝ := 40 -- in km/hr
def time : ℝ := 6 -- in hours

-- Theorem statement
theorem find_distance (speed : ℝ) (time : ℝ) : speed = 40 → time = 6 → speed * time = 240 :=
by
  intros h1 h2
  rw [h1, h2]
  -- skipping the proof with sorry
  sorry

end find_distance_l216_21648


namespace shortest_distance_between_circles_l216_21635

-- Conditions
def first_circle (x y : ℝ) : Prop := x^2 - 10 * x + y^2 - 4 * y - 7 = 0
def second_circle (x y : ℝ) : Prop := x^2 + 14 * x + y^2 + 6 * y + 49 = 0

-- Goal: Prove the shortest distance between the two circles is 4
theorem shortest_distance_between_circles : 
  -- Given conditions about the equations of the circles
  (∀ x y : ℝ, first_circle x y ↔ (x - 5)^2 + (y - 2)^2 = 36) ∧ 
  (∀ x y : ℝ, second_circle x y ↔ (x + 7)^2 + (y + 3)^2 = 9) →
  -- Assert the shortest distance between the two circles is 4
  13 - (6 + 3) = 4 :=
by
  sorry

end shortest_distance_between_circles_l216_21635


namespace chuck_total_play_area_l216_21670

noncomputable def chuck_play_area (leash_radius : ℝ) : ℝ :=
  let middle_arc_area := (1 / 2) * Real.pi * leash_radius^2
  let corner_arc_area := 2 * (1 / 4) * Real.pi * leash_radius^2
  middle_arc_area + corner_arc_area

theorem chuck_total_play_area (leash_radius : ℝ) (shed_width shed_length : ℝ) 
  (h_radius : leash_radius = 4) (h_width : shed_width = 4) (h_length : shed_length = 6) :
  chuck_play_area leash_radius = 16 * Real.pi :=
by
  sorry

end chuck_total_play_area_l216_21670


namespace order_of_a_b_c_l216_21681

noncomputable def ln : ℝ → ℝ := Real.log
noncomputable def a : ℝ := ln 3 / 3
noncomputable def b : ℝ := ln 5 / 5
noncomputable def c : ℝ := ln 6 / 6

theorem order_of_a_b_c : a > b ∧ b > c := by
  sorry

end order_of_a_b_c_l216_21681


namespace find_t_and_m_l216_21616

theorem find_t_and_m 
  (t m : ℝ) 
  (ineq : ∀ x : ℝ, x^2 - 3 * x + t < 0 ↔ 1 < x ∧ x < m) : 
  t = 2 ∧ m = 2 :=
sorry

end find_t_and_m_l216_21616


namespace task_candy_distribution_l216_21680

noncomputable def candy_distribution_eq_eventually (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∃ k : ℕ, ∀ m : ℕ, ∀ j : ℕ, m ≥ k → a (j + m * n) = a (0 + m * n)

theorem task_candy_distribution :
  ∀ n : ℕ, n > 0 →
  ∀ a : ℕ → ℕ,
  (∀ i : ℕ, a i = if a i % 2 = 1 then (a i) + 1 else a i) →
  (∀ i : ℕ, a (i + 1) = a i / 2 + a (i - 1) / 2) →
  candy_distribution_eq_eventually n a :=
by
  intros n n_positive a h_even h_transfer
  sorry

end task_candy_distribution_l216_21680


namespace scientific_notation_of_3100000_l216_21642

theorem scientific_notation_of_3100000 :
  ∃ (a : ℝ) (n : ℤ), 3100000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.1 ∧ n = 6 :=
  sorry

end scientific_notation_of_3100000_l216_21642


namespace fish_count_seventh_day_l216_21667

-- Define the initial state and transformations
def fish_count (n: ℕ) :=
  if n = 0 then 6
  else
    if n = 3 then fish_count (n-1) / 3 * 2 * 2 * 2 - fish_count (n-1) / 3
    else if n = 5 then (fish_count (n-1) * 2) / 4 * 3
    else if n = 6 then fish_count (n-1) * 2 + 15
    else fish_count (n-1) * 2

theorem fish_count_seventh_day : fish_count 7 = 207 :=
by
  sorry

end fish_count_seventh_day_l216_21667


namespace domain_of_sqrt_sum_l216_21610

theorem domain_of_sqrt_sum (x : ℝ) : (1 ≤ x ∧ x ≤ 3) ↔ (x - 1 ≥ 0 ∧ 3 - x ≥ 0) := by
  sorry

end domain_of_sqrt_sum_l216_21610


namespace paco_initial_sweet_cookies_l216_21603

theorem paco_initial_sweet_cookies (S : ℕ) (h1 : S - 15 = 7) : S = 22 :=
by
  sorry

end paco_initial_sweet_cookies_l216_21603


namespace ratio_of_vanilla_chips_l216_21638

-- Definitions from the conditions
variable (V_c S_c V_v S_v : ℕ)
variable (H1 : V_c = S_c + 5)
variable (H2 : S_c = 25)
variable (H3 : V_v = 20)
variable (H4 : V_c + S_c + V_v + S_v = 90)

-- The statement we want to prove
theorem ratio_of_vanilla_chips : S_v / V_v = 3 / 4 := by
  sorry

end ratio_of_vanilla_chips_l216_21638


namespace total_customers_in_line_l216_21694

-- Define the number of people behind the first person
def people_behind := 11

-- Define the total number of people in line
def people_in_line : Nat := people_behind + 1

-- Prove the total number of people in line is 12
theorem total_customers_in_line : people_in_line = 12 :=
by
  sorry

end total_customers_in_line_l216_21694


namespace power_function_below_identity_l216_21627

theorem power_function_below_identity {α : ℝ} :
  (∀ x : ℝ, 1 < x → x^α < x) → α < 1 :=
by
  intro h
  sorry

end power_function_below_identity_l216_21627


namespace no_real_x_solution_l216_21614

open Real

-- Define the conditions.
def log_defined (x : ℝ) : Prop :=
  0 < x + 5 ∧ 0 < x - 3 ∧ 0 < x^2 - 7*x - 18

-- Define the equation to prove.
def log_eqn (x : ℝ) : Prop :=
  log (x + 5) + log (x - 3) = log (x^2 - 7*x - 18)

-- The mathematicall equivalent proof problem.
theorem no_real_x_solution : ¬∃ x : ℝ, log_defined x ∧ log_eqn x :=
by
  sorry

end no_real_x_solution_l216_21614


namespace inequality_solution_l216_21613

theorem inequality_solution (x : ℝ) : 
  (x - 1) / (2 * x + 1) ≤ 0 ↔ -1 / 2 < x ∧ x ≤ 1 :=
sorry

end inequality_solution_l216_21613


namespace MrsYoung_puzzle_complete_l216_21654

theorem MrsYoung_puzzle_complete :
  let total_pieces := 500
  let children := 4
  let pieces_per_child := total_pieces / children
  let minutes := 120
  let pieces_Reyn := (25 * (minutes / 30))
  let pieces_Rhys := 2 * pieces_Reyn
  let pieces_Rory := 3 * pieces_Reyn
  let pieces_Rina := 4 * pieces_Reyn
  let total_pieces_placed := pieces_Reyn + pieces_Rhys + pieces_Rory + pieces_Rina
  total_pieces_placed >= total_pieces :=
by
  sorry

end MrsYoung_puzzle_complete_l216_21654


namespace lcm_of_two_numbers_l216_21669

theorem lcm_of_two_numbers (a b : ℕ) (h_hcf : Nat.gcd a b = 6) (h_product : a * b = 432) :
  Nat.lcm a b = 72 :=
by 
  sorry

end lcm_of_two_numbers_l216_21669


namespace decrement_value_is_15_l216_21640

noncomputable def decrement_value (n : ℕ) (original_mean updated_mean : ℕ) : ℕ :=
  (n * original_mean - n * updated_mean) / n

theorem decrement_value_is_15 : decrement_value 50 200 185 = 15 :=
by
  sorry

end decrement_value_is_15_l216_21640


namespace max_value_func1_l216_21615

theorem max_value_func1 (x : ℝ) (h : 0 < x ∧ x < 2) : 
  ∃ y, y = x * (4 - 2 * x) ∧ (∀ z, z = x * (4 - 2 * x) → z ≤ 2) :=
sorry

end max_value_func1_l216_21615


namespace find_x_l216_21644

theorem find_x (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 14) : x = 12 := by
  sorry

end find_x_l216_21644


namespace evaluate_expression_l216_21622

theorem evaluate_expression : 8 - 5 * (9 - (4 - 2)^2) * 2 = -42 := by
  sorry

end evaluate_expression_l216_21622


namespace smallest_integer_value_l216_21665

theorem smallest_integer_value (n : ℤ) : ∃ (n : ℤ), n = 5 ∧ n^2 - 11*n + 28 < 0 :=
by
  use 5
  sorry

end smallest_integer_value_l216_21665


namespace intersection_M_N_l216_21601

noncomputable def M : Set ℝ := {-1, 0, 1}
noncomputable def N : Set ℝ := {x | x * (x - 2) ≤ 0}

theorem intersection_M_N : M ∩ N = {0, 1} :=
by {
  sorry
}

end intersection_M_N_l216_21601


namespace total_time_to_climb_seven_flights_l216_21658

-- Define the conditions
def first_flight_time : ℕ := 15
def difference_between_flights : ℕ := 10
def num_of_flights : ℕ := 7

-- Define the sum of an arithmetic series function
def arithmetic_series_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Define the theorem
theorem total_time_to_climb_seven_flights :
  arithmetic_series_sum first_flight_time difference_between_flights num_of_flights = 315 :=
by
  sorry

end total_time_to_climb_seven_flights_l216_21658


namespace speed_of_stream_l216_21600

theorem speed_of_stream (b s : ℕ) 
  (h1 : b + s = 42) 
  (h2 : b - s = 24) :
  s = 9 := by sorry

end speed_of_stream_l216_21600


namespace numBaskets_l216_21691

noncomputable def numFlowersInitial : ℕ := 5 + 5
noncomputable def numFlowersAfterGrowth : ℕ := numFlowersInitial + 20
noncomputable def numFlowersFinal : ℕ := numFlowersAfterGrowth - 10
noncomputable def flowersPerBasket : ℕ := 4

theorem numBaskets : numFlowersFinal / flowersPerBasket = 5 := 
by
  sorry

end numBaskets_l216_21691


namespace salary_problem_l216_21678

theorem salary_problem
  (A B : ℝ)
  (h1 : A + B = 3000)
  (h2 : 0.05 * A = 0.15 * B) :
  A = 2250 :=
sorry

end salary_problem_l216_21678


namespace sean_total_spending_l216_21655

noncomputable def cost_first_bakery_euros : ℝ :=
  let almond_croissants := 2 * 4.00
  let salami_cheese_croissants := 3 * 5.00
  let total_before_discount := almond_croissants + salami_cheese_croissants
  total_before_discount * 0.90 -- 10% discount

noncomputable def cost_second_bakery_pounds : ℝ :=
  let plain_croissants := 3 * 3.50 -- buy-3-get-1-free
  let focaccia := 5.00
  let total_before_tax := plain_croissants + focaccia
  total_before_tax * 1.05 -- 5% tax

noncomputable def cost_cafe_dollars : ℝ :=
  let lattes := 3 * 3.00
  lattes * 0.85 -- 15% student discount

noncomputable def first_bakery_usd : ℝ :=
  cost_first_bakery_euros * 1.15 -- converting euros to dollars

noncomputable def second_bakery_usd : ℝ :=
  cost_second_bakery_pounds * 1.35 -- converting pounds to dollars

noncomputable def total_cost_sean_spends : ℝ :=
  first_bakery_usd + second_bakery_usd + cost_cafe_dollars

theorem sean_total_spending : total_cost_sean_spends = 53.44 :=
  by
  -- The proof can be handled here
  sorry

end sean_total_spending_l216_21655


namespace distance_from_A_to_origin_l216_21663

open Real

theorem distance_from_A_to_origin 
  (x1 y1 : ℝ)
  (hx1 : y1^2 = 4 * x1)
  (hratio : (x1 + 1) / abs y1 = 5 / 4)
  (hAF_gt_2 : dist (x1, y1) (1, 0) > 2) : 
  dist (x1, y1) (0, 0) = 4 * sqrt 2 :=
sorry

end distance_from_A_to_origin_l216_21663


namespace toucan_count_l216_21604

theorem toucan_count :
  (2 + 1 = 3) :=
by simp [add_comm]

end toucan_count_l216_21604


namespace janet_pairs_of_2_l216_21675

def total_pairs (x y z : ℕ) : Prop := x + y + z = 18

def total_cost (x y z : ℕ) : Prop := 2 * x + 5 * y + 7 * z = 60

theorem janet_pairs_of_2 (x y z : ℕ) (h1 : total_pairs x y z) (h2 : total_cost x y z) (hz : z = 3) : x = 12 :=
by
  -- Proof is currently skipped
  sorry

end janet_pairs_of_2_l216_21675


namespace boys_neither_happy_nor_sad_l216_21625

theorem boys_neither_happy_nor_sad : 
  (∀ children total happy sad neither boys girls happy_boys sad_girls : ℕ,
    total = 60 →
    happy = 30 →
    sad = 10 →
    neither = 20 →
    boys = 19 →
    girls = 41 →
    happy_boys = 6 →
    sad_girls = 4 →
    (boys - (happy_boys + (sad - sad_girls))) = 7) :=
by
  intros children total happy sad neither boys girls happy_boys sad_girls
  sorry

end boys_neither_happy_nor_sad_l216_21625


namespace percent_palindromes_containing_7_l216_21677

theorem percent_palindromes_containing_7 : 
  let num_palindromes := 90
  let num_palindrome_with_7 := 19
  (num_palindrome_with_7 / num_palindromes * 100) = 21.11 := 
by
  sorry

end percent_palindromes_containing_7_l216_21677


namespace minimum_discount_correct_l216_21630

noncomputable def minimum_discount (total_weight: ℝ) (cost_price: ℝ) (sell_price: ℝ) 
                                   (profit_required: ℝ) : ℝ :=
  let first_half_profit := (total_weight / 2) * (sell_price - cost_price)
  let second_half_profit_with_discount (x: ℝ) := (total_weight / 2) * (sell_price * x - cost_price)
  let required_profit_condition (x: ℝ) := first_half_profit + second_half_profit_with_discount x ≥ profit_required
  (1 - (7 / 11))

theorem minimum_discount_correct : minimum_discount 1000 7 10 2000 = 4 / 11 := 
by {
  -- We need to solve the inequality step by step to reach the final answer
  sorry
}

end minimum_discount_correct_l216_21630


namespace emily_cards_l216_21632

theorem emily_cards (initial_cards : ℕ) (total_cards : ℕ) (given_cards : ℕ) 
  (h1 : initial_cards = 63) (h2 : total_cards = 70) 
  (h3 : total_cards = initial_cards + given_cards) : 
  given_cards = 7 := 
by 
  sorry

end emily_cards_l216_21632


namespace jenna_water_cups_l216_21699

theorem jenna_water_cups (O S W : ℕ) (h1 : S = 3 * O) (h2 : W = 3 * S) (h3 : O = 4) : W = 36 :=
by
  sorry

end jenna_water_cups_l216_21699


namespace parallelepiped_inequality_l216_21689

theorem parallelepiped_inequality (a b c d : ℝ) (h : d^2 = a^2 + b^2 + c^2 + 2 * (a * b + a * c + b * c)) :
  a^2 + b^2 + c^2 ≥ (1 / 3) * d^2 :=
by
  sorry

end parallelepiped_inequality_l216_21689


namespace joan_original_seashells_l216_21660

theorem joan_original_seashells (a b total: ℕ) (h1 : a = 63) (h2 : b = 16) (h3: total = a + b) : total = 79 :=
by
  rw [h1, h2] at h3
  exact h3

end joan_original_seashells_l216_21660


namespace neg_parallelogram_is_rhombus_l216_21696

def parallelogram_is_rhombus := true

theorem neg_parallelogram_is_rhombus : ¬ parallelogram_is_rhombus := by
  sorry

end neg_parallelogram_is_rhombus_l216_21696


namespace rectangle_perimeter_eq_l216_21624

noncomputable def rectangle_perimeter (z w : ℕ) : ℕ :=
  let longer_side := w
  let shorter_side := (z - w) / 2
  2 * longer_side + 2 * shorter_side

theorem rectangle_perimeter_eq (z w : ℕ) : rectangle_perimeter z w = w + z := by
  sorry

end rectangle_perimeter_eq_l216_21624


namespace rabbit_population_2002_l216_21685

theorem rabbit_population_2002 :
  ∃ (x : ℕ) (k : ℝ), 
    (180 - 50 = k * x) ∧ 
    (255 - 75 = k * 180) ∧ 
    x = 130 :=
by
  sorry

end rabbit_population_2002_l216_21685


namespace largest_of_four_numbers_l216_21607

variables {x y z w : ℕ}

theorem largest_of_four_numbers
  (h1 : x + y + z = 180)
  (h2 : x + y + w = 197)
  (h3 : x + z + w = 208)
  (h4 : y + z + w = 222) :
  max x (max y (max z w)) = 89 :=
sorry

end largest_of_four_numbers_l216_21607


namespace z_is_imaginary_z_is_purely_imaginary_iff_z_on_angle_bisector_iff_l216_21633

open Complex

-- Problem definitions
def z (m : ℝ) : ℂ := (2 + I) * m^2 - 2 * (1 - I)

-- Prove that for all m in ℝ, z is imaginary
theorem z_is_imaginary (m : ℝ) : ∃ a : ℝ, z m = a * I :=
  sorry

-- Prove that z is purely imaginary iff m = ±1
theorem z_is_purely_imaginary_iff (m : ℝ) : (∃ b : ℝ, z m = b * I ∧ b ≠ 0) ↔ (m = 1 ∨ m = -1) :=
  sorry

-- Prove that z is on the angle bisector iff m = 0
theorem z_on_angle_bisector_iff (m : ℝ) : (z m).re = -((z m).im) ↔ (m = 0) :=
  sorry

end z_is_imaginary_z_is_purely_imaginary_iff_z_on_angle_bisector_iff_l216_21633


namespace weight_loss_in_april_l216_21662

-- Definitions based on given conditions
def total_weight_to_lose : ℕ := 10
def march_weight_loss : ℕ := 3
def may_weight_loss : ℕ := 3

-- Theorem statement
theorem weight_loss_in_april :
  total_weight_to_lose = march_weight_loss + 4 + may_weight_loss := 
sorry

end weight_loss_in_april_l216_21662


namespace Sydney_initial_rocks_l216_21687

variable (S₀ : ℕ)

def Conner_initial : ℕ := 723
def Sydney_collects_day1 : ℕ := 4
def Conner_collects_day1 : ℕ := 8 * Sydney_collects_day1
def Sydney_collects_day2 : ℕ := 0
def Conner_collects_day2 : ℕ := 123
def Sydney_collects_day3 : ℕ := 2 * Conner_collects_day1
def Conner_collects_day3 : ℕ := 27

def Total_Sydney_collects : ℕ := Sydney_collects_day1 + Sydney_collects_day2 + Sydney_collects_day3
def Total_Conner_collects : ℕ := Conner_collects_day1 + Conner_collects_day2 + Conner_collects_day3

def Total_Sydney_rocks : ℕ := S₀ + Total_Sydney_collects
def Total_Conner_rocks : ℕ := Conner_initial + Total_Conner_collects

theorem Sydney_initial_rocks :
  Total_Conner_rocks = Total_Sydney_rocks → S₀ = 837 :=
by
  sorry

end Sydney_initial_rocks_l216_21687


namespace tangent_line_through_point_l216_21621

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x

theorem tangent_line_through_point (x y : ℝ) (h₁ : y = 2 * Real.log x - x) (h₂ : (1 : ℝ)  ≠ 0) 
  (h₃ : (-1 : ℝ) ≠ 0):
  (x - y - 2 = 0) :=
sorry

end tangent_line_through_point_l216_21621
