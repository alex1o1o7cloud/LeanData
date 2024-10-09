import Mathlib

namespace bus_driver_total_hours_l904_90465

theorem bus_driver_total_hours
  (reg_rate : ℝ := 16)
  (ot_rate : ℝ := 28)
  (total_hours : ℝ)
  (total_compensation : ℝ := 920)
  (h : total_compensation = reg_rate * 40 + ot_rate * (total_hours - 40)) :
  total_hours = 50 := 
by 
  sorry

end bus_driver_total_hours_l904_90465


namespace mode_of_list_is_five_l904_90477

def list := [3, 4, 5, 5, 5, 5, 7, 11, 21]

def occurrence_count (l : List ℕ) (x : ℕ) : ℕ :=
  l.count x

def is_mode (l : List ℕ) (x : ℕ) : Prop :=
  ∀ y : ℕ, occurrence_count l x ≥ occurrence_count l y

theorem mode_of_list_is_five : is_mode list 5 := by
  sorry

end mode_of_list_is_five_l904_90477


namespace prime_integer_roots_l904_90476

theorem prime_integer_roots (p : ℕ) (hp : Prime p) 
  (hroots : ∀ (x1 x2 : ℤ), x1 * x2 = -512 * p ∧ x1 + x2 = -p) : p = 2 :=
by
  -- Proof omitted
  sorry

end prime_integer_roots_l904_90476


namespace solution_set_of_inequality_l904_90439

variable (f : ℝ → ℝ)
variable (h_inc : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y)

theorem solution_set_of_inequality :
  {x | 0 < x ∧ f x > f (2 * x - 4)} = {x | 2 < x ∧ x < 4} :=
by
  sorry

end solution_set_of_inequality_l904_90439


namespace subset_property_l904_90406

theorem subset_property : {2} ⊆ {x | x ≤ 10} := 
by 
  sorry

end subset_property_l904_90406


namespace albert_needs_more_money_l904_90456

def cost_paintbrush : Real := 1.50
def cost_paints : Real := 4.35
def cost_easel : Real := 12.65
def cost_canvas : Real := 7.95
def cost_palette : Real := 3.75
def money_albert_has : Real := 10.60
def total_cost : Real := cost_paintbrush + cost_paints + cost_easel + cost_canvas + cost_palette
def money_needed : Real := total_cost - money_albert_has

theorem albert_needs_more_money : money_needed = 19.60 := by
  sorry

end albert_needs_more_money_l904_90456


namespace typing_time_together_l904_90437

def meso_typing_rate : ℕ := 3 -- pages per minute
def tyler_typing_rate : ℕ := 5 -- pages per minute
def pages_to_type : ℕ := 40 -- pages

theorem typing_time_together :
  (meso_typing_rate + tyler_typing_rate) * 5 = pages_to_type :=
by
  sorry

end typing_time_together_l904_90437


namespace pears_sold_in_afternoon_l904_90470

theorem pears_sold_in_afternoon (m a total : ℕ) (h1 : a = 2 * m) (h2 : m = 120) (h3 : m + a = total) (h4 : total = 360) :
  a = 240 :=
by
  sorry

end pears_sold_in_afternoon_l904_90470


namespace goose_eggs_calculation_l904_90487

theorem goose_eggs_calculation (E : ℝ) (hatch_fraction : ℝ) (survived_first_month_fraction : ℝ) 
(survived_first_year_fraction : ℝ) (survived_first_year : ℝ) (no_more_than_one_per_egg : Prop) 
(h_hatch : hatch_fraction = 1/3) 
(h_month_survival : survived_first_month_fraction = 3/4)
(h_year_survival : survived_first_year_fraction = 2/5)
(h_survived120 : survived_first_year = 120)
(h_no_more_than_one : no_more_than_one_per_egg) :
  E = 1200 :=
by
  -- Convert the information from conditions to formulate the equation
  sorry


end goose_eggs_calculation_l904_90487


namespace total_volume_of_four_boxes_l904_90403

theorem total_volume_of_four_boxes :
  (∃ (V : ℕ), (∀ (edge_length : ℕ) (num_boxes : ℕ), edge_length = 6 → num_boxes = 4 → V = (edge_length ^ 3) * num_boxes)) :=
by
  let edge_length := 6
  let num_boxes := 4
  let volume := (edge_length ^ 3) * num_boxes
  use volume
  sorry

end total_volume_of_four_boxes_l904_90403


namespace basketball_surface_area_l904_90425

theorem basketball_surface_area (C : ℝ) (r : ℝ) (A : ℝ) (π : ℝ) 
  (h1 : C = 30) 
  (h2 : C = 2 * π * r) 
  (h3 : A = 4 * π * r^2) 
  : A = 900 / π := by
  sorry

end basketball_surface_area_l904_90425


namespace Wayne_blocks_count_l904_90415

-- Statement of the proof problem
theorem Wayne_blocks_count (initial_blocks additional_blocks total_blocks : ℕ) 
  (h1 : initial_blocks = 9) 
  (h2 : additional_blocks = 6) 
  (h3 : total_blocks = initial_blocks + additional_blocks) : 
  total_blocks = 15 := 
by 
  -- proof would go here, but we will use sorry for now
  sorry

end Wayne_blocks_count_l904_90415


namespace ratio_of_x_to_y_l904_90404

theorem ratio_of_x_to_y (x y : ℝ) (h : (8 * x - 5 * y) / (11 * x - 3 * y) = 4 / 7) : x / y = 23 / 12 := 
by
  sorry

end ratio_of_x_to_y_l904_90404


namespace gcd_gx_x_l904_90485

noncomputable def g (x : ℤ) : ℤ :=
  (3 * x + 5) * (9 * x + 4) * (11 * x + 8) * (x + 11)

theorem gcd_gx_x (x : ℤ) (h : 34914 ∣ x) : Int.gcd (g x) x = 1760 :=
by
  sorry

end gcd_gx_x_l904_90485


namespace cos_cofunction_identity_l904_90451

theorem cos_cofunction_identity (α : ℝ) (h : Real.sin (30 * Real.pi / 180 + α) = Real.sqrt 3 / 2) :
  Real.cos (60 * Real.pi / 180 - α) = Real.sqrt 3 / 2 := by
  sorry

end cos_cofunction_identity_l904_90451


namespace intersection_locus_l904_90436

theorem intersection_locus
  (a b : ℝ) (a_gt_b : a > b) (b_gt_zero : b > 0)
  (ellipse_eq : ∀ x y : ℝ, (x^2)/(a^2) + (y^2)/(b^2) = 1) :
  ∃ (x y : ℝ), (x^2)/(a^2) - (y^2)/(b^2) = 1 :=
sorry

end intersection_locus_l904_90436


namespace sequence_formula_l904_90497

noncomputable def a (n : ℕ) : ℕ :=
if n = 0 then 1 else (a (n - 1)) + 2^(n-1)

theorem sequence_formula (n : ℕ) (h : n > 0) : 
    a n = 2^n - 1 := 
sorry

end sequence_formula_l904_90497


namespace c_a_plus_c_b_geq_a_a_plus_b_b_l904_90486

theorem c_a_plus_c_b_geq_a_a_plus_b_b (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (c : ℚ) (h : c = (a^(a+1) + b^(b+1)) / (a^a + b^b)) :
  c^a + c^b ≥ a^a + b^b :=
sorry

end c_a_plus_c_b_geq_a_a_plus_b_b_l904_90486


namespace maximum_value_attains_maximum_value_l904_90472

theorem maximum_value
  (a b c : ℝ)
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c)
  (h₃ : a + b + c = 1) :
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) ≤ 1 / 2 :=
sorry

theorem attains_maximum_value :
  ∃ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 1 ∧
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) = 1 / 2 :=
sorry

end maximum_value_attains_maximum_value_l904_90472


namespace divisible_by_5886_l904_90448

theorem divisible_by_5886 (r b c : ℕ) (h1 : (523000 + r * 1000 + b * 100 + c * 10) % 89 = 0) (h2 : r * b * c = 180) : 
  (523000 + r * 1000 + b * 100 + c * 10) % 5886 = 0 := 
sorry

end divisible_by_5886_l904_90448


namespace evaluate_expression_l904_90413

theorem evaluate_expression (a b : ℚ) (h1 : a + b = 4) (h2 : a - b = 2) :
  ( (a^2 - 6 * a * b + 9 * b^2) / (a^2 - 2 * a * b) / ((5 * b^2 / (a - 2 * b)) - (a + 2 * b)) - 1 / a ) = -1 / 3 :=
by
  sorry

end evaluate_expression_l904_90413


namespace count_red_balls_l904_90419

/-- Given conditions:
  - The total number of balls in the bag is 100.
  - There are 50 white, 20 green, 10 yellow, and 3 purple balls.
  - The probability that a ball will be neither red nor purple is 0.8.
  Prove that the number of red balls is 17. -/
theorem count_red_balls (total_balls white_balls green_balls yellow_balls purple_balls red_balls : ℕ)
  (h1 : total_balls = 100)
  (h2 : white_balls = 50)
  (h3 : green_balls = 20)
  (h4 : yellow_balls = 10)
  (h5 : purple_balls = 3)
  (h6 : (white_balls + green_balls + yellow_balls) = 80)
  (h7 : (white_balls + green_balls + yellow_balls) / (total_balls : ℝ) = 0.8) :
  red_balls = 17 :=
by
  sorry

end count_red_balls_l904_90419


namespace solve_eq1_solve_eq2_l904_90402

theorem solve_eq1 (x : ℝ) : 3 * x * (x + 3) = 2 * (x + 3) ↔ (x = -3 ∨ x = 2 / 3) :=
by sorry

theorem solve_eq2 (x : ℝ) : x^2 - 4 * x - 5 = 0 ↔ (x = 5 ∨ x = -1) :=
by sorry

end solve_eq1_solve_eq2_l904_90402


namespace ratio_of_triangles_in_octagon_l904_90467

-- Conditions
def regular_octagon_division : Prop := 
  let L := 1 -- Area of each small congruent right triangle
  let ABJ := 2 * L -- Area of triangle ABJ
  let ADE := 6 * L -- Area of triangle ADE
  (ABJ / ADE = (1:ℝ) / 3)

-- Statement
theorem ratio_of_triangles_in_octagon : regular_octagon_division := by
  sorry

end ratio_of_triangles_in_octagon_l904_90467


namespace y_gets_per_rupee_l904_90489

theorem y_gets_per_rupee (a p : ℝ) (ha : a * p = 63) (htotal : p + a * p + 0.3 * p = 245) : a = 0.63 :=
by
  sorry

end y_gets_per_rupee_l904_90489


namespace sufficient_but_not_necessary_condition_l904_90427

theorem sufficient_but_not_necessary_condition : ∀ (y : ℝ), (y = 2 → y^2 = 4) ∧ (y^2 = 4 → (y = 2 ∨ y = -2)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l904_90427


namespace marbles_lost_l904_90420

theorem marbles_lost (initial_marbles lost_marbles gifted_marbles remaining_marbles : ℕ) 
  (h_initial : initial_marbles = 85)
  (h_gifted : gifted_marbles = 25)
  (h_remaining : remaining_marbles = 43)
  (h_before_gifting : remaining_marbles + gifted_marbles = initial_marbles - lost_marbles) :
  lost_marbles = 17 :=
by
  sorry

end marbles_lost_l904_90420


namespace can_choose_P_l904_90442

-- Define the objects in the problem,
-- types, constants, and assumptions as per the problem statement.

theorem can_choose_P (cube : ℝ) (P Q R S T A B C D : ℝ)
  (edge_length : cube = 10)
  (AR_RB_eq_CS_SB : ∀ AR RB CS SB, (AR / RB = 7 / 3) ∧ (CS / SB = 7 / 3))
  : ∃ P, 2 * (Q - R) = (P - Q) + (R - S) := by
  sorry

end can_choose_P_l904_90442


namespace smallest_k_for_64k_greater_than_6_l904_90499

theorem smallest_k_for_64k_greater_than_6 : ∃ (k : ℕ), 64 ^ k > 6 ∧ ∀ m : ℕ, m < k → 64 ^ m ≤ 6 :=
by
  use 1
  sorry

end smallest_k_for_64k_greater_than_6_l904_90499


namespace triangle_area_l904_90421

theorem triangle_area (x y : ℝ) (h : 3 * x + y = 9) : 
  (1 / 2) * 3 * 9 = 13.5 :=
sorry

end triangle_area_l904_90421


namespace tank_a_height_l904_90431

theorem tank_a_height (h_B : ℝ) (C_A C_B : ℝ) (V_A : ℝ → ℝ) (V_B : ℝ) :
  C_A = 4 ∧ C_B = 10 ∧ h_B = 8 ∧ (∀ h_A : ℝ, V_A h_A = 0.10000000000000002 * V_B) →
  ∃ h_A : ℝ, h_A = 5 :=
by sorry

end tank_a_height_l904_90431


namespace blue_face_area_factor_l904_90418

def cube_side_length : ℕ := 13

def original_red_face_area : ℕ := 6 * cube_side_length ^ 2
def total_mini_cube_face_area : ℕ := 6 * cube_side_length ^ 3
def blue_face_area : ℕ := total_mini_cube_face_area - original_red_face_area

theorem blue_face_area_factor :
  (blue_face_area / original_red_face_area) = 12 :=
by
  sorry

end blue_face_area_factor_l904_90418


namespace sum_of_first_ten_terms_seq_l904_90493

def a₁ : ℤ := -5
def d : ℤ := 6
def n : ℕ := 10

theorem sum_of_first_ten_terms_seq : (n * (a₁ + a₁ + (n - 1) * d)) / 2 = 220 :=
by
  sorry

end sum_of_first_ten_terms_seq_l904_90493


namespace robert_more_photos_than_claire_l904_90417

theorem robert_more_photos_than_claire
  (claire_photos : ℕ)
  (Lisa_photos : ℕ)
  (Robert_photos : ℕ)
  (Claire_takes_photos : claire_photos = 12)
  (Lisa_takes_photos : Lisa_photos = 3 * claire_photos)
  (Lisa_and_Robert_same_photos : Lisa_photos = Robert_photos) :
  Robert_photos - claire_photos = 24 := by
    sorry

end robert_more_photos_than_claire_l904_90417


namespace digits_base_d_l904_90414

theorem digits_base_d (d A B : ℕ) (h₀ : d > 7) (h₁ : A < d) (h₂ : B < d) 
  (h₃ : A * d + B + B * d + A = 2 * d^2 + 2) : A - B = 2 :=
by
  sorry

end digits_base_d_l904_90414


namespace snowman_volume_l904_90492

noncomputable def volume_snowman (r₁ r₂ r₃ r_c h_c : ℝ) : ℝ :=
  (4 / 3 * Real.pi * r₁^3) + (4 / 3 * Real.pi * r₂^3) + (4 / 3 * Real.pi * r₃^3) + (Real.pi * r_c^2 * h_c)

theorem snowman_volume 
  : volume_snowman 4 6 8 3 5 = 1101 * Real.pi := 
by 
  sorry

end snowman_volume_l904_90492


namespace order_of_f_l904_90440

variable (f : ℝ → ℝ)

/-- Conditions:
1. f is an even function for all x ∈ ℝ
2. f is increasing on [0, +∞)
Question:
Prove that the order of f(-2), f(-π), f(3) is f(-2) < f(3) < f(-π) -/
theorem order_of_f (h_even : ∀ x : ℝ, f (-x) = f x)
                   (h_incr : ∀ {x y : ℝ}, 0 ≤ x → x ≤ y → f x ≤ f y) : 
                   f (-2) < f 3 ∧ f 3 < f (-π) :=
by
  sorry

end order_of_f_l904_90440


namespace number_of_five_digit_numbers_without_repeating_digits_with_two_adjacent_odds_is_72_l904_90464

def five_digit_number_count : Nat :=
  -- Number of ways to select and arrange odd digits in two groups
  let group_odd_digits := (Nat.choose 3 2) * (Nat.factorial 2)
  -- Number of ways to arrange the even digits
  let arrange_even_digits := Nat.factorial 2
  -- Number of ways to insert two groups of odd digits into the gaps among even digits
  let insert_odd_groups := (Nat.factorial 3)
  -- Total ways
  group_odd_digits * arrange_even_digits * arrange_even_digits * insert_odd_groups

theorem number_of_five_digit_numbers_without_repeating_digits_with_two_adjacent_odds_is_72 :
  five_digit_number_count = 72 :=
by
  -- Placeholder for proof
  sorry

end number_of_five_digit_numbers_without_repeating_digits_with_two_adjacent_odds_is_72_l904_90464


namespace complex_fraction_simplification_l904_90411

theorem complex_fraction_simplification (a b c d : ℂ) (h₁ : a = 3 + i) (h₂ : b = 1 + i) (h₃ : c = 1 - i) (h₄ : d = 2 - i) : (a / b) = d := by
  sorry

end complex_fraction_simplification_l904_90411


namespace minimize_expression_l904_90410

theorem minimize_expression : ∃ c : ℝ, (∀ x : ℝ, (1/3 * x^2 + 7*x - 4) ≥ (1/3 * c^2 + 7*c - 4)) ∧ (c = -21/2) :=
sorry

end minimize_expression_l904_90410


namespace initial_eggs_count_l904_90446

theorem initial_eggs_count (harry_adds : ℕ) (total_eggs : ℕ) (initial_eggs : ℕ) :
  harry_adds = 5 → total_eggs = 52 → initial_eggs = total_eggs - harry_adds → initial_eggs = 47 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end initial_eggs_count_l904_90446


namespace find_ages_l904_90473

theorem find_ages (M F S : ℕ) 
  (h1 : M = 2 * F / 5)
  (h2 : M + 10 = (F + 10) / 2)
  (h3 : S + 10 = 3 * (F + 10) / 4) :
  M = 20 ∧ F = 50 ∧ S = 35 := 
by
  sorry

end find_ages_l904_90473


namespace pq_sum_l904_90496

open Real

section Problem
variables (p q : ℝ)
  (hp : p^3 - 21 * p^2 + 35 * p - 105 = 0)
  (hq : 5 * q^3 - 35 * q^2 - 175 * q + 1225 = 0)

theorem pq_sum : p + q = 21 / 2 :=
sorry
end Problem

end pq_sum_l904_90496


namespace final_price_is_correct_l904_90475

def cost_cucumber : ℝ := 5
def cost_tomato : ℝ := cost_cucumber - 0.2 * cost_cucumber
def cost_bell_pepper : ℝ := cost_cucumber + 0.5 * cost_cucumber
def total_cost_before_discount : ℝ := 2 * cost_tomato + 3 * cost_cucumber + 4 * cost_bell_pepper
def final_price : ℝ := total_cost_before_discount - 0.1 * total_cost_before_discount

theorem final_price_is_correct : final_price = 47.7 := sorry

end final_price_is_correct_l904_90475


namespace partition_count_l904_90474

noncomputable def count_partition (n : ℕ) : ℕ :=
  -- Function that counts the number of ways to partition n as per the given conditions
  n

theorem partition_count (n : ℕ) (h : n > 0) :
  count_partition n = n :=
sorry

end partition_count_l904_90474


namespace sufficient_m_value_l904_90453

theorem sufficient_m_value (m : ℕ) : 
  ((8 = m ∨ 9 = m) → 
  (m^2 + m^4 + m^6 + m^8 ≥ 6^3 + 6^5 + 6^7 + 6^9)) := 
by 
  sorry

end sufficient_m_value_l904_90453


namespace book_page_count_l904_90445

theorem book_page_count (pages_per_night : ℝ) (nights : ℝ) : pages_per_night = 120.0 → nights = 10.0 → pages_per_night * nights = 1200.0 :=
by
  sorry

end book_page_count_l904_90445


namespace expected_coin_worth_is_two_l904_90443

-- Define the conditions
def p_heads : ℚ := 4 / 5
def p_tails : ℚ := 1 / 5
def gain_heads : ℚ := 5
def loss_tails : ℚ := -10

-- Expected worth calculation
def expected_worth : ℚ := (p_heads * gain_heads) + (p_tails * loss_tails)

-- Lean 4 statement to prove
theorem expected_coin_worth_is_two : expected_worth = 2 := by
  sorry

end expected_coin_worth_is_two_l904_90443


namespace count_color_patterns_l904_90424

def regions := 6
def colors := 3

theorem count_color_patterns (h1 : regions = 6) (h2 : colors = 3) :
  3^6 - 3 * 2^6 + 3 * 1^6 = 540 := by
  sorry

end count_color_patterns_l904_90424


namespace smallest_angle_in_trapezoid_l904_90490

theorem smallest_angle_in_trapezoid 
  (a d : ℝ) 
  (h1 : a + 2 * d = 150) 
  (h2 : a + d + a + 2 * d = 180) : 
  a = 90 := 
sorry

end smallest_angle_in_trapezoid_l904_90490


namespace measure_of_angleA_l904_90408

theorem measure_of_angleA (A B : ℝ) 
  (h1 : ∀ (x : ℝ), x ≠ A → x ≠ B → x ≠ (3 * B - 20) → (3 * x - 20 ≠ A)) 
  (h2 : A = 3 * B - 20) :
  A = 10 ∨ A = 130 :=
by
  sorry

end measure_of_angleA_l904_90408


namespace tenfold_largest_two_digit_number_l904_90481

def largest_two_digit_number : ℕ := 99

theorem tenfold_largest_two_digit_number :
  10 * largest_two_digit_number = 990 :=
by
  sorry

end tenfold_largest_two_digit_number_l904_90481


namespace sum_of_possible_values_of_k_l904_90471

theorem sum_of_possible_values_of_k :
  (∀ j k : ℕ, 0 < j ∧ 0 < k → (1 / (j:ℚ)) + (1 / (k:ℚ)) = (1 / 5) → k = 6 ∨ k = 10 ∨ k = 30) ∧ 
  (46 = 6 + 10 + 30) :=
by
  sorry

end sum_of_possible_values_of_k_l904_90471


namespace percentage_cut_away_in_second_week_l904_90468

theorem percentage_cut_away_in_second_week :
  ∃(x : ℝ), (x / 100) * 142.5 * 0.9 = 109.0125 ∧ x = 15 :=
by
  sorry

end percentage_cut_away_in_second_week_l904_90468


namespace evaluate_expression_l904_90495

theorem evaluate_expression (x : ℝ) (h1 : x^3 + 2 ≠ 0) (h2 : x^3 - 2 ≠ 0) :
  (( (x+2)^3 * (x^2-x+2)^3 / (x^3+2)^3 )^3 * ( (x-2)^3 * (x^2+x+2)^3 / (x^3-2)^3 )^3 ) = 1 :=
by
  sorry

end evaluate_expression_l904_90495


namespace cargo_to_passenger_ratio_l904_90483

def total_cars : Nat := 71
def passenger_cars : Nat := 44
def engine_and_caboose : Nat := 2
def cargo_cars : Nat := total_cars - passenger_cars - engine_and_caboose

theorem cargo_to_passenger_ratio : cargo_cars = 25 ∧ passenger_cars = 44 →
  cargo_cars.toFloat / passenger_cars.toFloat = 25.0 / 44.0 :=
by
  intros h
  rw [h.1]
  rw [h.2]
  sorry

end cargo_to_passenger_ratio_l904_90483


namespace quadratic_function_correct_l904_90479

-- Defining the quadratic function a
def quadratic_function (x : ℝ) : ℝ := 2 * x^2 - 14 * x + 20

-- Theorem stating that the quadratic function passes through the points (2, 0) and (5, 0)
theorem quadratic_function_correct : 
  quadratic_function 2 = 0 ∧ quadratic_function 5 = 0 := 
by
  -- these proofs are skipped with sorry for now
  sorry

end quadratic_function_correct_l904_90479


namespace original_number_of_workers_l904_90462

theorem original_number_of_workers (W A : ℕ)
  (h1 : W * 75 = A)
  (h2 : (W + 10) * 65 = A) :
  W = 65 :=
by
  sorry

end original_number_of_workers_l904_90462


namespace crow_speed_l904_90484

/-- Definitions from conditions -/
def distance_between_nest_and_ditch : ℝ := 250 -- in meters
def total_trips : ℕ := 15
def total_hours : ℝ := 1.5 -- hours

/-- The statement to be proved -/
theorem crow_speed :
  let distance_per_trip := 2 * distance_between_nest_and_ditch
  let total_distance := (total_trips : ℝ) * distance_per_trip / 1000 -- convert to kilometers
  let speed := total_distance / total_hours
  speed = 5 := by
  let distance_per_trip := 2 * distance_between_nest_and_ditch
  let total_distance := (total_trips : ℝ) * distance_per_trip / 1000
  let speed := total_distance / total_hours
  sorry

end crow_speed_l904_90484


namespace find_student_hourly_rate_l904_90405

-- Definitions based on conditions
def janitor_work_time : ℝ := 8  -- Janitor can clean the school in 8 hours
def student_work_time : ℝ := 20  -- Student can clean the school in 20 hours
def janitor_hourly_rate : ℝ := 21  -- Janitor is paid $21 per hour
def cost_difference : ℝ := 8  -- The cost difference between janitor alone and both together is $8

-- The value we need to prove
def student_hourly_rate := 7

theorem find_student_hourly_rate
  (janitor_work_time : ℝ)
  (student_work_time : ℝ)
  (janitor_hourly_rate : ℝ)
  (cost_difference : ℝ) :
  S = 7 :=
by
  -- Calculations and logic can be filled here to prove the theorem
  sorry

end find_student_hourly_rate_l904_90405


namespace xiao_zhao_physical_education_grade_l904_90432

def classPerformanceScore : ℝ := 40
def midtermExamScore : ℝ := 50
def finalExamScore : ℝ := 45

def classPerformanceWeight : ℝ := 0.3
def midtermExamWeight : ℝ := 0.2
def finalExamWeight : ℝ := 0.5

def overallGrade : ℝ :=
  (classPerformanceScore * classPerformanceWeight) +
  (midtermExamScore * midtermExamWeight) +
  (finalExamScore * finalExamWeight)

theorem xiao_zhao_physical_education_grade : overallGrade = 44.5 := by
  sorry

end xiao_zhao_physical_education_grade_l904_90432


namespace reciprocal_of_neg_5_l904_90452

theorem reciprocal_of_neg_5 : (∃ r : ℚ, -5 * r = 1) ∧ r = -1 / 5 :=
by sorry

end reciprocal_of_neg_5_l904_90452


namespace least_number_to_add_l904_90460

theorem least_number_to_add (n : ℕ) : 
  (∀ k : ℕ, n = 1 + k * 425 ↔ n + 1019 % 425 = 0) → n = 256 := 
sorry

end least_number_to_add_l904_90460


namespace largest_divisor_of_three_consecutive_even_integers_is_sixteen_l904_90450

theorem largest_divisor_of_three_consecutive_even_integers_is_sixteen (n : ℕ) :
  ∃ d : ℕ, d = 16 ∧ 16 ∣ ((2 * n) * (2 * n + 2) * (2 * n + 4)) :=
by
  sorry

end largest_divisor_of_three_consecutive_even_integers_is_sixteen_l904_90450


namespace joan_took_marbles_l904_90433

-- Each condition is used as a definition.
def original_marbles : ℕ := 86
def remaining_marbles : ℕ := 61

-- The theorem states that the number of marbles Joan took equals 25.
theorem joan_took_marbles : (original_marbles - remaining_marbles) = 25 := by
  sorry    -- Add sorry to skip the proof.

end joan_took_marbles_l904_90433


namespace number_of_students_l904_90459

theorem number_of_students (n : ℕ) (h1 : 90 - n = n / 2) : n = 60 :=
by
  sorry

end number_of_students_l904_90459


namespace john_books_nights_l904_90430

theorem john_books_nights (n : ℕ) (cost_per_night discount amount_paid : ℕ) 
  (h1 : cost_per_night = 250)
  (h2 : discount = 100)
  (h3 : amount_paid = 650)
  (h4 : amount_paid = cost_per_night * n - discount) : 
  n = 3 :=
by
  sorry

end john_books_nights_l904_90430


namespace campaign_funds_total_l904_90416

variable (X : ℝ)

def campaign_funds (friends family remaining : ℝ) : Prop :=
  friends = 0.40 * X ∧
  family = 0.30 * (X - friends) ∧
  remaining = X - (friends + family) ∧
  remaining = 4200

theorem campaign_funds_total (X_val : ℝ) (friends family remaining : ℝ)
    (h : campaign_funds X friends family remaining) : X = 10000 :=
by
  have h_friends : friends = 0.40 * X := h.1
  have h_family : family = 0.30 * (X - friends) := h.2.1
  have h_remaining : remaining = X - (friends + family) := h.2.2.1
  have h_remaining_amount : remaining = 4200 := h.2.2.2
  sorry

end campaign_funds_total_l904_90416


namespace sum_of_uv_l904_90455

theorem sum_of_uv (u v : ℕ) (hu : 0 < u) (hv : 0 < v) (hv_lt_hu : v < u)
  (area_pent : 6 * u * v = 500) : u + v = 19 :=
by
  sorry

end sum_of_uv_l904_90455


namespace total_sticks_used_l904_90422

-- Define the number of sides an octagon has
def octagon_sides : ℕ := 8

-- Define the number of sticks each subsequent octagon needs, sharing one side with the previous one
def additional_sticks_per_octagon : ℕ := 7

-- Define the total number of octagons in the row
def total_octagons : ℕ := 700

-- Define the total number of sticks used
def total_sticks : ℕ := 
  let first_sticks := octagon_sides
  let additional_sticks := additional_sticks_per_octagon * (total_octagons - 1)
  first_sticks + additional_sticks

-- Statement to prove
theorem total_sticks_used : total_sticks = 4901 := by
  sorry

end total_sticks_used_l904_90422


namespace mabel_tomatoes_l904_90429

theorem mabel_tomatoes :
  ∃ (t1 t2 t3 t4 : ℕ), 
    t1 = 8 ∧ 
    t2 = t1 + 4 ∧ 
    t3 = 3 * (t1 + t2) ∧ 
    t4 = 3 * (t1 + t2) ∧ 
    (t1 + t2 + t3 + t4) = 140 :=
by
  sorry

end mabel_tomatoes_l904_90429


namespace min_cost_open_top_rectangular_pool_l904_90498

theorem min_cost_open_top_rectangular_pool
  (volume : ℝ)
  (depth : ℝ)
  (cost_bottom_per_sqm : ℝ)
  (cost_walls_per_sqm : ℝ)
  (h1 : volume = 18)
  (h2 : depth = 2)
  (h3 : cost_bottom_per_sqm = 200)
  (h4 : cost_walls_per_sqm = 150) :
  ∃ (min_cost : ℝ), min_cost = 5400 :=
by
  sorry

end min_cost_open_top_rectangular_pool_l904_90498


namespace width_of_shop_l904_90409

theorem width_of_shop 
  (monthly_rent : ℕ) 
  (length : ℕ) 
  (annual_rent_per_sqft : ℕ) 
  (h1 : monthly_rent = 3600) 
  (h2 : length = 18) 
  (h3 : annual_rent_per_sqft = 120) :
  ∃ width : ℕ, width = 20 :=
by
  sorry

end width_of_shop_l904_90409


namespace polar_to_cartesian_l904_90458

theorem polar_to_cartesian (θ ρ x y : ℝ) (h1 : ρ = 2 * Real.sin θ) (h2 : x = ρ * Real.cos θ) (h3 : y = ρ * Real.sin θ) :
  x^2 + (y - 1)^2 = 1 :=
sorry

end polar_to_cartesian_l904_90458


namespace fourth_quadrant_point_l904_90447

theorem fourth_quadrant_point (a : ℤ) (h1 : 2 * a + 6 > 0) (h2 : 3 * a + 3 < 0) :
  (2 * a + 6, 3 * a + 3) = (2, -3) :=
sorry

end fourth_quadrant_point_l904_90447


namespace sixty_three_times_fifty_seven_l904_90438

theorem sixty_three_times_fifty_seven : 63 * 57 = 3591 := by
  sorry

end sixty_three_times_fifty_seven_l904_90438


namespace first_player_can_ensure_distinct_rational_roots_l904_90488

theorem first_player_can_ensure_distinct_rational_roots :
  ∃ (a b c : ℚ), a + b + c = 0 ∧ (∀ x : ℚ, x^2 + (b/a) * x + (c/a) = 0 → False) :=
by
  sorry

end first_player_can_ensure_distinct_rational_roots_l904_90488


namespace largest_y_l904_90441

theorem largest_y (y : ℝ) (h : (⌊y⌋ / y) = 8 / 9) : y ≤ 63 / 8 :=
sorry

end largest_y_l904_90441


namespace sum_of_squares_of_roots_l904_90482

theorem sum_of_squares_of_roots :
  let a := 1
  let b := 8
  let c := -12
  let r1_r2_sum := -(b:ℝ) / a
  let r1_r2_product := (c:ℝ) / a
  (r1_r2_sum) ^ 2 - 2 * r1_r2_product = 88 :=
by
  sorry

end sum_of_squares_of_roots_l904_90482


namespace hens_count_l904_90401

theorem hens_count
  (H C : ℕ)
  (heads_eq : H + C = 48)
  (feet_eq : 2 * H + 4 * C = 136) :
  H = 28 :=
by
  sorry

end hens_count_l904_90401


namespace smallest_among_neg2_cube_neg3_square_neg_neg1_l904_90463

def smallest_among (a b c : ℤ) : ℤ :=
if a < b then
  if a < c then a else c
else
  if b < c then b else c

theorem smallest_among_neg2_cube_neg3_square_neg_neg1 :
  smallest_among ((-2)^3) (-(3^2)) (-(-1)) = -(3^2) :=
by
  sorry

end smallest_among_neg2_cube_neg3_square_neg_neg1_l904_90463


namespace car_tank_capacity_l904_90428

theorem car_tank_capacity
  (speed : ℝ) (usage_rate : ℝ) (time : ℝ) (used_fraction : ℝ) (distance : ℝ := speed * time) (gallons_used : ℝ := distance / usage_rate) 
  (fuel_used : ℝ := 10) (tank_capacity : ℝ := fuel_used / used_fraction)
  (h1 : speed = 60) (h2 : usage_rate = 30) (h3 : time = 5) (h4 : used_fraction = 0.8333333333333334) : 
  tank_capacity = 12 :=
by
  sorry

end car_tank_capacity_l904_90428


namespace mary_has_10_blue_marbles_l904_90434

-- Define the number of blue marbles Dan has
def dan_marbles : ℕ := 5

-- Define the factor by which Mary has more blue marbles than Dan
def factor : ℕ := 2

-- Define the number of blue marbles Mary has
def mary_marbles : ℕ := factor * dan_marbles

-- The theorem statement: Mary has 10 blue marbles
theorem mary_has_10_blue_marbles : mary_marbles = 10 :=
by
  -- Proof goes here
  sorry

end mary_has_10_blue_marbles_l904_90434


namespace ratio_proof_l904_90454

-- Define x and y as real numbers
variables (x y : ℝ)
-- Define the given condition
def given_condition : Prop := (3 * x - 2 * y) / (2 * x + y) = 3 / 4
-- Define the result to prove
def result : Prop := x / y = 11 / 6

-- State the theorem
theorem ratio_proof (h : given_condition x y) : result x y :=
by 
  sorry

end ratio_proof_l904_90454


namespace inequality_proof_l904_90461

theorem inequality_proof {x y z : ℝ} (n : ℕ) 
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x + y + z = 1)
  : (x^4 / (y * (1 - y^n))) + (y^4 / (z * (1 - z^n))) + (z^4 / (x * (1 - x^n))) 
    ≥ (3^n) / (3^(n - 2) - 9) :=
by
  sorry

end inequality_proof_l904_90461


namespace proof_problem_l904_90449

theorem proof_problem 
  (a b c : ℝ) 
  (h1 : ∀ x, (x < -4 ∨ (23 ≤ x ∧ x ≤ 27)) ↔ ((x - a) * (x - b) / (x - c) ≤ 0))
  (h2 : a < b) : 
  a + 2 * b + 3 * c = 65 :=
sorry

end proof_problem_l904_90449


namespace polynomial_remainder_l904_90412

theorem polynomial_remainder 
  (y: ℤ) 
  (root_cond: y^3 + y^2 + y + 1 = 0) 
  (beta_is_root: ∃ β: ℚ, β^3 + β^2 + β + 1 = 0) 
  (beta_four: ∀ β: ℚ, β^3 + β^2 + β + 1 = 0 → β^4 = 1) : 
  ∃ q r, (y^20 + y^15 + y^10 + y^5 + 1) = q * (y^3 + y^2 + y + 1) + r ∧ (r = 1) :=
by
  sorry

end polynomial_remainder_l904_90412


namespace problem_statement_l904_90407

theorem problem_statement (a b c x y z : ℂ)
  (h1 : a = (b + c) / (x - 2))
  (h2 : b = (c + a) / (y - 2))
  (h3 : c = (a + b) / (z - 2))
  (h4 : x * y + y * z + z * x = 67)
  (h5 : x + y + z = 2010) :
  x * y * z = -5892 :=
by {
  sorry
}

end problem_statement_l904_90407


namespace loss_equals_cost_price_of_some_balls_l904_90444

-- Conditions
def cost_price_per_ball := 60
def selling_price_for_17_balls := 720
def number_of_balls := 17

-- Calculations
def total_cost_price := number_of_balls * cost_price_per_ball
def loss := total_cost_price - selling_price_for_17_balls

-- Proof statement
theorem loss_equals_cost_price_of_some_balls : (loss / cost_price_per_ball) = 5 :=
by
  -- Proof would go here
  sorry

end loss_equals_cost_price_of_some_balls_l904_90444


namespace calculate_expression_l904_90423

def thirteen_power_thirteen_div_thirteen_power_twelve := 13 ^ 13 / 13 ^ 12
def expression := (thirteen_power_thirteen_div_thirteen_power_twelve ^ 3) * (3 ^ 3)
/- We define the main statement to be proven -/
theorem calculate_expression : (expression / 2 ^ 6) = 926 := sorry

end calculate_expression_l904_90423


namespace fifteen_percent_of_x_is_ninety_l904_90400

theorem fifteen_percent_of_x_is_ninety :
  ∃ (x : ℝ), (15 / 100) * x = 90 ↔ x = 600 :=
by
  sorry

end fifteen_percent_of_x_is_ninety_l904_90400


namespace simplify_to_quadratic_form_l904_90494

noncomputable def simplify_expression (p : ℝ) : ℝ :=
  ((6 * p + 2) - 3 * p * 5) ^ 2 + (5 - 2 / 4) * (8 * p - 12)

theorem simplify_to_quadratic_form (p : ℝ) : simplify_expression p = 81 * p ^ 2 - 50 :=
sorry

end simplify_to_quadratic_form_l904_90494


namespace Kelly_current_baking_powder_l904_90466

-- Definitions based on conditions
def yesterday_amount : ℝ := 0.4
def difference : ℝ := 0.1
def current_amount : ℝ := yesterday_amount - difference

-- Statement to prove the question == answer given the conditions
theorem Kelly_current_baking_powder : current_amount = 0.3 := 
by
  sorry

end Kelly_current_baking_powder_l904_90466


namespace number_of_boys_l904_90435

variables (total_girls total_teachers total_people : ℕ)
variables (total_girls_eq : total_girls = 315) (total_teachers_eq : total_teachers = 772) (total_people_eq : total_people = 1396)

theorem number_of_boys (total_boys : ℕ) : total_boys = total_people - total_girls - total_teachers :=
by sorry

end number_of_boys_l904_90435


namespace brown_eggs_survived_l904_90480

-- Conditions
variables (B : ℕ)  -- Number of brown eggs that survived

-- States that Linda had three times as many white eggs as brown eggs before the fall
def white_eggs_eq_3_times_brown : Prop := 3 * B + B = 12

-- Theorem statement
theorem brown_eggs_survived (h : white_eggs_eq_3_times_brown B) : B = 3 :=
sorry

end brown_eggs_survived_l904_90480


namespace tomatoes_price_per_pound_l904_90426

noncomputable def price_per_pound (cost_per_pound : ℝ) (loss_percent : ℝ) (profit_percent : ℝ) : ℝ :=
  let remaining_percent := 1 - loss_percent / 100
  let desired_total := (1 + profit_percent / 100) * cost_per_pound
  desired_total / remaining_percent

theorem tomatoes_price_per_pound :
  price_per_pound 0.80 15 8 = 1.02 :=
by
  sorry

end tomatoes_price_per_pound_l904_90426


namespace find_ellipse_l904_90469

noncomputable def standard_equation_ellipse (x y : ℝ) : Prop :=
  (x^2 / 9 + y^2 / 3 = 1)
  ∨ (x^2 / 18 + y^2 / 9 = 1)
  ∨ (y^2 / (45 / 2) + x^2 / (45 / 4) = 1)

variables 
  (P1 P2 : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (a b : ℝ)

def passes_through_points (P1 P2 : ℝ × ℝ) : Prop :=
  ∀ equation : (ℝ → ℝ → Prop), 
    equation P1.1 P1.2 ∧ equation P2.1 P2.2

def focus_conditions (focus : ℝ × ℝ) : Prop :=
  -- Condition indicating focus, relationship with the minor axis etc., will be precisely defined here
  true -- Placeholder, needs correct mathematical condition

theorem find_ellipse : 
  passes_through_points P1 P2 
  → focus_conditions focus 
  → standard_equation_ellipse x y :=
sorry

end find_ellipse_l904_90469


namespace div_by_5_mul_diff_l904_90457

theorem div_by_5_mul_diff (x y z : ℤ) (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) :
  5 ∣ ((x - y)^5 + (y - z)^5 + (z - x)^5) :=
by
  sorry

end div_by_5_mul_diff_l904_90457


namespace greatest_power_of_3_l904_90478

theorem greatest_power_of_3 (n : ℕ) : 
  (n = 603) → 
  3^603 ∣ (15^n - 6^n + 3^n) ∧ ¬ (3^(603+1) ∣ (15^n - 6^n + 3^n)) :=
by
  intro hn
  cases hn
  sorry

end greatest_power_of_3_l904_90478


namespace max_height_reached_threat_to_object_at_70km_l904_90491

noncomputable def initial_acceleration : ℝ := 20 -- m/s^2
noncomputable def duration : ℝ := 50 -- seconds
noncomputable def gravity : ℝ := 10 -- m/s^2
noncomputable def height_at_max_time : ℝ := 75000 -- meters (75km)

-- Proof that the maximum height reached is 75 km
theorem max_height_reached (a τ g : ℝ) (H : ℝ) (h₀: a = initial_acceleration) (h₁: τ = duration) (h₂: g = gravity) (h₃: H = height_at_max_time) :
  H = 75 * 1000 := 
sorry

-- Proof that the rocket poses a threat to an object located at 70 km
theorem threat_to_object_at_70km (a τ g : ℝ) (H : ℝ) (h₀: a = initial_acceleration) (h₁: τ = duration) (h₂: g = gravity) (h₃: H = height_at_max_time) :
  H > 70 * 1000 :=
sorry

end max_height_reached_threat_to_object_at_70km_l904_90491
