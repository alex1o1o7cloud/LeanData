import Mathlib

namespace NUMINAMATH_GPT_compare_fractions_compare_integers_l1734_173425

-- First comparison: Prove -4/7 > -2/3
theorem compare_fractions : - (4 : ℚ) / 7 > - (2 : ℚ) / 3 := 
by sorry

-- Second comparison: Prove -(-7) > -| -7 |
theorem compare_integers : -(-7) > -abs (-7) := 
by sorry

end NUMINAMATH_GPT_compare_fractions_compare_integers_l1734_173425


namespace NUMINAMATH_GPT_total_miles_walked_l1734_173458

-- Definition of the conditions
def num_islands : ℕ := 4
def miles_per_day_island1 : ℕ := 20
def miles_per_day_island2 : ℕ := 25
def days_per_island : ℚ := 1.5

-- Mathematically Equivalent Proof Problem
theorem total_miles_walked :
  let total_miles_island1 := 2 * (miles_per_day_island1 * days_per_island)
  let total_miles_island2 := 2 * (miles_per_day_island2 * days_per_island)
  total_miles_island1 + total_miles_island2 = 135 := by
  sorry

end NUMINAMATH_GPT_total_miles_walked_l1734_173458


namespace NUMINAMATH_GPT_equilateral_triangle_l1734_173407

noncomputable def angles_arithmetic_seq (A B C : ℝ) : Prop := B - A = C - B

noncomputable def sides_geometric_seq (a b c : ℝ) : Prop := b / a = c / b

theorem equilateral_triangle 
  (A B C a b c : ℝ) 
  (h_angles : angles_arithmetic_seq A B C) 
  (h_sides : sides_geometric_seq a b c) 
  (h_triangle : A + B + C = π) 
  (h_pos_sides : a > 0 ∧ b > 0 ∧ c > 0) :
  (A = B ∧ B = C) ∧ (a = b ∧ b = c) :=
sorry

end NUMINAMATH_GPT_equilateral_triangle_l1734_173407


namespace NUMINAMATH_GPT_gcd_gx_x_l1734_173423

def g (x : ℕ) : ℕ := (5 * x + 3) * (11 * x + 2) * (17 * x + 7) * (3 * x + 8)

theorem gcd_gx_x (x : ℕ) (h : 27720 ∣ x) : Nat.gcd (g x) x = 168 := by
  sorry

end NUMINAMATH_GPT_gcd_gx_x_l1734_173423


namespace NUMINAMATH_GPT_infinite_pairs_natural_numbers_l1734_173436

theorem infinite_pairs_natural_numbers :
  ∃ (infinite_pairs : ℕ × ℕ → Prop), (∀ a b : ℕ, infinite_pairs (a, b) ↔ (b ∣ (a^2 + 1) ∧ a ∣ (b^2 + 1))) ∧
    ∀ n : ℕ, ∃ (a b : ℕ), infinite_pairs (a, b) :=
sorry

end NUMINAMATH_GPT_infinite_pairs_natural_numbers_l1734_173436


namespace NUMINAMATH_GPT_entrepreneurs_not_attending_any_session_l1734_173427

theorem entrepreneurs_not_attending_any_session 
  (total_entrepreneurs : ℕ) 
  (digital_marketing_attendees : ℕ) 
  (e_commerce_attendees : ℕ) 
  (both_sessions_attendees : ℕ)
  (h1 : total_entrepreneurs = 40)
  (h2 : digital_marketing_attendees = 22) 
  (h3 : e_commerce_attendees = 18) 
  (h4 : both_sessions_attendees = 8) : 
  total_entrepreneurs - (digital_marketing_attendees + e_commerce_attendees - both_sessions_attendees) = 8 :=
by sorry

end NUMINAMATH_GPT_entrepreneurs_not_attending_any_session_l1734_173427


namespace NUMINAMATH_GPT_tennis_tournament_boxes_needed_l1734_173492

theorem tennis_tournament_boxes_needed (n : ℕ) (h : n = 199) : 
  ∃ m, m = 198 ∧
    (∀ k, k < n → (n - k - 1 = m)) :=
by
  sorry

end NUMINAMATH_GPT_tennis_tournament_boxes_needed_l1734_173492


namespace NUMINAMATH_GPT_find_b_l1734_173498

def direction_vector (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
  (x2 - x1, y2 - y1)

theorem find_b (b : ℝ)
  (hx1 : ℝ := -3) (hy1 : ℝ := 1) (hx2 : ℝ := 0) (hy2 : ℝ := 4)
  (hdir : direction_vector hx1 hy1 hx2 hy2 = (3, b)) :
  b = 3 :=
by
  -- Mathematical proof of b = 3 goes here
  sorry

end NUMINAMATH_GPT_find_b_l1734_173498


namespace NUMINAMATH_GPT_find_m_for_parallel_lines_l1734_173453

theorem find_m_for_parallel_lines (m : ℝ) :
  (∀ x y : ℝ, 3 * x - y + 2 = 0 → x + m * y - 3 = 0) →
  m = -1 / 3 := sorry

end NUMINAMATH_GPT_find_m_for_parallel_lines_l1734_173453


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3_not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3_sufficient_but_not_necessary_0_lt_x_lt_5_for_abs_x_minus_2_lt_3_l1734_173409

variable (x : ℝ)

theorem sufficient_but_not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3 :
        (0 < x ∧ x < 5) → (|x - 2| < 3) :=
by sorry

theorem not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3 :
        (|x - 2| < 3) → (0 < x ∧ x < 5) :=
by sorry

theorem sufficient_but_not_necessary_0_lt_x_lt_5_for_abs_x_minus_2_lt_3 :
        (0 < x ∧ x < 5) ↔ (|x - 2| < 3) → false :=
by sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3_not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3_sufficient_but_not_necessary_0_lt_x_lt_5_for_abs_x_minus_2_lt_3_l1734_173409


namespace NUMINAMATH_GPT_watermelons_with_seeds_l1734_173435

def ripe_watermelons : ℕ := 11
def unripe_watermelons : ℕ := 13
def seedless_watermelons : ℕ := 15
def total_watermelons := ripe_watermelons + unripe_watermelons

theorem watermelons_with_seeds :
  total_watermelons - seedless_watermelons = 9 :=
by
  sorry

end NUMINAMATH_GPT_watermelons_with_seeds_l1734_173435


namespace NUMINAMATH_GPT_mary_needs_to_add_6_25_more_cups_l1734_173465

def total_flour_needed : ℚ := 8.5
def flour_already_added : ℚ := 2.25
def flour_to_add : ℚ := total_flour_needed - flour_already_added

theorem mary_needs_to_add_6_25_more_cups :
  flour_to_add = 6.25 :=
sorry

end NUMINAMATH_GPT_mary_needs_to_add_6_25_more_cups_l1734_173465


namespace NUMINAMATH_GPT_max_three_kopecks_l1734_173456

def is_coin_placement_correct (n1 n2 n3 : ℕ) : Prop :=
  -- Conditions for the placement to be valid
  ∀ (i j : ℕ), i < j → 
  ((j - i > 1 → n1 = 0) ∧ (j - i > 2 → n2 = 0) ∧ (j - i > 3 → n3 = 0))

theorem max_three_kopecks (n1 n2 n3 : ℕ) (h : n1 + n2 + n3 = 101) (placement_correct : is_coin_placement_correct n1 n2 n3) :
  n3 = 25 ∨ n3 = 26 :=
sorry

end NUMINAMATH_GPT_max_three_kopecks_l1734_173456


namespace NUMINAMATH_GPT_calculate_r_when_n_is_3_l1734_173400

theorem calculate_r_when_n_is_3 : 
  ∀ (r s n : ℕ), r = 4^s - s → s = 3^n + 2 → n = 3 → r = 4^29 - 29 :=
by 
  intros r s n h1 h2 h3
  sorry

end NUMINAMATH_GPT_calculate_r_when_n_is_3_l1734_173400


namespace NUMINAMATH_GPT_abs_inequality_solution_set_l1734_173452

theorem abs_inequality_solution_set :
  { x : ℝ | |x - 1| + |x + 2| ≥ 5 } = { x : ℝ | x ≤ -3 } ∪ { x : ℝ | x ≥ 2 } := 
sorry

end NUMINAMATH_GPT_abs_inequality_solution_set_l1734_173452


namespace NUMINAMATH_GPT_original_wire_length_l1734_173437

theorem original_wire_length (S L : ℝ) (h1: S = 30) (h2: S = (3 / 5) * L) : S + L = 80 := by 
  sorry

end NUMINAMATH_GPT_original_wire_length_l1734_173437


namespace NUMINAMATH_GPT_evlyn_can_buy_grapes_l1734_173488

theorem evlyn_can_buy_grapes 
  (price_pears price_oranges price_lemons price_grapes : ℕ)
  (h1 : 10 * price_pears = 5 * price_oranges)
  (h2 : 4 * price_oranges = 6 * price_lemons)
  (h3 : 3 * price_lemons = 2 * price_grapes) :
  (20 * price_pears = 10 * price_grapes) :=
by
  -- The proof is omitted using sorry
  sorry

end NUMINAMATH_GPT_evlyn_can_buy_grapes_l1734_173488


namespace NUMINAMATH_GPT_polygon_sides_l1734_173426

theorem polygon_sides (n : ℕ) 
  (h : 3240 = 180 * (n - 2) - (360)) : n = 22 := 
by 
  sorry

end NUMINAMATH_GPT_polygon_sides_l1734_173426


namespace NUMINAMATH_GPT_rebecca_soda_left_l1734_173408

-- Definitions of the conditions
def total_bottles_purchased : ℕ := 3 * 6
def days_in_four_weeks : ℕ := 4 * 7
def total_half_bottles_drinks : ℕ := days_in_four_weeks
def total_whole_bottles_drinks : ℕ := total_half_bottles_drinks / 2

-- The final statement we aim to prove
theorem rebecca_soda_left : 
  total_bottles_purchased - total_whole_bottles_drinks = 4 := 
by
  -- proof is not required as per the guidelines
  sorry

end NUMINAMATH_GPT_rebecca_soda_left_l1734_173408


namespace NUMINAMATH_GPT_geometric_series_sum_l1734_173449

theorem geometric_series_sum :
  let a := 1
  let r := (1 : ℝ) / 5
  ∑' n : ℕ, a * r ^ n = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1734_173449


namespace NUMINAMATH_GPT_sin_double_angle_identity_l1734_173402

theorem sin_double_angle_identity (alpha : ℝ) (h : Real.cos (Real.pi / 4 - alpha) = -4 / 5) : 
  Real.sin (2 * alpha) = 7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_identity_l1734_173402


namespace NUMINAMATH_GPT_circumscribed_sphere_radius_l1734_173455

/-- Define the right triangular prism -/
structure RightTriangularPrism :=
(AB AC BC : ℝ)
(AA1 : ℝ)
(h_base : AB = 4 * Real.sqrt 2 ∧ AC = 4 * Real.sqrt 2 ∧ BC = 8)
(h_height : AA1 = 6)

/-- The condition that the base is an isosceles right-angled triangle -/
structure IsoscelesRightAngledTriangle :=
(A B C : ℝ)
(AB AC : ℝ)
(BC : ℝ)
(h_isosceles_right : AB = AC ∧ BC = Real.sqrt (AB^2 + AC^2))

/-- The main theorem stating the radius of the circumscribed sphere -/
theorem circumscribed_sphere_radius (prism : RightTriangularPrism) 
    (base : IsoscelesRightAngledTriangle) 
    (h_base_correct : base.AB = prism.AB ∧ base.AC = prism.AC ∧ base.BC = prism.BC):
    ∃ radius : ℝ, radius = 5 := 
by
    sorry

end NUMINAMATH_GPT_circumscribed_sphere_radius_l1734_173455


namespace NUMINAMATH_GPT_find_n_of_geometric_sum_l1734_173462

-- Define the first term and common ratio of the sequence
def a : ℚ := 1 / 3
def r : ℚ := 1 / 3

-- Define the sum of the first n terms of the geometric sequence
def S_n (n : ℕ) : ℚ := a * (1 - r^n) / (1 - r)

-- Mathematical statement to be proved
theorem find_n_of_geometric_sum (h : S_n 5 = 80 / 243) : ∃ n, S_n n = 80 / 243 ↔ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_n_of_geometric_sum_l1734_173462


namespace NUMINAMATH_GPT_vasim_share_l1734_173418

theorem vasim_share (x : ℕ) (F V R : ℕ) (h1 : F = 3 * x) (h2 : V = 5 * x) (h3 : R = 11 * x) (h4 : R - F = 2400) : V = 1500 :=
by sorry

end NUMINAMATH_GPT_vasim_share_l1734_173418


namespace NUMINAMATH_GPT_proof_probability_at_least_one_makes_both_shots_l1734_173431

-- Define the shooting percentages for Player A and Player B
def shooting_percentage_A : ℝ := 0.4
def shooting_percentage_B : ℝ := 0.5

-- Define the probability that Player A makes both shots
def prob_A_makes_both_shots : ℝ := shooting_percentage_A * shooting_percentage_A

-- Define the probability that Player B makes both shots
def prob_B_makes_both_shots : ℝ := shooting_percentage_B * shooting_percentage_B

-- Define the probability that neither makes both shots
def prob_neither_makes_both_shots : ℝ := (1 - prob_A_makes_both_shots) * (1 - prob_B_makes_both_shots)

-- Define the probability that at least one of them makes both shots
def prob_at_least_one_makes_both_shots : ℝ := 1 - prob_neither_makes_both_shots

-- Prove that the probability that at least one of them makes both shots is 0.37
theorem proof_probability_at_least_one_makes_both_shots :
  prob_at_least_one_makes_both_shots = 0.37 :=
sorry

end NUMINAMATH_GPT_proof_probability_at_least_one_makes_both_shots_l1734_173431


namespace NUMINAMATH_GPT_problem1_problem2_l1734_173447

-- For problem (1)
noncomputable def f (x : ℝ) := Real.sqrt ((1 - x) / (1 + x))

theorem problem1 (α : ℝ) (h_alpha : α ∈ Set.Ioo (Real.pi / 2) Real.pi) :
  f (Real.cos α) + f (-Real.cos α) = 2 / Real.sin α := by
  sorry

-- For problem (2)
theorem problem2 : Real.sin (Real.pi * 50 / 180) * (1 + Real.sqrt 3 * Real.tan (Real.pi * 10 / 180)) = 1 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1734_173447


namespace NUMINAMATH_GPT_symmetric_line_equation_l1734_173484

theorem symmetric_line_equation : ∀ (x y : ℝ), (2 * x + 3 * y - 6 = 0) ↔ (3 * (x + 2) + 2 * (-y - 2) + 16 = 0) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_line_equation_l1734_173484


namespace NUMINAMATH_GPT_expression_value_l1734_173439

theorem expression_value
  (x y z : ℝ)
  (hx : x = -5 / 4)
  (hy : y = -3 / 2)
  (hz : z = Real.sqrt 2) :
  -2 * x ^ 3 - y ^ 2 + Real.sin z = 53 / 32 + Real.sin (Real.sqrt 2) :=
by
  rw [hx, hy, hz]
  sorry

end NUMINAMATH_GPT_expression_value_l1734_173439


namespace NUMINAMATH_GPT_math_problem_l1734_173422

open Real

-- Conditions extracted from the problem
def cond1 (a b : ℝ) : Prop := -|2 - a| + b = 5
def cond2 (a b : ℝ) : Prop := -|8 - a| + b = 3
def cond3 (c d : ℝ) : Prop := |2 - c| + d = 5
def cond4 (c d : ℝ) : Prop := |8 - c| + d = 3
def cond5 (a c : ℝ) : Prop := 2 < a ∧ a < 8
def cond6 (a c : ℝ) : Prop := 2 < c ∧ c < 8

-- Proof problem: Given the conditions, prove that a + c = 10
theorem math_problem (a b c d : ℝ) (h1 : cond1 a b) (h2 : cond2 a b) (h3 : cond3 c d) (h4 : cond4 c d)
  (h5 : cond5 a c) (h6 : cond6 a c) : a + c = 10 := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l1734_173422


namespace NUMINAMATH_GPT_amy_total_equals_bob_total_l1734_173444

def original_price : ℝ := 120.00
def sales_tax_rate : ℝ := 0.08
def discount_rate : ℝ := 0.25
def additional_discount : ℝ := 0.10
def num_sweaters : ℕ := 4

def calculate_amy_total (original_price : ℝ) (sales_tax_rate : ℝ) (discount_rate : ℝ) (additional_discount : ℝ) (num_sweaters : ℕ) : ℝ :=
  let price_with_tax := original_price * (1.0 + sales_tax_rate)
  let discounted_price := price_with_tax * (1.0 - discount_rate)
  let final_price := discounted_price * (1.0 - additional_discount)
  final_price * (num_sweaters : ℝ)
  
def calculate_bob_total (original_price : ℝ) (sales_tax_rate : ℝ) (discount_rate : ℝ) (additional_discount : ℝ) (num_sweaters : ℕ) : ℝ :=
  let discounted_price := original_price * (1.0 - discount_rate)
  let further_discounted_price := discounted_price * (1.0 - additional_discount)
  let price_with_tax := further_discounted_price * (1.0 + sales_tax_rate)
  price_with_tax * (num_sweaters : ℝ)

theorem amy_total_equals_bob_total :
  calculate_amy_total original_price sales_tax_rate discount_rate additional_discount num_sweaters =
  calculate_bob_total original_price sales_tax_rate discount_rate additional_discount num_sweaters :=
by
  sorry

end NUMINAMATH_GPT_amy_total_equals_bob_total_l1734_173444


namespace NUMINAMATH_GPT_find_mini_cupcakes_l1734_173438

-- Definitions of the conditions
def number_of_donut_holes := 12
def number_of_students := 13
def desserts_per_student := 2

-- Statement of the theorem to prove the number of mini-cupcakes is 14
theorem find_mini_cupcakes :
  let D := number_of_donut_holes
  let N := number_of_students
  let total_desserts := N * desserts_per_student
  let C := total_desserts - D
  C = 14 :=
by
  sorry

end NUMINAMATH_GPT_find_mini_cupcakes_l1734_173438


namespace NUMINAMATH_GPT_rectangles_fit_l1734_173471

theorem rectangles_fit :
  let width := 50
  let height := 90
  let r_width := 1
  let r_height := (10 * Real.sqrt 2)
  ∃ n : ℕ, 
  n = 315 ∧
  (∃ w_cuts h_cuts : ℕ, 
    w_cuts = Int.floor (width / r_height) ∧
    h_cuts = Int.floor (height / r_height) ∧
    n = ((Int.floor (width / r_height) * Int.floor (height / r_height)) + 
         (Int.floor (height / r_width) * Int.floor (width / r_height)))) := 
sorry

end NUMINAMATH_GPT_rectangles_fit_l1734_173471


namespace NUMINAMATH_GPT_percentage_decrease_l1734_173470

variables (S : ℝ) (D : ℝ)
def initial_increase (S : ℝ) : ℝ := 1.5 * S
def final_gain (S : ℝ) : ℝ := 1.15 * S
def salary_after_decrease (S D : ℝ) : ℝ := (initial_increase S) * (1 - D)

theorem percentage_decrease :
  salary_after_decrease S D = final_gain S → D = 0.233333 :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_l1734_173470


namespace NUMINAMATH_GPT_price_adjustment_l1734_173445

theorem price_adjustment (P : ℝ) (x : ℝ) (hx : P * (1 - (x / 100)^2) = 0.75 * P) : 
  x = 50 :=
by
  -- skipping the proof with sorry
  sorry

end NUMINAMATH_GPT_price_adjustment_l1734_173445


namespace NUMINAMATH_GPT_sum_series_equals_three_fourths_l1734_173424

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end NUMINAMATH_GPT_sum_series_equals_three_fourths_l1734_173424


namespace NUMINAMATH_GPT_raise_3000_yuan_probability_l1734_173475

def prob_correct_1 : ℝ := 0.9
def prob_correct_2 : ℝ := 0.5
def prob_correct_3 : ℝ := 0.4
def prob_incorrect_3 : ℝ := 1 - prob_correct_3

def fund_first : ℝ := 1000
def fund_second : ℝ := 2000
def fund_third : ℝ := 3000

def prob_raise_3000_yuan : ℝ := prob_correct_1 * prob_correct_2 * prob_incorrect_3

theorem raise_3000_yuan_probability :
  prob_raise_3000_yuan = 0.27 :=
by
  sorry

end NUMINAMATH_GPT_raise_3000_yuan_probability_l1734_173475


namespace NUMINAMATH_GPT_min_unit_cubes_l1734_173454

theorem min_unit_cubes (l w h : ℕ) (S : ℕ) (hS : S = 52) 
  (hSurface : 2 * (l * w + l * h + w * h) = S) : 
  ∃ l w h, l * w * h = 16 :=
by
  -- start the proof here
  sorry

end NUMINAMATH_GPT_min_unit_cubes_l1734_173454


namespace NUMINAMATH_GPT_specialSignLanguage_l1734_173479

theorem specialSignLanguage (S : ℕ) 
  (h1 : (S + 2) * (S + 2) = S * S + 1288) : S = 321 := 
by
  sorry

end NUMINAMATH_GPT_specialSignLanguage_l1734_173479


namespace NUMINAMATH_GPT_top_four_cards_probability_l1734_173443

def num_cards : ℕ := 52

def num_hearts : ℕ := 13

def num_diamonds : ℕ := 13

def num_clubs : ℕ := 13

def prob_first_heart := (num_hearts : ℚ) / num_cards
def prob_second_heart := (num_hearts - 1 : ℚ) / (num_cards - 1)
def prob_third_diamond := (num_diamonds : ℚ) / (num_cards - 2)
def prob_fourth_club := (num_clubs : ℚ) / (num_cards - 3)

def combined_prob :=
  prob_first_heart * prob_second_heart * prob_third_diamond * prob_fourth_club

theorem top_four_cards_probability :
  combined_prob = 39 / 63875 := by
  sorry

end NUMINAMATH_GPT_top_four_cards_probability_l1734_173443


namespace NUMINAMATH_GPT_mary_initial_triangles_l1734_173466

theorem mary_initial_triangles (s t : ℕ) (h1 : s + t = 10) (h2 : 4 * s + 3 * t = 36) : t = 4 :=
by
  sorry

end NUMINAMATH_GPT_mary_initial_triangles_l1734_173466


namespace NUMINAMATH_GPT_scientific_notation_of_100000_l1734_173494

theorem scientific_notation_of_100000 :
  100000 = 1 * 10^5 :=
by sorry

end NUMINAMATH_GPT_scientific_notation_of_100000_l1734_173494


namespace NUMINAMATH_GPT_balls_in_boxes_l1734_173434

theorem balls_in_boxes : 
  ∀ (n k : ℕ), n = 6 ∧ k = 3 ∧ ∀ i, i < k → 1 ≤ i → 
             ( ∃ ways : ℕ, ways = Nat.choose ((n - k) + k - 1) (k - 1) ∧ ways = 10 ) :=
by
  sorry

end NUMINAMATH_GPT_balls_in_boxes_l1734_173434


namespace NUMINAMATH_GPT_number_of_bugs_seen_l1734_173463

-- Defining the conditions
def flowers_per_bug : ℕ := 2
def total_flowers_eaten : ℕ := 6

-- The statement to prove
theorem number_of_bugs_seen : total_flowers_eaten / flowers_per_bug = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_bugs_seen_l1734_173463


namespace NUMINAMATH_GPT_speed_of_man_l1734_173446

theorem speed_of_man (v_m v_s : ℝ) 
    (h1 : (v_m + v_s) * 4 = 32) 
    (h2 : (v_m - v_s) * 4 = 24) : v_m = 7 := 
by
  sorry

end NUMINAMATH_GPT_speed_of_man_l1734_173446


namespace NUMINAMATH_GPT_trader_profit_percentage_l1734_173405

-- Definitions for the conditions
def trader_buys_weight (indicated_weight: ℝ) : ℝ :=
  1.10 * indicated_weight

def trader_claimed_weight_to_customer (actual_weight: ℝ) : ℝ :=
  1.30 * actual_weight

-- Main theorem statement
theorem trader_profit_percentage (indicated_weight: ℝ) (actual_weight: ℝ) (claimed_weight: ℝ) :
  trader_buys_weight 1000 = 1100 →
  trader_claimed_weight_to_customer actual_weight = claimed_weight →
  claimed_weight = 1000 →
  (1000 - actual_weight) / actual_weight * 100 = 30 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_trader_profit_percentage_l1734_173405


namespace NUMINAMATH_GPT_change_in_total_berries_l1734_173413

theorem change_in_total_berries (B S : ℕ) (hB : B = 20) (hS : S + B = 50) : (S - B) = 10 := by
  sorry

end NUMINAMATH_GPT_change_in_total_berries_l1734_173413


namespace NUMINAMATH_GPT_journey_time_l1734_173429

variables (d1 d2 : ℝ) (T : ℝ)

theorem journey_time :
  (d1 / 30 + (150 - d1) / 4 = T) ∧
  (d1 / 30 + d2 / 30 + (150 - (d1 + d2)) / 4 = T) ∧
  (d2 / 4 + (150 - (d1 + d2)) / 4 = T) ∧
  (d1 = 3 / 2 * d2) 
  → T = 18 :=
by
  sorry

end NUMINAMATH_GPT_journey_time_l1734_173429


namespace NUMINAMATH_GPT_compute_f_f_f_19_l1734_173451

def f (x : Int) : Int :=
  if x < 10 then x^2 - 9 else x - 15

theorem compute_f_f_f_19 : f (f (f 19)) = 40 := by
  sorry

end NUMINAMATH_GPT_compute_f_f_f_19_l1734_173451


namespace NUMINAMATH_GPT_find_b_l1734_173420

theorem find_b (a b c : ℝ) (h1 : a = 6) (h2 : c = 3) (h3 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) : b = 15 :=
by
  rw [h1, h2] at h3
  sorry

end NUMINAMATH_GPT_find_b_l1734_173420


namespace NUMINAMATH_GPT_triangle_BC_length_l1734_173491

theorem triangle_BC_length (A B C X : Type) (AB AC BC BX CX : ℕ)
  (h1 : AB = 75)
  (h2 : AC = 85)
  (h3 : BC = BX + CX)
  (h4 : BX * (BX + CX) = 1600)
  (h5 : BX + CX = 80) :
  BC = 80 :=
by
  sorry

end NUMINAMATH_GPT_triangle_BC_length_l1734_173491


namespace NUMINAMATH_GPT_irene_overtime_pay_per_hour_l1734_173441

def irene_base_pay : ℝ := 500
def irene_base_hours : ℕ := 40
def irene_total_hours_last_week : ℕ := 50
def irene_total_income_last_week : ℝ := 700

theorem irene_overtime_pay_per_hour :
  (irene_total_income_last_week - irene_base_pay) / (irene_total_hours_last_week - irene_base_hours) = 20 := 
by
  sorry

end NUMINAMATH_GPT_irene_overtime_pay_per_hour_l1734_173441


namespace NUMINAMATH_GPT_rectangle_width_solution_l1734_173430

noncomputable def solve_rectangle_width (W L w l : ℝ) :=
  L = 2 * W ∧ 3 * w = W ∧ 2 * l = L ∧ 6 * l * w = 5400

theorem rectangle_width_solution (W L w l : ℝ) :
  solve_rectangle_width W L w l → w = 10 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_width_solution_l1734_173430


namespace NUMINAMATH_GPT_range_of_a_l1734_173473

noncomputable def f (x : ℝ) := Real.log x / Real.log 2

noncomputable def g (x a : ℝ) := Real.sqrt x + Real.sqrt (a - x)

theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x1 : ℝ, 0 <= x1 ∧ x1 <= a → ∃ x2 : ℝ, 4 ≤ x2 ∧ x2 ≤ 16 ∧ g x1 a = f x2) →
  4 ≤ a ∧ a ≤ 8 :=
sorry 

end NUMINAMATH_GPT_range_of_a_l1734_173473


namespace NUMINAMATH_GPT_min_letters_required_l1734_173460

theorem min_letters_required (n : ℕ) (hn : n = 26) : 
  ∃ k, (∀ (collectors : Fin n) (leader : Fin n), k = 2 * (n - 1)) := 
sorry

end NUMINAMATH_GPT_min_letters_required_l1734_173460


namespace NUMINAMATH_GPT_students_not_invited_count_l1734_173487

-- Define the total number of students
def total_students : ℕ := 30

-- Define the number of students not invited to the event
def not_invited_students : ℕ := 14

-- Define the sets representing different levels of friends of Anna
-- This demonstrates that the total invited students can be derived from given conditions

def anna_immediate_friends : ℕ := 4
def anna_second_level_friends : ℕ := (12 - anna_immediate_friends)
def anna_third_level_friends : ℕ := (16 - 12)

-- Define total invited students
def invited_students : ℕ := 
  anna_immediate_friends + 
  anna_second_level_friends +
  anna_third_level_friends

-- Prove that the number of not invited students is 14
theorem students_not_invited_count : (total_students - invited_students) = not_invited_students :=
by
  sorry

end NUMINAMATH_GPT_students_not_invited_count_l1734_173487


namespace NUMINAMATH_GPT_projection_of_sum_on_vec_a_l1734_173412

open Real

noncomputable def vector_projection (a b : ℝ) (angle : ℝ) : ℝ := 
  (cos angle) * (a * b) / a

theorem projection_of_sum_on_vec_a (a b : EuclideanSpace ℝ (Fin 3)) 
  (h₁ : ‖a‖ = 2) 
  (h₂ : ‖b‖ = 2) 
  (h₃ : inner a b = (2 * 2) * (cos (π / 3))):
  (inner (a + b) a) / ‖a‖ = 3 := 
by
  sorry

end NUMINAMATH_GPT_projection_of_sum_on_vec_a_l1734_173412


namespace NUMINAMATH_GPT_greatest_value_x_plus_y_l1734_173496

theorem greatest_value_x_plus_y (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) : x + y = 6 * Real.sqrt 5 ∨ x + y = -6 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_greatest_value_x_plus_y_l1734_173496


namespace NUMINAMATH_GPT_greening_task_equation_l1734_173410

variable (x : ℝ)

theorem greening_task_equation (h1 : 600000 = 600 * 1000)
    (h2 : ∀ a b : ℝ, a * 1.25 = b -> b = a * (1 + 25 / 100)) :
  (60 * (1 + 25 / 100)) / x - 60 / x = 30 := by
  sorry

end NUMINAMATH_GPT_greening_task_equation_l1734_173410


namespace NUMINAMATH_GPT_men_in_first_group_l1734_173490

theorem men_in_first_group (M : ℕ) : (M * 18 = 27 * 24) → M = 36 :=
by
  sorry

end NUMINAMATH_GPT_men_in_first_group_l1734_173490


namespace NUMINAMATH_GPT_alloy_gold_content_l1734_173406

theorem alloy_gold_content (x : ℝ) (w : ℝ) (p0 p1 : ℝ) (h_w : w = 16)
  (h_p0 : p0 = 0.50) (h_p1 : p1 = 0.80) (h_alloy : x = 24) :
  (p0 * w + x) / (w + x) = p1 :=
by sorry

end NUMINAMATH_GPT_alloy_gold_content_l1734_173406


namespace NUMINAMATH_GPT_correct_equation_l1734_173404

-- Definitions of the conditions
def january_turnover (T : ℝ) : Prop := T = 36
def march_turnover (T : ℝ) : Prop := T = 48
def average_monthly_growth_rate (x : ℝ) : Prop := True

-- The goal to be proved
theorem correct_equation (T_jan T_mar : ℝ) (x : ℝ) 
  (h_jan : january_turnover T_jan) 
  (h_mar : march_turnover T_mar) 
  (h_growth : average_monthly_growth_rate x) : 
  36 * (1 + x)^2 = 48 :=
sorry

end NUMINAMATH_GPT_correct_equation_l1734_173404


namespace NUMINAMATH_GPT_johnny_ways_to_choose_l1734_173476

def num_ways_to_choose_marbles (total_marbles : ℕ) (marbles_to_choose : ℕ) (blue_must_be_included : ℕ) : ℕ :=
  Nat.choose (total_marbles - blue_must_be_included) (marbles_to_choose - blue_must_be_included)

-- Given conditions
def total_marbles : ℕ := 9
def marbles_to_choose : ℕ := 4
def blue_must_be_included : ℕ := 1

-- Theorem to prove the number of ways to choose the marbles
theorem johnny_ways_to_choose :
  num_ways_to_choose_marbles total_marbles marbles_to_choose blue_must_be_included = 56 := by
  sorry

end NUMINAMATH_GPT_johnny_ways_to_choose_l1734_173476


namespace NUMINAMATH_GPT_arithmetic_sequence_property_l1734_173450

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_property
  (h1 : a 6 + a 8 = 10)
  (h2 : a 3 = 1)
  (property : ∀ m n p q : ℕ, m + n = p + q → a m + a n = a p + a q)
  : a 11 = 9 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_property_l1734_173450


namespace NUMINAMATH_GPT_exists_prime_among_15_numbers_l1734_173419

theorem exists_prime_among_15_numbers 
    (integers : Fin 15 → ℕ)
    (h1 : ∀ i, 1 < integers i)
    (h2 : ∀ i, integers i < 1998)
    (h3 : ∀ i j, i ≠ j → Nat.gcd (integers i) (integers j) = 1) :
    ∃ i, Nat.Prime (integers i) :=
by
  sorry

end NUMINAMATH_GPT_exists_prime_among_15_numbers_l1734_173419


namespace NUMINAMATH_GPT_value_of_a_l1734_173401

variable (a : ℤ)
def U : Set ℤ := {2, 4, 3 - a^2}
def P : Set ℤ := {2, a^2 + 2 - a}

theorem value_of_a (h : (U a) \ (P a) = {-1}) : a = 2 :=
sorry

end NUMINAMATH_GPT_value_of_a_l1734_173401


namespace NUMINAMATH_GPT_range_frequency_l1734_173497

-- Define the sample data
def sample_data : List ℝ := [10, 8, 6, 10, 13, 8, 10, 12, 11, 7, 8, 9, 11, 9, 12, 9, 10, 11, 12, 11]

-- Define the condition representing the frequency count
def frequency_count : ℝ := 0.2 * 20

-- Define the proof problem
theorem range_frequency (s : List ℝ) (range_start range_end : ℝ) : 
  s = sample_data → 
  range_start = 11.5 →
  range_end = 13.5 → 
  (s.filter (λ x => range_start ≤ x ∧ x < range_end)).length = frequency_count := 
by 
  intros
  sorry

end NUMINAMATH_GPT_range_frequency_l1734_173497


namespace NUMINAMATH_GPT_prime_implies_n_eq_3k_l1734_173464

theorem prime_implies_n_eq_3k (n : ℕ) (p : ℕ) (k : ℕ) (h_pos : k > 0)
  (h_prime : Prime p) (h_eq : p = 1 + 2^n + 4^n) :
  ∃ k : ℕ, k > 0 ∧ n = 3^k :=
by
  sorry

end NUMINAMATH_GPT_prime_implies_n_eq_3k_l1734_173464


namespace NUMINAMATH_GPT_div_expression_l1734_173411

theorem div_expression : 180 / (12 + 13 * 2) = 90 / 19 := 
  sorry

end NUMINAMATH_GPT_div_expression_l1734_173411


namespace NUMINAMATH_GPT_simplify_product_l1734_173489

theorem simplify_product : (18 : ℚ) * (8 / 12) * (1 / 6) = 2 := by
  sorry

end NUMINAMATH_GPT_simplify_product_l1734_173489


namespace NUMINAMATH_GPT_find_matrix_l1734_173467

theorem find_matrix (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : M^3 - 3 * M^2 + 2 * M = ![![8, 16], ![4, 8]]) : 
  M = ![![2, 4], ![1, 2]] :=
sorry

end NUMINAMATH_GPT_find_matrix_l1734_173467


namespace NUMINAMATH_GPT_find_side_PR_of_PQR_l1734_173461

open Real

noncomputable def triangle_PQR (PQ PM PH PR : ℝ) : Prop :=
  let HQ := sqrt (PQ^2 - PH^2)
  let MH := sqrt (PM^2 - PH^2)
  let MQ := MH - HQ
  let RH := HQ + 2 * MQ
  PR = sqrt (PH^2 + RH^2)

theorem find_side_PR_of_PQR (PQ PM PH : ℝ) (h_PQ : PQ = 3) (h_PM : PM = sqrt 14) (h_PH : PH = sqrt 5) (h_angle : ∀ QPR PRQ : ℝ, QPR + PRQ < 90) : 
  triangle_PQR PQ PM PH (sqrt 21) :=
by
  rw [h_PQ, h_PM, h_PH]
  exact sorry

end NUMINAMATH_GPT_find_side_PR_of_PQR_l1734_173461


namespace NUMINAMATH_GPT_max_value_fraction_l1734_173403

theorem max_value_fraction (e a b : ℝ) (h : ∀ x : ℝ, (e - a) * Real.exp x + x + b + 1 ≤ 0) : 
  (b + 1) / a ≤ 1 / e :=
sorry

end NUMINAMATH_GPT_max_value_fraction_l1734_173403


namespace NUMINAMATH_GPT_solve_table_assignment_l1734_173478

noncomputable def table_assignment (T_1 T_2 T_3 T_4 : Set (Fin 4 × Fin 4)) : Prop :=
  let Albert := T_4
  let Bogdan := T_2
  let Vadim := T_1
  let Denis := T_3
  (∀ x, x ∈ Vadim ↔ x ∉ (Albert ∪ Bogdan)) ∧
  (∀ x, x ∈ Denis ↔ x ∉ (Bogdan ∪ Vadim)) ∧
  Albert = T_4 ∧
  Bogdan = T_2 ∧
  Vadim = T_1 ∧
  Denis = T_3

theorem solve_table_assignment (T_1 T_2 T_3 T_4 : Set (Fin 4 × Fin 4)) :
  table_assignment T_1 T_2 T_3 T_4 :=
sorry

end NUMINAMATH_GPT_solve_table_assignment_l1734_173478


namespace NUMINAMATH_GPT_graph_does_not_pass_first_quadrant_l1734_173440

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - 2

theorem graph_does_not_pass_first_quadrant :
  ¬ ∃ x > 0, f x > 0 := by
sorry

end NUMINAMATH_GPT_graph_does_not_pass_first_quadrant_l1734_173440


namespace NUMINAMATH_GPT_alpha_values_l1734_173472

noncomputable def α := Complex

theorem alpha_values (α : Complex) :
  (α ≠ 1) ∧ 
  (Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1)) ∧ 
  (Complex.abs (α^4 - 1) = 5 * Complex.abs (α - 1)) ∧ 
  (Real.cos α.arg = 1 / 2) →
  α = Complex.mk ((-1 + Real.sqrt 33) / 4) (Real.sqrt (3 * (((-1 + Real.sqrt 33) / 4)^2))) ∨ 
  α = Complex.mk ((-1 - Real.sqrt 33) / 4) (Real.sqrt (3 * (((-1 - Real.sqrt 33) / 4)^2))) :=
sorry

end NUMINAMATH_GPT_alpha_values_l1734_173472


namespace NUMINAMATH_GPT_prob_A_and_B_truth_l1734_173474

-- Define the probabilities
def prob_A_truth := 0.70
def prob_B_truth := 0.60

-- State the theorem
theorem prob_A_and_B_truth : prob_A_truth * prob_B_truth = 0.42 :=
by
  sorry

end NUMINAMATH_GPT_prob_A_and_B_truth_l1734_173474


namespace NUMINAMATH_GPT_tan_alpha_minus_pi_over_4_l1734_173428

variable (α β : ℝ)

-- Given conditions
axiom h1 : Real.tan (α + β) = 2 / 5
axiom h2 : Real.tan β = 1 / 3

-- The goal to prove
theorem tan_alpha_minus_pi_over_4: 
  Real.tan (α - π / 4) = -8 / 9 := by
  sorry

end NUMINAMATH_GPT_tan_alpha_minus_pi_over_4_l1734_173428


namespace NUMINAMATH_GPT_clothing_price_reduction_l1734_173495

def price_reduction (original_profit_per_piece : ℕ) (original_sales_volume : ℕ) (target_profit : ℕ) (increase_in_sales_per_unit_price_reduction : ℕ) : ℕ :=
  sorry

theorem clothing_price_reduction :
  ∃ x : ℕ, (40 - x) * (20 + 2 * x) = 1200 :=
sorry

end NUMINAMATH_GPT_clothing_price_reduction_l1734_173495


namespace NUMINAMATH_GPT_additional_tiles_needed_l1734_173493

theorem additional_tiles_needed (blue_tiles : ℕ) (red_tiles : ℕ) (total_tiles_needed : ℕ)
  (h1 : blue_tiles = 48) (h2 : red_tiles = 32) (h3 : total_tiles_needed = 100) : 
  (total_tiles_needed - (blue_tiles + red_tiles)) = 20 :=
by 
  sorry

end NUMINAMATH_GPT_additional_tiles_needed_l1734_173493


namespace NUMINAMATH_GPT_compute_expression_l1734_173468

theorem compute_expression:
  let a := 3
  let b := 7
  (a + b) ^ 2 + Real.sqrt (a^2 + b^2) = 100 + Real.sqrt 58 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l1734_173468


namespace NUMINAMATH_GPT_remainder_7n_mod_5_l1734_173486

theorem remainder_7n_mod_5 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 5 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_7n_mod_5_l1734_173486


namespace NUMINAMATH_GPT_platform_length_is_260_meters_l1734_173481

noncomputable def train_speed_kmph : ℝ := 72
noncomputable def time_to_cross_platform_s : ℝ := 30
noncomputable def time_to_cross_man_s : ℝ := 17

noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def length_of_train_m : ℝ := train_speed_mps * time_to_cross_man_s
noncomputable def total_distance_cross_platform_m : ℝ := train_speed_mps * time_to_cross_platform_s
noncomputable def length_of_platform_m : ℝ := total_distance_cross_platform_m - length_of_train_m

theorem platform_length_is_260_meters :
  length_of_platform_m = 260 := by
  sorry

end NUMINAMATH_GPT_platform_length_is_260_meters_l1734_173481


namespace NUMINAMATH_GPT_tan_double_angle_third_quadrant_l1734_173485

theorem tan_double_angle_third_quadrant
  (α : ℝ)
  (sin_alpha : Real.sin α = -3/5)
  (h_quadrant : π < α ∧ α < 3 * π / 2) :
  Real.tan (2 * α) = 24 / 7 :=
sorry

end NUMINAMATH_GPT_tan_double_angle_third_quadrant_l1734_173485


namespace NUMINAMATH_GPT_work_completion_days_l1734_173477

theorem work_completion_days (Ry : ℝ) (R_combined : ℝ) (D : ℝ) :
  Ry = 1 / 40 ∧ R_combined = 1 / 13.333333333333332 → 1 / D + Ry = R_combined → D = 20 :=
by
  intros h_eqs h_combined
  sorry

end NUMINAMATH_GPT_work_completion_days_l1734_173477


namespace NUMINAMATH_GPT_twelfth_term_of_arithmetic_sequence_l1734_173457

/-- Condition: a_1 = 1/2 -/
def a1 : ℚ := 1 / 2

/-- Condition: common difference d = 1/3 -/
def d : ℚ := 1 / 3

/-- Prove that the 12th term in the arithmetic sequence is 25/6 given the conditions. -/
theorem twelfth_term_of_arithmetic_sequence : a1 + 11 * d = 25 / 6 := by
  sorry

end NUMINAMATH_GPT_twelfth_term_of_arithmetic_sequence_l1734_173457


namespace NUMINAMATH_GPT_hyperbola_standard_equation_l1734_173469

def a : ℕ := 5
def c : ℕ := 7
def b_squared : ℕ := c * c - a * a

theorem hyperbola_standard_equation (a_eq : a = 5) (c_eq : c = 7) :
    (b_squared = 24) →
    ( ∀ x y : ℝ, x^2 / (a^2 : ℝ) - y^2 / (b_squared : ℝ) = 1 ∨ 
                   y^2 / (a^2 : ℝ) - x^2 / (b_squared : ℝ) = 1) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_standard_equation_l1734_173469


namespace NUMINAMATH_GPT_vasya_is_not_mistaken_l1734_173421

theorem vasya_is_not_mistaken (X Y N A B : ℤ)
  (h_sum : X + Y = N)
  (h_tanya : A * X + B * Y ≡ 0 [ZMOD N]) :
  B * X + A * Y ≡ 0 [ZMOD N] :=
sorry

end NUMINAMATH_GPT_vasya_is_not_mistaken_l1734_173421


namespace NUMINAMATH_GPT_circumscribed_circle_area_l1734_173442

theorem circumscribed_circle_area (side_length : ℝ) (h : side_length = 12) :
  ∃ (A : ℝ), A = 48 * π :=
by
  sorry

end NUMINAMATH_GPT_circumscribed_circle_area_l1734_173442


namespace NUMINAMATH_GPT_chocolate_bars_cost_l1734_173416

variable (n : ℕ) (c : ℕ)

-- Jessica's purchase details
def gummy_bears_packs := 10
def gummy_bears_cost_per_pack := 2
def chocolate_chips_bags := 20
def chocolate_chips_cost_per_bag := 5

-- Calculated costs
def total_gummy_bears_cost := gummy_bears_packs * gummy_bears_cost_per_pack
def total_chocolate_chips_cost := chocolate_chips_bags * chocolate_chips_cost_per_bag

-- Total cost
def total_cost := 150

-- Remaining cost for chocolate bars
def remaining_cost_for_chocolate_bars := total_cost - (total_gummy_bears_cost + total_chocolate_chips_cost)

theorem chocolate_bars_cost (h : remaining_cost_for_chocolate_bars = n * c) : remaining_cost_for_chocolate_bars = 30 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_bars_cost_l1734_173416


namespace NUMINAMATH_GPT_min_c_value_l1734_173432

def y_eq_abs_sum (x a b c : ℝ) : ℝ := |x - a| + |x - b| + |x - c|
def y_eq_line (x : ℝ) : ℝ := -2 * x + 2023

theorem min_c_value (a b c : ℕ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (order : a ≤ b ∧ b < c)
  (unique_sol : ∃! x : ℝ, y_eq_abs_sum x a b c = y_eq_line x) :
  c = 2022 := sorry

end NUMINAMATH_GPT_min_c_value_l1734_173432


namespace NUMINAMATH_GPT_opposite_of_2023_l1734_173414

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end NUMINAMATH_GPT_opposite_of_2023_l1734_173414


namespace NUMINAMATH_GPT_find_divisor_l1734_173448

theorem find_divisor (D : ℕ) : 
  let dividend := 109
  let quotient := 9
  let remainder := 1
  (dividend = D * quotient + remainder) → D = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l1734_173448


namespace NUMINAMATH_GPT_claire_balloons_l1734_173480

def initial_balloons : ℕ := 50
def balloons_lost : ℕ := 12
def balloons_given_away : ℕ := 9
def balloons_received : ℕ := 11

theorem claire_balloons : initial_balloons - balloons_lost - balloons_given_away + balloons_received = 40 :=
by
  sorry

end NUMINAMATH_GPT_claire_balloons_l1734_173480


namespace NUMINAMATH_GPT_find_b_days_l1734_173459

theorem find_b_days 
  (a_days b_days c_days : ℕ)
  (a_wage b_wage c_wage : ℕ)
  (total_earnings : ℕ)
  (ratio_3_4_5 : a_wage * 5 = b_wage * 4 ∧ b_wage * 5 = c_wage * 4 ∧ a_wage * 5 = c_wage * 3)
  (c_wage_val : c_wage = 110)
  (a_days_val : a_days = 6)
  (c_days_val : c_days = 4) 
  (total_earnings_val : total_earnings = 1628)
  (earnings_eq : a_days * a_wage + b_days * b_wage + c_days * c_wage = total_earnings) :
  b_days = 9 := by
  sorry

end NUMINAMATH_GPT_find_b_days_l1734_173459


namespace NUMINAMATH_GPT_police_female_officers_l1734_173482

theorem police_female_officers (perc : ℝ) (total_on_duty: ℝ) (half_on_duty : ℝ) (F : ℝ) :
    perc = 0.18 →
    total_on_duty = 144 →
    half_on_duty = total_on_duty / 2 →
    half_on_duty = perc * F →
    F = 400 :=
by
  sorry

end NUMINAMATH_GPT_police_female_officers_l1734_173482


namespace NUMINAMATH_GPT_watch_cost_l1734_173417

theorem watch_cost (number_of_dimes : ℕ) (value_of_dime : ℝ) (h : number_of_dimes = 50) (hv : value_of_dime = 0.10) :
  number_of_dimes * value_of_dime = 5.00 :=
by
  sorry

end NUMINAMATH_GPT_watch_cost_l1734_173417


namespace NUMINAMATH_GPT_same_solution_for_equations_l1734_173483

theorem same_solution_for_equations (b x : ℝ) :
  (2 * x + 7 = 3) → 
  (b * x - 10 = -2) → 
  b = -4 :=
by
  sorry

end NUMINAMATH_GPT_same_solution_for_equations_l1734_173483


namespace NUMINAMATH_GPT_half_height_of_triangular_prism_l1734_173499

theorem half_height_of_triangular_prism (volume base_area height : ℝ) 
  (h_volume : volume = 576)
  (h_base_area : base_area = 3)
  (h_prism : volume = base_area * height) :
  height / 2 = 96 :=
by
  have h : height = volume / base_area := by sorry
  rw [h_volume, h_base_area] at h
  have h_height : height = 192 := by sorry
  rw [h_height]
  norm_num

end NUMINAMATH_GPT_half_height_of_triangular_prism_l1734_173499


namespace NUMINAMATH_GPT_geometric_sequence_property_l1734_173415

theorem geometric_sequence_property (a : ℕ → ℝ) (q : ℝ)
  (H_geo : ∀ n, a (n + 1) = a n * q)
  (H_cond1 : a 5 * a 7 = 2)
  (H_cond2 : a 2 + a 10 = 3) :
  (a 12 / a 4 = 2) ∨ (a 12 / a 4 = 1/2) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_property_l1734_173415


namespace NUMINAMATH_GPT_range_of_p_l1734_173433

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - 10 * x

-- A = { x | f'(x) ≤ 0 }
def A : Set ℝ := { x | deriv f x ≤ 0 }

-- B = { x | p + 1 ≤ x ≤ 2p - 1 }
def B (p : ℝ) : Set ℝ := { x | p + 1 ≤ x ∧ x ≤ 2 * p - 1 }

-- Given that A ∪ B = A, prove the range of values for p is ≤ 3.
theorem range_of_p (p : ℝ) : (A ∪ B p = A) → p ≤ 3 := sorry

end NUMINAMATH_GPT_range_of_p_l1734_173433
