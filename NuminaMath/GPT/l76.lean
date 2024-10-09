import Mathlib

namespace believe_more_blue_l76_7672

-- Define the conditions
def total_people : ℕ := 150
def more_green : ℕ := 90
def both_more_green_and_more_blue : ℕ := 40
def neither : ℕ := 20

-- Theorem statement: Prove that the number of people who believe teal is "more blue" is 80
theorem believe_more_blue : 
  total_people - neither - (more_green - both_more_green_and_more_blue) = 80 :=
by
  sorry

end believe_more_blue_l76_7672


namespace calculate_expression_l76_7632

theorem calculate_expression : (40 * 1505 - 20 * 1505) / 5 = 6020 := by
  sorry

end calculate_expression_l76_7632


namespace susan_betsy_ratio_l76_7643

theorem susan_betsy_ratio (betsy_wins : ℕ) (helen_wins : ℕ) (susan_wins : ℕ) (total_wins : ℕ)
  (h1 : betsy_wins = 5)
  (h2 : helen_wins = 2 * betsy_wins)
  (h3 : betsy_wins + helen_wins + susan_wins = total_wins)
  (h4 : total_wins = 30) :
  susan_wins / betsy_wins = 3 := by
  sorry

end susan_betsy_ratio_l76_7643


namespace find_b_amount_l76_7629

theorem find_b_amount (A B : ℝ) (h1 : A + B = 100) (h2 : (3 / 10) * A = (1 / 5) * B) : B = 60 := 
by 
  sorry

end find_b_amount_l76_7629


namespace quadrilateral_area_l76_7630

theorem quadrilateral_area :
  let a1 := 9  -- adjacent side length
  let a2 := 6  -- other adjacent side length
  let d := 20  -- diagonal
  let θ1 := 35  -- first angle in degrees
  let θ2 := 110  -- second angle in degrees
  let sin35 := Real.sin (θ1 * Real.pi / 180)
  let sin110 := Real.sin (θ2 * Real.pi / 180)
  let area_triangle1 := (1/2 : ℝ) * a1 * d * sin35
  let area_triangle2 := (1/2 : ℝ) * a2 * d * sin110
  area_triangle1 + area_triangle2 = 108.006 := 
by
  let a1 := 9
  let a2 := 6
  let d := 20
  let θ1 := 35
  let θ2 := 110
  let sin35 := Real.sin (θ1 * Real.pi / 180)
  let sin110 := Real.sin (θ2 * Real.pi / 180)
  let area_triangle1 := (1/2 : ℝ) * a1 * d * sin35
  let area_triangle2 := (1/2 : ℝ) * a2 * d * sin110
  show area_triangle1 + area_triangle2 = 108.006
  sorry

end quadrilateral_area_l76_7630


namespace determine_m_values_l76_7615

theorem determine_m_values (m : ℚ) :
  ((∃ x y : ℚ, x = -3 ∧ y = 0 ∧ (m^2 - 2 * m - 3) * x + (2 * m^2 + m - 1) * y = 2 * m - 6) ∨
  (∃ k : ℚ, k = -1 ∧ (m^2 - 2 * m - 3) + (2 * m^2 + m - 1) * k = 0)) →
  (m = -5/3 ∨ m = 4/3) :=
by
  sorry

end determine_m_values_l76_7615


namespace find_y_l76_7633

theorem find_y (y : ℝ) : 
  2 ≤ y / (3 * y - 4) ∧ y / (3 * y - 4) < 5 ↔ y ∈ Set.Ioc (10 / 7) (8 / 5) := 
sorry

end find_y_l76_7633


namespace distance_covered_at_40_kmph_l76_7662

theorem distance_covered_at_40_kmph
   (total_distance : ℝ)
   (speed1 : ℝ)
   (speed2 : ℝ)
   (total_time : ℝ)
   (part_distance1 : ℝ) :
   total_distance = 250 ∧
   speed1 = 40 ∧
   speed2 = 60 ∧
   total_time = 6 ∧
   (part_distance1 / speed1 + (total_distance - part_distance1) / speed2 = total_time) →
   part_distance1 = 220 :=
by sorry

end distance_covered_at_40_kmph_l76_7662


namespace perimeter_of_inner_polygon_le_outer_polygon_l76_7644

-- Definitions of polygons (for simplicity considered as list of points or sides)
structure Polygon where
  sides : List ℝ  -- assuming sides lengths are given as list of real numbers
  convex : Prop   -- a property stating that the polygon is convex

-- Definition of the perimeter of a polygon
def perimeter (p : Polygon) : ℝ := p.sides.sum

-- Conditions from the problem
variable {P_in P_out : Polygon}
variable (h_convex_in : P_in.convex) (h_convex_out : P_out.convex)
variable (h_inside : ∀ s ∈ P_in.sides, s ∈ P_out.sides) -- simplifying the "inside" condition

-- The theorem statement
theorem perimeter_of_inner_polygon_le_outer_polygon :
  perimeter P_in ≤ perimeter P_out :=
by {
  sorry
}

end perimeter_of_inner_polygon_le_outer_polygon_l76_7644


namespace gcd_18_24_l76_7677

theorem gcd_18_24 : Int.gcd 18 24 = 6 :=
by
  sorry

end gcd_18_24_l76_7677


namespace directrix_of_parabola_l76_7666

theorem directrix_of_parabola (a : ℝ) (P : ℝ × ℝ)
  (h1 : 3 * P.1 ^ 2 - P.2 ^ 2 = 3 * a ^ 2)
  (h2 : P.2 ^ 2 = 8 * a * P.1)
  (h3 : a > 0)
  (h4 : abs ((P.1 - 2 * a) ^ 2 + P.2 ^ 2) ^ (1 / 2) + abs ((P.1 + 2 * a) ^ 2 + P.2 ^ 2) ^ (1 / 2) = 12) :
  (a = 1) → P.1 = 6 - 3 * a → P.2 ^ 2 = 8 * a * (6 - 3 * a) → -2 * a = -2 := 
by
  sorry

end directrix_of_parabola_l76_7666


namespace perpendicular_lines_condition_l76_7663

theorem perpendicular_lines_condition (a : ℝ) :
  (6 * a + 3 * 4 = 0) ↔ (a = -2) :=
sorry

end perpendicular_lines_condition_l76_7663


namespace cubic_box_dimension_l76_7639

theorem cubic_box_dimension (a : ℤ) (h: 12 * a = 3 * (a^3)) : a = 2 :=
by
  sorry

end cubic_box_dimension_l76_7639


namespace tangent_product_l76_7612

-- Declarations for circles, points of tangency, and radii
variables (R r : ℝ) -- radii of the circles
variables (A B C : ℝ) -- distances related to the tangents

-- Conditions: Two circles, a common internal tangent intersecting at points A and B, tangent at point C
axiom tangent_conditions : A * B = R * r

-- Problem statement: Prove that A * C * C * B = R * r
theorem tangent_product (R r A B C : ℝ) (h : A * B = R * r) : A * C * C * B = R * r :=
by
  sorry

end tangent_product_l76_7612


namespace distinct_values_of_fx_l76_7631

theorem distinct_values_of_fx :
  let f (x : ℝ) := ⌊x⌋ + ⌊2 * x⌋ + ⌊3 * x⌋ + ⌊4 * x⌋
  ∃ (s : Finset ℤ), (∀ x, 0 ≤ x ∧ x ≤ 10 → f x ∈ s) ∧ s.card = 61 :=
by
  sorry

end distinct_values_of_fx_l76_7631


namespace jerry_removed_figures_l76_7669

-- Definitions based on conditions
def initialFigures : ℕ := 3
def addedFigures : ℕ := 4
def currentFigures : ℕ := 6

-- Total figures after adding
def totalFigures := initialFigures + addedFigures

-- Proof statement defining how many figures were removed
theorem jerry_removed_figures : (totalFigures - currentFigures) = 1 := by
  sorry

end jerry_removed_figures_l76_7669


namespace discount_percent_l76_7671

theorem discount_percent (CP MP SP : ℝ) (markup profit: ℝ) (h1 : CP = 100) (h2 : MP = CP + (markup * CP))
  (h3 : SP = CP + (profit * CP)) (h4 : markup = 0.75) (h5 : profit = 0.225) : 
  (MP - SP) / MP * 100 = 30 :=
by
  sorry

end discount_percent_l76_7671


namespace geometric_progression_fourth_term_l76_7684

theorem geometric_progression_fourth_term :
  let a1 := 2^(1/2)
  let a2 := 2^(1/4)
  let a3 := 2^(1/8)
  a4 = 2^(1/16) :=
by
  sorry

end geometric_progression_fourth_term_l76_7684


namespace solve_for_x_l76_7641

def f (x : ℝ) : ℝ := x^2 - 5 * x + 6

theorem solve_for_x :
  {x : ℝ | f (f x) = f x} = {0, 2, 3, 5} :=
by 
  sorry

end solve_for_x_l76_7641


namespace parabola_directrix_l76_7623

theorem parabola_directrix (x y : ℝ) (h : y = 4 * (x - 1)^2 + 3) : y = 11 / 4 :=
sorry

end parabola_directrix_l76_7623


namespace find_y_given_conditions_l76_7652

theorem find_y_given_conditions (a x y : ℝ) (h1 : y = a * x + (1 - a)) 
  (x_val : x = 3) (y_val : y = 7) (x_new : x = 8) :
  y = 22 := 
  sorry

end find_y_given_conditions_l76_7652


namespace eval_three_plus_three_cubed_l76_7600

theorem eval_three_plus_three_cubed : 3 + 3^3 = 30 := 
by 
  sorry

end eval_three_plus_three_cubed_l76_7600


namespace necessary_and_sufficient_condition_l76_7624

theorem necessary_and_sufficient_condition (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ (a ≥ 4) :=
by
  sorry

end necessary_and_sufficient_condition_l76_7624


namespace shuffleboard_total_games_l76_7667

theorem shuffleboard_total_games
    (jerry_wins : ℕ)
    (dave_wins : ℕ)
    (ken_wins : ℕ)
    (h1 : jerry_wins = 7)
    (h2 : dave_wins = jerry_wins + 3)
    (h3 : ken_wins = dave_wins + 5) :
    jerry_wins + dave_wins + ken_wins = 32 := 
by
  sorry

end shuffleboard_total_games_l76_7667


namespace largest_4_digit_congruent_to_15_mod_25_l76_7660

theorem largest_4_digit_congruent_to_15_mod_25 : 
  ∀ x : ℕ, (1000 ≤ x ∧ x < 10000 ∧ x % 25 = 15) → x = 9990 :=
by
  intros x h
  sorry

end largest_4_digit_congruent_to_15_mod_25_l76_7660


namespace min_cost_for_boxes_l76_7659

def box_volume (l w h : ℕ) : ℕ := l * w * h
def total_boxes_needed (total_volume box_volume : ℕ) : ℕ := (total_volume + box_volume - 1) / box_volume
def total_cost (num_boxes : ℕ) (cost_per_box : ℚ) : ℚ := num_boxes * cost_per_box

theorem min_cost_for_boxes : 
  let l := 20
  let w := 20
  let h := 15
  let cost_per_box := (7 : ℚ) / 10
  let total_volume := 3060000
  let volume_box := box_volume l w h
  let num_boxes_needed := total_boxes_needed total_volume volume_box
  (num_boxes_needed = 510) → 
  (total_cost num_boxes_needed cost_per_box = 357) :=
by
  intros
  sorry

end min_cost_for_boxes_l76_7659


namespace staff_discount_l76_7605

open Real

theorem staff_discount (d : ℝ) (h : d > 0) (final_price_eq : 0.14 * d = 0.35 * d * (1 - 0.6)) : 0.6 * 100 = 60 :=
by
  sorry

end staff_discount_l76_7605


namespace magic_island_red_parrots_l76_7601

noncomputable def total_parrots : ℕ := 120

noncomputable def green_parrots : ℕ := (5 * total_parrots) / 8

noncomputable def non_green_parrots : ℕ := total_parrots - green_parrots

noncomputable def red_parrots : ℕ := non_green_parrots / 3

theorem magic_island_red_parrots : red_parrots = 15 :=
by
  sorry

end magic_island_red_parrots_l76_7601


namespace financed_amount_correct_l76_7637

-- Define the conditions
def monthly_payment : ℝ := 150.0
def years : ℝ := 5.0
def months_in_a_year : ℝ := 12.0

-- Define the total number of months
def total_months : ℝ := years * months_in_a_year

-- Define the amount financed
def total_financed : ℝ := monthly_payment * total_months

-- State the theorem
theorem financed_amount_correct :
  total_financed = 9000 :=
by
  -- Provide the proof here
  sorry

end financed_amount_correct_l76_7637


namespace repeating_decimal_sum_in_lowest_terms_l76_7651

noncomputable def repeating_decimal_to_fraction (s : String) : ℚ := sorry

theorem repeating_decimal_sum_in_lowest_terms :
  let x := repeating_decimal_to_fraction "0.2"
  let y := repeating_decimal_to_fraction "0.03"
  x + y = 25 / 99 := sorry

end repeating_decimal_sum_in_lowest_terms_l76_7651


namespace fruit_seller_original_apples_l76_7645

theorem fruit_seller_original_apples (x : ℝ) (h : 0.50 * x = 5000) : x = 10000 :=
sorry

end fruit_seller_original_apples_l76_7645


namespace abc_product_l76_7627

theorem abc_product :
  ∃ (a b c P : ℕ), 
    b + c = 3 ∧ 
    c + a = 6 ∧ 
    a + b = 7 ∧ 
    P = a * b * c ∧ 
    P = 10 :=
by sorry

end abc_product_l76_7627


namespace john_total_distance_traveled_l76_7693

theorem john_total_distance_traveled :
  let d1 := 45 * 2.5
  let d2 := 60 * 3.5
  let d3 := 40 * 2
  let d4 := 55 * 3
  d1 + d2 + d3 + d4 = 567.5 := by
  sorry

end john_total_distance_traveled_l76_7693


namespace right_triangle_condition_l76_7699

theorem right_triangle_condition (a b c : ℝ) :
  (a^3 + b^3 + c^3 = a*b*(a + b) - b*c*(b + c) + a*c*(a + c)) ↔ (a^2 = b^2 + c^2) ∨ (b^2 = a^2 + c^2) ∨ (c^2 = a^2 + b^2) :=
by
  sorry

end right_triangle_condition_l76_7699


namespace negation_proposition_l76_7638

theorem negation_proposition :
  (¬ ∀ x : ℝ, x^2 + 2 * x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0) :=
by
  sorry

end negation_proposition_l76_7638


namespace mul_binom_expansion_l76_7686

variable (a : ℝ)

theorem mul_binom_expansion : (a + 1) * (a - 1) = a^2 - 1 :=
by
  sorry

end mul_binom_expansion_l76_7686


namespace base_10_representation_l76_7692

-- Conditions
variables (C D : ℕ)
variables (hC : 0 ≤ C ∧ C ≤ 7)
variables (hD : 0 ≤ D ∧ D ≤ 5)
variables (hEq : 8 * C + D = 6 * D + C)

-- Goal
theorem base_10_representation : 8 * C + D = 0 := by
  sorry

end base_10_representation_l76_7692


namespace ratio_length_to_breadth_l76_7695

theorem ratio_length_to_breadth (b l : ℕ) (A : ℕ) (h1 : b = 30) (h2 : A = 2700) (h3 : A = l * b) :
  l / b = 3 :=
by sorry

end ratio_length_to_breadth_l76_7695


namespace triangle_height_l76_7610

theorem triangle_height (area base height : ℝ) (h1 : area = 500) (h2 : base = 50) (h3 : area = (1 / 2) * base * height) : height = 20 :=
sorry

end triangle_height_l76_7610


namespace sufficient_condition_for_odd_l76_7688

noncomputable def f (a x : ℝ) : ℝ :=
  Real.log (Real.sqrt (x^2 + a^2) - x)

theorem sufficient_condition_for_odd (a : ℝ) :
  (∀ x : ℝ, f 1 (-x) = -f 1 x) ∧
  (∀ x : ℝ, f (-1) (-x) = -f (-1) x) → 
  (a = 1 → ∀ x : ℝ, f a (-x) = -f a x) ∧ 
  (a ≠ 1 → ∃ x : ℝ, f a (-x) ≠ -f a x) :=
by
  sorry

end sufficient_condition_for_odd_l76_7688


namespace two_real_solutions_only_if_c_zero_l76_7682

theorem two_real_solutions_only_if_c_zero (x y c : ℝ) :
  (|x + y| = 99 ∧ |x - y| = c → (∃! (x y : ℝ), |x + y| = 99 ∧ |x - y| = c)) ↔ c = 0 :=
by
  sorry

end two_real_solutions_only_if_c_zero_l76_7682


namespace willie_currency_exchange_l76_7611

theorem willie_currency_exchange :
  let euro_amount := 70
  let pound_amount := 50
  let franc_amount := 30

  let euro_to_dollar := 1.2
  let pound_to_dollar := 1.5
  let franc_to_dollar := 1.1

  let airport_euro_rate := 5 / 7
  let airport_pound_rate := 3 / 4
  let airport_franc_rate := 9 / 10

  let flat_fee := 5

  let official_euro_dollars := euro_amount * euro_to_dollar
  let official_pound_dollars := pound_amount * pound_to_dollar
  let official_franc_dollars := franc_amount * franc_to_dollar

  let airport_euro_dollars := official_euro_dollars * airport_euro_rate
  let airport_pound_dollars := official_pound_dollars * airport_pound_rate
  let airport_franc_dollars := official_franc_dollars * airport_franc_rate

  let final_euro_dollars := airport_euro_dollars - flat_fee
  let final_pound_dollars := airport_pound_dollars - flat_fee
  let final_franc_dollars := airport_franc_dollars - flat_fee

  let total_dollars := final_euro_dollars + final_pound_dollars + final_franc_dollars

  total_dollars = 130.95 :=
by
  sorry

end willie_currency_exchange_l76_7611


namespace college_girls_count_l76_7622

/-- Given conditions:
 1. The ratio of the numbers of boys to girls is 8:5.
 2. The total number of students in the college is 416.
 
 Prove: The number of girls in the college is 160.
 -/
theorem college_girls_count (B G : ℕ) (h1 : B = (8 * G) / 5) (h2 : B + G = 416) : G = 160 :=
by
  sorry

end college_girls_count_l76_7622


namespace mean_problem_l76_7640

theorem mean_problem : 
  (8 + 12 + 24) / 3 = (16 + z) / 2 → z = 40 / 3 :=
by
  intro h
  sorry

end mean_problem_l76_7640


namespace prob_B_independent_l76_7653

-- Definitions based on the problem's conditions
def prob_A := 0.7
def prob_A_union_B := 0.94

-- With these definitions established, we need to state the theorem.
-- The theorem should express that the probability of B solving the problem independently (prob_B) is 0.8.
theorem prob_B_independent : 
    (∃ (prob_B: ℝ), prob_A = 0.7 ∧ prob_A_union_B = 0.94 ∧ prob_B = 0.8) :=
by
    sorry

end prob_B_independent_l76_7653


namespace Malik_yards_per_game_l76_7676

-- Definitions of the conditions
def number_of_games : ℕ := 4
def josiah_yards_per_game : ℕ := 22
def darnell_average_yards_per_game : ℕ := 11
def total_yards_all_athletes : ℕ := 204

-- The statement to prove
theorem Malik_yards_per_game (M : ℕ) 
  (H1 : number_of_games = 4) 
  (H2 : josiah_yards_per_game = 22) 
  (H3 : darnell_average_yards_per_game = 11) 
  (H4 : total_yards_all_athletes = 204) :
  4 * M + 4 * 22 + 4 * 11 = 204 → M = 18 :=
by
  intros h
  sorry

end Malik_yards_per_game_l76_7676


namespace sum_of_decimals_l76_7613

theorem sum_of_decimals : (5.47 + 4.96) = 10.43 :=
by
  sorry

end sum_of_decimals_l76_7613


namespace price_per_postcard_is_correct_l76_7679

noncomputable def initial_postcards : ℕ := 18
noncomputable def sold_postcards : ℕ := initial_postcards / 2
noncomputable def price_per_postcard_sold : ℕ := 15
noncomputable def total_earned : ℕ := sold_postcards * price_per_postcard_sold
noncomputable def total_postcards_after : ℕ := 36
noncomputable def remaining_original_postcards : ℕ := initial_postcards - sold_postcards
noncomputable def new_postcards_bought : ℕ := total_postcards_after - remaining_original_postcards
noncomputable def price_per_new_postcard : ℕ := total_earned / new_postcards_bought

theorem price_per_postcard_is_correct:
  price_per_new_postcard = 5 :=
by
  sorry

end price_per_postcard_is_correct_l76_7679


namespace polygon_number_of_sides_l76_7628

theorem polygon_number_of_sides (P : ℝ) (L : ℝ) (n : ℕ) : P = 180 ∧ L = 15 ∧ n = P / L → n = 12 := by
  sorry

end polygon_number_of_sides_l76_7628


namespace Toms_out_of_pocket_cost_l76_7649

theorem Toms_out_of_pocket_cost (visit_cost cast_cost insurance_percent : ℝ) 
  (h1 : visit_cost = 300) 
  (h2 : cast_cost = 200) 
  (h3 : insurance_percent = 0.6) : 
  (visit_cost + cast_cost) - ((visit_cost + cast_cost) * insurance_percent) = 200 :=
by
  sorry

end Toms_out_of_pocket_cost_l76_7649


namespace abs_sqrt3_minus_1_sub_2_cos30_eq_neg1_l76_7617

theorem abs_sqrt3_minus_1_sub_2_cos30_eq_neg1 :
  |(Real.sqrt 3) - 1| - 2 * Real.cos (Real.pi / 6) = -1 := by
  sorry

end abs_sqrt3_minus_1_sub_2_cos30_eq_neg1_l76_7617


namespace smallest_angle_between_lines_l76_7678

theorem smallest_angle_between_lines (r1 r2 r3 : ℝ) (S U : ℝ) (h1 : r1 = 4) (h2 : r2 = 3) 
  (h3 : r3 = 2) (total_area : ℝ := π * (r1^2 + r2^2 + r3^2)) 
  (h4 : S = (5 / 8) * U) (h5 : S + U = total_area) : 
  ∃ θ : ℝ, θ = (5 * π) / 13 :=
by
  sorry

end smallest_angle_between_lines_l76_7678


namespace solution_set_inequality_l76_7668

theorem solution_set_inequality (x : ℝ) : (x^2 + x - 2 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 1) := 
sorry

end solution_set_inequality_l76_7668


namespace vector_dot_product_result_l76_7614

variable {α : Type*} [Field α]

structure Vector2 (α : Type*) :=
(x : α)
(y : α)

def vector_add (a b : Vector2 α) : Vector2 α :=
  ⟨a.x + b.x, a.y + b.y⟩

def vector_sub (a b : Vector2 α) : Vector2 α :=
  ⟨a.x - b.x, a.y - b.y⟩

def dot_product (a b : Vector2 α) : α :=
  a.x * b.x + a.y * b.y

variable (a b : Vector2 ℝ)

theorem vector_dot_product_result
  (h1 : vector_add a b = ⟨1, -3⟩)
  (h2 : vector_sub a b = ⟨3, 7⟩) :
  dot_product a b = -12 :=
by
  sorry

end vector_dot_product_result_l76_7614


namespace volume_of_cube_l76_7654

theorem volume_of_cube (a : ℕ) (h : a^3 - (a^3 - 4 * a) = 12) : a^3 = 27 :=
by 
  sorry

end volume_of_cube_l76_7654


namespace find_coordinates_of_B_l76_7689

-- Define points A and B, and vector a
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -1, y := 5 }
def a : Point := { x := 2, y := 3 }

-- Define the proof problem
theorem find_coordinates_of_B (B : Point) 
  (h1 : B.x + 1 = 3 * a.x)
  (h2 : B.y - 5 = 3 * a.y) : 
  B = { x := 5, y := 14 } := 
sorry

end find_coordinates_of_B_l76_7689


namespace cashier_overestimation_l76_7681

def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def half_dollar_value := 50

def nickels_counted_as_dimes := 15
def quarters_counted_as_half_dollars := 10

noncomputable def overestimation_due_to_nickels_as_dimes : Nat := 
  (dime_value - nickel_value) * nickels_counted_as_dimes

noncomputable def overestimation_due_to_quarters_as_half_dollars : Nat := 
  (half_dollar_value - quarter_value) * quarters_counted_as_half_dollars

noncomputable def total_overestimation : Nat := 
  overestimation_due_to_nickels_as_dimes + overestimation_due_to_quarters_as_half_dollars

theorem cashier_overestimation : total_overestimation = 325 := by
  sorry

end cashier_overestimation_l76_7681


namespace exists_x_in_interval_iff_m_lt_3_l76_7636

theorem exists_x_in_interval_iff_m_lt_3 (m : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ x^2 - 2 * x > m) ↔ m < 3 :=
by
  sorry

end exists_x_in_interval_iff_m_lt_3_l76_7636


namespace integers_exist_for_eqns_l76_7657

theorem integers_exist_for_eqns (a b c : ℤ) :
  ∃ (p1 q1 r1 p2 q2 r2 : ℤ), 
    a = q1 * r2 - q2 * r1 ∧ 
    b = r1 * p2 - r2 * p1 ∧ 
    c = p1 * q2 - p2 * q1 :=
  sorry

end integers_exist_for_eqns_l76_7657


namespace neighbors_receive_equal_mangoes_l76_7620

-- Definitions from conditions
def total_mangoes : ℕ := 560
def mangoes_sold : ℕ := total_mangoes / 2
def remaining_mangoes : ℕ := total_mangoes - mangoes_sold
def neighbors : ℕ := 8

-- The lean statement
theorem neighbors_receive_equal_mangoes :
  remaining_mangoes / neighbors = 35 :=
by
  -- This is where the proof would go, but we'll leave it with sorry for now.
  sorry

end neighbors_receive_equal_mangoes_l76_7620


namespace erin_walks_less_l76_7618

variable (total_distance : ℕ)
variable (susan_distance : ℕ)

theorem erin_walks_less (h1 : total_distance = 15) (h2 : susan_distance = 9) :
  susan_distance - (total_distance - susan_distance) = 3 := by
  sorry

end erin_walks_less_l76_7618


namespace school_bus_solution_l76_7635

-- Define the capacities
def bus_capacity : Prop := 
  ∃ x y : ℕ, x + y = 75 ∧ 3 * x + 2 * y = 180 ∧ x = 30 ∧ y = 45

-- Define the rental problem
def rental_plans : Prop :=
  ∃ a : ℕ, 6 ≤ a ∧ a ≤ 8 ∧ 
  (30 * a + 45 * (25 - a) ≥ 1000) ∧ 
  (320 * a + 400 * (25 - a) ≤ 9550) ∧ 
  3 = 3

-- The main theorem combines the two aspects
theorem school_bus_solution: bus_capacity ∧ rental_plans := 
  sorry -- Proof omitted

end school_bus_solution_l76_7635


namespace tangent_line_to_C1_and_C2_is_correct_l76_7621

def C1 (x : ℝ) : ℝ := x ^ 2
def C2 (x : ℝ) : ℝ := -(x - 2) ^ 2
def l (x : ℝ) : ℝ := -2 * x + 3

theorem tangent_line_to_C1_and_C2_is_correct :
  (∃ x1 : ℝ, C1 x1 = l x1 ∧ deriv C1 x1 = deriv l x1) ∧
  (∃ x2 : ℝ, C2 x2 = l x2 ∧ deriv C2 x2 = deriv l x2) :=
sorry

end tangent_line_to_C1_and_C2_is_correct_l76_7621


namespace remainder_97_pow_50_mod_100_l76_7650

theorem remainder_97_pow_50_mod_100 :
  (97 ^ 50) % 100 = 49 := 
by
  sorry

end remainder_97_pow_50_mod_100_l76_7650


namespace arithmetic_geometric_sequence_l76_7685

open Real

noncomputable def a_4 (a1 q : ℝ) : ℝ := a1 * q^3
noncomputable def sum_five_terms (a1 q : ℝ) : ℝ := a1 * (1 - q^5) / (1 - q)

theorem arithmetic_geometric_sequence :
  ∀ (a1 q : ℝ),
    (a1 + a1 * q^2 = 10) →
    (a1 * q^3 + a1 * q^5 = 5 / 4) →
    (a_4 a1 q = 1) ∧ (sum_five_terms a1 q = 31 / 2) :=
by
  intros a1 q h1 h2
  sorry

end arithmetic_geometric_sequence_l76_7685


namespace sum_of_cubes_pattern_l76_7625

theorem sum_of_cubes_pattern :
  (1^3 + 2^3 = 3^2) ->
  (1^3 + 2^3 + 3^3 = 6^2) ->
  (1^3 + 2^3 + 3^3 + 4^3 = 10^2) ->
  (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2) :=
by
  intros h1 h2 h3
  -- Proof follows here
  sorry

end sum_of_cubes_pattern_l76_7625


namespace tangent_line_intercept_l76_7674

theorem tangent_line_intercept:
  ∃ (m b : ℚ), 
    m > 0 ∧ 
    b = 135 / 28 ∧ 
    (∀ x y : ℚ, (y - 3)^2 + (x - 1)^2 ≥ 3^2 → (y - 8)^2 + (x - 10)^2 ≥ 6^2 → y = m * x + b) := 
sorry

end tangent_line_intercept_l76_7674


namespace fraction_of_students_l76_7616

theorem fraction_of_students {G B T : ℕ} (h1 : B = 2 * G) (h2 : T = G + B) (h3 : (1 / 2) * (G : ℝ) = (x : ℝ) * (T : ℝ)) : x = (1 / 6) :=
by sorry

end fraction_of_students_l76_7616


namespace sin_double_angle_log_simplification_l76_7697

-- Problem 1: Prove sin(2 * α) = 7 / 25 given sin(α - π / 4) = 3 / 5
theorem sin_double_angle (α : ℝ) (h : Real.sin (α - Real.pi / 4) = 3 / 5) : Real.sin (2 * α) = 7 / 25 :=
by
  sorry

-- Problem 2: Prove 2 * log₅ 10 + log₅ 0.25 = 2
theorem log_simplification : 2 * Real.log 10 / Real.log 5 + Real.log (0.25) / Real.log 5 = 2 :=
by
  sorry

end sin_double_angle_log_simplification_l76_7697


namespace solve_system_eq_l76_7698

theorem solve_system_eq (x y z b : ℝ) :
  (3 * x * y * z - x^3 - y^3 - z^3 = b^3) ∧ 
  (x + y + z = 2 * b) ∧ 
  (x^2 + y^2 - z^2 = b^2) → 
  ( ∃ t : ℝ, (x = (1 + t) * b) ∧ (y = (1 - t) * b) ∧ (z = 0) ∧ t^2 = -1/2 ) :=
by
  -- proof will be filled in here
  sorry

end solve_system_eq_l76_7698


namespace radio_cost_price_l76_7675

theorem radio_cost_price (SP : ℝ) (Loss : ℝ) (CP : ℝ) (h1 : SP = 1110) (h2 : Loss = 0.26) (h3 : SP = CP * (1 - Loss)) : CP = 1500 :=
  by
  sorry

end radio_cost_price_l76_7675


namespace fixed_point_on_line_AC_l76_7646

-- Given definitions and conditions directly from a)
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def line_through_P (x y : ℝ) : Prop := ∃ t : ℝ, x = t * y - 1
def reflection_across_x_axis (y : ℝ) : ℝ := -y

-- The final proof statement translating c)
theorem fixed_point_on_line_AC
  (A B C P : ℝ × ℝ)
  (hP : P = (-1, 0))
  (hA : parabola A.1 A.2)
  (hB : parabola B.1 B.2)
  (hAB : ∃ t : ℝ, line_through_P A.1 A.2 ∧ line_through_P B.1 B.2)
  (hRef : C = (B.1, reflection_across_x_axis B.2)) :
  ∃ x y : ℝ, (x, y) = (1, 0) ∧ line_through_P x y := 
sorry

end fixed_point_on_line_AC_l76_7646


namespace find_x_l76_7658

theorem find_x (A V R S x : ℝ) 
  (h1 : A + x = V - x)
  (h2 : V + 2 * x = A - 2 * x + 30)
  (h3 : (A + R / 2) + (V + R / 2) = 120)
  (h4 : S - 0.25 * S + 10 = 2 * (R / 2)) :
  x = 5 :=
  sorry

end find_x_l76_7658


namespace range_of_m_l76_7687

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 - 4*x ≥ m) → m ≤ -3 :=
by
  intro h
  sorry

end range_of_m_l76_7687


namespace celebration_women_count_l76_7602

theorem celebration_women_count (num_men : ℕ) (num_pairs : ℕ) (pairs_per_man : ℕ) (pairs_per_woman : ℕ) 
  (hm : num_men = 15) (hpm : pairs_per_man = 4) (hwp : pairs_per_woman = 3) (total_pairs : num_pairs = num_men * pairs_per_man) : 
  num_pairs / pairs_per_woman = 20 :=
by
  sorry

end celebration_women_count_l76_7602


namespace find_x_l76_7694

theorem find_x : 
  ∀ x : ℝ, (1 / (x + 4) + 1 / (x - 4) = 1 / (x - 4)) → x = 1 / 2 := 
by 
  sorry

end find_x_l76_7694


namespace find_line_l_l76_7661

theorem find_line_l :
  ∃ l : ℝ × ℝ → Prop,
    (∀ (B : ℝ × ℝ), (2 * B.1 + B.2 - 8 = 0) → 
      (∀ A : ℝ × ℝ, (A.1 = -B.1 ∧ A.2 = 2 * B.1 - 6 ) → 
        (A.1 - 3 * A.2 + 10 = 0) → 
          B.1 = 4 ∧ B.2 = 0 ∧ ∀ p : ℝ × ℝ, B.1 * p.1 + 4 * p.2 - 4 = 0)) := 
  sorry

end find_line_l_l76_7661


namespace scientific_notation_representation_l76_7656

theorem scientific_notation_representation :
  1300000 = 1.3 * 10^6 :=
sorry

end scientific_notation_representation_l76_7656


namespace speed_of_first_train_l76_7673

theorem speed_of_first_train
  (v : ℝ)
  (d : ℝ)
  (distance_between_stations : ℝ := 450)
  (speed_of_second_train : ℝ := 25)
  (additional_distance_first_train : ℝ := 50)
  (meet_time_condition : d / v = (d - additional_distance_first_train) / speed_of_second_train)
  (total_distance_condition : d + (d - additional_distance_first_train) = distance_between_stations) :
  v = 31.25 :=
by {
  sorry
}

end speed_of_first_train_l76_7673


namespace CaitlinAge_l76_7642

theorem CaitlinAge (age_AuntAnna : ℕ) (age_Brianna : ℕ) (age_Caitlin : ℕ)
  (h1 : age_AuntAnna = 42)
  (h2 : age_Brianna = age_AuntAnna / 2)
  (h3 : age_Caitlin = age_Brianna - 5) :
  age_Caitlin = 16 :=
by 
  sorry

end CaitlinAge_l76_7642


namespace rectangle_area_l76_7607

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 :=
by
  sorry

end rectangle_area_l76_7607


namespace johns_payment_ratio_is_one_half_l76_7696

-- Define the initial conditions
def num_members := 4
def join_fee_per_person := 4000
def monthly_cost_per_person := 1000
def johns_payment_per_year := 32000

-- Calculate total cost for joining
def total_join_fee := num_members * join_fee_per_person

-- Calculate total monthly cost for a year
def total_monthly_cost := num_members * monthly_cost_per_person * 12

-- Calculate total cost for the first year
def total_cost_for_year := total_join_fee + total_monthly_cost

-- The ratio of John's payment to the total cost
def johns_ratio := johns_payment_per_year / total_cost_for_year

-- The statement to be proved
theorem johns_payment_ratio_is_one_half : johns_ratio = (1 / 2) := by sorry

end johns_payment_ratio_is_one_half_l76_7696


namespace distance_from_point_A_l76_7608

theorem distance_from_point_A :
  ∀ (A : ℝ) (area : ℝ) (white_area : ℝ) (black_area : ℝ), area = 18 →
  (black_area = 2 * white_area) →
  A = (12 * Real.sqrt 2) / 5 := by
  intros A area white_area black_area h1 h2
  sorry

end distance_from_point_A_l76_7608


namespace second_largest_geometric_sum_l76_7665

theorem second_largest_geometric_sum {a r : ℕ} (h_sum: a + a * r + a * r^2 + a * r^3 = 1417) (h_geometric: 1 + r + r^2 + r^3 ∣ 1417) : (a * r^2 = 272) :=
sorry

end second_largest_geometric_sum_l76_7665


namespace c_value_difference_l76_7609

theorem c_value_difference (a b c : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a^2 + b^2 + c^2 = 18) : 
  max c - min c = 34 / 3 :=
sorry

end c_value_difference_l76_7609


namespace fraction_unspent_is_correct_l76_7604

noncomputable def fraction_unspent (S : ℝ) : ℝ :=
  let after_tax := S - 0.15 * S
  let after_first_week := after_tax - 0.25 * after_tax
  let after_second_week := after_first_week - 0.3 * after_first_week
  let after_third_week := after_second_week - 0.2 * S
  let after_fourth_week := after_third_week - 0.1 * after_third_week
  after_fourth_week / S

theorem fraction_unspent_is_correct (S : ℝ) (hS : S > 0) : 
  fraction_unspent S = 0.221625 :=
by
  sorry

end fraction_unspent_is_correct_l76_7604


namespace dogs_not_eat_either_l76_7606

-- Let's define the conditions
variables (total_dogs : ℕ) (dogs_like_carrots : ℕ) (dogs_like_chicken : ℕ) (dogs_like_both : ℕ)

-- Given conditions
def conditions : Prop :=
  total_dogs = 85 ∧
  dogs_like_carrots = 12 ∧
  dogs_like_chicken = 62 ∧
  dogs_like_both = 8

-- Problem to solve
theorem dogs_not_eat_either (h : conditions total_dogs dogs_like_carrots dogs_like_chicken dogs_like_both) :
  (total_dogs - (dogs_like_carrots - dogs_like_both + dogs_like_chicken - dogs_like_both + dogs_like_both)) = 19 :=
by {
  sorry 
}

end dogs_not_eat_either_l76_7606


namespace integer_product_l76_7683

open Real

theorem integer_product (P Q R S : ℕ) (h1 : P + Q + R + S = 48)
    (h2 : P + 3 = Q - 3) (h3 : P + 3 = R * 3) (h4 : P + 3 = S / 3) :
    P * Q * R * S = 5832 :=
sorry

end integer_product_l76_7683


namespace Karen_baked_50_cookies_l76_7655

def Karen_kept_cookies : ℕ := 10
def Karen_grandparents_cookies : ℕ := 8
def people_in_class : ℕ := 16
def cookies_per_person : ℕ := 2

theorem Karen_baked_50_cookies :
  Karen_kept_cookies + Karen_grandparents_cookies + (people_in_class * cookies_per_person) = 50 :=
by 
  sorry

end Karen_baked_50_cookies_l76_7655


namespace unique_solution_exists_l76_7634

theorem unique_solution_exists (k : ℝ) :
  (16 + 12 * k = 0) → ∃! x : ℝ, k * x^2 - 4 * x - 3 = 0 :=
by
  intro hk
  sorry

end unique_solution_exists_l76_7634


namespace geometric_sequence_ratio_l76_7603

theorem geometric_sequence_ratio (a1 q : ℝ) (h : (a1 * (1 - q^3) / (1 - q)) / (a1 * (1 - q^2) / (1 - q)) = 3 / 2) :
  q = 1 ∨ q = -1 / 2 := by
  sorry

end geometric_sequence_ratio_l76_7603


namespace percentage_loss_l76_7626

theorem percentage_loss (CP SP : ℝ) (hCP : CP = 1400) (hSP : SP = 1148) : 
  (CP - SP) / CP * 100 = 18 := by 
  sorry

end percentage_loss_l76_7626


namespace cost_of_art_book_l76_7691

theorem cost_of_art_book
  (total_cost m_c s_c : ℕ)
  (m_b s_b a_b : ℕ)
  (hm : m_c = 3)
  (hs : s_c = 3)
  (ht : total_cost = 30)
  (hm_books : m_b = 2)
  (hs_books : s_b = 6)
  (ha_books : a_b = 3)
  : ∃ (a_c : ℕ), a_c = 2 := 
by
  sorry

end cost_of_art_book_l76_7691


namespace solve_for_sum_l76_7647

theorem solve_for_sum (x y : ℝ) (h : x^2 + y^2 = 18 * x - 10 * y + 22) : x + y = 4 + 2 * Real.sqrt 42 :=
sorry

end solve_for_sum_l76_7647


namespace number_of_friends_l76_7680

theorem number_of_friends (n : ℕ) (total_bill : ℕ) :
  (total_bill = 12 * (n + 2)) → (total_bill = 16 * n) → n = 6 :=
by
  sorry

end number_of_friends_l76_7680


namespace problem_statement_l76_7619

-- Defining the condition x^3 = 8
def condition1 (x : ℝ) : Prop := x^3 = 8

-- Defining the function f(x) = (x-1)(x+1)(x^2 + x + 1)
def f (x : ℝ) : ℝ := (x - 1) * (x + 1) * (x^2 + x + 1)

-- The theorem we want to prove: For any x satisfying the condition, the function value is 21
theorem problem_statement (x : ℝ) (h : condition1 x) : f x = 21 := 
by
  sorry

end problem_statement_l76_7619


namespace remainder_of_130_div_k_l76_7648

theorem remainder_of_130_div_k (k a : ℕ) (hk : 90 = a * k^2 + 18) : 130 % k = 4 :=
sorry

end remainder_of_130_div_k_l76_7648


namespace community_service_arrangements_l76_7670

noncomputable def total_arrangements : ℕ :=
  let case1 := Nat.choose 6 3
  let case2 := 2 * Nat.choose 6 2
  let case3 := case2
  case1 + case2 + case3

theorem community_service_arrangements :
  total_arrangements = 80 :=
by
  sorry

end community_service_arrangements_l76_7670


namespace proposition_D_l76_7664

variable {A B C : Set α} (h1 : ∀ a (ha : a ∈ A), ∃ b ∈ B, a = b)
variable {A B C : Set α} (h2 : ∀ c (hc : c ∈ C), ∃ b ∈ B, b = c) 

theorem proposition_D (A B C : Set α) (h : A ∩ B = B ∪ C) : C ⊆ B :=
by 
  sorry

end proposition_D_l76_7664


namespace minimum_value_of_f_l76_7690

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem minimum_value_of_f :
  ∃ x : ℝ, (∀ y : ℝ, f x ≤ f y) ∧ f x = -1 / Real.exp 1 :=
by
  -- Proof to be provided
  sorry

end minimum_value_of_f_l76_7690
