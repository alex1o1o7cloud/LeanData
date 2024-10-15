import Mathlib

namespace NUMINAMATH_GPT_volume_conversion_l1673_167300

theorem volume_conversion (V_ft : ℕ) (h_V : V_ft = 216) (conversion_factor : ℕ) (h_cf : conversion_factor = 27) :
  V_ft / conversion_factor = 8 :=
by
  sorry

end NUMINAMATH_GPT_volume_conversion_l1673_167300


namespace NUMINAMATH_GPT_find_line_equation_of_ellipse_intersection_l1673_167304

-- Defining the ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 2 = 1

-- Defining the line intersects points
def line_intersects (A B : ℝ × ℝ) : Prop := 
  ∃ x1 y1 x2 y2 : ℝ, A = (x1, y1) ∧ B = (x2, y2) ∧ 
  (ellipse x1 y1) ∧ (ellipse x2 y2) ∧ 
  ((x1 + x2) / 2 = 1 / 2) ∧ ((y1 + y2) / 2 = -1)

-- Statement to prove the equation of the line
theorem find_line_equation_of_ellipse_intersection (A B : ℝ × ℝ)
  (h : line_intersects A B) : 
  ∃ m b : ℝ, (∀ x y : ℝ, y = m * x + b ↔ x - 4*y - (9/2) = 0) :=
sorry

end NUMINAMATH_GPT_find_line_equation_of_ellipse_intersection_l1673_167304


namespace NUMINAMATH_GPT_whiskers_ratio_l1673_167388

/-- Four cats live in the old grey house at the end of the road. Their names are Puffy, Scruffy, Buffy, and Juniper.
Puffy has three times more whiskers than Juniper, but a certain ratio as many as Scruffy. Buffy has the same number of whiskers
as the average number of whiskers on the three other cats. Prove that the ratio of Puffy's whiskers to Scruffy's whiskers is 1:2
given Juniper has 12 whiskers and Buffy has 40 whiskers. -/
theorem whiskers_ratio (J B P S : ℕ) (hJ : J = 12) (hB : B = 40) (hP : P = 3 * J) (hAvg : B = (P + S + J) / 3) :
  P / gcd P S = 1 ∧ S / gcd P S = 2 := by
  sorry

end NUMINAMATH_GPT_whiskers_ratio_l1673_167388


namespace NUMINAMATH_GPT_find_n_l1673_167361

theorem find_n (a b c : ℕ) (n : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : n > 2) 
    (h₃ : (a^n + b^n + c^n)^2 = 2 * (a^(2*n) + b^(2*n) + c^(2*n))) : n = 4 := 
sorry

end NUMINAMATH_GPT_find_n_l1673_167361


namespace NUMINAMATH_GPT_simplify_expression_l1673_167362

theorem simplify_expression (x : ℝ) : 
  (x^2 + 2 * x + 3) / 4 + (3 * x - 5) / 6 = (3 * x^2 + 12 * x - 1) / 12 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1673_167362


namespace NUMINAMATH_GPT_jack_sugar_l1673_167318

theorem jack_sugar (initial_sugar : ℕ) (sugar_used : ℕ) (sugar_bought : ℕ) (final_sugar : ℕ) 
  (h1 : initial_sugar = 65) (h2 : sugar_used = 18) (h3 : sugar_bought = 50) : 
  final_sugar = initial_sugar - sugar_used + sugar_bought := 
sorry

end NUMINAMATH_GPT_jack_sugar_l1673_167318


namespace NUMINAMATH_GPT_average_sales_per_month_after_discount_is_93_l1673_167379

theorem average_sales_per_month_after_discount_is_93 :
  let salesJanuary := 120
  let salesFebruary := 80
  let salesMarch := 70
  let salesApril := 150
  let salesMayBeforeDiscount := 50
  let discountRate := 0.10
  let discountedSalesMay := salesMayBeforeDiscount - (discountRate * salesMayBeforeDiscount)
  let totalSales := salesJanuary + salesFebruary + salesMarch + salesApril + discountedSalesMay
  let numberOfMonths := 5
  let averageSales := totalSales / numberOfMonths
  averageSales = 93 :=
by {
  -- The actual proof code would go here, but we will skip the proof steps as instructed.
  sorry
}

end NUMINAMATH_GPT_average_sales_per_month_after_discount_is_93_l1673_167379


namespace NUMINAMATH_GPT_prob_of_yellow_second_l1673_167376

-- Defining the probabilities based on the given conditions
def prob_white_from_X : ℚ := 5 / 8
def prob_black_from_X : ℚ := 3 / 8
def prob_yellow_from_Y : ℚ := 8 / 10
def prob_yellow_from_Z : ℚ := 3 / 7

-- Combining probabilities
def combined_prob_white_Y : ℚ := prob_white_from_X * prob_yellow_from_Y
def combined_prob_black_Z : ℚ := prob_black_from_X * prob_yellow_from_Z

-- Total probability of drawing a yellow marble in the second draw
def total_prob_yellow_second : ℚ := combined_prob_white_Y + combined_prob_black_Z

-- Proof statement
theorem prob_of_yellow_second :
  total_prob_yellow_second = 37 / 56 := 
sorry

end NUMINAMATH_GPT_prob_of_yellow_second_l1673_167376


namespace NUMINAMATH_GPT_final_weight_is_correct_l1673_167363

-- Define the initial weight of marble
def initial_weight := 300.0

-- Define the percentage reductions each week
def first_week_reduction := 0.3 * initial_weight
def second_week_reduction := 0.3 * (initial_weight - first_week_reduction)
def third_week_reduction := 0.15 * (initial_weight - first_week_reduction - second_week_reduction)

-- Calculate the final weight of the statue
def final_weight := initial_weight - first_week_reduction - second_week_reduction - third_week_reduction

-- The statement to prove
theorem final_weight_is_correct : final_weight = 124.95 := by
  -- Here would be the proof, which we are omitting
  sorry

end NUMINAMATH_GPT_final_weight_is_correct_l1673_167363


namespace NUMINAMATH_GPT_num_balls_box_l1673_167329

theorem num_balls_box (n : ℕ) (balls : Fin n → ℕ) (red blue : Fin n → Prop)
  (h_colors : ∀ i, red i ∨ blue i)
  (h_constraints : ∀ i j k,  red i ∨ red j ∨ red k ∧ blue i ∨ blue j ∨ blue k) : 
  n = 4 := 
sorry

end NUMINAMATH_GPT_num_balls_box_l1673_167329


namespace NUMINAMATH_GPT_intersection_A_B_l1673_167334

def A := {x : ℝ | x^2 - x - 2 ≤ 0}
def B := {y : ℝ | ∃ x : ℝ, y = 2^x}

theorem intersection_A_B :
  A ∩ {x : ℝ | x > 0} = {x : ℝ | 0 < x ∧ x ≤ 2} :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l1673_167334


namespace NUMINAMATH_GPT_area_of_midpoint_quadrilateral_l1673_167396

theorem area_of_midpoint_quadrilateral (length width : ℝ) (h_length : length = 15) (h_width : width = 8) :
  let A := (0, width / 2)
  let B := (length / 2, 0)
  let C := (length, width / 2)
  let D := (length / 2, width)
  let mid_quad_area := (length / 2) * (width / 2)
  mid_quad_area = 30 :=
by
  simp [h_length, h_width]
  sorry

end NUMINAMATH_GPT_area_of_midpoint_quadrilateral_l1673_167396


namespace NUMINAMATH_GPT_coloring_circle_impossible_l1673_167314

theorem coloring_circle_impossible (n : ℕ) (h : n = 2022) : 
  ¬ (∃ (coloring : ℕ → ℕ), (∀ i, 0 ≤ coloring i ∧ coloring i < 3) ∧ (∀ i, coloring ((i + 1) % n) ≠ coloring i)) :=
sorry

end NUMINAMATH_GPT_coloring_circle_impossible_l1673_167314


namespace NUMINAMATH_GPT_children_attended_play_l1673_167371

variables (A C : ℕ)

theorem children_attended_play
  (h1 : A + C = 610)
  (h2 : 2 * A + C = 960) : 
  C = 260 := 
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_children_attended_play_l1673_167371


namespace NUMINAMATH_GPT_average_marks_l1673_167307

-- Conditions
def marks_english : ℕ := 73
def marks_mathematics : ℕ := 69
def marks_physics : ℕ := 92
def marks_chemistry : ℕ := 64
def marks_biology : ℕ := 82
def number_of_subjects : ℕ := 5

-- Problem Statement
theorem average_marks :
  (marks_english + marks_mathematics + marks_physics + marks_chemistry + marks_biology) / number_of_subjects = 76 :=
by
  sorry

end NUMINAMATH_GPT_average_marks_l1673_167307


namespace NUMINAMATH_GPT_set_representation_l1673_167310

def is_nat_star (n : ℕ) : Prop := n > 0
def satisfies_eqn (x y : ℕ) : Prop := y = 6 / (x + 3)

theorem set_representation :
  {p : ℕ × ℕ | is_nat_star p.fst ∧ is_nat_star p.snd ∧ satisfies_eqn p.fst p.snd } = { (3, 1) } :=
by
  sorry

end NUMINAMATH_GPT_set_representation_l1673_167310


namespace NUMINAMATH_GPT_simplify_expression_correct_l1673_167344

def simplify_expression (i : ℂ) (h : i ^ 2 = -1) : ℂ :=
  3 * (4 - 2 * i) + 2 * i * (3 - i)

theorem simplify_expression_correct (i : ℂ) (h : i ^ 2 = -1) : simplify_expression i h = 14 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_correct_l1673_167344


namespace NUMINAMATH_GPT_if_a_gt_abs_b_then_a2_gt_b2_l1673_167309

theorem if_a_gt_abs_b_then_a2_gt_b2 (a b : ℝ) (h : a > abs b) : a^2 > b^2 :=
by sorry

end NUMINAMATH_GPT_if_a_gt_abs_b_then_a2_gt_b2_l1673_167309


namespace NUMINAMATH_GPT_least_number_of_cookies_l1673_167351

theorem least_number_of_cookies (c : ℕ) :
  (c % 6 = 5) ∧ (c % 8 = 7) ∧ (c % 9 = 6) → c = 23 :=
by
  sorry

end NUMINAMATH_GPT_least_number_of_cookies_l1673_167351


namespace NUMINAMATH_GPT_sum_of_integers_l1673_167390

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 240) : 
  x + y = 32 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_l1673_167390


namespace NUMINAMATH_GPT_initial_men_count_l1673_167366

theorem initial_men_count (M : ℕ) (P : ℝ) 
  (h1 : P = M * 12) 
  (h2 : P = (M + 300) * 9.662337662337663) :
  M = 1240 :=
sorry

end NUMINAMATH_GPT_initial_men_count_l1673_167366


namespace NUMINAMATH_GPT_opposite_of_one_half_l1673_167323

theorem opposite_of_one_half : -((1:ℚ)/2) = -1/2 := by
  -- Skipping the proof using sorry
  sorry

end NUMINAMATH_GPT_opposite_of_one_half_l1673_167323


namespace NUMINAMATH_GPT_midpoint_of_segment_l1673_167349

theorem midpoint_of_segment (A B : (ℤ × ℤ)) (hA : A = (12, 3)) (hB : B = (-8, -5)) :
  (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = -1 :=
by
  sorry

end NUMINAMATH_GPT_midpoint_of_segment_l1673_167349


namespace NUMINAMATH_GPT_complex_exp_sum_l1673_167356

def w : ℂ := sorry  -- We define w as a complex number, satisfying the given condition.

theorem complex_exp_sum (h : w^2 - w + 1 = 0) : 
  w^97 + w^98 + w^99 + w^100 + w^101 + w^102 = -2 + 2 * w :=
by
  sorry

end NUMINAMATH_GPT_complex_exp_sum_l1673_167356


namespace NUMINAMATH_GPT_total_tickets_l1673_167335

theorem total_tickets (R K : ℕ) (hR : R = 12) (h_income : 2 * R + (9 / 2) * K = 60) : R + K = 20 :=
sorry

end NUMINAMATH_GPT_total_tickets_l1673_167335


namespace NUMINAMATH_GPT_frank_total_cans_l1673_167399

def total_cans_picked_up (bags_saturday : ℕ) (bags_sunday : ℕ) (cans_per_bag : ℕ) : ℕ :=
  let total_bags := bags_saturday + bags_sunday
  total_bags * cans_per_bag

theorem frank_total_cans : total_cans_picked_up 5 3 5 = 40 := by
  sorry

end NUMINAMATH_GPT_frank_total_cans_l1673_167399


namespace NUMINAMATH_GPT_area_of_triangle_is_18_l1673_167331

-- Define the vertices of the triangle
def point1 : ℝ × ℝ := (1, 4)
def point2 : ℝ × ℝ := (7, 4)
def point3 : ℝ × ℝ := (1, 10)

-- Define a function to calculate the area of a triangle given three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2)

-- Statement of the problem
theorem area_of_triangle_is_18 :
  triangle_area point1 point2 point3 = 18 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_area_of_triangle_is_18_l1673_167331


namespace NUMINAMATH_GPT_black_circles_count_l1673_167354

theorem black_circles_count (a1 d n : ℕ) (h1 : a1 = 2) (h2 : d = 1) (h3 : n = 16) :
  (n * (a1 + (n - 1) * d) / 2) + n ≤ 160 :=
by
  rw [h1, h2, h3]
  -- Here we will carry out the arithmetic to prove the statement
  sorry

end NUMINAMATH_GPT_black_circles_count_l1673_167354


namespace NUMINAMATH_GPT_math_problem_l1673_167392

noncomputable def find_min_value (a m n : ℝ) (h_a_pos : a > 0) (h_bn : n = 2 * m + 1 / 2)
  (h_b : -a^2 / 2 + 3 * Real.log a = -1 / 2) : ℝ :=
  (3 * Real.sqrt 5 / 5) ^ 2

theorem math_problem (a m n : ℝ) (h_a_pos : a > 0) (h_bn : n = 2 * m + 1 / 2) :
  ∃ b : ℝ, b = -a^2 / 2 + 3 * Real.log a →
  (a - m) ^ 2 + (b - n) ^ 2 = 9 / 5 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1673_167392


namespace NUMINAMATH_GPT_carla_paints_120_square_feet_l1673_167397

def totalWork : ℕ := 360
def ratioAlex : ℕ := 3
def ratioBen : ℕ := 5
def ratioCarla : ℕ := 4
def ratioTotal : ℕ := ratioAlex + ratioBen + ratioCarla
def workPerPart : ℕ := totalWork / ratioTotal
def carlasWork : ℕ := ratioCarla * workPerPart

theorem carla_paints_120_square_feet : carlasWork = 120 := by
  sorry

end NUMINAMATH_GPT_carla_paints_120_square_feet_l1673_167397


namespace NUMINAMATH_GPT_count_valid_Q_l1673_167301

noncomputable def P (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 5)

def Q_degree (Q : Polynomial ℝ) : Prop :=
  Q.degree = 2

def R_degree (R : Polynomial ℝ) : Prop :=
  R.degree = 3

def P_Q_relation (Q R : Polynomial ℝ) : Prop :=
  ∀ x, P (Q.eval x) = P x * R.eval x

theorem count_valid_Q : 
  (∃ Qs : Finset (Polynomial ℝ), ∀ Q ∈ Qs, Q_degree Q ∧ (∃ R, R_degree R ∧ P_Q_relation Q R) 
    ∧ Qs.card = 22) :=
sorry

end NUMINAMATH_GPT_count_valid_Q_l1673_167301


namespace NUMINAMATH_GPT_correct_option_l1673_167327

-- Defining the conditions for each option
def optionA (m n : ℝ) : Prop := (m / n)^7 = m^7 * n^(1/7)
def optionB : Prop := (4)^(4/12) = (-3)^(1/3)
def optionC (x y : ℝ) : Prop := ((x^3 + y^3)^(1/4)) = (x + y)^(3/4)
def optionD : Prop := (9)^(1/6) = 3^(1/3)

-- Asserting that option D is correct
theorem correct_option : optionD :=
by
  sorry

end NUMINAMATH_GPT_correct_option_l1673_167327


namespace NUMINAMATH_GPT_first_to_receive_10_pieces_l1673_167348

-- Definitions and conditions
def children := [1, 2, 3, 4, 5, 6, 7, 8]
def distribution_cycle := [1, 3, 6, 8, 3, 5, 8, 2, 5, 7, 2, 4, 7, 1, 4, 6]

def count_occurrences (n : ℕ) (lst : List ℕ) : ℕ :=
  lst.count n

-- Theorem
theorem first_to_receive_10_pieces : ∃ k, k = 3 ∧ count_occurrences k distribution_cycle = 2 :=
by
  sorry

end NUMINAMATH_GPT_first_to_receive_10_pieces_l1673_167348


namespace NUMINAMATH_GPT_similar_triangle_longest_side_length_l1673_167311

-- Given conditions as definitions 
def originalTriangleSides (a b c : ℕ) : Prop := a = 8 ∧ b = 10 ∧ c = 12
def similarTrianglePerimeter (P : ℕ) : Prop := P = 150

-- Statement to be proved using the given conditions
theorem similar_triangle_longest_side_length (a b c P : ℕ) 
  (h1 : originalTriangleSides a b c) 
  (h2 : similarTrianglePerimeter P) : 
  ∃ x : ℕ, P = (a + b + c) * x ∧ 12 * x = 60 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_similar_triangle_longest_side_length_l1673_167311


namespace NUMINAMATH_GPT_recurring_decimal_to_fraction_l1673_167373

theorem recurring_decimal_to_fraction (a b : ℕ) (ha : a = 356) (hb : b = 999) (hab_gcd : Nat.gcd a b = 1)
  (x : ℚ) (hx : x = 356 / 999) 
  (hx_recurring : x = {num := 356, den := 999}): a + b = 1355 :=
by
  sorry  -- Proof is not required as per the instructions

end NUMINAMATH_GPT_recurring_decimal_to_fraction_l1673_167373


namespace NUMINAMATH_GPT_original_avg_is_40_l1673_167352

noncomputable def original_average (A : ℝ) := (15 : ℝ) * A

noncomputable def new_sum (A : ℝ) := (15 : ℝ) * A + 15 * (15 : ℝ)

theorem original_avg_is_40 (A : ℝ) (h : new_sum A / 15 = 55) :
  A = 40 :=
by sorry

end NUMINAMATH_GPT_original_avg_is_40_l1673_167352


namespace NUMINAMATH_GPT_find_x3_l1673_167395

noncomputable def x3 : ℝ :=
  Real.log ((2 / 3) + (1 / 3) * Real.exp 2)

theorem find_x3 
  (x1 x2 : ℝ)
  (h1 : x1 = 0)
  (h2 : x2 = 2)
  (A : ℝ × ℝ := (x1, Real.exp x1))
  (B : ℝ × ℝ := (x2, Real.exp x2))
  (C : ℝ × ℝ := ((2 * A.1 + B.1) / 3, (2 * A.2 + B.2) / 3))
  (yC : ℝ := (2 / 3) * A.2 + (1 / 3) * B.2)
  (E : ℝ × ℝ := (x3, yC)) :
  E.1 = Real.log ((2 / 3) + (1 / 3) * Real.exp x2) := sorry

end NUMINAMATH_GPT_find_x3_l1673_167395


namespace NUMINAMATH_GPT_total_cost_correct_l1673_167320

def shirt_price : ℕ := 5
def hat_price : ℕ := 4
def jeans_price : ℕ := 10
def jacket_price : ℕ := 20
def shoes_price : ℕ := 15

def num_shirts : ℕ := 4
def num_jeans : ℕ := 3
def num_hats : ℕ := 4
def num_jackets : ℕ := 3
def num_shoes : ℕ := 2

def third_jacket_discount : ℕ := jacket_price / 2
def discount_per_two_shirts : ℕ := 2
def free_hat : ℕ := if num_jeans ≥ 3 then 1 else 0
def shoes_discount : ℕ := (num_shirts / 2) * discount_per_two_shirts

def total_cost : ℕ :=
  (num_shirts * shirt_price) +
  (num_jeans * jeans_price) +
  ((num_hats - free_hat) * hat_price) +
  ((num_jackets - 1) * jacket_price + third_jacket_discount) +
  (num_shoes * shoes_price - shoes_discount)

theorem total_cost_correct : total_cost = 138 := by
  sorry

end NUMINAMATH_GPT_total_cost_correct_l1673_167320


namespace NUMINAMATH_GPT_paint_needed_to_buy_l1673_167380

def total_paint := 333
def existing_paint := 157

theorem paint_needed_to_buy : total_paint - existing_paint = 176 := by
  sorry

end NUMINAMATH_GPT_paint_needed_to_buy_l1673_167380


namespace NUMINAMATH_GPT_find_number_of_3cm_books_l1673_167326

-- Define the conditions
def total_books : ℕ := 46
def total_thickness : ℕ := 200
def thickness_3cm : ℕ := 3
def thickness_5cm : ℕ := 5

-- Let x be the number of 3 cm thick books, y be the number of 5 cm thick books
variable (x y : ℕ)

-- Define the system of equations based on the given conditions
axiom total_books_eq : x + y = total_books
axiom total_thickness_eq : thickness_3cm * x + thickness_5cm * y = total_thickness

-- The theorem to prove: x = 15
theorem find_number_of_3cm_books : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_3cm_books_l1673_167326


namespace NUMINAMATH_GPT_reciprocal_of_fraction_subtraction_l1673_167313

theorem reciprocal_of_fraction_subtraction : (1 / ((2 / 3) - (3 / 4))) = -12 := by
  sorry

end NUMINAMATH_GPT_reciprocal_of_fraction_subtraction_l1673_167313


namespace NUMINAMATH_GPT_johnny_fishes_l1673_167315

theorem johnny_fishes (total_fishes sony_multiple j : ℕ) (h1 : total_fishes = 120) (h2 : sony_multiple = 7) (h3 : total_fishes = j + sony_multiple * j) : j = 15 :=
by sorry

end NUMINAMATH_GPT_johnny_fishes_l1673_167315


namespace NUMINAMATH_GPT_prob_2022_2023_l1673_167360

theorem prob_2022_2023 (n : ℤ) (h : (n - 2022)^2 + (2023 - n)^2 = 1) : (n - 2022) * (2023 - n) = 0 :=
sorry

end NUMINAMATH_GPT_prob_2022_2023_l1673_167360


namespace NUMINAMATH_GPT_b_2023_value_l1673_167389

noncomputable def seq (b : ℕ → ℝ) : Prop := 
  ∀ n ≥ 2, b n = b (n - 1) * b (n + 1)

theorem b_2023_value (b : ℕ → ℝ) (h1 : seq b) (h2 : b 1 = 2 + Real.sqrt 5) (h3 : b 1984 = 12 + Real.sqrt 5) : 
  b 2023 = -4/3 + 10 * Real.sqrt 5 / 3 :=
sorry

end NUMINAMATH_GPT_b_2023_value_l1673_167389


namespace NUMINAMATH_GPT_functional_equation_to_linear_l1673_167372

-- Define that f satisfies the Cauchy functional equation
variable (f : ℕ → ℝ)
axiom cauchy_eq (x y : ℕ) : f (x + y) = f x + f y

-- The theorem we want to prove
theorem functional_equation_to_linear (h : ∀ n k : ℕ, f (n * k) = n * f k) : ∃ a : ℝ, ∀ n : ℕ, f n = a * n :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_to_linear_l1673_167372


namespace NUMINAMATH_GPT_factor_expression_l1673_167302

noncomputable def numerator (a b c : ℝ) : ℝ := 
(|a^2 + b^2|^3 + |b^2 + c^2|^3 + |c^2 + a^2|^3)

noncomputable def denominator (a b c : ℝ) : ℝ := 
(|a + b|^3 + |b + c|^3 + |c + a|^3)

theorem factor_expression (a b c : ℝ) : 
  (denominator a b c) ≠ 0 → 
  (numerator a b c) / (denominator a b c) = 1 :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1673_167302


namespace NUMINAMATH_GPT_letters_per_large_envelope_l1673_167308

theorem letters_per_large_envelope
  (total_letters : ℕ)
  (small_envelope_letters : ℕ)
  (large_envelopes : ℕ)
  (large_envelopes_count : ℕ)
  (h1 : total_letters = 80)
  (h2 : small_envelope_letters = 20)
  (h3 : large_envelopes_count = 30)
  (h4 : total_letters - small_envelope_letters = large_envelopes)
  : large_envelopes / large_envelopes_count = 2 :=
by
  sorry

end NUMINAMATH_GPT_letters_per_large_envelope_l1673_167308


namespace NUMINAMATH_GPT_original_number_is_13_l1673_167387

theorem original_number_is_13 (x : ℝ) (h : 3 * (2 * x + 7) = 99) : x = 13 :=
sorry

end NUMINAMATH_GPT_original_number_is_13_l1673_167387


namespace NUMINAMATH_GPT_power_calculation_l1673_167357

theorem power_calculation :
  ((8^5 / 8^3) * 4^6) = 262144 := by
  sorry

end NUMINAMATH_GPT_power_calculation_l1673_167357


namespace NUMINAMATH_GPT_sum_of_solutions_l1673_167333

theorem sum_of_solutions (x : ℝ) :
  (4 * x + 6) * (3 * x - 12) = 0 → (x = -3 / 2 ∨ x = 4) →
  (-3 / 2 + 4) = 5 / 2 :=
by
  intros Hsol Hsols
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l1673_167333


namespace NUMINAMATH_GPT_prove_m_range_l1673_167394

theorem prove_m_range (m : ℝ) :
  (∀ x : ℝ, (2 * x + 5) / 3 - 1 ≤ 2 - x → 3 * (x - 1) + 5 > 5 * x + 2 * (m + x)) → m < -3 / 5 := by
  sorry

end NUMINAMATH_GPT_prove_m_range_l1673_167394


namespace NUMINAMATH_GPT_percentage_reduction_l1673_167339

theorem percentage_reduction (S P : ℝ) (h : S - (P / 100) * S = S / 2) : P = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_reduction_l1673_167339


namespace NUMINAMATH_GPT_Z_real_Z_imaginary_Z_pure_imaginary_l1673_167303

-- Definitions

def Z (a : ℝ) : ℂ := (a^2 - 9 : ℝ) + (a^2 - 2 * a - 15 : ℂ)

-- Statement for the proof problems

theorem Z_real (a : ℝ) : 
  (Z a).im = 0 ↔ a = 5 ∨ a = -3 := sorry

theorem Z_imaginary (a : ℝ) : 
  (Z a).re = 0 ↔ a ≠ 5 ∧ a ≠ -3 := sorry

theorem Z_pure_imaginary (a : ℝ) : 
  (Z a).re = 0 ∧ (Z a).im ≠ 0 ↔ a = 3 := sorry

end NUMINAMATH_GPT_Z_real_Z_imaginary_Z_pure_imaginary_l1673_167303


namespace NUMINAMATH_GPT_bob_weight_l1673_167316

variable (j b : ℕ)

theorem bob_weight :
  j + b = 210 →
  b - j = b / 3 →
  b = 126 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_bob_weight_l1673_167316


namespace NUMINAMATH_GPT_find_weight_b_l1673_167367

theorem find_weight_b (A B C : ℕ) 
  (h1 : A + B + C = 90)
  (h2 : A + B = 50)
  (h3 : B + C = 56) : 
  B = 16 :=
sorry

end NUMINAMATH_GPT_find_weight_b_l1673_167367


namespace NUMINAMATH_GPT_profit_growth_rate_and_expected_profit_l1673_167328

theorem profit_growth_rate_and_expected_profit
  (profit_April : ℕ)
  (profit_June : ℕ)
  (months : ℕ)
  (avg_growth_rate : ℝ)
  (profit_July : ℕ) :
  profit_April = 6000 ∧ profit_June = 7260 ∧ months = 2 ∧ 
  (profit_April : ℝ) * (1 + avg_growth_rate)^months = profit_June →
  avg_growth_rate = 0.1 ∧ 
  (profit_June : ℝ) * (1 + avg_growth_rate) = profit_July →
  profit_July = 7986 := 
sorry

end NUMINAMATH_GPT_profit_growth_rate_and_expected_profit_l1673_167328


namespace NUMINAMATH_GPT_a_five_minus_a_divisible_by_five_l1673_167330

theorem a_five_minus_a_divisible_by_five (a : ℤ) : 5 ∣ (a^5 - a) :=
by
  -- proof steps
  sorry

end NUMINAMATH_GPT_a_five_minus_a_divisible_by_five_l1673_167330


namespace NUMINAMATH_GPT_set_intersection_complement_l1673_167321

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}
noncomputable def A : Set ℕ := {1, 3}
noncomputable def B : Set ℕ := {2, 3}

theorem set_intersection_complement :
  A ∩ (U \ B) = {1} :=
sorry

end NUMINAMATH_GPT_set_intersection_complement_l1673_167321


namespace NUMINAMATH_GPT_promotional_savings_l1673_167393

noncomputable def y (x : ℝ) : ℝ :=
if x ≤ 500 then x
else if x ≤ 1000 then 500 + 0.8 * (x - 500)
else 500 + 400 + 0.5 * (x - 1000)

theorem promotional_savings (payment : ℝ) (hx : y 2400 = 1600) : 2400 - payment = 800 :=
by sorry

end NUMINAMATH_GPT_promotional_savings_l1673_167393


namespace NUMINAMATH_GPT_power_function_passes_point_l1673_167317

noncomputable def f (k α x : ℝ) : ℝ := k * x^α

theorem power_function_passes_point (k α : ℝ) (h1 : f k α (1/2) = (Real.sqrt 2)/2) : 
  k + α = 3/2 :=
sorry

end NUMINAMATH_GPT_power_function_passes_point_l1673_167317


namespace NUMINAMATH_GPT_total_area_is_8_units_l1673_167365

-- Let s be the side length of the original square and x be the leg length of each isosceles right triangle
variables (s x : ℕ)

-- The side length of the smaller square is 8 units
axiom smaller_square_length : s - 2 * x = 8

-- The area of one isosceles right triangle
def area_triangle : ℕ := x * x / 2

-- There are four triangles
def total_area_triangles : ℕ := 4 * area_triangle x

-- The aim is to prove that the total area of the removed triangles is 8 square units
theorem total_area_is_8_units : total_area_triangles x = 8 :=
sorry

end NUMINAMATH_GPT_total_area_is_8_units_l1673_167365


namespace NUMINAMATH_GPT_multiplier_for_ab_to_equal_1800_l1673_167305

variable (a b m : ℝ)
variable (h1 : 4 * a = 30)
variable (h2 : 5 * b = 30)
variable (h3 : a * b = 45)
variable (h4 : m * (a * b) = 1800)

theorem multiplier_for_ab_to_equal_1800 (h1 : 4 * a = 30) (h2 : 5 * b = 30) (h3 : a * b = 45) (h4 : m * (a * b) = 1800) :
  m = 40 :=
sorry

end NUMINAMATH_GPT_multiplier_for_ab_to_equal_1800_l1673_167305


namespace NUMINAMATH_GPT_problem_KMO_16_l1673_167336

theorem problem_KMO_16
  (m : ℕ) (h_pos : m > 0) :
  (2^(m+1) + 1) ∣ (3^(2^m) + 1) ↔ Nat.Prime (2^(m+1) + 1) :=
by
  sorry

end NUMINAMATH_GPT_problem_KMO_16_l1673_167336


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1673_167375

-- Definitions of the conditions
def is_isosceles (a b : ℕ) : Prop :=
  a = b

def has_side_lengths (a b : ℕ) (c : ℕ) : Prop :=
  true

-- The statement to be proved
theorem isosceles_triangle_perimeter (a b c : ℕ) 
  (h₁ : is_isosceles a b) (h₂ : has_side_lengths a b c) :
  (a + b + c = 16 ∨ a + b + c = 17) :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1673_167375


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1673_167358

variable (x : ℝ)

def p : Prop := (x - 1) / (x + 2) ≥ 0
def q : Prop := (x - 1) * (x + 2) ≥ 0

theorem sufficient_but_not_necessary_condition : (p x) → (q x) ∧ ¬((q x) → (p x)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1673_167358


namespace NUMINAMATH_GPT_range_independent_variable_l1673_167341

def domain_of_function (x : ℝ) : Prop :=
  x ≥ -1 ∧ x ≠ 0

theorem range_independent_variable (x : ℝ) :
  domain_of_function x ↔ x ≥ -1 ∧ x ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_range_independent_variable_l1673_167341


namespace NUMINAMATH_GPT_sumata_family_miles_driven_l1673_167346

def total_miles_driven (days : ℝ) (miles_per_day : ℝ) : ℝ :=
  days * miles_per_day

theorem sumata_family_miles_driven :
  total_miles_driven 5 50 = 250 :=
by
  sorry

end NUMINAMATH_GPT_sumata_family_miles_driven_l1673_167346


namespace NUMINAMATH_GPT_simplify_arithmetic_expr1_simplify_arithmetic_expr2_l1673_167322

-- Problem 1 Statement
theorem simplify_arithmetic_expr1 (x y : ℝ) : 
  (x - 3 * y) - (y - 2 * x) = 3 * x - 4 * y :=
sorry

-- Problem 2 Statement
theorem simplify_arithmetic_expr2 (a b : ℝ) : 
  5 * a * b^2 - 3 * (2 * a^2 * b - 2 * (a^2 * b - 2 * a * b^2)) = -7 * a * b^2 :=
sorry

end NUMINAMATH_GPT_simplify_arithmetic_expr1_simplify_arithmetic_expr2_l1673_167322


namespace NUMINAMATH_GPT_division_rounded_nearest_hundredth_l1673_167343

theorem division_rounded_nearest_hundredth :
  Float.round (285 * 387 / (981^2) * 100) / 100 = 0.11 :=
by
  sorry

end NUMINAMATH_GPT_division_rounded_nearest_hundredth_l1673_167343


namespace NUMINAMATH_GPT_arithmetic_prog_triangle_l1673_167385

theorem arithmetic_prog_triangle (a b c : ℝ) (h : a < b ∧ b < c ∧ 2 * b = a + c)
    (hα : ∀ t, t = a ↔ t = min a (min b c))
    (hγ : ∀ t, t = c ↔ t = max a (max b c)) :
    3 * (Real.tan (α / 2)) * (Real.tan (γ / 2)) = 1 := sorry

end NUMINAMATH_GPT_arithmetic_prog_triangle_l1673_167385


namespace NUMINAMATH_GPT_overall_percentage_favoring_new_tool_l1673_167381

theorem overall_percentage_favoring_new_tool (teachers students : ℕ) 
  (favor_teachers favor_students : ℚ) 
  (surveyed_teachers surveyed_students : ℕ) : 
  surveyed_teachers = 200 → 
  surveyed_students = 800 → 
  favor_teachers = 0.4 → 
  favor_students = 0.75 → 
  ( ( (favor_teachers * surveyed_teachers) + (favor_students * surveyed_students) ) / (surveyed_teachers + surveyed_students) ) * 100 = 68 := 
by 
  sorry

end NUMINAMATH_GPT_overall_percentage_favoring_new_tool_l1673_167381


namespace NUMINAMATH_GPT_abs_neg_sqrt_six_l1673_167386

noncomputable def abs_val (x : ℝ) : ℝ :=
  if x < 0 then -x else x

theorem abs_neg_sqrt_six : abs_val (- Real.sqrt 6) = Real.sqrt 6 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_abs_neg_sqrt_six_l1673_167386


namespace NUMINAMATH_GPT_number_of_polynomials_satisfying_P_neg1_eq_neg12_l1673_167391

noncomputable def count_polynomials_satisfying_condition : ℕ := 
  sorry

theorem number_of_polynomials_satisfying_P_neg1_eq_neg12 :
  count_polynomials_satisfying_condition = 455 := 
  sorry

end NUMINAMATH_GPT_number_of_polynomials_satisfying_P_neg1_eq_neg12_l1673_167391


namespace NUMINAMATH_GPT_min_guests_l1673_167340

/-- Problem statement:
Given:
1. The total food consumed by all guests is 319 pounds.
2. Each guest consumes no more than 1.5 pounds of meat, 0.3 pounds of vegetables, and 0.2 pounds of dessert.
3. Each guest has equal proportions of meat, vegetables, and dessert.

Prove:
The minimum number of guests such that the total food consumed is less than or equal to 319 pounds is 160.
-/
theorem min_guests (total_food : ℝ) (meat_per_guest : ℝ) (veg_per_guest : ℝ) (dessert_per_guest : ℝ) (G : ℕ) :
  total_food = 319 ∧ meat_per_guest ≤ 1.5 ∧ veg_per_guest ≤ 0.3 ∧ dessert_per_guest ≤ 0.2 ∧
  (meat_per_guest + veg_per_guest + dessert_per_guest = 2.0) →
  G = 160 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_min_guests_l1673_167340


namespace NUMINAMATH_GPT_sum_positive_implies_at_least_one_positive_l1673_167347

theorem sum_positive_implies_at_least_one_positive (a b : ℝ) (h : a + b > 0) : a > 0 ∨ b > 0 :=
sorry

end NUMINAMATH_GPT_sum_positive_implies_at_least_one_positive_l1673_167347


namespace NUMINAMATH_GPT_product_of_roots_is_four_thirds_l1673_167377

theorem product_of_roots_is_four_thirds :
  (∀ p q r s : ℚ, (∃ a b c: ℚ, (3 * a^3 - 9 * a^2 + 5 * a - 4 = 0 ∧
                                   3 * b^3 - 9 * b^2 + 5 * b - 4 = 0 ∧
                                   3 * c^3 - 9 * c^2 + 5 * c - 4 = 0)) → 
  - s / p = (4 : ℚ) / 3) := sorry

end NUMINAMATH_GPT_product_of_roots_is_four_thirds_l1673_167377


namespace NUMINAMATH_GPT_range_and_period_range_of_m_l1673_167359

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos (x + Real.pi / 3) * (Real.sin (x + Real.pi / 3) - Real.sqrt 3 * Real.cos (x + Real.pi / 3))

theorem range_and_period (x : ℝ) :
  (Set.range f = Set.Icc (-2 - Real.sqrt 3) (2 - Real.sqrt 3)) ∧ (∀ x, f (x + Real.pi) = f x) := sorry

theorem range_of_m (x m : ℝ) (h1 : 0 ≤ x ∧ x ≤ Real.pi / 6) (h2 : m * (f x + Real.sqrt 3) + 2 = 0) :
  m ∈ Set.Icc (- 2 * Real.sqrt 3 / 3) (-1) := sorry

end NUMINAMATH_GPT_range_and_period_range_of_m_l1673_167359


namespace NUMINAMATH_GPT_number_of_terms_arithmetic_sequence_l1673_167312

theorem number_of_terms_arithmetic_sequence
  (a₁ d n : ℝ)
  (h1 : a₁ + (a₁ + d) + (a₁ + 2 * d) = 34)
  (h2 : (a₁ + (n-3) * d) + (a₁ + (n-2) * d) + (a₁ + (n-1) * d) = 146)
  (h3 : n / 2 * (2 * a₁ + (n-1) * d) = 390) :
  n = 11 :=
by sorry

end NUMINAMATH_GPT_number_of_terms_arithmetic_sequence_l1673_167312


namespace NUMINAMATH_GPT_negation_of_all_exp_monotonic_l1673_167384

theorem negation_of_all_exp_monotonic :
  ¬ (∀ f : ℝ → ℝ, (∀ x y : ℝ, x < y → f x < f y) → (∃ g : ℝ → ℝ, ∃ x y : ℝ, x < y ∧ g x ≥ g y)) :=
sorry

end NUMINAMATH_GPT_negation_of_all_exp_monotonic_l1673_167384


namespace NUMINAMATH_GPT_max_xy_l1673_167374

open Real

theorem max_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 6 * x + 8 * y = 72) (h4 : x = 2 * y) : 
  x * y = 25.92 := 
sorry

end NUMINAMATH_GPT_max_xy_l1673_167374


namespace NUMINAMATH_GPT_area_of_triangle_l1673_167332

theorem area_of_triangle {A B C : ℝ} {a b c : ℝ}
  (h1 : b = 2) (h2 : c = 2 * Real.sqrt 2) (h3 : C = Real.pi / 4) :
  1 / 2 * b * c * Real.sin (Real.pi - C - (1 / 2 * Real.pi / 3)) = Real.sqrt 3 + 1 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l1673_167332


namespace NUMINAMATH_GPT_petya_vasya_equal_again_l1673_167378

theorem petya_vasya_equal_again (n : ℤ) (hn : n ≠ 0) :
  ∃ (k m : ℕ), (∃ P V : ℤ, P = n + 10 * k ∧ V = n - 10 * k ∧ 2014 * P * V = n) :=
sorry

end NUMINAMATH_GPT_petya_vasya_equal_again_l1673_167378


namespace NUMINAMATH_GPT_gcd_153_119_l1673_167355

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  sorry

end NUMINAMATH_GPT_gcd_153_119_l1673_167355


namespace NUMINAMATH_GPT_mary_total_nickels_l1673_167364

-- Definitions for the conditions
def initial_nickels := 7
def dad_nickels := 5
def mom_nickels := 3 * dad_nickels
def chore_nickels := 2

-- The proof problem statement
theorem mary_total_nickels : 
  initial_nickels + dad_nickels + mom_nickels + chore_nickels = 29 := 
by
  sorry

end NUMINAMATH_GPT_mary_total_nickels_l1673_167364


namespace NUMINAMATH_GPT_race_distance_l1673_167342

variable (distance : ℝ)

theorem race_distance :
  (0.25 * distance = 50) → (distance = 200) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_race_distance_l1673_167342


namespace NUMINAMATH_GPT_train_speed_is_60_0131_l1673_167325

noncomputable def train_speed (speed_of_man_kmh : ℝ) (length_of_train_m : ℝ) (time_s : ℝ) : ℝ :=
  let speed_of_man_ms := speed_of_man_kmh * 1000 / 3600
  let relative_speed := length_of_train_m / time_s
  let train_speed_ms := relative_speed - speed_of_man_ms
  train_speed_ms * 3600 / 1000

theorem train_speed_is_60_0131 :
  train_speed 6 330 17.998560115190788 = 60.0131 := by
  sorry

end NUMINAMATH_GPT_train_speed_is_60_0131_l1673_167325


namespace NUMINAMATH_GPT_sequence_converges_and_limit_l1673_167319

theorem sequence_converges_and_limit {a : ℝ} (m : ℕ) (h_a_pos : 0 < a) (h_m_pos : 0 < m) :
  (∃ (x : ℕ → ℝ), 
  (x 1 = 1) ∧ 
  (x 2 = a) ∧ 
  (∀ n : ℕ, x (n + 2) = (x (n + 1) ^ m * x n) ^ (↑(1 : ℕ) / (m + 1))) ∧ 
  ∃ l : ℝ, (∀ ε > 0, ∃ N, ∀ n > N, |x n - l| < ε) ∧ l = a ^ (↑(m + 1) / ↑(m + 2))) :=
sorry

end NUMINAMATH_GPT_sequence_converges_and_limit_l1673_167319


namespace NUMINAMATH_GPT_andrew_bought_mangoes_l1673_167350

theorem andrew_bought_mangoes (m : ℕ) 
    (grapes_cost : 6 * 74 = 444) 
    (mangoes_cost : m * 59 = total_mangoes_cost) 
    (total_cost_eq_975 : 444 + total_mangoes_cost = 975) 
    (total_cost := 444 + total_mangoes_cost) 
    (total_mangoes_cost := 59 * m) 
    : m = 9 := 
sorry

end NUMINAMATH_GPT_andrew_bought_mangoes_l1673_167350


namespace NUMINAMATH_GPT_solution_set_f_x_sq_gt_2f_x_plus_1_l1673_167324

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_f_x_sq_gt_2f_x_plus_1
  (h_domain : ∀ x, 0 < x → ∃ y, f y = f x)
  (h_func_equation : ∀ x y, 0 < x → 0 < y → f (x + y) = f x * f y)
  (h_greater_than_2 : ∀ x, 1 < x → f x > 2)
  (h_f2 : f 2 = 4) :
  ∀ x, x^2 > x + 2 → x > 2 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_solution_set_f_x_sq_gt_2f_x_plus_1_l1673_167324


namespace NUMINAMATH_GPT_carrots_total_l1673_167398

def carrots_grown_by_sally := 6
def carrots_grown_by_fred := 4
def total_carrots := carrots_grown_by_sally + carrots_grown_by_fred

theorem carrots_total : total_carrots = 10 := 
by 
  sorry  -- proof to be filled in

end NUMINAMATH_GPT_carrots_total_l1673_167398


namespace NUMINAMATH_GPT_condition_for_all_real_solutions_l1673_167338

theorem condition_for_all_real_solutions (c : ℝ) :
  (∀ x : ℝ, x^2 + x + c > 0) ↔ c > 1 / 4 :=
sorry

end NUMINAMATH_GPT_condition_for_all_real_solutions_l1673_167338


namespace NUMINAMATH_GPT_cost_price_of_watch_l1673_167337

theorem cost_price_of_watch (C : ℝ) (h1 : ∃ C, 0.91 * C + 220 = 1.04 * C) : C = 1692.31 :=
sorry  -- proof to be provided

end NUMINAMATH_GPT_cost_price_of_watch_l1673_167337


namespace NUMINAMATH_GPT_posts_needed_l1673_167382

-- Define the main properties
def length_of_side_W_stone_wall := 80
def short_side := 50
def intervals (metres: ℕ) := metres / 10 + 1 

-- Define the conditions
def posts_along_w_stone_wall := intervals length_of_side_W_stone_wall
def posts_along_short_sides := 2 * (intervals short_side - 1)

-- Calculate total posts
def total_posts := posts_along_w_stone_wall + posts_along_short_sides

-- Define the theorem
theorem posts_needed : total_posts = 19 := 
by
  sorry

end NUMINAMATH_GPT_posts_needed_l1673_167382


namespace NUMINAMATH_GPT_find_k_l1673_167370

noncomputable def geometric_series_sum (k : ℝ) (h : k > 1) : ℝ :=
  ∑' n, ((7 * n - 2) / k ^ n)

theorem find_k (k : ℝ) (h : k > 1)
  (series_sum : geometric_series_sum k h = 18 / 5) :
  k = 3.42 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1673_167370


namespace NUMINAMATH_GPT_quadratic_roots_is_correct_l1673_167353

theorem quadratic_roots_is_correct (a b : ℝ) 
    (h1 : a + b = 16) 
    (h2 : a * b = 225) :
    (∀ x, x^2 - 16 * x + 225 = 0 ↔ x = a ∨ x = b) := sorry

end NUMINAMATH_GPT_quadratic_roots_is_correct_l1673_167353


namespace NUMINAMATH_GPT_min_value_eq_six_l1673_167369

theorem min_value_eq_six
    (α β : ℝ)
    (k : ℝ)
    (h1 : α^2 + 2 * (k + 3) * α + (k^2 + 3) = 0)
    (h2 : β^2 + 2 * (k + 3) * β + (k^2 + 3) = 0)
    (h3 : (2 * (k + 3))^2 - 4 * (k^2 + 3) ≥ 0) :
    ( (α - 1)^2 + (β - 1)^2 = 6 ) := 
sorry

end NUMINAMATH_GPT_min_value_eq_six_l1673_167369


namespace NUMINAMATH_GPT_sufficient_condition_of_square_inequality_l1673_167383

variables (a b : ℝ)

theorem sufficient_condition_of_square_inequality (ha : a > 0) (hb : b > 0) (h : a > b) : a^2 > b^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_sufficient_condition_of_square_inequality_l1673_167383


namespace NUMINAMATH_GPT_tan_pi_div_4_sub_theta_l1673_167306

theorem tan_pi_div_4_sub_theta (theta : ℝ) (h : Real.tan theta = 1 / 2) : 
  Real.tan (π / 4 - theta) = 1 / 3 := 
sorry

end NUMINAMATH_GPT_tan_pi_div_4_sub_theta_l1673_167306


namespace NUMINAMATH_GPT_fraction_of_students_speak_foreign_language_l1673_167345

noncomputable def students_speak_foreign_language_fraction (M F : ℕ) (h1 : M = F) (m_frac : ℚ) (f_frac : ℚ) : ℚ :=
  ((3 / 5) * M + (2 / 3) * F) / (M + F)

theorem fraction_of_students_speak_foreign_language (M F : ℕ) (h1 : M = F) :
  students_speak_foreign_language_fraction M F h1 (3 / 5) (2 / 3) = 19 / 30 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_of_students_speak_foreign_language_l1673_167345


namespace NUMINAMATH_GPT_find_second_number_l1673_167368

theorem find_second_number (x y z : ℚ) (h₁ : x + y + z = 150) (h₂ : x = (3 / 4) * y) (h₃ : z = (7 / 5) * y) : 
  y = 1000 / 21 :=
by sorry

end NUMINAMATH_GPT_find_second_number_l1673_167368
