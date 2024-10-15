import Mathlib

namespace NUMINAMATH_GPT_total_pages_l1629_162973

def Johnny_word_count : ℕ := 195
def Madeline_word_count : ℕ := 2 * Johnny_word_count
def Timothy_word_count : ℕ := Madeline_word_count + 50
def Samantha_word_count : ℕ := 3 * Madeline_word_count
def Ryan_word_count : ℕ := Johnny_word_count + 100
def Words_per_page : ℕ := 235

def pages_needed (words : ℕ) : ℕ :=
  if words % Words_per_page = 0 then words / Words_per_page else words / Words_per_page + 1

theorem total_pages :
  pages_needed Johnny_word_count +
  pages_needed Madeline_word_count +
  pages_needed Timothy_word_count +
  pages_needed Samantha_word_count +
  pages_needed Ryan_word_count = 12 :=
  by sorry

end NUMINAMATH_GPT_total_pages_l1629_162973


namespace NUMINAMATH_GPT_find_value_of_expression_l1629_162970

-- Conditions translated to Lean 4 definitions
variable (a b : ℝ)
axiom h1 : (a^2 * b^3) / 5 = 1000
axiom h2 : a * b = 2

-- The theorem stating what we need to prove
theorem find_value_of_expression :
  (a^3 * b^2) / 3 = 2 / 705 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l1629_162970


namespace NUMINAMATH_GPT_total_shoes_in_box_l1629_162965

theorem total_shoes_in_box (pairs : ℕ) (prob_matching : ℚ) (h1 : pairs = 7) (h2 : prob_matching = 1 / 13) : 
  ∃ (n : ℕ), n = 2 * pairs ∧ n = 14 :=
by 
  sorry

end NUMINAMATH_GPT_total_shoes_in_box_l1629_162965


namespace NUMINAMATH_GPT_perp_bisector_b_value_l1629_162941

theorem perp_bisector_b_value : ∃ b : ℝ, (∀ (x y : ℝ), x + y = b) ∧ (x + y = b) ∧ (x = (-1) ∧ y = 2) ∧ (x = 3 ∧ y = 8) := sorry

end NUMINAMATH_GPT_perp_bisector_b_value_l1629_162941


namespace NUMINAMATH_GPT_compare_negatives_l1629_162916

theorem compare_negatives : (- (3 : ℝ) / 5) > (- (5 : ℝ) / 7) :=
by
  sorry

end NUMINAMATH_GPT_compare_negatives_l1629_162916


namespace NUMINAMATH_GPT_first_term_of_geometric_series_l1629_162986

theorem first_term_of_geometric_series (r : ℚ) (S : ℚ) (a : ℚ) (h1 : r = 1/5) (h2 : S = 100) (h3 : S = a / (1 - r)) : a = 80 := 
by
  sorry

end NUMINAMATH_GPT_first_term_of_geometric_series_l1629_162986


namespace NUMINAMATH_GPT_value_of_x_abs_not_positive_l1629_162992

theorem value_of_x_abs_not_positive {x : ℝ} : |4 * x - 6| = 0 → x = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_abs_not_positive_l1629_162992


namespace NUMINAMATH_GPT_stratified_sampling_B_l1629_162919

-- Define the groups and their sizes
def num_people_A : ℕ := 18
def num_people_B : ℕ := 24
def num_people_C : ℕ := 30

-- Total number of people
def total_people : ℕ := num_people_A + num_people_B + num_people_C

-- Total sample size to be drawn
def sample_size : ℕ := 12

-- Proportion of group B
def proportion_B : ℚ := num_people_B / total_people

-- Number of people to be drawn from group B
def number_drawn_from_B : ℚ := sample_size * proportion_B

-- The theorem to be proved
theorem stratified_sampling_B : number_drawn_from_B = 4 := 
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_stratified_sampling_B_l1629_162919


namespace NUMINAMATH_GPT_find_even_odd_functions_l1629_162969

variable {X : Type} [AddGroup X]

def even_function (f : X → X) : Prop :=
∀ x, f (-x) = f x

def odd_function (f : X → X) : Prop :=
∀ x, f (-x) = -f x

theorem find_even_odd_functions
  (f g : X → X)
  (h_even : even_function f)
  (h_odd : odd_function g)
  (h_eq : ∀ x, f x + g x = 0) :
  (∀ x, f x = 0) ∧ (∀ x, g x = 0) :=
sorry

end NUMINAMATH_GPT_find_even_odd_functions_l1629_162969


namespace NUMINAMATH_GPT_midpoint_3d_l1629_162990

/-- Midpoint calculation in 3D space -/
theorem midpoint_3d (x1 y1 z1 x2 y2 z2 : ℝ) : 
  (x1, y1, z1) = (2, -3, 6) → 
  (x2, y2, z2) = (8, 5, -4) → 
  ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2) = (5, 1, 1) := 
by
  intros
  sorry

end NUMINAMATH_GPT_midpoint_3d_l1629_162990


namespace NUMINAMATH_GPT_equal_roots_of_quadratic_l1629_162932

theorem equal_roots_of_quadratic (k : ℝ) : (1 - 8 * k = 0) → (k = 1/8) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_equal_roots_of_quadratic_l1629_162932


namespace NUMINAMATH_GPT_correct_algorithm_statement_l1629_162924

def reversible : Prop := false -- Algorithms are generally not reversible.
def endless : Prop := false -- Algorithms should not run endlessly.
def unique_algo : Prop := false -- Not always one single algorithm for a task.
def simple_convenient : Prop := true -- Algorithms should be simple and convenient.

theorem correct_algorithm_statement : simple_convenient = true :=
by
  sorry

end NUMINAMATH_GPT_correct_algorithm_statement_l1629_162924


namespace NUMINAMATH_GPT_pentagon_angle_sum_l1629_162922

theorem pentagon_angle_sum (A B C D Q : ℝ) (hA : A = 118) (hB : B = 105) (hC : C = 87) (hD : D = 135) :
  (A + B + C + D + Q = 540) -> Q = 95 :=
by
  sorry

end NUMINAMATH_GPT_pentagon_angle_sum_l1629_162922


namespace NUMINAMATH_GPT_world_grain_supply_is_correct_l1629_162950

def world_grain_demand : ℝ := 2400000
def supply_ratio : ℝ := 0.75
def world_grain_supply (demand : ℝ) (ratio : ℝ) : ℝ := ratio * demand

theorem world_grain_supply_is_correct :
  world_grain_supply world_grain_demand supply_ratio = 1800000 := 
by 
  sorry

end NUMINAMATH_GPT_world_grain_supply_is_correct_l1629_162950


namespace NUMINAMATH_GPT_problem1_problem2_l1629_162934

-- Proving that (3*sqrt(8) - 12*sqrt(1/2) + sqrt(18)) * 2*sqrt(3) = 6*sqrt(6)
theorem problem1 :
  (3 * Real.sqrt 8 - 12 * Real.sqrt (1/2) + Real.sqrt 18) * 2 * Real.sqrt 3 = 6 * Real.sqrt 6 :=
sorry

-- Proving that (6*sqrt(x/4) - 2*x*sqrt(1/x)) / 3*sqrt(x) = 1/3
theorem problem2 (x : ℝ) (hx : 0 < x) :
  (6 * Real.sqrt (x/4) - 2 * x * Real.sqrt (1/x)) / (3 * Real.sqrt x) = 1/3 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1629_162934


namespace NUMINAMATH_GPT_right_triangle_of_three_colors_exists_l1629_162955

-- Define the type for color
inductive Color
| color1
| color2
| color3

open Color

-- Define the type for integer coordinate points
structure Point :=
(x : ℤ)
(y : ℤ)
(color : Color)

-- Define the conditions
def all_points_colored : Prop :=
∀ (p : Point), p.color = color1 ∨ p.color = color2 ∨ p.color = color3

def all_colors_used : Prop :=
∃ (p1 p2 p3 : Point), p1.color = color1 ∧ p2.color = color2 ∧ p3.color = color3

-- Define the right_triangle_exist problem
def right_triangle_exists : Prop :=
∃ (p1 p2 p3 : Point), 
  p1.color ≠ p2.color ∧ p2.color ≠ p3.color ∧ p3.color ≠ p1.color ∧
  (p1.x = p2.x ∧ p2.y = p3.y ∧ p1.y = p3.y ∨
   p1.y = p2.y ∧ p2.x = p3.x ∧ p1.x = p3.x ∨
   (p3.x - p1.x)*(p3.x - p1.x) + (p3.y - p1.y)*(p3.y - p1.y) = (p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y) ∧
   (p3.x - p2.x)*(p3.x - p2.x) + (p3.y - p2.y)*(p3.y - p2.y) = (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y))

theorem right_triangle_of_three_colors_exists (h1 : all_points_colored) (h2 : all_colors_used) : right_triangle_exists := 
sorry

end NUMINAMATH_GPT_right_triangle_of_three_colors_exists_l1629_162955


namespace NUMINAMATH_GPT_car_price_is_5_l1629_162967

variable (numCars : ℕ) (totalEarnings legoCost carCost : ℕ)

-- Conditions
axiom h1 : numCars = 3
axiom h2 : totalEarnings = 45
axiom h3 : legoCost = 30
axiom h4 : totalEarnings - legoCost = 15
axiom h5 : (totalEarnings - legoCost) / numCars = carCost

-- The proof problem statement
theorem car_price_is_5 : carCost = 5 :=
  by
    -- Here the proof steps would be filled in, but are not required for this task.
    sorry

end NUMINAMATH_GPT_car_price_is_5_l1629_162967


namespace NUMINAMATH_GPT_sasha_hometown_name_l1629_162903

theorem sasha_hometown_name :
  ∃ (sasha_hometown : String), 
  (∃ (vadik_last_column : String), vadik_last_column = "ВКСАМО") →
  (∃ (sasha_transformed : String), sasha_transformed = "мТТЛАРАЕкис") →
  (∃ (sasha_starts_with : Char), sasha_starts_with = 'с') →
  sasha_hometown = "СТЕРЛИТАМАК" :=
by
  sorry

end NUMINAMATH_GPT_sasha_hometown_name_l1629_162903


namespace NUMINAMATH_GPT_fraction_equality_l1629_162975

theorem fraction_equality (a b : ℝ) (h : a / b = 2 / 3) : a / (a + b) = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_equality_l1629_162975


namespace NUMINAMATH_GPT_polynomial_product_roots_l1629_162983

theorem polynomial_product_roots (a b c : ℝ) : 
  (∀ x, (x - (Real.sin (Real.pi / 6))) * (x - (Real.sin (Real.pi / 3))) * (x - (Real.sin (5 * Real.pi / 6))) = x^3 + a * x^2 + b * x + c) → 
  a * b * c = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_product_roots_l1629_162983


namespace NUMINAMATH_GPT_more_soccer_balls_than_basketballs_l1629_162901

theorem more_soccer_balls_than_basketballs :
  let soccer_boxes := 8
  let basketball_boxes := 5
  let balls_per_box := 12
  soccer_boxes * balls_per_box - basketball_boxes * balls_per_box = 36 := by
  sorry

end NUMINAMATH_GPT_more_soccer_balls_than_basketballs_l1629_162901


namespace NUMINAMATH_GPT_tenth_term_l1629_162906

noncomputable def sequence_term (n : ℕ) : ℝ :=
  (-1)^(n+1) * (Real.sqrt (1 + 2*(n - 1))) / (2^n)

theorem tenth_term :
  sequence_term 10 = Real.sqrt 19 / (2^10) :=
by
  sorry

end NUMINAMATH_GPT_tenth_term_l1629_162906


namespace NUMINAMATH_GPT_text_message_costs_equal_l1629_162943

theorem text_message_costs_equal (x : ℝ) : 
  (0.25 * x + 9 = 0.40 * x) ∧ (0.25 * x + 9 = 0.20 * x + 12) → x = 60 :=
by 
  sorry

end NUMINAMATH_GPT_text_message_costs_equal_l1629_162943


namespace NUMINAMATH_GPT_usual_time_is_36_l1629_162915

-- Definition: let S be the usual speed of the worker (not directly relevant to the final proof)
noncomputable def S : ℝ := sorry

-- Definition: let T be the usual time taken by the worker
noncomputable def T : ℝ := sorry

-- Condition: The worker's speed is (3/4) of her normal speed, resulting in a time (T + 12)
axiom speed_delay_condition : (3 / 4) * S * (T + 12) = S * T

-- Theorem: Prove that the usual time T taken to cover the distance is 36 minutes
theorem usual_time_is_36 : T = 36 := by
  -- Formally stating our proof based on given conditions
  sorry

end NUMINAMATH_GPT_usual_time_is_36_l1629_162915


namespace NUMINAMATH_GPT_no_real_x_satisfies_quadratic_ineq_l1629_162911

theorem no_real_x_satisfies_quadratic_ineq :
  ¬ ∃ x : ℝ, x^2 + 3 * x + 3 ≤ 0 :=
sorry

end NUMINAMATH_GPT_no_real_x_satisfies_quadratic_ineq_l1629_162911


namespace NUMINAMATH_GPT_semicircle_radius_correct_l1629_162981

noncomputable def semicircle_radius (P : ℝ) : ℝ := P / (Real.pi + 2)

theorem semicircle_radius_correct (h :127 =113): semicircle_radius 113 = 113 / (Real.pi + 2) :=
by
  sorry

end NUMINAMATH_GPT_semicircle_radius_correct_l1629_162981


namespace NUMINAMATH_GPT_remainder_of_division_l1629_162944

theorem remainder_of_division (L S R : ℕ) (hL : L = 1620) (h_diff : L - S = 1365) (h_div : L = 6 * S + R) : R = 90 :=
by {
  -- Since we are not providing the proof, we use sorry
  sorry
}

end NUMINAMATH_GPT_remainder_of_division_l1629_162944


namespace NUMINAMATH_GPT_profit_margin_A_cost_price_B_units_purchased_l1629_162907

variables (cost_price_A selling_price_A selling_price_B profit_margin_B total_units total_cost : ℕ)
variables (units_A units_B : ℕ)

-- Conditions
def condition1 : cost_price_A = 40 := sorry
def condition2 : selling_price_A = 60 := sorry
def condition3 : selling_price_B = 80 := sorry
def condition4 : profit_margin_B = 60 := sorry
def condition5 : total_units = 50 := sorry
def condition6 : total_cost = 2200 := sorry

-- Proof statements 
theorem profit_margin_A (h1 : cost_price_A = 40) (h2 : selling_price_A = 60) :
  (selling_price_A - cost_price_A) * 100 / cost_price_A = 50 :=
by sorry

theorem cost_price_B (h3 : selling_price_B = 80) (h4 : profit_margin_B = 60) :
  (selling_price_B * 100) / (100 + profit_margin_B) = 50 :=
by sorry

theorem units_purchased (h5 : 40 * units_A + 50 * units_B = 2200)
  (h6 : units_A + units_B = 50) :
  units_A = 30 ∧ units_B = 20 :=
by sorry


end NUMINAMATH_GPT_profit_margin_A_cost_price_B_units_purchased_l1629_162907


namespace NUMINAMATH_GPT_total_words_read_l1629_162997

/-- Proof Problem Statement:
  Given the following conditions:
  - Henri has 8 hours to watch movies and read.
  - He watches one movie for 3.5 hours.
  - He watches another movie for 1.5 hours.
  - He watches two more movies with durations of 1.25 hours and 0.75 hours, respectively.
  - He reads for the remaining time after watching movies.
  - For the first 30 minutes of reading, he reads at a speed of 12 words per minute.
  - For the following 20 minutes, his reading speed decreases to 8 words per minute.
  - In the last remaining minutes, his reading speed increases to 15 words per minute.
  Prove that the total number of words Henri reads during his free time is 670.
--/
theorem total_words_read : 8 * 60 - (7 * 60) = 60 ∧
  (30 * 12) + (20 * 8) + ((60 - 30 - 20) * 15) = 670 :=
by
  sorry

end NUMINAMATH_GPT_total_words_read_l1629_162997


namespace NUMINAMATH_GPT_find_b_l1629_162960

theorem find_b (a b : ℤ) 
  (h1 : a * b = 2 * (a + b) + 14) 
  (h2 : b - a = 3) : 
  b = 8 :=
sorry

end NUMINAMATH_GPT_find_b_l1629_162960


namespace NUMINAMATH_GPT_flooring_cost_correct_l1629_162962

noncomputable def cost_of_flooring (l w h_t b_t c : ℝ) : ℝ :=
  let area_rectangle := l * w
  let area_triangle := (b_t * h_t) / 2
  let area_to_be_floored := area_rectangle - area_triangle
  area_to_be_floored * c

theorem flooring_cost_correct :
  cost_of_flooring 10 7 3 4 900 = 57600 :=
by
  sorry

end NUMINAMATH_GPT_flooring_cost_correct_l1629_162962


namespace NUMINAMATH_GPT_completing_square_l1629_162963

-- Define the theorem statement
theorem completing_square (x : ℝ) : 
  x^2 - 2 * x = 2 -> (x - 1)^2 = 3 :=
by sorry

end NUMINAMATH_GPT_completing_square_l1629_162963


namespace NUMINAMATH_GPT_tan_theta_parallel_l1629_162926

theorem tan_theta_parallel (θ : ℝ) : 
  let a := (2, 3)
  let b := (Real.cos θ, Real.sin θ)
  (b.1 * a.2 = b.2 * a.1) → Real.tan θ = 3 / 2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_tan_theta_parallel_l1629_162926


namespace NUMINAMATH_GPT_sandy_total_sums_l1629_162957

theorem sandy_total_sums (C I : ℕ) (h1 : C = 22) (h2 : 3 * C - 2 * I = 50) :
  C + I = 30 :=
sorry

end NUMINAMATH_GPT_sandy_total_sums_l1629_162957


namespace NUMINAMATH_GPT_move_point_right_3_units_from_neg_2_l1629_162936

noncomputable def move_point_to_right (start : ℤ) (units : ℤ) : ℤ :=
start + units

theorem move_point_right_3_units_from_neg_2 : move_point_to_right (-2) 3 = 1 :=
by
  sorry

end NUMINAMATH_GPT_move_point_right_3_units_from_neg_2_l1629_162936


namespace NUMINAMATH_GPT_log_x_inequality_l1629_162984

noncomputable def log_x_over_x (x : ℝ) := (Real.log x) / x

theorem log_x_inequality {x : ℝ} (h1 : 1 < x) (h2 : x < 2) : 
  (log_x_over_x x) ^ 2 < log_x_over_x x ∧ log_x_over_x x < log_x_over_x (x * x) :=
by
  sorry

end NUMINAMATH_GPT_log_x_inequality_l1629_162984


namespace NUMINAMATH_GPT_solve_equation_l1629_162928

theorem solve_equation (x : ℚ) (h : x ≠ -5) : 
  (x^2 + 3*x + 4) / (x + 5) = x + 6 ↔ x = -13 / 4 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l1629_162928


namespace NUMINAMATH_GPT_total_boxes_l1629_162964

theorem total_boxes (r_cost y_cost : ℝ) (avg_cost : ℝ) (R Y : ℕ) (hc_r : r_cost = 1.30) (hc_y : y_cost = 2.00) 
                    (hc_avg : avg_cost = 1.72) (hc_R : R = 4) (hc_Y : Y = 4) : 
  R + Y = 8 :=
by
  sorry

end NUMINAMATH_GPT_total_boxes_l1629_162964


namespace NUMINAMATH_GPT_dogsled_course_distance_l1629_162953

theorem dogsled_course_distance 
    (t : ℕ)  -- time taken by Team B
    (speed_B : ℕ := 20)  -- average speed of Team B
    (speed_A : ℕ := 25)  -- average speed of Team A
    (tA_eq_tB_minus_3 : t - 3 = tA)  -- Team A’s time relation
    (speedA_eq_speedB_plus_5 : speed_A = speed_B + 5)  -- Team A's average speed in relation to Team B’s average speed
    (distance_eq : speed_B * t = speed_A * (t - 3))  -- Distance equality condition
    (t_eq_15 : t = 15)  -- Time taken by Team B to finish
    :
    (speed_B * t = 300) :=   -- Distance of the course
by
  sorry

end NUMINAMATH_GPT_dogsled_course_distance_l1629_162953


namespace NUMINAMATH_GPT_sum_minimal_area_k_l1629_162972

def vertices_triangle_min_area (k : ℤ) : Prop :=
  let x1 := 1
  let y1 := 7
  let x2 := 13
  let y2 := 16
  let x3 := 5
  ((y1 - k) * (x2 - x1) ≠ (x1 - x3) * (y2 - y1))

def minimal_area_sum_k : ℤ :=
  9 + 11

theorem sum_minimal_area_k :
  ∃ k1 k2 : ℤ, vertices_triangle_min_area k1 ∧ vertices_triangle_min_area k2 ∧ k1 + k2 = 20 := 
sorry

end NUMINAMATH_GPT_sum_minimal_area_k_l1629_162972


namespace NUMINAMATH_GPT_ratio_of_areas_l1629_162918

-- Define the side lengths of Squared B and Square C
variables (y : ℝ)

-- Define the areas of Square B and C
def area_B := (2 * y) * (2 * y)
def area_C := (8 * y) * (8 * y)

-- The theorem statement proving the ratio of the areas
theorem ratio_of_areas : area_B y / area_C y = 1 / 16 := 
by sorry

end NUMINAMATH_GPT_ratio_of_areas_l1629_162918


namespace NUMINAMATH_GPT_does_not_round_to_72_56_l1629_162999

-- Definitions for the numbers in question
def numA := 72.558
def numB := 72.563
def numC := 72.55999
def numD := 72.564
def numE := 72.555

-- Function to round a number to the nearest hundredth
def round_nearest_hundredth (x : Float) : Float :=
  (Float.round (x * 100) / 100 : Float)

-- Lean statement for the equivalent proof problem
theorem does_not_round_to_72_56 :
  round_nearest_hundredth numA = 72.56 ∧
  round_nearest_hundredth numB = 72.56 ∧
  round_nearest_hundredth numC = 72.56 ∧
  round_nearest_hundredth numD = 72.56 ∧
  round_nearest_hundredth numE ≠ 72.56 :=
by
  sorry

end NUMINAMATH_GPT_does_not_round_to_72_56_l1629_162999


namespace NUMINAMATH_GPT_root_expr_calculation_l1629_162991

theorem root_expr_calculation : (3 + Real.sqrt 10) * (Real.sqrt 2 - Real.sqrt 5) = -2 * Real.sqrt 2 - Real.sqrt 5 := 
by 
  sorry

end NUMINAMATH_GPT_root_expr_calculation_l1629_162991


namespace NUMINAMATH_GPT_crayons_count_l1629_162940

theorem crayons_count (l b f : ℕ) (h1 : l = b / 2) (h2 : b = 3 * f) (h3 : l = 27) : f = 18 :=
by
  sorry

end NUMINAMATH_GPT_crayons_count_l1629_162940


namespace NUMINAMATH_GPT_exists_divisible_by_2011_l1629_162958

def a (n : ℕ) : ℕ := (List.range n).foldl (λ acc i => acc + 10 ^ i) 0

theorem exists_divisible_by_2011 : ∃ n, 1 ≤ n ∧ n ≤ 2011 ∧ 2011 ∣ a n := by
  sorry

end NUMINAMATH_GPT_exists_divisible_by_2011_l1629_162958


namespace NUMINAMATH_GPT_find_f_2_find_f_neg2_l1629_162927

noncomputable def f : ℝ → ℝ := sorry -- This is left to be defined as a function on ℝ

axiom f_property : ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y
axiom f_at_1 : f 1 = 2

theorem find_f_2 : f 2 = 6 := by
  sorry

theorem find_f_neg2 : f (-2) = 2 := by
  sorry

end NUMINAMATH_GPT_find_f_2_find_f_neg2_l1629_162927


namespace NUMINAMATH_GPT_red_balls_in_bag_l1629_162939

theorem red_balls_in_bag (r : ℕ) (h1 : 0 ≤ r ∧ r ≤ 12)
  (h2 : (r * (r - 1)) / (12 * 11) = 1 / 10) : r = 12 :=
sorry

end NUMINAMATH_GPT_red_balls_in_bag_l1629_162939


namespace NUMINAMATH_GPT_train_stop_time_per_hour_l1629_162948

theorem train_stop_time_per_hour
    (speed_excl_stoppages : ℕ)
    (speed_incl_stoppages : ℕ)
    (h1 : speed_excl_stoppages = 48)
    (h2 : speed_incl_stoppages = 36) :
    ∃ (t : ℕ), t = 15 :=
by
  sorry

end NUMINAMATH_GPT_train_stop_time_per_hour_l1629_162948


namespace NUMINAMATH_GPT_value_of_expression_l1629_162930

theorem value_of_expression (m : ℝ) (h : m^2 - m - 110 = 0) : (m - 1)^2 + m = 111 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1629_162930


namespace NUMINAMATH_GPT_problem1_problem2_l1629_162925

open Real

theorem problem1: 
  ((25^(1/3) - 125^(1/2)) / 5^(1/4) = 5^(5/12) - 5^(5/4)) :=
sorry

theorem problem2 (a : ℝ) (h : 0 < a): 
  (a^2 / (a^(1/2) * a^(2/3)) = a^(5/6)) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1629_162925


namespace NUMINAMATH_GPT_statement_two_statement_three_l1629_162905

section
variables {R : Type*} [Field R]
variables (a b c p q : R)
noncomputable def f (x : R) := a * x^2 + b * x + c

-- Statement ②
theorem statement_two (hpq : f a b c p = f a b c q) (hpq_neq : p ≠ q) : 
  f a b c (p + q) = c :=
sorry

-- Statement ③
theorem statement_three (hf : f a b c (p + q) = c) (hpq_neq : p ≠ q) : 
  p + q = 0 ∨ f a b c p = f a b c q :=
sorry

end

end NUMINAMATH_GPT_statement_two_statement_three_l1629_162905


namespace NUMINAMATH_GPT_hose_filling_time_l1629_162978

theorem hose_filling_time :
  ∀ (P A B C : ℝ), 
  (P / 3 = A + B) →
  (P / 5 = A + C) →
  (P / 4 = B + C) →
  (P / (A + B + C) = 2.55) :=
by
  intros P A B C hAB hAC hBC
  sorry

end NUMINAMATH_GPT_hose_filling_time_l1629_162978


namespace NUMINAMATH_GPT_time_to_cross_second_platform_l1629_162933

-- Definition of the conditions
variables (l_train l_platform1 l_platform2 t1 : ℕ)
variable (v : ℕ)

-- The conditions given in the problem
def conditions : Prop :=
  l_train = 190 ∧
  l_platform1 = 140 ∧
  l_platform2 = 250 ∧
  t1 = 15 ∧
  v = (l_train + l_platform1) / t1

-- The statement to prove
theorem time_to_cross_second_platform
    (l_train l_platform1 l_platform2 t1 : ℕ)
    (v : ℕ)
    (h : conditions l_train l_platform1 l_platform2 t1 v) :
    (l_train + l_platform2) / v = 20 :=
  sorry

end NUMINAMATH_GPT_time_to_cross_second_platform_l1629_162933


namespace NUMINAMATH_GPT_mn_value_l1629_162909

noncomputable def log_base (a b : ℝ) := Real.log b / Real.log a

theorem mn_value (M N : ℝ) (a : ℝ) 
  (h1 : log_base M N = a * log_base N M)
  (h2 : M ≠ N) (h3 : M * N > 0) (h4 : M ≠ 1) (h5 : N ≠ 1) (h6 : a = 4)
  : M * N = N^(3/2) ∨ M * N = N^(1/2) := 
by
  sorry

end NUMINAMATH_GPT_mn_value_l1629_162909


namespace NUMINAMATH_GPT_gcd_of_consecutive_digit_sums_l1629_162917

theorem gcd_of_consecutive_digit_sums :
  ∀ x y z : ℕ, x + 1 = y → y + 1 = z → gcd (101 * (x + z) + 10 * y) 212 = 212 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_consecutive_digit_sums_l1629_162917


namespace NUMINAMATH_GPT_equidistant_cyclist_l1629_162989

-- Definition of key parameters
def speed_car := 60  -- in km/h
def speed_cyclist := 18  -- in km/h
def speed_pedestrian := 6  -- in km/h
def distance_AC := 10  -- in km
def angle_ACB := 60  -- in degrees
def time_car_start := (7, 58)  -- 7:58 AM
def time_cyclist_start := (8, 0)  -- 8:00 AM
def time_pedestrian_start := (6, 44) -- 6:44 AM
def time_solution := (8, 6)  -- 8:06 AM

-- Time difference function
def time_diff (t1 t2 : Nat × Nat) : Nat :=
  (t2.1 - t1.1) * 60 + (t2.2 - t1.2)  -- time difference in minutes

-- Convert minutes to hours
noncomputable def minutes_to_hours (m : Nat) : ℝ :=
  m / 60.0

-- Distances traveled by car, cyclist, and pedestrian by the given time
noncomputable def distance_car (t1 t2 : Nat × Nat) : ℝ :=
  speed_car * (minutes_to_hours (time_diff t1 t2) + 2 / 60.0)

noncomputable def distance_cyclist (t1 t2 : Nat × Nat) : ℝ :=
  speed_cyclist * minutes_to_hours (time_diff t1 t2)

noncomputable def distance_pedestrian (t1 t2 : Nat × Nat) : ℝ :=
  speed_pedestrian * (minutes_to_hours (time_diff t1 t2) + 136 / 60.0)

-- Verification statement
theorem equidistant_cyclist :
  distance_car time_car_start time_solution = distance_pedestrian time_pedestrian_start time_solution → 
  distance_cyclist time_cyclist_start time_solution = 
  distance_car time_car_start time_solution ∧
  distance_cyclist time_cyclist_start time_solution = 
  distance_pedestrian time_pedestrian_start time_solution :=
by
  -- Given conditions and the correctness to be shown
  sorry

end NUMINAMATH_GPT_equidistant_cyclist_l1629_162989


namespace NUMINAMATH_GPT_ricky_time_difference_l1629_162974

noncomputable def old_man_time_per_mile : ℚ := 300 / 8
noncomputable def young_man_time_per_mile : ℚ := 160 / 12
noncomputable def time_difference : ℚ := old_man_time_per_mile - young_man_time_per_mile

theorem ricky_time_difference :
  time_difference = 24 := by
sorry

end NUMINAMATH_GPT_ricky_time_difference_l1629_162974


namespace NUMINAMATH_GPT_wheels_on_each_other_axle_l1629_162910

def truck_toll_wheels (t : ℝ) (x : ℝ) (w : ℕ) : Prop :=
  t = 1.50 + 1.50 * (x - 2) ∧ (w = 18) ∧ (∀ y : ℕ, y = 18 - 2 - 4 *(x - 5) / 4)

theorem wheels_on_each_other_axle :
  ∀ t x w, truck_toll_wheels t x w → w = 18 ∧ x = 5 → (18 - 2) / 4 = 4 :=
by
  intros t x w h₁ h₂
  have h₃ : t = 6 := sorry
  have h₄ : x = 4 := sorry
  have h₅ : w = 18 := sorry
  have h₆ : (18 - 2) / 4 = 4 := sorry
  exact h₆

end NUMINAMATH_GPT_wheels_on_each_other_axle_l1629_162910


namespace NUMINAMATH_GPT_students_in_class_l1629_162959

theorem students_in_class (S : ℕ)
  (h₁ : S / 2 + 2 * S / 5 - S / 10 = 4 * S / 5)
  (h₂ : S / 5 = 4) :
  S = 20 :=
sorry

end NUMINAMATH_GPT_students_in_class_l1629_162959


namespace NUMINAMATH_GPT_four_digit_cubes_divisible_by_16_count_l1629_162920

theorem four_digit_cubes_divisible_by_16_count :
  ∃ (count : ℕ), count = 3 ∧
    ∀ (m : ℕ), 1000 ≤ 64 * m^3 ∧ 64 * m^3 ≤ 9999 → (m = 3 ∨ m = 4 ∨ m = 5) :=
by {
  -- our proof would go here
  sorry
}

end NUMINAMATH_GPT_four_digit_cubes_divisible_by_16_count_l1629_162920


namespace NUMINAMATH_GPT_equalize_rice_move_amount_l1629_162976

open Real

noncomputable def containerA_kg : Real := 12
noncomputable def containerA_g : Real := 400
noncomputable def containerB_g : Real := 7600

noncomputable def total_rice_in_A_g : Real := containerA_kg * 1000 + containerA_g
noncomputable def total_rice_in_A_and_B_g : Real := total_rice_in_A_g + containerB_g
noncomputable def equalized_rice_per_container_g : Real := total_rice_in_A_and_B_g / 2

noncomputable def amount_to_move_g : Real := total_rice_in_A_g - equalized_rice_per_container_g
noncomputable def amount_to_move_kg : Real := amount_to_move_g / 1000

theorem equalize_rice_move_amount :
  amount_to_move_kg = 2.4 :=
by
  sorry

end NUMINAMATH_GPT_equalize_rice_move_amount_l1629_162976


namespace NUMINAMATH_GPT_minimize_total_cost_l1629_162994

noncomputable def event_probability_without_measures : ℚ := 0.3
noncomputable def loss_if_event_occurs : ℚ := 4000000
noncomputable def cost_measure_A : ℚ := 450000
noncomputable def prob_event_not_occurs_measure_A : ℚ := 0.9
noncomputable def cost_measure_B : ℚ := 300000
noncomputable def prob_event_not_occurs_measure_B : ℚ := 0.85

noncomputable def total_cost_no_measures : ℚ :=
  event_probability_without_measures * loss_if_event_occurs

noncomputable def total_cost_measure_A : ℚ :=
  cost_measure_A + (1 - prob_event_not_occurs_measure_A) * loss_if_event_occurs

noncomputable def total_cost_measure_B : ℚ :=
  cost_measure_B + (1 - prob_event_not_occurs_measure_B) * loss_if_event_occurs

noncomputable def total_cost_measures_A_and_B : ℚ :=
  cost_measure_A + cost_measure_B + (1 - prob_event_not_occurs_measure_A) * (1 - prob_event_not_occurs_measure_B) * loss_if_event_occurs

theorem minimize_total_cost :
  min (min total_cost_no_measures total_cost_measure_A) (min total_cost_measure_B total_cost_measures_A_and_B) = total_cost_measures_A_and_B :=
by sorry

end NUMINAMATH_GPT_minimize_total_cost_l1629_162994


namespace NUMINAMATH_GPT_initial_boxes_l1629_162904

theorem initial_boxes (x : ℕ) (h : x + 6 = 14) : x = 8 :=
by sorry

end NUMINAMATH_GPT_initial_boxes_l1629_162904


namespace NUMINAMATH_GPT_ratio_of_votes_l1629_162923

theorem ratio_of_votes (total_votes ben_votes : ℕ) (h_total : total_votes = 60) (h_ben : ben_votes = 24) :
  (ben_votes : ℚ) / (total_votes - ben_votes : ℚ) = 2 / 3 :=
by sorry

end NUMINAMATH_GPT_ratio_of_votes_l1629_162923


namespace NUMINAMATH_GPT_geometric_seq_fraction_l1629_162971

theorem geometric_seq_fraction (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = q * a n) 
  (h2 : (a 1 + 3 * a 3) / (a 2 + 3 * a 4) = 1 / 2) : 
  (a 4 * a 6 + a 6 * a 8) / (a 6 * a 8 + a 8 * a 10) = 1 / 16 :=
by
  sorry

end NUMINAMATH_GPT_geometric_seq_fraction_l1629_162971


namespace NUMINAMATH_GPT_product_remainder_div_5_l1629_162942

theorem product_remainder_div_5 :
  (1234 * 1567 * 1912) % 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_product_remainder_div_5_l1629_162942


namespace NUMINAMATH_GPT_geometric_series_first_term_l1629_162980

theorem geometric_series_first_term (r a S : ℝ) (hr : r = 1 / 8) (hS : S = 60) (hS_formula : S = a / (1 - r)) : 
  a = 105 / 2 := by
  rw [hr, hS] at hS_formula
  sorry

end NUMINAMATH_GPT_geometric_series_first_term_l1629_162980


namespace NUMINAMATH_GPT_inequality_proof_l1629_162961

variable (a b c d : ℝ)

theorem inequality_proof (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : a + b + c + d = 1) : 
  (1 / (4 * a + 3 * b + c) + 1 / (3 * a + b + 4 * d) + 1 / (a + 4 * c + 3 * d) + 1 / (4 * b + 3 * c + d)) ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1629_162961


namespace NUMINAMATH_GPT_triangle_YZ_length_l1629_162913

/-- In triangle XYZ, sides XY and XZ have lengths 6 and 8 inches respectively, 
    and the median XM from vertex X to the midpoint of side YZ is 5 inches. 
    Prove that the length of YZ is 10 inches. -/
theorem triangle_YZ_length
  (XY XZ XM : ℝ)
  (hXY : XY = 6)
  (hXZ : XZ = 8)
  (hXM : XM = 5) :
  ∃ (YZ : ℝ), YZ = 10 := 
by
  sorry

end NUMINAMATH_GPT_triangle_YZ_length_l1629_162913


namespace NUMINAMATH_GPT_div_37_permutation_l1629_162900

-- Let A, B, C be digits of a three-digit number
variables (A B C : ℕ) -- these can take values from 0 to 9
variables (p : ℕ) -- integer multiplier for the divisibility condition

-- The main theorem stated as a Lean 4 problem
theorem div_37_permutation (h : 100 * A + 10 * B + C = 37 * p) : 
  ∃ (M : ℕ), (M = 100 * B + 10 * C + A ∨ M = 100 * C + 10 * A + B ∨ M = 100 * A + 10 * C + B ∨ M = 100 * C + 10 * B + A ∨ M = 100 * B + 10 * A + C) ∧ 37 ∣ M :=
by
  sorry

end NUMINAMATH_GPT_div_37_permutation_l1629_162900


namespace NUMINAMATH_GPT_common_tangent_y_intercept_l1629_162947

noncomputable def circle_center_a : ℝ × ℝ := (1, 5)
noncomputable def circle_radius_a : ℝ := 3

noncomputable def circle_center_b : ℝ × ℝ := (15, 10)
noncomputable def circle_radius_b : ℝ := 10

theorem common_tangent_y_intercept :
  ∃ m b: ℝ, (m > 0) ∧ m = 700/1197 ∧ b = 7.416 ∧
  ∀ x y: ℝ, (y = m * x + b → ((x - 1)^2 + (y - 5)^2 = 9 ∨ (x - 15)^2 + (y - 10)^2 = 100)) := by
{
  sorry
}

end NUMINAMATH_GPT_common_tangent_y_intercept_l1629_162947


namespace NUMINAMATH_GPT_BigJoe_is_8_feet_l1629_162956

variable (Pepe_height : ℝ) (h1 : Pepe_height = 4.5)
variable (Frank_height : ℝ) (h2 : Frank_height = Pepe_height + 0.5)
variable (Larry_height : ℝ) (h3 : Larry_height = Frank_height + 1)
variable (Ben_height : ℝ) (h4 : Ben_height = Larry_height + 1)
variable (BigJoe_height : ℝ) (h5 : BigJoe_height = Ben_height + 1)

theorem BigJoe_is_8_feet : BigJoe_height = 8 := by
  sorry

end NUMINAMATH_GPT_BigJoe_is_8_feet_l1629_162956


namespace NUMINAMATH_GPT_spurs_total_basketballs_l1629_162908

theorem spurs_total_basketballs (players : ℕ) (basketballs_per_player : ℕ) (h1 : players = 22) (h2 : basketballs_per_player = 11) : players * basketballs_per_player = 242 := by
  sorry

end NUMINAMATH_GPT_spurs_total_basketballs_l1629_162908


namespace NUMINAMATH_GPT_count_6_digit_palindromes_with_even_middle_digits_l1629_162931

theorem count_6_digit_palindromes_with_even_middle_digits :
  let a_values := 9
  let b_even_values := 5
  let c_values := 10
  a_values * b_even_values * c_values = 450 :=
by {
  sorry
}

end NUMINAMATH_GPT_count_6_digit_palindromes_with_even_middle_digits_l1629_162931


namespace NUMINAMATH_GPT_exists_three_distinct_nats_sum_prod_squares_l1629_162977

theorem exists_three_distinct_nats_sum_prod_squares :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (∃ (x : ℕ), a + b + c = x^2) ∧ 
  (∃ (y : ℕ), a * b * c = y^2) :=
sorry

end NUMINAMATH_GPT_exists_three_distinct_nats_sum_prod_squares_l1629_162977


namespace NUMINAMATH_GPT_max_k_value_l1629_162951

def maximum_k (k : ℕ) : ℕ := 2

theorem max_k_value
  (k : ℕ)
  (h1 : 2 * k + 1 ≤ 20)  -- Condition implicitly implied by having subsets of a 20-element set
  (h2 : ∀ (s t : Finset (Fin 20)), s.card = 7 → t.card = 7 → s ≠ t → (s ∩ t).card = k) : k ≤ maximum_k k := 
by {
  sorry
}

end NUMINAMATH_GPT_max_k_value_l1629_162951


namespace NUMINAMATH_GPT_vertex_position_l1629_162914

-- Definitions based on the conditions of the problem
def quadratic_function (x : ℝ) : ℝ := 3*x^2 + 9*x + 5

-- Theorem that the vertex of the parabola is at x = -1.5
theorem vertex_position : ∃ x : ℝ, x = -1.5 ∧ ∀ y : ℝ, quadratic_function y ≥ quadratic_function x :=
by
  sorry

end NUMINAMATH_GPT_vertex_position_l1629_162914


namespace NUMINAMATH_GPT_time_in_1867_minutes_correct_l1629_162998

def current_time := (3, 15) -- (hours, minutes)
def minutes_in_hour := 60
def total_minutes := 1867
def hours_after := total_minutes / minutes_in_hour
def remainder_minutes := total_minutes % minutes_in_hour
def result_time := ((current_time.1 + hours_after) % 24, current_time.2 + remainder_minutes)
def expected_time := (22, 22) -- 10:22 p.m. in 24-hour format

theorem time_in_1867_minutes_correct : result_time = expected_time := 
by
    -- No proof is required according to the instructions.
    sorry

end NUMINAMATH_GPT_time_in_1867_minutes_correct_l1629_162998


namespace NUMINAMATH_GPT_product_of_primes_l1629_162985

theorem product_of_primes : 5 * 7 * 997 = 34895 :=
by
  sorry

end NUMINAMATH_GPT_product_of_primes_l1629_162985


namespace NUMINAMATH_GPT_friend_initial_money_l1629_162946

theorem friend_initial_money (F : ℕ) : 
    (160 + 25 * 7 = F + 25 * 5) → 
    (F = 210) :=
by
  sorry

end NUMINAMATH_GPT_friend_initial_money_l1629_162946


namespace NUMINAMATH_GPT_inequality_solution_sets_l1629_162982

noncomputable def solve_inequality (m : ℝ) : Set ℝ :=
  if m = 0 then Set.Iic (-2)
  else if m > 0 then Set.Iic (-2) ∪ Set.Ici (1 / m)
  else if (-(1/2) < m ∧ m < 0) then Set.Ioo (1 / m) (-2)
  else if m = -(1 / 2) then ∅
  else Set.Ioo (-2) (1 / m)

theorem inequality_solution_sets (m : ℝ) :
  solve_inequality m = 
    if m = 0 then Set.Iic (-2)
    else if m > 0 then Set.Iic (-2) ∪ Set.Ici (1 / m)
    else if (-(1/2) < m ∧ m < 0) then Set.Ioo (1 / m) (-2)
    else if m = -(1 / 2) then ∅
    else Set.Ioo (-2) (1 / m) :=
sorry

end NUMINAMATH_GPT_inequality_solution_sets_l1629_162982


namespace NUMINAMATH_GPT_percentage_decrease_in_spring_l1629_162945

-- Given Conditions
variables (initial_members : ℕ) (increased_percent : ℝ) (total_decrease_percent : ℝ)
-- population changes
variables (fall_members : ℝ) (spring_members : ℝ)

-- The initial conditions given by the problem
axiom initial_membership : initial_members = 100
axiom fall_increase : increased_percent = 6
axiom total_decrease : total_decrease_percent = 14.14

-- Derived values based on conditions
axiom fall_members_calculated : fall_members = initial_members * (1 + increased_percent / 100)
axiom spring_members_calculated : spring_members = initial_members * (1 - total_decrease_percent / 100)

-- The correct answer which we need to prove
theorem percentage_decrease_in_spring : 
  ((fall_members - spring_members) / fall_members) * 100 = 19 := by
  sorry

end NUMINAMATH_GPT_percentage_decrease_in_spring_l1629_162945


namespace NUMINAMATH_GPT_Doris_spent_6_l1629_162996

variable (D : ℝ)

theorem Doris_spent_6 (h0 : 24 - (D + D / 2) = 15) : D = 6 :=
by
  sorry

end NUMINAMATH_GPT_Doris_spent_6_l1629_162996


namespace NUMINAMATH_GPT_cost_of_potatoes_l1629_162987

theorem cost_of_potatoes
  (per_person_potatoes : ℕ → ℕ → ℕ)
  (amount_of_people : ℕ)
  (bag_cost : ℕ)
  (bag_weight : ℕ)
  (people : ℕ)
  (cost : ℕ) :
  (per_person_potatoes people amount_of_people = 60) →
  (60 / bag_weight = 3) →
  (3 * bag_cost = cost) →
  cost = 15 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_potatoes_l1629_162987


namespace NUMINAMATH_GPT_bugs_eat_same_flowers_l1629_162979

theorem bugs_eat_same_flowers (num_bugs : ℕ) (total_flowers : ℕ) (flowers_per_bug : ℕ) 
  (h1 : num_bugs = 3) (h2 : total_flowers = 6) (h3 : flowers_per_bug = total_flowers / num_bugs) : 
  flowers_per_bug = 2 :=
by
  sorry

end NUMINAMATH_GPT_bugs_eat_same_flowers_l1629_162979


namespace NUMINAMATH_GPT_cannot_determine_c_l1629_162921

-- Definitions based on conditions
variables {a b c d : ℕ}
axiom h1 : a + b + c = 21
axiom h2 : a + b + d = 27
axiom h3 : a + c + d = 30

-- The statement that c cannot be determined exactly
theorem cannot_determine_c : ¬ (∃ c : ℕ, c = c) :=
by sorry

end NUMINAMATH_GPT_cannot_determine_c_l1629_162921


namespace NUMINAMATH_GPT_students_not_taking_test_l1629_162902

theorem students_not_taking_test (total_students students_q1 students_q2 students_both not_taken : ℕ)
  (h_total : total_students = 30)
  (h_q1 : students_q1 = 25)
  (h_q2 : students_q2 = 22)
  (h_both : students_both = 22)
  (h_not_taken : not_taken = total_students - students_q2) :
  not_taken = 8 := by
  sorry

end NUMINAMATH_GPT_students_not_taking_test_l1629_162902


namespace NUMINAMATH_GPT_value_independent_of_b_value_for_d_zero_l1629_162988

theorem value_independent_of_b
  (c b d h : ℝ)
  (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ)
  (h1 : x1 = b - d - h)
  (h2 : x2 = b - d)
  (h3 : x3 = b + d)
  (h4 : x4 = b + d + h)
  (hy1 : y1 = c * x1^2)
  (hy2 : y2 = c * x2^2)
  (hy3 : y3 = c * x3^2)
  (hy4 : y4 = c * x4^2) :
  (y1 + y4 - y2 - y3) = 2 * c * h * (2 * d + h) :=
by
  sorry

theorem value_for_d_zero
  (c b h : ℝ)
  (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ)
  (d : ℝ := 0)
  (h1 : x1 = b - h)
  (h2 : x2 = b)
  (h3 : x3 = b)
  (h4 : x4 = b + h)
  (hy1 : y1 = c * x1^2)
  (hy2 : y2 = c * x2^2)
  (hy3 : y3 = c * x3^2)
  (hy4 : y4 = c * x4^2) :
  (y1 + y4 - y2 - y3) = 2 * c * h^2 :=
by
  sorry

end NUMINAMATH_GPT_value_independent_of_b_value_for_d_zero_l1629_162988


namespace NUMINAMATH_GPT_root_shifted_is_root_of_quadratic_with_integer_coeffs_l1629_162993

theorem root_shifted_is_root_of_quadratic_with_integer_coeffs
  (a b c t : ℤ)
  (h : a ≠ 0)
  (h_root : a * t^2 + b * t + c = 0) :
  ∃ (x : ℤ), a * x^2 + (4 * a + b) * x + (4 * a + 2 * b + c) = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_root_shifted_is_root_of_quadratic_with_integer_coeffs_l1629_162993


namespace NUMINAMATH_GPT_masking_tape_problem_l1629_162935

variable (width_other : ℕ)

theorem masking_tape_problem
  (h1 : ∀ w : ℕ, (2 * 4 + 2 * w) = 20)
  : width_other = 6 :=
by
  have h2 : 8 + 2 * width_other = 20 := h1 width_other
  sorry

end NUMINAMATH_GPT_masking_tape_problem_l1629_162935


namespace NUMINAMATH_GPT_find_d_square_plus_5d_l1629_162966

theorem find_d_square_plus_5d (a b c d : ℤ) (h₁: a^2 + 2 * a = 65) (h₂: b^2 + 3 * b = 125) (h₃: c^2 + 4 * c = 205) (h₄: d = 5 + 6) :
  d^2 + 5 * d = 176 :=
by
  rw [h₄]
  sorry

end NUMINAMATH_GPT_find_d_square_plus_5d_l1629_162966


namespace NUMINAMATH_GPT_joe_max_money_l1629_162912

noncomputable def max_guaranteed_money (initial_money : ℕ) (max_bet : ℕ) (num_bets : ℕ) : ℕ :=
  if initial_money = 100 ∧ max_bet = 17 ∧ num_bets = 5 then 98 else 0

theorem joe_max_money : max_guaranteed_money 100 17 5 = 98 := by
  sorry

end NUMINAMATH_GPT_joe_max_money_l1629_162912


namespace NUMINAMATH_GPT_largest_regular_hexagon_proof_l1629_162954

noncomputable def largest_regular_hexagon_side_length (x : ℝ) (H : ConvexHexagon) 
  (hx : -5 < x ∧ x < 6) : ℝ := 11 / 2

-- Convex Hexagon Definition
structure ConvexHexagon :=
  (sides : Vector ℝ 6)
  (is_convex : true)  -- Placeholder for convex property

theorem largest_regular_hexagon_proof (x : ℝ) (H : ConvexHexagon) 
  (hx : -5 < x ∧ x < 6)
  (H_sides_length : H.sides = ⟨[5, 6, 7, 5+x, 6-x, 7+x], by simp⟩) :
  largest_regular_hexagon_side_length x H hx = 11 / 2 :=
sorry

end NUMINAMATH_GPT_largest_regular_hexagon_proof_l1629_162954


namespace NUMINAMATH_GPT_alex_needs_more_coins_l1629_162937

-- Define the conditions and problem statement 
def num_friends : ℕ := 15
def coins_alex_has : ℕ := 95 

-- The total number of coins required is
def total_coins_needed : ℕ := num_friends * (num_friends + 1) / 2

-- The minimum number of additional coins needed
def additional_coins_needed : ℕ := total_coins_needed - coins_alex_has

-- Formalize the theorem 
theorem alex_needs_more_coins : additional_coins_needed = 25 := by
  -- Here we would provide the actual proof steps
  sorry

end NUMINAMATH_GPT_alex_needs_more_coins_l1629_162937


namespace NUMINAMATH_GPT_fraction_product_l1629_162929

theorem fraction_product :
  (7 / 4) * (8 / 14) * (28 / 16) * (24 / 36) * (49 / 35) * (40 / 25) * (63 / 42) * (32 / 48) = 56 / 25 :=
by sorry

end NUMINAMATH_GPT_fraction_product_l1629_162929


namespace NUMINAMATH_GPT_smallest_crate_side_l1629_162968

/-- 
A crate measures some feet by 8 feet by 12 feet on the inside. 
A stone pillar in the shape of a right circular cylinder must fit into the crate for shipping so that 
it rests upright when the crate sits on at least one of its six sides. 
The radius of the pillar is 7 feet. 
Prove that the length of the crate's smallest side is 8 feet.
-/
theorem smallest_crate_side (x : ℕ) (hx : x >= 14) : min (min x 8) 12 = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_crate_side_l1629_162968


namespace NUMINAMATH_GPT_add_base_6_l1629_162995

theorem add_base_6 (a b c : ℕ) (h₀ : a = 3 * 6^3 + 4 * 6^2 + 2 * 6 + 1)
                    (h₁ : b = 4 * 6^3 + 5 * 6^2 + 2 * 6 + 5)
                    (h₂ : c = 1 * 6^4 + 2 * 6^3 + 3 * 6^2 + 5 * 6 + 0) : 
  a + b = c :=
by  
  sorry

end NUMINAMATH_GPT_add_base_6_l1629_162995


namespace NUMINAMATH_GPT_range_of_x_l1629_162938

theorem range_of_x (a : ℝ) (x : ℝ) (h_a : 1 ≤ a) : 
  ax^2 + (a - 3) * x + (a - 4) > 0 ↔ x < -1 ∨ x > 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l1629_162938


namespace NUMINAMATH_GPT_triangle_angles_are_equal_l1629_162949

theorem triangle_angles_are_equal
  (A B C : ℝ) (a b c : ℝ)
  (h1 : A + B + C = π)
  (h2 : A = B + (B - A))
  (h3 : B = C + (C - B))
  (h4 : 2 * (1 / b) = (1 / a) + (1 / c)) :
  A = π / 3 ∧ B = π / 3 ∧ C = π / 3 :=
sorry

end NUMINAMATH_GPT_triangle_angles_are_equal_l1629_162949


namespace NUMINAMATH_GPT_heights_inequality_l1629_162952

theorem heights_inequality (a b c h_a h_b h_c p R : ℝ) (h : a ≤ b ∧ b ≤ c) : 
  h_a + h_b + h_c ≤ (3 * b * (a^2 + a * c + c^2)) / (4 * p * R) := 
sorry

end NUMINAMATH_GPT_heights_inequality_l1629_162952
