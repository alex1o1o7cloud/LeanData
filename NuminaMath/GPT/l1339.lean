import Mathlib

namespace no_positive_integer_n_ge_2_1001_n_is_square_of_prime_l1339_133915

noncomputable def is_square_of_prime (m : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ m = p * p

theorem no_positive_integer_n_ge_2_1001_n_is_square_of_prime :
  ∀ n : ℕ, n ≥ 2 → ¬ is_square_of_prime (n^3 + 1) :=
by
  intro n hn
  sorry

end no_positive_integer_n_ge_2_1001_n_is_square_of_prime_l1339_133915


namespace james_selling_price_l1339_133993

variable (P : ℝ)  -- Selling price per candy bar

theorem james_selling_price 
  (boxes_sold : ℕ)
  (candy_bars_per_box : ℕ) 
  (cost_price_per_candy_bar : ℝ)
  (total_profit : ℝ)
  (H1 : candy_bars_per_box = 10)
  (H2 : boxes_sold = 5)
  (H3 : cost_price_per_candy_bar = 1)
  (H4 : total_profit = 25)
  (profit_eq : boxes_sold * candy_bars_per_box * (P - cost_price_per_candy_bar) = total_profit)
  : P = 1.5 :=
by 
  sorry

end james_selling_price_l1339_133993


namespace maximum_area_rhombus_l1339_133981

theorem maximum_area_rhombus 
    (x₀ y₀ k : ℝ)
    (h1 : 2 ≤ x₀ ∧ x₀ ≤ 4)
    (h2 : y₀ = k / x₀)
    (h3 : ∀ x > 0, ∃ y, y = k / x) :
    (∀ (x₀ : ℝ), 2 ≤ x₀ ∧ x₀ ≤ 4 → ∃ (S : ℝ), S = 3 * (Real.sqrt 2 / 2 * x₀^2) → S ≤ 24 * Real.sqrt 2) :=
by
  sorry

end maximum_area_rhombus_l1339_133981


namespace bus_distance_covered_l1339_133906

theorem bus_distance_covered (speedTrain speedCar speedBus distanceBus : ℝ) (h1 : speedTrain / speedCar = 16 / 15)
                            (h2 : speedBus = (3 / 4) * speedTrain) (h3 : 450 / 6 = speedCar) (h4 : distanceBus = 8 * speedBus) :
                            distanceBus = 480 :=
by
  sorry

end bus_distance_covered_l1339_133906


namespace people_on_trolley_l1339_133948

-- Given conditions
variable (X : ℕ)

def initial_people : ℕ := 10

def second_stop_people : ℕ := initial_people - 3 + 20

def third_stop_people : ℕ := second_stop_people - 18 + 2

def fourth_stop_people : ℕ := third_stop_people - 5 + X

-- Prove the current number of people on the trolley is 6 + X
theorem people_on_trolley (X : ℕ) : 
  fourth_stop_people X = 6 + X := 
by 
  unfold fourth_stop_people
  unfold third_stop_people
  unfold second_stop_people
  unfold initial_people
  sorry

end people_on_trolley_l1339_133948


namespace range_of_a_l1339_133959

theorem range_of_a (a : ℝ) :
  (∃ A : Finset ℝ, 
    (∀ x, x ∈ A ↔ x^3 - 2 * x^2 + a * x = 0) ∧ A.card = 3) ↔ (a < 0 ∨ (0 < a ∧ a < 1)) :=
by
  sorry

end range_of_a_l1339_133959


namespace min_value_fraction_l1339_133925

theorem min_value_fraction {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 1) :
  (∀ y : ℝ,  y > 0 → (∀ x : ℝ, x > 0 → x + 3 * y = 1 → (1/x + 1/(3*y)) ≥ 4)) :=
sorry

end min_value_fraction_l1339_133925


namespace average_excluding_highest_lowest_l1339_133947

-- Define the conditions
def batting_average : ℚ := 59
def innings : ℕ := 46
def highest_score : ℕ := 156
def score_difference : ℕ := 150
def lowest_score : ℕ := highest_score - score_difference

-- Prove the average excluding the highest and lowest innings is 58
theorem average_excluding_highest_lowest :
  let total_runs := batting_average * innings
  let runs_excluding := total_runs - highest_score - lowest_score
  let effective_innings := innings - 2
  runs_excluding / effective_innings = 58 := by
  -- Insert proof here
  sorry

end average_excluding_highest_lowest_l1339_133947


namespace age_difference_is_20_l1339_133907

-- Definitions for the ages of the two persons
def elder_age := 35
def younger_age := 15

-- Condition: Difference in ages
def age_difference := elder_age - younger_age

-- Theorem to prove the difference in ages is 20 years
theorem age_difference_is_20 : age_difference = 20 := by
  sorry

end age_difference_is_20_l1339_133907


namespace max_value_of_y_l1339_133922

theorem max_value_of_y (x : ℝ) (h : 0 < x ∧ x < 1 / 2) : (∃ y, y = x^2 * (1 - 2*x) ∧ y ≤ 1 / 27) :=
sorry

end max_value_of_y_l1339_133922


namespace jackson_running_increase_l1339_133928

theorem jackson_running_increase
    (initial_miles_per_day : ℕ)
    (final_miles_per_day : ℕ)
    (weeks_increasing : ℕ)
    (total_weeks : ℕ)
    (h1 : initial_miles_per_day = 3)
    (h2 : final_miles_per_day = 7)
    (h3 : weeks_increasing = 4)
    (h4 : total_weeks = 5) :
    (final_miles_per_day - initial_miles_per_day) / weeks_increasing = 1 := 
by
  -- provided steps from solution
  sorry

end jackson_running_increase_l1339_133928


namespace analytical_expression_of_odd_function_l1339_133931

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 2 * x + 3 
else if x = 0 then 0 
else -x^2 - 2 * x - 3

theorem analytical_expression_of_odd_function :
  ∀ x : ℝ, f x =
    if x > 0 then x^2 - 2 * x + 3 
    else if x = 0 then 0 
    else -x^2 - 2 * x - 3 :=
by
  sorry

end analytical_expression_of_odd_function_l1339_133931


namespace tree_leaves_remaining_after_three_weeks_l1339_133982

theorem tree_leaves_remaining_after_three_weeks :
  let initial_leaves := 1000
  let leaves_shed_first_week := (2 / 5 : ℝ) * initial_leaves
  let leaves_remaining_after_first_week := initial_leaves - leaves_shed_first_week
  let leaves_shed_second_week := (4 / 10 : ℝ) * leaves_remaining_after_first_week
  let leaves_remaining_after_second_week := leaves_remaining_after_first_week - leaves_shed_second_week
  let leaves_shed_third_week := (3 / 4 : ℝ) * leaves_shed_second_week
  let leaves_remaining_after_third_week := leaves_remaining_after_second_week - leaves_shed_third_week
  leaves_remaining_after_third_week = 180 :=
by
  sorry

end tree_leaves_remaining_after_three_weeks_l1339_133982


namespace bees_on_second_day_l1339_133924

-- Define the number of bees on the first day
def bees_first_day : ℕ := 144

-- Define the multiplication factor
def multiplication_factor : ℕ := 3

-- Define the number of bees on the second day
def bees_second_day : ℕ := bees_first_day * multiplication_factor

-- Theorem stating the number of bees on the second day is 432
theorem bees_on_second_day : bees_second_day = 432 := 
by
  sorry

end bees_on_second_day_l1339_133924


namespace negate_exists_implies_forall_l1339_133909

-- Define the original proposition
def prop1 (x : ℝ) : Prop := x^2 + 2 * x + 2 < 0

-- The negation of the proposition
def neg_prop1 := ∀ x : ℝ, x^2 + 2 * x + 2 ≥ 0

-- Statement of the equivalence
theorem negate_exists_implies_forall :
  ¬(∃ x : ℝ, prop1 x) ↔ neg_prop1 := by
  sorry

end negate_exists_implies_forall_l1339_133909


namespace Roja_speed_is_8_l1339_133900

def Pooja_speed : ℝ := 3
def time_in_hours : ℝ := 4
def distance_between_them : ℝ := 44

theorem Roja_speed_is_8 :
  ∃ R : ℝ, R + Pooja_speed = (distance_between_them / time_in_hours) ∧ R = 8 :=
by
  sorry

end Roja_speed_is_8_l1339_133900


namespace smallest_n_transform_l1339_133953

open Real

noncomputable def line1_angle : ℝ := π / 30
noncomputable def line2_angle : ℝ := π / 40
noncomputable def line_slope : ℝ := 2 / 45
noncomputable def transform_angle (theta : ℝ) (n : ℕ) : ℝ := theta + n * (7 * π / 120)

theorem smallest_n_transform (theta : ℝ) (n : ℕ) (m : ℕ)
  (h_line1 : line1_angle = π / 30)
  (h_line2 : line2_angle = π / 40)
  (h_slope : tan theta = line_slope)
  (h_transform : transform_angle theta n = theta + m * 2 * π) :
  n = 120 := 
sorry

end smallest_n_transform_l1339_133953


namespace value_of_x_plus_y_l1339_133908

theorem value_of_x_plus_y (x y : ℝ) 
  (h1 : 2 * x - y = -1) 
  (h2 : x + 4 * y = 22) : 
  x + y = 7 :=
sorry

end value_of_x_plus_y_l1339_133908


namespace find_x_l1339_133930

theorem find_x 
  (x : ℝ)
  (h : 3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * x * 0.5)) = 2800.0000000000005) : 
  x = 0.225 := 
sorry

end find_x_l1339_133930


namespace find_x_value_l1339_133992

def average_eq_condition (x : ℝ) : Prop :=
  (5050 + x) / 101 = 50 * (x + 1)

theorem find_x_value : ∃ x : ℝ, average_eq_condition x ∧ x = 0 :=
by
  use 0
  sorry

end find_x_value_l1339_133992


namespace gcd_12345_6789_l1339_133999

theorem gcd_12345_6789 : Int.gcd 12345 6789 = 3 :=
by
  sorry

end gcd_12345_6789_l1339_133999


namespace range_of_f_lt_zero_l1339_133950

noncomputable
def f : ℝ → ℝ := sorry

theorem range_of_f_lt_zero 
  (hf_even : ∀ x, f x = f (-x))
  (hf_decreasing : ∀ x y, x < y ∧ y ≤ 0 → f x > f y)
  (hf_at_neg2_zero : f (-2) = 0) :
  {x : ℝ | f x < 0} = {x : ℝ | -2 < x ∧ x < 2} :=
by
  sorry

end range_of_f_lt_zero_l1339_133950


namespace bottle_caps_per_box_l1339_133903

theorem bottle_caps_per_box (total_bottle_caps boxes : ℕ) (hb : total_bottle_caps = 316) (bn : boxes = 79) :
  total_bottle_caps / boxes = 4 :=
by
  sorry

end bottle_caps_per_box_l1339_133903


namespace f_is_periodic_l1339_133960

-- Define the conditions for the function f
def f (x : ℝ) : ℝ := sorry
axiom f_defined : ∀ x : ℝ, f x ≠ 0
axiom f_property : ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f (x - a) = 1 / f x

-- Formal problem statement to be proven
theorem f_is_periodic : ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f x = f (x + 2 * a) :=
by {
  sorry
}

end f_is_periodic_l1339_133960


namespace largest_possible_sum_l1339_133974

theorem largest_possible_sum (a b : ℤ) (h : a^2 - b^2 = 144) : a + b ≤ 72 :=
sorry

end largest_possible_sum_l1339_133974


namespace minimum_value_of_expression_l1339_133966

theorem minimum_value_of_expression {k x1 x2 : ℝ} 
  (h1 : x1 + x2 = -2 * k)
  (h2 : x1 * x2 = k^2 + k + 3) : 
  (x1 - 1)^2 + (x2 - 1)^2 ≥ 8 :=
sorry

end minimum_value_of_expression_l1339_133966


namespace log_expression_evaluation_l1339_133940

theorem log_expression_evaluation : 
  (4 * Real.log 2 + 3 * Real.log 5 - Real.log (1/5)) = 4 := 
  sorry

end log_expression_evaluation_l1339_133940


namespace min_value_inequality_l1339_133939

theorem min_value_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 3 * b = 1) :
  1 / a + 3 / b ≥ 16 := 
by
  sorry

end min_value_inequality_l1339_133939


namespace chairs_left_proof_l1339_133938

def red_chairs : ℕ := 4
def yellow_chairs : ℕ := 2 * red_chairs
def blue_chairs : ℕ := 3 * yellow_chairs
def green_chairs : ℕ := blue_chairs / 2
def orange_chairs : ℕ := green_chairs + 2
def total_chairs : ℕ := red_chairs + yellow_chairs + blue_chairs + green_chairs + orange_chairs
def borrowed_chairs : ℕ := 5 + 3
def chairs_left : ℕ := total_chairs - borrowed_chairs

theorem chairs_left_proof : chairs_left = 54 := by
  -- This is where the proof would go
  sorry

end chairs_left_proof_l1339_133938


namespace count_more_blue_l1339_133979

-- Definitions derived from the provided conditions
variables (total_people more_green both neither : ℕ)
variable (more_blue : ℕ)

-- Condition 1: There are 150 people in total
axiom total_people_def : total_people = 150

-- Condition 2: 90 people believe that teal is "more green"
axiom more_green_def : more_green = 90

-- Condition 3: 35 people believe it is both "more green" and "more blue"
axiom both_def : both = 35

-- Condition 4: 25 people think that teal is neither "more green" nor "more blue"
axiom neither_def : neither = 25


-- Theorem statement
theorem count_more_blue (total_people more_green both neither more_blue : ℕ) 
  (total_people_def : total_people = 150)
  (more_green_def : more_green = 90)
  (both_def : both = 35)
  (neither_def : neither = 25) :
  more_blue = 70 :=
by
  sorry

end count_more_blue_l1339_133979


namespace sum_squares_condition_l1339_133964

theorem sum_squares_condition
  (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 75)
  (h2 : ab + bc + ca = 40)
  (h3 : c = 5) :
  a + b + c = 5 * Real.sqrt 62 :=
by sorry

end sum_squares_condition_l1339_133964


namespace mike_total_spent_l1339_133978

noncomputable def total_spent_by_mike (food_cost wallet_cost shirt_cost shoes_cost belt_cost 
  discounted_shirt_cost discounted_shoes_cost discounted_belt_cost : ℝ) : ℝ :=
  food_cost + wallet_cost + discounted_shirt_cost + discounted_shoes_cost + discounted_belt_cost

theorem mike_total_spent :
  let food_cost := 30
  let wallet_cost := food_cost + 60
  let shirt_cost := wallet_cost / 3
  let shoes_cost := 2 * wallet_cost
  let belt_cost := shoes_cost - 45
  let discounted_shirt_cost := shirt_cost - (0.2 * shirt_cost)
  let discounted_shoes_cost := shoes_cost - (0.15 * shoes_cost)
  let discounted_belt_cost := belt_cost - (0.1 * belt_cost)
  total_spent_by_mike food_cost wallet_cost shirt_cost shoes_cost belt_cost
    discounted_shirt_cost discounted_shoes_cost discounted_belt_cost = 418.50 := by
  sorry

end mike_total_spent_l1339_133978


namespace ali_babas_cave_min_moves_l1339_133991

theorem ali_babas_cave_min_moves : 
  ∀ (counters : Fin 28 → Fin 2018) (decrease_by : ℕ → Fin 28 → ℕ),
    (∀ n, n < 28 → decrease_by n ≤ 2017) → 
    (∃ (k : ℕ), k ≤ 11 ∧ 
      ∀ n, (n < 28 → decrease_by (k - n) n = 0)) :=
sorry

end ali_babas_cave_min_moves_l1339_133991


namespace exists_same_color_points_distance_one_l1339_133927

theorem exists_same_color_points_distance_one
    (color : ℝ × ℝ → Fin 3)
    (h : ∀ p q : ℝ × ℝ, dist p q = 1 → color p ≠ color q) :
  ∃ p q : ℝ × ℝ, dist p q = 1 ∧ color p = color q :=
sorry

end exists_same_color_points_distance_one_l1339_133927


namespace intersection_A_B_l1339_133902

def A := {x : ℝ | -2 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | ∃ y : ℝ, y = x^2 + 2}

theorem intersection_A_B :
  {x : ℝ | x ∈ A ∧ ∃ y : ℝ, y = x^2 + 2} = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := sorry

end intersection_A_B_l1339_133902


namespace externally_tangent_internally_tangent_common_chord_and_length_l1339_133937

-- Definitions of Circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y - 1 = 0
def circle2 (x y : ℝ) (m : ℝ) : Prop := x^2 + y^2 - 10*x - 12*y + m = 0

-- Proof problem 1: Externally tangent
theorem externally_tangent (m : ℝ) : (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) → m = 25 + 10 * Real.sqrt 11 :=
sorry

-- Proof problem 2: Internally tangent
theorem internally_tangent (m : ℝ) : (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) → m = 25 - 10 * Real.sqrt 11 :=
sorry

-- Proof problem 3: Common chord and length when m = 45
theorem common_chord_and_length :
  (∃ x y : ℝ, circle2 x y 45) →
  (∃ l : ℝ, l = 4 * Real.sqrt 7 ∧ ∀ x y : ℝ, (circle1 x y ∧ circle2 x y 45) → (4*x + 3*y - 23 = 0)) :=
sorry

end externally_tangent_internally_tangent_common_chord_and_length_l1339_133937


namespace problem_a5_value_l1339_133917

def Sn (n : ℕ) : ℕ := 2 * n^2 + 3 * n - 1

theorem problem_a5_value : Sn 5 - Sn 4 = 21 := by
  sorry

end problem_a5_value_l1339_133917


namespace problem_1_problem_2_problem_3_l1339_133961

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := 8 * x^2 + 16 * x - k
noncomputable def g (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 + 4 * x
noncomputable def h (x : ℝ) (k : ℝ) : ℝ := g x - f x k

theorem problem_1 (k : ℝ) : (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x k ≤ g x) → 45 ≤ k := by
  sorry

theorem problem_2 (k : ℝ) : (∃ x : ℝ, -3 ≤ x ∧ x ≤ 3 ∧ f x k ≤ g x) → -7 ≤ k := by
  sorry

theorem problem_3 (k : ℝ) : (∀ x1 x2 : ℝ, (-3 ≤ x1 ∧ x1 ≤ 3) ∧ (-3 ≤ x2 ∧ x2 ≤ 3) → f x1 k ≤ g x2) → 141 ≤ k := by
  sorry

end problem_1_problem_2_problem_3_l1339_133961


namespace closest_multiple_of_18_2021_l1339_133926

def is_multiple_of (n k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def closest_multiple_of (n k : ℕ) : ℕ :=
if (n % k) * 2 < k then n - (n % k) else n + (k - n % k)

theorem closest_multiple_of_18_2021 :
  closest_multiple_of 2021 18 = 2016 := by
    sorry

end closest_multiple_of_18_2021_l1339_133926


namespace tangent_line_through_B_l1339_133912

theorem tangent_line_through_B (x : ℝ) (y : ℝ) (x₀ : ℝ) (y₀ : ℝ) :
  (y₀ = x₀^2) →
  (y - y₀ = 2*x₀*(x - x₀)) →
  (3, 5) ∈ ({p : ℝ × ℝ | ∃ t, p.2 - t^2 = 2*t*(p.1 - t)}) →
  (x = 2 * x₀) ∧ (y = y₀) →
  (2*x - y - 1 = 0 ∨ 10*x - y - 25 = 0) :=
by
  intros h1 h2 h3 h4
  sorry

end tangent_line_through_B_l1339_133912


namespace existence_of_point_N_l1339_133916

-- Given conditions
def is_point_on_ellipse (x y a b : ℝ) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

def is_ellipse (a b : ℝ) : Prop := 
  a > b ∧ b > 0 ∧ (a^2 = b^2 + (a * (Real.sqrt 2) / 2)^2)

def passes_through_point (x y a b : ℝ) (px py : ℝ) : Prop :=
  (px^2 / a^2) + (py^2 / b^2) = 1

def ellipse_with_eccentricity (a : ℝ) : Prop :=
  (Real.sqrt 2) / 2 = (Real.sqrt (a^2 - (a * (Real.sqrt 2) / 2)^2)) / a

def line_through_point (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1

def lines_intersect_ellipse (k a b : ℝ) : Prop :=
  ∃ x1 y1 x2 y2, line_through_point k x1 y1 ∧ line_through_point k x2 y2 ∧ is_point_on_ellipse x1 y1 a b ∧ is_point_on_ellipse x2 y2 a b

def angle_condition (k t a b : ℝ) : Prop :=
  ∃ x1 y1 x2 y2, line_through_point k x1 y1 ∧ line_through_point k x2 y2 ∧ is_point_on_ellipse x1 y1 a b ∧ is_point_on_ellipse x2 y2 a b ∧ 
  ((y1 - t) / x1) + ((y2 - t) / x2) = 0

-- Lean 4 statement
theorem existence_of_point_N (a b k t : ℝ) (hx : is_ellipse a b) (hp : passes_through_point 2 (Real.sqrt 2) a b 2 (Real.sqrt 2)) (he : ellipse_with_eccentricity a) (hl : ∀ (x1 y1 x2 y2 : ℝ), lines_intersect_ellipse k a b) :
  ∃ (N : ℝ), N = 4 ∧ angle_condition k N a b :=
sorry

end existence_of_point_N_l1339_133916


namespace charlie_share_l1339_133995

theorem charlie_share (A B C D E : ℝ) (h1 : A = (1/3) * B)
  (h2 : B = (1/2) * C) (h3 : C = 0.75 * D) (h4 : D = 2 * E) 
  (h5 : A + B + C + D + E = 15000) : C = 15000 * (3 / 11) :=
by
  sorry

end charlie_share_l1339_133995


namespace min_d_value_l1339_133986

noncomputable def minChordLength (a : ℝ) : ℝ :=
  let P1 := (Real.arcsin a, Real.arcsin a)
  let P2 := (Real.arccos a, -Real.arccos a)
  let d_sq := 2 * ((Real.arcsin a)^2 + (Real.arccos a)^2)
  Real.sqrt d_sq

theorem min_d_value {a : ℝ} (h₁ : a ∈ Set.Icc (-1) 1) : 
  ∃ d : ℝ, d = minChordLength a ∧ d ≥ (π / 2) :=
sorry

end min_d_value_l1339_133986


namespace joe_lowest_test_score_dropped_l1339_133942

theorem joe_lowest_test_score_dropped 
  (A B C D : ℝ) 
  (h1 : A + B + C + D = 360) 
  (h2 : A + B + C = 255) :
  D = 105 :=
sorry

end joe_lowest_test_score_dropped_l1339_133942


namespace bouquet_cost_l1339_133952

theorem bouquet_cost (c₁ : ℕ) (r₁ r₂ : ℕ) (c_discount : ℕ) (discount_percentage: ℕ) :
  (c₁ = 30) → (r₁ = 15) → (r₂ = 45) → (c_discount = 81) → (discount_percentage = 10) → 
  ((c₂ : ℕ) → (c₂ = (c₁ * r₂) / r₁) → (r₂ > 30) → 
  (c_discount = c₂ - (c₂ * discount_percentage / 100))) → 
  c_discount = 81 :=
by
  intros h1 h2 h3 h4 h5
  subst_vars
  sorry

end bouquet_cost_l1339_133952


namespace four_fours_to_seven_l1339_133923

theorem four_fours_to_seven :
  (∃ eq1 eq2 : ℕ, eq1 ≠ eq2 ∧
    (eq1 = 4 + 4 - (4 / 4) ∧
     eq2 = 44 / 4 - 4 ∧ eq1 = 7 ∧ eq2 = 7)) :=
by
  existsi (4 + 4 - (4 / 4))
  existsi (44 / 4 - 4)
  sorry

end four_fours_to_seven_l1339_133923


namespace curve_not_parabola_l1339_133984

theorem curve_not_parabola (k : ℝ) : ¬ ∃ a b c t : ℝ, a * t^2 + b * t + c = x^2 + k * y^2 - 1 := sorry

end curve_not_parabola_l1339_133984


namespace pencil_cost_l1339_133971

theorem pencil_cost (P : ℕ) (h1 : ∀ p : ℕ, p = 80) (h2 : ∀ p_est, ((16 * P) + (20 * 80)) = p_est → p_est = 2000) (h3 : 36 = 16 + 20) :
    P = 25 :=
  sorry

end pencil_cost_l1339_133971


namespace min_time_to_cook_noodles_l1339_133973

/-- 
Li Ming needs to cook noodles, following these steps: 
① Boil the noodles for 4 minutes; 
② Wash vegetables for 5 minutes; 
③ Prepare the noodles and condiments for 2 minutes; 
④ Boil the water in the pot for 10 minutes; 
⑤ Wash the pot and add water for 2 minutes. 
Apart from step ④, only one step can be performed at a time. 
Prove that the minimum number of minutes needed to complete these tasks is 16.
-/
def total_time : Nat :=
  let t5 := 2 -- Wash the pot and add water
  let t4 := 10 -- Boil the water in the pot
  let t2 := 5 -- Wash vegetables
  let t3 := 2 -- Prepare the noodles and condiments
  let t1 := 4 -- Boil the noodles
  t5 + t4.max (t2 + t3) + t1

theorem min_time_to_cook_noodles : total_time = 16 :=
by
  sorry

end min_time_to_cook_noodles_l1339_133973


namespace multiple_properties_l1339_133997

variables (a b : ℤ)

-- Definitions of the conditions
def is_multiple_of_4 (x : ℤ) : Prop := ∃ k : ℤ, x = 4 * k
def is_multiple_of_8 (x : ℤ) : Prop := ∃ k : ℤ, x = 8 * k

-- Problem statement
theorem multiple_properties (h1 : is_multiple_of_4 a) (h2 : is_multiple_of_8 b) :
  is_multiple_of_4 b ∧ is_multiple_of_4 (a + b) ∧ (∃ k : ℤ, a + b = 2 * k) :=
by
  sorry

end multiple_properties_l1339_133997


namespace asymptotes_of_hyperbola_l1339_133963

theorem asymptotes_of_hyperbola (k : ℤ) (h1 : (k - 2016) * (k - 2018) < 0) :
  ∀ x y: ℝ, (x ^ 2) - (y ^ 2) = 1 → ∃ a b: ℝ, y = x * a ∨ y = x * b :=
by
  sorry

end asymptotes_of_hyperbola_l1339_133963


namespace integer_solutions_l1339_133988

theorem integer_solutions (t : ℤ) : 
  ∃ x y : ℤ, 5 * x - 7 * y = 3 ∧ x = 7 * t - 12 ∧ y = 5 * t - 9 :=
by
  sorry

end integer_solutions_l1339_133988


namespace list_price_of_article_l1339_133985

theorem list_price_of_article (P : ℝ) 
  (first_discount second_discount final_price : ℝ)
  (h1 : first_discount = 0.10)
  (h2 : second_discount = 0.08235294117647069)
  (h3 : final_price = 56.16)
  (h4 : P * (1 - first_discount) * (1 - second_discount) = final_price) : P = 68 :=
sorry

end list_price_of_article_l1339_133985


namespace impossible_to_form_11x12x13_parallelepiped_l1339_133934

def is_possible_to_form_parallelepiped
  (brick_shapes_form_unit_cubes : Prop)
  (dimensions : ℕ × ℕ × ℕ) : Prop :=
  ∃ bricks : ℕ, 
    (bricks * 4 = dimensions.fst * dimensions.snd * dimensions.snd.fst)

theorem impossible_to_form_11x12x13_parallelepiped 
  (dimensions := (11, 12, 13)) 
  (brick_shapes_form_unit_cubes : Prop) : 
  ¬ is_possible_to_form_parallelepiped brick_shapes_form_unit_cubes dimensions := 
sorry

end impossible_to_form_11x12x13_parallelepiped_l1339_133934


namespace square_of_number_l1339_133970

theorem square_of_number (x : ℝ) (h : 2 * x = x / 5 + 9) : x^2 = 25 := 
sorry

end square_of_number_l1339_133970


namespace simplify_fraction_l1339_133968

theorem simplify_fraction (x : ℝ) : (2 * x - 3) / 4 + (4 * x + 5) / 3 = (22 * x + 11) / 12 := by
  sorry

end simplify_fraction_l1339_133968


namespace initial_value_l1339_133944

theorem initial_value (x k : ℤ) (h : x + 294 = k * 456) : x = 162 :=
sorry

end initial_value_l1339_133944


namespace hawks_points_l1339_133987

def touchdowns : ℕ := 3
def points_per_touchdown : ℕ := 7
def total_points (t : ℕ) (p : ℕ) : ℕ := t * p

theorem hawks_points : total_points touchdowns points_per_touchdown = 21 :=
by
  -- Proof will go here
  sorry

end hawks_points_l1339_133987


namespace range_of_a_l1339_133976

open Real

noncomputable def f (x : ℝ) : ℝ := abs (log x)

noncomputable def g (x : ℝ) : ℝ := 
  if 0 < x ∧ x ≤ 1 then 0 
  else abs (x^2 - 4) - 2

noncomputable def h (x : ℝ) : ℝ := f x + g x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |h x| = a → has_four_real_roots : Prop) ↔ (1 ≤ a ∧ a < 2 - log 2) := sorry

end range_of_a_l1339_133976


namespace walt_total_interest_l1339_133990

noncomputable def interest_8_percent (P_8 R_8 : ℝ) : ℝ :=
  P_8 * R_8

noncomputable def remaining_amount (P_total P_8 : ℝ) : ℝ :=
  P_total - P_8

noncomputable def interest_9_percent (P_9 R_9 : ℝ) : ℝ :=
  P_9 * R_9

noncomputable def total_interest (I_8 I_9 : ℝ) : ℝ :=
  I_8 + I_9

theorem walt_total_interest :
  let P_8 := 4000
  let R_8 := 0.08
  let P_total := 9000
  let R_9 := 0.09
  let I_8 := interest_8_percent P_8 R_8
  let P_9 := remaining_amount P_total P_8
  let I_9 := interest_9_percent P_9 R_9
  let I_total := total_interest I_8 I_9
  I_total = 770 := 
by
  sorry

end walt_total_interest_l1339_133990


namespace number_of_white_stones_is_3600_l1339_133914

-- Definitions and conditions
def total_stones : ℕ := 6000
def total_difference_to_4800 : ℕ := 4800
def W : ℕ := 3600

-- Conditions
def condition1 (B : ℕ) : Prop := total_stones - W + B = total_difference_to_4800
def condition2 (B : ℕ) : Prop := W + B = total_stones
def condition3 (B : ℕ) : Prop := W > B

-- Theorem statement
theorem number_of_white_stones_is_3600 :
  ∃ B : ℕ, condition1 B ∧ condition2 B ∧ condition3 B :=
by
  -- TODO: Complete the proof
  sorry

end number_of_white_stones_is_3600_l1339_133914


namespace octagon_area_difference_l1339_133958

theorem octagon_area_difference (side_length : ℝ) (h : side_length = 1) : 
  let A := 2 * (1 + Real.sqrt 2)
  let triangle_area := (1 / 2) * (1 / 2) * (1 / 2)
  let gray_area := 4 * triangle_area
  let part_with_lines := A - gray_area
  (gray_area - part_with_lines) = 1 / 4 :=
by
  sorry

end octagon_area_difference_l1339_133958


namespace longest_boat_length_l1339_133904

variable (saved money : ℕ) (license_fee docking_multiplier boat_cost : ℕ)

theorem longest_boat_length (h1 : saved = 20000) 
                           (h2 : license_fee = 500) 
                           (h3 : docking_multiplier = 3)
                           (h4 : boat_cost = 1500) : 
                           (saved - license_fee - docking_multiplier * license_fee) / boat_cost = 12 := 
by 
  sorry

end longest_boat_length_l1339_133904


namespace algebraic_expression_interpretation_l1339_133955

def donations_interpretation (m n : ℝ) : ℝ := 5 * m + 2 * n
def plazas_area_interpretation (a : ℝ) : ℝ := 6 * a^2

theorem algebraic_expression_interpretation (m n a : ℝ) :
  donations_interpretation m n = 5 * m + 2 * n ∧ plazas_area_interpretation a = 6 * a^2 :=
by
  sorry

end algebraic_expression_interpretation_l1339_133955


namespace k_equals_three_fourths_l1339_133920

theorem k_equals_three_fourths : ∀ a b c d : ℝ, a ∈ Set.Ici (-1) → b ∈ Set.Ici (-1) → c ∈ Set.Ici (-1) → d ∈ Set.Ici (-1) →
  a^3 + b^3 + c^3 + d^3 + 1 ≥ (3 / 4) * (a + b + c + d) :=
by
  intros
  sorry

end k_equals_three_fourths_l1339_133920


namespace train_length_l1339_133943

open Real

theorem train_length 
  (v : ℝ) -- speed of the train in km/hr
  (t : ℝ) -- time in seconds
  (d : ℝ) -- length of the bridge in meters
  (h_v : v = 36) -- condition 1
  (h_t : t = 50) -- condition 2
  (h_d : d = 140) -- condition 3
  : (v * 1000 / 3600) * t = 360 + 140 := 
sorry

end train_length_l1339_133943


namespace positive_difference_of_b_l1339_133965

def g (n : Int) : Int :=
  if n < 0 then n^2 + 3 else 2 * n - 25

theorem positive_difference_of_b :
  let s := g (-3) + g 3
  let t b := g b = -s
  ∃ a b, t a ∧ t b ∧ a ≠ b ∧ |a - b| = 18 :=
by
  sorry

end positive_difference_of_b_l1339_133965


namespace John_total_amount_l1339_133941

theorem John_total_amount (x : ℝ)
  (h1 : ∃ x : ℝ, (3 * x * 5 * 3 * x) = 300):
  (x + 3 * x + 15 * x) = 380 := by
  sorry

end John_total_amount_l1339_133941


namespace intersection_of_A_and_B_l1339_133911

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B :
  ∀ x : ℝ, (x ∈ A ∩ B) ↔ (x = 0 ∨ x = 1) := by
  sorry

end intersection_of_A_and_B_l1339_133911


namespace find_percentage_l1339_133936

theorem find_percentage (P : ℕ) (h1 : 0.20 * 650 = 130) (h2 : P * 800 / 100 = 320) : P = 40 := 
by { 
  sorry 
}

end find_percentage_l1339_133936


namespace triangle_ABC_two_solutions_l1339_133969

theorem triangle_ABC_two_solutions (x : ℝ) (h1 : x > 0) : 
  2 < x ∧ x < 2 * Real.sqrt 2 ↔
  (∃ a b B, a = x ∧ b = 2 ∧ B = Real.pi / 4 ∧ a * Real.sin B < b ∧ b < a) := by
  sorry

end triangle_ABC_two_solutions_l1339_133969


namespace fraction_subtraction_l1339_133989

theorem fraction_subtraction (x y : ℝ) (h : x / y = 3 / 2) : (x - y) / y = 1 / 2 := 
by 
  sorry

end fraction_subtraction_l1339_133989


namespace points_lie_on_line_l1339_133994

theorem points_lie_on_line (t : ℝ) (ht : t ≠ 0) :
  let x := (2 * t + 2) / t
  let y := (2 * t - 2) / t
  x + y = 4 :=
by
  let x := (2 * t + 2) / t
  let y := (2 * t - 2) / t
  sorry

end points_lie_on_line_l1339_133994


namespace bhanu_income_percentage_l1339_133956

variable {I P : ℝ}

theorem bhanu_income_percentage (h₁ : 300 = (P / 100) * I)
                                  (h₂ : 210 = 0.3 * (I - 300)) :
  P = 30 :=
by
  sorry

end bhanu_income_percentage_l1339_133956


namespace part_a_part_b_l1339_133949

-- Definition based on conditions
def S (n k : ℕ) : ℕ :=
  -- Placeholder: Actual definition would count the coefficients
  -- of (x+1)^n that are not divisible by k.
  sorry

-- Part (a) proof statement
theorem part_a : S 2012 3 = 324 :=
by sorry

-- Part (b) proof statement
theorem part_b : 2012 ∣ S (2012^2011) 2011 :=
by sorry

end part_a_part_b_l1339_133949


namespace olivia_savings_l1339_133905

noncomputable def compound_amount 
  (P : ℝ) -- Initial principal
  (r : ℝ) -- Annual interest rate
  (n : ℕ) -- Number of times interest is compounded per year
  (t : ℕ) -- Number of years
  : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem olivia_savings :
  compound_amount 2500 0.045 2 21 = 5077.14 :=
by
  sorry

end olivia_savings_l1339_133905


namespace value_of_b_l1339_133935

theorem value_of_b (b : ℝ) (x : ℝ) (h : x = 1) (h_eq : 3 * x^2 - b * x + 3 = 0) : b = 6 :=
by
  sorry

end value_of_b_l1339_133935


namespace no_solution_k_l1339_133932

theorem no_solution_k (k : ℝ) : 
  (∀ t s : ℝ, 
    ∃ (a : ℝ × ℝ) (b : ℝ × ℝ) (c : ℝ × ℝ) (d : ℝ × ℝ), 
      (a = (2, 7)) ∧ 
      (b = (5, -9)) ∧ 
      (c = (4, -3)) ∧ 
      (d = (-2, k)) ∧ 
      (a + t • b ≠ c + s • d)) ↔ k = 18 / 5 := 
by
  sorry

end no_solution_k_l1339_133932


namespace amy_school_year_hours_l1339_133954

noncomputable def summer_hours_per_week := 40
noncomputable def summer_weeks := 8
noncomputable def summer_earnings := 3200
noncomputable def school_year_weeks := 32
noncomputable def school_year_earnings_needed := 4800

theorem amy_school_year_hours
  (H1 : summer_earnings = summer_hours_per_week * summer_weeks * (summer_earnings / (summer_hours_per_week * summer_weeks)))
  (H2 : school_year_earnings_needed = school_year_weeks * (summer_earnings / (summer_hours_per_week * summer_weeks)))
  : (school_year_earnings_needed / school_year_weeks / (summer_earnings / (summer_hours_per_week * summer_weeks))) = 15 :=
by
  sorry

end amy_school_year_hours_l1339_133954


namespace telephone_charges_equal_l1339_133945

theorem telephone_charges_equal (m : ℝ) :
  (9 + 0.25 * m = 12 + 0.20 * m) → m = 60 :=
by
  intro h
  sorry

end telephone_charges_equal_l1339_133945


namespace souvenirs_total_cost_l1339_133972

theorem souvenirs_total_cost (T : ℝ) (H1 : 347 = T + 146) : T + 347 = 548 :=
by
  -- To ensure the validity of the Lean statement but without the proof.
  sorry

end souvenirs_total_cost_l1339_133972


namespace series_sum_correct_l1339_133918

noncomputable def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r ^ n) / (1 - r)

theorem series_sum_correct :
  geometric_series_sum (1 / 2) (-1 / 3) 6 = 91 / 243 :=
by
  -- Proof goes here
  sorry

end series_sum_correct_l1339_133918


namespace largest_4_digit_divisible_by_35_l1339_133998

theorem largest_4_digit_divisible_by_35 : ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 35 = 0) ∧ (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ (m % 35 = 0) → m ≤ n) ∧ n = 9985 := 
by sorry

end largest_4_digit_divisible_by_35_l1339_133998


namespace rectangle_length_fraction_l1339_133975

theorem rectangle_length_fraction 
  (s r : ℝ) 
  (A b ℓ : ℝ)
  (area_square : s * s = 1600)
  (radius_eq_side : r = s)
  (area_rectangle : A = ℓ * b)
  (breadth_rect : b = 10)
  (area_rect_val : A = 160) :
  ℓ / r = 2 / 5 := 
by
  sorry

end rectangle_length_fraction_l1339_133975


namespace solve_system1_solve_system2_l1339_133946

theorem solve_system1 (x y : ℝ) (h1 : y = x - 4) (h2 : x + y = 6) : x = 5 ∧ y = 1 :=
by sorry

theorem solve_system2 (x y : ℝ) (h1 : 2 * x + y = 1) (h2 : 4 * x - y = 5) : x = 1 ∧ y = -1 :=
by sorry

end solve_system1_solve_system2_l1339_133946


namespace packed_oranges_l1339_133967

theorem packed_oranges (oranges_per_box : ℕ) (boxes_used : ℕ) (total_oranges : ℕ) 
  (h1 : oranges_per_box = 10) (h2 : boxes_used = 265) : 
  total_oranges = 2650 :=
by 
  sorry

end packed_oranges_l1339_133967


namespace stacked_lego_volume_l1339_133901

theorem stacked_lego_volume 
  (lego_volume : ℝ)
  (rows columns layers : ℕ)
  (h1 : lego_volume = 1)
  (h2 : rows = 7)
  (h3 : columns = 5)
  (h4 : layers = 3) :
  rows * columns * layers * lego_volume = 105 :=
by
  sorry

end stacked_lego_volume_l1339_133901


namespace train_length_is_correct_l1339_133921

-- Definitions of speeds and time
def speedTrain_kmph := 100
def speedMotorbike_kmph := 64
def overtakingTime_s := 20

-- Calculate speeds in m/s
def speedTrain_mps := speedTrain_kmph * 1000 / 3600
def speedMotorbike_mps := speedMotorbike_kmph * 1000 / 3600

-- Calculate relative speed
def relativeSpeed_mps := speedTrain_mps - speedMotorbike_mps

-- Calculate the length of the train
def length_of_train := relativeSpeed_mps * overtakingTime_s

-- Theorem: Verifying the length of the train is 200 meters
theorem train_length_is_correct : length_of_train = 200 := by
  -- Sorry placeholder for proof
  sorry

end train_length_is_correct_l1339_133921


namespace complex_sum_power_l1339_133910

noncomputable def z : ℂ := sorry

theorem complex_sum_power (hz : z^2 + z + 1 = 0) :
  z^100 + z^101 + z^102 + z^103 + z^104 = -1 :=
sorry

end complex_sum_power_l1339_133910


namespace min_value_expr_l1339_133977

-- Define the given expression
def given_expr (x : ℝ) : ℝ :=
  (15 - x) * (8 - x) * (15 + x) * (8 + x) + 200

-- Define the minimum value we need to prove
def min_value : ℝ :=
  -6290.25

-- The statement of the theorem
theorem min_value_expr :
  ∃ x : ℝ, ∀ y : ℝ, given_expr y ≥ min_value := by
  sorry

end min_value_expr_l1339_133977


namespace problem1_problem2_l1339_133933

noncomputable def f (x : ℝ) : ℝ :=
  if h : 1 ≤ x then x else 1 / x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
  a * f x - |x - 2|

def problem1_statement (b : ℝ) : Prop :=
  ∀ x, x > 0 → g x 0 ≤ |x - 1| + b

def problem2_statement : Prop :=
  ∃ x, (0 < x) ∧ ∀ y, (0 < y) → g y 1 ≥ g x 1

theorem problem1 : ∀ b : ℝ, problem1_statement b ↔ b ∈ Set.Ici (-1) := sorry

theorem problem2 : ∃ x, problem2_statement ∧ g x 1 = 0 := sorry

end problem1_problem2_l1339_133933


namespace number_of_fish_disappeared_l1339_133913

-- First, define initial amounts of each type of fish
def goldfish_initial := 7
def catfish_initial := 12
def guppies_initial := 8
def angelfish_initial := 5

-- Define the total initial number of fish
def total_fish_initial := goldfish_initial + catfish_initial + guppies_initial + angelfish_initial

-- Define the current number of fish
def fish_current := 27

-- Define the number of fish disappeared
def fish_disappeared := total_fish_initial - fish_current

-- Proof statement
theorem number_of_fish_disappeared:
  fish_disappeared = 5 :=
by
  -- Sorry is a placeholder that indicates the proof is omitted.
  sorry

end number_of_fish_disappeared_l1339_133913


namespace n_multiple_of_40_and_infinite_solutions_l1339_133983

theorem n_multiple_of_40_and_infinite_solutions 
  (n : ℤ)
  (h1 : ∃ k₁ : ℤ, 2 * n + 1 = k₁^2)
  (h2 : ∃ k₂ : ℤ, 3 * n + 1 = k₂^2)
  : ∃ (m : ℤ), n = 40 * m ∧ ∃ (seq : ℕ → ℤ), 
    (∀ i : ℕ, ∃ k₁ k₂ : ℤ, (2 * (seq i) + 1 = k₁^2) ∧ (3 * (seq i) + 1 = k₂^2) ∧ 
     (i ≠ 0 → seq i ≠ seq (i - 1))) :=
by sorry

end n_multiple_of_40_and_infinite_solutions_l1339_133983


namespace ratio_of_earnings_l1339_133919

theorem ratio_of_earnings (jacob_hourly: ℕ) (jake_total: ℕ) (days: ℕ) (hours_per_day: ℕ) (jake_hourly: ℕ) (ratio: ℕ) 
  (h_jacob: jacob_hourly = 6)
  (h_jake_total: jake_total = 720)
  (h_days: days = 5)
  (h_hours_per_day: hours_per_day = 8)
  (h_jake_hourly: jake_hourly = jake_total / (days * hours_per_day))
  (h_ratio: ratio = jake_hourly / jacob_hourly) :
  ratio = 3 := 
sorry

end ratio_of_earnings_l1339_133919


namespace seating_arrangement_fixed_pairs_l1339_133957

theorem seating_arrangement_fixed_pairs 
  (total_chairs : ℕ) 
  (total_people : ℕ) 
  (specific_pair_adjacent : Prop)
  (comb : ℕ) 
  (four_factorial : ℕ) 
  (two_factorial : ℕ) 
  : total_chairs = 6 → total_people = 5 → specific_pair_adjacent → comb = Nat.choose 6 4 → 
    four_factorial = Nat.factorial 4 → two_factorial = Nat.factorial 2 → 
    Nat.choose 6 4 * Nat.factorial 4 * Nat.factorial 2 = 720 
  := by
  intros
  sorry

end seating_arrangement_fixed_pairs_l1339_133957


namespace ceil_and_floor_difference_l1339_133951

theorem ceil_and_floor_difference (x : ℝ) (ε : ℝ) 
  (h_cond : ⌈x + ε⌉ - ⌊x + ε⌋ = 1) (h_eps : 0 < ε ∧ ε < 1) :
  ⌈x + ε⌉ - (x + ε) = 1 - ε :=
sorry

end ceil_and_floor_difference_l1339_133951


namespace sum_of_coordinates_of_reflected_points_l1339_133929

theorem sum_of_coordinates_of_reflected_points (C D : ℝ × ℝ) (hx : C.1 = 3) (hy : C.2 = 8) (hD : D = (-C.1, C.2)) :
  C.1 + C.2 + D.1 + D.2 = 16 := by
  sorry

end sum_of_coordinates_of_reflected_points_l1339_133929


namespace common_ratio_of_geometric_sequence_l1339_133996

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

theorem common_ratio_of_geometric_sequence
  (a1 d : ℝ) (h1 : d ≠ 0)
  (h2 : (a_n a1 d 5) * (a_n a1 d 20) = (a_n a1 d 10) ^ 2) :
  (a_n a1 d 10) / (a_n a1 d 5) = 2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l1339_133996


namespace greatest_distance_between_vertices_l1339_133962

theorem greatest_distance_between_vertices 
    (inner_perimeter outer_perimeter : ℝ) 
    (inner_square_perimeter_eq : inner_perimeter = 16)
    (outer_square_perimeter_eq : outer_perimeter = 40)
    : ∃ max_distance, max_distance = 2 * Real.sqrt 34 :=
by
  sorry

end greatest_distance_between_vertices_l1339_133962


namespace ratio_meerkats_to_lion_cubs_l1339_133980

-- Defining the initial conditions 
def initial_animals : ℕ := 68
def gorillas_sent : ℕ := 6
def hippo_adopted : ℕ := 1
def rhinos_rescued : ℕ := 3
def lion_cubs : ℕ := 8
def final_animal_count : ℕ := 90

-- Calculating the number of meerkats
def animals_before_meerkats : ℕ := initial_animals - gorillas_sent + hippo_adopted + rhinos_rescued + lion_cubs
def meerkats : ℕ := final_animal_count - animals_before_meerkats

-- Proving the ratio of meerkats to lion cubs is 2:1
theorem ratio_meerkats_to_lion_cubs : meerkats / lion_cubs = 2 := by
  -- Placeholder for the proof
  sorry

end ratio_meerkats_to_lion_cubs_l1339_133980
