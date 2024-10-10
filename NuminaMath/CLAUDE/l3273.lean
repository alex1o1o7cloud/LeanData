import Mathlib

namespace descent_time_calculation_l3273_327397

theorem descent_time_calculation (climb_time : ℝ) (avg_speed_total : ℝ) (avg_speed_climb : ℝ) :
  climb_time = 4 →
  avg_speed_total = 2 →
  avg_speed_climb = 1.5 →
  ∃ (descent_time : ℝ),
    descent_time = 2 ∧
    avg_speed_total = (2 * avg_speed_climb * climb_time) / (climb_time + descent_time) :=
by sorry

end descent_time_calculation_l3273_327397


namespace fraction_of_a_equal_to_quarter_of_b_l3273_327387

theorem fraction_of_a_equal_to_quarter_of_b : ∀ (a b x : ℚ), 
  a + b = 1210 →
  b = 484 →
  x * a = (1/4) * b →
  x = 1/6 := by
sorry

end fraction_of_a_equal_to_quarter_of_b_l3273_327387


namespace trapezoid_circle_area_ratio_l3273_327358

/-- Given a trapezoid inscribed in a circle, where the larger base forms an angle α 
    with a lateral side and an angle β with the diagonal, the ratio of the area of 
    the circle to the area of the trapezoid is π / (2 sin²α sin(2β)). -/
theorem trapezoid_circle_area_ratio (α β : Real) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) : 
  ∃ (S_circle S_trapezoid : Real),
    S_circle > 0 ∧ S_trapezoid > 0 ∧
    S_circle / S_trapezoid = π / (2 * Real.sin α ^ 2 * Real.sin (2 * β)) :=
by sorry

end trapezoid_circle_area_ratio_l3273_327358


namespace arm_wrestling_tournament_rounds_l3273_327348

/-- Represents the rules and structure of the arm wrestling tournament -/
structure Tournament :=
  (num_athletes : Nat)
  (point_diff_limit : Nat)
  (extra_point_rule : Bool)

/-- Calculates the minimum number of rounds required for a tournament -/
def min_rounds (t : Tournament) : Nat :=
  sorry

/-- The main theorem stating that a tournament with 510 athletes requires at least 9 rounds -/
theorem arm_wrestling_tournament_rounds :
  ∀ (t : Tournament),
    t.num_athletes = 510 ∧
    t.point_diff_limit = 1 ∧
    t.extra_point_rule = true →
    min_rounds t ≥ 9 :=
by sorry

end arm_wrestling_tournament_rounds_l3273_327348


namespace expression_evaluation_l3273_327333

theorem expression_evaluation : 
  let b : ℚ := 4/3
  (6 * b^2 - 8 * b + 3) * (3 * b - 4) = 0 := by
sorry

end expression_evaluation_l3273_327333


namespace linear_function_not_in_quadrant_II_l3273_327353

/-- A linear function in the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- The four quadrants in a Cartesian coordinate system -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Determines if a point (x, y) is in a given quadrant -/
def inQuadrant (x y : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.I => x > 0 ∧ y > 0
  | Quadrant.II => x < 0 ∧ y > 0
  | Quadrant.III => x < 0 ∧ y < 0
  | Quadrant.IV => x > 0 ∧ y < 0

/-- Determines if a linear function passes through a given quadrant -/
def passesThrough (f : LinearFunction) (q : Quadrant) : Prop :=
  ∃ x y : ℝ, y = f.m * x + f.b ∧ inQuadrant x y q

/-- The main theorem: y = 2x - 3 does not pass through Quadrant II -/
theorem linear_function_not_in_quadrant_II :
  ¬ passesThrough { m := 2, b := -3 } Quadrant.II :=
  sorry

end linear_function_not_in_quadrant_II_l3273_327353


namespace relationship_abc_l3273_327347

theorem relationship_abc : 
  let a : ℝ := 1 + Real.sqrt 7
  let b : ℝ := Real.sqrt 3 + Real.sqrt 5
  let c : ℝ := 4
  c > b ∧ b > a := by sorry

end relationship_abc_l3273_327347


namespace square_difference_equality_l3273_327380

theorem square_difference_equality : (23 + 15)^2 - 3 * (23 - 15)^2 = 1252 := by sorry

end square_difference_equality_l3273_327380


namespace set_operations_l3273_327359

def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def B : Set ℕ := {4, 7, 8, 9}

theorem set_operations :
  (A ∪ B = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (A ∩ B = {4, 7, 8}) := by
  sorry

end set_operations_l3273_327359


namespace pencil_length_l3273_327313

theorem pencil_length : ∀ (L : ℝ),
  (L / 8 : ℝ) +  -- Black part
  ((7 * L / 8) / 2 : ℝ) +  -- White part
  (7 / 2 : ℝ) = L →  -- Blue part
  L = 16 := by
sorry

end pencil_length_l3273_327313


namespace scatter_plot_regression_role_l3273_327340

/-- The role of a scatter plot in regression analysis -/
def scatter_plot_role : String :=
  "to roughly judge whether variables are linearly related"

/-- The main theorem about the role of scatter plots in regression analysis -/
theorem scatter_plot_regression_role :
  scatter_plot_role = "to roughly judge whether variables are linearly related" := by
  sorry

end scatter_plot_regression_role_l3273_327340


namespace first_nonzero_digit_of_one_over_137_l3273_327393

theorem first_nonzero_digit_of_one_over_137 :
  ∃ (n : ℕ) (k : ℕ), 
    (1000 : ℚ) / 137 = 7 + (n : ℚ) / (10 ^ k) ∧ 
    0 < n ∧ 
    n < 10 ^ k ∧ 
    n % 10 = 7 :=
by sorry

end first_nonzero_digit_of_one_over_137_l3273_327393


namespace exponential_inequality_l3273_327303

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  3^a + 2*a = 3^b + 3*b → a > b :=
by sorry

end exponential_inequality_l3273_327303


namespace domain_of_f_l3273_327316

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (x + 5)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -5 ∨ x > -5} := by sorry

end domain_of_f_l3273_327316


namespace problem_statement_l3273_327346

theorem problem_statement (a b c : ℕ+) 
  (h : (18 ^ a.val) * (9 ^ (3 * a.val - 1)) * (c ^ (2 * a.val - 3)) = (2 ^ 7) * (3 ^ b.val)) :
  a = 7 := by
  sorry

end problem_statement_l3273_327346


namespace not_perfect_square_l3273_327368

theorem not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, 7 * n + 3 = k ^ 2 := by
  sorry

end not_perfect_square_l3273_327368


namespace opposite_sum_and_sum_opposite_l3273_327389

theorem opposite_sum_and_sum_opposite (a b : ℤ) (h1 : a = -6) (h2 : b = 4) : 
  (-a) + (-b) = 2 ∧ -(a + b) = 2 :=
by sorry

end opposite_sum_and_sum_opposite_l3273_327389


namespace ralph_sock_purchase_l3273_327327

/-- Represents the number of pairs of socks at each price point -/
structure SockPurchase where
  two_dollar : ℕ
  three_dollar : ℕ
  five_dollar : ℕ

/-- Checks if the purchase satisfies the given conditions -/
def is_valid_purchase (p : SockPurchase) : Prop :=
  p.two_dollar + p.three_dollar + p.five_dollar = 15 ∧
  2 * p.two_dollar + 3 * p.three_dollar + 5 * p.five_dollar = 36 ∧
  p.two_dollar ≥ 1 ∧ p.three_dollar ≥ 1 ∧ p.five_dollar ≥ 1

/-- The theorem to be proved -/
theorem ralph_sock_purchase :
  ∃ (p : SockPurchase), is_valid_purchase p ∧ p.two_dollar = 11 :=
sorry

end ralph_sock_purchase_l3273_327327


namespace quadratic_inequality_range_l3273_327330

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (x < 1 ∨ x > 5) → x^2 - 2*(a-2)*x + a > 0) → 
  a ∈ Set.Ioo 1 5 ∪ Set.singleton 5 :=
sorry

end quadratic_inequality_range_l3273_327330


namespace a1_lt_a3_neither_sufficient_nor_necessary_for_a2_lt_a4_l3273_327341

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q^(n-1)

theorem a1_lt_a3_neither_sufficient_nor_necessary_for_a2_lt_a4 :
  ∃ (a q : ℝ), 
    let seq := geometric_sequence a q
    (seq 1 < seq 3 ∧ seq 2 ≥ seq 4) ∧
    (seq 2 < seq 4 ∧ seq 1 ≥ seq 3) :=
sorry

end a1_lt_a3_neither_sufficient_nor_necessary_for_a2_lt_a4_l3273_327341


namespace simple_interest_rate_for_doubling_l3273_327320

/-- Given a sum of money that doubles itself in 5 years at simple interest,
    prove that the rate percent per annum is 20%. -/
theorem simple_interest_rate_for_doubling (P : ℝ) (h : P > 0) : 
  ∃ (R : ℝ), R > 0 ∧ R ≤ 100 ∧ P + (P * R * 5) / 100 = 2 * P ∧ R = 20 := by
  sorry

end simple_interest_rate_for_doubling_l3273_327320


namespace four_three_seating_chart_l3273_327391

/-- Represents a seating chart configuration -/
structure SeatingChart where
  columns : ℕ
  rows : ℕ

/-- Interprets a pair of natural numbers as a seating chart -/
def interpret (pair : ℕ × ℕ) : SeatingChart :=
  { columns := pair.1, rows := pair.2 }

/-- States that (4,3) represents 4 columns and 3 rows -/
theorem four_three_seating_chart :
  let chart := interpret (4, 3)
  chart.columns = 4 ∧ chart.rows = 3 := by
  sorry

end four_three_seating_chart_l3273_327391


namespace mans_speed_with_stream_l3273_327382

/-- 
Given a man's rate (speed in still water) and his speed against the stream,
this theorem proves his speed with the stream.
-/
theorem mans_speed_with_stream 
  (rate : ℝ) 
  (speed_against : ℝ) 
  (h1 : rate = 2) 
  (h2 : speed_against = 6) : 
  rate + (speed_against - rate) = 6 := by
sorry

end mans_speed_with_stream_l3273_327382


namespace common_prime_root_quadratics_l3273_327344

theorem common_prime_root_quadratics (a b : ℤ) : 
  (∃ p : ℕ, Prime p ∧ 
    (p : ℤ)^2 + a * p + b = 0 ∧ 
    (p : ℤ)^2 + b * p + 1100 = 0) →
  a = 274 ∨ a = 40 :=
by sorry

end common_prime_root_quadratics_l3273_327344


namespace hockey_games_played_total_games_played_l3273_327342

/-- Calculates the total number of hockey games played in a season -/
theorem hockey_games_played 
  (season_duration : ℕ) 
  (games_per_month : ℕ) 
  (cancelled_games : ℕ) 
  (postponed_games : ℕ) : ℕ :=
  season_duration * games_per_month - cancelled_games

/-- Proves that the total number of hockey games played is 172 -/
theorem total_games_played : 
  hockey_games_played 14 13 10 5 = 172 := by
  sorry

end hockey_games_played_total_games_played_l3273_327342


namespace prize_probabilities_l3273_327309

/-- Represents the number of balls of each color in a box -/
structure BallBox where
  red : Nat
  white : Nat

/-- Calculates the probability of drawing a specific number of red balls from two boxes -/
def probability_draw_red (box_a box_b : BallBox) (red_count : Nat) : Rat :=
  sorry

/-- The first box containing 4 red balls and 6 white balls -/
def box_a : BallBox := { red := 4, white := 6 }

/-- The second box containing 5 red balls and 5 white balls -/
def box_b : BallBox := { red := 5, white := 5 }

theorem prize_probabilities :
  probability_draw_red box_a box_b 4 = 4 / 135 ∧
  probability_draw_red box_a box_b 3 = 26 / 135 ∧
  (1 - probability_draw_red box_a box_b 0) = 75 / 81 :=
sorry

end prize_probabilities_l3273_327309


namespace closest_integer_to_cube_root_l3273_327384

theorem closest_integer_to_cube_root : 
  ∃ (n : ℤ), n = 10 ∧ ∀ (m : ℤ), |n - (7^3 + 9^3)^(1/3)| ≤ |m - (7^3 + 9^3)^(1/3)| :=
by sorry

end closest_integer_to_cube_root_l3273_327384


namespace half_percent_of_150_in_paise_l3273_327360

/-- Converts rupees to paise -/
def rupees_to_paise (r : ℚ) : ℚ := 100 * r

/-- Calculates the percentage of a given value -/
def percentage_of (p : ℚ) (v : ℚ) : ℚ := (p / 100) * v

theorem half_percent_of_150_in_paise : 
  rupees_to_paise (percentage_of 0.5 150) = 75 := by
  sorry

end half_percent_of_150_in_paise_l3273_327360


namespace gcd_36745_59858_l3273_327383

theorem gcd_36745_59858 : Nat.gcd 36745 59858 = 7 := by
  sorry

end gcd_36745_59858_l3273_327383


namespace sum_and_reciprocal_value_l3273_327370

theorem sum_and_reciprocal_value (x : ℝ) (h1 : x ≠ 0) (h2 : x^2 + (1/x)^2 = 23) : 
  x + (1/x) = 5 := by
  sorry

end sum_and_reciprocal_value_l3273_327370


namespace coupon_problem_l3273_327351

/-- Calculates the total number of bottles that would be received if no coupons were lost -/
def total_bottles (bottles_per_coupon : ℕ) (lost_coupons : ℕ) (remaining_coupons : ℕ) : ℕ :=
  (remaining_coupons + lost_coupons) * bottles_per_coupon

/-- Proves that given the conditions, the total number of bottles would be 21 -/
theorem coupon_problem :
  let bottles_per_coupon : ℕ := 3
  let lost_coupons : ℕ := 3
  let remaining_coupons : ℕ := 4
  total_bottles bottles_per_coupon lost_coupons remaining_coupons = 21 := by
  sorry

end coupon_problem_l3273_327351


namespace smallest_slope_tangent_line_l3273_327362

/-- The equation of the curve -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

/-- The derivative of the curve -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

theorem smallest_slope_tangent_line :
  ∃ (x₀ y₀ : ℝ), 
    (∀ x : ℝ, f' x₀ ≤ f' x) ∧ 
    y₀ = f x₀ ∧
    (3 : ℝ) * x - y - 11 = 0 :=
sorry

end smallest_slope_tangent_line_l3273_327362


namespace constant_function_from_surjective_injective_l3273_327337

theorem constant_function_from_surjective_injective
  (f g h : ℕ → ℕ)
  (h_injective : Function.Injective h)
  (g_surjective : Function.Surjective g)
  (f_def : ∀ n, f n = g n - h n + 1) :
  ∀ n, f n = 1 := by
sorry

end constant_function_from_surjective_injective_l3273_327337


namespace max_value_theorem_l3273_327321

theorem max_value_theorem (a b c : ℝ) 
  (ha : -1 ≤ a ∧ a ≤ 1) 
  (hb : -1 ≤ b ∧ b ≤ 1) 
  (hc : -1 ≤ c ∧ c ≤ 1) : 
  Real.sqrt (a^2 * b^2 * c^2) + Real.sqrt ((1 - a^2) * (1 - b^2) * (1 - c^2)) ≤ 1 ∧ 
  ∃ (x y z : ℝ), -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1 ∧ -1 ≤ z ∧ z ≤ 1 ∧ 
    Real.sqrt (x^2 * y^2 * z^2) + Real.sqrt ((1 - x^2) * (1 - y^2) * (1 - z^2)) = 1 :=
by sorry

end max_value_theorem_l3273_327321


namespace three_integer_chords_l3273_327365

/-- Represents a circle with a given radius and a point at a given distance from its center -/
structure CircleWithPoint where
  radius : ℝ
  distanceFromCenter : ℝ

/-- Counts the number of integer-length chords containing the given point -/
def countIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

theorem three_integer_chords :
  let c := CircleWithPoint.mk 13 5
  countIntegerChords c = 3 := by
  sorry

end three_integer_chords_l3273_327365


namespace circle_radius_l3273_327371

theorem circle_radius (A : ℝ) (h : A = 81 * Real.pi) : 
  ∃ r : ℝ, r > 0 ∧ A = Real.pi * r^2 ∧ r = 9 := by
  sorry

end circle_radius_l3273_327371


namespace average_weight_increase_l3273_327399

/-- Proves that replacing a person weighing 47 kg with a person weighing 68 kg in a group of 6 people increases the average weight by 3.5 kg -/
theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 6 →
  old_weight = 47 →
  new_weight = 68 →
  (new_weight - old_weight) / initial_count = 3.5 := by
  sorry

end average_weight_increase_l3273_327399


namespace line_tangent_to_ellipse_l3273_327395

theorem line_tangent_to_ellipse (m : ℝ) : 
  (∀ x y : ℝ, y = m * x + 1 ∧ x^2 + 4 * y^2 = 1 → 
    ∀ x' y' : ℝ, y' = m * x' + 1 ∧ x'^2 + 4 * y'^2 = 1 → x = x' ∧ y = y') →
  m^2 = 3/4 := by
sorry

end line_tangent_to_ellipse_l3273_327395


namespace red_crayons_count_l3273_327363

/-- Represents the number of crayons of each color in a crayon box. -/
structure CrayonBox where
  total : ℕ
  blue : ℕ
  green : ℕ
  pink : ℕ
  red : ℕ

/-- Calculates the number of red crayons in a crayon box. -/
def redCrayons (box : CrayonBox) : ℕ :=
  box.total - (box.blue + box.green + box.pink)

/-- Theorem stating the number of red crayons in the given crayon box. -/
theorem red_crayons_count (box : CrayonBox) 
  (h1 : box.total = 24)
  (h2 : box.blue = 6)
  (h3 : box.green = 2 * box.blue / 3)
  (h4 : box.pink = 6) :
  redCrayons box = 8 := by
  sorry

#eval redCrayons { total := 24, blue := 6, green := 4, pink := 6, red := 8 }

end red_crayons_count_l3273_327363


namespace airplane_flight_problem_l3273_327385

/-- Airplane flight problem -/
theorem airplane_flight_problem 
  (wind_speed : ℝ) 
  (time_with_wind : ℝ) 
  (time_against_wind : ℝ) 
  (h1 : wind_speed = 24)
  (h2 : time_with_wind = 2.8)
  (h3 : time_against_wind = 3) :
  ∃ (airplane_speed : ℝ) (distance : ℝ),
    airplane_speed = 696 ∧ 
    distance = 2016 ∧
    time_with_wind * (airplane_speed + wind_speed) = distance ∧
    time_against_wind * (airplane_speed - wind_speed) = distance :=
by
  sorry


end airplane_flight_problem_l3273_327385


namespace triangle_area_l3273_327392

/-- The area of a triangle with base 3 meters and height 4 meters is 6 square meters. -/
theorem triangle_area : 
  let base : ℝ := 3
  let height : ℝ := 4
  let area : ℝ := (base * height) / 2
  area = 6 := by sorry

end triangle_area_l3273_327392


namespace diophantine_equation_7z_squared_l3273_327364

theorem diophantine_equation_7z_squared (x y z : ℕ) : 
  x^2 + y^2 = 7 * z^2 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end diophantine_equation_7z_squared_l3273_327364


namespace max_value_2q_minus_r_l3273_327378

theorem max_value_2q_minus_r :
  ∀ q r : ℕ+, 
  965 = 22 * q + r → 
  ∀ q' r' : ℕ+, 
  965 = 22 * q' + r' → 
  2 * q - r ≤ 67 :=
by sorry

end max_value_2q_minus_r_l3273_327378


namespace solution_set_for_negative_one_range_of_a_for_subset_condition_l3273_327315

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |2*x - 1|

theorem solution_set_for_negative_one :
  {x : ℝ | f (-1) x ≤ 2} = {x : ℝ | x = 1/2 ∨ x = -1/2} := by sorry

theorem range_of_a_for_subset_condition :
  ∀ a : ℝ, (∀ x ∈ Set.Icc (1/2) 1, f a x ≤ |2*x + 1|) → a ∈ Set.Icc 0 3 := by sorry

end solution_set_for_negative_one_range_of_a_for_subset_condition_l3273_327315


namespace complex_number_problem_l3273_327323

theorem complex_number_problem (z₁ z₂ : ℂ) : 
  (z₁ - 2) * (1 + Complex.I) = 1 - Complex.I →
  z₂.im = 2 →
  (z₁ * z₂).im = 0 →
  z₂ = 4 + 2 * Complex.I :=
by sorry

end complex_number_problem_l3273_327323


namespace power_of_two_divisibility_l3273_327338

theorem power_of_two_divisibility (n : ℕ) (hn : n ≥ 1) :
  (∃ k : ℕ, 2^n - 1 = 3 * k) ∧
  (∃ m : ℕ, m ≥ 1 ∧ ∃ l : ℕ, (2^n - 1) / 3 * l = 4 * m^2 + 1) →
  ∃ r : ℕ, n = 2^r :=
by sorry

end power_of_two_divisibility_l3273_327338


namespace bridge_length_l3273_327377

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 130 ∧ 
  train_speed_kmh = 54 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 320 := by
  sorry

end bridge_length_l3273_327377


namespace problem_solution_l3273_327322

theorem problem_solution (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a^2 / b = 2) (h2 : b^2 / c = 3) (h3 : c^2 / a = 4) :
  a = 576^(1/7) := by
sorry

end problem_solution_l3273_327322


namespace nut_mixture_ratio_l3273_327308

/-- Given a mixture of nuts where the ratio of almonds to walnuts is x:2 by weight,
    and there are 200 pounds of almonds in 280 pounds of the mixture,
    prove that the ratio of almonds to walnuts is 2.5:1. -/
theorem nut_mixture_ratio (x : ℝ) : 
  x / 2 = 200 / 80 → x / 2 = 2.5 := by sorry

end nut_mixture_ratio_l3273_327308


namespace company_shares_l3273_327396

theorem company_shares (p v s i : Real) : 
  p + v + s + i = 1 → 
  2*p + v + s + i = 1.3 →
  p + 2*v + s + i = 1.4 →
  p + v + 3*s + i = 1.2 →
  ∃ k : Real, k > 3.75 ∧ k * i > 0.75 := by sorry

end company_shares_l3273_327396


namespace line_through_intersection_parallel_to_given_l3273_327328

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := 2 * x - 3 * y + 2 = 0
def l2 (x y : ℝ) : Prop := 3 * x - 4 * y + 2 = 0

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 4 * x + y - 4 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := l1 x y ∧ l2 x y

-- Define the resulting line
def result_line (x y : ℝ) : Prop := 4 * x + y - 10 = 0

-- Theorem statement
theorem line_through_intersection_parallel_to_given :
  ∃ (x₀ y₀ : ℝ), intersection_point x₀ y₀ ∧
  ∀ (x y : ℝ), (∃ (k : ℝ), y - y₀ = k * (x - x₀) ∧ 
                parallel_line (x₀ + 1) (y₀ + k)) ↔
               result_line x y := by sorry

end line_through_intersection_parallel_to_given_l3273_327328


namespace vector_operation_l3273_327381

/-- Given two vectors in ℝ², prove that their specific linear combination equals a certain vector. -/
theorem vector_operation (a b : ℝ × ℝ) : 
  a = (3, 5) → b = (-2, 1) → a - 2 • b = (7, 3) := by sorry

end vector_operation_l3273_327381


namespace integer_operation_proof_l3273_327361

theorem integer_operation_proof (n : ℤ) : 5 * (n - 2) = 85 → n = 19 := by
  sorry

end integer_operation_proof_l3273_327361


namespace simplify_expressions_l3273_327307

theorem simplify_expressions (x y a b : ℝ) :
  ((-3 * x + y) + (4 * x - 3 * y) = x - 2 * y) ∧
  (2 * a - (3 * b - 5 * a - (2 * a - 7 * b)) = 9 * a - 10 * b) := by
  sorry

end simplify_expressions_l3273_327307


namespace hall_length_l3273_327312

theorem hall_length (hall_breadth : ℝ) (stone_length stone_width : ℝ) (num_stones : ℕ) :
  hall_breadth = 15 →
  stone_length = 0.3 →
  stone_width = 0.5 →
  num_stones = 3600 →
  (hall_breadth * (num_stones * stone_length * stone_width / hall_breadth)) = 36 := by
  sorry

end hall_length_l3273_327312


namespace ben_winning_strategy_l3273_327301

/-- Represents the state of the chocolate bar game -/
structure ChocolateBar where
  m : ℕ
  n : ℕ

/-- Determines if a player has a winning strategy given the current state of the game -/
def has_winning_strategy (state : ChocolateBar) : Prop :=
  ∃ (k : ℕ), (state.m + 1) = 2^k * (state.n + 1) ∨
             (state.n + 1) = 2^k * (state.m + 1) ∨
             ∃ (x : ℕ), x ≤ state.m ∧
                        ((state.m + 1 - x) = 2^k * (state.n + 1) ∨
                         (state.n + 1) = 2^k * (state.m + 1 - x))

/-- Theorem: Ben has a winning strategy if and only if the ratio can be made a power of two -/
theorem ben_winning_strategy (initial_state : ChocolateBar) :
  has_winning_strategy initial_state ↔
  ∃ (a k : ℕ), a ≥ 2 ∧ k ≥ 0 ∧
    ((initial_state.m = a - 1 ∧ initial_state.n = 2^k * a - 1) ∨
     (initial_state.m = 2^k * a - 1 ∧ initial_state.n = a - 1)) :=
sorry

end ben_winning_strategy_l3273_327301


namespace dividend_calculation_l3273_327300

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h_divisor : divisor = 15)
  (h_quotient : quotient = 8)
  (h_remainder : remainder = 5) :
  divisor * quotient + remainder = 125 := by
  sorry

end dividend_calculation_l3273_327300


namespace slowest_racer_time_l3273_327339

/-- Represents the time taken by each person to reach the top floor -/
structure RaceTime where
  lola : ℕ
  tara : ℕ
  sam : ℕ

/-- Calculates the race times given the building parameters -/
def calculateRaceTimes (
  totalStories : ℕ
  ) (lolaTimePerStory : ℕ
  ) (samTimePerStory : ℕ
  ) (elevatorTimePerStory : ℕ
  ) (elevatorStopTime : ℕ
  ) (samSwitchFloor : ℕ
  ) (elevatorWaitTime : ℕ
  ) : RaceTime :=
  { lola := totalStories * lolaTimePerStory,
    tara := totalStories * elevatorTimePerStory + (totalStories - 1) * elevatorStopTime,
    sam := samSwitchFloor * samTimePerStory + elevatorWaitTime +
           (totalStories - samSwitchFloor) * elevatorTimePerStory +
           (totalStories - samSwitchFloor - 1) * elevatorStopTime }

/-- The main theorem to prove -/
theorem slowest_racer_time (
  totalStories : ℕ
  ) (lolaTimePerStory : ℕ
  ) (samTimePerStory : ℕ
  ) (elevatorTimePerStory : ℕ
  ) (elevatorStopTime : ℕ
  ) (samSwitchFloor : ℕ
  ) (elevatorWaitTime : ℕ
  ) (h1 : totalStories = 50
  ) (h2 : lolaTimePerStory = 12
  ) (h3 : samTimePerStory = 15
  ) (h4 : elevatorTimePerStory = 10
  ) (h5 : elevatorStopTime = 4
  ) (h6 : samSwitchFloor = 25
  ) (h7 : elevatorWaitTime = 20
  ) : (
    let times := calculateRaceTimes totalStories lolaTimePerStory samTimePerStory
                   elevatorTimePerStory elevatorStopTime samSwitchFloor elevatorWaitTime
    max times.lola (max times.tara times.sam) = 741
  ) := by
  sorry

end slowest_racer_time_l3273_327339


namespace rice_containers_l3273_327354

theorem rice_containers (total_weight : ℚ) (container_capacity : ℕ) (pound_to_ounce : ℕ) : 
  total_weight = 25 / 2 →
  container_capacity = 50 →
  pound_to_ounce = 16 →
  (total_weight * pound_to_ounce : ℚ) / container_capacity = 4 := by
  sorry

end rice_containers_l3273_327354


namespace watermelon_size_ratio_l3273_327302

/-- Given information about watermelons grown by Michael, Clay, and John, 
    prove the ratio of Clay's watermelon size to Michael's watermelon size. -/
theorem watermelon_size_ratio 
  (michael_weight : ℝ) 
  (john_weight : ℝ) 
  (h1 : michael_weight = 8)
  (h2 : john_weight = 12)
  (h3 : ∃ (clay_weight : ℝ), clay_weight = 2 * john_weight) :
  ∃ (clay_weight : ℝ), clay_weight / michael_weight = 3 := by
  sorry

end watermelon_size_ratio_l3273_327302


namespace students_answering_yes_for_R_l3273_327317

theorem students_answering_yes_for_R (total : ℕ) (only_M : ℕ) (neither : ℕ) (h1 : total = 800) (h2 : only_M = 150) (h3 : neither = 250) : 
  ∃ R : ℕ, R = 400 ∧ R = total - neither - only_M :=
by sorry

end students_answering_yes_for_R_l3273_327317


namespace chocolate_chip_cookie_batches_l3273_327366

/-- Given:
  - Each batch of chocolate chip cookies contains 3 cookies.
  - There are 4 oatmeal cookies.
  - The total number of cookies is 10.
Prove that the number of batches of chocolate chip cookies is 2. -/
theorem chocolate_chip_cookie_batches :
  ∀ (batch_size : ℕ) (oatmeal_cookies : ℕ) (total_cookies : ℕ),
    batch_size = 3 →
    oatmeal_cookies = 4 →
    total_cookies = 10 →
    (total_cookies - oatmeal_cookies) / batch_size = 2 :=
by sorry

end chocolate_chip_cookie_batches_l3273_327366


namespace length_MN_l3273_327386

/-- The length of MN where M and N are points on two lines and S is their midpoint -/
theorem length_MN (M N S : ℝ × ℝ) : 
  S = (10, 8) →
  (∃ x₁, M = (x₁, 14 * x₁ / 9)) →
  (∃ x₂, N = (x₂, 5 * x₂ / 12)) →
  S.1 = (M.1 + N.1) / 2 →
  S.2 = (M.2 + N.2) / 2 →
  ∃ length, length = Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) :=
by sorry


end length_MN_l3273_327386


namespace intersection_M_N_l3273_327357

def M : Set ℝ := {y | ∃ x, y = |Real.cos x ^ 2 - Real.sin x ^ 2|}

def N : Set ℝ := {x | ∃ y, y = Real.log (1 - x^2)}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by sorry

end intersection_M_N_l3273_327357


namespace michelle_boutique_two_ties_probability_l3273_327335

/-- Represents the probability of selecting 2 ties from a boutique with given items. -/
def probability_two_ties (shirts pants ties : ℕ) : ℚ :=
  let total := shirts + pants + ties
  (ties : ℚ) / total * ((ties - 1) : ℚ) / (total - 1)

/-- Theorem stating the probability of selecting 2 ties from Michelle's boutique. -/
theorem michelle_boutique_two_ties_probability : 
  probability_two_ties 4 8 18 = 51 / 145 := by
  sorry

end michelle_boutique_two_ties_probability_l3273_327335


namespace quadratic_roots_range_l3273_327331

theorem quadratic_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁ > 2 ∧ x₂ > 2 ∧ 
   x₁^2 + (k-2)*x₁ + 5 - k = 0 ∧ 
   x₂^2 + (k-2)*x₂ + 5 - k = 0) → 
  -5 < k ∧ k < -4 :=
by sorry

end quadratic_roots_range_l3273_327331


namespace additive_function_properties_l3273_327304

/-- A function f: ℝ → ℝ satisfying f(x+y) = f(x) + f(y) for all x, y ∈ ℝ -/
def AdditiveFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

theorem additive_function_properties (f : ℝ → ℝ) (hf : AdditiveFunction f) :
  (f 0 = 0) ∧ (∀ x : ℝ, f (-x) = -f x) := by sorry

end additive_function_properties_l3273_327304


namespace max_c_value_l3273_327398

theorem max_c_value (a b : ℝ) (h : a + 2*b = 2) : 
  ∃ c_max : ℝ, c_max = 3 ∧ 
  (∀ c : ℝ, (3:ℝ)^a + (9:ℝ)^b ≥ c^2 - c → c ≤ c_max) ∧
  ((3:ℝ)^a + (9:ℝ)^b ≥ c_max^2 - c_max) :=
sorry

end max_c_value_l3273_327398


namespace highest_page_number_with_19_sevens_l3273_327349

/-- Counts the number of occurrences of a digit in a natural number -/
def countDigit (n : ℕ) (d : ℕ) : ℕ :=
  sorry

/-- Counts the total occurrences of a digit in a range of natural numbers -/
def countDigitInRange (start finish : ℕ) (d : ℕ) : ℕ :=
  sorry

/-- The highest page number that can be reached with a given number of sevens -/
def highestPageNumber (numSevens : ℕ) : ℕ :=
  sorry

theorem highest_page_number_with_19_sevens :
  highestPageNumber 19 = 99 :=
sorry

end highest_page_number_with_19_sevens_l3273_327349


namespace polar_to_rectangular_l3273_327324

/-- The rectangular coordinate equation equivalent to the polar equation ρ = 4sin θ -/
theorem polar_to_rectangular (x y ρ θ : ℝ) 
  (h1 : ρ = 4 * Real.sin θ)
  (h2 : x = ρ * Real.cos θ)
  (h3 : y = ρ * Real.sin θ)
  (h4 : ρ^2 = x^2 + y^2) :
  x^2 + y^2 - 4*y = 0 := by
sorry

end polar_to_rectangular_l3273_327324


namespace meters_examined_l3273_327332

/-- The percentage of meters rejected as defective -/
def rejection_rate : ℝ := 0.10

/-- The number of defective meters found -/
def defective_meters : ℕ := 10

/-- The total number of meters examined -/
def total_meters : ℕ := 100

/-- Theorem stating that if the rejection rate is 10% and 10 defective meters are found,
    then the total number of meters examined is 100 -/
theorem meters_examined (h : ℝ) (defective : ℕ) (total : ℕ) 
  (h_rate : h = rejection_rate)
  (h_defective : defective = defective_meters)
  (h_total : total = total_meters) :
  ↑defective = h * ↑total := by
  sorry

end meters_examined_l3273_327332


namespace candy_distribution_theorem_l3273_327356

/-- Represents the candy distribution pattern -/
def candy_distribution (n : ℕ) (k : ℕ) : ℕ :=
  (k * (k + 1) / 2) % n

/-- Predicate to check if all children receive candy -/
def all_children_receive_candy (n : ℕ) : Prop :=
  ∀ a : ℕ, ∃ k : ℕ, candy_distribution n k = a % n

/-- Theorem: All children receive candy iff n is a power of 2 -/
theorem candy_distribution_theorem (n : ℕ) :
  all_children_receive_candy n ↔ ∃ k : ℕ, n = 2^k :=
sorry

end candy_distribution_theorem_l3273_327356


namespace sum_of_two_elements_equals_power_of_two_l3273_327314

def M : Set ℕ := {m : ℕ | ∃ n : ℕ, m = n * (n + 1)}

theorem sum_of_two_elements_equals_power_of_two :
  ∃ n : ℕ, n * (n - 1) ∈ M ∧ n * (n + 1) ∈ M ∧ n * (n - 1) + n * (n + 1) = 2^2021 := by
  sorry

end sum_of_two_elements_equals_power_of_two_l3273_327314


namespace cubic_quadratic_relation_l3273_327372

theorem cubic_quadratic_relation (A B C D : ℝ) (p q r : ℝ) (a b : ℝ) : 
  (A * p^3 + B * p^2 + C * p + D = 0) →
  (A * q^3 + B * q^2 + C * q + D = 0) →
  (A * r^3 + B * r^2 + C * r + D = 0) →
  ((p^2 + q)^2 + a * (p^2 + q) + b = 0) →
  ((q^2 + r)^2 + a * (q^2 + r) + b = 0) →
  ((r^2 + p)^2 + a * (r^2 + p) + b = 0) →
  (A ≠ 0) →
  a = (A * B + 2 * A * C - B^2) / A^2 := by
sorry

end cubic_quadratic_relation_l3273_327372


namespace breakfast_cost_is_17_l3273_327350

/-- The cost of breakfast for Francis and Kiera -/
def breakfast_cost (muffin_price fruit_cup_price : ℕ) 
  (francis_muffins francis_fruit_cups kiera_muffins kiera_fruit_cups : ℕ) : ℕ :=
  (francis_muffins + kiera_muffins) * muffin_price + 
  (francis_fruit_cups + kiera_fruit_cups) * fruit_cup_price

/-- Theorem stating that the total cost of breakfast for Francis and Kiera is $17 -/
theorem breakfast_cost_is_17 : 
  breakfast_cost 2 3 2 2 2 1 = 17 := by
  sorry

end breakfast_cost_is_17_l3273_327350


namespace sin_210_degrees_l3273_327355

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end sin_210_degrees_l3273_327355


namespace team_selection_count_l3273_327311

theorem team_selection_count (n : ℕ) (k : ℕ) (h1 : n = 17) (h2 : k = 4) :
  Nat.choose n k = 2380 := by
  sorry

end team_selection_count_l3273_327311


namespace initial_rate_is_three_l3273_327367

/-- Calculates the initial consumption rate per soldier per day -/
def initial_consumption_rate (initial_soldiers : ℕ) (initial_duration : ℕ) 
  (additional_soldiers : ℕ) (new_consumption_rate : ℚ) (new_duration : ℕ) : ℚ :=
  (((initial_soldiers + additional_soldiers) * new_consumption_rate * new_duration) / 
   (initial_soldiers * initial_duration))

/-- Theorem stating that the initial consumption rate is 3 kg per soldier per day -/
theorem initial_rate_is_three :
  initial_consumption_rate 1200 30 528 (5/2) 25 = 3 := by
  sorry

end initial_rate_is_three_l3273_327367


namespace system_solution_l3273_327388

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  x + 3 * y + 14 ≤ 0 ∧
  x^4 + 2 * x^2 * y^2 + y^4 + 64 - 20 * x^2 - 20 * y^2 = 8 * x * y

-- Theorem stating that the solution to the system is (-2, -4)
theorem system_solution :
  ∃! p : ℝ × ℝ, system p.1 p.2 ∧ p = (-2, -4) := by
  sorry


end system_solution_l3273_327388


namespace speed_difference_proof_l3273_327379

/-- Proves the speed difference between two vehicles given their travel conditions -/
theorem speed_difference_proof (base_speed : ℝ) (time : ℝ) (total_distance : ℝ) :
  base_speed = 44 →
  time = 4 →
  total_distance = 384 →
  ∃ (speed_diff : ℝ),
    speed_diff > 0 ∧
    total_distance = base_speed * time + (base_speed + speed_diff) * time ∧
    speed_diff = 8 := by
  sorry

end speed_difference_proof_l3273_327379


namespace lumberjack_chopped_25_trees_l3273_327326

/-- Represents the lumberjack's work -/
structure LumberjackWork where
  logs_per_tree : ℕ
  firewood_per_log : ℕ
  total_firewood : ℕ

/-- Calculates the number of trees chopped based on the lumberjack's work -/
def trees_chopped (work : LumberjackWork) : ℕ :=
  work.total_firewood / (work.logs_per_tree * work.firewood_per_log)

/-- Theorem stating that given the specific conditions, the lumberjack chopped 25 trees -/
theorem lumberjack_chopped_25_trees :
  let work := LumberjackWork.mk 4 5 500
  trees_chopped work = 25 := by
  sorry

#eval trees_chopped (LumberjackWork.mk 4 5 500)

end lumberjack_chopped_25_trees_l3273_327326


namespace movie_marathon_difference_l3273_327345

/-- The duration of a movie marathon with three movies. -/
structure MovieMarathon where
  first_movie : ℝ
  second_movie : ℝ
  last_movie : ℝ
  total_time : ℝ

/-- The conditions of the movie marathon problem. -/
def movie_marathon_conditions (m : MovieMarathon) : Prop :=
  m.first_movie = 2 ∧
  m.second_movie = m.first_movie * 1.5 ∧
  m.total_time = 9 ∧
  m.total_time = m.first_movie + m.second_movie + m.last_movie

/-- The theorem stating the difference between the combined time of the first two movies
    and the last movie is 1 hour. -/
theorem movie_marathon_difference (m : MovieMarathon) 
  (h : movie_marathon_conditions m) : 
  m.first_movie + m.second_movie - m.last_movie = 1 := by
  sorry

end movie_marathon_difference_l3273_327345


namespace leftover_snacks_problem_l3273_327376

/-- Calculates the number of leftover snacks when feeding goats with dietary restrictions --/
def leftover_snacks (total_goats : ℕ) (restricted_goats : ℕ) (baby_carrots : ℕ) (cherry_tomatoes : ℕ) : ℕ :=
  let unrestricted_goats := total_goats - restricted_goats
  let tomatoes_per_restricted_goat := cherry_tomatoes / restricted_goats
  let leftover_tomatoes := cherry_tomatoes % restricted_goats
  let carrots_per_unrestricted_goat := baby_carrots / unrestricted_goats
  let leftover_carrots := baby_carrots % unrestricted_goats
  leftover_tomatoes + leftover_carrots

/-- Theorem stating that given the problem conditions, 6 snacks will be left over --/
theorem leftover_snacks_problem :
  leftover_snacks 9 3 124 56 = 6 := by
  sorry

end leftover_snacks_problem_l3273_327376


namespace largest_constant_inequality_l3273_327318

theorem largest_constant_inequality (C : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + x*y + 1 ≥ C*(x + y)) ↔ C ≤ 2/Real.sqrt 3 :=
sorry

end largest_constant_inequality_l3273_327318


namespace machine_output_percentage_l3273_327369

theorem machine_output_percentage :
  let prob_defect_A : ℝ := 9 / 1000
  let prob_defect_B : ℝ := 1 / 50
  let total_prob_defect : ℝ := 0.0156
  ∃ p : ℝ, 
    0 ≤ p ∧ p ≤ 1 ∧
    total_prob_defect = p * prob_defect_A + (1 - p) * prob_defect_B ∧
    p = 0.4 := by
  sorry

end machine_output_percentage_l3273_327369


namespace p_toluidine_molecular_weight_l3273_327390

/-- The atomic weight of Carbon in atomic mass units (amu) -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in atomic mass units (amu) -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of Nitrogen in atomic mass units (amu) -/
def nitrogen_weight : ℝ := 14.01

/-- The chemical formula of p-Toluidine -/
structure ChemicalFormula where
  carbon : ℕ
  hydrogen : ℕ
  nitrogen : ℕ

/-- The chemical formula of p-Toluidine (C7H9N) -/
def p_toluidine : ChemicalFormula := ⟨7, 9, 1⟩

/-- Calculate the molecular weight of a chemical compound given its formula -/
def molecular_weight (formula : ChemicalFormula) : ℝ :=
  formula.carbon * carbon_weight + 
  formula.hydrogen * hydrogen_weight + 
  formula.nitrogen * nitrogen_weight

/-- Theorem: The molecular weight of p-Toluidine is 107.152 amu -/
theorem p_toluidine_molecular_weight : 
  molecular_weight p_toluidine = 107.152 := by
  sorry

end p_toluidine_molecular_weight_l3273_327390


namespace bee_swarm_count_l3273_327319

theorem bee_swarm_count : ∃ x : ℕ, 
  x > 0 ∧ 
  (x / 5 : ℚ) + (x / 3 : ℚ) + 3 * ((x / 3 : ℚ) - (x / 5 : ℚ)) + 1 = x ∧ 
  x = 15 := by
  sorry

end bee_swarm_count_l3273_327319


namespace min_value_of_sum_of_reciprocals_l3273_327373

theorem min_value_of_sum_of_reciprocals (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 2) :
  (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) ≥ 9/4 := by
sorry

end min_value_of_sum_of_reciprocals_l3273_327373


namespace total_legs_in_pasture_l3273_327305

/-- The number of cows in the pasture -/
def num_cows : ℕ := 115

/-- The number of legs each cow has -/
def legs_per_cow : ℕ := 4

/-- Theorem: The total number of legs seen in a pasture with 115 cows, 
    where each cow has 4 legs, is equal to 460. -/
theorem total_legs_in_pasture : num_cows * legs_per_cow = 460 := by
  sorry

end total_legs_in_pasture_l3273_327305


namespace initial_money_calculation_l3273_327352

/-- Proves that if a person spends half of their initial money, then half of the remaining money, 
    and is left with 1250 won, their initial amount was 5000 won. -/
theorem initial_money_calculation (initial_money : ℝ) : 
  (initial_money / 2) / 2 = 1250 → initial_money = 5000 := by
  sorry

end initial_money_calculation_l3273_327352


namespace square_of_two_times_sqrt_three_l3273_327343

theorem square_of_two_times_sqrt_three : (2 * Real.sqrt 3) ^ 2 = 12 := by
  sorry

end square_of_two_times_sqrt_three_l3273_327343


namespace worker_loading_time_l3273_327375

/-- The time taken by two workers to load a truck together -/
def combined_time : ℝ := 3.428571428571429

/-- The time taken by the second worker to load the truck alone -/
def second_worker_time : ℝ := 8

/-- The time taken by the first worker to load the truck alone -/
def first_worker_time : ℝ := 1.142857142857143

/-- Theorem stating the relationship between the workers' loading times -/
theorem worker_loading_time :
  (1 / combined_time) = (1 / first_worker_time) + (1 / second_worker_time) :=
by sorry

end worker_loading_time_l3273_327375


namespace student_count_l3273_327394

theorem student_count (n : ℕ) (rank_top rank_bottom : ℕ) 
  (h1 : rank_top = 75)
  (h2 : rank_bottom = 75)
  (h3 : n = rank_top + rank_bottom - 1) :
  n = 149 := by
  sorry

end student_count_l3273_327394


namespace right_triangle_with_specific_median_l3273_327334

/-- A right triangle with a median to the hypotenuse -/
structure RightTriangleWithMedian where
  a : ℝ  -- First leg
  b : ℝ  -- Second leg
  c : ℝ  -- Hypotenuse
  m : ℝ  -- Median to the hypotenuse
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem
  median_property : m = c / 2  -- Property of the median to the hypotenuse
  perimeter_difference : b - a = 1  -- Difference in perimeters of smaller triangles

/-- The sides of the triangle satisfy the given conditions -/
theorem right_triangle_with_specific_median (t : RightTriangleWithMedian) 
  (h1 : t.a + t.m = 8)  -- Perimeter of one smaller triangle
  (h2 : t.b + t.m = 9)  -- Perimeter of the other smaller triangle
  : t.a = 3 ∧ t.b = 4 ∧ t.c = 5 := by
  sorry


end right_triangle_with_specific_median_l3273_327334


namespace savanna_safari_snake_ratio_l3273_327325

-- Define the number of animals in Safari National Park
def safari_lions : ℕ := 100
def safari_snakes : ℕ := safari_lions / 2
def safari_giraffes : ℕ := safari_snakes - 10

-- Define the number of animals in Savanna National Park
def savanna_lions : ℕ := 2 * safari_lions
def savanna_giraffes : ℕ := safari_giraffes + 20

-- Define the total number of animals in Savanna National Park
def savanna_total : ℕ := 410

-- Define the ratio of snakes in Savanna to Safari
def snake_ratio : ℚ := 3

theorem savanna_safari_snake_ratio :
  snake_ratio = (savanna_total - savanna_lions - savanna_giraffes) / safari_snakes := by
  sorry

end savanna_safari_snake_ratio_l3273_327325


namespace final_statue_count_statue_count_increases_l3273_327336

/-- Represents the number of statues on Grandma Molly's lawn over four years -/
def statue_count : ℕ → ℕ
| 0 => 4  -- Initial number of statues
| 1 => 7  -- After year 2: 4 + 7 - 4
| 2 => 9  -- After year 3: 7 + 9 - 7
| 3 => 13 -- After year 4: 9 + 4
| _ => 13 -- Any year after 4

/-- The final number of statues after four years is 13 -/
theorem final_statue_count : statue_count 3 = 13 := by
  sorry

/-- The number of statues increases over the years -/
theorem statue_count_increases (n : ℕ) : n < 3 → statue_count n < statue_count (n + 1) := by
  sorry

end final_statue_count_statue_count_increases_l3273_327336


namespace tangent_line_is_correct_l3273_327329

/-- The circle with equation x^2 + y^2 = 20 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 20}

/-- The point M (2, -4) -/
def M : ℝ × ℝ := (2, -4)

/-- The proposed tangent line with equation x - 2y - 10 = 0 -/
def TangentLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - 2*p.2 - 10 = 0}

theorem tangent_line_is_correct :
  ∀ p ∈ TangentLine,
    (p ∈ Circle → p = M) ∧
    (∀ q ∈ Circle, q ≠ M → (p.1 - M.1) * (q.1 - M.1) + (p.2 - M.2) * (q.2 - M.2) = 0) :=
sorry

end tangent_line_is_correct_l3273_327329


namespace square_roots_problem_l3273_327374

theorem square_roots_problem (x a : ℝ) : 
  x > 0 ∧ (2*a - 3)^2 = x ∧ (5 - a)^2 = x → a = -2 ∧ x = 49 := by
  sorry

end square_roots_problem_l3273_327374


namespace expression_evaluation_l3273_327310

theorem expression_evaluation :
  let x : ℚ := 1/2
  (2*x - 1)^2 - (3*x + 1)*(3*x - 1) + 5*x*(x - 1) = -5/2 := by
  sorry

end expression_evaluation_l3273_327310


namespace midpoint_chain_l3273_327306

theorem midpoint_chain (A B C D E F G : ℝ) : 
  (C = (A + B) / 2) →  -- C is midpoint of AB
  (D = (A + C) / 2) →  -- D is midpoint of AC
  (E = (A + D) / 2) →  -- E is midpoint of AD
  (F = (A + E) / 2) →  -- F is midpoint of AE
  (G = (A + F) / 2) →  -- G is midpoint of AF
  (G - A = 4) →        -- AG = 4
  (B - A = 128) :=     -- AB = 128
by sorry

end midpoint_chain_l3273_327306
