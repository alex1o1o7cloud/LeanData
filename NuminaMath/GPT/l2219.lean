import Mathlib

namespace original_number_is_7_l2219_221968

theorem original_number_is_7 (N : ℕ) (h : ∃ (k : ℤ), N = 12 * k + 7) : N = 7 :=
sorry

end original_number_is_7_l2219_221968


namespace red_balls_in_total_color_of_158th_ball_l2219_221909

def totalBalls : Nat := 200
def redBallsPerCycle : Nat := 5
def whiteBallsPerCycle : Nat := 4
def blackBallsPerCycle : Nat := 3
def cycleLength : Nat := redBallsPerCycle + whiteBallsPerCycle + blackBallsPerCycle

theorem red_balls_in_total :
  (totalBalls / cycleLength) * redBallsPerCycle + min redBallsPerCycle (totalBalls % cycleLength) = 85 :=
by sorry

theorem color_of_158th_ball :
  let positionInCycle := (158 - 1) % cycleLength + 1
  positionInCycle ≤ redBallsPerCycle := by sorry

end red_balls_in_total_color_of_158th_ball_l2219_221909


namespace sandy_age_l2219_221991

theorem sandy_age (S M : ℕ) 
  (h1 : M = S + 16) 
  (h2 : (↑S : ℚ) / ↑M = 7 / 9) : 
  S = 56 :=
by sorry

end sandy_age_l2219_221991


namespace a_beats_b_by_4_rounds_l2219_221911

variable (T_a T_b : ℝ)
variable (race_duration : ℝ) -- duration of the 4-round race in minutes
variable (time_difference : ℝ) -- Time that a beats b by in the 4-round race

open Real

-- Given conditions
def conditions :=
  (T_a = 7.5) ∧                             -- a's time to complete one round
  (race_duration = T_a * 4 + 10) ∧          -- a beats b by 10 minutes in a 4-round race
  (time_difference = T_b - T_a)             -- The time difference per round is T_b - T_a

-- Mathematical proof statement
theorem a_beats_b_by_4_rounds
  (h : conditions T_a T_b race_duration time_difference) :
  10 / time_difference = 4 := by
  sorry

end a_beats_b_by_4_rounds_l2219_221911


namespace find_2a_minus_b_l2219_221922

-- Define conditions
def f (x : ℝ) (a b : ℝ) := a * x + b
def g (x : ℝ) := -5 * x + 7
def h (x : ℝ) (a b : ℝ) := f (g x) a b
def h_inv (x : ℝ) := x - 9

-- Statement to prove
theorem find_2a_minus_b (a b : ℝ) 
(h_eq : ∀ x, h x a b = a * (-5 * x + 7) + b)
(h_inv_eq : ∀ x, h_inv x = x - 9)
(h_hinv_eq : ∀ x, h (h_inv x) a b = x) :
  2 * a - b = -54 / 5 := sorry

end find_2a_minus_b_l2219_221922


namespace initial_depth_dug_l2219_221901

theorem initial_depth_dug :
  (∀ days : ℕ, 75 * 8 * days / D = 140 * 6 * days / 70) → D = 50 :=
by
  sorry

end initial_depth_dug_l2219_221901


namespace sugar_needed_for_third_layer_l2219_221921

-- Let cups be the amount of sugar, and define the layers
def first_layer_sugar : ℕ := 2
def second_layer_sugar : ℕ := 2 * first_layer_sugar
def third_layer_sugar : ℕ := 3 * second_layer_sugar

-- The theorem we want to prove
theorem sugar_needed_for_third_layer : third_layer_sugar = 12 := by
  sorry

end sugar_needed_for_third_layer_l2219_221921


namespace cos_theta_value_projection_value_l2219_221973

noncomputable def vec_a : (ℝ × ℝ) := (3, 1)
noncomputable def vec_b : (ℝ × ℝ) := (-2, 4)

theorem cos_theta_value :
  let a := vec_a
  let b := vec_b
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  dot_product / (magnitude_a * magnitude_b) = - Real.sqrt 2 / 10 :=
by 
  sorry

theorem projection_value :
  let a := vec_a
  let b := vec_b
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  let cos_theta := dot_product / (magnitude_a * magnitude_b)
  cos_theta = - Real.sqrt 2 / 10 →
  magnitude_a * cos_theta = - Real.sqrt 5 / 5 :=
by 
  sorry

end cos_theta_value_projection_value_l2219_221973


namespace combined_weight_of_candles_l2219_221966

theorem combined_weight_of_candles 
  (beeswax_weight_per_candle : ℕ)
  (coconut_oil_weight_per_candle : ℕ)
  (total_candles : ℕ)
  (candles_made : ℕ) 
  (total_weight: ℕ) 
  : 
  beeswax_weight_per_candle = 8 → 
  coconut_oil_weight_per_candle = 1 → 
  total_candles = 10 → 
  candles_made = total_candles - 3 →
  total_weight = candles_made * (beeswax_weight_per_candle + coconut_oil_weight_per_candle) →
  total_weight = 63 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end combined_weight_of_candles_l2219_221966


namespace utility_bills_l2219_221904

-- Definitions for the conditions
def four_hundred := 4 * 100
def five_fifty := 5 * 50
def seven_twenty := 7 * 20
def eight_ten := 8 * 10
def total := four_hundred + five_fifty + seven_twenty + eight_ten

-- Lean statement for the proof problem
theorem utility_bills : total = 870 :=
by
  -- inserting skip proof placeholder
  sorry

end utility_bills_l2219_221904


namespace similar_triangles_height_l2219_221976

theorem similar_triangles_height (h_small: ℝ) (area_ratio: ℝ) (h_large: ℝ) :
  h_small = 5 ∧ area_ratio = 1/9 ∧ h_large = 3 * h_small → h_large = 15 :=
by
  intro h 
  sorry

end similar_triangles_height_l2219_221976


namespace tangent_normal_lines_l2219_221918

theorem tangent_normal_lines :
  ∃ m_t b_t m_n b_n,
    (∀ x y, y = 1 / (1 + x^2) → y = m_t * x + b_t → 4 * x + 25 * y - 13 = 0) ∧
    (∀ x y, y = 1 / (1 + x^2) → y = m_n * x + b_n → 125 * x - 20 * y - 246 = 0) :=
by
  sorry

end tangent_normal_lines_l2219_221918


namespace krishan_nandan_investment_l2219_221964

def investment_ratio (k r₁ r₂ : ℕ) (N T Gn : ℕ) : Prop :=
  k = r₁ ∧ r₂ = 1 ∧ Gn = N * T ∧ k * N * 3 * T + Gn = 26000 ∧ Gn = 2000

/-- Given the conditions, the ratio of Krishan's investment to Nandan's investment is 4:1. -/
theorem krishan_nandan_investment :
  ∃ k N T Gn Gn_total : ℕ, 
    investment_ratio k 4 1 N T Gn  ∧ k * N * 3 * T = 24000 :=
by
  sorry

end krishan_nandan_investment_l2219_221964


namespace circle_center_coordinates_l2219_221990

theorem circle_center_coordinates :
  let p1 := (2, -3)
  let p2 := (8, 9)
  let midpoint (x₁ y₁ x₂ y₂ : ℝ) : ℝ × ℝ := ((x₁ + x₂) / 2, (y₁ + y₂) / 2)
  midpoint (2 : ℝ) (-3) 8 9 = (5, 3) :=
by
  sorry

end circle_center_coordinates_l2219_221990


namespace shyam_weight_increase_l2219_221936

theorem shyam_weight_increase (x : ℝ) 
    (h1 : x > 0)
    (ratio : ∀ Ram Shyam : ℝ, (Ram / Shyam) = 7 / 5)
    (ram_increase : ∀ Ram : ℝ, Ram' = Ram + 0.1 * Ram)
    (total_weight_after : Ram' + Shyam' = 82.8)
    (total_weight_increase : 82.8 = 1.15 * total_weight) :
    (Shyam' - Shyam) / Shyam * 100 = 22 :=
by
  sorry

end shyam_weight_increase_l2219_221936


namespace total_votes_l2219_221956

theorem total_votes (V : ℝ) (win_percentage : ℝ) (majority : ℝ) (lose_percentage : ℝ)
  (h1 : win_percentage = 0.75) (h2 : lose_percentage = 0.25) (h3 : majority = 420) :
  V = 840 :=
by
  sorry

end total_votes_l2219_221956


namespace horse_catches_up_l2219_221984

-- Definitions based on given conditions
def dog_speed := 20 -- derived from 5 steps * 4 meters
def horse_speed := 21 -- derived from 3 steps * 7 meters
def initial_distance := 30 -- dog has already run 30 meters

-- Statement to be proved
theorem horse_catches_up (d h : ℕ) (time : ℕ) :
  d = dog_speed → h = horse_speed →
  initial_distance = 30 →
  h * time = initial_distance + dog_speed * time →
  time = 600 / (h - d) ∧ h * time - initial_distance = 600 :=
by
  intros
  -- Proof placeholders
  sorry  -- Omit the actual proof steps

end horse_catches_up_l2219_221984


namespace no_prime_divisible_by_57_l2219_221932

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. --/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Given that 57 is equal to 3 times 19.--/
theorem no_prime_divisible_by_57 : ∀ p : ℕ, is_prime p → ¬ (57 ∣ p) :=
by
  sorry

end no_prime_divisible_by_57_l2219_221932


namespace calculate_final_price_l2219_221987

def original_price : ℝ := 120
def fixture_discount : ℝ := 0.20
def decor_discount : ℝ := 0.15

def discounted_price_after_first_discount (p : ℝ) (d : ℝ) : ℝ :=
  p * (1 - d)

def final_price (p : ℝ) (d1 : ℝ) (d2 : ℝ) : ℝ :=
  let price_after_first_discount := discounted_price_after_first_discount p d1
  price_after_first_discount * (1 - d2)

theorem calculate_final_price :
  final_price original_price fixture_discount decor_discount = 81.60 :=
by sorry

end calculate_final_price_l2219_221987


namespace reflected_line_equation_l2219_221985

-- Definitions based on given conditions
def incident_line (x : ℝ) : ℝ := 2 * x + 1
def reflection_line (x : ℝ) : ℝ := x

-- Statement of the mathematical problem
theorem reflected_line_equation :
  ∀ x y : ℝ, (incident_line x = y) → (reflection_line x = x) → y = (1/2) * x - (1/2) :=
sorry

end reflected_line_equation_l2219_221985


namespace kangaroo_fraction_sum_l2219_221972

theorem kangaroo_fraction_sum (G P : ℕ) (hG : 1 ≤ G) (hP : 1 ≤ P) (hTotal : G + P = 2016) : 
  (G * (P / G) + P * (G / P) = 2016) :=
by
  sorry

end kangaroo_fraction_sum_l2219_221972


namespace solve_for_a_l2219_221928

noncomputable def area_of_triangle (b c : ℝ) : ℝ :=
  1 / 2 * b * c * Real.sin (Real.pi / 3)

theorem solve_for_a (a b c : ℝ) (hA : 60 = 60) 
  (h_area : area_of_triangle b c = 3 * Real.sqrt 3 / 2)
  (h_sum_bc : b + c = 3 * Real.sqrt 3) :
  a = 3 :=
sorry

end solve_for_a_l2219_221928


namespace unique_number_not_in_range_l2219_221950

noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ :=
  (p * x + q) / (r * x + s)

theorem unique_number_not_in_range (p q r s : ℝ) (h₀ : p ≠ 0) (h₁ : q ≠ 0) (h₂ : r ≠ 0) (h₃ : s ≠ 0) 
  (h₄ : g p q r s 23 = 23) (h₅ : g p q r s 101 = 101) (h₆ : ∀ x, x ≠ -s/r → g p q r s (g p q r s x) = x) :
  p / r = 62 :=
sorry

end unique_number_not_in_range_l2219_221950


namespace inequality_for_pos_reals_l2219_221998

-- Definitions for positive real numbers
variables {x y : ℝ}
def is_pos_real (x : ℝ) : Prop := x > 0

-- Theorem statement
theorem inequality_for_pos_reals (hx : is_pos_real x) (hy : is_pos_real y) : 
  2 * (x^2 + y^2) ≥ (x + y)^2 :=
by
  sorry

end inequality_for_pos_reals_l2219_221998


namespace evaluate_expression_l2219_221926

theorem evaluate_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^(2*y) * y^(2*x)) / (y^(2*y) * x^(2*x)) = (x / y)^(2 * (y - x)) :=
by
  sorry

end evaluate_expression_l2219_221926


namespace cube_net_count_l2219_221903

/-- A net of a cube is a two-dimensional arrangement of six squares.
    A regular tetrahedron has exactly 2 unique nets.
    For a cube, consider all possible ways in which the six faces can be arranged such that they 
    form a cube when properly folded. -/
theorem cube_net_count : cube_nets_count = 11 :=
sorry

end cube_net_count_l2219_221903


namespace savings_for_mother_l2219_221919

theorem savings_for_mother (Liam_oranges Claire_oranges: ℕ) (Liam_price_per_pair Claire_price_per_orange: ℝ) :
  Liam_oranges = 40 ∧ Liam_price_per_pair = 2.50 ∧
  Claire_oranges = 30 ∧ Claire_price_per_orange = 1.20 →
  ((Liam_oranges / 2) * Liam_price_per_pair + Claire_oranges * Claire_price_per_orange) = 86 :=
by
  intros
  sorry

end savings_for_mother_l2219_221919


namespace miles_to_mall_l2219_221962

noncomputable def miles_to_grocery_store : ℕ := 10
noncomputable def miles_to_pet_store : ℕ := 5
noncomputable def miles_back_home : ℕ := 9
noncomputable def miles_per_gallon : ℕ := 15
noncomputable def cost_per_gallon : ℝ := 3.50
noncomputable def total_cost_of_gas : ℝ := 7.00
noncomputable def total_miles_driven := 2 * miles_per_gallon

theorem miles_to_mall : total_miles_driven -
  (miles_to_grocery_store + miles_to_pet_store + miles_back_home) = 6 :=
by
  -- proof omitted 
  sorry

end miles_to_mall_l2219_221962


namespace exists_integers_a_b_c_d_l2219_221913

-- Define the problem statement in Lean 4

theorem exists_integers_a_b_c_d (x y z : ℕ) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), x = a^2 + b^2 ∧ y = c^2 + d^2 ∧ z = a * c + b * d :=
by
  sorry

end exists_integers_a_b_c_d_l2219_221913


namespace arithmetic_mean_difference_l2219_221955

theorem arithmetic_mean_difference (p q r : ℝ)
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 22) :
  r - p = 24 :=
by
  sorry

end arithmetic_mean_difference_l2219_221955


namespace quadratic_has_two_distinct_real_roots_l2219_221960

theorem quadratic_has_two_distinct_real_roots (a : ℝ) (h : a ≠ 0): 
  (a < 4 / 3) ↔ (∃ x y : ℝ, x ≠ y ∧  a * x^2 - 4 * x + 3 = 0 ∧ a * y^2 - 4 * y + 3 = 0) := 
sorry

end quadratic_has_two_distinct_real_roots_l2219_221960


namespace curve_in_second_quadrant_l2219_221931

theorem curve_in_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0 → x < 0 ∧ y > 0) ↔ (a > 2) :=
sorry

end curve_in_second_quadrant_l2219_221931


namespace order_of_three_numbers_l2219_221992

theorem order_of_three_numbers :
  let a := (7 : ℝ) ^ (0.3 : ℝ)
  let b := (0.3 : ℝ) ^ (7 : ℝ)
  let c := Real.log (0.3 : ℝ)
  a > b ∧ b > c ∧ a > c :=
by
  sorry

end order_of_three_numbers_l2219_221992


namespace set_intersection_and_polynomial_solution_l2219_221935

theorem set_intersection_and_polynomial_solution {a b : ℝ} :
  let A := {x : ℝ | x + 2 < 0}
  let B := {x : ℝ | (x + 3) * (x - 1) > 0}
  (A ∩ B = {x | x < -3}) ∧ ((A ∪ B = {x | x < -2 ∨ x > 1}) →
    (a = 2 ∧ b = -4)) :=
by
  let A := {x : ℝ | x + 2 < 0}
  let B := {x : ℝ | (x + 3) * (x - 1) > 0}
  sorry

end set_intersection_and_polynomial_solution_l2219_221935


namespace find_integers_satisfying_condition_l2219_221943

-- Define the inequality condition
def condition (x : ℤ) : Prop := x * x < 3 * x

-- Prove that the set of integers satisfying the condition is {1, 2}
theorem find_integers_satisfying_condition :
  { x : ℤ | condition x } = {1, 2} := 
by {
  sorry
}

end find_integers_satisfying_condition_l2219_221943


namespace range_of_a_l2219_221980

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x| = ax + 1 → x < 0) → a > 1 :=
by
  sorry

end range_of_a_l2219_221980


namespace petyas_number_l2219_221997

theorem petyas_number :
  ∃ (N : ℕ), 
  (N % 2 = 1 ∧ ∃ (M : ℕ), N = 149 * M ∧ (M = Nat.mod (N : ℕ) (100))) →
  (N = 745 ∨ N = 3725) :=
by
  sorry

end petyas_number_l2219_221997


namespace infinite_solutions_implies_d_eq_five_l2219_221908

theorem infinite_solutions_implies_d_eq_five (d : ℝ) :
  (∀ y : ℝ, 3 * (5 + d * y) = 15 * y + 15) ↔ (d = 5) := by
sorry

end infinite_solutions_implies_d_eq_five_l2219_221908


namespace fireworks_display_l2219_221953

def num_digits_year : ℕ := 4
def fireworks_per_digit : ℕ := 6
def regular_letters_phrase : ℕ := 12
def fireworks_per_regular_letter : ℕ := 5

def fireworks_H : ℕ := 8
def fireworks_E : ℕ := 7
def fireworks_L : ℕ := 6
def fireworks_O : ℕ := 9

def num_boxes : ℕ := 100
def fireworks_per_box : ℕ := 10

def total_fireworks : ℕ :=
  (num_digits_year * fireworks_per_digit) +
  (regular_letters_phrase * fireworks_per_regular_letter) +
  (fireworks_H + fireworks_E + 2 * fireworks_L + fireworks_O) + 
  (num_boxes * fireworks_per_box)

theorem fireworks_display : total_fireworks = 1120 := by
  sorry

end fireworks_display_l2219_221953


namespace brenda_cakes_l2219_221948

-- Definitions based on the given conditions
def cakes_per_day : ℕ := 20
def days : ℕ := 9
def total_cakes_baked : ℕ := cakes_per_day * days
def cakes_sold : ℕ := total_cakes_baked / 2
def cakes_left : ℕ := total_cakes_baked - cakes_sold

-- Formulate the theorem
theorem brenda_cakes : cakes_left = 90 :=
by {
  -- To skip the proof steps
  sorry
}

end brenda_cakes_l2219_221948


namespace area_of_region_l2219_221983

theorem area_of_region : 
  (∃ A : ℝ, 
    (∀ x y : ℝ, 
      (|4 * x - 20| + |3 * y + 9| ≤ 4) → 
      A = (32 / 3))) :=
by 
  sorry

end area_of_region_l2219_221983


namespace given_equation_roots_sum_cubes_l2219_221923

theorem given_equation_roots_sum_cubes (r s t : ℝ) 
    (h1 : 6 * r ^ 3 + 1506 * r + 3009 = 0)
    (h2 : 6 * s ^ 3 + 1506 * s + 3009 = 0)
    (h3 : 6 * t ^ 3 + 1506 * t + 3009 = 0)
    (sum_roots : r + s + t = 0) :
    (r + s) ^ 3 + (s + t) ^ 3 + (t + r) ^ 3 = 1504.5 := 
by 
  -- proof omitted
  sorry

end given_equation_roots_sum_cubes_l2219_221923


namespace average_incorrect_l2219_221963

theorem average_incorrect : ¬( (1 + 1 + 0 + 2 + 4) / 5 = 2) :=
by {
  sorry
}

end average_incorrect_l2219_221963


namespace correct_value_of_A_sub_B_l2219_221975

variable {x y : ℝ}

-- Given two polynomials A and B where B = 3x - 2y, and a mistaken equation A + B = x - y,
-- we want to prove the correct value of A - B.
theorem correct_value_of_A_sub_B (A B : ℝ) (h1 : B = 3 * x - 2 * y) (h2 : A + B = x - y) :
  A - B = -5 * x + 3 * y :=
by
  sorry

end correct_value_of_A_sub_B_l2219_221975


namespace solve_for_n_l2219_221988

theorem solve_for_n (n : ℕ) : (9^n * 9^n * 9^n * 9^n = 729^4) -> n = 3 := 
by
  sorry

end solve_for_n_l2219_221988


namespace pre_bought_tickets_l2219_221994

theorem pre_bought_tickets (P : ℕ) 
  (h1 : ∃ P, 155 * P + 2900 = 6000) : P = 20 :=
by {
  -- Insert formalization of steps leading to P = 20
  sorry
}

end pre_bought_tickets_l2219_221994


namespace geometric_mean_condition_l2219_221934

variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}

theorem geometric_mean_condition
  (h_arith : (a 1 + a 2 + a 3 + a 4 + a 5 + a 6) / 6 = (a 3 + a 4) / 2)
  (h_geom_pos : ∀ n, 0 < b n) :
  Real.sqrt (b 1 * b 2 * b 3 * b 4 * b 5 * b 6) = Real.sqrt (b 3 * b 4) :=
sorry

end geometric_mean_condition_l2219_221934


namespace huangs_tax_is_65_yuan_l2219_221961

noncomputable def monthly_salary : ℝ := 2900
noncomputable def tax_free_portion : ℝ := 2000
noncomputable def tax_rate_5_percent : ℝ := 0.05
noncomputable def tax_rate_10_percent : ℝ := 0.10

noncomputable def taxable_income_amount (income : ℝ) (exemption : ℝ) : ℝ := income - exemption

noncomputable def personal_income_tax (income : ℝ) : ℝ :=
  let taxable_income := taxable_income_amount income tax_free_portion
  if taxable_income ≤ 500 then
    taxable_income * tax_rate_5_percent
  else
    (500 * tax_rate_5_percent) + ((taxable_income - 500) * tax_rate_10_percent)

theorem huangs_tax_is_65_yuan : personal_income_tax monthly_salary = 65 :=
by
  sorry

end huangs_tax_is_65_yuan_l2219_221961


namespace maximum_value_inequality_l2219_221917

theorem maximum_value_inequality (x y : ℝ) : 
  (3 * x + 4 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 50 :=
sorry

end maximum_value_inequality_l2219_221917


namespace no_valid_pairs_l2219_221959

theorem no_valid_pairs (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) :
  ¬(1000 * a + 100 * b + 32) % 99 = 0 :=
by
  sorry

end no_valid_pairs_l2219_221959


namespace rectangle_area_l2219_221927

theorem rectangle_area (length : ℝ) (width : ℝ) (increased_width : ℝ) (area : ℝ)
  (h1 : length = 12)
  (h2 : increased_width = width * 1.2)
  (h3 : increased_width = 12)
  (h4 : area = length * width) : 
  area = 120 := 
by
  sorry

end rectangle_area_l2219_221927


namespace neg_prop_p_l2219_221933

def prop_p (x : ℝ) : Prop := x ≥ 0 → Real.log (x^2 + 1) ≥ 0

theorem neg_prop_p : (¬ (∀ x ≥ 0, Real.log (x^2 + 1) ≥ 0)) ↔ (∃ x ≥ 0, Real.log (x^2 + 1) < 0) := by
  sorry

end neg_prop_p_l2219_221933


namespace inequality_solution_l2219_221940

theorem inequality_solution 
  (x : ℝ) : 
  (x^2 / (x+2)^2 ≥ 0) ↔ x ≠ -2 := 
by
  sorry

end inequality_solution_l2219_221940


namespace range_x_minus_2y_l2219_221920

variable (x y : ℝ)

def cond1 : Prop := -1 ≤ x ∧ x < 2
def cond2 : Prop := 0 < y ∧ y ≤ 1

theorem range_x_minus_2y 
  (h1 : cond1 x) 
  (h2 : cond2 y) : 
  -3 ≤ x - 2 * y ∧ x - 2 * y < 2 := 
by
  sorry

end range_x_minus_2y_l2219_221920


namespace find_f_neg_two_l2219_221981

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_neg_two (h : ∀ x : ℝ, x ≠ 0 → f (1 / x) + (1 / x) * f (-x) = 2 * x) :
  f (-2) = 7 / 2 :=
by
  sorry

end find_f_neg_two_l2219_221981


namespace cost_condition_shirt_costs_purchasing_plans_maximize_profit_l2219_221916

/-- Define the costs and prices of shirts A and B -/
def cost_A (m : ℝ) : ℝ := m
def cost_B (m : ℝ) : ℝ := m - 10
def price_A : ℝ := 260
def price_B : ℝ := 180

/-- Condition: total cost of 3 A shirts and 2 B shirts is 480 -/
theorem cost_condition (m : ℝ) : 3 * (cost_A m) + 2 * (cost_B m) = 480 := by
  sorry

/-- The cost of each A shirt is 100 and each B shirt is 90 -/
theorem shirt_costs : ∃ m, cost_A m = 100 ∧ cost_B m = 90 := by
  sorry

/-- Number of purchasing plans for at least $34,000 profit with 300 shirts and at most 110 A shirts -/
theorem purchasing_plans : ∃ x, 100 ≤ x ∧ x ≤ 110 ∧ 
  (260 * x + 180 * (300 - x) - 100 * x - 90 * (300 - x) ≥ 34000) := by
  sorry

/- Maximize profit given 60 < a < 80:
   - 60 < a < 70: 110 A shirts, 190 B shirts.
   - a = 70: any combination satisfying conditions.
   - 70 < a < 80: 100 A shirts, 200 B shirts. -/

theorem maximize_profit (a : ℝ) (ha : 60 < a ∧ a < 80) : 
  ∃ x, ((60 < a ∧ a < 70 ∧ x = 110 ∧ (300 - x) = 190) ∨ 
        (a = 70) ∨ 
        (70 < a ∧ a < 80 ∧ x = 100 ∧ (300 - x) = 200)) := by
  sorry

end cost_condition_shirt_costs_purchasing_plans_maximize_profit_l2219_221916


namespace find_x_l2219_221978

/-- Given vectors a and b, and a is parallel to b -/
def vectors (x : ℝ) : Prop :=
  let a := (x, 2)
  let b := (2, 1)
  a.1 * b.2 = a.2 * b.1

theorem find_x: ∀ x : ℝ, vectors x → x = 4 :=
by
  intros x h
  sorry

end find_x_l2219_221978


namespace min_value_of_expression_l2219_221999

theorem min_value_of_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : 
  (3 * x + y) * (x + 3 * z) * (y + z + 1) ≥ 48 :=
by
  sorry

end min_value_of_expression_l2219_221999


namespace initial_members_in_family_c_l2219_221941

theorem initial_members_in_family_c 
  (a b d e f : ℕ)
  (ha : a = 7)
  (hb : b = 8)
  (hd : d = 13)
  (he : e = 6)
  (hf : f = 10)
  (average_after_moving : (a - 1) + (b - 1) + (d - 1) + (e - 1) + (f - 1) + (x : ℕ) - 1 = 48) :
  x = 10 := by
  sorry

end initial_members_in_family_c_l2219_221941


namespace taco_price_theorem_l2219_221902

noncomputable def price_hard_shell_taco_proof
  (H : ℤ)
  (price_soft : ℤ := 2)
  (num_hard_tacos_family : ℤ := 4)
  (num_soft_tacos_family : ℤ := 3)
  (num_additional_customers : ℤ := 10)
  (total_earnings : ℤ := 66)
  : Prop :=
  4 * H + 3 * price_soft + 10 * 2 * price_soft = total_earnings → H = 5

theorem taco_price_theorem : price_hard_shell_taco_proof 5 := 
by
  sorry

end taco_price_theorem_l2219_221902


namespace fraction_identity_l2219_221910

variables {a b : ℝ}

theorem fraction_identity (h : a^2 + b^2 ≠ 0) :
  (a^4 + b^4) / (a^2 + b^2) = a^2 + b^2 - (2 * a^2 * b^2) / (a^2 + b^2) :=
by sorry

end fraction_identity_l2219_221910


namespace sandra_oranges_l2219_221942

theorem sandra_oranges (S E B: ℕ) (h1: E = 7 * S) (h2: E = 252) (h3: B = 12) : S / B = 3 := by
  sorry

end sandra_oranges_l2219_221942


namespace system_solution_l2219_221945

theorem system_solution (x b y : ℝ) (h1 : 4 * x + 2 * y = b) (h2 : 3 * x + 4 * y = 3 * b) (h3 : x = 3) :
  b = -1 :=
by
  -- proof to be filled in
  sorry

end system_solution_l2219_221945


namespace prism_cut_out_l2219_221957

theorem prism_cut_out (x y : ℕ)
  (H1 : 15 * 5 * 4 - y * 5 * x = 120)
  (H2 : x < 4) :
  x = 3 ∧ y = 12 :=
sorry

end prism_cut_out_l2219_221957


namespace find_constants_and_formula_l2219_221967

namespace ArithmeticSequence

variable {a : ℕ → ℤ} -- Sequence a : ℕ → ℤ

-- Given conditions
axiom a_5 : a 5 = 11
axiom a_12 : a 12 = 31

-- Definitions to be proved
def a_1 := -2
def d := 3
def a_formula (n : ℕ) := a_1 + (n - 1) * d

theorem find_constants_and_formula :
  (a 1 = a_1) ∧
  (a 2 - a 1 = d) ∧
  (a 20 = 55) ∧
  (∀ n, a n = a_formula n) := by
  sorry

end ArithmeticSequence

end find_constants_and_formula_l2219_221967


namespace original_average_l2219_221974

theorem original_average (n : ℕ) (A : ℝ) (new_avg : ℝ) 
  (h1 : n = 25) 
  (h2 : new_avg = 140) 
  (h3 : 2 * A = new_avg) : A = 70 :=
sorry

end original_average_l2219_221974


namespace solveInequalityRegion_l2219_221979

noncomputable def greatestIntegerLessThan (x : ℝ) : ℤ :=
  Int.floor x

theorem solveInequalityRegion :
  ∀ (x y : ℝ), abs x < 1 → abs y < 1 → x * y ≠ 0 → (greatestIntegerLessThan (x + y) ≤ 
  greatestIntegerLessThan x + greatestIntegerLessThan y) :=
by
  intros x y h1 h2 h3
  sorry

end solveInequalityRegion_l2219_221979


namespace percentage_increase_l2219_221939

theorem percentage_increase (employees_dec : ℝ) (employees_jan : ℝ) (inc : ℝ) (percentage : ℝ) :
  employees_dec = 470 →
  employees_jan = 408.7 →
  inc = employees_dec - employees_jan →
  percentage = (inc / employees_jan) * 100 →
  percentage = 15 := 
sorry

end percentage_increase_l2219_221939


namespace minimum_k_for_mutual_criticism_l2219_221971

theorem minimum_k_for_mutual_criticism (k : ℕ) (h1 : 15 * k > 105) : k ≥ 8 := by
  sorry

end minimum_k_for_mutual_criticism_l2219_221971


namespace evaluate_expression_l2219_221952

-- Given conditions
def a : ℕ := 3
def b : ℕ := 2

-- Proof problem statement
theorem evaluate_expression : (1 / 3 : ℝ) ^ (b - a) = 3 := sorry

end evaluate_expression_l2219_221952


namespace percent_of_absent_students_l2219_221970

noncomputable def absent_percentage : ℚ :=
  let total_students := 120
  let boys := 70
  let girls := 50
  let absent_boys := boys * (1/5 : ℚ)
  let absent_girls := girls * (1/4 : ℚ)
  let total_absent := absent_boys + absent_girls
  (total_absent / total_students) * 100

theorem percent_of_absent_students : absent_percentage = 22.5 := sorry

end percent_of_absent_students_l2219_221970


namespace fresh_grapes_weight_l2219_221947

theorem fresh_grapes_weight (F D : ℝ) (h1 : D = 0.625) (h2 : 0.10 * F = 0.80 * D) : F = 5 := by
  -- Using premises h1 and h2, we aim to prove that F = 5
  sorry

end fresh_grapes_weight_l2219_221947


namespace find_inverse_of_25_l2219_221958

-- Define the inverses and the modulo
def inverse_mod (a m i : ℤ) : Prop :=
  (a * i) % m = 1

-- The given condition in the problem
def condition (m : ℤ) : Prop :=
  inverse_mod 5 m 39

-- The theorem we want to prove
theorem find_inverse_of_25 (m : ℤ) (h : condition m) : inverse_mod 25 m 8 :=
by
  sorry

end find_inverse_of_25_l2219_221958


namespace arithmetic_sequence_a10_gt_0_l2219_221986

variable {α : Type*} [LinearOrderedField α]

-- Definitions of the conditions
def arithmetic_sequence (a : ℕ → α) := ∀ n1 n2, a n1 - a n2 = (n1 - n2) * (a 1 - a 0)
def a9_lt_0 (a : ℕ → α) := a 9 < 0
def a1_add_a18_gt_0 (a : ℕ → α) := a 1 + a 18 > 0

-- The proof statement
theorem arithmetic_sequence_a10_gt_0 
  (a : ℕ → α) 
  (h_arith : arithmetic_sequence a) 
  (h_a9 : a9_lt_0 a) 
  (h_a1_a18 : a1_add_a18_gt_0 a) : 
  a 10 > 0 := 
sorry

end arithmetic_sequence_a10_gt_0_l2219_221986


namespace pens_sold_to_recover_investment_l2219_221907

-- Given the conditions
variables (P C : ℝ) (N : ℝ)
-- P is the total cost of 30 pens
-- C is the cost price of each pen
-- N is the number of pens sold to recover the initial investment

-- Stating the conditions
axiom h1 : P = 30 * C
axiom h2 : N * 1.5 * C = P

-- Proving that N = 20
theorem pens_sold_to_recover_investment (P C N : ℝ) (h1 : P = 30 * C) (h2 : N * 1.5 * C = P) : N = 20 :=
by
  sorry

end pens_sold_to_recover_investment_l2219_221907


namespace test_average_score_l2219_221900

theorem test_average_score (A : ℝ) (h : 0.90 * A + 5 = 86) : A = 90 := 
by
  sorry

end test_average_score_l2219_221900


namespace no_solution_nat_x_satisfies_eq_l2219_221937

def sum_digits (x : ℕ) : ℕ := x.digits 10 |>.sum

theorem no_solution_nat_x_satisfies_eq (x : ℕ) :
  ¬ (x + sum_digits x + sum_digits (sum_digits x) = 2014) :=
by
  sorry

end no_solution_nat_x_satisfies_eq_l2219_221937


namespace range_of_a1_l2219_221982

theorem range_of_a1 (a1 : ℝ) :
  (∃ (a2 a3 : ℝ), 
    ((a2 = 2 * a1 - 12) ∨ (a2 = a1 / 2 + 12)) ∧
    ((a3 = 2 * a2 - 12) ∨ (a3 = a2 / 2 + 12)) ) →
  ((a3 > a1) ↔ ((a1 ≤ 12) ∨ (24 ≤ a1))) :=
by
  sorry

end range_of_a1_l2219_221982


namespace not_integer_20_diff_l2219_221905

theorem not_integer_20_diff (a b : ℝ) (hne : a ≠ b) 
  (no_roots1 : ∀ x, x^2 + 20 * a * x + 10 * b ≠ 0) 
  (no_roots2 : ∀ x, x^2 + 20 * b * x + 10 * a ≠ 0) : 
  ¬ (∃ k : ℤ, 20 * (b - a) = k) :=
by
  sorry

end not_integer_20_diff_l2219_221905


namespace painted_faces_cube_eq_54_l2219_221954

def painted_faces (n : ℕ) : ℕ :=
  if n = 5 then (3 * 3) * 6 else 0

theorem painted_faces_cube_eq_54 : painted_faces 5 = 54 := by {
  sorry
}

end painted_faces_cube_eq_54_l2219_221954


namespace asymptotes_of_hyperbola_l2219_221929

theorem asymptotes_of_hyperbola :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 / 9 = 1) → (y = 3/2 * x ∨ y = -3/2 * x) :=
by
  intro x y h
  -- Proof would go here
  sorry

end asymptotes_of_hyperbola_l2219_221929


namespace inequality_solution_eq_l2219_221996

theorem inequality_solution_eq :
  ∀ y : ℝ, 2 ≤ |y - 5| ∧ |y - 5| ≤ 8 ↔ (-3 ≤ y ∧ y ≤ 3) ∨ (7 ≤ y ∧ y ≤ 13) :=
by
  sorry

end inequality_solution_eq_l2219_221996


namespace trigonometric_identity_l2219_221925

theorem trigonometric_identity :
  Real.cos (70 * Real.pi / 180) * Real.sin (80 * Real.pi / 180) +
  Real.cos (20 * Real.pi / 180) * Real.sin (10 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end trigonometric_identity_l2219_221925


namespace peanuts_in_box_l2219_221995

   theorem peanuts_in_box (initial_peanuts : ℕ) (added_peanuts : ℕ) (total_peanuts : ℕ) 
     (h1 : initial_peanuts = 4) (h2 : added_peanuts = 6) : total_peanuts = initial_peanuts + added_peanuts :=
   by
     sorry

   example : peanuts_in_box 4 6 10 rfl rfl = rfl :=
   by
     sorry
   
end peanuts_in_box_l2219_221995


namespace original_triangle_area_l2219_221965

-- Define the scaling factor and given areas
def scaling_factor : ℕ := 2
def new_triangle_area : ℕ := 32

-- State that if the dimensions of the original triangle are doubled, the area becomes 32 square feet
theorem original_triangle_area (original_area : ℕ) : (scaling_factor * scaling_factor) * original_area = new_triangle_area → original_area = 8 := 
by
  intros h
  sorry

end original_triangle_area_l2219_221965


namespace probability_two_red_cards_l2219_221924

theorem probability_two_red_cards : 
  let total_cards := 100;
  let red_cards := 50;
  let black_cards := 50;
  (red_cards / total_cards : ℝ) * ((red_cards - 1) / (total_cards - 1) : ℝ) = 49 / 198 := 
by
  sorry

end probability_two_red_cards_l2219_221924


namespace sum_of_arithmetic_progression_l2219_221906

theorem sum_of_arithmetic_progression :
  let a := 30
  let d := -3
  let n := 20
  let S_n := n / 2 * (2 * a + (n - 1) * d)
  S_n = 30 :=
by
  sorry

end sum_of_arithmetic_progression_l2219_221906


namespace pizza_slices_leftover_l2219_221993

def slices_per_small_pizza := 4
def slices_per_large_pizza := 8
def small_pizzas_purchased := 3
def large_pizzas_purchased := 2

def george_slices := 3
def bob_slices := george_slices + 1
def susie_slices := bob_slices / 2
def bill_slices := 3
def fred_slices := 3
def mark_slices := 3

def total_slices := small_pizzas_purchased * slices_per_small_pizza + large_pizzas_purchased * slices_per_large_pizza
def total_eaten_slices := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices

def slices_leftover := total_slices - total_eaten_slices

theorem pizza_slices_leftover : slices_leftover = 10 := by
  sorry

end pizza_slices_leftover_l2219_221993


namespace trigonometry_expression_zero_l2219_221949

variable {r : ℝ} {A B C : ℝ}
variable (a b c : ℝ) (sinA sinB sinC : ℝ)

-- The conditions from the problem
axiom Law_of_Sines_a : a = 2 * r * sinA
axiom Law_of_Sines_b : b = 2 * r * sinB
axiom Law_of_Sines_c : c = 2 * r * sinC

-- The theorem statement
theorem trigonometry_expression_zero :
  a * (sinC - sinB) + b * (sinA - sinC) + c * (sinB - sinA) = 0 :=
by
  -- Skipping the proof
  sorry

end trigonometry_expression_zero_l2219_221949


namespace find_f_neg3_l2219_221912

theorem find_f_neg3 : 
  (∃ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → 5 * f (1 / x) + 3 * f x / x = 2 * x^2) ∧ f (-3) = 14029 / 72) :=
sorry

end find_f_neg3_l2219_221912


namespace ethanol_relationship_l2219_221915

variables (a b c x : ℝ)
def total_capacity := a + b + c = 300
def ethanol_content := x = 0.10 * a + 0.15 * b + 0.20 * c
def ethanol_bounds := 30 ≤ x ∧ x ≤ 60

theorem ethanol_relationship : total_capacity a b c → ethanol_bounds x → ethanol_content a b c x :=
by
  intros h_total h_bounds
  unfold total_capacity at h_total
  unfold ethanol_bounds at h_bounds
  unfold ethanol_content
  sorry

end ethanol_relationship_l2219_221915


namespace tables_needed_for_luncheon_l2219_221914

theorem tables_needed_for_luncheon (invited attending remaining tables_needed : ℕ) (H1 : invited = 24) (H2 : remaining = 10) (H3 : attending = invited - remaining) (H4 : tables_needed = attending / 7) : tables_needed = 2 :=
by
  sorry

end tables_needed_for_luncheon_l2219_221914


namespace total_packets_needed_l2219_221969

theorem total_packets_needed :
  let oak_seedlings := 420
  let oak_per_packet := 7
  let maple_seedlings := 825
  let maple_per_packet := 5
  let pine_seedlings := 2040
  let pine_per_packet := 12
  let oak_packets := oak_seedlings / oak_per_packet
  let maple_packets := maple_seedlings / maple_per_packet
  let pine_packets := pine_seedlings / pine_per_packet
  let total_packets := oak_packets + maple_packets + pine_packets
  total_packets = 395 := 
by {
  sorry
}

end total_packets_needed_l2219_221969


namespace algorithm_contains_sequential_structure_l2219_221930

theorem algorithm_contains_sequential_structure :
  (∀ algorithm : Type, ∃ seq_struct : Prop, seq_struct) ∧
  (∀ algorithm : Type, ∃ sel_struct : Prop, sel_struct ∨ ¬ sel_struct) ∧
  (∀ algorithm : Type, ∃ loop_struct : Prop, loop_struct) →
  (∀ algorithm : Type, ∃ seq_struct : Prop, seq_struct) := by
  sorry

end algorithm_contains_sequential_structure_l2219_221930


namespace n_value_l2219_221989

theorem n_value (n : ℤ) (h1 : (18888 - n) % 11 = 0) : n = 7 :=
sorry

end n_value_l2219_221989


namespace decreasing_interval_implies_range_of_a_l2219_221951

noncomputable def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem decreasing_interval_implies_range_of_a (a : ℝ)
  (h : ∀ x y : ℝ, x ≤ y → y ≤ 4 → f a x ≥ f a y) : a ≤ -3 :=
by
  sorry

end decreasing_interval_implies_range_of_a_l2219_221951


namespace arithmetic_sequence_a15_l2219_221977

theorem arithmetic_sequence_a15 (a_n S_n : ℕ → ℝ) (a_9 : a_n 9 = 4) (S_15 : S_n 15 = 30) :
  let a_1 := (-12 : ℝ)
  let d := (2 : ℝ)
  a_n 15 = 16 :=
by
  sorry

end arithmetic_sequence_a15_l2219_221977


namespace min_abs_sum_l2219_221944

theorem min_abs_sum (a b c d : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a^2 + b * c = 9) (h2 : b * c + d^2 = 9) (h3 : a * b + b * d = 0) (h4 : a * c + c * d = 0) :
  |a| + |b| + |c| + |d| = 8 :=
sorry

end min_abs_sum_l2219_221944


namespace xy_value_l2219_221938

theorem xy_value (x y : ℝ) (h : |x - 5| + |y + 3| = 0) : x * y = -15 := by
  sorry

end xy_value_l2219_221938


namespace finish_time_is_1_10_PM_l2219_221946

-- Definitions of the problem conditions
def start_time := 9 * 60 -- 9:00 AM in minutes past midnight
def third_task_finish_time := 11 * 60 + 30 -- 11:30 AM in minutes past midnight
def num_tasks := 5
def tasks1_to_3_duration := third_task_finish_time - start_time
def one_task_duration := tasks1_to_3_duration / 3
def total_duration := one_task_duration * num_tasks

-- Statement to prove the final time when John finishes the fifth task
theorem finish_time_is_1_10_PM : 
  start_time + total_duration = 13 * 60 + 10 := 
by 
  sorry

end finish_time_is_1_10_PM_l2219_221946
