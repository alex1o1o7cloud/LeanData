import Mathlib

namespace find_k_l48_4822

theorem find_k (x y k : ℝ) (h1 : 2 * x - y = 4) (h2 : k * x - 3 * y = 12) : k = 6 := by
  sorry

end find_k_l48_4822


namespace problem_solution_l48_4878

variables {a b c : ℝ}

theorem problem_solution (h : a + b + c = 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a^3 * b^3 / ((a^3 - b^2 * c) * (b^3 - a^2 * c)) +
  a^3 * c^3 / ((a^3 - b^2 * c) * (c^3 - a^2 * b)) +
  b^3 * c^3 / ((b^3 - a^2 * c) * (c^3 - a^2 * b))) = 1 :=
sorry

end problem_solution_l48_4878


namespace classify_discuss_l48_4801

theorem classify_discuss (a b c : ℚ) (h : a * b * c > 0) : 
  (|a| / a + |b| / b + |c| / c = 3) ∨ (|a| / a + |b| / b + |c| / c = -1) :=
sorry

end classify_discuss_l48_4801


namespace find_k_l48_4808

open BigOperators

def a (n : ℕ) : ℕ := 2 ^ n

theorem find_k (k : ℕ) (h : a (k+1) + a (k+2) + a (k+3) + a (k+4) + a (k+5) + a (k+6) + a (k+7) + a (k+8) + a (k+9) + a (k+10) = 2 ^ 15 - 2 ^ 5) : k = 4 :=
sorry

end find_k_l48_4808


namespace product_to_difference_l48_4890

def x := 88 * 1.25
def y := 150 * 0.60
def z := 60 * 1.15

def product := x * y * z
def difference := x - y

theorem product_to_difference :
  product ^ difference = 683100 ^ 20 := 
sorry

end product_to_difference_l48_4890


namespace prob_seven_heads_in_ten_tosses_l48_4806

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  (Nat.choose n k)

noncomputable def probability_of_heads (n k : ℕ) : ℚ :=
  (binomial_coefficient n k) * (0.5^k : ℚ) * (0.5^(n - k) : ℚ)

theorem prob_seven_heads_in_ten_tosses :
  probability_of_heads 10 7 = 15 / 128 :=
by
  sorry

end prob_seven_heads_in_ten_tosses_l48_4806


namespace remainder_when_P_divided_by_ab_l48_4831

-- Given conditions
variables {P a b c Q Q' R R' : ℕ}

-- Provided equations as conditions
def equation1 : P = a * Q + R :=
sorry

def equation2 : Q = (b + c) * Q' + R' :=
sorry

-- Proof problem statement
theorem remainder_when_P_divided_by_ab :
  P % (a * b) = (a * c * Q' + a * R' + R) % (a * b) :=
by
  sorry

end remainder_when_P_divided_by_ab_l48_4831


namespace correct_option_C_l48_4884

theorem correct_option_C (a b c : ℝ) : 2 * a^2 * b * c - a^2 * b * c = a^2 * b * c := 
sorry

end correct_option_C_l48_4884


namespace smallest_number_of_rectangles_needed_l48_4830

theorem smallest_number_of_rectangles_needed :
  ∃ n, (n * 12 = 144) ∧ (∀ k, (k * 12 = 144) → k ≥ n) := by
  sorry

end smallest_number_of_rectangles_needed_l48_4830


namespace percent_increase_expenditure_l48_4897

theorem percent_increase_expenditure (cost_per_minute_2005 minutes_2005 minutes_2020 total_expenditure_2005 total_expenditure_2020 : ℕ)
  (h1 : cost_per_minute_2005 = 10)
  (h2 : minutes_2005 = 200)
  (h3 : minutes_2020 = 2 * minutes_2005)
  (h4 : total_expenditure_2005 = minutes_2005 * cost_per_minute_2005)
  (h5 : total_expenditure_2020 = minutes_2020 * cost_per_minute_2005) :
  ((total_expenditure_2020 - total_expenditure_2005) * 100 / total_expenditure_2005) = 100 :=
by
  sorry

end percent_increase_expenditure_l48_4897


namespace line_intersects_circle_l48_4823

theorem line_intersects_circle 
  (radius : ℝ) 
  (distance_center_line : ℝ) 
  (h_radius : radius = 4) 
  (h_distance : distance_center_line = 3) : 
  radius > distance_center_line := 
by 
  sorry

end line_intersects_circle_l48_4823


namespace center_of_circle_l48_4868

-- Let's define the circle as a set of points satisfying the given condition.
def circle (x y : ℝ) : Prop := (x - 2) ^ 2 + (y + 1) ^ 2 = 4

-- Prove that the point (2, -1) is the center of this circle in ℝ².
theorem center_of_circle : ∀ (x y : ℝ), circle (x - 2) (y + 1) ↔ (x, y) = (2, -1) :=
by
  intros x y
  sorry

end center_of_circle_l48_4868


namespace sin_and_tan_inequality_l48_4858

theorem sin_and_tan_inequality (n : ℕ) (hn : 0 < n) :
  2 * Real.sin (1 / n) + Real.tan (1 / n) > 3 / n :=
sorry

end sin_and_tan_inequality_l48_4858


namespace white_paint_amount_l48_4828

theorem white_paint_amount (total_blue_paint additional_blue_paint total_mix blue_parts red_parts white_parts green_parts : ℕ) 
    (h_ratio: blue_parts = 7 ∧ red_parts = 2 ∧ white_parts = 1 ∧ green_parts = 1)
    (total_blue_paint_eq: total_blue_paint = 140)
    (max_total_mix: additional_blue_paint ≤ 220 - total_blue_paint) 
    : (white_parts * (total_blue_paint / blue_parts)) = 20 := 
by 
  sorry

end white_paint_amount_l48_4828


namespace remainder_b22_div_35_l48_4853

def b_n (n : ℕ) : Nat :=
  ((List.range (n + 1)).drop 1).foldl (λ acc k => acc * 10^(Nat.digits 10 k).length + k) 0

theorem remainder_b22_div_35 : (b_n 22) % 35 = 17 :=
  sorry

end remainder_b22_div_35_l48_4853


namespace square_perimeter_l48_4883

theorem square_perimeter (s : ℕ) (h : 5 * s / 2 = 40) : 4 * s = 64 := by
  sorry

end square_perimeter_l48_4883


namespace find_width_of_rectangle_l48_4818

-- Given conditions
variable (P l w : ℕ)
variable (h1 : P = 240)
variable (h2 : P = 3 * l)

-- Prove the width of the rectangular field is 40 meters
theorem find_width_of_rectangle : w = 40 :=
  by 
  -- Add the necessary logical steps here
  sorry

end find_width_of_rectangle_l48_4818


namespace printing_time_345_l48_4856

def printing_time (total_pages : ℕ) (rate : ℕ) : ℕ :=
  total_pages / rate

theorem printing_time_345 :
  printing_time 345 23 = 15 :=
by
  sorry

end printing_time_345_l48_4856


namespace red_light_probability_l48_4857

theorem red_light_probability :
  let red_duration := 30
  let yellow_duration := 5
  let green_duration := 40
  let total_duration := red_duration + yellow_duration + green_duration
  let probability_of_red := (red_duration:ℝ) / total_duration
  probability_of_red = 2 / 5 := by
    sorry

end red_light_probability_l48_4857


namespace remainder_of_3045_div_32_l48_4859

theorem remainder_of_3045_div_32 : 3045 % 32 = 5 :=
by sorry

end remainder_of_3045_div_32_l48_4859


namespace solve_equation_l48_4892

theorem solve_equation (x : ℝ) (h : x ≠ 1) : 
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) → 
  x = -4 ∨ x = -2 :=
by 
  sorry

end solve_equation_l48_4892


namespace range_of_reciprocal_sum_l48_4850

theorem range_of_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
    4 ≤ (1/x + 1/y) :=
by
  sorry

end range_of_reciprocal_sum_l48_4850


namespace sqrt_nested_l48_4832

theorem sqrt_nested (x : ℝ) (hx : 0 ≤ x) : Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (15 / 16) := by
  sorry

end sqrt_nested_l48_4832


namespace minimum_area_of_triangle_is_sqrt_58_div_2_l48_4825

noncomputable def smallest_area_of_triangle (t s : ℝ) : ℝ :=
  (1/2) * Real.sqrt (5 * s^2 - 4 * s * t - 4 * s + 2 * t^2 + 10 * t + 13)

theorem minimum_area_of_triangle_is_sqrt_58_div_2 : ∃ t s : ℝ, smallest_area_of_triangle t s = Real.sqrt 58 / 2 := 
  by
  sorry

end minimum_area_of_triangle_is_sqrt_58_div_2_l48_4825


namespace ellipse_problem_l48_4891

theorem ellipse_problem
  (a b : ℝ)
  (h₀ : 0 < a)
  (h₁ : 0 < b)
  (h₂ : a > b)
  (P Q : ℝ × ℝ)
  (ellipse_eq : ∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | (p.1 ^ 2) / (a ^ 2) + (p.2 ^ 2) / (b ^ 2) = 1})
  (A : ℝ × ℝ)
  (hA : A = (a, 0))
  (R : ℝ × ℝ)
  (O : ℝ × ℝ)
  (hO : O = (0, 0))
  (AQ_OP_parallels : ∀ (x y : ℝ) (Qx Qy Px Py : ℝ), 
    x = a ∧ y = 0  ∧ (Qx, Qy) = (x, y) ↔ (O.1, O.2) = (Px, Py)
    ) :
  ∀ (AQ AR OP : ℝ), 
  AQ = dist (a, 0) Q → 
  AR = dist A R → 
  OP = dist O P → 
  |AQ * AR| / (OP ^ 2) = 2 :=
  sorry

end ellipse_problem_l48_4891


namespace product_gcd_lcm_l48_4863

-- Conditions.
def num1 : ℕ := 12
def num2 : ℕ := 9

-- Theorem to prove.
theorem product_gcd_lcm (a b : ℕ) (h1 : a = num1) (h2 : b = num2) :
  (Nat.gcd a b) * (Nat.lcm a b) = 108 :=
by
  sorry

end product_gcd_lcm_l48_4863


namespace longer_side_length_l48_4817

theorem longer_side_length (total_rope_length shorter_side_length longer_side_length : ℝ) 
  (h1 : total_rope_length = 100) 
  (h2 : shorter_side_length = 22) 
  : 2 * shorter_side_length + 2 * longer_side_length = total_rope_length -> longer_side_length = 28 :=
by sorry

end longer_side_length_l48_4817


namespace find_p_plus_q_l48_4809

noncomputable def calculate_p_plus_q (DE EF FD WX : ℕ) (Area : ℕ → ℝ) : ℕ :=
  let s := (DE + EF + FD) / 2
  let triangle_area := (Real.sqrt (s * (s - DE) * (s - EF) * (s - FD))) / 2
  let delta := triangle_area / (225 * WX)
  let gcd := Nat.gcd 41 225
  let p := 41 / gcd
  let q := 225 / gcd
  p + q

theorem find_p_plus_q : calculate_p_plus_q 13 30 19 15 (fun θ => 30 * θ - (41 / 225) * θ^2) = 266 := by
  sorry

end find_p_plus_q_l48_4809


namespace tan_ratio_triangle_area_l48_4827

theorem tan_ratio (a b c A B C : ℝ) (h1 : c = -3 * b * Real.cos A) :
  Real.tan A / Real.tan B = -4 := by
  sorry

theorem triangle_area (a b c A B C : ℝ) (h1 : c = -3 * b * Real.cos A)
  (h2 : c = 2) (h3 : Real.tan C = 3 / 4) :
  ∃ S : ℝ, S = 1 / 2 * b * c * Real.sin A ∧ S = 4 / 3 := by
  sorry

end tan_ratio_triangle_area_l48_4827


namespace adults_on_field_trip_l48_4807

-- Define the conditions
def van_capacity : ℕ := 7
def num_students : ℕ := 33
def num_vans : ℕ := 6

-- Define the total number of people that can be transported given the number of vans and capacity per van
def total_people : ℕ := num_vans * van_capacity

-- The number of people that can be transported minus the number of students gives the number of adults
def num_adults : ℕ := total_people - num_students

-- Theorem to prove the number of adults is 9
theorem adults_on_field_trip : num_adults = 9 :=
by
  -- Skipping the proof
  sorry

end adults_on_field_trip_l48_4807


namespace fraction_ordering_l48_4812

theorem fraction_ordering :
  let a := (6 : ℚ) / 22
  let b := (8 : ℚ) / 32
  let c := (10 : ℚ) / 29
  a < b ∧ b < c :=
by
  sorry

end fraction_ordering_l48_4812


namespace probability_of_odd_sum_l48_4855

theorem probability_of_odd_sum (P : ℝ → Prop) 
    (P_even_sum : ℝ)
    (P_odd_sum : ℝ)
    (h1 : P_even_sum = 2 * P_odd_sum) 
    (h2 : P_even_sum + P_odd_sum = 1) :
    P_odd_sum = 4/9 := 
sorry

end probability_of_odd_sum_l48_4855


namespace eliminate_y_l48_4820

theorem eliminate_y (x y : ℝ) (h1 : 2 * x + 3 * y = 1) (h2 : 3 * x - 6 * y = 7) :
  (4 * x + 6 * y) + (3 * x - 6 * y) = 9 :=
by
  sorry

end eliminate_y_l48_4820


namespace plane_arrival_time_l48_4838

-- Define the conditions
def departure_time := 11 -- common departure time in hours (11:00)
def bus_speed := 100 -- bus speed in km/h
def train_speed := 300 -- train speed in km/h
def plane_speed := 900 -- plane speed in km/h
def bus_arrival := 20 -- bus arrival time in hours (20:00)
def train_arrival := 14 -- train arrival time in hours (14:00)

-- Given these conditions, we need to prove the plane arrival time
theorem plane_arrival_time : (departure_time + (900 / plane_speed)) = 12 := by
  sorry

end plane_arrival_time_l48_4838


namespace yuan_older_than_david_l48_4852

theorem yuan_older_than_david (David_age : ℕ) (Yuan_age : ℕ) 
  (h1 : Yuan_age = 2 * David_age) 
  (h2 : David_age = 7) : 
  Yuan_age - David_age = 7 := by
  sorry

end yuan_older_than_david_l48_4852


namespace even_digit_perfect_squares_odd_digit_perfect_squares_l48_4800

-- Define the property of being a four-digit number
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Define the property of having even digits
def is_even_digit_number (n : ℕ) : Prop :=
  ∀ digit ∈ (n.digits 10), digit % 2 = 0

-- Define the property of having odd digits
def is_odd_digit_number (n : ℕ) : Prop :=
  ∀ digit ∈ (n.digits 10), digit % 2 = 1

-- Part (a) statement
theorem even_digit_perfect_squares :
  ∀ n : ℕ, is_four_digit n ∧ is_even_digit_number n ∧ ∃ m : ℕ, n = m * m ↔ 
    n = 4624 ∨ n = 6084 ∨ n = 6400 ∨ n = 8464 :=
sorry

-- Part (b) statement
theorem odd_digit_perfect_squares :
  ∀ n : ℕ, is_four_digit n ∧ is_odd_digit_number n ∧ ∃ m : ℕ, n = m * m → false :=
sorry

end even_digit_perfect_squares_odd_digit_perfect_squares_l48_4800


namespace monotonically_increasing_l48_4861

variable {R : Type} [LinearOrderedField R]

def f (x : R) : R := 3 * x + 1

theorem monotonically_increasing : ∀ x₁ x₂ : R, x₁ < x₂ → f x₁ < f x₂ :=
by
  intro x₁ x₂ h
 -- this is where the proof would go
  sorry

end monotonically_increasing_l48_4861


namespace mixed_fruit_juice_litres_opened_l48_4877

theorem mixed_fruit_juice_litres_opened (cocktail_cost_per_litre : ℝ)
  (mixed_juice_cost_per_litre : ℝ) (acai_cost_per_litre : ℝ)
  (acai_litres_added : ℝ) (total_mixed_juice_opened : ℝ) :
  cocktail_cost_per_litre = 1399.45 ∧
  mixed_juice_cost_per_litre = 262.85 ∧
  acai_cost_per_litre = 3104.35 ∧
  acai_litres_added = 23.333333333333336 ∧
  (mixed_juice_cost_per_litre * total_mixed_juice_opened + 
  acai_cost_per_litre * acai_litres_added = 
  cocktail_cost_per_litre * (total_mixed_juice_opened + acai_litres_added)) →
  total_mixed_juice_opened = 35 :=
sorry

end mixed_fruit_juice_litres_opened_l48_4877


namespace key_lime_yield_l48_4811

def audrey_key_lime_juice_yield (cup_to_key_lime_juice_ratio: ℚ) (lime_juice_doubling_factor: ℚ) (tablespoons_per_cup: ℕ) (num_key_limes: ℕ) : ℚ :=
  let total_lime_juice_cups := cup_to_key_lime_juice_ratio * lime_juice_doubling_factor
  let total_lime_juice_tablespoons := total_lime_juice_cups * tablespoons_per_cup
  total_lime_juice_tablespoons / num_key_limes

-- Statement of the problem
theorem key_lime_yield :
  audrey_key_lime_juice_yield (1/4) 2 16 8 = 1 := 
by 
  sorry

end key_lime_yield_l48_4811


namespace minutes_to_seconds_l48_4841

theorem minutes_to_seconds (m : ℝ) (hm : m = 6.5) : m * 60 = 390 := by
  sorry

end minutes_to_seconds_l48_4841


namespace pages_filled_with_images_ratio_l48_4805

theorem pages_filled_with_images_ratio (total_pages intro_pages text_pages : ℕ) 
  (h_total : total_pages = 98)
  (h_intro : intro_pages = 11)
  (h_text : text_pages = 19)
  (h_blank : 2 * text_pages = total_pages - intro_pages - 2 * text_pages) :
  (total_pages - intro_pages - text_pages - text_pages) / total_pages = 1 / 2 :=
by
  sorry

end pages_filled_with_images_ratio_l48_4805


namespace ball_draw_probability_l48_4875

/-- 
Four balls labeled with numbers 1, 2, 3, 4 are placed in an urn. 
A ball is drawn, its number is recorded, and then the ball is returned to the urn. 
This process is repeated three times. Each ball is equally likely to be drawn on each occasion. 
Given that the sum of the numbers recorded is 7, the probability that the ball numbered 2 was drawn twice is 1/4. 
-/
theorem ball_draw_probability :
  let draws := [(1, 1, 5),(1, 2, 4),(1, 3, 3),(2, 2, 3)]
  (3 / 12 = 1 / 4) :=
by
  sorry

end ball_draw_probability_l48_4875


namespace ball_and_ring_problem_l48_4814

theorem ball_and_ring_problem (x y : ℕ) (m_x m_y : ℕ) : 
  m_x + 2 = y ∧ 
  m_y = x + 2 ∧
  x * m_x + y * m_y - 800 = 2 * (y - x) ∧
  x^2 + y^2 = 881 →
  (x = 25 ∧ y = 16) ∨ (x = 16 ∧ y = 25) := 
by 
  sorry

end ball_and_ring_problem_l48_4814


namespace no_representation_of_216p3_l48_4899

theorem no_representation_of_216p3 (p : ℕ) (hp_prime : Nat.Prime p)
  (hp_form : ∃ m : ℤ, p = 4 * m + 1) : ¬ ∃ x y z : ℤ, 216 * (p ^ 3) = x^2 + y^2 + z^9 := by
  sorry

end no_representation_of_216p3_l48_4899


namespace regular_polygon_exterior_angle_l48_4885

theorem regular_polygon_exterior_angle (n : ℕ) (h : 60 * n = 360) : n = 6 :=
sorry

end regular_polygon_exterior_angle_l48_4885


namespace anoop_joined_after_6_months_l48_4835

theorem anoop_joined_after_6_months (arjun_investment : ℕ) (anoop_investment : ℕ) (months_in_year : ℕ)
  (arjun_time : ℕ) (anoop_time : ℕ) :
  arjun_investment * arjun_time = anoop_investment * anoop_time →
  anoop_investment = 2 * arjun_investment →
  arjun_time = months_in_year →
  anoop_time + arjun_time = months_in_year →
  anoop_time = 6 :=
by sorry

end anoop_joined_after_6_months_l48_4835


namespace monotonicity_of_f_range_of_a_if_no_zeros_l48_4837

noncomputable def f (a x : ℝ) := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_of_f (a : ℝ) (h : a > 0) :
  (∀ x, x > 0 → x < 1/a → deriv (f a) x < 0) ∧
  (∀ x, x > 1/a → deriv (f a) x > 0) := sorry

theorem range_of_a_if_no_zeros 
  (h1 : ∀ x > 0, f a x ≠ 0) : a > 1 / Real.exp 1 := sorry

end monotonicity_of_f_range_of_a_if_no_zeros_l48_4837


namespace x_squared_y_squared_iff_x_squared_y_squared_not_sufficient_x_squared_y_squared_necessary_l48_4869

theorem x_squared_y_squared_iff (x y : ℝ) : x ^ 2 = y ^ 2 ↔ x = y ∨ x = -y := by
  sorry

theorem x_squared_y_squared_not_sufficient (x y : ℝ) : (x ^ 2 = y ^ 2) → (x = y ∨ x = -y) := by
  sorry

theorem x_squared_y_squared_necessary (x y : ℝ) : (x = y) → (x ^ 2 = y ^ 2) := by
  sorry

end x_squared_y_squared_iff_x_squared_y_squared_not_sufficient_x_squared_y_squared_necessary_l48_4869


namespace question1_question2_l48_4824

noncomputable def f (x b c : ℝ) := x^2 + b * x + c

theorem question1 (b c : ℝ) (h : ∀ x : ℝ, 2 * x + b ≤ f x b c) (x : ℝ) (hx : 0 ≤ x) :
  f x b c ≤ (x + c)^2 :=
sorry

theorem question2 (b c m : ℝ) (h : ∀ b c : ℝ, b ≠ c → f c b b - f b b b ≤ m * (c^2 - b^2)) :
  m ≥ 3/2 :=
sorry

end question1_question2_l48_4824


namespace largest_divisor_three_consecutive_l48_4816

theorem largest_divisor_three_consecutive (u v w : ℤ) (h1 : u + 1 = v) (h2 : v + 1 = w) (h3 : ∃ n : ℤ, (u = 5 * n) ∨ (v = 5 * n) ∨ (w = 5 * n)) : 
  ∀ d ∈ {d | ∀ a b c : ℤ, a * b * c = u * v * w → d ∣ a * b * c}, 
  15 ∈ {d | ∀ a b c : ℤ, a * b * c = u * v * w → d ∣ a * b * c} :=
sorry

end largest_divisor_three_consecutive_l48_4816


namespace total_turnips_grown_l48_4879

theorem total_turnips_grown 
  (melanie_turnips : ℕ) 
  (benny_turnips : ℕ) 
  (jack_turnips : ℕ) 
  (lynn_turnips : ℕ) : 
  melanie_turnips = 1395 ∧
  benny_turnips = 11380 ∧
  jack_turnips = 15825 ∧
  lynn_turnips = 23500 → 
  melanie_turnips + benny_turnips + jack_turnips + lynn_turnips = 52100 :=
by
  intros h
  rcases h with ⟨hm, hb, hj, hl⟩
  sorry

end total_turnips_grown_l48_4879


namespace children_on_bus_after_events_l48_4834

-- Definition of the given problem parameters
def initial_children : Nat := 21
def got_off : Nat := 10
def got_on : Nat := 5

-- The theorem we want to prove
theorem children_on_bus_after_events : initial_children - got_off + got_on = 16 :=
by
  -- This is where the proof would go, but we leave it as sorry for now
  sorry

end children_on_bus_after_events_l48_4834


namespace geometric_seq_problem_l48_4865

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n+1) = a n * r

theorem geometric_seq_problem (h_geom : geometric_sequence a) 
  (h_cond : a 8 * a 9 * a 10 = -a 13 ^ 2 ∧ -a 13 ^ 2 = -1000) :
  a 10 * a 12 = 100 * Real.sqrt 10 :=
by
  sorry

end geometric_seq_problem_l48_4865


namespace score_in_first_round_l48_4887

theorem score_in_first_round (cards : List ℕ) (scores : List ℕ) 
  (total_rounds : ℕ) (last_round_score : ℕ) (total_score : ℕ) : 
  cards = [2, 4, 7, 13] ∧ scores = [16, 17, 21, 24] ∧ total_rounds = 3 ∧ last_round_score = 2 ∧ total_score = 16 →
  ∃ first_round_score, first_round_score = 7 := by
  sorry

end score_in_first_round_l48_4887


namespace range_of_a_l48_4840

open Real

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → x^2 + 2 * x - a > 0) → a < 3 :=
by
  sorry

end range_of_a_l48_4840


namespace system_of_equations_solution_l48_4843

theorem system_of_equations_solution :
  ∀ (x y z : ℝ),
  4 * x + 2 * y + z = 20 →
  x + 4 * y + 2 * z = 26 →
  2 * x + y + 4 * z = 28 →
  20 * x^2 + 24 * x * y + 20 * y^2 + 12 * z^2 = 500 :=
by
  intros x y z h1 h2 h3
  sorry

end system_of_equations_solution_l48_4843


namespace percentage_difference_l48_4894

theorem percentage_difference (x : ℝ) : 
  (62 / 100) * 150 - (x / 100) * 250 = 43 → x = 20 :=
by
  intro h
  sorry

end percentage_difference_l48_4894


namespace average_billboards_per_hour_l48_4833

def first_hour_billboards : ℕ := 17
def second_hour_billboards : ℕ := 20
def third_hour_billboards : ℕ := 23

theorem average_billboards_per_hour : 
  (first_hour_billboards + second_hour_billboards + third_hour_billboards) / 3 = 20 := 
by
  sorry

end average_billboards_per_hour_l48_4833


namespace lateral_surface_area_of_cone_l48_4804

theorem lateral_surface_area_of_cone (r l : ℝ) (h₁ : r = 3) (h₂ : l = 5) :
  π * r * l = 15 * π :=
by sorry

end lateral_surface_area_of_cone_l48_4804


namespace seventh_term_value_l48_4847

open Nat

noncomputable def a : ℤ := sorry
noncomputable def d : ℤ := sorry
noncomputable def n : ℤ := sorry

-- Conditions as definitions
def sum_first_five : Prop := 5 * a + 10 * d = 34
def sum_last_five : Prop := 5 * a + 5 * (n - 1) * d = 146
def sum_all_terms : Prop := (n * (2 * a + (n - 1) * d)) / 2 = 234

-- Theorem statement
theorem seventh_term_value :
  sum_first_five ∧ sum_last_five ∧ sum_all_terms → a + 6 * d = 18 :=
by
  sorry

end seventh_term_value_l48_4847


namespace number_of_people_quit_l48_4848

-- Define the conditions as constants.
def initial_team_size : ℕ := 25
def new_members : ℕ := 13
def final_team_size : ℕ := 30

-- Define the question as a function.
def people_quit (Q : ℕ) : Prop :=
  initial_team_size - Q + new_members = final_team_size

-- Prove the main statement assuming the conditions.
theorem number_of_people_quit (Q : ℕ) (h : people_quit Q) : Q = 8 :=
by
  sorry -- Proof is not required, so we use sorry to skip it.

end number_of_people_quit_l48_4848


namespace cake_eaten_fraction_l48_4871

noncomputable def cake_eaten_after_four_trips : ℚ :=
  let consumption_ratio := (1/3 : ℚ)
  let first_trip := consumption_ratio
  let second_trip := consumption_ratio * consumption_ratio
  let third_trip := second_trip * consumption_ratio
  let fourth_trip := third_trip * consumption_ratio
  first_trip + second_trip + third_trip + fourth_trip

theorem cake_eaten_fraction : cake_eaten_after_four_trips = (40 / 81 : ℚ) :=
by
  sorry

end cake_eaten_fraction_l48_4871


namespace math_problem_l48_4846

theorem math_problem (a b c : ℝ) (h1 : a + 2 * b + 3 * c = 12) (h2 : a^2 + b^2 + c^2 = a * b + b * c + c * a) : a + b^2 + c^3 = 14 :=
by
  sorry

end math_problem_l48_4846


namespace ab_value_l48_4898

noncomputable def func (x : ℝ) (a b : ℝ) : ℝ := 4 * x ^ 3 - a * x ^ 2 - 2 * b * x + 2

theorem ab_value 
  (a b : ℝ)
  (h_max : func 1 a b = -3)
  (h_deriv : (12 - 2 * a - 2 * b) = 0) :
  a * b = 9 :=
by
  sorry

end ab_value_l48_4898


namespace find_n_l48_4866

-- Definitions based on conditions
def a := (5 + 10 + 15 + 20 + 25 + 30 + 35) / 7
def b (n : ℕ) := 2 * n

-- Theorem stating the problem
theorem find_n (n : ℕ) (h : a^2 - (b n)^2 = 0) : n = 10 :=
by sorry

end find_n_l48_4866


namespace closest_integer_to_cube_root_of_1728_l48_4872

theorem closest_integer_to_cube_root_of_1728: 
  ∃ n : ℕ, n^3 = 1728 ∧ (∀ m : ℤ, m^3 < 1728 → m < n) ∧ (∀ p : ℤ, p^3 > 1728 → p > n) :=
by
  sorry

end closest_integer_to_cube_root_of_1728_l48_4872


namespace sheets_in_a_bundle_l48_4870

variable (B : ℕ) -- Denotes the number of sheets in a bundle

-- Conditions
variable (NumBundles NumBunches NumHeaps : ℕ)
variable (SheetsPerBunch SheetsPerHeap TotalSheets : ℕ)

-- Definitions of given conditions
def numBundles := 3
def numBunches := 2
def numHeaps := 5
def sheetsPerBunch := 4
def sheetsPerHeap := 20
def totalSheets := 114

-- Theorem to prove
theorem sheets_in_a_bundle :
  3 * B + 2 * sheetsPerBunch + 5 * sheetsPerHeap = totalSheets → B = 2 := by
  intro h
  sorry

end sheets_in_a_bundle_l48_4870


namespace initial_people_count_l48_4821

theorem initial_people_count (C : ℝ) (n : ℕ) (h : n > 1) :
  ((C / (n - 1)) - (C / n) = 0.125) →
  n = 8 := by
  sorry

end initial_people_count_l48_4821


namespace minimum_average_cost_l48_4810

noncomputable def average_cost (x : ℝ) : ℝ :=
  let y := (x^2) / 10 - 30 * x + 4000
  y / x

theorem minimum_average_cost : 
  ∃ (x : ℝ), 150 ≤ x ∧ x ≤ 250 ∧ (∀ (x' : ℝ), 150 ≤ x' ∧ x' ≤ 250 → average_cost x ≤ average_cost x') ∧ average_cost x = 10 := 
by
  sorry

end minimum_average_cost_l48_4810


namespace find_b_l48_4880

theorem find_b
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = (1/12) * x^2 + a * x + b)
  (A C: ℝ × ℝ)
  (hA : A = (x1, 0))
  (hC : C = (x2, 0))
  (T : ℝ × ℝ)
  (hT : T = (3, 3))
  (h_TA : dist (3, 3) (x1, 0) = dist (3, 3) (0, b))
  (h_TB : dist (3, 3) (0, b) = dist (3, 3) (x2, 0))
  (vietas : x1 * x2 = 12 * b)
  : b = -6 := 
sorry

end find_b_l48_4880


namespace exponential_decreasing_iff_frac_inequality_l48_4803

theorem exponential_decreasing_iff_frac_inequality (a : ℝ) :
  (0 < a ∧ a < 1) ↔ (a ≠ 1 ∧ a * (a - 1) ≤ 0) :=
by
  sorry

end exponential_decreasing_iff_frac_inequality_l48_4803


namespace trigonometric_problem_l48_4873

theorem trigonometric_problem (θ : ℝ) (h : Real.tan θ = 2) :
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4 / 5 := by
  sorry

end trigonometric_problem_l48_4873


namespace problem1_problem2_problem3_problem4_l48_4886

theorem problem1 : -15 + (-23) - 26 - (-15) = -49 := 
by sorry

theorem problem2 : (- (1 / 2) + (2 / 3) - (1 / 4)) * (-24) = 2 := 
by sorry

theorem problem3 : -24 / (-6) * (- (1 / 4)) = -1 := 
by sorry

theorem problem4 : -1 ^ 2024 - (-2) ^ 3 - 3 ^ 2 + 2 / (2 / 3 * (3 / 2)) = 5 / 2 := 
by sorry

end problem1_problem2_problem3_problem4_l48_4886


namespace real_part_of_complex_l48_4881

theorem real_part_of_complex (a : ℝ) (h : a^2 + 2 * a - 15 = 0 ∧ a + 5 ≠ 0) : a = 3 :=
by sorry

end real_part_of_complex_l48_4881


namespace complement_of_A_in_U_l48_4895

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 4}

theorem complement_of_A_in_U : (U \ A) = {1, 3, 5} := by
  sorry

end complement_of_A_in_U_l48_4895


namespace greatest_non_fiction_books_l48_4896

def is_prime (p : ℕ) := p > 1 ∧ (∀ d : ℕ, d ∣ p → d = 1 ∨ d = p)

theorem greatest_non_fiction_books (n f k : ℕ) :
  (n + f = 100 ∧ f = n + k ∧ is_prime k) → n ≤ 49 :=
by
  sorry

end greatest_non_fiction_books_l48_4896


namespace pyramid_base_side_length_l48_4851

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (s : ℝ)
  (h_area_lateral_face : area_lateral_face = 144)
  (h_slant_height : slant_height = 24) :
  (1 / 2) * s * slant_height = area_lateral_face → s = 12 :=
by
  sorry

end pyramid_base_side_length_l48_4851


namespace legs_heads_difference_l48_4842

variables (D C L H : ℕ)

theorem legs_heads_difference
    (hC : C = 18)
    (hL : L = 2 * D + 4 * C)
    (hH : H = D + C) :
    L - 2 * H = 36 :=
by
  have h1 : C = 18 := hC
  have h2 : L = 2 * D + 4 * C := hL
  have h3 : H = D + C := hH
  sorry

end legs_heads_difference_l48_4842


namespace problem_statement_l48_4815

theorem problem_statement (n : ℕ) (h : ∀ (a b : ℕ), ¬ (n ∣ (2^a * 3^b + 1))) :
  ∀ (c d : ℕ), ¬ (n ∣ (2^c + 3^d)) := by
  sorry

end problem_statement_l48_4815


namespace fraction_zero_l48_4826

theorem fraction_zero (x : ℝ) (h : (x - 1) * (x + 2) = 0) (hne : x^2 - 1 ≠ 0) : x = -2 :=
by
  sorry

end fraction_zero_l48_4826


namespace xyz_value_l48_4889

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (xy + xz + yz) = 40) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) 
  : x * y * z = 10 :=
sorry

end xyz_value_l48_4889


namespace geometric_series_common_ratio_l48_4839

theorem geometric_series_common_ratio :
  ∀ (a r : ℝ), (r ≠ 1) → 
  (∑' n, a * r^n = 64 * ∑' n, a * r^(n+4)) →
  r = 1 / 2 :=
by
  intros a r hnr heq
  have hsum1 : ∑' n, a * r^n = a / (1 - r) := sorry
  have hsum2 : ∑' n, a * r^(n+4) = a * r^4 / (1 - r) := sorry
  rw [hsum1, hsum2] at heq
  -- Further steps to derive r = 1/2 are omitted
  sorry

end geometric_series_common_ratio_l48_4839


namespace solve_equation_l48_4802

theorem solve_equation (x : ℝ) (h1 : 2 * x + 1 ≠ 0) (h2 : 4 * x ≠ 0) : 
  (3 / (2 * x + 1) = 5 / (4 * x)) ↔ (x = 2.5) :=
by 
  sorry

end solve_equation_l48_4802


namespace bake_sale_money_made_l48_4862

theorem bake_sale_money_made :
  let clementine_cookies := 72
  let jake_cookies := 2 * clementine_cookies
  let combined_cookies := jake_cookies + clementine_cookies
  let tory_cookies := combined_cookies / 2
  let total_cookies := clementine_cookies + jake_cookies + tory_cookies
  let price_per_cookie := 2
  let total_money := total_cookies * price_per_cookie
  total_money = 648 :=
by
  let clementine_cookies := 72
  let jake_cookies := 2 * clementine_cookies
  let combined_cookies := jake_cookies + clementine_cookies
  let tory_cookies := combined_cookies / 2
  let total_cookies := clementine_cookies + jake_cookies + tory_cookies
  let price_per_cookie := 2
  let total_money := total_cookies * price_per_cookie
  sorry

end bake_sale_money_made_l48_4862


namespace N_is_even_l48_4813

def sum_of_digits : ℕ → ℕ := sorry

theorem N_is_even 
  (N : ℕ)
  (h1 : sum_of_digits N = 100)
  (h2 : sum_of_digits (5 * N) = 50) : 
  Even N :=
sorry

end N_is_even_l48_4813


namespace number_of_students_l48_4844

theorem number_of_students (x : ℕ) (h : x * (x - 1) = 210) : x = 15 := 
by sorry

end number_of_students_l48_4844


namespace unique_integer_cube_triple_l48_4876

theorem unique_integer_cube_triple (x : ℤ) (h : x^3 < 3 * x) : x = 1 := 
sorry

end unique_integer_cube_triple_l48_4876


namespace select_student_D_l48_4829

-- Define the scores and variances based on the conditions
def avg_A : ℝ := 96
def avg_B : ℝ := 94
def avg_C : ℝ := 93
def avg_D : ℝ := 96

def var_A : ℝ := 1.2
def var_B : ℝ := 1.2
def var_C : ℝ := 0.6
def var_D : ℝ := 0.4

-- Proof statement in Lean 4
theorem select_student_D (avg_A avg_B avg_C avg_D var_A var_B var_C var_D : ℝ) 
                         (h_avg_A : avg_A = 96)
                         (h_avg_B : avg_B = 94)
                         (h_avg_C : avg_C = 93)
                         (h_avg_D : avg_D = 96)
                         (h_var_A : var_A = 1.2)
                         (h_var_B : var_B = 1.2)
                         (h_var_C : var_C = 0.6)
                         (h_var_D : var_D = 0.4) 
                         (h_D_highest_avg : avg_D = max avg_A (max avg_B (max avg_C avg_D)))
                         (h_D_lowest_var : var_D = min (min (min var_A var_B) var_C) var_D) :
  avg_D = 96 ∧ var_D = 0.4 := 
by 
  -- As we're not asked to prove, we put sorry here to indicate the proof step is omitted.
  sorry

end select_student_D_l48_4829


namespace insects_legs_l48_4845

theorem insects_legs (L N : ℕ) (hL : L = 54) (hN : N = 9) : (L / N = 6) :=
by sorry

end insects_legs_l48_4845


namespace geom_sequence_ratio_l48_4849

-- Definitions and assumptions for the problem
noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, 0 < r ∧ r < 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geom_sequence_ratio (a : ℕ → ℝ) (r : ℝ) 
  (h_geom: geom_seq a)
  (h_r: 0 < r ∧ r < 1)
  (h_seq: ∀ n : ℕ, a (n + 1) = a n * r)
  (ha1: a 7 * a 14 = 6)
  (ha2: a 4 + a 17 = 5) :
  (a 5 / a 18) = (3 / 2) :=
sorry

end geom_sequence_ratio_l48_4849


namespace frac_eq_three_l48_4860

theorem frac_eq_three (a b c : ℝ) 
  (h₁ : a / b = 4 / 3) (h₂ : (a + c) / (b - c) = 5 / 2) : 
  (3 * a + 2 * b) / (3 * a - 2 * b) = 3 :=
by
  sorry

end frac_eq_three_l48_4860


namespace arithmetic_square_root_of_16_is_4_l48_4864

theorem arithmetic_square_root_of_16_is_4 : ∃ x : ℤ, x * x = 16 ∧ x = 4 := 
sorry

end arithmetic_square_root_of_16_is_4_l48_4864


namespace problem1_problem2_l48_4874

theorem problem1 (x : ℝ) : (x + 3) * (x - 1) ≤ 0 ↔ -3 ≤ x ∧ x ≤ 1 :=
sorry

theorem problem2 (x : ℝ) : (x + 2) * (x - 3) > 0 ↔ x < -2 ∨ x > 3 :=
sorry

end problem1_problem2_l48_4874


namespace arithmetic_sequence_properties_l48_4819

theorem arithmetic_sequence_properties 
  (a : ℕ → ℤ) 
  (h1 : a 1 + a 2 + a 3 = 21) 
  (h2 : a 1 * a 2 * a 3 = 231) :
  (a 2 = 7) ∧ (∀ n, a n = -4 * n + 15 ∨ a n = 4 * n - 1) := 
by
  sorry

end arithmetic_sequence_properties_l48_4819


namespace roots_satisfy_conditions_l48_4867

variable (a x1 x2 : ℝ)

-- Conditions
def condition1 : Prop := x1 * x2 + x1 + x2 - a = 0
def condition2 : Prop := x1 * x2 - a * (x1 + x2) + 1 = 0

-- Derived quadratic equation
def quadratic_eq : Prop := ∃ x : ℝ, x^2 - x + (a - 1) = 0

theorem roots_satisfy_conditions (h1: condition1 a x1 x2) (h2: condition2 a x1 x2) : quadratic_eq a :=
  sorry

end roots_satisfy_conditions_l48_4867


namespace children_less_than_adults_l48_4893

theorem children_less_than_adults (total_members : ℕ)
  (percent_adults : ℝ) (percent_teenagers : ℝ) (percent_children : ℝ) :
  total_members = 500 →
  percent_adults = 0.45 →
  percent_teenagers = 0.25 →
  percent_children = 1 - percent_adults - percent_teenagers →
  (percent_children * total_members) - (percent_adults * total_members) = -75 := 
by
  intros h_total h_adults h_teenagers h_children
  sorry

end children_less_than_adults_l48_4893


namespace students_didnt_make_cut_l48_4854

theorem students_didnt_make_cut (g b c : ℕ) (hg : g = 15) (hb : b = 25) (hc : c = 7) : g + b - c = 33 := by
  sorry

end students_didnt_make_cut_l48_4854


namespace total_bill_correct_l48_4882

def scoop_cost : ℕ := 2
def pierre_scoops : ℕ := 3
def mom_scoops : ℕ := 4

def pierre_total : ℕ := pierre_scoops * scoop_cost
def mom_total : ℕ := mom_scoops * scoop_cost
def total_bill : ℕ := pierre_total + mom_total

theorem total_bill_correct : total_bill = 14 :=
by
  sorry

end total_bill_correct_l48_4882


namespace find_plaid_shirts_l48_4888

def total_shirts : ℕ := 5
def total_pants : ℕ := 24
def total_items : ℕ := total_shirts + total_pants
def neither_plaid_nor_purple : ℕ := 21
def total_plaid_or_purple : ℕ := total_items - neither_plaid_nor_purple
def purple_pants : ℕ := 5
def plaid_shirts (p : ℕ) : Prop := total_plaid_or_purple - purple_pants = p

theorem find_plaid_shirts : plaid_shirts 3 := by
  unfold plaid_shirts
  repeat { sorry }

end find_plaid_shirts_l48_4888


namespace eric_running_time_l48_4836

-- Define the conditions
variables (jog_time to_park_time return_time : ℕ)
axiom jog_time_def : jog_time = 10
axiom return_time_def : return_time = 90
axiom trip_relation : return_time = 3 * to_park_time

-- Define the question
def run_time : ℕ := to_park_time - jog_time

-- State the problem: Prove that given the conditions, the running time is 20 minutes.
theorem eric_running_time : run_time = 20 :=
by
  -- Proof goes here
  sorry

end eric_running_time_l48_4836
