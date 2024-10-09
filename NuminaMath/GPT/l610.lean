import Mathlib

namespace rectangle_area_given_conditions_l610_61075

theorem rectangle_area_given_conditions
  (l w : ℝ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by sorry

end rectangle_area_given_conditions_l610_61075


namespace parabola_property_l610_61002

-- Define the conditions of the problem in Lean
variable (a b : ℝ)
variable (h1 : (a, b) ∈ {p : ℝ × ℝ | p.1^2 = 20 * p.2}) -- P lies on the parabola x^2 = 20y
variable (h2 : dist (a, b) (0, 5) = 25) -- Distance from P to focus F

theorem parabola_property : |a * b| = 400 := by
  sorry

end parabola_property_l610_61002


namespace max_value_at_2_l610_61063

noncomputable def f (x : ℝ) : ℝ := -x^3 + 12 * x

theorem max_value_at_2 : ∃ a : ℝ, (∀ x : ℝ, f x ≤ f a) ∧ a = 2 := 
by
  sorry

end max_value_at_2_l610_61063


namespace first_term_geometric_series_l610_61023

theorem first_term_geometric_series (a1 q : ℝ) (h1 : a1 / (1 - q) = 1)
  (h2 : |a1| / (1 - |q|) = 2) (h3 : -1 < q) (h4 : q < 1) (h5 : q ≠ 0) :
  a1 = 4 / 3 :=
by {
  sorry
}

end first_term_geometric_series_l610_61023


namespace pens_purchased_is_30_l610_61098

def num_pens_purchased (cost_total: ℕ) 
                       (num_pencils: ℕ) 
                       (price_per_pencil: ℚ) 
                       (price_per_pen: ℚ)
                       (expected_pens: ℕ): Prop :=
   let cost_pencils := num_pencils * price_per_pencil
   let cost_pens := cost_total - cost_pencils
   let num_pens := cost_pens / price_per_pen
   num_pens = expected_pens

theorem pens_purchased_is_30 : num_pens_purchased 630 75 2.00 16 30 :=
by
  -- Unfold the definition manually if needed
  sorry

end pens_purchased_is_30_l610_61098


namespace jill_travels_less_than_john_l610_61032

theorem jill_travels_less_than_john :
  ∀ (John Jill Jim : ℕ), 
  John = 15 → 
  Jim = 2 → 
  (Jim = (20 / 100) * Jill) → 
  (John - Jill) = 5 := 
by
  intros John Jill Jim HJohn HJim HJimJill
  -- Skip the proof for now
  sorry

end jill_travels_less_than_john_l610_61032


namespace part_1_part_2_l610_61096

noncomputable def f (a m x : ℝ) := a ^ m / x

theorem part_1 (a : ℝ) (m : ℝ) (H1 : a > 1) (H2 : ∀ x, x ∈ Set.Icc a (2*a) → f a m x ∈ Set.Icc (a^2) (a^3)) :
  a = 2 :=
sorry

theorem part_2 (t : ℝ) (s : ℝ) (H1 : ∀ x, x ∈ Set.Icc 0 s → (x + t) ^ 2 + 2 * (x + t) ≤ 3 * x) :
  s ∈ Set.Ioc 0 5 :=
sorry

end part_1_part_2_l610_61096


namespace smallest_n_for_terminating_decimal_l610_61058

theorem smallest_n_for_terminating_decimal :
  ∃ (n : ℕ), (∀ m, m < n → (∃ k1 k2 : ℕ, (m + 150 = 2^k1 * 5^k2 ∧ m > 0) → false)) ∧ (∃ k1 k2 : ℕ, (n + 150 = 2^k1 * 5^k2) ∧ n > 0) :=
sorry

end smallest_n_for_terminating_decimal_l610_61058


namespace pages_read_tonight_l610_61030

def sum_of_digits (n: ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem pages_read_tonight :
  let pages_3_nights_ago := 20
  let pages_2_nights_ago := 20^2 + 5
  let pages_last_night := sum_of_digits pages_2_nights_ago * 3
  let total_pages := 500
  total_pages - (pages_3_nights_ago + pages_2_nights_ago + pages_last_night) = 48 :=
by
  sorry

end pages_read_tonight_l610_61030


namespace pigeonhole_principle_f_m_l610_61060

theorem pigeonhole_principle_f_m :
  ∀ (n : ℕ) (f : ℕ × ℕ → Fin (n + 1)), n ≤ 44 →
    ∃ (i j l k p m : ℕ),
      1989 * m ≤ i ∧ i < l ∧ l < 1989 + 1989 * m ∧
      1989 * p ≤ j ∧ j < k ∧ k < 1989 + 1989 * p ∧
      f (i, j) = f (i, k) ∧ f (i, k) = f (l, j) ∧ f (l, j) = f (l, k) :=
by {
  sorry
}

end pigeonhole_principle_f_m_l610_61060


namespace integer_solutions_zero_l610_61042

theorem integer_solutions_zero (x y u t : ℤ) :
  x^2 + y^2 = 1974 * (u^2 + t^2) → 
  x = 0 ∧ y = 0 ∧ u = 0 ∧ t = 0 :=
by
  sorry

end integer_solutions_zero_l610_61042


namespace james_tv_watching_time_l610_61050

theorem james_tv_watching_time
  (ep_jeopardy : ℕ := 20) -- Each episode of Jeopardy is 20 minutes long
  (n_jeopardy : ℕ := 2) -- James watched 2 episodes of Jeopardy
  (n_wheel : ℕ := 2) -- James watched 2 episodes of Wheel of Fortune
  (wheel_factor : ℕ := 2) -- Wheel of Fortune episodes are twice as long as Jeopardy episodes
  : (ep_jeopardy * n_jeopardy + ep_jeopardy * wheel_factor * n_wheel) / 60 = 2 :=
by
  sorry

end james_tv_watching_time_l610_61050


namespace parabola_range_m_l610_61010

noncomputable def parabola (m : ℝ) (x : ℝ) : ℝ := x^2 - (4*m + 1)*x + (2*m - 1)

theorem parabola_range_m (m : ℝ) :
  (∀ x : ℝ, parabola m x = 0 → (1 < x ∧ x < 2) ∨ (x < 1 ∨ x > 2)) ∧
  parabola m 0 < -1/2 →
  1/6 < m ∧ m < 1/4 :=
by
  sorry

end parabola_range_m_l610_61010


namespace consecutive_numbers_equation_l610_61047

theorem consecutive_numbers_equation (x y z : ℤ) (h1 : z = 3) (h2 : y = z + 1) (h3 : x = y + 1) 
(h4 : 2 * x + 3 * y + 3 * z = 5 * y + n) : n = 11 :=
by
  sorry

end consecutive_numbers_equation_l610_61047


namespace find_central_angle_of_sector_l610_61052

variables (r θ : ℝ)

def sector_arc_length (r θ : ℝ) := r * θ
def sector_area (r θ : ℝ) := 0.5 * r^2 * θ

theorem find_central_angle_of_sector
  (l : ℝ)
  (A : ℝ)
  (hl : l = sector_arc_length r θ)
  (hA : A = sector_area r θ)
  (hl_val : l = 4)
  (hA_val : A = 2) :
  θ = 4 :=
sorry

end find_central_angle_of_sector_l610_61052


namespace sugar_water_inequality_one_sugar_water_inequality_two_l610_61065

variable (a b m : ℝ)

-- Condition constraints
variable (h1 : 0 < a) (h2 : a < b) (h3 : 0 < m)

-- Sugar Water Experiment One Inequality
theorem sugar_water_inequality_one : a / b > a / (b + m) := 
by
  sorry

-- Sugar Water Experiment Two Inequality
theorem sugar_water_inequality_two : a / b < (a + m) / b := 
by
  sorry

end sugar_water_inequality_one_sugar_water_inequality_two_l610_61065


namespace arithmetic_sequence_nth_term_l610_61076

theorem arithmetic_sequence_nth_term (S : ℕ → ℕ) (h : ∀ n, S n = 5 * n + 4 * n^2) (r : ℕ) : 
  S r - S (r - 1) = 8 * r + 1 := 
by
  sorry

end arithmetic_sequence_nth_term_l610_61076


namespace three_digit_numbers_count_l610_61008

def number_of_3_digit_numbers : ℕ := 
  let without_zero := 2 * Nat.choose 9 3
  let with_zero := Nat.choose 9 2
  without_zero + with_zero

theorem three_digit_numbers_count : number_of_3_digit_numbers = 204 := by
  -- Proof to be completed
  sorry

end three_digit_numbers_count_l610_61008


namespace total_games_is_272_l610_61069

-- Define the number of players
def n : ℕ := 17

-- Define the formula for the number of games played
def total_games (n : ℕ) : ℕ := n * (n - 1)

-- Define a theorem stating that the total games played is 272
theorem total_games_is_272 : total_games n = 272 := by
  -- Proof omitted
  sorry

end total_games_is_272_l610_61069


namespace sum_of_integers_l610_61040

theorem sum_of_integers (s : Finset ℕ) (h₀ : ∀ a ∈ s, 0 ≤ a ∧ a ≤ 124)
  (h₁ : ∀ a ∈ s, a^3 % 125 = 2) : s.sum id = 265 :=
sorry

end sum_of_integers_l610_61040


namespace smallest_number_divisible_by_20_and_36_is_180_l610_61003

theorem smallest_number_divisible_by_20_and_36_is_180 :
  ∃ x, (x % 20 = 0) ∧ (x % 36 = 0) ∧ (∀ y, (y % 20 = 0) ∧ (y % 36 = 0) → x ≤ y) ∧ x = 180 :=
by
  sorry

end smallest_number_divisible_by_20_and_36_is_180_l610_61003


namespace ratio_children_to_adults_l610_61043

variable (male_adults : ℕ) (female_adults : ℕ) (total_people : ℕ)
variable (total_adults : ℕ) (children : ℕ)

theorem ratio_children_to_adults :
  male_adults = 100 →
  female_adults = male_adults + 50 →
  total_people = 750 →
  total_adults = male_adults + female_adults →
  children = total_people - total_adults →
  children / total_adults = 2 :=
by
  intros h_male h_female h_total h_adults h_children
  sorry

end ratio_children_to_adults_l610_61043


namespace find_a2016_l610_61035

theorem find_a2016 (a : ℕ → ℤ) (h1 : a 1 = 4) (h2 : a 2 = 6) (h3 : ∀ n : ℕ, a (n + 2) = a (n + 1) - a n) : a 2016 = -2 := 
by sorry

end find_a2016_l610_61035


namespace fruit_cost_l610_61041

theorem fruit_cost:
  let strawberry_cost := 2.20
  let cherry_cost := 6 * strawberry_cost
  let blueberry_cost := cherry_cost / 2
  let strawberries_count := 3
  let cherries_count := 4.5
  let blueberries_count := 6.2
  let total_cost := (strawberries_count * strawberry_cost) + (cherries_count * cherry_cost) + (blueberries_count * blueberry_cost)
  total_cost = 106.92 :=
by
  sorry

end fruit_cost_l610_61041


namespace Ed_more_marbles_than_Doug_l610_61095

-- Definitions based on conditions
def Ed_marbles_initial : ℕ := 45
def Doug_loss : ℕ := 11
def Doug_marbles_initial : ℕ := Ed_marbles_initial - 10
def Doug_marbles_after_loss : ℕ := Doug_marbles_initial - Doug_loss

-- Theorem statement
theorem Ed_more_marbles_than_Doug :
  Ed_marbles_initial - Doug_marbles_after_loss = 21 :=
by
  -- Proof would go here
  sorry

end Ed_more_marbles_than_Doug_l610_61095


namespace right_triangle_area_l610_61036

/-- Given a right triangle where one leg is 18 cm and the hypotenuse is 30 cm,
    prove that the area of the triangle is 216 square centimeters. -/
theorem right_triangle_area (a b c : ℝ) 
    (ha : a = 18) 
    (hc : c = 30) 
    (h_right : a^2 + b^2 = c^2) :
    (1 / 2) * a * b = 216 :=
by
  -- Substitute the values given and solve the area.
  sorry

end right_triangle_area_l610_61036


namespace sum_of_consecutive_integers_product_336_l610_61079

theorem sum_of_consecutive_integers_product_336 :
  ∃ (x y : ℕ), x * (x + 1) = 336 ∧ (y - 1) * y * (y + 1) = 336 ∧ x + (x + 1) + (y - 1) + y + (y + 1) = 54 :=
by
  -- The formal proof would go here
  sorry

end sum_of_consecutive_integers_product_336_l610_61079


namespace find_integer_n_l610_61034

theorem find_integer_n (n : ℤ) (h : (⌊(n^2 : ℤ)/4⌋ - (⌊n/2⌋)^2 = 2)) : n = 5 :=
sorry

end find_integer_n_l610_61034


namespace complement_intersection_eq_l610_61062

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {1, 2, 5}) (hB : B = {1, 3, 4})

theorem complement_intersection_eq :
  (U \ A) ∩ B = {3, 4} :=
by
  rw [hU, hA, hB]
  sorry

end complement_intersection_eq_l610_61062


namespace ratio_trumpet_to_running_l610_61028

def basketball_hours := 10
def running_hours := 2 * basketball_hours
def trumpet_hours := 40

theorem ratio_trumpet_to_running : (trumpet_hours : ℚ) / running_hours = 2 :=
by
  sorry

end ratio_trumpet_to_running_l610_61028


namespace quadratic_eq_standard_form_coefficients_l610_61018

-- Define initial quadratic equation
def initial_eq (x : ℝ) : Prop := (x + 5) * (x + 3) = 2 * x^2

-- Define the quadratic equation in standard form
def standard_form (x : ℝ) : Prop := x^2 - 8 * x - 15 = 0

-- Prove that given the initial equation, it can be converted to its standard form
theorem quadratic_eq_standard_form (x : ℝ) :
  initial_eq x → standard_form x := 
sorry

-- Verify the coefficients of the quadratic term, linear term, and constant term
theorem coefficients (x : ℝ) :
  initial_eq x → 
  (∀ a b c : ℝ, (a = 1) ∧ (b = -8) ∧ (c = -15) → standard_form x) :=
sorry

end quadratic_eq_standard_form_coefficients_l610_61018


namespace power_eq_l610_61031

theorem power_eq (a b c : ℝ) (h₁ : a = 81) (h₂ : b = 4 / 3) : (a ^ b) = 243 * (3 ^ (1 / 3)) := by
  sorry

end power_eq_l610_61031


namespace men_left_bus_l610_61068

theorem men_left_bus (M W : ℕ) (initial_passengers : M + W = 72) 
  (women_half_men : W = M / 2) 
  (equal_men_women_after_changes : ∃ men_left : ℕ, ∀ W_new, W_new = W + 8 → M - men_left = W_new → M - men_left = 32) :
  ∃ men_left : ℕ, men_left = 16 :=
  sorry

end men_left_bus_l610_61068


namespace aunt_gemma_dog_food_l610_61013

theorem aunt_gemma_dog_food :
  ∀ (dogs : ℕ) (grams_per_meal : ℕ) (meals_per_day : ℕ) (sack_kg : ℕ) (days : ℕ), 
    dogs = 4 →
    grams_per_meal = 250 →
    meals_per_day = 2 →
    sack_kg = 50 →
    days = 50 →
    (dogs * meals_per_day * grams_per_meal * days) / (1000 * sack_kg) = 2 :=
by
  intros dogs grams_per_meal meals_per_day sack_kg days
  intros h_dogs h_grams_per_meal h_meals_per_day h_sack_kg h_days
  sorry

end aunt_gemma_dog_food_l610_61013


namespace minimum_value_l610_61019

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃z, z = (x^2 + y^2) / (x + y)^2 ∧ z ≥ 1/2 := 
sorry

end minimum_value_l610_61019


namespace number_of_connections_l610_61059

theorem number_of_connections (n : ℕ) (d : ℕ) (h₀ : n = 40) (h₁ : d = 4) : 
  (n * d) / 2 = 80 :=
by
  sorry

end number_of_connections_l610_61059


namespace max_members_in_band_l610_61066

theorem max_members_in_band (m : ℤ) (h1 : 30 * m % 31 = 6) (h2 : 30 * m < 1200) : 30 * m = 360 :=
by {
  sorry -- Proof steps are not required according to the procedure
}

end max_members_in_band_l610_61066


namespace ratio_50kg_to_05tons_not_100_to_1_l610_61091

theorem ratio_50kg_to_05tons_not_100_to_1 (weight1 : ℕ) (weight2 : ℕ) (r : ℕ) 
  (h1 : weight1 = 50) (h2 : weight2 = 500) (h3 : r = 100) : ¬ (weight1 * r = weight2) := 
by
  sorry

end ratio_50kg_to_05tons_not_100_to_1_l610_61091


namespace find_aa_l610_61025

-- Given conditions
def m : ℕ := 7

-- Definition for checking if a number's tens place is 1
def tens_place_one (n : ℕ) : Prop :=
  (n / 10) % 10 = 1

-- The main statement to prove
theorem find_aa : ∃ x : ℕ, x < 10 ∧ tens_place_one (m * x^3) ∧ x = 6 := by
  -- Proof would go here
  sorry

end find_aa_l610_61025


namespace surface_area_of_box_l610_61086

def cube_edge_length : ℕ := 1
def cubes_required : ℕ := 12

theorem surface_area_of_box (l w h : ℕ) (h1 : l * w * h = cubes_required / cube_edge_length ^ 3) :
  (2 * (l * w + w * h + h * l) = 32 ∨ 2 * (l * w + w * h + h * l) = 38 ∨ 2 * (l * w + w * h + h * l) = 40) :=
  sorry

end surface_area_of_box_l610_61086


namespace dawn_hours_l610_61071

-- Define the conditions
def pedestrian_walked_from_A_to_B (x : ℕ) : Prop :=
  x > 0

def pedestrian_walked_from_B_to_A (x : ℕ) : Prop :=
  x > 0

def met_at_noon (x : ℕ) : Prop :=
  x > 0

def arrived_at_B_at_4pm (x : ℕ) : Prop :=
  x > 0

def arrived_at_A_at_9pm (x : ℕ) : Prop :=
  x > 0

-- Define the theorem to prove
theorem dawn_hours (x : ℕ) :
  pedestrian_walked_from_A_to_B x ∧ 
  pedestrian_walked_from_B_to_A x ∧
  met_at_noon x ∧ 
  arrived_at_B_at_4pm x ∧ 
  arrived_at_A_at_9pm x → 
  x = 6 := 
sorry

end dawn_hours_l610_61071


namespace complex_mul_l610_61029

theorem complex_mul (i : ℂ) (h : i^2 = -1) :
    (1 - i) * (1 + 2 * i) = 3 + i :=
by
  sorry

end complex_mul_l610_61029


namespace find_the_added_number_l610_61081

theorem find_the_added_number (n : ℤ) : (1 + n) / (3 + n) = 3 / 4 → n = 5 :=
  sorry

end find_the_added_number_l610_61081


namespace minimum_value_is_one_l610_61061

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
  (1 / (3 * a + 2)) + (1 / (3 * b + 2)) + (1 / (3 * c + 2))

theorem minimum_value_is_one (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  minimum_value a b c = 1 := by
  sorry

end minimum_value_is_one_l610_61061


namespace range_of_m_l610_61006

variable (m : ℝ)

def p : Prop := (m^2 - 4 > 0) ∧ (m > 0)
def q : Prop := 16 * (m - 2)^2 - 16 < 0

theorem range_of_m :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → (1 < m ∧ m ≤ 2) ∨ (3 ≤ m) :=
by
  intro h
  sorry

end range_of_m_l610_61006


namespace polynomial_bound_implies_l610_61078

theorem polynomial_bound_implies :
  (∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) →
  (∀ x : ℝ, |x| ≤ 1 → |c * x^2 + b * x + a| ≤ 2) :=
by
  sorry

end polynomial_bound_implies_l610_61078


namespace value_of_expression_when_x_eq_4_l610_61085

theorem value_of_expression_when_x_eq_4 : (3 * 4 + 4)^2 = 256 := by
  sorry

end value_of_expression_when_x_eq_4_l610_61085


namespace find_B_l610_61083

theorem find_B (A B : ℕ) (h : 5 * 100 + 10 * A + 8 - (B * 100 + 14) = 364) : B = 2 :=
sorry

end find_B_l610_61083


namespace sum_infinite_geometric_series_l610_61012

theorem sum_infinite_geometric_series :
  let a := 1
  let r := (1 : ℝ) / 3
  ∑' (n : ℕ), a * r ^ n = (3 : ℝ) / 2 :=
by
  sorry

end sum_infinite_geometric_series_l610_61012


namespace main_theorem_l610_61005

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem main_theorem :
  (∀ x : ℝ, f (x + 5/2) + f x = 2) ∧
  (∀ x : ℝ, f (1 + 2*x) = f (1 - 2*x)) ∧
  (∀ x : ℝ, g (x + 2) = g (x - 2)) ∧
  (∀ x : ℝ, g (-x + 1) - 1 = -g (x + 1) + 1) ∧
  (∀ x : ℝ, 0 ≤ x → x ≤ 2 → f x + g x = 3^x + x^3) →
  f 2022 * g 2022 = 72 :=
sorry

end main_theorem_l610_61005


namespace rhombus_obtuse_angle_l610_61088

theorem rhombus_obtuse_angle (perimeter height : ℝ) (h_perimeter : perimeter = 8) (h_height : height = 1) : 
  ∃ θ : ℝ, θ = 150 :=
by
  sorry

end rhombus_obtuse_angle_l610_61088


namespace compute_pairs_a_b_l610_61004

noncomputable def f (x a b : ℝ) : ℝ := (x + a) / (x + b)

theorem compute_pairs_a_b (a b : ℝ) (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -b) :
  ((∀ x, f (f x a b) a b = -1 / x) ↔ (a = -1 ∧ b = 1)) :=
sorry

end compute_pairs_a_b_l610_61004


namespace sqrt_of_S_l610_61017

def initial_time := 16 * 3600 + 11 * 60 + 22
def initial_date := 16
def total_seconds_in_a_day := 86400
def total_seconds_in_an_hour := 3600

theorem sqrt_of_S (S : ℕ) (hS : S = total_seconds_in_a_day + total_seconds_in_an_hour) : 
  Real.sqrt S = 300 := 
sorry

end sqrt_of_S_l610_61017


namespace question_a_gt_b_neither_sufficient_nor_necessary_l610_61015

theorem question_a_gt_b_neither_sufficient_nor_necessary (a b : ℝ) :
  ¬ ((a > b → a^2 > b^2) ∧ (a^2 > b^2 → a > b)) :=
by
  sorry

end question_a_gt_b_neither_sufficient_nor_necessary_l610_61015


namespace fraction_is_terminating_decimal_l610_61027

noncomputable def fraction_to_decimal : ℚ :=
  58 / 160

theorem fraction_is_terminating_decimal : fraction_to_decimal = 3625 / 10000 :=
by
  sorry

end fraction_is_terminating_decimal_l610_61027


namespace carina_coffee_l610_61064

def total_coffee (t f : ℕ) : ℕ := 10 * t + 5 * f

theorem carina_coffee (t : ℕ) (h1 : t = 3) (f : ℕ) (h2 : f = t + 2) : total_coffee t f = 55 := by
  sorry

end carina_coffee_l610_61064


namespace least_weight_of_oranges_l610_61020

theorem least_weight_of_oranges :
  ∀ (a o : ℝ), (a ≥ 8 + 3 * o) → (a ≤ 4 * o) → (o ≥ 8) :=
by
  intros a o h1 h2
  sorry

end least_weight_of_oranges_l610_61020


namespace tangent_line_ellipse_l610_61093

variable {a b x x0 y y0 : ℝ}

theorem tangent_line_ellipse (h : a * x0^2 + b * y0^2 = 1) :
  a * x0 * x + b * y0 * y = 1 :=
sorry

end tangent_line_ellipse_l610_61093


namespace graduation_photo_arrangement_l610_61049

theorem graduation_photo_arrangement (teachers middle_positions other_students : Finset ℕ) (A B : ℕ) :
  teachers.card = 2 ∧ middle_positions.card = 2 ∧ 
  (other_students ∪ {A, B}).card = 4 ∧ ∀ t ∈ teachers, t ∈ middle_positions →
  ∃ arrangements : ℕ, arrangements = 8 :=
by
  sorry

end graduation_photo_arrangement_l610_61049


namespace percent_of_number_l610_61070

theorem percent_of_number (x : ℝ) (hx : (120 / x) = (75 / 100)) : x = 160 := 
sorry

end percent_of_number_l610_61070


namespace equal_students_initially_l610_61011

theorem equal_students_initially (B G : ℕ) (h1 : B = G) (h2 : B = 2 * (G - 8)) : B + G = 32 :=
by
  sorry

end equal_students_initially_l610_61011


namespace total_food_needed_l610_61045

-- Definitions for the conditions
def horses : ℕ := 4
def oats_per_meal : ℕ := 4
def oats_meals_per_day : ℕ := 2
def grain_per_day : ℕ := 3
def days : ℕ := 3

-- Theorem stating the problem
theorem total_food_needed :
  (horses * (days * (oats_per_meal * oats_meals_per_day) + days * grain_per_day)) = 132 :=
by sorry

end total_food_needed_l610_61045


namespace min_bound_of_gcd_condition_l610_61022

theorem min_bound_of_gcd_condition :
  ∃ c > 0, ∀ a b n : ℕ, 0 < a ∧ 0 < b ∧ 0 < n ∧
  (∀ i j : ℕ, i ≤ n ∧ j ≤ n → Nat.gcd (a + i) (b + j) > 1) →
  min a b > (c * n) ^ (n / 2) :=
sorry

end min_bound_of_gcd_condition_l610_61022


namespace hermione_utility_l610_61051

theorem hermione_utility (h : ℕ) : (h * (10 - h) = (4 - h) * (h + 2)) ↔ h = 4 := by
  sorry

end hermione_utility_l610_61051


namespace probability_not_grade_5_l610_61038

theorem probability_not_grade_5 :
  let A1 := 0.3
  let A2 := 0.4
  let A3 := 0.2
  let A4 := 0.1
  (A1 + A2 + A3 + A4 = 1) → (1 - A1 = 0.7) := by
  intros A1_def A2_def A3_def A4_def h
  sorry

end probability_not_grade_5_l610_61038


namespace expression_for_f_general_formula_a_n_sum_S_n_l610_61000

-- Definitions for conditions
def f (x : ℝ) : ℝ := x^2 + x

-- Given conditions
axiom f_zero : f 0 = 0
axiom f_recurrence : ∀ x : ℝ, f (x + 1) - f x = x + 1

-- Statements to prove
theorem expression_for_f (x : ℝ) : f x = x^2 + x := 
sorry

theorem general_formula_a_n (t : ℝ) (n : ℕ) (H : 0 < t) : 
    ∃ a_n : ℕ → ℝ, a_n n = t^n := 
sorry

theorem sum_S_n (t : ℝ) (n : ℕ) (H : 0 < t) :
    ∃ S_n : ℕ → ℝ, (S_n n = if t = 1 then ↑n else (t * (t^n - 1)) / (t - 1)) := 
sorry

end expression_for_f_general_formula_a_n_sum_S_n_l610_61000


namespace math_problem_l610_61007

def Q (f : ℝ → ℝ) : Prop :=
  (∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → x + y ≠ 0 → f (1 / (x + y)) = f (1 / x) + f (1 / y))
  ∧ (∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → x + y ≠ 0 → (x + y) * f (x + y) = x * y * f x * f y)
  ∧ f 1 = 1

theorem math_problem (f : ℝ → ℝ) : Q f → (∀ (x : ℝ), x ≠ 0 → f x = 1 / x) :=
by
  -- Proof goes here
  sorry

end math_problem_l610_61007


namespace calculate_expression_l610_61053

theorem calculate_expression : abs (-2) - Real.sqrt 4 + 3^2 = 9 := by
  sorry

end calculate_expression_l610_61053


namespace sqrt_identity_l610_61072

theorem sqrt_identity (x : ℝ) (hx : x = Real.sqrt 5 - 3) : Real.sqrt (x^2 + 6*x + 9) = Real.sqrt 5 :=
by
  sorry

end sqrt_identity_l610_61072


namespace simplify_fraction_l610_61001

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : (x^2 / (x - 1)) - (1 / (x - 1)) = x + 1 :=
by
  sorry

end simplify_fraction_l610_61001


namespace eq_margin_l610_61067

variables (C S n : ℝ) (M : ℝ)

theorem eq_margin (h : M = 1 / n * (2 * C - S)) : M = S / (n + 2) :=
sorry

end eq_margin_l610_61067


namespace total_difference_is_18_l610_61021

-- Define variables for Mike, Joe, and Anna's bills
variables (m j a : ℝ)

-- Define the conditions given in the problem
def MikeTipped := (0.15 * m = 3)
def JoeTipped := (0.25 * j = 3)
def AnnaTipped := (0.10 * a = 3)

-- Prove the total amount of money that was different between the highest and lowest bill is 18
theorem total_difference_is_18 (MikeTipped : 0.15 * m = 3) (JoeTipped : 0.25 * j = 3) (AnnaTipped : 0.10 * a = 3) :
  |a - j| = 18 := 
sorry

end total_difference_is_18_l610_61021


namespace total_profit_for_the_month_l610_61084

theorem total_profit_for_the_month (mean_profit_month : ℕ) (num_days_month : ℕ)
(mean_profit_first15 : ℕ) (num_days_first15 : ℕ) 
(mean_profit_last15 : ℕ) (num_days_last15 : ℕ) 
(h1 : mean_profit_month = 350) (h2 : num_days_month = 30) 
(h3 : mean_profit_first15 = 285) (h4 : num_days_first15 = 15) 
(h5 : mean_profit_last15 = 415) (h6 : num_days_last15 = 15) : 
(mean_profit_first15 * num_days_first15 + mean_profit_last15 * num_days_last15) = 10500 := by
  sorry

end total_profit_for_the_month_l610_61084


namespace negation_proposition_false_l610_61026

theorem negation_proposition_false : 
  (¬ ∃ x : ℝ, x^2 + 2 ≤ 0) :=
by sorry

end negation_proposition_false_l610_61026


namespace at_most_2n_div_3_good_triangles_l610_61056

-- Definitions based on problem conditions
universe u

structure Polygon (α : Type u) :=
(vertices : List α)
(convex : True)  -- Placeholder for convexity condition

-- Definition for a good triangle
structure Triangle (α : Type u) :=
(vertices : Fin 3 → α)
(unit_length : (Fin 3) → (Fin 3) → Bool)  -- Placeholder for unit length side condition

noncomputable def count_good_triangles {α : Type u} [Inhabited α] (P : Polygon α) : Nat := sorry

theorem at_most_2n_div_3_good_triangles {α : Type u} [Inhabited α] (P : Polygon α) :
  count_good_triangles P ≤ P.vertices.length * 2 / 3 := 
sorry

end at_most_2n_div_3_good_triangles_l610_61056


namespace parallel_lines_l610_61099

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, (ax + 2 * y + a = 0 ∧ 3 * a * x + (a - 1) * y + 7 = 0) →
    - (a / 2) = - (3 * a / (a - 1))) ↔ (a = 0 ∨ a = 7) :=
by
  sorry

end parallel_lines_l610_61099


namespace percentage_students_went_on_trip_l610_61033

theorem percentage_students_went_on_trip
  (total_students : ℕ)
  (students_march : ℕ)
  (students_march_more_than_100 : ℕ)
  (students_june : ℕ)
  (students_june_more_than_100 : ℕ)
  (total_more_than_100_either_trip : ℕ) :
  total_students = 100 → students_march = 20 → students_march_more_than_100 = 7 →
  students_june = 15 → students_june_more_than_100 = 6 →
  70 * total_more_than_100_either_trip = 7 * 100 →
  (students_march + students_june) * 100 / total_students = 35 :=
by
  intros h_total h_march h_march_100 h_june h_june_100 h_total_100
  sorry

end percentage_students_went_on_trip_l610_61033


namespace gcd_2025_2070_l610_61039

theorem gcd_2025_2070 : Nat.gcd 2025 2070 = 45 := by
  sorry

end gcd_2025_2070_l610_61039


namespace computer_multiplications_in_30_minutes_l610_61046

def multiplications_per_second : ℕ := 20000
def seconds_per_minute : ℕ := 60
def minutes : ℕ := 30
def total_seconds : ℕ := minutes * seconds_per_minute
def expected_multiplications : ℕ := 36000000

theorem computer_multiplications_in_30_minutes :
  multiplications_per_second * total_seconds = expected_multiplications :=
by
  sorry

end computer_multiplications_in_30_minutes_l610_61046


namespace problem1_problem2_l610_61037

-- Definitions of the sets A and B based on the given conditions
def A : Set ℝ := { x | x^2 - 6 * x + 8 < 0 }
def B (a : ℝ) : Set ℝ := { x | (x - a) * (x - 3 * a) < 0 }

-- Proof statement for problem (1)
theorem problem1 (a : ℝ) : (∀ x, x ∈ A → x ∈ (B a)) ↔ (4 / 3 ≤ a ∧ a ≤ 2) := by
  sorry

-- Proof statement for problem (2)
theorem problem2 (a : ℝ) : (∀ x, (x ∈ A ∧ x ∈ (B a)) ↔ (3 < x ∧ x < 4)) ↔ (a = 3) := by
  sorry

end problem1_problem2_l610_61037


namespace geometric_sequence_common_ratio_l610_61024

open scoped Nat

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ)
  (h : ∀ n : ℕ, a n * a (n + 1) = (16 : ℝ) ^ n) :
  ∃ r : ℝ, (∀ n : ℕ, a n = a 0 * r ^ n) ∧ (r = 4) :=
sorry

end geometric_sequence_common_ratio_l610_61024


namespace quadratic_inequality_solution_l610_61073

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * x + a ≥ 0) → a ≥ 1 :=
by
  sorry

end quadratic_inequality_solution_l610_61073


namespace min_points_to_win_l610_61055

theorem min_points_to_win : ∀ (points : ℕ), (∀ (race_results : ℕ → ℕ), 
  (points = race_results 1 * 4 + race_results 2 * 2 + race_results 3 * 1) 
  ∧ (∀ i, 1 ≤ race_results i ∧ race_results i ≤ 4) 
  ∧ (∀ i j, i ≠ j → race_results i ≠ race_results j) 
  ∧ (race_results 1 + race_results 2 + race_results 3 = 4)) → (15 ≤ points) :=
by
  sorry

end min_points_to_win_l610_61055


namespace number_of_spiders_l610_61087

theorem number_of_spiders (total_legs : ℕ) (legs_per_spider : ℕ) (h1 : total_legs = 32) (h2 : legs_per_spider = 8) :
  total_legs / legs_per_spider = 4 := by
  sorry

end number_of_spiders_l610_61087


namespace arithmetic_sequence_prop_l610_61080

theorem arithmetic_sequence_prop (a1 d : ℝ) (S : ℕ → ℝ) 
  (h1 : S 6 > S 7) (h2 : S 7 > S 5)
  (hSn : ∀ n, S n = n * a1 + (n * (n - 1) / 2) * d) :
  (d < 0) ∧ (S 11 > 0) ∧ (|a1 + 5 * d| > |a1 + 6 * d|) := 
by
  sorry

end arithmetic_sequence_prop_l610_61080


namespace greatest_three_digit_multiple_of_17_l610_61074

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l610_61074


namespace cannot_lie_on_line_l610_61089

theorem cannot_lie_on_line (m b : ℝ) (h : m * b < 0) : ¬ (0 = m * (-2022) + b) := 
  by
  sorry

end cannot_lie_on_line_l610_61089


namespace quadratic_no_real_roots_iff_m_gt_one_l610_61014

theorem quadratic_no_real_roots_iff_m_gt_one (m : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2 * x + m ≤ 0) ↔ m > 1 :=
sorry

end quadratic_no_real_roots_iff_m_gt_one_l610_61014


namespace perfect_square_n_l610_61054

theorem perfect_square_n (m : ℤ) :
  ∃ (n : ℤ), (n = 7 * m^2 + 6 * m + 1 ∨ n = 7 * m^2 - 6 * m + 1) ∧ ∃ (k : ℤ), 7 * n + 2 = k^2 :=
by
  sorry

end perfect_square_n_l610_61054


namespace find_t_value_l610_61090

theorem find_t_value (t : ℝ) (h1 : (t - 6) * (2 * t - 5) = (2 * t - 8) * (t - 5)) : t = 10 :=
sorry

end find_t_value_l610_61090


namespace average_speed_is_80_l610_61057

def distance : ℕ := 100

def time : ℚ := 5 / 4  -- 1.25 hours expressed as a rational number

noncomputable def average_speed : ℚ := distance / time

theorem average_speed_is_80 : average_speed = 80 := by
  sorry

end average_speed_is_80_l610_61057


namespace equilibrium_temperature_l610_61092

-- Initial conditions for heat capacities and masses
variables (c_B c_W m_B m_W : ℝ) (h : c_W * m_W = 3 * c_B * m_B)

-- Initial temperatures
def T_W_initial := 100
def T_B_initial := 20
def T_f_initial := 80

-- Final equilibrium temperature after second block is added
def final_temp := 68

theorem equilibrium_temperature (t : ℝ)
  (h_first_eq : c_W * m_W * (T_W_initial - T_f_initial) = c_B * m_B * (T_f_initial - T_B_initial))
  (h_second_eq : c_W * m_W * (T_f_initial - t) + c_B * m_B * (T_f_initial - t) = c_B * m_B * (t - T_B_initial)) :
  t = final_temp :=
by 
  sorry

end equilibrium_temperature_l610_61092


namespace change_in_max_value_l610_61009

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem change_in_max_value (a b c : ℝ) (h1 : -b^2 / (4 * (a + 1)) + c = -b^2 / (4 * a) + c + 27 / 2)
  (h2 : -b^2 / (4 * (a - 4)) + c = -b^2 / (4 * a) + c - 9) :
  -b^2 / (4 * (a - 2)) + c = -b^2 / (4 * a) + c - 27 / 4 :=
by
  sorry

end change_in_max_value_l610_61009


namespace reeya_average_l610_61082

theorem reeya_average (s1 s2 s3 s4 s5 : ℕ) (h1 : s1 = 65) (h2 : s2 = 67) (h3 : s3 = 76) (h4 : s4 = 82) (h5 : s5 = 85) :
  (s1 + s2 + s3 + s4 + s5) / 5 = 75 := by
  sorry

end reeya_average_l610_61082


namespace trips_needed_to_fill_pool_l610_61097

def caleb_gallons_per_trip : ℕ := 7
def cynthia_gallons_per_trip : ℕ := 8
def pool_capacity : ℕ := 105

theorem trips_needed_to_fill_pool : (pool_capacity / (caleb_gallons_per_trip + cynthia_gallons_per_trip) = 7) :=
by
  sorry

end trips_needed_to_fill_pool_l610_61097


namespace sine_ratio_triangle_area_l610_61094

variables {A B C : ℝ}
variables {a b c : ℝ}
variables {area : ℝ}

-- Main statement for part 1
theorem sine_ratio 
  (h1 : a * c * Real.cos B - b * c * Real.cos A = 3 * b^2) :
  (Real.sin A / Real.sin B) = Real.sqrt 7 := 
sorry

-- Main statement for part 2
theorem triangle_area 
  (h1 : a * c * Real.cos B - b * c * Real.cos A = 3 * b^2)
  (h2 : c = Real.sqrt 11)
  (h3 : Real.sin C = (2 * Real.sqrt 2)/3)
  (h4 : C < π / 2) :
  area = Real.sqrt 14 :=
sorry

end sine_ratio_triangle_area_l610_61094


namespace point_in_fourth_quadrant_l610_61044

theorem point_in_fourth_quadrant (a b : ℝ) (h1 : a > 0) (h2 : a * b < 0) : a > 0 ∧ b < 0 :=
by 
  have hb : b < 0 := sorry
  exact ⟨h1, hb⟩

end point_in_fourth_quadrant_l610_61044


namespace max_x_satisfies_inequality_l610_61016

theorem max_x_satisfies_inequality (k : ℝ) :
    (∀ x : ℝ, |x^2 - 4 * x + k| + |x - 3| ≤ 5 → x ≤ 3) → k = 8 :=
by
  intros h
  /- The proof goes here. -/
  sorry

end max_x_satisfies_inequality_l610_61016


namespace polynomial_coefficients_correct_l610_61077

-- Define the polynomial equation
def polynomial_equation (x a b c d : ℝ) : Prop :=
  x^4 + 4*x^3 + 3*x^2 + 2*x + 1 = (x+1)^4 + a*(x+1)^3 + b*(x+1)^2 + c*(x+1) + d

-- The problem to prove
theorem polynomial_coefficients_correct :
  ∀ x : ℝ, polynomial_equation x 0 (-3) 4 (-1) :=
by
  intro x
  unfold polynomial_equation
  sorry

end polynomial_coefficients_correct_l610_61077


namespace total_frogs_in_both_ponds_l610_61048

noncomputable def total_frogs_combined : Nat :=
let frogs_in_pond_a : Nat := 32
let frogs_in_pond_b : Nat := frogs_in_pond_a / 2
frogs_in_pond_a + frogs_in_pond_b

theorem total_frogs_in_both_ponds :
  total_frogs_combined = 48 := by
  sorry

end total_frogs_in_both_ponds_l610_61048
