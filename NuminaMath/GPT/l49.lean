import Mathlib

namespace percent_value_in_quarters_l49_49548

theorem percent_value_in_quarters (num_dimes num_quarters : ℕ) 
  (value_dime value_quarter total_value value_in_quarters : ℕ) 
  (h1 : num_dimes = 75)
  (h2 : num_quarters = 30)
  (h3 : value_dime = num_dimes * 10)
  (h4 : value_quarter = num_quarters * 25)
  (h5 : total_value = value_dime + value_quarter)
  (h6 : value_in_quarters = num_quarters * 25) :
  (value_in_quarters / total_value) * 100 = 50 :=
by
  sorry

end percent_value_in_quarters_l49_49548


namespace dan_blue_marbles_l49_49884

variable (m d : ℕ)
variable (h1 : m = 2 * d)
variable (h2 : m = 10)

theorem dan_blue_marbles : d = 5 :=
by
  sorry

end dan_blue_marbles_l49_49884


namespace could_be_simple_random_sampling_l49_49485

-- Conditions
def boys : Nat := 20
def girls : Nat := 30
def total_students : Nat := boys + girls
def sample_size : Nat := 10
def boys_in_sample : Nat := 4
def girls_in_sample : Nat := 6

-- Theorem Statement
theorem could_be_simple_random_sampling :
  boys = 20 ∧ girls = 30 ∧ sample_size = 10 ∧ boys_in_sample = 4 ∧ girls_in_sample = 6 →
  (∃ (sample_method : String), sample_method = "simple random sampling"):=
by 
  sorry

end could_be_simple_random_sampling_l49_49485


namespace min_abs_2x_minus_y_minus_2_l49_49593

open Real

theorem min_abs_2x_minus_y_minus_2
  (x y : ℝ)
  (h : x^2 + y^2 - 4*x + 6*y + 12 = 0) :
  ∃ (c : ℝ), c = 5 - sqrt 5 ∧ ∀ x y : ℝ, (x^2 + y^2 - 4*x + 6*y + 12 = 0) → |2*x - y - 2| ≥ c :=
sorry

end min_abs_2x_minus_y_minus_2_l49_49593


namespace interest_rate_per_annum_l49_49671

variable (P : ℝ := 1200) (T : ℝ := 1) (diff : ℝ := 2.999999999999936) (r : ℝ)
noncomputable def SI (P : ℝ) (r : ℝ) (T : ℝ) : ℝ := P * r * T
noncomputable def CI (P : ℝ) (r : ℝ) (T : ℝ) : ℝ := P * ((1 + r / 2) ^ (2 * T) - 1)

theorem interest_rate_per_annum :
  CI P r T - SI P r T = diff → r = 0.1 :=
by
  -- Proof to be provided
  sorry

end interest_rate_per_annum_l49_49671


namespace range_M_l49_49634

theorem range_M (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b < 1) :
  1 < (1 / (1 + a)) + (1 / (1 + b)) ∧ (1 / (1 + a)) + (1 / (1 + b)) < 2 := by
  sorry

end range_M_l49_49634


namespace cost_of_1000_pieces_of_gum_l49_49367

theorem cost_of_1000_pieces_of_gum
  (cost_per_piece : ℕ)
  (num_pieces : ℕ)
  (discount_threshold : ℕ)
  (discount_rate : ℚ)
  (conversion_rate : ℕ)
  (h_cost : cost_per_piece = 2)
  (h_pieces : num_pieces = 1000)
  (h_threshold : discount_threshold = 500)
  (h_discount : discount_rate = 0.90)
  (h_conversion : conversion_rate = 100)
  (h_more_than_threshold : num_pieces > discount_threshold) :
  (num_pieces * cost_per_piece * discount_rate) / conversion_rate = 18 := 
sorry

end cost_of_1000_pieces_of_gum_l49_49367


namespace old_edition_pages_l49_49148

-- Define the conditions
variables (new_edition : ℕ) (old_edition : ℕ)

-- The conditions given in the problem
axiom new_edition_pages : new_edition = 450
axiom pages_relationship : new_edition = 2 * old_edition - 230

-- Goal: Prove that the old edition Geometry book had 340 pages
theorem old_edition_pages : old_edition = 340 :=
by sorry

end old_edition_pages_l49_49148


namespace distinct_a_count_l49_49029

theorem distinct_a_count :
  ∃ (a_set : Set ℝ), (∀ x ∈ a_set, ∃ r s : ℤ, r + s = -x ∧ r * s = 9 * x) ∧ a_set.toFinset.card = 3 :=
by 
  sorry

end distinct_a_count_l49_49029


namespace necessary_but_not_sufficient_condition_l49_49122

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  {x | 1 / x ≤ 1} ⊆ {x | Real.log x ≥ 0} ∧ 
  ¬ ({x | Real.log x ≥ 0} ⊆ {x | 1 / x ≤ 1}) :=
by
  sorry

end necessary_but_not_sufficient_condition_l49_49122


namespace combined_capacity_is_40_l49_49985

/-- Define the bus capacity as 1/6 the train capacity -/
def bus_capacity (train_capacity : ℕ) := train_capacity / 6

/-- There are two buses in the problem -/
def number_of_buses := 2

/-- The train capacity given in the problem is 120 people -/
def train_capacity := 120

/-- The combined capacity of the two buses is -/
def combined_bus_capacity := number_of_buses * bus_capacity train_capacity

/-- Proof that the combined capacity of the two buses is 40 people -/
theorem combined_capacity_is_40 : combined_bus_capacity = 40 := by
  -- Proof will be filled in here
  sorry

end combined_capacity_is_40_l49_49985


namespace probability_odd_sum_is_one_half_probability_2x_plus_y_less_than_10_is_seven_eighteenths_l49_49688

def num_faces : ℕ := 6
def possible_outcomes : ℕ := num_faces * num_faces

def count_odd_sum_outcomes : ℕ := 18 -- From solution steps
def probability_odd_sum : ℚ := count_odd_sum_outcomes / possible_outcomes

def count_2x_plus_y_less_than_10 : ℕ := 14 -- From solution steps
def probability_2x_plus_y_less_than_10 : ℚ := count_2x_plus_y_less_than_10 / possible_outcomes

theorem probability_odd_sum_is_one_half :
  probability_odd_sum = 1 / 2 :=
sorry

theorem probability_2x_plus_y_less_than_10_is_seven_eighteenths :
  probability_2x_plus_y_less_than_10 = 7 / 18 :=
sorry

end probability_odd_sum_is_one_half_probability_2x_plus_y_less_than_10_is_seven_eighteenths_l49_49688


namespace ratio_of_q_to_p_l49_49820

theorem ratio_of_q_to_p (p q : ℝ) (h₀ : 0 < p) (h₁ : 0 < q) 
  (h₂ : Real.log p / Real.log 9 = Real.log q / Real.log 12) 
  (h₃ : Real.log q / Real.log 12 = Real.log (p + q) / Real.log 16) : 
  q / p = (1 + Real.sqrt 5) / 2 := 
by 
  sorry

end ratio_of_q_to_p_l49_49820


namespace triangle_side_difference_l49_49962

theorem triangle_side_difference (x : ℕ) : 3 < x ∧ x < 17 → (∃ a b : ℕ, 3 < a ∧ a < 17 ∧ 3 < b ∧ b < 17 ∧ a - b = 12) :=
by
  sorry

end triangle_side_difference_l49_49962


namespace probability_xi_leq_7_l49_49950

noncomputable def probability_ball_draw_score : ℚ :=
  let red_balls := 4
  let black_balls := 3
  let total_balls := red_balls + black_balls
  let score := λ (red black : ℕ), red + 3 * black
  let comb := λ n k, (nat.choose n k : ℚ)
  (comb red_balls 4 / comb total_balls 4) +
  (comb red_balls 3 * comb black_balls 1 / comb total_balls 4)

theorem probability_xi_leq_7 : probability_ball_draw_score = (13 / 35) := by
  sorry

end probability_xi_leq_7_l49_49950


namespace max_ratio_three_digit_l49_49019

theorem max_ratio_three_digit (x a b c : ℕ) (h1 : 100 * a + 10 * b + c = x) (h2 : 1 ≤ a ∧ a ≤ 9)
  (h3 : 0 ≤ b ∧ b ≤ 9) (h4 : 0 ≤ c ∧ c ≤ 9) : 
  (x : ℚ) / (a + b + c) ≤ 100 := sorry

end max_ratio_three_digit_l49_49019


namespace part1_part2_part3_l49_49223

-- Part 1
theorem part1 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : (x + y) * (y + z) * (z + x) ≥ 8 * x * y * z :=
sorry

-- Part 2
theorem part2 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : x^2 + y^2 + z^2 ≥ x * y + y * z + z * x :=
sorry

-- Part 3
theorem part3 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : x ^ x * y ^ y * z ^ z ≥ (x * y * z) ^ ((x + y + z) / 3) :=
sorry

#print axioms part1
#print axioms part2
#print axioms part3

end part1_part2_part3_l49_49223


namespace probability_same_number_l49_49285

def is_multiple (n factor : ℕ) : Prop :=
  ∃ k : ℕ, n = k * factor

def multiples_below (factor upper_limit : ℕ) : ℕ :=
  (upper_limit - 1) / factor

theorem probability_same_number :
  let upper_limit := 250
  let billy_factor := 20
  let bobbi_factor := 30
  let common_factor := 60
  let billy_multiples := multiples_below billy_factor upper_limit
  let bobbi_multiples := multiples_below bobbi_factor upper_limit
  let common_multiples := multiples_below common_factor upper_limit
  (common_multiples : ℚ) / (billy_multiples * bobbi_multiples) = 1 / 24 :=
by
  sorry

end probability_same_number_l49_49285


namespace ball_distribution_l49_49123

theorem ball_distribution (basketballs volleyballs classes balls : ℕ) 
  (h1 : basketballs = 2) 
  (h2 : volleyballs = 3) 
  (h3 : classes = 4) 
  (h4 : balls = 4) :
  (classes.choose 3) + (classes.choose 2) = 10 :=
by
  sorry

end ball_distribution_l49_49123


namespace circle_area_l49_49286

-- Condition: Given the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 10 * x + 4 * y + 20 = 0

-- Theorem: The area enclosed by the given circle equation is 9π
theorem circle_area : ∀ x y : ℝ, circle_eq x y → ∃ A : ℝ, A = 9 * Real.pi :=
by
  intros
  sorry

end circle_area_l49_49286


namespace nums_between_2000_and_3000_div_by_360_l49_49935

theorem nums_between_2000_and_3000_div_by_360 : 
  (∃ n1 n2 n3 : ℕ, 2000 ≤ n1 ∧ n1 ≤ 3000 ∧ 360 ∣ n1 ∧
                   2000 ≤ n2 ∧ n2 ≤ 3000 ∧ 360 ∣ n2 ∧
                   2000 ≤ n3 ∧ n3 ≤ 3000 ∧ 360 ∣ n3 ∧
                   n1 ≠ n2 ∧ n1 ≠ n3 ∧ n2 ≠ n3 ∧
                   ∀ m : ℕ, (2000 ≤ m ∧ m ≤ 3000 ∧ 360 ∣ m → m = n1 ∨ m = n2 ∨ m = n3)) := 
begin
  sorry
end

end nums_between_2000_and_3000_div_by_360_l49_49935


namespace part1_part2_part3_l49_49616

-- Part 1: Prove that B = 90° given a=20, b=29, c=21

theorem part1 (a b c : ℝ) (h1 : a = 20) (h2 : b = 29) (h3 : c = 21) : 
  ∃ B : ℝ, B = 90 := 
sorry

-- Part 2: Prove that b = 7 given a=3√3, c=2, B=150°

theorem part2 (a c B b : ℝ) (h1 : a = 3 * Real.sqrt 3) (h2 : c = 2) (h3 : B = 150) : 
  ∃ b : ℝ, b = 7 :=
sorry

-- Part 3: Prove that A = 45° given a=2, b=√2, c=√3 + 1

theorem part3 (a b c A : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 2) (h3 : c = Real.sqrt 3 + 1) : 
  ∃ A : ℝ, A = 45 :=
sorry

end part1_part2_part3_l49_49616


namespace parking_lot_wheels_l49_49290

noncomputable def total_car_wheels (guest_cars : Nat) (guest_car_wheels : Nat) (parent_cars : Nat) (parent_car_wheels : Nat) : Nat :=
  guest_cars * guest_car_wheels + parent_cars * parent_car_wheels

theorem parking_lot_wheels :
  total_car_wheels 10 4 2 4 = 48 :=
by
  sorry

end parking_lot_wheels_l49_49290


namespace compute_x_squared_y_plus_x_y_squared_l49_49047

open Real

theorem compute_x_squared_y_plus_x_y_squared (x y : ℝ) 
  (h1 : (1/x) + (1/y) = 5) 
  (h2 : x * y + 2 * x + 2 * y = 7) : 
  x^2 * y + x * y^2 = 245 / 121 := 
by 
  sorry

end compute_x_squared_y_plus_x_y_squared_l49_49047


namespace sweatshirt_sales_l49_49083

variables (S H : ℝ)

theorem sweatshirt_sales (h1 : 13 * S + 9 * H = 370) (h2 : 9 * S + 2 * H = 180) :
  12 * S + 6 * H = 300 :=
sorry

end sweatshirt_sales_l49_49083


namespace representation_of_1_l49_49170

theorem representation_of_1 (x y z : ℕ) (h : 1 = 1/x + 1/y + 1/z) : 
  (x = 2 ∧ y = 3 ∧ z = 6) ∨ (x = 2 ∧ y = 4 ∧ z = 4) ∨ (x = 3 ∧ y = 3 ∧ z = 3) :=
by
  sorry

end representation_of_1_l49_49170


namespace gypsy_pheasants_l49_49056

theorem gypsy_pheasants (x : ℚ) (h1 : (3 : ℚ) = x / 8 - 7 / 8) : x = 31 := 
sorry

end gypsy_pheasants_l49_49056


namespace num_real_values_for_integer_roots_l49_49032

theorem num_real_values_for_integer_roots : 
  (∃ (a : ℝ), ∀ (r s : ℤ), r + s = -a ∧ r * s = 9 * a) → ∃ (n : ℕ), n = 10 :=
by
  sorry

end num_real_values_for_integer_roots_l49_49032


namespace people_with_fewer_than_7_cards_l49_49785

theorem people_with_fewer_than_7_cards (num_cards : ℕ) (num_people : ℕ) (h₁ : num_cards = 60) (h₂ : num_people = 9) : 
  ∃ k, k = num_people - num_cards % num_people ∧ k < 7 :=
by
  have rem := num_cards % num_people
  have few_count := num_people - rem
  use few_count
  split
  sorry

end people_with_fewer_than_7_cards_l49_49785


namespace first_term_geometric_sequence_l49_49530

theorem first_term_geometric_sequence (a r : ℚ) 
    (h1 : a * r^2 = 8) 
    (h2 : a * r^4 = 27 / 4) : 
    a = 256 / 27 :=
by sorry

end first_term_geometric_sequence_l49_49530


namespace reciprocal_of_fraction_sum_l49_49294

theorem reciprocal_of_fraction_sum : 
  (1 / (1 / 3 + 1 / 4 - 1 / 12)) = 2 := sorry

end reciprocal_of_fraction_sum_l49_49294


namespace no_good_polygon_in_division_of_equilateral_l49_49972

def is_equilateral_polygon (P : List Point) : Prop :=
  -- Definition of equilateral polygon
  sorry

def is_good_polygon (P : List Point) : Prop :=
  -- Definition of good polygon (having a pair of parallel sides)
  sorry

def is_divided_by_non_intersecting_diagonals (P : List Point) (polygons : List (List Point)) : Prop :=
  -- Definition for dividing by non-intersecting diagonals into several polygons
  sorry

def have_same_odd_sides (polygons : List (List Point)) : Prop :=
  -- Definition for all polygons having the same odd number of sides
  sorry

theorem no_good_polygon_in_division_of_equilateral (P : List Point) (polygons : List (List Point)) :
  is_equilateral_polygon P →
  is_divided_by_non_intersecting_diagonals P polygons →
  have_same_odd_sides polygons →
  ¬ ∃ gp ∈ polygons, is_good_polygon gp :=
by
  intro h_eq h_div h_odd
  intro h_good
  -- Proof goes here
  sorry

end no_good_polygon_in_division_of_equilateral_l49_49972


namespace isosceles_triangle_base_length_l49_49664

theorem isosceles_triangle_base_length
  (b : ℕ)
  (congruent_side : ℕ)
  (perimeter : ℕ)
  (h1 : congruent_side = 8)
  (h2 : perimeter = 25)
  (h3 : 2 * congruent_side + b = perimeter) :
  b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l49_49664


namespace Donovan_Mitchell_goal_l49_49886

theorem Donovan_Mitchell_goal 
  (current_avg : ℕ) 
  (current_games : ℕ) 
  (target_avg : ℕ) 
  (total_games : ℕ) 
  (remaining_games : ℕ) 
  (points_scored_so_far : ℕ)
  (points_needed_total : ℕ)
  (points_needed_remaining : ℕ) :
  (current_avg = 26) ∧
  (current_games = 15) ∧
  (target_avg = 30) ∧
  (total_games = 20) ∧
  (remaining_games = 5) ∧
  (points_scored_so_far = current_avg * current_games) ∧
  (points_needed_total = target_avg * total_games) ∧
  (points_needed_remaining = points_needed_total - points_scored_so_far) →
  (points_needed_remaining / remaining_games = 42) :=
by
  sorry

end Donovan_Mitchell_goal_l49_49886


namespace cards_dealt_l49_49772

theorem cards_dealt (total_cards : ℕ) (num_people : ℕ) (fewer_cards : ℕ) :
  total_cards = 60 → num_people = 9 → fewer_cards = 3 →
  ∃ k : ℕ, total_cards = num_people * k + 6 ∧ k = 6 ∧ 
  (num_people - 6 = fewer_cards) :=
by
  intros h1 h2 h3
  use 6
  split;
  sorry

end cards_dealt_l49_49772


namespace find_m_l49_49932

variables {m : ℝ}
def vec_a : ℝ × ℝ := (-2, 3)
def vec_b (m : ℝ) : ℝ × ℝ := (3, m)
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem find_m (m : ℝ) (h : perpendicular vec_a (vec_b m)) : m = 2 :=
by
  sorry

end find_m_l49_49932


namespace beads_per_bracelet_is_10_l49_49806

-- Definitions of given conditions
def num_necklaces_Monday : ℕ := 10
def num_necklaces_Tuesday : ℕ := 2
def num_necklaces : ℕ := num_necklaces_Monday + num_necklaces_Tuesday

def beads_per_necklace : ℕ := 20
def beads_necklaces : ℕ := num_necklaces * beads_per_necklace

def num_earrings : ℕ := 7
def beads_per_earring : ℕ := 5
def beads_earrings : ℕ := num_earrings * beads_per_earring

def total_beads_used : ℕ := 325
def beads_used_for_necklaces_and_earrings : ℕ := beads_necklaces + beads_earrings
def beads_remaining_for_bracelets : ℕ := total_beads_used - beads_used_for_necklaces_and_earrings

def num_bracelets : ℕ := 5
def beads_per_bracelet : ℕ := beads_remaining_for_bracelets / num_bracelets

-- Theorem statement to prove
theorem beads_per_bracelet_is_10 : beads_per_bracelet = 10 := by
  sorry

end beads_per_bracelet_is_10_l49_49806


namespace combined_bus_capacity_l49_49981

-- Define conditions
def train_capacity : ℕ := 120
def bus_capacity : ℕ := train_capacity / 6
def number_of_buses : ℕ := 2

-- Define theorem for the combined capacity of two buses
theorem combined_bus_capacity : number_of_buses * bus_capacity = 40 := by
  -- We declare that the proof is skipped here
  sorry

end combined_bus_capacity_l49_49981


namespace binary_110101_is_53_l49_49013

def binary_to_decimal (n : Nat) : Nat :=
  let digits := [1, 1, 0, 1, 0, 1]  -- Define binary digits from the problem statement
  digits.reverse.foldr (λ d (acc, pow) => (acc + d * (2^pow), pow + 1)) (0, 0) |>.fst

theorem binary_110101_is_53 : binary_to_decimal 110101 = 53 := by
  sorry

end binary_110101_is_53_l49_49013


namespace price_reduction_proof_l49_49727

theorem price_reduction_proof (x : ℝ) : 256 * (1 - x) ^ 2 = 196 :=
sorry

end price_reduction_proof_l49_49727


namespace overall_profit_or_loss_l49_49629

def price_USD_to_INR(price_usd : ℝ) : ℝ := price_usd * 75
def price_EUR_to_INR(price_eur : ℝ) : ℝ := price_eur * 80
def price_GBP_to_INR(price_gbp : ℝ) : ℝ := price_gbp * 100
def price_JPY_to_INR(price_jpy : ℝ) : ℝ := price_jpy * 0.7

def CP_grinder : ℝ := price_USD_to_INR (150 + 0.1 * 150)
def SP_grinder : ℝ := price_USD_to_INR (165 - 0.04 * 165)

def CP_mobile_phone : ℝ := price_EUR_to_INR ((100 - 0.05 * 100) + 0.15 * (100 - 0.05 * 100))
def SP_mobile_phone : ℝ := price_EUR_to_INR ((109.25 : ℝ) + 0.1 * 109.25)

def CP_laptop : ℝ := price_GBP_to_INR (200 + 0.08 * 200)
def SP_laptop : ℝ := price_GBP_to_INR (216 - 0.08 * 216)

def CP_camera : ℝ := price_JPY_to_INR ((12000 - 0.12 * 12000) + 0.05 * (12000 - 0.12 * 12000))
def SP_camera : ℝ := price_JPY_to_INR (11088 + 0.15 * 11088)

def total_CP : ℝ := CP_grinder + CP_mobile_phone + CP_laptop + CP_camera
def total_SP : ℝ := SP_grinder + SP_mobile_phone + SP_laptop + SP_camera

theorem overall_profit_or_loss :
  (total_SP - total_CP) = -184.76 := 
sorry

end overall_profit_or_loss_l49_49629


namespace cards_dealt_l49_49774

theorem cards_dealt (total_cards : ℕ) (num_people : ℕ) (fewer_cards : ℕ) :
  total_cards = 60 → num_people = 9 → fewer_cards = 3 →
  ∃ k : ℕ, total_cards = num_people * k + 6 ∧ k = 6 ∧ 
  (num_people - 6 = fewer_cards) :=
by
  intros h1 h2 h3
  use 6
  split;
  sorry

end cards_dealt_l49_49774


namespace simplify_and_evaluate_expression_l49_49516

theorem simplify_and_evaluate_expression 
  (a b : ℚ) 
  (ha : a = 2) 
  (hb : b = 1 / 3) : 
  (a / (a - b)) * ((1 / b) - (1 / a)) + ((a - 1) / b) = 6 := 
by
  -- Place the steps verifying this here. For now:
  sorry

end simplify_and_evaluate_expression_l49_49516


namespace red_ball_second_given_red_ball_first_l49_49140

noncomputable def probability_of_red_second_given_first : ℚ :=
  let totalBalls := 6
  let redBallsOnFirst := 4
  let whiteBalls := 2
  let redBallsOnSecond := 3
  let remainingBalls := 5

  let P_A := redBallsOnFirst / totalBalls
  let P_AB := (redBallsOnFirst / totalBalls) * (redBallsOnSecond / remainingBalls)
  P_AB / P_A

theorem red_ball_second_given_red_ball_first :
  probability_of_red_second_given_first = 3 / 5 :=
sorry

end red_ball_second_given_red_ball_first_l49_49140


namespace find_tan_G_l49_49021

def right_triangle (FG GH FH : ℕ) : Prop :=
  FG^2 = GH^2 + FH^2

def tan_ratio (GH FH : ℕ) : ℚ :=
  FH / GH

theorem find_tan_G
  (FG GH : ℕ)
  (H1 : FG = 13)
  (H2 : GH = 12)
  (FH : ℕ)
  (H3 : right_triangle FG GH FH) :
  tan_ratio GH FH = 5 / 12 :=
by
  sorry

end find_tan_G_l49_49021


namespace old_edition_pages_l49_49146

theorem old_edition_pages (x : ℕ) (h : 2 * x - 230 = 450) : x = 340 :=
by {
  have eq1 : 2 * x = 450 + 230, from eq_add_of_sub_eq h,
  have eq2 : 2 * x = 680, from eq1,
  have eq3 : x = 680 / 2, from eq_of_mul_eq_mul_right (by norm_num) eq2,
  norm_num at eq3,
  exact eq3,
}

end old_edition_pages_l49_49146


namespace xiaopangs_score_is_16_l49_49544

-- Define the father's score
def fathers_score : ℕ := 48

-- Define Xiaopang's score in terms of father's score
def xiaopangs_score (fathers_score : ℕ) : ℕ := fathers_score / 2 - 8

-- The theorem to prove that Xiaopang's score is 16
theorem xiaopangs_score_is_16 : xiaopangs_score fathers_score = 16 := 
by
  sorry

end xiaopangs_score_is_16_l49_49544


namespace num_people_fewer_than_7_cards_l49_49776

theorem num_people_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      remainder := total_cards % total_people
  in total_people - remainder = 3 :=
by
  sorry

end num_people_fewer_than_7_cards_l49_49776


namespace bankers_gain_is_60_l49_49659

def banker's_gain (BD F PV R T : ℝ) : ℝ :=
  let TD := F - PV
  BD - TD

theorem bankers_gain_is_60 (BD F PV R T BG : ℝ) (h₁ : BD = 260) (h₂ : R = 0.10) (h₃ : T = 3)
  (h₄ : F = 260 / 0.3) (h₅ : PV = F / (1 + (R * T))) :
  banker's_gain BD F PV R T = 60 :=
by
  rw [banker's_gain, h₄, h₅]
  -- Further simplifications and exact equality steps would be added here with actual proof steps
  sorry

end bankers_gain_is_60_l49_49659


namespace unique_pair_a_b_l49_49436

open Complex

theorem unique_pair_a_b :
  ∃! (a b : ℂ), a^4 * b^3 = 1 ∧ a^6 * b^7 = 1 := by
  sorry

end unique_pair_a_b_l49_49436


namespace sequence_general_formula_l49_49600

theorem sequence_general_formula :
  (∃ a : ℕ → ℕ, a 1 = 4 ∧ a 2 = 6 ∧ a 3 = 8 ∧ a 4 = 10 ∧ (∀ n : ℕ, a n = 2 * (n + 1))) :=
by
  sorry

end sequence_general_formula_l49_49600


namespace units_digit_7_pow_1023_l49_49853

-- Define a function for the units digit of a number
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_7_pow_1023 :
  units_digit (7 ^ 1023) = 3 :=
by
  sorry

end units_digit_7_pow_1023_l49_49853


namespace intersection_of_sets_l49_49349

noncomputable def U : Set ℝ := Set.univ

noncomputable def M : Set ℝ := {x | x < -1 ∨ x > 1}

noncomputable def N : Set ℝ := {x | 0 < x ∧ x < 2}

noncomputable def complement_U_M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

noncomputable def intersection_N_complement_U_M : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem intersection_of_sets :
  N ∩ complement_U_M = intersection_N_complement_U_M := 
sorry

end intersection_of_sets_l49_49349


namespace mary_needs_more_apples_l49_49092

theorem mary_needs_more_apples :
  let pies := 15
  let apples_per_pie := 10
  let harvested_apples := 40
  let total_apples_needed := pies * apples_per_pie
  let more_apples_needed := total_apples_needed - harvested_apples
  more_apples_needed = 110 :=
by
  sorry

end mary_needs_more_apples_l49_49092


namespace tangent_expression_l49_49189

open Real

theorem tangent_expression
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (geom_seq : ∀ n m, a (n + m) = a n * a m) 
  (arith_seq : ∀ n, b (n + 1) = b n + (b 2 - b 1))
  (cond1 : a 1 * a 6 * a 11 = -3 * sqrt 3)
  (cond2 : b 1 + b 6 + b 11 = 7 * pi) :
  tan ( (b 3 + b 9) / (1 - a 4 * a 8) ) = -sqrt 3 :=
sorry

end tangent_expression_l49_49189


namespace roots_of_transformed_quadratic_l49_49318

theorem roots_of_transformed_quadratic (a b c d x : ℝ) :
  (∀ x, (x - a) * (x - b) - x = 0 → x = c ∨ x = d) →
  (x - c) * (x - d) + x = 0 → x = a ∨ x = b :=
by
  sorry

end roots_of_transformed_quadratic_l49_49318


namespace people_with_fewer_than_7_cards_l49_49791

theorem people_with_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (cards_per_person : ℕ) (remainder : ℕ) 
  (h : total_cards = total_people * cards_per_person + remainder)
  (h_cards : total_cards = 60)
  (h_people : total_people = 9)
  (h_cards_per_person : cards_per_person = 6)
  (h_remainder : remainder = 6) :
  (total_people - remainder) = 3 :=
by
  sorry

end people_with_fewer_than_7_cards_l49_49791


namespace solve_inequality_l49_49996

theorem solve_inequality (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) : 
  (x^2 - 9) / (x^2 - 1) > 0 ↔ (x > 3 ∨ x < -3 ∨ (-1 < x ∧ x < 1)) :=
sorry

end solve_inequality_l49_49996


namespace number_of_common_tangents_l49_49676

noncomputable def circle1_center : ℝ × ℝ := (-3, 0)
noncomputable def circle1_radius : ℝ := 4

noncomputable def circle2_center : ℝ × ℝ := (0, 3)
noncomputable def circle2_radius : ℝ := 6

theorem number_of_common_tangents 
  (center1 center2 : ℝ × ℝ)
  (radius1 radius2 : ℝ)
  (h_center1: center1 = (-3, 0))
  (h_radius1: radius1 = 4)
  (h_center2: center2 = (0, 3))
  (h_radius2: radius2 = 6) :
  -- The sought number of common tangents between the two circles
  2 = 2 :=
by
  sorry

end number_of_common_tangents_l49_49676


namespace exists_three_mutually_related_or_unrelated_l49_49102

variable {V : Type} [Fintype V] [DecidableRel (λ v w : V, v ≠ w)]

theorem exists_three_mutually_related_or_unrelated (hV : Fintype.card V = 6) 
  (knows : V → V → Prop) : 
  (∃ A B C : V, (knows A B ∧ knows A C ∧ knows B C) ∨ 
                  (¬ knows A B ∧ ¬ knows A C ∧ ¬ knows B C)) :=
sorry

end exists_three_mutually_related_or_unrelated_l49_49102


namespace calculate_value_l49_49126

-- Definition of the given values
def val1 : ℕ := 444
def val2 : ℕ := 44
def val3 : ℕ := 4

-- Theorem statement proving the value of the expression
theorem calculate_value : (val1 - val2 - val3) = 396 := 
by 
  sorry

end calculate_value_l49_49126


namespace find_possible_m_values_l49_49595

theorem find_possible_m_values (m : ℕ) (a : ℕ) (h₀ : m > 1) (h₁ : m * a + (m * (m - 1) / 2) = 33) :
  m = 2 ∨ m = 3 ∨ m = 6 :=
by
  sorry

end find_possible_m_values_l49_49595


namespace machine_makes_12_shirts_l49_49730

def shirts_per_minute : ℕ := 2
def minutes_worked : ℕ := 6

def total_shirts_made : ℕ := shirts_per_minute * minutes_worked

theorem machine_makes_12_shirts :
  total_shirts_made = 12 :=
by
  -- proof placeholder
  sorry

end machine_makes_12_shirts_l49_49730


namespace field_dimension_solution_l49_49268

theorem field_dimension_solution (m : ℝ) (h₁ : (3 * m + 10) * (m - 5) = 72) : m = 7 :=
sorry

end field_dimension_solution_l49_49268


namespace adam_lessons_on_monday_l49_49408

theorem adam_lessons_on_monday :
  (∃ (time_monday time_tuesday time_wednesday : ℝ) (n_monday_lessons : ℕ),
    time_tuesday = 3 ∧
    time_wednesday = 2 * time_tuesday ∧
    time_monday + time_tuesday + time_wednesday = 12 ∧
    n_monday_lessons = time_monday / 0.5 ∧
    n_monday_lessons = 6) :=
by
  sorry

end adam_lessons_on_monday_l49_49408


namespace number_of_added_groups_l49_49357

-- Define the total number of students in the class
def total_students : ℕ := 47

-- Define the number of students per table and the number of tables
def students_per_table : ℕ := 3
def number_of_tables : ℕ := 6

-- Define the number of girls in the bathroom and the multiplier for students in the canteen
def girls_in_bathroom : ℕ := 3
def canteen_multiplier : ℕ := 3

-- Define the number of foreign exchange students from each country
def foreign_exchange_germany : ℕ := 3
def foreign_exchange_france : ℕ := 3
def foreign_exchange_norway : ℕ := 3

-- Define the number of students per recently added group
def students_per_group : ℕ := 4

-- Calculate the number of students currently in the classroom
def students_in_classroom := number_of_tables * students_per_table

-- Calculate the number of students temporarily absent
def students_in_canteen := girls_in_bathroom * canteen_multiplier
def temporarily_absent := girls_in_bathroom + students_in_canteen

-- Calculate the number of foreign exchange students missing
def foreign_exchange_missing := foreign_exchange_germany + foreign_exchange_france + foreign_exchange_norway

-- Calculate the total number of students accounted for
def student_accounted_for := students_in_classroom + temporarily_absent + foreign_exchange_missing

-- The proof statement (main goal)
theorem number_of_added_groups : (total_students - student_accounted_for) / students_per_group = 2 :=
by
  sorry

end number_of_added_groups_l49_49357


namespace smartphone_price_l49_49283

theorem smartphone_price (S : ℝ) (pc_price : ℝ) (tablet_price : ℝ) 
  (total_cost : ℝ) (h1 : pc_price = S + 500) 
  (h2 : tablet_price = 2 * S + 500) 
  (h3 : S + pc_price + tablet_price = 2200) : 
  S = 300 :=
by
  sorry

end smartphone_price_l49_49283


namespace volume_of_revolved_region_l49_49496

theorem volume_of_revolved_region :
  let R := {p : ℝ × ℝ | |8 - p.1| + p.2 ≤ 10 ∧ 3 * p.2 - p.1 ≥ 15}
  let volume := (1 / 3) * Real.pi * (7 / Real.sqrt 10)^2 * (7 * Real.sqrt 10 / 4)
  let m := 343
  let n := 12
  let p := 10
  m + n + p = 365 := by
  sorry

end volume_of_revolved_region_l49_49496


namespace triangle_area_l49_49717

theorem triangle_area (a b c : ℕ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 54 := by
    sorry

end triangle_area_l49_49717


namespace mul_mod_l49_49851

theorem mul_mod (n1 n2 n3 : ℤ) (h1 : n1 = 2011) (h2 : n2 = 1537) (h3 : n3 = 450) : 
  (2011 * 1537) % 450 = 307 := by
  sorry

end mul_mod_l49_49851


namespace units_digit_of_7_pow_6_pow_5_l49_49903

theorem units_digit_of_7_pow_6_pow_5 : (7^(6^5)) % 10 = 1 := by
  -- Proof goes here
  sorry

end units_digit_of_7_pow_6_pow_5_l49_49903


namespace solve_equation_1_solve_equation_2_l49_49105

theorem solve_equation_1 (y: ℝ) : y^2 - 6 * y + 1 = 0 ↔ (y = 3 + 2 * Real.sqrt 2 ∨ y = 3 - 2 * Real.sqrt 2) :=
sorry

theorem solve_equation_2 (x: ℝ) : 2 * (x - 4)^2 = x^2 - 16 ↔ (x = 4 ∨ x = 12) :=
sorry

end solve_equation_1_solve_equation_2_l49_49105


namespace problem_1_problem_2_l49_49042

def f (x : ℝ) : ℝ := x^2 + 4 * x
def g (a : ℝ) : ℝ := |a - 2| + |a + 1|

theorem problem_1 (x : ℝ) :
    (f x ≥ g 3) ↔ (x ≥ 1 ∨ x ≤ -5) :=
  sorry

theorem problem_2 (a : ℝ) :
    (∃ x : ℝ, f x + g a = 0) → (-3 / 2 ≤ a ∧ a ≤ 5 / 2) :=
  sorry

end problem_1_problem_2_l49_49042


namespace correct_option_B_l49_49116

-- Given conditions
variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f x = f (-x))
variable (h_mono_inc : ∀ (a b : ℝ), 0 ≤ a ∧ a ≤ b → f a ≤ f b)

-- Theorem statement
theorem correct_option_B : f (-2) > f (-1) ∧ f (-1) > f (0) :=
by
  sorry

end correct_option_B_l49_49116


namespace f_periodic_l49_49503

noncomputable def f : ℝ → ℝ := sorry

variable (a : ℝ) (h_a : 0 < a)
variable (h_cond : ∀ x : ℝ, f (x + a) = 1 / 2 + sqrt (f x - (f x)^2))

theorem f_periodic : ∀ x : ℝ, f (x + 2 * a) = f x := sorry

end f_periodic_l49_49503


namespace div_m_by_18_equals_500_l49_49117

-- Define the conditions
noncomputable def m : ℕ := 9000 -- 'm' is given as 9000 since it fulfills all conditions described
def is_multiple_of_18 (n : ℕ) : Prop := n % 18 = 0
def all_digits_9_or_0 (n : ℕ) : Prop := ∀ (d : ℕ), (∃ (k : ℕ), n = 10^k * d) → (d = 0 ∨ d = 9)

-- Define the proof problem statement
theorem div_m_by_18_equals_500 
  (h1 : is_multiple_of_18 m) 
  (h2 : all_digits_9_or_0 m) 
  (h3 : ∀ n, is_multiple_of_18 n ∧ all_digits_9_or_0 n → n ≤ m) : 
  m / 18 = 500 :=
sorry

end div_m_by_18_equals_500_l49_49117


namespace function_values_l49_49930

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x + b

theorem function_values (a b : ℝ) (h1 : f 1 a b = 2) (h2 : a = 2) : f 2 a b = 4 := by
  sorry

end function_values_l49_49930


namespace probability_of_vowel_initials_l49_49979

def students_26_initials_distinct : Prop :=
  ∀ (students : Fin 26 → Char × Char), function.Injective students

def is_vowel (c : Char) : Prop :=
  c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U' ∨ c = 'Y'

theorem probability_of_vowel_initials : 
  (∀ (students : Fin 26 → Char × Char), students_26_initials_distinct students) →
  (let vowels := ['A', 'E', 'I', 'O', 'U', 'Y'] in
   let num_vowels := vowels.length in
   let total_initials := 26 in
   (num_vowels : ℚ) / total_initials = 3 / 13) :=
by
  sorry

end probability_of_vowel_initials_l49_49979


namespace distinct_a_count_l49_49030

theorem distinct_a_count :
  ∃ (a_set : Set ℝ), (∀ x ∈ a_set, ∃ r s : ℤ, r + s = -x ∧ r * s = 9 * x) ∧ a_set.toFinset.card = 3 :=
by 
  sorry

end distinct_a_count_l49_49030


namespace range_of_a_l49_49025

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l49_49025


namespace count_distinct_reals_a_with_integer_roots_l49_49027

-- Define the quadratic equation with its roots and conditions
theorem count_distinct_reals_a_with_integer_roots :
  ∃ (a_vals : Finset ℝ), a_vals.card = 6 ∧
    (∀ a ∈ a_vals, ∃ r s : ℤ, 
      (r + s : ℝ) = -a ∧ (r * s : ℝ) = 9 * a) :=
by
  sorry

end count_distinct_reals_a_with_integer_roots_l49_49027


namespace complex_square_l49_49010

theorem complex_square (i : ℂ) (hi : i^2 = -1) : (1 + i)^2 = 2 * i :=
by
  sorry

end complex_square_l49_49010


namespace part_a_part_b_l49_49501

-- Part (a)
theorem part_a (a b : ℕ) (h : Nat.lcm a (a + 5) = Nat.lcm b (b + 5)) : a = b :=
sorry

-- Part (b)
theorem part_b (a b c : ℕ) (gcd_abc : Nat.gcd a (Nat.gcd b c) = 1) :
  Nat.lcm a b = Nat.lcm (a + c) (b + c) → False :=
sorry

end part_a_part_b_l49_49501


namespace adjacent_irreducible_rationals_condition_l49_49801

theorem adjacent_irreducible_rationals_condition 
  (a b c d : ℕ) 
  (hab_cop : Nat.gcd a b = 1) (hcd_cop : Nat.gcd c d = 1) 
  (h_ab_prod : a * b < 1988) (h_cd_prod : c * d < 1988) 
  (adj : ∀ p q r s, (Nat.gcd p q = 1) → (Nat.gcd r s = 1) → 
                  (p * q < 1988) → (r * s < 1988) →
                  (p / q < r / s) → (p * s - q * r = 1)) : 
  b * c - a * d = 1 :=
sorry

end adjacent_irreducible_rationals_condition_l49_49801


namespace correct_barometric_pressure_l49_49204

noncomputable def true_barometric_pressure (p1 p2 v1 v2 T1 T2 observed_pressure_final observed_pressure_initial : ℝ) : ℝ :=
  let combined_gas_law : ℝ := (p1 * v1 * T2) / (v2 * T1)
  observed_pressure_final + combined_gas_law

theorem correct_barometric_pressure :
  true_barometric_pressure 58 56 143 155 288 303 692 704 = 748 :=
by
  sorry

end correct_barometric_pressure_l49_49204


namespace smallest_perimeter_l49_49933

theorem smallest_perimeter (m n : ℕ) 
  (h1 : (m - 4) * (n - 4) = 8) 
  (h2 : ∀ k l : ℕ, (k - 4) * (l - 4) = 8 → 2 * k + 2 * l ≥ 2 * m + 2 * n) : 
  (m = 6 ∧ n = 8) ∨ (m = 8 ∧ n = 6) :=
sorry

end smallest_perimeter_l49_49933


namespace area_union_of_reflected_triangles_l49_49725

def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (5, 7)
def C : ℝ × ℝ := (6, 2)
def A' : ℝ × ℝ := (3, 2)
def B' : ℝ × ℝ := (7, 5)
def C' : ℝ × ℝ := (2, 6)

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem area_union_of_reflected_triangles :
  let area_ABC := triangle_area A B C
  let area_A'B'C' := triangle_area A' B' C'
  area_ABC + area_A'B'C' = 19 := by
  sorry

end area_union_of_reflected_triangles_l49_49725


namespace sum_of_x_y_l49_49327

theorem sum_of_x_y (x y : ℝ) (h : (x + y + 2) * (x + y - 1) = 0) : x + y = -2 ∨ x + y = 1 :=
by sorry

end sum_of_x_y_l49_49327


namespace complementary_angle_measure_l49_49846

theorem complementary_angle_measure (x : ℝ) (h1 : 0 < x) (h2 : 4*x + x = 90) : 4*x = 72 :=
by
  sorry

end complementary_angle_measure_l49_49846


namespace range_of_a_l49_49915

theorem range_of_a (a : ℝ) : (∀ x > 1, x^2 ≥ a) ↔ (a ≤ 1) :=
by {
  sorry
}

end range_of_a_l49_49915


namespace minimum_value_fraction_l49_49045

theorem minimum_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  (2 / x + 1 / y) >= 2 * Real.sqrt 2 :=
sorry

end minimum_value_fraction_l49_49045


namespace old_edition_pages_l49_49150

theorem old_edition_pages (x : ℕ) 
  (h₁ : 2 * x - 230 = 450) : x = 340 := 
by sorry

end old_edition_pages_l49_49150


namespace probability_b_greater_than_a_l49_49139

open Probability

theorem probability_b_greater_than_a :
  (∃ (a : ℕ) (ha : a ∈ set.Icc 1 1000)
     (b : ℕ) (hb : b ∈ set.Icc 1 1000),
        b > a) →
  (∀ (a b : ℕ), (a ∈ set.Icc 1 1000) → (b ∈ set.Icc 1 1000) →
     P(b > a) = 0.4995) := sorry

end probability_b_greater_than_a_l49_49139


namespace polynomial_not_factorable_l49_49453

theorem polynomial_not_factorable (b c d : Int) (h₁ : (b * d + c * d) % 2 = 1) : 
  ¬ ∃ p q r : Int, (x + p) * (x^2 + q * x + r) = x^3 + b * x^2 + c * x + d :=
by 
  sorry

end polynomial_not_factorable_l49_49453


namespace min_value_xyz_l49_49499

-- Definition of the problem
theorem min_value_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z = 108):
  x^2 + 9 * x * y + 9 * y^2 + 3 * z^2 ≥ 324 :=
sorry

end min_value_xyz_l49_49499


namespace option_A_two_solutions_l49_49203

theorem option_A_two_solutions :
    (∀ (a b : ℝ) (A : ℝ), 
    (a = 3 ∧ b = 4 ∧ A = 45) ∨ 
    (a = 7 ∧ b = 14 ∧ A = 30) ∨ 
    (a = 2 ∧ b = 7 ∧ A = 60) ∨ 
    (a = 8 ∧ b = 5 ∧ A = 135) →
    (∃ a b A : ℝ, a = 3 ∧ b = 4 ∧ A = 45 ∧ 2 = 2)) :=
by
  sorry

end option_A_two_solutions_l49_49203


namespace num_pure_Gala_trees_l49_49269

-- Define the problem statement conditions
variables (T F G H : ℝ)
variables (c1 : 0.125 * F + 0.075 * F + F = 315)
variables (c2 : F = (2 / 3) * T)
variables (c3 : H = (1 / 6) * T)
variables (c4 : T = F + G + H)

-- Prove the number of pure Gala trees G is 66
theorem num_pure_Gala_trees : G = 66 :=
by
  -- Proof will be filled out here
  sorry

end num_pure_Gala_trees_l49_49269


namespace negation_of_exists_l49_49118

theorem negation_of_exists (x : ℝ) : x^2 + 2 * x + 2 > 0 := sorry

end negation_of_exists_l49_49118


namespace find_m_l49_49288

def point (α : Type) := (α × α)

def collinear {α : Type} [LinearOrderedField α] 
  (p1 p2 p3 : point α) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p2.1) = (p3.2 - p2.2) * (p2.1 - p1.1)

theorem find_m {m : ℚ} 
  (h : collinear (4, 10) (-3, m) (-12, 5)) : 
  m = 125 / 16 :=
by sorry

end find_m_l49_49288


namespace store_income_l49_49956

def pencil_store_income (p_with_eraser_qty p_with_eraser_cost p_regular_qty p_regular_cost p_short_qty p_short_cost : ℕ → ℝ) : ℝ :=
  (p_with_eraser_qty * p_with_eraser_cost) + (p_regular_qty * p_regular_cost) + (p_short_qty * p_short_cost)

theorem store_income : 
  pencil_store_income 200 0.8 40 0.5 35 0.4 = 194 := 
by sorry

end store_income_l49_49956


namespace two_point_distribution_success_prob_l49_49760

theorem two_point_distribution_success_prob (X : ℝ) (hX : E(X) = 0.7) :
  ∃ (p : ℝ), p = 0.7 :=
by
  sorry

end two_point_distribution_success_prob_l49_49760


namespace velocity_ratio_proof_l49_49390

noncomputable def velocity_ratio (V U : ℝ) : ℝ := V / U

-- The conditions:
-- 1. A smooth horizontal surface.
-- 2. The speed of the ball is perpendicular to the face of the block.
-- 3. The mass of the ball is much smaller than the mass of the block.
-- 4. The collision is elastic.
-- 5. After the collision, the ball’s speed is halved and it moves in the opposite direction.

def ball_block_collision 
    (V U U_final : ℝ) 
    (smooth_surface : Prop) 
    (perpendicular_impact : Prop) 
    (ball_much_smaller : Prop) 
    (elastic_collision : Prop) 
    (speed_halved : Prop) : Prop :=
  U_final = U ∧ V / U = 4

theorem velocity_ratio_proof : 
  ∀ (V U U_final : ℝ)
    (smooth_surface : Prop)
    (perpendicular_impact : Prop)
    (ball_much_smaller : Prop)
    (elastic_collision : Prop)
    (speed_halved : Prop),
    ball_block_collision V U U_final smooth_surface perpendicular_impact ball_much_smaller elastic_collision speed_halved := 
sorry

end velocity_ratio_proof_l49_49390


namespace tony_combined_lift_weight_l49_49842

theorem tony_combined_lift_weight :
  let curl_weight := 90
  let military_press_weight := 2 * curl_weight
  let squat_weight := 5 * military_press_weight
  let bench_press_weight := 1.5 * military_press_weight
  squat_weight + bench_press_weight = 1170 :=
by
  sorry

end tony_combined_lift_weight_l49_49842


namespace quadratic_has_real_roots_iff_l49_49947

theorem quadratic_has_real_roots_iff (k : ℝ) (hk : k ≠ 0) :
  (∃ x : ℝ, k * x^2 - x + 1 = 0) ↔ k ≤ 1 / 4 :=
by
  sorry

end quadratic_has_real_roots_iff_l49_49947


namespace lower_limit_of_b_l49_49470

theorem lower_limit_of_b (a : ℤ) (b : ℤ) (h₁ : 8 < a ∧ a < 15) (h₂ : ∃ x, x < b ∧ b < 21) (h₃ : (14 : ℚ) / b - (9 : ℚ) / b = 1.55) : b = 4 :=
by
  sorry

end lower_limit_of_b_l49_49470


namespace system_solution_l49_49916

theorem system_solution:
  let k := 115 / 12 
  ∃ x y z: ℝ, 
    x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ 
    (x + k * y + 5 * z = 0) ∧
    (4 * x + k * y - 3 * z = 0) ∧
    (3 * x + 5 * y - 4 * z = 0) ∧ 
    ((1 : ℝ) / 15 = (x * z) / (y * y)) := 
by sorry

end system_solution_l49_49916


namespace value_of_expression_l49_49854

theorem value_of_expression (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) : x * y - x = 9 := 
by
  sorry

end value_of_expression_l49_49854


namespace isosceles_triangle_perimeter_l49_49071

/-- 
  Given an isosceles triangle with two sides of length 6 and the third side of length 2,
  prove that the perimeter of the triangle is 14.
-/
theorem isosceles_triangle_perimeter (a b c : ℕ) (h1 : a = 6) (h2 : b = 6) (h3 : c = 2) 
  (triangle_ineq1 : a + b > c) (triangle_ineq2 : a + c > b) (triangle_ineq3 : b + c > a) :
  a + b + c = 14 :=
  sorry

end isosceles_triangle_perimeter_l49_49071


namespace smallest_whole_number_larger_than_perimeter_l49_49378

theorem smallest_whole_number_larger_than_perimeter (c : ℝ) (h1 : 13 < c) (h2 : c < 25) : 50 = Nat.ceil (6 + 19 + c) :=
by
  sorry

end smallest_whole_number_larger_than_perimeter_l49_49378


namespace count_negative_rationals_is_two_l49_49282

theorem count_negative_rationals_is_two :
  let a := (-1 : ℚ) ^ 2007
  let b := (|(-1 : ℚ)| ^ 3)
  let c := -(1 : ℚ) ^ 18
  let d := (18 : ℚ)
  (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) + (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0) = 2 := by
  sorry

end count_negative_rationals_is_two_l49_49282


namespace div_rule_2701_is_37_or_73_l49_49222

theorem div_rule_2701_is_37_or_73 (a b x : ℕ) (h1 : 10 * a + b = x) (h2 : a^2 + b^2 = 58) : 
  (x = 37 ∨ x = 73) ↔ 2701 % x = 0 :=
by
  sorry

end div_rule_2701_is_37_or_73_l49_49222


namespace translate_parabola_l49_49251

theorem translate_parabola :
  ∀ (x y : ℝ), y = -5*x^2 + 1 → y = -5*(x + 1)^2 - 1 := by
  sorry

end translate_parabola_l49_49251


namespace triangle_area_l49_49718

theorem triangle_area (a b c : ℕ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 54 := by
    sorry

end triangle_area_l49_49718


namespace eggs_in_basket_l49_49323

theorem eggs_in_basket (x : ℕ) (h₁ : 600 / x + 1 = 600 / (x - 20)) : x = 120 :=
sorry

end eggs_in_basket_l49_49323


namespace solve_for_x_l49_49103

theorem solve_for_x : ∀ x : ℝ, (x - 3) ≠ 0 → (x + 6) / (x - 3) = 4 → x = 6 :=
by 
  intros x hx h
  sorry

end solve_for_x_l49_49103


namespace intersection_of_A_and_B_l49_49186

def A : Set ℝ := { x | -1 ≤ x ∧ x < 3 }
def B : Set ℝ := { y | 2 ≤ y ∧ y ≤ 5 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | 2 ≤ x ∧ x < 3 } :=
sorry

end intersection_of_A_and_B_l49_49186


namespace domain_of_f_l49_49537

noncomputable def f (x : ℝ) := 1 / ((x - 3) + (x - 6))

theorem domain_of_f :
  (∀ x : ℝ, x ≠ 9/2 → ∃ y : ℝ, f x = y) ∧ (∀ x : ℝ, x = 9/2 → ¬ (∃ y : ℝ, f x = y)) :=
by
  sorry

end domain_of_f_l49_49537


namespace maria_walk_to_school_l49_49641

variable (w s : ℝ)

theorem maria_walk_to_school (h1 : 25 * w + 13 * s = 38) (h2 : 11 * w + 20 * s = 31) : 
  51 = 51 := by
  sorry

end maria_walk_to_school_l49_49641


namespace mark_total_cost_is_correct_l49_49355

variable (hours : ℕ) (hourly_rate part_cost : ℕ)

def total_cost (hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) :=
  hours * hourly_rate + part_cost

theorem mark_total_cost_is_correct : 
  hours = 2 → hourly_rate = 75 → part_cost = 150 → total_cost hours hourly_rate part_cost = 300 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end mark_total_cost_is_correct_l49_49355


namespace old_edition_pages_l49_49147

-- Define the conditions
variables (new_edition : ℕ) (old_edition : ℕ)

-- The conditions given in the problem
axiom new_edition_pages : new_edition = 450
axiom pages_relationship : new_edition = 2 * old_edition - 230

-- Goal: Prove that the old edition Geometry book had 340 pages
theorem old_edition_pages : old_edition = 340 :=
by sorry

end old_edition_pages_l49_49147


namespace smallest_integer_k_l49_49437

theorem smallest_integer_k (k : ℕ) : 
  (k > 1 ∧ 
   k % 13 = 1 ∧ 
   k % 7 = 1 ∧ 
   k % 5 = 1 ∧ 
   k % 3 = 1) ↔ k = 1366 := 
sorry

end smallest_integer_k_l49_49437


namespace primes_between_30_and_50_l49_49467

theorem primes_between_30_and_50 : (Finset.card (Finset.filter Nat.Prime (Finset.Ico 30 51))) = 5 :=
by
  sorry

end primes_between_30_and_50_l49_49467


namespace area_of_right_triangle_l49_49715

theorem area_of_right_triangle
    (a b c : ℝ)
    (h₀ : a = 9)
    (h₁ : b = 12)
    (h₂ : c = 15)
    (right_triangle : a^2 + b^2 = c^2) :
    (1 / 2) * a * b = 54 := by
  sorry

end area_of_right_triangle_l49_49715


namespace isosceles_triangle_base_length_l49_49667

theorem isosceles_triangle_base_length
  (a b : ℕ)
  (ha : a = 8)
  (hp : 2 * a + b = 25)
  : b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l49_49667


namespace number_of_mismatching_socks_l49_49236

-- Define the conditions
def total_socks : Nat := 25
def pairs_of_matching_socks : Nat := 4
def socks_per_pair : Nat := 2
def matching_socks : Nat := pairs_of_matching_socks * socks_per_pair

-- State the theorem
theorem number_of_mismatching_socks : total_socks - matching_socks = 17 :=
by
  -- Skip the proof
  sorry

end number_of_mismatching_socks_l49_49236


namespace donut_combinations_l49_49569

theorem donut_combinations {α : Type} (k1 k2 k3 k4 : α → Prop) :
  (∃ f : α → ℕ, (∀ x, k1 x → f x ≥ 1) ∧ (∀ x, k2 x → f x ≥ 1) ∧ (∀ x, k3 x → f x ≥ 1) ∧ (∀ x, k4 x → f x ≥ 1) ∧ ∑ x, f x = 6) →
  (∃ n : ℕ, n = 10) :=
by
  sorry

end donut_combinations_l49_49569


namespace piglets_straws_l49_49844

theorem piglets_straws (straws : ℕ) (fraction_adult_pigs : ℚ) (number_of_piglets : ℕ)
  (h₁ : straws = 300) (h₂ : fraction_adult_pigs = 3/5) (h₃ : number_of_piglets = 20) :
  let straws_for_adults := fraction_adult_pigs * straws in
  let straws_for_piglets := straws_for_adults in
  let straws_per_piglet := straws_for_piglets / number_of_piglets in
  straws_per_piglet = 9 :=
by
  sorry

end piglets_straws_l49_49844


namespace difference_is_24_l49_49487

namespace BuffaloesAndDucks

def numLegs (B D : ℕ) : ℕ := 4 * B + 2 * D

def numHeads (B D : ℕ) : ℕ := B + D

def diffLegsAndHeads (B D : ℕ) : ℕ := numLegs B D - 2 * numHeads B D

theorem difference_is_24 (D : ℕ) : diffLegsAndHeads 12 D = 24 := by
  sorry

end BuffaloesAndDucks

end difference_is_24_l49_49487


namespace union_of_sets_l49_49810

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

-- State the proof problem
theorem union_of_sets : A ∪ B = {x | -1 < x ∧ x < 3} := by
  sorry

end union_of_sets_l49_49810


namespace degree_of_monomial_x_l49_49693

def is_monomial (e : Expr) : Prop := sorry -- Placeholder definition
def degree (e : Expr) : Nat := sorry -- Placeholder definition

theorem degree_of_monomial_x :
  degree x = 1 :=
by
  sorry

end degree_of_monomial_x_l49_49693


namespace true_propositions_l49_49542

open Set

theorem true_propositions (M N : Set ℕ) (a b m : ℕ) (h1 : M ⊆ N) 
  (h2 : a > b) (h3 : b > 0) (h4 : m > 0) (p : ∀ x : ℝ, x > 0) :
  (M ⊆ M ∪ N) ∧ ((b + m) / (a + m) > b / a) ∧ 
  ¬(∀ (a b c : ℝ), a = b ↔ a * c ^ 2 = b * c ^ 2) ∧ 
  ¬(∃ x₀ : ℝ, x₀ ≤ 0) := sorry

end true_propositions_l49_49542


namespace number_of_primes_between_30_and_50_l49_49458

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the interval condition
def in_interval (n : ℕ) : Prop :=
  30 ≤ n ∧ n ≤ 50

-- Define the count of prime numbers in the interval
def prime_count_in_interval : ℕ :=
  (List.range' 30 21).countp (λ n, is_prime n)

-- We state that the above count is equal to 5
theorem number_of_primes_between_30_and_50 : prime_count_in_interval = 5 :=
  sorry

end number_of_primes_between_30_and_50_l49_49458


namespace range_m_l49_49757

-- Definitions for propositions p and q
def p (m : ℝ) : Prop :=
  (m^2 - 4 > 0) ∧ (-m < 0)

def q (m : ℝ) : Prop :=
  (m - 1 > 0)

-- Given conditions:
-- 1. p ∨ q is true
-- 2. p ∧ q is false

theorem range_m (m : ℝ) (h1: p m ∨ q m) (h2: ¬ (p m ∧ q m)) : 1 < m ∧ m ≤ 2 :=
by
  sorry

end range_m_l49_49757


namespace intersection_of_A_and_B_union_of_A_and_B_l49_49345

def A : Set ℝ := {x | x * (9 - x) > 0}
def B : Set ℝ := {x | x ≤ 3}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x ≤ 3} :=
sorry

theorem union_of_A_and_B : A ∪ B = {x | x < 9} :=
sorry

end intersection_of_A_and_B_union_of_A_and_B_l49_49345


namespace xy_yx_eq_zy_yz_eq_xz_zx_l49_49512

theorem xy_yx_eq_zy_yz_eq_xz_zx 
  (x y z : ℝ) 
  (h : x * (y + z - x) / x = y * (z + x - y) / y ∧ y * (z + x - y) / y = z * (x + y - z) / z): 
  x ^ y * y ^ x = z ^ y * y ^ z ∧ z ^ y * y ^ z = x ^ z * z ^ x :=
by
  sorry

end xy_yx_eq_zy_yz_eq_xz_zx_l49_49512


namespace rhombic_dodecahedron_surface_area_rhombic_dodecahedron_volume_l49_49660

noncomputable def surface_area_rhombic_dodecahedron (a : ℝ) : ℝ :=
  6 * (a ^ 2) * Real.sqrt 2

noncomputable def volume_rhombic_dodecahedron (a : ℝ) : ℝ :=
  2 * (a ^ 3)

theorem rhombic_dodecahedron_surface_area (a : ℝ) :
  surface_area_rhombic_dodecahedron a = 6 * (a ^ 2) * Real.sqrt 2 :=
by
  sorry

theorem rhombic_dodecahedron_volume (a : ℝ) :
  volume_rhombic_dodecahedron a = 2 * (a ^ 3) :=
by
  sorry

end rhombic_dodecahedron_surface_area_rhombic_dodecahedron_volume_l49_49660


namespace meet_at_midpoint_l49_49847

open Classical

noncomputable def distance_travel1 (t : ℝ) : ℝ :=
  4 * t

noncomputable def distance_travel2 (t : ℝ) : ℝ :=
  (t / 2) * (3.5 + 0.5 * t)

theorem meet_at_midpoint (t : ℝ) : 
  (4 * t + (t / 2) * (3.5 + 0.5 * t) = 72) → 
  (t = 9) ∧ (4 * t = 36) := 
 by 
  sorry

end meet_at_midpoint_l49_49847


namespace factorization_eq_l49_49746

variable (x y : ℝ)

theorem factorization_eq : 9 * y - 25 * x^2 * y = y * (3 + 5 * x) * (3 - 5 * x) :=
by sorry 

end factorization_eq_l49_49746


namespace repeatingDecimal_proof_l49_49579

noncomputable def repeatingDecimalToFraction (x : ℚ) (y : ℚ): ℚ :=
  0.3 + x

theorem repeatingDecimal_proof : (0.3 + 0.23 + 0.00023 + 0.0000023 + ...) = (527 / 990) :=
by
  sorry

end repeatingDecimal_proof_l49_49579


namespace distinct_a_count_l49_49031

theorem distinct_a_count :
  ∃ (a_set : Set ℝ), (∀ x ∈ a_set, ∃ r s : ℤ, r + s = -x ∧ r * s = 9 * x) ∧ a_set.toFinset.card = 3 :=
by 
  sorry

end distinct_a_count_l49_49031


namespace stock_price_end_of_third_year_l49_49745

def stock_price_after_years (initial_price : ℝ) (year1_increase : ℝ) (year2_decrease : ℝ) (year3_increase : ℝ) : ℝ :=
  let price_after_year1 := initial_price * (1 + year1_increase)
  let price_after_year2 := price_after_year1 * (1 - year2_decrease)
  let price_after_year3 := price_after_year2 * (1 + year3_increase)
  price_after_year3

theorem stock_price_end_of_third_year :
  stock_price_after_years 120 0.80 0.30 0.50 = 226.8 := 
by
  sorry

end stock_price_end_of_third_year_l49_49745


namespace solve_inequality_system_l49_49232

theorem solve_inequality_system (x : ℝ) (h1 : 3 * x - 2 < x) (h2 : (1 / 3) * x < -2) : x < -6 :=
sorry

end solve_inequality_system_l49_49232


namespace simplify_and_evaluate_l49_49226

theorem simplify_and_evaluate (x y : ℤ) (hx : x = -1) (hy : y = 2) : 
  x^2 - 2 * (3 * y^2 - x * y) + (y^2 - 2 * x * y) = -19 := 
by
  -- Proof will go here, but it's omitted as per instructions
  sorry

end simplify_and_evaluate_l49_49226


namespace initial_customers_l49_49277

theorem initial_customers (x : ℕ) (h1 : x - 31 + 26 = 28) : x = 33 := 
by 
  sorry

end initial_customers_l49_49277


namespace max_ratio_of_three_digit_to_sum_l49_49017

theorem max_ratio_of_three_digit_to_sum (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 9)
  (hb : 0 ≤ b ∧ b ≤ 9)
  (hc : 0 ≤ c ∧ c ≤ 9) :
  (100 * a + 10 * b + c) / (a + b + c) ≤ 100 :=
by sorry

end max_ratio_of_three_digit_to_sum_l49_49017


namespace normal_distribution_property_l49_49086

noncomputable def ξ : MeasureTheory.ProbabilityDistribution := 
  MeasureTheory.ProbabilityDistribution.normal 10 σ^2

theorem normal_distribution_property (hξ : ∀ a b, MeasureTheory.ProbabilityMeasure (N(10, σ^2)) a b) :
  ProbabilityTheory.Prob (λ x, |x - 10| < 1) = 0.8 :=
by
  -- Given: ξ ~ N(10, σ^2)
  have h1 : MeasureTheory.ProbabilityMeasure (N(10, σ^2)) := ξ
  -- Given: P(ξ < 11) = 0.9
  have h2 : ProbabilityTheory.Prob (λ x, x < 11) = 0.9 :=
    sorry -- this would follow from the distribution properties
  -- Prove: P(|ξ - 10| < 1) = 0.8
  sorry

end normal_distribution_property_l49_49086


namespace isosceles_triangle_l49_49796

-- Let ∆ABC be a triangle with angles A, B, and C
variables {A B C : ℝ}

-- Given condition: 2 * cos B * sin A = sin C
def condition (A B C : ℝ) : Prop := 2 * Real.cos B * Real.sin A = Real.sin C

-- Problem: Given the condition, we need to prove that ∆ABC is an isosceles triangle, meaning A = B.
theorem isosceles_triangle (A B C : ℝ) (h : condition A B C) : A = B :=
by
  sorry

end isosceles_triangle_l49_49796


namespace quadratic_has_single_real_root_l49_49475

theorem quadratic_has_single_real_root (n : ℝ) (h : (6 * n) ^ 2 - 4 * 1 * (2 * n) = 0) : n = 2 / 9 :=
by
  sorry

end quadratic_has_single_real_root_l49_49475


namespace exists_integers_a_b_l49_49582

theorem exists_integers_a_b : 
  ∃ (a b : ℤ), 2003 < a + b * (Real.sqrt 2) ∧ a + b * (Real.sqrt 2) < 2003.01 :=
by
  sorry

end exists_integers_a_b_l49_49582


namespace combined_bus_capacity_l49_49982

-- Define conditions
def train_capacity : ℕ := 120
def bus_capacity : ℕ := train_capacity / 6
def number_of_buses : ℕ := 2

-- Define theorem for the combined capacity of two buses
theorem combined_bus_capacity : number_of_buses * bus_capacity = 40 := by
  -- We declare that the proof is skipped here
  sorry

end combined_bus_capacity_l49_49982


namespace truck_weight_l49_49726

theorem truck_weight (T R : ℝ) (h1 : T + R = 7000) (h2 : R = 0.5 * T - 200) : T = 4800 :=
by sorry

end truck_weight_l49_49726


namespace joes_mean_score_is_88_83_l49_49210

def joesQuizScores : List ℕ := [88, 92, 95, 81, 90, 87]

noncomputable def mean (lst : List ℕ) : ℝ := (lst.sum : ℝ) / lst.length

theorem joes_mean_score_is_88_83 :
  mean joesQuizScores = 88.83 := 
sorry

end joes_mean_score_is_88_83_l49_49210


namespace total_money_of_james_and_ali_l49_49078

def jamesOwns : ℕ := 145
def jamesAliDifference : ℕ := 40
def aliOwns : ℕ := jamesOwns - jamesAliDifference

theorem total_money_of_james_and_ali :
  jamesOwns + aliOwns = 250 := by
  sorry

end total_money_of_james_and_ali_l49_49078


namespace unique_intersection_l49_49733

open Real

-- Defining the functions f and g as per the conditions
def f (b : ℝ) (x : ℝ) : ℝ := b * x^2 + 5 * x + 3
def g (x : ℝ) : ℝ := -2 * x - 2

-- The condition that the intersection occurs at one point translates to a specific b satisfying the discriminant condition.
theorem unique_intersection (b : ℝ) : (∃ x : ℝ, f b x = g x) ∧ (f b x = g x → ∀ y : ℝ, y ≠ x → f b y ≠ g y) ↔ b = 49 / 20 :=
by {
  sorry
}

end unique_intersection_l49_49733


namespace a_100_value_l49_49181

variables (S : ℕ → ℚ) (a : ℕ → ℚ)

def S_n (n : ℕ) : ℚ := S n
def a_n (n : ℕ) : ℚ := a n

axiom a1_eq_3 : a 1 = 3
axiom a_n_formula (n : ℕ) (hn : n ≥ 2) : a n = (3 * S n ^ 2) / (3 * S n - 2)

theorem a_100_value : a 100 = -3 / 88401 :=
sorry

end a_100_value_l49_49181


namespace Kara_books_proof_l49_49571

-- Let's define the conditions and the proof statement in Lean 4

def Candice_books : ℕ := 18
def Amanda_books := Candice_books / 3
def Kara_books := Amanda_books / 2

theorem Kara_books_proof : Kara_books = 3 := by
  -- setting up the conditions based on the given problem.
  have Amanda_books_correct : Amanda_books = 6 := by
    exact Nat.div_eq_of_eq_mul_right (Nat.zero_lt_succ 2) (rfl) -- 18 / 3 = 6

  have Kara_books_correct : Kara_books = 3 := by
    exact Nat.div_eq_of_eq_mul_right (Nat.zero_lt_succ 1) Amanda_books_correct -- 6 / 2 = 3

  exact Kara_books_correct

end Kara_books_proof_l49_49571


namespace swap_equality_l49_49511

theorem swap_equality {a1 b1 a2 b2 : ℝ} 
  (h1 : a1^2 + b1^2 = 1)
  (h2 : a2^2 + b2^2 = 1)
  (h3 : a1 * a2 + b1 * b2 = 0) :
  b1 = a2 ∨ b1 = -a2 :=
by sorry

end swap_equality_l49_49511


namespace evaluate_expression_l49_49159

theorem evaluate_expression (a b c : ℕ) (h1 : a = 12) (h2 : b = 8) (h3 : c = 3) :
  (a - b + c - (a - (b + c)) = 6) := by
  sorry

end evaluate_expression_l49_49159


namespace total_sticks_used_l49_49040

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

end total_sticks_used_l49_49040


namespace intersection_A_B_union_A_complement_B_subset_C_B_range_l49_49310

def set_A : Set ℝ := { x | 1 ≤ x ∧ x < 6 }
def set_B : Set ℝ := { x | 2 < x ∧ x < 9 }
def set_C (a : ℝ) : Set ℝ := { x | a < x ∧ x < a + 1 }

theorem intersection_A_B :
  set_A ∩ set_B = { x | 2 < x ∧ x < 6 } :=
sorry

theorem union_A_complement_B :
  set_A ∪ (compl set_B) = { x | x < 6 } ∪ { x | x ≥ 9 } :=
sorry

theorem subset_C_B_range (a : ℝ) :
  (set_C a ⊆ set_B) → (2 ≤ a ∧ a ≤ 8) :=
sorry

end intersection_A_B_union_A_complement_B_subset_C_B_range_l49_49310


namespace cards_distribution_l49_49789

open Nat

theorem cards_distribution : 
  ∀ (total_cards people : Nat), total_cards = 60 → people = 9 → 
  let base_cards := total_cards / people;
  let remainder := total_cards % people;
  let num_with_more := remainder;
  let num_with_fewer := people - remainder;
  num_with_fewer = 3 :=
by
  intros total_cards people h_total h_people
  let base_cards := total_cards / people
  let remainder := total_cards % people
  let num_with_more := remainder
  let num_with_fewer := people - remainder
  have h_base_cards : base_cards = 6 := by sorry
  have h_remainder : remainder = 6 := by sorry
  have h_num_with_more : num_with_more = 6 := by rw [h_remainder]; sorry
  have h_num_with_fewer : num_with_fewer = people - remainder := by sorry
  rw [h_people, h_remainder]
  exact rfl

end cards_distribution_l49_49789


namespace relationship_between_x_b_a_l49_49754

variable {x b a : ℝ}

theorem relationship_between_x_b_a 
  (hx : x < 0) (hb : b < 0) (ha : a < 0)
  (hxb : x < b) (hba : b < a) : x^2 > b * x ∧ b * x > b^2 :=
by sorry

end relationship_between_x_b_a_l49_49754


namespace find_f_find_g_l49_49171

-- Problem 1: Finding f(x) given f(x+1) = x^2 - 2x
theorem find_f (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 2 * x) :
  ∀ x, f x = x^2 - 4 * x + 3 :=
sorry

-- Problem 2: Finding g(x) given roots and a point
theorem find_g (g : ℝ → ℝ) (h1 : g (-2) = 0) (h2 : g 3 = 0) (h3 : g 0 = -3) :
  ∀ x, g x = (1 / 2) * x^2 - (1 / 2) * x - 3 :=
sorry

end find_f_find_g_l49_49171


namespace number_of_customers_l49_49572

-- Definitions based on conditions
def popularity (p : ℕ) (c w : ℕ) (k : ℝ) : Prop :=
  p = k * (w / c)

-- Given values
def given_values : Prop :=
  ∃ k : ℝ, popularity 15 500 1000 k

-- Problem statement
theorem number_of_customers:
  given_values →
  popularity 15 600 1200 7.5 :=
by
  intro h
  -- Proof omitted
  sorry

end number_of_customers_l49_49572


namespace complex_magnitude_condition_l49_49758

noncomputable def magnitude_of_z (z : ℂ) : ℝ :=
  Complex.abs z

theorem complex_magnitude_condition (z : ℂ) (i : ℂ) (h : i * i = -1) (h1 : z - 2 * i = 1 + z * i) :
  magnitude_of_z z = Real.sqrt (10) / 2 :=
by
  -- proof goes here
  sorry

end complex_magnitude_condition_l49_49758


namespace number_of_mismatching_socks_l49_49235

-- Define the conditions
def total_socks : Nat := 25
def pairs_of_matching_socks : Nat := 4
def socks_per_pair : Nat := 2
def matching_socks : Nat := pairs_of_matching_socks * socks_per_pair

-- State the theorem
theorem number_of_mismatching_socks : total_socks - matching_socks = 17 :=
by
  -- Skip the proof
  sorry

end number_of_mismatching_socks_l49_49235


namespace average_number_of_fish_is_75_l49_49507

-- Define the conditions
def BoastPool_fish := 75
def OnumLake_fish := BoastPool_fish + 25
def RiddlePond_fish := OnumLake_fish / 2

-- Prove the average number of fish
theorem average_number_of_fish_is_75 :
  (BoastPool_fish + OnumLake_fish + RiddlePond_fish) / 3 = 75 :=
by
  sorry

end average_number_of_fish_is_75_l49_49507


namespace bucket_fill_proof_l49_49381

variables (x y : ℕ)
def tank_capacity : ℕ := 4 * x

theorem bucket_fill_proof (hx: y = x + 4) (hy: 4 * x = 3 * y): tank_capacity x = 48 :=
by {
  -- Proof steps will be here, but are elided for now
  sorry 
}

end bucket_fill_proof_l49_49381


namespace no_finite_operations_l49_49604

noncomputable def P (x : ℝ) := (x^2 - 1)^2023
noncomputable def Q (x : ℝ) := (2 * x + 1)^14
noncomputable def R (x : ℝ) := (2 * x + 1 + 2 / x)^34

theorem no_finite_operations :
  ∀ S : set (ℝ → ℝ),
    (P ∈ S ∧ Q ∈ S) ∨ (P ∈ S ∧ R ∈ S) ∨ (Q ∈ S ∧ R ∈ S) →
    ∀ f : ℝ → ℝ,
      ((∃ p q ∈ S, f = p + q) ∨ (∃ p q ∈ S, f = p - q) ∨ (∃ p q ∈ S, f = p * q) ∨
       (∃ p ∈ S, ∃ k : ℕ, k > 0 ∧ f = p ^ k) ∨ (∃ p ∈ S, ∃ t : ℝ, f = p + t) ∨
       (∃ p ∈ S, ∃ t : ℝ, f = p - t) ∨ (∃ p ∈ S, ∃ t : ℝ, f = p * t)) →
      ¬ (f = P ∨ f = Q ∨ f = R) :=
by
  intro S h_initial f h_operations
  sorry

end no_finite_operations_l49_49604


namespace impossible_cube_configuration_l49_49710

theorem impossible_cube_configuration :
  ∀ (cube: ℕ → ℕ) (n : ℕ), 
    (∀ n, 1 ≤ n ∧ n ≤ 27 → ∃ k, 1 ≤ k ∧ k ≤ 27 ∧ cube k = n) →
    (∀ n, 1 ≤ n ∧ n ≤ 27 → (cube 27 = 27 ∧ ∀ m, 1 ≤ m ∧ m ≤ 26 → cube m = 27 - m)) → 
    false :=
by
  intros cube n hcube htarget
  -- any detailed proof steps would go here, skipping with sorry
  sorry

end impossible_cube_configuration_l49_49710


namespace total_cost_of_repair_l49_49353

theorem total_cost_of_repair (hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) (H1 : hours = 2) (H2 : hourly_rate = 75) (H3 : part_cost = 150) :
  hours * hourly_rate + part_cost = 300 := 
by
  sorry

end total_cost_of_repair_l49_49353


namespace stationery_store_sales_l49_49953

theorem stationery_store_sales :
  let price_pencil_eraser := 0.8
  let price_regular_pencil := 0.5
  let price_short_pencil := 0.4
  let num_pencil_eraser := 200
  let num_regular_pencil := 40
  let num_short_pencil := 35
  (num_pencil_eraser * price_pencil_eraser) +
  (num_regular_pencil * price_regular_pencil) +
  (num_short_pencil * price_short_pencil) = 194 :=
by
  sorry

end stationery_store_sales_l49_49953


namespace range_of_m_l49_49242

theorem range_of_m (m : ℝ) : 
    (∀ x : ℝ, mx^2 - 6 * m * x + m + 8 ≥ 0) ↔ (0 ≤ m ∧ m ≤ 1) :=
sorry

end range_of_m_l49_49242


namespace cards_distribution_l49_49788

open Nat

theorem cards_distribution : 
  ∀ (total_cards people : Nat), total_cards = 60 → people = 9 → 
  let base_cards := total_cards / people;
  let remainder := total_cards % people;
  let num_with_more := remainder;
  let num_with_fewer := people - remainder;
  num_with_fewer = 3 :=
by
  intros total_cards people h_total h_people
  let base_cards := total_cards / people
  let remainder := total_cards % people
  let num_with_more := remainder
  let num_with_fewer := people - remainder
  have h_base_cards : base_cards = 6 := by sorry
  have h_remainder : remainder = 6 := by sorry
  have h_num_with_more : num_with_more = 6 := by rw [h_remainder]; sorry
  have h_num_with_fewer : num_with_fewer = people - remainder := by sorry
  rw [h_people, h_remainder]
  exact rfl

end cards_distribution_l49_49788


namespace combined_boys_average_l49_49566

noncomputable def average_boys_score (C c D d : ℕ) : ℚ :=
  (68 * C + 74 * 3 * c / 4) / (C + 3 * c / 4)

theorem combined_boys_average:
  ∀ (C c D d : ℕ),
  (68 * C + 72 * c) / (C + c) = 70 →
  (74 * D + 88 * d) / (D + d) = 82 →
  (72 * c + 88 * d) / (c + d) = 83 →
  C = c →
  4 * D = 3 * d →
  average_boys_score C c D d = 48.57 :=
by
  intros C c D d h_clinton h_dixon h_combined_girls h_C_eq_c h_D_eq_d
  sorry

end combined_boys_average_l49_49566


namespace final_middle_pile_cards_l49_49617

-- Definitions based on conditions
def initial_cards_per_pile (n : ℕ) (h : n ≥ 2) := n

def left_pile_after_step_2 (n : ℕ) (h : n ≥ 2) := n - 2
def middle_pile_after_step_2 (n : ℕ) (h : n ≥ 2) := n + 2
def right_pile_after_step_2 (n : ℕ) (h : n ≥ 2) := n

def right_pile_after_step_3 (n : ℕ) (h : n ≥ 2) := n - 1
def middle_pile_after_step_3 (n : ℕ) (h : n ≥ 2) := n + 3

def left_pile_after_step_4 (n : ℕ) (h : n ≥ 2) := n
def middle_pile_after_step_4 (n : ℕ) (h : n ≥ 2) := (n + 3) - n

-- The proof problem to solve
theorem final_middle_pile_cards (n : ℕ) (h : n ≥ 2) : middle_pile_after_step_4 n h = 5 :=
sorry

end final_middle_pile_cards_l49_49617


namespace fraction_minimum_decimal_digits_l49_49130

def minimum_decimal_digits (n d : ℕ) : ℕ := sorry

theorem fraction_minimum_decimal_digits :
  minimum_decimal_digits 987654321 (2^28 * 5^3) = 28 :=
sorry

end fraction_minimum_decimal_digits_l49_49130


namespace total_worth_of_stock_l49_49871

theorem total_worth_of_stock (total_worth profit_fraction profit_rate loss_fraction loss_rate overall_loss : ℝ) :
  profit_fraction = 0.20 ->
  profit_rate = 0.20 -> 
  loss_fraction = 0.80 -> 
  loss_rate = 0.10 -> 
  overall_loss = 500 ->
  total_worth - (profit_fraction * total_worth * profit_rate) - (loss_fraction * total_worth * loss_rate) = overall_loss ->
  total_worth = 12500 :=
by
  sorry

end total_worth_of_stock_l49_49871


namespace total_students_l49_49798

theorem total_students (boys girls : ℕ) (h_ratio : 5 * girls = 7 * boys) (h_girls : girls = 140) :
  boys + girls = 240 :=
sorry

end total_students_l49_49798


namespace product_of_intersection_coordinates_l49_49163

noncomputable def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 5)^2 = 1
noncomputable def circle2 (x y : ℝ) : Prop := (x - 5)^2 + (y - 5)^2 = 4

theorem product_of_intersection_coordinates :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y ∧ x * y = 15 :=
by
  sorry

end product_of_intersection_coordinates_l49_49163


namespace number_divisible_by_37_l49_49803

def consecutive_ones_1998 : ℕ := (10 ^ 1998 - 1) / 9

theorem number_divisible_by_37 : 37 ∣ consecutive_ones_1998 :=
sorry

end number_divisible_by_37_l49_49803


namespace cube_net_count_l49_49120

/-- A net of a cube is a two-dimensional arrangement of six squares.
    A regular tetrahedron has exactly 2 unique nets.
    For a cube, consider all possible ways in which the six faces can be arranged such that they 
    form a cube when properly folded. -/
theorem cube_net_count : cube_nets_count = 11 :=
sorry

end cube_net_count_l49_49120


namespace count_primes_between_30_and_50_l49_49457

-- Define the range of numbers from 30 to 50
def range_30_to_50 := Set.of_list (List.range' 30 (51 - 30))

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Extract all prime numbers in the specified range
def primes_between_30_and_50 : List ℕ :=
  List.filter is_prime (Set.toList range_30_to_50)

theorem count_primes_between_30_and_50 : primes_between_30_and_50.length = 5 :=
by
  -- The proof goes here
  sorry

end count_primes_between_30_and_50_l49_49457


namespace derivative_of_f_l49_49824

variable (x : ℝ)
def f (x : ℝ) := (5 * x - 4) ^ 3

theorem derivative_of_f :
  (deriv f x) = 15 * (5 * x - 4) ^ 2 :=
sorry

end derivative_of_f_l49_49824


namespace solve_monomial_equation_l49_49795

theorem solve_monomial_equation (x : ℝ) (m n : ℝ) (a b : ℝ) 
  (h1 : m = 2) (h2 : n = 3) 
  (h3 : (1/3) * a^m * b^3 + (-2) * a^2 * b^n = (1/3) * a^2 * b^3 + (-2) * a^2 * b^3) :
  (x - 7) / n - (1 + x) / m = 1 → x = -23 := 
by
  sorry

end solve_monomial_equation_l49_49795


namespace problem_statement_l49_49919

theorem problem_statement
  (a b c d : ℕ)
  (h1 : (b + c + d) / 3 + 2 * a = 54)
  (h2 : (a + c + d) / 3 + 2 * b = 50)
  (h3 : (a + b + d) / 3 + 2 * c = 42)
  (h4 : (a + b + c) / 3 + 2 * d = 30) :
  a = 17 ∨ b = 17 ∨ c = 17 ∨ d = 17 :=
by
  sorry

end problem_statement_l49_49919


namespace base_length_of_isosceles_triangle_l49_49661

theorem base_length_of_isosceles_triangle (a b : ℕ) 
    (h₁ : a = 8) 
    (h₂ : 2 * a + b = 25) : 
    b = 9 :=
by
  -- This is the proof stub. Proof will be provided here.
  sorry

end base_length_of_isosceles_triangle_l49_49661


namespace city_roads_different_colors_l49_49619

-- Definitions and conditions
def Intersection (α : Type) := α × α × α

def City (α : Type) :=
  { intersections : α → Intersection α // 
    ∀ i : α, ∃ c₁ c₂ c₃ : α, intersections i = (c₁, c₂, c₃) 
    ∧ c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₃ ≠ c₁ 
  }

variables {α : Type}

-- Statement to prove that the three roads leading out of the city have different colors
theorem city_roads_different_colors (c : City α) 
  (roads_outside : α → Prop)
  (h : ∃ r₁ r₂ r₃, roads_outside r₁ ∧ roads_outside r₂ ∧ roads_outside r₃ ∧ 
  r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₃ ≠ r₁) : 
  true := 
sorry

end city_roads_different_colors_l49_49619


namespace soccer_ball_problem_l49_49141

-- Definitions of conditions
def price_eqs (x y : ℕ) : Prop :=
  x + 2 * y = 800 ∧ 3 * x + 2 * y = 1200

def total_cost_constraint (m : ℕ) : Prop :=
  200 * m + 300 * (20 - m) ≤ 5000 ∧ 1 ≤ m ∧ m ≤ 19

def store_discounts (x y : ℕ) (m : ℕ) : Prop :=
  200 * m + (3 / 5) * 300 * (20 - m) = (200 * m + (3 / 5) * 300 * (20 - m))

-- Main problem statement
theorem soccer_ball_problem :
  ∃ (x y m : ℕ), price_eqs x y ∧ total_cost_constraint m ∧ store_discounts x y m :=
sorry

end soccer_ball_problem_l49_49141


namespace prime_count_30_to_50_l49_49464

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

def primes_in_range (a b: ℕ) : list ℕ :=
  list.filter is_prime (list.range' a (b - a + 1))

theorem prime_count_30_to_50 : (primes_in_range 30 50).length = 5 :=
by sorry

end prime_count_30_to_50_l49_49464


namespace passed_candidates_count_l49_49859

theorem passed_candidates_count
    (average_total : ℝ)
    (number_candidates : ℕ)
    (average_passed : ℝ)
    (average_failed : ℝ)
    (total_marks : ℝ) :
    average_total = 35 →
    number_candidates = 120 →
    average_passed = 39 →
    average_failed = 15 →
    total_marks = average_total * number_candidates →
    (∃ P F, P + F = number_candidates ∧ 39 * P + 15 * F = total_marks ∧ P = 100) :=
by
  sorry

end passed_candidates_count_l49_49859


namespace collinear_iff_real_simple_ratio_l49_49698

theorem collinear_iff_real_simple_ratio (a b c : ℂ) : (∃ k : ℝ, a = k * b + (1 - k) * c) ↔ ∃ r : ℝ, (a - b) / (a - c) = r :=
sorry

end collinear_iff_real_simple_ratio_l49_49698


namespace dogs_on_mon_wed_fri_l49_49052

def dogs_on_tuesday : ℕ := 12
def dogs_on_thursday : ℕ := 9
def pay_per_dog : ℕ := 5
def total_earnings : ℕ := 210

theorem dogs_on_mon_wed_fri :
  ∃ (d : ℕ), d = 21 ∧ d * pay_per_dog = total_earnings - (dogs_on_tuesday + dogs_on_thursday) * pay_per_dog :=
by 
  sorry

end dogs_on_mon_wed_fri_l49_49052


namespace johns_old_cards_l49_49081

def cards_per_page : ℕ := 3
def new_cards : ℕ := 8
def total_pages : ℕ := 8

def total_cards := total_pages * cards_per_page
def old_cards := total_cards - new_cards

theorem johns_old_cards :
  old_cards = 16 :=
by
  -- Note: No specific solution steps needed here, just stating the theorem
  sorry

end johns_old_cards_l49_49081


namespace common_factor_l49_49366

-- Define the polynomials
def P1 (x : ℝ) : ℝ := x^3 + x^2
def P2 (x : ℝ) : ℝ := x^2 + 2*x + 1
def P3 (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem common_factor (x : ℝ) : ∃ (f : ℝ → ℝ), (f x = x + 1) ∧ (∃ g1 g2 g3 : ℝ → ℝ, P1 x = f x * g1 x ∧ P2 x = f x * g2 x ∧ P3 x = f x * g3 x) :=
sorry

end common_factor_l49_49366


namespace mathematically_excellent_related_to_gender_probability_of_selecting_at_least_one_140_150_l49_49555

-- Part (1) Definitions and Statement

def total_students : ℕ := 50

def students_mathematically_excellent (female male : ℕ) :=
  female + male ≥ 34

def total_female_students : ℕ :=
  1 + 4 + 5 + 5 + 3 + 2

def total_male_students : ℕ :=
  2 + 4 + 12 + 9 + 3

def calc_k2 (a b c d : ℕ) :=
  let n := a + b + c + d in
  (n * (a*d - b*c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem mathematically_excellent_related_to_gender :
  calc_k2 10 10 24 16 > 3.841 :=
sorry

-- Part (2) Definitions and Statement

def choose (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

def prob_at_least_one_in_range_140_150 (three_selected : ℕ) : ℚ :=
  1 - (choose 9 three_selected) / (choose 12 three_selected)

theorem probability_of_selecting_at_least_one_140_150 :
  prob_at_least_one_in_range_140_150 3 = 34 / 55 :=
sorry

end mathematically_excellent_related_to_gender_probability_of_selecting_at_least_one_140_150_l49_49555


namespace factory_workers_total_payroll_l49_49958

theorem factory_workers_total_payroll (total_office_payroll : ℝ) (number_factory_workers : ℝ) 
(number_office_workers : ℝ) (salary_difference : ℝ) 
(average_office_salary : ℝ) (average_factory_salary : ℝ) 
(h1 : total_office_payroll = 75000) (h2 : number_factory_workers = 15)
(h3 : number_office_workers = 30) (h4 : salary_difference = 500)
(h5 : average_office_salary = total_office_payroll / number_office_workers)
(h6 : average_office_salary = average_factory_salary + salary_difference) :
  number_factory_workers * average_factory_salary = 30000 :=
by
  sorry

end factory_workers_total_payroll_l49_49958


namespace max_sides_three_obtuse_l49_49426

theorem max_sides_three_obtuse (n : ℕ) (convex : Prop) (obtuse_angles : ℕ) :
  (convex = true ∧ obtuse_angles = 3) → n ≤ 6 :=
by
  sorry

end max_sides_three_obtuse_l49_49426


namespace value_of_p_l49_49821

theorem value_of_p (x y p : ℝ) 
  (h1 : 3 * x - 2 * y = 4 - p) 
  (h2 : 4 * x - 3 * y = 2 + p) 
  (h3 : x > y) : 
  p < -1 := 
sorry

end value_of_p_l49_49821


namespace count_real_numbers_a_with_integer_roots_l49_49035

theorem count_real_numbers_a_with_integer_roots :
  ∃ (S : Finset ℝ), (∀ (a : ℝ), (∃ (x y : ℤ), x^2 + a*x + 9*a = 0 ∧ y^2 + a*y + 9*a = 0) ↔ a ∈ S) ∧ S.card = 8 :=
by
  sorry

end count_real_numbers_a_with_integer_roots_l49_49035


namespace XiaoZhang_four_vcd_probability_l49_49697

noncomputable def probability_four_vcd (zhang_vcd zhang_dvd wang_vcd wang_dvd : ℕ) : ℚ :=
  (4 * 2 / (7 * 3)) + (3 * 1 / (7 * 3))

theorem XiaoZhang_four_vcd_probability :
  probability_four_vcd 4 3 2 1 = 11 / 21 :=
by
  sorry

end XiaoZhang_four_vcd_probability_l49_49697


namespace workbook_arrangement_l49_49976

-- Define the condition of having different Korean and English workbooks
variables (K1 K2 : Type) (E1 E2 : Type)

-- The main theorem statement
theorem workbook_arrangement :
  ∃ (koreanWorkbooks englishWorkbooks : List (Type)), 
  (koreanWorkbooks.length = 2) ∧
  (englishWorkbooks.length = 2) ∧
  (∀ wb ∈ (koreanWorkbooks ++ englishWorkbooks), wb ≠ wb) ∧
  (∃ arrangements : Nat,
    arrangements = 12) :=
  sorry

end workbook_arrangement_l49_49976


namespace inequality_with_a_eq_0_l49_49762

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
1 - Real.log x + a^2 * x^2 - a * x

theorem inequality_with_a_eq_0 (x : ℝ) (hx : 0 < x ∧ x < 1) : 
(1 - Real.log x) / Real.exp x + x^2 - 1 / x < 1 :=
by
  sorry

end inequality_with_a_eq_0_l49_49762


namespace find_m_l49_49612

theorem find_m (m : ℝ) :
  (∃ x a : ℝ, |x - 1| - |x + m| ≥ a ∧ a ≤ 5) ↔ (m = 4 ∨ m = -6) :=
by
  sorry

end find_m_l49_49612


namespace imo1983_q6_l49_49347

theorem imo1983_q6 (a b c : ℝ) (h : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 :=
by
  sorry

end imo1983_q6_l49_49347


namespace negation_of_proposition_l49_49815

theorem negation_of_proposition :
  (¬ ∀ (x : ℝ), |x| < 0) ↔ (∃ (x : ℝ), |x| ≥ 0) := 
sorry

end negation_of_proposition_l49_49815


namespace blocks_differ_in_two_ways_l49_49264

/-- 
A child has a set of 120 distinct blocks. Each block is one of 3 materials (plastic, wood, metal), 
3 sizes (small, medium, large), 4 colors (blue, green, red, yellow), and 5 shapes (circle, hexagon, 
square, triangle, pentagon). How many blocks in the set differ from the 'metal medium blue hexagon' 
in exactly 2 ways?
-/
def num_blocks_differ_in_two_ways : Nat := 44

theorem blocks_differ_in_two_ways (blocks : Fin 120)
    (materials : Fin 3)
    (sizes : Fin 3)
    (colors : Fin 4)
    (shapes : Fin 5)
    (fixed_block : {m // m = 2} × {s // s = 1} × {c // c = 0} × {sh // sh = 1}) :
    num_blocks_differ_in_two_ways = 44 :=
by
  -- proof steps are omitted
  sorry

end blocks_differ_in_two_ways_l49_49264


namespace first_discount_percentage_l49_49527

theorem first_discount_percentage (x : ℝ) (h : 450 * (1 - x / 100) * 0.85 = 306) : x = 20 :=
sorry

end first_discount_percentage_l49_49527


namespace apples_hand_out_l49_49365

theorem apples_hand_out (t p a h : ℕ) (h_t : t = 62) (h_p : p = 6) (h_a : a = 9) : h = t - (p * a) → h = 8 :=
by
  intros
  sorry

end apples_hand_out_l49_49365


namespace payment_to_C_l49_49707

theorem payment_to_C (A_days B_days total_payment days_taken : ℕ) 
  (A_work_rate B_work_rate : ℚ)
  (work_fraction_by_A_and_B : ℚ)
  (remaining_work_fraction_by_C : ℚ)
  (C_payment : ℚ) :
  A_days = 6 →
  B_days = 8 →
  total_payment = 3360 →
  days_taken = 3 →
  A_work_rate = 1/6 →
  B_work_rate = 1/8 →
  work_fraction_by_A_and_B = (A_work_rate + B_work_rate) * days_taken →
  remaining_work_fraction_by_C = 1 - work_fraction_by_A_and_B →
  C_payment = total_payment * remaining_work_fraction_by_C →
  C_payment = 420 := 
by
  intros hA hB hTP hD hAR hBR hWF hRWF hCP
  sorry

end payment_to_C_l49_49707


namespace color_of_85th_bead_l49_49400

/-- Definition for the repeating pattern of beads -/
def pattern : List String := ["red", "orange", "yellow", "yellow", "yellow", "green", "blue", "blue"]

/-- Definition for finding the color of the n-th bead -/
def bead_color (n : Nat) : Option String :=
  let index := (n - 1) % pattern.length
  pattern.get? index

theorem color_of_85th_bead : bead_color 85 = some "yellow" := by
  sorry

end color_of_85th_bead_l49_49400


namespace expand_product_l49_49890

theorem expand_product (x : ℝ) : 4 * (x + 3) * (2 * x + 7) = 8 * x ^ 2 + 52 * x + 84 := by
  sorry

end expand_product_l49_49890


namespace probability_same_value_after_reroll_l49_49230

theorem probability_same_value_after_reroll
  (initial_dice : Fin 6 → Fin 6)
  (rerolled_dice : Fin 4 → Fin 6)
  (initial_pair_num : Fin 6)
  (h_initial_no_four_of_a_kind : ∀ (n : Fin 6), (∃ i j : Fin 6, i ≠ j ∧ initial_dice i = n ∧ initial_dice j = n) →
    ∃ (i₁ i₂ i₃ i₄ : Fin 6), i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₁ ≠ i₄ ∧ i₂ ≠ i₃ ∧ i₂ ≠ i₄ ∧ i₃ ≠ i₄ ∧
    initial_dice i₁ ≠ n ∧ initial_dice i₂ ≠ n ∧ initial_dice i₃ ≠ n ∧ initial_dice i₄ ≠ n)
  (h_initial_pair : ∃ i j : Fin 6, i ≠ j ∧ initial_dice i = initial_pair_num ∧ initial_dice j = initial_pair_num) :
  (671 : ℚ) / 1296 = 671 / 1296 :=
by sorry

end probability_same_value_after_reroll_l49_49230


namespace probability_three_black_balls_probability_white_ball_l49_49007

-- Definitions representing conditions
def total_ratio (A B C : ℕ) := A / B = 5 / 4 ∧ B / C = 4 / 6

-- Proportions of black balls in each box
def proportion_black_A (black_A total_A : ℕ) := black_A = 40 * total_A / 100
def proportion_black_B (black_B total_B : ℕ) := black_B = 25 * total_B / 100
def proportion_black_C (black_C total_C : ℕ) := black_C = 50 * total_C / 100

-- Problem 1: Probability of selecting a black ball from each box
theorem probability_three_black_balls
  (A B C : ℕ)
  (total_A total_B total_C : ℕ)
  (black_A black_B black_C : ℕ)
  (h1 : total_ratio A B C)
  (h2 : proportion_black_A black_A total_A)
  (h3 : proportion_black_B black_B total_B)
  (h4 : proportion_black_C black_C total_C) :
  (black_A / total_A) * (black_B / total_B) * (black_C / total_C) = 1 / 20 :=
  sorry

-- Problem 2: Probability of selecting a white ball from the mixed total
theorem probability_white_ball
  (A B C : ℕ)
  (total_A total_B total_C : ℕ)
  (black_A black_B black_C : ℕ)
  (white_A white_B white_C : ℕ)
  (h1 : total_ratio A B C)
  (h2 : proportion_black_A black_A total_A)
  (h3 : proportion_black_B black_B total_B)
  (h4 : proportion_black_C black_C total_C)
  (h5 : white_A = total_A - black_A)
  (h6 : white_B = total_B - black_B)
  (h7 : white_C = total_C - black_C) :
  (white_A + white_B + white_C) / (total_A + total_B + total_C) = 3 / 5 :=
  sorry

end probability_three_black_balls_probability_white_ball_l49_49007


namespace sum_of_coefficients_l49_49598

theorem sum_of_coefficients (a b : ℝ)
  (h1 : 15 * a^4 * b^2 = 135)
  (h2 : 6 * a^5 * b = -18) :
  (a + b)^6 = 64 := by
  sorry

end sum_of_coefficients_l49_49598


namespace illuminated_area_correct_l49_49267

noncomputable def cube_illuminated_area (a ρ : ℝ) (h₁ : a = 1 / Real.sqrt 2) (h₂ : ρ = Real.sqrt (2 - Real.sqrt 3)) : ℝ :=
  (Real.sqrt 3 - 3 / 2) * (Real.pi + 3)

theorem illuminated_area_correct :
  cube_illuminated_area (1 / Real.sqrt 2) (Real.sqrt (2 - Real.sqrt 3)) (by norm_num) (by norm_num) = (Real.sqrt 3 - 3 / 2) * (Real.pi + 3) :=
sorry

end illuminated_area_correct_l49_49267


namespace inequality_proof_l49_49917

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)

theorem inequality_proof :
  (a / (a + b)) * ((a + 2 * b) / (a + 3 * b)) < Real.sqrt (a / (a + 4 * b)) :=
sorry

end inequality_proof_l49_49917


namespace isosceles_triangle_inequality_degenerate_triangle_a_zero_degenerate_triangle_double_b_l49_49823

section isosceles_triangle

variables (a b k : ℝ)

/-- Prove the inequality for an isosceles triangle -/
theorem isosceles_triangle_inequality (h_perimeter : k = a + 2 * b) (ha_pos : a > 0) :
  k / 2 < a + b ∧ a + b < 3 * k / 4 :=
sorry

/-- Prove the inequality for degenerate triangle with a = 0 -/
theorem degenerate_triangle_a_zero (b k : ℝ) (h_perimeter : k = 2 * b) :
  k / 2 ≤ b ∧ b < 3 * k / 4 :=
sorry

/-- Prove the inequality for degenerate triangle with a = 2b -/
theorem degenerate_triangle_double_b (b k : ℝ) (h_perimeter : k = 4 * b) :
  k / 2 < b ∧ b ≤ 3 * k / 4 :=
sorry

end isosceles_triangle

end isosceles_triangle_inequality_degenerate_triangle_a_zero_degenerate_triangle_double_b_l49_49823


namespace seating_arrangement_count_l49_49247

theorem seating_arrangement_count :
  let front_row := 11
  let back_row := 12
  let blocked_seats_front := 3
  let people := 2
  let valid_arrangements := 346
  (∀ positions : list ℕ,
    (length positions = front_row + back_row - blocked_seats_front) → 
    (∀ two_people : list (ℕ × ℕ), 
      (length two_people = people) ∧ 
      (∀ p1 p2, p1 ≠ p2 → abs (p1.1 - p2.1) > 1 ∧ abs (p1.2 - p2.2) > 1)) →
      count_valid_arrangements positions two_people = valid_arrangements) := 
sorry

end seating_arrangement_count_l49_49247


namespace angle_measure_l49_49529

theorem angle_measure (x y : ℝ) 
  (h1 : y = 3 * x + 10) 
  (h2 : x + y = 180) : x = 42.5 :=
by
  -- Proof goes here
  sorry

end angle_measure_l49_49529


namespace weight_of_one_fan_l49_49687

theorem weight_of_one_fan
  (total_weight_with_fans : ℝ)
  (num_fans : ℕ)
  (empty_box_weight : ℝ)
  (h1 : total_weight_with_fans = 11.14)
  (h2 : num_fans = 14)
  (h3 : empty_box_weight = 0.5) :
  (total_weight_with_fans - empty_box_weight) / num_fans = 0.76 :=
by
  simp [h1, h2, h3]
  sorry

end weight_of_one_fan_l49_49687


namespace correct_fraction_l49_49799

theorem correct_fraction (x y : ℤ) (h : (5 / 6 : ℚ) * 384 = (x / y : ℚ) * 384 + 200) : x / y = 5 / 16 :=
by
  sorry

end correct_fraction_l49_49799


namespace negation_proof_l49_49245

open Real

theorem negation_proof :
  (¬ ∃ x : ℕ, exp x - x - 1 ≤ 0) ↔ (∀ x : ℕ, exp x - x - 1 > 0) :=
by
  sorry

end negation_proof_l49_49245


namespace tangent_line_through_P_is_correct_l49_49317

-- Define the circle and the point
def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 3) ^ 2 = 25
def pointP : ℝ × ℝ := (-1, 7)

-- Define the equation of the tangent line
def tangent_line (x y : ℝ) : Prop := 3 * x - 4 * y + 31 = 0

-- State the theorem
theorem tangent_line_through_P_is_correct :
  (circle_eq (-1) 7) → 
  (tangent_line (-1) 7) :=
sorry

end tangent_line_through_P_is_correct_l49_49317


namespace binary_div_four_remainder_l49_49889

theorem binary_div_four_remainder (n : ℕ) (h : n = 0b111001001101) : n % 4 = 1 := 
sorry

end binary_div_four_remainder_l49_49889


namespace num_ways_to_select_sets_l49_49882

open Finset Function

-- Define the set T
def T : Finset ℕ := {u, v, w, x, y, z}

-- Define the constraints on subsets A and B
def valid_sets (A B : Finset ℕ) : Prop :=
  A ∪ B = T ∧ A ∩ B.card = 3

-- Main statement to be proved
theorem num_ways_to_select_sets : 
  (Finset.univ.filter (λ A => ∃ (B : Finset ℕ), valid_sets A B)).card = 80 :=
by sorry

end num_ways_to_select_sets_l49_49882


namespace taller_tree_height_l49_49125

theorem taller_tree_height :
  ∀ (h : ℕ), 
    ∃ (h_s : ℕ), (h_s = h - 24) ∧ (5 * h = 7 * h_s) → h = 84 :=
by
  sorry

end taller_tree_height_l49_49125


namespace ξ_and_η_are_normal_ξ_and_η_are_normal_dropping_iid_l49_49633

noncomputable def problem_statement (ξ η : ℝ) [IsFiniteVar ξ] [IsFiniteVar η] : Prop :=
  Independent ξ η ∧
  IdenticallyDistributed ξ η ∧
  Independent (ξ + η) (ξ - η)

theorem ξ_and_η_are_normal (ξ η : ℝ) [IsFiniteVar ξ] [IsFiniteVar η] 
  (h_ind0 : Independent ξ η) 
  (h_iid : IdenticallyDistributed ξ η) 
  (h_ind1 : Independent (ξ + η) (ξ - η)) : 
  IsNormal ξ ∧ IsNormal η := 
sorry

theorem ξ_and_η_are_normal_dropping_iid (ξ η : ℝ) [IsFiniteVar ξ] [IsFiniteVar η] 
  (h_ind0 : Independent ξ η) 
  (h_ind1 : Independent (ξ + η) (ξ - η)) : 
  IsNormal ξ ∧ IsNormal η := 
sorry

end ξ_and_η_are_normal_ξ_and_η_are_normal_dropping_iid_l49_49633


namespace jimmys_speed_l49_49973

theorem jimmys_speed 
(Mary_speed : ℕ) (total_distance : ℕ) (t : ℕ)
(h1 : Mary_speed = 5)
(h2 : total_distance = 9)
(h3 : t = 1)
: ∃ (Jimmy_speed : ℕ), Jimmy_speed = 4 :=
by
  -- calculation steps skipped here
  sorry

end jimmys_speed_l49_49973


namespace parabola_and_hyperbola_tangent_l49_49164

theorem parabola_and_hyperbola_tangent (m : ℝ) :
  (∀ (x y : ℝ), (y = x^2 + 6) → (y^2 - m * x^2 = 6) → (m = 12 + 10 * Real.sqrt 6 ∨ m = 12 - 10 * Real.sqrt 6)) :=
sorry

end parabola_and_hyperbola_tangent_l49_49164


namespace john_max_books_l49_49080

theorem john_max_books (h₁ : 4575 ≥ 0) (h₂ : 325 > 0) : 
  ∃ (x : ℕ), x = 14 ∧ ∀ n : ℕ, n ≤ x ↔ n * 325 ≤ 4575 := 
  sorry

end john_max_books_l49_49080


namespace distance_between_points_l49_49022

open Real

theorem distance_between_points : 
  let p1 := (2, 2)
  let p2 := (5, 9)
  dist (p1 : ℝ × ℝ) p2 = sqrt 58 :=
by
  let p1 := (2, 2)
  let p2 := (5, 9)
  have h1 : p1.1 = 2 := rfl
  have h2 : p1.2 = 2 := rfl
  have h3 : p2.1 = 5 := rfl
  have h4 : p2.2 = 9 := rfl
  sorry

end distance_between_points_l49_49022


namespace age_of_beckett_l49_49735

variables (B O S J : ℕ)

theorem age_of_beckett
  (h1 : B = O - 3)
  (h2 : S = O - 2)
  (h3 : J = 2 * S + 5)
  (h4 : B + O + S + J = 71) :
  B = 12 :=
by
  sorry

end age_of_beckett_l49_49735


namespace orange_segments_l49_49237

noncomputable def total_segments (H S B : ℕ) : ℕ :=
  H + S + B

theorem orange_segments
  (H S B : ℕ)
  (h1 : H = 2 * S)
  (h2 : S = B / 5)
  (h3 : B = S + 8) :
  total_segments H S B = 16 := by
  -- proof goes here
  sorry

end orange_segments_l49_49237


namespace cube_face_parallel_probability_l49_49304

theorem cube_face_parallel_probability :
  ∃ (n m : ℕ), (n = 15) ∧ (m = 3) ∧ (m / n = (1 / 5 : ℝ)) := 
sorry

end cube_face_parallel_probability_l49_49304


namespace temperature_range_for_5_percent_deviation_l49_49375

noncomputable def approx_formula (C : ℝ) : ℝ := 2 * C + 30
noncomputable def exact_formula (C : ℝ) : ℝ := (9/5 : ℝ) * C + 32
noncomputable def deviation (C : ℝ) : ℝ := approx_formula C - exact_formula C
noncomputable def percentage_deviation (C : ℝ) : ℝ := abs (deviation C / exact_formula C)

theorem temperature_range_for_5_percent_deviation :
  ∀ (C : ℝ), 1 + 11 / 29 ≤ C ∧ C ≤ 32 + 8 / 11 ↔ percentage_deviation C ≤ 0.05 := sorry

end temperature_range_for_5_percent_deviation_l49_49375


namespace axes_are_not_vectors_l49_49959

def is_vector (v : Type) : Prop :=
  ∃ (magnitude : ℝ) (direction : ℝ), magnitude > 0

def x_axis : Type := ℝ
def y_axis : Type := ℝ

-- The Cartesian x-axis and y-axis are not vectors
theorem axes_are_not_vectors : ¬ (is_vector x_axis) ∧ ¬ (is_vector y_axis) :=
by
  sorry

end axes_are_not_vectors_l49_49959


namespace find_a_n_l49_49922

-- Definitions from the conditions
def seq (a : ℕ → ℤ) : Prop :=
  ∀ n, (3 - a (n + 1)) * (6 + a n) = 18

-- The Lean statement of the problem
theorem find_a_n (a : ℕ → ℤ) (h_a0 : a 0 ≠ 3) (h_seq : seq a) :
  ∀ n, a n = 2 ^ (n + 2) - n - 3 :=
by
  sorry

end find_a_n_l49_49922


namespace expression_value_l49_49176

theorem expression_value (a b c : ℝ) (h : a * b + b * c + c * a = 3) : 
  (a * (b^2 + 3)) / (a + b) + (b * (c^2 + 3)) / (b + c) + (c * (a^2 + 3)) / (c + a) = 6 := 
by
  sorry

end expression_value_l49_49176


namespace g_five_l49_49968

def g (x : ℝ) : ℝ := 4 * x + 2

theorem g_five : g 5 = 22 := by
  sorry

end g_five_l49_49968


namespace cards_dealt_to_people_l49_49783

theorem cards_dealt_to_people (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  (∃ k, k = total_people - (total_cards % total_people) ∧ k = 3) := 
by
  sorry

end cards_dealt_to_people_l49_49783


namespace remainder_is_correct_l49_49885

noncomputable def remainder (P Q : Polynomial ℤ) : Polynomial ℤ :=
  (Polynomial.modByMonic P (Q.monic))

def P : Polynomial ℤ := (X^5 - 1) * (X^3 - 1)
def Q : Polynomial ℤ := X^3 + X^2 + 1

theorem remainder_is_correct : remainder P Q = -2 * X^2 + X + 1 := by
  sorry

end remainder_is_correct_l49_49885


namespace sum_of_invalid_domain_of_g_l49_49289

noncomputable def g (x : ℝ) : ℝ := 1 / (2 + (1 / (3 + (1 / x))))

theorem sum_of_invalid_domain_of_g : 
  (0 : ℝ) + (-1 / 3) + (-2 / 7) = -13 / 21 :=
by
  sorry

end sum_of_invalid_domain_of_g_l49_49289


namespace num_real_values_for_integer_roots_l49_49034

theorem num_real_values_for_integer_roots : 
  (∃ (a : ℝ), ∀ (r s : ℤ), r + s = -a ∧ r * s = 9 * a) → ∃ (n : ℕ), n = 10 :=
by
  sorry

end num_real_values_for_integer_roots_l49_49034


namespace necessary_and_sufficient_condition_l49_49535

variable {A B : Prop}

theorem necessary_and_sufficient_condition (h1 : A → B) (h2 : B → A) : A ↔ B := 
by 
  sorry

end necessary_and_sufficient_condition_l49_49535


namespace find_angle_A_l49_49331

theorem find_angle_A (a b c : ℝ) (h : a^2 - c^2 = b^2 - b * c) : 
  ∃ (A : ℝ), A = π / 3 :=
by
  sorry

end find_angle_A_l49_49331


namespace probability_score_l49_49949

/-- Given:
1. A bag with 4 red balls and 3 black balls.
2. 4 balls drawn from the bag.
3. Drawing 1 red ball scores 1 point.
4. Drawing 1 black ball scores 3 points.
5. Score is a random variable ξ.

Prove that the probability P(ξ ≤ 7) equals 13/35.
-/
theorem probability_score (R B : ℕ) (drawn : ℕ) (score_red score_black : ℕ) (ξ : ℕ → ℕ) :
  R = 4 → B = 3 → drawn = 4 → score_red = 1 → score_black = 3 →
  (∀ n, ξ n = if n = 0 then 4 else if n = 1 then 6 else if n = 2 then 8 else if n = 3 then 10 else 0) →
  ∑ i in finset.range (ξ 2 + 1), if ξ i ≤ 7 then 1 else 0 / (nat.choose (R + B) drawn) = 13 / 35 :=
by
  intros hR hB hDrawn hscore_red hscore_black hξ sorry

end probability_score_l49_49949


namespace hash_fn_triple_40_l49_49014

def hash_fn (N : ℝ) : ℝ := 0.6 * N + 2

theorem hash_fn_triple_40 : hash_fn (hash_fn (hash_fn 40)) = 12.56 := by
  sorry

end hash_fn_triple_40_l49_49014


namespace baker_work_alone_time_l49_49388

theorem baker_work_alone_time 
  (rate_baker_alone : ℕ) 
  (rate_baker_with_helper : ℕ) 
  (total_time : ℕ) 
  (total_flour : ℕ)
  (time_with_helper : ℕ)
  (flour_used_baker_alone_time : ℕ)
  (flour_used_with_helper_time : ℕ)
  (total_flour_used : ℕ) 
  (h1 : rate_baker_alone = total_flour / 6) 
  (h2 : rate_baker_with_helper = total_flour / 2) 
  (h3 : total_time = 150)
  (h4 : flour_used_baker_alone_time = total_flour * flour_used_baker_alone_time / 6)
  (h5 : flour_used_with_helper_time = total_flour * (total_time - flour_used_baker_alone_time) / 2)
  (h6 : total_flour_used = total_flour) :
  flour_used_baker_alone_time = 45 :=
by
  sorry

end baker_work_alone_time_l49_49388


namespace apple_cost_price_l49_49701

theorem apple_cost_price (SP : ℝ) (loss_frac : ℝ) (CP : ℝ) (h_SP : SP = 19) (h_loss_frac : loss_frac = 1 / 6) (h_loss : SP = CP - loss_frac * CP) : CP = 22.8 :=
by
  sorry

end apple_cost_price_l49_49701


namespace people_with_fewer_than_7_cards_l49_49770

-- Definitions based on conditions
def cards_total : ℕ := 60
def people_total : ℕ := 9

-- Statement of the theorem
theorem people_with_fewer_than_7_cards : 
  ∃ (x : ℕ), x = 3 ∧ (cards_total % people_total = 0 ∨ cards_total % people_total < people_total) :=
by
  sorry

end people_with_fewer_than_7_cards_l49_49770


namespace three_distinct_numbers_l49_49058

theorem three_distinct_numbers (s : ℕ) (A : Finset ℕ) (S : Finset ℕ) (hA : A = Finset.range (4 * s + 1) \ Finset.range 1)
  (hS : S ⊆ A) (hcard: S.card = 2 * s + 2) :
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x + y = 2 * z :=
by
  sorry

end three_distinct_numbers_l49_49058


namespace quarters_value_percentage_l49_49550

theorem quarters_value_percentage (dimes_count quarters_count dimes_value quarters_value : ℕ) (h1 : dimes_count = 75)
    (h2 : quarters_count = 30) (h3 : dimes_value = 10) (h4 : quarters_value = 25) :
    (quarters_count * quarters_value * 100) / (dimes_count * dimes_value + quarters_count * quarters_value) = 50 := 
by
    sorry

end quarters_value_percentage_l49_49550


namespace amount_each_student_should_pay_l49_49526

noncomputable def total_rental_fee_per_book_per_half_hour : ℕ := 4000 
noncomputable def total_books : ℕ := 4
noncomputable def total_students : ℕ := 6
noncomputable def total_hours : ℕ := 3
noncomputable def total_half_hours : ℕ := total_hours * 2

noncomputable def total_fee_one_book : ℕ := total_rental_fee_per_book_per_half_hour * total_half_hours
noncomputable def total_fee_all_books : ℕ := total_fee_one_book * total_books

theorem amount_each_student_should_pay : total_fee_all_books / total_students = 16000 := by
  sorry

end amount_each_student_should_pay_l49_49526


namespace triangle_area_is_54_l49_49719

-- Define the sides of the triangle
def side1 : ℕ := 9
def side2 : ℕ := 12
def side3 : ℕ := 15

-- Verify that it is a right triangle using the Pythagorean theorem
def isRightTriangle (a b c : ℕ) : Prop := a * a + b * b = c * c

-- Define the area calculation for a right triangle
def areaRightTriangle (a b : ℕ) : ℕ := Nat.div (a * b) 2

-- State the theorem (Problem) to prove
theorem triangle_area_is_54 :
  isRightTriangle side1 side2 side3 ∧ areaRightTriangle side1 side2 = 54 :=
by
  sorry

end triangle_area_is_54_l49_49719


namespace total_water_in_boxes_l49_49553

theorem total_water_in_boxes : 
  let boxes := 10 
  let bottles_per_box := 50 
  let capacity_per_bottle := 12 
  let filled_fraction := 3 / 4 in 
  let water_per_bottle := filled_fraction * capacity_per_bottle 
  let water_per_box := bottles_per_box * water_per_bottle 
  let total_water := boxes * water_per_box in 
  total_water = 4500 :=
by 
  sorry

end total_water_in_boxes_l49_49553


namespace coin_problem_l49_49135

theorem coin_problem :
  ∃ (p n d q : ℕ), p + n + d + q = 11 ∧ 
                   1 * p + 5 * n + 10 * d + 25 * q = 132 ∧
                   p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1 ∧ 
                   q = 3 :=
by
  sorry

end coin_problem_l49_49135


namespace restore_triangle_ABC_l49_49337

-- let I be the incenter of triangle ABC
variable (I : Point)
-- let Ic be the C-excenter of triangle ABC
variable (I_c : Point)
-- let H be the foot of the altitude from vertex C to side AB
variable (H : Point)

-- Claim: Given I, I_c, H, we can recover the original triangle ABC
theorem restore_triangle_ABC (I I_c H : Point) : ExistsTriangleABC :=
sorry

end restore_triangle_ABC_l49_49337


namespace other_root_of_quadratic_l49_49192

theorem other_root_of_quadratic (a b c : ℚ) (x₁ x₂ : ℚ) :
  a ≠ 0 →
  x₁ = 4 / 9 →
  (a * x₁^2 + b * x₁ + c = 0) →
  (a = 81) →
  (b = -145) →
  (c = 64) →
  x₂ = -16 / 9
:=
sorry

end other_root_of_quadratic_l49_49192


namespace interest_rate_same_l49_49065

theorem interest_rate_same (initial_amount: ℝ) (interest_earned: ℝ) 
  (time_period1: ℝ) (time_period2: ℝ) (principal: ℝ) (initial_rate: ℝ) : 
  initial_amount * initial_rate * time_period2 = interest_earned * 100 ↔ initial_rate = 12 
  :=
by
  sorry

end interest_rate_same_l49_49065


namespace cost_of_adult_ticket_is_10_l49_49751

-- Definitions based on the problem's conditions
def num_adults : ℕ := 5
def num_children : ℕ := 2
def cost_concessions : ℝ := 12
def total_cost : ℝ := 76
def cost_child_ticket : ℝ := 7

-- Statement to prove the cost of an adult ticket being $10
theorem cost_of_adult_ticket_is_10 :
  ∃ A : ℝ, (num_adults * A + num_children * cost_child_ticket + cost_concessions = total_cost) ∧ A = 10 :=
by
  sorry

end cost_of_adult_ticket_is_10_l49_49751


namespace radius_of_shorter_cylinder_l49_49532

theorem radius_of_shorter_cylinder (h r : ℝ) (V_s V_t : ℝ) (π : ℝ) : 
  V_s = 500 → 
  V_t = 500 → 
  V_t = π * 5^2 * 4 * h → 
  V_s = π * r^2 * h → 
  r = 10 :=
by 
  sorry

end radius_of_shorter_cylinder_l49_49532


namespace abs_x_plus_1_plus_abs_x_minus_3_ge_a_l49_49613

theorem abs_x_plus_1_plus_abs_x_minus_3_ge_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ a) ↔ a ≤ 4 :=
by
  sorry

end abs_x_plus_1_plus_abs_x_minus_3_ge_a_l49_49613


namespace hyperbola_equation_l49_49602

theorem hyperbola_equation:
  let F1 := (-Real.sqrt 10, 0)
  let F2 := (Real.sqrt 10, 0)
  ∃ P : ℝ × ℝ, 
    (let PF1 := (P.1 - F1.1, P.2 - F1.2);
     let PF2 := (P.1 - F2.1, P.2 - F2.2);
     (PF1.1 * PF2.1 + PF1.2 * PF2.2 = 0) ∧ 
     ((Real.sqrt (PF1.1^2 + PF1.2^2)) * (Real.sqrt (PF2.1^2 + PF2.2^2)) = 2)) →
    (∃ a b : ℝ, (a^2 = 9 ∧ b^2 = 1) ∧ 
                (∀ x y : ℝ, 
                 (a ≠ 0 ∧ (x^2 / a^2) - (y^2 / b^2) = 1 ↔ 
                  ∃ P : ℝ × ℝ, 
                    let PF1 := (P.1 - F1.1, P.2 - F1.2);
                    let PF2 := (P.1 - F2.1, P.2 - F2.2);
                    PF1.1 * PF2.1 + PF1.2 * PF2.2 = 0 ∧ 
                    (Real.sqrt (PF1.1^2 + PF1.2^2)) * (Real.sqrt (PF2.1^2 + PF2.2^2)) = 2)))
:= by
sorry

end hyperbola_equation_l49_49602


namespace cylinder_surface_area_and_volume_l49_49396

noncomputable def cylinder_total_surface_area (r h : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

noncomputable def cylinder_volume (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem cylinder_surface_area_and_volume (r h : ℝ) (hr : r = 5) (hh : h = 15) :
  cylinder_total_surface_area r h = 200 * Real.pi ∧ cylinder_volume r h = 375 * Real.pi :=
by
  sorry -- Proof omitted

end cylinder_surface_area_and_volume_l49_49396


namespace equal_partition_of_weights_l49_49069

theorem equal_partition_of_weights 
  (weights : Fin 2009 → ℕ) 
  (h1 : ∀ i : Fin 2008, (weights i + 1 = weights (i + 1)) ∨ (weights i = weights (i + 1) + 1))
  (h2 : ∀ i : Fin 2009, weights i ≤ 1000)
  (h3 : (Finset.univ.sum weights) % 2 = 0) :
  ∃ (A B : Finset (Fin 2009)), (A ∪ B = Finset.univ ∧ A ∩ B = ∅ ∧ A.sum weights = B.sum weights) :=
sorry

end equal_partition_of_weights_l49_49069


namespace mode_of_gold_medals_is_8_l49_49506

def countries : List String := ["Norway", "Germany", "China", "USA", "Sweden", "Netherlands", "Austria"]

def gold_medals : List Nat := [16, 12, 9, 8, 8, 8, 7]

def mode (lst : List Nat) : Nat :=
  lst.foldr
    (fun (x : Nat) acc =>
      if lst.count x > lst.count acc then x else acc)
    lst.head!

theorem mode_of_gold_medals_is_8 :
  mode gold_medals = 8 :=
by sorry

end mode_of_gold_medals_is_8_l49_49506


namespace spherical_to_rectangular_correct_l49_49743

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_correct :
  spherical_to_rectangular 3 (Real.pi / 2) (Real.pi / 3) = (0, (3 * Real.sqrt 3) / 2, 3 / 2) :=
by
  sorry

end spherical_to_rectangular_correct_l49_49743


namespace slope_tangent_line_at_x1_l49_49202

def f (x c : ℝ) : ℝ := (x-2)*(x^2 + c)
def f_prime (x c : ℝ) := (x^2 + c) + (x-2) * 2 * x

theorem slope_tangent_line_at_x1 (c : ℝ) (h : f_prime 2 c = 0) : f_prime 1 c = -5 := by
  sorry

end slope_tangent_line_at_x1_l49_49202


namespace dividend_is_686_l49_49486

theorem dividend_is_686 (divisor quotient remainder : ℕ) (h1 : divisor = 36) (h2 : quotient = 19) (h3 : remainder = 2) :
  divisor * quotient + remainder = 686 :=
by
  sorry

end dividend_is_686_l49_49486


namespace proof_PQ_expression_l49_49590

theorem proof_PQ_expression (P Q : ℝ) (h1 : P^2 - P * Q = 1) (h2 : 4 * P * Q - 3 * Q^2 = 2) : 
  P^2 + 3 * P * Q - 3 * Q^2 = 3 :=
by
  sorry

end proof_PQ_expression_l49_49590


namespace max_hedgehogs_l49_49673

theorem max_hedgehogs (S : ℕ) (n : ℕ) (hS : S = 65) (hn : ∀ m, m > n → (m * (m + 1)) / 2 > S) :
  n = 10 := 
sorry

end max_hedgehogs_l49_49673


namespace student_correct_answers_l49_49137

theorem student_correct_answers (C I : ℕ) 
  (h1 : C + I = 100) 
  (h2 : C - 2 * I = 61) : 
  C = 87 :=
by
  sorry

end student_correct_answers_l49_49137


namespace number_of_passed_candidates_l49_49383

-- Definitions based on conditions:
def total_candidates : ℕ := 120
def avg_total_marks : ℝ := 35
def avg_passed_marks : ℝ := 39
def avg_failed_marks : ℝ := 15

-- The number of candidates who passed the examination:
theorem number_of_passed_candidates :
  ∃ (P F : ℕ), 
    P + F = total_candidates ∧
    39 * P + 15 * F = total_candidates * avg_total_marks ∧
    P = 100 :=
by
  sorry

end number_of_passed_candidates_l49_49383


namespace expression_divisible_by_x_minus_1_squared_l49_49514

theorem expression_divisible_by_x_minus_1_squared :
  ∀ (n : ℕ) (x : ℝ), x ≠ 1 →
  (n * x^(n + 1) * (1 - 1 / x) - x^n * (1 - 1 / x^n)) / (x - 1)^2 = 
  (n * x^(n + 1) - n * x^n - x^n + 1) / (x - 1)^2 :=
by
  intro n x hx_ne_1
  sorry

end expression_divisible_by_x_minus_1_squared_l49_49514


namespace store_total_income_l49_49951

def pencil_with_eraser_cost : ℝ := 0.8
def regular_pencil_cost : ℝ := 0.5
def short_pencil_cost : ℝ := 0.4

def pencils_with_eraser_sold : ℕ := 200
def regular_pencils_sold : ℕ := 40
def short_pencils_sold : ℕ := 35

noncomputable def total_money_made : ℝ :=
  (pencil_with_eraser_cost * pencils_with_eraser_sold) +
  (regular_pencil_cost * regular_pencils_sold) +
  (short_pencil_cost * short_pencils_sold)

theorem store_total_income : total_money_made = 194 := by
  sorry

end store_total_income_l49_49951


namespace refund_amount_l49_49741

def income_tax_paid : ℝ := 156000
def education_expenses : ℝ := 130000
def medical_expenses : ℝ := 10000
def tax_rate : ℝ := 0.13

def eligible_expenses : ℝ := education_expenses + medical_expenses
def max_refund : ℝ := tax_rate * eligible_expenses

theorem refund_amount : min (max_refund) (income_tax_paid) = 18200 := by
  sorry

end refund_amount_l49_49741


namespace extreme_value_result_l49_49348

open Real

-- Conditions
def function_has_extreme_value_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop := 
  deriv f x₀ = 0

-- The given function
noncomputable def f (x : ℝ) : ℝ := x * sin x

-- The problem statement (to prove)
theorem extreme_value_result (x₀ : ℝ) 
  (h : function_has_extreme_value_at f x₀) :
  (1 + x₀^2) * (1 + cos (2 * x₀)) = 2 :=
sorry

end extreme_value_result_l49_49348


namespace radius_of_spheres_in_cube_l49_49657

noncomputable def sphere_radius (sides: ℝ) (spheres: ℕ) (tangent_pairs: ℕ) (tangent_faces: ℕ): ℝ :=
  if sides = 2 ∧ spheres = 10 ∧ tangent_pairs = 2 ∧ tangent_faces = 3 then 0.5 else 0

theorem radius_of_spheres_in_cube : sphere_radius 2 10 2 3 = 0.5 :=
by
  -- This is the main theorem that states the radius of each sphere given the problem conditions.
  sorry

end radius_of_spheres_in_cube_l49_49657


namespace linda_age_l49_49112

variable (s j l : ℕ)

theorem linda_age (h1 : (s + j + l) / 3 = 11) 
                  (h2 : l - 5 = s) 
                  (h3 : j + 4 = 3 * (s + 4) / 4) :
                  l = 14 := by
  sorry

end linda_age_l49_49112


namespace distance_AC_l49_49292

theorem distance_AC (t_Eddy t_Freddy : ℕ) (d_AB : ℝ) (speed_ratio : ℝ) : 
  t_Eddy = 3 ∧ t_Freddy = 4 ∧ d_AB = 510 ∧ speed_ratio = 2.2666666666666666 → 
  ∃ d_AC : ℝ, d_AC = 300 :=
by 
  intros h
  obtain ⟨hE, hF, hD, hR⟩ := h
  -- Declare velocities
  let v_Eddy : ℝ := d_AB / t_Eddy
  let v_Freddy : ℝ := v_Eddy / speed_ratio
  let d_AC : ℝ := v_Freddy * t_Freddy
  -- Prove the distance
  use d_AC
  sorry

end distance_AC_l49_49292


namespace student_passing_percentage_l49_49873

variable (marks_obtained failed_by max_marks : ℕ)

def passing_marks (marks_obtained failed_by : ℕ) : ℕ :=
  marks_obtained + failed_by

def percentage_needed (passing_marks max_marks : ℕ) : ℚ :=
  (passing_marks : ℚ) / (max_marks : ℚ) * 100

theorem student_passing_percentage
  (h1 : marks_obtained = 125)
  (h2 : failed_by = 40)
  (h3 : max_marks = 500) :
  percentage_needed (passing_marks marks_obtained failed_by) max_marks = 33 := by
  sorry

end student_passing_percentage_l49_49873


namespace isosceles_triangle_base_length_l49_49669

theorem isosceles_triangle_base_length
  (a b : ℕ)
  (ha : a = 8)
  (hp : 2 * a + b = 25)
  : b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l49_49669


namespace distribute_5_cousins_in_4_rooms_l49_49358

theorem distribute_5_cousins_in_4_rooms : 
  let rooms := 4
  let cousins := 5
  ∃ ways : ℕ, ways = 67 ∧ rooms = 4 ∧ cousins = 5 := sorry

end distribute_5_cousins_in_4_rooms_l49_49358


namespace h_value_at_3_l49_49631

noncomputable def f (x : ℝ) : ℝ := 3 * x + 4
noncomputable def g (x : ℝ) : ℝ := (Real.sqrt (f x) - 3) ^ 2
noncomputable def h (x : ℝ) : ℝ := f (g x)

theorem h_value_at_3 : h 3 = 70 - 18 * Real.sqrt 13 := 
by
  -- Proof goes here
  sorry

end h_value_at_3_l49_49631


namespace area_of_triangle_BEF_l49_49877

open Real

theorem area_of_triangle_BEF (a b x y : ℝ) (h1 : a * b = 30) (h2 : (1/2) * abs (x * (b - y) + a * b - a * y) = 2) (h3 : (1/2) * abs (x * (-y) + a * y - x * b) = 3) :
  (1/2) * abs (x * y) = 35 / 8 :=
by
  sorry

end area_of_triangle_BEF_l49_49877


namespace units_digit_of_7_pow_6_pow_5_l49_49895

-- Define the units digit cycle for powers of 7
def units_digit_cycle : List ℕ := [7, 9, 3, 1]

-- Define the function to calculate the units digit of 7^n
def units_digit (n : ℕ) : ℕ :=
  units_digit_cycle[(n % 4) - 1]

-- The main theorem stating the units digit of 7^(6^5) is 1
theorem units_digit_of_7_pow_6_pow_5 : units_digit (6^5) = 1 :=
by
  -- Skipping the proof, including a sorry placeholder
  sorry

end units_digit_of_7_pow_6_pow_5_l49_49895


namespace people_with_fewer_than_7_cards_l49_49786

theorem people_with_fewer_than_7_cards (num_cards : ℕ) (num_people : ℕ) (h₁ : num_cards = 60) (h₂ : num_people = 9) : 
  ∃ k, k = num_people - num_cards % num_people ∧ k < 7 :=
by
  have rem := num_cards % num_people
  have few_count := num_people - rem
  use few_count
  split
  sorry

end people_with_fewer_than_7_cards_l49_49786


namespace car_owners_without_motorcycles_l49_49480

theorem car_owners_without_motorcycles 
    (total_adults : ℕ) 
    (car_owners : ℕ) 
    (motorcycle_owners : ℕ) 
    (total_with_vehicles : total_adults = 500) 
    (total_car_owners : car_owners = 480) 
    (total_motorcycle_owners : motorcycle_owners = 120) : 
    car_owners - (car_owners + motorcycle_owners - total_adults) = 380 := 
by
    sorry

end car_owners_without_motorcycles_l49_49480


namespace old_edition_pages_l49_49145

theorem old_edition_pages (x : ℕ) (h : 2 * x - 230 = 450) : x = 340 :=
by {
  have eq1 : 2 * x = 450 + 230, from eq_add_of_sub_eq h,
  have eq2 : 2 * x = 680, from eq1,
  have eq3 : x = 680 / 2, from eq_of_mul_eq_mul_right (by norm_num) eq2,
  norm_num at eq3,
  exact eq3,
}

end old_edition_pages_l49_49145


namespace students_in_both_clubs_l49_49414

variables (Total Students RoboticClub ScienceClub EitherClub BothClubs : ℕ)

theorem students_in_both_clubs
  (h1 : Total = 300)
  (h2 : RoboticClub = 80)
  (h3 : ScienceClub = 130)
  (h4 : EitherClub = 190)
  (h5 : EitherClub = RoboticClub + ScienceClub - BothClubs) :
  BothClubs = 20 :=
by
  sorry

end students_in_both_clubs_l49_49414


namespace units_digit_pow_7_6_5_l49_49897

theorem units_digit_pow_7_6_5 :
  let units_digit (n : ℕ) : ℕ := n % 10
  in units_digit (7 ^ (6 ^ 5)) = 9 :=
by
  let units_digit (n : ℕ) := n % 10
  sorry

end units_digit_pow_7_6_5_l49_49897


namespace factory_toys_production_each_day_l49_49398

theorem factory_toys_production_each_day 
  (weekly_production : ℕ)
  (days_worked_per_week : ℕ)
  (h1 : weekly_production = 4560)
  (h2 : days_worked_per_week = 4) : 
  (weekly_production / days_worked_per_week) = 1140 :=
  sorry

end factory_toys_production_each_day_l49_49398


namespace find_line_eq_l49_49180

theorem find_line_eq (l : ℝ → ℝ → Prop) :
  (∃ A B : ℝ × ℝ, l A.fst A.snd ∧ l B.fst B.snd ∧ ((A.fst + 1)^2 + (A.snd - 2)^2 = 100 ∧ (B.fst + 1)^2 + (B.snd - 2)^2 = 100)) ∧
  (∃ M : ℝ × ℝ, M = (-2, 3) ∧ (l M.fst M.snd)) →
  (∀ x y : ℝ, l x y ↔ x - y + 5 = 0) :=
by
  sorry

end find_line_eq_l49_49180


namespace problem_statement_l49_49124

noncomputable def a_b (a b : ℚ) : Prop :=
  a + b = 6 ∧ a / b = 6

theorem problem_statement (a b : ℚ) (h : a_b a b) : 
  (a * b - (a - b)) = 6 / 49 :=
by
  sorry

end problem_statement_l49_49124


namespace pipe_fill_without_hole_l49_49646

theorem pipe_fill_without_hole :
  ∀ (T : ℝ), 
  (1 / T - 1 / 60 = 1 / 20) → 
  T = 15 := 
by
  intros T h
  sorry

end pipe_fill_without_hole_l49_49646


namespace polar_to_cartesian_l49_49418

theorem polar_to_cartesian (r θ : ℝ) (h_r : r = 2) (h_θ : θ = π / 6) :
  (r * Real.cos θ, r * Real.sin θ) = (Real.sqrt 3, 1) :=
by
  rw [h_r, h_θ]
  have h_cos : Real.cos (π / 6) = Real.sqrt 3 / 2 := sorry -- This identity can be used from trigonometric property.
  have h_sin : Real.sin (π / 6) = 1 / 2 := sorry -- This identity can be used from trigonometric property.
  rw [h_cos, h_sin]
  -- some algebraic steps to simplifiy left sides to (Real.sqrt 3, 1) should follow here. using multiplication and commmutaivity properties mainly.
  sorry

end polar_to_cartesian_l49_49418


namespace sixth_element_row_20_l49_49257

theorem sixth_element_row_20 : (Nat.choose 20 5) = 15504 := by
  sorry

end sixth_element_row_20_l49_49257


namespace count_real_numbers_a_with_integer_roots_l49_49036

theorem count_real_numbers_a_with_integer_roots :
  ∃ (S : Finset ℝ), (∀ (a : ℝ), (∃ (x y : ℤ), x^2 + a*x + 9*a = 0 ∧ y^2 + a*y + 9*a = 0) ↔ a ∈ S) ∧ S.card = 8 :=
by
  sorry

end count_real_numbers_a_with_integer_roots_l49_49036


namespace total_students_class_l49_49484

theorem total_students_class (S R : ℕ) 
  (h1 : 2 + 12 + 10 + R = S)
  (h2 : (0 * 2) + (1 * 12) + (2 * 10) + (3 * R) = 2 * S) :
  S = 40 := by
  sorry

end total_students_class_l49_49484


namespace total_payment_l49_49351

def work_hours := 2
def hourly_rate := 75
def part_cost := 150

theorem total_payment : work_hours * hourly_rate + part_cost = 300 := 
by 
  calc 
  2 * 75 + 150 = 150 + 150 : by rw mul_comm 2 75
             ... = 300 : by rw add_comm 150 150

# The term "sorry" is unnecessary due to the use of "by" tactic and commutativity rules simplifying the steps directly.

end total_payment_l49_49351


namespace group_1991_l49_49261

theorem group_1991 (n : ℕ) (h1 : 1 ≤ n) (h2 : 1991 = 2 * n ^ 2 - 1) : n = 32 := 
sorry

end group_1991_l49_49261


namespace ratio_is_correct_l49_49977

-- Define the constants
def total_students : ℕ := 47
def current_students : ℕ := 6 * 3
def girls_bathroom : ℕ := 3
def new_groups : ℕ := 2 * 4
def foreign_exchange_students : ℕ := 3 * 3

-- The total number of missing students
def missing_students : ℕ := girls_bathroom + new_groups + foreign_exchange_students

-- The number of students who went to the canteen
def students_canteen : ℕ := total_students - current_students - missing_students

-- The ratio of students who went to the canteen to girls who went to the bathroom
def canteen_to_bathroom_ratio : ℕ × ℕ := (students_canteen, girls_bathroom)

theorem ratio_is_correct : canteen_to_bathroom_ratio = (3, 1) :=
by
  -- Proof goes here
  sorry

end ratio_is_correct_l49_49977


namespace sum_of_coordinates_x_l49_49084

-- Given points Y and Z
def Y : ℝ × ℝ := (2, 8)
def Z : ℝ × ℝ := (0, -4)

-- Given ratio conditions
def ratio_condition (X Y Z : ℝ × ℝ) : Prop :=
  dist X Z / dist X Y = 1/3 ∧ dist Z Y / dist X Y = 1/3

-- Define X, ensuring Z is the midpoint of XY
def X : ℝ × ℝ := (4, 20)

-- Prove that sum of coordinates of X is 10
theorem sum_of_coordinates_x (h : ratio_condition X Y Z) : (X.1 + X.2) = 10 := 
  sorry

end sum_of_coordinates_x_l49_49084


namespace operation_not_equal_33_l49_49861

-- Definitions for the given conditions
def single_digit_positive_integer (n : ℤ) : Prop := 1 ≤ n ∧ n ≤ 9
def x (a : ℤ) := 1 / 5 * a
def z (b : ℤ) := 1 / 5 * b

-- The theorem to show that the operations involving x and z cannot equal 33
theorem operation_not_equal_33 (a b : ℤ) (ha : single_digit_positive_integer a) 
(hb : single_digit_positive_integer b) : 
((x a - z b = 33) ∨ (z b - x a = 33) ∨ (x a / z b = 33) ∨ (z b / x a = 33)) → false :=
by
  sorry

end operation_not_equal_33_l49_49861


namespace min_lcm_value_l49_49837

-- Definitions
def gcd_77 (a b c d : ℕ) : Prop :=
  Nat.gcd (Nat.gcd a b) (Nat.gcd c d) = 77

def lcm_n (a b c d n : ℕ) : Prop :=
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d) = n

-- Problem statement
theorem min_lcm_value :
  (∃ a b c d : ℕ, gcd_77 a b c d ∧ lcm_n a b c d 27720) ∧
  (∀ n : ℕ, (∃ a b c d : ℕ, gcd_77 a b c d ∧ lcm_n a b c d n) → 27720 ≤ n) :=
sorry

end min_lcm_value_l49_49837


namespace gwen_average_speed_l49_49050

def average_speed (distance1 distance2 speed1 speed2 : ℕ) : ℕ :=
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  total_distance / total_time

theorem gwen_average_speed :
  average_speed 40 40 15 30 = 20 :=
by
  sorry

end gwen_average_speed_l49_49050


namespace geom_seq_sum_l49_49219

theorem geom_seq_sum (a : ℕ → ℝ) (q : ℝ) (h1 : 0 < q)
  (h2 : ∀ n, a (n+1) = a n * q)
  (h3 : a 0 + a 1 = 3 / 4)
  (h4 : a 2 + a 3 + a 4 + a 5 = 15) :
  a 6 + a 7 + a 8 = 112 := by
  sorry

end geom_seq_sum_l49_49219


namespace find_multiple_of_brothers_l49_49278

theorem find_multiple_of_brothers : 
  ∃ x : ℕ, (x * 4) - 2 = 6 :=
by
  -- Provide the correct Lean statement for the problem
  sorry

end find_multiple_of_brothers_l49_49278


namespace greatest_divisor_l49_49750

theorem greatest_divisor :
  ∃ x, (∀ y : ℕ, y > 0 → x ∣ (7^y + 12*y - 1)) ∧ (∀ z, (∀ y : ℕ, y > 0 → z ∣ (7^y + 12*y - 1)) → z ≤ x) ∧ x = 18 :=
sorry

end greatest_divisor_l49_49750


namespace repeating_decimal_conversion_l49_49578

-- Definition of 0.\overline{23} as a rational number
def repeating_decimal_fraction : ℚ := 23 / 99

-- The main statement to prove
theorem repeating_decimal_conversion : (3 / 10) + (repeating_decimal_fraction) = 527 / 990 := 
by
  -- Placeholder for proof steps
  sorry

end repeating_decimal_conversion_l49_49578


namespace count_primes_between_30_and_50_l49_49462

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_30_and_50 : List ℕ :=
  [31, 37, 41, 43, 47]

theorem count_primes_between_30_and_50 : 
  (primes_between_30_and_50.filter is_prime).length = 5 :=
by
  sorry

end count_primes_between_30_and_50_l49_49462


namespace N_cannot_be_sum_of_three_squares_l49_49494

theorem N_cannot_be_sum_of_three_squares (K : ℕ) (L : ℕ) (N : ℕ) (h1 : N = 4^K * L) (h2 : L % 8 = 7) : ¬ ∃ (a b c : ℕ), N = a^2 + b^2 + c^2 := 
sorry

end N_cannot_be_sum_of_three_squares_l49_49494


namespace ratio_20_to_10_exists_l49_49860

theorem ratio_20_to_10_exists (x : ℕ) (h : x = 20 * 10) : x = 200 :=
by sorry

end ratio_20_to_10_exists_l49_49860


namespace total_people_going_to_zoo_l49_49387

def cars : ℝ := 3.0
def people_per_car : ℝ := 63.0

theorem total_people_going_to_zoo : cars * people_per_car = 189.0 :=
by 
  sorry

end total_people_going_to_zoo_l49_49387


namespace book_pages_total_l49_49224

-- Definitions based on conditions
def pages_first_three_days: ℕ := 3 * 28
def pages_next_three_days: ℕ := 3 * 35
def pages_following_three_days: ℕ := 3 * 42
def pages_last_day: ℕ := 15

-- Total pages read calculated from above conditions
def total_pages_read: ℕ :=
  pages_first_three_days + pages_next_three_days + pages_following_three_days + pages_last_day

-- Proof problem statement: prove that the total pages read equal 330
theorem book_pages_total:
  total_pages_read = 330 :=
by
  sorry

end book_pages_total_l49_49224


namespace units_digit_7_pow_6_pow_5_l49_49912

theorem units_digit_7_pow_6_pow_5 : (7 ^ (6 ^ 5)) % 10 = 1 :=
by
  -- Using the cyclic pattern of the units digits of powers of 7: 7, 9, 3, 1
  have h1 : 7 % 10 = 7, by norm_num,
  have h2 : (7 ^ 2) % 10 = 9, by norm_num,
  have h3 : (7 ^ 3) % 10 = 3, by norm_num,
  have h4 : (7 ^ 4) % 10 = 1, by norm_num,

  -- Calculate 6 ^ 5 and the modular position
  have h6_5 : (6 ^ 5) % 4 = 0, by norm_num,

  -- Therefore, 7 ^ (6 ^ 5) % 10 = 7 ^ 0 % 10 because the cycle is 4
  have h_final : (7 ^ (6 ^ 5 % 4)) % 10 = (7 ^ 0) % 10, by rw h6_5,
  have h_zero : (7 ^ 0) % 10 = 1, by norm_num,

  rw h_final,
  exact h_zero,

end units_digit_7_pow_6_pow_5_l49_49912


namespace man_l49_49379

noncomputable def man_saves (S : ℝ) : ℝ :=
0.20 * S

noncomputable def initial_expenses (S : ℝ) : ℝ :=
0.80 * S

noncomputable def new_expenses (S : ℝ) : ℝ :=
1.10 * (0.80 * S)

noncomputable def said_savings (S : ℝ) : ℝ :=
S - new_expenses S

theorem man's_monthly_salary (S : ℝ) (h : said_savings S = 500) : S = 4166.67 :=
by
  sorry

end man_l49_49379


namespace salon_fingers_l49_49870

theorem salon_fingers (clients non_clients total_fingers cost_per_client total_earnings : Nat)
  (h1 : cost_per_client = 20)
  (h2 : total_earnings = 200)
  (h3 : total_fingers = 210)
  (h4 : non_clients = 11)
  (h_clients : clients = total_earnings / cost_per_client)
  (h_people : total_fingers / 10 = clients + non_clients) :
  10 = total_fingers / (clients + non_clients) :=
by
  sorry

end salon_fingers_l49_49870


namespace evaluate_expr_l49_49887

theorem evaluate_expr (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) : 5 * x^(y+1) + 6 * y^(x+1) = 2751 :=
by
  rw [h₁, h₂]
  rfl

end evaluate_expr_l49_49887


namespace zoe_pictures_l49_49136

theorem zoe_pictures (pictures_taken : ℕ) (dolphin_show_pictures : ℕ)
  (h1 : pictures_taken = 28) (h2 : dolphin_show_pictures = 16) :
  pictures_taken + dolphin_show_pictures = 44 :=
sorry

end zoe_pictures_l49_49136


namespace no_real_roots_of_equation_l49_49677

theorem no_real_roots_of_equation :
  (∃ x : ℝ, 2 * Real.cos (x / 2) = 10^x + 10^(-x) + 1) -> False :=
by
  sorry

end no_real_roots_of_equation_l49_49677


namespace total_spent_correct_l49_49279

def cost_ornamental_plants : Float := 467.00
def cost_garden_tool_set : Float := 85.00
def cost_potting_soil : Float := 38.00

def discount_plants : Float := 0.15
def discount_tools : Float := 0.10
def discount_soil : Float := 0.00

def sales_tax_rate : Float := 0.08
def surcharge : Float := 12.00

def discounted_price (original_price : Float) (discount_rate : Float) : Float :=
  original_price * (1.0 - discount_rate)

def subtotal (price_plants : Float) (price_tools : Float) (price_soil : Float) : Float :=
  price_plants + price_tools + price_soil

def sales_tax (amount : Float) (tax_rate : Float) : Float :=
  amount * tax_rate

def total (subtotal : Float) (sales_tax : Float) (surcharge : Float) : Float :=
  subtotal + sales_tax + surcharge

def final_total_spent : Float :=
  let price_plants := discounted_price cost_ornamental_plants discount_plants
  let price_tools := discounted_price cost_garden_tool_set discount_tools
  let price_soil := cost_potting_soil
  let subtotal_amount := subtotal price_plants price_tools price_soil
  let tax_amount := sales_tax subtotal_amount sales_tax_rate
  total subtotal_amount tax_amount surcharge

theorem total_spent_correct : final_total_spent = 564.37 :=
  by sorry

end total_spent_correct_l49_49279


namespace triangle_area_is_54_l49_49720

-- Define the sides of the triangle
def side1 : ℕ := 9
def side2 : ℕ := 12
def side3 : ℕ := 15

-- Verify that it is a right triangle using the Pythagorean theorem
def isRightTriangle (a b c : ℕ) : Prop := a * a + b * b = c * c

-- Define the area calculation for a right triangle
def areaRightTriangle (a b : ℕ) : ℕ := Nat.div (a * b) 2

-- State the theorem (Problem) to prove
theorem triangle_area_is_54 :
  isRightTriangle side1 side2 side3 ∧ areaRightTriangle side1 side2 = 54 :=
by
  sorry

end triangle_area_is_54_l49_49720


namespace people_with_fewer_than_7_cards_l49_49769

-- Definitions based on conditions
def cards_total : ℕ := 60
def people_total : ℕ := 9

-- Statement of the theorem
theorem people_with_fewer_than_7_cards : 
  ∃ (x : ℕ), x = 3 ∧ (cards_total % people_total = 0 ∨ cards_total % people_total < people_total) :=
by
  sorry

end people_with_fewer_than_7_cards_l49_49769


namespace sequence_property_l49_49576

open Classical

variable (a : ℕ → ℤ)
variable (h0 : a 1 = 0)
variable (h1 : ∀ n, a n ∈ {0, 1})
variable (h2 : ∀ n, a n + a (n + 1) ≠ a (n + 2) + a (n + 3))
variable (h3 : ∀ n, a n + a (n + 1) + a (n + 2) ≠ a (n + 3) + a (n + 4) + a (n + 5))

theorem sequence_property : a 2020 = 1 :=
by
  sorry

end sequence_property_l49_49576


namespace min_value_expression_l49_49188

theorem min_value_expression (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 3) (hy : 1 ≤ y ∧ y ≤ 4) : 
  ∃ z, z = (x + y) / x ∧ z = 4 / 3 := by
  sorry

end min_value_expression_l49_49188


namespace frustum_lateral_surface_area_l49_49559

/-- A frustum of a right circular cone has the following properties:
  * Lower base radius r1 = 8 inches
  * Upper base radius r2 = 2 inches
  * Height h = 6 inches
  The lateral surface area of such a frustum is 60 * √2 * π square inches.
-/
theorem frustum_lateral_surface_area : 
  let r1 := 8 
  let r2 := 2 
  let h := 6 
  let s := Real.sqrt (h^2 + (r1 - r2)^2)
  A = π * (r1 + r2) * s :=
  sorry

end frustum_lateral_surface_area_l49_49559


namespace avg_marks_l49_49205

theorem avg_marks (P C M B E H G : ℝ) 
  (h1 : C = P + 75)
  (h2 : M = P + 105)
  (h3 : B = P - 15)
  (h4 : E = P - 25)
  (h5 : H = P - 25)
  (h6 : G = P - 25)
  (h7 : P + C + M + B + E + H + G = P + 520) :
  (M + B + H + G) / 4 = 82 :=
by 
  sorry

end avg_marks_l49_49205


namespace trigonometric_identity_l49_49513

open Real

theorem trigonometric_identity (α φ : ℝ) :
  cos α ^ 2 + cos φ ^ 2 + cos (α + φ) ^ 2 - 2 * cos α * cos φ * cos (α + φ) = 1 :=
sorry

end trigonometric_identity_l49_49513


namespace geometric_progression_solution_l49_49584

theorem geometric_progression_solution (b4 b2 b6 : ℚ) (h1 : b4 - b2 = -45 / 32) (h2 : b6 - b4 = -45 / 512) :
  (∃ (b1 q : ℚ), b4 = b1 * q^3 ∧ b2 = b1 * q ∧ b6 = b1 * q^5 ∧ 
    ((b1 = 6 ∧ q = 1 / 4) ∨ (b1 = -6 ∧ q = -1 / 4))) :=
by
  sorry

end geometric_progression_solution_l49_49584


namespace different_colors_of_roads_leading_out_l49_49621

-- Define the city with intersections and streets
variables (n : ℕ) -- number of intersections
variables (c₁ c₂ c₃ : ℕ) -- number of external roads of each color

-- Conditions
axiom intersections_have_three_streets : ∀ (i : ℕ), i < n → (∀ (color : ℕ), color < 3 → exists (s : ℕ → ℕ), s color < n ∧ s color ≠ s ((color + 1) % 3) ∧ s color ≠ s ((color + 2) % 3))
axiom streets_colored_differently : ∀ (i : ℕ), i < n → (∀ (color1 color2 : ℕ), color1 < 3 → color2 < 3 → color1 ≠ color2 → exists (s1 s2 : ℕ → ℕ), s1 color1 < n ∧ s2 color2 < n ∧ s1 color1 ≠ s2 color2)

-- Problem Statement
theorem different_colors_of_roads_leading_out (h₁ : n % 2 = 0) (h₂ : c₁ + c₂ + c₃ = 3) : c₁ = 1 ∧ c₂ = 1 ∧ c₃ = 1 :=
by sorry

end different_colors_of_roads_leading_out_l49_49621


namespace smaller_of_two_digit_numbers_l49_49263

theorem smaller_of_two_digit_numbers (a b : ℕ) (h1 : 10 ≤ a ∧ a < 100) (h2 : 10 ≤ b ∧ b < 100) (h3 : a * b = 4725) :
  min a b = 15 :=
sorry

end smaller_of_two_digit_numbers_l49_49263


namespace matrix_det_is_neg16_l49_49161

def matrix := Matrix (Fin 2) (Fin 2) ℤ
def given_matrix : matrix := ![![ -7, 5], ![6, -2]]

theorem matrix_det_is_neg16 : Matrix.det given_matrix = -16 := 
by
  sorry

end matrix_det_is_neg16_l49_49161


namespace price_per_kg_of_fruits_l49_49093

theorem price_per_kg_of_fruits (mangoes apples oranges : ℕ) (total_amount : ℕ)
  (h1 : mangoes = 400)
  (h2 : apples = 2 * mangoes)
  (h3 : oranges = mangoes + 200)
  (h4 : total_amount = 90000) :
  (total_amount / (mangoes + apples + oranges) = 50) :=
by
  sorry

end price_per_kg_of_fruits_l49_49093


namespace no_HCl_formed_l49_49023

-- Definitions
def NaCl_moles : Nat := 3
def HNO3_moles : Nat := 3
def HCl_moles : Nat := 0

-- Hypothetical reaction context
-- if the reaction would produce HCl
axiom hypothetical_reaction : (NaCl_moles = 3) → (HNO3_moles = 3) → (∃ h : Nat, h = 3)

-- Proof under normal conditions that no HCl is formed
theorem no_HCl_formed : (NaCl_moles = 3) → (HNO3_moles = 3) → HCl_moles = 0 := by
  intros hNaCl hHNO3
  sorry

end no_HCl_formed_l49_49023


namespace tangent_line_eq_monotonic_intervals_l49_49450

noncomputable def f (x : ℝ) (a : ℝ) := x - a * Real.log x
noncomputable def f' (x : ℝ) (a : ℝ) := 1 - (a / x)

theorem tangent_line_eq (x y : ℝ) (h : x = 1 ∧ a = 2) :
  y = f 1 2 → (x - 1) + (y - 1) - 2 * ((x - 1) + (y - 1)) = 0 := by sorry

theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x > 0, f' x a > 0) ∧
  (a > 0 → ∀ x > 0, (x < a → f' x a < 0) ∧ (x > a → f' x a > 0)) := by sorry

end tangent_line_eq_monotonic_intervals_l49_49450


namespace people_with_fewer_than_7_cards_l49_49784

theorem people_with_fewer_than_7_cards (num_cards : ℕ) (num_people : ℕ) (h₁ : num_cards = 60) (h₂ : num_people = 9) : 
  ∃ k, k = num_people - num_cards % num_people ∧ k < 7 :=
by
  have rem := num_cards % num_people
  have few_count := num_people - rem
  use few_count
  split
  sorry

end people_with_fewer_than_7_cards_l49_49784


namespace james_weekly_hours_l49_49491

def james_meditation_total : ℕ :=
  let weekly_minutes := (30 * 2 * 6) + (30 * 2 * 2) -- 1 hour/day for 6 days + 2 hours on Sunday
  weekly_minutes / 60

def james_yoga_total : ℕ :=
  let weekly_minutes := (45 * 2) -- 45 minutes on Monday and Friday
  weekly_minutes / 60

def james_bikeride_total : ℕ :=
  let weekly_minutes := 90
  weekly_minutes / 60

def james_dance_total : ℕ :=
  2 -- 2 hours on Saturday

def james_total_activity_hours : ℕ :=
  james_meditation_total + james_yoga_total + james_bikeride_total + james_dance_total

theorem james_weekly_hours : james_total_activity_hours = 13 := by
  sorry

end james_weekly_hours_l49_49491


namespace perpendicular_line_eq_l49_49872

theorem perpendicular_line_eq :
  ∃ (A B C : ℝ), (A * 0 + B * 4 + C = 0) ∧ (A = 3) ∧ (B = 1) ∧ (C = -4) ∧ (3 * 1 + 1 * -3 = 0) :=
sorry

end perpendicular_line_eq_l49_49872


namespace min_distance_from_curve_to_line_l49_49988

open Real

-- Definitions and conditions
def curve_eq (x y: ℝ) : Prop := (x^2 - y - 2 * log (sqrt x) = 0)
def line_eq (x y: ℝ) : Prop := (4 * x + 4 * y + 1 = 0)

-- The main statement
theorem min_distance_from_curve_to_line :
  ∃ (x y : ℝ), curve_eq x y ∧ y = x^2 - 2 * log (sqrt x) ∧ line_eq x y ∧ y = -x - 1/4 ∧ 
               |4 * (1/2) + 4 * ((1/4) + log 2) + 1| / sqrt 32 = sqrt 2 / 2 * (1 + log 2) :=
by
  -- We skip the proof as requested, using sorry:
  sorry

end min_distance_from_curve_to_line_l49_49988


namespace prove_trigonometric_identities_l49_49926

variable {α : ℝ}

theorem prove_trigonometric_identities
  (h1 : 0 < α ∧ α < π)
  (h2 : Real.cos α = -3/5) :
  Real.tan α = -4/3 ∧
  (Real.cos (2 * α) - Real.cos (π / 2 + α) = 13/25) := 
by
  sorry

end prove_trigonometric_identities_l49_49926


namespace number_of_white_lights_l49_49333

theorem number_of_white_lights (red_lights low_brightness med_brightness high_brightness : ℕ) 
                                (yellow_lights blue_lights green_lights purple_lights : ℕ)
                                (extra_blue_lights extra_red_lights : ℕ) 
                                (brightness_conversion : ℕ → ℚ)
                                (h1 : low_brightness = 16)
                                (h2 : med_brightness = 1)
                                (h3 : high_brightness = 1.5)
                                (h4 : yellow_lights = 4)
                                (h5 : blue_lights = 2 * yellow_lights)
                                (h6 : green_lights = 8)
                                (h7 : purple_lights = 3)
                                (h8 : extra_blue_lights = nat.floor (0.25 * (blue_lights : ℚ)))
                                (h9 : extra_red_lights = 10)
                                (total_brightness : ℚ) :
  total_brightness = (low_brightness * 0.5 + high_brightness * 4 * 1.5 + 2 * yellow_lights * med_brightness + 
                      green_lights * 0.5 + purple_lights * 1.5 + extra_blue_lights * med_brightness +
                      extra_red_lights * 0.5) →
  38 := sorry

end number_of_white_lights_l49_49333


namespace monthly_income_l49_49305

variable {I : ℝ} -- George's monthly income

def donated_to_charity (I : ℝ) := 0.60 * I -- 60% of the income left
def paid_in_taxes (I : ℝ) := 0.75 * donated_to_charity I -- 75% of the remaining income after donation
def saved_for_future (I : ℝ) := 0.80 * paid_in_taxes I -- 80% of the remaining income after taxes
def expenses (I : ℝ) := saved_for_future I - 125 -- Remaining income after groceries and transportation expenses
def remaining_for_entertainment := 150 -- $150 left for entertainment and miscellaneous expenses

theorem monthly_income : I = 763.89 := 
by
  -- Using the conditions of the problem
  sorry

end monthly_income_l49_49305


namespace factor_poly_l49_49580

theorem factor_poly (x : ℝ) : (75 * x^3 - 300 * x^7) = 75 * x^3 * (1 - 4 * x^4) :=
by sorry

end factor_poly_l49_49580


namespace fraction_exponentiation_l49_49012

theorem fraction_exponentiation :
  (⟨1/3⟩ : ℝ) ^ 5 = (⟨1/243⟩ : ℝ) :=
by
  sorry

end fraction_exponentiation_l49_49012


namespace solve_system_of_equations_l49_49997

theorem solve_system_of_equations (x y : ℝ) (h1 : 2 * x + 3 * y = 7) (h2 : 4 * x - 3 * y = 5) : x = 2 ∧ y = 1 :=
by
    -- The proof is not required, so we put a sorry here.
    sorry

end solve_system_of_equations_l49_49997


namespace cyclic_quadrilaterals_count_l49_49253

noncomputable def num_cyclic_quadrilaterals (n : ℕ) : ℕ :=
  if n = 32 then 568 else 0 -- encapsulating the problem's answer

theorem cyclic_quadrilaterals_count :
  num_cyclic_quadrilaterals 32 = 568 :=
sorry

end cyclic_quadrilaterals_count_l49_49253


namespace translate_parabola_l49_49531

theorem translate_parabola :
  (∀ x : ℝ, (y : ℝ) = 6 * x^2 -> y = 6 * (x + 2)^2 + 3) :=
by
  sorry

end translate_parabola_l49_49531


namespace shorter_leg_of_right_triangle_l49_49800

theorem shorter_leg_of_right_triangle {a b : ℕ} (hypotenuse : ℕ) (h : hypotenuse = 41) (h_right_triangle : a^2 + b^2 = hypotenuse^2) (h_ineq : a < b) : a = 9 :=
by {
  -- proof to be filled in 
  sorry
}

end shorter_leg_of_right_triangle_l49_49800


namespace chord_length_l49_49960

-- Definitions and conditions for the problem
variables (A D B C G E F : Point)

-- Lengths and radii in the problem
noncomputable def radius : Real := 10
noncomputable def AB : Real := 20
noncomputable def BC : Real := 20
noncomputable def CD : Real := 20

-- Centers of circles
variables (O N P : Circle) (AN ND : Real)

-- Tangent properties and intersection points
variable (tangent_AG : Tangent AG P G)
variable (intersect_AG_N : Intersects AG N E F)

-- Given the geometry setup, prove the length of chord EF.
theorem chord_length (EF_length : Real) :
  EF_length = 2 * Real.sqrt 93.75 := sorry

end chord_length_l49_49960


namespace triangle_area_l49_49724

theorem triangle_area (a b c : ℕ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) (h₄ : a^2 + b^2 = c^2) : 
  (1 / 2 : ℝ) * a * b = 54 := 
by
  rw [h₁, h₂, h₃]
  -- The proof goes here
  sorry

end triangle_area_l49_49724


namespace total_feed_per_week_l49_49992

-- Define the conditions
def daily_feed_per_pig : ℕ := 10
def number_of_pigs : ℕ := 2
def days_per_week : ℕ := 7

-- Theorem statement
theorem total_feed_per_week : daily_feed_per_pig * number_of_pigs * days_per_week = 140 := 
  sorry

end total_feed_per_week_l49_49992


namespace solve_equation_l49_49231

theorem solve_equation : ∀ x : ℝ, (x - (x + 2) / 2 = (2 * x - 1) / 3 - 1) → (x = 2) :=
by
  intros x h
  sorry

end solve_equation_l49_49231


namespace mismatching_socks_l49_49234

theorem mismatching_socks (total_socks : ℕ) (pairs : ℕ) (socks_per_pair : ℕ) 
  (h1 : total_socks = 25) (h2 : pairs = 4) (h3 : socks_per_pair = 2) : 
  total_socks - (socks_per_pair * pairs) = 17 :=
by
  sorry

end mismatching_socks_l49_49234


namespace external_tangent_inequality_l49_49377

variable (x y z : ℝ)
variable (a b c T : ℝ)

-- Definitions based on conditions
def a_def : a = x + y := sorry
def b_def : b = y + z := sorry
def c_def : c = z + x := sorry
def T_def : T = π * x^2 + π * y^2 + π * z^2 := sorry

-- The theorem to prove
theorem external_tangent_inequality
    (a_def : a = x + y) 
    (b_def : b = y + z) 
    (c_def : c = z + x) 
    (T_def : T = π * x^2 + π * y^2 + π * z^2) : 
    π * (a + b + c) ^ 2 ≤ 12 * T := 
sorry

end external_tangent_inequality_l49_49377


namespace isosceles_trapezoid_ratio_l49_49632

theorem isosceles_trapezoid_ratio (a b d : ℝ) (h1 : b = 2 * d) (h2 : a = d) : a / b = 1 / 2 :=
by
  sorry

end isosceles_trapezoid_ratio_l49_49632


namespace sqrt_sqrt4_of_decimal_l49_49880

theorem sqrt_sqrt4_of_decimal (h : 0.000625 = 625 / (10 ^ 6)) :
  Real.sqrt (Real.sqrt (Real.sqrt (Real.sqrt 625) / 1000)) = 0.4 :=
by
  sorry

end sqrt_sqrt4_of_decimal_l49_49880


namespace store_owner_loss_percentage_l49_49275

theorem store_owner_loss_percentage :
  ∀ (initial_value : ℝ) (profit_margin : ℝ) (loss1 : ℝ) (loss2 : ℝ) (loss3 : ℝ) (tax_rate : ℝ),
    initial_value = 100 → profit_margin = 0.10 → loss1 = 0.20 → loss2 = 0.30 → loss3 = 0.25 → tax_rate = 0.12 →
      ((initial_value - initial_value * (1 - loss1) * (1 - loss2) * (1 - loss3)) / initial_value * 100) = 58 :=
by
  intros initial_value profit_margin loss1 loss2 loss3 tax_rate h_initial_value h_profit_margin h_loss1 h_loss2 h_loss3 h_tax_rate
  -- Variable assignments as per given conditions
  have h1 : initial_value = 100 := h_initial_value
  have h2 : profit_margin = 0.10 := h_profit_margin
  have h3 : loss1 = 0.20 := h_loss1
  have h4 : loss2 = 0.30 := h_loss2
  have h5 : loss3 = 0.25 := h_loss3
  have h6 : tax_rate = 0.12 := h_tax_rate
  
  sorry

end store_owner_loss_percentage_l49_49275


namespace range_of_a_l49_49329

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a*x^2 + (a+1)*x + a < 0) → a ∈ Set.Iio (-2 / 3) := 
sorry

end range_of_a_l49_49329


namespace directrix_parabola_l49_49523

theorem directrix_parabola (p : ℝ) (h : 4 * p = 2) : 
  ∃ d : ℝ, d = -p / 2 ∧ d = -1/2 :=
by
  sorry

end directrix_parabola_l49_49523


namespace irreducible_fraction_l49_49360

theorem irreducible_fraction (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 := 
  sorry

end irreducible_fraction_l49_49360


namespace initial_candies_proof_l49_49399

noncomputable def initial_candies (n : ℕ) := 
  ∃ c1 c2 c3 c4 c5 : ℕ, 
    c5 = 1 ∧
    c5 = n * 1 / 6 ∧
    c4 = n * 5 / 6 ∧
    c3 = n * 4 / 5 ∧
    c2 = n * 3 / 4 ∧
    c1 = n * 2 / 3 ∧
    n = 2 * c1

theorem initial_candies_proof (n : ℕ) : initial_candies n → n = 720 :=
  by
    sorry

end initial_candies_proof_l49_49399


namespace y2_over_x2_plus_x2_over_y2_eq_9_over_4_l49_49518

theorem y2_over_x2_plus_x2_over_y2_eq_9_over_4 (x y : ℝ) 
  (h : (1 / x) - (1 / (2 * y)) = (1 / (2 * x + y))) : 
  (y^2 / x^2) + (x^2 / y^2) = 9 / 4 := 
by 
  sorry

end y2_over_x2_plus_x2_over_y2_eq_9_over_4_l49_49518


namespace store_income_l49_49955

def pencil_store_income (p_with_eraser_qty p_with_eraser_cost p_regular_qty p_regular_cost p_short_qty p_short_cost : ℕ → ℝ) : ℝ :=
  (p_with_eraser_qty * p_with_eraser_cost) + (p_regular_qty * p_regular_cost) + (p_short_qty * p_short_cost)

theorem store_income : 
  pencil_store_income 200 0.8 40 0.5 35 0.4 = 194 := 
by sorry

end store_income_l49_49955


namespace exists_b_for_a_ge_condition_l49_49893

theorem exists_b_for_a_ge_condition (a : ℝ) (h : a ≥ -Real.sqrt 2 - 1 / 4) :
  ∃ b : ℝ, ∃ x y : ℝ, 
    y = x^2 - a ∧
    x^2 + y^2 + 8 * b^2 = 4 * b * (y - x) + 1 :=
sorry

end exists_b_for_a_ge_condition_l49_49893


namespace number_of_multiples_in_range_l49_49937

-- Definitions based on given conditions
def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def in_range (x lower upper : ℕ) : Prop := lower ≤ x ∧ x ≤ upper

def lcm_18_24_30 := ((2^3) * (3^2) * 5) -- LCM of 18, 24, and 30

-- Main theorem statement
theorem number_of_multiples_in_range : 
  (∃ a b c : ℕ, in_range a 2000 3000 ∧ is_multiple_of a lcm_18_24_30 ∧ 
                in_range b 2000 3000 ∧ is_multiple_of b lcm_18_24_30 ∧ 
                in_range c 2000 3000 ∧ is_multiple_of c lcm_18_24_30 ∧
                a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                ∀ z, in_range z 2000 3000 ∧ is_multiple_of z lcm_18_24_30 → z = a ∨ z = b ∨ z = c) := sorry

end number_of_multiples_in_range_l49_49937


namespace coby_travel_time_l49_49160

theorem coby_travel_time :
  let d1 := 640
  let d2 := 400
  let d3 := 250
  let d4 := 380
  let s1 := 80
  let s2 := 65
  let s3 := 75
  let s4 := 50
  let time1 := d1 / s1
  let time2 := d2 / s2
  let time3 := d3 / s3
  let time4 := d4 / s4
  let total_time := time1 + time2 + time3 + time4
  total_time = 25.08 :=
by
  sorry

end coby_travel_time_l49_49160


namespace complex_multiplication_l49_49753

theorem complex_multiplication (i : ℂ) (hi : i^2 = -1) : (1 + i) * (1 - i) = 1 := 
by
  sorry

end complex_multiplication_l49_49753


namespace reciprocal_inequality_l49_49191

open Real

theorem reciprocal_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (1 / a) + (1 / b) > 1 / (a + b) :=
sorry

end reciprocal_inequality_l49_49191


namespace range_of_a_l49_49946

theorem range_of_a (a : ℝ) : 
  (∃ n : ℕ, (∀ x : ℕ, 1 ≤ x → x ≤ 5 → x < a) ∧ (∀ y : ℕ, x ≥ 1 → y ≥ 6 → y ≥ a)) ↔ (5 < a ∧ a < 6) :=
by
  sorry

end range_of_a_l49_49946


namespace allan_balloons_l49_49153

theorem allan_balloons (a j t : ℕ) (h1 : t = 6) (h2 : j = 4) (h3 : t = a + j) : a = 2 := by
  sorry

end allan_balloons_l49_49153


namespace parabolaIntersection_l49_49866

-- Definitions related to the problem conditions
def basePlane : Plane := plane.mk [... some data ...]
def horizontalPlane : Plane := plane.mk [... some data ...]
def rightCircularCone : Cone := cone.mk basePlane [... some parameters ...]

-- Condition 1: Positioning of the cone's base
def coneBaseOnPlane {x1 x2 h : ℝ} (h > 0) : basePlane = (x_1, x_2, 0)

-- Condition 2: Horizontal plane height
def horizontalPlaneHeight {x1 x2 h : ℝ} (h > 0) : horizontalPlane = (x_1, x_2, 2 / 3 * h)

-- Condition 3: Parabola intersection with plane
def intersectionParabola (S : Plane) (cone : Cone) : IsParabola (S ∩ cone)

-- Condition 4: Directrix definition via intersection
def directrixDefinition (horizontalPlane : Plane) : Directrix :=
  horizontalPlane ∩ line.from_direction [... some data ...]

-- Theorem statement without proof
theorem parabolaIntersection 
    (cone : Cone)
    (basePlane = (x_1, x_2, 0))
    (horizontalPlaneHeight = (x_1, x_2, 2 / 3 * h))
    (S : Plane parallel to (x_1, x_2))
    (parabola := intersectionParabola S cone):
  ∃ focus directrix,
    IsParabolaIntersection cone horizontalPlane S parabola focus directrix := 
sorry

end parabolaIntersection_l49_49866


namespace jury_deliberation_days_l49_49628

theorem jury_deliberation_days
  (jury_selection_days trial_times jury_duty_days deliberation_hours_per_day hours_in_day : ℕ)
  (h1 : jury_selection_days = 2)
  (h2 : trial_times = 4)
  (h3 : jury_duty_days = 19)
  (h4 : deliberation_hours_per_day = 16)
  (h5 : hours_in_day = 24) :
  (jury_duty_days - jury_selection_days - (trial_times * jury_selection_days)) * deliberation_hours_per_day / hours_in_day = 6 := 
by
  sorry

end jury_deliberation_days_l49_49628


namespace limsup_liminf_prob_zero_l49_49705

open ProbabilityTheory

variables {Ω : Type*} {A : ℕ → Event Ω} [ProbabilitySpace Ω]

theorem limsup_liminf_prob_zero
  (h : ∑ n, P (A n ∆ A (n + 1)) < ∞) :
  P (limsup A \ liminf A) = 0 :=
by sorry

end limsup_liminf_prob_zero_l49_49705


namespace count_real_numbers_a_with_integer_roots_l49_49037

theorem count_real_numbers_a_with_integer_roots :
  ∃ (S : Finset ℝ), (∀ (a : ℝ), (∃ (x y : ℤ), x^2 + a*x + 9*a = 0 ∧ y^2 + a*y + 9*a = 0) ↔ a ∈ S) ∧ S.card = 8 :=
by
  sorry

end count_real_numbers_a_with_integer_roots_l49_49037


namespace problem_l49_49319

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

variables (f : ℝ → ℝ)
variables (h_odd : is_odd_function f)
variables (h_f1 : f 1 = 5)
variables (h_period : ∀ x, f (x + 4) = -f x)

-- Prove that f(2012) + f(2015) = -5
theorem problem :
  f 2012 + f 2015 = -5 :=
sorry

end problem_l49_49319


namespace solve_inequality_l49_49652

theorem solve_inequality :
  { x : ℝ | (x - 5) / (x - 3)^2 < 0 } = { x : ℝ | x < 3 } ∪ { x : ℝ | 3 < x ∧ x < 5 } :=
by
  sorry

end solve_inequality_l49_49652


namespace smallest_positive_natural_number_l49_49534

theorem smallest_positive_natural_number (a b c d e : ℕ) 
    (h1 : a = 3) (h2 : b = 5) (h3 : c = 6) (h4 : d = 18) (h5 : e = 23) :
    ∃ (x y : ℕ), x = (e - a) / b - d / c ∨ x = e - d + b - c - a ∧ x = 1 := by
  sorry

end smallest_positive_natural_number_l49_49534


namespace divide_inequality_by_negative_l49_49425

theorem divide_inequality_by_negative {x : ℝ} (h : -6 * x > 2) : x < -1 / 3 :=
by sorry

end divide_inequality_by_negative_l49_49425


namespace number_of_primes_between_30_and_50_l49_49460

/-- 
  Prove that there are exactly 5 prime numbers in the range from 30 to 50. 
  These primes are 31, 37, 41, 43, and 47.
-/
theorem number_of_primes_between_30_and_50 : 
  (Finset.filter Nat.Prime (Finset.range 51)).card - 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 5 := 
by 
  sorry

end number_of_primes_between_30_and_50_l49_49460


namespace prime_q_exists_l49_49995

theorem prime_q_exists (p : ℕ) (pp : Nat.Prime p) : 
  ∃ q, Nat.Prime q ∧ (∀ n, n > 0 → ¬ q ∣ n ^ p - p) := 
sorry

end prime_q_exists_l49_49995


namespace difference_two_digit_interchanged_l49_49373

theorem difference_two_digit_interchanged
  (x y : ℕ)
  (h1 : y = 2 * x)
  (h2 : (10 * x + y) - (x + y) = 8) :
  (10 * y + x) - (10 * x + y) = 9 := by
sorry

end difference_two_digit_interchanged_l49_49373


namespace smallest_number_of_people_l49_49708

theorem smallest_number_of_people (N : ℕ) :
  (∃ (N : ℕ), ∀ seats : ℕ, seats = 80 → N ≤ 80 → ∀ n : ℕ, n > N → (∃ m : ℕ, (m < N) ∧ ((seats + m) % 80 < seats))) → N = 20 :=
by
  sorry

end smallest_number_of_people_l49_49708


namespace select_female_athletes_l49_49404

theorem select_female_athletes (males females sample_size total_size : ℕ)
    (h1 : males = 56) (h2 : females = 42) (h3 : sample_size = 28)
    (h4 : total_size = males + females) : 
    (females * sample_size / total_size = 12) := 
by
  sorry

end select_female_athletes_l49_49404


namespace box_internal_volume_in_cubic_feet_l49_49626

def box_length := 26 -- inches
def box_width := 26 -- inches
def box_height := 14 -- inches
def wall_thickness := 1 -- inch

def external_volume := box_length * box_width * box_height -- cubic inches
def internal_length := box_length - 2 * wall_thickness
def internal_width := box_width - 2 * wall_thickness
def internal_height := box_height - 2 * wall_thickness
def internal_volume := internal_length * internal_width * internal_height -- cubic inches

def cubic_inches_to_cubic_feet (v : ℕ) : ℕ := v / 1728

theorem box_internal_volume_in_cubic_feet : cubic_inches_to_cubic_feet internal_volume = 4 := by
  sorry

end box_internal_volume_in_cubic_feet_l49_49626


namespace total_words_in_poem_l49_49993

theorem total_words_in_poem (stanzas lines words : ℕ) 
  (h1 : stanzas = 20) 
  (h2 : lines = 10) 
  (h3 : words = 8) :
  stanzas * lines * words = 1600 :=
by
  rw [h1, h2, h3]
  norm_num

end total_words_in_poem_l49_49993


namespace beth_total_crayons_l49_49006

theorem beth_total_crayons (packs : ℕ) (crayons_per_pack : ℕ) (extra_crayons : ℕ) 
  (h1 : packs = 8) (h2 : crayons_per_pack = 20) (h3 : extra_crayons = 15) :
  packs * crayons_per_pack + extra_crayons = 175 :=
by
  sorry

end beth_total_crayons_l49_49006


namespace simplify_fraction_l49_49945

theorem simplify_fraction (a b x : ℝ) (h₁ : x = a / b) (h₂ : a ≠ b) (h₃ : b ≠ 0) : 
  (2 * a + b) / (a - 2 * b) = (2 * x + 1) / (x - 2) :=
sorry

end simplify_fraction_l49_49945


namespace find_numerator_l49_49614

variable {y : ℝ} (hy : y > 0) (n : ℝ)

theorem find_numerator (h: (2 * y / 10) + n = 1 / 2 * y) : n = 3 :=
sorry

end find_numerator_l49_49614


namespace units_digit_of_power_l49_49908

theorem units_digit_of_power (a b : ℕ) : (a % 10 = 7) → (b % 4 = 0) → ((a^b) % 10 = 1) :=
by
  intros
  sorry

end units_digit_of_power_l49_49908


namespace max_ratio_three_digit_l49_49018

theorem max_ratio_three_digit (x a b c : ℕ) (h1 : 100 * a + 10 * b + c = x) (h2 : 1 ≤ a ∧ a ≤ 9)
  (h3 : 0 ≤ b ∧ b ≤ 9) (h4 : 0 ≤ c ∧ c ≤ 9) : 
  (x : ℚ) / (a + b + c) ≤ 100 := sorry

end max_ratio_three_digit_l49_49018


namespace intersection_of_cylinders_within_sphere_l49_49990

theorem intersection_of_cylinders_within_sphere (a b c d e f : ℝ) :
    ∀ (x y z : ℝ), 
      (x - a)^2 + (y - b)^2 < 1 ∧ 
      (y - c)^2 + (z - d)^2 < 1 ∧ 
      (z - e)^2 + (x - f)^2 < 1 → 
      (x - (a + f) / 2)^2 + (y - (b + c) / 2)^2 + (z - (d + e) / 2)^2 < 3 / 2 :=
by
  sorry

end intersection_of_cylinders_within_sphere_l49_49990


namespace student_correct_numbers_l49_49704

theorem student_correct_numbers (x y : ℕ) 
  (h1 : (10 * x + 5) * y = 4500)
  (h2 : (10 * x + 3) * y = 4380) : 
  (10 * x + 5 = 75 ∧ y = 60) :=
by 
  sorry

end student_correct_numbers_l49_49704


namespace larry_wins_game_l49_49343

-- Defining probabilities for Larry and Julius
def larry_throw_prob : ℚ := 2 / 3
def julius_throw_prob : ℚ := 1 / 3

-- Calculating individual probabilities based on the description
def p1 : ℚ := larry_throw_prob
def p3 : ℚ := (julius_throw_prob ^ 2) * larry_throw_prob
def p5 : ℚ := (julius_throw_prob ^ 4) * larry_throw_prob

-- Aggregating the probability that Larry wins the game
def larry_wins_prob : ℚ := p1 + p3 + p5

-- The proof statement
theorem larry_wins_game : larry_wins_prob = 170 / 243 := by
  sorry

end larry_wins_game_l49_49343


namespace am_gm_inequality_l49_49442

-- Definitions of the variables and hypotheses
variables {a b : ℝ}

-- The theorem statement
theorem am_gm_inequality (h : a * b > 0) : a / b + b / a ≥ 2 :=
sorry

end am_gm_inequality_l49_49442


namespace intersection_of_M_and_N_l49_49049

def M : Set ℝ := { x | x ≤ 0 }
def N : Set ℝ := { -2, 0, 1 }

theorem intersection_of_M_and_N : M ∩ N = { -2, 0 } := 
by
  sorry

end intersection_of_M_and_N_l49_49049


namespace solve_inequality_and_find_positive_int_solutions_l49_49107

theorem solve_inequality_and_find_positive_int_solutions :
  ∀ (x : ℝ), (2 * x + 1) / 3 - 1 ≤ (2 / 5) * x → x ≤ 2.5 ∧ ∃ (n : ℕ), n = 1 ∨ n = 2 :=
by
  intro x
  intro h
  sorry

end solve_inequality_and_find_positive_int_solutions_l49_49107


namespace expression_value_l49_49177

theorem expression_value (a b c : ℝ) (h : a * b + b * c + c * a = 3) : 
  (a * (b^2 + 3)) / (a + b) + (b * (c^2 + 3)) / (b + c) + (c * (a^2 + 3)) / (c + a) = 6 := 
by
  sorry

end expression_value_l49_49177


namespace expand_and_simplify_l49_49427

theorem expand_and_simplify (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * (14 / x^3 + 15 * x - 6 * x^5) = (6 / x^3) + (45 * x / 7) - (18 * x^5 / 7) :=
by
  sorry

end expand_and_simplify_l49_49427


namespace four_dice_probability_l49_49228

open ProbabilityTheory
open Classical

noncomputable def dice_prob_space : ProbabilitySpace := sorry -- Define the probability space of rolling six 6-sided dice

def condition_no_four_of_a_kind (dice_outcome : Vector ℕ 6) : Prop :=
  ¬∃ n, dice_outcome.count n ≥ 4

def condition_pair_exists (dice_outcome : Vector ℕ 6) : Prop :=
  ∃ n, dice_outcome.count n = 2

def re_rolled_dice (initial_outcome : Vector ℕ 6) (re_roll : Vector ℕ 4) : Vector ℕ 6 :=
  sorry -- Combine initial pair and re-rolled outcomes

def at_least_four_same (dice_outcome : Vector ℕ 6) : Prop :=
  ∃ n, dice_outcome.count n ≥ 4

theorem four_dice_probability :
  ∀ (initial_outcome : Vector ℕ 6)
    (re_roll : Vector ℕ 4),
  (condition_no_four_of_a_kind initial_outcome) →
  (condition_pair_exists initial_outcome) →
  (∃ pr : ℚ, pr = 311 / 648 ∧ 
    (Pr[dice_prob_space, at_least_four_same (re_rolled_dice initial_outcome re_roll)] = pr)) :=
sorry

end four_dice_probability_l49_49228


namespace sum_of_largest_and_smallest_four_digit_numbers_is_11990_l49_49303

theorem sum_of_largest_and_smallest_four_digit_numbers_is_11990 (A B C D : ℕ) 
    (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : B ≠ C) (h5 : B ≠ D) (h6 : C ≠ D)
    (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0)
    (h_eq : 1001 * A + 110 * B + 110 * C + 1001 * D = 11990) :
    (min (1000 * A + 100 * B + 10 * C + D) (1000 * D + 100 * C + 10 * B + A) = 1999) ∧
    (max (1000 * A + 100 * B + 10 * C + D) (1000 * D + 100 * C + 10 * B + A) = 9991) :=
by
  sorry

end sum_of_largest_and_smallest_four_digit_numbers_is_11990_l49_49303


namespace speed_of_other_train_l49_49848

theorem speed_of_other_train :
  ∀ (d : ℕ) (v1 v2 : ℕ), d = 120 → v1 = 30 → 
    ∀ (d_remaining : ℕ), d_remaining = 70 → 
    v1 + v2 = d_remaining → 
    v2 = 40 :=
by
  intros d v1 v2 h_d h_v1 d_remaining h_d_remaining h_rel_speed
  sorry

end speed_of_other_train_l49_49848


namespace probability_of_diamond_or_ace_at_least_one_l49_49392

noncomputable def prob_at_least_one_diamond_or_ace : ℚ := 
  1 - (9 / 13) ^ 2

theorem probability_of_diamond_or_ace_at_least_one :
  prob_at_least_one_diamond_or_ace = 88 / 169 := 
by
  sorry

end probability_of_diamond_or_ace_at_least_one_l49_49392


namespace product_is_2008th_power_l49_49249

theorem product_is_2008th_power (a b c : ℕ) (h1 : a = (b + c) / 2) (h2 : b ≠ c) (h3 : c ≠ a) (h4 : a ≠ b) :
  ∃ k : ℕ, (a * b * c) = k^2008 :=
by
  sorry

end product_is_2008th_power_l49_49249


namespace tony_combined_lift_weight_l49_49840

noncomputable def tony_exercises :=
  let curl_weight := 90 -- pounds.
  let military_press_weight := 2 * curl_weight -- pounds.
  let squat_weight := 5 * military_press_weight -- pounds.
  let bench_press_weight := 1.5 * military_press_weight -- pounds.
  squat_weight + bench_press_weight

theorem tony_combined_lift_weight :
  tony_exercises = 1170 := by
  -- Here we will include the necessary proof steps
  sorry

end tony_combined_lift_weight_l49_49840


namespace product_of_four_integers_l49_49208

theorem product_of_four_integers 
  (w x y z : ℕ) 
  (h1 : x * y * z = 280)
  (h2 : w * y * z = 168)
  (h3 : w * x * z = 105)
  (h4 : w * x * y = 120) :
  w * x * y * z = 840 :=
by {
sorry
}

end product_of_four_integers_l49_49208


namespace john_quiz_goal_l49_49878

theorem john_quiz_goal
  (total_quizzes : ℕ)
  (goal_percentage : ℕ)
  (quizzes_completed : ℕ)
  (quizzes_remaining : ℕ)
  (quizzes_with_A_completed : ℕ)
  (total_quizzes_with_A_needed : ℕ)
  (additional_A_needed : ℕ)
  (quizzes_below_A_allowed : ℕ)
  (h1 : total_quizzes = 60)
  (h2 : goal_percentage = 75)
  (h3 : quizzes_completed = 40)
  (h4 : quizzes_remaining = total_quizzes - quizzes_completed)
  (h5 : quizzes_with_A_completed = 27)
  (h6 : total_quizzes_with_A_needed = total_quizzes * goal_percentage / 100)
  (h7 : additional_A_needed = total_quizzes_with_A_needed - quizzes_with_A_completed)
  (h8 : quizzes_below_A_allowed = quizzes_remaining - additional_A_needed)
  (h_goal : quizzes_below_A_allowed ≤ 2) : quizzes_below_A_allowed = 2 :=
by
  sorry

end john_quiz_goal_l49_49878


namespace abs_diff_gt_half_prob_l49_49648

noncomputable def coin_flip_probability : Real :=
  let outcomes : Finset ℝ := {0, 0.5, 1}
  let prob := λ x : ℝ, 
    if x = 0 then 1/4
    else if x = 0.5 then 1/2
    else if x = 1 then 1/4
    else 0
  1/16 + 1/16

theorem abs_diff_gt_half_prob : coin_flip_probability = 1/8 := 
by
  sorry

end abs_diff_gt_half_prob_l49_49648


namespace Bill_donut_combinations_correct_l49_49568

/-- Bill has to purchase exactly six donuts from a shop with four kinds of donuts, ensuring he gets at least one of each kind. -/
def Bill_donut_combinations : ℕ :=
  let k := 4  -- number of kinds of donuts
  let n := 6  -- total number of donuts Bill needs to buy
  let m := 2  -- remaining donuts after buying one of each kind
  let same_kind := k          -- ways to choose 2 donuts of the same kind
  let different_kind := (k * (k - 1)) / 2  -- ways to choose 2 donuts of different kinds
  same_kind + different_kind

theorem Bill_donut_combinations_correct : Bill_donut_combinations = 10 :=
  by
    sorry  -- Proof is omitted; we assert this statement is true

end Bill_donut_combinations_correct_l49_49568


namespace reciprocal_of_lcm_24_221_l49_49520

theorem reciprocal_of_lcm_24_221 : (1 / Nat.lcm 24 221) = (1 / 5304) :=
by 
  sorry

end reciprocal_of_lcm_24_221_l49_49520


namespace arithmetic_seq_term_ratio_l49_49322

-- Assume two arithmetic sequences a and b
def arithmetic_seq_a (n : ℕ) : ℕ := sorry
def arithmetic_seq_b (n : ℕ) : ℕ := sorry

-- Sum of first n terms of the sequences
def sum_a (n : ℕ) : ℕ := (List.range (n+1)).map arithmetic_seq_a |>.sum
def sum_b (n : ℕ) : ℕ := (List.range (n+1)).map arithmetic_seq_b |>.sum

-- The given condition: Sn / Tn = (7n + 2) / (n + 3)
axiom sum_condition (n : ℕ) : (sum_a n) / (sum_b n) = (7 * n + 2) / (n + 3)

-- The goal: a4 / b4 = 51 / 10
theorem arithmetic_seq_term_ratio : (arithmetic_seq_a 4 : ℚ) / (arithmetic_seq_b 4 : ℚ) = 51 / 10 :=
by
  sorry

end arithmetic_seq_term_ratio_l49_49322


namespace minimum_value_of_a_l49_49451

theorem minimum_value_of_a :
  ∀ (x : ℝ), (2 * x + 2 / (x - 1) ≥ 7) ↔ (3 ≤ x) :=
sorry

end minimum_value_of_a_l49_49451


namespace age_of_replaced_person_l49_49111

theorem age_of_replaced_person (avg_age x : ℕ) (h1 : 10 * avg_age - 10 * (avg_age - 3) = x - 18) : x = 48 := 
by
  -- The proof goes here, but we are omitting it as per instruction.
  sorry

end age_of_replaced_person_l49_49111


namespace range_of_x_l49_49320

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - (1 / x) + 2 * Real.sin x

theorem range_of_x (x : ℝ) (h₀ : x > 0) (h₁ : f (1 - x) > f x) : x < (1 / 2) :=
by
  sorry

end range_of_x_l49_49320


namespace units_digit_of_7_pow_6_pow_5_l49_49894

-- Define the units digit cycle for powers of 7
def units_digit_cycle : List ℕ := [7, 9, 3, 1]

-- Define the function to calculate the units digit of 7^n
def units_digit (n : ℕ) : ℕ :=
  units_digit_cycle[(n % 4) - 1]

-- The main theorem stating the units digit of 7^(6^5) is 1
theorem units_digit_of_7_pow_6_pow_5 : units_digit (6^5) = 1 :=
by
  -- Skipping the proof, including a sorry placeholder
  sorry

end units_digit_of_7_pow_6_pow_5_l49_49894


namespace no_valid_two_digit_factors_l49_49942

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Main theorem to show: there are no valid two-digit factorizations of 1976
theorem no_valid_two_digit_factors : 
  ∃ (factors : ℕ → ℕ → Prop), (∀ (a b : ℕ), factors a b → (a * b = 1976) → (is_two_digit a) → (is_two_digit b)) → 
  ∃ (count : ℕ), count = 0 := 
sorry

end no_valid_two_digit_factors_l49_49942


namespace cube_volume_and_surface_area_l49_49246

theorem cube_volume_and_surface_area (s : ℝ) (h : 12 * s = 72) :
  s^3 = 216 ∧ 6 * s^2 = 216 :=
by 
  sorry

end cube_volume_and_surface_area_l49_49246


namespace solve_inequality_system_l49_49106

theorem solve_inequality_system (x : ℝ) :
  (5 * x - 1 > 3 * (x + 1)) ∧ (x - 1 ≤ 7 - x) ↔ (2 < x ∧ x ≤ 4) :=
by
  sorry

end solve_inequality_system_l49_49106


namespace minimum_value_expr_l49_49131

theorem minimum_value_expr (x y : ℝ) : 
  ∃ (a b : ℝ), 2 * x^2 + 3 * y^2 - 12 * x + 6 * y + 25 = 2 * (a - 3)^2 + 3 * (b + 1)^2 + 4 ∧ 
  2 * (a - 3)^2 + 3 * (b + 1)^2 + 4 ≥ 4 :=
by 
  sorry

end minimum_value_expr_l49_49131


namespace dinner_cost_l49_49732

theorem dinner_cost (tax_rate : ℝ) (tip_rate : ℝ) (total_amount : ℝ) : 
  tax_rate = 0.12 → 
  tip_rate = 0.18 → 
  total_amount = 30 → 
  (total_amount / (1 + tax_rate + tip_rate)) = 23.08 :=
by
  intros h1 h2 h3
  sorry

end dinner_cost_l49_49732


namespace number_of_multiples_in_range_l49_49938

-- Definitions based on given conditions
def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def in_range (x lower upper : ℕ) : Prop := lower ≤ x ∧ x ≤ upper

def lcm_18_24_30 := ((2^3) * (3^2) * 5) -- LCM of 18, 24, and 30

-- Main theorem statement
theorem number_of_multiples_in_range : 
  (∃ a b c : ℕ, in_range a 2000 3000 ∧ is_multiple_of a lcm_18_24_30 ∧ 
                in_range b 2000 3000 ∧ is_multiple_of b lcm_18_24_30 ∧ 
                in_range c 2000 3000 ∧ is_multiple_of c lcm_18_24_30 ∧
                a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                ∀ z, in_range z 2000 3000 ∧ is_multiple_of z lcm_18_24_30 → z = a ∨ z = b ∨ z = c) := sorry

end number_of_multiples_in_range_l49_49938


namespace sufficient_but_not_necessary_l49_49944

theorem sufficient_but_not_necessary (a : ℝ) : ((a = 2) → ((a - 1) * (a - 2) = 0)) ∧ (¬(((a - 1) * (a - 2) = 0) → (a = 2))) := 
by 
sorry

end sufficient_but_not_necessary_l49_49944


namespace angle_C_is_110_degrees_l49_49640

def lines_are_parallel (l m : Type) : Prop := sorry
def angle_measure (A : Type) : ℝ := sorry
noncomputable def mangle (C : Type) : ℝ := sorry

theorem angle_C_is_110_degrees 
  (l m C D : Type) 
  (hlm : lines_are_parallel l m)
  (hCDl : lines_are_parallel C l)
  (hCDm : lines_are_parallel C m)
  (hA : angle_measure A = 100)
  (hB : angle_measure B = 150) :
  mangle C = 110 :=
by
  sorry

end angle_C_is_110_degrees_l49_49640


namespace number_of_keyboards_l49_49711

-- Definitions based on conditions
def keyboard_cost : ℕ := 20
def printer_cost : ℕ := 70
def printers_bought : ℕ := 25
def total_cost : ℕ := 2050

-- The variable we want to prove
variable (K : ℕ)

-- The main theorem statement
theorem number_of_keyboards (K : ℕ) (keyboard_cost printer_cost printers_bought total_cost : ℕ) :
  keyboard_cost * K + printer_cost * printers_bought = total_cost → K = 15 :=
by
  -- Placeholder for the proof
  sorry

end number_of_keyboards_l49_49711


namespace hyperbola_foci_asymptote_shared_l49_49368

noncomputable def hyperbola_equation (a b : ℝ) : String :=
  "y^2/" ++ a^2.repr ++ " - x^2/" ++ b^2.repr ++ " = 1"

theorem hyperbola_foci_asymptote_shared :
  let ellipse_foci := λ c : ℝ, Real.sqrt (49 - 24) = 5,
      hyperbola_asymptote := λ r : ℝ, r = 4 / 3,
      hyperbola_shared := hyperbola_equation 4 3
  in ellipse_foci 5 ∧ hyperbola_asymptote (4 / 3) ∧ hyperbola_shared = "y^2/16 - x^2/9 = 1" :=
by
  let ellipse_foci := Real.sqrt (49 - 24) = 5
  let hyperbola_asymptote := 4 / 3
  let hyperbola_shared := hyperbola_equation 4 3
  have h1 : ellipse_foci := by norm_num [ellipse_foci]
  have h2 : hyperbola_asymptote = 4 / 3 := by norm_num [hyperbola_asymptote]
  have h3 : hyperbola_shared = "y^2/16 - x^2/9 = 1" := by norm_num [hyperbola_shared]
  exact ⟨h1, h2, h3⟩

end hyperbola_foci_asymptote_shared_l49_49368


namespace alexandra_brianna_meeting_probability_l49_49115

noncomputable def probability_meeting (A B : ℕ × ℕ) : ℚ :=
if A = (0,0) ∧ B = (5,7) then 347 / 768 else 0

theorem alexandra_brianna_meeting_probability :
  probability_meeting (0,0) (5,7) = 347 / 768 := 
by sorry

end alexandra_brianna_meeting_probability_l49_49115


namespace integral_cos_plus_one_l49_49888

theorem integral_cos_plus_one :
  ∫ x in - (Real.pi / 2).. (Real.pi / 2), (1 + Real.cos x) = Real.pi + 2 :=
by
  sorry

end integral_cos_plus_one_l49_49888


namespace max_sum_of_lengths_l49_49700

theorem max_sum_of_lengths (x y : ℕ) (hx : 1 < x) (hy : 1 < y) (hxy : x + 3 * y < 5000) :
  ∃ a b : ℕ, x = 2^a ∧ y = 2^b ∧ a + b = 20 := sorry

end max_sum_of_lengths_l49_49700


namespace rectangle_new_area_l49_49521

theorem rectangle_new_area (l w : ℝ) (h_area : l * w = 540) : 
  (1.15 * l) * (0.8 * w) = 497 :=
by
  sorry

end rectangle_new_area_l49_49521


namespace dan_destroyed_l49_49039

def balloons_initial (fred: ℝ) (sam: ℝ) : ℝ := fred + sam

theorem dan_destroyed (fred: ℝ) (sam: ℝ) (final_balloons: ℝ) (destroyed_balloons: ℝ) :
  fred = 10.0 →
  sam = 46.0 →
  final_balloons = 40.0 →
  destroyed_balloons = (balloons_initial fred sam) - final_balloons →
  destroyed_balloons = 16.0 := by
  intros h1 h2 h3 h4
  sorry

end dan_destroyed_l49_49039


namespace number_of_rectangles_l49_49361

theorem number_of_rectangles (horizontal_lines : Fin 6) (vertical_lines : Fin 5) 
                             (point : ℕ × ℕ) (h₁ : point = (3, 4)) : 
  ∃ ways : ℕ, ways = 24 :=
by {
  sorry
}

end number_of_rectangles_l49_49361


namespace polar_to_cartesian_2_pi_over_6_l49_49421

theorem polar_to_cartesian_2_pi_over_6 :
  let r : ℝ := 2
  let θ : ℝ := (Real.pi / 6)
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (Real.sqrt 3, 1) := by
    -- Initialize the constants and their values
    let r := 2
    let θ := Real.pi / 6
    let x := r * Real.cos θ
    let y := r * Real.sin θ
    -- Placeholder for the actual proof
    sorry

end polar_to_cartesian_2_pi_over_6_l49_49421


namespace trigonometric_identity_solution_l49_49586

noncomputable def trigonometric_identity_problem : Prop :=
  let a := 96 * Real.pi / 180
  let b := 24 * Real.pi / 180
  let c := 66 * Real.pi / 180
  (Real.cos a * Real.cos b - Real.sin a * Real.cos c) = -1/2

theorem trigonometric_identity_solution : trigonometric_identity_problem :=
by
  sorry

end trigonometric_identity_solution_l49_49586


namespace Alan_shells_l49_49095

theorem Alan_shells (l b a : ℕ) (h1 : l = 36) (h2 : b = l / 3) (h3 : a = 4 * b) : a = 48 :=
by
sorry

end Alan_shells_l49_49095


namespace intersection_property_l49_49765

def universal_set : Set ℝ := Set.univ

def M : Set ℝ := {-1, 1, 2, 4}

def N : Set ℝ := {x : ℝ | x > 2}

theorem intersection_property : (M ∩ N) = {4} := by
  sorry

end intersection_property_l49_49765


namespace jack_total_yen_l49_49077

def pounds := 42
def euros := 11
def yen := 3000
def pounds_per_euro := 2
def yen_per_pound := 100

theorem jack_total_yen : (euros * pounds_per_euro + pounds) * yen_per_pound + yen = 9400 := by
  sorry

end jack_total_yen_l49_49077


namespace negation_of_universal_l49_49119

theorem negation_of_universal: (¬(∀ x : ℝ, x > 1 → x^2 > 1)) ↔ (∃ x : ℝ, x > 1 ∧ x^2 ≤ 1) :=
by 
  sorry

end negation_of_universal_l49_49119


namespace minimum_n_120n_divisibility_l49_49608

theorem minimum_n_120n_divisibility (n : ℕ) : 
  (4 ∣ 120 * n) ∧ (8 ∣ 120 * n) ∧ (12 ∣ 120 * n) ↔ n = 1 :=
by
  sorry

end minimum_n_120n_divisibility_l49_49608


namespace hexagon_perimeter_l49_49402

theorem hexagon_perimeter (s : ℝ) (h_area : s ^ 2 * (3 * Real.sqrt 3 / 2) = 54 * Real.sqrt 3) :
  6 * s = 36 :=
by
  sorry

end hexagon_perimeter_l49_49402


namespace linda_savings_l49_49090

theorem linda_savings :
  let original_savings := 880
  let cost_of_tv := 220
  let amount_spent_on_furniture := original_savings - cost_of_tv
  let fraction_spent_on_furniture := amount_spent_on_furniture / original_savings
  fraction_spent_on_furniture = 3 / 4 :=
by
  -- original savings
  let original_savings := 880
  -- cost of the TV
  let cost_of_tv := 220
  -- amount spent on furniture
  let amount_spent_on_furniture := original_savings - cost_of_tv
  -- fraction spent on furniture
  let fraction_spent_on_furniture := amount_spent_on_furniture / original_savings

  -- need to show that this fraction is 3/4
  sorry

end linda_savings_l49_49090


namespace lowest_temperature_in_january_2023_l49_49284

theorem lowest_temperature_in_january_2023 
  (T_Beijing T_Shanghai T_Shenzhen T_Jilin : ℝ)
  (h_Beijing : T_Beijing = -5)
  (h_Shanghai : T_Shanghai = 6)
  (h_Shenzhen : T_Shenzhen = 19)
  (h_Jilin : T_Jilin = -22) :
  T_Jilin < T_Beijing ∧ T_Jilin < T_Shanghai ∧ T_Jilin < T_Shenzhen :=
by
  sorry

end lowest_temperature_in_january_2023_l49_49284


namespace binom_P_X_4_eq_3_times_0_4_4_l49_49476

noncomputable def P_X_equals_4 (n : ℕ) (X : ℕ → ℝ) : ℝ := 
  if (E (pmf_of_fn (λ k, if k = 4 then n * 0.4 else 0)) = 2) then 
    (pmf_of_fn (λ k, if k = 4 then (n choose 4) * (0.4) ^ 4 * (0.6) ^ (n - 4) else 0)) 4
  else 0

theorem binom_P_X_4_eq_3_times_0_4_4 : 
  (X : ℕ → ℝ) (hX : E (pmf_of_fn (λ k, if k = 4 then n * 0.4 else 0)) = 2) :
  P_X_equals_4 5 X = 3 * (0.4) ^ 4 :=
by sorry

end binom_P_X_4_eq_3_times_0_4_4_l49_49476


namespace part_a_part_b_part_c_l49_49260

-- Define the conditions for Payneful pairs
def isPaynefulPair (f g : ℝ → ℝ) : Prop :=
  (∀ x, f x ∈ Set.univ) ∧
  (∀ x, g x ∈ Set.univ) ∧
  (∀ x y, f (x + y) = f x * g y + g x * f y) ∧
  (∀ x y, g (x + y) = g x * g y - f x * f y) ∧
  (∃ a, f a ≠ 0)

-- Questions and corresponding proofs as Lean theorems
theorem part_a (f g : ℝ → ℝ) (hf : isPaynefulPair f g) : f 0 = 0 ∧ g 0 = 1 := sorry

def h (f g : ℝ → ℝ) (x : ℝ) : ℝ := (f x) ^ 2 + (g x) ^ 2

theorem part_b (f g : ℝ → ℝ) (hf : isPaynefulPair f g) : h f g 5 * h f g (-5) = 1 := sorry

theorem part_c (f g : ℝ → ℝ) (hf : isPaynefulPair f g)
  (h_bound_f : ∀ x, -10 ≤ f x ∧ f x ≤ 10) (h_bound_g : ∀ x, -10 ≤ g x ∧ g x ≤ 10):
  h f g 2021 = 1 := sorry

end part_a_part_b_part_c_l49_49260


namespace random_event_is_crane_among_chickens_l49_49410

-- Definitions of the idioms as events
def coveringTheSkyWithOneHand : Prop := false
def fumingFromAllSevenOrifices : Prop := false
def stridingLikeAMeteor : Prop := false
def standingOutLikeACraneAmongChickens : Prop := ¬false

-- The theorem stating that Standing out like a crane among chickens is a random event
theorem random_event_is_crane_among_chickens :
  ¬coveringTheSkyWithOneHand ∧ ¬fumingFromAllSevenOrifices ∧ ¬stridingLikeAMeteor → standingOutLikeACraneAmongChickens :=
by 
  sorry

end random_event_is_crane_among_chickens_l49_49410


namespace total_campers_went_rowing_l49_49554

theorem total_campers_went_rowing (morning_campers afternoon_campers : ℕ) (h_morning : morning_campers = 35) (h_afternoon : afternoon_campers = 27) : morning_campers + afternoon_campers = 62 := by
  -- handle the proof
  sorry

end total_campers_went_rowing_l49_49554


namespace shaded_trapezoids_perimeter_l49_49156

theorem shaded_trapezoids_perimeter :
  let l := 8
  let w := 6
  let half_diagonal_1 := (l^2 + w^2) / 2
  let perimeter := 2 * (w + (half_diagonal_1 / l))
  let total_perimeter := perimeter + perimeter + half_diagonal_1
  total_perimeter = 48 :=
by 
  sorry

end shaded_trapezoids_perimeter_l49_49156


namespace books_received_l49_49508

theorem books_received (initial_books : ℕ) (total_books : ℕ) (h1 : initial_books = 54) (h2 : total_books = 77) : (total_books - initial_books) = 23 :=
by
  sorry

end books_received_l49_49508


namespace intersection_of_A_and_B_l49_49594

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {-1, 0, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end intersection_of_A_and_B_l49_49594


namespace problem_intersection_union_complement_l49_49603

open Set Real

noncomputable def A : Set ℝ := {x | x ≥ 2}
noncomputable def B : Set ℝ := {y | y ≤ 3}

theorem problem_intersection_union_complement :
  (A ∩ B = {x | 2 ≤ x ∧ x ≤ 3}) ∧ 
  (A ∪ B = univ) ∧ 
  (compl A ∩ compl B = ∅) :=
by
  sorry

end problem_intersection_union_complement_l49_49603


namespace prime_count_30_to_50_l49_49465

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

def primes_in_range (a b: ℕ) : list ℕ :=
  list.filter is_prime (list.range' a (b - a + 1))

theorem prime_count_30_to_50 : (primes_in_range 30 50).length = 5 :=
by sorry

end prime_count_30_to_50_l49_49465


namespace find_k_l49_49524

theorem find_k : 
  (∃ y, -3 * (-15) + y = k ∧ 0.3 * (-15) + y = 10) → k = 59.5 :=
by
  sorry

end find_k_l49_49524


namespace inequality_proof_l49_49596

open Real

variable (a b c : ℝ)

theorem inequality_proof
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c) :
  sqrt (a * b * c) * (sqrt a + sqrt b + sqrt c) + (a + b + c) ^ 2 ≥ 
  4 * sqrt (3 * a * b * c * (a + b + c)) :=
by sorry

end inequality_proof_l49_49596


namespace distinct_values_least_count_l49_49271

theorem distinct_values_least_count (total_integers : ℕ) (mode_count : ℕ) (unique_mode : Prop) 
  (h1 : total_integers = 3200)
  (h2 : mode_count = 17)
  (h3 : unique_mode):
  ∃ (least_count : ℕ), least_count = 200 := by
  sorry

end distinct_values_least_count_l49_49271


namespace alternating_factorial_base_sum_correct_l49_49295

noncomputable def alternating_factorial_base_sum : ℕ :=
  ∑ n in List.range 124, (if n % 2 = 0 then 16 * (n / 2 + 1)! else -(16 * ((n - 1) / 2 + 1)!))

theorem alternating_factorial_base_sum_correct :
  f_1 - f_2 + f_3 - f_4 + ⋯ + (-1) ^ (list_length + 1) * f_j = 495 :=
begin
  sorry
end

end alternating_factorial_base_sum_correct_l49_49295


namespace minimum_value_exists_l49_49423

-- Definitions of the components
noncomputable def quadratic_expression (k x y : ℝ) : ℝ := 
  9 * x^2 - 12 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 9 * y + 12

theorem minimum_value_exists (k : ℝ) :
  (∃ x y : ℝ, quadratic_expression k x y = 0) ↔ k = 2 := 
sorry

end minimum_value_exists_l49_49423


namespace postcards_per_day_l49_49250

variable (income_per_card total_income days : ℕ)
variable (x : ℕ)

theorem postcards_per_day
  (h1 : income_per_card = 5)
  (h2 : total_income = 900)
  (h3 : days = 6)
  (h4 : total_income = income_per_card * x * days) :
  x = 30 :=
by
  rw [h1, h2, h3] at h4
  linarith

end postcards_per_day_l49_49250


namespace people_with_fewer_than_7_cards_l49_49792

theorem people_with_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (cards_per_person : ℕ) (remainder : ℕ) 
  (h : total_cards = total_people * cards_per_person + remainder)
  (h_cards : total_cards = 60)
  (h_people : total_people = 9)
  (h_cards_per_person : cards_per_person = 6)
  (h_remainder : remainder = 6) :
  (total_people - remainder) = 3 :=
by
  sorry

end people_with_fewer_than_7_cards_l49_49792


namespace problem_statement_l49_49876

open Classical

variable (p q : Prop)

theorem problem_statement (h1 : p ∨ q) (h2 : ¬(p ∧ q)) (h3 : ¬ p) : (p = (5 + 2 = 6) ∧ q = (6 > 2)) :=
by
  have hp : p = False := by sorry
  have hq : q = True := by sorry
  exact ⟨by sorry, by sorry⟩

end problem_statement_l49_49876


namespace cos_pi_minus_alpha_l49_49312

theorem cos_pi_minus_alpha (α : ℝ) (hα : α > π ∧ α < 3 * π / 2) (h : Real.sin α = -5/13) :
  Real.cos (π - α) = 12 / 13 := 
by
  sorry

end cos_pi_minus_alpha_l49_49312


namespace number_of_primes_between_30_and_50_l49_49461

/-- 
  Prove that there are exactly 5 prime numbers in the range from 30 to 50. 
  These primes are 31, 37, 41, 43, and 47.
-/
theorem number_of_primes_between_30_and_50 : 
  (Finset.filter Nat.Prime (Finset.range 51)).card - 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 5 := 
by 
  sorry

end number_of_primes_between_30_and_50_l49_49461


namespace units_digit_7_power_l49_49911

theorem units_digit_7_power (n : ℕ) : 
  (7 ^ (6 ^ 5)) % 10 = 1 :=
by
  have h1 : 7 % 10 = 7 := by norm_num
  have h2 : (7 ^ 2) % 10 = 49 % 10 := by rfl    -- 49 % 10 = 9
  have h3 : (7 ^ 3) % 10 = 343 % 10 := by rfl   -- 343 % 10 = 3
  have h4 : (7 ^ 4) % 10 = 2401 % 10 := by rfl  -- 2401 % 10 = 1
  have h_pattern : ∀ k : ℕ, 7 ^ (4 * k) % 10 = 1 := 
    by intro k; cases k; norm_num [pow_succ, mul_comm] -- Pattern repeats every 4
  have h_mod : 6 ^ 5 % 4 = 0 := by
    have h51 : 6 % 4 = 2 := by norm_num
    have h62 : (6 ^ 2) % 4 = 0 := by norm_num
    have h63 : (6 ^ 5) % 4 = (6 * 6 ^ 4) % 4 := by ring_exp
    rw [← h62, h51]; norm_num
  exact h_pattern (6 ^ 5 / 4) -- Using the repetition pattern

end units_digit_7_power_l49_49911


namespace percentage_subtracted_l49_49517

theorem percentage_subtracted (a : ℝ) (p : ℝ) (h : (1 - p / 100) * a = 0.97 * a) : p = 3 :=
by
  sorry

end percentage_subtracted_l49_49517


namespace solve_inequality_system_l49_49362

theorem solve_inequality_system (x : ℝ) :
  (x / 3 + 2 > 0) ∧ (2 * x + 5 ≥ 3) ↔ (x ≥ -1) :=
by
  sorry

end solve_inequality_system_l49_49362


namespace negation_equiv_l49_49454

variable (f : ℝ → ℝ)

theorem negation_equiv :
  ¬ (∀ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  ∃ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) < 0 := by
sorry

end negation_equiv_l49_49454


namespace max_m_for_inequality_min_4a2_9b2_c2_l49_49293

theorem max_m_for_inequality (m : ℝ) : (∀ x : ℝ, |x - 3| + |x - m| ≥ 2 * m) → m ≤ 1 := 
sorry

theorem min_4a2_9b2_c2 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (4 * a^2 + 9 * b^2 + c^2) = 36 / 49 ∧ a = 9 / 49 ∧ b = 4 / 49 ∧ c = 36 / 49 :=
sorry

end max_m_for_inequality_min_4a2_9b2_c2_l49_49293


namespace product_fraction_simplification_l49_49742

theorem product_fraction_simplification : 
  (1^4 - 1) / (1^4 + 1) * (2^4 - 1) / (2^4 + 1) * (3^4 - 1) / (3^4 + 1) *
  (4^4 - 1) / (4^4 + 1) * (5^4 - 1) / (5^4 + 1) * (6^4 - 1) / (6^4 + 1) *
  (7^4 - 1) / (7^4 + 1) = 50 := 
  sorry

end product_fraction_simplification_l49_49742


namespace abba_divisible_by_11_l49_49471

-- Given any two-digit number with digits a and b
def is_divisible_by_11 (a b : ℕ) : Prop :=
  (1001 * a + 110 * b) % 11 = 0

theorem abba_divisible_by_11 (a b : ℕ) (ha : a < 10) (hb : b < 10) : is_divisible_by_11 a b :=
  sorry

end abba_divisible_by_11_l49_49471


namespace sum_m_n_l49_49768

theorem sum_m_n (m n : ℤ) (h1 : m^2 - n^2 = 18) (h2 : m - n = 9) : m + n = 2 := 
by
  sorry

end sum_m_n_l49_49768


namespace sum_not_divisible_by_6_1_to_100_l49_49961

def sum_not_divisible_by_6 (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ x, ¬(6 ∣ x)).sum id

theorem sum_not_divisible_by_6_1_to_100 : sum_not_divisible_by_6 100 = 4234 := sorry

end sum_not_divisible_by_6_1_to_100_l49_49961


namespace largest_int_less_than_100_with_remainder_5_l49_49431

theorem largest_int_less_than_100_with_remainder_5 (x : ℤ) (n : ℤ) (h₁ : x = 7 * n + 5) (h₂ : x < 100) : 
  x = 96 := by
  sorry

end largest_int_less_than_100_with_remainder_5_l49_49431


namespace polar_to_cartesian_l49_49419

theorem polar_to_cartesian (r θ : ℝ) (h_r : r = 2) (h_θ : θ = π / 6) :
  (r * Real.cos θ, r * Real.sin θ) = (Real.sqrt 3, 1) :=
by
  rw [h_r, h_θ]
  have h_cos : Real.cos (π / 6) = Real.sqrt 3 / 2 := sorry -- This identity can be used from trigonometric property.
  have h_sin : Real.sin (π / 6) = 1 / 2 := sorry -- This identity can be used from trigonometric property.
  rw [h_cos, h_sin]
  -- some algebraic steps to simplifiy left sides to (Real.sqrt 3, 1) should follow here. using multiplication and commmutaivity properties mainly.
  sorry

end polar_to_cartesian_l49_49419


namespace range_of_a_l49_49474

theorem range_of_a {a : ℝ} :
  (∀ x : ℝ, (a-2)*x^2 + 2*(a-2)*x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) :=
sorry

end range_of_a_l49_49474


namespace origin_movement_by_dilation_l49_49397

/-- Given a dilation of the plane that maps a circle with radius 4 centered at (3,3) 
to a circle of radius 6 centered at (7,9), calculate the distance the origin (0,0)
moves under this transformation to be 0.5 * sqrt(10). -/
theorem origin_movement_by_dilation :
  let B := (3, 3)
  let B' := (7, 9)
  let radius_B := 4
  let radius_B' := 6
  let dilation_factor := radius_B' / radius_B
  let center_of_dilation := (-1, -3)
  let initial_distance := Real.sqrt ((-1)^2 + (-3)^2) 
  let moved_distance := dilation_factor * initial_distance
  moved_distance - initial_distance = 0.5 * Real.sqrt (10) := 
by
  sorry

end origin_movement_by_dilation_l49_49397


namespace rotate_result_l49_49127

noncomputable def original_vector : ℝ × ℝ × ℝ := (2, 1, 2)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

def is_orthogonal (v w : ℝ × ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3 = 0

def rotate_90_deg_zaxis (v : ℝ × ℝ × ℝ) : Prop :=
  ∃ (w : ℝ × ℝ × ℝ), magnitude w = magnitude v ∧ is_orthogonal v w

theorem rotate_result :
  rotate_90_deg_zaxis original_vector →
  original_vector.magnitude = (3:ℝ) →
  ∃ (w : ℝ × ℝ × ℝ),
    w = (6 / real.sqrt 17, 6 / real.sqrt 17, -9 / real.sqrt 17) :=
begin
  sorry
end

end rotate_result_l49_49127


namespace remaining_files_l49_49849

def initial_music_files : ℕ := 16
def initial_video_files : ℕ := 48
def deleted_files : ℕ := 30

theorem remaining_files :
  initial_music_files + initial_video_files - deleted_files = 34 := 
by
  sorry

end remaining_files_l49_49849


namespace cards_dealt_to_people_l49_49781

theorem cards_dealt_to_people (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  (∃ k, k = total_people - (total_cards % total_people) ∧ k = 3) := 
by
  sorry

end cards_dealt_to_people_l49_49781


namespace truth_values_of_p_and_q_l49_49336

theorem truth_values_of_p_and_q (p q : Prop) (h1 : p ∨ q) (h2 : ¬(p ∧ q)) (h3 : ¬p) : ¬p ∧ q :=
by
  sorry

end truth_values_of_p_and_q_l49_49336


namespace integer_points_on_line_l49_49869

/-- Given a line that passes through points C(3, 3) and D(150, 250),
prove that the number of other points with integer coordinates
that lie strictly between C and D is 48. -/
theorem integer_points_on_line {C D : ℝ × ℝ} (hC : C = (3, 3)) (hD : D = (150, 250)) :
  ∃ (n : ℕ), n = 48 ∧ 
  ∀ p : ℝ × ℝ, C.1 < p.1 ∧ p.1 < D.1 ∧ 
  C.2 < p.2 ∧ p.2 < D.2 → 
  (∃ (k : ℤ), p.1 = ↑k ∧ p.2 = (5/3) * p.1 - 2) :=
sorry

end integer_points_on_line_l49_49869


namespace expression_value_l49_49174

variables {a b c : ℝ}

theorem expression_value (h : a * b + b * c + c * a = 3) :
  (a * (b^2 + 3) / (a + b)) + (b * (c^2 + 3) / (b + c)) + (c * (a^2 + 3) / (c + a)) = 6 := 
  sorry

end expression_value_l49_49174


namespace probability_even_heads_in_50_tosses_l49_49828

-- Define the biased coin and its probability of heads
def p_heads : ℝ := 2 / 3
def p_tails : ℝ := 1 / 3
def n : ℕ := 50

-- Statement of the problem in Lean
theorem probability_even_heads_in_50_tosses :
  P (even_heads 50) = 1 / 2 * (1 + 1 / 3 ^ 50) :=
sorry

end probability_even_heads_in_50_tosses_l49_49828


namespace complex_number_solution_l49_49759

open Complex

theorem complex_number_solution (z : ℂ) (h : (2 * z - I) * (2 - I) = 5) : 
  z = 1 + I :=
sorry

end complex_number_solution_l49_49759


namespace units_digit_pow_7_6_5_l49_49898

theorem units_digit_pow_7_6_5 :
  let units_digit (n : ℕ) : ℕ := n % 10
  in units_digit (7 ^ (6 ^ 5)) = 9 :=
by
  let units_digit (n : ℕ) := n % 10
  sorry

end units_digit_pow_7_6_5_l49_49898


namespace unique_solution_l49_49429

theorem unique_solution (x y a : ℝ) :
  (x^2 + y^2 = 2 * a ∧ x + Real.log (y^2 + 1) / Real.log 2 = a) ↔ a = 0 ∧ x = 0 ∧ y = 0 :=
by
  sorry

end unique_solution_l49_49429


namespace daily_average_books_l49_49805

theorem daily_average_books (x : ℝ) (h1 : 4 * x + 1.4 * x = 216) : x = 40 :=
by 
  sorry

end daily_average_books_l49_49805


namespace exists_Q_R_l49_49179

noncomputable def P (x : ℚ) : ℚ := x^4 + x^3 + x^2 + x + 1

theorem exists_Q_R : ∃ (Q R : Polynomial ℚ), 
  (Q.degree > 0 ∧ R.degree > 0) ∧
  (∀ (y : ℚ), (Q.eval y) * (R.eval y) = P (5 * y^2)) :=
sorry

end exists_Q_R_l49_49179


namespace find_A_l49_49809

noncomputable def f (A B x : ℝ) : ℝ := A * x - 3 * B ^ 2
def g (B x : ℝ) : ℝ := B * x
variable (B : ℝ) (hB : B ≠ 0)

theorem find_A (h : f (A := A) B (g B 2) = 0) : A = 3 * B / 2 := by
  sorry

end find_A_l49_49809


namespace mariela_cards_l49_49811

theorem mariela_cards (cards_after_home : ℕ) (total_cards : ℕ) (cards_in_hospital : ℕ) : 
  cards_after_home = 287 → 
  total_cards = 690 → 
  cards_in_hospital = total_cards - cards_after_home → 
  cards_in_hospital = 403 := 
by 
  intros h1 h2 h3 
  rw [h1, h2] at h3 
  exact h3


end mariela_cards_l49_49811


namespace find_ff_half_l49_49307

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then x + 1 else -x + 3

theorem find_ff_half : f (f (1 / 2)) = 3 / 2 := 
by 
  sorry

end find_ff_half_l49_49307


namespace baker_cakes_left_l49_49734

theorem baker_cakes_left (cakes_made cakes_bought : ℕ) (h1 : cakes_made = 155) (h2 : cakes_bought = 140) : cakes_made - cakes_bought = 15 := by
  sorry

end baker_cakes_left_l49_49734


namespace total_payment_l49_49350

def work_hours := 2
def hourly_rate := 75
def part_cost := 150

theorem total_payment : work_hours * hourly_rate + part_cost = 300 := 
by 
  calc 
  2 * 75 + 150 = 150 + 150 : by rw mul_comm 2 75
             ... = 300 : by rw add_comm 150 150

# The term "sorry" is unnecessary due to the use of "by" tactic and commutativity rules simplifying the steps directly.

end total_payment_l49_49350


namespace only_integer_square_less_double_l49_49691

theorem only_integer_square_less_double : ∀ x : ℤ, x^2 < 2 * x → x = 1 :=
begin
  sorry,
end

end only_integer_square_less_double_l49_49691


namespace total_crosswalk_lines_l49_49695

theorem total_crosswalk_lines 
  (intersections : ℕ) 
  (crosswalks_per_intersection : ℕ) 
  (lines_per_crosswalk : ℕ)
  (h1 : intersections = 10)
  (h2 : crosswalks_per_intersection = 8)
  (h3 : lines_per_crosswalk = 30) :
  intersections * crosswalks_per_intersection * lines_per_crosswalk = 2400 := 
by {
  sorry
}

end total_crosswalk_lines_l49_49695


namespace minimize_expression_l49_49258

theorem minimize_expression : ∃ c : ℝ, c = 6 ∧ ∀ x : ℝ, (3 / 4) * (x ^ 2) - 9 * x + 7 ≥ (3 / 4) * (6 ^ 2) - 9 * 6 + 7 :=
by
  sorry

end minimize_expression_l49_49258


namespace first_laptop_cost_l49_49000

variable (x : ℝ)

def cost_first_laptop (x : ℝ) : ℝ := x
def cost_second_laptop (x : ℝ) : ℝ := 3 * x
def total_cost (x : ℝ) : ℝ := cost_first_laptop x + cost_second_laptop x
def budget : ℝ := 2000

theorem first_laptop_cost : total_cost x = budget → x = 500 :=
by
  intros h
  sorry

end first_laptop_cost_l49_49000


namespace lcm_factor_l49_49244

theorem lcm_factor (A B : ℕ) (hcf : ℕ) (factor1 : ℕ) (factor2 : ℕ) 
  (hcf_eq : hcf = 15) (factor1_eq : factor1 = 11) (A_eq : A = 225) 
  (hcf_divides_A : hcf ∣ A) (lcm_eq : Nat.lcm A B = hcf * factor1 * factor2) : 
  factor2 = 15 :=
by
  sorry

end lcm_factor_l49_49244


namespace lesser_fraction_l49_49685

theorem lesser_fraction (x y : ℚ) (h_sum : x + y = 17 / 24) (h_prod : x * y = 1 / 8) : min x y = 1 / 3 := by
  sorry

end lesser_fraction_l49_49685


namespace combine_like_terms_problem1_combine_like_terms_problem2_l49_49881

-- Problem 1 Statement
theorem combine_like_terms_problem1 (x y : ℝ) : 
  2*x - (x - y) + (x + y) = 2*x + 2*y :=
by
  sorry

-- Problem 2 Statement
theorem combine_like_terms_problem2 (x : ℝ) : 
  3*x^2 - 9*x + 2 - x^2 + 4*x - 6 = 2*x^2 - 5*x - 4 :=
by
  sorry

end combine_like_terms_problem1_combine_like_terms_problem2_l49_49881


namespace locus_of_M_l49_49185

/-- Define the coordinates of points A and B, and given point M(x, y) with the 
    condition x ≠ ±1, ensure the equation of the locus of point M -/
theorem locus_of_M (x y : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) 
  (h3 : (y / (x + 1)) + (y / (x - 1)) = 2) : x^2 - x * y - 1 = 0 := 
sorry

end locus_of_M_l49_49185


namespace more_stable_performance_l49_49376

theorem more_stable_performance (S_A2 S_B2 : ℝ) (hA : S_A2 = 0.2) (hB : S_B2 = 0.09) (h : S_A2 > S_B2) : 
  "B" = "B" :=
by
  sorry

end more_stable_performance_l49_49376


namespace probability_of_winning_quiz_l49_49152

theorem probability_of_winning_quiz :
  let n := 4 -- number of questions
  let choices := 3 -- number of choices per question
  let probability_correct := 1 / choices -- probability of answering correctly
  let probability_incorrect := 1 - probability_correct -- probability of answering incorrectly
  let probability_all_correct := probability_correct^n -- probability of getting all questions correct
  let probability_exactly_three_correct := 4 * probability_correct^3 * probability_incorrect -- probability of getting exactly 3 questions correct
  probability_all_correct + probability_exactly_three_correct = 1 / 9 :=
by
  sorry

end probability_of_winning_quiz_l49_49152


namespace divisor_of_70th_number_l49_49970

-- Define the conditions
def s (d : ℕ) (n : ℕ) : ℕ := n * d + 5

-- Theorem stating the given problem
theorem divisor_of_70th_number (d : ℕ) (h : s d 70 = 557) : d = 8 :=
by
  -- The proof is to be filled in later. 
  -- Now, just create the structure.
  sorry

end divisor_of_70th_number_l49_49970


namespace range_of_a_l49_49313

/-- Definitions for propositions p and q --/
def p (a : ℝ) : Prop := a > 0 ∧ a < 1
def q (a : ℝ) : Prop := (2 * a - 3) ^ 2 - 4 > 0

/-- Theorem stating the range of possible values for a given conditions --/
theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : ¬(p a) ∧ ¬(q a) = false) (h4 : p a ∨ q a) :
  (1 / 2 ≤ a ∧ a < 1) ∨ (a ≥ 5 / 2) :=
sorry

end range_of_a_l49_49313


namespace grid_blue_probability_l49_49417

-- Define the problem in Lean
theorem grid_blue_probability :
  let n := 4
  let p_tile_blue := 1 / 2
  let invariant_prob := (p_tile_blue ^ (n / 2))
  let pair_prob := (p_tile_blue * p_tile_blue)
  let total_pairs := (n * n / 2 - n / 2)
  let final_prob := (invariant_prob ^ 2) * (pair_prob ^ total_pairs)
  final_prob = 1 / 65536 := by
  sorry

end grid_blue_probability_l49_49417


namespace number_of_digits_of_1234_in_base5_l49_49054

def base5_representation_digits (n : ℕ) : ℕ :=
  if h : n > 0 then
    Nat.find (λ k, n < 5^(k + 1)) + 1
  else
    1

theorem number_of_digits_of_1234_in_base5 : base5_representation_digits 1234 = 5 := 
by
  unfold base5_representation_digits
  have h : ∃ k, 1234 < 5^(k+1), from Exists.intro 4 (by norm_num)
  simp [Nat.find_spec h]
  rfl

end number_of_digits_of_1234_in_base5_l49_49054


namespace stephanie_total_remaining_bills_l49_49655

-- Conditions
def electricity_bill : ℕ := 60
def electricity_paid : ℕ := electricity_bill
def gas_bill : ℕ := 40
def gas_paid : ℕ := (3 * gas_bill) / 4 + 5
def water_bill : ℕ := 40
def water_paid : ℕ := water_bill / 2
def internet_bill : ℕ := 25
def internet_payment : ℕ := 5
def internet_paid : ℕ := 4 * internet_payment

-- Define
def remaining_electricity : ℕ := electricity_bill - electricity_paid
def remaining_gas : ℕ := gas_bill - gas_paid
def remaining_water : ℕ := water_bill - water_paid
def remaining_internet : ℕ := internet_bill - internet_paid

def total_remaining : ℕ := remaining_electricity + remaining_gas + remaining_water + remaining_internet

-- Problem Statement
theorem stephanie_total_remaining_bills :
  total_remaining = 30 :=
by
  -- proof goes here (not required as per the instructions)
  sorry

end stephanie_total_remaining_bills_l49_49655


namespace find_p_a_l49_49924

variables (p : ℕ → ℝ) (a b : ℕ)

-- Given conditions
axiom p_b : p b = 0.5
axiom p_b_given_a : p b / p a = 0.2 
axiom p_a_inter_b : p a * p b = 0.36

-- Problem statement
theorem find_p_a : p a = 1.8 :=
by
  sorry

end find_p_a_l49_49924


namespace total_bottles_per_day_l49_49143

def num_cases_per_day : ℕ := 7200
def bottles_per_case : ℕ := 10

theorem total_bottles_per_day : num_cases_per_day * bottles_per_case = 72000 := by
  sorry

end total_bottles_per_day_l49_49143


namespace infinite_geometric_series_sum_l49_49443

theorem infinite_geometric_series_sum (a : ℕ → ℝ) (a1 : a 1 = 1) (r : ℝ) (h : r = 1 / 3) (S : ℝ) (H : S = a 1 / (1 - r)) : S = 3 / 2 :=
by
  sorry

end infinite_geometric_series_sum_l49_49443


namespace cheese_cookie_packs_l49_49393

def packs_per_box (P : ℕ) : Prop :=
  let cartons := 12
  let boxes_per_carton := 12
  let total_boxes := cartons * boxes_per_carton
  let total_cost := 1440
  let box_cost := total_cost / total_boxes
  let pack_cost := 1
  P = box_cost / pack_cost

theorem cheese_cookie_packs : packs_per_box 10 := by
  sorry

end cheese_cookie_packs_l49_49393


namespace doubles_tournament_handshakes_l49_49003

theorem doubles_tournament_handshakes :
  let num_teams := 3
  let players_per_team := 2
  let total_players := num_teams * players_per_team
  let handshakes_per_player := total_players - 2
  let total_handshakes := total_players * handshakes_per_player / 2
  total_handshakes = 12 :=
by
  sorry

end doubles_tournament_handshakes_l49_49003


namespace hancho_tape_length_l49_49051

noncomputable def tape_length (x : ℝ) : Prop :=
  (1 / 4) * (4 / 5) * x = 1.5

theorem hancho_tape_length : ∃ x : ℝ, tape_length x ∧ x = 7.5 :=
by sorry

end hancho_tape_length_l49_49051


namespace negation_of_universal_l49_49752

theorem negation_of_universal : (¬ ∀ x : ℝ, x^2 + 2 * x - 1 = 0) ↔ ∃ x : ℝ, x^2 + 2 * x - 1 ≠ 0 :=
by sorry

end negation_of_universal_l49_49752


namespace last_four_digits_of_7_pow_5000_l49_49546

theorem last_four_digits_of_7_pow_5000 (h : 7 ^ 250 ≡ 1 [MOD 1250]) : 7 ^ 5000 ≡ 1 [MOD 1250] :=
by
  -- Proof (will be omitted)
  sorry

end last_four_digits_of_7_pow_5000_l49_49546


namespace fg_eval_l49_49324

def f (x : ℤ) : ℤ := x^3
def g (x : ℤ) : ℤ := 4 * x + 5

theorem fg_eval : f (g (-2)) = -27 := by
  sorry

end fg_eval_l49_49324


namespace number_of_officers_l49_49072

theorem number_of_officers
  (avg_all : ℝ := 120)
  (avg_officer : ℝ := 420)
  (avg_non_officer : ℝ := 110)
  (num_non_officer : ℕ := 450) :
  ∃ O : ℕ, avg_all * (O + num_non_officer) = avg_officer * O + avg_non_officer * num_non_officer ∧ O = 15 :=
by
  sorry

end number_of_officers_l49_49072


namespace calculate_f3_minus_f4_l49_49928

-- Defining the function f and the given conditions
variables (f : ℝ → ℝ)
variable (odd_f : ∀ x, f (-x) = -f x)
variable (periodic_f : ∀ x, f (x + 2) = -f x)
variable (f1 : f 1 = 1)

-- Proving the required equality
theorem calculate_f3_minus_f4 : f 3 - f 4 = -1 :=
by
  sorry

end calculate_f3_minus_f4_l49_49928


namespace finish_fourth_task_l49_49340

noncomputable def time_task_starts : ℕ := 12 -- Time in hours (12:00 PM)
noncomputable def time_task_ends : ℕ := 15 -- Time in hours (3:00 PM)
noncomputable def total_tasks : ℕ := 4 -- Total number of tasks
noncomputable def tasks_time (tasks: ℕ) := (time_task_ends - time_task_starts) * 60 / (total_tasks - 1) -- Time in minutes for each task

theorem finish_fourth_task : tasks_time 1 + ((total_tasks - 1) * tasks_time 1) = 240 := -- 4:00 PM expressed as 240 minutes from 12:00 PM
by
  sorry

end finish_fourth_task_l49_49340


namespace num_people_fewer_than_7_cards_l49_49775

theorem num_people_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      remainder := total_cards % total_people
  in total_people - remainder = 3 :=
by
  sorry

end num_people_fewer_than_7_cards_l49_49775


namespace find_angle_EBC_l49_49606

-- Definitions of the given angle measures and their relationships.
variables {x : ℝ}

-- Conditions
def is_parallel (a b : ℝ) : Prop := a = b
def angle_AEG_eq_1_5x (α : ℝ) : Prop := α = 1.5 * x
def angle_BEG_eq_2x (β : ℝ) : Prop := β = 2 * x
def supplementary_angle (α β : ℝ) : Prop := α + β = 180

-- Proof of the desired angle measure
theorem find_angle_EBC (α β : ℝ) 
  (h_parallel : is_parallel 1 1)
  (h_angle_AEG : angle_AEG_eq_1_5x α)
  (h_angle_BEG : angle_BEG_eq_2x β)
  (h_supplementary : supplementary_angle (1.5 * x) (2 * x)) : 
  2 * x = 102.86 :=
by
  -- just to complete with proof, but skip the details
  sorry

end find_angle_EBC_l49_49606


namespace rainy_day_probability_l49_49452

theorem rainy_day_probability 
  (P_A : ℝ) (P_B : ℝ) (P_A_and_B : ℝ) 
  (h1 : P_A = 0.20) 
  (h2 : P_B = 0.18) 
  (h3 : P_A_and_B = 0.12) :
  P_A_and_B / P_A = 0.60 :=
sorry

end rainy_day_probability_l49_49452


namespace calculate_value_l49_49739

theorem calculate_value (a b c : ℤ) (h₁ : a = 5) (h₂ : b = -3) (h₃ : c = 4) : 2 * c / (a + b) = 4 :=
by
  rw [h₁, h₂, h₃]
  sorry

end calculate_value_l49_49739


namespace participants_count_l49_49686

theorem participants_count (F M : ℕ)
  (hF2 : F / 2 = 110)
  (hM4 : M / 4 = 330 - F - M / 3)
  (hFm : (F + M) / 3 = F / 2 + M / 4) :
  F + M = 330 :=
sorry

end participants_count_l49_49686


namespace problem_statement_l49_49609

theorem problem_statement (x : ℝ) (h : 7 * x = 3) : 150 * (1 / x) = 350 :=
by
  sorry

end problem_statement_l49_49609


namespace triangle_inequality_sides_l49_49495

theorem triangle_inequality_sides {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (triangle_ineq1 : a + b > c) (triangle_ineq2 : b + c > a) (triangle_ineq3 : c + a > b) : 
  |(a / b) + (b / c) + (c / a) - (b / a) - (c / b) - (a / c)| < 1 :=
  sorry

end triangle_inequality_sides_l49_49495


namespace irrational_sqrt3_l49_49133

theorem irrational_sqrt3 : ¬ ∃ (a b : ℕ), b ≠ 0 ∧ (a * a = 3 * b * b) :=
by
  sorry

end irrational_sqrt3_l49_49133


namespace range_of_m_l49_49196

theorem range_of_m (m : ℝ) : 
  (¬(-2 ≤ 1 - (x - 1) / 3 ∧ (1 - (x - 1) / 3 ≤ 2)) → (∀ x, m > 0 → x^2 - 2*x + 1 - m^2 > 0)) → 
  (40 ≤ m ∧ m < 50) :=
by
  sorry

end range_of_m_l49_49196


namespace jack_total_yen_l49_49076

theorem jack_total_yen (pounds euros yen : ℕ) (pounds_per_euro yen_per_pound : ℕ) 
  (h_pounds : pounds = 42) 
  (h_euros : euros = 11) 
  (h_yen : yen = 3000) 
  (h_pounds_per_euro : pounds_per_euro = 2) 
  (h_yen_per_pound : yen_per_pound = 100) : 
  9400 = yen + (pounds * yen_per_pound) + ((euros * pounds_per_euro) * yen_per_pound) :=
by
  rw [h_pounds, h_euros, h_yen, h_pounds_per_euro, h_yen_per_pound]
  norm_num
  sorry

end jack_total_yen_l49_49076


namespace sequence_increasing_range_l49_49592

theorem sequence_increasing_range (a : ℝ) (n : ℕ) : 
  (∀ n ≤ 5, (a - 1) ^ (n - 4) < (a - 1) ^ ((n+1) - 4)) ∧
  (∀ n > 5, (7 - a) * n - 1 < (7 - a) * (n + 1) - 1) ∧
  (a - 1 < (7 - a) * 6 - 1) 
  → 2 < a ∧ a < 6 := 
sorry

end sequence_increasing_range_l49_49592


namespace parametric_to_general_eq_l49_49372

theorem parametric_to_general_eq (x y θ : ℝ) 
  (h1 : x = 2 + Real.sin θ ^ 2) 
  (h2 : y = -1 + Real.cos (2 * θ)) : 
  2 * x + y - 4 = 0 ∧ 2 ≤ x ∧ x ≤ 3 := 
sorry

end parametric_to_general_eq_l49_49372


namespace share_ratio_l49_49394

theorem share_ratio (A B C x : ℝ)
  (h1 : A = 280)
  (h2 : A + B + C = 700)
  (h3 : A = x * (B + C))
  (h4 : B = (6 / 9) * (A + C)) :
  A / (B + C) = 2 / 3 :=
by
  sorry

end share_ratio_l49_49394


namespace percentage_of_teachers_without_issues_l49_49712

theorem percentage_of_teachers_without_issues (total_teachers : ℕ) 
    (high_bp_teachers : ℕ) (heart_issue_teachers : ℕ) 
    (both_issues_teachers : ℕ) (h1 : total_teachers = 150) 
    (h2 : high_bp_teachers = 90) 
    (h3 : heart_issue_teachers = 60) 
    (h4 : both_issues_teachers = 30) : 
    (total_teachers - (high_bp_teachers + heart_issue_teachers - both_issues_teachers)) / total_teachers * 100 = 20 :=
by sorry

end percentage_of_teachers_without_issues_l49_49712


namespace jason_cost_l49_49642

variable (full_page_cost_per_square_inch : ℝ := 6.50)
variable (half_page_cost_per_square_inch : ℝ := 8)
variable (quarter_page_cost_per_square_inch : ℝ := 10)

variable (full_page_area : ℝ := 9 * 12)
variable (half_page_area : ℝ := full_page_area / 2)
variable (quarter_page_area : ℝ := full_page_area / 4)

variable (half_page_ads : ℝ := 1)
variable (quarter_page_ads : ℝ := 4)

variable (total_ads : ℝ := half_page_ads + quarter_page_ads)
variable (bulk_discount : ℝ := if total_ads >= 4 then 0.10 else 0.0)

variable (half_page_cost : ℝ := half_page_area * half_page_cost_per_square_inch)
variable (quarter_page_cost : ℝ := quarter_page_ads * (quarter_page_area * quarter_page_cost_per_square_inch))

variable (total_cost_before_discount : ℝ := half_page_cost + quarter_page_cost)
variable (discount_amount : ℝ := total_cost_before_discount * bulk_discount)
variable (final_cost : ℝ := total_cost_before_discount - discount_amount)

theorem jason_cost :
  final_cost = 1360.80 := by
  sorry

end jason_cost_l49_49642


namespace uncle_wang_withdraw_amount_l49_49533

noncomputable def total_amount (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal + principal * rate * time

theorem uncle_wang_withdraw_amount :
  total_amount 100000 (315/10000) 2 = 106300 := by
  sorry

end uncle_wang_withdraw_amount_l49_49533


namespace expression_value_l49_49175

variables {a b c : ℝ}

theorem expression_value (h : a * b + b * c + c * a = 3) :
  (a * (b^2 + 3) / (a + b)) + (b * (c^2 + 3) / (b + c)) + (c * (a^2 + 3) / (c + a)) = 6 := 
  sorry

end expression_value_l49_49175


namespace unique_intersection_points_l49_49862

noncomputable def line_segments_intersection_points (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) : ℕ :=
  Nat.choose m 2 * Nat.choose n 2

theorem unique_intersection_points (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  line_segments_intersection_points m n hm hn = Nat.choose m 2 * Nat.choose n 2 := by
  -- Start proof (actual proof is not required as per instructions)
  sorry

end unique_intersection_points_l49_49862


namespace find_PA_PB_sum_2sqrt6_l49_49335

noncomputable def polar_equation (ρ θ : ℝ) : Prop :=
  ρ - 2 * Real.cos θ - 6 * Real.sin θ + 1 / ρ = 0

noncomputable def parametric_line (t x y : ℝ) : Prop :=
  x = 3 + 1 / 2 * t ∧ y = 3 + Real.sqrt 3 / 2 * t

def point_P (x y : ℝ) : Prop :=
  x = 3 ∧ y = 3

theorem find_PA_PB_sum_2sqrt6 :
  (∃ ρ θ t₁ t₂, polar_equation ρ θ ∧ parametric_line t₁ 3 3 ∧ parametric_line t₂ 3 3 ∧
  point_P 3 3 ∧ |t₁| + |t₂| = 2 * Real.sqrt 6) := sorry

end find_PA_PB_sum_2sqrt6_l49_49335


namespace remainder_is_15x_minus_14_l49_49172

noncomputable def remainder_polynomial_division : Polynomial ℝ :=
  (Polynomial.X ^ 4) % (Polynomial.X ^ 2 - 3 * Polynomial.X + 2)

theorem remainder_is_15x_minus_14 :
  remainder_polynomial_division = 15 * Polynomial.X - 14 :=
by
  sorry

end remainder_is_15x_minus_14_l49_49172


namespace infinite_geometric_series_sum_l49_49570

/-
Mathematical problem: Calculate the sum of the infinite geometric series 1 + (1/2) + (1/2)^2 + (1/2)^3 + ... . Express your answer as a common fraction.

Conditions:
- The first term \( a \) is 1.
- The common ratio \( r \) is \(\frac{1}{2}\).

Answer:
- The sum of the series is 2.
-/

theorem infinite_geometric_series_sum :
  let a := 1
  let r := 1 / 2
  (a * (1 / (1 - r))) = 2 :=
by
  let a := 1
  let r := 1 / 2
  have h : 1 * (1 / (1 - r)) = 2 := by sorry
  exact h

end infinite_geometric_series_sum_l49_49570


namespace angles_sum_540_l49_49178

theorem angles_sum_540 (p q r s : ℝ) (h1 : ∀ a, a + (180 - a) = 180)
  (h2 : ∀ a b, (180 - a) + (180 - b) = 360 - a - b)
  (h3 : ∀ p q r, (360 - p - q) + (180 - r) = 540 - p - q - r) :
  p + q + r + s = 540 :=
sorry

end angles_sum_540_l49_49178


namespace MrFletcher_paid_l49_49218

noncomputable def total_payment (hours_day1 hours_day2 hours_day3 rate_per_hour men : ℕ) : ℕ :=
  let total_hours := hours_day1 + hours_day2 + hours_day3
  let total_man_hours := total_hours * men
  total_man_hours * rate_per_hour

theorem MrFletcher_paid
  (hours_day1 hours_day2 hours_day3 : ℕ)
  (rate_per_hour men : ℕ)
  (h1 : hours_day1 = 10)
  (h2 : hours_day2 = 8)
  (h3 : hours_day3 = 15)
  (h4 : rate_per_hour = 10)
  (h5 : men = 2) :
  total_payment hours_day1 hours_day2 hours_day3 rate_per_hour men = 660 := 
by {
  -- skipped proof details
  sorry
}

end MrFletcher_paid_l49_49218


namespace fraction_sum_eq_l49_49835

theorem fraction_sum_eq : (7 / 10 : ℚ) + (3 / 100) + (9 / 1000) = 0.739 := sorry

end fraction_sum_eq_l49_49835


namespace cone_height_is_2_sqrt_15_l49_49556

noncomputable def height_of_cone (radius : ℝ) (num_sectors : ℕ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let sector_arc_length := circumference / num_sectors
  let base_radius := sector_arc_length / (2 * Real.pi)
  let slant_height := radius
  Real.sqrt (slant_height ^ 2 - base_radius ^ 2)

theorem cone_height_is_2_sqrt_15 :
  height_of_cone 8 4 = 2 * Real.sqrt 15 :=
by
  sorry

end cone_height_is_2_sqrt_15_l49_49556


namespace ratio_passengers_i_to_ii_l49_49829

-- Definitions: Conditions from the problem
variables (total_fare : ℕ) (fare_ii_class : ℕ) (fare_i_class_ratio_to_ii : ℕ)

-- Given conditions
axiom total_fare_collected : total_fare = 1325
axiom fare_collected_from_ii_class : fare_ii_class = 1250
axiom i_to_ii_fare_ratio : fare_i_class_ratio_to_ii = 3

-- Define the fare for I class and II class passengers
def fare_i_class := 3 * (fare_ii_class / fare_i_class_ratio_to_ii)

-- Statement of the proof problem translating the question, conditions, and answer
theorem ratio_passengers_i_to_ii (x y : ℕ) (h1 : 3 * fare_i_class * x = total_fare - fare_ii_class)
    (h2 : (fare_ii_class / fare_i_class_ratio_to_ii) * y = fare_ii_class) : x = y / 50 :=
by
  sorry

end ratio_passengers_i_to_ii_l49_49829


namespace remainder_when_divided_by_10_l49_49852

theorem remainder_when_divided_by_10 :
  (2457 * 6291 * 9503) % 10 = 1 :=
by
  sorry

end remainder_when_divided_by_10_l49_49852


namespace coordinates_satisfy_l49_49645

theorem coordinates_satisfy (x y : ℝ) : y * (x + 1) = x^2 - 1 ↔ (x = -1 ∨ y = x - 1) :=
by
  sorry

end coordinates_satisfy_l49_49645


namespace mismatching_socks_l49_49233

theorem mismatching_socks (total_socks : ℕ) (pairs : ℕ) (socks_per_pair : ℕ) 
  (h1 : total_socks = 25) (h2 : pairs = 4) (h3 : socks_per_pair = 2) : 
  total_socks - (socks_per_pair * pairs) = 17 :=
by
  sorry

end mismatching_socks_l49_49233


namespace average_percentage_increase_l49_49341

def initial_income_A : ℝ := 60
def new_income_A : ℝ := 80
def initial_income_B : ℝ := 100
def new_income_B : ℝ := 130
def hours_worked_C : ℝ := 20
def initial_rate_C : ℝ := 8
def new_rate_C : ℝ := 10

theorem average_percentage_increase :
  let initial_weekly_income_C := hours_worked_C * initial_rate_C
  let new_weekly_income_C := hours_worked_C * new_rate_C
  let percentage_increase_A := (new_income_A - initial_income_A) / initial_income_A * 100
  let percentage_increase_B := (new_income_B - initial_income_B) / initial_income_B * 100
  let percentage_increase_C := (new_weekly_income_C - initial_weekly_income_C) / initial_weekly_income_C * 100
  let average_percentage_increase := (percentage_increase_A + percentage_increase_B + percentage_increase_C) / 3
  average_percentage_increase = 29.44 :=
by sorry

end average_percentage_increase_l49_49341


namespace solve_linear_eq_l49_49682

theorem solve_linear_eq : (∃ x : ℝ, 2 * x - 1 = 0) ↔ (∃ x : ℝ, x = 1/2) :=
by
  sorry

end solve_linear_eq_l49_49682


namespace base_length_of_isosceles_triangle_l49_49662

theorem base_length_of_isosceles_triangle (a b : ℕ) 
    (h₁ : a = 8) 
    (h₂ : 2 * a + b = 25) : 
    b = 9 :=
by
  -- This is the proof stub. Proof will be provided here.
  sorry

end base_length_of_isosceles_triangle_l49_49662


namespace math_proof_problem_l49_49216

noncomputable def discriminant (a : ℝ) : ℝ := a^2 - 4 * a + 2

def is_real_roots (a : ℝ) : Prop := discriminant a ≥ 0

def solution_set_a : Set ℝ := { a | is_real_roots a ∧ (a ≤ 2 - Real.sqrt 2 ∨ a ≥ 2 + Real.sqrt 2) }

def f (a : ℝ) : ℝ := -3 * a^2 + 16 * a - 8

def inequality_m (m t : ℝ) : Prop := m^2 + t * m + 4 * Real.sqrt 2 + 6 ≥ f (2 + Real.sqrt 2)

theorem math_proof_problem :
  (∀ a ∈ solution_set_a, ∃ m : ℝ, ∀ t ∈ Set.Icc (-1 : ℝ) (1 : ℝ), inequality_m m t) ∧
  (∀ m t, inequality_m m t → m ≤ -1 ∨ m = 0 ∨ m ≥ 1) :=
by
  sorry

end math_proof_problem_l49_49216


namespace alice_operations_terminate_l49_49438

theorem alice_operations_terminate (a : List ℕ) (h_pos : ∀ x ∈ a, x > 0) : 
(∀ x y z, (x, y) = (y + 1, x) ∨ (x, y) = (x - 1, x) → ∃ n, (x :: y :: z).sum ≤ n) :=
by sorry

end alice_operations_terminate_l49_49438


namespace sasha_remaining_questions_l49_49101

variable (rate : Int) (initial_questions : Int) (hours_worked : Int)

theorem sasha_remaining_questions
  (h1 : rate = 15)
  (h2 : initial_questions = 60)
  (h3 : hours_worked = 2) :
  initial_questions - rate * hours_worked = 30 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end sasha_remaining_questions_l49_49101


namespace sum_of_possible_areas_of_square_in_xy_plane_l49_49564

theorem sum_of_possible_areas_of_square_in_xy_plane (x1 x2 x3 : ℝ) (A : ℝ)
    (h1 : x1 = 2 ∨ x1 = 0 ∨ x1 = 18)
    (h2 : x2 = 2 ∨ x2 = 0 ∨ x2 = 18)
    (h3 : x3 = 2 ∨ x3 = 0 ∨ x3 = 18) :
  A = 1168 := sorry

end sum_of_possible_areas_of_square_in_xy_plane_l49_49564


namespace find_x_eq_14_4_l49_49747

theorem find_x_eq_14_4 (x : ℝ) (h : ⌈x⌉ * x = 216) : x = 14.4 :=
by
  sorry

end find_x_eq_14_4_l49_49747


namespace complement_intersection_l49_49195

open Set

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}
noncomputable def A : Set ℕ := {1, 2, 3}
noncomputable def B : Set ℕ := {3, 4, 5}

theorem complement_intersection (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {3, 4, 5}) :
  U \ (A ∩ B) = {1, 2, 4, 5} :=
by
  sorry

end complement_intersection_l49_49195


namespace sixth_distance_l49_49923

theorem sixth_distance (A B C D : Point)
  (dist_AB dist_AC dist_BC dist_AD dist_BD dist_CD : ℝ)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_lengths : (dist_AB = 1 ∧ dist_AC = 1 ∧ dist_BC = 1 ∧ dist_AD = 1) ∨
               (dist_AB = 1 ∧ dist_AC = 1 ∧ dist_BD = 1 ∧ dist_CD = 1) ∨
               (dist_AB = 1 ∧ dist_AD = 1 ∧ dist_BC = 1 ∧ dist_CD = 1) ∨
               (dist_AC = 1 ∧ dist_AD = 1 ∧ dist_BC = 1 ∧ dist_BD = 1) ∨
               (dist_AC = 1 ∧ dist_AD = 1 ∧ dist_BD = 1 ∧ dist_CD = 1) ∨
               (dist_AD = 1 ∧ dist_BC = 1 ∧ dist_BD = 1 ∧ dist_CD = 1))
  (h_one_point_two : dist_AB = 1.2 ∨ dist_AC = 1.2 ∨ dist_BC = 1.2 ∨ dist_AD = 1.2 ∨ dist_BD = 1.2 ∨ dist_CD = 1.2) :
  dist_AB = 1.84 ∨ dist_AB = 0.24 ∨ dist_AB = 1.6 ∨
  dist_AC = 1.84 ∨ dist_AC = 0.24 ∨ dist_AC = 1.6 ∨
  dist_BC = 1.84 ∨ dist_BC = 0.24 ∨ dist_BC = 1.6 ∨
  dist_AD = 1.84 ∨ dist_AD = 0.24 ∨ dist_AD = 1.6 ∨
  dist_BD = 1.84 ∨ dist_BD = 0.24 ∨ dist_BD = 1.6 ∨
  dist_CD = 1.84 ∨ dist_CD = 0.24 ∨ dist_CD = 1.6 :=
sorry

end sixth_distance_l49_49923


namespace race_position_problem_l49_49478

theorem race_position_problem 
  (Cara Bruno Emily David Fiona Alan: ℕ)
  (participants : Finset ℕ)
  (participants_card : participants.card = 12)
  (hCara_Bruno : Cara = Bruno - 3)
  (hEmily_David : Emily = David + 1)
  (hAlan_Bruno : Alan = Bruno + 4)
  (hDavid_Fiona : David = Fiona + 3)
  (hFiona_Cara : Fiona = Cara - 2)
  (hBruno : Bruno = 9)
  (Cara_in_participants : Cara ∈ participants)
  (Bruno_in_participants : Bruno ∈ participants)
  (Emily_in_participants : Emily ∈ participants)
  (David_in_participants : David ∈ participants)
  (Fiona_in_participants : Fiona ∈ participants)
  (Alan_in_participants : Alan ∈ participants)
  : David = 7 := 
sorry

end race_position_problem_l49_49478


namespace Kelly_egg_price_l49_49964

/-- Kelly has 8 chickens, and each chicken lays 3 eggs per day.
Kelly makes $280 in 4 weeks by selling all the eggs.
We want to prove that Kelly sells a dozen eggs for $5. -/
theorem Kelly_egg_price (chickens : ℕ) (eggs_per_day_per_chicken : ℕ) (earnings_in_4_weeks : ℕ)
  (days_in_4_weeks : ℕ) (eggs_per_dozen : ℕ) (price_per_dozen : ℕ) :
  chickens = 8 →
  eggs_per_day_per_chicken = 3 →
  earnings_in_4_weeks = 280 →
  days_in_4_weeks = 28 →
  eggs_per_dozen = 12 →
  price_per_dozen = earnings_in_4_weeks / ((chickens * eggs_per_day_per_chicken * days_in_4_weeks) / eggs_per_dozen) →
  price_per_dozen = 5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end Kelly_egg_price_l49_49964


namespace probability_of_same_color_when_rolling_two_24_sided_dice_l49_49128

-- Defining the conditions
def numSides : ℕ := 24
def purpleSides : ℕ := 5
def blueSides : ℕ := 8
def redSides : ℕ := 10
def goldSides : ℕ := 1

-- Required to use rational numbers for probabilities
def probability (eventSides : ℕ) (totalSides : ℕ) : ℚ := eventSides / totalSides

-- Main theorem statement
theorem probability_of_same_color_when_rolling_two_24_sided_dice :
  probability purpleSides numSides * probability purpleSides numSides +
  probability blueSides numSides * probability blueSides numSides +
  probability redSides numSides * probability redSides numSides +
  probability goldSides numSides * probability goldSides numSides =
  95 / 288 :=
by
  sorry

end probability_of_same_color_when_rolling_two_24_sided_dice_l49_49128


namespace alan_needs_more_wings_l49_49493

theorem alan_needs_more_wings 
  (kevin_wings : ℕ) (kevin_time : ℕ) (alan_rate : ℕ) (target_wings : ℕ) : 
  kevin_wings = 64 → kevin_time = 8 → alan_rate = 5 → target_wings = 3 → 
  (kevin_wings / kevin_time < alan_rate + target_wings) :=
by
  intros kevin_eq time_eq rate_eq target_eq
  sorry

end alan_needs_more_wings_l49_49493


namespace cashier_total_bills_l49_49865

theorem cashier_total_bills
  (total_value : ℕ)
  (num_ten_bills : ℕ)
  (num_twenty_bills : ℕ)
  (h1 : total_value = 330)
  (h2 : num_ten_bills = 27)
  (h3 : num_twenty_bills = 3) :
  num_ten_bills + num_twenty_bills = 30 :=
by
  -- Proof goes here
  sorry

end cashier_total_bills_l49_49865


namespace matrix_N_satisfies_l49_49435

open Matrix

variable {α : Type*} [Fintype α] [DecidableEq α]
variable {R : Type*} [CommRing R]

def N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 2; 1, 2]

theorem matrix_N_satisfies (N : Matrix (Fin 2) (Fin 2) ℝ)
  (h : N^3 - 3•N^2 + 4•N = !![6, 12; 3, 6]) :
  N = !![2, 2; 1, 2] := 
sorry

end matrix_N_satisfies_l49_49435


namespace variance_defect_rate_l49_49241

noncomputable def defect_rate : ℝ := 0.02
noncomputable def number_of_trials : ℕ := 100
noncomputable def variance_binomial (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem variance_defect_rate :
  variance_binomial number_of_trials defect_rate = 1.96 :=
by
  sorry

end variance_defect_rate_l49_49241


namespace sixty_percent_of_N_l49_49838

noncomputable def N : ℝ :=
  let x := (45 : ℝ)
  let frac := (3/4 : ℝ) * (1/3) * (2/5) * (1/2)
  20 * x / frac

theorem sixty_percent_of_N : (0.60 : ℝ) * N = 540 := by
  sorry

end sixty_percent_of_N_l49_49838


namespace batsman_inning_problem_l49_49864

-- Define the problem in Lean 4
theorem batsman_inning_problem (n R : ℕ) (h1 : R = 55 * n) (h2 : R + 110 = 60 * (n + 1)) : n + 1 = 11 := 
  sorry

end batsman_inning_problem_l49_49864


namespace triangle_perimeter_l49_49190

theorem triangle_perimeter (a b c : ℝ) (h1 : a = 2) (h2 : (b-2)^2 + |c-3| = 0) : a + b + c = 7 :=
by
  sorry

end triangle_perimeter_l49_49190


namespace total_number_of_students_is_40_l49_49481

variables (S R : ℕ)

-- Conditions
def students_not_borrowed_any_books := 2
def students_borrowed_1_book := 12
def students_borrowed_2_books := 10
def average_books_per_student := 2

-- Definition of total books borrowed
def total_books_borrowed := (0 * students_not_borrowed_any_books) + (1 * students_borrowed_1_book) + (2 * students_borrowed_2_books) + (3 * R)

-- Expression for total number of students
def total_students := students_not_borrowed_any_books + students_borrowed_1_book + students_borrowed_2_books + R

-- Mathematical statement to prove
theorem total_number_of_students_is_40 (h : total_books_borrowed R / total_students R = average_books_per_student) : total_students R = 40 :=
sorry

end total_number_of_students_is_40_l49_49481


namespace find_sum_of_angles_l49_49489

-- Given conditions
def angleP := 34
def angleQ := 76
def angleR := 28

-- Proposition to prove
theorem find_sum_of_angles (x z : ℝ) (h1 : x + z = 138) : x + z = 138 :=
by
  have angleP := 34
  have angleQ := 76
  have angleR := 28
  exact h1

end find_sum_of_angles_l49_49489


namespace largest_int_less_than_100_rem_5_by_7_l49_49433

theorem largest_int_less_than_100_rem_5_by_7 :
  ∃ k : ℤ, (7 * k + 5 = 96) ∧ ∀ n : ℤ, (7 * n + 5 < 100) → (n ≤ k) :=
sorry

end largest_int_less_than_100_rem_5_by_7_l49_49433


namespace stephanie_remaining_payment_l49_49654

theorem stephanie_remaining_payment:
  let electricity_bill := 60
  let gas_bill := 40
  let water_bill := 40
  let internet_bill := 25
  let gas_paid_fraction := 3/4
  let gas_additional_payment := 5
  let water_paid_fraction := 1/2
  let internet_payment_count := 4
  let internet_payment_each := 5
  let total_bills := electricity_bill + gas_bill + water_bill + internet_bill
  let gas_total_paid := gas_bill * gas_paid_fraction + gas_additional_payment
  let water_total_paid := water_bill * water_paid_fraction
  let internet_total_paid := internet_payment_each * internet_payment_count
  let total_paid := electricity_bill + gas_total_paid + water_total_paid + internet_total_paid
  let remaining_payment := total_bills - total_paid
  in remaining_payment = 30 := by
  sorry

end stephanie_remaining_payment_l49_49654


namespace total_number_of_students_is_40_l49_49482

variables (S R : ℕ)

-- Conditions
def students_not_borrowed_any_books := 2
def students_borrowed_1_book := 12
def students_borrowed_2_books := 10
def average_books_per_student := 2

-- Definition of total books borrowed
def total_books_borrowed := (0 * students_not_borrowed_any_books) + (1 * students_borrowed_1_book) + (2 * students_borrowed_2_books) + (3 * R)

-- Expression for total number of students
def total_students := students_not_borrowed_any_books + students_borrowed_1_book + students_borrowed_2_books + R

-- Mathematical statement to prove
theorem total_number_of_students_is_40 (h : total_books_borrowed R / total_students R = average_books_per_student) : total_students R = 40 :=
sorry

end total_number_of_students_is_40_l49_49482


namespace P_parity_Q_div_by_3_l49_49440

-- Define polynomial P(x)
def P (x p q : ℤ) : ℤ := x*x + p*x + q

-- Define polynomial Q(x)
def Q (x p q : ℤ) : ℤ := x*x*x + p*x + q

-- Part (a) proof statement
theorem P_parity (p q : ℤ) (h1 : Odd p) (h2 : Even q ∨ Odd q) :
  (∀ x : ℤ, Even (P x p q)) ∨ (∀ x : ℤ, Odd (P x p q)) :=
sorry

-- Part (b) proof statement
theorem Q_div_by_3 (p q : ℤ) (h1 : q % 3 = 0) (h2 : p % 3 = 2) :
  ∀ x : ℤ, Q x p q % 3 = 0 :=
sorry

end P_parity_Q_div_by_3_l49_49440


namespace total_fish_count_l49_49239

def number_of_tables : ℕ := 32
def fish_per_table : ℕ := 2
def additional_fish_table : ℕ := 1
def total_fish : ℕ := (number_of_tables * fish_per_table) + additional_fish_table

theorem total_fish_count : total_fish = 65 := by
  sorry

end total_fish_count_l49_49239


namespace triangle_area_l49_49716

theorem triangle_area (a b c : ℕ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 54 := by
    sorry

end triangle_area_l49_49716


namespace cars_on_river_road_l49_49681

variable (B C M : ℕ)

theorem cars_on_river_road
  (h1 : ∃ B C : ℕ, B / C = 1 / 3) -- ratio of buses to cars is 1:3
  (h2 : ∀ B C : ℕ, C = B + 40) -- 40 fewer buses than cars
  (h3 : ∃ B C M : ℕ, B + C + M = 720) -- total number of vehicles is 720
  : C = 60 :=
sorry

end cars_on_river_road_l49_49681


namespace exists_f_m_eq_n_plus_2017_l49_49024

theorem exists_f_m_eq_n_plus_2017 (m : ℕ) (h : m > 0) :
  (∃ f : ℤ → ℤ, ∀ n : ℤ, (f^[m] n = n + 2017)) ↔ (m = 1 ∨ m = 2017) :=
by
  sorry

end exists_f_m_eq_n_plus_2017_l49_49024


namespace vertex_and_maximum_l49_49173

-- Define the quadratic equation
def quadratic (x : ℝ) : ℝ := -3 * x^2 + 6 * x - 9

-- Prove that the vertex of the parabola quadratic is (1, -6) and it is a maximum point
theorem vertex_and_maximum :
  (∃ x y : ℝ, (quadratic x = y) ∧ (x = 1) ∧ (y = -6)) ∧
  (∀ x : ℝ, quadratic x ≤ quadratic 1) :=
sorry

end vertex_and_maximum_l49_49173


namespace solve_linear_system_l49_49764

variable {a b : ℝ}
variables {m n : ℝ}

theorem solve_linear_system
  (h1 : a * 2 - b * 1 = 3)
  (h2 : a * 2 + b * 1 = 5)
  (h3 : a * (m + 2 * n) - 2 * b * n = 6)
  (h4 : a * (m + 2 * n) + 2 * b * n = 10) :
  m = 2 ∧ n = 1 := 
sorry

end solve_linear_system_l49_49764


namespace Alan_shells_l49_49094

theorem Alan_shells (l b a : ℕ) (h1 : l = 36) (h2 : b = l / 3) (h3 : a = 4 * b) : a = 48 :=
by
sorry

end Alan_shells_l49_49094


namespace product_of_roots_l49_49215

theorem product_of_roots (p q r : ℝ) (hp : 3*p^3 - 9*p^2 + 5*p - 15 = 0) 
  (hq : 3*q^3 - 9*q^2 + 5*q - 15 = 0) (hr : 3*r^3 - 9*r^2 + 5*r - 15 = 0) :
  p * q * r = 5 :=
sorry

end product_of_roots_l49_49215


namespace area_of_triangle_l49_49797

noncomputable def triangle_area (AB AC θ : ℝ) : ℝ := 
  0.5 * AB * AC * Real.sin θ

theorem area_of_triangle (AB AC : ℝ) (θ : ℝ) (hAB : AB = 1) (hAC : AC = 2) (hθ : θ = 2 * Real.pi / 3) :
  triangle_area AB AC θ = 3 * Real.sqrt 3 / 14 :=
by
  rw [triangle_area, hAB, hAC, hθ]
  sorry

end area_of_triangle_l49_49797


namespace find_domain_l49_49299

noncomputable def domain (x : ℝ) : Prop :=
  (2 * x + 1 ≥ 0) ∧ (3 - 4 * x ≥ 0)

theorem find_domain :
  {x : ℝ | domain x} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 3/4} :=
by
  sorry

end find_domain_l49_49299


namespace johns_personal_payment_l49_49627

theorem johns_personal_payment 
  (cost_per_hearing_aid : ℕ)
  (num_hearing_aids : ℕ)
  (deductible : ℕ)
  (coverage_percent : ℕ)
  (coverage_limit : ℕ) 
  (total_payment : ℕ)
  (insurance_payment_over_limit : ℕ) : 
  cost_per_hearing_aid = 2500 ∧ 
  num_hearing_aids = 2 ∧ 
  deductible = 500 ∧ 
  coverage_percent = 80 ∧ 
  coverage_limit = 3500 →
  total_payment = cost_per_hearing_aid * num_hearing_aids - deductible →
  insurance_payment_over_limit = max 0 (coverage_percent * total_payment / 100 - coverage_limit) →
  (total_payment - min (coverage_percent * total_payment / 100) coverage_limit + deductible = 1500) :=
by
  intros
  sorry

end johns_personal_payment_l49_49627


namespace find_a_b_a_b_values_l49_49085

/-
Define the matrix M as given in the problem.
Define the constants a and b, and state the condition that proves their correct values such that M_inv = a * M + b * I.
-/

open Matrix

noncomputable def M : Matrix (Fin 2) (Fin 2) ℚ :=
  !![2, 0;
     1, -3]

noncomputable def M_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  !![1/2, 0;
     1/6, -1/3]

theorem find_a_b :
  ∃ (a b : ℚ), (M⁻¹) = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ) :=
sorry

theorem a_b_values :
  (∃ (a b : ℚ), (M⁻¹) = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ)) ∧
  (∃ a b : ℚ, a = 1/6 ∧ b = 1/6) :=
sorry

end find_a_b_a_b_values_l49_49085


namespace find_tangent_line_l49_49927

theorem find_tangent_line (k : ℝ) :
  (∃ k : ℝ, ∀ (x y : ℝ), y = k * (x - 1) + 3 ∧ k^2 + 1 = 1) →
  (∃ k : ℝ, k = 4 / 3 ∧ (k * x - y + 3 - k = 0) ∨ (x = 1)) :=
sorry

end find_tangent_line_l49_49927


namespace min_value_of_expr_l49_49445

theorem min_value_of_expr (a b c : ℝ) (h1 : 0 < a ∧ a ≤ b ∧ b ≤ c) (h2 : a * b * c = 1) :
    (1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (a + c)) + 1 / (c ^ 3 * (a + b))) ≥ 3 / 2 := 
by
  sorry

end min_value_of_expr_l49_49445


namespace loss_percent_l49_49551

theorem loss_percent (CP SP : ℝ) (h₁ : CP = 600) (h₂ : SP = 300) : 
  (CP - SP) / CP * 100 = 50 :=
by
  rw [h₁, h₂]
  norm_num

end loss_percent_l49_49551


namespace range_of_k_l49_49201

theorem range_of_k (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*y^2 = 2 ∧ 
  (∀ e : ℝ, (x^2 / 2 + y^2 / (2 / e) = 1 → (2 / e) > 2))) → 
  0 < k ∧ k < 1 :=
by 
sorry

end range_of_k_l49_49201


namespace parabola_min_value_l49_49098

variable {x0 y0 : ℝ}

def isOnParabola (x0 y0 : ℝ) : Prop := x0^2 = y0

noncomputable def expression (y0 x0 : ℝ) : ℝ :=
  Real.sqrt 2 * y0 + |x0 - y0 - 2|

theorem parabola_min_value :
  isOnParabola x0 y0 → ∃ (m : ℝ), m = (9 / 4 : ℝ) - (Real.sqrt 2 / 4) ∧ 
  ∀ y0 x0, expression y0 x0 ≥ (9 / 4 : ℝ) - (Real.sqrt 2 / 4) := 
by
  sorry

end parabola_min_value_l49_49098


namespace max_coconuts_needed_l49_49975

theorem max_coconuts_needed (goats : ℕ) (coconuts_per_crab : ℕ) (crabs_per_goat : ℕ) 
  (final_goats : ℕ) : 
  goats = 19 ∧ coconuts_per_crab = 3 ∧ crabs_per_goat = 6 →
  ∃ coconuts, coconuts = 342 :=
by
  sorry

end max_coconuts_needed_l49_49975


namespace Dave_ticket_count_l49_49155

variable (T C total : ℕ)

theorem Dave_ticket_count
  (hT1 : T = 12)
  (hC1 : C = 7)
  (hT2 : T = C + 5) :
  total = T + C → total = 19 := by
  sorry

end Dave_ticket_count_l49_49155


namespace equilateral_centered_triangle_l49_49807

noncomputable def is_equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

noncomputable def rotation_120 (A P : Point) : Point := sorry -- Assume this rotates point P around center A by 120 degrees

theorem equilateral_centered_triangle (A1 A2 A3 : Point) (P : ℕ → Point) (Q : ℕ → Point)
  (h1 : ∀ i, A1 = A1)
  (h2 : ∀ i, A2 = A2)
  (h3 : ∀ i, A3 = A3)
  (h4 : ∀ i, P (i + 3) = P i)
  (h5 : P 2020 = P 1)
  (h6 : ∀ i, is_equilateral_triangle (Q i) (P i) (P (i + 1))
    -- Direct equilateral triangle with center at A_i
    -- Utilized the fact that rotation should map correctly
    -- and the equilateral nature is preserved
    ∧ ∀ i, (P (i + 1)) = rotation_120 (A1, P i) ∧ (P (i + 2)) = rotation_120 (A2, P (i + 1))
      ∧ (P (i + 3)) = rotation_120 (A3, P (i + 2))
  : is_equilateral_triangle A1 A2 A3 :=
by
  sorry

end equilateral_centered_triangle_l49_49807


namespace correct_average_of_20_numbers_l49_49658

theorem correct_average_of_20_numbers 
  (incorrect_avg : ℕ) 
  (n : ℕ) 
  (incorrectly_read : ℕ) 
  (correction : ℕ) 
  (a b c d e f g h i j : ℤ) 
  (sum_a_b_c_d_e : ℤ)
  (sum_f_g_h_i_j : ℤ)
  (incorrect_sum : ℤ)
  (correction_sum : ℤ) 
  (corrected_sum : ℤ)
  (correct_avg : ℤ) : 
  incorrect_avg = 35 ∧ 
  n = 20 ∧ 
  incorrectly_read = 5 ∧ 
  correction = 136 ∧ 
  a = 90 ∧ b = 73 ∧ c = 85 ∧ d = -45 ∧ e = 64 ∧ 
  f = 45 ∧ g = 36 ∧ h = 42 ∧ i = -27 ∧ j = 35 ∧ 
  sum_a_b_c_d_e = a + b + c + d + e ∧
  sum_f_g_h_i_j = f + g + h + i + j ∧
  incorrect_sum = incorrect_avg * n ∧ 
  correction_sum = sum_a_b_c_d_e - sum_f_g_h_i_j ∧ 
  corrected_sum = incorrect_sum + correction_sum → correct_avg = corrected_sum / n := 
  by sorry

end correct_average_of_20_numbers_l49_49658


namespace isosceles_triangle_base_length_l49_49665

theorem isosceles_triangle_base_length
  (b : ℕ)
  (congruent_side : ℕ)
  (perimeter : ℕ)
  (h1 : congruent_side = 8)
  (h2 : perimeter = 25)
  (h3 : 2 * congruent_side + b = perimeter) :
  b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l49_49665


namespace simplify_fraction_part1_simplify_fraction_part2_l49_49818

-- Part 1
theorem simplify_fraction_part1 (x : ℝ) (h1 : x ≠ -2) :
  (x^2 / (x + 2)) + ((4 * x + 4) / (x + 2)) = x + 2 :=
sorry

-- Part 2
theorem simplify_fraction_part2 (x : ℝ) (h1 : x ≠ 1) :
  (x^2 / ((x - 1)^2)) / ((1 - 2 * x) / (x - 1) - (x - 1)) = -1 / (x - 1) :=
sorry

end simplify_fraction_part1_simplify_fraction_part2_l49_49818


namespace base_7_to_base_10_equiv_l49_49403

theorem base_7_to_base_10_equiv : 
  ∀ (d2 d1 d0 : ℕ), 
      d2 = 3 → d1 = 4 → d0 = 6 → 
      (d2 * 7^2 + d1 * 7^1 + d0 * 7^0) = 181 := 
by 
  sorry

end base_7_to_base_10_equiv_l49_49403


namespace modulo_4_equiv_2_l49_49344

open Nat

noncomputable def f (n : ℕ) [Fintype (ZMod n)] : ZMod n → ZMod n := sorry

theorem modulo_4_equiv_2 (n : ℕ) [hn : Fact (n > 0)] 
  (f : ZMod n → ZMod n)
  (h1 : ∀ x, f x ≠ x)
  (h2 : ∀ x, f (f x) = x)
  (h3 : ∀ x, f (f (f (x + 1) + 1) + 1) = x) : 
  n % 4 = 2 := 
sorry

end modulo_4_equiv_2_l49_49344


namespace andy_loss_more_likely_than_win_l49_49412

def prob_win_first := 0.30
def prob_lose_first := 0.70

def prob_win_second := 0.50
def prob_lose_second := 0.50

def prob_win_both := prob_win_first * prob_win_second
def prob_lose_both := prob_lose_first * prob_lose_second
def diff_probability := prob_lose_both - prob_win_both
def percentage_more_likely := (diff_probability / prob_win_both) * 100

theorem andy_loss_more_likely_than_win :
  percentage_more_likely = 133.33 := sorry

end andy_loss_more_likely_than_win_l49_49412


namespace coeff_x3y3_in_expansion_l49_49162

section
variable {R : Type*} [CommRing R]
open BigOperators

noncomputable def coefficient_x3y3 (x y : R) : R :=
  (range(6)).sum (λ r, (binom 5 r) * ((x^2 + x)^r) * (y^(5 - r)))

theorem coeff_x3y3_in_expansion : coefficient_x3y3 x y = 20 :=
sorry
end

end coeff_x3y3_in_expansion_l49_49162


namespace log_product_solution_l49_49543

theorem log_product_solution (x : ℝ) (hx : 0 < x) : 
  (Real.log x / Real.log 2) * (Real.log x / Real.log 5) = Real.log 10 / Real.log 2 ↔ 
  x = 2 ^ Real.sqrt (6 * Real.log 2) :=
sorry

end log_product_solution_l49_49543


namespace total_cost_of_repair_l49_49352

theorem total_cost_of_repair (hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) (H1 : hours = 2) (H2 : hourly_rate = 75) (H3 : part_cost = 150) :
  hours * hourly_rate + part_cost = 300 := 
by
  sorry

end total_cost_of_repair_l49_49352


namespace ellipse_area_is_12pi_l49_49287

noncomputable def ellipse_area : ℝ :=
  (1 / 2) * ∫ (t : ℝ) in 0..(2 * Real.pi), (4 * Real.cos t * (3 * Real.cos t) - 3 * Real.sin t * (-4 * Real.sin t))

theorem ellipse_area_is_12pi :
  ellipse_area = 12 * Real.pi := by
sorry

end ellipse_area_is_12pi_l49_49287


namespace downstream_speed_l49_49560

theorem downstream_speed 
  (upstream_speed : ℕ) 
  (still_water_speed : ℕ) 
  (hm_upstream : upstream_speed = 27) 
  (hm_still_water : still_water_speed = 31) 
  : (still_water_speed + (still_water_speed - upstream_speed)) = 35 :=
by
  sorry

end downstream_speed_l49_49560


namespace triangle_inradius_exradius_l49_49623

-- Define the properties of the triangle
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Define the inradius
def inradius (a b c : ℝ) (r : ℝ) : Prop :=
  r = (a + b - c) / 2

-- Define the exradius
def exradius (a b c : ℝ) (rc : ℝ) : Prop :=
  rc = (a + b + c) / 2

-- Formalize the Lean statement for the given proof problem
theorem triangle_inradius_exradius (a b c r rc: ℝ) 
  (h_triangle: right_triangle a b c) : 
  inradius a b c r ∧ exradius a b c rc :=
by
  sorry

end triangle_inradius_exradius_l49_49623


namespace largest_int_less_than_100_with_remainder_5_l49_49432

theorem largest_int_less_than_100_with_remainder_5 (x : ℤ) (n : ℤ) (h₁ : x = 7 * n + 5) (h₂ : x < 100) : 
  x = 96 := by
  sorry

end largest_int_less_than_100_with_remainder_5_l49_49432


namespace part1_part2_l49_49552

open Set

variable {α : Type*} [LinearOrderedField α]

def A : Set α := { x | abs (x - 1) ≤ 1 }
def B (a : α) : Set α := { x | x ≥ a }

theorem part1 {x : α} : x ∈ (A ∩ B 1) ↔ 1 ≤ x ∧ x ≤ 2 := by
  sorry

theorem part2 {a : α} : (A ⊆ B a) ↔ a ≤ 0 := by
  sorry

end part1_part2_l49_49552


namespace vertex_coordinates_l49_49113

-- Define the given parabola equation
def parabola (x : ℝ) : ℝ := (x + 3) ^ 2 - 1

-- Define the statement for the coordinates of the vertex of the parabola
theorem vertex_coordinates : ∃ (h k : ℝ), (∀ x : ℝ, parabola x = (x + 3) ^ 2 - 1) ∧ h = -3 ∧ k = -1 := 
  sorry

end vertex_coordinates_l49_49113


namespace Thabo_books_ratio_l49_49999

variable (P_f P_nf H_nf : ℕ)

theorem Thabo_books_ratio :
  P_f + P_nf + H_nf = 220 →
  H_nf = 40 →
  P_nf = H_nf + 20 →
  P_f / P_nf = 2 :=
by sorry

end Thabo_books_ratio_l49_49999


namespace determine_values_of_a_and_c_l49_49574

-- Definition of the projection matrix P
def P_matrix (a c : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, 20/36], ![c, 16/36]]

-- Definition to check if a matrix is a projection matrix
def is_projection_matrix (P : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  P * P = P

-- The values we want to prove are the solution
def a_value : ℝ := 1/27
def c_value : ℝ := 5/27

-- The proof statement: given P_matrix is a projection matrix, 
-- a and c must be the specific values we found.
theorem determine_values_of_a_and_c :
  is_projection_matrix (P_matrix a_value c_value) :=
sorry

end determine_values_of_a_and_c_l49_49574


namespace factorize_expression_l49_49891

variable (a b c : ℝ)

theorem factorize_expression : 
  (a - 2 * b) * (a - 2 * b - 4) + 4 - c ^ 2 = ((a - 2 * b) - 2 + c) * ((a - 2 * b) - 2 - c) := 
by
  sorry

end factorize_expression_l49_49891


namespace number_properties_l49_49528

theorem number_properties (a b x : ℝ) 
  (h1 : a + b = 40) 
  (h2 : a * b = 375) 
  (h3 : a - b = x) : 
  (a = 25 ∧ b = 15 ∧ x = 10) ∨ (a = 15 ∧ b = 25 ∧ x = 10) :=
by
  sorry

end number_properties_l49_49528


namespace num_people_fewer_than_7_cards_l49_49777

theorem num_people_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      remainder := total_cards % total_people
  in total_people - remainder = 3 :=
by
  sorry

end num_people_fewer_than_7_cards_l49_49777


namespace area_of_right_triangle_l49_49713

theorem area_of_right_triangle
    (a b c : ℝ)
    (h₀ : a = 9)
    (h₁ : b = 12)
    (h₂ : c = 15)
    (right_triangle : a^2 + b^2 = c^2) :
    (1 / 2) * a * b = 54 := by
  sorry

end area_of_right_triangle_l49_49713


namespace units_digit_7_pow_6_pow_5_l49_49901

def units_digit_of_power (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_6_pow_5 :
  units_digit_of_power (7 ^ (6 ^ 5)) = 1 :=
by
  -- Insert proof steps here
  sorry

end units_digit_7_pow_6_pow_5_l49_49901


namespace cone_fits_in_cube_l49_49639

noncomputable def height_cone : ℝ := 15
noncomputable def diameter_cone_base : ℝ := 8
noncomputable def side_length_cube : ℝ := 15
noncomputable def volume_cube : ℝ := side_length_cube ^ 3

theorem cone_fits_in_cube :
  (height_cone = 15) →
  (diameter_cone_base = 8) →
  (height_cone ≤ side_length_cube ∧ diameter_cone_base ≤ side_length_cube) →
  volume_cube = 3375 := by
  intros h_cone d_base fits
  sorry

end cone_fits_in_cube_l49_49639


namespace cucumber_kinds_l49_49647

theorem cucumber_kinds (x : ℕ) :
  (3 * 5) + (4 * x) + 30 + 85 = 150 → x = 5 :=
by
  intros h
  -- h : 15 + 4 * x + 30 + 85 = 150 

  -- Proof would go here
  sorry

end cucumber_kinds_l49_49647


namespace units_digit_of_7_pow_6_pow_5_l49_49904

theorem units_digit_of_7_pow_6_pow_5 : (7^(6^5)) % 10 = 1 := by
  -- Proof goes here
  sorry

end units_digit_of_7_pow_6_pow_5_l49_49904


namespace train_speed_l49_49276

/-- 
Given:
- Length of train L is 390 meters (0.39 km)
- Speed of man Vm is 2 km/h
- Time to cross man T is 52 seconds

Prove:
- The speed of the train Vt is 25 km/h
--/
theorem train_speed 
  (L : ℝ) (Vm : ℝ) (T : ℝ) (Vt : ℝ)
  (h1 : L = 0.39) 
  (h2 : Vm = 2) 
  (h3 : T = 52 / 3600) 
  (h4 : Vt + Vm = L / T) :
  Vt = 25 :=
by sorry

end train_speed_l49_49276


namespace rectangle_decomposition_l49_49703

theorem rectangle_decomposition (m n k : ℕ) : ((k ∣ m) ∨ (k ∣ n)) ↔ (∃ P : ℕ, m * n = P * k) :=
by
  sorry

end rectangle_decomposition_l49_49703


namespace at_least_one_misses_l49_49479

-- Definitions for the given conditions
variables {p q : Prop}

-- Lean 4 statement proving the equivalence
theorem at_least_one_misses (hp : p → false) (hq : q → false) : (¬p ∨ ¬q) :=
by sorry

end at_least_one_misses_l49_49479


namespace find_x_plus_y_l49_49193

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2010) (h2 : x + 2010 * Real.cos y = 2009) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2009 + Real.pi / 2 := 
by
  sorry

end find_x_plus_y_l49_49193


namespace denominator_of_expression_l49_49689

theorem denominator_of_expression (x : ℝ) (h : (1 / x) ^ 1 = 0.25) : x = 4 := by
  sorry

end denominator_of_expression_l49_49689


namespace find_number_l49_49380

theorem find_number (N : ℕ) :
  let sum := 555 + 445
  let difference := 555 - 445
  let divisor := sum
  let quotient := 2 * difference
  let remainder := 70
  N = divisor * quotient + remainder -> N = 220070 := 
by
  intro h
  sorry

end find_number_l49_49380


namespace find_first_number_in_second_set_l49_49110

theorem find_first_number_in_second_set: 
  ∃ x: ℕ, (20 + 40 + 60) / 3 = (x + 80 + 15) / 3 + 5 ∧ x = 10 :=
by
  sorry

end find_first_number_in_second_set_l49_49110


namespace solution_set_absolute_value_sum_eq_three_l49_49929

theorem solution_set_absolute_value_sum_eq_three (m n : ℝ) (h : ∀ x : ℝ, (|2 * x - 3| ≤ 1) ↔ (m ≤ x ∧ x ≤ n)) : m + n = 3 :=
sorry

end solution_set_absolute_value_sum_eq_three_l49_49929


namespace cube_cut_edges_l49_49558

theorem cube_cut_edges (original_edges new_edges_per_vertex vertices : ℕ) (h1 : original_edges = 12) (h2 : new_edges_per_vertex = 6) (h3 : vertices = 8) :
  original_edges + new_edges_per_vertex * vertices = 60 :=
by
  sorry

end cube_cut_edges_l49_49558


namespace tan_gt_neg_one_solution_set_l49_49834

def tangent_periodic_solution_set (x : ℝ) : Prop :=
  ∃ k : ℤ, k * Real.pi - Real.pi / 4 < x ∧ x < k * Real.pi + Real.pi / 2

theorem tan_gt_neg_one_solution_set (x : ℝ) :
  tangent_periodic_solution_set x ↔ Real.tan x > -1 :=
by
  sorry

end tan_gt_neg_one_solution_set_l49_49834


namespace division_result_is_correct_l49_49738

def division_result : ℚ := 132 / 6 / 3

theorem division_result_is_correct : division_result = 22 / 3 :=
by
  -- here, we would include the proof steps, but for now, we'll put sorry
  sorry

end division_result_is_correct_l49_49738


namespace solve_for_a_l49_49041

variable (a b x : ℝ)

theorem solve_for_a (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * x^3) (h3 : a - b = 3 * x) :
  a = 3 * x := sorry

end solve_for_a_l49_49041


namespace arithmetic_sequence_general_formula_geometric_sequence_sum_first_n_terms_l49_49182

noncomputable def arithmetic_sequence (a n d : ℝ) : ℝ := 
  a + (n - 1) * d

noncomputable def geometric_sequence_sum (b1 r n : ℝ) : ℝ := 
  b1 * (1 - r^n) / (1 - r)

theorem arithmetic_sequence_general_formula (a1 d : ℝ) (h1 : a1 + 2 * d = 2) (h2 : 3 * a1 + 3 * d = 9 / 2) : 
  ∀ n, arithmetic_sequence a1 n d = (n + 1) / 2 :=
by 
  sorry

theorem geometric_sequence_sum_first_n_terms (a1 d b1 b4 : ℝ) (h1 : a1 + 2 * d = 2) (h2 : 3 * a1 + 3 * d = 9 / 2) 
  (h3 : b1 = a1) (h4 : b4 = arithmetic_sequence a1 15 d) (h5 : b4 = 8) :
  ∀ n, geometric_sequence_sum b1 2 n = 2^n - 1 :=
by 
  sorry

end arithmetic_sequence_general_formula_geometric_sequence_sum_first_n_terms_l49_49182


namespace triangle_area_l49_49723

theorem triangle_area (a b c : ℕ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) (h₄ : a^2 + b^2 = c^2) : 
  (1 / 2 : ℝ) * a * b = 54 := 
by
  rw [h₁, h₂, h₃]
  -- The proof goes here
  sorry

end triangle_area_l49_49723


namespace positive_difference_l49_49540

noncomputable def calculate_diff : ℕ :=
  let first_term := (8^2 - 8^2) / 8
  let second_term := (8^2 * 8^2) / 8
  second_term - first_term

theorem positive_difference : calculate_diff = 512 := by
  sorry

end positive_difference_l49_49540


namespace units_digit_7_pow_6_pow_5_l49_49913

theorem units_digit_7_pow_6_pow_5 : (7 ^ (6 ^ 5)) % 10 = 1 :=
by
  -- Using the cyclic pattern of the units digits of powers of 7: 7, 9, 3, 1
  have h1 : 7 % 10 = 7, by norm_num,
  have h2 : (7 ^ 2) % 10 = 9, by norm_num,
  have h3 : (7 ^ 3) % 10 = 3, by norm_num,
  have h4 : (7 ^ 4) % 10 = 1, by norm_num,

  -- Calculate 6 ^ 5 and the modular position
  have h6_5 : (6 ^ 5) % 4 = 0, by norm_num,

  -- Therefore, 7 ^ (6 ^ 5) % 10 = 7 ^ 0 % 10 because the cycle is 4
  have h_final : (7 ^ (6 ^ 5 % 4)) % 10 = (7 ^ 0) % 10, by rw h6_5,
  have h_zero : (7 ^ 0) % 10 = 1, by norm_num,

  rw h_final,
  exact h_zero,

end units_digit_7_pow_6_pow_5_l49_49913


namespace max_chain_triangles_l49_49291

theorem max_chain_triangles (n : ℕ) (h : n > 0) : 
  ∃ k, k = n^2 - n + 1 := 
sorry

end max_chain_triangles_l49_49291


namespace pascal_sixth_element_row_20_l49_49254

theorem pascal_sixth_element_row_20 : (Nat.choose 20 5) = 7752 := 
  by
  sorry

end pascal_sixth_element_row_20_l49_49254


namespace max_ratio_of_three_digit_to_sum_l49_49016

theorem max_ratio_of_three_digit_to_sum (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 9)
  (hb : 0 ≤ b ∧ b ≤ 9)
  (hc : 0 ≤ c ∧ c ≤ 9) :
  (100 * a + 10 * b + c) / (a + b + c) ≤ 100 :=
by sorry

end max_ratio_of_three_digit_to_sum_l49_49016


namespace curve_is_circle_l49_49749

theorem curve_is_circle : ∀ (θ : ℝ), ∃ r : ℝ, r = 3 * Real.cos θ → ∃ (x y : ℝ), x^2 + y^2 = (3/2)^2 :=
by
  intro θ
  use 3 * Real.cos θ
  sorry

end curve_is_circle_l49_49749


namespace count_distinct_reals_a_with_integer_roots_l49_49028

-- Define the quadratic equation with its roots and conditions
theorem count_distinct_reals_a_with_integer_roots :
  ∃ (a_vals : Finset ℝ), a_vals.card = 6 ∧
    (∀ a ∈ a_vals, ∃ r s : ℤ, 
      (r + s : ℝ) = -a ∧ (r * s : ℝ) = 9 * a) :=
by
  sorry

end count_distinct_reals_a_with_integer_roots_l49_49028


namespace rectangle_dimensions_l49_49679

variable (w l : ℝ)
variable (h1 : l = w + 15)
variable (h2 : 2 * w + 2 * l = 150)

theorem rectangle_dimensions :
  w = 30 ∧ l = 45 :=
by
  sorry

end rectangle_dimensions_l49_49679


namespace total_words_in_poem_l49_49994

theorem total_words_in_poem 
  (stanzas : ℕ) 
  (lines_per_stanza : ℕ) 
  (words_per_line : ℕ) 
  (h_stanzas : stanzas = 20) 
  (h_lines_per_stanza : lines_per_stanza = 10) 
  (h_words_per_line : words_per_line = 8) : 
  stanzas * lines_per_stanza * words_per_line = 1600 := 
sorry

end total_words_in_poem_l49_49994


namespace greatest_temp_diff_on_tuesday_l49_49674

def highest_temp_mon : ℝ := 5
def lowest_temp_mon : ℝ := 2
def highest_temp_tue : ℝ := 4
def lowest_temp_tue : ℝ := -1
def highest_temp_wed : ℝ := 0
def lowest_temp_wed : ℝ := -4

def temp_diff (highest lowest : ℝ) : ℝ :=
  highest - lowest

theorem greatest_temp_diff_on_tuesday : temp_diff highest_temp_tue lowest_temp_tue 
  > temp_diff highest_temp_mon lowest_temp_mon 
  ∧ temp_diff highest_temp_tue lowest_temp_tue 
  > temp_diff highest_temp_wed lowest_temp_wed := 
by
  sorry

end greatest_temp_diff_on_tuesday_l49_49674


namespace total_fish_l49_49240

theorem total_fish (n : ℕ) (t : ℕ) (f : ℕ) :
  n = 32 ∧ t = 1 ∧ f = 31 ∧ ∃ (fish_count_table : ℕ → ℕ), 
  (fish_count_table(t) = 3) ∧ (∀ i, 1 ≤ i ∧ i <= f → fish_count_table(i + t) = 2) → 
  (∑ i in finset.range (t + f), fish_count_table (i + 1)) = 65 :=
by
  sorry

end total_fish_l49_49240


namespace rationalized_value_l49_49386

open Real

theorem rationalized_value :
  let A := 2
  let B := 4
  let C := 3
  A + B + C = 9 :=
by
  -- By rationalizing the denominator of 4 / (3 * (8 ^ (1/3))), it can be shown that:
  -- rationalized form is (2 * (4 ^ (1/3))) / 3
  -- thus A = 2, B = 4, C = 3 and 2 + 4 + 3 = 9
  sorry

end rationalized_value_l49_49386


namespace range_of_a_l49_49611

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → (2 * a + 1)^x > (2 * a + 1)^y) → (-1/2 < a ∧ a < 0) :=
by
  sorry

end range_of_a_l49_49611


namespace distance_between_poles_l49_49563

theorem distance_between_poles (length width : ℝ) (num_poles : ℕ) (h_length : length = 90)
  (h_width : width = 40) (h_num_poles : num_poles = 52) : 
  (2 * (length + width)) / (num_poles - 1) = 5.098 := 
by 
  -- Sorry to skip the proof
  sorry

end distance_between_poles_l49_49563


namespace find_initial_number_l49_49151

theorem find_initial_number (x : ℝ) (h : x + 12.808 - 47.80600000000004 = 3854.002) : x = 3889 := by
  sorry

end find_initial_number_l49_49151


namespace maximum_negative_roots_l49_49766

theorem maximum_negative_roots (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
    (discriminant1 : b^2 - 4 * a * c ≥ 0)
    (discriminant2 : c^2 - 4 * b * a ≥ 0)
    (discriminant3 : a^2 - 4 * c * b ≥ 0) :
    ∃ n : ℕ, n ≤ 2 ∧ ∀ x ∈ {x | a * x^2 + b * x + c = 0 ∨ b * x^2 + c * x + a = 0 ∨ c * x^2 + a * x + b = 0}, x < 0 ↔ n = 2 := 
sorry

end maximum_negative_roots_l49_49766


namespace part1_part2_l49_49638

-- Part 1: Define the sequence and sum function, then state the problem.
def a_1 : ℚ := 3 / 2
def d : ℚ := 1

def S_n (n : ℕ) : ℚ :=
  n * a_1 + (n * (n - 1) / 2) * d

theorem part1 (k : ℕ) (h : S_n (k^2) = (S_n k)^2) : k = 4 := sorry

-- Part 2: Define the general sequence and state the problem.
def arith_seq (a_1 : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a_1 + (n - 1) * d

def S_n_general (a_1 : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n * a_1) + (n * (n - 1) / 2) * d

theorem part2 (a_1 : ℚ) (d : ℚ) :
  (∀ k : ℕ, S_n_general a_1 d (k^2) = (S_n_general a_1 d k)^2) ↔
  (a_1 = 0 ∧ d = 0) ∨
  (a_1 = 1 ∧ d = 0) ∨
  (a_1 = 1 ∧ d = 2) := sorry

end part1_part2_l49_49638


namespace tasty_residue_count_2016_l49_49015

def tasty_residue (n : ℕ) (a : ℕ) : Prop :=
  1 < a ∧ a < n ∧ ∃ m : ℕ, m > 1 ∧ a ^ m ≡ a [MOD n]

theorem tasty_residue_count_2016 : 
  (∃ count : ℕ, count = 831 ∧ ∀ a : ℕ, 1 < a ∧ a < 2016 ↔ tasty_residue 2016 a) :=
sorry

end tasty_residue_count_2016_l49_49015


namespace diamond_two_three_l49_49794

def diamond (a b : ℕ) : ℕ := a * b^2 - b + 1

theorem diamond_two_three : diamond 2 3 = 16 := by
  sorry

end diamond_two_three_l49_49794


namespace arctan_sum_pi_over_four_l49_49338

theorem arctan_sum_pi_over_four (a b c : ℝ) (C : ℝ) (h : Real.sin C = c / (a + b + c)) :
  Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = Real.pi / 4 :=
sorry

end arctan_sum_pi_over_four_l49_49338


namespace range_of_expr_l49_49998

noncomputable def expr (x y : ℝ) : ℝ := (x + 2 * y + 3) / (x + 1)

theorem range_of_expr : 
  (∀ x y : ℝ, x ≥ 0 → y ≥ x → 4 * x + 3 * y ≤ 12 → 3 ≤ expr x y ∧ expr x y ≤ 11) :=
by
  sorry

end range_of_expr_l49_49998


namespace nums_between_2000_and_3000_div_by_360_l49_49936

theorem nums_between_2000_and_3000_div_by_360 : 
  (∃ n1 n2 n3 : ℕ, 2000 ≤ n1 ∧ n1 ≤ 3000 ∧ 360 ∣ n1 ∧
                   2000 ≤ n2 ∧ n2 ≤ 3000 ∧ 360 ∣ n2 ∧
                   2000 ≤ n3 ∧ n3 ≤ 3000 ∧ 360 ∣ n3 ∧
                   n1 ≠ n2 ∧ n1 ≠ n3 ∧ n2 ≠ n3 ∧
                   ∀ m : ℕ, (2000 ≤ m ∧ m ≤ 3000 ∧ 360 ∣ m → m = n1 ∨ m = n2 ∨ m = n3)) := 
begin
  sorry
end

end nums_between_2000_and_3000_div_by_360_l49_49936


namespace average_temperature_l49_49005

def highTemps : List ℚ := [51, 60, 56, 55, 48, 63, 59]
def lowTemps : List ℚ := [42, 50, 44, 43, 41, 46, 45]

def dailyAverage (high low : ℚ) : ℚ :=
  (high + low) / 2

def averageOfAverages (tempsHigh tempsLow : List ℚ) : ℚ :=
  (List.sum (List.zipWith dailyAverage tempsHigh tempsLow)) / tempsHigh.length

theorem average_temperature :
  averageOfAverages highTemps lowTemps = 50.2 :=
  sorry

end average_temperature_l49_49005


namespace percentage_decrease_increase_l49_49987

theorem percentage_decrease_increase (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 0.75 ↔ x = 50 :=
by
  sorry

end percentage_decrease_increase_l49_49987


namespace xiaoying_school_trip_l49_49545

theorem xiaoying_school_trip :
  ∃ (x y : ℝ), 
    (1200 / 1000) = (3 / 60) * x + (5 / 60) * y ∧ 
    x + y = 16 :=
by
  sorry

end xiaoying_school_trip_l49_49545


namespace minimum_detectors_203_l49_49121

def minimum_detectors (length : ℕ) : ℕ :=
  length / 3 * 2 -- This models the generalization for 1 × (3k + 2)

theorem minimum_detectors_203 : minimum_detectors 203 = 134 :=
by
  -- Length is 203, k = 67 which follows from the floor division
  -- Therefore, minimum detectors = 2 * 67 = 134
  sorry

end minimum_detectors_203_l49_49121


namespace height_of_spruce_tree_l49_49492

theorem height_of_spruce_tree (t : ℚ) (h1 : t = 25 / 64) :
  (∃ s : ℚ, s = 3 / (1 - t) ∧ s = 64 / 13) :=
by
  sorry

end height_of_spruce_tree_l49_49492


namespace units_digit_7_pow_6_pow_5_l49_49902

def units_digit_of_power (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_6_pow_5 :
  units_digit_of_power (7 ^ (6 ^ 5)) = 1 :=
by
  -- Insert proof steps here
  sorry

end units_digit_7_pow_6_pow_5_l49_49902


namespace pascal_sixth_element_row_20_l49_49255

theorem pascal_sixth_element_row_20 : (Nat.choose 20 5) = 7752 := 
  by
  sorry

end pascal_sixth_element_row_20_l49_49255


namespace four_at_three_equals_thirty_l49_49943

def custom_operation (a b : ℕ) : ℕ :=
  3 * a^2 - 2 * b^2

theorem four_at_three_equals_thirty : custom_operation 4 3 = 30 :=
by
  sorry

end four_at_three_equals_thirty_l49_49943


namespace decrease_in_area_of_equilateral_triangle_l49_49411

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

theorem decrease_in_area_of_equilateral_triangle :
  (equilateral_triangle_area 20 - equilateral_triangle_area 14) = 51 * Real.sqrt 3 := by
  sorry

end decrease_in_area_of_equilateral_triangle_l49_49411


namespace max_tetrahedron_in_cube_l49_49134

open Real

noncomputable def cube_edge_length : ℝ := 6
noncomputable def max_tetrahedron_edge_length (a : ℝ) : Prop :=
  ∃ x : ℝ, x = 2 * sqrt 6 ∧ 
          (∃ R : ℝ, R = (a * sqrt 3) / 2 ∧ x / sqrt (2 / 3) = 4 * R / 3)

theorem max_tetrahedron_in_cube : max_tetrahedron_edge_length cube_edge_length :=
sorry

end max_tetrahedron_in_cube_l49_49134


namespace die_prob_tangent_die_prob_isosceles_l49_49407

-- Defining the problem context
def die_faces := {1, 2, 3, 4, 5, 6}

-- Defining the probability calculation
def prob_tangent (a b : ℕ) : ℝ :=
  if a^2 + b^2 = 25 then 1 else 0

def valid_isosceles (a b : ℕ) :=
  (a = b ∨ a = 5 ∨ b = 5)

-- Main theorem statements
theorem die_prob_tangent :
  let outcomes := ∑ a in die_faces, ∑ b in die_faces, 1 in
  let valid_tangents := ∑ a in die_faces, ∑ b in die_faces, prob_tangent a b in
  valid_tangents / outcomes = (1 : ℝ) / 18 :=
sorry

theorem die_prob_isosceles :
  let outcomes := ∑ a in die_faces, ∑ b in die_faces, 1 in
  let valid_isosceles := ∑ a in die_faces, ∑ b in die_faces, if valid_isosceles a b then 1 else 0 in
  valid_isosceles / outcomes = (7 : ℝ) / 18 :=
sorry

end die_prob_tangent_die_prob_isosceles_l49_49407


namespace count_primes_between_30_and_50_l49_49463

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_30_and_50 : List ℕ :=
  [31, 37, 41, 43, 47]

theorem count_primes_between_30_and_50 : 
  (primes_between_30_and_50.filter is_prime).length = 5 :=
by
  sorry

end count_primes_between_30_and_50_l49_49463


namespace smallest_positive_period_and_symmetry_l49_49449

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x + (7 * Real.pi / 4)) + 
  Real.cos (x - (3 * Real.pi / 4))

theorem smallest_positive_period_and_symmetry :
  (∃ T > 0, T = 2 * Real.pi ∧ ∀ x, f (x + T) = f x) ∧ 
  (∃ a, a = - (Real.pi / 4) ∧ ∀ x, f (2 * a - x) = f x) :=
by
  sorry

end smallest_positive_period_and_symmetry_l49_49449


namespace bill_soaking_time_l49_49158

theorem bill_soaking_time 
  (G M : ℕ) 
  (h₁ : M = G + 7) 
  (h₂ : 3 * G + M = 19) : 
  G = 3 := 
by {
  sorry
}

end bill_soaking_time_l49_49158


namespace people_with_fewer_than_7_cards_l49_49771

-- Definitions based on conditions
def cards_total : ℕ := 60
def people_total : ℕ := 9

-- Statement of the theorem
theorem people_with_fewer_than_7_cards : 
  ∃ (x : ℕ), x = 3 ∧ (cards_total % people_total = 0 ∨ cards_total % people_total < people_total) :=
by
  sorry

end people_with_fewer_than_7_cards_l49_49771


namespace xyz_value_l49_49187

variable (x y z : ℝ)

theorem xyz_value :
  (x + y + z) * (x*y + x*z + y*z) = 36 →
  x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 24 →
  x * y * z = 4 :=
by
  intros h1 h2
  sorry

end xyz_value_l49_49187


namespace hearty_beads_count_l49_49455

theorem hearty_beads_count :
  let blue_packages := 3
  let red_packages := 5
  let beads_per_package := 40
  let total_beads := blue_packages * beads_per_package + red_packages * beads_per_package
  total_beads = 320 :=
by
  let blue_packages := 3
  let red_packages := 5
  let beads_per_package := 40
  let total_beads := blue_packages * beads_per_package + red_packages * beads_per_package
  show total_beads = 320
  sorry

end hearty_beads_count_l49_49455


namespace bruno_coconuts_per_trip_is_8_l49_49004

-- Definitions related to the problem conditions
def total_coconuts : ℕ := 144
def barbie_coconuts_per_trip : ℕ := 4
def trips : ℕ := 12
def bruno_coconuts_per_trip : ℕ := total_coconuts - (barbie_coconuts_per_trip * trips)

-- The main theorem stating the question and the answer
theorem bruno_coconuts_per_trip_is_8 : bruno_coconuts_per_trip / trips = 8 :=
by
  sorry

end bruno_coconuts_per_trip_is_8_l49_49004


namespace not_divisible_by_15_l49_49675

theorem not_divisible_by_15 (a : ℤ) : ¬ (15 ∣ (a^2 + a + 2)) :=
by
  sorry

end not_divisible_by_15_l49_49675


namespace tony_combined_lift_weight_l49_49843

theorem tony_combined_lift_weight :
  let curl_weight := 90
  let military_press_weight := 2 * curl_weight
  let squat_weight := 5 * military_press_weight
  let bench_press_weight := 1.5 * military_press_weight
  squat_weight + bench_press_weight = 1170 :=
by
  sorry

end tony_combined_lift_weight_l49_49843


namespace net_gain_loss_l49_49978

-- Definitions of the initial conditions
structure InitialState :=
  (cash_x : ℕ) (painting_value : ℕ) (cash_y : ℕ)

-- Definitions of transactions
structure Transaction :=
  (sell_price : ℕ) (commission_rate : ℕ)

def apply_transaction (initial_cash : ℕ) (tr : Transaction) : ℕ :=
  initial_cash + (tr.sell_price - (tr.sell_price * tr.commission_rate / 100))

def revert_transaction (initial_cash : ℕ) (tr : Transaction) : ℕ :=
  initial_cash - tr.sell_price + (tr.sell_price * tr.commission_rate / 100)

def compute_final_cash (initial_states : InitialState) (trans1 : Transaction) (trans2 : Transaction) : ℕ :=
  let cash_x_after_first := apply_transaction initial_states.cash_x trans1
  let cash_y_after_first := initial_states.cash_y - trans1.sell_price
  let cash_x_after_second := revert_transaction cash_x_after_first trans2
  let cash_y_after_second := cash_y_after_first + (trans2.sell_price - (trans2.sell_price * trans2.commission_rate / 100))
  cash_x_after_second - initial_states.cash_x + (cash_y_after_second - initial_states.cash_y)

-- Statement of the theorem
theorem net_gain_loss (initial_states : InitialState) (trans1 : Transaction) (trans2 : Transaction)
  (h1 : initial_states.cash_x = 15000)
  (h2 : initial_states.painting_value = 15000)
  (h3 : initial_states.cash_y = 18000)
  (h4 : trans1.sell_price = 20000)
  (h5 : trans1.commission_rate = 5)
  (h6 : trans2.sell_price = 14000)
  (h7 : trans2.commission_rate = 5) : 
  compute_final_cash initial_states trans1 trans2 = 5000 - 6700 :=
sorry

end net_gain_loss_l49_49978


namespace dan_blue_marbles_l49_49883

variable (m d : ℕ)
variable (h1 : m = 2 * d)
variable (h2 : m = 10)

theorem dan_blue_marbles : d = 5 :=
by
  sorry

end dan_blue_marbles_l49_49883


namespace range_of_a_l49_49500

theorem range_of_a (x y : ℝ) (a : ℝ) :
  (0 < x ∧ x ≤ 2) ∧ (0 < y ∧ y ≤ 2) ∧ (x * y = 2) ∧ (6 - 2 * x - y ≥ a * (2 - x) * (4 - y)) →
  a ≤ 1 :=
by sorry

end range_of_a_l49_49500


namespace old_edition_pages_l49_49149

theorem old_edition_pages (x : ℕ) 
  (h₁ : 2 * x - 230 = 450) : x = 340 := 
by sorry

end old_edition_pages_l49_49149


namespace sues_answer_l49_49406

theorem sues_answer (x : ℕ) (hx : x = 6) : 
  let b := 2 * (x + 1)
  let s := 2 * (b - 1)
  s = 26 :=
by
  sorry

end sues_answer_l49_49406


namespace approx_equal_e_l49_49536
noncomputable def a : ℝ := 69.28
noncomputable def b : ℝ := 0.004
noncomputable def c : ℝ := 0.03
noncomputable def d : ℝ := a * b
noncomputable def e : ℝ := d / c

theorem approx_equal_e : abs (e - 9.24) < 0.01 :=
by
  sorry

end approx_equal_e_l49_49536


namespace sum_of_polynomial_roots_l49_49967

theorem sum_of_polynomial_roots:
  ∀ (a b : ℝ),
  (a^2 - 5 * a + 6 = 0) ∧ (b^2 - 5 * b + 6 = 0) →
  a^3 + a^4 * b^2 + a^2 * b^4 + b^3 + a * b^3 + b * a^3 = 683 := by
  intros a b h
  sorry

end sum_of_polynomial_roots_l49_49967


namespace number_of_routes_l49_49416

variable {City : Type}
variable (A B C D E : City)
variable (AB_N AB_S AD AE BC BD CD DE : City → City → Prop)
  
theorem number_of_routes 
  (hAB_N : AB_N A B) (hAB_S : AB_S A B)
  (hAD : AD A D) (hAE : AE A E)
  (hBC : BC B C) (hBD : BD B D)
  (hCD : CD C D) (hDE : DE D E) :
  ∃ r : ℕ, r = 16 := 
sorry

end number_of_routes_l49_49416


namespace tony_combined_lift_weight_l49_49841

noncomputable def tony_exercises :=
  let curl_weight := 90 -- pounds.
  let military_press_weight := 2 * curl_weight -- pounds.
  let squat_weight := 5 * military_press_weight -- pounds.
  let bench_press_weight := 1.5 * military_press_weight -- pounds.
  squat_weight + bench_press_weight

theorem tony_combined_lift_weight :
  tony_exercises = 1170 := by
  -- Here we will include the necessary proof steps
  sorry

end tony_combined_lift_weight_l49_49841


namespace unique_positive_integers_sum_l49_49088

noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 77) / 3 + 5 / 3)

theorem unique_positive_integers_sum :
  ∃ (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c),
    x^100 = 3 * x^98 + 17 * x^96 + 13 * x^94 - 2 * x^50 + (a : ℝ) * x^46 + (b : ℝ) * x^44 + (c : ℝ) * x^40
    ∧ a + b + c = 167 := by
  sorry

end unique_positive_integers_sum_l49_49088


namespace polar_to_cartesian_2_pi_over_6_l49_49420

theorem polar_to_cartesian_2_pi_over_6 :
  let r : ℝ := 2
  let θ : ℝ := (Real.pi / 6)
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (Real.sqrt 3, 1) := by
    -- Initialize the constants and their values
    let r := 2
    let θ := Real.pi / 6
    let x := r * Real.cos θ
    let y := r * Real.sin θ
    -- Placeholder for the actual proof
    sorry

end polar_to_cartesian_2_pi_over_6_l49_49420


namespace count_divisibles_l49_49940

theorem count_divisibles (a b lcm : ℕ) (h_lcm: lcm = Nat.lcm 18 (Nat.lcm 24 30)) (h_a: a = 2000) (h_b: b = 3000) :
  (Finset.filter (λ x, x % lcm = 0) (Finset.Icc a b)).card = 3 :=
by
  sorry

end count_divisibles_l49_49940


namespace four_dice_probability_l49_49227

open ProbabilityTheory
open Classical

noncomputable def dice_prob_space : ProbabilitySpace := sorry -- Define the probability space of rolling six 6-sided dice

def condition_no_four_of_a_kind (dice_outcome : Vector ℕ 6) : Prop :=
  ¬∃ n, dice_outcome.count n ≥ 4

def condition_pair_exists (dice_outcome : Vector ℕ 6) : Prop :=
  ∃ n, dice_outcome.count n = 2

def re_rolled_dice (initial_outcome : Vector ℕ 6) (re_roll : Vector ℕ 4) : Vector ℕ 6 :=
  sorry -- Combine initial pair and re-rolled outcomes

def at_least_four_same (dice_outcome : Vector ℕ 6) : Prop :=
  ∃ n, dice_outcome.count n ≥ 4

theorem four_dice_probability :
  ∀ (initial_outcome : Vector ℕ 6)
    (re_roll : Vector ℕ 4),
  (condition_no_four_of_a_kind initial_outcome) →
  (condition_pair_exists initial_outcome) →
  (∃ pr : ℚ, pr = 311 / 648 ∧ 
    (Pr[dice_prob_space, at_least_four_same (re_rolled_dice initial_outcome re_roll)] = pr)) :=
sorry

end four_dice_probability_l49_49227


namespace distance_between_vertices_of_hyperbola_l49_49298

def hyperbola_equation (x y : ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ), c₁ = 4 ∧ c₂ = -4 ∧
    (c₁ * x^2 + 24 * x + c₂ * y^2 + 8 * y + 44 = 0)

theorem distance_between_vertices_of_hyperbola :
  (∀ x y : ℝ, hyperbola_equation x y) → (2 : ℝ) = 2 :=
by
  intro h
  sorry

end distance_between_vertices_of_hyperbola_l49_49298


namespace prize_distribution_l49_49557

theorem prize_distribution : 
  ∃ (n1 n2 n3 : ℕ), -- The number of 1st, 2nd, and 3rd prize winners
  n1 + n2 + n3 = 7 ∧ -- Total number of winners is 7
  n1 * 800 + n2 * 700 + n3 * 300 = 4200 ∧ -- Total prize money distributed is $4200
  n1 = 1 ∧ -- Number of 1st prize winners
  n2 = 4 ∧ -- Number of 2nd prize winners
  n3 = 2 -- Number of 3rd prize winners
:= sorry

end prize_distribution_l49_49557


namespace root_exists_l49_49297

noncomputable def f (x : ℝ) := log10 x + x - 3

theorem root_exists :
  ∃ (x : ℝ), (abs (f 2.6)) < 0.1 :=
begin
  sorry
end

end root_exists_l49_49297


namespace weight_of_b_l49_49238

theorem weight_of_b (A B C : ℝ)
  (h1 : A + B + C = 135)
  (h2 : A + B = 80)
  (h3 : B + C = 94) : 
  B = 39 := 
by 
  sorry

end weight_of_b_l49_49238


namespace melanie_attended_games_l49_49356

theorem melanie_attended_games 
(missed_games total_games attended_games : ℕ) 
(h1 : total_games = 64) 
(h2 : missed_games = 32)
(h3 : attended_games = total_games - missed_games) 
: attended_games = 32 :=
by sorry

end melanie_attended_games_l49_49356


namespace stacy_berries_multiple_l49_49653

theorem stacy_berries_multiple (Skylar_berries : ℕ) (Stacy_berries : ℕ) (Steve_berries : ℕ) (m : ℕ)
  (h1 : Skylar_berries = 20)
  (h2 : Steve_berries = Skylar_berries / 2)
  (h3 : Stacy_berries = m * Steve_berries + 2)
  (h4 : Stacy_berries = 32) :
  m = 3 :=
by
  sorry

end stacy_berries_multiple_l49_49653


namespace number_of_triangles_l49_49413

theorem number_of_triangles (points : List ℝ) (h₀ : points.length = 12)
  (h₁ : ∀ p ∈ points, p ≠ A ∧ p ≠ B ∧ p ≠ C ∧ p ≠ D): 
  (∃ triangles : ℕ, triangles = 216) :=
  sorry

end number_of_triangles_l49_49413


namespace find_initial_investment_l49_49469

-- Define the necessary parameters for the problem
variables (P r : ℝ)

-- Given conditions
def condition1 : Prop := P * (1 + r * 3) = 240
def condition2 : Prop := 150 * (1 + r * 6) = 210

-- The statement to be proved
theorem find_initial_investment (h1 : condition1 P r) (h2 : condition2 r) : P = 200 :=
sorry

end find_initial_investment_l49_49469


namespace dividend_is_176_l49_49850

theorem dividend_is_176 (divisor quotient remainder : ℕ) (h1 : divisor = 19) (h2 : quotient = 9) (h3 : remainder = 5) :
  divisor * quotient + remainder = 176 := by
  sorry

end dividend_is_176_l49_49850


namespace rachel_took_money_l49_49991

theorem rachel_took_money (x y : ℕ) (h₁ : x = 5) (h₂ : y = 3) : x - y = 2 :=
by {
  sorry
}

end rachel_took_money_l49_49991


namespace square_side_length_l49_49525

variables (s : ℝ) (π : ℝ)
  
theorem square_side_length (h : 4 * s = π * s^2 / 2) : s = 8 / π :=
by sorry

end square_side_length_l49_49525


namespace fencing_cost_per_foot_is_3_l49_49011

-- Definitions of the constants given in the problem
def side_length : ℕ := 9
def back_length : ℕ := 18
def total_cost : ℕ := 72
def neighbor_behind_rate : ℚ := 1/2
def neighbor_left_rate : ℚ := 1/3

-- The statement to be proved
theorem fencing_cost_per_foot_is_3 : 
  (total_cost / ((2 * side_length + back_length) - 
                (neighbor_behind_rate * back_length) -
                (neighbor_left_rate * side_length))) = 3 := 
by
  sorry

end fencing_cost_per_foot_is_3_l49_49011


namespace polynomial_inequality_l49_49214

-- Define P(x) as a polynomial with non-negative coefficients
def isNonNegativePolynomial (P : Polynomial ℝ) : Prop :=
  ∀ i, P.coeff i ≥ 0

-- The main theorem, which states that for any polynomial P with non-negative coefficients,
-- if P(1) * P(1) ≥ 1, then P(x) * P(1/x) ≥ 1 for all positive x.
theorem polynomial_inequality (P : Polynomial ℝ) (hP : isNonNegativePolynomial P) (hP1 : P.eval 1 * P.eval 1 ≥ 1) :
  ∀ x : ℝ, 0 < x → P.eval x * P.eval (1 / x) ≥ 1 :=
by {
  sorry
}

end polynomial_inequality_l49_49214


namespace inequality_for_large_exponent_l49_49339

theorem inequality_for_large_exponent (u : ℕ → ℕ) (x : ℕ) (k : ℕ) (hk : k = 100) (hu : u x = 2^x) : 
  2^(2^(x : ℕ)) > 2^(k * x) :=
by 
  sorry

end inequality_for_large_exponent_l49_49339


namespace min_squared_distance_l49_49597

theorem min_squared_distance : 
  ∀ (x y : ℝ), (x - y = 1) → (∃ (a b : ℝ), 
  ((a - 2) ^ 2 + (b - 2) ^ 2 <= (x - 2) ^ 2 + (y - 2) ^ 2) ∧ ((a - 2) ^ 2 + (b - 2) ^ 2 = 1 / 2)) := 
by
  sorry

end min_squared_distance_l49_49597


namespace primes_between_30_and_50_l49_49466

theorem primes_between_30_and_50 : (Finset.card (Finset.filter Nat.Prime (Finset.Ico 30 51))) = 5 :=
by
  sorry

end primes_between_30_and_50_l49_49466


namespace largest_k_sum_of_consecutive_odds_l49_49585

theorem largest_k_sum_of_consecutive_odds (k m : ℕ) (h1 : k * (2 * m + k) = 2^15) : k ≤ 128 :=
by {
  sorry
}

end largest_k_sum_of_consecutive_odds_l49_49585


namespace perfect_squares_of_diophantine_l49_49989

theorem perfect_squares_of_diophantine (a b : ℤ) (h : 2 * a^2 + a = 3 * b^2 + b) :
  ∃ k m : ℤ, (a - b) = k^2 ∧ (2 * a + 2 * b + 1) = m^2 := by
  sorry

end perfect_squares_of_diophantine_l49_49989


namespace combined_bus_capacity_eq_40_l49_49984

theorem combined_bus_capacity_eq_40 (train_capacity : ℕ) (fraction : ℚ) (num_buses : ℕ) 
  (h_train_capacity : train_capacity = 120)
  (h_fraction : fraction = 1/6)
  (h_num_buses : num_buses = 2) :
  num_buses * (train_capacity * fraction).toNat = 40 := by
  sorry

end combined_bus_capacity_eq_40_l49_49984


namespace watermelon_melon_weight_l49_49974

variables {W M : ℝ}

theorem watermelon_melon_weight :
  (2 * W > 3 * M ∨ 3 * W > 4 * M) ∧ ¬ (2 * W > 3 * M ∧ 3 * W > 4 * M) → 12 * W ≤ 18 * M :=
by
  sorry

end watermelon_melon_weight_l49_49974


namespace largest_int_less_than_100_rem_5_by_7_l49_49434

theorem largest_int_less_than_100_rem_5_by_7 :
  ∃ k : ℤ, (7 * k + 5 = 96) ∧ ∀ n : ℤ, (7 * n + 5 < 100) → (n ≤ k) :=
sorry

end largest_int_less_than_100_rem_5_by_7_l49_49434


namespace sandy_age_l49_49225

variables (S M J : ℕ)

def Q1 : Prop := S = M - 14  -- Sandy is younger than Molly by 14 years
def Q2 : Prop := J = S + 6  -- John is older than Sandy by 6 years
def Q3 : Prop := 7 * M = 9 * S  -- The ratio of Sandy's age to Molly's age is 7:9
def Q4 : Prop := 5 * J = 6 * S  -- The ratio of Sandy's age to John's age is 5:6

theorem sandy_age (h1 : Q1 S M) (h2 : Q2 S J) (h3 : Q3 S M) (h4 : Q4 S J) : S = 49 :=
by sorry

end sandy_age_l49_49225


namespace joey_pills_one_week_l49_49625

def pills_on_day (n : ℕ) : ℕ := 1 + 2 * (n - 1)

theorem joey_pills_one_week : (∑ i in Finset.range 7, pills_on_day (i + 1)) = 49 := by
  sorry

end joey_pills_one_week_l49_49625


namespace range_of_a_exists_x_ax2_ax_1_lt_0_l49_49509

theorem range_of_a_exists_x_ax2_ax_1_lt_0 :
  {a : ℝ | ∃ x : ℝ, a * x^2 + a * x + 1 < 0} = {a : ℝ | a < 0 ∨ a > 4} :=
sorry

end range_of_a_exists_x_ax2_ax_1_lt_0_l49_49509


namespace triangle_side_length_l49_49615

theorem triangle_side_length (B C : Real) (b c : Real) 
  (h1 : c * Real.cos B = 12) 
  (h2 : b * Real.sin C = 5) 
  (h3 : b * Real.sin B = 5) : 
  c = 13 := 
sorry

end triangle_side_length_l49_49615


namespace paired_products_not_equal_1000_paired_products_equal_10000_l49_49262

open Nat

theorem paired_products_not_equal_1000 :
  ∀ (a : Fin 1000 → ℤ), (∃ p n : Nat, p + n = 1000 ∧
    p * (p - 1) / 2 + n * (n - 1) / 2 = 2 * p * n) → False :=
by 
  sorry

theorem paired_products_equal_10000 :
  ∀ (a : Fin 10000 → ℤ), (∃ p n : Nat, p + n = 10000 ∧
    p * (p - 1) / 2 + n * (n - 1) / 2 = 2 * p * n) ↔ p = 5050 ∨ p = 4950 :=
by 
  sorry

end paired_products_not_equal_1000_paired_products_equal_10000_l49_49262


namespace units_digit_7_power_l49_49910

theorem units_digit_7_power (n : ℕ) : 
  (7 ^ (6 ^ 5)) % 10 = 1 :=
by
  have h1 : 7 % 10 = 7 := by norm_num
  have h2 : (7 ^ 2) % 10 = 49 % 10 := by rfl    -- 49 % 10 = 9
  have h3 : (7 ^ 3) % 10 = 343 % 10 := by rfl   -- 343 % 10 = 3
  have h4 : (7 ^ 4) % 10 = 2401 % 10 := by rfl  -- 2401 % 10 = 1
  have h_pattern : ∀ k : ℕ, 7 ^ (4 * k) % 10 = 1 := 
    by intro k; cases k; norm_num [pow_succ, mul_comm] -- Pattern repeats every 4
  have h_mod : 6 ^ 5 % 4 = 0 := by
    have h51 : 6 % 4 = 2 := by norm_num
    have h62 : (6 ^ 2) % 4 = 0 := by norm_num
    have h63 : (6 ^ 5) % 4 = (6 * 6 ^ 4) % 4 := by ring_exp
    rw [← h62, h51]; norm_num
  exact h_pattern (6 ^ 5 / 4) -- Using the repetition pattern

end units_digit_7_power_l49_49910


namespace total_students_class_l49_49483

theorem total_students_class (S R : ℕ) 
  (h1 : 2 + 12 + 10 + R = S)
  (h2 : (0 * 2) + (1 * 12) + (2 * 10) + (3 * R) = 2 * S) :
  S = 40 := by
  sorry

end total_students_class_l49_49483


namespace original_days_l49_49709

-- Definitions based on the given problem conditions
def totalLaborers : ℝ := 17.5
def absentLaborers : ℝ := 7
def workingLaborers : ℝ := totalLaborers - absentLaborers
def workDaysByWorkingLaborers : ℝ := 10
def totalLaborDays : ℝ := workingLaborers * workDaysByWorkingLaborers

theorem original_days (D : ℝ) (h : totalLaborers * D = totalLaborDays) : D = 6 := sorry

end original_days_l49_49709


namespace total_distance_correct_l49_49415

def day1_distance : ℕ := (5 * 4) + (3 * 2) + (4 * 3)
def day2_distance : ℕ := (6 * 3) + (2 * 1) + (6 * 3) + (3 * 4)
def day3_distance : ℕ := (4 * 2) + (2 * 1) + (7 * 3) + (5 * 2)

def total_distance : ℕ := day1_distance + day2_distance + day3_distance

theorem total_distance_correct :
  total_distance = 129 := by
  sorry

end total_distance_correct_l49_49415


namespace product_of_legs_divisible_by_12_l49_49817

theorem product_of_legs_divisible_by_12 
  (a b c : ℕ) 
  (h_triangle : a^2 + b^2 = c^2) 
  (h_int : ∃ a b c : ℕ, a^2 + b^2 = c^2) :
  ∃ k : ℕ, a * b = 12 * k :=
sorry

end product_of_legs_divisible_by_12_l49_49817


namespace ratio_black_haired_children_l49_49266

theorem ratio_black_haired_children 
  (n_red : ℕ) (n_total : ℕ) (ratio_red : ℕ) (ratio_blonde : ℕ) (ratio_black : ℕ)
  (h_ratio : ratio_red / ratio_red = 1 ∧ ratio_blonde / ratio_red = 2 ∧ ratio_black / ratio_red = 7 / 3)
  (h_n_red : n_red = 9)
  (h_n_total : n_total = 48) :
  (7 : ℚ) / (16 : ℚ) = (n_total * 7 / 16 : ℚ) :=
sorry

end ratio_black_haired_children_l49_49266


namespace minimum_m_plus_n_l49_49519

theorem minimum_m_plus_n (m n : ℕ) (h1 : 98 * m = n ^ 3) (h2 : 0 < m) (h3 : 0 < n) : m + n = 42 :=
sorry

end minimum_m_plus_n_l49_49519


namespace water_removal_l49_49075

theorem water_removal (n : ℕ) : 
  (∀n, (2:ℚ) / (n + 2) = 1 / 8) ↔ (n = 14) := 
by 
  sorry

end water_removal_l49_49075


namespace determine_number_on_reverse_side_l49_49980

variable (n : ℕ) (k : ℕ) (shown_cards : ℕ → Prop)

theorem determine_number_on_reverse_side :
    -- Conditions
    (∀ i, 1 ≤ i ∧ i ≤ n → (shown_cards (i - 1) ↔ shown_cards i)) →
    -- Prove
    (k = 0 ∨ k = n ∨ (1 ≤ k ∧ k < n ∧ (shown_cards (k - 1) ∨ shown_cards (k + 1)))) →
    (∃ j, (j = 1 ∧ k = 0) ∨ (j = n - 1 ∧ k = n) ∨ 
          (j = k - 1 ∧ k > 0 ∧ k < n ∧ shown_cards (k + 1)) ∨ 
          (j = k + 1 ∧ k > 0 ∧ k < n ∧ shown_cards (k - 1))) :=
by
  sorry

end determine_number_on_reverse_side_l49_49980


namespace evaluate_sixth_iteration_of_g_at_2_l49_49969

def g (x : ℤ) : ℤ := x^2 - 4 * x + 1

theorem evaluate_sixth_iteration_of_g_at_2 :
  g (g (g (g (g (g 2))))) = 59162302643740737293922 := by
  sorry

end evaluate_sixth_iteration_of_g_at_2_l49_49969


namespace cards_dealt_problem_l49_49778

theorem cards_dealt_problem (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      extra_cards := total_cards % total_people,
      people_with_extra_card := extra_cards,
      people_with_fewer_cards := total_people - people_with_extra_card
  in people_with_fewer_cards = 3 :=
by
  sorry

end cards_dealt_problem_l49_49778


namespace vertex_of_parabola_l49_49424

-- Define the parabolic function
def parabola (x : ℝ) : ℝ := 2 * x^2 + 16 * x + 50

-- Define the vertex coordinates we need to prove
def vertex_x : ℝ := -4
def vertex_y : ℝ := 18

-- Prove that the vertex of the parabola is at (-4, 18)
theorem vertex_of_parabola : ∀ x : ℝ, 
  (vertex_x, vertex_y) = (-4, 2 * ((x + 4)^2 - 4^2) + 50) := 
by 
  sorry

end vertex_of_parabola_l49_49424


namespace rhombus_area_correct_l49_49813

/-- Define the rhombus area calculation in miles given the lengths of its diagonals -/
def scale := 250
def d1 := 6 * scale -- first diagonal in miles
def d2 := 12 * scale -- second diagonal in miles
def areaOfRhombus (d1 d2 : ℕ) : ℕ := (d1 * d2) / 2

theorem rhombus_area_correct :
  areaOfRhombus d1 d2 = 2250000 :=
by
  sorry

end rhombus_area_correct_l49_49813


namespace f1_odd_f2_even_l49_49575

noncomputable def f1 (x : ℝ) : ℝ := x + x^3 + x^5
noncomputable def f2 (x : ℝ) : ℝ := x^2 + 1

theorem f1_odd : ∀ x : ℝ, f1 (-x) = - f1 x := 
by
  sorry

theorem f2_even : ∀ x : ℝ, f2 (-x) = f2 x := 
by
  sorry

end f1_odd_f2_even_l49_49575


namespace geometric_sequence_problem_l49_49308

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ a₁ q : ℝ, ∀ n, a n = a₁ * q^n

axiom a_3_eq_2 : ∃ a : ℕ → ℝ, geometric_sequence a ∧ a 3 = 2
axiom a_4a_6_eq_16 : ∃ a : ℕ → ℝ, geometric_sequence a ∧ a 4 * a 6 = 16

theorem geometric_sequence_problem :
  ∃ a : ℕ → ℝ, geometric_sequence a ∧ a 3 = 2 ∧ a 4 * a 6 = 16 →
  (a 9 - a 11) / (a 5 - a 7) = 4 :=
sorry

end geometric_sequence_problem_l49_49308


namespace product_of_four_integers_negative_l49_49857

theorem product_of_four_integers_negative {a b c d : ℤ}
  (h : a * b * c * d < 0) :
  (∃ n : ℕ, n ≤ 3 ∧ (n = (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) + (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0))) :=
sorry

end product_of_four_integers_negative_l49_49857


namespace probability_less_than_side_length_l49_49755

-- Summary of the given conditions
def is_vertex_or_center (p : ℝ × ℝ) (s : ℝ) : Prop :=
  (p = (0, 0)) ∨ (p = (s, 0)) ∨ (p = (0, s)) ∨ (p = (s, s)) ∨ (p = (s / 2, s / 2))

-- Side length of the square
def side_length : ℝ := 1

-- Set of all the possible points (vertices and center of the square with side_length 1)
def points : set (ℝ × ℝ) := {(0,0), (1,0), (0,1), (1,1), (1/2, 1/2)}

-- Set of all pairs of points
def pairs : set ((ℝ × ℝ) × (ℝ × ℝ)) := {((x1, y1), (x2, y2)) | (x1, y1) ∈ points ∧ (x2, y2) ∈ points ∧ (x1, y1) ≠ (x2, y2)}

-- Function to compute the distance between two points
def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Counting the pairs whose distance is less than the side length of the square
def valid_pairs : nat :=
  finset.card (finset.filter (λ (p : (ℝ × ℝ) × (ℝ × ℝ)), distance p.1 p.2 < side_length) (pairs.to_finset))

-- Total number of pairs
def total_pairs : nat :=
  finset.card pairs.to_finset

-- Required probability
def probability : ℝ :=
  valid_pairs.to_real / total_pairs.to_real

-- The Probability proof statement
theorem probability_less_than_side_length :
  probability = 2 / 5 := 
sorry

end probability_less_than_side_length_l49_49755


namespace theorem_incorrect_statement_D_l49_49194

open Real

def incorrect_statement_D (φ : ℝ) (hφ : φ > 0) (x : ℝ) : Prop :=
  cos (2*x + φ) ≠ cos (2*(x - φ/2))

theorem theorem_incorrect_statement_D (φ : ℝ) (hφ : φ > 0) : 
  ∃ x : ℝ, incorrect_statement_D φ hφ x :=
by
  sorry

end theorem_incorrect_statement_D_l49_49194


namespace count_divisibles_l49_49939

theorem count_divisibles (a b lcm : ℕ) (h_lcm: lcm = Nat.lcm 18 (Nat.lcm 24 30)) (h_a: a = 2000) (h_b: b = 3000) :
  (Finset.filter (λ x, x % lcm = 0) (Finset.Icc a b)).card = 3 :=
by
  sorry

end count_divisibles_l49_49939


namespace hanna_gives_roses_l49_49198

-- Conditions as Lean definitions
def initial_budget : ℕ := 300
def price_jenna : ℕ := 2
def price_imma : ℕ := 3
def price_ravi : ℕ := 4
def price_leila : ℕ := 5

def roses_for_jenna (budget : ℕ) : ℕ :=
  budget / price_jenna * 1 / 3

def roses_for_imma (budget : ℕ) : ℕ :=
  budget / price_imma * 1 / 4

def roses_for_ravi (budget : ℕ) : ℕ :=
  budget / price_ravi * 1 / 6

def roses_for_leila (budget : ℕ) : ℕ :=
  budget / price_leila

-- Calculations based on conditions
def roses_jenna : ℕ := Nat.floor (50 * 1/3)
def roses_imma : ℕ := Nat.floor ((100 / price_imma) * 1 / 4)
def roses_ravi : ℕ := Nat.floor ((50 / price_ravi) * 1 / 6)
def roses_leila : ℕ := 50 / price_leila

-- Final statement to be proven
theorem hanna_gives_roses :
  roses_jenna + roses_imma + roses_ravi + roses_leila = 36 := by
  sorry

end hanna_gives_roses_l49_49198


namespace car_mileage_city_l49_49391

theorem car_mileage_city (h c t : ℕ) 
  (h_eq_tank_mileage : 462 = h * t) 
  (c_eq_tank_mileage : 336 = c * t) 
  (mileage_diff : c = h - 3) : 
  c = 8 := 
by
  sorry

end car_mileage_city_l49_49391


namespace pizza_slices_left_l49_49736

/-- Blanch starts with 15 slices of pizza.
    During breakfast, she eats 4 slices.
    At lunch, Blanch eats 2 more slices.
    Blanch takes 2 slices as a snack.
    Finally, she consumes 5 slices for dinner.
    Prove that Blanch has 2 slices left after all meals and snacks. -/
theorem pizza_slices_left :
  let initial_slices := 15 in
  let breakfast := 4 in
  let lunch := 2 in
  let snack := 2 in
  let dinner := 5 in
  initial_slices - breakfast - lunch - snack - dinner = 2 :=
by
  sorry

end pizza_slices_left_l49_49736


namespace probability_of_specific_combination_l49_49057

def total_shirts : ℕ := 3
def total_shorts : ℕ := 7
def total_socks : ℕ := 4
def total_clothes : ℕ := total_shirts + total_shorts + total_socks
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
def favorable_outcomes : ℕ := (choose total_shirts 2) * (choose total_shorts 1) * (choose total_socks 1)
def total_outcomes : ℕ := choose total_clothes 4

theorem probability_of_specific_combination :
  favorable_outcomes / total_outcomes = 84 / 1001 :=
by
  -- Proof omitted
  sorry

end probability_of_specific_combination_l49_49057


namespace store_total_income_l49_49952

def pencil_with_eraser_cost : ℝ := 0.8
def regular_pencil_cost : ℝ := 0.5
def short_pencil_cost : ℝ := 0.4

def pencils_with_eraser_sold : ℕ := 200
def regular_pencils_sold : ℕ := 40
def short_pencils_sold : ℕ := 35

noncomputable def total_money_made : ℝ :=
  (pencil_with_eraser_cost * pencils_with_eraser_sold) +
  (regular_pencil_cost * regular_pencils_sold) +
  (short_pencil_cost * short_pencils_sold)

theorem store_total_income : total_money_made = 194 := by
  sorry

end store_total_income_l49_49952


namespace unique_zero_identity_l49_49515

theorem unique_zero_identity (n : ℤ) : (∀ z : ℤ, z + n = z ∧ z * n = 0) → n = 0 :=
by
  intro h
  have h1 : ∀ z : ℤ, z + n = z := fun z => (h z).left
  have h2 : ∀ z : ℤ, z * n = 0 := fun z => (h z).right
  sorry

end unique_zero_identity_l49_49515


namespace nonagon_diagonals_l49_49053

theorem nonagon_diagonals (n : ℕ) (h1 : n = 9) : (n * (n - 3)) / 2 = 27 := by
  sorry

end nonagon_diagonals_l49_49053


namespace binomial_arithmetic_series_l49_49428

theorem binomial_arithmetic_series {n k : ℕ} (h1 : 2 < k) (h2 : k < n)
  (h3 : nat.choose n (k-1) + nat.choose n (k+1) = 2 * nat.choose n k) :
  ∃ p : ℤ, n = 4 * p^2 - 2 ∧ (k = 2 * p^2 - 1 + p ∨ k = 2 * p^2 - 1 - p) :=
sorry

end binomial_arithmetic_series_l49_49428


namespace fraction_area_of_triangles_l49_49252

theorem fraction_area_of_triangles 
  (base_PQR : ℝ) (height_PQR : ℝ)
  (base_XYZ : ℝ) (height_XYZ : ℝ)
  (h_base_PQR : base_PQR = 3)
  (h_height_PQR : height_PQR = 2)
  (h_base_XYZ : base_XYZ = 6)
  (h_height_XYZ : height_XYZ = 3) :
  (1/2 * base_PQR * height_PQR) / (1/2 * base_XYZ * height_XYZ) = 1 / 3 :=
by
  sorry

end fraction_area_of_triangles_l49_49252


namespace quadratic_symmetry_l49_49948

noncomputable def f (x : ℝ) (a b : ℝ) := a * x^2 + b * x + 1

theorem quadratic_symmetry 
  (a b x1 x2 : ℝ) 
  (h_quad : f x1 a b = f x2 a b) 
  (h_diff : x1 ≠ x2) 
  (h_nonzero : a ≠ 0) :
  f (x1 + x2) a b = 1 := 
by
  sorry

end quadratic_symmetry_l49_49948


namespace number_of_boys_exceeds_girls_by_l49_49206

theorem number_of_boys_exceeds_girls_by (girls boys: ℕ) (h1: girls = 34) (h2: boys = 841) : boys - girls = 807 := by
  sorry

end number_of_boys_exceeds_girls_by_l49_49206


namespace cube_probability_problem_l49_49395

-- Definitions based on the problem's conditions
def total_faces : ℕ := 6
def faces_labeled_1 : ℕ := 1
def faces_labeled_2 : ℕ := 2
def faces_labeled_3 : ℕ := 3

-- Proposition (1): Probability of rolling a 2
def prob_rolling_2 : ℚ := faces_labeled_2 / total_faces

-- Proposition (2): Number with the highest probability of facing up
def highest_prob_number : ℕ := if (faces_labeled_1 > faces_labeled_2) ∧ (faces_labeled_1 > faces_labeled_3) then 1
  else if (faces_labeled_2 > faces_labeled_1) ∧ (faces_labeled_2 > faces_labeled_3) then 2
  else 3

-- Proposition (3): Chances of Winning
def player_A_wins_faces : ℕ := faces_labeled_1 + faces_labeled_2
def player_B_wins_faces : ℕ := faces_labeled_3
def chances_equal : Bool := player_A_wins_faces = player_B_wins_faces

-- The statement to be proved in Lean 4
theorem cube_probability_problem :
  prob_rolling_2 = 1/3 ∧
  highest_prob_number = 3 ∧
  chances_equal = true :=
sorry

end cube_probability_problem_l49_49395


namespace expand_expression_l49_49166

variable {R : Type} [CommRing R]
variable (a b x : R)

theorem expand_expression (a b x : R) :
  (a * x^2 + b) * (5 * x^3) = 35 * x^5 + (-15) * x^3 :=
by
  -- The proof goes here
  sorry

end expand_expression_l49_49166


namespace cards_distribution_l49_49787

open Nat

theorem cards_distribution : 
  ∀ (total_cards people : Nat), total_cards = 60 → people = 9 → 
  let base_cards := total_cards / people;
  let remainder := total_cards % people;
  let num_with_more := remainder;
  let num_with_fewer := people - remainder;
  num_with_fewer = 3 :=
by
  intros total_cards people h_total h_people
  let base_cards := total_cards / people
  let remainder := total_cards % people
  let num_with_more := remainder
  let num_with_fewer := people - remainder
  have h_base_cards : base_cards = 6 := by sorry
  have h_remainder : remainder = 6 := by sorry
  have h_num_with_more : num_with_more = 6 := by rw [h_remainder]; sorry
  have h_num_with_fewer : num_with_fewer = people - remainder := by sorry
  rw [h_people, h_remainder]
  exact rfl

end cards_distribution_l49_49787


namespace factorize_3m2_minus_12_l49_49169

theorem factorize_3m2_minus_12 (m : ℤ) : 
  3 * m^2 - 12 = 3 * (m - 2) * (m + 2) := 
sorry

end factorize_3m2_minus_12_l49_49169


namespace units_digit_of_7_pow_6_pow_5_l49_49896

-- Define the units digit cycle for powers of 7
def units_digit_cycle : List ℕ := [7, 9, 3, 1]

-- Define the function to calculate the units digit of 7^n
def units_digit (n : ℕ) : ℕ :=
  units_digit_cycle[(n % 4) - 1]

-- The main theorem stating the units digit of 7^(6^5) is 1
theorem units_digit_of_7_pow_6_pow_5 : units_digit (6^5) = 1 :=
by
  -- Skipping the proof, including a sorry placeholder
  sorry

end units_digit_of_7_pow_6_pow_5_l49_49896


namespace kanul_total_amount_l49_49630

-- Definitions based on the conditions
def raw_materials_cost : ℝ := 35000
def machinery_cost : ℝ := 40000
def marketing_cost : ℝ := 15000
def total_spent : ℝ := raw_materials_cost + machinery_cost + marketing_cost
def spending_percentage : ℝ := 0.25

-- The statement we want to prove
theorem kanul_total_amount (T : ℝ) (h : total_spent = spending_percentage * T) : T = 360000 :=
by
  sorry

end kanul_total_amount_l49_49630


namespace value_of_ab_over_cd_l49_49793

theorem value_of_ab_over_cd (a b c d : ℚ) (h₁ : a / b = 2 / 3) (h₂ : c / b = 1 / 5) (h₃ : c / d = 7 / 15) : (a * b) / (c * d) = 140 / 9 :=
by
  sorry

end value_of_ab_over_cd_l49_49793


namespace birth_year_1957_l49_49490

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem birth_year_1957 (y x : ℕ) (h : y = 2023) (h1 : sum_of_digits x = y - x) : x = 1957 :=
by
  sorry

end birth_year_1957_l49_49490


namespace initial_amount_l49_49577

theorem initial_amount (A : ℝ) (h : (9 / 8) * (9 / 8) * A = 40500) : 
  A = 32000 :=
sorry

end initial_amount_l49_49577


namespace positive_difference_l49_49541

def a : ℕ := (8^2 - 8^2) / 8
def b : ℕ := (8^2 * 8^2) / 8

theorem positive_difference : |b - a| = 512 :=
by
  sorry

end positive_difference_l49_49541


namespace kibble_consumption_rate_l49_49965

-- Kira fills her cat's bowl with 3 pounds of kibble before going to work.
def initial_kibble : ℚ := 3

-- There is still 1 pound left when she returns.
def remaining_kibble : ℚ := 1

-- Kira was away from home for 8 hours.
def time_away : ℚ := 8

-- Calculate the amount of kibble eaten
def kibble_eaten : ℚ := initial_kibble - remaining_kibble

-- Calculate the rate of consumption (hours per pound)
def rate_of_consumption (time: ℚ) (kibble: ℚ) : ℚ := time / kibble

-- Theorem statement: It takes 4 hours for Kira's cat to eat a pound of kibble.
theorem kibble_consumption_rate : rate_of_consumption time_away kibble_eaten = 4 := by
  sorry

end kibble_consumption_rate_l49_49965


namespace simplify_expression_l49_49635

theorem simplify_expression (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) :
  let x := q/r + r/q
  let y := p/r + r/p
  let z := p/q + q/p
  (x^2 + y^2 + z^2 - 2 * x * y * z) = 4 :=
by
  let x := q/r + r/q
  let y := p/r + r/p
  let z := p/q + q/p
  sorry

end simplify_expression_l49_49635


namespace square_area_of_inscribed_in_parabola_l49_49405

theorem square_area_of_inscribed_in_parabola : 
    ∀ (s : ℝ), -2s = (3 + s)^2 - 6*(3 + s) + 7 → (2*s)^2 = 16 - 8*Real.sqrt 3 :=
by
  intro s hs
  sorry

end square_area_of_inscribed_in_parabola_l49_49405


namespace discount_percentage_l49_49562

theorem discount_percentage 
    (original_price : ℝ) 
    (total_paid : ℝ) 
    (sales_tax_rate : ℝ) 
    (sale_price_before_tax : ℝ) 
    (discount_amount : ℝ) 
    (discount_percentage : ℝ) :
    original_price = 200 → total_paid = 165 → sales_tax_rate = 0.10 →
    total_paid = sale_price_before_tax * (1 + sales_tax_rate) →
    sale_price_before_tax = original_price - discount_amount →
    discount_percentage = (discount_amount / original_price) * 100 →
    discount_percentage = 25 :=
by
  intros h_original h_total h_tax h_eq1 h_eq2 h_eq3
  sorry

end discount_percentage_l49_49562


namespace third_part_of_156_division_proof_l49_49059

theorem third_part_of_156_division_proof :
  ∃ (x : ℚ), (2 * x + (1 / 2) * x + (1 / 4) * x + (1 / 8) * x = 156) ∧ ((1 / 4) * x = 13 + 15 / 23) :=
by
  sorry

end third_part_of_156_division_proof_l49_49059


namespace find_x_l49_49692

theorem find_x (x : ℝ) : 0.3 * x + 0.2 = 0.26 → x = 0.2 :=
by
  sorry

end find_x_l49_49692


namespace mutually_exclusive_event_l49_49064

variables {Ω : Type*} {P : MeasureTheory.Measure Ω}
variables {A B : Set ℝ}

theorem mutually_exclusive_event (h : A ∩ B = ∅) :
  MeasureTheory.Probability.ofSet P A * MeasureTheory.Probability.ofSet P B = 0 :=
by sorry

end mutually_exclusive_event_l49_49064


namespace sin_double_angle_given_condition_l49_49589

open Real

variable (x : ℝ)

theorem sin_double_angle_given_condition :
  sin (π / 4 - x) = 3 / 5 → sin (2 * x) = 7 / 25 :=
by
  intro h
  sorry

end sin_double_angle_given_condition_l49_49589


namespace value_of_expression_l49_49200

theorem value_of_expression (x y : ℝ) (h1 : x = 1 / 2) (h2 : y = 2) : (1 / 3) * x ^ 8 * y ^ 9 = 2 / 3 :=
by
  -- Proof can be filled in here
  sorry

end value_of_expression_l49_49200


namespace petya_vasya_meet_at_lamp_64_l49_49002

-- Definitions of positions of Petya and Vasya
def Petya_position (x : ℕ) : ℕ := x - 21 -- Petya starts from the 1st lamp and is at the 22nd lamp
def Vasya_position (x : ℕ) : ℕ := 88 - x -- Vasya starts from the 100th lamp and is at the 88th lamp

-- Condition that both lanes add up to 64
theorem petya_vasya_meet_at_lamp_64 : ∀ x y : ℕ, 
    Petya_position x = Vasya_position y -> x = 64 :=
by
  intro x y
  rw [Petya_position, Vasya_position]
  sorry

end petya_vasya_meet_at_lamp_64_l49_49002


namespace complex_number_conditions_l49_49441

open Complex Real

noncomputable def is_real (a : ℝ) : Prop :=
a ^ 2 - 2 * a - 15 = 0

noncomputable def is_imaginary (a : ℝ) : Prop :=
a ^ 2 - 2 * a - 15 ≠ 0

noncomputable def is_purely_imaginary (a : ℝ) : Prop :=
a ^ 2 - 9 = 0 ∧ a ^ 2 - 2 * a - 15 ≠ 0

theorem complex_number_conditions (a : ℝ) :
  (is_real a ↔ (a = 5 ∨ a = -3))
  ∧ (is_imaginary a ↔ (a ≠ 5 ∧ a ≠ -3))
  ∧ (¬(∃ a : ℝ, is_purely_imaginary a)) :=
by
  sorry

end complex_number_conditions_l49_49441


namespace train_speed_in_km_per_hr_l49_49874

-- Conditions
def time_in_seconds : ℕ := 9
def length_in_meters : ℕ := 175

-- Conversion factor from m/s to km/hr
def meters_per_sec_to_km_per_hr (speed_m_per_s : ℚ) : ℚ :=
  speed_m_per_s * 3.6

-- Question as statement
theorem train_speed_in_km_per_hr :
  meters_per_sec_to_km_per_hr ((length_in_meters : ℚ) / (time_in_seconds : ℚ)) = 70 := by
  sorry

end train_speed_in_km_per_hr_l49_49874


namespace smallest_m_condition_l49_49439

-- Define the function f_9
def f_9 (n : ℕ) : ℕ := (List.range' 1 9).count (λ d, d ∣ n)

-- The main theorem statement
theorem smallest_m_condition (m : ℕ) (b : ℕ → ℝ) (h : ∀ n > m, f_9 n = ∑ j in Finset.range(m), b j * f_9 (n - j)) : m = 28 :=
sorry

end smallest_m_condition_l49_49439


namespace tangency_condition_l49_49114

-- Define the equation for the ellipse
def ellipse_eq (x y : ℝ) : Prop :=
  3 * x^2 + 9 * y^2 = 9

-- Define the equation for the hyperbola
def hyperbola_eq (x y m : ℝ) : Prop :=
  (x - 2)^2 - m * (y + 1)^2 = 1

-- Prove that for the ellipse and hyperbola to be tangent, m must equal 3
theorem tangency_condition (m : ℝ) :
  (∀ x y : ℝ, ellipse_eq x y ∧ hyperbola_eq x y m) → m = 3 :=
by
  sorry

end tangency_condition_l49_49114


namespace evaluate_five_applications_of_f_l49_49505

def f (x : ℤ) : ℤ :=
  if x ≥ 0 then x + 5 else -x^2 - 3

theorem evaluate_five_applications_of_f :
  f (f (f (f (f (-1))))) = -17554795004 :=
by
  sorry

end evaluate_five_applications_of_f_l49_49505


namespace prove_ellipse_and_dot_product_l49_49756

open Real

-- Assume the given conditions
variables (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_ab : a > b)
variable (e : ℝ) (he : e = sqrt 2 / 2)
variable (h_chord : 2 = 2 * sqrt (a^2 - 1))
variables (k : ℝ) (hk : k ≠ 0)

-- Given equation of points on the line and the ellipse
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 2 + y^2 = 1
def line_eq (x y : ℝ) : Prop := y = k * (x - 1)

-- The points A and B lie on the ellipse and the line
variables (x1 y1 x2 y2 : ℝ)
variable (A : x1^2 / 2 + y1^2 = 1 ∧ y1 = k * (x1 - 1))
variable (B : x2^2 / 2 + y2^2 = 1 ∧ y2 = k * (x2 - 1))

-- Define the dot product condition
def MA_dot_MB (m : ℝ) : ℝ :=
  let x1_term := x1 - m
  let x2_term := x2 - m
  let dot_product := (x1_term * x2_term + y1 * y2)
  dot_product

-- The statement we need to prove
theorem prove_ellipse_and_dot_product :
  (a^2 = 2) ∧ (b = 1) ∧ (c = 1) ∧ (∃ (m : ℝ), m = 5 / 4 ∧ MA_dot_MB m = -7 / 16) :=
sorry

end prove_ellipse_and_dot_product_l49_49756


namespace find_some_value_l49_49858

-- Define the main variables and assumptions
variable (m n some_value : ℝ)

-- State the assumptions based on the conditions
axiom h1 : m = n / 2 - 2 / 5
axiom h2 : m + some_value = (n + 4) / 2 - 2 / 5

-- State the theorem we are trying to prove
theorem find_some_value : some_value = 2 :=
by
  -- Proof goes here, for now we just put sorry
  sorry

end find_some_value_l49_49858


namespace willam_land_percentage_l49_49699

-- Definitions from conditions
def farm_tax_rate : ℝ := 0.6
def total_tax_collected : ℝ := 3840
def mr_willam_tax_paid : ℝ := 500

-- Goal to prove: percentage of Mr. Willam's land over total taxable land of the village
noncomputable def percentage_mr_willam_land : ℝ :=
  (mr_willam_tax_paid / total_tax_collected) * 100

theorem willam_land_percentage :
  percentage_mr_willam_land = 13.02 := 
  by 
  sorry

end willam_land_percentage_l49_49699


namespace sequence_ninth_term_l49_49001

theorem sequence_ninth_term (a b : ℚ) :
  ∀ n : ℕ, n = 9 → (-1 : ℚ) ^ n * (n * b ^ n) / ((n + 1) * a ^ (n + 2)) = -9 * b^9 / (10 * a^11) :=
by
  sorry

end sequence_ninth_term_l49_49001


namespace combined_total_cost_is_correct_l49_49588

-- Define the number and costs of balloons for each person
def Fred_yellow_count : ℕ := 5
def Fred_red_count : ℕ := 3
def Fred_yellow_cost_per : ℕ := 3
def Fred_red_cost_per : ℕ := 4

def Sam_yellow_count : ℕ := 6
def Sam_red_count : ℕ := 4
def Sam_yellow_cost_per : ℕ := 4
def Sam_red_cost_per : ℕ := 5

def Mary_yellow_count : ℕ := 7
def Mary_red_count : ℕ := 5
def Mary_yellow_cost_per : ℕ := 5
def Mary_red_cost_per : ℕ := 6

def Susan_yellow_count : ℕ := 4
def Susan_red_count : ℕ := 6
def Susan_yellow_cost_per : ℕ := 6
def Susan_red_cost_per : ℕ := 7

def Tom_yellow_count : ℕ := 10
def Tom_red_count : ℕ := 8
def Tom_yellow_cost_per : ℕ := 2
def Tom_red_cost_per : ℕ := 3

-- Formula to calculate total cost for a given person
def total_cost (yellow_count red_count yellow_cost_per red_cost_per : ℕ) : ℕ :=
  (yellow_count * yellow_cost_per) + (red_count * red_cost_per)

-- Total costs for each person
def Fred_total_cost := total_cost Fred_yellow_count Fred_red_count Fred_yellow_cost_per Fred_red_cost_per
def Sam_total_cost := total_cost Sam_yellow_count Sam_red_count Sam_yellow_cost_per Sam_red_cost_per
def Mary_total_cost := total_cost Mary_yellow_count Mary_red_count Mary_yellow_cost_per Mary_red_cost_per
def Susan_total_cost := total_cost Susan_yellow_count Susan_red_count Susan_yellow_cost_per Susan_red_cost_per
def Tom_total_cost := total_cost Tom_yellow_count Tom_red_count Tom_yellow_cost_per Tom_red_cost_per

-- Combined total cost
def combined_total_cost : ℕ :=
  Fred_total_cost + Sam_total_cost + Mary_total_cost + Susan_total_cost + Tom_total_cost

-- Lean statement to prove
theorem combined_total_cost_is_correct : combined_total_cost = 246 :=
by
  dsimp [combined_total_cost, Fred_total_cost, Sam_total_cost, Mary_total_cost, Susan_total_cost, Tom_total_cost, total_cost]
  sorry

end combined_total_cost_is_correct_l49_49588


namespace combined_bus_capacity_eq_40_l49_49983

theorem combined_bus_capacity_eq_40 (train_capacity : ℕ) (fraction : ℚ) (num_buses : ℕ) 
  (h_train_capacity : train_capacity = 120)
  (h_fraction : fraction = 1/6)
  (h_num_buses : num_buses = 2) :
  num_buses * (train_capacity * fraction).toNat = 40 := by
  sorry

end combined_bus_capacity_eq_40_l49_49983


namespace min_value_of_expression_l49_49498

theorem min_value_of_expression (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_abc : a * b * c = 4) :
  (3 * a + b) * (2 * b + 3 * c) * (a * c + 4) ≥ 384 := 
by sorry

end min_value_of_expression_l49_49498


namespace min_abs_ab_perpendicular_lines_l49_49046

theorem min_abs_ab_perpendicular_lines (a b : ℝ) (h : a * b = a ^ 2 + 1) : |a * b| = 1 :=
by sorry

end min_abs_ab_perpendicular_lines_l49_49046


namespace auntie_em_can_park_l49_49561

noncomputable def parking_probability : ℚ :=
  let total_ways := (Nat.choose 20 5)
  let unfavorables := (Nat.choose 14 5)
  let probability_cannot_park := (unfavorables : ℚ) / total_ways
  1 - probability_cannot_park

theorem auntie_em_can_park :
  parking_probability = 964 / 1107 :=
by
  sorry

end auntie_em_can_park_l49_49561


namespace negation_of_p_l49_49610

namespace ProofProblem

variable (x : ℝ)

def p : Prop := ∃ x : ℝ, x^2 + x - 1 ≥ 0

def neg_p : Prop := ∀ x : ℝ, x^2 + x - 1 < 0

theorem negation_of_p : ¬p = neg_p := sorry

end ProofProblem

end negation_of_p_l49_49610


namespace courtyard_width_l49_49867

theorem courtyard_width 
  (L : ℝ) (N : ℕ) (brick_length brick_width : ℝ) (courtyard_area : ℝ)
  (hL : L = 18)
  (hN : N = 30000)
  (hbrick_length : brick_length = 0.12)
  (hbrick_width : brick_width = 0.06)
  (hcourtyard_area : courtyard_area = (N : ℝ) * (brick_length * brick_width)) :
  (courtyard_area / L) = 12 :=
by
  sorry

end courtyard_width_l49_49867


namespace statement_A_statement_B_statement_C_statement_D_statement_E_l49_49694

-- Define a statement for each case and prove each one
theorem statement_A (x : ℝ) (h : x ≥ 0) : x^2 ≥ x :=
sorry

theorem statement_B (x : ℝ) (h : x^2 ≥ 0) : abs x ≥ 0 :=
sorry

theorem statement_C (x : ℝ) (h : x^2 ≤ x) : ¬ (x ≤ 1) :=
sorry

theorem statement_D (x : ℝ) (h : x^2 ≥ x) : ¬ (x ≤ 0) :=
sorry

theorem statement_E (x : ℝ) (h : x ≤ -1) : x^2 ≥ abs x :=
sorry

end statement_A_statement_B_statement_C_statement_D_statement_E_l49_49694


namespace triangle_inequality_l49_49502

theorem triangle_inequality 
(a b c : ℝ) (α β γ : ℝ)
(h_t : a + b > c ∧ a + c > b ∧ b + c > a)
(h_opposite : 0 < α ∧ α < π ∧ 0 < β ∧ β < π ∧ 0 < γ ∧ γ < π ∧ α + β + γ = π) :
  a * α + b * β + c * γ ≥ a * β + b * γ + c * α :=
sorry

end triangle_inequality_l49_49502


namespace isosceles_triangle_base_length_l49_49666

theorem isosceles_triangle_base_length
  (b : ℕ)
  (congruent_side : ℕ)
  (perimeter : ℕ)
  (h1 : congruent_side = 8)
  (h2 : perimeter = 25)
  (h3 : 2 * congruent_side + b = perimeter) :
  b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l49_49666


namespace road_trip_mileage_base10_l49_49079

-- Defining the base 8 number 3452
def base8_to_base10 (n : Nat) : Nat :=
  3 * 8^3 + 4 * 8^2 + 5 * 8^1 + 2 * 8^0

-- Stating the problem as a theorem
theorem road_trip_mileage_base10 : base8_to_base10 3452 = 1834 := by
  sorry

end road_trip_mileage_base10_l49_49079


namespace population_time_interval_l49_49622

theorem population_time_interval (T : ℕ) 
  (birth_rate : ℕ) (death_rate : ℕ) (net_increase_day : ℕ) (seconds_in_day : ℕ)
  (h_birth_rate : birth_rate = 8) 
  (h_death_rate : death_rate = 6) 
  (h_net_increase_day : net_increase_day = 86400)
  (h_seconds_in_day : seconds_in_day = 86400) : 
  T = 2 := sorry

end population_time_interval_l49_49622


namespace range_of_a_l49_49473

theorem range_of_a (a : ℝ) :
  (-1 < x ∧ x < 0 → (x^2 - a * x + 2 * a) > 0) ∧
  (0 < x → (x^2 - a * x + 2 * a) < 0) ↔ -1 / 3 < a ∧ a < 0 :=
sorry

end range_of_a_l49_49473


namespace scientific_notation_of_number_l49_49855

theorem scientific_notation_of_number :
  ∀ (n : ℕ), n = 450000000 -> n = 45 * 10^7 := 
by
  sorry

end scientific_notation_of_number_l49_49855


namespace min_distance_from_P_to_origin_l49_49444

noncomputable def distance_to_origin : ℝ := 8 / 5

theorem min_distance_from_P_to_origin
  (P : ℝ × ℝ)
  (hA : P.1^2 + P.2^2 = 1)
  (hB : (P.1 - 3)^2 + (P.2 + 4)^2 = 10)
  (h_tangent : PE = PD) :
  dist P (0, 0) = distance_to_origin := 
sorry

end min_distance_from_P_to_origin_l49_49444


namespace min_slope_at_a_half_l49_49446

theorem min_slope_at_a_half (a : ℝ) (h : 0 < a) :
  (∀ b : ℝ, 0 < b → 4 * b + 1 / b ≥ 4) → (4 * a + 1 / a = 4) → a = 1 / 2 :=
by
  sorry

end min_slope_at_a_half_l49_49446


namespace games_new_friends_l49_49963

-- Definitions based on the conditions
def total_games_all_friends : ℕ := 141
def games_old_friends : ℕ := 53

-- Statement of the problem
theorem games_new_friends {games_new_friends : ℕ} :
  games_new_friends = total_games_all_friends - games_old_friends :=
sorry

end games_new_friends_l49_49963


namespace common_fraction_proof_l49_49296

def expr_as_common_fraction : Prop :=
  let numerator := (3 / 6) + (4 / 5)
  let denominator := (5 / 12) + (1 / 4)
  (numerator / denominator) = (39 / 20)

theorem common_fraction_proof : expr_as_common_fraction :=
by
  sorry

end common_fraction_proof_l49_49296


namespace triangle_side_calculation_l49_49332

theorem triangle_side_calculation
  (a : ℝ) (A B : ℝ)
  (ha : a = 3)
  (hA : A = 30)
  (hB : B = 15) :
  let C := 180 - A - B
  let c := a * (Real.sin C) / (Real.sin A)
  c = 3 * Real.sqrt 2 := by
  sorry

end triangle_side_calculation_l49_49332


namespace label_subsets_l49_49966

open Finset

theorem label_subsets (S : Finset α) :
  ∃ (A : List (Finset α)), 
    A.head = ∅ ∧ 
    (∀ (n : ℕ), n < A.length - 1 → 
      (A.get n ⊆ A.get (n + 1) ∧ (A.get (n + 1) \ A.get n).card = 1) ∨ 
      (A.get (n + 1) ⊆ A.get n ∧ (A.get n \ A.get (n + 1)).card = 1)) :=
sorry

end label_subsets_l49_49966


namespace compute_a4_b4_c4_l49_49363

theorem compute_a4_b4_c4 (a b c : ℝ) (h1 : a + b + c = 8) (h2 : ab + ac + bc = 13) (h3 : abc = -22) : a^4 + b^4 + c^4 = 1378 :=
by
  sorry

end compute_a4_b4_c4_l49_49363


namespace reading_minutes_per_disc_l49_49272

-- Define the total reading time
def total_reading_time := 630

-- Define the maximum capacity per disc
def max_capacity_per_disc := 80

-- Define the allowable unused space
def max_unused_space := 4

-- Define the effective capacity of each disc
def effective_capacity_per_disc := max_capacity_per_disc - max_unused_space

-- Define the number of discs needed, rounded up as a ceiling function
def number_of_discs := Nat.ceil (total_reading_time / effective_capacity_per_disc)

-- Theorem statement: Each disc will contain 70 minutes of reading if all conditions are met
theorem reading_minutes_per_disc : ∀ (total_reading_time : ℕ) (max_capacity_per_disc : ℕ) (max_unused_space : ℕ)
  (effective_capacity_per_disc := max_capacity_per_disc - max_unused_space) 
  (number_of_discs := Nat.ceil (total_reading_time / effective_capacity_per_disc)), 
  number_of_discs = 9 → total_reading_time / number_of_discs = 70 :=
by
  sorry

end reading_minutes_per_disc_l49_49272


namespace rebus_solution_l49_49744

-- We state the conditions:
variables (A B Γ D : ℤ)

-- Define the correct values
def A_correct := 2
def B_correct := 7
def Γ_correct := 1
def D_correct := 0

-- State the conditions as assumptions
axiom cond1 : A * B + 8 = 3 * B
axiom cond2 : Γ * D + B = 5  -- Adjusted assuming V = 5 from problem data
axiom cond3 : Γ * B + 3 = A * D

-- State the goal to be proved
theorem rebus_solution : A = A_correct ∧ B = B_correct ∧ Γ = Γ_correct ∧ D = D_correct :=
by
  sorry

end rebus_solution_l49_49744


namespace area_of_right_triangle_l49_49714

theorem area_of_right_triangle
    (a b c : ℝ)
    (h₀ : a = 9)
    (h₁ : b = 12)
    (h₂ : c = 15)
    (right_triangle : a^2 + b^2 = c^2) :
    (1 / 2) * a * b = 54 := by
  sorry

end area_of_right_triangle_l49_49714


namespace piglets_each_ate_6_straws_l49_49845

theorem piglets_each_ate_6_straws (total_straws : ℕ) (fraction_for_adult_pigs : ℚ) (piglets : ℕ) 
  (h1 : total_straws = 300) 
  (h2 : fraction_for_adult_pigs = 3/5) 
  (h3 : piglets = 20) :
  (total_straws * (1 - fraction_for_adult_pigs) / piglets) = 6 :=
by
  sorry

end piglets_each_ate_6_straws_l49_49845


namespace slices_of_pizza_left_l49_49737

theorem slices_of_pizza_left (initial_slices: ℕ) 
  (breakfast_slices: ℕ) (lunch_slices: ℕ) (snack_slices: ℕ) (dinner_slices: ℕ) :
  initial_slices = 15 →
  breakfast_slices = 4 →
  lunch_slices = 2 →
  snack_slices = 2 →
  dinner_slices = 5 →
  (initial_slices - breakfast_slices - lunch_slices - snack_slices - dinner_slices) = 2 :=
by
  intros
  repeat { sorry }

end slices_of_pizza_left_l49_49737


namespace isosceles_triangle_base_length_l49_49668

theorem isosceles_triangle_base_length
  (a b : ℕ)
  (ha : a = 8)
  (hp : 2 * a + b = 25)
  : b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l49_49668


namespace roots_of_quadratic_l49_49832

theorem roots_of_quadratic (x : ℝ) : 3 * (x - 3) = (x - 3) ^ 2 → x = 3 ∨ x = 6 :=
by
  intro h
  sorry

end roots_of_quadratic_l49_49832


namespace calculate_expression_l49_49879

theorem calculate_expression :
  let a := 2^4
  let b := 2^2
  let c := 2^3
  (a^2 / b^3) * c^3 = 2048 :=
by
  sorry -- Proof is omitted as per instructions

end calculate_expression_l49_49879


namespace find_special_three_digit_numbers_l49_49892

theorem find_special_three_digit_numbers :
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ a ≠ 0 ∧ 
  (100 * a + 10 * b + (c + 3)) % 10 + (100 * a + 10 * (b + 1) + c).div 10 % 10 + (100 * (a + 1) + 10 * b + c).div 100 % 10 + 3 = 
  (a + b + c) / 3)} → n = 117 ∨ n = 207 ∨ n = 108 :=
by
  sorry

end find_special_three_digit_numbers_l49_49892


namespace number_of_primes_between_30_and_50_l49_49459

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the interval condition
def in_interval (n : ℕ) : Prop :=
  30 ≤ n ∧ n ≤ 50

-- Define the count of prime numbers in the interval
def prime_count_in_interval : ℕ :=
  (List.range' 30 21).countp (λ n, is_prime n)

-- We state that the above count is equal to 5
theorem number_of_primes_between_30_and_50 : prime_count_in_interval = 5 :=
  sorry

end number_of_primes_between_30_and_50_l49_49459


namespace mark_total_cost_is_correct_l49_49354

variable (hours : ℕ) (hourly_rate part_cost : ℕ)

def total_cost (hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) :=
  hours * hourly_rate + part_cost

theorem mark_total_cost_is_correct : 
  hours = 2 → hourly_rate = 75 → part_cost = 150 → total_cost hours hourly_rate part_cost = 300 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end mark_total_cost_is_correct_l49_49354


namespace find_water_in_sport_formulation_l49_49259

noncomputable def standard_formulation : ℚ × ℚ × ℚ := (1, 12, 30)
noncomputable def sport_flavoring_to_corn : ℚ := 3 * (1 / 12)
noncomputable def sport_flavoring_to_water : ℚ := (1 / 2) * (1 / 30)
noncomputable def sport_formulation (f : ℚ) (c : ℚ) (w : ℚ) : Prop :=
  f / c = sport_flavoring_to_corn ∧ f / w = sport_flavoring_to_water

noncomputable def given_corn_syrup : ℚ := 8

theorem find_water_in_sport_formulation :
  ∀ (f c w : ℚ), sport_formulation f c w → c = given_corn_syrup → w = 120 :=
by
  sorry

end find_water_in_sport_formulation_l49_49259


namespace sum_first_n_terms_l49_49309

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_first_n_terms
  (a : ℕ → ℝ)
  (h_seq : arithmetic_sequence a)
  (h_a2a4 : a 2 + a 4 = 8)
  (h_common_diff : ∀ n : ℕ, a (n + 1) = a n + 2) :
  ∃ S_n : ℕ → ℝ, ∀ n : ℕ, S_n n = n^2 - n :=
by 
  sorry

end sum_first_n_terms_l49_49309


namespace count_players_studying_chemistry_l49_49567

theorem count_players_studying_chemistry :
  ∀ 
    (total_players : ℕ)
    (math_players : ℕ)
    (physics_players : ℕ)
    (math_and_physics_players : ℕ)
    (all_three_subjects_players : ℕ),
    total_players = 18 →
    math_players = 10 →
    physics_players = 6 →
    math_and_physics_players = 3 →
    all_three_subjects_players = 2 →
    (total_players - (math_players + physics_players - math_and_physics_players)) + all_three_subjects_players = 7 :=
by
  intros total_players math_players physics_players math_and_physics_players all_three_subjects_players
  sorry

end count_players_studying_chemistry_l49_49567


namespace fraction_division_l49_49009

theorem fraction_division :
  (3 / 4) / (5 / 8) = 6 / 5 :=
by
  sorry

end fraction_division_l49_49009


namespace expression_evaluation_l49_49325

theorem expression_evaluation (k : ℚ) (h : 3 * k = 10) : (6 / 5) * k - 2 = 2 :=
by
  sorry

end expression_evaluation_l49_49325


namespace num_enemies_left_l49_49802

-- Definitions of conditions
def points_per_enemy : Nat := 5
def total_enemies : Nat := 8
def earned_points : Nat := 10

-- Theorem statement to prove the number of undefeated enemies
theorem num_enemies_left (points_per_enemy total_enemies earned_points : Nat) : 
    (earned_points / points_per_enemy) <= total_enemies →
    total_enemies - (earned_points / points_per_enemy) = 6 := by
  sorry

end num_enemies_left_l49_49802


namespace factor_correct_l49_49167

noncomputable def factor_expression (x : ℝ) : ℝ :=
  66 * x^6 - 231 * x^12

theorem factor_correct (x : ℝ) :
  factor_expression x = 33 * x^6 * (2 - 7 * x^6) :=
by 
  sorry

end factor_correct_l49_49167


namespace cost_of_jeans_and_shirts_l49_49157

theorem cost_of_jeans_and_shirts 
  (S : ℕ) (J : ℕ) (X : ℕ)
  (hS : S = 18)
  (h2J3S : 2 * J + 3 * S = 76)
  (h3J2S : 3 * J + 2 * S = X) :
  X = 69 :=
by
  sorry

end cost_of_jeans_and_shirts_l49_49157


namespace sara_sent_letters_l49_49099

theorem sara_sent_letters (J : ℕ)
  (h1 : 9 + 3 * J + J = 33) : J = 6 :=
by
  sorry

end sara_sent_letters_l49_49099


namespace different_colors_of_roads_leading_out_l49_49620

-- Define the city with intersections and streets
variables (n : ℕ) -- number of intersections
variables (c₁ c₂ c₃ : ℕ) -- number of external roads of each color

-- Conditions
axiom intersections_have_three_streets : ∀ (i : ℕ), i < n → (∀ (color : ℕ), color < 3 → exists (s : ℕ → ℕ), s color < n ∧ s color ≠ s ((color + 1) % 3) ∧ s color ≠ s ((color + 2) % 3))
axiom streets_colored_differently : ∀ (i : ℕ), i < n → (∀ (color1 color2 : ℕ), color1 < 3 → color2 < 3 → color1 ≠ color2 → exists (s1 s2 : ℕ → ℕ), s1 color1 < n ∧ s2 color2 < n ∧ s1 color1 ≠ s2 color2)

-- Problem Statement
theorem different_colors_of_roads_leading_out (h₁ : n % 2 = 0) (h₂ : c₁ + c₂ + c₃ = 3) : c₁ = 1 ∧ c₂ = 1 ∧ c₃ = 1 :=
by sorry

end different_colors_of_roads_leading_out_l49_49620


namespace trajectory_of_point_l49_49328

theorem trajectory_of_point (P : ℝ × ℝ) 
  (h1 : dist P (0, 3) = dist P (x1, -3)) :
  ∃ p > 0, (P.fst)^2 = 2 * p * P.snd ∧ p = 6 :=
by {
  sorry
}

end trajectory_of_point_l49_49328


namespace impossible_transformation_l49_49728

-- Definition of the allowed operations
def operation_swap (p : ℕ × ℕ) : ℕ × ℕ := (p.2, p.1)
def operation_add (p : ℕ × ℕ) : ℕ × ℕ := (p.1 + p.2, p.2)
def operation_diff (p : ℕ × ℕ) : ℕ × ℕ := (p.1, abs (p.1 - p.2))

def automaton_operations (p : ℕ × ℕ) : list (ℕ × ℕ) :=
  [operation_swap p, operation_add p, operation_diff p]

-- Initial and target pairs
def initial_pair : ℕ × ℕ := (901, 1219)
def target_pair : ℕ × ℕ := (871, 1273)

-- Main theorem
theorem impossible_transformation :
  ¬ ∃ (seq : list (ℕ × ℕ → ℕ × ℕ)),
    foldl (λ p f, f p) initial_pair seq = target_pair :=
sorry

end impossible_transformation_l49_49728


namespace pool_capacity_l49_49274

noncomputable def total_capacity : ℝ := 1000

theorem pool_capacity
    (C : ℝ)
    (H1 : 0.75 * C = 0.45 * C + 300)
    (H2 : 300 / 0.3 = 1000)
    : C = total_capacity :=
by
  -- Solution steps are omitted, proof goes here.
  sorry

end pool_capacity_l49_49274


namespace problem_l49_49199

theorem problem (n : ℝ) (h : n + 1 / n = 10) : n ^ 2 + 1 / n ^ 2 + 5 = 103 :=
by sorry

end problem_l49_49199


namespace pluto_orbit_scientific_notation_l49_49680

theorem pluto_orbit_scientific_notation : 5900000000 = 5.9 * 10^9 := by
  sorry

end pluto_orbit_scientific_notation_l49_49680


namespace xy_sum_values_l49_49607

theorem xy_sum_values (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x < y) (h4 : x + y + x * y = 119) : 
  x + y = 27 ∨ x + y = 24 ∨ x + y = 21 ∨ x + y = 20 :=
sorry

end xy_sum_values_l49_49607


namespace question1_question2_l49_49587

-- Define the function representing the inequality
def inequality (a x : ℝ) : Prop := (a * x - 5) / (x - a) < 0

-- Question 1: Compute the solution set M when a=1
theorem question1 : (setOf (λ x : ℝ => inequality 1 x)) = {x : ℝ | 1 < x ∧ x < 5} :=
by
  sorry

-- Question 2: Determine the range for a such that 3 ∈ M but 5 ∉ M
theorem question2 : (setOf (λ a : ℝ => 3 ∈ (setOf (λ x : ℝ => inequality a x)) ∧ 5 ∉ (setOf (λ x : ℝ => inequality a x)))) = 
  {a : ℝ | (1 ≤ a ∧ a < 5 / 3) ∨ (3 < a ∧ a ≤ 5)} :=
by
  sorry

end question1_question2_l49_49587


namespace find_a_l49_49314

theorem find_a (a : ℕ) (h_pos : a > 0) (h_quadrant : 2 - a > 0) : a = 1 := by
  sorry

end find_a_l49_49314


namespace car_body_mass_l49_49839

theorem car_body_mass (m_model : ℕ) (scale : ℕ) : 
  m_model = 1 → scale = 11 → m_car = 1331 :=
by 
  intros h1 h2
  sorry

end car_body_mass_l49_49839


namespace cards_dealt_problem_l49_49780

theorem cards_dealt_problem (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      extra_cards := total_cards % total_people,
      people_with_extra_card := extra_cards,
      people_with_fewer_cards := total_people - people_with_extra_card
  in people_with_fewer_cards = 3 :=
by
  sorry

end cards_dealt_problem_l49_49780


namespace problem_l49_49920

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x ≤ 2}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def compN : Set ℝ := {x | x < -1 ∨ 1 < x}
def intersection : Set ℝ := {x | x < -1 ∨ (1 < x ∧ x ≤ 2)}

theorem problem (x : ℝ) : x ∈ (M ∩ compN) ↔ x ∈ intersection := by
  sorry

end problem_l49_49920


namespace students_in_each_class_l49_49814

theorem students_in_each_class (S : ℕ) 
  (h1 : 10 * S * 5 = 1750) : 
  S = 35 := 
by 
  sorry

end students_in_each_class_l49_49814


namespace time_to_walk_against_walkway_150_l49_49401

def v_p := 4 / 3
def v_w := 2 - v_p
def distance := 100
def time_against_walkway := distance / (v_p - v_w)

theorem time_to_walk_against_walkway_150 :
  time_against_walkway = 150 := by
  -- Note: Proof goes here (not required)
  sorry

end time_to_walk_against_walkway_150_l49_49401


namespace arithmetic_sequence_product_l49_49672

theorem arithmetic_sequence_product (a d : ℕ) :
  (a + 7 * d = 20) → (d = 2) → ((a + d) * (a + 2 * d) = 80) :=
by
  intros h₁ h₂
  sorry

end arithmetic_sequence_product_l49_49672


namespace micah_has_seven_fish_l49_49217

-- Definitions from problem conditions
def micahFish (M : ℕ) : Prop :=
  let kennethFish := 3 * M
  let matthiasFish := kennethFish - 15
  M + kennethFish + matthiasFish = 34

-- Main statement: prove that the number of fish Micah has is 7
theorem micah_has_seven_fish : ∃ M : ℕ, micahFish M ∧ M = 7 :=
by
  sorry

end micah_has_seven_fish_l49_49217


namespace team_size_is_nine_l49_49565

noncomputable def number_of_workers (n x y : ℕ) : ℕ :=
  if 7 * n = (n - 2) * x ∧ 7 * n = (n - 6) * y then n else 0

theorem team_size_is_nine (x y : ℕ) :
  number_of_workers 9 x y = 9 :=
by
  sorry

end team_size_is_nine_l49_49565


namespace units_digit_of_7_pow_6_pow_5_l49_49905

theorem units_digit_of_7_pow_6_pow_5 : (7^(6^5)) % 10 = 1 := by
  -- Proof goes here
  sorry

end units_digit_of_7_pow_6_pow_5_l49_49905


namespace inequality_condition_l49_49306

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + 5*x + 6

-- Define the main theorem to be proven
theorem inequality_condition (a b : ℝ) (h_a : a > 11 / 4) (h_b : b > 3 / 2) :
  (∀ x : ℝ, |x + 1| < b → |f x + 3| < a) :=
by
  -- We state the required proof without providing the steps
  sorry

end inequality_condition_l49_49306


namespace find_number_l49_49138

theorem find_number (x : ℝ) (h : 0.15 * 40 = 0.25 * x + 2) : x = 16 :=
by
  sorry

end find_number_l49_49138


namespace triangle_area_is_54_l49_49721

-- Define the sides of the triangle
def side1 : ℕ := 9
def side2 : ℕ := 12
def side3 : ℕ := 15

-- Verify that it is a right triangle using the Pythagorean theorem
def isRightTriangle (a b c : ℕ) : Prop := a * a + b * b = c * c

-- Define the area calculation for a right triangle
def areaRightTriangle (a b : ℕ) : ℕ := Nat.div (a * b) 2

-- State the theorem (Problem) to prove
theorem triangle_area_is_54 :
  isRightTriangle side1 side2 side3 ∧ areaRightTriangle side1 side2 = 54 :=
by
  sorry

end triangle_area_is_54_l49_49721


namespace distribution_function_for_closed_set_l49_49637

open MeasureTheory

theorem distribution_function_for_closed_set (C : set ℝ) (hC : is_closed C) :
  ∃ F : ℝ → ℝ, support F = C := sorry

end distribution_function_for_closed_set_l49_49637


namespace units_digit_7_power_l49_49909

theorem units_digit_7_power (n : ℕ) : 
  (7 ^ (6 ^ 5)) % 10 = 1 :=
by
  have h1 : 7 % 10 = 7 := by norm_num
  have h2 : (7 ^ 2) % 10 = 49 % 10 := by rfl    -- 49 % 10 = 9
  have h3 : (7 ^ 3) % 10 = 343 % 10 := by rfl   -- 343 % 10 = 3
  have h4 : (7 ^ 4) % 10 = 2401 % 10 := by rfl  -- 2401 % 10 = 1
  have h_pattern : ∀ k : ℕ, 7 ^ (4 * k) % 10 = 1 := 
    by intro k; cases k; norm_num [pow_succ, mul_comm] -- Pattern repeats every 4
  have h_mod : 6 ^ 5 % 4 = 0 := by
    have h51 : 6 % 4 = 2 := by norm_num
    have h62 : (6 ^ 2) % 4 = 0 := by norm_num
    have h63 : (6 ^ 5) % 4 = (6 * 6 ^ 4) % 4 := by ring_exp
    rw [← h62, h51]; norm_num
  exact h_pattern (6 ^ 5 / 4) -- Using the repetition pattern

end units_digit_7_power_l49_49909


namespace original_deck_size_l49_49144

noncomputable def initial_red_probability (r b : ℕ) : Prop := r / (r + b) = 1 / 4
noncomputable def added_black_probability (r b : ℕ) : Prop := r / (r + (b + 6)) = 1 / 6

theorem original_deck_size (r b : ℕ) 
  (h1 : initial_red_probability r b) 
  (h2 : added_black_probability r b) : 
  r + b = 12 := 
sorry

end original_deck_size_l49_49144


namespace equal_real_roots_quadratic_l49_49599

theorem equal_real_roots_quadratic (k : ℝ) : (∀ x : ℝ, (x^2 + 2*x + k = 0)) → k = 1 :=
by
sorry

end equal_real_roots_quadratic_l49_49599


namespace cistern_fill_time_l49_49265

theorem cistern_fill_time (hF : ∀ (F : ℝ), F = 1 / 3)
                         (hE : ∀ (E : ℝ), E = 1 / 5) : 
  ∃ (t : ℝ), t = 15 / 2 :=
by
  sorry

end cistern_fill_time_l49_49265


namespace sphere_shot_radius_l49_49941

theorem sphere_shot_radius (R : ℝ) (N : ℕ) (π : ℝ) (r : ℝ) 
  (h₀ : R = 4) (h₁ : N = 64) 
  (h₂ : (4 / 3) * π * (R ^ 3) / ((4 / 3) * π * (r ^ 3)) = N) : 
  r = 1 := 
by
  sorry

end sphere_shot_radius_l49_49941


namespace domain_of_f_l49_49825

noncomputable def f (x : ℝ) := 1 / (Real.log (x + 1)) + Real.sqrt (4 - x)

theorem domain_of_f :
  {x : ℝ | x + 1 > 0 ∧ x + 1 ≠ 1 ∧ 4 - x ≥ 0} = { x : ℝ | (-1 < x ∧ x ≤ 4) ∧ x ≠ 0 } :=
sorry

end domain_of_f_l49_49825


namespace arithmetic_sequence_sum_l49_49073

variable {a : ℕ → ℝ} 

-- Condition: Arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Condition: Given sum of specific terms in the sequence
def given_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 10 = 16

-- Problem: Proving the correct answer
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : given_condition a) :
  a 4 + a 6 + a 8 = 24 :=
by
  sorry

end arithmetic_sequence_sum_l49_49073


namespace complement_intersection_l49_49931

-- Define sets P and Q.
def P : Set ℝ := {x | x ≥ 2}
def Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- Define the complement of P.
def complement_P : Set ℝ := {x | x < 2}

-- The theorem we need to prove.
theorem complement_intersection : complement_P ∩ Q = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end complement_intersection_l49_49931


namespace div_1947_l49_49816

theorem div_1947 (n : ℕ) (hn : n % 2 = 1) : 1947 ∣ (46^n + 296 * 13^n) :=
by
  sorry

end div_1947_l49_49816


namespace shaded_area_of_octagon_l49_49209

def side_length := 12
def octagon_area := 288

theorem shaded_area_of_octagon (s : ℕ) (h0 : s = side_length):
  (2 * s * s - 2 * s * s / 2) * 2 / 2 = octagon_area :=
by
  skip
  sorry

end shaded_area_of_octagon_l49_49209


namespace gridPolygon_side_longer_than_one_l49_49510

-- Define the structure of a grid polygon
structure GridPolygon where
  area : ℕ  -- Area of the grid polygon
  perimeter : ℕ  -- Perimeter of the grid polygon
  no_holes : Prop  -- Polyon does not contain holes

-- Definition of a grid polygon with specific properties
def specificGridPolygon : GridPolygon :=
  { area := 300, perimeter := 300, no_holes := true }

-- The theorem we want to prove that ensures at least one side is longer than 1
theorem gridPolygon_side_longer_than_one (P : GridPolygon) (h_area : P.area = 300) (h_perimeter : P.perimeter = 300) (h_no_holes : P.no_holes) : ∃ side_length : ℝ, side_length > 1 :=
  by
  sorry

end gridPolygon_side_longer_than_one_l49_49510


namespace velociraptor_catch_time_l49_49696

/-- You encounter a velociraptor while out for a stroll. You run to the northeast at 10 m/s 
    with a 3-second head start. The velociraptor runs at 15√2 m/s but only runs either north or east at any given time. 
    Prove that the time until the velociraptor catches you is 6 seconds. -/
theorem velociraptor_catch_time (v_yours : ℝ) (t_head_start : ℝ) (v_velociraptor : ℝ)
  (v_eff : ℝ) (speed_advantage : ℝ) (headstart_distance : ℝ) :
  v_yours = 10 → t_head_start = 3 → v_velociraptor = 15 * Real.sqrt 2 →
  v_eff = 15 → speed_advantage = v_eff - v_yours → headstart_distance = v_yours * t_head_start →
  (headstart_distance / speed_advantage) = 6 :=
by
  sorry

end velociraptor_catch_time_l49_49696


namespace remainder_91_pow_91_mod_100_l49_49830

-- Definitions
def large_power_mod (a b n : ℕ) : ℕ :=
  (a^b) % n

-- Statement
theorem remainder_91_pow_91_mod_100 : large_power_mod 91 91 100 = 91 :=
by
  sorry

end remainder_91_pow_91_mod_100_l49_49830


namespace find_divisor_l49_49300

theorem find_divisor (X : ℕ) (h12 : 12 ∣ (1020 - 12)) (h24 : 24 ∣ (1020 - 12)) (h48 : 48 ∣ (1020 - 12)) (h56 : 56 ∣ (1020 - 12)) :
  X = 63 :=
sorry

end find_divisor_l49_49300


namespace range_of_a_l49_49601

theorem range_of_a (a : ℝ) 
  (h : ¬ ∃ x : ℝ, Real.exp x ≤ 2 * x + a) : a < 2 - 2 * Real.log 2 := 
  sorry

end range_of_a_l49_49601


namespace initial_ducks_l49_49108

theorem initial_ducks (D : ℕ) (h1 : D + 20 = 33) : D = 13 :=
by sorry

end initial_ducks_l49_49108


namespace tens_digit_of_9_pow_1024_l49_49690

theorem tens_digit_of_9_pow_1024 : 
  (9^1024 % 100) / 10 % 10 = 6 := 
sorry

end tens_digit_of_9_pow_1024_l49_49690


namespace maximum_value_expression_l49_49504

theorem maximum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 :=
by
  sorry

end maximum_value_expression_l49_49504


namespace solve_equation_l49_49819

theorem solve_equation : ∀ x : ℝ, 3 * x * (x - 2) = (x - 2) → (x = 2 ∨ x = 1 / 3) :=
by
  intro x
  intro h
  sorry

end solve_equation_l49_49819


namespace solve_system_of_eqns_l49_49651

theorem solve_system_of_eqns :
  ∃ x y : ℝ, (x^2 + x * y + y = 1 ∧ y^2 + x * y + x = 5) ∧ ((x = -1 ∧ y = 3) ∨ (x = -1 ∧ y = -2)) :=
by
  sorry

end solve_system_of_eqns_l49_49651


namespace compare_f_l49_49048

def f (x : ℝ) : ℝ := x^2 + 2*x + 4

theorem compare_f (x1 x2 : ℝ) (h1 : x1 < x2) (h2 : x1 + x2 = 0) : 
  f x1 < f x2 :=
by sorry

end compare_f_l49_49048


namespace ball_bounces_17_times_to_reach_below_2_feet_l49_49389

theorem ball_bounces_17_times_to_reach_below_2_feet:
  ∃ k: ℕ, (∀ n, n < k → (800 * ((2: ℝ) / 3) ^ n) ≥ 2) ∧ (800 * ((2: ℝ) / 3) ^ k < 2) ∧ k = 17 :=
by
  sorry

end ball_bounces_17_times_to_reach_below_2_feet_l49_49389


namespace journey_possibility_l49_49302

noncomputable def possible_start_cities 
  (routes : List (String × String)) 
  (visited : List String) : List String :=
sorry

theorem journey_possibility :
  possible_start_cities 
    [("Saint Petersburg", "Tver"), 
     ("Yaroslavl", "Nizhny Novgorod"), 
     ("Moscow", "Kazan"), 
     ("Nizhny Novgorod", "Kazan"), 
     ("Moscow", "Tver"), 
     ("Moscow", "Nizhny Novgorod")]
    ["Saint Petersburg", "Tver", "Yaroslavl", "Nizhny Novgorod", "Moscow", "Kazan"] 
  = ["Saint Petersburg", "Yaroslavl"] :=
sorry

end journey_possibility_l49_49302


namespace solution_to_system_l49_49061

theorem solution_to_system (x y a b : ℝ) 
  (h1 : x = 1) (h2 : y = 2) 
  (h3 : a * x + b * y = 4) 
  (h4 : b * x - a * y = 7) : 
  a + b = 1 :=
by
  sorry

end solution_to_system_l49_49061


namespace extended_morse_code_symbols_l49_49488

def symbol_count (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 2
  else if n = 3 then 1
  else if n = 4 then 1 + 4 + 1
  else if n = 5 then 1 + 8
  else 0

theorem extended_morse_code_symbols : 
  (symbol_count 1 + symbol_count 2 + symbol_count 3 + symbol_count 4 + symbol_count 5) = 20 :=
by sorry

end extended_morse_code_symbols_l49_49488


namespace streetlights_each_square_l49_49364

-- Define the conditions
def total_streetlights : Nat := 200
def total_squares : Nat := 15
def unused_streetlights : Nat := 20

-- State the question mathematically
def streetlights_installed := total_streetlights - unused_streetlights
def streetlights_per_square := streetlights_installed / total_squares

-- The theorem we need to prove
theorem streetlights_each_square : streetlights_per_square = 12 := sorry

end streetlights_each_square_l49_49364


namespace cards_dealt_to_people_l49_49782

theorem cards_dealt_to_people (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  (∃ k, k = total_people - (total_cards % total_people) ∧ k = 3) := 
by
  sorry

end cards_dealt_to_people_l49_49782


namespace circle_center_radius_l49_49447

theorem circle_center_radius :
  ∃ (h k r : ℝ), (∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 4 ↔ (x - h)^2 + (y - k)^2 = r^2) ∧ h = 2 ∧ k = -1 ∧ r = 2 :=
by
  sorry

end circle_center_radius_l49_49447


namespace simplify_fractions_l49_49649

theorem simplify_fractions :
  (30 / 45) * (75 / 128) * (256 / 150) = 1 / 6 := 
by
  sorry

end simplify_fractions_l49_49649


namespace largest_even_digit_multiple_of_five_l49_49538

theorem largest_even_digit_multiple_of_five : ∃ n : ℕ, n = 8860 ∧ n < 10000 ∧ (∀ digit ∈ (n.digits 10), digit % 2 = 0) ∧ n % 5 = 0 :=
by
  sorry

end largest_even_digit_multiple_of_five_l49_49538


namespace units_digit_of_power_l49_49906

theorem units_digit_of_power (a b : ℕ) : (a % 10 = 7) → (b % 4 = 0) → ((a^b) % 10 = 1) :=
by
  intros
  sorry

end units_digit_of_power_l49_49906


namespace probability_same_value_after_reroll_l49_49229

theorem probability_same_value_after_reroll
  (initial_dice : Fin 6 → Fin 6)
  (rerolled_dice : Fin 4 → Fin 6)
  (initial_pair_num : Fin 6)
  (h_initial_no_four_of_a_kind : ∀ (n : Fin 6), (∃ i j : Fin 6, i ≠ j ∧ initial_dice i = n ∧ initial_dice j = n) →
    ∃ (i₁ i₂ i₃ i₄ : Fin 6), i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₁ ≠ i₄ ∧ i₂ ≠ i₃ ∧ i₂ ≠ i₄ ∧ i₃ ≠ i₄ ∧
    initial_dice i₁ ≠ n ∧ initial_dice i₂ ≠ n ∧ initial_dice i₃ ≠ n ∧ initial_dice i₄ ≠ n)
  (h_initial_pair : ∃ i j : Fin 6, i ≠ j ∧ initial_dice i = initial_pair_num ∧ initial_dice j = initial_pair_num) :
  (671 : ℚ) / 1296 = 671 / 1296 :=
by sorry

end probability_same_value_after_reroll_l49_49229


namespace solve_log_eq_l49_49104

theorem solve_log_eq : ∀ x : ℝ, (2 : ℝ) ^ (Real.log x / Real.log 3) = (1 / 4 : ℝ) → x = 1 / 9 :=
by
  intro x
  sorry

end solve_log_eq_l49_49104


namespace range_of_b_l49_49330

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 4 * x + 4

theorem range_of_b (b : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (f x1 = b) ∧ (f x2 = b) ∧ (f x3 = b))
  ↔ (-4 / 3 < b ∧ b < 28 / 3) :=
by
  sorry

end range_of_b_l49_49330


namespace find_ethanol_percentage_l49_49729

noncomputable def ethanol_percentage_in_fuel_A (P_A : ℝ) (V_A : ℝ) : Prop :=
  (P_A / 100) * V_A + 0.16 * (200 - V_A) = 18

theorem find_ethanol_percentage (P_A : ℝ) (V_A : ℝ) (h₀ : V_A ≤ 200) (h₁ : 0 ≤ V_A) :
  ethanol_percentage_in_fuel_A P_A V_A :=
by
  sorry

end find_ethanol_percentage_l49_49729


namespace complex_modulus_squared_l49_49087

open Complex

theorem complex_modulus_squared (w : ℂ) (h : w^2 + abs w ^ 2 = 7 + 2 * I) : abs w ^ 2 = 53 / 14 :=
sorry

end complex_modulus_squared_l49_49087


namespace employee_n_salary_l49_49702

variable (m n : ℝ)

theorem employee_n_salary 
  (h1 : m + n = 605) 
  (h2 : m = 1.20 * n) : 
  n = 275 :=
by
  sorry

end employee_n_salary_l49_49702


namespace base_length_of_isosceles_triangle_l49_49663

theorem base_length_of_isosceles_triangle (a b : ℕ) 
    (h₁ : a = 8) 
    (h₂ : 2 * a + b = 25) : 
    b = 9 :=
by
  -- This is the proof stub. Proof will be provided here.
  sorry

end base_length_of_isosceles_triangle_l49_49663


namespace polynomial_coefficients_l49_49311

theorem polynomial_coefficients :
  ∃ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ),
  (1 - 2 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 ∧
  a_0 = 1 ∧
  (a_1 + a_2 + a_3 + a_4 + a_5 = -2) ∧
  (a_1 + a_3 + a_5 = -122) :=
begin
  sorry
end

end polynomial_coefficients_l49_49311


namespace area_percent_of_smaller_rectangle_l49_49273

-- Definitions of the main geometric elements and assumptions
def larger_rectangle (w h : ℝ) : Prop := (w > 0) ∧ (h > 0)
def radius_of_circle (w h r : ℝ) : Prop := r = Real.sqrt (w^2 + h^2)
def inscribed_smaller_rectangle (w h x y : ℝ) : Prop := 
  (0 < x) ∧ (x < 1) ∧ (0 < y) ∧ (y < 1) ∧
  ((h + 2 * y * h)^2 + (x * w)^2 = w^2 + h^2)

-- Prove the area percentage relationship
theorem area_percent_of_smaller_rectangle 
  (w h x y : ℝ) 
  (hw : w > 0) (hh : h > 0)
  (hcirc : radius_of_circle w h (Real.sqrt (w^2 + h^2)))
  (hsmall_rect : inscribed_smaller_rectangle w h x y) :
  (4 * x * y) / (4.0 * 1.0) * 100 = 8.33 := sorry

end area_percent_of_smaller_rectangle_l49_49273


namespace coefficient_x9_l49_49430

theorem coefficient_x9 (p : Polynomial ℚ) : 
  p = (1 + 3 * Polynomial.X - Polynomial.X^2)^5 →
  Polynomial.coeff p 9 = 15 := 
by
  intro h
  rw [h]
  -- additional lean tactics to prove the statement would go here
  sorry

end coefficient_x9_l49_49430


namespace same_terminal_side_l49_49706

theorem same_terminal_side (α : ℝ) (k : ℤ) : 
  ∃ k : ℤ, α = k * 360 + 60 → α = -300 := 
by
  sorry

end same_terminal_side_l49_49706


namespace ab_operation_l49_49826

theorem ab_operation (a b : ℤ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (h1 : a + b = 10) (h2 : a * b = 24) : 
  (1 / a + 1 / b) = 5 / 12 :=
by
  sorry

end ab_operation_l49_49826


namespace common_ratio_l49_49089

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem common_ratio (a₁ : ℝ) (h : a₁ ≠ 0) : 
  (∀ S4 S5 S6, S5 = geometric_sum a₁ q 5 ∧ S4 = geometric_sum a₁ q 4 ∧ S6 = geometric_sum a₁ q 6 → 
  2 * S4 = S5 + S6) → 
  q = -2 := 
by
  sorry

end common_ratio_l49_49089


namespace digits_in_base_5_l49_49055

theorem digits_in_base_5 (n : ℕ) (h : n = 1234) (h_largest_power : 5^4 < n ∧ n < 5^5) : 
  ∃ digits : ℕ, digits = 5 := 
sorry

end digits_in_base_5_l49_49055


namespace smallest_d_l49_49326

theorem smallest_d (d : ℕ) (h : 3150 * d = k ^ 2) : d = 14 :=
by
  -- assuming the condition: 3150 = 2 * 3 * 5^2 * 7
  have h_factorization : 3150 = 2 * 3 * 5^2 * 7 := by sorry
  -- based on the computation and verification, the smallest d that satisfies the condition is 14
  sorry

end smallest_d_l49_49326


namespace chocolate_candy_pieces_l49_49384

-- Define the initial number of boxes and the boxes given away
def initial_boxes : Nat := 12
def boxes_given : Nat := 7

-- Define the number of remaining boxes
def remaining_boxes := initial_boxes - boxes_given

-- Define the number of pieces per box
def pieces_per_box : Nat := 6

-- Calculate the total pieces Tom still has
def total_pieces := remaining_boxes * pieces_per_box

-- State the theorem
theorem chocolate_candy_pieces : total_pieces = 30 :=
by
  -- proof steps would go here
  sorry

end chocolate_candy_pieces_l49_49384


namespace num_of_3_digit_nums_with_one_even_digit_l49_49767

def is_even (n : Nat) : Bool :=
  n % 2 == 0

def count_3_digit_nums_with_exactly_one_even_digit : Nat :=
  let even_digits := [0, 2, 4, 6, 8]
  let odd_digits := [1, 3, 5, 7, 9]
  -- Case 1: A is even, B and C are odd
  let case1 := 4 * 5 * 5
  -- Case 2: B is even, A and C are odd
  let case2 := 5 * 5 * 5
  -- Case 3: C is even, A and B are odd
  let case3 := 5 * 5 * 5
  case1 + case2 + case3

theorem num_of_3_digit_nums_with_one_even_digit : count_3_digit_nums_with_exactly_one_even_digit = 350 := by
  sorry

end num_of_3_digit_nums_with_one_even_digit_l49_49767


namespace pardee_road_length_l49_49656

theorem pardee_road_length (t p : ℕ) (h1 : t = 162 * 1000) (h2 : t = p + 150 * 1000) : p = 12 * 1000 :=
by
  -- Proof goes here
  sorry

end pardee_road_length_l49_49656


namespace sequence_inequality_l49_49925

theorem sequence_inequality
  (a : ℕ → ℝ)
  (h₁ : a 1 = 0)
  (h₇ : a 7 = 0) :
  ∃ k : ℕ, k ≤ 5 ∧ a k + a (k + 2) ≤ a (k + 1) * Real.sqrt 3 := 
sorry

end sequence_inequality_l49_49925


namespace evaluate_expression_l49_49573

theorem evaluate_expression :
  (3^2016 + 3^2014 + 3^2012) / (3^2016 - 3^2014 + 3^2012) = 91 / 73 := 
  sorry

end evaluate_expression_l49_49573


namespace product_of_roots_l49_49316

variable {x1 x2 : ℝ}

theorem product_of_roots (h : ∀ x, -x^2 + 3*x = 0 → (x = x1 ∨ x = x2)) :
  x1 * x2 = 0 :=
by
  sorry

end product_of_roots_l49_49316


namespace random_event_l49_49281

theorem random_event (a b : ℝ) (h1 : a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0):
  ¬ (∀ a b, a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0 → a + b < 0) :=
by
  sorry

end random_event_l49_49281


namespace units_digit_7_pow_6_pow_5_l49_49900

def units_digit_of_power (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_6_pow_5 :
  units_digit_of_power (7 ^ (6 ^ 5)) = 1 :=
by
  -- Insert proof steps here
  sorry

end units_digit_7_pow_6_pow_5_l49_49900


namespace calculate_ff2_l49_49213

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x + 4

theorem calculate_ff2 : f (f 2) = 5450 := by
  sorry

end calculate_ff2_l49_49213


namespace min_value_expression_l49_49636

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
    9 ≤ (5 * z / (2 * x + y) + 5 * x / (y + 2 * z) + 2 * y / (x + z) + (x + y + z) / (x * y + y * z + z * x)) :=
sorry

end min_value_expression_l49_49636


namespace freshmen_count_l49_49109

theorem freshmen_count (n : ℕ) (h1 : n < 600) (h2 : n % 17 = 16) (h3 : n % 19 = 18) : n = 322 := 
by 
  sorry

end freshmen_count_l49_49109


namespace units_digit_of_power_l49_49907

theorem units_digit_of_power (a b : ℕ) : (a % 10 = 7) → (b % 4 = 0) → ((a^b) % 10 = 1) :=
by
  intros
  sorry

end units_digit_of_power_l49_49907


namespace percentage_commute_l49_49468

variable (x : Real)
variable (h : 0.20 * 0.10 * x = 12)

theorem percentage_commute :
  0.10 * 0.20 * x = 12 :=
by
  sorry

end percentage_commute_l49_49468


namespace build_bridge_l49_49804

/-- It took 6 days for 60 workers, all working together at the same rate, to build a bridge.
    Prove that if only 30 workers had been available, it would have taken 12 total days to build the bridge. -/
theorem build_bridge (days_60_workers : ℕ) (num_60_workers : ℕ) (same_rate : Prop) : 
  (days_60_workers = 6) → (num_60_workers = 60) → (same_rate = ∀ n m, n * days_60_workers = m * days_30_workers) → (days_30_workers = 12) :=
by
  sorry

end build_bridge_l49_49804


namespace solve_equation_l49_49650

theorem solve_equation (x : ℝ) : x*(x-3)^2*(5+x) = 0 ↔ x = 0 ∨ x = 3 ∨ x = -5 := 
by 
  sorry

end solve_equation_l49_49650


namespace units_digit_pow_7_6_5_l49_49899

theorem units_digit_pow_7_6_5 :
  let units_digit (n : ℕ) : ℕ := n % 10
  in units_digit (7 ^ (6 ^ 5)) = 9 :=
by
  let units_digit (n : ℕ) := n % 10
  sorry

end units_digit_pow_7_6_5_l49_49899


namespace age_of_first_man_replaced_l49_49522

theorem age_of_first_man_replaced (x : ℕ) (avg_before : ℝ) : avg_before * 15 + 30 = avg_before * 15 + 74 - (x + 23) → (37 * 2 - (x + 23) = 30) → x = 21 :=
sorry

end age_of_first_man_replaced_l49_49522


namespace triangle_angles_are_30_60_90_l49_49748

theorem triangle_angles_are_30_60_90
  (a b c OH R r : ℝ)
  (h1 : OH = c / 2)
  (h2 : OH = a)
  (h3 : a < b)
  (h4 : b < c)
  (h5 : a + b > c)
  (h6 : a + c > b)
  (h7 : b + c > a) :
  ∃ (A B C : ℝ), (A = π / 6 ∧ B = π / 3 ∧ C = π / 2) :=
sorry

end triangle_angles_are_30_60_90_l49_49748


namespace determine_ordered_triple_l49_49808

open Real

theorem determine_ordered_triple (a b c : ℝ) (h₁ : 5 < a) (h₂ : 5 < b) (h₃ : 5 < c) 
  (h₄ : (a + 3)^2 / (b + c - 3) + (b + 6)^2 / (c + a - 6) + (c + 9)^2 / (a + b - 9) = 81) : 
  a = 15 ∧ b = 12 ∧ c = 9 := 
sorry

end determine_ordered_triple_l49_49808


namespace Miriam_gave_brother_60_marbles_l49_49643

def Miriam_current_marbles : ℕ := 30
def Miriam_initial_marbles : ℕ := 300
def brother_marbles (B : ℕ) : Prop := B = 60
def sister_marbles (B : ℕ) : ℕ := 2 * B
def friend_marbles : ℕ := 90
def total_given_away_marbles (B : ℕ) : ℕ := B + sister_marbles B + friend_marbles

theorem Miriam_gave_brother_60_marbles (B : ℕ) 
    (h1 : Miriam_current_marbles = 30) 
    (h2 : Miriam_initial_marbles = 300)
    (h3 : total_given_away_marbles B = Miriam_initial_marbles - Miriam_current_marbles) : 
    brother_marbles B :=
by 
    sorry

end Miriam_gave_brother_60_marbles_l49_49643


namespace popsicles_consumed_l49_49359

def total_minutes (hours : ℕ) (additional_minutes : ℕ) : ℕ :=
  hours * 60 + additional_minutes

def popsicles_in_time (total_time : ℕ) (interval : ℕ) : ℕ :=
  total_time / interval

theorem popsicles_consumed : popsicles_in_time (total_minutes 4 30) 15 = 18 :=
by
  -- The proof is omitted
  sorry

end popsicles_consumed_l49_49359


namespace inequality_holds_if_and_only_if_l49_49063

variable (x : ℝ) (b : ℝ)

theorem inequality_holds_if_and_only_if (hx : |x-5| + |x-3| + |x-2| < b) : b > 4 :=
sorry

end inequality_holds_if_and_only_if_l49_49063


namespace arith_expression_evaluation_l49_49740

theorem arith_expression_evaluation :
  2 + (1/6:ℚ) + (((4.32:ℚ) - 1.68 - (1 + 8/25:ℚ)) * (5/11:ℚ) - (2/7:ℚ)) / (1 + 9/35:ℚ) = 2 + 101/210 := by
  sorry

end arith_expression_evaluation_l49_49740


namespace hours_worked_l49_49211

theorem hours_worked (w e : ℝ) (hw : w = 6.75) (he : e = 67.5) 
  : e / w = 10 := by
  sorry

end hours_worked_l49_49211


namespace abs_neg_six_l49_49385

theorem abs_neg_six : abs (-6) = 6 := by
  sorry

end abs_neg_six_l49_49385


namespace sixth_element_row_20_l49_49256

theorem sixth_element_row_20 : (Nat.choose 20 5) = 15504 := by
  sorry

end sixth_element_row_20_l49_49256


namespace find_a_l49_49477

theorem find_a (a : ℝ) :
  (∃ b : ℝ, 4 * b + 3 = 7 ∧ 5 * (-b) - 1 = 2 * (-b) + a) → a = -4 :=
by
  sorry

end find_a_l49_49477


namespace scientific_notation_of_million_l49_49220

theorem scientific_notation_of_million : 1000000 = 10^6 :=
by
  sorry

end scientific_notation_of_million_l49_49220


namespace f_inequality_l49_49971

-- Definition of odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- f is an odd function
variable {f : ℝ → ℝ}
variable (h1 : is_odd_function f)

-- f has a period of 4
variable (h2 : ∀ x, f (x + 4) = f x)

-- f is monotonically increasing on [0, 2)
variable (h3 : ∀ x y, 0 ≤ x → x < y → y < 2 → f x < f y)

theorem f_inequality : f 3 < 0 ∧ 0 < f 1 :=
by 
  -- Place proof here
  sorry

end f_inequality_l49_49971


namespace exam_percentage_l49_49066

theorem exam_percentage (x : ℝ) (h_cond : 100 - x >= 0 ∧ x >= 0 ∧ 60 * x + 90 * (100 - x) = 69 * 100) : x = 70 := by
  sorry

end exam_percentage_l49_49066


namespace cylinder_dimensions_l49_49836

theorem cylinder_dimensions (r_sphere : ℝ) (r_cylinder h d : ℝ)
  (h_d_eq : h = d) (r_sphere_val : r_sphere = 6) 
  (sphere_area_eq : 4 * Real.pi * r_sphere^2 = 2 * Real.pi * r_cylinder * h) :
  h = 12 ∧ d = 12 :=
by 
  sorry

end cylinder_dimensions_l49_49836


namespace cos_neg_3pi_plus_alpha_l49_49044

/-- Given conditions: 
  1. 𝚌𝚘𝚜(3π/2 + α) = -3/5,
  2. α is an angle in the fourth quadrant,
Prove: cos(-3π + α) = -4/5 -/
theorem cos_neg_3pi_plus_alpha (α : Real) (h1 : Real.cos (3 * Real.pi / 2 + α) = -3 / 5) (h2 : 0 ≤ α ∧ α < 2 * Real.pi ∧ Real.sin α < 0) :
  Real.cos (-3 * Real.pi + α) = -4 / 5 := 
sorry

end cos_neg_3pi_plus_alpha_l49_49044


namespace count_distinct_reals_a_with_integer_roots_l49_49026

-- Define the quadratic equation with its roots and conditions
theorem count_distinct_reals_a_with_integer_roots :
  ∃ (a_vals : Finset ℝ), a_vals.card = 6 ∧
    (∀ a ∈ a_vals, ∃ r s : ℤ, 
      (r + s : ℝ) = -a ∧ (r * s : ℝ) = 9 * a) :=
by
  sorry

end count_distinct_reals_a_with_integer_roots_l49_49026


namespace fathers_age_more_than_three_times_son_l49_49827

variable (F S x : ℝ)

theorem fathers_age_more_than_three_times_son :
  F = 27 →
  F = 3 * S + x →
  F + 3 = 2 * (S + 3) + 8 →
  x = 3 :=
by
  intros hF h1 h2
  sorry

end fathers_age_more_than_three_times_son_l49_49827


namespace first_player_wins_l49_49921

-- Definitions of dominos and moves
structure Domino :=
  (a : Nat)
  (b : Nat)

structure Game :=
  (dominos : Finset Domino)
  (chain : List Domino)
  (current_player : Bool) -- true for player 1, false for player 2

-- Assumptions needed
def valid_move (g : Game) (d : Domino) : Prop :=
  d ∈ g.dominos ∧ 
  (g.chain = [] ∨
   (d.a = g.chain.head.b ∨ d.b = g.chain.head.b ∨
    d.a = g.chain.last.a ∨ d.b = g.chain.last.a))

-- Move function to update the game state
def make_move (g : Game) (d : Domino) : Game :=
  { g with
    dominos := g.dominos.erase d,
    chain := if g.chain = [] then [d] else if d.a = g.chain.head.b then d :: g.chain else if d.b = g.chain.last.a then g.chain ++ [d] else g.chain,
    current_player := bnot g.current_player }

-- Winning strategy theorem
theorem first_player_wins (game : Game) :
  (∃ d : Domino, valid_move game d ∧ game.current_player = true) ∧
  (∀ g' : Game, (valid_move game d ∧ g' = make_move game d) → (∃ d' : Domino, valid_move g' d' ∧ g'.current_player = false)) :=
sorry

end first_player_wins_l49_49921


namespace money_weed_eating_l49_49221

-- Define the amounts and conditions
def money_mowing : ℕ := 68
def money_per_week : ℕ := 9
def weeks : ℕ := 9
def total_money : ℕ := money_per_week * weeks

-- Define the proof that the money made weed eating is 13 dollars
theorem money_weed_eating :
  total_money - money_mowing = 13 := sorry

end money_weed_eating_l49_49221


namespace book_cost_l49_49374

theorem book_cost (x y : ℝ) (h₁ : 2 * y = x) (h₂ : 100 + y = x - 100) : x = 200 := by
  sorry

end book_cost_l49_49374


namespace maximum_surface_area_of_inscribed_sphere_in_right_triangular_prism_l49_49074

open Real

theorem maximum_surface_area_of_inscribed_sphere_in_right_triangular_prism 
  (a b : ℝ)
  (ha : a^2 + b^2 = 25) 
  (AC_eq_5 : AC = 5) :
  ∃ (r : ℝ), 4 * π * r^2 = 25 * (3 - 3 * sqrt 2) * π :=
sorry

end maximum_surface_area_of_inscribed_sphere_in_right_triangular_prism_l49_49074


namespace evaluate_expression_l49_49165

noncomputable def cuberoot (x : ℝ) : ℝ := x ^ (1 / 3)

theorem evaluate_expression : 
  cuberoot (1 + 27) * cuberoot (1 + cuberoot 27) = cuberoot 112 := 
by 
  sorry

end evaluate_expression_l49_49165


namespace find_original_number_l49_49409

theorem find_original_number (x : ℝ)
  (h : (((x + 3) * 3 - 3) / 3) = 10) : x = 8 :=
sorry

end find_original_number_l49_49409


namespace field_ratio_l49_49370

theorem field_ratio (side pond_area_ratio : ℝ) (field_length : ℝ) 
  (pond_is_square: pond_area_ratio = 1/18) 
  (side_length: side = 8) 
  (field_len: field_length = 48) : 
  (field_length / (pond_area_ratio * side ^ 2 / side)) = 2 :=
by
  sorry

end field_ratio_l49_49370


namespace solve_parallelogram_l49_49833

variables (x y : ℚ)

def condition1 : Prop := 6 * y - 2 = 12 * y - 10
def condition2 : Prop := 4 * x + 5 = 8 * x + 1

theorem solve_parallelogram : condition1 y → condition2 x → x + y = 7 / 3 :=
by
  intros h1 h2
  sorry

end solve_parallelogram_l49_49833


namespace brogan_total_red_apples_l49_49008

def red_apples (total_apples percentage_red : ℕ) : ℕ :=
  (total_apples * percentage_red) / 100

theorem brogan_total_red_apples :
  red_apples 20 40 + red_apples 20 50 = 18 :=
by
  sorry

end brogan_total_red_apples_l49_49008


namespace min_weighings_to_determine_counterfeit_l49_49591

/-- 
  Given 2023 coins with two counterfeit coins and 2021 genuine coins, 
  and using a balance scale, determine whether the counterfeit coins 
  are heavier or lighter. Prove that the minimum number of weighings 
  required is 3. 
-/
theorem min_weighings_to_determine_counterfeit (n : ℕ) (k : ℕ) (l : ℕ) 
  (h : n = 2023) (h₁ : k = 2) (h₂ : l = 2021) 
  (w₁ w₂ : ℕ → ℝ) -- weights of coins
  (h_fake : ∀ i j, w₁ i = w₁ j) -- counterfeits have same weight
  (h_fake_diff : ∀ i j, i ≠ j → w₁ i ≠ w₂ j) -- fake different from genuine
  (h_genuine : ∀ i j, w₂ i = w₂ j) -- genuines have same weight
  (h_total : ∀ i, i ≤ l + k) -- total coins condition
  : ∃ min_weighings : ℕ, min_weighings = 3 :=
by
  sorry

end min_weighings_to_determine_counterfeit_l49_49591


namespace number_of_valid_4_digit_integers_l49_49605

/-- 
Prove that the number of 4-digit positive integers that satisfy the following conditions:
1. Each of the first two digits must be 2, 3, or 5.
2. The last two digits cannot be the same.
3. Each of the last two digits must be 4, 6, or 9.
is equal to 54.
-/
theorem number_of_valid_4_digit_integers : 
  ∃ n : ℕ, n = 54 ∧ 
  ∀ d1 d2 d3 d4 : ℕ, 
    (d1 = 2 ∨ d1 = 3 ∨ d1 = 5) ∧ 
    (d2 = 2 ∨ d2 = 3 ∨ d2 = 5) ∧ 
    (d3 = 4 ∨ d3 = 6 ∨ d3 = 9) ∧ 
    (d4 = 4 ∨ d4 = 6 ∨ d4 = 9) ∧ 
    (d3 ≠ d4) → 
    n = 54 := 
sorry

end number_of_valid_4_digit_integers_l49_49605


namespace compute_expression_l49_49062

variables (a b c : ℝ)

theorem compute_expression (h1 : a - b = 2) (h2 : a + c = 6) : 
  (2 * a + b + c) - 2 * (a - b - c) = 12 :=
by
  sorry

end compute_expression_l49_49062


namespace bathroom_area_is_eight_l49_49863

def bathroomArea (length width : ℕ) : ℕ :=
  length * width

theorem bathroom_area_is_eight : bathroomArea 4 2 = 8 := 
by
  -- Proof omitted.
  sorry

end bathroom_area_is_eight_l49_49863


namespace find_num_large_envelopes_l49_49154

def numLettersInSmallEnvelopes : Nat := 20
def totalLetters : Nat := 150
def totalLettersInMediumLargeEnvelopes := totalLetters - numLettersInSmallEnvelopes -- 130
def lettersPerLargeEnvelope : Nat := 5
def lettersPerMediumEnvelope : Nat := 3
def numLargeEnvelopes (L : Nat) : Prop := 5 * L + 6 * L = totalLettersInMediumLargeEnvelopes

theorem find_num_large_envelopes : ∃ L : Nat, numLargeEnvelopes L ∧ L = 11 := by
  sorry

end find_num_large_envelopes_l49_49154


namespace positive_difference_of_expressions_l49_49539

theorem positive_difference_of_expressions :
  let a := 8
  let expr1 := (a^2 - a^2) / a
  let expr2 := (a^2 * a^2) / a
  expr1 = 0 → expr2 = 512 → 512 - 0 = 512 := 
by
  introv h_expr1 h_expr2
  rw [h_expr1, h_expr2]
  norm_num
  exact rfl

end positive_difference_of_expressions_l49_49539


namespace units_digit_7_pow_6_pow_5_l49_49914

theorem units_digit_7_pow_6_pow_5 : (7 ^ (6 ^ 5)) % 10 = 1 :=
by
  -- Using the cyclic pattern of the units digits of powers of 7: 7, 9, 3, 1
  have h1 : 7 % 10 = 7, by norm_num,
  have h2 : (7 ^ 2) % 10 = 9, by norm_num,
  have h3 : (7 ^ 3) % 10 = 3, by norm_num,
  have h4 : (7 ^ 4) % 10 = 1, by norm_num,

  -- Calculate 6 ^ 5 and the modular position
  have h6_5 : (6 ^ 5) % 4 = 0, by norm_num,

  -- Therefore, 7 ^ (6 ^ 5) % 10 = 7 ^ 0 % 10 because the cycle is 4
  have h_final : (7 ^ (6 ^ 5 % 4)) % 10 = (7 ^ 0) % 10, by rw h6_5,
  have h_zero : (7 ^ 0) % 10 = 1, by norm_num,

  rw h_final,
  exact h_zero,

end units_digit_7_pow_6_pow_5_l49_49914


namespace num_triangles_with_perimeter_9_l49_49934

theorem num_triangles_with_perimeter_9 : 
  ∃! (S : Finset (ℕ × ℕ × ℕ)), 
  S.card = 6 ∧ 
  (∀ (a b c : ℕ), (a, b, c) ∈ S → a + b + c = 9 ∧ a + b > c ∧ b + c > a ∧ a + c > b ∧ a ≤ b ∧ b ≤ c) := 
sorry

end num_triangles_with_perimeter_9_l49_49934


namespace city_roads_different_colors_l49_49618

-- Definitions and conditions
def Intersection (α : Type) := α × α × α

def City (α : Type) :=
  { intersections : α → Intersection α // 
    ∀ i : α, ∃ c₁ c₂ c₃ : α, intersections i = (c₁, c₂, c₃) 
    ∧ c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₃ ≠ c₁ 
  }

variables {α : Type}

-- Statement to prove that the three roads leading out of the city have different colors
theorem city_roads_different_colors (c : City α) 
  (roads_outside : α → Prop)
  (h : ∃ r₁ r₂ r₃, roads_outside r₁ ∧ roads_outside r₂ ∧ roads_outside r₃ ∧ 
  r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₃ ≠ r₁) : 
  true := 
sorry

end city_roads_different_colors_l49_49618


namespace quarters_value_percentage_l49_49549

theorem quarters_value_percentage (dimes_count quarters_count dimes_value quarters_value : ℕ) (h1 : dimes_count = 75)
    (h2 : quarters_count = 30) (h3 : dimes_value = 10) (h4 : quarters_value = 25) :
    (quarters_count * quarters_value * 100) / (dimes_count * dimes_value + quarters_count * quarters_value) = 50 := 
by
    sorry

end quarters_value_percentage_l49_49549


namespace arithmetic_seq_a7_l49_49070

theorem arithmetic_seq_a7 (a : ℕ → ℤ) (d : ℤ)
  (h1 : a 1 = 2)
  (h2 : a 3 + a 5 = 10)
  (h3 : ∀ n, a (n + 1) = a n + d) : a 7 = 8 :=
sorry

end arithmetic_seq_a7_l49_49070


namespace rebus_decrypt_correct_l49_49422

-- Definitions
def is_digit (d : ℕ) : Prop := 0 ≤ d ∧ d ≤ 9
def is_odd (d : ℕ) : Prop := is_digit d ∧ d % 2 = 1
def is_even (d : ℕ) : Prop := is_digit d ∧ d % 2 = 0

-- Variables representing ċharacters H, Ч (C), A, D, Y, E, F, B, K
variables (H C A D Y E F B K : ℕ)

-- Conditions
axiom H_odd : is_odd H
axiom C_even : is_even C
axiom A_even : is_even A
axiom D_odd : is_odd D
axiom Y_even : is_even Y
axiom E_even : is_even E
axiom F_odd : is_odd F
axiom B_digit : is_digit B
axiom K_odd : is_odd K

-- Correct answers
def H_val : ℕ := 5
def C_val : ℕ := 3
def A_val : ℕ := 2
def D_val : ℕ := 9
def Y_val : ℕ := 8
def E_val : ℕ := 8
def F_val : ℕ := 5
def B_any : ℕ := B
def K_val : ℕ := 5

-- Proof statement
theorem rebus_decrypt_correct : 
  H = H_val ∧
  C = C_val ∧
  A = A_val ∧
  D = D_val ∧
  Y = Y_val ∧
  E = E_val ∧
  F = F_val ∧
  K = K_val :=
sorry

end rebus_decrypt_correct_l49_49422


namespace island_perimeter_l49_49731

-- Defining the properties of the island
def width : ℕ := 4
def length : ℕ := 7

-- The main theorem stating the condition to be proved
theorem island_perimeter : 2 * (length + width) = 22 := by
  sorry

end island_perimeter_l49_49731


namespace sasha_remaining_questions_l49_49100

theorem sasha_remaining_questions
  (qph : ℕ) (total_questions : ℕ) (hours_worked : ℕ)
  (h_qph : qph = 15) (h_total_questions : total_questions = 60) (h_hours_worked : hours_worked = 2) :
  total_questions - (qph * hours_worked) = 30 :=
by
  sorry

end sasha_remaining_questions_l49_49100


namespace triangle_area_l49_49722

theorem triangle_area (a b c : ℕ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) (h₄ : a^2 + b^2 = c^2) : 
  (1 / 2 : ℝ) * a * b = 54 := 
by
  rw [h₁, h₂, h₃]
  -- The proof goes here
  sorry

end triangle_area_l49_49722


namespace subcommittees_with_experts_l49_49371

def total_members : ℕ := 12
def experts : ℕ := 5
def non_experts : ℕ := total_members - experts
def subcommittee_size : ℕ := 5

theorem subcommittees_with_experts :
  (nat.choose total_members subcommittee_size) - (nat.choose non_experts subcommittee_size) = 771 := by
  sorry

end subcommittees_with_experts_l49_49371


namespace cards_dealt_problem_l49_49779

theorem cards_dealt_problem (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  let cards_per_person := total_cards / total_people,
      extra_cards := total_cards % total_people,
      people_with_extra_card := extra_cards,
      people_with_fewer_cards := total_people - people_with_extra_card
  in people_with_fewer_cards = 3 :=
by
  sorry

end cards_dealt_problem_l49_49779


namespace people_with_fewer_than_7_cards_l49_49790

theorem people_with_fewer_than_7_cards (total_cards : ℕ) (total_people : ℕ) (cards_per_person : ℕ) (remainder : ℕ) 
  (h : total_cards = total_people * cards_per_person + remainder)
  (h_cards : total_cards = 60)
  (h_people : total_people = 9)
  (h_cards_per_person : cards_per_person = 6)
  (h_remainder : remainder = 6) :
  (total_people - remainder) = 3 :=
by
  sorry

end people_with_fewer_than_7_cards_l49_49790


namespace combined_capacity_is_40_l49_49986

/-- Define the bus capacity as 1/6 the train capacity -/
def bus_capacity (train_capacity : ℕ) := train_capacity / 6

/-- There are two buses in the problem -/
def number_of_buses := 2

/-- The train capacity given in the problem is 120 people -/
def train_capacity := 120

/-- The combined capacity of the two buses is -/
def combined_bus_capacity := number_of_buses * bus_capacity train_capacity

/-- Proof that the combined capacity of the two buses is 40 people -/
theorem combined_capacity_is_40 : combined_bus_capacity = 40 := by
  -- Proof will be filled in here
  sorry

end combined_capacity_is_40_l49_49986


namespace salary_restoration_l49_49875

theorem salary_restoration (S : ℝ) : 
  let reduced_salary := 0.7 * S
  let restore_factor := 1 / 0.7
  let percentage_increase := restore_factor - 1
  percentage_increase * 100 = 42.857 :=
by
  sorry

end salary_restoration_l49_49875


namespace find_n_values_l49_49918

theorem find_n_values : {n : ℕ | n ≥ 1 ∧ n ≤ 6 ∧ ∃ a b c : ℤ, a^n + b^n = c^n + n} = {1, 2, 3} :=
by sorry

end find_n_values_l49_49918


namespace find_circle_center_l49_49583

-- Definition of the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 - 6*x + y^2 + 10*y - 7 = 0

-- The main statement to prove
theorem find_circle_center :
  (∃ center : ℝ × ℝ, center = (3, -5) ∧ ∀ x y : ℝ, circle_eq x y ↔ (x - 3)^2 + (y + 5)^2 = 41) :=
sorry

end find_circle_center_l49_49583


namespace total_apples_bought_l49_49342

def apples_bought_by_Junhyeok := 7 * 16
def apples_bought_by_Jihyun := 6 * 25

theorem total_apples_bought : apples_bought_by_Junhyeok + apples_bought_by_Jihyun = 262 := by
  sorry

end total_apples_bought_l49_49342


namespace cookie_portion_l49_49091

theorem cookie_portion :
  ∃ (total_cookies : ℕ) (remaining_cookies : ℕ) (cookies_senior_ate : ℕ) (cookies_senior_took_second_day : ℕ) 
    (cookies_senior_put_back : ℕ) (cookies_junior_took : ℕ),
  total_cookies = 22 ∧
  remaining_cookies = 11 ∧
  cookies_senior_ate = 3 ∧
  cookies_senior_took_second_day = 3 ∧
  cookies_senior_put_back = 2 ∧
  cookies_junior_took = 7 ∧
  4 / 22 = 2 / 11 :=
by
  existsi 22, 11, 3, 3, 2, 7
  sorry

end cookie_portion_l49_49091


namespace hexagon_diagonals_sum_correct_l49_49270

noncomputable def hexagon_diagonals_sum : ℝ :=
  let AB := 40
  let S := 100
  let AC := 140
  let AD := 240
  let AE := 340
  AC + AD + AE

theorem hexagon_diagonals_sum_correct : hexagon_diagonals_sum = 720 :=
  by
  show hexagon_diagonals_sum = 720
  sorry

end hexagon_diagonals_sum_correct_l49_49270


namespace actual_distance_is_correct_l49_49644

noncomputable def actual_distance_in_meters (scale : ℕ) (map_distance_cm : ℝ) : ℝ :=
  (map_distance_cm * scale) / 100

theorem actual_distance_is_correct
  (scale : ℕ)
  (map_distance_cm : ℝ)
  (h_scale : scale = 3000000)
  (h_map_distance : map_distance_cm = 4) :
  actual_distance_in_meters scale map_distance_cm = 1.2 * 10^5 :=
by
  sorry

end actual_distance_is_correct_l49_49644


namespace arc_length_calc_l49_49207

-- Defining the conditions
def circle_radius := 12 -- radius OR
def angle_RIP := 30 -- angle in degrees

-- Defining the goal
noncomputable def arc_length_RP := 4 * Real.pi -- length of arc RP

-- The statement to prove
theorem arc_length_calc :
  arc_length_RP = 4 * Real.pi :=
sorry

end arc_length_calc_l49_49207


namespace intersection_M_N_l49_49321

def set_M : Set ℝ := { x | x < 2 }
def set_N : Set ℝ := { x | x > 0 }
def set_intersection : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem intersection_M_N : set_M ∩ set_N = set_intersection := 
by
  sorry

end intersection_M_N_l49_49321


namespace quadratic_equation_factored_form_correct_l49_49280

theorem quadratic_equation_factored_form_correct :
  ∀ x : ℝ, (x^2 - 4 * x - 1 = 0) → (x - 2)^2 = 5 :=
by
  intros x h
  sorry

end quadratic_equation_factored_form_correct_l49_49280


namespace number_half_reduction_l49_49248

/-- Define the conditions -/
def percentage_more (percent : Float) (amount : Float) : Float := amount + (percent / 100) * amount

theorem number_half_reduction (x : Float) : percentage_more 30 75 = 97.5 → (x / 2) = 97.5 → x = 195 := by
  intros h1 h2
  sorry

end number_half_reduction_l49_49248


namespace prime_product_is_2009_l49_49038

theorem prime_product_is_2009 (a b c : ℕ) 
  (h_primeA : Prime a) 
  (h_primeB : Prime b) 
  (h_primeC : Prime c)
  (h_div1 : a ∣ (b + 8)) 
  (h_div2a : a ∣ (b^2 - 1)) 
  (h_div2c : c ∣ (b^2 - 1)) 
  (h_sum : b + c = a^2 - 1) : 
  a * b * c = 2009 := 
sorry

end prime_product_is_2009_l49_49038


namespace percent_sparrows_not_pigeons_l49_49068

-- Definitions of percentages
def crows_percent : ℝ := 0.20
def sparrows_percent : ℝ := 0.40
def pigeons_percent : ℝ := 0.15
def doves_percent : ℝ := 0.25

-- The statement to prove
theorem percent_sparrows_not_pigeons :
  (sparrows_percent / (1 - pigeons_percent)) = 0.47 :=
by
  sorry

end percent_sparrows_not_pigeons_l49_49068


namespace vasya_hits_ship_l49_49097

theorem vasya_hits_ship (board_size : ℕ) (ship_length : ℕ) (shots : ℕ) : 
  board_size = 10 ∧ ship_length = 4 ∧ shots = 24 → ∃ strategy : Fin board_size × Fin board_size → Prop, 
  (∀ pos, strategy pos → pos.1 * board_size + pos.2 < shots) ∧ 
  ∀ (ship_pos : Fin board_size × Fin board_size) (horizontal : Bool), 
  ∃ shot_pos, strategy shot_pos ∧ 
  (if horizontal then 
    ship_pos.1 = shot_pos.1 ∧ ship_pos.2 ≤ shot_pos.2 ∧ shot_pos.2 < ship_pos.2 + ship_length 
  else 
    ship_pos.2 = shot_pos.2 ∧ ship_pos.1 ≤ shot_pos.1 ∧ shot_pos.1 < ship_pos.1 + ship_length) :=
sorry

end vasya_hits_ship_l49_49097


namespace math_proof_problem_l49_49346

noncomputable def find_value (a b c : ℝ) : ℝ :=
  (a^3 + b^3 + c^3) / (a * b * c * (a * b + a * c + b * c))

theorem math_proof_problem (a b c : ℝ)
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 0) (h5 : a * b + a * c + b * c ≠ 0) :
  find_value a b c = 3 :=
by 
  -- sorry is used as we are only asked to provide the theorem statement in Lean.
  sorry

end math_proof_problem_l49_49346


namespace bakery_storage_l49_49382

theorem bakery_storage (S F B : ℕ) 
  (h1 : S * 4 = F * 5) 
  (h2 : F = 10 * B) 
  (h3 : F * 1 = (B + 60) * 8) : S = 3000 :=
sorry

end bakery_storage_l49_49382


namespace parabola_standard_equation_l49_49683

theorem parabola_standard_equation (x y : ℝ) :
  (3 * x - 4 * y - 12 = 0) →
  (y^2 = 16 * x ∨ x^2 = -12 * y) :=
sorry

end parabola_standard_equation_l49_49683


namespace shirts_sold_l49_49129

theorem shirts_sold (S : ℕ) (H_total : 69 = 7 * 7 + 5 * S) : S = 4 :=
by
  sorry -- Placeholder for the proof

end shirts_sold_l49_49129


namespace find_m_condition_l49_49060

theorem find_m_condition (m : ℕ) (h : 9^4 = 3^(2*m)) : m = 4 := by
  sorry

end find_m_condition_l49_49060


namespace vehicle_distribution_l49_49868

theorem vehicle_distribution :
  ∃ B T U : ℕ, 2 * B + 3 * T + U = 18 ∧ ∀ n : ℕ, n ≤ 18 → ∃ t : ℕ, ∃ (u : ℕ), 2 * (n - t) + u = 18 ∧ 2 * Nat.gcd t u + 3 * t + u = 18 ∧
  10 + 8 + 7 + 5 + 4 + 2 + 1 = 37 := by
  sorry

end vehicle_distribution_l49_49868


namespace modulus_of_complex_l49_49448

open Complex

theorem modulus_of_complex (z : ℂ) (h : (1 + z) / (1 - z) = ⟨0, 1⟩) : abs z = 1 := 
sorry

end modulus_of_complex_l49_49448


namespace stationery_store_sales_l49_49954

theorem stationery_store_sales :
  let price_pencil_eraser := 0.8
  let price_regular_pencil := 0.5
  let price_short_pencil := 0.4
  let num_pencil_eraser := 200
  let num_regular_pencil := 40
  let num_short_pencil := 35
  (num_pencil_eraser * price_pencil_eraser) +
  (num_regular_pencil * price_regular_pencil) +
  (num_short_pencil * price_short_pencil) = 194 :=
by
  sorry

end stationery_store_sales_l49_49954


namespace extremum_of_cubic_function_l49_49369

noncomputable def cubic_function (x : ℝ) : ℝ := 2 - x^2 - x^3

theorem extremum_of_cubic_function : 
  ∃ x_max x_min : ℝ, 
    cubic_function x_max = x_max_value ∧ 
    cubic_function x_min = x_min_value ∧ 
    ∀ x : ℝ, cubic_function x ≤ cubic_function x_max ∧ cubic_function x_min ≤ cubic_function x :=
sorry

end extremum_of_cubic_function_l49_49369


namespace roots_negative_reciprocals_l49_49132

theorem roots_negative_reciprocals (a b c r s : ℝ) (h1 : a * r^2 + b * r + c = 0)
    (h2 : a * s^2 + b * s + c = 0) (h3 : r = -1 / s) (h4 : s = -1 / r) :
    a = -c :=
by
  -- Insert clever tricks to auto-solve or reuse axioms here
  sorry

end roots_negative_reciprocals_l49_49132


namespace landscaping_charges_l49_49197

theorem landscaping_charges
    (x : ℕ)
    (h : 63 * x + 9 * 11 + 10 * 9 = 567) :
  x = 6 :=
by
  sorry

end landscaping_charges_l49_49197


namespace num_real_values_for_integer_roots_l49_49033

theorem num_real_values_for_integer_roots : 
  (∃ (a : ℝ), ∀ (r s : ℤ), r + s = -a ∧ r * s = 9 * a) → ∃ (n : ℕ), n = 10 :=
by
  sorry

end num_real_values_for_integer_roots_l49_49033


namespace correct_divisor_l49_49067

theorem correct_divisor (dividend incorrect_divisor quotient correct_quotient correct_divisor : ℕ) 
  (h1 : incorrect_divisor = 63) 
  (h2 : quotient = 24) 
  (h3 : correct_quotient = 42) 
  (h4 : dividend = incorrect_divisor * quotient) 
  (h5 : dividend / correct_divisor = correct_quotient) : 
  correct_divisor = 36 := 
by 
  sorry

end correct_divisor_l49_49067


namespace find_abc_l49_49096

theorem find_abc (a b c : ℤ) 
  (h₁ : a^4 - 2 * b^2 = a)
  (h₂ : b^4 - 2 * c^2 = b)
  (h₃ : c^4 - 2 * a^2 = c)
  (h₄ : a + b + c = -3) : 
  a = -1 ∧ b = -1 ∧ c = -1 := 
sorry

end find_abc_l49_49096


namespace ellipse_eq_elliptic_line_exist_l49_49183

-- Definition of ellipse
def ellipse_eq (a b : ℝ) : ℝ × ℝ → Prop :=
λ p => (p.fst ^ 2 / a ^ 2) + (p.snd ^ 2 / b ^ 2) = 1

-- Given conditions
def focal_condition : Prop :=
  let c := 2 in
  let a := 2 * c in
  let b := 2 * Real.sqrt 3 in
  a^2 - b^2 = c^2 ∧ a = 4 ∧ b = 2 * Real.sqrt 3

-- Problem 1
theorem ellipse_eq_elliptic :
  focal_condition →
  ∃ (a b : ℝ), a = 4 ∧ b = 2 * Real.sqrt 3 ∧ ellipse_eq a b = ellipse_eq 4 (2 * Real.sqrt 3) :=
by
  -- Note that the exact proof steps shall be included here when doing the actual proof in Lean
  sorry

-- Condition for line passing through E
def line_through_point (k : ℝ) (x y : ℝ) : Prop :=
  ∃ (l : ℝ → ℝ), l = λ x => k * x - 4 ∧ ∃ (r t : ℝ × ℝ), ellipse_eq 4 (2 * Real.sqrt 3) r ∧ ellipse_eq 4 (2 * Real.sqrt 3) t ∧ r.fst * t.fst + r.snd * t.snd = 16 / 7

-- Problem 2
theorem line_exist :
  ∃ l : ℝ → ℝ, line_through_point 1 0 (-4) ∧ line_through_point (-1) 0 (-4) :=
by
  -- Note that the exact proof steps shall be included here when doing the actual proof in Lean
  -- These represent the equations 'x + y + 4 = 0' and 'x - y - 4 = 0'
  sorry

end ellipse_eq_elliptic_line_exist_l49_49183


namespace area_of_rectangle_is_588_l49_49142

-- Define the conditions
def radius_of_circle := 7
def width_of_rectangle := 2 * radius_of_circle
def length_to_width_ratio := 3

-- Define the width and length of the rectangle based on the conditions
def width := width_of_rectangle
def length := length_to_width_ratio * width_of_rectangle

-- Define the area of the rectangle
def area_of_rectangle := length * width

-- The theorem to prove
theorem area_of_rectangle_is_588 : area_of_rectangle = 588 :=
by sorry -- Proof is not required

end area_of_rectangle_is_588_l49_49142


namespace segments_count_bound_l49_49212

-- Define the overall setup of the problem
variable (n : ℕ) (points : Finset ℕ)

-- The main hypothesis and goal
theorem segments_count_bound (hn : n ≥ 2) (hpoints : points.card = 3 * n) :
  ∃ A B : Finset (ℕ × ℕ), (∀ (i j : ℕ), i ∈ points → j ∈ points → i ≠ j → ((i, j) ∈ A ↔ (i, j) ∉ B)) ∧
  ∀ (X : Finset ℕ) (hX : X.card = n), ∃ C : Finset (ℕ × ℕ), (C ⊆ A) ∧ (X ⊆ points) ∧
  (∃ count : ℕ, count ≥ (n - 1) / 6 ∧ count = C.card ∧ ∀ (a b : ℕ), (a, b) ∈ C → a ∈ X ∧ b ∈ points \ X) := sorry

end segments_count_bound_l49_49212


namespace nat_number_solution_odd_l49_49678

theorem nat_number_solution_odd (x y z : ℕ) (h : x + y + z = 100) : 
  ∃ P : ℕ, P = 49 ∧ P % 2 = 1 := 
sorry

end nat_number_solution_odd_l49_49678


namespace unique_zero_point_of_quadratic_l49_49472

theorem unique_zero_point_of_quadratic (a : ℝ) :
  (∀ x : ℝ, (a * x^2 - x - 1 = 0 → x = -1)) ↔ (a = 0 ∨ a = -1 / 4) :=
by
  sorry

end unique_zero_point_of_quadratic_l49_49472


namespace roof_area_l49_49831

theorem roof_area (l w : ℝ) 
  (h1 : l = 4 * w) 
  (h2 : l - w = 28) : 
  l * w = 3136 / 9 := 
by 
  sorry

end roof_area_l49_49831


namespace factorize_expression_l49_49168

theorem factorize_expression (x : ℝ) : 2 * x - x^2 = x * (2 - x) := sorry

end factorize_expression_l49_49168


namespace fantasia_max_capacity_reach_l49_49334

def acre_per_person := 1
def land_acres := 40000
def base_population := 500
def population_growth_factor := 4
def years_per_growth_period := 20

def maximum_capacity := land_acres / acre_per_person

def population_at_time (years_from_2000 : ℕ) : ℕ :=
  base_population * population_growth_factor^(years_from_2000 / years_per_growth_period)

theorem fantasia_max_capacity_reach :
  ∃ t : ℕ, t = 60 ∧ population_at_time t = maximum_capacity := by sorry

end fantasia_max_capacity_reach_l49_49334


namespace percent_value_in_quarters_l49_49547

theorem percent_value_in_quarters (num_dimes num_quarters : ℕ) 
  (value_dime value_quarter total_value value_in_quarters : ℕ) 
  (h1 : num_dimes = 75)
  (h2 : num_quarters = 30)
  (h3 : value_dime = num_dimes * 10)
  (h4 : value_quarter = num_quarters * 25)
  (h5 : total_value = value_dime + value_quarter)
  (h6 : value_in_quarters = num_quarters * 25) :
  (value_in_quarters / total_value) * 100 = 50 :=
by
  sorry

end percent_value_in_quarters_l49_49547


namespace find_c_eq_3_l49_49581

theorem find_c_eq_3 (m b c : ℝ) :
  (∀ x y, y = m * x + c → ((x = b + 4 ∧ y = 5) ∨ (x = -2 ∧ y = 2))) →
  c = 3 :=
by
  sorry

end find_c_eq_3_l49_49581


namespace cost_price_of_radio_l49_49670

-- Define the conditions
def selling_price : ℝ := 1335
def loss_percentage : ℝ := 0.11

-- Define what we need to prove
theorem cost_price_of_radio (C : ℝ) (h1 : selling_price = 0.89 * C) : C = 1500 :=
by
  -- This is where we would put the proof, but we can leave it as a sorry for now.
  sorry

end cost_price_of_radio_l49_49670


namespace problem_l49_49020

-- Define the problem
theorem problem {a b c : ℤ} (h1 : a = c + 1) (h2 : b - 1 = a) :
  (a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2 = 6 := 
sorry

end problem_l49_49020


namespace fraction_of_problems_solved_by_Andrey_l49_49812

theorem fraction_of_problems_solved_by_Andrey (N x : ℕ) 
  (h1 : 0 < N) 
  (h2 : x = N / 2)
  (Boris_solves : ∀ y : ℕ, y = N - x → y / 3 = (N - x) / 3)
  (remaining_problems : ∀ y : ℕ, y = (N - x) - (N - x) / 3 → y = 2 * (N - x) / 3) 
  (Viktor_solves : (2 * (N - x) / 3 = N / 3)) :
  x / N = 1 / 2 := 
by {
  sorry
}

end fraction_of_problems_solved_by_Andrey_l49_49812


namespace tetrahedron_ineq_l49_49957

variable (P Q R S : ℝ)

-- Given conditions
axiom ortho_condition : S^2 = P^2 + Q^2 + R^2

theorem tetrahedron_ineq (P Q R S : ℝ) (ortho_condition : S^2 = P^2 + Q^2 + R^2) :
  (P + Q + R) / S ≤ Real.sqrt 3 := by
  sorry

end tetrahedron_ineq_l49_49957


namespace min_value_cx_plus_dy_squared_l49_49184

theorem min_value_cx_plus_dy_squared
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (∃ (x y : ℝ), a * x^2 + b * y^2 = 1 ∧ ∀ (x y : ℝ), a * x^2 + b * y^2 = 1 → c * x + d * y^2 ≥ -c / a.sqrt) :=
sorry

end min_value_cx_plus_dy_squared_l49_49184


namespace intersection_proof_l49_49763

-- Definitions of sets M and N
def M : Set ℝ := { x | x^2 < 4 }
def N : Set ℝ := { x | x < 1 }

-- The intersection of M and N
def intersection : Set ℝ := { x | -2 < x ∧ x < 1 }

-- Proposition to prove
theorem intersection_proof : M ∩ N = intersection :=
by sorry

end intersection_proof_l49_49763


namespace count_primes_between_30_and_50_l49_49456

-- Define the range of numbers from 30 to 50
def range_30_to_50 := Set.of_list (List.range' 30 (51 - 30))

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Extract all prime numbers in the specified range
def primes_between_30_and_50 : List ℕ :=
  List.filter is_prime (Set.toList range_30_to_50)

theorem count_primes_between_30_and_50 : primes_between_30_and_50.length = 5 :=
by
  -- The proof goes here
  sorry

end count_primes_between_30_and_50_l49_49456


namespace tan_13pi_div_3_eq_sqrt_3_l49_49301

theorem tan_13pi_div_3_eq_sqrt_3 : Real.tan (13 * Real.pi / 3) = Real.sqrt 3 :=
  sorry

end tan_13pi_div_3_eq_sqrt_3_l49_49301


namespace num_correct_conclusions_l49_49761

-- Definitions and conditions from the problem
variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}
variable (n : ℕ)
variable (hSn_eq : S n + S (n + 1) = n ^ 2)

-- Assert the conditions described in the comments
theorem num_correct_conclusions (hSn_eq : ∀ n, S n + S (n + 1) = n ^ 2) :
  (1:ℕ) = 3 ↔
  (-- Conclusion 1
   ¬(∀ n, a (n + 2) - a n = 2) ∧
   -- Conclusion 2: If a_1 = 0, then S_50 = 1225
   (S 50 = 1225) ∧
   -- Conclusion 3: If a_1 = 1, then S_50 = 1224
   (S 50 = 1224) ∧
   -- Conclusion 4: Monotonically increasing sequence
   (∀ a_1, (-1/4 : ℚ) < a_1 ∧ a_1 < 1/4)) :=
by
  sorry

end num_correct_conclusions_l49_49761


namespace cards_dealt_l49_49773

theorem cards_dealt (total_cards : ℕ) (num_people : ℕ) (fewer_cards : ℕ) :
  total_cards = 60 → num_people = 9 → fewer_cards = 3 →
  ∃ k : ℕ, total_cards = num_people * k + 6 ∧ k = 6 ∧ 
  (num_people - 6 = fewer_cards) :=
by
  intros h1 h2 h3
  use 6
  split;
  sorry

end cards_dealt_l49_49773


namespace axis_of_symmetry_shift_l49_49243

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem axis_of_symmetry_shift (f : ℝ → ℝ) (hf : is_even_function f) :
  (∃ a : ℝ, ∀ x : ℝ, f (x + 1) = f (-(x + 1))) :=
sorry

end axis_of_symmetry_shift_l49_49243


namespace joey_pills_sum_one_week_l49_49624

def joey_pills (n : ℕ) : ℕ :=
  1 + 2 * n

theorem joey_pills_sum_one_week : 
  (joey_pills 0) + (joey_pills 1) + (joey_pills 2) + (joey_pills 3) + (joey_pills 4) + (joey_pills 5) + (joey_pills 6) = 49 :=
by
  sorry

end joey_pills_sum_one_week_l49_49624


namespace max_value_of_expression_l49_49315

theorem max_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 8) : 
  (1 + x) * (1 + y) ≤ 25 :=
by
  sorry

end max_value_of_expression_l49_49315


namespace sin_cos_fourth_power_l49_49497

theorem sin_cos_fourth_power (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 4) : Real.sin θ ^ 4 + Real.cos θ ^ 4 = 63 / 64 :=
by
  sorry

end sin_cos_fourth_power_l49_49497


namespace reconstruct_quadrilateral_l49_49043

def quadrilateralVectors (W W' X X' Y Y' Z Z' : ℝ) :=
  (W - Z = W/2 + Z'/2) ∧
  (X - Y = Y'/2 + X'/2) ∧
  (Y - X = Y'/2 + X'/2) ∧
  (Z - W = W/2 + Z'/2)

theorem reconstruct_quadrilateral (W W' X X' Y Y' Z Z' : ℝ) :
  quadrilateralVectors W W' X X' Y Y' Z Z' →
  W = (1/2) * W' + 0 * X' + 0 * Y' + (1/2) * Z' :=
sorry

end reconstruct_quadrilateral_l49_49043


namespace inscribed_sphere_radius_eq_l49_49822

noncomputable def inscribed_sphere_radius (b α : ℝ) : ℝ :=
  b * (Real.sin α) / (4 * (Real.cos (α / 4))^2)

theorem inscribed_sphere_radius_eq
  (b α : ℝ) 
  (h1 : 0 < b)
  (h2 : 0 < α ∧ α < Real.pi) 
  : inscribed_sphere_radius b α = b * (Real.sin α) / (4 * (Real.cos (α / 4))^2) :=
sorry

end inscribed_sphere_radius_eq_l49_49822


namespace find_number_l49_49684

theorem find_number (x : ℝ) (h₁ : |x| + 1/x = 0) (h₂ : x ≠ 0) : x = -1 :=
sorry

end find_number_l49_49684


namespace convex_polygon_intersection_l49_49856

theorem convex_polygon_intersection {ABCD : Set (ℝ × ℝ)} 
  (h_square : ∀ x y ∈ ABCD, ∃ z w : ℝ, z ∈ Icc 0 1 ∧ w ∈ Icc 0 1 ∧ (x, y) = (z, w)) 
  {M : Set (ℝ × ℝ)} (hM_subset : M ⊆ ABCD) (hM_convex : convex ℝ M) (hM_area : measure_theory.measure.l_integral measure_theory.measure_space.volume (indicator M (λ _, 1)) > 1 / 2) :
  ∃ l : ℝ → ℝ, ∃ a : ℝ, parallel_to_side_ABCD l AB ∧ (M ∩ {p : ℝ × ℝ | p.2 = l a}).count > 1 / 2 := 
sorry

end convex_polygon_intersection_l49_49856


namespace johns_weight_l49_49082

-- Definitions based on the given conditions
def max_weight : ℝ := 1000
def safety_percentage : ℝ := 0.20
def bar_weight : ℝ := 550

-- Theorem stating the mathematically equivalent proof problem
theorem johns_weight : 
  (johns_safe_weight : ℝ) = max_weight - safety_percentage * max_weight 
  → (johns_safe_weight - bar_weight = 250) :=
by
  sorry

end johns_weight_l49_49082
