import Mathlib

namespace NUMINAMATH_GPT_inequality_system_no_solution_l1209_120932

theorem inequality_system_no_solution (k x : ℝ) (h₁ : 1 < x ∧ x ≤ 2) (h₂ : x > k) : k ≥ 2 :=
sorry

end NUMINAMATH_GPT_inequality_system_no_solution_l1209_120932


namespace NUMINAMATH_GPT_y_A_functional_relationship_y_B_functional_relationship_cost_effective_B_cost_effective_equal_cost_effective_A_l1209_120965

-- Definitions of cost functions for travel agencies
def full_ticket_price : ℕ := 240

def y_A (x : ℕ) : ℕ := 120 * x + 240
def y_B (x : ℕ) : ℕ := 144 * x + 144

-- Prove functional relationships for y_A and y_B
theorem y_A_functional_relationship (x : ℕ) : y_A x = 120 * x + 240 :=
by sorry

theorem y_B_functional_relationship (x : ℕ) : y_B x = 144 * x + 144 :=
by sorry

-- Prove conditions for cost-effectiveness
theorem cost_effective_B (x : ℕ) : x < 4 → y_A x > y_B x :=
by sorry

theorem cost_effective_equal (x : ℕ) : x = 4 → y_A x = y_B x :=
by sorry

theorem cost_effective_A (x : ℕ) : x > 4 → y_A x < y_B x :=
by sorry

end NUMINAMATH_GPT_y_A_functional_relationship_y_B_functional_relationship_cost_effective_B_cost_effective_equal_cost_effective_A_l1209_120965


namespace NUMINAMATH_GPT_combined_selling_price_l1209_120913

theorem combined_selling_price :
  let cost_price_A := 180
  let profit_percent_A := 0.15
  let cost_price_B := 220
  let profit_percent_B := 0.20
  let cost_price_C := 130
  let profit_percent_C := 0.25
  let selling_price_A := cost_price_A * (1 + profit_percent_A)
  let selling_price_B := cost_price_B * (1 + profit_percent_B)
  let selling_price_C := cost_price_C * (1 + profit_percent_C)
  selling_price_A + selling_price_B + selling_price_C = 633.50 := by
  sorry

end NUMINAMATH_GPT_combined_selling_price_l1209_120913


namespace NUMINAMATH_GPT_distance_A_to_B_is_64_yards_l1209_120915

theorem distance_A_to_B_is_64_yards :
  let south1 := 50
  let west := 80
  let north := 20
  let east := 30
  let south2 := 10
  let net_south := south1 + south2 - north
  let net_west := west - east
  let distance := Real.sqrt ((net_south ^ 2) + (net_west ^ 2))
  distance = 64 :=
  by
  let south1 := 50
  let west := 80
  let north := 20
  let east := 30
  let south2 := 10
  let net_south := south1 + south2 - north
  let net_west := west - east
  let distance := Real.sqrt ((net_south ^ 2) + (net_west ^ 2))
  sorry

end NUMINAMATH_GPT_distance_A_to_B_is_64_yards_l1209_120915


namespace NUMINAMATH_GPT_only_valid_set_is_b_l1209_120985

def can_form_triangle (a b c : Nat) : Prop :=
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

theorem only_valid_set_is_b :
  can_form_triangle 2 3 4 ∧ 
  ¬ can_form_triangle 1 2 3 ∧
  ¬ can_form_triangle 3 4 9 ∧
  ¬ can_form_triangle 2 2 4 := by
  sorry

end NUMINAMATH_GPT_only_valid_set_is_b_l1209_120985


namespace NUMINAMATH_GPT_min_value_y1_y2_sq_l1209_120910

theorem min_value_y1_y2_sq (k : ℝ) (y1 y2 : ℝ) :
  ∃ y1 y2, y1 + y2 = 4 / k ∧ y1 * y2 = -4 ∧ y1^2 + y2^2 = 8 :=
sorry

end NUMINAMATH_GPT_min_value_y1_y2_sq_l1209_120910


namespace NUMINAMATH_GPT_initial_velocity_is_three_l1209_120949

-- Define the displacement function s(t)
def s (t : ℝ) : ℝ := 3 * t - t ^ 2

-- Define the initial time condition
def initial_time : ℝ := 0

-- State the main theorem about the initial velocity
theorem initial_velocity_is_three : (deriv s) initial_time = 3 :=
by
  sorry

end NUMINAMATH_GPT_initial_velocity_is_three_l1209_120949


namespace NUMINAMATH_GPT_nalani_fraction_sold_is_3_over_8_l1209_120970

-- Definitions of conditions
def num_dogs : ℕ := 2
def puppies_per_dog : ℕ := 10
def total_amount_received : ℕ := 3000
def price_per_puppy : ℕ := 200

-- Calculation of total puppies and sold puppies
def total_puppies : ℕ := num_dogs * puppies_per_dog
def puppies_sold : ℕ := total_amount_received / price_per_puppy

-- Fraction of puppies sold
def fraction_sold : ℚ := puppies_sold / total_puppies

theorem nalani_fraction_sold_is_3_over_8 :
  fraction_sold = 3 / 8 :=
sorry

end NUMINAMATH_GPT_nalani_fraction_sold_is_3_over_8_l1209_120970


namespace NUMINAMATH_GPT_sum_of_coordinates_of_intersection_l1209_120958

def h : ℝ → ℝ := -- Define h(x). This would be specific to the function provided; we abstract it here for the proof.
sorry

theorem sum_of_coordinates_of_intersection (a b : ℝ) (h_eq: h a = h (a - 5)) : a + b = 6 :=
by
  -- We need a [step from the problem conditions], hence introducing the given conditions
  have : b = h a := sorry
  have : b = h (a - 5) := sorry
  exact sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_intersection_l1209_120958


namespace NUMINAMATH_GPT_inequality_holds_l1209_120971

theorem inequality_holds (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) : a^2 + b^2 ≥ 2 :=
sorry

end NUMINAMATH_GPT_inequality_holds_l1209_120971


namespace NUMINAMATH_GPT_polynomial_product_equals_expected_result_l1209_120978

-- Define the polynomials
def polynomial_product (x : ℝ) : ℝ := (x + 1) * (x^2 - x + 1)

-- Define the expected result of the product
def expected_result (x : ℝ) : ℝ := x^3 + 1

-- The main theorem to prove
theorem polynomial_product_equals_expected_result (x : ℝ) : polynomial_product x = expected_result x :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_polynomial_product_equals_expected_result_l1209_120978


namespace NUMINAMATH_GPT_find_a_and_b_l1209_120934

theorem find_a_and_b (a b c : ℝ) (h1 : a = 6 - b) (h2 : c^2 = a * b - 9) : a = 3 ∧ b = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_and_b_l1209_120934


namespace NUMINAMATH_GPT_area_enclosed_is_one_third_l1209_120924

theorem area_enclosed_is_one_third :
  ∫ x in (0:ℝ)..1, (x^(1/2) - x^2 : ℝ) = (1/3 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_area_enclosed_is_one_third_l1209_120924


namespace NUMINAMATH_GPT_exponent_fraction_law_l1209_120979

theorem exponent_fraction_law :
  (2 ^ 2017 + 2 ^ 2013) / (2 ^ 2017 - 2 ^ 2013) = 17 / 15 :=
  sorry

end NUMINAMATH_GPT_exponent_fraction_law_l1209_120979


namespace NUMINAMATH_GPT_max_property_l1209_120931

noncomputable def f : ℚ → ℚ := sorry

axiom f_zero : f 0 = 0
axiom f_pos_of_nonzero : ∀ α : ℚ, α ≠ 0 → f α > 0
axiom f_mul : ∀ α β : ℚ, f (α * β) = f α * f β
axiom f_add : ∀ α β : ℚ, f (α + β) ≤ f α + f β
axiom f_bounded_by_1989 : ∀ m : ℤ, f m ≤ 1989

theorem max_property (α β : ℚ) (h : f α ≠ f β) : f (α + β) = max (f α) (f β) := sorry

end NUMINAMATH_GPT_max_property_l1209_120931


namespace NUMINAMATH_GPT_sum_remainders_eq_two_l1209_120998

theorem sum_remainders_eq_two (a b c : ℤ) (h_a : a % 24 = 10) (h_b : b % 24 = 4) (h_c : c % 24 = 12) :
  (a + b + c) % 24 = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_remainders_eq_two_l1209_120998


namespace NUMINAMATH_GPT_evaluate_expression_l1209_120925

theorem evaluate_expression : (1 / (1 - 1 / (3 + 1 / 4))) = (13 / 9) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1209_120925


namespace NUMINAMATH_GPT_inequality_solution_l1209_120917

theorem inequality_solution (x : ℝ) : 
  (x - 1) / (2 * x + 1) ≤ 0 ↔ -1 / 2 < x ∧ x ≤ 1 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l1209_120917


namespace NUMINAMATH_GPT_complement_intersection_l1209_120959

open Set

variable (U : Set ℤ) (A B : Set ℤ)

theorem complement_intersection (hU : U = univ)
                               (hA : A = {3, 4})
                               (h_union : A ∪ B = {1, 2, 3, 4}) :
  (U \ A) ∩ B = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1209_120959


namespace NUMINAMATH_GPT_x_axis_intercept_of_line_l1209_120909

theorem x_axis_intercept_of_line (x : ℝ) : (∃ x, 2*x + 1 = 0) → x = - 1 / 2 :=
  by
    intro h
    obtain ⟨x, h1⟩ := h
    have : 2 * x + 1 = 0 := h1
    linarith [this]

end NUMINAMATH_GPT_x_axis_intercept_of_line_l1209_120909


namespace NUMINAMATH_GPT_train_length_proof_l1209_120966

noncomputable def train_length (speed_kmph : ℕ) (time_seconds : ℕ) : ℝ :=
  (speed_kmph * 1000 / 3600) * time_seconds

theorem train_length_proof : train_length 100 18 = 500.04 :=
  sorry

end NUMINAMATH_GPT_train_length_proof_l1209_120966


namespace NUMINAMATH_GPT_expression_evaluation_l1209_120964

theorem expression_evaluation :
  1 - (2 - (3 - 4 - (5 - 6))) = -1 :=
sorry

end NUMINAMATH_GPT_expression_evaluation_l1209_120964


namespace NUMINAMATH_GPT_comic_books_left_l1209_120975

theorem comic_books_left (total : ℕ) (sold : ℕ) (left : ℕ) (h1 : total = 90) (h2 : sold = 65) :
  left = total - sold → left = 25 := by
  sorry

end NUMINAMATH_GPT_comic_books_left_l1209_120975


namespace NUMINAMATH_GPT_boxes_left_l1209_120940

-- Define the initial number of boxes
def initial_boxes : ℕ := 10

-- Define the number of boxes sold
def boxes_sold : ℕ := 5

-- Define a theorem stating that the number of boxes left is 5
theorem boxes_left : initial_boxes - boxes_sold = 5 :=
by
  sorry

end NUMINAMATH_GPT_boxes_left_l1209_120940


namespace NUMINAMATH_GPT_slices_with_both_toppings_l1209_120933

theorem slices_with_both_toppings :
  ∀ (h p b : ℕ),
  (h + b = 9) ∧ (p + b = 12) ∧ (h + p + b = 15) → b = 6 :=
by
  sorry

end NUMINAMATH_GPT_slices_with_both_toppings_l1209_120933


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l1209_120997

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) (d : ℤ) (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_common_diff : d = 2) (h_geom : a 2 ^ 2 = a 1 * a 5) : 
  a 2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l1209_120997


namespace NUMINAMATH_GPT_max_ratio_of_two_digit_numbers_with_mean_55_l1209_120923

theorem max_ratio_of_two_digit_numbers_with_mean_55 (x y : ℕ) (h1 : 10 ≤ x) (h2 : x ≤ 99) (h3 : 10 ≤ y) (h4 : y ≤ 99) (h5 : (x + y) / 2 = 55) : x / y ≤ 9 :=
sorry

end NUMINAMATH_GPT_max_ratio_of_two_digit_numbers_with_mean_55_l1209_120923


namespace NUMINAMATH_GPT_cost_to_treat_dog_l1209_120927

variable (D : ℕ)
variable (cost_cat : ℕ := 40)
variable (num_dogs : ℕ := 20)
variable (num_cats : ℕ := 60)
variable (total_paid : ℕ := 3600)

theorem cost_to_treat_dog : 20 * D + 60 * cost_cat = total_paid → D = 60 := by
  intros h
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cost_to_treat_dog_l1209_120927


namespace NUMINAMATH_GPT_scientific_notation_of_3100000_l1209_120922

theorem scientific_notation_of_3100000 :
  ∃ (a : ℝ) (n : ℤ), 3100000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.1 ∧ n = 6 :=
  sorry

end NUMINAMATH_GPT_scientific_notation_of_3100000_l1209_120922


namespace NUMINAMATH_GPT_num_neg_values_of_x_l1209_120947

theorem num_neg_values_of_x 
  (n : ℕ) 
  (xn_pos_int : ∃ k, n = k ∧ k > 0) 
  (sqrt_x_169_pos_int : ∀ x, ∃ m, x + 169 = m^2 ∧ m > 0) :
  ∃ count, count = 12 := 
by
  sorry

end NUMINAMATH_GPT_num_neg_values_of_x_l1209_120947


namespace NUMINAMATH_GPT_min_elements_in_AS_l1209_120956

theorem min_elements_in_AS (n : ℕ) (h : n ≥ 2) (S : Finset ℝ) (h_card : S.card = n) :
  ∃ (A_S : Finset ℝ), ∀ T : Finset ℝ, (∀ a b : ℝ, a ≠ b → a ∈ S → b ∈ S → (a + b) / 2 ∈ T) → 
  T.card ≥ 2 * n - 3 :=
sorry

end NUMINAMATH_GPT_min_elements_in_AS_l1209_120956


namespace NUMINAMATH_GPT_max_ab_value_l1209_120968

noncomputable def max_ab (a b : ℝ) : ℝ :=
  if (a > 0 ∧ b > 0 ∧ 2 * a + b = 1) then a * b else 0

theorem max_ab_value (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : 2 * a + b = 1) :
  max_ab a b = 1 / 8 := sorry

end NUMINAMATH_GPT_max_ab_value_l1209_120968


namespace NUMINAMATH_GPT_city_mpg_l1209_120951

-- Definitions
def total_distance := 256.2 -- total distance in miles
def total_gallons := 21.0 -- total gallons of gasoline

-- Theorem statement
theorem city_mpg : total_distance / total_gallons = 12.2 :=
by sorry

end NUMINAMATH_GPT_city_mpg_l1209_120951


namespace NUMINAMATH_GPT_find_R_when_S_is_five_l1209_120981

theorem find_R_when_S_is_five (g : ℚ) :
  (∀ (S : ℚ), R = g * S^2 - 5) →
  (R = 25 ∧ S = 3) →
  R = (250 / 3) - 5 :=
by 
  sorry

end NUMINAMATH_GPT_find_R_when_S_is_five_l1209_120981


namespace NUMINAMATH_GPT_range_of_z_l1209_120908

theorem range_of_z (a b : ℝ) (h1 : 2 < a) (h2 : a < 3) (h3 : -2 < b) (h4 : b < -1) :
  5 < 2 * a - b ∧ 2 * a - b < 8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_z_l1209_120908


namespace NUMINAMATH_GPT_no_real_x_solution_l1209_120918

open Real

-- Define the conditions.
def log_defined (x : ℝ) : Prop :=
  0 < x + 5 ∧ 0 < x - 3 ∧ 0 < x^2 - 7*x - 18

-- Define the equation to prove.
def log_eqn (x : ℝ) : Prop :=
  log (x + 5) + log (x - 3) = log (x^2 - 7*x - 18)

-- The mathematicall equivalent proof problem.
theorem no_real_x_solution : ¬∃ x : ℝ, log_defined x ∧ log_eqn x :=
by
  sorry

end NUMINAMATH_GPT_no_real_x_solution_l1209_120918


namespace NUMINAMATH_GPT_keesha_total_cost_is_correct_l1209_120967

noncomputable def hair_cost : ℝ := 
  let cost := 50.0 
  let discount := cost * 0.10 
  let discounted_cost := cost - discount 
  let tip := discounted_cost * 0.20 
  discounted_cost + tip

noncomputable def nails_cost : ℝ := 
  let manicure_cost := 30.0 
  let pedicure_cost := 35.0 * 0.50 
  let total_without_tip := manicure_cost + pedicure_cost 
  let tip := total_without_tip * 0.20 
  total_without_tip + tip

noncomputable def makeup_cost : ℝ := 
  let cost := 40.0 
  let tax := cost * 0.07 
  let total_without_tip := cost + tax 
  let tip := total_without_tip * 0.20 
  total_without_tip + tip

noncomputable def facial_cost : ℝ := 
  let cost := 60.0 
  let discount := cost * 0.15 
  let discounted_cost := cost - discount 
  let tip := discounted_cost * 0.20 
  discounted_cost + tip

noncomputable def total_cost : ℝ := 
  hair_cost + nails_cost + makeup_cost + facial_cost

theorem keesha_total_cost_is_correct : total_cost = 223.56 := by
  sorry

end NUMINAMATH_GPT_keesha_total_cost_is_correct_l1209_120967


namespace NUMINAMATH_GPT_mark_age_l1209_120907

-- Definitions based on the conditions in the problem
variables (M J P : ℕ)  -- Current ages of Mark, John, and their parents respectively

-- Condition definitions
def condition1 : Prop := J = M - 10
def condition2 : Prop := P = 5 * J
def condition3 : Prop := P - 22 = M

-- The theorem to prove the correct answer
theorem mark_age : condition1 M J ∧ condition2 J P ∧ condition3 P M → M = 18 := by
  sorry

end NUMINAMATH_GPT_mark_age_l1209_120907


namespace NUMINAMATH_GPT_sum_of_ages_l1209_120999

-- Definitions from the problem conditions
def Maria_age : ℕ := 14
def age_difference_between_Jose_and_Maria : ℕ := 12
def Jose_age : ℕ := Maria_age + age_difference_between_Jose_and_Maria

-- To be proven: sum of their ages is 40
theorem sum_of_ages : Maria_age + Jose_age = 40 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_sum_of_ages_l1209_120999


namespace NUMINAMATH_GPT_final_result_l1209_120901

def a : ℕ := 2548
def b : ℕ := 364
def hcd := Nat.gcd a b
def result := hcd + 8 - 12

theorem final_result : result = 360 := by
  sorry

end NUMINAMATH_GPT_final_result_l1209_120901


namespace NUMINAMATH_GPT_boys_neither_happy_nor_sad_l1209_120900

theorem boys_neither_happy_nor_sad : 
  (∀ children total happy sad neither boys girls happy_boys sad_girls : ℕ,
    total = 60 →
    happy = 30 →
    sad = 10 →
    neither = 20 →
    boys = 19 →
    girls = 41 →
    happy_boys = 6 →
    sad_girls = 4 →
    (boys - (happy_boys + (sad - sad_girls))) = 7) :=
by
  intros children total happy sad neither boys girls happy_boys sad_girls
  sorry

end NUMINAMATH_GPT_boys_neither_happy_nor_sad_l1209_120900


namespace NUMINAMATH_GPT_area_difference_8_7_area_difference_9_8_l1209_120939

-- Define the side lengths of the tablets
def side_length_7 : ℕ := 7
def side_length_8 : ℕ := 8
def side_length_9 : ℕ := 9

-- Define the areas of the tablets
def area_7 := side_length_7 * side_length_7
def area_8 := side_length_8 * side_length_8
def area_9 := side_length_9 * side_length_9

-- Prove the differences in area
theorem area_difference_8_7 : area_8 - area_7 = 15 := by sorry
theorem area_difference_9_8 : area_9 - area_8 = 17 := by sorry

end NUMINAMATH_GPT_area_difference_8_7_area_difference_9_8_l1209_120939


namespace NUMINAMATH_GPT_total_stickers_l1209_120989

theorem total_stickers (r s t : ℕ) (h1 : r = 30) (h2 : s = 3 * r) (h3 : t = s + 20) : r + s + t = 230 :=
by sorry

end NUMINAMATH_GPT_total_stickers_l1209_120989


namespace NUMINAMATH_GPT_probability_of_rolling_five_l1209_120983

theorem probability_of_rolling_five (total_outcomes : ℕ) (favorable_outcomes : ℕ) 
  (h1 : total_outcomes = 6) (h2 : favorable_outcomes = 1) : 
  favorable_outcomes / total_outcomes = (1 / 6 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_rolling_five_l1209_120983


namespace NUMINAMATH_GPT_circles_intersect_l1209_120938

def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x = 0
def circle2_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * y = 0

theorem circles_intersect :
  ∃ x y : ℝ, circle1_eq x y ∧ circle2_eq x y := by
  sorry

end NUMINAMATH_GPT_circles_intersect_l1209_120938


namespace NUMINAMATH_GPT_number_of_machines_l1209_120993

def machine_problem : Prop :=
  ∃ (m : ℕ), (6 * 42) = 6 * 36 ∧ m = 7

theorem number_of_machines : machine_problem :=
  sorry

end NUMINAMATH_GPT_number_of_machines_l1209_120993


namespace NUMINAMATH_GPT_minimum_discount_correct_l1209_120914

noncomputable def minimum_discount (total_weight: ℝ) (cost_price: ℝ) (sell_price: ℝ) 
                                   (profit_required: ℝ) : ℝ :=
  let first_half_profit := (total_weight / 2) * (sell_price - cost_price)
  let second_half_profit_with_discount (x: ℝ) := (total_weight / 2) * (sell_price * x - cost_price)
  let required_profit_condition (x: ℝ) := first_half_profit + second_half_profit_with_discount x ≥ profit_required
  (1 - (7 / 11))

theorem minimum_discount_correct : minimum_discount 1000 7 10 2000 = 4 / 11 := 
by {
  -- We need to solve the inequality step by step to reach the final answer
  sorry
}

end NUMINAMATH_GPT_minimum_discount_correct_l1209_120914


namespace NUMINAMATH_GPT_tangent_line_parabola_l1209_120973

theorem tangent_line_parabola (k : ℝ) 
  (h : ∀ (x y : ℝ), 4 * x + 6 * y + k = 0 → y^2 = 32 * x) : k = 72 := 
sorry

end NUMINAMATH_GPT_tangent_line_parabola_l1209_120973


namespace NUMINAMATH_GPT_find_fx_l1209_120946

theorem find_fx (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x^2 + 1) = 2 * x^2 + 1) : ∀ x : ℝ, f x = 2 * x - 1 := 
sorry

end NUMINAMATH_GPT_find_fx_l1209_120946


namespace NUMINAMATH_GPT_evaluate_expression_l1209_120902

theorem evaluate_expression : 8 - 5 * (9 - (4 - 2)^2) * 2 = -42 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1209_120902


namespace NUMINAMATH_GPT_unit_cubes_fill_box_l1209_120963

theorem unit_cubes_fill_box (p : ℕ) (hp : Nat.Prime p) :
  let length := p
  let width := 2 * p
  let height := 3 * p
  length * width * height = 6 * p^3 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_unit_cubes_fill_box_l1209_120963


namespace NUMINAMATH_GPT_a5_eq_11_l1209_120929

variable (a : ℕ → ℚ) (S : ℕ → ℚ)
variable (n : ℕ) (d : ℚ) (a1 : ℚ)

-- The definitions as given in the conditions
def arithmetic_sequence (a : ℕ → ℚ) (a1 : ℚ) (d : ℚ) : Prop :=
  ∀ n, a n = a1 + (n - 1) * d

def sum_of_terms (S : ℕ → ℚ) (a1 : ℚ) (d : ℚ) : Prop :=
  ∀ n, S n = n / 2 * (2 * a1 + (n - 1) * d)

-- Given conditions
def cond1 (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  a 3 + S 3 = 22

def cond2 (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  a 4 - S 4 = -15

-- The statement to prove
theorem a5_eq_11 (a : ℕ → ℚ) (S : ℕ → ℚ) (a1 : ℚ) (d : ℚ)
  (h_arith : arithmetic_sequence a a1 d)
  (h_sum : sum_of_terms S a1 d)
  (h1 : cond1 a S)
  (h2 : cond2 a S) : a 5 = 11 := by
  sorry

end NUMINAMATH_GPT_a5_eq_11_l1209_120929


namespace NUMINAMATH_GPT_a_eq_b_if_fraction_is_integer_l1209_120990

theorem a_eq_b_if_fraction_is_integer (a b : ℕ) (h_pos_a : 1 ≤ a) (h_pos_b : 1 ≤ b) :
  ∃ k : ℕ, (a^4 + a^3 + 1) = k * (a^2 * b^2 + a * b^2 + 1) -> a = b :=
by
  sorry

end NUMINAMATH_GPT_a_eq_b_if_fraction_is_integer_l1209_120990


namespace NUMINAMATH_GPT_vector_combination_l1209_120937

-- Define the vectors and the conditions
def vec_a : ℝ × ℝ := (1, -2)
def vec_b (m : ℝ) : ℝ × ℝ := (2, m)
def parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ v = (k * u.1, k * u.2)

-- The main theorem to be proved
theorem vector_combination (m : ℝ) (h_parallel : parallel vec_a (vec_b m)) : 3 * vec_a + 2 * vec_b m = (7, -14) := by
  sorry

end NUMINAMATH_GPT_vector_combination_l1209_120937


namespace NUMINAMATH_GPT_octagon_area_half_l1209_120969

theorem octagon_area_half (parallelogram : ℝ) (h_parallelogram : parallelogram = 1) : 
  (octagon_area : ℝ) =
  1 / 2 := 
  sorry

end NUMINAMATH_GPT_octagon_area_half_l1209_120969


namespace NUMINAMATH_GPT_portrait_in_silver_box_l1209_120912

theorem portrait_in_silver_box
  (gold_box : Prop)
  (silver_box : Prop)
  (lead_box : Prop)
  (p : Prop) (q : Prop) (r : Prop)
  (h1 : p ↔ gold_box)
  (h2 : q ↔ ¬silver_box)
  (h3 : r ↔ ¬gold_box)
  (h4 : (p ∨ q ∨ r) ∧ ¬(p ∧ q) ∧ ¬(q ∧ r) ∧ ¬(r ∧ p)) :
  silver_box :=
sorry

end NUMINAMATH_GPT_portrait_in_silver_box_l1209_120912


namespace NUMINAMATH_GPT_swimming_speed_in_still_water_l1209_120988

theorem swimming_speed_in_still_water 
  (speed_of_water : ℝ) (distance : ℝ) (time : ℝ) (v : ℝ) 
  (h_water_speed : speed_of_water = 2) 
  (h_time_distance : time = 4 ∧ distance = 8) :
  v = 4 :=
by
  sorry

end NUMINAMATH_GPT_swimming_speed_in_still_water_l1209_120988


namespace NUMINAMATH_GPT_speed_of_stream_l1209_120911

theorem speed_of_stream (b s : ℕ) 
  (h1 : b + s = 42) 
  (h2 : b - s = 24) :
  s = 9 := by sorry

end NUMINAMATH_GPT_speed_of_stream_l1209_120911


namespace NUMINAMATH_GPT_sq_diff_eq_binom_identity_l1209_120954

variable (a b : ℝ)

theorem sq_diff_eq_binom_identity : (a - b) ^ 2 = a ^ 2 - 2 * a * b + b ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_sq_diff_eq_binom_identity_l1209_120954


namespace NUMINAMATH_GPT_proof_numbers_exist_l1209_120953

noncomputable def exists_numbers : Prop :=
  ∃ a b c : ℕ, a > 10^10 ∧ b > 10^10 ∧ c > 10^10 ∧
  (a * b % (a + 2012) = 0) ∧
  (a * c % (a + 2012) = 0) ∧
  (b * c % (b + 2012) = 0) ∧
  (a * b * c % (b + 2012) = 0) ∧
  (a * b * c % (c + 2012) = 0)

theorem proof_numbers_exist : exists_numbers :=
  sorry

end NUMINAMATH_GPT_proof_numbers_exist_l1209_120953


namespace NUMINAMATH_GPT_find_natural_numbers_l1209_120982

-- Problem statement: Find all natural numbers x, y, z such that 3^x + 4^y = 5^z
theorem find_natural_numbers (x y z : ℕ) (h : 3^x + 4^y = 5^z) : x = 2 ∧ y = 2 ∧ z = 2 :=
sorry

end NUMINAMATH_GPT_find_natural_numbers_l1209_120982


namespace NUMINAMATH_GPT_g_of_minus_1_eq_9_l1209_120950

-- defining f(x) and g(f(x)), and stating the objective to prove g(-1)=9
def f (x : ℝ) : ℝ := 4 * x - 9
def g (x : ℝ) : ℝ := 3 * x ^ 2 - 4 * x + 5

theorem g_of_minus_1_eq_9 : g (-1) = 9 :=
  sorry

end NUMINAMATH_GPT_g_of_minus_1_eq_9_l1209_120950


namespace NUMINAMATH_GPT_Sravan_travel_time_l1209_120972

theorem Sravan_travel_time :
  let total_distance := 540
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let speed_first_half := 45
  let speed_second_half := 30
  let time_first_half := first_half_distance / speed_first_half
  let time_second_half := second_half_distance / speed_second_half
  let total_time := time_first_half + time_second_half
  total_time = 15 :=
by
  let total_distance := 540
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let speed_first_half := 45
  let speed_second_half := 30
  let time_first_half := first_half_distance / speed_first_half
  let time_second_half := second_half_distance / speed_second_half
  let total_time := time_first_half + time_second_half
  sorry

end NUMINAMATH_GPT_Sravan_travel_time_l1209_120972


namespace NUMINAMATH_GPT_probability_top_card_is_king_or_queen_l1209_120935

-- Defining the basic entities of the problem
def standard_deck_size := 52
def ranks := 13
def suits := 4
def number_of_kings := 4
def number_of_queens := 4
def number_of_kings_and_queens := number_of_kings + number_of_queens

-- Statement: Calculating the probability that the top card is either a King or a Queen
theorem probability_top_card_is_king_or_queen :
  (number_of_kings_and_queens : ℚ) / standard_deck_size = 2 / 13 := by
  -- Skipping the proof for now
  sorry

end NUMINAMATH_GPT_probability_top_card_is_king_or_queen_l1209_120935


namespace NUMINAMATH_GPT_decrement_value_is_15_l1209_120904

noncomputable def decrement_value (n : ℕ) (original_mean updated_mean : ℕ) : ℕ :=
  (n * original_mean - n * updated_mean) / n

theorem decrement_value_is_15 : decrement_value 50 200 185 = 15 :=
by
  sorry

end NUMINAMATH_GPT_decrement_value_is_15_l1209_120904


namespace NUMINAMATH_GPT_savings_after_expense_increase_l1209_120962

-- Define the conditions
def monthly_salary : ℝ := 6500
def initial_savings_percentage : ℝ := 0.20
def increase_expenses_percentage : ℝ := 0.20

-- Define the statement we want to prove
theorem savings_after_expense_increase :
  (monthly_salary - (monthly_salary - (initial_savings_percentage * monthly_salary) + (increase_expenses_percentage * (monthly_salary - (initial_savings_percentage * monthly_salary))))) = 260 :=
sorry

end NUMINAMATH_GPT_savings_after_expense_increase_l1209_120962


namespace NUMINAMATH_GPT_area_of_region_enclosed_by_graph_l1209_120941

noncomputable def area_of_enclosed_region : ℝ :=
  let x1 := 41.67
  let x2 := 62.5
  let y1 := 8.33
  let y2 := -8.33
  0.5 * (x2 - x1) * (y1 - y2)

theorem area_of_region_enclosed_by_graph :
  area_of_enclosed_region = 173.28 :=
sorry

end NUMINAMATH_GPT_area_of_region_enclosed_by_graph_l1209_120941


namespace NUMINAMATH_GPT_fundraiser_successful_l1209_120928

-- Defining the conditions
def num_students_bringing_brownies := 30
def brownies_per_student := 12
def num_students_bringing_cookies := 20
def cookies_per_student := 24
def num_students_bringing_donuts := 15
def donuts_per_student := 12
def price_per_treat := 2

-- Calculating the total number of each type of treat
def total_brownies := num_students_bringing_brownies * brownies_per_student
def total_cookies := num_students_bringing_cookies * cookies_per_student
def total_donuts := num_students_bringing_donuts * donuts_per_student

-- Calculating the total number of treats
def total_treats := total_brownies + total_cookies + total_donuts

-- Calculating the total money raised
def total_money_raised := total_treats * price_per_treat

theorem fundraiser_successful : total_money_raised = 2040 := by
    -- We introduce a sorry here because we are not providing the proof steps.
    sorry

end NUMINAMATH_GPT_fundraiser_successful_l1209_120928


namespace NUMINAMATH_GPT_emily_cards_l1209_120921

theorem emily_cards (initial_cards : ℕ) (total_cards : ℕ) (given_cards : ℕ) 
  (h1 : initial_cards = 63) (h2 : total_cards = 70) 
  (h3 : total_cards = initial_cards + given_cards) : 
  given_cards = 7 := 
by 
  sorry

end NUMINAMATH_GPT_emily_cards_l1209_120921


namespace NUMINAMATH_GPT_possible_distances_AG_l1209_120952

theorem possible_distances_AG (A B V G : ℝ) (AB VG : ℝ) (x AG : ℝ) :
  (AB = 600) →
  (VG = 600) →
  (AG = 3 * x) →
  (AG = 900 ∨ AG = 1800) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_possible_distances_AG_l1209_120952


namespace NUMINAMATH_GPT_jake_sister_weight_ratio_l1209_120986

theorem jake_sister_weight_ratio
  (jake_present_weight : ℕ)
  (total_weight : ℕ)
  (weight_lost : ℕ)
  (sister_weight : ℕ)
  (jake_weight_after_loss : ℕ)
  (ratio : ℕ) :
  jake_present_weight = 188 →
  total_weight = 278 →
  weight_lost = 8 →
  jake_weight_after_loss = jake_present_weight - weight_lost →
  sister_weight = total_weight - jake_present_weight →
  ratio = jake_weight_after_loss / sister_weight →
  ratio = 2 := by
  sorry

end NUMINAMATH_GPT_jake_sister_weight_ratio_l1209_120986


namespace NUMINAMATH_GPT_find_f_neg_8point5_l1209_120961

def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodicity : ∀ x : ℝ, f (x + 2) = -f x
axiom initial_condition : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem find_f_neg_8point5 : f (-8.5) = -0.5 :=
by
  -- Expect this proof to follow the outlined logic
  sorry

end NUMINAMATH_GPT_find_f_neg_8point5_l1209_120961


namespace NUMINAMATH_GPT_exists_N_with_N_and_N2_ending_same_l1209_120945

theorem exists_N_with_N_and_N2_ending_same : 
  ∃ (N : ℕ), (N > 0) ∧ (N % 100000 = (N*N) % 100000) ∧ (N / 10000 ≠ 0) := sorry

end NUMINAMATH_GPT_exists_N_with_N_and_N2_ending_same_l1209_120945


namespace NUMINAMATH_GPT_catch_up_time_l1209_120936

theorem catch_up_time (x : ℕ) : 240 * x = 150 * x + 12 * 150 := by
  sorry

end NUMINAMATH_GPT_catch_up_time_l1209_120936


namespace NUMINAMATH_GPT_find_square_sum_l1209_120992

theorem find_square_sum (x y : ℝ) (h1 : 2 * x * (x + y) = 54) (h2 : 3 * y * (x + y) = 81) : (x + y) ^ 2 = 135 :=
sorry

end NUMINAMATH_GPT_find_square_sum_l1209_120992


namespace NUMINAMATH_GPT_rolling_green_probability_l1209_120903

/-- A cube with 5 green faces and 1 yellow face. -/
structure ColoredCube :=
  (green_faces : ℕ)
  (yellow_face : ℕ)
  (total_faces : ℕ)

def example_cube : ColoredCube :=
  { green_faces := 5, yellow_face := 1, total_faces := 6 }

/-- The probability of rolling a green face on a given cube. -/
def probability_of_rolling_green (cube : ColoredCube) : ℚ :=
  cube.green_faces / cube.total_faces

theorem rolling_green_probability :
  probability_of_rolling_green example_cube = 5 / 6 :=
by simp [probability_of_rolling_green, example_cube]

end NUMINAMATH_GPT_rolling_green_probability_l1209_120903


namespace NUMINAMATH_GPT_negation_necessary_not_sufficient_l1209_120974

theorem negation_necessary_not_sufficient (p q : Prop) : 
  ((¬ p) → ¬ (p ∨ q)) := 
sorry

end NUMINAMATH_GPT_negation_necessary_not_sufficient_l1209_120974


namespace NUMINAMATH_GPT_perimeter_of_fence_l1209_120977

noncomputable def n : ℕ := 18
noncomputable def w : ℝ := 0.5
noncomputable def d : ℝ := 4

theorem perimeter_of_fence : 3 * ((n / 3 - 1) * d + n / 3 * w) = 69 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_fence_l1209_120977


namespace NUMINAMATH_GPT_find_custom_operator_result_l1209_120994

def custom_operator (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem find_custom_operator_result :
  custom_operator 2 5 = 23 :=
by
  sorry

end NUMINAMATH_GPT_find_custom_operator_result_l1209_120994


namespace NUMINAMATH_GPT_intersection_M_N_l1209_120944

def M : Set ℝ := { x | x / (x - 1) ≥ 0 }
def N : Set ℝ := { y | ∃ x : ℝ, y = 3 * x^2 + 1 }

theorem intersection_M_N :
  { x | x / (x - 1) ≥ 0 } ∩ { y | ∃ x : ℝ, y = 3 * x^2 + 1 } = { x | x > 1 } :=
sorry

end NUMINAMATH_GPT_intersection_M_N_l1209_120944


namespace NUMINAMATH_GPT_susan_more_cats_than_bob_l1209_120991

-- Given problem: Initial and transaction conditions
def susan_initial_cats : ℕ := 21
def bob_initial_cats : ℕ := 3
def susan_additional_cats : ℕ := 5
def bob_additional_cats : ℕ := 7
def susan_gives_bob_cats : ℕ := 4

-- Declaration to find the difference between Susan's and Bob's cats
def final_susan_cats (initial : ℕ) (additional : ℕ) (given : ℕ) : ℕ := initial + additional - given
def final_bob_cats (initial : ℕ) (additional : ℕ) (received : ℕ) : ℕ := initial + additional + received

-- The proof statement which we need to show
theorem susan_more_cats_than_bob : 
  final_susan_cats susan_initial_cats susan_additional_cats susan_gives_bob_cats - 
  final_bob_cats bob_initial_cats bob_additional_cats susan_gives_bob_cats = 8 := by
  sorry

end NUMINAMATH_GPT_susan_more_cats_than_bob_l1209_120991


namespace NUMINAMATH_GPT_part1_part2_l1209_120996

def f (x a b : ℝ) : ℝ := |x - a| + |x - b|

theorem part1 (a b c x : ℝ) (h1 : |a - b| > c) : f x a b > c :=
  by sorry

theorem part2 (a : ℝ) (h1 : ∃ (x : ℝ), f x a 1 < 2 - |a - 2|) : 1/2 < a ∧ a < 5/2 :=
  by sorry

end NUMINAMATH_GPT_part1_part2_l1209_120996


namespace NUMINAMATH_GPT_contrapositive_example_l1209_120948

variable {a : ℕ → ℝ}

theorem contrapositive_example 
  (h₁ : ∀ n : ℕ, n > 0 → (a n + a (n + 2)) / 2 < a (n + 1)) :
  (∀ n : ℕ, n > 0 → a n ≤ a (n + 1)) → ∀ n : ℕ, n > 0 → (a n + a (n + 2)) / 2 ≥ a (n + 1) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_example_l1209_120948


namespace NUMINAMATH_GPT_find_first_offset_l1209_120920

theorem find_first_offset (x : ℝ) : 
  let area := 180
  let diagonal := 24
  let offset2 := 6
  (area = (diagonal * (x + offset2)) / 2) -> x = 9 :=
sorry

end NUMINAMATH_GPT_find_first_offset_l1209_120920


namespace NUMINAMATH_GPT_copper_percentage_l1209_120943

theorem copper_percentage (copperFirst copperSecond totalWeight1 totalWeight2: ℝ) 
    (h1 : copperFirst = 0.25)
    (h2 : copperSecond = 0.50) 
    (h3 : totalWeight1 = 200) 
    (h4 : totalWeight2 = 800) : 
    (copperFirst * totalWeight1 + copperSecond * totalWeight2) / (totalWeight1 + totalWeight2) * 100 = 45 := 
by 
  sorry

end NUMINAMATH_GPT_copper_percentage_l1209_120943


namespace NUMINAMATH_GPT_pool_capacity_l1209_120995

theorem pool_capacity (hose_rate leak_rate : ℝ) (fill_time : ℝ) (net_rate := hose_rate - leak_rate) (total_water := net_rate * fill_time) :
  hose_rate = 1.6 → 
  leak_rate = 0.1 → 
  fill_time = 40 → 
  total_water = 60 := by
  intros
  sorry

end NUMINAMATH_GPT_pool_capacity_l1209_120995


namespace NUMINAMATH_GPT_problem_statement_l1209_120957

noncomputable def sequence_def (a : ℝ) (S : ℕ → ℝ) (n : ℕ) : Prop :=
  (a ≠ 0) ∧
  (S 1 = a) ∧
  (S 2 = 2 / S 1) ∧
  (∀ n, n ≥ 3 → S n = 2 / S (n - 1))

theorem problem_statement (a : ℝ) (S : ℕ → ℝ) (h : sequence_def a S 2018) : 
  S 2018 = 2 / a := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l1209_120957


namespace NUMINAMATH_GPT_calc_expression_correct_l1209_120987

noncomputable def calc_expression : Real :=
  Real.sqrt 8 - (1 / 3)⁻¹ / Real.sqrt 3 + (1 - Real.sqrt 2)^2

theorem calc_expression_correct :
  calc_expression = 3 - Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_calc_expression_correct_l1209_120987


namespace NUMINAMATH_GPT_probability_distance_greater_than_2_l1209_120976

theorem probability_distance_greater_than_2 :
  let D := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}
  let area_square := 9
  let area_sector := Real.pi
  let area_shaded := area_square - area_sector
  let P := area_shaded / area_square
  P = (9 - Real.pi) / 9 :=
by
  sorry

end NUMINAMATH_GPT_probability_distance_greater_than_2_l1209_120976


namespace NUMINAMATH_GPT_sum_of_three_integers_eq_57_l1209_120960

theorem sum_of_three_integers_eq_57
  (a b c : ℕ) (h1: a * b * c = 7^3) (h2: a ≠ b) (h3: b ≠ c) (h4: a ≠ c) :
  a + b + c = 57 :=
sorry

end NUMINAMATH_GPT_sum_of_three_integers_eq_57_l1209_120960


namespace NUMINAMATH_GPT_find_distance_l1209_120906

-- Definitions based on given conditions
def speed : ℝ := 40 -- in km/hr
def time : ℝ := 6 -- in hours

-- Theorem statement
theorem find_distance (speed : ℝ) (time : ℝ) : speed = 40 → time = 6 → speed * time = 240 :=
by
  intros h1 h2
  rw [h1, h2]
  -- skipping the proof with sorry
  sorry

end NUMINAMATH_GPT_find_distance_l1209_120906


namespace NUMINAMATH_GPT_units_digit_of_product_l1209_120980

theorem units_digit_of_product (a b c : ℕ) (n m p : ℕ) (units_a : a ≡ 4 [MOD 10])
  (units_b : b ≡ 9 [MOD 10]) (units_c : c ≡ 16 [MOD 10])
  (exp_a : n = 150) (exp_b : m = 151) (exp_c : p = 152) :
  (a^n * b^m * c^p) % 10 = 4 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_product_l1209_120980


namespace NUMINAMATH_GPT_find_n_l1209_120984

theorem find_n (n : ℕ)
  (h1 : ∃ k : ℕ, k = n^3) -- the cube is cut into n^3 unit cubes
  (h2 : ∃ r : ℕ, r = 4 * n^2) -- 4 faces are painted, each with area n^2
  (h3 : 1 / 3 = r / (6 * k)) -- one-third of the total number of faces are red
  : n = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1209_120984


namespace NUMINAMATH_GPT_equal_roots_quadratic_l1209_120919

theorem equal_roots_quadratic (k : ℝ) : (∃ (x : ℝ), x*(x + 2) + k = 0 ∧ ∀ y z, (y, z) = (x, x)) → k = 1 :=
sorry

end NUMINAMATH_GPT_equal_roots_quadratic_l1209_120919


namespace NUMINAMATH_GPT_circle_passes_through_fixed_point_l1209_120905

theorem circle_passes_through_fixed_point (a : ℝ) (ha : a ≠ 1) : 
  ∃ P : ℝ × ℝ, P = (1, 1) ∧ ∀ (x y : ℝ), (x^2 + y^2 - 2*a*x + 2*(a-2)*y + 2 = 0) → (x, y) = P :=
sorry

end NUMINAMATH_GPT_circle_passes_through_fixed_point_l1209_120905


namespace NUMINAMATH_GPT_total_seeds_in_watermelons_l1209_120926

def slices1 : ℕ := 40
def seeds_per_slice1 : ℕ := 60
def slices2 : ℕ := 30
def seeds_per_slice2 : ℕ := 80
def slices3 : ℕ := 50
def seeds_per_slice3 : ℕ := 40

theorem total_seeds_in_watermelons :
  (slices1 * seeds_per_slice1) + (slices2 * seeds_per_slice2) + (slices3 * seeds_per_slice3) = 6800 := by
  sorry

end NUMINAMATH_GPT_total_seeds_in_watermelons_l1209_120926


namespace NUMINAMATH_GPT_quadratic_has_distinct_real_roots_expression_value_l1209_120930

variable (x m : ℝ)

-- Condition: Quadratic equation
def quadratic_eq := (x^2 - 2 * (m - 1) * x - m * (m + 2) = 0)

-- Prove that the quadratic equation always has two distinct real roots
theorem quadratic_has_distinct_real_roots (m : ℝ) : 
  ∃ a b : ℝ, a ≠ b ∧ quadratic_eq a m ∧ quadratic_eq b m :=
by
  sorry

-- Given that x = -2 is a root, prove that 2018 - 3(m-1)^2 = 2015
theorem expression_value (m : ℝ) (h : quadratic_eq (-2) m) : 
  2018 - 3 * (m - 1)^2 = 2015 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_distinct_real_roots_expression_value_l1209_120930


namespace NUMINAMATH_GPT_trust_meteorologist_l1209_120942

noncomputable def problem_statement : Prop :=
  let r := 0.74
  let p := 0.5
  let senators_forecast := (1 - 1.5 * p) * p^2 * r
  let meteorologist_forecast := 1.5 * p * (1 - p)^2 * (1 - r)
  meteorologist_forecast > senators_forecast

theorem trust_meteorologist : problem_statement :=
  sorry

end NUMINAMATH_GPT_trust_meteorologist_l1209_120942


namespace NUMINAMATH_GPT_rectangle_perimeter_eq_l1209_120916

noncomputable def rectangle_perimeter (z w : ℕ) : ℕ :=
  let longer_side := w
  let shorter_side := (z - w) / 2
  2 * longer_side + 2 * shorter_side

theorem rectangle_perimeter_eq (z w : ℕ) : rectangle_perimeter z w = w + z := by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_eq_l1209_120916


namespace NUMINAMATH_GPT_rectangle_dimensions_l1209_120955

theorem rectangle_dimensions (l w : ℝ) (h1 : l = 2 * w) (h2 : 2 * (l + w) = 3 * (l * w)) : 
  w = 1 ∧ l = 2 := by
  sorry

end NUMINAMATH_GPT_rectangle_dimensions_l1209_120955
