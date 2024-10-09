import Mathlib

namespace find_mn_l462_46252

theorem find_mn
  (AB BC : ℝ) -- Lengths of AB and BC
  (m n : ℝ)   -- Coefficients of the quadratic equation
  (h_perimeter : 2 * (AB + BC) = 12)
  (h_area : AB * BC = 5)
  (h_roots_sum : AB + BC = -m)
  (h_roots_product : AB * BC = n) :
  m * n = -30 :=
by
  sorry

end find_mn_l462_46252


namespace smallest_non_factor_product_l462_46204

open Nat

def is_factor (n d : ℕ) := d > 0 ∧ n % d = 0

theorem smallest_non_factor_product (a b : ℕ) (h1 : a ≠ b) (h2 : is_factor 48 a) (h3 : is_factor 48 b) (h4 : ¬ is_factor 48 (a * b)) : a * b = 18 :=
by
  sorry

end smallest_non_factor_product_l462_46204


namespace ralph_did_not_hit_110_balls_l462_46290

def tennis_problem : Prop :=
  ∀ (total_balls first_batch second_batch hit_first hit_second not_hit_first not_hit_second not_hit_total : ℕ),
  total_balls = 175 →
  first_batch = 100 →
  second_batch = 75 →
  hit_first = 2/5 * first_batch →
  hit_second = 1/3 * second_batch →
  not_hit_first = first_batch - hit_first →
  not_hit_second = second_batch - hit_second →
  not_hit_total = not_hit_first + not_hit_second →
  not_hit_total = 110

theorem ralph_did_not_hit_110_balls : tennis_problem := by
  unfold tennis_problem
  intros
  sorry

end ralph_did_not_hit_110_balls_l462_46290


namespace eiffel_tower_scale_l462_46207

theorem eiffel_tower_scale (height_tower_m : ℝ) (height_model_cm : ℝ) :
    height_tower_m = 324 →
    height_model_cm = 50 →
    (height_tower_m * 100) / height_model_cm = 648 →
    (648 / 100) = 6.48 :=
by
  intro h_tower h_model h_ratio
  rw [h_tower, h_model] at h_ratio
  sorry

end eiffel_tower_scale_l462_46207


namespace finish_11th_l462_46295

noncomputable def place_in_race (place: Fin 15) := ℕ

variables (Dana Ethan Alice Bob Chris Flora : Fin 15)

def conditions := 
  Dana.val + 3 = Ethan.val ∧
  Alice.val = Bob.val - 2 ∧
  Chris.val = Flora.val - 5 ∧
  Flora.val = Dana.val + 2 ∧
  Ethan.val = Alice.val - 3 ∧
  Bob.val = 6

theorem finish_11th (h : conditions Dana Ethan Alice Bob Chris Flora) : Flora.val = 10 :=
  by sorry

end finish_11th_l462_46295


namespace property_tax_increase_is_800_l462_46222

-- Define conditions as constants
def tax_rate : ℝ := 0.10
def initial_value : ℝ := 20000
def new_value : ℝ := 28000

-- Define the increase in property tax
def tax_increase : ℝ := (new_value * tax_rate) - (initial_value * tax_rate)

-- Statement to be proved
theorem property_tax_increase_is_800 : tax_increase = 800 :=
by
  sorry

end property_tax_increase_is_800_l462_46222


namespace operation_not_equal_33_l462_46239

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

end operation_not_equal_33_l462_46239


namespace total_books_sold_l462_46208

theorem total_books_sold (tuesday_books wednesday_books thursday_books : Nat) 
  (h1 : tuesday_books = 7) 
  (h2 : wednesday_books = 3 * tuesday_books) 
  (h3 : thursday_books = 3 * wednesday_books) : 
  tuesday_books + wednesday_books + thursday_books = 91 := 
by 
  sorry

end total_books_sold_l462_46208


namespace ratio_of_boys_l462_46282

theorem ratio_of_boys 
  (p : ℚ) 
  (h : p = (3/4) * (1 - p)) : 
  p = 3 / 7 :=
by
  sorry

end ratio_of_boys_l462_46282


namespace remaining_oil_quantity_check_remaining_oil_quantity_l462_46253

def initial_oil_quantity : Real := 40
def outflow_rate : Real := 0.2

theorem remaining_oil_quantity (t : Real) : Real :=
  initial_oil_quantity - outflow_rate * t

theorem check_remaining_oil_quantity (t : Real) : remaining_oil_quantity t = 40 - 0.2 * t := 
by 
  sorry

end remaining_oil_quantity_check_remaining_oil_quantity_l462_46253


namespace geometric_sequence_conditions_l462_46234

variable (a : ℕ → ℝ) (q : ℝ)

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_conditions (a : ℕ → ℝ) (q : ℝ)
  (h1 : geometric_sequence a q)
  (h2 : -1 < q)
  (h3 : q < 0) :
  (∀ n, a n * a (n + 1) < 0) ∧ (∀ n, |a n| > |a (n + 1)|) :=
by
  sorry

end geometric_sequence_conditions_l462_46234


namespace EF_side_length_l462_46299

def square_side_length (n : ℝ) : Prop := n = 10

def distance_parallel_line (d : ℝ) : Prop := d = 6.5

def area_difference (a : ℝ) : Prop := a = 13.8

theorem EF_side_length :
  ∃ (x : ℝ), square_side_length 10 ∧ distance_parallel_line 6.5 ∧ area_difference 13.8 ∧ x = 5.4 :=
sorry

end EF_side_length_l462_46299


namespace vector_equation_l462_46249

variable {V : Type} [AddCommGroup V]

variables (A B C : V)

theorem vector_equation :
  (B - A) - 2 • (C - A) + (C - B) = (A - C) :=
by
  sorry

end vector_equation_l462_46249


namespace remaining_hard_hats_l462_46279

theorem remaining_hard_hats 
  (pink_initial : ℕ)
  (green_initial : ℕ)
  (yellow_initial : ℕ)
  (carl_takes_pink : ℕ)
  (john_takes_pink : ℕ)
  (john_takes_green : ℕ) :
  john_takes_green = 2 * john_takes_pink →
  pink_initial = 26 →
  green_initial = 15 →
  yellow_initial = 24 →
  carl_takes_pink = 4 →
  john_takes_pink = 6 →
  ∃ pink_remaining green_remaining yellow_remaining total_remaining, 
    pink_remaining = pink_initial - carl_takes_pink - john_takes_pink ∧
    green_remaining = green_initial - john_takes_green ∧
    yellow_remaining = yellow_initial ∧
    total_remaining = pink_remaining + green_remaining + yellow_remaining ∧
    total_remaining = 43 :=
by
  sorry

end remaining_hard_hats_l462_46279


namespace math_problem_l462_46289

noncomputable def alpha_condition (α : ℝ) : Prop :=
  4 * Real.cos α - 2 * Real.sin α = 0

theorem math_problem (α : ℝ) (h : alpha_condition α) :
  (Real.sin α)^3 + (Real.cos α)^3 / (Real.sin α - Real.cos α) = 9 / 5 :=
  sorry

end math_problem_l462_46289


namespace johns_out_of_pocket_l462_46264

noncomputable def total_cost_after_discounts (computer_cost gaming_chair_cost accessories_cost : ℝ) 
  (comp_discount gaming_discount : ℝ) (tax : ℝ) : ℝ :=
  let comp_price := computer_cost * (1 - comp_discount)
  let chair_price := gaming_chair_cost * (1 - gaming_discount)
  let pre_tax_total := comp_price + chair_price + accessories_cost
  pre_tax_total * (1 + tax)

noncomputable def total_selling_price (playstation_value playstation_discount bicycle_price : ℝ) (exchange_rate : ℝ) : ℝ :=
  let playstation_price := playstation_value * (1 - playstation_discount)
  (playstation_price * exchange_rate) / exchange_rate + bicycle_price

theorem johns_out_of_pocket (computer_cost gaming_chair_cost accessories_cost comp_discount gaming_discount tax 
  playstation_value playstation_discount bicycle_price exchange_rate : ℝ) :
  computer_cost = 1500 →
  gaming_chair_cost = 400 →
  accessories_cost = 300 →
  comp_discount = 0.2 →
  gaming_discount = 0.1 →
  tax = 0.05 →
  playstation_value = 600 →
  playstation_discount = 0.2 →
  bicycle_price = 200 →
  exchange_rate = 100 →
  total_cost_after_discounts computer_cost gaming_chair_cost accessories_cost comp_discount gaming_discount tax -
  total_selling_price playstation_value playstation_discount bicycle_price exchange_rate = 1273 := by
  intros
  sorry

end johns_out_of_pocket_l462_46264


namespace dolphins_trained_next_month_l462_46271

theorem dolphins_trained_next_month
  (total_dolphins : ℕ) 
  (one_fourth_fully_trained : ℚ) 
  (two_thirds_in_training : ℚ)
  (h1 : total_dolphins = 20)
  (h2 : one_fourth_fully_trained = 1 / 4) 
  (h3 : two_thirds_in_training = 2 / 3) :
  (total_dolphins - total_dolphins * one_fourth_fully_trained) * two_thirds_in_training = 10 := 
by 
  sorry

end dolphins_trained_next_month_l462_46271


namespace challenging_math_problem_l462_46212

theorem challenging_math_problem :
  ((9^2 + (3^3 - 1) * 4^2) % 6) * Real.sqrt 49 + (15 - 3 * 5) = 35 :=
by
  sorry

end challenging_math_problem_l462_46212


namespace greatest_possible_value_l462_46235

theorem greatest_possible_value (A B C D : ℕ) 
    (h1 : A + B + C + D = 200) 
    (h2 : A + B = 70) 
    (h3 : 0 < A) 
    (h4 : 0 < B) 
    (h5 : 0 < C) 
    (h6 : 0 < D) : 
    C ≤ 129 := 
sorry

end greatest_possible_value_l462_46235


namespace mr_hernandez_tax_l462_46263

theorem mr_hernandez_tax :
  let taxable_income := 42500
  let resident_months := 9
  let standard_deduction := if resident_months > 6 then 5000 else 0
  let adjusted_income := taxable_income - standard_deduction
  let tax_bracket_1 := min adjusted_income 10000 * 0.01
  let tax_bracket_2 := min (max (adjusted_income - 10000) 0) 20000 * 0.03
  let tax_bracket_3 := min (max (adjusted_income - 30000) 0) 30000 * 0.05
  let total_tax_before_credit := tax_bracket_1 + tax_bracket_2 + tax_bracket_3
  let tax_credit := if resident_months < 10 then 500 else 0
  total_tax_before_credit - tax_credit = 575 := 
by
  sorry
  
end mr_hernandez_tax_l462_46263


namespace largest_mersenne_prime_less_than_500_l462_46272

-- Define what it means for a number to be prime
def is_prime (p : ℕ) : Prop :=
p > 1 ∧ ∀ (n : ℕ), n > 1 ∧ n < p → ¬ (p % n = 0)

-- Define what a Mersenne prime is
def is_mersenne_prime (m : ℕ) : Prop :=
∃ n : ℕ, is_prime n ∧ m = 2^n - 1

-- We state the main theorem we want to prove
theorem largest_mersenne_prime_less_than_500 : ∀ (m : ℕ), is_mersenne_prime m ∧ m < 500 → m ≤ 127 :=
by 
  sorry

end largest_mersenne_prime_less_than_500_l462_46272


namespace problem1_l462_46224

theorem problem1 :
  let total_products := 10
  let defective_products := 4
  let first_def_pos := 5
  let last_def_pos := 10
  ∃ (num_methods : Nat), num_methods = 103680 :=
by
  sorry

end problem1_l462_46224


namespace find_k_value_l462_46248

theorem find_k_value (k : ℝ) : 
  5 + ∑' n : ℕ, (5 + k + n) / 5^(n+1) = 12 → k = 18.2 :=
by 
  sorry

end find_k_value_l462_46248


namespace exists_circle_touching_given_circles_and_line_l462_46214

-- Define the given radii
def r1 := 1
def r2 := 3
def r3 := 4

-- Prove that there exists a circle with a specific radius touching the given circles and line AB
theorem exists_circle_touching_given_circles_and_line (x : ℝ) :
  ∃ (r : ℝ), r > 0 ∧ (r + r1) = x ∧ (r + r2) = x ∧ (r + r3) = x :=
sorry

end exists_circle_touching_given_circles_and_line_l462_46214


namespace correct_population_l462_46297

variable (P : ℕ) (S : ℕ)
variable (math_scores : ℕ → Type)

-- Assume P is the total number of students who took the exam.
-- Let math_scores(P) represent the math scores of P students.

def population_data (P : ℕ) : Prop := 
  P = 50000

def sample_data (S : ℕ) : Prop :=
  S = 2000

theorem correct_population (P : ℕ) (S : ℕ) (math_scores : ℕ → Type)
  (hP : population_data P) (hS : sample_data S) : 
  math_scores P = math_scores 50000 :=
by {
  sorry
}

end correct_population_l462_46297


namespace haircuts_away_from_next_free_l462_46250

def free_haircut (total_paid : ℕ) : ℕ := total_paid / 14

theorem haircuts_away_from_next_free (total_haircuts : ℕ) (free_haircuts : ℕ) (haircuts_per_free : ℕ) :
  total_haircuts = 79 → free_haircuts = 5 → haircuts_per_free = 14 → 
  (haircuts_per_free - (total_haircuts - free_haircuts)) % haircuts_per_free = 10 :=
by
  intros h1 h2 h3
  sorry

end haircuts_away_from_next_free_l462_46250


namespace vertex_coordinates_l462_46215

-- Define the given parabola equation
def parabola (x : ℝ) : ℝ := (x + 3) ^ 2 - 1

-- Define the statement for the coordinates of the vertex of the parabola
theorem vertex_coordinates : ∃ (h k : ℝ), (∀ x : ℝ, parabola x = (x + 3) ^ 2 - 1) ∧ h = -3 ∧ k = -1 := 
  sorry

end vertex_coordinates_l462_46215


namespace vertical_asymptote_condition_l462_46283

theorem vertical_asymptote_condition (c : ℝ) :
  (∀ x : ℝ, (x = 3 ∨ x = -6) → (x^2 - x + c = 0)) → 
  (c = -6 ∨ c = -42) :=
by
  sorry

end vertical_asymptote_condition_l462_46283


namespace trig_function_properties_l462_46232

theorem trig_function_properties :
  ∀ x : ℝ, 
    (1 - 2 * (Real.sin (x - π / 4))^2) = Real.sin (2 * x) ∧ 
    (∀ x : ℝ, Real.sin (2 * (-x)) = -Real.sin (2 * x)) ∧ 
    2 * π / 2 = π :=
by
  sorry

end trig_function_properties_l462_46232


namespace solve_for_y_l462_46255

theorem solve_for_y (y : ℕ) (h : 2^y + 8 = 4 * 2^y - 40) : y = 4 :=
by
  sorry

end solve_for_y_l462_46255


namespace max_value_y2_minus_x2_plus_x_plus_5_l462_46219

theorem max_value_y2_minus_x2_plus_x_plus_5 (x y : ℝ) (h : y^2 + x - 2 = 0) : 
  ∃ M, M = 7 ∧ ∀ u v, v^2 + u - 2 = 0 → y^2 - x^2 + x + 5 ≤ M :=
by
  sorry

end max_value_y2_minus_x2_plus_x_plus_5_l462_46219


namespace martha_jar_spices_cost_l462_46258

def price_per_jar_spices (p_beef p_fv p_oj : ℕ) (price_spices : ℕ) :=
  let total_spent := (3 * p_beef) + (8 * p_fv) + p_oj + (3 * price_spices)
  let total_points := (total_spent / 10) * 50 + if total_spent > 100 then 250 else 0
  total_points

theorem martha_jar_spices_cost (price_spices : ℕ) :
  price_per_jar_spices 11 4 37 price_spices = 850 → price_spices = 6 := by
  sorry

end martha_jar_spices_cost_l462_46258


namespace response_rate_percentage_l462_46225

theorem response_rate_percentage (number_of_responses_needed number_of_questionnaires_mailed : ℕ) 
  (h1 : number_of_responses_needed = 300) 
  (h2 : number_of_questionnaires_mailed = 500) : 
  (number_of_responses_needed / number_of_questionnaires_mailed : ℚ) * 100 = 60 :=
by 
  sorry

end response_rate_percentage_l462_46225


namespace product_of_two_numbers_l462_46293

theorem product_of_two_numbers (a b : ℤ) (h1 : lcm a b = 72) (h2 : gcd a b = 8) :
  a * b = 576 :=
sorry

end product_of_two_numbers_l462_46293


namespace school_boys_number_l462_46242

theorem school_boys_number (B G : ℕ) (h1 : B / G = 5 / 13) (h2 : G = B + 80) : B = 50 :=
by
  sorry

end school_boys_number_l462_46242


namespace min_boxes_to_eliminate_for_one_third_chance_l462_46211

-- Define the number of boxes
def total_boxes := 26

-- Define the number of boxes with at least $250,000
def boxes_with_at_least_250k := 6

-- Define the condition for having a 1/3 chance
def one_third_chance (remaining_boxes : ℕ) : Prop :=
  6 / remaining_boxes = 1 / 3

-- Define the target number of boxes to eliminate
def boxes_to_eliminate := total_boxes - 18

theorem min_boxes_to_eliminate_for_one_third_chance :
  ∃ remaining_boxes : ℕ, one_third_chance remaining_boxes ∧ total_boxes - remaining_boxes = boxes_to_eliminate :=
sorry

end min_boxes_to_eliminate_for_one_third_chance_l462_46211


namespace Sally_bought_20_pokemon_cards_l462_46205

theorem Sally_bought_20_pokemon_cards
  (initial_cards : ℕ)
  (cards_from_dan : ℕ)
  (total_cards : ℕ)
  (bought_cards : ℕ)
  (h1 : initial_cards = 27)
  (h2 : cards_from_dan = 41)
  (h3 : total_cards = 88)
  (h4 : total_cards = initial_cards + cards_from_dan + bought_cards) :
  bought_cards = 20 := 
by
  sorry

end Sally_bought_20_pokemon_cards_l462_46205


namespace cafeteria_problem_l462_46273

theorem cafeteria_problem (C : ℕ) 
    (h1 : ∃ h : ℕ, h = 4 * C)
    (h2 : 5 = 5)
    (h3 : C + 4 * C + 5 = 40) : 
    C = 7 := sorry

end cafeteria_problem_l462_46273


namespace P_at_7_eq_5760_l462_46213

noncomputable def P (x : ℝ) : ℝ :=
  12 * (x - 1) * (x - 2) * (x - 3)^2 * (x - 6)^4

theorem P_at_7_eq_5760 : P 7 = 5760 :=
by
  -- Proof goes here
  sorry

end P_at_7_eq_5760_l462_46213


namespace square_field_area_l462_46274

/-- 
  Statement: Prove that the area of the square field is 69696 square meters 
  given that the wire goes around the square field 15 times and the total 
  length of the wire is 15840 meters.
-/
theorem square_field_area (rounds : ℕ) (total_length : ℕ) (area : ℕ) 
  (h1 : rounds = 15) (h2 : total_length = 15840) : 
  area = 69696 := 
by 
  sorry

end square_field_area_l462_46274


namespace sum_of_all_angles_l462_46200

-- Defining the three triangles and their properties
structure Triangle :=
  (a1 a2 a3 : ℝ)
  (sum : a1 + a2 + a3 = 180)

def triangle_ABC : Triangle := {a1 := 1, a2 := 2, a3 := 3, sum := sorry}
def triangle_DEF : Triangle := {a1 := 4, a2 := 5, a3 := 6, sum := sorry}
def triangle_GHI : Triangle := {a1 := 7, a2 := 8, a3 := 9, sum := sorry}

theorem sum_of_all_angles :
  triangle_ABC.a1 + triangle_ABC.a2 + triangle_ABC.a3 +
  triangle_DEF.a1 + triangle_DEF.a2 + triangle_DEF.a3 +
  triangle_GHI.a1 + triangle_GHI.a2 + triangle_GHI.a3 = 540 := by
  sorry

end sum_of_all_angles_l462_46200


namespace smallest_value_of_x_l462_46291

theorem smallest_value_of_x (x : ℝ) (h : |x - 3| = 8) : x = -5 :=
sorry

end smallest_value_of_x_l462_46291


namespace sum_of_x_values_l462_46209

theorem sum_of_x_values (x : ℝ) (h : x ≠ -1) : 
  (∃ x, 3 = (x^3 - 3*x^2 - 4*x)/(x + 1)) →
  (x = 6) :=
by
  sorry

end sum_of_x_values_l462_46209


namespace jill_commute_time_l462_46240

theorem jill_commute_time :
  let dave_steps_per_min := 80
  let dave_cm_per_step := 70
  let dave_time_min := 20
  let dave_speed :=
    dave_steps_per_min * dave_cm_per_step
  let dave_distance :=
    dave_speed * dave_time_min
  let jill_steps_per_min := 120
  let jill_cm_per_step := 50
  let jill_speed :=
    jill_steps_per_min * jill_cm_per_step
  let jill_time :=
    dave_distance / jill_speed
  jill_time = 18 + 2 / 3 := by
  sorry

end jill_commute_time_l462_46240


namespace simplify_fraction_l462_46267

theorem simplify_fraction (d : ℝ) : (6 - 5 * d) / 9 - 3 = (-21 - 5 * d) / 9 :=
by
  sorry

end simplify_fraction_l462_46267


namespace total_amount_spent_l462_46201

-- Definitions for problem conditions
def mall_spent_before_discount : ℝ := 250
def clothes_discount_percent : ℝ := 0.15
def mall_tax_percent : ℝ := 0.08

def movie_ticket_price : ℝ := 24
def num_movies : ℝ := 3
def ticket_discount_percent : ℝ := 0.10
def movie_tax_percent : ℝ := 0.05

def beans_price : ℝ := 1.25
def num_beans : ℝ := 20
def cucumber_price : ℝ := 2.50
def num_cucumbers : ℝ := 5
def tomato_price : ℝ := 5.00
def num_tomatoes : ℝ := 3
def pineapple_price : ℝ := 6.50
def num_pineapples : ℝ := 2
def market_tax_percent : ℝ := 0.07

-- Proof statement
theorem total_amount_spent :
  let mall_spent_after_discount := mall_spent_before_discount * (1 - clothes_discount_percent)
  let mall_tax := mall_spent_after_discount * mall_tax_percent
  let total_mall_spent := mall_spent_after_discount + mall_tax

  let total_ticket_cost_before_discount := num_movies * movie_ticket_price
  let ticket_cost_after_discount := total_ticket_cost_before_discount * (1 - ticket_discount_percent)
  let movie_tax := ticket_cost_after_discount * movie_tax_percent
  let total_movie_spent := ticket_cost_after_discount + movie_tax

  let total_beans_cost := num_beans * beans_price
  let total_cucumbers_cost := num_cucumbers * cucumber_price
  let total_tomatoes_cost := num_tomatoes * tomato_price
  let total_pineapples_cost := num_pineapples * pineapple_price
  let total_market_spent_before_tax := total_beans_cost + total_cucumbers_cost + total_tomatoes_cost + total_pineapples_cost
  let market_tax := total_market_spent_before_tax * market_tax_percent
  let total_market_spent := total_market_spent_before_tax + market_tax
  
  let total_spent := total_mall_spent + total_movie_spent + total_market_spent
  total_spent = 367.63 :=
by
  sorry

end total_amount_spent_l462_46201


namespace translate_triangle_vertex_l462_46296

theorem translate_triangle_vertex 
    (a b : ℤ) 
    (hA : (-3, a) = (-1, 2) + (-2, a - 2)) 
    (hB : (b, 3) = (1, -1) + (b - 1, 4)) :
    (2 + (-3 - (-1)), 1 + (3 - (-1))) = (0, 5) :=
by 
  -- proof is omitted as instructed
  sorry

end translate_triangle_vertex_l462_46296


namespace solve_equation_l462_46285

theorem solve_equation (x : ℝ) : 
  (x - 1)^2 + 2 * x * (x - 1) = 0 ↔ x = 1 ∨ x = 1 / 3 :=
by sorry

end solve_equation_l462_46285


namespace find_people_got_off_at_first_stop_l462_46292

def total_seats (rows : ℕ) (seats_per_row : ℕ) : ℕ :=
  rows * seats_per_row

def occupied_seats (total_seats : ℕ) (initial_people : ℕ) : ℕ :=
  total_seats - initial_people

def occupied_seats_after_first_stop (initial_people : ℕ) (boarded_first_stop : ℕ) (got_off_first_stop : ℕ) : ℕ :=
  (initial_people + boarded_first_stop) - got_off_first_stop

def occupied_seats_after_second_stop (occupied_after_first_stop : ℕ) (boarded_second_stop : ℕ) (got_off_second_stop : ℕ) : ℕ :=
  (occupied_after_first_stop + boarded_second_stop) - got_off_second_stop

theorem find_people_got_off_at_first_stop
  (initial_people : ℕ := 16)
  (boarded_first_stop : ℕ := 15)
  (total_rows : ℕ := 23)
  (seats_per_row : ℕ := 4)
  (boarded_second_stop : ℕ := 17)
  (got_off_second_stop : ℕ := 10)
  (empty_seats_after_second_stop : ℕ := 57)
  : ∃ x, (occupied_seats_after_second_stop (occupied_seats_after_first_stop initial_people boarded_first_stop x) boarded_second_stop got_off_second_stop) = total_seats total_rows seats_per_row - empty_seats_after_second_stop :=
by
  sorry

end find_people_got_off_at_first_stop_l462_46292


namespace can_cover_101x101_with_102_cells_100_times_l462_46247

theorem can_cover_101x101_with_102_cells_100_times :
  ∃ f : Fin 100 → Fin 101 → Fin 101 → Bool,
  (∀ i j : Fin 101, (i ≠ 100 ∨ j ≠ 100) → ∃ t : Fin 100, 
    f t i j = true) :=
sorry

end can_cover_101x101_with_102_cells_100_times_l462_46247


namespace a10_is_b55_l462_46270

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℕ := 2 * n - 1

-- Define the new sequence b_n according to the given insertion rules
def b (k : ℕ) : ℕ := sorry

-- Prove that if a_10 = 19, then 19 is the 55th term in the new sequence b_n
theorem a10_is_b55 : b 55 = a 10 := sorry

end a10_is_b55_l462_46270


namespace gift_distribution_l462_46245

noncomputable section

structure Recipients :=
  (ondra : String)
  (matej : String)
  (kuba : String)

structure PetrStatements :=
  (ondra_fire_truck : Bool)
  (kuba_no_fire_truck : Bool)
  (matej_no_merkur : Bool)

def exactly_one_statement_true (s : PetrStatements) : Prop :=
  (s.ondra_fire_truck && ¬s.kuba_no_fire_truck && ¬s.matej_no_merkur)
  ∨ (¬s.ondra_fire_truck && s.kuba_no_fire_truck && ¬s.matej_no_merkur)
  ∨ (¬s.ondra_fire_truck && ¬s.kuba_no_fire_truck && s.matej_no_merkur)

def correct_recipients (r : Recipients) : Prop :=
  r.kuba = "fire truck" ∧ r.matej = "helicopter" ∧ r.ondra = "Merkur"

theorem gift_distribution
  (r : Recipients)
  (s : PetrStatements)
  (h : exactly_one_statement_true s)
  (h0 : ¬exactly_one_statement_true ⟨r.ondra = "fire truck", r.kuba ≠ "fire truck", r.matej ≠ "Merkur"⟩)
  (h1 : ¬exactly_one_statement_true ⟨r.ondra ≠ "fire truck", r.kuba ≠ "fire truck", r.matej ≠ "Merkur"⟩)
  : correct_recipients r := by
  -- Proof is omitted as per the instructions
  sorry

end gift_distribution_l462_46245


namespace no_multiple_of_2310_in_2_j_minus_2_i_l462_46246

theorem no_multiple_of_2310_in_2_j_minus_2_i (i j : ℕ) (h₀ : 0 ≤ i) (h₁ : i < j) (h₂ : j ≤ 50) :
  ¬ ∃ k : ℕ, 2^j - 2^i = 2310 * k :=
by 
  sorry

end no_multiple_of_2310_in_2_j_minus_2_i_l462_46246


namespace mike_initial_marbles_l462_46256

theorem mike_initial_marbles (n : ℕ) 
  (gave_to_sam : ℕ) (left_with_mike : ℕ)
  (h1 : gave_to_sam = 4)
  (h2 : left_with_mike = 4)
  (h3 : n = gave_to_sam + left_with_mike) : n = 8 := 
by
  sorry

end mike_initial_marbles_l462_46256


namespace david_still_has_less_than_750_l462_46286

theorem david_still_has_less_than_750 (S R : ℝ) 
  (h1 : S + R = 1500)
  (h2 : R < S) : 
  R < 750 :=
by 
  sorry

end david_still_has_less_than_750_l462_46286


namespace k_bounds_inequality_l462_46287

open Real

theorem k_bounds_inequality (k : ℝ) :
  (∀ x : ℝ, abs ((x^2 - k * x + 1) / (x^2 + x + 1)) < 3) ↔ -5 ≤ k ∧ k ≤ 1 := 
sorry

end k_bounds_inequality_l462_46287


namespace bananas_per_box_l462_46298

theorem bananas_per_box (total_bananas : ℕ) (num_boxes : ℕ) (h1 : total_bananas = 40) (h2 : num_boxes = 8) :
  total_bananas / num_boxes = 5 := by
  sorry

end bananas_per_box_l462_46298


namespace find_y_l462_46206

theorem find_y (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : x = 8) : y = 1 :=
by
  sorry

end find_y_l462_46206


namespace maximize_probability_sum_8_l462_46216

def L : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12]

theorem maximize_probability_sum_8 :
  (∀ x ∈ L, x ≠ 4 → (∃ y ∈ (List.erase L x), y = 8 - x)) ∧ 
  (∀ y ∈ List.erase L 4, ¬(∃ x ∈ List.erase L 4, x + y = 8)) :=
sorry

end maximize_probability_sum_8_l462_46216


namespace ben_has_10_fewer_stickers_than_ryan_l462_46217

theorem ben_has_10_fewer_stickers_than_ryan :
  ∀ (Karl_stickers Ryan_stickers Ben_stickers total_stickers : ℕ),
    Karl_stickers = 25 →
    Ryan_stickers = Karl_stickers + 20 →
    total_stickers = Karl_stickers + Ryan_stickers + Ben_stickers →
    total_stickers = 105 →
    (Ryan_stickers - Ben_stickers) = 10 :=
by
  intros Karl_stickers Ryan_stickers Ben_stickers total_stickers h1 h2 h3 h4
  -- Conditions mentioned in a)
  exact sorry

end ben_has_10_fewer_stickers_than_ryan_l462_46217


namespace volunteers_meet_again_in_360_days_l462_46233

theorem volunteers_meet_again_in_360_days :
  let Sasha := 5
  let Leo := 8
  let Uma := 9
  let Kim := 10
  Nat.lcm Sasha (Nat.lcm Leo (Nat.lcm Uma Kim)) = 360 :=
by
  sorry

end volunteers_meet_again_in_360_days_l462_46233


namespace total_amount_paid_l462_46202

-- Definitions from the conditions
def quantity_grapes : ℕ := 8
def rate_grapes : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 60

-- Main statement to prove
theorem total_amount_paid :
  (quantity_grapes * rate_grapes) + (quantity_mangoes * rate_mangoes) = 1100 :=
by
  sorry

end total_amount_paid_l462_46202


namespace no_integer_solution_k_range_l462_46229

theorem no_integer_solution_k_range (k : ℝ) :
  (∀ x : ℤ, ¬ ((k * x - k^2 - 4) * (x - 4) < 0)) → (1 ≤ k ∧ k ≤ 4) :=
by
  sorry

end no_integer_solution_k_range_l462_46229


namespace garden_stone_calculation_l462_46237

/-- A rectangular garden with dimensions 15m by 2m and patio stones of dimensions 0.5m by 0.5m requires 120 stones to be fully covered -/
theorem garden_stone_calculation :
  let garden_length := 15
  let garden_width := 2
  let stone_length := 0.5
  let stone_width := 0.5
  let area_garden := garden_length * garden_width
  let area_stone := stone_length * stone_width
  let num_stones := area_garden / area_stone
  num_stones = 120 :=
by
  sorry

end garden_stone_calculation_l462_46237


namespace percent_to_decimal_l462_46220

theorem percent_to_decimal : (2 : ℝ) / 100 = 0.02 :=
by
  -- Proof would go here
  sorry

end percent_to_decimal_l462_46220


namespace trapezoid_perimeter_and_area_l462_46218

theorem trapezoid_perimeter_and_area (PQ RS QR PS : ℝ) (hPQ_RS : PQ = RS)
  (hPQ_RS_positive : PQ > 0) (hQR : QR = 10) (hPS : PS = 20) (height : ℝ)
  (h_height : height = 5) :
  PQ = 5 * Real.sqrt 2 ∧
  QR = 10 ∧
  PS = 20 ∧ 
  height = 5 ∧
  (PQ + QR + RS + PS = 30 + 10 * Real.sqrt 2) ∧
  (1 / 2 * (QR + PS) * height = 75) :=
by
  sorry

end trapezoid_perimeter_and_area_l462_46218


namespace number_of_technicians_l462_46228

-- Definitions of the conditions
def average_salary_all_workers := 10000
def average_salary_technicians := 12000
def average_salary_rest := 8000
def total_workers := 14

-- Variables for the number of technicians and the rest of the workers
variable (T R : ℕ)

-- Problem statement in Lean
theorem number_of_technicians :
  (T + R = total_workers) →
  (T * average_salary_technicians + R * average_salary_rest = total_workers * average_salary_all_workers) →
  T = 7 :=
by
  -- leaving the proof as sorry
  sorry

end number_of_technicians_l462_46228


namespace avg_tickets_per_member_is_66_l462_46257

-- Definitions based on the problem's conditions
def avg_female_tickets : ℕ := 70
def male_to_female_ratio : ℕ := 2
def avg_male_tickets : ℕ := 58

-- Let the number of male members be M and number of female members be F
variables (M : ℕ) (F : ℕ)
def num_female_members : ℕ := male_to_female_ratio * M

-- Total tickets sold by males
def total_male_tickets : ℕ := avg_male_tickets * M

-- Total tickets sold by females
def total_female_tickets : ℕ := avg_female_tickets * num_female_members M

-- Total tickets sold by all members
def total_tickets_sold : ℕ := total_male_tickets M + total_female_tickets M

-- Total number of members
def total_members : ℕ := M + num_female_members M

-- Statement to prove: the average number of tickets sold per member is 66
theorem avg_tickets_per_member_is_66 : total_tickets_sold M / total_members M = 66 :=
by 
  sorry

end avg_tickets_per_member_is_66_l462_46257


namespace train_speed_excluding_stoppages_l462_46276

theorem train_speed_excluding_stoppages 
    (speed_including_stoppages : ℕ)
    (stoppage_time_per_hour : ℕ)
    (running_time_per_hour : ℚ)
    (h1 : speed_including_stoppages = 36)
    (h2 : stoppage_time_per_hour = 20)
    (h3 : running_time_per_hour = 2 / 3) :
    ∃ S : ℕ, S = 54 :=
by 
  sorry

end train_speed_excluding_stoppages_l462_46276


namespace fg_of_5_eq_140_l462_46251

def g (x : ℝ) : ℝ := 4 * x + 5
def f (x : ℝ) : ℝ := 6 * x - 10

theorem fg_of_5_eq_140 : f (g 5) = 140 := by
  sorry

end fg_of_5_eq_140_l462_46251


namespace max_value_x_plus_y_max_value_x_plus_y_achieved_l462_46244

theorem max_value_x_plus_y (x y : ℝ) (h1: x^2 + y^2 = 100) (h2: x * y = 40) : x + y ≤ 6 * Real.sqrt 5 :=
by
  sorry

theorem max_value_x_plus_y_achieved (x y : ℝ) (h1: x^2 + y^2 = 100) (h2: x * y = 40) : ∃ x y, x + y = 6 * Real.sqrt 5 :=
by
  sorry

end max_value_x_plus_y_max_value_x_plus_y_achieved_l462_46244


namespace population_in_2001_l462_46294

-- Define the populations at specific years
def pop_2000 := 50
def pop_2002 := 146
def pop_2003 := 350

-- Define the population difference condition
def pop_condition (n : ℕ) (pop : ℕ → ℕ) :=
  pop (n + 3) - pop n = 3 * pop (n + 2)

-- Given that the population condition holds, and specific populations are known,
-- the population in the year 2001 is 100
theorem population_in_2001 :
  (∃ (pop : ℕ → ℕ), pop 2000 = pop_2000 ∧ pop 2002 = pop_2002 ∧ pop 2003 = pop_2003 ∧ 
    pop_condition 2000 pop) → ∃ (pop : ℕ → ℕ), pop 2001 = 100 :=
by
  -- Placeholder for the actual proof
  sorry

end population_in_2001_l462_46294


namespace monthly_income_of_P_l462_46261

-- Define variables and assumptions
variables (P Q R : ℝ)
axiom avg_P_Q : (P + Q) / 2 = 5050
axiom avg_Q_R : (Q + R) / 2 = 6250
axiom avg_P_R : (P + R) / 2 = 5200

-- Prove that the monthly income of P is 4000
theorem monthly_income_of_P : P = 4000 :=
by
  sorry

end monthly_income_of_P_l462_46261


namespace exists_tangent_inequality_l462_46275

theorem exists_tangent_inequality {x : Fin 8 → ℝ} (h : Function.Injective x) :
  ∃ (i j : Fin 8), i ≠ j ∧ 0 < (x i - x j) / (1 + x i * x j) ∧ (x i - x j) / (1 + x i * x j) < Real.tan (Real.pi / 7) :=
by
  sorry

end exists_tangent_inequality_l462_46275


namespace cost_of_5_pound_bag_is_2_l462_46238

-- Define costs of each type of bag
def cost_10_pound_bag : ℝ := 20.40
def cost_25_pound_bag : ℝ := 32.25
def least_total_cost : ℝ := 98.75

-- Define the total weight constraint
def min_weight : ℕ := 65
def max_weight : ℕ := 80
def weight_25_pound_bags : ℕ := 75

-- Given condition: The total purchase fulfils the condition of minimum cost
def total_cost_3_bags_25 : ℝ := 3 * cost_25_pound_bag
def remaining_cost : ℝ := least_total_cost - total_cost_3_bags_25

-- Prove the cost of the 5-pound bag is $2.00
theorem cost_of_5_pound_bag_is_2 :
  ∃ (cost_5_pound_bag : ℝ), cost_5_pound_bag = remaining_cost :=
by
  sorry

end cost_of_5_pound_bag_is_2_l462_46238


namespace Carver_school_earnings_l462_46268

noncomputable def total_earnings_Carver_school : ℝ :=
  let base_payment := 20
  let total_payment := 900
  let Allen_days := 7 * 3
  let Balboa_days := 5 * 6
  let Carver_days := 4 * 10
  let total_student_days := Allen_days + Balboa_days + Carver_days
  let adjusted_total_payment := total_payment - 3 * base_payment
  let daily_wage := adjusted_total_payment / total_student_days
  daily_wage * Carver_days

theorem Carver_school_earnings : 
  total_earnings_Carver_school = 369.6 := 
by 
  sorry

end Carver_school_earnings_l462_46268


namespace product_of_consecutive_natural_numbers_l462_46277

theorem product_of_consecutive_natural_numbers (n : ℕ) : 
  (∃ t : ℕ, n = t * (t + 1) - 1) ↔ ∃ x : ℕ, n^2 - 1 = x * (x + 1) * (x + 2) * (x + 3) := 
sorry

end product_of_consecutive_natural_numbers_l462_46277


namespace project_completion_equation_l462_46288

variables (x : ℕ)

-- Project completion conditions
def person_A_time : ℕ := 12
def person_B_time : ℕ := 8
def A_initial_work_days : ℕ := 3

-- Work done by Person A when working alone for 3 days
def work_A_initial := (A_initial_work_days:ℚ) / person_A_time

-- Work done by Person A and B after the initial 3 days until completion
def combined_work_remaining := 
  (λ x:ℕ => ((x - A_initial_work_days):ℚ) * (1/person_A_time + 1/person_B_time))

-- The equation representing the total work done equals 1
theorem project_completion_equation (x : ℕ) : 
  (x:ℚ) / person_A_time + (x - A_initial_work_days:ℚ) / person_B_time = 1 :=
sorry

end project_completion_equation_l462_46288


namespace complex_number_in_first_quadrant_l462_46241

open Complex

theorem complex_number_in_first_quadrant (z : ℂ) (h : z = 1 / (1 - I)) : 
  z.re > 0 ∧ z.im > 0 :=
by
  sorry

end complex_number_in_first_quadrant_l462_46241


namespace compare_exp_square_l462_46254

theorem compare_exp_square (n : ℕ) : 
  (n ≥ 3 → 2^(2 * n) > (2 * n + 1)^2) ∧ ((n = 1 ∨ n = 2) → 2^(2 * n) < (2 * n + 1)^2) :=
by
  sorry

end compare_exp_square_l462_46254


namespace sequence_inequality_l462_46231
open Nat

variable (a : ℕ → ℝ)

noncomputable def conditions := 
  (a 1 ≥ 1) ∧ (∀ k : ℕ, a (k + 1) - a k ≥ 1)

theorem sequence_inequality (h : conditions a) : 
  ∀ n : ℕ, a (n + 1) ≥ n + 1 :=
sorry

end sequence_inequality_l462_46231


namespace largest_4_digit_integer_congruent_to_25_mod_26_l462_46236

theorem largest_4_digit_integer_congruent_to_25_mod_26 : ∃ x : ℕ, x < 10000 ∧ x ≥ 1000 ∧ x % 26 = 25 ∧ ∀ y : ℕ, y < 10000 ∧ y ≥ 1000 ∧ y % 26 = 25 → y ≤ x := by
  sorry

end largest_4_digit_integer_congruent_to_25_mod_26_l462_46236


namespace intersection_is_expected_l462_46260

open Set

def setA : Set ℝ := { x | (x + 1) / (x - 2) ≤ 0 }
def setB : Set ℝ := { x | x^2 - 4 * x + 3 ≤ 0 }
def expectedIntersection : Set ℝ := { x | 1 ≤ x ∧ x < 2 }

theorem intersection_is_expected :
  (setA ∩ setB) = expectedIntersection := by
  sorry

end intersection_is_expected_l462_46260


namespace smallest_y_l462_46227

theorem smallest_y (y : ℝ) :
  (3 * y ^ 2 + 33 * y - 90 = y * (y + 16)) → y = -10 :=
sorry

end smallest_y_l462_46227


namespace smallest_four_digit_multiple_of_18_l462_46226

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 18 ∣ n ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ 18 ∣ m → n ≤ m := by
  use 1008
  sorry

end smallest_four_digit_multiple_of_18_l462_46226


namespace proof_problem_solution_l462_46221

noncomputable def proof_problem (a b c d : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ (a * b + b * c + c * d + d * a = 1) →
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3)

theorem proof_problem_solution (a b c d : ℝ) : proof_problem a b c d :=
  sorry

end proof_problem_solution_l462_46221


namespace time_since_production_approximate_l462_46266

noncomputable def solve_time (N N₀ : ℝ) (t : ℝ) : Prop :=
  N = N₀ * (1 / 2) ^ (t / 5730) ∧
  N / N₀ = 3 / 8 ∧
  t = 8138

theorem time_since_production_approximate
  (N N₀ : ℝ)
  (h_decay : N = N₀ * (1 / 2) ^ (t / 5730))
  (h_ratio : N / N₀ = 3 / 8) :
  t = 8138 := 
sorry

end time_since_production_approximate_l462_46266


namespace polygon_area_is_12_l462_46265

def polygon_vertices := [(0,0), (4,0), (4,4), (2,4), (2,2), (0,2)]

def area_of_polygon (vertices : List (ℕ × ℕ)) : ℕ :=
  -- Function to compute the area (stub here for now)
  sorry

theorem polygon_area_is_12 :
  area_of_polygon polygon_vertices = 12 :=
by
  sorry

end polygon_area_is_12_l462_46265


namespace percentage_of_games_lost_l462_46230

theorem percentage_of_games_lost (games_won games_lost games_tied total_games : ℕ)
  (h_ratio : 5 * games_lost = 3 * games_won)
  (h_tied : games_tied * 5 = total_games) :
  (games_lost * 10 / total_games) = 3 :=
by sorry

end percentage_of_games_lost_l462_46230


namespace intersect_of_given_circles_l462_46203

noncomputable def circle_center (a b c : ℝ) : ℝ × ℝ :=
  let x := -a / 2
  let y := -b / 2
  (x, y)

noncomputable def radius_squared (a b c : ℝ) : ℝ :=
  (a / 2) ^ 2 + (b / 2) ^ 2 - c

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def circles_intersect (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  let center1 := circle_center a1 b1 c1
  let center2 := circle_center a2 b2 c2
  let r1 := Real.sqrt (radius_squared a1 b1 c1)
  let r2 := Real.sqrt (radius_squared a2 b2 c2)
  let d := distance center1 center2
  r1 - r2 < d ∧ d < r1 + r2

theorem intersect_of_given_circles :
  circles_intersect 4 3 2 2 3 1 :=
sorry

end intersect_of_given_circles_l462_46203


namespace problem_l462_46278

noncomputable def f (x : ℝ) := Real.log x + (x + 1) / x

noncomputable def g (x : ℝ) := x - 1/x - 2 * Real.log x

theorem problem 
  (x : ℝ) (hx : x > 0) (hxn1 : x ≠ 1) :
  f x > (x + 1) * Real.log x / (x - 1) :=
by
  sorry

end problem_l462_46278


namespace sin_cos_inequality_for_any_x_l462_46243

noncomputable def largest_valid_n : ℕ := 8

theorem sin_cos_inequality_for_any_x (n : ℕ) (h : n = largest_valid_n) :
  ∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 1 / n :=
sorry

end sin_cos_inequality_for_any_x_l462_46243


namespace discount_percentage_l462_46223

theorem discount_percentage (MP CP SP : ℝ)
  (h1 : CP = 0.64 * MP)
  (h2 : SP = CP * 1.375)
  (gain_percent : 37.5 = (SP - CP) / CP * 100) :
  (MP - SP) / MP * 100 = 12 :=
by
  sorry

end discount_percentage_l462_46223


namespace lines_passing_through_neg1_0_l462_46269

theorem lines_passing_through_neg1_0 (k : ℝ) :
  ∀ x y : ℝ, (y = k * (x + 1)) ↔ (x = -1 → y = 0 ∧ k ≠ 0) :=
by
  sorry

end lines_passing_through_neg1_0_l462_46269


namespace seeds_planted_l462_46284

theorem seeds_planted (seeds_per_bed : ℕ) (beds : ℕ) (total_seeds : ℕ) :
  seeds_per_bed = 10 → beds = 6 → total_seeds = seeds_per_bed * beds → total_seeds = 60 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end seeds_planted_l462_46284


namespace smallest_solution_is_neg_sqrt_13_l462_46210

noncomputable def smallest_solution (x : ℝ) : Prop :=
  x^4 - 26 * x^2 + 169 = 0 ∧ ∀ y : ℝ, y^4 - 26 * y^2 + 169 = 0 → x ≤ y

theorem smallest_solution_is_neg_sqrt_13 :
  smallest_solution (-Real.sqrt 13) :=
by
  sorry

end smallest_solution_is_neg_sqrt_13_l462_46210


namespace group_for_2019_is_63_l462_46259

def last_term_of_group (n : ℕ) : ℕ := (n * (n + 1)) / 2 + n

theorem group_for_2019_is_63 :
  ∃ n : ℕ, (2015 < 2019 ∧ 2019 ≤ 2079) :=
by
  sorry

end group_for_2019_is_63_l462_46259


namespace triangle_inequality_l462_46281

-- Define the lengths of the existing sticks
def a := 4
def b := 7

-- Define the list of potential third sticks
def potential_sticks := [3, 6, 11, 12]

-- Define the triangle inequality conditions
def valid_length (c : ℕ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

-- Prove that the valid length satisfying these conditions is 6
theorem triangle_inequality : ∃ c ∈ potential_sticks, valid_length c ∧ c = 6 :=
by
  sorry

end triangle_inequality_l462_46281


namespace Gwendolyn_will_take_50_hours_to_read_l462_46262

def GwendolynReadingTime (sentences_per_hour : ℕ) (sentences_per_paragraph : ℕ) (paragraphs_per_page : ℕ) (pages : ℕ) : ℕ :=
  (sentences_per_paragraph * paragraphs_per_page * pages) / sentences_per_hour

theorem Gwendolyn_will_take_50_hours_to_read 
  (h1 : 200 = 200)
  (h2 : 10 = 10)
  (h3 : 20 = 20)
  (h4 : 50 = 50) :
  GwendolynReadingTime 200 10 20 50 = 50 := by
  sorry

end Gwendolyn_will_take_50_hours_to_read_l462_46262


namespace calculation_l462_46280

theorem calculation : (1 / 2) ^ (-2 : ℤ) + (-1 : ℝ) ^ (2022 : ℤ) = 5 := by
  sorry

end calculation_l462_46280
