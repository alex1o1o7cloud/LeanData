import Mathlib

namespace percentage_of_b_l1880_188091

variable (a b c p : ℝ)

theorem percentage_of_b :
  (0.04 * a = 8) →
  (p * b = 4) →
  (c = b / a) →
  p = 1 / (50 * c) :=
by
  sorry

end percentage_of_b_l1880_188091


namespace total_birds_in_tree_l1880_188055

def initial_birds := 14
def additional_birds := 21

theorem total_birds_in_tree : initial_birds + additional_birds = 35 := by
  sorry

end total_birds_in_tree_l1880_188055


namespace compound_interest_rate_l1880_188009

theorem compound_interest_rate
  (A P : ℝ) (t n : ℝ)
  (HA : A = 1348.32)
  (HP : P = 1200)
  (Ht : t = 2)
  (Hn : n = 1) :
  ∃ r : ℝ, 0 ≤ r ∧ ((A / P) ^ (1 / (n * t)) - 1) = r ∧ r = 0.06 := 
sorry

end compound_interest_rate_l1880_188009


namespace white_stones_count_l1880_188082

/-- We define the total number of stones as a constant. -/
def total_stones : ℕ := 120

/-- We define the difference between white and black stones as a constant. -/
def white_minus_black : ℕ := 36

/-- The theorem states that if there are 120 go stones in total and 
    36 more white go stones than black go stones, then there are 78 white go stones. -/
theorem white_stones_count (W B : ℕ) (h1 : W = B + white_minus_black) (h2 : B + W = total_stones) : W = 78 := 
sorry

end white_stones_count_l1880_188082


namespace problem_statement_l1880_188005

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 4

theorem problem_statement (a x₁ x₂: ℝ) (ha : a > 0) (hx : x₁ < x₂) (hxsum : x₁ + x₂ = 0) :
  f a x₁ < f a x₂ := by
  sorry

end problem_statement_l1880_188005


namespace mutually_exclusive_not_complementary_l1880_188079

-- Define the basic events and conditions
structure Pocket :=
(red : ℕ)
(black : ℕ)

-- Define the event type
inductive Event
| atleast_one_black : Event
| both_black : Event
| atleast_one_red : Event
| both_red : Event
| exactly_one_black : Event
| exactly_two_black : Event
| none_black : Event

def is_mutually_exclusive (e1 e2 : Event) : Prop :=
  match e1, e2 with
  | Event.exactly_one_black, Event.exactly_two_black => true
  | Event.exactly_two_black, Event.exactly_one_black => true
  | _, _ => false

def is_complementary (e1 e2 : Event) : Prop :=
  e1 = Event.none_black ∧ e2 = Event.both_red ∨
  e1 = Event.both_red ∧ e2 = Event.none_black

-- Given conditions
def pocket : Pocket := { red := 2, black := 2 }

-- Proof problem setup
theorem mutually_exclusive_not_complementary : 
  is_mutually_exclusive Event.exactly_one_black Event.exactly_two_black ∧
  ¬ is_complementary Event.exactly_one_black Event.exactly_two_black :=
by
  sorry

end mutually_exclusive_not_complementary_l1880_188079


namespace fermat_little_theorem_l1880_188034

theorem fermat_little_theorem (p : ℕ) (a : ℤ) (hp : Nat.Prime p) (hcoprime : Int.gcd a p = 1) : 
  (a ^ (p - 1)) % p = 1 % p := 
sorry

end fermat_little_theorem_l1880_188034


namespace value_of_3_W_4_l1880_188085

def W (a b : ℤ) : ℤ := b + 5 * a - 3 * a ^ 2

theorem value_of_3_W_4 : W 3 4 = -8 :=
by
  sorry

end value_of_3_W_4_l1880_188085


namespace fermat_prime_solution_unique_l1880_188046

def is_fermat_prime (p : ℕ) : Prop :=
  ∃ r : ℕ, p = 2^(2^r) + 1

def problem_statement (p n k : ℕ) : Prop :=
  is_fermat_prime p ∧ p^n + n = (n + 1)^k

theorem fermat_prime_solution_unique (p n k : ℕ) :
  problem_statement p n k → (p, n, k) = (3, 1, 2) ∨ (p, n, k) = (5, 2, 3) :=
by
  sorry

end fermat_prime_solution_unique_l1880_188046


namespace ananthu_can_complete_work_in_45_days_l1880_188018

def amit_work_rate : ℚ := 1 / 15

def time_amit_worked : ℚ := 3

def total_work : ℚ := 1

def total_days : ℚ := 39

noncomputable def ananthu_days (x : ℚ) : Prop :=
  let amit_work_done := time_amit_worked * amit_work_rate
  let remaining_work := total_work - amit_work_done
  let ananthu_work_rate := remaining_work / (total_days - time_amit_worked)
  1 /x = ananthu_work_rate

theorem ananthu_can_complete_work_in_45_days :
  ananthu_days 45 :=
by
  sorry

end ananthu_can_complete_work_in_45_days_l1880_188018


namespace limit_of_p_n_is_tenth_l1880_188086

noncomputable def p_n (n : ℕ) : ℝ := sorry -- Definition of p_n needs precise formulation.

def tends_to_tenth_as_n_infty (p : ℕ → ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (p n - 1/10) < ε

theorem limit_of_p_n_is_tenth : tends_to_tenth_as_n_infty p_n := sorry

end limit_of_p_n_is_tenth_l1880_188086


namespace tangent_subtraction_identity_l1880_188023

theorem tangent_subtraction_identity (α β : ℝ) 
  (h1 : Real.tan α = -3/4) 
  (h2 : Real.tan (Real.pi - β) = 1/2) : 
  Real.tan (α - β) = -2/11 := 
sorry

end tangent_subtraction_identity_l1880_188023


namespace initial_volume_of_mixture_l1880_188057

theorem initial_volume_of_mixture (p q : ℕ) (x : ℕ) (h_ratio1 : p = 5 * x) (h_ratio2 : q = 3 * x) (h_added : q + 15 = 6 * x) (h_new_ratio : 5 * (3 * x + 15) = 6 * 5 * x) : 
  p + q = 40 :=
by
  sorry

end initial_volume_of_mixture_l1880_188057


namespace max_area_equilateral_in_rectangle_l1880_188004

-- Define the dimensions of the rectangle
def length_efgh : ℕ := 15
def width_efgh : ℕ := 8

-- The maximum possible area of an equilateral triangle inscribed in the rectangle
theorem max_area_equilateral_in_rectangle : 
  ∃ (s : ℝ), 
  s = ((16 * Real.sqrt 3) / 3) ∧ 
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ length_efgh → 
    (∃ (area : ℝ), area = (Real.sqrt 3 / 4 * s^2) ∧
      area = 64 * Real.sqrt 3)) :=
by sorry

end max_area_equilateral_in_rectangle_l1880_188004


namespace ratio_of_part_diminished_by_4_l1880_188071

theorem ratio_of_part_diminished_by_4 (N P : ℕ) (h1 : N = 160)
    (h2 : (1/5 : ℝ) * N + 4 = P - 4) : (P - 4) / N = 9 / 40 := 
by
  sorry

end ratio_of_part_diminished_by_4_l1880_188071


namespace trigonometric_identity_l1880_188000

theorem trigonometric_identity (θ : ℝ) (h : Real.tan (θ + Real.pi / 4) = 2) : 
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = -2 := 
sorry

end trigonometric_identity_l1880_188000


namespace percentage_decrease_of_b_l1880_188084

theorem percentage_decrease_of_b (a b x m : ℝ) (p : ℝ) 
  (a_pos : 0 < a) (b_pos : 0 < b)
  (h1 : a / b = 4 / 5)
  (h2 : x = a + 0.25 * a)
  (h3 : m = b * (1 - p / 100))
  (h4 : m / x = 0.4) :
  p = 60 :=
by
  sorry

end percentage_decrease_of_b_l1880_188084


namespace average_weight_of_16_boys_l1880_188035

theorem average_weight_of_16_boys :
  ∃ A : ℝ,
    (16 * A + 8 * 45.15 = 24 * 48.55) ∧
    A = 50.25 :=
by {
  -- Proof skipped, using sorry to denote the proof is required.
  sorry
}

end average_weight_of_16_boys_l1880_188035


namespace kendall_total_change_l1880_188083

-- Definition of values of coins
def value_of_quarters (q : ℕ) : ℝ := q * 0.25
def value_of_dimes (d : ℕ) : ℝ := d * 0.10
def value_of_nickels (n : ℕ) : ℝ := n * 0.05

-- Conditions
def quarters := 10
def dimes := 12
def nickels := 6

-- Theorem statement
theorem kendall_total_change : 
  value_of_quarters quarters + value_of_dimes dimes + value_of_nickels nickels = 4.00 :=
by
  sorry

end kendall_total_change_l1880_188083


namespace paving_stones_needed_l1880_188090

def length_courtyard : ℝ := 60
def width_courtyard : ℝ := 14
def width_stone : ℝ := 2
def paving_stones_required : ℕ := 140

theorem paving_stones_needed (L : ℝ) 
  (h1 : length_courtyard * width_courtyard = 840) 
  (h2 : paving_stones_required = 140)
  (h3 : (140 * (L * 2)) = 840) : 
  (length_courtyard * width_courtyard) / (L * width_stone) = 140 := 
by sorry

end paving_stones_needed_l1880_188090


namespace exists_fraction_equal_to_d_minus_1_l1880_188052

theorem exists_fraction_equal_to_d_minus_1 (n d : ℕ) (hdiv : d > 0 ∧ n % d = 0) :
  ∃ k : ℕ, k < n ∧ (n - k) / (n - (n - k)) = d - 1 :=
by
  sorry

end exists_fraction_equal_to_d_minus_1_l1880_188052


namespace smallest_k_sum_of_squares_multiple_of_200_l1880_188081

-- Define the sum of squares for positive integer k
def sum_of_squares (k : ℕ) : ℕ := (k * (k + 1) * (2 * k + 1)) / 6

-- Prove that the sum of squares for k = 112 is a multiple of 200
theorem smallest_k_sum_of_squares_multiple_of_200 :
  ∃ k : ℕ, sum_of_squares k = sum_of_squares 112 ∧ 200 ∣ sum_of_squares 112 :=
sorry

end smallest_k_sum_of_squares_multiple_of_200_l1880_188081


namespace Linda_original_savings_l1880_188025

variable (TV_cost : ℝ := 200) -- TV cost
variable (savings : ℝ) -- Linda's original savings

-- Prices, Discounts, Taxes
variable (sofa_price : ℝ := 600)
variable (sofa_discount : ℝ := 0.20)
variable (sofa_tax : ℝ := 0.05)

variable (dining_table_price : ℝ := 400)
variable (dining_table_discount : ℝ := 0.15)
variable (dining_table_tax : ℝ := 0.06)

variable (chair_set_price : ℝ := 300)
variable (chair_set_discount : ℝ := 0.25)
variable (chair_set_tax : ℝ := 0.04)

variable (coffee_table_price : ℝ := 100)
variable (coffee_table_discount : ℝ := 0.10)
variable (coffee_table_tax : ℝ := 0.03)

variable (service_charge_rate : ℝ := 0.02) -- Service charge rate

noncomputable def discounted_price_with_tax (price discount tax : ℝ) : ℝ :=
  let discounted_price := price * (1 - discount)
  let taxed_price := discounted_price * (1 + tax)
  taxed_price

noncomputable def total_furniture_cost : ℝ :=
  let sofa_cost := discounted_price_with_tax sofa_price sofa_discount sofa_tax
  let dining_table_cost := discounted_price_with_tax dining_table_price dining_table_discount dining_table_tax
  let chair_set_cost := discounted_price_with_tax chair_set_price chair_set_discount chair_set_tax
  let coffee_table_cost := discounted_price_with_tax coffee_table_price coffee_table_discount coffee_table_tax
  let combined_cost := sofa_cost + dining_table_cost + chair_set_cost + coffee_table_cost
  combined_cost * (1 + service_charge_rate)

theorem Linda_original_savings : savings = 4 * TV_cost ∧ savings / 4 * 3 = total_furniture_cost :=
by
  sorry -- Proof skipped

end Linda_original_savings_l1880_188025


namespace largest_x_satisfying_inequality_l1880_188047

theorem largest_x_satisfying_inequality :
  (∃ x : ℝ, 
    (∀ y : ℝ, |(y^2 - 4 * y - 39601)| ≥ |(y^2 + 4 * y - 39601)| → y ≤ x) ∧ 
    |(x^2 - 4 * x - 39601)| ≥ |(x^2 + 4 * x - 39601)|
  ) → x = 199 := 
sorry

end largest_x_satisfying_inequality_l1880_188047


namespace max_rectangle_area_l1880_188036

theorem max_rectangle_area (P : ℕ) (hP : P = 40) (l w : ℕ) (h : 2 * l + 2 * w = P) : ∃ A, A = l * w ∧ ∀ l' w', 2 * l' + 2 * w' = P → l' * w' ≤ 100 :=
by 
  sorry

end max_rectangle_area_l1880_188036


namespace carrots_picked_next_day_l1880_188073

-- Definitions based on conditions
def initial_carrots : Nat := 48
def carrots_thrown_away : Nat := 45
def total_carrots_next_day : Nat := 45

-- The proof problem statement
theorem carrots_picked_next_day : 
  (initial_carrots - carrots_thrown_away + x = total_carrots_next_day) → (x = 42) :=
by 
  sorry

end carrots_picked_next_day_l1880_188073


namespace cubed_sum_identity_l1880_188093

theorem cubed_sum_identity {x y : ℝ} (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end cubed_sum_identity_l1880_188093


namespace unreachable_y_l1880_188010

noncomputable def y_function (x : ℝ) : ℝ := (2 - 3 * x) / (5 * x - 1)

theorem unreachable_y : ¬ ∃ x : ℝ, y_function x = -3 / 5 ∧ x ≠ 1 / 5 :=
by {
  sorry
}

end unreachable_y_l1880_188010


namespace find_base_l1880_188044

theorem find_base (r : ℕ) : 
  (2 * r^2 + 1 * r + 0) + (2 * r^2 + 6 * r + 0) = 5 * r^2 + 0 * r + 0 → r = 7 :=
by
  sorry

end find_base_l1880_188044


namespace amaya_total_marks_l1880_188058

theorem amaya_total_marks 
  (m_a s_a a m m_s : ℕ) 
  (h_music : m_a = 70)
  (h_social_studies : s_a = m_a + 10)
  (h_maths_art_diff : m = a - 20)
  (h_maths_fraction : m = a - 1/10 * a)
  (h_maths_eq_fraction : m = 9/10 * a)
  (h_arts : 9/10 * a = a - 20)
  (h_total : m_a + s_a + a + m = 530) :
  m_a + s_a + a + m = 530 :=
by
  -- Proof to be completed
  sorry

end amaya_total_marks_l1880_188058


namespace find_N_l1880_188065

theorem find_N (N : ℕ) (h : (Real.sqrt 3 - 1)^N = 4817152 - 2781184 * Real.sqrt 3) : N = 16 :=
sorry

end find_N_l1880_188065


namespace number_of_new_terms_l1880_188097

theorem number_of_new_terms (n : ℕ) (h : n > 1) :
  (2^(n+1) - 1) - (2^n - 1) + 1 = 2^n := by
sorry

end number_of_new_terms_l1880_188097


namespace together_time_l1880_188043

theorem together_time (P_time Q_time : ℝ) (hP : P_time = 4) (hQ : Q_time = 6) : (1 / ((1 / P_time) + (1 / Q_time))) = 2.4 :=
by
  sorry

end together_time_l1880_188043


namespace turkey_2003_problem_l1880_188019

theorem turkey_2003_problem (x m n : ℕ) (hx : 0 < x) (hm : 0 < m) (hn : 0 < n) (h : x^m = 2^(2 * n + 1) + 2^n + 1) :
  x = 2^(2 * n + 1) + 2^n + 1 ∧ m = 1 ∨ x = 23 ∧ m = 2 ∧ n = 4 :=
sorry

end turkey_2003_problem_l1880_188019


namespace loss_percentage_is_11_l1880_188067

-- Constants for the given problem conditions
def cost_price : ℝ := 1500
def selling_price : ℝ := 1335

-- Formulation of the proof problem
theorem loss_percentage_is_11 :
  ((cost_price - selling_price) / cost_price) * 100 = 11 := by
  sorry

end loss_percentage_is_11_l1880_188067


namespace gas_cost_per_gallon_l1880_188029

-- Define the conditions as Lean definitions
def miles_per_gallon : ℕ := 32
def total_miles : ℕ := 336
def total_cost : ℕ := 42

-- Prove the cost of gas per gallon, which is $4 per gallon
theorem gas_cost_per_gallon : total_cost / (total_miles / miles_per_gallon) = 4 :=
by
  sorry

end gas_cost_per_gallon_l1880_188029


namespace construct_angle_from_19_l1880_188096

theorem construct_angle_from_19 (θ : ℝ) (h : θ = 19) : ∃ n : ℕ, (n * θ) % 360 = 75 :=
by
  -- Placeholder for the proof
  sorry

end construct_angle_from_19_l1880_188096


namespace unique_line_through_point_odd_x_prime_y_intercepts_l1880_188037

theorem unique_line_through_point_odd_x_prime_y_intercepts :
  ∃! (a b : ℕ), 0 < b ∧ Nat.Prime b ∧ a % 2 = 1 ∧
  (4 * b + 3 * a = a * b) :=
sorry

end unique_line_through_point_odd_x_prime_y_intercepts_l1880_188037


namespace circle_center_count_l1880_188015

noncomputable def num_circle_centers (b c d : ℝ) (h₁ : b < c) (h₂ : c ≤ d) : ℕ :=
  if (c = d) then 4 else 8

-- Here is the theorem statement
theorem circle_center_count (b c d : ℝ) (h₁ : b < c) (h₂ : c ≤ d) :
  num_circle_centers b c d h₁ h₂ = if (c = d) then 4 else 8 :=
sorry

end circle_center_count_l1880_188015


namespace calculate_difference_of_squares_l1880_188041

theorem calculate_difference_of_squares :
  (153^2 - 147^2) = 1800 :=
by
  sorry

end calculate_difference_of_squares_l1880_188041


namespace f_neg2_range_l1880_188003

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x

theorem f_neg2_range (h1 : 1 ≤ f (-1) ∧ f (-1) ≤ 2) (h2 : 2 ≤ f (1) ∧ f (1) ≤ 4) :
  ∀ k, f (-2) = k → 5 ≤ k ∧ k ≤ 10 :=
  sorry

end f_neg2_range_l1880_188003


namespace max_value_inequality_l1880_188088

theorem max_value_inequality (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  3 * x + 4 * y + 6 * z ≤ Real.sqrt 53 := by
  sorry

end max_value_inequality_l1880_188088


namespace games_in_each_box_l1880_188063

theorem games_in_each_box (start_games sold_games total_boxes remaining_games games_per_box : ℕ) 
  (h_start: start_games = 35) (h_sold: sold_games = 19) (h_boxes: total_boxes = 2) 
  (h_remaining: remaining_games = start_games - sold_games) 
  (h_per_box: games_per_box = remaining_games / total_boxes) : games_per_box = 8 :=
by
  sorry

end games_in_each_box_l1880_188063


namespace total_pencils_is_5_l1880_188012

-- Define the initial number of pencils and the number of pencils Tim added
def initial_pencils : Nat := 2
def pencils_added_by_tim : Nat := 3

-- Prove the total number of pencils is equal to 5
theorem total_pencils_is_5 : initial_pencils + pencils_added_by_tim = 5 := by
  sorry

end total_pencils_is_5_l1880_188012


namespace molly_total_cost_l1880_188062

def cost_per_package : ℕ := 5
def num_parents : ℕ := 2
def num_brothers : ℕ := 3
def num_children_per_brother : ℕ := 2
def num_spouse_per_brother : ℕ := 1

def total_num_relatives : ℕ := 
  let parents_and_siblings := num_parents + num_brothers
  let additional_relatives := num_brothers * (1 + num_spouse_per_brother + num_children_per_brother)
  parents_and_siblings + additional_relatives

def total_cost : ℕ :=
  total_num_relatives * cost_per_package

theorem molly_total_cost : total_cost = 85 := sorry

end molly_total_cost_l1880_188062


namespace gravel_amount_l1880_188087

theorem gravel_amount (total_material sand gravel : ℝ) 
  (h1 : total_material = 14.02) 
  (h2 : sand = 8.11) 
  (h3 : gravel = total_material - sand) : 
  gravel = 5.91 :=
  sorry

end gravel_amount_l1880_188087


namespace find_angle3_l1880_188080

theorem find_angle3 (angle1 angle2 angle3 : ℝ)
  (h1 : angle1 + angle2 = 90)
  (h2 : angle2 + angle3 = 180)
  (h3 : angle1 = 20) :
  angle3 = 110 :=
sorry

end find_angle3_l1880_188080


namespace quadratic_roots_square_l1880_188068

theorem quadratic_roots_square (q : ℝ) :
  (∃ a : ℝ, a + a^2 = 12 ∧ q = a * a^2) → (q = 27 ∨ q = -64) :=
by
  sorry

end quadratic_roots_square_l1880_188068


namespace simplify_expression_l1880_188061

theorem simplify_expression :
  1 + 1 / (1 + 1 / (2 + 2)) = 9 / 5 := by
  sorry

end simplify_expression_l1880_188061


namespace least_integer_greater_than_sqrt_500_l1880_188032

theorem least_integer_greater_than_sqrt_500 : 
  ∃ n : ℤ, (∀ m : ℤ, m * m ≤ 500 → m < n) ∧ n = 23 :=
by
  sorry

end least_integer_greater_than_sqrt_500_l1880_188032


namespace monthly_growth_rate_l1880_188060

-- Definitions based on the conditions given in the original problem.
def final_height : ℝ := 80
def current_height : ℝ := 20
def months_in_year : ℕ := 12

-- Prove the monthly growth rate.
theorem monthly_growth_rate : (final_height - current_height) / months_in_year = 5 := by
  sorry

end monthly_growth_rate_l1880_188060


namespace subtraction_correct_l1880_188050

theorem subtraction_correct :
  2222222222222 - 1111111111111 = 1111111111111 := by
  sorry

end subtraction_correct_l1880_188050


namespace cells_count_after_9_days_l1880_188028

theorem cells_count_after_9_days :
  let a := 5
  let r := 3
  let n := 3
  a * r^(n-1) = 45 :=
by
  let a := 5
  let r := 3
  let n := 3
  sorry

end cells_count_after_9_days_l1880_188028


namespace cos_triple_angle_l1880_188022

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 :=
by
  sorry

end cos_triple_angle_l1880_188022


namespace wine_division_l1880_188024

theorem wine_division (m n : ℕ) (m_pos : m > 0) (n_pos : n > 0) :
  (∃ k, k = (m + n) / 2 ∧ k * 2 = (m + n) ∧ k % Nat.gcd m n = 0) ↔ 
  (m + n) % 2 = 0 ∧ ((m + n) / 2) % Nat.gcd m n = 0 :=
by
  sorry

end wine_division_l1880_188024


namespace average_problem_l1880_188072

theorem average_problem
  (h : (20 + 40 + 60) / 3 = (x + 50 + 45) / 3 + 5) :
  x = 10 :=
by
  sorry

end average_problem_l1880_188072


namespace sum_of_squares_is_149_l1880_188011

-- Define the integers and their sum and product
def integers_sum (b : ℤ) : ℤ := (b - 1) + b + (b + 1)
def integers_product (b : ℤ) : ℤ := (b - 1) * b * (b + 1)

-- Define the condition given in the problem
def condition (b : ℤ) : Prop :=
  integers_product b = 12 * integers_sum b + b^2

-- Define the sum of squares of three consecutive integers
def sum_of_squares (b : ℤ) : ℤ :=
  (b - 1)^2 + b^2 + (b + 1)^2

-- The main statement to be proved
theorem sum_of_squares_is_149 (b : ℤ) (h : condition b) : sum_of_squares b = 149 :=
by
  sorry

end sum_of_squares_is_149_l1880_188011


namespace arithmetic_sequence_a15_l1880_188040

theorem arithmetic_sequence_a15 {a : ℕ → ℝ} (d : ℝ) (a7 a23 : ℝ) 
    (h1 : a 7 = 8) (h2 : a 23 = 22) : 
    a 15 = 15 := 
by
  sorry

end arithmetic_sequence_a15_l1880_188040


namespace part_one_part_two_part_three_l1880_188077

-- Define the sequence and the sum of its first n terms
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a n - 2 ^ n

-- Prove that a_1 = 2 and a_4 = 40
theorem part_one (a : ℕ → ℕ) (h : ∀ n, S a n = 2 * a n - 2 ^ n) : 
  a 1 = 2 ∧ a 4 = 40 := by
  sorry
  
-- Prove that the sequence {a_{n+1} - 2a_n} is a geometric sequence
theorem part_two (a : ℕ → ℕ) (h : ∀ n, S a n = 2 * a n - 2 ^ n) : 
  ∃ r : ℕ, (r = 2) ∧ (∀ n, (a (n + 1) - 2 * a n) = r ^ n) := by
  sorry

-- Prove the general term formula for the sequence {a_n}
theorem part_three (a : ℕ → ℕ) (h : ∀ n, S a n = 2 * a n - 2 ^ n) : 
  ∀ n, a n = 2 ^ (n + 1) - 2 := by
  sorry

end part_one_part_two_part_three_l1880_188077


namespace students_prefer_dogs_l1880_188031

theorem students_prefer_dogs (total_students : ℕ) (perc_dogs_vg perc_dogs_mv : ℕ) (h_total: total_students = 30)
  (h_perc_dogs_vg: perc_dogs_vg = 50) (h_perc_dogs_mv: perc_dogs_mv = 10) :
  total_students * perc_dogs_vg / 100 + total_students * perc_dogs_mv / 100 = 18 := by
  sorry

end students_prefer_dogs_l1880_188031


namespace sandbox_width_l1880_188064

theorem sandbox_width :
  ∀ (length area width : ℕ), length = 312 → area = 45552 →
  area = length * width → width = 146 :=
by
  intros length area width h_length h_area h_eq
  sorry

end sandbox_width_l1880_188064


namespace breadth_of_water_tank_l1880_188059

theorem breadth_of_water_tank (L H V : ℝ) (n : ℕ) (avg_displacement : ℝ) (total_displacement : ℝ)
  (h_len : L = 40)
  (h_height : H = 0.25)
  (h_avg_disp : avg_displacement = 4)
  (h_number : n = 50)
  (h_total_disp : total_displacement = avg_displacement * n)
  (h_displacement_value : total_displacement = 200) :
  (40 * B * 0.25 = 200) → B = 20 :=
by
  intro h_eq
  sorry

end breadth_of_water_tank_l1880_188059


namespace todd_savings_l1880_188042

def original_price : ℝ := 125
def sale_discount : ℝ := 0.20
def coupon : ℝ := 10
def credit_card_discount : ℝ := 0.10
def rebate : ℝ := 0.05
def sales_tax : ℝ := 0.08

def calculate_savings (original_price sale_discount coupon credit_card_discount rebate sales_tax : ℝ) : ℝ :=
  let after_sale := original_price * (1 - sale_discount)
  let after_coupon := after_sale - coupon
  let after_credit_card := after_coupon * (1 - credit_card_discount)
  let after_rebate := after_credit_card * (1 - rebate)
  let tax := after_credit_card * sales_tax
  let final_price := after_rebate + tax
  original_price - final_price

theorem todd_savings : calculate_savings 125 0.20 10 0.10 0.05 0.08 = 41.57 :=
by
  sorry

end todd_savings_l1880_188042


namespace binom_10_4_eq_210_l1880_188017

theorem binom_10_4_eq_210 : Nat.choose 10 4 = 210 :=
  by sorry

end binom_10_4_eq_210_l1880_188017


namespace dice_probability_l1880_188089

noncomputable def probability_event (event_count : ℕ) (total_count : ℕ) : ℚ := 
  event_count / total_count

theorem dice_probability :
  let event_first_die := 3
  let event_second_die := 3
  let total_outcomes_first := 8
  let total_outcomes_second := 8
  probability_event event_first_die total_outcomes_first * probability_event event_second_die total_outcomes_second = 9 / 64 :=
by
  sorry

end dice_probability_l1880_188089


namespace correct_answer_l1880_188075

variables (x y : ℝ)

def cost_equations (x y : ℝ) : Prop :=
  (2 * x + 3 * y = 120) ∧ (2 * x - y = 20)

theorem correct_answer : cost_equations x y :=
sorry

end correct_answer_l1880_188075


namespace math_competition_question_1_math_competition_question_2_l1880_188026

noncomputable def participant_score_probabilities : Prop :=
  let P1 := (3 / 5)^2 * (2 / 5)^2
  let P2 := 2 * (3 / 5) * (2 / 5)
  let P3 := 2 * (3 / 5) * (2 / 5)^2
  let P4 := (3 / 5)^2
  P1 + P2 + P3 + P4 = 208 / 625

noncomputable def winning_probabilities : Prop :=
  let P_100_or_more := (4 / 5)^8 * (3 / 5)^3 + 3 * (4 / 5)^8 * (3 / 5)^2 * (2 / 5) + 
                      (8 * (4 / 5)^7 * (1/5) * (3 / 5)^3 + 
                      28 * (4 / 5)^6 * (1/5)^2 * (3 / 5)^3)
  let winning_if_100_or_more := P_100_or_more * (9 / 10)
  let winning_if_less_100 := (1 - P_100_or_more) * (2 / 5)
  winning_if_100_or_more + winning_if_less_100 ≥ 1 / 2

theorem math_competition_question_1 : participant_score_probabilities :=
by sorry

theorem math_competition_question_2 : winning_probabilities :=
by sorry

end math_competition_question_1_math_competition_question_2_l1880_188026


namespace correct_expression_l1880_188020

theorem correct_expression :
  ¬ (|4| = -4) ∧
  ¬ (|4| = -4) ∧
  (-(4^2) ≠ 16)  ∧
  ((-4)^2 = 16) := by
  sorry

end correct_expression_l1880_188020


namespace set_representation_listing_method_l1880_188076

def is_in_set (a : ℤ) : Prop := 0 < 2 * a - 1 ∧ 2 * a - 1 ≤ 5

def M : Set ℤ := {a | is_in_set a}

theorem set_representation_listing_method :
  M = {1, 2, 3} :=
sorry

end set_representation_listing_method_l1880_188076


namespace vegetarian_count_l1880_188051

variables (v_only v_nboth vegan pesc nvboth : ℕ)
variables (hv_only : v_only = 13) (hv_nboth : v_nboth = 8)
          (hvegan_tot : vegan = 5) (hvegan_v : vveg1 = 3)
          (hpesc_tot : pesc = 4) (hpesc_vnboth : nvboth = 2)

theorem vegetarian_count (total_veg : ℕ) 
  (H_total : total_veg = v_only + v_nboth + (vegan - vveg1)) :
  total_veg = 23 :=
sorry

end vegetarian_count_l1880_188051


namespace find_fraction_l1880_188092

noncomputable def condition_eq : ℝ := 5
noncomputable def condition_gq : ℝ := 7

theorem find_fraction {FQ HQ : ℝ} (h : condition_eq * FQ = condition_gq * HQ) :
  FQ / HQ = 7 / 5 :=
by
  have eq_mul : condition_eq = 5 := by rfl
  have gq_mul : condition_gq = 7 := by rfl
  rw [eq_mul, gq_mul] at h
  have h': 5 * FQ = 7 * HQ := h
  field_simp [←h']
  sorry

end find_fraction_l1880_188092


namespace simplify_expression_l1880_188002

theorem simplify_expression (x : ℝ) (h : x^2 + x - 6 = 0) : 
  (x - 1) / ((2 / (x - 1)) - 1) = 8 / 3 :=
sorry

end simplify_expression_l1880_188002


namespace solve_for_x_l1880_188095

theorem solve_for_x (x : ℝ) : (3 : ℝ)^(4 * x^2 - 3 * x + 5) = (3 : ℝ)^(4 * x^2 + 9 * x - 6) ↔ x = 11 / 12 :=
by sorry

end solve_for_x_l1880_188095


namespace find_difference_l1880_188066

noncomputable def g : ℝ → ℝ := sorry    -- Definition of the function g (since it's graph-based and specific)

-- Given conditions
variables (c d : ℝ)
axiom h1 : Function.Injective g          -- g is an invertible function (injective functions have inverses)
axiom h2 : g c = d
axiom h3 : g d = 6

-- Theorem to prove
theorem find_difference : c - d = -2 :=
by {
  -- sorry is needed since the exact proof steps are not provided
  sorry
}

end find_difference_l1880_188066


namespace hiring_probabilities_l1880_188054

-- Define the candidates and their abilities
inductive Candidate : Type
| Strong
| Moderate
| Weak

open Candidate

-- Define the ordering rule and hiring rule
def interviewOrders : List (Candidate × Candidate × Candidate) :=
  [(Strong, Moderate, Weak), (Strong, Weak, Moderate), 
   (Moderate, Strong, Weak), (Moderate, Weak, Strong),
   (Weak, Strong, Moderate), (Weak, Moderate, Strong)]

def hiresStrong (order : Candidate × Candidate × Candidate) : Bool :=
  match order with
  | (Moderate, Strong, Weak) => true
  | (Moderate, Weak, Strong) => true
  | (Weak, Strong, Moderate) => true
  | _ => false

def hiresModerate (order : Candidate × Candidate × Candidate) : Bool :=
  match order with
  | (Strong, Weak, Moderate) => true
  | (Weak, Moderate, Strong) => true
  | _ => false

-- The main theorem to be proved
theorem hiring_probabilities :
  let orders := interviewOrders
  let p := (orders.filter hiresStrong).length / orders.length
  let q := (orders.filter hiresModerate).length / orders.length
  p = 1 / 2 ∧ q = 1 / 3 := by
  sorry

end hiring_probabilities_l1880_188054


namespace gigi_ate_33_bananas_l1880_188056

def gigi_bananas (total_bananas : ℕ) (days : ℕ) (diff : ℕ) (bananas_day_7 : ℕ) : Prop :=
  ∃ b, (days * b + diff * ((days * (days - 1)) / 2)) = total_bananas ∧ 
       (b + 6 * diff) = bananas_day_7

theorem gigi_ate_33_bananas :
  gigi_bananas 150 7 4 33 :=
by {
  sorry
}

end gigi_ate_33_bananas_l1880_188056


namespace probability_of_point_on_line_4_l1880_188099

-- Definitions as per conditions
def total_outcomes : ℕ := 36
def favorable_points : Finset (ℕ × ℕ) := {(1, 3), (2, 2), (3, 1)}
def probability : ℚ := (favorable_points.card : ℚ) / total_outcomes

-- Problem statement to prove
theorem probability_of_point_on_line_4 :
  probability = 1 / 12 :=
by
  sorry

end probability_of_point_on_line_4_l1880_188099


namespace problem1_problem2_l1880_188039

-- Define propositions P and Q under the given conditions
def P (a x : ℝ) : Prop := 2 * x^2 - 5 * a * x - 3 * a^2 < 0

def Q (x : ℝ) : Prop := (2 * Real.sin x > 1) ∧ (x^2 - x - 2 < 0)

-- Problem 1: Prove that if a = 2 and p ∧ q holds true, then the range of x is (π/6, 2)
theorem problem1 (x : ℝ) (hx1 : P 2 x ∧ Q x) : (Real.pi / 6 < x ∧ x < 2) :=
sorry

-- Problem 2: Prove that if ¬P is a sufficient but not necessary condition for ¬Q, then the range of a is [2/3, ∞)
theorem problem2 (a : ℝ) (h₁ : ∀ x, Q x → P a x) (h₂ : ∃ x, Q x → ¬P a x) : a ≥ 2 / 3 :=
sorry

end problem1_problem2_l1880_188039


namespace square_diff_correctness_l1880_188038

theorem square_diff_correctness (x y : ℝ) :
  let A := (x + y) * (x - 2*y)
  let B := (x + y) * (-x + y)
  let C := (x + y) * (-x - y)
  let D := (-x + y) * (x - y)
  (∃ (a b : ℝ), B = (a + b) * (a - b)) ∧ (∀ (p q : ℝ), A ≠ (p + q) * (p - q)) ∧ (∀ (r s : ℝ), C ≠ (r + s) * (r - s)) ∧ (∀ (t u : ℝ), D ≠ (t + u) * (t - u)) :=
by
  sorry

end square_diff_correctness_l1880_188038


namespace a_3_equals_35_l1880_188070

noncomputable def S (n : ℕ) : ℕ := 5 * n ^ 2 + 10 * n
noncomputable def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_3_equals_35 : a 3 = 35 := by
  sorry

end a_3_equals_35_l1880_188070


namespace taxi_fare_function_l1880_188078

theorem taxi_fare_function (x : ℝ) (h : x > 3) : 
  ∃ y : ℝ, y = 2 * x + 4 :=
by
  sorry

end taxi_fare_function_l1880_188078


namespace monotonically_decreasing_iff_l1880_188074

noncomputable def f (a x : ℝ) : ℝ := (x^2 - 2 * a * x) * Real.exp x

theorem monotonically_decreasing_iff (a : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≤ f a (-1) ∧ f a x ≤ f a 1) ↔ (a ≥ 3 / 4) :=
by
  sorry

end monotonically_decreasing_iff_l1880_188074


namespace desiree_age_l1880_188045

-- Definitions of the given variables and conditions
variables (D C : ℝ)

-- Given conditions
def condition1 : Prop := D = 2 * C
def condition2 : Prop := D + 30 = 0.6666666 * (C + 30) + 14
def condition3 : Prop := D = 2.99999835

-- Main theorem to prove
theorem desiree_age : D = 2.99999835 :=
by
  { sorry }

end desiree_age_l1880_188045


namespace ratio_of_sweater_vests_to_shirts_l1880_188016

theorem ratio_of_sweater_vests_to_shirts (S V O : ℕ) (h1 : S = 3) (h2 : O = 18) (h3 : O = V * S) : (V : ℚ) / (S : ℚ) = 2 := 
  by
  sorry

end ratio_of_sweater_vests_to_shirts_l1880_188016


namespace martin_rings_big_bell_l1880_188048

/-
Problem Statement:
Martin rings the small bell 4 times more than 1/3 as often as the big bell.
If he rings both of them a combined total of 52 times, prove that he rings the big bell 36 times.
-/

theorem martin_rings_big_bell (s b : ℕ) 
  (h1 : s + b = 52) 
  (h2 : s = 4 + (1 / 3 : ℚ) * b) : 
  b = 36 := 
by
  sorry

end martin_rings_big_bell_l1880_188048


namespace gcd_of_numbers_l1880_188021

theorem gcd_of_numbers :
  let a := 125^2 + 235^2 + 349^2
  let b := 124^2 + 234^2 + 350^2
  gcd a b = 1 := by
  sorry

end gcd_of_numbers_l1880_188021


namespace tax_rate_equals_65_l1880_188027

def tax_rate_percentage := 65
def tax_rate_per_dollars (rate_percentage : ℕ) : ℕ :=
  (rate_percentage / 100) * 100

theorem tax_rate_equals_65 :
  tax_rate_per_dollars tax_rate_percentage = 65 := by
  sorry

end tax_rate_equals_65_l1880_188027


namespace factorize_expression_l1880_188094

theorem factorize_expression (x : ℝ) : x^3 - 4 * x^2 + 4 * x = x * (x - 2)^2 :=
by
  sorry

end factorize_expression_l1880_188094


namespace mean_inequality_l1880_188001

variable (a b : ℝ)

-- Conditions: a and b are distinct and non-zero
axiom h₀ : a ≠ b
axiom h₁ : a ≠ 0
axiom h₂ : b ≠ 0

theorem mean_inequality (h₀ : a ≠ b) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : 
  (a^2 + b^2) / 2 > (a + b) / 2 ∧ (a + b) / 2 > Real.sqrt (a * b) :=
sorry -- Proof is not provided, only statement.

end mean_inequality_l1880_188001


namespace triangle_angle_D_l1880_188033

theorem triangle_angle_D (F E D : ℝ) (hF : F = 15) (hE : E = 3 * F) (h_triangle : D + E + F = 180) : D = 120 := by
  sorry

end triangle_angle_D_l1880_188033


namespace solve_system_and_find_6a_plus_b_l1880_188014

theorem solve_system_and_find_6a_plus_b (x y a b : ℝ)
  (h1 : 3 * x - 2 * y + 20 = 0)
  (h2 : 2 * x + 15 * y - 3 = 0)
  (h3 : a * x - b * y = 3) :
  6 * a + b = -3 := by
  sorry

end solve_system_and_find_6a_plus_b_l1880_188014


namespace sum_of_cubes_l1880_188030

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
sorry

end sum_of_cubes_l1880_188030


namespace raisins_in_other_boxes_l1880_188008

theorem raisins_in_other_boxes (total_raisins : ℕ) (raisins_box1 : ℕ) (raisins_box2 : ℕ) (other_boxes : ℕ) (num_other_boxes : ℕ) :
  total_raisins = 437 →
  raisins_box1 = 72 →
  raisins_box2 = 74 →
  num_other_boxes = 3 →
  other_boxes = (total_raisins - raisins_box1 - raisins_box2) / num_other_boxes →
  other_boxes = 97 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end raisins_in_other_boxes_l1880_188008


namespace distance_difference_l1880_188053

-- Definition of speeds and time
def speed_alberto : ℕ := 16
def speed_clara : ℕ := 12
def time_hours : ℕ := 5

-- Distance calculation functions
def distance (speed time : ℕ) : ℕ := speed * time

-- Main theorem statement
theorem distance_difference : 
  distance speed_alberto time_hours - distance speed_clara time_hours = 20 :=
by
  sorry

end distance_difference_l1880_188053


namespace unique_function_f_l1880_188049

theorem unique_function_f (f : ℝ → ℝ)
    (h1 : ∀ x : ℝ, f x = -f (-x))
    (h2 : ∀ x : ℝ, f (x + 1) = f x + 1)
    (h3 : ∀ x : ℝ, x ≠ 0 → f (1 / x) = 1 / x^2 * f x) :
    ∀ x : ℝ, f x = x := 
sorry

end unique_function_f_l1880_188049


namespace maximize_profit_l1880_188006

def cost_A : ℝ := 3
def price_A : ℝ := 3.3
def cost_B : ℝ := 2.4
def price_B : ℝ := 2.8
def total_devices : ℕ := 50

def profit (x : ℕ) : ℝ := (price_A - cost_A) * x + (price_B - cost_B) * (total_devices - x)

def functional_relationship (x : ℕ) : ℝ := -0.1 * x + 20

def purchase_condition (x : ℕ) : Prop := 4 * x ≥ total_devices - x

theorem maximize_profit :
    functional_relationship (10) = 19 ∧ 
    (∀ x : ℕ, purchase_condition x → functional_relationship x ≤ 19) :=
by {
    -- Proof omitted
    sorry
}

end maximize_profit_l1880_188006


namespace find_m_for_parallel_lines_l1880_188007

theorem find_m_for_parallel_lines (m : ℝ) :
  (∀ x y : ℝ, (3 + m) * x + 4 * y = 5 - 3 * m) →
  (∀ x y : ℝ, 2 * x + (5 + m) * y = 8) →
  m = -7 :=
by
  sorry

end find_m_for_parallel_lines_l1880_188007


namespace solve_for_B_l1880_188069

theorem solve_for_B (B : ℕ) (h : 3 * B + 2 = 20) : B = 6 :=
by 
  -- This is just a placeholder, the proof will go here
  sorry

end solve_for_B_l1880_188069


namespace total_area_three_plots_l1880_188013

variable (x y z A : ℝ)

theorem total_area_three_plots :
  (x = (2 / 5) * A) →
  (z = x - 16) →
  (y = (9 / 8) * z) →
  (A = x + y + z) →
  A = 96 :=
by
  intros h1 h2 h3 h4
  sorry

end total_area_three_plots_l1880_188013


namespace problem_l1880_188098

structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

def NotParallel (v1 v2 : Vector3D) : Prop := ¬ ∃ k : ℝ, v2 = ⟨k * v1.x, k * v1.y, k * v1.z⟩

def a : Vector3D := ⟨1, 2, -2⟩
def b : Vector3D := ⟨-2, -4, 4⟩
def c : Vector3D := ⟨1, 0, 0⟩
def d : Vector3D := ⟨-3, 0, 0⟩
def g : Vector3D := ⟨-2, 3, 5⟩
def h : Vector3D := ⟨16, 24, 40⟩
def e : Vector3D := ⟨2, 3, 0⟩
def f : Vector3D := ⟨0, 0, 0⟩

theorem problem : NotParallel g h := by
  sorry

end problem_l1880_188098
