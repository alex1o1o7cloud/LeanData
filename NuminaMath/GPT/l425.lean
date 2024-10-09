import Mathlib

namespace train_crosses_platform_in_15_seconds_l425_42512

-- Definitions based on conditions
def length_of_train : ℝ := 330 -- in meters
def tunnel_length : ℝ := 1200 -- in meters
def time_to_cross_tunnel : ℝ := 45 -- in seconds
def platform_length : ℝ := 180 -- in meters

-- Definition based on the solution but directly asserting the correct answer.
def time_to_cross_platform : ℝ := 15 -- in seconds

-- Lean statement
theorem train_crosses_platform_in_15_seconds :
  (length_of_train + platform_length) / ((length_of_train + tunnel_length) / time_to_cross_tunnel) = time_to_cross_platform :=
by
  sorry

end train_crosses_platform_in_15_seconds_l425_42512


namespace min_value_of_sum_of_squares_l425_42507

theorem min_value_of_sum_of_squares (a b c : ℝ) (h : a + b + c = 1) : a^2 + b^2 + c^2 ≥ 1 / 3 :=
by
  sorry

end min_value_of_sum_of_squares_l425_42507


namespace bill_due_months_l425_42583

theorem bill_due_months {TD A: ℝ} (R: ℝ) : 
  TD = 189 → A = 1764 → R = 16 → 
  ∃ M: ℕ, A - TD * (1 + (R/100) * (M/12)) = 1764 - 189 * (1 + (16/100) * (10/12)) ∧ M = 10 :=
by
  intro hTD hA hR
  use 10
  sorry

end bill_due_months_l425_42583


namespace base_length_of_isosceles_triangle_l425_42564

-- Definitions based on given conditions
def is_isosceles (a b : ℕ) (c : ℕ) :=
a = b ∧ c = c

def side_length : ℕ := 6
def perimeter : ℕ := 20

-- Theorem to prove the base length
theorem base_length_of_isosceles_triangle (b : ℕ) (h1 : 2 * side_length + b = perimeter) :
  b = 8 :=
sorry

end base_length_of_isosceles_triangle_l425_42564


namespace solve_for_x_l425_42575

def star (a b : ℝ) : ℝ := a * b + 3 * b - 2 * a

theorem solve_for_x (x : ℝ) : star 6 x = 45 ↔ x = 19 / 3 := by
  sorry

end solve_for_x_l425_42575


namespace marsha_first_package_miles_l425_42546

noncomputable def total_distance (x : ℝ) : ℝ := x + 28 + 14

noncomputable def earnings (x : ℝ) : ℝ := total_distance x * 2

theorem marsha_first_package_miles : ∃ x : ℝ, earnings x = 104 ∧ x = 10 :=
by
  use 10
  sorry

end marsha_first_package_miles_l425_42546


namespace intersection_A_B_complement_A_in_U_complement_B_in_U_l425_42585

-- Definitions and conditions
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {5, 6, 7, 8}
def B : Set ℕ := {2, 4, 6, 8}

-- Problems to prove
theorem intersection_A_B : A ∩ B = {6, 8} := by
  sorry

theorem complement_A_in_U : U \ A = {1, 2, 3, 4} := by
  sorry

theorem complement_B_in_U : U \ B = {1, 3, 5, 7} := by
  sorry

end intersection_A_B_complement_A_in_U_complement_B_in_U_l425_42585


namespace meals_without_restrictions_l425_42569

theorem meals_without_restrictions (total_clients vegan kosher gluten_free halal dairy_free nut_free vegan_kosher vegan_gluten_free kosher_gluten_free halal_dairy_free gluten_free_nut_free vegan_halal_gluten_free kosher_dairy_free_nut_free : ℕ) 
  (h_tc : total_clients = 80)
  (h_vegan : vegan = 15)
  (h_kosher : kosher = 18)
  (h_gluten_free : gluten_free = 12)
  (h_halal : halal = 10)
  (h_dairy_free : dairy_free = 8)
  (h_nut_free : nut_free = 4)
  (h_vegan_kosher : vegan_kosher = 5)
  (h_vegan_gluten_free : vegan_gluten_free = 6)
  (h_kosher_gluten_free : kosher_gluten_free = 3)
  (h_halal_dairy_free : halal_dairy_free = 4)
  (h_gluten_free_nut_free : gluten_free_nut_free = 2)
  (h_vegan_halal_gluten_free : vegan_halal_gluten_free = 2)
  (h_kosher_dairy_free_nut_free : kosher_dairy_free_nut_free = 1) : 
  (total_clients - (vegan + kosher + gluten_free + halal + dairy_free + nut_free 
  - vegan_kosher - vegan_gluten_free - kosher_gluten_free - halal_dairy_free - gluten_free_nut_free 
  + vegan_halal_gluten_free + kosher_dairy_free_nut_free) = 30) :=
by {
  -- solution steps here
  sorry
}

end meals_without_restrictions_l425_42569


namespace max_ski_trips_l425_42580

/--
The ski lift carries skiers from the bottom of the mountain to the top, taking 15 minutes each way, 
and it takes 5 minutes to ski back down the mountain. 
Given that the total available time is 2 hours, prove that the maximum number of trips 
down the mountain in that time is 6.
-/
theorem max_ski_trips (ride_up_time : ℕ) (ski_down_time : ℕ) (total_time : ℕ) :
  ride_up_time = 15 →
  ski_down_time = 5 →
  total_time = 120 →
  (total_time / (ride_up_time + ski_down_time) = 6) :=
by
  intros h1 h2 h3
  sorry

end max_ski_trips_l425_42580


namespace additional_oil_needed_l425_42501

variable (oil_per_cylinder : ℕ) (number_of_cylinders : ℕ) (oil_already_added : ℕ)

theorem additional_oil_needed (h1 : oil_per_cylinder = 8) (h2 : number_of_cylinders = 6) (h3 : oil_already_added = 16) :
  oil_per_cylinder * number_of_cylinders - oil_already_added = 32 :=
by
  -- proof here
  sorry

end additional_oil_needed_l425_42501


namespace greatest_sum_of_consecutive_odd_integers_lt_500_l425_42589

-- Define the consecutive odd integers and their conditions
def consecutive_odd_integers (n : ℤ) : Prop :=
  n % 2 = 1 ∧ (n + 2) % 2 = 1

-- Define the condition that their product must be less than 500
def prod_less_500 (n : ℤ) : Prop :=
  n * (n + 2) < 500

-- The theorem statement
theorem greatest_sum_of_consecutive_odd_integers_lt_500 : 
  ∃ n : ℤ, consecutive_odd_integers n ∧ prod_less_500 n ∧ ∀ m : ℤ, consecutive_odd_integers m ∧ prod_less_500 m → n + (n + 2) ≥ m + (m + 2) :=
sorry

end greatest_sum_of_consecutive_odd_integers_lt_500_l425_42589


namespace probability_of_even_product_l425_42592

-- Each die has faces numbered from 1 to 8.
def faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Calculate the number of outcomes where the product of two rolls is even.
def num_even_product_outcomes : ℕ := (64 - 16)

-- Calculate the total number of outcomes when two eight-sided dice are rolled.
def total_outcomes : ℕ := 64

-- The probability that the product is even.
def probability_even_product : ℚ := num_even_product_outcomes / total_outcomes

theorem probability_of_even_product :
  probability_even_product = 3 / 4 :=
  by
    sorry

end probability_of_even_product_l425_42592


namespace units_digit_7_pow_l425_42595

theorem units_digit_7_pow (n : ℕ) : 
  ∃ k, 7^n % 10 = k ∧ ((7^1 % 10 = 7) ∧ (7^2 % 10 = 9) ∧ (7^3 % 10 = 3) ∧ (7^4 % 10 = 1) ∧ (7^5 % 10 = 7)) → 
  7^2010 % 10 = 9 :=
by
  sorry

end units_digit_7_pow_l425_42595


namespace fifth_term_is_19_l425_42568

-- Define the first term and the common difference
def a₁ : Int := 3
def d : Int := 4

-- Define the formula for the nth term in the arithmetic sequence
def arithmetic_sequence (n : Int) : Int :=
  a₁ + (n - 1) * d

-- Define the Lean 4 statement proving that the 5th term is 19
theorem fifth_term_is_19 : arithmetic_sequence 5 = 19 :=
by
  sorry -- Proof to be filled in

end fifth_term_is_19_l425_42568


namespace inequality_solution_l425_42533

def solution_set_of_inequality (x : ℝ) : Prop :=
  x * (x - 1) < 0

theorem inequality_solution :
  { x : ℝ | solution_set_of_inequality x } = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end inequality_solution_l425_42533


namespace find_x_l425_42514

def sum_sequence (a b : ℕ) : ℕ :=
  (b * (2 * a + b - 1)) / 2  -- Sum of an arithmetic progression

theorem find_x (x : ℕ) (h1 : sum_sequence x 10 = 65) : x = 2 :=
by {
  -- the proof goes here
  sorry
}

end find_x_l425_42514


namespace age_ratio_five_years_later_l425_42542

theorem age_ratio_five_years_later (my_age : ℕ) (son_age : ℕ) (h1 : my_age = 45) (h2 : son_age = 15) :
  (my_age + 5) / gcd (my_age + 5) (son_age + 5) = 5 ∧ (son_age + 5) / gcd (my_age + 5) (son_age + 5) = 2 :=
by
  sorry

end age_ratio_five_years_later_l425_42542


namespace student_l425_42515

noncomputable def allowance_after_video_games (A : ℝ) : ℝ := (3 / 7) * A

noncomputable def allowance_after_comic_books (remaining_after_video_games : ℝ) : ℝ := (3 / 5) * remaining_after_video_games

noncomputable def allowance_after_trading_cards (remaining_after_comic_books : ℝ) : ℝ := (5 / 8) * remaining_after_comic_books

noncomputable def last_allowance (remaining_after_trading_cards : ℝ) : ℝ := remaining_after_trading_cards

theorem student's_monthly_allowance (A : ℝ) (h1 : last_allowance (allowance_after_trading_cards (allowance_after_comic_books (allowance_after_video_games A))) = 1.20) :
  A = 7.47 := 
sorry

end student_l425_42515


namespace four_digit_perfect_square_l425_42531

theorem four_digit_perfect_square (N : ℕ) (a b : ℤ) :
  N = 1100 * a + 11 * b ∧
  N >= 1000 ∧ N <= 9999 ∧
  a >= 0 ∧ a <= 9 ∧ b >= 0 ∧ b <= 9 ∧
  (∃ (x : ℤ), N = 11 * x^2) →
  N = 7744 := by
  sorry

end four_digit_perfect_square_l425_42531


namespace range_of_a_if_solution_non_empty_l425_42579

variable (f : ℝ → ℝ) (a : ℝ)

/-- Given that the solution set of f(x) < | -1 | is non-empty,
    we need to prove that |a| ≥ 4. -/
theorem range_of_a_if_solution_non_empty (h : ∃ x, f x < 1) : |a| ≥ 4 :=
sorry

end range_of_a_if_solution_non_empty_l425_42579


namespace modulus_z_eq_one_l425_42547

noncomputable def imaginary_unit : ℂ := Complex.I

noncomputable def z : ℂ := (1 - imaginary_unit) / (1 + imaginary_unit) 

theorem modulus_z_eq_one : Complex.abs z = 1 := 
sorry

end modulus_z_eq_one_l425_42547


namespace domain_of_function_correct_l425_42572

noncomputable def domain_of_function (x : ℝ) : Prop :=
  (x + 1 ≥ 0) ∧ (2 - x > 0) ∧ (Real.logb 10 (2 - x) ≠ 0)

theorem domain_of_function_correct :
  {x : ℝ | domain_of_function x} = {x : ℝ | x ∈ Set.Icc (-1 : ℝ) 1 \ {1}} ∪ {x : ℝ | x ∈ Set.Ioc 1 2} :=
by
  sorry

end domain_of_function_correct_l425_42572


namespace num_students_is_92_l425_42551

noncomputable def total_students (S : ℕ) : Prop :=
  let remaining := S - 20
  let biking := (5/8 : ℚ) * remaining
  let walking := (3/8 : ℚ) * remaining
  walking = 27

theorem num_students_is_92 : total_students 92 :=
by
  let remaining := 92 - 20
  let biking := (5/8 : ℚ) * remaining
  let walking := (3/8 : ℚ) * remaining
  have walk_eq : walking = 27 := by sorry
  exact walk_eq

end num_students_is_92_l425_42551


namespace y_in_terms_of_x_l425_42588

theorem y_in_terms_of_x (p x y : ℝ) (hx : x = 1 + 2^p) (hy : y = 1 + 2^(-p)) : y = x / (x - 1) := 
by 
  sorry

end y_in_terms_of_x_l425_42588


namespace geom_seq_sum_five_terms_l425_42573

theorem geom_seq_sum_five_terms (a : ℕ → ℝ) (q : ℝ) 
    (h_pos : ∀ n, 0 < a n)
    (h_a2 : a 2 = 8) 
    (h_arith : 2 * a 4 - a 3 = a 3 - 4 * a 5) :
    a 1 * (1 - q^5) / (1 - q) = 31 :=
by
    sorry

end geom_seq_sum_five_terms_l425_42573


namespace T7_value_l425_42520

-- Define the geometric sequence a_n
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q > 0, ∀ n, a (n + 1) = a n * q

-- Define the even function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (2 * a + 1) * x + 2 * a

-- The main theorem statement
theorem T7_value (a : ℕ → ℝ) (a2 a6 : ℝ) (a_val : ℝ) (q : ℝ) (T7 : ℝ) 
  (h1 : is_geometric_sequence a) 
  (h2 : a 2 = a2)
  (h3 : a 6 = a6)
  (h4 : a2 - 2 = f a_val 0)
  (h5 : a6 - 3 = f a_val 0)
  (h6 : q > 1)
  (h7 : T7 = a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7) : 
  T7 = 128 :=
sorry

end T7_value_l425_42520


namespace decreasing_function_range_l425_42521

theorem decreasing_function_range (k : ℝ) : (∀ x : ℝ, k + 2 < 0) ↔ k < -2 :=
by
  sorry

end decreasing_function_range_l425_42521


namespace total_flowers_correct_l425_42513

def rosa_original_flowers : ℝ := 67.5
def andre_gifted_flowers : ℝ := 90.75
def total_flowers (rosa : ℝ) (andre : ℝ) : ℝ := rosa + andre

theorem total_flowers_correct : total_flowers rosa_original_flowers andre_gifted_flowers = 158.25 :=
by 
  rw [total_flowers]
  sorry

end total_flowers_correct_l425_42513


namespace intersection_result_l425_42529

noncomputable def A : Set ℝ := { x | x^2 - 5*x - 6 < 0 }
noncomputable def B : Set ℝ := { x | 2022^x > Real.sqrt 2022 }
noncomputable def intersection : Set ℝ := { x | A x ∧ B x }

theorem intersection_result : intersection = Set.Ioo (1/2 : ℝ) 6 := by
  sorry

end intersection_result_l425_42529


namespace cone_volume_l425_42562

theorem cone_volume (S r : ℝ) : 
  ∃ V : ℝ, V = (1 / 3) * S * r :=
by
  sorry

end cone_volume_l425_42562


namespace initial_number_of_girls_l425_42570

theorem initial_number_of_girls (b g : ℕ) (h1 : b = 3 * (g - 20)) (h2 : 7 * (b - 54) = g - 20) : g = 39 :=
sorry

end initial_number_of_girls_l425_42570


namespace tens_digit_N_to_20_l425_42594

theorem tens_digit_N_to_20 (N : ℕ) (h1 : Even N) (h2 : ¬(∃ k : ℕ, N = 10 * k)) : 
  ((N ^ 20) / 10) % 10 = 7 := 
by 
  sorry

end tens_digit_N_to_20_l425_42594


namespace number_of_regular_pencils_l425_42566

def cost_eraser : ℝ := 0.8
def cost_regular : ℝ := 0.5
def cost_short : ℝ := 0.4
def num_eraser : ℕ := 200
def num_short : ℕ := 35
def total_revenue : ℝ := 194

theorem number_of_regular_pencils (num_regular : ℕ) :
  (num_eraser * cost_eraser) + (num_short * cost_short) + (num_regular * cost_regular) = total_revenue → 
  num_regular = 40 :=
by
  sorry

end number_of_regular_pencils_l425_42566


namespace discount_percent_l425_42544

theorem discount_percent
  (MP CP SP : ℝ)
  (h1 : CP = 0.55 * MP)
  (gainPercent : ℝ)
  (h2 : gainPercent = 54.54545454545454 / 100)
  (h3 : (SP - CP) / CP = gainPercent)
  : ((MP - SP) / MP) * 100 = 15 := by
  sorry

end discount_percent_l425_42544


namespace smallest_integer_to_make_perfect_square_l425_42599

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_integer_to_make_perfect_square :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, (n * y) = k^2) ∧ n = 6 :=
by
  sorry

end smallest_integer_to_make_perfect_square_l425_42599


namespace average_temperature_correct_l425_42524

-- Definition of the daily temperatures
def daily_temperatures : List ℕ := [51, 64, 61, 59, 48, 63, 55]

-- Define the number of days
def number_of_days : ℕ := 7

-- Prove the average temperature calculation
theorem average_temperature_correct :
  ((List.sum daily_temperatures : ℚ) / number_of_days : ℚ) = 57.3 :=
by
  sorry

end average_temperature_correct_l425_42524


namespace find_some_number_eq_0_3_l425_42552

theorem find_some_number_eq_0_3 (X : ℝ) (h : 2 * ((3.6 * 0.48 * 2.50) / (X * 0.09 * 0.5)) = 1600.0000000000002) :
  X = 0.3 :=
by sorry

end find_some_number_eq_0_3_l425_42552


namespace greatest_value_of_squares_l425_42557

theorem greatest_value_of_squares (a b c d : ℝ)
  (h1 : a + b = 18)
  (h2 : ab + c + d = 85)
  (h3 : ad + bc = 170)
  (h4 : cd = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 308 :=
sorry

end greatest_value_of_squares_l425_42557


namespace max_t_squared_value_l425_42559

noncomputable def max_t_squared (R : ℝ) : ℝ :=
  let PR_QR_sq_sum := 4 * R^2
  let max_PR_QR_prod := 2 * R^2
  PR_QR_sq_sum + 2 * max_PR_QR_prod

theorem max_t_squared_value (R : ℝ) : max_t_squared R = 8 * R^2 :=
  sorry

end max_t_squared_value_l425_42559


namespace alcohol_added_l425_42556

theorem alcohol_added (x : ℝ) :
  let initial_solution_volume := 40
  let initial_alcohol_percentage := 0.05
  let initial_alcohol_volume := initial_solution_volume * initial_alcohol_percentage
  let additional_water := 6.5
  let final_solution_volume := initial_solution_volume + x + additional_water
  let final_alcohol_percentage := 0.11
  let final_alcohol_volume := final_solution_volume * final_alcohol_percentage
  initial_alcohol_volume + x = final_alcohol_volume → x = 3.5 :=
by
  intros
  sorry

end alcohol_added_l425_42556


namespace sum_of_solutions_eq_320_l425_42550

theorem sum_of_solutions_eq_320 :
  ∃ (S : Finset ℝ), 
  (∀ x ∈ S, 0 < x ∧ x < 180 ∧ (1 + (Real.sin x / Real.sin (4 * x)) = (Real.sin (3 * x) / Real.sin (2 * x)))) 
  ∧ S.sum id = 320 :=
by {
  sorry
}

end sum_of_solutions_eq_320_l425_42550


namespace simplify_trig_expression_l425_42582

theorem simplify_trig_expression :
  (2 - Real.sin 21 * Real.sin 21 - Real.cos 21 * Real.cos 21 + 
  (Real.sin 17 * Real.sin 17) * (Real.sin 17 * Real.sin 17) + 
  (Real.sin 17 * Real.sin 17) * (Real.cos 17 * Real.cos 17) + 
  (Real.cos 17 * Real.cos 17)) = 2 :=
by
  sorry

end simplify_trig_expression_l425_42582


namespace circle_k_range_l425_42578

def circle_equation (k x y : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0

theorem circle_k_range (k : ℝ) (h : ∃ x y, circle_equation k x y) : k > 4 ∨ k < -1 :=
by
  sorry

end circle_k_range_l425_42578


namespace triangle_area_l425_42567

theorem triangle_area (a b c : ℝ) (h1 : a / b = 3 / 4) (h2 : b / c = 4 / 5) (h3 : a + b + c = 60) : 
  (1/2) * a * b = 150 :=
by
  sorry

end triangle_area_l425_42567


namespace steven_seeds_l425_42539

def average_seeds (fruit: String) : Nat :=
  match fruit with
  | "apple" => 6
  | "pear" => 2
  | "grape" => 3
  | "orange" => 10
  | "watermelon" => 300
  | _ => 0

def fruits := [("apple", 2), ("pear", 3), ("grape", 5), ("orange", 1), ("watermelon", 2)]

def required_seeds := 420

def total_seeds (fruit_list : List (String × Nat)) : Nat :=
  fruit_list.foldr (fun (fruit_qty : String × Nat) acc =>
    acc + (average_seeds fruit_qty.fst) * fruit_qty.snd) 0

theorem steven_seeds : total_seeds fruits - required_seeds = 223 := by
  sorry

end steven_seeds_l425_42539


namespace min_value_expression_l425_42548

noncomputable def min_expression (a b c : ℝ) : ℝ :=
  (9 / a) + (16 / b) + (25 / c)

theorem min_value_expression :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + b + c = 6 →
  min_expression a b c ≥ 18 :=
by
  intro a b c ha hb hc habc
  sorry

end min_value_expression_l425_42548


namespace ln_of_gt_of_pos_l425_42596

variable {a b : ℝ}

theorem ln_of_gt_of_pos (h1 : a > b) (h2 : b > 0) : Real.log a > Real.log b :=
sorry

end ln_of_gt_of_pos_l425_42596


namespace weightlifter_total_weight_l425_42534

theorem weightlifter_total_weight (weight_one_hand : ℕ) (num_hands : ℕ) (condition: weight_one_hand = 8 ∧ num_hands = 2) :
  2 * weight_one_hand = 16 :=
by
  sorry

end weightlifter_total_weight_l425_42534


namespace small_pizza_slices_correct_l425_42519

-- Defining the total number of people involved
def people_count : ℕ := 3

-- Defining the number of slices each person can eat
def slices_per_person : ℕ := 12

-- Calculating the total number of slices needed based on the number of people and slices per person
def total_slices_needed : ℕ := people_count * slices_per_person

-- Defining the number of slices in a large pizza
def large_pizza_slices : ℕ := 14

-- Defining the number of large pizzas ordered
def large_pizzas_count : ℕ := 2

-- Calculating the total number of slices provided by the large pizzas
def total_large_pizza_slices : ℕ := large_pizza_slices * large_pizzas_count

-- Defining the number of slices in a small pizza
def small_pizza_slices : ℕ := 8

-- Total number of slices provided needs to be at least the total slices needed
theorem small_pizza_slices_correct :
  total_slices_needed ≤ total_large_pizza_slices + small_pizza_slices := by
  sorry

end small_pizza_slices_correct_l425_42519


namespace find_root_floor_l425_42516

noncomputable def g (x : ℝ) := Real.sin x - Real.cos x + 4 * Real.tan x

theorem find_root_floor :
  ∃ s : ℝ, (g s = 0) ∧ (π / 2 < s) ∧ (s < 3 * π / 2) ∧ (Int.floor s = 3) :=
  sorry

end find_root_floor_l425_42516


namespace balls_in_each_package_l425_42526

theorem balls_in_each_package (x : ℕ) (h : 21 * x = 399) : x = 19 :=
by
  sorry

end balls_in_each_package_l425_42526


namespace scientific_notation_of_0point0000025_l425_42586

theorem scientific_notation_of_0point0000025 : ∃ (a : ℝ) (n : ℤ), 0.0000025 = a * 10 ^ n ∧ a = 2.5 ∧ n = -6 :=
by {
  sorry
}

end scientific_notation_of_0point0000025_l425_42586


namespace compound_interest_rate_l425_42545

theorem compound_interest_rate : 
  let P := 14800
  let interest := 4265.73
  let A := 19065.73
  let t := 2
  let n := 1
  let r := 0.13514
  (P : ℝ) * (1 + r)^t = A :=
by
-- Here we will provide the steps of the proof
sorry

end compound_interest_rate_l425_42545


namespace range_of_a_l425_42577

noncomputable def f (a : ℝ) (x : ℝ) := Real.sqrt (Real.exp x + (Real.exp 1 - 1) * x - a)
def exists_b_condition (a : ℝ) : Prop := ∃ b : ℝ, b ∈ Set.Icc 0 1 ∧ f a b = b

theorem range_of_a (a : ℝ) : exists_b_condition a → a ∈ Set.Icc 1 (2 * Real.exp 1 - 2) :=
sorry

end range_of_a_l425_42577


namespace no_polyhedron_with_surface_2015_l425_42537

/--
It is impossible to glue together 1 × 1 × 1 cubes to form a polyhedron whose surface area is 2015.
-/
theorem no_polyhedron_with_surface_2015 (n k : ℕ) : 6 * n - 2 * k ≠ 2015 :=
by
  sorry

end no_polyhedron_with_surface_2015_l425_42537


namespace sum_of_solutions_l425_42502

-- Define the system of equations as lean functions
def equation1 (x y : ℝ) : Prop := |x - 4| = |y - 10|
def equation2 (x y : ℝ) : Prop := |x - 10| = 3 * |y - 4|

-- Statement of the theorem
theorem sum_of_solutions : 
  ∃ (solutions : List (ℝ × ℝ)), 
    (∀ (sol : ℝ × ℝ), sol ∈ solutions → equation1 sol.1 sol.2 ∧ equation2 sol.1 sol.2) ∧ 
    (List.sum (solutions.map (fun sol => sol.1 + sol.2)) = 24) :=
  sorry

end sum_of_solutions_l425_42502


namespace milan_long_distance_bill_l425_42555

theorem milan_long_distance_bill
  (monthly_fee : ℝ := 2)
  (per_minute_cost : ℝ := 0.12)
  (minutes_used : ℕ := 178) :
  ((minutes_used : ℝ) * per_minute_cost + monthly_fee = 23.36) :=
by
  sorry

end milan_long_distance_bill_l425_42555


namespace stratified_sampling_l425_42510

variable (H M L total_sample : ℕ)
variable (H_fams M_fams L_fams : ℕ)

-- Conditions
def community : Prop := H_fams = 150 ∧ M_fams = 360 ∧ L_fams = 90
def total_population : Prop := H_fams + M_fams + L_fams = 600
def sample_size : Prop := total_sample = 100

-- Statement
theorem stratified_sampling (H_fams M_fams L_fams : ℕ) (total_sample : ℕ)
  (h_com : community H_fams M_fams L_fams)
  (h_total_pop : total_population H_fams M_fams L_fams)
  (h_sample_size : sample_size total_sample)
  : H = 25 ∧ M = 60 ∧ L = 15 :=
by
  sorry

end stratified_sampling_l425_42510


namespace find_weight_of_second_square_l425_42500

-- Define given conditions
def side_length1 : ℝ := 4
def weight1 : ℝ := 16
def side_length2 : ℝ := 6

-- Define the uniform density and thickness condition
def uniform_density (a₁ a₂ : ℝ) (w₁ w₂ : ℝ) : Prop :=
  (a₁ * w₂ = a₂ * w₁)

-- Problem statement:
theorem find_weight_of_second_square : 
  uniform_density (side_length1 ^ 2) (side_length2 ^ 2) weight1 w₂ → 
  w₂ = 36 :=
by
  sorry

end find_weight_of_second_square_l425_42500


namespace expression_equals_eight_l425_42505

theorem expression_equals_eight
  (a b c : ℝ)
  (h1 : a + b = 2 * c)
  (h2 : b + c = 2 * a)
  (h3 : a + c = 2 * b) :
  (a + b) * (b + c) * (a + c) / (a * b * c) = 8 := by
  sorry

end expression_equals_eight_l425_42505


namespace certain_number_example_l425_42518

theorem certain_number_example (x : ℝ) 
    (h1 : 213 * 16 = 3408)
    (h2 : 0.16 * x = 0.3408) : 
    x = 2.13 := 
by 
  sorry

end certain_number_example_l425_42518


namespace problem_part_a_problem_part_b_problem_part_e_problem_part_c_disproof_problem_part_d_disproof_l425_42593

-- Define Lean goals for the true statements
theorem problem_part_a (x : ℝ) (h : x < 0) : x^3 < x := sorry
theorem problem_part_b (x : ℝ) (h : x^3 > 0) : x > 0 := sorry
theorem problem_part_e (x : ℝ) (h : x > 1) : x^3 > x := sorry

-- Disprove the false statements by showing the negation
theorem problem_part_c_disproof (x : ℝ) (h : x^3 < x) : ¬ (|x| > 1) := sorry
theorem problem_part_d_disproof (x : ℝ) (h : x^3 > x) : ¬ (x > 1) := sorry

end problem_part_a_problem_part_b_problem_part_e_problem_part_c_disproof_problem_part_d_disproof_l425_42593


namespace leila_yards_l425_42522

variable (mile_yards : ℕ := 1760)
variable (marathon_miles : ℕ := 28)
variable (marathon_yards : ℕ := 1500)
variable (marathons_ran : ℕ := 15)

theorem leila_yards (m y : ℕ) (h1 : marathon_miles = 28) (h2 : marathon_yards = 1500) (h3 : mile_yards = 1760) (h4 : marathons_ran = 15) (hy : 0 ≤ y ∧ y < mile_yards) :
  y = 1200 :=
sorry

end leila_yards_l425_42522


namespace remainder_when_divided_by_9_l425_42540

noncomputable def base12_to_dec (x : ℕ) : ℕ :=
  (1 * 12^3) + (5 * 12^2) + (3 * 12) + 4
  
theorem remainder_when_divided_by_9 : base12_to_dec (1534) % 9 = 2 := by
  sorry

end remainder_when_divided_by_9_l425_42540


namespace find_annual_interest_rate_l425_42508

noncomputable def annual_interest_rate (P A n t : ℝ) : ℝ :=
  2 * ((A / P)^(1 / (n * t)) - 1)

theorem find_annual_interest_rate :
  Π (P A : ℝ) (n t : ℕ), P = 600 → A = 760 → n = 2 → t = 4 →
  annual_interest_rate P A n t = 0.06020727 :=
by
  intros P A n t hP hA hn ht
  rw [hP, hA, hn, ht]
  unfold annual_interest_rate
  sorry

end find_annual_interest_rate_l425_42508


namespace rem_neg_one_third_quarter_l425_42560

noncomputable def rem (x y : ℝ) : ℝ :=
  x - y * ⌊x / y⌋

theorem rem_neg_one_third_quarter :
  rem (-1/3) (1/4) = 1/6 :=
by
  sorry

end rem_neg_one_third_quarter_l425_42560


namespace regular_octagon_interior_angle_l425_42563

theorem regular_octagon_interior_angle : 
  (∀ (n : ℕ), n = 8 → ∀ (sum_of_interior_angles : ℕ), sum_of_interior_angles = (n - 2) * 180 → ∀ (each_angle : ℕ), each_angle = sum_of_interior_angles / n → each_angle = 135) :=
  sorry

end regular_octagon_interior_angle_l425_42563


namespace tomatoes_eaten_l425_42538

theorem tomatoes_eaten (initial_tomatoes : ℕ) (remaining_tomatoes : ℕ) (portion_eaten : ℚ)
  (h_init : initial_tomatoes = 21)
  (h_rem : remaining_tomatoes = 14)
  (h_portion : portion_eaten = 1/3) :
  initial_tomatoes - remaining_tomatoes = (portion_eaten * initial_tomatoes) :=
by
  sorry

end tomatoes_eaten_l425_42538


namespace find_n_l425_42503

noncomputable def f (x : ℤ) : ℤ := sorry -- f is some polynomial with integer coefficients

theorem find_n (n : ℤ) (h1 : f 1 = -1) (h4 : f 4 = 2) (h8 : f 8 = 34) (hn : f n = n^2 - 4 * n - 18) : n = 3 ∨ n = 6 :=
sorry

end find_n_l425_42503


namespace simplify_expression_l425_42587

theorem simplify_expression (x : ℝ) : (3 * x)^5 + (5 * x) * (x^4) - 7 * x^5 = 241 * x^5 := 
by
  sorry

end simplify_expression_l425_42587


namespace evaluate_expression_l425_42511

-- Definition of the conditions
def a : ℕ := 15
def b : ℕ := 19
def c : ℕ := 13

-- Problem statement
theorem evaluate_expression :
  (225 * (1 / a - 1 / b) + 361 * (1 / b - 1 / c) + 169 * (1 / c - 1 / a))
  /
  (a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)) = a + b + c :=
by
  sorry

end evaluate_expression_l425_42511


namespace negation_of_forall_ge_zero_l425_42527

theorem negation_of_forall_ge_zero :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x : ℝ, x^2 < 0) :=
sorry

end negation_of_forall_ge_zero_l425_42527


namespace zoe_earnings_from_zachary_l425_42571

noncomputable def babysitting_earnings 
  (total_earnings : ℕ) (pool_cleaning_earnings : ℕ) (earnings_julie_ratio : ℕ) 
  (earnings_chloe_ratio : ℕ) 
  (earnings_zachary : ℕ) : Prop := 
total_earnings = 8000 ∧ 
pool_cleaning_earnings = 2600 ∧ 
earnings_julie_ratio = 3 ∧ 
earnings_chloe_ratio = 5 ∧ 
9 * earnings_zachary = 5400

theorem zoe_earnings_from_zachary : babysitting_earnings 8000 2600 3 5 600 :=
by 
  unfold babysitting_earnings
  sorry

end zoe_earnings_from_zachary_l425_42571


namespace smallest_unpayable_amount_l425_42565

theorem smallest_unpayable_amount :
  ∀ (coins_1p coins_2p coins_3p coins_4p coins_5p : ℕ), 
    coins_1p = 1 → 
    coins_2p = 2 → 
    coins_3p = 3 → 
    coins_4p = 4 → 
    coins_5p = 5 → 
    ∃ (x : ℕ), x = 56 ∧ 
    ¬ (∃ (a b c d e : ℕ), a * 1 + b * 2 + c * 3 + d * 4 + e * 5 = x ∧ 
    a ≤ coins_1p ∧
    b ≤ coins_2p ∧
    c ≤ coins_3p ∧
    d ≤ coins_4p ∧
    e ≤ coins_5p) :=
by {
  -- Here we skip the actual proof
  sorry
}

end smallest_unpayable_amount_l425_42565


namespace minimum_perimeter_l425_42597

-- Define the area condition
def area_condition (l w : ℝ) : Prop := l * w = 64

-- Define the perimeter function
def perimeter (l w : ℝ) : ℝ := 2 * l + 2 * w

-- The theorem statement based on the conditions and the correct answer
theorem minimum_perimeter (l w : ℝ) (h : area_condition l w) : 
  perimeter l w ≥ 32 := by
sorry

end minimum_perimeter_l425_42597


namespace probability_of_satisfaction_l425_42528

-- Definitions for the conditions given in the problem
def dissatisfied_customers_leave_negative_review_probability : ℝ := 0.8
def satisfied_customers_leave_positive_review_probability : ℝ := 0.15
def negative_reviews : ℕ := 60
def positive_reviews : ℕ := 20
def expected_satisfaction_probability : ℝ := 0.64

-- The problem to prove
theorem probability_of_satisfaction :
  ∃ p : ℝ, (dissatisfied_customers_leave_negative_review_probability * (1 - p) = negative_reviews / (negative_reviews + positive_reviews)) ∧
           (satisfied_customers_leave_positive_review_probability * p = positive_reviews / (negative_reviews + positive_reviews)) ∧
           p = expected_satisfaction_probability := 
by
  sorry

end probability_of_satisfaction_l425_42528


namespace min_x2_y2_of_product_eq_zero_l425_42543

theorem min_x2_y2_of_product_eq_zero (x y : ℝ) (h : (x + 8) * (y - 8) = 0) : x^2 + y^2 = 64 :=
sorry

end min_x2_y2_of_product_eq_zero_l425_42543


namespace find_f_2010_l425_42525

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom functional_eq : ∀ x : ℝ, f (x + 6) = f x + f (3 - x)

theorem find_f_2010 : f 2010 = 0 := sorry

end find_f_2010_l425_42525


namespace glass_heavier_than_plastic_l425_42574

-- Define the conditions
def condition1 (G : ℕ) : Prop := 3 * G = 600
def condition2 (G P : ℕ) : Prop := 4 * G + 5 * P = 1050

-- Define the theorem to prove
theorem glass_heavier_than_plastic (G P : ℕ) (h1 : condition1 G) (h2 : condition2 G P) : G - P = 150 :=
by
  sorry

end glass_heavier_than_plastic_l425_42574


namespace one_third_of_five_times_seven_l425_42598

theorem one_third_of_five_times_seven:
  (1/3 : ℝ) * (5 * 7) = 35 / 3 := 
by
  -- Definitions and calculations go here
  sorry

end one_third_of_five_times_seven_l425_42598


namespace class_total_students_l425_42523

theorem class_total_students (x y : ℕ)
  (initial_absent : y = (1/6) * x)
  (after_sending_chalk : y = (1/5) * (x - 1)) :
  x + y = 7 :=
by
  sorry

end class_total_students_l425_42523


namespace max_value_inequality_l425_42536

theorem max_value_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 3) :
  (x^2 + x * y + y^2) * (y^2 + y * z + z^2) * (z^2 + z * x + x^2) ≤ 27 := 
sorry

end max_value_inequality_l425_42536


namespace prop_false_iff_a_lt_neg_13_over_2_l425_42541

theorem prop_false_iff_a_lt_neg_13_over_2 :
  (¬ ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 + a * x + 9 ≥ 0) ↔ a < -13 / 2 := 
sorry

end prop_false_iff_a_lt_neg_13_over_2_l425_42541


namespace min_sugar_l425_42553

theorem min_sugar (f s : ℝ) (h₁ : f ≥ 8 + (3/4) * s) (h₂ : f ≤ 2 * s) : s ≥ 32 / 5 :=
sorry

end min_sugar_l425_42553


namespace solve_for_a_l425_42532

theorem solve_for_a (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
(h_eq_exponents : a ^ b = b ^ a) (h_b_equals_3a : b = 3 * a) : a = Real.sqrt 3 :=
sorry

end solve_for_a_l425_42532


namespace largest_proper_fraction_smallest_improper_fraction_smallest_mixed_number_l425_42554

-- Definitions based on conditions
def isProperFraction (n d : ℕ) : Prop := n < d
def isImproperFraction (n d : ℕ) : Prop := n ≥ d
def isMixedNumber (w n d : ℕ) : Prop := w > 0 ∧ isProperFraction n d

-- Fractional part is 1/9, meaning all fractions considered have part = 1/9
def fractionalPart := 1 / 9

-- Lean 4 statements to verify the correct answers
theorem largest_proper_fraction : ∃ n d : ℕ, fractionalPart = n / d ∧ isProperFraction n d ∧ (n, d) = (8, 9) := sorry

theorem smallest_improper_fraction : ∃ n d : ℕ, fractionalPart = n / d ∧ isImproperFraction n d ∧ (n, d) = (9, 9) := sorry

theorem smallest_mixed_number : ∃ w n d : ℕ, fractionalPart = n / d ∧ isMixedNumber w n d ∧ ((w, n, d) = (1, 1, 9) ∨ (w, n, d) = (10, 9)) := sorry

end largest_proper_fraction_smallest_improper_fraction_smallest_mixed_number_l425_42554


namespace jill_peaches_l425_42581

open Nat

theorem jill_peaches (Jake Steven Jill : ℕ)
  (h1 : Jake = Steven - 6)
  (h2 : Steven = Jill + 18)
  (h3 : Jake = 17) :
  Jill = 5 := 
by
  sorry

end jill_peaches_l425_42581


namespace arithmetic_sequence_S10_l425_42509

-- Definition of an arithmetic sequence and the corresponding sums S_n.
def is_arithmetic_sequence (S : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, S (n + 1) = S n + d

theorem arithmetic_sequence_S10 
  (S : ℕ → ℕ)
  (h1 : S 1 = 10)
  (h2 : S 2 = 20)
  (h_arith : is_arithmetic_sequence S) :
  S 10 = 100 :=
sorry

end arithmetic_sequence_S10_l425_42509


namespace minimize_distance_postman_l425_42530

-- Let x be a function that maps house indices to coordinates.
def optimalPostOfficeLocation (n: ℕ) (x : ℕ → ℝ) : ℝ :=
  if n % 2 = 1 then 
    x (n / 2 + 1)
  else 
    x (n / 2)

theorem minimize_distance_postman (n: ℕ) (x : ℕ → ℝ)
  (h_sorted : ∀ i j, i < j → x i < x j) :
  optimalPostOfficeLocation n x = if n % 2 = 1 then 
    x (n / 2 + 1)
  else 
    x (n / 2) := 
  sorry

end minimize_distance_postman_l425_42530


namespace milk_price_per_liter_l425_42584

theorem milk_price_per_liter (M : ℝ) 
  (price_fruit_per_kg : ℝ) (price_each_fruit_kg_eq_2: price_fruit_per_kg = 2)
  (milk_liters_per_batch : ℝ) (milk_liters_per_batch_eq_10: milk_liters_per_batch = 10)
  (fruit_kg_per_batch : ℝ) (fruit_kg_per_batch_eq_3 : fruit_kg_per_batch = 3)
  (cost_three_batches : ℝ) (cost_three_batches_eq_63: cost_three_batches = 63) :
  M = 1.5 :=
by
  sorry

end milk_price_per_liter_l425_42584


namespace epicenter_distance_l425_42558

noncomputable def distance_from_epicenter (v1 v2 Δt: ℝ) : ℝ :=
  Δt / ((1 / v2) - (1 / v1))

theorem epicenter_distance : 
  distance_from_epicenter 5.94 3.87 11.5 = 128 := 
by
  -- The proof will use calculations shown in the solution.
  sorry

end epicenter_distance_l425_42558


namespace no_positive_integer_solutions_l425_42576

theorem no_positive_integer_solutions (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : x^4 * y^4 - 14 * x^2 * y^2 + 49 ≠ 0 := 
by sorry

end no_positive_integer_solutions_l425_42576


namespace ahmed_final_score_requirement_l425_42549

-- Define the given conditions
def total_assignments : ℕ := 9
def ahmed_initial_grade : ℕ := 91
def emily_initial_grade : ℕ := 92
def sarah_initial_grade : ℕ := 94
def final_assignment_weight := true -- Assuming each assignment has the same weight
def min_passing_score : ℕ := 70
def max_score : ℕ := 100
def emily_final_score : ℕ := 90

noncomputable def ahmed_min_final_score : ℕ := 98

-- The proof statement
theorem ahmed_final_score_requirement :
  let ahmed_initial_points := ahmed_initial_grade * total_assignments
  let emily_initial_points := emily_initial_grade * total_assignments
  let sarah_initial_points := sarah_initial_grade * total_assignments
  let emily_final_total := emily_initial_points + emily_final_score
  let sarah_final_total := sarah_initial_points + min_passing_score
  let ahmed_final_total_needed := sarah_final_total + 1
  let ahmed_needed_score := ahmed_final_total_needed - ahmed_initial_points
  ahmed_needed_score = ahmed_min_final_score :=
by
  sorry

end ahmed_final_score_requirement_l425_42549


namespace Toms_dog_age_in_6_years_l425_42506

-- Let's define the conditions
variables (B D : ℕ)
axiom h1 : B = 4 * D
axiom h2 : B + 6 = 30

-- Now we state the theorem
theorem Toms_dog_age_in_6_years :
  D + 6 = 12 :=
by
  sorry

end Toms_dog_age_in_6_years_l425_42506


namespace A_3_2_eq_29_l425_42590

-- Define the recursive function A(m, n).
def A : Nat → Nat → Nat
| 0, n => n + 1
| (m + 1), 0 => A m 1
| (m + 1), (n + 1) => A m (A (m + 1) n)

-- Prove that A(3, 2) = 29
theorem A_3_2_eq_29 : A 3 2 = 29 := by 
  sorry

end A_3_2_eq_29_l425_42590


namespace handshakes_total_l425_42517

def num_couples : ℕ := 15
def total_people : ℕ := 30
def men : ℕ := 15
def women : ℕ := 15
def youngest_man_handshakes : ℕ := 0
def men_handshakes : ℕ := (14 * 13) / 2
def men_women_handshakes : ℕ := 15 * 14

theorem handshakes_total : men_handshakes + men_women_handshakes = 301 :=
by
  -- Proof goes here
  sorry

end handshakes_total_l425_42517


namespace find_a9_l425_42504

variable {a : ℕ → ℝ} 
variable {q : ℝ}

-- Conditions
def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def a3_eq_1 (a : ℕ → ℝ) : Prop := 
  a 3 = 1

def a5_a6_a7_eq_8 (a : ℕ → ℝ) : Prop := 
  a 5 * a 6 * a 7 = 8

-- Theorem to prove
theorem find_a9 {a : ℕ → ℝ} {q : ℝ} 
  (geom : geom_seq a q)
  (ha3 : a3_eq_1 a)
  (ha5a6a7 : a5_a6_a7_eq_8 a) : a 9 = 4 := 
sorry

end find_a9_l425_42504


namespace parabola_latus_rectum_equation_l425_42591

theorem parabola_latus_rectum_equation :
  (∃ (y x : ℝ), y^2 = 4 * x) → (∀ x, x = -1) :=
by
  sorry

end parabola_latus_rectum_equation_l425_42591


namespace expression_is_perfect_cube_l425_42535

theorem expression_is_perfect_cube {x y z : ℝ} (h : x + y + z = 0) :
  ∃ m : ℝ, 
    (x^2 * y^2 + y^2 * z^2 + z^2 * x^2) * 
    (x^3 * y * z + x * y^3 * z + x * y * z^3) *
    (x^3 * y^2 * z + x^3 * y * z^2 + x^2 * y^3 * z + x * y^3 * z^2 + x^2 * y * z^3 + x * y^2 * z^3) =
    m ^ 3 := 
by 
  sorry

end expression_is_perfect_cube_l425_42535


namespace perp_bisector_chord_l425_42561

theorem perp_bisector_chord (x y : ℝ) :
  (2 * x + 3 * y + 1 = 0) ∧ (x^2 + y^2 - 2 * x + 4 * y = 0) → 
  ∃ k l m : ℝ, (3 * x - 2 * y - 7 = 0) :=
by
  sorry

end perp_bisector_chord_l425_42561
