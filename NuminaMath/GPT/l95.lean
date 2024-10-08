import Mathlib

namespace inappropriate_character_choice_l95_95953

-- Definitions and conditions
def is_main_character (c : String) : Prop := 
  c = "Gryphon" ∨ c = "Mock Turtle"

def characters : List String := ["Lobster", "Gryphon", "Mock Turtle"]

-- Theorem statement
theorem inappropriate_character_choice : 
  ¬ is_main_character "Lobster" :=
by 
  sorry

end inappropriate_character_choice_l95_95953


namespace overhead_percentage_l95_95024

def purchase_price : ℝ := 48
def markup : ℝ := 30
def net_profit : ℝ := 12

-- Define the theorem to be proved
theorem overhead_percentage : ((markup - net_profit) / purchase_price) * 100 = 37.5 := by
  sorry

end overhead_percentage_l95_95024


namespace x_squared_y_cubed_plus_y_squared_x_cubed_eq_zero_l95_95037

theorem x_squared_y_cubed_plus_y_squared_x_cubed_eq_zero
  (x y : ℝ)
  (h1 : x + y = 2)
  (h2 : x * y = 4) : x^2 * y^3 + y^2 * x^3 = 0 := 
sorry

end x_squared_y_cubed_plus_y_squared_x_cubed_eq_zero_l95_95037


namespace distinct_pen_distribution_l95_95730

theorem distinct_pen_distribution :
  ∃! (a b c d : ℕ), a + b + c + d = 10 ∧
                    1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ 1 ≤ d ∧
                    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d :=
sorry

end distinct_pen_distribution_l95_95730


namespace f_g_2_equals_169_l95_95083

-- Definitions of f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 2 * x^2 + x + 3

-- The theorem statement
theorem f_g_2_equals_169 : f (g 2) = 169 :=
by
  sorry

end f_g_2_equals_169_l95_95083


namespace trains_meet_at_9am_l95_95211

-- Definitions of conditions
def distance_AB : ℝ := 65
def speed_train_A : ℝ := 20
def speed_train_B : ℝ := 25
def start_time_train_A : ℝ := 7
def start_time_train_B : ℝ := 8

-- This function calculates the meeting time of the two trains
noncomputable def meeting_time (distance_AB : ℝ) (speed_train_A : ℝ) (speed_train_B : ℝ) 
    (start_time_train_A : ℝ) (start_time_train_B : ℝ) : ℝ :=
  let distance_train_A := speed_train_A * (start_time_train_B - start_time_train_A)
  let remaining_distance := distance_AB - distance_train_A
  let relative_speed := speed_train_A + speed_train_B
  start_time_train_B + remaining_distance / relative_speed

-- Theorem stating the time when the two trains meet
theorem trains_meet_at_9am :
    meeting_time distance_AB speed_train_A speed_train_B start_time_train_A start_time_train_B = 9 := sorry

end trains_meet_at_9am_l95_95211


namespace total_amount_paid_l95_95650

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

end total_amount_paid_l95_95650


namespace circle_equation_l95_95402

-- Definitions based on the conditions
def center_on_x_axis (a b r : ℝ) := b = 0
def tangent_at_point (a b r : ℝ) := (b - 1) / a = -1/2

-- Proof statement
theorem circle_equation (a b r : ℝ) (h1: center_on_x_axis a b r) (h2: tangent_at_point a b r) :
    ∃ (a b r : ℝ), (x - a)^2 + y^2 = r^2 ∧ a = 2 ∧ b = 0 ∧ r^2 = 5 :=
by 
  sorry

end circle_equation_l95_95402


namespace exp_sum_is_neg_one_l95_95748

noncomputable def sumExpExpressions : ℂ :=
  (Complex.exp (Real.pi * Complex.I / 7) +
   Complex.exp (2 * Real.pi * Complex.I / 7) +
   Complex.exp (3 * Real.pi * Complex.I / 7) +
   Complex.exp (4 * Real.pi * Complex.I / 7) +
   Complex.exp (5 * Real.pi * Complex.I / 7) +
   Complex.exp (6 * Real.pi * Complex.I / 7) +
   Complex.exp (2 * Real.pi * Complex.I / 9) +
   Complex.exp (4 * Real.pi * Complex.I / 9) +
   Complex.exp (6 * Real.pi * Complex.I / 9) +
   Complex.exp (8 * Real.pi * Complex.I / 9) +
   Complex.exp (10 * Real.pi * Complex.I / 9) +
   Complex.exp (12 * Real.pi * Complex.I / 9) +
   Complex.exp (14 * Real.pi * Complex.I / 9) +
   Complex.exp (16 * Real.pi * Complex.I / 9))

theorem exp_sum_is_neg_one : sumExpExpressions = -1 := by
  sorry

end exp_sum_is_neg_one_l95_95748


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l95_95767

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l95_95767


namespace tickets_needed_l95_95349

def tickets_per_roller_coaster : ℕ := 5
def tickets_per_giant_slide : ℕ := 3
def roller_coaster_rides : ℕ := 7
def giant_slide_rides : ℕ := 4

theorem tickets_needed : tickets_per_roller_coaster * roller_coaster_rides + tickets_per_giant_slide * giant_slide_rides = 47 := 
by
  sorry

end tickets_needed_l95_95349


namespace proof_2d_minus_r_l95_95419

theorem proof_2d_minus_r (d r: ℕ) (h1 : 1059 % d = r)
  (h2 : 1482 % d = r) (h3 : 2340 % d = r) (hd : d > 1) : 2 * d - r = 6 := 
by 
  sorry

end proof_2d_minus_r_l95_95419


namespace equivalence_gcd_prime_power_l95_95829

theorem equivalence_gcd_prime_power (a b n : ℕ) :
  (∀ m, 0 < m ∧ m < n → Nat.gcd n ((n - m) / Nat.gcd n m) = 1) ↔ 
  (∃ p k : ℕ, Nat.Prime p ∧ n = p ^ k) :=
by
  sorry

end equivalence_gcd_prime_power_l95_95829


namespace initial_oranges_l95_95531

theorem initial_oranges (O : ℕ) (h1 : (1 / 4 : ℚ) * (1 / 2 : ℚ) * O = 39) (h2 : (1 / 8 : ℚ) * (1 / 2 : ℚ) * O = 4 + 78 - (1 / 4 : ℚ) * (1 / 2 : ℚ) * O) :
  O = 96 :=
by
  sorry

end initial_oranges_l95_95531


namespace absolute_value_sum_l95_95217

theorem absolute_value_sum (a b : ℤ) (h_a : |a| = 5) (h_b : |b| = 3) : 
  (a + b = 8) ∨ (a + b = 2) ∨ (a + b = -2) ∨ (a + b = -8) :=
by
  sorry

end absolute_value_sum_l95_95217


namespace cube_inequality_l95_95080

theorem cube_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^3 + b^3) / 2 ≥ ((a + b) / 2)^3 :=
by 
  sorry

end cube_inequality_l95_95080


namespace isosceles_trapezoid_side_length_l95_95215

theorem isosceles_trapezoid_side_length (A b1 b2 : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 48) (hb1 : b1 = 9) (hb2 : b2 = 15) 
  (h_area : A = 1 / 2 * (b1 + b2) * h) 
  (h_h : h = 4)
  (h_s : s^2 = h^2 + ((b2 - b1) / 2)^2) :
  s = 5 :=
by sorry

end isosceles_trapezoid_side_length_l95_95215


namespace total_amount_spent_l95_95634

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

end total_amount_spent_l95_95634


namespace tree_growth_per_year_l95_95979

-- Defining the initial height and age.
def initial_height : ℕ := 5
def initial_age : ℕ := 1

-- Defining the height and age after a certain number of years.
def height_at_7_years : ℕ := 23
def age_at_7_years : ℕ := 7

-- Calculating the total growth and number of years.
def total_height_growth : ℕ := height_at_7_years - initial_height
def years_of_growth : ℕ := age_at_7_years - initial_age

-- Stating the theorem to be proven.
theorem tree_growth_per_year : total_height_growth / years_of_growth = 3 :=
by
  sorry

end tree_growth_per_year_l95_95979


namespace probability_of_A_l95_95477

variable (A B : Prop)
variable (P : Prop → ℝ)

-- Given conditions
variable (h1 : P (A ∧ B) = 0.72)
variable (h2 : P (A ∧ ¬B) = 0.18)

theorem probability_of_A: P A = 0.90 := sorry

end probability_of_A_l95_95477


namespace largest_angle_in_triangle_l95_95843

theorem largest_angle_in_triangle (A B C : ℝ) 
  (a b c : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hABC : A + B + C = 180) (h_sin : Real.sin A + Real.sin C = Real.sqrt 2 * Real.sin B)
  : B = 90 :=
by
  sorry

end largest_angle_in_triangle_l95_95843


namespace jack_black_balloons_l95_95208

def nancy_balloons := 7
def mary_balloons := 4 * nancy_balloons
def total_mary_nancy_balloons := nancy_balloons + mary_balloons
def jack_balloons := total_mary_nancy_balloons + 3

theorem jack_black_balloons : jack_balloons = 38 := by
  -- proof goes here
  sorry

end jack_black_balloons_l95_95208


namespace divisibility_by_seven_l95_95461

theorem divisibility_by_seven (n : ℤ) (b : ℤ) (a : ℤ) (h : n = 10 * a + b) 
  (hb : 0 ≤ b) (hb9 : b ≤ 9) (ha : 0 ≤ a) (d : ℤ) (hd : d = a - 2 * b) :
  (2 * n + d) % 7 = 0 ↔ n % 7 = 0 := 
by
  sorry

end divisibility_by_seven_l95_95461


namespace solve_equation_l95_95526

theorem solve_equation : ∃ x : ℚ, 3 * (x - 2) = x - (2 * x - 1) ∧ x = 7/4 := by
  sorry

end solve_equation_l95_95526


namespace simplify_and_evaluate_expression_l95_95129

theorem simplify_and_evaluate_expression 
  (a b : ℚ) 
  (ha : a = 2) 
  (hb : b = 1 / 3) : 
  (a / (a - b)) * ((1 / b) - (1 / a)) + ((a - 1) / b) = 6 := 
by
  -- Place the steps verifying this here. For now:
  sorry

end simplify_and_evaluate_expression_l95_95129


namespace valid_values_for_D_l95_95001

-- Definitions for the distinct digits and the non-zero condition
def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9
def distinct_nonzero_digits (A B C D : ℕ) : Prop :=
  is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

-- Condition for the carry situation
def carry_in_addition (A B C D : ℕ) : Prop :=
  ∃ carry1 carry2 carry3 carry4 : ℕ,
  (A + B + carry1) % 10 = D ∧ (B + C + carry2) % 10 = A ∧
  (C + C + carry3) % 10 = B ∧ (A + B + carry4) % 10 = C ∧
  (carry1 = 1 ∨ carry2 = 1 ∨ carry3 = 1 ∨ carry4 = 1)

-- Main statement
theorem valid_values_for_D (A B C D : ℕ) :
  distinct_nonzero_digits A B C D →
  carry_in_addition A B C D →
  ∃ n, n = 5 :=
sorry

end valid_values_for_D_l95_95001


namespace total_number_of_games_l95_95177

theorem total_number_of_games (n : ℕ) (k : ℕ) (teams : Finset ℕ)
  (h_n : n = 8) (h_k : k = 2) (h_teams : teams.card = n) :
  (teams.card.choose k) = 28 :=
by
  sorry

end total_number_of_games_l95_95177


namespace time_with_walkway_l95_95716

-- Definitions
def length_walkway : ℝ := 60
def time_against_walkway : ℝ := 120
def time_stationary_walkway : ℝ := 48

-- Theorem statement
theorem time_with_walkway (v w : ℝ)
  (h1 : 60 = 120 * (v - w))
  (h2 : 60 = 48 * v)
  (h3 : v = 1.25)
  (h4 : w = 0.75) :
  60 = 30 * (v + w) :=
by
  sorry

end time_with_walkway_l95_95716


namespace complex_solution_l95_95284

theorem complex_solution (i z : ℂ) (h : i^2 = -1) (hz : (z - 2 * i) * (2 - i) = 5) : z = 2 + 3 * i :=
sorry

end complex_solution_l95_95284


namespace right_triangle_legs_l95_95598

theorem right_triangle_legs (c a b : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : ab = c^2 / 4) :
  a = c * (Real.sqrt 6 + Real.sqrt 2) / 4 ∧ b = c * (Real.sqrt 6 - Real.sqrt 2) / 4 := 
sorry

end right_triangle_legs_l95_95598


namespace range_is_80_l95_95184

def dataSet : List ℕ := [60, 100, 80, 40, 20]

def minValue (l : List ℕ) : ℕ :=
  match l with
  | [] => 0
  | (x :: xs) => List.foldl min x xs

def maxValue (l : List ℕ) : ℕ :=
  match l with
  | [] => 0
  | (x :: xs) => List.foldl max x xs

def range (l : List ℕ) : ℕ :=
  maxValue l - minValue l

theorem range_is_80 : range dataSet = 80 :=
by
  sorry

end range_is_80_l95_95184


namespace eval_expression_l95_95928

theorem eval_expression : 68 + (156 / 12) + (11 * 19) - 250 - (450 / 9) = -10 := 
by
  sorry

end eval_expression_l95_95928


namespace find_x_l95_95139

theorem find_x (x : ℝ) (h : 3 * x = 36 - x + 16) : x = 13 :=
by
  sorry

end find_x_l95_95139


namespace determine_b_div_a_l95_95613

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a * x^2 + b * x - a^2 - 7 * a

theorem determine_b_div_a
  (a b : ℝ)
  (hf_deriv : ∀ x : ℝ, (deriv (f a b)) x = 3 * x^2 + 2 * a * x + b)
  (hf_max : f a b 1 = 10)
  (hf_deriv_at_1 : (deriv (f a b)) 1 = 0) :
  b / a = -3 / 2 :=
sorry

end determine_b_div_a_l95_95613


namespace sum_of_dimensions_l95_95859

theorem sum_of_dimensions (A B C : ℝ) (h1 : A * B = 30) (h2 : A * C = 60) (h3 : B * C = 90) : A + B + C = 24 := 
sorry

end sum_of_dimensions_l95_95859


namespace speed_of_train_l95_95529

-- Define the given conditions
def length_of_bridge : ℝ := 200
def length_of_train : ℝ := 100
def time_to_cross_bridge : ℝ := 60

-- Define the speed conversion factor
def m_per_s_to_km_per_h : ℝ := 3.6

-- Prove that the speed of the train is 18 km/h
theorem speed_of_train :
  (length_of_bridge + length_of_train) / time_to_cross_bridge * m_per_s_to_km_per_h = 18 :=
by
  sorry

end speed_of_train_l95_95529


namespace books_new_arrivals_false_implies_statements_l95_95857

variable (Books : Type) -- representing the set of books in the library
variable (isNewArrival : Books → Prop) -- predicate stating if a book is a new arrival

theorem books_new_arrivals_false_implies_statements (H : ¬ ∀ b : Books, isNewArrival b) :
  (∃ b : Books, ¬ isNewArrival b) ∧ (¬ ∀ b : Books, isNewArrival b) :=
by
  sorry

end books_new_arrivals_false_implies_statements_l95_95857


namespace value_of_a_l95_95312

theorem value_of_a (x : ℝ) (h : (1 - x^32) ≠ 0):
  (8 * a / (1 - x^32) = 
   2 / (1 - x) + 2 / (1 + x) + 
   4 / (1 + x^2) + 8 / (1 + x^4) + 
   16 / (1 + x^8) + 32 / (1 + x^16)) → 
  a = 8 := sorry

end value_of_a_l95_95312


namespace functional_equation_solution_l95_95156

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x * y * f y) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) := by
  intro h
  sorry

end functional_equation_solution_l95_95156


namespace cistern_length_l95_95103

-- Definitions of the given conditions
def width : ℝ := 4
def depth : ℝ := 1.25
def total_wet_surface_area : ℝ := 49

-- Mathematical problem: prove the length of the cistern
theorem cistern_length : ∃ (L : ℝ), (L * width + 2 * L * depth + 2 * width * depth = total_wet_surface_area) ∧ L = 6 :=
by
sorry

end cistern_length_l95_95103


namespace xyz_unique_solution_l95_95311

theorem xyz_unique_solution (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_eq : x + y^2 + z^3 = x * y * z)
  (h_gcd : z = Nat.gcd x y) : x = 5 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end xyz_unique_solution_l95_95311


namespace greatest_possible_value_l95_95628

theorem greatest_possible_value (A B C D : ℕ) 
    (h1 : A + B + C + D = 200) 
    (h2 : A + B = 70) 
    (h3 : 0 < A) 
    (h4 : 0 < B) 
    (h5 : 0 < C) 
    (h6 : 0 < D) : 
    C ≤ 129 := 
sorry

end greatest_possible_value_l95_95628


namespace coefficient_a7_l95_95720

theorem coefficient_a7 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ) (x : ℝ) 
  (h : x^9 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 
          + a_4 * (x - 1)^4 + a_5 * (x - 1)^5 + a_6 * (x - 1)^6 + a_7 * (x - 1)^7 
          + a_8 * (x - 1)^8 + a_9 * (x - 1)^9) : 
  a_7 = 36 := 
by
  sorry

end coefficient_a7_l95_95720


namespace simplify_fraction_l95_95342

theorem simplify_fraction (x y : ℝ) (h : x ≠ y) : ((x^2 - y^2) / (x - y)) = x + y :=
by
  -- This is a placeholder for the actual proof
  sorry

end simplify_fraction_l95_95342


namespace clerks_needed_eq_84_l95_95085

def forms_processed_per_hour : ℕ := 25
def type_a_forms_count : ℕ := 3000
def type_b_forms_count : ℕ := 4000
def type_a_form_time_minutes : ℕ := 3
def type_b_form_time_minutes : ℕ := 4
def working_hours_per_day : ℕ := 5
def total_minutes_in_an_hour : ℕ := 60
def forms_time_needed (count : ℕ) (time_per_form : ℕ) : ℕ := count * time_per_form
def total_forms_time_needed : ℕ := forms_time_needed type_a_forms_count type_a_form_time_minutes +
                                    forms_time_needed type_b_forms_count type_b_form_time_minutes
def total_hours_needed : ℕ := total_forms_time_needed / total_minutes_in_an_hour
def clerk_hours_needed : ℕ := total_hours_needed / working_hours_per_day
def required_clerks : ℕ := Nat.ceil (clerk_hours_needed)

theorem clerks_needed_eq_84 :
  required_clerks = 84 :=
by
  sorry

end clerks_needed_eq_84_l95_95085


namespace max_sum_integers_differ_by_60_l95_95412

theorem max_sum_integers_differ_by_60 (b : ℕ) (c : ℕ) (h_diff : 0 < b) (h_sqrt : (Nat.sqrt b : ℝ) + (Nat.sqrt (b + 60) : ℝ) = (Nat.sqrt c : ℝ)) (h_not_square : ¬ ∃ (k : ℕ), k * k = c) :
  ∃ (b : ℕ), b + (b + 60) = 156 := 
sorry

end max_sum_integers_differ_by_60_l95_95412


namespace best_fit_slope_is_correct_l95_95840

open Real

noncomputable def slope_regression_line (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) :=
  (-2.5 * y1 - 1.5 * y2 + 0.5 * y3 + 3.5 * y4) / 21

theorem best_fit_slope_is_correct (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) (h1 : x1 < x2) (h2 : x2 < x3) (h3 : x3 < x4)
  (h_arith : (x4 - x3 = 2 * (x3 - x2)) ∧ (x3 - x2 = 2 * (x2 - x1))) :
  slope_regression_line x1 x2 x3 x4 y1 y2 y3 y4 = (-2.5 * y1 - 1.5 * y2 + 0.5 * y3 + 3.5 * y4) / 21 := 
sorry

end best_fit_slope_is_correct_l95_95840


namespace brad_amount_l95_95316

-- Definitions for the conditions
def total_amount (j d b : ℚ) := j + d + b = 68
def josh_twice_brad (j b : ℚ) := j = 2 * b
def josh_three_fourths_doug (j d : ℚ) := j = (3 / 4) * d

-- The theorem we want to prove
theorem brad_amount : ∃ (b : ℚ), (∃ (j d : ℚ), total_amount j d b ∧ josh_twice_brad j b ∧ josh_three_fourths_doug j d) ∧ b = 12 :=
sorry

end brad_amount_l95_95316


namespace original_cost_of_luxury_bag_l95_95077

theorem original_cost_of_luxury_bag (SP : ℝ) (profit_margin : ℝ) (original_cost : ℝ) 
  (h1 : SP = 3450) (h2 : profit_margin = 0.15) (h3 : SP = original_cost * (1 + profit_margin)) : 
  original_cost = 3000 :=
by
  sorry

end original_cost_of_luxury_bag_l95_95077


namespace amount_r_has_l95_95487

variable (p q r : ℕ)
variable (total_amount : ℕ)
variable (two_thirdsOf_pq : ℕ)

def total_money : Prop := (p + q + r = 4000)
def two_thirds_of_pq : Prop := (r = 2 * (p + q) / 3)

theorem amount_r_has : total_money p q r → two_thirds_of_pq p q r → r = 1600 := by
  intro h1 h2
  sorry

end amount_r_has_l95_95487


namespace minimum_value_a_2b_3c_l95_95424

theorem minimum_value_a_2b_3c (a b c : ℝ)
  (h : ∀ x y : ℝ, x + 2*y - 3 ≤ a*x + b*y + c ∧ a*x + b*y + c ≤ x + 2*y + 3) :
  (a + 2*b - 3*c) = -4 :=
sorry

end minimum_value_a_2b_3c_l95_95424


namespace tan_theta_is_sqrt3_div_5_l95_95364

open Real

theorem tan_theta_is_sqrt3_div_5 (theta : ℝ) (h : 2 * sin (theta + π / 3) = 3 * sin (π / 3 - theta)) :
  tan theta = sqrt 3 / 5 :=
sorry

end tan_theta_is_sqrt3_div_5_l95_95364


namespace average_weighted_score_l95_95998

theorem average_weighted_score
  (score1 score2 score3 : ℕ)
  (weight1 weight2 weight3 : ℕ)
  (h_scores : score1 = 90 ∧ score2 = 85 ∧ score3 = 80)
  (h_weights : weight1 = 5 ∧ weight2 = 2 ∧ weight3 = 3) :
  (weight1 * score1 + weight2 * score2 + weight3 * score3) / (weight1 + weight2 + weight3) = 86 := 
by
  sorry

end average_weighted_score_l95_95998


namespace avg_three_numbers_l95_95352

theorem avg_three_numbers (A B C : ℝ) 
  (h1 : A + B = 53)
  (h2 : B + C = 69)
  (h3 : A + C = 58) : 
  (A + B + C) / 3 = 30 := 
by
  sorry

end avg_three_numbers_l95_95352


namespace total_steps_needed_l95_95384

def cycles_needed (dist : ℕ) : ℕ := dist
def steps_per_cycle : ℕ := 5
def effective_steps_per_pattern : ℕ := 1

theorem total_steps_needed (dist : ℕ) (h : dist = 66) : 
  steps_per_cycle * cycles_needed dist = 330 :=
by 
  -- Placeholder for proof
  sorry

end total_steps_needed_l95_95384


namespace cistern_filling_time_l95_95029

open Real

theorem cistern_filling_time :
  let rate1 := 1 / 10
  let rate2 := 1 / 12
  let rate3 := -1 / 25
  let rate4 := 1 / 15
  let rate5 := -1 / 30
  let combined_rate := rate1 + rate2 + rate4 + rate3 + rate5
  (300 / combined_rate) = (300 / 53) := by
  let rate1 := 1 / 10
  let rate2 := 1 / 12
  let rate3 := -1 / 25
  let rate4 := 1 / 15
  let rate5 := -1 / 30
  let combined_rate := rate1 + rate2 + rate4 + rate3 + rate5
  sorry

end cistern_filling_time_l95_95029


namespace walking_speed_l95_95954

theorem walking_speed (total_time : ℕ) (distance : ℕ) (rest_interval : ℕ) (rest_time : ℕ) (rest_periods: ℕ) 
  (total_rest_time: ℕ) (total_walking_time: ℕ) (hours: ℕ) 
  (H1 : total_time = 332) 
  (H2 : distance = 50) 
  (H3 : rest_interval = 10) 
  (H4 : rest_time = 8)
  (H5 : rest_periods = distance / rest_interval - 1) 
  (H6 : total_rest_time = rest_periods * rest_time)
  (H7 : total_walking_time = total_time - total_rest_time) 
  (H8 : hours = total_walking_time / 60) : 
  (distance / hours) = 10 :=
by {
  -- proof omitted
  sorry
}

end walking_speed_l95_95954


namespace fib_subsequence_fib_l95_95046

noncomputable def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | 1     => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

theorem fib_subsequence_fib (p : ℕ) (hp : p > 0) :
  ∀ n : ℕ, fibonacci ((n - 1) * p) + fibonacci (n * p) = fibonacci ((n + 1) * p) := 
by
  sorry

end fib_subsequence_fib_l95_95046


namespace temperature_at_4km_l95_95633

theorem temperature_at_4km (ground_temp : ℤ) (drop_rate : ℤ) (altitude : ℕ) (ΔT : ℤ) : 
  ground_temp = 15 ∧ drop_rate = -5 ∧ ΔT = altitude * drop_rate ∧ altitude = 4 → 
  ground_temp + ΔT = -5 :=
by
  sorry

end temperature_at_4km_l95_95633


namespace sum_of_x_values_l95_95648

theorem sum_of_x_values (x : ℝ) (h : x ≠ -1) : 
  (∃ x, 3 = (x^3 - 3*x^2 - 4*x)/(x + 1)) →
  (x = 6) :=
by
  sorry

end sum_of_x_values_l95_95648


namespace range_of_a_l95_95751

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * (a + 2) * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, deriv (f a) x ≥ 0) ↔ -1 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l95_95751


namespace max_rectangle_area_l95_95863

theorem max_rectangle_area (P : ℝ) (hP : P = 60) : ∃ A : ℝ, A = 225 :=
by 
  sorry

end max_rectangle_area_l95_95863


namespace A_intersect_B_l95_95899

def A : Set ℝ := { x | abs x < 2 }
def B : Set ℝ := { x | x^2 - 5 * x - 6 < 0 }

theorem A_intersect_B : A ∩ B = { x | -1 < x ∧ x < 2 } := by
  sorry

end A_intersect_B_l95_95899


namespace p_computation_l95_95849

def p (x y : Int) : Int :=
  if x >= 0 ∧ y >= 0 then x + y
  else if x < 0 ∧ y < 0 then x - 3 * y
  else if x + y > 0 then 2 * x + 2 * y
  else x + 4 * y

theorem p_computation : p (p 2 (-3)) (p (-3) (-4)) = 26 := by
  sorry

end p_computation_l95_95849


namespace no_integer_solution_k_range_l95_95623

theorem no_integer_solution_k_range (k : ℝ) :
  (∀ x : ℤ, ¬ ((k * x - k^2 - 4) * (x - 4) < 0)) → (1 ≤ k ∧ k ≤ 4) :=
by
  sorry

end no_integer_solution_k_range_l95_95623


namespace no_five_integer_solutions_divisibility_condition_l95_95444

variables (k : ℤ) 

-- Definition of equation
def equation (x y : ℤ) : Prop :=
  y^2 - k = x^3

-- Variables to capture the integer solutions
variables (x1 x2 x3 x4 x5 y1 : ℤ)

-- Prove that there do not exist five solutions satisfying the given forms
theorem no_five_integer_solutions :
  ¬(equation k x1 y1 ∧ 
    equation k x2 (y1 - 1) ∧ 
    equation k x3 (y1 - 2) ∧ 
    equation k x4 (y1 - 3) ∧ 
    equation k x5 (y1 - 4)) :=
sorry

-- Prove divisibility condition for the first four solutions
theorem divisibility_condition :
  (equation k x1 y1 ∧ 
   equation k x2 (y1 - 1) ∧ 
   equation k x3 (y1 - 2) ∧ 
   equation k x4 (y1 - 3)) → 
  63 ∣ (k - 17) :=
sorry

end no_five_integer_solutions_divisibility_condition_l95_95444


namespace smallest_y_l95_95635

theorem smallest_y (y : ℝ) :
  (3 * y ^ 2 + 33 * y - 90 = y * (y + 16)) → y = -10 :=
sorry

end smallest_y_l95_95635


namespace original_average_age_l95_95941

variable (A : ℕ)
variable (N : ℕ := 2)
variable (new_avg_age : ℕ := 32)
variable (age_decrease : ℕ := 4)

theorem original_average_age :
  (A * N + new_avg_age * 2) / (N + 2) = A - age_decrease → A = 40 := 
by
  sorry

end original_average_age_l95_95941


namespace fraction_of_income_to_taxes_l95_95544

noncomputable def joe_income : ℕ := 2120
noncomputable def joe_taxes : ℕ := 848

theorem fraction_of_income_to_taxes : (joe_taxes / gcd joe_taxes joe_income) / (joe_income / gcd joe_taxes joe_income) = 106 / 265 := sorry

end fraction_of_income_to_taxes_l95_95544


namespace profit_distribution_l95_95084

theorem profit_distribution (investment_LiWei investment_WangGang profit total_investment : ℝ)
  (h1 : investment_LiWei = 16000)
  (h2 : investment_WangGang = 12000)
  (h3 : profit = 14000)
  (h4 : total_investment = investment_LiWei + investment_WangGang) :
  (profit * (investment_LiWei / total_investment) = 8000) ∧ 
  (profit * (investment_WangGang / total_investment) = 6000) :=
by
  sorry

end profit_distribution_l95_95084


namespace express_in_scientific_notation_l95_95104

def scientific_notation (n : ℤ) (x : ℝ) :=
  ∃ (a : ℝ) (b : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ x = a * 10^b

theorem express_in_scientific_notation : scientific_notation (-8206000) (-8.206 * 10^6) :=
by
  sorry

end express_in_scientific_notation_l95_95104


namespace jill_commute_time_l95_95652

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

end jill_commute_time_l95_95652


namespace tangent_line_at_1_tangent_line_through_2_3_l95_95392

-- Define the function
def f (x : ℝ) : ℝ := x^3

-- Derivative of the function
def f' (x : ℝ) : ℝ := 3 * x^2

-- Problem 1: Prove that the tangent line at point (1, 1) is y = 3x - 2
theorem tangent_line_at_1 (x y : ℝ) (h : y = f 1 + f' 1 * (x - 1)) : y = 3 * x - 2 := 
sorry

-- Problem 2: Prove that the tangent line passing through (2/3, 0) is either y = 0 or y = 3x - 2
theorem tangent_line_through_2_3 (x y x0 : ℝ) 
  (hx0 : y = f x0 + f' x0 * (x - x0))
  (hp : 0 = f' x0 * (2/3 - x0)) :
  y = 0 ∨ y = 3 * x - 2 := 
sorry

end tangent_line_at_1_tangent_line_through_2_3_l95_95392


namespace max_age_l95_95447

-- Definitions of the conditions
def born_same_day (max_birth luka_turn4 : ℕ) : Prop := max_birth = luka_turn4
def age_difference (luka_age aubrey_age : ℕ) : Prop := luka_age = aubrey_age + 2
def aubrey_age_on_birthday : ℕ := 8

-- Prove that Max's age is 6 years when Aubrey is 8 years old
theorem max_age (luka_birth aubrey_birth max_birth : ℕ) 
                (h1 : born_same_day max_birth luka_birth) 
                (h2 : age_difference luka_birth aubrey_birth) : 
                (aubrey_birth + 4 - luka_birth) = 6 :=
by
  sorry

end max_age_l95_95447


namespace pipe_b_fills_tank_7_times_faster_l95_95886

theorem pipe_b_fills_tank_7_times_faster 
  (time_A : ℝ) 
  (time_B : ℝ)
  (combined_time : ℝ) 
  (hA : time_A = 30)
  (h_combined : combined_time = 3.75) 
  (hB : time_B = time_A / 7) :
  time_B =  30 / 7 :=
by
  sorry

end pipe_b_fills_tank_7_times_faster_l95_95886


namespace fraction_of_journey_asleep_l95_95766

theorem fraction_of_journey_asleep (x y : ℝ) (hx : x > 0) (hy : y = x / 3) :
  y / x = 1 / 3 :=
by
  sorry

end fraction_of_journey_asleep_l95_95766


namespace pencils_placed_by_sara_l95_95471

theorem pencils_placed_by_sara (initial_pencils final_pencils : ℕ) (h1 : initial_pencils = 115) (h2 : final_pencils = 215) : final_pencils - initial_pencils = 100 := by
  sorry

end pencils_placed_by_sara_l95_95471


namespace abs_f_sub_lt_abs_l95_95547

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 + x^2)

theorem abs_f_sub_lt_abs (a b : ℝ) (h : a ≠ b) : 
  |f a - f b| < |a - b| := 
by
  sorry

end abs_f_sub_lt_abs_l95_95547


namespace regions_first_two_sets_regions_all_sets_l95_95173

-- Definitions for the problem
def triangle_regions_first_two_sets (n : ℕ) : ℕ :=
  (n + 1) * (n + 1)

def triangle_regions_all_sets (n : ℕ) : ℕ :=
  3 * n * n + 3 * n + 1

-- Proof Problem 1: Given n points on AB and AC, prove the regions are (n + 1)^2
theorem regions_first_two_sets (n : ℕ) :
  (n * (n + 1) + (n + 1)) = (n + 1) * (n + 1) :=
by sorry

-- Proof Problem 2: Given n points on AB, AC, and BC, prove the regions are 3n^2 + 3n + 1
theorem regions_all_sets (n : ℕ) :
  ((n + 1) * (n + 1) + n * (2 * n + 1)) = 3 * n * n + 3 * n + 1 :=
by sorry

end regions_first_two_sets_regions_all_sets_l95_95173


namespace percentage_problem_l95_95929

theorem percentage_problem (N : ℕ) (P : ℕ) (h1 : N = 25) (h2 : N = (P * N / 100) + 21) : P = 16 :=
sorry

end percentage_problem_l95_95929


namespace smallest_positive_period_of_f_max_min_values_of_f_on_interval_l95_95307

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) ^ 2 + Real.cos (2 * x)

theorem smallest_positive_period_of_f : ∀ (x : ℝ), f (x + π) = f x :=
by sorry

theorem max_min_values_of_f_on_interval : ∃ (x₁ x₂ : ℝ), 0 ≤ x₁ ∧ x₁ ≤ π / 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ π / 2 ∧
  f x₁ = 0 ∧ f x₂ = 1 + Real.sqrt 2 :=
by sorry

end smallest_positive_period_of_f_max_min_values_of_f_on_interval_l95_95307


namespace common_root_iff_cond_l95_95246

theorem common_root_iff_cond (p1 p2 q1 q2 : ℂ) :
  (∃ x : ℂ, x^2 + p1 * x + q1 = 0 ∧ x^2 + p2 * x + q2 = 0) ↔
  (q2 - q1)^2 + (p1 - p2) * (p1 * q2 - q1 * p2) = 0 :=
by
  sorry

end common_root_iff_cond_l95_95246


namespace root_in_interval_l95_95592

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 2

theorem root_in_interval : 
  (f 1 < 0) → (f 2 > 0) → ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  intros h1 h2
  sorry

end root_in_interval_l95_95592


namespace initial_performers_count_l95_95787

theorem initial_performers_count (n : ℕ)
    (h1 : ∃ rows, 8 * rows = n)
    (h2 : ∃ (m : ℕ), n + 16 = m ∧ ∃ s, s * s = m)
    (h3 : ∃ (k : ℕ), n + 1 = k ∧ ∃ t, t * t = k) : 
    n = 48 := 
sorry

end initial_performers_count_l95_95787


namespace find_value_l95_95454

theorem find_value (x : ℝ) (h : 0.20 * x = 80) : 0.40 * x = 160 := 
by
  sorry

end find_value_l95_95454


namespace measure_85_liters_l95_95944

theorem measure_85_liters (C1 C2 C3 : ℕ) (capacity : ℕ) : 
  (C1 = 0 ∧ C2 = 0 ∧ C3 = 1 ∧ capacity = 85) → 
  (∃ weighings : ℕ, weighings ≤ 8 ∧ C1 = 85 ∨ C2 = 85 ∨ C3 = 85) :=
by 
  sorry

end measure_85_liters_l95_95944


namespace billy_restaurant_total_payment_l95_95188

noncomputable def cost_of_meal
  (adult_count child_count : ℕ)
  (adult_cost child_cost : ℕ) : ℕ :=
  adult_count * adult_cost + child_count * child_cost

noncomputable def cost_of_dessert
  (total_people : ℕ)
  (dessert_cost : ℕ) : ℕ :=
  total_people * dessert_cost

noncomputable def total_cost_before_discount
  (adult_count child_count : ℕ)
  (adult_cost child_cost dessert_cost : ℕ) : ℕ :=
  (cost_of_meal adult_count child_count adult_cost child_cost) +
  (cost_of_dessert (adult_count + child_count) dessert_cost)

noncomputable def discount_amount
  (total : ℕ)
  (discount_rate : ℝ) : ℝ :=
  total * discount_rate

noncomputable def total_amount_to_pay
  (total : ℕ)
  (discount : ℝ) : ℝ :=
  total - discount

theorem billy_restaurant_total_payment :
  total_amount_to_pay
  (total_cost_before_discount 2 5 7 3 2)
  (discount_amount (total_cost_before_discount 2 5 7 3 2) 0.15) = 36.55 := by
  sorry

end billy_restaurant_total_payment_l95_95188


namespace medieval_society_hierarchy_l95_95403

-- Given conditions
def members := 12
def king_choices := members
def remaining_after_king := members - 1
def duke_choices : ℕ := remaining_after_king * (remaining_after_king - 1) * (remaining_after_king - 2)
def knight_choices : ℕ := Nat.choose (remaining_after_king - 2) 2 * Nat.choose (remaining_after_king - 4) 2 * Nat.choose (remaining_after_king - 6) 2

-- The number of ways to establish the hierarchy can be stated as:
def total_ways : ℕ := king_choices * duke_choices * knight_choices

-- Our main theorem
theorem medieval_society_hierarchy : total_ways = 907200 := by
  -- Proof would go here, we skip it with sorry
  sorry

end medieval_society_hierarchy_l95_95403


namespace old_geometry_book_pages_l95_95939

def old_pages := 340
def new_pages := 450
def deluxe_pages := 915

theorem old_geometry_book_pages : 
  (new_pages = 2 * old_pages - 230) ∧ 
  (deluxe_pages = new_pages + old_pages + 125) ∧ 
  (deluxe_pages ≥ old_pages + old_pages / 10) 
  → old_pages = 340 := by
  sorry

end old_geometry_book_pages_l95_95939


namespace smallest_solution_is_neg_sqrt_13_l95_95649

noncomputable def smallest_solution (x : ℝ) : Prop :=
  x^4 - 26 * x^2 + 169 = 0 ∧ ∀ y : ℝ, y^4 - 26 * y^2 + 169 = 0 → x ≤ y

theorem smallest_solution_is_neg_sqrt_13 :
  smallest_solution (-Real.sqrt 13) :=
by
  sorry

end smallest_solution_is_neg_sqrt_13_l95_95649


namespace find_a4_l95_95405

variable {α : Type*} [Field α] [Inhabited α]

-- Definitions of the geometric sequence conditions
def geometric_sequence_condition1 (a₁ q : α) : Prop :=
  a₁ * (1 + q) = -1

def geometric_sequence_condition2 (a₁ q : α) : Prop :=
  a₁ * (1 - q^2) = -3

-- Definition of the geometric sequence
def geometric_sequence (a₁ q : α) (n : ℕ) : α :=
  a₁ * q^n

-- The theorem to be proven
theorem find_a4 (a₁ q : α) (h₁ : geometric_sequence_condition1 a₁ q) (h₂ : geometric_sequence_condition2 a₁ q) :
  geometric_sequence a₁ q 3 = -8 :=
  sorry

end find_a4_l95_95405


namespace find_u_value_l95_95946

theorem find_u_value (u : ℤ) : ∀ (y : ℤ → ℤ), 
  (y 2 = 8) → (y 4 = 14) → (y 6 = 20) → 
  (∀ x, (x % 2 = 0) → (y (x + 2) = y x + 6)) → 
  y 18 = u → u = 56 :=
by
  intros y h2 h4 h6 pattern h18
  sorry

end find_u_value_l95_95946


namespace boat_speed_ratio_l95_95141

variable (B S : ℝ)

theorem boat_speed_ratio (h : 1 / (B - S) = 2 * (1 / (B + S))) : B / S = 3 := 
by
  sorry

end boat_speed_ratio_l95_95141


namespace soldiers_count_l95_95799

-- Statements of conditions and proofs
theorem soldiers_count (n : ℕ) (s : ℕ) :
  (n * n + 30 = s) →
  ((n + 1) * (n + 1) - 50 = s) →
  s = 1975 :=
by
  intros h1 h2
  -- We know from h1 and h2 that there should be a unique solution for s and n that satisfies both
  -- conditions. Our goal is to show that s must be 1975.

  -- Initialize the proof structure
  sorry

end soldiers_count_l95_95799


namespace classroom_not_1_hectare_l95_95622

def hectare_in_sq_meters : ℕ := 10000
def classroom_area_approx : ℕ := 60

theorem classroom_not_1_hectare : ¬ (classroom_area_approx = hectare_in_sq_meters) :=
by 
  sorry

end classroom_not_1_hectare_l95_95622


namespace max_geometric_sequence_terms_l95_95722

theorem max_geometric_sequence_terms (a r : ℝ) (n : ℕ) (h_r : r > 1) 
    (h_seq : ∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → 100 ≤ a * r^(k-1) ∧ a * r^(k-1) ≤ 1000) :
  n ≤ 6 :=
sorry

end max_geometric_sequence_terms_l95_95722


namespace awareness_survey_sampling_l95_95915

theorem awareness_survey_sampling
  (students : Set ℝ) -- assumption that defines the set of students
  (grades : Set ℝ) -- assumption that defines the set of grades
  (awareness : ℝ → ℝ) -- assumption defining the awareness function
  (significant_differences : ∀ g1 g2 : ℝ, g1 ≠ g2 → awareness g1 ≠ awareness g2) -- significant differences in awareness among grades
  (first_grade_students : Set ℝ) -- assumption defining the set of first grade students
  (second_grade_students : Set ℝ) -- assumption defining the set of second grade students
  (third_grade_students : Set ℝ) -- assumption defining the set of third grade students
  (students_from_grades : students = first_grade_students ∪ second_grade_students ∪ third_grade_students) -- assumption that the students are from first, second, and third grades
  (representative_method : (simple_random_sampling → False) ∧ (systematic_sampling_method → False))
  : stratified_sampling_method := 
sorry

end awareness_survey_sampling_l95_95915


namespace find_balls_l95_95609

theorem find_balls (x y : ℕ) (h1 : (x + y : ℚ) / (x + y + 18) = (x + 18) / (x + y + 18) - 1 / 15)
                   (h2 : (y + 18 : ℚ) / (x + y + 18) = (x + 18) / (x + y + 18) * 11 / 10) :
  x = 12 ∧ y = 15 :=
sorry

end find_balls_l95_95609


namespace triangle_area_eq_40_sqrt_3_l95_95297

open Real

theorem triangle_area_eq_40_sqrt_3 
  (a : ℝ) (A : ℝ) (b c : ℝ)
  (h1 : a = 14)
  (h2 : A = π / 3) -- 60 degrees in radians
  (h3 : b / c = 8 / 5) :
  1 / 2 * b * c * sin A = 40 * sqrt 3 :=
by
  sorry

end triangle_area_eq_40_sqrt_3_l95_95297


namespace kosher_clients_count_l95_95231

def T := 30
def V := 7
def VK := 3
def Neither := 18

theorem kosher_clients_count (K : ℕ) : T - Neither = V + K - VK → K = 8 :=
by
  intro h
  sorry

end kosher_clients_count_l95_95231


namespace correct_propositions_l95_95837

theorem correct_propositions (a b c d m : ℝ) :
  (ab > 0 → a > b → (1 / a < 1 / b)) ∧
  (a > |b| → a ^ 2 > b ^ 2) ∧
  ¬ (a > b ∧ c < d → a - d > b - c) ∧
  ¬ (a < b ∧ m > 0 → a / b < (a + m) / (b + m)) :=
by sorry

end correct_propositions_l95_95837


namespace log_inequality_sqrt_inequality_l95_95582

-- Proof problem for part (1)
theorem log_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.log ((a + b) / 2) ≥ (Real.log a + Real.log b) / 2 :=
sorry

-- Proof problem for part (2)
theorem sqrt_inequality :
  Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 :=
sorry

end log_inequality_sqrt_inequality_l95_95582


namespace initial_apples_count_l95_95521

variable (initial_apples : ℕ)
variable (used_apples : ℕ := 2)
variable (bought_apples : ℕ := 23)
variable (final_apples : ℕ := 38)

theorem initial_apples_count :
  initial_apples - used_apples + bought_apples = final_apples ↔ initial_apples = 17 := by
  sorry

end initial_apples_count_l95_95521


namespace find_m_l95_95851

theorem find_m (x y m : ℤ) 
  (h1 : 4 * x + y = 34)
  (h2 : m * x - y = 20)
  (h3 : y ^ 2 = 4) 
  : m = 2 :=
sorry

end find_m_l95_95851


namespace ratio_w_to_y_l95_95812

theorem ratio_w_to_y (w x y z : ℝ) (h1 : w / x = 4 / 3) (h2 : y / z = 5 / 3) (h3 : z / x = 1 / 5) :
  w / y = 4 :=
by
  sorry

end ratio_w_to_y_l95_95812


namespace compare_neg5_neg2_compare_neg_third_neg_half_compare_absneg5_0_l95_95580

theorem compare_neg5_neg2 : -5 < -2 :=
by sorry

theorem compare_neg_third_neg_half : -(1/3) > -(1/2) :=
by sorry

theorem compare_absneg5_0 : abs (-5) > 0 :=
by sorry

end compare_neg5_neg2_compare_neg_third_neg_half_compare_absneg5_0_l95_95580


namespace arithmetic_sequence_sum_l95_95369

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_a7 : a 7 = 12) :
  a 3 + a 11 = 24 :=
by
  sorry

end arithmetic_sequence_sum_l95_95369


namespace solve_for_x_l95_95476

theorem solve_for_x (x : ℝ) : 0.05 * x + 0.1 * (30 + x) = 15.5 → x = 83 := by 
  sorry

end solve_for_x_l95_95476


namespace a_squared_plus_b_squared_less_than_c_squared_l95_95936

theorem a_squared_plus_b_squared_less_than_c_squared 
  (a b c : Real) 
  (h : a^2 + b^2 + a * b + b * c + c * a < 0) : 
  a^2 + b^2 < c^2 := 
  by 
  sorry

end a_squared_plus_b_squared_less_than_c_squared_l95_95936


namespace find_other_number_l95_95247

-- Given conditions
def sum_of_numbers (x y : ℕ) : Prop := x + y = 72
def number_difference (x y : ℕ) : Prop := x = y + 12
def one_number_is_30 (x : ℕ) : Prop := x = 30

-- Theorem to prove
theorem find_other_number (y : ℕ) : 
  sum_of_numbers y 30 ∧ number_difference 30 y → y = 18 := by
  sorry

end find_other_number_l95_95247


namespace find_m_values_l95_95590

theorem find_m_values :
  ∃ m : ℝ, (∀ (α β : ℝ), (3 * α^2 + m * α - 4 = 0 ∧ 3 * β^2 + m * β - 4 = 0) ∧ (α * β = -4 / 3) ∧ (α + β = -m / 3) ∧ (α * β = 2 * (α^3 + β^3))) ↔
  (m = -1.5 ∨ m = 6 ∨ m = -2.4) :=
sorry

end find_m_values_l95_95590


namespace a_wins_by_200_meters_l95_95900

-- Define the conditions
def race_distance : ℕ := 600
def speed_ratio_a_to_b := 5 / 4
def head_start_a : ℕ := 100

-- Define the proof statement
theorem a_wins_by_200_meters (x : ℝ) (ha_speed : ℝ := 5 * x) (hb_speed : ℝ := 4 * x)
  (ha_distance_to_win : ℝ := race_distance - head_start_a) :
  (ha_distance_to_win / ha_speed) = (400 / hb_speed) → 
  600 - (400) = 200 :=
by
  -- For now, skip the proof, focus on the statement.
  sorry

end a_wins_by_200_meters_l95_95900


namespace property_tax_increase_is_800_l95_95631

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

end property_tax_increase_is_800_l95_95631


namespace least_pos_int_div_by_four_distinct_primes_l95_95457

/-- 
The least positive integer that is divisible by four distinct primes is 210.
-/
theorem least_pos_int_div_by_four_distinct_primes : 
  ∃ (n : ℕ), (∀ (p : ℕ), Prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) ∧ 
             (∀ (m : ℕ), (∀ (p : ℕ), Prime p → p ∣ m → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → n ≤ m) ∧ 
             n = 210 := 
sorry

end least_pos_int_div_by_four_distinct_primes_l95_95457


namespace hemisphere_surface_area_l95_95265

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (area_base : ℝ) (surface_area_sphere : ℝ) (Q : ℝ) : 
  area_base = 3 ∧ surface_area_sphere = 4 * π * r^2 → Q = 9 :=
by
  sorry

end hemisphere_surface_area_l95_95265


namespace general_term_sum_formula_l95_95293

-- Conditions for the sequence
variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (S : ℕ → ℤ)

-- Given conditions
axiom a2_eq_5 : a 2 = 5
axiom S4_eq_28 : S 4 = 28

-- The sequence is an arithmetic sequence
axiom arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + d

-- Statement 1: Proof that a_n = 4n - 3
theorem general_term (n : ℕ) : a n = 4 * n - 3 :=
by
  sorry

-- Statement 2: Proof that S_n = 2n^2 - n
theorem sum_formula (n : ℕ) : S n = 2 * n^2 - n :=
by
  sorry

end general_term_sum_formula_l95_95293


namespace find_x_l95_95985

-- Define the variables and conditions
def x := 27
axiom h : (5 / 3) * x = 45

-- Main statement to be proved
theorem find_x : x = 27 :=
by
  have : (5 / 3) * x = 45 := h
  sorry

end find_x_l95_95985


namespace proof_problem_l95_95058

variable (γ θ α : ℝ)
variable (x y : ℝ)

def condition1 := x = γ * Real.sin ((θ - α) / 2)
def condition2 := y = γ * Real.sin ((θ + α) / 2)

theorem proof_problem
  (h1 : condition1 γ θ α x)
  (h2 : condition2 γ θ α y)
  : x^2 - 2*x*y*Real.cos α + y^2 = γ^2 * (Real.sin α)^2 :=
by
  sorry

end proof_problem_l95_95058


namespace sufficient_but_not_necessary_condition_l95_95095

theorem sufficient_but_not_necessary_condition (a : ℝ) : (a < -1) → (|a| > 1) ∧ ¬((|a| > 1) → (a < -1)) :=
by
-- This statement represents the required proof.
sorry

end sufficient_but_not_necessary_condition_l95_95095


namespace total_marks_l95_95105

theorem total_marks (k l d : ℝ) (hk : k = 3.5) (hl : l = 3.2 * k) (hd : d = l + 5.7) : k + l + d = 31.6 :=
by
  rw [hk] at hl
  rw [hl] at hd
  rw [hk, hl, hd]
  sorry

end total_marks_l95_95105


namespace order_of_abc_l95_95602

theorem order_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h1 : a^2 + b^2 < a^2 + c^2) (h2 : a^2 + c^2 < b^2 + c^2) : a < b ∧ b < c := 
by
  sorry

end order_of_abc_l95_95602


namespace line_equation_parametric_to_implicit_l95_95362

theorem line_equation_parametric_to_implicit (t : ℝ) :
  ∀ x y : ℝ, (x = 3 * t + 6 ∧ y = 5 * t - 7) → y = (5 / 3) * x - 17 :=
by
  intros x y h
  obtain ⟨hx, hy⟩ := h
  sorry

end line_equation_parametric_to_implicit_l95_95362


namespace sum_first_n_terms_eq_l95_95298

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1

noncomputable def b_n (n : ℕ) : ℕ := 2 ^ (n - 1)

noncomputable def c_n (n : ℕ) : ℕ := a_n n * b_n n

noncomputable def T_n (n : ℕ) : ℕ := (2 * n - 3) * 2 ^ n + 3

theorem sum_first_n_terms_eq (n : ℕ) : 
  (Finset.sum (Finset.range n.succ) (λ k => c_n k) = T_n n) :=
  sorry

end sum_first_n_terms_eq_l95_95298


namespace sum_of_a3_a4_a5_l95_95114

def geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a n = 3 * q ^ n

theorem sum_of_a3_a4_a5 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geometric : geometric_sequence_sum a q)
  (h_pos : ∀ n, a n > 0)
  (h_first_term : a 0 = 3)
  (h_sum_first_three : a 0 + a 1 + a 2 = 21) :
  a 2 + a 3 + a 4 = 84 :=
sorry

end sum_of_a3_a4_a5_l95_95114


namespace find_side_a_find_area_l95_95353

-- Definitions from the conditions
variables {A B C : ℝ} 
variables {a b c : ℝ}
variable (angle_B: B = 120 * Real.pi / 180)
variable (side_b: b = Real.sqrt 7)
variable (side_c: c = 1)

-- The first proof problem: Prove that a = 2 given the above conditions
theorem find_side_a (h_angle_B: B = 120 * Real.pi / 180)
  (h_side_b: b = Real.sqrt 7) (h_side_c: c = 1)
  (h_cos_formula: b^2 = a^2 + c^2 - 2 * a * c * Real.cos B) : a = 2 :=
  by
  sorry

-- The second proof problem: Prove that the area is sqrt(3)/2 given the above conditions
theorem find_area (h_angle_B: B = 120 * Real.pi / 180)
  (h_side_b: b = Real.sqrt 7) (h_side_c: c = 1)
  (h_side_a: a = 2) : (1 / 2) * a * c * Real.sin B = Real.sqrt 3 / 2 :=
  by
  sorry

end find_side_a_find_area_l95_95353


namespace repeating_block_digits_l95_95603

theorem repeating_block_digits (n d : ℕ) (h1 : n = 3) (h2 : d = 11) : 
  (∃ repeating_block_length, repeating_block_length = 2) := by
  sorry

end repeating_block_digits_l95_95603


namespace lunas_phone_bill_percentage_l95_95903

variables (H F P : ℝ)

theorem lunas_phone_bill_percentage :
  F = 0.60 * H ∧ H + F = 240 ∧ H + F + P = 249 →
  (P / F) * 100 = 10 :=
by
  intros
  sorry

end lunas_phone_bill_percentage_l95_95903


namespace centers_collinear_l95_95192

theorem centers_collinear (k : ℝ) (hk : k ≠ -1) :
    ∀ p : ℝ × ℝ, p = (-k, -2*k-5) → (2*p.1 - p.2 - 5 = 0) :=
by
  sorry

end centers_collinear_l95_95192


namespace ratio_of_fuji_trees_l95_95248

variable (F T : ℕ) -- Declaring F as number of pure Fuji trees, T as total number of trees
variables (C : ℕ) -- Declaring C as number of cross-pollinated trees 

theorem ratio_of_fuji_trees 
  (h1: 10 * C = T) 
  (h2: F + C = 221) 
  (h3: T = F + 39 + C) : 
  F * 52 = 39 * T := 
sorry

end ratio_of_fuji_trees_l95_95248


namespace negation_of_proposition_l95_95608

theorem negation_of_proposition (a b c : ℝ) :
  ¬ (a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c ≠ 3 → a^2 + b^2 + c^2 < 3) :=
sorry

end negation_of_proposition_l95_95608


namespace vector_perpendicular_solution_l95_95253

noncomputable def a (m : ℝ) : ℝ × ℝ := (1, m)
def b : ℝ × ℝ := (3, -2)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_perpendicular_solution (m : ℝ) (h : dot_product (a m + b) b = 0) : m = 8 := by
  sorry

end vector_perpendicular_solution_l95_95253


namespace wendy_full_face_time_l95_95439

-- Define the constants based on the conditions
def num_products := 5
def wait_time := 5
def makeup_time := 30

-- Calculate the total time to put on "full face"
def total_time (products : ℕ) (wait_time : ℕ) (makeup_time : ℕ) : ℕ :=
  (products - 1) * wait_time + makeup_time

-- The theorem stating that Wendy's full face routine takes 50 minutes
theorem wendy_full_face_time : total_time num_products wait_time makeup_time = 50 :=
by {
  -- the proof would be provided here, for now we use sorry
  sorry
}

end wendy_full_face_time_l95_95439


namespace min_cost_to_package_fine_arts_collection_l95_95351

theorem min_cost_to_package_fine_arts_collection :
  let box_length := 20
  let box_width := 20
  let box_height := 12
  let cost_per_box := 0.50
  let required_volume := 1920000
  let volume_of_one_box := box_length * box_width * box_height
  let number_of_boxes := required_volume / volume_of_one_box
  let total_cost := number_of_boxes * cost_per_box
  total_cost = 200 := 
by
  sorry

end min_cost_to_package_fine_arts_collection_l95_95351


namespace sum_xyz_l95_95201

variables (x y z : ℤ)

theorem sum_xyz (h1 : y = 3 * x) (h2 : z = 3 * y - x) : x + y + z = 12 * x :=
by 
  -- skip the proof
  sorry

end sum_xyz_l95_95201


namespace catherine_friends_count_l95_95779

/-
Definition and conditions:
- An equal number of pencils and pens, totaling 60 each.
- Gave away 8 pens and 6 pencils to each friend.
- Left with 22 pens and pencils.
Proof:
- The number of friends she gave pens and pencils to equals 7.
-/
theorem catherine_friends_count :
  ∀ (pencils pens friends : ℕ),
  pens = 60 →
  pencils = 60 →
  (pens + pencils) - friends * (8 + 6) = 22 →
  friends = 7 :=
sorry

end catherine_friends_count_l95_95779


namespace money_needed_to_finish_collection_l95_95522

-- Define the conditions
def initial_action_figures : ℕ := 9
def total_action_figures_needed : ℕ := 27
def cost_per_action_figure : ℕ := 12

-- Define the goal
theorem money_needed_to_finish_collection 
  (initial : ℕ) (total_needed : ℕ) (cost_per : ℕ) 
  (h1 : initial = initial_action_figures)
  (h2 : total_needed = total_action_figures_needed)
  (h3 : cost_per = cost_per_action_figure) :
  ((total_needed - initial) * cost_per = 216) := 
by
  sorry

end money_needed_to_finish_collection_l95_95522


namespace find_x_y_l95_95146

theorem find_x_y 
  (x y : ℚ)
  (h1 : (x / 6) * 12 = 10)
  (h2 : (y / 4) * 8 = x) :
  x = 5 ∧ y = 2.5 :=
by
  sorry

end find_x_y_l95_95146


namespace contestant_final_score_l95_95763

theorem contestant_final_score 
    (content_score : ℕ)
    (delivery_score : ℕ)
    (weight_content : ℕ)
    (weight_delivery : ℕ)
    (h1 : content_score = 90)
    (h2 : delivery_score = 85)
    (h3 : weight_content = 6)
    (h4 : weight_delivery = 4) : 
    (content_score * weight_content + delivery_score * weight_delivery) / (weight_content + weight_delivery) = 88 := 
sorry

end contestant_final_score_l95_95763


namespace range_of_b_l95_95072

-- Given a function f(x)
def f (b x : ℝ) : ℝ := x^3 - 3 * b * x + 3 * b

-- Derivative of the function f(x)
def f' (b x : ℝ) : ℝ := 3 * x^2 - 3 * b

-- The theorem to prove the range of b
theorem range_of_b (b : ℝ) : (∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ f' b x = 0) → (0 < b ∧ b < 1) := by
  sorry

end range_of_b_l95_95072


namespace complex_number_solution_l95_95452

theorem complex_number_solution
  (z : ℂ)
  (h : i * (z - 1) = 1 + i) :
  z = 2 - i :=
sorry

end complex_number_solution_l95_95452


namespace percentage_water_in_puree_l95_95455

/-- Given that tomato juice is 90% water and Heinz obtains 2.5 litres of tomato puree from 20 litres of tomato juice,
proves that the percentage of water in the tomato puree is 20%. -/
theorem percentage_water_in_puree (tj_volume : ℝ) (tj_water_content : ℝ) (tp_volume : ℝ) (tj_to_tp_ratio : ℝ) 
  (h1 : tj_water_content = 0.90) 
  (h2 : tj_volume = 20) 
  (h3 : tp_volume = 2.5) 
  (h4 : tj_to_tp_ratio = tj_volume / tp_volume) : 
  ((tp_volume - (1 - tj_water_content) * (tj_volume * (tp_volume / tj_volume))) / tp_volume) * 100 = 20 := 
sorry

end percentage_water_in_puree_l95_95455


namespace arithmetic_sequence_a3_value_l95_95559

theorem arithmetic_sequence_a3_value 
  (a : ℕ → ℤ) 
  (h1 : ∀ n, a (n + 1) = a n + 2) 
  (h2 : (a 1 + 2)^2 = a 1 * (a 1 + 8)) : 
  a 2 = 5 := 
by 
  sorry

end arithmetic_sequence_a3_value_l95_95559


namespace proof_ineq_l95_95389

noncomputable def P (f g : ℤ → ℤ) (m n k : ℕ) :=
  (∀ x y : ℤ, -1000 ≤ x ∧ x ≤ 1000 ∧ -1000 ≤ y ∧ y ≤ 1000 ∧ f x = g y → m = m + 1) ∧
  (∀ x y : ℤ, -1000 ≤ x ∧ x ≤ 1000 ∧ -1000 ≤ y ∧ y ≤ 1000 ∧ f x = f y → n = n + 1) ∧
  (∀ x y : ℤ, -1000 ≤ x ∧ x ≤ 1000 ∧ -1000 ≤ y ∧ y ≤ 1000 ∧ g x = g y → k = k + 1)

theorem proof_ineq (f g : ℤ → ℤ) (m n k : ℕ) (h : P f g m n k) : 
  2 * m ≤ n + k :=
  sorry

end proof_ineq_l95_95389


namespace totalInitialAmount_l95_95760

variable (a j t k x : ℝ)

-- Given conditions
def initialToyAmount : Prop :=
  t = 48

def kimRedistribution : Prop :=
  k = 4 * x - 144

def amyRedistribution : Prop :=
  (a = 3 * x) ∧ (j = 2 * x) ∧ (t = 2 * x)

def janRedistribution : Prop :=
  (a = 3 * x) ∧ (t = 4 * x)

def toyRedistribution : Prop :=
  (a = 6 * x) ∧ (j = -6 * x) ∧ (t = 48) 

def toyFinalAmount : Prop :=
  t = 48

-- Proof Problem
theorem totalInitialAmount
  (h1 : initialToyAmount t)
  (h2 : kimRedistribution k x)
  (h3 : amyRedistribution a j t x)
  (h4 : janRedistribution a t x)
  (h5 : toyRedistribution a j t x)
  (h6 : toyFinalAmount t) :
  a + j + t + k = 192 :=
sorry

end totalInitialAmount_l95_95760


namespace find_smallest_sphere_radius_squared_l95_95484

noncomputable def smallest_sphere_radius_squared
  (AB CD AD BC : ℝ) (angle_ABC : ℝ) (radius_AC_squared : ℝ) : ℝ :=
if AB = 6 ∧ CD = 6 ∧ AD = 10 ∧ BC = 10 ∧ angle_ABC = 120 then radius_AC_squared else 0

theorem find_smallest_sphere_radius_squared
  (AB CD AD BC : ℝ) (angle_ABC : ℝ) (radius_AC_squared : ℝ) :
  (AB = 6 ∧ CD = 6 ∧ AD = 10 ∧ BC = 10 ∧ angle_ABC = 120) →
  radius_AC_squared = 49 :=
by
  intros h
  have h_ABCD : AB = 6 ∧ CD = 6 ∧ AD = 10 ∧ BC = 10 ∧ angle_ABC = 120 := h
  sorry -- The proof steps would be filled in here

end find_smallest_sphere_radius_squared_l95_95484


namespace minimum_value_expression_l95_95678

theorem minimum_value_expression (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) :
  4 ≤ (5 * r) / (3 * p + q) + (5 * p) / (q + 3 * r) + (2 * q) / (p + r) :=
by sorry

end minimum_value_expression_l95_95678


namespace solve_for_x_l95_95233
-- Import the entire Mathlib library

-- Define the condition
def condition (x : ℝ) := (72 - x)^2 = x^2

-- State the theorem
theorem solve_for_x : ∃ x : ℝ, condition x ∧ x = 36 :=
by {
  -- The proof will be provided here
  sorry
}

end solve_for_x_l95_95233


namespace bottom_row_bricks_l95_95702

theorem bottom_row_bricks {x : ℕ} 
  (c1 : ∀ i, i < 5 → (x - i) > 0)
  (c2 : x + (x - 1) + (x - 2) + (x - 3) + (x - 4) = 100) : 
  x = 22 := 
by
  sorry

end bottom_row_bricks_l95_95702


namespace volume_of_rectangular_prism_l95_95698

-- Given conditions translated into Lean definitions
variables (AB AD AC1 AA1 : ℕ)

def rectangular_prism_properties : Prop :=
  AB = 2 ∧ AD = 2 ∧ AC1 = 3 ∧ AA1 = 1

-- The mathematical volume of the rectangular prism
def volume (AB AD AA1 : ℕ) := AB * AD * AA1

-- Prove that given the conditions, the volume of the rectangular prism is 4
theorem volume_of_rectangular_prism (h : rectangular_prism_properties AB AD AC1 AA1) : volume AB AD AA1 = 4 :=
by
  sorry

#check volume_of_rectangular_prism

end volume_of_rectangular_prism_l95_95698


namespace largest_difference_l95_95357

def A := 3 * 1005^1006
def B := 1005^1006
def C := 1004 * 1005^1005
def D := 3 * 1005^1005
def E := 1005^1005
def F := 1005^1004

theorem largest_difference : 
  A - B > B - C ∧ 
  A - B > C - D ∧ 
  A - B > D - E ∧ 
  A - B > E - F :=
by
  sorry

end largest_difference_l95_95357


namespace power_mod_residue_l95_95276

theorem power_mod_residue (n : ℕ) (h : n = 1234) : (7^n) % 19 = 9 := by
  sorry

end power_mod_residue_l95_95276


namespace inverse_proportion_inequality_l95_95990

theorem inverse_proportion_inequality :
  ∀ (y : ℝ → ℝ) (y_1 y_2 y_3 : ℝ),
  (∀ x, y x = 7 / x) →
  y (-3) = y_1 →
  y (-1) = y_2 →
  y (2) = y_3 →
  y_2 < y_1 ∧ y_1 < y_3 :=
by
  intros y y_1 y_2 y_3 hy hA hB hC
  sorry

end inverse_proportion_inequality_l95_95990


namespace smallest_integer_represented_as_AA6_and_BB8_l95_95595

def valid_digit_in_base (d : ℕ) (b : ℕ) : Prop := d < b

theorem smallest_integer_represented_as_AA6_and_BB8 :
  ∃ (n : ℕ) (A B : ℕ),
  valid_digit_in_base A 6 ∧ valid_digit_in_base B 8 ∧ 
  n = 7 * A ∧ n = 9 * B ∧ n = 63 :=
by
  sorry

end smallest_integer_represented_as_AA6_and_BB8_l95_95595


namespace inequality_proof_l95_95536

noncomputable def problem_statement (x y z : ℝ) (positive_x : 0 < x) (positive_y : 0 < y) (positive_z : 0 < z) 
  (condition : x * y * z + x * y + y * z + z * x = x + y + z + 1) : Prop :=
  (1 / 3) * (Real.sqrt ((1 + x^2) / (1 + x)) + Real.sqrt ((1 + y^2) / (1 + y)) + Real.sqrt ((1 + z^2) / (1 + z))) ≤ 
  ((x + y + z) / 3) ^ (5 / 8)

-- The statement below is what needs to be proven.
theorem inequality_proof (x y z : ℝ) (positive_x : 0 < x) (positive_y : 0 < y) (positive_z : 0 < z) 
  (condition : x * y * z + x * y + y * z + z * x = x + y + z + 1) : problem_statement x y z positive_x positive_y positive_z condition :=
sorry

end inequality_proof_l95_95536


namespace radical_product_l95_95963

def fourth_root (x : ℝ) : ℝ := x ^ (1/4)
def third_root (x : ℝ) : ℝ := x ^ (1/3)
def square_root (x : ℝ) : ℝ := x ^ (1/2)

theorem radical_product :
  fourth_root 81 * third_root 27 * square_root 9 = 27 := 
by
  sorry

end radical_product_l95_95963


namespace multiple_of_fair_tickets_l95_95223

theorem multiple_of_fair_tickets (fair_tickets_sold : ℕ) (game_tickets_sold : ℕ) (h : fair_tickets_sold = game_tickets_sold * x + 6) :
  25 = 56 * x + 6 → x = 19 / 56 := by
  sorry

end multiple_of_fair_tickets_l95_95223


namespace xy_is_perfect_cube_l95_95831

theorem xy_is_perfect_cube (x y : ℕ) (h₁ : x = 5 * 2^4 * 3^3) (h₂ : y = 2^2 * 5^2) : ∃ z : ℕ, (x * y) = z^3 :=
by
  sorry

end xy_is_perfect_cube_l95_95831


namespace length_of_plot_l95_95303

-- Definitions of the given conditions, along with the question.
def breadth (b : ℝ) : Prop := 2 * (b + 32) + 2 * b = 5300 / 26.50
def length (b : ℝ) := b + 32

theorem length_of_plot (b : ℝ) (h : breadth b) : length b = 66 := by 
  sorry

end length_of_plot_l95_95303


namespace no_odd_total_given_ratio_l95_95445

theorem no_odd_total_given_ratio (T : ℕ) (hT1 : 50 < T) (hT2 : T < 150) (hT3 : T % 2 = 1) : 
  ∀ (B : ℕ), T ≠ 8 * B + B / 4 :=
sorry

end no_odd_total_given_ratio_l95_95445


namespace problem_Z_value_l95_95437

def Z (a b : ℕ) : ℕ := 3 * (a - b) ^ 2

theorem problem_Z_value : Z 5 3 = 12 := by
  sorry

end problem_Z_value_l95_95437


namespace longer_side_is_40_l95_95675

-- Given the conditions
variable (small_rect_width : ℝ) (small_rect_length : ℝ)
variable (num_rects : ℕ)

-- Conditions 
axiom rect_width_is_10 : small_rect_width = 10
axiom length_is_twice_width : small_rect_length = 2 * small_rect_width
axiom four_rectangles : num_rects = 4

-- Prove length of the longer side of the large rectangle
theorem longer_side_is_40 :
  small_rect_width = 10 → small_rect_length = 2 * small_rect_width → num_rects = 4 →
  (2 * small_rect_length) = 40 := sorry

end longer_side_is_40_l95_95675


namespace num_valid_pairs_l95_95868

/-- 
Let S(n) denote the sum of the digits of a natural number n.
Define the predicate to check if the pair (m, n) satisfies the given conditions.
-/
def S (n : ℕ) : ℕ := (toString n).foldl (fun acc ch => acc + ch.toNat - '0'.toNat) 0

def valid_pair (m n : ℕ) : Prop :=
  m < 100 ∧ n < 100 ∧ m > n ∧ m + S n = n + 2 * S m

/-- 
Theorem: There are exactly 99 pairs (m, n) that satisfy the given conditions.
-/
theorem num_valid_pairs : ∃! (pairs : Finset (ℕ × ℕ)), pairs.card = 99 ∧
  ∀ (p : ℕ × ℕ), p ∈ pairs ↔ valid_pair p.1 p.2 :=
sorry

end num_valid_pairs_l95_95868


namespace smallest_fraction_numerator_l95_95557

theorem smallest_fraction_numerator :
  ∃ (a b : ℕ), (10 ≤ a ∧ a < 100) ∧ (10 ≤ b ∧ b < 100) ∧ (5 * b < 7 * a) ∧ 
    ∀ (a' b' : ℕ), (10 ≤ a' ∧ a' < 100) ∧ (10 ≤ b' ∧ b' < 100) ∧ (5 * b' < 7 * a') →
    (a * b' ≤ a' * b) → a = 68 :=
sorry

end smallest_fraction_numerator_l95_95557


namespace sqrt_nested_expr_l95_95679

theorem sqrt_nested_expr (x : ℝ) (hx : 0 ≤ x) : 
  (x * (x * (x * x)^(1 / 2))^(1 / 2))^(1 / 2) = (x^7)^(1 / 4) :=
sorry

end sqrt_nested_expr_l95_95679


namespace average_visitors_per_day_is_276_l95_95365

-- Define the number of days in the month
def num_days_in_month : ℕ := 30

-- Define the number of Sundays in the month
def num_sundays_in_month : ℕ := 4

-- Define the number of other days in the month
def num_other_days_in_month : ℕ := num_days_in_month - num_sundays_in_month * 7 / 7 + 2

-- Define the average visitors on Sundays
def avg_visitors_sunday : ℕ := 510

-- Define the average visitors on other days
def avg_visitors_other_days : ℕ := 240

-- Calculate total visitors on Sundays
def total_visitors_sundays : ℕ := num_sundays_in_month * avg_visitors_sunday

-- Calculate total visitors on other days
def total_visitors_other_days : ℕ := num_other_days_in_month * avg_visitors_other_days

-- Calculate total visitors in the month
def total_visitors_in_month : ℕ := total_visitors_sundays + total_visitors_other_days

-- Given conditions, prove average visitors per day in a month
theorem average_visitors_per_day_is_276 :
  total_visitors_in_month / num_days_in_month = 276 := by
  sorry

end average_visitors_per_day_is_276_l95_95365


namespace area_of_triangle_l95_95530

theorem area_of_triangle (m : ℝ) 
  (h : ∀ x y : ℝ, ((m + 3) * x + y = 3 * m - 4) → 
                  (7 * x + (5 - m) * y - 8 ≠ 0)
  ) : ((m = -2) → (1/2) * 2 * 2 = 2) := 
by {
  sorry
}

end area_of_triangle_l95_95530


namespace greatest_number_of_unit_segments_l95_95168

-- Define the conditions
def is_equilateral (n : ℕ) : Prop := n > 0

-- Define the theorem
theorem greatest_number_of_unit_segments (n : ℕ) (h : is_equilateral n) : 
  -- Prove the greatest number of unit segments such that no three of them form a single triangle
  ∃(m : ℕ), m = n * (n + 1) := 
sorry

end greatest_number_of_unit_segments_l95_95168


namespace calculate_probability_l95_95068

-- Definitions
def total_coins : ℕ := 16  -- Total coins (3 pennies + 5 nickels + 8 dimes)
def draw_coins : ℕ := 8    -- Coins drawn
def successful_outcomes : ℕ := 321  -- Number of successful outcomes
def total_outcomes : ℕ := Nat.choose total_coins draw_coins  -- Total number of ways to choose draw_coins from total_coins

-- Question statement in Lean 4: Probability of drawing coins worth at least 75 cents
theorem calculate_probability : (successful_outcomes : ℝ) / (total_outcomes : ℝ) = 321 / 12870 := by
  sorry

end calculate_probability_l95_95068


namespace empty_pipe_time_l95_95509

theorem empty_pipe_time (R1 R2 : ℚ) (t1 t2 t_total : ℕ) (h1 : t1 = 60) (h2 : t_total = 180) (H1 : R1 = 1 / t1) (H2 : R1 - R2 = 1 / t_total) :
  1 / R2 = 90 :=
by
  sorry

end empty_pipe_time_l95_95509


namespace platform_length_l95_95040

theorem platform_length (train_length : ℕ) (time_post : ℕ) (time_platform : ℕ) (speed : ℕ)
    (h1 : train_length = 150)
    (h2 : time_post = 15)
    (h3 : time_platform = 25)
    (h4 : speed = train_length / time_post)
    : (train_length + 100) / time_platform = speed :=
by
  sorry

end platform_length_l95_95040


namespace find_b_l95_95958

def h (x : ℝ) : ℝ := 5 * x + 7

theorem find_b (b : ℝ) : h b = 0 ↔ b = -7 / 5 := by
  sorry

end find_b_l95_95958


namespace barry_pretzels_l95_95889

theorem barry_pretzels (A S B : ℕ) (h1 : A = 3 * S) (h2 : S = B / 2) (h3 : A = 18) : B = 12 :=
  by
  sorry

end barry_pretzels_l95_95889


namespace exists_ints_for_inequalities_l95_95982

theorem exists_ints_for_inequalities (a b : ℝ) (ε : ℝ) (hε : ε > 0) :
  ∃ (n : ℕ) (k m : ℤ), |(n * a) - k| < ε ∧ |(n * b) - m| < ε :=
by
  sorry

end exists_ints_for_inequalities_l95_95982


namespace cape_may_vs_daytona_shark_sightings_diff_l95_95053

-- Definitions based on the conditions
def total_shark_sightings := 40
def cape_may_sightings : ℕ := 24
def daytona_beach_sightings : ℕ := total_shark_sightings - cape_may_sightings

-- The main theorem stating the problem in Lean
theorem cape_may_vs_daytona_shark_sightings_diff :
  (2 * daytona_beach_sightings - cape_may_sightings) = 8 := by
  sorry

end cape_may_vs_daytona_shark_sightings_diff_l95_95053


namespace area_of_yard_l95_95324

def length {w : ℝ} : ℝ := 2 * w + 30

def perimeter {w l : ℝ} (cond_len : l = 2 * w + 30) : Prop := 2 * w + 2 * l = 700

theorem area_of_yard {w l A : ℝ} 
  (cond_len : l = 2 * w + 30) 
  (cond_perim : 2 * w + 2 * l = 700) : 
  A = w * l := 
  sorry

end area_of_yard_l95_95324


namespace value_2_stddev_less_than_mean_l95_95181

theorem value_2_stddev_less_than_mean :
  let mean := 17.5
  let stddev := 2.5
  mean - 2 * stddev = 12.5 :=
by
  sorry

end value_2_stddev_less_than_mean_l95_95181


namespace shoe_size_percentage_difference_l95_95410

theorem shoe_size_percentage_difference :
  ∀ (size8_len size15_len size17_len : ℝ)
  (h1 : size8_len = size15_len - (7 * (1 / 5)))
  (h2 : size17_len = size15_len + (2 * (1 / 5)))
  (h3 : size15_len = 10.4),
  ((size17_len - size8_len) / size8_len) * 100 = 20 := by
  intros size8_len size15_len size17_len h1 h2 h3
  sorry

end shoe_size_percentage_difference_l95_95410


namespace top_angle_degrees_l95_95408

def isosceles_triangle_with_angle_ratio (x : ℝ) (a b c : ℝ) : Prop :=
  a = x ∧ b = 4 * x ∧ a + b + c = 180 ∧ (a = b ∨ a = c ∨ b = c)

theorem top_angle_degrees (x : ℝ) (a b c : ℝ) :
  isosceles_triangle_with_angle_ratio x a b c → c = 20 ∨ c = 120 :=
by
  sorry

end top_angle_degrees_l95_95408


namespace probability_of_sum_greater_than_15_l95_95880

-- Definition of the dice and outcomes
def total_outcomes : ℕ := 6 * 6 * 6
def favorable_outcomes : ℕ := 10

-- Probability calculation
def probability_sum_gt_15 : ℚ := favorable_outcomes / total_outcomes

-- Theorem to be proven
theorem probability_of_sum_greater_than_15 : probability_sum_gt_15 = 5 / 108 := by
  sorry

end probability_of_sum_greater_than_15_l95_95880


namespace smallest_value_div_by_13_l95_95148

theorem smallest_value_div_by_13 : 
  ∃ (A B : ℕ), 
    (0 ≤ A ∧ A ≤ 9) ∧ (0 ≤ B ∧ B ≤ 9) ∧ 
    A ≠ B ∧ 
    1001 * A + 110 * B = 1771 ∧ 
    (1001 * A + 110 * B) % 13 = 0 :=
by
  sorry

end smallest_value_div_by_13_l95_95148


namespace xiao_wang_parts_processed_l95_95127

-- Definitions for the processing rates and conditions
def xiao_wang_rate := 15 -- parts per hour
def xiao_wang_max_continuous_hours := 2
def xiao_wang_break_hours := 1

def xiao_li_rate := 12 -- parts per hour

-- Constants for the problem setup
def xiao_wang_process_time := 4 -- hours including breaks after first cycle
def xiao_li_process_time := 5 -- hours including no breaks

-- Total parts processed by both when they finish simultaneously
def parts_processed_when_finished_simultaneously := 60

theorem xiao_wang_parts_processed :
  (xiao_wang_rate * xiao_wang_max_continuous_hours) * (xiao_wang_process_time / 
  (xiao_wang_max_continuous_hours + xiao_wang_break_hours)) =
  parts_processed_when_finished_simultaneously :=
sorry

end xiao_wang_parts_processed_l95_95127


namespace sum_of_edges_l95_95614

theorem sum_of_edges (a r : ℝ) 
  (h_volume : (a^3 = 512))
  (h_surface_area : (2 * (a^2 / r + a^2 + a^2 * r) = 384))
  (h_geometric_progression : true) :
  (4 * ((a / r) + a + (a * r)) = 96) :=
by
  -- It is only necessary to provide the theorem statement
  sorry

end sum_of_edges_l95_95614


namespace triangle_side_AC_l95_95030

theorem triangle_side_AC (B : Real) (BC AB : Real) (AC : Real) (hB : B = 30 * Real.pi / 180) (hBC : BC = 2) (hAB : AB = Real.sqrt 3) : AC = 1 :=
by
  sorry

end triangle_side_AC_l95_95030


namespace smallest_number_l95_95749

theorem smallest_number (n : ℕ) :
  (n % 3 = 1) ∧
  (n % 5 = 3) ∧
  (n % 6 = 4) →
  n = 28 :=
sorry

end smallest_number_l95_95749


namespace number_of_movies_in_series_l95_95593

variables (watched_movies remaining_movies total_movies : ℕ)

theorem number_of_movies_in_series 
  (h_watched : watched_movies = 4) 
  (h_remaining : remaining_movies = 4) :
  total_movies = watched_movies + remaining_movies :=
by
  sorry

end number_of_movies_in_series_l95_95593


namespace volunteers_meet_again_in_360_days_l95_95640

theorem volunteers_meet_again_in_360_days :
  let Sasha := 5
  let Leo := 8
  let Uma := 9
  let Kim := 10
  Nat.lcm Sasha (Nat.lcm Leo (Nat.lcm Uma Kim)) = 360 :=
by
  sorry

end volunteers_meet_again_in_360_days_l95_95640


namespace heidi_zoe_paint_wall_l95_95497

theorem heidi_zoe_paint_wall :
  let heidi_rate := (1 : ℚ) / 60
  let zoe_rate := (1 : ℚ) / 90
  let combined_rate := heidi_rate + zoe_rate
  let painted_fraction_15_minutes := combined_rate * 15
  painted_fraction_15_minutes = (5 : ℚ) / 12 :=
by
  sorry

end heidi_zoe_paint_wall_l95_95497


namespace abs_eq_neg_imp_nonpos_l95_95540

theorem abs_eq_neg_imp_nonpos (a : ℝ) (h : |a| = -a) : a ≤ 0 :=
sorry

end abs_eq_neg_imp_nonpos_l95_95540


namespace power_function_through_point_l95_95367

-- Define the condition that the power function passes through the point (2, 8)
theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) (h : ∀ x, f x = x^α) (h₂ : f 2 = 8) :
  α = 3 ∧ ∀ x, f x = x^3 :=
by
  -- Proof will be provided here
  sorry

end power_function_through_point_l95_95367


namespace Randy_trip_distance_l95_95015

theorem Randy_trip_distance (x : ℝ) (h1 : x = 4 * (x / 4 + 30 + x / 6)) : x = 360 / 7 :=
by
  have h2 : x = ((3 * x + 36 * 30 + 2 * x) / 12) := sorry
  have h3 : x = (5 * x / 12 + 30) := sorry
  have h4 : 30 = x - (5 * x / 12) := sorry
  have h5 : 30 = 7 * x / 12 := sorry
  have h6 : x = (12 * 30) / 7 := sorry
  have h7 : x = 360 / 7 := sorry
  exact h7

end Randy_trip_distance_l95_95015


namespace sin_2pi_minus_alpha_l95_95894

noncomputable def alpha_condition (α : ℝ) : Prop :=
  (3 * Real.pi / 2 < α) ∧ (α < 2 * Real.pi) ∧ (Real.cos (Real.pi + α) = -1 / 2)

theorem sin_2pi_minus_alpha (α : ℝ) (h : alpha_condition α) : Real.sin (2 * Real.pi - α) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_2pi_minus_alpha_l95_95894


namespace problem_solution_l95_95672

def eq_A (x : ℝ) : Prop := 2 * x = 7
def eq_B (x y : ℝ) : Prop := x^2 + y = 5
def eq_C (x : ℝ) : Prop := x = 1 / x + 1
def eq_D (x : ℝ) : Prop := x^2 + x = 4

def is_quadratic (eq : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x : ℝ, eq x ↔ a * x^2 + b * x + c = 0

theorem problem_solution : is_quadratic eq_D := by
  sorry

end problem_solution_l95_95672


namespace woman_waits_for_man_l95_95291

noncomputable def man_speed := 5 / 60 -- miles per minute
noncomputable def woman_speed := 15 / 60 -- miles per minute
noncomputable def passed_time := 2 -- minutes

noncomputable def catch_up_time (man_speed woman_speed : ℝ) (passed_time : ℝ) : ℝ :=
  (woman_speed * passed_time) / man_speed

theorem woman_waits_for_man
  (man_speed woman_speed : ℝ)
  (passed_time : ℝ)
  (h_man_speed : man_speed = 5 / 60)
  (h_woman_speed : woman_speed = 15 / 60)
  (h_passed_time : passed_time = 2) :
  catch_up_time man_speed woman_speed passed_time = 6 := 
by
  -- actual proof skipped
  sorry

end woman_waits_for_man_l95_95291


namespace game_show_possible_guesses_l95_95343

theorem game_show_possible_guesses : 
  (∃ A B C : ℕ, 
    A + B + C = 8 ∧ 
    A > 0 ∧ B > 0 ∧ C > 0 ∧ 
    (A = 1 ∨ A = 4) ∧
    (B = 1 ∨ B = 4) ∧
    (C = 1 ∨ C = 4) ) →
  (number_of_possible_guesses : ℕ) = 210 :=
sorry

end game_show_possible_guesses_l95_95343


namespace coloring_scheme_count_l95_95919

/-- Given the set of points in the Cartesian plane, where each point (m, n) with
    1 <= m, n <= 6 is colored either red or blue, the number of ways to color these points
    such that each unit square has exactly two red vertices is 126. -/
theorem coloring_scheme_count 
  (color : Fin 6 → Fin 6 → Bool)
  (colored_correctly : ∀ m n, (1 ≤ m ∧ m ≤ 6) ∧ (1 ≤ n ∧ n ≤ 6) ∧ 
    (color m n = true ∨ color m n = false) :=
    sorry
  )
  : (∃ valid_coloring : Nat, valid_coloring = 126) :=
  sorry

end coloring_scheme_count_l95_95919


namespace prove_triangular_cake_volume_surface_area_sum_l95_95525

def triangular_cake_volume_surface_area_sum_proof : Prop :=
  let length : ℝ := 3
  let width : ℝ := 2
  let height : ℝ := 2
  let base_area : ℝ := (1 / 2) * length * width
  let volume : ℝ := base_area * height
  let top_area : ℝ := base_area
  let side_area : ℝ := (1 / 2) * width * height
  let icing_area : ℝ := top_area + 3 * side_area
  volume + icing_area = 15

theorem prove_triangular_cake_volume_surface_area_sum : triangular_cake_volume_surface_area_sum_proof := by
  sorry

end prove_triangular_cake_volume_surface_area_sum_l95_95525


namespace parallelogram_base_length_l95_95036

theorem parallelogram_base_length (b h : ℝ) (area : ℝ) (angle : ℝ) (h_area : area = 200) 
(h_altitude : h = 2 * b) (h_angle : angle = 60) : b = 10 :=
by
  -- Placeholder for proof
  sorry

end parallelogram_base_length_l95_95036


namespace anticipated_sedans_l95_95913

theorem anticipated_sedans (sales_sports_cars sedans_ratio sports_ratio sports_forecast : ℕ) 
  (h_ratio : sports_ratio = 5) (h_sedans_ratio : sedans_ratio = 8) (h_sports_forecast : sports_forecast = 35)
  (h_eq : sales_sports_cars = sports_ratio * sports_forecast) :
  sales_sports_cars * 8 / 5 = 56 :=
by
  sorry

end anticipated_sedans_l95_95913


namespace problem_proof_l95_95202

variables (a b : ℝ) (n : ℕ)

theorem problem_proof (h1: a > 0) (h2: b > 0) (h3: a + b = 1) (h4: n >= 2) :
  3/2 < 1/(a^n + 1) + 1/(b^n + 1) ∧ 1/(a^n + 1) + 1/(b^n + 1) ≤ (2^(n+1))/(2^n + 1) := sorry

end problem_proof_l95_95202


namespace german_mo_2016_problem_1_l95_95821

theorem german_mo_2016_problem_1 (a b : ℝ) :
  a^2 + b^2 = 25 ∧ 3 * (a + b) - a * b = 15 ↔
  (a = 0 ∧ b = 5) ∨ (a = 5 ∧ b = 0) ∨
  (a = 4 ∧ b = -3) ∨ (a = -3 ∧ b = 4) :=
sorry

end german_mo_2016_problem_1_l95_95821


namespace find_x_plus_inv_x_l95_95852

theorem find_x_plus_inv_x (x : ℝ) (hx : x^3 + 1 / x^3 = 110) : x + 1 / x = 5 := 
by 
  sorry

end find_x_plus_inv_x_l95_95852


namespace cricket_team_members_l95_95775

theorem cricket_team_members (n : ℕ)
    (captain_age : ℕ) (wicket_keeper_age : ℕ) (average_age : ℕ)
    (remaining_average_age : ℕ) (total_age : ℕ) (remaining_players : ℕ) :
    captain_age = 27 →
    wicket_keeper_age = captain_age + 3 →
    average_age = 24 →
    remaining_average_age = average_age - 1 →
    total_age = average_age * n →
    remaining_players = n - 2 →
    total_age = captain_age + wicket_keeper_age + remaining_average_age * remaining_players →
    n = 11 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end cricket_team_members_l95_95775


namespace new_quadratic_eq_l95_95196

def quadratic_roots_eq (a b c : ℝ) (x1 x2 : ℝ) : Prop :=
  a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0

theorem new_quadratic_eq
  (a b c : ℝ) (x1 x2 : ℝ)
  (h1 : quadratic_roots_eq a b c x1 x2)
  (h_sum : x1 + x2 = -b / a)
  (h_prod : x1 * x2 = c / a) :
  a^3 * x^2 - a * b^2 * x + 2 * c * (b^2 - 2 * a * c) = 0 :=
sorry

end new_quadratic_eq_l95_95196


namespace carl_took_4_pink_hard_hats_l95_95308

-- Define the initial number of hard hats
def initial_pink : ℕ := 26
def initial_green : ℕ := 15
def initial_yellow : ℕ := 24

-- Define the number of hard hats John took
def john_pink : ℕ := 6
def john_green : ℕ := 2 * john_pink
def john_total : ℕ := john_pink + john_green

-- Define the total initial number of hard hats
def total_initial : ℕ := initial_pink + initial_green + initial_yellow

-- Define the number of hard hats remaining after John's removal
def remaining_after_john : ℕ := total_initial - john_total

-- Define the total number of hard hats that remained in the truck
def total_remaining : ℕ := 43

-- Define the number of pink hard hats Carl took away
def carl_pink : ℕ := remaining_after_john - total_remaining

-- State the proof problem
theorem carl_took_4_pink_hard_hats : carl_pink = 4 := by
  sorry

end carl_took_4_pink_hard_hats_l95_95308


namespace hyperbola_asymptotes_angle_l95_95924

-- Define the given conditions and the proof problem
theorem hyperbola_asymptotes_angle (a b c : ℝ) (e : ℝ) (h1 : e = 2) 
  (h2 : e = c / a) (h3 : c = 2 * a) (h4 : b^2 + a^2 = c^2) : 
  ∃ θ : ℝ, θ = 60 :=
by 
  sorry -- Proof is omitted

end hyperbola_asymptotes_angle_l95_95924


namespace part_I_part_II_l95_95371

open Real

noncomputable def f (x : ℝ) : ℝ := log ((2 / (x + 1)) - 1)

def g (x a : ℝ) : ℝ := -x^2 + 2 * x + a

-- Domain of function f
def A : Set ℝ := {x | -1 < x ∧ x < 1}

-- Range of function g with a given condition on x
def B (a : ℝ) : Set ℝ := {y | ∃ x, 0 ≤ x ∧ x ≤ 3 ∧ y = g x a}

theorem part_I : f (1 / 2015) + f (-1 / 2015) = 0 := sorry

theorem part_II (a : ℝ) : (A ∩ B a) = ∅ ↔ a ≤ -2 ∨ a ≥ 4 := sorry

end part_I_part_II_l95_95371


namespace smallest_denominator_between_l95_95713

theorem smallest_denominator_between :
  ∃ (a b : ℕ), b > 0 ∧ a < b ∧ 6 / 17 < (a : ℚ) / b ∧ (a : ℚ) / b < 9 / 25 ∧ (∀ (c d : ℕ), d > 0 → c < d → 6 / 17 < (c : ℚ) / d → (c : ℚ) / d < 9 / 25 → b ≤ d) ∧ a = 5 ∧ b = 14 :=
by
  existsi 5
  existsi 14
  sorry

end smallest_denominator_between_l95_95713


namespace gcd_8p_18q_l95_95134

theorem gcd_8p_18q (p q : ℕ) (hp : p > 0) (hq : q > 0) (hg : Nat.gcd p q = 9) : Nat.gcd (8 * p) (18 * q) = 18 := 
sorry

end gcd_8p_18q_l95_95134


namespace distance_between_lines_l95_95152

/-- Define the lines by their equations -/
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0
def line2 (x y : ℝ) : Prop := 6 * x + 8 * y + 6 = 0

/-- Define the simplified form of the second line -/
def simplified_line2 (x y : ℝ) : Prop := 3 * x + 4 * y + 3 = 0

/-- Prove the distance between the two lines is 3 -/
theorem distance_between_lines : 
  let A : ℝ := 3
  let B : ℝ := 4
  let C1 : ℝ := -12
  let C2 : ℝ := 3
  (|C2 - C1| / Real.sqrt (A^2 + B^2) = 3) :=
by
  sorry

end distance_between_lines_l95_95152


namespace keats_library_percentage_increase_l95_95071

theorem keats_library_percentage_increase :
  let total_books_A := 8000
  let total_books_B := 10000
  let total_books_C := 12000
  let initial_bio_A := 0.20 * total_books_A
  let initial_bio_B := 0.25 * total_books_B
  let initial_bio_C := 0.28 * total_books_C
  let total_initial_bio := initial_bio_A + initial_bio_B + initial_bio_C
  let final_bio_A := 0.32 * total_books_A
  let final_bio_B := 0.35 * total_books_B
  let final_bio_C := 0.40 * total_books_C
  --
  let total_final_bio := final_bio_A + final_bio_B + final_bio_C
  let increase_in_bio := total_final_bio - total_initial_bio
  let percentage_increase := (increase_in_bio / total_initial_bio) * 100
  --
  percentage_increase = 45.58 := 
by
  sorry

end keats_library_percentage_increase_l95_95071


namespace shop_sold_price_l95_95075

noncomputable def clock_selling_price (C : ℝ) : ℝ :=
  let buy_back_price := 0.60 * C
  let maintenance_cost := 0.10 * buy_back_price
  let total_spent := buy_back_price + maintenance_cost
  let selling_price := 1.80 * total_spent
  selling_price

theorem shop_sold_price (C : ℝ) (h1 : C - 0.60 * C = 100) :
  clock_selling_price C = 297 := by
  sorry

end shop_sold_price_l95_95075


namespace find_constants_l95_95050

theorem find_constants (a b : ℝ) (h₀ : ∀ x : ℝ, (x^3 + 3*a*x^2 + b*x + a^2 = 0 → x = -1)) :
    a = 2 ∧ b = 9 :=
by
  sorry

end find_constants_l95_95050


namespace find_a_l95_95501

noncomputable def a := 1/2

theorem find_a (a : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : 1 - a^2 = 3/4) : a = 1/2 :=
sorry

end find_a_l95_95501


namespace profit_percent_is_25_l95_95743

-- Define the cost price (CP) and selling price (SP) based on the given ratio.
def CP (x : ℝ) := 4 * x
def SP (x : ℝ) := 5 * x

-- Calculate the profit percent based on the given conditions.
noncomputable def profitPercent (x : ℝ) := ((SP x - CP x) / CP x) * 100

-- Prove that the profit percent is 25% given the ratio of CP to SP is 4:5.
theorem profit_percent_is_25 (x : ℝ) : profitPercent x = 25 := by
  sorry

end profit_percent_is_25_l95_95743


namespace davonte_ran_further_than_mercedes_l95_95379

-- Conditions
variable (jonathan_distance : ℝ) (mercedes_distance : ℝ) (davonte_distance : ℝ)

-- Given conditions
def jonathan_ran := jonathan_distance = 7.5
def mercedes_ran_twice_jonathan := mercedes_distance = 2 * jonathan_distance
def mercedes_and_davonte_total := mercedes_distance + davonte_distance = 32

-- Prove the distance Davonte ran farther than Mercedes is 2 kilometers
theorem davonte_ran_further_than_mercedes :
  jonathan_ran jonathan_distance ∧
  mercedes_ran_twice_jonathan jonathan_distance mercedes_distance ∧
  mercedes_and_davonte_total mercedes_distance davonte_distance →
  davonte_distance - mercedes_distance = 2 :=
by
  sorry

end davonte_ran_further_than_mercedes_l95_95379


namespace length_of_each_stone_l95_95226

theorem length_of_each_stone {L : ℝ} (hall_length hall_breadth : ℝ) (stone_breadth : ℝ) (num_stones : ℕ) (area_hall : ℝ) (area_stone : ℝ) :
  hall_length = 36 * 10 ∧ hall_breadth = 15 * 10 ∧ stone_breadth = 5 ∧ num_stones = 3600 ∧
  area_hall = hall_length * hall_breadth ∧ area_stone = L * stone_breadth ∧
  area_stone * num_stones = area_hall →
  L = 3 :=
by
  sorry

end length_of_each_stone_l95_95226


namespace find_p_q_sum_l95_95956

noncomputable def roots (r1 r2 r3 : ℝ) := (r1 + r2 + r3 = 11 ∧ r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3) ∧ 
                                         (∀ x : ℝ, x^3 - 11*x^2 + (r1 * r2 + r2 * r3 + r3 * r1) * x - r1 * r2 * r3 = 0)

theorem find_p_q_sum : ∃ (p q : ℝ), roots 2 4 5 → p + q = 78 :=
by
  sorry

end find_p_q_sum_l95_95956


namespace bird_families_flew_away_l95_95011

theorem bird_families_flew_away (original : ℕ) (left : ℕ) (flew_away : ℕ) (h1 : original = 67) (h2 : left = 35) (h3 : flew_away = original - left) : flew_away = 32 :=
by
  rw [h1, h2] at h3
  exact h3

end bird_families_flew_away_l95_95011


namespace tom_and_mary_age_l95_95049

-- Define Tom's and Mary's ages
variables (T M : ℕ)

-- Define the two given conditions
def condition1 : Prop := T^2 + M = 62
def condition2 : Prop := M^2 + T = 176

-- State the theorem
theorem tom_and_mary_age (h1 : condition1 T M) (h2 : condition2 T M) : T = 7 ∧ M = 13 :=
by {
  -- sorry acts as a placeholder for the proof
  sorry
}

end tom_and_mary_age_l95_95049


namespace perimeter_of_square_C_l95_95062

theorem perimeter_of_square_C (s_A s_B s_C : ℕ) (hpA : 4 * s_A = 16) (hpB : 4 * s_B = 32) (hC : s_C = s_A + s_B - 2) :
  4 * s_C = 40 := 
by
  sorry

end perimeter_of_square_C_l95_95062


namespace sqrt_prime_geometric_progression_impossible_l95_95282

theorem sqrt_prime_geometric_progression_impossible {p1 p2 p3 : ℕ} (hp1 : Nat.Prime p1) (hp2 : Nat.Prime p2) (hp3 : Nat.Prime p3) (hneq12 : p1 ≠ p2) (hneq23 : p2 ≠ p3) (hneq31 : p3 ≠ p1) :
  ¬ ∃ (a r : ℝ) (n1 n2 n3 : ℤ), (a * r^n1 = Real.sqrt p1) ∧ (a * r^n2 = Real.sqrt p2) ∧ (a * r^n3 = Real.sqrt p3) := sorry

end sqrt_prime_geometric_progression_impossible_l95_95282


namespace x_gt_neg2_is_necessary_for_prod_lt_0_l95_95523

theorem x_gt_neg2_is_necessary_for_prod_lt_0 (x : Real) :
  (x > -2) ↔ (((x + 2) * (x - 3)) < 0) → (x > -2) :=
by
  sorry

end x_gt_neg2_is_necessary_for_prod_lt_0_l95_95523


namespace slices_eaten_l95_95441

theorem slices_eaten (slices_cheese : ℕ) (slices_pepperoni : ℕ) (slices_left_per_person : ℕ) (phil_andre_slices_left : ℕ) :
  (slices_cheese + slices_pepperoni = 22) →
  (slices_left_per_person = 2) →
  (phil_andre_slices_left = 2 + 2) →
  (slices_cheese + slices_pepperoni - phil_andre_slices_left = 18) :=
by
  intros
  sorry

end slices_eaten_l95_95441


namespace find_product_stu_l95_95866

-- Define hypotheses
variables (a x y c : ℕ)
variables (s t u : ℕ)
variable (h_eq : a^8 * x * y - a^7 * x - a^6 * y = a^5 * (c^5 - 2))

-- Statement to prove the equivalent form and stu product
theorem find_product_stu (h_eq : a^8 * x * y - a^7 * x - a^6 * y = a^5 * (c^5 - 2)) :
  ∃ s t u : ℕ, (a^s * x - a^t) * (a^u * y - a^3) = a^5 * c^5 ∧ s * t * u = 12 :=
sorry

end find_product_stu_l95_95866


namespace frosting_cupcakes_l95_95453

noncomputable def Cagney_rate := 1 / 20 -- cupcakes per second
noncomputable def Lacey_rate := 1 / 30 -- cupcakes per second
noncomputable def Hardy_rate := 1 / 40 -- cupcakes per second

noncomputable def combined_rate := Cagney_rate + Lacey_rate + Hardy_rate
noncomputable def total_time := 600 -- seconds (10 minutes)

theorem frosting_cupcakes :
  total_time * combined_rate = 65 := 
by 
  sorry

end frosting_cupcakes_l95_95453


namespace tan_beta_eq_neg13_l95_95286

variables (α β : Real)

theorem tan_beta_eq_neg13 (h1 : Real.tan α = 2) (h2 : Real.tan (α - β) = -3/5) : 
  Real.tan β = -13 := 
by 
  sorry

end tan_beta_eq_neg13_l95_95286


namespace power_comparison_l95_95185

theorem power_comparison : (9^20 : ℝ) < (9999^10 : ℝ) :=
sorry

end power_comparison_l95_95185


namespace wall_paint_area_l95_95090

theorem wall_paint_area
  (A₁ : ℕ) (A₂ : ℕ) (A₃ : ℕ) (A₄ : ℕ)
  (H₁ : A₁ = 32)
  (H₂ : A₂ = 48)
  (H₃ : A₃ = 32)
  (H₄ : A₄ = 48) :
  A₁ + A₂ + A₃ + A₄ = 160 :=
by
  sorry

end wall_paint_area_l95_95090


namespace students_with_both_l95_95753

/-- There are 28 students in a class -/
def total_students : ℕ := 28

/-- Number of students with a cat -/
def students_with_cat : ℕ := 17

/-- Number of students with a dog -/
def students_with_dog : ℕ := 10

/-- Number of students with neither a cat nor a dog -/
def students_with_neither : ℕ := 5

/-- Number of students having both a cat and a dog -/
theorem students_with_both :
  students_with_cat + students_with_dog - (total_students - students_with_neither) = 4 :=
sorry

end students_with_both_l95_95753


namespace problem1_l95_95651

theorem problem1 :
  let total_products := 10
  let defective_products := 4
  let first_def_pos := 5
  let last_def_pos := 10
  ∃ (num_methods : Nat), num_methods = 103680 :=
by
  sorry

end problem1_l95_95651


namespace fraction_value_l95_95287

theorem fraction_value : (1 - 1 / 4) / (1 - 1 / 5) = 15 / 16 := sorry

end fraction_value_l95_95287


namespace tan_of_alpha_intersects_unit_circle_l95_95328

theorem tan_of_alpha_intersects_unit_circle (α : ℝ) (hα : ∃ P : ℝ × ℝ, P = (12 / 13, -5 / 13) ∧ ∀ x y : ℝ, P = (x, y) → x^2 + y^2 = 1) : 
  Real.tan α = -5 / 12 :=
by
  -- proof to be completed
  sorry

end tan_of_alpha_intersects_unit_circle_l95_95328


namespace total_cost_fencing_l95_95734

-- Define the conditions
def length : ℝ := 75
def breadth : ℝ := 25
def cost_per_meter : ℝ := 26.50

-- Define the perimeter of the rectangular plot
def perimeter : ℝ := 2 * length + 2 * breadth

-- Define the total cost of fencing
def total_cost : ℝ := perimeter * cost_per_meter

-- The theorem statement
theorem total_cost_fencing : total_cost = 5300 := 
by 
  -- This is the statement we want to prove
  sorry

end total_cost_fencing_l95_95734


namespace lauri_ate_days_l95_95199

theorem lauri_ate_days
    (simone_rate : ℚ)
    (simone_days : ℕ)
    (lauri_rate : ℚ)
    (total_apples : ℚ)
    (simone_apples : ℚ)
    (lauri_apples : ℚ)
    (lauri_days : ℚ) :
  simone_rate = 1/2 → 
  simone_days = 16 →
  lauri_rate = 1/3 →
  total_apples = 13 →
  simone_apples = simone_rate * simone_days →
  lauri_apples = total_apples - simone_apples →
  lauri_days = lauri_apples / lauri_rate →
  lauri_days = 15 :=
by
  intros
  sorry

end lauri_ate_days_l95_95199


namespace time_increases_with_water_speed_increase_l95_95065

variable (S : ℝ) -- Total distance
variable (V : ℝ) -- Speed of the ferry in still water
variable (V1 V2 : ℝ) -- Speed of the water flow before and after increase

-- Ensure realistic conditions
axiom V_pos : 0 < V
axiom V1_pos : 0 < V1
axiom V2_pos : 0 < V2
axiom V1_less_V : V1 < V
axiom V2_less_V : V2 < V
axiom V1_less_V2 : V1 < V2

theorem time_increases_with_water_speed_increase :
  (S / (V + V1) + S / (V - V1)) < (S / (V + V2) + S / (V - V2)) :=
sorry

end time_increases_with_water_speed_increase_l95_95065


namespace largest_common_term_in_range_l95_95239

theorem largest_common_term_in_range :
  ∃ (a : ℕ), a < 150 ∧ (∃ (n : ℕ), a = 3 + 8 * n) ∧ (∃ (n : ℕ), a = 5 + 9 * n) ∧ a = 131 :=
by
  sorry

end largest_common_term_in_range_l95_95239


namespace sufficient_but_not_necessary_condition_for_monotonicity_l95_95132

theorem sufficient_but_not_necessary_condition_for_monotonicity
  (a : ℕ → ℝ)
  (h_recurrence : ∀ n : ℕ, n > 0 → a (n + 1) = a n ^ 2)
  (h_initial : a 1 = 2) :
  (∀ n : ℕ, n > 0 → a n > a 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_monotonicity_l95_95132


namespace triangle_BD_length_l95_95876

noncomputable def triangle_length_BD : ℝ :=
  let AB := 45
  let AC := 60
  let BC := Real.sqrt (AB^2 + AC^2)
  let area := (1 / 2) * AB * AC
  let AD := (2 * area) / BC
  let BD := Real.sqrt (BC^2 - AD^2)
  BD

theorem triangle_BD_length : triangle_length_BD = 63 :=
by
  -- Definitions and assumptions
  let AB := 45
  let AC := 60
  let BC := Real.sqrt (AB^2 + AC^2)
  let area := (1 / 2) * AB * AC
  let AD := (2 * area) / BC
  let BD := Real.sqrt (BC^2 - AD^2)

  -- Formal proof logic corresponding to solution steps
  sorry

end triangle_BD_length_l95_95876


namespace find_f1_find_fx_find_largest_m_l95_95047

noncomputable def f (a b c : ℝ) (x : ℝ) := a * x ^ 2 + b * x + c

axiom min_value_eq_zero (a b c : ℝ) : ∀ x : ℝ, f a b c x ≥ 0 ∨ f a b c x ≤ 0
axiom symmetry_condition (a b c : ℝ) : ∀ x : ℝ, f a b c (x - 1) = f a b c (-x - 1)
axiom inequality_condition (a b c : ℝ) : ∀ x : ℝ, 0 < x ∧ x < 5 → x ≤ f a b c x ∧ f a b c x ≤ 2 * |x - 1| + 1

theorem find_f1 (a b c : ℝ) : f a b c 1 = 1 := sorry

theorem find_fx (a b c : ℝ) : ∀ x : ℝ, f a b c x = (1 / 4) * (x + 1) ^ 2 := sorry

theorem find_largest_m (a b c : ℝ) : ∃ m : ℝ, m > 1 ∧ ∀ t x : ℝ, 1 ≤ x ∧ x ≤ m → f a b c (x + t) ≤ x := sorry

end find_f1_find_fx_find_largest_m_l95_95047


namespace jean_total_cost_l95_95687

theorem jean_total_cost 
  (num_pants : ℕ)
  (original_price_per_pant : ℝ)
  (discount_rate : ℝ)
  (tax_rate : ℝ)
  (num_pants_eq : num_pants = 10)
  (original_price_per_pant_eq : original_price_per_pant = 45)
  (discount_rate_eq : discount_rate = 0.2)
  (tax_rate_eq : tax_rate = 0.1) : 
  ∃ total_cost : ℝ, total_cost = 396 :=
by
  sorry

end jean_total_cost_l95_95687


namespace train_length_l95_95601

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (h_speed : speed_kmph = 36) (h_time : time_sec = 6.5) : 
  (speed_kmph * 1000 / 3600) * time_sec = 65 := 
by {
  -- Placeholder for proof
  sorry
}

end train_length_l95_95601


namespace Mia_studied_fraction_l95_95161

-- Define the conditions
def total_minutes_per_day := 1440
def time_spent_watching_TV := total_minutes_per_day * 1 / 5
def time_spent_studying := 288
def remaining_time := total_minutes_per_day - time_spent_watching_TV
def fraction_studying := time_spent_studying / remaining_time

-- State the proof goal
theorem Mia_studied_fraction : fraction_studying = 1 / 4 := by
  sorry

end Mia_studied_fraction_l95_95161


namespace find_principal_amount_l95_95594

theorem find_principal_amount
  (P r : ℝ) -- P for Principal amount, r for interest rate
  (simple_interest : 800 = P * r / 100 * 2) -- Condition 1: Simple Interest Formula
  (compound_interest : 820 = P * ((1 + r / 100) ^ 2 - 1)) -- Condition 2: Compound Interest Formula
  : P = 8000 := 
sorry

end find_principal_amount_l95_95594


namespace ned_initial_lives_l95_95344

-- Define the initial number of lives Ned had
def initial_lives (start_lives current_lives lost_lives : ℕ) : ℕ :=
  current_lives + lost_lives

-- Define the conditions
def current_lives := 70
def lost_lives := 13

-- State the theorem
theorem ned_initial_lives : initial_lives current_lives current_lives lost_lives = 83 := by
  sorry

end ned_initial_lives_l95_95344


namespace inequality_iff_positive_l95_95548

variable (x y : ℝ)

theorem inequality_iff_positive :
  x + y > abs (x - y) ↔ x > 0 ∧ y > 0 :=
sorry

end inequality_iff_positive_l95_95548


namespace ratio_black_bears_to_white_bears_l95_95000

theorem ratio_black_bears_to_white_bears
  (B W Br : ℕ)
  (hB : B = 60)
  (hBr : Br = B + 40)
  (h_total : B + W + Br = 190) :
  B / W = 2 :=
by
  sorry

end ratio_black_bears_to_white_bears_l95_95000


namespace difference_is_20_l95_95415

def x : ℕ := 10

def a : ℕ := 3 * x

def b : ℕ := 20 - x

theorem difference_is_20 : a - b = 20 := 
by 
  sorry

end difference_is_20_l95_95415


namespace min_value_l95_95443

theorem min_value (a b c x y z : ℝ) (h1 : a + b + c = 1) (h2 : x + y + z = 1) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  ∃ val : ℝ, val = -1 / 4 ∧ ∀ a b c x y z : ℝ, 0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ x → 0 ≤ y → 0 ≤ z → a + b + c = 1 → x + y + z = 1 → (a - x^2) * (b - y^2) * (c - z^2) ≥ val :=
sorry

end min_value_l95_95443


namespace sin_cos_ratio_value_sin_cos_expression_value_l95_95519

variable (α : ℝ)

-- Given condition
def tan_alpha_eq_3 := Real.tan α = 3

-- Goal (1)
theorem sin_cos_ratio_value 
  (h : tan_alpha_eq_3 α) : 
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 4 / 5 := 
  sorry

-- Goal (2)
theorem sin_cos_expression_value
  (h : tan_alpha_eq_3 α) : 
  Real.sin α ^ 2 + Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 15 := 
  sorry

end sin_cos_ratio_value_sin_cos_expression_value_l95_95519


namespace missed_interior_angle_l95_95741

  theorem missed_interior_angle (n : ℕ) (x : ℝ) 
    (h1 : (n - 2) * 180 = 2750 + x) : x = 130 := 
  by sorry
  
end missed_interior_angle_l95_95741


namespace bug_travel_distance_half_l95_95073

-- Define the conditions
def isHexagonalGrid (side_length : ℝ) : Prop :=
  side_length = 1

def shortest_path_length (path_length : ℝ) : Prop :=
  path_length = 100

-- Define a theorem that encapsulates the problem statement
theorem bug_travel_distance_half (side_length path_length : ℝ)
  (H1 : isHexagonalGrid side_length)
  (H2 : shortest_path_length path_length) :
  ∃ one_direction_distance : ℝ, one_direction_distance = path_length / 2 :=
sorry -- Proof to be provided.

end bug_travel_distance_half_l95_95073


namespace wickets_before_last_match_l95_95113

theorem wickets_before_last_match (R W : ℝ) (h1 : R = 12.4 * W) (h2 : R + 26 = 12 * (W + 7)) :
  W = 145 := 
by 
  sorry

end wickets_before_last_match_l95_95113


namespace winning_candidate_votes_l95_95360

-- Define the conditions as hypotheses in Lean.
def two_candidates (candidates : ℕ) : Prop := candidates = 2
def winner_received_62_percent (V : ℝ) (votes_winner : ℝ) : Prop := votes_winner = 0.62 * V
def winning_margin (V : ℝ) : Prop := 0.24 * V = 384

-- The main theorem to prove: the winner candidate received 992 votes.
theorem winning_candidate_votes (V votes_winner : ℝ) (candidates : ℕ) 
  (h1 : two_candidates candidates) 
  (h2 : winner_received_62_percent V votes_winner)
  (h3 : winning_margin V) : 
  votes_winner = 992 :=
by
  sorry

end winning_candidate_votes_l95_95360


namespace total_buckets_poured_l95_95854

-- Define given conditions
def initial_buckets : ℝ := 1
def additional_buckets : ℝ := 8.8

-- Theorem to prove the total number of buckets poured
theorem total_buckets_poured : 
  initial_buckets + additional_buckets = 9.8 :=
by
  sorry

end total_buckets_poured_l95_95854


namespace arithmetic_sequence_c_d_sum_l95_95106

theorem arithmetic_sequence_c_d_sum (c d : ℕ) 
  (h1 : 10 - 3 = 7) 
  (h2 : 17 - 10 = 7) 
  (h3 : c = 17 + 7) 
  (h4 : d = c + 7) 
  (h5 : d + 7 = 38) :
  c + d = 55 := 
by
  sorry

end arithmetic_sequence_c_d_sum_l95_95106


namespace parker_net_income_after_taxes_l95_95576

noncomputable def parker_income : Real := sorry

theorem parker_net_income_after_taxes :
  let daily_pay := 63
  let hours_per_day := 8
  let hourly_rate := daily_pay / hours_per_day
  let overtime_rate := 1.5 * hourly_rate
  let overtime_hours_per_weekend_day := 3
  let weekends_in_6_weeks := 6
  let days_per_week := 7
  let total_days_in_6_weeks := days_per_week * weekends_in_6_weeks
  let regular_earnings := daily_pay * total_days_in_6_weeks
  let total_overtime_earnings := overtime_rate * overtime_hours_per_weekend_day * 2 * weekends_in_6_weeks
  let gross_income := regular_earnings + total_overtime_earnings
  let tax_rate := 0.1
  let net_income_after_taxes := gross_income * (1 - tax_rate)
  net_income_after_taxes = 2764.125 := by sorry

end parker_net_income_after_taxes_l95_95576


namespace milk_water_ratio_l95_95784

theorem milk_water_ratio
  (vessel1_milk_ratio : ℚ)
  (vessel1_water_ratio : ℚ)
  (vessel2_milk_ratio : ℚ)
  (vessel2_water_ratio : ℚ)
  (equal_mixture_units  : ℚ)
  (h1 : vessel1_milk_ratio / vessel1_water_ratio = 4 / 1)
  (h2 : vessel2_milk_ratio / vessel2_water_ratio = 7 / 3)
  :
  (vessel1_milk_ratio + vessel2_milk_ratio) / 
  (vessel1_water_ratio + vessel2_water_ratio) = 11 / 4 :=
by
  sorry

end milk_water_ratio_l95_95784


namespace nurse_distribution_l95_95270

theorem nurse_distribution (nurses hospitals : ℕ) (h1 : nurses = 3) (h2 : hospitals = 6) 
  (h3 : ∀ (a b c : ℕ), a = b → b = c → a = c → a ≤ 2) : 
  (hospitals^nurses - hospitals) = 210 := 
by 
  sorry

end nurse_distribution_l95_95270


namespace restaurant_hamburgers_l95_95754

-- Define the conditions
def hamburgers_served : ℕ := 3
def hamburgers_left_over : ℕ := 6

-- Define the total hamburgers made
def hamburgers_made : ℕ := hamburgers_served + hamburgers_left_over

-- State and prove the theorem
theorem restaurant_hamburgers : hamburgers_made = 9 := by
  sorry

end restaurant_hamburgers_l95_95754


namespace rectangle_area_l95_95235

theorem rectangle_area (side_length : ℝ) (rect_width : ℝ) (rect_length : ℝ) : 
  side_length^2 = 64 → 
  rect_width = side_length →
  rect_length = 3 * rect_width →
  rect_width * rect_length = 192 := 
by
  intros h1 h2 h3
  sorry

end rectangle_area_l95_95235


namespace solve_diophantine_equations_l95_95107

theorem solve_diophantine_equations :
  { (a, b, c, d) : ℤ × ℤ × ℤ × ℤ |
    a * b - 2 * c * d = 3 ∧
    a * c + b * d = 1 } =
  { (1, 3, 1, 0), (-1, -3, -1, 0), (3, 1, 0, 1), (-3, -1, 0, -1) } :=
by
  sorry

end solve_diophantine_equations_l95_95107


namespace value_of_square_l95_95485

variable (x y : ℝ)

theorem value_of_square (h1 : x * (x + y) = 30) (h2 : y * (x + y) = 60) :
  (x + y) ^ 2 = 90 := sorry

end value_of_square_l95_95485


namespace rd_expense_necessary_for_increase_l95_95588

theorem rd_expense_necessary_for_increase :
  ∀ (R_and_D_t : ℝ) (delta_APL_t1 : ℝ),
  R_and_D_t = 3289.31 → delta_APL_t1 = 1.55 →
  R_and_D_t / delta_APL_t1 = 2122 := 
by
  intros R_and_D_t delta_APL_t1 hR hD
  rw [hR, hD]
  norm_num
  sorry

end rd_expense_necessary_for_increase_l95_95588


namespace sum_of_15_consecutive_integers_perfect_square_l95_95933

open Nat

-- statement that defines the conditions and the objective of the problem
theorem sum_of_15_consecutive_integers_perfect_square :
  ∃ n k : ℕ, 15 * (n + 7) = k^2 ∧ 15 * (n + 7) ≥ 225 := 
sorry

end sum_of_15_consecutive_integers_perfect_square_l95_95933


namespace sum_digits_single_digit_l95_95699

theorem sum_digits_single_digit (n : ℕ) (h : n = 2^100) : (n % 9) = 7 := 
sorry

end sum_digits_single_digit_l95_95699


namespace part1_part2_l95_95300

def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

theorem part1 : {x : ℝ | f x ≤ 4} = {x : ℝ | -5 / 3 ≤ x ∧ x ≤ 1} :=
by
  sorry

theorem part2 {a : ℝ} :
  ({x : ℝ | f x ≤ 4} ⊆ {x : ℝ | |x + 3| + |x + a| < x + 6}) ↔ (-4 / 3 < a ∧ a < 2) :=
by
  sorry

end part1_part2_l95_95300


namespace minimum_bailing_rate_is_seven_l95_95759

noncomputable def minimum_bailing_rate (shore_distance : ℝ) (paddling_speed : ℝ) 
                                       (water_intake_rate : ℝ) (max_capacity : ℝ) : ℝ := 
  let time_to_shore := shore_distance / paddling_speed
  let intake_total := water_intake_rate * time_to_shore
  let required_rate := (intake_total - max_capacity) / time_to_shore
  required_rate

theorem minimum_bailing_rate_is_seven 
  (shore_distance : ℝ) (paddling_speed : ℝ) (water_intake_rate : ℝ) (max_capacity : ℝ) :
  shore_distance = 2 →
  paddling_speed = 3 →
  water_intake_rate = 8 →
  max_capacity = 40 →
  minimum_bailing_rate shore_distance paddling_speed water_intake_rate max_capacity = 7 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end minimum_bailing_rate_is_seven_l95_95759


namespace eccentricity_of_hyperbola_l95_95339

noncomputable def hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (h : (4:ℝ) * a^2 = c^2) : ℝ :=
  c / a

theorem eccentricity_of_hyperbola (a b c : ℝ) (ha : a > 0) (hb : b > 0) (h : (4:ℝ) * a^2 = c^2) :
  hyperbola_eccentricity a b c ha hb h = 2 :=
by
  sorry


end eccentricity_of_hyperbola_l95_95339


namespace find_y_l95_95658

theorem find_y (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : x = 8) : y = 1 :=
by
  sorry

end find_y_l95_95658


namespace series_sum_equals_1_over_400_l95_95179

noncomputable def series_term (n : ℕ) : ℝ :=
  (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

theorem series_sum_equals_1_over_400 :
  ∑' n, series_term (n + 1) = 1 / 400 := by
  sorry

end series_sum_equals_1_over_400_l95_95179


namespace total_simple_interest_l95_95673

noncomputable def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

theorem total_simple_interest : simple_interest 2500 10 4 = 1000 := 
by
  sorry

end total_simple_interest_l95_95673


namespace one_three_digit_cube_divisible_by_16_l95_95260

theorem one_three_digit_cube_divisible_by_16 :
  ∃! (n : ℕ), (100 ≤ n ∧ n < 1000 ∧ ∃ (k : ℕ), n = k^3 ∧ 16 ∣ n) :=
sorry

end one_three_digit_cube_divisible_by_16_l95_95260


namespace helen_cookies_till_last_night_l95_95221

theorem helen_cookies_till_last_night 
  (cookies_yesterday : Nat := 31) 
  (cookies_day_before_yesterday : Nat := 419) : 
  cookies_yesterday + cookies_day_before_yesterday = 450 := 
by
  sorry

end helen_cookies_till_last_night_l95_95221


namespace black_squares_in_45th_row_l95_95803

-- Definitions based on the conditions
def number_of_squares_in_row (n : ℕ) : ℕ := 2 * n + 1

def number_of_black_squares (total_squares : ℕ) : ℕ := (total_squares - 1) / 2

-- The theorem statement
theorem black_squares_in_45th_row : number_of_black_squares (number_of_squares_in_row 45) = 45 :=
by sorry

end black_squares_in_45th_row_l95_95803


namespace x_midpoint_of_MN_l95_95034

-- Definition: Given the parabola y^2 = 4x
def parabola (y x : ℝ) : Prop := y^2 = 4 * x

-- Definition: Point F is the focus of the parabola y^2 = 4x
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Definition: Points M and N are on the parabola
def on_parabola (M N : ℝ × ℝ) : Prop :=
  parabola M.2 M.1 ∧ parabola N.2 N.1

-- Definition: The sum of distances |MF| + |NF| = 6
def sum_of_distances (M N : ℝ × ℝ) (F : ℝ × ℝ) : Prop :=
  dist M F + dist N F = 6

-- Theorem: Prove that the x-coordinate of the midpoint of MN is 2
theorem x_midpoint_of_MN (M N : ℝ × ℝ) (F : ℝ × ℝ) 
  (hF : focus F) (hM_N : on_parabola M N) (hDist : sum_of_distances M N F) :
  (M.1 + N.1) / 2 = 2 :=
sorry

end x_midpoint_of_MN_l95_95034


namespace sqrt_221_range_l95_95556

theorem sqrt_221_range : 14 < Real.sqrt 221 ∧ Real.sqrt 221 < 15 := by
  sorry

end sqrt_221_range_l95_95556


namespace range_of_a_l95_95579

noncomputable def f (a x : ℝ) : ℝ :=
  x^3 + 3 * a * x^2 + 3 * ((a + 2) * x + 1)

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ deriv (f a) x = 0 ∧ deriv (f a) y = 0) ↔ a < -1 ∨ a > 2 :=
by
  sorry

end range_of_a_l95_95579


namespace acute_triangle_altitude_inequality_l95_95009

theorem acute_triangle_altitude_inequality (a b c d e f : ℝ) 
  (A B C : ℝ) 
  (acute_triangle : (d = b * Real.sin C) ∧ (d = c * Real.sin B) ∧
                    (e = a * Real.sin C) ∧ (f = a * Real.sin B))
  (projections : (de = b * Real.cos B) ∧ (df = c * Real.cos C))
  : (de + df ≤ a) := 
sorry

end acute_triangle_altitude_inequality_l95_95009


namespace find_principal_amount_l95_95325

variable (x y : ℝ)

-- conditions given in the problem
def simple_interest_condition : Prop :=
  600 = (x * y * 2) / 100

def compound_interest_condition : Prop :=
  615 = x * ((1 + y / 100)^2 - 1)

-- target statement to be proven
theorem find_principal_amount (h1 : simple_interest_condition x y) (h2 : compound_interest_condition x y) :
  x = 285.7142857 :=
  sorry

end find_principal_amount_l95_95325


namespace father_20_bills_count_l95_95289

-- Defining the conditions from the problem.
variables (mother50 mother20 mother10 father50 father10 : ℕ)
def mother_total := mother50 * 50 + mother20 * 20 + mother10 * 10
def father_total (x : ℕ) := father50 * 50 + x * 20 + father10 * 10

-- Given conditions
axiom mother_given : mother50 = 1 ∧ mother20 = 2 ∧ mother10 = 3
axiom father_given : father50 = 4 ∧ father10 = 1
axiom school_fee : 350 = 350

-- Theorem to prove
theorem father_20_bills_count (x : ℕ) :
  mother_total 1 2 3 + father_total 4 x 1 = 350 → x = 1 :=
by sorry

end father_20_bills_count_l95_95289


namespace opposite_of_neg_three_l95_95254

theorem opposite_of_neg_three : -(-3) = 3 := by
  sorry

end opposite_of_neg_three_l95_95254


namespace not_directly_nor_inversely_proportional_A_not_directly_nor_inversely_proportional_D_l95_95972

def equationA (x y : ℝ) : Prop := 2 * x + 3 * y = 5
def equationD (x y : ℝ) : Prop := 4 * x + 2 * y = 8

def directlyProportional (x y : ℝ) : Prop := ∃ k : ℝ, y = k * x
def inverselyProportional (x y : ℝ) : Prop := ∃ k : ℝ, x * y = k

theorem not_directly_nor_inversely_proportional_A (x y : ℝ) :
  equationA x y → ¬ (directlyProportional x y ∨ inverselyProportional x y) := 
sorry

theorem not_directly_nor_inversely_proportional_D (x y : ℝ) :
  equationD x y → ¬ (directlyProportional x y ∨ inverselyProportional x y) := 
sorry

end not_directly_nor_inversely_proportional_A_not_directly_nor_inversely_proportional_D_l95_95972


namespace geometric_sequence_conditions_l95_95641

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

end geometric_sequence_conditions_l95_95641


namespace right_triangle_sin_sum_l95_95366

/--
In a right triangle ABC with ∠A = 90°, prove that sin A + sin^2 B + sin^2 C = 2.
-/
theorem right_triangle_sin_sum (A B C : ℝ) (hA : A = 90) (hABC : A + B + C = 180) :
  Real.sin (A * π / 180) + Real.sin (B * π / 180) ^ 2 + Real.sin (C * π / 180) ^ 2 = 2 :=
sorry

end right_triangle_sin_sum_l95_95366


namespace max_value_of_f_product_of_zeros_l95_95970

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := Real.log x - a * x + b
 
theorem max_value_of_f (a b x1 x2 : ℝ) (h : 0 < a) (hz1 : Real.log x1 - a * x1 + b = 0) (hz2 : Real.log x2 - a * x2 + b = 0) : f (1 / a) a b = -Real.log a - 1 + b :=
by
  sorry

theorem product_of_zeros (a b x1 x2 : ℝ) (h : 0 < a) (hz1 : Real.log x1 - a * x1 + b = 0) (hz2 : Real.log x2 - a * x2 + b = 0) (hx_ne : x1 ≠ x2) : x1 * x2 < 1 / (a * a) :=
by
  sorry

end max_value_of_f_product_of_zeros_l95_95970


namespace total_money_collected_l95_95883

def hourly_wage : ℕ := 10 -- Marta's hourly wage 
def tips_collected : ℕ := 50 -- Tips collected by Marta
def hours_worked : ℕ := 19 -- Hours Marta worked

theorem total_money_collected : (hourly_wage * hours_worked + tips_collected = 240) :=
  sorry

end total_money_collected_l95_95883


namespace ratio_of_areas_l95_95922

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = (9 / 4) :=
by
  sorry

end ratio_of_areas_l95_95922


namespace set_union_proof_l95_95252

  open Set

  def M : Set ℕ := {0, 1, 3}
  def N : Set ℕ := {x | ∃ a, a ∈ M ∧ x = 3 * a}

  theorem set_union_proof : M ∪ N = {0, 1, 3, 9} :=
  by
    sorry
  
end set_union_proof_l95_95252


namespace challenging_math_problem_l95_95665

theorem challenging_math_problem :
  ((9^2 + (3^3 - 1) * 4^2) % 6) * Real.sqrt 49 + (15 - 3 * 5) = 35 :=
by
  sorry

end challenging_math_problem_l95_95665


namespace theatre_fraction_l95_95872

noncomputable def fraction_theatre_took_elective_last_year (T P Th M : ℕ) : Prop :=
  (P = 1 / 2 * T) ∧
  (Th + M = T - P) ∧
  (1 / 3 * P + M = 2 / 3 * T) ∧
  (Th = 1 / 6 * T)

theorem theatre_fraction (T P Th M : ℕ) :
  fraction_theatre_took_elective_last_year T P Th M →
  Th / T = 1 / 6 :=
by
  intro h
  cases h
  sorry

end theatre_fraction_l95_95872


namespace difference_q_r_share_l95_95069

theorem difference_q_r_share (x : ℝ) (h1 : 7 * x - 3 * x = 2800) :
  12 * x - 7 * x = 3500 :=
by
  sorry

end difference_q_r_share_l95_95069


namespace smallest_b_for_factoring_l95_95732

theorem smallest_b_for_factoring (b : ℕ) : 
  (∃ r s : ℤ, x^2 + b*x + (1200 : ℤ) = (x + r)*(x + s) ∧ b = r + s ∧ r * s = 1200) →
  b = 70 := 
sorry

end smallest_b_for_factoring_l95_95732


namespace parabola_vertex_example_l95_95780

-- Definitions based on conditions
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def vertex (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 + 3

-- Conditions given in the problem
def condition1 (a b c : ℝ) : Prop := parabola a b c 2 = 5
def condition2 (a : ℝ) : Prop := vertex a 1 = 3

-- Goal statement to be proved
theorem parabola_vertex_example : ∃ (a b c : ℝ), 
  condition1 a b c ∧ condition2 a ∧ a - b + c = 11 :=
by
  sorry

end parabola_vertex_example_l95_95780


namespace karl_sticker_count_l95_95340

theorem karl_sticker_count : 
  ∀ (K R B : ℕ), 
    (R = K + 20) → 
    (B = R - 10) → 
    (K + R + B = 105) → 
    K = 25 := 
by
  intros K R B hR hB hSum
  sorry

end karl_sticker_count_l95_95340


namespace evaluate_f_a_plus_1_l95_95691

variable (a : ℝ)  -- The variable a is a real number.

def f (x : ℝ) : ℝ := x^2 + 1  -- The function f is defined as x^2 + 1.

theorem evaluate_f_a_plus_1 : f (a + 1) = a^2 + 2 * a + 2 := by
  -- Provide the proof here
  sorry

end evaluate_f_a_plus_1_l95_95691


namespace hexagon_angle_U_l95_95888

theorem hexagon_angle_U 
  (F I U G E R : ℝ)
  (h1 : F = I) 
  (h2 : I = U)
  (h3 : G + E = 180)
  (h4 : R + U = 180)
  (h5 : F + I + G + U + R + E = 720) :
  U = 120 := by
  sorry

end hexagon_angle_U_l95_95888


namespace f_1987_is_3_l95_95897

noncomputable def f : ℕ → ℕ :=
sorry

axiom f_is_defined : ∀ x : ℕ, f x ≠ 0
axiom f_initial : f 1 = 3
axiom f_functional_equation : ∀ (a b : ℕ), f (a + b) = f a + f b - 2 * f (a * b) + 1

theorem f_1987_is_3 : f 1987 = 3 :=
by
  -- Here we would provide the mathematical proof
  sorry

end f_1987_is_3_l95_95897


namespace max_value_trig_formula_l95_95480

theorem max_value_trig_formula (x : ℝ) : ∃ (M : ℝ), M = 5 ∧ ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ M := 
sorry

end max_value_trig_formula_l95_95480


namespace pi_over_2_irrational_l95_95957

def is_rational (x : ℝ) : Prop :=
  ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def is_irrational (x : ℝ) : Prop :=
  ¬ is_rational x

theorem pi_over_2_irrational : is_irrational (Real.pi / 2) :=
by sorry

end pi_over_2_irrational_l95_95957


namespace interval_monotonicity_no_zeros_min_a_l95_95097

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

theorem interval_monotonicity (a : ℝ) :
  a = 1 →
  (∀ x, 0 < x ∧ x ≤ 2 → f a x < f a (x+1)) ∧
  (∀ x, x ≥ 2 → f a x < f a (x-1)) :=
by
  sorry

theorem no_zeros_min_a : 
  (∀ x, x ∈ Set.Ioo 0 (1/2 : ℝ) → f a x ≠ 0) →
  a ≥ 2 - 4 * Real.log 2 :=
by
  sorry

end interval_monotonicity_no_zeros_min_a_l95_95097


namespace negation_of_universal_prop_l95_95025

theorem negation_of_universal_prop :
  (¬ (∀ x : ℝ, x^2 - 5 * x + 3 ≤ 0)) ↔ (∃ x : ℝ, x^2 - 5 * x + 3 > 0) :=
by sorry

end negation_of_universal_prop_l95_95025


namespace classroom_student_count_l95_95136

theorem classroom_student_count (n : ℕ) (students_avg : ℕ) (teacher_age : ℕ) (combined_avg : ℕ) 
  (h1 : students_avg = 8) (h2 : teacher_age = 32) (h3 : combined_avg = 11) 
  (h4 : (8 * n + 32) / (n + 1) = 11) : n + 1 = 8 :=
by
  sorry

end classroom_student_count_l95_95136


namespace shorter_leg_of_right_triangle_l95_95974

theorem shorter_leg_of_right_triangle {a b : ℕ} (hypotenuse : ℕ) (h : hypotenuse = 41) (h_right_triangle : a^2 + b^2 = hypotenuse^2) (h_ineq : a < b) : a = 9 :=
by {
  -- proof to be filled in 
  sorry
}

end shorter_leg_of_right_triangle_l95_95974


namespace sum_of_consecutive_integers_l95_95304

theorem sum_of_consecutive_integers (n a : ℕ) (h₁ : 2 ≤ n) (h₂ : (n * (2 * a + n - 1)) = 36) :
    ∃! (a' n' : ℕ), 2 ≤ n' ∧ (n' * (2 * a' + n' - 1)) = 36 :=
  sorry

end sum_of_consecutive_integers_l95_95304


namespace max_a_for_f_l95_95725

theorem max_a_for_f :
  ∀ (a : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |a * x^2 - a * x + 1| ≤ 1) → a ≤ 8 :=
sorry

end max_a_for_f_l95_95725


namespace min_dwarfs_l95_95745

theorem min_dwarfs (chairs : Fin 30 → Prop) 
  (h1 : ∀ i : Fin 30, chairs i ∨ chairs ((i + 1) % 30) ∨ chairs ((i + 2) % 30)) :
  ∃ S : Finset (Fin 30), (S.card = 10) ∧ (∀ i : Fin 30, i ∈ S) :=
sorry

end min_dwarfs_l95_95745


namespace find_C_l95_95098

theorem find_C (A B C : ℕ) 
  (h1 : A + B + C = 500) 
  (h2 : A + C = 200) 
  (h3 : B + C = 320) : 
  C = 20 := 
by 
  sorry

end find_C_l95_95098


namespace next_tutoring_day_lcm_l95_95117

theorem next_tutoring_day_lcm :
  Nat.lcm (Nat.lcm (Nat.lcm 4 5) 6) 8 = 120 := by
  sorry

end next_tutoring_day_lcm_l95_95117


namespace extracellular_proof_l95_95249

-- Define the components
def component1 : Set String := {"Na＋", "antibodies", "plasma proteins"}
def component2 : Set String := {"Hemoglobin", "O2", "glucose"}
def component3 : Set String := {"glucose", "CO2", "insulin"}
def component4 : Set String := {"Hormones", "neurotransmitter vesicles", "amino acids"}

-- Define the properties of being a part of the extracellular fluid
def is_extracellular (x : Set String) : Prop :=
  x = component1 ∨ x = component3

-- State the theorem to prove
theorem extracellular_proof : is_extracellular component1 ∧ ¬is_extracellular component2 ∧ is_extracellular component3 ∧ ¬is_extracellular component4 :=
by
  sorry

end extracellular_proof_l95_95249


namespace chess_group_players_count_l95_95194

theorem chess_group_players_count (n : ℕ)
  (h1 : ∀ (x y : ℕ), x ≠ y → ∃ k, k = 2)
  (h2 : n * (n - 1) / 2 = 45) :
  n = 10 := sorry

end chess_group_players_count_l95_95194


namespace smallest_AAB_value_l95_95183

theorem smallest_AAB_value : ∃ (A B : ℕ), 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9 ∧ 110 * A + B = 8 * (10 * A + B) ∧ ¬ (A = B) ∧ 110 * A + B = 773 :=
by sorry

end smallest_AAB_value_l95_95183


namespace perpendicular_lines_l95_95275

theorem perpendicular_lines (a : ℝ) : 
  ∀ x y : ℝ, 3 * y - x + 4 = 0 → 4 * y + a * x + 5 = 0 → a = 12 :=
by
  sorry

end perpendicular_lines_l95_95275


namespace pies_baked_l95_95905

theorem pies_baked (days : ℕ) (eddie_rate : ℕ) (sister_rate : ℕ) (mother_rate : ℕ)
  (H1 : eddie_rate = 3) (H2 : sister_rate = 6) (H3 : mother_rate = 8) (days_eq : days = 7) :
  eddie_rate * days + sister_rate * days + mother_rate * days = 119 :=
by
  sorry

end pies_baked_l95_95905


namespace wilfred_carrots_on_tuesday_l95_95163

theorem wilfred_carrots_on_tuesday :
  ∀ (carrots_eaten_Wednesday carrots_eaten_Thursday total_carrots desired_total: ℕ),
    carrots_eaten_Wednesday = 6 →
    carrots_eaten_Thursday = 5 →
    desired_total = 15 →
    desired_total - (carrots_eaten_Wednesday + carrots_eaten_Thursday) = 4 :=
by
  intros
  sorry

end wilfred_carrots_on_tuesday_l95_95163


namespace find_N_l95_95373

theorem find_N : ∃ (N : ℕ), (1000 ≤ N ∧ N < 10000) ∧ (N^2 % 10000 = N) ∧ (N % 16 = 7) ∧ N = 3751 := 
by sorry

end find_N_l95_95373


namespace problem_1_problem_2_l95_95546

-- Define the given function
def f (x : ℝ) := |x - 1|

-- Problem 1: Prove if f(x) + f(1 - x) ≥ a always holds, then a ≤ 1
theorem problem_1 (a : ℝ) : 
  (∀ x : ℝ, f x + f (1 - x) ≥ a) → a ≤ 1 :=
  sorry

-- Problem 2: Prove if a + 2b = 8, then f(a)^2 + f(b)^2 ≥ 5
theorem problem_2 (a b : ℝ) : 
  (a + 2 * b = 8) → (f a)^2 + (f b)^2 ≥ 5 :=
  sorry

end problem_1_problem_2_l95_95546


namespace cost_of_5_pound_bag_is_2_l95_95656

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

end cost_of_5_pound_bag_is_2_l95_95656


namespace cone_lateral_surface_area_l95_95281

theorem cone_lateral_surface_area (r : ℝ) (V : ℝ) (h : ℝ) (l : ℝ) 
  (h₁ : r = 3)
  (h₂ : V = 12 * Real.pi)
  (h₃ : V = (1 / 3) * Real.pi * r^2 * h)
  (h₄ : l = Real.sqrt (r^2 + h^2)) : 
  ∃ A : ℝ, A = Real.pi * r * l ∧ A = 15 * Real.pi := 
by
  use Real.pi * r * l
  have hr : r = 3 := by exact h₁
  have hV : V = 12 * Real.pi := by exact h₂
  have volume_formula : V = (1 / 3) * Real.pi * r^2 * h := by exact h₃
  have slant_height : l = Real.sqrt (r^2 + h^2) := by exact h₄
  sorry

end cone_lateral_surface_area_l95_95281


namespace Connie_correct_number_l95_95935

theorem Connie_correct_number (x : ℤ) (h : x + 2 = 80) : x - 2 = 76 := by
  sorry

end Connie_correct_number_l95_95935


namespace angle_sum_l95_95723

-- Define the angles in the isosceles triangles
def angle_BAC := 40
def angle_EDF := 50

-- Using the property of isosceles triangles to calculate other angles
def angle_ABC := (180 - angle_BAC) / 2
def angle_DEF := (180 - angle_EDF) / 2

-- Since AD is parallel to CE, angles DAC and ACB are equal as are ADE and DEF
def angle_DAC := angle_ABC
def angle_ADE := angle_DEF

-- The theorem to be proven
theorem angle_sum :
  angle_DAC + angle_ADE = 135 :=
by
  sorry

end angle_sum_l95_95723


namespace whale_consumption_third_hour_l95_95481

theorem whale_consumption_third_hour (x : ℕ) :
  (x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 450) → ((x + 6) = 90) :=
by
  intro h
  sorry

end whale_consumption_third_hour_l95_95481


namespace set_intersection_l95_95879

def U : Set ℤ := {-1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1, 2}

theorem set_intersection :
  (U \ A) ∩ B = {1, 2} :=
by
  sorry

end set_intersection_l95_95879


namespace range_of_a_l95_95218

theorem range_of_a (p q : Prop)
  (hp : ∀ a : ℝ, (1 < a ↔ p))
  (hq : ∀ a : ℝ, (2 ≤ a ∨ a ≤ -2 ↔ q))
  (hpq : ∀ a : ℝ, ∀ (p : Prop), ∀ (q : Prop), (p ∧ q) → p ∧ q) :
    ∀ a : ℝ, p ∧ q → 2 ≤ a :=
sorry

end range_of_a_l95_95218


namespace find_side_length_S2_l95_95567

-- Define the variables and conditions
variables (r s : ℕ)
def is_solution (r s : ℕ) : Prop :=
  2 * r + s = 2160 ∧ 2 * r + 3 * s = 3450

-- Define the problem statement
theorem find_side_length_S2 (r s : ℕ) (h : is_solution r s) : s = 645 :=
sorry

end find_side_length_S2_l95_95567


namespace sufficient_but_not_necessary_l95_95361

variables {p q : Prop}

theorem sufficient_but_not_necessary :
  (p → q) ∧ (¬q → ¬p) ∧ ¬(q → p) → (¬q → ¬p) ∧ (¬(q → p)) :=
by
  sorry

end sufficient_but_not_necessary_l95_95361


namespace local_language_letters_l95_95969

theorem local_language_letters (n : ℕ) (h : 1 + 2 * n = 139) : n = 69 :=
by
  -- Proof skipped
  sorry

end local_language_letters_l95_95969


namespace shorter_tree_height_l95_95008

theorem shorter_tree_height
  (s : ℝ)
  (h₁ : ∀ s, s > 0 )
  (h₂ : s + (s + 20) = 240)
  (h₃ : s / (s + 20) = 5 / 7) :
  s = 110 :=
by
sorry

end shorter_tree_height_l95_95008


namespace quadratic_inequality_solution_set_l95_95869

theorem quadratic_inequality_solution_set (a b : ℝ) (h : ∀ x, (1 < x ∧ x < 2) ↔ x^2 + a * x + b < 0) : b = 2 :=
sorry

end quadratic_inequality_solution_set_l95_95869


namespace P_at_7_eq_5760_l95_95655

noncomputable def P (x : ℝ) : ℝ :=
  12 * (x - 1) * (x - 2) * (x - 3)^2 * (x - 6)^4

theorem P_at_7_eq_5760 : P 7 = 5760 :=
by
  -- Proof goes here
  sorry

end P_at_7_eq_5760_l95_95655


namespace sale_price_with_50_percent_profit_l95_95305

theorem sale_price_with_50_percent_profit (CP SP₁ SP₃ : ℝ) 
(h1 : SP₁ - CP = CP - 448) 
(h2 : SP₃ = 1.5 * CP) 
(h3 : SP₃ = 1020) : 
SP₃ = 1020 := 
by 
  sorry

end sale_price_with_50_percent_profit_l95_95305


namespace polynomial_equivalence_l95_95890

def polynomial_expression (x : ℝ) : ℝ :=
  (3 * x ^ 2 + 2 * x - 5) * (x - 2) - (x - 2) * (x ^ 2 - 5 * x + 28) + (4 * x - 7) * (x - 2) * (x + 4)

theorem polynomial_equivalence (x : ℝ) : 
  polynomial_expression x = 6 * x ^ 3 + 4 * x ^ 2 - 93 * x + 122 :=
by {
  sorry
}

end polynomial_equivalence_l95_95890


namespace complement_set_l95_95010

def U : Set ℝ := Set.univ
def M : Set ℝ := {y | ∃ x : ℝ, 0 < x ∧ x < 1 ∧ y = Real.log x / Real.log 2}

theorem complement_set :
  Set.compl M = {y : ℝ | y ≥ 0} :=
by
  sorry

end complement_set_l95_95010


namespace polygon_interior_angle_sum_l95_95610

theorem polygon_interior_angle_sum (n : ℕ) (hn : 3 ≤ n) :
  (n - 2) * 180 + 180 = 2007 → n = 13 := by
  sorry

end polygon_interior_angle_sum_l95_95610


namespace total_tiles_number_l95_95395

-- Define the conditions based on the problem statement
def square_floor_tiles (s : ℕ) : ℕ := s * s

def black_tiles_count (s : ℕ) : ℕ := 3 * s - 3

-- The main theorem statement: given the number of black tiles as 201,
-- prove that the total number of tiles is 4624
theorem total_tiles_number (s : ℕ) (h₁ : black_tiles_count s = 201) : 
  square_floor_tiles s = 4624 :=
by
  -- This is where the proof would go
  sorry

end total_tiles_number_l95_95395


namespace surface_area_of_each_smaller_cube_l95_95513

theorem surface_area_of_each_smaller_cube
  (L : ℝ) (l : ℝ)
  (h1 : 6 * L^2 = 600)
  (h2 : 125 * l^3 = L^3) :
  6 * l^2 = 24 := by
  sorry

end surface_area_of_each_smaller_cube_l95_95513


namespace solution_set_of_inequality_l95_95962

theorem solution_set_of_inequality :
  {x : ℝ | (x^2 - 2*x - 3) * (x^2 + 1) < 0} = {x : ℝ | -1 < x ∧ x < 3} :=
by
  sorry

end solution_set_of_inequality_l95_95962


namespace melissa_driving_time_l95_95398

theorem melissa_driving_time
  (trips_per_month: ℕ)
  (months_per_year: ℕ)
  (total_hours_per_year: ℕ)
  (total_trips: ℕ)
  (hours_per_trip: ℕ) :
  trips_per_month = 2 ∧
  months_per_year = 12 ∧
  total_hours_per_year = 72 ∧
  total_trips = (trips_per_month * months_per_year) ∧
  hours_per_trip = (total_hours_per_year / total_trips) →
  hours_per_trip = 3 :=
by
  intro h
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  sorry

end melissa_driving_time_l95_95398


namespace inverse_negation_l95_95237

theorem inverse_negation :
  (∀ x : ℝ, x ≥ 3 → x < 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ ¬ (x < 3)) :=
by
  sorry

end inverse_negation_l95_95237


namespace rhombus_area_l95_95463

theorem rhombus_area (R1 R2 : ℝ) (x y : ℝ)
  (hR1 : R1 = 15) (hR2 : R2 = 30)
  (hx : x = 15) (hy : y = 2 * x):
  (x * y / 2 = 225) :=
by 
  -- Lean 4 proof not required here
  sorry

end rhombus_area_l95_95463


namespace jake_fewer_peaches_l95_95807

theorem jake_fewer_peaches (steven_peaches : ℕ) (jake_peaches : ℕ) (h1 : steven_peaches = 19) (h2 : jake_peaches = 7) : steven_peaches - jake_peaches = 12 :=
sorry

end jake_fewer_peaches_l95_95807


namespace speed_of_man_in_still_water_l95_95612

theorem speed_of_man_in_still_water (v_m v_s : ℝ) (h1 : v_m + v_s = 18) (h2 : v_m - v_s = 13) : v_m = 15.5 :=
by {
  -- Proof is not required as per the instructions
  sorry
}

end speed_of_man_in_still_water_l95_95612


namespace pie_eating_fraction_l95_95674

theorem pie_eating_fraction :
  (1 / 3 + 1 / 3^2 + 1 / 3^3 + 1 / 3^4 + 1 / 3^5 + 1 / 3^6 + 1 / 3^7) = 1093 / 2187 := 
sorry

end pie_eating_fraction_l95_95674


namespace find_quadruples_l95_95213

def valid_quadruple (x1 x2 x3 x4 : ℝ) : Prop :=
  x1 + x2 * x3 * x4 = 2 ∧ 
  x2 + x3 * x4 * x1 = 2 ∧ 
  x3 + x4 * x1 * x2 = 2 ∧ 
  x4 + x1 * x2 * x3 = 2

theorem find_quadruples (x1 x2 x3 x4 : ℝ) :
  valid_quadruple x1 x2 x3 x4 ↔ (x1, x2, x3, x4) = (1, 1, 1, 1) ∨ 
                                   (x1, x2, x3, x4) = (3, -1, -1, -1) ∨ 
                                   (x1, x2, x3, x4) = (-1, 3, -1, -1) ∨ 
                                   (x1, x2, x3, x4) = (-1, -1, 3, -1) ∨ 
                                   (x1, x2, x3, x4) = (-1, -1, -1, 3) := by
  sorry

end find_quadruples_l95_95213


namespace range_of_m_l95_95700

noncomputable def f (m x : ℝ) : ℝ := m * (x - 2 * m) * (x + m + 3)
noncomputable def g (x : ℝ) : ℝ := 2^x - 2

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f m x < 0 ∨ g x < 0) ∧ (∃ x : ℝ, x < -4 ∧ f m x * g x < 0) → (-4 < m ∧ m < -2) :=
by
  sorry

end range_of_m_l95_95700


namespace cost_of_each_art_book_l95_95278

-- Define the conditions
def total_cost : ℕ := 30
def cost_per_math_and_science_book : ℕ := 3
def num_math_books : ℕ := 2
def num_art_books : ℕ := 3
def num_science_books : ℕ := 6

-- The proof problem statement
theorem cost_of_each_art_book :
  (total_cost - (num_math_books * cost_per_math_and_science_book + num_science_books * cost_per_math_and_science_book)) / num_art_books = 2 :=
by
  sorry -- proof goes here,

end cost_of_each_art_book_l95_95278


namespace find_y_value_l95_95542

theorem find_y_value (y : ℕ) : (1/8 * 2^36 = 2^33) ∧ (8^y = 2^(3 * y)) → y = 11 :=
by
  intros h
  -- additional elaboration to verify each step using Lean, skipped for simplicity
  sorry

end find_y_value_l95_95542


namespace cost_D_to_E_l95_95245

def distance_DF (DF DE EF : ℝ) : Prop :=
  DE^2 = DF^2 + EF^2

def cost_to_fly (distance : ℝ) (per_kilometer_cost booking_fee : ℝ) : ℝ :=
  distance * per_kilometer_cost + booking_fee

noncomputable def total_cost_to_fly_from_D_to_E : ℝ :=
  let DE := 3750 -- Distance from D to E (km)
  let booking_fee := 120 -- Booking fee in dollars
  let per_kilometer_cost := 0.12 -- Cost per kilometer in dollars
  cost_to_fly DE per_kilometer_cost booking_fee

theorem cost_D_to_E : total_cost_to_fly_from_D_to_E = 570 := by
  sorry

end cost_D_to_E_l95_95245


namespace largest_value_expression_l95_95348

theorem largest_value_expression (a b c : ℝ) (ha : a ∈ ({1, 2, 4} : Set ℝ)) (hb : b ∈ ({1, 2, 4} : Set ℝ)) (hc : c ∈ ({1, 2, 4} : Set ℝ)) (habc_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (a / 2) / (b / c) ≤ 4 :=
sorry

end largest_value_expression_l95_95348


namespace valid_combinations_l95_95145

theorem valid_combinations :
  ∀ (x y z : ℕ), 
  10 ≤ x ∧ x ≤ 20 → 
  10 ≤ y ∧ y ≤ 20 →
  10 ≤ z ∧ z ≤ 20 →
  3 * x^2 - y^2 - 7 * z = 99 →
  (x, y, z) = (15, 10, 12) ∨ (x, y, z) = (16, 12, 11) ∨ (x, y, z) = (18, 15, 13) := 
by
  intros x y z hx hy hz h
  sorry

end valid_combinations_l95_95145


namespace appropriate_weight_design_l95_95123

def weight_design (w_l w_s w_r w_w : ℕ) : Prop :=
  w_l > w_s ∧ w_l > w_w ∧ w_w > w_r ∧ w_s = w_w

theorem appropriate_weight_design :
  weight_design 5 2 1 2 :=
by {
  sorry -- skipped proof
}

end appropriate_weight_design_l95_95123


namespace chess_pieces_present_l95_95586

theorem chess_pieces_present (total_pieces : ℕ) (missing_pieces : ℕ) (h1 : total_pieces = 32) (h2 : missing_pieces = 4) : (total_pieces - missing_pieces) = 28 := 
by sorry

end chess_pieces_present_l95_95586


namespace domain_of_g_l95_95088

noncomputable def g (x : ℝ) : ℝ := Real.logb 3 (Real.logb 4 (Real.logb 5 (Real.logb 6 x)))

theorem domain_of_g : {x : ℝ | x > 6^625} = {x : ℝ | ∃ y : ℝ, y = g x } := sorry

end domain_of_g_l95_95088


namespace calculate_y_position_l95_95549

/--
Given a number line with equally spaced markings, if eight steps are taken from \( 0 \) to \( 32 \),
then the position \( y \) after five steps can be calculated.
-/
theorem calculate_y_position : 
    ∃ y : ℕ, (∀ (step length : ℕ), (8 * step = 32) ∧ (y = 5 * length) → y = 20) :=
by
  -- Provide initial definitions based on the conditions
  let step := 4
  let length := 4
  use (5 * length)
  sorry

end calculate_y_position_l95_95549


namespace find_four_digit_number_l95_95167

/-- 
  If there exists a positive integer M and M² both end in the same sequence of 
  five digits abcde in base 10 where a ≠ 0, 
  then the four-digit number abcd derived from M = 96876 is 9687.
-/
theorem find_four_digit_number
  (M : ℕ)
  (h_end_digits : (M % 100000) = (M * M % 100000))
  (h_first_digit_nonzero : 10000 ≤ M % 100000  ∧ M % 100000 < 100000)
  : (M = 96876 → (M / 10 % 10000 = 9687)) :=
by { sorry }

end find_four_digit_number_l95_95167


namespace orangeade_price_l95_95272

theorem orangeade_price (O W : ℝ) (h1 : O = W) (price_day1 : ℝ) (price_day2 : ℝ) 
    (volume_day1 : ℝ) (volume_day2 : ℝ) (revenue_day1 : ℝ) (revenue_day2 : ℝ) : 
    volume_day1 = 2 * O ∧ volume_day2 = 3 * O ∧ revenue_day1 = revenue_day2 ∧ price_day1 = 0.82 
    → price_day2 = 0.55 :=
by
    intros
    sorry

end orangeade_price_l95_95272


namespace s_is_arithmetic_progression_l95_95867

variables (s : ℕ → ℕ) (ds1 ds2 : ℕ)

-- Conditions
axiom strictly_increasing : ∀ n, s n < s (n + 1)
axiom s_is_positive : ∀ n, 0 < s n
axiom s_s_is_arithmetic : ∃ d1, ∀ k, s (s k) = s (s 0) + k * d1
axiom s_s_plus1_is_arithmetic : ∃ d2, ∀ k, s (s k + 1) = s (s 0 + 1) + k * d2

-- Statement to prove
theorem s_is_arithmetic_progression : ∃ d, ∀ k, s (k + 1) = s 0 + k * d :=
sorry

end s_is_arithmetic_progression_l95_95867


namespace payback_period_l95_95038

def system_unit_cost : ℕ := 9499 -- cost in RUB
def graphics_card_cost : ℕ := 20990 -- cost per card in RUB
def num_graphics_cards : ℕ := 2
def system_unit_power : ℕ := 120 -- power in watts
def graphics_card_power : ℕ := 185 -- power per card in watts
def earnings_per_card_per_day_ethereum : ℚ := 0.00630
def ethereum_to_rub : ℚ := 27790.37 -- RUB per ETH
def electricity_cost_per_kwh : ℚ := 5.38 -- RUB per kWh
def total_investment : ℕ := system_unit_cost + num_graphics_cards * graphics_card_cost
def total_power_consumption_watts : ℕ := system_unit_power + num_graphics_cards * graphics_card_power
def total_power_consumption_kwh_per_day : ℚ := total_power_consumption_watts / 1000 * 24
def daily_earnings_rub : ℚ := earnings_per_card_per_day_ethereum * num_graphics_cards * ethereum_to_rub
def daily_energy_cost : ℚ := total_power_consumption_kwh_per_day * electricity_cost_per_kwh
def net_daily_profit : ℚ := daily_earnings_rub - daily_energy_cost

theorem payback_period : total_investment / net_daily_profit = 179 := by
  sorry

end payback_period_l95_95038


namespace complex_number_first_quadrant_l95_95267

theorem complex_number_first_quadrant (z : ℂ) (h : z = (i - 1) / i) : 
  ∃ x y : ℝ, z = x + y * I ∧ x > 0 ∧ y > 0 := 
sorry

end complex_number_first_quadrant_l95_95267


namespace bus_seating_options_l95_95977

theorem bus_seating_options :
  ∃! (x y : ℕ), 21*x + 10*y = 241 :=
sorry

end bus_seating_options_l95_95977


namespace rhombus_area_outside_circle_l95_95560

theorem rhombus_area_outside_circle (d : ℝ) (r : ℝ) (h_d : d = 10) (h_r : r = 3) : 
  (d * d / 2 - 9 * Real.pi) > 9 :=
by
  sorry

end rhombus_area_outside_circle_l95_95560


namespace molecular_weight_l95_95528

theorem molecular_weight :
  let H_weight := 1.008
  let Br_weight := 79.904
  let O_weight := 15.999
  let C_weight := 12.011
  let N_weight := 14.007
  let S_weight := 32.065
  (2 * H_weight + 1 * Br_weight + 3 * O_weight + 1 * C_weight + 1 * N_weight + 2 * S_weight) = 220.065 :=
by
  let H_weight := 1.008
  let Br_weight := 79.904
  let O_weight := 15.999
  let C_weight := 12.011
  let N_weight := 14.007
  let S_weight := 32.065
  sorry

end molecular_weight_l95_95528


namespace eval_expr_l95_95227

theorem eval_expr : 3^2 * 4 * 6^3 * Nat.factorial 7 = 39191040 := by
  -- the proof will be filled in here
  sorry

end eval_expr_l95_95227


namespace exists_x0_in_interval_l95_95306

noncomputable def f (x : ℝ) : ℝ := (2 : ℝ) / x + Real.log (1 / (x - 1))

theorem exists_x0_in_interval :
  ∃ x0 ∈ Set.Ioo (2 : ℝ) (3 : ℝ), f x0 = 0 := 
sorry  -- Proof is left as an exercise

end exists_x0_in_interval_l95_95306


namespace max_angle_line_plane_l95_95397

theorem max_angle_line_plane (θ : ℝ) (h_angle : θ = 72) :
  ∃ φ : ℝ, φ = 90 ∧ (72 ≤ φ ∧ φ ≤ 90) :=
by sorry

end max_angle_line_plane_l95_95397


namespace set1_eq_set2_eq_set3_eq_set4_eq_set5_eq_l95_95968

open Set

-- (1) The set of integers whose absolute value is not greater than 2
theorem set1_eq : { x : ℤ | |x| ≤ 2 } = {-2, -1, 0, 1, 2} := sorry

-- (2) The set of positive numbers less than 10 that are divisible by 3
theorem set2_eq : { x : ℕ | x < 10 ∧ x > 0 ∧ x % 3 = 0 } = {3, 6, 9} := sorry

-- (3) The set {x | x = |x|, x < 5, x ∈ 𝕫}
theorem set3_eq : { x : ℕ | x < 5 } = {0, 1, 2, 3, 4} := sorry

-- (4) The set {(x, y) | x + y = 6, x ∈ ℕ⁺, y ∈ ℕ⁺}
theorem set4_eq : { p : ℕ × ℕ | p.1 + p.2 = 6 ∧ p.1 > 0 ∧ p.2 > 0 } = {(1, 5), (2, 4), (3, 3), (4, 2), (5, 1) } := sorry

-- (5) The set {-3, -1, 1, 3, 5}
theorem set5_eq : {-3, -1, 1, 3, 5} = { x : ℤ | ∃ k : ℤ, x = 2 * k - 1 ∧ -1 ≤ k ∧ k ≤ 3 } := sorry

end set1_eq_set2_eq_set3_eq_set4_eq_set5_eq_l95_95968


namespace problem_1_problem_2_problem_3_l95_95438

section MathProblems

variable (a b c m n x y : ℝ)
-- Problem 1
theorem problem_1 :
  (-6 * a^2 * b^5 * c) / (-2 * a * b^2)^2 = (3/2) * b * c := sorry

-- Problem 2
theorem problem_2 :
  (-3 * m - 2 * n) * (3 * m + 2 * n) = -9 * m^2 - 12 * m * n - 4 * n^2 := sorry

-- Problem 3
theorem problem_3 :
  ((x - 2 * y)^2 - (x - 2 * y) * (x + 2 * y)) / (2 * y) = -2 * x + 4 * y := sorry

end MathProblems

end problem_1_problem_2_problem_3_l95_95438


namespace rent_for_additional_hour_l95_95280

theorem rent_for_additional_hour (x : ℝ) :
  (25 + 10 * x = 125) → (x = 10) :=
by 
  sorry

end rent_for_additional_hour_l95_95280


namespace total_toothpicks_grid_area_l95_95066

open Nat

-- Definitions
def grid_length : Nat := 30
def grid_width : Nat := 50

-- Prove the total number of toothpicks
theorem total_toothpicks : (31 * grid_width + 51 * grid_length) = 3080 := by
  sorry

-- Prove the area enclosed by the grid
theorem grid_area : (grid_length * grid_width) = 1500 := by
  sorry

end total_toothpicks_grid_area_l95_95066


namespace pet_store_cages_l95_95413

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage remaining_puppies num_cages : ℕ)
  (h1 : initial_puppies = 102) 
  (h2 : sold_puppies = 21) 
  (h3 : puppies_per_cage = 9) 
  (h4 : remaining_puppies = initial_puppies - sold_puppies)
  (h5 : num_cages = remaining_puppies / puppies_per_cage) : 
  num_cages = 9 := 
by
  sorry

end pet_store_cages_l95_95413


namespace inverse_proposition_l95_95358

   theorem inverse_proposition (x a b : ℝ) :
     (x ≥ a^2 + b^2 → x ≥ 2 * a * b) →
     (x ≥ 2 * a * b → x ≥ a^2 + b^2) :=
   sorry
   
end inverse_proposition_l95_95358


namespace chromium_percentage_l95_95769

theorem chromium_percentage (c1 c2 : ℝ) (w1 w2 : ℝ) (percentage1 percentage2 : ℝ) : 
  percentage1 = 0.1 → 
  percentage2 = 0.08 → 
  w1 = 15 → 
  w2 = 35 → 
  (c1 = percentage1 * w1) → 
  (c2 = percentage2 * w2) → 
  (c1 + c2 = 4.3) → 
  ((w1 + w2) = 50) →
  ((c1 + c2) / (w1 + w2) * 100 = 8.6) := 
by 
  sorry

end chromium_percentage_l95_95769


namespace volume_of_56_ounces_is_24_cubic_inches_l95_95555

-- Given information as premises
def directlyProportional (V W : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ V = k * W

-- The specific conditions in the problem
def initial_volume := 48   -- in cubic inches
def initial_weight := 112  -- in ounces
def target_weight := 56    -- in ounces
def target_volume := 24    -- in cubic inches (the value we need to prove)

-- The theorem statement 
theorem volume_of_56_ounces_is_24_cubic_inches
  (h1 : directlyProportional initial_volume initial_weight)
  (h2 : directlyProportional target_volume target_weight)
  (h3 : target_weight = 56)
  (h4 : initial_volume = 48)
  (h5 : initial_weight = 112) :
  target_volume = 24 :=
sorry -- Proof not required as per instructions

end volume_of_56_ounces_is_24_cubic_inches_l95_95555


namespace Jesse_pages_left_to_read_l95_95466

def pages_read := [10, 15, 27, 12, 19]
def total_pages_read := pages_read.sum
def fraction_read : ℚ := 1 / 3
def total_pages : ℚ := total_pages_read / fraction_read
def pages_left_to_read : ℚ := total_pages - total_pages_read

theorem Jesse_pages_left_to_read :
  pages_left_to_read = 166 := by
  sorry

end Jesse_pages_left_to_read_l95_95466


namespace find_value_of_x8_plus_x4_plus_1_l95_95057

theorem find_value_of_x8_plus_x4_plus_1 (x : ℂ) (hx : x^2 + x + 1 = 0) : x^8 + x^4 + 1 = 0 :=
sorry

end find_value_of_x8_plus_x4_plus_1_l95_95057


namespace percentage_failed_both_l95_95563

theorem percentage_failed_both (p_hindi p_english p_pass_both x : ℝ)
  (h₁ : p_hindi = 0.25)
  (h₂ : p_english = 0.5)
  (h₃ : p_pass_both = 0.5)
  (h₄ : (p_hindi + p_english - x) = 0.5) : 
  x = 0.25 := 
sorry

end percentage_failed_both_l95_95563


namespace rectangle_area_l95_95268

-- Conditions
def radius : ℝ := 6
def diameter : ℝ := 2 * radius
def width : ℝ := diameter
def ratio_length_to_width : ℝ := 3

-- Given the ratio of the length to the width is 3:1
def length : ℝ := ratio_length_to_width * width

-- Theorem stating the area of the rectangle
theorem rectangle_area :
  let area := length * width
  area = 432 := by
    sorry

end rectangle_area_l95_95268


namespace find_m_range_l95_95690

noncomputable def range_m (a b c m : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) : Prop :=
  m ≥ 4

-- Here is the theorem statement
theorem find_m_range (a b c m : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) : range_m a b c m h1 h2 h3 :=
sorry

end find_m_range_l95_95690


namespace refrigerator_cost_l95_95005

theorem refrigerator_cost
  (R : ℝ)
  (mobile_phone_cost : ℝ := 8000)
  (loss_percent_refrigerator : ℝ := 0.04)
  (profit_percent_mobile_phone : ℝ := 0.09)
  (overall_profit : ℝ := 120)
  (selling_price_refrigerator : ℝ := 0.96 * R)
  (selling_price_mobile_phone : ℝ := 8720)
  (total_selling_price : ℝ := selling_price_refrigerator + selling_price_mobile_phone)
  (total_cost_price : ℝ := R + mobile_phone_cost)
  (balance_profit_eq : total_selling_price = total_cost_price + overall_profit):
  R = 15000 :=
by
  sorry

end refrigerator_cost_l95_95005


namespace triangle_formation_l95_95996

theorem triangle_formation (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : x₁ ≠ x₂) (h₂ : x₁ ≠ x₃) (h₃ : x₁ ≠ x₄) (h₄ : x₂ ≠ x₃) (h₅ : x₂ ≠ x₄) (h₆ : x₃ ≠ x₄)
  (h₇ : 0 < x₁) (h₈ : 0 < x₂) (h₉ : 0 < x₃) (h₁₀ : 0 < x₄)
  (h₁₁ : (x₁ + x₂ + x₃ + x₄) * (1/x₁ + 1/x₂ + 1/x₃ + 1/x₄) < 17) :
  (x₁ + x₂ > x₃) ∧ (x₂ + x₃ > x₄) ∧ (x₁ + x₃ > x₂) ∧ 
  (x₁ + x₄ > x₃) ∧ (x₁ + x₂ > x₄) ∧ (x₃ + x₄ > x₁) ∧ 
  (x₂ + x₄ > x₁) ∧ (x₂ + x₃ > x₁) :=
sorry

end triangle_formation_l95_95996


namespace simplify_expression_l95_95892

theorem simplify_expression (x y : ℝ) : x^2 * y - 3 * x * y^2 + 2 * y * x^2 - y^2 * x = 3 * x^2 * y - 4 * x * y^2 :=
by
  sorry

end simplify_expression_l95_95892


namespace total_number_of_members_l95_95989

variables (b g : Nat)
def girls_twice_boys : Prop := g = 2 * b
def boys_twice_remaining_girls (b g : Nat) : Prop := b = 2 * (g - 24)

theorem total_number_of_members (b g : Nat) 
  (h1 : girls_twice_boys b g) 
  (h2 : boys_twice_remaining_girls b g) : 
  b + g = 48 := by
  sorry

end total_number_of_members_l95_95989


namespace diff_not_equal_l95_95440

variable (A B : Set ℕ)

def diff (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem diff_not_equal (A B : Set ℕ) :
  A ≠ ∅ ∧ B ≠ ∅ → (diff A B ≠ diff B A) :=
by
  sorry

end diff_not_equal_l95_95440


namespace probability_heart_then_king_of_clubs_l95_95475

theorem probability_heart_then_king_of_clubs : 
  let deck := 52
  let hearts := 13
  let remaining_cards := deck - 1
  let king_of_clubs := 1
  let first_card_heart_probability := (hearts : ℝ) / deck
  let second_card_king_of_clubs_probability := (king_of_clubs : ℝ) / remaining_cards
  first_card_heart_probability * second_card_king_of_clubs_probability = 1 / 204 :=
by
  sorry

end probability_heart_then_king_of_clubs_l95_95475


namespace house_number_count_l95_95135

noncomputable def count_valid_house_numbers : Nat :=
  let two_digit_primes := [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  let valid_combinations := two_digit_primes.product two_digit_primes |>.filter (λ (WX, YZ) => WX ≠ YZ)
  valid_combinations.length

theorem house_number_count : count_valid_house_numbers = 110 :=
  by
    sorry

end house_number_count_l95_95135


namespace intersection_A_compl_B_subset_E_B_l95_95432

namespace MathProof

-- Definitions
def A := {x : ℝ | (x + 3) * (x - 6) ≥ 0}
def B := {x : ℝ | (x + 2) / (x - 14) < 0}
def compl_R_B := {x : ℝ | x ≤ -2 ∨ x ≥ 14}
def E (a : ℝ) := {x : ℝ | 2 * a < x ∧ x < a + 1}

-- Theorem for intersection of A and complement of B
theorem intersection_A_compl_B : A ∩ compl_R_B = {x : ℝ | x ≤ -3 ∨ x ≥ 14} :=
by
  sorry

-- Theorem for subset relationship to determine range of a
theorem subset_E_B (a : ℝ) : (E a ⊆ B) → a ≥ -1 :=
by
  sorry

end MathProof

end intersection_A_compl_B_subset_E_B_l95_95432


namespace cornelia_age_l95_95710

theorem cornelia_age :
  ∃ C : ℕ, 
  (∃ K : ℕ, K = 30 ∧ (C + 20 = 2 * (K + 20))) ∧
  ((K - 5)^2 = 3 * (C - 5)) := by
  sorry

end cornelia_age_l95_95710


namespace DeepakAgeProof_l95_95850

def RahulAgeAfter10Years (RahulAge : ℕ) : Prop := RahulAge + 10 = 26

def DeepakPresentAge (ratioRahul ratioDeepak : ℕ) (RahulAge : ℕ) : ℕ :=
  (2 * RahulAge) / ratioRahul

theorem DeepakAgeProof {DeepakCurrentAge : ℕ}
  (ratioRahul ratioDeepak RahulAge : ℕ)
  (hRatio : ratioRahul = 4)
  (hDeepakRatio : ratioDeepak = 2) :
  RahulAgeAfter10Years RahulAge →
  DeepakCurrentAge = DeepakPresentAge ratioRahul ratioDeepak RahulAge :=
  sorry

end DeepakAgeProof_l95_95850


namespace total_blocks_l95_95938

def initial_blocks := 2
def multiplier := 3
def father_blocks := multiplier * initial_blocks

theorem total_blocks :
  initial_blocks + father_blocks = 8 :=
by 
  -- skipping the proof with sorry
  sorry

end total_blocks_l95_95938


namespace solve_for_k_l95_95407

-- Definition and conditions
def ellipse_eq (k : ℝ) : Prop := ∀ x y, k * x^2 + 5 * y^2 = 5

-- Problem: Prove k = 1 given the above definitions
theorem solve_for_k (k : ℝ) :
  (exists (x y : ℝ), ellipse_eq k ∧ x = 2 ∧ y = 0) -> k = 1 :=
sorry

end solve_for_k_l95_95407


namespace max_modulus_l95_95923

open Complex

noncomputable def max_modulus_condition (z : ℂ) : Prop :=
  abs (z - (0 + 2*Complex.I)) = 1

theorem max_modulus : ∀ z : ℂ, max_modulus_condition z → abs z ≤ 3 :=
  by sorry

end max_modulus_l95_95923


namespace solution_set_of_inequality_l95_95055

variable {a b x : ℝ}

theorem solution_set_of_inequality (h : ∃ y, y = 3*(-5) + a ∧ y = -2*(-5) + b) :
  (3*x + a < -2*x + b) ↔ (x < -5) :=
by sorry

end solution_set_of_inequality_l95_95055


namespace trains_clear_time_l95_95926

theorem trains_clear_time :
  ∀ (length_A length_B length_C : ℕ)
    (speed_A_kmph speed_B_kmph speed_C_kmph : ℕ)
    (distance_AB distance_BC : ℕ),
  length_A = 160 ∧ length_B = 320 ∧ length_C = 480 ∧
  speed_A_kmph = 42 ∧ speed_B_kmph = 30 ∧ speed_C_kmph = 48 ∧
  distance_AB = 200 ∧ distance_BC = 300 →
  ∃ (time_clear : ℚ), time_clear = 50.78 :=
by
  intros length_A length_B length_C
         speed_A_kmph speed_B_kmph speed_C_kmph
         distance_AB distance_BC h
  sorry

end trains_clear_time_l95_95926


namespace expression_eq_49_l95_95599

theorem expression_eq_49 (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 := 
by 
  sorry

end expression_eq_49_l95_95599


namespace range_of_a_l95_95425

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log (x + 1) - x^2

theorem range_of_a (a : ℝ) (p q : ℝ) (h₀ : p ≠ q) (h₁ : -1 < p ∧ p < 0) (h₂ : -1 < q ∧ q < 0) :
  (∀ p q : ℝ, -1 < p ∧ p < 0 → -1 < q ∧ q < 0 → p ≠ q → (f a (p + 1) - f a (q + 1)) / (p - q) > 1) ↔ (6 ≤ a) :=
by
  -- proof is omitted
  sorry

end range_of_a_l95_95425


namespace twelfth_term_l95_95230

noncomputable def a (n : ℕ) : ℕ := 
  if n = 0 then 0 
  else (n * (n + 2)) - ((n - 1) * (n + 1))

theorem twelfth_term : a 12 = 25 :=
by sorry

end twelfth_term_l95_95230


namespace four_digit_numbers_with_3_or_7_l95_95507

theorem four_digit_numbers_with_3_or_7 : 
  let total_four_digit_numbers := 9000
  let numbers_without_3_or_7 := 3584
  total_four_digit_numbers - numbers_without_3_or_7 = 5416 :=
by
  trivial

end four_digit_numbers_with_3_or_7_l95_95507


namespace sum_of_number_and_reverse_is_perfect_square_iff_l95_95993

def is_two_digit (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100

def reverse_of (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  10 * b + a

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem sum_of_number_and_reverse_is_perfect_square_iff :
  ∀ n : ℕ, is_two_digit n →
    is_perfect_square (n + reverse_of n) ↔
      n = 29 ∨ n = 38 ∨ n = 47 ∨ n = 56 ∨ n = 65 ∨ n = 74 ∨ n = 83 ∨ n = 92 :=
by
  sorry

end sum_of_number_and_reverse_is_perfect_square_iff_l95_95993


namespace raul_money_left_l95_95313

theorem raul_money_left (initial_money : ℕ) (cost_per_comic : ℕ) (number_of_comics : ℕ) (money_left : ℕ)
  (h1 : initial_money = 87)
  (h2 : cost_per_comic = 4)
  (h3 : number_of_comics = 8)
  (h4 : money_left = initial_money - (number_of_comics * cost_per_comic)) :
  money_left = 55 :=
by 
  rw [h1, h2, h3] at h4
  exact h4

end raul_money_left_l95_95313


namespace gcd_values_count_l95_95847

theorem gcd_values_count (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) : 
  (∃ S : Finset ℕ, S.card = 12 ∧ ∀ d ∈ S, d = Nat.gcd a b) :=
by
  sorry

end gcd_values_count_l95_95847


namespace values_of_fractions_l95_95170

theorem values_of_fractions (A B : ℝ) :
  (∀ x : ℝ, 3 * x ^ 2 + 2 * x - 8 ≠ 0) →
  (∀ x : ℝ, (6 * x - 7) / (3 * x ^ 2 + 2 * x - 8) = A / (x - 2) + B / (3 * x + 4)) →
  A = 1 / 2 ∧ B = 4.5 :=
by
  intros h1 h2
  sorry

end values_of_fractions_l95_95170


namespace gcd_a_b_l95_95464

-- Define a and b
def a : ℕ := 333333
def b : ℕ := 9999999

-- Prove that gcd(a, b) = 3
theorem gcd_a_b : Nat.gcd a b = 3 := by
  sorry

end gcd_a_b_l95_95464


namespace measles_cases_in_1990_l95_95044

noncomputable def measles_cases_1970 := 480000
noncomputable def measles_cases_2000 := 600
noncomputable def years_between := 2000 - 1970
noncomputable def total_decrease := measles_cases_1970 - measles_cases_2000
noncomputable def decrease_per_year := total_decrease / years_between
noncomputable def years_from_1970_to_1990 := 1990 - 1970
noncomputable def decrease_to_1990 := years_from_1970_to_1990 * decrease_per_year
noncomputable def measles_cases_1990 := measles_cases_1970 - decrease_to_1990

theorem measles_cases_in_1990 : measles_cases_1990 = 160400 := by
  sorry

end measles_cases_in_1990_l95_95044


namespace value_of_x_plus_y_squared_l95_95154

theorem value_of_x_plus_y_squared (x y : ℝ) 
  (h₁ : x^2 + y^2 = 20) 
  (h₂ : x * y = 6) : 
  (x + y)^2 = 32 :=
by
  sorry

end value_of_x_plus_y_squared_l95_95154


namespace prove_n_eq_1_l95_95726

-- Definitions of the given conditions
def is_prime (x : ℕ) : Prop := Nat.Prime x

variable {p q r n : ℕ}
variable (hp : is_prime p) (hq : is_prime q) (hr : is_prime r)
variable (hn_pos : n > 0)
variable (h_eq : p^n + q^n = r^2)

-- Statement to prove
theorem prove_n_eq_1 : n = 1 :=
  sorry

end prove_n_eq_1_l95_95726


namespace painted_cells_solutions_l95_95345

def painted_cells (k l : ℕ) : ℕ := (2 * k + 1) * (2 * l + 1) - 74

theorem painted_cells_solutions : ∃ k l : ℕ, k * l = 74 ∧ (painted_cells k l = 373 ∨ painted_cells k l = 301) :=
by
  sorry

end painted_cells_solutions_l95_95345


namespace exists_circle_touching_given_circles_and_line_l95_95663

-- Define the given radii
def r1 := 1
def r2 := 3
def r3 := 4

-- Prove that there exists a circle with a specific radius touching the given circles and line AB
theorem exists_circle_touching_given_circles_and_line (x : ℝ) :
  ∃ (r : ℝ), r > 0 ∧ (r + r1) = x ∧ (r + r2) = x ∧ (r + r3) = x :=
sorry

end exists_circle_touching_given_circles_and_line_l95_95663


namespace sqrt_product_l95_95893

theorem sqrt_product (h1 : Real.sqrt 81 = 9) 
                     (h2 : Real.sqrt 16 = 4) 
                     (h3 : Real.sqrt (Real.sqrt (Real.sqrt 64)) = 2 * Real.sqrt 2) : 
                     Real.sqrt 81 * Real.sqrt 16 * Real.sqrt (Real.sqrt (Real.sqrt 64)) = 72 * Real.sqrt 2 :=
by
  sorry

end sqrt_product_l95_95893


namespace nine_a_minus_six_b_l95_95387

-- Define the variables and conditions.
variables (a b : ℚ)

-- Assume the given conditions.
def condition1 : Prop := 3 * a + 4 * b = 0
def condition2 : Prop := a = 2 * b - 3

-- Formalize the statement to prove.
theorem nine_a_minus_six_b (h1 : condition1 a b) (h2 : condition2 a b) : 9 * a - 6 * b = -81 / 5 :=
sorry

end nine_a_minus_six_b_l95_95387


namespace complex_number_in_first_quadrant_l95_95653

open Complex

theorem complex_number_in_first_quadrant (z : ℂ) (h : z = 1 / (1 - I)) : 
  z.re > 0 ∧ z.im > 0 :=
by
  sorry

end complex_number_in_first_quadrant_l95_95653


namespace eighty_percent_of_number_l95_95396

theorem eighty_percent_of_number (x : ℝ) (h : 0.20 * x = 60) : 0.80 * x = 240 := 
by sorry

end eighty_percent_of_number_l95_95396


namespace anya_hairs_wanted_more_l95_95198

def anya_initial_number_of_hairs : ℕ := 0 -- for simplicity, assume she starts with 0 hairs
def hairs_lost_washing : ℕ := 32
def hairs_lost_brushing : ℕ := hairs_lost_washing / 2
def total_hairs_lost : ℕ := hairs_lost_washing + hairs_lost_brushing
def hairs_to_grow_back : ℕ := 49

theorem anya_hairs_wanted_more : total_hairs_lost + hairs_to_grow_back = 97 :=
by
  sorry

end anya_hairs_wanted_more_l95_95198


namespace trig_function_properties_l95_95645

theorem trig_function_properties :
  ∀ x : ℝ, 
    (1 - 2 * (Real.sin (x - π / 4))^2) = Real.sin (2 * x) ∧ 
    (∀ x : ℝ, Real.sin (2 * (-x)) = -Real.sin (2 * x)) ∧ 
    2 * π / 2 = π :=
by
  sorry

end trig_function_properties_l95_95645


namespace sqrt_meaningful_l95_95404

theorem sqrt_meaningful (x : ℝ) : (x - 2 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end sqrt_meaningful_l95_95404


namespace value_of_sine_neg_10pi_over_3_l95_95819

theorem value_of_sine_neg_10pi_over_3 : Real.sin (-10 * Real.pi / 3) = Real.sqrt 3 / 2 :=
by
  sorry

end value_of_sine_neg_10pi_over_3_l95_95819


namespace ratio_b_a_4_l95_95251

theorem ratio_b_a_4 (a b : ℚ) (h1 : b / a = 4) (h2 : b = 15 - 6 * a) : a = 3 / 2 :=
by
  sorry

end ratio_b_a_4_l95_95251


namespace simon_legos_l95_95824

theorem simon_legos (B : ℝ) (K : ℝ) (x : ℝ) (simon_has : ℝ) 
  (h1 : simon_has = B * 1.20)
  (h2 : K = 40)
  (h3 : B = K + x)
  (h4 : simon_has = 72) : simon_has = 72 := by
  sorry

end simon_legos_l95_95824


namespace min_boxes_to_eliminate_for_one_third_chance_l95_95664

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

end min_boxes_to_eliminate_for_one_third_chance_l95_95664


namespace find_c_if_lines_parallel_l95_95569

theorem find_c_if_lines_parallel (c : ℝ) : 
  (∀ x : ℝ, 5 * x - 3 = (3 * c) * x + 1) → 
  c = 5 / 3 :=
by
  intro h
  sorry

end find_c_if_lines_parallel_l95_95569


namespace reduced_price_per_kg_l95_95667

variables (P P' : ℝ)

-- Given conditions
def condition1 := P' = P / 2
def condition2 := 800 / P' = 800 / P + 5

-- Proof problem statement
theorem reduced_price_per_kg (P P' : ℝ) (h1 : condition1 P P') (h2 : condition2 P P') :
  P' = 80 :=
by
  sorry

end reduced_price_per_kg_l95_95667


namespace arithmetic_sequence_number_of_terms_l95_95190

def arithmetic_sequence_terms_count (a d l : ℕ) : ℕ :=
  sorry

theorem arithmetic_sequence_number_of_terms :
  arithmetic_sequence_terms_count 13 3 73 = 21 :=
sorry

end arithmetic_sequence_number_of_terms_l95_95190


namespace manuscript_typing_cost_l95_95618

theorem manuscript_typing_cost :
  let total_pages := 100
  let revised_once_pages := 30
  let revised_twice_pages := 20
  let cost_first_time := 10
  let cost_revision := 5
  let cost_first_typing := total_pages * cost_first_time
  let cost_revisions_once := revised_once_pages * cost_revision
  let cost_revisions_twice := revised_twice_pages * (2 * cost_revision)
  cost_first_typing + cost_revisions_once + cost_revisions_twice = 1350 :=
by
  let total_pages := 100
  let revised_once_pages := 30
  let revised_twice_pages := 20
  let cost_first_time := 10
  let cost_revision := 5
  let cost_first_typing := total_pages * cost_first_time
  let cost_revisions_once := revised_once_pages * cost_revision
  let cost_revisions_twice := revised_twice_pages * (2 * cost_revision)
  have : cost_first_typing + cost_revisions_once + cost_revisions_twice = 1350 := sorry
  exact this

end manuscript_typing_cost_l95_95618


namespace largest_four_digit_integer_congruent_to_17_mod_26_l95_95473

theorem largest_four_digit_integer_congruent_to_17_mod_26 :
  ∃ x : ℤ, 1000 ≤ x ∧ x < 10000 ∧ x % 26 = 17 ∧ x = 9978 :=
by
  sorry

end largest_four_digit_integer_congruent_to_17_mod_26_l95_95473


namespace comparison_b_a_c_l95_95143

noncomputable def a : ℝ := Real.sqrt 1.2
noncomputable def b : ℝ := Real.exp 0.1
noncomputable def c : ℝ := 1 + Real.log 1.1

theorem comparison_b_a_c : b > a ∧ a > c :=
by
  unfold a b c
  sorry

end comparison_b_a_c_l95_95143


namespace rate_for_gravelling_roads_l95_95752

variable (length breadth width cost : ℕ)
variable (rate per_square_meter : ℕ)

def total_area_parallel_length : ℕ := length * width
def total_area_parallel_breadth : ℕ := (breadth * width) - (width * width)
def total_area : ℕ := total_area_parallel_length length width + total_area_parallel_breadth breadth width

def rate_per_square_meter := cost / total_area length breadth width

theorem rate_for_gravelling_roads :
  (length = 70) →
  (breadth = 30) →
  (width = 5) →
  (cost = 1900) →
  rate_per_square_meter length breadth width cost = 4 := by
  intros; exact sorry

end rate_for_gravelling_roads_l95_95752


namespace max_value_y2_minus_x2_plus_x_plus_5_l95_95643

theorem max_value_y2_minus_x2_plus_x_plus_5 (x y : ℝ) (h : y^2 + x - 2 = 0) : 
  ∃ M, M = 7 ∧ ∀ u v, v^2 + u - 2 = 0 → y^2 - x^2 + x + 5 ≤ M :=
by
  sorry

end max_value_y2_minus_x2_plus_x_plus_5_l95_95643


namespace restore_original_problem_l95_95834

theorem restore_original_problem (X Y : ℕ) (hX : X = 17) (hY : Y = 8) :
  (5 + 1/X) * (Y + 1/2) = 43 := by
  sorry

end restore_original_problem_l95_95834


namespace y_intercept_of_line_l95_95813

theorem y_intercept_of_line : ∃ y : ℝ, 4 * 0 + 7 * y = 28 ∧ 0 = 0 ∧ y = 4 := by
  sorry

end y_intercept_of_line_l95_95813


namespace find_triple_l95_95575
-- Import necessary libraries

-- Define the required predicates and conditions
def satisfies_conditions (x y z : ℕ) : Prop :=
  x ≤ y ∧ y ≤ z ∧ x^3 * (y^3 + z^3) = 2012 * (x * y * z + 2)

-- The main theorem statement
theorem find_triple : 
  ∀ (x y z : ℕ), satisfies_conditions x y z → (x, y, z) = (2, 251, 252) :=
by
  sorry

end find_triple_l95_95575


namespace relatively_prime_divisibility_l95_95133

theorem relatively_prime_divisibility (x y : ℕ) (h1 : Nat.gcd x y = 1) (h2 : y^2 * (y - x)^2 ∣ x^2 * (x + y)) :
  (x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 1) :=
sorry

end relatively_prime_divisibility_l95_95133


namespace stratified_sampling_third_year_l95_95180

theorem stratified_sampling_third_year :
  ∀ (total students_first_year students_second_year sample_size students_third_year sampled_students : ℕ),
  (total = 900) →
  (students_first_year = 240) →
  (students_second_year = 260) →
  (sample_size = 45) →
  (students_third_year = total - students_first_year - students_second_year) →
  (sampled_students = sample_size * students_third_year / total) →
  sampled_students = 20 :=
by
  intros
  sorry

end stratified_sampling_third_year_l95_95180


namespace train_cross_signal_pole_time_l95_95003

theorem train_cross_signal_pole_time :
  ∀ (l_t l_p t_p : ℕ), l_t = 450 → l_p = 525 → t_p = 39 → 
  (l_t * t_p) / (l_t + l_p) = 18 := by
  sorry

end train_cross_signal_pole_time_l95_95003


namespace tablet_battery_life_l95_95273

noncomputable def battery_life_remaining
  (no_use_life : ℝ) (use_life : ℝ) (total_on_time : ℝ) (use_time : ℝ) : ℝ :=
  let no_use_consumption_rate := 1 / no_use_life
  let use_consumption_rate := 1 / use_life
  let no_use_time := total_on_time - use_time
  let total_battery_used := no_use_time * no_use_consumption_rate + use_time * use_consumption_rate
  let remaining_battery := 1 - total_battery_used
  remaining_battery / no_use_consumption_rate

theorem tablet_battery_life (no_use_life : ℝ) (use_life : ℝ) (total_on_time : ℝ) (use_time : ℝ) :
  battery_life_remaining no_use_life use_life total_on_time use_time = 6 :=
by
  -- The proof will go here, we use sorry for now to skip the proof step.
  sorry

end tablet_battery_life_l95_95273


namespace period_started_at_7_am_l95_95334

-- Define the end time of the period
def end_time : ℕ := 16 -- 4 pm in 24-hour format

-- Define the total duration in hours
def duration : ℕ := 9

-- Define the start time of the period
def start_time : ℕ := end_time - duration

-- Prove that the start time is 7 am
theorem period_started_at_7_am : start_time = 7 := by
  sorry

end period_started_at_7_am_l95_95334


namespace find_a_plus_b_l95_95431
-- Definition of the problem variables and conditions
variables (a b : ℝ)
def condition1 : Prop := a - b = 3
def condition2 : Prop := a^2 - b^2 = -12

-- Goal: Prove that a + b = -4 given the conditions
theorem find_a_plus_b (h1 : condition1 a b) (h2 : condition2 a b) : a + b = -4 :=
  sorry

end find_a_plus_b_l95_95431


namespace min_square_sum_l95_95789

theorem min_square_sum (a b : ℝ) (h : a + b = 3) : a^2 + b^2 ≥ 9 / 2 :=
by 
  sorry

end min_square_sum_l95_95789


namespace fraction_values_l95_95758

theorem fraction_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 2 * x^2 + 2 * y^2 = 5 * x * y) :
  ∃ k ∈ ({3, -3} : Set ℝ), (x + y) / (x - y) = k :=
by
  sorry

end fraction_values_l95_95758


namespace smallest_integer_M_exists_l95_95126

theorem smallest_integer_M_exists :
  ∃ (M : ℕ), 
    (M > 0) ∧ 
    (∃ (x y z : ℕ), 
      (x = M ∨ x = M + 1 ∨ x = M + 2) ∧ 
      (y = M ∨ y = M + 1 ∨ y = M + 2) ∧ 
      (z = M ∨ z = M + 1 ∨ z = M + 2) ∧ 
      ((x = M ∨ x = M + 1 ∨ x = M + 2) ∧ x % 8 = 0) ∧ 
      ((y = M ∨ y = M + 1 ∨ y = M + 2) ∧ y % 9 = 0) ∧ 
      ((z = M ∨ z = M + 1 ∨ z = M + 2) ∧ z % 25 = 0) ) ∧ 
    M = 200 := 
by
  sorry

end smallest_integer_M_exists_l95_95126


namespace not_a_factorization_method_l95_95333

def factorization_methods : Set String := 
  {"Taking out the common factor", "Cross multiplication method", "Formula method", "Group factorization"}

theorem not_a_factorization_method : 
  ¬ ("Addition and subtraction elimination method" ∈ factorization_methods) :=
sorry

end not_a_factorization_method_l95_95333


namespace repeating_decimals_sum_l95_95383

theorem repeating_decimals_sum :
  let x := (0.2222222222 : ℚ)
          -- Repeating decimal 0.222... represented up to some precision in rational form.
          -- Of course, internally it is understood with perpetuity.
  let y := (0.0303030303 : ℚ)
          -- Repeating decimal 0.0303... represented up to some precision in rational form.
  x + y = 25 / 99 :=
by
  let x := 2 / 9
  let y := 1 / 33
  sorry

end repeating_decimals_sum_l95_95383


namespace johns_age_l95_95550

-- Define the variables and conditions
def age_problem (j d : ℕ) : Prop :=
j = d - 34 ∧ j + d = 84

-- State the theorem to prove that John's age is 25
theorem johns_age : ∃ (j d : ℕ), age_problem j d ∧ j = 25 :=
by {
  sorry
}

end johns_age_l95_95550


namespace packs_of_chewing_gum_zero_l95_95995

noncomputable def frozen_yogurt_price : ℝ := sorry
noncomputable def chewing_gum_price : ℝ := frozen_yogurt_price / 2
noncomputable def packs_of_chewing_gum : ℕ := sorry

theorem packs_of_chewing_gum_zero 
  (F : ℝ) -- Price of a pint of frozen yogurt
  (G : ℝ) -- Price of a pack of chewing gum
  (x : ℕ) -- Number of packs of chewing gum
  (H1 : G = F / 2)
  (H2 : 5 * F + x * G + 25 = 55)
  : x = 0 :=
sorry

end packs_of_chewing_gum_zero_l95_95995


namespace geometric_series_solution_l95_95543

noncomputable def geometric_series_sums (b1 q : ℝ) : Prop :=
  (b1 / (1 - q) = 16) ∧ (b1^2 / (1 - q^2) = 153.6) ∧ (|q| < 1)

theorem geometric_series_solution (b1 q : ℝ) (h : geometric_series_sums b1 q) :
  q = 2 / 3 ∧ b1 * q^3 = 32 / 9 :=
by
  sorry

end geometric_series_solution_l95_95543


namespace relationship_of_y_values_l95_95264

noncomputable def quadratic_function (x : ℝ) (c : ℝ) := x^2 - 6*x + c

theorem relationship_of_y_values (c : ℝ) (y1 y2 y3 : ℝ) :
  quadratic_function 1 c = y1 →
  quadratic_function (2 * Real.sqrt 2) c = y2 →
  quadratic_function 4 c = y3 →
  y3 < y2 ∧ y2 < y1 :=
by
  intros hA hB hC
  sorry

end relationship_of_y_values_l95_95264


namespace village_population_rate_l95_95043

theorem village_population_rate (r : ℕ) :
  let PX := 72000
  let PY := 42000
  let decrease_rate_X := 1200
  let years := 15
  let population_X_after_years := PX - decrease_rate_X * years
  let population_Y_after_years := PY + r * years
  population_X_after_years = population_Y_after_years → r = 800 :=
by
  sorry

end village_population_rate_l95_95043


namespace intersection_empty_l95_95731

def A : Set ℝ := {x | x^2 + 2 * x - 3 < 0}
def B : Set ℝ := {-3, 1, 2}

theorem intersection_empty : A ∩ B = ∅ := 
by
  sorry

end intersection_empty_l95_95731


namespace exam_max_marks_l95_95119

theorem exam_max_marks (M : ℝ) (h1: 0.30 * M = 66) : M = 220 :=
by
  sorry

end exam_max_marks_l95_95119


namespace principal_amount_l95_95374

theorem principal_amount (SI : ℝ) (R : ℝ) (T : ℕ) (P : ℝ) :
  SI = 3.45 → R = 0.05 → T = 3 → SI = P * R * T → P = 23 :=
by
  -- The proof steps would go here but are omitted as specified.
  sorry

end principal_amount_l95_95374


namespace largest_4_digit_integer_congruent_to_25_mod_26_l95_95626

theorem largest_4_digit_integer_congruent_to_25_mod_26 : ∃ x : ℕ, x < 10000 ∧ x ≥ 1000 ∧ x % 26 = 25 ∧ ∀ y : ℕ, y < 10000 ∧ y ≥ 1000 ∧ y % 26 = 25 → y ≤ x := by
  sorry

end largest_4_digit_integer_congruent_to_25_mod_26_l95_95626


namespace gray_eyed_brunettes_l95_95474

-- Given conditions
def total_students : ℕ := 60
def brunettes : ℕ := 35
def green_eyed_blondes : ℕ := 20
def gray_eyed_total : ℕ := 25

-- Conclude that the number of gray-eyed brunettes is 20
theorem gray_eyed_brunettes :
    (gray_eyed_total - (total_students - brunettes - green_eyed_blondes)) = 20 := by
    sorry

end gray_eyed_brunettes_l95_95474


namespace present_age_of_son_l95_95942

theorem present_age_of_son :
  (∃ (S F : ℕ), F = S + 22 ∧ (F + 2) = 2 * (S + 2)) → ∃ (S : ℕ), S = 20 :=
by
  sorry

end present_age_of_son_l95_95942


namespace max_halls_visited_l95_95510

theorem max_halls_visited (side_len large_tri small_tri: ℕ) 
  (h1 : side_len = 100)
  (h2 : large_tri = 100)
  (h3 : small_tri = 10)
  (div : large_tri = (side_len / small_tri) ^ 2) :
  ∃ m : ℕ, m = 91 → m ≤ large_tri - 9 := 
sorry

end max_halls_visited_l95_95510


namespace g_1987_l95_95551

def g (x : ℕ) : ℚ := sorry

axiom g_defined_for_all (x : ℕ) : true

axiom g1 : g 1 = 1

axiom g_rec (a b : ℕ) : g (a + b) = g a + g b - 3 * g (a * b) + 1

theorem g_1987 : g 1987 = 2 := sorry

end g_1987_l95_95551


namespace solution_set_inequality_l95_95561

theorem solution_set_inequality (x : ℝ) : 3 * x - 2 > x → x > 1 := by
  sorry

end solution_set_inequality_l95_95561


namespace drone_height_l95_95735

theorem drone_height (TR TS TU : ℝ) (UR : TU^2 + TR^2 = 180^2) (US : TU^2 + TS^2 = 150^2) (RS : TR^2 + TS^2 = 160^2) : 
  TU = Real.sqrt 14650 :=
by
  sorry

end drone_height_l95_95735


namespace f_comp_g_eq_g_comp_f_iff_l95_95984

variable {R : Type} [CommRing R]

def f (m n : R) (x : R) : R := m * x ^ 2 + n
def g (p q : R) (x : R) : R := p * x + q

theorem f_comp_g_eq_g_comp_f_iff (m n p q : R) :
  (∀ x : R, f m n (g p q x) = g p q (f m n x)) ↔ n * (1 - p ^ 2) - q * (1 - m) = 0 :=
by
  sorry

end f_comp_g_eq_g_comp_f_iff_l95_95984


namespace isosceles_triangles_with_perimeter_27_count_l95_95844

theorem isosceles_triangles_with_perimeter_27_count :
  ∃ n, (∀ (a : ℕ), 7 ≤ a ∧ a ≤ 13 → ∃ (b : ℕ), b = 27 - 2*a ∧ b < 2*a) ∧ n = 7 :=
sorry

end isosceles_triangles_with_perimeter_27_count_l95_95844


namespace salary_january_l95_95420

theorem salary_january
  (J F M A May : ℝ)  -- declare the salaries as real numbers
  (h1 : (J + F + M + A) / 4 = 8000)  -- condition 1
  (h2 : (F + M + A + May) / 4 = 9500)  -- condition 2
  (h3 : May = 6500) :  -- condition 3
  J = 500 := 
by
  sorry

end salary_january_l95_95420


namespace x_sq_add_x_add_1_divides_x_pow_3p_add_x_pow_3m1_add_x_pow_3n2_l95_95498

theorem x_sq_add_x_add_1_divides_x_pow_3p_add_x_pow_3m1_add_x_pow_3n2 
  (x : ℤ) (p m n : ℕ) (hp : 0 < p) (hm : 0 < m) (hn : 0 < n) :
  (x^2 + x + 1) ∣ (x^(3 * p) + x^(3 * m + 1) + x^(3 * n + 2)) :=
by
  sorry

end x_sq_add_x_add_1_divides_x_pow_3p_add_x_pow_3m1_add_x_pow_3n2_l95_95498


namespace problem_statement_l95_95818

def operation (a b : ℝ) := (a + b) ^ 2

theorem problem_statement (x y : ℝ) : operation ((x + y) ^ 2) ((y + x) ^ 2) = 4 * (x + y) ^ 4 :=
by
  sorry

end problem_statement_l95_95818


namespace B_time_l95_95323

-- Define the work rates of A, B, and C in terms of how long they take to complete the work
variable (A B C : ℝ)

-- Conditions provided in the problem
axiom A_rate : A = 1 / 3
axiom BC_rate : B + C = 1 / 3
axiom AC_rate : A + C = 1 / 2

-- Prove that B alone will take 6 hours to complete the work
theorem B_time : B = 1 / 6 → (1 / B) = 6 := by
  intro hB
  sorry

end B_time_l95_95323


namespace sweeties_remainder_l95_95335

theorem sweeties_remainder (m : ℕ) (h : m % 6 = 4) : (2 * m) % 6 = 2 :=
by {
  sorry
}

end sweeties_remainder_l95_95335


namespace find_a_and_x_l95_95811

theorem find_a_and_x (a x : ℝ) (ha1 : x = (2 * a - 1)^2) (ha2 : x = (-a + 2)^2) : a = -1 ∧ x = 9 := 
by
  sorry

end find_a_and_x_l95_95811


namespace range_of_a_l95_95814

open Set Real

theorem range_of_a :
  let p := ∀ x : ℝ, |4 * x - 3| ≤ 1
  let q := ∀ x : ℝ, x^2 - (2 * a + 1) * x + (a * (a + 1)) ≤ 0
  (¬ p → ¬ q) ∧ ¬ (¬ p ↔ ¬ q)
  → (∀ x : Icc (0 : ℝ) (1 / 2 : ℝ), a = x) :=
by
  intros
  sorry

end range_of_a_l95_95814


namespace Sarah_brother_apples_l95_95696

theorem Sarah_brother_apples (n : Nat) (h1 : 45 = 5 * n) : n = 9 := 
  sorry

end Sarah_brother_apples_l95_95696


namespace tan_105_eq_neg2_sub_sqrt3_l95_95162

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l95_95162


namespace exists_positive_integer_divisible_by_14_with_sqrt_between_25_and_25_3_l95_95028

theorem exists_positive_integer_divisible_by_14_with_sqrt_between_25_and_25_3 :
  ∃ (x : ℕ), x % 14 = 0 ∧ 625 <= x ∧ x <= 640 ∧ x = 630 := 
by 
  sorry

end exists_positive_integer_divisible_by_14_with_sqrt_between_25_and_25_3_l95_95028


namespace ellipse_problem_l95_95577

noncomputable def point_coordinates (x y b : ℝ) : Prop :=
  x = 1 ∧ y = 1 ∧ (4 * x^2 = 4) ∧ (4 * b^2 / (4 + b^2) = 1)

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  (Real.sqrt (a^2 - b^2)) / a

theorem ellipse_problem (b : ℝ) (h₁ : 4 * b^2 / (4 + b^2) = 1) :
  ∃ x y, point_coordinates x y b 
  ∧ eccentricity 2 b = Real.sqrt 6 / 3 := 
by 
  sorry

end ellipse_problem_l95_95577


namespace smallest_non_factor_product_l95_95661

open Nat

def is_factor (n d : ℕ) := d > 0 ∧ n % d = 0

theorem smallest_non_factor_product (a b : ℕ) (h1 : a ≠ b) (h2 : is_factor 48 a) (h3 : is_factor 48 b) (h4 : ¬ is_factor 48 (a * b)) : a * b = 18 :=
by
  sorry

end smallest_non_factor_product_l95_95661


namespace number_of_technicians_l95_95630

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

end number_of_technicians_l95_95630


namespace min_marked_price_l95_95777

theorem min_marked_price 
  (x : ℝ) 
  (sets : ℝ) 
  (cost_per_set : ℝ) 
  (discount : ℝ) 
  (desired_profit : ℝ) 
  (purchase_cost : ℝ) 
  (total_revenue : ℝ) 
  (cost : ℝ)
  (h1 : sets = 40)
  (h2 : cost_per_set = 80)
  (h3 : discount = 0.9)
  (h4 : desired_profit = 4000)
  (h5 : cost = sets * cost_per_set)
  (h6 : total_revenue = sets * (discount * x))
  (h7 : total_revenue - cost ≥ desired_profit) : x ≥ 200 := by
  sorry

end min_marked_price_l95_95777


namespace platform_length_correct_l95_95620

noncomputable def platform_length : ℝ :=
  let T := 180
  let v_kmph := 72
  let t := 20
  let v_ms := v_kmph * 1000 / 3600
  let total_distance := v_ms * t
  total_distance - T

theorem platform_length_correct : platform_length = 220 := by
  sorry

end platform_length_correct_l95_95620


namespace meaningful_fraction_range_l95_95715

theorem meaningful_fraction_range (x : ℝ) : (3 - x) ≠ 0 ↔ x ≠ 3 :=
by sorry

end meaningful_fraction_range_l95_95715


namespace number_of_ways_to_read_BANANA_l95_95792

/-- 
In a 3x3 grid, there are 84 different ways to read the word BANANA 
by moving from one cell to another cell with which it shares an edge,
and cells may be visited more than once.
-/
theorem number_of_ways_to_read_BANANA (grid : Matrix (Fin 3) (Fin 3) Char) (word : String := "BANANA") : 
  ∃! n : ℕ, n = 84 :=
by
  sorry

end number_of_ways_to_read_BANANA_l95_95792


namespace walking_rate_on_escalator_l95_95283

theorem walking_rate_on_escalator (v : ℝ)
  (escalator_speed : ℝ := 12)
  (escalator_length : ℝ := 196)
  (travel_time : ℝ := 14)
  (effective_speed : ℝ := v + escalator_speed)
  (distance_eq : effective_speed * travel_time = escalator_length) :
  v = 2 := by
  sorry

end walking_rate_on_escalator_l95_95283


namespace eqn_of_line_through_intersection_parallel_eqn_of_line_perpendicular_distance_l95_95243

-- Proof 1: Line through intersection and parallel
theorem eqn_of_line_through_intersection_parallel :
  ∃ k : ℝ, (9 : ℝ) * (x: ℝ) + (18: ℝ) * (y: ℝ) - 4 = 0 ∧
           (∀ x y : ℝ, (2 * x + 3 * y - 5 = 0) → (7 * x + 15 * y + 1 = 0) → (x + 2 * y + k = 0)) :=
sorry

-- Proof 2: Line perpendicular and specific distance from origin
theorem eqn_of_line_perpendicular_distance :
  ∃ k : ℝ, (∃ m : ℝ, (k = 30 ∨ k = -30) ∧ (4 * (x: ℝ) - 3 * (y: ℝ) + m = 0 ∧ (∃ d : ℝ, d = 6 ∧ (|m| / (4 ^ 2 + (-3) ^ 2).sqrt) = d))) :=
sorry

end eqn_of_line_through_intersection_parallel_eqn_of_line_perpendicular_distance_l95_95243


namespace LindaCandiesLeft_l95_95224

variable (initialCandies : ℝ)
variable (candiesGiven : ℝ)

theorem LindaCandiesLeft (h1 : initialCandies = 34.0) (h2 : candiesGiven = 28.0) : initialCandies - candiesGiven = 6.0 := by
  sorry

end LindaCandiesLeft_l95_95224


namespace selling_price_of_cycle_l95_95495

theorem selling_price_of_cycle (cost_price : ℕ) (loss_percent : ℕ) (selling_price : ℕ) :
  cost_price = 1400 → loss_percent = 25 → selling_price = 1050 := by
  sorry

end selling_price_of_cycle_l95_95495


namespace min_value_expr_l95_95757

theorem min_value_expr (x : ℝ) (hx : x > 0) : 4 * x + 1 / x^2 ≥ 5 :=
by
  sorry

end min_value_expr_l95_95757


namespace sum_of_powers_l95_95689

theorem sum_of_powers : 5^5 + 5^5 + 5^5 + 5^5 = 4 * 5^5 :=
by
  sorry

end sum_of_powers_l95_95689


namespace delivery_time_is_40_minutes_l95_95768

-- Define the conditions
def total_pizzas : Nat := 12
def two_pizza_stops : Nat := 2
def pizzas_per_stop_with_two_pizzas : Nat := 2
def time_per_stop_minutes : Nat := 4

-- Define the number of pizzas covered by stops with two pizzas
def pizzas_covered_by_two_pizza_stops : Nat := two_pizza_stops * pizzas_per_stop_with_two_pizzas

-- Define the number of single pizza stops
def single_pizza_stops : Nat := total_pizzas - pizzas_covered_by_two_pizza_stops

-- Define the total number of stops
def total_stops : Nat := two_pizza_stops + single_pizza_stops

-- Total time to deliver all pizzas
def total_delivery_time_minutes : Nat := total_stops * time_per_stop_minutes

theorem delivery_time_is_40_minutes : total_delivery_time_minutes = 40 := by
  sorry

end delivery_time_is_40_minutes_l95_95768


namespace solution_to_equation_l95_95212

theorem solution_to_equation (x : ℝ) : x * (x - 2) = 2 * x ↔ (x = 0 ∨ x = 4) := by
  sorry

end solution_to_equation_l95_95212


namespace cyclists_meet_at_start_l95_95833

theorem cyclists_meet_at_start (T : ℚ) (h1 : T = 5 * 7 * 9 / gcd (5 * 7) (gcd (7 * 9) (9 * 5))) : T = 157.5 :=
by
  sorry

end cyclists_meet_at_start_l95_95833


namespace min_n_A0_An_ge_200_l95_95853

theorem min_n_A0_An_ge_200 :
  (∃ n : ℕ, (n * (n + 1)) / 3 ≥ 200) ∧
  (∀ m < 24, (m * (m + 1)) / 3 < 200) :=
sorry

end min_n_A0_An_ge_200_l95_95853


namespace sin_315_eq_neg_sqrt_2_div_2_l95_95712

theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * (Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt_2_div_2_l95_95712


namespace find_tabitha_age_l95_95301

-- Define the conditions
variable (age_started : ℕ) (colors_started : ℕ) (years_future : ℕ) (future_colors : ℕ)

-- Let's specify the given problem's conditions:
axiom h1 : age_started = 15          -- Tabitha started at age 15
axiom h2 : colors_started = 2        -- with 2 colors
axiom h3 : years_future = 3          -- in three years
axiom h4 : future_colors = 8         -- she will have 8 different colors

-- The proof problem we need to state:
theorem find_tabitha_age : ∃ age_now : ℕ, age_now = age_started + (future_colors - colors_started) - years_future := by
  sorry

end find_tabitha_age_l95_95301


namespace verify_exact_countries_attended_l95_95532

theorem verify_exact_countries_attended :
  let start_year := 1990
  let years_between_festivals := 3
  let total_festivals := 12
  let attended_countries := 68
  (attended_countries = 68) :=
by
  let start_year := 1990
  let years_between_festivals := 3
  let total_festivals := 12
  let attended_countries := 68
  have : attended_countries = 68 := rfl
  exact this

end verify_exact_countries_attended_l95_95532


namespace value_when_x_is_neg1_l95_95826

theorem value_when_x_is_neg1 (p q : ℝ) (h : p + q = 2022) : 
  (p * (-1)^3 + q * (-1) + 1) = -2021 := by
  sorry

end value_when_x_is_neg1_l95_95826


namespace total_annual_gain_l95_95570

-- Definitions based on given conditions
variable (A B C : Type) [Field ℝ]

-- Assume initial investments and time factors
variable (x : ℝ) (A_share : ℝ := 5000) -- A's share is Rs. 5000

-- Total annual gain to be proven
theorem total_annual_gain (x : ℝ) (A_share B_share C_share Total_Profit : ℝ) :
  A_share = 5000 → 
  B_share = (2 * x) * (6 / 12) → 
  C_share = (3 * x) * (4 / 12) → 
  (A_share / (x * 12)) * Total_Profit = 5000 → -- A's determined share from profit
  Total_Profit = 15000 := 
by 
  sorry

end total_annual_gain_l95_95570


namespace eiffel_tower_scale_l95_95659

theorem eiffel_tower_scale (height_tower_m : ℝ) (height_model_cm : ℝ) :
    height_tower_m = 324 →
    height_model_cm = 50 →
    (height_tower_m * 100) / height_model_cm = 648 →
    (648 / 100) = 6.48 :=
by
  intro h_tower h_model h_ratio
  rw [h_tower, h_model] at h_ratio
  sorry

end eiffel_tower_scale_l95_95659


namespace total_rowing_proof_l95_95032

def morning_rowing := 13
def afternoon_rowing := 21
def total_rowing := 34

theorem total_rowing_proof :
  morning_rowing + afternoon_rowing = total_rowing :=
by
  sorry

end total_rowing_proof_l95_95032


namespace ratio_perimeter_to_breadth_l95_95881

-- Definitions of the conditions
def area_of_rectangle (length breadth : ℝ) := length * breadth
def perimeter_of_rectangle (length breadth : ℝ) := 2 * (length + breadth)

-- The problem statement: prove the ratio of perimeter to breadth
theorem ratio_perimeter_to_breadth (L B : ℝ) (hL : L = 18) (hA : area_of_rectangle L B = 216) :
  (perimeter_of_rectangle L B) / B = 5 :=
by 
  -- Given definitions and conditions, we skip the proof.
  sorry

end ratio_perimeter_to_breadth_l95_95881


namespace podcast_ratio_l95_95061

theorem podcast_ratio
  (total_drive_time : ℕ)
  (first_podcast : ℕ)
  (third_podcast : ℕ)
  (fourth_podcast : ℕ)
  (next_podcast : ℕ)
  (second_podcast : ℕ) :
  total_drive_time = 360 →
  first_podcast = 45 →
  third_podcast = 105 →
  fourth_podcast = 60 →
  next_podcast = 60 →
  second_podcast = total_drive_time - (first_podcast + third_podcast + fourth_podcast + next_podcast) →
  second_podcast / first_podcast = 2 :=
by
  sorry

end podcast_ratio_l95_95061


namespace brooke_total_jumping_jacks_l95_95750

def sj1 : Nat := 20
def sj2 : Nat := 36
def sj3 : Nat := 40
def sj4 : Nat := 50
def Brooke_jumping_jacks : Nat := 3 * (sj1 + sj2 + sj3 + sj4)

theorem brooke_total_jumping_jacks : Brooke_jumping_jacks = 438 := by
  sorry

end brooke_total_jumping_jacks_l95_95750


namespace coordinates_of_B_l95_95992

theorem coordinates_of_B (x y : ℝ) (A : ℝ × ℝ) (a : ℝ × ℝ) :
  A = (2, 4) ∧ a = (3, 4) ∧ (x - 2, y - 4) = (2 * a.1, 2 * a.2) → (x, y) = (8, 12) :=
by
  intros h
  sorry

end coordinates_of_B_l95_95992


namespace derivative_at_pi_over_4_l95_95704

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem derivative_at_pi_over_4 :
  deriv f (π / 4) = (Real.sqrt 2 / 2) + (Real.sqrt 2 * π / 8) :=
by
  -- Since the focus is only on the statement, the proof is not required.
  sorry

end derivative_at_pi_over_4_l95_95704


namespace handshake_count_l95_95676

theorem handshake_count (num_companies : ℕ) (num_representatives : ℕ) 
  (total_handshakes : ℕ) (h1 : num_companies = 5) (h2 : num_representatives = 5)
  (h3 : total_handshakes = (num_companies * num_representatives * 
   (num_companies * num_representatives - 1 - (num_representatives - 1)) / 2)) :
  total_handshakes = 250 :=
by
  rw [h1, h2] at h3
  exact h3

end handshake_count_l95_95676


namespace inequality_solution_l95_95756

theorem inequality_solution (x : ℝ) :
  (7 / 36 + (abs (2 * x - (1 / 6)))^2 < 5 / 12) ↔
  (x ∈ Set.Ioo ((1 / 12 - (Real.sqrt 2 / 6))) ((1 / 12 + (Real.sqrt 2 / 6)))) :=
by
  sorry

end inequality_solution_l95_95756


namespace problem_1_problem_2_l95_95448

noncomputable def f (x : ℝ) := Real.sin x + (x - 1) / Real.exp x

theorem problem_1 (x : ℝ) (h₀ : x ∈ Set.Icc (-Real.pi) (Real.pi / 2)) :
  MonotoneOn f (Set.Icc (-Real.pi) (Real.pi / 2)) :=
sorry

theorem problem_2 (k : ℝ) :
  ∀ x ∈ Set.Icc (-Real.pi) 0, ((f x - Real.sin x) * Real.exp x - Real.cos x) ≤ k * Real.sin x → 
  k ∈ Set.Iic (1 + Real.pi / 2) :=
sorry

end problem_1_problem_2_l95_95448


namespace compare_neg5_neg7_l95_95761

theorem compare_neg5_neg7 : -5 > -7 := 
by
  sorry

end compare_neg5_neg7_l95_95761


namespace one_meter_to_leaps_l95_95378

theorem one_meter_to_leaps 
  (x y z w u v : ℕ)
  (h1 : x * leaps = y * strides) 
  (h2 : z * bounds = w * leaps) 
  (h3 : u * bounds = v * meters) :
  1 * meters = (uw / vz) * leaps :=
sorry

end one_meter_to_leaps_l95_95378


namespace line_intersects_ellipse_two_points_l95_95504

theorem line_intersects_ellipse_two_points (k b : ℝ) : 
  (-2 < b) ∧ (b < 2) ↔ ∀ x y : ℝ, (y = k * x + b) ↔ (x ^ 2 / 9 + y ^ 2 / 4 = 1) → true :=
sorry

end line_intersects_ellipse_two_points_l95_95504


namespace books_bought_l95_95033

theorem books_bought (initial_books bought_books total_books : ℕ) 
    (h_initial : initial_books = 35)
    (h_total : total_books = 56) :
    bought_books = total_books - initial_books → bought_books = 21 := 
by
  sorry

end books_bought_l95_95033


namespace damage_in_dollars_l95_95785

noncomputable def euros_to_dollars (euros : ℝ) : ℝ := euros * (1 / 0.9)

theorem damage_in_dollars :
  euros_to_dollars 45000000 = 49995000 :=
by
  -- This is where the proof would go
  sorry

end damage_in_dollars_l95_95785


namespace find_numbers_l95_95910

theorem find_numbers (S P : ℝ) (h : S^2 - 4 * P ≥ 0) :
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧
             ((x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨
              (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2)) :=
by
  sorry

end find_numbers_l95_95910


namespace dot_product_in_triangle_l95_95949

noncomputable def ab := 3
noncomputable def ac := 2
noncomputable def bc := Real.sqrt 10

theorem dot_product_in_triangle : 
  let AB := ab
  let AC := ac
  let BC := bc
  (AB = 3) → (AC = 2) → (BC = Real.sqrt 10) → 
  ∃ cosA, (cosA = (AB^2 + AC^2 - BC^2) / (2 * AB * AC)) →
  ∃ dot_product, (dot_product = AB * AC * cosA) ∧ dot_product = 3 / 2 :=
by
  sorry

end dot_product_in_triangle_l95_95949


namespace annual_income_earned_by_both_investments_l95_95375

noncomputable def interest (principal: ℝ) (rate: ℝ) (time: ℝ) : ℝ :=
  principal * rate * time

theorem annual_income_earned_by_both_investments :
  let total_amount := 8000
  let first_investment := 3000
  let first_interest_rate := 0.085
  let second_interest_rate := 0.064
  let second_investment := total_amount - first_investment
  interest first_investment first_interest_rate 1 + interest second_investment second_interest_rate 1 = 575 :=
by
  sorry

end annual_income_earned_by_both_investments_l95_95375


namespace player_1_winning_strategy_l95_95102

-- Define the properties and rules of the game
def valid_pair (a b : ℕ) : Prop := a > 0 ∧ b > 0 ∧ a + b = 2005

def move (current t a b : ℕ) : Prop := 
  current = t - a ∨ current = t - b

def first_player_wins (t a b : ℕ) : Prop :=
  ∀ k : ℕ, t > k * 2005 → ∃ m : ℕ, move (t - m) t a b

-- Main theorem statement
theorem player_1_winning_strategy : ∃ (t : ℕ) (a b : ℕ), valid_pair a b ∧ first_player_wins t a b :=
sorry

end player_1_winning_strategy_l95_95102


namespace marcus_point_value_l95_95925

theorem marcus_point_value 
  (team_total_points : ℕ)
  (marcus_percentage : ℚ)
  (three_point_goals : ℕ)
  (num_goals_type2 : ℕ)
  (score_type1 : ℕ)
  (score_type2 : ℕ)
  (total_marcus_points : ℚ)
  (points_type2 : ℚ)
  (three_point_value : ℕ := 3):
  team_total_points = 70 →
  marcus_percentage = 0.5 →
  three_point_goals = 5 →
  num_goals_type2 = 10 →
  total_marcus_points = marcus_percentage * team_total_points →
  score_type1 = three_point_goals * three_point_value →
  points_type2 = total_marcus_points - score_type1 →
  score_type2 = points_type2 / num_goals_type2 →
  score_type2 = 2 :=
by
  intros
  sorry

end marcus_point_value_l95_95925


namespace aluminum_foil_thickness_l95_95076

-- Define the variables and constants
variables (d l m w t : ℝ)

-- Define the conditions
def density_condition : Prop := d = m / (l * w * t)
def volume_formula : Prop := t = m / (d * l * w)

-- The theorem to prove
theorem aluminum_foil_thickness (h1 : density_condition d l m w t) : volume_formula d l m w t :=
sorry

end aluminum_foil_thickness_l95_95076


namespace second_group_men_count_l95_95855

-- Define the conditions given in the problem
def men1 := 8
def days1 := 80
def days2 := 32

-- The question we need to answer
theorem second_group_men_count : 
  ∃ (men2 : ℕ), men1 * days1 = men2 * days2 ∧ men2 = 20 :=
by
  sorry

end second_group_men_count_l95_95855


namespace prime_cubic_condition_l95_95688

theorem prime_cubic_condition (p : ℕ) (hp : Nat.Prime p) (hp_prime : Nat.Prime (p^4 - 3 * p^2 + 9)) : p = 2 :=
sorry

end prime_cubic_condition_l95_95688


namespace RupertCandles_l95_95210

-- Definitions corresponding to the conditions
def PeterAge : ℕ := 10
def RupertRelativeAge : ℝ := 3.5

-- Define Rupert's age based on Peter's age and the given relative age factor
def RupertAge : ℝ := RupertRelativeAge * PeterAge

-- Statement of the theorem
theorem RupertCandles : RupertAge = 35 := by
  -- Proof is omitted
  sorry

end RupertCandles_l95_95210


namespace problem1_problem2_problem3_problem4_l95_95554

-- Problem 1
theorem problem1 : (-3 / 8) + ((-5 / 8) * (-6)) = 27 / 8 :=
by sorry

-- Problem 2
theorem problem2 : 12 + (7 * (-3)) - (18 / (-3)) = -3 :=
by sorry

-- Problem 3
theorem problem3 : -((2:ℤ)^2) - (4 / 7) * (2:ℚ) - (-((3:ℤ)^2:ℤ) : ℤ) = -99 / 7 :=
by sorry

-- Problem 4
theorem problem4 : -(((-1) ^ 2020 : ℤ)) + ((6 : ℚ) / (-(2 : ℤ) ^ 3)) * (-1 / 3) = -3 / 4 :=
by sorry

end problem1_problem2_problem3_problem4_l95_95554


namespace preston_high_school_teachers_l95_95684

theorem preston_high_school_teachers 
  (num_students : ℕ)
  (classes_per_student : ℕ)
  (classes_per_teacher : ℕ)
  (students_per_class : ℕ)
  (teachers_per_class : ℕ)
  (H : num_students = 1500)
  (C : classes_per_student = 6)
  (T : classes_per_teacher = 5)
  (S : students_per_class = 30)
  (P : teachers_per_class = 1) : 
  (num_students * classes_per_student / students_per_class / classes_per_teacher = 60) :=
by sorry

end preston_high_school_teachers_l95_95684


namespace clock_chime_time_l95_95964

theorem clock_chime_time (t : ℕ) (h : t = 12) (k : 4 * (t / (4 - 1)) = 12) :
  12 * (t / (4 - 1)) - (12 - 1) * (t / (4 - 1)) = 44 :=
by {
  sorry
}

end clock_chime_time_l95_95964


namespace mean_of_elements_increased_by_2_l95_95762

noncomputable def calculate_mean_after_increase (m : ℝ) (median_value : ℝ) (increase_value : ℝ) : ℝ :=
  let set := [m, m + 2, m + 4, m + 7, m + 11, m + 13]
  let increased_set := set.map (λ x => x + increase_value)
  increased_set.sum / increased_set.length

theorem mean_of_elements_increased_by_2 (m : ℝ) (h : (m + 4 + m + 7) / 2 = 10) :
  calculate_mean_after_increase m 10 2 = 38 / 3 :=
by 
  sorry

end mean_of_elements_increased_by_2_l95_95762


namespace smallest_a_exists_l95_95158

theorem smallest_a_exists : ∃ a b c : ℤ, a > 0 ∧ b^2 > 4*a*c ∧ 
  (∀ x : ℝ, x > 0 ∧ x < 1 → (a * x^2 - b * x + c) = 0 → false) 
  ∧ a = 5 :=
by sorry

end smallest_a_exists_l95_95158


namespace no_distinct_positive_integers_l95_95382

noncomputable def P (x : ℕ) : ℕ := x^2000 - x^1000 + 1

theorem no_distinct_positive_integers (a : Fin 2001 → ℕ) (h_distinct : Function.Injective a) :
  ¬ (∀ i j, i ≠ j → a i * a j ∣ P (a i) * P (a j)) :=
sorry

end no_distinct_positive_integers_l95_95382


namespace min_value_of_algebraic_sum_l95_95310

theorem min_value_of_algebraic_sum 
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : a + 3 * b = 3) :
  ∃ (min_value : ℝ), min_value = 16 / 3 ∧ (∀ a b, a > 0 → b > 0 → a + 3 * b = 3 → 1 / a + 3 / b ≥ min_value) :=
sorry

end min_value_of_algebraic_sum_l95_95310


namespace combined_motion_properties_l95_95465

noncomputable def y (x : ℝ) := Real.sin x + (Real.sin x) ^ 2

theorem combined_motion_properties :
  (∀ x: ℝ, - (1/4: ℝ) ≤ y x ∧ y x ≤ 2) ∧ 
  (∃ x: ℝ, y x = 2) ∧
  (∃ x: ℝ, y x = -(1/4: ℝ)) :=
by
  -- The complete proofs for these statements are omitted.
  -- This theorem specifies the required properties of the function y.
  sorry

end combined_motion_properties_l95_95465


namespace marbles_total_l95_95368

def marbles_initial := 22
def marbles_given := 20

theorem marbles_total : marbles_initial + marbles_given = 42 := by
  sorry

end marbles_total_l95_95368


namespace cube_volume_surface_area_value_l95_95045

theorem cube_volume_surface_area_value (x : ℝ) : 
  (∃ s : ℝ, s = (6 * x)^(1 / 3) ∧ 6 * s^2 = 2 * x) → 
  x = 1 / 972 :=
by {
  sorry
}

end cube_volume_surface_area_value_l95_95045


namespace sum_of_altitudes_of_triangle_l95_95116

-- Define the line equation as a condition
def line_eq (x y : ℝ) : Prop := 15 * x + 8 * y = 120

-- Define the triangle formed by the line with the coordinate axes
def forms_triangle_with_axes (x y : ℝ) : Prop := 
  line_eq x 0 ∧ line_eq 0 y

-- Prove the sum of the lengths of the altitudes is 511/17
theorem sum_of_altitudes_of_triangle : 
  ∃ x y : ℝ, forms_triangle_with_axes x y → 
  15 + 8 + (120 / 17) = 511 / 17 :=
by
  sorry

end sum_of_altitudes_of_triangle_l95_95116


namespace solve_expression_l95_95619

theorem solve_expression :
  (27 ^ (2 / 3) - 2 ^ (Real.log 3 / Real.log 2) * (Real.logb 2 (1 / 8)) +
    Real.logb 10 4 + Real.logb 10 25 = 20) :=
by
  sorry

end solve_expression_l95_95619


namespace overall_average_is_63_point_4_l95_95467

theorem overall_average_is_63_point_4 : 
  ∃ (n total_marks : ℕ) (avg_marks : ℚ), 
  n = 50 ∧ 
  (∃ (marks_group1 marks_group2 marks_group3 marks_remaining : ℕ), 
    marks_group1 = 6 * 95 ∧
    marks_group2 = 4 * 0 ∧
    marks_group3 = 10 * 80 ∧
    marks_remaining = (n - 20) * 60 ∧
    total_marks = marks_group1 + marks_group2 + marks_group3 + marks_remaining) ∧ 
  avg_marks = total_marks / n ∧ 
  avg_marks = 63.4 := 
by 
  sorry

end overall_average_is_63_point_4_l95_95467


namespace shaded_area_inequality_l95_95299

theorem shaded_area_inequality 
    (A : ℝ) -- All three triangles have the same total area, A.
    {a1 a2 a3 : ℝ} -- a1, a2, a3 are the shaded areas of Triangle I, II, and III respectively.
    (h1 : a1 = A / 6) 
    (h2 : a2 = A / 2) 
    (h3 : a3 = (2 * A) / 3) : 
    a1 ≠ a2 ∧ a1 ≠ a3 ∧ a2 ≠ a3 :=
by
  -- Proof steps would go here, but they are not required as per the instructions
  sorry

end shaded_area_inequality_l95_95299


namespace maximize_probability_sum_8_l95_95625

def L : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12]

theorem maximize_probability_sum_8 :
  (∀ x ∈ L, x ≠ 4 → (∃ y ∈ (List.erase L x), y = 8 - x)) ∧ 
  (∀ y ∈ List.erase L 4, ¬(∃ x ∈ List.erase L 4, x + y = 8)) :=
sorry

end maximize_probability_sum_8_l95_95625


namespace number_of_ways_to_choose_committee_l95_95514

-- Definitions of the conditions
def eligible_members : ℕ := 30
def new_members : ℕ := 3
def committee_size : ℕ := 5
def eligible_pool : ℕ := eligible_members - new_members

-- Problem statement to prove
theorem number_of_ways_to_choose_committee : (Nat.choose eligible_pool committee_size) = 80730 := by
  -- This space is reserved for the proof which is not required per instructions.
  sorry

end number_of_ways_to_choose_committee_l95_95514


namespace square_of_binomial_example_l95_95456

theorem square_of_binomial_example : (23^2 + 2 * 23 * 2 + 2^2 = 625) :=
by
  sorry

end square_of_binomial_example_l95_95456


namespace total_books_proof_l95_95885

-- Define the number of books Lily finished last month.
def books_last_month : ℕ := 4

-- Define the number of books Lily wants to finish this month.
def books_this_month : ℕ := books_last_month * 2

-- Define the total number of books Lily will finish in two months.
def total_books_two_months : ℕ := books_last_month + books_this_month

-- Theorem to prove the total number of books Lily will finish in two months is 12.
theorem total_books_proof : total_books_two_months = 12 := by
  -- Here would be the proof steps.
  sorry

end total_books_proof_l95_95885


namespace cloth_cost_price_l95_95041

theorem cloth_cost_price
  (meters_of_cloth : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ)
  (total_profit : ℕ) (total_cost_price : ℕ) (cost_price_per_meter : ℕ) :
  meters_of_cloth = 45 →
  selling_price = 4500 →
  profit_per_meter = 14 →
  total_profit = profit_per_meter * meters_of_cloth →
  total_cost_price = selling_price - total_profit →
  cost_price_per_meter = total_cost_price / meters_of_cloth →
  cost_price_per_meter = 86 :=
by
  intros
  sorry

end cloth_cost_price_l95_95041


namespace final_color_all_blue_l95_95685

-- Definitions based on the problem's initial conditions
def initial_blue_sheep : ℕ := 22
def initial_red_sheep : ℕ := 18
def initial_green_sheep : ℕ := 15

-- The final problem statement: prove that all sheep end up being blue
theorem final_color_all_blue (B R G : ℕ) 
  (hB : B = initial_blue_sheep) 
  (hR : R = initial_red_sheep) 
  (hG : G = initial_green_sheep) 
  (interaction : ∀ (B R G : ℕ), (B > 0 ∨ R > 0 ∨ G > 0) → (R ≡ G [MOD 3])) :
  ∃ b, b = B + R + G ∧ R = 0 ∧ G = 0 ∧ b % 3 = 1 ∧ B = b :=
by
  -- Proof to be provided
  sorry

end final_color_all_blue_l95_95685


namespace wombat_clawing_l95_95391

variable (W : ℕ)
variable (R : ℕ := 1)

theorem wombat_clawing :
    (9 * W + 3 * R = 39) → (W = 4) :=
by 
  sorry

end wombat_clawing_l95_95391


namespace geometric_sequence_sum_is_five_eighths_l95_95115

noncomputable def geometric_sequence_sum (a₁ : ℝ) (q : ℝ) : ℝ :=
  if q = 1 then 4 * a₁ else a₁ * (1 - q^4) / (1 - q)

theorem geometric_sequence_sum_is_five_eighths
  (a₁ q : ℝ)
  (h₀ : q ≠ 1)
  (h₁ : a₁ * (a₁ * q) * (a₁ * q^2) = -1 / 8)
  (h₂ : 2 * (a₁ * q^2) = a₁ * q + a₁ * q^2) :
  geometric_sequence_sum a₁ q = 5 / 8 := by
sorry

end geometric_sequence_sum_is_five_eighths_l95_95115


namespace paul_and_paula_cookies_l95_95258

-- Define the number of cookies per pack type
def cookies_in_pack (pack : ℕ) : ℕ :=
  match pack with
  | 1 => 15
  | 2 => 30
  | 3 => 45
  | 4 => 60
  | _ => 0

-- Paul's purchase: 2 packs of Pack B and 1 pack of Pack A
def pauls_cookies : ℕ :=
  2 * cookies_in_pack 2 + cookies_in_pack 1

-- Paula's purchase: 1 pack of Pack A and 1 pack of Pack C
def paulas_cookies : ℕ :=
  cookies_in_pack 1 + cookies_in_pack 3

-- Total number of cookies Paul and Paula have
def total_cookies : ℕ :=
  pauls_cookies + paulas_cookies

theorem paul_and_paula_cookies : total_cookies = 135 :=
by
  sorry

end paul_and_paula_cookies_l95_95258


namespace polynomial_value_at_2_l95_95589

def f (x : ℕ) : ℕ := 8 * x^7 + 5 * x^6 + 3 * x^4 + 2 * x + 1

theorem polynomial_value_at_2 : f 2 = 1397 := by
  sorry

end polynomial_value_at_2_l95_95589


namespace cos_R_in_triangle_PQR_l95_95746

theorem cos_R_in_triangle_PQR
  (P Q R : ℝ) (hP : P = 90) (hQ : Real.sin Q = 3/5)
  (h_sum : P + Q + R = 180) (h_PQ_comp : P + Q = 90) :
  Real.cos R = 3 / 5 := 
sorry

end cos_R_in_triangle_PQR_l95_95746


namespace hyperbola_equation_correct_l95_95214

noncomputable def hyperbola_equation (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (focal_len : Real.sqrt (a^2 + b^2) = 5) (asymptote_slope : b = 2 * a) :=
  (x y : ℝ) -> (x^2 / 5) - (y^2 / 20) = 1

theorem hyperbola_equation_correct {a b : ℝ} (a_pos : 0 < a) (b_pos : 0 < b) 
  (focal_len : Real.sqrt (a^2 + b^2) = 5) (asymptote_slope : b = 2 * a) :
  hyperbola_equation a b a_pos b_pos focal_len asymptote_slope :=
by {
  sorry
}

end hyperbola_equation_correct_l95_95214


namespace Joey_age_l95_95904

-- Define the basic data
def ages : List ℕ := [4, 6, 8, 10, 12]

-- Define the conditions
def cinema_ages (x y : ℕ) : Prop := x + y = 18
def soccer_ages (x y : ℕ) : Prop := x < 11 ∧ y < 11
def stays_home (x : ℕ) : Prop := x = 6

-- The goal is to prove Joey's age
theorem Joey_age : ∃ j, j ∈ ages ∧ stays_home 6 ∧ (∀ x y, cinema_ages x y → x ≠ j ∧ y ≠ j) ∧ 
(∃ x y, soccer_ages x y ∧ x ≠ 6 ∧ y ≠ 6) ∧ j = 8 := by
  sorry

end Joey_age_l95_95904


namespace linear_if_abs_k_eq_1_l95_95701

theorem linear_if_abs_k_eq_1 (k : ℤ) : |k| = 1 ↔ (k = 1 ∨ k = -1) := by
  sorry

end linear_if_abs_k_eq_1_l95_95701


namespace seq_a_general_term_seq_b_general_term_inequality_k_l95_95263

def seq_a (n : ℕ) : ℕ :=
if n = 1 then 2 else 2 * n - 1

def S (n : ℕ) : ℕ := 
match n with
| 0       => 0
| (n + 1) => S n + seq_a (n + 1)

def seq_b (n : ℕ) : ℕ := 3 ^ n

def T (n : ℕ) : ℕ := (3 ^ (n + 1) - 3) / 2

theorem seq_a_general_term (n : ℕ) : seq_a n = if n = 1 then 2 else 2 * n - 1 :=
sorry

theorem seq_b_general_term (n : ℕ) : seq_b n = 3 ^ n :=
sorry

theorem inequality_k (k : ℝ) : (∀ n : ℕ, n > 0 → (T n + 3/2 : ℝ) * k ≥ 3 * n - 6) ↔ k ≥ 2 / 27 :=
sorry

end seq_a_general_term_seq_b_general_term_inequality_k_l95_95263


namespace find_f_4_l95_95166

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x : ℝ, f x + 2 * f (1 - x) = 3 * x^2

theorem find_f_4 : f 4 = 2 := 
by {
    -- The proof is omitted as per the task.
    sorry
}

end find_f_4_l95_95166


namespace find_xy_l95_95100

theorem find_xy (x y : ℝ) (h1 : x + y = 5) (h2 : x^3 + y^3 = 125) : x * y = 0 :=
by
  sorry

end find_xy_l95_95100


namespace quadratic_inequality_l95_95470

-- Defining the quadratic expression
def quadratic_expr (a x : ℝ) : ℝ :=
  (a + 2) * x^2 + 2 * (a + 2) * x + 4

-- Statement to be proven
theorem quadratic_inequality {a : ℝ} :
  (∀ x : ℝ, quadratic_expr a x > 0) ↔ -2 ≤ a ∧ a < 2 :=
by
  sorry -- Proof omitted

end quadratic_inequality_l95_95470


namespace dodecagon_diagonals_l95_95891

/--
The formula for the number of diagonals in a convex n-gon is given by (n * (n - 3)) / 2.
-/
def number_of_diagonals (n : Nat) : Nat := (n * (n - 3)) / 2

/--
A dodecagon has 12 sides.
-/
def dodecagon_sides : Nat := 12

/--
The number of diagonals in a convex dodecagon is 54.
-/
theorem dodecagon_diagonals : number_of_diagonals dodecagon_sides = 54 := by
  sorry

end dodecagon_diagonals_l95_95891


namespace bridge_length_is_correct_l95_95729

noncomputable def train_length : ℝ := 135
noncomputable def train_speed_kmh : ℝ := 45
noncomputable def bridge_crossing_time : ℝ := 30

noncomputable def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600
noncomputable def total_distance_crossed : ℝ := train_speed_ms * bridge_crossing_time
noncomputable def bridge_length : ℝ := total_distance_crossed - train_length

theorem bridge_length_is_correct : bridge_length = 240 := by
  sorry

end bridge_length_is_correct_l95_95729


namespace total_bouncy_balls_l95_95782

def red_packs := 4
def yellow_packs := 8
def green_packs := 4
def balls_per_pack := 10

theorem total_bouncy_balls:
  (red_packs * balls_per_pack + yellow_packs * balls_per_pack + green_packs * balls_per_pack) = 160 :=
by 
  sorry

end total_bouncy_balls_l95_95782


namespace intersection_A_B_l95_95321

noncomputable def A : Set ℝ := { x | (x - 1) / (x + 3) < 0 }
noncomputable def B : Set ℝ := { x | abs x < 2 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | -2 < x ∧ x < 1 } :=
by
  sorry

end intersection_A_B_l95_95321


namespace david_number_sum_l95_95220

theorem david_number_sum :
  ∃ (x y : ℕ), (10 ≤ x ∧ x < 100) ∧ (100 ≤ y ∧ y < 1000) ∧ (1000 * x + y = 4 * x * y) ∧ (x + y = 266) :=
sorry

end david_number_sum_l95_95220


namespace least_positive_integer_x_20y_l95_95607

theorem least_positive_integer_x_20y (x y : ℤ) (h : Int.gcd x (20 * y) = 4) : 
  ∃ k : ℕ, k > 0 ∧ k * (x + 20 * y) = 4 := 
sorry

end least_positive_integer_x_20y_l95_95607


namespace cost_per_pack_l95_95257

theorem cost_per_pack (total_bill : ℕ) (change_given : ℕ) (packs : ℕ) (total_cost := total_bill - change_given) (cost_per_pack := total_cost / packs) 
  (h1 : total_bill = 20) 
  (h2 : change_given = 11) 
  (h3 : packs = 3) : 
  cost_per_pack = 3 := by
  sorry

end cost_per_pack_l95_95257


namespace fraction_of_cream_in_cup1_after_operations_l95_95018

/-
We consider two cups of liquids with the following contents initially:
Cup 1 has 6 ounces of coffee.
Cup 2 has 2 ounces of coffee and 4 ounces of cream.
After pouring half of Cup 1's content into Cup 2, stirring, and then pouring half of Cup 2's new content back into Cup 1, we need to show that 
the fraction of the liquid in Cup 1 that is now cream is 4/15.
-/

theorem fraction_of_cream_in_cup1_after_operations :
  let cup1_initial_coffee := 6
  let cup2_initial_coffee := 2
  let cup2_initial_cream := 4
  let cup2_initial_liquid := cup2_initial_coffee + cup2_initial_cream
  let cup1_to_cup2_coffee := cup1_initial_coffee / 2
  let cup1_final_coffee := cup1_initial_coffee - cup1_to_cup2_coffee
  let cup2_final_coffee := cup2_initial_coffee + cup1_to_cup2_coffee
  let cup2_final_liquid := cup2_final_coffee + cup2_initial_cream
  let cup2_to_cup1_liquid := cup2_final_liquid / 2
  let cup2_coffee_fraction := cup2_final_coffee / cup2_final_liquid
  let cup2_cream_fraction := cup2_initial_cream / cup2_final_liquid
  let cup2_to_cup1_coffee := cup2_to_cup1_liquid * cup2_coffee_fraction
  let cup2_to_cup1_cream := cup2_to_cup1_liquid * cup2_cream_fraction
  let cup1_final_liquid_coffee := cup1_final_coffee + cup2_to_cup1_coffee
  let cup1_final_liquid_cream := cup2_to_cup1_cream
  let cup1_final_liquid := cup1_final_liquid_coffee + cup1_final_liquid_cream
  (cup1_final_liquid_cream / cup1_final_liquid) = 4 / 15 :=
by
  sorry

end fraction_of_cream_in_cup1_after_operations_l95_95018


namespace calc_square_uncovered_area_l95_95488

theorem calc_square_uncovered_area :
  ∀ (side_length : ℕ) (circle_diameter : ℝ) (num_circles : ℕ),
    side_length = 16 →
    circle_diameter = (16 / 3) →
    num_circles = 9 →
    (side_length ^ 2) - num_circles * (Real.pi * (circle_diameter / 2) ^ 2) = 256 - 64 * Real.pi :=
by
  intros side_length circle_diameter num_circles h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end calc_square_uncovered_area_l95_95488


namespace response_rate_percentage_l95_95639

theorem response_rate_percentage (number_of_responses_needed number_of_questionnaires_mailed : ℕ) 
  (h1 : number_of_responses_needed = 300) 
  (h2 : number_of_questionnaires_mailed = 500) : 
  (number_of_responses_needed / number_of_questionnaires_mailed : ℚ) * 100 = 60 :=
by 
  sorry

end response_rate_percentage_l95_95639


namespace total_boys_in_class_l95_95841

theorem total_boys_in_class (n : ℕ)
  (h1 : 19 + 19 - 1 = n) :
  n = 37 :=
  sorry

end total_boys_in_class_l95_95841


namespace select_team_with_smaller_variance_l95_95805

theorem select_team_with_smaller_variance 
    (variance_A variance_B : ℝ)
    (hA : variance_A = 1.5)
    (hB : variance_B = 2.8)
    : variance_A < variance_B → "Team A" = "Team A" :=
by
  intros h
  sorry

end select_team_with_smaller_variance_l95_95805


namespace possible_k_values_l95_95796

variables (p q r s k : ℂ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : s ≠ 0)
          (h5 : p * q = r * s)
          (h6 : p * k ^ 3 + q * k ^ 2 + r * k + s = 0)
          (h7 : q * k ^ 3 + r * k ^ 2 + s * k + p = 0)

noncomputable def roots_of_unity := {k : ℂ | k ^ 4 = 1}

theorem possible_k_values : k ∈ roots_of_unity :=
by {
  sorry
}

end possible_k_values_l95_95796


namespace a_b_total_money_l95_95721

variable (A B : ℝ)

theorem a_b_total_money (h1 : (4 / 15) * A = (2 / 5) * 484) (h2 : B = 484) : A + B = 1210 := by
  sorry

end a_b_total_money_l95_95721


namespace shop_discount_percentage_l95_95092

-- Definitions based on conditions
def original_price := 800
def price_paid := 560
def discount_amount := original_price - price_paid
def percentage_discount := (discount_amount / original_price) * 100

-- Proposition to prove
theorem shop_discount_percentage : percentage_discount = 30 := by
  sorry

end shop_discount_percentage_l95_95092


namespace total_expenditure_is_3500_l95_95203

def expenditure_mon : ℕ := 450
def expenditure_tue : ℕ := 600
def expenditure_wed : ℕ := 400
def expenditure_thurs : ℕ := 500
def expenditure_sat : ℕ := 550
def expenditure_sun : ℕ := 300
def cost_earphone : ℕ := 620
def cost_pen : ℕ := 30
def cost_notebook : ℕ := 50

def expenditure_fri : ℕ := cost_earphone + cost_pen + cost_notebook
def total_expenditure : ℕ := expenditure_mon + expenditure_tue + expenditure_wed + expenditure_thurs + expenditure_fri + expenditure_sat + expenditure_sun

theorem total_expenditure_is_3500 : total_expenditure = 3500 := by
  sorry

end total_expenditure_is_3500_l95_95203


namespace sample_and_size_correct_l95_95740

structure SchoolSurvey :=
  (students_selected : ℕ)
  (classes_selected : ℕ)

def survey_sample (survey : SchoolSurvey) : String :=
  "the physical condition of " ++ toString survey.students_selected ++ " students"

def survey_sample_size (survey : SchoolSurvey) : ℕ :=
  survey.students_selected

theorem sample_and_size_correct (survey : SchoolSurvey)
  (h_selected : survey.students_selected = 190)
  (h_classes : survey.classes_selected = 19) :
  survey_sample survey = "the physical condition of 190 students" ∧ 
  survey_sample_size survey = 190 :=
by
  sorry

end sample_and_size_correct_l95_95740


namespace max_value_fractions_l95_95052

noncomputable def maxFractions (a b c : ℝ) : ℝ :=
  (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c)

theorem max_value_fractions (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
    (h_sum : a + b + c = 2) :
    maxFractions a b c ≤ 1 ∧ 
    (a = 2 / 3 ∧ b = 2 / 3 ∧ c = 2 / 3 → maxFractions a b c = 1) := 
  by
    sorry

end max_value_fractions_l95_95052


namespace total_length_circle_l95_95315

-- Definitions based on conditions
def num_strips : ℕ := 16
def length_each_strip : ℝ := 10.4
def overlap_each_strip : ℝ := 3.5

-- Theorem stating the total length of the circle-shaped colored tape
theorem total_length_circle : 
  (num_strips * length_each_strip) - (num_strips * overlap_each_strip) = 110.4 := 
by 
  sorry

end total_length_circle_l95_95315


namespace smallest_positive_n_l95_95234

theorem smallest_positive_n (n : ℕ) : n > 0 → (3 * n ≡ 1367 [MOD 26]) → n = 5 :=
by
  intros _ _
  sorry

end smallest_positive_n_l95_95234


namespace division_remainder_l95_95122

theorem division_remainder :
  ∃ (r : ℝ), ∀ (z : ℝ), (4 * z^3 - 5 * z^2 - 17 * z + 4) = (4 * z + 6) * (z^2 - 4 * z + 1/2) + r ∧ r = 1 :=
sorry

end division_remainder_l95_95122


namespace eggs_per_week_is_84_l95_95943

-- Define the number of pens
def number_of_pens : Nat := 4

-- Define the number of emus per pen
def emus_per_pen : Nat := 6

-- Define the number of days in a week
def days_in_week : Nat := 7

-- Define the number of eggs per female emu per day
def eggs_per_female_emu_per_day : Nat := 1

-- Calculate the total number of emus
def total_emus : Nat := number_of_pens * emus_per_pen

-- Calculate the number of female emus
def female_emus : Nat := total_emus / 2

-- Calculate the number of eggs per day
def eggs_per_day : Nat := female_emus * eggs_per_female_emu_per_day

-- Calculate the number of eggs per week
def eggs_per_week : Nat := eggs_per_day * days_in_week

-- The theorem to prove
theorem eggs_per_week_is_84 : eggs_per_week = 84 := by
  sorry

end eggs_per_week_is_84_l95_95943


namespace total_earnings_l95_95330

theorem total_earnings (d_a : ℕ) (h : 57 * d_a + 684 + 380 = 1406) : d_a = 6 :=
by {
  -- The proof will involve algebraic manipulations similar to the solution steps
  sorry
}

end total_earnings_l95_95330


namespace dagger_computation_l95_95416

def dagger (m n p q : ℕ) (hn : n ≠ 0) (hm : m ≠ 0) : ℚ :=
  (m^2 * p * (q / n)) + ((p : ℚ) / m)

theorem dagger_computation :
  dagger 5 9 6 2 (by norm_num) (by norm_num) = 518 / 15 :=
sorry

end dagger_computation_l95_95416


namespace ratio_of_cows_to_bulls_l95_95581

-- Define the total number of cattle
def total_cattle := 555

-- Define the number of bulls
def number_of_bulls := 405

-- Compute the number of cows
def number_of_cows := total_cattle - number_of_bulls

-- Define the expected ratio of cows to bulls
def expected_ratio_cows_to_bulls := (10, 27)

-- Prove that the ratio of cows to bulls is equal to the expected ratio
theorem ratio_of_cows_to_bulls : 
  (number_of_cows / (gcd number_of_cows number_of_bulls), number_of_bulls / (gcd number_of_cows number_of_bulls)) = expected_ratio_cows_to_bulls :=
sorry

end ratio_of_cows_to_bulls_l95_95581


namespace problem_equivalence_l95_95242

theorem problem_equivalence : 4 * 4^3 - 16^60 / 16^57 = -3840 := by
  sorry

end problem_equivalence_l95_95242


namespace nancy_hourly_wage_l95_95468

-- Definitions based on conditions
def tuition_per_semester : ℕ := 22000
def parents_contribution : ℕ := tuition_per_semester / 2
def scholarship_amount : ℕ := 3000
def loan_amount : ℕ := 2 * scholarship_amount
def nancy_contributions : ℕ := parents_contribution + scholarship_amount + loan_amount
def remaining_tuition : ℕ := tuition_per_semester - nancy_contributions
def total_working_hours : ℕ := 200

-- Theorem to prove based on the formulated problem
theorem nancy_hourly_wage :
  (remaining_tuition / total_working_hours) = 10 :=
by
  sorry

end nancy_hourly_wage_l95_95468


namespace total_opaque_stackings_l95_95228

-- Define the glass pane and its rotation
inductive Rotation
| deg_0 | deg_90 | deg_180 | deg_270
deriving DecidableEq, Repr

-- The property of opacity for a stack of glass panes
def isOpaque (stack : List (List Rotation)) : Bool :=
  -- The implementation of this part depends on the specific condition in the problem
  -- and here is abstracted out for the problem statement.
  sorry

-- The main problem stating the required number of ways
theorem total_opaque_stackings : ∃ (n : ℕ), n = 7200 :=
  sorry

end total_opaque_stackings_l95_95228


namespace sequence_term_2012_l95_95823

theorem sequence_term_2012 :
  ∃ (a : ℕ → ℤ), a 1 = 3 ∧ a 2 = 6 ∧ (∀ n, a (n + 2) = a (n + 1) - a n) ∧ a 2012 = 6 :=
sorry

end sequence_term_2012_l95_95823


namespace solve_equation_l95_95515

theorem solve_equation (x : ℝ) (h₀ : x ≠ 1) (h₁ : x ≠ -1) :
  ( -15 * x / (x^2 - 1) = 3 * x / (x + 1) - 9 / (x - 1) + 1 )
  ↔ x = 5 / 4 ∨ x = -2 :=
by sorry

end solve_equation_l95_95515


namespace circumscribed_circle_diameter_l95_95144

theorem circumscribed_circle_diameter (a : ℝ) (A : ℝ) (h_a : a = 16) (h_A : A = 30) :
    let D := a / Real.sin (A * Real.pi / 180)
    D = 32 := by
  sorry

end circumscribed_circle_diameter_l95_95144


namespace pair_with_gcf_20_l95_95795

theorem pair_with_gcf_20 (a b : ℕ) (h1 : a = 20) (h2 : b = 40) : Nat.gcd a b = 20 := by
  rw [h1, h2]
  sorry

end pair_with_gcf_20_l95_95795


namespace units_digit_of_7_pow_6_pow_5_l95_95727

theorem units_digit_of_7_pow_6_pow_5 : ((7 : ℕ)^ (6^5) % 10) = 1 := by
  sorry

end units_digit_of_7_pow_6_pow_5_l95_95727


namespace only_one_P_Q_l95_95472

def P (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def Q (a : ℝ) : Prop := ∀ x : ℝ, x^2 - x + a = 0

theorem only_one_P_Q (a : ℝ) :
  (P a ∧ ¬ Q a) ∨ (Q a ∧ ¬ P a) ↔
  (a < 0) ∨ (1/4 < a ∧ a < 4) :=
sorry

end only_one_P_Q_l95_95472


namespace find_equation_AC_l95_95271

noncomputable def triangleABC (A B C : (ℝ × ℝ)) : Prop :=
  B = (-2, 0) ∧ 
  ∃ (lineAB : ℝ × ℝ → ℝ), ∀ P, lineAB P = 3 * P.1 - P.2 + 6 

noncomputable def conditions (A B : (ℝ × ℝ)) : Prop :=
  (3 * B.1 - B.2 + 6 = 0) ∧ 
  (B.1 + 3 * B.2 - 26 = 0) ∧
  (A.1 + A.2 - 2 = 0)

noncomputable def equationAC (A C : (ℝ × ℝ)) : Prop :=
  (C.1 - 3 * C.2 + 10 = 0)

theorem find_equation_AC (A B C : (ℝ × ℝ)) (h₁ : triangleABC A B C) (h₂ : conditions A B) : 
  equationAC A C :=
sorry

end find_equation_AC_l95_95271


namespace simplified_fraction_l95_95778

noncomputable def simplify_and_rationalize (a b c d e f : ℝ) : ℝ :=
  (Real.sqrt a / Real.sqrt b) * (Real.sqrt c / Real.sqrt d) * (Real.sqrt e / Real.sqrt f)

theorem simplified_fraction :
  simplify_and_rationalize 3 7 5 9 6 8 = Real.sqrt 35 / 14 :=
by
  sorry

end simplified_fraction_l95_95778


namespace cost_of_five_plastic_chairs_l95_95692

theorem cost_of_five_plastic_chairs (C T : ℕ) (h1 : 3 * C = T) (h2 : T + 2 * C = 55) : 5 * C = 55 :=
by {
  sorry
}

end cost_of_five_plastic_chairs_l95_95692


namespace camera_value_l95_95739

variables (V : ℝ)

def rental_fee_per_week (V : ℝ) := 0.1 * V
def total_rental_fee(V : ℝ) := 4 * rental_fee_per_week V
def johns_share_of_fee(V : ℝ) := 0.6 * (0.4 * total_rental_fee V)

theorem camera_value (h : johns_share_of_fee V = 1200): 
  V = 5000 :=
by
  sorry

end camera_value_l95_95739


namespace equation_of_line_l95_95506

theorem equation_of_line (x y : ℝ) 
  (l1 : 4 * x + y + 6 = 0) 
  (l2 : 3 * x - 5 * y - 6 = 0) 
  (midpoint_origin : ∃ x₁ y₁ x₂ y₂ : ℝ, 
    (4 * x₁ + y₁ + 6 = 0) ∧ 
    (3 * x₂ - 5 * y₂ - 6 = 0) ∧ 
    (x₁ + x₂ = 0) ∧ 
    (y₁ + y₂ = 0)) : 
  7 * x + 4 * y = 0 :=
sorry

end equation_of_line_l95_95506


namespace number_of_valid_N_count_valid_N_is_seven_l95_95140

theorem number_of_valid_N (N : ℕ) :  ∃ (m : ℕ), (m > 3) ∧ (m ∣ 48) ∧ (N = m - 3) := sorry

theorem count_valid_N_is_seven : 
  (∃ f : Fin 7 → ℕ, ∀ k, ∃ m, (m > 3) ∧ (m ∣ 48) ∧ (f k = m - 3)) ∧
  (∀ g : Fin 8 → ℕ, (∀ k, ∃ m, (m > 3) ∧ (m ∣ 48) ∧ (g k = m - 3)) → False) := sorry

end number_of_valid_N_count_valid_N_is_seven_l95_95140


namespace hyperbola_eccentricity_is_sqrt2_l95_95615

noncomputable def hyperbola_eccentricity (a b : ℝ) (hyp1 : a > 0) (hyp2 : b > 0) 
(hyp3 : b = a) : ℝ :=
    let c := Real.sqrt (2) * a
    c / a

theorem hyperbola_eccentricity_is_sqrt2 
(a b : ℝ) (hyp1 : a > 0) (hyp2 : b > 0) (hyp3 : b = a) :
hyperbola_eccentricity a b hyp1 hyp2 hyp3 = Real.sqrt 2 := sorry

end hyperbola_eccentricity_is_sqrt2_l95_95615


namespace dot_product_is_ten_l95_95718

-- Define the vectors
def a : ℝ × ℝ := (1, -2)
def b (m : ℝ) : ℝ × ℝ := (2, m)

-- Define the condition that the vectors are parallel
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 / v2.1 = v1.2 / v2.2

-- The main theorem statement
theorem dot_product_is_ten (m : ℝ) (h : parallel a (b m)) : 
  a.1 * (b m).1 + a.2 * (b m).2 = 10 := by
  sorry

end dot_product_is_ten_l95_95718


namespace geo_seq_sum_S4_l95_95527

noncomputable def geom_seq_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geo_seq_sum_S4 {a : ℝ} {q : ℝ} (h1 : a * q^2 - a = 15) (h2 : a * q - a = 5) :
  geom_seq_sum a q 4 = 75 :=
by
  sorry

end geo_seq_sum_S4_l95_95527


namespace mike_peaches_l95_95697

theorem mike_peaches (initial_peaches picked_peaches : ℝ) (h1 : initial_peaches = 34.0) (h2 : picked_peaches = 86.0) : initial_peaches + picked_peaches = 120.0 :=
by
  rw [h1, h2]
  norm_num

end mike_peaches_l95_95697


namespace number_of_positive_solutions_l95_95204

theorem number_of_positive_solutions (x y z : ℕ) (h_cond : x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 12) :
    ∃ (n : ℕ), n = 55 :=
by 
  sorry

end number_of_positive_solutions_l95_95204


namespace correctStatements_l95_95327

-- Definitions based on conditions
def isFunctionalRelationshipDeterministic (S1 : Prop) := 
  S1 = true

def isCorrelationNonDeterministic (S2 : Prop) := 
  S2 = true

def regressionAnalysisFunctionalRelation (S3 : Prop) :=
  S3 = false

def regressionAnalysisCorrelation (S4 : Prop) :=
  S4 = true

-- The translated proof problem statement
theorem correctStatements :
  ∀ (S1 S2 S3 S4 : Prop), 
    isFunctionalRelationshipDeterministic S1 →
    isCorrelationNonDeterministic S2 →
    regressionAnalysisFunctionalRelation S3 →
    regressionAnalysisCorrelation S4 →
    (S1 ∧ S2 ∧ ¬S3 ∧ S4) →
    (S1 ∧ S2 ∧ ¬S3 ∧ S4) = (true ∧ true ∧ true ∧ true) :=
by
  intros S1 S2 S3 S4 H1 H2 H3 H4 H5
  sorry

end correctStatements_l95_95327


namespace xyz_value_l95_95931

theorem xyz_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x + 1/y = 5) (h2 : y + 1/z = 3) (h3 : z + 1/x = 2) :
  x * y * z = 10 + 3 * Real.sqrt 11 :=
by
  sorry

end xyz_value_l95_95931


namespace ben_has_10_fewer_stickers_than_ryan_l95_95666

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

end ben_has_10_fewer_stickers_than_ryan_l95_95666


namespace extra_cost_from_online_purchase_l95_95401

-- Define the in-store price
def inStorePrice : ℝ := 150.00

-- Define the online payment and processing fee
def onlinePayment : ℝ := 35.00
def processingFee : ℝ := 12.00

-- Calculate the total online cost
def totalOnlineCost : ℝ := (4 * onlinePayment) + processingFee

-- Calculate the difference in cents
def differenceInCents : ℝ := (totalOnlineCost - inStorePrice) * 100

-- The proof statement
theorem extra_cost_from_online_purchase : differenceInCents = 200 :=
by
  -- Proof steps go here
  sorry

end extra_cost_from_online_purchase_l95_95401


namespace base_conversion_l95_95294

theorem base_conversion (b : ℕ) (h : 1 * 6^2 + 4 * 6 + 2 = 2 * b^2 + b + 5) : b = 5 :=
by
  sorry

end base_conversion_l95_95294


namespace school_boys_number_l95_95654

theorem school_boys_number (B G : ℕ) (h1 : B / G = 5 / 13) (h2 : G = B + 80) : B = 50 :=
by
  sorry

end school_boys_number_l95_95654


namespace largest_n_for_divisibility_l95_95426

theorem largest_n_for_divisibility (n : ℕ) (h : (n + 20) ∣ (n^3 + 1000)) : n ≤ 180 := 
sorry

example : ∃ n : ℕ, (n + 20) ∣ (n^3 + 1000) ∧ n = 180 :=
by
  use 180
  sorry

end largest_n_for_divisibility_l95_95426


namespace largest_possible_value_n_l95_95562

theorem largest_possible_value_n (n : ℕ) (h : ∀ m : ℕ, m ≠ n → n % m = 0 → m ≤ 35) : n = 35 :=
sorry

end largest_possible_value_n_l95_95562


namespace total_surface_area_l95_95817

theorem total_surface_area (r h : ℝ) (pi : ℝ) (area_base : ℝ) (curved_area_hemisphere : ℝ) (lateral_area_cylinder : ℝ) :
  (pi * r^2 = 144 * pi) ∧ (h = 10) ∧ (curved_area_hemisphere = 2 * pi * r^2) ∧ (lateral_area_cylinder = 2 * pi * r * h) →
  (curved_area_hemisphere + lateral_area_cylinder + area_base = 672 * pi) :=
by
  sorry

end total_surface_area_l95_95817


namespace largest_constant_inequality_l95_95973

theorem largest_constant_inequality :
  ∃ C, C = 3 ∧
  (∀ (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ),
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 ≥ 
  C * (x₁ * (x₂ + x₃) + x₂ * (x₃ + x₄) + x₃ * (x₄ + x₅) + x₄ * (x₅ + x₆) + x₅ * (x₆ + x₁) + x₆ * (x₁ + x₂))) :=

sorry

end largest_constant_inequality_l95_95973


namespace owen_profit_l95_95159

/-- Given the initial purchases and sales, calculate Owen's overall profit. -/
theorem owen_profit :
  let boxes_9_dollars := 8
  let boxes_12_dollars := 4
  let cost_9_dollars := 9
  let cost_12_dollars := 12
  let masks_per_box := 50
  let packets_25_pieces := 100
  let price_25_pieces := 5
  let packets_100_pieces := 28
  let price_100_pieces := 12
  let remaining_masks1 := 150
  let price_remaining1 := 3
  let remaining_masks2 := 150
  let price_remaining2 := 4
  let total_cost := (boxes_9_dollars * cost_9_dollars) + (boxes_12_dollars * cost_12_dollars)
  let total_repacked_masks := (packets_25_pieces * price_25_pieces) + (packets_100_pieces * price_100_pieces)
  let total_remaining_masks := (remaining_masks1 * price_remaining1) + (remaining_masks2 * price_remaining2)
  let total_revenue := total_repacked_masks + total_remaining_masks
  let overall_profit := total_revenue - total_cost
  overall_profit = 1766 := by
  sorry

end owen_profit_l95_95159


namespace solve_quadratic_equation_l95_95945

theorem solve_quadratic_equation (x : ℝ) : x^2 - 4*x + 3 = 0 ↔ (x = 1 ∨ x = 3) := 
by 
  sorry

end solve_quadratic_equation_l95_95945


namespace children_playing_tennis_l95_95302

theorem children_playing_tennis
  (Total : ℕ) (S : ℕ) (N : ℕ) (B : ℕ) (T : ℕ) 
  (hTotal : Total = 38) (hS : S = 21) (hN : N = 10) (hB : B = 12) :
  T = 38 - 21 + 12 - 10 :=
by
  sorry

end children_playing_tennis_l95_95302


namespace sufficient_condition_l95_95896

theorem sufficient_condition (a : ℝ) (h : a > 0) : a^2 + a ≥ 0 :=
sorry

end sufficient_condition_l95_95896


namespace multiply_polynomials_l95_95433

open Polynomial

variable {R : Type*} [CommRing R]

theorem multiply_polynomials (x : R) :
  (x^4 + 6*x^2 + 9) * (x^2 - 3) = x^4 + 6*x^2 :=
  sorry

end multiply_polynomials_l95_95433


namespace arithmetic_sequence_second_term_l95_95773

theorem arithmetic_sequence_second_term (S₃: ℕ) (a₁: ℕ) (h1: S₃ = 9) (h2: a₁ = 1) : 
∃ d a₂, 3 * a₁ + 3 * d = S₃ ∧ a₂ = a₁ + d ∧ a₂ = 3 :=
by
  sorry

end arithmetic_sequence_second_term_l95_95773


namespace nearest_integer_x_sub_y_l95_95240

theorem nearest_integer_x_sub_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : |x| - y = 4) 
  (h2 : |x| * y - x^3 = 1) : 
  abs (x - y - 4) < 1 :=
sorry

end nearest_integer_x_sub_y_l95_95240


namespace tan_of_fourth_quadrant_l95_95680

theorem tan_of_fourth_quadrant (α : ℝ) (h₁ : Real.sin α = -5 / 13) (h₂ : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) : Real.tan α = -5 / 12 :=
sorry

end tan_of_fourth_quadrant_l95_95680


namespace max_n_l95_95063

noncomputable def a (n : ℕ) : ℕ := n

noncomputable def b (n : ℕ) : ℕ := 2 ^ a n

theorem max_n (n : ℕ) (h1 : a 2 = 2) (h2 : ∀ n, b n = 2 ^ a n)
  (h3 : b 4 = 4 * b 2) : n ≤ 9 :=
by 
  sorry

end max_n_l95_95063


namespace smallest_number_among_given_l95_95538

theorem smallest_number_among_given :
  ∀ (a b c d : ℚ), a = -2 → b = -5/2 → c = 0 → d = 1/5 →
  (min (min (min a b) c) d) = b :=
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  sorry

end smallest_number_among_given_l95_95538


namespace angle_B_l95_95200

theorem angle_B (A B C a b c : ℝ) (h : 2 * b * (Real.cos A) = 2 * c - Real.sqrt 3 * a) :
  B = Real.pi / 6 :=
sorry

end angle_B_l95_95200


namespace arithmetic_sequence_properties_l95_95261

-- Definitions and conditions
def S (n : ℕ) : ℤ := -2 * n^2 + 15 * n

-- Statement of the problem as a theorem
theorem arithmetic_sequence_properties :
  (∀ n : ℕ, S (n + 1) - S n = 17 - 4 * (n + 1)) ∧
  (∃ n : ℕ, S n = 28 ∧ ∀ m : ℕ, S m ≤ S n) :=
by {sorry}

end arithmetic_sequence_properties_l95_95261


namespace academy_league_total_games_l95_95288

theorem academy_league_total_games (teams : ℕ) (plays_each_other_twice games_non_conference : ℕ) 
  (h_teams : teams = 8)
  (h_plays_each_other_twice : plays_each_other_twice = 2 * teams * (teams - 1) / 2)
  (h_games_non_conference : games_non_conference = 6 * teams) :
  (plays_each_other_twice + games_non_conference) = 104 :=
by
  sorry

end academy_league_total_games_l95_95288


namespace cookies_in_the_fridge_l95_95517

-- Define the conditions
def total_baked : ℕ := 256
def tim_cookies : ℕ := 15
def mike_cookies : ℕ := 23
def anna_cookies : ℕ := 2 * tim_cookies

-- Define the proof problem
theorem cookies_in_the_fridge : (total_baked - (tim_cookies + mike_cookies + anna_cookies)) = 188 :=
by
  -- insert proof here
  sorry

end cookies_in_the_fridge_l95_95517


namespace parabola_focus_distance_l95_95776

theorem parabola_focus_distance (p : ℝ) (h : 2 * p = 8) : p = 4 :=
  by
  sorry

end parabola_focus_distance_l95_95776


namespace largest_n_digit_number_divisible_by_61_correct_l95_95611

def largest_n_digit_number (n : ℕ) : ℕ :=
10^n - 1

def largest_n_digit_number_divisible_by_61 (n : ℕ) : ℕ :=
largest_n_digit_number n - (largest_n_digit_number n % 61)

theorem largest_n_digit_number_divisible_by_61_correct (n : ℕ) :
  ∃ k : ℕ, largest_n_digit_number_divisible_by_61 n = 61 * k :=
by
  sorry

end largest_n_digit_number_divisible_by_61_correct_l95_95611


namespace quadratic_roots_l95_95737

theorem quadratic_roots (A B C : ℝ) (r s p : ℝ) (h1 : 2 * A * r^2 + 3 * B * r + 4 * C = 0)
  (h2 : 2 * A * s^2 + 3 * B * s + 4 * C = 0) (h3 : r + s = -3 * B / (2 * A)) (h4 : r * s = 2 * C / A) :
  p = (16 * A * C - 9 * B^2) / (4 * A^2) :=
by
  sorry

end quadratic_roots_l95_95737


namespace long_show_episode_duration_is_one_hour_l95_95101

-- Definitions for the given conditions
def total_shows : ℕ := 2
def short_show_length : ℕ := 24
def short_show_episode_duration : ℝ := 0.5
def long_show_episodes : ℕ := 12
def total_viewing_time : ℝ := 24

-- Definition of the length of each episode of the longer show
def long_show_episode_length (L : ℝ) : Prop :=
  (short_show_length * short_show_episode_duration) + (long_show_episodes * L) = total_viewing_time

-- Main statement to prove
theorem long_show_episode_duration_is_one_hour : long_show_episode_length 1 :=
by
  -- Proof placeholder
  sorry

end long_show_episode_duration_is_one_hour_l95_95101


namespace geometric_series_first_term_l95_95786

theorem geometric_series_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 90)
  (hrange : |r| < 1) :
  a = 60 / 11 :=
by 
  sorry

end geometric_series_first_term_l95_95786


namespace sqrt_fourth_root_l95_95422

theorem sqrt_fourth_root (h : Real.sqrt (Real.sqrt (0.00000081)) = 0.1732) : Real.sqrt (Real.sqrt (0.00000081)) = 0.2 :=
by
  sorry

end sqrt_fourth_root_l95_95422


namespace three_pair_probability_l95_95274

theorem three_pair_probability :
  let total_combinations := Nat.choose 52 5
  let three_pair_combinations := 13 * 4 * 12 * 4
  total_combinations = 2598960 ∧ three_pair_combinations = 2496 →
  (three_pair_combinations : ℚ) / total_combinations = 2496 / 2598960 :=
by
  -- Definitions and computations can be added here if necessary
  sorry

end three_pair_probability_l95_95274


namespace total_servings_of_vegetables_l95_95585

def carrot_plant_serving : ℕ := 4
def num_green_bean_plants : ℕ := 10
def num_carrot_plants : ℕ := 8
def num_corn_plants : ℕ := 12
def num_tomato_plants : ℕ := 15
def corn_plant_serving : ℕ := 5 * carrot_plant_serving
def green_bean_plant_serving : ℕ := corn_plant_serving / 2
def tomato_plant_serving : ℕ := carrot_plant_serving + 3

theorem total_servings_of_vegetables :
  (num_carrot_plants * carrot_plant_serving) +
  (num_corn_plants * corn_plant_serving) +
  (num_green_bean_plants * green_bean_plant_serving) +
  (num_tomato_plants * tomato_plant_serving) = 477 := by
  sorry

end total_servings_of_vegetables_l95_95585


namespace angle_C_in_triangle_l95_95801

theorem angle_C_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by 
  sorry

end angle_C_in_triangle_l95_95801


namespace train_speed_in_kmh_l95_95845

-- Definitions from the conditions
def length_of_train : ℝ := 800 -- in meters
def time_to_cross_pole : ℝ := 20 -- in seconds
def conversion_factor : ℝ := 3.6 -- (km/h) per (m/s)

-- Statement to prove the train's speed in km/h
theorem train_speed_in_kmh :
  (length_of_train / time_to_cross_pole * conversion_factor) = 144 :=
  sorry

end train_speed_in_kmh_l95_95845


namespace problem_Ashwin_Sah_l95_95621

def sqrt_int (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

theorem problem_Ashwin_Sah (a b : ℕ) (k : ℤ) (x y : ℕ) :
  (∀ a b : ℕ, ∃ k : ℤ, (a^2 + b^2 + 2 = k * a * b )) →
  (∀ (a b : ℕ), a ≤ b ∨ b < a) →
  (∀ (a b : ℕ), sqrt_int (((k * a) * (k * a) - 4 * (a^2 + 2)))) →
  ∀ (x y : ℕ), (x + y) % 2017 = 24 := by
  sorry

end problem_Ashwin_Sah_l95_95621


namespace perfect_square_formula_l95_95842

theorem perfect_square_formula (x y : ℝ) :
  ¬∃ a b : ℝ, (x^2 + (1/4)*x + (1/4)) = (a + b)^2 ∧
  ¬∃ c d : ℝ, (x^2 + 2*x*y - y^2) = (c + d)^2 ∧
  ¬∃ e f : ℝ, (x^2 + x*y + y^2) = (e + f)^2 ∧
  ∃ g h : ℝ, (4*x^2 + 4*x + 1) = (g + h)^2 :=
sorry

end perfect_square_formula_l95_95842


namespace vectors_parallel_l95_95596

theorem vectors_parallel (m : ℝ) : 
    (∃ k : ℝ, (m, 4) = (k * 5, k * -2)) → m = -10 := 
by
  sorry

end vectors_parallel_l95_95596


namespace aisha_probability_l95_95385

noncomputable def prob_one_head (prob_tail : ℝ) (num_coins : ℕ) : ℝ :=
  1 - (prob_tail ^ num_coins)

theorem aisha_probability : 
  prob_one_head (1/2) 4 = 15 / 16 := 
by 
  sorry

end aisha_probability_l95_95385


namespace count_divisible_by_90_four_digit_numbers_l95_95354

theorem count_divisible_by_90_four_digit_numbers :
  ∃ (n : ℕ), (n = 10) ∧ (∀ (x : ℕ), 1000 ≤ x ∧ x < 10000 ∧ x % 90 = 0 ∧ x % 100 = 90 → (x = 1890 ∨ x = 2790 ∨ x = 3690 ∨ x = 4590 ∨ x = 5490 ∨ x = 6390 ∨ x = 7290 ∨ x = 8190 ∨ x = 9090 ∨ x = 9990)) :=
by
  sorry

end count_divisible_by_90_four_digit_numbers_l95_95354


namespace inequality_solution_l95_95683

theorem inequality_solution (a b : ℝ)
  (h : ∀ x : ℝ, 0 ≤ x → (x^2 + 1 ≥ a * x + b ∧ a * x + b ≥ (3 / 2) * x^(2 / 3) )) :
  (2 - Real.sqrt 2) / 4 ≤ b ∧ b ≤ (2 + Real.sqrt 2) / 4 ∧
  (1 / Real.sqrt (2 * b)) ≤ a ∧ a ≤ 2 * Real.sqrt (1 - b) :=
  sorry

end inequality_solution_l95_95683


namespace range_of_a_l95_95693

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - 2| ≥ a) → a ≤ 3 :=
by
  intro h
  sorry

end range_of_a_l95_95693


namespace mutually_exclusive_events_l95_95747

/-- A group consists of 3 boys and 2 girls. Two students are to be randomly selected to participate in a speech competition. -/
def num_boys : ℕ := 3
def num_girls : ℕ := 2
def total_selected : ℕ := 2

/-- Possible events under consideration:
  A*: Exactly one boy is selected or exactly two girls are selected -/
def is_boy (s : ℕ) (boys : ℕ) : Prop := s ≤ boys 
def is_girl (s : ℕ) (girls : ℕ) : Prop := s ≤ girls
def one_boy_selected (selected : ℕ) (boys : ℕ) := selected = 1 ∧ is_boy selected boys
def two_girls_selected (selected : ℕ) (girls : ℕ) := selected = 2 ∧ is_girl selected girls

theorem mutually_exclusive_events 
  (selected_boy : ℕ) (selected_girl : ℕ) :
  one_boy_selected selected_boy num_boys ∧ selected_boy + selected_girl = total_selected 
  ∧ two_girls_selected selected_girl num_girls 
  → (one_boy_selected selected_boy num_boys ∨ two_girls_selected selected_girl num_girls) :=
by
  sorry

end mutually_exclusive_events_l95_95747


namespace depth_of_grass_sheet_l95_95809

-- Given conditions
def playground_area : ℝ := 5900
def grass_cost_per_cubic_meter : ℝ := 2.80
def total_cost : ℝ := 165.2

-- Variable to solve for
variable (d : ℝ)

-- Theorem statement
theorem depth_of_grass_sheet
  (h : total_cost = (playground_area * d) * grass_cost_per_cubic_meter) :
  d = 0.01 :=
by
  sorry

end depth_of_grass_sheet_l95_95809


namespace expected_value_is_correct_l95_95874

noncomputable def expected_value_of_heads : ℝ :=
  let penny := 1 / 2 * 1
  let nickel := 1 / 2 * 5
  let dime := 1 / 2 * 10
  let quarter := 1 / 2 * 25
  let half_dollar := 1 / 2 * 50
  (penny + nickel + dime + quarter + half_dollar : ℝ)

theorem expected_value_is_correct : expected_value_of_heads = 45.5 := by
  sorry

end expected_value_is_correct_l95_95874


namespace oil_bill_january_l95_95099

-- Declare the constants for January and February oil bills
variables (J F : ℝ)

-- State the conditions
def condition_1 : Prop := F / J = 3 / 2
def condition_2 : Prop := (F + 20) / J = 5 / 3

-- State the theorem based on the conditions and the target statement
theorem oil_bill_january (h1 : condition_1 F J) (h2 : condition_2 F J) : J = 120 :=
by
  sorry

end oil_bill_january_l95_95099


namespace rectangular_coordinates_of_polar_2_pi_over_3_l95_95110

noncomputable def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem rectangular_coordinates_of_polar_2_pi_over_3 :
  polar_to_rectangular 2 (Real.pi / 3) = (1, Real.sqrt 3) :=
by
  sorry

end rectangular_coordinates_of_polar_2_pi_over_3_l95_95110


namespace sufficient_but_not_necessary_condition_l95_95322

variables (a b : ℝ)

def p : Prop := a > b ∧ b > 1
def q : Prop := a - b < a^2 - b^2

theorem sufficient_but_not_necessary_condition (h : p a b) : q a b :=
  sorry

end sufficient_but_not_necessary_condition_l95_95322


namespace axis_of_symmetry_compare_m_n_range_of_t_for_y1_leq_y2_maximum_value_of_t_l95_95314

-- (1) Axis of symmetry
theorem axis_of_symmetry (t : ℝ) :
  ∀ x y : ℝ, (y = x^2 - 2*t*x + 1) → (x = t) := sorry

-- (2) Comparison of m and n
theorem compare_m_n (t m n : ℝ) :
  (t - 2)^2 - 2*t*(t - 2) + 1 = m*1 →
  (t + 3)^2 - 2*t*(t + 3) + 1 = n*1 →
  n > m := sorry

-- (3) Range of t for y₁ ≤ y₂
theorem range_of_t_for_y1_leq_y2 (t x1 x2 y1 y2 : ℝ) :
  (-1 ≤ x1) → (x1 < 3) → (x2 = 3) → 
  (y1 = x1^2 - 2*t*x1 + 1) → 
  (y2 = x2^2 - 2*t*x2 + 1) → 
  y1 ≤ y2 →
  t ≤ 1 := sorry

-- (4) Maximum value of t
theorem maximum_value_of_t (t y1 y2 : ℝ) :
  (y1 = (t + 1)^2 - 2*t*(t + 1) + 1) →
  (y2 = (2*t - 4)^2 - 2*t*(2*t - 4) + 1) →
  y1 ≥ y2 →
  t = 5 := sorry

end axis_of_symmetry_compare_m_n_range_of_t_for_y1_leq_y2_maximum_value_of_t_l95_95314


namespace symmetric_points_sum_l95_95597

theorem symmetric_points_sum (a b : ℝ) (P Q : ℝ × ℝ) 
    (hP : P = (3, a)) (hQ : Q = (b, 2))
    (symm : P = (-Q.1, Q.2)) : a + b = -1 := by
  sorry

end symmetric_points_sum_l95_95597


namespace find_arith_seq_params_l95_95336

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- The conditions given in the problem
theorem find_arith_seq_params :
  ∃ a d : ℤ, 
  (arithmetic_sequence a d 8) = 5 * (arithmetic_sequence a d 1) ∧
  (arithmetic_sequence a d 12) = 2 * (arithmetic_sequence a d 5) + 5 ∧
  a = 3 ∧
  d = 4 :=
by
  sorry

end find_arith_seq_params_l95_95336


namespace employee_B_paid_l95_95736

variable (A B : ℝ)

/-- Two employees A and B are paid a total of Rs. 550 per week by their employer. 
A is paid 120 percent of the sum paid to B. -/
theorem employee_B_paid (h₁ : A + B = 550) (h₂ : A = 1.2 * B) : B = 250 := by
  -- Proof will go here
  sorry

end employee_B_paid_l95_95736


namespace tin_silver_ratio_l95_95864

/-- Assuming a metal bar made of an alloy of tin and silver weighs 40 kg, 
    and loses 4 kg in weight when submerged in water,
    where 10 kg of tin loses 1.375 kg in water and 5 kg of silver loses 0.375 kg, 
    prove that the ratio of tin to silver in the bar is 2 : 3. -/
theorem tin_silver_ratio :
  ∃ (T S : ℝ), 
    T + S = 40 ∧ 
    0.1375 * T + 0.075 * S = 4 ∧ 
    T / S = 2 / 3 := 
by
  sorry

end tin_silver_ratio_l95_95864


namespace rachel_older_than_leah_l95_95081

theorem rachel_older_than_leah (rachel_age leah_age : ℕ) (h1 : rachel_age = 19) (h2 : rachel_age + leah_age = 34) :
  rachel_age - leah_age = 4 :=
by sorry

end rachel_older_than_leah_l95_95081


namespace goldfish_count_equal_in_6_months_l95_95781

def initial_goldfish_brent : ℕ := 3
def initial_goldfish_gretel : ℕ := 243

def goldfish_brent (n : ℕ) : ℕ := initial_goldfish_brent * 4^n
def goldfish_gretel (n : ℕ) : ℕ := initial_goldfish_gretel * 3^n

theorem goldfish_count_equal_in_6_months : 
  (∃ n : ℕ, goldfish_brent n = goldfish_gretel n) ↔ n = 6 :=
by
  sorry

end goldfish_count_equal_in_6_months_l95_95781


namespace cos_alpha_add_beta_over_2_l95_95019

variable (α β : ℝ)

-- Conditions
variables (h1 : 0 < α ∧ α < π / 2)
variables (h2 : -π / 2 < β ∧ β < 0)
variables (h3 : Real.cos (π / 4 + α) = 1 / 3)
variables (h4 : Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3)

-- Result
theorem cos_alpha_add_beta_over_2 :
  Real.cos (α + β / 2) = 5 * Real.sqrt 3 / 9 :=
sorry

end cos_alpha_add_beta_over_2_l95_95019


namespace trapezoid_area_l95_95014

def isosceles_triangle (Δ : Type) (A B C : Δ) : Prop :=
  -- Define the property that triangle ABC is isosceles with AB = AC
  sorry

def similar_triangles (Δ₁ Δ₂ : Type) (A₁ B₁ C₁ : Δ₁) (A₂ B₂ C₂ : Δ₂) : Prop :=
  -- Define the property that triangles Δ₁ and Δ₂ are similar
  sorry

def area (Δ : Type) (A B C : Δ) : ℝ :=
  -- Define the area of a triangle Δ with vertices A, B, and C
  sorry

theorem trapezoid_area
  (Δ : Type)
  {A B C D E : Δ}
  (ABC_is_isosceles : isosceles_triangle Δ A B C)
  (all_similar : ∀ (Δ₁ Δ₂ : Type) (A₁ B₁ C₁ : Δ₁) (A₂ B₂ C₂ : Δ₂), 
    similar_triangles Δ₁ Δ₂ A₁ B₁ C₁ A₂ B₂ C₂ → (area Δ₁ A₁ B₁ C₁ = 1 → area Δ₂ A₂ B₂ C₂ = 1))
  (smallest_triangles_area : area Δ A B C = 50)
  (area_ADE : area Δ A D E = 5) :
  area Δ D B C + area Δ C E B = 45 := 
sorry

end trapezoid_area_l95_95014


namespace salt_weight_l95_95153

theorem salt_weight {S : ℝ} (h1 : 16 + S = 46) : S = 30 :=
by
  sorry

end salt_weight_l95_95153


namespace class_students_l95_95317

theorem class_students :
  ∃ n : ℕ,
    (∃ m : ℕ, 2 * m = n) ∧
    (∃ q : ℕ, 4 * q = n) ∧
    (∃ l : ℕ, 7 * l = n) ∧
    (∀ f : ℕ, f < 6 → n - (n / 2) - (n / 4) - (n / 7) = f) ∧
    n = 28 :=
by
  sorry

end class_students_l95_95317


namespace domain_of_f_lg_x_l95_95865

theorem domain_of_f_lg_x : 
  ({x : ℝ | -1 ≤ x ∧ x ≤ 1} = {x | 10 ≤ x ∧ x ≤ 100}) ↔ (∃ f : ℝ → ℝ, ∀ x ∈ {x : ℝ | -1 ≤ x ∧ x ≤ 1}, f (x * x + 1) = f (Real.log x)) :=
sorry

end domain_of_f_lg_x_l95_95865


namespace find_f_of_13_l95_95794

def f : ℤ → ℤ := sorry  -- We define f as a function from integers to integers

theorem find_f_of_13 : 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x k : ℤ, f (x + 4 * k) = f x) ∧ 
  (f (-1) = 2) → 
  f 13 = -2 := 
by 
  sorry

end find_f_of_13_l95_95794


namespace semicircle_radius_l95_95479

theorem semicircle_radius (π : ℝ) (P : ℝ) (r : ℝ) (hπ : π ≠ 0) (hP : P = 162) (hPerimeter : P = π * r + 2 * r) : r = 162 / (π + 2) :=
by
  sorry

end semicircle_radius_l95_95479


namespace range_of_a_l95_95686

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + 1| + |x - a| < 4) ↔ (-5 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l95_95686


namespace range_of_a_sufficient_but_not_necessary_condition_l95_95921

theorem range_of_a_sufficient_but_not_necessary_condition (a : ℝ) : 
  (-2 < x ∧ x < -1) → ((x + a) * (x + 1) < 0) → (a > 2) :=
sorry

end range_of_a_sufficient_but_not_necessary_condition_l95_95921


namespace zhiqiang_series_l95_95802

theorem zhiqiang_series (a b : ℝ) (n : ℕ) (n_pos : 0 < n) (h : a * b = 1) (h₀ : b ≠ 1):
  (1 + a^n) / (1 + b^n) = ((1 + a) / (1 + b)) ^ n :=
by
  sorry

end zhiqiang_series_l95_95802


namespace fair_coin_heads_probability_l95_95094

theorem fair_coin_heads_probability
  (fair_coin : ∀ n : ℕ, (∀ (heads tails : ℕ), heads + tails = n → (heads / n = 1 / 2) ∧ (tails / n = 1 / 2)))
  (n : ℕ)
  (heads : ℕ)
  (tails : ℕ)
  (h1 : n = 20)
  (h2 : heads = 8)
  (h3 : tails = 12)
  (h4 : heads + tails = n)
  : heads / n = 1 / 2 :=
by
  sorry

end fair_coin_heads_probability_l95_95094


namespace A_finish_work_in_6_days_l95_95262

theorem A_finish_work_in_6_days :
  ∃ (x : ℕ), (1 / (12:ℚ) + 1 / (x:ℚ) = 1 / (4:ℚ)) → x = 6 :=
by
  sorry

end A_finish_work_in_6_days_l95_95262


namespace inverse_of_49_mod_89_l95_95012

theorem inverse_of_49_mod_89 (h : (7 * 55 ≡ 1 [MOD 89])) : (49 * 1 ≡ 1 [MOD 89]) := 
by
  sorry

end inverse_of_49_mod_89_l95_95012


namespace johns_gym_time_l95_95449

noncomputable def time_spent_at_gym (day : String) : ℝ :=
  match day with
  | "Monday" => 1 + 0.5
  | "Tuesday" => 40/60 + 20/60 + 15/60
  | "Thursday" => 40/60 + 20/60 + 15/60
  | "Saturday" => 1.5 + 0.75
  | "Sunday" => 10/60 + 50/60 + 10/60
  | _ => 0

noncomputable def total_hours_per_week : ℝ :=
  time_spent_at_gym "Monday" 
  + 2 * time_spent_at_gym "Tuesday" 
  + time_spent_at_gym "Saturday" 
  + time_spent_at_gym "Sunday"

theorem johns_gym_time : total_hours_per_week = 7.4167 := by
  sorry

end johns_gym_time_l95_95449


namespace area_of_circle_below_line_l95_95393

theorem area_of_circle_below_line (x y : ℝ) :
  (x - 3)^2 + (y - 5)^2 = 9 →
  y ≤ 8 →
  ∃ (A : ℝ), A = 9 * Real.pi :=
sorry

end area_of_circle_below_line_l95_95393


namespace base7_subtraction_l95_95827

theorem base7_subtraction (a b : ℕ) (ha : a = 4 * 7^3 + 3 * 7^2 + 2 * 7 + 1)
                            (hb : b = 1 * 7^3 + 2 * 7^2 + 3 * 7 + 4) :
                            a - b = 3 * 7^3 + 0 * 7^2 + 5 * 7 + 4 :=
by
  sorry

end base7_subtraction_l95_95827


namespace binary_operation_l95_95534

-- Definitions of the binary numbers.
def a : ℕ := 0b10110      -- 10110_2 in base 10
def b : ℕ := 0b10100      -- 10100_2 in base 10
def c : ℕ := 0b10         -- 10_2 in base 10
def result : ℕ := 0b11011100 -- 11011100_2 in base 10

-- The theorem to be proven
theorem binary_operation : (a * b) / c = result := by
  -- Placeholder for the proof
  sorry

end binary_operation_l95_95534


namespace triangle_area_is_17_point_5_l95_95499

-- Define the points A, B, and C as tuples of coordinates
def A : (ℝ × ℝ) := (2, 2)
def B : (ℝ × ℝ) := (7, 2)
def C : (ℝ × ℝ) := (4, 9)

-- Function to calculate the area of a triangle given its vertices
noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1 / 2) * abs ((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)))

-- The theorem statement asserting the area of the triangle is 17.5 square units
theorem triangle_area_is_17_point_5 :
  area_of_triangle A B C = 17.5 :=
by
  sorry -- Proof is omitted

end triangle_area_is_17_point_5_l95_95499


namespace train_cross_bridge_time_l95_95006

noncomputable def time_to_cross_bridge (L_train : ℕ) (v_kmph : ℕ) (L_bridge : ℕ) : ℝ :=
  let v_mps := (v_kmph * 1000) / 3600
  let total_distance := L_train + L_bridge
  total_distance / v_mps

theorem train_cross_bridge_time :
  time_to_cross_bridge 145 54 660 = 53.67 := by
    sorry

end train_cross_bridge_time_l95_95006


namespace second_yellow_probability_l95_95256

-- Define the conditions in Lean
def BagA : Type := {marble : Int // marble ≥ 0}
def BagB : Type := {marble : Int // marble ≥ 0}
def BagC : Type := {marble : Int // marble ≥ 0}
def BagD : Type := {marble : Int // marble ≥ 0}

noncomputable def marbles_in_A := 4 + 5 + 2
noncomputable def marbles_in_B := 7 + 5
noncomputable def marbles_in_C := 3 + 7
noncomputable def marbles_in_D := 8 + 2

-- Probabilities of drawing specific colors from Bag A
noncomputable def prob_white_A := 4 / 11
noncomputable def prob_black_A := 5 / 11
noncomputable def prob_red_A := 2 / 11

-- Probabilities of drawing a yellow marble from Bags B, C and D
noncomputable def prob_yellow_B := 7 / 12
noncomputable def prob_yellow_C := 3 / 10
noncomputable def prob_yellow_D := 8 / 10

-- Expected probability that the second marble is yellow
noncomputable def prob_second_yellow : ℚ :=
  (prob_white_A * prob_yellow_B) + (prob_black_A * prob_yellow_C) + (prob_red_A * prob_yellow_D)

/-- Prove that the total probability the second marble drawn is yellow is 163/330. -/
theorem second_yellow_probability :
  prob_second_yellow = 163 / 330 := sorry

end second_yellow_probability_l95_95256


namespace parallel_segments_have_equal_slopes_l95_95255

theorem parallel_segments_have_equal_slopes
  (A B X : ℝ × ℝ)
  (Y : ℝ × ℝ)
  (hA : A = (-5, -1))
  (hB : B = (2, -8))
  (hX : X = (2, 10))
  (hY1 : Y.1 = 20)
  (h_parallel : (B.2 - A.2) / (B.1 - A.1) = (Y.2 - X.2) / (Y.1 - X.1)) :
  Y.2 = -8 :=
by
  sorry

end parallel_segments_have_equal_slopes_l95_95255


namespace number_of_girls_l95_95430

open Rat

theorem number_of_girls 
  (G B : ℕ) 
  (h1 : G / B = 5 / 8)
  (h2 : G + B = 300) 
  : G = 116 := 
by
  sorry

end number_of_girls_l95_95430


namespace weaving_problem_l95_95724

theorem weaving_problem
  (a : ℕ → ℝ) -- the sequence
  (a_arith_seq : ∀ n, a n = a 0 + n * (a 1 - a 0)) -- arithmetic sequence condition
  (sum_seven_days : 7 * a 0 + 21 * (a 1 - a 0) = 21) -- sum in seven days
  (sum_days_2_5_8 : 3 * a 1 + 12 * (a 1 - a 0) = 15) -- sum on 2nd, 5th, and 8th days
  : a 10 = 15 := sorry

end weaving_problem_l95_95724


namespace sum_of_all_angles_l95_95646

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

end sum_of_all_angles_l95_95646


namespace Sally_bought_20_pokemon_cards_l95_95629

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

end Sally_bought_20_pokemon_cards_l95_95629


namespace Bruce_initial_eggs_l95_95042

variable (B : ℕ)

theorem Bruce_initial_eggs (h : B - 70 = 5) : B = 75 := by
  sorry

end Bruce_initial_eggs_l95_95042


namespace fraction_lost_down_sewer_l95_95059

-- Definitions of the conditions derived from the problem
def initial_marbles := 100
def street_loss_percent := 60 / 100
def sewer_loss := 40 - 20
def remaining_marbles_after_street := initial_marbles - (initial_marbles * street_loss_percent)
def marbles_left := 20

-- The theorem statement proving the fraction of remaining marbles lost down the sewer
theorem fraction_lost_down_sewer :
  (sewer_loss / remaining_marbles_after_street) = 1 / 2 :=
by
  sorry

end fraction_lost_down_sewer_l95_95059


namespace peter_stamps_l95_95329

theorem peter_stamps (M : ℕ) (h1 : M % 5 = 2) (h2 : M % 11 = 2) (h3 : M % 13 = 2) (h4 : M > 1) : M = 717 :=
by
  -- proof will be filled in
  sorry

end peter_stamps_l95_95329


namespace line_equation_with_slope_angle_135_and_y_intercept_neg1_l95_95902

theorem line_equation_with_slope_angle_135_and_y_intercept_neg1 :
  ∃ k b : ℝ, k = -1 ∧ b = -1 ∧ ∀ x y : ℝ, y = k * x + b ↔ y = -x - 1 :=
by
  sorry

end line_equation_with_slope_angle_135_and_y_intercept_neg1_l95_95902


namespace speed_of_stream_l95_95703

theorem speed_of_stream (downstream_speed upstream_speed : ℝ) (h1 : downstream_speed = 14) (h2 : upstream_speed = 8) :
  (downstream_speed - upstream_speed) / 2 = 3 :=
by
  rw [h1, h2]
  norm_num

end speed_of_stream_l95_95703


namespace bob_overtime_pay_rate_l95_95155

theorem bob_overtime_pay_rate :
  let regular_pay_rate := 5
  let total_hours := (44, 48)
  let total_pay := 472
  let overtime_hours (hours : Nat) := max 0 (hours - 40)
  let regular_hours (hours : Nat) := min 40 hours
  let total_regular_hours := regular_hours 44 + regular_hours 48
  let total_regular_pay := total_regular_hours * regular_pay_rate
  let total_overtime_hours := overtime_hours 44 + overtime_hours 48
  let total_overtime_pay := total_pay - total_regular_pay
  let overtime_pay_rate := total_overtime_pay / total_overtime_hours
  overtime_pay_rate = 6 := by sorry

end bob_overtime_pay_rate_l95_95155


namespace canyon_trail_length_l95_95458

theorem canyon_trail_length
  (a b c d e : ℝ)
  (h1 : a + b + c = 36)
  (h2 : b + c + d = 42)
  (h3 : c + d + e = 45)
  (h4 : a + d = 29) :
  a + b + c + d + e = 71 :=
by sorry

end canyon_trail_length_l95_95458


namespace number_of_pupils_l95_95714

theorem number_of_pupils
  (pupil_mark_wrong : ℕ)
  (pupil_mark_correct : ℕ)
  (average_increase : ℚ)
  (n : ℕ)
  (h1 : pupil_mark_wrong = 73)
  (h2 : pupil_mark_correct = 45)
  (h3 : average_increase = 1/2)
  (h4 : 28 / n = average_increase) : n = 56 := 
sorry

end number_of_pupils_l95_95714


namespace problem_l95_95816

theorem problem (a b : ℝ) : a^6 + b^6 ≥ a^4 * b^2 + a^2 * b^4 := 
by sorry

end problem_l95_95816


namespace eulers_formula_l95_95399

structure PlanarGraph :=
(vertices : ℕ)
(edges : ℕ)
(faces : ℕ)
(connected : Prop)

theorem eulers_formula (G: PlanarGraph) (H_conn: G.connected) : G.vertices - G.edges + G.faces = 2 :=
sorry

end eulers_formula_l95_95399


namespace not_function_of_x_l95_95616

theorem not_function_of_x : 
  ∃ x : ℝ, ∃ y1 y2 : ℝ, (|y1| = 2 * x ∧ |y2| = 2 * x ∧ y1 ≠ y2) := sorry

end not_function_of_x_l95_95616


namespace oak_trees_remaining_is_7_l95_95186

-- Define the number of oak trees initially in the park
def initial_oak_trees : ℕ := 9

-- Define the number of oak trees cut down by workers
def oak_trees_cut_down : ℕ := 2

-- Define the remaining oak trees calculation
def remaining_oak_trees : ℕ := initial_oak_trees - oak_trees_cut_down

-- Prove that the remaining oak trees is equal to 7
theorem oak_trees_remaining_is_7 : remaining_oak_trees = 7 := by
  sorry

end oak_trees_remaining_is_7_l95_95186


namespace expression_equivalence_l95_95765

-- Define the initial expression
def expr (w : ℝ) : ℝ := 3 * w + 4 - 2 * w^2 - 5 * w - 6 + w^2 + 7 * w + 8 - 3 * w^2

-- Define the simplified expression
def simplified_expr (w : ℝ) : ℝ := 5 * w - 4 * w^2 + 6

-- Theorem stating the equivalence
theorem expression_equivalence (w : ℝ) : expr w = simplified_expr w :=
by
  -- we would normally simplify and prove here, but we state the theorem and skip the proof for now.
  sorry

end expression_equivalence_l95_95765


namespace tennis_preference_combined_percentage_l95_95004

theorem tennis_preference_combined_percentage :
  let total_north_students := 1500
  let total_south_students := 1800
  let north_tennis_percentage := 0.30
  let south_tennis_percentage := 0.35
  let north_tennis_students := total_north_students * north_tennis_percentage
  let south_tennis_students := total_south_students * south_tennis_percentage
  let total_tennis_students := north_tennis_students + south_tennis_students
  let total_students := total_north_students + total_south_students
  let combined_percentage := (total_tennis_students / total_students) * 100
  combined_percentage = 33 := 
by
  sorry

end tennis_preference_combined_percentage_l95_95004


namespace fred_money_last_week_l95_95533

-- Definitions for the conditions in the problem
variables {f j : ℕ} (current_fred : ℕ) (current_jason : ℕ) (last_week_jason : ℕ)
variable (earning : ℕ)

-- Conditions
axiom Fred_current_money : current_fred = 115
axiom Jason_current_money : current_jason = 44
axiom Jason_last_week_money : last_week_jason = 40
axiom Earning_amount : earning = 4

-- Theorem statement: prove Fred's money last week
theorem fred_money_last_week (current_fred last_week_jason current_jason earning : ℕ)
  (Fred_current_money : current_fred = 115)
  (Jason_current_money : current_jason = 44)
  (Jason_last_week_money : last_week_jason = 40)
  (Earning_amount : earning = 4)
  : current_fred - earning = 111 :=
sorry

end fred_money_last_week_l95_95533


namespace trapezoid_perimeter_and_area_l95_95662

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

end trapezoid_perimeter_and_area_l95_95662


namespace portrait_is_in_Silver_l95_95150

def Gold_inscription (located_in : String → Prop) : Prop := located_in "Gold"
def Silver_inscription (located_in : String → Prop) : Prop := ¬located_in "Silver"
def Lead_inscription (located_in : String → Prop) : Prop := ¬located_in "Gold"

def is_true (inscription : Prop) : Prop := inscription
def is_false (inscription : Prop) : Prop := ¬inscription

noncomputable def portrait_in_Silver_Given_Statements : Prop :=
  ∃ located_in : String → Prop,
    (is_true (Gold_inscription located_in) ∨ is_true (Silver_inscription located_in) ∨ is_true (Lead_inscription located_in)) ∧
    (is_false (Gold_inscription located_in) ∨ is_false (Silver_inscription located_in) ∨ is_false (Lead_inscription located_in)) ∧
    located_in "Silver"

theorem portrait_is_in_Silver : portrait_in_Silver_Given_Statements :=
by {
    sorry
}

end portrait_is_in_Silver_l95_95150


namespace find_s5_l95_95350

noncomputable def s (a b x y : ℝ) (n : ℕ) : ℝ :=
if n = 1 then (a * x + b * y) else
if n = 2 then (a * x^2 + b * y^2) else
if n = 3 then (a * x^3 + b * y^3) else
if n = 4 then (a * x^4 + b * y^4) else
if n = 5 then (a * x^5 + b * y^5) else 0

theorem find_s5 
  (a b x y : ℝ) :
  s a b x y 1 = 5 →
  s a b x y 2 = 11 →
  s a b x y 3 = 24 →
  s a b x y 4 = 58 →
  s a b x y 5 = 262.88 :=
by
  intros h1 h2 h3 h4
  sorry

end find_s5_l95_95350


namespace car_fewer_minutes_than_bus_l95_95078

-- Conditions translated into Lean definitions
def bus_time_to_beach : ℕ := 40
def car_round_trip_time : ℕ := 70

-- Derived condition
def car_one_way_time : ℕ := car_round_trip_time / 2

-- Theorem statement to be proven
theorem car_fewer_minutes_than_bus : car_one_way_time = bus_time_to_beach - 5 := by
  -- This is the placeholder for the proof
  sorry

end car_fewer_minutes_than_bus_l95_95078


namespace N_is_85714_l95_95873

theorem N_is_85714 (N : ℕ) (hN : 10000 ≤ N ∧ N < 100000) 
  (P : ℕ := 200000 + N) 
  (Q : ℕ := 10 * N + 2) 
  (hQ_eq_3P : Q = 3 * P) 
  : N = 85714 := 
by 
  sorry

end N_is_85714_l95_95873


namespace bob_spending_over_limit_l95_95950

theorem bob_spending_over_limit : 
  ∀ (necklace_price book_price limit total_cost amount_over_limit : ℕ),
  necklace_price = 34 →
  book_price = necklace_price + 5 →
  limit = 70 →
  total_cost = necklace_price + book_price →
  amount_over_limit = total_cost - limit →
  amount_over_limit = 3 :=
by
  intros
  sorry

end bob_spending_over_limit_l95_95950


namespace exponential_function_solution_l95_95887

theorem exponential_function_solution (a : ℝ) (h : a > 1)
  (h_max_min_diff : a - a⁻¹ = 1) : a = (Real.sqrt 5 + 1) / 2 :=
sorry

end exponential_function_solution_l95_95887


namespace negation_abs_lt_zero_l95_95967

theorem negation_abs_lt_zero : ¬ (∀ x : ℝ, |x| < 0) ↔ ∃ x : ℝ, |x| ≥ 0 := 
by 
  sorry

end negation_abs_lt_zero_l95_95967


namespace unique_solution_for_exponential_eq_l95_95961

theorem unique_solution_for_exponential_eq (a y : ℕ) (h_a : a ≥ 1) (h_y : y ≥ 1) :
  3^(2*a-1) + 3^a + 1 = 7^y ↔ (a = 1 ∧ y = 1) := by
  sorry

end unique_solution_for_exponential_eq_l95_95961


namespace rectangle_vertex_area_y_value_l95_95832

theorem rectangle_vertex_area_y_value (y : ℕ) (hy : 0 ≤ y) :
  let A := (0, y)
  let B := (10, y)
  let C := (0, 4)
  let D := (10, 4)
  10 * (y - 4) = 90 → y = 13 :=
by
  sorry

end rectangle_vertex_area_y_value_l95_95832


namespace smallest_possible_value_of_n_l95_95971

theorem smallest_possible_value_of_n :
  ∃ n : ℕ, (60 * n = (x + 6) * x * (x + 6) ∧ (x > 0) ∧ gcd 60 n = x + 6) ∧ n = 93 :=
by
  sorry

end smallest_possible_value_of_n_l95_95971


namespace find_c_l95_95035

theorem find_c (a b c : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c)
(h_asc : a < b) (h_asc2 : b < c)
(h_sum : a + b + c = 11)
(h_eq : 1 / a + 1 / b + 1 / c = 1) : c = 6 := 
sorry

end find_c_l95_95035


namespace range_of_m_l95_95174

theorem range_of_m (x y m : ℝ) : (∃ (x y : ℝ), x + y^2 - x + y + m = 0) → m < 1/2 :=
by
  sorry

end range_of_m_l95_95174


namespace green_disks_count_l95_95583

-- Definitions of the conditions given in the problem
def total_disks : ℕ := 14
def red_disks (g : ℕ) : ℕ := 2 * g
def blue_disks (g : ℕ) : ℕ := g / 2

-- The theorem statement to prove
theorem green_disks_count (g : ℕ) (h : 2 * g + g + g / 2 = total_disks) : g = 4 :=
sorry

end green_disks_count_l95_95583


namespace symmetrical_character_l95_95520

def is_symmetrical (char : String) : Prop := 
  sorry  -- Here the definition for symmetry will be elaborated

theorem symmetrical_character : 
  let A : String := "坡"
  let B : String := "上"
  let C : String := "草"
  let D : String := "原"
  is_symmetrical C := 
  sorry

end symmetrical_character_l95_95520


namespace speed_difference_l95_95460

theorem speed_difference (distance : ℝ) (time_heavy : ℝ) (time_no_traffic : ℝ) (d : distance = 200) (th : time_heavy = 5) (tn : time_no_traffic = 4) :
  (distance / time_no_traffic) - (distance / time_heavy) = 10 :=
by
  -- Proof goes here
  sorry

end speed_difference_l95_95460


namespace sum_placed_on_SI_l95_95755

theorem sum_placed_on_SI :
  let P₁ := 4000
  let r₁ := 0.10
  let t₁ := 2
  let CI := P₁ * ((1 + r₁)^t₁ - 1)

  let SI := (1 / 2 * CI : ℝ)
  let r₂ := 0.08
  let t₂ := 3
  let P₂ := SI / (r₂ * t₂)

  P₂ = 1750 :=
by
  sorry

end sum_placed_on_SI_l95_95755


namespace compute_expression_l95_95671

theorem compute_expression : ((-5) * 3) - (7 * (-2)) + ((-4) * (-6)) = 23 := by
  sorry

end compute_expression_l95_95671


namespace grocery_store_total_bottles_l95_95793

def total_bottles (regular_soda : Nat) (diet_soda : Nat) : Nat :=
  regular_soda + diet_soda

theorem grocery_store_total_bottles :
 (total_bottles 9 8 = 17) :=
 by
   sorry

end grocery_store_total_bottles_l95_95793


namespace regular_polygon_sides_l95_95442

theorem regular_polygon_sides (n : ℕ) (h1 : 2 ≤ n) (h2 : (n - 2) * 180 / n = 120) : n = 6 :=
by
  sorry

end regular_polygon_sides_l95_95442


namespace jessica_age_proof_l95_95605

-- Definitions based on conditions
def grandmother_age (j : ℚ) : ℚ := 15 * j
def age_difference (g j : ℚ) : Prop := g - j = 60

-- Proposed age of Jessica
def jessica_age : ℚ := 30 / 7

-- Main statement to prove
theorem jessica_age_proof : ∃ j : ℚ, grandmother_age j = 15 * j ∧ age_difference (grandmother_age j) j ∧ j = jessica_age :=
by sorry

end jessica_age_proof_l95_95605


namespace sequence_general_term_l95_95290

theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = (1 / 2) * a n + 1) :
  ∀ n, a n = 2 - (1 / 2) ^ (n - 1) :=
by
  sorry

end sequence_general_term_l95_95290


namespace men_in_first_group_l95_95541

theorem men_in_first_group (M : ℕ) (h1 : (M * 15) = (M + 0) * 15) (h2 : (15 * 36) = 540) : M = 36 :=
by
  -- Proof would go here
  sorry

end men_in_first_group_l95_95541


namespace no_even_sum_of_four_consecutive_in_circle_l95_95380

theorem no_even_sum_of_four_consecutive_in_circle (n : ℕ) (h1 : n = 2018) :
  ¬ ∃ (f : ℕ → ℕ), (∀ i, 1 ≤ f i ∧ f i ≤ n) ∧ (∀ i, i < n → (f (i % n) + f ((i + 1) % n) + f ((i + 2) % n) + f ((i + 3) % n)) % 2 = 1) :=
by { sorry }

end no_even_sum_of_four_consecutive_in_circle_l95_95380


namespace mod_remainder_l95_95552

open Int

theorem mod_remainder (n : ℤ) : 
  (1125 * 1127 * n) % 12 = 3 ↔ n % 12 = 1 :=
by
  sorry

end mod_remainder_l95_95552


namespace probability_toner_never_displayed_l95_95056

theorem probability_toner_never_displayed:
  let total_votes := 129
  let toner_votes := 63
  let celery_votes := 66
  (toner_votes + celery_votes = total_votes) →
  let probability := (celery_votes - toner_votes) / (celery_votes + toner_votes)
  probability = 1 / 43 := 
by
  sorry

end probability_toner_never_displayed_l95_95056


namespace green_pill_cost_l95_95959

theorem green_pill_cost (p g : ℕ) (h1 : g = p + 1) (h2 : 14 * (p + g) = 546) : g = 20 :=
by
  sorry

end green_pill_cost_l95_95959


namespace intersection_A_B_l95_95400

def A := {y : ℝ | ∃ x : ℝ, y = 2^x}
def B := {y : ℝ | ∃ x : ℝ, y = -x^2 + 2}
def Intersection := {y : ℝ | 0 < y ∧ y ≤ 2}

theorem intersection_A_B :
  (A ∩ B) = Intersection :=
by
  sorry

end intersection_A_B_l95_95400


namespace sin_cos_inequality_for_any_x_l95_95627

noncomputable def largest_valid_n : ℕ := 8

theorem sin_cos_inequality_for_any_x (n : ℕ) (h : n = largest_valid_n) :
  ∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 1 / n :=
sorry

end sin_cos_inequality_for_any_x_l95_95627


namespace steiner_ellipse_equation_l95_95436

theorem steiner_ellipse_equation
  (α β γ : ℝ) 
  (h : α + β + γ = 1) :
  β * γ + α * γ + α * β = 0 := 
sorry

end steiner_ellipse_equation_l95_95436


namespace find_g_of_nine_l95_95292

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_of_nine (h : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = x) : g 9 = 2 :=
by
  sorry

end find_g_of_nine_l95_95292


namespace basis_vetors_correct_options_l95_95355

def is_basis (e1 e2 : ℝ × ℝ) : Prop :=
  e1 ≠ (0, 0) ∧ e2 ≠ (0, 0) ∧ e1.1 * e2.2 - e1.2 * e2.1 ≠ 0

def option_A : ℝ × ℝ := (0, 0)
def option_A' : ℝ × ℝ := (1, 2)

def option_B : ℝ × ℝ := (2, -1)
def option_B' : ℝ × ℝ := (1, 2)

def option_C : ℝ × ℝ := (-1, -2)
def option_C' : ℝ × ℝ := (1, 2)

def option_D : ℝ × ℝ := (1, 1)
def option_D' : ℝ × ℝ := (1, 2)

theorem basis_vetors_correct_options:
  ¬ is_basis option_A option_A' ∧ ¬ is_basis option_C option_C' ∧ 
  is_basis option_B option_B' ∧ is_basis option_D option_D' := 
by
  sorry

end basis_vetors_correct_options_l95_95355


namespace complex_quadrant_l95_95250

open Complex

theorem complex_quadrant (z : ℂ) (h : z = (2 - I) / (2 + I)) : 
  z.re > 0 ∧ z.im < 0 := 
by
  sorry

end complex_quadrant_l95_95250


namespace percentage_answered_first_correctly_l95_95908

variable (A B C D : ℝ)

-- Conditions translated to Lean
variable (hB : B = 0.65)
variable (hC : C = 0.20)
variable (hD : D = 0.60)

-- Statement to prove
theorem percentage_answered_first_correctly (hI : A + B - D = 1 - C) : A = 0.75 := by
  -- import conditions
  rw [hB, hC, hD] at hI
  -- solve the equation
  sorry

end percentage_answered_first_correctly_l95_95908


namespace initial_profit_price_reduction_for_target_profit_l95_95109

-- Define given conditions
def purchase_price : ℝ := 280
def initial_selling_price : ℝ := 360
def items_sold_per_month : ℕ := 60
def target_profit : ℝ := 7200
def increment_per_reduced_yuan : ℕ := 5

-- Problem 1: Prove the initial profit per month before the price reduction
theorem initial_profit : 
  items_sold_per_month * (initial_selling_price - purchase_price) = 4800 := by
sorry

-- Problem 2: Prove that reducing the price by 60 yuan achieves the target profit
theorem price_reduction_for_target_profit : 
  ∃ x : ℝ, 
    ((initial_selling_price - x) - purchase_price) * (items_sold_per_month + (increment_per_reduced_yuan * x)) = target_profit ∧
    x = 60 := by
sorry

end initial_profit_price_reduction_for_target_profit_l95_95109


namespace total_legs_correct_l95_95937

def num_ants : ℕ := 12
def num_spiders : ℕ := 8
def legs_per_ant : ℕ := 6
def legs_per_spider : ℕ := 8
def total_legs := num_ants * legs_per_ant + num_spiders * legs_per_spider

theorem total_legs_correct : total_legs = 136 :=
by
  sorry

end total_legs_correct_l95_95937


namespace problem_part_1_problem_part_2_l95_95511

theorem problem_part_1 
  (p_A : ℚ) (p_B : ℚ)
  (hA : p_A = 2 / 3) 
  (hB : p_B = 3 / 4) : 
  1 - p_A ^ 3 = 19 / 27 :=
by sorry

theorem problem_part_2 
  (p_A : ℚ) (p_B : ℚ)
  (hA : p_A = 2 / 3) 
  (hB : p_B = 3 / 4) 
  (h1 : 3 * (p_A ^ 2) * (1 - p_A) = 4 / 9)
  (h2 : 3 * p_B * ((1 - p_B) ^ 2) = 9 / 64) : 
  (4 / 9) * (9 / 64) = 1 / 16 :=
by sorry

end problem_part_1_problem_part_2_l95_95511


namespace solve_for_a_l95_95783

theorem solve_for_a (a x : ℤ) (h : x + 2 * a = -3) (hx : x = 1) : a = -2 := by
  sorry

end solve_for_a_l95_95783


namespace product_in_A_l95_95788

def A : Set ℤ := { z | ∃ a b : ℤ, z = a^2 + 4 * a * b + b^2 }

theorem product_in_A (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ A := 
by
  sorry

end product_in_A_l95_95788


namespace incorrect_mark_l95_95193

theorem incorrect_mark (n : ℕ) (correct_mark incorrect_entry : ℕ) (average_increase : ℕ) :
  n = 40 → correct_mark = 63 → average_increase = 1/2 →
  incorrect_entry - correct_mark = average_increase * n →
  incorrect_entry = 83 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end incorrect_mark_l95_95193


namespace grading_options_count_l95_95564

theorem grading_options_count :
  (4 ^ 15) = 1073741824 :=
by
  sorry

end grading_options_count_l95_95564


namespace sequence_bound_100_l95_95573

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n ≥ 2, a n = a (n - 1) + 1 / a (n - 1)

theorem sequence_bound_100 (a : ℕ → ℝ) (h : seq a) : 
  14 < a 100 ∧ a 100 < 18 := 
sorry

end sequence_bound_100_l95_95573


namespace investment_problem_l95_95492

theorem investment_problem :
  ∃ (S G : ℝ), S + G = 10000 ∧ 0.06 * G = 0.05 * S + 160 ∧ S = 4000 :=
by
  sorry

end investment_problem_l95_95492


namespace class_duration_l95_95920

theorem class_duration (x : ℝ) (h : 3 * x = 6) : x = 2 :=
by
  sorry

end class_duration_l95_95920


namespace money_sum_l95_95719

theorem money_sum (A B : ℕ) (h₁ : (1 / 3 : ℝ) * A = (1 / 4 : ℝ) * B) (h₂ : B = 484) : A + B = 847 := by
  sorry

end money_sum_l95_95719


namespace celine_change_l95_95285

theorem celine_change
  (price_laptop : ℕ)
  (price_smartphone : ℕ)
  (num_laptops : ℕ)
  (num_smartphones : ℕ)
  (total_money : ℕ)
  (h1 : price_laptop = 600)
  (h2 : price_smartphone = 400)
  (h3 : num_laptops = 2)
  (h4 : num_smartphones = 4)
  (h5 : total_money = 3000) :
  total_money - (num_laptops * price_laptop + num_smartphones * price_smartphone) = 200 :=
by
  sorry

end celine_change_l95_95285


namespace operation_not_equal_33_l95_95642

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

end operation_not_equal_33_l95_95642


namespace find_r_s_l95_95089

noncomputable def r_s_proof_problem (r s : ℕ) (h1 : Nat.gcd r s = 1) (h2 : (r : ℝ) / s = (2 * (Real.sqrt 2 + Real.sqrt 10)) / (5 * Real.sqrt (3 + Real.sqrt 5))) : Prop :=
(r, s) = (4, 5)

theorem find_r_s (r s : ℕ) (h1 : Nat.gcd r s = 1) (h2 : (r : ℝ) / s = (2 * (Real.sqrt 2 + Real.sqrt 10)) / (5 * Real.sqrt (3 + Real.sqrt 5))) : r_s_proof_problem r s h1 h2 :=
sorry

end find_r_s_l95_95089


namespace Caitlin_age_l95_95296

theorem Caitlin_age (Aunt_Anna_age : ℕ) (Brianna_age : ℕ) (Caitlin_age : ℕ)
    (h1 : Aunt_Anna_age = 48)
    (h2 : Brianna_age = Aunt_Anna_age / 3)
    (h3 : Caitlin_age = Brianna_age - 6) : 
    Caitlin_age = 10 := by 
  -- proof here
  sorry

end Caitlin_age_l95_95296


namespace find_numbers_l95_95423

-- Define the conditions
def condition_1 (L S : ℕ) : Prop := L - S = 8327
def condition_2 (L S : ℕ) : Prop := ∃ q r, L = q * S + r ∧ q = 21 ∧ r = 125

-- Define the math proof problem
theorem find_numbers (S L : ℕ) (h1 : condition_1 L S) (h2 : condition_2 L S) : S = 410 ∧ L = 8735 :=
by
  sorry

end find_numbers_l95_95423


namespace diameter_increase_l95_95459

theorem diameter_increase (π : ℝ) (D : ℝ) (A A' D' : ℝ)
  (hA : A = (π / 4) * D^2)
  (hA' : A' = 4 * A)
  (hA'_def : A' = (π / 4) * D'^2) :
  D' = 2 * D :=
by
  sorry

end diameter_increase_l95_95459


namespace parallelogram_height_l95_95682

theorem parallelogram_height (A : ℝ) (b : ℝ) (h : ℝ) (h1 : A = 320) (h2 : b = 20) :
  h = A / b → h = 16 := by
  sorry

end parallelogram_height_l95_95682


namespace presidency_meeting_ways_l95_95319

theorem presidency_meeting_ways :
  let total_schools := 4
  let members_per_school := 4
  let host_school_choices := total_schools
  let choose_3_from_4 := Nat.choose 4 3
  let choose_1_from_4 := Nat.choose 4 1
  let ways_per_host := choose_3_from_4 * choose_1_from_4 ^ 3
  let total_ways := host_school_choices * ways_per_host
  total_ways = 1024 := by
  sorry

end presidency_meeting_ways_l95_95319


namespace relationship_between_exponents_l95_95406

theorem relationship_between_exponents 
  (p r : ℝ) (u v s t m n : ℝ)
  (h1 : p^u = r^s)
  (h2 : r^v = p^t)
  (h3 : m = r^s)
  (h4 : n = r^v)
  (h5 : m^2 = n^3) :
  (s / u = v / t) ∧ (2 * s = 3 * v) :=
  by
  sorry

end relationship_between_exponents_l95_95406


namespace benny_spent_on_baseball_gear_l95_95346

theorem benny_spent_on_baseball_gear (initial_amount left_over spent : ℕ) 
  (h_initial : initial_amount = 67) 
  (h_left : left_over = 33) 
  (h_spent : spent = initial_amount - left_over) : 
  spent = 34 :=
by
  rw [h_initial, h_left] at h_spent
  exact h_spent

end benny_spent_on_baseball_gear_l95_95346


namespace fraction_simplification_l95_95764

theorem fraction_simplification :
  (1^2 + 1) * (2^2 + 1) * (3^2 + 1) / ((2^2 - 1) * (3^2 - 1) * (4^2 - 1)) = 5 / 18 :=
by
  sorry

end fraction_simplification_l95_95764


namespace m_range_l95_95118

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin (2018 * Real.pi - x) * Real.sin (3 * Real.pi / 2 + x) 
  - Real.cos x ^ 2 + 1

def valid_m (m : ℝ) : Prop := 
  ∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2), abs (f x - m) ≤ 1

theorem m_range : 
  ∀ m : ℝ, valid_m m ↔ (m ∈ Set.Icc (1 / 2) ((3 - Real.sqrt 3) / 2)) :=
by sorry

end m_range_l95_95118


namespace ratio_lions_l95_95862

variable (Safari_Lions : Nat)
variable (Safari_Snakes : Nat)
variable (Safari_Giraffes : Nat)
variable (Savanna_Lions_Ratio : ℕ)
variable (Savanna_Snakes : Nat)
variable (Savanna_Giraffes : Nat)
variable (Savanna_Total : Nat)

-- Conditions
def conditions := 
  (Safari_Lions = 100) ∧
  (Safari_Snakes = Safari_Lions / 2) ∧
  (Safari_Giraffes = Safari_Snakes - 10) ∧
  (Savanna_Lions_Ratio * Safari_Lions + Savanna_Snakes + Savanna_Giraffes = Savanna_Total) ∧
  (Savanna_Snakes = 3 * Safari_Snakes) ∧
  (Savanna_Giraffes = Safari_Giraffes + 20) ∧
  (Savanna_Total = 410)

-- Theorem to prove
theorem ratio_lions : conditions Safari_Lions Safari_Snakes Safari_Giraffes Savanna_Lions_Ratio Savanna_Snakes Savanna_Giraffes Savanna_Total → Savanna_Lions_Ratio = 2 := by
  sorry

end ratio_lions_l95_95862


namespace fixer_used_30_percent_kitchen_l95_95031

def fixer_percentage (x : ℝ) : Prop :=
  let initial_nails := 400
  let remaining_after_kitchen := initial_nails * ((100 - x) / 100)
  let remaining_after_fence := remaining_after_kitchen * 0.3
  remaining_after_fence = 84

theorem fixer_used_30_percent_kitchen : fixer_percentage 30 :=
by
  exact sorry

end fixer_used_30_percent_kitchen_l95_95031


namespace shopkeeper_gain_percentage_l95_95236

noncomputable def gain_percentage (false_weight: ℕ) (true_weight: ℕ) : ℝ :=
  (↑(true_weight - false_weight) / ↑false_weight) * 100

theorem shopkeeper_gain_percentage :
  gain_percentage 960 1000 = 4.166666666666667 := 
sorry

end shopkeeper_gain_percentage_l95_95236


namespace find_female_employees_l95_95636

-- Definitions from conditions
def total_employees (E : ℕ) := True
def female_employees (F : ℕ) := True
def male_employees (M : ℕ) := True
def female_managers (F_mgrs : ℕ) := F_mgrs = 280
def fraction_of_managers : ℚ := 2 / 5
def fraction_of_male_managers : ℚ := 2 / 5

-- Statements as conditions in Lean
def managers_total (E M : ℕ) := (fraction_of_managers * E : ℚ) = (fraction_of_male_managers * M : ℚ) + 280
def employees_total (E F M : ℕ) := E = F + M

-- The proof target
theorem find_female_employees (E F M : ℕ) (F_mgrs : ℕ)
    (h1 : female_managers F_mgrs)
    (h2 : managers_total E M)
    (h3 : employees_total E F M) : F = 700 := by
  sorry

end find_female_employees_l95_95636


namespace prime_cond_l95_95516

theorem prime_cond (p q n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hn : n > 1) : 
  (p^(2*n+1) - 1) / (p - 1) = (q^3 - 1) / (q - 1) → (p = 2 ∧ q = 5 ∧ n = 2) :=
  sorry

end prime_cond_l95_95516


namespace factorial_product_square_root_square_l95_95121

theorem factorial_product_square_root_square :
  (Real.sqrt (Nat.factorial 5 * Nat.factorial 4 * Nat.factorial 3))^2 = 17280 := 
by
  sorry

end factorial_product_square_root_square_l95_95121


namespace combinations_with_common_subjects_l95_95744

-- Conditions and known facts
def subjects : Finset String := {"politics", "history", "geography", "physics", "chemistry", "biology", "technology"}
def personA_must_choose : Finset String := {"physics", "politics"}
def personB_cannot_choose : String := "technology"
def total_combinations : Nat := Nat.choose 7 3
def valid_combinations : Nat := Nat.choose 5 1 * Nat.choose 6 3
def non_common_subject_combinations : Nat := 4 + 4

-- We need to prove this statement
theorem combinations_with_common_subjects : valid_combinations - non_common_subject_combinations = 92 := by
  sorry

end combinations_with_common_subjects_l95_95744


namespace solution_set_eq_two_l95_95681

theorem solution_set_eq_two (m : ℝ) (h : ∀ x : ℝ, mx + 2 > 0 ↔ x < 2) :
  m = -1 :=
sorry

end solution_set_eq_two_l95_95681


namespace find_c_in_triangle_l95_95411

theorem find_c_in_triangle
  (A : Real) (a b S : Real) (c : Real)
  (hA : A = 60) 
  (ha : a = 6 * Real.sqrt 3)
  (hb : b = 12)
  (hS : S = 18 * Real.sqrt 3) :
  c = 6 := by
  sorry

end find_c_in_triangle_l95_95411


namespace percentage_error_equals_l95_95493

noncomputable def correct_fraction_calc : ℚ :=
  let num := (3/4 : ℚ) * 16 - (7/8 : ℚ) * 8
  let denom := (3/10 : ℚ) - (1/8 : ℚ)
  num / denom

noncomputable def incorrect_fraction_calc : ℚ :=
  let num := (3/4 : ℚ) * 16 - (7 / 8 : ℚ) * 8
  num * (3/5 : ℚ)

def percentage_error (correct incorrect : ℚ) : ℚ :=
  abs (correct - incorrect) / correct * 100

theorem percentage_error_equals :
  percentage_error correct_fraction_calc incorrect_fraction_calc = 89.47 :=
by
  sorry

end percentage_error_equals_l95_95493


namespace conference_attendees_l95_95986

theorem conference_attendees (w m : ℕ) (h1 : w + m = 47) (h2 : 16 + (w - 1) = m) : w = 16 ∧ m = 31 :=
by
  sorry

end conference_attendees_l95_95986


namespace intersect_of_given_circles_l95_95660

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

end intersect_of_given_circles_l95_95660


namespace triangle_area_202_2192_pi_squared_l95_95027

noncomputable def triangle_area (a b c : ℝ) : ℝ := 
  let r := (a + b + c) / (2 * Real.pi)
  let theta := 20.0 * Real.pi / 180.0  -- converting 20 degrees to radians
  let angle1 := 5 * theta
  let angle2 := 6 * theta
  let angle3 := 7 * theta
  (1 / 2) * r * r * (Real.sin angle1 + Real.sin angle2 + Real.sin angle3)

theorem triangle_area_202_2192_pi_squared (a b c : ℝ) (h1 : a = 5) (h2 : b = 6) (h3 : c = 7) : 
  triangle_area a b c = 202.2192 / (Real.pi * Real.pi) := 
by {
  sorry
}

end triangle_area_202_2192_pi_squared_l95_95027


namespace Grant_room_count_l95_95838

-- Defining the number of rooms in each person's apartments
def Danielle_rooms : ℕ := 6
def Heidi_rooms : ℕ := 3 * Danielle_rooms
def Jenny_rooms : ℕ := Danielle_rooms + 5

-- Combined total rooms
def Total_rooms : ℕ := Danielle_rooms + Heidi_rooms + Jenny_rooms

-- Division operation to determine Grant's room count
def Grant_rooms (total_rooms : ℕ) : ℕ := total_rooms / 9

-- Statement to be proved
theorem Grant_room_count : Grant_rooms Total_rooms = 3 := by
  sorry

end Grant_room_count_l95_95838


namespace derivative_at_one_l95_95207

open Real

noncomputable def f (x : ℝ) : ℝ := exp x / x

theorem derivative_at_one : deriv f 1 = 0 :=
by
  sorry

end derivative_at_one_l95_95207


namespace Grant_score_is_100_l95_95370

/-- Definition of scores --/
def Hunter_score : ℕ := 45

def John_score (H : ℕ) : ℕ := 2 * H

def Grant_score (J : ℕ) : ℕ := J + 10

/-- Theorem to prove Grant's score --/
theorem Grant_score_is_100 : Grant_score (John_score Hunter_score) = 100 := 
  sorry

end Grant_score_is_100_l95_95370


namespace find_x_l95_95128

theorem find_x (x y : ℝ) (h1 : 0.65 * x = 0.20 * y)
  (h2 : y = 617.5 ^ 2 - 42) : 
  x = 117374.3846153846 :=
by
  sorry

end find_x_l95_95128


namespace arrange_abc_l95_95377

noncomputable def a : ℝ := Real.log (4) / Real.log (0.3)
noncomputable def b : ℝ := Real.log (0.2) / Real.log (0.3)
noncomputable def c : ℝ := (1 / Real.exp 1) ^ Real.pi

theorem arrange_abc (a := a) (b := b) (c := c) : b > c ∧ c > a := by
  sorry

end arrange_abc_l95_95377


namespace spilled_bag_candies_l95_95191

theorem spilled_bag_candies (c1 c2 c3 c4 c5 c6 c7 : ℕ) (avg_candies_per_bag : ℕ) (x : ℕ) 
  (h_counts : c1 = 12 ∧ c2 = 14 ∧ c3 = 18 ∧ c4 = 22 ∧ c5 = 24 ∧ c6 = 26 ∧ c7 = 29)
  (h_avg : avg_candies_per_bag = 22)
  (h_total : c1 + c2 + c3 + c4 + c5 + c6 + c7 + x = 8 * avg_candies_per_bag) : x = 31 := 
by
  sorry

end spilled_bag_candies_l95_95191


namespace minute_hand_angle_l95_95064

theorem minute_hand_angle (minutes_slow : ℕ) (total_minutes : ℕ) (full_rotation : ℝ) (h1 : minutes_slow = 5) (h2 : total_minutes = 60) (h3 : full_rotation = 2 * Real.pi) : 
  (minutes_slow / total_minutes : ℝ) * full_rotation = Real.pi / 6 :=
by
  sorry

end minute_hand_angle_l95_95064


namespace inequality_relations_l95_95386

variable {R : Type} [OrderedAddCommGroup R]
variables (x y z : R)

theorem inequality_relations (h1 : x - y > x + z) (h2 : x + y < y + z) : y < -z ∧ x < z :=
by
  sorry

end inequality_relations_l95_95386


namespace determine_d_l95_95994

theorem determine_d (d c f : ℚ) :
  (3 * x^3 - 2 * x^2 + x - (5/4)) * (3 * x^3 + d * x^2 + c * x + f) = 9 * x^6 - 5 * x^5 - x^4 + 20 * x^3 - (25/4) * x^2 + (15/4) * x - (5/2) →
  d = 1 / 3 :=
by
  sorry

end determine_d_l95_95994


namespace consecutive_numbers_perfect_square_l95_95960

theorem consecutive_numbers_perfect_square (a : ℕ) (h : a ≥ 1) : 
  (a * (a + 1) * (a + 2) * (a + 3) + 1) = (a^2 + 3 * a + 1)^2 :=
by sorry

end consecutive_numbers_perfect_square_l95_95960


namespace mary_shirt_fraction_l95_95909

theorem mary_shirt_fraction (f : ℝ) : 
  26 * (1 - f) + 36 - 36 / 3 = 37 → f = 1 / 2 :=
by
  sorry

end mary_shirt_fraction_l95_95909


namespace gum_lcm_l95_95624

theorem gum_lcm (strawberry blueberry cherry : ℕ) (h₁ : strawberry = 6) (h₂ : blueberry = 5) (h₃ : cherry = 8) :
  Nat.lcm (Nat.lcm strawberry blueberry) cherry = 120 :=
by
  rw [h₁, h₂, h₃]
  -- LCM(6, 5, 8) = LCM(LCM(6, 5), 8)
  sorry

end gum_lcm_l95_95624


namespace ratio_of_c_d_l95_95537

theorem ratio_of_c_d 
  (x y c d : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hd : d ≠ 0)
  (h1 : 8 * x - 6 * y = c)
  (h2 : 12 * y - 18 * x = d) :
  c / d = -4 / 3 :=
by 
  sorry

end ratio_of_c_d_l95_95537


namespace larger_number_is_437_l95_95093

-- Definitions from the conditions
def hcf : ℕ := 23
def factor1 : ℕ := 13
def factor2 : ℕ := 19

-- The larger number should be the product of H.C.F and the larger factor.
theorem larger_number_is_437 : hcf * factor2 = 437 := by
  sorry

end larger_number_is_437_l95_95093


namespace find_c_l95_95948

-- Define the quadratic polynomial with given conditions
def quadratic (b c x y : ℝ) : Prop :=
  y = x^2 + b * x + c

-- Define the condition that the polynomial passes through two particular points
def passes_through_points (b c : ℝ) : Prop :=
  (quadratic b c 1 4) ∧ (quadratic b c 5 4)

-- The theorem stating c is 9 given the conditions
theorem find_c (b c : ℝ) (h : passes_through_points b c) : c = 9 :=
by {
  sorry
}

end find_c_l95_95948


namespace each_vaccine_costs_45_l95_95978

theorem each_vaccine_costs_45
    (num_vaccines : ℕ)
    (doctor_visit_cost : ℝ)
    (insurance_coverage : ℝ)
    (trip_cost : ℝ)
    (total_payment : ℝ) :
    num_vaccines = 10 ->
    doctor_visit_cost = 250 ->
    insurance_coverage = 0.80 ->
    trip_cost = 1200 ->
    total_payment = 1340 ->
    (∃ (vaccine_cost : ℝ), vaccine_cost = 45) :=
by {
    sorry
}

end each_vaccine_costs_45_l95_95978


namespace cube_difference_positive_l95_95539

theorem cube_difference_positive (a b : ℝ) (h : a > b) : a^3 - b^3 > 0 :=
sorry

end cube_difference_positive_l95_95539


namespace smallest_n_l95_95241

variable {a : ℕ → ℝ} -- the arithmetic sequence
noncomputable def d := a 2 - a 1  -- common difference

variable {S : ℕ → ℝ}  -- sum of the first n terms

-- conditions
axiom cond1 : a 66 < 0
axiom cond2 : a 67 > 0
axiom cond3 : a 67 > abs (a 66)

-- sum of the first n terms of the arithmetic sequence
noncomputable def sum_n (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem smallest_n (n : ℕ) : S n > 0 → n = 132 :=
by
  sorry

end smallest_n_l95_95241


namespace number_of_license_plates_l95_95259

-- Define the alphabet size and digit size constants.
def num_letters : ℕ := 26
def num_digits : ℕ := 10

-- Define the number of letters in the license plate.
def letters_in_plate : ℕ := 3

-- Define the number of digits in the license plate.
def digits_in_plate : ℕ := 4

-- Calculating the total number of license plates possible as (26^3) * (10^4).
theorem number_of_license_plates : 
  (num_letters ^ letters_in_plate) * (num_digits ^ digits_in_plate) = 175760000 :=
by
  sorry

end number_of_license_plates_l95_95259


namespace sum_of_digits_of_triangular_number_2010_l95_95774

noncomputable def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_of_triangular_number_2010 (N : ℕ)
  (h₁ : triangular_number N = 2010) :
  sum_of_digits N = 9 :=
sorry

end sum_of_digits_of_triangular_number_2010_l95_95774


namespace total_space_needed_for_trees_l95_95835

def appleTreeWidth : ℕ := 10
def spaceBetweenAppleTrees : ℕ := 12
def numAppleTrees : ℕ := 2

def peachTreeWidth : ℕ := 12
def spaceBetweenPeachTrees : ℕ := 15
def numPeachTrees : ℕ := 2

def totalSpace : ℕ :=
  numAppleTrees * appleTreeWidth + spaceBetweenAppleTrees +
  numPeachTrees * peachTreeWidth + spaceBetweenPeachTrees

theorem total_space_needed_for_trees : totalSpace = 71 := by
  sorry

end total_space_needed_for_trees_l95_95835


namespace sequence_sum_l95_95216

-- Defining the sequence terms
variables (J K L M N O P Q R S : ℤ)
-- Condition N = 7
def N_value : Prop := N = 7
-- Condition sum of any four consecutive terms is 40
def sum_of_consecutive : Prop := 
  J + K + L + M = 40 ∧
  K + L + M + N = 40 ∧
  L + M + N + O = 40 ∧
  M + N + O + P = 40 ∧
  N + O + P + Q = 40 ∧
  O + P + Q + R = 40 ∧
  P + Q + R + S = 40

-- The main theorem stating J + S = 40 given the conditions
theorem sequence_sum (N_value : N = 7) (sum_of_consecutive : 
  J + K + L + M = 40 ∧
  K + L + M + N = 40 ∧
  L + M + N + O = 40 ∧
  M + N + O + P = 40 ∧
  N + O + P + Q = 40 ∧
  O + P + Q + R = 40 ∧
  P + Q + R + S = 40) : 
  J + S = 40 := sorry

end sequence_sum_l95_95216


namespace simplify_expression_l95_95007

theorem simplify_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) *
  (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 := 
by sorry

end simplify_expression_l95_95007


namespace find_pairs_l95_95489

def sequence_a : Nat → Int
| 0 => 0
| 1 => 0
| n+2 => 2 * sequence_a (n+1) - sequence_a n + 2

def sequence_b : Nat → Int
| 0 => 8
| 1 => 8
| n+2 => 2 * sequence_b (n+1) - sequence_b n

theorem find_pairs :
  (sequence_a 1992 = 31872 ∧ sequence_b 1992 = 31880) ∨
  (sequence_a 1992 = -31872 ∧ sequence_b 1992 = -31864) :=
sorry

end find_pairs_l95_95489


namespace max_pieces_l95_95875

namespace CakeProblem

-- Define the dimensions of the cake and the pieces.
def cake_side : ℕ := 16
def piece_side : ℕ := 4

-- Define the areas of the cake and the pieces.
def cake_area : ℕ := cake_side * cake_side
def piece_area : ℕ := piece_side * piece_side

-- State the main problem to prove.
theorem max_pieces : cake_area / piece_area = 16 :=
by
  -- The proof is omitted.
  sorry

end CakeProblem

end max_pieces_l95_95875


namespace sweets_ratio_l95_95848

theorem sweets_ratio (x : ℕ) (h1 : x + 4 + 7 = 22) : x / 22 = 1 / 2 :=
by
  sorry

end sweets_ratio_l95_95848


namespace expected_number_of_edges_same_color_3x3_l95_95733

noncomputable def expected_edges_same_color (board_size : ℕ) (blackened_count : ℕ) : ℚ :=
  let total_pairs := 12       -- 6 horizontal pairs + 6 vertical pairs
  let prob_both_white := 1 / 6
  let prob_both_black := 5 / 18
  let prob_same_color := prob_both_white + prob_both_black
  total_pairs * prob_same_color

theorem expected_number_of_edges_same_color_3x3 :
  expected_edges_same_color 3 5 = 16 / 3 :=
by
  sorry

end expected_number_of_edges_same_color_3x3_l95_95733


namespace malvina_card_value_sum_l95_95836

noncomputable def possible_values_sum: ℝ :=
  let value1 := 1
  let value2 := (-1 + Real.sqrt 5) / 2
  (value1 + value2) / 2

theorem malvina_card_value_sum
  (hx : ∃ x : ℝ, 0 < x ∧ x < Real.pi / 2 ∧ 
                 (x = Real.pi / 4 ∨ (Real.sin x = (-1 + Real.sqrt 5) / 2))):
  possible_values_sum = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end malvina_card_value_sum_l95_95836


namespace proper_subset_A_B_l95_95572

theorem proper_subset_A_B (a : ℝ) : 
  (∀ x, 1 < x ∧ x < 2 → x < a) ∧ (∃ b, b < a ∧ ¬(1 < b ∧ b < 2)) ↔ 2 ≤ a :=
by
  sorry

end proper_subset_A_B_l95_95572


namespace find_integers_l95_95414

theorem find_integers (n : ℕ) (h1 : n < 10^100)
  (h2 : n ∣ 2^n) (h3 : n - 1 ∣ 2^n - 1) (h4 : n - 2 ∣ 2^n - 2) :
  n = 2^2 ∨ n = 2^4 ∨ n = 2^16 ∨ n = 2^256 := by
  sorry

end find_integers_l95_95414


namespace sin_70_eq_1_minus_2k_squared_l95_95927

theorem sin_70_eq_1_minus_2k_squared (k : ℝ) (h : Real.sin 10 = k) : Real.sin 70 = 1 - 2 * k^2 := 
by
  sorry

end sin_70_eq_1_minus_2k_squared_l95_95927


namespace circle_passing_through_points_l95_95409

noncomputable def parabola (x: ℝ) (a b: ℝ) : ℝ :=
  x^2 + a * x + b

theorem circle_passing_through_points (a b α β k: ℝ) :
  parabola 0 a b = b ∧ parabola α a b = 0 ∧ parabola β a b = 0 ∧
  ((0 - (α + β) / 2)^2 + (1 - k)^2 = ((α + β) / 2)^2 + (k - b)^2) →
  b = 1 :=
by
  sorry

end circle_passing_through_points_l95_95409


namespace stratified_sampling_result_l95_95898

-- Define the total number of students in each grade
def students_grade10 : ℕ := 1600
def students_grade11 : ℕ := 1200
def students_grade12 : ℕ := 800

-- Define the condition
def stratified_sampling (x : ℕ) : Prop :=
  (x / (students_grade10 + students_grade11 + students_grade12) = (20 / students_grade12))

-- The main statement to be proven
theorem stratified_sampling_result 
  (students_grade10 : ℕ)
  (students_grade11 : ℕ)
  (students_grade12 : ℕ)
  (sampled_from_grade12 : ℕ)
  (h_sampling : stratified_sampling 90)
  (h_sampled12 : sampled_from_grade12 = 20) :
  (90 - sampled_from_grade12 = 70) :=
  by
    sorry

end stratified_sampling_result_l95_95898


namespace melted_ice_cream_depth_l95_95376

noncomputable def radius_sphere : ℝ := 3
noncomputable def radius_cylinder : ℝ := 10
noncomputable def height_cylinder : ℝ := 36 / 100

theorem melted_ice_cream_depth :
  (4 / 3) * Real.pi * radius_sphere^3 = Real.pi * radius_cylinder^2 * height_cylinder :=
by
  sorry

end melted_ice_cream_depth_l95_95376


namespace isosceles_triangle_perimeter_l95_95983

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : (a = 3 ∨ a = 7)) (h2 : (b = 3 ∨ b = 7)) (h3 : a ≠ b) : 
  ∃ (c : ℕ), (a = 7 ∧ b = 3 ∧ c = 17) ∨ (a = 3 ∧ b = 7 ∧ c = 17) := 
by
  sorry

end isosceles_triangle_perimeter_l95_95983


namespace toothpicks_needed_for_8_step_staircase_l95_95016

theorem toothpicks_needed_for_8_step_staircase:
  ∀ n toothpicks : ℕ, n = 4 → toothpicks = 30 → 
  (∃ additional_toothpicks : ℕ, additional_toothpicks = 88) :=
by
  sorry

end toothpicks_needed_for_8_step_staircase_l95_95016


namespace remainder_of_power_l95_95026

theorem remainder_of_power :
  (4^215) % 9 = 7 := by
sorry

end remainder_of_power_l95_95026


namespace fraction_sum_geq_zero_l95_95266

theorem fraction_sum_geq_zero (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (1 / (a - b) + 1 / (b - c) + 4 / (c - a)) ≥ 0 := 
by 
  sorry

end fraction_sum_geq_zero_l95_95266


namespace discount_percentage_l95_95632

theorem discount_percentage (MP CP SP : ℝ)
  (h1 : CP = 0.64 * MP)
  (h2 : SP = CP * 1.375)
  (gain_percent : 37.5 = (SP - CP) / CP * 100) :
  (MP - SP) / MP * 100 = 12 :=
by
  sorry

end discount_percentage_l95_95632


namespace flowers_per_bouquet_l95_95491

theorem flowers_per_bouquet (total_flowers wilted_flowers : ℕ) (bouquets : ℕ) (remaining_flowers : ℕ)
    (h1 : total_flowers = 45)
    (h2 : wilted_flowers = 35)
    (h3 : bouquets = 2)
    (h4 : remaining_flowers = total_flowers - wilted_flowers)
    (h5 : bouquets * (remaining_flowers / bouquets) = remaining_flowers) :
  remaining_flowers / bouquets = 5 :=
by
  sorry

end flowers_per_bouquet_l95_95491


namespace reading_time_difference_in_minutes_l95_95604

noncomputable def xanthia_reading_speed : ℝ := 120 -- pages per hour
noncomputable def molly_reading_speed : ℝ := 60 -- pages per hour
noncomputable def book_length : ℝ := 360 -- pages

theorem reading_time_difference_in_minutes :
  let time_for_xanthia := book_length / xanthia_reading_speed
  let time_for_molly := book_length / molly_reading_speed
  let difference_in_hours := time_for_molly - time_for_xanthia
  difference_in_hours * 60 = 180 :=
by
  sorry

end reading_time_difference_in_minutes_l95_95604


namespace solve_quadratic_l95_95429

theorem solve_quadratic (x : ℝ) (h : (9 / x^2) - (6 / x) + 1 = 0) : 2 / x = 2 / 3 :=
by
  sorry

end solve_quadratic_l95_95429


namespace burt_net_profit_l95_95356

theorem burt_net_profit
  (cost_seeds : ℝ := 2.00)
  (cost_soil : ℝ := 8.00)
  (num_plants : ℕ := 20)
  (price_per_plant : ℝ := 5.00) :
  let total_cost := cost_seeds + cost_soil
  let total_revenue := num_plants * price_per_plant
  let net_profit := total_revenue - total_cost
  net_profit = 90.00 :=
by sorry

end burt_net_profit_l95_95356


namespace train_ride_time_in_hours_l95_95446

-- Definition of conditions
def lukes_total_trip_time_hours : ℕ := 8
def bus_ride_minutes : ℕ := 75
def walk_to_train_center_minutes : ℕ := 15
def wait_time_minutes : ℕ := 2 * walk_to_train_center_minutes

-- Convert total trip time to minutes
def lukes_total_trip_time_minutes : ℕ := lukes_total_trip_time_hours * 60

-- Calculate the total time spent on bus, walking, and waiting
def bus_walk_wait_time_minutes : ℕ :=
  bus_ride_minutes + walk_to_train_center_minutes + wait_time_minutes

-- Calculate the train ride time in minutes
def train_ride_time_minutes : ℕ :=
  lukes_total_trip_time_minutes - bus_walk_wait_time_minutes

-- Prove the train ride time in hours
theorem train_ride_time_in_hours : train_ride_time_minutes / 60 = 6 :=
by
  sorry

end train_ride_time_in_hours_l95_95446


namespace cost_per_slice_in_cents_l95_95269

def loaves : ℕ := 3
def slices_per_loaf : ℕ := 20
def total_payment : ℕ := 2 * 20
def change : ℕ := 16
def total_cost : ℕ := total_payment - change
def total_slices : ℕ := loaves * slices_per_loaf

theorem cost_per_slice_in_cents :
  (total_cost : ℕ) * 100 / total_slices = 40 :=
by
  sorry

end cost_per_slice_in_cents_l95_95269


namespace belfried_industries_payroll_l95_95861

theorem belfried_industries_payroll (P : ℝ) (tax_paid : ℝ) : 
  ((P > 200000) ∧ (tax_paid = 0.002 * (P - 200000)) ∧ (tax_paid = 200)) → P = 300000 :=
by
  sorry

end belfried_industries_payroll_l95_95861


namespace total_seashells_l95_95895

theorem total_seashells (a b : Nat) (h1 : a = 5) (h2 : b = 7) : 
  let total_first_two_days := a + b
  let third_day := 2 * total_first_two_days
  let total := total_first_two_days + third_day
  total = 36 := 
by
  sorry

end total_seashells_l95_95895


namespace area_difference_is_correct_l95_95197

noncomputable def area_rectangle (length width : ℝ) : ℝ := length * width

noncomputable def area_equilateral_triangle (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side ^ 2

noncomputable def area_circle (diameter : ℝ) : ℝ := (Real.pi * (diameter / 2) ^ 2)

noncomputable def combined_area_difference : ℝ :=
  (area_rectangle 11 11 + area_rectangle 5.5 11) - 
  (area_equilateral_triangle 6 + area_circle 4)
 
theorem area_difference_is_correct :
  |combined_area_difference - 153.35| < 0.001 :=
by
  sorry

end area_difference_is_correct_l95_95197


namespace range_of_fraction_l95_95209

-- Definition of the quadratic equation with roots within specified intervals
variables (a b : ℝ)
variables (x1 x2 : ℝ)
variables (h_distinct_roots : x1 ≠ x2)
variables (h_interval_x1 : 0 < x1 ∧ x1 < 1)
variables (h_interval_x2 : 1 < x2 ∧ x2 < 2)
variables (h_quadratic : ∀ x : ℝ, x^2 + a * x + 2 * b - 2 = 0)

-- Prove range of expression
theorem range_of_fraction (a b : ℝ)
  (x1 x2 h_distinct_roots : ℝ) (h_interval_x1 : 0 < x1 ∧ x1 < 1)
  (h_interval_x2 : 1 < x2 ∧ x2 < 2)
  (h_quadratic : ∀ x, x^2 + a * x + 2 * b - 2 = 0) :
  (1/2 < (b - 4) / (a - 1)) ∧ ((b - 4) / (a - 1) < 3/2) :=
by
  -- proof placeholder
  sorry

end range_of_fraction_l95_95209


namespace find_books_second_purchase_profit_l95_95871

-- For part (1)
theorem find_books (x y : ℕ) (h₁ : 12 * x + 10 * y = 1200) (h₂ : 3 * x + 2 * y = 270) :
  x = 50 ∧ y = 60 :=
by 
  sorry

-- For part (2)
theorem second_purchase_profit (m : ℕ) (h₃ : 50 * (m - 12) + 2 * 60 * (12 - 10) ≥ 340) :
  m ≥ 14 :=
by 
  sorry

end find_books_second_purchase_profit_l95_95871


namespace find_number_l95_95553

theorem find_number : ∃ x : ℝ, (x / 6 * 12 = 10) ∧ x = 5 :=
by
 sorry

end find_number_l95_95553


namespace expression_evaluation_l95_95309

-- Define the numbers and operations
def expr : ℚ := 10 * (1 / 2) * 3 / (1 / 6)

-- Formalize the proof problem
theorem expression_evaluation : expr = 90 := 
by 
  -- Start the proof, which is not required according to the instruction, so we replace it with 'sorry'
  sorry

end expression_evaluation_l95_95309


namespace tan_2x_eq_sin_x_has_three_solutions_l95_95182

theorem tan_2x_eq_sin_x_has_three_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.tan (2 * x) = Real.sin x) ∧ S.card = 3 :=
by
  sorry

end tan_2x_eq_sin_x_has_three_solutions_l95_95182


namespace eval_expression_l95_95797

theorem eval_expression (a b c : ℕ) (h₀ : a = 3) (h₁ : b = 2) (h₂ : c = 1) : 
  (a^3 + b^2 + c)^2 - (a^3 + b^2 - c)^2 = 124 :=
by
  sorry

end eval_expression_l95_95797


namespace minutes_before_4_angle_same_as_4_l95_95388

def hour_hand_angle_at_4 := 120
def minute_hand_angle_at_4 := 0
def minute_hand_angle_per_minute := 6
def hour_hand_angle_per_minute := 0.5

theorem minutes_before_4_angle_same_as_4 :
  ∃ m : ℚ, abs (hour_hand_angle_at_4 - 5.5 * m) = hour_hand_angle_at_4 ∧ 
           (60 - m) = 21 + 9 / 11 := by
  sorry

end minutes_before_4_angle_same_as_4_l95_95388


namespace increase_in_p_does_not_imply_increase_in_equal_points_probability_l95_95020

noncomputable def probability_equal_points (p : ℝ) : ℝ :=
  (3 * p^2 - 2 * p + 1) / 4

theorem increase_in_p_does_not_imply_increase_in_equal_points_probability :
  ¬ ∀ p1 p2 : ℝ, p1 < p2 → p1 ≥ 0 → p2 ≤ 1 → probability_equal_points p1 < probability_equal_points p2 := 
sorry

end increase_in_p_does_not_imply_increase_in_equal_points_probability_l95_95020


namespace sum_of_ages_l95_95178

theorem sum_of_ages (J M R : ℕ) (hJ : J = 10) (hM : M = J - 3) (hR : R = J + 2) : M + R = 19 :=
by
  sorry

end sum_of_ages_l95_95178


namespace box_volume_l95_95932

theorem box_volume (l w h V : ℝ) 
  (h1 : l * w = 30) 
  (h2 : w * h = 18) 
  (h3 : l * h = 10) 
  : V = l * w * h → V = 90 :=
by 
  intro volume_eq
  sorry

end box_volume_l95_95932


namespace sale_price_is_correct_l95_95054

def initial_price : ℝ := 560
def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.30
def discount3 : ℝ := 0.15
def tax_rate : ℝ := 0.12

noncomputable def final_price : ℝ :=
  let price_after_first_discount := initial_price * (1 - discount1)
  let price_after_second_discount := price_after_first_discount * (1 - discount2)
  let price_after_third_discount := price_after_second_discount * (1 - discount3)
  let price_after_tax := price_after_third_discount * (1 + tax_rate)
  price_after_tax

theorem sale_price_is_correct :
  final_price = 298.55 :=
sorry

end sale_price_is_correct_l95_95054


namespace log_identity_l95_95566

noncomputable def my_log (base x : ℝ) := Real.log x / Real.log base

theorem log_identity (x : ℝ) (h : x > 0) (h1 : x ≠ 1) : 
  (my_log 4 x) * (my_log x 5) = my_log 4 5 :=
by
  sorry

end log_identity_l95_95566


namespace laptop_full_price_l95_95108

theorem laptop_full_price (p : ℝ) (deposit : ℝ) (h1 : deposit = 0.25 * p) (h2 : deposit = 400) : p = 1600 :=
by
  sorry

end laptop_full_price_l95_95108


namespace find_principal_l95_95478

-- Define the conditions
def simple_interest (P R T : ℕ) : ℕ :=
  (P * R * T) / 100

-- Given values
def SI : ℕ := 750
def R : ℕ := 6
def T : ℕ := 5

-- Proof statement
theorem find_principal : ∃ P : ℕ, simple_interest P R T = SI ∧ P = 2500 := by
  aesop

end find_principal_l95_95478


namespace kiril_konstantinovich_age_is_full_years_l95_95147

theorem kiril_konstantinovich_age_is_full_years
  (years : ℕ)
  (months : ℕ)
  (weeks : ℕ)
  (days : ℕ)
  (hours : ℕ) :
  (years = 48) →
  (months = 48) →
  (weeks = 48) →
  (days = 48) →
  (hours = 48) →
  Int.floor (
    years + 
    (months / 12 : ℝ) + 
    (weeks * 7 / 365 : ℝ) + 
    (days / 365 : ℝ) + 
    (hours / (24 * 365) : ℝ)
  ) = 53 :=
by
  intro hyears hmonths hweeks hdays hhours
  rw [hyears, hmonths, hweeks, hdays, hhours]
  sorry

end kiril_konstantinovich_age_is_full_years_l95_95147


namespace sum_of_squares_l95_95149

theorem sum_of_squares (a b : ℕ) (h_side_lengths : 20^2 = a^2 + b^2) : a + b = 28 :=
sorry

end sum_of_squares_l95_95149


namespace eccentricity_of_ellipse_equation_of_ellipse_l95_95187

variable {a b : ℝ}
variable {x y : ℝ}

/-- Problem 1: Eccentricity of the given ellipse --/
theorem eccentricity_of_ellipse (ha : a = 2 * b) (hb0 : 0 < b) :
  ∃ e : ℝ, e = Real.sqrt (1 - (b / a) ^ 2) ∧ e = Real.sqrt 3 / 2 := by
  sorry

/-- Problem 2: Equation of the ellipse with respect to maximizing the area of triangle OMN --/
theorem equation_of_ellipse (ha : a = 2 * b) (hb0 : 0 < b) :
  ∃ l : ℝ → ℝ, (∃ k : ℝ, ∀ x, l x = k * x + 2) →
  ∀ x y : ℝ, (x^2 / (a^2) + y^2 / (b^2) = 1) →
  (∀ x' y' : ℝ, (x'^2 + 4 * y'^2 = 4 * b^2) ∧ y' = k * x' + 2) →
  (∃ a b : ℝ, a = 8 ∧ b = 2 ∧ x^2 / a + y^2 / b = 1) := by
  sorry

end eccentricity_of_ellipse_equation_of_ellipse_l95_95187


namespace star_intersections_l95_95219

theorem star_intersections (n k : ℕ) (h_coprime : Nat.gcd n k = 1) (h_n_ge_5 : 5 ≤ n) (h_k_lt_n_div_2 : k < n / 2) :
    k = 25 → n = 2018 → n * (k - 1) = 48432 := by
  intros
  sorry

end star_intersections_l95_95219


namespace max_element_sum_l95_95023

-- Definitions based on conditions
def S : Set ℚ :=
  {r | ∃ (p q : ℕ), r = p / q ∧ q ≤ 2009 ∧ p / q < 1257/2009}

-- Maximum element of S in reduced form
def max_element_S (r : ℚ) : Prop := r ∈ S ∧ ∀ s ∈ S, r ≥ s

-- Main statement to be proven
theorem max_element_sum : 
  ∃ p0 q0 : ℕ, max_element_S (p0 / q0) ∧ Nat.gcd p0 q0 = 1 ∧ p0 + q0 = 595 := 
sorry

end max_element_sum_l95_95023


namespace simplify_and_evaluate_expression_l95_95295

variable (x y : ℝ)

theorem simplify_and_evaluate_expression
  (hx : x = 2)
  (hy : y = -0.5) :
  2 * (2 * x - 3 * y) - (3 * x + 2 * y + 1) = 5 :=
by
  sorry

end simplify_and_evaluate_expression_l95_95295


namespace positive_diff_solutions_l95_95951

theorem positive_diff_solutions : 
  (∃ x₁ x₂ : ℝ, ( (9 - x₁^2 / 4)^(1/3) = -3) ∧ ((9 - x₂^2 / 4)^(1/3) = -3) ∧ ∃ (d : ℝ), d = |x₁ - x₂| ∧ d = 24) :=
by
  sorry

end positive_diff_solutions_l95_95951


namespace find_positive_solution_l95_95524

-- Defining the variables x, y, and z as real numbers
variables (x y z : ℝ)

-- Define the conditions from the problem statement
def condition1 : Prop := x * y + 3 * x + 4 * y + 10 = 30
def condition2 : Prop := y * z + 4 * y + 2 * z + 8 = 6
def condition3 : Prop := x * z + 4 * x + 3 * z + 12 = 30

-- The theorem that states the positive solution for x is 3
theorem find_positive_solution (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 x z) : x = 3 :=
by {
  sorry
}

end find_positive_solution_l95_95524


namespace polygon_diagonals_with_one_non_connecting_vertex_l95_95856

-- Define the number of sides in the polygon
def num_sides : ℕ := 17

-- Define the formula to calculate the number of diagonals in a polygon
def total_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- Define the number of non-connecting vertex to any diagonal
def non_connected_diagonals (n : ℕ) : ℕ :=
  n - 3

-- The theorem to state and prove
theorem polygon_diagonals_with_one_non_connecting_vertex :
  total_diagonals num_sides - non_connected_diagonals num_sides = 105 :=
by
  -- The formal proof would go here
  sorry

end polygon_diagonals_with_one_non_connecting_vertex_l95_95856


namespace intersection_M_N_l95_95508

open Set

noncomputable def M : Set ℕ := {x | x < 6}
noncomputable def N : Set ℕ := {x | x^2 - 11 * x + 18 < 0}

theorem intersection_M_N : M ∩ N = {3, 4, 5} := by
  sorry

end intersection_M_N_l95_95508


namespace brushes_cost_l95_95394

-- Define the conditions
def canvas_cost (B : ℝ) : ℝ := 3 * B
def paint_cost : ℝ := 5 * 8
def total_material_cost (B : ℝ) : ℝ := B + canvas_cost B + paint_cost
def earning_from_sale : ℝ := 200 - 80

-- State the question as a theorem in Lean
theorem brushes_cost (B : ℝ) (h : total_material_cost B = earning_from_sale) : B = 20 :=
sorry

end brushes_cost_l95_95394


namespace reciprocal_of_neg_five_l95_95705

theorem reciprocal_of_neg_five: 
  ∃ x : ℚ, -5 * x = 1 ∧ x = -1 / 5 := 
sorry

end reciprocal_of_neg_five_l95_95705


namespace bread_rise_time_l95_95822

theorem bread_rise_time (x : ℕ) (kneading_time : ℕ) (baking_time : ℕ) (total_time : ℕ) 
  (h1 : kneading_time = 10) 
  (h2 : baking_time = 30) 
  (h3 : total_time = 280) 
  (h4 : kneading_time + baking_time + 2 * x = total_time) : 
  x = 120 :=
sorry

end bread_rise_time_l95_95822


namespace bottle_cap_cost_l95_95363

-- Define the conditions given in the problem.
def caps_cost (n : ℕ) (cost : ℝ) : Prop := n * cost = 12

-- Prove that the cost of each bottle cap is $2 given 6 bottle caps cost $12.
theorem bottle_cap_cost (h : caps_cost 6 cost) : cost = 2 :=
sorry

end bottle_cap_cost_l95_95363


namespace codecracker_total_combinations_l95_95565

theorem codecracker_total_combinations (colors slots : ℕ) (h_colors : colors = 6) (h_slots : slots = 5) :
  colors ^ slots = 7776 :=
by
  rw [h_colors, h_slots]
  norm_num

end codecracker_total_combinations_l95_95565


namespace escalator_length_l95_95584

theorem escalator_length
  (escalator_speed : ℝ)
  (person_speed : ℝ)
  (time_taken : ℝ)
  (combined_speed := escalator_speed + person_speed)
  (distance := combined_speed * time_taken) :
  escalator_speed = 10 → person_speed = 4 → time_taken = 8 → distance = 112 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end escalator_length_l95_95584


namespace vertex_coordinates_l95_95638

-- Define the given parabola equation
def parabola (x : ℝ) : ℝ := (x + 3) ^ 2 - 1

-- Define the statement for the coordinates of the vertex of the parabola
theorem vertex_coordinates : ∃ (h k : ℝ), (∀ x : ℝ, parabola x = (x + 3) ^ 2 - 1) ∧ h = -3 ∧ k = -1 := 
  sorry

end vertex_coordinates_l95_95638


namespace find_base_a_l95_95955

theorem find_base_a 
  (a : ℕ)
  (C_a : ℕ := 12) :
  (3 * a^2 + 4 * a + 7) + (5 * a^2 + 7 * a + 9) = 9 * a^2 + 2 * a + C_a →
  a = 14 :=
by
  intros h
  sorry

end find_base_a_l95_95955


namespace max_x4_y6_l95_95175

noncomputable def maximum_product (x y : ℝ) := x^4 * y^6

theorem max_x4_y6 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 100) :
  maximum_product x y ≤ maximum_product 40 60 := sorry

end max_x4_y6_l95_95175


namespace solution_set_l95_95877

noncomputable def solve_inequality : Set ℝ :=
  {x | (1 / (x - 1)) >= -1}

theorem solution_set :
  solve_inequality = {x | x ≤ 0} ∪ {x | x > 1} :=
by
  sorry

end solution_set_l95_95877


namespace multiple_of_three_l95_95587

theorem multiple_of_three (a b : ℤ) : ∃ k : ℤ, (a + b = 3 * k) ∨ (ab = 3 * k) ∨ (a - b = 3 * k) :=
sorry

end multiple_of_three_l95_95587


namespace percent_to_decimal_l95_95644

theorem percent_to_decimal : (2 : ℝ) / 100 = 0.02 :=
by
  -- Proof would go here
  sorry

end percent_to_decimal_l95_95644


namespace determine_n_l95_95815

noncomputable def polynomial (n : ℕ) : ℕ → ℕ := sorry  -- Placeholder for the actual polynomial function

theorem determine_n (n : ℕ) 
  (h_deg : ∀ a, polynomial n a = 2 → (3 ∣ a) ∨ a = 0)
  (h_deg' : ∀ a, polynomial n a = 1 → (3 ∣ (a + 2)))
  (h_deg'' : ∀ a, polynomial n a = 0 → (3 ∣ (a + 1)))
  (h_val : polynomial n (3*n+1) = 730) :
  n = 4 :=
sorry

end determine_n_l95_95815


namespace system_of_equations_m_value_l95_95798

theorem system_of_equations_m_value {x y m : ℝ} 
  (h1 : 2 * x + y = 4)
  (h2 : x + 2 * y = m)
  (h3 : x + y = 1) : m = -1 := 
sorry

end system_of_equations_m_value_l95_95798


namespace circles_intersect_l95_95858

-- Definition of the first circle
def circleC := { p : ℝ × ℝ | (p.1 + 1)^2 + p.2^2 = 4 }

-- Definition of the second circle
def circleM := { p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 9 }

-- Prove that the circles intersect
theorem circles_intersect : 
  ∃ p : ℝ × ℝ, p ∈ circleC ∧ p ∈ circleM := 
sorry

end circles_intersect_l95_95858


namespace sum_of_roots_l95_95164

theorem sum_of_roots (x : ℝ) :
  (x + 2) * (x - 3) = 16 →
  ∃ a b : ℝ, (a ≠ x ∧ b ≠ x ∧ (x - a) * (x - b) = 0) ∧
             (a + b = 1) :=
by
  intro h
  sorry

end sum_of_roots_l95_95164


namespace proof_problem_solution_l95_95668

noncomputable def proof_problem (a b c d : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ (a * b + b * c + c * d + d * a = 1) →
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3)

theorem proof_problem_solution (a b c d : ℝ) : proof_problem a b c d :=
  sorry

end proof_problem_solution_l95_95668


namespace ratio_of_board_pieces_l95_95320

theorem ratio_of_board_pieces (S L : ℕ) (hS : S = 23) (hTotal : S + L = 69) : L / S = 2 :=
by
  sorry

end ratio_of_board_pieces_l95_95320


namespace least_number_to_add_divisible_l95_95591

theorem least_number_to_add_divisible (n d : ℕ) (h1 : n = 929) (h2 : d = 30) : 
  ∃ x, (n + x) % d = 0 ∧ x = 1 := 
by 
  sorry

end least_number_to_add_divisible_l95_95591


namespace find_x_l95_95808

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0) (h : x + y + x * y = 143) : x = 15 :=
by sorry

end find_x_l95_95808


namespace find_values_l95_95120

def isInInterval (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

theorem find_values 
  (a b c d e : ℝ)
  (ha : isInInterval a) 
  (hb : isInInterval b) 
  (hc : isInInterval c) 
  (hd : isInInterval d)
  (he : isInInterval e)
  (h1 : a + b + c + d + e = 0)
  (h2 : a^3 + b^3 + c^3 + d^3 + e^3 = 0)
  (h3 : a^5 + b^5 + c^5 + d^5 + e^5 = 10) : 
  (a = 2 ∧ b = (Real.sqrt 5 - 1) / 2 ∧ c = (Real.sqrt 5 - 1) / 2 ∧ d = - (1 + Real.sqrt 5) / 2 ∧ e = - (1 + Real.sqrt 5) / 2) ∨
  (a = (Real.sqrt 5 - 1) / 2 ∧ b = 2 ∧ c = (Real.sqrt 5 - 1) / 2 ∧ d = - (1 + Real.sqrt 5) / 2 ∧ e = - (1 + Real.sqrt 5) / 2) ∨
  (a = (Real.sqrt 5 - 1) / 2 ∧ b = (Real.sqrt 5 - 1) / 2 ∧ c = 2 ∧ d = - (1 + Real.sqrt 5) / 2 ∧ e = - (1 + Real.sqrt 5) / 2) ∨
  (a = (Real.sqrt 5 - 1) / 2 ∧ b = (Real.sqrt 5 - 1) / 2 ∧ c = - (1 + Real.sqrt 5) / 2 ∧ d = 2 ∧ e = - (1 + Real.sqrt 5) / 2) ∨
  (a = (Real.sqrt 5 - 1) / 2 ∧ b = (Real.sqrt 5 - 1) / 2 ∧ c = - (1 + Real.sqrt 5) / 2 ∧ d = - (1 + Real.sqrt 5) / 2 ∧ e = 2) :=
sorry

end find_values_l95_95120


namespace multiply_equality_l95_95545

variable (a b c d e : ℝ)

theorem multiply_equality
  (h1 : a = 2994)
  (h2 : b = 14.5)
  (h3 : c = 173)
  (h4 : d = 29.94)
  (h5 : e = 1.45)
  (h6 : a * b = c) : d * e = 1.73 :=
sorry

end multiply_equality_l95_95545


namespace smallest_four_digit_multiple_of_18_l95_95637

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 18 ∣ n ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ 18 ∣ m → n ≤ m := by
  use 1008
  sorry

end smallest_four_digit_multiple_of_18_l95_95637


namespace polygon_sides_l95_95013

/-- If the sum of the interior angles of a polygon is three times the sum of its exterior angles,
    then the number of sides of the polygon is 8. -/
theorem polygon_sides (n : ℕ) (h1 : 180 * (n - 2) = 3 * 360) : n = 8 :=
sorry

end polygon_sides_l95_95013


namespace sum_of_ratios_of_squares_l95_95882

theorem sum_of_ratios_of_squares (r : ℚ) (a b c : ℤ) (h1 : r = 45 / 64) 
  (h2 : r = (a * (Real.sqrt b)) / c) 
  (ha : a = 3) 
  (hb : b = 5) 
  (hc : c = 8) : a + b + c = 16 := 
by
  sorry

end sum_of_ratios_of_squares_l95_95882


namespace adults_wearing_sunglasses_l95_95326

def total_adults : ℕ := 2400
def one_third_of_adults (total : ℕ) : ℕ := total / 3
def women_wearing_sunglasses (women : ℕ) : ℕ := (15 * women) / 100
def men_wearing_sunglasses (men : ℕ) : ℕ := (12 * men) / 100

theorem adults_wearing_sunglasses : 
  let women := one_third_of_adults total_adults
  let men := total_adults - women
  let women_in_sunglasses := women_wearing_sunglasses women
  let men_in_sunglasses := men_wearing_sunglasses men
  women_in_sunglasses + men_in_sunglasses = 312 :=
by
  sorry

end adults_wearing_sunglasses_l95_95326


namespace find_x_l95_95087

open Real

def vector (a b : ℝ) : ℝ × ℝ := (a, b)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

def problem_statement (x : ℝ) : Prop :=
  let m := vector 2 x
  let n := vector 4 (-2)
  let m_minus_n := vector (2 - 4) (x - (-2))
  perpendicular m m_minus_n → x = -1 + sqrt 5 ∨ x = -1 - sqrt 5

-- We assert the theorem based on the problem statement
theorem find_x (x : ℝ) : problem_statement x :=
  sorry

end find_x_l95_95087


namespace num_prime_factors_30_fact_l95_95417

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def is_prime (n : ℕ) : Bool :=
  if h : n ≤ 1 then false else
    let divisors := List.range (n - 2) |>.map (· + 2)
    !divisors.any (· ∣ n)

def primes_upto (n : ℕ) : List ℕ :=
  List.range (n - 1) |>.map (· + 1) |>.filter is_prime

def count_primes_factorial_upto (n : ℕ) : ℕ :=
  (primes_upto n).length

theorem num_prime_factors_30_fact : count_primes_factorial_upto 30 = 10 := sorry

end num_prime_factors_30_fact_l95_95417


namespace symmetry_y_axis_l95_95079

theorem symmetry_y_axis (A B C D : ℝ → ℝ → Prop) 
  (A_eq : ∀ x y : ℝ, A x y ↔ (x^2 - x + y^2 = 1))
  (B_eq : ∀ x y : ℝ, B x y ↔ (x^2 * y + x * y^2 = 1))
  (C_eq : ∀ x y : ℝ, C x y ↔ (x^2 - y^2 = 1))
  (D_eq : ∀ x y : ℝ, D x y ↔ (x - y = 1)) : 
  (∀ x y : ℝ, C x y ↔ C (-x) y) ∧ 
  ¬(∀ x y : ℝ, A x y ↔ A (-x) y) ∧ 
  ¬(∀ x y : ℝ, B x y ↔ B (-x) y) ∧ 
  ¬(∀ x y : ℝ, D x y ↔ D (-x) y) :=
by
  -- Proof goes here
  sorry

end symmetry_y_axis_l95_95079


namespace calculate_x_l95_95505

theorem calculate_x :
  let a := 3
  let b := 5
  let c := 2
  let d := 4
  let term1 := (a ^ 2) * b * 0.47 * 1442
  let term2 := c * d * 0.36 * 1412
  (term1 - term2) + 63 = 26544.74 := by
  sorry

end calculate_x_l95_95505


namespace problem_six_circles_l95_95707

noncomputable def six_circles_centers : List (ℝ × ℝ) := [(1,1), (1,3), (3,1), (3,3), (5,1), (5,3)]

noncomputable def slope_of_line_dividing_circles := (2 : ℝ)

def gcd_is_1 (p q r : ℕ) : Prop := Nat.gcd (Nat.gcd p q) r = 1

theorem problem_six_circles (p q r : ℕ) (h_gcd : gcd_is_1 p q r)
  (h_line_eq : ∀ x y, y = slope_of_line_dividing_circles * x - 3 → px = qy + r) :
  p^2 + q^2 + r^2 = 14 :=
sorry

end problem_six_circles_l95_95707


namespace evaluate_expression_l95_95600

theorem evaluate_expression (x : ℝ) (h : 3 * x - 2 = 13) : 6 * x - 4 = 26 :=
by {
    sorry
}

end evaluate_expression_l95_95600


namespace percentage_of_games_lost_l95_95669

theorem percentage_of_games_lost (games_won games_lost games_tied total_games : ℕ)
  (h_ratio : 5 * games_lost = 3 * games_won)
  (h_tied : games_tied * 5 = total_games) :
  (games_lost * 10 / total_games) = 3 :=
by sorry

end percentage_of_games_lost_l95_95669


namespace number_of_parakeets_per_cage_l95_95846

def num_cages : ℕ := 9
def parrots_per_cage : ℕ := 2
def total_birds : ℕ := 72

theorem number_of_parakeets_per_cage : (total_birds - (num_cages * parrots_per_cage)) / num_cages = 6 := by
  sorry

end number_of_parakeets_per_cage_l95_95846


namespace people_to_right_of_taehyung_l95_95503

-- Given conditions
def total_people : Nat := 11
def people_to_left_of_taehyung : Nat := 5

-- Question and proof: How many people are standing to Taehyung's right?
theorem people_to_right_of_taehyung : total_people - people_to_left_of_taehyung - 1 = 4 :=
by
  sorry

end people_to_right_of_taehyung_l95_95503


namespace find_y_l95_95048

theorem find_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x = 2 + 1/y) (h2 : y = 2 + 1/x) : y = x :=
sorry

end find_y_l95_95048


namespace sum_of_terms_l95_95870

def sequence_sum (n : ℕ) : ℕ :=
  n^2 + 2*n + 5

theorem sum_of_terms : sequence_sum 9 - sequence_sum 6 = 51 :=
by
  sorry

end sum_of_terms_l95_95870


namespace analytical_expression_l95_95694

theorem analytical_expression (k : ℝ) (h : k ≠ 0) (x y : ℝ) (hx : x = 4) (hy : y = 6) 
  (eqn : y = k * x) : y = (3 / 2) * x :=
by {
  sorry
}

end analytical_expression_l95_95694


namespace find_a_l95_95999

noncomputable def binomialExpansion (a : ℚ) (x : ℚ) := (x - a / x) ^ 6

theorem find_a (a : ℚ) (A : ℚ) (B : ℚ) (hA : A = 15 * a ^ 2) (hB : B = -20 * a ^ 3) (hB_value : B = 44) :
  a = -22 / 5 :=
by
  sorry -- skipping the proof

end find_a_l95_95999


namespace solve_for_x_l95_95771

theorem solve_for_x : ∀ (x : ℝ), (4 * x^2 - 3 * x + 2) / (x + 2) = 4 * x - 5 → x = 2 :=
by
  intros x h
  sorry

end solve_for_x_l95_95771


namespace pythagorean_triple_divisibility_l95_95390

theorem pythagorean_triple_divisibility (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (∃ k₃, k₃ ∣ a ∨ k₃ ∣ b) ∧
  (∃ k₄, k₄ ∣ a ∨ k₄ ∣ b ∧ 2 ∣ k₄) ∧
  (∃ k₅, k₅ ∣ a ∨ k₅ ∣ b ∨ k₅ ∣ c) :=
by
  sorry

end pythagorean_triple_divisibility_l95_95390


namespace inequality_proof_l95_95232

noncomputable def a (x1 x2 x3 x4 x5 : ℝ) := x1 + x2 + x3 + x4 + x5
noncomputable def b (x1 x2 x3 x4 x5 : ℝ) := x1 * x2 + x1 * x3 + x1 * x4 + x1 * x5 + x2 * x3 + x2 * x4 + x2 * x5 + x3 * x4 + x3 * x5 + x4 * x5
noncomputable def c (x1 x2 x3 x4 x5 : ℝ) := x1 * x2 * x3 + x1 * x2 * x4 + x1 * x2 * x5 + x1 * x3 * x4 + x1 * x3 * x5 + x1 * x4 * x5 + x2 * x3 * x4 + x2 * x3 * x5 + x2 * x4 * x5 + x3 * x4 * x5
noncomputable def d (x1 x2 x3 x4 x5 : ℝ) := x1 * x2 * x3 * x4 + x1 * x2 * x3 * x5 + x1 * x2 * x4 * x5 + x1 * x3 * x4 * x5 + x2 * x3 * x4 * x5

theorem inequality_proof (x1 x2 x3 x4 x5 : ℝ) (hx1x2x3x4x5 : x1 * x2 * x3 * x4 * x5 = 1) :
  (1 / a x1 x2 x3 x4 x5) + (1 / b x1 x2 x3 x4 x5) + (1 / c x1 x2 x3 x4 x5) + (1 / d x1 x2 x3 x4 x5) ≤ 3 / 5 := 
sorry

end inequality_proof_l95_95232


namespace cost_per_chair_l95_95137

theorem cost_per_chair (total_spent : ℕ) (chairs_bought : ℕ) (cost : ℕ) 
  (h1 : total_spent = 180) 
  (h2 : chairs_bought = 12) 
  (h3 : cost = total_spent / chairs_bought) : 
  cost = 15 :=
by
  -- Proof steps go here (skipped with sorry)
  sorry

end cost_per_chair_l95_95137


namespace average_length_is_21_08_l95_95171

def lengths : List ℕ := [20, 21, 22]
def quantities : List ℕ := [23, 64, 32]

def total_length := List.sum (List.zipWith (· * ·) lengths quantities)
def total_quantity := List.sum quantities

def average_length := total_length / total_quantity

theorem average_length_is_21_08 :
  average_length = 2508 / 119 := by
  sorry

end average_length_is_21_08_l95_95171


namespace arithmetic_sequence_max_value_l95_95496

theorem arithmetic_sequence_max_value 
  (S : ℕ → ℤ)
  (k : ℕ)
  (h1 : 2 ≤ k)
  (h2 : S (k - 1) = 8)
  (h3 : S k = 0)
  (h4 : S (k + 1) = -10) :
  ∃ n, S n = 20 ∧ (∀ m, S m ≤ 20) :=
sorry

end arithmetic_sequence_max_value_l95_95496


namespace solve_inequality_l95_95706

open Real

theorem solve_inequality (f : ℝ → ℝ)
  (h_cos : ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f (cos x) ≥ 0) :
  ∀ k : ℤ, ∀ x, (2 * ↑k * π ≤ x ∧ x ≤ 2 * ↑k * π + π) → f (sin x) ≥ 0 :=
by
  intros k x hx
  sorry

end solve_inequality_l95_95706


namespace cosine_expression_rewrite_l95_95806

theorem cosine_expression_rewrite (x : ℝ) :
  ∃ a b c d : ℕ, 
    a * (Real.cos (b * x) * Real.cos (c * x) * Real.cos (d * x)) = 
    Real.cos (2 * x) + Real.cos (6 * x) + Real.cos (14 * x) + Real.cos (18 * x) 
    ∧ a + b + c + d = 22 := sorry

end cosine_expression_rewrite_l95_95806


namespace liu_xiang_hurdles_l95_95165

theorem liu_xiang_hurdles :
  let total_distance := 110
  let first_hurdle_distance := 13.72
  let last_hurdle_distance := 14.02
  let best_time_first_segment := 2.5
  let best_time_last_segment := 1.4
  let hurdle_cycle_time := 0.96
  let num_hurdles := 10
  (total_distance - first_hurdle_distance - last_hurdle_distance) / num_hurdles = 8.28 ∧
  best_time_first_segment + num_hurdles * hurdle_cycle_time + best_time_last_segment  = 12.1 :=
by
  sorry

end liu_xiang_hurdles_l95_95165


namespace find_f_of_power_function_l95_95469

theorem find_f_of_power_function (a : ℝ) (alpha : ℝ) (f : ℝ → ℝ) 
  (h1 : 0 < a ∧ a ≠ 1) 
  (h2 : ∀ x, f x = x^alpha) 
  (h3 : ∀ x, a^(x-2) + 3 = f (2)): 
  f 2 = 4 := 
  sorry

end find_f_of_power_function_l95_95469


namespace tournament_participants_l95_95195

theorem tournament_participants (n : ℕ) (h : (n * (n - 1)) / 2 = 171) : n = 19 :=
by
  sorry

end tournament_participants_l95_95195


namespace speed_of_sound_l95_95975

theorem speed_of_sound (time_heard : ℕ) (time_occured : ℕ) (distance : ℝ) : 
  time_heard = 30 * 60 + 20 → 
  time_occured = 30 * 60 → 
  distance = 6600 → 
  (distance / ((time_heard - time_occured) / 3600)) / 3600 = 330 :=
by 
  intros h1 h2 h3
  sorry

end speed_of_sound_l95_95975


namespace value_of_linear_combination_l95_95067

theorem value_of_linear_combination :
  ∀ (x1 x2 x3 x4 x5 : ℝ),
    2*x1 + x2 + x3 + x4 + x5 = 6 →
    x1 + 2*x2 + x3 + x4 + x5 = 12 →
    x1 + x2 + 2*x3 + x4 + x5 = 24 →
    x1 + x2 + x3 + 2*x4 + x5 = 48 →
    x1 + x2 + x3 + x4 + 2*x5 = 96 →
    3*x4 + 2*x5 = 181 :=
by
  intros x1 x2 x3 x4 x5 h1 h2 h3 h4 h5
  sorry

end value_of_linear_combination_l95_95067


namespace gcd_228_1995_base3_to_base6_conversion_l95_95051

-- Proof Problem 1: GCD of 228 and 1995 is 57
theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 :=
by
  sorry

-- Proof Problem 2: Converting base-3 number 11102 to base-6
theorem base3_to_base6_conversion : Nat.ofDigits 6 [3, 1, 5] = Nat.ofDigits 10 [1, 1, 1, 0, 2] :=
by
  sorry

end gcd_228_1995_base3_to_base6_conversion_l95_95051


namespace correct_figure_is_D_l95_95810

def option_A : Prop := sorry -- placeholder for option A as a diagram representation
def option_B : Prop := sorry -- placeholder for option B as a diagram representation
def option_C : Prop := sorry -- placeholder for option C as a diagram representation
def option_D : Prop := sorry -- placeholder for option D as a diagram representation
def equilateral_triangle (figure : Prop) : Prop := sorry -- placeholder for the condition representing an equilateral triangle in the oblique projection method

theorem correct_figure_is_D : equilateral_triangle option_D := 
sorry

end correct_figure_is_D_l95_95810


namespace candidate_lost_by_votes_l95_95229

theorem candidate_lost_by_votes :
  let candidate_votes := (31 / 100) * 6450
  let rival_votes := (69 / 100) * 6450
  candidate_votes <= 6450 ∧ rival_votes <= 6450 ∧ rival_votes - candidate_votes = 2451 :=
by
  let candidate_votes := (31 / 100) * 6450
  let rival_votes := (69 / 100) * 6450
  have h1: candidate_votes <= 6450 := sorry
  have h2: rival_votes <= 6450 := sorry
  have h3: rival_votes - candidate_votes = 2451 := sorry
  exact ⟨h1, h2, h3⟩

end candidate_lost_by_votes_l95_95229


namespace paint_required_for_frame_l95_95138

theorem paint_required_for_frame :
  ∀ (width height thickness : ℕ) 
    (coverage : ℚ),
  width = 6 →
  height = 9 →
  thickness = 1 →
  coverage = 5 →
  (width * height - (width - 2 * thickness) * (height - 2 * thickness) + 2 * width * thickness + 2 * height * thickness) / coverage = 11.2 :=
by
  intros
  sorry

end paint_required_for_frame_l95_95138


namespace garden_stone_calculation_l95_95657

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

end garden_stone_calculation_l95_95657


namespace total_number_of_notes_l95_95381

-- The total amount of money in Rs.
def total_amount : ℕ := 400

-- The number of each type of note is equal.
variable (n : ℕ)

-- The total value equation given the number of each type of note.
def total_value : ℕ := n * 1 + n * 5 + n * 10

-- Prove that if the total value equals 400, the total number of notes is 75.
theorem total_number_of_notes : total_value n = total_amount → 3 * n = 75 :=
by
  sorry

end total_number_of_notes_l95_95381


namespace locus_of_intersection_l95_95916

-- Define the conditions
def line_e (m_e x y : ℝ) : Prop := y = m_e * (x - 1) + 1
def line_f (m_f x y : ℝ) : Prop := y = m_f * (x + 1) + 1
def slope_diff_cond (m_e m_f : ℝ) : Prop := (m_e - m_f = 2 ∨ m_f - m_e = 2)
def not_at_points (x y : ℝ) : Prop := (x, y) ≠ (1, 1) ∧ (x, y) ≠ (-1, 1)

-- Define the proof problem
theorem locus_of_intersection (x y m_e m_f : ℝ) :
  line_e m_e x y → line_f m_f x y → slope_diff_cond m_e m_f → not_at_points x y →
  (y = x^2 ∨ y = 2 - x^2) :=
by
  intros he hf h_diff h_not_at
  sorry

end locus_of_intersection_l95_95916


namespace odd_two_digit_combinations_l95_95338

theorem odd_two_digit_combinations (digits : Finset ℕ) (h_digits : digits = {1, 3, 5, 7, 9}) :
  ∃ n : ℕ, n = 20 ∧ (∃ a b : ℕ, a ∈ digits ∧ b ∈ digits ∧ a ≠ b ∧ (10 * a + b) % 2 = 1) :=
by
  sorry

end odd_two_digit_combinations_l95_95338


namespace double_acute_angle_is_positive_and_less_than_180_l95_95450

variable (α : ℝ) (h : 0 < α ∧ α < π / 2)

theorem double_acute_angle_is_positive_and_less_than_180 :
  0 < 2 * α ∧ 2 * α < π :=
by
  sorry

end double_acute_angle_is_positive_and_less_than_180_l95_95450


namespace g_solution_l95_95568

noncomputable def g : ℝ → ℝ := sorry

axiom g_0 : g 0 = 2
axiom g_functional : ∀ x y : ℝ, g (x * y) = g ((x^2 + y^2) / 2) + (x - y)^2 + x^2

theorem g_solution :
  ∀ x : ℝ, g x = 2 - 2 * x := sorry

end g_solution_l95_95568


namespace bake_sale_money_raised_correct_l95_95151

def bake_sale_money_raised : Prop :=
  let chocolate_chip_cookies_baked := 4 * 12
  let oatmeal_raisin_cookies_baked := 6 * 12
  let regular_brownies_baked := 2 * 12
  let sugar_cookies_baked := 6 * 12
  let blondies_baked := 3 * 12
  let cream_cheese_swirled_brownies_baked := 5 * 12
  let chocolate_chip_cookies_price := 1.50
  let oatmeal_raisin_cookies_price := 1.00
  let regular_brownies_price := 2.50
  let sugar_cookies_price := 1.25
  let blondies_price := 2.75
  let cream_cheese_swirled_brownies_price := 3.00
  let chocolate_chip_cookies_sold := 0.75 * chocolate_chip_cookies_baked
  let oatmeal_raisin_cookies_sold := 0.85 * oatmeal_raisin_cookies_baked
  let regular_brownies_sold := 0.60 * regular_brownies_baked
  let sugar_cookies_sold := 0.90 * sugar_cookies_baked
  let blondies_sold := 0.80 * blondies_baked
  let cream_cheese_swirled_brownies_sold := 0.50 * cream_cheese_swirled_brownies_baked
  let total_money_raised := 
    chocolate_chip_cookies_sold * chocolate_chip_cookies_price + 
    oatmeal_raisin_cookies_sold * oatmeal_raisin_cookies_price + 
    regular_brownies_sold * regular_brownies_price + 
    sugar_cookies_sold * sugar_cookies_price + 
    blondies_sold * blondies_price + 
    cream_cheese_swirled_brownies_sold * cream_cheese_swirled_brownies_price
  total_money_raised = 397.00

theorem bake_sale_money_raised_correct : bake_sale_money_raised := by
  sorry

end bake_sale_money_raised_correct_l95_95151


namespace number_of_divisors_of_720_l95_95130

theorem number_of_divisors_of_720 : 
  let n := 720
  let prime_factorization := [(2, 4), (3, 2), (5, 1)] 
  let num_divisors := (4 + 1) * (2 + 1) * (1 + 1)
  n = 2^4 * 3^2 * 5^1 →
  num_divisors = 30 := 
by
  -- Placeholder for the proof
  sorry

end number_of_divisors_of_720_l95_95130


namespace total_books_sold_l95_95647

theorem total_books_sold (tuesday_books wednesday_books thursday_books : Nat) 
  (h1 : tuesday_books = 7) 
  (h2 : wednesday_books = 3 * tuesday_books) 
  (h3 : thursday_books = 3 * wednesday_books) : 
  tuesday_books + wednesday_books + thursday_books = 91 := 
by 
  sorry

end total_books_sold_l95_95647


namespace smallest_integer_inequality_l95_95738

theorem smallest_integer_inequality:
  ∃ x : ℤ, (2 * x < 3 * x - 10) ∧ ∀ y : ℤ, (2 * y < 3 * y - 10) → y ≥ 11 := by
  sorry

end smallest_integer_inequality_l95_95738


namespace ratio_of_sphere_surface_areas_l95_95111

noncomputable def inscribed_sphere_radius (a : ℝ) : ℝ := a / 2
noncomputable def circumscribed_sphere_radius (a : ℝ) : ℝ := a * (Real.sqrt 3) / 2
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem ratio_of_sphere_surface_areas (a : ℝ) (h : 0 < a) : 
  (sphere_surface_area (circumscribed_sphere_radius a)) / (sphere_surface_area (inscribed_sphere_radius a)) = 3 :=
by
  sorry

end ratio_of_sphere_surface_areas_l95_95111


namespace problem1_solution_problem2_solution_l95_95571

noncomputable def problem1 (α : ℝ) (h : Real.tan α = -2) : Real :=
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α)

theorem problem1_solution (α : ℝ) (h : Real.tan α = -2) : problem1 α h = 5 := by
  sorry

noncomputable def problem2 (α : ℝ) (h : Real.tan α = -2) : Real :=
  1 / (Real.sin α * Real.cos α)

theorem problem2_solution (α : ℝ) (h : Real.tan α = -2) : problem2 α h = -5 / 2 := by
  sorry

end problem1_solution_problem2_solution_l95_95571


namespace sequence_inequality_l95_95670
open Nat

variable (a : ℕ → ℝ)

noncomputable def conditions := 
  (a 1 ≥ 1) ∧ (∀ k : ℕ, a (k + 1) - a k ≥ 1)

theorem sequence_inequality (h : conditions a) : 
  ∀ n : ℕ, a (n + 1) ≥ n + 1 :=
sorry

end sequence_inequality_l95_95670


namespace evaluate_expression_l95_95997

theorem evaluate_expression : (3 / (2 - (4 / (-5)))) = (15 / 14) :=
by
  sorry

end evaluate_expression_l95_95997


namespace translation_correct_l95_95169

def vector_a : ℝ × ℝ := (1, 1)

def translate_right (v : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (v.1 + d, v.2)
def translate_down (v : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (v.1, v.2 - d)

def vector_b := translate_down (translate_right vector_a 2) 1

theorem translation_correct :
  vector_b = (3, 0) :=
by
  -- proof steps will go here
  sorry

end translation_correct_l95_95169


namespace tyler_bought_10_erasers_l95_95790

/--
Given that Tyler initially has $100, buys 8 scissors for $5 each, buys some erasers for $4 each,
and has $20 remaining after these purchases, prove that he bought 10 erasers.
-/
theorem tyler_bought_10_erasers : ∀ (initial_money scissors_cost erasers_cost remaining_money : ℕ), 
  initial_money = 100 →
  scissors_cost = 5 →
  erasers_cost = 4 →
  remaining_money = 20 →
  ∃ (scissors_count erasers_count : ℕ),
    scissors_count = 8 ∧ 
    initial_money - scissors_count * scissors_cost - erasers_count * erasers_cost = remaining_money ∧ 
    erasers_count = 10 :=
by
  intros
  sorry

end tyler_bought_10_erasers_l95_95790


namespace reflect_P_across_x_axis_l95_95711

def point_reflection_over_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

theorem reflect_P_across_x_axis : 
  point_reflection_over_x_axis (-3, 1) = (-3, -1) :=
  by
    sorry

end reflect_P_across_x_axis_l95_95711


namespace cube_edge_length_l95_95451

theorem cube_edge_length (sum_of_edges : ℕ) (num_edges : ℕ) (h : sum_of_edges = 144) (num_edges_h : num_edges = 12) :
  sum_of_edges / num_edges = 12 :=
by
  -- The proof is skipped.
  sorry

end cube_edge_length_l95_95451


namespace more_oranges_than_apples_l95_95791

-- Definitions based on conditions
def apples : ℕ := 14
def oranges : ℕ := 2 * 12  -- 2 dozen oranges

-- Statement to prove
theorem more_oranges_than_apples : oranges - apples = 10 := by
  sorry

end more_oranges_than_apples_l95_95791


namespace terminal_side_of_angle_l95_95839

theorem terminal_side_of_angle (θ : Real) (h_cos : Real.cos θ < 0) (h_tan : Real.tan θ > 0) :
  θ ∈ {φ : Real | π < φ ∧ φ < 3 * π / 2} :=
sorry

end terminal_side_of_angle_l95_95839


namespace general_form_of_equation_l95_95535

theorem general_form_of_equation : 
  ∀ x : ℝ, (x - 1) * (x - 2) = 4 → x^2 - 3 * x - 2 = 0 := by
  sorry

end general_form_of_equation_l95_95535


namespace edward_games_start_l95_95825

theorem edward_games_start (sold_games : ℕ) (boxes : ℕ) (games_per_box : ℕ) 
  (h_sold : sold_games = 19) (h_boxes : boxes = 2) (h_game_box : games_per_box = 8) : 
  sold_games + boxes * games_per_box = 35 := 
  by 
    sorry

end edward_games_start_l95_95825


namespace power_addition_l95_95347

theorem power_addition {a m n : ℝ} (h1 : a^m = 2) (h2 : a^n = 8) : a^(m + n) = 16 :=
sorry

end power_addition_l95_95347


namespace quadratic_inequality_solution_l95_95952

theorem quadratic_inequality_solution (m: ℝ) (h: m > 1) :
  { x : ℝ | x^2 + (m - 1) * x - m ≥ 0 } = { x | x ≤ -m ∨ x ≥ 1 } :=
sorry

end quadratic_inequality_solution_l95_95952


namespace equality_equiv_l95_95086

-- Problem statement
theorem equality_equiv (a b c : ℝ) :
  (a + b + c ≠ 0 → ( (a * (b - c)) / (b + c) + (b * (c - a)) / (c + a) + (c * (a - b)) / (a + b) = 0 ↔
  (a^2 * (b - c)) / (b + c) + (b^2 * (c - a)) / (c + a) + (c^2 * (a - b)) / (a + b) = 0)) ∧
  (a + b + c = 0 → ∀ w x y z: ℝ, w * x + y * z = 0) :=
by
  sorry

end equality_equiv_l95_95086


namespace first_bag_weight_l95_95980

def weight_of_first_bag (initial_weight : ℕ) (second_bag : ℕ) (total_weight : ℕ) : ℕ :=
  total_weight - second_bag - initial_weight

theorem first_bag_weight : weight_of_first_bag 15 10 40 = 15 :=
by
  unfold weight_of_first_bag
  sorry

end first_bag_weight_l95_95980


namespace sequence_product_mod_five_l95_95742

theorem sequence_product_mod_five : 
  let seq := List.range 20 |>.map (λ k => 10 * k + 3)
  seq.prod % 5 = 1 := 
by
  sorry

end sequence_product_mod_five_l95_95742


namespace length_of_train_is_135_l95_95906

noncomputable def length_of_train (v : ℝ) (t : ℝ) : ℝ :=
  ((v * 1000) / 3600) * t

theorem length_of_train_is_135 :
  length_of_train 140 3.4711508793582233 = 135 :=
sorry

end length_of_train_is_135_l95_95906


namespace option_d_is_true_l95_95820

theorem option_d_is_true (x : ℝ) : (4 * x) / (x^2 + 4) ≤ 1 := 
  sorry

end option_d_is_true_l95_95820


namespace find_subsequence_with_sum_n_l95_95914

theorem find_subsequence_with_sum_n (n : ℕ) (a : Fin n → ℕ) (h1 : ∀ i, a i ∈ Finset.range n) 
  (h2 : (Finset.univ.sum a) < 2 * n) : 
  ∃ s : Finset (Fin n), s.sum a = n := 
sorry

end find_subsequence_with_sum_n_l95_95914


namespace max_digits_product_l95_95359

def digitsProduct (A B : ℕ) : ℕ := A * B

theorem max_digits_product 
  (A B : ℕ) 
  (h1 : A + B + 5 ≡ 0 [MOD 9]) 
  (h2 : 0 ≤ A ∧ A ≤ 9) 
  (h3 : 0 ≤ B ∧ B ≤ 9) 
  : digitsProduct A B = 42 := 
sorry

end max_digits_product_l95_95359


namespace find_g2_l95_95988

theorem find_g2
  (g : ℝ → ℝ)
  (h : ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = x ^ 2) :
  g 2 = 19 / 16 := 
sorry

end find_g2_l95_95988


namespace max_soap_boxes_l95_95070

theorem max_soap_boxes 
  (base_width base_length top_width top_length height soap_width soap_length soap_height max_weight soap_weight : ℝ)
  (h_base_dims : base_width = 25)
  (h_base_len : base_length = 42)
  (h_top_width : top_width = 20)
  (h_top_length : top_length = 35)
  (h_height : height = 60)
  (h_soap_width : soap_width = 7)
  (h_soap_length : soap_length = 6)
  (h_soap_height : soap_height = 10)
  (h_max_weight : max_weight = 150)
  (h_soap_weight : soap_weight = 3) :
  (50 = 
    min 
      (⌊top_width / soap_width⌋ * ⌊top_length / soap_length⌋ * ⌊height / soap_height⌋)
      (⌊max_weight / soap_weight⌋)) := by sorry

end max_soap_boxes_l95_95070


namespace smallest_y_l95_95878

theorem smallest_y (y : ℤ) (h : y < 3 * y - 15) : y = 8 :=
  sorry

end smallest_y_l95_95878


namespace train_journey_time_l95_95131

theorem train_journey_time {X : ℝ} (h1 : 0 < X) (h2 : X < 60) (h3 : ∀ T_A M_A T_B M_B : ℝ, M_A - T_A = X ∧ M_B - T_B = X) :
    X = 360 / 7 :=
by
  sorry

end train_journey_time_l95_95131


namespace value_of_a6_l95_95987

theorem value_of_a6 (a : ℕ → ℝ) (h_positive : ∀ n, 0 < a n)
  (h_a1 : a 1 = 1) (h_a2 : a 2 = 2)
  (h_recurrence : ∀ n, 2 * (a n)^2 = (a (n + 1))^2 + (a (n - 1))^2) :
  a 6 = 4 := 
sorry

end value_of_a6_l95_95987


namespace value_of_y_l95_95606

theorem value_of_y (x y z : ℕ) (h1 : 3 * x = 3 / 4 * y) (h2 : x + z = 24) (h3 : z = 8) : y = 64 :=
by
  -- Proof omitted
  sorry

end value_of_y_l95_95606


namespace circles_positional_relationship_l95_95512

theorem circles_positional_relationship :
  ∃ R r : ℝ, (R * r = 2 ∧ R + r = 3) ∧ 3 = R + r → "externally tangent" = "externally tangent" :=
by
  sorry

end circles_positional_relationship_l95_95512


namespace inverse_function_fixed_point_l95_95421

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-2) - 1

theorem inverse_function_fixed_point
  (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  ∃ g : ℝ → ℝ, (∀ y : ℝ, g (f a y) = y) ∧ g 0 = 2 :=
sorry

end inverse_function_fixed_point_l95_95421


namespace max_a4a7_value_l95_95907

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := ∀ n m : ℕ, a (n + 1) = a n + d

-- Given conditions
def given_conditions (a : ℕ → ℝ) (d : ℝ) : Prop := 
  arithmetic_sequence a d ∧ a 5 = 4 -- a6 = 4 so we use index 5 since Lean is 0-indexed

-- Define the product a4 * a7
def a4a7_product (a : ℕ → ℝ) (d : ℝ) : ℝ := (a 5 - 2 * d) * (a 5 + d)

-- The maximum value of a4 * a7
def max_a4a7 (a : ℕ → ℝ) (d : ℝ) : ℝ := 18

-- The proof problem statement
theorem max_a4a7_value (a : ℕ → ℝ) (d : ℝ) :
  given_conditions a d → a4a7_product a d = max_a4a7 a d :=
by
  sorry

end max_a4a7_value_l95_95907


namespace line_relation_with_plane_l95_95434

variables {P : Type} [Infinite P] [MetricSpace P]

variables (a b : Line P) (α : Plane P)

-- Conditions
axiom intersecting_lines : ∃ p : P, p ∈ a ∧ p ∈ b
axiom line_parallel_plane : ∀ p : P, p ∈ a → p ∈ α

-- Theorem statement for the proof problem
theorem line_relation_with_plane : (∀ p : P, p ∈ b → p ∈ α) ∨ (∃ q : P, q ∈ α ∧ q ∈ b) :=
sorry

end line_relation_with_plane_l95_95434


namespace oranges_per_glass_l95_95502

theorem oranges_per_glass (total_oranges glasses_of_juice oranges_per_glass : ℕ)
    (h_oranges : total_oranges = 12)
    (h_glasses : glasses_of_juice = 6) : 
    total_oranges / glasses_of_juice = oranges_per_glass :=
by 
    sorry

end oranges_per_glass_l95_95502


namespace common_chord_and_length_l95_95418

-- Define the two circles
def circle1 (x y : ℝ) := x^2 + y^2 + 2*x - 4*y - 5 = 0
def circle2 (x y : ℝ) := x^2 + y^2 + 2*x - 1 = 0

-- The theorem statement with the conditions and expected solutions
theorem common_chord_and_length :
  (∀ x y : ℝ, circle1 x y ∧ circle2 x y → y = -1)
  ∧
  (∃ A B : (ℝ × ℝ), (circle1 A.1 A.2 ∧ circle2 A.1 A.2) ∧ 
                    (circle1 B.1 B.2 ∧ circle2 B.1 B.2) ∧ 
                    (|A.1 - B.1|^2 + |A.2 - B.2|^2 = 4)) :=
by
  sorry

end common_chord_and_length_l95_95418


namespace rightmost_four_digits_of_5_pow_2023_l95_95708

theorem rightmost_four_digits_of_5_pow_2023 :
  (5 ^ 2023) % 10000 = 8125 :=
sorry

end rightmost_four_digits_of_5_pow_2023_l95_95708


namespace ratio_of_candies_l95_95074

theorem ratio_of_candies (candiesEmily candiesBob : ℕ) (candiesJennifer : ℕ) 
  (hEmily : candiesEmily = 6) 
  (hBob : candiesBob = 4)
  (hJennifer : candiesJennifer = 3 * candiesBob) : 
  (candiesJennifer / Nat.gcd candiesJennifer candiesEmily) = 2 ∧ (candiesEmily / Nat.gcd candiesJennifer candiesEmily) = 1 := 
by
  sorry

end ratio_of_candies_l95_95074


namespace x_squared_y_minus_xy_squared_l95_95860

theorem x_squared_y_minus_xy_squared (x y : ℝ) (h1 : x - y = -2) (h2 : x * y = 3) : x^2 * y - x * y^2 = -6 := 
by 
  sorry

end x_squared_y_minus_xy_squared_l95_95860


namespace solve_system_l95_95096

-- Define the conditions
def system_of_equations (x y : ℝ) : Prop :=
  (x + y = 8) ∧ (2 * x - y = 7)

-- Define the proof problem statement
theorem solve_system : 
  system_of_equations 5 3 :=
by
  -- Proof will be filled in here
  sorry

end solve_system_l95_95096


namespace point_in_or_on_circle_l95_95341

theorem point_in_or_on_circle (θ : Real) :
  let P := (5 * Real.cos θ, 4 * Real.sin θ)
  let C_eq := ∀ (x y : Real), x^2 + y^2 = 25
  25 * Real.cos θ ^ 2 + 16 * Real.sin θ ^ 2 ≤ 25 := 
by 
  sorry

end point_in_or_on_circle_l95_95341


namespace stratified_sampling_l95_95238

-- Definitions of the classes and their student counts
def class1_students : Nat := 54
def class2_students : Nat := 42

-- Definition of total students to be sampled
def total_sampled_students : Nat := 16

-- Definition of the number of students to be selected from each class
def students_selected_from_class1 : Nat := 9
def students_selected_from_class2 : Nat := 7

-- The proof problem
theorem stratified_sampling :
  students_selected_from_class1 + students_selected_from_class2 = total_sampled_students ∧ 
  students_selected_from_class1 * (class2_students + class1_students) = class1_students * total_sampled_students :=
by
  sorry

end stratified_sampling_l95_95238


namespace min_value_frac_inv_l95_95157

theorem min_value_frac_inv (a b : ℝ) (h1: a > 0) (h2: b > 0) (h3: a + 3 * b = 2) : 
  (2 + Real.sqrt 3) ≤ (1 / a + 1 / b) :=
sorry

end min_value_frac_inv_l95_95157


namespace problem_1992_AHSME_43_l95_95912

theorem problem_1992_AHSME_43 (a b c : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h1 : Odd a) (h2 : Odd b) : Odd (3^a + (b-1)^2 * c) :=
sorry

end problem_1992_AHSME_43_l95_95912


namespace fraction_equation_correct_l95_95225

theorem fraction_equation_correct : (1 / 2 - 1 / 6) / (1 / 6009) = 2003 := by
  sorry

end fraction_equation_correct_l95_95225


namespace least_money_Moe_l95_95244

theorem least_money_Moe (Bo Coe Flo Jo Moe Zoe : ℝ)
  (H1 : Flo > Jo) 
  (H2 : Flo > Bo) 
  (H3 : Bo > Zoe) 
  (H4 : Coe > Zoe) 
  (H5 : Jo > Zoe) 
  (H6 : Bo > Jo) 
  (H7 : Zoe > Moe) : 
  (Moe < Bo) ∧ (Moe < Coe) ∧ (Moe < Flo) ∧ (Moe < Jo) ∧ (Moe < Zoe) :=
by
  sorry

end least_money_Moe_l95_95244


namespace hall_ratio_l95_95021

open Real

theorem hall_ratio (w l : ℝ) (h_area : w * l = 288) (h_diff : l - w = 12) : w / l = 1 / 2 :=
by sorry

end hall_ratio_l95_95021


namespace five_people_six_chairs_l95_95039

/-- Number of ways to sit 5 people in 6 chairs -/
def ways_to_sit_in_chairs : ℕ :=
  6 * 5 * 4 * 3 * 2

theorem five_people_six_chairs : ways_to_sit_in_chairs = 720 := by
  -- placeholder for the proof
  sorry

end five_people_six_chairs_l95_95039


namespace calculate_expression_is_correct_l95_95617

noncomputable def calculate_expression : ℝ :=
  -(-2) + 2 * Real.cos (Real.pi / 3) + (-1 / 8)⁻¹ + (Real.pi - 3.14) ^ 0

theorem calculate_expression_is_correct :
  calculate_expression = -4 :=
by
  -- the conditions as definitions
  have h1 : Real.cos (Real.pi / 3) = 1 / 2 := by sorry
  have h2 : (Real.pi - 3.14) ^ 0 = 1 := by sorry
  -- use these conditions to prove the main statement
  sorry

end calculate_expression_is_correct_l95_95617


namespace ratio_initial_to_doubled_l95_95574

theorem ratio_initial_to_doubled (x : ℕ) (h : 3 * (2 * x + 9) = 63) : x / (2 * x) = 1 / 2 := 
by
  sorry

end ratio_initial_to_doubled_l95_95574


namespace closest_approx_of_q_l95_95558

theorem closest_approx_of_q :
  let result : ℝ := 69.28 * 0.004
  let q : ℝ := result / 0.03
  abs (q - 9.24) < 0.005 := 
by 
  let result : ℝ := 69.28 * 0.004
  let q : ℝ := result / 0.03
  sorry

end closest_approx_of_q_l95_95558


namespace two_digit_decimal_bounds_l95_95279

def is_approximate (original approx : ℝ) : Prop :=
  abs (original - approx) < 0.05

theorem two_digit_decimal_bounds :
  ∃ max min : ℝ, is_approximate 15.6 max ∧ max = 15.64 ∧ is_approximate 15.6 min ∧ min = 15.55 :=
by
  sorry

end two_digit_decimal_bounds_l95_95279


namespace symmetric_point_yOz_l95_95222

-- Given point A in 3D Cartesian system
def A : ℝ × ℝ × ℝ := (1, -3, 5)

-- Plane yOz where x = 0
def symmetric_yOz (point : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := point
  (-x, y, z)

-- Proof statement (without the actual proof)
theorem symmetric_point_yOz : symmetric_yOz A = (-1, -3, 5) :=
by sorry

end symmetric_point_yOz_l95_95222


namespace least_five_digit_is_15625_l95_95884

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l95_95884


namespace odd_function_expression_l95_95091

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_expression (x : ℝ) (h1 : x < 0 → f x = x^2 - x) (h2 : ∀ x, f (-x) = -f x) (h3 : 0 < x) :
  f x = -x^2 - x :=
sorry

end odd_function_expression_l95_95091


namespace find_k_for_infinite_solutions_l95_95022

noncomputable def has_infinitely_many_solutions (k : ℝ) : Prop :=
  ∀ x : ℝ, 5 * (3 * x - k) = 3 * (5 * x + 15)

theorem find_k_for_infinite_solutions :
  has_infinitely_many_solutions (-9) :=
by
  sorry

end find_k_for_infinite_solutions_l95_95022


namespace square_area_eq_1296_l95_95490

theorem square_area_eq_1296 (x : ℝ) (side : ℝ) (h1 : side = 6 * x - 18) (h2 : side = 3 * x + 9) : side ^ 2 = 1296 := sorry

end square_area_eq_1296_l95_95490


namespace afternoon_sales_l95_95082

theorem afternoon_sales (x : ℕ) (h : 3 * x = 510) : 2 * x = 340 :=
by sorry

end afternoon_sales_l95_95082


namespace no_real_satisfies_absolute_value_equation_l95_95331

theorem no_real_satisfies_absolute_value_equation :
  ∀ x : ℝ, ¬ (|x - 2| = |x - 1| + |x - 5|) :=
by
  sorry

end no_real_satisfies_absolute_value_equation_l95_95331


namespace analytic_expression_of_f_max_min_of_f_on_interval_l95_95483

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem analytic_expression_of_f :
  ∀ A ω φ : ℝ, (∀ x, f x = A * Real.sin (ω * x + φ)) →
  A = 2 ∧ ω = 2 ∧ φ = Real.pi / 6 :=
by
  sorry -- Placeholder for the actual proof

theorem max_min_of_f_on_interval :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 12 → f x ≤ Real.sqrt 3) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 12 → f x ≥ 1) :=
by
  sorry -- Placeholder for the actual proof

end analytic_expression_of_f_max_min_of_f_on_interval_l95_95483


namespace combination_mod_100_l95_95901

def totalDistinctHands : Nat := Nat.choose 60 12

def remainder (n : Nat) (m : Nat) : Nat := n % m

theorem combination_mod_100 :
  remainder totalDistinctHands 100 = R :=
sorry

end combination_mod_100_l95_95901


namespace number_exceeds_35_percent_by_245_l95_95482

theorem number_exceeds_35_percent_by_245 : 
  ∃ (x : ℝ), (0.35 * x + 245 = x) ∧ x = 376.92 := 
by
  sorry

end number_exceeds_35_percent_by_245_l95_95482


namespace tea_customers_count_l95_95518

theorem tea_customers_count :
  ∃ T : ℕ, 7 * 5 + T * 4 = 67 ∧ T = 8 :=
by
  sorry

end tea_customers_count_l95_95518


namespace interest_rate_l95_95017

theorem interest_rate (SI P T : ℕ) (h1 : SI = 2000) (h2 : P = 5000) (h3 : T = 10) :
  (SI = (P * R * T) / 100) -> R = 4 :=
by
  sorry

end interest_rate_l95_95017


namespace solve_for_x_l95_95318

theorem solve_for_x : ∃ x : ℝ, 3 * x - 6 = |(-20 + 5)| ∧ x = 7 := by
  sorry

end solve_for_x_l95_95318


namespace circle_tangent_to_parabola_directrix_l95_95427

theorem circle_tangent_to_parabola_directrix (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + m * x - 1/4 = 0 → y^2 = 4 * x → x = -1) → m = 3/4 :=
by
  sorry

end circle_tangent_to_parabola_directrix_l95_95427


namespace carB_highest_avg_speed_l95_95930

-- Define the distances and times for each car
def distanceA : ℕ := 715
def timeA : ℕ := 11
def distanceB : ℕ := 820
def timeB : ℕ := 12
def distanceC : ℕ := 950
def timeC : ℕ := 14

-- Define the average speeds
def avgSpeedA : ℚ := distanceA / timeA
def avgSpeedB : ℚ := distanceB / timeB
def avgSpeedC : ℚ := distanceC / timeC

theorem carB_highest_avg_speed : avgSpeedB > avgSpeedA ∧ avgSpeedB > avgSpeedC :=
by
  -- Proof will be filled in here
  sorry

end carB_highest_avg_speed_l95_95930


namespace Gake_needs_fewer_boards_than_Tom_l95_95500

noncomputable def Tom_boards_needed : ℕ :=
  let width_board : ℕ := 5
  let width_char : ℕ := 9
  (3 * width_char + 2 * 6) / width_board

noncomputable def Gake_boards_needed : ℕ :=
  let width_board : ℕ := 5
  let width_char : ℕ := 9
  (4 * width_char + 3 * 1) / width_board

theorem Gake_needs_fewer_boards_than_Tom :
  Gake_boards_needed < Tom_boards_needed :=
by
  -- Here you will put the actual proof steps
  sorry

end Gake_needs_fewer_boards_than_Tom_l95_95500


namespace range_of_a_l95_95486

theorem range_of_a (a : ℝ) : (∃ x₀ : ℝ, a * x₀^2 + 2 * x₀ + 1 < 0) ↔ a < 1 :=
by
  sorry

end range_of_a_l95_95486


namespace triangle_area_is_zero_l95_95976

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def vector_sub (p1 p2 : Point3D) : Point3D := {
  x := p1.x - p2.x,
  y := p1.y - p2.y,
  z := p1.z - p2.z
}

def scalar_vector_mult (k : ℝ) (v : Point3D) : Point3D := {
  x := k * v.x,
  y := k * v.y,
  z := k * v.z
}

theorem triangle_area_is_zero : 
  let u := Point3D.mk 2 1 (-1)
  let v := Point3D.mk 5 4 1
  let w := Point3D.mk 11 10 5
  vector_sub w u = scalar_vector_mult 3 (vector_sub v u) →
-- If the points u, v, w are collinear, the area of the triangle formed by these points is zero:
  ∃ area : ℝ, area = 0 :=
by {
  sorry
}

end triangle_area_is_zero_l95_95976


namespace find_common_difference_l95_95966

theorem find_common_difference 
  (a : ℕ → ℝ)
  (a1 : a 1 = 5)
  (a25 : a 25 = 173)
  (h : ∀ n : ℕ, a (n+1) = a 1 + n * (a 2 - a 1)) : 
  a 2 - a 1 = 7 :=
by 
  sorry

end find_common_difference_l95_95966


namespace new_total_lifting_capacity_is_correct_l95_95991

-- Define the initial lifting capacities and improvements
def initial_clean_and_jerk : ℕ := 80
def initial_snatch : ℕ := 50
def clean_and_jerk_multiplier : ℕ := 2
def snatch_increment_percentage : ℕ := 80

-- Calculated values
def new_clean_and_jerk := initial_clean_and_jerk * clean_and_jerk_multiplier
def snatch_increment := initial_snatch * snatch_increment_percentage / 100
def new_snatch := initial_snatch + snatch_increment
def new_total_lifting_capacity := new_clean_and_jerk + new_snatch

-- Theorem statement to be proven
theorem new_total_lifting_capacity_is_correct :
  new_total_lifting_capacity = 250 := 
sorry

end new_total_lifting_capacity_is_correct_l95_95991


namespace find_angle_B_l95_95677

theorem find_angle_B 
  (a b : ℝ) (A B : ℝ) 
  (ha : a = 2 * Real.sqrt 2) 
  (hb : b = 2)
  (hA : A = Real.pi / 4) -- 45 degrees in radians
  (h_triangle : ∃ c, a^2 + b^2 - 2*a*b*Real.cos A = c^2 ∧ a^2 * Real.sin 45 = b^2 * Real.sin B) :
  B = Real.pi / 6 := -- 30 degrees in radians
sorry

end find_angle_B_l95_95677


namespace find_two_sets_l95_95125

theorem find_two_sets :
  ∃ (a1 a2 a3 a4 a5 b1 b2 b3 b4 b5 : ℕ),
    a1 + a2 + a3 + a4 + a5 = a1 * a2 * a3 * a4 * a5 ∧
    b1 + b2 + b3 + b4 + b5 = b1 * b2 * b3 * b4 * b5 ∧
    (a1, a2, a3, a4, a5) ≠ (b1, b2, b3, b4, b5) := by
  sorry

end find_two_sets_l95_95125


namespace rattlesnakes_count_l95_95828

theorem rattlesnakes_count (P B R V : ℕ) (h1 : P = 3 * B / 2) (h2 : V = 2 * 420 / 100) (h3 : P + R = 3 * 420 / 4) (h4 : P + B + R + V = 420) : R = 162 :=
by
  sorry

end rattlesnakes_count_l95_95828


namespace eighth_box_contains_65_books_l95_95176

theorem eighth_box_contains_65_books (total_books boxes first_seven_books per_box eighth_box : ℕ) :
  total_books = 800 →
  boxes = 8 →
  first_seven_books = 7 →
  per_box = 105 →
  eighth_box = total_books - (first_seven_books * per_box) →
  eighth_box = 65 := by
  sorry

end eighth_box_contains_65_books_l95_95176


namespace angles_terminal_yaxis_l95_95206

theorem angles_terminal_yaxis :
  {θ : ℝ | ∃ k : ℤ, θ = 2 * k * Real.pi + Real.pi / 2 ∨ θ = 2 * k * Real.pi + 3 * Real.pi / 2} =
  {θ : ℝ | ∃ n : ℤ, θ = n * Real.pi + Real.pi / 2} :=
by sorry

end angles_terminal_yaxis_l95_95206


namespace product_fraction_l95_95189

theorem product_fraction :
  (1 + 1/2) * (1 + 1/4) * (1 + 1/6) * (1 + 1/8) * (1 + 1/10) = 693 / 256 := by
  sorry

end product_fraction_l95_95189


namespace problem_equivalence_l95_95830

section ProblemDefinitions

def odd_function_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def statement_A (f : ℝ → ℝ) : Prop :=
  (∀ x < 0, f x = -Real.log (-x)) →
  odd_function_condition f →
  ∀ x > 0, f x ≠ -Real.log x

def statement_B (a : ℝ) : Prop :=
  Real.logb a (1 / 2) < 1 →
  (0 < a ∧ a < 1 / 2) ∨ (1 < a)

def statement_C : Prop :=
  ∀ x, (Real.logb 2 (Real.sqrt (x-1)) = (1/2) * Real.logb 2 x)

def statement_D (x1 x2 : ℝ) : Prop :=
  (x1 + Real.log x1 = 2) →
  (Real.log (1 - x2) - x2 = 1) →
  x1 + x2 = 1

end ProblemDefinitions

structure MathProofProblem :=
  (A : ∀ f : ℝ → ℝ, statement_A f)
  (B : ∀ a : ℝ, statement_B a)
  (C : statement_C)
  (D : ∀ x1 x2 : ℝ, statement_D x1 x2)

theorem problem_equivalence : MathProofProblem :=
  { A := sorry,
    B := sorry,
    C := sorry,
    D := sorry }

end problem_equivalence_l95_95830


namespace math_proof_problem_l95_95494

noncomputable def problem_statement : Prop :=
  let a_bound := 14
  let b_bound := 7
  let c_bound := 14
  let num_square_divisors := (a_bound / 2 + 1) * (b_bound / 2 + 1) * (c_bound / 2 + 1)
  let num_cube_divisors := (a_bound / 3 + 1) * (b_bound / 3 + 1) * (c_bound / 3 + 1)
  let num_sixth_power_divisors := (a_bound / 6 + 1) * (b_bound / 6 + 1) * (c_bound / 6 + 1)
  
  num_square_divisors + num_cube_divisors - num_sixth_power_divisors = 313

theorem math_proof_problem : problem_statement := by sorry

end math_proof_problem_l95_95494


namespace Tracy_sold_paintings_l95_95917

theorem Tracy_sold_paintings (num_people : ℕ) (group1_customers : ℕ) (group1_paintings : ℕ)
    (group2_customers : ℕ) (group2_paintings : ℕ) (group3_customers : ℕ) (group3_paintings : ℕ) 
    (total_paintings : ℕ) :
    num_people = 20 →
    group1_customers = 4 →
    group1_paintings = 2 →
    group2_customers = 12 →
    group2_paintings = 1 →
    group3_customers = 4 →
    group3_paintings = 4 →
    total_paintings = (group1_customers * group1_paintings) + (group2_customers * group2_paintings) + 
                      (group3_customers * group3_paintings) →
    total_paintings = 36 :=
by
  intros 
  -- including this to ensure the lean code passes syntax checks
  sorry

end Tracy_sold_paintings_l95_95917


namespace smallest_angle_of_triangle_l95_95709

theorem smallest_angle_of_triangle :
  ∀ a b c : ℝ, a = 2 * Real.sqrt 10 → b = 3 * Real.sqrt 5 → c = 5 → 
  ∃ α β γ : ℝ, α + β + γ = π ∧ α = 45 * (π / 180) ∧ (a = c → α < β ∧ α < γ) ∧ (b = c → β < α ∧ β < γ) ∧ (c = a → γ < α ∧ γ < β) → 
  α = 45 * (π / 180) := 
sorry

end smallest_angle_of_triangle_l95_95709


namespace tens_digit_17_pow_1993_l95_95800

theorem tens_digit_17_pow_1993 :
  (17 ^ 1993) % 100 / 10 = 3 := by
  sorry

end tens_digit_17_pow_1993_l95_95800


namespace probability_of_dime_l95_95332

noncomputable def num_quarters := 12 / 0.25
noncomputable def num_dimes := 8 / 0.10
noncomputable def num_pennies := 5 / 0.01
noncomputable def total_coins := num_quarters + num_dimes + num_pennies

theorem probability_of_dime : (num_dimes / total_coins) = (40 / 314) :=
by
  sorry

end probability_of_dime_l95_95332


namespace selection_probabilities_l95_95160

-- Define the probabilities of selection for Ram, Ravi, and Rani
def prob_ram : ℚ := 5 / 7
def prob_ravi : ℚ := 1 / 5
def prob_rani : ℚ := 3 / 4

-- State the theorem that combines these probabilities
theorem selection_probabilities : prob_ram * prob_ravi * prob_rani = 3 / 28 :=
by
  sorry


end selection_probabilities_l95_95160


namespace solve_x_squared_eq_four_l95_95772

theorem solve_x_squared_eq_four (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 := 
by sorry

end solve_x_squared_eq_four_l95_95772


namespace candy_cost_l95_95172

theorem candy_cost (J H C : ℕ) (h1 : J + 7 = C) (h2 : H + 1 = C) (h3 : J + H < C) : C = 7 :=
by
  sorry

end candy_cost_l95_95172


namespace kevin_total_distance_l95_95142

noncomputable def kevin_hop_total_distance_after_seven_leaps : ℚ :=
  let a := (1 / 4 : ℚ)
  let r := (3 / 4 : ℚ)
  let n := 7
  a * (1 - r^n) / (1 - r)

theorem kevin_total_distance (total_distance : ℚ) :
  total_distance = kevin_hop_total_distance_after_seven_leaps → 
  total_distance = 14197 / 16384 := by
  intro h
  sorry

end kevin_total_distance_l95_95142


namespace max_S_n_of_arithmetic_seq_l95_95695

theorem max_S_n_of_arithmetic_seq (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a 1 + n * d)
  (h2 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h3 : a 1 + a 3 + a 5 = 15)
  (h4 : a 2 + a 4 + a 6 = 0) : 
  ∃ n : ℕ, S n = 40 ∧ (∀ m : ℕ, S m ≤ 40) :=
sorry

end max_S_n_of_arithmetic_seq_l95_95695


namespace bernardo_larger_probability_l95_95435

-- Mathematical definitions
def bernardo_set : Finset ℕ := {1,2,3,4,5,6,7,8,10}
def silvia_set : Finset ℕ := {1,2,3,4,5,6}

-- Probability calculation function (you need to define the detailed implementation)
noncomputable def probability_bernardo_gt_silvia : ℚ := sorry

-- The proof statement
theorem bernardo_larger_probability : 
  probability_bernardo_gt_silvia = 13 / 20 :=
sorry

end bernardo_larger_probability_l95_95435


namespace natural_number_pairs_lcm_gcd_l95_95804

theorem natural_number_pairs_lcm_gcd (a b : ℕ) (h1 : lcm a b * gcd a b = a * b)
  (h2 : lcm a b - gcd a b = (a * b) / 5) : 
  (a = 4 ∧ b = 20) ∨ (a = 20 ∧ b = 4) :=
  sorry

end natural_number_pairs_lcm_gcd_l95_95804


namespace rancher_cattle_count_l95_95462

theorem rancher_cattle_count
  (truck_capacity : ℕ)
  (distance_to_higher_ground : ℕ)
  (truck_speed : ℕ)
  (total_transport_time : ℕ)
  (h1 : truck_capacity = 20)
  (h2 : distance_to_higher_ground = 60)
  (h3 : truck_speed = 60)
  (h4 : total_transport_time = 40):
  ∃ (number_of_cattle : ℕ), number_of_cattle = 400 :=
by {
  sorry
}

end rancher_cattle_count_l95_95462


namespace plane_through_points_and_perpendicular_l95_95981

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def plane_eq (A B C D : ℝ) (P : Point3D) : Prop :=
  A * P.x + B * P.y + C * P.z + D = 0

def vector_sub (P Q : Point3D) : Point3D :=
  ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

def cross_product (u v : Point3D) : Point3D :=
  ⟨u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x⟩

def is_perpendicular (normal1 normal2 : Point3D) : Prop :=
  normal1.x * normal2.x + normal1.y * normal2.y + normal1.z * normal2.z = 0

theorem plane_through_points_and_perpendicular
  (P1 P2 : Point3D)
  (A B C D : ℝ)
  (n_perp : Point3D)
  (normal1_eq : n_perp = ⟨2, -1, 4⟩)
  (eqn_given : plane_eq 2 (-1) 4 7 P1)
  (vec := vector_sub P1 P2)
  (n := cross_product vec n_perp)
  (eqn : plane_eq 11 (-10) (-9) (-33) P1) :
  (plane_eq 11 (-10) (-9) (-33) P2 ∧ is_perpendicular n n_perp) :=
sorry

end plane_through_points_and_perpendicular_l95_95981


namespace jane_current_age_l95_95934

theorem jane_current_age (J : ℕ) (h1 : ∀ t : ℕ, t = 13 → 25 + t = 2 * (J + t)) : J = 6 :=
by {
  sorry
}

end jane_current_age_l95_95934


namespace general_formula_an_general_formula_bn_exists_arithmetic_sequence_bn_l95_95002

variable (a_n : ℕ → ℝ)
variable (b_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)
variable (d : ℝ)

-- Define the initial conditions
axiom a2_a3_condition : a_n 2 * a_n 3 = 15
axiom S4_condition : S_n 4 = 16
axiom b_recursion : ∀ (n : ℕ), b_n (n + 1) - b_n n = 1 / (a_n n * a_n (n + 1))

-- Define the proofs
theorem general_formula_an : ∀ (n : ℕ), a_n n = 2 * n - 1 :=
sorry

theorem general_formula_bn : ∀ (n : ℕ), b_n n = (3 * n - 2) / (2 * n - 1) :=
sorry

theorem exists_arithmetic_sequence_bn : ∃ (m n : ℕ), m ≠ n ∧ b_n 2 + b_n n = 2 * b_n m ∧ b_n 2 = 4 / 3 ∧ (n = 8 ∧ m = 3) :=
sorry

end general_formula_an_general_formula_bn_exists_arithmetic_sequence_bn_l95_95002


namespace minimum_surface_area_of_combined_cuboids_l95_95728

noncomputable def cuboid_combinations (l w h : ℕ) (n : ℕ) : ℕ :=
sorry

theorem minimum_surface_area_of_combined_cuboids :
  ∃ n, cuboid_combinations 2 1 3 3 = 4 ∧ n = 42 :=
sorry

end minimum_surface_area_of_combined_cuboids_l95_95728


namespace prime_geq_7_div_240_l95_95717

theorem prime_geq_7_div_240 (p : ℕ) (hp : Nat.Prime p) (h7 : p ≥ 7) : 240 ∣ p^4 - 1 :=
sorry

end prime_geq_7_div_240_l95_95717


namespace chord_segments_division_l95_95277

-- Definitions based on the conditions
variables (R OM : ℝ) (AB : ℝ)
-- Setting the values as the problem provides 
def radius : ℝ := 15
def distance_from_center : ℝ := 13
def chord_length : ℝ := 18

-- Formulate the problem statement as a theorem
theorem chord_segments_division :
  ∃ (AM MB : ℝ), AM = 14 ∧ MB = 4 :=
by
  let CB := chord_length / 2
  let OC := Real.sqrt (radius^2 - CB^2)
  let MC := Real.sqrt (distance_from_center^2 - OC^2)
  let AM := CB + MC
  let MB := CB - MC
  use AM, MB
  sorry

end chord_segments_division_l95_95277


namespace polygon_sides_l95_95124

theorem polygon_sides (n : ℕ) 
    (h1 : (n-2) * 180 = 3 * 360 - 180) 
    (h2 : ∀ k, k > 2 → (k-2) * 180 = 180 * (k - 2)) 
    (h3 : 360 = 360) : n = 5 := 
by
  sorry

end polygon_sides_l95_95124


namespace fixed_point_of_function_l95_95205

-- Definition: The function passes through a fixed point (a, b) for all real numbers k.
def passes_through_fixed_point (f : ℝ → ℝ) (a b : ℝ) := ∀ k : ℝ, f a = b

-- Given the function y = 9x^2 + 3kx - 6k, we aim to prove the fixed point is (2, 36).
theorem fixed_point_of_function : passes_through_fixed_point (fun x => 9 * x^2 + 3 * k * x - 6 * k) 2 36 := by
  sorry

end fixed_point_of_function_l95_95205


namespace exponential_inequality_l95_95578

theorem exponential_inequality (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a^m < a^n) : ¬ (m < n) := 
sorry

end exponential_inequality_l95_95578


namespace find_p_l95_95770

theorem find_p 
  (p q x y : ℤ)
  (h1 : p * x + q * y = 8)
  (h2 : 3 * x - q * y = 38)
  (hx : x = 2)
  (hy : y = -4) : 
  p = 20 := 
by 
  subst hx
  subst hy
  sorry

end find_p_l95_95770


namespace scientific_notation_21600_l95_95911

theorem scientific_notation_21600 : ∃ (a : ℝ) (n : ℤ), 21600 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 2.16 ∧ n = 4 :=
by
  sorry

end scientific_notation_21600_l95_95911


namespace negation_of_proposition_l95_95965

theorem negation_of_proposition (p : ∀ (x : ℝ), x^2 + 1 > 0) :
  ∃ (x : ℝ), x^2 + 1 ≤ 0 ↔ ¬ (∀ (x : ℝ), x^2 + 1 > 0) :=
by
  sorry

end negation_of_proposition_l95_95965


namespace third_week_cases_l95_95337

-- Define the conditions as Lean definitions
def first_week_cases : ℕ := 5000
def second_week_cases : ℕ := first_week_cases / 2
def total_cases_after_three_weeks : ℕ := 9500

-- The statement to be proven
theorem third_week_cases :
  first_week_cases + second_week_cases + 2000 = total_cases_after_three_weeks :=
by
  sorry

end third_week_cases_l95_95337


namespace one_fourth_of_six_point_three_as_fraction_l95_95428

noncomputable def one_fourth_of_six_point_three_is_simplified : ℚ :=
  6.3 / 4

theorem one_fourth_of_six_point_three_as_fraction :
  one_fourth_of_six_point_three_is_simplified = 63 / 40 :=
by
  sorry

end one_fourth_of_six_point_three_as_fraction_l95_95428


namespace maximum_sphere_radius_squared_l95_95112

def cone_base_radius : ℝ := 4
def cone_height : ℝ := 10
def axes_intersection_distance_from_base : ℝ := 4

theorem maximum_sphere_radius_squared :
  let m : ℕ := 144
  let n : ℕ := 29
  m + n = 173 :=
by
  sorry

end maximum_sphere_radius_squared_l95_95112


namespace brett_total_miles_l95_95060

def miles_per_hour : ℕ := 75
def hours_driven : ℕ := 12

theorem brett_total_miles : miles_per_hour * hours_driven = 900 := 
by 
  sorry

end brett_total_miles_l95_95060


namespace probability_of_odd_sum_given_even_product_l95_95918

open Nat

noncomputable def probability_odd_sum_given_even_product : ℚ :=
  let total_outcomes := 6^5
  let odd_outcomes := 3^5
  let even_outcomes := total_outcomes - odd_outcomes
  let favorable_outcomes := 15 * 3^5
  favorable_outcomes / even_outcomes

theorem probability_of_odd_sum_given_even_product :
  probability_odd_sum_given_even_product = 91 / 324 :=
by
  sorry

end probability_of_odd_sum_given_even_product_l95_95918


namespace mean_daily_profit_l95_95947

theorem mean_daily_profit 
  (mean_first_15_days : ℝ) 
  (mean_last_15_days : ℝ) 
  (n : ℝ) 
  (m1_days : ℝ) 
  (m2_days : ℝ) : 
  (mean_first_15_days = 245) → 
  (mean_last_15_days = 455) → 
  (m1_days = 15) → 
  (m2_days = 15) → 
  (n = 30) →
  (∀ P, P = (245 * 15 + 455 * 15) / 30) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end mean_daily_profit_l95_95947


namespace find_a_b_sum_l95_95940

theorem find_a_b_sum (a b : ℕ) (h1 : 830 - (400 + 10 * a + 7) = 300 + 10 * b + 4)
    (h2 : ∃ k : ℕ, 300 + 10 * b + 4 = 7 * k) : a + b = 2 :=
by
  sorry

end find_a_b_sum_l95_95940


namespace intersection_is_correct_l95_95372

def setA := {x : ℝ | 3 * x - x^2 > 0}
def setB := {x : ℝ | x ≤ 1}

theorem intersection_is_correct : 
  setA ∩ setB = {x | 0 < x ∧ x ≤ 1} :=
sorry

end intersection_is_correct_l95_95372
