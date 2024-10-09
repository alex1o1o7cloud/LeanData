import Mathlib

namespace evaluate_g_at_6_l1740_174038

def g (x : ℝ) : ℝ := 3 * x^4 - 20 * x^3 + 30 * x^2 - 35 * x - 75

theorem evaluate_g_at_6 : g 6 = 363 :=
by
  -- Proof skipped
  sorry

end evaluate_g_at_6_l1740_174038


namespace count_polynomials_l1740_174083

def is_polynomial (expr : String) : Bool :=
  match expr with
  | "-7"            => true
  | "x"             => true
  | "m^2 + 1/m"     => false
  | "x^2*y + 5"     => true
  | "(x + y)/2"     => true
  | "-5ab^3c^2"     => true
  | "1/y"           => false
  | _               => false

theorem count_polynomials :
  let expressions := ["-7", "x", "m^2 + 1/m", "x^2*y + 5", "(x + y)/2", "-5ab^3c^2", "1/y"]
  List.filter is_polynomial expressions |>.length = 5 :=
by
  sorry

end count_polynomials_l1740_174083


namespace equality_of_floor_squares_l1740_174057

theorem equality_of_floor_squares (n : ℕ) (hn : 0 < n) :
  (⌊Real.sqrt n + Real.sqrt (n + 1)⌋ : ℤ) = ⌊Real.sqrt (4 * n + 1)⌋ ∧
  (⌊Real.sqrt (4 * n + 1)⌋ : ℤ) = ⌊Real.sqrt (4 * n + 2)⌋ ∧
  (⌊Real.sqrt (4 * n + 2)⌋ : ℤ) = ⌊Real.sqrt (4 * n + 3)⌋ :=
by
  sorry

end equality_of_floor_squares_l1740_174057


namespace find_divisor_l1740_174056

theorem find_divisor (d : ℕ) (H1 : 199 = d * 11 + 1) : d = 18 := 
sorry

end find_divisor_l1740_174056


namespace directrix_of_parabola_l1740_174044

theorem directrix_of_parabola (a b c : ℝ) (h_eqn : ∀ x, b = -4 * x^2 + c) : 
  b = 5 → c = 0 → (∃ y, y = 81 / 16) :=
by
  sorry

end directrix_of_parabola_l1740_174044


namespace g_g_g_g_15_eq_3_l1740_174047

def g (x : ℕ) : ℕ :=
if x % 3 = 0 then x / 3 else 5 * x + 2

theorem g_g_g_g_15_eq_3 : g (g (g (g 15))) = 3 := 
by
  sorry

end g_g_g_g_15_eq_3_l1740_174047


namespace minimum_value_correct_l1740_174070

noncomputable def minimum_value (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_eq : x^2 + y^2 + z^2 = 1) : ℝ :=
  (z + 1)^2 / (2 * x * y * z)

theorem minimum_value_correct {x y z : ℝ}
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_eq : x^2 + y^2 + z^2 = 1) :
  minimum_value x y z h_pos h_eq = 3 + 2 * Real.sqrt 2 :=
sorry

end minimum_value_correct_l1740_174070


namespace div_by_7_l1740_174059

theorem div_by_7 (n : ℕ) (h : n ≥ 1) : 7 ∣ (8^n + 6) :=
sorry

end div_by_7_l1740_174059


namespace min_troublemakers_in_class_l1740_174081

noncomputable def min_troublemakers : ℕ :=
  10

theorem min_troublemakers_in_class :
  (∃ t l : ℕ, t + l = 29 ∧ t + l - 1 = 29 ∧
   (∀ i : ℕ, i < 29 → (i % 3 = 0 → ∃ t : ℕ, t = 1) ∧ 
   (i % 3 ≠ 0 → ∃ t : ℕ, t = 2))) →
   min_troublemakers = 10 :=
by
  sorry

end min_troublemakers_in_class_l1740_174081


namespace percentage_reduction_l1740_174076

theorem percentage_reduction (P : ℝ) (h1 : 700 / P + 3 = 700 / 70) : 
  ((P - 70) / P) * 100 = 30 :=
by
  sorry

end percentage_reduction_l1740_174076


namespace shari_effective_distance_l1740_174082

-- Define the given conditions
def constant_rate : ℝ := 4 -- miles per hour
def wind_resistance : ℝ := 0.5 -- miles per hour
def walking_time : ℝ := 2 -- hours

-- Define the effective walking speed considering wind resistance
def effective_speed : ℝ := constant_rate - wind_resistance

-- Define the effective walking distance
def effective_distance : ℝ := effective_speed * walking_time

-- State that Shari effectively walks 7.0 miles
theorem shari_effective_distance :
  effective_distance = 7.0 :=
by
  sorry

end shari_effective_distance_l1740_174082


namespace hydras_never_die_l1740_174085

theorem hydras_never_die (heads_A heads_B : ℕ) (grow_heads : ℕ → ℕ → Prop) : 
  (heads_A = 2016) → 
  (heads_B = 2017) →
  (∀ a b : ℕ, grow_heads a b → (a = 5 ∨ a = 7) ∧ (b = 5 ∨ b = 7)) →
  (∀ (a b : ℕ), grow_heads a b → (heads_A + a - 2) ≠ (heads_B + b - 2)) :=
by
  intros hA hB hGrow
  intro hEq
  sorry

end hydras_never_die_l1740_174085


namespace initial_white_cookies_l1740_174063

theorem initial_white_cookies (B W : ℕ) 
  (h1 : B = W + 50)
  (h2 : (1 / 2 : ℚ) * B + (1 / 4 : ℚ) * W = 85) :
  W = 80 :=
by
  sorry

end initial_white_cookies_l1740_174063


namespace distance_from_y_axis_l1740_174062

theorem distance_from_y_axis (dx dy : ℝ) (h1 : dx = 8) (h2 : dx = (1/2) * dy) : dy = 16 :=
by
  sorry

end distance_from_y_axis_l1740_174062


namespace product_simplification_l1740_174041

theorem product_simplification :
  (10 * (1 / 5) * (1 / 2) * 4 / 2 : ℝ) = 2 :=
by
  sorry

end product_simplification_l1740_174041


namespace max_intersection_l1740_174049

open Finset

def n (S : Finset α) : ℕ := (2 : ℕ) ^ S.card

theorem max_intersection (A B C : Finset ℕ)
  (h1 : A.card = 2016)
  (h2 : B.card = 2016)
  (h3 : n A + n B + n C = n (A ∪ B ∪ C)) :
  (A ∩ B ∩ C).card ≤ 2015 :=
sorry

end max_intersection_l1740_174049


namespace find_7th_term_l1740_174072

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem find_7th_term 
    (a d : ℤ) 
    (h3 : a + 2 * d = 17) 
    (h5 : a + 4 * d = 39) : 
    arithmetic_sequence a d 7 = 61 := 
sorry

end find_7th_term_l1740_174072


namespace det_A_is_2_l1740_174090

-- Define the matrix A
def A (a d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, 2], ![-3, d]]

-- Define the inverse of matrix A 
noncomputable def A_inv (a d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  (1 / (a * d + 6)) • ![![d, -2], ![3, a]]

-- Condition: A + A_inv = 0
def condition (a d : ℝ) : Prop := A a d + A_inv a d = 0

-- Main theorem: determinant of A under the given condition
theorem det_A_is_2 (a d : ℝ) (h : condition a d) : Matrix.det (A a d) = 2 :=
by sorry

end det_A_is_2_l1740_174090


namespace maxwell_distance_when_meeting_l1740_174032

theorem maxwell_distance_when_meeting 
  (distance_between_homes : ℕ)
  (maxwell_speed : ℕ) 
  (brad_speed : ℕ) 
  (total_distance : ℕ) 
  (h : distance_between_homes = 36) 
  (h1 : maxwell_speed = 2)
  (h2 : brad_speed = 4) 
  (h3 : 6 * (total_distance / 6) = distance_between_homes) :
  total_distance = 12 :=
sorry

end maxwell_distance_when_meeting_l1740_174032


namespace sequence_expression_l1740_174036

theorem sequence_expression (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) - 2 * a n = 2^n) :
  ∀ n, a n = n * 2^(n - 1) :=
by
  sorry

end sequence_expression_l1740_174036


namespace three_digit_number_satisfies_conditions_l1740_174091

-- Definitions for the digits of the number
def x := 9
def y := 6
def z := 4

-- Define the three-digit number
def number := 100 * x + 10 * y + z

-- Define the conditions
def geometric_progression := y * y = x * z

def reverse_order_condition := (number - 495) = 100 * z + 10 * y + x

def arithmetic_progression := (z - 1) + (x - 2) = 2 * (y - 1)

-- The theorem to prove
theorem three_digit_number_satisfies_conditions :
  geometric_progression ∧ reverse_order_condition ∧ arithmetic_progression :=
by {
  sorry
}

end three_digit_number_satisfies_conditions_l1740_174091


namespace min_shoeing_time_l1740_174078

theorem min_shoeing_time
  (num_blacksmiths : ℕ) (num_horses : ℕ) (hooves_per_horse : ℕ) (minutes_per_hoof : ℕ)
  (h_blacksmiths : num_blacksmiths = 48)
  (h_horses : num_horses = 60)
  (h_hooves_per_horse : hooves_per_horse = 4)
  (h_minutes_per_hoof : minutes_per_hoof = 5) :
  (num_horses * hooves_per_horse * minutes_per_hoof) / num_blacksmiths = 25 := 
by
  sorry

end min_shoeing_time_l1740_174078


namespace rain_difference_l1740_174035

variable (R : ℝ) -- Amount of rain in the second hour
variable (r1 : ℝ) -- Amount of rain in the first hour

-- Conditions
axiom h1 : r1 = 5
axiom h2 : R + r1 = 22

-- Theorem to prove
theorem rain_difference (R r1 : ℝ) (h1 : r1 = 5) (h2 : R + r1 = 22) : R - 2 * r1 = 7 := by
  sorry

end rain_difference_l1740_174035


namespace fraction_even_odd_phonenumbers_l1740_174040

-- Define a predicate for valid phone numbers
def isValidPhoneNumber (n : Nat) : Prop :=
  1000000 ≤ n ∧ n < 10000000 ∧ (n / 1000000 ≠ 0) ∧ (n / 1000000 ≠ 1)

-- Calculate the total number of valid phone numbers
def totalValidPhoneNumbers : Nat :=
  4 * 10^6

-- Calculate the number of valid phone numbers that begin with an even digit and end with an odd digit
def validEvenOddPhoneNumbers : Nat :=
  4 * (10^5) * 5

-- Determine the fraction of such phone numbers (valid ones and valid even-odd ones)
theorem fraction_even_odd_phonenumbers : 
  (validEvenOddPhoneNumbers) / (totalValidPhoneNumbers) = 1 / 2 :=
by {
  sorry
}

end fraction_even_odd_phonenumbers_l1740_174040


namespace part1_part2_part3_part4_l1740_174061

-- Part 1: Prove that 1/42 is equal to 1/6 - 1/7
theorem part1 : (1/42 : ℚ) = (1/6 : ℚ) - (1/7 : ℚ) := sorry

-- Part 2: Prove that 1/240 is equal to 1/15 - 1/16
theorem part2 : (1/240 : ℚ) = (1/15 : ℚ) - (1/16 : ℚ) := sorry

-- Part 3: Prove the general rule for all natural numbers m
theorem part3 (m : ℕ) (hm : m > 0) : (1 / (m * (m + 1)) : ℚ) = (1 / m : ℚ) - (1 / (m + 1) : ℚ) := sorry

-- Part 4: Prove the given expression evaluates to 0 for any x
theorem part4 (x : ℚ) (h1 : x ≠ 1) (h2 : x ≠ 2) (h3 : x ≠ 3) : 
  (1 / ((x - 2) * (x - 3)) : ℚ) - (2 / ((x - 1) * (x - 3)) : ℚ) + (1 / ((x - 1) * (x - 2)) : ℚ) = 0 := sorry

end part1_part2_part3_part4_l1740_174061


namespace total_birds_l1740_174073

theorem total_birds (g d : Nat) (h₁ : g = 58) (h₂ : d = 37) : g + d = 95 :=
by
  sorry

end total_birds_l1740_174073


namespace perfect_square_condition_l1740_174005

theorem perfect_square_condition (x y : ℕ) :
  ∃ k : ℕ, (x + y)^2 + 3*x + y + 1 = k^2 ↔ x = y := 
by 
  sorry

end perfect_square_condition_l1740_174005


namespace total_distance_traveled_eq_l1740_174012

-- Define the conditions as speeds and times for each segment of Jeff's trip.
def speed1 : ℝ := 80
def time1 : ℝ := 6
def speed2 : ℝ := 60
def time2 : ℝ := 4
def speed3 : ℝ := 40
def time3 : ℝ := 2

-- Define the distance function given speed and time.
def distance (speed time : ℝ) : ℝ := speed * time

-- Calculate the individual distances for each segment.
def distance1 : ℝ := distance speed1 time1
def distance2 : ℝ := distance speed2 time2
def distance3 : ℝ := distance speed3 time3

-- State the proof problem to show that the total distance is 800 miles.
theorem total_distance_traveled_eq : distance1 + distance2 + distance3 = 800 :=
by
  -- Placeholder for actual proof
  sorry

end total_distance_traveled_eq_l1740_174012


namespace base6_to_base10_l1740_174064

theorem base6_to_base10 (c d : ℕ) (h1 : 524 = 2 * (10 * c + d)) (hc : c < 10) (hd : d < 10) :
  (c * d : ℚ) / 12 = 3 / 4 := by
  sorry

end base6_to_base10_l1740_174064


namespace school_enrollment_l1740_174024

theorem school_enrollment
  (X Y : ℝ)
  (h1 : X + Y = 4000)
  (h2 : 1.07 * X > X)
  (h3 : 1.03 * Y > Y)
  (h4 : 0.07 * X - 0.03 * Y = 40) :
  Y = 2400 :=
by
  -- problem reduction
  sorry

end school_enrollment_l1740_174024


namespace b_20_value_l1740_174009

-- Definitions based on conditions
def a (n : ℕ) : ℕ := 2 * n - 1

def b (n : ℕ) : ℕ := a n  -- Given that \( b_n = a_n \)

-- The theorem stating that \( b_{20} = 39 \)
theorem b_20_value : b 20 = 39 :=
by
  -- Skipping the proof
  sorry

end b_20_value_l1740_174009


namespace max_cut_strings_preserving_net_l1740_174030

-- Define the conditions of the problem
def volleyball_net_width : ℕ := 50
def volleyball_net_height : ℕ := 600

-- The vertices count is calculated as (width + 1) * (height + 1)
def vertices_count : ℕ := (volleyball_net_width + 1) * (volleyball_net_height + 1)

-- The total edges count is the sum of vertical and horizontal edges
def total_edges_count : ℕ := volleyball_net_width * (volleyball_net_height + 1) + (volleyball_net_width + 1) * volleyball_net_height

-- The edges needed to keep the graph connected (number of vertices - 1)
def edges_in_tree : ℕ := vertices_count - 1

-- The maximum removable edges (total edges - edges needed in tree)
def max_removable_edges : ℕ := total_edges_count - edges_in_tree

-- Define the theorem to prove
theorem max_cut_strings_preserving_net : max_removable_edges = 30000 := by
  sorry

end max_cut_strings_preserving_net_l1740_174030


namespace fraction_of_product_l1740_174011

theorem fraction_of_product (c d: ℕ) 
  (h1: 5 * 64 + 4 * 8 + 3 = 355)
  (h2: 2 * (10 * c + d) = 355)
  (h3: c < 10)
  (h4: d < 10):
  (c * d : ℚ) / 12 = 5 / 4 :=
by
  sorry

end fraction_of_product_l1740_174011


namespace polynomial_evaluation_l1740_174029

theorem polynomial_evaluation :
  (5 * 3^3 - 3 * 3^2 + 7 * 3 - 2 = 127) :=
by
  sorry

end polynomial_evaluation_l1740_174029


namespace relationship_between_abc_l1740_174006

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

theorem relationship_between_abc (h1 : 2^a = Real.log (1/a) / Real.log 2)
                                 (h2 : Real.log b / Real.log 2 = 2)
                                 (h3 : c = Real.log 2 + Real.log 3 - Real.log 7) :
  b > a ∧ a > c :=
sorry

end relationship_between_abc_l1740_174006


namespace max_value_sqrt43_l1740_174095

noncomputable def max_value_expr (x y z : ℝ) : ℝ :=
  3 * x * z * Real.sqrt 2 + 5 * x * y

theorem max_value_sqrt43 (x y z : ℝ) (h₁ : 0 ≤ x) (h₂ : 0 ≤ y) (h₃ : 0 ≤ z) (h₄ : x^2 + y^2 + z^2 = 1) :
  max_value_expr x y z ≤ Real.sqrt 43 :=
sorry

end max_value_sqrt43_l1740_174095


namespace consecutive_odd_integers_l1740_174050

theorem consecutive_odd_integers (n : ℕ) (h1 : n > 0) (h2 : (1 : ℚ) / n * ((n : ℚ) * 154) = 154) : n = 10 :=
sorry

end consecutive_odd_integers_l1740_174050


namespace jim_saving_amount_l1740_174019

theorem jim_saving_amount
    (sara_initial_savings : ℕ)
    (sara_weekly_savings : ℕ)
    (jim_weekly_savings : ℕ)
    (weeks_elapsed : ℕ)
    (sara_total_savings : ℕ := sara_initial_savings + weeks_elapsed * sara_weekly_savings)
    (jim_total_savings : ℕ := weeks_elapsed * jim_weekly_savings)
    (savings_equal: sara_total_savings = jim_total_savings)
    (sara_initial_savings_value : sara_initial_savings = 4100)
    (sara_weekly_savings_value : sara_weekly_savings = 10)
    (weeks_elapsed_value : weeks_elapsed = 820) :
    jim_weekly_savings = 15 := 
by
  sorry

end jim_saving_amount_l1740_174019


namespace rooster_weight_l1740_174008

variable (W : ℝ)  -- The weight of the first rooster

theorem rooster_weight (h1 : 0.50 * W + 0.50 * 40 = 35) : W = 30 :=
by
  sorry

end rooster_weight_l1740_174008


namespace different_routes_calculation_l1740_174093

-- Definitions for the conditions
def west_blocks := 3
def south_blocks := 2
def east_blocks := 3
def north_blocks := 3

-- Calculation of combinations for the number of sequences
def house_to_sw_corner_routes := Nat.choose (west_blocks + south_blocks) south_blocks
def ne_corner_to_school_routes := Nat.choose (east_blocks + north_blocks) east_blocks

-- Proving the total number of routes
theorem different_routes_calculation : 
  house_to_sw_corner_routes * 1 * ne_corner_to_school_routes = 200 :=
by
  -- Mathematical proof steps (to be filled)
  sorry

end different_routes_calculation_l1740_174093


namespace middle_card_is_five_l1740_174033

theorem middle_card_is_five 
    (a b c : ℕ) 
    (h1 : a ≠ b ∧ a ≠ c ∧ b ≠ c) 
    (h2 : a + b + c = 16)
    (h3 : a < b ∧ b < c)
    (casey : ¬(∃ y z, y ≠ z ∧ y + z + a = 16 ∧ a < y ∧ y < z))
    (tracy : ¬(∃ x y, x ≠ y ∧ x + y + c = 16 ∧ x < y ∧ y < c))
    (stacy : ¬(∃ x z, x ≠ z ∧ x + z + b = 16 ∧ x < b ∧ b < z)) 
    : b = 5 :=
sorry

end middle_card_is_five_l1740_174033


namespace range_of_function_l1740_174068

theorem range_of_function :
  ∀ x, -1 ≤ Real.sin x ∧ Real.sin x ≤ 1 → -3 ≤ 2 * Real.sin x - 1 ∧ 2 * Real.sin x - 1 ≤ 1 :=
by
  intros x h
  sorry

end range_of_function_l1740_174068


namespace power_of_exponents_l1740_174058

theorem power_of_exponents (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 := by
  sorry

end power_of_exponents_l1740_174058


namespace triangle_inequality_l1740_174071

theorem triangle_inequality 
  (a b c : ℝ)
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) : 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := 
sorry

end triangle_inequality_l1740_174071


namespace english_speaking_students_l1740_174039

theorem english_speaking_students (T H B E : ℕ) (hT : T = 40) (hH : H = 30) (hB : B = 10) (h_inclusion_exclusion : T = H + E - B) : E = 20 :=
by
  sorry

end english_speaking_students_l1740_174039


namespace tip_percentage_is_20_l1740_174092

theorem tip_percentage_is_20 (total_spent price_before_tax_and_tip : ℝ) (sales_tax_rate : ℝ) (h1 : total_spent = 158.40) (h2 : price_before_tax_and_tip = 120) (h3 : sales_tax_rate = 0.10) :
  ((total_spent - (price_before_tax_and_tip * (1 + sales_tax_rate))) / (price_before_tax_and_tip * (1 + sales_tax_rate))) * 100 = 20 :=
by
  sorry

end tip_percentage_is_20_l1740_174092


namespace Carissa_ran_at_10_feet_per_second_l1740_174094

theorem Carissa_ran_at_10_feet_per_second :
  ∀ (n : ℕ), 
  (∃ (a : ℕ), 
    (2 * a + 2 * n^2 * a = 260) ∧ -- Total distance
    (a + n * a = 30)) → -- Total time spent
  (2 * n = 10) :=
by
  intro n
  intro h
  sorry

end Carissa_ran_at_10_feet_per_second_l1740_174094


namespace find_a_l1740_174016

def f (x : ℝ) : ℝ := -x^2 - 2 * x + 3

theorem find_a : ∃ a : ℝ, (a > -1) ∧ (a < 2) ∧ (∀ x : ℝ, a ≤ x ∧ x ≤ 2 → f x ≤ f a) ∧ f a = 15 / 4 :=
by
  exists -1 / 2
  sorry

end find_a_l1740_174016


namespace log_simplification_l1740_174086

theorem log_simplification :
  (1 / (Real.log 3 / Real.log 12 + 2))
  + (1 / (Real.log 2 / Real.log 8 + 2))
  + (1 / (Real.log 3 / Real.log 9 + 2)) = 2 :=
  sorry

end log_simplification_l1740_174086


namespace remainder_when_divided_by_11_l1740_174048

theorem remainder_when_divided_by_11 (N : ℕ)
  (h₁ : N = 5 * 5 + 0) :
  N % 11 = 3 := 
sorry

end remainder_when_divided_by_11_l1740_174048


namespace no_alpha_exists_l1740_174046

theorem no_alpha_exists (α : ℝ) (hα1 : 0 < α) (hα2 : α < 1) :
  ¬(∃ (a : ℕ → ℝ), (∀ n : ℕ, 0 < a n) ∧ (∀ n : ℕ, 1 + a (n+1) ≤ a n + (α / n.succ) * a n)) :=
by
  sorry

end no_alpha_exists_l1740_174046


namespace universal_negation_example_l1740_174001

theorem universal_negation_example :
  (∀ x : ℝ, x^2 - 3 * x + 1 ≤ 0) →
  (¬ (∀ x : ℝ, x^2 - 3 * x + 1 ≤ 0) = (∃ x : ℝ, x^2 - 3 * x + 1 > 0)) :=
by
  intro h
  sorry

end universal_negation_example_l1740_174001


namespace moles_of_water_produced_l1740_174075

-- Definitions for the chemical reaction
def moles_NaOH := 4
def moles_H₂SO₄ := 2

-- The balanced chemical equation tells us the ratio of NaOH to H₂O
def chemical_equation (moles_NaOH moles_H₂SO₄ moles_H₂O moles_Na₂SO₄: ℕ) : Prop :=
  2 * moles_NaOH = 2 * moles_H₂O ∧ moles_H₂SO₄ = 1 ∧ moles_Na₂SO₄ = 1

-- The actual proof statement
theorem moles_of_water_produced : 
  ∀ (m_NaOH m_H₂SO₄ m_Na₂SO₄ : ℕ), 
  chemical_equation m_NaOH m_H₂SO₄ 4 m_Na₂SO₄ → moles_H₂O = 4 :=
by
  intros m_NaOH m_H₂SO₄ m_Na₂SO₄ chem_eq
  -- Placeholder for the actual proof.
  sorry

end moles_of_water_produced_l1740_174075


namespace rem_product_eq_l1740_174043

theorem rem_product_eq 
  (P Q R k : ℤ) 
  (hk : k > 0) 
  (hPQ : P * Q = R) : 
  ((P % k) * (Q % k)) % k = R % k :=
by
  sorry

end rem_product_eq_l1740_174043


namespace first_person_work_days_l1740_174080

theorem first_person_work_days (x : ℝ) (h1 : 0 < x) :
  (1/x + 1/40 = 1/15) → x = 24 :=
by
  intro h
  sorry

end first_person_work_days_l1740_174080


namespace rows_seating_nine_people_l1740_174055

theorem rows_seating_nine_people (x y : ℕ) (h : 9 * x + 7 * y = 74) : x = 2 :=
by sorry

end rows_seating_nine_people_l1740_174055


namespace emilia_blueberries_l1740_174054

def cartons_needed : Nat := 42
def cartons_strawberries : Nat := 2
def cartons_bought : Nat := 33

def cartons_blueberries (needed : Nat) (strawberries : Nat) (bought : Nat) : Nat :=
  needed - (strawberries + bought)

theorem emilia_blueberries : cartons_blueberries cartons_needed cartons_strawberries cartons_bought = 7 :=
by
  sorry

end emilia_blueberries_l1740_174054


namespace diff_eq_40_l1740_174066

theorem diff_eq_40 (x y : ℤ) (h1 : x + y = 24) (h2 : x = 32) : x - y = 40 := by
  sorry

end diff_eq_40_l1740_174066


namespace number_of_valid_house_numbers_l1740_174098

def is_two_digit_prime (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ Prime n

def digit_sum_odd (n : ℕ) : Prop :=
  (n / 10 + n % 10) % 2 = 1

def valid_house_number (W X Y Z : ℕ) : Prop :=
  W ≠ 0 ∧ X ≠ 0 ∧ Y ≠ 0 ∧ Z ≠ 0 ∧
  is_two_digit_prime (10 * W + X) ∧ is_two_digit_prime (10 * Y + Z) ∧
  10 * W + X ≠ 10 * Y + Z ∧
  10 * W + X < 60 ∧ 10 * Y + Z < 60 ∧
  digit_sum_odd (10 * W + X)

theorem number_of_valid_house_numbers : ∃ n, n = 108 ∧
  (∀ W X Y Z, valid_house_number W X Y Z → valid_house_number_count = 108) :=
sorry

end number_of_valid_house_numbers_l1740_174098


namespace square_root_of_9_l1740_174052

theorem square_root_of_9 : {x : ℝ // x^2 = 9} = {x : ℝ // x = 3 ∨ x = -3} :=
by
  sorry

end square_root_of_9_l1740_174052


namespace cost_of_brushes_and_canvas_minimum_canvases_cost_effectiveness_l1740_174079

-- Part 1: Prove the cost of one box of brushes and one canvas each.
theorem cost_of_brushes_and_canvas (x y : ℕ) 
    (h₁ : 2 * x + 4 * y = 94) (h₂ : 4 * x + 2 * y = 98) :
    x = 17 ∧ y = 15 := by
  sorry

-- Part 2: Prove the minimum number of canvases.
theorem minimum_canvases (m : ℕ) 
    (h₃ : m + (10 - m) = 10) (h₄ : 17 * (10 - m) + 15 * m ≤ 157) :
    m ≥ 7 := by
  sorry

-- Part 3: Prove the cost-effective purchasing plan.
theorem cost_effectiveness (m n : ℕ) 
    (h₃ : m + n = 10) (h₄ : 17 * n + 15 * m ≤ 157) (h₅ : m ≤ 8) :
    (m = 8 ∧ n = 2) := by
  sorry

end cost_of_brushes_and_canvas_minimum_canvases_cost_effectiveness_l1740_174079


namespace ellipse_foci_coordinates_l1740_174031

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ), (x^2 / 64 + y^2 / 100 = 1) → (x = 0 ∧ (y = 6 ∨ y = -6)) :=
by
  sorry

end ellipse_foci_coordinates_l1740_174031


namespace arithmetic_sequence_sum_l1740_174023

variable (S : ℕ → ℝ)
variable (a_n : ℕ → ℝ)

theorem arithmetic_sequence_sum (h₁ : S 5 = 8) (h₂ : S 10 = 20) : S 15 = 36 := 
by
  sorry

end arithmetic_sequence_sum_l1740_174023


namespace remainder_when_2x_divided_by_7_l1740_174045

theorem remainder_when_2x_divided_by_7 (x y r : ℤ) (h1 : x = 10 * y + 3)
    (h2 : 2 * x = 7 * (3 * y) + r) (h3 : 11 * y - x = 2) : r = 1 := by
  sorry

end remainder_when_2x_divided_by_7_l1740_174045


namespace find_days_l1740_174018

variables (a d e k m : ℕ) (y : ℕ)

-- Assumptions based on the problem
def workers_efficiency_condition : Prop := 
  (a * e * (d * k) / (a * e)) = d

-- Conclusion we aim to prove
def target_days_condition : Prop :=
  y = (a * a) / (d * k * m)

theorem find_days (h : workers_efficiency_condition a d e k) : target_days_condition a d k m y :=
  sorry

end find_days_l1740_174018


namespace problem_statement_l1740_174065

noncomputable def S (k : ℕ) : ℚ := sorry

theorem problem_statement (k : ℕ) (a_k : ℚ) :
  S (k - 1) < 10 → S k > 10 → a_k = 6 / 7 :=
sorry

end problem_statement_l1740_174065


namespace find_two_angles_of_scalene_obtuse_triangle_l1740_174000

def is_scalene (a b c : ℝ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c
def is_obtuse (a : ℝ) : Prop := a > 90
def is_triangle (a b c : ℝ) : Prop := a + b + c = 180

theorem find_two_angles_of_scalene_obtuse_triangle
  (a b c : ℝ)
  (ha : is_obtuse a) (h_scalene : is_scalene a b c) 
  (h_sum : is_triangle a b c) 
  (ha_val : a = 108)
  (h_half : b = 2 * c) :
  b = 48 ∧ c = 24 :=
by
  sorry

end find_two_angles_of_scalene_obtuse_triangle_l1740_174000


namespace solution_for_b_l1740_174042

theorem solution_for_b (x y b : ℚ) (h1 : 4 * x + 3 * y = b) (h2 : 3 * x + 4 * y = 3 * b) (hx : x = 3) : b = -21 / 5 := by
  sorry

end solution_for_b_l1740_174042


namespace find_smallest_n_l1740_174025

-- defining the geometric sequence and its sum for the given conditions
def a_n (n : ℕ) := 3 * (4 ^ n)

def S_n (n : ℕ) := (a_n n - 1) / (4 - 1) -- simplification step

-- statement of the problem: finding the smallest natural number n such that S_n > 3000
theorem find_smallest_n :
  ∃ n : ℕ, S_n n > 3000 ∧ ∀ m : ℕ, m < n → S_n m ≤ 3000 := by
  sorry

end find_smallest_n_l1740_174025


namespace sheep_to_horses_ratio_l1740_174087

-- Define the known quantities
def number_of_sheep := 32
def total_horse_food := 12880
def food_per_horse := 230

-- Calculate number of horses
def number_of_horses := total_horse_food / food_per_horse

-- Calculate and simplify the ratio of sheep to horses
def ratio_of_sheep_to_horses := (number_of_sheep : ℚ) / (number_of_horses : ℚ)

-- Define the expected simplified ratio
def expected_ratio_of_sheep_to_horses := (4 : ℚ) / (7 : ℚ)

-- The statement we want to prove
theorem sheep_to_horses_ratio : ratio_of_sheep_to_horses = expected_ratio_of_sheep_to_horses :=
by
  -- Proof will be here
  sorry

end sheep_to_horses_ratio_l1740_174087


namespace fixed_point_linear_l1740_174022

-- Define the linear function y = kx + k + 2
def linear_function (k x : ℝ) : ℝ := k * x + k + 2

-- Prove that the point (-1, 2) lies on the graph of the function for any k
theorem fixed_point_linear (k : ℝ) : linear_function k (-1) = 2 := by
  sorry

end fixed_point_linear_l1740_174022


namespace second_less_than_first_l1740_174089

-- Define the given conditions
def third_number : ℝ := sorry
def first_number : ℝ := 0.65 * third_number
def second_number : ℝ := 0.58 * third_number

-- Problem statement: Prove that the second number is approximately 10.77% less than the first number
theorem second_less_than_first : 
  (first_number - second_number) / first_number * 100 = 10.77 := 
sorry

end second_less_than_first_l1740_174089


namespace three_digit_integers_count_l1740_174097

theorem three_digit_integers_count (N : ℕ) :
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
            n % 7 = 4 ∧ 
            n % 8 = 3 ∧ 
            n % 10 = 2) → N = 3 :=
by
  sorry

end three_digit_integers_count_l1740_174097


namespace train_length_l1740_174027

noncomputable def train_speed_kmph : ℝ := 72
noncomputable def train_speed_mps : ℝ := 20
noncomputable def crossing_time : ℝ := 20
noncomputable def platform_length : ℝ := 220.032
noncomputable def total_distance : ℝ := train_speed_mps * crossing_time

theorem train_length :
  total_distance - platform_length = 179.968 := by
  sorry

end train_length_l1740_174027


namespace quadratic_equation_general_form_l1740_174037

theorem quadratic_equation_general_form :
  ∀ (x : ℝ), 3 * x^2 + 1 = 7 * x ↔ 3 * x^2 - 7 * x + 1 = 0 :=
by
  intro x
  constructor
  · intro h
    sorry
  · intro h
    sorry

end quadratic_equation_general_form_l1740_174037


namespace standard_equation_of_hyperbola_l1740_174099

noncomputable def ellipse_eccentricity_problem
  (e : ℚ) (a_maj : ℕ) (f_1 f_2 : ℝ × ℝ) (d : ℕ) : Prop :=
  e = 5 / 13 ∧
  a_maj = 26 ∧
  f_1 = (-5, 0) ∧
  f_2 = (5, 0) ∧
  d = 8 →
  ∃ b, (2 * b = 3) ∧ (2 * b ≠ 0) ∧
  ∃ h k : ℝ, (0 ≤  h) ∧ (0 ≤ k) ∧
  ((h^2)/(4^2)) - ((k^2)/(3^2)) = 1

-- problem statement: 
theorem standard_equation_of_hyperbola
  (e : ℚ) (a_maj : ℕ) (f_1 f_2 : ℝ × ℝ) (d : ℕ)
  (h : e = 5 / 13)
  (a_maj_length : a_maj = 26)
  (f1_coords : f_1 = (-5, 0))
  (f2_coords : f_2 = (5, 0))
  (distance_diff : d = 8) :
  ellipse_eccentricity_problem e a_maj f_1 f_2 d :=
sorry

end standard_equation_of_hyperbola_l1740_174099


namespace fourth_ball_black_probability_l1740_174067

noncomputable def prob_fourth_is_black : Prop :=
  let total_balls := 8
  let black_balls := 4
  let prob_black := black_balls / total_balls
  prob_black = 1 / 2

theorem fourth_ball_black_probability :
  prob_fourth_is_black :=
sorry

end fourth_ball_black_probability_l1740_174067


namespace triangle_angle_l1740_174034

-- Definitions of the conditions and theorem
variables {a b c : ℝ}
variables {A B C : ℝ}

theorem triangle_angle (h : b^2 + c^2 - a^2 = bc)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hA : 0 < A) (hA_max : A < π) :
  A = π / 3 :=
by
  sorry

end triangle_angle_l1740_174034


namespace polygon_side_count_l1740_174053

theorem polygon_side_count (n : ℕ) 
    (h : (n - 2) * 180 + 1350 - (n - 2) * 180 = 1350) : n = 9 :=
by
  sorry

end polygon_side_count_l1740_174053


namespace find_k_l1740_174007

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V) (k : ℝ)

-- Conditions
def not_collinear (a b : V) : Prop := ¬ ∃ (m : ℝ), b = m • a
def collinear (u v : V) : Prop := ∃ (m : ℝ), u = m • v

theorem find_k (h1 : not_collinear a b) (h2 : collinear (2 • a + k • b) (a - b)) : k = -2 :=
by
  sorry

end find_k_l1740_174007


namespace park_trees_after_planting_l1740_174003

theorem park_trees_after_planting (current_trees trees_today trees_tomorrow : ℕ)
  (h1 : current_trees = 7)
  (h2 : trees_today = 5)
  (h3 : trees_tomorrow = 4) :
  current_trees + trees_today + trees_tomorrow = 16 :=
by
  sorry

end park_trees_after_planting_l1740_174003


namespace weight_of_B_l1740_174051

-- Definitions for the weights
variables (A B C : ℝ)

-- Conditions from the problem
def avg_ABC : Prop := (A + B + C) / 3 = 45
def avg_AB : Prop := (A + B) / 2 = 40
def avg_BC : Prop := (B + C) / 2 = 43

-- The theorem to prove the weight of B
theorem weight_of_B (h1 : avg_ABC A B C) (h2 : avg_AB A B) (h3 : avg_BC B C) : B = 31 :=
sorry

end weight_of_B_l1740_174051


namespace nate_age_when_ember_is_14_l1740_174028

theorem nate_age_when_ember_is_14 (nate_age : ℕ) (ember_age : ℕ) 
  (h1 : ember_age = nate_age / 2) (h2 : nate_age = 14) :
  ∃ (years_later : ℕ), ember_age + years_later = 14 ∧ nate_age + years_later = 21 :=
by
  -- sorry to skip the proof, adhering to the instructions
  sorry

end nate_age_when_ember_is_14_l1740_174028


namespace skirt_price_is_13_l1740_174096

-- Definitions based on conditions
def skirts_cost (S : ℝ) : ℝ := 2 * S
def blouses_cost : ℝ := 3 * 6
def total_cost (S : ℝ) : ℝ := skirts_cost S + blouses_cost
def amount_spent : ℝ := 100 - 56

-- The statement we want to prove
theorem skirt_price_is_13 (S : ℝ) (h : total_cost S = amount_spent) : S = 13 :=
by sorry

end skirt_price_is_13_l1740_174096


namespace geometric_series_sum_y_equals_nine_l1740_174002

theorem geometric_series_sum_y_equals_nine : 
  (∑' n : ℕ, (1 / 3) ^ n) * (∑' n : ℕ, (-1 / 3) ^ n) = ∑' n : ℕ, (1 / (9 ^ n)) :=
by
  sorry

end geometric_series_sum_y_equals_nine_l1740_174002


namespace jane_brown_sheets_l1740_174020

theorem jane_brown_sheets :
  ∀ (total_sheets yellow_sheets brown_sheets : ℕ),
    total_sheets = 55 →
    yellow_sheets = 27 →
    brown_sheets = total_sheets - yellow_sheets →
    brown_sheets = 28 := 
by
  intros total_sheets yellow_sheets brown_sheets ht hy hb
  rw [ht, hy] at hb
  simp at hb
  exact hb

end jane_brown_sheets_l1740_174020


namespace lateral_surface_area_of_rotated_triangle_l1740_174077

theorem lateral_surface_area_of_rotated_triangle :
  let AC := 3
  let BC := 4
  let AB := Real.sqrt (AC ^ 2 + BC ^ 2)
  let radius := BC
  let slant_height := AB
  let lateral_surface_area := Real.pi * radius * slant_height
  lateral_surface_area = 20 * Real.pi := by
  sorry

end lateral_surface_area_of_rotated_triangle_l1740_174077


namespace ratio_of_areas_of_concentric_circles_l1740_174004

theorem ratio_of_areas_of_concentric_circles 
  (C1 C2 : ℝ) (r1 r2 : ℝ)
  (h1 : r1 * C1 = 2 * π * r1)
  (h2 : r2 * C2 = 2 * π * r2)
  (h_c1 : 60 / 360 * C1 = 48 / 360 * C2) :
  (π * r1^2) / (π * r2^2) = 16 / 25 := by
  sorry

end ratio_of_areas_of_concentric_circles_l1740_174004


namespace polygon_triangle_division_l1740_174013

theorem polygon_triangle_division (n k : ℕ) (h₁ : n ≥ 3) (h₂ : k ≥ 1):
  k ≥ n - 2 :=
sorry

end polygon_triangle_division_l1740_174013


namespace range_of_a_l1740_174060

noncomputable def A (x : ℝ) : Prop := x^2 - x ≤ 0
noncomputable def B (x : ℝ) (a : ℝ) : Prop := 2^(1 - x) + a ≤ 0

theorem range_of_a (a : ℝ) : (∀ x, A x → B x a) → a ≤ -2 := by
  intro h
  -- Proof steps would go here
  sorry

end range_of_a_l1740_174060


namespace fraction_of_odd_products_is_0_25_l1740_174069

noncomputable def fraction_of_odd_products : ℝ :=
  let odd_products := 8 * 8
  let total_products := 16 * 16
  (odd_products / total_products : ℝ)

theorem fraction_of_odd_products_is_0_25 :
  fraction_of_odd_products = 0.25 :=
by sorry

end fraction_of_odd_products_is_0_25_l1740_174069


namespace no_such_triangle_exists_l1740_174074

theorem no_such_triangle_exists (a b c : ℝ) (h1 : c = 0.2 * a) (h2 : b = 0.25 * (a + b + c)) :
  ¬ (a + b > c ∧ a + c > b ∧ b + c > a) :=
by
  sorry

end no_such_triangle_exists_l1740_174074


namespace largest_quantity_l1740_174084

theorem largest_quantity 
  (A := (2010 / 2009) + (2010 / 2011))
  (B := (2012 / 2011) + (2010 / 2011))
  (C := (2011 / 2010) + (2011 / 2012)) : C > A ∧ C > B := 
by {
  sorry
}

end largest_quantity_l1740_174084


namespace distance_between_trains_l1740_174010

theorem distance_between_trains
  (v1 v2 : ℕ) (d_diff : ℕ)
  (h_v1 : v1 = 50) (h_v2 : v2 = 60) (h_d_diff : d_diff = 100) :
  ∃ d, d = 1100 :=
by
  sorry

-- Explanation:
-- v1 is the speed of the first train.
-- v2 is the speed of the second train.
-- d_diff is the difference in the distances traveled by the two trains at the time of meeting.
-- h_v1 states that the speed of the first train is 50 kmph.
-- h_v2 states that the speed of the second train is 60 kmph.
-- h_d_diff states that the second train travels 100 km more than the first train.
-- The existential statement asserts that there exists a distance d such that d equals 1100 km.

end distance_between_trains_l1740_174010


namespace horizontal_asymptote_degree_l1740_174088

noncomputable def degree (p : Polynomial ℝ) : ℕ := Polynomial.natDegree p

theorem horizontal_asymptote_degree (p : Polynomial ℝ) :
  (∃ l : ℝ, ∀ ε > 0, ∃ N, ∀ x > N, |(p.eval x / (3 * x^7 - 2 * x^3 + x - 4)) - l| < ε) →
  degree p ≤ 7 :=
sorry

end horizontal_asymptote_degree_l1740_174088


namespace shaded_area_correct_l1740_174017

-- Given definitions
def square_side_length : ℝ := 1
def grid_rows : ℕ := 3
def grid_columns : ℕ := 9

def triangle1_area : ℝ := 3
def triangle2_area : ℝ := 1
def triangle3_area : ℝ := 3
def triangle4_area : ℝ := 3

def total_grid_area := (grid_rows * grid_columns : ℕ) * square_side_length^2
def total_unshaded_area := triangle1_area + triangle2_area + triangle3_area + triangle4_area

-- Problem statement
theorem shaded_area_correct :
  total_grid_area - total_unshaded_area = 17 := 
by
  sorry

end shaded_area_correct_l1740_174017


namespace area_of_shaded_region_l1740_174021

theorem area_of_shaded_region 
  (ABCD : Type) 
  (BC : ℝ)
  (height : ℝ)
  (BE : ℝ)
  (CF : ℝ)
  (BC_length : BC = 12)
  (height_length : height = 10)
  (BE_length : BE = 5)
  (CF_length : CF = 3) :
  (BC * height - (1 / 2 * BE * height) - (1 / 2 * CF * height)) = 80 :=
by
  sorry

end area_of_shaded_region_l1740_174021


namespace first_number_is_twenty_l1740_174026

theorem first_number_is_twenty (x : ℕ) : 
  (x + 40 + 60) / 3 = ((10 + 70 + 16) / 3) + 8 → x = 20 := 
by 
  sorry

end first_number_is_twenty_l1740_174026


namespace first_group_correct_l1740_174015

/-- Define the total members in the choir --/
def total_members : ℕ := 70

/-- Define members in the second group --/
def second_group_members : ℕ := 30

/-- Define members in the third group --/
def third_group_members : ℕ := 15

/-- Define the number of members in the first group by subtracting second and third groups members from total members --/
def first_group_members : ℕ := total_members - (second_group_members + third_group_members)

/-- Prove that the first group has 25 members --/
theorem first_group_correct : first_group_members = 25 := by
  -- insert the proof steps here
  sorry

end first_group_correct_l1740_174015


namespace ones_divisible_by_d_l1740_174014

theorem ones_divisible_by_d (d : ℕ) (h1 : ¬ (2 ∣ d)) (h2 : ¬ (5 ∣ d))  : 
  ∃ n, (∃ k : ℕ, n = 10^k - 1) ∧ n % d = 0 := 
sorry

end ones_divisible_by_d_l1740_174014
