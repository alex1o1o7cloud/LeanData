import Mathlib

namespace find_A_of_trig_max_bsquared_plus_csquared_l732_73256

-- Given the geometric conditions and trigonometric identities.

-- Prove: Given 2a * sin B = b * tan A, we have A = π / 3
theorem find_A_of_trig (a b c A B C : Real) (h1 : 2 * a * Real.sin B = b * Real.tan A) :
  A = Real.pi / 3 := sorry

-- Prove: Given a = 2, the maximum value of b^2 + c^2 is 8
theorem max_bsquared_plus_csquared (a b c A : Real) (hA : A = Real.pi / 3) (ha : a = 2) :
  b^2 + c^2 ≤ 8 :=
by
  have hcos : Real.cos A = 1 / 2 := by sorry
  have h : 4 = b^2 + c^2 - b * c * (1/2) := by sorry
  have hmax : b^2 + c^2 + b * c ≤ 8 := by sorry
  sorry -- Proof steps to reach the final result

end find_A_of_trig_max_bsquared_plus_csquared_l732_73256


namespace b_earns_more_than_a_l732_73274

-- Definitions for the conditions
def investments_ratio := (3, 4, 5)
def returns_ratio := (6, 5, 4)
def total_earnings := 10150

-- We need to prove the statement
theorem b_earns_more_than_a (x y : ℕ) (hx : 58 * x * y = 10150) : 2 * x * y = 350 := by
  -- Conditions based on ratios
  let earnings_a := 3 * x * 6 * y
  let earnings_b := 4 * x * 5 * y
  let difference := earnings_b - earnings_a
  
  -- To complete the proof, sorry is used
  sorry

end b_earns_more_than_a_l732_73274


namespace total_bathing_suits_l732_73257

def men_bathing_suits : ℕ := 14797
def women_bathing_suits : ℕ := 4969

theorem total_bathing_suits : men_bathing_suits + women_bathing_suits = 19766 := by
  sorry

end total_bathing_suits_l732_73257


namespace min_sum_of_segments_is_305_l732_73282

noncomputable def min_sum_of_segments : ℕ := 
  let a : ℕ := 3
  let b : ℕ := 5
  100 * a + b

theorem min_sum_of_segments_is_305 : min_sum_of_segments = 305 := by
  sorry

end min_sum_of_segments_is_305_l732_73282


namespace statement_B_false_l732_73215

def f (x : ℝ) : ℝ := 3 * x

def diamondsuit (x y : ℝ) : ℝ := abs (f x - f y)

theorem statement_B_false (x y : ℝ) : 3 * diamondsuit x y ≠ diamondsuit (3 * x) (3 * y) :=
by
  sorry

end statement_B_false_l732_73215


namespace count_two_digit_perfect_squares_divisible_by_4_l732_73213

-- Define what it means to be a two-digit number perfect square divisible by 4
def two_digit_perfect_squares_divisible_by_4 : List ℕ :=
  [16, 36, 64] -- Manually identified two-digit perfect squares which are divisible by 4

-- 6^2 = 36 and 8^2 = 64 both fit, hypothesis checks are already done manually in solution steps
def valid_two_digit_perfect_squares : List ℕ :=
  [16, 25, 36, 49, 64, 81] -- all two-digit perfect squares

-- Define the theorem statement
theorem count_two_digit_perfect_squares_divisible_by_4 :
  (two_digit_perfect_squares_divisible_by_4.count 16 + 
   two_digit_perfect_squares_divisible_by_4.count 36 +
   two_digit_perfect_squares_divisible_by_4.count 64) = 3 :=
by
  -- Proof would go here, omitted by "sorry"
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l732_73213


namespace change_given_back_l732_73242

theorem change_given_back
  (p s t a : ℕ)
  (hp : p = 140)
  (hs : s = 43)
  (ht : t = 15)
  (ha : a = 200) :
  (a - (p + s + t)) = 2 :=
by
  sorry

end change_given_back_l732_73242


namespace AM_GM_inequality_AM_GM_equality_l732_73211

theorem AM_GM_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) / 3 ≤ Real.sqrt ((a^2 + b^2 + c^2) / 3) :=
by
  sorry

theorem AM_GM_equality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) / 3 = Real.sqrt ((a^2 + b^2 + c^2) / 3) ↔ a = b ∧ b = c :=
by
  sorry

end AM_GM_inequality_AM_GM_equality_l732_73211


namespace gcd_840_1764_l732_73267

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l732_73267


namespace volunteer_arrangements_l732_73249

theorem volunteer_arrangements (students : Fin 5 → String) (events : Fin 3 → String)
  (A : String) (high_jump : String)
  (h : ∀ (arrange : Fin 3 → Fin 5), ¬(students (arrange 0) = A ∧ events 0 = high_jump)) :
  ∃! valid_arrangements, valid_arrangements = 48 :=
by
  sorry

end volunteer_arrangements_l732_73249


namespace percent_democrats_l732_73234

/-- The percentage of registered voters in the city who are democrats and republicans -/
def D : ℝ := sorry -- Percent of democrats
def R : ℝ := sorry -- Percent of republicans

-- Given conditions
axiom H1 : D + R = 100
axiom H2 : 0.65 * D + 0.20 * R = 47

-- Statement to prove
theorem percent_democrats : D = 60 :=
by
  sorry

end percent_democrats_l732_73234


namespace hyperbola_k_range_l732_73295

theorem hyperbola_k_range (k : ℝ) : ((k + 2) * (6 - 2 * k) > 0) ↔ (-2 < k ∧ k < 3) := 
sorry

end hyperbola_k_range_l732_73295


namespace smallest_non_consecutive_product_not_factor_of_48_l732_73221

def is_factor (a b : ℕ) : Prop := b % a = 0

def non_consecutive_pairs (x y : ℕ) : Prop := (x ≠ y) ∧ (x + 1 ≠ y) ∧ (y + 1 ≠ x)

theorem smallest_non_consecutive_product_not_factor_of_48 :
  ∃ x y, x ∣ 48 ∧ y ∣ 48 ∧ non_consecutive_pairs x y ∧ ¬ (x * y ∣ 48) ∧ (∀ x' y', x' ∣ 48 ∧ y' ∣ 48 ∧ non_consecutive_pairs x' y' ∧ ¬ (x' * y' ∣ 48) → x' * y' ≥ 18) :=
by
  sorry

end smallest_non_consecutive_product_not_factor_of_48_l732_73221


namespace remainder_a_cubed_l732_73290

theorem remainder_a_cubed {a n : ℤ} (hn : 0 < n) (hinv : a * a ≡ 1 [ZMOD n]) (ha : a ≡ -1 [ZMOD n]) : a^3 ≡ -1 [ZMOD n] := 
sorry

end remainder_a_cubed_l732_73290


namespace paint_problem_l732_73251

-- Definitions based on conditions
def roomsInitiallyPaintable := 50
def roomsAfterLoss := 40
def cansLost := 5

-- The number of rooms each can could paint
def roomsPerCan := (roomsInitiallyPaintable - roomsAfterLoss) / cansLost

-- The total number of cans originally owned
def originalCans := roomsInitiallyPaintable / roomsPerCan

-- Theorem to prove the number of original cans equals 25
theorem paint_problem : originalCans = 25 := by
  sorry

end paint_problem_l732_73251


namespace train_speed_l732_73205

noncomputable def jogger_speed : ℝ := 9 -- speed in km/hr
noncomputable def jogger_distance : ℝ := 150 / 1000 -- distance in km
noncomputable def train_length : ℝ := 100 / 1000 -- length in km
noncomputable def time_to_pass : ℝ := 25 -- time in seconds

theorem train_speed 
  (v_j : ℝ := jogger_speed)
  (d_j : ℝ := jogger_distance)
  (L : ℝ := train_length)
  (t : ℝ := time_to_pass) :
  (train_speed_in_kmh : ℝ) = 36 :=
by 
  sorry

end train_speed_l732_73205


namespace contradiction_proof_l732_73224

theorem contradiction_proof (a b c : ℝ) (h1 : 0 < a ∧ a < 2) (h2 : 0 < b ∧ b < 2) (h3 : 0 < c ∧ c < 2) :
  ¬ (a * (2 - b) > 1 ∧ b * (2 - c) > 1 ∧ c * (2 - a) > 1) :=
sorry

end contradiction_proof_l732_73224


namespace game_winning_strategy_l732_73220

theorem game_winning_strategy (n : ℕ) : (n % 2 = 0 → ∃ strategy : ℕ → ℕ, ∀ m, strategy m = 1) ∧ (n % 2 = 1 → ∃ strategy : ℕ → ℕ, ∀ m, strategy m = 2) :=
by
  sorry

end game_winning_strategy_l732_73220


namespace smallest_possible_e_l732_73296

-- Definitions based on given conditions
def polynomial (x : ℝ) (a b c d e : ℝ) : ℝ := a * x^4 + b * x^3 + c * x^2 + d * x + e

-- The given polynomial has roots -3, 4, 8, and -1/4, and e is positive integer
theorem smallest_possible_e :
  ∃ (a b c d e : ℤ), polynomial x a b c d e = 4*x^4 - 32*x^3 - 23*x^2 + 104*x + 96 ∧ e > 0 ∧ e = 96 :=
by
  sorry

end smallest_possible_e_l732_73296


namespace parallel_lines_distance_sum_l732_73270

theorem parallel_lines_distance_sum (b c : ℝ) 
  (h1 : ∃ k : ℝ, 6 = 3 * k ∧ b = 4 * k) 
  (h2 : (abs ((c / 2) - 5) / (Real.sqrt (3^2 + 4^2))) = 3) : 
  b + c = 48 ∨ b + c = -12 := by
  sorry

end parallel_lines_distance_sum_l732_73270


namespace maximize_revenue_l732_73200

-- Define the revenue function
def revenue (p : ℝ) : ℝ :=
  p * (150 - 4 * p)

-- Define the price constraints
def price_constraint (p : ℝ) : Prop :=
  0 ≤ p ∧ p ≤ 30

-- The theorem statement to prove that p = 19 maximizes the revenue
theorem maximize_revenue : ∀ p: ℕ, price_constraint p → revenue p ≤ revenue 19 :=
by
  sorry

end maximize_revenue_l732_73200


namespace find_a_l732_73243
-- Import the entire Mathlib to ensure all necessary primitives and theorems are available.

-- Define a constant equation representing the conditions.
def equation (x a : ℝ) := 3 * x + 2 * a

-- Define a theorem to prove the condition => result structure.
theorem find_a (h : equation 2 a = 0) : a = -3 :=
by sorry

end find_a_l732_73243


namespace largest_n_for_factoring_l732_73275

theorem largest_n_for_factoring :
  ∃ (n : ℤ), 
    (∀ A B : ℤ, (5 * B + A = n ∧ A * B = 60) → (5 * B + A ≤ n)) ∧
    n = 301 :=
by sorry

end largest_n_for_factoring_l732_73275


namespace smallest_w_factor_l732_73287

theorem smallest_w_factor (w : ℕ) (hw : w > 0) :
  (∃ w, 2^4 ∣ 1452 * w ∧ 3^3 ∣ 1452 * w ∧ 13^3 ∣ 1452 * w) ↔ w = 79092 :=
by sorry

end smallest_w_factor_l732_73287


namespace prob_defective_l732_73265

/-- Assume there are two boxes of components. 
    The first box contains 10 pieces, including 2 defective ones; 
    the second box contains 20 pieces, including 3 defective ones. --/
def box1_total : ℕ := 10
def box1_defective : ℕ := 2
def box2_total : ℕ := 20
def box2_defective : ℕ := 3

/-- Randomly select one box from the two boxes, 
    and then randomly pick 1 component from that box. --/
def prob_select_box : ℚ := 1 / 2

/-- Probability of selecting a defective component given that box 1 was selected. --/
def prob_defective_given_box1 : ℚ := box1_defective / box1_total

/-- Probability of selecting a defective component given that box 2 was selected. --/
def prob_defective_given_box2 : ℚ := box2_defective / box2_total

/-- The probability of selecting a defective component is 7/40. --/
theorem prob_defective :
  prob_select_box * prob_defective_given_box1 + prob_select_box * prob_defective_given_box2 = 7 / 40 :=
sorry

end prob_defective_l732_73265


namespace simplify_expression_l732_73229

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : (2 / (x^2 - 1)) / (1 / (x - 1)) = 2 / (x + 1) :=
by sorry

end simplify_expression_l732_73229


namespace sum_of_squares_of_sum_and_difference_l732_73217

theorem sum_of_squares_of_sum_and_difference (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 8) : 
  (x + y)^2 + (x - y)^2 = 640 :=
by
  sorry

end sum_of_squares_of_sum_and_difference_l732_73217


namespace binom_9_5_l732_73268

open Nat

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Theorem to prove binom(9, 5) = 126
theorem binom_9_5 : binom 9 5 = 126 := by
  sorry

end binom_9_5_l732_73268


namespace inequality_system_range_l732_73241

theorem inequality_system_range (a : ℝ) :
  (∃ (x : ℤ), (6 * (x : ℝ) + 2 > 3 * (x : ℝ) + 5) ∧ (2 * (x : ℝ) - a ≤ 0)) ∧
  (∀ x : ℤ, (6 * (x : ℝ) + 2 > 3 * (x : ℝ) + 5) ∧ (2 * (x : ℝ) - a ≤ 0) → (x = 2 ∨ x = 3)) →
  6 ≤ a ∧ a < 8 :=
by
  sorry

end inequality_system_range_l732_73241


namespace max_halls_l732_73255

theorem max_halls (n : ℕ) (hall : ℕ → ℕ) (H : ∀ n, hall n = hall (3 * n + 1) ∧ hall n = hall (n + 10)) :
  ∃ (m : ℕ), m = 3 :=
by
  sorry

end max_halls_l732_73255


namespace committee_meeting_l732_73208

theorem committee_meeting : 
  ∃ (A B : ℕ), 2 * A + B = 7 ∧ A + 2 * B = 11 ∧ A + B = 6 :=
by 
  sorry

end committee_meeting_l732_73208


namespace tiffany_lives_l732_73227

theorem tiffany_lives (initial_lives lives_lost lives_after_next_level lives_gained : ℕ)
  (h1 : initial_lives = 43)
  (h2 : lives_lost = 14)
  (h3 : lives_after_next_level = 56)
  (h4 : lives_gained = lives_after_next_level - (initial_lives - lives_lost)) :
  lives_gained = 27 :=
by {
  sorry
}

end tiffany_lives_l732_73227


namespace olaf_travels_miles_l732_73216

-- Define the given conditions
def men : ℕ := 25
def per_day_water_per_man : ℚ := 1 / 2
def boat_mileage_per_day : ℕ := 200
def total_water : ℚ := 250

-- Define the daily water consumption for the crew
def daily_water_consumption : ℚ := men * per_day_water_per_man

-- Define the number of days the water will last
def days_water_lasts : ℚ := total_water / daily_water_consumption

-- Define the total miles traveled
def total_miles_traveled : ℚ := days_water_lasts * boat_mileage_per_day

-- Theorem statement to prove the total miles traveled is 4000 miles
theorem olaf_travels_miles : total_miles_traveled = 4000 := by
  sorry

end olaf_travels_miles_l732_73216


namespace least_possible_value_of_D_l732_73228

-- Defining the conditions as theorems
theorem least_possible_value_of_D :
  ∃ (A B C D : ℕ), 
  (A + B + C + D) / 4 = 18 ∧
  A = 3 * B ∧
  B = C - 2 ∧
  C = 3 / 2 * D ∧
  (∀ x : ℕ, x ≥ 10 → D = x) := 
sorry

end least_possible_value_of_D_l732_73228


namespace area_of_CEF_l732_73259

-- Definitions of points and triangles based on given ratios
def is_right_triangle (A B C : Type) : Prop := sorry -- Placeholder for right triangle condition

def divides_ratio (A B : Type) (ratio : ℚ) : Prop := sorry -- Placeholder for ratio division condition

def area_of_triangle (A B C : Type) : ℚ := sorry -- Function to calculate area of triangle - placeholder

theorem area_of_CEF {A B C E F : Type} 
  (h1 : is_right_triangle A B C)
  (h2 : divides_ratio A C (1/4))
  (h3 : divides_ratio A B (2/3))
  (h4 : area_of_triangle A B C = 50) : 
  area_of_triangle C E F = 25 :=
sorry

end area_of_CEF_l732_73259


namespace total_cost_is_eight_times_short_cost_l732_73277

variables (x : ℝ)
def cost_shorts := x
def cost_shirt := x
def cost_boots := 4 * x
def cost_shin_guards := 2 * x

theorem total_cost_is_eight_times_short_cost
    (cond_shirt : cost_shorts x + cost_shirt x = 2*x)
    (cond_boots : cost_shorts x + cost_boots x = 5*x)
    (cond_shin_guards : cost_shorts x + cost_shin_guards x = 3*x) :
    cost_shorts x + cost_shirt x + cost_boots x + cost_shin_guards x = 8*x :=
by
  sorry

end total_cost_is_eight_times_short_cost_l732_73277


namespace walkway_area_296_l732_73240

theorem walkway_area_296 :
  let bed_length := 4
  let bed_width := 3
  let num_rows := 4
  let num_columns := 3
  let walkway_width := 2
  let total_bed_area := num_rows * num_columns * bed_length * bed_width
  let total_garden_width := num_columns * bed_length + (num_columns + 1) * walkway_width
  let total_garden_height := num_rows * bed_width + (num_rows + 1) * walkway_width
  let total_garden_area := total_garden_width * total_garden_height
  let total_walkway_area := total_garden_area - total_bed_area
  total_walkway_area = 296 :=
by 
  sorry

end walkway_area_296_l732_73240


namespace range_of_m_l732_73204

theorem range_of_m {x m : ℝ} 
  (h1 : 1 / 3 < x) 
  (h2 : x < 1 / 2) 
  (h3 : |x - m| < 1) : 
  -1 / 2 ≤ m ∧ m ≤ 4 / 3 :=
by
  sorry

end range_of_m_l732_73204


namespace max_leap_years_l732_73271

theorem max_leap_years (years : ℕ) (leap_interval : ℕ) (total_years : ℕ) :
  leap_interval = 5 ∧ total_years = 200 → (years = total_years / leap_interval) :=
by
  sorry

end max_leap_years_l732_73271


namespace find_x_l732_73252

theorem find_x (x : ℝ) (h : 6 * x + 3 * x + 4 * x + 2 * x = 360) : x = 24 :=
sorry

end find_x_l732_73252


namespace sum_of_A_and_B_l732_73209

theorem sum_of_A_and_B:
  ∃ A B : ℕ, (A = 2 + 4) ∧ (B - 3 = 1) ∧ (A < 10) ∧ (B < 10) ∧ (A + B = 10) :=
by 
  sorry

end sum_of_A_and_B_l732_73209


namespace number_of_integer_pairs_satisfying_conditions_l732_73238

noncomputable def count_integer_pairs (n m : ℕ) : ℕ := Nat.choose (n-1) (m-1)

theorem number_of_integer_pairs_satisfying_conditions :
  ∃ (a b c x y : ℕ), a + b + c = 55 ∧ a + b + c + x + y = 71 ∧ x + y > a + b + c → count_integer_pairs 55 3 * count_integer_pairs 16 2 = 21465 := sorry

end number_of_integer_pairs_satisfying_conditions_l732_73238


namespace platform_length_is_500_l732_73222

-- Define the length of the train, the time to cross a tree, and the time to cross a platform as given conditions
def train_length := 1500 -- in meters
def time_to_cross_tree := 120 -- in seconds
def time_to_cross_platform := 160 -- in seconds

-- Define the speed based on the train crossing the tree
def train_speed := train_length / time_to_cross_tree -- in meters/second

-- Define the total distance covered when crossing the platform
def total_distance_crossing_platform (platform_length : ℝ) := train_length + platform_length

-- State the main theorem to prove the platform length is 500 meters
theorem platform_length_is_500 (platform_length : ℝ) :
  (train_speed * time_to_cross_platform = total_distance_crossing_platform platform_length) → platform_length = 500 :=
by
  sorry

end platform_length_is_500_l732_73222


namespace total_weight_l732_73299

def w1 : ℝ := 9.91
def w2 : ℝ := 4.11

theorem total_weight : w1 + w2 = 14.02 := by 
  sorry

end total_weight_l732_73299


namespace polygon_sides_l732_73260

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 3 * 360) : 
  n = 8 := 
sorry

end polygon_sides_l732_73260


namespace total_time_correct_l732_73235

-- Definitions based on problem conditions
def first_time : ℕ := 15
def time_increment : ℕ := 7
def number_of_flights : ℕ := 7

-- Time taken for a specific flight
def time_for_nth_flight (n : ℕ) : ℕ := first_time + (n - 1) * time_increment

-- Sum of the times for the first seven flights
def total_time : ℕ := (number_of_flights * (first_time + time_for_nth_flight number_of_flights)) / 2

-- Statement to be proven
theorem total_time_correct : total_time = 252 := 
by
  sorry

end total_time_correct_l732_73235


namespace Hayley_l732_73245

-- Definitions based on the given conditions
def num_friends : ℕ := 9
def stickers_per_friend : ℕ := 8

-- Theorem statement
theorem Hayley's_total_stickers : num_friends * stickers_per_friend = 72 := by
  sorry

end Hayley_l732_73245


namespace quadratic_solution_l732_73263

theorem quadratic_solution (x : ℝ) : (x^2 - 3 * x + 2 < 0) ↔ (1 < x ∧ x < 2) :=
by
  sorry

end quadratic_solution_l732_73263


namespace which_set_forms_triangle_l732_73212

def satisfies_triangle_inequality (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem which_set_forms_triangle : 
  satisfies_triangle_inequality 4 3 6 ∧ 
  ¬ satisfies_triangle_inequality 1 2 3 ∧ 
  ¬ satisfies_triangle_inequality 7 8 16 ∧ 
  ¬ satisfies_triangle_inequality 9 10 20 :=
by
  sorry

end which_set_forms_triangle_l732_73212


namespace polynomial_sum_l732_73266

def p (x : ℝ) := -4 * x^2 + 2 * x - 5
def q (x : ℝ) := -6 * x^2 + 4 * x - 9
def r (x : ℝ) := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l732_73266


namespace problem_statement_l732_73225

theorem problem_statement (d : ℕ) (h1 : d > 0) (h2 : d ∣ (5 + 2022^2022)) :
  (∃ x y : ℤ, d = 2 * x^2 + 2 * x * y + 3 * y^2) ↔ (d % 20 = 3 ∨ d % 20 = 7) :=
by
  sorry

end problem_statement_l732_73225


namespace range_of_a_value_of_a_l732_73291

-- Problem 1
theorem range_of_a (a : ℝ) :
  (∃ x, (2 < x ∧ x < 4) ∧ (a < x ∧ x < 3 * a)) ↔ (4 / 3 ≤ a ∧ a < 4) :=
sorry

-- Problem 2
theorem value_of_a (a : ℝ) :
  (∀ x, (2 < x ∧ x < 4) ∨ (a < x ∧ x < 3 * a) ↔ (2 < x ∧ x < 6)) ↔ (a = 2) :=
sorry

end range_of_a_value_of_a_l732_73291


namespace difference_between_extremes_l732_73233

/-- Define the structure of a 3-digit integer and its digits. -/
structure ThreeDigitInteger where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  val : ℕ := 100 * hundreds + 10 * tens + units

/-- Define the problem conditions. -/
def satisfiesConditions (x : ThreeDigitInteger) : Prop :=
  x.hundreds > 0 ∧
  4 * x.hundreds = 2 * x.tens ∧
  2 * x.tens = x.units

/-- Given conditions prove the difference between the two greatest possible values of x is 124. -/
theorem difference_between_extremes :
  ∃ (x₁ x₂ : ThreeDigitInteger), 
    satisfiesConditions x₁ ∧ satisfiesConditions x₂ ∧
    (x₁.val = 248 ∧ x₂.val = 124 ∧ (x₁.val - x₂.val = 124)) :=
sorry

end difference_between_extremes_l732_73233


namespace favorite_movies_total_hours_l732_73214

theorem favorite_movies_total_hours (michael_hrs joyce_hrs nikki_hrs ryn_hrs sam_hrs alex_hrs : ℕ)
  (H1 : nikki_hrs = 30)
  (H2 : michael_hrs = nikki_hrs / 3)
  (H3 : joyce_hrs = michael_hrs + 2)
  (H4 : ryn_hrs = (4 * nikki_hrs) / 5)
  (H5 : sam_hrs = (3 * joyce_hrs) / 2)
  (H6 : alex_hrs = 2 * michael_hrs) :
  michael_hrs + joyce_hrs + nikki_hrs + ryn_hrs + sam_hrs + alex_hrs = 114 := 
sorry

end favorite_movies_total_hours_l732_73214


namespace largest_multiple_of_8_less_than_100_l732_73283

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, 8 * n < 100 ∧ (∀ m : ℕ, 8 * m < 100 → m ≤ n) ∧ 8 * n = 96 :=
by
  sorry

end largest_multiple_of_8_less_than_100_l732_73283


namespace price_of_each_apple_l732_73272

theorem price_of_each_apple
  (bike_cost: ℝ) (repair_cost_percent: ℝ) (remaining_percentage: ℝ)
  (total_apples_sold: ℕ) (repair_cost: ℝ) (total_money_earned: ℝ)
  (price_per_apple: ℝ) :
  bike_cost = 80 →
  repair_cost_percent = 0.25 →
  remaining_percentage = 0.2 →
  total_apples_sold = 20 →
  repair_cost = repair_cost_percent * bike_cost →
  total_money_earned = repair_cost / (1 - remaining_percentage) →
  price_per_apple = total_money_earned / total_apples_sold →
  price_per_apple = 1.25 := 
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end price_of_each_apple_l732_73272


namespace pamela_skittles_correct_l732_73269

def pamela_initial_skittles := 50
def pamela_gives_skittles_to_karen := 7
def pamela_receives_skittles_from_kevin := 3
def pamela_shares_percentage := 20

def pamela_final_skittles : Nat :=
  let after_giving := pamela_initial_skittles - pamela_gives_skittles_to_karen
  let after_receiving := after_giving + pamela_receives_skittles_from_kevin
  let share_amount := (after_receiving * pamela_shares_percentage) / 100
  let rounded_share := Nat.floor share_amount
  let final_count := after_receiving - rounded_share
  final_count

theorem pamela_skittles_correct :
  pamela_final_skittles = 37 := by
  sorry

end pamela_skittles_correct_l732_73269


namespace arrange_students_l732_73226

theorem arrange_students (students : Fin 7 → Prop) : 
  ∃ arrangements : ℕ, arrangements = 140 :=
by
  -- Define selection of 6 out of 7
  let selection_ways := Nat.choose 7 6
  -- Define arrangement of 6 into two groups of 3 each
  let arrangement_ways := (Nat.choose 6 3) * (Nat.choose 3 3)
  -- Calculate total arrangements by multiplying the two values
  let total_arrangements := selection_ways * arrangement_ways
  use total_arrangements
  simp [selection_ways, arrangement_ways, total_arrangements]
  exact rfl

end arrange_students_l732_73226


namespace opposite_of_neg_five_l732_73237

/-- Definition of the opposite of a number -/
def opposite (a : Int) : Int := -a

theorem opposite_of_neg_five : opposite (-5) = 5 := by
  sorry

end opposite_of_neg_five_l732_73237


namespace prove_3a_3b_3c_l732_73288

variable (a b c : ℝ)

def condition1 := b + c = 15 - 2 * a
def condition2 := a + c = -18 - 3 * b
def condition3 := a + b = 8 - 4 * c
def condition4 := a - b + c = 3

theorem prove_3a_3b_3c (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b c) (h4 : condition4 a b c) :
  3 * a + 3 * b + 3 * c = 24 / 5 :=
sorry

end prove_3a_3b_3c_l732_73288


namespace scientific_notation_of_3300000_l732_73297

theorem scientific_notation_of_3300000 : 3300000 = 3.3 * 10^6 :=
by
  sorry

end scientific_notation_of_3300000_l732_73297


namespace least_four_digit_perfect_square_and_cube_l732_73219

theorem least_four_digit_perfect_square_and_cube :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (∃ m1 : ℕ, n = m1^2) ∧ (∃ m2 : ℕ, n = m2^3) ∧ n = 4096 := sorry

end least_four_digit_perfect_square_and_cube_l732_73219


namespace evaluate_expression_l732_73298

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by {
  sorry -- Proof goes here
}

end evaluate_expression_l732_73298


namespace slope_of_line_through_origin_and_A_l732_73203

theorem slope_of_line_through_origin_and_A :
  ∀ (x1 y1 x2 y2 : ℝ), (x1 = 0) → (y1 = 0) → (x2 = -2) → (y2 = -2) →
  (y2 - y1) / (x2 - x1) = 1 :=
by intros; sorry

end slope_of_line_through_origin_and_A_l732_73203


namespace makenna_garden_larger_by_160_l732_73207

def area (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def karl_length : ℕ := 22
def karl_width : ℕ := 50
def makenna_length : ℕ := 28
def makenna_width : ℕ := 45

def karl_area : ℕ := area karl_length karl_width
def makenna_area : ℕ := area makenna_length makenna_width

theorem makenna_garden_larger_by_160 :
  makenna_area = karl_area + 160 := by
  sorry

end makenna_garden_larger_by_160_l732_73207


namespace train_length_l732_73281

theorem train_length (L : ℕ) :
  (L + 350) / 15 = (L + 500) / 20 → L = 100 := 
by
  intro h
  sorry

end train_length_l732_73281


namespace domain_log_base_4_l732_73223

theorem domain_log_base_4 (x : ℝ) : {x // x + 2 > 0} = {x | x > -2} :=
by
  sorry

end domain_log_base_4_l732_73223


namespace combination_15_3_l732_73264

theorem combination_15_3 :
  (Nat.choose 15 3 = 455) :=
by
  sorry

end combination_15_3_l732_73264


namespace negation_of_universal_prop_l732_73230

-- Define the proposition p
def p : Prop := ∀ x : ℝ, Real.sin x ≤ 1

-- Define the negation of p
def neg_p : Prop := ∃ x : ℝ, Real.sin x > 1

-- The theorem stating the equivalence
theorem negation_of_universal_prop : ¬p ↔ neg_p := 
by sorry

end negation_of_universal_prop_l732_73230


namespace final_price_difference_l732_73250

noncomputable def OP : ℝ := 78.2 / 0.85
noncomputable def IP : ℝ := 78.2 + 0.25 * 78.2
noncomputable def DP : ℝ := 97.75 - 0.10 * 97.75
noncomputable def FP : ℝ := 87.975 + 0.0725 * 87.975

theorem final_price_difference : OP - FP = -2.3531875 := 
by sorry

end final_price_difference_l732_73250


namespace brownies_pieces_count_l732_73239

theorem brownies_pieces_count
  (pan_length pan_width piece_length piece_width : ℕ)
  (h1 : pan_length = 24)
  (h2 : pan_width = 15)
  (h3 : piece_length = 3)
  (h4 : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 60 :=
by
  sorry

end brownies_pieces_count_l732_73239


namespace ball_hits_ground_l732_73210

noncomputable def ball_height (t : ℝ) : ℝ := -9 * t^2 + 15 * t + 72

theorem ball_hits_ground :
  (∃ t : ℝ, t = (5 + Real.sqrt 313) / 6 ∧ ball_height t = 0) :=
sorry

end ball_hits_ground_l732_73210


namespace volume_of_tetrahedron_l732_73262

theorem volume_of_tetrahedron 
  (A B C D E : ℝ)
  (AB AD AE: ℝ)
  (h_AB : AB = 3)
  (h_AD : AD = 4)
  (h_AE : AE = 1)
  (V : ℝ) :
  (V = (4 * Real.sqrt 3) / 3) :=
sorry

end volume_of_tetrahedron_l732_73262


namespace order_theorems_l732_73206

theorem order_theorems : 
  ∃ a b c d e f g : String,
    (a = "H") ∧ (b = "M") ∧ (c = "P") ∧ (d = "C") ∧ 
    (e = "V") ∧ (f = "S") ∧ (g = "E") ∧
    (a = "Heron's Theorem") ∧
    (b = "Menelaus' Theorem") ∧
    (c = "Pascal's Theorem") ∧
    (d = "Ceva's Theorem") ∧
    (e = "Varignon's Theorem") ∧
    (f = "Stewart's Theorem") ∧
    (g = "Euler's Theorem") := 
  sorry

end order_theorems_l732_73206


namespace SetC_not_right_angled_triangle_l732_73258

theorem SetC_not_right_angled_triangle :
  ¬ (7^2 + 24^2 = 26^2) :=
by 
  have h : 7^2 + 24^2 ≠ 26^2 := by decide
  exact h

end SetC_not_right_angled_triangle_l732_73258


namespace increasing_function_inv_condition_l732_73293

-- Given a strictly increasing real-valued function f on ℝ with an inverse,
-- satisfying the condition f(x) + f⁻¹(x) = 2x for all x in ℝ,
-- prove that f(x) = x + b, where b is a real constant.

theorem increasing_function_inv_condition (f : ℝ → ℝ) (hf_strict_mono : StrictMono f)
  (hf_inv : ∀ x, f (f⁻¹ x) = x ∧ f⁻¹ (f x) = x)
  (hf_condition : ∀ x, f x + f⁻¹ x = 2 * x) :
  ∃ b : ℝ, ∀ x, f x = x + b :=
sorry

end increasing_function_inv_condition_l732_73293


namespace fraction_simplification_l732_73254

theorem fraction_simplification : 
  (1/5 - 1/6) / (1/3 - 1/4) = 2/5 := 
by 
  sorry

end fraction_simplification_l732_73254


namespace line_slope_intercept_l732_73279

theorem line_slope_intercept :
  (∀ (x y : ℝ), 3 * (x + 2) - 4 * (y - 8) = 0 → y = (3/4) * x + 9.5) :=
sorry

end line_slope_intercept_l732_73279


namespace range_of_a_l732_73285

-- Define the conditions and the problem
def neg_p (x : ℝ) : Prop := -3 < x ∧ x < 0
def neg_q (x : ℝ) (a : ℝ) : Prop := x > a
def p (x : ℝ) : Prop := x ≤ -3 ∨ x ≥ 0
def q (x : ℝ) (a : ℝ) : Prop := x ≤ a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, neg_p x → ¬ p x) ∧
  (∀ x : ℝ, neg_q x a → ¬ q x a) ∧
  (∀ x : ℝ, q x a → p x) ∧
  (∃ x : ℝ, ¬ (q x a → p x)) →
  a ≤ -3 :=
by
  sorry

end range_of_a_l732_73285


namespace largest_of_options_l732_73294

theorem largest_of_options :
  max (2 + 0 + 1 + 3) (max (2 * 0 + 1 + 3) (max (2 + 0 * 1 + 3) (max (2 + 0 + 1 * 3) (2 * 0 * 1 * 3)))) = 2 + 0 + 1 + 3 := by sorry

end largest_of_options_l732_73294


namespace a_2018_mod_49_l732_73284

def a (n : ℕ) : ℕ := 6^n + 8^n

theorem a_2018_mod_49 : (a 2018) % 49 = 0 := by
  sorry

end a_2018_mod_49_l732_73284


namespace Adam_total_candy_l732_73246

theorem Adam_total_candy :
  (2 + 5) * 4 = 28 := 
by 
  sorry

end Adam_total_candy_l732_73246


namespace roots_quartic_sum_l732_73201

theorem roots_quartic_sum (c d : ℝ) (h1 : c + d = 3) (h2 : c * d = 1) (hc : Polynomial.eval c (Polynomial.C (-1) + Polynomial.X ^ 4 - 6 * Polynomial.X ^ 3 - 4 * Polynomial.X) = 0) (hd : Polynomial.eval d (Polynomial.C (-1) + Polynomial.X ^ 4 - 6 * Polynomial.X ^ 3 - 4 * Polynomial.X) = 0) :
  c * d + c + d = 4 :=
by
  sorry

end roots_quartic_sum_l732_73201


namespace Tucker_last_number_l732_73273

-- Define the sequence of numbers said by Todd, Tadd, and Tucker
def game_sequence (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 3
  else if n = 4 then 4
  else if n = 5 then 5
  else if n = 6 then 6
  else sorry -- Define recursively for subsequent rounds

-- Condition: The game ends when they reach the number 1000.
def game_end := 1000

-- Define the function to determine the last number said by Tucker
def last_number_said_by_Tucker (end_num : ℕ) : ℕ :=
  -- Assuming this function correctly calculates the last number said by Tucker
  if end_num = game_end then 1000 else sorry

-- Problem statement to prove
theorem Tucker_last_number : last_number_said_by_Tucker game_end = 1000 := by
  sorry

end Tucker_last_number_l732_73273


namespace ellipse_focal_length_l732_73202

theorem ellipse_focal_length {m : ℝ} : 
  (m > 2 ∧ 4 ≤ 10 - m ∧ 4 ≤ m - 2) → 
  (10 - m - (m - 2) = 4) ∨ (m - 2 - (10 - m) = 4) :=
by
  sorry

end ellipse_focal_length_l732_73202


namespace base_7_units_digit_l732_73276

theorem base_7_units_digit : ((156 + 97) % 7) = 1 := 
by
  sorry

end base_7_units_digit_l732_73276


namespace sequence_S_n_a_n_l732_73232

noncomputable def sequence_S (n : ℕ) : ℝ := -1 / (n : ℝ)

noncomputable def sequence_a (n : ℕ) : ℝ :=
  if n = 1 then -1 else 1 / ((n : ℝ) * (n - 1))

theorem sequence_S_n_a_n (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  a 1 = -1 →
  (∀ n, (a (n + 1)) / (S (n + 1)) = S n) →
  S n = sequence_S n ∧ a n = sequence_a n :=
by
  intros h1 h2
  sorry

end sequence_S_n_a_n_l732_73232


namespace salaries_proof_l732_73289

-- Define salaries as real numbers
variables (a b c d : ℝ)

-- Define assumptions
def conditions := 
  (a + b + c + d = 4000) ∧
  (0.05 * a + 0.15 * b = c) ∧ 
  (0.25 * d = 0.3 * b) ∧
  (b = 3 * c)

-- Define the solution as found
def solution :=
  (a = 2365.55) ∧
  (b = 645.15) ∧
  (c = 215.05) ∧
  (d = 774.18)

-- Prove that given the conditions, the solution holds
theorem salaries_proof : 
  (conditions a b c d) → (solution a b c d) := by
  sorry

end salaries_proof_l732_73289


namespace gcd_7_nplus2_8_2nplus1_l732_73247

theorem gcd_7_nplus2_8_2nplus1 : 
  ∃ d : ℕ, (∀ n : ℕ, d ∣ (7^(n+2) + 8^(2*n+1))) ∧ (∀ n : ℕ, d = 57) :=
sorry

end gcd_7_nplus2_8_2nplus1_l732_73247


namespace find_sets_A_B_l732_73280

def C : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

def S : Finset ℕ := {4, 5, 9, 14, 23, 37}

theorem find_sets_A_B :
  ∃ (A B : Finset ℕ), 
  (A ∩ B = ∅) ∧ 
  (A ∪ B = C) ∧ 
  (∀ (x y : ℕ), x ≠ y → x ∈ A → y ∈ A → x + y ∉ S) ∧ 
  (∀ (x y : ℕ), x ≠ y → x ∈ B → y ∈ B → x + y ∉ S) ∧ 
  (A = {1, 2, 5, 6, 10, 11, 14, 15, 16, 19, 20}) ∧ 
  (B = {3, 4, 7, 8, 9, 12, 13, 17, 18}) :=
by
  sorry

end find_sets_A_B_l732_73280


namespace number_ordering_l732_73236

theorem number_ordering : (10^5 < 2^20) ∧ (2^20 < 5^10) :=
by {
  -- We place the proof steps here
  sorry
}

end number_ordering_l732_73236


namespace original_gift_card_value_l732_73231

def gift_card_cost_per_pound : ℝ := 8.58
def coffee_pounds_bought : ℕ := 4
def remaining_balance_after_purchase : ℝ := 35.68

theorem original_gift_card_value :
  (remaining_balance_after_purchase + coffee_pounds_bought * gift_card_cost_per_pound) = 70.00 :=
by
  -- Proof goes here
  sorry

end original_gift_card_value_l732_73231


namespace solve_for_a_and_b_l732_73278
-- Import the necessary library

open Classical

variable (a b x : ℝ)

theorem solve_for_a_and_b (h1 : 0 ≤ x) (h2 : x < 1) (h3 : x + 2 * a ≥ 4) (h4 : (2 * x - b) / 3 < 1) : a + b = 1 := 
by
  sorry

end solve_for_a_and_b_l732_73278


namespace pizza_problem_l732_73286

theorem pizza_problem (diameter : ℝ) (sectors : ℕ) (h1 : diameter = 18) (h2 : sectors = 4) : 
  let R := diameter / 2 
  let θ := (2 * Real.pi / sectors : ℝ)
  let m := 2 * R * Real.sin (θ / 2) 
  (m^2 = 162) := by
  sorry

end pizza_problem_l732_73286


namespace farmer_field_area_l732_73292

theorem farmer_field_area (m : ℝ) (h : (3 * m + 5) * (m + 1) = 104) : m = 4.56 :=
sorry

end farmer_field_area_l732_73292


namespace set_M_properties_l732_73244

def f (x : ℝ) : ℝ := |x| - |2 * x - 1|

def M : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem set_M_properties :
  M = { x | 0 < x ∧ x < 2 } ∧
  (∀ a, a ∈ M → 
    ((0 < a ∧ a < 1) → (a^2 - a + 1 < 1 / a)) ∧
    (a = 1 → (a^2 - a + 1 = 1 / a)) ∧
    ((1 < a ∧ a < 2) → (a^2 - a + 1 > 1 / a))) := 
by
  sorry

end set_M_properties_l732_73244


namespace initial_number_of_earning_members_l732_73248

theorem initial_number_of_earning_members (n : ℕ) 
  (h1 : 840 * n - 650 * (n - 1) = 1410) : n = 4 :=
by {
  -- Proof omitted
  sorry
}

end initial_number_of_earning_members_l732_73248


namespace complement_union_l732_73218

open Set

variable (U : Set ℝ) (A B : Set ℝ)

def C_U (s : Set ℝ) : Set ℝ := U \ s

theorem complement_union (U : Set ℝ) (A B : Set ℝ) (hU : U = univ) (hA : A = { x | x < 0 }) (hB : B = { x | x ≥ 2 }) :
  C_U U (A ∪ B) = { x | 0 ≤ x ∧ x < 2 } :=
by
  sorry

end complement_union_l732_73218


namespace weight_of_currants_l732_73253

noncomputable def packing_density : ℝ := 0.74
noncomputable def water_density : ℝ := 1000
noncomputable def bucket_volume : ℝ := 0.01

theorem weight_of_currants :
  (water_density * (packing_density * bucket_volume)) = 7.4 :=
by
  sorry

end weight_of_currants_l732_73253


namespace intersection_of_A_and_B_l732_73261

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := { x | ∃ m : ℕ, x = 2 * m }

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := 
by sorry

end intersection_of_A_and_B_l732_73261
