import Mathlib

namespace compound_statement_false_l820_82069

theorem compound_statement_false (p q : Prop) (h : ¬ (p ∧ q)) : ¬ p ∨ ¬ q :=
sorry

end compound_statement_false_l820_82069


namespace log2_bounds_158489_l820_82004

theorem log2_bounds_158489 :
  (2^16 = 65536) ∧ (2^17 = 131072) ∧ (65536 < 158489 ∧ 158489 < 131072) →
  (16 < Real.log 158489 / Real.log 2 ∧ Real.log 158489 / Real.log 2 < 17) ∧ 16 + 17 = 33 :=
by
  intro h
  have h1 : 2^16 = 65536 := h.1
  have h2 : 2^17 = 131072 := h.2.1
  have h3 : 65536 < 158489 := h.2.2.1
  have h4 : 158489 < 131072 := h.2.2.2
  sorry

end log2_bounds_158489_l820_82004


namespace olivia_time_spent_l820_82086

theorem olivia_time_spent :
  ∀ (x : ℕ), 7 * x + 3 = 31 → x = 4 :=
by
  intro x
  intro h
  sorry

end olivia_time_spent_l820_82086


namespace find_guest_sets_l820_82083

-- Definitions based on conditions
def cost_per_guest_set : ℝ := 32.0
def cost_per_master_set : ℝ := 40.0
def num_master_sets : ℕ := 4
def total_cost : ℝ := 224.0

-- The mathematical problem
theorem find_guest_sets (G : ℕ) (total_cost_eq : total_cost = cost_per_guest_set * G + cost_per_master_set * num_master_sets) : G = 2 :=
by
  sorry

end find_guest_sets_l820_82083


namespace eggs_processed_per_day_l820_82012

/-- In a certain egg-processing plant, every egg must be inspected, and is either accepted for processing or rejected. For every 388 eggs accepted for processing, 12 eggs are rejected.

If, on a particular day, 37 additional eggs were accepted, but the overall number of eggs inspected remained the same, the ratio of those accepted to those rejected would be 405 to 3.

Prove that the number of eggs processed per day, given these conditions, is 125763.
-/
theorem eggs_processed_per_day : ∃ (E : ℕ), (∃ (R : ℕ), 38 * R = 3 * (E - 37) ∧  E = 32 * R + E / 33 ) ∧ (E = 125763) :=
sorry

end eggs_processed_per_day_l820_82012


namespace Mark_hours_left_l820_82033

theorem Mark_hours_left (sick_days vacation_days : ℕ) (hours_per_day : ℕ) 
  (h1 : sick_days = 10) (h2 : vacation_days = 10) (h3 : hours_per_day = 8) 
  (used_sick_days : ℕ) (used_vacation_days : ℕ) 
  (h4 : used_sick_days = sick_days / 2) (h5 : used_vacation_days = vacation_days / 2) 
  : (sick_days + vacation_days - used_sick_days - used_vacation_days) * hours_per_day = 80 :=
by
  sorry

end Mark_hours_left_l820_82033


namespace line_tangent_to_ellipse_l820_82037

theorem line_tangent_to_ellipse (k : ℝ) :
  (∃ x : ℝ, 2 * x ^ 2 + 8 * (k * x + 2) ^ 2 = 8 ∧
             ∀ x1 x2 : ℝ, (2 + 8 * k ^ 2) * x1 ^ 2 + 32 * k * x1 + 24 = 0 →
             (2 + 8 * k ^ 2) * x2 ^ 2 + 32 * k * x2 + 24 = 0 → x1 = x2) →
  k^2 = 3 / 4 := by
  sorry

end line_tangent_to_ellipse_l820_82037


namespace linear_function_positive_in_interval_abc_sum_greater_negative_one_l820_82006

-- Problem 1
theorem linear_function_positive_in_interval (f : ℝ → ℝ) (k h m n : ℝ) (hk : k ≠ 0) (hmn : m < n)
  (hf_m : f m > 0) (hf_n : f n > 0) : (∀ x : ℝ, m < x ∧ x < n → f x > 0) :=
sorry

-- Problem 2
theorem abc_sum_greater_negative_one (a b c : ℝ)
  (ha : abs a < 1) (hb : abs b < 1) (hc : abs c < 1) : a * b + b * c + c * a > -1 :=
sorry

end linear_function_positive_in_interval_abc_sum_greater_negative_one_l820_82006


namespace value_of_a_b_c_l820_82010

theorem value_of_a_b_c 
    (a b c : Int)
    (h1 : ∀ x : Int, x^2 + 10*x + 21 = (x + a) * (x + b))
    (h2 : ∀ x : Int, x^2 + 3*x - 88 = (x + b) * (x - c))
    :
    a + b + c = 18 := 
sorry

end value_of_a_b_c_l820_82010


namespace radius_of_semi_circle_l820_82096

-- Given definitions and conditions
def perimeter : ℝ := 33.934511513692634
def pi_approx : ℝ := 3.141592653589793

-- The formula for the perimeter of a semi-circle
def semi_circle_perimeter (r : ℝ) : ℝ := pi_approx * r + 2 * r

-- The theorem we want to prove
theorem radius_of_semi_circle (r : ℝ) (h: semi_circle_perimeter r = perimeter) : r = 6.6 :=
sorry

end radius_of_semi_circle_l820_82096


namespace cost_of_fruits_l820_82032

-- Definitions based on the conditions
variables (x y z : ℝ)

-- Conditions
axiom h1 : 2 * x + y + 4 * z = 6
axiom h2 : 4 * x + 2 * y + 2 * z = 4

-- Question to prove
theorem cost_of_fruits : 4 * x + 2 * y + 5 * z = 8 :=
sorry

end cost_of_fruits_l820_82032


namespace remainder_sum_modulo_l820_82063

theorem remainder_sum_modulo :
  (9156 + 9157 + 9158 + 9159 + 9160) % 9 = 7 :=
by
sorry

end remainder_sum_modulo_l820_82063


namespace equilateral_triangle_l820_82038

theorem equilateral_triangle
  (a b c : ℝ) (α β γ : ℝ) (p R : ℝ)
  (h : (a * Real.cos α + b * Real.cos β + c * Real.cos γ) / (a * Real.sin β + b * Real.sin γ + c * Real.sin α) = p / (9 * R)) :
  a = b ∧ b = c ∧ a = c :=
sorry

end equilateral_triangle_l820_82038


namespace determine_k_l820_82046

theorem determine_k 
  (k : ℝ) 
  (r s : ℝ) 
  (h1 : r + s = -k) 
  (h2 : r * s = 6) 
  (h3 : (r + 5) + (s + 5) = k) : 
  k = 5 := 
by 
  sorry

end determine_k_l820_82046


namespace no_square_divisible_by_six_in_range_l820_82076

theorem no_square_divisible_by_six_in_range : ¬ ∃ y : ℕ, (∃ k : ℕ, y = k * k) ∧ (6 ∣ y) ∧ (50 ≤ y ∧ y ≤ 120) :=
by
  sorry

end no_square_divisible_by_six_in_range_l820_82076


namespace number_of_laborers_in_crew_l820_82075

theorem number_of_laborers_in_crew (present : ℕ) (percentage : ℝ) (total : ℕ) 
    (h1 : present = 70) (h2 : percentage = 44.9 / 100) (h3 : present = percentage * total) : 
    total = 156 := 
sorry

end number_of_laborers_in_crew_l820_82075


namespace Joan_attended_games_l820_82007

def total_games : ℕ := 864
def games_missed_by_Joan : ℕ := 469
def games_attended_by_Joan : ℕ := total_games - games_missed_by_Joan

theorem Joan_attended_games : games_attended_by_Joan = 395 := 
by 
  -- Proof omitted
  sorry

end Joan_attended_games_l820_82007


namespace soccer_uniform_probability_l820_82044

-- Definitions for the conditions of the problem
def colorsSocks : List String := ["red", "blue"]
def colorsShirts : List String := ["red", "blue", "green"]

noncomputable def differentColorConfigurations : Nat :=
  let validConfigs := [("red", "blue"), ("red", "green"), ("blue", "red"), ("blue", "green")]
  validConfigs.length

noncomputable def totalConfigurations : Nat :=
  colorsSocks.length * colorsShirts.length

noncomputable def probabilityDifferentColors : ℚ :=
  (differentColorConfigurations : ℚ) / (totalConfigurations : ℚ)

-- The theorem to prove
theorem soccer_uniform_probability :
  probabilityDifferentColors = 2 / 3 :=
by
  sorry

end soccer_uniform_probability_l820_82044


namespace function_properties_l820_82035

noncomputable def f (x : ℝ) : ℝ := 3^x - 3^(-x)

theorem function_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
by
  sorry

end function_properties_l820_82035


namespace original_prices_sum_l820_82029

theorem original_prices_sum
  (new_price_candy_box : ℝ)
  (new_price_soda_can : ℝ)
  (increase_candy_box : ℝ)
  (increase_soda_can : ℝ)
  (h1 : new_price_candy_box = 10)
  (h2 : new_price_soda_can = 9)
  (h3 : increase_candy_box = 0.25)
  (h4 : increase_soda_can = 0.50) :
  let original_price_candy_box := new_price_candy_box / (1 + increase_candy_box)
  let original_price_soda_can := new_price_soda_can / (1 + increase_soda_can)
  original_price_candy_box + original_price_soda_can = 19 :=
by
  sorry

end original_prices_sum_l820_82029


namespace problem_solution_l820_82041

/-- Define the repeating decimal 0.\overline{49} as a rational number. --/
def rep49 := 7 / 9

/-- Define the repeating decimal 0.\overline{4} as a rational number. --/
def rep4 := 4 / 9

/-- The main theorem stating that 99 times the difference between 
    the repeating decimals 0.\overline{49} and 0.\overline{4} equals 5. --/
theorem problem_solution : 99 * (rep49 - rep4) = 5 := by
  sorry

end problem_solution_l820_82041


namespace arithmetic_example_l820_82026

theorem arithmetic_example : 15 * 30 + 45 * 15 = 1125 := by
  sorry

end arithmetic_example_l820_82026


namespace product_of_solutions_l820_82034

theorem product_of_solutions :
  (∀ y : ℝ, (|y| = 2 * (|y| - 1)) → y = 2 ∨ y = -2) →
  (∀ y1 y2 : ℝ, (y1 = 2 ∧ y2 = -2) → y1 * y2 = -4) :=
by
  intro h
  have h1 := h 2
  have h2 := h (-2)
  sorry

end product_of_solutions_l820_82034


namespace largest_uncovered_squares_l820_82095

theorem largest_uncovered_squares (board_size : ℕ) (total_squares : ℕ) (domino_size : ℕ) 
  (odd_property : ∀ (n : ℕ), n % 2 = 1 → (n - domino_size) % 2 = 1)
  (can_place_more : ∀ (placed_squares odd_squares : ℕ), placed_squares + domino_size ≤ total_squares → odd_squares - domino_size % 2 = 1 → odd_squares ≥ 0)
  : ∃ max_uncovered : ℕ, max_uncovered = 7 := by
  sorry

end largest_uncovered_squares_l820_82095


namespace rectangle_sides_l820_82059

def side_length_square : ℝ := 18
def num_rectangles : ℕ := 5

variable (a b : ℝ)
variables (h1 : 2 * (a + b) = side_length_square) (h2 : 3 * a = side_length_square)

theorem rectangle_sides : a = 6 ∧ b = 3 :=
by {
  sorry
}

end rectangle_sides_l820_82059


namespace sums_ratio_l820_82061

theorem sums_ratio (total_sums : ℕ) (sums_right : ℕ) (sums_wrong: ℕ) (h1 : total_sums = 24) (h2 : sums_right = 8) (h3 : sums_wrong = total_sums - sums_right) :
  sums_wrong / Nat.gcd sums_wrong sums_right = 2 ∧ sums_right / Nat.gcd sums_wrong sums_right = 1 := by
  sorry

end sums_ratio_l820_82061


namespace original_price_l820_82049

theorem original_price (P : ℝ) (h : 0.684 * P = 6800) : P = 10000 :=
by
  sorry

end original_price_l820_82049


namespace whole_process_time_is_9_l820_82023

variable (BleachingTime : ℕ)
variable (DyeingTime : ℕ)

-- Conditions
axiom bleachingTime_is_3 : BleachingTime = 3
axiom dyeingTime_is_twice_bleachingTime : DyeingTime = 2 * BleachingTime

-- Question and Proof Problem
theorem whole_process_time_is_9 (BleachingTime : ℕ) (DyeingTime : ℕ)
  (h1 : BleachingTime = 3) (h2 : DyeingTime = 2 * BleachingTime) : 
  (BleachingTime + DyeingTime) = 9 :=
  by
  sorry

end whole_process_time_is_9_l820_82023


namespace fractional_sides_l820_82062

variable {F : ℕ} -- Number of fractional sides
variable {D : ℕ} -- Number of diagonals

theorem fractional_sides (h1 : D = 2 * F) (h2 : D = F * (F - 3) / 2) : F = 7 :=
by
  sorry

end fractional_sides_l820_82062


namespace nonoverlapping_area_difference_l820_82077

theorem nonoverlapping_area_difference :
  let radius := 3
  let side := 2
  let circle_area := Real.pi * radius^2
  let square_area := side^2
  ∃ (x : ℝ), (circle_area - x) - (square_area - x) = 9 * Real.pi - 4 :=
by
  sorry

end nonoverlapping_area_difference_l820_82077


namespace fraction_remain_same_l820_82002

theorem fraction_remain_same (x y : ℝ) : (2 * x + y) / (3 * x + y) = (2 * (10 * x) + (10 * y)) / (3 * (10 * x) + (10 * y)) :=
by sorry

end fraction_remain_same_l820_82002


namespace largest_positive_integer_not_sum_of_multiple_30_and_composite_l820_82085

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m k : ℕ, 2 ≤ m ∧ 2 ≤ k ∧ n = m * k

noncomputable def largest_not_sum_of_multiple_30_and_composite : ℕ :=
  211

theorem largest_positive_integer_not_sum_of_multiple_30_and_composite {m : ℕ} :
  m = largest_not_sum_of_multiple_30_and_composite ↔ 
  (∀ a b : ℕ, a > 0 ∧ b > 0 ∧ (∃ k : ℕ, (b = k * 30) ∨ is_composite b) → m ≠ 30 * a + b) :=
sorry

end largest_positive_integer_not_sum_of_multiple_30_and_composite_l820_82085


namespace relationship_between_a_b_c_l820_82079

noncomputable def a : ℝ := 2^(4/3)
noncomputable def b : ℝ := 4^(2/5)
noncomputable def c : ℝ := 25^(1/3)

theorem relationship_between_a_b_c : c > a ∧ a > b := 
by
  have ha : a = 2^(4/3) := rfl
  have hb : b = 4^(2/5) := rfl
  have hc : c = 25^(1/3) := rfl

  sorry

end relationship_between_a_b_c_l820_82079


namespace cube_sum_identity_l820_82098

theorem cube_sum_identity (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
by
  sorry

end cube_sum_identity_l820_82098


namespace find_minimum_m_l820_82000

theorem find_minimum_m (m : ℕ) (h1 : 1350 + 36 * m < 2136) (h2 : 1500 + 45 * m ≥ 2365) :
  m = 20 :=
by
  sorry

end find_minimum_m_l820_82000


namespace abs_x_minus_one_sufficient_not_necessary_l820_82056

variable (x : ℝ) -- x is a real number

theorem abs_x_minus_one_sufficient_not_necessary (h : |x - 1| > 2) :
  (x^2 > 1) ∧ (∃ (y : ℝ), x^2 > 1 ∧ |y - 1| ≤ 2) := by
  sorry

end abs_x_minus_one_sufficient_not_necessary_l820_82056


namespace min_value_one_over_a_plus_two_over_b_l820_82074

theorem min_value_one_over_a_plus_two_over_b :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 2 * b = 2) →
  ∃ (min_val : ℝ), min_val = (1 / a + 2 / b) ∧ min_val = 9 / 2 :=
by
  sorry

end min_value_one_over_a_plus_two_over_b_l820_82074


namespace greatest_odd_factors_under_150_l820_82009

theorem greatest_odd_factors_under_150 : ∃ (n : ℕ), n < 150 ∧ ( ∃ (k : ℕ), n = k * k ) ∧ (∀ m : ℕ, m < 150 ∧ ( ∃ (k : ℕ), m = k * k ) → m ≤ 144) :=
by
  sorry

end greatest_odd_factors_under_150_l820_82009


namespace highlights_part_to_whole_relation_l820_82088

/-- A predicate representing different types of statistical graphs. -/
inductive StatGraphType where
  | BarGraph : StatGraphType
  | PieChart : StatGraphType
  | LineGraph : StatGraphType
  | FrequencyDistributionHistogram : StatGraphType

/-- A lemma specifying that the PieChart is the graph type that highlights the relationship between a part and the whole. -/
theorem highlights_part_to_whole_relation (t : StatGraphType) : t = StatGraphType.PieChart :=
  sorry

end highlights_part_to_whole_relation_l820_82088


namespace advanced_purchase_tickets_sold_l820_82073

theorem advanced_purchase_tickets_sold (A D : ℕ) 
  (h1 : A + D = 140)
  (h2 : 8 * A + 14 * D = 1720) : 
  A = 40 :=
by
  sorry

end advanced_purchase_tickets_sold_l820_82073


namespace initial_apples_l820_82064

-- Define the initial conditions
def r : Nat := 14
def s : Nat := 2 * r
def remaining : Nat := 32
def total_removed : Nat := r + s

-- The proof problem: Prove that the initial number of apples is 74
theorem initial_apples : (total_removed + remaining = 74) :=
by
  sorry

end initial_apples_l820_82064


namespace probability_of_selecting_red_books_is_3_div_14_l820_82082

-- Define the conditions
def total_books : ℕ := 8
def red_books : ℕ := 4
def blue_books : ℕ := 4
def books_selected : ℕ := 2

-- Define the calculation of the probability
def probability_red_books_selected : ℚ :=
  let total_outcomes := Nat.choose total_books books_selected
  let favorable_outcomes := Nat.choose red_books books_selected
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

-- State the theorem
theorem probability_of_selecting_red_books_is_3_div_14 :
  probability_red_books_selected = 3 / 14 :=
by
  sorry

end probability_of_selecting_red_books_is_3_div_14_l820_82082


namespace range_of_k_l820_82011

theorem range_of_k :
  ∀ (a k : ℝ) (f : ℝ → ℝ),
    (∀ x, f x = if x ≥ 0 then k^2 * x + a^2 - k else x^2 + (a^2 + 4 * a) * x + (2 - a)^2) →
    (∀ x1 x2 : ℝ, x1 ≠ 0 → x2 ≠ 0 → x1 ≠ x2 → f x1 = f x2 → False) →
    -20 ≤ k ∧ k ≤ -4 :=
by
  sorry

end range_of_k_l820_82011


namespace smallest_three_digit_perfect_square_l820_82019

theorem smallest_three_digit_perfect_square :
  ∀ (a : ℕ), 100 ≤ a ∧ a < 1000 → (∃ (n : ℕ), 1001 * a + 1 = n^2) → a = 183 :=
by
  sorry

end smallest_three_digit_perfect_square_l820_82019


namespace cost_price_of_watch_l820_82031

theorem cost_price_of_watch (C : ℝ) 
  (h1 : ∃ (SP1 SP2 : ℝ), SP1 = 0.54 * C ∧ SP2 = 1.04 * C ∧ SP2 = SP1 + 140) : 
  C = 280 :=
by
  obtain ⟨SP1, SP2, H1, H2, H3⟩ := h1
  sorry

end cost_price_of_watch_l820_82031


namespace invest_today_for_future_value_l820_82018

-- Define the given future value, interest rate, and number of years as constants
def FV : ℝ := 600000
def r : ℝ := 0.04
def n : ℕ := 15
def target : ℝ := 333087.66

-- Define the present value calculation
noncomputable def PV : ℝ := FV / (1 + r)^n

-- State the theorem that PV is approximately equal to the target value
theorem invest_today_for_future_value : PV = target := 
by sorry

end invest_today_for_future_value_l820_82018


namespace quadratic_inequality_solution_l820_82045

theorem quadratic_inequality_solution
  (a b c : ℝ)
  (h1: ∀ x : ℝ, (-1/3 < x ∧ x < 2) → (ax^2 + bx + c) > 0)
  (h2: a < 0):
  ∀ x : ℝ, ((-3 < x ∧ x < 1/2) ↔ (cx^2 + bx + a) < 0) :=
by
  sorry

end quadratic_inequality_solution_l820_82045


namespace range_of_2x_minus_y_l820_82024

variable {x y : ℝ}

theorem range_of_2x_minus_y (h1 : 2 < x) (h2 : x < 4) (h3 : -1 < y) (h4 : y < 3) :
  ∃ (a b : ℝ), (1 < a) ∧ (a < 2 * x - y) ∧ (2 * x - y < b) ∧ (b < 9) :=
by
  sorry

end range_of_2x_minus_y_l820_82024


namespace distinct_real_roots_l820_82008

theorem distinct_real_roots (p : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 2 * |x1| - p = 0) ∧ (x2^2 - 2 * |x2| - p = 0)) → p > -1 :=
by
  intro h
  sorry

end distinct_real_roots_l820_82008


namespace largest_sum_of_ABCD_l820_82078

theorem largest_sum_of_ABCD :
  ∃ (A B C D : ℕ), 10 ≤ A ∧ A < 100 ∧ 10 ≤ B ∧ B < 100 ∧ 10 ≤ C ∧ C < 100 ∧ 10 ≤ D ∧ D < 100 ∧
  B = 3 * C ∧ D = 2 * B - C ∧ A = B + D ∧ A + B + C + D = 204 :=
by
  sorry

end largest_sum_of_ABCD_l820_82078


namespace arithmetic_progression_sum_squares_l820_82092

theorem arithmetic_progression_sum_squares (a1 a2 a3 : ℚ)
  (h1 : a2 = (a1 + a3) / 2)
  (h2 : a1 + a2 + a3 = 2)
  (h3 : a1^2 + a2^2 + a3^2 = 14/9) :
  (a1 = 1/3 ∧ a2 = 2/3 ∧ a3 = 1) ∨ (a1 = 1 ∧ a2 = 2/3 ∧ a3 = 1/3) :=
sorry

end arithmetic_progression_sum_squares_l820_82092


namespace find_x_for_mean_l820_82097

theorem find_x_for_mean 
(x : ℝ) 
(h_mean : (3 + 11 + 7 + 9 + 15 + 13 + 8 + 19 + 17 + 21 + 14 + x) / 12 = 12) : 
x = 7 :=
sorry

end find_x_for_mean_l820_82097


namespace question1_question2_l820_82071

def A (x : ℝ) : Prop := x^2 - 2*x - 3 ≤ 0
def B (m : ℝ) (x : ℝ) : Prop := x^2 - 2*m*x + m^2 - 4 ≤ 0

-- Question 1: If A ∩ B = [1, 3], then m = 3
theorem question1 (m : ℝ) : (∀ x, A x ∧ B m x ↔ (1 ≤ x ∧ x ≤ 3)) → m = 3 :=
sorry

-- Question 2: If A is a subset of the complement of B in ℝ, then m > 5 or m < -3
theorem question2 (m : ℝ) : (∀ x, A x → ¬ B m x) → (m > 5 ∨ m < -3) :=
sorry

end question1_question2_l820_82071


namespace good_pair_exists_l820_82091

theorem good_pair_exists (m : ℕ) : ∃ n : ℕ, n > m ∧ (∃ k1 k2 : ℕ, m * n = k1 * k1 ∧ (m + 1) * (n + 1) = k2 * k2) :=
by
  sorry

end good_pair_exists_l820_82091


namespace range_of_n_l820_82094

noncomputable def parabola (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x + b

variable {a b n y1 y2 : ℝ}

theorem range_of_n (h_a : a > 0) 
  (hA : parabola a b (2*n + 3) = y1) 
  (hB : parabola a b (n - 1) = y2)
  (h_sym : y1 < y2) 
  (h_opposite_sides : (2*n + 3 - 1) * (n - 1 - 1) < 0) :
  -1 < n ∧ n < 0 :=
sorry

end range_of_n_l820_82094


namespace wizard_elixir_combinations_l820_82060

theorem wizard_elixir_combinations :
  let herbs := 4
  let crystals := 6
  let invalid_combinations := 3
  herbs * crystals - invalid_combinations = 21 := 
by
  sorry

end wizard_elixir_combinations_l820_82060


namespace product_of_place_values_l820_82017

theorem product_of_place_values : 
  let place_value_1 := 800000
  let place_value_2 := 80
  let place_value_3 := 0.08
  place_value_1 * place_value_2 * place_value_3 = 5120000 := 
by 
  -- proof will be provided here 
  sorry

end product_of_place_values_l820_82017


namespace present_age_of_son_l820_82067

variable (S M : ℕ)

-- Conditions
def condition1 := M = S + 28
def condition2 := M + 2 = 2 * (S + 2)

-- Theorem to be proven
theorem present_age_of_son : condition1 S M ∧ condition2 S M → S = 26 := by
  sorry

end present_age_of_son_l820_82067


namespace curve_C2_eqn_l820_82001

theorem curve_C2_eqn (p : ℝ) (x y : ℝ) :
  (∃ x y, (x^2 - y^2 = 1) ∧ (y^2 = 2 * p * x) ∧ (2 * p = 3/4)) →
  (y^2 = (3/2) * x) :=
by
  sorry

end curve_C2_eqn_l820_82001


namespace parabola_vertex_l820_82099

theorem parabola_vertex:
  ∀ x: ℝ, ∀ y: ℝ, (y = (1 / 2) * x ^ 2 - 4 * x + 3) → (x = 4 ∧ y = -5) :=
sorry

end parabola_vertex_l820_82099


namespace geometric_seq_20th_term_l820_82066

theorem geometric_seq_20th_term (a r : ℕ)
  (h1 : a * r ^ 4 = 5)
  (h2 : a * r ^ 11 = 1280) :
  a * r ^ 19 = 2621440 :=
sorry

end geometric_seq_20th_term_l820_82066


namespace curves_intersect_exactly_three_points_l820_82013

theorem curves_intersect_exactly_three_points (a : ℝ) :
  (∃! (p : ℝ × ℝ), p.1 ^ 2 + p.2 ^ 2 = a ^ 2 ∧ p.2 = p.1 ^ 2 - a) ↔ a > (1 / 2) :=
by sorry

end curves_intersect_exactly_three_points_l820_82013


namespace find_n_l820_82081

theorem find_n (n : ℤ) (hn : -180 ≤ n ∧ n ≤ 180) (hsin : Real.sin (n * Real.pi / 180) = Real.sin (750 * Real.pi / 180)) :
  n = 30 ∨ n = 150 ∨ n = -30 ∨ n = -150 :=
by
  sorry

end find_n_l820_82081


namespace fill_tank_time_is_18_l820_82055

def rate1 := 1 / 20
def rate2 := 1 / 30
def combined_rate := rate1 + rate2
def effective_rate := (2 / 3) * combined_rate
def T := 1 / effective_rate

theorem fill_tank_time_is_18 : T = 18 := by
  sorry

end fill_tank_time_is_18_l820_82055


namespace units_digit_A_is_1_l820_82016

def units_digit (n : ℕ) : ℕ := n % 10

noncomputable def A : ℕ := 2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) + 1

theorem units_digit_A_is_1 : units_digit A = 1 := by
  sorry

end units_digit_A_is_1_l820_82016


namespace inequality_system_solution_l820_82027

theorem inequality_system_solution (x: ℝ) (h1: 5 * x - 2 < 3 * (x + 2)) (h2: (2 * x - 1) / 3 - (5 * x + 1) / 2 <= 1) : 
  -1 ≤ x ∧ x < 4 :=
sorry

end inequality_system_solution_l820_82027


namespace solution_set_inequality_l820_82022

open Set

variable {a b : ℝ}

/-- Proof Problem Statement -/
theorem solution_set_inequality (h : ∀ x : ℝ, -3 < x ∧ x < -1 ↔ a * x^2 - 1999 * x + b > 0) : 
  ∀ x : ℝ, 1 < x ∧ x < 3 ↔ a * x^2 + 1999 * x + b > 0 :=
sorry

end solution_set_inequality_l820_82022


namespace sector_area_sexagesimal_l820_82089

theorem sector_area_sexagesimal (r : ℝ) (n : ℝ) (α_sex : ℝ) (π : ℝ) (two_pi : ℝ):
  r = 4 →
  n = 6000 →
  α_sex = 625 →
  two_pi = 2 * π →
  (1/2 * (α_sex / n * two_pi) * r^2) = (5 * π) / 3 :=
by
  intros
  sorry

end sector_area_sexagesimal_l820_82089


namespace scientific_notation_correct_l820_82028

-- Define the number to be converted
def number : ℕ := 3790000

-- Define the correct scientific notation representation
def scientific_notation : ℝ := 3.79 * (10 ^ 6)

-- Statement to prove that number equals scientific_notation
theorem scientific_notation_correct :
  number = 3790000 → scientific_notation = 3.79 * (10 ^ 6) :=
by
  sorry

end scientific_notation_correct_l820_82028


namespace marble_remainder_l820_82042

theorem marble_remainder
  (r p : ℕ)
  (h_r : r % 5 = 2)
  (h_p : p % 5 = 4) :
  (r + p) % 5 = 1 :=
by
  sorry

end marble_remainder_l820_82042


namespace cylinder_ratio_l820_82005

theorem cylinder_ratio (h r : ℝ) (h_eq : h = 2 * Real.pi * r) : 
  h / r = 2 * Real.pi := 
by 
  sorry

end cylinder_ratio_l820_82005


namespace president_vice_president_ways_l820_82068

theorem president_vice_president_ways :
  let boys := 14
  let girls := 10
  let total_boys_ways := boys * (boys - 1)
  let total_girls_ways := girls * (girls - 1)
  total_boys_ways + total_girls_ways = 272 := 
by
  sorry

end president_vice_president_ways_l820_82068


namespace simplify_expression_l820_82048

theorem simplify_expression (x : ℝ) : 2 * x * (x - 4) - (2 * x - 3) * (x + 2) = -9 * x + 6 :=
by
  sorry

end simplify_expression_l820_82048


namespace value_added_to_each_number_is_12_l820_82030

theorem value_added_to_each_number_is_12
    (sum_original : ℕ)
    (sum_new : ℕ)
    (n : ℕ)
    (avg_original : ℕ)
    (avg_new : ℕ)
    (value_added : ℕ) :
  (n = 15) →
  (avg_original = 40) →
  (avg_new = 52) →
  (sum_original = n * avg_original) →
  (sum_new = n * avg_new) →
  (value_added = (sum_new - sum_original) / n) →
  value_added = 12 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end value_added_to_each_number_is_12_l820_82030


namespace quadratic_roots_l820_82058

theorem quadratic_roots (x : ℝ) : (x ^ 2 - 3 = 0) → (x = Real.sqrt 3 ∨ x = -Real.sqrt 3) :=
by
  intro h
  sorry

end quadratic_roots_l820_82058


namespace a_and_b_together_30_days_l820_82036

variable (R_a R_b : ℝ)

-- Conditions
axiom condition1 : R_a = 3 * R_b
axiom condition2 : R_a * 40 = (R_a + R_b) * 30

-- Question: prove that a and b together can complete the work in 30 days.
theorem a_and_b_together_30_days (R_a R_b : ℝ) (condition1 : R_a = 3 * R_b) (condition2 : R_a * 40 = (R_a + R_b) * 30) : true :=
by
  sorry

end a_and_b_together_30_days_l820_82036


namespace area_triangle_ABC_l820_82054

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let x1 := A.1
  let y1 := A.2
  let x2 := B.1
  let y2 := B.2
  let x3 := C.1
  let y3 := C.2
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem area_triangle_ABC :
  area_of_triangle (2, 4) (-1, 1) (1, -1) = 6 :=
by
  sorry

end area_triangle_ABC_l820_82054


namespace person6_number_l820_82072

theorem person6_number (a : ℕ → ℕ) (x : ℕ → ℕ) 
  (mod12 : ∀ i, a (i % 12) = a i)
  (h5 : x 5 = 5)
  (h6 : x 6 = 8)
  (h7 : x 7 = 11) 
  (h_avg : ∀ i, x i = (a (i-1) + a (i+1)) / 2) : 
  a 6 = 6 := sorry

end person6_number_l820_82072


namespace eggs_per_chicken_per_week_l820_82020

-- Define the conditions
def chickens : ℕ := 10
def price_per_dozen : ℕ := 2  -- in dollars
def earnings_in_2_weeks : ℕ := 20  -- in dollars
def weeks : ℕ := 2
def eggs_per_dozen : ℕ := 12

-- Define the question as a theorem to be proved
theorem eggs_per_chicken_per_week : 
  (earnings_in_2_weeks / price_per_dozen) * eggs_per_dozen / (chickens * weeks) = 6 :=
by
  -- proof steps
  sorry

end eggs_per_chicken_per_week_l820_82020


namespace three_integers_product_sum_l820_82065

theorem three_integers_product_sum (a b c : ℤ) (h : a * b * c = -5) :
    a + b + c = 5 ∨ a + b + c = -3 ∨ a + b + c = -7 :=
sorry

end three_integers_product_sum_l820_82065


namespace original_number_l820_82040

theorem original_number (x y : ℝ) (h1 : 10 * x + 22 * y = 780) (h2 : y = 34) : x + y = 37.2 :=
sorry

end original_number_l820_82040


namespace find_d_l820_82084

theorem find_d (c : ℕ) (d : ℕ) : 
  (∀ n : ℕ, c = 3 ∧ ∀ k : ℕ, k ≠ 30 → ((1 : ℚ) * (29 / 30) * (28 / 30) = 203 / 225) → d = 203) := 
by
  intros
  sorry

end find_d_l820_82084


namespace number_of_trees_is_eleven_l820_82047

variables (N : ℕ)

-- Conditions
def Anya (N : ℕ) := N = 15
def Borya (N : ℕ) := 11 ∣ N
def Vera (N : ℕ) := N < 25
def Gena (N : ℕ) := 22 ∣ N

axiom OneBoyOneGirlTruth :
  (∃ (b : Prop) (g : Prop),
    (b ∨ ¬ b) ∧ (g ∨ ¬ g) ∧
    ((b = (Borya N ∨ Gena N)) ∧ (g = (Anya N ∨ Vera N)) ∧
     (b ↔ ¬g) ∧
     ((Anya N ∨ ¬Vera N) ∨ (¬Anya N ∨ Vera N)) ∧
     (Anya N = (N = 15)) ∧
     (Borya N = (11 ∣ N)) ∧
     (Vera N = (N < 25)) ∧
     (Gena N = (22 ∣ N))))

theorem number_of_trees_is_eleven: N = 11 :=
sorry

end number_of_trees_is_eleven_l820_82047


namespace ascending_order_conversion_l820_82015

def convert_base (num : Nat) (base : Nat) : Nat :=
  match num with
  | 0 => 0
  | _ => (num / 10) * base + (num % 10)

theorem ascending_order_conversion :
  let num16 := 12
  let num7 := 25
  let num4 := 33
  let base16 := 16
  let base7 := 7
  let base4 := 4
  convert_base num4 base4 < convert_base num16 base16 ∧ 
  convert_base num16 base16 < convert_base num7 base7 :=
by
  -- Here would be the proof, but we skip it
  sorry

end ascending_order_conversion_l820_82015


namespace at_least_two_solutions_l820_82025

theorem at_least_two_solutions (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  (∃ x, (x - a) * (x - b) = x - c) ∨ (∃ x, (x - b) * (x - c) = x - a) ∨ (∃ x, (x - c) * (x - a) = x - b) ∨
    (((x - a) * (x - b) = x - c) ∧ ((x - b) * (x - c) = x - a)) ∨ 
    (((x - b) * (x + c) = x - a) ∧ ((x - c) * (x - a) = x - b)) ∨ 
    (((x - c) * (x - a) = x - b) ∧ ((x - a) * (x - b) = x - c)) :=
sorry

end at_least_two_solutions_l820_82025


namespace factor_quadratic_expression_l820_82039

theorem factor_quadratic_expression (a b : ℤ) :
  (∃ a b : ℤ, (5 * a + 5 * b = -125) ∧ (a * b = -100) → (a + b = -25)) → (25 * x^2 - 125 * x - 100 = (5 * x + a) * (5 * x + b)) := 
by
  sorry

end factor_quadratic_expression_l820_82039


namespace statement_A_l820_82093

theorem statement_A (x : ℝ) (h : x > 1) : x^2 > x := 
by
  sorry

end statement_A_l820_82093


namespace bags_total_on_next_day_l820_82052

def bags_on_monday : ℕ := 7
def additional_bags : ℕ := 5
def bags_on_next_day : ℕ := bags_on_monday + additional_bags

theorem bags_total_on_next_day : bags_on_next_day = 12 := by
  unfold bags_on_next_day
  unfold bags_on_monday
  unfold additional_bags
  sorry

end bags_total_on_next_day_l820_82052


namespace shaded_area_of_hexagon_with_semicircles_l820_82070

theorem shaded_area_of_hexagon_with_semicircles :
  let s := 3
  let r := 3 / 2
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let semicircle_area := 3 * (1/2 * Real.pi * r^2)
  let shaded_area := hexagon_area - semicircle_area
  shaded_area = 13.5 * Real.sqrt 3 - 27 * Real.pi / 8 :=
by
  sorry

end shaded_area_of_hexagon_with_semicircles_l820_82070


namespace parallelogram_coordinates_l820_82053

/-- Given points A, B, and C, prove the coordinates of point D for the parallelogram -/
theorem parallelogram_coordinates (A B C: (ℝ × ℝ)) 
  (hA : A = (3, 7)) 
  (hB : B = (4, 6))
  (hC : C = (1, -2)) :
  D = (0, -1) ∨ D = (2, -3) ∨ D = (6, 15) :=
sorry

end parallelogram_coordinates_l820_82053


namespace eq_of_frac_eq_and_neq_neg_one_l820_82090

theorem eq_of_frac_eq_and_neq_neg_one
  (a b c d : ℝ)
  (h : (a + b) / (c + d) = (b + c) / (a + d))
  (h_neq : (a + b) / (c + d) ≠ -1) :
  a = c :=
sorry

end eq_of_frac_eq_and_neq_neg_one_l820_82090


namespace bus_empty_seats_l820_82014

theorem bus_empty_seats : 
  let initial_seats : ℕ := 23 * 4
  let people_at_start : ℕ := 16
  let first_board : ℕ := 15
  let first_alight : ℕ := 3
  let second_board : ℕ := 17
  let second_alight : ℕ := 10
  let seats_after_init : ℕ := initial_seats - people_at_start
  let seats_after_first : ℕ := seats_after_init - (first_board - first_alight)
  let seats_after_second : ℕ := seats_after_first - (second_board - second_alight)
  seats_after_second = 57 :=
by
  sorry

end bus_empty_seats_l820_82014


namespace original_number_of_men_l820_82057

theorem original_number_of_men 
  (x : ℕ) 
  (H1 : x * 15 = (x - 8) * 18) : 
  x = 48 := 
sorry

end original_number_of_men_l820_82057


namespace solve_quadratic_eq_l820_82087

theorem solve_quadratic_eq (x : ℝ) : x^2 + 8 * x = 9 ↔ x = -9 ∨ x = 1 :=
by
  sorry

end solve_quadratic_eq_l820_82087


namespace next_in_sequence_is_80_l820_82043

def seq (n : ℕ) : ℕ := n^2 - 1

theorem next_in_sequence_is_80 :
  seq 9 = 80 :=
by
  sorry

end next_in_sequence_is_80_l820_82043


namespace workers_planted_33_walnut_trees_l820_82051

def initial_walnut_trees : ℕ := 22
def total_walnut_trees_after_planting : ℕ := 55
def walnut_trees_planted (initial : ℕ) (total : ℕ) : ℕ := total - initial

theorem workers_planted_33_walnut_trees :
  walnut_trees_planted initial_walnut_trees total_walnut_trees_after_planting = 33 :=
by
  unfold walnut_trees_planted
  rfl

end workers_planted_33_walnut_trees_l820_82051


namespace box_contents_l820_82050

-- Definitions for the boxes and balls
inductive Ball
| Black | White | Green

-- Define the labels on each box
def label_box1 := "white"
def label_box2 := "black"
def label_box3 := "white or green"

-- Conditions based on the problem
def box1_label := label_box1
def box2_label := label_box2
def box3_label := label_box3

-- Statement of the problem
theorem box_contents (b1 b2 b3 : Ball) 
  (h1 : b1 ≠ Ball.White) 
  (h2 : b2 ≠ Ball.Black) 
  (h3 : b3 = Ball.Black) 
  (h4 : ∀ (x y z : Ball), x ≠ y ∧ y ≠ z ∧ z ≠ x → 
        (x = b1 ∨ y = b1 ∨ z = b1) ∧
        (x = b2 ∨ y = b2 ∨ z = b2) ∧
        (x = b3 ∨ y = b3 ∨ z = b3)) : 
  b1 = Ball.Green ∧ b2 = Ball.White ∧ b3 = Ball.Black :=
sorry

end box_contents_l820_82050


namespace positive_difference_of_complementary_angles_in_ratio_5_to_4_l820_82021

-- Definitions for given conditions
def is_complementary (a b : ℝ) : Prop :=
  a + b = 90

def ratio_5_to_4 (a b : ℝ) : Prop :=
  ∃ x : ℝ, a = 5 * x ∧ b = 4 * x

-- Theorem to prove the measure of their positive difference is 10 degrees
theorem positive_difference_of_complementary_angles_in_ratio_5_to_4
  {a b : ℝ} (h_complementary : is_complementary a b) (h_ratio : ratio_5_to_4 a b) :
  abs (a - b) = 10 :=
by 
  sorry

end positive_difference_of_complementary_angles_in_ratio_5_to_4_l820_82021


namespace increasing_iff_a_gt_neg1_l820_82003

noncomputable def increasing_function_condition (a : ℝ) (b : ℝ) (x : ℝ) : Prop :=
  let y := (a + 1) * x + b
  a > -1

theorem increasing_iff_a_gt_neg1 (a : ℝ) (b : ℝ) : (∀ x : ℝ, (a + 1) > 0) ↔ a > -1 :=
by
  sorry

end increasing_iff_a_gt_neg1_l820_82003


namespace steers_cows_unique_solution_l820_82080

-- Definition of the problem
def steers_and_cows_problem (s c : ℕ) : Prop :=
  25 * s + 26 * c = 1000 ∧ s > 0 ∧ c > 0

-- The theorem statement to be proved
theorem steers_cows_unique_solution :
  ∃! (s c : ℕ), steers_and_cows_problem s c ∧ c > s :=
sorry

end steers_cows_unique_solution_l820_82080
