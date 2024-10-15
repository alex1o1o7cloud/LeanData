import Mathlib

namespace NUMINAMATH_GPT_students_per_bench_l342_34245

-- Definitions based on conditions
def num_male_students : ℕ := 29
def num_female_students : ℕ := 4 * num_male_students
def num_benches : ℕ := 29
def total_students : ℕ := num_male_students + num_female_students

-- Theorem to prove
theorem students_per_bench : total_students / num_benches = 5 := by
  sorry

end NUMINAMATH_GPT_students_per_bench_l342_34245


namespace NUMINAMATH_GPT_brendan_weekly_capacity_l342_34265

/-- Brendan can cut 8 yards of grass per day on flat terrain under normal weather conditions. Bought a lawnmower that improved his cutting speed by 50 percent on flat terrain. On uneven terrain, his speed is reduced by 35 percent. Rain reduces his cutting capacity by 20 percent. Extreme heat reduces his cutting capacity by 10 percent. The conditions for each day of the week are given and we want to prove that the total yards Brendan can cut in a week is 65.46 yards.
  Monday: Flat terrain, normal weather
  Tuesday: Flat terrain, rain
  Wednesday: Uneven terrain, normal weather
  Thursday: Flat terrain, extreme heat
  Friday: Uneven terrain, rain
  Saturday: Flat terrain, normal weather
  Sunday: Uneven terrain, extreme heat
-/
def brendan_cutting_capacity : ℝ :=
  let base_capacity := 8.0
  let flat_terrain_boost := 1.5
  let uneven_terrain_penalty := 0.65
  let rain_penalty := 0.8
  let extreme_heat_penalty := 0.9
  let monday_capacity := base_capacity * flat_terrain_boost
  let tuesday_capacity := monday_capacity * rain_penalty
  let wednesday_capacity := monday_capacity * uneven_terrain_penalty
  let thursday_capacity := monday_capacity * extreme_heat_penalty
  let friday_capacity := wednesday_capacity * rain_penalty
  let saturday_capacity := monday_capacity
  let sunday_capacity := wednesday_capacity * extreme_heat_penalty
  monday_capacity + tuesday_capacity + wednesday_capacity + thursday_capacity + friday_capacity + saturday_capacity + sunday_capacity

theorem brendan_weekly_capacity : brendan_cutting_capacity = 65.46 := 
by 
  sorry

end NUMINAMATH_GPT_brendan_weekly_capacity_l342_34265


namespace NUMINAMATH_GPT_find_prime_pair_l342_34289
open Int

theorem find_prime_pair :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ ∃ (p : ℕ), Prime p ∧ p = a * b^2 / (a + b) ∧ (a, b) = (6, 2) := by
  sorry

end NUMINAMATH_GPT_find_prime_pair_l342_34289


namespace NUMINAMATH_GPT_karen_packs_cookies_l342_34256

-- Conditions stated as definitions
def school_days := 5
def peanut_butter_days := 2
def ham_sandwich_days := school_days - peanut_butter_days
def cake_days := 1
def probability_ham_and_cake := 0.12

-- Lean theorem statement
theorem karen_packs_cookies : 
  (school_days - cake_days - peanut_butter_days) = 2 :=
by
  sorry

end NUMINAMATH_GPT_karen_packs_cookies_l342_34256


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l342_34213

theorem simplify_and_evaluate_expression (x y : ℝ) (hx : x = -1) (hy : y = -1) :
  (5 * x ^ 2 - 2 * (3 * y ^ 2 + 6 * x) + (2 * y ^ 2 - 5 * x ^ 2)) = 8 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l342_34213


namespace NUMINAMATH_GPT_number_of_students_l342_34241

def candiesPerStudent : ℕ := 2
def totalCandies : ℕ := 18
def expectedStudents : ℕ := 9

theorem number_of_students :
  totalCandies / candiesPerStudent = expectedStudents :=
sorry

end NUMINAMATH_GPT_number_of_students_l342_34241


namespace NUMINAMATH_GPT_cost_of_paving_floor_l342_34204

-- Definitions of the constants
def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sq_meter : ℝ := 400

-- Definitions of the calculated area and cost
def area : ℝ := length * width
def cost : ℝ := area * rate_per_sq_meter

-- Statement to prove
theorem cost_of_paving_floor : cost = 8250 := by
  sorry

end NUMINAMATH_GPT_cost_of_paving_floor_l342_34204


namespace NUMINAMATH_GPT_igors_number_l342_34220

-- Define the initial lineup of players
def initialLineup : List ℕ := [9, 7, 11, 10, 6, 8, 5, 4, 1]

-- Define the condition for a player running to the locker room
def runsToLockRoom (n : ℕ) (left : Option ℕ) (right : Option ℕ) : Prop :=
  match left, right with
  | some l, some r => n < l ∨ n < r
  | some l, none   => n < l
  | none, some r   => n < r
  | none, none     => False

-- Define the process of players running to the locker room iteratively
def runProcess : List ℕ → List ℕ := 
  sorry   -- Implementation of the run process is skipped

-- Define the remaining players after repeated commands until 3 players are left
def remainingPlayers (lineup : List ℕ) : List ℕ :=
  sorry  -- Implementation to find the remaining players is skipped

-- Statement of the theorem
theorem igors_number (afterIgorRanOff : List ℕ := remainingPlayers initialLineup)
  (finalLineup : List ℕ := [9, 11, 10]) :
  ∃ n, n ∈ initialLineup ∧ ¬(n ∈ finalLineup) ∧ afterIgorRanOff.length = 3 → n = 5 :=
  sorry

end NUMINAMATH_GPT_igors_number_l342_34220


namespace NUMINAMATH_GPT_smallest_number_greater_than_500000_has_56_positive_factors_l342_34238

/-- Let n be the smallest number greater than 500,000 
    that is the product of the first four terms of both
    an arithmetic sequence and a geometric sequence.
    Prove that n has 56 positive factors. -/
theorem smallest_number_greater_than_500000_has_56_positive_factors :
  ∃ n : ℕ,
    (500000 < n) ∧
    (∀ a d b r, a > 0 → d > 0 → b > 0 → r > 0 →
      n = (a * (a + d) * (a + 2 * d) * (a + 3 * d)) ∧
          n = (b * (b * r) * (b * r^2) * (b * r^3))) ∧
    (n.factors.length = 56) :=
by sorry

end NUMINAMATH_GPT_smallest_number_greater_than_500000_has_56_positive_factors_l342_34238


namespace NUMINAMATH_GPT_comparison_of_prices_l342_34228

theorem comparison_of_prices:
  ∀ (x y : ℝ), (6 * x + 3 * y > 24) → (4 * x + 5 * y < 22) → (2 * x > 3 * y) :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_comparison_of_prices_l342_34228


namespace NUMINAMATH_GPT_base9_4318_is_base10_3176_l342_34219

def base9_to_base10 (n : Nat) : Nat :=
  let d₀ := (n % 10) * 9^0
  let d₁ := ((n / 10) % 10) * 9^1
  let d₂ := ((n / 100) % 10) * 9^2
  let d₃ := ((n / 1000) % 10) * 9^3
  d₀ + d₁ + d₂ + d₃

theorem base9_4318_is_base10_3176 :
  base9_to_base10 4318 = 3176 :=
by
  sorry

end NUMINAMATH_GPT_base9_4318_is_base10_3176_l342_34219


namespace NUMINAMATH_GPT_impossible_all_matches_outside_own_country_l342_34253

theorem impossible_all_matches_outside_own_country (n : ℕ) (h_teams : n = 16) : 
  ¬ ∀ (T : Fin n → Fin n → Prop), (∀ i j, i ≠ j → T i j) ∧ 
  (∀ i, ∀ j, i ≠ j → T i j → T j i) ∧ 
  (∀ i, T i i = false) → 
  ∀ i, ∃ j, T i j ∧ i ≠ j :=
by
  intro H
  sorry

end NUMINAMATH_GPT_impossible_all_matches_outside_own_country_l342_34253


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l342_34242

theorem necessary_but_not_sufficient (x y : ℝ) :
  (x = 0) → (x^2 + y^2 = 0) ↔ (x = 0 ∧ y = 0) :=
by sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l342_34242


namespace NUMINAMATH_GPT_possible_value_of_a_l342_34247

variable {a b x : ℝ}

theorem possible_value_of_a (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * x^3) (h3 : a - b = 3 * x) :
  a = 3 * x :=
sorry

end NUMINAMATH_GPT_possible_value_of_a_l342_34247


namespace NUMINAMATH_GPT_area_enclosed_by_region_l342_34217

theorem area_enclosed_by_region :
  (∃ (x y : ℝ), x^2 + y^2 - 4*x + 6*y - 3 = 0) → 
  (∃ r : ℝ, r = 4 ∧ area = (π * r^2)) :=
by
  -- Starting proof setup
  sorry

end NUMINAMATH_GPT_area_enclosed_by_region_l342_34217


namespace NUMINAMATH_GPT_log_comparison_l342_34271

theorem log_comparison (a b : ℝ) (h1 : 0 < a) (h2 : a < e) (h3 : 0 < b) (h4 : b < e) (h5 : a < b) :
  a * Real.log b > b * Real.log a := sorry

end NUMINAMATH_GPT_log_comparison_l342_34271


namespace NUMINAMATH_GPT_correctly_calculated_value_l342_34250

theorem correctly_calculated_value (n : ℕ) (h : 5 * n = 30) : n / 6 = 1 :=
sorry

end NUMINAMATH_GPT_correctly_calculated_value_l342_34250


namespace NUMINAMATH_GPT_more_supermarkets_in_us_l342_34274

-- Definitions based on conditions
def total_supermarkets : ℕ := 84
def us_supermarkets : ℕ := 47
def canada_supermarkets : ℕ := total_supermarkets - us_supermarkets

-- Prove that the number of more FGH supermarkets in the US than in Canada is 10
theorem more_supermarkets_in_us : us_supermarkets - canada_supermarkets = 10 :=
by
  -- adding 'sorry' as the proof
  sorry

end NUMINAMATH_GPT_more_supermarkets_in_us_l342_34274


namespace NUMINAMATH_GPT_avg_of_numbers_l342_34255

theorem avg_of_numbers (a b c d : ℕ) (avg : ℕ) (h₁ : a = 6) (h₂ : b = 16) (h₃ : c = 8) (h₄ : d = 22) (h₅ : avg = 13) :
  (a + b + c + d) / 4 = avg := by
  -- Proof here
  sorry

end NUMINAMATH_GPT_avg_of_numbers_l342_34255


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l342_34273

theorem necessary_but_not_sufficient (x y : ℕ) : x + y = 3 → (x = 1 ∧ y = 2) ↔ (¬ (x = 0 ∧ y = 3)) := by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l342_34273


namespace NUMINAMATH_GPT_inv_func_eval_l342_34254

theorem inv_func_eval (a : ℝ) (h : 8^(1/3) = a) : (fun y => (Real.log y / Real.log 8)) (a + 2) = 2/3 :=
by
  sorry

end NUMINAMATH_GPT_inv_func_eval_l342_34254


namespace NUMINAMATH_GPT_total_boxes_packed_l342_34232

-- Definitions of the conditions
def initial_boxes : ℕ := 400
def food_value_per_box : ℕ := 80
def supplies_value_per_box : ℕ := 165
def donor_multiplier : ℕ := 4

-- Total value of one box
def total_value_per_box : ℕ := food_value_per_box + supplies_value_per_box

-- Initial spending
def initial_spending : ℕ := initial_boxes * total_value_per_box

-- Donation amount
def donation_amount : ℕ := donor_multiplier * initial_spending

-- Number of additional boxes packed with the donation
def additional_boxes : ℕ := donation_amount / total_value_per_box

-- Total number of boxes packed
def total_boxes : ℕ := initial_boxes + additional_boxes

-- Statement to be proven
theorem total_boxes_packed : total_boxes = 2000 := by
  -- Proof for this theorem goes here...
  -- The proof is omitted in this statement as requested.
  sorry

end NUMINAMATH_GPT_total_boxes_packed_l342_34232


namespace NUMINAMATH_GPT_tank_filled_in_96_minutes_l342_34214

-- conditions
def pipeA_fill_time : ℝ := 6
def pipeB_empty_time : ℝ := 24
def time_with_both_pipes_open : ℝ := 96

-- rate computations and final proof
noncomputable def pipeA_fill_rate : ℝ := 1 / pipeA_fill_time
noncomputable def pipeB_empty_rate : ℝ := 1 / pipeB_empty_time
noncomputable def net_fill_rate : ℝ := pipeA_fill_rate - pipeB_empty_rate
noncomputable def tank_filled_in_time_with_both : ℝ := time_with_both_pipes_open * net_fill_rate

theorem tank_filled_in_96_minutes (HA : pipeA_fill_time = 6) (HB : pipeB_empty_time = 24)
  (HT : time_with_both_pipes_open = 96) : tank_filled_in_time_with_both = 1 :=
by
  sorry

end NUMINAMATH_GPT_tank_filled_in_96_minutes_l342_34214


namespace NUMINAMATH_GPT_rice_mixture_price_l342_34269

-- Defining the costs per kg for each type of rice
def rice_cost1 : ℝ := 16
def rice_cost2 : ℝ := 24

-- Defining the given ratio
def mixing_ratio : ℝ := 3

-- Main theorem stating the problem
theorem rice_mixture_price
  (x : ℝ)  -- The common measure of quantity in the ratio
  (h1 : 3 * x * rice_cost1 + x * rice_cost2 = 72 * x)
  (h2 : 3 * x + x = 4 * x) :
  (3 * x * rice_cost1 + x * rice_cost2) / (3 * x + x) = 18 :=
by
  sorry

end NUMINAMATH_GPT_rice_mixture_price_l342_34269


namespace NUMINAMATH_GPT_count_integers_divis_by_8_l342_34208

theorem count_integers_divis_by_8 : 
  ∃ k : ℕ, k = 49 ∧ ∀ n : ℕ, 2 ≤ n ∧ n ≤ 80 → (∃ m : ℤ, (n-1) * n * (n+1) = 8 * m) ↔ (∃ m : ℕ, m ≤ k) :=
by 
  sorry

end NUMINAMATH_GPT_count_integers_divis_by_8_l342_34208


namespace NUMINAMATH_GPT_find_values_of_a_l342_34280

-- Definitions for sets A and B
def A : Set ℝ := {x | x^2 - x - 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 6 = 0}

-- The theorem we want to prove
theorem find_values_of_a (a : ℝ) : (A ∪ B a = A) ↔ (a = -6 ∨ a = 0 ∨ a = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_values_of_a_l342_34280


namespace NUMINAMATH_GPT_total_students_calculation_l342_34239

variable (x : ℕ)
variable (girls_jelly_beans boys_jelly_beans total_jelly_beans : ℕ)
variable (total_students : ℕ)
variable (remaining_jelly_beans : ℕ)

-- Defining the number of boys as per the problem's conditions
def boys (x : ℕ) : ℕ := 2 * x + 3

-- Defining the jelly beans given to girls
def jelly_beans_given_to_girls (x girls_jelly_beans : ℕ) : Prop :=
  girls_jelly_beans = 2 * x * x

-- Defining the jelly beans given to boys
def jelly_beans_given_to_boys (x boys_jelly_beans : ℕ) : Prop :=
  boys_jelly_beans = 3 * (2 * x + 3) * (2 * x + 3)

-- Defining the total jelly beans given out
def total_jelly_beans_given_out (girls_jelly_beans boys_jelly_beans total_jelly_beans : ℕ) : Prop :=
  total_jelly_beans = girls_jelly_beans + boys_jelly_beans

-- Defining the total number of students
def total_students_in_class (x total_students : ℕ) : Prop :=
  total_students = x + boys x

-- Proving that the total number of students is 18 under given conditions
theorem total_students_calculation (h1 : jelly_beans_given_to_girls x girls_jelly_beans)
                                   (h2 : jelly_beans_given_to_boys x boys_jelly_beans)
                                   (h3 : total_jelly_beans_given_out girls_jelly_beans boys_jelly_beans total_jelly_beans)
                                   (h4 : total_jelly_beans - remaining_jelly_beans = 642)
                                   (h5 : remaining_jelly_beans = 3) :
                                   total_students = 18 :=
by
  sorry

end NUMINAMATH_GPT_total_students_calculation_l342_34239


namespace NUMINAMATH_GPT_exists_six_digit_number_l342_34243

theorem exists_six_digit_number : ∃ (n : ℕ), 100000 ≤ n ∧ n < 1000000 ∧ (∃ (x y : ℕ), n = 1000 * x + y ∧ 0 ≤ x ∧ x < 1000 ∧ 0 ≤ y ∧ y < 1000 ∧ 6 * n = 1000 * y + x) :=
by
  sorry

end NUMINAMATH_GPT_exists_six_digit_number_l342_34243


namespace NUMINAMATH_GPT_find_monthly_salary_l342_34210

variable (S : ℝ)

theorem find_monthly_salary
  (h1 : 0.20 * S - 0.20 * (0.20 * S) = 220) :
  S = 1375 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_monthly_salary_l342_34210


namespace NUMINAMATH_GPT_length_of_square_side_is_correct_l342_34235

noncomputable def length_of_square_side : ℚ :=
  let PQ : ℚ := 7
  let QR : ℚ := 24
  let hypotenuse := (PQ^2 + QR^2).sqrt
  (25 * 175) / (24 * 32)

theorem length_of_square_side_is_correct :
  length_of_square_side = 4375 / 768 := 
by 
  sorry

end NUMINAMATH_GPT_length_of_square_side_is_correct_l342_34235


namespace NUMINAMATH_GPT_remaining_amount_is_12_l342_34249

-- Define initial amount and amount spent
def initial_amount : ℕ := 90
def amount_spent : ℕ := 78

-- Define the remaining amount after spending
def remaining_amount : ℕ := initial_amount - amount_spent

-- Theorem asserting the remaining amount is 12
theorem remaining_amount_is_12 : remaining_amount = 12 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_remaining_amount_is_12_l342_34249


namespace NUMINAMATH_GPT_f_even_l342_34258

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem f_even : ∀ x : ℝ, f x = f (-x) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_f_even_l342_34258


namespace NUMINAMATH_GPT_find_m_value_l342_34297

theorem find_m_value (m : ℤ) : (x^2 + m * x - 35 = (x - 7) * (x + 5)) → m = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l342_34297


namespace NUMINAMATH_GPT_last_number_is_four_l342_34252

theorem last_number_is_four (a b c d e last_number : ℕ) (h_counts : a = 6 ∧ b = 12 ∧ c = 1 ∧ d = 12 ∧ e = 7)
    (h_mean : (a + b + c + d + e + last_number) / 6 = 7) : last_number = 4 := 
sorry

end NUMINAMATH_GPT_last_number_is_four_l342_34252


namespace NUMINAMATH_GPT_max_value_2x_plus_y_l342_34293

def max_poly_value : ℝ :=
  sorry

theorem max_value_2x_plus_y (x y : ℝ) (h1 : x + 2 * y ≤ 3) (h2 : 0 ≤ x) (h3 : 0 ≤ y) : 
  2 * x + y ≤ 6 :=
sorry

example (x y : ℝ) (h1 : x + 2 * y ≤ 3) (h2 : 0 ≤ x) (h3 : 0 ≤ y) : 2 * x + y = 6 
  ↔ x = 3 ∧ y = 0 :=
by exact sorry

end NUMINAMATH_GPT_max_value_2x_plus_y_l342_34293


namespace NUMINAMATH_GPT_price_reduction_after_markup_l342_34230

theorem price_reduction_after_markup (p : ℝ) (x : ℝ) (h₁ : 0 < p) (h₂ : 0 ≤ x ∧ x < 1) :
  (1.25 : ℝ) * (1 - x) = 1 → x = 0.20 := by
  sorry

end NUMINAMATH_GPT_price_reduction_after_markup_l342_34230


namespace NUMINAMATH_GPT_ms_hatcher_total_students_l342_34205

theorem ms_hatcher_total_students :
  let third_graders := 20
  let fourth_graders := 2 * third_graders
  let fifth_graders := third_graders / 2
  third_graders + fourth_graders + fifth_graders = 70 :=
by 
  let third_graders := 20
  let fourth_graders := 2 * third_graders
  let fifth_graders := third_graders / 2
  show third_graders + fourth_graders + fifth_graders = 70
  sorry

end NUMINAMATH_GPT_ms_hatcher_total_students_l342_34205


namespace NUMINAMATH_GPT_part1_part2_part3_l342_34262

-- Problem Definitions
def air_conditioner_cost (A B : ℕ → ℕ) :=
  A 3 + B 2 = 39000 ∧ 4 * A 1 - 5 * B 1 = 6000

def possible_schemes (A B : ℕ → ℕ) :=
  ∀ a b, a ≥ b / 2 ∧ 9000 * a + 6000 * b ≤ 217000 ∧ a + b = 30

def minimize_cost (A B : ℕ → ℕ) :=
  ∃ a, (a = 10 ∧ 9000 * a + 6000 * (30 - a) = 210000) ∧
  ∀ b, b ≥ 10 → b ≤ 12 → 9000 * b + 6000 * (30 - b) ≥ 210000

-- Theorem Statements
theorem part1 (A B : ℕ → ℕ) : air_conditioner_cost A B → A 1 = 9000 ∧ B 1 = 6000 :=
by sorry

theorem part2 (A B : ℕ → ℕ) : air_conditioner_cost A B →
  possible_schemes A B :=
by sorry

theorem part3 (A B : ℕ → ℕ) : air_conditioner_cost A B ∧ possible_schemes A B →
  minimize_cost A B :=
by sorry

end NUMINAMATH_GPT_part1_part2_part3_l342_34262


namespace NUMINAMATH_GPT_no_groups_of_six_l342_34291

theorem no_groups_of_six (x y z : ℕ) 
  (h1 : (2 * x + 6 * y + 10 * z) / (x + y + z) = 5)
  (h2 : (2 * x + 30 * y + 90 * z) / (2 * x + 6 * y + 10 * z) = 7) : 
  y = 0 := 
sorry

end NUMINAMATH_GPT_no_groups_of_six_l342_34291


namespace NUMINAMATH_GPT_baker_price_l342_34259

theorem baker_price
  (P : ℝ)
  (h1 : 8 * P = 320)
  (h2 : 10 * (0.80 * P) = 320)
  : P = 40 := sorry

end NUMINAMATH_GPT_baker_price_l342_34259


namespace NUMINAMATH_GPT_silvia_escalator_time_l342_34285

noncomputable def total_time_standing (v s : ℝ) : ℝ := 
  let d := 80 * v
  d / s

theorem silvia_escalator_time (v s t : ℝ) (h1 : 80 * v = 28 * (v + s)) (h2 : t = total_time_standing v s) : 
  t = 43 := by
  sorry

end NUMINAMATH_GPT_silvia_escalator_time_l342_34285


namespace NUMINAMATH_GPT_evaluate_fraction_sum_squared_l342_34248

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6

theorem evaluate_fraction_sum_squared :
  ( (1 / a + 1 / b + 1 / c + 1 / d)^2 = (11 + 2 * Real.sqrt 30) / 9 ) := 
by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_sum_squared_l342_34248


namespace NUMINAMATH_GPT_moles_of_NaHSO4_l342_34267

def react_eq (naoh h2so4 nahso4 h2o : ℕ) : Prop :=
  naoh + h2so4 = nahso4 + h2o

theorem moles_of_NaHSO4
  (naoh h2so4 : ℕ)
  (h : 2 = naoh ∧ 2 = h2so4)
  (react : react_eq naoh h2so4 2 2):
  2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_NaHSO4_l342_34267


namespace NUMINAMATH_GPT_unique_triple_sum_l342_34206

theorem unique_triple_sum :
  ∃ (a b c : ℕ), 
    (10 ≤ a ∧ a < 100) ∧ 
    (10 ≤ b ∧ b < 100) ∧ 
    (10 ≤ c ∧ c < 100) ∧ 
    (a^3 + 3 * b^3 + 9 * c^3 = 9 * a * b * c + 1) ∧ 
    (a + b + c = 9) := 
sorry

end NUMINAMATH_GPT_unique_triple_sum_l342_34206


namespace NUMINAMATH_GPT_remainder_is_162_l342_34209

def polynomial (x : ℝ) : ℝ := 2 * x^4 - x^3 + 4 * x^2 - 5 * x + 6

theorem remainder_is_162 : polynomial 3 = 162 :=
by 
  sorry

end NUMINAMATH_GPT_remainder_is_162_l342_34209


namespace NUMINAMATH_GPT_solution_cos_eq_l342_34261

open Real

theorem solution_cos_eq (x : ℝ) :
  (cos x)^2 + (cos (2 * x))^2 + (cos (3 * x))^2 = 1 ↔
  (∃ k : ℤ, x = k * π / 2 + π / 4) ∨ (∃ k : ℤ, x = k * π / 3 + π / 6) :=
by sorry

end NUMINAMATH_GPT_solution_cos_eq_l342_34261


namespace NUMINAMATH_GPT_find_a_l342_34233

variable (a : ℝ) (h_pos : a > 0) (h_integral : ∫ x in 0..a, (2 * x - 2) = 3)

theorem find_a : a = 3 :=
by sorry

end NUMINAMATH_GPT_find_a_l342_34233


namespace NUMINAMATH_GPT_first_expression_second_expression_l342_34246

-- Define the variables
variables {a x y : ℝ}

-- Statement for the first expression
theorem first_expression (a : ℝ) : (2 * a^2)^3 + (-3 * a^3)^2 = 17 * a^6 := sorry

-- Statement for the second expression
theorem second_expression (x y : ℝ) : (x + 3 * y) * (x - y) = x^2 + 2 * x * y - 3 * y^2 := sorry

end NUMINAMATH_GPT_first_expression_second_expression_l342_34246


namespace NUMINAMATH_GPT_choir_average_age_l342_34251

theorem choir_average_age 
  (avg_f : ℝ) (n_f : ℕ)
  (avg_m : ℝ) (n_m : ℕ)
  (h_f : avg_f = 28) 
  (h_nf : n_f = 12) 
  (h_m : avg_m = 40) 
  (h_nm : n_m = 18) 
  : (n_f * avg_f + n_m * avg_m) / (n_f + n_m) = 35.2 := 
by 
  sorry

end NUMINAMATH_GPT_choir_average_age_l342_34251


namespace NUMINAMATH_GPT_parabola_vertex_l342_34201

theorem parabola_vertex :
  ∃ (x y : ℤ), ((∀ x : ℝ, 2 * x^2 - 4 * x - 7 = y) ∧ x = 1 ∧ y = -9) := 
sorry

end NUMINAMATH_GPT_parabola_vertex_l342_34201


namespace NUMINAMATH_GPT_part_a_part_b_l342_34298

-- Define the function with the given conditions
variable {f : ℝ → ℝ}
variable (h_nonneg : ∀ x, 0 ≤ x → 0 ≤ f x)
variable (h_f1 : f 1 = 1)
variable (h_subadditivity : ∀ (x₁ x₂ : ℝ), 0 ≤ x₁ → 0 ≤ x₂ → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂)

-- Part (a): Prove that f(x) ≤ 2x for all x ∈ [0, 1]
theorem part_a : ∀ x, 0 ≤ x → x ≤ 1 → f x ≤ 2 * x :=
by
  sorry -- Proof required.

-- Part (b): Prove that it is not true that f(x) ≤ 1.9x for all x ∈ [0,1]
theorem part_b : ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ 1.9 * x < f x :=
by
  sorry -- Proof required.

end NUMINAMATH_GPT_part_a_part_b_l342_34298


namespace NUMINAMATH_GPT_smallest_prime_10_less_than_perfect_square_l342_34216

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_10_less_than_perfect_square :
  ∃ (a : ℕ), is_prime a ∧ (∃ (n : ℕ), a = n^2 - 10) ∧ (∀ (b : ℕ), is_prime b ∧ (∃ (m : ℕ), b = m^2 - 10) → a ≤ b) ∧ a = 71 := 
by
  sorry

end NUMINAMATH_GPT_smallest_prime_10_less_than_perfect_square_l342_34216


namespace NUMINAMATH_GPT_ab_equals_six_l342_34264

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end NUMINAMATH_GPT_ab_equals_six_l342_34264


namespace NUMINAMATH_GPT_inequality_holds_l342_34277

noncomputable def positive_real_numbers := { x : ℝ // 0 < x }

theorem inequality_holds (a b c : positive_real_numbers) (h : (a.val * b.val + b.val * c.val + c.val * a.val) = 1) :
    (a.val / b.val + b.val / c.val + c.val / a.val) ≥ (a.val^2 + b.val^2 + c.val^2 + 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l342_34277


namespace NUMINAMATH_GPT_last_child_loses_l342_34283

-- Definitions corresponding to conditions
def num_children := 11
def child_sequence := List.range' 1 num_children
def valid_two_digit_numbers := 90
def invalid_digit_sum_6 := 6
def invalid_digit_sum_9 := 9
def valid_numbers := valid_two_digit_numbers - invalid_digit_sum_6 - invalid_digit_sum_9
def complete_cycles := valid_numbers / num_children
def remaining_numbers := valid_numbers % num_children

-- Statement to be proven
theorem last_child_loses (h1 : num_children = 11)
                         (h2 : valid_two_digit_numbers = 90)
                         (h3 : invalid_digit_sum_6 = 6)
                         (h4 : invalid_digit_sum_9 = 9)
                         (h5 : valid_numbers = valid_two_digit_numbers - invalid_digit_sum_6 - invalid_digit_sum_9)
                         (h6 : remaining_numbers = valid_numbers % num_children) :
  (remaining_numbers = 9) ∧ (num_children - remaining_numbers = 2) :=
by
  sorry

end NUMINAMATH_GPT_last_child_loses_l342_34283


namespace NUMINAMATH_GPT_coloring_satisfies_conditions_l342_34227

/-- Define what it means for a point to be a lattice point -/
def is_lattice_point (x y : ℤ) : Prop := true

/-- Define the coloring function based on coordinates -/
def color (x y : ℤ) : Prop :=
  (x % 2 = 1 ∧ y % 2 = 1) ∨   -- white
  (x % 2 = 1 ∧ y % 2 = 0) ∨   -- black
  (x % 2 = 0)                 -- red (both (even even) and (even odd) are included)

/-- Proving the method of coloring lattice points satisfies the given conditions -/
theorem coloring_satisfies_conditions :
  (∀ x y : ℤ, is_lattice_point x y → 
    color x y ∧ 
    ∃ (A B C : ℤ × ℤ), 
      (is_lattice_point A.fst A.snd ∧ 
       is_lattice_point B.fst B.snd ∧ 
       is_lattice_point C.fst C.snd ∧ 
       color A.fst A.snd ∧ 
       color B.fst B.snd ∧ 
       color C.fst C.snd ∧
       ∃ D : ℤ × ℤ, 
         (is_lattice_point D.fst D.snd ∧ 
          color D.fst D.snd ∧ 
          D.fst = A.fst + C.fst - B.fst ∧ 
          D.snd = A.snd + C.snd - B.snd))) :=
sorry

end NUMINAMATH_GPT_coloring_satisfies_conditions_l342_34227


namespace NUMINAMATH_GPT_arithmetic_mean_of_set_l342_34294

theorem arithmetic_mean_of_set {x : ℝ} (mean_eq_12 : (8 + 16 + 20 + x + 12) / 5 = 12) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_set_l342_34294


namespace NUMINAMATH_GPT_probability_first_spade_last_ace_l342_34236

-- Define the problem parameters
def standard_deck : ℕ := 52
def spades_count : ℕ := 13
def aces_count : ℕ := 4
def ace_of_spades : ℕ := 1

-- Probability of drawing a spade but not an ace as the first card
def prob_spade_not_ace_first : ℚ := 12 / 52

-- Probability of drawing any of the four aces among the two remaining cards
def prob_ace_among_two_remaining : ℚ := 4 / 50

-- Probability of drawing the ace of spades as the first card
def prob_ace_of_spades_first : ℚ := 1 / 52

-- Probability of drawing one of three remaining aces among two remaining cards
def prob_three_aces_among_two_remaining : ℚ := 3 / 50

-- Combined probability according to the cases
def final_probability : ℚ := (prob_spade_not_ace_first * prob_ace_among_two_remaining) + (prob_ace_of_spades_first * prob_three_aces_among_two_remaining)

-- The theorem stating that the computed probability matches the expected result
theorem probability_first_spade_last_ace : final_probability = 51 / 2600 := 
  by
    -- inserting proof steps here would solve the theorem
    sorry

end NUMINAMATH_GPT_probability_first_spade_last_ace_l342_34236


namespace NUMINAMATH_GPT_simplify_sqrt1_simplify_sqrt2_find_a_l342_34237

-- Part 1
theorem simplify_sqrt1 : ∃ m n : ℝ, m^2 + n^2 = 6 ∧ m * n = Real.sqrt 5 ∧ Real.sqrt (6 + 2 * Real.sqrt 5) = m + n :=
by sorry

-- Part 2
theorem simplify_sqrt2 : ∃ m n : ℝ, m^2 + n^2 = 5 ∧ m * n = -Real.sqrt 6 ∧ Real.sqrt (5 - 2 * Real.sqrt 6) = abs (m - n) :=
by sorry

-- Part 3
theorem find_a (a : ℝ) : (Real.sqrt (a^2 + 4 * Real.sqrt 5) = 2 + Real.sqrt 5) → (a = 3 ∨ a = -3) :=
by sorry

end NUMINAMATH_GPT_simplify_sqrt1_simplify_sqrt2_find_a_l342_34237


namespace NUMINAMATH_GPT_find_x_l342_34276

theorem find_x (x : ℕ) (hx1 : 1 ≤ x) (hx2 : x ≤ 100) (hx3 : (31 + 58 + 98 + 3 * x) / 6 = 2 * x) : x = 21 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l342_34276


namespace NUMINAMATH_GPT_equal_lengths_l342_34295

noncomputable def F (x y z : ℝ) := (x+y+z) * (x+y-z) * (y+z-x) * (x+z-y)

variables {a b c d e f : ℝ}

axiom acute_angled_triangle (x y z : ℝ) : Prop

axiom altitudes_sum_greater (x y z : ℝ) : Prop

axiom cond1 : acute_angled_triangle a b c
axiom cond2 : acute_angled_triangle b d f
axiom cond3 : acute_angled_triangle a e f
axiom cond4 : acute_angled_triangle e c d

axiom cond5 : altitudes_sum_greater a b c
axiom cond6 : altitudes_sum_greater b d f
axiom cond7 : altitudes_sum_greater a e f
axiom cond8 : altitudes_sum_greater e c d

axiom cond9 : F a b c = F b d f
axiom cond10 : F a e f = F e c d

theorem equal_lengths : a = d ∧ b = e ∧ c = f := by
  sorry -- Proof not required.

end NUMINAMATH_GPT_equal_lengths_l342_34295


namespace NUMINAMATH_GPT_domain_of_f_l342_34278

open Real

noncomputable def f (x : ℝ) : ℝ := log (log x)

theorem domain_of_f : { x : ℝ | 1 < x } = { x : ℝ | ∃ y > 1, x = y } :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l342_34278


namespace NUMINAMATH_GPT_trigonometric_identity_l342_34229

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l342_34229


namespace NUMINAMATH_GPT_abs_difference_lt_2t_l342_34202

/-- Given conditions of absolute values with respect to t -/
theorem abs_difference_lt_2t (x y s t : ℝ) (h₁ : |x - s| < t) (h₂ : |y - s| < t) :
  |x - y| < 2 * t :=
sorry

end NUMINAMATH_GPT_abs_difference_lt_2t_l342_34202


namespace NUMINAMATH_GPT_exp_ineq_solution_set_l342_34286

theorem exp_ineq_solution_set (e : ℝ) (h : e = Real.exp 1) :
  {x : ℝ | e^(2*x - 1) < 1} = {x : ℝ | x < 1 / 2} :=
sorry

end NUMINAMATH_GPT_exp_ineq_solution_set_l342_34286


namespace NUMINAMATH_GPT_shells_put_back_l342_34221

def shells_picked_up : ℝ := 324.0
def shells_left : ℝ := 32.0

theorem shells_put_back : shells_picked_up - shells_left = 292 := by
  sorry

end NUMINAMATH_GPT_shells_put_back_l342_34221


namespace NUMINAMATH_GPT_julie_initial_savings_l342_34281

-- Definition of the simple interest condition
def simple_interest_condition (P : ℝ) : Prop :=
  575 = P * 0.04 * 5

-- Definition of the compound interest condition
def compound_interest_condition (P : ℝ) : Prop :=
  635 = P * ((1 + 0.05) ^ 5 - 1)

-- The final proof problem
theorem julie_initial_savings (P : ℝ) :
  simple_interest_condition P →
  compound_interest_condition P →
  2 * P = 5750 :=
by sorry

end NUMINAMATH_GPT_julie_initial_savings_l342_34281


namespace NUMINAMATH_GPT_sixth_graders_count_l342_34299

theorem sixth_graders_count (total_students seventh_graders_percentage sixth_graders_percentage : ℝ)
                            (seventh_graders_count : ℕ)
                            (h1 : seventh_graders_percentage = 0.32)
                            (h2 : seventh_graders_count = 64)
                            (h3 : sixth_graders_percentage = 0.38)
                            (h4 : seventh_graders_count = seventh_graders_percentage * total_students) :
                            sixth_graders_percentage * total_students = 76 := by
  sorry

end NUMINAMATH_GPT_sixth_graders_count_l342_34299


namespace NUMINAMATH_GPT_solve_abs_linear_eq_l342_34200

theorem solve_abs_linear_eq (x : ℝ) : (|x - 1| + x - 1 = 0) ↔ (x ≤ 1) :=
sorry

end NUMINAMATH_GPT_solve_abs_linear_eq_l342_34200


namespace NUMINAMATH_GPT_negation_of_p_l342_34207

open Real

def p : Prop := ∃ x : ℝ, sin x < (1 / 2) * x

theorem negation_of_p : ¬p ↔ ∀ x : ℝ, sin x ≥ (1 / 2) * x := 
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l342_34207


namespace NUMINAMATH_GPT_num_false_statements_is_three_l342_34222

-- Definitions of the statements on the card
def s1 : Prop := ∀ (false_statements : ℕ), false_statements = 1
def s2 : Prop := ∀ (false_statements_card1 false_statements_card2 : ℕ), false_statements_card1 + false_statements_card2 = 2
def s3 : Prop := ∀ (false_statements : ℕ), false_statements = 3
def s4 : Prop := ∀ (false_statements_card1 false_statements_card2 : ℕ), false_statements_card1 = false_statements_card2

-- Main proof problem: The number of false statements on this card is 3
theorem num_false_statements_is_three 
  (h_s1 : ¬ s1)
  (h_s2 : ¬ s2)
  (h_s3 : s3)
  (h_s4 : ¬ s4) :
  ∃ (n : ℕ), n = 3 :=
by
  sorry

end NUMINAMATH_GPT_num_false_statements_is_three_l342_34222


namespace NUMINAMATH_GPT_find_minimum_fuse_length_l342_34234

def safeZone : ℝ := 70
def fuseBurningSpeed : ℝ := 0.112
def personSpeed : ℝ := 7
def minimumFuseLength : ℝ := 1.1

theorem find_minimum_fuse_length (x : ℝ) (h1 : x ≥ 0):
  (safeZone / personSpeed) * fuseBurningSpeed ≤ x :=
by
  sorry

end NUMINAMATH_GPT_find_minimum_fuse_length_l342_34234


namespace NUMINAMATH_GPT_value_of_a4_l342_34263

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d
def sum_of_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) := ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem value_of_a4 {a : ℕ → ℝ} {S : ℕ → ℝ} (h1 : arithmetic_sequence a)
  (h2 : sum_of_arithmetic_sequence S a) (h3 : S 7 = 28) :
  a 4 = 4 := 
  sorry

end NUMINAMATH_GPT_value_of_a4_l342_34263


namespace NUMINAMATH_GPT_inequality_holds_for_all_real_l342_34211

theorem inequality_holds_for_all_real (x : ℝ) : x^2 + 6 * x + 8 ≥ -(x + 4) * (x + 6) :=
  sorry

end NUMINAMATH_GPT_inequality_holds_for_all_real_l342_34211


namespace NUMINAMATH_GPT_simplify_fraction_l342_34226

theorem simplify_fraction (a : ℕ) (h : a = 5) : (15 * a^4) / (75 * a^3) = 1 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l342_34226


namespace NUMINAMATH_GPT_robe_initial_savings_l342_34270

noncomputable def initial_savings (repair_fee corner_light_cost brake_disk_cost tires_cost remaining_savings : ℕ) : ℕ :=
  remaining_savings + repair_fee + corner_light_cost + 2 * brake_disk_cost + tires_cost

theorem robe_initial_savings :
  let R := 10
  let corner_light := 2 * R
  let brake_disk := 3 * corner_light
  let tires := corner_light + 2 * brake_disk
  let remaining := 480
  initial_savings R corner_light brake_disk tires remaining = 770 :=
by
  sorry

end NUMINAMATH_GPT_robe_initial_savings_l342_34270


namespace NUMINAMATH_GPT_trapezoid_height_l342_34279

-- Definitions of the problem conditions
def is_isosceles_trapezoid (a b : ℝ) : Prop :=
  ∃ (AB CD BM CN h : ℝ), a = 24 ∧ b = 10 ∧ AB = 25 ∧ CD = 25 ∧ BM = h ∧ CN = h ∧
  BM ^ 2 + ((24 - 10) / 2) ^ 2 = AB ^ 2

-- The theorem to prove
theorem trapezoid_height (a b : ℝ) (h : ℝ) 
  (H : is_isosceles_trapezoid a b) : h = 24 :=
sorry

end NUMINAMATH_GPT_trapezoid_height_l342_34279


namespace NUMINAMATH_GPT_largest_minus_smallest_eq_13_l342_34224

theorem largest_minus_smallest_eq_13 :
  let a := (-1 : ℤ) ^ 3
  let b := (-1 : ℤ) ^ 2
  let c := -(2 : ℤ) ^ 2
  let d := (-3 : ℤ) ^ 2
  max (max a (max b c)) d - min (min a (min b c)) d = 13 := by
  sorry

end NUMINAMATH_GPT_largest_minus_smallest_eq_13_l342_34224


namespace NUMINAMATH_GPT_alice_has_ball_after_two_turns_l342_34260

noncomputable def probability_alice_has_ball_twice_turns : ℚ :=
  let P_AB_A : ℚ := 1/2 * 1/3
  let P_ABC_A : ℚ := 1/2 * 1/3 * 1/2
  let P_AA : ℚ := 1/2 * 1/2
  P_AB_A + P_ABC_A + P_AA

theorem alice_has_ball_after_two_turns :
  probability_alice_has_ball_twice_turns = 1/2 := 
by
  sorry

end NUMINAMATH_GPT_alice_has_ball_after_two_turns_l342_34260


namespace NUMINAMATH_GPT_num_dogs_l342_34288

-- Define the conditions
def total_animals := 11
def ducks := 6
def total_legs := 32
def legs_per_duck := 2
def legs_per_dog := 4

-- Calculate intermediate values based on conditions
def duck_legs := ducks * legs_per_duck
def remaining_legs := total_legs - duck_legs

-- The proof statement
theorem num_dogs : ∃ D : ℕ, D = remaining_legs / legs_per_dog ∧ D + ducks = total_animals :=
by
  sorry

end NUMINAMATH_GPT_num_dogs_l342_34288


namespace NUMINAMATH_GPT_total_price_correct_l342_34240

-- Define the initial price, reduction, and the number of boxes
def initial_price : ℝ := 104
def price_reduction : ℝ := 24
def number_of_boxes : ℕ := 20

-- Define the new price as initial price minus the reduction
def new_price := initial_price - price_reduction

-- Define the total price as the new price times the number of boxes
def total_price := (number_of_boxes : ℝ) * new_price

-- The goal is to prove the total price equals 1600
theorem total_price_correct : total_price = 1600 := by
  sorry

end NUMINAMATH_GPT_total_price_correct_l342_34240


namespace NUMINAMATH_GPT_cosine_inequality_l342_34215

theorem cosine_inequality (a b c : ℝ) : ∃ x : ℝ, 
    a * Real.cos x + b * Real.cos (3 * x) + c * Real.cos (9 * x) ≥ (|a| + |b| + |c|) / 2 :=
sorry

end NUMINAMATH_GPT_cosine_inequality_l342_34215


namespace NUMINAMATH_GPT_smallest_number_of_digits_to_append_l342_34244

theorem smallest_number_of_digits_to_append (n : ℕ) (d : ℕ) : n = 2014 → d = 4 → 
  ∃ m : ℕ, (m = n * 10^d + 4506) ∧ (m % 2520 = 0) :=
by
  intros
  sorry

end NUMINAMATH_GPT_smallest_number_of_digits_to_append_l342_34244


namespace NUMINAMATH_GPT_weight_ratios_l342_34231

theorem weight_ratios {x y z k : ℝ} (h1 : x + y = k * z) (h2 : y + z = k * x) (h3 : z + x = k * y) : x = y ∧ y = z :=
by 
  -- Proof to be filled in later
  sorry

end NUMINAMATH_GPT_weight_ratios_l342_34231


namespace NUMINAMATH_GPT_color_column_l342_34223

theorem color_column (n : ℕ) (color : ℕ) (board : ℕ → ℕ → ℕ) 
  (h_colors : ∀ i j, 1 ≤ board i j ∧ board i j ≤ n^2)
  (h_block : ∀ i j, (∀ k l : ℕ, k < n → l < n → ∃ c, ∀ a b : ℕ, k + a * n < n → l + b * n < n → board (i + k + a * n) (j + l + b * n) = c))
  (h_row : ∃ r, ∀ k, k < n → ∃ c, 1 ≤ c ∧ c ≤ n ∧ board r k = c) :
  ∃ c, (∀ j, 1 ≤ board c j ∧ board c j ≤ n) :=
sorry

end NUMINAMATH_GPT_color_column_l342_34223


namespace NUMINAMATH_GPT_union_of_sets_l342_34257

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Prove that A ∪ B = {x | -1 < x ∧ x ≤ 2}
theorem union_of_sets (x : ℝ) : x ∈ (A ∪ B) ↔ x ∈ {x | -1 < x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_union_of_sets_l342_34257


namespace NUMINAMATH_GPT_smallest_integer_representation_l342_34203

theorem smallest_integer_representation :
  ∃ a b : ℕ, a > 3 ∧ b > 3 ∧ (13 = a + 3 ∧ 13 = 3 * b + 1) := by
  sorry

end NUMINAMATH_GPT_smallest_integer_representation_l342_34203


namespace NUMINAMATH_GPT_trader_gain_percentage_l342_34275

structure PenType :=
  (pens_sold : ℕ)
  (cost_per_pen : ℕ)

def total_cost (pen : PenType) : ℕ :=
  pen.pens_sold * pen.cost_per_pen

def gain (pen : PenType) (multiplier : ℕ) : ℕ :=
  multiplier * pen.cost_per_pen

def weighted_average_gain_percentage (penA penB penC : PenType) (gainA gainB gainC : ℕ) : ℚ :=
  (((gainA + gainB + gainC):ℚ) / ((total_cost penA + total_cost penB + total_cost penC):ℚ)) * 100

theorem trader_gain_percentage :
  ∀ (penA penB penC : PenType)
  (gainA gainB gainC : ℕ),
  penA.pens_sold = 60 →
  penA.cost_per_pen = 2 →
  penB.pens_sold = 40 →
  penB.cost_per_pen = 3 →
  penC.pens_sold = 50 →
  penC.cost_per_pen = 4 →
  gainA = 20 * penA.cost_per_pen →
  gainB = 15 * penB.cost_per_pen →
  gainC = 10 * penC.cost_per_pen →
  weighted_average_gain_percentage penA penB penC gainA gainB gainC = 28.41 := 
by
  intros
  sorry

end NUMINAMATH_GPT_trader_gain_percentage_l342_34275


namespace NUMINAMATH_GPT_ursula_initial_money_l342_34284

def cost_per_hot_dog : ℝ := 1.50
def number_of_hot_dogs : ℕ := 5
def cost_per_salad : ℝ := 2.50
def number_of_salads : ℕ := 3
def change_received : ℝ := 5.00

def total_cost_of_hot_dogs : ℝ := number_of_hot_dogs * cost_per_hot_dog
def total_cost_of_salads : ℝ := number_of_salads * cost_per_salad
def total_cost : ℝ := total_cost_of_hot_dogs + total_cost_of_salads
def amount_paid : ℝ := total_cost + change_received

theorem ursula_initial_money : amount_paid = 20.00 := by
  /- Proof here, which is not required for the task -/
  sorry

end NUMINAMATH_GPT_ursula_initial_money_l342_34284


namespace NUMINAMATH_GPT_find_point_A_l342_34292

-- Define the point -3, 4
def pointP : ℝ × ℝ := (-3, 4)

-- Define the point 0, 2
def pointB : ℝ × ℝ := (0, 2)

-- Define the coordinates of point A
def pointA (x : ℝ) : ℝ × ℝ := (x, 0)

-- The hypothesis using the condition derived from the problem
def ray_reflection_condition (x : ℝ) : Prop :=
  4 / (x + 3) = -2 / x

-- The main theorem we need to prove that the coordinates of point A are (-1, 0)
theorem find_point_A :
  ∃ x : ℝ, ray_reflection_condition x ∧ pointA x = (-1, 0) :=
sorry

end NUMINAMATH_GPT_find_point_A_l342_34292


namespace NUMINAMATH_GPT_cs_competition_hits_l342_34225

theorem cs_competition_hits :
  (∃ x y z : ℕ, 5 * x + 4 * y + 3 * z = 15 ∧ x + y + z ≥ 1)
  ∧ (∀ x y z : ℕ, 5 * x + 4 * y + 3 * z = 15 ∧ x + y + z ≥ 1 → (x = 1 ∧ y = 1 ∧ z = 2) ∨ (x = 0 ∧ y = 3 ∧ z = 1)) :=
by
  sorry

end NUMINAMATH_GPT_cs_competition_hits_l342_34225


namespace NUMINAMATH_GPT_percent_defective_units_shipped_l342_34212

theorem percent_defective_units_shipped (h1 : 8 / 100 * 4 / 100 = 32 / 10000) :
  (32 / 10000) * 100 = 0.32 := 
sorry

end NUMINAMATH_GPT_percent_defective_units_shipped_l342_34212


namespace NUMINAMATH_GPT_range_f_l342_34290

noncomputable def f (x : ℝ) : ℝ := if x = -5 then 0 else 3 * (x - 4)

theorem range_f : (Set.range f) = (Set.univ \ { -27 }) :=
by
  sorry

end NUMINAMATH_GPT_range_f_l342_34290


namespace NUMINAMATH_GPT_speed_of_man_cycling_l342_34268

theorem speed_of_man_cycling (L B : ℝ) (h1 : L / B = 1 / 3) (h2 : B = 3 * L)
  (h3 : L * B = 30000) (h4 : ∀ t : ℝ, t = 4 / 60): 
  ( (2 * L + 2 * B) / (4 / 60) ) = 12000 :=
by
  -- Assume given conditions
  sorry

end NUMINAMATH_GPT_speed_of_man_cycling_l342_34268


namespace NUMINAMATH_GPT_problem_l342_34287

open Real

noncomputable def f (x : ℝ) : ℝ := x + 1

theorem problem (f : ℝ → ℝ)
  (h : ∀ x, 2 * f x - f (-x) = 3 * x + 1) :
  f 1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_l342_34287


namespace NUMINAMATH_GPT_simplify_and_evaluate_l342_34296

noncomputable def x := Real.tan (Real.pi / 4) + Real.cos (Real.pi / 6)

theorem simplify_and_evaluate :
  ((x / (x ^ 2 - 1)) * ((x - 1) / x - 2)) = - (2 * Real.sqrt 3) / 3 := 
sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l342_34296


namespace NUMINAMATH_GPT_line_ellipse_common_points_l342_34266

def point (P : Type*) := P → ℝ × ℝ

theorem line_ellipse_common_points
  (m n : ℝ)
  (no_common_points_with_circle : ∀ (x y : ℝ), mx + ny - 3 = 0 → x^2 + y^2 ≠ 3) :
  ∀ (Px Py : ℝ), (Px = m ∧ Py = n) →
  (∃ (x1 y1 x2 y2 : ℝ), ((x1^2 / 7) + (y1^2 / 3) = 1 ∧ (x2^2 / 7) + (y2^2 / 3) = 1) ∧ (x1, y1) ≠ (x2, y2)) :=
by
  sorry

end NUMINAMATH_GPT_line_ellipse_common_points_l342_34266


namespace NUMINAMATH_GPT_glenda_speed_is_8_l342_34218

noncomputable def GlendaSpeed : ℝ :=
  let AnnSpeed := 6
  let Hours := 3
  let Distance := 42
  let AnnDistance := AnnSpeed * Hours
  let GlendaDistance := Distance - AnnDistance
  GlendaDistance / Hours

theorem glenda_speed_is_8 : GlendaSpeed = 8 := by
  sorry

end NUMINAMATH_GPT_glenda_speed_is_8_l342_34218


namespace NUMINAMATH_GPT_width_of_room_l342_34272

theorem width_of_room (length room_area cost paving_rate : ℝ) 
  (H_length : length = 5.5) 
  (H_cost : cost = 17600)
  (H_paving_rate : paving_rate = 800)
  (H_area : room_area = cost / paving_rate) :
  room_area = length * 4 :=
by
  -- sorry to skip proof
  sorry

end NUMINAMATH_GPT_width_of_room_l342_34272


namespace NUMINAMATH_GPT_math_problem_l342_34282

theorem math_problem : 
  (Real.sqrt 4) * (4 ^ (1 / 2: ℝ)) + (16 / 4) * 2 - (8 ^ (1 / 2: ℝ)) = 12 - 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l342_34282
