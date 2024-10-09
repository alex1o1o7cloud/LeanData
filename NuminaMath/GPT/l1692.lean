import Mathlib

namespace proportional_function_decreases_l1692_169225

theorem proportional_function_decreases
  (k : ℝ) (h : k ≠ 0) (h_point : ∃ k, (-4 : ℝ) = k * 2) :
  ∀ x1 x2 : ℝ, x1 < x2 → (k * x1) > (k * x2) :=
by
  sorry

end proportional_function_decreases_l1692_169225


namespace sufficient_not_necessary_condition_l1692_169271

theorem sufficient_not_necessary_condition (x y : ℝ) : 
  (x - y) * x^4 < 0 → x < y ∧ ¬(x < y → (x - y) * x^4 < 0) := 
sorry

end sufficient_not_necessary_condition_l1692_169271


namespace book_has_125_pages_l1692_169250

-- Define the number of pages in each chapter
def chapter1_pages : ℕ := 66
def chapter2_pages : ℕ := 35
def chapter3_pages : ℕ := 24

-- Define the total number of pages in the book
def total_pages : ℕ := chapter1_pages + chapter2_pages + chapter3_pages

-- State the theorem to prove that the total number of pages is 125
theorem book_has_125_pages : total_pages = 125 := 
by 
  -- The proof is omitted for the purpose of this task
  sorry

end book_has_125_pages_l1692_169250


namespace volume_of_red_tetrahedron_in_colored_cube_l1692_169207

noncomputable def red_tetrahedron_volume (side_length : ℝ) : ℝ :=
  let cube_volume := side_length ^ 3
  let clear_tetrahedron_volume := (1/3) * (1/2 * side_length * side_length) * side_length
  let red_tetrahedron_volume := (cube_volume - 4 * clear_tetrahedron_volume)
  red_tetrahedron_volume

theorem volume_of_red_tetrahedron_in_colored_cube 
: red_tetrahedron_volume 8 = 512 / 3 := by
  sorry

end volume_of_red_tetrahedron_in_colored_cube_l1692_169207


namespace point_Q_in_third_quadrant_l1692_169298

theorem point_Q_in_third_quadrant (m : ℝ) :
  (2 * m + 4 = 0 → (m - 3, m).fst < 0 ∧ (m - 3, m).snd < 0) :=
by
  sorry

end point_Q_in_third_quadrant_l1692_169298


namespace sat_production_correct_highest_lowest_diff_correct_total_weekly_wage_correct_l1692_169286

def avg_daily_production := 400
def weekly_planned_production := 2800
def daily_deviations := [15, -5, 21, 16, -7, 0, -8]
def total_weekly_deviation := 80

-- Calculation for sets produced on Saturday
def sat_production_exceeds_plan := total_weekly_deviation - (daily_deviations.take (daily_deviations.length - 1)).sum
def sat_production := avg_daily_production + sat_production_exceeds_plan

-- Calculation for the difference between the max and min production days
def max_deviation := max sat_production_exceeds_plan (daily_deviations.maximum.getD 0)
def min_deviation := min sat_production_exceeds_plan (daily_deviations.minimum.getD 0)
def highest_lowest_diff := max_deviation - min_deviation

-- Calculation for the weekly wage for each worker
def workers := 20
def daily_wage := 200
def basic_weekly_wage := daily_wage * 7
def additional_wage := (15 + 21 + 16 + sat_production_exceeds_plan) * 10 - (5 + 7 + 8) * 15
def total_bonus := additional_wage / workers
def total_weekly_wage := basic_weekly_wage + total_bonus

theorem sat_production_correct : sat_production = 448 := by
  sorry

theorem highest_lowest_diff_correct : highest_lowest_diff = 56 := by
  sorry

theorem total_weekly_wage_correct : total_weekly_wage = 1435 := by
  sorry

end sat_production_correct_highest_lowest_diff_correct_total_weekly_wage_correct_l1692_169286


namespace ladder_rung_length_l1692_169243

noncomputable def ladder_problem : Prop :=
  let total_height_ft := 50
  let spacing_in := 6
  let wood_ft := 150
  let feet_to_inches(ft : ℕ) : ℕ := ft * 12
  let total_height_in := feet_to_inches total_height_ft
  let wood_in := feet_to_inches wood_ft
  let number_of_rungs := total_height_in / spacing_in
  let length_of_each_rung := wood_in / number_of_rungs
  length_of_each_rung = 18

theorem ladder_rung_length : ladder_problem := sorry

end ladder_rung_length_l1692_169243


namespace storage_temperature_overlap_l1692_169270

theorem storage_temperature_overlap (T_A_min T_A_max T_B_min T_B_max : ℝ) 
  (hA : T_A_min = 0)
  (hA' : T_A_max = 5)
  (hB : T_B_min = 2)
  (hB' : T_B_max = 7) : 
  (max T_A_min T_B_min, min T_A_max T_B_max) = (2, 5) := by 
{
  sorry -- The proof is omitted as per instructions.
}

end storage_temperature_overlap_l1692_169270


namespace problem1_problem2_l1692_169218

-- Definitions for the sets and conditions
def setA : Set ℝ := {x | -1 < x ∧ x < 2}
def setB (a : ℝ) : Set ℝ := if a > 0 then {x | x ≤ -2 ∨ x ≥ (1 / a)} else ∅

-- Problem 1: Prove the intersection for a == 1
theorem problem1 : (setB 1) ∩ setA = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

-- Problem 2: Prove the range of a
theorem problem2 (a : ℝ) (h : setB a ⊆ setAᶜ) : 0 < a ∧ a ≤ 1/2 :=
by
  sorry

end problem1_problem2_l1692_169218


namespace average_of_original_set_l1692_169266

-- Average of 8 numbers is some value A and the average of the new set where each number is 
-- multiplied by 8 is 168. We need to show that the original average A is 21.

theorem average_of_original_set (A : ℝ) (h1 : (64 * A) / 8 = 168) : A = 21 :=
by {
  -- This is the theorem statement, we add the proof next
  sorry -- proof placeholder
}

end average_of_original_set_l1692_169266


namespace cakes_and_bread_weight_l1692_169249

theorem cakes_and_bread_weight 
  (B : ℕ)
  (cake_weight : ℕ := B + 100)
  (h1 : 4 * cake_weight = 800)
  : 3 * cake_weight + 5 * B = 1100 := by
  sorry

end cakes_and_bread_weight_l1692_169249


namespace range_of_a_l1692_169257

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x >= 0 then -x + 3 * a else x^2 - a * x + 1

theorem range_of_a (a : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → f a x1 ≥ f a x2) ↔ (0 <= a ∧ a <= 1/3) :=
by
  sorry

end range_of_a_l1692_169257


namespace total_length_of_scale_l1692_169284

theorem total_length_of_scale (num_parts : ℕ) (length_per_part : ℕ) 
  (h1: num_parts = 4) (h2: length_per_part = 20) : 
  num_parts * length_per_part = 80 := by
  sorry

end total_length_of_scale_l1692_169284


namespace problem1_problem2_l1692_169202

-- Definitions for the inequalities
def f (x a : ℝ) : ℝ := abs (x - a) - 1

-- Problem 1: Given a = 2, solve the inequality f(x) + |2x - 3| > 0
theorem problem1 (x : ℝ) (h1 : abs (x - 2) + abs (2 * x - 3) > 1) : (x ≥ 2 ∨ x ≤ 4 / 3) := sorry

-- Problem 2: If the inequality f(x) > |x - 3| has solutions, find the range of a
theorem problem2 (a : ℝ) (h2 : ∃ x : ℝ, abs (x - a) - abs (x - 3) > 1) : a < 2 ∨ a > 4 := sorry

end problem1_problem2_l1692_169202


namespace smallest_seating_l1692_169280

theorem smallest_seating (N : ℕ) (h: ∀ (chairs : ℕ) (occupants : ℕ), 
  chairs = 100 ∧ occupants = 25 → 
  ∃ (adjacent_occupied: ℕ), adjacent_occupied > 0 ∧ adjacent_occupied < chairs ∧
  adjacent_occupied ≠ occupants) : 
  N = 25 :=
sorry

end smallest_seating_l1692_169280


namespace max_b_integer_l1692_169245

theorem max_b_integer (b : ℤ) : (∀ x : ℝ, x^2 + (b : ℝ) * x + 20 ≠ -10) → b ≤ 10 :=
by
  sorry

end max_b_integer_l1692_169245


namespace area_of_rectangle_l1692_169275

theorem area_of_rectangle (a b : ℝ) (area : ℝ) 
(h1 : a = 5.9) 
(h2 : b = 3) 
(h3 : area = a * b) : 
area = 17.7 := 
by 
  -- proof goes here
  sorry

-- Definitions and conditions alignment:
-- a represents one side of the rectangle.
-- b represents the other side of the rectangle.
-- area represents the area of the rectangle.
-- h1: a = 5.9 corresponds to the first condition.
-- h2: b = 3 corresponds to the second condition.
-- h3: area = a * b connects the conditions to the formula to find the area.
-- The goal is to show that area = 17.7, which matches the correct answer.

end area_of_rectangle_l1692_169275


namespace price_per_pot_l1692_169228

-- Definitions based on conditions
def total_pots : ℕ := 80
def proportion_not_cracked : ℚ := 3 / 5
def total_revenue : ℚ := 1920

-- The Lean statement to prove she sold each clay pot for $40
theorem price_per_pot : (total_revenue / (total_pots * proportion_not_cracked)) = 40 := 
by sorry

end price_per_pot_l1692_169228


namespace eccentricity_proof_l1692_169299

variables (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
def ellipse_eq (x y: ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def circle_eq (x y: ℝ) := x^2 + y^2 = b^2

-- Conditions
def a_eq_3b : Prop := a = 3 * b
def major_minor_axis_relation : Prop := a^2 = b^2 + c^2

-- To prove
theorem eccentricity_proof 
  (h3 : a_eq_3b a b)
  (h4 : major_minor_axis_relation a b c) :
  (c / a) = (2 * Real.sqrt 2 / 3) := 
  sorry

end eccentricity_proof_l1692_169299


namespace valid_digit_cancel_fractions_l1692_169277

def digit_cancel_fraction (a b c d : ℕ) : Prop :=
  10 * a + b == 0 ∧ 10 * c + d == 0 ∧ 
  (b == d ∨ b == c ∨ a == d ∨ a == c) ∧
  (b ≠ a ∨ d ≠ c) ∧
  ((10 * a + b) ≠ (10 * c + d)) ∧
  ((10 * a + b) * d == (10 * c + d) * a)

theorem valid_digit_cancel_fractions :
  ∀ (a b c d : ℕ), 
  digit_cancel_fraction a b c d → 
  (10 * a + b == 26 ∧ 10 * c + d == 65) ∨
  (10 * a + b == 16 ∧ 10 * c + d == 64) ∨
  (10 * a + b == 19 ∧ 10 * c + d == 95) ∨
  (10 * a + b == 49 ∧ 10 * c + d == 98) :=
by {sorry}

end valid_digit_cancel_fractions_l1692_169277


namespace smallest_yellow_marbles_l1692_169238

-- Definitions for given conditions
def total_marbles (n : ℕ): Prop := n > 0
def blue_marbles (n : ℕ) : ℕ := n / 4
def red_marbles (n : ℕ) : ℕ := n / 6
def green_marbles : ℕ := 7
def yellow_marbles (n : ℕ) : ℕ := n - (blue_marbles n + red_marbles n + green_marbles)

-- Lean statement that verifies the smallest number of yellow marbles is 0
theorem smallest_yellow_marbles (n : ℕ) (h : total_marbles n) : yellow_marbles n = 0 :=
  sorry

end smallest_yellow_marbles_l1692_169238


namespace number_solution_l1692_169203

theorem number_solution : ∃ x : ℝ, x + 9 = x^2 ∧ x = (1 + Real.sqrt 37) / 2 :=
by
  use (1 + Real.sqrt 37) / 2
  simp
  sorry

end number_solution_l1692_169203


namespace coterminal_angle_l1692_169247

theorem coterminal_angle (α : ℤ) : 
  ∃ k : ℤ, α = k * 360 + 283 ↔ ∃ k : ℤ, α = k * 360 - 437 :=
sorry

end coterminal_angle_l1692_169247


namespace true_propositions_for_quadratic_equations_l1692_169208

theorem true_propositions_for_quadratic_equations :
  (∀ (a b c : ℤ), a ≠ 0 → (∃ Δ : ℤ, Δ^2 = b^2 - 4 * a * c → ∃ x : ℚ, x^2 + (b / a) * x + (c / a) = 0)) ∧
  (∀ (a b c : ℤ), a ≠ 0 → (∃ x : ℚ, x^2 + (b / a) * x + (c / a) = 0 → ∃ Δ : ℤ, Δ^2 = b^2 - 4 * a * c)) ∧
  (¬ ∀ (a b c : ℝ), a ≠ 0 → (∃ x : ℚ, x^2 + (b / a) * x + (c / a) = 0)) ∧
  (∀ (a b c : ℤ), a ≠ 0 ∧ a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 → ¬∃ x : ℚ, x^2 + (b / a) * x + (c / a) = 0) :=
by sorry

end true_propositions_for_quadratic_equations_l1692_169208


namespace squirrels_in_tree_l1692_169268

theorem squirrels_in_tree (nuts : ℕ) (squirrels : ℕ) (h1 : nuts = 2) (h2 : squirrels = nuts + 2) : squirrels = 4 :=
by
    rw [h1] at h2
    exact h2

end squirrels_in_tree_l1692_169268


namespace proof_problem_l1692_169262

variables {a b c : ℝ}

theorem proof_problem (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 4 * a^2 + b^2 + 16 * c^2 = 1) :
  (0 < a * b ∧ a * b < 1 / 4) ∧ (1 / a^2 + 1 / b^2 + 1 / (4 * a * b * c^2) > 49) :=
by
  sorry

end proof_problem_l1692_169262


namespace picked_tomatoes_eq_53_l1692_169276

-- Definitions based on the conditions
def initial_tomatoes : ℕ := 177
def initial_potatoes : ℕ := 12
def items_left : ℕ := 136

-- Define what we need to prove
theorem picked_tomatoes_eq_53 : initial_tomatoes + initial_potatoes - items_left = 53 :=
by sorry

end picked_tomatoes_eq_53_l1692_169276


namespace f_at_neg2_l1692_169289

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2^x - Real.log (x^2 - 3*x + 5) / Real.log 3 
else -2^(-x) + Real.log ((-x)^2 + 3*(-x) + 5) / Real.log 3 

theorem f_at_neg2 : f (-2) = -3 := by
  sorry

end f_at_neg2_l1692_169289


namespace find_q_l1692_169259

theorem find_q (P J T : ℝ) (Q : ℝ) (q : ℚ) 
  (h1 : J = 0.75 * P)
  (h2 : J = 0.80 * T)
  (h3 : T = P * (1 - Q))
  (h4 : Q = q / 100) :
  q = 6.25 := 
by
  sorry

end find_q_l1692_169259


namespace factorize_expression_l1692_169221

theorem factorize_expression (a x y : ℤ) : a * x - a * y = a * (x - y) :=
  sorry

end factorize_expression_l1692_169221


namespace find_number_l1692_169292

theorem find_number (x : ℝ) (h : 0.15 * 0.30 * 0.50 * x = 126) : x = 5600 := 
by
  -- Proof goes here
  sorry

end find_number_l1692_169292


namespace train_length_l1692_169214

theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (conversion_factor : ℝ) (speed_ms : ℝ) (distance_m : ℝ) 
  (h1 : speed_kmh = 36) 
  (h2 : time_s = 28)
  (h3 : conversion_factor = 1000 / 3600) -- convert km/hr to m/s
  (h4 : speed_ms = speed_kmh * conversion_factor)
  (h5 : distance_m = speed_ms * time_s) :
  distance_m = 280 := 
by
  sorry

end train_length_l1692_169214


namespace original_cost_of_car_l1692_169297

noncomputable def original_cost (C : ℝ) : ℝ :=
  if h : C + 13000 ≠ 0 then (60900 - (C + 13000)) / (C + 13000) * 100 else 0

theorem original_cost_of_car 
  (C : ℝ) 
  (h1 : original_cost C = 10.727272727272727)
  (h2 : 60900 - (C + 13000) > 0) :
  C = 433500 :=
by
  sorry

end original_cost_of_car_l1692_169297


namespace taylor_one_basket_probability_l1692_169204

-- Definitions based on conditions
def not_make_basket_prob : ℚ := 1 / 3
def make_basket_prob : ℚ := 1 - not_make_basket_prob
def trials : ℕ := 3
def successes : ℕ := 1

def binomial_coefficient (n k : ℕ) : ℕ := n.choose k

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coefficient n k) * (p^k) * ((1 - p)^(n - k))

theorem taylor_one_basket_probability : 
  binomial_probability trials successes make_basket_prob = 2 / 9 :=
by
  rw [binomial_probability, binomial_coefficient]
  -- The rest of the proof steps can involve simplifications 
  -- and calculations that were mentioned in the solution.
  sorry

end taylor_one_basket_probability_l1692_169204


namespace remaining_milk_correct_l1692_169254

def arranged_milk : ℝ := 21.52
def sold_milk : ℝ := 12.64
def remaining_milk (total : ℝ) (sold : ℝ) : ℝ := total - sold

theorem remaining_milk_correct :
  remaining_milk arranged_milk sold_milk = 8.88 :=
by
  sorry

end remaining_milk_correct_l1692_169254


namespace non_degenerate_ellipse_l1692_169241

theorem non_degenerate_ellipse (x y k : ℝ) : (∃ k, (2 * x^2 + 9 * y^2 - 12 * x - 27 * y = k) → k > -135 / 4) := sorry

end non_degenerate_ellipse_l1692_169241


namespace total_budget_is_correct_l1692_169217

-- Define the costs of TV, fridge, and computer based on the given conditions
def cost_tv : ℕ := 600
def cost_computer : ℕ := 250
def cost_fridge : ℕ := cost_computer + 500

-- Statement to prove the total budget
theorem total_budget_is_correct : cost_tv + cost_computer + cost_fridge = 1600 :=
by
  sorry

end total_budget_is_correct_l1692_169217


namespace domain_of_log_function_l1692_169278

theorem domain_of_log_function (x : ℝ) :
  (-1 < x ∧ x < 1) ↔ (1 - x) / (1 + x) > 0 :=
by sorry

end domain_of_log_function_l1692_169278


namespace Claire_photos_is_5_l1692_169201

variable (Claire_photos : ℕ)
variable (Lisa_photos : ℕ := 3 * Claire_photos)
variable (Robert_photos : ℕ := Claire_photos + 10)

theorem Claire_photos_is_5
  (h1 : Lisa_photos = Robert_photos) :
  Claire_photos = 5 :=
by
  sorry

end Claire_photos_is_5_l1692_169201


namespace smallest_value_y_l1692_169234

theorem smallest_value_y : ∃ y : ℝ, 3 * y ^ 2 + 33 * y - 90 = y * (y + 18) ∧ (∀ z : ℝ, 3 * z ^ 2 + 33 * z - 90 = z * (z + 18) → y ≤ z) ∧ y = -18 := 
sorry

end smallest_value_y_l1692_169234


namespace num_green_balls_l1692_169291

theorem num_green_balls (G : ℕ) (h : (3 * 2 : ℚ) / ((5 + G) * (4 + G)) = 1/12) : G = 4 :=
by
  sorry

end num_green_balls_l1692_169291


namespace encode_mathematics_l1692_169223

def robotCipherMapping : String → String := sorry

theorem encode_mathematics :
  robotCipherMapping "MATHEMATICS" = "2232331122323323132" := sorry

end encode_mathematics_l1692_169223


namespace unique_largest_negative_integer_l1692_169294

theorem unique_largest_negative_integer :
  ∃! x : ℤ, x = -1 ∧ (∀ y : ℤ, y < 0 → x ≥ y) :=
by
  sorry

end unique_largest_negative_integer_l1692_169294


namespace range_of_x_l1692_169293

theorem range_of_x (x p : ℝ) (h₀ : 0 ≤ p ∧ p ≤ 4) :
  x^2 + p * x > 4 * x + p - 3 → (x < 1 ∨ x > 3) :=
sorry

end range_of_x_l1692_169293


namespace point_on_y_axis_is_zero_l1692_169240

-- Given conditions
variables (m : ℝ) (y : ℝ)
-- \( P(m, 2) \) lies on the y-axis
def point_on_y_axis (m y : ℝ) : Prop := (m = 0)

-- Proof statement: Prove that if \( P(m, 2) \) lies on the y-axis, then \( m = 0 \)
theorem point_on_y_axis_is_zero (h : point_on_y_axis m 2) : m = 0 :=
by 
  -- the proof would go here
  sorry

end point_on_y_axis_is_zero_l1692_169240


namespace binary_mul_correct_l1692_169261

def bin_to_nat (l : List ℕ) : ℕ :=
  l.foldl (λ n b => 2 * n + b) 0

def p : List ℕ := [1,0,1,1,0,1]
def q : List ℕ := [1,1,0,1]
def r : List ℕ := [1,0,0,0,1,0,0,0,1,1]

theorem binary_mul_correct :
  bin_to_nat p * bin_to_nat q = bin_to_nat r := by
  sorry

end binary_mul_correct_l1692_169261


namespace european_fraction_is_one_fourth_l1692_169265

-- Define the total number of passengers
def P : ℕ := 108

-- Define the fractions and the number of passengers from each continent
def northAmerica := (1 / 12) * P
def africa := (1 / 9) * P
def asia := (1 / 6) * P
def otherContinents := 42

-- Define the total number of non-European passengers
def totalNonEuropean := northAmerica + africa + asia + otherContinents

-- Define the number of European passengers
def european := P - totalNonEuropean

-- Define the fraction of European passengers
def europeanFraction := european / P

-- Prove that the fraction of European passengers is 1/4
theorem european_fraction_is_one_fourth : europeanFraction = 1 / 4 := 
by
  unfold europeanFraction european totalNonEuropean northAmerica africa asia P
  sorry

end european_fraction_is_one_fourth_l1692_169265


namespace lisa_needs_change_probability_l1692_169237

theorem lisa_needs_change_probability :
  let quarters := 16
  let toy_prices := List.range' 2 10 |> List.map (fun n => n * 25) -- List of toy costs: (50,75,...,300)
  let favorite_toy_price := 275
  let factorial := Nat.factorial
  let favorable := (factorial 9) + 9 * (factorial 8)
  let total_permutations := factorial 10
  let p_no_change := (favorable.toFloat / total_permutations.toFloat) -- Convert to Float for probability calculations
  let p_change_needed := Float.round ((1.0 - p_no_change) * 100.0) / 100.0
  p_change_needed = 4.0 / 5.0 := sorry

end lisa_needs_change_probability_l1692_169237


namespace min_value_of_sum_squares_l1692_169200

theorem min_value_of_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) :
    x^2 + y^2 + z^2 ≥ 10 :=
sorry

end min_value_of_sum_squares_l1692_169200


namespace temperature_difference_l1692_169253

-- Define the temperatures given in the problem.
def T_noon : ℝ := 10
def T_midnight : ℝ := -150

-- State the theorem to prove the temperature difference.
theorem temperature_difference :
  T_noon - T_midnight = 160 :=
by
  -- We skip the proof and add sorry.
  sorry

end temperature_difference_l1692_169253


namespace ratio_of_diagonals_l1692_169283

theorem ratio_of_diagonals (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (4 * b) / (4 * a) = 11) : (b * Real.sqrt 2) / (a * Real.sqrt 2) = 11 := 
by 
  sorry

end ratio_of_diagonals_l1692_169283


namespace fill_question_mark_l1692_169248

def sudoku_grid : Type := 
  List (List (Option ℕ))

def initial_grid : sudoku_grid := 
  [ [some 3, none, none, none],
    [none, none, none, some 1], 
    [none, none, some 2, none], 
    [some 1, none, none, none] ]

def valid_sudoku (grid : sudoku_grid) : Prop :=
  -- Ensure the grid is a valid 4x4 Sudoku grid
  -- Adding necessary constraints for rows, columns and 2x2 subgrids.
  sorry

def solve_sudoku (grid : sudoku_grid) : sudoku_grid :=
  -- Function that solves the Sudoku (not implemented for this proof statement)
  sorry

theorem fill_question_mark : solve_sudoku initial_grid = 
  [ [some 3, some 2, none, none],
    [none, none, none, some 1], 
    [none, none, some 2, none], 
    [some 1, none, none, none] ] :=
  sorry

end fill_question_mark_l1692_169248


namespace max_projection_area_of_tetrahedron_l1692_169233

/-- 
Two adjacent faces of a tetrahedron are isosceles right triangles with a hypotenuse of 2,
and they form a dihedral angle of 60 degrees. The tetrahedron rotates around the common edge
of these faces. The maximum area of the projection of the rotating tetrahedron onto 
the plane containing the given edge is 1.
-/
theorem max_projection_area_of_tetrahedron (S hypotenuse dihedral max_proj_area : ℝ)
  (is_isosceles_right_triangle : ∀ (a b : ℝ), a^2 + b^2 = hypotenuse^2)
  (hypotenuse_len : hypotenuse = 2)
  (dihedral_angle : dihedral = 60) :
  max_proj_area = 1 :=
  sorry

end max_projection_area_of_tetrahedron_l1692_169233


namespace area_of_room_in_square_inches_l1692_169287

-- Defining the conversion from feet to inches
def feet_to_inches (ft : ℕ) : ℕ := ft * 12

-- Given conditions
def length_in_feet : ℕ := 10
def width_in_feet : ℕ := 10

-- Calculate length and width in inches
def length_in_inches := feet_to_inches length_in_feet
def width_in_inches := feet_to_inches width_in_feet

-- Calculate area in square inches
def area_in_square_inches := length_in_inches * width_in_inches

-- Theorem statement
theorem area_of_room_in_square_inches
  (h1 : length_in_feet = 10)
  (h2 : width_in_feet = 10)
  (conversion : feet_to_inches 1 = 12) :
  area_in_square_inches = 14400 :=
sorry

end area_of_room_in_square_inches_l1692_169287


namespace total_students_l1692_169215

theorem total_students (x : ℝ) :
  (x - (1/2)*x - (1/4)*x - (1/8)*x = 3) → x = 24 :=
by
  intro h
  sorry

end total_students_l1692_169215


namespace calculate_sequences_l1692_169226

-- Definitions of sequences and constants
def a (n : ℕ) := 2 * n + 1
def b (n : ℕ) := 3 ^ n
def S (n : ℕ) := n * (n + 2)
def T (n : ℕ) := (3 / 4 : ℚ) - (2 * n + 3) / (2 * (n + 1) * (n + 2))

-- Hypotheses and proofs
theorem calculate_sequences (d : ℕ) (a1 : ℕ) (h_d : d = 2) (h_a1 : a1 = 3) :
  ∀ n, (a n = 2 * n + 1) ∧ (b 1 = a 1) ∧ (b 2 = a 4) ∧ (b 3 = a 13) ∧ (b n = 3 ^ n) ∧
  (S n = n * (n + 2)) ∧ (T n = (3 / 4 : ℚ) - (2 * n + 3) / (2 * (n + 1) * (n + 2))) :=
by
  intros
  -- Skipping proof steps with sorry
  sorry

end calculate_sequences_l1692_169226


namespace find_value_of_expression_l1692_169229

theorem find_value_of_expression (x y : ℝ) (h1 : |x| = 2) (h2 : |y| = 3) (h3 : x / y < 0) :
  (2 * x - y = 7) ∨ (2 * x - y = -7) :=
by
  sorry

end find_value_of_expression_l1692_169229


namespace equal_share_payment_l1692_169219

theorem equal_share_payment (A B C : ℝ) (h : A < B) (h2 : B < C) :
  (B + C + (A + C - 2 * B) / 3) + (A + C - 2 * B / 3) = 2 * C - A - B / 3 :=
sorry

end equal_share_payment_l1692_169219


namespace number_of_buses_l1692_169288

theorem number_of_buses (total_students : ℕ) (students_per_bus : ℕ) (students_in_cars : ℕ) (buses : ℕ)
  (h1 : total_students = 375)
  (h2 : students_per_bus = 53)
  (h3 : students_in_cars = 4)
  (h4 : buses = (total_students - students_in_cars + students_per_bus - 1) / students_per_bus) :
  buses = 8 := by
  -- We will demonstrate that the number of buses indeed equals 8 under the given conditions.
  sorry

end number_of_buses_l1692_169288


namespace sequence_formula_l1692_169212

-- Define the problem when n >= 2
theorem sequence_formula (n : ℕ) (h : n ≥ 2) : 
  1 / (n^2 - 1) = (1 / 2) * (1 / (n - 1) - 1 / (n + 1)) := 
by {
  sorry
}

end sequence_formula_l1692_169212


namespace value_of_y_l1692_169246

theorem value_of_y (x y : ℤ) (h1 : x^2 = y - 8) (h2 : x = -7) : y = 57 :=
sorry

end value_of_y_l1692_169246


namespace hyperbola_asymptotes_n_l1692_169295

theorem hyperbola_asymptotes_n {y x : ℝ} (n : ℝ) (H : ∀ x y, (y^2 / 16) - (x^2 / 9) = 1 → y = n * x ∨ y = -n * x) : n = 4/3 :=
  sorry

end hyperbola_asymptotes_n_l1692_169295


namespace four_digit_numbers_permutations_l1692_169216

theorem four_digit_numbers_permutations (a b : ℕ) (h1 : a = 3) (h2 : b = 0) : 
  (if a = 3 ∧ b = 0 then 3 else 0) = 3 :=
by
  sorry

end four_digit_numbers_permutations_l1692_169216


namespace factorization_of_difference_of_squares_l1692_169252

theorem factorization_of_difference_of_squares (m : ℝ) : 
  m^2 - 16 = (m + 4) * (m - 4) := 
by 
  sorry

end factorization_of_difference_of_squares_l1692_169252


namespace number_added_is_8_l1692_169231

theorem number_added_is_8
  (x y : ℕ)
  (h1 : x = 265)
  (h2 : x / 5 + y = 61) :
  y = 8 :=
by
  sorry

end number_added_is_8_l1692_169231


namespace average_after_modifications_l1692_169227

theorem average_after_modifications (S : ℕ) (sum_initial : S = 1080)
  (sum_after_removals : S - 80 - 85 = 915)
  (sum_after_additions : 915 + 75 + 75 = 1065) :
  (1065 / 12 : ℚ) = 88.75 :=
by sorry

end average_after_modifications_l1692_169227


namespace problem_sequence_inequality_l1692_169269

def a (n : ℕ) : ℚ := 15 + (n - 1 : ℚ) * (-(2 / 3))

theorem problem_sequence_inequality :
  ∃ k : ℕ, (a k) * (a (k + 1)) < 0 ∧ k = 23 :=
by {
  use 23,
  sorry
}

end problem_sequence_inequality_l1692_169269


namespace geom_seq_find_b3_l1692_169296

-- Given conditions
def is_geometric_seq (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

def geom_seq_condition (b : ℕ → ℝ) : Prop :=
  is_geometric_seq b ∧ b 2 * b 3 * b 4 = 8

-- Proof statement: We need to prove that b 3 = 2
theorem geom_seq_find_b3 (b : ℕ → ℝ) (h : geom_seq_condition b) : b 3 = 2 :=
  sorry

end geom_seq_find_b3_l1692_169296


namespace smallest_five_digit_multiple_of_9_starting_with_7_l1692_169213

theorem smallest_five_digit_multiple_of_9_starting_with_7 :
  ∃ (n : ℕ), (70000 ≤ n ∧ n < 80000) ∧ (n % 9 = 0) ∧ n = 70002 :=
sorry

end smallest_five_digit_multiple_of_9_starting_with_7_l1692_169213


namespace find_m_l1692_169274

theorem find_m (m : ℝ) : (∀ x : ℝ, x - m > 5 ↔ x > 2) → m = -3 :=
by
  sorry

end find_m_l1692_169274


namespace paint_needed_for_snake_l1692_169279

open Nat

def total_paint (paint_per_segment segments additional_paint : Nat) : Nat :=
  paint_per_segment * segments + additional_paint

theorem paint_needed_for_snake :
  total_paint 240 336 20 = 80660 :=
by
  sorry

end paint_needed_for_snake_l1692_169279


namespace b_2023_equals_one_fifth_l1692_169258

theorem b_2023_equals_one_fifth (b : ℕ → ℚ) (h1 : b 1 = 4) (h2 : b 2 = 5)
    (h_rec : ∀ (n : ℕ), n ≥ 3 → b n = b (n - 1) / b (n - 2)) :
    b 2023 = 1 / 5 := by
  sorry

end b_2023_equals_one_fifth_l1692_169258


namespace harmonic_mean_pairs_count_l1692_169267

open Nat

theorem harmonic_mean_pairs_count :
  ∃! n : ℕ, (∀ x y : ℕ, x < y ∧ x > 0 ∧ y > 0 ∧ (2 * x * y) / (x + y) = 4^15 → n = 29) :=
sorry

end harmonic_mean_pairs_count_l1692_169267


namespace different_testing_methods_1_different_testing_methods_2_l1692_169282

-- Definitions used in Lean 4 statement should be derived from the conditions in a).
def total_products := 10
def defective_products := 4
def non_defective_products := total_products - defective_products
def choose (n k : Nat) : Nat := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Statement (1)
theorem different_testing_methods_1 :
  let first_defective := 5
  let last_defective := 10
  let non_defective_in_first_4 := choose 6 4
  let defective_in_middle_5 := choose 5 3
  let total_methods := non_defective_in_first_4 * defective_in_middle_5 * Nat.factorial 5 * Nat.factorial 4
  total_methods = 103680 := sorry

-- Statement (2)
theorem different_testing_methods_2 :
  let first_defective := 5
  let remaining_defective := 4
  let non_defective_in_first_4 := choose 6 4
  let total_methods := non_defective_in_first_4 * Nat.factorial 5
  total_methods = 576 := sorry

end different_testing_methods_1_different_testing_methods_2_l1692_169282


namespace range_of_a_l1692_169263

theorem range_of_a (a : ℝ) :
  (∀ x, (3 ≤ x → 2*a*x + 4 ≤ 2*a*(x+1) + 4) ∧ (2 < x ∧ x < 3 → (a + (2*a + 2)/(x-2) ≤ a + (2*a + 2)/(x-1))) ) →
  -1 < a ∧ a ≤ -2/3 :=
by
  intros h
  sorry

end range_of_a_l1692_169263


namespace solve_inequality_l1692_169251

theorem solve_inequality (x : ℝ) :
  (0 ≤ x^2 - x - 2 ∧ x^2 - x - 2 ≤ 4) ↔
  (-2 ≤ x ∧ x ≤ -1) ∨ (2 ≤ x ∧ x ≤ 3) :=
by sorry

end solve_inequality_l1692_169251


namespace units_digit_of_product_l1692_169290

-- Definitions for units digit patterns for powers of 5 and 7
def units_digit (n : ℕ) : ℕ := n % 10

def power5_units_digit := 5
def power7_units_cycle := [7, 9, 3, 1]

-- Statement of the problem
theorem units_digit_of_product :
  units_digit ((5 ^ 3) * (7 ^ 52)) = 5 :=
by
  sorry

end units_digit_of_product_l1692_169290


namespace part1_part2_l1692_169220

def f (x : ℝ) := |x + 4| - |x - 1|
def g (x : ℝ) := |2 * x - 1| + 3

theorem part1 (x : ℝ) : (f x > 3) → x > 0 :=
by sorry

theorem part2 (a : ℝ) : (∃ x, f x + 1 < 4^a - 5 * 2^a) ↔ (a < 0 ∨ a > 2) :=
by sorry

end part1_part2_l1692_169220


namespace Johnson_Smith_tied_end_May_l1692_169235

def home_runs_Johnson : List ℕ := [2, 12, 15, 8, 14, 11, 9, 16]
def home_runs_Smith : List ℕ := [5, 9, 10, 12, 15, 12, 10, 17]

def total_without_June (runs: List ℕ) : Nat := List.sum (runs.take 5 ++ runs.drop 5)
def estimated_June (total: Nat) : Nat := total / 8

theorem Johnson_Smith_tied_end_May :
  let total_Johnson := total_without_June home_runs_Johnson;
  let total_Smith := total_without_June home_runs_Smith;
  let estimated_June_Johnson := estimated_June total_Johnson;
  let estimated_June_Smith := estimated_June total_Smith;
  let total_with_June_Johnson := total_Johnson + estimated_June_Johnson;
  let total_with_June_Smith := total_Smith + estimated_June_Smith;
  (List.sum (home_runs_Johnson.take 5) = List.sum (home_runs_Smith.take 5)) :=
by
  sorry

end Johnson_Smith_tied_end_May_l1692_169235


namespace OBrien_current_hats_l1692_169222

-- Definition of the number of hats that Fire chief Simpson has
def Simpson_hats : ℕ := 15

-- Definition of the number of hats that Policeman O'Brien had before losing one
def OBrien_initial_hats (Simpson_hats : ℕ) : ℕ := 2 * Simpson_hats + 5

-- Final proof statement that Policeman O'Brien now has 34 hats
theorem OBrien_current_hats : OBrien_initial_hats Simpson_hats - 1 = 34 := by
  -- Proof will go here, but is skipped for now
  sorry

end OBrien_current_hats_l1692_169222


namespace water_evaporation_l1692_169206

theorem water_evaporation (m : ℝ) 
  (evaporation_day1 : m' = m * (0.1)) 
  (evaporation_day2 : m'' = (m * 0.9) * 0.1) 
  (total_evaporation : total = m' + m'')
  (water_added : 15 = total) 
  : m = 1500 / 19 := by
  sorry

end water_evaporation_l1692_169206


namespace solve_inequality_system_l1692_169230

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l1692_169230


namespace amount_of_p_l1692_169239

theorem amount_of_p (p q r : ℝ) (h1 : q = (1 / 6) * p) (h2 : r = (1 / 6) * p) 
  (h3 : p = (q + r) + 32) : p = 48 :=
by
  sorry

end amount_of_p_l1692_169239


namespace digits_of_2_pow_100_last_three_digits_of_2_pow_100_l1692_169272

-- Prove that 2^100 has 31 digits.
theorem digits_of_2_pow_100 : (10^30 ≤ 2^100) ∧ (2^100 < 10^31) :=
by
  sorry

-- Prove that the last three digits of 2^100 are 376.
theorem last_three_digits_of_2_pow_100 : 2^100 % 1000 = 376 :=
by
  sorry

end digits_of_2_pow_100_last_three_digits_of_2_pow_100_l1692_169272


namespace part_a_part_a_rev_l1692_169236

variable (x y : ℝ)

theorem part_a (hx : x > 0) (hy : y > 0) : x + y > |x - y| :=
sorry

theorem part_a_rev (h : x + y > |x - y|) : x > 0 ∧ y > 0 :=
sorry

end part_a_part_a_rev_l1692_169236


namespace solve_quadratic_eq_l1692_169285

theorem solve_quadratic_eq : (x : ℝ) → (x^2 - 4 = 0) → (x = 2 ∨ x = -2) :=
by
  sorry

end solve_quadratic_eq_l1692_169285


namespace total_charge_for_first_4_minutes_under_plan_A_is_0_60_l1692_169242

def planA_charges (X : ℝ) (minutes : ℕ) : ℝ :=
  if minutes <= 4 then X
  else X + (minutes - 4) * 0.06

def planB_charges (minutes : ℕ) : ℝ :=
  minutes * 0.08

theorem total_charge_for_first_4_minutes_under_plan_A_is_0_60
  (X : ℝ)
  (h : planA_charges X 18 = planB_charges 18) :
  X = 0.60 :=
by
  sorry

end total_charge_for_first_4_minutes_under_plan_A_is_0_60_l1692_169242


namespace sentence_structure_diff_l1692_169209

-- Definitions based on sentence structures.
def sentence_A := "得不焚，殆有神护者" -- passive
def sentence_B := "重为乡党所笑" -- passive
def sentence_C := "而文采不表于后也" -- post-positioned prepositional
def sentence_D := "是以见放" -- passive

-- Definition to check if the given sentence is passive
def is_passive (s : String) : Prop :=
  s = sentence_A ∨ s = sentence_B ∨ s = sentence_D

-- Definition to check if the given sentence is post-positioned prepositional
def is_post_positioned_prepositional (s : String) : Prop :=
  s = sentence_C

-- Theorem to prove
theorem sentence_structure_diff :
  (is_post_positioned_prepositional sentence_C) ∧ ¬(is_passive sentence_C) :=
by
  sorry

end sentence_structure_diff_l1692_169209


namespace amy_music_files_l1692_169260

-- Define the number of total files on the flash drive
def files_on_flash_drive := 48.0

-- Define the number of video files on the flash drive
def video_files := 21.0

-- Define the number of picture files on the flash drive
def picture_files := 23.0

-- Define the number of music files, derived from the conditions
def music_files := files_on_flash_drive - (video_files + picture_files)

-- The theorem we need to prove
theorem amy_music_files : music_files = 4.0 := by
  sorry

end amy_music_files_l1692_169260


namespace last_two_digits_2007_pow_20077_l1692_169224

theorem last_two_digits_2007_pow_20077 : (2007 ^ 20077) % 100 = 7 := 
by sorry

end last_two_digits_2007_pow_20077_l1692_169224


namespace common_property_of_rhombus_and_rectangle_l1692_169255

structure Rhombus :=
  (bisect_perpendicular : ∀ d₁ d₂ : ℝ, ∃ p : ℝ × ℝ, p = (0, 0))
  (diagonals_not_equal : ∀ d₁ d₂ : ℝ, ¬(d₁ = d₂))

structure Rectangle :=
  (bisect_each_other : ∀ d₁ d₂ : ℝ, ∃ p : ℝ × ℝ, p = (0, 0))
  (diagonals_equal : ∀ d₁ d₂ : ℝ, d₁ = d₂)

theorem common_property_of_rhombus_and_rectangle (R : Rhombus) (S : Rectangle) :
  ∀ d₁ d₂ : ℝ, ∃ p : ℝ × ℝ, p = (0, 0) :=
by
  -- Assuming the properties of Rhombus R and Rectangle S
  sorry

end common_property_of_rhombus_and_rectangle_l1692_169255


namespace find_n_l1692_169210

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (n : ℕ)

def isArithmeticSeq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sumTo (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

theorem find_n 
  (h_arith : isArithmeticSeq a)
  (h_a2 : a 2 = 2) 
  (h_S_diff : ∀ n, n > 3 → S n - S (n - 3) = 54)
  (h_Sn : S n = 100)
  : n = 10 := 
by
  sorry

end find_n_l1692_169210


namespace union_A_B_interval_l1692_169264

def setA (x : ℝ) : Prop := x ≥ -1
def setB (y : ℝ) : Prop := y ≥ 1

theorem union_A_B_interval :
  {x | setA x} ∪ {y | setB y} = {z : ℝ | z ≥ -1} :=
by
  sorry

end union_A_B_interval_l1692_169264


namespace marble_cut_in_third_week_l1692_169273

def percentage_cut_third_week := 
  let initial_weight : ℝ := 250 
  let final_weight : ℝ := 105
  let percent_cut_first_week : ℝ := 0.30
  let percent_cut_second_week : ℝ := 0.20
  let weight_after_first_week := initial_weight * (1 - percent_cut_first_week)
  let weight_after_second_week := weight_after_first_week * (1 - percent_cut_second_week)
  (weight_after_second_week - final_weight) / weight_after_second_week * 100 = 25

theorem marble_cut_in_third_week :
  percentage_cut_third_week = true :=
by
  sorry

end marble_cut_in_third_week_l1692_169273


namespace number_of_valid_pairs_is_343_l1692_169256

-- Define the given problem conditions
def given_number : Nat := 1003003001

-- Define the expression for LCM calculation
def LCM (x y : Nat) : Nat := (x * y) / (Nat.gcd x y)

-- Define the prime factorization of the given number
def is_prime_factorization_correct : Prop :=
  given_number = 7^3 * 11^3 * 13^3

-- Define x and y form as described
def is_valid_form (x y : Nat) : Prop :=
  ∃ (a b c d e f : ℕ), x = 7^a * 11^b * 13^c ∧ y = 7^d * 11^e * 13^f

-- Define the LCM condition for the ordered pairs
def meets_lcm_condition (x y : Nat) : Prop :=
  LCM x y = given_number

-- State the theorem to prove an equivalent problem
theorem number_of_valid_pairs_is_343 : is_prime_factorization_correct →
  (∃ (n : ℕ), n = 343 ∧ 
    (∀ (x y : ℕ), is_valid_form x y → meets_lcm_condition x y → x > 0 → y > 0 → True)
  ) :=
by
  intros h
  use 343
  sorry

end number_of_valid_pairs_is_343_l1692_169256


namespace find_x_l1692_169211

theorem find_x (x : ℝ) :
  (1 / 3) * ((3 * x + 4) + (7 * x - 5) + (4 * x + 9)) = (5 * x - 3) → x = 17 :=
by
  sorry

end find_x_l1692_169211


namespace sculpture_exposed_surface_area_l1692_169205

theorem sculpture_exposed_surface_area :
  let l₁ := 9
  let l₂ := 6
  let l₃ := 4
  let l₄ := 1

  let exposed_bottom_layer := 9 + 16
  let exposed_second_layer := 6 + 10
  let exposed_third_layer := 4 + 8
  let exposed_top_layer := 5

  l₁ + l₂ + l₃ + l₄ = 20 →
  exposed_bottom_layer + exposed_second_layer + exposed_third_layer + exposed_top_layer = 58 :=
by {
  sorry
}

end sculpture_exposed_surface_area_l1692_169205


namespace swim_back_distance_l1692_169281

variables (swimming_speed_still_water : ℝ) (water_speed : ℝ) (time_back : ℝ) (distance_back : ℝ)

theorem swim_back_distance :
  swimming_speed_still_water = 12 → 
  water_speed = 10 → 
  time_back = 4 →
  distance_back = (swimming_speed_still_water - water_speed) * time_back →
  distance_back = 8 :=
by
  intros swimming_speed_still_water_eq water_speed_eq time_back_eq distance_back_eq
  have swim_speed : (swimming_speed_still_water - water_speed) = 2 := by sorry
  rw [swim_speed, time_back_eq] at distance_back_eq
  sorry

end swim_back_distance_l1692_169281


namespace field_trip_cost_l1692_169232

def candy_bar_price : ℝ := 1.25
def candy_bars_sold : ℤ := 188
def money_from_grandma : ℝ := 250

theorem field_trip_cost : (candy_bars_sold * candy_bar_price + money_from_grandma) = 485 := 
by
  sorry

end field_trip_cost_l1692_169232


namespace original_team_players_l1692_169244

theorem original_team_players (n : ℕ) (W : ℝ)
    (h1 : W = n * 76)
    (h2 : (W + 110 + 60) / (n + 2) = 78) : n = 7 :=
  sorry

end original_team_players_l1692_169244
