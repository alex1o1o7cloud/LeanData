import Mathlib

namespace NUMINAMATH_GPT_range_of_a_for_empty_solution_set_l622_62204

theorem range_of_a_for_empty_solution_set : 
  (∀ a : ℝ, (∀ x : ℝ, |x - 4| + |3 - x| < a → false) ↔ a ≤ 1) := 
sorry

end NUMINAMATH_GPT_range_of_a_for_empty_solution_set_l622_62204


namespace NUMINAMATH_GPT_unit_triangle_count_bound_l622_62211

variable {L : ℝ} (L_pos : L > 0)
variable {n : ℕ}

/--
  Let \( \Delta \) be an equilateral triangle with side length \( L \), and suppose that \( n \) unit 
  equilateral triangles are drawn inside \( \Delta \) with non-overlapping interiors and each having 
  sides parallel to \( \Delta \) but with opposite orientation. Then,
  we must have \( n \leq \frac{2}{3} L^2 \).
-/
theorem unit_triangle_count_bound (L_pos : L > 0) (n : ℕ) :
  n ≤ (2 / 3) * (L ^ 2) := 
sorry

end NUMINAMATH_GPT_unit_triangle_count_bound_l622_62211


namespace NUMINAMATH_GPT_max_ab_bc_cd_l622_62214

theorem max_ab_bc_cd (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) (h_sum : a + b + c + d = 200) : 
    ab + bc + cd ≤ 10000 := by
  sorry

end NUMINAMATH_GPT_max_ab_bc_cd_l622_62214


namespace NUMINAMATH_GPT_exam_items_count_l622_62276

theorem exam_items_count (x : ℝ) (hLiza : Liza_correct = 0.9 * x) (hRoseCorrect : Rose_correct = 0.9 * x + 2) (hRoseTotal : Rose_total = x) (hRoseIncorrect : Rose_incorrect = x - (0.9 * x + 2) ):
    Liza_correct + Rose_incorrect = Rose_total :=
by
    sorry

end NUMINAMATH_GPT_exam_items_count_l622_62276


namespace NUMINAMATH_GPT_vacation_cost_l622_62241

theorem vacation_cost (C : ℝ)
  (h1 : C / 5 - C / 8 = 60) :
  C = 800 :=
sorry

end NUMINAMATH_GPT_vacation_cost_l622_62241


namespace NUMINAMATH_GPT_f_is_even_l622_62281

-- Given an odd function g
def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g (x)

-- Define the function f as given by the problem
def f (g : ℝ → ℝ) (x : ℝ) : ℝ :=
  abs (g (x^2))

-- The theorem stating that f is an even function
theorem f_is_even (g : ℝ → ℝ) (h_odd : is_odd_function g) : ∀ x, f g x = f g (-x) :=
by
  sorry

end NUMINAMATH_GPT_f_is_even_l622_62281


namespace NUMINAMATH_GPT_max_value_expression_l622_62217

theorem max_value_expression (a b c : ℝ) (h : a * b * c + a + c - b = 0) : 
  ∃ m, (m = (1/(1+a^2) - 1/(1+b^2) + 1/(1+c^2))) ∧ (m = 5 / 4) :=
by 
  sorry

end NUMINAMATH_GPT_max_value_expression_l622_62217


namespace NUMINAMATH_GPT_min_diff_between_y_and_x_l622_62275

theorem min_diff_between_y_and_x (x y z : ℤ)
    (h1 : x < y)
    (h2 : y < z)
    (h3 : Even x)
    (h4 : Odd y)
    (h5 : Odd z)
    (h6 : z - x = 9) :
    y - x = 1 := 
  by sorry

end NUMINAMATH_GPT_min_diff_between_y_and_x_l622_62275


namespace NUMINAMATH_GPT_MN_eq_l622_62226

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}
def operation (A B : Set ℕ) : Set ℕ := { x | x ∈ A ∪ B ∧ x ∉ A ∩ B }

theorem MN_eq : operation M N = {1, 4} :=
sorry

end NUMINAMATH_GPT_MN_eq_l622_62226


namespace NUMINAMATH_GPT_value_of_a_l622_62202

-- Define the three lines as predicates
def line1 (x y : ℝ) : Prop := x + y = 1
def line2 (x y : ℝ) : Prop := x - y = 1
def line3 (a x y : ℝ) : Prop := a * x + y = 1

-- Define the condition that the lines do not form a triangle
def lines_do_not_form_triangle (a x y : ℝ) : Prop :=
  (∀ x y, line1 x y → ¬line3 a x y) ∨
  (∀ x y, line2 x y → ¬line3 a x y) ∨
  (a = 1)

theorem value_of_a (a : ℝ) :
  (¬ ∃ x y, line1 x y ∧ line2 x y ∧ line3 a x y) →
  lines_do_not_form_triangle a 1 0 →
  a = -1 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_value_of_a_l622_62202


namespace NUMINAMATH_GPT_sum_interior_angles_convex_polygon_number_of_triangles_convex_polygon_l622_62220

-- Define a convex n-gon and prove that the sum of its interior angles is (n-2) * 180 degrees
theorem sum_interior_angles_convex_polygon (n : ℕ) (h : 3 ≤ n) :
  ∃ (sum_of_angles : ℝ), sum_of_angles = (n-2) * 180 :=
sorry

-- Define a convex n-gon and prove that the number of triangles formed by dividing with non-intersecting diagonals is n-2
theorem number_of_triangles_convex_polygon (n : ℕ) (h : 3 ≤ n) :
  ∃ (num_of_triangles : ℕ), num_of_triangles = n-2 :=
sorry

end NUMINAMATH_GPT_sum_interior_angles_convex_polygon_number_of_triangles_convex_polygon_l622_62220


namespace NUMINAMATH_GPT_area_of_fourth_square_l622_62259

open Real

theorem area_of_fourth_square
  (EF FG GH : ℝ)
  (hEF : EF = 5)
  (hFG : FG = 7)
  (hGH : GH = 8) :
  let EG := sqrt (EF^2 + FG^2)
  let EH := sqrt (EG^2 + GH^2)
  EH^2 = 138 :=
by
  sorry

end NUMINAMATH_GPT_area_of_fourth_square_l622_62259


namespace NUMINAMATH_GPT_max_sector_area_l622_62289

theorem max_sector_area (r l : ℝ) (hp : 2 * r + l = 40) : (1 / 2) * l * r ≤ 100 := 
by
  sorry

end NUMINAMATH_GPT_max_sector_area_l622_62289


namespace NUMINAMATH_GPT_average_marks_in_6_subjects_l622_62268

/-- The average marks Ashok secured in 6 subjects is 72
Given:
1. The average of marks in 5 subjects is 74.
2. Ashok secured 62 marks in the 6th subject.
-/
theorem average_marks_in_6_subjects (avg_5 : ℕ) (marks_6th : ℕ) (h_avg_5 : avg_5 = 74) (h_marks_6th : marks_6th = 62) : 
  ((avg_5 * 5 + marks_6th) / 6) = 72 :=
  by
  sorry

end NUMINAMATH_GPT_average_marks_in_6_subjects_l622_62268


namespace NUMINAMATH_GPT_remaining_miles_to_be_built_l622_62219

-- Definitions from problem conditions
def current_length : ℕ := 200
def target_length : ℕ := 650
def first_day_miles : ℕ := 50
def second_day_miles : ℕ := 3 * first_day_miles

-- Lean theorem statement
theorem remaining_miles_to_be_built : 
  (target_length - current_length) - (first_day_miles + second_day_miles) = 250 := 
by 
  sorry

end NUMINAMATH_GPT_remaining_miles_to_be_built_l622_62219


namespace NUMINAMATH_GPT_num_dogs_with_spots_l622_62212

variable (D P : ℕ)

theorem num_dogs_with_spots (h1 : D / 2 = D / 2) (h2 : D / 5 = P) : (5 * P) / 2 = D / 2 := 
by
  have h3 : 5 * P = D := by
    sorry
  have h4 : (5 * P) / 2 = D / 2 := by
    rw [h3]
  exact h4

end NUMINAMATH_GPT_num_dogs_with_spots_l622_62212


namespace NUMINAMATH_GPT_mary_money_left_l622_62290

def initial_amount : Float := 150
def game_cost : Float := 60
def discount_percent : Float := 15 / 100
def remaining_percent_for_goggles : Float := 20 / 100
def tax_on_goggles : Float := 8 / 100

def money_left_after_shopping_trip (initial_amount : Float) (game_cost : Float) (discount_percent : Float) (remaining_percent_for_goggles : Float) (tax_on_goggles : Float) : Float :=
  let discount := game_cost * discount_percent
  let discounted_price := game_cost - discount
  let remainder_after_game := initial_amount - discounted_price
  let goggles_cost_before_tax := remainder_after_game * remaining_percent_for_goggles
  let tax := goggles_cost_before_tax * tax_on_goggles
  let final_goggles_cost := goggles_cost_before_tax + tax
  let remainder_after_goggles := remainder_after_game - final_goggles_cost
  remainder_after_goggles

#eval money_left_after_shopping_trip initial_amount game_cost discount_percent remaining_percent_for_goggles tax_on_goggles -- expected: 77.62

theorem mary_money_left (initial_amount game_cost discount_percent remaining_percent_for_goggles tax_on_goggles : Float) : 
  money_left_after_shopping_trip initial_amount game_cost discount_percent remaining_percent_for_goggles tax_on_goggles = 77.62 :=
by sorry

end NUMINAMATH_GPT_mary_money_left_l622_62290


namespace NUMINAMATH_GPT_tangent_line_equation_l622_62270

theorem tangent_line_equation :
  ∀ (x : ℝ) (y : ℝ), y = 4 * x - x^3 → 
  (x = -1) → (y = -3) →
  (∀ (m : ℝ), m = 4 - 3 * (-1)^2) →
  ∃ (line_eq : ℝ → ℝ), (∀ x, line_eq x = x - 2) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_equation_l622_62270


namespace NUMINAMATH_GPT_solution_set_of_inequality_l622_62253

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_tangent : ∀ x₀ y₀, y₀ = f x₀ → (∀ x, f x = y₀ + (3*x₀^2 - 6*x₀)*(x - x₀)))
  (h_at_3 : f 3 = 0) :
  {x : ℝ | ((x - 1) / f x) ≥ 0} = {x : ℝ | x < 0} ∪ {x : ℝ | 0 < x ∧ x ≤ 1} ∪ {x : ℝ | x > 3} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l622_62253


namespace NUMINAMATH_GPT_simple_interest_rate_l622_62266

def principal : ℕ := 600
def amount : ℕ := 950
def time : ℕ := 5
def expected_rate : ℚ := 11.67

theorem simple_interest_rate (P A T : ℕ) (R : ℚ) :
  P = principal → A = amount → T = time → R = expected_rate →
  (A = P + P * R * T / 100) :=
by
  intros hP hA hT hR
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l622_62266


namespace NUMINAMATH_GPT_max_principals_and_assistant_principals_l622_62288

theorem max_principals_and_assistant_principals : 
  ∀ (years term_principal term_assistant), (years = 10) ∧ (term_principal = 3) ∧ (term_assistant = 2) 
  → ∃ n, n = 9 :=
by
  sorry

end NUMINAMATH_GPT_max_principals_and_assistant_principals_l622_62288


namespace NUMINAMATH_GPT_find_values_of_a_l622_62229

theorem find_values_of_a :
  ∃ (a : ℝ), 
    (∀ x y, (|y + 2| + |x - 11| - 3) * (x^2 + y^2 - 13) = 0 ∧ 
             (x - 5)^2 + (y + 2)^2 = a) ↔ 
    a = 9 ∨ a = 42 + 2 * Real.sqrt 377 :=
sorry

end NUMINAMATH_GPT_find_values_of_a_l622_62229


namespace NUMINAMATH_GPT_seventh_grade_caps_collection_l622_62271

theorem seventh_grade_caps_collection (A B C : ℕ) 
  (h1 : A = (B + C) / 2) 
  (h2 : B = (A + C) / 3)
  (h3 : C = 150) : A + B + C = 360 := 
by 
  sorry

end NUMINAMATH_GPT_seventh_grade_caps_collection_l622_62271


namespace NUMINAMATH_GPT_product_gcd_lcm_l622_62243

-- Define the numbers
def a : ℕ := 24
def b : ℕ := 60

-- Define the gcd and lcm
def gcd_ab := Nat.gcd a b
def lcm_ab := Nat.lcm a b

-- Statement to prove: the product of gcd and lcm of 24 and 60 equals 1440
theorem product_gcd_lcm : gcd_ab * lcm_ab = 1440 := by
  -- gcd_ab = 12
  -- lcm_ab = 120
  -- Thus, 12 * 120 = 1440
  sorry

end NUMINAMATH_GPT_product_gcd_lcm_l622_62243


namespace NUMINAMATH_GPT_martha_total_cost_l622_62282

def weight_cheese : ℝ := 1.5
def weight_meat : ℝ := 0.55    -- converting grams to kg
def weight_pasta : ℝ := 0.28   -- converting grams to kg
def weight_tomatoes : ℝ := 2.2

def price_cheese_per_kg : ℝ := 6.30
def price_meat_per_kg : ℝ := 8.55
def price_pasta_per_kg : ℝ := 2.40
def price_tomatoes_per_kg : ℝ := 1.79

def tax_cheese : ℝ := 0.07
def tax_meat : ℝ := 0.06
def tax_pasta : ℝ := 0.08
def tax_tomatoes : ℝ := 0.05

def total_cost : ℝ :=
  let cost_cheese := weight_cheese * price_cheese_per_kg * (1 + tax_cheese)
  let cost_meat := weight_meat * price_meat_per_kg * (1 + tax_meat)
  let cost_pasta := weight_pasta * price_pasta_per_kg * (1 + tax_pasta)
  let cost_tomatoes := weight_tomatoes * price_tomatoes_per_kg * (1 + tax_tomatoes)
  cost_cheese + cost_meat + cost_pasta + cost_tomatoes

theorem martha_total_cost : total_cost = 19.9568 := by
  sorry

end NUMINAMATH_GPT_martha_total_cost_l622_62282


namespace NUMINAMATH_GPT_turtles_received_l622_62285

theorem turtles_received (martha_turtles : ℕ) (marion_turtles : ℕ) (h1 : martha_turtles = 40) 
    (h2 : marion_turtles = martha_turtles + 20) : martha_turtles + marion_turtles = 100 := 
by {
    sorry
}

end NUMINAMATH_GPT_turtles_received_l622_62285


namespace NUMINAMATH_GPT_sock_pairs_proof_l622_62295

noncomputable def numPairsOfSocks : ℕ :=
  let n : ℕ := sorry
  n

theorem sock_pairs_proof : numPairsOfSocks = 6 := by
  sorry

end NUMINAMATH_GPT_sock_pairs_proof_l622_62295


namespace NUMINAMATH_GPT_solve_for_x_l622_62231

-- Define the conditions as mathematical statements in Lean
def conditions (x y : ℝ) : Prop :=
  (2 * x - 3 * y = 10) ∧ (y = -x)

-- State the theorem that needs to be proven
theorem solve_for_x : ∃ x : ℝ, ∃ y : ℝ, conditions x y ∧ x = 2 :=
by 
  -- Provide a sketch of the proof to show that the statement is well-formed
  sorry

end NUMINAMATH_GPT_solve_for_x_l622_62231


namespace NUMINAMATH_GPT_cara_age_is_40_l622_62218

-- Defining the conditions
def grandmother_age : ℕ := 75
def mom_age : ℕ := grandmother_age - 15
def cara_age : ℕ := mom_age - 20

-- Proving the question
theorem cara_age_is_40 : cara_age = 40 := by
  sorry

end NUMINAMATH_GPT_cara_age_is_40_l622_62218


namespace NUMINAMATH_GPT_analytical_expression_of_C3_l622_62293

def C1 (x : ℝ) : ℝ := x^2 - 2*x + 3
def C2 (x : ℝ) : ℝ := C1 (x + 1)
def C3 (x : ℝ) : ℝ := C2 (-x)

theorem analytical_expression_of_C3 :
  ∀ x, C3 x = x^2 + 2 := by
  sorry

end NUMINAMATH_GPT_analytical_expression_of_C3_l622_62293


namespace NUMINAMATH_GPT_tony_average_time_l622_62222

-- Definitions based on the conditions
def distance_to_store : ℕ := 4 -- in miles
def walking_speed : ℕ := 2 -- in MPH
def running_speed : ℕ := 10 -- in MPH

-- Conditions
def time_walking : ℕ := (distance_to_store / walking_speed) * 60 -- in minutes
def time_running : ℕ := (distance_to_store / running_speed) * 60 -- in minutes

def total_time : ℕ := time_walking + 2 * time_running -- Total time spent in minutes
def number_of_days : ℕ := 3 -- Number of days

def average_time : ℕ := total_time / number_of_days -- Average time in minutes

-- Statement to prove
theorem tony_average_time : average_time = 56 := by 
  sorry

end NUMINAMATH_GPT_tony_average_time_l622_62222


namespace NUMINAMATH_GPT_value_of_first_equation_l622_62236

theorem value_of_first_equation (x y a : ℝ) 
  (h₁ : 2 * x + y = a) 
  (h₂ : x + 2 * y = 10) 
  (h₃ : (x + y) / 3 = 4) : 
  a = 12 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_first_equation_l622_62236


namespace NUMINAMATH_GPT_total_number_of_animals_l622_62261

-- Prove that the total number of animals is 300 given the conditions described.
theorem total_number_of_animals (A : ℕ) (H₁ : 4 * (A / 3) = 400) : A = 300 :=
sorry

end NUMINAMATH_GPT_total_number_of_animals_l622_62261


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l622_62240

variable {x k : ℝ}

def p (x k : ℝ) : Prop := x ≥ k
def q (x : ℝ) : Prop := (2 - x) / (x + 1) < 0

theorem sufficient_but_not_necessary_condition (h_suff : ∀ x, p x k → q x) (h_not_necessary : ∃ x, q x ∧ ¬p x k) : k > 2 :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l622_62240


namespace NUMINAMATH_GPT_chip_placement_count_l622_62292

def grid := Fin 4 × Fin 3

def grid_positions (n : Nat) := {s : Finset grid // s.card = n}

def no_direct_adjacency (positions : Finset grid) : Prop :=
  ∀ (x y : grid), x ∈ positions → y ∈ positions →
  (x.fst ≠ y.fst ∨ x.snd ≠ y.snd)

noncomputable def count_valid_placements : Nat :=
  -- Function to count valid placements
  sorry

theorem chip_placement_count :
  count_valid_placements = 4 :=
  sorry

end NUMINAMATH_GPT_chip_placement_count_l622_62292


namespace NUMINAMATH_GPT_find_integer_n_l622_62267

theorem find_integer_n : ∃ (n : ℤ), 0 ≤ n ∧ n < 23 ∧ 54126 % 23 = n :=
by
  use 13
  sorry

end NUMINAMATH_GPT_find_integer_n_l622_62267


namespace NUMINAMATH_GPT_bono_jelly_beans_l622_62233

variable (t A B C : ℤ)

theorem bono_jelly_beans (h₁ : A + B = 6 * t + 3) 
                         (h₂ : A + C = 4 * t + 5) 
                         (h₃ : B + C = 6 * t) : 
                         B = 4 * t - 1 := by
  sorry

end NUMINAMATH_GPT_bono_jelly_beans_l622_62233


namespace NUMINAMATH_GPT_arrangement_non_adjacent_l622_62208

theorem arrangement_non_adjacent :
  let total_arrangements := Nat.factorial 30
  let adjacent_arrangements := 2 * Nat.factorial 29
  let non_adjacent_arrangements := total_arrangements - adjacent_arrangements
  non_adjacent_arrangements = 28 * Nat.factorial 29 :=
by
  sorry

end NUMINAMATH_GPT_arrangement_non_adjacent_l622_62208


namespace NUMINAMATH_GPT_tan_alpha_l622_62200

theorem tan_alpha (α : ℝ) (h1 : Real.sin (Real.pi - α) = 3/5) (h2 : Real.pi / 2 < α ∧ α < Real.pi) : Real.tan α = -3/4 := 
  sorry

end NUMINAMATH_GPT_tan_alpha_l622_62200


namespace NUMINAMATH_GPT_number_of_consecutive_sum_sets_eq_18_l622_62251

theorem number_of_consecutive_sum_sets_eq_18 :
  ∃! (S : ℕ → ℕ) (n a : ℕ), (n ≥ 2) ∧ (S n = (n * (2 * a + n - 1)) / 2) ∧ (S n = 18) :=
sorry

end NUMINAMATH_GPT_number_of_consecutive_sum_sets_eq_18_l622_62251


namespace NUMINAMATH_GPT_prime_divisor_exponent_l622_62213

theorem prime_divisor_exponent (a n : ℕ) (p : ℕ) 
    (ha : a ≥ 2)
    (hn : n ≥ 1) 
    (hp : Nat.Prime p) 
    (hdiv : p ∣ a^(2^n) + 1) :
    2^(n+1) ∣ (p-1) :=
by
  sorry

end NUMINAMATH_GPT_prime_divisor_exponent_l622_62213


namespace NUMINAMATH_GPT_not_divisible_l622_62274

theorem not_divisible (n k : ℕ) : ¬ (5 ^ n + 1) ∣ (5 ^ k - 1) :=
sorry

end NUMINAMATH_GPT_not_divisible_l622_62274


namespace NUMINAMATH_GPT_frequency_of_rolling_six_is_0_point_19_l622_62232

theorem frequency_of_rolling_six_is_0_point_19 :
  ∀ (total_rolls number_six_appeared : ℕ), total_rolls = 100 → number_six_appeared = 19 → 
  (number_six_appeared : ℝ) / (total_rolls : ℝ) = 0.19 := 
by 
  intros total_rolls number_six_appeared h_total_rolls h_number_six_appeared
  sorry

end NUMINAMATH_GPT_frequency_of_rolling_six_is_0_point_19_l622_62232


namespace NUMINAMATH_GPT_value_of_x_l622_62286

theorem value_of_x (x y : ℝ) (h₁ : x = y - 0.10 * y) (h₂ : y = 125 + 0.10 * 125) : x = 123.75 := 
by
  sorry

end NUMINAMATH_GPT_value_of_x_l622_62286


namespace NUMINAMATH_GPT_smallest_multiple_of_36_with_digit_product_divisible_by_9_l622_62287

theorem smallest_multiple_of_36_with_digit_product_divisible_by_9 :
  ∃ n : ℕ, n > 0 ∧ n % 36 = 0 ∧ (∃ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 ∧ (d1 * d2 * d3) % 9 = 0) ∧ n = 936 := 
by
  sorry

end NUMINAMATH_GPT_smallest_multiple_of_36_with_digit_product_divisible_by_9_l622_62287


namespace NUMINAMATH_GPT_a4_minus_1_divisible_5_l622_62296

theorem a4_minus_1_divisible_5 (a : ℤ) (h : ¬ (∃ k : ℤ, a = 5 * k)) : 
  (a^4 - 1) % 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_a4_minus_1_divisible_5_l622_62296


namespace NUMINAMATH_GPT_soccer_ball_cost_l622_62209

theorem soccer_ball_cost (F S : ℝ) 
  (h1 : 3 * F + S = 155) 
  (h2 : 2 * F + 3 * S = 220) :
  S = 50 := 
sorry

end NUMINAMATH_GPT_soccer_ball_cost_l622_62209


namespace NUMINAMATH_GPT_cake_area_l622_62256

theorem cake_area (n : ℕ) (a area_per_piece : ℕ) 
  (h1 : n = 25) 
  (h2 : a = 16) 
  (h3 : area_per_piece = 4 * 4) 
  (h4 : a = area_per_piece) : 
  n * a = 400 := 
by
  sorry

end NUMINAMATH_GPT_cake_area_l622_62256


namespace NUMINAMATH_GPT_twenty_five_percent_of_five_hundred_l622_62223

theorem twenty_five_percent_of_five_hundred : 0.25 * 500 = 125 := 
by 
  sorry

end NUMINAMATH_GPT_twenty_five_percent_of_five_hundred_l622_62223


namespace NUMINAMATH_GPT_check_correct_conditional_expression_l622_62215
-- importing the necessary library for basic algebraic constructions and predicates

-- defining a predicate to denote the symbolic representation of conditional expressions validity
def valid_conditional_expression (expr: String) : Prop :=
  expr = "x <> 1" ∨ expr = "x > 1" ∨ expr = "x >= 1" ∨ expr = "x < 1" ∨ expr = "x <= 1" ∨ expr = "x = 1"

-- theorem to check for the valid conditional expression among the given options
theorem check_correct_conditional_expression :
  (valid_conditional_expression "1 < x < 2") = false ∧ 
  (valid_conditional_expression "x > < 1") = false ∧ 
  (valid_conditional_expression "x <> 1") = true ∧ 
  (valid_conditional_expression "x ≤ 1") = true :=
by sorry

end NUMINAMATH_GPT_check_correct_conditional_expression_l622_62215


namespace NUMINAMATH_GPT_goldfish_initial_count_l622_62255

theorem goldfish_initial_count (catsfish : ℕ) (fish_left : ℕ) (fish_disappeared : ℕ) (goldfish_initial : ℕ) :
  catsfish = 12 →
  fish_left = 15 →
  fish_disappeared = 4 →
  goldfish_initial = (fish_left + fish_disappeared) - catsfish →
  goldfish_initial = 7 :=
by
  intros h1 h2 h3 h4
  rw [h2, h3, h1] at h4
  exact h4

end NUMINAMATH_GPT_goldfish_initial_count_l622_62255


namespace NUMINAMATH_GPT_how_many_years_older_is_a_than_b_l622_62203

variable (a b c : ℕ)

theorem how_many_years_older_is_a_than_b
  (hb : b = 4)
  (hc : c = b / 2)
  (h_ages_sum : a + b + c = 12) :
  a - b = 2 := by
  sorry

end NUMINAMATH_GPT_how_many_years_older_is_a_than_b_l622_62203


namespace NUMINAMATH_GPT_no_descending_multiple_of_111_l622_62239

theorem no_descending_multiple_of_111 :
  ¬ ∃ (n : ℕ), (∀ (i j : ℕ), (i < j ∧ (n / 10^i % 10) < (n / 10^j % 10)) ∨ (i = j)) ∧ 111 ∣ n :=
by
  sorry

end NUMINAMATH_GPT_no_descending_multiple_of_111_l622_62239


namespace NUMINAMATH_GPT_calculate_value_is_neg_seventeen_l622_62201

theorem calculate_value_is_neg_seventeen : -3^2 + (-2)^3 = -17 :=
by
  sorry

end NUMINAMATH_GPT_calculate_value_is_neg_seventeen_l622_62201


namespace NUMINAMATH_GPT_fish_kept_l622_62234

theorem fish_kept (Leo_caught Agrey_more Sierra_more Leo_fish Returned : ℕ) 
                  (Agrey_caught : Agrey_more = 20) 
                  (Sierra_caught : Sierra_more = 15) 
                  (Leo_caught_cond : Leo_fish = 40) 
                  (Returned_cond : Returned = 30) : 
                  (Leo_fish + (Leo_fish + Agrey_more) + ((Leo_fish + Agrey_more) + Sierra_more) - Returned) = 145 :=
by
  sorry

end NUMINAMATH_GPT_fish_kept_l622_62234


namespace NUMINAMATH_GPT_billiard_expected_reflections_l622_62237

noncomputable def expected_reflections : ℝ :=
  (2 / Real.pi) * (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4))

theorem billiard_expected_reflections :
  expected_reflections = (2 / Real.pi) * (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_billiard_expected_reflections_l622_62237


namespace NUMINAMATH_GPT_paint_leftover_l622_62224

theorem paint_leftover (containers total_walls tiles_wall paint_ceiling : ℕ) 
  (h_containers : containers = 16) 
  (h_total_walls : total_walls = 4) 
  (h_tiles_wall : tiles_wall = 1) 
  (h_paint_ceiling : paint_ceiling = 1) : 
  containers - ((total_walls - tiles_wall) * (containers / total_walls)) - paint_ceiling = 3 :=
by 
  sorry

end NUMINAMATH_GPT_paint_leftover_l622_62224


namespace NUMINAMATH_GPT_spade_5_7_8_l622_62247

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_5_7_8 : spade 5 (spade 7 8) = -200 :=
by
  sorry

end NUMINAMATH_GPT_spade_5_7_8_l622_62247


namespace NUMINAMATH_GPT_initial_elephants_l622_62262

theorem initial_elephants (E : ℕ) :
  (E + 35 + 135 + 125 = 315) → (5 * 35 / 7 = 25) → (5 * 25 = 125) → (135 = 125 + 10) →
  E = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_initial_elephants_l622_62262


namespace NUMINAMATH_GPT_students_passing_course_l622_62280

theorem students_passing_course :
  let students_three_years_ago := 200
  let increase_factor := 1.5
  let students_two_years_ago := students_three_years_ago * increase_factor
  let students_last_year := students_two_years_ago * increase_factor
  let students_this_year := students_last_year * increase_factor
  students_this_year = 675 :=
by
  sorry

end NUMINAMATH_GPT_students_passing_course_l622_62280


namespace NUMINAMATH_GPT_range_of_m_l622_62254

theorem range_of_m (m : ℝ) : 
  (∀ x, x^2 + 2 * x - m > 0 ↔ (x = 1 → x^2 + 2 * x - m ≤ 0) ∧ (x = 2 → x^2 + 2 * x - m > 0)) ↔ (3 ≤ m ∧ m < 8) := 
sorry

end NUMINAMATH_GPT_range_of_m_l622_62254


namespace NUMINAMATH_GPT_initial_position_is_minus_one_l622_62250

def initial_position_of_A (A B C : ℤ) : Prop :=
  B = A - 3 ∧ C = B + 5 ∧ C = 1 ∧ A = -1

theorem initial_position_is_minus_one (A B C : ℤ) (h1 : B = A - 3) (h2 : C = B + 5) (h3 : C = 1) : A = -1 :=
  by sorry

end NUMINAMATH_GPT_initial_position_is_minus_one_l622_62250


namespace NUMINAMATH_GPT_mike_planted_50_l622_62205

-- Definitions for conditions
def mike_morning (M : ℕ) := M
def ted_morning (M : ℕ) := 2 * M
def mike_afternoon := 60
def ted_afternoon := 40
def total_planted (M : ℕ) := mike_morning M + ted_morning M + mike_afternoon + ted_afternoon

-- Statement to prove
theorem mike_planted_50 (M : ℕ) (h : total_planted M = 250) : M = 50 :=
by
  sorry

end NUMINAMATH_GPT_mike_planted_50_l622_62205


namespace NUMINAMATH_GPT_inequality_reciprocal_l622_62235

theorem inequality_reciprocal (a b : Real) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) : (1 / a < 1 / b) :=
by
  sorry

end NUMINAMATH_GPT_inequality_reciprocal_l622_62235


namespace NUMINAMATH_GPT_smallest_positive_multiple_of_32_l622_62299

theorem smallest_positive_multiple_of_32 : ∃ (n : ℕ), n > 0 ∧ ∃ k : ℕ, k > 0 ∧ n = 32 * k ∧ n = 32 := by
  use 32
  constructor
  · exact Nat.zero_lt_succ 31
  · use 1
    constructor
    · exact Nat.zero_lt_succ 0
    · constructor
      · rfl
      · rfl

end NUMINAMATH_GPT_smallest_positive_multiple_of_32_l622_62299


namespace NUMINAMATH_GPT_triangle_right_angle_l622_62245

theorem triangle_right_angle (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : γ = α + β) : γ = 90 :=
by
  sorry

end NUMINAMATH_GPT_triangle_right_angle_l622_62245


namespace NUMINAMATH_GPT_problem_l622_62279

open Real

noncomputable def f (x : ℝ) : ℝ := log x / log 2

theorem problem (f : ℝ → ℝ) (h : ∀ (x y : ℝ), f (x * y) = f x + f y) : 
  (∀ x : ℝ, f x = log x / log 2) :=
sorry

end NUMINAMATH_GPT_problem_l622_62279


namespace NUMINAMATH_GPT_spadesuit_calculation_l622_62284

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_calculation : spadesuit 5 (spadesuit 3 2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_spadesuit_calculation_l622_62284


namespace NUMINAMATH_GPT_prime_digit_B_l622_62272

-- Mathematical description
def six_digit_form (B : Nat) : Nat := 3 * 10^5 + 0 * 10^4 + 3 * 10^3 + 7 * 10^2 + 0 * 10^1 + B

-- Prime condition
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

theorem prime_digit_B (B : Nat) : is_prime (six_digit_form B) ↔ B = 3 :=
sorry

end NUMINAMATH_GPT_prime_digit_B_l622_62272


namespace NUMINAMATH_GPT_arithmetic_sequence_a10_l622_62242

variable {a : ℕ → ℝ}

-- Given the sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, (n < m) → (a (m + 1) - a m = a (n + 1) - a n)

-- Conditions
theorem arithmetic_sequence_a10 (h_arith : is_arithmetic_sequence a) 
                                (h1 : a 6 + a 8 = 16)
                                (h2 : a 4 = 1) :
  a 10 = 15 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a10_l622_62242


namespace NUMINAMATH_GPT_boat_speed_still_water_l622_62246

theorem boat_speed_still_water (downstream_speed upstream_speed : ℝ) (h1 : downstream_speed = 16) (h2 : upstream_speed = 9) : 
  (downstream_speed + upstream_speed) / 2 = 12.5 := 
by
  -- conditions explicitly stated above
  sorry

end NUMINAMATH_GPT_boat_speed_still_water_l622_62246


namespace NUMINAMATH_GPT_cone_volume_increase_l622_62252

open Real

noncomputable def cone_volume (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h
noncomputable def new_height (h : ℝ) : ℝ := 2 * h
noncomputable def new_volume (r h : ℝ) : ℝ := cone_volume r (new_height h)

theorem cone_volume_increase (r h : ℝ) : new_volume r h = 2 * (cone_volume r h) :=
by
  sorry

end NUMINAMATH_GPT_cone_volume_increase_l622_62252


namespace NUMINAMATH_GPT_total_games_for_18_players_l622_62273

-- Define the number of players
def num_players : ℕ := 18

-- Define the function to calculate total number of games
def total_games (n : ℕ) : ℕ := n * (n - 1) * 2

-- Theorem statement asserting the total number of games for 18 players
theorem total_games_for_18_players : total_games num_players = 612 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_games_for_18_players_l622_62273


namespace NUMINAMATH_GPT_probability_target_A_destroyed_probability_exactly_one_target_destroyed_l622_62258

-- Definition of probabilities
def prob_A_hits_target_A := 1 / 2
def prob_A_hits_target_B := 1 / 2
def prob_B_hits_target_A := 1 / 3
def prob_B_hits_target_B := 2 / 5

-- The event of target A being destroyed
def prob_target_A_destroyed := prob_A_hits_target_A * prob_B_hits_target_A

-- The event of target B being destroyed
def prob_target_B_destroyed := prob_A_hits_target_B * prob_B_hits_target_B

-- Complementary events
def prob_target_A_not_destroyed := 1 - prob_target_A_destroyed
def prob_target_B_not_destroyed := 1 - prob_target_B_destroyed

-- Exactly one target being destroyed
def prob_exactly_one_target_destroyed := 
  (prob_target_A_destroyed * prob_target_B_not_destroyed) +
  (prob_target_B_destroyed * prob_target_A_not_destroyed)

theorem probability_target_A_destroyed : prob_target_A_destroyed = 1 / 6 := by
  -- Proof needed here
  sorry

theorem probability_exactly_one_target_destroyed : prob_exactly_one_target_destroyed = 3 / 10 := by
  -- Proof needed here
  sorry

end NUMINAMATH_GPT_probability_target_A_destroyed_probability_exactly_one_target_destroyed_l622_62258


namespace NUMINAMATH_GPT_solve_a_b_c_d_l622_62263

theorem solve_a_b_c_d (n a b c d : ℕ) (h0 : 0 ≤ a) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : 2^n = a^2 + b^2 + c^2 + d^2) : 
  (a, b, c, d) ∈ {p | p = (↑0, ↑0, ↑0, 2^n.div (↑4)) ∨
                  p = (↑0, ↑0, 2^n.div (↑4), 2^n.div (↑4)) ∨
                  p = (2^n.div (↑4), 2^n.div (↑4), 2^n.div (↑4), 2^n.div (↑4)) ∨
                  p = (2^n.div (↑4), 0, 0, 0) ∨
                  p = (0, 2^n.div (↑4), 0, 0) ∨
                  p = (0, 0, 2^n.div (↑4), 0) ∨
                  p = (0, 0, 0, 2^n.div (↑4))} :=
sorry

end NUMINAMATH_GPT_solve_a_b_c_d_l622_62263


namespace NUMINAMATH_GPT_Claire_photos_l622_62297

variable (C : ℕ)

def Lisa_photos := 3 * C
def Robert_photos := C + 28

theorem Claire_photos :
  Lisa_photos C = Robert_photos C → C = 14 :=
by
  sorry

end NUMINAMATH_GPT_Claire_photos_l622_62297


namespace NUMINAMATH_GPT_sum_of_roots_l622_62265

theorem sum_of_roots (a b c : ℝ) (h_eq : a = 1) (h_b : b = -5) (h_c : c = 6) :
  (-b / a) = 5 := by
sorry

end NUMINAMATH_GPT_sum_of_roots_l622_62265


namespace NUMINAMATH_GPT_six_power_six_div_two_l622_62249

theorem six_power_six_div_two : 6 ^ (6 / 2) = 216 := by
  sorry

end NUMINAMATH_GPT_six_power_six_div_two_l622_62249


namespace NUMINAMATH_GPT_most_reasonable_plan_l622_62277

-- Defining the conditions as a type
inductive SurveyPlans
| A -- Surveying students in the second grade of School B
| C -- Randomly surveying 150 teachers
| B -- Surveying 600 students randomly selected from School C
| D -- Randomly surveying 150 students from each of the four schools

-- Define the main theorem asserting that the most reasonable plan is Option D
theorem most_reasonable_plan : SurveyPlans.D = SurveyPlans.D :=
by
  sorry

end NUMINAMATH_GPT_most_reasonable_plan_l622_62277


namespace NUMINAMATH_GPT_expression_evaluation_l622_62291

theorem expression_evaluation : (50 - (2050 - 250)) + (2050 - (250 - 50)) = 100 := by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l622_62291


namespace NUMINAMATH_GPT_seq_max_value_l622_62269

theorem seq_max_value {a_n : ℕ → ℝ} (h : ∀ n, a_n n = (↑n + 2) * (3 / 4) ^ n) : 
  ∃ n, a_n n = max (a_n 1) (a_n 2) → (n = 1 ∨ n = 2) :=
by 
  sorry

end NUMINAMATH_GPT_seq_max_value_l622_62269


namespace NUMINAMATH_GPT_induction_step_l622_62298

theorem induction_step 
  (k : ℕ) 
  (hk : ∃ m: ℕ, 5^k - 2^k = 3 * m) : 
  ∃ n: ℕ, 5^(k+1) - 2^(k+1) = 5 * (5^k - 2^k) + 3 * 2^k :=
by
  sorry

end NUMINAMATH_GPT_induction_step_l622_62298


namespace NUMINAMATH_GPT_divide_number_l622_62221

theorem divide_number (x : ℝ) (h : 0.3 * x = 0.2 * (80 - x) + 10) : min x (80 - x) = 28 := 
by 
  sorry

end NUMINAMATH_GPT_divide_number_l622_62221


namespace NUMINAMATH_GPT_simplify_expression_l622_62260

theorem simplify_expression :
  (Real.sin (Real.pi / 6) + (1 / 2) - 2007^0 + abs (-2) = 2) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l622_62260


namespace NUMINAMATH_GPT_square_garden_area_l622_62206

theorem square_garden_area (P A : ℕ)
  (h1 : P = 40)
  (h2 : A = 2 * P + 20) :
  A = 100 :=
by
  rw [h1] at h2 -- Substitute h1 (P = 40) into h2 (A = 2P + 20)
  norm_num at h2 -- Normalize numeric expressions in h2
  exact h2 -- Conclude by showing h2 (A = 100) holds

-- The output should be able to build successfully without solving the proof.

end NUMINAMATH_GPT_square_garden_area_l622_62206


namespace NUMINAMATH_GPT_avg_student_headcount_l622_62283

def student_headcount (yr1 yr2 yr3 yr4 : ℕ) : ℕ :=
  (yr1 + yr2 + yr3 + yr4) / 4

theorem avg_student_headcount :
  student_headcount 10600 10800 10500 10400 = 10825 :=
by
  sorry

end NUMINAMATH_GPT_avg_student_headcount_l622_62283


namespace NUMINAMATH_GPT_cos_squared_pi_over_4_minus_alpha_l622_62210

theorem cos_squared_pi_over_4_minus_alpha (α : ℝ) (h : Real.tan (α + Real.pi / 4) = 3 / 4) :
  Real.cos (Real.pi / 4 - α) ^ 2 = 9 / 25 :=
by
  sorry

end NUMINAMATH_GPT_cos_squared_pi_over_4_minus_alpha_l622_62210


namespace NUMINAMATH_GPT_required_run_rate_l622_62278

theorem required_run_rate
  (run_rate_first_10_overs : ℝ)
  (target_runs : ℝ)
  (overs_first : ℕ)
  (overs_remaining : ℕ)
  (H_run_rate_10_overs : run_rate_first_10_overs = 3.2)
  (H_target_runs : target_runs = 222)
  (H_overs_first : overs_first = 10)
  (H_overs_remaining : overs_remaining = 40) :
  ((target_runs - run_rate_first_10_overs * overs_first) / overs_remaining) = 4.75 := 
by
  sorry

end NUMINAMATH_GPT_required_run_rate_l622_62278


namespace NUMINAMATH_GPT_expected_coincidences_l622_62225

-- Definitions for the given conditions
def num_questions : ℕ := 20
def vasya_correct : ℕ := 6
def misha_correct : ℕ := 8

def prob_correct (correct : ℕ) : ℚ := correct / num_questions
def prob_incorrect (correct : ℕ) : ℚ := 1 - prob_correct correct 

def prob_vasya_correct := prob_correct vasya_correct
def prob_vasya_incorrect := prob_incorrect vasya_correct
def prob_misha_correct := prob_correct misha_correct
def prob_misha_incorrect := prob_incorrect misha_correct

-- Probability that both guessed correctly or incorrectly
def prob_both_correct_or_incorrect : ℚ := 
  (prob_vasya_correct * prob_misha_correct) + (prob_vasya_incorrect * prob_misha_incorrect)

-- Expected value for one question being a coincidence
def expected_I_k : ℚ := prob_both_correct_or_incorrect

-- Definition of the total number of coincidences
def total_coincidences : ℚ := num_questions * expected_I_k

-- Proof statement
theorem expected_coincidences : 
  total_coincidences = 10.8 := by
  -- calculation for the expected number
  sorry

end NUMINAMATH_GPT_expected_coincidences_l622_62225


namespace NUMINAMATH_GPT_parabola_line_intersection_distance_l622_62244

theorem parabola_line_intersection_distance :
  ∀ (x y : ℝ), x^2 = -4 * y ∧ y = x - 1 ∧ x^2 + 4 * x + 4 = 0 →
  abs (y - -1 + (-1 - y)) = 8 :=
by
  sorry

end NUMINAMATH_GPT_parabola_line_intersection_distance_l622_62244


namespace NUMINAMATH_GPT_recommended_water_intake_l622_62216

theorem recommended_water_intake (current_intake : ℕ) (increase_percentage : ℚ) (recommended_intake : ℕ) : 
  current_intake = 15 → increase_percentage = 0.40 → recommended_intake = 21 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_recommended_water_intake_l622_62216


namespace NUMINAMATH_GPT_inscribed_square_area_l622_62227

def isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ c = (a ^ 2 + b ^ 2) ^ (1 / 2)

def square_area (s : ℝ) : ℝ := s * s

theorem inscribed_square_area
  (a b c : ℝ) (s₁ s₂ : ℝ)
  (ha : a = 16 * 2) -- Leg lengths equal to 2 * 16 cm
  (hb : b = 16 * 2)
  (hc : c = 32 * Real.sqrt 2) -- Hypotenuse of the triangle
  (hiso : isosceles_right_triangle a b c)
  (harea₁ : square_area 16 = 256) -- Given square area
  (hS : s₂ = 16 * Real.sqrt 2 - 8) -- Side length of the new square
  : square_area s₂ = 576 - 256 * Real.sqrt 2 := sorry

end NUMINAMATH_GPT_inscribed_square_area_l622_62227


namespace NUMINAMATH_GPT_voltage_relationship_l622_62264

variables (x y z : ℝ) -- Coordinates representing positions on the lines
variables (I R U : ℝ) -- Representing current, resistance, and voltage respectively

-- Conditions translated into Lean
def I_def := I = 10^x
def R_def := R = 10^(-2 * y)
def U_def := U = 10^(-z)
def coord_relation := x + z = 2 * y

-- The final theorem to prove V = I * R under given conditions
theorem voltage_relationship : I = 10^x → R = 10^(-2 * y) → U = 10^(-z) → (x + z = 2 * y) → U = I * R :=
by 
  intros hI hR hU hXYZ
  sorry

end NUMINAMATH_GPT_voltage_relationship_l622_62264


namespace NUMINAMATH_GPT_four_digit_positive_integers_count_l622_62257

def first_two_digit_choices : Finset ℕ := {2, 3, 6}
def last_two_digit_choices : Finset ℕ := {3, 7, 9}

theorem four_digit_positive_integers_count :
  (first_two_digit_choices.card * first_two_digit_choices.card) *
  (last_two_digit_choices.card * (last_two_digit_choices.card - 1)) = 54 := by
sorry

end NUMINAMATH_GPT_four_digit_positive_integers_count_l622_62257


namespace NUMINAMATH_GPT_find_opposite_endpoint_l622_62207

/-- A utility function to model coordinate pairs as tuples -/
def coord_pair := (ℝ × ℝ)

-- Define the center and one endpoint
def center : coord_pair := (4, 6)
def endpoint1 : coord_pair := (2, 1)

-- Define the expected endpoint
def expected_endpoint2 : coord_pair := (6, 11)

/-- Definition of the opposite endpoint given the center and one endpoint -/
def opposite_endpoint (c : coord_pair) (p : coord_pair) : coord_pair :=
  let dx := c.1 - p.1
  let dy := c.2 - p.2
  (c.1 + dx, c.2 + dy)

/-- The proof statement for the problem -/
theorem find_opposite_endpoint :
  opposite_endpoint center endpoint1 = expected_endpoint2 :=
sorry

end NUMINAMATH_GPT_find_opposite_endpoint_l622_62207


namespace NUMINAMATH_GPT_division_into_rectangles_l622_62248

theorem division_into_rectangles (figure : Type) (valid_division : figure → Prop) : (∃ ways, ways = 8) :=
by {
  -- assume given conditions related to valid_division using "figure"
  sorry
}

end NUMINAMATH_GPT_division_into_rectangles_l622_62248


namespace NUMINAMATH_GPT_B_contains_only_one_element_l622_62294

def setA := { x | (x - 1/2) * (x - 3) = 0 }

def setB (a : ℝ) := { x | Real.log (x^2 + a * x + a + 9 / 4) = 0 }

theorem B_contains_only_one_element (a : ℝ) :
  (∃ x, setB a x ∧ ∀ y, setB a y → y = x) →
  (a = 5 ∨ a = -1) :=
by
  intro h
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_B_contains_only_one_element_l622_62294


namespace NUMINAMATH_GPT_beth_crayon_packs_l622_62228

theorem beth_crayon_packs (P : ℕ) (h1 : 10 * P + 6 = 46) : P = 4 :=
by
  sorry

end NUMINAMATH_GPT_beth_crayon_packs_l622_62228


namespace NUMINAMATH_GPT_fraction_bounds_l622_62230

theorem fraction_bounds (x y : ℝ) (h : x^2 * y^2 + x * y + 1 = 3 * y^2) : 
0 ≤ (y - x) / (x + 4 * y) ∧ (y - x) / (x + 4 * y) ≤ 4 := 
sorry

end NUMINAMATH_GPT_fraction_bounds_l622_62230


namespace NUMINAMATH_GPT_sum_first_11_terms_l622_62238

variable {a : ℕ → ℕ} -- a is the arithmetic sequence

-- Condition: a_4 + a_8 = 26
axiom condition : a 4 + a 8 = 26

-- Definition of arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

-- Definition of the sum of the first 11 terms
def S_11 (a : ℕ → ℕ) : ℕ := (11 * (a 1 + a 11)) / 2

-- The proof problem statement
theorem sum_first_11_terms (h : is_arithmetic_sequence a) : S_11 a = 143 := 
by 
  sorry

end NUMINAMATH_GPT_sum_first_11_terms_l622_62238
