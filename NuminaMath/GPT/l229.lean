import Mathlib

namespace cost_price_of_article_l229_229614

theorem cost_price_of_article :
  ∃ (CP : ℝ), (616 = 1.10 * (1.17 * CP)) → CP = 478.77 :=
by
  sorry

end cost_price_of_article_l229_229614


namespace period_and_symmetry_of_function_l229_229041

-- Given conditions
variables (f : ℝ → ℝ)
variable (hf_odd : ∀ x, f (-x) = -f x)
variable (hf_cond : ∀ x, f (-2 * x + 4) = -f (2 * x))
variable (hf_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 ^ x - 1)

-- Prove that 4 is a period and x=1 is a line of symmetry for the graph of f(x)
theorem period_and_symmetry_of_function :
  (∀ x, f (x + 4) = f x) ∧ (∀ x, f (x) + f (4 - x) = 0) :=
by sorry

end period_and_symmetry_of_function_l229_229041


namespace first_three_workers_time_l229_229813

open Real

-- Define a function to represent the total work done by any set of workers
noncomputable def total_work (a b c d : ℝ) (t : ℝ) := (a + b + c + d) * t

-- Conditions
variables (a b c d : ℝ)
variables (h1 : a + b + c + d = 1 / 6)
variables (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
variables (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4)

-- Question: Time for the first three workers to dig the trench
def trench_time : ℝ :=
  (a + b + c + d) * (6 / (a + b + c + d))

theorem first_three_workers_time : trench_time a b c = 6 :=
  sorry

end first_three_workers_time_l229_229813


namespace column_of_2023_l229_229168

theorem column_of_2023 : 
  let columns := ["G", "H", "I", "J", "K", "L", "M"]
  let pattern := ["H", "I", "J", "K", "L", "M", "L", "K", "J", "I", "H", "G"]
  let n := 2023
  (pattern.get! ((n - 2) % 12)) = "I" :=
by
  -- Sorry is a placeholder for the proof
  sorry

end column_of_2023_l229_229168


namespace trench_dig_time_l229_229823

theorem trench_dig_time (a b c d : ℝ) (h1 : a + b + c + d = 1/6)
  (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
  (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4) :
  a + b + c = 1 / 6 := sorry

end trench_dig_time_l229_229823


namespace harkamal_grapes_purchase_l229_229839

-- Define the conditions as parameters and constants
def cost_per_kg_grapes := 70
def kg_mangoes := 9
def cost_per_kg_mangoes := 45
def total_payment := 965

-- The theorem stating Harkamal purchased 8 kg of grapes
theorem harkamal_grapes_purchase : 
  ∃ G : ℕ, (cost_per_kg_grapes * G + cost_per_kg_mangoes * kg_mangoes = total_payment) ∧ G = 8 :=
by
  use 8
  unfold cost_per_kg_grapes cost_per_kg_mangoes kg_mangoes total_payment
  show 70 * 8 + 45 * 9 = 965 ∧ 8 = 8
  sorry

end harkamal_grapes_purchase_l229_229839


namespace pizza_topping_combinations_l229_229581

theorem pizza_topping_combinations :
  (Nat.choose 7 3) = 35 :=
sorry

end pizza_topping_combinations_l229_229581


namespace smallest_four_digit_number_in_pascals_triangle_l229_229424

theorem smallest_four_digit_number_in_pascals_triangle :
  (pascal : ℕ → ℕ → ℕ) 
  (h_start : ∀ n, pascal n 0 = 1) 
  (h_increase : ∀ n k, pascal n k ≤ pascal n (k + 1)) : 
  ∃ n k, pascal n k = 1000 := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l229_229424


namespace first_three_workers_time_l229_229814

open Real

-- Define a function to represent the total work done by any set of workers
noncomputable def total_work (a b c d : ℝ) (t : ℝ) := (a + b + c + d) * t

-- Conditions
variables (a b c d : ℝ)
variables (h1 : a + b + c + d = 1 / 6)
variables (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
variables (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4)

-- Question: Time for the first three workers to dig the trench
def trench_time : ℝ :=
  (a + b + c + d) * (6 / (a + b + c + d))

theorem first_three_workers_time : trench_time a b c = 6 :=
  sorry

end first_three_workers_time_l229_229814


namespace christina_total_weekly_distance_l229_229514

-- Definitions based on conditions
def daily_distance_to_school : ℕ := 7
def daily_round_trip : ℕ := 2 * daily_distance_to_school
def days_per_week : ℕ := 5
def extra_trip_distance_one_way : ℕ := 2
def extra_trip_round_trip : ℕ := 2 * extra_trip_distance_one_way

-- Theorem statement
theorem christina_total_weekly_distance :
  let weekly_distance := daily_round_trip * days_per_week in
  let final_week_distance := weekly_distance + extra_trip_round_trip in
  final_week_distance = 74 := by
sorry

end christina_total_weekly_distance_l229_229514


namespace bacteria_exceeds_day_l229_229100

theorem bacteria_exceeds_day :
  ∃ n : ℕ, 5 * 3^n > 200 ∧ ∀ m : ℕ, (m < n → 5 * 3^m ≤ 200) :=
sorry

end bacteria_exceeds_day_l229_229100


namespace smallest_four_digit_number_in_pascals_triangle_l229_229451

theorem smallest_four_digit_number_in_pascals_triangle :
  ∃ (n : ℕ), 1000 = (n choose min n) ∧ (n > 0 ∧ (∀ m < n, (m choose min m) < 1000)) :=
begin
  sorry
end

end smallest_four_digit_number_in_pascals_triangle_l229_229451


namespace member_age_greater_than_zero_l229_229559

def num_members : ℕ := 23
def avg_age : ℤ := 0
def age_range : Set ℤ := {x | x ≥ -20 ∧ x ≤ 20}
def num_negative_members : ℕ := 5

theorem member_age_greater_than_zero :
  ∃ n : ℕ, n ≤ 18 ∧ (avg_age = 0 ∧ num_members = 23 ∧ num_negative_members = 5 ∧ ∀ age ∈ age_range, age ≥ -20 ∧ age ≤ 20) :=
sorry

end member_age_greater_than_zero_l229_229559


namespace James_balloons_correct_l229_229874

def Amy_balloons : ℕ := 101
def diff_balloons : ℕ := 131
def James_balloons (a : ℕ) (d : ℕ) : ℕ := a + d

theorem James_balloons_correct : James_balloons Amy_balloons diff_balloons = 232 :=
by
  sorry

end James_balloons_correct_l229_229874


namespace probability_of_two_red_balls_l229_229045

theorem probability_of_two_red_balls :
  let red_balls := 4
  let blue_balls := 4
  let green_balls := 2
  let total_balls := red_balls + blue_balls + green_balls
  let prob_red1 := (red_balls : ℚ) / total_balls
  let prob_red2 := ((red_balls - 1 : ℚ) / (total_balls - 1))
  (prob_red1 * prob_red2 = (2 : ℚ) / 15) :=
by
  sorry

end probability_of_two_red_balls_l229_229045


namespace second_part_of_ratio_l229_229624

theorem second_part_of_ratio (first_part : ℝ) (whole second_part : ℝ) (h1 : first_part = 5) (h2 : first_part / whole = 25 / 100) : second_part = 15 :=
by
  sorry

end second_part_of_ratio_l229_229624


namespace correct_option_D_l229_229037

theorem correct_option_D (a : ℝ) (h : a ≠ 0) : a^0 = 1 :=
by sorry

end correct_option_D_l229_229037


namespace probability_diff_by_3_l229_229946

def roll_probability_diff_three (x y : ℕ) : ℚ :=
  if abs (x - y) = 3 then 1 else 0

theorem probability_diff_by_3 :
  let total_outcomes := 36 in
  let successful_outcomes := (finset.univ.product finset.univ).filter (λ (p : ℕ × ℕ), roll_probability_diff_three p.1 p.2 = 1) in
  (successful_outcomes.card : ℚ) / total_outcomes = 5 / 36 :=
by
  sorry

end probability_diff_by_3_l229_229946


namespace lock_rings_l229_229162

theorem lock_rings (n : ℕ) (h : 6 ^ n - 1 ≤ 215) : n = 3 :=
sorry

end lock_rings_l229_229162


namespace fraction_subtraction_l229_229651

theorem fraction_subtraction : (18 : ℚ) / 45 - (3 : ℚ) / 8 = (1 : ℚ) / 40 := by
  sorry

end fraction_subtraction_l229_229651


namespace problem_solution_l229_229654

noncomputable def S : ℝ :=
  1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 1 / (Real.sqrt 14 - Real.sqrt 13) - 1 / (Real.sqrt 13 - 3)

theorem problem_solution : S = (13 / 4) + (3 / 4) * Real.sqrt 13 :=
by sorry

end problem_solution_l229_229654


namespace turnip_difference_l229_229119

theorem turnip_difference (melanie_turnips benny_turnips : ℕ) (h1 : melanie_turnips = 139) (h2 : benny_turnips = 113) : melanie_turnips - benny_turnips = 26 := by
  sorry

end turnip_difference_l229_229119


namespace no_prime_divisible_by_77_l229_229691

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_77 : ∀ p : ℕ, is_prime p → ¬ (77 ∣ p) :=
by
  sorry

end no_prime_divisible_by_77_l229_229691


namespace isosceles_perimeter_l229_229829

noncomputable def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem isosceles_perimeter
  (k : ℝ)
  (a b : ℝ)
  (h1 : 4 = a)
  (h2 : k * b^2 - (k + 8) * b + 8 = 0)
  (h3 : k ≠ 0)
  (h4 : is_triangle 4 a a) : a + 4 + a = 9 :=
sorry

end isosceles_perimeter_l229_229829


namespace smallest_four_digit_in_pascals_triangle_l229_229448

theorem smallest_four_digit_in_pascals_triangle : ∃ n, ∃ k, k ≤ n ∧ 1000 ≤ Nat.choose n k :=
sorry

end smallest_four_digit_in_pascals_triangle_l229_229448


namespace smallest_positive_integer_ending_in_9_divisible_by_13_l229_229025

theorem smallest_positive_integer_ending_in_9_divisible_by_13 :
  ∃ n : ℕ, (n % 10 = 9) ∧ (n % 13 = 0) ∧ (∀ m : ℕ, (m % 10 = 9) ∧ (m % 13 = 0) → m ≥ n) :=
sorry

end smallest_positive_integer_ending_in_9_divisible_by_13_l229_229025


namespace smallest_four_digit_in_pascal_l229_229416

-- define what it means for a number to be in Pascal's triangle
def in_pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- define the concept of a smallest four-digit number
def smallest_four_digit : ℕ := 1000

-- prove that the smallest four-digit number in Pascal's Triangle is 1000
theorem smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), in_pascal n k = smallest_four_digit := 
sorry

end smallest_four_digit_in_pascal_l229_229416


namespace correct_calculation_B_l229_229605

theorem correct_calculation_B :
  (∀ (a : ℕ), 3 * a^3 * 2 * a^2 ≠ 6 * a^6) ∧
  (∀ (x : ℕ), 2 * x^2 * 3 * x^2 = 6 * x^4) ∧
  (∀ (x : ℕ), 3 * x^2 * 4 * x^2 ≠ 12 * x^2) ∧
  (∀ (y : ℕ), 5 * y^3 * 3 * y^5 ≠ 8 * y^8) →
  (∀ (x : ℕ), 2 * x^2 * 3 * x^2 = 6 * x^4) := 
by
  sorry

end correct_calculation_B_l229_229605


namespace omega_range_l229_229990

namespace Problem

def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem omega_range (ω : ℝ) (hω : ω > 0) : 
  (∃ (a b : ℝ), a ∈ set.Icc 0 (Real.pi / 2) ∧ b ∈ set.Icc 0 (Real.pi / 2) ∧ a ≠ b ∧ f ω a + f ω b = 4) ↔ 
  5 ≤ ω ∧ ω < 9 :=
begin
  sorry
end

end Problem

end omega_range_l229_229990


namespace solve_x_l229_229825

theorem solve_x (x : ℝ) : (x - 1)^2 = 4 → (x = 3 ∨ x = -1) := 
by 
  sorry

end solve_x_l229_229825


namespace rhombus_diagonal_length_l229_229898

-- Define a rhombus with one diagonal of 10 cm and a perimeter of 52 cm.
theorem rhombus_diagonal_length (d : ℝ) 
  (h1 : ∃ a b c : ℝ, a = 10 ∧ b = d ∧ c = 13) -- The diagonals and side of rhombus.
  (h2 : 52 = 4 * c) -- The perimeter condition.
  (h3 : c^2 = (d/2)^2 + (10/2)^2) -- The relationship from Pythagorean theorem.
  : d = 24 :=
by
  sorry

end rhombus_diagonal_length_l229_229898


namespace find_some_number_l229_229680

theorem find_some_number :
  ∃ some_number : ℝ, (3.242 * 10 / some_number) = 0.032420000000000004 ∧ some_number = 1000 :=
by
  sorry

end find_some_number_l229_229680


namespace min_a_for_inequality_l229_229094

theorem min_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x ≤ 1/2) → (x^2 + a*x + 1 ≥ 0)) ↔ a ≥ -5/2 :=
by
  sorry

end min_a_for_inequality_l229_229094


namespace mary_initially_selected_10_l229_229505

-- Definitions based on the conditions
def price_apple := 40
def price_orange := 60
def avg_price_initial := 54
def avg_price_after_putting_back := 48
def num_oranges_put_back := 5

-- Definition of Mary_initially_selected as the total number of pieces of fruit initially selected by Mary
def Mary_initially_selected (A O : ℕ) := A + O

-- Theorem statement
theorem mary_initially_selected_10 (A O : ℕ) 
  (h1 : (price_apple * A + price_orange * O) / (A + O) = avg_price_initial)
  (h2 : (price_apple * A + price_orange * (O - num_oranges_put_back)) / (A + O - num_oranges_put_back) = avg_price_after_putting_back) : 
  Mary_initially_selected A O = 10 := 
sorry

end mary_initially_selected_10_l229_229505


namespace average_speed_l229_229913

theorem average_speed
    (distance1 distance2 : ℕ)
    (time1 time2 : ℕ)
    (h1 : distance1 = 100)
    (h2 : distance2 = 80)
    (h3 : time1 = 1)
    (h4 : time2 = 1) :
    (distance1 + distance2) / (time1 + time2) = 90 :=
by
  sorry

end average_speed_l229_229913


namespace triangle_perimeter_correct_l229_229234

noncomputable def triangle_perimeter (a b c : ℕ) : ℕ :=
    a + b + c

theorem triangle_perimeter_correct (a b c : ℕ) (h1 : a = b - 1) (h2 : b = c - 1) (h3 : c = 2 * a) : triangle_perimeter a b c = 15 :=
    sorry

end triangle_perimeter_correct_l229_229234


namespace minimum_value_expression_l229_229246

noncomputable def problem_statement (x y z : ℝ) : ℝ :=
  (x + y) / z + (x + z) / y + (y + z) / x + 3

theorem minimum_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  problem_statement x y z ≥ 9 :=
sorry

end minimum_value_expression_l229_229246


namespace total_animals_on_farm_l229_229251

theorem total_animals_on_farm :
  let coop1 := 60
  let coop2 := 45
  let coop3 := 55
  let coop4 := 40
  let coop5 := 35
  let coop6 := 20
  let coop7 := 50
  let coop8 := 10
  let coop9 := 10
  let first_shed := 2 * 10
  let second_shed := 10
  let third_shed := 6
  let section1 := 15
  let section2 := 25
  let section3 := 2 * 15
  coop1 + coop2 + coop3 + coop4 + coop5 + coop6 + coop7 + coop8 + coop9 + first_shed + second_shed + third_shed + section1 + section2 + section3 = 431 :=
by
  sorry

end total_animals_on_farm_l229_229251


namespace unit_prices_purchase_plans_exchange_methods_l229_229966

theorem unit_prices (x r : ℝ) (hx : r = 2 * x) 
  (h_eq : (40/(2*r)) + 4 = 30/x) : 
  x = 2.5 ∧ r = 5 := sorry

theorem purchase_plans (x r : ℝ) (a b : ℕ)
  (hx : x = 2.5) (hr : r = 5) (h_eq : x * a + r * b = 200)
  (h_ge_20 : 20 ≤ a ∧ 20 ≤ b) (h_mult_10 : a % 10 = 0) :
  (a, b) = (20, 30) ∨ (a, b) = (30, 25) ∨ (a, b) = (40, 20) := sorry

theorem exchange_methods (a b t m : ℕ) 
  (hx : x = 2.5) (hr : r = 5) 
  (h_leq : 1 < m ∧ m < 10) 
  (h_eq : a + 2 * t = b + (m - t))
  (h_planA : (a = 20 ∧ b = 30) ∨ (a = 30 ∧ b = 25) ∨ (a = 40 ∧ b = 20)) :
  (m = 5 ∧ t = 5 ∧ b = 30) ∨
  (m = 8 ∧ t = 6 ∧ b = 25) ∨
  (m = 5 ∧ t = 0 ∧ b = 25) ∨
  (m = 8 ∧ t = 1 ∧ b = 20) := sorry

end unit_prices_purchase_plans_exchange_methods_l229_229966


namespace next_divisor_of_4_digit_even_number_l229_229118

theorem next_divisor_of_4_digit_even_number (n : ℕ) (h1 : 1000 ≤ n ∧ n < 10000)
  (h2 : n % 2 = 0) (hDiv : n % 221 = 0) :
  ∃ d, d > 221 ∧ d < n ∧ d % 13 = 0 ∧ d % 17 = 0 ∧ d = 442 :=
by
  use 442
  sorry

end next_divisor_of_4_digit_even_number_l229_229118


namespace complete_square_quadratic_t_l229_229388

theorem complete_square_quadratic_t : 
  ∀ x : ℝ, (16 * x^2 - 32 * x - 512 = 0) → (∃ q t : ℝ, (x + q)^2 = t ∧ t = 33) :=
by sorry

end complete_square_quadratic_t_l229_229388


namespace smallest_two_digit_prime_with_conditions_l229_229522

open Nat

def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def is_composite (n : ℕ) : Prop := n ≥ 2 ∧ ∃ m, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

def reverse_digits (n : ℕ) : ℕ :=
  let digits := to_digits 10 n
  of_digits 10 digits.reverse

noncomputable def smallest_prime_with_conditions : ℕ :=
  23 -- The smallest prime satisfying the conditions, verified manually

theorem smallest_two_digit_prime_with_conditions :
  ∃ p, is_prime p ∧ (10 ≤ p ∧ p < 100) ∧ (p % 100 / 10 = 2) ∧ is_composite (reverse_digits p) ∧ 
  (∀ q, is_prime q ∧ (10 ≤ q ∧ q < 100) ∧ (q % 100 / 10 = 2) ∧ is_composite (reverse_digits q) → p ≤ q) := 
begin
  use 23,
  split, {
    dsimp [is_prime],
    split,
    { exact dec_trivial }, -- 23 ≥ 2
    { intros m hm,
      interval_cases m; simp [hm],
    },
  },
  split, {
    split,
    { norm_num }, -- 10 ≤ 23
    { norm_num }, -- 23 < 100
  },
  split, {
    norm_num, -- tens digit is 2
  },
  split, {
    dsimp [is_composite, reverse_digits],
    use 2,
    split,
    { exact dec_trivial, }, -- 10 ≤ 32
    split,
    { exact dec_trivial, }, -- 2 ≠ 1
    { exact dec_trivial },  -- 2 ≠ 32
  },
  intros q hq,
  rcases hq with ⟨prm_q, ⟨hq_low, hq_high⟩, t_d, q_comp⟩,
  rw t_d at *,
  linarith,
end

end smallest_two_digit_prime_with_conditions_l229_229522


namespace number_of_girls_with_no_pets_l229_229864

-- Define total number of students
def total_students : ℕ := 30

-- Define the fraction of boys in the class
def fraction_boys : ℚ := 1 / 3

-- Define the percentages of girls with pets
def percentage_girls_with_dogs : ℚ := 0.40
def percentage_girls_with_cats : ℚ := 0.20

-- Calculate the number of boys
def number_of_boys : ℕ := (fraction_boys * total_students).toNat

-- Calculate the number of girls
def number_of_girls : ℕ := total_students - number_of_boys

-- Calculate the number of girls who own dogs
def number_of_girls_with_dogs : ℕ := (percentage_girls_with_dogs * number_of_girls).toNat

-- Calculate the number of girls who own cats
def number_of_girls_with_cats : ℕ := (percentage_girls_with_cats * number_of_girls).toNat

-- Define the statement to be proved
theorem number_of_girls_with_no_pets : number_of_girls - (number_of_girls_with_dogs + number_of_girls_with_cats) = 8 := by
  sorry

end number_of_girls_with_no_pets_l229_229864


namespace exists_points_with_distance_one_l229_229566

open Set Real

noncomputable def regionK (K : Set (ℝ × ℝ)) : Prop :=
  Convex ℝ K ∧ (∃ (L : List (Set (ℝ × ℝ))), Union L = boundary K ∧ ∀ l ∈ L, ∃ a b : ℝ × ℝ, l = segment ℝ a b) ∧ measure_univ K ≥ π / 4

theorem exists_points_with_distance_one {K : Set (ℝ × ℝ)} (hK : regionK K) :
  ∃ (P Q : ℝ × ℝ), P ∈ K ∧ Q ∈ K ∧ dist P Q = 1 :=
sorry

end exists_points_with_distance_one_l229_229566


namespace parabola_focus_l229_229199

-- Define the given conditions
def parabola_equation (x : ℝ) : ℝ := 4 * x^2 - 8 * x + 5

-- The proof statement that we need to show the focus of the given parabola
theorem parabola_focus :
  (∃ (h k : ℝ), (k = 1) ∧ (h = 1) ∧ (parabola_equation h = k) ∧ ((h, k + 1 / (4 * 4)) = (1, 17 / 16))) := 
sorry

end parabola_focus_l229_229199


namespace amy_bike_total_l229_229172

-- Define the miles Amy biked yesterday
def y : ℕ := 12

-- Define the miles Amy biked today
def t : ℕ := 2 * y - 3

-- Define the total miles Amy biked in two days
def total : ℕ := y + t

-- The theorem stating the total distance biked equals 33 miles
theorem amy_bike_total : total = 33 := by
  sorry

end amy_bike_total_l229_229172


namespace sum_of_squares_l229_229886

theorem sum_of_squares (a d : Int) : 
  ∃ y1 y2 : Int, a^2 + 2*(a+d)^2 + 3*(a+2*d)^2 + 4*(a+3*d)^2 = (3*a + y1*d)^2 + (a + y2*d)^2 :=
by
  sorry

end sum_of_squares_l229_229886


namespace find_number_l229_229477

theorem find_number (x : ℕ) (h : 23 + x = 34) : x = 11 :=
by
  sorry

end find_number_l229_229477


namespace first_cube_weight_l229_229770

-- Given definitions of cubes and their relationships
def weight_of_cube (s : ℝ) (weight : ℝ) : Prop :=
  ∃ v : ℝ, v = s^3 ∧ weight = v

def cube_relationship (s1 s2 weight2 : ℝ) : Prop :=
  s2 = 2 * s1 ∧ weight2 = 32

-- The proof problem
theorem first_cube_weight (s1 s2 weight1 weight2 : ℝ) (h1 : cube_relationship s1 s2 weight2) : weight1 = 4 :=
  sorry

end first_cube_weight_l229_229770


namespace range_k_l229_229082

noncomputable def point (α : Type*) := (α × α)

def M : point ℝ := (0, 2)
def N : point ℝ := (-2, 0)

def line (k : ℝ) (P : point ℝ) := k * P.1 - P.2 - 2 * k + 2 = 0
def angle_condition (M N P : point ℝ) := true -- placeholder for the condition that ∠MPN ≥ π/2

theorem range_k (k : ℝ) (P : point ℝ)
  (hP_on_line : line k P)
  (h_angle_cond : angle_condition M N P) :
  (1 / 7 : ℝ) ≤ k ∧ k ≤ 1 :=
sorry

end range_k_l229_229082


namespace minimum_average_cost_l229_229932

noncomputable def average_cost (x : ℝ) : ℝ :=
  let y := (x^2) / 10 - 30 * x + 4000
  y / x

theorem minimum_average_cost : 
  ∃ (x : ℝ), 150 ≤ x ∧ x ≤ 250 ∧ (∀ (x' : ℝ), 150 ≤ x' ∧ x' ≤ 250 → average_cost x ≤ average_cost x') ∧ average_cost x = 10 := 
by
  sorry

end minimum_average_cost_l229_229932


namespace dice_rolls_diff_by_3_probability_l229_229953

-- Define a function to encapsulate the problem's statement
def probability_dice_diff_by_3 : ℚ := 1 / 6

-- Prove that given the conditions, the probability of rolling integers 
-- that differ by 3 when rolling a standard 6-sided die twice is 1/6.
theorem dice_rolls_diff_by_3_probability : 
  (probability (λ (x y : ℕ), x != y ∧ x - y = 3 ∨ y - x = 3) (finset.range 1 7 ×ˢ finset.range 1 7)) = probability_dice_diff_by_3 :=
sorry

end dice_rolls_diff_by_3_probability_l229_229953


namespace smallest_four_digit_in_pascal_triangle_l229_229472

noncomputable section

def is_in_pascal_triangle (n : ℕ) : Prop :=
  ∃ r c : ℕ, c ≤ r ∧ binomial r c = n

theorem smallest_four_digit_in_pascal_triangle :
  ¬ ∃ n : ℕ, is_in_pascal_triangle n ∧ n < 1000 ∧ (n ≥ 1000 ∨ n < 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_triangle_l229_229472


namespace time_for_first_three_workers_l229_229812

def work_rate_equations (a b c d : ℝ) :=
  a + b + c + d = 1 / 6 ∧
  2 * a + (1 / 2) * b + c + d = 1 / 6 ∧
  (1 / 2) * a + 2 * b + c + d = 1 / 4

theorem time_for_first_three_workers (a b c d : ℝ) (h : work_rate_equations a b c d) :
  (a + b + c) * 6 = 1 :=
sorry

end time_for_first_three_workers_l229_229812


namespace problem_range_of_k_l229_229213

theorem problem_range_of_k (k : ℝ) : 
  (∀ x : ℝ, x^2 - 11 * x + (30 + k) = 0 → x > 5) → (0 < k ∧ k ≤ 1 / 4) :=
by
  sorry

end problem_range_of_k_l229_229213


namespace greatest_integer_is_8_l229_229603

theorem greatest_integer_is_8 {a b : ℤ} (h_sum : a + b + 8 = 21) : max a (max b 8) = 8 :=
by
  sorry

end greatest_integer_is_8_l229_229603


namespace fourth_vertex_of_tetrahedron_exists_l229_229625

theorem fourth_vertex_of_tetrahedron_exists (x y z : ℤ) :
  (∃ (x y z : ℤ), 
     ((x - 1) ^ 2 + y ^ 2 + (z - 3) ^ 2 = 26) ∧ 
     ((x - 5) ^ 2 + (y - 3) ^ 2 + (z - 2) ^ 2 = 26) ∧ 
     ((x - 4) ^ 2 + y ^ 2 + (z - 6) ^ 2 = 26)) :=
sorry

end fourth_vertex_of_tetrahedron_exists_l229_229625


namespace manager_monthly_salary_l229_229399

theorem manager_monthly_salary :
  let avg_salary := 1200
  let num_employees := 20
  let total_salary := num_employees * avg_salary
  let new_avg_salary := avg_salary + 100
  let num_people_with_manager := num_employees + 1
  let new_total_salary := num_people_with_manager * new_avg_salary
  let manager_salary := new_total_salary - total_salary
  manager_salary = 3300 := by
  sorry

end manager_monthly_salary_l229_229399


namespace smallest_four_digit_in_pascals_triangle_l229_229457

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, 1000 ≤ nat.choose n k ∧ nat.choose n k < 10000 ∧ 
  (∀ m l : ℕ, (nat.choose m l ≥ 1000 ∧ nat.choose m l < 10000) → nat.choose n k ≤ nat.choose m l) :=
begin
  -- This is a statement only, no proof is provided
  sorry
end

end smallest_four_digit_in_pascals_triangle_l229_229457


namespace units_digit_17_pow_2023_l229_229035

theorem units_digit_17_pow_2023 : (17 ^ 2023) % 10 = 3 :=
by
  have units_cycle_7 : ∀ (n : ℕ), (7 ^ n) % 10 = [7, 9, 3, 1].nth (n % 4) :=
    sorry
  have units_pattern_equiv : (17 ^ n) % 10 = (7 ^ n) % 10 :=
    sorry
  calc
    (17 ^ 2023) % 10
        = (7 ^ 2023) % 10  : by rw [units_pattern_equiv]
    ... = 3               : by rw [units_cycle_7, nat.mod_eq_of_lt, List.nth]

end units_digit_17_pow_2023_l229_229035


namespace h_at_neg_eight_l229_229879

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + x + 1

noncomputable def h (x : ℝ) (a b c : ℝ) : ℝ := (x - a^3) * (x - b^3) * (x - c^3)

theorem h_at_neg_eight (a b c : ℝ) (hf : f a = 0) (hf_b : f b = 0) (hf_c : f c = 0) :
  h (-8) a b c = -115 :=
  sorry

end h_at_neg_eight_l229_229879


namespace largest_common_term_l229_229266

theorem largest_common_term (b : ℕ) (h1 : b ≡ 1 [MOD 3]) (h2 : b ≡ 2 [MOD 10]) (h3 : b < 300) : b = 290 :=
sorry

end largest_common_term_l229_229266


namespace arithmetic_sum_property_l229_229560

variable {a : ℕ → ℤ} -- declare the sequence as a sequence of integers

-- Define the condition of the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 2 - a 1

-- Given condition: sum of specific terms in the sequence equals 400
def sum_condition (a : ℕ → ℤ) : Prop :=
  a 3 + a 4 + a 5 + a 6 + a 7 = 400

-- The goal: if the sum_condition holds, then a_2 + a_8 = 160
theorem arithmetic_sum_property
  (h_sum : sum_condition a)
  (h_arith : arithmetic_sequence a) :
  a 2 + a 8 = 160 := by
  sorry

end arithmetic_sum_property_l229_229560


namespace problem_statement_l229_229606

theorem problem_statement : 25 * 15 * 9 * 5.4 * 3.24 = 3 ^ 10 := 
by 
  sorry

end problem_statement_l229_229606


namespace amy_bike_total_l229_229173

-- Define the miles Amy biked yesterday
def y : ℕ := 12

-- Define the miles Amy biked today
def t : ℕ := 2 * y - 3

-- Define the total miles Amy biked in two days
def total : ℕ := y + t

-- The theorem stating the total distance biked equals 33 miles
theorem amy_bike_total : total = 33 := by
  sorry

end amy_bike_total_l229_229173


namespace divisors_form_l229_229922

theorem divisors_form (p n : ℕ) (h_prime : Nat.Prime p) (h_pos : 0 < n) :
  ∃ k : ℕ, (p^n - 1 = 2^k - 1 ∨ p^n - 1 ∣ 48) :=
sorry

end divisors_form_l229_229922


namespace correct_calculation_l229_229764

theorem correct_calculation (x : ℤ) (h : x + 54 = 78) : x + 45 = 69 :=
by
  sorry

end correct_calculation_l229_229764


namespace jennifer_cards_left_l229_229875

-- Define the initial number of cards and the number of cards eaten
def initial_cards : ℕ := 72
def eaten_cards : ℕ := 61

-- Define the final number of cards
def final_cards (initial_cards eaten_cards : ℕ) : ℕ :=
  initial_cards - eaten_cards

-- Proposition stating that Jennifer has 11 cards left
theorem jennifer_cards_left : final_cards initial_cards eaten_cards = 11 :=
by
  -- Proof here
  sorry

end jennifer_cards_left_l229_229875


namespace find_a_plus_b_l229_229676

noncomputable def A : ℝ := 3
noncomputable def B : ℝ := -1

noncomputable def l : ℝ := -1 -- Slope of line l (since angle is 3π/4)

noncomputable def l1_slope : ℝ := 1 -- Slope of line l1 which is perpendicular to l

noncomputable def a : ℝ := 0 -- Calculated from k_{AB} = 1

noncomputable def b : ℝ := -2 -- Calculated from line parallel condition

theorem find_a_plus_b : a + b = -2 :=
by
  sorry

end find_a_plus_b_l229_229676


namespace probability_sum_six_two_dice_l229_229299

theorem probability_sum_six_two_dice :
  let total_outcomes := 36
  let favorable_outcomes := 5
  favorable_outcomes / total_outcomes = 5 / 36 := by
  let total_outcomes := 6 * 6
  let favorable_outcomes := 5
  sorry

end probability_sum_six_two_dice_l229_229299


namespace find_a_l229_229069

def star (a b : ℝ) : ℝ := 2 * a - b^2

theorem find_a (a : ℝ) (h : star a 5 = 9) : a = 17 := by
  sorry

end find_a_l229_229069


namespace parameterization_theorem_l229_229401

theorem parameterization_theorem (a b c d : ℝ) (h1 : b = 1) (h2 : d = -3) (h3 : a + b = 4) (h4 : c + d = 5) :
  a^2 + b^2 + c^2 + d^2 = 83 :=
by
  sorry

end parameterization_theorem_l229_229401


namespace inclination_angle_l229_229351

theorem inclination_angle (α : ℝ) (t : ℝ) (h : 0 < α ∧ α < π / 2) :
  let x := 1 + t * Real.cos (α + 3 * π / 2)
  let y := 2 + t * Real.sin (α + 3 * π / 2)
  ∃ θ, θ = α + π / 2 := by
  sorry

end inclination_angle_l229_229351


namespace calculate_product_value_l229_229327

theorem calculate_product_value :
    (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 :=
by
  sorry

end calculate_product_value_l229_229327


namespace rhombus_longer_diagonal_l229_229634

theorem rhombus_longer_diagonal (a b d : ℝ) (h : a = 65) (h_d : d = 56) :
  ∃ l, l = 118 :=
by
  sorry

end rhombus_longer_diagonal_l229_229634


namespace base_any_number_l229_229555

open Nat

theorem base_any_number (n k : ℕ) (h1 : k ≥ 0) (h2 : (30 ^ k) ∣ 929260) (h3 : n ^ k - k ^ 3 = 1) : true :=
by
  sorry

end base_any_number_l229_229555


namespace min_value_expression_l229_229210

variable (a b m n : ℝ)

-- Conditions: a, b, m, n are positive, a + b = 1, mn = 2
def conditions (a b m n : ℝ) : Prop := 
  0 < a ∧ 0 < b ∧ 0 < m ∧ 0 < n ∧ a + b = 1 ∧ m * n = 2

-- Statement to prove: The minimum value of (am + bn) * (bm + an) is 2
theorem min_value_expression (a b m n : ℝ) (h : conditions a b m n) : 
  ∃ c : ℝ, c = 2 ∧ (∀ (x y z w : ℝ), conditions x y z w → (x * z + y * w) * (y * z + x * w) ≥ c) :=
by
  sorry

end min_value_expression_l229_229210


namespace sym_diff_A_B_l229_229524

def set_diff (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}
def sym_diff (M N : Set ℝ) : Set ℝ := set_diff M N ∪ set_diff N M

def A : Set ℝ := {x | -1 ≤ x ∧ x < 1}
def B : Set ℝ := {x | x < 0}

theorem sym_diff_A_B :
  sym_diff A B = {x | x < -1} ∪ {x | 0 ≤ x ∧ x < 1} := by
  sorry

end sym_diff_A_B_l229_229524


namespace quadratic_function_correct_l229_229010

-- Defining the quadratic function a
def quadratic_function (x : ℝ) : ℝ := 2 * x^2 - 14 * x + 20

-- Theorem stating that the quadratic function passes through the points (2, 0) and (5, 0)
theorem quadratic_function_correct : 
  quadratic_function 2 = 0 ∧ quadratic_function 5 = 0 := 
by
  -- these proofs are skipped with sorry for now
  sorry

end quadratic_function_correct_l229_229010


namespace avg_visitors_per_day_l229_229039

theorem avg_visitors_per_day 
  (avg_visitors_sundays : ℕ) 
  (avg_visitors_other_days : ℕ) 
  (total_days : ℕ) 
  (sundays : ℕ) 
  (other_days : ℕ)
  (hs : avg_visitors_sundays = 630)
  (ho : avg_visitors_other_days = 240)
  (td : total_days = 30)
  (sd : sundays = 4)
  (od : other_days = 26)
  : (4 * avg_visitors_sundays + 26 * avg_visitors_other_days) / 30 = 292 := 
by
  sorry

end avg_visitors_per_day_l229_229039


namespace expand_expression_l229_229191

theorem expand_expression : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  intro x
  sorry

end expand_expression_l229_229191


namespace alpha_range_midpoint_trajectory_l229_229376

noncomputable def circle_parametric_eqn (θ : ℝ) : ℝ × ℝ :=
  ⟨Real.cos θ, Real.sin θ⟩

theorem alpha_range (α : ℝ) (h1 : 0 < α ∧ α < 2 * Real.pi) :
  (Real.tan α) > 1 ∨ (Real.tan α) < -1 ↔ (Real.pi / 4 < α ∧ α < 3 * Real.pi / 4) ∨ 
                                          (5 * Real.pi / 4 < α ∧ α < 7 * Real.pi / 4) := 
  sorry

theorem midpoint_trajectory (m : ℝ) (h2 : -1 < m ∧ m < 1) :
  ∃ x y : ℝ, x = (Real.sqrt 2 * m) / (m^2 + 1) ∧ 
             y = -(Real.sqrt 2 * m^2) / (m^2 + 1) :=
  sorry

end alpha_range_midpoint_trajectory_l229_229376


namespace smallest_four_digit_number_in_pascals_triangle_l229_229438

theorem smallest_four_digit_number_in_pascals_triangle : ∃ n k, binom n k = 1000 ∧ (∀ n' k', binom n' k' > 1000 → nat.zero < n' k' ∧ n' k' < (binom n k)) := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l229_229438


namespace remaining_volume_correct_l229_229500

noncomputable def diameter_sphere : ℝ := 24
noncomputable def radius_sphere : ℝ := diameter_sphere / 2
noncomputable def height_hole1 : ℝ := 10
noncomputable def diameter_hole1 : ℝ := 3
noncomputable def radius_hole1 : ℝ := diameter_hole1 / 2
noncomputable def height_hole2 : ℝ := 10
noncomputable def diameter_hole2 : ℝ := 3
noncomputable def radius_hole2 : ℝ := diameter_hole2 / 2
noncomputable def height_hole3 : ℝ := 5
noncomputable def diameter_hole3 : ℝ := 4
noncomputable def radius_hole3 : ℝ := diameter_hole3 / 2

noncomputable def volume_sphere : ℝ := (4 / 3) * Real.pi * (radius_sphere ^ 3)
noncomputable def volume_hole1 : ℝ := Real.pi * (radius_hole1 ^ 2) * height_hole1
noncomputable def volume_hole2 : ℝ := Real.pi * (radius_hole2 ^ 2) * height_hole2
noncomputable def volume_hole3 : ℝ := Real.pi * (radius_hole3 ^ 2) * height_hole3

noncomputable def remaining_volume : ℝ := 
  volume_sphere - (2 * volume_hole1 + volume_hole3)

theorem remaining_volume_correct : remaining_volume = 2239 * Real.pi := by
  sorry

end remaining_volume_correct_l229_229500


namespace probability_of_rolling_five_l229_229777

theorem probability_of_rolling_five (total_outcomes : ℕ) (favorable_outcomes : ℕ) 
  (h1 : total_outcomes = 6) (h2 : favorable_outcomes = 1) : 
  favorable_outcomes / total_outcomes = (1 / 6 : ℚ) :=
by
  sorry

end probability_of_rolling_five_l229_229777


namespace cats_eat_fish_l229_229767

theorem cats_eat_fish (c d: ℕ) (h1 : 1 < c) (h2 : c < 10) (h3 : c * d = 91) : c + d = 20 := by
  sorry

end cats_eat_fish_l229_229767


namespace part1_part2_l229_229085

def setA (m : ℝ) : Set ℝ := {x | 0 < x - m ∧ x - m < 3}
def setB : Set ℝ := {x | x ≤ 0 ∨ x ≥ 3}

theorem part1 (m : ℝ) (h : m = 1) : 
  {x | x ∈ setA m} ∩ {x | x ∈ setB} = {x | 3 ≤ x ∧ x < 4} :=
by {
  sorry
}

theorem part2 (m : ℝ): 
  ({x | x ∈ setA m} ∪ {x | x ∈ setB} = {x | x ∈ setB}) ↔ (m ≥ 3 ∨ m ≤ -3) :=
by {
  sorry
}

end part1_part2_l229_229085


namespace distribution_y_value_l229_229899

theorem distribution_y_value :
  ∀ (x y : ℝ),
  (x + 0.1 + 0.3 + y = 1) →
  (7 * x + 8 * 0.1 + 9 * 0.3 + 10 * y = 8.9) →
  y = 0.4 :=
by
  intros x y h1 h2
  sorry

end distribution_y_value_l229_229899


namespace parallel_vectors_x_value_angle_between_vectors_pi_div_2_l229_229089

open Real

-- Problem 1
theorem parallel_vectors_x_value (x : ℝ) (h : (1 : ℝ) * x - 3 * 3 = 0) : x = 9 :=
sorry

-- Problem 2
theorem angle_between_vectors_pi_div_2 (x : ℝ) (h₁ : x = -1)
  (h₂ : (1 : ℝ) * 3 + 3 * x = 0) : angle (1, 3) (3, x) = π / 2 :=
sorry

end parallel_vectors_x_value_angle_between_vectors_pi_div_2_l229_229089


namespace smallest_four_digit_number_in_pascals_triangle_l229_229426

theorem smallest_four_digit_number_in_pascals_triangle :
  (pascal : ℕ → ℕ → ℕ) 
  (h_start : ∀ n, pascal n 0 = 1) 
  (h_increase : ∀ n k, pascal n k ≤ pascal n (k + 1)) : 
  ∃ n k, pascal n k = 1000 := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l229_229426


namespace smallest_four_digit_in_pascal_l229_229432

-- Pascal's triangle definition
def pascal (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

-- Condition definition
def exists_in_pascal (m : ℕ) : Prop :=
  ∃ n k, pascal n k = m

theorem smallest_four_digit_in_pascal :
  ∃ n k, pascal n k = 1000 ∧ ∀ m, (exists_in_pascal m) → (m >= 1000) → (m = 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_l229_229432


namespace probability_5_of_6_odd_rolls_l229_229011

def binom_coeff : ℕ → ℕ → ℕ
| n k := Nat.choose n k

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binom_coeff n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_5_of_6_odd_rolls :
  binomial_probability 6 5 (1/2) = 3/16 :=
by
  -- Proof will go here, but we skip it with sorry for now.
  sorry

end probability_5_of_6_odd_rolls_l229_229011


namespace rachel_plant_placement_l229_229736

def num_ways_to_place_plants : ℕ :=
  let plants := ["basil", "basil", "aloe", "cactus"]
  let lamps := ["white", "white", "red", "red"]
  -- we need to compute the number of ways to place 4 plants under 4 lamps
  22

theorem rachel_plant_placement :
  num_ways_to_place_plants = 22 :=
by
  -- Proof omitted for brevity
  sorry

end rachel_plant_placement_l229_229736


namespace relationship_of_inequalities_l229_229483

theorem relationship_of_inequalities (a b : ℝ) : 
  ¬ (∀ a b : ℝ, (a > b) → (a^2 > b^2)) ∧ 
  ¬ (∀ a b : ℝ, (a^2 > b^2) → (a > b)) := 
by 
  sorry

end relationship_of_inequalities_l229_229483


namespace problem_inequality_l229_229726

theorem problem_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 :=
sorry

end problem_inequality_l229_229726


namespace smallest_positive_expr_l229_229149

theorem smallest_positive_expr (m n : ℤ) : ∃ (m n : ℤ), 216 * m + 493 * n = 1 := 
sorry

end smallest_positive_expr_l229_229149


namespace last_three_digits_7_pow_123_l229_229343

theorem last_three_digits_7_pow_123 : (7^123 % 1000) = 717 := sorry

end last_three_digits_7_pow_123_l229_229343


namespace true_converses_count_l229_229653

-- Definitions according to the conditions
def parallel_lines (L1 L2 : Prop) : Prop := L1 ↔ L2
def congruent_triangles (T1 T2 : Prop) : Prop := T1 ↔ T2
def vertical_angles (A1 A2 : Prop) : Prop := A1 = A2
def squares_equal (m n : ℝ) : Prop := m = n → (m^2 = n^2)

-- Propositions with their converses
def converse_parallel (L1 L2 : Prop) : Prop := parallel_lines L1 L2 → parallel_lines L2 L1
def converse_congruent (T1 T2 : Prop) : Prop := congruent_triangles T1 T2 → congruent_triangles T2 T1
def converse_vertical (A1 A2 : Prop) : Prop := vertical_angles A1 A2 → vertical_angles A2 A1
def converse_squares (m n : ℝ) : Prop := (m^2 = n^2) → (m = n)

-- Proving the number of true converses
theorem true_converses_count : 
  (∃ L1 L2, converse_parallel L1 L2) →
  (∃ T1 T2, ¬converse_congruent T1 T2) →
  (∃ A1 A2, converse_vertical A1 A2) →
  (∃ m n : ℝ, ¬converse_squares m n) →
  (2 = 2) := by
  intros _ _ _ _
  sorry

end true_converses_count_l229_229653


namespace exist_amusing_numbers_l229_229728

/-- Definitions for an amusing number -/
def is_amusing (x : ℕ) : Prop :=
  (x >= 1000) ∧ (x <= 9999) ∧
  ∃ y : ℕ, y ≠ x ∧
  ((∀ d ∈ [x / 1000 % 10, x / 100 % 10, x / 10 % 10, x % 10],
    (d ≠ 0 → (y % 1000) ∈ [(d-1)*1000 + x / 100 % 10*100 + x / 10 % 10*10 + x % 10, (d+1)*1000 + x / 100 % 10*100 + x / 10 % 10*10 + x % 10]) ∧
    (d ≠ 9 → (y % 1000) ∈ [(d-1)*1000 + x / 100 % 10*100 + x / 10 % 10*10 + x % 10, (d+1)*1000 + x / 100 % 10*100 + x / 10 % 10*10 + x % 10]))) ∧
  (y % x = 0)

/-- Prove the existence of four amusing four-digit numbers -/
theorem exist_amusing_numbers :
  ∃ x1 x2 x3 x4 : ℕ, is_amusing x1 ∧ is_amusing x2 ∧ is_amusing x3 ∧ is_amusing x4 ∧ 
                   x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 :=
by sorry

end exist_amusing_numbers_l229_229728


namespace coeff_x3_in_product_l229_229980

open Polynomial

noncomputable def p : Polynomial ℤ := 3 * X^3 + 2 * X^2 + 5 * X + 3
noncomputable def q : Polynomial ℤ := 4 * X^3 + 5 * X^2 + 6 * X + 8

theorem coeff_x3_in_product :
  (p * q).coeff 3 = 61 :=
by sorry

end coeff_x3_in_product_l229_229980


namespace largest_four_digit_neg_int_congruent_mod_17_l229_229291

theorem largest_four_digit_neg_int_congruent_mod_17 :
  ∃ (n : ℤ), (-10000 < n) ∧ (n < -100) ∧ (n % 17 = 2) ∧ ∀ m, (-10000 < m) ∧ (m < -100) ∧ (m % 17 = 2) → m ≤ n :=
begin
  use -1001,
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  { sorry }
end

end largest_four_digit_neg_int_congruent_mod_17_l229_229291


namespace number_of_unanswered_questions_l229_229231

theorem number_of_unanswered_questions (n p q : ℕ) (h1 : p = 8) (h2 : q = 5) (h3 : n = 20)
(h4: ∃ s, s % 13 = 0) (hy : y = 0 ∨ y = 13) : 
  ∃ k, k = 20 ∨ k = 7 := by
  sorry

end number_of_unanswered_questions_l229_229231


namespace trigonometric_identity_l229_229583

open Real

theorem trigonometric_identity :
  sin (72 * pi / 180) * cos (12 * pi / 180) - cos (72 * pi / 180) * sin (12 * pi / 180) = sqrt 3 / 2 :=
by
  sorry

end trigonometric_identity_l229_229583


namespace num_ordered_triples_unique_l229_229344

theorem num_ordered_triples_unique : 
  (∃! (x y z : ℝ), x + y = 2 ∧ xy - z^2 = 1) := 
by 
  sorry 

end num_ordered_triples_unique_l229_229344


namespace oreo_shop_total_ways_l229_229964

def oreo_shop_ways : ℕ :=
  let flavors_oreos := 6
  let flavors_milk := 3
  let total_items := 3
  let total_choices := flavors_oreos + flavors_milk
  let no_item_more_than := 2
  
  if total_items > total_choices then 0 else
    -- Total ways of purchasing 3 items collectively given conditions
    ∑ i in [0, 1, 2, 3], 
      (if i ≤ total_items then 
        (Nat.choose total_choices i) * (Nat.choose total_choices (total_items - i))
      else 0)
  
theorem oreo_shop_total_ways : oreo_shop_ways = 708 := by
  sorry

end oreo_shop_total_ways_l229_229964


namespace smallest_four_digit_number_in_pascals_triangle_l229_229452

theorem smallest_four_digit_number_in_pascals_triangle :
  ∃ (n : ℕ), 1000 = (n choose min n) ∧ (n > 0 ∧ (∀ m < n, (m choose min m) < 1000)) :=
begin
  sorry
end

end smallest_four_digit_number_in_pascals_triangle_l229_229452


namespace smallest_four_digit_in_pascals_triangle_l229_229421

-- Define Pascal's triangle
def pascals_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 1
| n+1, k+1 => pascals_triangle n k + pascals_triangle n (k+1)

-- State the theorem
theorem smallest_four_digit_in_pascals_triangle : ∃ (n k : ℕ), pascals_triangle n k = 1000 ∧ ∀ m, m < 1000 → ∀ r s, pascals_triangle r s ≠ m :=
sorry

end smallest_four_digit_in_pascals_triangle_l229_229421


namespace sequence_increasing_l229_229828

theorem sequence_increasing (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ∀ n : ℕ, a^n / n^b < a^(n+1) / (n+1)^b :=
by sorry

end sequence_increasing_l229_229828


namespace better_fit_model_l229_229087

-- Define the residual sums of squares
def RSS_1 : ℝ := 152.6
def RSS_2 : ℝ := 159.8

-- Define the statement that the model with RSS_1 is the better fit
theorem better_fit_model : RSS_1 < RSS_2 → RSS_1 = 152.6 :=
by
  sorry

end better_fit_model_l229_229087


namespace proof_number_of_subsets_l229_229907

open Finset

-- Definition of the main problem statement
theorem proof_number_of_subsets (S : Finset ℕ) (T : Finset ℕ) 
  (hS : S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
  (hT : T.card = 4) 
  (h_diff : ∀ (x y ∈ T), x ≠ y → (x - y).nat_abs ≠ 1) :
  T.card = 35 := sorry

end proof_number_of_subsets_l229_229907


namespace sequence_sum_is_100_then_n_is_10_l229_229109

theorem sequence_sum_is_100_then_n_is_10 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 1 = 1) →
  (∀ n, a (n + 1) = a n + 2) →
  (∀ n, S n = n * a 1 + n * (n - 1)) →
  (∃ n, S n = 100) → 
  n = 10 :=
by sorry

end sequence_sum_is_100_then_n_is_10_l229_229109


namespace int_solutions_exist_for_x2_plus_15y2_eq_4n_l229_229387

theorem int_solutions_exist_for_x2_plus_15y2_eq_4n (n : ℕ) (hn : n > 0) : 
  ∃ S : Finset (ℤ × ℤ), S.card ≥ n ∧ ∀ (xy : ℤ × ℤ), xy ∈ S → xy.1^2 + 15 * xy.2^2 = 4^n :=
by
  sorry

end int_solutions_exist_for_x2_plus_15y2_eq_4n_l229_229387


namespace unique_integer_in_ranges_l229_229786

theorem unique_integer_in_ranges {x : ℤ} :
  1 < x ∧ x < 9 → 
  2 < x ∧ x < 15 → 
  -1 < x ∧ x < 7 → 
  0 < x ∧ x < 4 → 
  x + 1 < 5 → 
  x = 3 := by
  intros _ _ _ _ _
  sorry

end unique_integer_in_ranges_l229_229786


namespace triangle_area_B_equals_C_tan_ratio_sum_l229_229710

-- First part: Proving the area of the triangle
theorem triangle_area_B_equals_C {A B C a b c : ℝ} (h1 : B = C) (h2 : a = 2) (h3 : b^2 + c^2 = 3 * b * c * cos A) 
    (h4 : B + C = 180) :
    0.5 * b * c * sin A = sqrt(5) := 
by
  sorry

-- Second part: Proving the value of tan A / tan B + tan A / tan C
theorem tan_ratio_sum {A B C a b c : ℝ} (h1 : b^2 + c^2 = 3 * b * c * cos A) (h2 : A + B + C = 180) :
    (tan A / tan B) + (tan A / tan C) = 1 :=
by
  sorry

end triangle_area_B_equals_C_tan_ratio_sum_l229_229710


namespace smallest_four_digit_in_pascal_l229_229459

theorem smallest_four_digit_in_pascal :
  ∃ n k : ℕ, binomial n k = 1000 ∧ 
             (∀ a b : ℕ, binomial a b ∈ set.Icc 1000 9999 → a ≥ n) :=
sorry

end smallest_four_digit_in_pascal_l229_229459


namespace distance_travelled_downstream_in_12_minutes_l229_229926

noncomputable def speed_boat_still : ℝ := 15 -- in km/hr
noncomputable def rate_current : ℝ := 3 -- in km/hr
noncomputable def time_downstream : ℝ := 12 / 60 -- in hr (since 12 minutes is 12/60 hours)
noncomputable def effective_speed_downstream : ℝ := speed_boat_still + rate_current -- in km/hr
noncomputable def distance_downstream := effective_speed_downstream * time_downstream -- in km

theorem distance_travelled_downstream_in_12_minutes :
  distance_downstream = 3.6 := 
by
  sorry

end distance_travelled_downstream_in_12_minutes_l229_229926


namespace smallest_four_digit_in_pascals_triangle_l229_229422

-- Define Pascal's triangle
def pascals_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 1
| n+1, k+1 => pascals_triangle n k + pascals_triangle n (k+1)

-- State the theorem
theorem smallest_four_digit_in_pascals_triangle : ∃ (n k : ℕ), pascals_triangle n k = 1000 ∧ ∀ m, m < 1000 → ∀ r s, pascals_triangle r s ≠ m :=
sorry

end smallest_four_digit_in_pascals_triangle_l229_229422


namespace not_prime_p_l229_229233

theorem not_prime_p (x k p : ℕ) (h : x^5 + 2 * x + 3 = p * k) : ¬ (Nat.Prime p) :=
by
  sorry -- Placeholder for the proof

end not_prime_p_l229_229233


namespace fireworks_display_l229_229643

-- Define numbers and conditions
def display_fireworks_for_number (n : ℕ) : ℕ := 6
def display_fireworks_for_letter (c : Char) : ℕ := 5
def fireworks_per_box : ℕ := 8
def number_boxes : ℕ := 50

-- Calculate fireworks for the year 2023
def fireworks_for_year : ℕ :=
  display_fireworks_for_number 2 * 2 +
  display_fireworks_for_number 0 * 1 +
  display_fireworks_for_number 3 * 1

-- Calculate fireworks for "HAPPY NEW YEAR"
def fireworks_for_phrase : ℕ :=
  12 * display_fireworks_for_letter 'H'

-- Calculate fireworks for 50 boxes
def fireworks_for_boxes : ℕ := number_boxes * fireworks_per_box

-- Total fireworks calculation
def total_fireworks : ℕ := fireworks_for_year + fireworks_for_phrase + fireworks_for_boxes

-- Proof statement
theorem fireworks_display : total_fireworks = 476 := 
  by
  -- This is where the proof would go.
  sorry

end fireworks_display_l229_229643


namespace Christina_weekly_distance_l229_229513

/-- 
Prove that Christina covered 74 kilometers that week given the following conditions:
1. Christina walks 7km to school every day from Monday to Friday.
2. She returns home covering the same distance each day.
3. Last Friday, she had to pass by her friend, which is another 2km away from the school in the opposite direction from home.
-/
theorem Christina_weekly_distance : 
  let distance_to_school := 7
  let days_school := 5
  let extra_distance_Friday := 2
  let daily_distance := 2 * distance_to_school
  let total_distance_from_Monday_to_Thursday := 4 * daily_distance
  let distance_on_Friday := daily_distance + 2 * extra_distance_Friday
  total_distance_from_Monday_to_Thursday + distance_on_Friday = 74 := 
by
  sorry

end Christina_weekly_distance_l229_229513


namespace age_equation_correct_l229_229138

-- Define the main theorem
theorem age_equation_correct (x : ℕ) (h1 : ∀ (b : ℕ), b = 2 * x) (h2 : ∀ (b4 s4 : ℕ), b4 = b - 4 ∧ s4 = x - 4 ∧ b4 = 3 * s4) : 
  2 * x - 4 = 3 * (x - 4) :=
by
  sorry

end age_equation_correct_l229_229138


namespace difference_divisible_by_9_l229_229125

-- Define the integers a and b
variables (a b : ℤ)

-- Define the theorem statement
theorem difference_divisible_by_9 (a b : ℤ) : 9 ∣ ((3 * a + 2) ^ 2 - (3 * b + 2) ^ 2) :=
sorry

end difference_divisible_by_9_l229_229125


namespace common_difference_l229_229532

theorem common_difference (a : ℕ → ℤ) (d : ℤ) 
    (h1 : ∀ n : ℕ, a (n + 1) = a 1 + n * d)
    (h2 : a 1 + a 3 + a 5 = 15)
    (h3 : a 4 = 3) : 
    d = -2 := 
sorry

end common_difference_l229_229532


namespace total_time_l229_229313

/-- Define the different time periods in years --/
def getting_in_shape : ℕ := 2
def learning_to_climb : ℕ := 2 * getting_in_shape
def months_climbing : ℕ := 7 * 5
def climbing : ℚ := months_climbing / 12
def break_after_climbing : ℚ := 13 / 12
def diving : ℕ := 2

/-- Prove that the total time taken to achieve all goals is 12 years --/
theorem total_time : getting_in_shape + learning_to_climb + climbing + break_after_climbing + diving = 12 := by
  sorry

end total_time_l229_229313


namespace no_polyhedron_without_triangles_and_three_valent_vertices_l229_229735

-- Definitions and assumptions based on the problem's conditions
def f_3 := 0 -- no triangular faces
def p_3 := 0 -- no vertices with degree three

-- Euler's formula for convex polyhedra
def euler_formula (f p a : ℕ) : Prop := f + p - a = 2

-- Define general properties for faces and vertices in polyhedra
def polyhedron_no_triangular_no_three_valent (f p a f_4 f_5 p_4 p_5: ℕ) : Prop :=
  f_3 = 0 ∧ p_3 = 0 ∧ 2 * a ≥ 4 * (f_4 + f_5) ∧ 2 * a ≥ 4 * (p_4 + p_5) ∧ euler_formula f p a

-- Theorem to prove there does not exist such a polyhedron
theorem no_polyhedron_without_triangles_and_three_valent_vertices :
  ¬ ∃ (f p a f_4 f_5 p_4 p_5 : ℕ), polyhedron_no_triangular_no_three_valent f p a f_4 f_5 p_4 p_5 :=
by
  sorry

end no_polyhedron_without_triangles_and_three_valent_vertices_l229_229735


namespace edward_skee_ball_tickets_l229_229660

theorem edward_skee_ball_tickets (w_tickets : Nat) (candy_cost : Nat) (num_candies : Nat) (total_tickets : Nat) (skee_ball_tickets : Nat) :
  w_tickets = 3 ∧ candy_cost = 4 ∧ num_candies = 2 ∧ total_tickets = num_candies * candy_cost ∧ total_tickets - w_tickets = skee_ball_tickets → 
  skee_ball_tickets = 5 :=
by
  sorry

end edward_skee_ball_tickets_l229_229660


namespace find_m_and_c_l229_229677

-- Definitions & conditions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -1, y := 3 }
def B (m : ℝ) : Point := { x := -6, y := m }

def line (c : ℝ) (p : Point) : Prop := p.x + p.y + c = 0

-- Theorem statement
theorem find_m_and_c (m : ℝ) (c : ℝ) (hc : line c A) (hcB : line c (B m)) :
  m = 3 ∧ c = -2 :=
  by
  sorry

end find_m_and_c_l229_229677


namespace circle_center_radius_l229_229400

open Real

theorem circle_center_radius (x y : ℝ) :
  x^2 + y^2 - 6*x = 0 ↔ (x - 3)^2 + y^2 = 9 :=
by sorry

end circle_center_radius_l229_229400


namespace minValueExpr_ge_9_l229_229242

noncomputable def minValueExpr (x y z : ℝ) : ℝ :=
  (x + y) / z + (x + z) / y + (y + z) / x + 3

theorem minValueExpr_ge_9 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  minValueExpr x y z ≥ 9 :=
by sorry

end minValueExpr_ge_9_l229_229242


namespace geometric_sequence_product_correct_l229_229377

noncomputable def geometric_sequence_product (a_1 a_5 : ℝ) (a_2 a_3 a_4 : ℝ) :=
  a_1 = 1 / 2 ∧ a_5 = 8 ∧ a_2 * a_4 = a_1 * a_5 ∧ a_3^2 = a_1 * a_5

theorem geometric_sequence_product_correct:
  ∃ a_2 a_3 a_4 : ℝ, geometric_sequence_product (1 / 2) 8 a_2 a_3 a_4 ∧ (a_2 * a_3 * a_4 = 8) :=
by
  sorry

end geometric_sequence_product_correct_l229_229377


namespace constant_term_expansion_l229_229144

theorem constant_term_expansion (x : ℝ) : 
    (constant_term (3 * x + (2 / x)) ^ 8) = 90720 :=
by sorry

end constant_term_expansion_l229_229144


namespace sum_due_is_correct_l229_229746

-- Definitions of the given conditions
def BD : ℝ := 78
def TD : ℝ := 66

-- Definition of the sum due (S)
noncomputable def S : ℝ := (TD^2) / (BD - TD) + TD

-- The theorem to be proved
theorem sum_due_is_correct : S = 429 := by
  sorry

end sum_due_is_correct_l229_229746


namespace function_passes_through_A_l229_229681

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 + Real.log x / Real.log a

theorem function_passes_through_A 
  (a : ℝ) 
  (h1 : 0 < a) 
  (h2 : a ≠ 1)
  : f a 2 = 4 := sorry

end function_passes_through_A_l229_229681


namespace units_digit_17_pow_2023_l229_229036

theorem units_digit_17_pow_2023 : (17 ^ 2023) % 10 = 3 :=
by
  have units_cycle_7 : ∀ (n : ℕ), (7 ^ n) % 10 = [7, 9, 3, 1].nth (n % 4) :=
    sorry
  have units_pattern_equiv : (17 ^ n) % 10 = (7 ^ n) % 10 :=
    sorry
  calc
    (17 ^ 2023) % 10
        = (7 ^ 2023) % 10  : by rw [units_pattern_equiv]
    ... = 3               : by rw [units_cycle_7, nat.mod_eq_of_lt, List.nth]

end units_digit_17_pow_2023_l229_229036


namespace money_left_after_purchase_l229_229797

def initial_money : ℕ := 7
def cost_candy_bar : ℕ := 2
def cost_chocolate : ℕ := 3

def total_spent : ℕ := cost_candy_bar + cost_chocolate
def money_left : ℕ := initial_money - total_spent

theorem money_left_after_purchase : 
  money_left = 2 := by
  sorry

end money_left_after_purchase_l229_229797


namespace r_cube_plus_inv_r_cube_eq_zero_l229_229699

theorem r_cube_plus_inv_r_cube_eq_zero {r : ℝ} (h : (r + 1/r)^2 = 3) : r^3 + 1/r^3 = 0 := 
sorry

end r_cube_plus_inv_r_cube_eq_zero_l229_229699


namespace smallest_x_l229_229475

theorem smallest_x (x : ℕ) (h : 67 * 89 * x % 35 = 0) : x = 35 := 
by sorry

end smallest_x_l229_229475


namespace ezekiel_third_day_hike_l229_229342

-- Ezekiel's total hike distance
def total_distance : ℕ := 50

-- Distance covered on the first day
def first_day_distance : ℕ := 10

-- Distance covered on the second day
def second_day_distance : ℕ := total_distance / 2

-- Distance remaining for the third day
def third_day_distance : ℕ := total_distance - first_day_distance - second_day_distance

-- The distance Ezekiel had to hike on the third day
theorem ezekiel_third_day_hike : third_day_distance = 15 := by
  sorry

end ezekiel_third_day_hike_l229_229342


namespace ratio_final_to_initial_l229_229054

theorem ratio_final_to_initial (P R T : ℝ) (hR : R = 5) (hT : T = 20) :
  let SI := P * R * T / 100
  let A := P + SI
  A / P = 2 := 
by
  sorry

end ratio_final_to_initial_l229_229054


namespace number_of_paths_A_to_D_l229_229986

-- Definition of conditions
def ways_A_to_B : Nat := 2
def ways_B_to_C : Nat := 2
def ways_C_to_D : Nat := 2
def direct_A_to_D : Nat := 1

-- Theorem statement for the total number of paths from A to D
theorem number_of_paths_A_to_D : ways_A_to_B * ways_B_to_C * ways_C_to_D + direct_A_to_D = 9 := by
  sorry

end number_of_paths_A_to_D_l229_229986


namespace trigonometric_identity_l229_229655

theorem trigonometric_identity (α : ℝ) : 
  - (Real.sin α) + (Real.sqrt 3) * (Real.cos α) = 2 * (Real.sin (α + 2 * Real.pi / 3)) :=
by
  sorry

end trigonometric_identity_l229_229655


namespace problem_statement_l229_229214

variable {R : Type} [LinearOrderedField R]
variable (f : R → R)

theorem problem_statement
  (hf1 : ∀ x y : R, 0 < x ∧ x < 2 ∧ 0 < y ∧ y < 2 ∧ x < y → f x < f y)
  (hf2 : ∀ x : R, f (x + 2) = f (- (x + 2))) :
  f (7 / 2) < f 1 ∧ f 1 < f (5 / 2) :=
by
  sorry

end problem_statement_l229_229214


namespace general_formula_sum_of_b_l229_229675

variable {a : ℕ → ℕ} (b : ℕ → ℕ)
variable (n : ℕ)

def is_geometric_sequence (a : ℕ → ℕ) (q : ℕ) : Prop :=
  ∀ n : ℕ, a (n+2) = q * a (n+1)

def initial_conditions (a : ℕ → ℕ) : Prop :=
  a 1 + a 2 = 9 ∧ a 2 + a 3 = 18

theorem general_formula (q : ℕ) (h1 : is_geometric_sequence a q) (h2 : initial_conditions a) :
  a n = 3 * 2^(n - 1) :=
sorry

def b_n (a : ℕ → ℕ) (n : ℕ) : ℕ := a n + 2 * n

def sum_b (b : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum b

theorem sum_of_b (h1 : ∀ m : ℕ, b m = a m + 2 * m) (h2 : initial_conditions a) :
  sum_b b n = 3 * 2^n + n * (n + 1) - 3 :=
sorry

end general_formula_sum_of_b_l229_229675


namespace hall_length_l229_229613

theorem hall_length (b : ℕ) (h1 : b + 5 > 0) (h2 : (b + 5) * b = 750) : b + 5 = 30 :=
by {
  -- Proof goes here
  sorry
}

end hall_length_l229_229613


namespace minimum_value_expression_l229_229245

variable (x y z : ℝ)
variable (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)

theorem minimum_value_expression : 
  (\frac{x + y}{z} + \frac{x + z}{y} + \frac{y + z}{x} + 3) ≥ 9 :=
by
  sorry

end minimum_value_expression_l229_229245


namespace problem_sin_cos_k_l229_229130

open Real

theorem problem_sin_cos_k {k : ℝ} :
  (∃ x : ℝ, sin x ^ 2 + cos x + k = 0) ↔ -2 ≤ k ∧ k ≤ 0 := by
  sorry

end problem_sin_cos_k_l229_229130


namespace largest_fraction_consecutive_primes_l229_229531

theorem largest_fraction_consecutive_primes (p q r s : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s)
  (h0 : 0 < p) (h1 : p < q) (h2 : q < r) (h3 : r < s)
  (hconsec : p + 2 = q ∧ q + 2 = r ∧ r + 2 = s) :
  (r + s) / (p + q) > max ((p + q) / (r + s)) (max ((p + s) / (q + r)) (max ((q + r) / (p + s)) ((q + s) / (p + r)))) :=
sorry

end largest_fraction_consecutive_primes_l229_229531


namespace truncated_pyramid_lateral_surface_area_l229_229320

noncomputable def lateralSurfaceAreaTruncatedPyramid (s1 s2 h : ℝ) :=
  let l := Real.sqrt (h^2 + ((s1 - s2) / 2)^2)
  let P1 := 4 * s1
  let P2 := 4 * s2
  (1 / 2) * (P1 + P2) * l

theorem truncated_pyramid_lateral_surface_area :
  lateralSurfaceAreaTruncatedPyramid 10 5 7 = 222.9 :=
by
  sorry

end truncated_pyramid_lateral_surface_area_l229_229320


namespace tan_pi_over_4_plus_alpha_eq_two_l229_229535

theorem tan_pi_over_4_plus_alpha_eq_two
  (α : ℂ) 
  (h : Complex.tan ((π / 4) + α) = 2) : 
  (1 / (2 * Complex.sin α * Complex.cos α + (Complex.cos α)^2)) = (2 / 3) :=
by
  sorry

end tan_pi_over_4_plus_alpha_eq_two_l229_229535


namespace width_of_plot_is_60_l229_229497

-- Defining the conditions
def length_of_plot := 90
def distance_between_poles := 5
def number_of_poles := 60

-- The theorem statement
theorem width_of_plot_is_60 :
  ∃ width : ℕ, 2 * (length_of_plot + width) = number_of_poles * distance_between_poles ∧ width = 60 :=
sorry

end width_of_plot_is_60_l229_229497


namespace girls_with_no_pets_l229_229865

-- Define the conditions
def total_students : ℕ := 30
def fraction_boys : ℚ := 1 / 3
def fraction_girls : ℚ := 1 - fraction_boys
def girls_with_dogs_fraction : ℚ := 40 / 100
def girls_with_cats_fraction : ℚ := 20 / 100
def girls_with_no_pets_fraction : ℚ := 1 - (girls_with_dogs_fraction + girls_with_cats_fraction)

-- Calculate the number of girls
def total_girls : ℕ := total_students * fraction_girls.to_nat
def number_girls_with_no_pets : ℕ := total_girls * girls_with_no_pets_fraction.to_nat

-- Theorem statement
theorem girls_with_no_pets : number_girls_with_no_pets = 8 :=
by sorry

end girls_with_no_pets_l229_229865


namespace max_quadratic_in_interval_l229_229132

-- Define the quadratic function
noncomputable def quadratic_fun (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define the closed interval
def interval (a b : ℝ) (x : ℝ) : Prop := a ≤ x ∧ x ≤ b

-- Define the maximum value property
def is_max_value (f : ℝ → ℝ) (a b max_val : ℝ) : Prop :=
  ∀ x, interval a b x → f x ≤ max_val

-- State the problem in Lean 4
theorem max_quadratic_in_interval : 
  is_max_value quadratic_fun (-5) 3 36 := 
sorry

end max_quadratic_in_interval_l229_229132


namespace solve_z_squared_eq_l229_229805

open Complex

theorem solve_z_squared_eq : 
  ∀ z : ℂ, z^2 = -100 - 64 * I → (z = 4 - 8 * I ∨ z = -4 + 8 * I) :=
by
  sorry

end solve_z_squared_eq_l229_229805


namespace sum_of_first_nine_terms_l229_229368

theorem sum_of_first_nine_terms (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 1 = 3 * a 3 - 6) : 
  (9 * (a 0 + a 8)) / 2 = 27 := 
sorry

end sum_of_first_nine_terms_l229_229368


namespace probability_of_5_odd_numbers_in_6_rolls_is_3_over_32_l229_229013

def probability_of_5_odd_numbers_in_6_rolls (prob_odd : ℚ) : ℚ :=
  (nat.choose 6 5 * (prob_odd^5) * ((1 - prob_odd)^1)) / (2^6)

theorem probability_of_5_odd_numbers_in_6_rolls_is_3_over_32 :
  probability_of_5_odd_numbers_in_6_rolls (1/2) = 3 / 32 :=
by sorry

end probability_of_5_odd_numbers_in_6_rolls_is_3_over_32_l229_229013


namespace correct_calc_value_l229_229225

theorem correct_calc_value (x : ℕ) (h : 2 * (3 * x + 14) = 946) : 2 * (x / 3 + 14) = 130 := 
by
  sorry

end correct_calc_value_l229_229225


namespace sum_of_997_lemons_l229_229869

-- Define x and y as functions of k
def x (k : ℕ) := 1 + 9 * k
def y (k : ℕ) := 110 - 7 * k

-- The theorem we need to prove
theorem sum_of_997_lemons :
  ∃ (k : ℕ), 0 ≤ k ∧ k ≤ 15 ∧ 7 * (x k) + 9 * (y k) = 997 := 
by
  sorry -- Proof to be filled in

end sum_of_997_lemons_l229_229869


namespace range_of_x_l229_229405

theorem range_of_x (x : ℝ) (h : 1 - x ≥ 0) : x ≤ 1 :=
  sorry

end range_of_x_l229_229405


namespace book_donation_growth_rate_l229_229156

theorem book_donation_growth_rate (x : ℝ) : 
  400 + 400 * (1 + x) + 400 * (1 + x)^2 = 1525 :=
sorry

end book_donation_growth_rate_l229_229156


namespace rising_number_fifty_l229_229212

/-- Define a valid four-digit rising number from digits 1 to 8, represented as a list of four integers where each list element is greater than its predecessor. -/
def is_rising_four_digit (num: List ℕ) : Prop :=
  num.length = 4 ∧ ∀ i j, (i < j ∧ j < 4) → num.nth i < num.nth j

/-- Define our specific conditions for the problem -/
def fifty_rising_number : List ℕ := [2, 3, 6, 7]

/-- The set of digits we are investigating -/
def digit_set : Set ℕ := {3, 4, 5, 6, 7}

/-- The proof problem statement to be shown in Lean -/
theorem rising_number_fifty :
  is_rising_four_digit fifty_rising_number ∧ (∀ d ∈ digit_set, d ≠ 2 ∧ d ≠ 3 ∧ d ≠ 6 ∧ d ≠ 7 → d = 4) :=
by
  sorry

end rising_number_fifty_l229_229212


namespace greatest_possible_bent_strips_l229_229136

theorem greatest_possible_bent_strips (strip_count : ℕ) (cube_length cube_faces flat_strip_cover : ℕ) 
  (unit_squares_per_face total_squares flat_strips unit_squares_covered_by_flats : ℕ):
  strip_count = 18 →
  cube_length = 3 →
  cube_faces = 6 →
  flat_strip_cover = 3 →
  unit_squares_per_face = cube_length * cube_length →
  total_squares = cube_faces * unit_squares_per_face →
  flat_strips = 4 →
  unit_squares_covered_by_flats = flat_strips * flat_strip_cover →
  ∃ bent_strips,
  flat_strips * flat_strip_cover + bent_strips * flat_strip_cover = total_squares 
  ∧ bent_strips = 14 := by
  intros
  -- skipped proof
  sorry

end greatest_possible_bent_strips_l229_229136


namespace diana_wins_l229_229337

noncomputable def probability_diana_wins : ℚ :=
  45 / 100

theorem diana_wins (d : ℕ) (a : ℕ) (hd : 1 ≤ d ∧ d ≤ 10) (ha : 1 ≤ a ∧ a ≤ 10) :
  probability_diana_wins = 9 / 20 :=
by
  sorry

end diana_wins_l229_229337


namespace stockholm_to_malmo_road_distance_l229_229748

-- Define constants based on the conditions
def map_distance_cm : ℕ := 120
def scale_factor : ℕ := 10
def road_distance_multiplier : ℚ := 1.15

-- Define the real distances based on the conditions
def straight_line_distance_km : ℕ :=
  map_distance_cm * scale_factor

def road_distance_km : ℚ :=
  straight_line_distance_km * road_distance_multiplier

-- Assert the final statement
theorem stockholm_to_malmo_road_distance :
  road_distance_km = 1380 := 
sorry

end stockholm_to_malmo_road_distance_l229_229748


namespace hyperbola_asymptote_m_l229_229517

def isAsymptote (x y : ℝ) (m : ℝ) : Prop :=
  y = m * x ∨ y = -m * x

theorem hyperbola_asymptote_m (m : ℝ) : 
  (∀ x y, (x^2 / 25 - y^2 / 16 = 1 → isAsymptote x y m)) ↔ m = 4 / 5 := 
by
  sorry

end hyperbola_asymptote_m_l229_229517


namespace inequal_f_i_sum_mn_ii_l229_229084

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 3 / 2 then -2 
  else if x > -5 / 2 then -x - 1 / 2 
  else 2

theorem inequal_f_i (a : ℝ) : (∀ x : ℝ, f x ≥ a^2 - 3 * a) ↔ (1 ≤ a ∧ a ≤ 2) :=
sorry

theorem sum_mn_ii (m n : ℝ) (h1 : f m + f n = 4) (h2 : m < n) : m + n < -5 :=
sorry

end inequal_f_i_sum_mn_ii_l229_229084


namespace fraction_a_over_b_l229_229096

theorem fraction_a_over_b (x y a b : ℝ) (h1 : 2 * x - y = a) (h2 : 3 * y - 6 * x = b) (hb : b ≠ 0) : a / b = -1 / 3 :=
by
  sorry

end fraction_a_over_b_l229_229096


namespace soda_cost_is_20_l229_229944

noncomputable def cost_of_soda (b s : ℕ) : Prop :=
  4 * b + 3 * s = 500 ∧ 3 * b + 2 * s = 370

theorem soda_cost_is_20 {b s : ℕ} (h : cost_of_soda b s) : s = 20 :=
  by sorry

end soda_cost_is_20_l229_229944


namespace prove_f_2_eq_10_prove_f_4_eq_10_prove_f_prove_f_l229_229668

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom f_cond1 : ∀ x : ℝ, f x + (deriv g x) = 10
axiom f_cond2 : ∀ x : ℝ, f x - (deriv g (4 - x)) = 10
axiom g_even : ∀ x : ℝ, g x = g (-x)

theorem prove_f_2_eq_10 : f 2 = 10 := sorry
theorem prove_f_4_eq_10 : f 4 = 10 := sorry
theorem prove_f'_neg1_eq_f'_neg3 : deriv f (-1) = deriv f (-3) := sorry
theorem prove_f'_2023_ne_0 : deriv f 2023 ≠ 0 := sorry

end prove_f_2_eq_10_prove_f_4_eq_10_prove_f_prove_f_l229_229668


namespace compute_alpha_l229_229379

-- Define the main hypothesis with complex numbers
variable (α γ : ℂ)
variable (h1 : γ = 4 + 3 * Complex.I)
variable (h2 : ∃r1 r2: ℝ, r1 > 0 ∧ r2 > 0 ∧ (α + γ = r1) ∧ (Complex.I * (α - 3 * γ) = r2))

-- The main theorem
theorem compute_alpha : α = 12 + 3 * Complex.I :=
by
  sorry

end compute_alpha_l229_229379


namespace variation_relationship_l229_229362

theorem variation_relationship (k j : ℝ) (y z x : ℝ) (h1 : x = k * y^3) (h2 : y = j * z^(1/5)) :
  ∃ m : ℝ, x = m * z^(3/5) :=
by
  sorry

end variation_relationship_l229_229362


namespace prime_5p_plus_4p4_is_perfect_square_l229_229183

theorem prime_5p_plus_4p4_is_perfect_square (p : ℕ) [Fact (Nat.Prime p)] :
  ∃ q : ℕ, 5^p + 4 * p^4 = q^2 ↔ p = 5 :=
by
  sorry

end prime_5p_plus_4p4_is_perfect_square_l229_229183


namespace smallest_positive_integer_ends_in_9_and_divisible_by_13_l229_229023

theorem smallest_positive_integer_ends_in_9_and_divisible_by_13 :
  ∃ n : ℕ, n % 10 = 9 ∧ 13 ∣ n ∧ n > 0 ∧ ∀ m, m % 10 = 9 → 13 ∣ m ∧ m > 0 → m ≥ n := 
begin
  use 99,
  split,
  { exact mod_eq_of_lt (10*k + 9) 10 99 9 (by norm_num), },
  split,
  { exact dvd_refl 99, },
  split,
  { exact zero_lt_99, },
  intros m hm1 hm2 hpos,
  by_contradiction hmn,
  sorry
end

end smallest_positive_integer_ends_in_9_and_divisible_by_13_l229_229023


namespace selection_methods_l229_229627

theorem selection_methods (students : ℕ) (boys : ℕ) (girls : ℕ) (selected : ℕ) (h1 : students = 8) (h2 : boys = 6) (h3 : girls = 2) (h4 : selected = 4) : 
  ∃ methods, methods = 40 :=
by
  have h5 : students = boys + girls := by linarith
  sorry

end selection_methods_l229_229627


namespace number_of_true_propositions_l229_229411

noncomputable def proposition1 : Prop := ∀ (x : ℝ), x^2 - 3 * x + 2 > 0
noncomputable def proposition2 : Prop := ∃ (x : ℚ), x^2 = 2
noncomputable def proposition3 : Prop := ∃ (x : ℝ), x^2 - 1 = 0
noncomputable def proposition4 : Prop := ∀ (x : ℝ), 4 * x^2 > 2 * x - 1 + 3 * x^2

theorem number_of_true_propositions : (¬ proposition1 ∧ ¬ proposition2 ∧ proposition3 ∧ ¬ proposition4) → 1 = 1 :=
by
  intros
  sorry

end number_of_true_propositions_l229_229411


namespace james_collects_15_gallons_per_inch_l229_229235

def rain_gallons_per_inch (G : ℝ) : Prop :=
  let monday_rain := 4
  let tuesday_rain := 3
  let price_per_gallon := 1.2
  let total_money := 126
  let total_rain := monday_rain + tuesday_rain
  (total_rain * G = total_money / price_per_gallon)

theorem james_collects_15_gallons_per_inch : rain_gallons_per_inch 15 :=
by
  -- This is the theorem statement; the proof is not required.
  sorry

end james_collects_15_gallons_per_inch_l229_229235


namespace average_cd_l229_229590

theorem average_cd (c d : ℝ) (h : (4 + 6 + 8 + c + d) / 5 = 18) : (c + d) / 2 = 36 := 
by
  -- The proof goes here
  sorry

end average_cd_l229_229590


namespace matrix_expression_l229_229539

variable {F : Type} [Field F] {n : Type} [Fintype n] [DecidableEq n]
variable (B : Matrix n n F)

-- Suppose B is invertible
variable [Invertible B]

-- Condition given in the problem
theorem matrix_expression (h : (B - 3 • (1 : Matrix n n F)) * (B - 5 • (1 : Matrix n n F)) = 0) :
  B + 10 • (B⁻¹) = 10 • (B⁻¹) + (32 / 3 : F) • (1 : Matrix n n F) :=
sorry

end matrix_expression_l229_229539


namespace factorial_expression_l229_229066

theorem factorial_expression :
  7 * (Nat.factorial 7) + 6 * (Nat.factorial 6) + 2 * (Nat.factorial 6) = 41040 := by
  sorry

end factorial_expression_l229_229066


namespace irreducible_fraction_l229_229887

theorem irreducible_fraction (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 := 
by sorry

end irreducible_fraction_l229_229887


namespace joan_initial_balloons_l229_229112

-- Definitions using conditions from a)
def initial_balloons (lost : ℕ) (current : ℕ) : ℕ := lost + current

-- Statement of our equivalent math proof problem
theorem joan_initial_balloons : initial_balloons 2 7 = 9 := 
by
  -- Proof skipped using sorry
  sorry

end joan_initial_balloons_l229_229112


namespace find_m_plus_n_l229_229047

def rolls : Type := {x : Fin 16 // x < 4}

def guests (rolls : List (Fin 4)) : Prop :=
  ∀ g, g.length = 4 ∧ rolls.nodup ∧ (∀ r ∈ g, r < 4)

noncomputable def probability_each_gets_one_of_each (rolls : List (Fin 4)) : ℚ :=
  if guests rolls then 
    (16/455) * (6/165) * (15/168)
  else 0

theorem find_m_plus_n : ∀ (rolls : List (Fin 4)),
  guests rolls -> 
  let p := probability_each_gets_one_of_each rolls in
  let m := Rat.num p in
  let n := Rat.denom p in
  (Nat.gcd m n = 1) -> 
  m + n = 8783 :=
by
  intros
  have h : p = (10/8773) := sorry
  have prime_gcd : Nat.gcd 10 8773 = 1 := sorry
  sorry

end find_m_plus_n_l229_229047


namespace g_at_5_l229_229271

-- Define the function g(x) that satisfies the given condition
def g (x : ℝ) : ℝ := sorry

-- Axiom stating that the function g satisfies the given equation for all x ∈ ℝ
axiom g_condition : ∀ x : ℝ, 3 * g x + 4 * g (1 - x) = 6 * x^2

-- The theorem to prove
theorem g_at_5 : g 5 = -66 / 7 :=
by
  -- Proof will be added here.
  sorry

end g_at_5_l229_229271


namespace disjoint_subsets_same_sum_l229_229121

-- Define the main theorem
theorem disjoint_subsets_same_sum (S : Finset ℕ) (hS_len : S.card = 10) (hS_range : ∀ x ∈ S, 10 ≤ x ∧ x ≤ 99) :
  ∃ A B : Finset ℕ, A ∩ B = ∅ ∧ A ≠ ∅ ∧ B ≠ ∅ ∧ A.sum id = B.sum id :=
by {
  sorry
}

end disjoint_subsets_same_sum_l229_229121


namespace not_possible_to_obtain_target_triple_l229_229057

def is_target_triple_achievable (a1 a2 a3 b1 b2 b3 : ℝ) : Prop :=
  ∀ x y : ℝ, (x, y) = (0.6 * x - 0.8 * y, 0.8 * x + 0.6 * y) →
    (b1^2 + b2^2 + b3^2 = 169 → False)

theorem not_possible_to_obtain_target_triple :
  ¬ is_target_triple_achievable 3 4 12 2 8 10 :=
by sorry

end not_possible_to_obtain_target_triple_l229_229057


namespace adjacent_number_in_grid_l229_229616

def adjacent_triangle_number (k n: ℕ) : ℕ :=
  if k % 2 = 1 then n - k else n + k

theorem adjacent_number_in_grid (n : ℕ) (bound: n = 350) :
  let k := Nat.ceil (Real.sqrt n)
  let m := (k * k) - n
  k = 19 ∧ m = 19 →
  adjacent_triangle_number k n = 314 :=
by
  sorry

end adjacent_number_in_grid_l229_229616


namespace no_prime_divisible_by_77_l229_229694

def is_prime (n : ℕ) : Prop := 
  (n > 1) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_divisible (a b : ℕ) : Prop := b ∣ a

theorem no_prime_divisible_by_77 : ∀ p : ℕ, is_prime p → is_divisible p 77 → false :=
by
  sorry

end no_prime_divisible_by_77_l229_229694


namespace sum_of_fully_paintable_numbers_l229_229861

def is_fully_paintable (h t u : ℕ) : Prop :=
  (∀ n : ℕ, (∀ k1 : ℕ, n ≠ 1 + k1 * h) ∧ (∀ k2 : ℕ, n ≠ 3 + k2 * t) ∧ (∀ k3 : ℕ, n ≠ 2 + k3 * u)) → False

theorem sum_of_fully_paintable_numbers :  ∃ L : List ℕ, (∀ x ∈ L, ∃ (h t u : ℕ), is_fully_paintable h t u ∧ 100 * h + 10 * t + u = x) ∧ L.sum = 944 :=
sorry

end sum_of_fully_paintable_numbers_l229_229861


namespace combination_simplify_l229_229515

theorem combination_simplify : (Nat.choose 6 2) + 3 = 18 := by
  sorry

end combination_simplify_l229_229515


namespace find_a1_l229_229248

variable {α : Type*} [LinearOrderedField α]

noncomputable def arithmeticSequence (a d : α) (n : ℕ) : α :=
  a + (n - 1) * d

noncomputable def sumOfArithmeticSequence (a d : α) (n : ℕ) : α :=
  n * a + d * (n * (n - 1) / 2)

theorem find_a1 (a1 d : α) :
  arithmeticSequence a1 d 2 + arithmeticSequence a1 d 8 = 34 →
  sumOfArithmeticSequence a1 d 4 = 38 →
  a1 = 5 :=
by
  intros h1 h2
  sorry

end find_a1_l229_229248


namespace smallest_four_digit_in_pascals_triangle_l229_229468

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, (1000 ≤ (nat.choose n k)) ∧ (∀ m l : ℕ, 1000 ≤ nat.choose m l → nat.choose m l ≥ 1000) :=
begin
  use [1000, 1],
  split,
  { exact nat.choose_eq_one_right 1000 1 },
  { intros m l hml,
    linarith }
end

end smallest_four_digit_in_pascals_triangle_l229_229468


namespace number_of_pencils_l229_229909

-- Definitions based on the conditions
def ratio_pens_pencils (P L : ℕ) : Prop := P * 6 = 5 * L
def pencils_more_than_pens (P L : ℕ) : Prop := L = P + 4

-- Statement to prove the number of pencils
theorem number_of_pencils : ∃ L : ℕ, (∃ P : ℕ, ratio_pens_pencils P L ∧ pencils_more_than_pens P L) ∧ L = 24 :=
by
  sorry

end number_of_pencils_l229_229909


namespace lambda_5_geq_2_sin_54_l229_229527

-- Define the problem
theorem lambda_5_geq_2_sin_54 (points : Fin 5 → ℝ × ℝ) :
  let distances := {dist | ∃ i j, i ≠ j ∧ dist = (dist.points points i j)} in
  let λ5 := (distances.max' sorry) / (distances.min' sorry) in
  λ5 ≥ 2 * Real.sin (54 * Real.pi / 180) ∧
  (λ5 = 2 * Real.sin (54 * Real.pi / 180) ↔
   ∃ pentagon, (∀ i, ∃ j, j ∈ pentagon ∧ j ≠ i ∧ dist.points pentagon i j) ∧
   RegularPentagon pentagon) :=
sorry

-- Define helper functions
noncomputable def dist.points (points : Fin 5 → ℝ × ℝ) (i j : Fin 5) : ℝ :=
  Real.sqrt (points i - points j).1^2 + (points i - points j).2^2

-- Regular Pentagon
structure RegularPentagon (points : Fin 5 → ℝ × ℝ) : Prop :=
(equilateral : ∀ i j, dist.points points i j = dist.points points (Fin.iterate Fin.succ 2 i) (Fin.iterate Fin.succ 2 j))

end lambda_5_geq_2_sin_54_l229_229527


namespace books_per_shelf_l229_229281

theorem books_per_shelf (total_books : ℕ) (total_shelves : ℕ) (h_total_books : total_books = 2250) (h_total_shelves : total_shelves = 150) :
  total_books / total_shelves = 15 :=
by
  sorry

end books_per_shelf_l229_229281


namespace not_prime_5n_plus_3_l229_229227

def isSquare (x : ℕ) : Prop := ∃ (k : ℕ), k * k = x

theorem not_prime_5n_plus_3 (n k m : ℕ) (h₁ : 2 * n + 1 = k * k) (h₂ : 3 * n + 1 = m * m) (n_pos : 0 < n) (k_pos : 0 < k) (m_pos : 0 < m) :
  ¬ Nat.Prime (5 * n + 3) :=
sorry -- Proof to be completed

end not_prime_5n_plus_3_l229_229227


namespace reroll_probability_two_dice_l229_229237

noncomputable def optimized_reroll_probability : ℚ :=
  1 / 72

/-- 
Jason rolls three fair six-sided dice. He then decides to reroll any subset of these dice.
Jason wins if the sum of the numbers face up on the three dice after rerolls is exactly 9.
Jason always plays to optimize his chances of winning.
-/
theorem reroll_probability_two_dice : 
  let prob := 1 / 72 in
  optimized_reroll_probability = prob :=
by
  sorry

end reroll_probability_two_dice_l229_229237


namespace whale_population_ratio_l229_229595

theorem whale_population_ratio 
  (W_last : ℕ)
  (W_this : ℕ)
  (W_next : ℕ)
  (h1 : W_last = 4000)
  (h2 : W_next = W_this + 800)
  (h3 : W_next = 8800) :
  (W_this / W_last) = 2 := by
  sorry

end whale_population_ratio_l229_229595


namespace number_of_ways_to_feed_animals_l229_229309

-- Definitions for the conditions
def pairs_of_animals := 5
def alternating_feeding (start_with_female : Bool) (remaining_pairs : ℕ) : ℕ :=
if start_with_female then
  (pairs_of_animals.factorial / 2 ^ pairs_of_animals)
else
  0 -- we can ignore this case as it is not needed

-- Theorem statement
theorem number_of_ways_to_feed_animals :
  alternating_feeding true pairs_of_animals = 2880 :=
sorry

end number_of_ways_to_feed_animals_l229_229309


namespace rhombus_longer_diagonal_l229_229630

theorem rhombus_longer_diagonal (a b : ℕ) (d1 : ℕ) (d2 : ℕ) (h0 : a = 65) (h1 : d1 = 56) (h2 : a * a = (d1 / 2) * (d1 / 2) + (d2 / 2) * (d2 / 2)) :
  d2 = 118 :=
by
  have h3 : (d1 / 2) * (d1 / 2) + (d2 / 2) * (d2 / 2) = 65 * 65, from h2,
  rw [←h0, ←h1] at h3,
  have h4 : (28 : ℕ) = d1 / 2, sorry,
  have h5 : (28 : ℕ) * (28 : ℕ) + ((d2 / 2) * (d2 / 2)) = 65 * 65, from h3,
  have h6 : (28 * 28) + (d2 / 2) * (d2 / 2) = (65*65), by rw h5,
  have h7 : (65 * 65) - 784 = (d2 / 2) * (d2 / 2), sorry,
  have h8 : (d2 / 2) * (d2 / 2) = 3441, by simp[h7],
  have h9 : (d2 / 2) = 59, from nat.sqrt_eq 3441,
  have h10 : d2 = 2 * 59, by rw [h9, two_mul],
  exact h10
 

end rhombus_longer_diagonal_l229_229630


namespace bill_face_value_l229_229480

theorem bill_face_value
  (TD : ℝ) (T : ℝ) (r : ℝ) (FV : ℝ)
  (h1 : TD = 210)
  (h2 : T = 0.75)
  (h3 : r = 0.16) :
  FV = 1960 :=
by 
  sorry

end bill_face_value_l229_229480


namespace coles_average_speed_l229_229652

theorem coles_average_speed (t_work : ℝ) (t_round : ℝ) (s_return : ℝ) (t_return : ℝ) (d : ℝ) (t_work_min : ℕ) :
  t_work_min = 72 ∧ t_round = 2 ∧ s_return = 90 ∧ 
  t_work = t_work_min / 60 ∧ t_return = t_round - t_work ∧ d = s_return * t_return →
  d / t_work = 60 := 
by
  intro h
  sorry

end coles_average_speed_l229_229652


namespace worksheets_turned_in_l229_229502

def initial_worksheets : ℕ := 34
def graded_worksheets : ℕ := 7
def remaining_worksheets : ℕ := initial_worksheets - graded_worksheets
def current_worksheets : ℕ := 63

theorem worksheets_turned_in :
  current_worksheets - remaining_worksheets = 36 :=
by
  sorry

end worksheets_turned_in_l229_229502


namespace rhombus_longer_diagonal_l229_229636

theorem rhombus_longer_diagonal
  (a b : ℝ)
  (ha : a = 65) -- Side length
  (hb : b = 28) -- Half the length of the shorter diagonal
  : 2 * Real.sqrt (a^2 - b^2) = 118 :=  -- Statement of the longer diagonal length
by
  sorry -- Proof goes here

end rhombus_longer_diagonal_l229_229636


namespace smallest_four_digit_in_pascals_triangle_l229_229439

/-- Pascal's triangle is defined using binomial coefficients.
    We want to prove that the smallest four-digit number in Pascal's triangle is 1001. -/
theorem smallest_four_digit_in_pascals_triangle : ∃ n k : ℕ, n ≥ k ∧ (binomial n k = 1001 ∧ ∀ m l, m ≥ l → binomial m l ≥ 1000 → binomial m l = 1001 → n = m ∧ k = l) := by
  sorry

end smallest_four_digit_in_pascals_triangle_l229_229439


namespace car_speed_on_local_roads_l229_229486

theorem car_speed_on_local_roads
    (v : ℝ) -- Speed of the car on local roads
    (h1 : v > 0) -- The speed is positive
    (h2 : 40 / v + 3 = 5) -- Given equation based on travel times and distances
    : v = 20 := 
sorry

end car_speed_on_local_roads_l229_229486


namespace intersection_A_B_subset_A_B_l229_229357

-- Definition of sets A and B
def set_A (a : ℝ) : Set ℝ := {x | 0 < 2 * x + a ∧ 2 * x + a ≤ 3}
def set_B : Set ℝ := {x | -1 / 2 < x ∧ x < 2}

-- Problem 1: Prove A ∩ B when a = -1
theorem intersection_A_B (a : ℝ) (h : a = -1) : set_A a ∩ set_B = {x | 1 / 2 < x ∧ x < 2} :=
sorry

-- Problem 2: Find the range of a such that A ⊆ B
theorem subset_A_B (a : ℝ) : (-1 < a ∧ a ≤ 1) ↔ (set_A a ⊆ set_B) :=
sorry

end intersection_A_B_subset_A_B_l229_229357


namespace sum_of_y_neg_l229_229911

-- Define the conditions from the problem
def condition1 (x y : ℝ) : Prop := x + y = 7
def condition2 (x z : ℝ) : Prop := x * z = -180
def condition3 (x y z : ℝ) : Prop := (x + y + z)^2 = 4

-- Define the main theorem to prove
theorem sum_of_y_neg (x y z : ℝ) (S : ℝ) :
  (condition1 x y) ∧ (condition2 x z) ∧ (condition3 x y z) →
  (S = (-29) + (-13)) →
  -S = 42 :=
by
  sorry

end sum_of_y_neg_l229_229911


namespace sum_of_reflected_midpoint_coords_l229_229256

theorem sum_of_reflected_midpoint_coords (P R : ℝ × ℝ) 
  (hP : P = (2, 1)) (hR : R = (12, 15)) :
  let M := ((P.1 + R.1) / 2, (P.2 + R.2) / 2)
  let P' := (-P.1, P.2)
  let R' := (-R.1, R.2)
  let M' := ((P'.1 + R'.1) / 2, (P'.2 + R'.2) / 2)
  M'.1 + M'.2 = 1 :=
by
  sorry

end sum_of_reflected_midpoint_coords_l229_229256


namespace smallest_four_digit_in_pascal_l229_229461

theorem smallest_four_digit_in_pascal :
  ∃ n k : ℕ, binomial n k = 1000 ∧ 
             (∀ a b : ℕ, binomial a b ∈ set.Icc 1000 9999 → a ≥ n) :=
sorry

end smallest_four_digit_in_pascal_l229_229461


namespace division_multiplication_expression_l229_229620

theorem division_multiplication_expression : 377 / 13 / 29 * 1 / 4 / 2 = 0.125 :=
by
  sorry

end division_multiplication_expression_l229_229620


namespace added_number_is_nine_l229_229495

theorem added_number_is_nine (y : ℤ) : 
  3 * (2 * 4 + y) = 51 → y = 9 :=
by
  sorry

end added_number_is_nine_l229_229495


namespace average_salary_l229_229591

def A_salary : ℝ := 9000
def B_salary : ℝ := 5000
def C_salary : ℝ := 11000
def D_salary : ℝ := 7000
def E_salary : ℝ := 9000
def number_of_people : ℝ := 5
def total_salary : ℝ := A_salary + B_salary + C_salary + D_salary + E_salary

theorem average_salary : (total_salary / number_of_people) = 8200 := by
  sorry

end average_salary_l229_229591


namespace three_point_seven_five_minus_one_point_four_six_l229_229647

theorem three_point_seven_five_minus_one_point_four_six : 3.75 - 1.46 = 2.29 :=
by sorry

end three_point_seven_five_minus_one_point_four_six_l229_229647


namespace rectangle_length_35_l229_229888

theorem rectangle_length_35
  (n_rectangles : ℕ) (area_abcd : ℝ) (rect_length_multiple : ℕ) (rect_width_multiple : ℕ) 
  (n_rectangles_eq : n_rectangles = 6)
  (area_abcd_eq : area_abcd = 4800)
  (rect_length_multiple_eq : rect_length_multiple = 3)
  (rect_width_multiple_eq : rect_width_multiple = 2) :
  ∃ y : ℝ, round y = 35 ∧ y^2 * (4/3) = area_abcd :=
by
  sorry


end rectangle_length_35_l229_229888


namespace calvin_weight_after_one_year_l229_229335

theorem calvin_weight_after_one_year :
  ∀ (initial_weight weight_loss_per_month months : ℕ),
  initial_weight = 250 →
  weight_loss_per_month = 8 →
  months = 12 →
  (initial_weight - (weight_loss_per_month * months) = 154) := by
  intros initial_weight weight_loss_per_month months 
  intro h1 h2 h3
  rw [h1, h2, h3]
  show (250 - (8 * 12) = 154)
  norm_num
  sorry

end calvin_weight_after_one_year_l229_229335


namespace problem_l229_229575

noncomputable def f (x : ℝ) : ℝ := 3 / (x + 1)

theorem problem (x : ℝ) (h1 : 3 ≤ x) (h2 : x ≤ 5) :
  (∀ x₁ x₂, 3 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 5 → f x₁ > f x₁) ∧
  f 3 = 3/4 ∧
  f 5 = 1/2 :=
by sorry

end problem_l229_229575


namespace rectangular_prism_edges_vertices_faces_sum_l229_229775

theorem rectangular_prism_edges_vertices_faces_sum (a b c : ℕ) (h1: a = 2) (h2: b = 3) (h3: c = 4) : 
  12 + 8 + 6 = 26 :=
by
  sorry

end rectangular_prism_edges_vertices_faces_sum_l229_229775


namespace initial_jelly_beans_l229_229598

theorem initial_jelly_beans (total_children : ℕ) (percentage : ℕ) (jelly_per_child : ℕ) (remaining_jelly : ℕ) :
  (percentage = 80) → (total_children = 40) → (jelly_per_child = 2) → (remaining_jelly = 36) →
  (total_children * percentage / 100 * jelly_per_child + remaining_jelly = 100) :=
by
  intros h1 h2 h3 h4
  sorry

end initial_jelly_beans_l229_229598


namespace eggs_needed_for_recipe_l229_229781

noncomputable section

theorem eggs_needed_for_recipe 
  (total_eggs : ℕ) 
  (rotten_eggs : ℕ) 
  (prob_all_rotten : ℝ)
  (h_total : total_eggs = 36)
  (h_rotten : rotten_eggs = 3)
  (h_prob : prob_all_rotten = 0.0047619047619047615) 
  : (2 : ℕ) = 2 :=
by
  sorry

end eggs_needed_for_recipe_l229_229781


namespace triangle_area_is_9_point_5_l229_229628

def Point : Type := (ℝ × ℝ)

def A : Point := (0, 1)
def B : Point := (4, 0)
def C : Point := (3, 5)

noncomputable def areaOfTriangle (A B C : Point) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem triangle_area_is_9_point_5 :
  areaOfTriangle A B C = 9.5 :=
by
  sorry

end triangle_area_is_9_point_5_l229_229628


namespace probability_two_green_balls_l229_229621

theorem probability_two_green_balls:
  let total_balls := 3 + 5 + 4 in
  let total_ways := Nat.choose total_balls 3 in
  let ways_to_choose_green := Nat.choose 4 2 in
  let ways_to_choose_non_green := Nat.choose 8 1 in
  let favorable_ways := ways_to_choose_green * ways_to_choose_non_green in
  let probability := favorable_ways / total_ways in
  probability = 12 / 55 :=
by
  sorry

end probability_two_green_balls_l229_229621


namespace rhombus_longer_diagonal_length_l229_229641

theorem rhombus_longer_diagonal_length (side_length : ℕ) (shorter_diagonal : ℕ) 
  (h_side : side_length = 65) (h_short_diag : shorter_diagonal = 56) : 
  ∃ longer_diagonal : ℕ, longer_diagonal = 118 :=
by
  sorry

end rhombus_longer_diagonal_length_l229_229641


namespace prime_pairs_divisibility_l229_229973

theorem prime_pairs_divisibility:
  ∀ (p q : ℕ), (Nat.Prime p ∧ Nat.Prime q ∧ p ≤ q ∧ p * q ∣ ((5 ^ p - 2 ^ p) * (7 ^ q - 2 ^ q))) ↔ 
                (p = 3 ∧ q = 5) ∨ 
                (p = 3 ∧ q = 3) ∨ 
                (p = 5 ∧ q = 37) ∨ 
                (p = 5 ∧ q = 83) := by
  sorry

end prime_pairs_divisibility_l229_229973


namespace initial_days_planned_l229_229161

-- We define the variables and conditions given in the problem.
variables (men_original men_absent men_remaining days_remaining days_initial : ℕ)
variable (work_equivalence : men_original * days_initial = men_remaining * days_remaining)

-- Conditions from the problem
axiom men_original_cond : men_original = 48
axiom men_absent_cond : men_absent = 8
axiom men_remaining_cond : men_remaining = men_original - men_absent
axiom days_remaining_cond : days_remaining = 18

-- Theorem to be proved
theorem initial_days_planned : days_initial = 15 :=
by
  -- Insert proof steps here
  sorry

end initial_days_planned_l229_229161


namespace eval_expression_l229_229187

theorem eval_expression :
  -((18 / 3 * 8) - 80 + (4 ^ 2 * 2)) = 0 :=
by
  sorry

end eval_expression_l229_229187


namespace marble_distribution_l229_229202

theorem marble_distribution (x : ℝ) (h : 49 = (3 * x + 2) + (x + 1) + (2 * x - 1) + x) :
  (3 * x + 2 = 22) ∧ (x + 1 = 8) ∧ (2 * x - 1 = 12) ∧ (x = 7) :=
by
  sorry

end marble_distribution_l229_229202


namespace probability_of_5_odd_numbers_l229_229014

-- Define a function to represent the binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

-- Axiom that defines the probability of getting an odd number
axiom fair_die_prob : ∀ (x : ℕ), 0 < x ∧ x ≤ 6 -> (1/2)

-- Define the problem statement about the probability
theorem probability_of_5_odd_numbers (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 5) : 
  (binom n k) / 2^n = 3 / 32 := sorry

end probability_of_5_odd_numbers_l229_229014


namespace triangle_square_ratio_l229_229317

theorem triangle_square_ratio :
  ∀ (x y : ℝ), (x = 60 / 17) → (y = 780 / 169) → (x / y = 78 / 102) :=
by
  intros x y hx hy
  rw [hx, hy]
  -- the proof is skipped, as instructed
  sorry

end triangle_square_ratio_l229_229317


namespace simplify_expression1_simplify_expression2_l229_229262

-- Problem 1 statement
theorem simplify_expression1 (a b : ℤ) : 2 * (2 * b - 3 * a) + 3 * (2 * a - 3 * b) = -5 * b :=
  by
  sorry

-- Problem 2 statement
theorem simplify_expression2 (a b : ℤ) : 4 * a^2 + 2 * (3 * a * b - 2 * a^2) - (7 * a * b - 1) = -a * b + 1 :=
  by
  sorry

end simplify_expression1_simplify_expression2_l229_229262


namespace solve_problem_l229_229798

def spadesuit (a b : ℤ) : ℤ := abs (a - b)

theorem solve_problem : spadesuit 3 (spadesuit 5 (spadesuit 8 11)) = 1 :=
by
  -- Proof is omitted
  sorry

end solve_problem_l229_229798


namespace factor_expression_l229_229662

theorem factor_expression (x y : ℝ) : x * y^2 - 4 * x = x * (y + 2) * (y - 2) := 
by
  sorry

end factor_expression_l229_229662


namespace smallest_four_digit_in_pascal_l229_229429

-- Pascal's triangle definition
def pascal (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

-- Condition definition
def exists_in_pascal (m : ℕ) : Prop :=
  ∃ n k, pascal n k = m

theorem smallest_four_digit_in_pascal :
  ∃ n k, pascal n k = 1000 ∧ ∀ m, (exists_in_pascal m) → (m >= 1000) → (m = 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_l229_229429


namespace no_prime_divisible_by_77_l229_229688

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_77 :
  ∀ p : ℕ, is_prime p → ¬ (77 ∣ p) :=
by
  intros p h_prime h_div
  sorry

end no_prime_divisible_by_77_l229_229688


namespace quadratic_root_unique_l229_229751

theorem quadratic_root_unique 
  (a b c : ℝ)
  (hf1 : b^2 - 4 * a * c = 0)
  (hf2 : (b - 30 * a)^2 - 4 * a * (17 * a - 7 * b + c) = 0)
  (ha_pos : a ≠ 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x = -11 := 
by
  sorry

end quadratic_root_unique_l229_229751


namespace ploughing_solution_l229_229295

/-- Definition representing the problem of A and B ploughing the field together and alone --/
noncomputable def ploughing_problem : Prop :=
  ∃ (A : ℝ), (A > 0) ∧ (1 / A + 1 / 30 = 1 / 10) ∧ A = 15

theorem ploughing_solution : ploughing_problem :=
  by sorry

end ploughing_solution_l229_229295


namespace proportional_increase_l229_229288

theorem proportional_increase (x y : ℝ) (h : 3 * x - 2 * y = 7) : y = (3 / 2) * x - 7 / 2 :=
by
  sorry

end proportional_increase_l229_229288


namespace tv_show_duration_l229_229383

theorem tv_show_duration (total_airing_time_in_hours : ℝ) (num_commercials : ℕ) (commercial_duration_in_minutes : ℝ) :
  num_commercials = 3 → commercial_duration_in_minutes = 10 → total_airing_time_in_hours = 1.5 →
  (total_airing_time_in_hours * 60 - num_commercials * commercial_duration_in_minutes) / 60 = 1 := 
by {
  intros,
  simp,
  sorry
}

end tv_show_duration_l229_229383


namespace Alfonso_daily_earnings_l229_229943

-- Define the conditions given in the problem
def helmet_cost : ℕ := 340
def current_savings : ℕ := 40
def days_per_week : ℕ := 5
def weeks_to_work : ℕ := 10

-- Define the question as a property to prove
def daily_earnings : ℕ := 6

-- Prove that the daily earnings are $6 given the conditions
theorem Alfonso_daily_earnings :
  (helmet_cost - current_savings) / (days_per_week * weeks_to_work) = daily_earnings :=
by
  sorry

end Alfonso_daily_earnings_l229_229943


namespace sum_infinite_geometric_series_l229_229969

theorem sum_infinite_geometric_series (a r : ℚ) (h : a = 1) (h2 : r = 1/4) : 
  (∀ S, S = a / (1 - r) → S = 4 / 3) :=
by
  intros S hS
  rw [h, h2] at hS
  simp [hS]
  sorry

end sum_infinite_geometric_series_l229_229969


namespace total_length_of_scale_l229_229499

theorem total_length_of_scale 
  (n : ℕ) (len_per_part : ℕ) 
  (h_n : n = 5) 
  (h_len_per_part : len_per_part = 25) :
  n * len_per_part = 125 :=
by
  sorry

end total_length_of_scale_l229_229499


namespace first_three_workers_dig_time_l229_229817

variables 
  (a b c d : ℝ) -- work rates of the four workers
  (hours : ℝ) -- time to dig the trench

def work_together (a b c d hours : ℝ) := (a + b + c + d) * hours = 1

def scenario1 (a b c d : ℝ) := (2 * a + (1/2) * b + c + d) * 6 = 1

def scenario2 (a b c d : ℝ) := (a/2 + 2 * b + c + d) * 4 = 1

theorem first_three_workers_dig_time
  (h1 : work_together a b c d 6)
  (h2 : scenario1 a b c d)
  (h3 : scenario2 a b c d) :
  hours = 6 := 
sorry

end first_three_workers_dig_time_l229_229817


namespace area_of_triangle_is_sqrt_5_sum_of_tangents_eq_1_l229_229711

-- Definitions and conditions
variable {A B C a b c : ℝ}
variable (cosA : ℝ) (sinA : ℝ)
variable (area : ℝ)
variable (tanA tanB tanC : ℝ)

-- Given conditions
axiom angle_identity : b^2 + c^2 = 3 * b * c * cosA
axiom sin_cos_identity : sinA^2 + cosA^2 = 1
axiom law_of_cosines : a^2 = b^2 + c^2 - 2 * b * c * cosA

-- Part (1) statement
theorem area_of_triangle_is_sqrt_5 (B_eq_C : B = C) (a_eq_2 : a = 2) 
    (cosA_eq_2_3 : cosA = 2/3) 
    (b_eq_sqrt6 : b = Real.sqrt 6) 
    (sinA_eq_sqrt5_3 : sinA = Real.sqrt 5 / 3) 
    : area = Real.sqrt 5 := sorry

-- Part (2) statement
theorem sum_of_tangents_eq_1 (tanA_eq : tanA = sinA / cosA)
    (tanB_eq : tanB = sinA * sinA / (cosA * cosA))
    (tanC_eq : tanC = sinA * sinA / (cosA * cosA))
    : (tanA / tanB) + (tanA / tanC) = 1 := sorry

end area_of_triangle_is_sqrt_5_sum_of_tangents_eq_1_l229_229711


namespace find_z_l229_229481

-- Definitions from the problem statement
variables {x y z : ℤ}
axiom consecutive (h1: x = z + 2) (h2: y = z + 1) : true
axiom ordered (h3: x > y) (h4: y > z) : true
axiom equation (h5: 2 * x + 3 * y + 3 * z = 5 * y + 8) : true

-- The proof goal
theorem find_z (h1: x = z + 2) (h2: y = z + 1) (h3: x > y) (h4: y > z) (h5: 2 * x + 3 * y + 3 * z = 5 * y + 8) : z = 2 :=
by 
  sorry

end find_z_l229_229481


namespace rhombus_longer_diagonal_l229_229635

theorem rhombus_longer_diagonal (a b d : ℝ) (h : a = 65) (h_d : d = 56) :
  ∃ l, l = 118 :=
by
  sorry

end rhombus_longer_diagonal_l229_229635


namespace expression_never_equals_33_l229_229573

theorem expression_never_equals_33 (x y : ℤ) : 
  x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 := 
sorry

end expression_never_equals_33_l229_229573


namespace jake_fewer_peaches_than_steven_l229_229564

theorem jake_fewer_peaches_than_steven :
  ∀ (jill steven jake : ℕ),
    jill = 87 →
    steven = jill + 18 →
    jake = jill + 13 →
    steven - jake = 5 :=
by
  intros jill steven jake hjill hsteven hjake
  sorry

end jake_fewer_peaches_than_steven_l229_229564


namespace parts_of_second_liquid_l229_229386

theorem parts_of_second_liquid (x : ℝ) :
    (0.10 * 5 + 0.15 * x) / (5 + x) = 11.42857142857143 / 100 ↔ x = 2 :=
by
  sorry

end parts_of_second_liquid_l229_229386


namespace hall_volume_l229_229038

theorem hall_volume (length width : ℝ) (h : ℝ) 
  (h_length : length = 6) 
  (h_width : width = 6) 
  (h_areas : 2 * (length * width) = 4 * (length * h)) :
  length * width * h = 108 :=
by
  sorry

end hall_volume_l229_229038


namespace planes_parallel_l229_229725

-- Given definitions and conditions
variables {Line Plane : Type}
variables (a b : Line) (α β γ : Plane)

-- Conditions from the problem
axiom perp_line_plane (line : Line) (plane : Plane) : Prop
axiom parallel_line_plane (line : Line) (plane : Plane) : Prop
axiom parallel_plane_plane (plane1 plane2 : Plane) : Prop

-- Conditions
variable (h1 : parallel_plane_plane γ α)
variable (h2 : parallel_plane_plane γ β)

-- Proof statement
theorem planes_parallel (h1 : parallel_plane_plane γ α) (h2 : parallel_plane_plane γ β) : parallel_plane_plane α β := sorry

end planes_parallel_l229_229725


namespace probability_differ_by_three_is_one_sixth_l229_229952

def probability_of_differ_by_three (outcomes : ℕ) : ℚ :=
  let successful_outcomes := 6
  successful_outcomes / outcomes

theorem probability_differ_by_three_is_one_sixth :
  probability_of_differ_by_three (6 * 6) = 1 / 6 :=
by sorry

end probability_differ_by_three_is_one_sixth_l229_229952


namespace total_distance_biked_two_days_l229_229171

def distance_yesterday : ℕ := 12
def distance_today : ℕ := (2 * distance_yesterday) - 3
def total_distance_biked : ℕ := distance_yesterday + distance_today

theorem total_distance_biked_two_days : total_distance_biked = 33 :=
by {
  -- Given distance_yesterday = 12
  -- distance_today calculated as (2 * distance_yesterday) - 3 = 21
  -- total_distance_biked = distance_yesterday + distance_today = 33
  sorry
}

end total_distance_biked_two_days_l229_229171


namespace at_least_one_ge_two_l229_229996

theorem at_least_one_ge_two (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  a + 1 / b ≥ 2 ∨ b + 1 / c ≥ 2 ∨ c + 1 / a ≥ 2 := 
sorry

end at_least_one_ge_two_l229_229996


namespace tangent_line_eq_l229_229749

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 3

def M : ℝ×ℝ := (2, -3)

theorem tangent_line_eq (x y : ℝ) (h : y = f x) (h' : (x, y) = M) :
  2 * x - y - 7 = 0 :=
sorry

end tangent_line_eq_l229_229749


namespace max_product_of_functions_l229_229265

theorem max_product_of_functions (f h : ℝ → ℝ) (hf : ∀ x, -5 ≤ f x ∧ f x ≤ 3) (hh : ∀ x, -3 ≤ h x ∧ h x ≤ 4) :
  ∃ x, f x * h x = 20 :=
by {
  sorry
}

end max_product_of_functions_l229_229265


namespace monotonic_increasing_interval_f_l229_229404

noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 2 * x + 8)

theorem monotonic_increasing_interval_f :
  ∃ I : Set ℝ, (I = Set.Icc (-2) 1) ∧ (∀x1 ∈ I, ∀x2 ∈ I, x1 ≤ x2 → f x1 ≤ f x2) :=
sorry

end monotonic_increasing_interval_f_l229_229404


namespace root_exists_in_interval_l229_229238

noncomputable def f (x : ℝ) := (1 / 2) ^ x - x + 1

theorem root_exists_in_interval :
  (0 < f 1) ∧ (f 1.5 < 0) ∧ (f 2 < 0) ∧ (f 3 < 0) → ∃ x, 1 < x ∧ x < 1.5 ∧ f x = 0 :=
by
  -- use the intermediate value theorem and bisection method here
  sorry

end root_exists_in_interval_l229_229238


namespace problem_statement_l229_229115

noncomputable def expr (x y z : ℝ) : ℝ :=
  (x^2 * y^2) / ((x^2 - y*z) * (y^2 - x*z)) +
  (x^2 * z^2) / ((x^2 - y*z) * (z^2 - x*y)) +
  (y^2 * z^2) / ((y^2 - x*z) * (z^2 - x*y))

theorem problem_statement (x y z : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : z ≠ 0) (h₄ : x + y + z = -1) :
  expr x y z = 1 := by
  sorry

end problem_statement_l229_229115


namespace all_odd_digits_n_squared_l229_229194

/-- Helper function to check if all digits in a number are odd -/
def all_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d % 2 = 1

/-- Main theorem stating that the only positive integers n such that all the digits of n^2 are odd are 1 and 3 -/
theorem all_odd_digits_n_squared (n : ℕ) :
  (n > 0) → (all_odd_digits (n^2)) → (n = 1 ∨ n = 3) :=
by
  sorry

end all_odd_digits_n_squared_l229_229194


namespace product_uvw_l229_229901

theorem product_uvw (a x y c : ℝ) (u v w : ℤ) :
  (a^u * x - a^v) * (a^w * y - a^3) = a^5 * c^5 → 
  a^8 * x * y - a^7 * y - a^6 * x = a^5 * (c^5 - 1) → 
  u * v * w = 6 :=
by
  intros h1 h2
  -- Proof will go here
  sorry

end product_uvw_l229_229901


namespace ypsilon_calendar_l229_229255

theorem ypsilon_calendar (x y z : ℕ) 
  (h1 : 28 * x + 30 * y + 31 * z = 365) : x + y + z = 12 :=
sorry

end ypsilon_calendar_l229_229255


namespace dice_diff_by_three_probability_l229_229962

theorem dice_diff_by_three_probability : 
  let outcomes := [(1, 4), (2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let successful_outcomes := 6 in
  let total_outcomes := 6 * 6 in
  let probability := successful_outcomes / total_outcomes in
  probability = 1 / 6 :=
by
  sorry

end dice_diff_by_three_probability_l229_229962


namespace arc_length_ln_sin_eval_l229_229300

noncomputable def arc_length_ln_sin : ℝ :=
  ∫ x in (Real.pi / 3)..(Real.pi / 2), Real.sqrt (1 + (Real.cot x)^2)

theorem arc_length_ln_sin_eval :
  arc_length_ln_sin = (1 / 2) * Real.log 3 :=
by
  sorry

end arc_length_ln_sin_eval_l229_229300


namespace number_of_different_ways_to_travel_l229_229510

-- Define the conditions
def number_of_morning_flights : ℕ := 2
def number_of_afternoon_flights : ℕ := 3

-- Assert the question and the answer
theorem number_of_different_ways_to_travel : 
  (number_of_morning_flights * number_of_afternoon_flights) = 6 :=
by
  sorry

end number_of_different_ways_to_travel_l229_229510


namespace find_number_l229_229701

theorem find_number (x : ℤ) (h : x - (28 - (37 - (15 - 16))) = 55) : x = 65 :=
sorry

end find_number_l229_229701


namespace inequality_D_holds_l229_229997

theorem inequality_D_holds (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 := 
sorry

end inequality_D_holds_l229_229997


namespace smallest_four_digit_in_pascal_l229_229430

-- Pascal's triangle definition
def pascal (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

-- Condition definition
def exists_in_pascal (m : ℕ) : Prop :=
  ∃ n k, pascal n k = m

theorem smallest_four_digit_in_pascal :
  ∃ n k, pascal n k = 1000 ∧ ∀ m, (exists_in_pascal m) → (m >= 1000) → (m = 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_l229_229430


namespace smallest_four_digit_in_pascal_l229_229418

-- define what it means for a number to be in Pascal's triangle
def in_pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- define the concept of a smallest four-digit number
def smallest_four_digit : ℕ := 1000

-- prove that the smallest four-digit number in Pascal's Triangle is 1000
theorem smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), in_pascal n k = smallest_four_digit := 
sorry

end smallest_four_digit_in_pascal_l229_229418


namespace smallest_possible_difference_l229_229110

theorem smallest_possible_difference :
  ∃ (x y z : ℕ), 
    x + y + z = 1801 ∧ x < y ∧ y ≤ z ∧ x + y > z ∧ y + z > x ∧ z + x > y ∧ (y - x = 1) := 
by
  sorry

end smallest_possible_difference_l229_229110


namespace product_of_values_l229_229808

-- Define the condition
def satisfies_eq (x : ℝ) : Prop := |2 * x| + 4 = 38

-- State the theorem
theorem product_of_values : ∃ x1 x2 : ℝ, satisfies_eq x1 ∧ satisfies_eq x2 ∧ x1 * x2 = -289 := 
by
  sorry

end product_of_values_l229_229808


namespace Jordan_length_is_8_l229_229793

-- Definitions of the conditions given in the problem
def Carol_length := 5
def Carol_width := 24
def Jordan_width := 15

-- Definition to calculate the area of Carol's rectangle
def Carol_area : ℕ := Carol_length * Carol_width

-- Definition to calculate the length of Jordan's rectangle
def Jordan_length (area : ℕ) (width : ℕ) : ℕ := area / width

-- Proposition to prove the length of Jordan's rectangle
theorem Jordan_length_is_8 : Jordan_length Carol_area Jordan_width = 8 :=
by
  -- skipping the proof
  sorry

end Jordan_length_is_8_l229_229793


namespace probability_of_difference_three_l229_229949

def is_valid_pair (a b : ℕ) : Prop :=
  (a = 3 ∧ b = 6) ∨ (a = 4 ∧ b = 1) ∨ (a = 5 ∧ b = 2) ∨ (a = 6 ∧ b = 3)

def number_of_successful_outcomes : ℕ := 4

def total_number_of_outcomes : ℕ := 36

def probability_of_valid_pairs : ℚ := number_of_successful_outcomes / total_number_of_outcomes

theorem probability_of_difference_three : probability_of_valid_pairs = 1 / 9 := by
  sorry

end probability_of_difference_three_l229_229949


namespace cos_neg_1500_eq_half_l229_229978

theorem cos_neg_1500_eq_half : Real.cos (-1500 * Real.pi / 180) = 1/2 := by
  sorry

end cos_neg_1500_eq_half_l229_229978


namespace payment_to_N_l229_229141

variable (x : ℝ)

/-- Conditions stating the total payment and the relationship between M and N's payment --/
axiom total_payment : x + 1.20 * x = 550

/-- Statement to prove the amount paid to N per week --/
theorem payment_to_N : x = 250 :=
by
  sorry

end payment_to_N_l229_229141


namespace pure_alcohol_addition_l229_229223

variables (P : ℝ) (V : ℝ := 14.285714285714286 ) (initial_volume : ℝ := 100) (final_percent_alcohol : ℝ := 0.30)

theorem pure_alcohol_addition :
  P / 100 * initial_volume + V = final_percent_alcohol * (initial_volume + V) :=
by
  sorry

end pure_alcohol_addition_l229_229223


namespace probability_neither_red_nor_purple_correct_l229_229305

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def green_balls : ℕ := 10
def yellow_balls : ℕ := 7
def red_balls : ℕ := 15
def purple_balls : ℕ := 6

def neither_red_nor_purple_balls : ℕ := total_balls - (red_balls + purple_balls)
def probability_neither_red_nor_purple : ℚ := (neither_red_nor_purple_balls : ℚ) / (total_balls : ℚ)

theorem probability_neither_red_nor_purple_correct : 
  probability_neither_red_nor_purple = 13 / 20 := 
by sorry

end probability_neither_red_nor_purple_correct_l229_229305


namespace range_of_a_l229_229837

noncomputable def f (a x : ℝ) : ℝ := (a * x + 1) / (x + 2)

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, -2 < x → -2 < y → x < y → f a x < f a y) → (a > 1/2) :=
by
  sorry

end range_of_a_l229_229837


namespace trigonometric_identity_l229_229673

theorem trigonometric_identity (A B C : ℝ) (h : A + B + C = Real.pi) :
  (Real.cos (A / 2)) ^ 2 = (Real.cos (B / 2)) ^ 2 + (Real.cos (C / 2)) ^ 2 - 2 * (Real.cos (B / 2)) * (Real.cos (C / 2)) * (Real.sin (A / 2)) :=
sorry

end trigonometric_identity_l229_229673


namespace tetrahedron_volume_from_pentagon_l229_229667

noncomputable def volume_of_tetrahedron (side_length : ℝ) (diagonal_length : ℝ) (base_area : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

theorem tetrahedron_volume_from_pentagon :
  ∀ (s : ℝ), s = 1 →
  volume_of_tetrahedron s ((1 + Real.sqrt 5) / 2) ((Real.sqrt 3) / 4) (Real.sqrt ((5 + 2 * Real.sqrt 5) / 4)) =
  (1 + Real.sqrt 5) / 24 :=
by
  intros s hs
  rw [hs]
  sorry

end tetrahedron_volume_from_pentagon_l229_229667


namespace find_k_l229_229278

-- Definitions for the conditions and the main theorem.
variables {x y k : ℝ}

-- The first equation of the system
def eq1 (x y k : ℝ) : Prop := 2 * x + 5 * y = k

-- The second equation of the system
def eq2 (x y : ℝ) : Prop := x - 4 * y = 15

-- Condition that x and y are opposites
def are_opposites (x y : ℝ) : Prop := x + y = 0

-- The theorem to prove
theorem find_k (hk : ∃ (x y : ℝ), eq1 x y k ∧ eq2 x y ∧ are_opposites x y) : k = -9 :=
sorry

end find_k_l229_229278


namespace smallest_four_digit_in_pascals_triangle_l229_229465

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, (1000 ≤ (nat.choose n k)) ∧ (∀ m l : ℕ, 1000 ≤ nat.choose m l → nat.choose m l ≥ 1000) :=
begin
  use [1000, 1],
  split,
  { exact nat.choose_eq_one_right 1000 1 },
  { intros m l hml,
    linarith }
end

end smallest_four_digit_in_pascals_triangle_l229_229465


namespace probability_diff_by_3_l229_229945

def roll_probability_diff_three (x y : ℕ) : ℚ :=
  if abs (x - y) = 3 then 1 else 0

theorem probability_diff_by_3 :
  let total_outcomes := 36 in
  let successful_outcomes := (finset.univ.product finset.univ).filter (λ (p : ℕ × ℕ), roll_probability_diff_three p.1 p.2 = 1) in
  (successful_outcomes.card : ℚ) / total_outcomes = 5 / 36 :=
by
  sorry

end probability_diff_by_3_l229_229945


namespace frank_pie_consumption_l229_229061

theorem frank_pie_consumption :
  let Erik := 0.6666666666666666
  let MoreThanFrank := 0.3333333333333333
  let Frank := Erik - MoreThanFrank
  Frank = 0.3333333333333333 := by
sorry

end frank_pie_consumption_l229_229061


namespace probability_of_difference_three_l229_229950

def is_valid_pair (a b : ℕ) : Prop :=
  (a = 3 ∧ b = 6) ∨ (a = 4 ∧ b = 1) ∨ (a = 5 ∧ b = 2) ∨ (a = 6 ∧ b = 3)

def number_of_successful_outcomes : ℕ := 4

def total_number_of_outcomes : ℕ := 36

def probability_of_valid_pairs : ℚ := number_of_successful_outcomes / total_number_of_outcomes

theorem probability_of_difference_three : probability_of_valid_pairs = 1 / 9 := by
  sorry

end probability_of_difference_three_l229_229950


namespace taxi_fare_function_l229_229863

theorem taxi_fare_function (x : ℝ) (h : x > 3) : 
  ∃ y : ℝ, y = 2 * x + 4 :=
by
  sorry

end taxi_fare_function_l229_229863


namespace largest_four_digit_negative_integer_congruent_to_2_mod_17_l229_229290

theorem largest_four_digit_negative_integer_congruent_to_2_mod_17 :
  ∃ (n : ℤ), (n % 17 = 2 ∧ n > -10000 ∧ n < -999) ∧ ∀ m : ℤ, (m % 17 = 2 ∧ m > -10000 ∧ m < -999) → m ≤ n :=
sorry

end largest_four_digit_negative_integer_congruent_to_2_mod_17_l229_229290


namespace runs_by_running_percentage_l229_229765

def total_runs := 125
def boundaries := 5
def boundary_runs := boundaries * 4
def sixes := 5
def sixes_runs := sixes * 6
def runs_by_running := total_runs - (boundary_runs + sixes_runs)
def percentage_runs_by_running := (runs_by_running : ℚ) / total_runs * 100

theorem runs_by_running_percentage :
  percentage_runs_by_running = 60 := by sorry

end runs_by_running_percentage_l229_229765


namespace female_athletes_drawn_l229_229056

theorem female_athletes_drawn (total_athletes male_athletes female_athletes sample_size : ℕ)
  (h_total : total_athletes = male_athletes + female_athletes)
  (h_team : male_athletes = 48 ∧ female_athletes = 36)
  (h_sample_size : sample_size = 35) :
  (female_athletes * sample_size) / total_athletes = 15 :=
by
  sorry

end female_athletes_drawn_l229_229056


namespace correct_value_of_A_sub_B_l229_229165

variable {x y : ℝ}

-- Given two polynomials A and B where B = 3x - 2y, and a mistaken equation A + B = x - y,
-- we want to prove the correct value of A - B.
theorem correct_value_of_A_sub_B (A B : ℝ) (h1 : B = 3 * x - 2 * y) (h2 : A + B = x - y) :
  A - B = -5 * x + 3 * y :=
by
  sorry

end correct_value_of_A_sub_B_l229_229165


namespace ella_incorrect_answers_l229_229381

theorem ella_incorrect_answers
  (marion_score : ℕ)
  (ella_score : ℕ)
  (total_items : ℕ)
  (h1 : marion_score = 24)
  (h2 : marion_score = (ella_score / 2) + 6)
  (h3 : total_items = 40) : 
  total_items - ella_score = 4 :=
by
  sorry

end ella_incorrect_answers_l229_229381


namespace beth_sells_half_of_coins_l229_229507

theorem beth_sells_half_of_coins (x y : ℕ) (h₁ : x = 125) (h₂ : y = 35) : (x + y) / 2 = 80 :=
by
  sorry

end beth_sells_half_of_coins_l229_229507


namespace tan_pi_over_4_plus_alpha_eq_two_l229_229534

theorem tan_pi_over_4_plus_alpha_eq_two
  (α : ℂ) 
  (h : Complex.tan ((π / 4) + α) = 2) : 
  (1 / (2 * Complex.sin α * Complex.cos α + (Complex.cos α)^2)) = (2 / 3) :=
by
  sorry

end tan_pi_over_4_plus_alpha_eq_two_l229_229534


namespace range_of_b_l229_229702

noncomputable def f (x b : ℝ) : ℝ := -1/2 * (x - 2)^2 + b * Real.log (x + 2)
noncomputable def derivative (x b : ℝ) := -(x - 2) + b / (x + 2)

-- Lean theorem statement
theorem range_of_b (b : ℝ) :
  (∀ x > 1, derivative x b ≤ 0) → b ≤ -3 :=
by
  sorry

end range_of_b_l229_229702


namespace cost_per_pie_eq_l229_229254

-- We define the conditions
def price_per_piece : ℝ := 4
def pieces_per_pie : ℕ := 3
def pies_per_hour : ℕ := 12
def actual_revenue : ℝ := 138

-- Lean theorem statement
theorem cost_per_pie_eq : (price_per_piece * pieces_per_pie * pies_per_hour - actual_revenue) / pies_per_hour = 0.50 := by
  -- Proof would go here
  sorry

end cost_per_pie_eq_l229_229254


namespace smallest_four_digit_number_in_pascals_triangle_l229_229437

theorem smallest_four_digit_number_in_pascals_triangle : ∃ n k, binom n k = 1000 ∧ (∀ n' k', binom n' k' > 1000 → nat.zero < n' k' ∧ n' k' < (binom n k)) := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l229_229437


namespace gcd_lcm_product_l229_229345

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 75) (h2 : b = 90) : Nat.gcd a b * Nat.lcm a b = 6750 :=
by
  sorry

end gcd_lcm_product_l229_229345


namespace promotional_event_probabilities_l229_229707

def P_A := 1 / 1000
def P_B := 1 / 100
def P_C := 1 / 20
def P_A_B_C := P_A + P_B + P_C
def P_A_B := P_A + P_B
def P_complement_A_B := 1 - P_A_B

theorem promotional_event_probabilities :
  P_A = 1 / 1000 ∧
  P_B = 1 / 100 ∧
  P_C = 1 / 20 ∧
  P_A_B_C = 61 / 1000 ∧
  P_complement_A_B = 989 / 1000 :=
by
  sorry

end promotional_event_probabilities_l229_229707


namespace child_tickets_sold_l229_229049

noncomputable def price_adult_ticket : ℝ := 7
noncomputable def price_child_ticket : ℝ := 4
noncomputable def total_tickets_sold : ℝ := 900
noncomputable def total_revenue : ℝ := 5100

theorem child_tickets_sold : ∃ (C : ℝ), price_child_ticket * C + price_adult_ticket * (total_tickets_sold - C) = total_revenue ∧ C = 400 :=
by
  sorry

end child_tickets_sold_l229_229049


namespace subtract_two_decimals_l229_229650

theorem subtract_two_decimals : 3.75 - 1.46 = 2.29 := by
  sorry

end subtract_two_decimals_l229_229650


namespace solve_quadratic_equation_l229_229408

theorem solve_quadratic_equation (x : ℝ) : x^2 = 100 → x = -10 ∨ x = 10 :=
by
  intro h
  sorry

end solve_quadratic_equation_l229_229408


namespace product_mod_23_l229_229920

theorem product_mod_23 :
  (2003 * 2004 * 2005 * 2006 * 2007 * 2008) % 23 = 3 :=
by 
  sorry

end product_mod_23_l229_229920


namespace average_salary_without_manager_l229_229745

theorem average_salary_without_manager (A : ℝ) (H : 15 * A + 4200 = 16 * (A + 150)) : A = 1800 :=
by {
  sorry
}

end average_salary_without_manager_l229_229745


namespace simplest_square_root_among_choices_l229_229607

variable {x : ℝ}

def is_simplest_square_root (n : ℝ) : Prop :=
  ∀ m, (m^2 = n) → (m = n)

theorem simplest_square_root_among_choices :
  is_simplest_square_root 7 ∧ ∀ n, n = 24 ∨ n = 1/3 ∨ n = 0.2 → ¬ is_simplest_square_root n :=
by
  sorry

end simplest_square_root_among_choices_l229_229607


namespace blue_tile_fraction_l229_229785

theorem blue_tile_fraction :
  let num_tiles := 8 * 8
  let corner_blue_tiles := 4 * 4 - 2 * 2
  let total_blue_tiles := 4 * corner_blue_tiles
  total_blue_tiles / num_tiles = (3 : ℚ) / 4 := 
by 
  let num_tiles := 8 * 8
  let corner_blue_tiles := 4 * 4 - 2 * 2
  let total_blue_tiles := 4 * corner_blue_tiles
  have frac_eq : total_blue_tiles / num_tiles = 48 / 64 := by sorry
  rw frac_eq,
  norm_num
  sorry

end blue_tile_fraction_l229_229785


namespace find_other_number_l229_229128

theorem find_other_number (a b : ℕ) (hcf_ab : Nat.gcd a b = 14) (lcm_ab : Nat.lcm a b = 396) (h : a = 36) : b = 154 :=
by
  sorry

end find_other_number_l229_229128


namespace find_weight_of_silver_in_metal_bar_l229_229485

noncomputable def weight_loss_ratio_tin : ℝ := 1.375 / 10
noncomputable def weight_loss_ratio_silver : ℝ := 0.375
noncomputable def ratio_tin_silver : ℝ := 0.6666666666666664

theorem find_weight_of_silver_in_metal_bar (T S : ℝ)
  (h1 : T + S = 70)
  (h2 : T / S = ratio_tin_silver)
  (h3 : weight_loss_ratio_tin * T + weight_loss_ratio_silver * S = 7) :
  S = 15 :=
by
  sorry

end find_weight_of_silver_in_metal_bar_l229_229485


namespace calculate_fraction_l229_229792

theorem calculate_fraction: (1 / (2 + 1 / (3 + 1 / 4))) = 13 / 30 := by
  sorry

end calculate_fraction_l229_229792


namespace total_profit_is_35000_l229_229294

-- Definitions based on the conditions
variables (IB TB : ℝ) -- IB: Investment of B, TB: Time period of B's investment
def IB_times_TB := IB * TB
def IA := 3 * IB
def TA := 2 * TB
def profit_share_B := IB_times_TB
def profit_share_A := 6 * IB_times_TB
variable (profit_B : ℝ)
def profit_B_val := 5000

-- Ensure these definitions are used
def total_profit := profit_share_A + profit_share_B

-- Lean 4 statement showing that the total profit is Rs 35000
theorem total_profit_is_35000 : total_profit = 35000 := by
  sorry

end total_profit_is_35000_l229_229294


namespace decreasing_sequence_b_l229_229827

def seq_a (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, 2 * a n * a (n + 1) = (a n)^2 + 1

def b_n (a b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n = (a n - 1) / (a n + 1)

theorem decreasing_sequence_b {a b : ℕ → ℝ} (h1 : seq_a a) (h2 : b_n a b) :
  ∀ n : ℕ, b (n + 1) < b n :=
by
  sorry

end decreasing_sequence_b_l229_229827


namespace least_number_with_remainder_4_l229_229615

theorem least_number_with_remainder_4 : ∃ n : ℕ, n = 184 ∧ 
  (∀ d ∈ [5, 9, 12, 18], (n - 4) % d = 0) ∧
  (∀ m : ℕ, (∀ d ∈ [5, 9, 12, 18], (m - 4) % d = 0) → m ≥ n) :=
by
  sorry

end least_number_with_remainder_4_l229_229615


namespace sum_of_digits_third_smallest_multiple_l229_229249

noncomputable def LCM_upto_7 : ℕ := Nat.lcm (Nat.lcm 1 2) (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7))))

noncomputable def third_smallest_multiple : ℕ := 3 * LCM_upto_7

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_third_smallest_multiple : sum_of_digits third_smallest_multiple = 9 := 
sorry

end sum_of_digits_third_smallest_multiple_l229_229249


namespace area_calculation_l229_229490

variable (x : ℝ)

def area_large_rectangle : ℝ := (2 * x + 9) * (x + 6)
def area_rectangular_hole : ℝ := (x - 1) * (2 * x - 5)
def area_square : ℝ := (x + 3) ^ 2
def area_remaining : ℝ := area_large_rectangle x - area_rectangular_hole x - area_square x

theorem area_calculation : area_remaining x = -x^2 + 22 * x + 40 := by
  sorry

end area_calculation_l229_229490


namespace CarlYardAreaIsCorrect_l229_229067

noncomputable def CarlRectangularYardArea (post_count : ℕ) (distance_between_posts : ℕ) (long_side_factor : ℕ) :=
  let x := post_count / (2 * (1 + long_side_factor))
  let short_side := (x - 1) * distance_between_posts
  let long_side := (long_side_factor * x - 1) * distance_between_posts
  short_side * long_side

theorem CarlYardAreaIsCorrect :
  CarlRectangularYardArea 24 5 3 = 825 := 
by
  -- calculation steps if needed or
  sorry

end CarlYardAreaIsCorrect_l229_229067


namespace salary_for_may_l229_229896

theorem salary_for_may
  (J F M A May : ℝ)
  (h1 : J + F + M + A = 32000)
  (h2 : F + M + A + May = 34400)
  (h3 : J = 4100) :
  May = 6500 := 
by 
  sorry

end salary_for_may_l229_229896


namespace consecutive_numbers_sum_digits_l229_229642

noncomputable def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem consecutive_numbers_sum_digits :
  ∃ n : ℕ, sum_of_digits n = 52 ∧ sum_of_digits (n + 4) = 20 := 
sorry

end consecutive_numbers_sum_digits_l229_229642


namespace price_of_item_a_l229_229762

theorem price_of_item_a : 
  let coins_1000 := 7
  let coins_100 := 4
  let coins_10 := 5
  let price_1000 := coins_1000 * 1000
  let price_100 := coins_100 * 100
  let price_10 := coins_10 * 10
  let total_price := price_1000 + price_100 + price_10
  total_price = 7450 := by
    sorry

end price_of_item_a_l229_229762


namespace uncovered_side_length_l229_229052

theorem uncovered_side_length {L W : ℕ} (h1 : L * W = 680) (h2 : 2 * W + L = 74) : L = 40 :=
sorry

end uncovered_side_length_l229_229052


namespace problem_l229_229842

theorem problem (x y : ℝ)
  (h1 : 1 / x + 1 / y = 4)
  (h2 : 1 / x - 1 / y = -6) :
  x + y = -4 / 5 :=
sorry

end problem_l229_229842


namespace remainder_of_5n_minus_9_l229_229476

theorem remainder_of_5n_minus_9 (n : ℤ) (h : n % 11 = 3) : (5 * n - 9) % 11 = 6 :=
by
  sorry -- Proof is omitted, as per instruction.

end remainder_of_5n_minus_9_l229_229476


namespace calculate_product_value_l229_229326

theorem calculate_product_value :
    (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 :=
by
  sorry

end calculate_product_value_l229_229326


namespace Sequential_structure_not_conditional_l229_229372

-- Definitions based on provided conditions
def is_conditional (s : String) : Prop :=
  s = "Loop structure" ∨ s = "If structure" ∨ s = "Until structure"

-- Theorem stating that Sequential structure is the one that doesn't contain a conditional judgment box
theorem Sequential_structure_not_conditional :
  ¬ is_conditional "Sequential structure" :=
by
  intro h
  cases h <;> contradiction

end Sequential_structure_not_conditional_l229_229372


namespace smallest_number_ending_in_9_divisible_by_13_l229_229028

theorem smallest_number_ending_in_9_divisible_by_13 :
  ∃ (n : ℕ), (n % 10 = 9) ∧ (13 ∣ n) ∧ (∀ (m : ℕ), (m % 10 = 9) ∧ (13 ∣ m) ∧ (m < n) -> false) :=
sorry

end smallest_number_ending_in_9_divisible_by_13_l229_229028


namespace number_of_yellow_parrots_l229_229120

-- Given conditions
def fraction_red : ℚ := 5 / 8
def total_parrots : ℕ := 120

-- Proof statement
theorem number_of_yellow_parrots : 
    (total_parrots : ℚ) * (1 - fraction_red) = 45 :=
by 
    sorry

end number_of_yellow_parrots_l229_229120


namespace trig_expression_value_l229_229999

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 2) : 
  (6 * Real.sin α + 8 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 5 := 
by
  sorry

end trig_expression_value_l229_229999


namespace triangle_perimeter_l229_229273

theorem triangle_perimeter (A r p : ℝ) (hA : A = 60) (hr : r = 2.5) (h_eq : A = r * p / 2) : p = 48 := 
by
  sorry

end triangle_perimeter_l229_229273


namespace eliana_steps_total_l229_229072

def eliana_walks_first_day_steps := 200 + 300
def eliana_walks_second_day_steps := 2 * eliana_walks_first_day_steps
def eliana_walks_third_day_steps := eliana_walks_second_day_steps + 100
def eliana_total_steps := eliana_walks_first_day_steps + eliana_walks_second_day_steps + eliana_walks_third_day_steps

theorem eliana_steps_total : eliana_total_steps = 2600 := by
  sorry

end eliana_steps_total_l229_229072


namespace total_money_l229_229479

theorem total_money (p q r : ℕ)
  (h1 : r = 2000)
  (h2 : r = (2 / 3) * (p + q)) : 
  p + q + r = 5000 :=
by
  sorry

end total_money_l229_229479


namespace sum_ages_l229_229005

theorem sum_ages (A_years B_years C_years : ℕ) (h1 : B_years = 30)
  (h2 : 10 * (B_years - 10) = (A_years - 10) * 2)
  (h3 : 10 * (B_years - 10) = (C_years - 10) * 3) :
  A_years + B_years + C_years = 90 :=
sorry

end sum_ages_l229_229005


namespace sin_transform_l229_229078

theorem sin_transform (θ : ℝ) (h : Real.sin (θ - π / 12) = 3 / 4) :
  Real.sin (2 * θ + π / 3) = -1 / 8 :=
by
  -- Proof would go here
  sorry

end sin_transform_l229_229078


namespace smallest_four_digit_in_pascals_triangle_l229_229444

theorem smallest_four_digit_in_pascals_triangle : ∃ n, ∃ k, k ≤ n ∧ 1000 ≤ Nat.choose n k :=
sorry

end smallest_four_digit_in_pascals_triangle_l229_229444


namespace inequality_ge_9_l229_229204

theorem inequality_ge_9 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) : 
  (2 / a + 1 / b) ≥ 9 :=
sorry

end inequality_ge_9_l229_229204


namespace determine_digit_square_l229_229974

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_palindrome (n : ℕ) : Prop :=
  let d1 := (n / 100000) % 10
  let d2 := (n / 10000) % 10
  let d3 := (n / 1000) % 10
  let d4 := (n / 100) % 10
  let d5 := (n / 10) % 10
  let d6 := n % 10
  d1 = d6 ∧ d2 = d5 ∧ d3 = d4

def is_multiple_of_6 (n : ℕ) : Prop := is_even (n % 10) ∧ is_divisible_by_3 (List.sum (Nat.digits 10 n))

theorem determine_digit_square :
  ∃ (square : ℕ),
  (is_palindrome (53700000 + square * 10 + 735) ∧ is_multiple_of_6 (53700000 + square * 10 + 735)) ∧ square = 6 := by
  sorry

end determine_digit_square_l229_229974


namespace trip_total_charge_l229_229877

noncomputable def initial_fee : ℝ := 2.25
noncomputable def additional_charge_per_increment : ℝ := 0.25
noncomputable def increment_length : ℝ := 2 / 5
noncomputable def trip_length : ℝ := 3.6

theorem trip_total_charge :
  initial_fee + (trip_length / increment_length) * additional_charge_per_increment = 4.50 :=
by
  sorry

end trip_total_charge_l229_229877


namespace range_of_m_l229_229215

theorem range_of_m {m : ℝ} :
  (3 * 3 - 2 * 1 + m) * (3 * (-4) - 2 * 6 + m) < 0 ↔ -7 < m ∧ m < 24 :=
by
  sorry

end range_of_m_l229_229215


namespace age_of_second_replaced_man_l229_229894

theorem age_of_second_replaced_man (avg_age_increase : ℕ) (avg_new_men_age : ℕ) (first_replaced_age : ℕ) (total_men : ℕ) (new_age_sum : ℕ) :
  avg_age_increase = 1 →
  avg_new_men_age = 34 →
  first_replaced_age = 21 →
  total_men = 12 →
  new_age_sum = 2 * avg_new_men_age →
  47 - (new_age_sum - (first_replaced_age + x)) = 12 →
  x = 35 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end age_of_second_replaced_man_l229_229894


namespace smallest_four_digit_number_in_pascals_triangle_l229_229449

theorem smallest_four_digit_number_in_pascals_triangle :
  ∃ (n : ℕ), 1000 = (n choose min n) ∧ (n > 0 ∧ (∀ m < n, (m choose min m) < 1000)) :=
begin
  sorry
end

end smallest_four_digit_number_in_pascals_triangle_l229_229449


namespace equation_of_line_bisecting_chord_l229_229487

theorem equation_of_line_bisecting_chord
  (P : ℝ × ℝ) 
  (A B : ℝ × ℝ)
  (P_bisects_AB : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (P_on_ellipse : 3 * P.1^2 + 4 * P.2^2 = 24)
  (A_on_ellipse : 3 * A.1^2 + 4 * A.2^2 = 24)
  (B_on_ellipse : 3 * B.1^2 + 4 * B.2^2 = 24) :
  ∃ (a b c : ℝ), a * P.2 + b * P.1 + c = 0 ∧ a = 2 ∧ b = -3 ∧ c = 7 :=
by 
  sorry

end equation_of_line_bisecting_chord_l229_229487


namespace find_a_b_value_l229_229229

-- Define the variables
variables {a b : ℤ}

-- Define the conditions for the monomials to be like terms
def exponents_match_x (a : ℤ) : Prop := a + 2 = 1
def exponents_match_y (b : ℤ) : Prop := b + 1 = 3

-- Main statement
theorem find_a_b_value (ha : exponents_match_x a) (hb : exponents_match_y b) : a + b = 1 :=
by
  sorry

end find_a_b_value_l229_229229


namespace inverse_proportion_relation_l229_229750

variable (k : ℝ) (y1 y2 : ℝ) (h1 : y1 = - (2 / (-1))) (h2 : y2 = - (2 / (-2)))

theorem inverse_proportion_relation : y1 > y2 := by
  sorry

end inverse_proportion_relation_l229_229750


namespace rectangle_difference_l229_229834

theorem rectangle_difference (A d x y : ℝ) (h1 : x * y = A) (h2 : x^2 + y^2 = d^2) :
  x - y = 2 * Real.sqrt A := 
sorry

end rectangle_difference_l229_229834


namespace evaluate_expression_l229_229340

theorem evaluate_expression : 
  - (16 / 2 * 8 - 72 + 4^2) = -8 :=
by 
  -- here, the proof would typically go
  sorry

end evaluate_expression_l229_229340


namespace probability_diff_by_three_l229_229955

theorem probability_diff_by_three : 
  let outcomes := (Finset.product (Finset.range 1 7) (Finset.range 1 7)) in
  let successful_outcomes := Finset.filter (λ (x : ℕ × ℕ), abs (x.1 - x.2) = 3) outcomes in
  (successful_outcomes.card : ℚ) / outcomes.card = 1 / 6 :=
by
  sorry

end probability_diff_by_three_l229_229955


namespace average_age_is_26_l229_229338

noncomputable def devin_age : ℕ := 12
noncomputable def eden_age : ℕ := 2 * devin_age
noncomputable def eden_mom_age : ℕ := 2 * eden_age
noncomputable def eden_grandfather_age : ℕ := (devin_age + eden_age + eden_mom_age) / 2
noncomputable def eden_aunt_age : ℕ := eden_mom_age / devin_age

theorem average_age_is_26 : 
  (devin_age + eden_age + eden_mom_age + eden_grandfather_age + eden_aunt_age) / 5 = 26 :=
by {
  sorry
}

end average_age_is_26_l229_229338


namespace smallest_number_ending_in_9_divisible_by_13_l229_229030

theorem smallest_number_ending_in_9_divisible_by_13 :
  ∃ (n : ℕ), (n % 10 = 9) ∧ (13 ∣ n) ∧ (∀ (m : ℕ), (m % 10 = 9) ∧ (13 ∣ m) ∧ (m < n) -> false) :=
sorry

end smallest_number_ending_in_9_divisible_by_13_l229_229030


namespace length_of_uncovered_side_l229_229051

theorem length_of_uncovered_side (L W : ℕ) (h1 : L * W = 680) (h2 : 2 * W + L = 74) : L = 40 :=
sorry

end length_of_uncovered_side_l229_229051


namespace students_not_finding_parents_funny_l229_229099

theorem students_not_finding_parents_funny:
  ∀ (total_students funny_dad funny_mom funny_both : ℕ),
  total_students = 50 →
  funny_dad = 25 →
  funny_mom = 30 →
  funny_both = 18 →
  (total_students - (funny_dad + funny_mom - funny_both) = 13) :=
by
  intros total_students funny_dad funny_mom funny_both
  sorry

end students_not_finding_parents_funny_l229_229099


namespace inequality_one_inequality_two_l229_229738

variable {a b r s : ℝ}

theorem inequality_one (h_a : 0 < a) (h_b : 0 < b) :
  a^2 * b ≤ 4 * ((a + b) / 3)^3 :=
sorry

theorem inequality_two (h_a : 0 < a) (h_b : 0 < b) (h_r : 0 < r) (h_s : 0 < s) 
  (h_eq : 1 / r + 1 / s = 1) : 
  (a^r / r) + (b^s / s) ≥ a * b :=
sorry

end inequality_one_inequality_two_l229_229738


namespace smallest_four_digit_in_pascals_triangle_l229_229458

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, 1000 ≤ nat.choose n k ∧ nat.choose n k < 10000 ∧ 
  (∀ m l : ℕ, (nat.choose m l ≥ 1000 ∧ nat.choose m l < 10000) → nat.choose n k ≤ nat.choose m l) :=
begin
  -- This is a statement only, no proof is provided
  sorry
end

end smallest_four_digit_in_pascals_triangle_l229_229458


namespace value_of_x_plus_y_l229_229848

theorem value_of_x_plus_y (x y : ℝ) (h1 : 1 / x + 1 / y = 4) (h2 : 1 / x - 1 / y = -6) : x + y = -4 / 5 :=
sorry

end value_of_x_plus_y_l229_229848


namespace work_days_l229_229549

theorem work_days (m r d : ℕ) (h : 2 * m * d = 2 * (m + r) * (md / (m + r))) : d = md / (m + r) :=
by
  sorry

end work_days_l229_229549


namespace emma_chocolates_l229_229686

theorem emma_chocolates 
  (x : ℕ) 
  (h1 : ∃ l : ℕ, x = l + 10) 
  (h2 : ∃ l : ℕ, l = x / 3) : 
  x = 15 := 
  sorry

end emma_chocolates_l229_229686


namespace find_f_7_5_l229_229880

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_2  : ∀ x, f (x + 2) = -f x
axiom initial_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem find_f_7_5 : f 7.5 = -0.5 :=
by
  -- Proof goes here
  sorry

end find_f_7_5_l229_229880


namespace range_of_m_l229_229355

def A (x : ℝ) : Prop := x^2 - x - 6 > 0
def B (x m : ℝ) : Prop := (x - m) * (x - 2 * m) ≤ 0
def is_disjoint (A B : ℝ → Prop) : Prop := ∀ x, ¬ (A x ∧ B x)

theorem range_of_m (m : ℝ) : 
  is_disjoint (A) (B m) ↔ -1 ≤ m ∧ m ≤ 3 / 2 := by
  sorry

end range_of_m_l229_229355


namespace exponential_function_passes_through_01_l229_229904

theorem exponential_function_passes_through_01 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : (a^0 = 1) :=
by
  sorry

end exponential_function_passes_through_01_l229_229904


namespace calculate_expr1_calculate_expr2_l229_229178

/-- Statement 1: -5 * 3 - 8 / -2 = -11 -/
theorem calculate_expr1 : (-5) * 3 - 8 / -2 = -11 :=
by sorry

/-- Statement 2: (-1)^3 + (5 - (-3)^2) / 6 = -5/3 -/
theorem calculate_expr2 : (-1)^3 + (5 - (-3)^2) / 6 = -(5 / 3) :=
by sorry

end calculate_expr1_calculate_expr2_l229_229178


namespace solve_x_l229_229985

theorem solve_x :
  ∃ x : ℝ, 2.5 * ( ( x * 0.48 * 2.50 ) / ( 0.12 * 0.09 * 0.5 ) ) = 2000.0000000000002 ∧ x = 3.6 :=
by sorry

end solve_x_l229_229985


namespace correct_result_l229_229154

theorem correct_result (x : ℤ) (h : x * 3 - 5 = 103) : (x / 3) - 5 = 7 :=
sorry

end correct_result_l229_229154


namespace smallest_four_digit_in_pascal_l229_229414

-- define what it means for a number to be in Pascal's triangle
def in_pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- define the concept of a smallest four-digit number
def smallest_four_digit : ℕ := 1000

-- prove that the smallest four-digit number in Pascal's Triangle is 1000
theorem smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), in_pascal n k = smallest_four_digit := 
sorry

end smallest_four_digit_in_pascal_l229_229414


namespace appropriate_length_of_presentation_l229_229382

theorem appropriate_length_of_presentation (wpm : ℕ) (min_time min_words max_time max_words total_words : ℕ) 
  (h1 : total_words = 160) 
  (h2 : min_time = 45) 
  (h3 : min_words = min_time * wpm) 
  (h4 : max_time = 60) 
  (h5 : max_words = max_time * wpm) : 
  7200 ≤ 9400 ∧ 9400 ≤ 9600 :=
by 
  sorry

end appropriate_length_of_presentation_l229_229382


namespace ratio_sub_div_a_l229_229226

theorem ratio_sub_div_a (a b : ℝ) (h : a / b = 5 / 8) : (b - a) / a = 3 / 5 :=
sorry

end ratio_sub_div_a_l229_229226


namespace root_of_quadratic_poly_l229_229754

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

def poly_has_root (a b c : ℝ) (r : ℝ) : Prop := a * r^2 + b * r + c = 0

theorem root_of_quadratic_poly 
  (a b c : ℝ)
  (h1 : discriminant a b c = 0)
  (h2 : discriminant (-a) (b - 30 * a) (17 * a - 7 * b + c) = 0):
  poly_has_root a b c (-11) :=
sorry

end root_of_quadratic_poly_l229_229754


namespace value_of_x_plus_y_l229_229849

theorem value_of_x_plus_y (x y : ℝ) (h1 : 1 / x + 1 / y = 4) (h2 : 1 / x - 1 / y = -6) : x + y = -4 / 5 :=
sorry

end value_of_x_plus_y_l229_229849


namespace negation_of_universal_l229_229133

theorem negation_of_universal :
  (¬ (∀ k : ℝ, ∃ x y : ℝ, x^2 + y^2 = 2 ∧ y = k * x + 1)) ↔ 
  (∃ k : ℝ, ¬ ∃ x y : ℝ, x^2 + y^2 = 2 ∧ y = k * x + 1) :=
by
  sorry

end negation_of_universal_l229_229133


namespace mandy_chocolate_pieces_l229_229304

def chocolate_pieces_total : ℕ := 60
def half (n : ℕ) : ℕ := n / 2

def michael_taken : ℕ := half chocolate_pieces_total
def paige_taken : ℕ := half (chocolate_pieces_total - michael_taken)
def ben_taken : ℕ := half (chocolate_pieces_total - michael_taken - paige_taken)
def mandy_left : ℕ := chocolate_pieces_total - michael_taken - paige_taken - ben_taken

theorem mandy_chocolate_pieces : mandy_left = 8 :=
  by
  -- proof to be provided here
  sorry

end mandy_chocolate_pieces_l229_229304


namespace solve_for_c_l229_229541

variables (m c b a : ℚ) -- Declaring variables as rationals for added precision

theorem solve_for_c (h : m = (c * b * a) / (a - c)) : 
  c = (m * a) / (m + b * a) := 
by 
  sorry -- Proof not required as per the instructions

end solve_for_c_l229_229541


namespace order_xyz_l229_229838

theorem order_xyz (x : ℝ) (h1 : 0.8 < x) (h2 : x < 0.9) :
  let y := x^x
  let z := x^(x^x)
  x < z ∧ z < y :=
by
  sorry

end order_xyz_l229_229838


namespace smallest_four_digit_number_in_pascals_triangle_l229_229436

theorem smallest_four_digit_number_in_pascals_triangle : ∃ n k, binom n k = 1000 ∧ (∀ n' k', binom n' k' > 1000 → nat.zero < n' k' ∧ n' k' < (binom n k)) := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l229_229436


namespace smallest_four_digit_in_pascals_triangle_l229_229454

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, 1000 ≤ nat.choose n k ∧ nat.choose n k < 10000 ∧ 
  (∀ m l : ℕ, (nat.choose m l ≥ 1000 ∧ nat.choose m l < 10000) → nat.choose n k ≤ nat.choose m l) :=
begin
  -- This is a statement only, no proof is provided
  sorry
end

end smallest_four_digit_in_pascals_triangle_l229_229454


namespace max_regions_with_6_chords_l229_229975

-- Definition stating the number of regions created by k chords
def regions_by_chords (k : ℕ) : ℕ :=
  1 + (k * (k + 1)) / 2

-- Lean statement for the proof problem
theorem max_regions_with_6_chords : regions_by_chords 6 = 22 :=
  by sorry

end max_regions_with_6_chords_l229_229975


namespace exists_unique_integer_pair_l229_229348

theorem exists_unique_integer_pair (a : ℕ) (ha : 0 < a) :
  ∃! (x y : ℕ), 0 < x ∧ 0 < y ∧ x + (x + y - 1) * (x + y - 2) / 2 = a :=
by
  sorry

end exists_unique_integer_pair_l229_229348


namespace smallest_four_digit_in_pascals_triangle_l229_229440

/-- Pascal's triangle is defined using binomial coefficients.
    We want to prove that the smallest four-digit number in Pascal's triangle is 1001. -/
theorem smallest_four_digit_in_pascals_triangle : ∃ n k : ℕ, n ≥ k ∧ (binomial n k = 1001 ∧ ∀ m l, m ≥ l → binomial m l ≥ 1000 → binomial m l = 1001 → n = m ∧ k = l) := by
  sorry

end smallest_four_digit_in_pascals_triangle_l229_229440


namespace chocolate_chips_per_member_l229_229715

/-
Define the problem conditions:
-/
def family_members := 4
def batches_choc_chip := 3
def cookies_per_batch_choc_chip := 12
def chips_per_cookie_choc_chip := 2
def batches_double_choc_chip := 2
def cookies_per_batch_double_choc_chip := 10
def chips_per_cookie_double_choc_chip := 4

/-
State the theorem to be proved:
-/
theorem chocolate_chips_per_member : 
  let total_choc_chip_cookies := batches_choc_chip * cookies_per_batch_choc_chip
  let total_choc_chips_choc_chip := total_choc_chip_cookies * chips_per_cookie_choc_chip
  let total_double_choc_chip_cookies := batches_double_choc_chip * cookies_per_batch_double_choc_chip
  let total_choc_chips_double_choc_chip := total_double_choc_chip_cookies * chips_per_cookie_double_choc_chip
  let total_choc_chips := total_choc_chips_choc_chip + total_choc_chips_double_choc_chip
  let chips_per_member := total_choc_chips / family_members
  chips_per_member = 38 :=
by
  sorry

end chocolate_chips_per_member_l229_229715


namespace v_is_82_875_percent_of_z_l229_229860

theorem v_is_82_875_percent_of_z (x y z w v : ℝ) 
  (h1 : x = 1.30 * y)
  (h2 : y = 0.60 * z)
  (h3 : w = 1.25 * x)
  (h4 : v = 0.85 * w) : 
  v = 0.82875 * z :=
by
  sorry

end v_is_82_875_percent_of_z_l229_229860


namespace lcm_153_180_560_l229_229148

theorem lcm_153_180_560 : Nat.lcm (Nat.lcm 153 180) 560 = 85680 :=
by
  sorry

end lcm_153_180_560_l229_229148


namespace solve_congruence_l229_229741

theorem solve_congruence (n : ℕ) (h₀ : 0 ≤ n ∧ n < 47) (h₁ : 13 * n ≡ 5 [MOD 47]) :
  n = 4 :=
sorry

end solve_congruence_l229_229741


namespace right_triangle_hypotenuse_length_l229_229101

theorem right_triangle_hypotenuse_length (a b c : ℕ) (h₁ : a = 5) (h₂ : b = 12)
  (h₃ : c^2 = a^2 + b^2) : c = 13 :=
by
  -- We should provide the actual proof here, but we'll use sorry for now.
  sorry

end right_triangle_hypotenuse_length_l229_229101


namespace sum_of_other_endpoint_coordinates_l229_229908

theorem sum_of_other_endpoint_coordinates (x y : ℤ) :
  (7 + x) / 2 = 5 ∧ (4 + y) / 2 = -8 → x + y = -17 :=
by 
  sorry

end sum_of_other_endpoint_coordinates_l229_229908


namespace largest_divisor_36_l229_229556

theorem largest_divisor_36 (n : ℕ) (h : n > 0) (h_div : 36 ∣ n^3) : 6 ∣ n := 
sorry

end largest_divisor_36_l229_229556


namespace probability_of_rolling_five_on_six_sided_die_l229_229778

theorem probability_of_rolling_five_on_six_sided_die :
  let S := {1, 2, 3, 4, 5, 6}
  let |S| := 6
  let A := {5}
  let |A| := 1
  probability A S = 1 / 6 := by
  -- Proof goes here
  sorry

end probability_of_rolling_five_on_six_sided_die_l229_229778


namespace necessary_and_sufficient_condition_l229_229832

theorem necessary_and_sufficient_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (|a + b| = |a| + |b|) ↔ (a * b > 0) :=
sorry

end necessary_and_sufficient_condition_l229_229832


namespace sum_50th_set_correct_l229_229391

noncomputable def sum_of_fiftieth_set : ℕ := 195 + 197

theorem sum_50th_set_correct : sum_of_fiftieth_set = 392 :=
by 
  -- The proof would go here
  sorry

end sum_50th_set_correct_l229_229391


namespace volume_increase_l229_229131

theorem volume_increase (L B H : ℝ) :
  let L_new := 1.25 * L
  let B_new := 0.85 * B
  let H_new := 1.10 * H
  (L_new * B_new * H_new) = 1.16875 * (L * B * H) := 
by
  sorry

end volume_increase_l229_229131


namespace total_difference_in_cups_l229_229395

theorem total_difference_in_cups (h1: Nat) (h2: Nat) (h3: Nat) (hrs: Nat) : 
  h1 = 4 → h2 = 7 → h3 = 5 → hrs = 3 → 
  ((h2 * hrs - h1 * hrs) + (h3 * hrs - h1 * hrs) + (h2 * hrs - h3 * hrs)) = 18 :=
by
  intros h1_eq h2_eq h3_eq hrs_eq
  sorry

end total_difference_in_cups_l229_229395


namespace inequality_correctness_l229_229151

theorem inequality_correctness (a b c : ℝ) (h : c^2 > 0) : (a * c^2 > b * c^2) ↔ (a > b) := by 
sorry

end inequality_correctness_l229_229151


namespace range_of_a_l229_229852

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_f : ∀ x, f x = x^3 - 3*x + a) 
  (has_3_distinct_roots : ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0) : 
  -2 < a ∧ a < 2 := 
by
  sorry

end range_of_a_l229_229852


namespace find_k_of_sequence_l229_229354

theorem find_k_of_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n, S n = n^2 - 9 * n)
  (hS_recurr : ∀ n ≥ 2, a n = S n - S (n-1)) (h_a_k : ∃ k, 5 < a k ∧ a k < 8) : ∃ k, k = 8 :=
by
  sorry

end find_k_of_sequence_l229_229354


namespace ellipse_foci_on_y_axis_l229_229352

theorem ellipse_foci_on_y_axis (theta : ℝ) (h1 : 0 < theta ∧ theta < π)
  (h2 : Real.sin theta + Real.cos theta = 1 / 2) :
  (0 < theta ∧ theta < π / 2) → 
  (0 < theta ∧ theta < 3 * π / 4) → 
  -- The equation x^2 * sin theta - y^2 * cos theta = 1 represents an ellipse with foci on the y-axis
  ∃ foci_on_y_axis : Prop, foci_on_y_axis := 
sorry

end ellipse_foci_on_y_axis_l229_229352


namespace hexagon_ratio_l229_229568

noncomputable def ratio_of_hexagon_areas (s : ℝ) : ℝ :=
  let area_ABCDEF := (3 * Real.sqrt 3 / 2) * s^2
  let side_smaller := (3 * s) / 2
  let area_smaller := (3 * Real.sqrt 3 / 2) * side_smaller^2
  area_smaller / area_ABCDEF

theorem hexagon_ratio (s : ℝ) : ratio_of_hexagon_areas s = 9 / 4 :=
by
  sorry

end hexagon_ratio_l229_229568


namespace problem1_inequality_problem2_inequality_l229_229891

theorem problem1_inequality (x : ℝ) (h1 : 2 * x + 10 ≤ 5 * x + 1) (h2 : 3 * (x - 1) > 9) : x > 4 := sorry

theorem problem2_inequality (x : ℝ) (h1 : 3 * (x + 2) ≥ 2 * x + 5) (h2 : 2 * x - (3 * x + 1) / 2 < 1) : -1 ≤ x ∧ x < 3 := sorry

end problem1_inequality_problem2_inequality_l229_229891


namespace eq_of_line_through_points_l229_229906

noncomputable def line_eqn (x y : ℝ) : Prop :=
  x - y + 3 = 0

theorem eq_of_line_through_points :
  ∀ (x1 y1 x2 y2 : ℝ), 
    x1 = -1 → y1 = 2 → x2 = 2 → y2 = 5 → 
    line_eqn (x1 + y1 - x2) (y2 - y1) :=
by
  intros x1 y1 x2 y2 hx1 hy1 hx2 hy2
  rw [hx1, hy1, hx2, hy2]
  sorry -- Proof steps would go here.

end eq_of_line_through_points_l229_229906


namespace largest_angle_of_consecutive_integer_angles_of_hexagon_l229_229398

theorem largest_angle_of_consecutive_integer_angles_of_hexagon 
  (angles : Fin 6 → ℝ)
  (h_consecutive : ∃ (x : ℝ), angles = ![
    x - 3, x - 2, x - 1, x, x + 1, x + 2 ])
  (h_sum : (angles 0 + angles 1 + angles 2 + angles 3 + angles 4 + angles 5) = 720) :
  (angles 5 = 122.5) :=
by
  sorry

end largest_angle_of_consecutive_integer_angles_of_hexagon_l229_229398


namespace smallest_four_digit_in_pascal_l229_229460

theorem smallest_four_digit_in_pascal :
  ∃ n k : ℕ, binomial n k = 1000 ∧ 
             (∀ a b : ℕ, binomial a b ∈ set.Icc 1000 9999 → a ≥ n) :=
sorry

end smallest_four_digit_in_pascal_l229_229460


namespace intervals_increasing_max_min_value_range_of_m_l229_229221

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 1/2

theorem intervals_increasing : ∀ (x : ℝ), ∃ k : ℤ, -π/6 + k * π ≤ x ∧ x ≤ π/3 + k * π := sorry

theorem max_min_value (x : ℝ) (hx : π/4 ≤ x ∧ x ≤ π/2) :
  (f (π/3) = 0) ∧ (f (π/2) = -1/2) :=
  sorry

theorem range_of_m (x : ℝ) (hx : π/4 ≤ x ∧ x ≤ π/2) :
  ∀ m : ℝ, (∀ y : ℝ, (π/4 ≤ y ∧ y ≤ π/2) → |f y - m| < 1) ↔ (-1 < m ∧ m < 1/2) :=
  sorry

end intervals_increasing_max_min_value_range_of_m_l229_229221


namespace find_f_1_0_plus_f_2_0_general_form_F_l229_229992

variable {F : ℝ → ℝ → ℝ}

-- Conditions
axiom cond1 : ∀ a, F a a = a
axiom cond2 : ∀ (k a b : ℝ), F (k * a) (k * b) = k * F a b
axiom cond3 : ∀ (a1 a2 b1 b2 : ℝ), F (a1 + a2) (b1 + b2) = F a1 b1 + F a2 b2
axiom cond4 : ∀ (a b : ℝ), F a b = F b ((a + b) / 2)

-- Proof problem
theorem find_f_1_0_plus_f_2_0 : F 1 0 + F 2 0 = 0 :=
sorry

theorem general_form_F : ∀ (x y : ℝ), F x y = y :=
sorry

end find_f_1_0_plus_f_2_0_general_form_F_l229_229992


namespace arccos_sin_three_pi_over_two_eq_pi_l229_229794

theorem arccos_sin_three_pi_over_two_eq_pi : 
  Real.arccos (Real.sin (3 * Real.pi / 2)) = Real.pi :=
by
  sorry

end arccos_sin_three_pi_over_two_eq_pi_l229_229794


namespace volume_conversion_l229_229629

-- Define the given conditions
def V_feet : ℕ := 216
def C_factor : ℕ := 27

-- State the theorem to prove
theorem volume_conversion : V_feet / C_factor = 8 :=
  sorry

end volume_conversion_l229_229629


namespace necessary_but_not_sufficient_l229_229669

theorem necessary_but_not_sufficient (p q : Prop) : 
  (p ∨ q) → (p ∧ q) → False :=
by
  sorry

end necessary_but_not_sufficient_l229_229669


namespace fraction_transformation_l229_229364

variables (a b : ℝ)

theorem fraction_transformation (ha : a ≠ 0) (hb : b ≠ 0) : 
  (4 * a * b) / (2 * (2 * a) + 2 * b) = 2 * (a * b) / (2 * a + b) :=
by
  sorry

end fraction_transformation_l229_229364


namespace shaded_area_concentric_circles_l229_229903

theorem shaded_area_concentric_circles 
  (CD_length : ℝ) (r1 r2 : ℝ) 
  (h_tangent : CD_length = 100) 
  (h_radius1 : r1 = 60) 
  (h_radius2 : r2 = 40) 
  (tangent_condition : CD_length = 2 * real.sqrt (r1^2 - r2^2)) :
  ∃ area : ℝ, area = π * (r1^2 - r2^2) ∧ area = 2000 * π :=
by
  use π * (r1^2 - r2^2)
  have h1 : r1^2 = 3600 := by { rw h_radius1, norm_num }
  have h2 : r2^2 = 1600 := by { rw h_radius2, norm_num }
  rw [h1, h2]
  simp
  sorry

end shaded_area_concentric_circles_l229_229903


namespace area_of_rhombus_l229_229881

variable (a b θ : ℝ)
variable (h_a : 0 < a) (h_b : 0 < b)

theorem area_of_rhombus (h : true) : (2 * a) * (2 * b) / 2 = 2 * a * b := by
  sorry

end area_of_rhombus_l229_229881


namespace converse_of_x_eq_one_implies_x_squared_eq_one_l229_229897

theorem converse_of_x_eq_one_implies_x_squared_eq_one (x : ℝ) : x^2 = 1 → x = 1 := 
sorry

end converse_of_x_eq_one_implies_x_squared_eq_one_l229_229897


namespace arithmetic_mean_is_correct_l229_229968

-- Define the numbers
def num1 : ℕ := 18
def num2 : ℕ := 27
def num3 : ℕ := 45

-- Define the number of terms
def n : ℕ := 3

-- Define the sum of the numbers
def total_sum : ℕ := num1 + num2 + num3

-- Define the arithmetic mean
def arithmetic_mean : ℕ := total_sum / n

-- Theorem stating that the arithmetic mean of the numbers is 30
theorem arithmetic_mean_is_correct : arithmetic_mean = 30 := by
  -- Proof goes here
  sorry

end arithmetic_mean_is_correct_l229_229968


namespace P_lt_Q_l229_229347

theorem P_lt_Q (x : ℝ) (hx : x > 0) (P : ℝ) (Q : ℝ) 
  (hP : P = Real.sqrt (1 + x)) 
  (hQ : Q = 1 + x / 2) : P < Q := 
by
  sorry

end P_lt_Q_l229_229347


namespace number_of_pupils_in_class_l229_229496

-- Defining the conditions
def wrongMark : ℕ := 79
def correctMark : ℕ := 45
def averageIncreasedByHalf : ℕ := 2  -- Condition representing average increased by half

-- The goal is to prove the number of pupils is 68
theorem number_of_pupils_in_class (n S : ℕ) (h1 : wrongMark = 79) (h2 : correctMark = 45)
(h3 : averageIncreasedByHalf = 2) 
(h4 : S + (wrongMark - correctMark) = (3 / 2) * S) :
  n = 68 :=
  sorry

end number_of_pupils_in_class_l229_229496


namespace solve_fractional_equation_l229_229002

theorem solve_fractional_equation (x : ℝ) (h : (3 / (x + 1) - 2 / (x - 1)) = 0) : x = 5 :=
sorry

end solve_fractional_equation_l229_229002


namespace tetrahedron_faces_congruent_iff_face_angle_sum_straight_l229_229123

-- Defining the Tetrahedron and its properties
structure Tetrahedron (V : Type*) :=
(A B C D : V)
(face_angle_sum_at_vertex : V → Prop)
(congruent_faces : Prop)

-- Translating the problem into a Lean 4 theorem statement
theorem tetrahedron_faces_congruent_iff_face_angle_sum_straight (V : Type*) 
  (T : Tetrahedron V) :
  T.face_angle_sum_at_vertex T.A = T.face_angle_sum_at_vertex T.B ∧ 
  T.face_angle_sum_at_vertex T.B = T.face_angle_sum_at_vertex T.C ∧ 
  T.face_angle_sum_at_vertex T.C = T.face_angle_sum_at_vertex T.D ↔ T.congruent_faces :=
sorry


end tetrahedron_faces_congruent_iff_face_angle_sum_straight_l229_229123


namespace train_length_l229_229296

theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) (h_speed : speed_kmh = 30) (h_time : time_sec = 6) :
  ∃ length_meters : ℝ, abs (length_meters - 50) < 1 :=
by
  -- Converting speed from km/hr to m/s
  let speed_ms := speed_kmh * (1000 / 3600)
  
  -- Calculating length of the train using the distance formula
  let length_meters := speed_ms * time_sec

  use length_meters
  -- Proof would go here showing abs (length_meters - 50) < 1
  sorry

end train_length_l229_229296


namespace remainder_when_divided_by_39_l229_229306

theorem remainder_when_divided_by_39 (N k : ℤ) (h : N = 13 * k + 4) : (N % 39) = 4 :=
sorry

end remainder_when_divided_by_39_l229_229306


namespace tv_show_duration_l229_229384

theorem tv_show_duration (total_air_time : ℝ) (num_commercials : ℕ) (commercial_duration_min : ℝ) :
  total_air_time = 1.5 ∧ num_commercials = 3 ∧ commercial_duration_min = 10 →
  (total_air_time - (num_commercials * commercial_duration_min / 60)) = 1 :=
by
  sorry

end tv_show_duration_l229_229384


namespace quadratic_inequality_range_l229_229228

variable (x : ℝ)

-- Statement of the mathematical problem
theorem quadratic_inequality_range (h : ¬ (x^2 - 5 * x + 4 > 0)) : 1 ≤ x ∧ x ≤ 4 :=
by
  sorry

end quadratic_inequality_range_l229_229228


namespace units_digit_17_pow_2023_l229_229031

theorem units_digit_17_pow_2023 : (17^2023 % 10) = 3 := sorry

end units_digit_17_pow_2023_l229_229031


namespace ratio_of_trees_l229_229315

theorem ratio_of_trees (plums pears apricots : ℕ) (h_plums : plums = 3) (h_pears : pears = 3) (h_apricots : apricots = 3) :
  plums = pears ∧ pears = apricots :=
by
  sorry

end ratio_of_trees_l229_229315


namespace tires_in_parking_lot_l229_229855

theorem tires_in_parking_lot (num_cars : ℕ) (regular_tires_per_car spare_tire : ℕ) (h1 : num_cars = 30) (h2 : regular_tires_per_car = 4) (h3 : spare_tire = 1) :
  num_cars * (regular_tires_per_car + spare_tire) = 150 :=
by
  sorry

end tires_in_parking_lot_l229_229855


namespace prime_gt_three_times_n_l229_229571

def nth_prime (n : ℕ) : ℕ :=
  -- Define the nth prime function, can use mathlib functionality
  sorry

theorem prime_gt_three_times_n (n : ℕ) (h : 12 ≤ n) : nth_prime n > 3 * n :=
  sorry

end prime_gt_three_times_n_l229_229571


namespace angle_A_range_find_b_l229_229230

-- Definitions based on problem conditions
variable {a b c S : ℝ}
variable {A B C : ℝ}
variable {x : ℝ}

-- First statement: range of values for A
theorem angle_A_range (h1 : c * b * Real.cos A ≤ 2 * Real.sqrt 3 * S)
                      (h2 : S = 1/2 * b * c * Real.sin A)
                      (h3 : 0 < A ∧ A < π) : π / 6 ≤ A ∧ A < π := 
sorry

-- Second statement: value of b
theorem find_b (h1 : Real.tan A = x ∧ Real.tan B = 2 * x ∧ Real.tan C = 3 * x)
               (h2 : x = 1)
               (h3 : c = 1) : b = 2 * Real.sqrt 2 / 3 :=
sorry

end angle_A_range_find_b_l229_229230


namespace problem_l229_229843

theorem problem (x y : ℝ)
  (h1 : 1 / x + 1 / y = 4)
  (h2 : 1 / x - 1 / y = -6) :
  x + y = -4 / 5 :=
sorry

end problem_l229_229843


namespace baskets_count_l229_229008

theorem baskets_count (total_apples apples_per_basket : ℕ) (h1 : total_apples = 629) (h2 : apples_per_basket = 17) : (total_apples / apples_per_basket) = 37 :=
by
  sorry

end baskets_count_l229_229008


namespace last_digit_101_pow_100_l229_229413

theorem last_digit_101_pow_100 :
  (101^100) % 10 = 1 :=
by
  sorry

end last_digit_101_pow_100_l229_229413


namespace largest_y_coordinate_of_graph_l229_229083

theorem largest_y_coordinate_of_graph :
  ∀ (x y : ℝ), (x^2 / 49 + (y - 3)^2 / 25 = 0) → y = 3 :=
by
  sorry

end largest_y_coordinate_of_graph_l229_229083


namespace equal_distribution_l229_229788

namespace MoneyDistribution

def Ann_initial := 777
def Bill_initial := 1111
def Charlie_initial := 1555
def target_amount := 1148
def Bill_to_Ann := 371
def Charlie_to_Bill := 408

theorem equal_distribution :
  (Bill_initial - Bill_to_Ann + Charlie_to_Bill = target_amount) ∧
  (Ann_initial + Bill_to_Ann = target_amount) ∧
  (Charlie_initial - Charlie_to_Bill = target_amount) :=
by
  sorry

end MoneyDistribution

end equal_distribution_l229_229788


namespace sum_of_x_and_y_l229_229844

theorem sum_of_x_and_y :
  ∀ (x y : ℚ), (1/x + 1/y = 4) → (1/x - 1/y = -6) → x + y = -4/5 :=
by
  intros x y h1 h2
  sorry

end sum_of_x_and_y_l229_229844


namespace mail_total_correct_l229_229720

def Monday_mail : ℕ := 65
def Tuesday_mail : ℕ := Monday_mail + 10
def Wednesday_mail : ℕ := Tuesday_mail - 5
def Thursday_mail : ℕ := Wednesday_mail + 15
def total_mail : ℕ := Monday_mail + Tuesday_mail + Wednesday_mail + Thursday_mail

theorem mail_total_correct : total_mail = 295 := by
  sorry

end mail_total_correct_l229_229720


namespace area_of_triangle_l229_229097

theorem area_of_triangle (a b c : ℝ) (C : ℝ) 
  (h1 : c^2 = (a - b)^2 + 6)
  (h2 : C = Real.pi / 3) 
  : (1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) :=
sorry

end area_of_triangle_l229_229097


namespace ab_value_l229_229287

theorem ab_value (a b : ℝ) (h1 : a + b = 7) (h2 : a^3 + b^3 = 91) : a * b = 12 :=
by
  sorry

end ab_value_l229_229287


namespace pig_duck_ratio_l229_229410

theorem pig_duck_ratio (G C D P : ℕ)
(h₁ : G = 66)
(h₂ : C = 2 * G)
(h₃ : D = (G + C) / 2)
(h₄ : P = G - 33) :
  P / D = 1 / 3 :=
by {
  sorry
}

end pig_duck_ratio_l229_229410


namespace amgm_inequality_proof_l229_229574

noncomputable def amgm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : Prop :=
  1 < (a / (Real.sqrt (a^2 + b^2))) + (b / (Real.sqrt (b^2 + c^2))) + (c / (Real.sqrt (c^2 + a^2))) 
  ∧ (a / (Real.sqrt (a^2 + b^2))) + (b / (Real.sqrt (b^2 + c^2))) + (c / (Real.sqrt (c^2 + a^2))) 
  ≤ (3 * Real.sqrt 2) / 2

theorem amgm_inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  amgm_inequality a b c ha hb hc := 
sorry

end amgm_inequality_proof_l229_229574


namespace part1_solution_count_part2_solution_count_l229_229303

theorem part1_solution_count :
  ∃ (solutions : Finset (ℕ × ℕ × ℕ)), solutions.card = 7 ∧
    ∀ (m n r : ℕ), (m, n, r) ∈ solutions ↔ mn + nr + mr = 2 * (m + n + r) := sorry

theorem part2_solution_count (k : ℕ) (h : 1 < k) :
  ∃ (solutions : Finset (ℕ × ℕ × ℕ)), solutions.card ≥ 3 * k + 1 ∧
    ∀ (m n r : ℕ), (m, n, r) ∈ solutions ↔ mn + nr + mr = k * (m + n + r) := sorry

end part1_solution_count_part2_solution_count_l229_229303


namespace correct_answer_l229_229759

noncomputable def original_number (y : ℝ) :=
  (y - 14) / 2 = 50

theorem correct_answer (y : ℝ) (h : original_number y) :
  (y - 5) / 7 = 15 :=
by
  sorry

end correct_answer_l229_229759


namespace first_math_festival_divisibility_largest_ordinal_number_divisibility_l229_229280

-- Definition of the conditions for part (a)
def first_math_festival_year : ℕ := 1990
def first_ordinal_number : ℕ := 1

-- Statement for part (a)
theorem first_math_festival_divisibility : first_math_festival_year % first_ordinal_number = 0 :=
sorry

-- Definition of the conditions for part (b)
def nth_math_festival_year (N : ℕ) : ℕ := 1989 + N

-- Statement for part (b)
theorem largest_ordinal_number_divisibility : ∀ N : ℕ, 
  (nth_math_festival_year N) % N = 0 → N ≤ 1989 :=
sorry

end first_math_festival_divisibility_largest_ordinal_number_divisibility_l229_229280


namespace gcf_75_100_l229_229336

theorem gcf_75_100 : Nat.gcd 75 100 = 25 :=
by
  sorry

end gcf_75_100_l229_229336


namespace smallest_four_digit_number_in_pascals_triangle_l229_229450

theorem smallest_four_digit_number_in_pascals_triangle :
  ∃ (n : ℕ), 1000 = (n choose min n) ∧ (n > 0 ∧ (∀ m < n, (m choose min m) < 1000)) :=
begin
  sorry
end

end smallest_four_digit_number_in_pascals_triangle_l229_229450


namespace carpet_needed_correct_l229_229498

def length_room : ℕ := 15
def width_room : ℕ := 9
def length_closet : ℕ := 3
def width_closet : ℕ := 2

def area_room : ℕ := length_room * width_room
def area_closet : ℕ := length_closet * width_closet
def area_to_carpet : ℕ := area_room - area_closet
def sq_ft_to_sq_yd (sqft: ℕ) : ℕ := (sqft + 8) / 9  -- Adding 8 to ensure proper rounding up

def carpet_needed : ℕ := sq_ft_to_sq_yd area_to_carpet

theorem carpet_needed_correct :
  carpet_needed = 15 := by
  sorry

end carpet_needed_correct_l229_229498


namespace probability_of_differ_by_three_l229_229957

def is_valid_roll (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6
def differ_by_three (a b : ℕ) : Prop := abs (a - b) = 3

theorem probability_of_differ_by_three :
  let successful_outcomes := ([
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ] : List (ℕ × ℕ)) in
  let total_outcomes := 6 * 6 in
  (List.length successful_outcomes : ℝ) / total_outcomes = 1 / 6 :=
by
  -- Definitions and assumptions
  let successful_outcomes := [
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ]
  let total_outcomes := 6 * 6
  
  -- Statement of the theorem
  have h_successful : successful_outcomes.length = 6 := sorry
  have h_total : total_outcomes = 36 := by norm_num
  have h_probability := h_successful
    ▸ h_total ▸ (6 / 36 : ℝ) = (1 / 6 : ℝ) := by norm_num
  exact h_probability

end probability_of_differ_by_three_l229_229957


namespace product_evaluation_l229_229328

theorem product_evaluation :
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 :=
by sorry

end product_evaluation_l229_229328


namespace fraction_expression_eq_l229_229092

theorem fraction_expression_eq (x y : ℕ) (hx : x = 4) (hy : y = 5) : 
  ((1 / y) + (1 / x)) / (1 / x) = 9 / 5 :=
by
  rw [hx, hy]
  sorry

end fraction_expression_eq_l229_229092


namespace smallest_four_digit_in_pascals_triangle_l229_229447

theorem smallest_four_digit_in_pascals_triangle : ∃ n, ∃ k, k ≤ n ∧ 1000 ≤ Nat.choose n k :=
sorry

end smallest_four_digit_in_pascals_triangle_l229_229447


namespace natural_numbers_fitting_description_l229_229782

theorem natural_numbers_fitting_description (n : ℕ) (h : 1 / (n : ℚ) + 1 / 2 = 1 / 3 + 2 / (n + 1)) : n = 2 ∨ n = 3 :=
by
  sorry

end natural_numbers_fitting_description_l229_229782


namespace range_of_fx_l229_229533

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^k

theorem range_of_fx (k : ℝ) (x : ℝ) (h1 : k < -1) (h2 : x ∈ Set.Ici (0.5)) :
  Set.Icc (0 : ℝ) 2 = {y | ∃ x, f x k = y ∧ x ∈ Set.Ici 0.5} :=
sorry

end range_of_fx_l229_229533


namespace units_digit_17_pow_2023_l229_229033

theorem units_digit_17_pow_2023 
  (cycle : ℕ → ℕ)
  (h1 : cycle 0 = 7)
  (h2 : cycle 1 = 9)
  (h3 : cycle 2 = 3)
  (h4 : cycle 3 = 1)
  (units_digit : ℕ → ℕ)
  (h_units : ∀ n, units_digit (17^n) = units_digit (7^n))
  (h_units_cycle : ∀ n, units_digit (7^n) = cycle (n % 4)) :
  units_digit (17^2023) = 3 :=
by
  sorry

end units_digit_17_pow_2023_l229_229033


namespace upper_bound_y_l229_229552

/-- 
  Theorem:
  For any real numbers x and y such that 3 < x < 6 and 6 < y, 
  if the greatest possible positive integer difference between x and y is 6,
  then the upper bound for y is 11.
 -/
theorem upper_bound_y (x y : ℝ) (h₁ : 3 < x) (h₂ : x < 6) (h₃ : 6 < y) (h₄ : y < some_number) (h₅ : y - x = 6) : y = 11 := 
by
  sorry

end upper_bound_y_l229_229552


namespace find_integer_a_l229_229664

theorem find_integer_a (a : ℤ) : 
  (∃ p : ℤ[X], x^13 + x + 90 = (x^2 - x + a) * p) → 
  a ∣ 90 → 
  a ∣ 92 → 
  (a + 2) ∣ 88 → 
  a = 2 :=
begin
  sorry
end

end find_integer_a_l229_229664


namespace probability_two_dice_same_face_l229_229044

theorem probability_two_dice_same_face :
  let total_outcomes := 6 ^ 4 in
  let favorable_outcomes := 6 * 6 * 5 * 4 in
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 9 :=
by
  sorry

end probability_two_dice_same_face_l229_229044


namespace min_value_of_sum_squares_on_circle_l229_229095

theorem min_value_of_sum_squares_on_circle :
  ∃ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = 1 ∧ x^2 + y^2 = 6 - 2 * Real.sqrt 5 :=
sorry

end min_value_of_sum_squares_on_circle_l229_229095


namespace monotonicity_l229_229672

noncomputable def f (a x : ℝ) : ℝ := abs (x^2 - a*x) - log x

-- Define the domain
def domain (x : ℝ) : Prop := 0 < x

-- Correctness of monotonicity results:

theorem monotonicity (a : ℝ) : 
  if a ≤ 0 then 
    ∀ x, domain x → x < (a + sqrt (a^2 + 8)) / 4 → 
    f a x ≤ f a ((a + sqrt (a^2 + 8)) / 4) 
    ∧ ∀ x, domain x → x > (a + sqrt (a^2 + 8)) / 4 → 
    f a x ≥ f a ((a + sqrt (a^2 + 8)) / 4)
  else if 0 < a ∧ a < 1 then 
    ∀ x, domain x → a < x ∧ x < (a + sqrt (a^2 + 8)) / 4 → 
    f a x ≥ f a a 
    ∧ ∀ x, domain x → x > (a + sqrt (a^2 + 8)) / 4 → 
    f a x ≥ f a ((a + sqrt (a^2 + 8)) / 4)
  else if 1 ≤ a ∧ a ≤ 2*sqrt 2 then 
    ∀ x, domain x → x < a → 
    f a x ≤ f a a 
    ∧ ∀ x, domain x → x > a → 
    f a x ≥ f a a 
  else if a > 2*sqrt 2 then 
    ∀ x, domain x → x < (a - sqrt (a^2  - 8)) / 4 ∨ (a + sqrt (a^2 - 8)) / 4 < x ∧ x < a → 
    f a x ≤ f a ((a - sqrt (a^2 - 8)) / 4) ∨ 
    ∀ x, domain x → (a - sqrt (a^2 - 8)) / 4 < x ∧ x < (a + sqrt (a^2 - 8)) / 4 → 
    f a x ≥ f a ((a + sqrt (a^2 - 8)) / 4) 
    ∧ ∀ x, domain x → x > a → 
    f a x ≥ f a a 
else 
    false :=
by sorry

end monotonicity_l229_229672


namespace arithmetic_sequence_common_difference_l229_229831

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 = 2)
  (h3 : ∃ r, a 2 = r * a 1 ∧ a 5 = r * a 2) :
  d = 4 :=
sorry

end arithmetic_sequence_common_difference_l229_229831


namespace total_time_is_12_years_l229_229312

noncomputable def total_time_spent (shape_years climb_years_per_summit dive_months cave_years : ℕ) : ℕ :=
  shape_years + (2 * shape_years) + (7 * climb_years_per_summit) / 12 + ((7 * climb_years_per_summit) % 12) / 12 + (dive_months + 12) / 12 + cave_years

theorem total_time_is_12_years :
  total_time_spent 2 5 13 2 = 12 :=
by
  sorry

end total_time_is_12_years_l229_229312


namespace max_profit_l229_229623

noncomputable def revenue (x : ℝ) : ℝ := 
  if (0 < x ∧ x ≤ 10) then 13.5 - (1 / 30) * x^2 
  else if (x > 10) then (168 / x) - (2000 / (3 * x^2)) 
  else 0

noncomputable def cost (x : ℝ) : ℝ := 
  20 + 5.4 * x

noncomputable def profit (x : ℝ) : ℝ := revenue x * x - cost x

theorem max_profit : 
  ∃ (x : ℝ), 0 < x ∧ x ≤ 10 ∧ (profit x = 8.1 * x - (1 / 30) * x^3 - 20) ∧ 
    (∀ (y : ℝ), 0 < y ∧ y ≤ 10 → profit y ≤ profit 9) ∧ 
    ∀ (z : ℝ), z > 10 → profit z ≤ profit 9 :=
by
  sorry

end max_profit_l229_229623


namespace nails_painted_purple_l229_229378

variable (P S : ℕ)

theorem nails_painted_purple :
  (P + 8 + S = 20) ∧ ((8 / 20 : ℚ) * 100 - (S / 20 : ℚ) * 100 = 10) → P = 6 :=
by
  sorry

end nails_painted_purple_l229_229378


namespace polynomial_horner_value_l229_229917

def f (x : ℤ) : ℤ :=
  7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

def horner (x : ℤ) : ℤ :=
  ((((((7 * x + 6) * x + 5) * x + 4) * x + 3) * x + 2) * x + 1)

theorem polynomial_horner_value :
  horner 3 = 262 := by
  sorry

end polynomial_horner_value_l229_229917


namespace inequality_proof_l229_229576

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (b + c)) + (1 / (a + c)) + (1 / (a + b)) ≥ 9 / (2 * (a + b + c)) :=
by
  sorry

end inequality_proof_l229_229576


namespace alice_ride_average_speed_l229_229784

theorem alice_ride_average_speed
    (d1 d2 : ℝ) 
    (s1 s2 : ℝ)
    (h_d1 : d1 = 40)
    (h_d2 : d2 = 20)
    (h_s1 : s1 = 8)
    (h_s2 : s2 = 40) :
    (d1 + d2) / (d1 / s1 + d2 / s2) = 10.909 :=
by
  simp [h_d1, h_d2, h_s1, h_s2]
  norm_num
  sorry

end alice_ride_average_speed_l229_229784


namespace finite_transformation_l229_229666

-- Define the function representing the number transformation
def transform (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n + 5

-- Define the predicate stating that the process terminates
def process_terminates (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ transform^[k] n = 1

-- Lean 4 statement for the theorem
theorem finite_transformation (n : ℕ) (h : n > 1) : process_terminates n ↔ ¬ (∃ m : ℕ, m > 0 ∧ n = 5 * m) :=
by
  sorry

end finite_transformation_l229_229666


namespace back_wheel_revolutions_l229_229252

theorem back_wheel_revolutions
  (front_diameter : ℝ) (back_diameter : ℝ) (front_revolutions : ℝ) (back_revolutions : ℝ)
  (front_diameter_eq : front_diameter = 28)
  (back_diameter_eq : back_diameter = 20)
  (front_revolutions_eq : front_revolutions = 50)
  (distance_eq : ∀ {d₁ d₂}, 2 * Real.pi * d₁ / 2 * front_revolutions = back_revolutions * (2 * Real.pi * d₂ / 2)) :
  back_revolutions = 70 :=
by
  have front_circumference : ℝ := 2 * Real.pi * front_diameter / 2
  have back_circumference : ℝ := 2 * Real.pi * back_diameter / 2
  have total_distance : ℝ := front_circumference * front_revolutions
  have revolutions : ℝ := total_distance / back_circumference 
  sorry

end back_wheel_revolutions_l229_229252


namespace smallest_pos_int_ending_in_9_divisible_by_13_l229_229021

theorem smallest_pos_int_ending_in_9_divisible_by_13 :
  ∃ (n : ℕ), n % 10 = 9 ∧ n % 13 = 0 ∧ ∀ m, m % 10 = 9 ∧ m % 13 = 0 → n ≤ m :=
by
  sorry

end smallest_pos_int_ending_in_9_divisible_by_13_l229_229021


namespace inequality_xyz_l229_229114

theorem inequality_xyz (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) : 
    x + y + 1 / (x * y) ≤ 1 / x + 1 / y + x * y := 
    sorry

end inequality_xyz_l229_229114


namespace intersection_complement_U_l229_229219

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

def B_complement_U : Set ℕ := U \ B

theorem intersection_complement_U (hU : U = {1, 3, 5, 7}) 
                                  (hA : A = {3, 5}) 
                                  (hB : B = {1, 3, 7}) : 
  A ∩ (B_complement_U U B) = {5} := by
  sorry

end intersection_complement_U_l229_229219


namespace find_innings_l229_229585

noncomputable def calculate_innings (A : ℕ) (n : ℕ) : Prop :=
  (n * A + 140 = (n + 1) * (A + 8)) ∧ (A + 8 = 28)

theorem find_innings (n : ℕ) (A : ℕ) :
  calculate_innings A n → n = 14 :=
by
  intros h
  -- Here you would prove that h implies n = 14, but we use sorry to skip the proof steps.
  sorry

end find_innings_l229_229585


namespace ineq_power_sum_lt_pow_two_l229_229257

theorem ineq_power_sum_lt_pow_two (x : ℝ) (n : ℕ) (hx : |x| < 1) (hn : 2 ≤ n) : 
  (1 + x)^n + (1 - x)^n < 2^n :=
by
  sorry

end ineq_power_sum_lt_pow_two_l229_229257


namespace total_time_l229_229314

/-- Define the different time periods in years --/
def getting_in_shape : ℕ := 2
def learning_to_climb : ℕ := 2 * getting_in_shape
def months_climbing : ℕ := 7 * 5
def climbing : ℚ := months_climbing / 12
def break_after_climbing : ℚ := 13 / 12
def diving : ℕ := 2

/-- Prove that the total time taken to achieve all goals is 12 years --/
theorem total_time : getting_in_shape + learning_to_climb + climbing + break_after_climbing + diving = 12 := by
  sorry

end total_time_l229_229314


namespace compare_neg_fractions_l229_229972

theorem compare_neg_fractions : (-3 / 5) < (-1 / 3) := 
by {
  sorry
}

end compare_neg_fractions_l229_229972


namespace complex_fraction_simplification_l229_229512

theorem complex_fraction_simplification (a b c d : ℂ) (h₁ : a = 3 + i) (h₂ : b = 1 + i) (h₃ : c = 1 - i) (h₄ : d = 2 - i) : (a / b) = d := by
  sorry

end complex_fraction_simplification_l229_229512


namespace prove_height_ratio_oil_barrel_l229_229933

noncomputable def height_ratio_oil_barrel (h R: ℝ) : ℝ :=
  let area_horizontal := (1/4 * π * R^2 - 1/2 * R^2);
  let volume_horizontal: ℝ := area_horizontal * h;
  let volume_vertical (x: ℝ) : ℝ := π * R^2 * x;
  volume_horizontal = volume_vertical ((π - 2) / (4 * π) * h)

theorem prove_height_ratio_oil_barrel (h R: ℝ) :
  (∀ a : ℝ, a = height_ratio_oil_barrel h R) → 
  ∃ (ratio: ℝ), ratio = ((1 / 4) - (1 / (2 * π))) :=
begin 
  intro hR,
  use ((1 / 4) - (1 / (2 * π))),
  sorry
end

end prove_height_ratio_oil_barrel_l229_229933


namespace smallest_four_digit_number_in_pascals_triangle_l229_229428

theorem smallest_four_digit_number_in_pascals_triangle :
  (pascal : ℕ → ℕ → ℕ) 
  (h_start : ∀ n, pascal n 0 = 1) 
  (h_increase : ∀ n k, pascal n k ≤ pascal n (k + 1)) : 
  ∃ n k, pascal n k = 1000 := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l229_229428


namespace smallest_square_perimeter_l229_229071

theorem smallest_square_perimeter (P_largest : ℕ) (units_apart : ℕ) (num_squares : ℕ) (H1 : P_largest = 96) (H2 : units_apart = 1) (H3 : num_squares = 8) : 
  ∃ P_smallest : ℕ, P_smallest = 40 := by
  sorry

end smallest_square_perimeter_l229_229071


namespace value_of_x_plus_y_l229_229847

theorem value_of_x_plus_y (x y : ℝ) (h1 : 1 / x + 1 / y = 4) (h2 : 1 / x - 1 / y = -6) : x + y = -4 / 5 :=
sorry

end value_of_x_plus_y_l229_229847


namespace hydrangea_cost_l229_229392

def cost_of_each_plant : ℕ :=
  let total_years := 2021 - 1989
  let total_amount_spent := 640
  total_amount_spent / total_years

theorem hydrangea_cost :
  cost_of_each_plant = 20 :=
by
  -- skipping the proof for Lean statement
  sorry

end hydrangea_cost_l229_229392


namespace beth_sold_l229_229509

theorem beth_sold {initial_coins additional_coins total_coins sold_coins : ℕ} 
  (h_init : initial_coins = 125)
  (h_add : additional_coins = 35)
  (h_total : total_coins = initial_coins + additional_coins)
  (h_sold : sold_coins = total_coins / 2) :
  sold_coins = 80 := 
sorry

end beth_sold_l229_229509


namespace expand_expression_l229_229192

theorem expand_expression : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  intro x
  sorry

end expand_expression_l229_229192


namespace sin_P_equals_one_l229_229093

theorem sin_P_equals_one
  (x y : ℝ) (h1 : (1 / 2) * x * y * Real.sin 1 = 50) (h2 : x * y = 100) :
  Real.sin 1 = 1 :=
by sorry

end sin_P_equals_one_l229_229093


namespace Adeline_hourly_wage_l229_229321

theorem Adeline_hourly_wage
  (hours_per_day : ℕ) 
  (days_per_week : ℕ) 
  (weeks : ℕ) 
  (total_earnings : ℕ) 
  (h1 : hours_per_day = 9) 
  (h2 : days_per_week = 5) 
  (h3 : weeks = 7) 
  (h4 : total_earnings = 3780) :
  total_earnings = 12 * (hours_per_day * days_per_week * weeks) :=
by
  sorry

end Adeline_hourly_wage_l229_229321


namespace impossible_gather_all_coins_in_one_sector_l229_229307

-- Definition of the initial condition with sectors and coins
def initial_coins_in_sectors := [1, 1, 1, 1, 1, 1] -- Each sector has one coin, represented by a list

-- Function to check if all coins are in one sector
def all_coins_in_one_sector (coins : List ℕ) := coins.count 6 == 1

-- Function to make a move (this is a helper; its implementation isn't necessary here but illustrates the idea)
def make_move (coins : List ℕ) (src dst : ℕ) : List ℕ := sorry

-- Proving that after 20 moves, coins cannot be gathered in one sector due to parity constraints
theorem impossible_gather_all_coins_in_one_sector : 
  ¬ ∃ (moves : List (ℕ × ℕ)), moves.length = 20 ∧ all_coins_in_one_sector (List.foldl (λ coins move => make_move coins move.1 move.2) initial_coins_in_sectors moves) :=
sorry

end impossible_gather_all_coins_in_one_sector_l229_229307


namespace rugged_terrain_distance_ratio_l229_229882

theorem rugged_terrain_distance_ratio (D k : ℝ) 
  (hD : D > 0) 
  (hk : k > 0) 
  (v_M v_P : ℝ) 
  (hm : v_M = 2 * k) 
  (hp : v_P = 3 * k)
  (v_Mr v_Pr : ℝ) 
  (hmr : v_Mr = k) 
  (hpr : v_Pr = 3 * k / 2) :
  ∀ (x y a b : ℝ), (x + y = D / 2) → (a + b = D / 2) → (y + b = 2 * D / 3) →
  (x / (2 * k) + y / k = a / (3 * k) + 2 * b / (3 * k)) → 
  (y / b = 1 / 3) := 
sorry

end rugged_terrain_distance_ratio_l229_229882


namespace proof_y_solves_diff_eqn_l229_229582

noncomputable def y (x : ℝ) : ℝ := Real.exp (2 * x)

theorem proof_y_solves_diff_eqn : ∀ x : ℝ, (deriv^[3] y x) - 8 * y x = 0 := by
  sorry

end proof_y_solves_diff_eqn_l229_229582


namespace no_prime_divisible_by_77_l229_229695

def is_prime (n : ℕ) : Prop := 
  (n > 1) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_divisible (a b : ℕ) : Prop := b ∣ a

theorem no_prime_divisible_by_77 : ∀ p : ℕ, is_prime p → is_divisible p 77 → false :=
by
  sorry

end no_prime_divisible_by_77_l229_229695


namespace smallest_number_ending_in_9_divisible_by_13_l229_229029

theorem smallest_number_ending_in_9_divisible_by_13 :
  ∃ (n : ℕ), (n % 10 = 9) ∧ (13 ∣ n) ∧ (∀ (m : ℕ), (m % 10 = 9) ∧ (13 ∣ m) ∧ (m < n) -> false) :=
sorry

end smallest_number_ending_in_9_divisible_by_13_l229_229029


namespace fraction_equivalence_1_algebraic_identity_l229_229259

/-- First Problem: Prove the equivalence of the fractions 171717/252525 and 17/25. -/
theorem fraction_equivalence_1 : 
  (171717 : ℚ) / 252525 = 17 / 25 := 
sorry

/-- Second Problem: Prove the equivalence of the algebraic expressions on both sides. -/
theorem algebraic_identity (a b : ℚ) : 
  2 * b^5 + (a^4 + a^3 * b + a^2 * b^2 + a * b^3 + b^4) * (a - b) = 
  (a^4 - a^3 * b + a^2 * b^2 - a * b^3 + b^4) * (a + b) := 
sorry

end fraction_equivalence_1_algebraic_identity_l229_229259


namespace number_of_mappings_l229_229544

-- Define the sets A and B
def A : Finset ℝ := {a | ∃ i : Fin 100, a = a i}.to_finset
def B : Finset ℝ := {b | ∃ i : Fin 50, b = b i}.to_finset

-- Define the mapping condition
def mapping_condition (f : ℝ → ℝ) : Prop :=
  ∀ a1 a2 ∈ A, a1 ≤ a2 → f a1 ≤ f a2 ∧ ∀ b ∈ B, ∃ a ∈ A, f a = b

-- Prove the number of such mappings
theorem number_of_mappings : ∃ (f : ℝ → ℝ), mapping_condition f →
  ∃ (ways : ℤ), ways = nat.choose 99 49 :=
sorry

end number_of_mappings_l229_229544


namespace function_monotonicity_l229_229588

theorem function_monotonicity (a b : ℝ) : 
  (∀ x, -1 < x ∧ x < 1 → (3 * x^2 + a) < 0) ∧ 
  (∀ x, 1 < x → (3 * x^2 + a) > 0) → 
  (a = -3 ∧ ∃ b : ℝ, true) :=
by {
  sorry
}

end function_monotonicity_l229_229588


namespace base8_subtraction_correct_l229_229325

theorem base8_subtraction_correct :
  ∀ (a b : ℕ) (h1 : a = 7534) (h2 : b = 3267),
      (a - b) % 8 = 4243 % 8 := by
  sorry

end base8_subtraction_correct_l229_229325


namespace number_of_insects_l229_229790

-- Conditions
def total_legs : ℕ := 30
def legs_per_insect : ℕ := 6

-- Theorem statement
theorem number_of_insects (total_legs legs_per_insect : ℕ) : 
  total_legs / legs_per_insect = 5 := 
by
  sorry

end number_of_insects_l229_229790


namespace ShepherdProblem_l229_229103

theorem ShepherdProblem (x y : ℕ) :
  (x + 9 = 2 * (y - 9) ∧ y + 9 = x - 9) ↔
  ((x + 9 = 2 * (y - 9)) ∧ (y + 9 = x - 9)) :=
by
  sorry

end ShepherdProblem_l229_229103


namespace first_three_workers_time_l229_229816

open Real

-- Define a function to represent the total work done by any set of workers
noncomputable def total_work (a b c d : ℝ) (t : ℝ) := (a + b + c + d) * t

-- Conditions
variables (a b c d : ℝ)
variables (h1 : a + b + c + d = 1 / 6)
variables (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
variables (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4)

-- Question: Time for the first three workers to dig the trench
def trench_time : ℝ :=
  (a + b + c + d) * (6 / (a + b + c + d))

theorem first_three_workers_time : trench_time a b c = 6 :=
  sorry

end first_three_workers_time_l229_229816


namespace sum_of_cubes_of_real_roots_eq_11_l229_229292

-- Define the polynomial f(x) = x^3 - 2x^2 - x + 1
def poly (x : ℝ) : ℝ := x^3 - 2 * x^2 - x + 1

-- State that the polynomial has exactly three real roots
axiom three_real_roots : ∃ (x1 x2 x3 : ℝ), poly x1 = 0 ∧ poly x2 = 0 ∧ poly x3 = 0

-- Prove that the sum of the cubes of the real roots is 11
theorem sum_of_cubes_of_real_roots_eq_11 (x1 x2 x3 : ℝ)
  (hx1 : poly x1 = 0) (hx2 : poly x2 = 0) (hx3 : poly x3 = 0) : 
  x1^3 + x2^3 + x3^3 = 11 :=
by
  sorry

end sum_of_cubes_of_real_roots_eq_11_l229_229292


namespace expand_expression_l229_229193

theorem expand_expression : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  intro x
  sorry

end expand_expression_l229_229193


namespace minimum_time_for_five_horses_l229_229009

theorem minimum_time_for_five_horses :
  let horses := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let min_time (hs : Finset ℕ) := hs.fold lcm 1
  let t := Finset.fold min 12 (Finset.filter (λ hs, hs.card ≥ 5) (Finset.powerset horses)).map min_time
  (Finset.card (Finset.filter (λ hs, hs.card ≥ 5 ∧ min_time hs = 12) (Finset.powerset horses)) > 0) ∧
  (t = 12) :=
sorry

end minimum_time_for_five_horses_l229_229009


namespace gauss_family_mean_age_l229_229397

theorem gauss_family_mean_age :
  let ages := [8, 8, 8, 8, 16, 17]
  let num_children := 6
  let sum_ages := 65
  (sum_ages : ℚ) / (num_children : ℚ) = 65 / 6 :=
by
  sorry

end gauss_family_mean_age_l229_229397


namespace trigonometric_identity_l229_229665

theorem trigonometric_identity
  (θ : ℝ)
  (h : Real.tan θ = 1 / 3) :
  Real.sin (3 / 2 * Real.pi + 2 * θ) = -4 / 5 :=
by sorry

end trigonometric_identity_l229_229665


namespace negation_of_universal_statement_l229_229685

theorem negation_of_universal_statement:
  (∀ x : ℝ, x ≥ 2) ↔ ¬ (∃ x : ℝ, x < 2) :=
by {
  sorry
}

end negation_of_universal_statement_l229_229685


namespace part1_part2_l229_229528

variables {p m : ℝ}

-- Definitions for given problem conditions
def parabola_eq (y x : ℝ) (p : ℝ) : Prop := y^2 = 4 * p * x
def line_bc_eq (x y : ℝ) : Prop := 4 * x + y - 20 = 0
def centroid_eq_focus (x₁ x₂ x₃ y₁ y₂ y₃ p : ℝ) : Prop :=
  (x₁ + x₂ + x₃) / 3 = p ∧ (y₁ + y₂ + y₃) / 3 = 0

-- Tangent line and perpendicular condition
def tangent_slope (m : ℝ) : ℝ := 8 / m
def perpendicular_slope (m : ℝ) : ℝ := - m / 8
def angle_lambda_eq_one (m : ℝ) : Prop :=
  let k := tangent_slope m in
  let mn_slope := perpendicular_slope m in
  ∃ λ : ℝ, λ = 1

-- Proof statements
theorem part1 : ∃ (p : ℝ), ∀ (x y : ℝ), 
  centroid_eq_focus x₁ x₂ x₃ y₁ y₂ y₃ p ∧ line_bc_eq x₂ y₂ → 
  parabola_eq y x 4 :=
sorry

theorem part2 : ∃ (λ : ℝ), ∀ (m : ℝ),
  angle_lambda_eq_one m :=
sorry

end part1_part2_l229_229528


namespace multiply_preserve_equiv_l229_229142

noncomputable def conditions_equiv_eqn (N D F : Polynomial ℝ) : Prop :=
  (D = F * (D / F)) ∧ (N.degree ≥ F.degree) ∧ (D ≠ 0)

theorem multiply_preserve_equiv (N D F : Polynomial ℝ) :
  conditions_equiv_eqn N D F →
  (N / D = 0 ↔ (N * F) / (D * F) = 0) :=
by
  sorry

end multiply_preserve_equiv_l229_229142


namespace geom_seq_sixth_term_l229_229587

theorem geom_seq_sixth_term (a : ℝ) (r : ℝ) (h1: a * r^3 = 512) (h2: a * r^8 = 8) : 
  a * r^5 = 128 := 
by 
  sorry

end geom_seq_sixth_term_l229_229587


namespace problem_a_problem_b_l229_229930

-- Problem a
theorem problem_a (p q : ℕ) (h1 : ∃ n : ℤ, 2 * p - q = n^2) (h2 : ∃ m : ℤ, 2 * p + q = m^2) : ∃ k : ℤ, q = 2 * k :=
sorry

-- Problem b
theorem problem_b (m : ℕ) (h1 : ∃ n : ℕ, 2 * m - 4030 = n^2) (h2 : ∃ k : ℕ, 2 * m + 4030 = k^2) : (m = 2593 ∨ m = 12097 ∨ m = 81217 ∨ m = 2030113) :=
sorry

end problem_a_problem_b_l229_229930


namespace tan_alpha_plus_beta_tan_beta_l229_229077

variable (α β : ℝ)

-- Given conditions
def tan_condition_1 : Prop := Real.tan (Real.pi + α) = -1 / 3
def tan_condition_2 : Prop := Real.tan (α + β) = (Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α)

-- Proving the results
theorem tan_alpha_plus_beta (h1 : tan_condition_1 α) (h2 : tan_condition_2 α β) : 
  Real.tan (α + β) = 5 / 16 :=
sorry

theorem tan_beta (h1 : tan_condition_1 α) (h2 : tan_condition_2 α β) :
  Real.tan β = 31 / 43 :=
sorry

end tan_alpha_plus_beta_tan_beta_l229_229077


namespace min_value_of_n_l229_229159

/-!
    Given:
    - There are 53 students.
    - Each student must join one club and can join at most two clubs.
    - There are three clubs: Science, Culture, and Lifestyle.

    Prove:
    The minimum value of n, where n is the maximum number of people who join exactly the same set of clubs, is 9.
-/

def numStudents : ℕ := 53
def numClubs : ℕ := 3
def numSets : ℕ := 6

theorem min_value_of_n : ∃ n : ℕ, n = 9 ∧ 
  ∀ (students clubs sets : ℕ), students = numStudents → clubs = numClubs → sets = numSets →
  (students / sets + if students % sets = 0 then 0 else 1) = 9 :=
by
  sorry -- proof to be filled out

end min_value_of_n_l229_229159


namespace positive_number_y_l229_229937

theorem positive_number_y (y : ℕ) (h1 : y > 0) (h2 : y^2 / 100 = 9) : y = 30 :=
by
  sorry

end positive_number_y_l229_229937


namespace sandwiches_count_l229_229520

def total_sandwiches : ℕ :=
  let meats := 12
  let cheeses := 8
  let condiments := 5
  meats * (Nat.choose cheeses 2) * condiments

theorem sandwiches_count : total_sandwiches = 1680 := by
  sorry

end sandwiches_count_l229_229520


namespace CaitlinAge_l229_229324

theorem CaitlinAge (age_AuntAnna : ℕ) (age_Brianna : ℕ) (age_Caitlin : ℕ)
  (h1 : age_AuntAnna = 42)
  (h2 : age_Brianna = age_AuntAnna / 2)
  (h3 : age_Caitlin = age_Brianna - 5) :
  age_Caitlin = 16 :=
by 
  sorry

end CaitlinAge_l229_229324


namespace integer_triangle_cosines_rational_l229_229122

theorem integer_triangle_cosines_rational (a b c : ℕ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  ∃ (cos_α cos_β cos_γ : ℚ), 
    cos_γ = (a^2 + b^2 - c^2) / (2 * a * b) ∧
    cos_β = (a^2 + c^2 - b^2) / (2 * a * c) ∧
    cos_α = (b^2 + c^2 - a^2) / (2 * b * c) :=
by
  sorry

end integer_triangle_cosines_rational_l229_229122


namespace no_integer_valued_function_l229_229261

theorem no_integer_valued_function (f : ℤ → ℤ) (h : ∀ (m n : ℤ), f (m + f n) = f m - n) : False :=
sorry

end no_integer_valued_function_l229_229261


namespace largest_study_only_Biology_l229_229963

-- Let's define the total number of students
def total_students : ℕ := 500

-- Define the given conditions
def S : ℕ := 65 * total_students / 100
def M : ℕ := 55 * total_students / 100
def B : ℕ := 50 * total_students / 100
def P : ℕ := 15 * total_students / 100

def MS : ℕ := 35 * total_students / 100
def MB : ℕ := 25 * total_students / 100
def BS : ℕ := 20 * total_students / 100
def MSB : ℕ := 10 * total_students / 100

-- Required to prove that the largest number of students who study only Biology is 75
theorem largest_study_only_Biology : 
  (B - MB - BS + MSB) = 75 :=
by 
  sorry

end largest_study_only_Biology_l229_229963


namespace rhombus_longer_diagonal_length_l229_229640

theorem rhombus_longer_diagonal_length (side_length : ℕ) (shorter_diagonal : ℕ) 
  (h_side : side_length = 65) (h_short_diag : shorter_diagonal = 56) : 
  ∃ longer_diagonal : ℕ, longer_diagonal = 118 :=
by
  sorry

end rhombus_longer_diagonal_length_l229_229640


namespace least_beans_l229_229203

-- Define the conditions 
variables (r b : ℕ)

-- State the theorem 
theorem least_beans (h1 : r ≥ 2 * b + 8) (h2 : r ≤ 3 * b) : b ≥ 8 :=
by
  sorry

end least_beans_l229_229203


namespace inverse_proportion_graph_l229_229703

theorem inverse_proportion_graph (k : ℝ) (x : ℝ) (y : ℝ) (h1 : y = k / x) (h2 : (3, -4) ∈ {p : ℝ × ℝ | p.snd = k / p.fst}) :
  k < 0 → ∀ x1 x2 : ℝ, x1 < x2 → y1 = k / x1 → y2 = k / x2 → y1 < y2 := by
  sorry

end inverse_proportion_graph_l229_229703


namespace max_value_of_PQ_l229_229589

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 12)
noncomputable def g (x : ℝ) := Real.sqrt 3 * Real.cos (2 * x - Real.pi / 12)

theorem max_value_of_PQ (t : ℝ) : abs (f t - g t) ≤ 2 :=
by sorry

end max_value_of_PQ_l229_229589


namespace Daniel_correct_answers_l229_229705

theorem Daniel_correct_answers
  (c w : ℕ)
  (h1 : c + w = 12)
  (h2 : 4 * c - 3 * w = 21) :
  c = 9 :=
sorry

end Daniel_correct_answers_l229_229705


namespace find_letter_l229_229732

def consecutive_dates (A B C D E F G : ℕ) : Prop :=
  B = A + 1 ∧ C = A + 2 ∧ D = A + 3 ∧ E = A + 4 ∧ F = A + 5 ∧ G = A + 6

theorem find_letter (A B C D E F G : ℕ) 
  (h_consecutive : consecutive_dates A B C D E F G) 
  (h_condition : ∃ y, (B + y = 2 * A + 6)) :
  y = F :=
by
  sorry

end find_letter_l229_229732


namespace balance_of_three_squares_and_two_heartsuits_l229_229282

-- Definitions
variable {x y z w : ℝ}

-- Given conditions
axiom h1 : 3 * x + 4 * y + z = 12 * w
axiom h2 : x = z + 2 * w

-- Problem to prove
theorem balance_of_three_squares_and_two_heartsuits :
  (3 * y + 2 * z) = (26 / 9) * w :=
sorry

end balance_of_three_squares_and_two_heartsuits_l229_229282


namespace time_for_first_three_workers_l229_229811

def work_rate_equations (a b c d : ℝ) :=
  a + b + c + d = 1 / 6 ∧
  2 * a + (1 / 2) * b + c + d = 1 / 6 ∧
  (1 / 2) * a + 2 * b + c + d = 1 / 4

theorem time_for_first_three_workers (a b c d : ℝ) (h : work_rate_equations a b c d) :
  (a + b + c) * 6 = 1 :=
sorry

end time_for_first_three_workers_l229_229811


namespace find_ratio_of_d1_and_d2_l229_229220

theorem find_ratio_of_d1_and_d2
  (x y d1 d2 : ℝ)
  (h1 : x + 4 * d1 = y)
  (h2 : x + 5 * d2 = y)
  (h3 : d1 ≠ 0)
  (h4 : d2 ≠ 0) :
  d1 / d2 = 5 / 4 := 
by 
  sorry

end find_ratio_of_d1_and_d2_l229_229220


namespace concentration_proof_l229_229942

noncomputable def newConcentration (vol1 vol2 vol3 : ℝ) (perc1 perc2 perc3 : ℝ) (totalVol : ℝ) (finalVol : ℝ) :=
  (vol1 * perc1 + vol2 * perc2 + vol3 * perc3) / finalVol

theorem concentration_proof : 
  newConcentration 2 6 4 0.2 0.55 0.35 (12 : ℝ) (15 : ℝ) = 0.34 := 
by 
  sorry

end concentration_proof_l229_229942


namespace probability_of_5_out_of_6_rolls_odd_l229_229012

theorem probability_of_5_out_of_6_rolls_odd : 
  (nat.choose 6 5 : ℚ) / (2 ^ 6 : ℚ) = 3 / 32 := 
by
  sorry

end probability_of_5_out_of_6_rolls_odd_l229_229012


namespace find_x_plus_y_l229_229349

-- Define the initial assumptions and conditions
variables {x y : ℝ}
axiom geom_sequence : 1 > 0 ∧ x > 0 ∧ y > 0 ∧ 3 > 0 ∧ 1 * x = y
axiom arith_sequence : 2 * y = x + 3

-- Prove that x + y = 15 / 4
theorem find_x_plus_y : x + y = 15 / 4 := sorry

end find_x_plus_y_l229_229349


namespace intersection_of_A_and_B_l229_229086

def A : Set ℝ := { x | -2 < x ∧ x < 2 }
def B : Set ℝ := { x | x ≤ 1 ∨ x ≥ 3 }

theorem intersection_of_A_and_B : 
  (A ∩ B) = { x : ℝ | -2 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_of_A_and_B_l229_229086


namespace simplify_expression_l229_229330

theorem simplify_expression (m : ℝ) (h : m ≠ 0) : (4 * m^2 - 2 * m) / (2 * m) = 2 * m - 1 := by
  sorry

end simplify_expression_l229_229330


namespace distinct_real_numbers_satisfying_system_l229_229518

theorem distinct_real_numbers_satisfying_system :
  ∃! (x y z : ℝ),
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  (x^2 + y^2 = -x + 3 * y + z) ∧
  (y^2 + z^2 = x + 3 * y - z) ∧
  (x^2 + z^2 = 2 * x + 2 * y - z) ∧
  ((x = 0 ∧ y = 1 ∧ z = -2) ∨ (x = -3/2 ∧ y = 5/2 ∧ z = -1/2)) :=
sorry

end distinct_real_numbers_satisfying_system_l229_229518


namespace apples_distribution_l229_229872

theorem apples_distribution (total_apples : ℕ) (given_to_father : ℕ) (total_people : ℕ) : total_apples = 55 → given_to_father = 10 → total_people = 5 → (total_apples - given_to_father) / total_people = 9 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end apples_distribution_l229_229872


namespace complex_number_value_l229_229570

-- Declare the imaginary unit 'i'
noncomputable def i : ℂ := Complex.I

-- Define the problem statement
theorem complex_number_value : (i / ((1 - i) ^ 2)) = -1/2 := 
by
  sorry

end complex_number_value_l229_229570


namespace no_prime_divisible_by_77_l229_229689

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_77 :
  ∀ p : ℕ, is_prime p → ¬ (77 ∣ p) :=
by
  intros p h_prime h_div
  sorry

end no_prime_divisible_by_77_l229_229689


namespace sum_of_segments_eq_twice_side_length_l229_229708

universe u
variables {α : Type u}

-- Define the equilateral triangle and internal point properties
structure Triangle (α : Type*) :=
  (A B C : α) 
  (side_length : ℝ)
  (equilateral : ∀ (P : α), dist A B = side_length ∧
                            dist B C = side_length ∧
                            dist C A = side_length)
                            
-- Define lines parallel through an internal point
structure ParallelSegs (α : Type*) :=
  (D E F G H I P : α)
  (PD_AB : ℝ)
  (PE_AB : ℝ)
  (PF_BC : ℝ)
  (PG_BC : ℝ)
  (PH_CA : ℝ)
  (PI_CA : ℝ)

-- Declare the theorem to prove sum of segments is twice the side length
theorem sum_of_segments_eq_twice_side_length 
  (T : Triangle α) (P : α) 
  (S : ParallelSegs α)
  (h1 : ∀ (k : α), T.equilateral k)
  (h2 : ¬(P = T.A ∨ P = T.B ∨ P = T.C)) 
  : S.PD_AB + S.PE_AB + S.PF_BC + S.PG_BC + S.PH_CA + S.PI_CA = 2 * T.side_length := 
begin
  sorry
end

end sum_of_segments_eq_twice_side_length_l229_229708


namespace time_for_first_three_workers_l229_229810

def work_rate_equations (a b c d : ℝ) :=
  a + b + c + d = 1 / 6 ∧
  2 * a + (1 / 2) * b + c + d = 1 / 6 ∧
  (1 / 2) * a + 2 * b + c + d = 1 / 4

theorem time_for_first_three_workers (a b c d : ℝ) (h : work_rate_equations a b c d) :
  (a + b + c) * 6 = 1 :=
sorry

end time_for_first_three_workers_l229_229810


namespace solve_expression_l229_229618

noncomputable def given_expression : ℝ :=
  (10 * 1.8 - 2 * 1.5) / 0.3 + 3^(2 / 3) - Real.log 4 + Real.sin (Real.pi / 6) - Real.cos (Real.pi / 4) + Nat.factorial 4 / Nat.factorial 2

theorem solve_expression : given_expression = 59.6862 :=
by
  sorry

end solve_expression_l229_229618


namespace gcd_lcm_identity_l229_229594

def GCD (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := a * (b / GCD a b)

theorem gcd_lcm_identity (a b c : ℕ) :
    (LCM a (LCM b c))^2 / (LCM a b * LCM b c * LCM c a) = (GCD a (GCD b c))^2 / (GCD a b * GCD b c * GCD c a) :=
by
  sorry

end gcd_lcm_identity_l229_229594


namespace first_three_workers_dig_time_l229_229820

variables 
  (a b c d : ℝ) -- work rates of the four workers
  (hours : ℝ) -- time to dig the trench

def work_together (a b c d hours : ℝ) := (a + b + c + d) * hours = 1

def scenario1 (a b c d : ℝ) := (2 * a + (1/2) * b + c + d) * 6 = 1

def scenario2 (a b c d : ℝ) := (a/2 + 2 * b + c + d) * 4 = 1

theorem first_three_workers_dig_time
  (h1 : work_together a b c d 6)
  (h2 : scenario1 a b c d)
  (h3 : scenario2 a b c d) :
  hours = 6 := 
sorry

end first_three_workers_dig_time_l229_229820


namespace totalWeightAlF3_is_correct_l229_229604

-- Define the atomic weights of Aluminum and Fluorine
def atomicWeightAl : ℝ := 26.98
def atomicWeightF : ℝ := 19.00

-- Define the number of atoms of Fluorine in Aluminum Fluoride (AlF3)
def numFluorineAtoms : ℕ := 3

-- Define the number of moles of Aluminum Fluoride
def numMolesAlF3 : ℕ := 7

-- Calculate the molecular weight of Aluminum Fluoride (AlF3)
noncomputable def molecularWeightAlF3 : ℝ :=
  atomicWeightAl + (numFluorineAtoms * atomicWeightF)

-- Calculate the total weight of the given moles of AlF3
noncomputable def totalWeight : ℝ :=
  molecularWeightAlF3 * numMolesAlF3

-- Theorem stating the total weight of 7 moles of AlF3
theorem totalWeightAlF3_is_correct : totalWeight = 587.86 := sorry

end totalWeightAlF3_is_correct_l229_229604


namespace intersection_P_Q_l229_229569

def P : Set ℝ := {x | x^2 - 9 < 0}
def Q : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}

theorem intersection_P_Q :
  {x : ℤ | (x : ℝ) ∈ P} ∩ Q = {-1, 0, 1, 2} := 
by
  sorry

end intersection_P_Q_l229_229569


namespace smallest_four_digit_in_pascal_l229_229431

-- Pascal's triangle definition
def pascal (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

-- Condition definition
def exists_in_pascal (m : ℕ) : Prop :=
  ∃ n k, pascal n k = m

theorem smallest_four_digit_in_pascal :
  ∃ n k, pascal n k = 1000 ∧ ∀ m, (exists_in_pascal m) → (m >= 1000) → (m = 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_l229_229431


namespace exists_small_area_triangle_l229_229830

structure Point :=
(x : ℝ)
(y : ℝ)

def is_valid_point (p : Point) : Prop :=
(|p.x| ≤ 2) ∧ (|p.y| ≤ 2)

def no_three_collinear (points : List Point) : Prop :=
  ∀ (p1 p2 p3 : Point), p1 ∈ points → p2 ∈ points → p3 ∈ points →
  (p1 ≠ p2) → (p1 ≠ p3) → (p2 ≠ p3) →
  ((p1.y - p2.y) * (p1.x - p3.x) ≠ (p1.y - p3.y) * (p1.x - p2.x))

noncomputable def triangle_area (p1 p2 p3: Point) : ℝ :=
(abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))) / 2

theorem exists_small_area_triangle (points : List Point)
  (h_valid : ∀ p ∈ points, is_valid_point p)
  (h_no_collinear : no_three_collinear points)
  (h_len : points.length = 6) :
  ∃ (p1 p2 p3: Point), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
  triangle_area p1 p2 p3 ≤ 2 :=
sorry

end exists_small_area_triangle_l229_229830


namespace no_prime_divisible_by_77_l229_229692

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_77 : ∀ p : ℕ, is_prime p → ¬ (77 ∣ p) :=
by
  sorry

end no_prime_divisible_by_77_l229_229692


namespace janna_wrote_more_words_than_yvonne_l229_229923

theorem janna_wrote_more_words_than_yvonne :
  ∃ (janna_words_written yvonne_words_written : ℕ), 
    yvonne_words_written = 400 ∧
    janna_words_written > yvonne_words_written ∧
    ∃ (removed_words added_words : ℕ),
      removed_words = 20 ∧
      added_words = 2 * removed_words ∧
      (janna_words_written + yvonne_words_written - removed_words + added_words + 30 = 1000) ∧
      (janna_words_written - yvonne_words_written = 130) :=
by
  sorry

end janna_wrote_more_words_than_yvonne_l229_229923


namespace min_value_of_sequence_l229_229241

variable (b1 b2 b3 : ℝ)

def geometric_sequence (b1 b2 b3 : ℝ) : Prop :=
  ∃ s : ℝ, b2 = b1 * s ∧ b3 = b1 * s^2 

theorem min_value_of_sequence (h1 : b1 = 2) (h2 : geometric_sequence b1 b2 b3) :
  ∃ s : ℝ, 3 * b2 + 4 * b3 = -9 / 8 :=
sorry

end min_value_of_sequence_l229_229241


namespace length_EQ_l229_229870

-- Define the square EFGH with side length 8
def square_EFGH (a : ℝ) (b : ℝ): Prop := a = 8 ∧ b = 8

-- Define the rectangle IJKL with IL = 12 and JK = 8
def rectangle_IJKL (l : ℝ) (w : ℝ): Prop := l = 12 ∧ w = 8

-- Define the perpendicularity of EH and IJ
def perpendicular_EH_IJ : Prop := true

-- Define the shaded area condition
def shaded_area_condition (area_IJKL : ℝ) (shaded_area : ℝ): Prop :=
  shaded_area = (1/3) * area_IJKL

-- Theorem to prove
theorem length_EQ (a b l w area_IJKL shaded_area EH HG HQ EQ : ℝ):
  square_EFGH a b →
  rectangle_IJKL l w →
  perpendicular_EH_IJ →
  shaded_area_condition area_IJKL shaded_area →
  HQ * HG = shaded_area →
  EQ = EH - HQ →
  EQ = 4 := by
  intros hSquare hRectangle hPerpendicular hShadedArea hHQHG hEQ
  sorry

end length_EQ_l229_229870


namespace solution_set_of_x_abs_x_lt_x_l229_229912

theorem solution_set_of_x_abs_x_lt_x :
  {x : ℝ | x * |x| < x} = {x : ℝ | 0 < x ∧ x < 1} ∪ {x : ℝ | x < -1} :=
by
  sorry

end solution_set_of_x_abs_x_lt_x_l229_229912


namespace positive_integers_congruent_to_2_mod_7_lt_500_count_l229_229687

theorem positive_integers_congruent_to_2_mod_7_lt_500_count : 
  ∃ n : ℕ, n = 72 ∧ ∀ k : ℕ, (k < n → (∃ m : ℕ, (m < 500 ∧ m % 7 = 2) ∧ m = 2 + 7 * k)) := 
by
  sorry

end positive_integers_congruent_to_2_mod_7_lt_500_count_l229_229687


namespace length_of_pipe_is_correct_l229_229780

-- Definitions of the conditions
def step_length : ℝ := 0.8
def steps_same_direction : ℤ := 210
def steps_opposite_direction : ℤ := 100

-- The distance moved by the tractor in one step
noncomputable def tractor_step_distance : ℝ := (steps_same_direction * step_length - steps_opposite_direction * step_length) / (steps_opposite_direction + steps_same_direction : ℝ)

-- The length of the pipe
noncomputable def length_of_pipe (steps_same_direction steps_opposite_direction : ℤ) (step_length : ℝ) : ℝ :=
 steps_same_direction * (step_length - tractor_step_distance)

-- Proof statement
theorem length_of_pipe_is_correct :
  length_of_pipe steps_same_direction steps_opposite_direction step_length = 108 :=
sorry

end length_of_pipe_is_correct_l229_229780


namespace pie_left_in_fridge_l229_229394

theorem pie_left_in_fridge (weight_ate : ℚ) (fraction_ate : ℚ) (total_pie_weight : ℚ) : 
  weight_ate = 240 ∧ fraction_ate = 1 / 6 ∧ total_pie_weight = weight_ate * 6 →
  (5 / 6) * total_pie_weight = 1200 :=
by
  intro h
  obtain ⟨hw, hf, ht⟩ := h
  rw [hw, hf] at ht
  rw ht
  linarith

end pie_left_in_fridge_l229_229394


namespace product_of_geometric_terms_l229_229079

noncomputable def arithmeticSeq (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

noncomputable def geometricSeq (b1 r : ℕ) (n : ℕ) : ℕ :=
  b1 * r^(n - 1)

theorem product_of_geometric_terms :
  ∃ (a1 d b1 r : ℕ),
    (3 * a1 - (arithmeticSeq a1 d 8)^2 + 3 * (arithmeticSeq a1 d 15) = 0) ∧ 
    (arithmeticSeq a1 d 8 = geometricSeq b1 r 10) ∧ 
    (geometricSeq b1 r 3 * geometricSeq b1 r 17 = 36) :=
sorry

end product_of_geometric_terms_l229_229079


namespace present_age_of_son_l229_229610

variable (S M : ℕ)

theorem present_age_of_son :
  (M = S + 30) ∧ (M + 2 = 2 * (S + 2)) → S = 28 :=
by
  sorry

end present_age_of_son_l229_229610


namespace num_numerators_count_l229_229239

open Rat -- open rational numbers namespace

-- Define the set T
def T := {r : ℚ | r > 0 ∧ r < 1 ∧ ∃ d e f : ℕ, d < 10 ∧ e < 10 ∧ f < 10 ∧ r = d / 10 + e / 100 + f / 1000}

-- Define the main problem statement
theorem num_numerators_count : 
  let nums := {def_value : ℕ | def_value < 1000 ∧ gcd def_value 1001 = 1} 
  in nums.card = 400 :=
by
  sorry

end num_numerators_count_l229_229239


namespace cost_per_book_l229_229602

theorem cost_per_book (num_animal_books : ℕ) (num_space_books : ℕ) (num_train_books : ℕ) (total_cost : ℕ) 
                      (h1 : num_animal_books = 10) (h2 : num_space_books = 1) (h3 : num_train_books = 3) (h4 : total_cost = 224) :
  (total_cost / (num_animal_books + num_space_books + num_train_books) = 16) :=
by sorry

end cost_per_book_l229_229602


namespace problem_statement_l229_229117

variable (a b : ℝ)

open Real

noncomputable def inequality_holds (a b : ℝ) : Prop :=
  0 ≤ a ∧ 0 ≤ b ∧ a + b < 2 → (1 / (1 + a^2)) + (1 / (1 + b^2)) ≤ 2 / (1 + a * b)

noncomputable def equality_condition (a b : ℝ) : Prop :=
  0 ≤ a ∧ 0 ≤ b ∧ a + b < 2 → ((1 / (1 + a^2)) + (1 / (1 + b^2)) = 2 / (1 + a * b) ↔ a = b)

theorem problem_statement (a b : ℝ) : inequality_holds a b ∧ equality_condition a b :=
by
  sorry

end problem_statement_l229_229117


namespace saline_drip_duration_l229_229774

theorem saline_drip_duration (rate_drops_per_minute : ℕ) (drop_to_ml_rate : ℕ → ℕ → Prop)
  (ml_received : ℕ) (time_hours : ℕ) :
  rate_drops_per_minute = 20 ->
  drop_to_ml_rate 100 5 ->
  ml_received = 120 ->
  time_hours = 2 :=
by {
  sorry
}

end saline_drip_duration_l229_229774


namespace combined_salaries_A_B_C_D_l229_229135

-- To ensure the whole calculation is noncomputable due to ℝ
noncomputable section

-- Let's define the variables
def salary_E : ℝ := 9000
def average_salary_group : ℝ := 8400
def num_people : ℕ := 5

-- combined salary A + B + C + D represented as a definition
def combined_salaries : ℝ := (average_salary_group * num_people) - salary_E

-- We need to prove that the combined salaries equals 33000
theorem combined_salaries_A_B_C_D : combined_salaries = 33000 := by
  sorry

end combined_salaries_A_B_C_D_l229_229135


namespace correct_calculation_l229_229921

theorem correct_calculation :
  (∀ x : ℤ, x^5 + x^3 ≠ x^8) ∧
  (∀ x : ℤ, x^5 - x^3 ≠ x^2) ∧
  (∀ x : ℤ, x^5 * x^3 = x^8) ∧
  (∀ x : ℤ, (-3 * x)^3 ≠ -9 * x^3) :=
by
  sorry

end correct_calculation_l229_229921


namespace football_championship_min_games_l229_229867

theorem football_championship_min_games :
  (∃ (teams : Finset ℕ) (games : Finset (ℕ × ℕ)),
    teams.card = 20 ∧
    (∀ (a b c : ℕ), a ∈ teams → b ∈ teams → c ∈ teams → a ≠ b → b ≠ c → c ≠ a →
      (a, b) ∈ games ∨ (b, c) ∈ games ∨ (c, a) ∈ games) ∧
    games.card = 90) :=
sorry

end football_championship_min_games_l229_229867


namespace lauren_total_money_made_is_correct_l229_229722

-- Define the rate per commercial view
def rate_per_commercial_view : ℝ := 0.50
-- Define the rate per subscriber
def rate_per_subscriber : ℝ := 1.00
-- Define the number of commercial views on Tuesday
def commercial_views : ℕ := 100
-- Define the number of new subscribers on Tuesday
def subscribers : ℕ := 27
-- Calculate the total money Lauren made on Tuesday
def total_money_made (rate_com_view : ℝ) (rate_sub : ℝ) (com_views : ℕ) (subs : ℕ) : ℝ :=
  (rate_com_view * com_views) + (rate_sub * subs)

-- Theorem stating that the total money Lauren made on Tuesday is $77.00
theorem lauren_total_money_made_is_correct : total_money_made rate_per_commercial_view rate_per_subscriber commercial_views subscribers = 77.00 :=
by
  sorry

end lauren_total_money_made_is_correct_l229_229722


namespace ending_number_l229_229608

theorem ending_number (h : ∃ n, 3 * n = 99 ∧ n = 33) : ∃ m, m = 99 :=
by
  sorry

end ending_number_l229_229608


namespace base9_is_decimal_l229_229795

-- Define the base-9 number and its decimal equivalent
def base9_to_decimal (n : Nat := 85) (base : Nat := 9) : Nat := 8 * base^1 + 5 * base^0

-- Theorem stating the proof problem
theorem base9_is_decimal : base9_to_decimal 85 9 = 77 := by
  unfold base9_to_decimal
  simp
  sorry

end base9_is_decimal_l229_229795


namespace range_of_quadratic_function_l229_229134

theorem range_of_quadratic_function : 
  ∀ x : ℝ, ∃ y : ℝ, y = x^2 - 1 :=
by
  sorry

end range_of_quadratic_function_l229_229134


namespace least_positive_integer_lemma_l229_229075

theorem least_positive_integer_lemma :
  ∃ x : ℕ, x > 0 ∧ x + 7237 ≡ 5017 [MOD 12] ∧ (∀ y : ℕ, y > 0 ∧ y + 7237 ≡ 5017 [MOD 12] → x ≤ y) :=
by
  sorry

end least_positive_integer_lemma_l229_229075


namespace day_in_43_days_is_wednesday_l229_229140

-- Define a function to represent the day of the week after a certain number of days
def day_of_week (n : ℕ) : ℕ := n % 7

-- Use an enum or some notation to represent the days of the week, but this is implicit in our setup.
-- We assume the days are numbered from 0 to 6 with 0 representing Tuesday.
def Tuesday : ℕ := 0
def Wednesday : ℕ := 1

-- Theorem to prove that 43 days after Tuesday is a Wednesday
theorem day_in_43_days_is_wednesday : day_of_week (Tuesday + 43) = Wednesday :=
by
  sorry

end day_in_43_days_is_wednesday_l229_229140


namespace diophantine_eq_solutions_count_l229_229516

theorem diophantine_eq_solutions_count :
  let eq := (∃ x y : ℤ, 2 * x + 3 * y = 780 ∧ x > 0 ∧ y > 0)
  in (∃ count : ℕ, count = 130) :=
begin
  -- Problem statement and definition based on given conditions
  let eq := (∃ x y : ℤ, 2 * x + 3 * y = 780 ∧ x > 0 ∧ y > 0),

  -- Providing the conclusion that there exist 130 solutions
  have count := 130,
  
  -- Existential quantifier asserting the number of positive integer solutions 
  existsi count,
  
  -- Statement of equality confirming the number
  exact rfl,
end

end diophantine_eq_solutions_count_l229_229516


namespace minValueExpr_ge_9_l229_229243

noncomputable def minValueExpr (x y z : ℝ) : ℝ :=
  (x + y) / z + (x + z) / y + (y + z) / x + 3

theorem minValueExpr_ge_9 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  minValueExpr x y z ≥ 9 :=
by sorry

end minValueExpr_ge_9_l229_229243


namespace smallest_four_digit_number_in_pascals_triangle_l229_229425

theorem smallest_four_digit_number_in_pascals_triangle :
  (pascal : ℕ → ℕ → ℕ) 
  (h_start : ∀ n, pascal n 0 = 1) 
  (h_increase : ∀ n k, pascal n k ≤ pascal n (k + 1)) : 
  ∃ n k, pascal n k = 1000 := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l229_229425


namespace evaluate_expression_correct_l229_229802

noncomputable def evaluate_expression : ℤ :=
  6 - 8 * (9 - 4 ^ 2) * 5 + 2

theorem evaluate_expression_correct : evaluate_expression = 288 := by
  sorry

end evaluate_expression_correct_l229_229802


namespace proof_theorem_l229_229700

noncomputable def proof_problem : Prop :=
  let a := 6
  let b := 15
  let c := 7
  let lhs := a * b * c
  let rhs := (Real.sqrt ((a^2) + (2 * a) + (b^3) - (b^2) + (3 * b))) / (c^2 + c + 1) + 629.001
  lhs = rhs

theorem proof_theorem : proof_problem :=
  by
  sorry

end proof_theorem_l229_229700


namespace david_overall_average_l229_229656

open Real

noncomputable def english_weighted_average := (74 * 0.20) + (80 * 0.25) + (77 * 0.55)
noncomputable def english_modified := english_weighted_average * 1.5

noncomputable def math_weighted_average := (65 * 0.15) + (75 * 0.25) + (90 * 0.60)
noncomputable def math_modified := math_weighted_average * 2.0

noncomputable def physics_weighted_average := (82 * 0.40) + (85 * 0.60)
noncomputable def physics_modified := physics_weighted_average * 1.2

noncomputable def chemistry_weighted_average := (67 * 0.35) + (89 * 0.65)
noncomputable def chemistry_modified := chemistry_weighted_average * 1.0

noncomputable def biology_weighted_average := (90 * 0.30) + (95 * 0.70)
noncomputable def biology_modified := biology_weighted_average * 1.5

noncomputable def overall_average := (english_modified + math_modified + physics_modified + chemistry_modified + biology_modified) / 5

theorem david_overall_average :
  overall_average = 120.567 :=
by
  -- Proof to be filled in
  sorry

end david_overall_average_l229_229656


namespace red_chips_probability_l229_229098

-- Define the setup and the probability calculation in Lean 4.
theorem red_chips_probability {r g : Nat} (hr : r = 4) (hg : g = 3) :
  let totalChips := r + g
  let totalArrangements := Nat.choose totalChips g
  let favorableArrangements := Nat.choose (r + g - 1) g in
  totalArrangements = 35 →
  favorableArrangements = 20 →
  (favorableArrangements : ℚ) / totalArrangements = 4 / 7 :=
by
  simp [hr, hg, Nat.choose]
  intros
  exact sorry

end red_chips_probability_l229_229098


namespace chairs_left_after_selling_l229_229935

-- Definitions based on conditions
def chairs_before_selling : ℕ := 15
def difference_after_selling : ℕ := 12

-- Theorem statement based on the question
theorem chairs_left_after_selling : (chairs_before_selling - 3 = difference_after_selling) → (chairs_before_selling - difference_after_selling = 3) := by
  intro h
  sorry

end chairs_left_after_selling_l229_229935


namespace isosceles_trapezoid_larger_base_l229_229272

theorem isosceles_trapezoid_larger_base (AD BC AC : ℝ) (h1 : AD = 10) (h2 : BC = 6) (h3 : AC = 14) :
  ∃ (AB : ℝ), AB = 16 :=
by
  sorry

end isosceles_trapezoid_larger_base_l229_229272


namespace trench_dig_time_l229_229824

theorem trench_dig_time (a b c d : ℝ) (h1 : a + b + c + d = 1/6)
  (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
  (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4) :
  a + b + c = 1 / 6 := sorry

end trench_dig_time_l229_229824


namespace condition_neither_sufficient_nor_necessary_l229_229929
-- Import necessary library

-- Define the function and conditions
def f (x a : ℝ) : ℝ := x^2 + a * x + 1

-- State the proof problem
theorem condition_neither_sufficient_nor_necessary :
  ∀ a : ℝ, (∀ x : ℝ, f x a = 0 -> x = 1/2) ↔ a^2 - 4 = 0 ∧ a ≤ -2 := sorry

end condition_neither_sufficient_nor_necessary_l229_229929


namespace no_solution_for_p_eq_7_l229_229799

theorem no_solution_for_p_eq_7 : ∀ x : ℝ, x ≠ 4 → x ≠ 8 → ( (x-3)/(x-4) = (x-7)/(x-8) ) → false := by
  intro x h1 h2 h
  sorry

end no_solution_for_p_eq_7_l229_229799


namespace units_digit_17_pow_2023_l229_229032

theorem units_digit_17_pow_2023 : (17^2023 % 10) = 3 := sorry

end units_digit_17_pow_2023_l229_229032


namespace ratio_of_juniors_to_seniors_l229_229175

theorem ratio_of_juniors_to_seniors (j s : ℕ) (h : (1 / 3) * j = (2 / 3) * s) : j / s = 2 :=
by
  sorry

end ratio_of_juniors_to_seniors_l229_229175


namespace rectangle_shorter_side_length_l229_229064

theorem rectangle_shorter_side_length (rope_length : ℕ) (long_side : ℕ) : 
  rope_length = 100 → long_side = 28 → 
  ∃ short_side : ℕ, (2 * long_side + 2 * short_side = rope_length) ∧ short_side = 22 :=
by
  sorry

end rectangle_shorter_side_length_l229_229064


namespace teacher_a_probability_correct_total_assessments_three_correct_l229_229580

noncomputable def teacher_a_pass_probability : ℚ :=
  let p_a : ℚ := 2/3 in
  p_a + (1 - p_a) * p_a

theorem teacher_a_probability_correct : teacher_a_pass_probability = 8/9 :=
by
  -- Proof will be filled in later
  sorry

noncomputable def total_assessments_three_probability : ℚ :=
  let p_a : ℚ := 2/3 in
  let p_b : ℚ := 1/2 in
  (p_a * (1 - p_b)) + ((1 - p_a) * p_b)

theorem total_assessments_three_correct : total_assessments_three_probability = 1/2 :=
by
  -- Proof will be filled in later
  sorry

end teacher_a_probability_correct_total_assessments_three_correct_l229_229580


namespace find_larger_number_l229_229040

theorem find_larger_number (L S : ℤ) (h1 : L - S = 1365) (h2 : L = 6 * S + 5) : L = 1637 :=
by
  sorry

end find_larger_number_l229_229040


namespace tom_initial_amount_l229_229916

variables (t s j : ℝ)

theorem tom_initial_amount :
  t + s + j = 1200 →
  t - 200 + 3 * s + 2 * j = 1800 →
  t = 400 :=
by
  intros h1 h2
  sorry

end tom_initial_amount_l229_229916


namespace general_term_of_arithmetic_sequence_l229_229994

theorem general_term_of_arithmetic_sequence
  (a : ℕ → ℤ)
  (h_arith : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a3 : a 3 = -2)
  (h_a7 : a 7 = -10) :
  ∀ n : ℕ, a n = 4 - 2 * n :=
sorry

end general_term_of_arithmetic_sequence_l229_229994


namespace sequence_all_perfect_squares_l229_229260

theorem sequence_all_perfect_squares (n : ℕ) : 
  ∃ k : ℕ, (∃ m : ℕ, 2 * 10^n + 1 = 3 * m) ∧ (x_n = (m^2 / 9)) :=
by
  sorry

end sequence_all_perfect_squares_l229_229260


namespace solve_for_r_l229_229803

theorem solve_for_r (r : ℚ) (h : 4 * (r - 10) = 3 * (3 - 3 * r) + 9) : r = 58 / 13 :=
by
  sorry

end solve_for_r_l229_229803


namespace no_prime_divisible_by_77_l229_229696

def is_prime (n : ℕ) : Prop := 
  (n > 1) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_divisible (a b : ℕ) : Prop := b ∣ a

theorem no_prime_divisible_by_77 : ∀ p : ℕ, is_prime p → is_divisible p 77 → false :=
by
  sorry

end no_prime_divisible_by_77_l229_229696


namespace finite_fraction_n_iff_l229_229617

theorem finite_fraction_n_iff (n : ℕ) (h_pos : 0 < n) :
  (∃ (a b : ℕ), n * (n + 1) = 2^a * 5^b) ↔ (n = 1 ∨ n = 4) :=
by
  sorry

end finite_fraction_n_iff_l229_229617


namespace fermat_prime_solution_unique_l229_229073

def is_fermat_prime (p : ℕ) : Prop :=
  ∃ r : ℕ, p = 2^(2^r) + 1

def problem_statement (p n k : ℕ) : Prop :=
  is_fermat_prime p ∧ p^n + n = (n + 1)^k

theorem fermat_prime_solution_unique (p n k : ℕ) :
  problem_statement p n k → (p, n, k) = (3, 1, 2) ∨ (p, n, k) = (5, 2, 3) :=
by
  sorry

end fermat_prime_solution_unique_l229_229073


namespace each_person_gets_9_apples_l229_229873

-- Define the initial number of apples and the number of apples given to Jack's father
def initial_apples : ℕ := 55
def apples_given_to_father : ℕ := 10

-- Define the remaining apples after giving to Jack's father
def remaining_apples : ℕ := initial_apples - apples_given_to_father

-- Define the number of people sharing the remaining apples
def number_of_people : ℕ := 1 + 4

-- Define the number of apples each person will get
def apples_per_person : ℕ := remaining_apples / number_of_people

-- Prove that each person gets 9 apples
theorem each_person_gets_9_apples (h₁ : initial_apples = 55) 
                                  (h₂ : apples_given_to_father = 10) 
                                  (h₃ : number_of_people = 5) 
                                  (h₄ : remaining_apples = initial_apples - apples_given_to_father) 
                                  (h₅ : apples_per_person = remaining_apples / number_of_people) : 
  apples_per_person = 9 :=
by sorry

end each_person_gets_9_apples_l229_229873


namespace larger_integer_value_l229_229277

theorem larger_integer_value
  (a b : ℕ)
  (h1 : a ≥ b)
  (h2 : ↑a / ↑b = 7 / 3)
  (h3 : a * b = 294) :
  a = 7 * Int.sqrt 14 := 
sorry

end larger_integer_value_l229_229277


namespace solve_cryptarithm_l229_229126

-- Declare non-computable constants for the letters
variables {A B C : ℕ}

-- Conditions from the problem
-- Different letters represent different digits
axiom diff_digits : A ≠ B ∧ B ≠ C ∧ C ≠ A

-- A ≠ 0
axiom A_nonzero : A ≠ 0

-- Given cryptarithm equation
axiom cryptarithm_eq : 100 * C + 10 * B + A + 100 * A + 10 * A + A = 100 * B + A

-- The proof to show the correct values
theorem solve_cryptarithm : A = 5 ∧ B = 9 ∧ C = 3 :=
sorry

end solve_cryptarithm_l229_229126


namespace constant_term_of_expansion_l229_229143

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the expansion term
def expansion_term (n k : ℕ) (a b : ℚ) : ℚ :=
  (binom n k) * (a ^ k) * (b ^ (n - k))

-- Define the specific example
def specific_expansion_term : ℚ :=
  expansion_term 8 4 3 (2 : ℚ)

theorem constant_term_of_expansion : specific_expansion_term = 90720 :=
by
  -- The proof is omitted
  sorry

end constant_term_of_expansion_l229_229143


namespace find_f_l229_229543

theorem find_f (f : ℝ → ℝ) (h : ∀ x : ℝ, f (Real.sqrt x + 4) = x + 8 * Real.sqrt x) :
  ∀ (x : ℝ), x ≥ 4 → f x = x^2 - 16 :=
by
  sorry

end find_f_l229_229543


namespace cost_of_pure_milk_l229_229163

theorem cost_of_pure_milk (C : ℝ) (total_milk : ℝ) (pure_milk : ℝ) (water : ℝ) (profit : ℝ) :
  total_milk = pure_milk + water → profit = (total_milk * C) - (pure_milk * C) → profit = 35 → C = 7 :=
by
  intros h1 h2 h3
  sorry

end cost_of_pure_milk_l229_229163


namespace quadratic_inequality_solution_l229_229853

theorem quadratic_inequality_solution (a b: ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 + a * x + b < 0) ∧
  (2 + 3 = -a) ∧
  (2 * 3 = b) →
  ∀ x : ℝ, (b * x^2 + a * x + 1 > 0) ↔ (x < 1/3 ∨ x > 1/2) :=
by
  sorry

end quadratic_inequality_solution_l229_229853


namespace quadratic_completion_l229_229661

theorem quadratic_completion (x : ℝ) : 
  (2 * x^2 + 3 * x - 1) = 2 * (x + 3 / 4)^2 - 17 / 8 := 
by 
  -- Proof isn't required, we just state the theorem.
  sorry

end quadratic_completion_l229_229661


namespace product_of_consecutive_multiples_of_4_divisible_by_768_l229_229146

theorem product_of_consecutive_multiples_of_4_divisible_by_768 (n : ℤ) :
  (4 * n) * (4 * (n + 1)) * (4 * (n + 2)) % 768 = 0 :=
by
  sorry

end product_of_consecutive_multiples_of_4_divisible_by_768_l229_229146


namespace roots_absolute_value_l229_229091

noncomputable def quadratic_roots_property (p : ℝ) (r1 r2 : ℝ) : Prop :=
  r1 ≠ r2 ∧
  r1 + r2 = -p ∧
  r1 * r2 = 16 ∧
  ∃ r : ℝ, r = r1 ∨ r = r2 ∧ abs r > 4

theorem roots_absolute_value (p : ℝ) (r1 r2 : ℝ) :
  quadratic_roots_property p r1 r2 → ∃ r : ℝ, (r = r1 ∨ r = r2) ∧ abs r > 4 :=
sorry

end roots_absolute_value_l229_229091


namespace larger_integer_is_7sqrt14_l229_229274

theorem larger_integer_is_7sqrt14 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a / b = 7 / 3) (h2 : a * b = 294) : max a b = 7 * Real.sqrt 14 :=
by 
  sorry

end larger_integer_is_7sqrt14_l229_229274


namespace mail_total_correct_l229_229721

def Monday_mail : ℕ := 65
def Tuesday_mail : ℕ := Monday_mail + 10
def Wednesday_mail : ℕ := Tuesday_mail - 5
def Thursday_mail : ℕ := Wednesday_mail + 15
def total_mail : ℕ := Monday_mail + Tuesday_mail + Wednesday_mail + Thursday_mail

theorem mail_total_correct : total_mail = 295 := by
  sorry

end mail_total_correct_l229_229721


namespace dice_sum_four_l229_229293

def possible_outcomes (x : Nat) : Set (Nat × Nat) :=
  { (d1, d2) | d1 + d2 = x ∧ 1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 }

theorem dice_sum_four :
  possible_outcomes 4 = {(3, 1), (1, 3), (2, 2)} :=
by
  sorry -- We acknowledge that this outline is equivalent to the provided math problem.

end dice_sum_four_l229_229293


namespace range_f_l229_229186

noncomputable def f (x : ℝ) : ℝ := 1 / (2 - x) ^ 3

theorem range_f : Set.range f = Set.Ioi 0 ∪ Set.Iio 0 := by
  sorry

end range_f_l229_229186


namespace cube_root_abs_diff_root_l229_229070

theorem cube_root_abs_diff_root :
  let roots := { x : ℝ | x^3 - 3 * x^2 - x + 3 = 0 }
  ∃ largest smallest, largest ∈ roots ∧ smallest ∈ roots ∧ largest >= smallest ∧ 
  real.cbrt (| largest - smallest |) = real.cbrt 4 := 
by
  let roots := { x : ℝ | x^3 - 3 * x^2 - x + 3 = 0 }
  have h_roots : roots = { -1, 1, 3 } := sorry
  let largest := 3
  let smallest := -1
  have h1 : largest >= smallest := sorry
  have h2 : largest ∈ roots := sorry
  have h3 : smallest ∈ roots := sorry
  have h5: | largest - smallest | = 4 := sorry
  show ∃ largest smallest, largest ∈ roots ∧ smallest ∈ roots ∧ largest >= smallest ∧ real.cbrt (| largest - smallest |) = real.cbrt 4, from
  ⟨largest, smallest, h2, h3, h1, by simp [h5]⟩

end cube_root_abs_diff_root_l229_229070


namespace sum_of_numbers_l229_229910

theorem sum_of_numbers (x : ℝ) (h1 : x^2 + (2 * x)^2 + (4 * x)^2 = 1701) : x + 2 * x + 4 * x = 63 :=
sorry

end sum_of_numbers_l229_229910


namespace correct_bushes_needed_l229_229185

def yield_per_bush := 10
def containers_per_zucchini := 3
def zucchinis_needed := 36
def bushes_needed (yield_per_bush containers_per_zucchini zucchinis_needed : ℕ) : ℕ :=
  Nat.ceil ((zucchinis_needed * containers_per_zucchini : ℕ) / yield_per_bush)

theorem correct_bushes_needed : bushes_needed yield_per_bush containers_per_zucchini zucchinis_needed = 11 := 
by
  sorry

end correct_bushes_needed_l229_229185


namespace find_n_l229_229915

-- Define the two numbers a and b
def a : ℕ := 4665
def b : ℕ := 6905

-- Calculate their difference
def diff : ℕ := b - a

-- Define that n is the greatest common divisor of the difference
def n : ℕ := Nat.gcd (b - a) 0 -- 0 here because b - a is non-zero and gcd(n, 0) = n

-- Define a function to calculate the sum of the digits of a number
def digit_sum (x : ℕ) : ℕ :=
  (Nat.digits 10 x).sum

-- Define our required properties
def has_property : Prop :=
  Nat.gcd (b - a) 0 = n ∧ digit_sum n = 4

-- Finally, state the theorem we need to prove
theorem find_n (h : has_property) : n = 1120 :=
sorry

end find_n_l229_229915


namespace magnitude_of_parallel_vector_l229_229546

theorem magnitude_of_parallel_vector {x : ℝ} 
  (h_parallel : 2 / x = -1 / 3) : 
  (Real.sqrt (x^2 + 3^2)) = 3 * Real.sqrt 5 := 
sorry

end magnitude_of_parallel_vector_l229_229546


namespace find_radius_l229_229768

-- Definitions based on conditions
def circle_radius (r : ℝ) : Prop := r = 2

-- Specification based on the question and conditions
theorem find_radius (r : ℝ) : circle_radius r :=
by
  -- Skip the proof
  sorry

end find_radius_l229_229768


namespace smallest_four_digit_in_pascal_triangle_l229_229473

noncomputable section

def is_in_pascal_triangle (n : ℕ) : Prop :=
  ∃ r c : ℕ, c ≤ r ∧ binomial r c = n

theorem smallest_four_digit_in_pascal_triangle :
  ¬ ∃ n : ℕ, is_in_pascal_triangle n ∧ n < 1000 ∧ (n ≥ 1000 ∨ n < 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_triangle_l229_229473


namespace part1_subsets_m_0_part2_range_m_l229_229350

namespace MathProof

variables {α : Type*} {m : ℝ}

def A := {x : ℝ | x^2 + 5 * x - 6 = 0}
def B (m : ℝ) := {x : ℝ | x^2 + 2 * (m + 1) * x + m^2 - 3 = 0}
def subsets (A : Set ℝ) := {s : Set ℝ | s ⊆ A}

theorem part1_subsets_m_0 :
  subsets (A ∪ B 0) = {∅, {-6}, {1}, {-3}, {-6,1}, {-6,-3}, {1,-3}, {-6,1,-3}} :=
sorry

theorem part2_range_m (h : ∀ x, x ∈ B m → x ∈ A) : m ≤ -2 :=
sorry

end MathProof

end part1_subsets_m_0_part2_range_m_l229_229350


namespace correct_limiting_reagent_and_yield_l229_229988

noncomputable def balanced_reaction_theoretical_yield : Prop :=
  let Fe2O3_initial : ℕ := 4
  let CaCO3_initial : ℕ := 10
  let moles_Fe2O3_needed_for_CaCO3 := Fe2O3_initial * (6 / 2)
  let limiting_reagent := if CaCO3_initial < moles_Fe2O3_needed_for_CaCO3 then true else false
  let theoretical_yield := (CaCO3_initial * (3 / 6))
  limiting_reagent = true ∧ theoretical_yield = 5

theorem correct_limiting_reagent_and_yield : balanced_reaction_theoretical_yield :=
by
  sorry

end correct_limiting_reagent_and_yield_l229_229988


namespace find_remainder_l229_229200

theorem find_remainder (G : ℕ) (Q1 Q2 R1 : ℕ) (hG : G = 127) (h1 : 1661 = G * Q1 + R1) (h2 : 2045 = G * Q2 + 13) : R1 = 10 :=
by
  sorry

end find_remainder_l229_229200


namespace Robert_diff_C_l229_229572

/- Define the conditions as hypotheses -/
variables (C : ℕ) -- Assuming the number of photos Claire has taken as a natural number.

-- Lisa has taken 3 times as many photos as Claire.
def Lisa_photos := 3 * C

-- Robert has taken the same number of photos as Lisa.
def Robert_photos := Lisa_photos C -- which will be 3 * C

-- Proof of the difference.
theorem Robert_diff_C : (Robert_photos C) - C = 2 * C :=
by
  sorry

end Robert_diff_C_l229_229572


namespace min_value_of_inverse_sum_l229_229991

theorem min_value_of_inverse_sum {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : (1/x) + (1/y) ≥ 4 :=
by
  sorry

end min_value_of_inverse_sum_l229_229991


namespace volume_of_stone_l229_229310

theorem volume_of_stone 
  (width length initial_height final_height : ℕ)
  (h_width : width = 15)
  (h_length : length = 20)
  (h_initial_height : initial_height = 10)
  (h_final_height : final_height = 15)
  : (width * length * final_height - width * length * initial_height = 1500) :=
by
  sorry

end volume_of_stone_l229_229310


namespace probability_disco_music_two_cassettes_returned_probability_disco_music_two_cassettes_not_returned_l229_229918

noncomputable def total_cassettes : ℕ := 30
noncomputable def disco_cassettes : ℕ := 12
noncomputable def classical_cassettes : ℕ := 18

-- Part (a): DJ returns the first cassette before taking the second one
theorem probability_disco_music_two_cassettes_returned :
  (disco_cassettes / total_cassettes) * (disco_cassettes / total_cassettes) = 4 / 25 :=
by
  sorry

-- Part (b): DJ does not return the first cassette before taking the second one
theorem probability_disco_music_two_cassettes_not_returned :
  (disco_cassettes / total_cassettes) * ((disco_cassettes - 1) / (total_cassettes - 1)) = 22 / 145 :=
by
  sorry

end probability_disco_music_two_cassettes_returned_probability_disco_music_two_cassettes_not_returned_l229_229918


namespace inverse_variation_l229_229743

theorem inverse_variation (w : ℝ) (h1 : ∃ (c : ℝ), ∀ (x : ℝ), x^4 * w^(1/4) = c)
  (h2 : (3 : ℝ)^4 * (16 : ℝ)^(1/4) = (6 : ℝ)^4 * w^(1/4)) : 
  w = 1 / 4096 :=
by
  sorry

end inverse_variation_l229_229743


namespace flea_can_visit_all_points_l229_229048

def flea_maximum_jump (max_point : ℕ) : ℕ :=
  1006

theorem flea_can_visit_all_points (n : ℕ) (max_point : ℕ) (h_nonneg_max_point : 0 ≤ max_point) (h_segment : max_point = 2013) :
  n ≤ flea_maximum_jump max_point :=
by
  sorry

end flea_can_visit_all_points_l229_229048


namespace sum_of_angles_equal_360_l229_229709

variables (A B C D F G : ℝ)

-- Given conditions.
def is_quadrilateral_interior_sum (A B C D : ℝ) : Prop := A + B + C + D = 360
def split_internal_angles (F G : ℝ) (C D : ℝ) : Prop := F + G = C + D

-- Proof problem statement.
theorem sum_of_angles_equal_360
  (h1 : is_quadrilateral_interior_sum A B C D)
  (h2 : split_internal_angles F G C D) :
  A + B + C + D + F + G = 360 :=
sorry

end sum_of_angles_equal_360_l229_229709


namespace no_rational_solution_l229_229995

theorem no_rational_solution 
  (a b c : ℤ) 
  (ha : a % 2 = 1) 
  (hb : b % 2 = 1) 
  (hc : c % 2 = 1) : 
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 :=
by sorry

end no_rational_solution_l229_229995


namespace units_digit_27_mul_46_l229_229984

-- Define the function to calculate the units digit of a number
def units_digit (n : ℕ) : ℕ :=
  n % 10

-- Problem statement: The units digit of 27 * 46 is 2
theorem units_digit_27_mul_46 : units_digit (27 * 46) = 2 :=
  sorry

end units_digit_27_mul_46_l229_229984


namespace findPositiveRealSolutions_l229_229807

noncomputable def onlySolutions (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  (a^2 - b * d) / (b + 2 * c + d) +
  (b^2 - c * a) / (c + 2 * d + a) +
  (c^2 - d * b) / (d + 2 * a + b) +
  (d^2 - a * c) / (a + 2 * b + c) = 0

theorem findPositiveRealSolutions :
  ∀ a b c d : ℝ,
  onlySolutions a b c d →
  ∃ k m : ℝ, k > 0 ∧ m > 0 ∧ a = k ∧ b = m ∧ c = k ∧ d = m :=
by
  intros a b c d h
  -- proof steps (if required) go here
  sorry

end findPositiveRealSolutions_l229_229807


namespace final_statue_weight_l229_229501

-- Define the initial weight of the statue
def initial_weight : ℝ := 250

-- Define the percentage of weight remaining after each week
def remaining_after_week1 (w : ℝ) : ℝ := 0.70 * w
def remaining_after_week2 (w : ℝ) : ℝ := 0.80 * w
def remaining_after_week3 (w : ℝ) : ℝ := 0.75 * w

-- Define the final weight of the statue after three weeks
def final_weight : ℝ := 
  remaining_after_week3 (remaining_after_week2 (remaining_after_week1 initial_weight))

-- Prove the weight of the final statue is 105 kg
theorem final_statue_weight : final_weight = 105 := 
  by
    sorry

end final_statue_weight_l229_229501


namespace set_difference_correct_l229_229353

-- Define the sets A and B
def A : Set ℤ := {-1, 1, 3, 5, 7, 9}
def B : Set ℤ := {-1, 5, 7}

-- Define the set difference A - B
def A_minus_B : Set ℤ := {x | x ∈ A ∧ x ∉ B} -- This is the operation A - B

-- The theorem stating the required proof
theorem set_difference_correct : A_minus_B = {1, 3, 9} :=
by {
  -- Proof goes here; however, we have requested no proof, so we put sorry.
  sorry
}

end set_difference_correct_l229_229353


namespace total_pikes_l229_229577

theorem total_pikes (x : ℝ) (h : x = 4 + (1/2) * x) : x = 8 :=
sorry

end total_pikes_l229_229577


namespace distinct_weights_count_l229_229756

theorem distinct_weights_count (n : ℕ) (h : n = 4) : 
  -- Given four weights and a two-pan balance scale without a pointer,
  ∃ m : ℕ, 
  -- prove that the number of distinct weights of cargo
  (m = 40) ∧  
  -- that can be exactly measured if the weights can be placed on both pans of the scale is 40.
  m = 3^n - 1 ∧ (m / 2 = 40) := by
  sorry

end distinct_weights_count_l229_229756


namespace difference_largest_smallest_l229_229597

def num1 : ℕ := 10
def num2 : ℕ := 11
def num3 : ℕ := 12

theorem difference_largest_smallest :
  (max num1 (max num2 num3)) - (min num1 (min num2 num3)) = 2 :=
by
  -- Proof can be filled here
  sorry

end difference_largest_smallest_l229_229597


namespace total_mail_l229_229716

def monday_mail := 65
def tuesday_mail := monday_mail + 10
def wednesday_mail := tuesday_mail - 5
def thursday_mail := wednesday_mail + 15

theorem total_mail : 
  monday_mail + tuesday_mail + wednesday_mail + thursday_mail = 295 := by
  sorry

end total_mail_l229_229716


namespace pointC_on_same_side_as_point1_l229_229169

-- Definitions of points and the line equation
def is_on_same_side (p1 p2 : ℝ × ℝ) (line : ℝ × ℝ → ℝ) : Prop :=
  (line p1 > 0) ↔ (line p2 > 0)

def line_eq (p : ℝ × ℝ) : ℝ := p.1 + p.2 - 1

def point1 : ℝ × ℝ := (1, 2)
def pointC : ℝ × ℝ := (-1, 3)

-- Theorem to prove the equivalence
theorem pointC_on_same_side_as_point1 :
  is_on_same_side point1 pointC line_eq :=
sorry

end pointC_on_same_side_as_point1_l229_229169


namespace smallest_four_digit_in_pascals_triangle_l229_229423

-- Define Pascal's triangle
def pascals_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 1
| n+1, k+1 => pascals_triangle n k + pascals_triangle n (k+1)

-- State the theorem
theorem smallest_four_digit_in_pascals_triangle : ∃ (n k : ℕ), pascals_triangle n k = 1000 ∧ ∀ m, m < 1000 → ∀ r s, pascals_triangle r s ≠ m :=
sorry

end smallest_four_digit_in_pascals_triangle_l229_229423


namespace expand_expression_l229_229188

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  sorry

end expand_expression_l229_229188


namespace find_x_plus_y_l229_229850

theorem find_x_plus_y (x y : ℝ) (h1 : |x| + x + y = 12) (h2 : x + |y| - y = 16) :
  x + y = 4 := 
by
  sorry

end find_x_plus_y_l229_229850


namespace day_of_week_nminus1_l229_229111

theorem day_of_week_nminus1 (N : ℕ) 
  (h1 : (250 % 7 = 3 ∧ (250 / 7 * 7 + 3 = 250)) ∧ (150 % 7 = 3 ∧ (150 / 7 * 7 + 3 = 150))) :
  (50 % 7 = 0 ∧ (50 / 7 * 7 = 50)) := 
sorry

end day_of_week_nminus1_l229_229111


namespace first_three_workers_time_l229_229815

open Real

-- Define a function to represent the total work done by any set of workers
noncomputable def total_work (a b c d : ℝ) (t : ℝ) := (a + b + c + d) * t

-- Conditions
variables (a b c d : ℝ)
variables (h1 : a + b + c + d = 1 / 6)
variables (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
variables (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4)

-- Question: Time for the first three workers to dig the trench
def trench_time : ℝ :=
  (a + b + c + d) * (6 / (a + b + c + d))

theorem first_three_workers_time : trench_time a b c = 6 :=
  sorry

end first_three_workers_time_l229_229815


namespace total_respondents_l229_229153

theorem total_respondents (x_preference resp_y : ℕ) (h1 : x_preference = 360) (h2 : 9 * resp_y = x_preference) : 
  resp_y + x_preference = 400 :=
by 
  sorry

end total_respondents_l229_229153


namespace twelve_percent_greater_l229_229859

theorem twelve_percent_greater :
  ∃ x : ℝ, x = 80 + (12 / 100) * 80 := sorry

end twelve_percent_greater_l229_229859


namespace beth_sold_l229_229508

theorem beth_sold {initial_coins additional_coins total_coins sold_coins : ℕ} 
  (h_init : initial_coins = 125)
  (h_add : additional_coins = 35)
  (h_total : total_coins = initial_coins + additional_coins)
  (h_sold : sold_coins = total_coins / 2) :
  sold_coins = 80 := 
sorry

end beth_sold_l229_229508


namespace marked_price_of_article_l229_229152

noncomputable def marked_price (discounted_total : ℝ) (num_articles : ℕ) (discount_rate : ℝ) : ℝ :=
  let selling_price_each := discounted_total / num_articles
  let discount_factor := 1 - discount_rate
  selling_price_each / discount_factor

theorem marked_price_of_article :
  marked_price 50 2 0.10 = 250 / 9 :=
by
  unfold marked_price
  -- Instantiate values:
  -- discounted_total = 50
  -- num_articles = 2
  -- discount_rate = 0.10
  sorry

end marked_price_of_article_l229_229152


namespace find_quotient_l229_229733

theorem find_quotient
    (dividend divisor remainder : ℕ)
    (h1 : dividend = 136)
    (h2 : divisor = 15)
    (h3 : remainder = 1)
    (h4 : dividend = divisor * quotient + remainder) :
    quotient = 9 :=
by
  sorry

end find_quotient_l229_229733


namespace greatest_x_plus_z_l229_229835

theorem greatest_x_plus_z (x y z c d : ℕ) (h1 : 1 ≤ x ∧ x ≤ 9) (h2 : 1 ≤ y ∧ y ≤ 9) (h3 : 1 ≤ z ∧ z ≤ 9)
  (h4 : 700 - c = 700)
  (h5 : 100 * x + 10 * y + z - (100 * z + 10 * y + x) = 693)
  (h6 : x > z) :
  d = 11 :=
by
  sorry

end greatest_x_plus_z_l229_229835


namespace star_compound_l229_229068

noncomputable def star (A B : ℝ) : ℝ := (A + B) / 4

theorem star_compound : star (star 3 11) 6 = 2.375 := by
  sorry

end star_compound_l229_229068


namespace tangent_line_at_0_eq_x_plus_1_l229_229198

theorem tangent_line_at_0_eq_x_plus_1 :
  ∀ (x y : ℝ), y = sin x + 1 → y = x + 1 → x = 0 → y = 1 :=
by
  intros x y curve_eq tangent_line_eq x_at_tangent_point
  sorry

end tangent_line_at_0_eq_x_plus_1_l229_229198


namespace smallest_four_digit_in_pascal_triangle_l229_229469

noncomputable section

def is_in_pascal_triangle (n : ℕ) : Prop :=
  ∃ r c : ℕ, c ≤ r ∧ binomial r c = n

theorem smallest_four_digit_in_pascal_triangle :
  ¬ ∃ n : ℕ, is_in_pascal_triangle n ∧ n < 1000 ∧ (n ≥ 1000 ∨ n < 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_triangle_l229_229469


namespace total_surface_area_l229_229279

theorem total_surface_area (a b c : ℝ)
    (h1 : a + b + c = 40)
    (h2 : a^2 + b^2 + c^2 = 625)
    (h3 : a * b * c = 600) : 
    2 * (a * b + b * c + c * a) = 975 :=
by
  sorry

end total_surface_area_l229_229279


namespace tank_capacity_l229_229609

theorem tank_capacity (C : ℕ) 
  (leak_rate : C / 4 = C / 4)               -- Condition: Leak rate is C/4 litres per hour
  (inlet_rate : 6 * 60 = 360)                -- Condition: Inlet rate is 360 litres per hour
  (net_emptying_rate : C / 12 = (360 - C / 4))  -- Condition: Net emptying rate for 12 hours
  : C = 1080 := 
by 
  -- Conditions imply that C = 1080 
  sorry

end tank_capacity_l229_229609


namespace minimum_value_expression_l229_229247

noncomputable def problem_statement (x y z : ℝ) : ℝ :=
  (x + y) / z + (x + z) / y + (y + z) / x + 3

theorem minimum_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  problem_statement x y z ≥ 9 :=
sorry

end minimum_value_expression_l229_229247


namespace smallest_positive_integer_ends_in_9_and_divisible_by_13_l229_229022

theorem smallest_positive_integer_ends_in_9_and_divisible_by_13 :
  ∃ n : ℕ, n % 10 = 9 ∧ 13 ∣ n ∧ n > 0 ∧ ∀ m, m % 10 = 9 → 13 ∣ m ∧ m > 0 → m ≥ n := 
begin
  use 99,
  split,
  { exact mod_eq_of_lt (10*k + 9) 10 99 9 (by norm_num), },
  split,
  { exact dvd_refl 99, },
  split,
  { exact zero_lt_99, },
  intros m hm1 hm2 hpos,
  by_contradiction hmn,
  sorry
end

end smallest_positive_integer_ends_in_9_and_divisible_by_13_l229_229022


namespace no_real_x_condition_l229_229366

theorem no_real_x_condition (a : ℝ) :
  (¬ ∃ x : ℝ, |x - 3| + |x - 1| ≤ a) ↔ a < 2 := 
by
  sorry

end no_real_x_condition_l229_229366


namespace rectangle_ratio_ratio_simplification_l229_229106

theorem rectangle_ratio (w : ℕ) (h : w + 10 = 10) (p : 2 * w + 2 * 10 = 30) :
  w = 5 := by
  sorry

theorem ratio_simplification (x y : ℕ) (h : x * 10 = y * 5) (rel_prime : Nat.gcd x y = 1) :
  (x, y) = (1, 2) := by
  sorry

end rectangle_ratio_ratio_simplification_l229_229106


namespace degree_difference_l229_229893

variable (S J : ℕ)

theorem degree_difference :
  S = 150 → S + J = 295 → S - J = 5 :=
by
  intros h₁ h₂
  sorry

end degree_difference_l229_229893


namespace calvin_weight_after_one_year_l229_229334

theorem calvin_weight_after_one_year
  (initial_weight : ℕ)
  (monthly_weight_loss: ℕ)
  (months_in_year: ℕ)
  (one_year: ℕ)
  (total_loss: ℕ)
  (final_weight: ℕ) :
  initial_weight = 250 ∧ monthly_weight_loss = 8 ∧ months_in_year = 12 ∧ one_year = 12 ∧ total_loss = monthly_weight_loss * months_in_year →
  final_weight = initial_weight - total_loss →
  final_weight = 154 :=
by
  intros
  sorry

end calvin_weight_after_one_year_l229_229334


namespace opposite_face_of_lime_is_black_l229_229393

-- Define the colors
inductive Color
| P | C | M | S | K | L

-- Define the problem conditions
def face_opposite (c : Color) : Color := sorry

-- Theorem statement
theorem opposite_face_of_lime_is_black : face_opposite Color.L = Color.K := sorry

end opposite_face_of_lime_is_black_l229_229393


namespace find_original_number_l229_229491

theorem find_original_number (x : ℤ) (h : 3 * (2 * x + 9) = 57) : x = 5 := by
  sorry

end find_original_number_l229_229491


namespace circle_equation_l229_229836

theorem circle_equation :
  ∃ x y : ℝ, x = 2 ∧ y = 0 ∧ ∀ (p q : ℝ), ((p - x)^2 + q^2 = 4) ↔ (p^2 + q^2 - 4 * p = 0) :=
sorry

end circle_equation_l229_229836


namespace smallest_a_l229_229250

def f (x : ℕ) : ℕ :=
  if x % 21 = 0 then x / 21
  else if x % 7 = 0 then 3 * x
  else if x % 3 = 0 then 7 * x
  else x + 3

def f_iterate (a : ℕ) (x : ℕ) : ℕ :=
  Nat.iterate f a x

theorem smallest_a (a : ℕ) : a > 1 ∧ f_iterate a 2 = f 2 ↔ a = 7 := 
sorry

end smallest_a_l229_229250


namespace rhombus_longer_diagonal_l229_229638

theorem rhombus_longer_diagonal
  (a b : ℝ)
  (ha : a = 65) -- Side length
  (hb : b = 28) -- Half the length of the shorter diagonal
  : 2 * Real.sqrt (a^2 - b^2) = 118 :=  -- Statement of the longer diagonal length
by
  sorry -- Proof goes here

end rhombus_longer_diagonal_l229_229638


namespace percentage_material_B_new_mixture_l229_229385

theorem percentage_material_B_new_mixture :
  let mixtureA := 8 -- kg of Mixture A
  let addOil := 2 -- kg of additional oil
  let addMixA := 6 -- kg of additional Mixture A
  let oil_percent := 0.20 -- 20% oil in Mixture A
  let materialB_percent := 0.80 -- 80% material B in Mixture A

  -- Initial amounts in 8 kg of Mixture A
  let initial_oil := oil_percent * mixtureA
  let initial_materialB := materialB_percent * mixtureA

  -- New mixture after adding 2 kg oil
  let new_oil := initial_oil + addOil
  let new_materialB := initial_materialB

  -- Adding 6 kg of Mixture A
  let added_oil := oil_percent * addMixA
  let added_materialB := materialB_percent * addMixA

  -- Total amounts in the new mixture
  let total_oil := new_oil + added_oil
  let total_materialB := new_materialB + added_materialB
  let total_weight := mixtureA + addOil + addMixA

  -- Percent calculation
  let percent_materialB := (total_materialB / total_weight) * 100

  percent_materialB = 70 := sorry

end percentage_material_B_new_mixture_l229_229385


namespace beth_sells_half_of_coins_l229_229506

theorem beth_sells_half_of_coins (x y : ℕ) (h₁ : x = 125) (h₂ : y = 35) : (x + y) / 2 = 80 :=
by
  sorry

end beth_sells_half_of_coins_l229_229506


namespace g_value_at_50_l229_229626

noncomputable def g : ℝ → ℝ := sorry

theorem g_value_at_50 (g : ℝ → ℝ)
  (H1 : ∀ x y : ℝ, 0 < x → 0 < y → x * g y + y * g x = g (x * y)) :
  g 50 = 0 :=
sorry

end g_value_at_50_l229_229626


namespace smallest_four_digit_in_pascal_l229_229415

-- define what it means for a number to be in Pascal's triangle
def in_pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- define the concept of a smallest four-digit number
def smallest_four_digit : ℕ := 1000

-- prove that the smallest four-digit number in Pascal's Triangle is 1000
theorem smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), in_pascal n k = smallest_four_digit := 
sorry

end smallest_four_digit_in_pascal_l229_229415


namespace bottle_cap_cost_l229_229801

-- Define the conditions given in the problem.
def caps_cost (n : ℕ) (cost : ℝ) : Prop := n * cost = 12

-- Prove that the cost of each bottle cap is $2 given 6 bottle caps cost $12.
theorem bottle_cap_cost (h : caps_cost 6 cost) : cost = 2 :=
sorry

end bottle_cap_cost_l229_229801


namespace ratio_of_sums_l229_229779

theorem ratio_of_sums (total_sums : ℕ) (correct_sums : ℕ) (incorrect_sums : ℕ)
  (h1 : total_sums = 75)
  (h2 : incorrect_sums = 2 * correct_sums)
  (h3 : total_sums = correct_sums + incorrect_sums) :
  incorrect_sums / correct_sums = 2 :=
by
  -- Proof placeholder
  sorry

end ratio_of_sums_l229_229779


namespace find_a_when_lines_perpendicular_l229_229081

theorem find_a_when_lines_perpendicular (a : ℝ) : 
  (∃ x y : ℝ, ax + 3 * y - 1 = 0 ∧  2 * x + (a^2 - a) * y + 3 = 0) ∧ 
  (∃ m₁ m₂ : ℝ, m₁ = -a / 3 ∧ m₂ = -2 / (a^2 - a) ∧ m₁ * m₂ = -1)
  → a = 0 ∨ a = 5 / 3 :=
by {
  sorry
}

end find_a_when_lines_perpendicular_l229_229081


namespace trapezoid_height_l229_229166

theorem trapezoid_height (A : ℝ) (d1 d2 : ℝ) (h : ℝ) :
  A = 2 ∧ d1 + d2 = 4 → h = Real.sqrt 2 :=
by
  sorry

end trapezoid_height_l229_229166


namespace rhombus_longer_diagonal_l229_229632

theorem rhombus_longer_diagonal (a b : ℕ) (d1 : ℕ) (d2 : ℕ) (h0 : a = 65) (h1 : d1 = 56) (h2 : a * a = (d1 / 2) * (d1 / 2) + (d2 / 2) * (d2 / 2)) :
  d2 = 118 :=
by
  have h3 : (d1 / 2) * (d1 / 2) + (d2 / 2) * (d2 / 2) = 65 * 65, from h2,
  rw [←h0, ←h1] at h3,
  have h4 : (28 : ℕ) = d1 / 2, sorry,
  have h5 : (28 : ℕ) * (28 : ℕ) + ((d2 / 2) * (d2 / 2)) = 65 * 65, from h3,
  have h6 : (28 * 28) + (d2 / 2) * (d2 / 2) = (65*65), by rw h5,
  have h7 : (65 * 65) - 784 = (d2 / 2) * (d2 / 2), sorry,
  have h8 : (d2 / 2) * (d2 / 2) = 3441, by simp[h7],
  have h9 : (d2 / 2) = 59, from nat.sqrt_eq 3441,
  have h10 : d2 = 2 * 59, by rw [h9, two_mul],
  exact h10
 

end rhombus_longer_diagonal_l229_229632


namespace symmetric_codes_count_l229_229776

def isSymmetric (grid : List (List Bool)) : Prop :=
  -- condition for symmetry: rotational and reflectional symmetry
  sorry

def isValidCode (grid : List (List Bool)) : Prop :=
  -- condition for valid scanning code with at least one black and one white
  sorry

noncomputable def numberOfSymmetricCodes : Nat :=
  -- function to count the number of symmetric valid codes
  sorry

theorem symmetric_codes_count :
  numberOfSymmetricCodes = 62 := 
  sorry

end symmetric_codes_count_l229_229776


namespace value_of_2a_plus_b_l229_229854

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

def is_tangent_perpendicular (a b : ℝ) : Prop :=
  let f' := (fun x => (1 : ℝ) / x - a)
  let slope_perpendicular_line := - (1/3 : ℝ)
  f' 1 * slope_perpendicular_line = -1 

def point_on_function (a b : ℝ) : Prop :=
  f a 1 = b

theorem value_of_2a_plus_b (a b : ℝ) 
  (h_tangent_perpendicular : is_tangent_perpendicular a b)
  (h_point_on_function : point_on_function a b) : 
  2 * a + b = -2 := sorry

end value_of_2a_plus_b_l229_229854


namespace greatest_integer_function_of_pi_plus_3_l229_229791

noncomputable def pi_plus_3 : Real := Real.pi + 3

theorem greatest_integer_function_of_pi_plus_3 : Int.floor pi_plus_3 = 6 := 
by
  -- sorry is used to skip the proof
  sorry

end greatest_integer_function_of_pi_plus_3_l229_229791


namespace plan_b_more_cost_effective_l229_229167

theorem plan_b_more_cost_effective (x : ℕ) : 
  (12 * x : ℤ) > (3000 + 8 * x : ℤ) → x ≥ 751 :=
sorry

end plan_b_more_cost_effective_l229_229167


namespace tangency_point_l229_229201

theorem tangency_point (x y : ℝ) : 
  y = x ^ 2 + 20 * x + 70 ∧ x = y ^ 2 + 70 * y + 1225 →
  (x, y) = (-19 / 2, -69 / 2) :=
by {
  sorry
}

end tangency_point_l229_229201


namespace circle_radius_twice_value_l229_229971

theorem circle_radius_twice_value (r_x r_y v : ℝ) (h1 : π * r_x^2 = π * r_y^2)
  (h2 : 2 * π * r_x = 12 * π) (h3 : r_y = 2 * v) : v = 3 := by
  sorry

end circle_radius_twice_value_l229_229971


namespace min_value_of_z_l229_229209

noncomputable def min_z (x y : ℝ) : ℝ :=
  2 * x + (Real.sqrt 3) * y

theorem min_value_of_z :
  ∃ x y : ℝ, 3 * x^2 + 4 * y^2 = 12 ∧ min_z x y = -5 :=
sorry

end min_value_of_z_l229_229209


namespace smallest_four_digit_in_pascals_triangle_l229_229464

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, (1000 ≤ (nat.choose n k)) ∧ (∀ m l : ℕ, 1000 ≤ nat.choose m l → nat.choose m l ≥ 1000) :=
begin
  use [1000, 1],
  split,
  { exact nat.choose_eq_one_right 1000 1 },
  { intros m l hml,
    linarith }
end

end smallest_four_digit_in_pascals_triangle_l229_229464


namespace needs_change_probability_l229_229006

noncomputable def probability_needs_change (n_toys : ℕ) (quarters : ℕ) (twenty_dollar_bill : ℕ) (fav_toy_cost : ℝ) : ℝ :=
  (n_toys = 10) → 
  (quarters = 10) → 
  (twenty_dollar_bill = 20) → 
  (fav_toy_cost = 2.25) →
  ((10! : ℝ) / ((10 * 9) : ℝ)) * (1 - (((1 / 10) * (1 / 9) * 8!) + ((7 / 10) * (1 / 9) * 8!)) / (10!)) = (5 / 6)

-- Define a theorem stating the probability as described in the solution.
theorem needs_change_probability: 
  probability_needs_change 10 10 20 2.25 = (5 / 6) :=
by
  sorry

end needs_change_probability_l229_229006


namespace a_13_eq_30_l229_229678

variable (a : ℕ → ℕ)
variable (d : ℕ)

-- Define arithmetic sequence condition
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop := ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
axiom a_5_eq_6 : a 5 = 6
axiom a_8_eq_15 : a 8 = 15

-- Required proof
theorem a_13_eq_30 (h : arithmetic_sequence a d) : a 13 = 30 :=
  sorry

end a_13_eq_30_l229_229678


namespace min_value_abc2_l229_229579

variables (a b c d : ℝ)

def condition_1 : Prop := a + b = 9 / (c - d)
def condition_2 : Prop := c + d = 25 / (a - b)

theorem min_value_abc2 :
  condition_1 a b c d → condition_2 a b c d → (a^2 + b^2 + c^2 + d^2) = 34 :=
by
  intros h1 h2
  sorry

end min_value_abc2_l229_229579


namespace divisor_of_425904_l229_229919

theorem divisor_of_425904 :
  ∃ (d : ℕ), d = 7 ∧ ∃ (n : ℕ), n = 425897 + 7 ∧ 425904 % d = 0 :=
by
  sorry

end divisor_of_425904_l229_229919


namespace four_digit_number_perfect_square_l229_229771

theorem four_digit_number_perfect_square (abcd : ℕ) (h1 : abcd ≥ 1000 ∧ abcd < 10000) (h2 : ∃ k : ℕ, k^2 = 4000000 + abcd) :
  abcd = 4001 ∨ abcd = 8004 :=
sorry

end four_digit_number_perfect_square_l229_229771


namespace smallest_four_digit_in_pascal_triangle_l229_229470

noncomputable section

def is_in_pascal_triangle (n : ℕ) : Prop :=
  ∃ r c : ℕ, c ≤ r ∧ binomial r c = n

theorem smallest_four_digit_in_pascal_triangle :
  ¬ ∃ n : ℕ, is_in_pascal_triangle n ∧ n < 1000 ∧ (n ≥ 1000 ∨ n < 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_triangle_l229_229470


namespace carly_trimmed_nails_correct_l229_229179

-- Definitions based on the conditions
def total_dogs : Nat := 11
def three_legged_dogs : Nat := 3
def paws_per_four_legged_dog : Nat := 4
def paws_per_three_legged_dog : Nat := 3
def nails_per_paw : Nat := 4

-- Mathematically equivalent proof problem in Lean 4 statement
theorem carly_trimmed_nails_correct :
  let four_legged_dogs := total_dogs - three_legged_dogs
  let nails_per_four_legged_dog := paws_per_four_legged_dog * nails_per_paw
  let nails_per_three_legged_dog := paws_per_three_legged_dog * nails_per_paw
  let total_nails_trimmed :=
    (four_legged_dogs * nails_per_four_legged_dog) +
    (three_legged_dogs * nails_per_three_legged_dog)
  total_nails_trimmed = 164 := by
  sorry

end carly_trimmed_nails_correct_l229_229179


namespace chewbacca_gum_packs_l229_229970

theorem chewbacca_gum_packs (x : ℕ) :
  (30 - 2 * x) * (40 + 4 * x) = 1200 → x = 5 :=
by
  -- This is where the proof would go. We'll leave it as sorry for now.
  sorry

end chewbacca_gum_packs_l229_229970


namespace increased_cost_per_person_l229_229744

-- Declaration of constants
def initial_cost : ℕ := 30000000000 -- 30 billion dollars in dollars
def people_sharing : ℕ := 300000000 -- 300 million people
def inflation_rate : ℝ := 0.10 -- 10% inflation rate

-- Calculation of increased cost per person
theorem increased_cost_per_person : (initial_cost * (1 + inflation_rate) / people_sharing) = 110 :=
by sorry

end increased_cost_per_person_l229_229744


namespace difference_of_squares_l229_229593

theorem difference_of_squares (x y : ℕ) (h1 : x + y = 26) (h2 : x * y = 168) : x^2 - y^2 = 52 := by
  sorry

end difference_of_squares_l229_229593


namespace tan_alpha_sub_beta_l229_229670

theorem tan_alpha_sub_beta (α β : ℝ) (h₁ : Real.tan α = 9) (h₂ : Real.tan β = 6) : Real.tan (α - β) = 3 / 55 := 
sorry

end tan_alpha_sub_beta_l229_229670


namespace total_receipts_correct_l229_229503

def cost_adult_ticket : ℝ := 5.50
def cost_children_ticket : ℝ := 2.50
def number_of_adults : ℕ := 152
def number_of_children : ℕ := number_of_adults / 2

def receipts_from_adults : ℝ := number_of_adults * cost_adult_ticket
def receipts_from_children : ℝ := number_of_children * cost_children_ticket
def total_receipts : ℝ := receipts_from_adults + receipts_from_children

theorem total_receipts_correct : total_receipts = 1026 := 
by
  -- Proof omitted, proof needed to validate theorem statement.
  sorry

end total_receipts_correct_l229_229503


namespace smallest_four_digit_in_pascals_triangle_l229_229445

theorem smallest_four_digit_in_pascals_triangle : ∃ n, ∃ k, k ≤ n ∧ 1000 ≤ Nat.choose n k :=
sorry

end smallest_four_digit_in_pascals_triangle_l229_229445


namespace addition_correct_l229_229484

-- Define the integers involved
def num1 : ℤ := 22
def num2 : ℤ := 62
def result : ℤ := 84

-- Theorem stating the relationship between the given numbers
theorem addition_correct : num1 + num2 = result :=
by {
  -- proof goes here
  sorry
}

end addition_correct_l229_229484


namespace smallest_four_digit_in_pascals_triangle_l229_229467

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, (1000 ≤ (nat.choose n k)) ∧ (∀ m l : ℕ, 1000 ≤ nat.choose m l → nat.choose m l ≥ 1000) :=
begin
  use [1000, 1],
  split,
  { exact nat.choose_eq_one_right 1000 1 },
  { intros m l hml,
    linarith }
end

end smallest_four_digit_in_pascals_triangle_l229_229467


namespace perpendicular_line_through_circle_center_l229_229074

theorem perpendicular_line_through_circle_center :
  ∃ (a b c : ℝ), (∀ (x y : ℝ), x^2 + y^2 - 2*x - 8 = 0 → x + 2*y = 0 → a * x + b * y + c = 0) ∧
  a = 2 ∧ b = -1 ∧ c = -2 :=
by
  sorry

end perpendicular_line_through_circle_center_l229_229074


namespace smallest_four_digit_in_pascals_triangle_l229_229443

/-- Pascal's triangle is defined using binomial coefficients.
    We want to prove that the smallest four-digit number in Pascal's triangle is 1001. -/
theorem smallest_four_digit_in_pascals_triangle : ∃ n k : ℕ, n ≥ k ∧ (binomial n k = 1001 ∧ ∀ m l, m ≥ l → binomial m l ≥ 1000 → binomial m l = 1001 → n = m ∧ k = l) := by
  sorry

end smallest_four_digit_in_pascals_triangle_l229_229443


namespace number_of_girls_with_no_pet_l229_229866

-- Definitions based on the conditions
def total_students : ℕ := 30
def fraction_boys : ℚ := 1 / 3
def percentage_girls_own_dogs : ℚ := 40 / 100
def percentage_girls_own_cats : ℚ := 20 / 100

-- Prove that the number of girls with no pets is 8
theorem number_of_girls_with_no_pet :
  let girls := total_students * (1 - fraction_boys),
      percentage_girls_no_pets := 1 - percentage_girls_own_dogs - percentage_girls_own_cats,
      girls_with_no_pets := girls * percentage_girls_no_pets
  in girls_with_no_pets = 8 := by
{
  sorry
}

end number_of_girls_with_no_pet_l229_229866


namespace final_movie_ticket_price_l229_229076

variable (initial_price : ℝ) (price_year1 price_year2 price_year3 price_year4 price_year5 : ℝ)

def price_after_years (initial_price : ℝ) : ℝ :=
  let price_year1 := initial_price * 1.12
  let price_year2 := price_year1 * 0.95
  let price_year3 := price_year2 * 1.08
  let price_year4 := price_year3 * 0.96
  let price_year5 := price_year4 * 1.06
  price_year5

theorem final_movie_ticket_price :
  price_after_years 100 = 116.9344512 :=
by
  sorry

end final_movie_ticket_price_l229_229076


namespace minimum_value_expression_l229_229244

variable (x y z : ℝ)
variable (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)

theorem minimum_value_expression : 
  (\frac{x + y}{z} + \frac{x + z}{y} + \frac{y + z}{x} + 3) ≥ 9 :=
by
  sorry

end minimum_value_expression_l229_229244


namespace system_solution_unique_l229_229806

theorem system_solution_unique (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (eq1 : x ^ 3 + 2 * y ^ 2 + 1 / (4 * z) = 1)
  (eq2 : y ^ 3 + 2 * z ^ 2 + 1 / (4 * x) = 1)
  (eq3 : z ^ 3 + 2 * x ^ 2 + 1 / (4 * y) = 1) :
  (x, y, z) = ( ( (-1 + Real.sqrt 3) / 2), ((-1 + Real.sqrt 3) / 2), ((-1 + Real.sqrt 3) / 2) ) := 
by
  sorry

end system_solution_unique_l229_229806


namespace smallest_four_digit_in_pascals_triangle_l229_229420

-- Define Pascal's triangle
def pascals_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 1
| n+1, k+1 => pascals_triangle n k + pascals_triangle n (k+1)

-- State the theorem
theorem smallest_four_digit_in_pascals_triangle : ∃ (n k : ℕ), pascals_triangle n k = 1000 ∧ ∀ m, m < 1000 → ∀ r s, pascals_triangle r s ≠ m :=
sorry

end smallest_four_digit_in_pascals_triangle_l229_229420


namespace problem_l229_229116

noncomputable def a : ℝ := (Real.sqrt 5 + Real.sqrt 3) / (Real.sqrt 5 - Real.sqrt 3)
noncomputable def b : ℝ := (Real.sqrt 5 - Real.sqrt 3) / (Real.sqrt 5 + Real.sqrt 3)

theorem problem :
  a^4 + b^4 + (a + b)^4 = 7938 := by
  sorry

end problem_l229_229116


namespace minimum_value_expression_l229_229557

theorem minimum_value_expression 
  (a b : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_eq : 1 / a + 1 / b = 1) : 
  (∃ (x : ℝ), x = (1 / (a-1) + 9 / (b-1)) ∧ x = 6) :=
sorry

end minimum_value_expression_l229_229557


namespace constructible_iff_multiple_of_8_l229_229258

def is_constructible_with_L_tetromino (m n : ℕ) : Prop :=
  ∃ (k : ℕ), 4 * k = m * n

theorem constructible_iff_multiple_of_8 (m n : ℕ) (hm : 1 < m) (hn : 1 < n) : 
  is_constructible_with_L_tetromino m n ↔ 8 ∣ m * n :=
sorry

end constructible_iff_multiple_of_8_l229_229258


namespace largest_undefined_x_value_l229_229147

theorem largest_undefined_x_value :
  ∃ x : ℝ, (6 * x^2 - 65 * x + 54 = 0) ∧ (∀ y : ℝ, (6 * y^2 - 65 * y + 54 = 0) → y ≤ x) :=
sorry

end largest_undefined_x_value_l229_229147


namespace time_for_first_three_workers_l229_229809

def work_rate_equations (a b c d : ℝ) :=
  a + b + c + d = 1 / 6 ∧
  2 * a + (1 / 2) * b + c + d = 1 / 6 ∧
  (1 / 2) * a + 2 * b + c + d = 1 / 4

theorem time_for_first_three_workers (a b c d : ℝ) (h : work_rate_equations a b c d) :
  (a + b + c) * 6 = 1 :=
sorry

end time_for_first_three_workers_l229_229809


namespace sqrt_mul_sqrt_eq_l229_229332

theorem sqrt_mul_sqrt_eq (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : (Real.sqrt a) * (Real.sqrt b) = Real.sqrt (a * b) :=
by {
  sorry
}

example : Real.sqrt 2 * Real.sqrt 8 = 4 :=
by {
  have h: Real.sqrt 2 * Real.sqrt 8 = Real.sqrt (2 * 8) := sqrt_mul_sqrt_eq 2 8 (by norm_num) (by norm_num),
  rw h,
  norm_num
}

end sqrt_mul_sqrt_eq_l229_229332


namespace find_x_l229_229553

theorem find_x
  (p q : ℝ)
  (h1 : 3 / p = x)
  (h2 : 3 / q = 18)
  (h3 : p - q = 0.33333333333333337) :
  x = 6 :=
sorry

end find_x_l229_229553


namespace trench_dig_time_l229_229822

theorem trench_dig_time (a b c d : ℝ) (h1 : a + b + c + d = 1/6)
  (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
  (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4) :
  a + b + c = 1 / 6 := sorry

end trench_dig_time_l229_229822


namespace solve_inequality_l229_229264

theorem solve_inequality (x : ℝ) (h : 3 * x + 4 ≠ 0) :
  3 - 1 / (3 * x + 4) < 5 ↔ -3 / 2 < x :=
by
  sorry

end solve_inequality_l229_229264


namespace rectangle_ratio_ratio_simplification_l229_229105

theorem rectangle_ratio (w : ℕ) (h : w + 10 = 10) (p : 2 * w + 2 * 10 = 30) :
  w = 5 := by
  sorry

theorem ratio_simplification (x y : ℕ) (h : x * 10 = y * 5) (rel_prime : Nat.gcd x y = 1) :
  (x, y) = (1, 2) := by
  sorry

end rectangle_ratio_ratio_simplification_l229_229105


namespace price_decrease_percentage_l229_229622

theorem price_decrease_percentage (original_price : ℝ) :
  let first_sale_price := (4/5) * original_price
  let second_sale_price := (1/2) * original_price
  let decrease := first_sale_price - second_sale_price
  let percentage_decrease := (decrease / first_sale_price) * 100
  percentage_decrease = 37.5 := by
  sorry

end price_decrease_percentage_l229_229622


namespace gcd_1728_1764_l229_229145

theorem gcd_1728_1764 : Int.gcd 1728 1764 = 36 := by
  sorry

end gcd_1728_1764_l229_229145


namespace sum_of_x_and_y_l229_229846

theorem sum_of_x_and_y :
  ∀ (x y : ℚ), (1/x + 1/y = 4) → (1/x - 1/y = -6) → x + y = -4/5 :=
by
  intros x y h1 h2
  sorry

end sum_of_x_and_y_l229_229846


namespace most_likely_dissatisfied_passengers_correct_expected_dissatisfied_passengers_correct_variance_dissatisfied_passengers_correct_l229_229914

noncomputable def most_likely_dissatisfied_passengers (n : ℕ) : ℕ :=
  1

theorem most_likely_dissatisfied_passengers_correct (n : ℕ) :
  let volume := 2 * n in
  let chicken := n in
  let fish := n in
  let preferred_chicken := 0.5 in
  let preferred_fish := 0.5 in
  most_likely_dissatisfied_passengers n = 1 :=
by
  sorry

noncomputable def expected_dissatisfied_passengers (n : ℕ) : ℝ :=
  sqrt (n / π)

theorem expected_dissatisfied_passengers_correct (n : ℕ) :
  let volume := 2 * n in
  let chicken := n in
  let fish := n in 
  let preferred_chicken := 0.5 in
  let preferred_fish := 0.5 in
  expected_dissatisfied_passengers n ≈ sqrt (n / π) :=
by
  sorry

noncomputable def variance_dissatisfied_passengers (n : ℕ) : ℝ :=
  0.182 * n

theorem variance_dissatisfied_passengers_correct (n : ℕ) :
  let volume := 2 * n in
  let chicken := n in
  let fish := n in
  let preferred_chicken := 0.5 in
  let preferred_fish := 0.5 in
  variance_dissatisfied_passengers n ≈ 0.182 * n :=
by
  sorry

end most_likely_dissatisfied_passengers_correct_expected_dissatisfied_passengers_correct_variance_dissatisfied_passengers_correct_l229_229914


namespace distance_traveled_by_second_hand_l229_229488

theorem distance_traveled_by_second_hand (r : ℝ) (minutes : ℝ) (h1 : r = 10) (h2 : minutes = 45) :
  (2 * Real.pi * r) * (minutes / 1) = 900 * Real.pi := by
  -- Given:
  -- r = length of the second hand = 10 cm
  -- minutes = 45
  -- To prove: distance traveled by the tip = 900π cm
  sorry

end distance_traveled_by_second_hand_l229_229488


namespace rhombus_longer_diagonal_l229_229631

theorem rhombus_longer_diagonal (a b : ℕ) (d1 : ℕ) (d2 : ℕ) (h0 : a = 65) (h1 : d1 = 56) (h2 : a * a = (d1 / 2) * (d1 / 2) + (d2 / 2) * (d2 / 2)) :
  d2 = 118 :=
by
  have h3 : (d1 / 2) * (d1 / 2) + (d2 / 2) * (d2 / 2) = 65 * 65, from h2,
  rw [←h0, ←h1] at h3,
  have h4 : (28 : ℕ) = d1 / 2, sorry,
  have h5 : (28 : ℕ) * (28 : ℕ) + ((d2 / 2) * (d2 / 2)) = 65 * 65, from h3,
  have h6 : (28 * 28) + (d2 / 2) * (d2 / 2) = (65*65), by rw h5,
  have h7 : (65 * 65) - 784 = (d2 / 2) * (d2 / 2), sorry,
  have h8 : (d2 / 2) * (d2 / 2) = 3441, by simp[h7],
  have h9 : (d2 / 2) = 59, from nat.sqrt_eq 3441,
  have h10 : d2 = 2 * 59, by rw [h9, two_mul],
  exact h10
 

end rhombus_longer_diagonal_l229_229631


namespace gcd_odd_multiple_1187_l229_229211

theorem gcd_odd_multiple_1187 (b: ℤ) (h1: b % 2 = 1) (h2: ∃ k: ℤ, b = 1187 * k) :
  Int.gcd (3 * b ^ 2 + 34 * b + 76) (b + 16) = 1 :=
by
  sorry

end gcd_odd_multiple_1187_l229_229211


namespace symmetric_point_coordinates_l229_229868

theorem symmetric_point_coordinates :
  ∀ (M N : ℝ × ℝ), M = (3, -4) ∧ M.fst = -N.fst ∧ M.snd = N.snd → N = (-3, -4) :=
by
  intro M N h
  sorry

end symmetric_point_coordinates_l229_229868


namespace problem_solution_l229_229356

noncomputable def set_M (x : ℝ) : Prop := x^2 - 4*x < 0
noncomputable def set_N (m x : ℝ) : Prop := m < x ∧ x < 5
noncomputable def set_intersection (x : ℝ) : Prop := 3 < x ∧ x < 4

theorem problem_solution (m n : ℝ) :
  (∀ x, set_M x ↔ (0 < x ∧ x < 4)) →
  (∀ x, set_N m x ↔ (m < x ∧ x < 5)) →
  (∀ x, (set_M x ∧ set_N m x) ↔ set_intersection x) →
  m + n = 7 :=
by
  intros H1 H2 H3
  sorry

end problem_solution_l229_229356


namespace probability_differ_by_three_is_one_sixth_l229_229951

def probability_of_differ_by_three (outcomes : ℕ) : ℚ :=
  let successful_outcomes := 6
  successful_outcomes / outcomes

theorem probability_differ_by_three_is_one_sixth :
  probability_of_differ_by_three (6 * 6) = 1 / 6 :=
by sorry

end probability_differ_by_three_is_one_sixth_l229_229951


namespace rectangular_prism_total_count_l229_229939

-- Define the dimensions of the rectangular prism
def length : ℕ := 4
def width : ℕ := 3
def height : ℕ := 5

-- Define the total count of edges, corners, and faces
def total_count : ℕ := 12 + 8 + 6

-- The proof statement that the total count is 26
theorem rectangular_prism_total_count : total_count = 26 :=
by
  sorry

end rectangular_prism_total_count_l229_229939


namespace right_triangle_point_selection_l229_229253

theorem right_triangle_point_selection : 
  let n := 200 
  let rows := 2
  (rows * (n - 22 + 1)) + 2 * (rows * (n - 122 + 1)) + (n * (2 * (n - 1))) = 80268 := 
by 
  let rows := 2
  let n := 200
  let case1a := rows * (n - 22 + 1)
  let case1b := 2 * (rows * (n - 122 + 1))
  let case2 := n * (2 * (n - 1))
  have h : case1a + case1b + case2 = 80268 := by sorry
  exact h

end right_triangle_point_selection_l229_229253


namespace ellipse_eccentricity_range_of_ratio_l229_229322

-- The setup conditions
variables {a b c : ℝ}
variables (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b) (h1 : a^2 - b^2 = c^2)
variables (M : ℝ) (m : ℝ)
variables (hM : M = a + c) (hm : m = a - c) (hMm : M * m = 3 / 4 * a^2)

-- Proof statement for the eccentricity of the ellipse
theorem ellipse_eccentricity : c / a = 1 / 2 := by
  sorry

-- The setup for the second part
variables {S1 S2 : ℝ}
variables (ellipse_eq : ∀ x y : ℝ, (x^2 / (4 * c^2) + y^2 / (3 * c^2) = 1) → x + y = 0)
variables (range_S : S1 / S2 > 9)

-- Proof statement for the range of the given ratio
theorem range_of_ratio : 0 < (2 * S1 * S2) / (S1^2 + S2^2) ∧ (2 * S1 * S2) / (S1^2 + S2^2) < 9 / 41 := by
  sorry

end ellipse_eccentricity_range_of_ratio_l229_229322


namespace boys_and_girls_total_l229_229059

theorem boys_and_girls_total (c : ℕ) (h_lollipop_fraction : c = 90) 
  (h_one_third_lollipops : c / 3 = 30)
  (h_lollipops_shared : 30 / 3 = 10) 
  (h_candy_caness_shared : 60 / 2 = 30) : 
  10 + 30 = 40 :=
by
  simp [h_one_third_lollipops, h_lollipops_shared, h_candy_caness_shared]

end boys_and_girls_total_l229_229059


namespace solve_fraction_equation_l229_229889

theorem solve_fraction_equation (x : ℝ) (h : (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2) : x = -2 / 3 :=
by
  sorry

end solve_fraction_equation_l229_229889


namespace quadratic_root_unique_l229_229752

theorem quadratic_root_unique 
  (a b c : ℝ)
  (hf1 : b^2 - 4 * a * c = 0)
  (hf2 : (b - 30 * a)^2 - 4 * a * (17 * a - 7 * b + c) = 0)
  (ha_pos : a ≠ 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x = -11 := 
by
  sorry

end quadratic_root_unique_l229_229752


namespace smallest_four_digit_in_pascals_triangle_l229_229466

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, (1000 ≤ (nat.choose n k)) ∧ (∀ m l : ℕ, 1000 ≤ nat.choose m l → nat.choose m l ≥ 1000) :=
begin
  use [1000, 1],
  split,
  { exact nat.choose_eq_one_right 1000 1 },
  { intros m l hml,
    linarith }
end

end smallest_four_digit_in_pascals_triangle_l229_229466


namespace find_center_angle_l229_229129

noncomputable def pi : ℝ := Real.pi
/-- Given conditions from the math problem -/
def radius : ℝ := 12
def area : ℝ := 67.88571428571429

theorem find_center_angle (θ : ℝ) 
  (area_def : area = (θ / 360) * pi * radius ^ 2) : 
  θ = 54 :=
sorry

end find_center_angle_l229_229129


namespace pizzas_returned_l229_229316

theorem pizzas_returned (total_pizzas served_pizzas : ℕ) (h_total : total_pizzas = 9) (h_served : served_pizzas = 3) : (total_pizzas - served_pizzas) = 6 :=
by
  sorry

end pizzas_returned_l229_229316


namespace polar_to_rectangular_l229_229182

theorem polar_to_rectangular : 
  ∀ (r θ : ℝ), 
  r = 8 → 
  θ = 7 * Real.pi / 4 → 
  (r * Real.cos θ, r * Real.sin θ) = (4 * Real.sqrt 2, -4 * Real.sqrt 2) :=
by 
  intros r θ hr hθ
  rw [hr, hθ]
  sorry

end polar_to_rectangular_l229_229182


namespace skittles_distribution_l229_229646

-- Given problem conditions
variable (Brandon_initial : ℕ := 96) (Bonnie_initial : ℕ := 4) 
variable (Brandon_loss : ℕ := 9)
variable (combined_skittles : ℕ := (Brandon_initial - Brandon_loss) + Bonnie_initial)
variable (individual_share : ℕ := combined_skittles / 4)
variable (remainder : ℕ := combined_skittles % 4)
variable (Chloe_share : ℕ := individual_share)
variable (Dylan_share_initial : ℕ := individual_share)
variable (Chloe_to_Dylan : ℕ := Chloe_share / 2)
variable (Dylan_new_share : ℕ := Dylan_share_initial + Chloe_to_Dylan)
variable (Dylan_to_Bonnie : ℕ := Dylan_new_share / 3)
variable (final_Bonnie : ℕ := individual_share + Dylan_to_Bonnie)
variable (final_Chloe : ℕ := Chloe_share - Chloe_to_Dylan)
variable (final_Dylan : ℕ := Dylan_new_share - Dylan_to_Bonnie)

-- The theorem to be proved
theorem skittles_distribution : 
  individual_share = 22 ∧ final_Bonnie = 33 ∧ final_Chloe = 11 ∧ final_Dylan = 22 :=
by
  -- The proof would go here, but it’s not required for this task.
  sorry

end skittles_distribution_l229_229646


namespace probability_of_5_odd_in_6_rolls_l229_229015

open Classical

noncomputable def prob_odd_in_six_rolls : ℚ :=
  let num_rolls := 6
  let prob_odd_single := 1 / 2
  let binom_coeff := Nat.choose num_rolls 5
  let total_outcomes := (2 : ℕ) ^ num_rolls
  binom_coeff * ((prob_odd_single ^ 5) * ((1 - prob_odd_single) ^ (num_rolls - 5))) / total_outcomes

theorem probability_of_5_odd_in_6_rolls :
  prob_odd_in_six_rolls = 3 / 32 :=
by
  sorry

end probability_of_5_odd_in_6_rolls_l229_229015


namespace expand_expression_l229_229190

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  sorry

end expand_expression_l229_229190


namespace root_of_quadratic_poly_l229_229753

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

def poly_has_root (a b c : ℝ) (r : ℝ) : Prop := a * r^2 + b * r + c = 0

theorem root_of_quadratic_poly 
  (a b c : ℝ)
  (h1 : discriminant a b c = 0)
  (h2 : discriminant (-a) (b - 30 * a) (17 * a - 7 * b + c) = 0):
  poly_has_root a b c (-11) :=
sorry

end root_of_quadratic_poly_l229_229753


namespace ethanol_concentration_l229_229757

theorem ethanol_concentration
  (w1 : ℕ) (c1 : ℝ) (w2 : ℕ) (c2 : ℝ)
  (hw1 : w1 = 400) (hc1 : c1 = 0.30)
  (hw2 : w2 = 600) (hc2 : c2 = 0.80) :
  (c1 * w1 + c2 * w2) / (w1 + w2) = 0.60 := 
by
  sorry

end ethanol_concentration_l229_229757


namespace prize_selection_count_l229_229090

theorem prize_selection_count :
  (Nat.choose 20 1) * (Nat.choose 19 2) * (Nat.choose 17 4) = 8145600 := 
by 
  sorry

end prize_selection_count_l229_229090


namespace tens_digit_seven_last_digit_six_l229_229698

theorem tens_digit_seven_last_digit_six (n : ℕ) (h : ((n * n) / 10) % 10 = 7) :
  (n * n) % 10 = 6 :=
sorry

end tens_digit_seven_last_digit_six_l229_229698


namespace sum_of_digits_b_n_l229_229727

def a_n (n : ℕ) : ℕ := 10^(2^n) - 1

def b_n (n : ℕ) : ℕ :=
  List.prod (List.map a_n (List.range (n + 1)))

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_b_n (n : ℕ) : sum_of_digits (b_n n) = 9 * 2^n :=
  sorry

end sum_of_digits_b_n_l229_229727


namespace area_of_shaded_region_l229_229000

noncomputable def area_of_triangle_in_hexagon : ℝ :=
  let s := 12 in
  36 * Real.sqrt 3

theorem area_of_shaded_region (s : ℝ) (h_s : s = 12) :
    ∃ (area : ℝ), area = 36 * Real.sqrt 3 :=
by
  use 36 * Real.sqrt 3
  rw h_s
  sorry

end area_of_shaded_region_l229_229000


namespace total_mail_l229_229717

def monday_mail := 65
def tuesday_mail := monday_mail + 10
def wednesday_mail := tuesday_mail - 5
def thursday_mail := wednesday_mail + 15

theorem total_mail : 
  monday_mail + tuesday_mail + wednesday_mail + thursday_mail = 295 := by
  sorry

end total_mail_l229_229717


namespace yeast_population_at_130pm_l229_229062

noncomputable def yeast_population (initial_population : ℕ) (time_increments : ℕ) (growth_factor : ℕ) : ℕ :=
  initial_population * growth_factor ^ time_increments

theorem yeast_population_at_130pm : yeast_population 30 3 3 = 810 :=
by
  sorry

end yeast_population_at_130pm_l229_229062


namespace marbles_difference_l229_229007

theorem marbles_difference {red_marbles blue_marbles : ℕ} 
  (h₁ : red_marbles = 288) (bags_red : ℕ) (h₂ : bags_red = 12) 
  (h₃ : blue_marbles = 243) (bags_blue : ℕ) (h₄ : bags_blue = 9) :
  (blue_marbles / bags_blue) - (red_marbles / bags_red) = 3 :=
by
  sorry

end marbles_difference_l229_229007


namespace find_original_number_l229_229492

theorem find_original_number (x : ℤ) (h : 3 * (2 * x + 9) = 57) : x = 5 := by
  sorry

end find_original_number_l229_229492


namespace percent_increase_decrease_condition_l229_229361

theorem percent_increase_decrease_condition (p q M : ℝ) (hp : 0 < p) (hq : 0 < q) (hM : 0 < M) (hq50 : q < 50) :
  (M * (1 + p / 100) * (1 - q / 100) < M) ↔ (p < 100 * q / (100 - q)) := 
sorry

end percent_increase_decrease_condition_l229_229361


namespace question1_is_random_event_question2_probability_xiuShui_l229_229139

-- Definitions for projects
inductive Project
| A | B | C | D

-- Definition for the problem context and probability computation
def xiuShuiProjects : List Project := [Project.A, Project.B]
def allProjects : List Project := [Project.A, Project.B, Project.C, Project.D]

-- Question 1
def isRandomEvent (event : Project) : Prop :=
  event = Project.C ∧ event ∈ allProjects

theorem question1_is_random_event : isRandomEvent Project.C := by
sorry

-- Question 2: Probability both visit Xiu Shui projects is 1/4
def favorable_outcomes : List (Project × Project) :=
  [(Project.A, Project.A), (Project.A, Project.B), (Project.B, Project.A), (Project.B, Project.B)]

def total_outcomes : List (Project × Project) :=
  List.product allProjects allProjects

def probability (fav : ℕ) (total : ℕ) : ℚ := fav / total

theorem question2_probability_xiuShui : probability favorable_outcomes.length total_outcomes.length = 1 / 4 := by
sorry

end question1_is_random_event_question2_probability_xiuShui_l229_229139


namespace Johnson_Martinez_tied_at_end_of_september_l229_229269

open Nat

-- Define the monthly home runs for Johnson and Martinez
def Johnson_runs : List Nat := [3, 8, 15, 12, 5, 7, 14]
def Martinez_runs : List Nat := [0, 3, 9, 20, 7, 12, 13]

-- Define the cumulated home runs for Johnson and Martinez over the months
def total_runs (runs : List Nat) : List Nat :=
  runs.scanl (· + ·) 0

-- State the theorem to prove that they are tied in total runs at the end of September
theorem Johnson_Martinez_tied_at_end_of_september :
  (total_runs Johnson_runs).getLast (by decide) =
  (total_runs Martinez_runs).getLast (by decide) := by
  sorry

end Johnson_Martinez_tied_at_end_of_september_l229_229269


namespace smallest_four_digit_in_pascal_l229_229433

-- Pascal's triangle definition
def pascal (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

-- Condition definition
def exists_in_pascal (m : ℕ) : Prop :=
  ∃ n k, pascal n k = m

theorem smallest_four_digit_in_pascal :
  ∃ n k, pascal n k = 1000 ∧ ∀ m, (exists_in_pascal m) → (m >= 1000) → (m = 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_l229_229433


namespace algebraic_expression_value_l229_229663

-- Definitions for the problem conditions
def x := -1
def y := 1 / 2
def expr := 2 * (x^2 - 5 * x * y) - 3 * (x^2 - 6 * x * y)

-- The problem statement to be proved
theorem algebraic_expression_value : expr = 3 :=
by
  sorry

end algebraic_expression_value_l229_229663


namespace cone_radius_correct_l229_229001

noncomputable def cone_radius (CSA l : ℝ) : ℝ := CSA / (Real.pi * l)

theorem cone_radius_correct :
  cone_radius 1539.3804002589986 35 = 13.9 :=
by
  -- Proof omitted
  sorry

end cone_radius_correct_l229_229001


namespace largest_house_number_l229_229737

theorem largest_house_number (house_num : ℕ) : 
  house_num ≤ 981 :=
  sorry

end largest_house_number_l229_229737


namespace winnie_servings_l229_229763

theorem winnie_servings:
  ∀ (x : ℝ), 
  (2 / 5) * x + (21 / 25) * x = 82 →
  x = 30 :=
by
  sorry

end winnie_servings_l229_229763


namespace smallest_four_digit_number_in_pascals_triangle_l229_229434

theorem smallest_four_digit_number_in_pascals_triangle : ∃ n k, binom n k = 1000 ∧ (∀ n' k', binom n' k' > 1000 → nat.zero < n' k' ∧ n' k' < (binom n k)) := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l229_229434


namespace proof_triangle_inequality_l229_229240

noncomputable def proof_statement (a b c: ℝ) (h: a + b > c ∧ b + c > a ∧ c + a > b) : Prop :=
  a * b * c ≥ (-a + b + c) * (a - b + c) * (a + b - c)

-- Proof statement without the proof
theorem proof_triangle_inequality (a b c: ℝ) (h: a + b > c ∧ b + c > a ∧ c + a > b) : 
  proof_statement a b c h :=
sorry

end proof_triangle_inequality_l229_229240


namespace final_price_correct_l229_229263

noncomputable def final_price_per_litre : Real :=
  let cost_1 := 70 * 43 * (1 - 0.15)
  let cost_2 := 50 * 51 * (1 + 0.10)
  let cost_3 := 15 * 60 * (1 - 0.08)
  let cost_4 := 25 * 62 * (1 + 0.12)
  let cost_5 := 40 * 67 * (1 - 0.05)
  let cost_6 := 10 * 75 * (1 - 0.18)
  let total_cost := cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6
  let total_volume := 70 + 50 + 15 + 25 + 40 + 10
  total_cost / total_volume

theorem final_price_correct : final_price_per_litre = 52.80 := by
  sorry

end final_price_correct_l229_229263


namespace no_prime_divisible_by_77_l229_229693

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_77 : ∀ p : ℕ, is_prime p → ¬ (77 ∣ p) :=
by
  sorry

end no_prime_divisible_by_77_l229_229693


namespace tan_pi_over_4_plus_alpha_l229_229536

theorem tan_pi_over_4_plus_alpha (α : ℝ) 
  (h : Real.tan (Real.pi / 4 + α) = 2) : 
  1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 5 / 7 := 
by {
  sorry
}

end tan_pi_over_4_plus_alpha_l229_229536


namespace students_bought_pencils_l229_229755

theorem students_bought_pencils (h1 : 2 * 2 + 6 * 3 + 2 * 1 = 24) : 
  2 + 6 + 2 = 10 := by
  sorry

end students_bought_pencils_l229_229755


namespace committee_probability_l229_229489

theorem committee_probability :
  let total_ways := Nat.choose 30 6
  let all_boys_ways := Nat.choose 12 6
  let all_girls_ways := Nat.choose 18 6
  let complementary_prob := (all_boys_ways + all_girls_ways) / total_ways
  let desired_prob := 1 - complementary_prob
  desired_prob = (574287 : ℚ) / 593775 :=
by
  sorry

end committee_probability_l229_229489


namespace tan_alpha_eq_neg_one_l229_229540

-- Define the point P and the angle α
def P : ℝ × ℝ := (-1, 1)
def α : ℝ := sorry  -- α is the angle whose terminal side passes through P

-- Statement to be proved
theorem tan_alpha_eq_neg_one (h : (P.1, P.2) = (-1, 1)) : Real.tan α = -1 :=
by
  sorry

end tan_alpha_eq_neg_one_l229_229540


namespace smallest_pos_int_ending_in_9_divisible_by_13_l229_229020

theorem smallest_pos_int_ending_in_9_divisible_by_13 :
  ∃ (n : ℕ), n % 10 = 9 ∧ n % 13 = 0 ∧ ∀ m, m % 10 = 9 ∧ m % 13 = 0 → n ≤ m :=
by
  sorry

end smallest_pos_int_ending_in_9_divisible_by_13_l229_229020


namespace find_original_number_l229_229493

theorem find_original_number (x : ℕ) (h : 3 * (2 * x + 9) = 57) : x = 5 := 
by sorry

end find_original_number_l229_229493


namespace average_score_remaining_students_l229_229308

theorem average_score_remaining_students (n : ℕ) (h : n > 15) (avg_all : ℚ) (avg_15 : ℚ) :
  avg_all = 12 → avg_15 = 20 →
  (∃ avg_remaining : ℚ, avg_remaining = (12 * n - 300) / (n - 15)) :=
by
  sorry

end average_score_remaining_students_l229_229308


namespace john_sarah_money_total_l229_229565

theorem john_sarah_money_total (j_money s_money : ℚ) (H1 : j_money = 5/8) (H2 : s_money = 7/16) :
  (j_money + s_money : ℚ) = 1.0625 := 
by
  sorry

end john_sarah_money_total_l229_229565


namespace avg_service_hours_is_17_l229_229158

-- Define the number of students and their corresponding service hours
def num_students : ℕ := 10
def students_15_hours : ℕ := 2
def students_16_hours : ℕ := 5
def students_20_hours : ℕ := 3

-- Define the service hours corresponding to each group
def service_hours_15 : ℕ := 15
def service_hours_16 : ℕ := 16
def service_hours_20 : ℕ := 20

-- Calculate the total service hours
def total_service_hours : ℕ := 
  (service_hours_15 * students_15_hours) + 
  (service_hours_16 * students_16_hours) + 
  (service_hours_20 * students_20_hours)

-- Average service hours calculation, cast to rational for precise division 
def average_service_hours : ℚ :=
  (total_service_hours : ℚ) / num_students

-- Statement of the theorem
theorem avg_service_hours_is_17 : average_service_hours = 17 := by
  sorry

end avg_service_hours_is_17_l229_229158


namespace rhombus_longer_diagonal_l229_229637

theorem rhombus_longer_diagonal
  (a b : ℝ)
  (ha : a = 65) -- Side length
  (hb : b = 28) -- Half the length of the shorter diagonal
  : 2 * Real.sqrt (a^2 - b^2) = 118 :=  -- Statement of the longer diagonal length
by
  sorry -- Proof goes here

end rhombus_longer_diagonal_l229_229637


namespace tom_blue_marbles_l229_229236

-- Definitions based on conditions
def jason_blue_marbles : Nat := 44
def total_blue_marbles : Nat := 68

-- The problem statement to prove
theorem tom_blue_marbles : (total_blue_marbles - jason_blue_marbles) = 24 :=
by
  sorry

end tom_blue_marbles_l229_229236


namespace distance_reflection_x_axis_l229_229284

/--
Given points C and its reflection over the x-axis C',
prove that the distance between C and C' is 6.
-/
theorem distance_reflection_x_axis :
  let C := (-2, 3)
  let C' := (-2, -3)
  dist C C' = 6 := by
  sorry

end distance_reflection_x_axis_l229_229284


namespace total_numbers_l229_229895

theorem total_numbers (n : ℕ) (a : ℕ → ℝ) 
  (h1 : (a 0 + a 1 + a 2 + a 3) / 4 = 25)
  (h2 : (a (n - 3) + a (n - 2) + a (n - 1)) / 3 = 35)
  (h3 : a 3 = 25)
  (h4 : (Finset.sum (Finset.range n) a) / n = 30) :
  n = 6 :=
sorry

end total_numbers_l229_229895


namespace volume_of_box_l229_229504

noncomputable def volume_expression (y : ℝ) : ℝ :=
  (15 - 2 * y) * (12 - 2 * y) * y

theorem volume_of_box (y : ℝ) :
  volume_expression y = 4 * y^3 - 54 * y^2 + 180 * y :=
by
  sorry

end volume_of_box_l229_229504


namespace upload_time_l229_229714

theorem upload_time (file_size upload_speed : ℕ) (h_file_size : file_size = 160) (h_upload_speed : upload_speed = 8) : file_size / upload_speed = 20 :=
by
  sorry

end upload_time_l229_229714


namespace equal_points_per_person_l229_229878

theorem equal_points_per_person :
  let blue_eggs := 12
  let blue_points := 2
  let pink_eggs := 5
  let pink_points := 3
  let golden_eggs := 3
  let golden_points := 5
  let total_people := 4
  (blue_eggs * blue_points + pink_eggs * pink_points + golden_eggs * golden_points) / total_people = 13 :=
by
  -- place the steps based on the conditions and calculations
  sorry

end equal_points_per_person_l229_229878


namespace smallest_four_digit_in_pascals_triangle_l229_229441

/-- Pascal's triangle is defined using binomial coefficients.
    We want to prove that the smallest four-digit number in Pascal's triangle is 1001. -/
theorem smallest_four_digit_in_pascals_triangle : ∃ n k : ℕ, n ≥ k ∧ (binomial n k = 1001 ∧ ∀ m l, m ≥ l → binomial m l ≥ 1000 → binomial m l = 1001 → n = m ∧ k = l) := by
  sorry

end smallest_four_digit_in_pascals_triangle_l229_229441


namespace watermelon_price_in_units_of_1000_l229_229363

theorem watermelon_price_in_units_of_1000
  (initial_price discounted_price: ℝ)
  (h_price: initial_price = 5000)
  (h_discount: discounted_price = initial_price - 200) :
  discounted_price / 1000 = 4.8 :=
by
  sorry

end watermelon_price_in_units_of_1000_l229_229363


namespace profit_share_difference_l229_229938

theorem profit_share_difference (P : ℝ) (hP : P = 1000) 
  (rX rY : ℝ) (hRatio : rX / rY = (1/2) / (1/3)) : 
  let total_parts := (1/2) + (1/3)
  let value_per_part := P / total_parts
  let x_share := (1/2) * value_per_part
  let y_share := (1/3) * value_per_part
  x_share - y_share = 200 := by 
  sorry

end profit_share_difference_l229_229938


namespace tan_pi_over_4_plus_alpha_l229_229537

theorem tan_pi_over_4_plus_alpha (α : ℝ) 
  (h : Real.tan (Real.pi / 4 + α) = 2) : 
  1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 5 / 7 := 
by {
  sorry
}

end tan_pi_over_4_plus_alpha_l229_229537


namespace yard_length_eq_250_l229_229375

noncomputable def number_of_trees : ℕ := 26
noncomputable def distance_between_trees : ℕ := 10
noncomputable def number_of_gaps := number_of_trees - 1
noncomputable def length_of_yard := number_of_gaps * distance_between_trees

theorem yard_length_eq_250 : 
  length_of_yard = 250 := 
sorry

end yard_length_eq_250_l229_229375


namespace uncovered_side_length_l229_229053

theorem uncovered_side_length {L W : ℕ} (h1 : L * W = 680) (h2 : 2 * W + L = 74) : L = 40 :=
sorry

end uncovered_side_length_l229_229053


namespace combined_class_average_score_l229_229319

theorem combined_class_average_score
  (avg_A : ℕ := 65) (avg_B : ℕ := 90) (avg_C : ℕ := 77)
  (ratio_A : ℕ := 4) (ratio_B : ℕ := 6) (ratio_C : ℕ := 5) :
  ((avg_A * ratio_A + avg_B * ratio_B + avg_C * ratio_C) / (ratio_A + ratio_B + ratio_C) = 79) :=
by 
  sorry

end combined_class_average_score_l229_229319


namespace jim_age_l229_229876

variable (J F S : ℕ)

theorem jim_age (h1 : J = 2 * F) (h2 : F = S + 9) (h3 : J - 6 = 5 * (S - 6)) : J = 46 := 
by
  sorry

end jim_age_l229_229876


namespace problem_l229_229804

def f : ℕ → ℕ → ℕ := sorry

theorem problem (m n : ℕ) (h1 : m ≥ 1) (h2 : n ≥ 1) :
  2 * f m n = 2 + f (m + 1) (n - 1) + f (m - 1) (n + 1) ∧
  (f m 0 = 0) ∧ (f 0 n = 0) → f m n = m * n :=
by sorry

end problem_l229_229804


namespace sum_of_series_is_918_l229_229004

-- Define the first term a, common difference d, last term a_n,
-- and the number of terms n calculated from the conditions.
def first_term : Int := -300
def common_difference : Int := 3
def last_term : Int := 309
def num_terms : Int := 204 -- calculated as per the solution

-- Compute the sum of the arithmetic series
def sum_arithmetic_series (a d : Int) (n : Int) : Int :=
  n * (2 * a + (n - 1) * d) / 2

-- Prove that the sum of the series is 918
theorem sum_of_series_is_918 :
  sum_arithmetic_series first_term common_difference num_terms = 918 :=
by
  sorry

end sum_of_series_is_918_l229_229004


namespace square_root_25_pm5_l229_229592

-- Define that a number x satisfies the equation x^2 = 25
def square_root_of_25 (x : ℝ) : Prop := x * x = 25

-- The theorem states that the square root of 25 is ±5
theorem square_root_25_pm5 : ∀ x : ℝ, square_root_of_25 x ↔ x = 5 ∨ x = -5 :=
by
  intros x
  sorry

end square_root_25_pm5_l229_229592


namespace max_regions_7_dots_l229_229619

-- Definitions based on conditions provided.
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def R (n : ℕ) : ℕ := 1 + binom n 2 + binom n 4

-- The goal is to state the proposition that the maximum number of regions created by joining 7 dots on a circle is 57.
theorem max_regions_7_dots : R 7 = 57 :=
by
  -- The proof is to be filled in here
  sorry

end max_regions_7_dots_l229_229619


namespace perimeter_shaded_area_is_942_l229_229789

-- Definition involving the perimeter of the shaded area of the circles
noncomputable def perimeter_shaded_area (s : ℝ) : ℝ := 
  4 * 75 * 3.14

-- Main theorem stating that if the side length of the octagon is 100 cm,
-- then the perimeter of the shaded area is 942 cm.
theorem perimeter_shaded_area_is_942 :
  perimeter_shaded_area 100 = 942 := 
  sorry

end perimeter_shaded_area_is_942_l229_229789


namespace ellipse_equation_line_l_existence_chord_parallel_constant_l229_229208

noncomputable def ellipse := 
{ x y : ℝ // (x^2)/4 + (y^2)/3 = 1 }

theorem ellipse_equation :
  ∃ (C : set (ℝ × ℝ)), 
    (∀ p ∈ C, p.1^2 / 4 + p.2^2 / 3 = 1) :=
begin
  use {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1},
  simp,
end

theorem line_l_existence (l : ℝ → ℝ) :
  ∃ k : ℝ, (k ≠ 0) ∧ (∀ p ∈ set_of (λ (x : ℝ), l (x - 1) = x^2 / 4 + (l x)^2 / 3 = 1),
    let (x1, y1) := p in let (x2, y2) := -p in
      x1 * x2 + y1 * y2 = -2) ∧ 
      (l = λ x, sqrt 2 * (x - 1) ∨ l = λ x, -sqrt 2 * (x - 1)) :=
begin
  existsi sqrt 2,
  split,
  { exact sqrt 2_ne_zero },
  split,
  { sorry },  -- Proof of this step is omitted
  {
    split,
    { rw ← sqrt_sqr (sqrt 2),
      exact (mul_pos (sqrt_pos.mpr zero_lt_two) (sub_pos.mpr zero_lt_one)).ne' },
    { rw ← sqrt_sqr (sqrt 2),
      exact (mul_neg (sqrt_pos.mpr zero_lt_two).ne' (sub_pos.mpr zero_lt_one)).ne' }
  },
end

theorem chord_parallel_constant (AB MN : set (ℝ × ℝ)) :
  (∀ (O : ℝ × ℝ) (AB_par : ∀ (x y : ℝ), (x^2 + y^2) = AB_par) (MN_par : ∀ (x y : ℝ), (x^2 + y^2) = MN)
    (k : ℝ) (h_AB_par : ∀ p ∈ AB, p.1^2 / 4 + p.2^2 / 3 = 1)
    (h_MN_par : ∀ p ∈ MN, p.1^2 / 4 + p.2^2 / 3 = 1),
    |AB_par / MN_par| = 4) :=
begin
  sorry  -- Proof of this step is omitted
end

end ellipse_equation_line_l_existence_chord_parallel_constant_l229_229208


namespace percent_increase_correct_l229_229339

noncomputable def last_year_ticket_price : ℝ := 85
noncomputable def last_year_tax_rate : ℝ := 0.10
noncomputable def this_year_ticket_price : ℝ := 102
noncomputable def this_year_tax_rate : ℝ := 0.12
noncomputable def student_discount_rate : ℝ := 0.15

noncomputable def last_year_total_cost : ℝ := last_year_ticket_price * (1 + last_year_tax_rate)
noncomputable def discounted_ticket_price_this_year : ℝ := this_year_ticket_price * (1 - student_discount_rate)
noncomputable def total_cost_this_year : ℝ := discounted_ticket_price_this_year * (1 + this_year_tax_rate)

noncomputable def percent_increase : ℝ := ((total_cost_this_year - last_year_total_cost) / last_year_total_cost) * 100

theorem percent_increase_correct :
  abs (percent_increase - 3.854) < 0.001 := sorry

end percent_increase_correct_l229_229339


namespace three_point_seven_five_minus_one_point_four_six_l229_229648

theorem three_point_seven_five_minus_one_point_four_six : 3.75 - 1.46 = 2.29 :=
by sorry

end three_point_seven_five_minus_one_point_four_six_l229_229648


namespace count_positive_integers_m_l229_229547

theorem count_positive_integers_m :
  ∃ m_values : Finset ℕ, m_values.card = 4 ∧ ∀ m ∈ m_values, 
    ∃ k : ℕ, k > 0 ∧ (7 * m + 2 = m * k + 2 * m) := 
sorry

end count_positive_integers_m_l229_229547


namespace cistern_width_l229_229157

theorem cistern_width (w : ℝ) (h : 8 * w + 2 * (1.25 * 8) + 2 * (1.25 * w) = 83) : w = 6 :=
by
  sorry

end cistern_width_l229_229157


namespace factorize_binomial_square_l229_229979

theorem factorize_binomial_square (x y : ℝ) : x^2 + 2*x*y + y^2 = (x + y)^2 :=
by
  sorry

end factorize_binomial_square_l229_229979


namespace smallest_positive_integer_ending_in_9_divisible_by_13_l229_229026

theorem smallest_positive_integer_ending_in_9_divisible_by_13 :
  ∃ n : ℕ, (n % 10 = 9) ∧ (n % 13 = 0) ∧ (∀ m : ℕ, (m % 10 = 9) ∧ (m % 13 = 0) → m ≥ n) :=
sorry

end smallest_positive_integer_ending_in_9_divisible_by_13_l229_229026


namespace minimum_value_of_y_l229_229551

theorem minimum_value_of_y (x : ℝ) (h : x > 2) : 
  ∃ y, y = x + 4 / (x - 2) ∧ y = 6 :=
by
  sorry

end minimum_value_of_y_l229_229551


namespace smallest_integer_consecutive_set_l229_229706

theorem smallest_integer_consecutive_set :
  ∃ m : ℤ, (m+3 < 3*m - 5) ∧ (∀ n : ℤ, (n+3 < 3*n - 5) → n ≥ m) ∧ m = 5 :=
by
  sorry

end smallest_integer_consecutive_set_l229_229706


namespace expand_expression_l229_229189

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  sorry

end expand_expression_l229_229189


namespace find_x_eq_2_l229_229936

theorem find_x_eq_2 (x : ℕ) (h : 7899665 - 36 * x = 7899593) : x = 2 := 
by
  sorry

end find_x_eq_2_l229_229936


namespace problem_l229_229841

theorem problem (x y : ℝ)
  (h1 : 1 / x + 1 / y = 4)
  (h2 : 1 / x - 1 / y = -6) :
  x + y = -4 / 5 :=
sorry

end problem_l229_229841


namespace min_groups_required_l229_229046

/-!
  Prove that if a coach has 30 athletes and wants to arrange them into equal groups with no more than 12 athletes each, 
  then the minimum number of groups required is 3.
-/

theorem min_groups_required (total_athletes : ℕ) (max_athletes_per_group : ℕ) (h_total : total_athletes = 30) (h_max : max_athletes_per_group = 12) :
  ∃ (min_groups : ℕ), min_groups = total_athletes / 10 ∧ (total_athletes % 10 = 0) := by
  sorry

end min_groups_required_l229_229046


namespace jill_draws_spade_probability_l229_229713

/-- Given the conditions of the card game, we want to prove that the probability that Jill 
    draws the spade is 12/37. -/
theorem jill_draws_spade_probability :
  let jack_spade_prob : ℚ := 1 / 4
      jill_spade_prob : ℚ := (3 / 4) * (1 / 4)
      john_spade_prob : ℚ := ((3 / 4) * (3 / 4)) * (1 / 4)
      total_spade_prob : ℚ := jack_spade_prob + jill_spade_prob + john_spade_prob
      jill_conditional_prob : ℚ := jill_spade_prob / total_spade_prob
  in
  jill_conditional_prob = 12 / 37 :=
by
  sorry

end jill_draws_spade_probability_l229_229713


namespace find_original_number_l229_229494

theorem find_original_number (x : ℕ) (h : 3 * (2 * x + 9) = 57) : x = 5 := 
by sorry

end find_original_number_l229_229494


namespace trench_dig_time_l229_229821

theorem trench_dig_time (a b c d : ℝ) (h1 : a + b + c + d = 1/6)
  (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
  (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4) :
  a + b + c = 1 / 6 := sorry

end trench_dig_time_l229_229821


namespace arithmetic_geo_sum_l229_229993

theorem arithmetic_geo_sum (a : ℕ → ℤ) (d : ℤ) :
  (∀ n, a (n + 1) = a n + d) →
  (d = 2) →
  (a 3) ^ 2 = (a 1) * (a 4) →
  (a 2 + a 3 = -10) := 
by
  intros h_arith h_d h_geo
  sorry

end arithmetic_geo_sum_l229_229993


namespace inversely_proportional_find_p_l229_229396

theorem inversely_proportional_find_p (p q : ℕ) (h1 : p * 8 = 160) (h2 : q = 10) : p * q = 160 → p = 16 :=
by
  intro h
  sorry

end inversely_proportional_find_p_l229_229396


namespace smallest_pos_int_for_congruence_l229_229474

theorem smallest_pos_int_for_congruence :
  ∃ (n : ℕ), 5 * n % 33 = 980 % 33 ∧ n > 0 ∧ n = 19 := 
by {
  sorry
}

end smallest_pos_int_for_congruence_l229_229474


namespace candy_problem_l229_229740

theorem candy_problem (N a S : ℕ) 
  (h1 : ∀ i : ℕ, i < N → a = S - 7 - a)
  (h2 : ∀ i : ℕ, i < N → a > 1)
  (h3 : S = N * a) : 
  S = 21 :=
by
  sorry

end candy_problem_l229_229740


namespace find_new_person_age_l229_229267

variables (A X : ℕ) -- A is the original average age, X is the age of the new person

def original_total_age (A : ℕ) := 10 * A
def new_total_age (A X : ℕ) := 10 * (A - 3)

theorem find_new_person_age (A : ℕ) (h : new_total_age A X = original_total_age A - 45 + X) : X = 15 :=
by
  sorry

end find_new_person_age_l229_229267


namespace greatest_drop_in_price_is_august_l229_229137

-- Define the months and their respective price changes
def price_changes : List (String × ℝ) :=
  [("January", -1.00), ("February", 1.50), ("March", -3.00), ("April", 2.50), 
   ("May", -0.75), ("June", -2.25), ("July", 1.00), ("August", -4.00)]

-- Define the statement that August has the greatest drop in price
theorem greatest_drop_in_price_is_august :
  ∀ month ∈ price_changes, month.snd ≤ -4.00 → month.fst = "August" :=
by
  sorry

end greatest_drop_in_price_is_august_l229_229137


namespace units_digit_27_mul_46_l229_229983

-- Define the function to calculate the units digit of a number
def units_digit (n : ℕ) : ℕ :=
  n % 10

-- Problem statement: The units digit of 27 * 46 is 2
theorem units_digit_27_mul_46 : units_digit (27 * 46) = 2 :=
  sorry

end units_digit_27_mul_46_l229_229983


namespace interval_of_increase_of_f_l229_229657

noncomputable def f (x : ℝ) := Real.logb (0.5) (x - x^2)

theorem interval_of_increase_of_f :
  ∀ x : ℝ, x ∈ Set.Ioo (1/2) 1 → ∃ ε > 0, ∀ y : ℝ, y ∈ Set.Ioo (x - ε) (x + ε) → f y > f x :=
  by
    sorry

end interval_of_increase_of_f_l229_229657


namespace smallest_four_digit_in_pascal_l229_229462

theorem smallest_four_digit_in_pascal :
  ∃ n k : ℕ, binomial n k = 1000 ∧ 
             (∀ a b : ℕ, binomial a b ∈ set.Icc 1000 9999 → a ≥ n) :=
sorry

end smallest_four_digit_in_pascal_l229_229462


namespace probability_of_diff_3_is_1_over_9_l229_229947

theorem probability_of_diff_3_is_1_over_9 :
  let outcomes := [(a, b) | a in [1, 2, 3, 4, 5, 6], b in [1, 2, 3, 4, 5, 6]],
      valid_pairs := [(2, 5), (3, 6), (4, 1), (5, 2)],
      total_outcomes := 36,
      successful_outcomes := 4
  in
  successful_outcomes.to_rat / total_outcomes.to_rat = 1 / 9 := 
  sorry

end probability_of_diff_3_is_1_over_9_l229_229947


namespace smallest_four_digit_in_pascals_triangle_l229_229446

theorem smallest_four_digit_in_pascals_triangle : ∃ n, ∃ k, k ≤ n ∧ 1000 ≤ Nat.choose n k :=
sorry

end smallest_four_digit_in_pascals_triangle_l229_229446


namespace range_of_m_necessary_condition_range_of_m_not_sufficient_condition_l229_229205

-- Problem I Statement
theorem range_of_m_necessary_condition (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 8 * x - 20 ≤ 0) → (1 - m^2 ≤ x ∧ x ≤ 1 + m^2)) →
  (-Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3) :=
by sorry

-- Problem II Statement
theorem range_of_m_not_sufficient_condition (m : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 8 * x - 20 ≤ 0) → ¬(1 - m^2 ≤ x ∧ x ≤ 1 + m^2)) →
  (m ≤ -3 ∨ m ≥ 3) :=
by sorry

end range_of_m_necessary_condition_range_of_m_not_sufficient_condition_l229_229205


namespace professor_has_to_grade_405_more_problems_l229_229940

theorem professor_has_to_grade_405_more_problems
  (problems_per_paper : ℕ)
  (total_papers : ℕ)
  (graded_papers : ℕ)
  (remaining_papers := total_papers - graded_papers)
  (p : ℕ := remaining_papers * problems_per_paper) :
  problems_per_paper = 15 ∧ total_papers = 45 ∧ graded_papers = 18 → p = 405 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end professor_has_to_grade_405_more_problems_l229_229940


namespace tires_in_parking_lot_l229_229857

theorem tires_in_parking_lot (n : ℕ) (m : ℕ) (h : 30 = n) (h' : m = 5) : n * m = 150 := by
  sorry

end tires_in_parking_lot_l229_229857


namespace at_least_one_negative_l229_229367

theorem at_least_one_negative (a b : ℝ) (h : a + b < 0) : a < 0 ∨ b < 0 := by
  sorry

end at_least_one_negative_l229_229367


namespace distance_between_5th_and_29th_red_light_in_feet_l229_229584

-- Define the repeating pattern length and individual light distance
def pattern_length := 7
def red_light_positions := {k | k % pattern_length < 3}
def distance_between_lights := 8 / 12  -- converting inches to feet

-- Positions of the 5th and 29th red lights in terms of pattern repetition
def position_of_nth_red_light (n : ℕ) : ℕ :=
  ((n-1) / 3) * pattern_length + (n-1) % 3 + 1

def position_5th_red_light := position_of_nth_red_light 5
def position_29th_red_light := position_of_nth_red_light 29

theorem distance_between_5th_and_29th_red_light_in_feet :
  (position_29th_red_light - position_5th_red_light - 1) * distance_between_lights = 37 := by
  sorry

end distance_between_5th_and_29th_red_light_in_feet_l229_229584


namespace points_on_opposite_sides_of_line_l229_229683

theorem points_on_opposite_sides_of_line (a : ℝ) :
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) < 0 ↔ -7 < a ∧ a < 24 :=
by 
  sorry

end points_on_opposite_sides_of_line_l229_229683


namespace repeated_1991_mod_13_l229_229206

theorem repeated_1991_mod_13 (k : ℕ) : 
  ((10^4 - 9) * (1991 * (10^(4*k) - 1)) / 9) % 13 = 8 :=
by
  sorry

end repeated_1991_mod_13_l229_229206


namespace fiona_considers_pairs_l229_229346

theorem fiona_considers_pairs : 
  ∀ (students : ℕ) (ignored_pairs : ℕ), students = 12 → ignored_pairs = 3 → (Nat.choose students 2 - ignored_pairs) = 63 :=
by
  intros students ignored_pairs h1 h2
  rw [h1, h2]
  exact (by decide : (Nat.choose 12 2 - 3) = 63)
  sorry

end fiona_considers_pairs_l229_229346


namespace total_units_in_building_l229_229931

theorem total_units_in_building (x y : ℕ) (cost_1_bedroom cost_2_bedroom total_cost : ℕ)
  (h1 : cost_1_bedroom = 360) (h2 : cost_2_bedroom = 450)
  (h3 : total_cost = 4950) (h4 : y = 7) (h5 : total_cost = cost_1_bedroom * x + cost_2_bedroom * y) :
  x + y = 12 :=
sorry

end total_units_in_building_l229_229931


namespace yellow_balls_l229_229102

theorem yellow_balls (total_balls : ℕ) (prob_yellow : ℚ) (x : ℕ) :
  total_balls = 40 ∧ prob_yellow = 0.30 → (x : ℚ) = 12 := 
by 
  sorry

end yellow_balls_l229_229102


namespace production_line_B_units_l229_229160

theorem production_line_B_units (total_units : ℕ) (A_units B_units C_units : ℕ) 
  (h1 : total_units = 16800)
  (h2 : ∃ d : ℕ, A_units + d = B_units ∧ B_units + d = C_units) :
  B_units = 5600 := 
sorry

end production_line_B_units_l229_229160


namespace distinct_arrangements_BOOKKEEPER_l229_229519

theorem distinct_arrangements_BOOKKEEPER :
  let n := 9
  let nO := 2
  let nK := 2
  let nE := 3
  ∃ arrangements : ℕ,
  arrangements = Nat.factorial n / (Nat.factorial nO * Nat.factorial nK * Nat.factorial nE) ∧
  arrangements = 15120 :=
by { sorry }

end distinct_arrangements_BOOKKEEPER_l229_229519


namespace geometric_sequence_a5_eq_8_l229_229562

variable (a : ℕ → ℝ)
variable (n : ℕ)

-- Conditions
axiom pos (n : ℕ) : a n > 0
axiom prod_eq (a3 a7 : ℝ) : a 3 * a 7 = 64

-- Statement to prove
theorem geometric_sequence_a5_eq_8
  (pos : ∀ n, a n > 0)
  (prod_eq : a 3 * a 7 = 64) :
  a 5 = 8 :=
sorry

end geometric_sequence_a5_eq_8_l229_229562


namespace part_a_part_b_part_c_part_d_part_e_part_f_part_g_part_h_part_i_part_j_part_k_part_m_l229_229180

open Real

variables (a b c d : ℝ)

-- Assumptions
axiom a_neg : a < 0
axiom b_neg : b < 0
axiom c_pos : 0 < c
axiom d_pos : 0 < d
axiom abs_conditions : (0 < abs c) ∧ (abs c < 1) ∧ (abs b < 2) ∧ (1 < abs b) ∧ (1 < abs d) ∧ (abs d < 2) ∧ (abs a < 4) ∧ (2 < abs a)

-- Theorem Statements
theorem part_a : abs a < 4 := sorry
theorem part_b : abs b < 2 := sorry
theorem part_c : abs c < 2 := sorry
theorem part_d : abs a > abs b := sorry
theorem part_e : abs c < abs d := sorry
theorem part_f : ¬ (abs a < abs d) := sorry
theorem part_g : abs (a - b) < 4 := sorry
theorem part_h : ¬ (abs (a - b) ≥ 3) := sorry
theorem part_i : ¬ (abs (c - d) < 1) := sorry
theorem part_j : abs (b - c) < 2 := sorry
theorem part_k : ¬ (abs (b - c) > 3) := sorry
theorem part_m : abs (c - a) > 1 := sorry

end part_a_part_b_part_c_part_d_part_e_part_f_part_g_part_h_part_i_part_j_part_k_part_m_l229_229180


namespace evaluate_expression_l229_229521

def improper_fraction (n : Int) (a : Int) (b : Int) : Rat :=
  n + (a : Rat) / b

def expression (x : Rat) : Rat :=
  (x * 1.65 - x + (7 / 20) * x) * 47.5 * 0.8 * 2.5

theorem evaluate_expression : 
  expression (improper_fraction 20 94 95) = 1994 := 
by 
  sorry

end evaluate_expression_l229_229521


namespace scott_monthly_miles_l229_229390

theorem scott_monthly_miles :
  let miles_per_mon_wed := 3
  let mon_wed_days := 3
  let thur_fri_factor := 2
  let thur_fri_days := 2
  let weeks_per_month := 4
  let miles_mon_wed := miles_per_mon_wed * mon_wed_days
  let miles_thur_fri_per_day := thur_fri_factor * miles_per_mon_wed
  let miles_thur_fri := miles_thur_fri_per_day * thur_fri_days
  let miles_per_week := miles_mon_wed + miles_thur_fri
  let total_miles_in_month := miles_per_week * weeks_per_month
  total_miles_in_month = 84 := 
  by
    sorry

end scott_monthly_miles_l229_229390


namespace junk_mail_per_house_l229_229773

theorem junk_mail_per_house (total_junk_mail : ℕ) (houses_per_block : ℕ) 
  (h1 : total_junk_mail = 14) (h2 : houses_per_block = 7) : 
  (total_junk_mail / houses_per_block) = 2 :=
by 
  sorry

end junk_mail_per_house_l229_229773


namespace problem_statement_l229_229538

variable {α : Type*} [LinearOrder α] [AddCommGroup α] [Nontrivial α]

def is_monotone_increasing (f : α → α) (s : Set α) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x ≤ f y

theorem problem_statement (f : ℝ → ℝ) (x1 x2 : ℝ)
  (h1 : ∀ x, f (-x) = -f (x + 4))
  (h2 : is_monotone_increasing f {x | x > 2})
  (hx1 : x1 < 2) (hx2 : 2 < x2) (h_sum : x1 + x2 < 4) :
  f (x1) + f (x2) < 0 :=
sorry

end problem_statement_l229_229538


namespace expansive_sequence_in_interval_l229_229016

-- Definition of an expansive sequence
def expansive_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (i j : ℕ), (i < j) → (|a i - a j| ≥ 1 / j)

-- Upper bound condition for C
def upper_bound_C (C : ℝ) : Prop :=
  C ≥ 2 * Real.log 2

-- The main statement combining both definitions into a proof problem
theorem expansive_sequence_in_interval (C : ℝ) (a : ℕ → ℝ) 
  (h_exp : expansive_sequence a) (h_bound : upper_bound_C C) :
  ∀ n, 0 ≤ a n ∧ a n ≤ C :=
sorry

end expansive_sequence_in_interval_l229_229016


namespace smallest_four_digit_number_in_pascals_triangle_l229_229427

theorem smallest_four_digit_number_in_pascals_triangle :
  (pascal : ℕ → ℕ → ℕ) 
  (h_start : ∀ n, pascal n 0 = 1) 
  (h_increase : ∀ n k, pascal n k ≤ pascal n (k + 1)) : 
  ∃ n k, pascal n k = 1000 := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l229_229427


namespace average_cost_correct_l229_229772

-- Defining the conditions
def groups_of_4_oranges := 11
def cost_of_4_oranges_bundle := 15
def groups_of_7_oranges := 2
def cost_of_7_oranges_bundle := 25

-- Calculating the relevant quantities as per the conditions
def total_cost : ℕ := (groups_of_4_oranges * cost_of_4_oranges_bundle) + (groups_of_7_oranges * cost_of_7_oranges_bundle)
def total_oranges : ℕ := (groups_of_4_oranges * 4) + (groups_of_7_oranges * 7)
def average_cost_per_orange := (total_cost:ℚ) / (total_oranges:ℚ)

-- Proving the average cost per orange matches the correct answer
theorem average_cost_correct : average_cost_per_orange = 215 / 58 := by
  sorry

end average_cost_correct_l229_229772


namespace glen_pop_l229_229563

/-- In the village of Glen, the total population can be formulated as 21h + 6c
given the relationships between people, horses, sheep, cows, and ducks.
We need to prove that 96 cannot be expressed in the form 21h + 6c for
non-negative integers h and c. -/
theorem glen_pop (h c : ℕ) : 21 * h + 6 * c ≠ 96 :=
by
sorry

end glen_pop_l229_229563


namespace probability_of_differ_by_three_l229_229958

def is_valid_roll (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6
def differ_by_three (a b : ℕ) : Prop := abs (a - b) = 3

theorem probability_of_differ_by_three :
  let successful_outcomes := ([
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ] : List (ℕ × ℕ)) in
  let total_outcomes := 6 * 6 in
  (List.length successful_outcomes : ℝ) / total_outcomes = 1 / 6 :=
by
  -- Definitions and assumptions
  let successful_outcomes := [
    (4, 1),
    (5, 2),
    (6, 3),
    (1, 4),
    (2, 5),
    (3, 6)
  ]
  let total_outcomes := 6 * 6
  
  -- Statement of the theorem
  have h_successful : successful_outcomes.length = 6 := sorry
  have h_total : total_outcomes = 36 := by norm_num
  have h_probability := h_successful
    ▸ h_total ▸ (6 / 36 : ℝ) = (1 / 6 : ℝ) := by norm_num
  exact h_probability

end probability_of_differ_by_three_l229_229958


namespace printing_presses_equivalence_l229_229871

theorem printing_presses_equivalence :
  ∃ P : ℕ, (500000 / 12) / P = (500000 / 14) / 30 ∧ P = 26 :=
by
  sorry

end printing_presses_equivalence_l229_229871


namespace probability_of_rolling_integers_with_difference_3_is_1_div_6_l229_229960

noncomputable def rolls_differ_by_three_probability : ℚ :=
  let successful_outcomes := [(2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let total_outcomes := 6 * 6 in
  (successful_outcomes.length : ℚ) / total_outcomes

theorem probability_of_rolling_integers_with_difference_3_is_1_div_6 :
  rolls_differ_by_three_probability = 1 / 6 := by
  sorry

end probability_of_rolling_integers_with_difference_3_is_1_div_6_l229_229960


namespace range_of_m_l229_229530

theorem range_of_m (m : ℝ) (p : |m + 1| ≤ 2) (q : ¬(m^2 - 4 ≥ 0)) : -2 < m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l229_229530


namespace trapezoid_area_l229_229289

noncomputable def area_of_trapezoid : ℝ :=
  let y1 := 12
  let y2 := 5
  let x1 := 12 / 2
  let x2 := 5 / 2
  ((x1 + x2) / 2) * (y1 - y2)

theorem trapezoid_area : area_of_trapezoid = 29.75 := by
  sorry

end trapezoid_area_l229_229289


namespace smallest_four_digit_in_pascals_triangle_l229_229442

/-- Pascal's triangle is defined using binomial coefficients.
    We want to prove that the smallest four-digit number in Pascal's triangle is 1001. -/
theorem smallest_four_digit_in_pascals_triangle : ∃ n k : ℕ, n ≥ k ∧ (binomial n k = 1001 ∧ ∀ m l, m ≥ l → binomial m l ≥ 1000 → binomial m l = 1001 → n = m ∧ k = l) := by
  sorry

end smallest_four_digit_in_pascals_triangle_l229_229442


namespace tan_half_angle_sum_identity_l229_229550

theorem tan_half_angle_sum_identity
  (α β γ : ℝ)
  (h : Real.sin α + Real.sin γ = 2 * Real.sin β) :
  Real.tan ((α + β) / 2) + Real.tan ((β + γ) / 2) = 2 * Real.tan ((γ + α) / 2) :=
sorry

end tan_half_angle_sum_identity_l229_229550


namespace expansion_sum_l229_229369

theorem expansion_sum (A B C : ℤ) (h1 : A = (2 - 1)^10) (h2 : B = (2 + 0)^10) (h3 : C = -5120) : 
A + B + C = -4095 :=
by 
  sorry

end expansion_sum_l229_229369


namespace seq_a_2012_value_l229_229684

theorem seq_a_2012_value :
  ∀ (a : ℕ → ℕ),
  (a 1 = 0) →
  (∀ n : ℕ, a (n + 1) = a n + 2 * n) →
  a 2012 = 2011 * 2012 :=
by
  intros a h₁ h₂
  sorry

end seq_a_2012_value_l229_229684


namespace largest_number_le_1_1_from_set_l229_229862

def is_largest_le (n : ℚ) (l : List ℚ) (bound : ℚ) : Prop :=
  (n ∈ l ∧ n ≤ bound) ∧ ∀ m ∈ l, m ≤ bound → m ≤ n

theorem largest_number_le_1_1_from_set : 
  is_largest_le (9/10) [14/10, 9/10, 12/10, 5/10, 13/10] (11/10) :=
by 
  sorry

end largest_number_le_1_1_from_set_l229_229862


namespace common_ratio_is_0_88_second_term_is_475_2_l229_229058

-- Define the first term and the sum of the infinite geometric series
def first_term : Real := 540
def sum_infinite_series : Real := 4500

-- Required properties of the common ratio
def common_ratio (r : Real) : Prop :=
  abs r < 1 ∧ sum_infinite_series = first_term / (1 - r)

-- Prove the common ratio is 0.88 given the conditions
theorem common_ratio_is_0_88 : ∃ r : Real, common_ratio r ∧ r = 0.88 :=
by 
  sorry

-- Calculate the second term of the series
def second_term (r : Real) : Real := first_term * r

-- Prove the second term is 475.2 given the common ratio is 0.88
theorem second_term_is_475_2 : second_term 0.88 = 475.2 :=
by 
  sorry

end common_ratio_is_0_88_second_term_is_475_2_l229_229058


namespace sum_of_digits_divisible_by_six_l229_229554

theorem sum_of_digits_divisible_by_six (A B : ℕ) (h1 : 10 * A + B % 6 = 0) (h2 : A + B = 12) : A + B = 12 :=
by
  sorry

end sum_of_digits_divisible_by_six_l229_229554


namespace mapping_f_correct_l229_229542

theorem mapping_f_correct (a1 a2 a3 a4 b1 b2 b3 b4 : ℤ) :
  (∀ (x : ℤ), x^4 + a1 * x^3 + a2 * x^2 + a3 * x + a4 = (x + 1)^4 + b1 * (x + 1)^3 + b2 * (x + 1)^2 + b3 * (x + 1) + b4) →
  a1 = 4 → a2 = 3 → a3 = 2 → a4 = 1 →
  b1 = 0 → b1 + b2 + b3 + b4 = 0 →
  (b1, b2, b3, b4) = (0, -3, 4, -1) :=
by
  intros
  sorry

end mapping_f_correct_l229_229542


namespace transistor_length_scientific_notation_l229_229060

theorem transistor_length_scientific_notation :
  0.000000006 = 6 * 10^(-9) := 
sorry

end transistor_length_scientific_notation_l229_229060


namespace inverse_variation_proof_l229_229742

variable (x w : ℝ)

-- Given conditions
def varies_inversely (k : ℝ) : Prop :=
  x^4 * w^(1/4) = k

-- Specific instances
def specific_instance1 : Prop :=
  varies_inversely x w 162 ∧ x = 3 ∧ w = 16

def specific_instance2 : Prop :=
  varies_inversely x w 162 ∧ x = 6 → w = 1/4096

theorem inverse_variation_proof : 
  specific_instance1 → specific_instance2 :=
sorry

end inverse_variation_proof_l229_229742


namespace first_three_workers_dig_time_l229_229818

variables 
  (a b c d : ℝ) -- work rates of the four workers
  (hours : ℝ) -- time to dig the trench

def work_together (a b c d hours : ℝ) := (a + b + c + d) * hours = 1

def scenario1 (a b c d : ℝ) := (2 * a + (1/2) * b + c + d) * 6 = 1

def scenario2 (a b c d : ℝ) := (a/2 + 2 * b + c + d) * 4 = 1

theorem first_three_workers_dig_time
  (h1 : work_together a b c d 6)
  (h2 : scenario1 a b c d)
  (h3 : scenario2 a b c d) :
  hours = 6 := 
sorry

end first_three_workers_dig_time_l229_229818


namespace roots_triangle_ineq_l229_229679

variable {m : ℝ}

def roots_form_triangle (x1 x2 x3 : ℝ) : Prop :=
  x1 + x2 > x3 ∧ x1 + x3 > x2 ∧ x2 + x3 > x1

theorem roots_triangle_ineq (h : ∀ x, (x - 2) * (x^2 - 4*x + m) = 0) :
  3 < m ∧ m < 4 :=
by
  sorry

end roots_triangle_ineq_l229_229679


namespace edward_initial_money_l229_229976

theorem edward_initial_money (cars qty : Nat) (car_cost race_track_cost left_money initial_money : ℝ) 
    (h1 : cars = 4) 
    (h2 : car_cost = 0.95) 
    (h3 : race_track_cost = 6.00)
    (h4 : left_money = 8.00)
    (h5 : initial_money = (cars * car_cost) + race_track_cost + left_money) :
  initial_money = 17.80 := sorry

end edward_initial_money_l229_229976


namespace sum_of_x_and_y_l229_229845

theorem sum_of_x_and_y :
  ∀ (x y : ℚ), (1/x + 1/y = 4) → (1/x - 1/y = -6) → x + y = -4/5 :=
by
  intros x y h1 h2
  sorry

end sum_of_x_and_y_l229_229845


namespace cost_per_book_l229_229601

theorem cost_per_book (num_animal_books : ℕ) (num_space_books : ℕ) (num_train_books : ℕ) (total_cost : ℕ) 
                      (h1 : num_animal_books = 10) (h2 : num_space_books = 1) (h3 : num_train_books = 3) (h4 : total_cost = 224) :
  (total_cost / (num_animal_books + num_space_books + num_train_books) = 16) :=
by sorry

end cost_per_book_l229_229601


namespace no_prime_divisible_by_77_l229_229690

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_77 :
  ∀ p : ℕ, is_prime p → ¬ (77 ∣ p) :=
by
  intros p h_prime h_div
  sorry

end no_prime_divisible_by_77_l229_229690


namespace find_m_l229_229545

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (-1, m)
def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_m (m : ℝ) (h : is_perpendicular vector_a (vector_b m)) : m = 1 / 2 :=
by 
  sorry

end find_m_l229_229545


namespace range_of_omega_l229_229989

noncomputable def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∃ (a b : ℝ), a ≠ b ∧ 0 ≤ a ∧ a ≤ π/2 ∧ 0 ≤ b ∧ b ≤ π/2 ∧ f ω a + f ω b = 4) ↔ 5 ≤ ω ∧ ω < 9 :=
sorry

end range_of_omega_l229_229989


namespace tires_in_parking_lot_l229_229856

theorem tires_in_parking_lot (num_cars : ℕ) (regular_tires_per_car spare_tire : ℕ) (h1 : num_cars = 30) (h2 : regular_tires_per_car = 4) (h3 : spare_tire = 1) :
  num_cars * (regular_tires_per_car + spare_tire) = 150 :=
by
  sorry

end tires_in_parking_lot_l229_229856


namespace difference_fraction_reciprocal_l229_229747

theorem difference_fraction_reciprocal :
  let f := (4 : ℚ) / 5
  let r := (5 : ℚ) / 4
  f - r = 9 / 20 :=
by
  sorry

end difference_fraction_reciprocal_l229_229747


namespace permutations_of_six_digit_number_l229_229222

/-- 
Theorem: The number of distinct permutations of the digits 1, 1, 3, 3, 3, 8 
to form six-digit positive integers is 60. 
-/
theorem permutations_of_six_digit_number : 
  (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 3)) = 60 := 
by 
  sorry

end permutations_of_six_digit_number_l229_229222


namespace penny_half_dollar_same_probability_l229_229127

def probability_penny_half_dollar_same : ℚ :=
  1 / 2

theorem penny_half_dollar_same_probability :
  probability_penny_half_dollar_same = 1 / 2 :=
by
  sorry

end penny_half_dollar_same_probability_l229_229127


namespace rational_solution_unique_l229_229482

theorem rational_solution_unique
  (n : ℕ) (x y : ℚ)
  (hn : Odd n)
  (hx_eqn : x ^ n + 2 * y = y ^ n + 2 * x) :
  x = y :=
sorry

end rational_solution_unique_l229_229482


namespace smallest_four_digit_in_pascals_triangle_l229_229456

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, 1000 ≤ nat.choose n k ∧ nat.choose n k < 10000 ∧ 
  (∀ m l : ℕ, (nat.choose m l ≥ 1000 ∧ nat.choose m l < 10000) → nat.choose n k ≤ nat.choose m l) :=
begin
  -- This is a statement only, no proof is provided
  sorry
end

end smallest_four_digit_in_pascals_triangle_l229_229456


namespace vincent_books_cost_l229_229599

theorem vincent_books_cost :
  let num_animals := 10
  let num_outer_space := 1
  let num_trains := 3
  let total_books := num_animals + num_outer_space + num_trains
  let total_spent := 224
  let cost_per_book := total_spent / total_books
  cost_per_book = 16 :=
by
  let num_animals := 10
  let num_outer_space := 1
  let num_trains := 3
  let total_books := num_animals + num_outer_space + num_trains
  let total_spent := 224
  let cost_per_book := total_spent / total_books
  show cost_per_book = 16
  sorry

end vincent_books_cost_l229_229599


namespace rhombus_longer_diagonal_l229_229633

theorem rhombus_longer_diagonal (a b d : ℝ) (h : a = 65) (h_d : d = 56) :
  ∃ l, l = 118 :=
by
  sorry

end rhombus_longer_diagonal_l229_229633


namespace factory_employees_l229_229558

def num_employees (n12 n14 n17 : ℕ) : ℕ := n12 + n14 + n17

def total_cost (n12 n14 n17 : ℕ) : ℕ := 
    (200 * 12 * 8) + (40 * 14 * 8) + (n17 * 17 * 8)

theorem factory_employees (n17 : ℕ) 
    (h_cost : total_cost 200 40 n17 = 31840) : 
    num_employees 200 40 n17 = 300 := 
by 
    sorry

end factory_employees_l229_229558


namespace length_of_uncovered_side_l229_229050

theorem length_of_uncovered_side (L W : ℕ) (h1 : L * W = 680) (h2 : 2 * W + L = 74) : L = 40 :=
sorry

end length_of_uncovered_side_l229_229050


namespace angle_between_line_and_plane_l229_229561

variables (α β : ℝ) -- angles in radians
-- Definitions to capture the provided conditions
def dihedral_angle (α : ℝ) : Prop := true -- The angle between the planes γ₁ and γ₂
def angle_with_edge (β : ℝ) : Prop := true -- The angle between line AB and edge l

-- The angle between line AB and the plane γ₂
theorem angle_between_line_and_plane (α β : ℝ) (h1 : dihedral_angle α) (h2 : angle_with_edge β) : 
  ∃ θ : ℝ, θ = Real.arcsin (Real.sin α * Real.sin β) :=
by
  sorry

end angle_between_line_and_plane_l229_229561


namespace no_positive_integer_makes_sum_prime_l229_229659

theorem no_positive_integer_makes_sum_prime : ¬ ∃ n : ℕ, 0 < n ∧ Prime (4^n + n^4) :=
by
  sorry

end no_positive_integer_makes_sum_prime_l229_229659


namespace smallest_four_digit_number_in_pascals_triangle_l229_229435

theorem smallest_four_digit_number_in_pascals_triangle : ∃ n k, binom n k = 1000 ∧ (∀ n' k', binom n' k' > 1000 → nat.zero < n' k' ∧ n' k' < (binom n k)) := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l229_229435


namespace sufficient_but_not_necessary_l229_229928

-- Define the conditions
def abs_value_condition (x : ℝ) : Prop := |x| < 2
def quadratic_condition (x : ℝ) : Prop := x^2 - x - 6 < 0

-- Theorem statement
theorem sufficient_but_not_necessary : (∀ x : ℝ, abs_value_condition x → quadratic_condition x) ∧ ¬ (∀ x : ℝ, quadratic_condition x → abs_value_condition x) :=
by
  sorry

end sufficient_but_not_necessary_l229_229928


namespace max_value_of_sample_l229_229758

theorem max_value_of_sample 
  (x : Fin 5 → ℤ)
  (h_different : ∀ i j, i ≠ j → x i ≠ x j)
  (h_mean : (x 0 + x 1 + x 2 + x 3 + x 4) / 5 = 7)
  (h_variance : ((x 0 - 7)^2 + (x 1 - 7)^2 + (x 2 - 7)^2 + (x 3 - 7)^2 + (x 4 - 7)^2) / 5 = 4)
  : ∃ i, x i = 10 := 
sorry

end max_value_of_sample_l229_229758


namespace min_value_sin_cos_expr_l229_229851

open Real

theorem min_value_sin_cos_expr (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  ∃ min_val : ℝ, min_val = 3 * sqrt 2 ∧ ∀ β, (0 < β ∧ β < π / 2) → 
    sin β + cos β + (2 * sqrt 2) / sin (β + π / 4) ≥ min_val :=
by
  sorry

end min_value_sin_cos_expr_l229_229851


namespace evaluate_expression_l229_229977

theorem evaluate_expression : 6^4 - 4 * 6^3 + 6^2 - 2 * 6 + 1 = 457 := 
  by
    sorry

end evaluate_expression_l229_229977


namespace total_students_l229_229612

-- Condition 1: 20% of students are below 8 years of age.
-- Condition 2: The number of students of 8 years of age is 72.
-- Condition 3: The number of students above 8 years of age is 2/3 of the number of students of 8 years of age.

variable {T : ℝ} -- Total number of students

axiom cond1 : 0.20 * T = (T - (72 + (2 / 3) * 72))
axiom cond2 : 72 = 72
axiom cond3 : (T - 72 - (2 / 3) * 72) = 0

theorem total_students : T = 150 := by
  -- Proof goes here
  sorry

end total_students_l229_229612


namespace x_intercept_of_line_l229_229197

theorem x_intercept_of_line : ∃ x, 4 * x + 7 * 0 = 28 ∧ (x, 0) = (7, 0) :=
by
  existsi 7
  split
  · norm_num
  · rfl

end x_intercept_of_line_l229_229197


namespace ratio_of_width_to_length_l229_229107

variable {w: ℕ}

theorem ratio_of_width_to_length (w: ℕ) (h1: 2*w + 2*10 = 30) (h2: w = 5) :
  ∃ (x y : ℕ), x = 1 ∧ y = 2 ∧ x.gcd y = 1 ∧ w / 10 = 1 / 2 := 
by
  sorry

end ratio_of_width_to_length_l229_229107


namespace smallest_four_digit_in_pascal_l229_229417

-- define what it means for a number to be in Pascal's triangle
def in_pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- define the concept of a smallest four-digit number
def smallest_four_digit : ℕ := 1000

-- prove that the smallest four-digit number in Pascal's Triangle is 1000
theorem smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), in_pascal n k = smallest_four_digit := 
sorry

end smallest_four_digit_in_pascal_l229_229417


namespace n_prime_of_divisors_l229_229884

theorem n_prime_of_divisors (n k : ℕ) (h₁ : n > 1) 
  (h₂ : ∀ d : ℕ, d ∣ n → (d + k ∣ n) ∨ (d - k ∣ n)) : Prime n :=
  sorry

end n_prime_of_divisors_l229_229884


namespace solve_equation_l229_229890

theorem solve_equation : ∃ x : ℝ, (1 + x) / (2 - x) - 1 = 1 / (x - 2) ↔ x = 0 := 
by
  sorry

end solve_equation_l229_229890


namespace distance_between_homes_l229_229729

-- Define the conditions as Lean functions and values
def walking_speed_maxwell : ℝ := 3
def running_speed_brad : ℝ := 5
def distance_traveled_maxwell : ℝ := 15

-- State the theorem
theorem distance_between_homes : 
  ∃ D : ℝ, 
    (15 = walking_speed_maxwell * (distance_traveled_maxwell / walking_speed_maxwell)) ∧ 
    (D - 15 = running_speed_brad * (distance_traveled_maxwell / walking_speed_maxwell)) ∧ 
    D = 40 :=
by 
  sorry

end distance_between_homes_l229_229729


namespace lawrence_walking_speed_l229_229567

theorem lawrence_walking_speed :
  let distance := 4
  let time := (4 : ℝ) / 3
  let speed := distance / time
  speed = 3 := 
by
  sorry

end lawrence_walking_speed_l229_229567


namespace probability_of_rolling_integers_with_difference_3_is_1_div_6_l229_229959

noncomputable def rolls_differ_by_three_probability : ℚ :=
  let successful_outcomes := [(2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let total_outcomes := 6 * 6 in
  (successful_outcomes.length : ℚ) / total_outcomes

theorem probability_of_rolling_integers_with_difference_3_is_1_div_6 :
  rolls_differ_by_three_probability = 1 / 6 := by
  sorry

end probability_of_rolling_integers_with_difference_3_is_1_div_6_l229_229959


namespace smallest_positive_integer_ending_in_9_divisible_by_13_l229_229027

theorem smallest_positive_integer_ending_in_9_divisible_by_13 :
  ∃ n : ℕ, (n % 10 = 9) ∧ (n % 13 = 0) ∧ (∀ m : ℕ, (m % 10 = 9) ∧ (m % 13 = 0) → m ≥ n) :=
sorry

end smallest_positive_integer_ending_in_9_divisible_by_13_l229_229027


namespace lauren_tuesday_earnings_l229_229723

noncomputable def money_from_commercials (commercials_viewed : ℕ) : ℕ :=
  (1 / 2) * commercials_viewed

noncomputable def money_from_subscriptions (subscribers : ℕ) : ℕ :=
  1 * subscribers

theorem lauren_tuesday_earnings :
  let commercials_viewed := 100 in
  let subscribers := 27 in
  let total_money := money_from_commercials commercials_viewed + money_from_subscriptions subscribers in
  total_money = 77 :=
by 
  sorry

end lauren_tuesday_earnings_l229_229723


namespace triangle_area_l229_229298

theorem triangle_area (base height : ℕ) (h_base : base = 10) (h_height : height = 5) :
  (base * height) / 2 = 25 := by
  -- Proof is not required as per instructions.
  sorry

end triangle_area_l229_229298


namespace solutions_to_shifted_parabola_l229_229217

noncomputable def solution_equation := ∀ (a b : ℝ) (m : ℝ) (x : ℝ),
  (a ≠ 0) →
  ((a * (x + m) ^ 2 + b = 0) → (x = 2 ∨ x = -1)) →
  (a * (x - m + 2) ^ 2 + b = 0 → (x = -3 ∨ x = 0))

-- We'll leave the proof for this theorem as 'sorry'
theorem solutions_to_shifted_parabola (a b m : ℝ) (h : a ≠ 0)
  (h1 : ∀ (x : ℝ), a * (x + m) ^ 2 + b = 0 → (x = 2 ∨ x = -1)) 
  (x : ℝ) : 
  (a * (x - m + 2) ^ 2 + b = 0 → (x = -3 ∨ x = 0)) := sorry

end solutions_to_shifted_parabola_l229_229217


namespace expression_value_l229_229800

theorem expression_value :
  3 * ((18 + 7)^2 - (7^2 + 18^2)) = 756 := 
sorry

end expression_value_l229_229800


namespace smallest_prime_with_conditions_l229_229523

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n
def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def reverse_digits (n : ℕ) : ℕ := 
  let tens := n / 10 
  let units := n % 10 
  units * 10 + tens

theorem smallest_prime_with_conditions : 
  ∃ (p : ℕ), is_prime p ∧ 20 ≤ p ∧ p < 30 ∧ (reverse_digits p) < 100 ∧ is_composite (reverse_digits p) ∧ p = 23 :=
by
  sorry

end smallest_prime_with_conditions_l229_229523


namespace dice_diff_by_three_probability_l229_229961

theorem dice_diff_by_three_probability : 
  let outcomes := [(1, 4), (2, 5), (3, 6), (4, 1), (5, 2), (6, 3)] in
  let successful_outcomes := 6 in
  let total_outcomes := 6 * 6 in
  let probability := successful_outcomes / total_outcomes in
  probability = 1 / 6 :=
by
  sorry

end dice_diff_by_three_probability_l229_229961


namespace weight_of_gravel_l229_229155

theorem weight_of_gravel (total_weight : ℝ) (weight_sand : ℝ) (weight_water : ℝ) (weight_gravel : ℝ) 
  (h1 : total_weight = 48)
  (h2 : weight_sand = (1/3) * total_weight)
  (h3 : weight_water = (1/2) * total_weight)
  (h4 : weight_gravel = total_weight - (weight_sand + weight_water)) :
  weight_gravel = 8 :=
sorry

end weight_of_gravel_l229_229155


namespace second_athlete_triple_jump_l229_229760

theorem second_athlete_triple_jump
  (long_jump1 triple_jump1 high_jump1 : ℕ) 
  (long_jump2 high_jump2 : ℕ)
  (average_winner : ℕ) 
  (H1 : long_jump1 = 26) (H2 : triple_jump1 = 30) (H3 : high_jump1 = 7)
  (H4 : long_jump2 = 24) (H5 : high_jump2 = 8) (H6 : average_winner = 22)
  : ∃ x : ℕ, (24 + x + 8) / 3 = 22 ∧ x = 34 := 
by
  sorry

end second_athlete_triple_jump_l229_229760


namespace southton_capsule_depth_l229_229892

theorem southton_capsule_depth :
  ∃ S : ℕ, 4 * S + 12 = 48 ∧ S = 9 :=
by
  sorry

end southton_capsule_depth_l229_229892


namespace tan_alpha_plus_pi_over_4_l229_229697

noncomputable def sin_cos_identity (α : ℝ) : Prop :=
  (3 * Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 8 / 3

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : sin_cos_identity α) :
  Real.tan (α + Real.pi / 4) = -3 :=
  by
  sorry

end tan_alpha_plus_pi_over_4_l229_229697


namespace y_coordinate_of_third_vertex_eq_l229_229645

theorem y_coordinate_of_third_vertex_eq (x1 x2 y1 y2 : ℝ)
    (h1 : x1 = 0) 
    (h2 : y1 = 3) 
    (h3 : x2 = 10) 
    (h4 : y2 = 3) 
    (h5 : x1 ≠ x2) 
    (h6 : y1 = y2) 
    : ∃ y3 : ℝ, y3 = 3 + 5 * Real.sqrt 3 := 
by
  sorry

end y_coordinate_of_third_vertex_eq_l229_229645


namespace exists_composite_arith_sequence_pairwise_coprime_l229_229043

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem exists_composite_arith_sequence_pairwise_coprime (n : ℕ) : 
  ∃ seq : Fin n → ℕ, (∀ i, ∃ k, seq i = factorial n + k) ∧ 
  (∀ i j, i ≠ j → gcd (seq i) (seq j) = 1) :=
by
  sorry

end exists_composite_arith_sequence_pairwise_coprime_l229_229043


namespace y_work_time_l229_229927

theorem y_work_time (x_days : ℕ) (x_work_time : ℕ) (y_work_time : ℕ) :
  x_days = 40 ∧ x_work_time = 8 ∧ y_work_time = 20 →
  let x_rate := 1 / 40
  let work_done_by_x := 8 * x_rate
  let remaining_work := 1 - work_done_by_x
  let y_rate := remaining_work / 20
  y_rate * 25 = 1 :=
by {
  sorry
}

end y_work_time_l229_229927


namespace rectangle_shorter_side_length_l229_229063

theorem rectangle_shorter_side_length (rope_length : ℕ) (long_side : ℕ) : 
  rope_length = 100 → long_side = 28 → 
  ∃ short_side : ℕ, (2 * long_side + 2 * short_side = rope_length) ∧ short_side = 22 :=
by
  sorry

end rectangle_shorter_side_length_l229_229063


namespace find_m_l229_229682

noncomputable def is_power_function (y : ℝ → ℝ) := 
  ∃ (c : ℝ), ∃ (n : ℝ), ∀ x : ℝ, y x = c * x ^ n

theorem find_m (m : ℝ) :
  (∀ x : ℝ, (∃ c : ℝ, (m^2 - 2 * m + 1) * x^(m - 1) = c * x^n) ∧ (∀ x : ℝ, true)) → m = 2 :=
sorry

end find_m_l229_229682


namespace smallest_four_digit_in_pascal_l229_229463

theorem smallest_four_digit_in_pascal :
  ∃ n k : ℕ, binomial n k = 1000 ∧ 
             (∀ a b : ℕ, binomial a b ∈ set.Icc 1000 9999 → a ≥ n) :=
sorry

end smallest_four_digit_in_pascal_l229_229463


namespace school_days_per_week_l229_229184

-- Definitions based on the conditions given
def paper_per_class_per_day : ℕ := 200
def total_paper_per_week : ℕ := 9000
def number_of_classes : ℕ := 9

-- The theorem stating the main claim to prove
theorem school_days_per_week :
  total_paper_per_week / (paper_per_class_per_day * number_of_classes) = 5 :=
  by
  sorry

end school_days_per_week_l229_229184


namespace mean_equality_l229_229402

theorem mean_equality (x : ℝ) 
  (h : (7 + 9 + 23) / 3 = (16 + x) / 2) : 
  x = 10 := 
sorry

end mean_equality_l229_229402


namespace paperclips_volume_75_l229_229769

noncomputable def paperclips (v : ℝ) : ℝ := 60 / Real.sqrt 27 * Real.sqrt v

theorem paperclips_volume_75 :
  paperclips 75 = 100 :=
by
  sorry

end paperclips_volume_75_l229_229769


namespace min_value_of_quadratic_l229_229658

theorem min_value_of_quadratic (x : ℝ) : ∃ z : ℝ, z = 2 * x^2 + 16 * x + 40 ∧ z = 8 :=
by {
  sorry
}

end min_value_of_quadratic_l229_229658


namespace intersecting_lines_l229_229286

theorem intersecting_lines (c d : ℝ)
  (h1 : 16 = 2 * 4 + c)
  (h2 : 16 = 5 * 4 + d) :
  c + d = 4 :=
sorry

end intersecting_lines_l229_229286


namespace geom_seq_product_arith_seq_l229_229283

theorem geom_seq_product_arith_seq (a b c r : ℝ) (h1 : c = b * r)
  (h2 : b = a * r)
  (h3 : a * b * c = 512)
  (h4 : b = 8)
  (h5 : 2 * b = (a - 2) + (c - 2)) :
  (a = 4 ∧ b = 8 ∧ c = 16) ∨ (a = 16 ∧ b = 8 ∧ c = 4) :=
by
  sorry

end geom_seq_product_arith_seq_l229_229283


namespace maximum_value_expression_l229_229018

noncomputable def expression (s t : ℝ) := -2 * s^2 + 24 * s + 3 * t - 38

theorem maximum_value_expression : ∀ (s : ℝ), expression s 4 ≤ 46 :=
by sorry

end maximum_value_expression_l229_229018


namespace det_A_eq_neg15_l229_229724

open Matrix

variable {x y : ℝ}
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![x, 3], ![-4, y]]
def B := 3 • A⁻¹
def I : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem det_A_eq_neg15 (h : A + B = I) : det A = -15 :=
by
  sorry

end det_A_eq_neg15_l229_229724


namespace find_k_l229_229323

def distances (S x y k : ℝ) := (S - x * 0.75) * x / (x + y) + 0.75 * x = S * x / (x + y) - 18 ∧
                              S * x / (x + y) - (S - y / 3) * x / (x + y) = k

theorem find_k (S x y k : ℝ) (h₁ : x * y / (x + y) = 24) (h₂ : k = 24 / 3)
  : k = 8 :=
by 
  -- We need to fill in the proof steps here
  sorry

end find_k_l229_229323


namespace incenter_bisects_BB1_l229_229905

noncomputable theory
open_locale euclidean_geometry

variables {a b c d : ℝ} (A B C O B1 : Point)

-- Condition 1: The sides of the triangle form an arithmetic progression
variables (h_arith_prog : a + d = b ∧ b + d = c)

-- Condition 2: The bisector of angle B intersects the circumcircle at point B1
variables (h_bisector : angle_bisector B A C B1)
variables [incircle_configuration A B C O] (h3 : O = incircle_center A B C) (h4 : concyclic_points [A, B, C, B1])

-- Conclusion: O bisects BB1.
theorem incenter_bisects_BB1 (h_arith_prog : a + d = b ∧ b + d = c)
(h_bisector : angle_bisector B A C B1)
(h3 : O = incircle_center A B C)
(h4 : concyclic_points [A, B, C, B1]):
line_segment O B = line_segment O B1 :=
begin
  sorry
end

end incenter_bisects_BB1_l229_229905


namespace smallest_base_b_l229_229150

theorem smallest_base_b (b : ℕ) : (b ≥ 1) → (b^2 ≤ 82) → (82 < b^3) → b = 5 := by
  sorry

end smallest_base_b_l229_229150


namespace cat_mouse_position_after_300_moves_l229_229104

def move_pattern_cat_mouse :=
  let cat_cycle_length := 4
  let mouse_cycle_length := 8
  let cat_moves := 300
  let mouse_moves := (3 / 2) * cat_moves
  let cat_position := (cat_moves % cat_cycle_length)
  let mouse_position := (mouse_moves % mouse_cycle_length)
  (cat_position, mouse_position)

theorem cat_mouse_position_after_300_moves :
  move_pattern_cat_mouse = (0, 2) :=
by
  sorry

end cat_mouse_position_after_300_moves_l229_229104


namespace lauren_mail_total_l229_229719

theorem lauren_mail_total : 
  let monday := 65
  let tuesday := monday + 10
  let wednesday := tuesday - 5
  let thursday := wednesday + 15
  monday + tuesday + wednesday + thursday = 295 :=
by
  have monday := 65
  have tuesday := monday + 10
  have wednesday := tuesday - 5
  have thursday := wednesday + 15
  calc
    monday + tuesday + wednesday + thursday 
    = 65 + (65 + 10) + (65 + 10 - 5) + (65 + 10 - 5 + 15) : by rfl
    ... = 65 + 75 + 70 + 85 : by rfl
    ... = 295 : by rfl

end lauren_mail_total_l229_229719


namespace bike_sharing_problem_l229_229967

def combinations (n k : ℕ) : ℕ := (Nat.choose n k)

theorem bike_sharing_problem:
  let total_bikes := 10
  let blue_bikes := 4
  let yellow_bikes := 6
  let inspected_bikes := 4
  let way_two_blue := combinations blue_bikes 2 * combinations yellow_bikes 2
  let way_three_blue := combinations blue_bikes 3 * combinations yellow_bikes 1
  let way_four_blue := combinations blue_bikes 4
  way_two_blue + way_three_blue + way_four_blue = 115 :=
by
  sorry

end bike_sharing_problem_l229_229967


namespace arithmetic_sequence_sum_l229_229080

noncomputable def S (n : ℕ) (a1 d : ℝ) : ℝ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_sequence_sum (a15 a6 : ℝ) (h : a15 + a6 = 1) : 
  S 20 ((a15 * 2) / 2) (((a6 * 2) / 2) - ((a15 * 2) / 2) / 19) = 10 :=
by 
  sorry

end arithmetic_sequence_sum_l229_229080


namespace first_three_workers_dig_time_l229_229819

variables 
  (a b c d : ℝ) -- work rates of the four workers
  (hours : ℝ) -- time to dig the trench

def work_together (a b c d hours : ℝ) := (a + b + c + d) * hours = 1

def scenario1 (a b c d : ℝ) := (2 * a + (1/2) * b + c + d) * 6 = 1

def scenario2 (a b c d : ℝ) := (a/2 + 2 * b + c + d) * 4 = 1

theorem first_three_workers_dig_time
  (h1 : work_together a b c d 6)
  (h2 : scenario1 a b c d)
  (h3 : scenario2 a b c d) :
  hours = 6 := 
sorry

end first_three_workers_dig_time_l229_229819


namespace minimum_words_to_learn_l229_229224

-- Definition of the problem
def total_words : ℕ := 600
def required_percentage : ℕ := 90

-- Lean statement of the problem
theorem minimum_words_to_learn : ∃ x : ℕ, (x / total_words : ℚ) = required_percentage / 100 ∧ x = 540 :=
sorry

end minimum_words_to_learn_l229_229224


namespace juniors_score_l229_229371

theorem juniors_score (juniors seniors total_students avg_score avg_seniors_score total_score : ℝ)
  (hj: juniors = 0.2 * total_students)
  (hs: seniors = 0.8 * total_students)
  (ht: total_students = 20)
  (ha: avg_score = 78)
  (hp: (seniors * avg_seniors_score + juniors * c) / total_students = avg_score)
  (havg_seniors: avg_seniors_score = 76)
  (hts: total_score = total_students * avg_score)
  (total_seniors_score : ℝ)
  (hts_seniors: total_seniors_score = seniors * avg_seniors_score)
  (total_juniors_score : ℝ)
  (hts_juniors: total_juniors_score = total_score - total_seniors_score)
  (hjs: c = total_juniors_score / juniors) :
  c = 86 :=
sorry

end juniors_score_l229_229371


namespace probability_of_diff_3_is_1_over_9_l229_229948

theorem probability_of_diff_3_is_1_over_9 :
  let outcomes := [(a, b) | a in [1, 2, 3, 4, 5, 6], b in [1, 2, 3, 4, 5, 6]],
      valid_pairs := [(2, 5), (3, 6), (4, 1), (5, 2)],
      total_outcomes := 36,
      successful_outcomes := 4
  in
  successful_outcomes.to_rat / total_outcomes.to_rat = 1 / 9 := 
  sorry

end probability_of_diff_3_is_1_over_9_l229_229948


namespace decreasing_interval_l229_229403

noncomputable def f (x : ℝ) := x^2 * Real.exp x

theorem decreasing_interval : ∀ x : ℝ, x > -2 ∧ x < 0 → deriv f x < 0 := 
by
  intro x h
  sorry

end decreasing_interval_l229_229403


namespace polynomial_expansion_l229_229511

theorem polynomial_expansion (x : ℝ) : 
  (1 - x^3) * (1 + x^4 - x^5) = 1 - x^3 + x^4 - x^5 - x^7 + x^8 :=
by
  sorry

end polynomial_expansion_l229_229511


namespace larger_integer_is_7sqrt14_l229_229275

theorem larger_integer_is_7sqrt14 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a / b = 7 / 3) (h2 : a * b = 294) : max a b = 7 * Real.sqrt 14 :=
by 
  sorry

end larger_integer_is_7sqrt14_l229_229275


namespace n_power_of_3_l229_229380

theorem n_power_of_3 (n : ℕ) (h_prime : Prime (4^n + 2^n + 1)) : ∃ k : ℕ, n = 3^k := 
sorry

end n_power_of_3_l229_229380


namespace total_dogs_l229_229409

-- Definitions of conditions
def brown_dogs : Nat := 20
def white_dogs : Nat := 10
def black_dogs : Nat := 15

-- Theorem to prove the total number of dogs
theorem total_dogs : brown_dogs + white_dogs + black_dogs = 45 := by
  -- Placeholder for proof
  sorry

end total_dogs_l229_229409


namespace average_visitors_per_day_l229_229924

/-- The average number of visitors per day in a month of 30 days that begins with a Sunday is 188, 
given that the library has 500 visitors on Sundays and 140 visitors on other days. -/
theorem average_visitors_per_day (visitors_sunday visitors_other : ℕ) (days_in_month : ℕ) 
   (starts_on_sunday : Bool) (sundays : ℕ) 
   (visitors_sunday_eq_500 : visitors_sunday = 500)
   (visitors_other_eq_140 : visitors_other = 140)
   (days_in_month_eq_30 : days_in_month = 30)
   (starts_on_sunday_eq_true : starts_on_sunday = true)
   (sundays_eq_4 : sundays = 4) :
   (visitors_sunday * sundays + visitors_other * (days_in_month - sundays)) / days_in_month = 188 := 
by {
  sorry
}

end average_visitors_per_day_l229_229924


namespace sqrt_product_is_four_l229_229331

theorem sqrt_product_is_four : (Real.sqrt 2 * Real.sqrt 8) = 4 := 
by
  sorry

end sqrt_product_is_four_l229_229331


namespace x_intercept_is_7_0_l229_229196

-- Define the given line equation
def line_eq (x y : ℚ) : Prop := 4 * x + 7 * y = 28

-- State the theorem we want to prove
theorem x_intercept_is_7_0 :
  ∃ x : ℚ, ∃ y : ℚ, line_eq x y ∧ y = 0 ∧ x = 7 :=
by
  sorry

end x_intercept_is_7_0_l229_229196


namespace mr_johnson_fencing_l229_229883

variable (Length Width : ℕ)

def perimeter_of_rectangle (Length Width : ℕ) : ℕ :=
  2 * (Length + Width)

theorem mr_johnson_fencing
  (hLength : Length = 25)
  (hWidth : Width = 15) :
  perimeter_of_rectangle Length Width = 80 := by
  sorry

end mr_johnson_fencing_l229_229883


namespace sufficient_but_not_necessary_l229_229268

theorem sufficient_but_not_necessary (a : ℝ) :
  ((a + 2) * (3 * a - 4) - (a - 2) ^ 2 = 0 → a = 2 ∨ a = 1 / 2) →
  (a = 1 / 2 → ∃ x y : ℝ, (a+2) * x + (a-2) * y = 1 ∧ (a-2) * x + (3*a-4) * y = 2) →
  ( (∀ x y : ℝ, (a+2) * x + (a-2) * y = 1 ∧ (a-2) * x + (3*a-4) * y = 2 → a = 1/2) ∧ 
  (∃ x y : ℝ, (a+2) * x + (a-2) * y = 1 ∧ (a-2) * x + (3*a-4) * y = 2) → a ≠ 1/2) :=
by
  sorry

end sufficient_but_not_necessary_l229_229268


namespace least_times_to_eat_l229_229611

theorem least_times_to_eat (A B C : ℕ) (h1 : A = (9 * B) / 5) (h2 : B = C / 8) : 
  A = 2 ∧ B = 1 ∧ C = 8 :=
sorry

end least_times_to_eat_l229_229611


namespace percentage_increase_l229_229704

variable (x r : ℝ)

theorem percentage_increase (h_x : x = 78.4) (h_r : x = 70 * (1 + r)) : r = 0.12 :=
by
  -- Proof goes here
  sorry

end percentage_increase_l229_229704


namespace gcd_40_120_45_l229_229412

theorem gcd_40_120_45 : Nat.gcd (Nat.gcd 40 120) 45 = 5 :=
by
  sorry

end gcd_40_120_45_l229_229412


namespace vincent_books_cost_l229_229600

theorem vincent_books_cost :
  let num_animals := 10
  let num_outer_space := 1
  let num_trains := 3
  let total_books := num_animals + num_outer_space + num_trains
  let total_spent := 224
  let cost_per_book := total_spent / total_books
  cost_per_book = 16 :=
by
  let num_animals := 10
  let num_outer_space := 1
  let num_trains := 3
  let total_books := num_animals + num_outer_space + num_trains
  let total_spent := 224
  let cost_per_book := total_spent / total_books
  show cost_per_book = 16
  sorry

end vincent_books_cost_l229_229600


namespace smallest_positive_integer_ends_in_9_and_divisible_by_13_l229_229024

theorem smallest_positive_integer_ends_in_9_and_divisible_by_13 :
  ∃ n : ℕ, n % 10 = 9 ∧ 13 ∣ n ∧ n > 0 ∧ ∀ m, m % 10 = 9 → 13 ∣ m ∧ m > 0 → m ≥ n := 
begin
  use 99,
  split,
  { exact mod_eq_of_lt (10*k + 9) 10 99 9 (by norm_num), },
  split,
  { exact dvd_refl 99, },
  split,
  { exact zero_lt_99, },
  intros m hm1 hm2 hpos,
  by_contradiction hmn,
  sorry
end

end smallest_positive_integer_ends_in_9_and_divisible_by_13_l229_229024


namespace calvin_final_weight_l229_229333

def initial_weight : ℕ := 250
def weight_loss_per_month : ℕ := 8
def duration_months : ℕ := 12
def total_weight_loss : ℕ := weight_loss_per_month * duration_months
def final_weight : ℕ := initial_weight - total_weight_loss

theorem calvin_final_weight : final_weight = 154 :=
by {
  have h1 : total_weight_loss = 96 := by norm_num,
  rw [h1],
  norm_num,
  sorry
}

-- We have used 'sorry' to mark the place where the proof would be completed.

end calvin_final_weight_l229_229333


namespace min_sum_of_arithmetic_seq_l229_229529

variables {a_n : ℕ → ℝ} (S_n : ℕ → ℝ) (d : ℝ)

-- Given Conditions
def arithmetic_sequence (a_n : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a_n (n+1) = a_n n + d

def sum_of_first_n_terms (a_n : ℕ → ℝ) (n : ℕ) :=
  (n : ℝ) * (a_n 1) + (n * (n - 1) * d / 2)

-- The specific conditions from the problem
def conditions :=
  a_n 1 = -9 ∧ (a_n 1 + 7 * d) + (a_n 1 + d) = -2

-- The goal statement which needs to be proven
theorem min_sum_of_arithmetic_seq : 
  conditions a_n d → 
  (∃ n : ℕ, S_n n = (n:ℝ) * a_n 1 + (n * (n-1) * d / 2) ∧ (S_n n) = n^2 - 10 * n ∧ n = 5) :=
begin
  sorry
end

end min_sum_of_arithmetic_seq_l229_229529


namespace probability_green_ball_l229_229787

theorem probability_green_ball 
  (total_balls : ℕ) 
  (green_balls : ℕ) 
  (white_balls : ℕ) 
  (h_total : total_balls = 9) 
  (h_green : green_balls = 7)
  (h_white : white_balls = 2)
  (h_total_eq : total_balls = green_balls + white_balls) : 
  (green_balls / total_balls : ℚ) = 7 / 9 := 
by
  sorry

end probability_green_ball_l229_229787


namespace subtract_two_decimals_l229_229649

theorem subtract_two_decimals : 3.75 - 1.46 = 2.29 := by
  sorry

end subtract_two_decimals_l229_229649


namespace total_seashells_l229_229730

-- Define the conditions from part a)
def unbroken_seashells : ℕ := 2
def broken_seashells : ℕ := 4

-- Define the proof problem
theorem total_seashells :
  unbroken_seashells + broken_seashells = 6 :=
by
  sorry

end total_seashells_l229_229730


namespace positions_after_317_moves_l229_229232

-- Define positions for the cat and dog
inductive ArchPosition
| North | East | South | West
deriving DecidableEq

inductive PathPosition
| North | Northeast | East | Southeast | South | Southwest
deriving DecidableEq

-- Define the movement function for cat and dog
def cat_position (n : Nat) : ArchPosition :=
  match n % 4 with
  | 0 => ArchPosition.North
  | 1 => ArchPosition.East
  | 2 => ArchPosition.South
  | _ => ArchPosition.West

def dog_position (n : Nat) : PathPosition :=
  match n % 6 with
  | 0 => PathPosition.North
  | 1 => PathPosition.Northeast
  | 2 => PathPosition.East
  | 3 => PathPosition.Southeast
  | 4 => PathPosition.South
  | _ => PathPosition.Southwest

-- Theorem statement to prove the positions after 317 moves
theorem positions_after_317_moves :
  cat_position 317 = ArchPosition.North ∧
  dog_position 317 = PathPosition.South :=
by
  sorry

end positions_after_317_moves_l229_229232


namespace right_triangle_side_length_l229_229181

theorem right_triangle_side_length (r f : ℝ) (h : f < 2 * r) :
  let c := 2 * r
  let a := (f / (4 * c)) * (f + Real.sqrt (f^2 + 8 * c^2))
  a = (f / (4 * (2 * r))) * (f + Real.sqrt (f^2 + 8 * (2 * r)^2)) :=
by
  let c := 2 * r
  let a := (f / (4 * c)) * (f + Real.sqrt (f^2 + 8 * c^2))
  have acalc : a = (f / (4 * (2 * r))) * (f + Real.sqrt (f^2 + 8 * (2 * r)^2)) := by sorry
  exact acalc

end right_triangle_side_length_l229_229181


namespace probability_diff_by_three_l229_229956

theorem probability_diff_by_three : 
  let outcomes := (Finset.product (Finset.range 1 7) (Finset.range 1 7)) in
  let successful_outcomes := Finset.filter (λ (x : ℕ × ℕ), abs (x.1 - x.2) = 3) outcomes in
  (successful_outcomes.card : ℚ) / outcomes.card = 1 / 6 :=
by
  sorry

end probability_diff_by_three_l229_229956


namespace third_number_sixth_row_l229_229644

/-- Define the arithmetic sequence and related properties. -/
def sequence (n : ℕ) : ℕ := 2 * n - 1

/-- Define sum of first k terms in a series where each row length doubles the previous row length. -/
def sum_of_rows (k : ℕ) : ℕ :=
  2^k - 1

/-- Statement of the problem: Prove that the third number in the sixth row is 67. -/
theorem third_number_sixth_row : sequence (sum_of_rows 5 + 3) = 67 := by
  sorry

end third_number_sixth_row_l229_229644


namespace greatest_4_digit_base7_divisible_by_7_l229_229017

-- Definitions and conditions
def is_base7_number (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 7, d < 7

def is_4_digit_base7 (n : ℕ) : Prop :=
  is_base7_number n ∧ 343 ≤ n ∧ n < 2401 -- 343 = 7^3 (smallest 4-digit base 7) and 2401 = 7^4

def is_divisible_by_7 (n : ℕ) : Prop :=
  n % 7 = 0

-- Proof problem statement
theorem greatest_4_digit_base7_divisible_by_7 :
  ∃ (n : ℕ), is_4_digit_base7 n ∧ is_divisible_by_7 n ∧ n = 2346 :=
sorry

end greatest_4_digit_base7_divisible_by_7_l229_229017


namespace price_change_on_eggs_and_apples_l229_229374

theorem price_change_on_eggs_and_apples :
  let initial_egg_price := 1.00
  let initial_apple_price := 1.00
  let egg_drop_percent := 0.10
  let apple_increase_percent := 0.02
  let new_egg_price := initial_egg_price * (1 - egg_drop_percent)
  let new_apple_price := initial_apple_price * (1 + apple_increase_percent)
  let initial_total := initial_egg_price + initial_apple_price
  let new_total := new_egg_price + new_apple_price
  let percent_change := ((new_total - initial_total) / initial_total) * 100
  percent_change = -4 :=
by
  sorry

end price_change_on_eggs_and_apples_l229_229374


namespace smallest_pos_int_ending_in_9_divisible_by_13_l229_229019

theorem smallest_pos_int_ending_in_9_divisible_by_13 :
  ∃ (n : ℕ), n % 10 = 9 ∧ n % 13 = 0 ∧ ∀ m, m % 10 = 9 ∧ m % 13 = 0 → n ≤ m :=
by
  sorry

end smallest_pos_int_ending_in_9_divisible_by_13_l229_229019


namespace larger_integer_value_l229_229276

theorem larger_integer_value
  (a b : ℕ)
  (h1 : a ≥ b)
  (h2 : ↑a / ↑b = 7 / 3)
  (h3 : a * b = 294) :
  a = 7 * Int.sqrt 14 := 
sorry

end larger_integer_value_l229_229276


namespace apples_mass_left_l229_229783

theorem apples_mass_left (initial_kidney golden canada fuji granny : ℕ)
                         (sold_kidney golden canada fuji granny : ℕ)
                         (left_kidney golden canada fuji granny : ℕ) :
  initial_kidney = 26 → sold_kidney = 15 → left_kidney = 11 →
  initial_golden = 42 → sold_golden = 28 → left_golden = 14 →
  initial_canada = 19 → sold_canada = 12 → left_canada = 7 →
  initial_fuji = 35 → sold_fuji = 20 → left_fuji = 15 →
  initial_granny = 22 → sold_granny = 18 → left_granny = 4 →
  left_kidney = initial_kidney - sold_kidney ∧
  left_golden = initial_golden - sold_golden ∧
  left_canada = initial_canada - sold_canada ∧
  left_fuji = initial_fuji - sold_fuji ∧
  left_granny = initial_granny - sold_granny := by sorry

end apples_mass_left_l229_229783


namespace find_abc_l229_229270

theorem find_abc (a b c : ℝ) (x y : ℝ) :
  (x^2 + y^2 + 2*a*x - b*y + c = 0) ∧
  ((-a, b / 2) = (2, 2)) ∧
  (4 = b^2 / 4 + a^2 - c) →
  a = -2 ∧ b = 4 ∧ c = 4 := by
  sorry

end find_abc_l229_229270


namespace sin_cos_value_l229_229526

noncomputable def tan_plus_pi_div_two_eq_two (θ : ℝ) : Prop :=
  Real.tan (θ + Real.pi / 2) = 2

theorem sin_cos_value (θ : ℝ) (h : tan_plus_pi_div_two_eq_two θ) :
  Real.sin θ * Real.cos θ = -2 / 5 :=
sorry

end sin_cos_value_l229_229526


namespace hannah_books_per_stocking_l229_229358

theorem hannah_books_per_stocking
  (candy_canes_per_stocking : ℕ)
  (beanie_babies_per_stocking : ℕ)
  (num_kids : ℕ)
  (total_stuffers : ℕ)
  (books_per_stocking : ℕ) :
  candy_canes_per_stocking = 4 →
  beanie_babies_per_stocking = 2 →
  num_kids = 3 →
  total_stuffers = 21 →
  books_per_stocking = (total_stuffers - (candy_canes_per_stocking + beanie_babies_per_stocking) * num_kids) / num_kids →
  books_per_stocking = 1 := 
by 
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h5
  simp at h5
  sorry

end hannah_books_per_stocking_l229_229358


namespace find_phi_l229_229218

open Real

theorem find_phi (φ : ℝ) : (∃ k : ℤ, φ = k * π - π / 3) →
  tan (π / 3 + φ) = 0 →
  φ = -π / 3 :=
by
  intros hφ htan
  sorry

end find_phi_l229_229218


namespace time_to_wash_car_l229_229712

theorem time_to_wash_car (W : ℕ) 
    (t_oil : ℕ := 15) 
    (t_tires : ℕ := 30) 
    (n_wash : ℕ := 9) 
    (n_oil : ℕ := 6) 
    (n_tires : ℕ := 2) 
    (total_time : ℕ := 240) 
    (h : n_wash * W + n_oil * t_oil + n_tires * t_tires = total_time) 
    : W = 10 := by
  sorry

end time_to_wash_car_l229_229712


namespace problem_proof_l229_229998

noncomputable def f : ℝ → ℝ := sorry

theorem problem_proof (h1 : ∀ x : ℝ, f (-x) = f x)
    (h2 : ∀ x y : ℝ, x < y ∧ y ≤ -1 → f x < f y) : 
    f 2 < f (-3 / 2) ∧ f (-3 / 2) < f (-1) :=
by
  sorry

end problem_proof_l229_229998


namespace ganesh_ram_together_l229_229525

theorem ganesh_ram_together (G R S : ℝ) (h1 : G + R + S = 1 / 16) (h2 : S = 1 / 48) : (G + R) = 1 / 24 :=
by
  sorry

end ganesh_ram_together_l229_229525


namespace total_time_is_12_years_l229_229311

noncomputable def total_time_spent (shape_years climb_years_per_summit dive_months cave_years : ℕ) : ℕ :=
  shape_years + (2 * shape_years) + (7 * climb_years_per_summit) / 12 + ((7 * climb_years_per_summit) % 12) / 12 + (dive_months + 12) / 12 + cave_years

theorem total_time_is_12_years :
  total_time_spent 2 5 13 2 = 12 :=
by
  sorry

end total_time_is_12_years_l229_229311


namespace sum_of_coefficients_factors_l229_229902

theorem sum_of_coefficients_factors :
  ∃ (a b c d e : ℤ), 
    (343 * (x : ℤ)^3 + 125 = (a * x + b) * (c * x^2 + d * x + e)) ∧ 
    (a + b + c + d + e = 51) :=
sorry

end sum_of_coefficients_factors_l229_229902


namespace salary_increase_l229_229407

theorem salary_increase (original_salary reduced_salary : ℝ) (hx : reduced_salary = original_salary * 0.5) : 
  (reduced_salary + reduced_salary * 1) = original_salary :=
by
  -- Prove the required increase percent to return to original salary
  sorry

end salary_increase_l229_229407


namespace john_increased_bench_press_factor_l229_229113

theorem john_increased_bench_press_factor (initial current : ℝ) (decrease_percent : ℝ) 
  (h_initial : initial = 500) 
  (h_current : current = 300) 
  (h_decrease : decrease_percent = 0.80) : 
  current / (initial * (1 - decrease_percent)) = 3 := 
by
  -- We'll provide the proof here later
  sorry

end john_increased_bench_press_factor_l229_229113


namespace lauren_mail_total_l229_229718

theorem lauren_mail_total : 
  let monday := 65
  let tuesday := monday + 10
  let wednesday := tuesday - 5
  let thursday := wednesday + 15
  monday + tuesday + wednesday + thursday = 295 :=
by
  have monday := 65
  have tuesday := monday + 10
  have wednesday := tuesday - 5
  have thursday := wednesday + 15
  calc
    monday + tuesday + wednesday + thursday 
    = 65 + (65 + 10) + (65 + 10 - 5) + (65 + 10 - 5 + 15) : by rfl
    ... = 65 + 75 + 70 + 85 : by rfl
    ... = 295 : by rfl

end lauren_mail_total_l229_229718


namespace arcade_ticket_problem_l229_229965

-- Define all the conditions given in the problem
def initial_tickets : Nat := 13
def used_tickets : Nat := 8
def more_tickets_for_clothes : Nat := 10
def tickets_for_toys : Nat := 8
def tickets_for_clothes := tickets_for_toys + more_tickets_for_clothes

-- The proof statement (goal)
theorem arcade_ticket_problem : tickets_for_clothes = 18 := by
  -- This is where the proof would go
  sorry

end arcade_ticket_problem_l229_229965


namespace problem1_problem2_l229_229302

-- Problem 1 equivalent proof problem
theorem problem1 : 
  (Real.sqrt 3 * Real.sqrt 6 - (Real.sqrt (1 / 2) - Real.sqrt 8)) = (9 * Real.sqrt 2 / 2) :=
by
  sorry

-- Problem 2 equivalent proof problem
theorem problem2 (x : Real) (hx : x = Real.sqrt 5) : 
  ((1 + 1 / x) / ((x^2 + x) / x)) = (Real.sqrt 5 / 5) :=
by
  sorry

end problem1_problem2_l229_229302


namespace sufficient_not_necessary_condition_l229_229301

theorem sufficient_not_necessary_condition {x : ℝ} (h : 1 < x ∧ x < 2) : x < 2 ∧ ¬(∀ x, x < 2 → (1 < x ∧ x < 2)) :=
by
  sorry

end sufficient_not_necessary_condition_l229_229301


namespace ratio_AB_CD_lengths_AB_CD_l229_229766

theorem ratio_AB_CD 
  (AM MD BN NC : ℝ)
  (h_AM : AM = 25 / 7)
  (h_MD : MD = 10 / 7)
  (h_BN : BN = 3 / 2)
  (h_NC : NC = 9 / 2)
  : (AM / MD) / (BN / NC) = 5 / 6 :=
by
  sorry

theorem lengths_AB_CD
  (AM MD BN NC AB CD : ℝ)
  (h_AM : AM = 25 / 7)
  (h_MD : MD = 10 / 7)
  (h_BN : BN = 3 / 2)
  (h_NC : NC = 9 / 2)
  (AB_div_CD : (AM / MD) / (BN / NC) = 5 / 6)
  (h_touch : true)  -- A placeholder condition indicating circles touch each other
  : AB = 5 ∧ CD = 6 :=
by
  sorry

end ratio_AB_CD_lengths_AB_CD_l229_229766


namespace smallest_four_digit_in_pascals_triangle_l229_229455

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, 1000 ≤ nat.choose n k ∧ nat.choose n k < 10000 ∧ 
  (∀ m l : ℕ, (nat.choose m l ≥ 1000 ∧ nat.choose m l < 10000) → nat.choose n k ≤ nat.choose m l) :=
begin
  -- This is a statement only, no proof is provided
  sorry
end

end smallest_four_digit_in_pascals_triangle_l229_229455


namespace stingrays_count_l229_229176

theorem stingrays_count (Sh S : ℕ) (h1 : Sh = 2 * S) (h2 : S + Sh = 84) : S = 28 :=
by
  -- Proof will be filled here
  sorry

end stingrays_count_l229_229176


namespace at_least_two_pairs_in_one_drawer_l229_229207

theorem at_least_two_pairs_in_one_drawer (n : ℕ) (hn : n > 0) : 
  ∃ i j : ℕ, i ≠ j ∧ i < n ∧ j < n :=
by {
  sorry
}

end at_least_two_pairs_in_one_drawer_l229_229207


namespace additional_time_to_empty_tank_l229_229174

-- Definitions based on conditions
def tankCapacity : ℕ := 3200  -- litres
def outletTimeAlone : ℕ := 5  -- hours
def inletRate : ℕ := 4  -- litres/min

-- Calculate rates
def outletRate : ℕ := tankCapacity / outletTimeAlone  -- litres/hour
def inletRatePerHour : ℕ := inletRate * 60  -- Convert litres/min to litres/hour

-- Calculate effective_rate when both pipes open
def effectiveRate : ℕ := outletRate - inletRatePerHour  -- litres/hour

-- Calculate times
def timeWithInletOpen : ℕ := tankCapacity / effectiveRate  -- hours
def additionalTime : ℕ := timeWithInletOpen - outletTimeAlone  -- hours

-- Proof statement
theorem additional_time_to_empty_tank : additionalTime = 3 := by
  -- It's clear from calculation above, we just add sorry for now to skip the proof
  sorry

end additional_time_to_empty_tank_l229_229174


namespace find_x_intercept_l229_229195

theorem find_x_intercept : ∃ x y : ℚ, (4 * x + 7 * y = 28) ∧ (y = 0) ∧ (x = 7) ∧ (y = 0) :=
by
  use 7, 0
  split
  · simp
  · exact rfl
  · exact rfl
  · exact rfl

end find_x_intercept_l229_229195


namespace max_vertex_sum_l229_229987

theorem max_vertex_sum
  (a U : ℤ)
  (hU : U ≠ 0)
  (hA : 0 = a * 0 * (0 - 3 * U))
  (hB : 0 = a * (3 * U) * ((3 * U) - 3 * U))
  (hC : 12 = a * (3 * U - 1) * ((3 * U - 1) - 3 * U))
  : ∃ N : ℝ, N = (3 * U) / 2 - (9 * a * U^2) / 4 ∧ N ≤ 17.75 :=
by sorry

end max_vertex_sum_l229_229987


namespace number_of_b_objects_l229_229003

theorem number_of_b_objects
  (total_objects : ℕ) 
  (a_objects : ℕ) 
  (b_objects : ℕ) 
  (h1 : total_objects = 35) 
  (h2 : a_objects = 17) 
  (h3 : total_objects = a_objects + b_objects) :
  b_objects = 18 :=
by
  sorry

end number_of_b_objects_l229_229003


namespace find_k_l229_229840

theorem find_k (x k : ℝ) (h : ((x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) ∧ k ≠ 0) : k = 6 :=
by
  sorry

end find_k_l229_229840


namespace coin_toss_probability_l229_229373

-- Define the sample space of the coin toss
inductive Coin
| heads : Coin
| tails : Coin

-- Define the probability function
def probability (outcome : Coin) : ℝ :=
  match outcome with
  | Coin.heads => 0.5
  | Coin.tails => 0.5

-- The theorem to be proved: In a fair coin toss, the probability of getting "heads" or "tails" is 0.5
theorem coin_toss_probability (outcome : Coin) : probability outcome = 0.5 :=
sorry

end coin_toss_probability_l229_229373


namespace total_distance_biked_two_days_l229_229170

def distance_yesterday : ℕ := 12
def distance_today : ℕ := (2 * distance_yesterday) - 3
def total_distance_biked : ℕ := distance_yesterday + distance_today

theorem total_distance_biked_two_days : total_distance_biked = 33 :=
by {
  -- Given distance_yesterday = 12
  -- distance_today calculated as (2 * distance_yesterday) - 3 = 21
  -- total_distance_biked = distance_yesterday + distance_today = 33
  sorry
}

end total_distance_biked_two_days_l229_229170


namespace percentage_not_even_integers_l229_229341

variable (T : ℝ) (E : ℝ)
variables (h1 : 0.36 * T = E * 0.60) -- Condition 1 translated: 36% of T are even multiples of 3.
variables (h2 : E * 0.40)            -- Condition 2 translated: 40% of E are not multiples of 3.

theorem percentage_not_even_integers : 0.40 * T = T - E :=
by
  sorry

end percentage_not_even_integers_l229_229341


namespace benny_spent_amount_l229_229065

-- Definitions based on given conditions
def initial_amount : ℕ := 79
def amount_left : ℕ := 32

-- Proof problem statement
theorem benny_spent_amount :
  initial_amount - amount_left = 47 :=
sorry

end benny_spent_amount_l229_229065


namespace common_ratio_l229_229216

def geom_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q
def arith_seq (a : ℕ → ℝ) (x y z : ℕ) := 2 * a z = a x + a y

theorem common_ratio (a : ℕ → ℝ) (q : ℝ) (h_geom : geom_seq a q) (h_arith : arith_seq a 0 1 2) (h_nonzero : a 0 ≠ 0) : q = 1 ∨ q = -1/2 :=
by
  sorry

end common_ratio_l229_229216


namespace base9_to_decimal_l229_229796

theorem base9_to_decimal : (8 * 9^1 + 5 * 9^0) = 77 := 
by
  sorry

end base9_to_decimal_l229_229796


namespace cyclist_C_speed_l229_229761

variable (c d : ℕ)

def distance_to_meeting (c d : ℕ) : Prop :=
  d = c + 6 ∧
  90 + 30 = 120 ∧
  ((90 - 30) / c) = (120 / d) ∧
  (60 / c) = (120 / (c + 6))

theorem cyclist_C_speed : distance_to_meeting c d → c = 6 :=
by
  intro h
  -- To be filled in with the proof using the conditions
  sorry

end cyclist_C_speed_l229_229761


namespace find_f_prime_zero_l229_229833

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}

-- Condition given in the problem.
def f_def : ∀ x : ℝ, f x = x^2 + 2 * x * f' 1 := 
sorry

-- Statement we want to prove.
theorem find_f_prime_zero : f' 0 = -4 := 
sorry

end find_f_prime_zero_l229_229833


namespace cookie_sales_l229_229925

theorem cookie_sales (n : ℕ) (h1 : 1 ≤ n - 11) (h2 : 1 ≤ n - 2) (h3 : (n - 11) + (n - 2) < n) : n = 12 :=
sorry

end cookie_sales_l229_229925


namespace units_digit_27_mul_46_l229_229981

theorem units_digit_27_mul_46 : (27 * 46) % 10 = 2 :=
by 
  -- Definition of units digit
  have def_units_digit :=  (n : ℕ) => n % 10

  -- Step 1: units digit of 27 is 7
  have units_digit_27 : 27 % 10 = 7 := by norm_num
  
  -- Step 2: units digit of 46 is 6
  have units_digit_46 : 46 % 10 = 6 := by norm_num

  -- Step 3: multiple the units digits
  have step3 : 7 * 6 = 42 := by norm_num

  -- Step 4: Find the units digit of 42
  have units_digit_42 : 42 % 10 = 2 := by norm_num

  exact units_digit_42

end units_digit_27_mul_46_l229_229981


namespace smallest_number_100_divisors_l229_229042

theorem smallest_number_100_divisors : ∃ n : ℕ, (∀ d : ℕ, d ∣ n ↔ d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 28, 30, 35, 36, 40, 42, 45, 48, 56, 60, 63, 70, 72, 80, 84, 90, 105, 112, 120, 126, 140, 144, 168, 180, 210, 224, 240, 252, 280, 315, 336, 360, 420, 504, 560, 630, 840, 882, 1260, 1680, 1764, 1890, 2520, 3360, 5040, 7560, 12600, 15120, 25200, 45360}) ∧ n = 45360 := 
sorry

end smallest_number_100_divisors_l229_229042


namespace trader_gain_percentage_l229_229478

variable (x : ℝ) (cost_of_one_pen : ℝ := x) (selling_cost_90_pens : ℝ := 90 * x) (gain : ℝ := 30 * x)

theorem trader_gain_percentage :
  30 * cost_of_one_pen / (90 * cost_of_one_pen) * 100 = 33.33 := by
  sorry

end trader_gain_percentage_l229_229478


namespace simplify_power_multiplication_l229_229739

theorem simplify_power_multiplication (x : ℝ) : (-x) ^ 3 * (-x) ^ 2 = -x ^ 5 :=
by sorry

end simplify_power_multiplication_l229_229739


namespace range_of_x_l229_229586

theorem range_of_x (x p : ℝ) (h₀ : 0 ≤ p ∧ p ≤ 4) :
  x^2 + p * x > 4 * x + p - 3 → (x < 1 ∨ x > 3) :=
sorry

end range_of_x_l229_229586


namespace arctan_sum_l229_229826

theorem arctan_sum (θ₁ θ₂ : ℝ) (h₁ : θ₁ = Real.arctan (1/2))
                              (h₂ : θ₂ = Real.arctan 2) :
  θ₁ + θ₂ = Real.pi / 2 :=
by
  have : θ₁ + θ₂ + Real.pi / 2 = Real.pi := sorry
  linarith

end arctan_sum_l229_229826


namespace rhombus_longer_diagonal_length_l229_229639

theorem rhombus_longer_diagonal_length (side_length : ℕ) (shorter_diagonal : ℕ) 
  (h_side : side_length = 65) (h_short_diag : shorter_diagonal = 56) : 
  ∃ longer_diagonal : ℕ, longer_diagonal = 118 :=
by
  sorry

end rhombus_longer_diagonal_length_l229_229639


namespace percent_workday_in_meetings_l229_229389

def first_meeting_duration : ℕ := 30
def second_meeting_duration : ℕ := 3 * first_meeting_duration
def third_meeting_duration : ℕ := 2 * second_meeting_duration
def total_meeting_time : ℕ := first_meeting_duration + second_meeting_duration + third_meeting_duration
def workday_duration : ℕ := 10 * 60

theorem percent_workday_in_meetings : (total_meeting_time : ℚ) / workday_duration * 100 = 50 := by
  sorry

end percent_workday_in_meetings_l229_229389


namespace bread_last_days_l229_229596

def total_consumption_per_member_breakfast : ℕ := 4
def total_consumption_per_member_snacks : ℕ := 3
def total_consumption_per_member : ℕ := total_consumption_per_member_breakfast + total_consumption_per_member_snacks
def family_members : ℕ := 6
def daily_family_consumption : ℕ := family_members * total_consumption_per_member
def slices_per_loaf : ℕ := 10
def total_loaves : ℕ := 5
def total_bread_slices : ℕ := total_loaves * slices_per_loaf

theorem bread_last_days : total_bread_slices / daily_family_consumption = 1 :=
by
  sorry

end bread_last_days_l229_229596


namespace probability_of_yellow_second_is_one_third_l229_229177

noncomputable def P_red_A : ℚ := 3 / 9
noncomputable def P_yellow_B_given_red_A : ℚ := 6 / 10
noncomputable def P_black_A : ℚ := 6 / 9
noncomputable def P_yellow_C_given_black_A : ℚ := 2 / 10

def P_yellow_second : ℚ := 
  (P_red_A * P_yellow_B_given_red_A) + 
  (P_black_A * P_yellow_C_given_black_A)

theorem probability_of_yellow_second_is_one_third :
  P_yellow_second = 1 / 3 :=
by sorry

end probability_of_yellow_second_is_one_third_l229_229177


namespace units_digit_17_pow_2023_l229_229034

theorem units_digit_17_pow_2023 
  (cycle : ℕ → ℕ)
  (h1 : cycle 0 = 7)
  (h2 : cycle 1 = 9)
  (h3 : cycle 2 = 3)
  (h4 : cycle 3 = 1)
  (units_digit : ℕ → ℕ)
  (h_units : ∀ n, units_digit (17^n) = units_digit (7^n))
  (h_units_cycle : ∀ n, units_digit (7^n) = cycle (n % 4)) :
  units_digit (17^2023) = 3 :=
by
  sorry

end units_digit_17_pow_2023_l229_229034


namespace units_digit_27_mul_46_l229_229982

theorem units_digit_27_mul_46 : (27 * 46) % 10 = 2 :=
by 
  -- Definition of units digit
  have def_units_digit :=  (n : ℕ) => n % 10

  -- Step 1: units digit of 27 is 7
  have units_digit_27 : 27 % 10 = 7 := by norm_num
  
  -- Step 2: units digit of 46 is 6
  have units_digit_46 : 46 % 10 = 6 := by norm_num

  -- Step 3: multiple the units digits
  have step3 : 7 * 6 = 42 := by norm_num

  -- Step 4: Find the units digit of 42
  have units_digit_42 : 42 % 10 = 2 := by norm_num

  exact units_digit_42

end units_digit_27_mul_46_l229_229982


namespace common_tangent_y_intercept_l229_229285

noncomputable def circle_center_a : ℝ × ℝ := (1, 5)
noncomputable def circle_radius_a : ℝ := 3

noncomputable def circle_center_b : ℝ × ℝ := (15, 10)
noncomputable def circle_radius_b : ℝ := 10

theorem common_tangent_y_intercept :
  ∃ m b: ℝ, (m > 0) ∧ m = 700/1197 ∧ b = 7.416 ∧
  ∀ x y: ℝ, (y = m * x + b → ((x - 1)^2 + (y - 5)^2 = 9 ∨ (x - 15)^2 + (y - 10)^2 = 100)) := by
{
  sorry
}

end common_tangent_y_intercept_l229_229285


namespace sequences_correct_l229_229088

def arithmetic_sequence (a b c : ℕ) : Prop :=
  2 * b = a + c

def geometric_sequence (b a₁ b₁ : ℕ) : Prop :=
  a₁ * a₁ = b * b₁

noncomputable def sequence_a (n : ℕ) :=
  (n * (n + 1)) / 2

noncomputable def sequence_b (n : ℕ) :=
  ((n + 1) * (n + 1)) / 2

theorem sequences_correct :
  (∀ n : ℕ,
    n ≥ 1 →
    arithmetic_sequence (sequence_a n) (sequence_b n) (sequence_a (n + 1)) ∧
    geometric_sequence (sequence_b n) (sequence_a (n + 1)) (sequence_b (n + 1))) ∧
  (sequence_a 1 = 1) ∧
  (sequence_b 1 = 2) ∧
  (sequence_a 2 = 3) :=
by
  sorry

end sequences_correct_l229_229088


namespace max_profit_at_800_l229_229934

open Nat

def P (x : ℕ) : ℝ :=
  if h : 0 < x ∧ x ≤ 100 then 80
  else if h : 100 < x ∧ x ≤ 1000 then 82 - 0.02 * x
  else 0

def f (x : ℕ) : ℝ :=
  if h : 0 < x ∧ x ≤ 100 then 30 * x
  else if h : 100 < x ∧ x ≤ 1000 then 32 * x - 0.02 * x^2
  else 0

theorem max_profit_at_800 :
  ∀ x : ℕ, f x ≤ 12800 ∧ f 800 = 12800 :=
sorry

end max_profit_at_800_l229_229934


namespace triangle_area_inscribed_in_circle_l229_229941

theorem triangle_area_inscribed_in_circle (R : ℝ) 
    (h_pos : R > 0) 
    (h_ratio : ∃ (x : ℝ)(hx : x > 0), 2*x + 5*x + 17*x = 2*π) :
  (∃ (area : ℝ), area = (R^2 / 4)) :=
by
  sorry

end triangle_area_inscribed_in_circle_l229_229941


namespace mixedGasTemperature_is_correct_l229_229548

noncomputable def mixedGasTemperature (V₁ V₂ p₁ p₂ T₁ T₂ : ℝ) : ℝ := 
  (p₁ * V₁ + p₂ * V₂) / ((p₁ * V₁) / T₁ + (p₂ * V₂) / T₂)

theorem mixedGasTemperature_is_correct :
  mixedGasTemperature 2 3 3 4 400 500 = 462 := by
    sorry

end mixedGasTemperature_is_correct_l229_229548


namespace problem_EF_fraction_of_GH_l229_229885

theorem problem_EF_fraction_of_GH (E F G H : Type) 
  (GE EH GH GF FH EF : ℝ) 
  (h1 : GE = 3 * EH) 
  (h2 : GF = 8 * FH)
  (h3 : GH = GE + EH)
  (h4 : GH = GF + FH) : 
  EF = 5 / 36 * GH :=
by
  sorry

end problem_EF_fraction_of_GH_l229_229885


namespace rhombus_area_l229_229674

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 5) (h2 : d2 = 6) : 
  1 / 2 * d1 * d2 = 15 :=
by
  rw [h1, h2]
  norm_num

end rhombus_area_l229_229674


namespace cost_of_baking_soda_l229_229055

-- Definitions of the condition
def students : ℕ := 23
def cost_of_bow : ℕ := 5
def cost_of_vinegar : ℕ := 2
def total_cost_of_supplies : ℕ := 184

-- Main statement to prove
theorem cost_of_baking_soda : 
  (∀ (students : ℕ) (cost_of_bow : ℕ) (cost_of_vinegar : ℕ) (total_cost_of_supplies : ℕ),
    total_cost_of_supplies = students * (cost_of_bow + cost_of_vinegar) + students) → 
  total_cost_of_supplies = 23 * (5 + 2) + 23 → 
  184 = 23 * (5 + 2 + 1) :=
by
  sorry

end cost_of_baking_soda_l229_229055


namespace jake_weight_l229_229297

theorem jake_weight:
  ∃ (J S : ℝ), (J - 8 = 2 * S) ∧ (J + S = 290) ∧ (J = 196) :=
by
  sorry

end jake_weight_l229_229297


namespace number_of_combinations_with_constraints_l229_229318

theorem number_of_combinations_with_constraints :
  let n : ℕ := 6
  let k : ℕ := 2
  let total_combinations := Nat.choose n k
  let invalid_combinations := 2
  total_combinations - invalid_combinations = 13 :=
by
  let n : ℕ := 6
  let k : ℕ := 2
  let total_combinations := Nat.choose 6 2
  let invalid_combinations := 2
  show total_combinations - invalid_combinations = 13
  sorry

end number_of_combinations_with_constraints_l229_229318


namespace revision_cost_is_3_l229_229578

def cost_first_time (pages : ℕ) : ℝ := 5 * pages

def cost_for_revisions (rev1 rev2 : ℕ) (rev_cost : ℝ) : ℝ := (rev1 * rev_cost) + (rev2 * 2 * rev_cost)

def total_cost (pages rev1 rev2 : ℕ) (rev_cost : ℝ) : ℝ := 
  cost_first_time pages + cost_for_revisions rev1 rev2 rev_cost

theorem revision_cost_is_3 :
  ∀ (pages rev1 rev2 : ℕ) (total : ℝ),
      pages = 100 →
      rev1 = 30 →
      rev2 = 20 →
      total = 710 →
      total_cost pages rev1 rev2 3 = total :=
by
  intros pages rev1 rev2 total pages_eq rev1_eq rev2_eq total_eq
  sorry

end revision_cost_is_3_l229_229578


namespace ecology_club_probability_l229_229900

theorem ecology_club_probability :
  let total_ways := nat.choose 30 4,
      all_boys := nat.choose 12 4,
      all_girls := nat.choose 18 4,
      probability_all_boys_or_all_girls := (all_boys + all_girls) / total_ways,
      probability_at_least_one_boy_and_one_girl := 1 - probability_all_boys_or_all_girls in
  probability_at_least_one_boy_and_one_girl = 530 / 609 :=
by
  let total_ways := nat.choose 30 4
  let all_boys := nat.choose 12 4
  let all_girls := nat.choose 18 4
  let probability_all_boys_or_all_girls := (all_boys + all_girls) / total_ways
  let probability_at_least_one_boy_and_one_girl := 1 - probability_all_boys_or_all_girls
  have h1 : total_ways = 27405 := by norm_num
  have h2 : all_boys = 495 := by norm_num
  have h3 : all_girls = 3060 := by norm_num
  have h4 : probability_all_boys_or_all_girls = (495 + 3060) / 27405 := by norm_num
  have h5 : (495 + 3060) / 27405 = 395 / 3045 := by norm_num
  have h6 : 1 - 395 / 3045 = 2650 / 3045 := by norm_num
  have h7 : 2650 / 3045 = 530 / 609 := by norm_num
  show 1 - probability_all_boys_or_all_girls = 530 / 609
  exact congr_arg (λ x, 1 - x) h5 ▸ h7

end ecology_club_probability_l229_229900


namespace tires_in_parking_lot_l229_229858

theorem tires_in_parking_lot (n : ℕ) (m : ℕ) (h : 30 = n) (h' : m = 5) : n * m = 150 := by
  sorry

end tires_in_parking_lot_l229_229858


namespace product_evaluation_l229_229329

theorem product_evaluation :
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 :=
by sorry

end product_evaluation_l229_229329


namespace ratio_of_width_to_length_l229_229108

variable {w: ℕ}

theorem ratio_of_width_to_length (w: ℕ) (h1: 2*w + 2*10 = 30) (h2: w = 5) :
  ∃ (x y : ℕ), x = 1 ∧ y = 2 ∧ x.gcd y = 1 ∧ w / 10 = 1 / 2 := 
by
  sorry

end ratio_of_width_to_length_l229_229108


namespace smallest_four_digit_number_in_pascals_triangle_l229_229453

theorem smallest_four_digit_number_in_pascals_triangle :
  ∃ (n : ℕ), 1000 = (n choose min n) ∧ (n > 0 ∧ (∀ m < n, (m choose min m) < 1000)) :=
begin
  sorry
end

end smallest_four_digit_number_in_pascals_triangle_l229_229453


namespace smallest_four_digit_in_pascals_triangle_l229_229419

-- Define Pascal's triangle
def pascals_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 1
| n+1, k+1 => pascals_triangle n k + pascals_triangle n (k+1)

-- State the theorem
theorem smallest_four_digit_in_pascals_triangle : ∃ (n k : ℕ), pascals_triangle n k = 1000 ∧ ∀ m, m < 1000 → ∀ r s, pascals_triangle r s ≠ m :=
sorry

end smallest_four_digit_in_pascals_triangle_l229_229419


namespace sara_picked_6_pears_l229_229124

def total_pears : ℕ := 11
def tim_pears : ℕ := 5
def sara_pears : ℕ := total_pears - tim_pears

theorem sara_picked_6_pears : sara_pears = 6 := by
  sorry

end sara_picked_6_pears_l229_229124


namespace sum_of_reciprocals_negative_l229_229406

theorem sum_of_reciprocals_negative {a b c : ℝ} (h₁ : a + b + c = 0) (h₂ : a * b * c > 0) :
  1/a + 1/b + 1/c < 0 :=
sorry

end sum_of_reciprocals_negative_l229_229406


namespace moe_mowing_time_l229_229731

noncomputable def effective_swath_width_inches : ℝ := 30 - 6
noncomputable def effective_swath_width_feet : ℝ := (effective_swath_width_inches / 12)
noncomputable def lawn_width : ℝ := 180
noncomputable def lawn_length : ℝ := 120
noncomputable def walking_rate : ℝ := 4500
noncomputable def total_strips : ℝ := lawn_width / effective_swath_width_feet
noncomputable def total_distance : ℝ := total_strips * lawn_length
noncomputable def time_required : ℝ := total_distance / walking_rate

theorem moe_mowing_time :
  time_required = 2.4 := by
  sorry

end moe_mowing_time_l229_229731


namespace find_n_l229_229370

theorem find_n (q d : ℕ) (hq : q = 25) (hd : d = 10) (h : 30 * q + 20 * d = 5 * q + n * d) : n = 83 := by
  sorry

end find_n_l229_229370


namespace dice_rolls_diff_by_3_probability_l229_229954

-- Define a function to encapsulate the problem's statement
def probability_dice_diff_by_3 : ℚ := 1 / 6

-- Prove that given the conditions, the probability of rolling integers 
-- that differ by 3 when rolling a standard 6-sided die twice is 1/6.
theorem dice_rolls_diff_by_3_probability : 
  (probability (λ (x y : ℕ), x != y ∧ x - y = 3 ∨ y - x = 3) (finset.range 1 7 ×ˢ finset.range 1 7)) = probability_dice_diff_by_3 :=
sorry

end dice_rolls_diff_by_3_probability_l229_229954


namespace smallest_four_digit_in_pascal_triangle_l229_229471

noncomputable section

def is_in_pascal_triangle (n : ℕ) : Prop :=
  ∃ r c : ℕ, c ≤ r ∧ binomial r c = n

theorem smallest_four_digit_in_pascal_triangle :
  ¬ ∃ n : ℕ, is_in_pascal_triangle n ∧ n < 1000 ∧ (n ≥ 1000 ∨ n < 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_triangle_l229_229471


namespace relative_prime_in_consecutive_integers_l229_229734

theorem relative_prime_in_consecutive_integers (n : ℤ) : 
  ∃ k, n ≤ k ∧ k ≤ n + 5 ∧ ∀ m, n ≤ m ∧ m ≤ n + 5 ∧ m ≠ k → Int.gcd k m = 1 :=
sorry

end relative_prime_in_consecutive_integers_l229_229734


namespace geom_seq_mult_l229_229671

variable {α : Type*} [LinearOrderedField α]

def is_geom_seq (a : ℕ → α) :=
  ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)

theorem geom_seq_mult (a : ℕ → α) (h : is_geom_seq a) (hpos : ∀ n, 0 < a n) (h4_8 : a 4 * a 8 = 4) :
  a 5 * a 6 * a 7 = 8 := 
sorry

end geom_seq_mult_l229_229671


namespace range_of_a_l229_229365

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |2 * x - 1| + |x + 1| > a) ↔ a < 3 / 2 := by
  sorry

end range_of_a_l229_229365


namespace hans_deposit_l229_229359

noncomputable def calculate_deposit : ℝ :=
  let flat_fee := 30
  let kid_deposit := 2 * 3
  let adult_deposit := 8 * 6
  let senior_deposit := 5 * 4
  let student_deposit := 3 * 4.5
  let employee_deposit := 2 * 2.5
  let total_deposit_before_service := flat_fee + kid_deposit + adult_deposit + senior_deposit + student_deposit + employee_deposit
  let service_charge := total_deposit_before_service * 0.05
  total_deposit_before_service + service_charge

theorem hans_deposit : calculate_deposit = 128.63 :=
by
  sorry

end hans_deposit_l229_229359


namespace no_solution_for_steers_and_cows_purchase_l229_229164

theorem no_solution_for_steers_and_cows_purchase :
  ¬ ∃ (s c : ℕ), 30 * s + 32 * c = 1200 ∧ c > s :=
by
  sorry

end no_solution_for_steers_and_cows_purchase_l229_229164


namespace cos_alpha_minus_half_beta_l229_229360

theorem cos_alpha_minus_half_beta
  (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : -π / 2 < β ∧ β < 0)
  (h3 : Real.cos (π / 4 + α) = 1 / 3)
  (h4 : Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3) :
  Real.cos (α - β / 2) = Real.sqrt 6 / 3 :=
by
  sorry

end cos_alpha_minus_half_beta_l229_229360
