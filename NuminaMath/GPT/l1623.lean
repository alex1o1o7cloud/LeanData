import Mathlib

namespace remainder_when_divided_l1623_162359

theorem remainder_when_divided (k : ℕ) (h_pos : 0 < k) (h_rem : 80 % k = 8) : 150 % (k^2) = 69 := by 
  sorry

end remainder_when_divided_l1623_162359


namespace numbers_divisible_l1623_162364

theorem numbers_divisible (n : ℕ) (d1 d2 : ℕ) (lcm_d1_d2 : ℕ) (limit : ℕ) (h_lcm: lcm d1 d2 = lcm_d1_d2) (h_limit : limit = 2011)
(h_d1 : d1 = 117) (h_d2 : d2 = 2) : 
  ∃ k : ℕ, k = 8 ∧ ∀ m : ℕ, m < limit → (m % lcm_d1_d2 = 0 ↔ ∃ i : ℕ, i < k ∧ m = lcm_d1_d2 * (i + 1)) :=
by
  sorry

end numbers_divisible_l1623_162364


namespace geometric_sequence_common_ratio_l1623_162380

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, S n = a 0 * (1 - q ^ n) / (1 - q))  -- Sum of geometric series
  (h2 : a 3 = S 3 + 1) : q = 3 :=
by sorry

end geometric_sequence_common_ratio_l1623_162380


namespace max_value_of_f_l1623_162322

noncomputable def f (x : Real) := 2 * (Real.sin x) ^ 2 - (Real.tan x) ^ 2

theorem max_value_of_f : 
  ∃ (x : Real), f x = 3 - 2 * Real.sqrt 2 := 
sorry

end max_value_of_f_l1623_162322


namespace simple_interest_initial_amount_l1623_162379

theorem simple_interest_initial_amount :
  ∃ P : ℝ, (P + P * 0.04 * 5 = 900) ∧ P = 750 :=
by
  sorry

end simple_interest_initial_amount_l1623_162379


namespace ratio_of_buyers_l1623_162352

theorem ratio_of_buyers (B Y T : ℕ) (hB : B = 50) 
  (hT : T = Y + 40) (hTotal : B + Y + T = 140) : 
  (Y : ℚ) / B = 1 / 2 :=
by 
  sorry

end ratio_of_buyers_l1623_162352


namespace fruit_basket_count_l1623_162389

/-- We have seven identical apples and twelve identical oranges.
    A fruit basket must contain at least one piece of fruit.
    Prove that the number of different fruit baskets we can make
    is 103. -/
theorem fruit_basket_count :
  let apples := 7
  let oranges := 12
  let total_possible_baskets := (apples + 1) * (oranges + 1) - 1
  total_possible_baskets = 103 :=
by
  let apples := 7
  let oranges := 12
  let total_possible_baskets := (apples + 1) * (oranges + 1) - 1
  show total_possible_baskets = 103
  sorry

end fruit_basket_count_l1623_162389


namespace quadratic_expression_odd_quadratic_expression_not_square_l1623_162382

theorem quadratic_expression_odd (n : ℕ) : 
  (n^2 + n + 1) % 2 = 1 := 
by sorry

theorem quadratic_expression_not_square (n : ℕ) : 
  ¬ ∃ (m : ℕ), m^2 = n^2 + n + 1 := 
by sorry

end quadratic_expression_odd_quadratic_expression_not_square_l1623_162382


namespace simplify_and_evaluate_l1623_162313

variable (x y : ℝ)

theorem simplify_and_evaluate (h : x / y = 3) : 
  (1 + y^2 / (x^2 - y^2)) * (x - y) / x = 3 / 4 :=
by
  sorry

end simplify_and_evaluate_l1623_162313


namespace possible_items_l1623_162387

-- Mathematical definitions derived from the conditions.
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 20083

-- The theorem stating the possible number of items Kolya could have bought.
theorem possible_items (a n : ℕ) (hn : n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
sorry

end possible_items_l1623_162387


namespace basketball_weight_calc_l1623_162345

-- Define the variables and conditions
variable (weight_basketball weight_watermelon : ℕ)
variable (h1 : 8 * weight_basketball = 4 * weight_watermelon)
variable (h2 : weight_watermelon = 32)

-- Statement to prove
theorem basketball_weight_calc : weight_basketball = 16 :=
by
  sorry

end basketball_weight_calc_l1623_162345


namespace experiment_variance_l1623_162381

noncomputable def probability_of_success : ℚ := 5/9

noncomputable def variance_of_binomial (n : ℕ) (p : ℚ) : ℚ :=
  n * p * (1 - p)

def number_of_experiments : ℕ := 30

theorem experiment_variance :
  variance_of_binomial number_of_experiments probability_of_success = 200/27 :=
by
  sorry

end experiment_variance_l1623_162381


namespace division_of_cubics_l1623_162317

theorem division_of_cubics (a b c : ℕ) (h_a : a = 7) (h_b : b = 6) (h_c : c = 1) :
  (a^3 + b^3) / (a^2 - a * b + b^2 + c) = 559 / 44 :=
by
  rw [h_a, h_b, h_c]
  -- After these substitutions, the problem is reduced to proving
  -- (7^3 + 6^3) / (7^2 - 7 * 6 + 6^2 + 1) = 559 / 44
  sorry

end division_of_cubics_l1623_162317


namespace fixed_point_of_function_l1623_162393

theorem fixed_point_of_function :
  (4, 4) ∈ { p : ℝ × ℝ | ∃ x : ℝ, p = (x, 2^(x-4) + 3) } :=
by
  sorry

end fixed_point_of_function_l1623_162393


namespace unique_real_solution_system_l1623_162329

/-- There is exactly one real solution (x, y, z, w) to the given system of equations:
  x + 1 = z + w + z * w * x,
  y - 1 = w + x + w * x * y,
  z + 2 = x + y + x * y * z,
  w - 2 = y + z + y * z * w
-/
theorem unique_real_solution_system :
  let eq1 (x y z w : ℝ) := x + 1 = z + w + z * w * x
  let eq2 (x y z w : ℝ) := y - 1 = w + x + w * x * y
  let eq3 (x y z w : ℝ) := z + 2 = x + y + x * y * z
  let eq4 (x y z w : ℝ) := w - 2 = y + z + y * z * w
  ∃! (x y z w : ℝ), eq1 x y z w ∧ eq2 x y z w ∧ eq3 x y z w ∧ eq4 x y z w := by {
  sorry
}

end unique_real_solution_system_l1623_162329


namespace ruth_hours_per_week_l1623_162339

theorem ruth_hours_per_week :
  let daily_hours := 8
  let days_per_week := 5
  let monday_wednesday_friday := 3
  let tuesday_thursday := 2
  let percentage_to_hours (percent : ℝ) (hours : ℕ) : ℝ := percent * hours
  let total_weekly_hours := daily_hours * days_per_week
  let monday_wednesday_friday_math_hours := percentage_to_hours 0.25 daily_hours
  let monday_wednesday_friday_science_hours := percentage_to_hours 0.15 daily_hours
  let tuesday_thursday_math_hours := percentage_to_hours 0.2 daily_hours
  let tuesday_thursday_science_hours := percentage_to_hours 0.35 daily_hours
  let tuesday_thursday_history_hours := percentage_to_hours 0.15 daily_hours
  let weekly_math_hours := monday_wednesday_friday_math_hours * monday_wednesday_friday + tuesday_thursday_math_hours * tuesday_thursday
  let weekly_science_hours := monday_wednesday_friday_science_hours * monday_wednesday_friday + tuesday_thursday_science_hours * tuesday_thursday
  let weekly_history_hours := tuesday_thursday_history_hours * tuesday_thursday
  let total_hours := weekly_math_hours + weekly_science_hours + weekly_history_hours
  total_hours = 20.8 := by
  sorry

end ruth_hours_per_week_l1623_162339


namespace percentage_decrease_l1623_162315

-- Define conditions and variables
def original_selling_price : ℝ := 659.9999999999994
def profit_rate1 : ℝ := 0.10
def increase_in_selling_price : ℝ := 42
def profit_rate2 : ℝ := 0.30

-- Define the actual proof problem
theorem percentage_decrease (C C_prime : ℝ) 
    (h1 : 1.10 * C = original_selling_price) 
    (h2 : 1.30 * C_prime = original_selling_price + increase_in_selling_price) : 
    ((C - C_prime) / C) * 100 = 10 := 
sorry

end percentage_decrease_l1623_162315


namespace selling_price_is_correct_l1623_162328

-- Define the constants used in the problem
noncomputable def cost_price : ℝ := 540
noncomputable def markup_percentage : ℝ := 0.15
noncomputable def discount_percentage : ℝ := 26.570048309178745 / 100

-- Define the conditions in the problem
noncomputable def marked_price : ℝ := cost_price * (1 + markup_percentage)
noncomputable def discount_amount : ℝ := marked_price * discount_percentage
noncomputable def selling_price : ℝ := marked_price - discount_amount

-- Theorem stating the problem
theorem selling_price_is_correct : selling_price = 456 := by 
  sorry

end selling_price_is_correct_l1623_162328


namespace find_theta_even_fn_l1623_162369

noncomputable def f (x θ : ℝ) := Real.sin (x + θ) + Real.cos (x + θ)

theorem find_theta_even_fn (θ : ℝ) (hθ: 0 ≤ θ ∧ θ ≤ π / 2) 
  (h: ∀ x : ℝ, f x θ = f (-x) θ) : θ = π / 4 :=
by sorry

end find_theta_even_fn_l1623_162369


namespace island_solution_l1623_162395

-- Definitions based on conditions
def is_liar (n : ℕ) (m : ℕ) : Prop := n = m + 2 ∨ n = m - 2
def is_truth_teller (n : ℕ) (m : ℕ) : Prop := n = m

-- Residents' statements
def first_resident_statement (liars : ℕ) (truth_tellers : ℕ) : Prop :=
  is_truth_teller liars 1001 ∧ is_truth_teller truth_tellers 1002 ∨
  is_liar liars 1001 ∧ is_liar truth_tellers 1002

def second_resident_statement (liars : ℕ) (truth_tellers : ℕ) : Prop :=
  is_truth_teller liars 1000 ∧ is_truth_teller truth_tellers 999 ∨
  is_liar liars 1000 ∧ is_liar truth_tellers 999

-- Proving the correct number of liars and truth-tellers, and identifying the residents
theorem island_solution :
  ∃ (liars : ℕ) (truth_tellers : ℕ),
    first_resident_statement (liars + 1) (truth_tellers + 1) ∧
    second_resident_statement (liars + 1) (truth_tellers + 1) ∧
    liars = 1000 ∧ truth_tellers = 1000 ∧
    first_resident_statement liars truth_tellers ∧ second_resident_statement liars truth_tellers :=
by
  sorry

end island_solution_l1623_162395


namespace fraction_meaningful_l1623_162343

theorem fraction_meaningful (x : ℝ) : (x ≠ 1) ↔ ¬ (x - 1 = 0) :=
by
  sorry

end fraction_meaningful_l1623_162343


namespace arithmetic_series_sum_after_multiplication_l1623_162357

theorem arithmetic_series_sum_after_multiplication :
  let s : List ℕ := [110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
  3 * s.sum = 3435 := by
  sorry

end arithmetic_series_sum_after_multiplication_l1623_162357


namespace frood_points_smallest_frood_points_l1623_162361

theorem frood_points (n : ℕ) (h : n > 9) : (n * (n + 1)) / 2 > 5 * n :=
by {
  sorry
}

noncomputable def smallest_n : ℕ := 10

theorem smallest_frood_points (m : ℕ) (h : (m * (m + 1)) / 2 > 5 * m) : 10 ≤ m :=
by {
  sorry
}

end frood_points_smallest_frood_points_l1623_162361


namespace solve_quadratic_eq_l1623_162326

theorem solve_quadratic_eq (x : ℝ) : x^2 - x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
by sorry

end solve_quadratic_eq_l1623_162326


namespace sum_lent_is_correct_l1623_162354

variable (P : ℝ) -- Sum lent
variable (R : ℝ) -- Interest rate
variable (T : ℝ) -- Time period
variable (I : ℝ) -- Simple interest

-- Conditions
axiom interest_rate : R = 8
axiom time_period : T = 8
axiom simple_interest_formula : I = (P * R * T) / 100
axiom interest_condition : I = P - 900

-- The proof problem
theorem sum_lent_is_correct : P = 2500 := by
  -- The proof is skipped
  sorry

end sum_lent_is_correct_l1623_162354


namespace arithmetic_sequence_length_l1623_162370

theorem arithmetic_sequence_length : 
  let a := 11
  let d := 5
  let l := 101
  ∃ n : ℕ, a + (n-1) * d = l ∧ n = 19 := 
by
  sorry

end arithmetic_sequence_length_l1623_162370


namespace fraction_sum_is_ten_l1623_162375

theorem fraction_sum_is_ten :
  (1 / 10) + (2 / 10) + (3 / 10) + (4 / 10) + (5 / 10) + (6 / 10) + (7 / 10) + (8 / 10) + (9 / 10) + (55 / 10) = 10 :=
by
  sorry

end fraction_sum_is_ten_l1623_162375


namespace dot_product_of_PA_PB_l1623_162378

theorem dot_product_of_PA_PB
  (A B P: ℝ × ℝ)
  (h_circle : ∀ (x y : ℝ), x ^ 2 + y ^ 2 + 4 * x - 5 = 0 → (x, y) = A ∨ (x, y) = B)
  (h_midpoint : (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 1)
  (h_x_axis_intersect : P.2 = 0 ∧ (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = -5) :
  (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = -5 :=
sorry

end dot_product_of_PA_PB_l1623_162378


namespace number_of_sets_given_to_sister_l1623_162321

-- Defining the total number of cards, sets given to his brother and friend, total cards given away,
-- number of cards per set, and expected answer for sets given to his sister.
def total_cards := 365
def sets_given_to_brother := 8
def sets_given_to_friend := 2
def total_cards_given_away := 195
def cards_per_set := 13
def sets_given_to_sister := 5

theorem number_of_sets_given_to_sister :
  sets_given_to_brother * cards_per_set + 
  sets_given_to_friend * cards_per_set + 
  sets_given_to_sister * cards_per_set = total_cards_given_away :=
by
  -- It skips the proof but ensures the statement is set up correctly.
  sorry

end number_of_sets_given_to_sister_l1623_162321


namespace base_seven_representation_l1623_162311

theorem base_seven_representation 
  (k : ℕ) 
  (h1 : 4 ≤ k) 
  (h2 : k < 8) 
  (h3 : 500 / k^3 < k) 
  (h4 : 500 ≥ k^3) 
  : ∃ n m o p : ℕ, (500 = n * k^3 + m * k^2 + o * k + p) ∧ (p % 2 = 1) ∧ (n ≠ 0 ) :=
sorry

end base_seven_representation_l1623_162311


namespace cost_of_jam_l1623_162350

theorem cost_of_jam (N B J H : ℕ) (h : N > 1) (cost_eq : N * (6 * B + 7 * J + 4 * H) = 462) : 7 * J * N = 462 :=
by
  sorry

end cost_of_jam_l1623_162350


namespace steak_chicken_ratio_l1623_162366

variable (S C : ℕ)

theorem steak_chicken_ratio (h1 : S + C = 80) (h2 : 25 * S + 18 * C = 1860) : S = 3 * C :=
by
  sorry

end steak_chicken_ratio_l1623_162366


namespace prove_a_minus_b_plus_c_eq_3_l1623_162300

variable {a b c m n : ℝ}

theorem prove_a_minus_b_plus_c_eq_3 
    (h : ∀ x : ℝ, m * x^2 - n * x + 3 = a * (x - 1)^2 + b * (x - 1) + c) :
    a - b + c = 3 :=
sorry

end prove_a_minus_b_plus_c_eq_3_l1623_162300


namespace san_antonio_to_austin_buses_passed_l1623_162319

def departure_schedule (departure_time_A_to_S departure_time_S_to_A travel_time : ℕ) : Prop :=
  ∀ t, (t < travel_time) →
       (∃ n, t = (departure_time_A_to_S + n * 60)) ∨
       (∃ m, t = (departure_time_S_to_A + m * 60)) →
       t < travel_time

theorem san_antonio_to_austin_buses_passed :
  let departure_time_A_to_S := 30  -- Austin to San Antonio buses leave every hour on the half-hour (e.g., 00:30, 1:30, ...)
  let departure_time_S_to_A := 0   -- San Antonio to Austin buses leave every hour on the hour (e.g., 00:00, 1:00, ...)
  let travel_time := 6 * 60        -- The trip takes 6 hours, or 360 minutes
  departure_schedule departure_time_A_to_S departure_time_S_to_A travel_time →
  ∃ count, count = 12 := 
by
  sorry

end san_antonio_to_austin_buses_passed_l1623_162319


namespace regular_dinosaur_weight_l1623_162318

namespace DinosaurWeight

-- Given Conditions
def Barney_weight (x : ℝ) : ℝ := 5 * x + 1500
def combined_weight (x : ℝ) : ℝ := Barney_weight x + 5 * x

-- Target Proof
theorem regular_dinosaur_weight :
  (∃ x : ℝ, combined_weight x = 9500) -> 
  ∃ x : ℝ, x = 800 :=
by {
  sorry
}

end DinosaurWeight

end regular_dinosaur_weight_l1623_162318


namespace speed_of_stream_l1623_162392

theorem speed_of_stream (v : ℝ) (h_still : ∀ (d : ℝ), d / (3 - v) = 2 * d / (3 + v)) : v = 1 :=
by
  sorry

end speed_of_stream_l1623_162392


namespace child_tickets_sold_l1623_162377

theorem child_tickets_sold (A C : ℕ) (h1 : A + C = 130) (h2 : 12 * A + 4 * C = 840) : C = 90 := by
  sorry

end child_tickets_sold_l1623_162377


namespace ninety_eight_squared_l1623_162363

theorem ninety_eight_squared : 98^2 = 9604 := by
  sorry

end ninety_eight_squared_l1623_162363


namespace price_of_sugar_and_salt_l1623_162331

theorem price_of_sugar_and_salt:
  (∀ (sugar_price salt_price : ℝ), 2 * sugar_price + 5 * salt_price = 5.50 ∧ sugar_price = 1.50 →
  3 * sugar_price + salt_price = 5) := 
by 
  sorry

end price_of_sugar_and_salt_l1623_162331


namespace compound_interest_l1623_162372

theorem compound_interest (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) (CI : ℝ) :
  SI = 40 → R = 5 → T = 2 → SI = (P * R * T) / 100 → CI = P * ((1 + R / 100) ^ T - 1) → CI = 41 :=
by sorry

end compound_interest_l1623_162372


namespace valid_plantings_count_l1623_162304

-- Define the grid structure
structure Grid3x3 :=
  (sections : Fin 9 → String)

noncomputable def crops := ["corn", "wheat", "soybeans", "potatoes", "oats"]

-- Define the adjacency relationships and restrictions as predicates
def adjacent (i j : Fin 9) : Prop :=
  (i = j + 1 ∧ j % 3 ≠ 2) ∨ (i = j - 1 ∧ i % 3 ≠ 2) ∨ (i = j + 3) ∨ (i = j - 3)

def valid_crop_planting (g : Grid3x3) : Prop :=
  ∀ i j, adjacent i j →
    (¬(g.sections i = "corn" ∧ g.sections j = "wheat") ∧ 
    ¬(g.sections i = "wheat" ∧ g.sections j = "corn") ∧
    ¬(g.sections i = "soybeans" ∧ g.sections j = "potatoes") ∧
    ¬(g.sections i = "potatoes" ∧ g.sections j = "soybeans") ∧
    ¬(g.sections i = "oats" ∧ g.sections j = "potatoes") ∧ 
    ¬(g.sections i = "potatoes" ∧ g.sections j = "oats"))

noncomputable def count_valid_plantings : Nat :=
  -- Placeholder for the actual count computing function
  sorry

theorem valid_plantings_count : count_valid_plantings = 5 :=
  sorry

end valid_plantings_count_l1623_162304


namespace radius_of_inscribed_circle_l1623_162399

variable (A p s r : ℝ)

theorem radius_of_inscribed_circle (h1 : A = 2 * p) (h2 : A = r * s) (h3 : p = 2 * s) : r = 4 := by
  sorry

end radius_of_inscribed_circle_l1623_162399


namespace wreaths_per_greek_l1623_162365

variable (m : ℕ) (m_pos : m > 0)

theorem wreaths_per_greek : ∃ x, x = 4 * m := 
sorry

end wreaths_per_greek_l1623_162365


namespace combined_tax_rate_l1623_162391

theorem combined_tax_rate (Mork_income Mindy_income : ℝ) (h1 : Mindy_income = 4 * Mork_income) :
  let Mork_tax := 0.45 * Mork_income;
  let Mindy_tax := 0.15 * Mindy_income;
  let combined_tax := Mork_tax + Mindy_tax;
  let combined_income := Mork_income + Mindy_income;
  combined_tax / combined_income * 100 = 21 := 
by
  sorry

end combined_tax_rate_l1623_162391


namespace stationery_box_cost_l1623_162335

theorem stationery_box_cost (unit_price : ℕ) (quantity : ℕ) (total_cost : ℕ) :
  unit_price = 23 ∧ quantity = 3 ∧ total_cost = 3 * 23 → total_cost = 69 :=
by
  sorry

end stationery_box_cost_l1623_162335


namespace harry_travel_time_l1623_162368

variables (bus_time1 bus_time2 : ℕ) (walk_ratio : ℕ)
-- Conditions based on the problem
-- Harry has already been sat on the bus for 15 minutes.
def part1_time : ℕ := 15

-- and he knows the rest of the journey will take another 25 minutes.
def part2_time : ℕ := 25

-- The total bus journey time
def total_bus_time : ℕ := part1_time + part2_time

-- The walk from the bus stop to his house will take half the amount of time the bus journey took.
def walk_time : ℕ := total_bus_time / 2

-- Total travel time
def total_travel_time : ℕ := total_bus_time + walk_time

-- Rewrite the proof problem statement
theorem harry_travel_time : total_travel_time = 60 := by
  sorry

end harry_travel_time_l1623_162368


namespace solution_l1623_162314

noncomputable def F (a b c : ℝ) := a * (b ^ 3) + c

theorem solution (a : ℝ) (h : F a 2 3 = F a 3 10) : a = -7 / 19 := sorry

end solution_l1623_162314


namespace range_of_a_l1623_162344

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a * x > 0) → a < 1 :=
by
  sorry

end range_of_a_l1623_162344


namespace least_k_cubed_divisible_by_168_l1623_162398

theorem least_k_cubed_divisible_by_168 : ∃ k : ℤ, (k ^ 3) % 168 = 0 ∧ ∀ n : ℤ, (n ^ 3) % 168 = 0 → k ≤ n :=
sorry

end least_k_cubed_divisible_by_168_l1623_162398


namespace opposite_of_one_sixth_l1623_162307

theorem opposite_of_one_sixth : (-(1 / 6) : ℚ) = -1 / 6 := 
by
  sorry

end opposite_of_one_sixth_l1623_162307


namespace number_of_outcomes_exactly_two_evening_l1623_162341

theorem number_of_outcomes_exactly_two_evening (chickens : Finset ℕ) (h_chickens : chickens.card = 4) 
    (day_places evening_places : ℕ) (h_day_places : day_places = 2) (h_evening_places : evening_places = 3) :
    ∃ n, n = (chickens.card.choose 2) ∧ n = 6 :=
by
  sorry

end number_of_outcomes_exactly_two_evening_l1623_162341


namespace minimum_colors_needed_l1623_162371

def paint_fence_colors (B : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, B i ≠ B (i + 2)) ∧
  (∀ i : ℕ, B i ≠ B (i + 3)) ∧
  (∀ i : ℕ, B i ≠ B (i + 5))

theorem minimum_colors_needed : ∃ (c : ℕ), 
  (∀ B : ℕ → ℕ, paint_fence_colors B → c ≥ 3) ∧
  (∃ B : ℕ → ℕ, paint_fence_colors B ∧ c = 3) :=
sorry

end minimum_colors_needed_l1623_162371


namespace prop_range_a_l1623_162337

theorem prop_range_a (a : ℝ) 
  (p : ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 4 → x^2 ≥ a)
  (q : ∃ (x : ℝ), x^2 + 2 * a * x + (2 - a) = 0)
  : a = 1 ∨ a ≤ -2 :=
sorry

end prop_range_a_l1623_162337


namespace sin_2theta_value_l1623_162388

theorem sin_2theta_value (θ : ℝ) (h : ∑' n, (Real.sin θ)^(2 * n) = 3) : Real.sin (2 * θ) = (2 * Real.sqrt 2) / 3 :=
by
  sorry

end sin_2theta_value_l1623_162388


namespace pears_remaining_l1623_162367

theorem pears_remaining (K_picked : ℕ) (M_picked : ℕ) (S_picked : ℕ)
                        (K_gave : ℕ) (M_gave : ℕ) (S_gave : ℕ)
                        (hK_pick : K_picked = 47)
                        (hM_pick : M_picked = 12)
                        (hS_pick : S_picked = 22)
                        (hK_give : K_gave = 46)
                        (hM_give : M_gave = 5)
                        (hS_give : S_gave = 15) :
  (K_picked - K_gave) + (M_picked - M_gave) + (S_picked - S_gave) = 15 :=
by
  sorry

end pears_remaining_l1623_162367


namespace find_max_z_plus_x_l1623_162394

theorem find_max_z_plus_x : 
  (∃ (x y z t: ℝ), x^2 + y^2 = 4 ∧ z^2 + t^2 = 9 ∧ xt + yz ≥ 6 ∧ z + x = 5) :=
sorry

end find_max_z_plus_x_l1623_162394


namespace stock_price_end_of_third_year_l1623_162310

def stock_price_after_years (initial_price : ℝ) (year1_increase : ℝ) (year2_decrease : ℝ) (year3_increase : ℝ) : ℝ :=
  let price_after_year1 := initial_price * (1 + year1_increase)
  let price_after_year2 := price_after_year1 * (1 - year2_decrease)
  let price_after_year3 := price_after_year2 * (1 + year3_increase)
  price_after_year3

theorem stock_price_end_of_third_year :
  stock_price_after_years 120 0.80 0.30 0.50 = 226.8 := 
by
  sorry

end stock_price_end_of_third_year_l1623_162310


namespace problem_solution_l1623_162349

noncomputable def f (x : ℝ) : ℝ := if 0 ≤ x ∧ x ≤ 1 then x^2 else sorry

lemma f_odd (x : ℝ) : f (-x) = -f x := sorry

lemma f_xplus1_even (x : ℝ) : f (x + 1) = f (-x + 1) := sorry

theorem problem_solution : f 2015 = -1 := 
by 
  sorry

end problem_solution_l1623_162349


namespace remainder_of_modified_expression_l1623_162373

theorem remainder_of_modified_expression (x y u v : ℕ) (h : x = u * y + v) (hy_pos : y > 0) (hv_bound : 0 ≤ v ∧ v < y) :
  (x + 3 * u * y + 4) % y = v + 4 :=
by sorry

end remainder_of_modified_expression_l1623_162373


namespace find_k_for_two_identical_solutions_l1623_162396

theorem find_k_for_two_identical_solutions (k : ℝ) :
  (∃ x : ℝ, x^2 = 4 * x + k) ∧ (∀ x : ℝ, x^2 = 4 * x + k → x = 2) ↔ k = -4 :=
by
  sorry

end find_k_for_two_identical_solutions_l1623_162396


namespace sqrt_meaningful_range_l1623_162362

theorem sqrt_meaningful_range (x : ℝ) (h : 3 * x - 5 ≥ 0) : x ≥ 5 / 3 :=
sorry

end sqrt_meaningful_range_l1623_162362


namespace train_length_l1623_162353

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (speed_conversion : speed_kmh = 40) 
  (time_condition : time_s = 27) : 
  (speed_kmh * 1000 / 3600 * time_s = 300) := 
by
  sorry

end train_length_l1623_162353


namespace calculate_total_weight_l1623_162324

-- Define the given conditions as constants and calculations
def silverware_weight_per_piece : ℕ := 4
def pieces_per_setting : ℕ := 3
def plate_weight_per_piece : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def backup_settings : ℕ := 20

-- Calculate individual weights and total settings
def silverware_weight_per_setting := silverware_weight_per_piece * pieces_per_setting
def plate_weight_per_setting := plate_weight_per_piece * plates_per_setting
def weight_per_setting := silverware_weight_per_setting + plate_weight_per_setting
def total_settings := (tables * settings_per_table) + backup_settings

-- Calculate the total weight of all settings
def total_weight : ℕ := total_settings * weight_per_setting

-- The theorem to prove that the total weight is 5040 ounces
theorem calculate_total_weight : total_weight = 5040 :=
by
  -- The proof steps are omitted
  sorry

end calculate_total_weight_l1623_162324


namespace fewest_seats_to_be_occupied_l1623_162355

theorem fewest_seats_to_be_occupied (n : ℕ) (h : n = 120) : ∃ m, m = 40 ∧
  ∀ a b, a + b = n → a ≥ m → ∀ x, (x > 0 ∧ x ≤ n) → (x > 1 → a = m → a + (b / 2) ≥ n / 3) :=
sorry

end fewest_seats_to_be_occupied_l1623_162355


namespace eventually_periodic_of_rational_cubic_l1623_162312

noncomputable def is_rational_sequence (P : ℚ → ℚ) (q : ℕ → ℚ) :=
  ∀ n : ℕ, q (n + 1) = P (q n)

theorem eventually_periodic_of_rational_cubic (P : ℚ → ℚ) (q : ℕ → ℚ) (hP : ∃ a b c d : ℚ, ∀ x : ℚ, P x = a * x^3 + b * x^2 + c * x + d) (hq : is_rational_sequence P q) : 
  ∃ k ≥ 1, ∀ n ≥ 1, q (n + k) = q n := 
sorry

end eventually_periodic_of_rational_cubic_l1623_162312


namespace min_value_of_expression_min_value_achieved_l1623_162384

theorem min_value_of_expression (x : ℝ) (h : x > 0) : 
  (x + 3 / (x + 1)) ≥ 2 * Real.sqrt 3 - 1 := 
sorry

theorem min_value_achieved (x : ℝ) (h : x = Real.sqrt 3 - 1) : 
  (x + 3 / (x + 1)) = 2 * Real.sqrt 3 - 1 := 
sorry

end min_value_of_expression_min_value_achieved_l1623_162384


namespace triangle_third_side_lengths_l1623_162383

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_third_side_lengths :
  ∃ (n : ℕ), n = 15 ∧ ∀ x : ℕ, (3 < x ∧ x < 19) → (∃ k, x = k) :=
by
  sorry

end triangle_third_side_lengths_l1623_162383


namespace expression_evaluation_l1623_162348

theorem expression_evaluation : -20 + 8 * (5 ^ 2 - 3) = 156 := by
  sorry

end expression_evaluation_l1623_162348


namespace problem1_problem2_l1623_162303

-- Problem 1
theorem problem1 : 3 * (Real.sqrt 3 + Real.sqrt 2) - 2 * (Real.sqrt 3 - Real.sqrt 2) = Real.sqrt 3 + 5 * Real.sqrt 2 :=
by
  sorry

-- Problem 2
theorem problem2 : abs (Real.sqrt 3 - Real.sqrt 2) + abs (Real.sqrt 3 - 2) + Real.sqrt 4 = 4 - Real.sqrt 2 :=
by
  sorry

end problem1_problem2_l1623_162303


namespace trigonometric_identity_l1623_162386

variable (α : Real)

theorem trigonometric_identity 
  (h : Real.sin (π / 6 + α) = Real.sqrt 3 / 3) :
  Real.cos (π / 3 - α) = Real.sqrt 3 / 3 :=
sorry

end trigonometric_identity_l1623_162386


namespace brick_length_l1623_162330

theorem brick_length (L : ℝ) :
  (∀ (V_wall V_brick : ℝ),
    V_wall = 29 * 100 * 2 * 100 * 0.75 * 100 ∧
    V_wall = 29000 * V_brick ∧
    V_brick = L * 10 * 7.5) →
  L = 20 :=
by
  intro h
  sorry

end brick_length_l1623_162330


namespace problem_statement_l1623_162325

/-- A predicate that checks if the numbers from 1 to 2n can be split into two groups 
    such that the sum of the product of the elements of each group is divisible by 2n - 1. -/
def valid_split (n : ℕ) : Prop :=
  ∃ (a b : Finset ℕ), 
  a ∪ b = Finset.range (2 * n) ∧
  a ∩ b = ∅ ∧
  (2 * n) ∣ (a.prod id + b.prod id - 1)

theorem problem_statement : 
  ∀ n : ℕ, n > 0 → valid_split n ↔ (n = 1 ∨ ∃ a : ℕ, n = 2^a ∧ a ≥ 1) :=
by
  sorry

end problem_statement_l1623_162325


namespace at_least_one_boy_selected_l1623_162333

-- Define the number of boys and girls
def boys : ℕ := 6
def girls : ℕ := 2

-- Define the total group and the total selected
def total_people : ℕ := boys + girls
def selected_people : ℕ := 3

-- Statement: In any selection of 3 people from the group, the selection contains at least one boy
theorem at_least_one_boy_selected :
  ∀ (selection : Finset ℕ), selection.card = selected_people → selection.card > girls :=
sorry

end at_least_one_boy_selected_l1623_162333


namespace sequence_sixth_term_is_364_l1623_162356

theorem sequence_sixth_term_is_364 (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 7) (h3 : a 3 = 20)
  (h4 : ∀ n, a (n + 1) = 1 / 3 * (a n + a (n + 2))) :
  a 6 = 364 :=
by
  -- Proof skipped
  sorry

end sequence_sixth_term_is_364_l1623_162356


namespace solution_1_solution_2_l1623_162390

noncomputable def problem_1 : Real :=
  Real.log 25 + Real.log 2 * Real.log 50 + (Real.log 2)^2

noncomputable def problem_2 : Real :=
  (Real.logb 3 2 + Real.logb 9 2) * (Real.logb 4 3 + Real.logb 8 3)

theorem solution_1 : problem_1 = 2 := by
  sorry

theorem solution_2 : problem_2 = 5 / 4 := by
  sorry

end solution_1_solution_2_l1623_162390


namespace intersection_of_sets_l1623_162351

variable (M : Set ℤ) (N : Set ℤ)

theorem intersection_of_sets :
  M = {-2, -1, 0, 1, 2} →
  N = {x | x ≥ 3 ∨ x ≤ -2} →
  M ∩ N = {-2} :=
by
  intros hM hN
  sorry

end intersection_of_sets_l1623_162351


namespace solution_of_system_of_equations_l1623_162309

-- Define the conditions of the problem.
def system_of_equations (x y : ℝ) : Prop :=
  (x + y = 6) ∧ (x = 2 * y)

-- Define the correct answer as a set.
def solution_set : Set (ℝ × ℝ) :=
  { (4, 2) }

-- State the proof problem.
theorem solution_of_system_of_equations : 
  {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ system_of_equations x y} = solution_set :=
  sorry

end solution_of_system_of_equations_l1623_162309


namespace schlaf_flachs_divisible_by_271_l1623_162301

theorem schlaf_flachs_divisible_by_271 
(S C F H L A : ℕ) 
(hS : S ≠ 0) 
(hF : F ≠ 0) 
(hS_digit : S < 10)
(hC_digit : C < 10)
(hF_digit : F < 10)
(hH_digit : H < 10)
(hL_digit : L < 10)
(hA_digit : A < 10) :
  (100000 * S + 10000 * C + 1000 * H + 100 * L + 10 * A + F - 
   (100000 * F + 10000 * L + 1000 * A + 100 * C + 10 * H + S)) % 271 = 0 ↔ 
  C = L ∧ H = A := 
sorry

end schlaf_flachs_divisible_by_271_l1623_162301


namespace z_is_greater_by_50_percent_of_w_l1623_162323

variable (w q y z : ℝ)

def w_is_60_percent_q : Prop := w = 0.60 * q
def q_is_60_percent_y : Prop := q = 0.60 * y
def z_is_54_percent_y : Prop := z = 0.54 * y

theorem z_is_greater_by_50_percent_of_w (h1 : w_is_60_percent_q w q) 
                                        (h2 : q_is_60_percent_y q y) 
                                        (h3 : z_is_54_percent_y z y) : 
  ((z - w) / w) * 100 = 50 :=
sorry

end z_is_greater_by_50_percent_of_w_l1623_162323


namespace triangle_perimeter_ratio_l1623_162347

theorem triangle_perimeter_ratio : 
  let side := 10
  let hypotenuse := Real.sqrt (side^2 + (side / 2) ^ 2)
  let triangle_perimeter := side + (side / 2) + hypotenuse
  let square_perimeter := 4 * side
  (triangle_perimeter / square_perimeter) = (15 + Real.sqrt 125) / 40 := 
by
  sorry

end triangle_perimeter_ratio_l1623_162347


namespace small_trucks_needed_l1623_162385

-- Defining the problem's conditions
def total_flour : ℝ := 500
def large_truck_capacity : ℝ := 9.6
def num_large_trucks : ℝ := 40
def small_truck_capacity : ℝ := 4

-- Theorem statement to find the number of small trucks needed
theorem small_trucks_needed : (total_flour - (num_large_trucks * large_truck_capacity)) / small_truck_capacity = (500 - (40 * 9.6)) / 4 :=
by
  sorry

end small_trucks_needed_l1623_162385


namespace area_R3_l1623_162374

-- Define the initial dimensions of rectangle R1
def length_R1 := 8
def width_R1 := 4

-- Define the dimensions of rectangle R2 after bisecting R1
def length_R2 := length_R1 / 2
def width_R2 := width_R1

-- Define the dimensions of rectangle R3 after bisecting R2
def length_R3 := length_R2 / 2
def width_R3 := width_R2

-- Prove that the area of R3 is 8
theorem area_R3 : (length_R3 * width_R3) = 8 := by
  -- Calculation for the theorem
  sorry

end area_R3_l1623_162374


namespace volume_ratio_of_cubes_l1623_162332

theorem volume_ratio_of_cubes (s2 : ℝ) : 
  let s1 := s2 * (Real.sqrt 3)
  let V1 := s1^3
  let V2 := s2^3
  V1 / V2 = 3 * (Real.sqrt 3) :=
by
  admit -- si



end volume_ratio_of_cubes_l1623_162332


namespace points_on_curve_l1623_162305

theorem points_on_curve (x y : ℝ) :
  (∃ p : ℝ, y = p^2 + (2 * p - 1) * x + 2 * x^2) ↔ y ≥ x^2 - x :=
by
  sorry

end points_on_curve_l1623_162305


namespace bricks_required_l1623_162327

theorem bricks_required (courtyard_length_m : ℕ) (courtyard_width_m : ℕ)
  (brick_length_cm : ℕ) (brick_width_cm : ℕ)
  (h1 : courtyard_length_m = 30) (h2 : courtyard_width_m = 16)
  (h3 : brick_length_cm = 20) (h4 : brick_width_cm = 10) :
  (3000 * 1600) / (20 * 10) = 24000 :=
by sorry

end bricks_required_l1623_162327


namespace bus_costs_unique_min_buses_cost_A_l1623_162306

-- Defining the main conditions
def condition1 (x y : ℕ) : Prop := x + 2 * y = 300
def condition2 (x y : ℕ) : Prop := 2 * x + y = 270

-- Part 1: Proving individual bus costs
theorem bus_costs_unique (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) :
  x = 80 ∧ y = 110 := 
by 
  sorry

-- Part 2: Minimum buses of type A and total cost constraint
def total_buses := 10
def total_cost (x y a : ℕ) : Prop := 
  x * a + y * (total_buses - a) ≤ 1000

theorem min_buses_cost_A (x y : ℕ) (hx : x = 80) (hy : y = 110) :
  ∃ a cost, total_cost x y a ∧ a >= 4 ∧ cost = x * 4 + y * (total_buses - 4) ∧ cost = 980 :=
by
  sorry

end bus_costs_unique_min_buses_cost_A_l1623_162306


namespace rectangular_prism_volume_l1623_162342

theorem rectangular_prism_volume (a b c : ℝ) (h1 : a * b = 15) (h2 : b * c = 10) (h3 : a * c = 6) (h4 : c^2 = a^2 + b^2) : 
  a * b * c = 30 := 
sorry

end rectangular_prism_volume_l1623_162342


namespace arithmetic_sqrt_9_l1623_162340

theorem arithmetic_sqrt_9 :
  (∃ y : ℝ, y * y = 9 ∧ y ≥ 0) → (∃ y : ℝ, y = 3) :=
by
  sorry

end arithmetic_sqrt_9_l1623_162340


namespace leak_drain_time_l1623_162360

theorem leak_drain_time (P L : ℝ) (h1 : P = 0.5) (h2 : (P - L) = (6 / 13)) :
    (1 / L) = 26 := by
  sorry

end leak_drain_time_l1623_162360


namespace frustum_lateral_surface_area_l1623_162336

/-- A frustum of a right circular cone has the following properties:
  * Lower base radius r1 = 8 inches
  * Upper base radius r2 = 2 inches
  * Height h = 6 inches
  The lateral surface area of such a frustum is 60 * √2 * π square inches.
-/
theorem frustum_lateral_surface_area : 
  let r1 := 8 
  let r2 := 2 
  let h := 6 
  let s := Real.sqrt (h^2 + (r1 - r2)^2)
  A = π * (r1 + r2) * s :=
  sorry

end frustum_lateral_surface_area_l1623_162336


namespace x3_y3_sum_sq_sq_leq_4a10_equality_holds_when_x_eq_y_l1623_162338

-- Conditions: x, y are positive real numbers and x + y = 2a
variables {x y a : ℝ}
variable (hxy : x + y = 2 * a)
variable (hx_pos : 0 < x)
variable (hy_pos : 0 < y)

-- Math proof problem: Prove the inequality
theorem x3_y3_sum_sq_sq_leq_4a10 : 
  x^3 * y^3 * (x^2 + y^2)^2 ≤ 4 * a^10 :=
by sorry

-- Equality condition: Equality holds when x = y
theorem equality_holds_when_x_eq_y (h : x = y) :
  x^3 * y^3 * (x^2 + y^2)^2 = 4 * a^10 :=
by sorry

end x3_y3_sum_sq_sq_leq_4a10_equality_holds_when_x_eq_y_l1623_162338


namespace natural_number_base_conversion_l1623_162316

theorem natural_number_base_conversion (n : ℕ) (h7 : n = 4 * 7 + 1) (h9 : n = 3 * 9 + 2) : 
  n = 3 * 8 + 5 := 
by 
  sorry

end natural_number_base_conversion_l1623_162316


namespace theta_value_l1623_162320

theorem theta_value (Theta : ℕ) (h_digit : Θ < 10) (h_eq : 252 / Θ = 30 + 2 * Θ) : Θ = 6 := 
by
  sorry

end theta_value_l1623_162320


namespace stocks_higher_price_l1623_162308

theorem stocks_higher_price
  (total_stocks : ℕ)
  (percent_increase : ℝ)
  (H L : ℝ)
  (H_eq : H = 1.35 * L)
  (sum_eq : H + L = 4200)
  (percent_increase_eq : percent_increase = 0.35)
  (total_stocks_eq : ↑total_stocks = 4200) :
  total_stocks = 2412 :=
by 
  sorry

end stocks_higher_price_l1623_162308


namespace spider_eyes_solution_l1623_162302

def spider_eyes_problem: Prop :=
  ∃ (x : ℕ), (3 * x) + (50 * 2) = 124 ∧ x = 8

theorem spider_eyes_solution : spider_eyes_problem :=
  sorry

end spider_eyes_solution_l1623_162302


namespace race_distance_l1623_162334

theorem race_distance (d v_A v_B v_C : ℝ) (h1 : d / v_A = (d - 20) / v_B)
  (h2 : d / v_B = (d - 10) / v_C) (h3 : d / v_A = (d - 28) / v_C) : d = 100 :=
by
  sorry

end race_distance_l1623_162334


namespace clothing_discount_l1623_162376

theorem clothing_discount (P : ℝ) :
  let first_sale_price := (4 / 5) * P
  let second_sale_price := first_sale_price * 0.60
  second_sale_price = (12 / 25) * P :=
by
  sorry

end clothing_discount_l1623_162376


namespace segment_length_segment_fraction_three_segments_fraction_l1623_162397

noncomputable def total_length : ℝ := 4
noncomputable def number_of_segments : ℕ := 5

theorem segment_length (L : ℝ) (n : ℕ) (hL : L = total_length) (hn : n = number_of_segments) :
  L / n = (4 / 5 : ℝ) := by
sorry

theorem segment_fraction (n : ℕ) (hn : n = number_of_segments) :
  (1 / n : ℝ) = (1 / 5 : ℝ) := by
sorry

theorem three_segments_fraction (n : ℕ) (hn : n = number_of_segments) :
  (3 / n : ℝ) = (3 / 5 : ℝ) := by
sorry

end segment_length_segment_fraction_three_segments_fraction_l1623_162397


namespace exponent_equivalence_l1623_162346

open Real

theorem exponent_equivalence (a : ℝ) (h : a > 0) : 
  (a^2 / (sqrt a * a^(2/3))) = a^(5/6) :=
  sorry

end exponent_equivalence_l1623_162346


namespace quadratic_not_factored_l1623_162358

theorem quadratic_not_factored
  (a b c : ℕ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (p : ℕ)
  (h_prime_p : Nat.Prime p)
  (h_p : a * 1991^2 + b * 1991 + c = p) :
  ¬ (∃ d₁ d₂ e₁ e₂ : ℤ, a = d₁ * d₂ ∧ b = d₁ * e₂ + d₂ * e₁ ∧ c = e₁ * e₂) :=
sorry

end quadratic_not_factored_l1623_162358
