import Mathlib

namespace NUMINAMATH_GPT_option_b_represents_factoring_l1722_172247

theorem option_b_represents_factoring (x y : ℤ) :
  x^2 - 2*x*y = x * (x - 2*y) :=
sorry

end NUMINAMATH_GPT_option_b_represents_factoring_l1722_172247


namespace NUMINAMATH_GPT_base6_addition_correct_l1722_172225

theorem base6_addition_correct (S H E : ℕ) (h1 : S < 6) (h2 : H < 6) (h3 : E < 6) 
  (distinct : S ≠ H ∧ H ≠ E ∧ S ≠ E) 
  (h4: S + H * 6 + E * 6^2 +  H * 6 = H + E * 6 + H * 6^2 + E * 6^1) :
  S + H + E = 12 :=
by sorry

end NUMINAMATH_GPT_base6_addition_correct_l1722_172225


namespace NUMINAMATH_GPT_original_percentage_of_acid_l1722_172286

theorem original_percentage_of_acid 
  (a w : ℝ) 
  (h1 : a + w = 6) 
  (h2 : a / (a + w + 2) = 15 / 100) 
  (h3 : (a + 2) / (a + w + 4) = 25 / 100) :
  (a / 6) * 100 = 20 :=
  sorry

end NUMINAMATH_GPT_original_percentage_of_acid_l1722_172286


namespace NUMINAMATH_GPT_opposite_number_l1722_172253

theorem opposite_number (x : ℤ) (h : -x = -2) : x = 2 :=
sorry

end NUMINAMATH_GPT_opposite_number_l1722_172253


namespace NUMINAMATH_GPT_find_initial_population_l1722_172275

-- Define the initial population, conditions and the final population
variable (P : ℕ)

noncomputable def initial_population (P : ℕ) :=
  (0.85 * (0.92 * P) : ℝ) = 3553

theorem find_initial_population (P : ℕ) (h : initial_population P) : P = 4546 := sorry

end NUMINAMATH_GPT_find_initial_population_l1722_172275


namespace NUMINAMATH_GPT_john_got_rolls_l1722_172289

def cost_per_dozen : ℕ := 5
def money_spent : ℕ := 15
def rolls_per_dozen : ℕ := 12

theorem john_got_rolls : (money_spent / cost_per_dozen) * rolls_per_dozen = 36 :=
by sorry

end NUMINAMATH_GPT_john_got_rolls_l1722_172289


namespace NUMINAMATH_GPT_triangle_area_half_l1722_172270

theorem triangle_area_half (AB AC BC : ℝ) (h₁ : AB = 8) (h₂ : AC = BC) (h₃ : AC * AC = AB * AB / 2) (h₄ : AC = BC) : 
  (1 / 2) * (1 / 2 * AB * AB) = 16 :=
  by
  sorry

end NUMINAMATH_GPT_triangle_area_half_l1722_172270


namespace NUMINAMATH_GPT_collectively_behind_l1722_172213

noncomputable def sleep_hours_behind (weeknights weekend nights_ideal: ℕ) : ℕ :=
  let total_sleep := (weeknights * 5) + (weekend * 2)
  let ideal_sleep := nights_ideal * 7
  ideal_sleep - total_sleep

def tom_weeknight := 5
def tom_weekend := 6

def jane_weeknight := 7
def jane_weekend := 9

def mark_weeknight := 6
def mark_weekend := 7

def ideal_night := 8

theorem collectively_behind :
  sleep_hours_behind tom_weeknight tom_weekend ideal_night +
  sleep_hours_behind jane_weeknight jane_weekend ideal_night +
  sleep_hours_behind mark_weeknight mark_weekend ideal_night = 34 :=
by
  sorry

end NUMINAMATH_GPT_collectively_behind_l1722_172213


namespace NUMINAMATH_GPT_rhombus_height_l1722_172262

theorem rhombus_height (a d1 d2 : ℝ) (h : ℝ)
  (h_a_positive : 0 < a)
  (h_d1_positive : 0 < d1)
  (h_d2_positive : 0 < d2)
  (h_side_geometric_mean : a^2 = d1 * d2) :
  h = a / 2 :=
sorry

end NUMINAMATH_GPT_rhombus_height_l1722_172262


namespace NUMINAMATH_GPT_frank_money_l1722_172266

-- Define the initial amount, expenses, and incomes as per the conditions
def initialAmount : ℕ := 11
def spentOnGame : ℕ := 3
def spentOnKeychain : ℕ := 2
def receivedFromAlice : ℕ := 4
def allowance : ℕ := 14
def spentOnBusTicket : ℕ := 5

-- Define the total money left for Frank
def finalAmount (initial : ℕ) (game : ℕ) (keychain : ℕ) (gift : ℕ) (allowance : ℕ) (bus : ℕ) : ℕ :=
  initial - game - keychain + gift + allowance - bus

-- Define the theorem stating that the final amount is 19
theorem frank_money : finalAmount initialAmount spentOnGame spentOnKeychain receivedFromAlice allowance spentOnBusTicket = 19 :=
by
  sorry

end NUMINAMATH_GPT_frank_money_l1722_172266


namespace NUMINAMATH_GPT_union_of_A_and_B_l1722_172226

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} := 
by 
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1722_172226


namespace NUMINAMATH_GPT_forty_percent_jacqueline_candy_l1722_172223

def fred_candy : ℕ := 12
def uncle_bob_candy : ℕ := fred_candy + 6
def total_fred_uncle_bob_candy : ℕ := fred_candy + uncle_bob_candy
def jacqueline_candy : ℕ := 10 * total_fred_uncle_bob_candy

theorem forty_percent_jacqueline_candy : (40 * jacqueline_candy) / 100 = 120 := by
  sorry

end NUMINAMATH_GPT_forty_percent_jacqueline_candy_l1722_172223


namespace NUMINAMATH_GPT_limit_perimeters_eq_l1722_172250

universe u

noncomputable def limit_perimeters (s : ℝ) : ℝ :=
  let a := 4 * s
  let r := 1 / 2
  a / (1 - r)

theorem limit_perimeters_eq (s : ℝ) : limit_perimeters s = 8 * s := by
  sorry

end NUMINAMATH_GPT_limit_perimeters_eq_l1722_172250


namespace NUMINAMATH_GPT_solve_system_l1722_172237

theorem solve_system (x y : ℝ) (h1 : 5 * x + y = 19) (h2 : x + 3 * y = 1) : 3 * x + 2 * y = 10 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l1722_172237


namespace NUMINAMATH_GPT_support_percentage_correct_l1722_172293

-- Define the total number of government employees and the percentage supporting the project
def num_gov_employees : ℕ := 150
def perc_gov_support : ℝ := 0.70

-- Define the total number of citizens and the percentage supporting the project
def num_citizens : ℕ := 800
def perc_citizens_support : ℝ := 0.60

-- Calculate the number of supporters among government employees
def gov_supporters : ℝ := perc_gov_support * num_gov_employees

-- Calculate the number of supporters among citizens
def citizens_supporters : ℝ := perc_citizens_support * num_citizens

-- Calculate the total number of people surveyed and the total number of supporters
def total_surveyed : ℝ := num_gov_employees + num_citizens
def total_supporters : ℝ := gov_supporters + citizens_supporters

-- Define the expected correct answer percentage
def correct_percentage_supporters : ℝ := 61.58

-- Prove that the percentage of overall supporters is equal to the expected correct percentage 
theorem support_percentage_correct :
  (total_supporters / total_surveyed * 100) = correct_percentage_supporters :=
by
  sorry

end NUMINAMATH_GPT_support_percentage_correct_l1722_172293


namespace NUMINAMATH_GPT_base8_subtraction_l1722_172227

def subtract_base_8 (a b : Nat) : Nat :=
  sorry  -- This is a placeholder for the actual implementation.

theorem base8_subtraction :
  subtract_base_8 0o5374 0o2645 = 0o1527 :=
by
  sorry

end NUMINAMATH_GPT_base8_subtraction_l1722_172227


namespace NUMINAMATH_GPT_relationship_between_x_y_z_l1722_172277

theorem relationship_between_x_y_z (x y z : ℕ) (a b c d : ℝ)
  (h1 : x ≤ y ∧ y ≤ z)
  (h2 : (x:ℝ)^a = 70^d ∧ (y:ℝ)^b = 70^d ∧ (z:ℝ)^c = 70^d)
  (h3 : 1/a + 1/b + 1/c = 1/d) :
  x + y = z := 
sorry

end NUMINAMATH_GPT_relationship_between_x_y_z_l1722_172277


namespace NUMINAMATH_GPT_f_2012_l1722_172248

noncomputable def f : ℝ → ℝ := sorry -- provided as a 'sorry' to be determined

axiom odd_function (hf : ℝ → ℝ) : ∀ x : ℝ, hf (-x) = -hf (x)

axiom f_shift : ∀ x : ℝ, f (x + 3) = -f (x)
axiom f_one : f 1 = 2

theorem f_2012 : f 2012 = 2 :=
by
  -- proofs would go here, but 'sorry' is enough to define the theorem statement
  sorry

end NUMINAMATH_GPT_f_2012_l1722_172248


namespace NUMINAMATH_GPT_max_min_values_l1722_172254

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 2

theorem max_min_values :
  let max_val := 2
  let min_val := -25
  ∃ x_max x_min, 
    0 ≤ x_max ∧ x_max ≤ 4 ∧ f x_max = max_val ∧ 
    0 ≤ x_min ∧ x_min ≤ 4 ∧ f x_min = min_val :=
sorry

end NUMINAMATH_GPT_max_min_values_l1722_172254


namespace NUMINAMATH_GPT_simplify_expression_l1722_172232

theorem simplify_expression (y : ℝ) : 3 * y + 5 * y + 6 * y + 10 = 14 * y + 10 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1722_172232


namespace NUMINAMATH_GPT_intersection_necessary_but_not_sufficient_l1722_172267

variables {M N P : Set α}

theorem intersection_necessary_but_not_sufficient : 
  (M ∩ P = N ∩ P) → (M ≠ N) :=
sorry

end NUMINAMATH_GPT_intersection_necessary_but_not_sufficient_l1722_172267


namespace NUMINAMATH_GPT_largest_angle_consecutive_even_pentagon_l1722_172200

theorem largest_angle_consecutive_even_pentagon :
  ∀ (n : ℕ), (2 * n + (2 * n + 2) + (2 * n + 4) + (2 * n + 6) + (2 * n + 8) = 540) →
  (2 * n + 8 = 112) :=
by
  intros n h
  sorry

end NUMINAMATH_GPT_largest_angle_consecutive_even_pentagon_l1722_172200


namespace NUMINAMATH_GPT_apple_tree_total_production_l1722_172202

-- Definitions for conditions
def first_year_production : ℕ := 40
def second_year_production : ℕ := 2 * first_year_production + 8
def third_year_production : ℕ := second_year_production - second_year_production / 4

-- Theorem statement
theorem apple_tree_total_production :
  first_year_production + second_year_production + third_year_production = 194 :=
by
  sorry

end NUMINAMATH_GPT_apple_tree_total_production_l1722_172202


namespace NUMINAMATH_GPT_simplify_expression_l1722_172295

theorem simplify_expression : 
  2 ^ (-1: ℤ) + Real.sqrt 16 - (3 - Real.sqrt 3) ^ 0 + |Real.sqrt 2 - 1 / 2| = 3 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1722_172295


namespace NUMINAMATH_GPT_garden_length_l1722_172249

open Nat

def perimeter : ℕ → ℕ → ℕ := λ l w => 2 * (l + w)

theorem garden_length (width : ℕ) (perimeter_val : ℕ) (length : ℕ) 
  (h1 : width = 15) 
  (h2 : perimeter_val = 80) 
  (h3 : perimeter length width = perimeter_val) :
  length = 25 := by
  sorry

end NUMINAMATH_GPT_garden_length_l1722_172249


namespace NUMINAMATH_GPT_greatest_drop_june_increase_april_l1722_172265

-- January price change
def jan : ℝ := -1.00

-- February price change
def feb : ℝ := 3.50

-- March price change
def mar : ℝ := -3.00

-- April price change
def apr : ℝ := 4.50

-- May price change
def may : ℝ := -1.50

-- June price change
def jun : ℝ := -3.50

def greatest_drop : List (ℝ × String) := [(jan, "January"), (mar, "March"), (may, "May"), (jun, "June")]

def greatest_increase : List (ℝ × String) := [(feb, "February"), (apr, "April")]

theorem greatest_drop_june_increase_april :
  (∀ d ∈ greatest_drop, d.1 ≤ jun) ∧ (∀ i ∈ greatest_increase, i.1 ≤ apr) :=
by
  sorry

end NUMINAMATH_GPT_greatest_drop_june_increase_april_l1722_172265


namespace NUMINAMATH_GPT_cost_of_fencing_each_side_l1722_172278

theorem cost_of_fencing_each_side (total_cost : ℕ) (x : ℕ) (h : total_cost = 276) (hx : 4 * x = total_cost) : x = 69 :=
by {
  sorry
}

end NUMINAMATH_GPT_cost_of_fencing_each_side_l1722_172278


namespace NUMINAMATH_GPT_johns_pool_depth_l1722_172224

theorem johns_pool_depth : 
  ∀ (j s : ℕ), (j = 2 * s + 5) → (s = 5) → (j = 15) := 
by 
  intros j s h1 h2
  rw [h2] at h1
  exact h1

end NUMINAMATH_GPT_johns_pool_depth_l1722_172224


namespace NUMINAMATH_GPT_neither_jia_nor_yi_has_winning_strategy_l1722_172214

/-- 
  There are 99 points, each marked with a number from 1 to 99, placed 
  on 99 equally spaced points on a circle. Jia and Yi take turns 
  placing one piece at a time, with Jia going first. The player who 
  first makes the numbers on three consecutive points form an 
  arithmetic sequence wins. Prove that neither Jia nor Yi has a 
  guaranteed winning strategy, and both possess strategies to avoid 
  losing.
-/
theorem neither_jia_nor_yi_has_winning_strategy :
  ∀ (points : Fin 99 → ℕ), -- 99 points on the circle
  (∀ i, 1 ≤ points i ∧ points i ≤ 99) → -- Each point is numbered between 1 and 99
  ¬(∃ (player : Fin 99 → ℕ) (h : ∀ (i : Fin 99), player i ≠ 0 ∧ (player i = 1 ∨ player i = 2)),
    ∃ i : Fin 99, (points i + points (i + 1) + points (i + 2)) / 3 = points i)
:=
by
  sorry

end NUMINAMATH_GPT_neither_jia_nor_yi_has_winning_strategy_l1722_172214


namespace NUMINAMATH_GPT_speed_conversion_l1722_172240

def speed_mps : ℝ := 10.0008
def conversion_factor : ℝ := 3.6

theorem speed_conversion : speed_mps * conversion_factor = 36.003 :=
by
  sorry

end NUMINAMATH_GPT_speed_conversion_l1722_172240


namespace NUMINAMATH_GPT_value_of_m_l1722_172288

noncomputable def has_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

noncomputable def has_no_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c < 0

theorem value_of_m (m : ℝ) :
  (has_distinct_real_roots 1 m 1 ∧ has_no_real_roots 4 (4 * (m + 2)) 1) ↔ (-3 < m ∧ m < -2) :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l1722_172288


namespace NUMINAMATH_GPT_geometric_arithmetic_sequence_relation_l1722_172251

theorem geometric_arithmetic_sequence_relation 
    (a : ℕ → ℝ) (b : ℕ → ℝ) (q d a1 : ℝ)
    (h1 : a 1 = a1) (h2 : b 1 = a1) (h3 : a 3 = a1 * q^2)
    (h4 : b 3 = a1 + 2 * d) (h5 : a 3 = b 3) (h6 : a1 > 0) (h7 : q^2 ≠ 1) :
    a 5 > b 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_geometric_arithmetic_sequence_relation_l1722_172251


namespace NUMINAMATH_GPT_marathon_distance_l1722_172236

theorem marathon_distance (marathons : ℕ) (miles_per_marathon : ℕ) (extra_yards_per_marathon : ℕ) (yards_per_mile : ℕ) (total_miles_run : ℕ) (total_yards_run : ℕ) (remaining_yards : ℕ) :
  marathons = 15 →
  miles_per_marathon = 26 →
  extra_yards_per_marathon = 385 →
  yards_per_mile = 1760 →
  total_miles_run = (marathons * miles_per_marathon + extra_yards_per_marathon * marathons / yards_per_mile) →
  total_yards_run = (marathons * (miles_per_marathon * yards_per_mile + extra_yards_per_marathon)) →
  remaining_yards = total_yards_run - (total_miles_run * yards_per_mile) →
  0 ≤ remaining_yards ∧ remaining_yards < yards_per_mile →
  remaining_yards = 1500 :=
by
  intros
  sorry

end NUMINAMATH_GPT_marathon_distance_l1722_172236


namespace NUMINAMATH_GPT_butterflies_left_l1722_172271

theorem butterflies_left (initial_butterflies : ℕ) (one_third_left : ℕ)
  (h1 : initial_butterflies = 9) (h2 : one_third_left = initial_butterflies / 3) :
  initial_butterflies - one_third_left = 6 :=
by
  sorry

end NUMINAMATH_GPT_butterflies_left_l1722_172271


namespace NUMINAMATH_GPT_mike_drive_average_rate_l1722_172297

open Real

variables (total_distance first_half_distance second_half_distance first_half_speed second_half_speed first_half_time second_half_time total_time avg_rate j : ℝ)

theorem mike_drive_average_rate :
  total_distance = 640 ∧
  first_half_distance = total_distance / 2 ∧
  second_half_distance = total_distance / 2 ∧
  first_half_speed = 80 ∧
  first_half_distance / first_half_speed = first_half_time ∧
  second_half_time = 3 * first_half_time ∧
  second_half_distance / second_half_time = second_half_speed ∧
  total_time = first_half_time + second_half_time ∧
  avg_rate = total_distance / total_time →
  j = 40 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_mike_drive_average_rate_l1722_172297


namespace NUMINAMATH_GPT_range_of_m_l1722_172287

theorem range_of_m {m : ℝ} (h1 : m^2 - 1 < 0) (h2 : m > 0) : 0 < m ∧ m < 1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1722_172287


namespace NUMINAMATH_GPT_number_of_trees_l1722_172242

-- Define the yard length and the distance between consecutive trees
def yard_length : ℕ := 300
def distance_between_trees : ℕ := 12

-- Prove that the number of trees planted in the garden is 26
theorem number_of_trees (yard_length distance_between_trees : ℕ) 
  (h1 : yard_length = 300) (h2 : distance_between_trees = 12) : 
  ∃ n : ℕ, n = 26 :=
by
  sorry

end NUMINAMATH_GPT_number_of_trees_l1722_172242


namespace NUMINAMATH_GPT_range_of_a_l1722_172218

theorem range_of_a (a : ℝ) : (2 * a - 8) / 3 < 0 → a < 4 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1722_172218


namespace NUMINAMATH_GPT_sufficient_food_supply_l1722_172243

variable {L S : ℝ}

theorem sufficient_food_supply (h1 : L + 4 * S = 14) (h2 : L > S) : L + 3 * S ≥ 11 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_food_supply_l1722_172243


namespace NUMINAMATH_GPT_remainder_of_1999_pow_11_mod_8_l1722_172257

theorem remainder_of_1999_pow_11_mod_8 :
  (1999 ^ 11) % 8 = 7 :=
  sorry

end NUMINAMATH_GPT_remainder_of_1999_pow_11_mod_8_l1722_172257


namespace NUMINAMATH_GPT_num_valid_seating_arrangements_l1722_172220

-- Define the dimensions of the examination room
def rows : Nat := 5
def columns : Nat := 6
def total_seats : Nat := rows * columns

-- Define the condition for students not sitting next to each other
def valid_seating_arrangements (rows columns : Nat) : Nat := sorry

-- The theorem to prove the number of seating arrangements
theorem num_valid_seating_arrangements : valid_seating_arrangements rows columns = 772 := 
by 
  sorry

end NUMINAMATH_GPT_num_valid_seating_arrangements_l1722_172220


namespace NUMINAMATH_GPT_exponentiation_identity_l1722_172282

theorem exponentiation_identity :
  (5^4)^2 = 390625 :=
  by sorry

end NUMINAMATH_GPT_exponentiation_identity_l1722_172282


namespace NUMINAMATH_GPT_add_numerator_denominator_add_numerator_denominator_gt_one_l1722_172274

variable {a b n : ℕ}

/-- Adding the same natural number to both the numerator and the denominator of a fraction 
    increases the fraction if it is less than one, and decreases the fraction if it is greater than one. -/
theorem add_numerator_denominator (h1: a < b) : (a + n) / (b + n) > a / b := sorry

theorem add_numerator_denominator_gt_one (h2: a > b) : (a + n) / (b + n) < a / b := sorry

end NUMINAMATH_GPT_add_numerator_denominator_add_numerator_denominator_gt_one_l1722_172274


namespace NUMINAMATH_GPT_people_and_cars_equation_l1722_172238

theorem people_and_cars_equation (x : ℕ) :
  3 * (x - 2) = 2 * x + 9 :=
sorry

end NUMINAMATH_GPT_people_and_cars_equation_l1722_172238


namespace NUMINAMATH_GPT_geom_seq_sum_first_eight_l1722_172252

def geom_seq (a₀ r : ℚ) (n : ℕ) : ℚ := a₀ * r^n

def sum_geom_seq (a₀ r : ℚ) (n : ℕ) : ℚ :=
  if r = 1 then a₀ * n else a₀ * (1 - r^n) / (1 - r)

theorem geom_seq_sum_first_eight :
  let a₀ := 1 / 3
  let r := 1 / 3
  let n := 8
  sum_geom_seq a₀ r n = 3280 / 6561 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_sum_first_eight_l1722_172252


namespace NUMINAMATH_GPT_pool_capacity_l1722_172273

theorem pool_capacity (C : ℝ) (h1 : 300 = 0.30 * C) : C = 1000 :=
by
  sorry

end NUMINAMATH_GPT_pool_capacity_l1722_172273


namespace NUMINAMATH_GPT_next_month_has_5_Wednesdays_l1722_172231

-- The current month characteristics
def current_month_has_5_Saturdays : Prop := ∃ month : ℕ, month = 30 ∧ ∃ day : ℕ, day = 5
def current_month_has_5_Sundays : Prop := ∃ month : ℕ, month = 30 ∧ ∃ day : ℕ, day = 5
def current_month_has_4_Mondays : Prop := ∃ month : ℕ, month = 30 ∧ ∃ day : ℕ, day = 4
def current_month_has_4_Fridays : Prop := ∃ month : ℕ, month = 30 ∧ ∃ day : ℕ, day = 4
def month_ends_on_Sunday : Prop := ∃ day : ℕ, day = 30 ∧ day % 7 = 0

-- Prove next month has 5 Wednesdays
theorem next_month_has_5_Wednesdays 
  (h1 : current_month_has_5_Saturdays) 
  (h2 : current_month_has_5_Sundays)
  (h3 : current_month_has_4_Mondays)
  (h4 : current_month_has_4_Fridays)
  (h5 : month_ends_on_Sunday) :
  ∃ month : ℕ, month = 31 ∧ ∃ day : ℕ, day = 5 := 
sorry

end NUMINAMATH_GPT_next_month_has_5_Wednesdays_l1722_172231


namespace NUMINAMATH_GPT_ordering_of_powers_l1722_172217

theorem ordering_of_powers : (3 ^ 17) < (8 ^ 9) ∧ (8 ^ 9) < (4 ^ 15) := 
by 
  -- We proved (3 ^ 17) < (8 ^ 9)
  have h1 : (3 ^ 17) < (8 ^ 9) := sorry
  
  -- We proved (8 ^ 9) < (4 ^ 15)
  have h2 : (8 ^ 9) < (4 ^ 15) := sorry

  -- Therefore, combining both
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_ordering_of_powers_l1722_172217


namespace NUMINAMATH_GPT_divisible_values_l1722_172216

def is_digit (n : ℕ) : Prop :=
  n >= 0 ∧ n <= 9

def N (x y : ℕ) : ℕ :=
  30 * 10^7 + x * 10^6 + 7 * 10^4 + y * 10^3 + 3

def is_divisible_by_37 (n : ℕ) : Prop :=
  n % 37 = 0

theorem divisible_values :
  ∃ (x y : ℕ), is_digit x ∧ is_digit y ∧ is_divisible_by_37 (N x y) ∧ ((x, y) = (8, 1) ∨ (x, y) = (4, 4) ∨ (x, y) = (0, 7)) :=
by {
  sorry
}

end NUMINAMATH_GPT_divisible_values_l1722_172216


namespace NUMINAMATH_GPT_ratio_of_ages_three_years_ago_l1722_172284

theorem ratio_of_ages_three_years_ago (k Y_c : ℕ) (h1 : 45 - 3 = k * (Y_c - 3)) (h2 : (45 + 7) + (Y_c + 7) = 83) : (45 - 3) / (Y_c - 3) = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_of_ages_three_years_ago_l1722_172284


namespace NUMINAMATH_GPT_odd_prime_divides_seq_implies_power_of_two_divides_l1722_172201

theorem odd_prime_divides_seq_implies_power_of_two_divides (a : ℕ → ℤ) (p n : ℕ)
  (h0 : a 0 = 2)
  (hk : ∀ k, a (k + 1) = 2 * (a k) ^ 2 - 1)
  (h_odd_prime : Nat.Prime p)
  (h_odd : p % 2 = 1)
  (h_divides : ↑p ∣ a n) :
  2^(n + 3) ∣ p^2 - 1 :=
sorry

end NUMINAMATH_GPT_odd_prime_divides_seq_implies_power_of_two_divides_l1722_172201


namespace NUMINAMATH_GPT_real_part_of_complex_pow_l1722_172260

open Complex

theorem real_part_of_complex_pow (a b : ℝ) : a = 1 → b = -2 → (realPart ((a : ℂ) + (b : ℂ) * Complex.I)^5) = 41 :=
by
  sorry

end NUMINAMATH_GPT_real_part_of_complex_pow_l1722_172260


namespace NUMINAMATH_GPT_arithmetic_expression_l1722_172235

theorem arithmetic_expression : 4 * 6 * 8 + 18 / 3 - 2 ^ 3 = 190 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_arithmetic_expression_l1722_172235


namespace NUMINAMATH_GPT_roots_of_equation_l1722_172272

theorem roots_of_equation (x : ℝ) : 3 * x * (x - 1) = 2 * (x - 1) → (x = 1 ∨ x = 2 / 3) :=
by 
  intros h
  sorry

end NUMINAMATH_GPT_roots_of_equation_l1722_172272


namespace NUMINAMATH_GPT_sequences_with_both_properties_are_constant_l1722_172259

-- Definitions according to the problem's conditions
def arithmetic_sequence (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) - seq n = seq (n + 2) - seq (n + 1)

def geometric_sequence (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) / seq n = seq (n + 2) / seq (n + 1)

-- Definition of the sequence properties combined
def arithmetic_and_geometric_sequence (seq : ℕ → ℝ) : Prop :=
  arithmetic_sequence seq ∧ geometric_sequence seq

-- Problem to prove
theorem sequences_with_both_properties_are_constant (seq : ℕ → ℝ) :
  arithmetic_and_geometric_sequence seq → ∀ n m : ℕ, seq n = seq m :=
sorry

end NUMINAMATH_GPT_sequences_with_both_properties_are_constant_l1722_172259


namespace NUMINAMATH_GPT_find_a_b_l1722_172255

theorem find_a_b :
  ∃ (a b : ℚ), 
    (∀ x : ℚ, x = 2 → (a * x^3 - 6 * x^2 + b * x - 5 - 3 = 0)) ∧
    (∀ x : ℚ, x = -1 → (a * x^3 - 6 * x^2 + b * x - 5 - 7 = 0)) ∧
    (a = -2/3 ∧ b = -52/3) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_b_l1722_172255


namespace NUMINAMATH_GPT_calculate_L_l1722_172209

theorem calculate_L (T H K : ℝ) (hT : T = 2 * Real.sqrt 5) (hH : H = 10) (hK : K = 2) :
  L = 100 :=
by
  let L := 50 * T^4 / (H^2 * K)
  have : T = 2 * Real.sqrt 5 := hT
  have : H = 10 := hH
  have : K = 2 := hK
  sorry

end NUMINAMATH_GPT_calculate_L_l1722_172209


namespace NUMINAMATH_GPT_perpendicular_condition_l1722_172246

theorem perpendicular_condition (a : ℝ) :
  let l1 (x y : ℝ) := x + a * y - 2
  let l2 (x y : ℝ) := x - a * y - 1
  (∀ x y, (l1 x y = 0 ↔ l2 x y ≠ 0) ↔ 1 - a * a = 0) →
  (a = -1) ∨ (a = 1) :=
by
  intro
  sorry

end NUMINAMATH_GPT_perpendicular_condition_l1722_172246


namespace NUMINAMATH_GPT_find_f6_l1722_172241

noncomputable def f : ℝ → ℝ :=
sorry

theorem find_f6 (h1 : ∀ x y : ℝ, f (x + y) = f x + f y)
                (h2 : f 5 = 6) :
  f 6 = 36 / 5 :=
sorry

end NUMINAMATH_GPT_find_f6_l1722_172241


namespace NUMINAMATH_GPT_amount_spent_on_belt_correct_l1722_172264

variable (budget shirt pants coat socks shoes remaining : ℕ)

-- Given conditions
def initial_budget : ℕ := 200
def spent_shirt : ℕ := 30
def spent_pants : ℕ := 46
def spent_coat : ℕ := 38
def spent_socks : ℕ := 11
def spent_shoes : ℕ := 41
def remaining_amount : ℕ := 16

-- The amount spent on the belt
def amount_spent_on_belt : ℕ :=
  budget - remaining - (shirt + pants + coat + socks + shoes)

-- The theorem statement we need to prove
theorem amount_spent_on_belt_correct :
  initial_budget = budget →
  spent_shirt = shirt →
  spent_pants = pants →
  spent_coat = coat →
  spent_socks = socks →
  spent_shoes = shoes →
  remaining_amount = remaining →
  amount_spent_on_belt budget shirt pants coat socks shoes remaining = 18 := by
    simp [initial_budget, spent_shirt, spent_pants, spent_coat, spent_socks, spent_shoes, remaining_amount, amount_spent_on_belt]
    sorry

end NUMINAMATH_GPT_amount_spent_on_belt_correct_l1722_172264


namespace NUMINAMATH_GPT_star_contains_2011_l1722_172283

theorem star_contains_2011 :
  ∃ (n : ℕ), n = 183 ∧ 
  (∃ (seq : List ℕ), seq = List.range' (2003) 11 ∧ 2011 ∈ seq) :=
by
  sorry

end NUMINAMATH_GPT_star_contains_2011_l1722_172283


namespace NUMINAMATH_GPT_correct_exponential_rule_l1722_172212

theorem correct_exponential_rule (a : ℝ) : (a^3)^2 = a^6 :=
by sorry

end NUMINAMATH_GPT_correct_exponential_rule_l1722_172212


namespace NUMINAMATH_GPT_base9_perfect_square_l1722_172221

theorem base9_perfect_square (a b d : ℕ) (h1 : a ≠ 0) (h2 : a < 9) (h3 : b < 9) (h4 : d < 9) (h5 : ∃ n : ℕ, (729 * a + 81 * b + 36 + d) = n * n) : d = 0 ∨ d = 1 ∨ d = 4 :=
by sorry

end NUMINAMATH_GPT_base9_perfect_square_l1722_172221


namespace NUMINAMATH_GPT_tim_total_spent_l1722_172258

-- Define the given conditions
def lunch_cost : ℝ := 50.20
def tip_percentage : ℝ := 0.20

-- Define the total amount spent
def total_amount_spent : ℝ := 60.24

-- Prove the total amount spent given the conditions
theorem tim_total_spent : lunch_cost + (tip_percentage * lunch_cost) = total_amount_spent := by
  -- This is the proof statement corresponding to the problem; the proof itself is not required for this task
  sorry

end NUMINAMATH_GPT_tim_total_spent_l1722_172258


namespace NUMINAMATH_GPT_find_center_of_circle_l1722_172219

theorem find_center_of_circle :
  ∃ (a b : ℝ), a = 0 ∧ b = 3/2 ∧
  ( ∀ (x y : ℝ), ( (x = 1 ∧ y = 2) ∨ (x = 1 ∧ y = 1) ∨ (∃ t : ℝ, y = 2 * t + 3) ) → 
  (x - a)^2 + (y - b)^2 = (1 - a)^2 + (1 - b)^2 ) :=
sorry

end NUMINAMATH_GPT_find_center_of_circle_l1722_172219


namespace NUMINAMATH_GPT_find_m_n_l1722_172230

theorem find_m_n (m n x1 x2 : ℕ) (hm : 0 < m) (hn : 0 < n) (hx1 : 0 < x1) (hx2 : 0 < x2) 
  (h_eq : x1 * x2 = m + n) (h_sum : x1 + x2 = m * n) :
  (m = 2 ∧ n = 3) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 2) ∨ (m = 1 ∧ n = 5) ∨ (m = 5 ∧ n = 1) := 
sorry

end NUMINAMATH_GPT_find_m_n_l1722_172230


namespace NUMINAMATH_GPT_danielle_travel_time_is_30_l1722_172207

noncomputable def chase_speed : ℝ := sorry
noncomputable def chase_time : ℝ := 180 -- in minutes
noncomputable def cameron_speed : ℝ := 2 * chase_speed
noncomputable def danielle_speed : ℝ := 3 * cameron_speed
noncomputable def distance : ℝ := chase_speed * chase_time
noncomputable def danielle_time : ℝ := distance / danielle_speed

theorem danielle_travel_time_is_30 :
  danielle_time = 30 :=
sorry

end NUMINAMATH_GPT_danielle_travel_time_is_30_l1722_172207


namespace NUMINAMATH_GPT_hyperbola_slopes_l1722_172268

variables {x1 y1 x2 y2 x y k1 k2 : ℝ}

theorem hyperbola_slopes (h1 : y1^2 - (x1^2 / 2) = 1)
  (h2 : y2^2 - (x2^2 / 2) = 1)
  (hx : x1 + x2 = 2 * x)
  (hy : y1 + y2 = 2 * y)
  (hk1 : k1 = (y2 - y1) / (x2 - x1))
  (hk2 : k2 = y / x) :
  k1 * k2 = 1 / 2 :=
sorry

end NUMINAMATH_GPT_hyperbola_slopes_l1722_172268


namespace NUMINAMATH_GPT_petya_payment_l1722_172205

theorem petya_payment (x y : ℤ) (h₁ : 14 * x + 3 * y = 107) (h₂ : |x - y| ≤ 5) : x + y = 10 :=
sorry

end NUMINAMATH_GPT_petya_payment_l1722_172205


namespace NUMINAMATH_GPT_theodore_total_monthly_earning_l1722_172299

def total_earnings (stone_statues: Nat) (wooden_statues: Nat) (cost_stone: Nat) (cost_wood: Nat) (tax_rate: Rat) : Rat :=
  let pre_tax_earnings := stone_statues * cost_stone + wooden_statues * cost_wood
  let tax := tax_rate * pre_tax_earnings
  pre_tax_earnings - tax

theorem theodore_total_monthly_earning : total_earnings 10 20 20 5 0.10 = 270 :=
by
  sorry

end NUMINAMATH_GPT_theodore_total_monthly_earning_l1722_172299


namespace NUMINAMATH_GPT_sum_of_roots_l1722_172211

theorem sum_of_roots (α β : ℝ) (h1 : α^2 - 4 * α + 3 = 0) (h2 : β^2 - 4 * β + 3 = 0) (h3 : α ≠ β) :
  α + β = 4 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l1722_172211


namespace NUMINAMATH_GPT_sum_is_18_l1722_172298

/-- Define the distinct non-zero digits, Hen, Xin, Chun, satisfying the given equation. -/
theorem sum_is_18 (Hen Xin Chun : ℕ) (h1 : Hen ≠ Xin) (h2 : Xin ≠ Chun) (h3 : Hen ≠ Chun)
  (h4 : 1 ≤ Hen ∧ Hen ≤ 9) (h5 : 1 ≤ Xin ∧ Xin ≤ 9) (h6 : 1 ≤ Chun ∧ Chun ≤ 9) :
  Hen + Xin + Chun = 18 :=
sorry

end NUMINAMATH_GPT_sum_is_18_l1722_172298


namespace NUMINAMATH_GPT_xy_fraction_equivalence_l1722_172263

theorem xy_fraction_equivalence
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (x^2 + 4 * x * y) / (y^2 - 4 * x * y) = 3) :
  (x^2 - 4 * x * y) / (y^2 + 4 * x * y) = -1 :=
sorry

end NUMINAMATH_GPT_xy_fraction_equivalence_l1722_172263


namespace NUMINAMATH_GPT_necessary_not_sufficient_condition_l1722_172222

theorem necessary_not_sufficient_condition (x : ℝ) : 
  x^2 - 2 * x - 3 < 0 → -2 < x ∧ x < 3 :=
by  
  sorry

end NUMINAMATH_GPT_necessary_not_sufficient_condition_l1722_172222


namespace NUMINAMATH_GPT_water_fee_part1_water_fee_part2_water_fee_usage_l1722_172208

theorem water_fee_part1 (x : ℕ) (h : 0 < x ∧ x ≤ 6) : y = 2 * x :=
sorry

theorem water_fee_part2 (x : ℕ) (h : x > 6) : y = 3 * x - 6 :=
sorry

theorem water_fee_usage (y : ℕ) (h : y = 27) : x = 11 :=
sorry

end NUMINAMATH_GPT_water_fee_part1_water_fee_part2_water_fee_usage_l1722_172208


namespace NUMINAMATH_GPT_larger_integer_l1722_172291

-- Definitions based on the given conditions
def two_integers (x : ℤ) (y : ℤ) :=
  y = 4 * x ∧ (x + 12) * 2 = y

-- Statement of the problem
theorem larger_integer (x : ℤ) (y : ℤ) (h : two_integers x y) : y = 48 :=
by sorry

end NUMINAMATH_GPT_larger_integer_l1722_172291


namespace NUMINAMATH_GPT_sin_cos_fraction_l1722_172239

theorem sin_cos_fraction (α : ℝ) (h1 : Real.sin α - Real.cos α = 1 / 5) (h2 : α ∈ Set.Ioo (-Real.pi / 2) (Real.pi / 2)) :
    Real.sin α * Real.cos α / (Real.sin α + Real.cos α) = 12 / 35 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_fraction_l1722_172239


namespace NUMINAMATH_GPT_average_rainfall_correct_l1722_172294

-- Define the leap year condition and days in February
def leap_year_february_days : ℕ := 29

-- Define total hours in a day
def hours_in_day : ℕ := 24

-- Define total rainfall in February 2012 in inches
def total_rainfall : ℕ := 420

-- Define total hours in February 2012
def total_hours_february : ℕ := leap_year_february_days * hours_in_day

-- Define the average rainfall calculation
def average_rainfall_per_hour : ℚ :=
  total_rainfall / total_hours_february

-- Theorem to prove the average rainfall is 35/58 inches per hour
theorem average_rainfall_correct :
  average_rainfall_per_hour = 35 / 58 :=
by 
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_average_rainfall_correct_l1722_172294


namespace NUMINAMATH_GPT_budget_allocation_degrees_l1722_172292

theorem budget_allocation_degrees :
  let microphotonics := 12.3
  let home_electronics := 17.8
  let food_additives := 9.4
  let gmo := 21.7
  let industrial_lubricants := 6.2
  let artificial_intelligence := 4.1
  let nanotechnology := 5.3
  let basic_astrophysics := 100 - (microphotonics + home_electronics + food_additives + gmo + industrial_lubricants + artificial_intelligence + nanotechnology)
  (basic_astrophysics * 3.6) + (artificial_intelligence * 3.6) + (nanotechnology * 3.6) = 117.36 :=
by
  sorry

end NUMINAMATH_GPT_budget_allocation_degrees_l1722_172292


namespace NUMINAMATH_GPT_min_area_and_line_eq_l1722_172203

theorem min_area_and_line_eq (a b : ℝ) (l : ℝ → ℝ → Prop)
    (h1 : l 3 2)
    (h2: ∀ x y: ℝ, l x y → (x/a + y/b = 1))
    (h3: a > 0)
    (h4: b > 0)
    : 
    a = 6 ∧ b = 4 ∧ 
    (∀ x y : ℝ, l x y ↔ (4 * x + 6 * y - 24 = 0)) ∧ 
    (∃ min_area : ℝ, min_area = 12) :=
by
  sorry

end NUMINAMATH_GPT_min_area_and_line_eq_l1722_172203


namespace NUMINAMATH_GPT_line_intersects_y_axis_at_point_l1722_172234

theorem line_intersects_y_axis_at_point :
  let x1 := 3
  let y1 := 20
  let x2 := -7
  let y2 := 2

  -- line equation from 2 points: y - y1 = m * (x - x1)
  -- slope m = (y2 - y1) / (x2 - x1)
  -- y-intercept when x = 0:
  
  (0, 14.6) ∈ { p : ℝ × ℝ | ∃ m b, p.2 = m * p.1 + b ∧ 
    m = (y2 - y1) / (x2 - x1) ∧ 
    b = y1 - m * x1 }
  :=
  sorry

end NUMINAMATH_GPT_line_intersects_y_axis_at_point_l1722_172234


namespace NUMINAMATH_GPT_log_sum_even_l1722_172276

noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

-- Define the condition for maximum value at x = 1
def has_max_value_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ y : ℝ, f y ≤ f x

-- Main theorem statement: Prove that lg x + lg y is an even function
theorem log_sum_even (A ω φ : ℝ) (hA : 0 < A) (hω : 0 < ω) 
  (hf_max : has_max_value_at (f A ω φ) 1) : 
  ∀ x y : ℝ, Real.log x + Real.log y = Real.log y + Real.log x := by
  sorry

end NUMINAMATH_GPT_log_sum_even_l1722_172276


namespace NUMINAMATH_GPT_latest_time_for_temperature_at_60_l1722_172206

theorem latest_time_for_temperature_at_60
  (t : ℝ) (h : -t^2 + 10 * t + 40 = 60) : t = 12 :=
sorry

end NUMINAMATH_GPT_latest_time_for_temperature_at_60_l1722_172206


namespace NUMINAMATH_GPT_number_of_teams_l1722_172290

theorem number_of_teams (n : ℕ) (h : n * (n - 1) / 2 = 36) : n = 9 :=
sorry

end NUMINAMATH_GPT_number_of_teams_l1722_172290


namespace NUMINAMATH_GPT_square_TU_squared_l1722_172204

theorem square_TU_squared (P Q R S T U : ℝ × ℝ)
  (side : ℝ) (RT SU PT QU : ℝ)
  (hpqrs : (P.1 - S.1)^2 + (P.2 - S.2)^2 = side^2 ∧ (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = side^2 ∧ 
            (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = side^2 ∧ (S.1 - R.1)^2 + (S.2 - R.2)^2 = side^2)
  (hRT : (R.1 - T.1)^2 + (R.2 - T.2)^2 = RT^2)
  (hSU : (S.1 - U.1)^2 + (S.2 - U.2)^2 = SU^2)
  (hPT : (P.1 - T.1)^2 + (P.2 - T.2)^2 = PT^2)
  (hQU : (Q.1 - U.1)^2 + (Q.2 - U.2)^2 = QU^2)
  (side_eq_17 : side = 17) (RT_SU_eq_8 : RT = 8) (PT_QU_eq_15 : PT = 15) :
  (T.1 - U.1)^2 + (T.2 - U.2)^2 = 979.5 :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_square_TU_squared_l1722_172204


namespace NUMINAMATH_GPT_solve_for_a_l1722_172281

def E (a b c : ℝ) : ℝ := a * b^2 + b * c + c

theorem solve_for_a : (E (-5/8) 3 2 = E (-5/8) 5 3) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l1722_172281


namespace NUMINAMATH_GPT_length_of_room_l1722_172285

theorem length_of_room (area_in_sq_inches : ℕ) (length_of_side_in_feet : ℕ) (h1 : area_in_sq_inches = 14400)
  (h2 : length_of_side_in_feet * length_of_side_in_feet = area_in_sq_inches / 144) : length_of_side_in_feet = 10 :=
  by
  sorry

end NUMINAMATH_GPT_length_of_room_l1722_172285


namespace NUMINAMATH_GPT_combined_balance_l1722_172210

theorem combined_balance (b : ℤ) (g1 g2 : ℤ) (h1 : b = 3456) (h2 : g1 = b / 4) (h3 : g2 = b / 4) : g1 + g2 = 1728 :=
by {
  sorry
}

end NUMINAMATH_GPT_combined_balance_l1722_172210


namespace NUMINAMATH_GPT_fourth_polygon_is_square_l1722_172261

theorem fourth_polygon_is_square
  (angle_triangle angle_square angle_hexagon : ℕ)
  (h_triangle : angle_triangle = 60)
  (h_square : angle_square = 90)
  (h_hexagon : angle_hexagon = 120)
  (h_total : angle_triangle + angle_square + angle_hexagon = 270) :
  ∃ angle_fourth : ℕ, angle_fourth = 90 ∧ (angle_fourth + angle_triangle + angle_square + angle_hexagon = 360) :=
sorry

end NUMINAMATH_GPT_fourth_polygon_is_square_l1722_172261


namespace NUMINAMATH_GPT_area_of_second_side_l1722_172215

theorem area_of_second_side 
  (L W H : ℝ) 
  (h1 : L * H = 120) 
  (h2 : L * W = 60) 
  (h3 : L * W * H = 720) : 
  W * H = 72 :=
sorry

end NUMINAMATH_GPT_area_of_second_side_l1722_172215


namespace NUMINAMATH_GPT_probability_of_Xiaojia_selection_l1722_172229

theorem probability_of_Xiaojia_selection : 
  let students := 2500
  let teachers := 350
  let support_staff := 150
  let total_individuals := students + teachers + support_staff
  let sampled_individuals := 300
  let student_sample := (students : ℝ)/total_individuals * sampled_individuals
  (student_sample / students) = (1 / 10) := 
by
  sorry

end NUMINAMATH_GPT_probability_of_Xiaojia_selection_l1722_172229


namespace NUMINAMATH_GPT_inequality_solution_l1722_172233

theorem inequality_solution :
  ∀ x : ℝ, (x - 3) / (x^2 + 4 * x + 10) ≥ 0 ↔ x ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1722_172233


namespace NUMINAMATH_GPT_problem_1_problem_2_l1722_172256

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |x + 1|

theorem problem_1 : {x : ℝ | f x < 4} = {x : ℝ | -4 / 3 < x ∧ x < 4 / 3} :=
by 
  sorry

theorem problem_2 (x₀ : ℝ) (h : ∀ t : ℝ, f x₀ < |m + t| + |t - m|) : 
  {m : ℝ | ∃ x t, f x < |m + t| + |t - m|} = {m : ℝ | m < -3 / 4 ∨ m > 3 / 4} :=
by 
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1722_172256


namespace NUMINAMATH_GPT_Monica_class_ratio_l1722_172269

theorem Monica_class_ratio : 
  (20 + 25 + 25 + x + 28 + 28 = 136) → 
  (x = 10) → 
  (x / 20 = 1 / 2) :=
by 
  intros h h_x
  sorry

end NUMINAMATH_GPT_Monica_class_ratio_l1722_172269


namespace NUMINAMATH_GPT_cube_root_approx_l1722_172296

open Classical

theorem cube_root_approx (n : ℤ) (x : ℝ) (h₁ : 2^n = x^3) (h₂ : abs (x - 50) <  1) : n = 17 := by
  sorry

end NUMINAMATH_GPT_cube_root_approx_l1722_172296


namespace NUMINAMATH_GPT_triangle_side_a_l1722_172228

theorem triangle_side_a {a b c : ℝ} (A : ℝ) (hA : A = (2 * Real.pi / 3)) (hb : b = Real.sqrt 2) 
(h_area : 1 / 2 * b * c * Real.sin A = Real.sqrt 3) :
  a = Real.sqrt 14 :=
by 
  sorry

end NUMINAMATH_GPT_triangle_side_a_l1722_172228


namespace NUMINAMATH_GPT_solve_otimes_eq_l1722_172244

def otimes (a b : ℝ) : ℝ := (a - 2) * (b + 1)

theorem solve_otimes_eq : ∃ x : ℝ, otimes (-4) (x + 3) = 6 ↔ x = -5 :=
by
  use -5
  simp [otimes]
  sorry

end NUMINAMATH_GPT_solve_otimes_eq_l1722_172244


namespace NUMINAMATH_GPT_trader_profit_l1722_172280

theorem trader_profit (donation goal extra profit : ℝ) (half_profit : ℝ) 
  (H1 : donation = 310) (H2 : goal = 610) (H3 : extra = 180)
  (H4 : half_profit = profit / 2) 
  (H5 : half_profit + donation = goal + extra) : 
  profit = 960 := 
by
  sorry

end NUMINAMATH_GPT_trader_profit_l1722_172280


namespace NUMINAMATH_GPT_inequality_solution_set_l1722_172279

theorem inequality_solution_set :
  {x : ℝ | x ≠ 0 ∧ x ≠ 2 ∧ (x / (x - 2) + (x + 3) / (3 * x) ≥ 4)} =
  {x : ℝ | 0 < x ∧ x ≤ 1 / 8} ∪ {x : ℝ | 2 < x ∧ x ≤ 6} :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1722_172279


namespace NUMINAMATH_GPT_simplify_expression_l1722_172245

theorem simplify_expression : 4 * (18 / 5) * (25 / -72) = -5 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1722_172245
