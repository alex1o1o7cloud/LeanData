import Mathlib

namespace largest_divisor_l700_70099

theorem largest_divisor (n : ℕ) (hn : n > 0) (h : 360 ∣ n^3) :
  ∃ w : ℕ, w > 0 ∧ w ∣ n ∧ ∀ d : ℕ, (d > 0 ∧ d ∣ n) → d ≤ 30 := 
sorry

end largest_divisor_l700_70099


namespace price_per_litre_mixed_oil_l700_70042

-- Define the given conditions
def cost_oil1 : ℝ := 100 * 45
def cost_oil2 : ℝ := 30 * 57.50
def cost_oil3 : ℝ := 20 * 72
def total_cost : ℝ := cost_oil1 + cost_oil2 + cost_oil3
def total_volume : ℝ := 100 + 30 + 20

-- Define the statement to be proved
theorem price_per_litre_mixed_oil : (total_cost / total_volume) = 51.10 :=
by
  sorry

end price_per_litre_mixed_oil_l700_70042


namespace unique_x1_exists_l700_70010

theorem unique_x1_exists (x : ℕ → ℝ) :
  (∀ n : ℕ+, x (n+1) = x n * (x n + 1 / n)) →
  ∃! (x1 : ℝ), (∀ n : ℕ+, 0 < x n ∧ x n < x (n+1) ∧ x (n+1) < 1) :=
sorry

end unique_x1_exists_l700_70010


namespace min_value_expression_l700_70075

theorem min_value_expression (x : ℝ) (hx : x > 0) : 9 * x + 1 / x^3 ≥ 10 :=
sorry

end min_value_expression_l700_70075


namespace ratio_cost_to_marked_l700_70062

variable (m : ℝ)

def marked_price (m : ℝ) := m

def selling_price (m : ℝ) : ℝ := 0.75 * m

def cost_price (m : ℝ) : ℝ := 0.60 * selling_price m

theorem ratio_cost_to_marked (m : ℝ) : 
  cost_price m / marked_price m = 0.45 := 
by
  sorry

end ratio_cost_to_marked_l700_70062


namespace find_shorter_parallel_side_l700_70061

variable (x : ℝ) (a : ℝ) (b : ℝ) (h : ℝ)

def is_trapezium_area (a b h : ℝ) (area : ℝ) : Prop :=
  area = 1/2 * (a + b) * h

theorem find_shorter_parallel_side
  (h28 : a = 28)
  (h15 : h = 15)
  (hArea : area = 345)
  (hIsTrapezium : is_trapezium_area a b h area):
  b = 18 := 
sorry

end find_shorter_parallel_side_l700_70061


namespace probability_of_sphere_in_cube_l700_70006

noncomputable def cube_volume : Real :=
  (4 : Real)^3

noncomputable def sphere_volume : Real :=
  (4 / 3) * Real.pi * (2 : Real)^3

noncomputable def probability : Real :=
  sphere_volume / cube_volume

theorem probability_of_sphere_in_cube : probability = Real.pi / 6 := by
  sorry

end probability_of_sphere_in_cube_l700_70006


namespace Flynn_tv_minutes_weekday_l700_70028

theorem Flynn_tv_minutes_weekday :
  ∀ (tv_hours_per_weekend : ℕ)
    (tv_hours_per_year : ℕ)
    (weeks_per_year : ℕ) 
    (weekdays_per_week : ℕ),
  tv_hours_per_weekend = 2 →
  tv_hours_per_year = 234 →
  weeks_per_year = 52 →
  weekdays_per_week = 5 →
  (tv_hours_per_year - (tv_hours_per_weekend * weeks_per_year)) / (weekdays_per_week * weeks_per_year) * 60
  = 30 :=
by
  intros tv_hours_per_weekend tv_hours_per_year weeks_per_year weekdays_per_week
        h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end Flynn_tv_minutes_weekday_l700_70028


namespace andy_wrong_questions_l700_70018

theorem andy_wrong_questions
  (a b c d : ℕ)
  (h1 : a + b = c + d + 6)
  (h2 : a + d = b + c + 4)
  (h3 : c = 10) :
  a = 15 :=
by
  sorry

end andy_wrong_questions_l700_70018


namespace neg_proposition_equiv_l700_70069

theorem neg_proposition_equiv :
  (¬ ∀ n : ℕ, n^2 ≤ 2^n) ↔ (∃ n : ℕ, n^2 > 2^n) :=
by
  sorry

end neg_proposition_equiv_l700_70069


namespace radius_increase_rate_l700_70031

theorem radius_increase_rate (r : ℝ) (u : ℝ)
  (h : r = 20) (dS_dt : ℝ) (h_dS_dt : dS_dt = 10 * Real.pi) :
  u = 1 / 4 :=
by
  have S := Real.pi * r^2
  have dS_dt_eq : dS_dt = 2 * Real.pi * r * u := sorry
  rw [h_dS_dt, h] at dS_dt_eq
  exact sorry

end radius_increase_rate_l700_70031


namespace factorial_fraction_simplification_l700_70024

theorem factorial_fraction_simplification : 
  (4 * (Nat.factorial 6) + 24 * (Nat.factorial 5)) / (Nat.factorial 7) = 8 / 7 :=
by
  sorry

end factorial_fraction_simplification_l700_70024


namespace integer_solutions_conditions_even_l700_70032

theorem integer_solutions_conditions_even (n : ℕ) (x : ℕ → ℤ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → 
    x i ^ 2 + x ((i % n) + 1) ^ 2 + 50 = 16 * x i + 12 * x ((i % n) + 1) ) → 
  n % 2 = 0 :=
by 
sorry

end integer_solutions_conditions_even_l700_70032


namespace annies_initial_amount_l700_70014

theorem annies_initial_amount :
  let hamburger_cost := 4
  let cheeseburger_cost := 5
  let french_fries_cost := 3
  let milkshake_cost := 5
  let smoothie_cost := 6
  let people_count := 8
  let burger_discount := 1
  let milkshake_discount := 2
  let smoothie_discount_buy2_get1free := 6
  let sales_tax := 0.08
  let tip_rate := 0.15
  let max_single_person_cost := cheeseburger_cost + french_fries_cost + smoothie_cost
  let total_cost := people_count * max_single_person_cost
  let total_burger_discount := people_count * burger_discount
  let total_milkshake_discount := 4 * milkshake_discount
  let total_smoothie_discount := smoothie_discount_buy2_get1free
  let total_discount := total_burger_discount + total_milkshake_discount + total_smoothie_discount
  let discounted_cost := total_cost - total_discount
  let tax_amount := discounted_cost * sales_tax
  let subtotal_with_tax := discounted_cost + tax_amount
  let original_total_cost := people_count * max_single_person_cost
  let tip_amount := original_total_cost * tip_rate
  let final_amount := subtotal_with_tax + tip_amount
  let annie_has_left := 30
  let annies_initial_money := final_amount + annie_has_left
  annies_initial_money = 144 :=
by
  sorry

end annies_initial_amount_l700_70014


namespace contrapositive_statement_l700_70087

theorem contrapositive_statement {a b : ℤ} :
  (∀ a b : ℤ, (a % 2 = 1 ∧ b % 2 = 1) → (a + b) % 2 = 0) →
  (∀ a b : ℤ, ¬((a + b) % 2 = 0) → ¬(a % 2 = 1 ∧ b % 2 = 1)) :=
by 
  intros h a b
  sorry

end contrapositive_statement_l700_70087


namespace factor_81_minus_4y4_l700_70053

theorem factor_81_minus_4y4 (y : ℝ) : 81 - 4 * y^4 = (9 + 2 * y^2) * (9 - 2 * y^2) := by 
    sorry

end factor_81_minus_4y4_l700_70053


namespace locus_of_M_l700_70067

/-- Define the coordinates of points A and B, and given point M(x, y) with the 
    condition x ≠ ±1, ensure the equation of the locus of point M -/
theorem locus_of_M (x y : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) 
  (h3 : (y / (x + 1)) + (y / (x - 1)) = 2) : x^2 - x * y - 1 = 0 := 
sorry

end locus_of_M_l700_70067


namespace find_original_number_l700_70043

theorem find_original_number (x : ℝ) 
  (h1 : x * 16 = 3408) 
  (h2 : 1.6 * 21.3 = 34.080000000000005) : 
  x = 213 :=
sorry

end find_original_number_l700_70043


namespace money_left_for_lunch_and_snacks_l700_70041

-- Definitions according to the conditions
def ticket_cost_per_person : ℝ := 5
def bus_fare_one_way_per_person : ℝ := 1.50
def total_budget : ℝ := 40
def number_of_people : ℝ := 2

-- The proposition to be proved
theorem money_left_for_lunch_and_snacks : 
  let total_zoo_cost := ticket_cost_per_person * number_of_people
  let total_bus_fare := bus_fare_one_way_per_person * number_of_people * 2
  let total_expense := total_zoo_cost + total_bus_fare
  total_budget - total_expense = 24 :=
by
  sorry

end money_left_for_lunch_and_snacks_l700_70041


namespace ab_cd_eq_neg190_over_9_l700_70046

theorem ab_cd_eq_neg190_over_9 (a b c d : ℝ)
  (h1 : a + b + c = 3)
  (h2 : a + b + d = -2)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = -1) :
  a * b + c * d = -190 / 9 :=
by
  sorry

end ab_cd_eq_neg190_over_9_l700_70046


namespace tidal_power_station_location_l700_70076

-- Define the conditions
def tidal_power_plants : ℕ := 9
def first_bidirectional_plant := 1980
def significant_bidirectional_plant_location : String := "Jiangxia"
def largest_bidirectional_plant : Prop := true

-- Assumptions based on conditions
axiom china_has_9_tidal_power_plants : tidal_power_plants = 9
axiom first_bidirectional_in_1980 : (first_bidirectional_plant = 1980) -> significant_bidirectional_plant_location = "Jiangxia"
axiom largest_bidirectional_in_world : largest_bidirectional_plant

-- Definition of the problem
theorem tidal_power_station_location : significant_bidirectional_plant_location = "Jiangxia" :=
by
  sorry

end tidal_power_station_location_l700_70076


namespace fraction_sum_is_one_l700_70033

theorem fraction_sum_is_one
    (a b c d w x y z : ℝ)
    (h1 : 17 * w + b * x + c * y + d * z = 0)
    (h2 : a * w + 29 * x + c * y + d * z = 0)
    (h3 : a * w + b * x + 37 * y + d * z = 0)
    (h4 : a * w + b * x + c * y + 53 * z = 0)
    (a_ne_17 : a ≠ 17)
    (b_ne_29 : b ≠ 29)
    (c_ne_37 : c ≠ 37)
    (wxyz_nonzero : w ≠ 0 ∨ x ≠ 0 ∨ y ≠ 0) :
    (a / (a - 17)) + (b / (b - 29)) + (c / (c - 37)) + (d / (d - 53)) = 1 := 
sorry

end fraction_sum_is_one_l700_70033


namespace asha_remaining_money_l700_70017

theorem asha_remaining_money :
  let brother := 20
  let father := 40
  let mother := 30
  let granny := 70
  let savings := 100
  let total_money := brother + father + mother + granny + savings
  let spent := (3 / 4) * total_money
  let remaining := total_money - spent
  remaining = 65 :=
by
  sorry

end asha_remaining_money_l700_70017


namespace find_a_l700_70026

-- Given conditions and definitions
def circle_eq (x y : ℝ) : Prop := (x^2 + y^2 - 2*x - 2*y + 1 = 0)
def line_eq (x y a : ℝ) : Prop := (x - 2*y + a = 0)
def chord_length (r : ℝ) : ℝ := 2 * r

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, circle_eq x y) → 
  (∀ x y : ℝ, line_eq x y a) → 
  (∃ x y : ℝ, (x = 1 ∧ y = 1) ∧ (line_eq x y a ∧ chord_length 1 = 2)) → 
  a = 1 := by sorry

end find_a_l700_70026


namespace ants_meet_distance_is_half_total_l700_70090

-- Definitions given in the problem
structure Tile :=
  (width : ℤ)
  (length : ℤ)

structure Ant :=
  (start_position : String)

-- Conditions from the problem
def tile : Tile := ⟨4, 6⟩
def maricota : Ant := ⟨"M"⟩
def nandinha : Ant := ⟨"N"⟩
def total_lengths := 14
def total_widths := 12

noncomputable
def calculate_total_distance (total_lengths : ℤ) (total_widths : ℤ) (tile : Tile) := 
  (total_lengths * tile.length) + (total_widths * tile.width)

-- Question stated as a theorem
theorem ants_meet_distance_is_half_total :
  calculate_total_distance total_lengths total_widths tile = 132 →
  (calculate_total_distance total_lengths total_widths tile) / 2 = 66 :=
by
  intro h
  sorry

end ants_meet_distance_is_half_total_l700_70090


namespace fish_ratio_l700_70081

variables (O R B : ℕ)
variables (h1 : O = B + 25)
variables (h2 : B = 75)
variables (h3 : (O + B + R) / 3 = 75)

theorem fish_ratio : R / O = 1 / 2 :=
sorry

end fish_ratio_l700_70081


namespace exponential_inequality_l700_70013

-- Define the conditions for the problem
variables {x y a : ℝ}
axiom h1 : x > y
axiom h2 : y > 1
axiom h3 : 0 < a
axiom h4 : a < 1

-- State the problem to be proved
theorem exponential_inequality (h1 : x > y) (h2 : y > 1) (h3 : 0 < a) (h4 : a < 1) : a ^ x < a ^ y :=
sorry

end exponential_inequality_l700_70013


namespace sachin_age_l700_70093

theorem sachin_age {Sachin_age Rahul_age : ℕ} (h1 : Sachin_age + 14 = Rahul_age) (h2 : Sachin_age * 9 = Rahul_age * 7) : Sachin_age = 49 := by
sorry

end sachin_age_l700_70093


namespace parabola_properties_l700_70077

-- Definitions of the conditions
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def point_A (a b c : ℝ) : Prop := parabola a b c (-1) = 0
def point_B (a b c m : ℝ) : Prop := parabola a b c m = 0
def opens_downwards (a : ℝ) : Prop := a < 0
def valid_m (m : ℝ) : Prop := 1 < m ∧ m < 2

-- Conclusion ①
def conclusion_1 (a b : ℝ) : Prop := b > 0

-- Conclusion ②
def conclusion_2 (a c : ℝ) : Prop := 3 * a + 2 * c < 0

-- Conclusion ③
def conclusion_3 (a b c x1 x2 y1 y2 : ℝ) : Prop :=
  x1 < x2 ∧ x1 + x2 > 1 ∧ parabola a b c x1 = y1 ∧ parabola a b c x2 = y2 → y1 > y2

-- Conclusion ④
def conclusion_4 (a b c : ℝ) : Prop :=
  a ≤ -1 → ∃ x1 x2 : ℝ, (a * x1^2 + b * x1 + c = 1) ∧ (a * x2^2 + b * x2 + c = 1) ∧ (x1 ≠ x2)

-- The theorem to prove
theorem parabola_properties (a b c m : ℝ) :
  (opens_downwards a) →
  (point_A a b c) →
  (point_B a b c m) →
  (valid_m m) →
  (conclusion_1 a b) ∧ (conclusion_2 a c → false) ∧ (∀ x1 x2 y1 y2, conclusion_3 a b c x1 x2 y1 y2) ∧ (conclusion_4 a b c) :=
by
  sorry

end parabola_properties_l700_70077


namespace sum_of_money_l700_70050

theorem sum_of_money (J C P : ℕ) 
  (h1 : P = 60)
  (h2 : P = 3 * J)
  (h3 : C + 7 = 2 * J) : 
  J + P + C = 113 := 
by
  sorry

end sum_of_money_l700_70050


namespace evaluate_expression_l700_70019

/- The mathematical statement to prove:

Evaluate the expression 2/10 + 4/20 + 6/30, then multiply the result by 3
and show that it equals to 9/5.
-/

theorem evaluate_expression : 
  (2 / 10 + 4 / 20 + 6 / 30) * 3 = 9 / 5 := 
by 
  sorry

end evaluate_expression_l700_70019


namespace distance_to_nearest_town_l700_70057

theorem distance_to_nearest_town (d : ℝ) :
  ¬ (d ≥ 6) → ¬ (d ≤ 5) → ¬ (d ≤ 4) → (d > 5 ∧ d < 6) :=
by
  intro h1 h2 h3
  sorry

end distance_to_nearest_town_l700_70057


namespace greatest_divisor_l700_70082

theorem greatest_divisor (n : ℕ) (h1 : 1428 % n = 9) (h2 : 2206 % n = 13) : n = 129 :=
sorry

end greatest_divisor_l700_70082


namespace additional_time_needed_l700_70049

theorem additional_time_needed (total_parts apprentice_first_phase remaining_parts apprentice_rate master_rate combined_rate : ℕ)
  (h1 : total_parts = 500)
  (h2 : apprentice_first_phase = 45)
  (h3 : remaining_parts = total_parts - apprentice_first_phase)
  (h4 : apprentice_rate = 15)
  (h5 : master_rate = 20)
  (h6 : combined_rate = apprentice_rate + master_rate) :
  remaining_parts / combined_rate = 13 := 
by {
  sorry
}

end additional_time_needed_l700_70049


namespace math_problem_l700_70040

theorem math_problem (a b c : ℚ) 
  (h1 : a * (-2) = 1)
  (h2 : |b + 2| = 5)
  (h3 : c = 5 - 6) :
  4 * a - b + 3 * c = -8 ∨ 4 * a - b + 3 * c = 2 :=
by
  sorry

end math_problem_l700_70040


namespace problem_solution_l700_70084

def x : ℤ := -2 + 3
def y : ℤ := abs (-5)
def z : ℤ := 4 * (-1/4)

theorem problem_solution : x + y + z = 5 := 
by
  -- Definitions based on the problem statement
  have h1 : x = -2 + 3 := rfl
  have h2 : y = abs (-5) := rfl
  have h3 : z = 4 * (-1/4) := rfl
  
  -- Exact result required to be proved. Adding placeholder for steps.
  sorry

end problem_solution_l700_70084


namespace simplify_expression_l700_70098

variable {R : Type} [LinearOrderedField R]

theorem simplify_expression (x y z : R) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) (h_sum : x + y + z = 3) :
  (1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2)) =
    3 / (-9 + 6 * y + 6 * z - 2 * y * z) :=
  sorry

end simplify_expression_l700_70098


namespace average_birds_seen_l700_70035

def MarcusBirds : Nat := 7
def HumphreyBirds : Nat := 11
def DarrelBirds : Nat := 9
def IsabellaBirds : Nat := 15

def totalBirds : Nat := MarcusBirds + HumphreyBirds + DarrelBirds + IsabellaBirds
def numberOfIndividuals : Nat := 4

theorem average_birds_seen : (totalBirds / numberOfIndividuals : Real) = 10.5 := 
by
  -- Proof skipped
  sorry

end average_birds_seen_l700_70035


namespace positive_three_digit_integers_divisible_by_12_and_7_l700_70027

theorem positive_three_digit_integers_divisible_by_12_and_7 : 
  ∃ n : ℕ, n = 11 ∧ ∀ k : ℕ, (k ∣ 12) ∧ (k ∣ 7) ∧ (100 ≤ k) ∧ (k < 1000) :=
by
  sorry

end positive_three_digit_integers_divisible_by_12_and_7_l700_70027


namespace rolls_remaining_to_sell_l700_70034

-- Conditions
def total_rolls_needed : ℕ := 45
def rolls_sold_to_grandmother : ℕ := 1
def rolls_sold_to_uncle : ℕ := 10
def rolls_sold_to_neighbor : ℕ := 6

-- Theorem statement
theorem rolls_remaining_to_sell : (total_rolls_needed - (rolls_sold_to_grandmother + rolls_sold_to_uncle + rolls_sold_to_neighbor) = 28) :=
by
  sorry

end rolls_remaining_to_sell_l700_70034


namespace Gunther_free_time_left_l700_70037

def vacuuming_time := 45
def dusting_time := 60
def folding_laundry_time := 25
def mopping_time := 30
def cleaning_bathroom_time := 40
def wiping_windows_time := 15
def brushing_cats_time := 4 * 5
def washing_dishes_time := 20
def first_tasks_total_time := 2 * 60 + 30
def available_free_time := 5 * 60

theorem Gunther_free_time_left : 
  (available_free_time - 
   (vacuuming_time + dusting_time + folding_laundry_time + 
    mopping_time + cleaning_bathroom_time + 
    wiping_windows_time + brushing_cats_time + 
    washing_dishes_time) = 45) := 
by 
  sorry

end Gunther_free_time_left_l700_70037


namespace relationship_among_a_b_c_l700_70054

noncomputable def a : ℝ := 3 ^ Real.cos (Real.pi / 6)
noncomputable def b : ℝ := Real.log (Real.sin (Real.pi / 6)) / Real.log (1 / 3)
noncomputable def c : ℝ := Real.log (Real.tan (Real.pi / 6)) / Real.log 2

theorem relationship_among_a_b_c : a > b ∧ b > c := 
by
  sorry

end relationship_among_a_b_c_l700_70054


namespace prove_values_of_a_l700_70064

-- Definitions of the conditions
def condition_1 (a x y : ℝ) : Prop := (x * y)^(1/3) = a^(a^2)
def condition_2 (a x y : ℝ) : Prop := (Real.log x / Real.log a * Real.log y / Real.log a) + (Real.log y / Real.log a * Real.log x / Real.log a) = 3 * a^3

-- The proof problem
theorem prove_values_of_a (a x y : ℝ) (h1 : condition_1 a x y) (h2 : condition_2 a x y) : a > 0 ∧ a ≤ 2/3 :=
sorry

end prove_values_of_a_l700_70064


namespace simplify_fraction_l700_70023

noncomputable def sin_15 := Real.sin (15 * Real.pi / 180)
noncomputable def cos_15 := Real.cos (15 * Real.pi / 180)
noncomputable def angle_15 := 15 * Real.pi / 180

theorem simplify_fraction : (1 / sin_15 - 1 / cos_15 = 2 * Real.sqrt 2) :=
by
  sorry

end simplify_fraction_l700_70023


namespace sum_of_fractions_removal_l700_70080

theorem sum_of_fractions_removal :
  (1 / 3 + 1 / 6 + 1 / 9 + 1 / 12 + 1 / 15 + 1 / 18 + 1 / 21) 
  - (1 / 12 + 1 / 21) = 3 / 4 := 
 by sorry

end sum_of_fractions_removal_l700_70080


namespace inequality_proof_l700_70089

variable (u v w : ℝ)

theorem inequality_proof (h1 : u > 0) (h2 : v > 0) (h3 : w > 0) (h4 : u + v + w + Real.sqrt (u * v * w) = 4) :
    Real.sqrt (u * v / w) + Real.sqrt (v * w / u) + Real.sqrt (w * u / v) ≥ u + v + w := 
  sorry

end inequality_proof_l700_70089


namespace min_value_fraction_l700_70096

noncomputable section

open Real

theorem min_value_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 4) : 
  ∃ t : ℝ, (∀ x' y' : ℝ, (x' > 0 ∧ y' > 0 ∧ x' + 2 * y' = 4) → (2 / x' + 1 / y') ≥ t) ∧ t = 2 :=
by
  sorry

end min_value_fraction_l700_70096


namespace sequence_is_constant_l700_70094

noncomputable def sequence_condition (a : ℕ → ℝ) :=
  a 1 = 1 ∧ ∀ m n : ℕ, m > 0 → n > 0 → |a n - a m| ≤ 2 * m * n / (m ^ 2 + n ^ 2)

theorem sequence_is_constant (a : ℕ → ℝ) 
  (h : sequence_condition a) :
  ∀ n : ℕ, n > 0 → a n = 1 :=
by
  sorry

end sequence_is_constant_l700_70094


namespace product_eval_l700_70003

theorem product_eval (a : ℤ) (h : a = 3) : (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 0 :=
by
  sorry

end product_eval_l700_70003


namespace min_value_of_T_l700_70085

noncomputable def T (x p : ℝ) : ℝ := |x - p| + |x - 15| + |x - (15 + p)|

theorem min_value_of_T (p : ℝ) (hp : 0 < p ∧ p < 15) :
  ∃ x, p ≤ x ∧ x ≤ 15 ∧ T x p = 15 :=
sorry

end min_value_of_T_l700_70085


namespace sqrt_inequality_l700_70001

theorem sqrt_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt (a^2 + b^2) ≥ (Real.sqrt 2 / 2) * (a + b) :=
sorry

end sqrt_inequality_l700_70001


namespace find_distance_walker_l700_70055

noncomputable def distance_walked (x t d : ℝ) : Prop :=
  (d = x * t) ∧
  (d = (x + 1) * (3 / 4) * t) ∧
  (d = (x - 1) * (t + 3))

theorem find_distance_walker (x t d : ℝ) (h : distance_walked x t d) : d = 18 := 
sorry

end find_distance_walker_l700_70055


namespace largest_adjacent_to_1_number_of_good_cells_l700_70083

def table_width := 51
def table_height := 3
def total_cells := 153

-- Conditions
def condition_1_present (n : ℕ) : Prop := n ∈ Finset.range (total_cells + 1)
def condition_2_bottom_left : Prop := (1 = 1)
def condition_3_adjacent (a b : ℕ) : Prop := 
  (a = b + 1) ∨ 
  (a + 1 = b) ∧ 
  (condition_1_present a) ∧ 
  (condition_1_present b)

-- Part (a): Largest number adjacent to cell containing 1 is 152.
theorem largest_adjacent_to_1 : ∃ b, b = 152 ∧ condition_3_adjacent 1 b :=
by sorry

-- Part (b): Number of good cells that can contain the number 153 is 76.
theorem number_of_good_cells : ∃ count, count = 76 ∧ 
  ∀ (i : ℕ) (j: ℕ), (i, j) ∈ (Finset.range table_height).product (Finset.range table_width) →
  condition_1_present 153 ∧
  (i = table_height - 1 ∨ j = 0 ∨ j = table_width - 1 ∨ j ∈ (Finset.range (table_width - 2)).erase 1) →
  (condition_3_adjacent (i*table_width + j) 153) :=
by sorry

end largest_adjacent_to_1_number_of_good_cells_l700_70083


namespace area_of_circle_l700_70048

theorem area_of_circle (x y : ℝ) :
  (x^2 + y^2 - 8*x - 6*y = -9) → 
  (∃ (R : ℝ), (x - 4)^2 + (y - 3)^2 = R^2 ∧ π * R^2 = 16 * π) :=
by
  sorry

end area_of_circle_l700_70048


namespace any_nat_as_difference_or_element_l700_70009

noncomputable def seq (q : ℕ → ℕ) : Prop :=
∀ n, q n < 2 * n

theorem any_nat_as_difference_or_element (q : ℕ → ℕ) (h_seq : seq q) (m : ℕ) :
  (∃ k, q k = m) ∨ (∃ k l, q l - q k = m) :=
sorry

end any_nat_as_difference_or_element_l700_70009


namespace min_value_sum_l700_70065

def positive_real (x : ℝ) : Prop := x > 0

theorem min_value_sum (x y : ℝ) (hx : positive_real x) (hy : positive_real y)
  (h : 1 / (x + 2) + 1 / (y + 2) = 1 / 6) : x + y ≥ 20 :=
sorry

end min_value_sum_l700_70065


namespace pow_mult_rule_l700_70008

variable (x : ℝ)

theorem pow_mult_rule : (x^3) * (x^2) = x^5 :=
by sorry

end pow_mult_rule_l700_70008


namespace range_of_m_l700_70092

theorem range_of_m (m : ℝ) (x : ℝ) :
  (|1 - (x - 1) / 2| ≤ 3) →
  (x^2 - 2 * x + 1 - m^2 ≤ 0) →
  (m > 0) →
  (∃ (q_is_necessary_but_not_sufficient_for_p : Prop), q_is_necessary_but_not_sufficient_for_p →
  (m ≥ 8)) :=
by
  sorry

end range_of_m_l700_70092


namespace calculate_group5_students_l700_70073

variable (total_students : ℕ) (freq_group1 : ℕ) (sum_freq_group2_3 : ℝ) (freq_group4 : ℝ)

theorem calculate_group5_students
  (h1 : total_students = 50)
  (h2 : freq_group1 = 7)
  (h3 : sum_freq_group2_3 = 0.46)
  (h4 : freq_group4 = 0.2) :
  (total_students * (1 - (freq_group1 / total_students + sum_freq_group2_3 + freq_group4)) = 10) :=
by
  sorry

end calculate_group5_students_l700_70073


namespace find_smallest_A_divisible_by_51_l700_70007

theorem find_smallest_A_divisible_by_51 :
  ∃ (x y : ℕ), (A = 1100 * x + 11 * y) ∧ 
    (0 ≤ x) ∧ (x ≤ 9) ∧ 
    (0 ≤ y) ∧ (y ≤ 9) ∧ 
    (A % 51 = 0) ∧ 
    (A = 1122) :=
sorry

end find_smallest_A_divisible_by_51_l700_70007


namespace factor_expression_l700_70044

theorem factor_expression (x y z : ℝ) :
  (x - y)^3 + (y - z)^3 + (z - x)^3 ≠ 0 →
  ((x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / ((x - y)^3 + (y - z)^3 + (z - x)^3) =
    (x + y) * (y + z) * (z + x) :=
by
  intro h
  sorry

end factor_expression_l700_70044


namespace sum_of_midpoint_coordinates_l700_70072

theorem sum_of_midpoint_coordinates 
  (x1 y1 z1 x2 y2 z2 : ℝ) 
  (h1 : (x1, y1, z1) = (2, 3, 4)) 
  (h2 : (x2, y2, z2) = (8, 15, 12)) : 
  (x1 + x2) / 2 + (y1 + y2) / 2 + (z1 + z2) / 2 = 22 := 
by
  sorry

end sum_of_midpoint_coordinates_l700_70072


namespace fraction_planted_of_field_is_correct_l700_70068

/-- Given a right triangle with legs 5 units and 12 units, and a small unplanted square S
at the right-angle vertex such that the shortest distance from S to the hypotenuse is 3 units,
prove that the fraction of the field that is planted is 52761/857430. -/
theorem fraction_planted_of_field_is_correct :
  let area_triangle := (5 * 12) / 2
  let area_square := (180 / 169) ^ 2
  let area_planted := area_triangle - area_square
  let fraction_planted := area_planted / area_triangle
  fraction_planted = 52761 / 857430 :=
sorry

end fraction_planted_of_field_is_correct_l700_70068


namespace percent_voters_for_candidate_A_l700_70051

theorem percent_voters_for_candidate_A (d r i u p_d p_r p_i p_u : ℝ) 
  (hd : d = 0.45) (hr : r = 0.30) (hi : i = 0.20) (hu : u = 0.05)
  (hp_d : p_d = 0.75) (hp_r : p_r = 0.25) (hp_i : p_i = 0.50) (hp_u : p_u = 0.50) :
  d * p_d + r * p_r + i * p_i + u * p_u = 0.5375 :=
by
  sorry

end percent_voters_for_candidate_A_l700_70051


namespace arithmetic_problem_l700_70063

theorem arithmetic_problem : 72 * 1313 - 32 * 1313 = 52520 := by
  sorry

end arithmetic_problem_l700_70063


namespace square_area_from_diagonal_l700_70039

theorem square_area_from_diagonal (d : ℝ) (hd : d = 3.8) : 
  ∃ (A : ℝ), A = 7.22 ∧ (∀ s : ℝ, d^2 = 2 * (s^2) → A = s^2) :=
by
  sorry

end square_area_from_diagonal_l700_70039


namespace three_not_divide_thirtyone_l700_70058

theorem three_not_divide_thirtyone : ¬ ∃ q : ℤ, 31 = 3 * q := sorry

end three_not_divide_thirtyone_l700_70058


namespace triangle_properties_l700_70030

theorem triangle_properties :
  (∀ (α β γ : ℝ), α + β + γ = 180 → 
    (α = β ∨ α = γ ∨ β = γ ∨ 
     (α = 60 ∧ β = 60 ∧ γ = 60) ∨
     ¬(α = 90 ∧ β = 90))) :=
by
  -- Placeholder for the actual proof, ensuring the theorem can build
  intros α β γ h₁
  sorry

end triangle_properties_l700_70030


namespace length_of_bridge_is_correct_l700_70059

noncomputable def length_of_bridge (length_of_train : ℕ) (time_in_seconds : ℕ) (speed_in_kmph : ℝ) : ℝ :=
  let speed_in_mps := speed_in_kmph * (1000 / 3600)
  time_in_seconds * speed_in_mps - length_of_train

theorem length_of_bridge_is_correct :
  length_of_bridge 150 40 42.3 = 320 := by
  sorry

end length_of_bridge_is_correct_l700_70059


namespace total_pokemon_cards_l700_70060

-- Definitions based on the problem statement

def dozen_to_cards (dozen : ℝ) : ℝ :=
  dozen * 12

def melanie_cards : ℝ :=
  dozen_to_cards 7.5

def benny_cards : ℝ :=
  dozen_to_cards 9

def sandy_cards : ℝ :=
  dozen_to_cards 5.2

def jessica_cards : ℝ :=
  dozen_to_cards 12.8

def total_cards : ℝ :=
  melanie_cards + benny_cards + sandy_cards + jessica_cards

theorem total_pokemon_cards : total_cards = 414 := 
  by sorry

end total_pokemon_cards_l700_70060


namespace prime_divides_sequence_term_l700_70004

theorem prime_divides_sequence_term (k : ℕ) (h_prime : Nat.Prime k) (h_ne_two : k ≠ 2) (h_ne_five : k ≠ 5) :
  ∃ n ≤ k, k ∣ (Nat.ofDigits 10 (List.replicate n 1)) :=
by
  sorry

end prime_divides_sequence_term_l700_70004


namespace fathers_age_more_than_4_times_son_l700_70066

-- Let F (Father's age) be 44 and S (Son's age) be 10 as given by solving the equations
def X_years_more_than_4_times_son_age (F S X : ℕ) : Prop :=
  F = 4 * S + X ∧ F + 4 = 2 * (S + 4) + 20

theorem fathers_age_more_than_4_times_son (F S X : ℕ) (h1 : F = 44) (h2 : F = 4 * S + X) (h3 : F + 4 = 2 * (S + 4) + 20) :
  X = 4 :=
by
  -- The proof would go here
  sorry

end fathers_age_more_than_4_times_son_l700_70066


namespace point_satisfies_equation_l700_70005

theorem point_satisfies_equation (x y : ℝ) :
  (-1 ≤ x ∧ x ≤ 3) ∧ (-5 ≤ y ∧ y ≤ 1) ∧
  ((3 * x + 2 * y = 5) ∨ (-3 * x + 2 * y = -1) ∨ (3 * x - 2 * y = 13) ∨ (-3 * x - 2 * y = 7))
  → 3 * |x - 1| + 2 * |y + 2| = 6 := 
by 
  sorry

end point_satisfies_equation_l700_70005


namespace calculate_minutes_worked_today_l700_70022

-- Define the conditions
def production_rate := 6 -- shirts per minute
def total_shirts_today := 72 

-- The statement to prove
theorem calculate_minutes_worked_today :
  total_shirts_today / production_rate = 12 := 
by
  sorry

end calculate_minutes_worked_today_l700_70022


namespace remaining_liquid_weight_l700_70038

theorem remaining_liquid_weight 
  (liqX_content : ℝ := 0.20)
  (water_content : ℝ := 0.80)
  (initial_solution : ℝ := 8)
  (evaporated_water : ℝ := 2)
  (added_solution : ℝ := 2)
  (new_solution_fraction : ℝ := 0.25) :
  ∃ (remaining_liquid : ℝ), remaining_liquid = 6 := 
by
  -- Skip the proof to ensure the statement is built successfully
  sorry

end remaining_liquid_weight_l700_70038


namespace average_reading_time_l700_70091

theorem average_reading_time (t_Emery t_Serena : ℕ) (h1 : t_Emery = 20) (h2 : t_Serena = 5 * t_Emery) : 
  (t_Emery + t_Serena) / 2 = 60 := 
by
  sorry

end average_reading_time_l700_70091


namespace min_sum_of_factors_of_2310_l700_70095

theorem min_sum_of_factors_of_2310 : ∃ a b c : ℕ, a * b * c = 2310 ∧ a + b + c = 52 :=
by
  sorry

end min_sum_of_factors_of_2310_l700_70095


namespace roque_bike_time_l700_70097

-- Definitions of conditions
def roque_walk_time_per_trip : ℕ := 2
def roque_walk_trips_per_week : ℕ := 3
def roque_bike_trips_per_week : ℕ := 2
def total_commuting_time_per_week : ℕ := 16

-- Statement of the problem to prove
theorem roque_bike_time (B : ℕ) :
  (roque_walk_time_per_trip * 2 * roque_walk_trips_per_week + roque_bike_trips_per_week * 2 * B = total_commuting_time_per_week) → 
  B = 1 :=
by
  sorry

end roque_bike_time_l700_70097


namespace minimum_percentage_increase_mean_l700_70088

def mean (s : List ℤ) : ℚ :=
  (s.sum : ℚ) / s.length

theorem minimum_percentage_increase_mean (F : List ℤ) (p1 p2 : ℤ) (F' : List ℤ)
  (hF : F = [ -4, -1, 0, 6, 9 ])
  (hp1 : p1 = 2) (hp2 : p2 = 3)
  (hF' : F' = [p1, p2, 0, 6, 9])
  : (mean F' - mean F) / mean F * 100 = 100 := 
sorry

end minimum_percentage_increase_mean_l700_70088


namespace cyclic_sum_inequality_l700_70052

theorem cyclic_sum_inequality (x y z a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (x / (a * y + b * z)) + (y / (a * z + b * x)) + (z / (a * x + b * y)) ≥ 3 / (a + b) :=
  sorry

end cyclic_sum_inequality_l700_70052


namespace x_is_48_percent_of_z_l700_70012

variable {x y z : ℝ}

theorem x_is_48_percent_of_z (h1 : x = 1.20 * y) (h2 : y = 0.40 * z) : x = 0.48 * z :=
by
  sorry

end x_is_48_percent_of_z_l700_70012


namespace tate_initial_tickets_l700_70020

theorem tate_initial_tickets (T : ℕ) (h1 : T + 2 + (T + 2)/2 = 51) : T = 32 := 
by
  sorry

end tate_initial_tickets_l700_70020


namespace problem1_problem2_problem3_l700_70070

-- Definition of sets A, B, and U
def A : Set ℤ := {1, 2, 3, 4, 5}
def B : Set ℤ := {-1, 1, 2, 3}
def U : Set ℤ := {x | -1 ≤ x ∧ x < 6}

-- The complement of B in U
def C_U (B : Set ℤ) : Set ℤ := {x ∈ U | x ∉ B}

-- Problem statements
theorem problem1 : A ∩ B = {1, 2, 3} := by sorry
theorem problem2 : A ∪ B = {-1, 1, 2, 3, 4, 5} := by sorry
theorem problem3 : (C_U B) ∩ A = {4, 5} := by sorry

end problem1_problem2_problem3_l700_70070


namespace hexagon_perimeter_l700_70036

def side_length : ℝ := 4
def number_of_sides : ℕ := 6

theorem hexagon_perimeter :
  6 * side_length = 24 := by
    sorry

end hexagon_perimeter_l700_70036


namespace point_P_inside_circle_l700_70078

theorem point_P_inside_circle
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a > b)
  (e : ℝ)
  (h4 : e = 1 / 2)
  (x1 x2 : ℝ)
  (hx1 : a * x1 ^ 2 + b * x1 - c = 0)
  (hx2 : a * x2 ^ 2 + b * x2 - c = 0) :
  x1 ^ 2 + x2 ^ 2 < 2 :=
by
  sorry

end point_P_inside_circle_l700_70078


namespace find_a_l700_70011

noncomputable def A (a : ℝ) : Set ℝ := {x | x^2 < a^2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}
def C : Set ℝ := {x | 1 < x ∧ x < 2}

theorem find_a (a : ℝ) (h : A a ∩ B = C) : a = 2 ∨ a = -2 := by
  sorry

end find_a_l700_70011


namespace area_of_triangle_ABC_l700_70025

/--
Given a triangle \(ABC\) with points \(D\) and \(E\) on sides \(BC\) and \(AC\) respectively,
where \(BD = 4\), \(DE = 2\), \(EC = 6\), and \(BF = FC = 3\),
proves that the area of triangle \( \triangle ABC \) is \( 18\sqrt{3} \).
-/
theorem area_of_triangle_ABC :
  ∀ (ABC D E : Type) (BD DE EC BF FC : ℝ),
    BD = 4 → DE = 2 → EC = 6 → BF = 3 → FC = 3 → 
    ∃ area, area = 18 * Real.sqrt 3 :=
by
  intros ABC D E BD DE EC BF FC hBD hDE hEC hBF hFC
  sorry

end area_of_triangle_ABC_l700_70025


namespace tournament_game_count_l700_70045

/-- In a tournament with 25 players where each player plays 4 games against each other,
prove that the total number of games played is 1200. -/
theorem tournament_game_count : 
  let n := 25
  let games_per_pair := 4
  let total_games := (n * (n - 1) / 2) * games_per_pair
  total_games = 1200 :=
by
  -- Definitions based on the conditions
  let n := 25
  let games_per_pair := 4

  -- Calculating the total number of games
  let total_games := (n * (n - 1) / 2) * games_per_pair

  -- This is the main goal to prove
  have h : total_games = 1200 := sorry
  exact h

end tournament_game_count_l700_70045


namespace principal_amount_borrowed_l700_70074

theorem principal_amount_borrowed (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) 
  (R_eq : R = 12) (T_eq : T = 3) (SI_eq : SI = 7200) :
  (SI = (P * R * T) / 100) → P = 20000 :=
by sorry

end principal_amount_borrowed_l700_70074


namespace find_opposite_pair_l700_70086

def is_opposite (x y : ℤ) : Prop := x = -y

theorem find_opposite_pair :
  ¬is_opposite 4 4 ∧ ¬is_opposite 2 2 ∧ ¬is_opposite (-8) (-8) ∧ is_opposite 4 (-4) := 
by
  sorry

end find_opposite_pair_l700_70086


namespace volcano_ash_height_l700_70071

theorem volcano_ash_height (r d : ℝ) (h : r = 2700) (h₁ : 2 * r = 18 * d) : d = 300 :=
by
  sorry

end volcano_ash_height_l700_70071


namespace values_of_x_l700_70079

theorem values_of_x (x : ℤ) :
  (∃ t : ℤ, x = 105 * t + 22) ∨ (∃ t : ℤ, x = 105 * t + 37) ↔ 
  (5 * x^3 - x + 17) % 15 = 0 ∧ (2 * x^2 + x - 3) % 7 = 0 :=
by {
  sorry
}

end values_of_x_l700_70079


namespace peach_difference_proof_l700_70002

def red_peaches_odd := 12
def green_peaches_odd := 22
def red_peaches_even := 15
def green_peaches_even := 20
def num_baskets := 20
def num_odd_baskets := num_baskets / 2
def num_even_baskets := num_baskets / 2

def total_red_peaches := (red_peaches_odd * num_odd_baskets) + (red_peaches_even * num_even_baskets)
def total_green_peaches := (green_peaches_odd * num_odd_baskets) + (green_peaches_even * num_even_baskets)
def difference := total_green_peaches - total_red_peaches

theorem peach_difference_proof : difference = 150 := by
  sorry

end peach_difference_proof_l700_70002


namespace calc_3a2b_times_neg_a_squared_l700_70015

variables {a b : ℝ}

theorem calc_3a2b_times_neg_a_squared : 3 * a^2 * b * (-a)^2 = 3 * a^4 * b :=
by
  sorry

end calc_3a2b_times_neg_a_squared_l700_70015


namespace total_food_each_day_l700_70000

-- Definitions as per conditions
def soldiers_first_side : Nat := 4000
def food_per_soldier_first_side : Nat := 10
def soldiers_difference : Nat := 500
def food_difference : Nat := 2

-- Proving the total amount of food
theorem total_food_each_day : 
  let soldiers_second_side := soldiers_first_side - soldiers_difference
  let food_per_soldier_second_side := food_per_soldier_first_side - food_difference
  let total_food_first_side := soldiers_first_side * food_per_soldier_first_side
  let total_food_second_side := soldiers_second_side * food_per_soldier_second_side
  total_food_first_side + total_food_second_side = 68000 := by
  -- Proof is omitted
  sorry

end total_food_each_day_l700_70000


namespace expand_product_l700_70021

-- Define the expressions (x + 3)(x + 8) and x^2 + 11x + 24
def expr1 (x : ℝ) : ℝ := (x + 3) * (x + 8)
def expr2 (x : ℝ) : ℝ := x^2 + 11 * x + 24

-- Prove that the two expressions are equal
theorem expand_product (x : ℝ) : expr1 x = expr2 x := by
  sorry

end expand_product_l700_70021


namespace revenue_fall_percentage_l700_70016

theorem revenue_fall_percentage:
  let oldRevenue := 72.0
  let newRevenue := 48.0
  (oldRevenue - newRevenue) / oldRevenue * 100 = 33.33 :=
by
  let oldRevenue := 72.0
  let newRevenue := 48.0
  sorry

end revenue_fall_percentage_l700_70016


namespace ordered_pair_of_positive_integers_l700_70047

theorem ordered_pair_of_positive_integers :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ (x^y + 4 = y^x) ∧ (3 * x^y = y^x + 10) ∧ (x = 7 ∧ y = 1) :=
by
  sorry

end ordered_pair_of_positive_integers_l700_70047


namespace yacht_capacity_l700_70029

theorem yacht_capacity :
  ∀ (x y : ℕ), (3 * x + 2 * y = 68) → (2 * x + 3 * y = 57) → (3 * x + 6 * y = 96) :=
by
  intros x y h1 h2
  sorry

end yacht_capacity_l700_70029


namespace geometric_sequence_100th_term_l700_70056

theorem geometric_sequence_100th_term :
  ∀ (a₁ a₂ : ℤ) (r : ℤ), a₁ = 5 → a₂ = -15 → r = a₂ / a₁ → 
  (a₁ * r ^ 99 = -5 * 3 ^ 99) :=
by
  intros a₁ a₂ r ha₁ ha₂ hr
  sorry

end geometric_sequence_100th_term_l700_70056
