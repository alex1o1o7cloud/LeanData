import Mathlib

namespace percentage_B_of_C_l302_302891

variable (A B C : ℝ)

theorem percentage_B_of_C (h1 : A = 0.08 * C) (h2 : A = 0.5 * B) : B = 0.16 * C :=
by
  sorry

end percentage_B_of_C_l302_302891


namespace value_of_other_bills_is_40_l302_302256

-- Define the conditions using Lean definitions
def class_fund_contains_only_10_and_other_bills (total_amount : ℕ) (num_other_bills num_10_bills : ℕ) : Prop :=
  total_amount = 120 ∧ num_other_bills = 3 ∧ num_10_bills = 2 * num_other_bills

def value_of_each_other_bill (total_amount num_other_bills : ℕ) : ℕ :=
  total_amount / num_other_bills

-- The theorem we want to prove
theorem value_of_other_bills_is_40 (total_amount num_other_bills : ℕ) 
  (h : class_fund_contains_only_10_and_other_bills total_amount num_other_bills (2 * num_other_bills)) :
  value_of_each_other_bill total_amount num_other_bills = 40 := 
by 
  -- We use the conditions here to ensure they are part of the proof even if we skip the actual proof with sorry
  have h1 : total_amount = 120 := by sorry
  have h2 : num_other_bills = 3 := by sorry
  -- Skipping the proof
  sorry

end value_of_other_bills_is_40_l302_302256


namespace carnival_tickets_l302_302854

theorem carnival_tickets (x : ℕ) (won_tickets : ℕ) (found_tickets : ℕ) (ticket_value : ℕ) (total_value : ℕ)
  (h1 : won_tickets = 5 * x)
  (h2 : found_tickets = 5)
  (h3 : ticket_value = 3)
  (h4 : total_value = 30)
  (h5 : total_value = (won_tickets + found_tickets) * ticket_value) :
  x = 1 :=
by
  -- Proof omitted
  sorry

end carnival_tickets_l302_302854


namespace restaurant_customer_problem_l302_302330

theorem restaurant_customer_problem (x y z : ℕ) 
  (h1 : x = 2 * z)
  (h2 : y = x - 3)
  (h3 : 3 + x + y - z = 8) :
  x = 6 ∧ y = 3 ∧ z = 3 ∧ (x + y = 9) :=
by
  sorry

end restaurant_customer_problem_l302_302330


namespace diamond_sum_l302_302794

def diamond (x : ℚ) : ℚ := (x^3 + 2 * x^2 + 3 * x) / 6

theorem diamond_sum : diamond 2 + diamond 3 + diamond 4 = 92 / 3 := by
  sorry

end diamond_sum_l302_302794


namespace value_of_k_plus_p_l302_302475

theorem value_of_k_plus_p
  (k p : ℝ)
  (h1 : ∀ x : ℝ, 3*x^2 - k*x + p = 0)
  (h_sum_roots : k / 3 = -3)
  (h_prod_roots : p / 3 = -6)
  : k + p = -27 :=
by
  sorry

end value_of_k_plus_p_l302_302475


namespace total_pencils_proof_l302_302466

noncomputable def total_pencils (Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff : ℕ) : ℕ :=
  Asaf_pencils + Alexander_pencils

theorem total_pencils_proof :
  ∀ (Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff : ℕ),
  Asaf_age = 50 →
  Alexander_age = 140 - Asaf_age →
  total_age_diff = Alexander_age - Asaf_age →
  Asaf_pencils = 2 * total_age_diff →
  Alexander_pencils = Asaf_pencils + 60 →
  total_pencils Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff = 220 :=
by
  intros
  sorry

end total_pencils_proof_l302_302466


namespace simplify_sqrt_450_l302_302123

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l302_302123


namespace proportional_relationships_l302_302924

-- Let l, v, t be real numbers indicating distance, velocity, and time respectively.
variables (l v t : ℝ)

-- Define the relationships according to the given formulas
def distance_formula := l = v * t
def velocity_formula := v = l / t
def time_formula := t = l / v

-- Definitions of proportionality
def directly_proportional (x y : ℝ) := ∃ k : ℝ, x = k * y
def inversely_proportional (x y : ℝ) := ∃ k : ℝ, x * y = k

-- The main theorem
theorem proportional_relationships (const_t const_v const_l : ℝ) :
  (distance_formula l v const_t → directly_proportional l v) ∧
  (distance_formula l const_v t → directly_proportional l t) ∧
  (velocity_formula const_l v t → inversely_proportional v t) :=
by
  sorry

end proportional_relationships_l302_302924


namespace haley_spent_32_dollars_l302_302167

noncomputable def total_spending (ticket_price : ℕ) (tickets_bought_self_friends : ℕ) (extra_tickets : ℕ) : ℕ :=
  ticket_price * (tickets_bought_self_friends + extra_tickets)

theorem haley_spent_32_dollars :
  total_spending 4 3 5 = 32 :=
by
  sorry

end haley_spent_32_dollars_l302_302167


namespace ksyusha_travel_time_l302_302270

variables (v S : ℝ)

theorem ksyusha_travel_time :
  (2 * S / v) + (S / (2 * v)) = 30 →
  (S / v) + (2 * S / (2 * v)) = 24 :=
by sorry

end ksyusha_travel_time_l302_302270


namespace Ksyusha_travel_time_l302_302260

theorem Ksyusha_travel_time:
  ∀ (S v : ℝ), 
  (S > 0 ∧ v > 0) →
  (∀ tuesday_distance tuesday_time wednesday_distance wednesday_time: ℝ, 
    tuesday_distance = 3 * S ∧ 
    tuesday_distance = 2 * S + S ∧ 
    tuesday_time = (2 * S) / v + (S / (2 * v)) ∧ 
    tuesday_time = 30) → 
  (∀ wednesday_distance wednesday_time : ℝ, 
    wednesday_distance = S + 2 * S ∧ 
    wednesday_time = S / v + (2 * S) / (2 * v)) →
  wednesday_time = 24 :=
by
  intros S v S_pos v_pos tuesday_distance tuesday_time wednesday_distance wednesday_time
  sorry

end Ksyusha_travel_time_l302_302260


namespace grandpa_uncle_ratio_l302_302723

def initial_collection := 150
def dad_gift := 10
def mum_gift := dad_gift + 5
def auntie_gift := 6
def uncle_gift := auntie_gift - 1
def final_collection := 196
def total_cars_needed := final_collection - initial_collection
def other_gifts := dad_gift + mum_gift + auntie_gift + uncle_gift
def grandpa_gift := total_cars_needed - other_gifts

theorem grandpa_uncle_ratio : grandpa_gift = 2 * uncle_gift := by
  sorry

end grandpa_uncle_ratio_l302_302723


namespace number_of_schools_is_23_l302_302200

-- Conditions and definitions
noncomputable def number_of_students_per_school : ℕ := 3
def beth_rank : ℕ := 37
def carla_rank : ℕ := 64

-- Statement of the proof problem
theorem number_of_schools_is_23
  (n : ℕ)
  (h1 : ∀ i < n, ∃ r1 r2 r3: ℕ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3)
  (h2 : ∀ i < n, ∃ A B C: ℕ, A = (2 * B + 1) ∧ C = A ∧ B = 35 ∧ A < beth_rank ∧ beth_rank < carla_rank):
  n = 23 :=
by
  sorry

end number_of_schools_is_23_l302_302200


namespace total_worth_of_stock_l302_302778

theorem total_worth_of_stock (W : ℝ) 
    (h1 : 0.2 * W * 0.1 = 0.02 * W)
    (h2 : 0.6 * (0.8 * W) * 0.05 = 0.024 * W)
    (h3 : 0.2 * (0.8 * W) = 0.16 * W)
    (h4 : (0.024 * W) - (0.02 * W) = 400) 
    : W = 100000 := 
sorry

end total_worth_of_stock_l302_302778


namespace simplify_sqrt_450_l302_302065

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l302_302065


namespace length_SR_l302_302289

theorem length_SR (cos_S : ℝ) (SP : ℝ) (SR : ℝ) (h1 : cos_S = 0.5) (h2 : SP = 10) (h3 : cos_S = SP / SR) : SR = 20 := by
  sorry

end length_SR_l302_302289


namespace baker_sold_pastries_l302_302187

theorem baker_sold_pastries : 
  ∃ P : ℕ, (97 = P + 89) ∧ P = 8 :=
by 
  sorry

end baker_sold_pastries_l302_302187


namespace correct_calculated_value_l302_302942

theorem correct_calculated_value (N : ℕ) (h : N ≠ 0) :
  N * 16 = 2048 * (N / 128) := by 
  sorry

end correct_calculated_value_l302_302942


namespace length_of_AB_l302_302973

variables {A B P Q : ℝ}
variables (x y : ℝ)

-- Conditions
axiom h1 : A < P ∧ P < Q ∧ Q < B
axiom h2 : P - A = 3 * x
axiom h3 : B - P = 5 * x
axiom h4 : Q - A = 2 * y
axiom h5 : B - Q = 3 * y
axiom h6 : Q - P = 3

-- Theorem statement
theorem length_of_AB : B - A = 120 :=
by
  sorry

end length_of_AB_l302_302973


namespace sqrt_450_eq_15_sqrt_2_l302_302113

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302113


namespace henry_books_l302_302938

theorem henry_books (initial_books packed_boxes each_box room_books coffee_books kitchen_books taken_books : ℕ)
  (h1 : initial_books = 99)
  (h2 : packed_boxes = 3)
  (h3 : each_box = 15)
  (h4 : room_books = 21)
  (h5 : coffee_books = 4)
  (h6 : kitchen_books = 18)
  (h7 : taken_books = 12) :
  initial_books - (packed_boxes * each_box + room_books + coffee_books + kitchen_books) + taken_books = 23 :=
by
  sorry

end henry_books_l302_302938


namespace prove_remainder_l302_302950

def problem_statement : Prop := (33333332 % 8 = 4)

theorem prove_remainder : problem_statement := 
by
  sorry

end prove_remainder_l302_302950


namespace necessary_but_not_sufficient_l302_302275

theorem necessary_but_not_sufficient (a : ℝ) : (a - 1 < 0 ↔ a < 1) ∧ (|a| < 1 → a < 1) ∧ ¬ (a < 1 → |a| < 1) := by
  sorry

end necessary_but_not_sufficient_l302_302275


namespace range_of_m_l302_302552

noncomputable def inequality_solutions (x m : ℝ) := |x + 2| - |x + 3| > m

theorem range_of_m (m : ℝ) : (∃ x : ℝ, inequality_solutions x m) → m < 1 :=
by
  sorry

end range_of_m_l302_302552


namespace total_lives_l302_302488

theorem total_lives (initial_players new_players lives_per_person : ℕ)
  (h_initial : initial_players = 8)
  (h_new : new_players = 2)
  (h_lives : lives_per_person = 6)
  : (initial_players + new_players) * lives_per_person = 60 := 
by
  sorry

end total_lives_l302_302488


namespace units_digit_calculation_l302_302915

-- Define a function to compute the units digit of a number in base 10
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_calculation :
  units_digit (8 * 18 * 1988 - 8^3) = 0 := by
  sorry

end units_digit_calculation_l302_302915


namespace number_of_chickens_l302_302186

def eggs_per_chicken : ℕ := 6
def eggs_per_carton : ℕ := 12
def full_cartons : ℕ := 10

theorem number_of_chickens :
  (full_cartons * eggs_per_carton) / eggs_per_chicken = 20 :=
by
  sorry

end number_of_chickens_l302_302186


namespace simplify_sqrt_450_l302_302092

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l302_302092


namespace maximum_value_of_reciprocals_l302_302548

theorem maximum_value_of_reciprocals (c b : ℝ) (h0 : 0 < b ∧ b < c)
  (e1 : ℝ) (e2 : ℝ)
  (h1 : e1 = c / (Real.sqrt (c^2 + (2 * b)^2)))
  (h2 : e2 = c / (Real.sqrt (c^2 - b^2)))
  (h3 : 1 / e1^2 + 4 / e2^2 = 5) :
  ∃ max_val, max_val = 5 / 2 :=
by
  sorry

end maximum_value_of_reciprocals_l302_302548


namespace sum_of_other_endpoint_coordinates_l302_302972

theorem sum_of_other_endpoint_coordinates (x y : ℤ)
  (h1 : (6 + x) / 2 = 3)
  (h2 : (-1 + y) / 2 = 6) :
  x + y = 13 :=
by
  sorry

end sum_of_other_endpoint_coordinates_l302_302972


namespace tommy_saw_100_wheels_l302_302751

-- Define the parameters
def trucks : ℕ := 12
def cars : ℕ := 13
def wheels_per_truck : ℕ := 4
def wheels_per_car : ℕ := 4

-- Define the statement to prove
theorem tommy_saw_100_wheels : (trucks * wheels_per_truck + cars * wheels_per_car) = 100 := by
  sorry 

end tommy_saw_100_wheels_l302_302751


namespace sqrt_450_eq_15_sqrt_2_l302_302116

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302116


namespace power_mod_equiv_l302_302496

theorem power_mod_equiv :
  2^1000 % 17 = 1 := by
  sorry

end power_mod_equiv_l302_302496


namespace Peter_bought_5_kilos_of_cucumbers_l302_302282

/-- 
Peter carried $500 to the market. 
He bought 6 kilos of potatoes for $2 per kilo, 
9 kilos of tomato for $3 per kilo, 
some kilos of cucumbers for $4 per kilo, 
and 3 kilos of bananas for $5 per kilo. 
After buying all these items, Peter has $426 remaining. 
How many kilos of cucumbers did Peter buy? 
-/
theorem Peter_bought_5_kilos_of_cucumbers : 
   ∃ (kilos_cucumbers : ℕ),
   (500 - (6 * 2 + 9 * 3 + 3 * 5 + kilos_cucumbers * 4) = 426) →
   kilos_cucumbers = 5 :=
sorry

end Peter_bought_5_kilos_of_cucumbers_l302_302282


namespace ratio_unchanged_l302_302738

-- Define the initial ratio
def initial_ratio (a b : ℕ) : ℚ := a / b

-- Define the new ratio after transformation
def new_ratio (a b : ℕ) : ℚ := (3 * a) / (b / (1/3))

-- The theorem stating that the ratio remains unchanged
theorem ratio_unchanged (a b : ℕ) (hb : b ≠ 0) :
  initial_ratio a b = new_ratio a b :=
by
  sorry

end ratio_unchanged_l302_302738


namespace number_of_lines_l302_302247

-- Define points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 1)

-- Define the distances from the points
def d_A : ℝ := 1
def d_B : ℝ := 2

-- A theorem stating the number of lines under the given conditions
theorem number_of_lines (A B : ℝ × ℝ) (d_A d_B : ℝ) (hA : A = (1, 2)) (hB : B = (3, 1)) (hdA : d_A = 1) (hdB : d_B = 2) :
  ∃ n : ℕ, n = 2 :=
by {
  sorry
}

end number_of_lines_l302_302247


namespace probability_at_least_two_same_row_col_l302_302667

-- Define the total number of ways to select 3 numbers from 9
def total_ways (n k : ℕ) : ℕ := Nat.choose n k

-- Define the number of ways to pick 3 numbers such that no two are in the same row/column
def ways_no_two_same_row_col : ℕ := 6

-- Define the probability calculation
def calculate_probability (total ways_excluded : ℕ) : ℚ :=
  (total - ways_excluded) / total

-- The problem to prove
theorem probability_at_least_two_same_row_col :
  calculate_probability (total_ways 9 3) ways_no_two_same_row_col = 13 / 14 :=
by
  sorry

end probability_at_least_two_same_row_col_l302_302667


namespace initial_overs_l302_302244

theorem initial_overs (initial_run_rate remaining_run_rate target runs initially remaining_overs : ℝ)
    (h_target : target = 282)
    (h_remaining_overs : remaining_overs = 40)
    (h_initial_run_rate : initial_run_rate = 3.6)
    (h_remaining_run_rate : remaining_run_rate = 6.15)
    (h_target_eq : initial_run_rate * initially + remaining_run_rate * remaining_overs = target) :
    initially = 10 :=
by
  sorry

end initial_overs_l302_302244


namespace benny_total_hours_l302_302189

-- Define the conditions
def hours_per_day : ℕ := 3
def days_worked : ℕ := 6

-- State the theorem (problem) to be proved
theorem benny_total_hours : hours_per_day * days_worked = 18 :=
by
  -- Sorry to skip the actual proof
  sorry

end benny_total_hours_l302_302189


namespace fourth_circle_radius_l302_302438

theorem fourth_circle_radius (c : ℝ) (h : c > 0) :
  let r := c / 5
  let fourth_radius := (3 * c) / 10
  fourth_radius = (c / 2) - r :=
by
  let r := c / 5
  let fourth_radius := (3 * c) / 10
  sorry

end fourth_circle_radius_l302_302438


namespace translate_statement_to_inequality_l302_302907

theorem translate_statement_to_inequality (y : ℝ) : (1/2) * y + 5 > 0 ↔ True := 
sorry

end translate_statement_to_inequality_l302_302907


namespace tutors_meeting_schedule_l302_302439

/-- In a school, five tutors, Jaclyn, Marcelle, Susanna, Wanda, and Thomas, 
are scheduled to work in the library. Their schedules are as follows: 
Jaclyn works every fifth school day, Marcelle works every sixth school day, 
Susanna works every seventh school day, Wanda works every eighth school day, 
and Thomas works every ninth school day. Today, all five tutors are working 
in the library. Prove that the least common multiple of 5, 6, 7, 8, and 9 is 2520 days. 
-/
theorem tutors_meeting_schedule : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9))) = 2520 := 
by
  sorry

end tutors_meeting_schedule_l302_302439


namespace range_of_a_l302_302861

theorem range_of_a (a : ℝ) (A : Set ℝ) (hA : ∀ x, x ∈ A ↔ a / (x - 1) < 1) (h_not_in : 2 ∉ A) : a ≥ 1 := 
sorry

end range_of_a_l302_302861


namespace log_bound_sum_l302_302905

theorem log_bound_sum (c d : ℕ) (h_c : c = 10) (h_d : d = 11) (h_bound : 10 < Real.log 1350 / Real.log 2 ∧ Real.log 1350 / Real.log 2 < 11) : c + d = 21 :=
by
  -- omitted proof
  sorry

end log_bound_sum_l302_302905


namespace animal_products_sampled_l302_302771

theorem animal_products_sampled
  (grains : ℕ)
  (oils : ℕ)
  (animal_products : ℕ)
  (fruits_vegetables : ℕ)
  (total_sample : ℕ)
  (total_food_types : grains + oils + animal_products + fruits_vegetables = 100)
  (sample_size : total_sample = 20)
  : (animal_products * total_sample / 100) = 6 := by
  sorry

end animal_products_sampled_l302_302771


namespace simplify_sqrt_450_l302_302024

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l302_302024


namespace curved_surface_area_of_sphere_l302_302737

theorem curved_surface_area_of_sphere (r : ℝ) (h : r = 4) : 4 * π * r^2 = 64 * π :=
by
  rw [h, sq]
  norm_num
  sorry

end curved_surface_area_of_sphere_l302_302737


namespace balanced_phrases_not_detected_l302_302632

def reduction_rules : Set String :=
  { "(()) -> A", "(A) -> A", "AA -> A" }

def formula_f (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 1
  else (formula_f (n-1)) + (formula_f (n-2)) + ∑ i in finset.range (n-2), (formula_f i) * (formula_f (n-i-1))

def C_n (n : ℕ) : ℕ :=
  (finset.range (n+1)).binom (2*n)

theorem balanced_phrases_not_detected (n : ℕ) (hn : n = 7) :
  C_n n - formula_f n = 392 :=
by
  rw [hn]
  have : C_n 7 = 429 := by
    rw [C_n]
    norm_num
  have : formula_f 7 = 37 := by
    norm_num
  norm_cast
  norm_num
  sorry

end balanced_phrases_not_detected_l302_302632


namespace marcel_potatoes_eq_l302_302721

-- Define the given conditions
def marcel_corn := 10
def dale_corn := marcel_corn / 2
def dale_potatoes := 8
def total_vegetables := 27

-- Define the fact that they bought 27 vegetables in total
def total_corn := marcel_corn + dale_corn
def total_potatoes := total_vegetables - total_corn

-- State the theorem
theorem marcel_potatoes_eq :
  (total_potatoes - dale_potatoes) = 4 :=
by
  -- Lean proof would go here
  sorry

end marcel_potatoes_eq_l302_302721


namespace simplify_sqrt_450_l302_302097

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l302_302097


namespace chapatis_ordered_l302_302897

theorem chapatis_ordered (C : ℕ) 
  (chapati_cost : ℕ) (plates_rice : ℕ) (rice_cost : ℕ)
  (plates_mixed_veg : ℕ) (mixed_veg_cost : ℕ)
  (ice_cream_cups : ℕ) (ice_cream_cost : ℕ)
  (total_amount_paid : ℕ)
  (cost_eq : chapati_cost = 6)
  (plates_rice_eq : plates_rice = 5)
  (rice_cost_eq : rice_cost = 45)
  (plates_mixed_veg_eq : plates_mixed_veg = 7)
  (mixed_veg_cost_eq : mixed_veg_cost = 70)
  (ice_cream_cups_eq : ice_cream_cups = 6)
  (ice_cream_cost_eq : ice_cream_cost = 40)
  (total_paid_eq : total_amount_paid = 1051) :
  6 * C + 5 * 45 + 7 * 70 + 6 * 40 = 1051 → C = 16 :=
by
  intro h
  sorry

end chapatis_ordered_l302_302897


namespace complement_of_A_l302_302279

open Set

def U : Set ℕ := {n | 1 ≤ n ∧ n ≤ 10}
def A : Set ℕ := {1, 2, 3, 5, 8}

theorem complement_of_A :
  (U \ A) = {4, 6, 7, 9, 10} :=
by
sorry

end complement_of_A_l302_302279


namespace telephone_number_problem_l302_302895

theorem telephone_number_problem :
  ∃ A B C D E F G H I J : ℕ,
    (A > B) ∧ (B > C) ∧ (D > E) ∧ (E > F) ∧ (G > H) ∧ (H > I) ∧ (I > J) ∧
    (D = E + 1) ∧ (E = F + 1) ∧ (D % 2 = 0) ∧ 
    (G = H + 2) ∧ (H = I + 2) ∧ (I = J + 2) ∧ (G % 2 = 1) ∧ (H % 2 = 1) ∧ (I % 2 = 1) ∧ (J % 2 = 1) ∧
    (A + B + C = 7) ∧ (B + C + F = 10) ∧ (A = 7) :=
sorry

end telephone_number_problem_l302_302895


namespace sqrt_450_eq_15_sqrt_2_l302_302069

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l302_302069


namespace find_m_l302_302231

theorem find_m (m : ℝ) (A : Set ℝ) (hA : A = {0, m, m^2 - 3 * m + 2}) (h2 : 2 ∈ A) : m = 3 :=
  sorry

end find_m_l302_302231


namespace problem_statement_l302_302502

def a := 596
def b := 130
def c := 270

theorem problem_statement : a - b - c = a - (b + c) := by
  sorry

end problem_statement_l302_302502


namespace sqrt_450_eq_15_sqrt_2_l302_302056

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302056


namespace Jeremy_payment_total_l302_302830

theorem Jeremy_payment_total :
  let room_rate := (13 : ℚ) / 3
  let rooms_cleaned := (8 : ℚ) / 5
  let window_rate := (5 : ℚ) / 2
  let windows_cleaned := (11 : ℚ) / 4
  let payment_rooms := room_rate * rooms_cleaned
  let payment_windows := window_rate * windows_cleaned
  let total_payment := payment_rooms + payment_windows
  total_payment = (553 : ℚ) / 40 :=
by {
  -- Definitions
  let room_rate := (13 : ℚ) / 3
  let rooms_cleaned := (8 : ℚ) / 5
  let window_rate := (5 : ℚ) / 2
  let windows_cleaned := (11 : ℚ) / 4
  let payment_rooms := room_rate * rooms_cleaned
  let payment_windows := window_rate * windows_cleaned
  let total_payment := payment_rooms + payment_windows
  
  -- Main goal
  sorry
}

end Jeremy_payment_total_l302_302830


namespace percentage_decrease_second_year_l302_302440

-- Define initial population
def initial_population : ℝ := 14999.999999999998

-- Define the population at the end of the first year after 12% increase
def population_end_year_1 : ℝ := initial_population * 1.12

-- Define the final population at the end of the second year
def final_population : ℝ := 14784.0

-- Define the proof statement
theorem percentage_decrease_second_year :
  ∃ D : ℝ, final_population = population_end_year_1 * (1 - D / 100) ∧ D = 12 :=
by
  sorry

end percentage_decrease_second_year_l302_302440


namespace harry_worked_16_hours_l302_302310

-- Define the given conditions
def harrys_pay_first_30_hours (x : ℝ) : ℝ := 30 * x
def harrys_pay_additional_hours (x H : ℝ) : ℝ := (H - 30) * 2 * x
def james_pay_first_40_hours (x : ℝ) : ℝ := 40 * x
def james_pay_additional_hour (x : ℝ) : ℝ := 2 * x
def james_total_hours : ℝ := 41

-- Given that Harry and James are paid the same amount 
-- Prove that Harry worked 16 hours last week
theorem harry_worked_16_hours (x H : ℝ) 
  (h1 : harrys_pay_first_30_hours x + harrys_pay_additional_hours x H = james_pay_first_40_hours x + james_pay_additional_hour x) 
  : H = 16 :=
by
  sorry

end harry_worked_16_hours_l302_302310


namespace age_ratio_l302_302506

theorem age_ratio (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 10) (h3 : a + b + c = 27) : b / c = 2 := by
  sorry

end age_ratio_l302_302506


namespace cost_price_per_meter_l302_302874

theorem cost_price_per_meter (number_of_meters : ℕ) (selling_price : ℝ) (profit_per_meter : ℝ) (total_cost_price : ℝ) (cost_per_meter : ℝ) :
  number_of_meters = 85 →
  selling_price = 8925 →
  profit_per_meter = 15 →
  total_cost_price = selling_price - (profit_per_meter * number_of_meters) →
  cost_per_meter = total_cost_price / number_of_meters →
  cost_per_meter = 90 :=
by
  intros h1 h2 h3 h4 h5 
  sorry

end cost_price_per_meter_l302_302874


namespace sqrt_450_eq_15_sqrt_2_l302_302071

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l302_302071


namespace simplify_sqrt_450_l302_302028

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l302_302028


namespace trigonometric_identity_l302_302945

open Real

theorem trigonometric_identity (α : ℝ) (h : α ∈ Set.Ioo (-π) (-π / 2)) : 
  sqrt ((1 + cos α) / (1 - cos α)) - sqrt ((1 - cos α) / (1 + cos α)) = 2 / tan α := 
by
  sorry

end trigonometric_identity_l302_302945


namespace subset_single_element_l302_302435

-- Define the set X
def X : Set ℝ := { x | x > -1 }

-- The proof statement
-- We need to prove that {0} ⊆ X
theorem subset_single_element : {0} ⊆ X :=
sorry

end subset_single_element_l302_302435


namespace sum_of_products_l302_302224

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 390) 
  (h2 : a + b + c = 20) : 
  ab + bc + ca = 5 :=
by 
  sorry

end sum_of_products_l302_302224


namespace complement_union_l302_302688

open Set

def U := { x : ℤ | x^2 - 5*x - 6 ≤ 0 }
def A := { x : ℤ | x * (2 - x) ≥ 0 }
def B := {1, 2, 3}

theorem complement_union (x : ℤ) : 
  x ∈ U \ (A ∪ B) ↔ x ∈ {-1, 4, 5, 6} := by
  sorry

end complement_union_l302_302688


namespace units_digit_calculation_l302_302916

-- Define a function to compute the units digit of a number in base 10
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_calculation :
  units_digit (8 * 18 * 1988 - 8^3) = 0 := by
  sorry

end units_digit_calculation_l302_302916


namespace Ksyusha_time_to_school_l302_302274

/-- Ksyusha runs twice as fast as she walks (both speeds are constant).
On Tuesday, Ksyusha walked a distance twice the distance she ran, and she reached school in 30 minutes.
On Wednesday, Ksyusha ran twice the distance she walked.
Prove that it took her 24 minutes to get from home to school on Wednesday. -/
theorem Ksyusha_time_to_school
  (S v : ℝ)                  -- S for distance unit, v for walking speed
  (htuesday : 2 * (2 * S / v) + S / (2 * v) = 30) -- Condition on Tuesday's travel
  :
  (S / v + 2 * S / (2 * v) = 24)                 -- Claim about Wednesday's travel time
:=
  sorry

end Ksyusha_time_to_school_l302_302274


namespace find_a_l302_302223

theorem find_a (a b c : ℕ) (h1 : (18 ^ a) * (9 ^ (3 * a - 1)) * (c ^ (2 * a - 3)) = (2 ^ 7) * (3 ^ b)) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) : a = 7 :=
by
  sorry

end find_a_l302_302223


namespace common_ratio_arithmetic_progression_l302_302185

theorem common_ratio_arithmetic_progression (a3 q : ℝ) (h1 : a3 = 9) (h2 : a3 + a3 * q + 9 = 27) :
  q = 1 ∨ q = -1 / 2 :=
by
  sorry

end common_ratio_arithmetic_progression_l302_302185


namespace complement_intersection_l302_302834

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

theorem complement_intersection (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {3, 4, 5}) :
  U \ (A ∩ B) = {1, 2, 4, 5} :=
by
  sorry

end complement_intersection_l302_302834


namespace geometric_sequence_ratio_l302_302806

/-
Given a geometric sequence {a_n} with common ratio q ≠ -1 and q ≠ 1,
and S_n is the sum of the first n terms of the geometric sequence.
Given S_{12} = 7 S_{4}, prove:
S_{8}/S_{4} = 3
-/

theorem geometric_sequence_ratio {a_n : ℕ → ℝ} (q : ℝ) (h₁ : q ≠ -1) (h₂ : q ≠ 1)
  (S : ℕ → ℝ) (hSn : ∀ n, S n = a_n 0 * (1 - q ^ n) / (1 - q)) (h : S 12 = 7 * S 4) :
  S 8 / S 4 = 3 :=
by
  sorry

end geometric_sequence_ratio_l302_302806


namespace quadratic_solve_l302_302731

theorem quadratic_solve (x : ℝ) : (x + 4)^2 = 5 * (x + 4) → x = -4 ∨ x = 1 :=
by sorry

end quadratic_solve_l302_302731


namespace cost_of_large_tubs_l302_302513

theorem cost_of_large_tubs (L : ℝ) (h1 : 3 * L + 6 * 5 = 48) : L = 6 :=
by {
  sorry
}

end cost_of_large_tubs_l302_302513


namespace infection_in_fourth_round_l302_302504

-- Define the initial conditions and the function for the geometric sequence
def initial_infected : ℕ := 1
def infection_ratio : ℕ := 20

noncomputable def infected_computers (rounds : ℕ) : ℕ :=
  initial_infected * infection_ratio^(rounds - 1)

-- The theorem to prove
theorem infection_in_fourth_round : infected_computers 4 = 8000 :=
by
  -- proof will be added later
  sorry

end infection_in_fourth_round_l302_302504


namespace sum_y_coordinates_of_other_vertices_of_parallelogram_l302_302943

theorem sum_y_coordinates_of_other_vertices_of_parallelogram :
  let x1 := 4
  let y1 := 26
  let x2 := 12
  let y2 := -8
  let midpoint_y := (y1 + y2) / 2
  2 * midpoint_y = 18 := by
    sorry

end sum_y_coordinates_of_other_vertices_of_parallelogram_l302_302943


namespace sandwiches_with_ten_loaves_l302_302182

def sandwiches_per_loaf : ℕ := 18 / 3

def num_sandwiches (loaves: ℕ) : ℕ := sandwiches_per_loaf * loaves

theorem sandwiches_with_ten_loaves :
  num_sandwiches 10 = 60 := by
  sorry

end sandwiches_with_ten_loaves_l302_302182


namespace unique_two_digit_solution_exists_l302_302482

theorem unique_two_digit_solution_exists :
  ∃! (s : ℤ), 10 ≤ s ∧ s < 100 ∧ 13 * s % 100 = 52 :=
begin
  use 4,
  split,
  { split,
    { linarith },
    { linarith },
    { norm_num }
  },
  { intros y hy,
    cases hy with hy1 hy2,
    cases hy2 with hy2 hy3,
    have : 77 * 52 % 100 = 4,
    { norm_num },
    have h : y ≡ 4 [MOD 100] := (congr_arg (λ x, 77 * x % 100) hy3).trans this,
    norm_num at h,
    linarith }
end

end unique_two_digit_solution_exists_l302_302482


namespace num_ways_to_place_balls_in_boxes_l302_302786

theorem num_ways_to_place_balls_in_boxes (num_balls num_boxes : ℕ) (hB : num_balls = 4) (hX : num_boxes = 3) : 
  (num_boxes ^ num_balls) = 81 := by
  rw [hB, hX]
  sorry

end num_ways_to_place_balls_in_boxes_l302_302786


namespace quadratic_equation_has_root_l302_302725

theorem quadratic_equation_has_root (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) :
  ∃ (x : ℝ), (a * x^2 + 2 * b * x + c = 0) ∨
             (b * x^2 + 2 * c * x + a = 0) ∨
             (c * x^2 + 2 * a * x + b = 0) :=
sorry

end quadratic_equation_has_root_l302_302725


namespace time_for_worker_C_l302_302747

theorem time_for_worker_C (time_A time_B time_total : ℝ) (time_A_pos : 0 < time_A) (time_B_pos : 0 < time_B) (time_total_pos : 0 < time_total) 
  (hA : time_A = 12) (hB : time_B = 15) (hTotal : time_total = 6) : 
  (1 / (1 / time_total - 1 / time_A - 1 / time_B) = 60) :=
by 
  sorry

end time_for_worker_C_l302_302747


namespace find_base_l302_302350

theorem find_base (b : ℕ) : (b^3 ≤ 64 ∧ 64 < b^4) ↔ b = 4 := 
by
  sorry

end find_base_l302_302350


namespace max_value_f_period_f_l302_302352

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos x) ^ 2 - (Real.cos x) ^ 4

theorem max_value_f : ∃ x : ℝ, (f x) = 1 / 4 :=
sorry

theorem period_f : ∃ p : ℝ, p = π / 2 ∧ ∀ x : ℝ, f (x + p) = f x :=
sorry

end max_value_f_period_f_l302_302352


namespace remaining_figure_area_l302_302637

-- Definitions based on conditions
def original_semi_circle_radius_from_chord (L : ℝ) : ℝ := L / 2

def area_of_semi_circle (r : ℝ) : ℝ := (π * r^2) / 2

def remaining_area (L : ℝ) : ℝ :=
  let R := original_semi_circle_radius_from_chord L
  in area_of_semi_circle R - 2 * area_of_semi_circle (R / 2)

-- Given problem rewritten in Lean
theorem remaining_figure_area (hL : 8) : 
  abs (remaining_area 8 - 12.57) < 0.01 := by
  sorry

end remaining_figure_area_l302_302637


namespace christina_walking_speed_l302_302446

noncomputable def christina_speed : ℕ :=
  let distance_between := 270
  let jack_speed := 4
  let lindy_total_distance := 240
  let lindy_speed := 8
  let meeting_time := lindy_total_distance / lindy_speed
  let jack_covered := jack_speed * meeting_time
  let remaining_distance := distance_between - jack_covered
  remaining_distance / meeting_time

theorem christina_walking_speed : christina_speed = 5 := by
  -- Proof will be provided here to verify the theorem, but for now, we use sorry to skip it
  sorry

end christina_walking_speed_l302_302446


namespace geom_sequence_a4_times_a7_l302_302953

theorem geom_sequence_a4_times_a7 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_q : q = 2) 
  (h_a2_a5 : a 2 * a 5 = 32) : 
  a 4 * a 7 = 512 :=
by 
  sorry

end geom_sequence_a4_times_a7_l302_302953


namespace solve_for_k_l302_302932

theorem solve_for_k (x y k : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h : (1 / 2)^(25 * x) * (1 / 81)^k = 1 / (18 ^ (25 * y))) :
  k = 25 * y / 2 :=
by
  sorry

end solve_for_k_l302_302932


namespace Ksyusha_time_to_school_l302_302272

/-- Ksyusha runs twice as fast as she walks (both speeds are constant).
On Tuesday, Ksyusha walked a distance twice the distance she ran, and she reached school in 30 minutes.
On Wednesday, Ksyusha ran twice the distance she walked.
Prove that it took her 24 minutes to get from home to school on Wednesday. -/
theorem Ksyusha_time_to_school
  (S v : ℝ)                  -- S for distance unit, v for walking speed
  (htuesday : 2 * (2 * S / v) + S / (2 * v) = 30) -- Condition on Tuesday's travel
  :
  (S / v + 2 * S / (2 * v) = 24)                 -- Claim about Wednesday's travel time
:=
  sorry

end Ksyusha_time_to_school_l302_302272


namespace line_passing_quadrants_l302_302160

-- Definition of the line
def line (x : ℝ) : ℝ := 5 * x - 2

-- Prove that the line passes through Quadrants I, III, and IV
theorem line_passing_quadrants :
  ∃ x_1 > 0, line x_1 > 0 ∧      -- Quadrant I
  ∃ x_3 < 0, line x_3 < 0 ∧      -- Quadrant III
  ∃ x_4 > 0, line x_4 < 0 :=     -- Quadrant IV
sorry

end line_passing_quadrants_l302_302160


namespace line_through_intersection_parallel_to_given_line_l302_302204

theorem line_through_intersection_parallel_to_given_line :
  ∃ k : ℝ, (∀ x y : ℝ, (2 * x + 3 * y + k = 0 ↔ (x, y) = (2, 1)) ∧
  (∀ m n : ℝ, (2 * m + 3 * n + 5 = 0 → 2 * m + 3 * n + k = 0))) →
  2 * x + 3 * y - 7 = 0 :=
sorry

end line_through_intersection_parallel_to_given_line_l302_302204


namespace cone_inscribed_spheres_distance_l302_302180

noncomputable def distance_between_sphere_centers (R α : ℝ) : ℝ :=
  R * (Real.sqrt 2) * Real.sin (α / 2) / (2 * Real.cos (α / 8) * Real.cos ((45 : ℝ) - α / 8))

theorem cone_inscribed_spheres_distance (R α : ℝ) (h1 : R > 0) (h2 : α > 0) :
  distance_between_sphere_centers R α = R * (Real.sqrt 2) * Real.sin (α / 2) / (2 * Real.cos (α / 8) * Real.cos ((45 : ℝ) - α / 8)) :=
by 
  sorry

end cone_inscribed_spheres_distance_l302_302180


namespace task1_on_time_task2_not_on_time_prob_l302_302497

def task1_on_time_prob : ℚ := 3 / 8
def task2_on_time_prob : ℚ := 3 / 5

theorem task1_on_time_task2_not_on_time_prob :
  task1_on_time_prob * (1 - task2_on_time_prob) = 3 / 20 := by
  sorry

end task1_on_time_task2_not_on_time_prob_l302_302497


namespace min_value_expression_is_4_l302_302872

noncomputable def min_value_expression (x : ℝ) : ℝ :=
(3 * x^2 + 6 * x + 5) / (0.5 * x^2 + x + 1)

theorem min_value_expression_is_4 : ∃ x : ℝ, min_value_expression x = 4 :=
sorry

end min_value_expression_is_4_l302_302872


namespace determine_x_l302_302658

theorem determine_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 5 * x - 7.5 = 0) → x = 3 / 2 :=
by
  intro h
  sorry

end determine_x_l302_302658


namespace mrs_franklin_gave_38_packs_l302_302841

-- Define the initial number of Valentines
def initial_valentines : Int := 450

-- Define the remaining Valentines after giving some away
def remaining_valentines : Int := 70

-- Define the size of each pack
def pack_size : Int := 10

-- Define the number of packs given away
def packs_given (initial remaining pack_size : Int) : Int :=
  (initial - remaining) / pack_size

theorem mrs_franklin_gave_38_packs :
  packs_given 450 70 10 = 38 := sorry

end mrs_franklin_gave_38_packs_l302_302841


namespace arithmetic_sequence_primes_l302_302985

theorem arithmetic_sequence_primes (a : ℕ) (d : ℕ) (primes_seq : ∀ n : ℕ, n < 15 → Nat.Prime (a + n * d))
  (distinct_primes : ∀ m n : ℕ, m < 15 → n < 15 → m ≠ n → a + m * d ≠ a + n * d) :
  d > 30000 := 
sorry

end arithmetic_sequence_primes_l302_302985


namespace arithmetic_seq_a4_l302_302927

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Define the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions and the goal to prove
theorem arithmetic_seq_a4 (h₁ : is_arithmetic_sequence a d) (h₂ : a 2 + a 6 = 10) : 
  a 4 = 5 :=
by
  sorry

end arithmetic_seq_a4_l302_302927


namespace total_wheels_in_parking_lot_l302_302572

theorem total_wheels_in_parking_lot :
  let cars := 5
  let trucks := 3
  let bikes := 2
  let three_wheelers := 4
  let wheels_per_car := 4
  let wheels_per_truck := 6
  let wheels_per_bike := 2
  let wheels_per_three_wheeler := 3
  (cars * wheels_per_car + trucks * wheels_per_truck + bikes * wheels_per_bike + three_wheelers * wheels_per_three_wheeler) = 54 := by
  sorry

end total_wheels_in_parking_lot_l302_302572


namespace ksyusha_wednesday_time_l302_302268

def speed_relation (v : ℝ) := ∀ (d_walk d_run1 d_run2 time1 time2: ℝ), 
  (2 * d_run1 = d_walk) → -- she walked twice the distance she ran on Tuesday
  (2 * v = 2 * v) → -- she runs twice as fast as she walks
  ((2 * d_walk / v) + (d_run1 / (2 * v)) = time1) → -- she took 30 minutes on Tuesday
  (time1 = 30) → -- Tuesday time equals 30
  (d_run2 = 2 * d_walk) → -- she had to run twice the distance she walked on Wednesday
  ((d_walk / v) + (d_run2 / (2 * v)) = time2) → -- calculate the time for Wednesday
  time2 = 24 -- time taken on Wednesday

theorem ksyusha_wednesday_time :
  ∃ (v : ℝ), speed_relation v :=
begin
  existsi 1,
  unfold speed_relation,
  intros,
  sorry,
end

end ksyusha_wednesday_time_l302_302268


namespace arithmetic_sequence_length_l302_302690

theorem arithmetic_sequence_length :
  ∀ (a₁ d an : ℤ), a₁ = -5 → d = 3 → an = 40 → (∃ n : ℕ, an = a₁ + (n - 1) * d ∧ n = 16) :=
by
  intros a₁ d an h₁ hd han
  sorry

end arithmetic_sequence_length_l302_302690


namespace arithmetic_geometric_sum_l302_302292

theorem arithmetic_geometric_sum {n : ℕ} (a : ℕ → ℤ) (S : ℕ → ℚ) 
  (h1 : ∀ k, a (k + 1) = a k + 2) 
  (h2 : (a 1) * (a 1 + a 4) = (a 1 + a 2) ^ 2 / 2) :
  S n = 6 - (4 * n + 6) / 2^n :=
by
  sorry

end arithmetic_geometric_sum_l302_302292


namespace quadratic_completing_the_square_q_l302_302339

theorem quadratic_completing_the_square_q (x p q : ℝ) (h : 4 * x^2 + 8 * x - 468 = 0) :
  (∃ p, (x + p)^2 = q) → q = 116 := sorry

end quadratic_completing_the_square_q_l302_302339


namespace students_average_age_l302_302613

theorem students_average_age (A : ℝ) (students_count teacher_age total_average new_count : ℝ) 
  (h1 : students_count = 30)
  (h2 : teacher_age = 45)
  (h3 : new_count = students_count + 1)
  (h4 : total_average = 15) 
  (h5 : total_average = (A * students_count + teacher_age) / new_count) : 
  A = 14 :=
by
  sorry

end students_average_age_l302_302613


namespace tangent_length_difference_l302_302246

open EuclideanGeometry

theorem tangent_length_difference (AB BC AF CG GF : ℝ) (AB_lt_BC : AB < BC)
  (E : Point) (circle_centered_at_E : Circle)
  (tangent_from_D : TangentLine)
  (tangent_at_AB : Point)
  (tangent_at_BC : Point)
  (intersection_from_D : SegmentIntersection tangent_from_D.1 tangent_at_AB)
  (tangent_intersects_G : SegmentIntersection tangent_from_D.1 tangent_at_BC):
  GF = AF - CG := by sorry

end tangent_length_difference_l302_302246


namespace sqrt_450_eq_15_sqrt_2_l302_302111

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302111


namespace Felipe_time_l302_302629

theorem Felipe_time (together_years : ℝ) (felipe_ratio : ℝ) : 
  together_years = 7.5 → felipe_ratio = 0.5 → 
  (F : ℝ), 12 * F = 30 :=
by
  intro h_together h_ratio
  let F := 2.5
  sorry

end Felipe_time_l302_302629


namespace ball_distribution_l302_302795

theorem ball_distribution (n : ℕ) (P_white P_red P_yellow : ℚ) (num_white num_red num_yellow : ℕ) 
  (total_balls : n = 6)
  (prob_white : P_white = 1/2)
  (prob_red : P_red = 1/3)
  (prob_yellow : P_yellow = 1/6) :
  num_white = 3 ∧ num_red = 2 ∧ num_yellow = 1 := 
sorry

end ball_distribution_l302_302795


namespace min_payment_max_payment_expected_value_payment_l302_302001

-- Proof Problem 1
theorem min_payment (prices : List ℕ) (H1 : prices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) :
  let optimized_groups := [[1000, 900, 800], [700, 600, 500], [400, 300, 200], [100]] in
  (∑ g in optimized_groups, ∑ l in (g.erase (g.min' (λ x y => x ≤ y))), l) = 4000 := by
  sorry

-- Proof Problem 2
theorem max_payment (prices : List ℕ) (H1 : prices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) :
  let suboptimized_groups := [[1000, 900, 100], [800, 700, 200], [600, 500, 300], [400]] in
  (∑ g in suboptimized_groups, ∑ l in (g.erase (g.min' (λ x y => x ≤ y))), l) = 4900 := by
  sorry

-- Proof Problem 3
theorem expected_value_payment (prices : List ℕ) (H1 : prices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) :
  let expected_savings := 100 * (∑ k in List.range 9, k * ((10 - k) * (9 - k)) / 72) in
  (5500 - expected_savings) = 4583.33 := by 
  sorry

end min_payment_max_payment_expected_value_payment_l302_302001


namespace bottle_caps_total_l302_302784

theorem bottle_caps_total (groups : ℕ) (bottle_caps_per_group : ℕ) (h1 : groups = 7) (h2 : bottle_caps_per_group = 5) : (groups * bottle_caps_per_group = 35) :=
by
  sorry

end bottle_caps_total_l302_302784


namespace sum_of_coeffs_eq_59049_l302_302306

-- Definition of the polynomial
def poly (x y z : ℕ) : ℕ :=
  (2 * x - 3 * y + 4 * z) ^ 10

-- Conjecture: The sum of the numerical coefficients in poly when x, y, and z are set to 1 is 59049
theorem sum_of_coeffs_eq_59049 : poly 1 1 1 = 59049 := by
  sorry

end sum_of_coeffs_eq_59049_l302_302306


namespace tetrahedron_altitudes_l302_302489

theorem tetrahedron_altitudes (r h₁ h₂ h₃ h₄ : ℝ)
  (h₁_def : h₁ = 3 * r)
  (h₂_def : h₂ = 4 * r)
  (h₃_def : h₃ = 4 * r)
  (altitude_sum : 1/h₁ + 1/h₂ + 1/h₃ + 1/h₄ = 1/r) : 
  h₄ = 6 * r :=
by
  rw [h₁_def, h₂_def, h₃_def] at altitude_sum
  sorry

end tetrahedron_altitudes_l302_302489


namespace problem_l302_302397

theorem problem (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by sorry

end problem_l302_302397


namespace sqrt_450_eq_15_sqrt_2_l302_302129

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302129


namespace digitalEarthFunctions_l302_302183

axiom OptionA (F : Type) : Prop
axiom OptionB (F : Type) : Prop
axiom OptionC (F : Type) : Prop
axiom OptionD (F : Type) : Prop

axiom isRemoteSensing (F : Type) : OptionA F
axiom isGIS (F : Type) : OptionB F
axiom isGPS (F : Type) : OptionD F

theorem digitalEarthFunctions {F : Type} : OptionC F :=
sorry

end digitalEarthFunctions_l302_302183


namespace ratio_triangle_square_l302_302574

noncomputable def square_area (s : ℝ) : ℝ := s * s

noncomputable def triangle_PTU_area (s : ℝ) : ℝ := 1 / 2 * (s / 2) * (s / 2)

theorem ratio_triangle_square (s : ℝ) (h : s > 0) : 
  triangle_PTU_area s / square_area s = 1 / 8 := 
sorry

end ratio_triangle_square_l302_302574


namespace matrix_power_four_l302_302194

def A : Matrix (Fin 2) (Fin 2) ℝ :=
  !![2, -Real.sqrt 3; Real.sqrt 3, 2]

theorem matrix_power_four :
  A ^ 4 = !![
    -49 / 2, -49 * Real.sqrt 3 / 2;
    49 * Real.sqrt 3 / 2, -49 / 2
  ] :=
by
  sorry

end matrix_power_four_l302_302194


namespace solve_Mary_height_l302_302839

theorem solve_Mary_height :
  ∃ (m s : ℝ), 
  s = 150 ∧ 
  s * 1.2 = 180 ∧ 
  m = s + (180 - s) / 2 ∧ 
  m = 165 :=
by
  sorry

end solve_Mary_height_l302_302839


namespace sqrt_450_eq_15_sqrt_2_l302_302130

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302130


namespace small_glass_cost_l302_302845

theorem small_glass_cost 
  (S : ℝ)
  (small_glass_cost : ℝ)
  (large_glass_cost : ℝ := 5)
  (initial_money : ℝ := 50)
  (num_small : ℝ := 8)
  (change : ℝ := 1)
  (num_large : ℝ := 5)
  (spent_money : ℝ := initial_money - change)
  (total_large_cost : ℝ := num_large * large_glass_cost)
  (total_cost : ℝ := num_small * S + total_large_cost)
  (total_cost_eq : total_cost = spent_money) :
  S = 3 :=
by
  sorry

end small_glass_cost_l302_302845


namespace juniors_more_than_seniors_l302_302333

theorem juniors_more_than_seniors
  (j s : ℕ)
  (h1 : (1 / 3) * j = (2 / 3) * s)
  (h2 : j + s = 300) :
  j - s = 100 := 
sorry

end juniors_more_than_seniors_l302_302333


namespace percentage_differences_equal_l302_302757

noncomputable def calculation1 : ℝ := 0.60 * 50
noncomputable def calculation2 : ℝ := 0.30 * 30
noncomputable def calculation3 : ℝ := 0.45 * 90
noncomputable def calculation4 : ℝ := 0.20 * 40

noncomputable def diff1 : ℝ := abs (calculation1 - calculation2)
noncomputable def diff2 : ℝ := abs (calculation2 - calculation3)
noncomputable def diff3 : ℝ := abs (calculation3 - calculation4)
noncomputable def largest_diff1 : ℝ := max diff1 (max diff2 diff3)

noncomputable def calculation5 : ℝ := 0.40 * 120
noncomputable def calculation6 : ℝ := 0.25 * 80
noncomputable def calculation7 : ℝ := 0.35 * 150
noncomputable def calculation8 : ℝ := 0.55 * 60

noncomputable def diff4 : ℝ := abs (calculation5 - calculation6)
noncomputable def diff5 : ℝ := abs (calculation6 - calculation7)
noncomputable def diff6 : ℝ := abs (calculation7 - calculation8)
noncomputable def largest_diff2 : ℝ := max diff4 (max diff5 diff6)

theorem percentage_differences_equal :
  largest_diff1 = largest_diff2 :=
sorry

end percentage_differences_equal_l302_302757


namespace Luka_water_requirement_l302_302588

-- Declare variables and conditions
variables (L S W O : ℕ)  -- All variables are natural numbers
-- Conditions
variable (h1 : S = 2 * L)  -- Twice as much sugar as lemon juice
variable (h2 : W = 5 * S)  -- 5 times as much water as sugar
variable (h3 : O = S)      -- Orange juice equals the amount of sugar 
variable (L_eq_5 : L = 5)  -- Lemon juice is 5 cups

-- The goal statement to prove
theorem Luka_water_requirement : W = 50 :=
by
  -- Note: The proof steps would go here, but as per instructions, we leave it as sorry.
  sorry

end Luka_water_requirement_l302_302588


namespace negation_of_exists_l302_302553

open Classical

theorem negation_of_exists (p : Prop) : 
  (∃ x : ℝ, 2^x ≥ 2 * x + 1) ↔ ¬ ∀ x : ℝ, 2^x < 2 * x + 1 :=
by
  sorry

end negation_of_exists_l302_302553


namespace friends_meeting_games_only_l302_302883

theorem friends_meeting_games_only 
  (M P G MP MG PG MPG : ℕ) 
  (h1 : M + MP + MG + MPG = 10) 
  (h2 : P + MP + PG + MPG = 20) 
  (h3 : MP = 4) 
  (h4 : MG = 2) 
  (h5 : PG = 0) 
  (h6 : MPG = 2) 
  (h7 : M + P + G + MP + MG + PG + MPG = 31) : 
  G = 1 := 
by
  sorry

end friends_meeting_games_only_l302_302883


namespace fraction_value_l302_302424

theorem fraction_value (a b c d : ℕ)
  (h₁ : a = 4 * b)
  (h₂ : b = 3 * c)
  (h₃ : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end fraction_value_l302_302424


namespace sqrt_simplify_l302_302029

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l302_302029


namespace find_value_of_expression_l302_302937

theorem find_value_of_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) : x^2 + 2*x + 3 = 4 := by
  sorry

end find_value_of_expression_l302_302937


namespace unique_two_digit_solution_l302_302484

theorem unique_two_digit_solution:
  ∃! (s : ℕ), 10 ≤ s ∧ s ≤ 99 ∧ (13 * s ≡ 52 [MOD 100]) :=
  sorry

end unique_two_digit_solution_l302_302484


namespace find_initial_candies_l302_302192

-- Define the initial number of candies as x
def initial_candies (x : ℕ) : ℕ :=
  let first_day := (3 * x) / 4 - 3
  let second_day := (3 * first_day) / 5 - 5
  let third_day := second_day - 7
  let final_candies := (5 * third_day) / 6
  final_candies

-- Formal statement of the theorem
theorem find_initial_candies (x : ℕ) (h : initial_candies x = 10) : x = 44 :=
  sorry

end find_initial_candies_l302_302192


namespace total_pencils_is_220_l302_302467

theorem total_pencils_is_220
  (A : ℕ) (B : ℕ) (P : ℕ) (Q : ℕ)
  (hA : A = 50)
  (h_sum : A + B = 140)
  (h_diff : B - A = P/2)
  (h_pencils : Q = P + 60)
  : P + Q = 220 :=
by
  sorry

end total_pencils_is_220_l302_302467


namespace sqrt_450_eq_15_sqrt_2_l302_302070

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l302_302070


namespace x1_x2_in_M_l302_302684

-- Definitions of the set M and the condition x ∈ M
def M : Set ℕ := { x | ∃ a b : ℤ, x = a^2 + b^2 }

-- Statement of the problem
theorem x1_x2_in_M (x1 x2 : ℕ) (h1 : x1 ∈ M) (h2 : x2 ∈ M) : (x1 * x2) ∈ M :=
sorry

end x1_x2_in_M_l302_302684


namespace value_of_expression_l302_302542

-- Given conditions as definitions
axiom cond1 (x y : ℝ) : -x + 2*y = 5

-- The theorem we want to prove
theorem value_of_expression (x y : ℝ) (h : -x + 2*y = 5) : 
  5 * (x - 2 * y)^2 - 3 * (x - 2 * y) - 60 = 80 :=
by
  -- The proof part is omitted here.
  sorry

end value_of_expression_l302_302542


namespace total_water_filled_jars_l302_302706

theorem total_water_filled_jars (x : ℕ) (h : 4 * x + 2 * x + x = 14 * 4) : 3 * x = 24 :=
by
  sorry

end total_water_filled_jars_l302_302706


namespace value_of_fraction_l302_302430

-- Definitions based on conditions
variables {a b c d : ℝ}
hypothesis h1 : a = 4 * b
hypothesis h2 : b = 3 * c
hypothesis h3 : c = 5 * d

-- The statement we want to prove
theorem value_of_fraction : (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_fraction_l302_302430


namespace no_such_functions_l302_302848

open Real

theorem no_such_functions : ¬ ∃ (f g : ℝ → ℝ), (∀ x y : ℝ, f (x^2 + g y) - f (x^2) + g (y) - g (x) ≤ 2 * y) ∧ (∀ x : ℝ, f (x) ≥ x^2) := by
  sorry

end no_such_functions_l302_302848


namespace cliff_rock_collection_l302_302311

theorem cliff_rock_collection (S I : ℕ) 
  (h1 : I = S / 2) 
  (h2 : 2 * I / 3 = 40) : S + I = 180 := by
  sorry

end cliff_rock_collection_l302_302311


namespace find_real_solutions_l302_302798

variable (x : ℝ)

theorem find_real_solutions :
  (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 8) ↔ 
  (x = -2 * Real.sqrt 14 ∨ x = 2 * Real.sqrt 14) := 
sorry

end find_real_solutions_l302_302798


namespace distance_karen_covers_l302_302147

theorem distance_karen_covers
  (books_per_shelf : ℕ)
  (shelves : ℕ)
  (distance_to_library : ℕ)
  (h1 : books_per_shelf = 400)
  (h2 : shelves = 4)
  (h3 : distance_to_library = books_per_shelf * shelves) :
  2 * distance_to_library = 3200 := 
by
  sorry

end distance_karen_covers_l302_302147


namespace john_took_more_chickens_l302_302605

theorem john_took_more_chickens (john_chickens mary_chickens ray_chickens : ℕ)
  (h_ray : ray_chickens = 10)
  (h_mary : mary_chickens = ray_chickens + 6)
  (h_john : john_chickens = mary_chickens + 5) :
  john_chickens - ray_chickens = 11 := by
  sorry

end john_took_more_chickens_l302_302605


namespace simplify_sqrt_450_l302_302008

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l302_302008


namespace marcel_potatoes_eq_l302_302720

-- Define the given conditions
def marcel_corn := 10
def dale_corn := marcel_corn / 2
def dale_potatoes := 8
def total_vegetables := 27

-- Define the fact that they bought 27 vegetables in total
def total_corn := marcel_corn + dale_corn
def total_potatoes := total_vegetables - total_corn

-- State the theorem
theorem marcel_potatoes_eq :
  (total_potatoes - dale_potatoes) = 4 :=
by
  -- Lean proof would go here
  sorry

end marcel_potatoes_eq_l302_302720


namespace boat_speed_in_still_water_l302_302700

/-- In one hour, a boat goes 9 km along the stream and 5 km against the stream.
Prove that the speed of the boat in still water is 7 km/hr. -/
theorem boat_speed_in_still_water (B S : ℝ) 
  (h1 : B + S = 9) 
  (h2 : B - S = 5) : 
  B = 7 :=
by
  sorry

end boat_speed_in_still_water_l302_302700


namespace greatest_y_l302_302290

theorem greatest_y (x y : ℤ) (h : x * y + 6 * x + 5 * y = -6) : y ≤ 24 :=
sorry

end greatest_y_l302_302290


namespace prob1_prob2_prob3_l302_302226

noncomputable def f (x : ℝ) : ℝ :=
  if h : x ≥ 0 then x^2 + 2
  else x

theorem prob1 :
  (∀ x, x ≥ 0 → f x = x^2 + 2) ∧
  (∀ x, x < 0 → f x = x) :=
by
  sorry

theorem prob2 : f 5 = 27 :=
by 
  sorry

theorem prob3 : ∀ (x : ℝ), f x = 0 → false :=
by
  sorry

end prob1_prob2_prob3_l302_302226


namespace f_prime_at_zero_l302_302577

-- Lean definition of the conditions.
def a (n : ℕ) : ℝ := 2 * (2 ^ (1/7)) ^ (n - 1)

-- The function f(x) based on the given conditions.
noncomputable def f (x : ℝ) : ℝ := 
  x * (x - a 1) * (x - a 2) * (x - a 3) * (x - a 4) * 
  (x - a 5) * (x - a 6) * (x - a 7) * (x - a 8)

-- The main goal to prove: f'(0) = 2^12
theorem f_prime_at_zero : deriv f 0 = 2^12 := by
  sorry

end f_prime_at_zero_l302_302577


namespace area_common_to_all_four_circles_l302_302865

noncomputable def common_area (R : ℝ) : ℝ :=
  (R^2 * (2 * Real.pi - 3 * Real.sqrt 3)) / 6

theorem area_common_to_all_four_circles (R : ℝ) :
  ∃ (O1 O2 A B : ℝ × ℝ),
    dist O1 O2 = R ∧
    dist O1 A = R ∧
    dist O2 A = R ∧
    dist O1 B = R ∧
    dist O2 B = R ∧
    dist A B = R ∧
    common_area R = (R^2 * (2 * Real.pi - 3 * Real.sqrt 3)) / 6 :=
by
  sorry

end area_common_to_all_four_circles_l302_302865


namespace compare_neg_fractions_l302_302902

theorem compare_neg_fractions : - (2 / 3 : ℝ) > - (3 / 4 : ℝ) :=
sorry

end compare_neg_fractions_l302_302902


namespace volume_ratio_l302_302868

def cube_volume (side_length : ℝ) : ℝ :=
  side_length ^ 3

theorem volume_ratio : 
  let a := (4 : ℝ) / 12   -- 4 inches converted to feet
  let b := (2 : ℝ)       -- 2 feet
  cube_volume a / cube_volume b = 1 / 216 :=
by
  sorry

end volume_ratio_l302_302868


namespace percentage_increase_is_50_l302_302884

def initial : ℝ := 110
def final : ℝ := 165

theorem percentage_increase_is_50 :
  ((final - initial) / initial) * 100 = 50 := by
  sorry

end percentage_increase_is_50_l302_302884


namespace carl_wins_in_4950_configurations_l302_302836

noncomputable def num_distinct_configurations_at_Carl_win : ℕ :=
  sorry
  
theorem carl_wins_in_4950_configurations :
  num_distinct_configurations_at_Carl_win = 4950 :=
sorry

end carl_wins_in_4950_configurations_l302_302836


namespace distinct_paths_from_C_to_D_l302_302820

-- Definitions based on conditions
def grid_rows : ℕ := 7
def grid_columns : ℕ := 8
def total_steps : ℕ := grid_rows + grid_columns -- 15 in this case
def steps_right : ℕ := grid_columns -- 8 in this case

-- Theorem statement
theorem distinct_paths_from_C_to_D :
  Nat.choose total_steps steps_right = 6435 :=
by
  -- The proof itself
  sorry

end distinct_paths_from_C_to_D_l302_302820


namespace value_of_f1_plus_g3_l302_302715

def f (x : ℝ) := 3 * x - 4
def g (x : ℝ) := x + 2

theorem value_of_f1_plus_g3 : f (1 + g 3) = 14 := by
  sorry

end value_of_f1_plus_g3_l302_302715


namespace polynomial_coefficients_correct_l302_302825

-- Define the polynomial equation
def polynomial_equation (x a b c d : ℝ) : Prop :=
  x^4 + 4*x^3 + 3*x^2 + 2*x + 1 = (x+1)^4 + a*(x+1)^3 + b*(x+1)^2 + c*(x+1) + d

-- The problem to prove
theorem polynomial_coefficients_correct :
  ∀ x : ℝ, polynomial_equation x 0 (-3) 4 (-1) :=
by
  intro x
  unfold polynomial_equation
  sorry

end polynomial_coefficients_correct_l302_302825


namespace sqrt_450_eq_15_sqrt_2_l302_302128

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302128


namespace sqrt_450_equals_15_sqrt_2_l302_302082

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l302_302082


namespace range_of_a_l302_302375

open Function

def f (x : ℝ) : ℝ := -2 * x^5 - x^3 - 7 * x + 2

theorem range_of_a (a : ℝ) : f (a^2) + f (a - 2) > 4 → -2 < a ∧ a < 1 := 
by
  sorry

end range_of_a_l302_302375


namespace sqrt_simplify_l302_302030

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l302_302030


namespace sqrt_450_equals_15_sqrt_2_l302_302083

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l302_302083


namespace value_of_fraction_l302_302427

-- Definitions based on conditions
variables {a b c d : ℝ}
hypothesis h1 : a = 4 * b
hypothesis h2 : b = 3 * c
hypothesis h3 : c = 5 * d

-- The statement we want to prove
theorem value_of_fraction : (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_fraction_l302_302427


namespace f_minus_5_eq_12_l302_302377

def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem f_minus_5_eq_12 : f (-5) = 12 := 
by sorry

end f_minus_5_eq_12_l302_302377


namespace not_washed_shirts_l302_302645

-- Definitions based on given conditions
def short_sleeve_shirts : ℕ := 9
def long_sleeve_shirts : ℕ := 21
def washed_shirts : ℕ := 29

-- Theorem to prove the number of shirts not washed
theorem not_washed_shirts : (short_sleeve_shirts + long_sleeve_shirts) - washed_shirts = 1 := by
  sorry

end not_washed_shirts_l302_302645


namespace product_of_two_numbers_l302_302866

-- Definitions and conditions
def HCF (a b : ℕ) : ℕ := 9
def LCM (a b : ℕ) : ℕ := 200

-- Theorem statement
theorem product_of_two_numbers (a b : ℕ) (H₁ : HCF a b = 9) (H₂ : LCM a b = 200) : a * b = 1800 :=
by
  -- Injecting HCF and LCM conditions into the problem
  sorry

end product_of_two_numbers_l302_302866


namespace problem_l302_302392

theorem problem (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by sorry

end problem_l302_302392


namespace total_wages_l302_302171

-- Definitions and conditions
def A_one_day_work : ℚ := 1 / 10
def B_one_day_work : ℚ := 1 / 15
def A_share_wages : ℚ := 2040

-- Stating the problem in Lean
theorem total_wages (X : ℚ) : (3 / 5) * X = A_share_wages → X = 3400 := 
  by 
  sorry

end total_wages_l302_302171


namespace east_high_school_students_l302_302621

theorem east_high_school_students (S : ℝ) 
  (h1 : 0.52 * S * 0.125 = 26) :
  S = 400 :=
by
  -- The proof is omitted for this exercise
  sorry

end east_high_school_students_l302_302621


namespace reduce_expression_l302_302229

-- Define the variables a, b, c as real numbers
variables (a b c : ℝ)

-- State the theorem with the given condition that expressions are defined and non-zero
theorem reduce_expression :
  (a^2 + b^2 - c^2 - 2*a*b) / (a^2 + c^2 - b^2 - 2*a*c) = ((a - b + c) * (a - b - c)) / ((a - c + b) * (a - c - b)) :=
by
  sorry

end reduce_expression_l302_302229


namespace total_sales_correct_l302_302988

def normal_sales_per_month : ℕ := 21122
def additional_sales_in_june : ℕ := 3922
def sales_in_june : ℕ := normal_sales_per_month + additional_sales_in_june
def sales_in_july : ℕ := normal_sales_per_month
def total_sales : ℕ := sales_in_june + sales_in_july

theorem total_sales_correct :
  total_sales = 46166 :=
by
  -- Proof goes here
  sorry

end total_sales_correct_l302_302988


namespace problem_statement_l302_302974

theorem problem_statement (x y : ℝ) (h1 : y ≥ 0) (h2 : y * (y + 1) ≤ (x + 1)^2) : y * (y - 1) ≤ x^2 := 
sorry

end problem_statement_l302_302974


namespace only_positive_odd_integer_dividing_3n_plus_1_l302_302349

theorem only_positive_odd_integer_dividing_3n_plus_1 : 
  ∀ (n : ℕ), (0 < n) → (n % 2 = 1) → (n ∣ (3 ^ n + 1)) → n = 1 := by
  sorry

end only_positive_odd_integer_dividing_3n_plus_1_l302_302349


namespace value_of_frac_l302_302387

theorem value_of_frac (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_frac_l302_302387


namespace even_function_exists_l302_302971

def f (x m : ℝ) : ℝ := x^2 + m * x

theorem even_function_exists : ∃ m : ℝ, ∀ x : ℝ, f x m = f (-x) m :=
by
  use 0
  intros x
  unfold f
  simp

end even_function_exists_l302_302971


namespace maximum_value_quadratic_l302_302309

def quadratic_function (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

theorem maximum_value_quadratic :
  ∃ x : ℝ, quadratic_function x = 2 ∧ ∀ y : ℝ, quadratic_function y ≤ 2 :=
sorry

end maximum_value_quadratic_l302_302309


namespace value_of_expression_l302_302401

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := by
  sorry

end value_of_expression_l302_302401


namespace range_of_x_l302_302949

theorem range_of_x (x : ℝ) (h1 : 1 / x < 4) (h2 : 1 / x > -2) : x > -(1 / 2) :=
sorry

end range_of_x_l302_302949


namespace simplify_sqrt_450_l302_302018
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l302_302018


namespace find_t_l302_302583

theorem find_t (t : ℝ) :
  let P := (t - 5, -2)
  let Q := (-3, t + 4)
  let M := ((t - 8) / 2, (t + 2) / 2)
  (dist M P) ^ 2 = t^2 / 3 →
  t = -12 + 2 * Real.sqrt 21 ∨ t = -12 - 2 * Real.sqrt 21 := sorry

end find_t_l302_302583


namespace unique_two_digit_solution_l302_302485

theorem unique_two_digit_solution:
  ∃! (s : ℕ), 10 ≤ s ∧ s ≤ 99 ∧ (13 * s ≡ 52 [MOD 100]) :=
  sorry

end unique_two_digit_solution_l302_302485


namespace Ksyusha_travel_time_l302_302261

theorem Ksyusha_travel_time:
  ∀ (S v : ℝ), 
  (S > 0 ∧ v > 0) →
  (∀ tuesday_distance tuesday_time wednesday_distance wednesday_time: ℝ, 
    tuesday_distance = 3 * S ∧ 
    tuesday_distance = 2 * S + S ∧ 
    tuesday_time = (2 * S) / v + (S / (2 * v)) ∧ 
    tuesday_time = 30) → 
  (∀ wednesday_distance wednesday_time : ℝ, 
    wednesday_distance = S + 2 * S ∧ 
    wednesday_time = S / v + (2 * S) / (2 * v)) →
  wednesday_time = 24 :=
by
  intros S v S_pos v_pos tuesday_distance tuesday_time wednesday_distance wednesday_time
  sorry

end Ksyusha_travel_time_l302_302261


namespace johns_elevation_after_descent_l302_302708

def starting_elevation : ℝ := 400
def rate_of_descent : ℝ := 10
def travel_time : ℝ := 5

theorem johns_elevation_after_descent :
  starting_elevation - (rate_of_descent * travel_time) = 350 :=
by
  sorry

end johns_elevation_after_descent_l302_302708


namespace postcard_cost_l302_302842

theorem postcard_cost (x : ℕ) (h₁ : 9 * x < 1000) (h₂ : 10 * x > 1100) : x = 111 :=
by
  sorry

end postcard_cost_l302_302842


namespace difference_sum_even_odd_1000_l302_302758

open Nat

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

def sum_first_n_even (n : ℕ) : ℕ :=
  n * (n + 1)

theorem difference_sum_even_odd_1000 :
  sum_first_n_even 1000 - sum_first_n_odd 1000 = 1000 :=
by
  sorry

end difference_sum_even_odd_1000_l302_302758


namespace lucy_total_fish_l302_302452

variable (current_fish additional_fish : ℕ)

def total_fish (current_fish additional_fish : ℕ) : ℕ :=
  current_fish + additional_fish

theorem lucy_total_fish (h1 : current_fish = 212) (h2 : additional_fish = 68) : total_fish current_fish additional_fish = 280 :=
by
  sorry

end lucy_total_fish_l302_302452


namespace expand_binomials_l302_302346

theorem expand_binomials (x : ℝ) : 
  (1 + x + x^3) * (1 - x^4) = 1 + x + x^3 - x^4 - x^5 - x^7 :=
by
  sorry

end expand_binomials_l302_302346


namespace bananas_in_each_box_l302_302447

theorem bananas_in_each_box 
    (bananas : ℕ) (boxes : ℕ) 
    (h_bananas : bananas = 40) 
    (h_boxes : boxes = 10) : 
    bananas / boxes = 4 := by
  sorry

end bananas_in_each_box_l302_302447


namespace remainder_of_789987_div_8_l302_302156

theorem remainder_of_789987_div_8 : (789987 % 8) = 3 := by
  sorry

end remainder_of_789987_div_8_l302_302156


namespace draw_sequence_count_l302_302863

noncomputable def total_sequences : ℕ :=
  (Nat.choose 4 3) * (Nat.factorial 4) * 5

theorem draw_sequence_count : total_sequences = 480 := by
  sorry

end draw_sequence_count_l302_302863


namespace value_of_frac_l302_302388

theorem value_of_frac (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_frac_l302_302388


namespace simplify_sqrt_450_l302_302012

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l302_302012


namespace residue_12_2040_mod_19_l302_302492

theorem residue_12_2040_mod_19 :
  12^2040 % 19 = 7 := 
sorry

end residue_12_2040_mod_19_l302_302492


namespace prove_d_value_l302_302212

-- Definitions of the conditions
def d (x : ℝ) : ℝ := x^4 - 2*x^3 + x^2 - 12*x - 5

-- The statement to prove
theorem prove_d_value (x : ℝ) (h : x^2 - 2*x - 5 = 0) : d x = 25 :=
sorry

end prove_d_value_l302_302212


namespace range_of_a_l302_302809

-- Let us define the problem conditions and statement in Lean
theorem range_of_a
  (a : ℝ)
  (h : ∀ x y : ℝ, x < y → (3 - a)^x > (3 - a)^y) :
  2 < a ∧ a < 3 :=
sorry

end range_of_a_l302_302809


namespace John_took_more_chickens_than_Ray_l302_302610

theorem John_took_more_chickens_than_Ray
  (r m j : ℕ)
  (h1 : r = 10)
  (h2 : r = m - 6)
  (h3 : j = m + 5) : j - r = 11 :=
by
  sorry

end John_took_more_chickens_than_Ray_l302_302610


namespace value_of_fraction_l302_302431

-- Definitions based on conditions
variables {a b c d : ℝ}
hypothesis h1 : a = 4 * b
hypothesis h2 : b = 3 * c
hypothesis h3 : c = 5 * d

-- The statement we want to prove
theorem value_of_fraction : (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_fraction_l302_302431


namespace bank_teller_bills_l302_302503

theorem bank_teller_bills (x y : ℕ) (h1 : x + y = 54) (h2 : 5 * x + 20 * y = 780) : x = 20 :=
by
  sorry

end bank_teller_bills_l302_302503


namespace sqrt_450_equals_15_sqrt_2_l302_302078

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l302_302078


namespace unique_two_digit_solution_exists_l302_302483

theorem unique_two_digit_solution_exists :
  ∃! (s : ℤ), 10 ≤ s ∧ s < 100 ∧ 13 * s % 100 = 52 :=
begin
  use 4,
  split,
  { split,
    { linarith },
    { linarith },
    { norm_num }
  },
  { intros y hy,
    cases hy with hy1 hy2,
    cases hy2 with hy2 hy3,
    have : 77 * 52 % 100 = 4,
    { norm_num },
    have h : y ≡ 4 [MOD 100] := (congr_arg (λ x, 77 * x % 100) hy3).trans this,
    norm_num at h,
    linarith }
end

end unique_two_digit_solution_exists_l302_302483


namespace simplify_sqrt_450_l302_302016
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l302_302016


namespace minimum_value_reciprocals_l302_302294

theorem minimum_value_reciprocals (a b : Real) (h1 : a > 0) (h2 : b > 0) (h3 : 2 / Real.sqrt (a^2 + 4 * b^2) = Real.sqrt 2) :
  (1 / a^2 + 1 / b^2) = 9 / 2 :=
sorry

end minimum_value_reciprocals_l302_302294


namespace distance_equal_axes_l302_302614

theorem distance_equal_axes (m : ℝ) :
  (abs (3 * m + 1) = abs (2 * m - 5)) ↔ (m = -6 ∨ m = 4 / 5) :=
by 
  sorry

end distance_equal_axes_l302_302614


namespace find_inverse_l302_302278

theorem find_inverse :
  ∀ (f : ℝ → ℝ), (∀ x, f x = 3 * x ^ 3 + 9) → (f⁻¹ 90 = 3) :=
by
  intros f h
  sorry

end find_inverse_l302_302278


namespace sin_alpha_two_alpha_plus_beta_l302_302370

variable {α β : ℝ}
variable (h₁ : 0 < α ∧ α < π / 2)
variable (h₂ : 0 < β ∧ β < π / 2)
variable (h₃ : Real.tan (α / 2) = 1 / 3)
variable (h₄ : Real.cos (α - β) = -4 / 5)

theorem sin_alpha (h₁ : 0 < α ∧ α < π / 2)
                  (h₃ : Real.tan (α / 2) = 1 / 3) :
                  Real.sin α = 3 / 5 :=
by
  sorry

theorem two_alpha_plus_beta (h₁ : 0 < α ∧ α < π / 2)
                            (h₂ : 0 < β ∧ β < π / 2)
                            (h₄ : Real.cos (α - β) = -4 / 5) :
                            2 * α + β = π :=
by
  sorry

end sin_alpha_two_alpha_plus_beta_l302_302370


namespace quadratic_inequality_solution_set_l302_302477

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2 * x - 3 < 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end quadratic_inequality_solution_set_l302_302477


namespace jackson_spends_on_school_supplies_l302_302703

theorem jackson_spends_on_school_supplies :
  let num_students := 50
  let pens_per_student := 7
  let notebooks_per_student := 5
  let binders_per_student := 3
  let highlighters_per_student := 4
  let folders_per_student := 2
  let cost_pen := 0.70
  let cost_notebook := 1.60
  let cost_binder := 5.10
  let cost_highlighter := 0.90
  let cost_folder := 1.15
  let teacher_discount := 135
  let bulk_discount := 25
  let sales_tax_rate := 0.05
  let total_cost := 
    (num_students * pens_per_student * cost_pen) + 
    (num_students * notebooks_per_student * cost_notebook) + 
    (num_students * binders_per_student * cost_binder) + 
    (num_students * highlighters_per_student * cost_highlighter) + 
    (num_students * folders_per_student * cost_folder)
  let discounted_cost := total_cost - teacher_discount - bulk_discount
  let sales_tax := discounted_cost * sales_tax_rate
  let final_cost := discounted_cost + sales_tax
  final_cost = 1622.25 := by
  sorry

end jackson_spends_on_school_supplies_l302_302703


namespace fraction_value_l302_302426

theorem fraction_value (a b c d : ℕ)
  (h₁ : a = 4 * b)
  (h₂ : b = 3 * c)
  (h₃ : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end fraction_value_l302_302426


namespace number_of_solutions_l302_302233

theorem number_of_solutions : ∃ n : ℕ, 1 < n ∧ 
  (∃ a b : ℕ, gcd a b = 1 ∧
  (∃ x y : ℕ, x^(a*n) + y^(b*n) = 2^2010)) ∧
  (∃ count : ℕ, count = 54) :=
sorry

end number_of_solutions_l302_302233


namespace fraction_value_l302_302420

theorem fraction_value (a b c d : ℕ)
  (h₁ : a = 4 * b)
  (h₂ : b = 3 * c)
  (h₃ : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end fraction_value_l302_302420


namespace ratio_expression_value_l302_302384

variable {A B C : ℚ}

theorem ratio_expression_value (h : A / B = 3 / 2 ∧ A / C = 3 / 6) : (4 * A - 3 * B) / (5 * C + 2 * A) = 1 / 4 := 
sorry

end ratio_expression_value_l302_302384


namespace sum_of_multiples_of_6_and_9_is_multiple_of_3_l302_302600

theorem sum_of_multiples_of_6_and_9_is_multiple_of_3 
  (x y : ℤ) (hx : ∃ m : ℤ, x = 6 * m) (hy : ∃ n : ℤ, y = 9 * n) : 
  ∃ k : ℤ, x + y = 3 * k := 
by 
  sorry

end sum_of_multiples_of_6_and_9_is_multiple_of_3_l302_302600


namespace sqrt_17_estimation_l302_302535

theorem sqrt_17_estimation :
  4 < Real.sqrt 17 ∧ Real.sqrt 17 < 5 := 
sorry

end sqrt_17_estimation_l302_302535


namespace equivalent_fraction_l302_302655

theorem equivalent_fraction :
  (6 + 6 + 6 + 6) / ((-2) * (-2) * (-2) * (-2)) = (4 * 6) / ((-2)^4) :=
by 
  sorry

end equivalent_fraction_l302_302655


namespace triangular_prism_sliced_faces_l302_302510

noncomputable def resulting_faces_count : ℕ :=
  let initial_faces := 5 -- 2 bases + 3 lateral faces
  let additional_faces := 3 -- from the slices
  initial_faces + additional_faces

theorem triangular_prism_sliced_faces :
  resulting_faces_count = 8 := by
  sorry

end triangular_prism_sliced_faces_l302_302510


namespace find_n_l302_302582

noncomputable def C (n : ℕ) : ℝ :=
  352 * (1 - 1 / 2 ^ n) / (1 - 1 / 2)

noncomputable def D (n : ℕ) : ℝ :=
  992 * (1 - 1 / (-2) ^ n) / (1 + 1 / 2)

theorem find_n (n : ℕ) (h : 1 ≤ n) : C n = D n ↔ n = 1 := by
  sorry

end find_n_l302_302582


namespace simplify_sqrt_450_l302_302013
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l302_302013


namespace additional_money_earned_l302_302174

-- Define the conditions as variables
def price_duck : ℕ := 10
def price_chicken : ℕ := 8
def num_chickens_sold : ℕ := 5
def num_ducks_sold : ℕ := 2
def half (x : ℕ) : ℕ := x / 2
def double (x : ℕ) : ℕ := 2 * x

-- Define the calculations based on the conditions
def earnings_chickens : ℕ := num_chickens_sold * price_chicken 
def earnings_ducks : ℕ := num_ducks_sold * price_duck 
def total_earnings : ℕ := earnings_chickens + earnings_ducks 
def cost_wheelbarrow : ℕ := half total_earnings
def selling_price_wheelbarrow : ℕ := double cost_wheelbarrow
def additional_earnings : ℕ := selling_price_wheelbarrow - cost_wheelbarrow

-- The theorem to prove the correct additional earnings
theorem additional_money_earned : additional_earnings = 30 := by
  sorry

end additional_money_earned_l302_302174


namespace solve_exponential_problem_l302_302858

noncomputable def satisfies_condition (a : ℝ) : Prop :=
  let max_value := if a > 1 then a^2 else a
  let min_value := if a > 1 then a else a^2
  max_value - min_value = a / 2

theorem solve_exponential_problem (a : ℝ) (hpos : a > 0) (hne1 : a ≠ 1) :
  satisfies_condition a ↔ (a = 1 / 2 ∨ a = 3 / 2) :=
sorry

end solve_exponential_problem_l302_302858


namespace factor_of_increase_l302_302780

-- Define the conditions
def interest_rate : ℝ := 0.25
def time_period : ℕ := 4

-- Define the principal amount as a variable
variable (P : ℝ)

-- Define the simple interest formula
def simple_interest (P : ℝ) (R : ℝ) (T : ℕ) : ℝ := P * R * (T : ℝ)

-- Define the total amount function
def total_amount (P : ℝ) (SI : ℝ) : ℝ := P + SI

-- The theorem that we need to prove: The factor by which the sum of money increases is 2
theorem factor_of_increase :
  total_amount P (simple_interest P interest_rate time_period) = 2 * P := by
  sorry

end factor_of_increase_l302_302780


namespace gcd_2720_1530_l302_302739

theorem gcd_2720_1530 : Nat.gcd 2720 1530 = 170 := by
  sorry

end gcd_2720_1530_l302_302739


namespace compute_expression_l302_302903

theorem compute_expression :
  24 * 42 + 58 * 24 + 12 * 24 = 2688 := by
  sorry

end compute_expression_l302_302903


namespace smallest_x_for_M_cube_l302_302743

theorem smallest_x_for_M_cube (x M : ℤ) (h1 : 1890 * x = M^3) : x = 4900 :=
sorry

end smallest_x_for_M_cube_l302_302743


namespace fraction_value_l302_302422

theorem fraction_value (a b c d : ℕ)
  (h₁ : a = 4 * b)
  (h₂ : b = 3 * c)
  (h₃ : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end fraction_value_l302_302422


namespace find_m_and_max_profit_l302_302173

theorem find_m_and_max_profit (m : ℝ) (y : ℝ) (x : ℝ) (ln : ℝ → ℝ) 
    (h1 : y = m * ln x - 1 / 100 * x ^ 2 + 101 / 50 * x + ln 10)
    (h2 : 10 < x) 
    (h3 : y = 35.7) 
    (h4 : x = 20)
    (ln_2 : ln 2 = 0.7) 
    (ln_5 : ln 5 = 1.6) :
    m = -1 ∧ ∃ x, (x = 50 ∧ (-ln x - 1 / 100 * x ^ 2 + 51 / 50 * x + ln 10 - x) = 24.4) := by
  sorry

end find_m_and_max_profit_l302_302173


namespace angle_bisector_length_l302_302726

variable (a b : ℝ) (α l : ℝ)

theorem angle_bisector_length (ha : 0 < a) (hb : 0 < b) (hα : 0 < α) (hl : l = (2 * a * b * Real.cos (α / 2)) / (a + b)) :
  l = (2 * a * b * Real.cos (α / 2)) / (a + b) := by
  -- problem assumptions
  have h1 : a > 0 := ha
  have h2 : b > 0 := hb
  have h3 : α > 0 := hα
  -- conclusion
  exact hl

end angle_bisector_length_l302_302726


namespace sequence_property_l302_302912

theorem sequence_property :
  ∃ (a_0 a_1 a_2 a_3 : ℕ),
    a_0 + a_1 + a_2 + a_3 = 4 ∧
    (a_0 = ([a_0, a_1, a_2, a_3].count 0)) ∧
    (a_1 = ([a_0, a_1, a_2, a_3].count 1)) ∧
    (a_2 = ([a_0, a_1, a_2, a_3].count 2)) ∧
    (a_3 = ([a_0, a_1, a_2, a_3].count 3)) :=
sorry

end sequence_property_l302_302912


namespace find_x_l302_302919

theorem find_x (x : ℝ) :
  (x^2 - 7 * x + 12) / (x^2 - 9 * x + 20) = (x^2 - 4 * x - 21) / (x^2 - 5 * x - 24) -> x = 11 :=
by
  sorry

end find_x_l302_302919


namespace workers_and_days_l302_302505

theorem workers_and_days (x y : ℕ) (h1 : x * y = (x - 20) * (y + 5)) (h2 : x * y = (x + 15) * (y - 2)) :
  x = 60 ∧ y = 10 := 
by {
  sorry
}

end workers_and_days_l302_302505


namespace scatter_plot_can_be_made_l302_302662

theorem scatter_plot_can_be_made
    (data : List (ℝ × ℝ)) :
    ∃ (scatter_plot : List (ℝ × ℝ)), scatter_plot = data :=
by
  sorry

end scatter_plot_can_be_made_l302_302662


namespace living_room_floor_area_l302_302170

-- Define the problem conditions
def carpet_length : ℝ := 4
def carpet_width : ℝ := 9
def carpet_area : ℝ := carpet_length * carpet_width    -- Area of the carpet

def percentage_covered_by_carpet : ℝ := 0.75

-- Theorem to prove: the area of the living room floor is 48 square feet
theorem living_room_floor_area (carpet_area : ℝ) (percentage_covered_by_carpet : ℝ) : 
  (A_floor : ℝ) = carpet_area / percentage_covered_by_carpet :=
by
  let carpet_area := 36
  let percentage_covered_by_carpet := 0.75
  let A_floor := 48
  sorry

end living_room_floor_area_l302_302170


namespace jenna_eel_length_l302_302250

theorem jenna_eel_length (j b : ℕ) (h1 : b = 3 * j) (h2 : b + j = 64) : j = 16 := by 
  sorry

end jenna_eel_length_l302_302250


namespace fouad_double_ahmed_l302_302332

/-- Proof that in 4 years, Fouad's age will be double of Ahmed's age given their current ages. -/
theorem fouad_double_ahmed (x : ℕ) (ahmed_age fouad_age : ℕ) (h1 : ahmed_age = 11) (h2 : fouad_age = 26) :
  (fouad_age + x = 2 * (ahmed_age + x)) → x = 4 :=
by
  -- This is the statement only, proof is omitted
  sorry

end fouad_double_ahmed_l302_302332


namespace seeds_per_flowerbed_l302_302844

theorem seeds_per_flowerbed :
  ∀ (total_seeds flowerbeds seeds_per_bed : ℕ), 
  total_seeds = 32 → 
  flowerbeds = 8 → 
  seeds_per_bed = total_seeds / flowerbeds → 
  seeds_per_bed = 4 :=
  by 
    intros total_seeds flowerbeds seeds_per_bed h_total h_flowerbeds h_calc
    rw [h_total, h_flowerbeds] at h_calc
    exact h_calc

end seeds_per_flowerbed_l302_302844


namespace total_students_shook_hands_l302_302622

theorem total_students_shook_hands (S3 S2 S1 : ℕ) (h1 : S3 = 200) (h2 : S2 = S3 + 40) (h3 : S1 = 2 * S2) : 
  S1 + S2 + S3 = 920 :=
by
  sorry

end total_students_shook_hands_l302_302622


namespace number_of_red_balls_l302_302699

theorem number_of_red_balls
    (black_balls : ℕ)
    (frequency : ℝ)
    (total_balls : ℕ)
    (red_balls : ℕ) 
    (h_black : black_balls = 5)
    (h_frequency : frequency = 0.25)
    (h_total : total_balls = black_balls / frequency) :
    red_balls = total_balls - black_balls → red_balls = 15 :=
by
  intros h_red
  sorry

end number_of_red_balls_l302_302699


namespace fgf_3_is_299_l302_302450

def f (x : ℕ) : ℕ := 5 * x + 4
def g (x : ℕ) : ℕ := 3 * x + 2
def h : ℕ := 3

theorem fgf_3_is_299 : f (g (f h)) = 299 :=
by
  sorry

end fgf_3_is_299_l302_302450


namespace find_xy_l302_302679

theorem find_xy (x y : ℝ) (h : x * (x + 2 * y) = x^2 + 10) : x * y = 5 :=
by
  sorry

end find_xy_l302_302679


namespace number_of_schools_l302_302197

theorem number_of_schools (N : ℕ) :
  (∀ i j : ℕ, i < j → i ≠ j) →
  (∀ i : ℕ, i < 2 * 35 → i = 35 ∨ ((i = 37 → ¬ (i = 35))) ∧ ((i = 64 → ¬ (i = 35)))) →
  N = (2 * (35) - 1) / 3 →
  N = 23 :=
by
  sorry

end number_of_schools_l302_302197


namespace painting_problem_l302_302822

theorem painting_problem
    (H_rate : ℝ := 1 / 60)
    (T_rate : ℝ := 1 / 90)
    (combined_rate : ℝ := H_rate + T_rate)
    (time_worked : ℝ := 15)
    (wall_painted : ℝ := time_worked * combined_rate):
  wall_painted = 5 / 12 := 
by
  sorry

end painting_problem_l302_302822


namespace min_amount_to_pay_max_amount_to_pay_expected_amount_to_pay_l302_302000

/- Part (a): Minimum amount the customer will pay -/
theorem min_amount_to_pay : 
  (∑ i in {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000} \ 0, id i) = 4000 := 
sorry

/- Part (b): Maximum amount the customer will pay -/
theorem max_amount_to_pay : 
  (∑ i in {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000} \ 0, id i) = 4900 := 
sorry

/- Part (c): Expected value the customer will pay -/
theorem expected_amount_to_pay :
  (∑ i in {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000} \ 0, id i) = 4583.33 := 
sorry

end min_amount_to_pay_max_amount_to_pay_expected_amount_to_pay_l302_302000


namespace simplify_sqrt_450_l302_302118

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l302_302118


namespace simplify_sqrt_450_l302_302122

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l302_302122


namespace circumcenter_on_AK_l302_302644

variable {α β γ : Real}
variable (A B C L H K O : Type)
variable [Triangle ABC] (circumcenter : Triangle ABC → Point O)
variable [AngleBisector A B C L]

theorem circumcenter_on_AK
  (h₁ : AL_is_angle_bisector ABC L)
  (h₂ : Height_from_B_on_AL B A L H)
  (h₃ : K_on_circumcircle_ABL B A L K)
  : Lies_on_line (circumcenter ABC) A K :=
sorry

end circumcenter_on_AK_l302_302644


namespace simplify_sqrt_450_l302_302106

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l302_302106


namespace xyz_range_l302_302595

theorem xyz_range (x y z : ℝ) (h1 : x + y + z = 1) (h2 : x^2 + y^2 + z^2 = 3) : 
  -1 ≤ x * y * z ∧ x * y * z ≤ 5 / 27 := 
sorry

end xyz_range_l302_302595


namespace mans_speed_against_current_l302_302323

theorem mans_speed_against_current
  (speed_with_current : ℝ)
  (speed_of_current : ℝ)
  (h1 : speed_with_current = 25)
  (h2 : speed_of_current = 2.5) :
  speed_with_current - 2 * speed_of_current = 20 := 
by
  sorry

end mans_speed_against_current_l302_302323


namespace sqrt_450_simplified_l302_302043

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l302_302043


namespace red_balls_in_box_l302_302862

theorem red_balls_in_box (initial_red_balls added_red_balls : ℕ) (initial_blue_balls : ℕ) 
  (h_initial : initial_red_balls = 5) (h_added : added_red_balls = 2) : 
  initial_red_balls + added_red_balls = 7 :=
by {
  sorry
}

end red_balls_in_box_l302_302862


namespace smallest_integer_l302_302760

theorem smallest_integer :
  ∃ (M : ℕ), M > 0 ∧
             M % 3 = 2 ∧
             M % 4 = 3 ∧
             M % 5 = 4 ∧
             M % 6 = 5 ∧
             M % 7 = 6 ∧
             M % 11 = 10 ∧
             M = 4619 :=
by
  sorry

end smallest_integer_l302_302760


namespace distance_NYC_to_DC_l302_302238

noncomputable def horse_speed := 10 -- miles per hour
noncomputable def travel_time := 24 -- hours

theorem distance_NYC_to_DC : horse_speed * travel_time = 240 := by
  sorry

end distance_NYC_to_DC_l302_302238


namespace area_of_triangle_formed_by_line_l_and_axes_equation_of_line_l_l302_302372

-- Defining the lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Intersection point P of line1 and line2
def P : ℝ × ℝ := (-2, 2)

-- Perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Line l, passing through P and perpendicular to perpendicular_line
def line_l (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Intercepts of line_l with axes
def x_intercept : ℝ := -1
def y_intercept : ℝ := -2

-- Verifying area of the triangle formed by the intercepts
def area_of_triangle (base height : ℝ) : ℝ := 0.5 * base * height

#check line1
#check line2
#check P
#check perpendicular_line
#check line_l
#check x_intercept
#check y_intercept
#check area_of_triangle

theorem area_of_triangle_formed_by_line_l_and_axes :
  ∀ (x : ℝ) (y : ℝ),
    line_l x 0 → line_l 0 y →
    area_of_triangle (abs x) (abs y) = 1 :=
by
  intros x y hx hy
  sorry

theorem equation_of_line_l :
  ∀ (x y : ℝ),
    (line1 x y ∧ line2 x y) →
    (perpendicular_line x y) →
    line_l x y :=
by
  intros x y h1 h2
  sorry

end area_of_triangle_formed_by_line_l_and_axes_equation_of_line_l_l302_302372


namespace three_point_three_seven_five_as_fraction_l302_302491

theorem three_point_three_seven_five_as_fraction :
  3.375 = (27 / 8 : ℚ) :=
by sorry

end three_point_three_seven_five_as_fraction_l302_302491


namespace solve_for_q_l302_302567

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : q = -25 / 11 :=
by
  sorry

end solve_for_q_l302_302567


namespace time_to_cross_pole_l302_302329

-- Conditions
def train_speed_kmh : ℕ := 108
def train_length_m : ℕ := 210

-- Conversion functions
def km_per_hr_to_m_per_sec (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 1000 / 3600

-- Theorem to be proved
theorem time_to_cross_pole : (train_length_m : ℕ) / (km_per_hr_to_m_per_sec train_speed_kmh) = 7 := by
  -- we'll use sorry here to skip the actual proof steps.
  sorry

end time_to_cross_pole_l302_302329


namespace solve_system_of_equations_l302_302540

theorem solve_system_of_equations (n : ℕ) (hn : n ≥ 3) (x : ℕ → ℝ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n →
    x i ^ 3 = (x ((i % n) + 1) + x ((i % n) + 2) + 1)) →
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n →
    (x i = -1 ∨ x i = (1 + Real.sqrt 5) / 2 ∨ x i = (1 - Real.sqrt 5) / 2)) :=
sorry

end solve_system_of_equations_l302_302540


namespace Brenda_new_lead_l302_302792

noncomputable def Brenda_initial_lead : ℤ := 22
noncomputable def Brenda_play_points : ℤ := 15
noncomputable def David_play_points : ℤ := 32

theorem Brenda_new_lead : 
  Brenda_initial_lead + Brenda_play_points - David_play_points = 5 := 
by
  sorry

end Brenda_new_lead_l302_302792


namespace circumradius_of_triangle_APQ_l302_302966

variable {A B C P Q: Type*}
variable [DecidableEq A]
variable [DecidableEq B]
variable [DecidableEq C]
variable [DecidableEq P]
variable [DecidableEq Q]
variable r1 r2 : ℝ

-- Given triangle ABC with angle BAC = 60 degrees, 
-- P is the intersection of angle bisector of ABC with side AC,
-- Q is the intersection of angle bisector of ACB with side AB,
-- r1 is the in-radius of triangle ABC,
-- r2 is the in-radius of triangle APQ

theorem circumradius_of_triangle_APQ (h₁ : ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a^2 + b^2 - a * b = c^2) 
  (h₂ : ∃ (p q : ℝ), 0 < p ∧ 0 < q ∧ (p + q)^2 = 2(p^2 + q^2 - p * q)) :
  circumradius (triangle APQ) = 2 * (r1 - r2) :=
sorry

end circumradius_of_triangle_APQ_l302_302966


namespace simplify_sqrt_450_l302_302063

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l302_302063


namespace simplify_sqrt_450_l302_302086

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l302_302086


namespace shooting_competition_sequences_l302_302698

/-- In a shooting competition, eight clay targets are set up in three hanging columns with the
configuration: three targets in the first column (A), two targets in the second column (B), 
and three targets in the third column (C). We need to count the number of different sequences
the shooter can follow to break all targets, following these rules:
1) The shooter selects any one of the columns to shoot a target from.
2) The shooter must then hit the lowest remaining target in the selected column.
-/
theorem shooting_competition_sequences : 
  let A := 3 in let B := 2 in let C := 3 in (A + B + C = 8) →
  (∑ n : ℕ in {n | (n = 8)}, 
    Multinomial (A + B + C) ! ! (A !, B !, C !)) = 560 :=
by {
  have hA : 3,
  have hB : 2,
  have hC : 3,
  have h_total : A + B + C = 8 := by norm_num,
  sorry -- proof to be completed
}

end shooting_competition_sequences_l302_302698


namespace inequality_holds_for_all_real_l302_302922

open Real -- Open the real numbers namespace

theorem inequality_holds_for_all_real (x : ℝ) : 
  2^((sin x)^2) + 2^((cos x)^2) ≥ 2 * sqrt 2 :=
by
  sorry

end inequality_holds_for_all_real_l302_302922


namespace john_took_11_more_l302_302608

/-- 
If Ray took 10 chickens, Ray took 6 chickens less than Mary, and 
John took 5 more chickens than Mary, then John took 11 more 
chickens than Ray. 
-/
theorem john_took_11_more (R M J : ℕ) (h1 : R = 10) 
  (h2 : R + 6 = M) (h3 : M + 5 = J) : J - R = 11 :=
by
  sorry

end john_took_11_more_l302_302608


namespace probability_of_A_chosen_l302_302656

-- Define the set of persons
inductive Person
| A : Person
| B : Person
| C : Person

open Person

-- Function to determine the selections
def selection (p : Person) : Finset (Finset Person) :=
if p = A then { {A, B}, {A, C} } else if p = B then { {B, A}, {B, C} } else { {C, A}, {C, B} }

-- Prove the probability of A being chosen when selecting two representatives from three people
theorem probability_of_A_chosen : (2 : ℚ) / 3 = 2 / 3 :=
by
  sorry

end probability_of_A_chosen_l302_302656


namespace total_cases_is_8_l302_302293

def num_blue_cards : Nat := 3
def num_yellow_cards : Nat := 5

def total_cases : Nat := num_blue_cards + num_yellow_cards

theorem total_cases_is_8 : total_cases = 8 := by
  sorry

end total_cases_is_8_l302_302293


namespace imo1987_q6_l302_302202

theorem imo1987_q6 (m n : ℤ) (h : n = m + 2) :
  ⌊(n : ℝ) * Real.sqrt 2⌋ = 2 + ⌊(m : ℝ) * Real.sqrt 2⌋ := 
by
  sorry -- We skip the detailed proof steps here.

end imo1987_q6_l302_302202


namespace henry_books_count_l302_302939

theorem henry_books_count
  (initial_books : ℕ)
  (boxes : ℕ)
  (books_per_box : ℕ)
  (room_books : ℕ)
  (table_books : ℕ)
  (kitchen_books : ℕ)
  (picked_books : ℕ) :
  initial_books = 99 →
  boxes = 3 →
  books_per_box = 15 →
  room_books = 21 →
  table_books = 4 →
  kitchen_books = 18 →
  picked_books = 12 →
  initial_books - (boxes * books_per_box + room_books + table_books + kitchen_books) + picked_books = 23 :=
by
  intros initial_books_eq boxes_eq books_per_box_eq room_books_eq table_books_eq kitchen_books_eq picked_books_eq
  rw [initial_books_eq, boxes_eq, books_per_box_eq, room_books_eq, table_books_eq, kitchen_books_eq, picked_books_eq]
  norm_num
  sorry

end henry_books_count_l302_302939


namespace sqrt_450_eq_15_sqrt_2_l302_302053

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302053


namespace average_annual_population_increase_l302_302499

theorem average_annual_population_increase 
    (initial_population : ℝ) 
    (final_population : ℝ) 
    (years : ℝ) 
    (initial_population_pos : initial_population > 0) 
    (years_pos : years > 0)
    (initial_population_eq : initial_population = 175000) 
    (final_population_eq : final_population = 297500) 
    (years_eq : years = 10) : 
    (final_population - initial_population) / initial_population / years * 100 = 7 :=
by
    sorry

end average_annual_population_increase_l302_302499


namespace total_cost_of_toys_l302_302453

-- Define the costs of the yoyo and the whistle
def cost_yoyo : Nat := 24
def cost_whistle : Nat := 14

-- Prove the total cost of the yoyo and the whistle is 38 cents
theorem total_cost_of_toys : cost_yoyo + cost_whistle = 38 := by
  sorry

end total_cost_of_toys_l302_302453


namespace power_mod_eight_l302_302869

theorem power_mod_eight : 3 ^ 2007 % 8 = 3 % 8 := by
  sorry

end power_mod_eight_l302_302869


namespace cos_beta_value_l302_302810

theorem cos_beta_value (α β : ℝ) (hα1 : 0 < α ∧ α < π/2) (hβ1 : 0 < β ∧ β < π/2) 
  (h1 : Real.sin α = 4/5) (h2 : Real.cos (α + β) = -12/13) : 
  Real.cos β = -16/65 := 
by 
  sorry

end cos_beta_value_l302_302810


namespace remainder_2pow33_minus_1_div_9_l302_302992

theorem remainder_2pow33_minus_1_div_9 : (2^33 - 1) % 9 = 7 := 
  sorry

end remainder_2pow33_minus_1_div_9_l302_302992


namespace johns_elevation_after_travel_l302_302710

-- Definitions based on conditions:
def initial_elevation : ℝ := 400
def downward_rate : ℝ := 10
def time_travelled : ℕ := 5

-- Proof statement:
theorem johns_elevation_after_travel:
  initial_elevation - (downward_rate * time_travelled) = 350 :=
by
  sorry

end johns_elevation_after_travel_l302_302710


namespace jack_jill_meeting_distance_l302_302702

-- Definitions for Jack's and Jill's initial conditions
def jack_speed_uphill := 12 -- km/hr
def jack_speed_downhill := 15 -- km/hr
def jill_speed_uphill := 14 -- km/hr
def jill_speed_downhill := 18 -- km/hr

def head_start := 0.2 -- hours
def total_distance := 12 -- km
def turn_point_distance := 7 -- km
def return_distance := 5 -- km

-- Statement of the problem to prove the distance from the turning point where they meet
theorem jack_jill_meeting_distance :
  let jack_time_to_turn := (turn_point_distance : ℚ) / jack_speed_uphill
  let jill_time_to_turn := (turn_point_distance : ℚ) / jill_speed_uphill
  let x_meet := (8.95 : ℚ) / 29
  7 - (14 * ((x_meet - 0.2) / 1)) = (772 / 145 : ℚ) := 
sorry

end jack_jill_meeting_distance_l302_302702


namespace fraction_draw_l302_302624

/-
Theorem: Given the win probabilities for Amy, Lily, and Eve, the fraction of the time they end up in a draw is 3/10.
-/

theorem fraction_draw (P_Amy P_Lily P_Eve : ℚ) (h_Amy : P_Amy = 2/5) (h_Lily : P_Lily = 1/5) (h_Eve : P_Eve = 1/10) : 
  1 - (P_Amy + P_Lily + P_Eve) = 3 / 10 := by
  sorry

end fraction_draw_l302_302624


namespace hexagon_ratio_l302_302338

theorem hexagon_ratio 
  (hex_area : ℝ)
  (rs_bisects_area : ∃ (a b : ℝ), a + b = hex_area / 2 ∧ ∃ (x r s : ℝ), x = 4 ∧ r * s = (hex_area / 2 - 1))
  : ∀ (XR RS : ℝ), XR = RS → XR / RS = 1 :=
by
  sorry

end hexagon_ratio_l302_302338


namespace square_areas_l302_302899

variables (a b : ℝ)

def is_perimeter_difference (a b : ℝ) : Prop :=
  4 * a - 4 * b = 12

def is_area_difference (a b : ℝ) : Prop :=
  a^2 - b^2 = 69

theorem square_areas (a b : ℝ) (h1 : is_perimeter_difference a b) (h2 : is_area_difference a b) :
  a^2 = 169 ∧ b^2 = 100 :=
by {
  sorry
}

end square_areas_l302_302899


namespace running_time_constant_pace_l302_302235

/-!
# Running Time Problem

We are given that the running pace is constant, it takes 30 minutes to run 5 miles,
and we need to find out how long it will take to run 2.5 miles.
-/

theorem running_time_constant_pace :
  ∀ (distance_to_store distance_to_cousin distance_run time_run : ℝ)
  (constant_pace : Prop),
  distance_to_store = 5 → time_run = 30 → distance_to_cousin = 2.5 →
  constant_pace → 
  time_run / distance_to_store * distance_to_cousin = 15 :=
by 
  intros distance_to_store distance_to_cousin distance_run time_run constant_pace 
         hds htr hdc hcp
  rw [hds, htr, hdc]
  exact sorry

end running_time_constant_pace_l302_302235


namespace simplify_expression_l302_302461

theorem simplify_expression (x : ℤ) : 
  (12*x^10 + 5*x^9 + 3*x^8) + (2*x^12 + 9*x^10 + 4*x^8 + 6*x^4 + 7*x^2 + 10)
  = 2*x^12 + 21*x^10 + 5*x^9 + 7*x^8 + 6*x^4 + 7*x^2 + 10 :=
by sorry

end simplify_expression_l302_302461


namespace smallest_integer_in_ratio_l302_302625

theorem smallest_integer_in_ratio {a b c : ℕ} (h1 : a = 2 * b / 3) (h2 : c = 5 * b / 3) (h3 : a + b + c = 60) : b = 12 := 
  sorry

end smallest_integer_in_ratio_l302_302625


namespace find_value_of_fraction_l302_302414

theorem find_value_of_fraction (a b c d: ℕ) (h1: a = 4 * b) (h2: b = 3 * c) (h3: c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end find_value_of_fraction_l302_302414


namespace base_subtraction_problem_l302_302342

theorem base_subtraction_problem (b : ℕ) (C_b : ℕ) (hC : C_b = 12) : 
  b = 15 :=
by
  sorry

end base_subtraction_problem_l302_302342


namespace find_m_value_l302_302214

variable (m : ℝ)

theorem find_m_value (h1 : m^2 - 3 * m = 4)
                     (h2 : m^2 = 5 * m + 6) : m = -1 :=
sorry

end find_m_value_l302_302214


namespace find_a_even_function_l302_302617

theorem find_a_even_function (a : ℝ) :
  (∀ x : ℝ, (x ^ 2 + a * x - 4) = ((-x) ^ 2 + a * (-x) - 4)) → a = 0 :=
by
  intro h
  sorry

end find_a_even_function_l302_302617


namespace orthocentric_tetrahedron_edge_tangent_iff_l302_302975

structure Tetrahedron :=
(V : Type*)
(a b c d e f : V)
(is_orthocentric : Prop)
(has_edge_tangent_sphere : Prop)
(face_equilateral : Prop)
(edges_converging_equal : Prop)

variable (T : Tetrahedron)

noncomputable def edge_tangent_iff_equilateral_edges_converging_equal : Prop :=
T.has_edge_tangent_sphere ↔ (T.face_equilateral ∧ T.edges_converging_equal)

-- Now create the theorem statement
theorem orthocentric_tetrahedron_edge_tangent_iff :
  T.is_orthocentric →
  (∀ a d b e c f p r : ℝ, 
    a + d = b + e ∧ b + e = c + f ∧ a^2 + d^2 = b^2 + e^2 ∧ b^2 + e^2 = c^2 + f^2 ) → 
    edge_tangent_iff_equilateral_edges_converging_equal T := 
by
  intros
  unfold edge_tangent_iff_equilateral_edges_converging_equal
  sorry

end orthocentric_tetrahedron_edge_tangent_iff_l302_302975


namespace parallel_lines_m_eq_neg2_l302_302382

def l1_equation (m : ℝ) (x y: ℝ) : Prop :=
  (m+1) * x + y - 1 = 0

def l2_equation (m : ℝ) (x y: ℝ) : Prop :=
  2 * x + m * y - 1 = 0

theorem parallel_lines_m_eq_neg2 (m : ℝ) :
  (∀ x y : ℝ, l1_equation m x y) →
  (∀ x y : ℝ, l2_equation m x y) →
  (m ≠ 1) →
  (m = -2) :=
sorry

end parallel_lines_m_eq_neg2_l302_302382


namespace exists_integers_for_x_squared_minus_y_squared_eq_a_fifth_l302_302767

theorem exists_integers_for_x_squared_minus_y_squared_eq_a_fifth (a : ℤ) : 
  ∃ x y : ℤ, x^2 - y^2 = a^5 :=
sorry

end exists_integers_for_x_squared_minus_y_squared_eq_a_fifth_l302_302767


namespace fem_current_age_l302_302722

theorem fem_current_age (F : ℕ) 
  (h1 : ∃ M : ℕ, M = 4 * F) 
  (h2 : (F + 2) + (4 * F + 2) = 59) : 
  F = 11 :=
sorry

end fem_current_age_l302_302722


namespace value_of_expression_l302_302403

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := by
  sorry

end value_of_expression_l302_302403


namespace work_together_l302_302500

theorem work_together (W : ℝ) (Dx Dy : ℝ) (hx : Dx = 15) (hy : Dy = 30) : 
  (Dx * Dy) / (Dx + Dy) = 10 := 
by
  sorry

end work_together_l302_302500


namespace ksyusha_travel_time_l302_302271

variables (v S : ℝ)

theorem ksyusha_travel_time :
  (2 * S / v) + (S / (2 * v)) = 30 →
  (S / v) + (2 * S / (2 * v)) = 24 :=
by sorry

end ksyusha_travel_time_l302_302271


namespace problem1_solution_l302_302983

theorem problem1_solution : ∀ x : ℝ, x^2 - 6 * x + 9 = (5 - 2 * x)^2 → (x = 8/3 ∨ x = 2) :=
sorry

end problem1_solution_l302_302983


namespace value_of_ac_over_bd_l302_302408

theorem value_of_ac_over_bd (a b c d : ℝ) 
  (h1 : a = 4 * b)
  (h2 : b = 3 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by
  sorry

end value_of_ac_over_bd_l302_302408


namespace bears_in_shipment_l302_302328

theorem bears_in_shipment
  (initial_bears : ℕ) (shelves : ℕ) (bears_per_shelf : ℕ)
  (total_bears_after_shipment : ℕ) 
  (initial_bears_eq : initial_bears = 5)
  (shelves_eq : shelves = 2)
  (bears_per_shelf_eq : bears_per_shelf = 6)
  (total_bears_calculation : total_bears_after_shipment = shelves * bears_per_shelf)
  : total_bears_after_shipment - initial_bears = 7 :=
by
  sorry

end bears_in_shipment_l302_302328


namespace cheryl_distance_walked_l302_302526

theorem cheryl_distance_walked (speed : ℕ) (time : ℕ) (distance_away : ℕ) (distance_home : ℕ) 
  (h1 : speed = 2) 
  (h2 : time = 3) 
  (h3 : distance_away = speed * time) 
  (h4 : distance_home = distance_away) : 
  distance_away + distance_home = 12 := 
by
  sorry

end cheryl_distance_walked_l302_302526


namespace find_total_original_cost_l302_302569

noncomputable def original_total_cost (x y z : ℝ) : ℝ :=
x + y + z

theorem find_total_original_cost (x y z : ℝ)
  (h1 : x * 1.30 = 351)
  (h2 : y * 1.25 = 275)
  (h3 : z * 1.20 = 96) :
  original_total_cost x y z = 570 :=
sorry

end find_total_original_cost_l302_302569


namespace additional_land_cost_l302_302790

noncomputable def initial_land := 300
noncomputable def final_land := 900
noncomputable def cost_per_square_meter := 20

theorem additional_land_cost : (final_land - initial_land) * cost_per_square_meter = 12000 :=
by
  -- Define the amount of additional land purchased
  let additional_land := final_land - initial_land
  -- Calculate the cost of the additional land            
  show additional_land * cost_per_square_meter = 12000
  sorry

end additional_land_cost_l302_302790


namespace intersection_eq_l302_302451

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {x | x ≤ 2}

theorem intersection_eq : P ∩ Q = {1, 2} :=
by
  sorry

end intersection_eq_l302_302451


namespace simplify_sqrt_450_l302_302067

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l302_302067


namespace find_a8_l302_302575

-- Define the arithmetic sequence aₙ
def arithmetic_sequence (a₁ d : ℕ) (n : ℕ) := a₁ + (n - 1) * d

-- The given condition
def condition (a₁ d : ℕ) :=
  arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 7 + arithmetic_sequence a₁ d 15 = 12

-- The value we want to prove
def a₈ (a₁ d : ℕ ) : ℕ :=
  arithmetic_sequence a₁ d 8

theorem find_a8 (a₁ d : ℕ) (h : condition a₁ d) : a₈ a₁ d = 4 :=
  sorry

end find_a8_l302_302575


namespace distinct_patterns_4x4_3_shaded_l302_302689

def num_distinct_patterns (n : ℕ) (shading : ℕ) : ℕ :=
  if n = 4 ∧ shading = 3 then 15
  else 0 -- Placeholder for other cases, not relevant for our problem

theorem distinct_patterns_4x4_3_shaded :
  num_distinct_patterns 4 3 = 15 :=
by {
  -- The proof would go here
  sorry
}

end distinct_patterns_4x4_3_shaded_l302_302689


namespace divisible_by_17_l302_302860

theorem divisible_by_17 (a b c d : ℕ) (h1 : a + b + c + d = 2023)
    (h2 : 2023 ∣ (a * b - c * d))
    (h3 : 2023 ∣ (a^2 + b^2 + c^2 + d^2))
    (h4 : ∀ x, x = a ∨ x = b ∨ x = c ∨ x = d → 7 ∣ x) :
    (∀ x, x = a ∨ x = b ∨ x = c ∨ x = d → 17 ∣ x) := 
sorry

end divisible_by_17_l302_302860


namespace maximum_M_k_l302_302358

-- Define the problem
def J (k : ℕ) : ℕ := 10^(k + 2) + 128

-- Define M(k) as the number of factors of 2 in the prime factorization of J(k)
def M (k : ℕ) : ℕ :=
  -- implementation details omitted
  sorry

-- The core theorem to prove
theorem maximum_M_k : ∃ k > 0, M k = 8 :=
by sorry

end maximum_M_k_l302_302358


namespace min_payment_max_payment_expected_payment_l302_302002

-- Given Prices
def item_prices : List ℕ := [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

-- Function to compute the actual paid amount given groups of three items
def paid_amount (groups : List (List ℕ)) : ℕ :=
  groups.foldr (λ group sum => sum + group.foldr (λ x s => s + x) 0 - group.minimum') 0

-- Optimal arrangement of items for minimal payment
def optimal_groups : List (List ℕ) :=
  [[1000, 900, 800], [700, 600, 500], [400, 300, 200], [100]]

-- Suboptimal arrangement of items for maximal payment
def suboptimal_groups : List (List ℕ) :=
  [[1000, 900, 100], [800, 700, 200], [600, 500, 300], [400]]

-- Expected value calculation's configuration
def num_items := 10
def num_groups := (num_items / 3).natCeil

noncomputable def expected_amount : ℕ :=
  let total_sum := item_prices.foldr (λ x s => s + x) 0
  let expected_savings := 100 * (660 / 72)
  total_sum - expected_savings

theorem min_payment : paid_amount optimal_groups = 4000 := by
  -- Proof steps and details here
  sorry

theorem max_payment : paid_amount suboptimal_groups = 4900 := by
  -- Proof steps and details here
  sorry

theorem expected_payment : expected_amount ≈ 4583 := by
  -- Proof steps and details here
  sorry

end min_payment_max_payment_expected_payment_l302_302002


namespace domain_of_sqrt_log_l302_302615

theorem domain_of_sqrt_log {x : ℝ} : (2 < x ∧ x ≤ 5 / 2) ↔ 
  (5 - 2 * x > 0 ∧ 0 ≤ Real.logb (1 / 2) (5 - 2 * x)) :=
sorry

end domain_of_sqrt_log_l302_302615


namespace total_cost_of_hotel_stay_l302_302996

-- Define the necessary conditions
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- State the problem
theorem total_cost_of_hotel_stay :
  (cost_per_night_per_person * number_of_people * number_of_nights) = 360 := by
  sorry

end total_cost_of_hotel_stay_l302_302996


namespace sqrt_450_simplified_l302_302051

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l302_302051


namespace sqrt_450_eq_15_sqrt_2_l302_302059

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302059


namespace sqrt_450_simplified_l302_302038

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l302_302038


namespace unique_tangent_circle_of_radius_2_l302_302745

noncomputable def is_tangent (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  dist c₁ c₂ = r₁ + r₂

theorem unique_tangent_circle_of_radius_2
    (C1_center C2_center C3_center : ℝ × ℝ)
    (h_C1_C2 : is_tangent C1_center C2_center 1 1)
    (h_C2_C3 : is_tangent C2_center C3_center 1 1)
    (h_C3_C1 : is_tangent C3_center C1_center 1 1):
    ∃! center : ℝ × ℝ, is_tangent center C1_center 2 1 ∧
                        is_tangent center C2_center 2 1 ∧
                        is_tangent center C3_center 2 1 := sorry

end unique_tangent_circle_of_radius_2_l302_302745


namespace seq_problem_l302_302717

theorem seq_problem (a : ℕ → ℚ) (d : ℚ) (h_arith : ∀ n : ℕ, a (n + 1) = a n + d )
 (h1 : a 1 = 2)
 (h_geom : (a 1 - 1) * (a 5 + 5) = (a 3)^2) :
  a 2017 = 1010 := 
sorry

end seq_problem_l302_302717


namespace find_a_b_l302_302666

theorem find_a_b (a b : ℝ) (z : ℂ) (hz : z = 1 + Complex.I) 
  (h : (z^2 + a*z + b) / (z^2 - z + 1) = 1 - Complex.I) : a = -1 ∧ b = 2 :=
by
  sorry

end find_a_b_l302_302666


namespace range_of_a_l302_302677

noncomputable def f (x : ℝ) : ℝ := x + Real.log x
noncomputable def g (a x : ℝ) : ℝ := a * x - 2 * Real.sin x

theorem range_of_a (a : ℝ) :
  (∀ x₁ > 0, ∃ x₂, (1 + 1 / x₁) * (a - 2 * Real.cos x₂) = -1) →
  -2 ≤ a ∧ a ≤ 1 :=
by {
  sorry
}

end range_of_a_l302_302677


namespace monotonic_interval_range_l302_302237

noncomputable def f (a x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

theorem monotonic_interval_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < 2 → 1 < x₂ → x₂ < 2 → x₁ < x₂ → f a x₁ ≤ f a x₂ ∨ f a x₁ ≥ f a x₂) ↔
  (a ∈ Set.Iic (-1) ∪ Set.Ici 0) :=
sorry

end monotonic_interval_range_l302_302237


namespace total_pencils_proof_l302_302465

noncomputable def total_pencils (Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff : ℕ) : ℕ :=
  Asaf_pencils + Alexander_pencils

theorem total_pencils_proof :
  ∀ (Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff : ℕ),
  Asaf_age = 50 →
  Alexander_age = 140 - Asaf_age →
  total_age_diff = Alexander_age - Asaf_age →
  Asaf_pencils = 2 * total_age_diff →
  Alexander_pencils = Asaf_pencils + 60 →
  total_pencils Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff = 220 :=
by
  intros
  sorry

end total_pencils_proof_l302_302465


namespace find_k_l302_302933

theorem find_k (k : ℝ) : 
  (∀ α β : ℝ, (α * β = 15 ∧ α + β = -k ∧ (α + 3 + β + 3 = k)) → k = 3) :=
by 
  sorry

end find_k_l302_302933


namespace sqrt_simplify_l302_302031

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l302_302031


namespace value_of_ac_over_bd_l302_302407

theorem value_of_ac_over_bd (a b c d : ℝ) 
  (h1 : a = 4 * b)
  (h2 : b = 3 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by
  sorry

end value_of_ac_over_bd_l302_302407


namespace cross_section_area_of_truncated_pyramid_l302_302993

-- Given conditions
variables (a b : ℝ) (α : ℝ)
-- Constraints
variable (h : a > b ∧ b > 0 ∧ α > 0 ∧ α < Real.pi / 2)

-- Proposed theorem
theorem cross_section_area_of_truncated_pyramid (h : a > b ∧ b > 0 ∧ α > 0 ∧ α < Real.pi / 2) :
    ∃ area : ℝ, area = (7 * a + 3 * b) / (144 * Real.cos α) * Real.sqrt (3 * (a^2 + b^2 + 2 * a * b * Real.cos (2 * α))) :=
sorry

end cross_section_area_of_truncated_pyramid_l302_302993


namespace price_reduction_l302_302888

theorem price_reduction (x : ℝ) :
  (20 + 2 * x) * (40 - x) = 1200 → x = 20 :=
by
  sorry

end price_reduction_l302_302888


namespace find_pairs_of_positive_integers_l302_302203

theorem find_pairs_of_positive_integers (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x^3 + y^3 = 4 * (x^2 * y + x * y^2 - 5) → (x = 1 ∧ y = 3) ∨ (x = 3 ∧ y = 1) :=
by
  sorry

end find_pairs_of_positive_integers_l302_302203


namespace at_least_one_genuine_l302_302827

theorem at_least_one_genuine (batch : Finset ℕ) 
  (h_batch_size : batch.card = 12) 
  (genuine_items : Finset ℕ)
  (h_genuine_size : genuine_items.card = 10)
  (defective_items : Finset ℕ)
  (h_defective_size : defective_items.card = 2)
  (h_disjoint : genuine_items ∩ defective_items = ∅)
  (drawn_items : Finset ℕ)
  (h_draw_size : drawn_items.card = 3)
  (h_subset : drawn_items ⊆ batch)
  (h_union : genuine_items ∪ defective_items = batch) :
  (∃ (x : ℕ), x ∈ drawn_items ∧ x ∈ genuine_items) :=
sorry

end at_least_one_genuine_l302_302827


namespace p_implies_q_q_not_implies_p_p_sufficient_but_not_necessary_l302_302367

variable (x : ℝ)

def p := |x| = x
def q := x^2 + x ≥ 0

theorem p_implies_q : p x → q x :=
by sorry

theorem q_not_implies_p : q x → ¬p x :=
by sorry

theorem p_sufficient_but_not_necessary : (p x → q x) ∧ ¬(q x → p x) :=
by sorry

end p_implies_q_q_not_implies_p_p_sufficient_but_not_necessary_l302_302367


namespace minimum_value_x_squared_plus_y_squared_l302_302824

-- We define our main proposition in Lean
theorem minimum_value_x_squared_plus_y_squared (x y : ℝ) 
  (h : (x + 5)^2 + (y - 12)^2 = 196) : x^2 + y^2 ≥ 169 :=
sorry

end minimum_value_x_squared_plus_y_squared_l302_302824


namespace solve_fish_tank_problem_l302_302986

def fish_tank_problem : Prop :=
  ∃ (first_tank_fish second_tank_fish third_tank_fish : ℕ),
  first_tank_fish = 7 + 8 ∧
  second_tank_fish = 2 * first_tank_fish ∧
  third_tank_fish = 10 ∧
  (third_tank_fish : ℚ) / second_tank_fish = 1 / 3

theorem solve_fish_tank_problem : fish_tank_problem :=
by
  sorry

end solve_fish_tank_problem_l302_302986


namespace arithmetic_sequence_sum_l302_302835

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (h_arith_seq: ∀ n: ℕ, S n = S 0 + n * (S 1 - S 0)) 
  (h5 : S 5 = 10) (h10 : S 10 = 30) : S 15 = 60 :=
by
  sorry

end arithmetic_sequence_sum_l302_302835


namespace find_pyramid_volume_l302_302735

noncomputable def volume_of_pyramid (α β R : ℝ) : ℝ :=
  (1/3) * R^3 * (Real.sin α)^2 * Real.cos (α/2) * Real.tan (Real.pi / 4 - α/2) * Real.tan β

theorem find_pyramid_volume (α β R : ℝ) 
  (base_isosceles : ∀ {a b c : ℝ}, a = b) -- Represents the isosceles triangle condition
  (dihedral_angles_equal : ∀ {angle : ℝ}, angle = β) -- Dihedral angle at the base
  (circumcircle_radius : {radius : ℝ // radius = R}) -- Radius of the circumcircle
  (height_through_point : true) -- Condition: height passes through a point inside the triangle
  :
  volume_of_pyramid α β R = (1/3) * R^3 * (Real.sin α)^2 * Real.cos (α/2) * Real.tan (Real.pi / 4 - α/2) * Real.tan β :=
by {
  sorry
}

end find_pyramid_volume_l302_302735


namespace simplify_sqrt_450_l302_302011

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l302_302011


namespace remainder_of_x_500_div_x2_plus_1_x2_minus_1_l302_302353

theorem remainder_of_x_500_div_x2_plus_1_x2_minus_1 :
  (x^500) % ((x^2 + 1) * (x^2 - 1)) = 1 :=
sorry

end remainder_of_x_500_div_x2_plus_1_x2_minus_1_l302_302353


namespace simplify_sqrt_450_l302_302119

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l302_302119


namespace quadratic_roots_range_l302_302359

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x^2 + (a^2 - 1) * x + a - 2 = 0 ∧ y^2 + (a^2 - 1) * y + a - 2 = 0 ∧ x ≠ y ∧ x > 1 ∧ y < 1) ↔ -2 < a ∧ a < 1 := 
sorry

end quadratic_roots_range_l302_302359


namespace Ksyusha_travel_time_l302_302262

theorem Ksyusha_travel_time:
  ∀ (S v : ℝ), 
  (S > 0 ∧ v > 0) →
  (∀ tuesday_distance tuesday_time wednesday_distance wednesday_time: ℝ, 
    tuesday_distance = 3 * S ∧ 
    tuesday_distance = 2 * S + S ∧ 
    tuesday_time = (2 * S) / v + (S / (2 * v)) ∧ 
    tuesday_time = 30) → 
  (∀ wednesday_distance wednesday_time : ℝ, 
    wednesday_distance = S + 2 * S ∧ 
    wednesday_time = S / v + (2 * S) / (2 * v)) →
  wednesday_time = 24 :=
by
  intros S v S_pos v_pos tuesday_distance tuesday_time wednesday_distance wednesday_time
  sorry

end Ksyusha_travel_time_l302_302262


namespace simplify_sqrt_450_l302_302089

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l302_302089


namespace jana_distance_l302_302249

theorem jana_distance (time_to_walk_one_mile : ℝ) (time_to_walk : ℝ) :
  (time_to_walk_one_mile = 18) → (time_to_walk = 15) →
  ((time_to_walk / time_to_walk_one_mile) * 1 = 0.8) :=
  by
    intros h1 h2
    rw [h1, h2]
    -- Here goes the proof, but it is skipped as per requirements
    sorry

end jana_distance_l302_302249


namespace non_shaded_area_l302_302829

theorem non_shaded_area (r : ℝ) (A : ℝ) (shaded : ℝ) (non_shaded : ℝ) :
  (r = 5) ∧ (A = 4 * (π * r^2)) ∧ (shaded = 8 * (1 / 4 * π * r^2 - (1 / 2 * r * r))) ∧
  (non_shaded = A - shaded) → 
  non_shaded = 50 * π + 100 :=
by
  intro h
  obtain ⟨r_eq_5, A_eq, shaded_eq, non_shaded_eq⟩ := h
  rw [r_eq_5] at *
  sorry

end non_shaded_area_l302_302829


namespace find_square_side_length_l302_302670

noncomputable def square_side_length (a : ℝ) : Prop :=
  let angle_deg := 30
  let a_sqr_minus_1 := Real.sqrt (a ^ 2 - 1)
  let a_sqr_minus_4 := Real.sqrt (a ^ 2 - 4)
  let dihedral_cos := Real.cos (Real.pi / 6)  -- 30 degrees in radians
  let dihedral_sin := Real.sin (Real.pi / 6)
  let area_1 := 0.5 * a_sqr_minus_1 * a_sqr_minus_4 * dihedral_sin
  let area_2 := 0.5 * Real.sqrt (a ^ 4 - 5 * a ^ 2)
  dihedral_cos = (Real.sqrt 3 / 2) -- Using the provided angle
  ∧ dihedral_sin = 0.5
  ∧ area_1 = area_2
  ∧ a = 2 * Real.sqrt 5

-- The theorem stating that the side length of the square is 2\sqrt{5}
theorem find_square_side_length (a : ℝ) (H : square_side_length a) : a = 2 * Real.sqrt 5 := by
  sorry

end find_square_side_length_l302_302670


namespace conversion_problem_l302_302910

noncomputable def conversion1 : ℚ :=
  35 * (1/1000)  -- to convert cubic decimeters to cubic meters

noncomputable def conversion2 : ℚ :=
  53 * (1/60)  -- to convert seconds to minutes

noncomputable def conversion3 : ℚ :=
  5 * (1/60)  -- to convert minutes to hours

noncomputable def conversion4 : ℚ :=
  1 * (1/100)  -- to convert square centimeters to square decimeters

noncomputable def conversion5 : ℚ :=
  450 * (1/1000)  -- to convert milliliters to liters

theorem conversion_problem : 
  (conversion1 = 7 / 200) ∧ 
  (conversion2 = 53 / 60) ∧ 
  (conversion3 = 1 / 12) ∧ 
  (conversion4 = 1 / 100) ∧ 
  (conversion5 = 9 / 20) :=
by
  sorry

end conversion_problem_l302_302910


namespace meeting_point_distance_l302_302579

theorem meeting_point_distance
  (distance_to_top : ℝ)
  (total_distance : ℝ)
  (jack_start_time : ℝ)
  (jack_uphill_speed : ℝ)
  (jack_downhill_speed : ℝ)
  (jill_uphill_speed : ℝ)
  (jill_downhill_speed : ℝ)
  (meeting_point_distance : ℝ):
  distance_to_top = 5 -> total_distance = 10 -> jack_start_time = 10 / 60 ->
  jack_uphill_speed = 15 -> jack_downhill_speed = 20 ->
  jill_uphill_speed = 16 -> jill_downhill_speed = 22 ->
  meeting_point_distance = 35 / 27 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end meeting_point_distance_l302_302579


namespace total_students_at_competition_l302_302626
-- Import necessary Lean libraries for arithmetic and logic

-- Define the conditions as variables and expressions
namespace ScienceFair

variables (K KKnowItAll KarenHigh NovelCoronaHigh : ℕ)
variables (hK : KKnowItAll = 50)
variables (hKH : KarenHigh = 3 * KKnowItAll / 5)
variables (hNCH : NovelCoronaHigh = 2 * (KKnowItAll + KarenHigh))

-- Define the proof problem
theorem total_students_at_competition (KKnowItAll KarenHigh NovelCoronaHigh : ℕ)
  (hK : KKnowItAll = 50)
  (hKH : KarenHigh = 3 * KKnowItAll / 5)
  (hNCH : NovelCoronaHigh = 2 * (KKnowItAll + KarenHigh)) :
  KKnowItAll + KarenHigh + NovelCoronaHigh = 240 := by
  sorry

end ScienceFair

end total_students_at_competition_l302_302626


namespace find_larger_number_l302_302560

theorem find_larger_number (x y : ℕ) (h1 : x * y = 56) (h2 : x + y = 15) : max x y = 8 :=
sorry

end find_larger_number_l302_302560


namespace find_c_l302_302931

theorem find_c (a b c : ℝ) : 
  (a * x^2 + b * x - 5) * (a * x^2 + b * x + 25) + c = (a * x^2 + b * x + 10)^2 → 
  c = 225 :=
by sorry

end find_c_l302_302931


namespace Arianna_time_at_work_l302_302654

theorem Arianna_time_at_work : 
  (24 - (5 + 13)) = 6 := 
by 
  sorry

end Arianna_time_at_work_l302_302654


namespace units_digit_of_7_pow_6_cubed_l302_302870

-- Define the repeating cycle of unit digits for powers of 7
def unit_digit_of_power_of_7 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 0 -- This case is actually unreachable given the modulus operation

-- Define the main problem statement
theorem units_digit_of_7_pow_6_cubed : unit_digit_of_power_of_7 (6 ^ 3) = 1 :=
by
  sorry

end units_digit_of_7_pow_6_cubed_l302_302870


namespace ksyusha_wednesday_time_l302_302266

def speed_relation (v : ℝ) := ∀ (d_walk d_run1 d_run2 time1 time2: ℝ), 
  (2 * d_run1 = d_walk) → -- she walked twice the distance she ran on Tuesday
  (2 * v = 2 * v) → -- she runs twice as fast as she walks
  ((2 * d_walk / v) + (d_run1 / (2 * v)) = time1) → -- she took 30 minutes on Tuesday
  (time1 = 30) → -- Tuesday time equals 30
  (d_run2 = 2 * d_walk) → -- she had to run twice the distance she walked on Wednesday
  ((d_walk / v) + (d_run2 / (2 * v)) = time2) → -- calculate the time for Wednesday
  time2 = 24 -- time taken on Wednesday

theorem ksyusha_wednesday_time :
  ∃ (v : ℝ), speed_relation v :=
begin
  existsi 1,
  unfold speed_relation,
  intros,
  sorry,
end

end ksyusha_wednesday_time_l302_302266


namespace never_attains_95_l302_302319

def dihedral_angle_condition (α β : ℝ) : Prop :=
  0 < α ∧ 0 < β ∧ α + β < 90

theorem never_attains_95 (α β : ℝ) (h : dihedral_angle_condition α β) :
  α + β ≠ 95 :=
by
  sorry

end never_attains_95_l302_302319


namespace seq_nonzero_l302_302363

def seq (a : ℕ → ℤ) : Prop :=
  (a 1 = 1) ∧ (a 2 = 2) ∧ (∀ n, n ≥ 3 → 
    (if (a (n - 2) * a (n - 1)) % 2 = 0 
     then a n = 5 * a (n - 1) - 3 * a (n - 2) 
     else a n = a (n - 1) - a (n - 2)))

theorem seq_nonzero (a : ℕ → ℤ) (h : seq a) : ∀ n, n > 0 → a n ≠ 0 :=
  sorry

end seq_nonzero_l302_302363


namespace alice_coins_percentage_l302_302782

theorem alice_coins_percentage :
  let penny := 1
  let dime := 10
  let quarter := 25
  let half_dollar := 50
  let total_cents := penny + dime + quarter + half_dollar
  (total_cents / 100) * 100 = 86 :=
by
  sorry

end alice_coins_percentage_l302_302782


namespace speed_in_still_water_l302_302176

-- We define the given conditions for the man's rowing speeds
def upstream_speed : ℕ := 25
def downstream_speed : ℕ := 35

-- We want to prove that the speed in still water is 30 kmph
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 30 := by
  sorry

end speed_in_still_water_l302_302176


namespace intersection_P_Q_l302_302685

-- Define set P
def P : Set ℕ := {x | 1 ≤ x ∧ x ≤ 10}

-- Define set Q (using real numbers, but we will be interested in natural number intersections)
def Q : Set ℝ := {x | x^2 + x - 6 ≤ 0}

-- The intersection of P with Q in the natural numbers should be {1, 2}
theorem intersection_P_Q :
  {x : ℕ | x ∈ P ∧ (x : ℝ) ∈ Q} = {1, 2} :=
by
  sorry

end intersection_P_Q_l302_302685


namespace Ravi_Prakash_finish_together_l302_302979

-- Definitions based on conditions
def Ravi_time := 24
def Prakash_time := 40

-- Main theorem statement
theorem Ravi_Prakash_finish_together :
  (1 / Ravi_time + 1 / Prakash_time) = 1 / 15 :=
by
  sorry

end Ravi_Prakash_finish_together_l302_302979


namespace games_went_this_year_l302_302253

theorem games_went_this_year (t l : ℕ) (h1 : t = 13) (h2 : l = 9) : (t - l = 4) :=
by
  sorry

end games_went_this_year_l302_302253


namespace area_of_remaining_figure_l302_302638
noncomputable def π := Real.pi

theorem area_of_remaining_figure (R : ℝ) (chord_length : ℝ) (C : ℝ) 
  (h : chord_length = 8) (hC : C = R) : (π * R^2 - 2 * π * (R / 2)^2) = 12.57 := by
  sorry

end area_of_remaining_figure_l302_302638


namespace intersection_eq_l302_302220

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x^2 ≤ 2}

theorem intersection_eq : A ∩ B = {-1, 0, 1} := by
  sorry

end intersection_eq_l302_302220


namespace count_distinct_product_divisors_l302_302967

-- Define the properties of 8000 and its divisors
def isDivisor (n d : ℕ) := d > 0 ∧ n % d = 0

def T := {d : ℕ | isDivisor 8000 d}

-- The main statement to prove
theorem count_distinct_product_divisors : 
    (∃ n : ℕ, n ∈ { m | ∃ a b : ℕ, a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ m = a * b } ∧ n = 88) :=
by {
  sorry
}

end count_distinct_product_divisors_l302_302967


namespace segment_EC_length_l302_302807

noncomputable def length_of_segment_EC (a b c : ℕ) (angle_A_deg BC : ℝ) (BD_perp_AC CE_perp_AB : Prop) (angle_DBC_eq_3_angle_ECB : Prop) : ℝ :=
  a * (Real.sqrt b + Real.sqrt c)

theorem segment_EC_length
  (a b c : ℕ)
  (angle_A_deg BC : ℝ)
  (BD_perp_AC CE_perp_AB : Prop)
  (angle_DBC_eq_3_angle_ECB : Prop)
  (h1 : angle_A_deg = 45)
  (h2 : BC = 10)
  (h3 : BD_perp_AC)
  (h4 : CE_perp_AB)
  (h5 : angle_DBC_eq_3_angle_ECB)
  (h6 : length_of_segment_EC a b c angle_A_deg BC BD_perp_AC CE_perp_AB angle_DBC_eq_3_angle_ECB = 5 * (Real.sqrt 3 + Real.sqrt 1)) :
  a + b + c = 9 :=
  by
    sorry

end segment_EC_length_l302_302807


namespace complement_union_l302_302687

-- Definition of the universal set U
def U : Set ℤ := {x | x^2 - 5 * x - 6 ≤ 0}

-- Definition of set A
def A : Set ℤ := {x | x * (2 - x) ≥ 0}

-- Definition of set B
def B : Set ℤ := {1, 2, 3}

-- The proof statement
theorem complement_union (h : U = {x | x^2 - 5 * x - 6 ≤ 0} ∧ 
                           A = {x | x * (2 - x) ≥ 0} ∧ 
                           B = {1, 2, 3}) : 
  U \ (A ∪ B) = {-1, 4, 5, 6} :=
by {
  sorry
}

end complement_union_l302_302687


namespace parallel_vectors_l302_302587

theorem parallel_vectors (m : ℝ) (a b : ℝ × ℝ) (h_a : a = (1, -2)) (h_b : b = (-1, m)) (h_parallel : ∃ k : ℝ, b = k • a) : m = 2 :=
by {
  sorry
}

end parallel_vectors_l302_302587


namespace find_range_of_a_l302_302151

theorem find_range_of_a (a : ℝ) (x : ℝ) (y : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 2 ≤ y ∧ y ≤ 3) 
    (hineq : x * y ≤ a * x^2 + 2 * y^2) : 
    -1 ≤ a := sorry

end find_range_of_a_l302_302151


namespace ratio_julia_bill_l302_302457

variable (B M : ℕ)

def total_miles := B + (B + 4) + M * (B + 4)

theorem ratio_julia_bill (h : total_miles B M = 32) :
  (M * (B + 4)) / (B + 4) = M :=
by sorry

end ratio_julia_bill_l302_302457


namespace value_of_frac_l302_302386

theorem value_of_frac (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_frac_l302_302386


namespace james_pre_injury_miles_600_l302_302334

-- Define the conditions
def james_pre_injury_miles (x : ℝ) : Prop :=
  ∃ goal_increase : ℝ, ∃ days : ℝ, ∃ weekly_increase : ℝ,
  goal_increase = 1.2 * x ∧
  days = 280 ∧
  weekly_increase = 3 ∧
  (days / 7) * weekly_increase = (goal_increase - x)

-- Define the main theorem to be proved
theorem james_pre_injury_miles_600 : james_pre_injury_miles 600 :=
sorry

end james_pre_injury_miles_600_l302_302334


namespace value_of_fraction_l302_302433

-- Definitions based on conditions
variables {a b c d : ℝ}
hypothesis h1 : a = 4 * b
hypothesis h2 : b = 3 * c
hypothesis h3 : c = 5 * d

-- The statement we want to prove
theorem value_of_fraction : (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_fraction_l302_302433


namespace reciprocal_neg_sqrt_2_l302_302618

theorem reciprocal_neg_sqrt_2 : 1 / (-Real.sqrt 2) = -Real.sqrt 2 / 2 :=
by
  sorry

end reciprocal_neg_sqrt_2_l302_302618


namespace longest_diagonal_length_l302_302652

-- Define the conditions
variables {a b : ℝ} (h_area : 135 = 1/2 * a * b) (h_ratio : a / b = 5 / 3)

-- Define the target to prove
theorem longest_diagonal_length (a b : ℝ) (h_area : 135 = 1/2 * a * b) (h_ratio : a / b = 5 / 3) :
    a = 15 * Real.sqrt 2 :=
sorry

end longest_diagonal_length_l302_302652


namespace picnic_total_cost_is_correct_l302_302744

-- Define the conditions given in the problem
def number_of_people : Nat := 4
def cost_per_sandwich : Nat := 5
def cost_per_fruit_salad : Nat := 3
def sodas_per_person : Nat := 2
def cost_per_soda : Nat := 2
def number_of_snack_bags : Nat := 3
def cost_per_snack_bag : Nat := 4

-- Calculate the total cost based on the given conditions
def total_cost_sandwiches : Nat := number_of_people * cost_per_sandwich
def total_cost_fruit_salads : Nat := number_of_people * cost_per_fruit_salad
def total_cost_sodas : Nat := number_of_people * sodas_per_person * cost_per_soda
def total_cost_snack_bags : Nat := number_of_snack_bags * cost_per_snack_bag

def total_spent : Nat := total_cost_sandwiches + total_cost_fruit_salads + total_cost_sodas + total_cost_snack_bags

-- The statement we want to prove
theorem picnic_total_cost_is_correct : total_spent = 60 :=
by
  -- Proof would be written here
  sorry

end picnic_total_cost_is_correct_l302_302744


namespace difference_of_triangular_2010_2009_l302_302516

def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

theorem difference_of_triangular_2010_2009 :
  triangular 2010 - triangular 2009 = 2010 :=
by
  sorry

end difference_of_triangular_2010_2009_l302_302516


namespace math_proof_problem_l302_302365

noncomputable def ellipse_standard_eq (a b : ℝ) : Prop :=
  a = 4 ∧ b = 2 * Real.sqrt 3

noncomputable def conditions (e : ℝ) (vertex : ℝ × ℝ) (p q : ℝ × ℝ) : Prop :=
  e = 1 / 2
  ∧ vertex = (0, 2 * Real.sqrt 3)  -- focus of the parabola
  ∧ p = (-2, -3)
  ∧ q = (-2, 3)

noncomputable def max_area_quadrilateral (area : ℝ) : Prop :=
  area = 12 * Real.sqrt 3

theorem math_proof_problem : 
  ∃ a b p q area, ellipse_standard_eq a b ∧ conditions (1/2) (0, 2 * Real.sqrt 3) p q 
  ∧ p = (-2, -3) ∧ q = (-2, 3) → max_area_quadrilateral area := 
  sorry

end math_proof_problem_l302_302365


namespace simplify_sqrt_450_l302_302025

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l302_302025


namespace value_of_fraction_l302_302429

-- Definitions based on conditions
variables {a b c d : ℝ}
hypothesis h1 : a = 4 * b
hypothesis h2 : b = 3 * c
hypothesis h3 : c = 5 * d

-- The statement we want to prove
theorem value_of_fraction : (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_fraction_l302_302429


namespace sqrt_450_simplified_l302_302052

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l302_302052


namespace johns_elevation_after_descent_l302_302707

def starting_elevation : ℝ := 400
def rate_of_descent : ℝ := 10
def travel_time : ℝ := 5

theorem johns_elevation_after_descent :
  starting_elevation - (rate_of_descent * travel_time) = 350 :=
by
  sorry

end johns_elevation_after_descent_l302_302707


namespace simplify_expression_l302_302519

variable (a b : ℕ)

theorem simplify_expression (a b : ℕ) : 5 * a * b - 7 * a * b + 3 * a * b = a * b := by
  sorry

end simplify_expression_l302_302519


namespace problem_solution_l302_302544

def arithmetic_sequence (a_1 : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a_1 + (n - 1) * d

def sum_of_terms (a_1 : ℕ) (a_n : ℕ) (n : ℕ) : ℕ :=
  n * (a_1 + a_n) / 2

theorem problem_solution 
  (a_1 : ℕ) (d : ℕ) (a_n : ℕ) (S_n : ℕ)
  (h1 : a_1 = 2)
  (h2 : S_2 = arithmetic_sequence a_1 d 3):
  a_2 = 4 ∧ S_10 = 110 :=
by
  sorry

end problem_solution_l302_302544


namespace crayons_lost_or_given_away_l302_302281

theorem crayons_lost_or_given_away (P E L : ℕ) (h1 : P = 479) (h2 : E = 134) (h3 : L = P - E) : L = 345 :=
by
  rw [h1, h2] at h3
  exact h3

end crayons_lost_or_given_away_l302_302281


namespace simplify_sqrt_450_l302_302102

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l302_302102


namespace train_passing_time_l302_302877

theorem train_passing_time (length_of_train : ℝ) (speed_of_train_kmhr : ℝ) :
  length_of_train = 180 → speed_of_train_kmhr = 36 → (length_of_train / (speed_of_train_kmhr * (1000 / 3600))) = 18 :=
by
  intro h1 h2
  sorry

end train_passing_time_l302_302877


namespace arithmetic_common_difference_l302_302543

-- Define the conditions of the arithmetic sequence
def a (n : ℕ) := 0 -- This is a placeholder definition since we only care about a_5 and a_12
def a5 : ℝ := 10
def a12 : ℝ := 31

-- State the proof problem
theorem arithmetic_common_difference :
  ∃ d : ℝ, a5 + 7 * d = a12 :=
by
  use 3
  simp [a5, a12]
  sorry

end arithmetic_common_difference_l302_302543


namespace flower_bed_length_l302_302856

theorem flower_bed_length (a b : ℝ) :
  ∀ width : ℝ, (6 * a^2 - 4 * a * b + 2 * a = 2 * a * width) → width = 3 * a - 2 * b + 1 :=
by
  intros width h
  sorry

end flower_bed_length_l302_302856


namespace sum_of_first_15_terms_is_largest_l302_302955

theorem sum_of_first_15_terms_is_largest
  (a : ℕ → ℝ)
  (s : ℕ → ℝ)
  (d : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_sum : ∀ n, s n = n * a 1 + (n * (n - 1) * d) / 2)
  (h1: 13 * a 6 = 19 * (a 6 + 3 * d))
  (h2: a 1 > 0) : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≠ 15 → s 15 > s n :=
by
  sorry

end sum_of_first_15_terms_is_largest_l302_302955


namespace union_M_N_l302_302570

def M : Set ℝ := { x | -1 < x ∧ x < 3 }
def N : Set ℝ := { x | x ≥ 1 }

theorem union_M_N : M ∪ N = { x | x > -1 } := 
by sorry

end union_M_N_l302_302570


namespace number_of_schools_l302_302198

theorem number_of_schools (N : ℕ) :
  (∀ i j : ℕ, i < j → i ≠ j) →
  (∀ i : ℕ, i < 2 * 35 → i = 35 ∨ ((i = 37 → ¬ (i = 35))) ∧ ((i = 64 → ¬ (i = 35)))) →
  N = (2 * (35) - 1) / 3 →
  N = 23 :=
by
  sorry

end number_of_schools_l302_302198


namespace negation_of_proposition_l302_302295

theorem negation_of_proposition : 
  (¬ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0)) ↔ (∃ x : ℝ, x^2 + 2*x + 5 = 0) :=
by sorry

end negation_of_proposition_l302_302295


namespace four_digit_even_numbers_divisible_by_4_l302_302940

noncomputable def number_of_4_digit_even_numbers_divisible_by_4 : Nat :=
  500

theorem four_digit_even_numbers_divisible_by_4 : 
  (∃ count : Nat, count = number_of_4_digit_even_numbers_divisible_by_4) :=
sorry

end four_digit_even_numbers_divisible_by_4_l302_302940


namespace problem_statement_l302_302218

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f (x)

theorem problem_statement (f : ℝ → ℝ) :
  is_odd_function f →
  (∀ x : ℝ, f (x + 6) = f (x) + 3) →
  f 1 = 1 →
  f 2015 + f 2016 = 2015 :=
by
  sorry

end problem_statement_l302_302218


namespace sqrt_450_simplified_l302_302041

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l302_302041


namespace sqrt_450_equals_15_sqrt_2_l302_302080

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l302_302080


namespace exponent_multiplication_l302_302190

-- Define the variables and exponentiation property
variable (a : ℝ)

-- State the theorem
theorem exponent_multiplication : a^3 * a^2 = a^5 := by
  sorry

end exponent_multiplication_l302_302190


namespace sqrt_450_simplified_l302_302049

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l302_302049


namespace increasing_function_range_l302_302362

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x + 1 else a^x

theorem increasing_function_range (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ 2 ≤ a ∧ a < 3 :=
  sorry

end increasing_function_range_l302_302362


namespace intersection_nonempty_iff_l302_302713

/-- Define sets A and B as described in the problem. -/
def A (x : ℝ) : Prop := -2 < x ∧ x ≤ 1
def B (x : ℝ) (k : ℝ) : Prop := x ≥ k

/-- The main theorem to prove the range of k where the intersection of A and B is non-empty. -/
theorem intersection_nonempty_iff (k : ℝ) : (∃ x, A x ∧ B x k) ↔ k ≤ 1 :=
by
  sorry

end intersection_nonempty_iff_l302_302713


namespace total_number_of_girls_is_13_l302_302288

def number_of_girls (n : ℕ) (B : ℕ) : Prop :=
  ∃ A : ℕ, (A = B - 5) ∧ (A = B + 8)

theorem total_number_of_girls_is_13 (n : ℕ) (B : ℕ) :
  number_of_girls n B → n = 13 :=
by
  intro h
  sorry

end total_number_of_girls_is_13_l302_302288


namespace concrete_pillars_l302_302963

-- Definitions based on the conditions of the problem
def C_deck : ℕ := 1600
def C_anchor : ℕ := 700
def C_total : ℕ := 4800

-- Theorem to prove the concrete required for supporting pillars
theorem concrete_pillars : C_total - (C_deck + 2 * C_anchor) = 1800 :=
by sorry

end concrete_pillars_l302_302963


namespace halfway_between_frac_l302_302206

theorem halfway_between_frac : (1 / 7 + 1 / 9) / 2 = 8 / 63 := by
  sorry

end halfway_between_frac_l302_302206


namespace price_alloy_per_kg_l302_302507

-- Defining the costs of the two metals.
def cost_metal1 : ℝ := 68
def cost_metal2 : ℝ := 96

-- Defining the mixture ratio.
def ratio : ℝ := 1

-- The proposition that the price per kg of the alloy is 82 Rs.
theorem price_alloy_per_kg (C1 C2 r : ℝ) (hC1 : C1 = 68) (hC2 : C2 = 96) (hr : r = 1) :
  (C1 + C2) / (r + r) = 82 :=
by
  sorry

end price_alloy_per_kg_l302_302507


namespace roots_square_sum_l302_302195

theorem roots_square_sum {a b c : ℝ} (h1 : 3 * a^3 + 2 * a^2 - 3 * a - 8 = 0)
                                  (h2 : 3 * b^3 + 2 * b^2 - 3 * b - 8 = 0)
                                  (h3 : 3 * c^3 + 2 * c^2 - 3 * c - 8 = 0)
                                  (sum_roots : a + b + c = -2/3)
                                  (product_pairs : a * b + b * c + c * a = -1) : 
  a^2 + b^2 + c^2 = 22 / 9 := by
  sorry

end roots_square_sum_l302_302195


namespace E_eq_F_l302_302686

noncomputable def E : Set ℝ := { x | ∃ n : ℤ, x = Real.cos (n * Real.pi / 3) }

noncomputable def F : Set ℝ := { x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6) }

theorem E_eq_F : E = F := 
sorry

end E_eq_F_l302_302686


namespace line_through_point_intersects_yaxis_triangular_area_l302_302773

theorem line_through_point_intersects_yaxis_triangular_area 
  (a T : ℝ) 
  (h : 0 < a) 
  (line_eqn : ∀ x y : ℝ, x = -a * y + a → 2 * T * x + a^2 * y - 2 * a * T = 0) 
  : ∃ (m b : ℝ), (forall x y : ℝ, y = m * x + b) := 
by
  sorry

end line_through_point_intersects_yaxis_triangular_area_l302_302773


namespace polynomial_divisible_by_square_l302_302976

def f (x : ℝ) (a1 a2 a3 a4 : ℝ) : ℝ := x^4 + a1 * x^3 + a2 * x^2 + a3 * x + a4
def f' (x : ℝ) (a1 a2 a3 : ℝ) : ℝ := 4 * x^3 + 3 * a1 * x^2 + 2 * a2 * x + a3

theorem polynomial_divisible_by_square (x0 a1 a2 a3 a4 : ℝ) 
  (h1 : f x0 a1 a2 a3 a4 = 0) 
  (h2 : f' x0 a1 a2 a3 = 0) : 
  ∃ g : ℝ → ℝ, ∀ x : ℝ, f x a1 a2 a3 a4 = (x - x0)^2 * (g x) :=
sorry

end polynomial_divisible_by_square_l302_302976


namespace integral_even_odd_l302_302946

open Real

theorem integral_even_odd (a : ℝ) :
  (∫ x in -a..a, x^2 + sin x) = 18 → a = 3 :=
by
  intros h
  -- We'll skip the proof
  sorry

end integral_even_odd_l302_302946


namespace toys_gained_l302_302774

theorem toys_gained
  (sp : ℕ) -- selling price of 18 toys
  (cp_per_toy : ℕ) -- cost price per toy
  (sp_val : sp = 27300) -- given selling price value
  (cp_per_val : cp_per_toy = 1300) -- given cost price per toy value
  : (sp - 18 * cp_per_toy) / cp_per_toy = 3 := by
  -- Conditions of the problem are stated
  -- Proof is omitted with 'sorry'
  sorry

end toys_gained_l302_302774


namespace ratio_problem_l302_302821

theorem ratio_problem 
  (x y z w : ℚ) 
  (h1 : x / y = 12) 
  (h2 : z / y = 4) 
  (h3 : z / w = 3 / 4) : 
  w / x = 4 / 9 := 
  sorry

end ratio_problem_l302_302821


namespace first_number_in_sum_l302_302498

theorem first_number_in_sum (a b c : ℝ) (h : a + b + c = 3.622) : a = 3.15 :=
by
  -- Assume the given values of b and c
  have hb : b = 0.014 := sorry
  have hc : c = 0.458 := sorry
  -- From the assumption h and hb, hc, we deduce a = 3.15
  sorry

end first_number_in_sum_l302_302498


namespace value_of_fraction_l302_302947

open Real

theorem value_of_fraction (a : ℝ) (h : a^2 + a - 1 = 0) : (1 - a) / a + a / (1 + a) = 1 := 
by { sorry }

end value_of_fraction_l302_302947


namespace number_of_schools_is_23_l302_302199

-- Conditions and definitions
noncomputable def number_of_students_per_school : ℕ := 3
def beth_rank : ℕ := 37
def carla_rank : ℕ := 64

-- Statement of the proof problem
theorem number_of_schools_is_23
  (n : ℕ)
  (h1 : ∀ i < n, ∃ r1 r2 r3: ℕ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3)
  (h2 : ∀ i < n, ∃ A B C: ℕ, A = (2 * B + 1) ∧ C = A ∧ B = 35 ∧ A < beth_rank ∧ beth_rank < carla_rank):
  n = 23 :=
by
  sorry

end number_of_schools_is_23_l302_302199


namespace center_of_circumcircle_lies_on_AK_l302_302642

variable {α β γ : Real} -- Angles in triangle ABC
variable (A B C L H K O : Point) -- Points in the configuration
variable (circumcircle_ABC : TriangularCircumcircle A B C) -- Circumcircle of triangle ABC

-- Definitions based on the given conditions
variable (is_angle_bisector : angle_bisector A B C L)
variable (is_height : height_from_point_to_line B A L H)
variable (intersects_circle_at_K : intersects_circumcircle A B L K circumcircle_ABC)
variable (is_circumcenter : circumcenter A B C O circumcircle_ABC)

theorem center_of_circumcircle_lies_on_AK
  (h_angle_bisector : is_angle_bisector)
  (h_height : is_height)
  (h_intersects_circle_at_K : intersects_circle_at_K)
  (h_circumcenter : is_circumcenter) 
    : lies_on_line O A K := 
sorry -- Proof is omitted

end center_of_circumcircle_lies_on_AK_l302_302642


namespace quadratic_function_expression_quadratic_function_range_quadratic_function_m_range_l302_302811

-- Part 1: Expression of the quadratic function
theorem quadratic_function_expression (a : ℝ) (h : a = 0) : 
  ∀ x, (x^2 + (a-2)*x + 3) = x^2 - 2*x + 3 :=
by sorry

-- Part 2: Range of y for 0 < x < 3
theorem quadratic_function_range (x y : ℝ) (h : ∀ x, y = x^2 - 2*x + 3) (hx : 0 < x ∧ x < 3) :
  2 ≤ y ∧ y < 6 :=
by sorry

-- Part 3: Range of m for y1 > y2
theorem quadratic_function_m_range (m y1 y2 : ℝ) (P Q : ℝ × ℝ)
  (h1 : P = (m - 1, y1)) (h2 : Q = (m, y2)) (h3 : y1 > y2) :
  m < 3 / 2 :=
by sorry

end quadratic_function_expression_quadratic_function_range_quadratic_function_m_range_l302_302811


namespace fourth_person_height_l302_302619

theorem fourth_person_height 
  (height1 height2 height3 height4 : ℝ)
  (diff12 : height2 = height1 + 2)
  (diff23 : height3 = height2 + 2)
  (diff34 : height4 = height3 + 6)
  (avg_height : (height1 + height2 + height3 + height4) / 4 = 76) :
  height4 = 82 :=
by
  sorry

end fourth_person_height_l302_302619


namespace gain_percentage_is_30_l302_302785

def sellingPrice : ℕ := 195
def gain : ℕ := 45
def costPrice : ℕ := sellingPrice - gain

def gainPercentage : ℚ := (gain : ℚ) / (costPrice : ℚ) * 100

theorem gain_percentage_is_30 :
  gainPercentage = 30 := 
sorry

end gain_percentage_is_30_l302_302785


namespace find_f_1991_l302_302712

namespace FunctionProof

-- Defining the given conditions as statements in Lean
def func_f (f : ℤ → ℤ) : Prop :=
  ∀ m n : ℤ, f (m + f (f n)) = -f (f (m + 1)) - n

def poly_g (f g : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, g n = g (f n)

-- Statement of the problem
theorem find_f_1991 
  (f g : ℤ → ℤ)
  (Hf : func_f f)
  (Hg : poly_g f g) :
  f 1991 = -1992 := 
sorry

end FunctionProof

end find_f_1991_l302_302712


namespace fractions_are_integers_l302_302970

theorem fractions_are_integers (a b c : ℤ) (h : ∃ k : ℤ, (a * b / c) + (a * c / b) + (b * c / a) = k) :
  ∃ k1 k2 k3 : ℤ, (a * b / c) = k1 ∧ (a * c / b) = k2 ∧ (b * c / a) = k3 :=
by
  sorry

end fractions_are_integers_l302_302970


namespace probability_of_rain_on_both_days_l302_302697

variable {Ω : Type} {P : Measure Ω} 
variable {M T N : {x : Ω // True}}

-- Defining the probabilities based on the given conditions
variable (pM pT pN pMcapT : ℝ)
variable (hpM : P M = 0.62)
variable (hpT : P T = 0.54)
variable (hpN : P N = 0.28)

-- Statement for the problem
theorem probability_of_rain_on_both_days :
  pMcapT = (pM + pT - (1 - pN)) :=
sorry

end probability_of_rain_on_both_days_l302_302697


namespace total_votes_l302_302161

-- Conditions
variables (V : ℝ)
def candidate_votes := 0.31 * V
def rival_votes := 0.31 * V + 2451

-- Problem statement
theorem total_votes (h : candidate_votes V + rival_votes V = V) : V = 6450 :=
sorry

end total_votes_l302_302161


namespace field_area_l302_302325

theorem field_area (L W : ℝ) (hL : L = 20) (h_fencing : 2 * W + L = 59) :
  L * W = 390 :=
by {
  -- We will skip the proof
  sorry
}

end field_area_l302_302325


namespace range_of_a_l302_302680

open Real

noncomputable def f (x : ℝ) := x - sqrt (x^2 + x)

noncomputable def g (x a : ℝ) := log x / log 27 - log x / log 9 + a * log x / log 3

theorem range_of_a (a : ℝ) : (∀ x1 ∈ Set.Ioi 1, ∃ x2 ∈ Set.Icc 3 9, f x1 > g x2 a) → a ≤ -1/12 :=
by
  intro h
  sorry

end range_of_a_l302_302680


namespace simplify_sqrt_450_l302_302121

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l302_302121


namespace total_length_of_ribbon_l302_302851

-- Define the conditions
def length_per_piece : ℕ := 73
def number_of_pieces : ℕ := 51

-- The theorem to prove
theorem total_length_of_ribbon : length_per_piece * number_of_pieces = 3723 :=
by
  sorry

end total_length_of_ribbon_l302_302851


namespace simplify_sqrt_450_l302_302103

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l302_302103


namespace geom_seq_sum_first_eight_l302_302538

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

end geom_seq_sum_first_eight_l302_302538


namespace sqrt_simplify_l302_302032

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l302_302032


namespace infinite_geometric_series_sum_l302_302968

theorem infinite_geometric_series_sum (p q : ℝ)
  (h : (∑' n : ℕ, p / q ^ (n + 1)) = 5) :
  (∑' n : ℕ, p / (p^2 + q) ^ (n + 1)) = 5 * (q - 1) / (25 * q^2 - 50 * q + 26) :=
sorry

end infinite_geometric_series_sum_l302_302968


namespace water_outflow_time_l302_302777

theorem water_outflow_time (H R : ℝ) (flow_rate : ℝ → ℝ)
  (h_initial : ℝ) (t_initial : ℝ) (empty_height : ℝ) :
  H = 12 →
  R = 3 →
  (∀ h, flow_rate h = -h) →
  h_initial = 12 →
  t_initial = 0 →
  empty_height = 0 →
  ∃ t, t = (72 : ℝ) * π / 16 :=
by
  intros hL R_eq flow_rate_eq h_initial_eq t_initial_eq empty_height_eq
  sorry

end water_outflow_time_l302_302777


namespace solution_set_ineq_l302_302355

theorem solution_set_ineq (x : ℝ) :
  (x - 1) / (1 - 2 * x) ≥ 0 ↔ (1 / 2 < x ∧ x ≤ 1) :=
by
  sorry

end solution_set_ineq_l302_302355


namespace simplify_sqrt_450_l302_302095

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l302_302095


namespace parabola_focus_coordinates_l302_302736

theorem parabola_focus_coordinates :
  (∃ f : ℝ × ℝ, f = (0, 2) ∧ ∀ x y : ℝ, y = (1/8) * x^2 ↔ f = (0, 2)) :=
sorry

end parabola_focus_coordinates_l302_302736


namespace prove_p_and_q_l302_302368

def p (m : ℝ) : Prop :=
  (∀ x : ℝ, x^2 + x + m > 0) → m > 1 / 4

def q (A B : ℝ) : Prop :=
  A > B ↔ Real.sin A > Real.sin B

theorem prove_p_and_q :
  (∀ m : ℝ, p m) ∧ (∀ A B : ℝ, q A B) :=
by
  sorry

end prove_p_and_q_l302_302368


namespace P_sufficient_but_not_necessary_for_Q_l302_302210

-- Define the propositions P and Q
def P (x : ℝ) : Prop := |x - 2| ≤ 3
def Q (x : ℝ) : Prop := x ≥ -1 ∨ x ≤ 5

-- Define the statement to prove
theorem P_sufficient_but_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ (∃ x : ℝ, Q x ∧ ¬ P x) :=
by
  sorry

end P_sufficient_but_not_necessary_for_Q_l302_302210


namespace regular_17gon_symmetries_l302_302528

theorem regular_17gon_symmetries : 
  let L := 17
  let R := 360 / 17
  L + R = 17 + 360 / 17 :=
by
  sorry

end regular_17gon_symmetries_l302_302528


namespace div_recurring_decimal_l302_302305

def recurringDecimalToFraction (q : ℚ) (h : q = 36/99) : ℚ := by
  sorry

theorem div_recurring_decimal : 12 / recurringDecimalToFraction 0.36 sorry = 33 :=
by
  sorry

end div_recurring_decimal_l302_302305


namespace solve_for_q_l302_302564

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : q = -25 / 11 :=
by
  sorry

end solve_for_q_l302_302564


namespace qualifying_rate_l302_302741

theorem qualifying_rate (a b : ℝ) (h1 : 0 ≤ a ∧ a ≤ 1) (h2 : 0 ≤ b ∧ b ≤ 1) :
  (1 - a) * (1 - b) = 1 - a - b + a * b :=
by sorry

end qualifying_rate_l302_302741


namespace total_robots_correct_l302_302150

def number_of_shapes : ℕ := 3
def number_of_colors : ℕ := 4
def total_types_of_robots : ℕ := number_of_shapes * number_of_colors

theorem total_robots_correct : total_types_of_robots = 12 := by
  sorry

end total_robots_correct_l302_302150


namespace equalize_nuts_l302_302481

open Nat

noncomputable def transfer (p1 p2 p3 : ℕ) : Prop :=
  ∃ (m1 m2 m3 : ℕ), 
    m1 ≤ p1 ∧ m1 ≤ p2 ∧ 
    m2 ≤ (p2 + m1) ∧ m2 ≤ p3 ∧ 
    m3 ≤ (p3 + m2) ∧ m3 ≤ (p1 - m1) ∧
    (p1 - m1 + m3 = 16) ∧ 
    (p2 + m1 - m2 = 16) ∧ 
    (p3 + m2 - m3 = 16)

theorem equalize_nuts : transfer 22 14 12 := 
  sorry

end equalize_nuts_l302_302481


namespace simplify_sqrt_450_l302_302107

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l302_302107


namespace num_ways_to_make_change_l302_302559

-- Define the standard U.S. coins
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Define the total amount
def total_amount : ℕ := 50

-- Condition to exclude two quarters
def valid_combination (num_pennies num_nickels num_dimes num_quarters : ℕ) : Prop :=
  (num_quarters != 2) ∧ (num_pennies + 5 * num_nickels + 10 * num_dimes + 25 * num_quarters = total_amount)

-- Prove that there are 39 ways to make change for 50 cents
theorem num_ways_to_make_change : 
  ∃ count : ℕ, count = 39 ∧ (∀ 
    (num_pennies num_nickels num_dimes num_quarters : ℕ),
    valid_combination num_pennies num_nickels num_dimes num_quarters → 
    (num_pennies, num_nickels, num_dimes, num_quarters) = count) :=
sorry

end num_ways_to_make_change_l302_302559


namespace sqrt_450_eq_15_sqrt_2_l302_302073

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l302_302073


namespace total_items_sold_at_garage_sale_l302_302517

-- Define the conditions for the problem
def items_more_expensive_than_radio : Nat := 16
def items_less_expensive_than_radio : Nat := 23

-- Declare the total number of items using the given conditions
theorem total_items_sold_at_garage_sale 
  (h1 : items_more_expensive_than_radio = 16)
  (h2 : items_less_expensive_than_radio = 23) :
  items_more_expensive_than_radio + 1 + items_less_expensive_than_radio = 40 :=
by
  sorry

end total_items_sold_at_garage_sale_l302_302517


namespace minimize_sum_areas_l302_302896

theorem minimize_sum_areas (x : ℝ) (h_wire_length : 0 < x ∧ x < 1) :
    let side_length := x / 4
    let square_area := (side_length ^ 2)
    let circle_radius := (1 - x) / (2 * Real.pi)
    let circle_area := Real.pi * (circle_radius ^ 2)
    let total_area := square_area + circle_area
    total_area = (x^2 / 16 + (1 - x)^2 / (4 * Real.pi)) -> 
    x = Real.pi / (Real.pi + 4) :=
by
  sorry

end minimize_sum_areas_l302_302896


namespace area_triangle_AMC_l302_302701

open Real

-- Definitions: Define the points A, B, C, D such that they form a rectangle
-- Define midpoint M of \overline{AD}

structure Point :=
(x : ℝ)
(y : ℝ)

noncomputable def A : Point := {x := 0, y := 0}
noncomputable def B : Point := {x := 6, y := 0}
noncomputable def D : Point := {x := 0, y := 8}
noncomputable def C : Point := {x := 6, y := 8}
noncomputable def M : Point := {x := 0, y := 4} -- midpoint of AD

-- Function to compute the area of triangle AMC
noncomputable def triangle_area (A M C : Point) : ℝ :=
  (1 / 2 : ℝ) * abs ((A.x - C.x) * (M.y - A.y) - (A.x - M.x) * (C.y - A.y))

-- The theorem to prove
theorem area_triangle_AMC : triangle_area A M C = 12 :=
by
  sorry

end area_triangle_AMC_l302_302701


namespace find_k_l302_302302

noncomputable def proof_problem (x1 x2 x3 x4 : ℝ) (k : ℝ) : Prop :=
  (x1 + x2) / (x3 + x4) = k ∧
  (x3 + x4) / (x1 + x2) = k ∧
  (x1 + x3) / (x2 + x4) = k ∧
  (x2 + x4) / (x1 + x3) = k ∧
  (x1 + x4) / (x2 + x3) = k ∧
  (x2 + x3) / (x1 + x4) = k ∧
  x1 ≠ x2 ∨ x2 ≠ x3 ∨ x3 ≠ x4 ∨ x4 ≠ x1

theorem find_k (x1 x2 x3 x4 : ℝ) (h : proof_problem x1 x2 x3 x4 k) : k = -1 :=
  sorry

end find_k_l302_302302


namespace smallest_integer_representation_l302_302157

theorem smallest_integer_representation :
  ∃ (A B C : ℕ), 0 ≤ A ∧ A < 5 ∧ 0 ≤ B ∧ B < 7 ∧ 0 ≤ C ∧ C < 4 ∧ 6 * A = 8 * B ∧ 6 * A = 5 * C ∧ 8 * B = 5 * C ∧ (6 * A) = 24 :=
  sorry

end smallest_integer_representation_l302_302157


namespace compare_nsquare_pow2_pos_int_l302_302929

-- Proposition that captures the given properties of comparing n^2 and 2^n
theorem compare_nsquare_pow2_pos_int (n : ℕ) (hn : n > 0) : 
  (n = 1 → n^2 < 2^n) ∧
  (n = 2 → n^2 = 2^n) ∧
  (n = 3 → n^2 > 2^n) ∧
  (n = 4 → n^2 = 2^n) ∧
  (n ≥ 5 → n^2 < 2^n) :=
by
  sorry

end compare_nsquare_pow2_pos_int_l302_302929


namespace ksyusha_travel_time_wednesday_l302_302265

-- Definitions based on the conditions
def speed_walk (v : ℝ) : ℝ := v
def speed_run (v : ℝ) : ℝ := 2 * v
def distance_walk_tuesday (S : ℝ) : ℝ := 2 * S
def distance_run_tuesday (S : ℝ) : ℝ := S
def total_time_tuesday (v S : ℝ) : ℝ := (distance_walk_tuesday S) / (speed_walk v) + (distance_run_tuesday S) / (speed_run v)

-- Theorem statement
theorem ksyusha_travel_time_wednesday (S v : ℝ) (h : total_time_tuesday v S = 30) :
  (S / v + S / v) = 24 := 
by sorry

end ksyusha_travel_time_wednesday_l302_302265


namespace kelvin_classes_l302_302556

theorem kelvin_classes (c : ℕ) (h1 : Grant = 4 * c) (h2 : c + Grant = 450) : c = 90 :=
by sorry

end kelvin_classes_l302_302556


namespace distance_between_tangent_and_parallel_line_l302_302340

noncomputable def distance_between_parallel_lines 
  (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (r : ℝ) (M : ℝ × ℝ) 
  (l : ℝ × ℝ → Prop) (a : ℝ) (l1 : ℝ × ℝ → Prop) : ℝ :=
sorry

variable (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (r : ℝ) (M : ℝ × ℝ)
variable (l : ℝ × ℝ → Prop) (a : ℝ) (l1 : ℝ × ℝ → Prop)

axiom tangent_line_at_point (M : ℝ × ℝ) (C : Set (ℝ × ℝ)) : (ℝ × ℝ → Prop)

theorem distance_between_tangent_and_parallel_line
  (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (r : ℝ) (M : ℝ × ℝ)
  (l : ℝ × ℝ → Prop) (a : ℝ) (l1 : ℝ × ℝ → Prop) :
  C = { p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2 } →
  M = (-2, 4) →
  l = tangent_line_at_point M C →
  l1 = { p | a * p.1 + 3 * p.2 + 2 * a = 0 } →
  distance_between_parallel_lines C center r M l a l1 = 12/5 :=
by
  intros hC hM hl hl1
  sorry

end distance_between_tangent_and_parallel_line_l302_302340


namespace necessary_but_not_sufficient_l302_302674

theorem necessary_but_not_sufficient {a b c d : ℝ} (hcd : c > d) : 
  (a - c > b - d) → (a > b) ∧ ¬((a > b) → (a - c > b - d)) :=
by
  sorry

end necessary_but_not_sufficient_l302_302674


namespace systematic_sampling_fourth_group_number_l302_302196

theorem systematic_sampling_fourth_group_number (n : ℕ) (step_size : ℕ) (first_number : ℕ) : 
  n = 4 → step_size = 6 → first_number = 4 → (first_number + step_size * 3) = 22 :=
by
  intros h_n h_step_size h_first_number
  sorry

end systematic_sampling_fourth_group_number_l302_302196


namespace union_sets_l302_302571

open Set

variable (α : Type) [LinearOrder α]

-- Definitions
def M : Set α := { x | -1 < x ∧ x < 3 }
def N : Set α := { x | 1 ≤ x }

-- Theorem statement
theorem union_sets : M α ∪ N α = { x | -1 < x } := sorry

end union_sets_l302_302571


namespace total_reduction_500_l302_302885

noncomputable def total_price_reduction (P : ℝ) (first_reduction_percent : ℝ) (second_reduction_percent : ℝ) : ℝ :=
  let first_reduction := P * first_reduction_percent / 100
  let intermediate_price := P - first_reduction
  let second_reduction := intermediate_price * second_reduction_percent / 100
  let final_price := intermediate_price - second_reduction
  P - final_price

theorem total_reduction_500 (P : ℝ) (first_reduction_percent : ℝ)  (second_reduction_percent: ℝ) (h₁ : P = 500) (h₂ : first_reduction_percent = 5) (h₃ : second_reduction_percent = 4):
  total_price_reduction P first_reduction_percent second_reduction_percent = 44 := 
by
  sorry

end total_reduction_500_l302_302885


namespace max_value_neg_expr_l302_302948

theorem max_value_neg_expr (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 1) :
  - (1 / (2 * a)) - (2 / b) ≤ - (9 / 2) :=
by 
  sorry

end max_value_neg_expr_l302_302948


namespace solve_m_eq_4_l302_302213

noncomputable def perpendicular_vectors (m : ℝ) : Prop :=
  let a := (1, -2)
  let b := (m, m - 2)
  (a.1 * b.1 + a.2 * b.2 = 0)

theorem solve_m_eq_4 : ∀ (m : ℝ), perpendicular_vectors m → m = 4 :=
by
  intros m h
  sorry

end solve_m_eq_4_l302_302213


namespace minimum_frac_seq_l302_302216

noncomputable def seq (n : ℕ) : ℝ :=
  match n with
  | 0 => 0  -- Define for n=0
  | 1 => 33
  | (k+2) => seq (k+1) + 2*(k+1)

theorem minimum_frac_seq (n : ℕ) (h_pos : n > 0) :
  ∃ m, (m ∈ { n | n > 0 }) ∧ (∀ k, k > 0 → (2 * seq k) / k ≥ (2 * seq m) /m) ∧ (2 * seq m / m) = 22 :=
sorry

end minimum_frac_seq_l302_302216


namespace cookies_guests_l302_302518

theorem cookies_guests (cc_cookies : ℕ) (oc_cookies : ℕ) (sc_cookies : ℕ) (cc_per_guest : ℚ) (oc_per_guest : ℚ) (sc_per_guest : ℕ)
    (cc_total : cc_cookies = 45) (oc_total : oc_cookies = 62) (sc_total : sc_cookies = 38) (cc_ratio : cc_per_guest = 1.5)
    (oc_ratio : oc_per_guest = 2.25) (sc_ratio : sc_per_guest = 1) :
    (cc_cookies / cc_per_guest) ≥ 0 ∧ (oc_cookies / oc_per_guest) ≥ 0 ∧ (sc_cookies / sc_per_guest) ≥ 0 → 
    Nat.floor (oc_cookies / oc_per_guest) = 27 :=
by
  sorry

end cookies_guests_l302_302518


namespace percentage_reduction_is_correct_l302_302612

-- Definitions and initial conditions
def initial_price_per_model := 100
def models_for_kindergarten := 2
def models_for_elementary := 2 * models_for_kindergarten
def total_models := models_for_kindergarten + models_for_elementary
def total_cost_without_reduction := total_models * initial_price_per_model
def total_cost_paid := 570

-- Goal statement in Lean 4
theorem percentage_reduction_is_correct :
  (total_models > 5) →
  total_cost_paid = 570 →
  models_for_kindergarten = 2 →
  (total_cost_without_reduction - total_cost_paid) / total_models / initial_price_per_model * 100 = 5 :=
by
  -- sorry to skip the proof
  sorry

end percentage_reduction_is_correct_l302_302612


namespace scaling_transformation_l302_302753

theorem scaling_transformation (a b : ℝ) :
  (∀ x y : ℝ, (y = 1 - x → y' = b * (1 - x))
    → (y' = b - b * x)) 
  ∧
  (∀ x' y' : ℝ, (y = (2 / 3) * x' + 2)
    → (y' = (2 / 3) * (a * x) + 2))
  → a = 3 ∧ b = 2 := by
  sorry

end scaling_transformation_l302_302753


namespace number_b_is_three_times_number_a_l302_302742

theorem number_b_is_three_times_number_a (A B : ℕ) (h1 : A = 612) (h2 : B = 3 * A) : B = 1836 :=
by
  -- This is where the proof would go
  sorry

end number_b_is_three_times_number_a_l302_302742


namespace find_matrix_calculate_M5_alpha_l302_302669

-- Define the matrix M, eigenvalues, eigenvectors and vector α
def M : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 2], ![3, 2]]
def alpha : Fin 2 → ℝ := ![-1, 1]
def e1 : Fin 2 → ℝ := ![2, 3]
def e2 : Fin 2 → ℝ := ![1, -1]
def lambda1 : ℝ := 4
def lambda2 : ℝ := -1

-- Conditions: eigenvalues and their corresponding eigenvectors
axiom h1 : M.mulVec e1 = lambda1 • e1
axiom h2 : M.mulVec e2 = lambda2 • e2

-- Condition: given vector α
axiom h3 : alpha = - e2

-- Prove that M is the matrix given by the components
theorem find_matrix : M = ![![1, 2], ![3, 2]] :=
sorry

-- Prove that M^5 times α equals the given vector
theorem calculate_M5_alpha : (M^5).mulVec alpha = ![-1, 1] :=
sorry

end find_matrix_calculate_M5_alpha_l302_302669


namespace value_of_frac_l302_302385

theorem value_of_frac (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_frac_l302_302385


namespace process_cannot_continue_indefinitely_l302_302859

theorem process_cannot_continue_indefinitely (n : ℕ) (hn : 2018 ∣ n) :
  ¬(∀ m, ∃ k, (10*m + k) % 11 = 0 ∧ (10*m + k) / 11 ∣ n) :=
sorry

end process_cannot_continue_indefinitely_l302_302859


namespace lines_with_equal_intercepts_l302_302890

theorem lines_with_equal_intercepts (A : ℝ × ℝ) (hA : A = (1, 2)) :
  ∃ (n : ℕ), n = 3 ∧ (∀ l : ℝ → ℝ, (l 1 = 2) → ((l 0 = l (-0)) ∨ (l (-0) = l 0))) :=
by
  sorry

end lines_with_equal_intercepts_l302_302890


namespace pool_perimeter_l302_302783

theorem pool_perimeter (garden_length : ℝ) (plot_area : ℝ) (plot_count : ℕ) : 
  garden_length = 9 ∧ plot_area = 20 ∧ plot_count = 4 →
  ∃ (pool_perimeter : ℝ), pool_perimeter = 18 :=
by
  intros h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end pool_perimeter_l302_302783


namespace problem_l302_302398

theorem problem (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by sorry

end problem_l302_302398


namespace hot_drinks_prediction_at_2_deg_l302_302779

-- Definition of the regression equation as a function
def regression_equation (x : ℝ) : ℝ :=
  -2.35 * x + 147.77

-- The statement to be proved
theorem hot_drinks_prediction_at_2_deg :
  abs (regression_equation 2 - 143) < 1 :=
sorry

end hot_drinks_prediction_at_2_deg_l302_302779


namespace min_white_surface_area_is_five_over_ninety_six_l302_302904

noncomputable def fraction_white_surface_area (total_surface_area white_surface_area : ℕ) :=
  (white_surface_area : ℚ) / (total_surface_area : ℚ)

theorem min_white_surface_area_is_five_over_ninety_six :
  let total_surface_area := 96
  let white_surface_area := 5
  fraction_white_surface_area total_surface_area white_surface_area = 5 / 96 :=
by
  sorry

end min_white_surface_area_is_five_over_ninety_six_l302_302904


namespace find_m_l302_302926

theorem find_m (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : m = Real.sqrt 10 :=
sorry

end find_m_l302_302926


namespace total_apples_l302_302999

theorem total_apples (A B C : ℕ) (h1 : A + B = 11) (h2 : B + C = 18) (h3 : A + C = 19) : A + B + C = 24 :=  
by
  -- Skip the proof
  sorry

end total_apples_l302_302999


namespace octahedron_cut_area_l302_302776

theorem octahedron_cut_area:
  let a := 9
  let b := 3
  let c := 8
  a + b + c = 20 :=
by
  sorry

end octahedron_cut_area_l302_302776


namespace scientific_notation_of_86_million_l302_302740

theorem scientific_notation_of_86_million :
  86000000 = 8.6 * 10^7 :=
sorry

end scientific_notation_of_86_million_l302_302740


namespace det_dilation_matrix_l302_302714

def E : Matrix (Fin 2) (Fin 2) ℝ := ![![12, 0], ![0, 12]]

theorem det_dilation_matrix : Matrix.det E = 144 := by
  sorry

end det_dilation_matrix_l302_302714


namespace algebraic_expression_value_l302_302209

theorem algebraic_expression_value (a : ℝ) (h : 2 * a^2 + 3 * a - 5 = 0) : 6 * a^2 + 9 * a - 5 = 10 :=
by
  sorry

end algebraic_expression_value_l302_302209


namespace sqrt_450_simplified_l302_302044

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l302_302044


namespace shirt_price_is_correct_l302_302300

noncomputable def sweater_price (T : ℝ) : ℝ := T + 7.43 

def discounted_price (S : ℝ) : ℝ := S * 0.90

theorem shirt_price_is_correct :
  ∃ (T S : ℝ), T + discounted_price S = 80.34 ∧ T = S - 7.43 ∧ T = 38.76 :=
by
  sorry

end shirt_price_is_correct_l302_302300


namespace max_value_expression_l302_302545

theorem max_value_expression (a b c : ℝ) (h : a * b * c + a + c - b = 0) : 
  ∃ m, (m = (1/(1+a^2) - 1/(1+b^2) + 1/(1+c^2))) ∧ (m = 5 / 4) :=
by 
  sorry

end max_value_expression_l302_302545


namespace kylie_stamps_l302_302711

theorem kylie_stamps (K N : ℕ) (h1 : N = K + 44) (h2 : K + N = 112) : K = 34 :=
by
  sorry

end kylie_stamps_l302_302711


namespace lateral_surface_area_of_square_pyramid_l302_302471

-- Definitions based on the conditions in a)
def baseEdgeLength : ℝ := 4
def slantHeight : ℝ := 3

-- Lean 4 statement for the proof problem
theorem lateral_surface_area_of_square_pyramid :
  let height := Real.sqrt (slantHeight^2 - (baseEdgeLength / 2)^2)
  let lateralArea := (1 / 2) * 4 * (baseEdgeLength * height)
  lateralArea = 8 * Real.sqrt 5 :=
by
  sorry

end lateral_surface_area_of_square_pyramid_l302_302471


namespace total_cookies_l302_302558

-- Define the number of bags and cookies per bag
def num_bags : Nat := 37
def cookies_per_bag : Nat := 19

-- The theorem stating the total number of cookies
theorem total_cookies : num_bags * cookies_per_bag = 703 := by
  sorry

end total_cookies_l302_302558


namespace sqrt_450_equals_15_sqrt_2_l302_302081

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l302_302081


namespace sqrt_450_simplified_l302_302046

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l302_302046


namespace no_third_quadrant_l302_302944

theorem no_third_quadrant {a b : ℝ} (h1 : 0 < a) (h2 : a < 1) (h3 : -1 < b) : ∀ x y : ℝ, (y = a^x + b) → ¬ (x < 0 ∧ y < 0) :=
by
  intro x y h
  sorry

end no_third_quadrant_l302_302944


namespace cube_difference_l302_302371

theorem cube_difference (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 135 :=
sorry

end cube_difference_l302_302371


namespace ratio_of_triangle_areas_l302_302298

theorem ratio_of_triangle_areas
  (a b c S : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (angle_bisector_theorem : ∀ {A B C K : ℝ}, A / B = C / K)
  (area_ABC : S = (b * c * (a + b + c)))
  : ∃ (area_BOK : ℝ), area_BOK = acS / ((a + b) * (a + b + c)) :=
sorry

end ratio_of_triangle_areas_l302_302298


namespace second_consecutive_odd_integer_l302_302994

theorem second_consecutive_odd_integer (n : ℤ) : 
  (n - 2) + (n + 2) = 152 → n = 76 := 
by 
  sorry

end second_consecutive_odd_integer_l302_302994


namespace probability_of_success_l302_302473

def prob_successful_attempt := 0.5

def prob_unsuccessful_attempt := 1 - prob_successful_attempt

def all_fail_prob := prob_unsuccessful_attempt ^ 4

def at_least_one_success_prob := 1 - all_fail_prob

theorem probability_of_success :
  at_least_one_success_prob = 0.9375 :=
by
  -- Proof would be here
  sorry

end probability_of_success_l302_302473


namespace problem_statement_l302_302586

noncomputable def f (x : ℝ) : ℝ := x + 1 / x - Real.sqrt 2

theorem problem_statement (x : ℝ) (h₁ : x ∈ Set.Ioc (Real.sqrt 2 / 2) 1) :
  Real.sqrt 2 / 2 < f (f x) ∧ f (f x) < x :=
by
  sorry

end problem_statement_l302_302586


namespace sqrt_450_eq_15_sqrt_2_l302_302114

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302114


namespace simplify_sqrt_450_l302_302019
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l302_302019


namespace fraction_value_l302_302423

theorem fraction_value (a b c d : ℕ)
  (h₁ : a = 4 * b)
  (h₂ : b = 3 * c)
  (h₃ : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end fraction_value_l302_302423


namespace average_speed_of_bus_l302_302649

theorem average_speed_of_bus (speed_bicycle : ℝ)
  (start_distance : ℝ) (catch_up_time : ℝ)
  (h1 : speed_bicycle = 15)
  (h2 : start_distance = 195)
  (h3 : catch_up_time = 3) : 
  (start_distance + speed_bicycle * catch_up_time) / catch_up_time = 80 :=
by
  sorry

end average_speed_of_bus_l302_302649


namespace negation_of_exists_l302_302554

open Classical

theorem negation_of_exists (p : Prop) : 
  (∃ x : ℝ, 2^x ≥ 2 * x + 1) ↔ ¬ ∀ x : ℝ, 2^x < 2 * x + 1 :=
by
  sorry

end negation_of_exists_l302_302554


namespace range_of_a_l302_302239

variable {x a : ℝ}

theorem range_of_a (hx : 1 ≤ x ∧ x ≤ 2) (h : 2 * x > a - x^2) : a < 8 :=
by sorry

end range_of_a_l302_302239


namespace problem_1_problem_2_problem_3_l302_302551

-- Problem 1
theorem problem_1 (a : ℝ) (t : ℝ) (h1 : a ≠ 1) (h2 : f 3 - g 3 = 0) : t = -4 := sorry

-- Problem 2
theorem problem_2 (a : ℝ) (t : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : t = 1) (x : ℝ) (h4 : f x ≤ g x) : -1/2 < x ∧ x ≤ 0 := sorry

-- Problem 3
theorem problem_3 (a : ℝ) (t : ℝ) (h1 : a ≠ 1) (h2 : ∃ x : ℝ, -1 < x ∧ x ≤ 3 ∧ F x = 0) :
  t ≤ -5/7 ∨ t ≥ (2 + real.sqrt 2) / 4 := sorry

/-- Definitions of functions used in the problems -/
def f (a : ℝ) (x : ℝ) := real.log (x + 1) / real.log a

def g (a : ℝ) (x : ℝ) (t : ℝ) := 2 * real.log (2 * x + t) / real.log a

def F (a : ℝ) (x : ℝ) (t : ℝ) := (a : ℝ) ^ (f a x) + t * x^2 - 2 * t + 1

end problem_1_problem_2_problem_3_l302_302551


namespace jeff_average_skips_is_14_l302_302730

-- Definitions of the given conditions in the problem
def sam_skips_per_round : ℕ := 16
def rounds : ℕ := 4

-- Number of skips by Jeff in each round based on the conditions
def jeff_first_round_skips : ℕ := sam_skips_per_round - 1
def jeff_second_round_skips : ℕ := sam_skips_per_round - 3
def jeff_third_round_skips : ℕ := sam_skips_per_round + 4
def jeff_fourth_round_skips : ℕ := sam_skips_per_round / 2

-- Total skips by Jeff in all rounds
def jeff_total_skips : ℕ := jeff_first_round_skips + 
                           jeff_second_round_skips + 
                           jeff_third_round_skips + 
                           jeff_fourth_round_skips

-- Average skips per round by Jeff
def jeff_average_skips : ℕ := jeff_total_skips / rounds

-- Theorem statement
theorem jeff_average_skips_is_14 : jeff_average_skips = 14 := 
by 
    sorry

end jeff_average_skips_is_14_l302_302730


namespace sqrt_simplify_l302_302036

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l302_302036


namespace system_of_equations_proof_l302_302232

theorem system_of_equations_proof (a b x A B C : ℝ) (h1: a ≠ 0) 
  (h2: a * Real.sin x + b * Real.cos x = 0) 
  (h3: A * Real.sin (2 * x) + B * Real.cos (2 * x) = C) : 
  2 * a * b * A + (b ^ 2 - a ^ 2) * B + (a ^ 2 + b ^ 2) * C = 0 := 
sorry

end system_of_equations_proof_l302_302232


namespace gcd_1260_924_l302_302857

theorem gcd_1260_924 : Nat.gcd 1260 924 = 84 :=
by
  sorry

end gcd_1260_924_l302_302857


namespace time_difference_l302_302511

theorem time_difference (dist1 dist2 : ℕ) (speed : ℕ) (h_dist : dist1 = 600) (h_dist2 : dist2 = 550) (h_speed : speed = 40) :
  (dist1 - dist2) / speed * 60 = 75 := by
  sorry

end time_difference_l302_302511


namespace store_loses_out_l302_302326

theorem store_loses_out (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) (x y : ℝ)
    (h1 : a = b * x) (h2 : b = a * y) : x + y > 2 :=
by
  sorry

end store_loses_out_l302_302326


namespace ksyusha_travel_time_wednesday_l302_302263

-- Definitions based on the conditions
def speed_walk (v : ℝ) : ℝ := v
def speed_run (v : ℝ) : ℝ := 2 * v
def distance_walk_tuesday (S : ℝ) : ℝ := 2 * S
def distance_run_tuesday (S : ℝ) : ℝ := S
def total_time_tuesday (v S : ℝ) : ℝ := (distance_walk_tuesday S) / (speed_walk v) + (distance_run_tuesday S) / (speed_run v)

-- Theorem statement
theorem ksyusha_travel_time_wednesday (S v : ℝ) (h : total_time_tuesday v S = 30) :
  (S / v + S / v) = 24 := 
by sorry

end ksyusha_travel_time_wednesday_l302_302263


namespace john_took_11_more_l302_302606

/-- 
If Ray took 10 chickens, Ray took 6 chickens less than Mary, and 
John took 5 more chickens than Mary, then John took 11 more 
chickens than Ray. 
-/
theorem john_took_11_more (R M J : ℕ) (h1 : R = 10) 
  (h2 : R + 6 = M) (h3 : M + 5 = J) : J - R = 11 :=
by
  sorry

end john_took_11_more_l302_302606


namespace log_expression_simplifies_to_zero_l302_302787

theorem log_expression_simplifies_to_zero : 
  (1/2 : ℝ) * (Real.log 4) + Real.log 5 - Real.exp (0 * Real.log (Real.pi + 1)) = 0 := 
by
  sorry

end log_expression_simplifies_to_zero_l302_302787


namespace half_angle_quadrant_l302_302236

-- Define the given condition
def is_angle_in_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 < α ∧ α < k * 360 + 90

-- Define the result that needs to be proved
def is_angle_in_first_or_third_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, (k * 180 < α / 2 ∧ α / 2 < k * 180 + 45) ∨ (k * 180 + 180 < α / 2 ∧ α / 2 < k * 180 + 225)

-- The main theorem statement
theorem half_angle_quadrant (α : ℝ) (h : is_angle_in_first_quadrant α) : is_angle_in_first_or_third_quadrant α :=
sorry

end half_angle_quadrant_l302_302236


namespace integer_solutions_exist_l302_302981

theorem integer_solutions_exist (m n : ℤ) :
  ∃ (w x y z : ℤ), 
  (w + x + 2 * y + 2 * z = m) ∧ 
  (2 * w - 2 * x + y - z = n) := sorry

end integer_solutions_exist_l302_302981


namespace John_took_more_chickens_than_Ray_l302_302611

theorem John_took_more_chickens_than_Ray
  (r m j : ℕ)
  (h1 : r = 10)
  (h2 : r = m - 6)
  (h3 : j = m + 5) : j - r = 11 :=
by
  sorry

end John_took_more_chickens_than_Ray_l302_302611


namespace inequality_preserved_l302_302369

variable {a b c : ℝ}

theorem inequality_preserved (h : abs ((a^2 + b^2 - c^2) / (a * b)) < 2) :
    abs ((b^2 + c^2 - a^2) / (b * c)) < 2 ∧ abs ((c^2 + a^2 - b^2) / (c * a)) < 2 := 
sorry

end inequality_preserved_l302_302369


namespace simplify_sqrt_450_l302_302091

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l302_302091


namespace DanielCandies_l302_302590

noncomputable def initialCandies (x : ℝ) : Prop :=
  (3 / 8) * x - (3 / 2) - 16 = 10

theorem DanielCandies : ∃ x : ℝ, initialCandies x ∧ x = 93 :=
by
  use 93
  simp [initialCandies]
  norm_num
  sorry

end DanielCandies_l302_302590


namespace sally_out_of_pocket_cost_l302_302729

/-- Definitions of the given conditions -/
def given_money : Int := 320
def cost_per_book : Int := 15
def number_of_students : Int := 35

/-- Theorem to prove the amount Sally needs to pay out of pocket -/
theorem sally_out_of_pocket_cost : 
  let total_cost := number_of_students * cost_per_book
  let amount_given := given_money
  let out_of_pocket_cost := total_cost - amount_given
  out_of_pocket_cost = 205 := by
  sorry

end sally_out_of_pocket_cost_l302_302729


namespace value_of_ac_over_bd_l302_302412

theorem value_of_ac_over_bd (a b c d : ℝ) 
  (h1 : a = 4 * b)
  (h2 : b = 3 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by
  sorry

end value_of_ac_over_bd_l302_302412


namespace find_n_in_arithmetic_sequence_l302_302898

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 2) = a (n + 1) + d

theorem find_n_in_arithmetic_sequence (x : ℝ) (n : ℕ) (b : ℕ → ℝ)
  (h1 : b 1 = Real.exp x) 
  (h2 : b 2 = x) 
  (h3 : is_arithmetic_sequence b) : 
  b n = 1 + Real.exp x ↔ n = (1 + x) / (x - Real.exp x) :=
sorry

end find_n_in_arithmetic_sequence_l302_302898


namespace corrected_multiplication_result_l302_302763

theorem corrected_multiplication_result :
  ∃ n : ℕ, 987 * n = 559989 ∧ 987 * n ≠ 559981 ∧ 559981 % 100 = 98 :=
by
  sorry

end corrected_multiplication_result_l302_302763


namespace find_x_value_l302_302494

theorem find_x_value : (8 = 2^3) ∧ (8 * 8^32 = 8^33) ∧ (8^33 = 2^99) → ∃ x, 8^32 + 8^32 + 8^32 + 8^32 + 8^32 + 8^32 + 8^32 + 8^32 = 2^x ∧ x = 99 :=
by
  intros h
  sorry

end find_x_value_l302_302494


namespace func_translation_right_symm_yaxis_l302_302987

def f (x : ℝ) : ℝ := sorry

theorem func_translation_right_symm_yaxis (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x - 1) = e ^ (-x)) :
  ∀ x, f x = e ^ (-x - 1) := sorry

end func_translation_right_symm_yaxis_l302_302987


namespace sqrt_450_eq_15_sqrt_2_l302_302060

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302060


namespace value_of_frac_l302_302390

theorem value_of_frac (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_frac_l302_302390


namespace find_asymptotes_l302_302529

def hyperbola_eq (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 9 = 1

def shifted_hyperbola_asymptotes (x y : ℝ) : Prop :=
  y = 4 / 3 * x + 5 ∨ y = -4 / 3 * x + 5

theorem find_asymptotes (x y : ℝ) :
  (∃ y', y = y' + 5 ∧ hyperbola_eq x y')
  ↔ shifted_hyperbola_asymptotes x y :=
by
  sorry

end find_asymptotes_l302_302529


namespace soda_price_before_increase_l302_302331

theorem soda_price_before_increase
  (candy_box_after : ℝ)
  (soda_after : ℝ)
  (candy_box_increase : ℝ)
  (soda_increase : ℝ)
  (new_price_soda : soda_after = 9)
  (new_price_candy_box : candy_box_after = 10)
  (percent_candy_box_increase : candy_box_increase = 0.25)
  (percent_soda_increase : soda_increase = 0.50) :
  ∃ P : ℝ, 1.5 * P = 9 ∧ P = 6 := 
by
  sorry

end soda_price_before_increase_l302_302331


namespace smallest_common_multiple_l302_302158

theorem smallest_common_multiple : Nat.lcm 18 35 = 630 := by
  sorry

end smallest_common_multiple_l302_302158


namespace problem_l302_302366

noncomputable def f : ℝ → ℝ := sorry 

theorem problem
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_func : ∀ x : ℝ, f (2 + x) = -f (2 - x))
  (h_value : f (-3) = -2) :
  f 2007 = 2 :=
sorry

end problem_l302_302366


namespace find_x_given_inverse_relationship_l302_302479

variable {x y : ℝ}

theorem find_x_given_inverse_relationship 
  (h₀ : x > 0) 
  (h₁ : y > 0) 
  (initial_condition : 3^2 * 25 = 225)
  (inversion_condition : x^2 * y = 225)
  (query : y = 1200) :
  x = Real.sqrt (3 / 16) :=
by
  sorry

end find_x_given_inverse_relationship_l302_302479


namespace percentage_of_boys_l302_302240

theorem percentage_of_boys (total_students boys girls : ℕ) (h_ratio : boys * 4 = girls * 3) (h_total : boys + girls = total_students) (h_total_students : total_students = 42) : (boys : ℚ) * 100 / total_students = 42.857 :=
by
  sorry

end percentage_of_boys_l302_302240


namespace problem_l302_302393

theorem problem (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by sorry

end problem_l302_302393


namespace alpha_beta_range_l302_302952

theorem alpha_beta_range (α β : ℝ) (h1 : - (π / 2) < α) (h2 : α < β) (h3 : β < π) : 
- 3 * (π / 2) < α - β ∧ α - β < 0 :=
by
  sorry

end alpha_beta_range_l302_302952


namespace sqrt_450_eq_15_sqrt_2_l302_302109

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302109


namespace complement_M_in_U_l302_302719

open Set

theorem complement_M_in_U : 
  let U : Set ℕ := {1, 3, 5, 7}
  let M : Set ℕ := {1, 5}
  U \ M = {3, 7} := 
by
  let U : Set ℕ := {1, 3, 5, 7}
  let M : Set ℕ := {1, 5}
  sorry

end complement_M_in_U_l302_302719


namespace five_g_speeds_l302_302280

theorem five_g_speeds (m : ℝ) :
  (1400 / 50) - (1400 / (50 * m)) = 24 → m = 7 :=
by
  sorry

end five_g_speeds_l302_302280


namespace tangent_position_is_six_l302_302472

def clock_radius : ℝ := 30
def disk_radius : ℝ := 15
def initial_tangent_position := 12
def final_tangent_position := 6

theorem tangent_position_is_six :
  (∃ (clock_radius disk_radius : ℝ), clock_radius = 30 ∧ disk_radius = 15) →
  (initial_tangent_position = 12) →
  (final_tangent_position = 6) :=
by
  intros h1 h2
  sorry

end tangent_position_is_six_l302_302472


namespace determine_x_l302_302343

theorem determine_x (p q : ℝ) (hpq : p ≠ q) : 
  ∃ (c d : ℝ), (x = c*p + d*q) ∧ c = 2 ∧ d = -2 :=
by 
  sorry

end determine_x_l302_302343


namespace sin_neg_4_div_3_pi_l302_302882

theorem sin_neg_4_div_3_pi : Real.sin (- (4 / 3) * Real.pi) = Real.sqrt 3 / 2 :=
by sorry

end sin_neg_4_div_3_pi_l302_302882


namespace john_took_11_more_l302_302607

/-- 
If Ray took 10 chickens, Ray took 6 chickens less than Mary, and 
John took 5 more chickens than Mary, then John took 11 more 
chickens than Ray. 
-/
theorem john_took_11_more (R M J : ℕ) (h1 : R = 10) 
  (h2 : R + 6 = M) (h3 : M + 5 = J) : J - R = 11 :=
by
  sorry

end john_took_11_more_l302_302607


namespace sqrt_450_eq_15_sqrt_2_l302_302131

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302131


namespace sqrt_450_simplified_l302_302045

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l302_302045


namespace sum_of_first_11_terms_l302_302576

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Condition: the sequence is an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Conditions given in the problem
axiom h1 : a 1 + a 5 + a 9 = 39
axiom h2 : a 3 + a 7 + a 11 = 27
axiom h3 : is_arithmetic_sequence a d

-- Proof statement
theorem sum_of_first_11_terms : (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11) = 121 := 
sorry

end sum_of_first_11_terms_l302_302576


namespace tommy_saw_100_wheels_l302_302752

-- Define the parameters
def trucks : ℕ := 12
def cars : ℕ := 13
def wheels_per_truck : ℕ := 4
def wheels_per_car : ℕ := 4

-- Define the statement to prove
theorem tommy_saw_100_wheels : (trucks * wheels_per_truck + cars * wheels_per_car) = 100 := by
  sorry 

end tommy_saw_100_wheels_l302_302752


namespace sqrt_450_eq_15_sqrt_2_l302_302132

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302132


namespace max_value_of_function_f_l302_302155

noncomputable def f (t : ℝ) : ℝ := (4^t - 2 * t) * t / 16^t

theorem max_value_of_function_f : ∃ t : ℝ, ∀ x : ℝ, f x ≤ f t ∧ f t = 1 / 8 := sorry

end max_value_of_function_f_l302_302155


namespace sum_of_functions_positive_l302_302816

open Real

noncomputable def f (x : ℝ) : ℝ := (exp x - exp (-x)) / 2

theorem sum_of_functions_positive (x1 x2 x3 : ℝ) (h1 : x1 + x2 > 0) (h2 : x2 + x3 > 0) (h3 : x3 + x1 > 0) : f x1 + f x2 + f x3 > 0 := by
  sorry

end sum_of_functions_positive_l302_302816


namespace range_of_x_for_function_l302_302474

theorem range_of_x_for_function :
  ∀ x : ℝ, (2 - x ≥ 0 ∧ x - 1 ≠ 0) ↔ (x ≤ 2 ∧ x ≠ 1) := by
  sorry

end range_of_x_for_function_l302_302474


namespace difference_twice_cecil_and_catherine_l302_302193

theorem difference_twice_cecil_and_catherine
  (Cecil Catherine Carmela : ℕ)
  (h1 : Cecil = 600)
  (h2 : Carmela = 2 * 600 + 50)
  (h3 : 600 + (2 * 600 - Catherine) + Carmela = 2800) :
  2 * 600 - Catherine = 250 := by
  sorry

end difference_twice_cecil_and_catherine_l302_302193


namespace instructors_meeting_l302_302727

theorem instructors_meeting (R P E M : ℕ) (hR : R = 5) (hP : P = 8) (hE : E = 10) (hM : M = 9) :
  Nat.lcm (Nat.lcm R P) (Nat.lcm E M) = 360 :=
by
  rw [hR, hP, hE, hM]
  sorry

end instructors_meeting_l302_302727


namespace greatest_difference_54_l302_302531

theorem greatest_difference_54 (board : ℕ → ℕ → ℕ) (h : ∀ i j, 1 ≤ board i j ∧ board i j ≤ 100) :
  ∃ i j k l, (i = k ∨ j = l) ∧ (board i j - board k l ≥ 54 ∨ board k l - board i j ≥ 54) :=
sorry

end greatest_difference_54_l302_302531


namespace alley_width_l302_302245

theorem alley_width (b u v : ℝ)
(h₁ : u = b * real.sin (real.pi / 3))
(h₂ : v = b * real.sin (real.pi / 6)) :
  ∃ w, w = b * (1 + real.sqrt 3) / 2 :=
by
  sorry

end alley_width_l302_302245


namespace simplify_and_evaluate_l302_302460

variable (x y : ℤ)

theorem simplify_and_evaluate (h1 : x = 1) (h2 : y = 1) :
    2 * (x - 2 * y) ^ 2 - (2 * y + x) * (-2 * y + x) = 5 := by
    sorry

end simplify_and_evaluate_l302_302460


namespace Johnson_Carter_Tie_August_l302_302137

structure MonthlyHomeRuns where
  March : Nat
  April : Nat
  May : Nat
  June : Nat
  July : Nat
  August : Nat
  September : Nat

def Johnson_runs : MonthlyHomeRuns := { March:= 2, April:= 11, May:= 15, June:= 9, July:= 7, August:= 9, September:= 0 }
def Carter_runs : MonthlyHomeRuns := { March:= 1, April:= 9, May:= 8, June:= 19, July:= 6, August:= 10, September:= 0 }

noncomputable def cumulative_runs (runs: MonthlyHomeRuns) (month: String) : Nat :=
  match month with
  | "March" => runs.March
  | "April" => runs.March + runs.April
  | "May" => runs.March + runs.April + runs.May
  | "June" => runs.March + runs.April + runs.May + runs.June
  | "July" => runs.March + runs.April + runs.May + runs.June + runs.July
  | "August" => runs.March + runs.April + runs.May + runs.June + runs.July + runs.August
  | _ => 0

theorem Johnson_Carter_Tie_August :
  cumulative_runs Johnson_runs "August" = cumulative_runs Carter_runs "August" := 
  by
  sorry

end Johnson_Carter_Tie_August_l302_302137


namespace number_of_children_admitted_l302_302734

variable (children adults : ℕ)

def admission_fee_children : ℝ := 1.5
def admission_fee_adults  : ℝ := 4

def total_people : ℕ := 315
def total_fees   : ℝ := 810

theorem number_of_children_admitted :
  ∃ (C A : ℕ), C + A = total_people ∧ admission_fee_children * C + admission_fee_adults * A = total_fees ∧ C = 180 :=
by
  sorry

end number_of_children_admitted_l302_302734


namespace find_real_solutions_l302_302799

theorem find_real_solutions : 
  ∀ x : ℝ, 1 / ((x - 2) * (x - 3)) 
         + 1 / ((x - 3) * (x - 4)) 
         + 1 / ((x - 4) * (x - 5)) 
         = 1 / 8 ↔ x = 7 ∨ x = -2 :=
by
  intro x
  sorry

end find_real_solutions_l302_302799


namespace domain_sqrt_sin_cos_l302_302616

open Real

theorem domain_sqrt_sin_cos (k : ℤ) :
  {x : ℝ | ∃ k : ℤ, (2 * k * π + π / 4 ≤ x) ∧ (x ≤ 2 * k * π + 5 * π / 4)} = 
  {x : ℝ | sin x - cos x ≥ 0} :=
sorry

end domain_sqrt_sin_cos_l302_302616


namespace triangle_perimeter_l302_302178

theorem triangle_perimeter (a b c : ℕ) (ha : a = 14) (hb : b = 8) (hc : c = 9) : a + b + c = 31 := 
by
  sorry

end triangle_perimeter_l302_302178


namespace concrete_pillars_l302_302962

-- Definitions based on the conditions of the problem
def C_deck : ℕ := 1600
def C_anchor : ℕ := 700
def C_total : ℕ := 4800

-- Theorem to prove the concrete required for supporting pillars
theorem concrete_pillars : C_total - (C_deck + 2 * C_anchor) = 1800 :=
by sorry

end concrete_pillars_l302_302962


namespace katie_bead_necklaces_l302_302832

theorem katie_bead_necklaces (B : ℕ) (gemstone_necklaces : ℕ := 3) (cost_each_necklace : ℕ := 3) (total_earnings : ℕ := 21) :
  gemstone_necklaces * cost_each_necklace + B * cost_each_necklace = total_earnings → B = 4 :=
by
  intro h
  sorry

end katie_bead_necklaces_l302_302832


namespace domain_of_g_l302_302678

-- Define the function f and specify the domain of f(x+1)
def f : ℝ → ℝ := sorry
def domain_f_x_plus_1 : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3} -- Domain of f(x+1) is [-1, 3]

-- Define the definition of the function g where g(x) = f(x^2)
def g (x : ℝ) : ℝ := f (x^2)

-- Prove that the domain of g(x) is [-2, 2]
theorem domain_of_g : {x | -2 ≤ x ∧ x ≤ 2} = {x | ∃ (y : ℝ), (0 ≤ y ∧ y ≤ 4) ∧ (x = y ∨ x = -y)} :=
by 
  sorry

end domain_of_g_l302_302678


namespace expression_evaluation_valid_l302_302814

theorem expression_evaluation_valid (a : ℝ) (h1 : a = 4) :
  (1 + (4 / (a ^ 2 - 4))) * ((a + 2) / a) = 2 := by
  sorry

end expression_evaluation_valid_l302_302814


namespace radius_of_surrounding_circles_is_correct_l302_302317

noncomputable def r : Real := 1 + Real.sqrt 2

theorem radius_of_surrounding_circles_is_correct (r: ℝ)
  (h₁: ∃c : ℝ, c = 2) -- central circle radius is 2
  (h₂: ∃far: ℝ, far = (1 + (Real.sqrt 2))) -- r is the solution as calculated
: 2 * r = 1 + Real.sqrt 2 :=
by
  sorry

end radius_of_surrounding_circles_is_correct_l302_302317


namespace intersection_of_A_and_B_l302_302380

open Set

def A : Set ℝ := { x | 3 * x + 2 > 0 }
def B : Set ℝ := { x | (x + 1) * (x - 3) > 0 }

theorem intersection_of_A_and_B : A ∩ B = { x : ℝ | x > 3 } :=
by 
  sorry

end intersection_of_A_and_B_l302_302380


namespace red_balls_unchanged_l302_302692

-- Definitions: 
def initial_red_balls : ℕ := 3
def initial_blue_balls : ℕ := 2
def initial_yellow_balls : ℕ := 5

def remove_blue_ball (blue_balls : ℕ) : ℕ :=
  if blue_balls > 0 then blue_balls - 1 else blue_balls

-- Condition after one blue ball is removed
def blue_balls_after_removal := remove_blue_ball initial_blue_balls

-- Prove that the number of red balls remain unchanged
theorem red_balls_unchanged : initial_red_balls = 3 :=
by
  sorry

end red_balls_unchanged_l302_302692


namespace not_and_implication_l302_302550

variable (p q : Prop)

theorem not_and_implication : ¬ (p ∧ q) → (¬ p ∨ ¬ q) :=
by
  sorry

end not_and_implication_l302_302550


namespace ksyusha_wednesday_time_l302_302267

def speed_relation (v : ℝ) := ∀ (d_walk d_run1 d_run2 time1 time2: ℝ), 
  (2 * d_run1 = d_walk) → -- she walked twice the distance she ran on Tuesday
  (2 * v = 2 * v) → -- she runs twice as fast as she walks
  ((2 * d_walk / v) + (d_run1 / (2 * v)) = time1) → -- she took 30 minutes on Tuesday
  (time1 = 30) → -- Tuesday time equals 30
  (d_run2 = 2 * d_walk) → -- she had to run twice the distance she walked on Wednesday
  ((d_walk / v) + (d_run2 / (2 * v)) = time2) → -- calculate the time for Wednesday
  time2 = 24 -- time taken on Wednesday

theorem ksyusha_wednesday_time :
  ∃ (v : ℝ), speed_relation v :=
begin
  existsi 1,
  unfold speed_relation,
  intros,
  sorry,
end

end ksyusha_wednesday_time_l302_302267


namespace books_per_day_l302_302454

-- Define the condition: Mrs. Hilt reads 15 books in 3 days.
def reads_books_in_days (total_books : ℕ) (days : ℕ) : Prop :=
  total_books = 15 ∧ days = 3

-- Define the theorem to prove that Mrs. Hilt reads 5 books per day.
theorem books_per_day (total_books : ℕ) (days : ℕ) (h : reads_books_in_days total_books days) : total_books / days = 5 :=
by
  -- Stub proof
  sorry

end books_per_day_l302_302454


namespace Q_coordinates_l302_302296

structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩
def P : Point := ⟨0, 3⟩
def R : Point := ⟨5, 0⟩

def isRectangle (A B C D : Point) : Prop :=
  -- replace this with the actual implementation of rectangle properties
  sorry

theorem Q_coordinates :
  ∃ Q : Point, isRectangle O P Q R ∧ Q.x = 5 ∧ Q.y = 3 :=
by
  -- replace this with the actual proof
  sorry

end Q_coordinates_l302_302296


namespace isosceles_triangle_perimeter_l302_302436

def is_isosceles (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem isosceles_triangle_perimeter :
  ∃ (a b c : ℕ), is_isosceles a b c ∧ ((a = 3 ∧ b = 3 ∧ c = 4 ∧ a + b + c = 10) ∨ (a = 3 ∧ b = 4 ∧ c = 4 ∧ a + b + c = 11)) :=
by
  sorry

end isosceles_triangle_perimeter_l302_302436


namespace find_x1_l302_302221

theorem find_x1 (x1 x2 x3 : ℝ) (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1-x1)^2 + 2*(x1-x2)^2 + (x2-x3)^2 + x3^2 = 1/2) :
  x1 = (3*Real.sqrt 2 - 3)/7 :=
by
  sorry

end find_x1_l302_302221


namespace concrete_pillars_correct_l302_302964

-- Definitions based on conditions
def concrete_for_roadway := 1600
def concrete_for_one_anchor := 700
def total_concrete_for_bridge := 4800

-- Total concrete for both anchors
def concrete_for_both_anchors := 2 * concrete_for_one_anchor

-- Total concrete needed for the roadway and anchors
def concrete_for_roadway_and_anchors := concrete_for_roadway + concrete_for_both_anchors

-- Concrete needed for the supporting pillars
def concrete_for_pillars := total_concrete_for_bridge - concrete_for_roadway_and_anchors

-- Proof problem statement, verify that the concrete for the supporting pillars is 1800 tons
theorem concrete_pillars_correct : concrete_for_pillars = 1800 := by
  sorry

end concrete_pillars_correct_l302_302964


namespace speed_of_current_l302_302322

-- Definitions for the conditions
variables (m c : ℝ)

-- Condition 1: man's speed with the current
def speed_with_current := m + c = 16

-- Condition 2: man's speed against the current
def speed_against_current := m - c = 9.6

-- The goal is to prove c = 3.2 given the conditions
theorem speed_of_current (h1 : speed_with_current m c) 
                         (h2 : speed_against_current m c) :
  c = 3.2 := 
sorry

end speed_of_current_l302_302322


namespace maximize_f_l302_302308

noncomputable def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

theorem maximize_f : ∃ x : ℝ, f x = 2 ∧ (∀ y : ℝ, f x ≥ f y) :=
by
  use 2
  split
  { show f 2 = 2
    sorry }
  { intro y
    show f 2 ≥ f y
    sorry }

end maximize_f_l302_302308


namespace sqrt_450_eq_15_sqrt_2_l302_302125

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302125


namespace increase_in_sets_when_profit_38_price_reduction_for_1200_profit_l302_302464

-- Definitions for conditions
def original_profit_per_set := 40
def original_sets_sold_per_day := 20
def additional_sets_per_dollar_drop := 2

-- The proof problems

-- Part 1: Prove the increase in sets when profit reduces to $38
theorem increase_in_sets_when_profit_38 :
  let decrease_in_profit := (original_profit_per_set - 38)
  additional_sets_per_dollar_drop * decrease_in_profit = 4 :=
by
  sorry

-- Part 2: Prove the price reduction needed for $1200 daily profit
theorem price_reduction_for_1200_profit :
  ∃ x, (original_profit_per_set - x) * (original_sets_sold_per_day + 2 * x) = 1200 ∧ x = 20 :=
by
  sorry

end increase_in_sets_when_profit_38_price_reduction_for_1200_profit_l302_302464


namespace pat_moved_chairs_l302_302789

theorem pat_moved_chairs (total_chairs : ℕ) (carey_moved : ℕ) (left_to_move : ℕ) (pat_moved : ℕ) :
  total_chairs = 74 →
  carey_moved = 28 →
  left_to_move = 17 →
  pat_moved = total_chairs - left_to_move - carey_moved →
  pat_moved = 29 :=
by
  intros h_total h_carey h_left h_equation
  rw [h_total, h_carey, h_left] at h_equation
  exact h_equation

end pat_moved_chairs_l302_302789


namespace xyz_identity_l302_302501

theorem xyz_identity (x y z : ℝ) (h : x + y + z = x * y * z) :
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) = 4 * x * y * z :=
by
  sorry

end xyz_identity_l302_302501


namespace factorial_quotient_l302_302191

/-- Prove that the quotient of the factorial of 4! divided by 4! simplifies to 23!. -/
theorem factorial_quotient : (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := 
by
  sorry

end factorial_quotient_l302_302191


namespace fraction_spent_on_dvd_l302_302977

theorem fraction_spent_on_dvd (r l m d x : ℝ) (h1 : r = 200) (h2 : l = (1/4) * r) (h3 : m = r - l) (h4 : x = 50) (h5 : d = m - x) : d / r = 1 / 2 :=
by
  sorry

end fraction_spent_on_dvd_l302_302977


namespace gcd_fx_x_l302_302676

def f (x: ℕ) := (5 * x + 4) * (9 * x + 7) * (11 * x + 3) * (x + 12)

theorem gcd_fx_x (x: ℕ) (h: x % 54896 = 0) : Nat.gcd (f x) x = 112 :=
  sorry

end gcd_fx_x_l302_302676


namespace simplify_sqrt_450_l302_302010

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l302_302010


namespace value_of_fraction_l302_302432

-- Definitions based on conditions
variables {a b c d : ℝ}
hypothesis h1 : a = 4 * b
hypothesis h2 : b = 3 * c
hypothesis h3 : c = 5 * d

-- The statement we want to prove
theorem value_of_fraction : (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_fraction_l302_302432


namespace cosA_value_area_of_triangle_l302_302826

noncomputable def cosA (a b c : ℝ) (cos_C : ℝ) : ℝ :=
  if (a ≠ 0 ∧ cos_C ≠ 0) then (2 * b - c) * cos_C / a else 1 / 2

noncomputable def area_triangle (a b c : ℝ) (cosA_val : ℝ) : ℝ :=
  let S := a * b * (Real.sqrt (1 - cosA_val ^ 2)) / 2
  S

theorem cosA_value (a b c : ℝ) (cos_C : ℝ) : a * cos_C = (2 * b - c) * (cosA a b c cos_C) → cosA a b c cos_C = 1 / 2 :=
by
  sorry

theorem area_of_triangle (a b c : ℝ) (cos_A : ℝ) (cos_A_proof : a * cos_C = (2 * b - c) * cos_A) (h₀ : a = 6) (h₁ : b + c = 8) : area_triangle a b c cos_A = 7 * Real.sqrt 3 / 3 :=
by
  sorry

end cosA_value_area_of_triangle_l302_302826


namespace simplify_sqrt_450_l302_302098

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l302_302098


namespace total_pencils_is_220_l302_302468

theorem total_pencils_is_220
  (A : ℕ) (B : ℕ) (P : ℕ) (Q : ℕ)
  (hA : A = 50)
  (h_sum : A + B = 140)
  (h_diff : B - A = P/2)
  (h_pencils : Q = P + 60)
  : P + Q = 220 :=
by
  sorry

end total_pencils_is_220_l302_302468


namespace converse_inverse_contrapositive_l302_302168

-- The original statement
def original_statement (x y : ℕ) : Prop :=
  (x + y = 5) → (x = 3 ∧ y = 2)

-- Converse of the original statement
theorem converse (x y : ℕ) : (x = 3 ∧ y = 2) → (x + y = 5) :=
by
  sorry

-- Inverse of the original statement
theorem inverse (x y : ℕ) : (x + y ≠ 5) → (x ≠ 3 ∨ y ≠ 2) :=
by
  sorry

-- Contrapositive of the original statement
theorem contrapositive (x y : ℕ) : (x ≠ 3 ∨ y ≠ 2) → (x + y ≠ 5) :=
by
  sorry

end converse_inverse_contrapositive_l302_302168


namespace problem_1_1_and_2_problem_1_2_l302_302444

section Sequence

variables (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Conditions
axiom a_1 : a 1 = 3
axiom a_n_recurr : ∀ n ≥ 2, a n = 2 * a (n - 1) + (n - 2)

-- Prove that {a_n + n} is a geometric sequence and find the general term formula for {a_n}
theorem problem_1_1_and_2 :
  (∀ n ≥ 2, (a (n - 1) + (n - 1) ≠ 0)) ∧ ((a 1 + 1) * 2^(n - 1) = a n + n) ∧
  (∀ n, a n = 2^(n + 1) - n) :=
sorry

-- Find the sum of the first n terms, S_n, of the sequence {a_n}
theorem problem_1_2 (n : ℕ) : S n = 2^(n + 2) - 4 - (n^2 + n) / 2 :=
sorry

end Sequence

end problem_1_1_and_2_problem_1_2_l302_302444


namespace fifth_term_arithmetic_sequence_is_19_l302_302828

def arithmetic_sequence_nth_term (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem fifth_term_arithmetic_sequence_is_19 :
  arithmetic_sequence_nth_term 3 4 5 = 19 := 
  by
  sorry

end fifth_term_arithmetic_sequence_is_19_l302_302828


namespace chess_tournament_game_count_l302_302315

theorem chess_tournament_game_count (n : ℕ) (h1 : ∃ n, ∀ i j, 1 ≤ i ∧ i ≤ 6 ∧ 1 ≤ j ∧ j ≤ 6 ∧ i ≠ j → ∃ games_between, games_between = n ∧ games_between * (Nat.choose 6 2) = 30) : n = 2 :=
by
  sorry

end chess_tournament_game_count_l302_302315


namespace simplify_sqrt_450_l302_302101

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l302_302101


namespace determine_a_from_equation_l302_302693

theorem determine_a_from_equation (a : ℝ) (x : ℝ) (h1 : x = 1) (h2 : a * x + 3 * x = 2) : a = -1 := by
  sorry

end determine_a_from_equation_l302_302693


namespace fraction_value_l302_302421

theorem fraction_value (a b c d : ℕ)
  (h₁ : a = 4 * b)
  (h₂ : b = 3 * c)
  (h₃ : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end fraction_value_l302_302421


namespace value_of_ac_over_bd_l302_302411

theorem value_of_ac_over_bd (a b c d : ℝ) 
  (h1 : a = 4 * b)
  (h2 : b = 3 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by
  sorry

end value_of_ac_over_bd_l302_302411


namespace total_paint_remaining_l302_302797

-- Definitions based on the conditions
def paint_per_statue : ℚ := 1 / 16
def statues_to_paint : ℕ := 14

-- Theorem statement to prove the answer
theorem total_paint_remaining : (statues_to_paint : ℚ) * paint_per_statue = 7 / 8 := 
by sorry

end total_paint_remaining_l302_302797


namespace expression_undefined_l302_302541

theorem expression_undefined (a : ℝ) : (a = 2 ∨ a = -2) ↔ (a^2 - 4 = 0) :=
by sorry

end expression_undefined_l302_302541


namespace solve_equation_l302_302462

theorem solve_equation : ∃ x : ℤ, (x - 15) / 3 = (3 * x + 11) / 8 ∧ x = -153 := 
by
  use -153
  sorry

end solve_equation_l302_302462


namespace simplify_sqrt_450_l302_302020
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l302_302020


namespace abc_inequality_l302_302283

theorem abc_inequality (a b c : ℝ) : a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
sorry

end abc_inequality_l302_302283


namespace sqrt_450_eq_15_sqrt_2_l302_302126

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302126


namespace Felipe_time_to_build_house_l302_302628

variables (E F : ℝ)
variables (Felipe_building_time_months : ℝ) (Combined_time : ℝ := 7.5) (Half_time_relation : F = 1 / 2 * E)

-- Felipe finished his house in 30 months
theorem Felipe_time_to_build_house :
  (F = 1 / 2 * E) →
  (F + E = Combined_time) →
  (Felipe_building_time_months = F * 12) →
  Felipe_building_time_months = 30 :=
by
  intros h1 h2 h3
  -- Combining the given conditions to prove the statement
  sorry

end Felipe_time_to_build_house_l302_302628


namespace circumcenter_lies_on_AK_l302_302643

noncomputable def is_circumcenter (O : Point) (A B C : Point) : Prop :=
  dist O A = dist O B ∧ dist O B = dist O C

noncomputable def lies_on_line (P Q R : Point) : Prop :=
  ∃ (k : ℝ), Q = P + k • (R - P)

theorem circumcenter_lies_on_AK
  (A B C L H K O : Point)
  (h_triangle : ∀ (X Y Z : Point), X ≠ Y → X ≠ Z → Y ≠ Z → is_triangle X Y Z)
  (h_AL : is_angle_bisector A L B C)
  (h_H : foot B L H)
  (h_K : foot_on_circumcircle B L K (set_circumcircle A B L))
  (h_circ_A : O = is_circumcenter O A B C) :
  lies_on_line A K O :=
sorry

end circumcenter_lies_on_AK_l302_302643


namespace ratio_shirt_to_coat_l302_302534

-- Define the given conditions
def total_cost := 600
def shirt_cost := 150

-- Define the coat cost based on the given conditions
def coat_cost := total_cost - shirt_cost

-- State the theorem to prove the ratio of shirt cost to coat cost is 1:3
theorem ratio_shirt_to_coat : (shirt_cost : ℚ) / (coat_cost : ℚ) = 1 / 3 :=
by
  -- The proof would go here
  sorry

end ratio_shirt_to_coat_l302_302534


namespace exponential_function_solution_l302_302217

theorem exponential_function_solution (a : ℝ) (h : a > 1)
  (h_max_min_diff : a - a⁻¹ = 1) : a = (Real.sqrt 5 + 1) / 2 :=
sorry

end exponential_function_solution_l302_302217


namespace average_of_remaining_two_l302_302469

theorem average_of_remaining_two (a1 a2 a3 a4 a5 : ℚ)
  (h1 : (a1 + a2 + a3 + a4 + a5) / 5 = 11)
  (h2 : (a1 + a2 + a3) / 3 = 4) :
  ((a4 + a5) / 2 = 21.5) :=
sorry

end average_of_remaining_two_l302_302469


namespace probability_a_b_c_is_divisible_by_4_l302_302584

open Probability

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def probability_divisible_by_4 : ℚ := 
  let set := (Finset.range 4032).map Nat.succ in
  if h0 : ¬ Finset.Nonempty set then 0 else
  let S := Finset (set.product (set.product set)) in
  let events := S.filter (λ ⟨a, ⟨b, c⟩⟩, is_divisible_by_4 (a * (b * c + b + 1))) in
  (events.card : ℚ) / (S.card : ℚ)

theorem probability_a_b_c_is_divisible_by_4 : 
  probability_divisible_by_4 = 3 / 8 :=
sorry

end probability_a_b_c_is_divisible_by_4_l302_302584


namespace jenna_eel_is_16_l302_302251

open Real

def jenna_eel_len (J B : ℝ) : Prop :=
  J = (1 / 3) * B ∧ J + B = 64

theorem jenna_eel_is_16 : ∃ J B : ℝ, jenna_eel_len J B ∧ J = 16 :=
by
  exists 16
  exists 64 * (3 / 4)    -- which is 48
  unfold jenna_eel_len
  split
  { norm_num }
  { norm_num }

end jenna_eel_is_16_l302_302251


namespace same_different_color_ways_equal_l302_302573

-- Definitions based on conditions in the problem
def num_black : ℕ := 15
def num_white : ℕ := 10

def same_color_ways : ℕ :=
  Nat.choose num_black 2 + Nat.choose num_white 2

def different_color_ways : ℕ :=
  num_black * num_white

-- The proof statement
theorem same_different_color_ways_equal : same_color_ways = different_color_ways :=
by
  sorry

end same_different_color_ways_equal_l302_302573


namespace sqrt_450_simplified_l302_302040

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l302_302040


namespace sqrt_450_equals_15_sqrt_2_l302_302084

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l302_302084


namespace electric_poles_count_l302_302344

theorem electric_poles_count (dist interval: ℕ) (h_interval: interval = 25) (h_dist: dist = 1500):
  (dist / interval) + 1 = 61 := 
by
  -- Sorry to skip the proof steps
  sorry

end electric_poles_count_l302_302344


namespace least_number_subtracted_l302_302871

/-- The least number that must be subtracted from 50248 so that the 
remaining number is divisible by both 20 and 37 is 668. -/
theorem least_number_subtracted (n : ℕ) (x : ℕ ) (y : ℕ ) (a : ℕ) (b : ℕ) :
  n = 50248 → x = 20 → y = 37 → (a = 20 * 37) →
  (50248 - b) % a = 0 → 50248 - b < a → b = 668 :=
by
  sorry

end least_number_subtracted_l302_302871


namespace find_D_l302_302307

theorem find_D (A B C D : ℕ) (h₁ : A + A = 6) (h₂ : B - A = 4) (h₃ : C + B = 9) (h₄ : D - C = 7) : D = 9 :=
sorry

end find_D_l302_302307


namespace brenda_leads_by_5_l302_302793

theorem brenda_leads_by_5
  (initial_lead : ℕ)
  (brenda_play : ℕ)
  (david_play : ℕ)
  (h_initial : initial_lead = 22)
  (h_brenda_play : brenda_play = 15)
  (h_david_play : david_play = 32) :
  initial_lead + brenda_play - david_play = 5 :=
by {
  rw [h_initial, h_brenda_play, h_david_play],
  norm_num, -- simplify to get the answer
  sorry
}

end brenda_leads_by_5_l302_302793


namespace speed_boat_in_still_water_l302_302321

variable (V_b V_s t : ℝ)

def speed_of_boat := V_b

axiom stream_speed : V_s = 26

axiom time_relation : 2 * (t : ℝ) = 2 * t

axiom distance_relation : (V_b + V_s) * t = (V_b - V_s) * (2 * t)

theorem speed_boat_in_still_water : V_b = 78 :=
by {
  sorry
}

end speed_boat_in_still_water_l302_302321


namespace max_magnitude_vector_sub_l302_302682

open Real

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
sqrt (v.1^2 + v.2^2)

noncomputable def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
(v1.1 - v2.1, v1.2 - v2.2)

theorem max_magnitude_vector_sub (a b : ℝ × ℝ)
  (ha : vector_magnitude a = 2)
  (hb : vector_magnitude b = 1) :
  ∃ θ : ℝ, |vector_magnitude (vector_sub a b)| = 3 :=
by
  use π  -- θ = π to minimize cos θ to be -1
  sorry

end max_magnitude_vector_sub_l302_302682


namespace solution_set_l302_302805

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn (a b : ℝ) : f (a + b) = f a + f b - 1
axiom monotonic (x y : ℝ) : x ≤ y → f x ≤ f y
axiom initial_condition : f 4 = 5

theorem solution_set : {m : ℝ | f (3 * m^2 - m - 2) < 3} = {m : ℝ | -4/3 < m ∧ m < 1} :=
by
  sorry

end solution_set_l302_302805


namespace power_of_power_l302_302873

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := 
by sorry

end power_of_power_l302_302873


namespace number_to_add_l302_302766

theorem number_to_add (a m : ℕ) (h₁ : a = 7844213) (h₂ : m = 549) :
  ∃ n, (a + n) % m = 0 ∧ n = m - (a % m) :=
by
  sorry

end number_to_add_l302_302766


namespace molecular_weight_CaOH2_l302_302914

def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

theorem molecular_weight_CaOH2 :
  (atomic_weight_Ca + 2 * atomic_weight_O + 2 * atomic_weight_H = 74.10) := 
by 
  sorry

end molecular_weight_CaOH2_l302_302914


namespace probability_three_primes_l302_302188

def primes : List ℕ := [2, 3, 5, 7]

def is_prime (n : ℕ) : Prop := n ∈ primes

noncomputable def probability_prime : ℚ := 4/10
noncomputable def probability_non_prime : ℚ := 1 - probability_prime

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def calculation :
  ℚ := (choose 5 3) * (probability_prime ^ 3) * (probability_non_prime ^ 2)

theorem probability_three_primes :
  calculation = 720 / 3125 := by
  sorry

end probability_three_primes_l302_302188


namespace count_total_wheels_l302_302750

theorem count_total_wheels (trucks : ℕ) (cars : ℕ) (truck_wheels : ℕ) (car_wheels : ℕ) :
  trucks = 12 → cars = 13 → truck_wheels = 4 → car_wheels = 4 →
  (trucks * truck_wheels + cars * car_wheels) = 100 :=
by
  intros h_trucks h_cars h_truck_wheels h_car_wheels
  sorry

end count_total_wheels_l302_302750


namespace additional_investment_interest_rate_l302_302900

theorem additional_investment_interest_rate :
  let initial_investment := 2400
  let initial_rate := 0.05
  let additional_investment := 600
  let total_investment := initial_investment + additional_investment
  let desired_total_income := 0.06 * total_investment
  let income_from_initial := initial_rate * initial_investment
  let additional_income_needed := desired_total_income - income_from_initial
  let additional_rate := additional_income_needed / additional_investment
  additional_rate * 100 = 10 :=
by
  sorry

end additional_investment_interest_rate_l302_302900


namespace select_best_player_l302_302923

theorem select_best_player : 
  (average_A = 9.6 ∧ variance_A = 0.25) ∧ 
  (average_B = 9.5 ∧ variance_B = 0.27) ∧ 
  (average_C = 9.5 ∧ variance_C = 0.30) ∧ 
  (average_D = 9.6 ∧ variance_D = 0.23) → 
  best_player = D := 
by 
  sorry

end select_best_player_l302_302923


namespace decode_plaintext_l302_302925

theorem decode_plaintext (a x y : ℕ) (h1 : y = a^x - 2) (h2 : 6 = a^3 - 2) (h3 : y = 14) : x = 4 := by
  sorry

end decode_plaintext_l302_302925


namespace bridge_length_is_correct_l302_302878

def speed_km_hr : ℝ := 45
def train_length_m : ℝ := 120
def crossing_time_s : ℝ := 30

noncomputable def speed_m_s : ℝ := speed_km_hr * 1000 / 3600
noncomputable def total_distance_m : ℝ := speed_m_s * crossing_time_s
noncomputable def bridge_length_m : ℝ := total_distance_m - train_length_m

theorem bridge_length_is_correct : bridge_length_m = 255 := by
  sorry

end bridge_length_is_correct_l302_302878


namespace largest_value_of_c_l302_302205

theorem largest_value_of_c : ∃ c, (∀ x : ℝ, x^2 - 6 * x + c = 1 → c ≤ 10) :=
sorry

end largest_value_of_c_l302_302205


namespace sqrt_450_eq_15_sqrt_2_l302_302054

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302054


namespace f_at_3_l302_302803

theorem f_at_3 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 2) = x + 3) : f 3 = 4 := 
sorry

end f_at_3_l302_302803


namespace sine_probability_l302_302562

noncomputable def probability_sine_inequality : ℝ :=
Real.pi / 2 / Real.pi

theorem sine_probability (theta : ℝ) (h : 0 ≤ theta ∧ theta ≤ Real.pi) :
  prob (fun theta => Real.sin (theta + Real.pi / 3) < 1 / 2) [0, Real.pi] = (1 : ℝ) / 2 :=
begin
  -- Outline of proof needed
  sorry
end

end sine_probability_l302_302562


namespace molecular_weight_compound_l302_302633

def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00

def num_H : ℝ := 1
def num_Br : ℝ := 1
def num_O : ℝ := 3

def molecular_weight (num_H num_Br num_O atomic_weight_H atomic_weight_Br atomic_weight_O : ℝ) : ℝ :=
  (num_H * atomic_weight_H) + (num_Br * atomic_weight_Br) + (num_O * atomic_weight_O)

theorem molecular_weight_compound : molecular_weight num_H num_Br num_O atomic_weight_H atomic_weight_Br atomic_weight_O = 128.91 :=
by
  sorry

end molecular_weight_compound_l302_302633


namespace remainder_div_x_plus_2_l302_302636

def f (x : ℤ) : ℤ := x^15 + 3

theorem remainder_div_x_plus_2 : f (-2) = -32765 := by
  sorry

end remainder_div_x_plus_2_l302_302636


namespace remaining_cookies_l302_302837

variable (total_initial_cookies : ℕ)
variable (cookies_taken_day1 : ℕ := 3)
variable (cookies_taken_day2 : ℕ := 3)
variable (cookies_eaten_day2 : ℕ := 1)
variable (cookies_put_back_day2 : ℕ := 2)
variable (cookies_taken_by_junior : ℕ := 7)

theorem remaining_cookies (total_initial_cookies cookies_taken_day1 cookies_taken_day2
                          cookies_eaten_day2 cookies_put_back_day2 cookies_taken_by_junior : ℕ) :
  (total_initial_cookies = 2 * (cookies_taken_day1 + cookies_eaten_day2 + cookies_taken_by_junior))
  → (total_initial_cookies - (cookies_taken_day1 + cookies_eaten_day2 + cookies_taken_by_junior) = 11) :=
by
  sorry

end remaining_cookies_l302_302837


namespace bales_in_barn_now_l302_302998

-- Define the initial number of bales
def initial_bales : ℕ := 28

-- Define the number of bales added by Tim
def added_bales : ℕ := 26

-- Define the total number of bales
def total_bales : ℕ := initial_bales + added_bales

-- Theorem stating the total number of bales
theorem bales_in_barn_now : total_bales = 54 := by
  sorry

end bales_in_barn_now_l302_302998


namespace find_range_of_m_l302_302219

def proposition_p (m : ℝ) : Prop := 0 < m ∧ m < 1/3
def proposition_q (m : ℝ) : Prop := 0 < m ∧ m < 15
def proposition_r (m : ℝ) : Prop := proposition_p m ∨ proposition_q m
def proposition_s (m : ℝ) : Prop := proposition_p m ∧ proposition_q m = False
def range_of_m (m : ℝ) : Prop := 1/3 ≤ m ∧ m < 15

theorem find_range_of_m (m : ℝ) : proposition_r m ∧ proposition_s m → range_of_m m := by
  sorry

end find_range_of_m_l302_302219


namespace shaded_area_is_correct_l302_302443

-- Defining the conditions
def grid_width : ℝ := 15 -- in units
def grid_height : ℝ := 5 -- in units
def total_grid_area : ℝ := grid_width * grid_height -- in square units

def larger_triangle_base : ℝ := grid_width -- in units
def larger_triangle_height : ℝ := grid_height -- in units
def larger_triangle_area : ℝ := 0.5 * larger_triangle_base * larger_triangle_height -- in square units

def smaller_triangle_base : ℝ := 3 -- in units
def smaller_triangle_height : ℝ := 2 -- in units
def smaller_triangle_area : ℝ := 0.5 * smaller_triangle_base * smaller_triangle_height -- in square units

-- The total area of the triangles that are not shaded
def unshaded_areas : ℝ := larger_triangle_area + smaller_triangle_area

-- The area of the shaded region
def shaded_area : ℝ := total_grid_area - unshaded_areas

-- The statement to be proven
theorem shaded_area_is_correct : shaded_area = 34.5 := 
by 
  -- This is a placeholder for the actual proof, which would normally go here
  sorry

end shaded_area_is_correct_l302_302443


namespace simplify_sqrt_450_l302_302014
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l302_302014


namespace simplify_sqrt_450_l302_302108

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l302_302108


namespace find_a33_l302_302665

theorem find_a33 : 
  ∀ (a : ℕ → ℤ), a 1 = 3 → a 2 = 6 → (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) → a 33 = 3 :=
by
  intros a h1 h2 h_rec
  sorry

end find_a33_l302_302665


namespace units_digit_proof_l302_302917

def units_digit (n : ℤ) : ℤ := n % 10

theorem units_digit_proof :
  ∀ (a b c : ℤ),
  a = 8 →
  b = 18 →
  c = 1988 →
  units_digit (a * b * c - a^3) = 0 :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc]
  -- Proof will go here
  sorry

end units_digit_proof_l302_302917


namespace minimum_possible_value_l302_302716

-- Define the set of distinct elements
def distinct_elems : Set ℤ := {-8, -6, -4, -1, 1, 3, 7, 12}

-- Define the existence of distinct elements
def elem_distinct (p q r s t u v w : ℤ) : Prop :=
  p ∈ distinct_elems ∧ q ∈ distinct_elems ∧ r ∈ distinct_elems ∧ s ∈ distinct_elems ∧ 
  t ∈ distinct_elems ∧ u ∈ distinct_elems ∧ v ∈ distinct_elems ∧ w ∈ distinct_elems ∧ 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧ 
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
  r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧ 
  s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧ 
  t ≠ u ∧ t ≠ v ∧ t ≠ w ∧ 
  u ≠ v ∧ u ≠ w ∧ 
  v ≠ w

-- The main proof problem
theorem minimum_possible_value :
  ∀ (p q r s t u v w : ℤ), elem_distinct p q r s t u v w ->
  (p + q + r + s)^2 + (t + u + v + w)^2 = 10 := 
sorry

end minimum_possible_value_l302_302716


namespace sqrt_450_simplified_l302_302037

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l302_302037


namespace set_intersection_problem_l302_302341

def set_product (A B : Set ℕ) : Set ℕ := {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y}
def A : Set ℕ := {0, 2}
def B : Set ℕ := {1, 3}
def C : Set ℕ := {x | x^2 - 3 * x + 2 = 0}

theorem set_intersection_problem :
  (set_product A B) ∩ (set_product B C) = {2, 6} :=
by
  sorry

end set_intersection_problem_l302_302341


namespace original_length_of_wood_l302_302650

theorem original_length_of_wood (s cl ol : ℝ) (h1 : s = 2.3) (h2 : cl = 6.6) (h3 : ol = cl + s) : 
  ol = 8.9 := 
by 
  sorry

end original_length_of_wood_l302_302650


namespace Sn_solution_l302_302812

-- Definitions for the problem
def sequence_sum (a : ℕ → ℝ) (n : ℕ) := ∑ i in range n, a (i + 1)
def a_1 : ℝ := 2
def S_n (a : ℕ → ℝ) (n : ℕ) := sequence_sum a n
def S_eq (a : ℕ → ℝ) (n : ℕ) : ℝ := S_n a n

-- Condition for n >= 2
def cond (a : ℕ → ℝ) (n : ℕ) : Prop :=
  n ≥ 2 ∧ S_eq a n ≠ S_eq a (n - 1)

-- Theorem statement
theorem Sn_solution (a : ℕ → ℝ) (n : ℕ) (hn : n ≥ 2) (h_sum_cond : ∀ n, S_eq a n * S_eq a (n - 1) + a n = 0) :
  S_eq a n = 2 / (2 * n - 1) :=
by {
  sorry
}

end Sn_solution_l302_302812


namespace total_students_l302_302563

theorem total_students (h1 : 15 * 70 = 1050) 
                       (h2 : 10 * 95 = 950) 
                       (h3 : 1050 + 950 = 2000)
                       (h4 : 80 * N = 2000) :
  N = 25 :=
by sorry

end total_students_l302_302563


namespace term_transition_addition_l302_302592

theorem term_transition_addition (k : Nat) :
  (2:ℚ) / ((k + 1) * (k + 2)) = ((2:ℚ) / ((k * (k + 1))) - ((2:ℚ) / ((k + 1) * (k + 2)))) := 
sorry

end term_transition_addition_l302_302592


namespace pencils_bought_l302_302383

theorem pencils_bought (payment change pencil_cost glue_cost : ℕ)
  (h_payment : payment = 1000)
  (h_change : change = 100)
  (h_pencil_cost : pencil_cost = 210)
  (h_glue_cost : glue_cost = 270) :
  (payment - change - glue_cost) / pencil_cost = 3 :=
by sorry

end pencils_bought_l302_302383


namespace sufficient_not_necessary_condition_l302_302768

theorem sufficient_not_necessary_condition (a : ℝ) : (a = 2 → (a^2 - a) * 1 + 1 = 0) ∧ (¬ ((a^2 - a) * 1 + 1 = 0 → a = 2)) :=
by sorry

end sufficient_not_necessary_condition_l302_302768


namespace points_per_right_answer_l302_302463

variable (p : ℕ)
variable (total_problems : ℕ := 25)
variable (wrong_problems : ℕ := 3)
variable (score : ℤ := 85)

theorem points_per_right_answer :
  (total_problems - wrong_problems) * p - wrong_problems = score -> p = 4 :=
  sorry

end points_per_right_answer_l302_302463


namespace value_of_expression_l302_302402

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := by
  sorry

end value_of_expression_l302_302402


namespace solution_1_solution_2_l302_302815

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - (a + 1) * x + Real.log x

def critical_point_condition (a x : ℝ) : Prop :=
  (x = 1 / 4) → deriv (f a) x = 0

def pseudo_symmetry_point_condition (a : ℝ) (x0 : ℝ) : Prop :=
  let f' := fun x => 2 * x^2 - 5 * x + Real.log x
  let g := fun x => (4 * x0^2 - 5 * x0 + 1) / x0 * (x - x0) + 2 * x0^2 - 5 * x0 + Real.log x0
  ∀ x : ℝ, 
    (0 < x ∧ x < x0) → (f' x - g x < 0) ∧ 
    (x > x0) → (f' x - g x > 0)

theorem solution_1 (a : ℝ) (h1 : a > 0) (h2 : critical_point_condition a (1/4)) :
  a = 4 := 
sorry

theorem solution_2 (x0 : ℝ) (h1 : x0 = 1/2) :
  pseudo_symmetry_point_condition 4 x0 :=
sorry


end solution_1_solution_2_l302_302815


namespace num_tickets_bought_l302_302887

-- Defining the cost and discount conditions
def ticket_cost : ℝ := 40
def discount_rate : ℝ := 0.05
def total_paid : ℝ := 476
def base_tickets : ℕ := 10

-- Definition to calculate the cost of the first 10 tickets
def cost_first_10_tickets : ℝ := base_tickets * ticket_cost
-- Definition of the discounted price for tickets exceeding 10
def discounted_ticket_cost : ℝ := ticket_cost * (1 - discount_rate)
-- Definition of the total cost for the tickets exceeding 10
def cost_discounted_tickets (num_tickets_exceeding_10 : ℕ) : ℝ := num_tickets_exceeding_10 * discounted_ticket_cost
-- Total amount spent on the tickets exceeding 10
def amount_spent_on_discounted_tickets : ℝ := total_paid - cost_first_10_tickets

-- Main theorem statement proving the total number of tickets Mr. Benson bought
theorem num_tickets_bought : ∃ x : ℕ, x = base_tickets + (amount_spent_on_discounted_tickets / discounted_ticket_cost) ∧ x = 12 := 
by
  sorry

end num_tickets_bought_l302_302887


namespace ksyusha_travel_time_l302_302258

theorem ksyusha_travel_time :
  ∀ (S v : ℝ), 
  (2 * S / v + S / (2 * v) = 30) → 
  (2 * (S / v) = 24) :=
by
  intros S v h1
  have h2 : S / v = 12 := by
    calc 
      S / v = (30 * 2) / 5 := by sorry
  calc 
    2 * (S / v) = 2 * 12 := by sorry
  ... = 24 := by norm_num

end ksyusha_travel_time_l302_302258


namespace sqrt_450_eq_15_sqrt_2_l302_302057

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302057


namespace simplify_expression_l302_302134

variable (x y : ℝ)

theorem simplify_expression : 
  (3 * x^2 * y - 2 * x * y^2) - (x * y^2 - 2 * x^2 * y) - 2 * (-3 * x^2 * y - x * y^2) = 26 := by
  -- Given conditions
  let x := -1
  let y := 2
  -- Proof to be provided
  sorry

end simplify_expression_l302_302134


namespace sqrt_450_eq_15_sqrt_2_l302_302112

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302112


namespace gcd_sequence_property_l302_302144

theorem gcd_sequence_property (a : ℕ → ℕ) (m n : ℕ) (h : ∀ m n, m > n → Nat.gcd (a m) (a n) = Nat.gcd (a (m - n)) (a n)) : 
  Nat.gcd (a m) (a n) = a (Nat.gcd m n) :=
by
  sorry

end gcd_sequence_property_l302_302144


namespace intersection_of_A_and_complement_of_B_l302_302555

noncomputable def U : Set ℝ := Set.univ

noncomputable def A : Set ℝ := { x : ℝ | 2^x * (x - 2) < 1 }
noncomputable def B : Set ℝ := { x : ℝ | ∃ y : ℝ, y = Real.log (1 - x) }
noncomputable def B_complement : Set ℝ := { x : ℝ | x ≥ 1 }

theorem intersection_of_A_and_complement_of_B :
  A ∩ B_complement = { x : ℝ | 1 ≤ x ∧ x < 2 } :=
by
  sorry

end intersection_of_A_and_complement_of_B_l302_302555


namespace arctan_sum_of_roots_l302_302984

open Real

theorem arctan_sum_of_roots : 
  let x1 := (-sin (3 * π / 5) + sqrt (sin (3 * π / 5)^2 - 4 * cos (3 * π / 5))) / 2
  let x2 := (-sin (3 * π / 5) - sqrt (sin (3 * π / 5)^2 - 4 * cos (3 * π / 5))) / 2 in
  arctan x1 + arctan x2 = π / 5 :=
sorry

end arctan_sum_of_roots_l302_302984


namespace average_salary_for_company_l302_302162

variable (n_m : ℕ) -- number of managers
variable (n_a : ℕ) -- number of associates
variable (avg_salary_m : ℕ) -- average salary of managers
variable (avg_salary_a : ℕ) -- average salary of associates

theorem average_salary_for_company (h_n_m : n_m = 15) (h_n_a : n_a = 75) 
  (h_avg_salary_m : avg_salary_m = 90000) (h_avg_salary_a : avg_salary_a = 30000) : 
  (n_m * avg_salary_m + n_a * avg_salary_a) / (n_m + n_a) = 40000 := 
by
  sorry

end average_salary_for_company_l302_302162


namespace fifth_observation_l302_302470

theorem fifth_observation (O1 O2 O3 O4 O5 O6 O7 O8 O9 : ℝ)
  (h1 : O1 + O2 + O3 + O4 + O5 + O6 + O7 + O8 + O9 = 72)
  (h2 : O1 + O2 + O3 + O4 + O5 = 50)
  (h3 : O5 + O6 + O7 + O8 + O9 = 40) :
  O5 = 18 := 
  sorry

end fifth_observation_l302_302470


namespace simplify_sqrt_450_l302_302093

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l302_302093


namespace living_room_area_l302_302314

-- Define the conditions
def carpet_area (length width : ℕ) : ℕ :=
  length * width

def percentage_coverage (carpet_area living_room_area : ℕ) : ℕ :=
  (carpet_area * 100) / living_room_area

-- State the problem
theorem living_room_area (A : ℕ) (carpet_len carpet_wid : ℕ) (carpet_coverage : ℕ) :
  carpet_len = 4 → carpet_wid = 9 → carpet_coverage = 20 →
  20 * A = 36 * 100 → A = 180 :=
by
  intros h_len h_wid h_coverage h_proportion
  sorry

end living_room_area_l302_302314


namespace find_initial_order_l302_302515

variables (x : ℕ)

def initial_order (x : ℕ) :=
  x + 60 = 72 * (x / 90 + 1)

theorem find_initial_order (h1 : initial_order x) : x = 60 :=
  sorry

end find_initial_order_l302_302515


namespace simplify_sqrt_450_l302_302099

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l302_302099


namespace find_value_of_fraction_l302_302415

theorem find_value_of_fraction (a b c d: ℕ) (h1: a = 4 * b) (h2: b = 3 * c) (h3: c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end find_value_of_fraction_l302_302415


namespace ksyusha_travel_time_l302_302257

theorem ksyusha_travel_time :
  ∀ (S v : ℝ), 
  (2 * S / v + S / (2 * v) = 30) → 
  (2 * (S / v) = 24) :=
by
  intros S v h1
  have h2 : S / v = 12 := by
    calc 
      S / v = (30 * 2) / 5 := by sorry
  calc 
    2 * (S / v) = 2 * 12 := by sorry
  ... = 24 := by norm_num

end ksyusha_travel_time_l302_302257


namespace coin_difference_l302_302301

variables (x y : ℕ)

theorem coin_difference (h1 : x + y = 15) (h2 : 2 * x + 5 * y = 51) : x - y = 1 := by
  sorry

end coin_difference_l302_302301


namespace sqrt_450_eq_15_sqrt_2_l302_302055

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302055


namespace cubes_difference_divisible_91_l302_302459

theorem cubes_difference_divisible_91 (cubes : Fin 16 → ℤ) (h : ∀ n : Fin 16, ∃ m : ℤ, cubes n = m^3) :
  ∃ (a b : Fin 16), a ≠ b ∧ 91 ∣ (cubes a - cubes b) :=
sorry

end cubes_difference_divisible_91_l302_302459


namespace black_white_ratio_l302_302242

theorem black_white_ratio 
  (x y : ℕ) 
  (h1 : (y - 1) * 7 = x * 9) 
  (h2 : y * 5 = (x - 1) * 7) : 
  y - x = 7 := 
by 
  sorry

end black_white_ratio_l302_302242


namespace problem_statement_l302_302360

noncomputable def tangent_sum_formula (x y : ℝ) : ℝ :=
  (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y)

theorem problem_statement
  (α β : ℝ)
  (hαβ1 : 0 < α ∧ α < π)
  (hαβ2 : 0 < β ∧ β < π)
  (h1 : Real.tan (α - β) = 1 / 2)
  (h2 : Real.tan β = - 1 / 7)
  : 2 * α - β = - (3 * π / 4) :=
sorry

end problem_statement_l302_302360


namespace neon_sign_blink_interval_l302_302152

theorem neon_sign_blink_interval :
  ∃ (b : ℕ), (∀ t : ℕ, t > 0 → (t % 9 = 0 ∧ t % b = 0 ↔ t % 45 = 0)) → b = 15 :=
by
  sorry

end neon_sign_blink_interval_l302_302152


namespace total_calories_in_jerrys_breakfast_l302_302252

theorem total_calories_in_jerrys_breakfast :
  let pancakes := 7 * 120
  let bacon := 3 * 100
  let orange_juice := 2 * 300
  let cereal := 1 * 200
  let chocolate_muffin := 1 * 350
  pancakes + bacon + orange_juice + cereal + chocolate_muffin = 2290 :=
by
  -- Proof omitted
  sorry

end total_calories_in_jerrys_breakfast_l302_302252


namespace final_price_after_discounts_l302_302589

theorem final_price_after_discounts (original_price : ℝ)
  (first_discount_pct : ℝ) (second_discount_pct : ℝ) (third_discount_pct : ℝ) :
  original_price = 200 → 
  first_discount_pct = 0.40 → 
  second_discount_pct = 0.20 → 
  third_discount_pct = 0.10 → 
  (original_price * (1 - first_discount_pct) * (1 - second_discount_pct) * (1 - third_discount_pct) = 86.40) := 
by
  intros
  sorry

end final_price_after_discounts_l302_302589


namespace cheryl_distance_walked_l302_302525

theorem cheryl_distance_walked (speed : ℕ) (time : ℕ) (distance_away : ℕ) (distance_home : ℕ) 
  (h1 : speed = 2) 
  (h2 : time = 3) 
  (h3 : distance_away = speed * time) 
  (h4 : distance_home = distance_away) : 
  distance_away + distance_home = 12 := 
by
  sorry

end cheryl_distance_walked_l302_302525


namespace compute_a1d1_a2d2_a3d3_l302_302448

noncomputable def polynomial_equation (a1 a2 a3 d1 d2 d3: ℝ) : Prop :=
  ∀ x : ℝ, (x^6 + x^5 + x^4 + x^3 + x^2 + x + 1) = (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3)

theorem compute_a1d1_a2d2_a3d3 (a1 a2 a3 d1 d2 d3 : ℝ) (h : polynomial_equation a1 a2 a3 d1 d2 d3) : 
  a1 * d1 + a2 * d2 + a3 * d3 = 1 :=
  sorry

end compute_a1d1_a2d2_a3d3_l302_302448


namespace solve_abs_equation_l302_302287

theorem solve_abs_equation (y : ℝ) (h8 : y < 8) (h_eq : |y - 8| + 2 * y = 12) : y = 4 :=
sorry

end solve_abs_equation_l302_302287


namespace valid_schedule_count_l302_302941

theorem valid_schedule_count :
  ∃ (valid_schedules : Finset (Fin 8 → Option (Fin 4))),
    valid_schedules.card = 488 ∧
    (∀ (schedule : Fin 8 → Option (Fin 4)), schedule ∈ valid_schedules →
      (∀ i : Fin 7, schedule i ≠ none ∧ schedule (i + 1) ≠ schedule i) ∧
      schedule 4 = none) :=
sorry

end valid_schedule_count_l302_302941


namespace ksyusha_travel_time_l302_302269

variables (v S : ℝ)

theorem ksyusha_travel_time :
  (2 * S / v) + (S / (2 * v)) = 30 →
  (S / v) + (2 * S / (2 * v)) = 24 :=
by sorry

end ksyusha_travel_time_l302_302269


namespace simplify_expression_l302_302133

variable (x y : ℝ)

theorem simplify_expression : 
  (3 * x^2 * y - 2 * x * y^2) - (x * y^2 - 2 * x^2 * y) - 2 * (-3 * x^2 * y - x * y^2) = 26 := by
  -- Given conditions
  let x := -1
  let y := 2
  -- Proof to be provided
  sorry

end simplify_expression_l302_302133


namespace ksyusha_travel_time_wednesday_l302_302264

-- Definitions based on the conditions
def speed_walk (v : ℝ) : ℝ := v
def speed_run (v : ℝ) : ℝ := 2 * v
def distance_walk_tuesday (S : ℝ) : ℝ := 2 * S
def distance_run_tuesday (S : ℝ) : ℝ := S
def total_time_tuesday (v S : ℝ) : ℝ := (distance_walk_tuesday S) / (speed_walk v) + (distance_run_tuesday S) / (speed_run v)

-- Theorem statement
theorem ksyusha_travel_time_wednesday (S v : ℝ) (h : total_time_tuesday v S = 30) :
  (S / v + S / v) = 24 := 
by sorry

end ksyusha_travel_time_wednesday_l302_302264


namespace simplify_sqrt_450_l302_302064

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l302_302064


namespace speed_of_sound_l302_302770

theorem speed_of_sound (d₁ d₂ t : ℝ) (speed_car : ℝ) (speed_km_hr_to_m_s : ℝ) :
  d₁ = 1200 ∧ speed_car = 108 ∧ speed_km_hr_to_m_s = (speed_car * 1000 / 3600) ∧ t = 3.9669421487603307 →
  (d₁ + speed_km_hr_to_m_s * t) / t = 332.59 :=
by sorry

end speed_of_sound_l302_302770


namespace geometric_sequence_a1_range_l302_302549

theorem geometric_sequence_a1_range (a : ℕ → ℝ) (b : ℕ → ℝ) (a1 : ℝ) :
  (∀ n, a (n+1) = a n / 2) ∧ (∀ n, b n = n / 2) ∧ (∃! n : ℕ, a n > b n) →
  (6 < a1 ∧ a1 ≤ 16) :=
by
  sorry

end geometric_sequence_a1_range_l302_302549


namespace expression_divisible_by_11_l302_302594

theorem expression_divisible_by_11 (n : ℕ) : (6^(2*n) + 3^(n+2) + 3^n) % 11 = 0 := 
sorry

end expression_divisible_by_11_l302_302594


namespace notebook_costs_2_20_l302_302324

theorem notebook_costs_2_20 (n c : ℝ) (h1 : n + c = 2.40) (h2 : n = 2 + c) : n = 2.20 :=
by
  sorry

end notebook_costs_2_20_l302_302324


namespace number_of_possible_k_values_l302_302442

theorem number_of_possible_k_values : 
  ∃ k_values : Finset ℤ, 
    (∀ k ∈ k_values, ∃ (x y : ℤ), y = x - 3 ∧ y = k * x - k) ∧
    k_values.card = 3 := 
sorry

end number_of_possible_k_values_l302_302442


namespace johns_elevation_after_travel_l302_302709

-- Definitions based on conditions:
def initial_elevation : ℝ := 400
def downward_rate : ℝ := 10
def time_travelled : ℕ := 5

-- Proof statement:
theorem johns_elevation_after_travel:
  initial_elevation - (downward_rate * time_travelled) = 350 :=
by
  sorry

end johns_elevation_after_travel_l302_302709


namespace find_value_of_fraction_l302_302413

theorem find_value_of_fraction (a b c d: ℕ) (h1: a = 4 * b) (h2: b = 3 * c) (h3: c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end find_value_of_fraction_l302_302413


namespace find_m_l302_302664

theorem find_m (m l : ℝ) (a b : ℝ × ℝ) (h_a : a = (2, m)) (h_b : b = (l, -2))
  (h_parallel : ∃ k : ℝ, k ≠ 0 ∧ a = k • (a + 2 • b)) :
  m = -4 :=
by
  sorry

end find_m_l302_302664


namespace number_of_ways_to_prepare_all_elixirs_l302_302445

def fairy_methods : ℕ := 2
def elf_methods : ℕ := 2
def fairy_elixirs : ℕ := 3
def elf_elixirs : ℕ := 4

theorem number_of_ways_to_prepare_all_elixirs : 
  (fairy_methods * fairy_elixirs) + (elf_methods * elf_elixirs) = 14 :=
by
  sorry

end number_of_ways_to_prepare_all_elixirs_l302_302445


namespace roman_created_171_roman_created_1513_m1_roman_created_1513_m2_roman_created_largest_l302_302850

-- Lean 4 statements to capture the proofs without computation.
theorem roman_created_171 (a b : ℕ) (h_sum : a + b = 17) (h_diff : a - b = 1) : 
  a = 9 ∧ b = 8 ∨ a = 8 ∧ b = 9 := 
  sorry

theorem roman_created_1513_m1 (a b : ℕ) (h_sum : a + b = 15) (h_diff : a - b = 13) : 
  a = 14 ∧ b = 1 ∨ a = 1 ∧ b = 14 := 
  sorry

theorem roman_created_1513_m2 (a b : ℕ) (h_sum : a + b = 151) (h_diff : a - b = 3) : 
  a = 77 ∧ b = 74 ∨ a = 74 ∧ b = 77 := 
  sorry

theorem roman_created_largest (a b : ℕ) (h_sum : a + b = 188) (h_diff : a - b = 10) : 
  a = 99 ∧ b = 89 ∨ a = 89 ∧ b = 99 := 
  sorry

end roman_created_171_roman_created_1513_m1_roman_created_1513_m2_roman_created_largest_l302_302850


namespace ball_returns_to_bella_after_13_throws_l302_302623

theorem ball_returns_to_bella_after_13_throws:
  ∀ (girls : Fin 13) (n : ℕ), (∃ k, k > 0 ∧ (1 + k * 5) % 13 = 1) → (n = 13) :=
by
  sorry

end ball_returns_to_bella_after_13_throws_l302_302623


namespace nancy_kept_chips_correct_l302_302456

/-- Define the initial conditions -/
def total_chips : ℕ := 22
def chips_to_brother : ℕ := 7
def chips_to_sister : ℕ := 5

/-- Define the number of chips Nancy kept -/
def chips_kept : ℕ := total_chips - (chips_to_brother + chips_to_sister)

theorem nancy_kept_chips_correct : chips_kept = 10 := by
  /- This is a placeholder. The proof would go here. -/
  sorry

end nancy_kept_chips_correct_l302_302456


namespace sixth_graders_bought_more_pencils_23_l302_302602

open Int

-- Conditions
def pencils_cost_whole_number_cents : Prop := ∃ n : ℕ, n > 0
def seventh_graders_total_cents := 165
def sixth_graders_total_cents := 234
def number_of_sixth_graders := 30

-- The number of sixth graders who bought more pencils than seventh graders
theorem sixth_graders_bought_more_pencils_23 :
  (seventh_graders_total_cents / 3 = 55) ∧
  (sixth_graders_total_cents / 3 = 78) →
  78 - 55 = 23 :=
by
  sorry

end sixth_graders_bought_more_pencils_23_l302_302602


namespace relationship_among_log_sin_exp_l302_302808

theorem relationship_among_log_sin_exp (x : ℝ) (h₁ : 0 < x) (h₂ : x < 1) (a b c : ℝ) 
(h₃ : a = Real.log 3 / Real.log x) (h₄ : b = Real.sin x)
(h₅ : c = 2 ^ x) : a < b ∧ b < c := 
sorry

end relationship_among_log_sin_exp_l302_302808


namespace arithmetic_mean_l302_302800

variable {x b c : ℝ}

theorem arithmetic_mean (hx : x ≠ 0) (hb : b ≠ c) : 
  (1 / 2) * ((x + b) / x + (x - c) / x) = 1 + (b - c) / (2 * x) :=
by
  sorry

end arithmetic_mean_l302_302800


namespace incorrect_conclusion_intersection_l302_302921

theorem incorrect_conclusion_intersection :
  ∀ (x : ℝ), (0 = -2 * x + 4) → (x = 2) :=
by
  intro x h
  sorry

end incorrect_conclusion_intersection_l302_302921


namespace sum_of_sequence_l302_302813

theorem sum_of_sequence (n : ℕ) (h : n ≥ 2) 
  (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, n ≥ 2 → S n * S (n-1) + a n = 0) :
  S n = 2 / (2 * n - 1) := by
  sorry

end sum_of_sequence_l302_302813


namespace complex_multiplication_quadrant_l302_302228

-- Given conditions
def complex_mul (z1 z2 : ℂ) : ℂ := z1 * z2

-- Proving point is in the fourth quadrant
theorem complex_multiplication_quadrant
  (a b : ℝ) (z : ℂ)
  (h1 : z = a + b * Complex.I)
  (h2 : z = complex_mul (1 + Complex.I) (3 - Complex.I)) :
  b < 0 ∧ a > 0 :=
by
  sorry

end complex_multiplication_quadrant_l302_302228


namespace find_number_l302_302177

theorem find_number (n : ℝ) (h : n - (1004 / 20.08) = 4970) : n = 5020 := 
by {
  sorry
}

end find_number_l302_302177


namespace sqrt_450_eq_15_sqrt_2_l302_302072

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l302_302072


namespace simplify_sqrt_450_l302_302022

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l302_302022


namespace sqrt_450_eq_15_sqrt_2_l302_302110

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302110


namespace simplify_expression_l302_302521

theorem simplify_expression (x : ℝ) (h₁ : x ≠ -1) (h₂: x ≠ -3) :
  (x - 1 - 8 / (x + 1)) / ( (x + 3) / (x + 1) ) = x - 3 :=
by
  sorry

end simplify_expression_l302_302521


namespace frogs_count_l302_302958

variables (Alex Brian Chris LeRoy Mike : Type) 

-- Definitions for the species
def toad (x : Type) : Prop := ∃ p : Prop, p -- Dummy definition for toads
def frog (x : Type) : Prop := ∃ p : Prop, ¬p -- Dummy definition for frogs

-- Conditions
axiom Alex_statement : (toad Alex) → (∃ x : ℕ, x = 3) ∧ (frog Alex) → (¬(∃ x : ℕ, x = 3))
axiom Brian_statement : (toad Brian) → (toad Mike) ∧ (frog Brian) → (frog Mike)
axiom Chris_statement : (toad Chris) → (toad LeRoy) ∧ (frog Chris) → (frog LeRoy)
axiom LeRoy_statement : (toad LeRoy) → (toad Chris) ∧ (frog LeRoy) → (frog Chris)
axiom Mike_statement : (toad Mike) → (∃ x : ℕ, x < 3) ∧ (frog Mike) → (¬(∃ x : ℕ, x < 3))

theorem frogs_count (total : ℕ) : total = 5 → 
  (∃ frog_count : ℕ, frog_count = 2) :=
by
  -- Leaving the proof as a sorry placeholder
  sorry

end frogs_count_l302_302958


namespace johns_average_speed_is_correct_l302_302255

noncomputable def johnsAverageSpeed : ℝ :=
  let total_time : ℝ := 6 + 0.5 -- Total driving time in hours
  let total_distance : ℝ := 210 -- Total distance covered in miles
  total_distance / total_time -- Average speed formula

theorem johns_average_speed_is_correct :
  johnsAverageSpeed = 32.31 :=
by
  -- This is a placeholder for the proof
  sorry

end johns_average_speed_is_correct_l302_302255


namespace sample_group_b_correct_l302_302748

noncomputable def stratified_sample_group_b (total_cities: ℕ) (group_b_cities: ℕ) (sample_size: ℕ) : ℕ :=
  (sample_size * group_b_cities) / total_cities

theorem sample_group_b_correct : stratified_sample_group_b 36 12 12 = 4 := by
  sorry

end sample_group_b_correct_l302_302748


namespace sampling_method_selection_l302_302762

-- Define the sampling methods as data type
inductive SamplingMethod
| SimpleRandomSampling : SamplingMethod
| SystematicSampling : SamplingMethod
| StratifiedSampling : SamplingMethod
| SamplingWithReplacement : SamplingMethod

-- Define our conditions
def basketballs : Nat := 10
def is_random_selection : Bool := true
def no_obvious_stratification : Bool := true

-- The theorem to prove the correct sampling method
theorem sampling_method_selection 
  (b : Nat) 
  (random_selection : Bool) 
  (no_stratification : Bool) : 
  SamplingMethod :=
  if b = 10 ∧ random_selection ∧ no_stratification then SamplingMethod.SimpleRandomSampling 
  else sorry

-- Prove the correct sampling method given our conditions
example : sampling_method_selection basketballs is_random_selection no_obvious_stratification = SamplingMethod.SimpleRandomSampling := 
by
-- skipping the proof here with sorry
sorry

end sampling_method_selection_l302_302762


namespace tan_product_identity_l302_302880

theorem tan_product_identity : (1 + Real.tan (Real.pi / 180 * 17)) * (1 + Real.tan (Real.pi / 180 * 28)) = 2 := by
  sorry

end tan_product_identity_l302_302880


namespace find_g_l302_302277

noncomputable def g (x : ℝ) : ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, (g x * g y - g (x * y)) / 4 = x + y + 4

theorem find_g :
  (∀ x : ℝ, g x = x + 5) ∨ (∀ x : ℝ, g x = -x - 3) := 
by
  sorry

end find_g_l302_302277


namespace two_numbers_sum_gcd_l302_302539

theorem two_numbers_sum_gcd (x y : ℕ) (h1 : x + y = 432) (h2 : Nat.gcd x y = 36) :
  (x = 36 ∧ y = 396) ∨ (x = 180 ∧ y = 252) ∨ (x = 396 ∧ y = 36) ∨ (x = 252 ∧ y = 180) :=
by
  -- Proof TBD
  sorry

end two_numbers_sum_gcd_l302_302539


namespace multiple_is_eight_l302_302648

theorem multiple_is_eight (m : ℝ) (h : 17 = m * 2.625 - 4) : m = 8 :=
by
  sorry

end multiple_is_eight_l302_302648


namespace simplify_sqrt_450_l302_302026

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l302_302026


namespace sqrt_450_eq_15_sqrt_2_l302_302075

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l302_302075


namespace percent_non_unionized_women_is_80_l302_302876

noncomputable def employeeStatistics :=
  let total_employees := 100
  let percent_men := 50
  let percent_unionized := 60
  let percent_unionized_men := 70
  let men := (percent_men / 100) * total_employees
  let unionized := (percent_unionized / 100) * total_employees
  let unionized_men := (percent_unionized_men / 100) * unionized
  let non_unionized_men := men - unionized_men
  let non_unionized := total_employees - unionized
  let non_unionized_women := non_unionized - non_unionized_men
  let percent_non_unionized_women := (non_unionized_women / non_unionized) * 100
  percent_non_unionized_women

theorem percent_non_unionized_women_is_80 :
  employeeStatistics = 80 :=
by
  sorry

end percent_non_unionized_women_is_80_l302_302876


namespace numberOfColoringWays_l302_302208

-- Define the problem parameters
def totalBalls : Nat := 5
def redBalls : Nat := 1
def blueBalls : Nat := 1
def yellowBalls : Nat := 2
def whiteBalls : Nat := 1

-- Show that the number of permutations of the multiset is 60
theorem numberOfColoringWays : (Nat.factorial totalBalls) / ((Nat.factorial redBalls) * (Nat.factorial blueBalls) * (Nat.factorial yellowBalls) * (Nat.factorial whiteBalls)) = 60 :=
  by
  simp [totalBalls, redBalls, blueBalls, yellowBalls, whiteBalls]
  sorry

end numberOfColoringWays_l302_302208


namespace vertex_x_coordinate_of_quadratic_l302_302495

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - 8 * x + 15

-- Define the x-coordinate of the vertex
def vertex_x_coordinate (f : ℝ → ℝ) : ℝ := 4

-- The theorem to prove
theorem vertex_x_coordinate_of_quadratic :
  vertex_x_coordinate quadratic_function = 4 :=
by
  -- Proof skipped
  sorry

end vertex_x_coordinate_of_quadratic_l302_302495


namespace men_became_absent_l302_302772

theorem men_became_absent (num_men absent : ℤ) 
  (num_men_eq : num_men = 180) 
  (days_planned : ℤ) (days_planned_eq : days_planned = 55)
  (days_taken : ℤ) (days_taken_eq : days_taken = 60)
  (work_planned : ℤ) (work_planned_eq : work_planned = num_men * days_planned)
  (work_taken : ℤ) (work_taken_eq : work_taken = (num_men - absent) * days_taken)
  (work_eq : work_planned = work_taken) :
  absent = 15 :=
  by sorry

end men_became_absent_l302_302772


namespace imaginary_unit_power_l302_302184

theorem imaginary_unit_power (i : ℂ) (n : ℕ) (h_i : i^2 = -1) : ∃ (n : ℕ), i^n = -1 :=
by
  use 6
  have h1 : i^4 = 1 := by sorry  -- Need to show i^4 = 1
  have h2 : i^6 = -1 := by sorry  -- Use i^4 and additional steps to show i^6 = -1
  exact h2

end imaginary_unit_power_l302_302184


namespace sqrt_simplify_l302_302034

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l302_302034


namespace snake_alligator_consumption_l302_302316

theorem snake_alligator_consumption :
  (616 / 7) = 88 :=
by
  sorry

end snake_alligator_consumption_l302_302316


namespace negation_exists_l302_302313

theorem negation_exists (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ ∀ x : ℝ, x^2 - a * x + 1 ≥ 0 :=
sorry

end negation_exists_l302_302313


namespace relation_of_M_and_N_l302_302823

-- Define the functions for M and N
def M (x : ℝ) : ℝ := (x - 3) * (x - 4)
def N (x : ℝ) : ℝ := (x - 1) * (x - 6)

-- Formulate the theorem to prove M < N for all x
theorem relation_of_M_and_N (x : ℝ) : M x < N x := sorry

end relation_of_M_and_N_l302_302823


namespace no_positive_a_b_for_all_primes_l302_302849

theorem no_positive_a_b_for_all_primes :
  ∀ (a b : ℕ), 0 < a → 0 < b → ∃ (p q : ℕ), p > 1000 ∧ q > 1000 ∧ p ≠ q ∧ Prime p ∧ Prime q ∧ ¬Prime (a * p + b * q) :=
by
  sorry

end no_positive_a_b_for_all_primes_l302_302849


namespace simplify_expression_l302_302135

theorem simplify_expression (x y : ℤ) (h1 : x = -1) (h2 : y = 2) :
  (3 * x^2 * y - 2 * x * y^2) - (x * y^2 - 2 * x^2 * y) - 2 * (-3 * x^2 * y - x * y^2) = 26 :=
by
  rw [h1, h2]
  sorry

end simplify_expression_l302_302135


namespace smallest_x_fraction_floor_l302_302207

theorem smallest_x_fraction_floor (x : ℝ) (h : ⌊x⌋ / x = 7 / 8) : x = 48 / 7 :=
sorry

end smallest_x_fraction_floor_l302_302207


namespace boys_from_other_communities_l302_302165

theorem boys_from_other_communities :
  ∀ (total_boys : ℕ) (percentage_muslims percentage_hindus percentage_sikhs : ℕ),
  total_boys = 400 →
  percentage_muslims = 44 →
  percentage_hindus = 28 →
  percentage_sikhs = 10 →
  (total_boys * (100 - (percentage_muslims + percentage_hindus + percentage_sikhs)) / 100) = 72 := 
by
  intros total_boys percentage_muslims percentage_hindus percentage_sikhs h1 h2 h3 h4
  sorry

end boys_from_other_communities_l302_302165


namespace evaluate_expression_l302_302881

theorem evaluate_expression :
  2 * 7^(-1/3 : ℝ) + (1/2 : ℝ) * Real.log (1/64) / Real.log 2 = -3 := 
  sorry

end evaluate_expression_l302_302881


namespace fgh_deriv_at_0_l302_302867

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def h : ℝ → ℝ := sorry

-- Function Values at x = 0
axiom f_zero : f 0 = 1
axiom g_zero : g 0 = 2
axiom h_zero : h 0 = 3

-- Derivatives of the pairwise products at x = 0
axiom d_gh_zero : (deriv (λ x => g x * h x)) 0 = 4
axiom d_hf_zero : (deriv (λ x => h x * f x)) 0 = 5
axiom d_fg_zero : (deriv (λ x => f x * g x)) 0 = 6

-- We need to prove that the derivative of the product of f, g, h at x = 0 is 16
theorem fgh_deriv_at_0 : (deriv (λ x => f x * g x * h x)) 0 = 16 := by
  sorry

end fgh_deriv_at_0_l302_302867


namespace ellie_sam_in_photo_probability_l302_302906

-- Definitions of the conditions
def lap_time_ellie := 120 -- seconds
def lap_time_sam := 75 -- seconds
def start_time := 10 * 60 -- 10 minutes in seconds
def photo_duration := 60 -- 1 minute in seconds
def photo_section := 1 / 3 -- fraction of the track captured in the photo

-- The probability that both Ellie and Sam are in the photo section between 10 to 11 minutes
theorem ellie_sam_in_photo_probability :
  let ellie_time := start_time;
  let sam_time := start_time;
  let ellie_range := (ellie_time - (photo_section * lap_time_ellie / 2), ellie_time + (photo_section * lap_time_ellie / 2));
  let sam_range := (sam_time - (photo_section * lap_time_sam / 2), sam_time + (photo_section * lap_time_sam / 2));
  let overlap_start := max ellie_range.1 sam_range.1;
  let overlap_end := min ellie_range.2 sam_range.2;
  let overlap_duration := max 0 (overlap_end - overlap_start);
  let overlap_probability := overlap_duration / photo_duration;
  overlap_probability = 5 / 12 :=
by
  sorry

end ellie_sam_in_photo_probability_l302_302906


namespace dance_lessons_l302_302630

theorem dance_lessons (cost_per_lesson : ℕ) (free_lessons : ℕ) (amount_paid : ℕ) 
  (H1 : cost_per_lesson = 10) 
  (H2 : free_lessons = 2) 
  (H3 : amount_paid = 80) : 
  (amount_paid / cost_per_lesson + free_lessons = 10) :=
by
  sorry

end dance_lessons_l302_302630


namespace Ksyusha_time_to_school_l302_302273

/-- Ksyusha runs twice as fast as she walks (both speeds are constant).
On Tuesday, Ksyusha walked a distance twice the distance she ran, and she reached school in 30 minutes.
On Wednesday, Ksyusha ran twice the distance she walked.
Prove that it took her 24 minutes to get from home to school on Wednesday. -/
theorem Ksyusha_time_to_school
  (S v : ℝ)                  -- S for distance unit, v for walking speed
  (htuesday : 2 * (2 * S / v) + S / (2 * v) = 30) -- Condition on Tuesday's travel
  :
  (S / v + 2 * S / (2 * v) = 24)                 -- Claim about Wednesday's travel time
:=
  sorry

end Ksyusha_time_to_school_l302_302273


namespace simplify_sqrt_450_l302_302117

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l302_302117


namespace sqrt_450_eq_15_sqrt_2_l302_302058

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302058


namespace smallest_number_divisible_1_through_12_and_15_l302_302493

theorem smallest_number_divisible_1_through_12_and_15 :
  ∃ n, (∀ i, 1 ≤ i ∧ i ≤ 12 → i ∣ n) ∧ 15 ∣ n ∧ n = 27720 :=
by {
  sorry
}

end smallest_number_divisible_1_through_12_and_15_l302_302493


namespace tank_capacity_l302_302164

theorem tank_capacity (C : ℝ) (h₁ : 3/4 * C + 7 = 7/8 * C) : C = 56 :=
by
  sorry

end tank_capacity_l302_302164


namespace div_polynomials_l302_302522

variable (a b : ℝ)

theorem div_polynomials :
  10 * a^3 * b^2 / (-5 * a^2 * b) = -2 * a * b := 
by sorry

end div_polynomials_l302_302522


namespace uncle_kahn_total_cost_l302_302437

noncomputable def base_price : ℝ := 10
noncomputable def child_discount : ℝ := 0.3
noncomputable def senior_discount : ℝ := 0.1
noncomputable def handling_fee : ℝ := 5
noncomputable def discounted_senior_ticket_price : ℝ := 14
noncomputable def num_child_tickets : ℝ := 2
noncomputable def num_senior_tickets : ℝ := 2

theorem uncle_kahn_total_cost :
  let child_ticket_cost := (1 - child_discount) * base_price + handling_fee
  let senior_ticket_cost := discounted_senior_ticket_price
  num_child_tickets * child_ticket_cost + num_senior_tickets * senior_ticket_cost = 52 :=
by
  sorry

end uncle_kahn_total_cost_l302_302437


namespace domain_of_g_l302_302154

noncomputable def g (x : ℝ) : ℝ := Real.logb 3 (Real.logb 4 (Real.logb 5 (Real.logb 6 x)))

theorem domain_of_g : {x : ℝ | x > 6^625} = {x : ℝ | ∃ y : ℝ, y = g x } := sorry

end domain_of_g_l302_302154


namespace simplify_sqrt_450_l302_302006

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l302_302006


namespace inclination_angle_of_line_equation_l302_302140

noncomputable def inclination_angle_of_line (a b c : ℝ) (h : a + b ≠ 0) : ℝ :=
if h : b ≠ 0 then
  real.arctan (-a / b)
else if h' : a ≠ 0 then
  if a > 0 then real.pi / 2 else -real.pi / 2
else 0

theorem inclination_angle_of_line_equation :
  inclination_angle_of_line 1 1 1 1 ≠ 0 = 3 * real.pi / 4 :=
begin
  sorry
end

end inclination_angle_of_line_equation_l302_302140


namespace rectangle_A_plus_P_ne_162_l302_302893

theorem rectangle_A_plus_P_ne_162 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (A : ℕ) (P : ℕ) 
  (hA : A = a * b) (hP : P = 2 * a + 2 * b) : A + P ≠ 162 :=
by
  sorry

end rectangle_A_plus_P_ne_162_l302_302893


namespace Cheryl_total_distance_l302_302523

theorem Cheryl_total_distance :
  let speed := 2
  let duration := 3
  let distance_away := speed * duration
  let distance_home := distance_away
  let total_distance := distance_away + distance_home
  total_distance = 12 := by
  sorry

end Cheryl_total_distance_l302_302523


namespace simplify_sqrt_450_l302_302104

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l302_302104


namespace value_of_expression_l302_302399

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := by
  sorry

end value_of_expression_l302_302399


namespace Cheryl_total_distance_l302_302524

theorem Cheryl_total_distance :
  let speed := 2
  let duration := 3
  let distance_away := speed * duration
  let distance_home := distance_away
  let total_distance := distance_away + distance_home
  total_distance = 12 := by
  sorry

end Cheryl_total_distance_l302_302524


namespace units_digit_proof_l302_302918

def units_digit (n : ℤ) : ℤ := n % 10

theorem units_digit_proof :
  ∀ (a b c : ℤ),
  a = 8 →
  b = 18 →
  c = 1988 →
  units_digit (a * b * c - a^3) = 0 :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc]
  -- Proof will go here
  sorry

end units_digit_proof_l302_302918


namespace sum_reciprocal_geo_seq_l302_302957

theorem sum_reciprocal_geo_seq {a_5 a_6 a_7 a_8 : ℝ}
  (h_sum : a_5 + a_6 + a_7 + a_8 = 15 / 8)
  (h_prod : a_6 * a_7 = -9 / 8) :
  (1 / a_5) + (1 / a_6) + (1 / a_7) + (1 / a_8) = -5 / 3 := by
  sorry

end sum_reciprocal_geo_seq_l302_302957


namespace complement_P_inter_Q_l302_302817

def P : Set ℝ := {x | x^2 - 2 * x ≥ 0}
def Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}
def complement_P : Set ℝ := {x | 0 < x ∧ x < 2}

theorem complement_P_inter_Q : (complement_P ∩ Q) = {x | 1 < x ∧ x < 2} := by
  sorry

end complement_P_inter_Q_l302_302817


namespace original_price_l302_302651

variables (p q d : ℝ)


theorem original_price (x : ℝ) (h : x * (1 + p / 100) * (1 - q / 100) = d) :
  x = 100 * d / (100 + p - q - p * q / 100) := 
sorry

end original_price_l302_302651


namespace solve_for_x_l302_302597

theorem solve_for_x (x : ℝ) (h : (3 * x - 15) / 4 = (x + 9) / 5) : x = 10 :=
by {
  sorry
}

end solve_for_x_l302_302597


namespace simplify_expression_l302_302285

theorem simplify_expression : 
  (6^8 - 4^7) * (2^3 - (-2)^3) ^ 10 = 1663232 * 16 ^ 10 := 
by {
  sorry
}

end simplify_expression_l302_302285


namespace value_of_ac_over_bd_l302_302409

theorem value_of_ac_over_bd (a b c d : ℝ) 
  (h1 : a = 4 * b)
  (h2 : b = 3 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by
  sorry

end value_of_ac_over_bd_l302_302409


namespace ratio_D_to_C_l302_302788

-- Defining the terms and conditions
def speed_ratio (C Ch D : ℝ) : Prop :=
  (C = 2 * Ch) ∧
  (D / Ch = 6)

-- The theorem statement
theorem ratio_D_to_C (C Ch D : ℝ) (h : speed_ratio C Ch D) : (D / C = 3) :=
by
  sorry

end ratio_D_to_C_l302_302788


namespace smallest_n_l302_302354

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def condition_for_n (n : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ n → ∀ x : ℕ, x ∈ M k → ∃ y : ℕ, y ∈ M k ∧ y ≠ x ∧ is_perfect_square (x + y)
  where M (k : ℕ) := { m : ℕ | m > 0 ∧ m ≤ k }

theorem smallest_n : ∃ n : ℕ, (condition_for_n n) ∧ (∀ m < n, ¬ condition_for_n m) :=
  sorry

end smallest_n_l302_302354


namespace find_cos_C_l302_302695

noncomputable def cos_C_eq (A B C a b c : ℝ) (h1 : 8 * b = 5 * c) (h2 : C = 2 * B) : Prop :=
  Real.cos C = 7 / 25

theorem find_cos_C (A B C a b c : ℝ) (h1 : 8 * b = 5 * c) (h2 : C = 2 * B) :
  cos_C_eq A B C a b c h1 h2 :=
sorry

end find_cos_C_l302_302695


namespace solve_for_q_l302_302565

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : q = -25 / 11 :=
by
  sorry

end solve_for_q_l302_302565


namespace simplify_sqrt_450_l302_302090

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l302_302090


namespace unique_triple_solution_l302_302660

theorem unique_triple_solution (x y z : ℝ) :
  x = y^3 + y - 8 ∧ y = z^3 + z - 8 ∧ z = x^3 + x - 8 → (x, y, z) = (2, 2, 2) :=
by
  sorry

end unique_triple_solution_l302_302660


namespace find_value_of_fraction_l302_302418

theorem find_value_of_fraction (a b c d: ℕ) (h1: a = 4 * b) (h2: b = 3 * c) (h3: c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end find_value_of_fraction_l302_302418


namespace stone_statue_cost_l302_302480

theorem stone_statue_cost :
  ∃ S : Real, 
    let total_earnings := 10 * S + 20 * 5
    let earnings_after_taxes := 0.9 * total_earnings
    earnings_after_taxes = 270 ∧ S = 20 :=
sorry

end stone_statue_cost_l302_302480


namespace fraction_to_decimal_l302_302201

theorem fraction_to_decimal :
  (51 / 160 : ℝ) = 0.31875 := 
by
  sorry

end fraction_to_decimal_l302_302201


namespace problem_l302_302775

def g (x : ℝ) (d e f : ℝ) := d * x^2 + e * x + f

theorem problem (d e f : ℝ) (h_vertex : ∀ x : ℝ, g d e f (x + 2) = -1 * (x + 2)^2 + 5) :
  d + e + 3 * f = 14 := 
sorry

end problem_l302_302775


namespace plane_through_points_l302_302351

-- Define the vectors as tuples of three integers
def point := (ℤ × ℤ × ℤ)

-- The given points
def p : point := (2, -1, 3)
def q : point := (4, -1, 5)
def r : point := (5, -3, 4)

-- A function to find the equation of the plane given three points
def plane_equation (p q r : point) : ℤ × ℤ × ℤ × ℤ :=
  let (px, py, pz) := p
  let (qx, qy, qz) := q
  let (rx, ry, rz) := r
  let a := (qy - py) * (rz - pz) - (qy - py) * (rz - pz)
  let b := (qx - px) * (rz - pz) - (qx - px) * (rz - pz)
  let c := (qx - px) * (ry - py) - (qx - px) * (ry - py)
  let d := -(a * px + b * py + c * pz)
  (a, b, c, d)

-- The proof statement
theorem plane_through_points : plane_equation (2, -1, 3) (4, -1, 5) (5, -3, 4) = (1, 2, -2, 6) :=
  by sorry

end plane_through_points_l302_302351


namespace find_roots_combination_l302_302673

theorem find_roots_combination 
  (α β : ℝ)
  (hα : α^2 - 3 * α + 1 = 0)
  (hβ : β^2 - 3 * β + 1 = 0) :
  7 * α^3 + 10 * β^4 = 697 := by
  sorry

end find_roots_combination_l302_302673


namespace exponent_equivalence_l302_302520

open Real

theorem exponent_equivalence (a : ℝ) (h : a > 0) : 
  (a^2 / (sqrt a * a^(2/3))) = a^(5/6) :=
  sorry

end exponent_equivalence_l302_302520


namespace find_rate_percent_l302_302639

theorem find_rate_percent
  (P : ℝ) (SI : ℝ) (T : ℝ) (R : ℝ) 
  (hP : P = 1600)
  (hSI : SI = 200)
  (hT : T = 4)
  (hSI_eq : SI = (P * R * T) / 100) :
  R = 3.125 :=
by {
  sorry
}

end find_rate_percent_l302_302639


namespace sum_numbers_l302_302635

theorem sum_numbers : 3456 + 4563 + 5634 + 6345 = 19998 := by
  sorry

end sum_numbers_l302_302635


namespace adjacent_sum_constant_l302_302248

theorem adjacent_sum_constant (x y : ℤ) (k : ℤ) (h1 : 2 + x = k) (h2 : x + y = k) (h3 : y + 5 = k) : x - y = 3 := 
by 
  sorry

end adjacent_sum_constant_l302_302248


namespace concrete_for_supporting_pillars_l302_302960

-- Define the given conditions
def roadway_deck_concrete : ℕ := 1600
def one_anchor_concrete : ℕ := 700
def total_bridge_concrete : ℕ := 4800

-- State the theorem
theorem concrete_for_supporting_pillars :
  let total_anchors_concrete := 2 * one_anchor_concrete in
  let total_deck_and_anchors_concrete := roadway_deck_concrete + total_anchors_concrete in
  total_bridge_concrete - total_deck_and_anchors_concrete = 1800 :=
by
  sorry

end concrete_for_supporting_pillars_l302_302960


namespace find_quadratic_minimum_value_l302_302683

noncomputable def quadraticMinimumPoint (a b c : ℝ) : ℝ :=
  -b / (2 * a)

theorem find_quadratic_minimum_value :
  quadraticMinimumPoint 3 6 9 = -1 :=
by
  sorry

end find_quadratic_minimum_value_l302_302683


namespace fraction_value_l302_302425

theorem fraction_value (a b c d : ℕ)
  (h₁ : a = 4 * b)
  (h₂ : b = 3 * c)
  (h₃ : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end fraction_value_l302_302425


namespace shaded_area_correct_l302_302596

-- Define points as vectors in the 2D plane.
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define points K, L, M, J based on the given coordinates.
def K : Point := {x := 0, y := 0}
def L : Point := {x := 5, y := 0}
def M : Point := {x := 5, y := 6}
def J : Point := {x := 0, y := 6}

-- Define intersection point N based on the equations of lines.
def N : Point := {x := 2.5, y := 3}

-- Define the function to calculate area of a trapezoid.
def trapezoid_area (b1 b2 h : ℝ) : ℝ :=
  0.5 * (b1 + b2) * h

-- Define the function to calculate area of a triangle.
def triangle_area (b h : ℝ) : ℝ :=
  0.5 * b * h

-- Compute total shaded area according to the problem statement.
def shaded_area (K L M J N : Point) : ℝ :=
  trapezoid_area 5 2.5 3 + triangle_area 2.5 1

theorem shaded_area_correct : shaded_area K L M J N = 12.5 := by
  sorry

end shaded_area_correct_l302_302596


namespace ordering_of_abc_l302_302211

def a : ℝ := 2^(4/3)
def b : ℝ := 4^(2/5)
def c : ℝ := 25^(1/3)

theorem ordering_of_abc : b < a ∧ a < c := by
  sorry

end ordering_of_abc_l302_302211


namespace solve_for_q_l302_302566

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : q = -25 / 11 :=
by
  sorry

end solve_for_q_l302_302566


namespace maximum_value_is_16_l302_302969

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
(x^2 - 2 * x * y + 2 * y^2) * (x^2 - 2 * x * z + 2 * z^2) * (y^2 - 2 * y * z + 2 * z^2)

theorem maximum_value_is_16 (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) :
  maximum_value x y z ≤ 16 :=
by
  sorry

end maximum_value_is_16_l302_302969


namespace simplify_sqrt_450_l302_302023

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l302_302023


namespace maximize_abs_sum_solution_problem_l302_302982

theorem maximize_abs_sum_solution :
ℤ → ℤ → Ennreal := sorry

theorem problem :
  (∃ (x y : ℤ), 6 * x^2 + 5 * x * y + y^2 = 6 * x + 2 * y + 7 ∧ 
  x = -8 ∧ y = 25 ∧ (maximize_abs_sum_solution x y = 33)) := sorry

end maximize_abs_sum_solution_problem_l302_302982


namespace actual_diameter_layer_3_is_20_micrometers_l302_302508

noncomputable def magnified_diameter_to_actual (magnified_diameter_cm : ℕ) (magnification_factor : ℕ) : ℕ :=
  (magnified_diameter_cm * 10000) / magnification_factor

def layer_3_magnified_diameter_cm : ℕ := 3
def layer_3_magnification_factor : ℕ := 1500

theorem actual_diameter_layer_3_is_20_micrometers :
  magnified_diameter_to_actual layer_3_magnified_diameter_cm layer_3_magnification_factor = 20 :=
by
  sorry

end actual_diameter_layer_3_is_20_micrometers_l302_302508


namespace find_f_2010_l302_302222

def f (x : ℝ) : ℝ := sorry

theorem find_f_2010 (h₁ : ∀ x, f (x + 1) = - f x) (h₂ : f 1 = 4) : f 2010 = -4 :=
by 
  sorry

end find_f_2010_l302_302222


namespace geometric_progression_theorem_l302_302593

theorem geometric_progression_theorem 
  (a b c d : ℝ) (q : ℝ) 
  (h1 : b = a * q) 
  (h2 : c = a * q^2) 
  (h3 : d = a * q^3) 
  : (a - d)^2 = (a - c)^2 + (b - c)^2 + (b - d)^2 := 
by sorry

end geometric_progression_theorem_l302_302593


namespace simplify_sqrt_450_l302_302061

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l302_302061


namespace y_at_x_equals_2sqrt3_l302_302434

theorem y_at_x_equals_2sqrt3 (k : ℝ) (y : ℝ → ℝ)
  (h1 : ∀ x : ℝ, y x = k * x^(1/3))
  (h2 : y 64 = 4 * Real.sqrt 3) :
  y 8 = 2 * Real.sqrt 3 :=
sorry

end y_at_x_equals_2sqrt3_l302_302434


namespace tangent_line_eq_max_min_values_l302_302376

noncomputable def f (x : ℝ) : ℝ := (1 / (3:ℝ)) * x^3 - 4 * x + 4

theorem tangent_line_eq (x y : ℝ) : 
    y = f 1 → 
    y = -3 * (x - 1) + f 1 → 
    3 * x + y - 10 / 3 = 0 := 
sorry

theorem max_min_values (x : ℝ) (h : 0 ≤ x ∧ x ≤ 3) : 
    (∀ x, (0 ≤ x ∧ x ≤ 3) → f x ≤ 4) ∧ 
    (∀ x, (0 ≤ x ∧ x ≤ 3) → f x ≥ -4 / 3) := 
sorry

end tangent_line_eq_max_min_values_l302_302376


namespace equivalent_single_discount_l302_302599

theorem equivalent_single_discount (p : ℝ) : 
  let discount1 := 0.15
  let discount2 := 0.25
  let price_after_first_discount := (1 - discount1) * p
  let price_after_second_discount := (1 - discount2) * price_after_first_discount
  let equivalent_single_discount := 1 - price_after_second_discount / p
  equivalent_single_discount = 0.3625 :=
by
  sorry

end equivalent_single_discount_l302_302599


namespace max_possible_acute_angled_triangles_l302_302843
-- Define the sets of points on lines a and b
def maxAcuteAngledTriangles (n : Nat) : Nat :=
  let sum1 := (n * (n - 1) / 2)  -- Sum of first (n-1) natural numbers
  let sum2 := (sum1 * 50) - (n * (n - 1) * (2 * n - 1) / 6) -- Applying the given formula
  (2 * sum2)  -- Multiply by 2 for both colors of alternating points

-- Define the main theorem
theorem max_possible_acute_angled_triangles : maxAcuteAngledTriangles 50 = 41650 := by
  sorry

end max_possible_acute_angled_triangles_l302_302843


namespace find_a_b_c_sum_l302_302512

-- Define the necessary conditions and constants
def radius : ℝ := 10  -- tower radius in feet
def rope_length : ℝ := 30  -- length of the rope in feet
def unicorn_height : ℝ := 6  -- height of the unicorn from ground in feet
def rope_end_distance : ℝ := 6  -- distance from the unicorn to the nearest point on the tower

def a : ℕ := 30
def b : ℕ := 900
def c : ℕ := 10  -- assuming c is not necessarily prime for the purpose of this exercise

-- The theorem we want to prove
theorem find_a_b_c_sum : a + b + c = 940 :=
by
  sorry

end find_a_b_c_sum_l302_302512


namespace first_month_sale_l302_302889

def sale2 : ℕ := 5768
def sale3 : ℕ := 5922
def sale4 : ℕ := 5678
def sale5 : ℕ := 6029
def sale6 : ℕ := 4937
def average_sale : ℕ := 5600

theorem first_month_sale :
  let total_sales := average_sale * 6
  let known_sales := sale2 + sale3 + sale4 + sale5 + sale6
  let sale1 := total_sales - known_sales
  sale1 = 5266 :=
by
  sorry

end first_month_sale_l302_302889


namespace prime_pairs_perfect_square_l302_302657

theorem prime_pairs_perfect_square (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  ∃ k : ℕ, p^(q-1) + q^(p-1) = k^2 ↔ (p = 2 ∧ q = 2) :=
by
  sorry

end prime_pairs_perfect_square_l302_302657


namespace arithmetic_sequence_value_of_n_l302_302143

theorem arithmetic_sequence_value_of_n :
  ∀ (a n d : ℕ), a = 1 → d = 3 → (a + (n - 1) * d = 2005) → n = 669 :=
by
  intros a n d h_a1 h_d ha_n
  sorry

end arithmetic_sequence_value_of_n_l302_302143


namespace remainder_of_349_divided_by_17_l302_302159

theorem remainder_of_349_divided_by_17 : 
  (349 % 17 = 9) := 
by
  sorry

end remainder_of_349_divided_by_17_l302_302159


namespace total_area_of_region_l302_302894

variable (a b c d : ℝ)
variable (ha : a > 0) (hb : b > 0) (hd : d > 0)

theorem total_area_of_region : (a + b) * d + (1 / 2) * Real.pi * c ^ 2 = (a + b) * d + (1 / 2) * Real.pi * c ^ 2 := by
  sorry

end total_area_of_region_l302_302894


namespace average_rate_of_change_l302_302661

noncomputable def f (x : ℝ) := 2 * x + 1

theorem average_rate_of_change :
  (f 2 - f 1) / (2 - 1) = 2 :=
by
  sorry

end average_rate_of_change_l302_302661


namespace sandy_ordered_three_cappuccinos_l302_302336

-- Definitions and conditions
def cost_cappuccino : ℝ := 2
def cost_iced_tea : ℝ := 3
def cost_cafe_latte : ℝ := 1.5
def cost_espresso : ℝ := 1
def num_iced_teas : ℕ := 2
def num_cafe_lattes : ℕ := 2
def num_espressos : ℕ := 2
def change_received : ℝ := 3
def amount_paid : ℝ := 20

-- Calculation of costs
def total_cost_iced_teas : ℝ := num_iced_teas * cost_iced_tea
def total_cost_cafe_lattes : ℝ := num_cafe_lattes * cost_cafe_latte
def total_cost_espressos : ℝ := num_espressos * cost_espresso
def total_cost_other_drinks : ℝ := total_cost_iced_teas + total_cost_cafe_lattes + total_cost_espressos
def total_spent : ℝ := amount_paid - change_received
def cost_cappuccinos := total_spent - total_cost_other_drinks

-- Proof statement
theorem sandy_ordered_three_cappuccinos (num_cappuccinos : ℕ) : cost_cappuccinos = num_cappuccinos * cost_cappuccino → num_cappuccinos = 3 :=
by sorry

end sandy_ordered_three_cappuccinos_l302_302336


namespace expand_product_l302_302348

theorem expand_product (x : ℝ) : (3 * x + 4) * (2 * x + 6) = 6 * x^2 + 26 * x + 24 := 
by 
  sorry

end expand_product_l302_302348


namespace simplify_sqrt_450_l302_302066

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l302_302066


namespace base4_addition_l302_302990

def base4_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | n + 1 => (n % 10) + 4 * base4_to_base10 (n / 10)

def base10_to_base4 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | n + 1 => (n % 4) + 10 * base10_to_base4 (n / 4)

theorem base4_addition :
  base10_to_base4 (base4_to_base10 234 + base4_to_base10 73) = 1203 := by
  sorry

end base4_addition_l302_302990


namespace fourth_competitor_jump_distance_l302_302142

theorem fourth_competitor_jump_distance (a b c d : ℕ) 
    (h1 : a = 22) 
    (h2 : b = a + 1)
    (h3 : c = b - 2)
    (h4 : d = c + 3):
    d = 24 :=
by
  rw [h1, h2, h3, h4]
  sorry

end fourth_competitor_jump_distance_l302_302142


namespace sum_x_coords_of_A_l302_302755

open Real

noncomputable def area_triangle (A B C : (ℝ × ℝ)) : ℝ := 
  abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2

theorem sum_x_coords_of_A :
  ∀ A : (ℝ × ℝ),
  (area_triangle B C A = 2010) →
  (area_triangle A D F = 8020) →
  B = (0, 0) →
  C = (226, 0) →
  D = (680, 380) →
  F = (700, 400) →
  (sum_of_possible_x_coords_of_A A = -635.6)
sorry

end sum_x_coords_of_A_l302_302755


namespace count_odd_3_digit_integers_l302_302691

open Nat

/-- Prove that there are no odd positive 3-digit integers which are divisible by 5
    and do not contain the digit 5. -/
theorem count_odd_3_digit_integers : 
  ∃ n : ℕ, (n = 0 ↔ ∀ x, 100 ≤ x ∧ x < 1000 ∧ (x % 2 = 1) ∧ (x % 5 = 0) ∧ ¬('5' ∈ (show String, from toDigits 10 x).data)) :=
by 
  sorry

end count_odd_3_digit_integers_l302_302691


namespace expected_value_ξ_l302_302243

-- Define conditions
def n : ℕ := 5
def p : ℚ := 1 / 3

-- Define random variable ξ as binomial distribution's expected value
def ξ_expected_value : ℚ := n * p

-- The statement to be proved
theorem expected_value_ξ : ξ_expected_value = 5 / 3 :=
by
  -- proof would go here
  sorry

end expected_value_ξ_l302_302243


namespace sqrt_450_eq_15_sqrt_2_l302_302074

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l302_302074


namespace concrete_pillars_correct_l302_302965

-- Definitions based on conditions
def concrete_for_roadway := 1600
def concrete_for_one_anchor := 700
def total_concrete_for_bridge := 4800

-- Total concrete for both anchors
def concrete_for_both_anchors := 2 * concrete_for_one_anchor

-- Total concrete needed for the roadway and anchors
def concrete_for_roadway_and_anchors := concrete_for_roadway + concrete_for_both_anchors

-- Concrete needed for the supporting pillars
def concrete_for_pillars := total_concrete_for_bridge - concrete_for_roadway_and_anchors

-- Proof problem statement, verify that the concrete for the supporting pillars is 1800 tons
theorem concrete_pillars_correct : concrete_for_pillars = 1800 := by
  sorry

end concrete_pillars_correct_l302_302965


namespace triangle_area_difference_l302_302754

theorem triangle_area_difference 
  (b h : ℝ)
  (hb : 0 < b)
  (hh : 0 < h)
  (A_base : ℝ) (A_height : ℝ)
  (hA_base: A_base = 1.20 * b)
  (hA_height: A_height = 0.80 * h)
  (A_area: ℝ) (B_area: ℝ)
  (hA_area: A_area = 0.5 * A_base * A_height)
  (hB_area: B_area = 0.5 * b * h) :
  (B_area - A_area) / B_area = 0.04 := 
by sorry

end triangle_area_difference_l302_302754


namespace john_took_more_chickens_l302_302604

theorem john_took_more_chickens (john_chickens mary_chickens ray_chickens : ℕ)
  (h_ray : ray_chickens = 10)
  (h_mary : mary_chickens = ray_chickens + 6)
  (h_john : john_chickens = mary_chickens + 5) :
  john_chickens - ray_chickens = 11 := by
  sorry

end john_took_more_chickens_l302_302604


namespace verify_trig_identity_l302_302312

noncomputable def trig_identity_eqn : Prop :=
  2 * Real.sqrt (1 - Real.sin 8) + Real.sqrt (2 + 2 * Real.cos 8) = -2 * Real.sin 4

theorem verify_trig_identity : trig_identity_eqn := by
  sorry

end verify_trig_identity_l302_302312


namespace area_of_square_field_l302_302879

-- Define side length
def side_length : ℕ := 20

-- Theorem statement about the area of the square field
theorem area_of_square_field : (side_length * side_length) = 400 := by
  sorry

end area_of_square_field_l302_302879


namespace mean_score_is_76_l302_302163

noncomputable def mean_stddev_problem := 
  ∃ (M SD : ℝ), (M - 2 * SD = 60) ∧ (M + 3 * SD = 100) ∧ (M = 76)

theorem mean_score_is_76 : mean_stddev_problem :=
sorry

end mean_score_is_76_l302_302163


namespace max_S_value_l302_302449

noncomputable def maximize_S (a b c : ℝ) : ℝ :=
  (a^2 - a * b + b^2) * (b^2 - b * c + c^2) * (c^2 - c * a + a^2)

theorem max_S_value :
  ∀ (a b c : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = 3 →
  maximize_S a b c ≤ 12 :=
by sorry

end max_S_value_l302_302449


namespace simplify_expression_l302_302136

theorem simplify_expression (x y : ℤ) (h1 : x = -1) (h2 : y = 2) :
  (3 * x^2 * y - 2 * x * y^2) - (x * y^2 - 2 * x^2 * y) - 2 * (-3 * x^2 * y - x * y^2) = 26 :=
by
  rw [h1, h2]
  sorry

end simplify_expression_l302_302136


namespace trigonometric_identity_l302_302802

theorem trigonometric_identity (α : ℝ) (h : Real.tan (Real.pi + α) = 2) :
  4 * Real.sin α * Real.cos α + 3 * (Real.cos α) ^ 2 = 11 / 5 :=
sorry

end trigonometric_identity_l302_302802


namespace find_perpendicular_vector_l302_302818

def vector_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

def vector_magnitude_equal (v1 v2 : ℝ × ℝ) : Prop :=
  (v1.1 ^ 2 + v1.2 ^ 2) = (v2.1 ^ 2 + v2.2 ^ 2)

theorem find_perpendicular_vector (a b : ℝ) :
  ∃ n : ℝ × ℝ, vector_perpendicular (a, b) n ∧ vector_magnitude_equal (a, b) n ∧ n = (b, -a) :=
by
  sorry

end find_perpendicular_vector_l302_302818


namespace sin_product_identity_l302_302284

theorem sin_product_identity (φ : ℝ) (n : ℕ) (h : n > 0) :
  (∏ k in Finset.range n, Real.sin (φ + (k * Real.pi) / n)) = (Real.sin (n * φ)) / (2 ^ (n - 1)) := 
sorry

end sin_product_identity_l302_302284


namespace budget_allocation_l302_302886

theorem budget_allocation 
  (total_degrees : ℝ := 360)
  (total_budget : ℝ := 100)
  (degrees_basic_astrophysics : ℝ := 43.2)
  (percent_microphotonics : ℝ := 12)
  (percent_home_electronics : ℝ := 24)
  (percent_food_additives : ℝ := 15)
  (percent_industrial_lubricants : ℝ := 8) :
  ∃ percent_genetically_modified_microorganisms : ℝ,
  percent_genetically_modified_microorganisms = 29 :=
sorry

end budget_allocation_l302_302886


namespace unique_two_digit_integer_s_l302_302487

-- We define s to satisfy the two given conditions.
theorem unique_two_digit_integer_s (s : ℕ) (h1 : 13 * s % 100 = 52) (h2 : 1 ≤ s) (h3 : s ≤ 99) : s = 4 :=
sorry

end unique_two_digit_integer_s_l302_302487


namespace jessica_journey_total_distance_l302_302959

theorem jessica_journey_total_distance
  (y : ℝ)
  (h1 : y = (y / 4) + 25 + (y / 4)) :
  y = 50 :=
by
  sorry

end jessica_journey_total_distance_l302_302959


namespace minimum_value_l302_302546

open Real

theorem minimum_value (x y : ℝ) (hx : 1 < x) (hy : 1 < y) (h : x * y - 2 * x - y + 1 = 0) :
  ∃ z : ℝ, (z = (3 / 2) * x^2 + y^2) ∧ z = 15 :=
by
  sorry

end minimum_value_l302_302546


namespace white_paint_amount_l302_302478

theorem white_paint_amount (total_paint green_paint brown_paint : ℕ) 
  (h_total : total_paint = 69)
  (h_green : green_paint = 15)
  (h_brown : brown_paint = 34) :
  total_paint - (green_paint + brown_paint) = 20 := by
  sorry

end white_paint_amount_l302_302478


namespace value_of_expression_l302_302404

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := by
  sorry

end value_of_expression_l302_302404


namespace total_biking_distance_l302_302146

-- Define the problem conditions 
def shelves := 4
def books_per_shelf := 400
def one_way_distance := shelves * books_per_shelf

-- Prove that the total distance for a round trip is 3200 miles
theorem total_biking_distance : 2 * one_way_distance = 3200 :=
by sorry

end total_biking_distance_l302_302146


namespace triangle_inequality_l302_302833

theorem triangle_inequality (ABC: Triangle) (M : Point) (a b c : ℝ)
  (h1 : a = BC) (h2 : b = CA) (h3 : c = AB) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ 3 / (MA^2 + MB^2 + MC^2) := 
sorry

end triangle_inequality_l302_302833


namespace value_of_expression_l302_302400

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := by
  sorry

end value_of_expression_l302_302400


namespace sqrt_450_simplified_l302_302050

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l302_302050


namespace marcy_minimum_avg_score_l302_302291

variables (s1 s2 s3 : ℝ)
variable (qualified_avg : ℝ := 90)
variable (required_total : ℝ := 5 * qualified_avg)
variable (first_three_total : ℝ := s1 + s2 + s3)
variable (needed_points : ℝ := required_total - first_three_total)
variable (required_avg : ℝ := needed_points / 2)

/-- The admission criteria for a mathematics contest require a contestant to 
    achieve an average score of at least 90% over five rounds to qualify for the final round.
    Marcy scores 87%, 92%, and 85% in the first three rounds. 
    Prove that Marcy must average at least 93% in the next two rounds to qualify for the final. --/
theorem marcy_minimum_avg_score 
    (h1 : s1 = 87) (h2 : s2 = 92) (h3 : s3 = 85)
    : required_avg ≥ 93 :=
sorry

end marcy_minimum_avg_score_l302_302291


namespace first_pump_time_l302_302458

-- Definitions for the conditions provided
def newer_model_rate := 1 / 6
def combined_rate := 1 / 3.6
def time_for_first_pump : ℝ := 9

-- The theorem to be proven
theorem first_pump_time (T : ℝ) (h1 : 1 / 6 + 1 / T = 1 / 3.6) : T = 9 :=
sorry

end first_pump_time_l302_302458


namespace abs_sqrt2_sub_2_l302_302286

theorem abs_sqrt2_sub_2 (h : 1 < Real.sqrt 2 ∧ Real.sqrt 2 < 2) : |Real.sqrt 2 - 2| = 2 - Real.sqrt 2 :=
by
  sorry

end abs_sqrt2_sub_2_l302_302286


namespace simplify_sqrt_450_l302_302096

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l302_302096


namespace positive_reals_condition_l302_302911

theorem positive_reals_condition (a : ℝ) (h_pos : 0 < a) : a < 2 :=
by
  -- Problem conditions:
  -- There exists a positive integer n and n pairwise disjoint infinite sets A_i
  -- such that A_1 ∪ ... ∪ A_n = ℕ* and for any two numbers b > c in each A_i,
  -- b - c ≥ a^i.

  sorry

end positive_reals_condition_l302_302911


namespace arithmetic_sequence_formula_l302_302671

theorem arithmetic_sequence_formula (x : ℤ) (a : ℕ → ℤ) 
  (h1 : a 1 = x - 1) (h2 : a 2 = x + 1) (h3 : a 3 = 2 * x + 3) :
  ∃ c d : ℤ, (∀ n : ℕ, a n = c + d * (n - 1)) ∧ ∀ n : ℕ, a n = 2 * n - 3 :=
by {
  sorry
}

end arithmetic_sequence_formula_l302_302671


namespace total_students_l302_302620

theorem total_students (students_per_classroom : ℕ) (num_classrooms : ℕ) (h1 : students_per_classroom = 30) (h2 : num_classrooms = 13) : students_per_classroom * num_classrooms = 390 :=
by
  -- Begin the proof
  sorry

end total_students_l302_302620


namespace simplify_sqrt_450_l302_302120

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l302_302120


namespace derivative_at_0_l302_302568

def f (x : ℝ) : ℝ := x + x^2

theorem derivative_at_0 : deriv f 0 = 1 := by
  -- Proof goes here
  sorry

end derivative_at_0_l302_302568


namespace catch_up_distance_l302_302875

def v_a : ℝ := 10 -- A's speed in kmph
def v_b : ℝ := 20 -- B's speed in kmph
def t : ℝ := 10 -- Time in hours when B starts after A

theorem catch_up_distance : v_b * t + v_a * t = 200 :=
by sorry

end catch_up_distance_l302_302875


namespace problem_l302_302394

theorem problem (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by sorry

end problem_l302_302394


namespace seashell_count_l302_302631

def initialSeashells : Nat := 5
def givenSeashells : Nat := 2
def remainingSeashells : Nat := initialSeashells - givenSeashells

theorem seashell_count : remainingSeashells = 3 := by
  sorry

end seashell_count_l302_302631


namespace imaginary_part_of_complex_number_l302_302989

def imaginary_unit (i : ℂ) : Prop := i * i = -1

def complex_number (z : ℂ) (i : ℂ) : Prop := z = i * (1 - 3 * i)

theorem imaginary_part_of_complex_number (i z : ℂ) (h1 : imaginary_unit i) (h2 : complex_number z i) : z.im = 1 :=
by
  sorry

end imaginary_part_of_complex_number_l302_302989


namespace alice_forest_walks_l302_302347

theorem alice_forest_walks
  (morning_distance : ℕ)
  (total_distance : ℕ)
  (days_per_week : ℕ)
  (forest_distance : ℕ) :
  morning_distance = 10 →
  total_distance = 110 →
  days_per_week = 5 →
  (total_distance - morning_distance * days_per_week) / days_per_week = forest_distance →
  forest_distance = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end alice_forest_walks_l302_302347


namespace sqrt_simplify_l302_302035

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l302_302035


namespace arithmetic_sequence_sum_l302_302476

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
  (h_arith : ∀ n, a n = a 0 + n * d)
  (h1 : a 0 + a 3 + a 6 = 45)
  (h2 : a 1 + a 4 + a 7 = 39) :
  a 2 + a 5 + a 8 = 33 := 
by
  sorry

end arithmetic_sequence_sum_l302_302476


namespace john_took_more_chickens_l302_302603

theorem john_took_more_chickens (john_chickens mary_chickens ray_chickens : ℕ)
  (h_ray : ray_chickens = 10)
  (h_mary : mary_chickens = ray_chickens + 6)
  (h_john : john_chickens = mary_chickens + 5) :
  john_chickens - ray_chickens = 11 := by
  sorry

end john_took_more_chickens_l302_302603


namespace seating_5_out_of_6_around_circle_l302_302765

def number_of_ways_to_seat_5_out_of_6_in_circle : Nat :=
  Nat.factorial 4

theorem seating_5_out_of_6_around_circle : number_of_ways_to_seat_5_out_of_6_in_circle = 24 :=
by {
  -- proof would be here
  sorry
}

end seating_5_out_of_6_around_circle_l302_302765


namespace complex_expression_eq_l302_302646

open Real

theorem complex_expression_eq (p q : ℝ) (hpq : p ≠ q) :
  (sqrt ((p^4 + q^4)/(p^4 - p^2 * q^2) + (2 * q^2)/(p^2 - q^2)) * (p^3 - p * q^2) - 2 * q * sqrt p) /
  (sqrt (p / (p - q) - q / (p + q) - 2 * p * q / (p^2 - q^2)) * (p - q)) = 
  sqrt (p^2 - q^2) / sqrt p := 
sorry

end complex_expression_eq_l302_302646


namespace total_pieces_of_gum_l302_302728

def packages := 43
def pieces_per_package := 23
def extra_pieces := 8

theorem total_pieces_of_gum :
  (packages * pieces_per_package) + extra_pieces = 997 := sorry

end total_pieces_of_gum_l302_302728


namespace johns_average_speed_l302_302254

noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

theorem johns_average_speed :
  average_speed 210 6.5 ≈ 32.31 := 
by
  sorry

end johns_average_speed_l302_302254


namespace jovana_added_23_pounds_l302_302831

def initial_weight : ℕ := 5
def final_weight : ℕ := 28

def added_weight : ℕ := final_weight - initial_weight

theorem jovana_added_23_pounds : added_weight = 23 := 
by sorry

end jovana_added_23_pounds_l302_302831


namespace sqrt_450_simplified_l302_302042

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l302_302042


namespace value_of_frac_l302_302389

theorem value_of_frac (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_frac_l302_302389


namespace sqrt_simplify_l302_302033

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l302_302033


namespace value_of_expression_l302_302405

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := by
  sorry

end value_of_expression_l302_302405


namespace simplify_sqrt_450_l302_302007

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l302_302007


namespace triangle_inequality_x_range_l302_302930

theorem triangle_inequality_x_range {x : ℝ} (h1 : 3 + 6 > x) (h2 : x + 3 > 6) :
  3 < x ∧ x < 9 :=
by 
  sorry

end triangle_inequality_x_range_l302_302930


namespace remainder_of_sum_is_12_l302_302761

theorem remainder_of_sum_is_12 (D k1 k2 : ℤ) (h1 : 242 = k1 * D + 4) (h2 : 698 = k2 * D + 8) : (242 + 698) % D = 12 :=
by
  sorry

end remainder_of_sum_is_12_l302_302761


namespace pigs_total_l302_302303

theorem pigs_total (initial_pigs : ℕ) (joined_pigs : ℕ) (total_pigs : ℕ) 
  (h1 : initial_pigs = 64) 
  (h2 : joined_pigs = 22) 
  : total_pigs = 86 :=
by
  sorry

end pigs_total_l302_302303


namespace original_population_correct_l302_302781

def original_population_problem :=
  let original_population := 6731
  let final_population := 4725
  let initial_disappeared := 0.10 * original_population
  let remaining_after_disappearance := original_population - initial_disappeared
  let left_after_remaining := 0.25 * remaining_after_disappearance
  let remaining_after_leaving := remaining_after_disappearance - left_after_remaining
  let disease_affected := 0.05 * original_population
  let disease_died := 0.02 * disease_affected
  let disease_migrated := 0.03 * disease_affected
  let remaining_after_disease := remaining_after_leaving - (disease_died + disease_migrated)
  let moved_to_village := 0.04 * remaining_after_disappearance
  let total_after_moving := remaining_after_disease + moved_to_village
  let births := 0.008 * original_population
  let deaths := 0.01 * original_population
  let final_population_calculated := total_after_moving + (births - deaths)
  final_population_calculated = final_population

theorem original_population_correct :
  original_population_problem ↔ True :=
by
  sorry

end original_population_correct_l302_302781


namespace distance_karen_covers_l302_302148

theorem distance_karen_covers
  (books_per_shelf : ℕ)
  (shelves : ℕ)
  (distance_to_library : ℕ)
  (h1 : books_per_shelf = 400)
  (h2 : shelves = 4)
  (h3 : distance_to_library = books_per_shelf * shelves) :
  2 * distance_to_library = 3200 := 
by
  sorry

end distance_karen_covers_l302_302148


namespace solve_equation_and_find_c_d_l302_302852

theorem solve_equation_and_find_c_d : 
  ∃ (c d : ℕ), (∃ x : ℝ, x^2 + 14 * x = 84 ∧ x = Real.sqrt c - d) ∧ c + d = 140 := 
sorry

end solve_equation_and_find_c_d_l302_302852


namespace value_of_ac_over_bd_l302_302406

theorem value_of_ac_over_bd (a b c d : ℝ) 
  (h1 : a = 4 * b)
  (h2 : b = 3 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by
  sorry

end value_of_ac_over_bd_l302_302406


namespace problem1_problem2_l302_302769

-- For Problem (1)
theorem problem1 (x : ℝ) : 2 * x - 3 > x + 1 → x > 4 := 
by sorry

-- For Problem (2)
theorem problem2 (a b : ℝ) (h : a^2 + 3 * a * b = 5) : (a + b) * (a + 2 * b) - 2 * b^2 = 5 := 
by sorry

end problem1_problem2_l302_302769


namespace find_x_from_conditions_l302_302956

theorem find_x_from_conditions (x y : ℝ)
  (h1 : (6 : ℝ) = (1 / 2 : ℝ) * x)
  (h2 : y = (1 / 2 :ℝ) * 10)
  (h3 : x * y = 60) : x = 12 := by
  sorry

end find_x_from_conditions_l302_302956


namespace find_value_of_fraction_l302_302419

theorem find_value_of_fraction (a b c d: ℕ) (h1: a = 4 * b) (h2: b = 3 * c) (h3: c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end find_value_of_fraction_l302_302419


namespace factorize_expression_l302_302909

variable {X M N : ℕ}

theorem factorize_expression (x m n : ℕ) : x * m - x * n = x * (m - n) :=
sorry

end factorize_expression_l302_302909


namespace card_game_probability_l302_302640

theorem card_game_probability :
  let n := 6
  let prob_empty (p q : ℕ) :=
    (p, q) = (9, 385) ∧ Nat.coprime p q
  let solution (p q : ℕ) :=
    p + q = 394
  ∃ p q : ℕ, prob_empty p q ∧ solution p q :=
begin
  have h : ∃ p q : ℕ, (p, q) = (9, 385) ∧ Nat.coprime p q,
  { use [9, 385],
    exact ⟨rfl, by norm_num⟩, },
  exact h,
end

end card_game_probability_l302_302640


namespace sqrt_450_eq_15_sqrt_2_l302_302076

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l302_302076


namespace determinant_A_l302_302791

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![
    ![  2,  4, -2],
    ![  3, -1,  5],
    ![-1,  3,  2]
  ]

theorem determinant_A : det A = -94 := by
  sorry

end determinant_A_l302_302791


namespace math_problem_l302_302928

theorem math_problem 
  (a : Int) (b : Int) (c : Int)
  (h_a : a = -1)
  (h_b : b = 1)
  (h_c : c = 0) :
  a + c - b = -2 := 
by
  sorry

end math_problem_l302_302928


namespace find_7th_term_l302_302601

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem find_7th_term 
    (a d : ℤ) 
    (h3 : a + 2 * d = 17) 
    (h5 : a + 4 * d = 39) : 
    arithmetic_sequence a d 7 = 61 := 
sorry

end find_7th_term_l302_302601


namespace circle_radius_integer_l302_302578

theorem circle_radius_integer (r : ℤ)
  (center : ℝ × ℝ)
  (inside_point : ℝ × ℝ)
  (outside_point : ℝ × ℝ)
  (h1 : center = (-2, -3))
  (h2 : inside_point = (-2, 2))
  (h3 : outside_point = (5, -3))
  (h4 : (dist center inside_point : ℝ) < r)
  (h5 : (dist center outside_point : ℝ) > r) 
  : r = 6 :=
sorry

end circle_radius_integer_l302_302578


namespace profit_percentage_l302_302653

theorem profit_percentage (CP SP : ℝ) (hCP : CP = 150) (hSP : SP = 216.67) :
  SP = 0.9 * LP ∧ LP = SP / 0.9 ∧ Profit = SP - CP ∧ Profit_Percentage = (Profit / CP) * 100 ∧ Profit_Percentage = 44.44 :=
by
  sorry

end profit_percentage_l302_302653


namespace ratio_jerky_l302_302704

/-
  Given conditions:
  1. Janette camps for 5 days.
  2. She has an initial 40 pieces of beef jerky.
  3. She eats 4 pieces of beef jerky per day.
  4. She will have 10 pieces of beef jerky left after giving some to her brother.

  Prove that the ratio of the pieces of beef jerky she gives to her brother 
  to the remaining pieces is 1:1.
-/

theorem ratio_jerky (days : ℕ) (initial_jerky : ℕ) (jerky_per_day : ℕ) (jerky_left_after_trip : ℕ)
  (h1 : days = 5) (h2 : initial_jerky = 40) (h3 : jerky_per_day = 4) (h4 : jerky_left_after_trip = 10) :
  (initial_jerky - days * jerky_per_day - jerky_left_after_trip) = jerky_left_after_trip :=
by
  sorry

end ratio_jerky_l302_302704


namespace team_A_games_42_l302_302855

noncomputable def team_games (a b : ℕ) : Prop :=
  (a * 2 / 3 + 7) = b * 5 / 8

theorem team_A_games_42 (a b : ℕ) (h1 : a * 2 / 3 = b * 5 / 8 - 7)
                                 (h2 : b = a + 14) :
  a = 42 :=
by
  sorry

end team_A_games_42_l302_302855


namespace sally_seashells_l302_302003

variable (M : ℝ)

theorem sally_seashells : 
  (1.20 * (M + M / 2) = 54) → M = 30 := 
by
  sorry

end sally_seashells_l302_302003


namespace problem_l302_302395

theorem problem (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by sorry

end problem_l302_302395


namespace total_students_at_competition_l302_302627

def KnowItAllHigh : ℕ := 50
def KarenHigh : ℕ := (3 / 5 : ℚ) * KnowItAllHigh
def CombinedSchools : ℕ := KnowItAllHigh + KarenHigh
def NovelCoronaHigh : ℕ := 2 * CombinedSchools
def TotalStudents := CombinedSchools + NovelCoronaHigh

theorem total_students_at_competition : TotalStudents = 240 := by
  sorry

end total_students_at_competition_l302_302627


namespace braden_total_amount_after_winning_l302_302901

noncomputable def initial_amount := 400
noncomputable def multiplier := 2

def total_amount_after_winning (initial: ℕ) (mult: ℕ) : ℕ := initial + (mult * initial)

theorem braden_total_amount_after_winning : total_amount_after_winning initial_amount multiplier = 1200 := by
  sorry

end braden_total_amount_after_winning_l302_302901


namespace seq_geometric_and_k_range_l302_302718

-- Definitions and Conditions
def seq (b : ℕ → ℝ) : Prop :=
  b 1 = 7 / 2 ∧ ∀ n : ℕ, b (n + 1) = (1 / 2) * b n + 1 / 4

def T (b : ℕ → ℝ) (n : ℕ) : ℝ := (Finset.range n).sum b

theorem seq_geometric_and_k_range (b : ℕ → ℝ) (k : ℝ) :
  seq b →
  (∃ a r, ∀ n, b (n + 1) - 1 / 2 = r * (b n - 1 / 2) ∧ a = b 1 - 1 / 2 ∧ r = 1 / 2) →
  (∀ n : ℕ, b n = 3 * (1 / 2)^(n-1) + 1 / 2) →
  (∀ n : ℕ, ∀ k : ℝ, (2 * T b n + 3 * 2^(2 * n - 1) - 10) / k ≤ n^2 + 4 * n + 5 → k ≥ 3 / 10) :=
by
  intros
  sorry

end seq_geometric_and_k_range_l302_302718


namespace circumcenter_lies_on_ak_l302_302641

noncomputable def triangle_circumcenter_lies_on_ak
  {α β γ : ℝ}
  (A B C L H K O : Type*)
  [triangle A B C]
  [angle_bisector A L]
  [height_from B H]
  [circumcircle_of_triangle A (B ∧ L) K]
  [circumcenter_of_triangle A B C O]
  : Prop :=
  lies_on_line O (line_through A K)

-- We'll add the assumptions as hypotheses to the lemma
theorem circumcenter_lies_on_ak 
  {α β γ : ℝ} {A B C L H K O : Type*}
  [triangle A B C]
  [angle_bisector A L]
  [height_from B H]
  [circumcircle_of_triangle A (B ∧ L) K]
  [circumcenter_of_triangle A B C O]
  : lies_on_line O (line_through A K) :=
sorry

end circumcenter_lies_on_ak_l302_302641


namespace length_of_bridge_l302_302166

theorem length_of_bridge (length_train : ℕ) (speed_train_kmh : ℕ) (crossing_time_sec : ℕ)
    (h_length_train : length_train = 125)
    (h_speed_train_kmh : speed_train_kmh = 45)
    (h_crossing_time_sec : crossing_time_sec = 30) : 
    ∃ (length_bridge : ℕ), length_bridge = 250 := by
  sorry

end length_of_bridge_l302_302166


namespace integral_evaluation_l302_302659

noncomputable def integral_result : ℝ :=
  ∫ x in (0:ℝ)..(1:ℝ), (Real.sqrt (1 - x^2) - x)

theorem integral_evaluation :
  integral_result = (Real.pi - 2) / 4 :=
by
  sorry

end integral_evaluation_l302_302659


namespace number_of_cities_experienced_protests_l302_302490

variables (days_of_protest : ℕ) (arrests_per_day : ℕ) (days_pre_trial : ℕ) 
          (days_post_trial_in_weeks : ℕ) (combined_weeks_jail : ℕ)

def total_days_in_jail_per_person := days_pre_trial + (days_post_trial_in_weeks * 7) / 2

theorem number_of_cities_experienced_protests 
  (h1 : days_of_protest = 30) 
  (h2 : arrests_per_day = 10) 
  (h3 : days_pre_trial = 4) 
  (h4 : days_post_trial_in_weeks = 2) 
  (h5 : combined_weeks_jail = 9900) : 
  (combined_weeks_jail * 7) / total_days_in_jail_per_person 
  = 21 :=
by
  sorry

end number_of_cities_experienced_protests_l302_302490


namespace min_f_of_shangmei_number_l302_302364

def is_shangmei_number (a b c d : ℕ) : Prop :=
  a + c = 11 ∧ b + d = 11

def f (a b : ℕ) : ℚ :=
  (b - (11 - b) : ℚ) / (a - (11 - a))

def G (a b : ℕ) : ℤ :=
  20 * a + 2 * b - 121

def is_multiple_of_7 (x : ℤ) : Prop :=
  ∃ k : ℤ, x = 7 * k

theorem min_f_of_shangmei_number :
  ∃ (a b c d : ℕ), a < b ∧ is_shangmei_number a b c d ∧ is_multiple_of_7 (G a b) ∧ f a b = -3 :=
sorry

end min_f_of_shangmei_number_l302_302364


namespace integer_solutions_l302_302537

theorem integer_solutions (x y z : ℤ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : x + y + z ≠ 0) :
  (1 / x + 1 / y + 1 / z = 1 / (x + y + z)) ↔ (z = -x - y) :=
sorry

end integer_solutions_l302_302537


namespace length_of_symmetric_chord_l302_302934

noncomputable theory

-- Define the parabola y = -x^2 + 3
def parabola (x : ℝ) : ℝ :=
  -x^2 + 3

-- Given that points A and B are symmetric about the line x + y = 0
-- Define the symmetric condition
def symmetric (A B : ℝ × ℝ) : Prop :=
  A.1 + A.2 = 0 ∧ B.1 + B.2 = 0 ∧ A.1 = -B.1 ∧ A.2 = -B.2

-- Define line passing through points a and b.
def line_segment_length (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem length_of_symmetric_chord :
  ∀ A B : ℝ × ℝ, symmetric A B ∧ A.2 = parabola A.1 ∧ B.2 = parabola B.1 → line_segment_length A B = 3 * real.sqrt 2 :=
sorry

end length_of_symmetric_chord_l302_302934


namespace find_digit_l302_302759

theorem find_digit (a : ℕ) (n1 n2 n3 : ℕ) (h1 : n1 = a * 1000) (h2 : n2 = a * 1000 + 998) (h3 : n3 = a * 1000 + 999) (h4 : n1 + n2 + n3 = 22997) :
  a = 7 :=
by
  sorry

end find_digit_l302_302759


namespace evaluate_expression_l302_302536

noncomputable def repeating_to_fraction_06 : ℚ := 2 / 3
noncomputable def repeating_to_fraction_02 : ℚ := 2 / 9
noncomputable def repeating_to_fraction_04 : ℚ := 4 / 9

theorem evaluate_expression : 
  ((repeating_to_fraction_06 * repeating_to_fraction_02) - repeating_to_fraction_04) = -8 / 27 := 
by 
  sorry

end evaluate_expression_l302_302536


namespace regular_dodecahedron_has_12_faces_l302_302819

-- Define a structure to represent a regular dodecahedron
structure RegularDodecahedron where

-- The main theorem to state that a regular dodecahedron has 12 faces
theorem regular_dodecahedron_has_12_faces (D : RegularDodecahedron) : ∃ faces : ℕ, faces = 12 := by
  sorry

end regular_dodecahedron_has_12_faces_l302_302819


namespace min_value_of_mn_squared_l302_302675

theorem min_value_of_mn_squared 
  (a b c : ℝ) 
  (h : a^2 + b^2 = c^2) 
  (m n : ℝ) 
  (h_point : a * m + b * n + 2 * c = 0) : 
  m^2 + n^2 = 4 :=
sorry

end min_value_of_mn_squared_l302_302675


namespace ratio_fraction_l302_302991

theorem ratio_fraction (A B C : ℕ) (h1 : 7 * B = 3 * A) (h2 : 6 * C = 5 * B) :
  (C : ℚ) / (A : ℚ) = 5 / 14 ∧ (A : ℚ) / (C : ℚ) = 14 / 5 :=
by
  sorry

end ratio_fraction_l302_302991


namespace man_speed_is_4_kmph_l302_302181

noncomputable def speed_of_man (train_length : ℝ) (train_speed_kmph : ℝ) (time_to_pass_seconds : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let relative_speed_mps := train_length / time_to_pass_seconds
  let relative_speed_kmph := relative_speed_mps * 3600 / 1000
  relative_speed_kmph - train_speed_kmph

theorem man_speed_is_4_kmph : speed_of_man 140 50 9.332586726395222 = 4 := by
  sorry

end man_speed_is_4_kmph_l302_302181


namespace sqrt_450_simplified_l302_302047

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l302_302047


namespace john_bought_packs_l302_302580

def students_in_classes : List ℕ := [20, 25, 18, 22, 15]
def packs_per_student : ℕ := 3

theorem john_bought_packs :
  (students_in_classes.sum) * packs_per_student = 300 := by
  sorry

end john_bought_packs_l302_302580


namespace number_of_perfect_square_factors_450_l302_302234

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def prime_factorization_450 := (2, 1) :: (3, 2) :: (5, 2) :: []

def perfect_square_factors (n : ℕ) : ℕ :=
  if n = 450 then 4 else 0

theorem number_of_perfect_square_factors_450 : perfect_square_factors 450 = 4 :=
by
  sorry

end number_of_perfect_square_factors_450_l302_302234


namespace ram_total_distance_l302_302978

noncomputable def total_distance 
  (speed1 speed2 time1 total_time : ℝ) 
  (h_speed1 : speed1 = 20) 
  (h_speed2 : speed2 = 70)
  (h_time1 : time1 = 3.2)
  (h_total_time : total_time = 8) 
  : ℝ := 
  speed1 * time1 + speed2 * (total_time - time1)

theorem ram_total_distance
  (speed1 speed2 time1 total_time : ℝ)
  (h_speed1 : speed1 = 20)
  (h_speed2 : speed2 = 70)
  (h_time1 : time1 = 3.2)
  (h_total_time : total_time = 8)
  : total_distance speed1 speed2 time1 total_time h_speed1 h_speed2 h_time1 h_total_time = 400 :=
  sorry

end ram_total_distance_l302_302978


namespace quadratic_root_value_l302_302215
-- Import the entirety of the necessary library

-- Define the quadratic equation with one root being -1
theorem quadratic_root_value 
    (m : ℝ)
    (h1 : ∀ x : ℝ, x^2 + m * x + 3 = 0)
    (root1 : -1 ∈ {x : ℝ | x^2 + m * x + 3 = 0}) :
    m = 4 ∧ ∃ root2 : ℝ, root2 = -3 ∧ root2 ∈ {x : ℝ | x^2 + m * x + 3 = 0} :=
by
  sorry

end quadratic_root_value_l302_302215


namespace simplify_sqrt_450_l302_302027

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l302_302027


namespace maximum_sequence_length_l302_302318

theorem maximum_sequence_length
  (seq : List ℚ) 
  (h1 : ∀ i : ℕ, i + 2 < seq.length → (seq.get! i + seq.get! (i+1) + seq.get! (i+2)) < 0)
  (h2 : ∀ i : ℕ, i + 3 < seq.length → (seq.get! i + seq.get! (i+1) + seq.get! (i+2) + seq.get! (i+3)) > 0) 
  : seq.length ≤ 5 := 
sorry

end maximum_sequence_length_l302_302318


namespace polynomial_value_l302_302357

variables (x y p q : ℝ)

theorem polynomial_value (h1 : x + y = -p) (h2 : xy = q) :
  x * (1 + y) - y * (x * y - 1) - x^2 * y = pq + q - p :=
by
  sorry

end polynomial_value_l302_302357


namespace value_of_ac_over_bd_l302_302410

theorem value_of_ac_over_bd (a b c d : ℝ) 
  (h1 : a = 4 * b)
  (h2 : b = 3 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by
  sorry

end value_of_ac_over_bd_l302_302410


namespace geometric_sequence_sum_l302_302668

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) 
  (h_geo : ∀ n, a (n + 1) = (3 : ℝ) * ((-2 : ℝ) ^ n))
  (h_first : a 1 = 3)
  (h_ratio_ne_1 : -2 ≠ 1)
  (h_arith : 2 * a 3 = a 4 + a 5) :
  a 1 + a 2 + a 3 + a 4 + a 5 = 33 := 
sorry

end geometric_sequence_sum_l302_302668


namespace inversions_range_l302_302598

/-- Given any permutation of 10 elements, 
    the number of inversions (or disorders) in the permutation 
    can take any value from 0 to 45.
-/
theorem inversions_range (perm : List ℕ) (h_length : perm.length = 10):
  ∃ S, 0 ≤ S ∧ S ≤ 45 :=
sorry

end inversions_range_l302_302598


namespace fraction_not_going_l302_302169

theorem fraction_not_going (S J : ℕ) (h1 : J = (2:ℕ)/3 * S) 
  (h_not_junior : 3/4 * J = 3/4 * (2/3 * S)) 
  (h_not_senior : 1/3 * S = (1:ℕ)/3 * S) :
  3/4 * (2/3 * S) + 1/3 * S = 5/6 * S :=
by 
  sorry

end fraction_not_going_l302_302169


namespace first_step_is_remove_parentheses_l302_302139

variable (x : ℝ)

def equation : Prop := 2 * x + 3 * (2 * x - 1) = 16 - (x + 1)

theorem first_step_is_remove_parentheses (x : ℝ) (eq : equation x) : 
  ∃ step : String, step = "remove the parentheses" := 
  sorry

end first_step_is_remove_parentheses_l302_302139


namespace projection_is_negative_sqrt_10_l302_302373

noncomputable def projection_of_AB_in_direction_of_AC : ℝ :=
  let A := (1, 1)
  let B := (-3, 3)
  let C := (4, 2)
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let dot_product := AB.1 * AC.1 + AB.2 * AC.2
  let magnitude_AC := Real.sqrt (AC.1^2 + AC.2^2)
  dot_product / magnitude_AC

theorem projection_is_negative_sqrt_10 :
  projection_of_AB_in_direction_of_AC = -Real.sqrt 10 :=
by
  sorry

end projection_is_negative_sqrt_10_l302_302373


namespace sqrt_450_equals_15_sqrt_2_l302_302077

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l302_302077


namespace simplify_sqrt_450_l302_302062

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l302_302062


namespace total_chickens_l302_302455

theorem total_chickens (x : ℕ) (h : 40 + (5 * x) / 12 = (x + 40) / 2) : x + 40 = 280 :=
by
  sorry

end total_chickens_l302_302455


namespace minimum_days_l302_302509

theorem minimum_days (n : ℕ) (rain_afternoon : ℕ) (sunny_afternoon : ℕ) (sunny_morning : ℕ) :
  rain_afternoon + sunny_afternoon = 7 ∧
  sunny_afternoon <= 5 ∧
  sunny_morning <= 6 ∧
  sunny_morning + rain_afternoon = 7 ∧
  n = 11 :=
by
  sorry

end minimum_days_l302_302509


namespace sqrt_450_eq_15_sqrt_2_l302_302115

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302115


namespace sqrt_450_equals_15_sqrt_2_l302_302079

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l302_302079


namespace remainder_2365947_div_8_l302_302634

theorem remainder_2365947_div_8 : (2365947 % 8) = 3 :=
by
  sorry

end remainder_2365947_div_8_l302_302634


namespace smallest_Y_l302_302801

theorem smallest_Y (U : ℕ) (Y : ℕ) (hU : U = 15 * Y) 
  (digits_U : ∀ d ∈ Nat.digits 10 U, d = 0 ∨ d = 1) 
  (div_15 : U % 15 = 0) : Y = 74 :=
sorry

end smallest_Y_l302_302801


namespace ratio_equivalence_l302_302532

theorem ratio_equivalence (m n s u : ℚ) (h1 : m / n = 5 / 4) (h2 : s / u = 8 / 15) :
  (5 * m * s - 2 * n * u) / (7 * n * u - 10 * m * s) = 4 / 3 :=
by
  sorry

end ratio_equivalence_l302_302532


namespace contrapositive_example_l302_302724

theorem contrapositive_example (a b : ℝ) (h : a^2 + b^2 < 4) : a + b ≠ 3 :=
sorry

end contrapositive_example_l302_302724


namespace simplify_sqrt_450_l302_302005

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l302_302005


namespace rhombus_side_length_l302_302175

-- Define the rhombus properties and the problem conditions
variables (p q x : ℝ)

-- State the problem as a theorem in Lean 4
theorem rhombus_side_length (h : x^2 = p * q) : x = Real.sqrt (p * q) :=
sorry

end rhombus_side_length_l302_302175


namespace minimum_value_abs_a_plus_2_abs_b_l302_302585

open Real

theorem minimum_value_abs_a_plus_2_abs_b 
  (a b : ℝ)
  (f : ℝ → ℝ)
  (x₁ x₂ x₃ : ℝ)
  (f_def : ∀ x, f x = x^3 + a*x^2 + b*x)
  (roots_cond : x₁ + 1 ≤ x₂ ∧ x₂ ≤ x₃ - 1)
  (equal_values : f x₁ = f x₂ ∧ f x₂ = f x₃) :
  ∃ minimum, minimum = (sqrt 3) ∧ (∀ (a b : ℝ), |a| + 2*|b| ≥ sqrt 3) :=
by
  sorry

end minimum_value_abs_a_plus_2_abs_b_l302_302585


namespace part1_l302_302647

theorem part1 (a b c : ℝ) (h : 1 / (a + b) + 1 / (b + c) = 2 / (c + a)) : 2 * b^2 = a^2 + c^2 :=
sorry

end part1_l302_302647


namespace radius_squared_l302_302804

-- Definitions of the conditions
def point_A := (2, -1)
def line_l1 (x y : ℝ) := x + y = 1
def line_l2 (x y : ℝ) := 2 * x + y = 0

-- Circle with center (h, k) and radius r
def circle_equation (h k r x y : ℝ) := (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2

-- Prove statement: r^2 = 2 given the conditions
theorem radius_squared (h k r : ℝ) 
  (H1 : circle_equation h k r 2 (-1))
  (H2 : line_l1 h k)
  (H3 : line_l2 h k):
  r ^ 2 = 2 := sorry

end radius_squared_l302_302804


namespace phi_varphi_difference_squared_l302_302561

theorem phi_varphi_difference_squared :
  ∀ (Φ φ : ℝ), (Φ ≠ φ) → (Φ^2 - 2*Φ - 1 = 0) → (φ^2 - 2*φ - 1 = 0) →
  (Φ - φ)^2 = 8 :=
by
  intros Φ φ distinct hΦ hφ
  sorry

end phi_varphi_difference_squared_l302_302561


namespace cannot_form_set_l302_302514

-- Definitions based on the conditions
def GroupA := {x : ℝ | abs (x - 2) < δ} -- This is not well-defined without δ
def GroupB := {x : ℝ | x^2 - 1 = 0}
def GroupC := {T : Type | T = EquilateralTriangle}
def GroupD := {n : ℕ | n < 10}

theorem cannot_form_set (δ : ℝ) : 
  (∀ (S : Set ℝ), S = GroupA → ¬(∃ δ > 0, ∀ x ∈ S, abs (x - 2) < δ)) ∧ 
  ((∃ (S : Set ℝ), S = GroupB) ∧ 
   (∃ (S : Set Type), S = GroupC) ∧ 
   (∃ (S : Set ℕ), S = GroupD)) := 
by
  sorry

end cannot_form_set_l302_302514


namespace count_total_wheels_l302_302749

theorem count_total_wheels (trucks : ℕ) (cars : ℕ) (truck_wheels : ℕ) (car_wheels : ℕ) :
  trucks = 12 → cars = 13 → truck_wheels = 4 → car_wheels = 4 →
  (trucks * truck_wheels + cars * car_wheels) = 100 :=
by
  intros h_trucks h_cars h_truck_wheels h_car_wheels
  sorry

end count_total_wheels_l302_302749


namespace find_value_of_fraction_l302_302417

theorem find_value_of_fraction (a b c d: ℕ) (h1: a = 4 * b) (h2: b = 3 * c) (h3: c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end find_value_of_fraction_l302_302417


namespace total_distance_hiked_l302_302846

theorem total_distance_hiked
  (a b c d e : ℕ)
  (h1 : a + b + c = 34)
  (h2 : b + c = 24)
  (h3 : c + d + e = 40)
  (h4 : a + c + e = 38)
  (h5 : d = 14) :
  a + b + c + d + e = 48 :=
by
  sorry

end total_distance_hiked_l302_302846


namespace discriminant_of_quadratic_l302_302796

-- Define the quadratic equation coefficients
def a : ℝ := 5
def b : ℝ := -11
def c : ℝ := 4

-- Prove the discriminant of the quadratic equation
theorem discriminant_of_quadratic :
    b^2 - 4 * a * c = 41 :=
by
  sorry

end discriminant_of_quadratic_l302_302796


namespace min_value_x_plus_y_l302_302672

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : (2 * x + Real.sqrt (4 * x^2 + 1)) * (Real.sqrt (y^2 + 4) - 2) ≥ y) : 
  x + y >= 2 := 
by
  sorry

end min_value_x_plus_y_l302_302672


namespace sequence_two_cases_l302_302179

noncomputable def sequence_property (a : ℕ → ℕ) : Prop :=
  (∀ n, a n ≥ a (n-1)) ∧  -- nondecreasing
  (∃ n m, n ≠ m ∧ a n ≠ a m) ∧  -- nonconstant
  (∀ n, a n ∣ n^2)  -- a_n | n^2

theorem sequence_two_cases (a : ℕ → ℕ) :
  sequence_property a →
  (∃ n1, ∀ n ≥ n1, a n = n) ∨ (∃ n2, ∀ n ≥ n2, a n = n^2) :=
by {
  sorry
}

end sequence_two_cases_l302_302179


namespace fourth_competitor_jump_l302_302141

theorem fourth_competitor_jump :
  let first_jump := 22
  let second_jump := first_jump + 1
  let third_jump := second_jump - 2
  let fourth_jump := third_jump + 3
  fourth_jump = 24 := by
  sorry

end fourth_competitor_jump_l302_302141


namespace ksyusha_travel_time_l302_302259

theorem ksyusha_travel_time :
  ∀ (S v : ℝ), 
  (2 * S / v + S / (2 * v) = 30) → 
  (2 * (S / v) = 24) :=
by
  intros S v h1
  have h2 : S / v = 12 := by
    calc 
      S / v = (30 * 2) / 5 := by sorry
  calc 
    2 * (S / v) = 2 * 12 := by sorry
  ... = 24 := by norm_num

end ksyusha_travel_time_l302_302259


namespace roger_collected_nickels_l302_302980

theorem roger_collected_nickels 
  (N : ℕ)
  (initial_pennies : ℕ := 42) 
  (initial_dimes : ℕ := 15)
  (donated_coins : ℕ := 66)
  (left_coins : ℕ := 27)
  (h_total_coins_initial : initial_pennies + N + initial_dimes - donated_coins = left_coins) :
  N = 36 := 
sorry

end roger_collected_nickels_l302_302980


namespace smallest_positive_integer_exists_l302_302337

theorem smallest_positive_integer_exists :
  ∃ (n : ℕ), n > 0 ∧ (∃ (k m : ℕ), n = 5 * k + 3 ∧ n = 12 * m) ∧ n = 48 :=
by
  sorry

end smallest_positive_integer_exists_l302_302337


namespace maria_threw_out_carrots_l302_302838

theorem maria_threw_out_carrots (initially_picked: ℕ) (picked_next_day: ℕ) (total_now: ℕ) (carrots_thrown_out: ℕ) :
  initially_picked = 48 → 
  picked_next_day = 15 → 
  total_now = 52 → 
  (initially_picked + picked_next_day - total_now = carrots_thrown_out) → 
  carrots_thrown_out = 11 :=
by
  intros
  sorry

end maria_threw_out_carrots_l302_302838


namespace value_of_frac_l302_302391

theorem value_of_frac (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_frac_l302_302391


namespace chess_group_players_l302_302304

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 190) : n = 20 :=
sorry

end chess_group_players_l302_302304


namespace probability_inside_octahedron_l302_302892

noncomputable def probability_of_octahedron : ℝ := 
  let cube_volume := 8
  let octahedron_volume := 4 / 3
  octahedron_volume / cube_volume

theorem probability_inside_octahedron :
  probability_of_octahedron = 1 / 6 :=
  by
    sorry

end probability_inside_octahedron_l302_302892


namespace sum_abc_equals_16_l302_302276

theorem sum_abc_equals_16 (a b c : ℝ) (h : (a - 2)^2 + (b - 6)^2 + (c - 8)^2 = 0) : 
  a + b + c = 16 :=
by
  sorry

end sum_abc_equals_16_l302_302276


namespace wanda_walks_days_per_week_l302_302756

theorem wanda_walks_days_per_week 
  (daily_distance : ℝ) (weekly_distance : ℝ) (weeks : ℕ) (total_distance : ℝ) 
  (h_daily_walk: daily_distance = 2) 
  (h_total_walk: total_distance = 40) 
  (h_weeks: weeks = 4) : 
  ∃ d : ℕ, (d * daily_distance * weeks = total_distance) ∧ (d = 5) := 
by 
  sorry

end wanda_walks_days_per_week_l302_302756


namespace unique_two_digit_integer_s_l302_302486

-- We define s to satisfy the two given conditions.
theorem unique_two_digit_integer_s (s : ℕ) (h1 : 13 * s % 100 = 52) (h2 : 1 ≤ s) (h3 : s ≤ 99) : s = 4 :=
sorry

end unique_two_digit_integer_s_l302_302486


namespace five_digit_number_is_40637_l302_302533

theorem five_digit_number_is_40637 
  (A B C D E F G : ℕ) 
  (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ 
        B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ 
        C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ 
        D ≠ E ∧ D ≠ F ∧ D ≠ G ∧
        E ≠ F ∧ E ≠ G ∧ 
        F ≠ G)
  (h2 : 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D ∧ 0 < E ∧ 0 < F ∧ 0 < G)
  (h3 : A + 11 * A = 2 * (10 * B + A))
  (h4 : A + 10 * C + D = 2 * (10 * A + B))
  (h5 : 10 * C + D = 20 * A)
  (h6 : 20 + 62 = 2 * (10 * C + A)) -- for sequences formed by AB, CA, EF
  (h7 : 21 + 63 = 2 * (10 * G + A)) -- for sequences formed by BA, CA, GA
  : ∃ (C D E F G : ℕ), C * 10000 + D * 1000 + E * 100 + F * 10 + G = 40637 := 
sorry

end five_digit_number_is_40637_l302_302533


namespace simplify_sqrt_450_l302_302085

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l302_302085


namespace michael_regular_hours_l302_302840

-- Define the constants and conditions
def regular_rate : ℝ := 7
def overtime_rate : ℝ := 14
def total_earnings : ℝ := 320
def total_hours : ℝ := 42.857142857142854

-- Declare the proof problem
theorem michael_regular_hours :
  ∃ R O : ℝ, (regular_rate * R + overtime_rate * O = total_earnings) ∧ (R + O = total_hours) ∧ (R = 40) :=
by
  sorry

end michael_regular_hours_l302_302840


namespace problem_l302_302396

theorem problem (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by sorry

end problem_l302_302396


namespace find_a2_l302_302935

noncomputable def a_sequence (k : ℕ+) (n : ℕ) : ℚ :=
  -(1 / 2 : ℚ) * n^2 + k * n

theorem find_a2
  (k : ℕ+)
  (max_S : ∀ n : ℕ, a_sequence k n ≤ 8)
  (max_reached : ∃ n : ℕ, a_sequence k n = 8) :
  a_sequence 4 2 - a_sequence 4 1 = 5 / 2 :=
by
  -- To be proved, insert appropriate steps here
  sorry

end find_a2_l302_302935


namespace sqrt_450_simplified_l302_302048

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l302_302048


namespace max_correct_answers_l302_302241

/--
In a 50-question multiple-choice math contest, students receive 5 points for a correct answer, 
0 points for an answer left blank, and -2 points for an incorrect answer. Jesse’s total score 
on the contest was 115. Prove that the maximum number of questions that Jesse could have answered 
correctly is 30.
-/
theorem max_correct_answers (a b c : ℕ) (h1 : a + b + c = 50) (h2 : 5 * a - 2 * c = 115) : a ≤ 30 :=
by
  sorry

end max_correct_answers_l302_302241


namespace households_neither_car_nor_bike_l302_302954

-- Define the given conditions
def total_households : ℕ := 90
def car_and_bike : ℕ := 18
def households_with_car : ℕ := 44
def bike_only : ℕ := 35

-- Prove the number of households with neither car nor bike
theorem households_neither_car_nor_bike :
  (total_households - ((households_with_car + bike_only) - car_and_bike)) = 11 :=
by
  sorry

end households_neither_car_nor_bike_l302_302954


namespace John_ASMC_score_l302_302581

def ASMC_score (c w : ℕ) : ℕ := 25 + 5 * c - 2 * w

theorem John_ASMC_score (c w : ℕ) (h1 : ASMC_score c w = 100) (h2 : c + w ≤ 25) :
  c = 19 ∧ w = 10 :=
by {
  sorry
}

end John_ASMC_score_l302_302581


namespace find_a_minimum_value_at_x_2_l302_302230

def f (x a : ℝ) := x^3 - a * x

theorem find_a_minimum_value_at_x_2 (a : ℝ) :
  (∃ x : ℝ, x = 2 ∧ ∀ y ≠ 2, f y a ≥ f 2 a) → a = 12 :=
by 
  -- Here we should include the proof steps
  sorry

end find_a_minimum_value_at_x_2_l302_302230


namespace value_of_fraction_l302_302428

-- Definitions based on conditions
variables {a b c d : ℝ}
hypothesis h1 : a = 4 * b
hypothesis h2 : b = 3 * c
hypothesis h3 : c = 5 * d

-- The statement we want to prove
theorem value_of_fraction : (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_fraction_l302_302428


namespace probability_not_math_physics_together_l302_302327

open Classical

def subjects := {"Chinese", "Mathematics", "English", "Physics", "Chemistry", "Biology"}

noncomputable def totalWays := Nat.choose 6 3
noncomputable def waysMathPhysicsTogether := Nat.choose 4 1
noncomputable def probabilityNotMathPhysicsTogether := 1 - (waysMathPhysicsTogether / totalWays : ℚ)

theorem probability_not_math_physics_together : probabilityNotMathPhysicsTogether = 4 / 5 := sorry

end probability_not_math_physics_together_l302_302327


namespace water_percentage_in_tomato_juice_l302_302557

-- Definitions from conditions
def tomato_juice_volume := 80 -- in liters
def tomato_puree_volume := 10 -- in liters
def tomato_puree_water_percentage := 20 -- in percent (20%)

-- Need to prove percentage of water in tomato juice is 20%
theorem water_percentage_in_tomato_juice : 
  (100 - tomato_puree_water_percentage) * tomato_puree_volume / tomato_juice_volume = 20 :=
by
  -- Skip the proof
  sorry

end water_percentage_in_tomato_juice_l302_302557


namespace find_m_l302_302936

theorem find_m (m : ℝ) 
  (h : (1 : ℝ) * (-3 : ℝ) + (3 : ℝ) * ((3 : ℝ) + 2 * m) = 0) : 
  m = -1 :=
by sorry

end find_m_l302_302936


namespace simplify_sqrt_450_l302_302015
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l302_302015


namespace simplify_sqrt_450_l302_302017
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l302_302017


namespace find_multiple_of_games_l302_302864

-- declaring the number of video games each person has
def Tory_videos := 6
def Theresa_videos := 11
def Julia_videos := Tory_videos / 3

-- declaring the multiple we need to find
def multiple_of_games := Theresa_videos - Julia_videos * 5

-- Theorem stating the problem
theorem find_multiple_of_games : ∃ m : ℕ, Julia_videos * m + 5 = Theresa_videos :=
by
  sorry

end find_multiple_of_games_l302_302864


namespace sum_of_first_13_terms_is_39_l302_302227

-- Definition of arithmetic sequence and the given condition
def arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

noncomputable def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

-- Given condition
axiom given_condition {a : ℕ → ℤ} (h : arithmetic_sequence a) : a 5 + a 6 + a 7 = 9

-- The main theorem
theorem sum_of_first_13_terms_is_39 {a : ℕ → ℤ} (h : arithmetic_sequence a) (h9 : a 5 + a 6 + a 7 = 9) : sum_of_first_n_terms a 12 = 39 :=
sorry

end sum_of_first_13_terms_is_39_l302_302227


namespace suraj_new_average_l302_302733

noncomputable def suraj_average (A : ℝ) : ℝ := A + 8

theorem suraj_new_average (A : ℝ) (h_conditions : 14 * A + 140 = 15 * (A + 8)) :
  suraj_average A = 28 :=
by
  sorry

end suraj_new_average_l302_302733


namespace annual_decrease_rate_l302_302297

def initial_population : ℝ := 8000
def population_after_two_years : ℝ := 3920

theorem annual_decrease_rate :
  ∃ r : ℝ, (0 < r ∧ r < 1) ∧ (initial_population * (1 - r)^2 = population_after_two_years) ∧ r = 0.3 :=
by
  sorry

end annual_decrease_rate_l302_302297


namespace intersection_M_N_l302_302381

open Set

def M : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def N : Set ℝ := {-3, -2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {-2, -1, 0} :=
by
  sorry

end intersection_M_N_l302_302381


namespace simplify_sqrt_450_l302_302068

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l302_302068


namespace gain_percent_correct_l302_302764

variable (CP SP Gain : ℝ)
variable (H₁ : CP = 900)
variable (H₂ : SP = 1125)
variable (H₃ : Gain = SP - CP)

theorem gain_percent_correct : (Gain / CP) * 100 = 25 :=
by
  sorry

end gain_percent_correct_l302_302764


namespace price_of_fruits_l302_302920

theorem price_of_fruits
  (x y : ℝ)
  (h1 : 9 * x + 10 * y = 73.8)
  (h2 : 17 * x + 6 * y = 69.8)
  (hx : x = 2.2)
  (hy : y = 5.4) : 
  9 * 2.2 + 10 * 5.4 = 73.8 ∧ 17 * 2.2 + 6 * 5.4 = 69.8 :=
by
  sorry

end price_of_fruits_l302_302920


namespace simplify_sqrt_450_l302_302009

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l302_302009


namespace isle_of_unluckiness_l302_302591

-- Definitions:
def is_knight (i : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k = i * n / 100 ∧ k > 0

-- Main statement:
theorem isle_of_unluckiness (n : ℕ) (h : n ∈ [1, 2, 4, 5, 10, 20, 25, 50, 100]) :
  ∃ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ is_knight i n := by
  sorry

end isle_of_unluckiness_l302_302591


namespace extreme_values_l302_302374

def f (x : ℝ) : ℝ := x^3 - 4 * x^2 + 5 * x - 4

theorem extreme_values :
  (∃ (x1 x2 : ℝ), x1 = 1 ∧ x2 = 5 / 3 ∧ f x1 = -2 ∧ f x2 = -58 / 27) ∧ 
  (∃ (a b : ℝ), a = 2 ∧ b = f 2 ∧ (∀ (x : ℝ), (a, b) = (x, f x) → (∀ y : ℝ, y = x - 4))) :=
by
  sorry

end extreme_values_l302_302374


namespace common_difference_of_AP_l302_302913

theorem common_difference_of_AP (a T_12 : ℝ) (d : ℝ) (n : ℕ) (h1 : a = 2) (h2 : T_12 = 90) (h3 : n = 12) 
(h4 : T_12 = a + (n - 1) * d) : d = 8 := 
by sorry

end common_difference_of_AP_l302_302913


namespace basic_computer_price_l302_302149

theorem basic_computer_price (C P : ℝ)
  (h1 : C + P = 2500)
  (h2 : P = (C + 500 + P) / 3)
  : C = 1500 :=
sorry

end basic_computer_price_l302_302149


namespace percent_problem_l302_302951

theorem percent_problem :
  ∀ (x : ℝ), 0.60 * 600 = 0.50 * x → x = 720 :=
by
  intros x h
  sorry

end percent_problem_l302_302951


namespace find_value_of_fraction_l302_302416

theorem find_value_of_fraction (a b c d: ℕ) (h1: a = 4 * b) (h2: b = 3 * c) (h3: c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end find_value_of_fraction_l302_302416


namespace simplify_sqrt_450_l302_302094

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l302_302094


namespace initial_children_on_bus_l302_302853

-- Define the conditions
variables (x : ℕ)

-- Define the problem statement
theorem initial_children_on_bus (h : x + 7 = 25) : x = 18 :=
sorry

end initial_children_on_bus_l302_302853


namespace total_biking_distance_l302_302145

-- Define the problem conditions 
def shelves := 4
def books_per_shelf := 400
def one_way_distance := shelves * books_per_shelf

-- Prove that the total distance for a round trip is 3200 miles
theorem total_biking_distance : 2 * one_way_distance = 3200 :=
by sorry

end total_biking_distance_l302_302145


namespace line_intersect_curve_area_l302_302441

noncomputable def line_param_eq (α : ℝ) (t : ℝ) : ℝ × ℝ :=
  (1 + t * Real.cos α, t * Real.sin α)

noncomputable def curve_cart_eq (x : ℝ) : ℝ :=
  Real.sqrt (8 * x)

theorem line_intersect_curve_area (α : ℝ) (t1 t2 : ℝ) (hα : α = Real.pi / 4) :
  let l_param (t : ℝ) := line_param_eq α t in
  let curve := curve_cart_eq in
  l_param t1 = (some_pt_x, curve some_pt_x)
  ∧ l_param t2 = (some_pt_y, curve some_pt_y) →
  ∃ area : ℝ, area = 2 * Real.sqrt 6 := 
sorry

end line_intersect_curve_area_l302_302441


namespace concrete_for_supporting_pillars_l302_302961

-- Define the given conditions
def roadway_deck_concrete : ℕ := 1600
def one_anchor_concrete : ℕ := 700
def total_bridge_concrete : ℕ := 4800

-- State the theorem
theorem concrete_for_supporting_pillars :
  let total_anchors_concrete := 2 * one_anchor_concrete in
  let total_deck_and_anchors_concrete := roadway_deck_concrete + total_anchors_concrete in
  total_bridge_concrete - total_deck_and_anchors_concrete = 1800 :=
by
  sorry

end concrete_for_supporting_pillars_l302_302961


namespace simplify_sqrt_450_l302_302100

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l302_302100


namespace parabola_directrix_l302_302138

theorem parabola_directrix (a : ℝ) :
  (∃ y : ℝ, y = ax^2 ∧ y = -2) → a = 1/8 :=
by
  -- Solution steps are omitted.
  sorry

end parabola_directrix_l302_302138


namespace sqrt_450_eq_15_sqrt_2_l302_302127

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l302_302127


namespace simplify_sqrt_450_l302_302105

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l302_302105


namespace minimize_magnitude_x_l302_302547

-- Conditions
variables (a b : ℝ)
variable (θ : ℝ)

-- Given conditions
variables (h1 : ∥a∥ = 2)
variables (h2 : ∥b∥ = 1)
variables (h3 : θ = real.pi / 3) -- 60 degrees in radian

-- The target result to prove
theorem minimize_magnitude_x (x : ℝ) (h4 : ∥a∥ = 2) (h5 : ∥b∥ = 1) (h6 : θ = real.pi / 3) : 
  (∥a - x * b∥) minimized by x = 1 := sorry

end minimize_magnitude_x_l302_302547


namespace tan_x_value_complex_trig_expression_value_l302_302361

theorem tan_x_value (x : ℝ) (h : Real.sin (x / 2) - 2 * Real.cos (x / 2) = 0) :
  Real.tan x = -4 / 3 :=
sorry

theorem complex_trig_expression_value (x : ℝ) (h : Real.sin (x / 2) - 2 * Real.cos (x / 2) = 0) :
  Real.cos (2 * x) / (Real.sqrt 2 * Real.cos (Real.pi / 4 + x) * Real.sin x) = 1 / 4 :=
sorry

end tan_x_value_complex_trig_expression_value_l302_302361


namespace positive_integers_sequence_l302_302530

theorem positive_integers_sequence (a b c d : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d) 
  (h4 : a ∣ (b + c + d)) (h5 : b ∣ (a + c + d)) 
  (h6 : c ∣ (a + b + d)) (h7 : d ∣ (a + b + c)) : 
  (a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 6) ∨ 
  (a = 1 ∧ b = 2 ∧ c = 6 ∧ d = 9) ∨ 
  (a = 1 ∧ b = 3 ∧ c = 8 ∧ d = 12) ∨ 
  (a = 1 ∧ b = 4 ∧ c = 5 ∧ d = 10) ∨ 
  (a = 1 ∧ b = 6 ∧ c = 14 ∧ d = 21) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 10 ∧ d = 15) :=
sorry

end positive_integers_sequence_l302_302530


namespace gcd_same_remainder_mod_three_l302_302847

theorem gcd_same_remainder_mod_three (a b c d e f g : ℕ) (h_distinct : list.nodup [a, b, c, d, e, f, g]) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g) :
  ∃ x y z, (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧ (gcd x y % 3 = gcd y z % 3 ∧ gcd y z % 3 = gcd z x % 3) :=
by sorry

end gcd_same_remainder_mod_three_l302_302847


namespace total_students_high_school_l302_302172

theorem total_students_high_school (students_first_grade : ℕ) (total_sample : ℕ) 
  (sample_second_grade : ℕ) (sample_third_grade : ℕ) (total_students : ℕ) 
  (h1 : students_first_grade = 600) (h2 : total_sample = 45) 
  (h3 : sample_second_grade = 20) (h4 : sample_third_grade = 10)
  (h5 : 15 = total_sample - sample_second_grade - sample_third_grade) 
  (h6 : 15 * total_students = students_first_grade * total_sample) :
  total_students = 1800 :=
sorry

end total_students_high_school_l302_302172


namespace additional_donation_l302_302320

theorem additional_donation
  (t : ℕ) (c d₁ d₂ T a : ℝ)
  (h1 : t = 25)
  (h2 : c = 2.00)
  (h3 : d₁ = 15.00) 
  (h4 : d₂ = 15.00)
  (h5 : T = 100.00)
  (h6 : t * c + d₁ + d₂ + a = T) :
  a = 20.00 :=
by
  sorry

end additional_donation_l302_302320


namespace count_consecutive_integers_l302_302995

theorem count_consecutive_integers : 
  ∃ n : ℕ, (∀ x : ℕ, (1 < x ∧ x < 111) → (x - 1) + x + (x + 1) < 333) ∧ n = 109 := 
  by
    sorry

end count_consecutive_integers_l302_302995


namespace simplify_sqrt_450_l302_302021

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l302_302021


namespace quad_root_magnitude_l302_302663

theorem quad_root_magnitude (m : ℝ) :
  (∃ x : ℝ, x^2 - x + m^2 - 4 = 0 ∧ x = 1) → m = 2 ∨ m = -2 :=
by
  sorry

end quad_root_magnitude_l302_302663


namespace sum_of_first_seven_terms_l302_302299

variable {a_n : ℕ → ℝ} {d : ℝ}

-- Define the arithmetic progression condition.
def arithmetic_progression (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a_n n = a_n 0 + n * d

-- We are given that the sequence is an arithmetic progression.
axiom sequence_is_arithmetic_progression : arithmetic_progression a_n d

-- We are also given that the sum of the 3rd, 4th, and 5th terms is 12.
axiom sum_of_terms_is_12 : a_n 2 + a_n 3 + a_n 4 = 12

-- We need to prove that the sum of the first seven terms is 28.
theorem sum_of_first_seven_terms : (a_n 0) + (a_n 1) + (a_n 2) + (a_n 3) + (a_n 4) + (a_n 5) + (a_n 6) = 28 := 
  sorry

end sum_of_first_seven_terms_l302_302299


namespace probability_perfect_square_sum_l302_302746

def is_perfect_square_sum (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

def count_perfect_square_sums : ℕ :=
  let possible_outcomes := 216
  let favorable_outcomes := 32
  favorable_outcomes

theorem probability_perfect_square_sum :
  (count_perfect_square_sums : ℚ) / 216 = 4 / 27 :=
by
  sorry

end probability_perfect_square_sum_l302_302746


namespace simplify_sqrt_450_l302_302087

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l302_302087


namespace ursula_initial_money_l302_302153

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

end ursula_initial_money_l302_302153


namespace atomic_number_cannot_be_x_plus_4_l302_302345

-- Definitions for atomic numbers and elements in the same main group
def in_same_main_group (A B : Type) (atomic_num_A atomic_num_B : ℕ) : Prop :=
  atomic_num_B ≠ atomic_num_A + 4

-- Noncomputable definition is likely needed as the problem involves non-algorithmic aspects.
noncomputable def periodic_table_condition (A B : Type) (x : ℕ) : Prop :=
  in_same_main_group A B x (x + 4)

-- Main theorem stating the mathematical proof problem
theorem atomic_number_cannot_be_x_plus_4
  (A B : Type)
  (x : ℕ)
  (h : periodic_table_condition A B x) : false :=
  by
    sorry

end atomic_number_cannot_be_x_plus_4_l302_302345


namespace John_took_more_chickens_than_Ray_l302_302609

theorem John_took_more_chickens_than_Ray
  (r m j : ℕ)
  (h1 : r = 10)
  (h2 : r = m - 6)
  (h3 : j = m + 5) : j - r = 11 :=
by
  sorry

end John_took_more_chickens_than_Ray_l302_302609


namespace aunt_money_calculation_l302_302335

variable (total_money_received aunt_money : ℕ)
variable (bank_amount grandfather_money : ℕ := 150)

theorem aunt_money_calculation (h1 : bank_amount = 45) (h2 : bank_amount = total_money_received / 5) (h3 : total_money_received = aunt_money + grandfather_money) :
  aunt_money = 75 :=
by
  -- The proof is captured in these statements:
  sorry

end aunt_money_calculation_l302_302335


namespace determine_f4_l302_302378

def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem determine_f4 (f : ℝ → ℝ) (h_odd : odd_function f) (h_f_neg : ∀ x, x < 0 → f x = x * (2 - x)) : f 4 = 24 :=
by
  sorry

end determine_f4_l302_302378


namespace sqrt_450_simplified_l302_302039

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l302_302039


namespace jerome_gave_to_meg_l302_302705

theorem jerome_gave_to_meg (init_money half_money given_away meg bianca : ℝ) 
    (h1 : half_money = 43) 
    (h2 : init_money = 2 * half_money) 
    (h3 : 54 = init_money - given_away)
    (h4 : given_away = meg + bianca)
    (h5 : bianca = 3 * meg) : 
    meg = 8 :=
by
  sorry

end jerome_gave_to_meg_l302_302705


namespace simplify_sqrt_450_l302_302088

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l302_302088


namespace factorize_expression_l302_302908

theorem factorize_expression (x y : ℝ) : 
  x * y^2 - 6 * x * y + 9 * x = x * (y - 3)^2 := 
by sorry

end factorize_expression_l302_302908


namespace min_value_proof_l302_302681

noncomputable def minimum_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (3/2) * a + b = 1) : ℝ :=
  (3 / a) + (2 / b)

theorem min_value_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (3/2) * a + b = 1) : minimum_value a b h1 h2 h3 = 25 / 2 :=
sorry

end min_value_proof_l302_302681


namespace simplify_sqrt_450_l302_302124

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l302_302124


namespace range_of_a_l302_302379

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ (a < -3 ∨ a > 1) :=
by
  sorry

end range_of_a_l302_302379


namespace inequality_solution_set_l302_302356

theorem inequality_solution_set : 
  {x : ℝ | -x^2 + 4*x + 5 < 0} = {x : ℝ | x < -1 ∨ x > 5} := 
by
  sorry

end inequality_solution_set_l302_302356


namespace word_count_in_language_l302_302696

theorem word_count_in_language :
  let vowels := 3
  let consonants := 5
  let num_syllables := (vowels * consonants) + (consonants * vowels)
  let num_words := num_syllables * num_syllables
  num_words = 900 :=
by
  let vowels := 3
  let consonants := 5
  let num_syllables := (vowels * consonants) + (consonants * vowels)
  let num_words := num_syllables * num_syllables
  have : num_words = 900 := sorry
  exact this

end word_count_in_language_l302_302696


namespace probability_two_defective_after_two_tests_l302_302225

variable {α : Type*}

-- Assume a finite set of components
def components : Finset α := {1, 2, 3, 4, 5, 6}

-- Assume 2 defective components
def defective : Finset α := {1, 2}

-- Assume 4 qualified components
def qualified : Finset α := {3, 4, 5, 6}

-- Define a probability measure space
noncomputable def prob_space (s : Finset α) := 
  MeasureTheory.MeasureSpace (λ _, 1.0 / s.card)

-- Probability of finding exactly the 2 defective components after 2 tests without replacement
theorem probability_two_defective_after_two_tests :
  @prob_space α components (defective.card = 2 ∧
                            qualified.card = 4 ∧
                            (defective ∪ qualified) = components) ->
  probability (finset.pair_combinations components 2) (λ s, s = defective) = 1/15 := by sorry

end probability_two_defective_after_two_tests_l302_302225


namespace balloons_floated_away_l302_302527

theorem balloons_floated_away (starting_balloons given_away grabbed_balloons final_balloons flattened_balloons : ℕ)
  (h1 : starting_balloons = 50)
  (h2 : given_away = 10)
  (h3 : grabbed_balloons = 11)
  (h4 : final_balloons = 39)
  : flattened_balloons = starting_balloons - given_away + grabbed_balloons - final_balloons → flattened_balloons = 12 :=
by
  sorry

end balloons_floated_away_l302_302527


namespace base_of_second_term_l302_302997

theorem base_of_second_term (e : ℕ) (base : ℝ) 
  (h1 : e = 35) 
  (h2 : (1/5)^e * base^18 = 1 / (2 * (10)^35)) : 
  base = 1/4 :=
by
  sorry

end base_of_second_term_l302_302997


namespace calculate_cherry_pies_l302_302004

-- Definitions for the conditions
def total_pies : ℕ := 40
def ratio_parts_apple : ℕ := 2
def ratio_parts_blueberry : ℕ := 5
def ratio_parts_cherry : ℕ := 3
def total_ratio_parts := ratio_parts_apple + ratio_parts_blueberry + ratio_parts_cherry

-- Calculating the number of pies per part and then the number of cherry pies
def pies_per_part : ℕ := total_pies / total_ratio_parts
def cherry_pies : ℕ := ratio_parts_cherry * pies_per_part

-- Proof statement
theorem calculate_cherry_pies : cherry_pies = 12 :=
by
  -- Lean proof goes here
  sorry

end calculate_cherry_pies_l302_302004


namespace range_of_a_l302_302694

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬ x^2 + (a - 1) * x + 1 < 0) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l302_302694


namespace first_six_divisors_l302_302732

theorem first_six_divisors (a b : ℤ) (h : 5 * b = 14 - 3 * a) : 
  ∃ n, n = 5 ∧ ∀ k ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ), (3 * b + 18) % k = 0 ↔ k ∈ ({1, 2, 3, 5, 6} : Finset ℕ) :=
by
  sorry

end first_six_divisors_l302_302732
