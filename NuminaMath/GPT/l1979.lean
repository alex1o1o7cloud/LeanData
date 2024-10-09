import Mathlib

namespace sum_of_eight_numbers_l1979_197901

theorem sum_of_eight_numbers (nums : List ℝ) (h_len : nums.length = 8) (h_avg : (nums.sum / 8) = 5.5) : nums.sum = 44 :=
by
  sorry

end sum_of_eight_numbers_l1979_197901


namespace remainder_444_power_444_mod_13_l1979_197994

theorem remainder_444_power_444_mod_13 : (444 ^ 444) % 13 = 1 := by
  sorry

end remainder_444_power_444_mod_13_l1979_197994


namespace parabola_y_values_order_l1979_197952

theorem parabola_y_values_order :
  let y1 := 2 * (-3 - 2) ^ 2 + 1
  let y2 := 2 * (3 - 2) ^ 2 + 1
  let y3 := 2 * (4 - 2) ^ 2 + 1
  y2 < y3 ∧ y3 < y1 :=
by
  let y1 := 2 * (-3 - 2) ^ 2 + 1
  let y2 := 2 * (3 - 2) ^ 2 + 1
  let y3 := 2 * (4 - 2) ^ 2 + 1
  -- The proof is omitted
  sorry

end parabola_y_values_order_l1979_197952


namespace fraction_never_simplifiable_l1979_197976

theorem fraction_never_simplifiable (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end fraction_never_simplifiable_l1979_197976


namespace arithmetic_seq_a3_a9_zero_l1979_197960

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_11_zero (a : ℕ → ℝ) : Prop :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 = 0

theorem arithmetic_seq_a3_a9_zero (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : sum_first_11_zero a) :
  a 3 + a 9 = 0 :=
sorry

end arithmetic_seq_a3_a9_zero_l1979_197960


namespace sticks_problem_solution_l1979_197964

theorem sticks_problem_solution :
  ∃ n : ℕ, n > 0 ∧ 1012 = 2 * n * (n + 1) ∧ 1012 > 1000 ∧ 
           1012 % 3 = 1 ∧ 1012 % 5 = 2 :=
by
  sorry

end sticks_problem_solution_l1979_197964


namespace socks_difference_l1979_197923

-- Definitions of the conditions
def week1 : ℕ := 12
def week2 (S : ℕ) : ℕ := S
def week3 (S : ℕ) : ℕ := (12 + S) / 2
def week4 (S : ℕ) : ℕ := (12 + S) / 2 - 3
def total (S : ℕ) : ℕ := week1 + week2 S + week3 S + week4 S

-- Statement of the theorem
theorem socks_difference (S : ℕ) (h : total S = 57) : S - week1 = 1 :=
by 
  -- Proof is not required
  sorry

end socks_difference_l1979_197923


namespace train_B_departure_time_l1979_197978

def distance : ℕ := 65
def speed_A : ℕ := 20
def speed_B : ℕ := 25
def departure_A := 7
def meeting_time := 9

theorem train_B_departure_time : ∀ (d : ℕ) (vA : ℕ) (vB : ℕ) (tA : ℕ) (m : ℕ), 
  d = 65 → vA = 20 → vB = 25 → tA = 7 → m = 9 → ((9 - (m - tA + (d - (2 * vA)) / vB)) = 1) → 
  8 = ((9 - (meeting_time - departure_A + (distance - (2 * speed_A)) / speed_B))) := 
  by {
    sorry
  }

end train_B_departure_time_l1979_197978


namespace tomatoes_first_shipment_l1979_197916

theorem tomatoes_first_shipment :
  ∃ X : ℕ, 
    (∀Y : ℕ, 
      (Y = 300) → -- Saturday sale
      (X - Y = X - 300) ∧
      (∀Z : ℕ, 
        (Z = 200) → -- Sunday rotting
        (X - 300 - Z = X - 500) ∧
        (∀W : ℕ, 
          (W = 2 * X) → -- Monday new shipment
          (X - 500 + W = 2500) →
          (X = 1000)
        )
      )
    ) :=
by
  sorry

end tomatoes_first_shipment_l1979_197916


namespace total_time_for_12000_dolls_l1979_197937

noncomputable def total_combined_machine_operation_time (num_dolls : ℕ) (shoes_per_doll bags_per_doll cosmetics_per_doll hats_per_doll : ℕ) (time_per_doll time_per_accessory : ℕ) : ℕ :=
  let total_accessories_per_doll := shoes_per_doll + bags_per_doll + cosmetics_per_doll + hats_per_doll
  let total_accessories := num_dolls * total_accessories_per_doll
  let time_for_dolls := num_dolls * time_per_doll
  let time_for_accessories := total_accessories * time_per_accessory
  time_for_dolls + time_for_accessories

theorem total_time_for_12000_dolls (h1 : ∀ (x : ℕ), x = 12000) (h2 : ∀ (x : ℕ), x = 2) (h3 : ∀ (x : ℕ), x = 3) (h4 : ∀ (x : ℕ), x = 1) (h5 : ∀ (x : ℕ), x = 5) (h6 : ∀ (x : ℕ), x = 45) (h7 : ∀ (x : ℕ), x = 10) :
  total_combined_machine_operation_time 12000 2 3 1 5 45 10 = 1860000 := by 
  sorry

end total_time_for_12000_dolls_l1979_197937


namespace clients_number_l1979_197924

theorem clients_number (C : ℕ) (total_cars : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ)
  (h1 : total_cars = 12)
  (h2 : cars_per_client = 4)
  (h3 : selections_per_car = 3)
  (h4 : C * cars_per_client = total_cars * selections_per_car) : C = 9 :=
by sorry

end clients_number_l1979_197924


namespace class_groups_l1979_197988

open Nat

def combinations (n k : ℕ) : ℕ :=
  n.choose k

theorem class_groups (boys girls : ℕ) (group_size : ℕ) :
  boys = 9 → girls = 12 → group_size = 3 →
  (combinations boys 1 * combinations girls 2) + (combinations boys 2 * combinations girls 1) = 1026 :=
by
  intros
  sorry

end class_groups_l1979_197988


namespace min_squares_to_cover_5x5_l1979_197997

theorem min_squares_to_cover_5x5 : 
  (∀ (cover : ℕ → ℕ), (cover 1 + cover 2 + cover 3 + cover 4) * (1^2 + 2^2 + 3^2 + 4^2) = 25 → 
  cover 1 + cover 2 + cover 3 + cover 4 = 10) :=
sorry

end min_squares_to_cover_5x5_l1979_197997


namespace time_to_cover_escalator_l1979_197903

noncomputable def average_speed (initial_speed final_speed : ℝ) : ℝ :=
  (initial_speed + final_speed) / 2

noncomputable def combined_speed (escalator_speed person_average_speed : ℝ) : ℝ :=
  escalator_speed + person_average_speed

noncomputable def coverage_time (length combined_speed : ℝ) : ℝ :=
  length / combined_speed

theorem time_to_cover_escalator
  (escalator_speed : ℝ := 20)
  (length : ℝ := 300)
  (initial_person_speed : ℝ := 3)
  (final_person_speed : ℝ := 5) :
  coverage_time length (combined_speed escalator_speed (average_speed initial_person_speed final_person_speed)) = 12.5 :=
by
  sorry

end time_to_cover_escalator_l1979_197903


namespace tan_15_degree_identity_l1979_197974

theorem tan_15_degree_identity : (1 + Real.tan (15 * Real.pi / 180)) / (1 - Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 :=
by sorry

end tan_15_degree_identity_l1979_197974


namespace difference_of_distances_l1979_197927

-- Definition of John's walking distance to school
def John_distance : ℝ := 0.7

-- Definition of Nina's walking distance to school
def Nina_distance : ℝ := 0.4

-- Assertion that the difference in walking distance is 0.3 miles
theorem difference_of_distances : (John_distance - Nina_distance) = 0.3 := 
by 
  sorry

end difference_of_distances_l1979_197927


namespace find_monotonic_bijections_l1979_197915

variable {f : ℝ → ℝ}

-- Define the properties of the function f
def bijective (f : ℝ → ℝ) : Prop :=
  Function.Bijective f

def condition (f : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, f t + f (f t) = 2 * t

theorem find_monotonic_bijections (f : ℝ → ℝ) (hf_bij : bijective f) (hf_cond : condition f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
sorry

end find_monotonic_bijections_l1979_197915


namespace compound_interest_correct_l1979_197919

noncomputable def compoundInterest (P: ℝ) (r: ℝ) (n: ℝ) (t: ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem compound_interest_correct :
  compoundInterest 5000 0.04 1 3 - 5000 = 624.32 :=
by
  sorry

end compound_interest_correct_l1979_197919


namespace range_of_a_l1979_197921

open Set

theorem range_of_a (a : ℝ) :
  (M : Set ℝ) = { x | -1 ≤ x ∧ x ≤ 2 } →
  (N : Set ℝ) = { x | 1 - 3 * a < x ∧ x ≤ 2 * a } →
  M ∩ N = M →
  1 ≤ a :=
by
  intro hM hN h_inter
  sorry

end range_of_a_l1979_197921


namespace eval_infinite_series_eq_4_l1979_197939

open BigOperators

noncomputable def infinite_series_sum : ℝ :=
  ∑' k, (k^2) / (3^k)

theorem eval_infinite_series_eq_4 : infinite_series_sum = 4 := 
  sorry

end eval_infinite_series_eq_4_l1979_197939


namespace greater_number_l1979_197929

theorem greater_number (a b : ℕ) (h1 : a + b = 36) (h2 : a - b = 8) : a = 22 :=
by
  sorry

end greater_number_l1979_197929


namespace factorization_correct_l1979_197928

-- Defining the expressions
def expr1 (x : ℝ) : ℝ := 4 * x^2 + 4 * x
def expr2 (x : ℝ) : ℝ := 4 * x * (x + 1)

-- Theorem statement: Prove that expr1 and expr2 are equivalent
theorem factorization_correct (x : ℝ) : expr1 x = expr2 x :=
by 
  sorry

end factorization_correct_l1979_197928


namespace production_cost_percentage_l1979_197989

theorem production_cost_percentage
    (initial_cost final_cost : ℝ)
    (final_cost_eq : final_cost = 48)
    (initial_cost_eq : initial_cost = 50)
    (h : (initial_cost + 0.5 * x) * (1 - x / 100) = final_cost) :
    x = 20 :=
by
  sorry

end production_cost_percentage_l1979_197989


namespace percentage_of_number_l1979_197932

/-- 
  Given a certain percentage \( P \) of 600 is 90.
  If 30% of 50% of a number 4000 is 90,
  Then P equals to 15%.
-/
theorem percentage_of_number (P : ℝ) (h1 : (0.30 : ℝ) * (0.50 : ℝ) * 4000 = 600) (h2 : P * 600 = 90) :
  P = 0.15 :=
  sorry

end percentage_of_number_l1979_197932


namespace harmony_numbers_with_first_digit_2_count_l1979_197926

def is_harmony_number (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  (1000 ≤ n ∧ n < 10000) ∧ (a + b + c + d = 6)

noncomputable def count_harmony_numbers_with_first_digit_2 : ℕ :=
  Nat.card { n : ℕ // is_harmony_number n ∧ n / 1000 = 2 }

theorem harmony_numbers_with_first_digit_2_count :
  count_harmony_numbers_with_first_digit_2 = 15 :=
sorry

end harmony_numbers_with_first_digit_2_count_l1979_197926


namespace remaining_fish_l1979_197906

theorem remaining_fish (initial_fish : ℝ) (moved_fish : ℝ) (remaining_fish : ℝ) : initial_fish = 212.0 → moved_fish = 68.0 → remaining_fish = 144.0 → initial_fish - moved_fish = remaining_fish := by sorry

end remaining_fish_l1979_197906


namespace sum_of_first_three_terms_is_zero_l1979_197946

variable (a d : ℤ) 

-- Definitions from the conditions
def a₄ := a + 3 * d
def a₅ := a + 4 * d
def a₆ := a + 5 * d

-- Theorem statement
theorem sum_of_first_three_terms_is_zero 
  (h₁ : a₄ = 8) 
  (h₂ : a₅ = 12) 
  (h₃ : a₆ = 16) : 
  a + (a + d) + (a + 2 * d) = 0 := 
by 
  sorry

end sum_of_first_three_terms_is_zero_l1979_197946


namespace largest_band_members_l1979_197957

theorem largest_band_members 
  (r x m : ℕ) 
  (h1 : (r * x + 3 = m)) 
  (h2 : ((r - 3) * (x + 1) = m))
  (h3 : m < 100) : 
  m = 75 :=
sorry

end largest_band_members_l1979_197957


namespace cyclist_speed_l1979_197981

theorem cyclist_speed:
  ∀ (c : ℝ), 
  ∀ (hiker_speed : ℝ), 
  (hiker_speed = 4) → 
  (4 * (5 / 60) + 4 * (25 / 60) = c * (5 / 60)) → 
  c = 24 := 
by
  intros c hiker_speed hiker_speed_def distance_eq
  sorry

end cyclist_speed_l1979_197981


namespace find_s_l1979_197998

theorem find_s (s : ℝ) :
  let P := (s - 3, 2)
  let Q := (1, s + 2)
  let M := ((s - 2) / 2, (s + 4) / 2)
  let dist_sq := (M.1 - P.1) ^ 2 + (M.2 - P.2) ^ 2
  dist_sq = 3 * s^2 / 4 →
  s = -5 + 5 * Real.sqrt 2 ∨ s = -5 - 5 * Real.sqrt 2 :=
by
  intros P Q M dist_sq h
  sorry

end find_s_l1979_197998


namespace number_division_l1979_197995

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end number_division_l1979_197995


namespace pen_sales_average_l1979_197933

theorem pen_sales_average :
  ∃ d : ℕ, (48 = (96 + 44 * d) / (d + 1)) → d = 12 :=
by
  sorry

end pen_sales_average_l1979_197933


namespace evaluate_expression_l1979_197909

theorem evaluate_expression : (20 + 22) / 2 = 21 := by
  sorry

end evaluate_expression_l1979_197909


namespace triple_f_of_3_l1979_197969

def f (x : ℤ) : ℤ := -3 * x + 5

theorem triple_f_of_3 : f (f (f 3)) = -46 := by
  sorry

end triple_f_of_3_l1979_197969


namespace perimeter_of_monster_is_correct_l1979_197955

/-
  The problem is to prove that the perimeter of a shaded sector of a circle
  with radius 2 cm and a central angle of 120 degrees (where the mouth is a chord)
  is equal to (8 * π / 3 + 2 * sqrt 3) cm.
-/

noncomputable def perimeter_of_monster (r : ℝ) (theta_deg : ℝ) : ℝ :=
  let theta_rad := theta_deg * Real.pi / 180
  let chord_length := 2 * r * Real.sin (theta_rad / 2)
  let arc_length := (2 * (2 * Real.pi) * (240 / 360))
  arc_length + chord_length

theorem perimeter_of_monster_is_correct : perimeter_of_monster 2 120 = (8 * Real.pi / 3 + 2 * Real.sqrt 3) :=
by
  sorry

end perimeter_of_monster_is_correct_l1979_197955


namespace find_mangoes_l1979_197962

def cost_of_grapes : ℕ := 8 * 70
def total_amount_paid : ℕ := 1165
def cost_per_kg_of_mangoes : ℕ := 55

theorem find_mangoes (m : ℕ) : cost_of_grapes + m * cost_per_kg_of_mangoes = total_amount_paid → m = 11 :=
by
  sorry

end find_mangoes_l1979_197962


namespace a_lt_c_lt_b_l1979_197917

noncomputable def a : ℝ := Real.sin (14 * Real.pi / 180) + Real.cos (14 * Real.pi / 180)
noncomputable def b : ℝ := 2 * Real.sqrt 2 * Real.sin (30.5 * Real.pi / 180) * Real.cos (30.5 * Real.pi / 180)
noncomputable def c : ℝ := Real.sqrt 6 / 2

theorem a_lt_c_lt_b : a < c ∧ c < b := by
  sorry

end a_lt_c_lt_b_l1979_197917


namespace no_ordered_triples_l1979_197940

theorem no_ordered_triples (x y z : ℕ)
  (h1 : 1 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) :
  x * y * z + 2 * (x * y + y * z + z * x) ≠ 2 * (2 * (x * y + y * z + z * x)) + 12 :=
by {
  sorry
}

end no_ordered_triples_l1979_197940


namespace amn_div_l1979_197938

theorem amn_div (a m n : ℕ) (a_pos : a > 1) (h : a > 1 ∧ (a^m + 1) ∣ (a^n + 1)) : m ∣ n :=
by sorry

end amn_div_l1979_197938


namespace pauls_weekly_spending_l1979_197991

def mowing_lawns : ℕ := 3
def weed_eating : ℕ := 3
def total_weeks : ℕ := 2
def total_money : ℕ := mowing_lawns + weed_eating
def spending_per_week : ℕ := total_money / total_weeks

theorem pauls_weekly_spending : spending_per_week = 3 := by
  sorry

end pauls_weekly_spending_l1979_197991


namespace trig_proof_l1979_197943

theorem trig_proof (α : ℝ) (h1 : Real.tan α = Real.sqrt 3) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.cos (2 * α) - Real.sin (π / 2 + α) = 0 :=
sorry

end trig_proof_l1979_197943


namespace tangent_line_equation_at_1_2_l1979_197902

noncomputable def f (x : ℝ) : ℝ := 2 / x

theorem tangent_line_equation_at_1_2 :
  let x₀ := 1
  let y₀ := 2
  let slope := -2
  ∀ (x y : ℝ),
    y - y₀ = slope * (x - x₀) →
    2 * x + y - 4 = 0 :=
by
  sorry

end tangent_line_equation_at_1_2_l1979_197902


namespace totalPoundsOfFoodConsumed_l1979_197925

def maxConsumptionPerGuest : ℝ := 2.5
def minNumberOfGuests : ℕ := 165

theorem totalPoundsOfFoodConsumed : 
    maxConsumptionPerGuest * (minNumberOfGuests : ℝ) = 412.5 := by
  sorry

end totalPoundsOfFoodConsumed_l1979_197925


namespace positive_difference_sum_of_squares_l1979_197972

-- Given definitions
def sum_of_squares_even (n : ℕ) : ℕ :=
  4 * (n * (n + 1) * (2 * n + 1)) / 6

def sum_of_squares_odd (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

-- The explicit values for this problem
def sum_of_squares_first_25_even := sum_of_squares_even 25
def sum_of_squares_first_20_odd := sum_of_squares_odd 20

-- The required proof statement
theorem positive_difference_sum_of_squares : 
  (sum_of_squares_first_25_even - sum_of_squares_first_20_odd) = 19230 := by
  sorry

end positive_difference_sum_of_squares_l1979_197972


namespace Isabella_redeem_day_l1979_197958

def is_coupon_day_closed_sunday (start_day : ℕ) (num_coupons : ℕ) (cycle_days : ℕ) : Prop :=
  ∃ n, n < num_coupons ∧ (start_day + n * cycle_days) % 7 = 0

theorem Isabella_redeem_day: 
  ∀ (day : ℕ), day ≡ 1 [MOD 7]
  → ¬ is_coupon_day_closed_sunday day 6 11 :=
by
  intro day h_mod
  simp [is_coupon_day_closed_sunday]
  sorry

end Isabella_redeem_day_l1979_197958


namespace tangent_line_at_one_l1979_197904

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem tangent_line_at_one : ∀ x y, (x = 1 ∧ y = 0) → (x - y - 1 = 0) :=
by 
  intro x y h
  sorry

end tangent_line_at_one_l1979_197904


namespace total_charge_for_3_hours_l1979_197979

namespace TherapyCharges

-- Conditions
variables (A F : ℝ)
variable (h1 : F = A + 20)
variable (h2 : F + 4 * A = 300)

-- Prove that the total charge for 3 hours of therapy is 188
theorem total_charge_for_3_hours : F + 2 * A = 188 :=
by
  sorry

end TherapyCharges

end total_charge_for_3_hours_l1979_197979


namespace bahs_equivalent_to_1500_yahs_l1979_197935

-- Definitions from conditions
def bahs := ℕ
def rahs := ℕ
def yahs := ℕ

-- Conversion ratios given in conditions
def ratio_bah_rah : ℚ := 10 / 16
def ratio_rah_yah : ℚ := 9 / 15

-- Given the conditions
def condition1 (b r : ℚ) : Prop := b / r = ratio_bah_rah
def condition2 (r y : ℚ) : Prop := r / y = ratio_rah_yah

-- Goal: proving the question
theorem bahs_equivalent_to_1500_yahs (b : ℚ) (r : ℚ) (y : ℚ)
  (h1 : condition1 b r) (h2 : condition2 r y) : b * (1500 / y) = 562.5
:=
sorry

end bahs_equivalent_to_1500_yahs_l1979_197935


namespace base_not_divisible_by_5_l1979_197918

def is_not_divisible_by_5 (c : ℤ) : Prop :=
  ¬(∃ k : ℤ, c = 5 * k)

def check_not_divisible_by_5 (b : ℤ) : Prop :=
  is_not_divisible_by_5 (3 * b^3 - 3 * b^2 - b)

theorem base_not_divisible_by_5 :
  check_not_divisible_by_5 6 ∧ check_not_divisible_by_5 8 :=
by 
  sorry

end base_not_divisible_by_5_l1979_197918


namespace purely_imaginary_x_value_l1979_197912

theorem purely_imaginary_x_value (x : ℝ) (h1 : x^2 - 1 = 0) (h2 : x + 1 ≠ 0) : x = 1 :=
by
  sorry

end purely_imaginary_x_value_l1979_197912


namespace factor_expression_l1979_197992

theorem factor_expression (b : ℤ) : 53 * b^2 + 159 * b = 53 * b * (b + 3) :=
by
  sorry

end factor_expression_l1979_197992


namespace seating_arrangements_l1979_197922

theorem seating_arrangements :
  let total_arrangements := Nat.factorial 8
  let jwp_together := (Nat.factorial 6) * (Nat.factorial 3)
  total_arrangements - jwp_together = 36000 := by
  sorry

end seating_arrangements_l1979_197922


namespace problem_statement_l1979_197914

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement (h1 : ∀ ⦃x y⦄, x > 4 → y > x → f y < f x)
                          (h2 : ∀ x, f (4 + x) = f (4 - x)) : f 3 > f 6 :=
by 
  sorry

end problem_statement_l1979_197914


namespace part1_part2_l1979_197985

-- Define the conditions for part (1)
def nonEmptyBoxes := ∀ i j k: Nat, (i ≠ j ∧ i ≠ k ∧ j ≠ k)
def ball3inBoxB := ∀ (b3: Nat) (B: Nat), b3 = 3 ∧ B > 0

-- Define the conditions for part (2)
def ball1notInBoxA := ∀ (b1: Nat) (A: Nat), b1 ≠ 1 ∧ A > 0
def ball2notInBoxB := ∀ (b2: Nat) (B: Nat), b2 ≠ 2 ∧ B > 0

-- Theorems to be proved
theorem part1 (h1: nonEmptyBoxes) (h2: ball3inBoxB) : ∃ n, n = 12 := by sorry

theorem part2 (h3: ball1notInBoxA) (h4: ball2notInBoxB) : ∃ n, n = 36 := by sorry

end part1_part2_l1979_197985


namespace greatest_integer_less_than_M_over_100_l1979_197900

theorem greatest_integer_less_than_M_over_100 :
  (1 / (Nat.factorial 3 * Nat.factorial 16) +
   1 / (Nat.factorial 4 * Nat.factorial 15) +
   1 / (Nat.factorial 5 * Nat.factorial 14) +
   1 / (Nat.factorial 6 * Nat.factorial 13) +
   1 / (Nat.factorial 7 * Nat.factorial 12) +
   1 / (Nat.factorial 8 * Nat.factorial 11) +
   1 / (Nat.factorial 9 * Nat.factorial 10) = M / (Nat.factorial 2 * Nat.factorial 17)) →
  (⌊(M : ℚ) / 100⌋ = 27) := 
sorry

end greatest_integer_less_than_M_over_100_l1979_197900


namespace inverse_proportion_l1979_197987

theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x = 3) (h3 : y = 15) (h4 : y = -30) : x = -3 / 2 :=
by
  sorry

end inverse_proportion_l1979_197987


namespace explicit_expression_l1979_197950

variable {α : Type*} [LinearOrder α] {f : α → α}

/-- Given that the function satisfies a specific condition, prove the function's explicit expression. -/
theorem explicit_expression (f : ℝ → ℝ) (h : ∀ x, f (3 * x + 2) = 9 * x + 8) : 
  ∀ x, f x = 3 * x + 2 :=
by
  sorry

end explicit_expression_l1979_197950


namespace statement_3_correct_l1979_197936

-- Definitions based on the conditions
def DeductiveReasoningGeneralToSpecific := True
def SyllogismForm := True
def ConclusionDependsOnPremisesAndForm := True

-- Proof problem statement
theorem statement_3_correct : SyllogismForm := by
  exact True.intro

end statement_3_correct_l1979_197936


namespace simplify_expression_l1979_197986

variable (a b c d x : ℝ)
variable (hab : a ≠ b)
variable (hac : a ≠ c)
variable (had : a ≠ d)
variable (hbc : b ≠ c)
variable (hbd : b ≠ d)
variable (hcd : c ≠ d)

theorem simplify_expression :
  ( ( (x + a)^4 / ((a - b)*(a - c)*(a - d)) )
  + ( (x + b)^4 / ((b - a)*(b - c)*(b - d)) )
  + ( (x + c)^4 / ((c - a)*(c - b)*(c - d)) )
  + ( (x + d)^4 / ((d - a)*(d - b)*(d - c)) ) = a + b + c + d + 4*x ) :=
  sorry

end simplify_expression_l1979_197986


namespace quadratic_common_root_distinct_real_numbers_l1979_197990

theorem quadratic_common_root_distinct_real_numbers:
  ∀ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ a →
  (∃ x, x^2 + a * x + b = 0 ∧ x^2 + c * x + a = 0) ∧
  (∃ y, y^2 + a * y + b = 0 ∧ y^2 + b * y + c = 0) ∧
  (∃ z, z^2 + b * z + c = 0 ∧ z^2 + c * z + a = 0) →
  a^2 + b^2 + c^2 = 6 :=
by
  intros a b c h_distinct h_common_root
  sorry

end quadratic_common_root_distinct_real_numbers_l1979_197990


namespace find_value_of_a_l1979_197947

theorem find_value_of_a (a : ℝ) (h : a^3 = 21 * 25 * 315 * 7) : a = 105 := by
  sorry

end find_value_of_a_l1979_197947


namespace total_goals_is_15_l1979_197956

-- Define the conditions as variables
def KickersFirstPeriodGoals : ℕ := 2
def KickersSecondPeriodGoals : ℕ := 2 * KickersFirstPeriodGoals
def SpidersFirstPeriodGoals : ℕ := KickersFirstPeriodGoals / 2
def SpidersSecondPeriodGoals : ℕ := 2 * KickersSecondPeriodGoals

-- Define total goals by each team
def TotalKickersGoals : ℕ := KickersFirstPeriodGoals + KickersSecondPeriodGoals
def TotalSpidersGoals : ℕ := SpidersFirstPeriodGoals + SpidersSecondPeriodGoals

-- Define total goals by both teams
def TotalGoals : ℕ := TotalKickersGoals + TotalSpidersGoals

-- Prove the statement
theorem total_goals_is_15 : TotalGoals = 15 :=
by
  sorry

end total_goals_is_15_l1979_197956


namespace total_waiting_time_l1979_197944

def t1 : ℕ := 20
def t2 : ℕ := 4 * t1 + 14
def T : ℕ := t1 + t2

theorem total_waiting_time : T = 114 :=
by {
  -- Preliminary calculations and justification would go here
  sorry
}

end total_waiting_time_l1979_197944


namespace ed_money_left_l1979_197954

theorem ed_money_left
  (cost_per_hour_night : ℝ := 1.5)
  (cost_per_hour_morning : ℝ := 2)
  (initial_money : ℝ := 80)
  (hours_night : ℝ := 6)
  (hours_morning : ℝ := 4) :
  initial_money - (cost_per_hour_night * hours_night + cost_per_hour_morning * hours_morning) = 63 := 
  by
  sorry

end ed_money_left_l1979_197954


namespace inequality_abc_l1979_197967

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) := 
by 
  sorry

end inequality_abc_l1979_197967


namespace selection_methods_count_l1979_197971

theorem selection_methods_count
  (multiple_choice_questions : ℕ)
  (fill_in_the_blank_questions : ℕ)
  (h1 : multiple_choice_questions = 9)
  (h2 : fill_in_the_blank_questions = 3) :
  multiple_choice_questions + fill_in_the_blank_questions = 12 := by
  sorry

end selection_methods_count_l1979_197971


namespace ounces_per_cup_l1979_197910

theorem ounces_per_cup (total_ounces : ℕ) (total_cups : ℕ) 
  (h : total_ounces = 264 ∧ total_cups = 33) : total_ounces / total_cups = 8 :=
by
  sorry

end ounces_per_cup_l1979_197910


namespace total_distance_correct_l1979_197963

def day1_distance : ℕ := (5 * 4) + (3 * 2) + (4 * 3)
def day2_distance : ℕ := (6 * 3) + (2 * 1) + (6 * 3) + (3 * 4)
def day3_distance : ℕ := (4 * 2) + (2 * 1) + (7 * 3) + (5 * 2)

def total_distance : ℕ := day1_distance + day2_distance + day3_distance

theorem total_distance_correct :
  total_distance = 129 := by
  sorry

end total_distance_correct_l1979_197963


namespace find_number_l1979_197905

theorem find_number (x : ℚ) (h : x / 5 = 3 * (x / 6) - 40) : x = 400 / 3 :=
sorry

end find_number_l1979_197905


namespace union_of_A_and_B_l1979_197966

def A : Set ℤ := {-1, 0, 2}
def B : Set ℤ := {-1, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} :=
by
  sorry

end union_of_A_and_B_l1979_197966


namespace inf_solutions_integers_l1979_197965

theorem inf_solutions_integers (x y z : ℕ) : ∃ (n : ℕ), ∀ n > 0, (x = 2^(32 + 72 * n)) ∧ (y = 2^(28 + 63 * n)) ∧ (z = 2^(25 + 56 * n)) → x^7 + y^8 = z^9 :=
by {
  sorry
}

end inf_solutions_integers_l1979_197965


namespace isosceles_triangle_angles_l1979_197930

theorem isosceles_triangle_angles (y : ℝ) (h : y > 0) :
  let P := y
  let R := 5 * y
  let Q := R
  P + Q + R = 180 → Q = 81.82 :=
by
  sorry

end isosceles_triangle_angles_l1979_197930


namespace sum_of_ages_five_years_ago_l1979_197945

-- Definitions from the conditions
variables (A B : ℕ) -- Angela's current age and Beth's current age

-- Conditions
def angela_is_four_times_as_old_as_beth := A = 4 * B
def angela_will_be_44_in_five_years := A + 5 = 44

-- Theorem statement to prove the sum of their ages five years ago
theorem sum_of_ages_five_years_ago (h1 : angela_is_four_times_as_old_as_beth A B) (h2 : angela_will_be_44_in_five_years A) : 
  (A - 5) + (B - 5) = 39 :=
by sorry

end sum_of_ages_five_years_ago_l1979_197945


namespace primes_in_arithmetic_sequence_have_specific_ones_digit_l1979_197983

-- Define the properties of the primes and the arithmetic sequence
theorem primes_in_arithmetic_sequence_have_specific_ones_digit
  (p q r s : ℕ) 
  (prime_p : Nat.Prime p)
  (prime_q : Nat.Prime q)
  (prime_r : Nat.Prime r)
  (prime_s : Nat.Prime s)
  (arithmetic_sequence : q = p + 4 ∧ r = q + 4 ∧ s = r + 4)
  (p_gt_3 : p > 3) : 
  p % 10 = 9 := 
sorry

end primes_in_arithmetic_sequence_have_specific_ones_digit_l1979_197983


namespace total_students_l1979_197913

theorem total_students (girls boys : ℕ) (h1 : girls = 300) (h2 : boys = 8 * (girls / 5)) : girls + boys = 780 := by
  sorry

end total_students_l1979_197913


namespace ceil_of_fractional_square_l1979_197968

theorem ceil_of_fractional_square :
  (Int.ceil ((- (7/4) + 1/4) ^ 2) = 3) :=
by
  sorry

end ceil_of_fractional_square_l1979_197968


namespace find_B_squared_l1979_197982

noncomputable def g (x : ℝ) : ℝ := Real.sqrt 31 + 85 / x

theorem find_B_squared :
  let x1 := (Real.sqrt 31 + Real.sqrt 371) / 2
  let x2 := (Real.sqrt 31 - Real.sqrt 371) / 2
  let B := |x1| + |x2|
  B^2 = 371 :=
by
  sorry

end find_B_squared_l1979_197982


namespace prob_four_vertical_faces_same_color_l1979_197975

noncomputable def painted_cube_probability : ℚ :=
  let total_arrangements := 3^6
  let suitable_arrangements := 3 + 18 + 6
  suitable_arrangements / total_arrangements

theorem prob_four_vertical_faces_same_color : 
  painted_cube_probability = 1 / 27 := by
  sorry

end prob_four_vertical_faces_same_color_l1979_197975


namespace cricket_team_matches_in_august_l1979_197911

noncomputable def cricket_matches_played_in_august (M W W_new: ℕ) : Prop :=
  W = 26 * M / 100 ∧
  W_new = 52 * (M + 65) / 100 ∧ 
  W_new = W + 65

theorem cricket_team_matches_in_august (M W W_new: ℕ) : cricket_matches_played_in_august M W W_new → M = 120 := 
by
  sorry

end cricket_team_matches_in_august_l1979_197911


namespace smaller_cuboid_length_l1979_197959

theorem smaller_cuboid_length
  (width_sm : ℝ)
  (height_sm : ℝ)
  (length_lg : ℝ)
  (width_lg : ℝ)
  (height_lg : ℝ)
  (num_sm : ℝ)
  (h1 : width_sm = 2)
  (h2 : height_sm = 3)
  (h3 : length_lg = 18)
  (h4 : width_lg = 15)
  (h5 : height_lg = 2)
  (h6 : num_sm = 18) :
  ∃ (length_sm : ℝ), (108 * length_sm = 540) ∧ (length_sm = 5) :=
by
  -- proof logic will be here
  sorry

end smaller_cuboid_length_l1979_197959


namespace find_A_l1979_197931

def clubsuit (A B : ℤ) : ℤ := 4 * A + 2 * B + 6

theorem find_A : ∃ A : ℤ, clubsuit A 6 = 70 → A = 13 := 
by
  sorry

end find_A_l1979_197931


namespace capacity_of_bucket_in_first_scenario_l1979_197996

theorem capacity_of_bucket_in_first_scenario (x : ℝ) 
  (h1 : 28 * x = 378) : x = 13.5 :=
by
  sorry

end capacity_of_bucket_in_first_scenario_l1979_197996


namespace emails_left_in_inbox_l1979_197970

-- Define the initial conditions and operations
def initial_emails : ℕ := 600

def move_half_to_trash (emails : ℕ) : ℕ := emails / 2
def move_40_percent_to_work (emails : ℕ) : ℕ := emails - (emails * 40 / 100)
def move_25_percent_to_personal (emails : ℕ) : ℕ := emails - (emails * 25 / 100)
def move_10_percent_to_miscellaneous (emails : ℕ) : ℕ := emails - (emails * 10 / 100)
def filter_30_percent_to_subfolders (emails : ℕ) : ℕ := emails - (emails * 30 / 100)
def archive_20_percent (emails : ℕ) : ℕ := emails - (emails * 20 / 100)

-- Statement we need to prove
theorem emails_left_in_inbox : 
  archive_20_percent
    (filter_30_percent_to_subfolders
      (move_10_percent_to_miscellaneous
        (move_25_percent_to_personal
          (move_40_percent_to_work
            (move_half_to_trash initial_emails))))) = 69 := 
by sorry

end emails_left_in_inbox_l1979_197970


namespace rocky_miles_total_l1979_197951

-- Defining the conditions
def m1 : ℕ := 4
def m2 : ℕ := 2 * m1
def m3 : ℕ := 3 * m2

-- The statement to be proven
theorem rocky_miles_total : m1 + m2 + m3 = 36 := by
  sorry

end rocky_miles_total_l1979_197951


namespace deepak_age_is_21_l1979_197907

noncomputable def DeepakCurrentAge (x : ℕ) : Prop :=
  let Rahul := 4 * x
  let Deepak := 3 * x
  let Karan := 5 * x
  Rahul + 6 = 34 ∧
  (Rahul + 6) / 7 = (Deepak + 6) / 5 ∧ (Rahul + 6) / 7 = (Karan + 6) / 9 → 
  Deepak = 21

theorem deepak_age_is_21 : ∃ x : ℕ, DeepakCurrentAge x :=
by
  use 7
  sorry

end deepak_age_is_21_l1979_197907


namespace total_amount_spent_l1979_197941

theorem total_amount_spent (T : ℝ) (h1 : 5000 + 200 + 0.30 * T = T) : 
  T = 7428.57 :=
by
  sorry

end total_amount_spent_l1979_197941


namespace max_cos_x_l1979_197961

theorem max_cos_x (x y : ℝ) (h : Real.cos (x - y) = Real.cos x - Real.cos y) : 
  ∃ M, (∀ x, Real.cos x <= M) ∧ M = 1 := 
sorry

end max_cos_x_l1979_197961


namespace remainder_product_191_193_197_mod_23_l1979_197993

theorem remainder_product_191_193_197_mod_23 :
  (191 * 193 * 197) % 23 = 14 := by
  sorry

end remainder_product_191_193_197_mod_23_l1979_197993


namespace john_recreation_percent_l1979_197973

theorem john_recreation_percent (W : ℝ) (P : ℝ) (H1 : 0 ≤ P ∧ P ≤ 1) (H2 : 0 ≤ W) (H3 : 0.15 * W = 0.50 * (P * W)) :
  P = 0.30 :=
by
  sorry

end john_recreation_percent_l1979_197973


namespace proof_problem_l1979_197953

def f (x : ℤ) : ℤ := 2 * x - 1
def g (x : ℤ) : ℤ := x^2 + 2 * x + 1

theorem proof_problem : f (g 3) - g (f 3) = -5 := by
  sorry

end proof_problem_l1979_197953


namespace possible_values_of_expression_l1979_197999

theorem possible_values_of_expression (x : ℝ) (h : 3 ≤ x ∧ x ≤ 4) : 
  40 ≤ x^2 + 7 * x + 10 ∧ x^2 + 7 * x + 10 ≤ 54 := 
sorry

end possible_values_of_expression_l1979_197999


namespace calculate_sum_of_triangles_l1979_197934

def operation_triangle (a b c : Int) : Int :=
  a * b - c 

theorem calculate_sum_of_triangles :
  operation_triangle 3 4 5 + operation_triangle 1 2 4 + operation_triangle 2 5 6 = 9 :=
by 
  sorry

end calculate_sum_of_triangles_l1979_197934


namespace simplify_polynomial_l1979_197977

def p (x : ℝ) : ℝ := 3 * x^5 - x^4 + 2 * x^3 + 5 * x^2 - 3 * x + 7
def q (x : ℝ) : ℝ := -x^5 + 4 * x^4 + x^3 - 6 * x^2 + 5 * x - 4
def r (x : ℝ) : ℝ := 2 * x^5 - 3 * x^4 + 4 * x^3 - x^2 - x + 2

theorem simplify_polynomial (x : ℝ) :
  (p x) + (q x) - (r x) = 6 * x^4 - x^3 + 3 * x + 1 :=
by sorry

end simplify_polynomial_l1979_197977


namespace jessica_initial_money_l1979_197949

def amount_spent : ℝ := 10.22
def amount_left : ℝ := 1.51
def initial_amount : ℝ := 11.73

theorem jessica_initial_money :
  amount_spent + amount_left = initial_amount := 
  by
    sorry

end jessica_initial_money_l1979_197949


namespace people_got_rid_of_some_snails_l1979_197980

namespace SnailProblem

def originalSnails : ℕ := 11760
def remainingSnails : ℕ := 8278
def snailsGotRidOf : ℕ := 3482

theorem people_got_rid_of_some_snails :
  originalSnails - remainingSnails = snailsGotRidOf :=
by 
  sorry

end SnailProblem

end people_got_rid_of_some_snails_l1979_197980


namespace time_to_drain_tank_l1979_197942

theorem time_to_drain_tank (P L: ℝ) (hP : P = 1/3) (h_combined : P - L = 2/7) : 1 / L = 21 :=
by
  -- Proof omitted. Use the conditions given to show that 1 / L = 21.
  sorry

end time_to_drain_tank_l1979_197942


namespace run_to_cafe_time_l1979_197948

theorem run_to_cafe_time (h_speed_const : ∀ t1 t2 d1 d2 : ℝ, (t1 / d1) = (t2 / d2))
  (h_store_time : 24 = 3 * (24 / 3))
  (h_cafe_halfway : ∀ d : ℝ, d = 1.5) :
  ∃ t : ℝ, t = 12 :=
by
  sorry

end run_to_cafe_time_l1979_197948


namespace rest_area_location_l1979_197984

theorem rest_area_location :
  ∀ (A B : ℝ), A = 50 → B = 230 → (5 / 8 * (B - A) + A = 162.5) :=
by
  intros A B hA hB
  rw [hA, hB]
  -- doing the computation to show the rest area is at 162.5 km
  sorry

end rest_area_location_l1979_197984


namespace consecutive_integers_sqrt19_sum_l1979_197908

theorem consecutive_integers_sqrt19_sum :
  ∃ a b : ℤ, (a < ⌊Real.sqrt 19⌋ ∧ ⌊Real.sqrt 19⌋ < b ∧ a + 1 = b) ∧ a + b = 9 := 
by
  sorry

end consecutive_integers_sqrt19_sum_l1979_197908


namespace contrapositive_necessary_condition_l1979_197920

theorem contrapositive_necessary_condition {p q : Prop} (h : p → q) : ¬p → ¬q :=
  by sorry

end contrapositive_necessary_condition_l1979_197920
