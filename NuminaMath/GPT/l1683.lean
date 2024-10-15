import Mathlib

namespace NUMINAMATH_GPT_inequality_div_l1683_168340

theorem inequality_div (m n : ℝ) (h : m > n) : (m / 5) > (n / 5) :=
sorry

end NUMINAMATH_GPT_inequality_div_l1683_168340


namespace NUMINAMATH_GPT_power_equality_l1683_168314

theorem power_equality : 
  ( (11 : ℝ) ^ (1 / 5) / (11 : ℝ) ^ (1 / 7) ) = (11 : ℝ) ^ (2 / 35) := 
by sorry

end NUMINAMATH_GPT_power_equality_l1683_168314


namespace NUMINAMATH_GPT_highest_avg_speed_2_to_3_l1683_168375

-- Define the time periods and distances traveled in those periods
def distance_8_to_9 : ℕ := 50
def distance_9_to_10 : ℕ := 70
def distance_10_to_11 : ℕ := 60
def distance_2_to_3 : ℕ := 80
def distance_3_to_4 : ℕ := 40

-- Define the average speed calculation for each period
def avg_speed (distance : ℕ) (hours : ℕ) : ℕ := distance / hours

-- Proposition stating that the highest average speed is from 2 pm to 3 pm
theorem highest_avg_speed_2_to_3 : 
  avg_speed distance_2_to_3 1 > avg_speed distance_8_to_9 1 ∧ 
  avg_speed distance_2_to_3 1 > avg_speed distance_9_to_10 1 ∧ 
  avg_speed distance_2_to_3 1 > avg_speed distance_10_to_11 1 ∧ 
  avg_speed distance_2_to_3 1 > avg_speed distance_3_to_4 1 := 
by 
  sorry

end NUMINAMATH_GPT_highest_avg_speed_2_to_3_l1683_168375


namespace NUMINAMATH_GPT_polygon_sides_l1683_168368

theorem polygon_sides :
  ∀ (n : ℕ), (n > 2) → (n - 2) * 180 < 360 → n = 3 :=
by
  intros n hn1 hn2
  sorry

end NUMINAMATH_GPT_polygon_sides_l1683_168368


namespace NUMINAMATH_GPT_find_second_number_in_second_set_l1683_168364

theorem find_second_number_in_second_set :
    (14 + 32 + 53) / 3 = 3 + (21 + x + 22) / 3 → x = 47 :=
by intro h
   sorry

end NUMINAMATH_GPT_find_second_number_in_second_set_l1683_168364


namespace NUMINAMATH_GPT_unique_tangent_lines_through_point_l1683_168370

theorem unique_tangent_lines_through_point (P : ℝ × ℝ) (hP : P = (2, 4)) :
  ∃! l : ℝ × ℝ → Prop, (l P) ∧ (∀ p : ℝ × ℝ, l p → p ∈ {p : ℝ × ℝ | p.2 ^ 2 = 8 * p.1}) := sorry

end NUMINAMATH_GPT_unique_tangent_lines_through_point_l1683_168370


namespace NUMINAMATH_GPT_p_is_contradictory_to_q_l1683_168357

variable (a : ℝ)

def p := a > 0 → a^2 ≠ 0
def q := a ≤ 0 → a^2 = 0

theorem p_is_contradictory_to_q : (p a) ↔ ¬ (q a) :=
by
  sorry

end NUMINAMATH_GPT_p_is_contradictory_to_q_l1683_168357


namespace NUMINAMATH_GPT_cost_per_first_30_kg_is_10_l1683_168327

-- Definitions of the constants based on the conditions
def cost_per_33_kg (p q : ℝ) : Prop := 30 * p + 3 * q = 360
def cost_per_36_kg (p q : ℝ) : Prop := 30 * p + 6 * q = 420
def cost_per_25_kg (p : ℝ) : Prop := 25 * p = 250

-- The statement we want to prove
theorem cost_per_first_30_kg_is_10 (p q : ℝ) 
  (h1 : cost_per_33_kg p q)
  (h2 : cost_per_36_kg p q)
  (h3 : cost_per_25_kg p) : 
  p = 10 :=
sorry

end NUMINAMATH_GPT_cost_per_first_30_kg_is_10_l1683_168327


namespace NUMINAMATH_GPT_four_nabla_seven_l1683_168328

-- Define the operation ∇
def nabla (a b : ℤ) : ℚ :=
  (a + b) / (1 + a * b)

theorem four_nabla_seven :
  nabla 4 7 = 11 / 29 :=
by
  sorry

end NUMINAMATH_GPT_four_nabla_seven_l1683_168328


namespace NUMINAMATH_GPT_reciprocal_inequality_of_negatives_l1683_168321

variable (a b : ℝ)

/-- Given that a < b < 0, prove that 1/a > 1/b. -/
theorem reciprocal_inequality_of_negatives (h1 : a < b) (h2 : b < 0) : (1/a) > (1/b) :=
sorry

end NUMINAMATH_GPT_reciprocal_inequality_of_negatives_l1683_168321


namespace NUMINAMATH_GPT_sqrt_inequality_l1683_168343

theorem sqrt_inequality (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) (habc : a + b + c = 9) : 
  Real.sqrt (a * b + b * c + c * a) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c := 
sorry

end NUMINAMATH_GPT_sqrt_inequality_l1683_168343


namespace NUMINAMATH_GPT_Taran_original_number_is_12_l1683_168304

open Nat

theorem Taran_original_number_is_12 (x : ℕ)
  (h1 : (5 * x) + 5 - 5 = 73 ∨ (5 * x) + 5 - 6 = 73 ∨ (5 * x) + 6 - 5 = 73 ∨ (5 * x) + 6 - 6 = 73 ∨ 
       (6 * x) + 5 - 5 = 73 ∨ (6 * x) + 5 - 6 = 73 ∨ (6 * x) + 6 - 5 = 73 ∨ (6 * x) + 6 - 6 = 73) : x = 12 := by
  sorry

end NUMINAMATH_GPT_Taran_original_number_is_12_l1683_168304


namespace NUMINAMATH_GPT_remainder_25197629_mod_4_l1683_168325

theorem remainder_25197629_mod_4 : 25197629 % 4 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_25197629_mod_4_l1683_168325


namespace NUMINAMATH_GPT_smallest_sum_of_pairwise_distinct_squares_l1683_168335

theorem smallest_sum_of_pairwise_distinct_squares :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  ∃ x y z : ℕ, a + b = x^2 ∧ b + c = z^2 ∧ c + a = y^2 ∧ a + b + c = 55 :=
sorry

end NUMINAMATH_GPT_smallest_sum_of_pairwise_distinct_squares_l1683_168335


namespace NUMINAMATH_GPT_meeting_distance_from_top_l1683_168377

section

def total_distance : ℝ := 12
def uphill_distance : ℝ := 6
def downhill_distance : ℝ := 6
def john_start_time : ℝ := 0.25
def john_uphill_speed : ℝ := 12
def john_downhill_speed : ℝ := 18
def jenny_uphill_speed : ℝ := 14
def jenny_downhill_speed : ℝ := 21

theorem meeting_distance_from_top : 
  ∃ (d : ℝ), d = 6 - 14 * ((0.25) + 6 / 14 - (1 / 2) - (6 - 18 * ((1 / 2) + d / 18))) / 14 ∧ d = 45 / 32 :=
sorry

end

end NUMINAMATH_GPT_meeting_distance_from_top_l1683_168377


namespace NUMINAMATH_GPT_d_divisibility_l1683_168365

theorem d_divisibility (p d : ℕ) (h_p : 0 < p) (h_d : 0 < d)
  (h1 : Prime p) 
  (h2 : Prime (p + d)) 
  (h3 : Prime (p + 2 * d)) 
  (h4 : Prime (p + 3 * d)) 
  (h5 : Prime (p + 4 * d)) 
  (h6 : Prime (p + 5 * d)) : 
  (2 ∣ d) ∧ (3 ∣ d) ∧ (5 ∣ d) :=
by
  sorry

end NUMINAMATH_GPT_d_divisibility_l1683_168365


namespace NUMINAMATH_GPT_expression_is_integer_if_k_eq_2_l1683_168348

def binom (n k : ℕ) := n.factorial / (k.factorial * (n-k).factorial)

theorem expression_is_integer_if_k_eq_2 
  (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) (h3 : k = 2) : 
  ∃ (m : ℕ), m = (n - 3 * k + 2) * binom n k / (k + 2) := sorry

end NUMINAMATH_GPT_expression_is_integer_if_k_eq_2_l1683_168348


namespace NUMINAMATH_GPT_pythagorean_triple_correct_l1683_168381

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_correct : 
  is_pythagorean_triple 9 12 15 ∧ ¬ is_pythagorean_triple 3 4 6 ∧ ¬ is_pythagorean_triple 1 2 3 ∧ ¬ is_pythagorean_triple 6 12 13 :=
by
  sorry

end NUMINAMATH_GPT_pythagorean_triple_correct_l1683_168381


namespace NUMINAMATH_GPT_algebraic_expression_value_l1683_168391

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y = -2) : 
  (2 * y - x) ^ 2 - 2 * x + 4 * y - 1 = 7 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1683_168391


namespace NUMINAMATH_GPT_second_term_arithmetic_sequence_l1683_168332

theorem second_term_arithmetic_sequence (a d : ℝ) (h : a + (a + 2 * d) = 10) : 
  a + d = 5 :=
by
  sorry

end NUMINAMATH_GPT_second_term_arithmetic_sequence_l1683_168332


namespace NUMINAMATH_GPT_sam_total_cents_l1683_168313

def dimes_to_cents (dimes : ℕ) : ℕ := dimes * 10
def quarters_to_cents (quarters : ℕ) : ℕ := quarters * 25
def nickels_to_cents (nickels : ℕ) : ℕ := nickels * 5
def dollars_to_cents (dollars : ℕ) : ℕ := dollars * 100

noncomputable def total_cents (initial_dimes dad_dimes mom_dimes grandma_dollars sister_quarters_initial : ℕ)
                             (initial_quarters dad_quarters mom_quarters grandma_transform sister_quarters_donation : ℕ)
                             (initial_nickels dad_nickels mom_nickels grandma_conversion sister_nickels_donation : ℕ) : ℕ :=
  dimes_to_cents initial_dimes +
  quarters_to_cents initial_quarters +
  nickels_to_cents initial_nickels +
  dimes_to_cents dad_dimes +
  quarters_to_cents dad_quarters -
  nickels_to_cents mom_nickels -
  dimes_to_cents mom_dimes +
  dollars_to_cents grandma_dollars +
  quarters_to_cents sister_quarters_donation +
  nickels_to_cents sister_nickels_donation

theorem sam_total_cents :
  total_cents 9 7 2 3 4 5 2 0 0 3 2 1 = 735 := 
  by exact sorry

end NUMINAMATH_GPT_sam_total_cents_l1683_168313


namespace NUMINAMATH_GPT_part1_proof_part2_proof_part3_proof_part4_proof_l1683_168337

variable {A B C : Type}
variables {a b c : ℝ}  -- Sides of the triangle
variables {h_a h_b h_c r r_a r_b r_c : ℝ}  -- Altitudes, inradius, and exradii of \triangle ABC

-- Part 1: Proving the sum of altitudes related to sides and inradius
theorem part1_proof : h_a + h_b + h_c = r * (a + b + c) * (1 / a + 1 / b + 1 / c) := sorry

-- Part 2: Proving the sum of reciprocals of altitudes related to the reciprocal of inradius and exradii
theorem part2_proof : (1 / h_a) + (1 / h_b) + (1 / h_c) = 1 / r ∧ 1 / r = (1 / r_a) + (1 / r_b) + (1 / r_c) := sorry

-- Part 3: Combining results of parts 1 and 2 to prove product of sums
theorem part3_proof : (h_a + h_b + h_c) * ((1 / h_a) + (1 / h_b) + (1 / h_c)) = (a + b + c) * (1 / a + 1 / b + 1 / c) := sorry

-- Part 4: Final geometric identity
theorem part4_proof : (h_a + h_c) / r_a + (h_c + h_a) / r_b + (h_a + h_b) / r_c = 6 := sorry

end NUMINAMATH_GPT_part1_proof_part2_proof_part3_proof_part4_proof_l1683_168337


namespace NUMINAMATH_GPT_S2016_value_l1683_168312

theorem S2016_value (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h1 : a 1 = -2016)
  (h2 : ∀ n, S (n+1) = S n + a (n+1))
  (h3 : ∀ n, a (n+1) = a n + d)
  (h4 : (S 2015) / 2015 - (S 2012) / 2012 = 3) : S 2016 = -2016 := 
sorry

end NUMINAMATH_GPT_S2016_value_l1683_168312


namespace NUMINAMATH_GPT_blocks_to_get_home_l1683_168330

-- Definitions based on conditions provided
def blocks_to_park := 4
def blocks_to_school := 7
def trips_per_day := 3
def total_daily_blocks := 66

-- The proof statement for the number of blocks Ray walks to get back home
theorem blocks_to_get_home 
  (h1: blocks_to_park = 4)
  (h2: blocks_to_school = 7)
  (h3: trips_per_day = 3)
  (h4: total_daily_blocks = 66) : 
  (total_daily_blocks / trips_per_day - (blocks_to_park + blocks_to_school) = 11) :=
by
  sorry

end NUMINAMATH_GPT_blocks_to_get_home_l1683_168330


namespace NUMINAMATH_GPT_number_of_triangles_l1683_168306

open Nat

-- Define the number of combinations
def comb : Nat → Nat → Nat
  | n, k => if k > n then 0 else n.choose k

-- The given conditions
def points_on_OA := 5
def points_on_OB := 6
def point_O := 1
def total_points := points_on_OA + points_on_OB + point_O -- should equal 12

-- Lean proof problem statement
theorem number_of_triangles : comb total_points 3 - comb points_on_OA 3 - comb points_on_OB 3 = 165 := by
  sorry

end NUMINAMATH_GPT_number_of_triangles_l1683_168306


namespace NUMINAMATH_GPT_max_s_value_l1683_168334

theorem max_s_value (r s : ℕ) (hr : r ≥ s) (hs : s ≥ 3)
  (h : ((r - 2) * 180 / r : ℚ) / ((s - 2) * 180 / s) = 60 / 59) :
  s = 117 :=
by
  sorry

end NUMINAMATH_GPT_max_s_value_l1683_168334


namespace NUMINAMATH_GPT_sum_powers_of_5_mod_8_l1683_168362

theorem sum_powers_of_5_mod_8 :
  (List.sum (List.map (fun n => (5^n % 8)) (List.range 2011))) % 8 = 4 := 
  sorry

end NUMINAMATH_GPT_sum_powers_of_5_mod_8_l1683_168362


namespace NUMINAMATH_GPT_value_of_a_l1683_168300

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, x > 0.5 → 1 - a / 2^x > 0) → a = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1683_168300


namespace NUMINAMATH_GPT_integer_not_always_greater_decimal_l1683_168367

-- Definitions based on conditions
def is_decimal (d : ℚ) : Prop :=
  ∃ (i : ℤ) (f : ℚ), 0 ≤ f ∧ f < 1 ∧ d = i + f

def is_greater (a : ℤ) (b : ℚ) : Prop :=
  (a : ℚ) > b

theorem integer_not_always_greater_decimal : ¬ ∀ n : ℤ, ∀ d : ℚ, is_decimal d → (is_greater n d) :=
by
  sorry

end NUMINAMATH_GPT_integer_not_always_greater_decimal_l1683_168367


namespace NUMINAMATH_GPT_largest_base5_to_base7_l1683_168359

-- Define the largest four-digit number in base-5
def largest_base5_four_digit_number : ℕ := 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

-- Convert this number to base-7
def convert_to_base7 (n : ℕ) : ℕ := 
  let d3 := n / (7^3)
  let r3 := n % (7^3)
  let d2 := r3 / (7^2)
  let r2 := r3 % (7^2)
  let d1 := r2 / (7^1)
  let r1 := r2 % (7^1)
  let d0 := r1
  (d3 * 10^3) + (d2 * 10^2) + (d1 * 10^1) + d0

-- Theorem to prove m in base-7
theorem largest_base5_to_base7 : 
  convert_to_base7 largest_base5_four_digit_number = 1551 :=
by 
  -- skip the proof
  sorry

end NUMINAMATH_GPT_largest_base5_to_base7_l1683_168359


namespace NUMINAMATH_GPT_river_flow_volume_l1683_168356

noncomputable def river_depth : ℝ := 2
noncomputable def river_width : ℝ := 45
noncomputable def flow_rate_kmph : ℝ := 4
noncomputable def flow_rate_mpm := flow_rate_kmph * 1000 / 60
noncomputable def cross_sectional_area := river_depth * river_width
noncomputable def volume_per_minute := cross_sectional_area * flow_rate_mpm

theorem river_flow_volume :
  volume_per_minute = 6000.3 := by
  sorry

end NUMINAMATH_GPT_river_flow_volume_l1683_168356


namespace NUMINAMATH_GPT_solution_interval_l1683_168322

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ x - x^(1 / 3)

theorem solution_interval (x₀ : ℝ) 
  (h_solution : (1 / 2)^x₀ = x₀^(1 / 3)) : x₀ ∈ Set.Ioo (1 / 3) (1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_solution_interval_l1683_168322


namespace NUMINAMATH_GPT_arithmetic_sequence_term_2011_is_671st_l1683_168333

theorem arithmetic_sequence_term_2011_is_671st:
  ∀ (a1 d n : ℕ), a1 = 1 → d = 3 → (3 * n - 2 = 2011) → n = 671 :=
by 
  intros a1 d n ha1 hd h_eq;
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_term_2011_is_671st_l1683_168333


namespace NUMINAMATH_GPT_find_positive_number_l1683_168344

theorem find_positive_number (x : ℝ) (h_pos : 0 < x) (h_eq : (2 / 3) * x = (49 / 216) * (1 / x)) : x = 24.5 :=
by
  sorry

end NUMINAMATH_GPT_find_positive_number_l1683_168344


namespace NUMINAMATH_GPT_number_of_coins_l1683_168361

-- Define the conditions
def equal_number_of_coins (x : ℝ) :=
  ∃ n : ℝ, n = x

-- Define the total value condition
def total_value (x : ℝ) :=
  x + 0.50 * x + 0.25 * x = 70

-- The theorem to be proved
theorem number_of_coins (x : ℝ) (h1 : equal_number_of_coins x) (h2 : total_value x) : x = 40 :=
by sorry

end NUMINAMATH_GPT_number_of_coins_l1683_168361


namespace NUMINAMATH_GPT_perfect_square_461_l1683_168302

theorem perfect_square_461 (x : ℤ) (y : ℤ) (hx : 5 ∣ x) (hy : 5 ∣ y) 
  (h : x^2 + 461 = y^2) : x^2 = 52900 :=
  sorry

end NUMINAMATH_GPT_perfect_square_461_l1683_168302


namespace NUMINAMATH_GPT_jason_remaining_pokemon_cards_l1683_168392

theorem jason_remaining_pokemon_cards :
  (3 - 2) = 1 :=
by 
  sorry

end NUMINAMATH_GPT_jason_remaining_pokemon_cards_l1683_168392


namespace NUMINAMATH_GPT_greatest_number_that_divides_54_87_172_l1683_168363

noncomputable def gcdThree (a b c : ℤ) : ℤ :=
  gcd (gcd a b) c

theorem greatest_number_that_divides_54_87_172
  (d r : ℤ)
  (h1 : 54 % d = r)
  (h2 : 87 % d = r)
  (h3 : 172 % d = r) :
  d = gcdThree 33 85 118 := by
  -- We would start the proof here, but it's omitted per instructions
  sorry

end NUMINAMATH_GPT_greatest_number_that_divides_54_87_172_l1683_168363


namespace NUMINAMATH_GPT_number_of_one_dollar_coins_l1683_168320

theorem number_of_one_dollar_coins (t : ℕ) :
  (∃ k : ℕ, 3 * k = t) → ∃ k : ℕ, k = t / 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_one_dollar_coins_l1683_168320


namespace NUMINAMATH_GPT_trip_time_difference_l1683_168353

def travel_time (distance speed : ℕ) : ℕ :=
  distance / speed

theorem trip_time_difference
  (speed : ℕ)
  (speed_pos : 0 < speed)
  (distance1 : ℕ)
  (distance2 : ℕ)
  (time_difference : ℕ)
  (h1 : distance1 = 540)
  (h2 : distance2 = 600)
  (h_speed : speed = 60)
  (h_time_diff : time_difference = (travel_time distance2 speed) - (travel_time distance1 speed) * 60)
  : time_difference = 60 :=
by
  sorry

end NUMINAMATH_GPT_trip_time_difference_l1683_168353


namespace NUMINAMATH_GPT_prove_x_minus_y_squared_l1683_168382

noncomputable section

variables {x y a b : ℝ}

theorem prove_x_minus_y_squared (h1 : x * y = b) (h2 : x / y + y / x = a) : (x - y) ^ 2 = a * b - 2 * b := 
  sorry

end NUMINAMATH_GPT_prove_x_minus_y_squared_l1683_168382


namespace NUMINAMATH_GPT_annika_total_kilometers_east_l1683_168338

def annika_constant_rate : ℝ := 10 -- 10 minutes per kilometer
def distance_hiked_initially : ℝ := 2.5 -- 2.5 kilometers
def total_time_to_return : ℝ := 35 -- 35 minutes

theorem annika_total_kilometers_east :
  (total_time_to_return - (distance_hiked_initially * annika_constant_rate)) / annika_constant_rate + distance_hiked_initially = 3.5 := by
  sorry

end NUMINAMATH_GPT_annika_total_kilometers_east_l1683_168338


namespace NUMINAMATH_GPT_total_cost_second_set_l1683_168395

variable (A V : ℝ)

-- Condition declarations
axiom cost_video_cassette : V = 300
axiom cost_second_set : 7 * A + 3 * V = 1110

-- Proof goal
theorem total_cost_second_set :
  7 * A + 3 * V = 1110 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_second_set_l1683_168395


namespace NUMINAMATH_GPT_elsa_data_usage_l1683_168326

theorem elsa_data_usage (D : ℝ) 
  (h_condition : D - 300 - (2/5) * (D - 300) = 120) : D = 500 := 
sorry

end NUMINAMATH_GPT_elsa_data_usage_l1683_168326


namespace NUMINAMATH_GPT_percentage_paid_to_A_l1683_168371

theorem percentage_paid_to_A (A B : ℝ) (h1 : A + B = 550) (h2 : B = 220) : (A / B) * 100 = 150 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_percentage_paid_to_A_l1683_168371


namespace NUMINAMATH_GPT_george_borrow_amount_l1683_168374

-- Define the conditions
def initial_fee_rate : ℝ := 0.05
def doubling_rate : ℝ := 2
def total_weeks : ℕ := 2
def total_fee : ℝ := 15

-- Define the problem statement
theorem george_borrow_amount : 
  ∃ (P : ℝ), (initial_fee_rate * P + initial_fee_rate * doubling_rate * P = total_fee) ∧ P = 100 :=
by
  -- Statement only, proof is skipped
  sorry

end NUMINAMATH_GPT_george_borrow_amount_l1683_168374


namespace NUMINAMATH_GPT_triangle_with_incircle_radius_one_has_sides_5_4_3_l1683_168387

variable {a b c : ℕ} (h1 : a ≥ b ∧ b ≥ c)
variable (h2 : ∃ (a b c : ℕ), (a + b + c) / 2 * 1 = (a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))

theorem triangle_with_incircle_radius_one_has_sides_5_4_3 :
  a = 5 ∧ b = 4 ∧ c = 3 :=
by
    sorry

end NUMINAMATH_GPT_triangle_with_incircle_radius_one_has_sides_5_4_3_l1683_168387


namespace NUMINAMATH_GPT_tangent_circles_m_values_l1683_168389

noncomputable def is_tangent (m : ℝ) : Prop :=
  let o1_center := (m, 0)
  let o2_center := (-1, 2 * m)
  let distance := Real.sqrt ((m + 1)^2 + (2 * m)^2)
  (distance = 5 ∨ distance = 1)

theorem tangent_circles_m_values :
  {m : ℝ | is_tangent m} = {-12 / 5, -2 / 5, 0, 2} := by
  sorry

end NUMINAMATH_GPT_tangent_circles_m_values_l1683_168389


namespace NUMINAMATH_GPT_expression_equals_one_l1683_168351

variable {R : Type*} [Field R]
variables (x y z : R)

theorem expression_equals_one (h₁ : x ≠ y) (h₂ : x ≠ z) (h₃ : y ≠ z) :
    (x^2 / ((x - y) * (x - z)) + y^2 / ((y - x) * (y - z)) + z^2 / ((z - x) * (z - y))) = 1 :=
by sorry

end NUMINAMATH_GPT_expression_equals_one_l1683_168351


namespace NUMINAMATH_GPT_oranges_taken_from_basket_l1683_168331

-- Define the original number of oranges and the number left after taking some out.
def original_oranges : ℕ := 8
def oranges_left : ℕ := 3

-- Prove that the number of oranges taken from the basket equals 5.
theorem oranges_taken_from_basket : original_oranges - oranges_left = 5 := by
  sorry

end NUMINAMATH_GPT_oranges_taken_from_basket_l1683_168331


namespace NUMINAMATH_GPT_equation_has_real_roots_l1683_168379

theorem equation_has_real_roots (k : ℝ) : ∀ (x : ℝ), 
  ∃ x, x = k^2 * (x - 1) * (x - 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_equation_has_real_roots_l1683_168379


namespace NUMINAMATH_GPT_exam_papers_count_l1683_168336

theorem exam_papers_count (F x : ℝ) :
  (∀ n : ℕ, n = 5) →    -- condition 1: equivalence of n to proportions count
  (6 * x + 7 * x + 8 * x + 9 * x + 10 * x = 40 * x) →    -- condition 2: sum of proportions
  (40 * x = 0.60 * n * F) →   -- condition 3: student obtained 60% of total marks
  (7 * x > 0.50 * F ∧ 8 * x > 0.50 * F ∧ 9 * x > 0.50 * F ∧ 10 * x > 0.50 * F ∧ 6 * x ≤ 0.50 * F) →  -- condition 4: more than 50% in 4 papers
  ∃ n : ℕ, n = 5 :=    -- prove: number of papers is 5
sorry

end NUMINAMATH_GPT_exam_papers_count_l1683_168336


namespace NUMINAMATH_GPT_compute_expression_l1683_168342

theorem compute_expression : 1007^2 - 993^2 - 1005^2 + 995^2 = 8000 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l1683_168342


namespace NUMINAMATH_GPT_purely_imaginary_implies_m_eq_neg_half_simplify_z_squared_over_z_add_5_plus_2i_l1683_168345

def z (m : ℝ) : Complex := Complex.mk (2 * m^2 - 3 * m - 2) (m^2 - 3 * m + 2)

theorem purely_imaginary_implies_m_eq_neg_half (m : ℝ) : 
  (z m).re = 0 ↔ m = -1 / 2 := sorry

theorem simplify_z_squared_over_z_add_5_plus_2i (z_zero : ℂ) :
  z 0 = ⟨-2, 2⟩ →
  (z 0)^2 / (z 0 + Complex.mk 5 2) = ⟨-32 / 25, -24 / 25⟩ := sorry

end NUMINAMATH_GPT_purely_imaginary_implies_m_eq_neg_half_simplify_z_squared_over_z_add_5_plus_2i_l1683_168345


namespace NUMINAMATH_GPT_min_sum_xy_l1683_168341

theorem min_sum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y + x * y = 3) : x + y ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_min_sum_xy_l1683_168341


namespace NUMINAMATH_GPT_union_sets_l1683_168369

-- Define the sets A and B
def A : Set ℤ := {0, 1}
def B : Set ℤ := {x : ℤ | (x + 2) * (x - 1) < 0}

-- The theorem to be proven
theorem union_sets : A ∪ B = {-1, 0, 1} :=
by
  sorry

end NUMINAMATH_GPT_union_sets_l1683_168369


namespace NUMINAMATH_GPT_equality_condition_l1683_168358

theorem equality_condition (a b c : ℝ) :
  a + b + c = (a + b) * (a + c) → a = 1 ∧ b = 1 ∧ c = 1 :=
by
  sorry

end NUMINAMATH_GPT_equality_condition_l1683_168358


namespace NUMINAMATH_GPT_max_min_sundays_in_month_l1683_168346

def week_days : ℕ := 7
def min_month_days : ℕ := 28
def months_days (d : ℕ) : Prop := d = 28 ∨ d = 30 ∨ d = 31

theorem max_min_sundays_in_month (d : ℕ) (h1 : months_days d) :
  4 ≤ (d / week_days) + ite (d % week_days > 0) 1 0 ∧ (d / week_days) + ite (d % week_days > 0) 1 0 ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_max_min_sundays_in_month_l1683_168346


namespace NUMINAMATH_GPT_option_A_two_solutions_l1683_168399

theorem option_A_two_solutions :
    (∀ (a b : ℝ) (A : ℝ), 
    (a = 3 ∧ b = 4 ∧ A = 45) ∨ 
    (a = 7 ∧ b = 14 ∧ A = 30) ∨ 
    (a = 2 ∧ b = 7 ∧ A = 60) ∨ 
    (a = 8 ∧ b = 5 ∧ A = 135) →
    (∃ a b A : ℝ, a = 3 ∧ b = 4 ∧ A = 45 ∧ 2 = 2)) :=
by
  sorry

end NUMINAMATH_GPT_option_A_two_solutions_l1683_168399


namespace NUMINAMATH_GPT_geometric_sequence_third_term_l1683_168349

theorem geometric_sequence_third_term :
  ∀ (a r : ℕ), a = 2 ∧ a * r ^ 3 = 162 → a * r ^ 2 = 18 :=
by
  intros a r
  intro h
  have ha : a = 2 := h.1
  have h_fourth_term : a * r ^ 3 = 162 := h.2
  sorry

end NUMINAMATH_GPT_geometric_sequence_third_term_l1683_168349


namespace NUMINAMATH_GPT_arc_length_l1683_168303

theorem arc_length (r : ℝ) (α : ℝ) (h_r : r = 10) (h_α : α = 2 * Real.pi / 3) : 
  r * α = 20 * Real.pi / 3 := 
by {
sorry
}

end NUMINAMATH_GPT_arc_length_l1683_168303


namespace NUMINAMATH_GPT_calculation_equals_106_25_l1683_168376

noncomputable def calculation : ℝ := 2.5 * 8.5 * (5.2 - 0.2)

theorem calculation_equals_106_25 : calculation = 106.25 := 
by
  sorry

end NUMINAMATH_GPT_calculation_equals_106_25_l1683_168376


namespace NUMINAMATH_GPT_triangle_interior_angle_ge_60_l1683_168301

theorem triangle_interior_angle_ge_60 (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A < 60) (h3 : B < 60) (h4 : C < 60) : false := 
by
  sorry

end NUMINAMATH_GPT_triangle_interior_angle_ge_60_l1683_168301


namespace NUMINAMATH_GPT_girls_more_than_boys_l1683_168315

variables (B G : ℕ)
def ratio_condition : Prop := 3 * G = 4 * B
def total_students_condition : Prop := B + G = 49

theorem girls_more_than_boys
  (h1 : ratio_condition B G)
  (h2 : total_students_condition B G) :
  G = B + 7 :=
sorry

end NUMINAMATH_GPT_girls_more_than_boys_l1683_168315


namespace NUMINAMATH_GPT_other_x_intercept_l1683_168385

def foci1 := (0, -3)
def foci2 := (4, 0)
def x_intercept1 := (0, 0)

theorem other_x_intercept :
  (∃ x : ℝ, (|x - 4| + |-3| * x = 7)) → x = 11 / 4 := by
  sorry

end NUMINAMATH_GPT_other_x_intercept_l1683_168385


namespace NUMINAMATH_GPT_product_is_zero_l1683_168307

theorem product_is_zero (n : ℤ) (h : n = 3) :
  (n - 3) * (n - 2) * (n - 1) * n * (n + 1) * (n + 4) = 0 := 
by
  sorry

end NUMINAMATH_GPT_product_is_zero_l1683_168307


namespace NUMINAMATH_GPT_smallest_f_for_perfect_square_l1683_168339

theorem smallest_f_for_perfect_square (f : ℕ) (h₁: 3150 = 2 * 3 * 5^2 * 7) (h₂: ∃ m : ℕ, 3150 * f = m^2) :
  f = 14 :=
sorry

end NUMINAMATH_GPT_smallest_f_for_perfect_square_l1683_168339


namespace NUMINAMATH_GPT_correct_factorization_A_l1683_168398

theorem correct_factorization_A (x : ℝ) : x^2 - 4 * x + 4 = (x - 2)^2 :=
by sorry

end NUMINAMATH_GPT_correct_factorization_A_l1683_168398


namespace NUMINAMATH_GPT_squares_with_center_35_65_l1683_168316

theorem squares_with_center_35_65 : 
  (∃ (n : ℕ), n = 1190 ∧ ∀ (x y : ℕ), x ≠ y → (x, y) = (35, 65)) :=
sorry

end NUMINAMATH_GPT_squares_with_center_35_65_l1683_168316


namespace NUMINAMATH_GPT_exists_nat_square_starting_with_digits_l1683_168373

theorem exists_nat_square_starting_with_digits (S : ℕ) : 
  ∃ (N k : ℕ), S * 10^k ≤ N^2 ∧ N^2 < (S + 1) * 10^k := 
by {
  sorry
}

end NUMINAMATH_GPT_exists_nat_square_starting_with_digits_l1683_168373


namespace NUMINAMATH_GPT_p_plus_q_identity_l1683_168396

variable {α : Type*} [CommRing α]

-- Definitions derived from conditions
def p (x : α) : α := 3 * (x - 2)
def q (x : α) : α := (x + 2) * (x - 4)

-- Lean theorem stating the problem
theorem p_plus_q_identity (x : α) : p x + q x = x^2 + x - 14 :=
by
  unfold p q
  sorry

end NUMINAMATH_GPT_p_plus_q_identity_l1683_168396


namespace NUMINAMATH_GPT_additional_investment_l1683_168372

-- Given the conditions
variables (x y : ℝ)
def interest_rate_1 := 0.02
def interest_rate_2 := 0.04
def invested_amount := 1000
def total_interest := 92

-- Theorem to prove
theorem additional_investment : 
  0.02 * invested_amount + 0.04 * (invested_amount + y) = total_interest → 
  y = 800 :=
by
  sorry

end NUMINAMATH_GPT_additional_investment_l1683_168372


namespace NUMINAMATH_GPT_sum_a1_to_a12_l1683_168323

variable {a : ℕ → ℕ}

axiom geom_seq (n : ℕ) : a n * a (n + 1) * a (n + 2) = 8
axiom a_1 : a 1 = 1
axiom a_2 : a 2 = 2

theorem sum_a1_to_a12 : (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12) = 28 :=
by
  sorry

end NUMINAMATH_GPT_sum_a1_to_a12_l1683_168323


namespace NUMINAMATH_GPT_a_2023_value_l1683_168394

theorem a_2023_value :
  ∀ (a : ℕ → ℚ),
  a 1 = 5 ∧
  a 2 = 5 / 11 ∧
  (∀ n, 3 ≤ n → a n = (a (n - 2)) * (a (n - 1)) / (3 * (a (n - 2)) - (a (n - 1)))) →
  a 2023 = 5 / 10114 ∧ 5 + 10114 = 10119 :=
by
  sorry

end NUMINAMATH_GPT_a_2023_value_l1683_168394


namespace NUMINAMATH_GPT_James_beat_record_by_72_l1683_168308

-- Define the conditions as given in the problem
def touchdowns_per_game : ℕ := 4
def points_per_touchdown : ℕ := 6
def games_in_season : ℕ := 15
def conversions : ℕ := 6
def points_per_conversion : ℕ := 2
def old_record : ℕ := 300

-- Define the necessary calculations based on the conditions
def points_from_touchdowns_per_game : ℕ := touchdowns_per_game * points_per_touchdown
def points_from_touchdowns_in_season : ℕ := games_in_season * points_from_touchdowns_per_game
def points_from_conversions : ℕ := conversions * points_per_conversion
def total_points_in_season : ℕ := points_from_touchdowns_in_season + points_from_conversions
def points_above_old_record : ℕ := total_points_in_season - old_record

-- State the proof problem
theorem James_beat_record_by_72 : points_above_old_record = 72 :=
by
  sorry

end NUMINAMATH_GPT_James_beat_record_by_72_l1683_168308


namespace NUMINAMATH_GPT_original_number_l1683_168350

theorem original_number (x : ℝ) (h : x * 1.5 = 105) : x = 70 :=
sorry

end NUMINAMATH_GPT_original_number_l1683_168350


namespace NUMINAMATH_GPT_find_x_l1683_168383

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := 
by 
sorry

end NUMINAMATH_GPT_find_x_l1683_168383


namespace NUMINAMATH_GPT_only_positive_integer_x_l1683_168319

theorem only_positive_integer_x (x : ℕ) (k : ℕ) (h1 : 2 * x + 1 = k^2) (h2 : x > 0) :
  ¬ (∃ y : ℕ, (y >= 2 * x + 2 ∧ y <= 3 * x + 2 ∧ ∃ m : ℕ, y = m^2)) → x = 4 := 
by sorry

end NUMINAMATH_GPT_only_positive_integer_x_l1683_168319


namespace NUMINAMATH_GPT_cube_volume_ratio_l1683_168352

theorem cube_volume_ratio (a b : ℝ) (h : (a^2 / b^2) = 9 / 25) :
  (b^3 / a^3) = 125 / 27 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_ratio_l1683_168352


namespace NUMINAMATH_GPT_range_of_m_l1683_168380

def A (x : ℝ) := x^2 - 3 * x - 10 ≤ 0
def B (x m : ℝ) := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem range_of_m (m : ℝ) (h : ∀ x, B x m → A x) : m ≤ 3 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l1683_168380


namespace NUMINAMATH_GPT_remainder_when_divided_by_7_l1683_168378

theorem remainder_when_divided_by_7
  (x : ℤ) (k : ℤ) (h : x = 52 * k + 19) : x % 7 = 5 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_7_l1683_168378


namespace NUMINAMATH_GPT_certain_number_plus_two_l1683_168305

theorem certain_number_plus_two (x : ℤ) (h : x - 2 = 5) : x + 2 = 9 := by
  sorry

end NUMINAMATH_GPT_certain_number_plus_two_l1683_168305


namespace NUMINAMATH_GPT_tangent_line_to_circle_l1683_168360

theorem tangent_line_to_circle :
  ∀ (x y : ℝ), x^2 + y^2 = 5 → (x = 2 → y = -1 → 2 * x - y - 5 = 0) :=
by
  intros x y h_circle hx hy
  sorry

end NUMINAMATH_GPT_tangent_line_to_circle_l1683_168360


namespace NUMINAMATH_GPT_sum_of_digits_base2_345_l1683_168310

open Nat -- open natural numbers namespace

theorem sum_of_digits_base2_345 : (Nat.digits 2 345).sum = 5 := by
  sorry -- proof to be filled in later

end NUMINAMATH_GPT_sum_of_digits_base2_345_l1683_168310


namespace NUMINAMATH_GPT_initial_avg_weight_l1683_168388

theorem initial_avg_weight (A : ℝ) (h : 6 * A + 121 = 7 * 151) : A = 156 :=
by
sorry

end NUMINAMATH_GPT_initial_avg_weight_l1683_168388


namespace NUMINAMATH_GPT_length_AB_l1683_168355

open Real

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

theorem length_AB (x1 y1 x2 y2 : ℝ) 
  (hA : y1^2 = 4 * x1) (hB : y2^2 = 4 * x2) 
  (hLine: (y2 - y1) * 1 = (x2 - x1) *0)
  (hSum : x1 + x2 = 6) : 
  dist (x1, y1) (x2, y2) = 8 := 
sorry

end NUMINAMATH_GPT_length_AB_l1683_168355


namespace NUMINAMATH_GPT_tony_lottery_winning_l1683_168347

theorem tony_lottery_winning
  (tickets : ℕ) (winning_numbers : ℕ) (worth_per_number : ℕ) (identical_numbers : Prop)
  (h_tickets : tickets = 3) (h_winning_numbers : winning_numbers = 5) (h_worth_per_number : worth_per_number = 20)
  (h_identical_numbers : identical_numbers) :
  (tickets * (winning_numbers * worth_per_number) = 300) :=
by
  sorry

end NUMINAMATH_GPT_tony_lottery_winning_l1683_168347


namespace NUMINAMATH_GPT_min_width_for_fence_area_least_200_l1683_168386

theorem min_width_for_fence_area_least_200 (w : ℝ) (h : w * (w + 20) ≥ 200) : w ≥ 10 :=
sorry

end NUMINAMATH_GPT_min_width_for_fence_area_least_200_l1683_168386


namespace NUMINAMATH_GPT_average_speed_for_trip_l1683_168366

-- Define the total distance of the trip
def total_distance : ℕ := 850

--  Define the distance and speed for the first part of the trip
def distance1 : ℕ := 400
def speed1 : ℕ := 20

-- Define the distance and speed for the remaining part of the trip
def distance2 : ℕ := 450
def speed2 : ℕ := 15

-- Define the calculated average speed for the entire trip
def average_speed : ℕ := 17

theorem average_speed_for_trip 
  (d_total : ℕ)
  (d1 : ℕ) (s1 : ℕ)
  (d2 : ℕ) (s2 : ℕ)
  (hsum : d1 + d2 = d_total)
  (d1_eq : d1 = distance1)
  (s1_eq : s1 = speed1)
  (d2_eq : d2 = distance2)
  (s2_eq : s2 = speed2) :
  (d_total / ((d1 / s1) + (d2 / s2))) = average_speed := by
  sorry

end NUMINAMATH_GPT_average_speed_for_trip_l1683_168366


namespace NUMINAMATH_GPT_inequality_solution_l1683_168309

noncomputable def g (x : ℝ) : ℝ := (3 * x - 8) * (x - 4) * (x + 1) / (x - 2)

theorem inequality_solution :
  { x : ℝ | g x ≥ 0 } = { x : ℝ | x ≤ -1 } ∪ { x : ℝ | 2 < x ∧ x ≤ 8/3 } ∪ { x : ℝ | 4 ≤ x } :=
by sorry

end NUMINAMATH_GPT_inequality_solution_l1683_168309


namespace NUMINAMATH_GPT_angle_in_third_quadrant_l1683_168329

/-- 
Given that the terminal side of angle α is in the third quadrant,
prove that the terminal side of α/3 cannot be in the second quadrant.
-/
theorem angle_in_third_quadrant (α : ℝ) (k : ℤ)
  (h : π + 2 * k * π < α ∧ α < 3 / 2 * π + 2 * k * π) :
  ¬ (π / 2 < α / 3 ∧ α / 3 < π) :=
sorry

end NUMINAMATH_GPT_angle_in_third_quadrant_l1683_168329


namespace NUMINAMATH_GPT_tom_annual_car_leasing_cost_l1683_168324

theorem tom_annual_car_leasing_cost :
  let miles_mwf := 50 * 3  -- Miles driven on Monday, Wednesday, and Friday
  let miles_other_days := 100 * 4 -- Miles driven on the other days (Sunday, Tuesday, Thursday, Saturday)
  let weekly_miles := miles_mwf + miles_other_days -- Total miles driven per week

  let cost_per_mile := 0.1 -- Cost per mile
  let weekly_fee := 100 -- Weekly fee

  let weekly_cost := weekly_miles * cost_per_mile + weekly_fee -- Total weekly cost

  let weeks_per_year := 52
  let annual_cost := weekly_cost * weeks_per_year -- Annual cost

  annual_cost = 8060 :=
by
  sorry

end NUMINAMATH_GPT_tom_annual_car_leasing_cost_l1683_168324


namespace NUMINAMATH_GPT_range_of_a_l1683_168390

theorem range_of_a (a : ℝ) : 
  {x : ℝ | x^2 - 4 * x + 3 < 0} ⊆ {x : ℝ | 2^(1 - x) + a ≤ 0 ∧ x^2 - 2 * (a + 7) * x + 5 ≤ 0} → 
  -4 ≤ a ∧ a ≤ -1 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1683_168390


namespace NUMINAMATH_GPT_solve_for_x_l1683_168393

theorem solve_for_x (x : ℤ) (h : 2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) : 
  x = -15 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1683_168393


namespace NUMINAMATH_GPT_reflection_over_line_y_eq_x_l1683_168354

theorem reflection_over_line_y_eq_x {x y x' y' : ℝ} (h_c : (x, y) = (6, -5)) (h_reflect : (x', y') = (y, x)) :
  (x', y') = (-5, 6) :=
  by
    simp [h_c, h_reflect]
    sorry

end NUMINAMATH_GPT_reflection_over_line_y_eq_x_l1683_168354


namespace NUMINAMATH_GPT_trim_hedges_purpose_l1683_168384

-- Given possible answers
inductive Answer
| A : Answer
| B : Answer
| C : Answer
| D : Answer

-- Define the purpose of trimming hedges
def trimmingHedges : Answer :=
  Answer.B

-- Formal problem statement
theorem trim_hedges_purpose : trimmingHedges = Answer.B :=
  sorry

end NUMINAMATH_GPT_trim_hedges_purpose_l1683_168384


namespace NUMINAMATH_GPT_gray_region_area_l1683_168317

theorem gray_region_area (r : ℝ) : 
  let inner_circle_radius := r
  let outer_circle_radius := r + 3
  let inner_circle_area := Real.pi * (r ^ 2)
  let outer_circle_area := Real.pi * ((r + 3) ^ 2)
  let gray_region_area := outer_circle_area - inner_circle_area
  gray_region_area = 6 * Real.pi * r + 9 * Real.pi := 
by
  sorry

end NUMINAMATH_GPT_gray_region_area_l1683_168317


namespace NUMINAMATH_GPT_mike_total_work_time_l1683_168318

theorem mike_total_work_time :
  let wash_time := 10
  let oil_change_time := 15
  let tire_change_time := 30
  let paint_time := 45
  let engine_service_time := 60

  let num_wash := 9
  let num_oil_change := 6
  let num_tire_change := 2
  let num_paint := 4
  let num_engine_service := 3
  
  let total_minutes := 
        num_wash * wash_time +
        num_oil_change * oil_change_time +
        num_tire_change * tire_change_time +
        num_paint * paint_time +
        num_engine_service * engine_service_time

  let total_hours := total_minutes / 60

  total_hours = 10 :=
  by
    -- Definitions of times per task
    let wash_time := 10
    let oil_change_time := 15
    let tire_change_time := 30
    let paint_time := 45
    let engine_service_time := 60

    -- Definitions of number of tasks performed
    let num_wash := 9
    let num_oil_change := 6
    let num_tire_change := 2
    let num_paint := 4
    let num_engine_service := 3

    -- Calculate total minutes
    let total_minutes := 
      num_wash * wash_time +
      num_oil_change * oil_change_time +
      num_tire_change * tire_change_time +
      num_paint * paint_time +
      num_engine_service * engine_service_time
    
    -- Calculate total hours
    let total_hours := total_minutes / 60

    -- Required equality to prove
    have : total_hours = 10 := sorry
    exact this

end NUMINAMATH_GPT_mike_total_work_time_l1683_168318


namespace NUMINAMATH_GPT_part1_part2_case1_part2_case2_part2_case3_l1683_168311

namespace InequalityProof

variable {a x : ℝ}

def f (a x : ℝ) := a * x^2 + x - a

theorem part1 (h : a = 1) : (x > 1 ∨ x < -2) → f a x > 1 :=
by sorry

theorem part2_case1 (h1 : a < 0) (h2 : a < -1/2) : (- (a + 1) / a) < x ∧ x < 1 → f a x > 1 :=
by sorry

theorem part2_case2 (h1 : a < 0) (h2 : a = -1/2) : x ≠ 1 → f a x > 1 :=
by sorry

theorem part2_case3 (h1 : a < 0) (h2 : 0 > a) (h3 : a > -1/2) : 1 < x ∧ x < - (a + 1) / a → f a x > 1 :=
by sorry

end InequalityProof

end NUMINAMATH_GPT_part1_part2_case1_part2_case2_part2_case3_l1683_168311


namespace NUMINAMATH_GPT_sin_C_eq_63_over_65_l1683_168397

theorem sin_C_eq_63_over_65 (A B C : Real) (h₁ : 0 < A) (h₂ : A < π)
  (h₃ : 0 < B) (h₄ : B < π) (h₅ : 0 < C) (h₆ : C < π)
  (h₇ : A + B + C = π)
  (h₈ : Real.sin A = 5 / 13) (h₉ : Real.cos B = 3 / 5) : Real.sin C = 63 / 65 := 
by
  sorry

end NUMINAMATH_GPT_sin_C_eq_63_over_65_l1683_168397
