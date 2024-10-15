import Mathlib

namespace NUMINAMATH_GPT_largest_divisor_of_consecutive_odd_integers_l1802_180206

theorem largest_divisor_of_consecutive_odd_integers :
  ∀ (x : ℤ), (∃ (d : ℤ) (m : ℤ), d = 48 ∧ (x * (x + 2) * (x + 4) * (x + 6)) = d * m) :=
by 
-- We assert that for any integer x, 48 always divides the product of
-- four consecutive odd integers starting from x
sorry

end NUMINAMATH_GPT_largest_divisor_of_consecutive_odd_integers_l1802_180206


namespace NUMINAMATH_GPT_total_bills_54_l1802_180287

/-- A bank teller has some 5-dollar and 20-dollar bills in her cash drawer, 
and the total value of the bills is 780 dollars, with 20 5-dollar bills.
Show that the total number of bills is 54. -/
theorem total_bills_54 (value_total : ℕ) (num_5dollar : ℕ) (num_5dollar_value : ℕ) (num_20dollar : ℕ) :
    value_total = 780 ∧ num_5dollar = 20 ∧ num_5dollar_value = 5 ∧ num_20dollar * 20 + num_5dollar * num_5dollar_value = value_total
    → num_20dollar + num_5dollar = 54 :=
by
  sorry

end NUMINAMATH_GPT_total_bills_54_l1802_180287


namespace NUMINAMATH_GPT_canoe_speed_downstream_l1802_180223

theorem canoe_speed_downstream (V_upstream V_s V_c V_downstream : ℝ) 
    (h1 : V_upstream = 6) 
    (h2 : V_s = 2) 
    (h3 : V_upstream = V_c - V_s) 
    (h4 : V_downstream = V_c + V_s) : 
  V_downstream = 10 := 
by 
  sorry

end NUMINAMATH_GPT_canoe_speed_downstream_l1802_180223


namespace NUMINAMATH_GPT_xiao_yu_reading_days_l1802_180286

-- Definition of Xiao Yu's reading problem
def number_of_pages_per_day := 15
def total_number_of_days := 24
def additional_pages_per_day := 3
def new_number_of_pages_per_day := number_of_pages_per_day + additional_pages_per_day
def total_pages := number_of_pages_per_day * total_number_of_days
def new_total_number_of_days := total_pages / new_number_of_pages_per_day

-- Theorem statement in Lean 4
theorem xiao_yu_reading_days : new_total_number_of_days = 20 :=
  sorry

end NUMINAMATH_GPT_xiao_yu_reading_days_l1802_180286


namespace NUMINAMATH_GPT_exists_natural_number_n_l1802_180239

theorem exists_natural_number_n (t : ℕ) (ht : t > 0) :
  ∃ n : ℕ, n > 1 ∧ Nat.gcd n t = 1 ∧ ∀ k : ℕ, k > 0 → ∃ m : ℕ, m > 1 → n^k + t ≠ m^m :=
by
  sorry

end NUMINAMATH_GPT_exists_natural_number_n_l1802_180239


namespace NUMINAMATH_GPT_compare_logs_l1802_180246

noncomputable def a : ℝ := Real.log 2
noncomputable def b : ℝ := Real.logb 2 3
noncomputable def c : ℝ := Real.logb 5 8

theorem compare_logs : a < c ∧ c < b := by
  sorry

end NUMINAMATH_GPT_compare_logs_l1802_180246


namespace NUMINAMATH_GPT_order_of_numbers_l1802_180281

theorem order_of_numbers :
  2^30 < 10^10 ∧ 10^10 < 5^15 :=
by
  sorry

end NUMINAMATH_GPT_order_of_numbers_l1802_180281


namespace NUMINAMATH_GPT_divisibility_by_24_l1802_180205

theorem divisibility_by_24 (n : ℤ) : 24 ∣ n * (n + 2) * (5 * n - 1) * (5 * n + 1) :=
sorry

end NUMINAMATH_GPT_divisibility_by_24_l1802_180205


namespace NUMINAMATH_GPT_expected_audience_l1802_180235

theorem expected_audience (Sat Mon Wed Fri : ℕ) (extra_people expected_total : ℕ)
  (h1 : Sat = 80)
  (h2 : Mon = 80 - 20)
  (h3 : Wed = Mon + 50)
  (h4 : Fri = Sat + Mon)
  (h5 : extra_people = 40)
  (h6 : expected_total = Sat + Mon + Wed + Fri - extra_people) :
  expected_total = 350 := 
sorry

end NUMINAMATH_GPT_expected_audience_l1802_180235


namespace NUMINAMATH_GPT_find_k_l1802_180269

theorem find_k (k : ℚ) (h : ∃ k : ℚ, (3 * (4 - k) = 2 * (-5 - 3))): k = -4 / 3 := by
  sorry

end NUMINAMATH_GPT_find_k_l1802_180269


namespace NUMINAMATH_GPT_tom_found_dimes_l1802_180272

theorem tom_found_dimes :
  let quarters := 10
  let nickels := 4
  let pennies := 200
  let total_value := 5
  let value_quarters := 0.25 * quarters
  let value_nickels := 0.05 * nickels
  let value_pennies := 0.01 * pennies
  let total_other := value_quarters + value_nickels + value_pennies
  let value_dimes := total_value - total_other
  value_dimes / 0.10 = 3 := sorry

end NUMINAMATH_GPT_tom_found_dimes_l1802_180272


namespace NUMINAMATH_GPT_sum_in_Q_l1802_180245

open Set

def is_set_P (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k
def is_set_Q (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k - 1
def is_set_M (x : ℤ) : Prop := ∃ k : ℤ, x = 4 * k + 1

variables (a b : ℤ)

theorem sum_in_Q (ha : is_set_P a) (hb : is_set_Q b) : is_set_Q (a + b) := 
sorry

end NUMINAMATH_GPT_sum_in_Q_l1802_180245


namespace NUMINAMATH_GPT_bricks_needed_for_wall_l1802_180213

noncomputable def number_of_bricks_needed
    (brick_length : ℕ)
    (brick_width : ℕ)
    (brick_height : ℕ)
    (wall_length_m : ℕ)
    (wall_height_m : ℕ)
    (wall_thickness_cm : ℕ) : ℕ :=
  let wall_length_cm := wall_length_m * 100
  let wall_height_cm := wall_height_m * 100
  let wall_volume := wall_length_cm * wall_height_cm * wall_thickness_cm
  let brick_volume := brick_length * brick_width * brick_height
  (wall_volume + brick_volume - 1) / brick_volume -- This rounds up to the nearest whole number.

theorem bricks_needed_for_wall : number_of_bricks_needed 5 11 6 8 6 2 = 2910 :=
sorry

end NUMINAMATH_GPT_bricks_needed_for_wall_l1802_180213


namespace NUMINAMATH_GPT_smallest_enclosing_sphere_radius_l1802_180285

-- Define the conditions
def sphere_radius : ℝ := 2

-- Define the sphere center coordinates in each octant
def sphere_centers : List (ℝ × ℝ × ℝ) :=
  [ (2, 2, 2), (2, 2, -2), (2, -2, 2), (2, -2, -2),
    (-2, 2, 2), (-2, 2, -2), (-2, -2, 2), (-2, -2, -2) ]

-- Define the theorem statement
theorem smallest_enclosing_sphere_radius :
  (∃ (r : ℝ), r = 2 * Real.sqrt 3 + 2) :=
by
  -- conditions and proof will go here
  sorry

end NUMINAMATH_GPT_smallest_enclosing_sphere_radius_l1802_180285


namespace NUMINAMATH_GPT_exponential_inequality_example_l1802_180293

theorem exponential_inequality_example (a b : ℝ) (h : 1.5 > 0 ∧ 1.5 ≠ 1) (h2 : 2.3 < 3.2) : 1.5 ^ 2.3 < 1.5 ^ 3.2 :=
by 
  sorry

end NUMINAMATH_GPT_exponential_inequality_example_l1802_180293


namespace NUMINAMATH_GPT_gcd_m_l1802_180230

def m' : ℕ := 33333333
def n' : ℕ := 555555555

theorem gcd_m'_n' : Nat.gcd m' n' = 3 := by
  sorry

end NUMINAMATH_GPT_gcd_m_l1802_180230


namespace NUMINAMATH_GPT_complement_of_A_in_U_intersection_of_A_and_B_union_of_A_and_B_union_of_complements_of_A_and_B_l1802_180249

-- Definitions of the sets U, A, B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 5}

-- Complement of a set
def C_A : Set ℕ := U \ A
def C_B : Set ℕ := U \ B

-- Questions rephrased as theorem statements
theorem complement_of_A_in_U : C_A = {2, 4, 5} := by sorry
theorem intersection_of_A_and_B : A ∩ B = ∅ := by sorry
theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 5} := by sorry
theorem union_of_complements_of_A_and_B : C_A ∪ C_B = U := by sorry

end NUMINAMATH_GPT_complement_of_A_in_U_intersection_of_A_and_B_union_of_A_and_B_union_of_complements_of_A_and_B_l1802_180249


namespace NUMINAMATH_GPT_find_AC_l1802_180253

theorem find_AC (AB DC AD : ℕ) (hAB : AB = 13) (hDC : DC = 20) (hAD : AD = 5) : 
  AC = 24.2 := 
sorry

end NUMINAMATH_GPT_find_AC_l1802_180253


namespace NUMINAMATH_GPT_four_lines_set_l1802_180208

-- Define the ⬩ operation
def clubsuit (a b : ℝ) := a^3 * b - a * b^3

-- Define the main theorem
theorem four_lines_set (x y : ℝ) : 
  (clubsuit x y = clubsuit y x) ↔ (y = 0 ∨ x = 0 ∨ y = x ∨ y = -x) :=
by sorry

end NUMINAMATH_GPT_four_lines_set_l1802_180208


namespace NUMINAMATH_GPT_no_positive_integer_makes_sum_prime_l1802_180242

theorem no_positive_integer_makes_sum_prime : ¬ ∃ n : ℕ, 0 < n ∧ Prime (4^n + n^4) :=
by
  sorry

end NUMINAMATH_GPT_no_positive_integer_makes_sum_prime_l1802_180242


namespace NUMINAMATH_GPT_solve_for_k_l1802_180255

theorem solve_for_k (k : ℝ) (h : 2 * (5:ℝ)^2 + 3 * (5:ℝ) - k = 0) : k = 65 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_k_l1802_180255


namespace NUMINAMATH_GPT_rectangle_length_l1802_180203

theorem rectangle_length (side_length_square : ℝ) (width_rectangle : ℝ) (area_equal : ℝ) 
  (square_area : side_length_square * side_length_square = area_equal) 
  (rectangle_area : width_rectangle * (width_rectangle * length) = area_equal) : 
  length = 24 :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_length_l1802_180203


namespace NUMINAMATH_GPT_max_days_for_same_shift_l1802_180220

open BigOperators

-- We define the given conditions
def nurses : ℕ := 15
def shifts_per_day : ℕ := 24 / 8
noncomputable def total_pairs : ℕ := (nurses.choose 2)

-- The main statement to prove
theorem max_days_for_same_shift : 
  35 = total_pairs / shifts_per_day := by
  sorry

end NUMINAMATH_GPT_max_days_for_same_shift_l1802_180220


namespace NUMINAMATH_GPT_weight_measurement_l1802_180256

theorem weight_measurement :
  ∀ (w : Set ℕ), w = {1, 3, 9, 27} → (∀ n ∈ w, ∃ k, k = n ∧ k ∈ w) →
  ∃ (num_sets : ℕ), num_sets = 41 := by
  intros w hw hcomb
  sorry

end NUMINAMATH_GPT_weight_measurement_l1802_180256


namespace NUMINAMATH_GPT_train_speed_l1802_180214

theorem train_speed (distance time : ℝ) (h₁ : distance = 240) (h₂ : time = 4) : 
  ((distance / time) * 3.6) = 216 := 
by 
  rw [h₁, h₂] 
  sorry

end NUMINAMATH_GPT_train_speed_l1802_180214


namespace NUMINAMATH_GPT_volleyball_tournament_l1802_180294

theorem volleyball_tournament (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
sorry

end NUMINAMATH_GPT_volleyball_tournament_l1802_180294


namespace NUMINAMATH_GPT_gcd_of_ten_digit_same_five_digit_l1802_180295

def ten_digit_same_five_digit (n : ℕ) : Prop :=
  n > 9999 ∧ n < 100000 ∧ ∃ k : ℕ, k = n * (10^10 + 10^5 + 1)

theorem gcd_of_ten_digit_same_five_digit :
  (∀ n : ℕ, ten_digit_same_five_digit n → ∃ d : ℕ, d = 10000100001 ∧ ∀ m : ℕ, m ∣ d) := 
sorry

end NUMINAMATH_GPT_gcd_of_ten_digit_same_five_digit_l1802_180295


namespace NUMINAMATH_GPT_stock_and_bond_value_relation_l1802_180241

-- Definitions for conditions
def more_valuable_shares : ℕ := 14
def less_valuable_shares : ℕ := 26
def face_value_bond : ℝ := 1000
def coupon_rate_bond : ℝ := 0.06
def discount_rate_bond : ℝ := 0.03
def total_assets_value : ℝ := 2106

-- Lean statement for the proof problem
theorem stock_and_bond_value_relation (x y : ℝ) 
    (h1 : face_value_bond * (1 - discount_rate_bond) = 970)
    (h2 : 27 * x + y = total_assets_value) :
    y = 2106 - 27 * x :=
by
  sorry

end NUMINAMATH_GPT_stock_and_bond_value_relation_l1802_180241


namespace NUMINAMATH_GPT_cake_icing_l1802_180233

/-- Define the cake conditions -/
structure Cake :=
  (dimension : ℕ)
  (small_cube_dimension : ℕ)
  (total_cubes : ℕ)
  (iced_faces : ℕ)

/-- Define the main theorem to prove the number of smaller cubes with icing on exactly two sides -/
theorem cake_icing (c : Cake) : 
  c.dimension = 5 ∧ c.small_cube_dimension = 1 ∧ c.total_cubes = 125 ∧ c.iced_faces = 4 →
  ∃ n, n = 20 :=
by
  sorry

end NUMINAMATH_GPT_cake_icing_l1802_180233


namespace NUMINAMATH_GPT_isabella_hair_length_l1802_180264

-- Define the conditions and the question in Lean
def current_length : ℕ := 9
def length_cut_off : ℕ := 9

-- Main theorem statement
theorem isabella_hair_length 
  (current_length : ℕ) 
  (length_cut_off : ℕ) 
  (H1 : current_length = 9) 
  (H2 : length_cut_off = 9) : 
  current_length + length_cut_off = 18 :=
  sorry

end NUMINAMATH_GPT_isabella_hair_length_l1802_180264


namespace NUMINAMATH_GPT_identify_worst_player_l1802_180236

-- Define the participants
inductive Participant
| father
| sister
| son
| daughter

open Participant

-- Conditions
def participants : List Participant :=
  [father, sister, son, daughter]

def twins (p1 p2 : Participant) : Prop := 
  (p1 = father ∧ p2 = sister) ∨
  (p1 = sister ∧ p2 = father) ∨
  (p1 = son ∧ p2 = daughter) ∨
  (p1 = daughter ∧ p2 = son)

def not_same_sex (p1 p2 : Participant) : Prop :=
  (p1 = father ∧ p2 = sister) ∨
  (p1 = sister ∧ p2 = father) ∨
  (p1 = son ∧ p2 = daughter) ∨
  (p1 = daughter ∧ p2 = son)

def older_by_one_year (p1 p2 : Participant) : Prop :=
  (p1 = father ∧ p2 = sister) ∨
  (p1 = sister ∧ p2 = father)

-- Question: who is the worst player?
def worst_player : Participant := sister

-- Proof statement
theorem identify_worst_player
  (h_twins : ∃ p1 p2, twins p1 p2)
  (h_not_same_sex : ∀ p1 p2, twins p1 p2 → not_same_sex p1 p2)
  (h_age_diff : ∀ p1 p2, twins p1 p2 → older_by_one_year p1 p2) :
  worst_player = sister :=
sorry

end NUMINAMATH_GPT_identify_worst_player_l1802_180236


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l1802_180222

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ)
  (h_d : d ≠ 0)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_S : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * d)
  (h_geo : (a 1 + 2 * d) ^ 2 = a 1 * (a 1 + 3 * d)) :
  (S 4 - S 2) / (S 5 - S 3) = 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l1802_180222


namespace NUMINAMATH_GPT_taxi_trip_distance_l1802_180211

theorem taxi_trip_distance
  (initial_fee : ℝ)
  (per_segment_charge : ℝ)
  (segment_distance : ℝ)
  (total_charge : ℝ)
  (segments_traveled : ℝ)
  (total_miles : ℝ) :
  initial_fee = 2.25 →
  per_segment_charge = 0.3 →
  segment_distance = 2/5 →
  total_charge = 4.95 →
  total_miles = segments_traveled * segment_distance →
  segments_traveled = (total_charge - initial_fee) / per_segment_charge →
  total_miles = 3.6 :=
by
  intros h_initial_fee h_per_segment_charge h_segment_distance h_total_charge h_total_miles h_segments_traveled
  sorry

end NUMINAMATH_GPT_taxi_trip_distance_l1802_180211


namespace NUMINAMATH_GPT_factorizable_trinomial_l1802_180266

theorem factorizable_trinomial (k : ℤ) : (∃ a b : ℤ, a + b = k ∧ a * b = 5) ↔ (k = 6 ∨ k = -6) :=
by
  sorry

end NUMINAMATH_GPT_factorizable_trinomial_l1802_180266


namespace NUMINAMATH_GPT_problem_1_problem_2_l1802_180290

theorem problem_1 (p x : ℝ) (h1 : |p| ≤ 2) (h2 : x^2 + p*x + 1 > 2*x + p) : x < -1 ∨ x > 3 :=
sorry

theorem problem_2 (p x : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ 4) (h3 : x^2 + p*x + 1 > 2*x + p) : p > -1 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1802_180290


namespace NUMINAMATH_GPT_find_number_l1802_180271

-- Define the conditions
def satisfies_condition (x : ℝ) : Prop := x * 4 * 25 = 812

-- The main theorem stating that the number satisfying the condition is 8.12
theorem find_number (x : ℝ) (h : satisfies_condition x) : x = 8.12 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1802_180271


namespace NUMINAMATH_GPT_tetrahedron_volume_l1802_180296

theorem tetrahedron_volume 
  (R S₁ S₂ S₃ S₄ : ℝ) : 
  V = R * (S₁ + S₂ + S₃ + S₄) :=
sorry

end NUMINAMATH_GPT_tetrahedron_volume_l1802_180296


namespace NUMINAMATH_GPT_diamond_cut_1_3_loss_diamond_max_loss_ratio_l1802_180278

noncomputable def value (w : ℝ) : ℝ := 6000 * w^2

theorem diamond_cut_1_3_loss (a : ℝ) :
  (value a - (value (1/4 * a) + value (3/4 * a))) / value a = 0.375 :=
by sorry

theorem diamond_max_loss_ratio :
  ∀ (m n : ℝ), (m > 0) → (n > 0) → 
  (1 - (value (m/(m + n)) + value (n/(m + n))) ≤ 0.5) :=
by sorry

end NUMINAMATH_GPT_diamond_cut_1_3_loss_diamond_max_loss_ratio_l1802_180278


namespace NUMINAMATH_GPT_find_x_if_opposites_l1802_180279

theorem find_x_if_opposites (x : ℝ) (h : 2 * (x - 3) = - 4 * (1 - x)) : x = -1 := 
by
  sorry

end NUMINAMATH_GPT_find_x_if_opposites_l1802_180279


namespace NUMINAMATH_GPT_tangent_lines_from_point_to_circle_l1802_180216

theorem tangent_lines_from_point_to_circle : 
  ∀ (P : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ), 
  P = (2, 3) → C = (1, 1) → r = 1 → 
  (∃ k : ℝ, ((3 : ℝ) * P.1 - (4 : ℝ) * P.2 + 6 = 0) ∨ (P.1 = 2)) :=
by
  intros P C r hP hC hr
  sorry

end NUMINAMATH_GPT_tangent_lines_from_point_to_circle_l1802_180216


namespace NUMINAMATH_GPT_orchestra_members_l1802_180270

theorem orchestra_members : ∃ (x : ℕ), (130 < x) ∧ (x < 260) ∧ (x % 6 = 1) ∧ (x % 5 = 2) ∧ (x % 7 = 3) ∧ (x = 241) :=
by
  sorry

end NUMINAMATH_GPT_orchestra_members_l1802_180270


namespace NUMINAMATH_GPT_hummus_serving_amount_proof_l1802_180297

/-- Given conditions: 
    one_can is the number of ounces of chickpeas in one can,
    total_cans is the number of cans Thomas buys,
    total_servings is the number of servings of hummus Thomas needs to make,
    to_produce_one_serving is the amount of chickpeas needed for one serving,
    we prove that to_produce_one_serving = 6.4 given the above conditions. -/
theorem hummus_serving_amount_proof 
  (one_can : ℕ) 
  (total_cans : ℕ) 
  (total_servings : ℕ) 
  (to_produce_one_serving : ℚ) 
  (h_one_can : one_can = 16) 
  (h_total_cans : total_cans = 8)
  (h_total_servings : total_servings = 20) 
  (h_total_ounces : total_cans * one_can = 128) : 
  to_produce_one_serving = 128 / 20 := 
by
  sorry

end NUMINAMATH_GPT_hummus_serving_amount_proof_l1802_180297


namespace NUMINAMATH_GPT_number_of_partners_l1802_180231

def total_profit : ℝ := 80000
def majority_owner_share := 0.25 * total_profit
def remaining_profit := total_profit - majority_owner_share
def partner_share := 0.25 * remaining_profit
def combined_share := majority_owner_share + 2 * partner_share

theorem number_of_partners : combined_share = 50000 → remaining_profit / partner_share = 4 := by
  intro h1
  have h_majority : majority_owner_share = 0.25 * total_profit := by sorry
  have h_remaining : remaining_profit = total_profit - majority_owner_share := by sorry
  have h_partner : partner_share = 0.25 * remaining_profit := by sorry
  have h_combined : combined_share = majority_owner_share + 2 * partner_share := by sorry
  calc
    remaining_profit / partner_share = _ := by sorry
    4 = 4 := by sorry

end NUMINAMATH_GPT_number_of_partners_l1802_180231


namespace NUMINAMATH_GPT_sequence_a_n_term_l1802_180283

theorem sequence_a_n_term :
  ∃ a : ℕ → ℕ, 
  a 1 = 1 ∧
  (∀ n : ℕ, a (n+1) = 2 * a n + 1) ∧
  a 10 = 1023 := by
  sorry

end NUMINAMATH_GPT_sequence_a_n_term_l1802_180283


namespace NUMINAMATH_GPT_determine_f4_l1802_180215

def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem determine_f4 (f : ℝ → ℝ) (h_odd : odd_function f) (h_f_neg : ∀ x, x < 0 → f x = x * (2 - x)) : f 4 = 24 :=
by
  sorry

end NUMINAMATH_GPT_determine_f4_l1802_180215


namespace NUMINAMATH_GPT_radius_of_circle_l1802_180292

theorem radius_of_circle
  (d PQ QR : ℝ) (h1 : d = 15) (h2 : PQ = 7) (h3 : QR = 8) :
  ∃ r : ℝ, r = 2 * Real.sqrt 30 ∧ (PQ * (PQ + QR) = (d - r) * (d + r)) :=
by
  -- All necessary non-proof related statements
  sorry

end NUMINAMATH_GPT_radius_of_circle_l1802_180292


namespace NUMINAMATH_GPT_replacement_paint_intensity_l1802_180218

theorem replacement_paint_intensity 
  (P_original : ℝ) (P_new : ℝ) (f : ℝ) (I : ℝ) :
  P_original = 50 →
  P_new = 45 →
  f = 0.2 →
  0.8 * P_original + f * I = P_new →
  I = 25 :=
by
  intros
  sorry

end NUMINAMATH_GPT_replacement_paint_intensity_l1802_180218


namespace NUMINAMATH_GPT_vector_solution_l1802_180262

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_solution (a x : V) (h : 2 • x - 3 • (x - 2 • a) = 0) : x = 6 • a :=
by sorry

end NUMINAMATH_GPT_vector_solution_l1802_180262


namespace NUMINAMATH_GPT_sum_of_z_values_l1802_180238

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 2

theorem sum_of_z_values (z : ℝ) : 
  (f (4 * z) = 13) → (∃ z1 z2 : ℝ, z1 = 1/8 ∧ z2 = -1/4 ∧ z1 + z2 = -1/8) :=
sorry

end NUMINAMATH_GPT_sum_of_z_values_l1802_180238


namespace NUMINAMATH_GPT_fully_filled_boxes_l1802_180284

theorem fully_filled_boxes (total_cards : ℕ) (cards_per_box : ℕ) (h1 : total_cards = 94) (h2 : cards_per_box = 8) : total_cards / cards_per_box = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_fully_filled_boxes_l1802_180284


namespace NUMINAMATH_GPT_functional_equation_unique_solution_l1802_180258

theorem functional_equation_unique_solution (f : ℝ → ℝ) :
  (∀ a b c : ℝ, a + f b + f (f c) = 0 → f a ^ 3 + b * f b ^ 2 + c ^ 2 * f c = 3 * a * b * c) →
  (∀ x : ℝ, f x = x ∨ f x = -x ∨ f x = 0) :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_unique_solution_l1802_180258


namespace NUMINAMATH_GPT_pictures_deleted_l1802_180228

theorem pictures_deleted (zoo_pics museum_pics remaining_pics : ℕ) 
  (h1 : zoo_pics = 15) 
  (h2 : museum_pics = 18) 
  (h3 : remaining_pics = 2) : 
  zoo_pics + museum_pics - remaining_pics = 31 :=
by 
  sorry

end NUMINAMATH_GPT_pictures_deleted_l1802_180228


namespace NUMINAMATH_GPT_arithmetic_sequence_minimization_l1802_180289

theorem arithmetic_sequence_minimization (a b : ℕ) (h_range : 1 ≤ a ∧ b ≤ 17) (h_seq : a + b = 18) (h_min : ∀ x y, (1 ≤ x ∧ y ≤ 17 ∧ x + y = 18) → (1 / x + 25 / y) ≥ (1 / a + 25 / b)) : ∃ n : ℕ, n = 9 :=
by
  -- We'd usually follow by proving the conditions and defining the sequence correctly.
  -- Definitions and steps leading to finding n = 9 will be elaborated here.
  -- This placeholder is to satisfy the requirement only.
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_minimization_l1802_180289


namespace NUMINAMATH_GPT_graph_passes_through_quadrants_l1802_180204

theorem graph_passes_through_quadrants :
  ∀ x : ℝ, (4 * x + 2 > 0 → (x > 0)) ∨ (4 * x + 2 > 0 → (x < 0)) ∨ (4 * x + 2 < 0 → (x < 0)) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_graph_passes_through_quadrants_l1802_180204


namespace NUMINAMATH_GPT_enchilada_taco_cost_l1802_180273

variables (e t : ℝ)

theorem enchilada_taco_cost 
  (h1 : 4 * e + 5 * t = 4.00) 
  (h2 : 5 * e + 3 * t = 3.80) 
  (h3 : 7 * e + 6 * t = 6.10) : 
  4 * e + 7 * t = 4.75 := 
sorry

end NUMINAMATH_GPT_enchilada_taco_cost_l1802_180273


namespace NUMINAMATH_GPT_shortest_time_between_ships_l1802_180232

theorem shortest_time_between_ships 
  (AB : ℝ) (speed_A : ℝ) (speed_B : ℝ) (angle_ABA' : ℝ) : (AB = 10) → (speed_A = 4) → (speed_B = 6) → (angle_ABA' = 60) →
  ∃ t : ℝ, (t = 150/7 / 60) :=
by
  intro hAB hSpeedA hSpeedB hAngle
  sorry

end NUMINAMATH_GPT_shortest_time_between_ships_l1802_180232


namespace NUMINAMATH_GPT_geometric_sequence_property_l1802_180298

variables {a : ℕ → ℝ} {S : ℕ → ℝ}

noncomputable def a_n (n : ℕ) : ℝ := 2 * 3^(n - 1)
noncomputable def S_n (n : ℕ) : ℝ := 
  if n = 0 then 0
  else (2 * (1 - 3^n)) / (1 - 3)

theorem geometric_sequence_property 
  (h₁ : a 1 + a 2 + a 3 = 26)
  (h₂ : S 6 = 728)
  (h₃ : ∀ n, a n = a_n n)
  (h₄ : ∀ n, S n = S_n n) :
  ∀ n, S (n + 1) ^ 2 - S n * S (n + 2) = 4 * 3 ^ n :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_property_l1802_180298


namespace NUMINAMATH_GPT_expand_product_l1802_180263

theorem expand_product (x : ℝ) : (x^2 - 2*x + 2) * (x^2 + 2*x + 2) = x^4 + 4 :=
by
  sorry

end NUMINAMATH_GPT_expand_product_l1802_180263


namespace NUMINAMATH_GPT_determine_plane_by_trapezoid_legs_l1802_180288

-- Defining basic objects
structure Point := (x : ℝ) (y : ℝ) (z : ℝ)
structure Line := (p1 : Point) (p2 : Point)
structure Plane := (l1 : Line) (l2 : Line)

-- Theorem statement for the problem
theorem determine_plane_by_trapezoid_legs (trapezoid_legs : Line) :
  ∃ (pl : Plane), ∀ (l1 l2 : Line), (l1 = trapezoid_legs) ∧ (l2 = trapezoid_legs) → (pl = Plane.mk l1 l2) :=
sorry

end NUMINAMATH_GPT_determine_plane_by_trapezoid_legs_l1802_180288


namespace NUMINAMATH_GPT_zeros_of_quadratic_l1802_180210

theorem zeros_of_quadratic : ∃ x : ℝ, x^2 - x - 2 = 0 -> (x = -1 ∨ x = 2) :=
by
  sorry

end NUMINAMATH_GPT_zeros_of_quadratic_l1802_180210


namespace NUMINAMATH_GPT_Kaleb_got_rid_of_7_shirts_l1802_180257

theorem Kaleb_got_rid_of_7_shirts (initial_shirts : ℕ) (remaining_shirts : ℕ) 
    (h1 : initial_shirts = 17) (h2 : remaining_shirts = 10) : initial_shirts - remaining_shirts = 7 := 
by
  sorry

end NUMINAMATH_GPT_Kaleb_got_rid_of_7_shirts_l1802_180257


namespace NUMINAMATH_GPT_min_value_expression_l1802_180280

open Real

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4) :
  ∃ z : ℝ, z = 16 / 7 ∧ ∀ u > 0, ∀ v > 0, u + v = 4 → ((u^2 / (u + 1)) + (v^2 / (v + 2))) ≥ z :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1802_180280


namespace NUMINAMATH_GPT_range_of_a_l1802_180240

noncomputable def f (x a : ℝ) : ℝ := (Real.sqrt x) / (x^3 - 3 * x + a)

theorem range_of_a (a : ℝ) :
    (∀ x, 0 ≤ x → x^3 - 3 * x + a ≠ 0) ↔ 2 < a := 
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l1802_180240


namespace NUMINAMATH_GPT_least_comic_books_l1802_180207

theorem least_comic_books (n : ℕ) (h1 : n % 7 = 3) (h2 : n % 4 = 1) : n = 17 :=
sorry

end NUMINAMATH_GPT_least_comic_books_l1802_180207


namespace NUMINAMATH_GPT_negation_equivalence_l1802_180277

-- Define the angles in a triangle as three real numbers
def is_triangle (a b c : ℝ) : Prop :=
  a + b + c = 180 ∧ a > 0 ∧ b > 0 ∧ c > 0

-- Define the proposition
def at_least_one_angle_not_greater_than_60 (a b c : ℝ) : Prop :=
  a ≤ 60 ∨ b ≤ 60 ∨ c ≤ 60

-- Negate the proposition
def all_angles_greater_than_60 (a b c : ℝ) : Prop :=
  a > 60 ∧ b > 60 ∧ c > 60

-- Prove that the negation of the proposition is equivalent
theorem negation_equivalence (a b c : ℝ) (h_triangle : is_triangle a b c) :
  ¬ at_least_one_angle_not_greater_than_60 a b c ↔ all_angles_greater_than_60 a b c :=
by
  sorry

end NUMINAMATH_GPT_negation_equivalence_l1802_180277


namespace NUMINAMATH_GPT_initial_weight_cucumbers_l1802_180217

theorem initial_weight_cucumbers (W : ℝ) (h1 : 0.99 * W + 0.01 * W = W) 
                                  (h2 : W = (50 - 0.98 * 50 + 0.01 * W))
                                  (h3 : 50 > 0) : W = 100 := 
sorry

end NUMINAMATH_GPT_initial_weight_cucumbers_l1802_180217


namespace NUMINAMATH_GPT_difference_is_correct_l1802_180219

-- Define the digits
def digits : List ℕ := [9, 2, 1, 5]

-- Define the largest number that can be formed by these digits
def largestNumber : ℕ :=
  1000 * 9 + 100 * 5 + 10 * 2 + 1 * 1

-- Define the smallest number that can be formed by these digits
def smallestNumber : ℕ :=
  1000 * 1 + 100 * 2 + 10 * 5 + 1 * 9

-- Define the correct difference
def difference : ℕ :=
  largestNumber - smallestNumber

-- Theorem statement
theorem difference_is_correct : difference = 8262 :=
by
  sorry

end NUMINAMATH_GPT_difference_is_correct_l1802_180219


namespace NUMINAMATH_GPT_arrange_magnitudes_l1802_180212

theorem arrange_magnitudes (x : ℝ) (hx : 0.8 < x ∧ x < 0.9) :
  let y := x^x
  let z := x^(x^x)
  x < z ∧ z < y := by
  sorry

end NUMINAMATH_GPT_arrange_magnitudes_l1802_180212


namespace NUMINAMATH_GPT_angle_of_inclination_l1802_180268

-- The statement of the mathematically equivalent proof problem in Lean 4
theorem angle_of_inclination
  (k: ℝ)
  (α: ℝ)
  (line_eq: ∀ x, ∃ y, y = (k-1) * x + 2)
  (circle_eq: ∀ x y, x^2 + y^2 + k * x + 2 * y + k^2 = 0) :
  α = 3 * Real.pi / 4 :=
sorry -- Proof to be provided

end NUMINAMATH_GPT_angle_of_inclination_l1802_180268


namespace NUMINAMATH_GPT_number_of_distinguishable_arrangements_l1802_180243

-- Define the conditions
def num_blue_tiles : Nat := 1
def num_red_tiles : Nat := 2
def num_green_tiles : Nat := 3
def num_yellow_tiles : Nat := 2
def total_tiles : Nat := num_blue_tiles + num_red_tiles + num_green_tiles + num_yellow_tiles

-- The goal is to prove the number of distinguishable arrangements
theorem number_of_distinguishable_arrangements : 
  (Nat.factorial total_tiles) / ((Nat.factorial num_green_tiles) * 
                                (Nat.factorial num_red_tiles) * 
                                (Nat.factorial num_yellow_tiles) * 
                                (Nat.factorial num_blue_tiles)) = 1680 := by
  sorry

end NUMINAMATH_GPT_number_of_distinguishable_arrangements_l1802_180243


namespace NUMINAMATH_GPT_meryll_remaining_questions_l1802_180265

variables (total_mc total_ps total_tf : ℕ)
variables (frac_mc frac_ps frac_tf : ℚ)

-- Conditions as Lean definitions:
def written_mc (total_mc : ℕ) (frac_mc : ℚ) := (frac_mc * total_mc).floor
def written_ps (total_ps : ℕ) (frac_ps : ℚ) := (frac_ps * total_ps).floor
def written_tf (total_tf : ℕ) (frac_tf : ℚ) := (frac_tf * total_tf).floor

def remaining_mc (total_mc : ℕ) (frac_mc : ℚ) := total_mc - written_mc total_mc frac_mc
def remaining_ps (total_ps : ℕ) (frac_ps : ℚ) := total_ps - written_ps total_ps frac_ps
def remaining_tf (total_tf : ℕ) (frac_tf : ℚ) := total_tf - written_tf total_tf frac_tf

def total_remaining (total_mc total_ps total_tf : ℕ) (frac_mc frac_ps frac_tf : ℚ) :=
  remaining_mc total_mc frac_mc + remaining_ps total_ps frac_ps + remaining_tf total_tf frac_tf

-- The statement to prove:
theorem meryll_remaining_questions :
  total_remaining 50 30 40 (5/8) (7/12) (2/5) = 56 :=
by
  sorry

end NUMINAMATH_GPT_meryll_remaining_questions_l1802_180265


namespace NUMINAMATH_GPT_constant_speed_l1802_180259

open Real

def total_trip_time := 50.0
def total_distance := 2790.0
def break_interval := 5.0
def break_duration := 0.5
def hotel_search_time := 0.5

theorem constant_speed :
  let number_of_breaks := total_trip_time / break_interval
  let total_break_time := number_of_breaks * break_duration
  let actual_driving_time := total_trip_time - total_break_time - hotel_search_time
  let constant_speed := total_distance / actual_driving_time
  constant_speed = 62.7 :=
by
  -- Provide proof here
  sorry

end NUMINAMATH_GPT_constant_speed_l1802_180259


namespace NUMINAMATH_GPT_fraction_zero_l1802_180261

theorem fraction_zero (x : ℝ) (h₁ : 2 * x = 0) (h₂ : x + 2 ≠ 0) : (2 * x) / (x + 2) = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_zero_l1802_180261


namespace NUMINAMATH_GPT_outlet_pipe_emptying_time_l1802_180201

theorem outlet_pipe_emptying_time :
  let rate1 := 1 / 18
  let rate2 := 1 / 20
  let fill_time := 0.08333333333333333
  ∃ x : ℝ, (rate1 + rate2 - 1 / x = 1 / fill_time) → x = 45 :=
by
  intro rate1 rate2 fill_time
  use 45
  intro h
  sorry

end NUMINAMATH_GPT_outlet_pipe_emptying_time_l1802_180201


namespace NUMINAMATH_GPT_successful_combinations_l1802_180234

def herbs := 4
def gems := 6
def incompatible_combinations := 3

theorem successful_combinations : herbs * gems - incompatible_combinations = 21 := by
  sorry

end NUMINAMATH_GPT_successful_combinations_l1802_180234


namespace NUMINAMATH_GPT_minimum_value_a_l1802_180251

theorem minimum_value_a (a : ℝ) : (∃ x0 : ℝ, |x0 + 1| + |x0 - 2| ≤ a) → a ≥ 3 :=
by 
  sorry

end NUMINAMATH_GPT_minimum_value_a_l1802_180251


namespace NUMINAMATH_GPT_property_P_difference_l1802_180226

noncomputable def f (n : ℕ) : ℕ :=
  if n % 2 = 0 then 
    6 * 2^(n / 2) - n - 5 
  else 
    4 * 2^((n + 1) / 2) - n - 5

theorem property_P_difference : f 9 - f 8 = 31 := by
  sorry

end NUMINAMATH_GPT_property_P_difference_l1802_180226


namespace NUMINAMATH_GPT_problem_sum_150_consecutive_integers_l1802_180250

theorem problem_sum_150_consecutive_integers : 
  ∃ k : ℕ, 150 * k + 11325 = 5310375 :=
sorry

end NUMINAMATH_GPT_problem_sum_150_consecutive_integers_l1802_180250


namespace NUMINAMATH_GPT_inverse_variation_l1802_180209

variable (a b : ℝ)

theorem inverse_variation (h_ab : a * b = 400) :
  (b = 0.25 ∧ a = 1600) ∨ (b = 1.0 ∧ a = 400) :=
  sorry

end NUMINAMATH_GPT_inverse_variation_l1802_180209


namespace NUMINAMATH_GPT_Nick_total_money_l1802_180248

variable (nickels : Nat) (dimes : Nat) (quarters : Nat)
variable (value_nickel : Nat := 5) (value_dime : Nat := 10) (value_quarter : Nat := 25)

def total_value (nickels dimes quarters : Nat) : Nat :=
  nickels * value_nickel + dimes * value_dime + quarters * value_quarter

theorem Nick_total_money :
  total_value 6 2 1 = 75 := by
  sorry

end NUMINAMATH_GPT_Nick_total_money_l1802_180248


namespace NUMINAMATH_GPT_ratio_of_juniors_to_seniors_l1802_180244

theorem ratio_of_juniors_to_seniors (j s : ℕ) (h : (1 / 3) * j = (2 / 3) * s) : j / s = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_juniors_to_seniors_l1802_180244


namespace NUMINAMATH_GPT_molecular_weight_compound_l1802_180237

-- Definitions of atomic weights
def atomic_weight_Cu : ℝ := 63.546
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999

-- Definitions of the number of atoms in the compound
def num_Cu : ℝ := 1
def num_C : ℝ := 1
def num_O : ℝ := 3

-- The molecular weight of the compound
def molecular_weight : ℝ := (num_Cu * atomic_weight_Cu) + (num_C * atomic_weight_C) + (num_O * atomic_weight_O)

-- Statement to prove
theorem molecular_weight_compound : molecular_weight = 123.554 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_compound_l1802_180237


namespace NUMINAMATH_GPT_solve_functional_equation_l1802_180275

theorem solve_functional_equation
  (f g h : ℝ → ℝ)
  (H : ∀ x y : ℝ, f x - g y = (x - y) * h (x + y)) :
  ∃ d c : ℝ, (∀ x, f x = d * x^2 + c) ∧ (∀ x, g x = d * x^2 + c) :=
sorry

end NUMINAMATH_GPT_solve_functional_equation_l1802_180275


namespace NUMINAMATH_GPT_equation_of_parabola_passing_through_points_l1802_180291

noncomputable def parabola (x : ℝ) (b c : ℝ) : ℝ :=
  x^2 + b * x + c

theorem equation_of_parabola_passing_through_points :
  ∃ (b c : ℝ), 
    (parabola 0 b c = 5) ∧ (parabola 3 b c = 2) ∧
    (∀ x, parabola x b c = x^2 - 4 * x + 5) := 
by
  sorry

end NUMINAMATH_GPT_equation_of_parabola_passing_through_points_l1802_180291


namespace NUMINAMATH_GPT_find_ab_l1802_180227

noncomputable def validate_ab : Prop :=
  let n : ℕ := 8
  let a : ℕ := n^2 - 1
  let b : ℕ := n
  a = 63 ∧ b = 8

theorem find_ab : validate_ab :=
by
  sorry

end NUMINAMATH_GPT_find_ab_l1802_180227


namespace NUMINAMATH_GPT_monotonic_intervals_range_of_values_l1802_180202

-- Part (1): Monotonic intervals of the function
theorem monotonic_intervals (a : ℝ) (h_a : a = 0) :
  (∀ x, 0 < x ∧ x < 1 → (1 + Real.log x) / x > 0) ∧ (∀ x, 1 < x → (1 + Real.log x) / x < 0) :=
by
  sorry

-- Part (2): Range of values for \(a\)
theorem range_of_values (a : ℝ) (h_f : ∀ x, 0 < x → (1 + Real.log x) / x - a ≤ 0) : 
  1 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_monotonic_intervals_range_of_values_l1802_180202


namespace NUMINAMATH_GPT_blue_candy_count_l1802_180274

def total_pieces : ℕ := 3409
def red_pieces : ℕ := 145
def blue_pieces : ℕ := total_pieces - red_pieces

theorem blue_candy_count :
  blue_pieces = 3264 := by
  sorry

end NUMINAMATH_GPT_blue_candy_count_l1802_180274


namespace NUMINAMATH_GPT_identify_vanya_l1802_180200

structure Twin :=
(name : String)
(truth_teller : Bool)

def is_vanya_truth_teller (twin : Twin) (vanya vitya : Twin) : Prop :=
  twin = vanya ∧ twin.truth_teller ∨ twin = vitya ∧ ¬twin.truth_teller

theorem identify_vanya
  (vanya vitya : Twin)
  (h_vanya : vanya.name = "Vanya")
  (h_vitya : vitya.name = "Vitya")
  (h_one_truth : ∃ t : Twin, t = vanya ∨ t = vitya ∧ (t.truth_teller = true ∨ t.truth_teller = false))
  (h_one_lie : ∀ t : Twin, t = vanya ∨ t = vitya → ¬(t.truth_teller = true ∧ t = vitya) ∧ ¬(t.truth_teller = false ∧ t = vanya)) :
  ∀ twin : Twin, twin = vanya ∨ twin = vitya →
  (is_vanya_truth_teller twin vanya vitya ↔ (twin = vanya ∧ twin.truth_teller = true)) :=
by
  sorry

end NUMINAMATH_GPT_identify_vanya_l1802_180200


namespace NUMINAMATH_GPT_mans_rate_is_19_l1802_180282

-- Define the given conditions
def downstream_speed : ℝ := 25
def upstream_speed : ℝ := 13

-- Define the man's rate in still water and state the theorem
theorem mans_rate_is_19 : (downstream_speed + upstream_speed) / 2 = 19 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_mans_rate_is_19_l1802_180282


namespace NUMINAMATH_GPT_Leonard_is_11_l1802_180267

def Leonard_age (L N J P T: ℕ) : Prop :=
  (L = N - 4) ∧
  (N = J / 2) ∧
  (P = 2 * L) ∧
  (T = P - 3) ∧
  (L + N + J + P + T = 75)

theorem Leonard_is_11 (L N J P T : ℕ) (h : Leonard_age L N J P T) : L = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_Leonard_is_11_l1802_180267


namespace NUMINAMATH_GPT_dividend_divisor_quotient_l1802_180247

theorem dividend_divisor_quotient (x y z : ℕ) 
  (h1 : x = 6 * y) 
  (h2 : y = 6 * z) 
  (h3 : x = y * z) : 
  x = 216 ∧ y = 36 ∧ z = 6 := 
by
  sorry

end NUMINAMATH_GPT_dividend_divisor_quotient_l1802_180247


namespace NUMINAMATH_GPT_total_handshakes_l1802_180299

def total_people := 40
def group_x_people := 25
def group_x_known_others := 5
def group_y_people := 15
def handshakes_between_x_y := group_x_people * group_y_people
def handshakes_within_x := 25 * (25 - 1 - 5) / 2
def handshakes_within_y := (15 * (15 - 1)) / 2

theorem total_handshakes 
    (h1 : total_people = 40)
    (h2 : group_x_people = 25)
    (h3 : group_x_known_others = 5)
    (h4 : group_y_people = 15) :
    handshakes_between_x_y + handshakes_within_x + handshakes_within_y = 717 := 
by
  sorry

end NUMINAMATH_GPT_total_handshakes_l1802_180299


namespace NUMINAMATH_GPT_point_in_third_quadrant_l1802_180224

theorem point_in_third_quadrant (m : ℝ) : 
  (-1 < 0 ∧ -2 + m < 0) ↔ (m < 2) :=
by 
  sorry

end NUMINAMATH_GPT_point_in_third_quadrant_l1802_180224


namespace NUMINAMATH_GPT_total_number_of_girls_is_13_l1802_180225

def number_of_girls (n : ℕ) (B : ℕ) : Prop :=
  ∃ A : ℕ, (A = B - 5) ∧ (A = B + 8)

theorem total_number_of_girls_is_13 (n : ℕ) (B : ℕ) :
  number_of_girls n B → n = 13 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_total_number_of_girls_is_13_l1802_180225


namespace NUMINAMATH_GPT_vertex_of_parabola_l1802_180229

theorem vertex_of_parabola (c d : ℝ) :
  (∀ x, -2 * x^2 + c * x + d ≤ 0 ↔ x ≥ -7 / 2) →
  ∃ k, k = (-7 / 2 : ℝ) ∧ y = -2 * (x + 7 / 2)^2 + 0 := 
sorry

end NUMINAMATH_GPT_vertex_of_parabola_l1802_180229


namespace NUMINAMATH_GPT_negation_of_exists_l1802_180276

theorem negation_of_exists:
  (¬ ∃ x₀ : ℝ, x₀^2 - x₀ + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) := sorry

end NUMINAMATH_GPT_negation_of_exists_l1802_180276


namespace NUMINAMATH_GPT_percent_difference_l1802_180260

def boys := 100
def girls := 125
def diff := girls - boys
def boys_less_than_girls_percent := (diff : ℚ) / girls  * 100
def girls_more_than_boys_percent := (diff : ℚ) / boys  * 100

theorem percent_difference :
  boys_less_than_girls_percent = 20 ∧ girls_more_than_boys_percent = 25 :=
by
  -- The proof here demonstrates the percentage calculations.
  sorry

end NUMINAMATH_GPT_percent_difference_l1802_180260


namespace NUMINAMATH_GPT_range_of_k_l1802_180252

theorem range_of_k (k : ℝ) : 
  (∃ a b : ℝ, x^2 + ky^2 = 2 ∧ a^2 = 2/k ∧ b^2 = 2 ∧ a > b) → 0 < k ∧ k < 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_k_l1802_180252


namespace NUMINAMATH_GPT_parameterization_solution_l1802_180254

/-- Proof problem statement:
  Given the line equation y = 3x - 11 and its parameterization representation,
  the ordered pair (s, h) that satisfies both conditions is (3, 15).
-/
theorem parameterization_solution : ∃ s h : ℝ, 
  (∀ t : ℝ, (∃ x y : ℝ, (x, y) = (s, -2) + t • (5, h)) ∧ y = 3 * x - 11) → 
  (s = 3 ∧ h = 15) :=
by
  -- introduce s and h 
  use 3
  use 15
  -- skip the proof
  sorry

end NUMINAMATH_GPT_parameterization_solution_l1802_180254


namespace NUMINAMATH_GPT_edward_toy_cars_l1802_180221

def initial_amount : ℝ := 17.80
def cost_per_car : ℝ := 0.95
def cost_of_race_track : ℝ := 6.00
def remaining_amount : ℝ := 8.00

theorem edward_toy_cars : ∃ (n : ℕ), initial_amount - remaining_amount = n * cost_per_car + cost_of_race_track ∧ n = 4 := by
  sorry

end NUMINAMATH_GPT_edward_toy_cars_l1802_180221
