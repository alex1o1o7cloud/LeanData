import Mathlib

namespace ratio_a_c_l1985_198566

variable (a b c d : ℕ)

/-- The given conditions -/
axiom ratio_a_b : a / b = 5 / 2
axiom ratio_c_d : c / d = 4 / 1
axiom ratio_d_b : d / b = 1 / 3

/-- The proof problem -/
theorem ratio_a_c : a / c = 15 / 8 := by
  sorry

end ratio_a_c_l1985_198566


namespace lauren_change_l1985_198565

-- Define the given conditions as Lean terms.
def price_meat_per_pound : ℝ := 3.5
def pounds_meat : ℝ := 2.0
def price_buns : ℝ := 1.5
def price_lettuce : ℝ := 1.0
def pounds_tomato : ℝ := 1.5
def price_tomato_per_pound : ℝ := 2.0
def price_pickles : ℝ := 2.5
def coupon_value : ℝ := 1.0
def amount_paid : ℝ := 20.0

-- Define the total cost of each item.
def cost_meat : ℝ := pounds_meat * price_meat_per_pound
def cost_tomato : ℝ := pounds_tomato * price_tomato_per_pound
def total_cost_before_coupon : ℝ := cost_meat + price_buns + price_lettuce + cost_tomato + price_pickles

-- Define the final total cost after applying the coupon.
def final_total_cost : ℝ := total_cost_before_coupon - coupon_value

-- Define the expected change.
def expected_change : ℝ := amount_paid - final_total_cost

-- Prove that the expected change is $6.00.
theorem lauren_change : expected_change = 6.0 := by
  sorry

end lauren_change_l1985_198565


namespace Masha_thought_of_numbers_l1985_198571

theorem Masha_thought_of_numbers : ∃ a b : ℕ, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (a % 2 = 0 ∨ b % 2 = 0) ∧ (a = 12 ∧ b = 16 ∨ a = 16 ∧ b = 12) :=
by
  sorry

end Masha_thought_of_numbers_l1985_198571


namespace find_constants_l1985_198554

theorem find_constants (A B : ℚ) : 
  (∀ x : ℚ, x ≠ 10 ∧ x ≠ -5 → (8 * x - 3) / (x^2 - 5 * x - 50) = A / (x - 10) + B / (x + 5)) 
  → (A = 77 / 15 ∧ B = 43 / 15) := by 
  sorry

end find_constants_l1985_198554


namespace solve_inequality_system_l1985_198572

theorem solve_inequality_system (x : ℝ) 
  (h1 : 3 * x - 1 > x + 1) 
  (h2 : (4 * x - 5) / 3 ≤ x) 
  : 1 < x ∧ x ≤ 5 :=
by
  sorry

end solve_inequality_system_l1985_198572


namespace Olivia_hours_worked_on_Monday_l1985_198529

/-- Olivia works on multiple days in a week with given wages per hour and total income -/
theorem Olivia_hours_worked_on_Monday 
  (M : ℕ)  -- Hours worked on Monday
  (rate_per_hour : ℕ := 9) -- Olivia’s earning rate per hour
  (hours_Wednesday : ℕ := 3)  -- Hours worked on Wednesday
  (hours_Friday : ℕ := 6)  -- Hours worked on Friday
  (total_income : ℕ := 117)  -- Total income earned this week
  (hours_total : ℕ := hours_Wednesday + hours_Friday + M)
  (income_calc : ℕ := rate_per_hour * hours_total) :
  -- Prove that the hours worked on Monday is 4 given the conditions
  income_calc = total_income → M = 4 :=
by
  sorry

end Olivia_hours_worked_on_Monday_l1985_198529


namespace right_triangle_area_l1985_198517

-- Define the lengths of the legs of the right triangle
def leg_length : ℝ := 1

-- State the theorem
theorem right_triangle_area (a b : ℝ) (h1 : a = leg_length) (h2 : b = leg_length) : 
  (1 / 2) * a * b = 1 / 2 :=
by
  rw [h1, h2]
  -- From the substitutions above, it simplifies to:
  sorry

end right_triangle_area_l1985_198517


namespace inequality_solution_1_inequality_system_solution_2_l1985_198552

theorem inequality_solution_1 (x : ℝ) : 
  (2 * x - 1) / 2 ≥ 1 - (x + 1) / 3 ↔ x ≥ 7 / 8 := 
sorry

theorem inequality_system_solution_2 (x : ℝ) : 
  (-2 * x ≤ -3) ∧ (x / 2 < 2) ↔ (3 / 2 ≤ x) ∧ (x < 4) :=
sorry

end inequality_solution_1_inequality_system_solution_2_l1985_198552


namespace cities_with_fewer_than_200000_residents_l1985_198590

def percentage_of_cities_with_fewer_than_50000 : ℕ := 20
def percentage_of_cities_with_50000_to_199999 : ℕ := 65

theorem cities_with_fewer_than_200000_residents :
  percentage_of_cities_with_fewer_than_50000 + percentage_of_cities_with_50000_to_199999 = 85 :=
by
  sorry

end cities_with_fewer_than_200000_residents_l1985_198590


namespace third_divisor_l1985_198523

theorem third_divisor (x : ℕ) (h12 : 12 ∣ (x + 3)) (h15 : 15 ∣ (x + 3)) (h40 : 40 ∣ (x + 3)) :
  ∃ d : ℕ, d ≠ 12 ∧ d ≠ 15 ∧ d ≠ 40 ∧ d ∣ (x + 3) ∧ d = 2 :=
by
  sorry

end third_divisor_l1985_198523


namespace triangle_perimeter_l1985_198555

theorem triangle_perimeter (a b c : ℕ) (ha : a = 7) (hb : b = 10) (hc : c = 15) :
  a + b + c = 32 :=
by
  -- Given the lengths of the sides
  have H1 : a = 7 := ha
  have H2 : b = 10 := hb
  have H3 : c = 15 := hc
  
  -- Therefore, we need to prove the sum
  sorry

end triangle_perimeter_l1985_198555


namespace no_integer_soln_x_y_l1985_198509

theorem no_integer_soln_x_y (x y : ℤ) : x^2 + 5 ≠ y^3 := 
sorry

end no_integer_soln_x_y_l1985_198509


namespace horner_method_v3_value_l1985_198551

theorem horner_method_v3_value :
  let f (x : ℤ) := 3 * x^6 + 5 * x^5 + 6 * x^4 + 79 * x^3 - 8 * x^2 + 35 * x + 12
  let v : ℤ := 3
  let v1 (x : ℤ) : ℤ := v * x + 5
  let v2 (x : ℤ) (v1x : ℤ) : ℤ := v1x * x + 6
  let v3 (x : ℤ) (v2x : ℤ) : ℤ := v2x * x + 79
  x = -4 →
  v3 x (v2 x (v1 x)) = -57 :=
by
  sorry

end horner_method_v3_value_l1985_198551


namespace no_perfect_square_l1985_198577

-- Define the given polynomial
def poly (n : ℕ) : ℤ := n^6 + 3*n^5 - 5*n^4 - 15*n^3 + 4*n^2 + 12*n + 3

-- The theorem to prove
theorem no_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, poly n = k^2 := by
  sorry

end no_perfect_square_l1985_198577


namespace find_original_number_l1985_198531

def original_number_divide_multiply (x : ℝ) : Prop :=
  (x / 12) * 24 = x + 36

theorem find_original_number (x : ℝ) (h : original_number_divide_multiply x) : x = 36 :=
by
  sorry

end find_original_number_l1985_198531


namespace simplest_square_root_l1985_198560

theorem simplest_square_root : 
  let a1 := Real.sqrt 20
  let a2 := Real.sqrt 2
  let a3 := Real.sqrt (1 / 2)
  let a4 := Real.sqrt 0.2
  a2 = Real.sqrt 2 ∧
  (a1 ≠ Real.sqrt 2 ∧ a3 ≠ Real.sqrt 2 ∧ a4 ≠ Real.sqrt 2) :=
by {
  -- Here, we fill in the necessary proof steps, but it's omitted for now.
  sorry
}

end simplest_square_root_l1985_198560


namespace distance_between_5th_and_23rd_red_light_l1985_198589

theorem distance_between_5th_and_23rd_red_light :
  let inch_to_feet (inches : ℕ) : ℝ := inches / 12.0
  let distance_in_inches := 40 * 8
  inch_to_feet distance_in_inches = 26.67 :=
by
  sorry

end distance_between_5th_and_23rd_red_light_l1985_198589


namespace abs_add_opposite_signs_l1985_198596

theorem abs_add_opposite_signs (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a * b < 0) : |a + b| = 1 := 
sorry

end abs_add_opposite_signs_l1985_198596


namespace contrapositive_proposition_l1985_198520

theorem contrapositive_proposition (a b : ℝ) :
  (¬ ((a - b) * (a + b) = 0) → ¬ (a - b = 0)) :=
sorry

end contrapositive_proposition_l1985_198520


namespace law_I_law_II_l1985_198586

section
variable (x y z : ℝ)

def op_at (a b : ℝ) : ℝ := a + 2 * b
def op_hash (a b : ℝ) : ℝ := 2 * a - b

theorem law_I (x y z : ℝ) : op_at x (op_hash y z) = op_hash (op_at x y) (op_at x z) := 
by
  unfold op_at op_hash
  sorry

theorem law_II (x y z : ℝ) : x + op_at y z ≠ op_at (x + y) (x + z) := 
by
  unfold op_at
  sorry

end

end law_I_law_II_l1985_198586


namespace max_value_of_a_l1985_198525

theorem max_value_of_a :
  ∀ (a : ℚ),
  (∀ (m : ℚ), 1/3 < m ∧ m < a →
   (∀ (x : ℤ), 0 < x ∧ x ≤ 200 →
    ¬ (∃ (y : ℤ), y = m * x + 3 ∨ y = m * x + 1))) →
  a = 68/201 :=
by
  sorry

end max_value_of_a_l1985_198525


namespace compare_magnitudes_l1985_198583

noncomputable def A : ℝ := Real.sin (Real.sin (3 * Real.pi / 8))
noncomputable def B : ℝ := Real.sin (Real.cos (3 * Real.pi / 8))
noncomputable def C : ℝ := Real.cos (Real.sin (3 * Real.pi / 8))
noncomputable def D : ℝ := Real.cos (Real.cos (3 * Real.pi / 8))

theorem compare_magnitudes : B < C ∧ C < A ∧ A < D :=
by
  sorry

end compare_magnitudes_l1985_198583


namespace ordered_pair_a_82_a_28_l1985_198507

-- Definitions for the conditions
def a (i j : ℕ) : ℕ :=
  if i % 2 = 1 then
    if j = 1 then i * i else i * i - (j - 1)
  else
    if j = 1 then (i-1) * i + 1 else i * i - (j - 1)

theorem ordered_pair_a_82_a_28 : (a 8 2, a 2 8) = (51, 63) := by
  sorry

end ordered_pair_a_82_a_28_l1985_198507


namespace min_value_quadratic_l1985_198541

noncomputable def quadratic_min (a c : ℝ) : ℝ :=
  (2 / a) + (2 / c)

theorem min_value_quadratic {a c : ℝ} (ha : a > 0) (hc : c > 0) (hac : a * c = 1/4) : 
  quadratic_min a c = 8 :=
sorry

end min_value_quadratic_l1985_198541


namespace used_computer_lifespan_l1985_198506

-- Problem statement
theorem used_computer_lifespan (cost_new : ℕ) (lifespan_new : ℕ) (cost_used : ℕ) (num_used : ℕ) (savings : ℕ) :
  cost_new = 600 →
  lifespan_new = 6 →
  cost_used = 200 →
  num_used = 2 →
  savings = 200 →
  ((cost_new - savings = num_used * cost_used) → (2 * (lifespan_new / 2) = 6) → lifespan_new / 2 = 3)
:= by
  intros
  sorry

end used_computer_lifespan_l1985_198506


namespace maximize_log_power_l1985_198553

theorem maximize_log_power (a b : ℝ) (ha : 1 < a) (hb : 1 < b) (hab : a * b = 100) :
  ∃ x : ℝ, (a ^ (Real.logb 10 b)^2 = 10^x) ∧ x = 32 / 27 :=
by
  sorry

end maximize_log_power_l1985_198553


namespace find_rhombus_acute_angle_l1985_198519

-- Definitions and conditions
def rhombus_angle (V1 V2 : ℝ) (α : ℝ) : Prop :=
  V1 / V2 = 1 / (2 * Real.sqrt 5)
  
-- Theorem statement
theorem find_rhombus_acute_angle (V1 V2 a : ℝ) (α : ℝ) (h : rhombus_angle V1 V2 α) :
  α = Real.arccos (1 / 9) :=
sorry

end find_rhombus_acute_angle_l1985_198519


namespace count_multiples_of_four_between_100_and_350_l1985_198570

-- Define the problem conditions
def is_multiple_of_four (n : ℕ) : Prop := n % 4 = 0
def in_range (n : ℕ) : Prop := 100 < n ∧ n < 350

-- Problem statement
theorem count_multiples_of_four_between_100_and_350 : 
  ∃ (k : ℕ), k = 62 ∧ ∀ n : ℕ, is_multiple_of_four n ∧ in_range n ↔ (100 < n ∧ n < 350 ∧ is_multiple_of_four n)
:= sorry

end count_multiples_of_four_between_100_and_350_l1985_198570


namespace minimum_value_of_y_at_l1985_198593

def y (x : ℝ) : ℝ := |x + 1| + |x + 2| + |x + 3|

theorem minimum_value_of_y_at (x : ℝ) :
  (∀ x : ℝ, y x ≥ 2) ∧ (y (-2) = 2) :=
by 
  sorry

end minimum_value_of_y_at_l1985_198593


namespace one_thirds_in_eight_halves_l1985_198524

theorem one_thirds_in_eight_halves : (8 / 2) / (1 / 3) = 12 := by
  sorry

end one_thirds_in_eight_halves_l1985_198524


namespace range_of_m_l1985_198503

open Real

theorem range_of_m (a m y1 y2 : ℝ) (h_a_pos : a > 0)
  (hA : y1 = a * (m - 1)^2 + 4 * a * (m - 1) + 3)
  (hB : y2 = a * m^2 + 4 * a * m + 3)
  (h_y1_lt_y2 : y1 < y2) : 
  m > -3 / 2 := 
sorry

end range_of_m_l1985_198503


namespace cost_formula_correct_l1985_198575

def cost_of_ride (T : ℤ) : ℤ :=
  if T > 5 then 10 + 5 * T - 10 else 10 + 5 * T

theorem cost_formula_correct (T : ℤ) : cost_of_ride T = 10 + 5 * T - (if T > 5 then 10 else 0) := by
  sorry

end cost_formula_correct_l1985_198575


namespace find_a_squared_l1985_198539

-- Defining the conditions for the problem
structure RectangleConditions :=
  (a : ℝ) 
  (side_length : ℝ := 36)
  (hinges_vertex : Bool := true)
  (hinges_midpoint : Bool := true)
  (pressed_distance : ℝ := 24)
  (hexagon_area_equiv : Bool := true)

-- Stating the theorem
theorem find_a_squared (cond : RectangleConditions) (ha : 36 * cond.a = 
  (24 * cond.a) + 2 * 15 * Real.sqrt (cond.a^2 - 36)) : 
  cond.a^2 = 720 :=
sorry

end find_a_squared_l1985_198539


namespace hyperbola_eccentricity_l1985_198545

-- Definitions based on the conditions
def hyperbola (a b : ℝ) : Prop := (a > 0) ∧ (b > 0)

def distance_from_focus_to_asymptote (a b c : ℝ) : Prop :=
  (b^2 * c) / (a^2 + b^2).sqrt = b ∧ b = 2 * Real.sqrt 3

def minimum_distance_point_to_focus (a c : ℝ) : Prop :=
  c - a = 2

def eccentricity (a c e : ℝ) : Prop :=
  e = c / a

-- Problem statement
theorem hyperbola_eccentricity (a b c e : ℝ) 
  (h_hyperbola : hyperbola a b)
  (h_dist_asymptote : distance_from_focus_to_asymptote a b c)
  (h_min_dist_focus : minimum_distance_point_to_focus a c)
  (h_eccentricity : eccentricity a c e) :
  e = 2 :=
sorry

end hyperbola_eccentricity_l1985_198545


namespace balls_drawn_ensure_single_color_ge_20_l1985_198567

theorem balls_drawn_ensure_single_color_ge_20 (r g y b w bl : ℕ) (h_r : r = 34) (h_g : g = 28) (h_y : y = 23) (h_b : b = 18) (h_w : w = 12) (h_bl : bl = 11) : 
  ∃ (n : ℕ), n ≥ 20 →
    (r + g + y + b + w + bl - n) + 1 > 20 :=
by
  sorry

end balls_drawn_ensure_single_color_ge_20_l1985_198567


namespace average_speed_to_first_summit_l1985_198550

theorem average_speed_to_first_summit 
  (time_first_summit : ℝ := 3)
  (time_descend_partially : ℝ := 1)
  (time_second_uphill : ℝ := 2)
  (time_descend_back : ℝ := 2)
  (avg_speed_whole_journey : ℝ := 3) :
  avg_speed_whole_journey = 3 →
  time_first_summit = 3 →
  avg_speed_whole_journey * (time_first_summit + time_descend_partially + time_second_uphill + time_descend_back) = 24 →
  avg_speed_whole_journey = 3 := 
by
  intros h_avg_speed h_time_first_summit h_total_distance
  sorry

end average_speed_to_first_summit_l1985_198550


namespace find_a_l1985_198556

theorem find_a (a : ℝ) (U A CU: Set ℝ) (hU : U = {2, 3, a^2 - a - 1}) (hA : A = {2, 3}) (hCU : CU = {1}) (hComplement : CU = U \ A) :
  a = -1 ∨ a = 2 :=
by
  sorry

end find_a_l1985_198556


namespace intersection_A_B_find_a_b_l1985_198527

noncomputable def A : Set ℝ := { x | x^2 - 5 * x + 6 > 0 }
noncomputable def B : Set ℝ := { x | Real.log (x + 1) / Real.log 2 < 2 }

theorem intersection_A_B :
  A ∩ B = { x | -1 < x ∧ x < 2 } :=
by
  -- Proof will be provided
  sorry

theorem find_a_b :
  ∃ a b : ℝ, (∀ x : ℝ, x^2 + a * x - b < 0 ↔ -1 < x ∧ x < 2) ∧ a = -1 ∧ b = 2 :=
by
  -- Proof will be provided
  sorry

end intersection_A_B_find_a_b_l1985_198527


namespace increasing_sum_sequence_l1985_198588

theorem increasing_sum_sequence (a : ℕ → ℝ) (Sn : ℕ → ℝ)
  (ha : ∀ n : ℕ, 0 < a (n + 1))
  (hSn : ∀ n : ℕ, Sn (n + 1) = Sn n + a (n + 1)) :
  (∀ n : ℕ, Sn (n + 1) > Sn n)
  ∧ ¬ (∀ n : ℕ, Sn (n + 1) > Sn n → 0 < a (n + 1)) :=
sorry

end increasing_sum_sequence_l1985_198588


namespace sum_of_xyz_l1985_198598

theorem sum_of_xyz (x y z : ℝ) (h : (x - 5)^2 + (y - 3)^2 + (z - 1)^2 = 0) : x + y + z = 9 :=
by {
  sorry
}

end sum_of_xyz_l1985_198598


namespace no_solution_for_given_m_l1985_198504

theorem no_solution_for_given_m (x m : ℝ) (h1 : x ≠ 5) (h2 : x ≠ 8) :
  (∀ y : ℝ, (y - 2) / (y - 5) = (y - m) / (y - 8) → false) ↔ m = 5 :=
by
  sorry

end no_solution_for_given_m_l1985_198504


namespace gcd_12012_21021_l1985_198505

-- Definitions
def factors_12012 : List ℕ := [2, 2, 3, 7, 11, 13] -- Factors of 12,012
def factors_21021 : List ℕ := [3, 7, 7, 11, 13] -- Factors of 21,021

def common_factors := [3, 7, 11, 13] -- Common factors between 12,012 and 21,021

def gcd (ls : List ℕ) : ℕ :=
ls.foldr Nat.gcd 0 -- Function to calculate gcd of list of numbers

-- Main statement
theorem gcd_12012_21021 : gcd common_factors = 1001 := by
  -- Proof is not required, so we use sorry to skip the proof.
  sorry

end gcd_12012_21021_l1985_198505


namespace older_sister_age_l1985_198508

theorem older_sister_age (x : ℕ) (older_sister_age : ℕ) (h1 : older_sister_age = 3 * x)
  (h2 : older_sister_age + 2 = 2 * (x + 2)) : older_sister_age = 6 :=
by
  sorry

end older_sister_age_l1985_198508


namespace perfect_cube_prime_l1985_198591

theorem perfect_cube_prime (p : ℕ) (h_prime : Nat.Prime p) (h_cube : ∃ x : ℕ, 2 * p + 1 = x^3) : 
  2 * p + 1 = 27 ∧ p = 13 :=
by
  sorry

end perfect_cube_prime_l1985_198591


namespace chef_meals_prepared_for_dinner_l1985_198579

theorem chef_meals_prepared_for_dinner (lunch_meals_prepared lunch_meals_sold dinner_meals_total : ℕ) 
  (h1 : lunch_meals_prepared = 17)
  (h2 : lunch_meals_sold = 12)
  (h3 : dinner_meals_total = 10) :
  (dinner_meals_total - (lunch_meals_prepared - lunch_meals_sold)) = 5 :=
by
  -- Lean proof code to proceed from here
  sorry

end chef_meals_prepared_for_dinner_l1985_198579


namespace min_value_at_neg7_l1985_198512

noncomputable def f (x : ℝ) : ℝ := x^2 + 14 * x + 24

theorem min_value_at_neg7 : ∀ x : ℝ, f (-7) ≤ f x :=
by
  sorry

end min_value_at_neg7_l1985_198512


namespace exists_sum_of_150_consecutive_integers_l1985_198580

theorem exists_sum_of_150_consecutive_integers :
  ∃ a : ℕ, 1627395075 = 150 * a + 11175 :=
by
  sorry

end exists_sum_of_150_consecutive_integers_l1985_198580


namespace grandfather_age_5_years_back_l1985_198578

variable (F S G : ℕ)

-- Conditions
def father_age : Prop := F = 58
def son_current_age : Prop := S = 58 - S
def son_grandfather_age_relation : Prop := S - 5 = 1 / 2 * (G - 5)

-- Theorem: Prove the grandfather's age 5 years back given the conditions.
theorem grandfather_age_5_years_back (h1 : father_age F) (h2 : son_current_age S) (h3 : son_grandfather_age_relation S G) : G = 2 * S - 5 :=
sorry

end grandfather_age_5_years_back_l1985_198578


namespace chosen_number_eq_l1985_198568

-- Given a number x, if (x / 2) - 100 = 4, then x = 208.
theorem chosen_number_eq (x : ℝ) (h : (x / 2) - 100 = 4) : x = 208 := 
by
  sorry

end chosen_number_eq_l1985_198568


namespace Francie_remaining_money_l1985_198536

theorem Francie_remaining_money :
  let weekly_allowance_8_weeks : ℕ := 5 * 8
  let weekly_allowance_6_weeks : ℕ := 6 * 6
  let cash_gift : ℕ := 20
  let initial_total_savings := weekly_allowance_8_weeks + weekly_allowance_6_weeks + cash_gift

  let investment_amount : ℕ := 10
  let expected_return_investment_1 : ℚ := 0.05 * 10
  let expected_return_investment_2 : ℚ := (0.5 * 0.10 * 10) + (0.5 * 0.02 * 10)
  let best_investment_return := max expected_return_investment_1 expected_return_investment_2
  let final_savings_after_investment : ℚ := initial_total_savings - investment_amount + best_investment_return

  let amount_for_clothes : ℚ := final_savings_after_investment / 2
  let remaining_after_clothes := final_savings_after_investment - amount_for_clothes
  let cost_of_video_game : ℕ := 35
  
  remaining_after_clothes.sub cost_of_video_game = 8.30 :=
by
  intros
  sorry

end Francie_remaining_money_l1985_198536


namespace intersecting_lines_angle_difference_l1985_198513

-- Define the conditions
def angle_y : ℝ := 40
def straight_angle_sum : ℝ := 180

-- Define the variables representing the angles
variable (x y : ℝ)

-- Define the proof problem
theorem intersecting_lines_angle_difference : 
  ∀ x y : ℝ, 
  y = angle_y → 
  (∃ (a b : ℝ), a + b = straight_angle_sum ∧ a = y ∧ b = x) → 
  x - y = 100 :=
by
  intros x y hy h
  sorry

end intersecting_lines_angle_difference_l1985_198513


namespace clock_equiv_to_square_l1985_198543

theorem clock_equiv_to_square : ∃ h : ℕ, h > 5 ∧ (h^2 - h) % 24 = 0 ∧ h = 9 :=
by 
  let h := 9
  use h
  refine ⟨by decide, by decide, rfl⟩ 

end clock_equiv_to_square_l1985_198543


namespace factorization_identity_l1985_198585

theorem factorization_identity (a b : ℝ) : 3 * a^2 + 6 * a * b + 3 * b^2 = 3 * (a + b)^2 :=
by
  sorry

end factorization_identity_l1985_198585


namespace log_ordering_l1985_198559

noncomputable def P : ℝ := Real.log 3 / Real.log 2
noncomputable def Q : ℝ := Real.log 2 / Real.log 3
noncomputable def R : ℝ := Real.log (Real.log 2 / Real.log 3) / Real.log 2

theorem log_ordering (P Q R : ℝ) (h₁ : P = Real.log 3 / Real.log 2)
  (h₂ : Q = Real.log 2 / Real.log 3) (h₃ : R = Real.log (Real.log 2 / Real.log 3) / Real.log 2) :
  R < Q ∧ Q < P := by
  sorry

end log_ordering_l1985_198559


namespace correct_option_is_B_l1985_198535

def satisfy_triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem correct_option_is_B :
  satisfy_triangle_inequality 3 4 5 ∧
  ¬ satisfy_triangle_inequality 1 1 2 ∧
  ¬ satisfy_triangle_inequality 1 4 6 ∧
  ¬ satisfy_triangle_inequality 2 3 7 :=
by
  sorry

end correct_option_is_B_l1985_198535


namespace first_term_of_geometric_sequence_l1985_198597

theorem first_term_of_geometric_sequence :
  ∀ (a b c : ℝ), 
    (∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ 16 = b * r ∧ c = 16 * r ∧ 128 = c * r) →
    a = 1 / 4 :=
by
  intros a b c
  rintro ⟨r, hr0, hbr, h16r, hcr, h128r⟩
  sorry

end first_term_of_geometric_sequence_l1985_198597


namespace determine_a_b_l1985_198542

-- Define the polynomial expression
def poly (x a b : ℝ) : ℝ := x^2 + a * x + b

-- Define the factored form
def factored_poly (x : ℝ) : ℝ := (x + 1) * (x - 3)

-- State the theorem
theorem determine_a_b (a b : ℝ) (h : ∀ x, poly x a b = factored_poly x) : a = -2 ∧ b = -3 :=
by 
  sorry

end determine_a_b_l1985_198542


namespace total_games_played_l1985_198516

-- Define the conditions as parameters
def ratio_games_won_lost (W L : ℕ) : Prop := W / 2 = L / 3

-- Let's state the problem formally in Lean
theorem total_games_played (W L : ℕ) (h1 : ratio_games_won_lost W L) (h2 : W = 18) : W + L = 30 :=
by 
  sorry  -- The proof will be filled in


end total_games_played_l1985_198516


namespace min_right_triangle_side_l1985_198526

theorem min_right_triangle_side (s : ℕ) : 
  (7^2 + 24^2 = s^2 ∧ 7 + 24 > s ∧ 24 + s > 7 ∧ 7 + s > 24) → s = 25 :=
by
  intro h
  sorry

end min_right_triangle_side_l1985_198526


namespace min_sum_of_primes_l1985_198562

open Classical

theorem min_sum_of_primes (k m n p : ℕ) (h1 : 47 + m = k) (h2 : 53 + n = k) (h3 : 71 + p = k)
  (pm : Prime m) (pn : Prime n) (pp : Prime p) :
  m + n + p = 57 ↔ (k = 76 ∧ m = 29 ∧ n = 23 ∧ p = 5) :=
by {
  sorry
}

end min_sum_of_primes_l1985_198562


namespace find_f1_l1985_198573

theorem find_f1 (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f (3 * x + 1) = x^2 + 3*x + 2) :
  f 1 = 2 :=
by
  -- Proof is omitted
  sorry

end find_f1_l1985_198573


namespace area_increase_percentage_area_percentage_increase_length_to_width_ratio_l1985_198500

open Real

-- Part (a)
theorem area_increase_percentage (a b : ℝ) :
  (1.12 * a) * (1.15 * b) = 1.288 * (a * b) :=
  sorry

theorem area_percentage_increase (a b : ℝ) :
  ((1.12 * a) * (1.15 * b)) / (a * b) = 1.288 :=
  sorry

-- Part (b)
theorem length_to_width_ratio (a b : ℝ) (h : 2 * ((1.12 * a) + (1.15 * b)) = 1.13 * 2 * (a + b)) :
  a = 2 * b :=
  sorry

end area_increase_percentage_area_percentage_increase_length_to_width_ratio_l1985_198500


namespace first_worker_time_l1985_198557

theorem first_worker_time
  (T : ℝ) 
  (hT : T ≠ 0)
  (h_comb : (T + 8) / (8 * T) = 1 / 3.428571428571429) :
  T = 8 / 7 :=
by
  sorry

end first_worker_time_l1985_198557


namespace flower_combinations_count_l1985_198582

/-- Prove that there are exactly 3 combinations of tulips and sunflowers that sum up to $60,
    where tulips cost $4 each and sunflowers cost $3 each, and the number of sunflowers is greater than the number 
    of tulips. -/
theorem flower_combinations_count :
  ∃ n : ℕ, n = 3 ∧
    ∃ t s : ℕ, 4 * t + 3 * s = 60 ∧ s > t :=
by {
  sorry
}

end flower_combinations_count_l1985_198582


namespace range_of_a_l1985_198514

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + a * x + 1 ≥ 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l1985_198514


namespace percent_of_x_is_y_minus_z_l1985_198574

variable (x y z : ℝ)

axiom condition1 : 0.60 * (x - y) = 0.30 * (x + y + z)
axiom condition2 : 0.40 * (y - z) = 0.20 * (y + x - z)

theorem percent_of_x_is_y_minus_z :
  (y - z) = x := by
  sorry

end percent_of_x_is_y_minus_z_l1985_198574


namespace car_speed_l1985_198501

/-- Given a car covers a distance of 624 km in 2 3/5 hours,
    prove that the speed of the car is 240 km/h. -/
theorem car_speed (distance : ℝ) (time : ℝ)
  (h_distance : distance = 624)
  (h_time : time = 13 / 5) :
  (distance / time) = 240 :=
by
  sorry

end car_speed_l1985_198501


namespace fraction_shaded_is_one_tenth_l1985_198547

theorem fraction_shaded_is_one_tenth :
  ∀ (A L S: ℕ), A = 300 → L = 5 → S = 2 → 
  ((15 * 20 = A) → (A / L = 60) → (60 / S = 30) → (30 / A = 1 / 10)) :=
by sorry

end fraction_shaded_is_one_tenth_l1985_198547


namespace one_sixth_time_l1985_198510

-- Conditions
def total_kids : ℕ := 40
def kids_less_than_6_minutes : ℕ := total_kids * 10 / 100
def kids_less_than_8_minutes : ℕ := 3 * kids_less_than_6_minutes
def remaining_kids : ℕ := total_kids - (kids_less_than_6_minutes + kids_less_than_8_minutes)
def kids_more_than_certain_minutes : ℕ := 4
def one_sixth_remaining_kids : ℕ := remaining_kids / 6

-- Statement to prove the equivalence
theorem one_sixth_time :
  one_sixth_remaining_kids = kids_more_than_certain_minutes := 
sorry

end one_sixth_time_l1985_198510


namespace find_number_l1985_198522

/--
A number is added to 5, then multiplied by 5, then subtracted by 5, and then divided by 5. 
The result is still 5. Prove that the number is 1.
-/
theorem find_number (x : ℝ) (h : ((5 * (x + 5) - 5) / 5 = 5)) : x = 1 := 
  sorry

end find_number_l1985_198522


namespace find_prime_p_l1985_198563

def is_prime (p: ℕ) : Prop := Nat.Prime p

def is_product_of_three_distinct_primes (n: ℕ) : Prop :=
  ∃ (p1 p2 p3: ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ n = p1 * p2 * p3

theorem find_prime_p (p: ℕ) (hp: is_prime p) :
  (∃ x y z: ℕ, x^p + y^p + z^p - x - y - z = 30) ↔ (p = 2 ∨ p = 3 ∨ p = 5) := 
sorry

end find_prime_p_l1985_198563


namespace zero_neither_positive_nor_negative_l1985_198584

def is_positive (n : ℤ) : Prop := n > 0
def is_negative (n : ℤ) : Prop := n < 0
def is_rational (n : ℤ) : Prop := ∃ p q : ℤ, q ≠ 0 ∧ n = p / q

theorem zero_neither_positive_nor_negative : ¬is_positive 0 ∧ ¬is_negative 0 :=
by
  sorry

end zero_neither_positive_nor_negative_l1985_198584


namespace range_of_m_for_hyperbola_l1985_198530

theorem range_of_m_for_hyperbola (m : ℝ) :
  (∃ (x y : ℝ), (m+2) ≠ 0 ∧ (m-2) ≠ 0 ∧ (x^2)/(m+2) + (y^2)/(m-2) = 1) ↔ (-2 < m ∧ m < 2) :=
by
  sorry

end range_of_m_for_hyperbola_l1985_198530


namespace distinct_sequences_l1985_198533

theorem distinct_sequences (N : ℕ) (α : ℝ) 
  (cond1 : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ N → 1 ≤ j ∧ j ≤ N → i ≠ j → 
    Int.floor (i * α) ≠ Int.floor (j * α)) 
  (cond2 : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ N → 1 ≤ j ∧ j ≤ N → i ≠ j → 
    Int.floor (i / α) ≠ Int.floor (j / α)) : 
  (↑(N - 1) / ↑N : ℝ) ≤ α ∧ α ≤ (↑N / ↑(N - 1) : ℝ) := 
sorry

end distinct_sequences_l1985_198533


namespace find_number_l1985_198581

theorem find_number (N x : ℝ) (h1 : x = 1) (h2 : N / (4 + 1 / x) = 1) : N = 5 := 
by 
  sorry

end find_number_l1985_198581


namespace log_50_between_integers_l1985_198549

open Real

-- Declaration of the proof problem
theorem log_50_between_integers (a b : ℤ) (h1 : log 10 = 1) (h2 : log 100 = 2) (h3 : 10 < 50) (h4 : 50 < 100) :
  a + b = 3 :=
by
  sorry

end log_50_between_integers_l1985_198549


namespace solve_for_X_l1985_198548

theorem solve_for_X (X : ℝ) (h : (X ^ (5 / 4)) = 32 * (32 ^ (1 / 16))) :
  X =  16 * (2 ^ (1 / 4)) :=
sorry

end solve_for_X_l1985_198548


namespace members_not_playing_either_l1985_198534

variable (total_members badminton_players tennis_players both_players : ℕ)

theorem members_not_playing_either (h1 : total_members = 40)
                                   (h2 : badminton_players = 20)
                                   (h3 : tennis_players = 18)
                                   (h4 : both_players = 3) :
  total_members - (badminton_players + tennis_players - both_players) = 5 := by
  sorry

end members_not_playing_either_l1985_198534


namespace max_k_inequality_l1985_198538

theorem max_k_inequality (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : a ≤ 1) 
                                      (h₂ : 0 ≤ b) (h₃ : b ≤ 1) 
                                      (h₄ : 0 ≤ c) (h₅ : c ≤ 1) 
                                      (h₆ : 0 ≤ d) (h₇ : d ≤ 1) :
  a^2 * b + b^2 * c + c^2 * d + d^2 * a + 4 ≥ 2 * (a^2 + b^2 + c^2 + d^2) :=
sorry

end max_k_inequality_l1985_198538


namespace termites_count_l1985_198595

theorem termites_count (total_workers monkeys : ℕ) (h1 : total_workers = 861) (h2 : monkeys = 239) : total_workers - monkeys = 622 :=
by
  -- The proof steps will go here
  sorry

end termites_count_l1985_198595


namespace Anne_Katherine_savings_l1985_198515

theorem Anne_Katherine_savings :
  ∃ A K : ℕ, (A - 150 = K / 3) ∧ (2 * K = 3 * A) ∧ (A + K = 750) := 
sorry

end Anne_Katherine_savings_l1985_198515


namespace tan_sum_trig_identity_l1985_198540

variable {A B C : ℝ} -- Angles
variable {a b c : ℝ} -- Sides opposite to angles A, B and C

-- Acute triangle implies A, B, C are all less than π/2 and greater than 0
variable (hAcute : 0 < A ∧ A < pi / 2 ∧ 0 < B ∧ B < pi / 2 ∧ 0 < C ∧ C < pi / 2)

-- Given condition in the problem
variable (hCondition : b / a + a / b = 6 * Real.cos C)

theorem tan_sum_trig_identity : 
  Real.tan C / Real.tan A + Real.tan C / Real.tan B = 4 :=
sorry

end tan_sum_trig_identity_l1985_198540


namespace tony_quilt_square_side_length_l1985_198546

theorem tony_quilt_square_side_length (length width : ℝ) (h_length : length = 6) (h_width : width = 24) : 
  ∃ s, s * s = length * width ∧ s = 12 :=
by
  sorry

end tony_quilt_square_side_length_l1985_198546


namespace max_non_multiples_of_3_l1985_198511

theorem max_non_multiples_of_3 (a b c d e f : ℕ) (h1 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) (h2 : a * b * c * d * e * f % 3 = 0) : 
  ¬ ∃ (count : ℕ), count > 5 ∧ (∀ x ∈ [a, b, c, d, e, f], x % 3 ≠ 0) :=
by
  sorry

end max_non_multiples_of_3_l1985_198511


namespace tamara_has_30_crackers_l1985_198587

theorem tamara_has_30_crackers :
  ∀ (Tamara Nicholas Marcus Mona : ℕ),
    Tamara = 2 * Nicholas →
    Marcus = 3 * Mona →
    Nicholas = Mona + 6 →
    Marcus = 27 →
    Tamara = 30 :=
by
  intros Tamara Nicholas Marcus Mona h1 h2 h3 h4
  sorry

end tamara_has_30_crackers_l1985_198587


namespace find_e_of_conditions_l1985_198502

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := x^3 + d * x^2 + e * x + f

theorem find_e_of_conditions (d e f : ℝ) 
  (h1 : f = 6) 
  (h2 : -d / 3 = -f)
  (h3 : -f = d + e + f - 1) : 
  e = -30 :=
by 
  sorry

end find_e_of_conditions_l1985_198502


namespace inequality_proof_l1985_198521

theorem inequality_proof
  (x y z : ℝ) (hxpos : 0 < x) (hypos : 0 < y) (hzpos : 0 < z)
  (hineq : x * y + y * z + z * x ≤ 1) :
  (x + 1 / x) * (y + 1 / y) * (z + 1 / z) ≥ 8 * (x + y) * (y + z) * (z + x) :=
sorry

end inequality_proof_l1985_198521


namespace star_comm_l1985_198537

section SymmetricOperation

variable {S : Type*} 
variable (star : S → S → S)
variable (symm : ∀ a b : S, star a b = star (star b a) (star b a)) 

theorem star_comm (a b : S) : star a b = star b a := 
by 
  sorry

end SymmetricOperation

end star_comm_l1985_198537


namespace calc_expression_l1985_198561

noncomputable def x := (3 + Real.sqrt 5) / 2 -- chosen from one of the roots of the quadratic equation x^2 - 3x + 1

theorem calc_expression (h : x + 1 / x = 3) : 
  (x - 1) ^ 2 + 16 / (x - 1) ^ 2 = 7 + 3 * Real.sqrt 5 := 
by 
  sorry

end calc_expression_l1985_198561


namespace series_evaluation_l1985_198528

noncomputable def series_sum : ℝ :=
  ∑' m : ℕ, (∑' n : ℕ, (m^2 * n) / (3^m * (n * 3^m + m * 3^n)))

theorem series_evaluation : series_sum = 9 / 32 :=
by
  sorry

end series_evaluation_l1985_198528


namespace arithmetic_mean_is_correct_l1985_198532

variable (x a : ℝ)
variable (hx : x ≠ 0)

theorem arithmetic_mean_is_correct : 
  (1/2 * ((x + 2 * a) / x - 1 + (x - 3 * a) / x + 1)) = (1 - a / (2 * x)) := 
  sorry

end arithmetic_mean_is_correct_l1985_198532


namespace greatest_area_difference_l1985_198544

theorem greatest_area_difference 
    (a b c d : ℕ) 
    (H1 : 2 * (a + b) = 100)
    (H2 : 2 * (c + d) = 100)
    (H3 : ∀i j : ℕ, 2 * (i + j) = 100 → i * j ≤ a * b)
    : 373 ≤ a * b - (c * d) := 
sorry

end greatest_area_difference_l1985_198544


namespace pigeon_distance_l1985_198569

-- Define the conditions
def pigeon_trip (d : ℝ) (v : ℝ) (wind : ℝ) (time_nowind : ℝ) (time_wind : ℝ) :=
  (2 * d / v = time_nowind) ∧
  (d / (v + wind) + d / (v - wind) = time_wind)

-- Define the theorems to be proven
theorem pigeon_distance : ∃ (d : ℝ), pigeon_trip d 40 10 3.75 4 ∧ d = 75 :=
  by {
  sorry
}

end pigeon_distance_l1985_198569


namespace percentage_of_mixture_X_is_13_333_l1985_198592

variable (X Y : ℝ) (P : ℝ)

-- Conditions
def mixture_X_contains_40_percent_ryegrass : Prop := X = 0.40
def mixture_Y_contains_25_percent_ryegrass : Prop := Y = 0.25
def final_mixture_contains_27_percent_ryegrass : Prop := 0.4 * P + 0.25 * (100 - P) = 27

-- The goal
theorem percentage_of_mixture_X_is_13_333
    (h1 : mixture_X_contains_40_percent_ryegrass X)
    (h2 : mixture_Y_contains_25_percent_ryegrass Y)
    (h3 : final_mixture_contains_27_percent_ryegrass P) :
  P = 200 / 15 := by
  sorry

end percentage_of_mixture_X_is_13_333_l1985_198592


namespace ratio_a_to_c_l1985_198594

theorem ratio_a_to_c (a b c d : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 7) : 
  a / c = 105 / 16 :=
by sorry

end ratio_a_to_c_l1985_198594


namespace find_savings_l1985_198564

theorem find_savings (income expenditure : ℕ) (ratio_income_expenditure : ℕ × ℕ) (income_value : income = 40000)
    (ratio_condition : ratio_income_expenditure = (8, 7)) :
    income - expenditure = 5000 :=
by
  sorry

end find_savings_l1985_198564


namespace common_difference_l1985_198576

def Sn (S : Nat → ℝ) (n : Nat) : ℝ := S n

theorem common_difference (S : Nat → ℝ) (H : Sn S 2016 / 2016 = Sn S 2015 / 2015 + 1) : 2 = 2 := 
by
  sorry

end common_difference_l1985_198576


namespace front_view_heights_l1985_198599

-- Define conditions
def column1 := [4, 2]
def column2 := [3, 0, 3]
def column3 := [1, 5]

-- Define a function to get the max height in each column
def max_height (col : List Nat) : Nat :=
  col.foldr Nat.max 0

-- Define the statement to prove the frontal view heights
theorem front_view_heights : 
  max_height column1 = 4 ∧ 
  max_height column2 = 3 ∧ 
  max_height column3 = 5 :=
by 
  sorry

end front_view_heights_l1985_198599


namespace maximize_expr_at_neg_5_l1985_198518

-- Definition of the expression
def expr (x : ℝ) : ℝ := 1 - (x + 5) ^ 2

-- Prove that when x = -5, the expression has its maximum value
theorem maximize_expr_at_neg_5 : ∀ x : ℝ, expr x ≤ expr (-5) :=
by
  -- Placeholder for the proof
  sorry

end maximize_expr_at_neg_5_l1985_198518


namespace volume_ratio_of_cube_cut_l1985_198558

/-
  The cube ABCDEFGH has its side length assumed to be 1.
  The points K, L, M divide the vertical edges AA', BB', CC'
  respectively, in the ratios 1:2, 1:3, 1:4. 
  We need to prove that the plane KLM cuts the cube into
  two parts such that the volume ratio of the two parts is 4:11.
-/
theorem volume_ratio_of_cube_cut (s : ℝ) (K L M : ℝ) :
  ∃ (Vbelow Vabove : ℝ), 
    s = 1 → 
    K = 1/3 → 
    L = 1/4 → 
    M = 1/5 → 
    Vbelow / Vabove = 4 / 11 :=
sorry

end volume_ratio_of_cube_cut_l1985_198558
