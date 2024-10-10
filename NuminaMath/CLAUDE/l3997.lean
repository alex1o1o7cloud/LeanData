import Mathlib

namespace megan_current_seashells_l3997_399773

-- Define the number of seashells Megan currently has
def current_seashells : ℕ := 19

-- Define the number of additional seashells Megan needs
def additional_seashells : ℕ := 6

-- Define the total number of seashells Megan will have after adding more
def total_seashells : ℕ := 25

-- Theorem stating that Megan currently has 19 seashells
theorem megan_current_seashells : 
  current_seashells = total_seashells - additional_seashells :=
by
  sorry

end megan_current_seashells_l3997_399773


namespace vector_collinearity_problem_l3997_399733

/-- Given two 2D vectors are collinear if the cross product of their coordinates is zero -/
def collinear (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 - v.2 * w.1 = 0

/-- The problem statement -/
theorem vector_collinearity_problem (m : ℝ) :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (-1, 2)
  collinear (m * a.1 + 4 * b.1, m * a.2 + 4 * b.2) (a.1 - 2 * b.1, a.2 - 2 * b.2) →
  m = -2 := by
  sorry

end vector_collinearity_problem_l3997_399733


namespace h_zero_iff_b_eq_neg_six_fifths_l3997_399739

def h (x : ℝ) : ℝ := 5 * x + 6

theorem h_zero_iff_b_eq_neg_six_fifths :
  ∀ b : ℝ, h b = 0 ↔ b = -6/5 := by sorry

end h_zero_iff_b_eq_neg_six_fifths_l3997_399739


namespace syrup_cost_is_fifty_cents_l3997_399772

/-- The cost of a Build Your Own Hot Brownie dessert --/
def dessert_cost (brownie_cost ice_cream_cost nuts_cost syrup_cost : ℚ) : ℚ :=
  brownie_cost + 2 * ice_cream_cost + nuts_cost + 2 * syrup_cost

/-- Theorem: The syrup cost is $0.50 per serving --/
theorem syrup_cost_is_fifty_cents :
  ∃ (syrup_cost : ℚ),
    dessert_cost 2.5 1 1.5 syrup_cost = 7 ∧
    syrup_cost = 0.5 := by
  sorry

end syrup_cost_is_fifty_cents_l3997_399772


namespace sum_of_integers_l3997_399724

theorem sum_of_integers (x y : ℕ+) (h1 : x.val^2 + y.val^2 = 250) (h2 : x.val * y.val = 108) :
  x.val + y.val = Real.sqrt 466 := by
  sorry

end sum_of_integers_l3997_399724


namespace triangle_angle_b_value_l3997_399720

theorem triangle_angle_b_value 
  (A B C : Real) 
  (a b c : Real) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : A + B + C = π) 
  (h3 : 0 < A ∧ A < π) 
  (h4 : 0 < B ∧ B < π) 
  (h5 : 0 < C ∧ C < π) 
  (h6 : (c - b) / (Real.sqrt 2 * c - a) = Real.sin A / (Real.sin B + Real.sin C)) : 
  B = π / 4 := by
  sorry

end triangle_angle_b_value_l3997_399720


namespace xyz_problem_l3997_399783

theorem xyz_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * (y + z) = 132)
  (h2 : y * (z + x) = 152)
  (h3 : x * y * z = 160) :
  z * (x + y) = 131.92 := by
  sorry

end xyz_problem_l3997_399783


namespace inequality_solution_l3997_399761

theorem inequality_solution : 
  ∀ x : ℝ, (|x - 1| + |x + 2| + |x| < 7) ↔ (-2 < x ∧ x < 2) := by
sorry

end inequality_solution_l3997_399761


namespace distance_AD_between_41_and_42_l3997_399732

-- Define points A, B, C, and D in a 2D plane
variable (A B C D : ℝ × ℝ)

-- Define the conditions
variable (h1 : B.1 > A.1 ∧ B.2 = A.2) -- B is due east of A
variable (h2 : C.1 = B.1 ∧ C.2 > B.2) -- C is due north of B
variable (h3 : (C.1 - A.1)^2 + (C.2 - A.2)^2 = 300) -- AC = 10√3
variable (h4 : Real.cos (Real.arctan ((C.2 - A.2) / (C.1 - A.1))) = 1/2) -- Angle BAC = 60°
variable (h5 : D.1 = C.1 ∧ D.2 = C.2 + 30) -- D is 30 meters due north of C

-- Theorem statement
theorem distance_AD_between_41_and_42 :
  41 < Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) ∧
  Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) < 42 :=
sorry

end distance_AD_between_41_and_42_l3997_399732


namespace multiples_of_three_never_reach_one_l3997_399766

def operation (n : ℕ) : ℕ :=
  (n + 3 * (5 - n % 5) % 5) / 5

theorem multiples_of_three_never_reach_one (k : ℕ) :
  ∀ n : ℕ, (∃ m : ℕ, n = operation^[m] (3 * k)) → n ≠ 1 :=
sorry

end multiples_of_three_never_reach_one_l3997_399766


namespace race_time_difference_per_hurdle_l3997_399746

/-- Given a race with the following parameters:
  * Total distance: 120 meters
  * Hurdles placed every 20 meters
  * Runner A's total time: 36 seconds
  * Runner B's total time: 45 seconds
Prove that the time difference between the runners at each hurdle is 1.5 seconds. -/
theorem race_time_difference_per_hurdle 
  (total_distance : ℝ) 
  (hurdle_interval : ℝ)
  (runner_a_time : ℝ)
  (runner_b_time : ℝ)
  (h1 : total_distance = 120)
  (h2 : hurdle_interval = 20)
  (h3 : runner_a_time = 36)
  (h4 : runner_b_time = 45) :
  (runner_b_time - runner_a_time) / (total_distance / hurdle_interval) = 1.5 := by
sorry

end race_time_difference_per_hurdle_l3997_399746


namespace min_value_of_expression_l3997_399705

theorem min_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  2 * x^2 + 8 * x * y + 32 * y^2 + 16 * y * z + 8 * z^2 ≥ 72 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    2 * x₀^2 + 8 * x₀ * y₀ + 32 * y₀^2 + 16 * y₀ * z₀ + 8 * z₀^2 = 72 :=
by sorry

end min_value_of_expression_l3997_399705


namespace business_investment_l3997_399760

/-- Prove that the total investment is 90000 given the conditions of the business problem -/
theorem business_investment (a b c : ℕ) (total_profit a_share : ℕ) : 
  a = b + 6000 →
  c = b + 3000 →
  total_profit = 8640 →
  a_share = 3168 →
  a_share * (a + b + c) = a * total_profit →
  a + b + c = 90000 :=
by sorry

end business_investment_l3997_399760


namespace arithmetic_sequence_sum_l3997_399765

theorem arithmetic_sequence_sum (d : ℤ) : ∃ (S : ℕ → ℤ) (a : ℕ → ℤ), 
  (∀ n, a (n + 1) = a n + d) ∧  -- Arithmetic sequence definition
  (∀ n, S n = n * a 1 + (n * (n - 1) / 2) * d) ∧  -- Sum formula
  a 1 = 190 ∧  -- First term
  S 20 > 0 ∧  -- S₂₀ > 0
  S 24 < 0 ∧  -- S₂₄ < 0
  d = -17  -- One possible value for d
  := by sorry

end arithmetic_sequence_sum_l3997_399765


namespace white_marbles_count_l3997_399718

theorem white_marbles_count (total : ℕ) (blue : ℕ) (red : ℕ) (prob_red_or_white : ℚ) 
  (h1 : total = 30)
  (h2 : blue = 5)
  (h3 : red = 9)
  (h4 : prob_red_or_white = 25/30) :
  total - (blue + red) = 16 := by
  sorry

end white_marbles_count_l3997_399718


namespace unique_solution_for_all_z_l3997_399774

theorem unique_solution_for_all_z (x : ℚ) : 
  (∀ z : ℚ, 10 * x * z - 15 * z + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := by
sorry

end unique_solution_for_all_z_l3997_399774


namespace complementary_angles_ratio_l3997_399731

theorem complementary_angles_ratio (a b : ℝ) : 
  a > 0 → b > 0 → -- angles are positive
  a + b = 90 → -- angles are complementary
  a = 4 * b → -- ratio of angles is 4:1
  b = 18 := by
sorry

end complementary_angles_ratio_l3997_399731


namespace isabella_exchange_l3997_399741

/-- Exchange rate from U.S. dollars to Euros -/
def exchange_rate : ℚ := 5 / 8

/-- The amount of Euros Isabella spent -/
def euros_spent : ℕ := 80

theorem isabella_exchange (d : ℕ) : 
  (exchange_rate * d : ℚ) - euros_spent = 2 * d → d = 58 := by
  sorry

end isabella_exchange_l3997_399741


namespace quadratic_completion_sum_l3997_399716

/-- For the quadratic x^2 - 24x + 50, when written as (x+b)^2 + c, b+c equals -106 -/
theorem quadratic_completion_sum (b c : ℝ) : 
  (∀ x, x^2 - 24*x + 50 = (x+b)^2 + c) → b + c = -106 := by
sorry

end quadratic_completion_sum_l3997_399716


namespace complex_fraction_equality_l3997_399751

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_equality : (3 : ℂ) / (1 - i)^2 = (3/2 : ℂ) * i := by
  sorry

end complex_fraction_equality_l3997_399751


namespace dormitory_problem_l3997_399740

theorem dormitory_problem (x : ℕ) 
  (h1 : x > 0) 
  (h2 : 4 * x + 18 < 6 * x) 
  (h3 : 4 * x + 18 > 6 * (x - 1)) : 
  x = 10 ∨ x = 11 := by
sorry

end dormitory_problem_l3997_399740


namespace complementary_angles_are_acute_l3997_399726

/-- Two angles are complementary if their sum is 90 degrees -/
def complementary (a b : ℝ) : Prop := a + b = 90

/-- An angle is acute if it is less than 90 degrees -/
def acute (a : ℝ) : Prop := a < 90

/-- For any two complementary angles, both angles are always acute -/
theorem complementary_angles_are_acute (a b : ℝ) (h : complementary a b) : 
  acute a ∧ acute b := by sorry

end complementary_angles_are_acute_l3997_399726


namespace remainder_of_98765432101_mod_240_l3997_399784

theorem remainder_of_98765432101_mod_240 :
  98765432101 % 240 = 61 := by sorry

end remainder_of_98765432101_mod_240_l3997_399784


namespace gcf_of_48_and_14_l3997_399709

theorem gcf_of_48_and_14 :
  let n : ℕ := 48
  let m : ℕ := 14
  let lcm_nm : ℕ := 56
  Nat.lcm n m = lcm_nm →
  Nat.gcd n m = 12 := by
sorry

end gcf_of_48_and_14_l3997_399709


namespace triangle_area_angle_relation_l3997_399755

theorem triangle_area_angle_relation (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let A := Real.sqrt 3 / 12 * (a^2 + c^2 - b^2)
  (∃ (B : ℝ), 0 < B ∧ B < π ∧ A = 1/2 * a * c * Real.sin B) → 
  (∃ (B : ℝ), 0 < B ∧ B < π ∧ B = π/6) :=
by sorry

end triangle_area_angle_relation_l3997_399755


namespace equation_solution_l3997_399775

theorem equation_solution :
  ∃ x : ℝ, (x + 6) / (x - 3) = 4 ↔ x = 6 := by
  sorry

end equation_solution_l3997_399775


namespace min_dot_product_l3997_399754

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define a point on the hyperbola in the first quadrant
def point_on_hyperbola (M : ℝ × ℝ) : Prop :=
  hyperbola M.1 M.2 ∧ M.1 > 0 ∧ M.2 > 0

-- Define the tangent line at point M
def tangent_line (M : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop :=
  point_on_hyperbola M ∧ 
  ∃ (t : ℝ), P = (M.1 + t, M.2 + t) ∧ Q = (M.1 - t, M.2 - t)

-- Define P in the first quadrant
def P_in_first_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 > 0

-- Define R on the same asymptote as Q
def R_on_asymptote (Q R : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), R = (k * Q.1, k * Q.2)

-- Theorem statement
theorem min_dot_product 
  (M P Q R : ℝ × ℝ) 
  (h1 : tangent_line M P Q)
  (h2 : P_in_first_quadrant P)
  (h3 : R_on_asymptote Q R) :
  ∃ (min_value : ℝ), 
    (∀ (R' : ℝ × ℝ), R_on_asymptote Q R' → 
      (R'.1 - P.1) * (R'.1 - Q.1) + (R'.2 - P.2) * (R'.2 - Q.2) ≥ min_value) ∧
    min_value = -1/2 :=
sorry

end min_dot_product_l3997_399754


namespace grain_demand_formula_l3997_399756

/-- World grain supply and demand model -/
structure GrainModel where
  S : ℝ  -- World grain supply
  D : ℝ  -- World grain demand
  F : ℝ  -- Production fluctuations
  P : ℝ  -- Population growth
  S0 : ℝ  -- Base supply value
  D0 : ℝ  -- Initial demand value

/-- Conditions for the grain model -/
def GrainModelConditions (m : GrainModel) : Prop :=
  m.S = 0.75 * m.D ∧
  m.S = m.S0 * (1 + m.F) ∧
  m.D = m.D0 * (1 + m.P) ∧
  m.S0 = 1800000

/-- Theorem: Given the conditions, the world grain demand D can be expressed as D = (1,800,000 * (1 + F)) / 0.75 -/
theorem grain_demand_formula (m : GrainModel) (h : GrainModelConditions m) :
  m.D = (1800000 * (1 + m.F)) / 0.75 := by
  sorry


end grain_demand_formula_l3997_399756


namespace smallest_y_in_geometric_sequence_125_l3997_399702

/-- A geometric sequence of three positive integers with product 125 -/
structure GeometricSequence125 where
  x : ℕ+
  y : ℕ+
  z : ℕ+
  geometric : ∃ (r : ℚ), y = x * r ∧ z = y * r
  product : x * y * z = 125

/-- The smallest possible value of y in a geometric sequence of three positive integers with product 125 -/
theorem smallest_y_in_geometric_sequence_125 : 
  ∀ (seq : GeometricSequence125), seq.y ≥ 5 :=
sorry

end smallest_y_in_geometric_sequence_125_l3997_399702


namespace bob_distance_when_meeting_l3997_399789

/-- Prove that Bob walked 8 miles when he met Yolanda given the following conditions:
  - The total distance between X and Y is 17 miles
  - Yolanda starts walking from X to Y
  - Bob starts walking from Y to X one hour after Yolanda
  - Yolanda's walking rate is 3 miles per hour
  - Bob's walking rate is 4 miles per hour
-/
theorem bob_distance_when_meeting (total_distance : ℝ) (yolanda_rate : ℝ) (bob_rate : ℝ) 
  (h1 : total_distance = 17)
  (h2 : yolanda_rate = 3)
  (h3 : bob_rate = 4) :
  ∃ t : ℝ, t > 0 ∧ yolanda_rate * (t + 1) + bob_rate * t = total_distance ∧ bob_rate * t = 8 := by
  sorry


end bob_distance_when_meeting_l3997_399789


namespace base_10_to_base_7_conversion_l3997_399745

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The problem statement --/
theorem base_10_to_base_7_conversion :
  base7ToBase10 [5, 0, 2, 2] = 789 := by
  sorry

end base_10_to_base_7_conversion_l3997_399745


namespace pool_tiles_l3997_399747

theorem pool_tiles (total_needed : ℕ) (blue_tiles : ℕ) (additional_needed : ℕ) 
  (h1 : total_needed = 100)
  (h2 : blue_tiles = 48)
  (h3 : additional_needed = 20) :
  total_needed - additional_needed - blue_tiles = 32 := by
  sorry

#check pool_tiles

end pool_tiles_l3997_399747


namespace birthday_stickers_l3997_399799

/-- Represents the number of stickers Luke has at different stages --/
structure StickerCount where
  initial : ℕ
  bought : ℕ
  birthday : ℕ
  givenAway : ℕ
  used : ℕ
  final : ℕ

/-- Theorem stating the number of stickers Luke got for his birthday --/
theorem birthday_stickers (s : StickerCount) 
  (h1 : s.initial = 20)
  (h2 : s.bought = 12)
  (h3 : s.givenAway = 5)
  (h4 : s.used = 8)
  (h5 : s.final = 39)
  (h6 : s.final = s.initial + s.bought + s.birthday - s.givenAway - s.used) :
  s.birthday = 20 := by
  sorry

end birthday_stickers_l3997_399799


namespace triangle_longest_side_range_l3997_399727

/-- Given a triangle with perimeter 12 and b as the longest side, prove the range of b -/
theorem triangle_longest_side_range (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- positive side lengths
  a + b + c = 12 →         -- perimeter is 12
  b ≥ a ∧ b ≥ c →          -- b is the longest side
  4 < b ∧ b < 6 :=         -- range of b
by sorry

end triangle_longest_side_range_l3997_399727


namespace initial_average_production_l3997_399794

theorem initial_average_production (n : ℕ) (today_production : ℕ) (new_average : ℕ) 
  (h1 : n = 14)
  (h2 : today_production = 90)
  (h3 : new_average = 62) :
  (n * (n + 1) * new_average - n * today_production) / n = 60 := by
  sorry

end initial_average_production_l3997_399794


namespace price_difference_l3997_399708

theorem price_difference (P : ℝ) (P_positive : P > 0) : 
  let new_price := P * 1.2
  let discounted_price := new_price * 0.8
  new_price - discounted_price = P * 0.24 :=
by sorry

end price_difference_l3997_399708


namespace correct_sum_is_45250_l3997_399758

/-- Represents the sum with errors --/
def incorrect_sum : ℕ := 52000

/-- Represents the error in the first number's tens place --/
def tens_error : ℤ := 50

/-- Represents the error in the first number's hundreds place --/
def hundreds_error : ℤ := -300

/-- Represents the error in the second number's thousands place --/
def thousands_error : ℤ := 7000

/-- The total error introduced by the mistakes --/
def total_error : ℤ := tens_error + hundreds_error + thousands_error

/-- The correct sum after adjusting for errors --/
def correct_sum : ℕ := incorrect_sum - total_error.toNat

theorem correct_sum_is_45250 : correct_sum = 45250 := by
  sorry

end correct_sum_is_45250_l3997_399758


namespace savings_account_balance_l3997_399796

theorem savings_account_balance 
  (total : ℕ) 
  (checking : ℕ) 
  (h1 : total = 9844)
  (h2 : checking = 6359) :
  total - checking = 3485 :=
by sorry

end savings_account_balance_l3997_399796


namespace men_sent_to_project_l3997_399759

/-- Represents the number of men sent to another project -/
def men_sent : ℕ := 33

/-- Represents the original number of men -/
def original_men : ℕ := 50

/-- Represents the original number of days to complete the work -/
def original_days : ℕ := 10

/-- Represents the new number of days to complete the work -/
def new_days : ℕ := 30

/-- Theorem stating that given the original conditions and the new completion time,
    the number of men sent to another project is 33 -/
theorem men_sent_to_project :
  (original_men * original_days = (original_men - men_sent) * new_days) →
  men_sent = 33 := by
  sorry


end men_sent_to_project_l3997_399759


namespace marys_garbage_bill_is_164_l3997_399717

/-- Calculates Mary's garbage bill based on the given conditions --/
def calculate_garbage_bill : ℝ :=
  let weeks_in_month : ℕ := 4
  let trash_bin_charge : ℝ := 10
  let recycling_bin_charge : ℝ := 5
  let green_waste_bin_charge : ℝ := 3
  let trash_bins : ℕ := 2
  let recycling_bins : ℕ := 1
  let green_waste_bins : ℕ := 1
  let flat_service_fee : ℝ := 15
  let trash_discount : ℝ := 0.18
  let recycling_discount : ℝ := 0.12
  let green_waste_discount : ℝ := 0.10
  let recycling_fine : ℝ := 20
  let overfilling_fine : ℝ := 15
  let unsorted_green_waste_fine : ℝ := 10
  let late_payment_fee : ℝ := 10

  let weekly_cost : ℝ := trash_bin_charge * trash_bins + recycling_bin_charge * recycling_bins + green_waste_bin_charge * green_waste_bins
  let monthly_cost : ℝ := weekly_cost * weeks_in_month
  let weekly_discount : ℝ := trash_bin_charge * trash_bins * trash_discount + recycling_bin_charge * recycling_bins * recycling_discount + green_waste_bin_charge * green_waste_bins * green_waste_discount
  let monthly_discount : ℝ := weekly_discount * weeks_in_month
  let adjusted_monthly_cost : ℝ := monthly_cost - monthly_discount + flat_service_fee
  let total_fines : ℝ := recycling_fine + overfilling_fine + unsorted_green_waste_fine + late_payment_fee

  adjusted_monthly_cost + total_fines

/-- Theorem stating that Mary's garbage bill is equal to $164 --/
theorem marys_garbage_bill_is_164 : calculate_garbage_bill = 164 := by
  sorry

end marys_garbage_bill_is_164_l3997_399717


namespace arithmetic_square_root_of_sqrt_81_l3997_399734

theorem arithmetic_square_root_of_sqrt_81 : Real.sqrt (Real.sqrt 81) = 9 := by
  sorry

end arithmetic_square_root_of_sqrt_81_l3997_399734


namespace school_referendum_l3997_399730

theorem school_referendum (total_students : ℕ) (first_issue : ℕ) (second_issue : ℕ) (against_both : ℕ)
  (h1 : total_students = 150)
  (h2 : first_issue = 110)
  (h3 : second_issue = 95)
  (h4 : against_both = 15) :
  first_issue + second_issue - (total_students - against_both) = 70 := by
  sorry

end school_referendum_l3997_399730


namespace partial_fraction_sum_l3997_399750

theorem partial_fraction_sum (A B C D E : ℚ) :
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) = 
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5)) →
  A + B + C + D + E = 1 / 30 := by
sorry

end partial_fraction_sum_l3997_399750


namespace polygon_diagonals_l3997_399703

/-- The number of diagonals in a polygon with exterior angles of 10 degrees each -/
def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem polygon_diagonals :
  ∃ n : ℕ,
    n > 0 ∧
    n * 10 = 360 ∧
    num_diagonals n = 594 := by
  sorry

end polygon_diagonals_l3997_399703


namespace largest_multiple_of_8_with_negation_greater_than_neg_80_l3997_399742

theorem largest_multiple_of_8_with_negation_greater_than_neg_80 : 
  ∀ n : ℤ, (∃ k : ℤ, n = 8 * k) → -n > -80 → n ≤ 72 :=
by
  sorry

end largest_multiple_of_8_with_negation_greater_than_neg_80_l3997_399742


namespace square_equation_implies_m_equals_negative_one_l3997_399798

theorem square_equation_implies_m_equals_negative_one :
  (∀ a : ℝ, a^2 + m * a + 1/4 = (a - 1/2)^2) → m = -1 := by
  sorry

end square_equation_implies_m_equals_negative_one_l3997_399798


namespace no_real_roots_l3997_399738

theorem no_real_roots : ∀ x : ℝ, x ≠ 2 → 
  (3 * x^2) / (x - 2) - (3 * x + 8) / 2 + (5 - 9 * x) / (x - 2) + 2 ≠ 0 := by
  sorry

end no_real_roots_l3997_399738


namespace triangle_inequality_squared_l3997_399788

theorem triangle_inequality_squared (a b c : ℝ) 
  (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle : a < b + c ∧ b < a + c ∧ c < a + b) : 
  a^2 < a*b + a*c := by
  sorry

end triangle_inequality_squared_l3997_399788


namespace set_of_multiples_of_six_l3997_399714

def is_closed_under_addition_subtraction (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x + y) ∈ S ∧ (x - y) ∈ S

def smallest_positive (S : Set ℝ) (a : ℝ) : Prop :=
  a ∈ S ∧ a > 0 ∧ ∀ x ∈ S, x > 0 → x ≥ a

theorem set_of_multiples_of_six (S : Set ℝ) :
  S.Nonempty →
  is_closed_under_addition_subtraction S →
  smallest_positive S 6 →
  S = {x : ℝ | ∃ n : ℤ, x = 6 * n} :=
by sorry

end set_of_multiples_of_six_l3997_399714


namespace total_days_2000_to_2003_l3997_399763

def isLeapYear (year : Nat) : Bool :=
  year % 4 == 0 && (year % 100 ≠ 0 || year % 400 == 0)

def daysInYear (year : Nat) : Nat :=
  if isLeapYear year then 366 else 365

def totalDaysInRange (startYear endYear : Nat) : Nat :=
  (List.range (endYear - startYear + 1)).map (fun i => daysInYear (startYear + i))
    |>.sum

theorem total_days_2000_to_2003 :
  totalDaysInRange 2000 2003 = 1461 :=
by
  sorry

end total_days_2000_to_2003_l3997_399763


namespace tank_capacity_l3997_399769

theorem tank_capacity (x : ℝ) 
  (h1 : x / 8 + 120 = x / 2) : x = 320 := by
  sorry

end tank_capacity_l3997_399769


namespace cuboid_surface_area_l3997_399790

/-- Surface area of a cuboid -/
def surface_area (l b h : ℝ) : ℝ := 2 * (l * b + b * h + h * l)

/-- Theorem: The surface area of a cuboid with length 8 cm, breadth 10 cm, and height 12 cm is 592 square centimeters -/
theorem cuboid_surface_area :
  surface_area 8 10 12 = 592 := by
  sorry

end cuboid_surface_area_l3997_399790


namespace total_towels_folded_per_hour_l3997_399700

/-- Represents the number of towels a person can fold in one hour -/
def towels_per_hour (towels : ℕ) (minutes : ℕ) : ℕ :=
  (60 / minutes) * towels

/-- Proves that Jane, Kyla, and Anthony can fold 87 towels together in one hour -/
theorem total_towels_folded_per_hour :
  let jane_rate := towels_per_hour 3 5
  let kyla_rate := towels_per_hour 5 10
  let anthony_rate := towels_per_hour 7 20
  jane_rate + kyla_rate + anthony_rate = 87 := by
  sorry

#eval towels_per_hour 3 5 + towels_per_hour 5 10 + towels_per_hour 7 20

end total_towels_folded_per_hour_l3997_399700


namespace c_2017_value_l3997_399770

/-- Sequence a_n -/
def a : ℕ → ℕ
  | 0 => 3
  | n + 1 => a n + 3

/-- Sequence b_n -/
def b : ℕ → ℕ
  | 0 => 3
  | n + 1 => 3 * b n

/-- Sequence c_n -/
def c (n : ℕ) : ℕ := b (a n - 1)

theorem c_2017_value : c 2016 = 27^2017 := by sorry

end c_2017_value_l3997_399770


namespace carrot_distribution_l3997_399752

theorem carrot_distribution (total : ℕ) (leftover : ℕ) (people : ℕ) : 
  total = 74 → 
  leftover = 2 → 
  people > 1 → 
  people < 72 → 
  (total - leftover) % people = 0 → 
  72 % people = 0 := by
sorry

end carrot_distribution_l3997_399752


namespace total_cost_proof_l3997_399704

def bow_cost : ℕ := 5
def vinegar_cost : ℕ := 2
def baking_soda_cost : ℕ := 1
def num_students : ℕ := 23

def total_cost_per_student : ℕ := bow_cost + vinegar_cost + baking_soda_cost

theorem total_cost_proof : 
  total_cost_per_student * num_students = 184 :=
by sorry

end total_cost_proof_l3997_399704


namespace no_real_solutions_l3997_399786

theorem no_real_solutions : ∀ x y : ℝ, x^2 + 3*y^2 - 4*x - 12*y + 36 ≠ 0 := by
  sorry

end no_real_solutions_l3997_399786


namespace ellipse_m_value_l3997_399764

/-- An ellipse with equation x² + y²/m = 1, foci on x-axis, and major axis twice the minor axis -/
structure Ellipse (m : ℝ) :=
  (equation : ∀ (x y : ℝ), x^2 + y^2/m = 1)
  (foci_on_x_axis : True)  -- This is a placeholder, as we can't directly represent this geometrically
  (major_twice_minor : True)  -- This is a placeholder for the condition

/-- The value of m for the given ellipse properties is 1/4 -/
theorem ellipse_m_value :
  ∀ (m : ℝ), Ellipse m → m = 1/4 := by
  sorry

end ellipse_m_value_l3997_399764


namespace function_identity_l3997_399735

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_identity :
  (∀ x : ℝ, f (x + 1) = x^2 - 2*x) →
  (∀ x : ℝ, f x = x^2 - 4*x + 3) :=
by sorry

end function_identity_l3997_399735


namespace brownie_distribution_l3997_399743

theorem brownie_distribution (columns rows people : ℕ) : 
  columns = 6 → rows = 3 → people = 6 → (columns * rows) / people = 3 := by
  sorry

end brownie_distribution_l3997_399743


namespace cube_square_fraction_inequality_l3997_399711

theorem cube_square_fraction_inequality (s r : ℝ) (hs : s > 0) (hr : r > 0) (hsr : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) := by
  sorry

end cube_square_fraction_inequality_l3997_399711


namespace minimum_detectors_l3997_399707

/-- Represents a position on the board -/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents a detector on the board -/
structure Detector :=
  (pos : Position)

/-- Represents a ship on the board -/
structure Ship :=
  (pos : Position)
  (size : Nat)

def boardSize : Nat := 2015
def shipSize : Nat := 1500

/-- Checks if a detector can detect a ship at a given position -/
def canDetect (d : Detector) (s : Ship) : Prop :=
  d.pos.x ≥ s.pos.x ∧ d.pos.x < s.pos.x + s.size ∧
  d.pos.y ≥ s.pos.y ∧ d.pos.y < s.pos.y + s.size

/-- Checks if a set of detectors can determine the position of any ship -/
def canDetermineShipPosition (detectors : List Detector) : Prop :=
  ∀ (s1 s2 : Ship),
    (∀ (d : Detector), d ∈ detectors → (canDetect d s1 ↔ canDetect d s2)) →
    s1 = s2

theorem minimum_detectors :
  ∃ (detectors : List Detector),
    detectors.length = 1030 ∧
    canDetermineShipPosition detectors ∧
    ∀ (d : List Detector),
      d.length < 1030 →
      ¬ canDetermineShipPosition d :=
sorry

end minimum_detectors_l3997_399707


namespace combined_tax_rate_l3997_399757

/-- Given Mork's and Mindy's tax rates and relative incomes, compute their combined tax rate -/
theorem combined_tax_rate (mork_rate mindy_rate : ℚ) (income_ratio : ℕ) :
  mork_rate = 45/100 →
  mindy_rate = 25/100 →
  income_ratio = 4 →
  (mork_rate + income_ratio * mindy_rate) / (1 + income_ratio) = 29/100 := by
sorry

end combined_tax_rate_l3997_399757


namespace true_propositions_count_l3997_399753

/-- Represents the four propositions about geometric solids -/
inductive GeometricProposition
| RegularPyramidLateralEdges
| RightPrismLateralFaces
| CylinderGeneratrix
| ConeSectionIsoscelesTriangles

/-- Determines if a given geometric proposition is true -/
def isTrue (prop : GeometricProposition) : Bool :=
  match prop with
  | .RegularPyramidLateralEdges => true
  | .RightPrismLateralFaces => false
  | .CylinderGeneratrix => true
  | .ConeSectionIsoscelesTriangles => true

/-- The list of all geometric propositions -/
def allPropositions : List GeometricProposition :=
  [.RegularPyramidLateralEdges, .RightPrismLateralFaces, .CylinderGeneratrix, .ConeSectionIsoscelesTriangles]

/-- Counts the number of true propositions -/
def countTruePropositions (props : List GeometricProposition) : Nat :=
  props.filter isTrue |>.length

/-- Theorem stating that the number of true propositions is 3 -/
theorem true_propositions_count :
  countTruePropositions allPropositions = 3 := by
  sorry


end true_propositions_count_l3997_399753


namespace m_range_condition_l3997_399737

def A : Set ℝ := Set.Ioo (-2) 2
def B (m : ℝ) : Set ℝ := Set.Ici (m - 1)

theorem m_range_condition (m : ℝ) : A ⊆ B m ↔ m ≤ -1 := by
  sorry

end m_range_condition_l3997_399737


namespace two_sided_iced_subcubes_count_l3997_399795

/-- Represents a cube with icing on all sides -/
structure IcedCube where
  size : Nat
  deriving Repr

/-- Counts the number of subcubes with icing on exactly two sides -/
def count_two_sided_iced_subcubes (cube : IcedCube) : Nat :=
  sorry

/-- Theorem stating that a 5×5×5 iced cube has 40 subcubes with icing on exactly two sides -/
theorem two_sided_iced_subcubes_count (cube : IcedCube) (h : cube.size = 5) : 
  count_two_sided_iced_subcubes cube = 40 := by
  sorry

end two_sided_iced_subcubes_count_l3997_399795


namespace solution_satisfies_system_l3997_399776

noncomputable def x₁ (t C₁ C₂ : ℝ) : ℝ :=
  (C₁ + C₂ - 2 * t^2) / (2 * (C₁ - t^2) * (C₂ - t^2))

noncomputable def x₂ (t C₁ C₂ : ℝ) : ℝ :=
  (C₂ - C₁) / (2 * (C₁ - t^2) * (C₂ - t^2))

theorem solution_satisfies_system (t C₁ C₂ : ℝ) :
  deriv (fun t => x₁ t C₁ C₂) t = 2 * (x₁ t C₁ C₂)^2 * t + 2 * (x₂ t C₁ C₂)^2 * t ∧
  deriv (fun t => x₂ t C₁ C₂) t = 4 * (x₁ t C₁ C₂) * (x₂ t C₁ C₂) * t :=
by sorry

end solution_satisfies_system_l3997_399776


namespace protest_jail_time_l3997_399780

/-- Calculates the total combined weeks of jail time given protest conditions --/
theorem protest_jail_time 
  (days_of_protest : ℕ) 
  (num_cities : ℕ) 
  (arrests_per_day_per_city : ℕ) 
  (days_in_jail_before_trial : ℕ) 
  (sentence_weeks : ℕ) 
  (h1 : days_of_protest = 30)
  (h2 : num_cities = 21)
  (h3 : arrests_per_day_per_city = 10)
  (h4 : days_in_jail_before_trial = 4)
  (h5 : sentence_weeks = 2) :
  (days_of_protest * num_cities * arrests_per_day_per_city * days_in_jail_before_trial) / 7 +
  (days_of_protest * num_cities * arrests_per_day_per_city * sentence_weeks) / 2 = 9900 := by
  sorry


end protest_jail_time_l3997_399780


namespace sqrt_equation_l3997_399792

theorem sqrt_equation (n : ℝ) : Real.sqrt (5 + n) = 7 → n = 44 := by sorry

end sqrt_equation_l3997_399792


namespace sphere_surface_area_rectangular_solid_l3997_399736

/-- The surface area of a sphere containing all vertices of a rectangular solid -/
theorem sphere_surface_area_rectangular_solid (l w h : ℝ) (r : ℝ) : 
  l = 2 → w = 1 → h = 2 → 
  r^2 = (l^2 + w^2 + h^2) / 4 →
  4 * Real.pi * r^2 = 9 * Real.pi := by sorry

end sphere_surface_area_rectangular_solid_l3997_399736


namespace min_sum_last_three_digits_equal_l3997_399787

/-- 
Given two positive integers m and n, where n > m ≥ 1, 
and the last three digits of 1978^n and 1978^m are equal,
prove that the minimum value of m + n is 106.
-/
theorem min_sum_last_three_digits_equal (m n : ℕ) : 
  m ≥ 1 → n > m → 
  (1978^n) % 1000 = (1978^m) % 1000 → 
  ∃ (m₀ n₀ : ℕ), m₀ ≥ 1 ∧ n₀ > m₀ ∧ 
    (1978^n₀) % 1000 = (1978^m₀) % 1000 ∧
    m₀ + n₀ = 106 ∧ 
    ∀ (m' n' : ℕ), m' ≥ 1 → n' > m' → 
      (1978^n') % 1000 = (1978^m') % 1000 → 
      m' + n' ≥ 106 :=
by sorry

end min_sum_last_three_digits_equal_l3997_399787


namespace isosceles_triangle_circle_properties_l3997_399715

/-- An isosceles triangle with base 48 and side length 30 -/
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  isIsosceles : base = 48 ∧ side = 30

/-- Properties of the inscribed and circumscribed circles of the isosceles triangle -/
def CircleProperties (t : IsoscelesTriangle) : Prop :=
  ∃ (r R d : ℝ),
    r = 8 ∧  -- radius of inscribed circle
    R = 25 ∧  -- radius of circumscribed circle
    d = 15 ∧  -- distance between centers
    r > 0 ∧ R > 0 ∧ d > 0

/-- Theorem stating the properties of the inscribed and circumscribed circles -/
theorem isosceles_triangle_circle_properties (t : IsoscelesTriangle) :
  CircleProperties t :=
sorry

end isosceles_triangle_circle_properties_l3997_399715


namespace permutations_of_four_objects_l3997_399785

theorem permutations_of_four_objects : Nat.factorial 4 = 24 := by
  sorry

end permutations_of_four_objects_l3997_399785


namespace initial_hours_were_eight_l3997_399701

/-- Represents the highway construction scenario -/
structure HighwayConstruction where
  initial_workforce : ℕ
  total_length : ℕ
  initial_duration : ℕ
  partial_duration : ℕ
  partial_completion : ℚ
  additional_workforce : ℕ
  new_daily_hours : ℕ

/-- Calculates the initial daily working hours -/
def calculate_initial_hours (scenario : HighwayConstruction) : ℚ :=
  (scenario.new_daily_hours * (scenario.initial_workforce + scenario.additional_workforce) * scenario.partial_duration * (1 - scenario.partial_completion)) /
  (scenario.initial_workforce * scenario.partial_duration * scenario.partial_completion)

/-- Theorem stating that the initial daily working hours were 8 -/
theorem initial_hours_were_eight (scenario : HighwayConstruction) 
  (h1 : scenario.initial_workforce = 100)
  (h2 : scenario.total_length = 2)
  (h3 : scenario.initial_duration = 50)
  (h4 : scenario.partial_duration = 25)
  (h5 : scenario.partial_completion = 1/3)
  (h6 : scenario.additional_workforce = 60)
  (h7 : scenario.new_daily_hours = 10) :
  calculate_initial_hours scenario = 8 := by
  sorry

end initial_hours_were_eight_l3997_399701


namespace root_sum_reciprocal_products_l3997_399771

theorem root_sum_reciprocal_products (p q r s : ℂ) : 
  (p^4 + 10*p^3 + 20*p^2 + 15*p + 6 = 0) →
  (q^4 + 10*q^3 + 20*q^2 + 15*q + 6 = 0) →
  (r^4 + 10*r^3 + 20*r^2 + 15*r + 6 = 0) →
  (s^4 + 10*s^3 + 20*s^2 + 15*s + 6 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 10/3 :=
by sorry

end root_sum_reciprocal_products_l3997_399771


namespace no_rectangle_satisfies_conditions_l3997_399728

theorem no_rectangle_satisfies_conditions (p q : ℝ) (hp : p > q) (hq : q > 0) :
  ¬∃ x y : ℝ, x < p ∧ y < q ∧ x + y = (p + q) / 2 ∧ x * y = p * q / 4 := by
  sorry

end no_rectangle_satisfies_conditions_l3997_399728


namespace cosine_equation_solution_l3997_399725

theorem cosine_equation_solution (x : ℝ) : 
  (Real.cos x + 2 * Real.cos (6 * x))^2 = 9 + (Real.sin (3 * x))^2 ↔ 
  ∃ k : ℤ, x = 2 * k * Real.pi := by
sorry

end cosine_equation_solution_l3997_399725


namespace left_handed_jazz_no_glasses_l3997_399767

/-- Represents a club with members having various characteristics -/
structure Club where
  total : Nat
  leftHanded : Nat
  jazzLovers : Nat
  rightHandedJazzDislikers : Nat
  glassesWearers : Nat

/-- The main theorem to be proved -/
theorem left_handed_jazz_no_glasses (c : Club)
  (h_total : c.total = 50)
  (h_left : c.leftHanded = 22)
  (h_jazz : c.jazzLovers = 35)
  (h_right_no_jazz : c.rightHandedJazzDislikers = 5)
  (h_glasses : c.glassesWearers = 10)
  (h_hand_exclusive : c.leftHanded + (c.total - c.leftHanded) = c.total)
  (h_glasses_independent : True) :
  ∃ x : Nat, x = 4 ∧ 
    x = c.leftHanded + c.jazzLovers - c.total + c.rightHandedJazzDislikers - c.glassesWearers :=
sorry


end left_handed_jazz_no_glasses_l3997_399767


namespace toms_age_ratio_l3997_399722

theorem toms_age_ratio (T N : ℝ) (h1 : T > 0) (h2 : N > 0) 
  (h3 : T - N = 2 * (T - 3 * N)) : T / N = 5 := by
  sorry

end toms_age_ratio_l3997_399722


namespace f_is_even_l3997_399706

-- Define a real-valued function g
variable (g : ℝ → ℝ)

-- Define the property of g being an odd function
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Define function f in terms of g
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := |g (x^4)|

-- Define the property of f being an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem statement
theorem f_is_even (g : ℝ → ℝ) (h : IsOdd g) : IsEven (f g) := by
  sorry

end f_is_even_l3997_399706


namespace buses_met_count_l3997_399729

/-- Represents the schedule of buses between Moscow and Voronezh -/
structure BusSchedule where
  moscow_departure_minute : Nat
  voronezh_departure_minute : Nat
  travel_time_hours : Nat

/-- Calculates the number of buses from Voronezh that a bus from Moscow will meet -/
def buses_met (schedule : BusSchedule) : Nat :=
  2 * schedule.travel_time_hours

/-- Theorem stating that a bus from Moscow will meet 16 buses from Voronezh -/
theorem buses_met_count (schedule : BusSchedule) 
  (h1 : schedule.moscow_departure_minute = 0)
  (h2 : schedule.voronezh_departure_minute = 30)
  (h3 : schedule.travel_time_hours = 8) : 
  buses_met schedule = 16 := by
  sorry

#eval buses_met { moscow_departure_minute := 0, voronezh_departure_minute := 30, travel_time_hours := 8 }

end buses_met_count_l3997_399729


namespace xy_system_solution_l3997_399793

theorem xy_system_solution (x y : ℝ) 
  (h1 : x * y = 8)
  (h2 : x^2 * y + x * y^2 + x + y = 80) : 
  x^2 + y^2 = 5104 / 81 := by
sorry

end xy_system_solution_l3997_399793


namespace different_color_chips_probability_l3997_399748

theorem different_color_chips_probability : 
  let total_chips := 20
  let blue_chips := 4
  let red_chips := 3
  let yellow_chips := 2
  let green_chips := 5
  let orange_chips := 6
  let prob_diff_color := 
    (blue_chips / total_chips) * ((total_chips - blue_chips) / total_chips) +
    (red_chips / total_chips) * ((total_chips - red_chips) / total_chips) +
    (yellow_chips / total_chips) * ((total_chips - yellow_chips) / total_chips) +
    (green_chips / total_chips) * ((total_chips - green_chips) / total_chips) +
    (orange_chips / total_chips) * ((total_chips - orange_chips) / total_chips)
  prob_diff_color = 31 / 40 := by
  sorry

end different_color_chips_probability_l3997_399748


namespace recurring_decimal_equiv_recurring_decimal_lowest_terms_l3997_399713

def recurring_decimal : ℚ := 433 / 990

theorem recurring_decimal_equiv : recurring_decimal = 0.4375375375375375375375375375375 := by sorry

theorem recurring_decimal_lowest_terms : ∀ a b : ℤ, (a : ℚ) / b = recurring_decimal → Nat.gcd a.natAbs b.natAbs = 1 := by sorry

end recurring_decimal_equiv_recurring_decimal_lowest_terms_l3997_399713


namespace article_pricing_gain_l3997_399749

/-- Proves that if selling an article at 2/3 of its original price results in a 10% loss,
    then selling it at the original price results in a 35% gain. -/
theorem article_pricing_gain (P : ℝ) (P_pos : P > 0) :
  (2 / 3 : ℝ) * P = (9 / 10 : ℝ) * ((20 / 27 : ℝ) * P) →
  ((P - (20 / 27 : ℝ) * P) / ((20 / 27 : ℝ) * P)) * 100 = 35 := by
sorry

end article_pricing_gain_l3997_399749


namespace arctan_sum_equation_l3997_399744

theorem arctan_sum_equation (n : ℕ) : 
  (n > 0) → 
  (Real.arctan (1/2) + Real.arctan (1/3) + Real.arctan (1/7) + Real.arctan (1/n : ℝ) = π/2) → 
  n = 46 := by
sorry

end arctan_sum_equation_l3997_399744


namespace h_in_terms_of_f_l3997_399797

-- Define the domain of f
def I : Set ℝ := Set.Icc (-3 : ℝ) 3

-- Define f as a function on the interval I
variable (f : I → ℝ)

-- Define h as a function derived from f
def h (x : ℝ) : ℝ := -(f ⟨x + 6, sorry⟩)

-- Theorem statement
theorem h_in_terms_of_f (x : ℝ) : h f x = -f ⟨x + 6, sorry⟩ := by sorry

end h_in_terms_of_f_l3997_399797


namespace chocolate_cost_is_75_cents_l3997_399778

/-- The cost of a candy bar in cents -/
def candy_bar_cost : ℕ := 25

/-- The cost of a pack of juice in cents -/
def juice_cost : ℕ := 50

/-- The total cost in cents for 3 candy bars, 2 pieces of chocolate, and 1 pack of juice -/
def total_cost : ℕ := 275

/-- The number of candy bars purchased -/
def num_candy_bars : ℕ := 3

/-- The number of chocolate pieces purchased -/
def num_chocolates : ℕ := 2

/-- The number of juice packs purchased -/
def num_juice_packs : ℕ := 1

theorem chocolate_cost_is_75_cents :
  ∃ (chocolate_cost : ℕ),
    chocolate_cost * num_chocolates + 
    candy_bar_cost * num_candy_bars + 
    juice_cost * num_juice_packs = total_cost ∧
    chocolate_cost = 75 := by
  sorry

end chocolate_cost_is_75_cents_l3997_399778


namespace power_product_equality_l3997_399710

theorem power_product_equality (a b : ℝ) : a^3 * b^3 = (a*b)^3 := by
  sorry

end power_product_equality_l3997_399710


namespace picture_placement_l3997_399779

/-- Given a wall and a picture with specified widths and offset, calculate the distance from the nearest end of the wall to the nearest edge of the picture. -/
theorem picture_placement (wall_width picture_width offset : ℝ) 
  (hw : wall_width = 25)
  (hp : picture_width = 5)
  (ho : offset = 2) :
  let center := (wall_width - picture_width) / 2
  let distance_to_nearest_edge := center - offset
  distance_to_nearest_edge = 8 := by sorry

end picture_placement_l3997_399779


namespace stratified_sampling_theorem_l3997_399719

/-- Represents a chicken farm with its population -/
structure Farm where
  population : ℕ

/-- Calculates the sample size for a farm given the total population and total sample size -/
def sampleSize (farm : Farm) (totalPopulation : ℕ) (totalSample : ℕ) : ℕ :=
  (farm.population * totalSample) / totalPopulation

theorem stratified_sampling_theorem (farmA farmB farmC : Farm) 
    (h1 : farmA.population = 12000)
    (h2 : farmB.population = 8000)
    (h3 : farmC.population = 4000)
    (totalSample : ℕ)
    (h4 : totalSample = 120) :
  let totalPopulation := farmA.population + farmB.population + farmC.population
  (sampleSize farmA totalPopulation totalSample = 60) ∧
  (sampleSize farmB totalPopulation totalSample = 40) ∧
  (sampleSize farmC totalPopulation totalSample = 20) := by
  sorry

end stratified_sampling_theorem_l3997_399719


namespace min_value_expression_l3997_399777

theorem min_value_expression (m n : ℝ) (h : m - n^2 = 8) :
  58 ≤ m^2 - 3*n^2 + m - 14 := by
  sorry

end min_value_expression_l3997_399777


namespace sin_eighth_integral_l3997_399712

theorem sin_eighth_integral : ∫ x in (0)..(2*Real.pi), (Real.sin x)^8 = (35 * Real.pi) / 64 := by sorry

end sin_eighth_integral_l3997_399712


namespace fox_jeans_price_l3997_399721

/-- The regular price of Fox jeans -/
def F : ℝ := 15

/-- The regular price of Pony jeans -/
def pony_price : ℝ := 18

/-- The discount rate on Pony jeans -/
def pony_discount : ℝ := 0.1

/-- The sum of the two discount rates -/
def total_discount : ℝ := 0.22

/-- The total savings when purchasing 5 pairs of jeans -/
def total_savings : ℝ := 9

/-- The number of Fox jeans purchased -/
def fox_count : ℕ := 3

/-- The number of Pony jeans purchased -/
def pony_count : ℕ := 2

theorem fox_jeans_price :
  F = 15 ∧
  pony_price = 18 ∧
  pony_discount = 0.1 ∧
  total_discount = 0.22 ∧
  total_savings = 9 ∧
  fox_count = 3 ∧
  pony_count = 2 →
  F = 15 := by sorry

end fox_jeans_price_l3997_399721


namespace smaller_number_of_product_323_and_difference_2_l3997_399781

theorem smaller_number_of_product_323_and_difference_2 :
  ∀ x y : ℕ+,
  (x : ℕ) * y = 323 →
  (x : ℕ) - y = 2 →
  y = 17 := by
sorry

end smaller_number_of_product_323_and_difference_2_l3997_399781


namespace multiply_95_105_l3997_399782

theorem multiply_95_105 : 95 * 105 = 9975 := by
  sorry

end multiply_95_105_l3997_399782


namespace smallest_n_for_342_fraction_l3997_399791

/-- Checks if two numbers are relatively prime -/
def are_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- Checks if the decimal representation of m/n contains 342 consecutively -/
def contains_342 (m n : ℕ) : Prop :=
  ∃ k : ℕ, 342 * n ≤ 1000 * k * m ∧ 1000 * k * m < 343 * n

theorem smallest_n_for_342_fraction :
  (∃ n : ℕ, n > 0 ∧
    (∃ m : ℕ, m > 0 ∧ m < n ∧
      are_relatively_prime m n ∧
      contains_342 m n)) ∧
  (∀ n : ℕ, n > 0 →
    (∃ m : ℕ, m > 0 ∧ m < n ∧
      are_relatively_prime m n ∧
      contains_342 m n) →
    n ≥ 331) :=
sorry

end smallest_n_for_342_fraction_l3997_399791


namespace continuity_at_6_delta_formula_l3997_399768

def f (x : ℝ) : ℝ := 3 * x^2 + 7

theorem continuity_at_6 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 6| < δ → |f x - f 6| < ε :=
by
  sorry

theorem delta_formula (ε : ℝ) (h : ε > 0) : 
  ∃ δ > 0, δ = ε / 36 ∧ ∀ x, |x - 6| < δ → |f x - f 6| < ε :=
by
  sorry

end continuity_at_6_delta_formula_l3997_399768


namespace brians_books_l3997_399723

/-- The number of chapters in the first book Brian read -/
def book1_chapters : ℕ := 20

/-- The total number of chapters Brian read -/
def total_chapters : ℕ := 75

/-- The number of identical books Brian read -/
def identical_books : ℕ := 2

theorem brians_books (x : ℕ) : 
  book1_chapters + identical_books * x + (book1_chapters + identical_books * x) / 2 = total_chapters → 
  x = 15 :=
by sorry

end brians_books_l3997_399723


namespace smallest_absolute_value_l3997_399762

theorem smallest_absolute_value : ∀ (a b c : ℤ), 
  a = -3 → b = -2 → c = 1 → 
  |0| < |a| ∧ |0| < |b| ∧ |0| < |c| :=
by
  sorry

end smallest_absolute_value_l3997_399762
