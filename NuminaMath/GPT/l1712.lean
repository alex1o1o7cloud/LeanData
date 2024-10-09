import Mathlib

namespace steven_has_15_more_peaches_than_jill_l1712_171280

-- Definitions based on conditions
def peaches_jill : ℕ := 12
def peaches_jake : ℕ := peaches_jill - 1
def peaches_steven : ℕ := peaches_jake + 16

-- The proof problem
theorem steven_has_15_more_peaches_than_jill : peaches_steven - peaches_jill = 15 := by
  sorry

end steven_has_15_more_peaches_than_jill_l1712_171280


namespace candidate_votes_percentage_l1712_171200

-- Conditions
variables {P : ℝ} 
variables (totalVotes : ℝ := 8000)
variables (differenceVotes : ℝ := 2400)

-- Proof Problem
theorem candidate_votes_percentage (h : ((P / 100) * totalVotes + ((P / 100) * totalVotes + differenceVotes) = totalVotes)) : P = 35 :=
by
  sorry

end candidate_votes_percentage_l1712_171200


namespace positive_integers_ab_divides_asq_bsq_implies_a_eq_b_l1712_171294

theorem positive_integers_ab_divides_asq_bsq_implies_a_eq_b
  (a b : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hdiv : a * b ∣ a^2 + b^2) : a = b := by
  sorry

end positive_integers_ab_divides_asq_bsq_implies_a_eq_b_l1712_171294


namespace larry_initial_money_l1712_171235

theorem larry_initial_money
  (M : ℝ)
  (spent_maintenance : ℝ := 0.04 * M)
  (saved_for_emergencies : ℝ := 0.30 * M)
  (snack_cost : ℝ := 5)
  (souvenir_cost : ℝ := 25)
  (lunch_cost : ℝ := 12)
  (loan_cost : ℝ := 10)
  (remaining_money : ℝ := 368)
  (total_spent : ℝ := snack_cost + souvenir_cost + lunch_cost + loan_cost) :
  M - spent_maintenance - saved_for_emergencies - total_spent = remaining_money →
  M = 636.36 :=
by
  sorry

end larry_initial_money_l1712_171235


namespace percentage_exceed_l1712_171298

theorem percentage_exceed (x y : ℝ) (h : y = x + (0.25 * x)) : (y - x) / x * 100 = 25 :=
by
  sorry

end percentage_exceed_l1712_171298


namespace defect_free_product_probability_is_correct_l1712_171282

noncomputable def defect_free_probability : ℝ :=
  let p1 := 0.2
  let p2 := 0.3
  let p3 := 0.5
  let d1 := 0.95
  let d2 := 0.90
  let d3 := 0.80
  p1 * d1 + p2 * d2 + p3 * d3

theorem defect_free_product_probability_is_correct :
  defect_free_probability = 0.86 :=
by
  sorry

end defect_free_product_probability_is_correct_l1712_171282


namespace max_range_of_temps_l1712_171237

noncomputable def max_temp_range (T1 T2 T3 T4 T5 : ℝ) : ℝ := 
  max (max (max (max T1 T2) T3) T4) T5 - min (min (min (min T1 T2) T3) T4) T5

theorem max_range_of_temps :
  ∀ (T1 T2 T3 T4 T5 : ℝ), 
  (T1 + T2 + T3 + T4 + T5) / 5 = 60 →
  T1 = 40 →
  (max_temp_range T1 T2 T3 T4 T5) = 100 :=
by
  intros T1 T2 T3 T4 T5 Havg Hlowest
  sorry

end max_range_of_temps_l1712_171237


namespace find_b_l1712_171270

-- Define the constants and assumptions
variables {a b c d : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d)

-- The function completes 5 periods between 0 and 2π
def completes_5_periods (b : ℝ) : Prop :=
  (2 * Real.pi) / b = (2 * Real.pi) / 5

theorem find_b (h : completes_5_periods b) : b = 5 :=
sorry

end find_b_l1712_171270


namespace subtraction_problem_solution_l1712_171245

theorem subtraction_problem_solution :
  ∃ x : ℝ, (8 - x) / (9 - x) = 4 / 5 :=
by
  use 4
  sorry

end subtraction_problem_solution_l1712_171245


namespace problem_statement_l1712_171236
noncomputable def a : ℕ := 10
noncomputable def b : ℕ := a^3

theorem problem_statement (a b : ℕ) (a_pos : 0 < a) (b_eq : b = a^3)
    (log_ab : Real.logb a (b : ℝ) = 3) (b_minus_a : b = a + 891) :
    a + b = 1010 :=
by
  sorry

end problem_statement_l1712_171236


namespace sum_consecutive_triangular_sum_triangular_2020_l1712_171290

-- Define triangular numbers
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

-- The theorem to be proved
theorem sum_consecutive_triangular (n : ℕ) : triangular n + triangular (n + 1) = (n + 1)^2 :=
by 
  sorry

-- Applying the theorem for the specific case of n = 2020
theorem sum_triangular_2020 : triangular 2020 + triangular 2021 = 2021^2 :=
by 
  exact sum_consecutive_triangular 2020

end sum_consecutive_triangular_sum_triangular_2020_l1712_171290


namespace class3_qualifies_l1712_171221

/-- Data structure representing a class's tardiness statistics. -/
structure ClassStats where
  mean : ℕ
  median : ℕ
  variance : ℕ
  mode : Option ℕ -- mode is optional because not all classes might have a unique mode.

def class1 : ClassStats := { mean := 3, median := 3, variance := 0, mode := none }
def class2 : ClassStats := { mean := 2, median := 0, variance := 1, mode := none }
def class3 : ClassStats := { mean := 2, median := 0, variance := 2, mode := none }
def class4 : ClassStats := { mean := 0, median := 2, variance := 0, mode := some 2 }

/-- Predicate to check if a class qualifies for the flag, meaning no more than 5 students tardy each day for 5 consecutive days. -/
def qualifies (cs : ClassStats) : Prop :=
  cs.mean = 2 ∧ cs.variance = 2

theorem class3_qualifies : qualifies class3 :=
by
  sorry

end class3_qualifies_l1712_171221


namespace sum_of_cubes_eq_three_l1712_171240

theorem sum_of_cubes_eq_three (k : ℤ) : 
  (1 + 6 * k^3)^3 + (1 - 6 * k^3)^3 + (-6 * k^2)^3 + 1^3 = 3 :=
by 
  sorry

end sum_of_cubes_eq_three_l1712_171240


namespace projectile_reaches_50_first_at_0point5_l1712_171277

noncomputable def height_at_time (t : ℝ) : ℝ := -16 * t^2 + 100 * t

theorem projectile_reaches_50_first_at_0point5 :
  ∃ t : ℝ, (height_at_time t = 50) ∧ (t = 0.5) :=
sorry

end projectile_reaches_50_first_at_0point5_l1712_171277


namespace triangle_angles_l1712_171216

theorem triangle_angles (A B C : ℝ) 
  (h1 : B = 4 * A)
  (h2 : C - B = 27)
  (h3 : A + B + C = 180) : 
  A = 17 ∧ B = 68 ∧ C = 95 :=
by {
  -- Sorry will be replaced once the actual proof is provided
  sorry 
}

end triangle_angles_l1712_171216


namespace part1_part2_l1712_171288

-- Definition of the operation '※'
def operation (a b : ℝ) : ℝ := a^2 - b^2

-- Part 1: Proving 2※(-4) = -12
theorem part1 : operation 2 (-4) = -12 := 
by
  sorry

-- Part 2: Proving the solutions to the equation (x + 5)※3 = 0 are x = -8 and x = -2
theorem part2 : (∃ x : ℝ, operation (x + 5) 3 = 0) ↔ (x = -8 ∨ x = -2) := 
by
  sorry

end part1_part2_l1712_171288


namespace four_digit_integers_with_repeated_digits_l1712_171267

noncomputable def count_four_digit_integers_with_repeated_digits : ℕ := sorry

theorem four_digit_integers_with_repeated_digits : 
  count_four_digit_integers_with_repeated_digits = 1984 :=
sorry

end four_digit_integers_with_repeated_digits_l1712_171267


namespace candy_distribution_l1712_171285

theorem candy_distribution (A B C : ℕ) (x y : ℕ)
  (h1 : A > 2 * B)
  (h2 : B > 3 * C)
  (h3 : A + B + C = 200) :
  (A = 121) ∧ (C = 19) :=
  sorry

end candy_distribution_l1712_171285


namespace quadratic_form_l1712_171246

-- Define the constants b and c based on the problem conditions
def b : ℤ := 900
def c : ℤ := -807300

-- Create a statement that represents the proof goal
theorem quadratic_form (c_eq : c = -807300) (b_eq : b = 900) : c / b = -897 :=
by
  sorry

end quadratic_form_l1712_171246


namespace series_sum_is_6_over_5_l1712_171223

noncomputable def series_sum : ℝ := ∑' n : ℕ, if n % 4 == 0 then 1 / (4^(n/4)) else 
                                          if n % 4 == 1 then 1 / (2 * 4^(n/4)) else 
                                          if n % 4 == 2 then -1 / (4^(n/4) * 4^(1/2)) else 
                                          -1 / (2 * 4^(n/4 + 1/2))

theorem series_sum_is_6_over_5 : series_sum = 6 / 5 := 
  sorry

end series_sum_is_6_over_5_l1712_171223


namespace f_14_52_eq_364_l1712_171261

def f : ℕ → ℕ → ℕ := sorry  -- Placeholder definition

axiom f_xx (x : ℕ) : f x x = x
axiom f_sym (x y : ℕ) : f x y = f y x
axiom f_rec (x y : ℕ) (h : x + y > 0) : (x + y) * f x y = y * f x (x + y)

theorem f_14_52_eq_364 : f 14 52 = 364 := 
by {
  sorry  -- Placeholder for the proof steps
}

end f_14_52_eq_364_l1712_171261


namespace sample_capacity_l1712_171262

theorem sample_capacity 
  (n : ℕ) 
  (model_A : ℕ) 
  (model_B model_C : ℕ) 
  (ratio_A ratio_B ratio_C : ℕ)
  (r_A : ratio_A = 2)
  (r_B : ratio_B = 3)
  (r_C : ratio_C = 5)
  (total_production_ratio : ratio_A + ratio_B + ratio_C = 10)
  (items_model_A : model_A = 15)
  (proportion : (model_A : ℚ) / (ratio_A : ℚ) = (n : ℚ) / 10) :
  n = 75 :=
by sorry

end sample_capacity_l1712_171262


namespace football_game_spectators_l1712_171299

theorem football_game_spectators (total_wristbands wristbands_per_person : ℕ)
  (h1 : total_wristbands = 250) (h2 : wristbands_per_person = 2) : 
  total_wristbands / wristbands_per_person = 125 :=
by
  sorry

end football_game_spectators_l1712_171299


namespace remainder_2503_div_28_l1712_171289

theorem remainder_2503_div_28 : 2503 % 28 = 11 := 
by
  -- The proof goes here
  sorry

end remainder_2503_div_28_l1712_171289


namespace lisa_savings_l1712_171258

-- Define the conditions
def originalPricePerNotebook : ℝ := 3
def numberOfNotebooks : ℕ := 8
def discountRate : ℝ := 0.30
def additionalDiscount : ℝ := 5

-- Define the total savings calculation
def calculateSavings (originalPricePerNotebook : ℝ) (numberOfNotebooks : ℕ) (discountRate : ℝ) (additionalDiscount : ℝ) : ℝ := 
  let totalPriceWithoutDiscount := originalPricePerNotebook * numberOfNotebooks
  let discountedPricePerNotebook := originalPricePerNotebook * (1 - discountRate)
  let totalPriceWith30PercentDiscount := discountedPricePerNotebook * numberOfNotebooks
  let totalPriceWithAllDiscounts := totalPriceWith30PercentDiscount - additionalDiscount
  totalPriceWithoutDiscount - totalPriceWithAllDiscounts

-- Theorem for the proof problem
theorem lisa_savings :
  calculateSavings originalPricePerNotebook numberOfNotebooks discountRate additionalDiscount = 12.20 :=
by
  -- Inserting the proof as sorry
  sorry

end lisa_savings_l1712_171258


namespace rain_probability_in_two_locations_l1712_171296

noncomputable def probability_no_rain_A : ℝ := 0.3
noncomputable def probability_no_rain_B : ℝ := 0.4

-- The probability of raining at a location is 1 - the probability of no rain at that location
noncomputable def probability_rain_A : ℝ := 1 - probability_no_rain_A
noncomputable def probability_rain_B : ℝ := 1 - probability_no_rain_B

-- The rain status in location A and location B are independent
theorem rain_probability_in_two_locations :
  probability_rain_A * probability_rain_B = 0.42 := by
  sorry

end rain_probability_in_two_locations_l1712_171296


namespace arithmetic_seq_sixth_term_l1712_171203

theorem arithmetic_seq_sixth_term
  (a d : ℤ)
  (h1 : a + d = 14)
  (h2 : a + 3 * d = 32) : a + 5 * d = 50 := 
by
  sorry

end arithmetic_seq_sixth_term_l1712_171203


namespace janice_purchase_l1712_171213

theorem janice_purchase (a b c : ℕ) (h1 : a + b + c = 30) (h2 : 30 * a + 200 * b + 300 * c = 3000) : a = 20 :=
sorry

end janice_purchase_l1712_171213


namespace gigi_has_15_jellybeans_l1712_171284

variable (G : ℕ) -- G is the number of jellybeans Gigi has
variable (R : ℕ) -- R is the number of jellybeans Rory has
variable (L : ℕ) -- L is the number of jellybeans Lorelai has eaten

-- Conditions
def condition1 := R = G + 30
def condition2 := L = 3 * (G + R)
def condition3 := L = 180

-- Proof statement
theorem gigi_has_15_jellybeans (G R L : ℕ) (h1 : condition1 G R) (h2 : condition2 G R L) (h3 : condition3 L) : G = 15 := by
  sorry

end gigi_has_15_jellybeans_l1712_171284


namespace find_x_l1712_171256

-- Definitions for the vectors and their relationships
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)
def u (x : ℝ) : ℝ × ℝ := (a.1 + 2 * (b x).1, a.2 + 2 * (b x).2)
def v (x : ℝ) : ℝ × ℝ := (2 * a.1 - (b x).1, 2 * a.2 - (b x).2)

-- Given condition that u is parallel to v
def u_parallel_v (x : ℝ) : Prop := u x = v x

-- Prove that the value of x is 1/2
theorem find_x : ∃ x : ℝ, u_parallel_v x ∧ x = 1 / 2 := 
sorry

end find_x_l1712_171256


namespace regular_polygon_angle_not_divisible_by_five_l1712_171254

theorem regular_polygon_angle_not_divisible_by_five :
  ∃ (n_values : Finset ℕ), n_values.card = 5 ∧
    ∀ n ∈ n_values, 3 ≤ n ∧ n ≤ 15 ∧
      ¬ (∃ k : ℕ, (180 * (n - 2)) / n = 5 * k) := 
by
  sorry

end regular_polygon_angle_not_divisible_by_five_l1712_171254


namespace sum_of_positive_ks_l1712_171244

theorem sum_of_positive_ks :
  ∃ (S : ℤ), S = 39 ∧ ∀ k : ℤ, 
  (∃ α β : ℤ, α * β = 18 ∧ α + β = k) →
  (k > 0 → S = 19 + 11 + 9) := sorry

end sum_of_positive_ks_l1712_171244


namespace integer_solution_l1712_171247

theorem integer_solution (a b : ℤ) (h : 6 * a * b = 9 * a - 10 * b + 303) : a + b = 15 :=
sorry

end integer_solution_l1712_171247


namespace problem_statement_l1712_171211

theorem problem_statement (x y : ℝ) (h₁ : x + y = 5) (h₂ : x * y = 3) : 
  x + (x^2 / y) + (y^2 / x) + y = 95 / 3 := 
sorry

end problem_statement_l1712_171211


namespace direct_proportion_function_l1712_171231

-- Definitions of the given functions
def fA (x : ℝ) : ℝ := 3 * x - 4
def fB (x : ℝ) : ℝ := -2 * x + 1
def fC (x : ℝ) : ℝ := 3 * x
def fD (x : ℝ) : ℝ := 4

-- Direct proportion function definition
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∀ x, f 0 = 0 ∧ (f x) / x = f 1 / 1

-- Prove that fC (x) is the only direct proportion function among the given options
theorem direct_proportion_function :
  is_direct_proportion fC ∧ ¬ is_direct_proportion fA ∧ ¬ is_direct_proportion fB ∧ ¬ is_direct_proportion fD :=
by
  sorry

end direct_proportion_function_l1712_171231


namespace longer_part_length_l1712_171273

-- Conditions
def total_length : ℕ := 180
def diff_length : ℕ := 32

-- Hypothesis for the shorter part of the wire
def shorter_part (x : ℕ) : Prop :=
  x + (x + diff_length) = total_length

-- The goal is to find the longer part's length
theorem longer_part_length (x : ℕ) (h : shorter_part x) : x + diff_length = 106 := by
  sorry

end longer_part_length_l1712_171273


namespace set_intersection_l1712_171259

def A := {x : ℝ | -5 < x ∧ x < 2}
def B := {x : ℝ | |x| < 3}

theorem set_intersection : {x : ℝ | x ∈ A ∧ x ∈ B} = {x : ℝ | -3 < x ∧ x < 2} :=
by
  sorry

end set_intersection_l1712_171259


namespace no_intersect_M1_M2_l1712_171219

theorem no_intersect_M1_M2 (A B : ℤ) : ∃ C : ℤ, 
  ∀ x y : ℤ, (x^2 + A * x + B) ≠ (2 * y^2 + 2 * y + C) := by
  sorry

end no_intersect_M1_M2_l1712_171219


namespace correct_average_l1712_171251

-- let's define the numbers as a list
def numbers : List ℕ := [1200, 1300, 1510, 1520, 1530, 1200]

-- the condition given in the problem: the stated average is 1380
def stated_average : ℕ := 1380

-- given the correct calculation of average, let's write the theorem statement
theorem correct_average : (numbers.foldr (· + ·) 0) / numbers.length = 1460 :=
by
  -- we would prove it here
  sorry

end correct_average_l1712_171251


namespace f_expr_for_nonneg_l1712_171212

-- Define the function f piecewise as per the given conditions
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then
    Real.exp (-x) + 2 * x - 1
  else
    -Real.exp x + 2 * x + 1

-- Prove that for x > 0, f(x) = -e^x + 2x + 1 given the conditions
theorem f_expr_for_nonneg (x : ℝ) (h : x ≥ 0) : f x = -Real.exp x + 2 * x + 1 := by
  sorry

end f_expr_for_nonneg_l1712_171212


namespace max_expression_value_l1712_171260

noncomputable def A : ℝ := 15682 + (1 / 3579)
noncomputable def B : ℝ := 15682 - (1 / 3579)
noncomputable def C : ℝ := 15682 * (1 / 3579)
noncomputable def D : ℝ := 15682 / (1 / 3579)
noncomputable def E : ℝ := 15682.3579

theorem max_expression_value :
  D = 56109138 ∧ D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end max_expression_value_l1712_171260


namespace first_group_hours_per_day_l1712_171248

theorem first_group_hours_per_day :
  ∃ H : ℕ, 
    (39 * 12 * H = 30 * 26 * 3) ∧
    H = 5 :=
by sorry

end first_group_hours_per_day_l1712_171248


namespace john_not_stronger_than_ivan_l1712_171281

-- Define strength relations
axiom stronger (a b : Type) : Prop

variable (whiskey liqueur vodka beer : Type)

axiom whiskey_stronger_than_vodka : stronger whiskey vodka
axiom liqueur_stronger_than_beer : stronger liqueur beer

-- Define types for cocktails and their strengths
variable (John_cocktail Ivan_cocktail : Type)

axiom John_mixed_whiskey_liqueur : John_cocktail
axiom Ivan_mixed_vodka_beer : Ivan_cocktail

-- Prove that it can't be asserted that John's cocktail is stronger
theorem john_not_stronger_than_ivan :
  ¬ (stronger John_cocktail Ivan_cocktail) :=
sorry

end john_not_stronger_than_ivan_l1712_171281


namespace divides_expression_l1712_171266

theorem divides_expression (y : ℕ) (h : y ≠ 0) : (y - 1) ∣ (y ^ (y ^ 2) - 2 * y ^ (y + 1) + 1) :=
sorry

end divides_expression_l1712_171266


namespace unique_triple_l1712_171205

theorem unique_triple (x y z : ℤ) (h₁ : x + y = z) (h₂ : y + z = x) (h₃ : z + x = y) :
  (x = 0) ∧ (y = 0) ∧ (z = 0) :=
sorry

end unique_triple_l1712_171205


namespace train_travel_distance_l1712_171283

theorem train_travel_distance (speed time: ℕ) (h1: speed = 85) (h2: time = 4) : speed * time = 340 :=
by
-- Given: speed = 85 km/hr and time = 4 hr
-- To prove: speed * time = 340
-- Since speed = 85 and time = 4, then 85 * 4 = 340
sorry

end train_travel_distance_l1712_171283


namespace angles_between_plane_and_catheti_l1712_171224

theorem angles_between_plane_and_catheti
  (α β : ℝ)
  (h_alpha : 0 < α ∧ α < π / 2)
  (h_beta : 0 < β ∧ β < π / 2) :
  ∃ γ θ : ℝ,
    γ = Real.arcsin (Real.sin β * Real.cos α) ∧
    θ = Real.arcsin (Real.sin β * Real.sin α) :=
by
  sorry

end angles_between_plane_and_catheti_l1712_171224


namespace pants_and_coat_cost_l1712_171287

noncomputable def pants_shirt_costs : ℕ := 100
noncomputable def coat_cost_times_shirt : ℕ := 5
noncomputable def coat_cost : ℕ := 180

theorem pants_and_coat_cost (p s c : ℕ) 
  (h1 : p + s = pants_shirt_costs)
  (h2 : c = coat_cost_times_shirt * s)
  (h3 : c = coat_cost) :
  p + c = 244 :=
by
  sorry

end pants_and_coat_cost_l1712_171287


namespace zeroes_y_minus_a_l1712_171217

open Real

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 2 then |2 ^ x - 1| else 3 / (x - 1)

theorem zeroes_y_minus_a (a : ℝ) : (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧ f x₁ = a ∧ f x₂ = a ∧ f x₃ = a) → (0 < a ∧ a < 1) :=
sorry

end zeroes_y_minus_a_l1712_171217


namespace tan_alpha_plus_pi_over_4_equals_3_over_22_l1712_171215

theorem tan_alpha_plus_pi_over_4_equals_3_over_22
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  Real.tan (α + π / 4) = 3 / 22 :=
sorry

end tan_alpha_plus_pi_over_4_equals_3_over_22_l1712_171215


namespace prime_factorization_sum_l1712_171238

theorem prime_factorization_sum (w x y z k : ℕ) (h : 2^w * 3^x * 5^y * 7^z * 11^k = 2310) :
  2 * w + 3 * x + 5 * y + 7 * z + 11 * k = 28 :=
sorry

end prime_factorization_sum_l1712_171238


namespace percent_students_two_novels_l1712_171292

theorem percent_students_two_novels :
  let total_students := 240
  let students_three_or_more := (1/6 : ℚ) * total_students
  let students_one := (5/12 : ℚ) * total_students
  let students_none := 16
  let students_two := total_students - students_three_or_more - students_one - students_none
  (students_two / total_students) * 100 = 35 := 
by
  sorry

end percent_students_two_novels_l1712_171292


namespace angles_around_point_sum_l1712_171230

theorem angles_around_point_sum 
  (x y : ℝ)
  (h1 : 130 + x + y = 360)
  (h2 : y = x + 30) :
  x = 100 ∧ y = 130 :=
by
  sorry

end angles_around_point_sum_l1712_171230


namespace max_profit_at_max_price_l1712_171218

-- Definitions based on the given problem's conditions
def cost_price : ℝ := 30
def profit_margin : ℝ := 0.5
def max_price : ℝ := cost_price * (1 + profit_margin)
def min_price : ℝ := 35
def base_sales : ℝ := 350
def sales_decrease_per_price_increase : ℝ := 50
def price_increase_step : ℝ := 5

-- Profit function based on the conditions
def profit (x : ℝ) : ℝ := (-10 * x^2 + 1000 * x - 21000)

-- Maximum profit and corresponding price
theorem max_profit_at_max_price :
  ∀ x, min_price ≤ x ∧ x ≤ max_price →
  profit x ≤ profit max_price ∧ profit max_price = 3750 :=
by sorry

end max_profit_at_max_price_l1712_171218


namespace sequence_sum_a1_a3_l1712_171252

theorem sequence_sum_a1_a3 (S : ℕ → ℕ) (a : ℕ → ℤ) 
  (h1 : ∀ n, n ≥ 2 → S n + S (n - 1) = 2 * n - 1) 
  (h2 : S 2 = 3) : 
  a 1 + a 3 = -1 := by
  sorry

end sequence_sum_a1_a3_l1712_171252


namespace lamps_on_after_n2_minus_1_lamps_on_after_n2_minus_n_plus_1_l1712_171204

def lamps_on_again (n : ℕ) (steps : ℕ → Bool → Bool) : ∃ M : ℕ, ∀ s, (s ≥ M) → (n > 1 → ∀ i : ℕ, steps i true = true) := 
sorry

theorem lamps_on_after_n2_minus_1 (n : ℕ) (k : ℕ) (hk : n = 2^k) (steps : ℕ → Bool → Bool) : 
∀ s, (s ≥ n^2 - 1) → (n > 1 → ∀ i : ℕ, steps i true = true) := 
sorry

theorem lamps_on_after_n2_minus_n_plus_1 (n : ℕ) (k : ℕ) (hk : n = 2^k + 1) (steps : ℕ → Bool → Bool) : 
∀ s, (s ≥ n^2 - n + 1) → (n > 1 → ∀ i : ℕ, steps i true = true) := 
sorry

end lamps_on_after_n2_minus_1_lamps_on_after_n2_minus_n_plus_1_l1712_171204


namespace correct_growth_equation_l1712_171263

-- Define the parameters
def initial_income : ℝ := 2.36
def final_income : ℝ := 2.7
def growth_period : ℕ := 2

-- Define the growth rate x
variable (x : ℝ)

-- The theorem we want to prove
theorem correct_growth_equation : initial_income * (1 + x)^growth_period = final_income :=
sorry

end correct_growth_equation_l1712_171263


namespace mason_internet_speed_l1712_171229

-- Definitions based on the conditions
def total_data : ℕ := 880
def downloaded_data : ℕ := 310
def remaining_time : ℕ := 190

-- Statement: The speed of Mason's Internet connection after it slows down
theorem mason_internet_speed :
  (total_data - downloaded_data) / remaining_time = 3 :=
by
  sorry

end mason_internet_speed_l1712_171229


namespace recipe_required_ingredients_l1712_171279

-- Define the number of cups required for each ingredient in the recipe
def sugar_cups : Nat := 11
def flour_cups : Nat := 8
def cocoa_cups : Nat := 5

-- Define the cups of flour and cocoa already added
def flour_already_added : Nat := 3
def cocoa_already_added : Nat := 2

-- Define the cups of flour and cocoa that still need to be added
def flour_needed_to_add : Nat := 6
def cocoa_needed_to_add : Nat := 3

-- Sum the total amount of flour and cocoa powder based on already added and still needed amounts
def total_flour: Nat := flour_already_added + flour_needed_to_add
def total_cocoa: Nat := cocoa_already_added + cocoa_needed_to_add

-- Total ingredients calculation according to the problem's conditions
def total_ingredients : Nat := sugar_cups + total_flour + total_cocoa

-- The theorem to be proved
theorem recipe_required_ingredients : total_ingredients = 24 := by
  sorry

end recipe_required_ingredients_l1712_171279


namespace circle_eq1_circle_eq2_l1712_171255

-- Problem 1: Circle with center M(-5, 3) and passing through point A(-8, -1)
theorem circle_eq1 : ∀ (x y : ℝ), (x + 5) ^ 2 + (y - 3) ^ 2 = 25 :=
by
  sorry

-- Problem 2: Circle passing through three points A(-2, 4), B(-1, 3), C(2, 6)
theorem circle_eq2 : ∀ (x y : ℝ), x ^ 2 + (y - 5) ^ 2 = 5 :=
by
  sorry

end circle_eq1_circle_eq2_l1712_171255


namespace man_l1712_171222

variable (V_m V_c : ℝ)

theorem man's_speed_against_current :
  (V_m + V_c = 21 ∧ V_c = 2.5) → (V_m - V_c = 16) :=
by
  sorry

end man_l1712_171222


namespace total_cost_to_plant_flowers_l1712_171210

noncomputable def flower_cost : ℕ := 9
noncomputable def clay_pot_cost : ℕ := flower_cost + 20
noncomputable def soil_bag_cost : ℕ := flower_cost - 2
noncomputable def total_cost : ℕ := flower_cost + clay_pot_cost + soil_bag_cost

theorem total_cost_to_plant_flowers : total_cost = 45 := by
  sorry

end total_cost_to_plant_flowers_l1712_171210


namespace maria_dozen_flowers_l1712_171257

theorem maria_dozen_flowers (x : ℕ) (h : 12 * x + 2 * x = 42) : x = 3 :=
by
  sorry

end maria_dozen_flowers_l1712_171257


namespace cheyenne_clay_pots_l1712_171239

theorem cheyenne_clay_pots (P : ℕ) (cracked_ratio sold_ratio : ℝ) (total_revenue price_per_pot : ℝ) 
    (P_sold : ℕ) :
  cracked_ratio = (2 / 5) →
  sold_ratio = (3 / 5) →
  total_revenue = 1920 →
  price_per_pot = 40 →
  P_sold = 48 →
  (sold_ratio * P = P_sold) →
  P = 80 :=
by
  sorry

end cheyenne_clay_pots_l1712_171239


namespace hamburgers_made_l1712_171226

theorem hamburgers_made (initial_hamburgers additional_hamburgers total_hamburgers : ℝ)
    (h_initial : initial_hamburgers = 9.0)
    (h_additional : additional_hamburgers = 3.0)
    (h_total : total_hamburgers = initial_hamburgers + additional_hamburgers) :
    total_hamburgers = 12.0 :=
by
    sorry

end hamburgers_made_l1712_171226


namespace part1_part2_part3_l1712_171201

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x + 1

theorem part1 (a : ℝ) (x : ℝ) (h : 0 < x) :
  (a ≤ 0 → (∀ x > 0, f a x < 0)) ∧
  (a > 0 → (∀ x ∈ Set.Ioo 0 a, f a x > 0) ∧ (∀ x ∈ Set.Ioi a, f a x < 0)) :=
sorry

theorem part2 {a : ℝ} : (∀ x > 0, f a x ≤ 0) → a = 1 :=
sorry

theorem part3 (n : ℕ) (h : 0 < n) :
  (1 + 1 / n : ℝ)^n < Real.exp 1 ∧ Real.exp 1 < (1 + 1 / n : ℝ)^(n + 1) :=
sorry

end part1_part2_part3_l1712_171201


namespace main_theorem_l1712_171274

noncomputable def circle_center : Prop :=
  ∃ x y : ℝ, 2*x - y - 7 = 0 ∧ y = -3 ∧ x = 2

noncomputable def circle_equation : Prop :=
  (∀ (x y : ℝ), (x - 2)^2 + (y + 3)^2 = 5)

noncomputable def tangent_condition (k : ℝ) : Prop :=
  (3 + 3*k)^2 / (1 + k^2) = 5

noncomputable def symmetric_circle_center : Prop :=
  ∃ x y : ℝ, x = -22/5 ∧ y = 1/5

noncomputable def symmetric_circle_equation : Prop :=
  (∀ (x y : ℝ), (x + 22/5)^2 + (y - 1/5)^2 = 5)

theorem main_theorem : circle_center → circle_equation ∧ (∃ k : ℝ, tangent_condition k) ∧ symmetric_circle_center → symmetric_circle_equation :=
  by sorry

end main_theorem_l1712_171274


namespace total_sum_lent_l1712_171278

theorem total_sum_lent (x : ℝ) (second_part : ℝ) (total_sum : ℝ) 
  (h1 : second_part = 1640) 
  (h2 : (x * 8 * 0.03) = (second_part * 3 * 0.05)) :
  total_sum = x + second_part → total_sum = 2665 := by
  sorry

end total_sum_lent_l1712_171278


namespace alice_walking_speed_l1712_171242

theorem alice_walking_speed:
  ∃ v : ℝ, 
  (∀ t : ℝ, t = 1 → ∀ d_a d_b : ℝ, d_a = 25 → d_b = 41 - d_a → 
  ∀ s_b : ℝ, s_b = 4 → 
  d_b / s_b + t = d_a / v) ∧ v = 5 :=
by
  sorry

end alice_walking_speed_l1712_171242


namespace pinky_pig_apples_l1712_171291

variable (P : ℕ)

theorem pinky_pig_apples (h : P + 73 = 109) : P = 36 := sorry

end pinky_pig_apples_l1712_171291


namespace mangoes_in_shop_l1712_171243

-- Define the conditions
def ratio_mango_to_apple := 10 / 3
def apples := 36

-- Problem statement to prove
theorem mangoes_in_shop : ∃ (m : ℕ), m = 120 ∧ m = apples * ratio_mango_to_apple :=
by
  sorry

end mangoes_in_shop_l1712_171243


namespace polynomial_real_root_condition_l1712_171228

theorem polynomial_real_root_condition (b : ℝ) :
    (∃ x : ℝ, x^4 + b * x^3 + x^2 + b * x - 1 = 0) ↔ (b ≥ 1 / 2) :=
by sorry

end polynomial_real_root_condition_l1712_171228


namespace ceil_square_of_neg_seven_fourths_l1712_171220

/-- Evaluate the ceiling of the square of -7/4 --/
theorem ceil_square_of_neg_seven_fourths : (Int.ceil ((-7/4 : ℚ)^2 : ℚ) = 4) :=
sorry

end ceil_square_of_neg_seven_fourths_l1712_171220


namespace digit_product_inequality_l1712_171214

noncomputable def digit_count_in_n (n : ℕ) (d : ℕ) : ℕ :=
  (n.digits 10).count d

theorem digit_product_inequality (n : ℕ) (a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℕ)
  (h1 : a1 = digit_count_in_n n 1)
  (h2 : a2 = digit_count_in_n n 2)
  (h3 : a3 = digit_count_in_n n 3)
  (h4 : a4 = digit_count_in_n n 4)
  (h5 : a5 = digit_count_in_n n 5)
  (h6 : a6 = digit_count_in_n n 6)
  (h7 : a7 = digit_count_in_n n 7)
  (h8 : a8 = digit_count_in_n n 8)
  (h9 : a9 = digit_count_in_n n 9)
  : 2^a1 * 3^a2 * 4^a3 * 5^a4 * 6^a5 * 7^a6 * 8^a7 * 9^a8 * 10^a9 ≤ n + 1 :=
  sorry

end digit_product_inequality_l1712_171214


namespace relay_race_time_l1712_171227

-- Define the time it takes for each runner.
def Rhonda_time : ℕ := 24
def Sally_time : ℕ := Rhonda_time + 2
def Diane_time : ℕ := Rhonda_time - 3

-- Define the total time for the relay race.
def total_relay_time : ℕ := Rhonda_time + Sally_time + Diane_time

-- State the theorem we want to prove: the total relay time is 71 seconds.
theorem relay_race_time : total_relay_time = 71 := 
by 
  -- The following "sorry" indicates a step where the proof would be completed.
  sorry

end relay_race_time_l1712_171227


namespace eleven_step_paths_l1712_171241

def H : (ℕ × ℕ) := (0, 0)
def K : (ℕ × ℕ) := (4, 3)
def J : (ℕ × ℕ) := (6, 5)

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem eleven_step_paths (H K J : (ℕ × ℕ)) (H_coords : H = (0, 0)) (K_coords : K = (4, 3)) (J_coords : J = (6, 5)) : 
  (binomial 7 4) * (binomial 4 2) = 210 := by 
  sorry

end eleven_step_paths_l1712_171241


namespace chairs_to_remove_l1712_171206

/-- Given conditions:
1. Each row holds 13 chairs.
2. There are 169 chairs initially.
3. There are 95 expected attendees.

Task: 
Prove that the number of chairs to be removed to ensure complete rows and minimize empty seats is 65. -/
theorem chairs_to_remove (chairs_per_row total_chairs expected_attendees : ℕ)
  (h1 : chairs_per_row = 13)
  (h2 : total_chairs = 169)
  (h3 : expected_attendees = 95) :
  ∃ chairs_to_remove : ℕ, chairs_to_remove = 65 :=
by
  sorry -- proof omitted

end chairs_to_remove_l1712_171206


namespace min_xy_eq_nine_l1712_171286

theorem min_xy_eq_nine (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y = x + y + 3) : x * y = 9 :=
sorry

end min_xy_eq_nine_l1712_171286


namespace average_age_increase_l1712_171275

theorem average_age_increase (n : ℕ) (m : ℕ) (a b : ℝ) (h1 : n = 19) (h2 : m = 20) (h3 : a = 20) (h4 : b = 40) :
  ((n * a + b) / (n + 1)) - a = 1 :=
by
  -- Proof omitted
  sorry

end average_age_increase_l1712_171275


namespace total_cans_l1712_171232

theorem total_cans (total_oil : ℕ) (oil_in_8_liter_cans : ℕ) (number_of_8_liter_cans : ℕ) (remaining_oil : ℕ) 
(oil_per_15_liter_can : ℕ) (number_of_15_liter_cans : ℕ) :
  total_oil = 290 ∧ oil_in_8_liter_cans = 8 ∧ number_of_8_liter_cans = 10 ∧ oil_per_15_liter_can = 15 ∧
  remaining_oil = total_oil - (number_of_8_liter_cans * oil_in_8_liter_cans) ∧
  number_of_15_liter_cans = remaining_oil / oil_per_15_liter_can →
  (number_of_8_liter_cans + number_of_15_liter_cans) = 24 := sorry

end total_cans_l1712_171232


namespace stratified_sampling_grade10_sampled_count_l1712_171297

def total_students : ℕ := 2000
def grade10_students : ℕ := 600
def grade11_students : ℕ := 680
def grade12_students : ℕ := 720
def total_sampled_students : ℕ := 50

theorem stratified_sampling_grade10_sampled_count :
  15 = (total_sampled_students * grade10_students / total_students) :=
by sorry

end stratified_sampling_grade10_sampled_count_l1712_171297


namespace equivalent_problem_l1712_171293

variable (x y : ℝ)
variable (hx_ne_zero : x ≠ 0)
variable (hy_ne_zero : y ≠ 0)
variable (h : (3 * x + y) / (x - 3 * y) = -2)

theorem equivalent_problem : (x + 3 * y) / (3 * x - y) = 2 :=
by
  sorry

end equivalent_problem_l1712_171293


namespace range_of_a_l1712_171249

theorem range_of_a (a : Real) : 
  (∀ x y : Real, (x^2 + y^2 + 2 * a * x - 4 * a * y + 5 * a^2 - 4 = 0 → x < 0 ∧ y > 0)) ↔ (a > 2) := 
sorry

end range_of_a_l1712_171249


namespace new_average_weight_l1712_171208

-- noncomputable theory can be enabled if necessary for real number calculations.
-- noncomputable theory

def original_players : Nat := 7
def original_avg_weight : Real := 103
def new_players : Nat := 2
def weight_first_new_player : Real := 110
def weight_second_new_player : Real := 60

theorem new_average_weight :
  let original_total_weight : Real := original_players * original_avg_weight
  let total_weight : Real := original_total_weight + weight_first_new_player + weight_second_new_player
  let total_players : Nat := original_players + new_players
  total_weight / total_players = 99 := by
  sorry

end new_average_weight_l1712_171208


namespace group_A_can_form_triangle_l1712_171264

def can_form_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem group_A_can_form_triangle : can_form_triangle 9 6 13 :=
by
  sorry

end group_A_can_form_triangle_l1712_171264


namespace mini_toy_height_difference_l1712_171202

variables (H_standard H_toy H_mini_diff : ℝ)

def poodle_heights : Prop :=
  H_standard = 28 ∧ H_toy = 14 ∧ H_standard - 8 = H_mini_diff + H_toy

theorem mini_toy_height_difference (H_standard H_toy H_mini_diff: ℝ) (h: poodle_heights H_standard H_toy H_mini_diff) :
  H_mini_diff = 6 :=
by {
  sorry
}

end mini_toy_height_difference_l1712_171202


namespace isabella_euros_l1712_171268

theorem isabella_euros (d : ℝ) : 
  (5 / 8) * d - 80 = 2 * d → d = 58 :=
by
  sorry

end isabella_euros_l1712_171268


namespace calculate_f_at_2_l1712_171271

def f (x : ℝ) : ℝ := 15 * x ^ 5 - 24 * x ^ 4 + 33 * x ^ 3 - 42 * x ^ 2 + 51 * x

theorem calculate_f_at_2 : f 2 = 294 := by
  sorry

end calculate_f_at_2_l1712_171271


namespace bricks_needed_per_square_meter_l1712_171234

theorem bricks_needed_per_square_meter 
  (num_rooms : ℕ) (room_length room_breadth : ℕ) (total_bricks : ℕ)
  (h1 : num_rooms = 5)
  (h2 : room_length = 4)
  (h3 : room_breadth = 5)
  (h4 : total_bricks = 340) : 
  (total_bricks / (room_length * room_breadth)) = 17 := 
by
  sorry

end bricks_needed_per_square_meter_l1712_171234


namespace least_number_subtraction_l1712_171265

theorem least_number_subtraction (n : ℕ) (h₀ : n = 3830) (k : ℕ) (h₁ : k = 5) : (n - k) % 15 = 0 :=
by {
  sorry
}

end least_number_subtraction_l1712_171265


namespace daily_harvest_l1712_171276

theorem daily_harvest (sacks_per_section : ℕ) (num_sections : ℕ) 
  (h1 : sacks_per_section = 45) (h2 : num_sections = 8) : 
  sacks_per_section * num_sections = 360 :=
by
  sorry

end daily_harvest_l1712_171276


namespace find_sum_of_vars_l1712_171207

-- Definitions of the quadratic polynomials
def quadratic1 (x : ℝ) : ℝ := 2 * x^2 + 8 * x + 11
def quadratic2 (y : ℝ) : ℝ := y^2 - 10 * y + 29
def quadratic3 (z : ℝ) : ℝ := 3 * z^2 - 18 * z + 32

-- Theorem statement
theorem find_sum_of_vars (x y z : ℝ) :
  quadratic1 x * quadratic2 y * quadratic3 z ≤ 60 → x + y - z = 0 :=
by 
-- here we would complete the proof steps
sorry

end find_sum_of_vars_l1712_171207


namespace symmetric_points_l1712_171250

theorem symmetric_points (a b : ℝ) (h1 : 2 * a + 1 = -1) (h2 : 4 = -(3 * b - 1)) :
  2 * a + b = -3 := 
sorry

end symmetric_points_l1712_171250


namespace initial_cookies_count_l1712_171225

def cookies_left : ℕ := 9
def cookies_eaten : ℕ := 9

theorem initial_cookies_count : cookies_left + cookies_eaten = 18 :=
by sorry

end initial_cookies_count_l1712_171225


namespace rachel_remaining_pictures_l1712_171295

theorem rachel_remaining_pictures 
  (p1 p2 p_colored : ℕ)
  (h1 : p1 = 23)
  (h2 : p2 = 32)
  (h3 : p_colored = 44) :
  (p1 + p2 - p_colored = 11) :=
by
  sorry

end rachel_remaining_pictures_l1712_171295


namespace rhombus_side_length_l1712_171209

theorem rhombus_side_length (a b s K : ℝ)
  (h1 : b = 3 * a)
  (h2 : K = (1 / 2) * a * b)
  (h3 : s ^ 2 = (a / 2) ^ 2 + (3 * a / 2) ^ 2) :
  s = Real.sqrt (5 * K / 3) :=
by
  sorry

end rhombus_side_length_l1712_171209


namespace triangle_angle_C_triangle_max_area_l1712_171233

noncomputable def cos (θ : Real) : Real := sorry
noncomputable def sin (θ : Real) : Real := sorry

theorem triangle_angle_C (a b c : Real) (A B C : Real) (h1: 0 < A ∧ A < Real.pi)
  (h2: 0 < B ∧ B < Real.pi) (h3: 0 < C ∧ C < Real.pi)
  (h4: (2 * a + b) * cos C + c * cos B = 0) : C = (2 * Real.pi) / 3 :=
sorry

theorem triangle_max_area (a b c : Real) (A B C : Real) (h1: 0 < A ∧ A < Real.pi)
  (h2: 0 < B ∧ B < Real.pi) (h3: 0 < C ∧ C < Real.pi)
  (h4: (2 * a + b) * cos C + c * cos B = 0) (hc : c = 6)
  (hC : C = (2 * Real.pi) / 3) : 
  ∃ (S : Real), S = 3 * Real.sqrt 3 := 
sorry

end triangle_angle_C_triangle_max_area_l1712_171233


namespace isosceles_triangle_perimeter_l1712_171269

theorem isosceles_triangle_perimeter (a b : ℕ) (h_eq : a = 5 ∨ a = 9) (h_side : b = 9 ∨ b = 5) (h_neq : a ≠ b) : 
  (a + a + b = 19 ∨ a + a + b = 23) :=
by
  sorry

end isosceles_triangle_perimeter_l1712_171269


namespace find_three_leaf_clovers_l1712_171272

-- Define the conditions
def total_leaves : Nat := 1000

-- Define the statement
theorem find_three_leaf_clovers (n : Nat) (h : 3 * n + 4 = total_leaves) : n = 332 :=
  sorry

end find_three_leaf_clovers_l1712_171272


namespace temperature_difference_l1712_171253

def highest_temperature : ℤ := 8
def lowest_temperature : ℤ := -2

theorem temperature_difference :
  highest_temperature - lowest_temperature = 10 := by
  sorry

end temperature_difference_l1712_171253
