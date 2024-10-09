import Mathlib

namespace length_decreased_by_l718_71810

noncomputable def length_decrease_proof : Prop :=
  let length := 33.333333333333336
  let breadth := length / 2
  let new_length := length - 2.833333333333336
  let new_breadth := breadth + 4
  let original_area := length * breadth
  let new_area := new_length * new_breadth
  (new_area = original_area + 75) ↔ (new_length = length - 2.833333333333336)

theorem length_decreased_by : length_decrease_proof := sorry

end length_decreased_by_l718_71810


namespace quad_equiv_proof_l718_71832

theorem quad_equiv_proof (a b : ℝ) (h : a ≠ 0) (hroot : a * 2019^2 + b * 2019 + 2 = 0) :
  ∃ x : ℝ, a * (x - 1)^2 + b * (x - 1) = -2 ∧ x = 2019 :=
sorry

end quad_equiv_proof_l718_71832


namespace cards_probability_l718_71835

-- Definitions based on conditions
def total_cards := 52
def suits := 4
def cards_per_suit := 13

-- Introducing probabilities for the conditions mentioned
def prob_first := 1
def prob_second := 39 / 52
def prob_third := 26 / 52
def prob_fourth := 13 / 52
def prob_fifth := 26 / 52

-- The problem statement
theorem cards_probability :
  (prob_first * prob_second * prob_third * prob_fourth * prob_fifth) = (3 / 64) :=
by
  sorry

end cards_probability_l718_71835


namespace math_enthusiast_gender_relation_female_success_probability_l718_71842

-- Constants and probabilities
def a : ℕ := 24
def b : ℕ := 36
def c : ℕ := 12
def d : ℕ := 28
def n : ℕ := 100
def P_male_success : ℚ := 3 / 4
def P_female_success : ℚ := 2 / 3
def K_threshold : ℚ := 6.635

-- Computation of K^2
def K_square : ℚ := n * (a * d - b * c) ^ 2 / ((a + b) * (c + d) * (a + c) * (b + d))

-- The first part of the proof comparing K^2 with threshold
theorem math_enthusiast_gender_relation : K_square < K_threshold := sorry

-- The second part calculating given conditions for probability calculation
def P_A : ℚ := (P_male_success ^ 2 * (1 - P_female_success)) + (2 * (1 - P_male_success) * P_male_success * P_female_success)
def P_AB : ℚ := 2 * (1 - P_male_success) * P_male_success * P_female_success
def P_B_given_A : ℚ := P_AB / P_A

theorem female_success_probability : P_B_given_A = 4 / 7 := sorry

end math_enthusiast_gender_relation_female_success_probability_l718_71842


namespace three_equal_mass_piles_l718_71886

theorem three_equal_mass_piles (n : ℕ) (h : n > 3) : 
  (∃ (A B C : Finset ℕ), 
    (A ∪ B ∪ C = Finset.range (n + 1)) ∧ 
    (A ∩ B = ∅) ∧ 
    (A ∩ C = ∅) ∧ 
    (B ∩ C = ∅) ∧ 
    (A.sum id = B.sum id) ∧ 
    (B.sum id = C.sum id)) 
  ↔ (n % 3 = 0 ∨ n % 3 = 2) :=
sorry

end three_equal_mass_piles_l718_71886


namespace sum_difference_of_consecutive_integers_l718_71800

theorem sum_difference_of_consecutive_integers (n : ℤ) :
  let set1 := [(n-3), (n-2), (n-1), n, (n+1), (n+2), (n+3)]
  let set2 := [(n+1), (n+2), (n+3), (n+4), (n+5), (n+6), (n+7)]
  let S1 := set1.sum
  let S2 := set2.sum
  S2 - S1 = 28 :=
by
  let set1 := [(n-3), (n-2), (n-1), n, (n+1), (n+2), (n+3)]
  let set2 := [(n+1), (n+2), (n+3), (n+4), (n+5), (n+6), (n+7)]
  let S1 := set1.sum
  let S2 := set2.sum
  have hS1 : S1 = (n-3) + (n-2) + (n-1) + n + (n+1) + (n+2) + (n+3) := by sorry
  have hS2 : S2 = (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) := by sorry
  have h_diff : S2 - S1 = 28 := by sorry
  exact h_diff

end sum_difference_of_consecutive_integers_l718_71800


namespace oranges_in_bin_l718_71808

theorem oranges_in_bin (initial_oranges : ℕ) (oranges_thrown_away : ℕ) (oranges_added : ℕ) 
  (h1 : initial_oranges = 50) (h2 : oranges_thrown_away = 40) (h3 : oranges_added = 24) 
  : initial_oranges - oranges_thrown_away + oranges_added = 34 := 
by
  -- Simplification and calculation here
  sorry

end oranges_in_bin_l718_71808


namespace max_odd_integers_l718_71801

theorem max_odd_integers (a1 a2 a3 a4 a5 a6 a7 : ℕ) (hpos : ∀ i, i ∈ [a1, a2, a3, a4, a5, a6, a7] → i > 0) 
  (hprod : a1 * a2 * a3 * a4 * a5 * a6 * a7 % 2 = 0) : 
  ∃ l : List ℕ, l.length = 6 ∧ (∀ i, i ∈ l → i % 2 = 1) ∧ ∃ e : ℕ, e % 2 = 0 ∧ e ∈ [a1, a2, a3, a4, a5, a6, a7] :=
by
  sorry

end max_odd_integers_l718_71801


namespace total_money_left_l718_71859

theorem total_money_left (david_start john_start emily_start : ℝ) 
  (david_percent_left john_percent_spent emily_percent_spent : ℝ) : 
  (david_start = 3200) → 
  (david_percent_left = 0.65) → 
  (john_start = 2500) → 
  (john_percent_spent = 0.60) → 
  (emily_start = 4000) → 
  (emily_percent_spent = 0.45) → 
  let david_spent := david_start / (1 + david_percent_left)
  let david_remaining := david_start - david_spent
  let john_remaining := john_start * (1 - john_percent_spent)
  let emily_remaining := emily_start * (1 - emily_percent_spent)
  david_remaining + john_remaining + emily_remaining = 4460.61 :=
by
  sorry

end total_money_left_l718_71859


namespace tens_digit_of_7_pow_2011_l718_71857

-- Define the conditions for the problem
def seven_power := 7
def exponent := 2011
def modulo := 100

-- Define the target function to find the tens digit
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- State the theorem formally
theorem tens_digit_of_7_pow_2011 : tens_digit (seven_power ^ exponent % modulo) = 4 := by
  sorry

end tens_digit_of_7_pow_2011_l718_71857


namespace smallest_integer_consecutive_set_l718_71843

theorem smallest_integer_consecutive_set 
(n : ℤ) (h : 7 * n + 21 > 4 * n) : n > -7 :=
by
  sorry

end smallest_integer_consecutive_set_l718_71843


namespace coordinates_P_wrt_origin_l718_71841

/-- Define a point P with coordinates we are given. -/
def P : ℝ × ℝ := (-1, 2)

/-- State that the coordinates of P with respect to the origin O are (-1, 2). -/
theorem coordinates_P_wrt_origin : P = (-1, 2) :=
by
  -- Proof would go here
  sorry

end coordinates_P_wrt_origin_l718_71841


namespace waste_in_scientific_notation_l718_71829

def water_waste_per_person : ℝ := 0.32
def number_of_people : ℝ := 10^6

def total_daily_waste : ℝ := water_waste_per_person * number_of_people

def scientific_notation (x : ℝ) : Prop :=
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ x = a * 10^n

theorem waste_in_scientific_notation :
  scientific_notation total_daily_waste ∧ total_daily_waste = 3.2 * 10^5 :=
by
  sorry

end waste_in_scientific_notation_l718_71829


namespace concert_cost_l718_71882

-- Definitions of the given conditions
def ticket_price : ℝ := 50.00
def num_tickets : ℕ := 2
def processing_fee_rate : ℝ := 0.15
def parking_fee : ℝ := 10.00
def entrance_fee_per_person : ℝ := 5.00
def num_people : ℕ := 2

-- Function to compute the total cost
def total_cost : ℝ :=
  let ticket_total := num_tickets * ticket_price
  let processing_fee := processing_fee_rate * ticket_total
  let total_with_processing := ticket_total + processing_fee
  let total_with_parking := total_with_processing + parking_fee
  let entrance_fee_total := num_people * entrance_fee_per_person
  total_with_parking + entrance_fee_total

-- The proof statement
theorem concert_cost :
  total_cost = 135.00 :=
by
  -- Using the assumptions defined
  let ticket_total := num_tickets * ticket_price
  let processing_fee := processing_fee_rate * ticket_total
  let total_with_processing := ticket_total + processing_fee
  let total_with_parking := total_with_processing + parking_fee
  let entrance_fee_total := num_people * entrance_fee_per_person
  let final_total := total_with_parking + entrance_fee_total
  
  -- Proving the final total
  show final_total = 135.00
  sorry

end concert_cost_l718_71882


namespace shorten_other_side_area_l718_71850

-- Assuming initial dimensions and given conditions
variable (length1 length2 : ℕ)
variable (new_length : ℕ)
variable (area1 area2 : ℕ)

-- Initial dimensions of the index card
def initial_dimensions (length1 length2 : ℕ) : Prop :=
  length1 = 3 ∧ length2 = 7

-- Area when one side is shortened to a specific new length
def shortened_area (length1 length2 new_length : ℕ) : ℕ :=
  if new_length = length1 - 1 then new_length * length2 else length1 * (length2 - 1)

-- Condition that the area is 15 square inches when one side is shortened
def condition_area_15 (length1 length2 : ℕ) : Prop :=
  (shortened_area length1 length2 (length1 - 1) = 15 ∨
   shortened_area length1 length2 (length2 - 1) = 15)

-- Area when the other side is shortened by 1 inch
def new_area (length1 new_length : ℕ) : ℕ :=
  new_length * (length1 - 1)

-- Proving the final area when the other side is shortened
theorem shorten_other_side_area :
  initial_dimensions length1 length2 →
  condition_area_15 length1 length2 →
  new_area length2 (length2 - 1) = 10 :=
by
  intros hdim hc15
  have hlength1 : length1 = 3 := hdim.1
  have hlength2 : length2 = 7 := hdim.2
  sorry

end shorten_other_side_area_l718_71850


namespace find_k_solution_l718_71878

theorem find_k_solution 
  (k : ℝ)
  (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → |k * x - 4| ≤ 2) : 
  k = 2 :=
sorry

end find_k_solution_l718_71878


namespace expand_product_l718_71871

theorem expand_product (y : ℝ) (h : y ≠ 0) : 
  (3 / 7) * ((7 / y) - 14 * y^3 + 21) = (3 / y) - 6 * y^3 + 9 := 
by 
  sorry

end expand_product_l718_71871


namespace find_n_l718_71827

-- Defining the conditions.
def condition_one : Prop :=
  ∀ (c d : ℕ), 
  (80 * 2 * c = 320) ∧ (80 * 2 * d = 160)

def condition_two : Prop :=
  ∀ (c d : ℕ), 
  (100 * 3 * c = 450) ∧ (100 * 3 * d = 300)

def condition_three (n : ℕ) : Prop :=
  ∀ (c d : ℕ), 
  (40 * 4 * c = n) ∧ (40 * 4 * d = 160)

-- Statement of the proof problem using the conditions.
theorem find_n : 
  condition_one ∧ condition_two ∧ condition_three 160 :=
by
  sorry

end find_n_l718_71827


namespace find_q_l718_71863

noncomputable def Sn (n : ℕ) (d : ℚ) : ℚ :=
  d^2 * (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def Tn (n : ℕ) (d : ℚ) (q : ℚ) : ℚ :=
  d^2 * (1 - q^n) / (1 - q)

theorem find_q (d : ℚ) (q : ℚ) (hd : d ≠ 0) (hq : 0 < q ∧ q < 1) :
  Sn 3 d / Tn 3 d q = 14 → q = 1 / 2 :=
by
  sorry

end find_q_l718_71863


namespace find_smaller_number_l718_71844

theorem find_smaller_number (u v : ℝ) (hu : u > 0) (hv : v > 0)
  (h_ratio : u / v = 3 / 5) (h_sum : u + v = 16) : u = 6 :=
by
  sorry

end find_smaller_number_l718_71844


namespace inequality_proof_l718_71817

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 9 * x * y * z) :
    x / Real.sqrt (x^2 + 2 * y * z + 2) + y / Real.sqrt (y^2 + 2 * z * x + 2) + z / Real.sqrt (z^2 + 2 * x * y + 2) ≥ 1 :=
by
  sorry

end inequality_proof_l718_71817


namespace find_nat_number_l718_71866

theorem find_nat_number (N : ℕ) (d : ℕ) (hd : d < 10) (h : N = 5 * d + d) : N = 25 :=
by
  sorry

end find_nat_number_l718_71866


namespace derivative_f_l718_71879

noncomputable def f (x : ℝ) : ℝ := x + (1 / x)

theorem derivative_f (x : ℝ) (hx : x ≠ 0) :
  deriv f x = 1 - (1 / (x ^ 2)) :=
by
  -- The proof goes here
  sorry

end derivative_f_l718_71879


namespace exponentiation_addition_l718_71897

theorem exponentiation_addition : (3^3)^2 + 1 = 730 := by
  sorry

end exponentiation_addition_l718_71897


namespace inequality_abc_l718_71858

theorem inequality_abc (a b c : ℝ) : a^2 + 4 * b^2 + 8 * c^2 ≥ 3 * a * b + 4 * b * c + 2 * c * a :=
by
  sorry

end inequality_abc_l718_71858


namespace exponential_comparison_l718_71814

theorem exponential_comparison
  (a : ℕ := 3^55)
  (b : ℕ := 4^44)
  (c : ℕ := 5^33) :
  c < a ∧ a < b :=
by
  sorry

end exponential_comparison_l718_71814


namespace points_within_distance_5_l718_71820

noncomputable def distance (x y z : ℝ) : ℝ := Real.sqrt (x^2 + y^2 + z^2)

def within_distance (x y z : ℝ) (d : ℝ) : Prop := distance x y z ≤ d

def A := (1, 1, 1)
def B := (1, 2, 2)
def C := (2, -3, 5)
def D := (3, 0, 4)

theorem points_within_distance_5 :
  within_distance 1 1 1 5 ∧
  within_distance 1 2 2 5 ∧
  ¬ within_distance 2 (-3) 5 5 ∧
  within_distance 3 0 4 5 :=
by {
  sorry
}

end points_within_distance_5_l718_71820


namespace my_and_mothers_ages_l718_71826

-- Definitions based on conditions
noncomputable def my_age (x : ℕ) := x
noncomputable def mothers_age (x : ℕ) := 3 * x
noncomputable def sum_of_ages (x : ℕ) := my_age x + mothers_age x

-- Proposition that needs to be proved
theorem my_and_mothers_ages (x : ℕ) (h : sum_of_ages x = 40) :
  my_age x = 10 ∧ mothers_age x = 30 :=
by
  sorry

end my_and_mothers_ages_l718_71826


namespace Djibo_sister_age_l718_71892

variable (d s : ℕ)
variable (h1 : d = 17)
variable (h2 : d - 5 + (s - 5) = 35)

theorem Djibo_sister_age : s = 28 :=
by sorry

end Djibo_sister_age_l718_71892


namespace swimming_speed_in_still_water_l718_71834

variable (v : ℝ) -- the person's swimming speed in still water

-- Conditions
variable (water_speed : ℝ := 4) -- speed of the water
variable (time : ℝ := 2) -- time taken to swim 12 km against the current
variable (distance : ℝ := 12) -- distance swam against the current

theorem swimming_speed_in_still_water :
  (v - water_speed) = distance / time → v = 10 :=
by
  sorry

end swimming_speed_in_still_water_l718_71834


namespace mango_production_l718_71856

-- Conditions
def num_papaya_trees := 2
def papayas_per_tree := 10
def num_mango_trees := 3
def total_fruits := 80

-- Definition to be proven
def mangos_per_mango_tree : Nat :=
  (total_fruits - num_papaya_trees * papayas_per_tree) / num_mango_trees

theorem mango_production :
  mangos_per_mango_tree = 20 := by
  sorry

end mango_production_l718_71856


namespace find_sample_size_l718_71896

def ratio_A : ℕ := 2
def ratio_B : ℕ := 3
def ratio_C : ℕ := 5
def total_ratio : ℕ := ratio_A + ratio_B + ratio_C
def num_B_selected : ℕ := 24

theorem find_sample_size : ∃ n : ℕ, num_B_selected * total_ratio = ratio_B * n :=
by
  sorry

end find_sample_size_l718_71896


namespace rectangle_equation_l718_71862

-- Given points in the problem, we define the coordinates
def A : ℝ × ℝ := (5, 5)
def B : ℝ × ℝ := (9, 2)
def C (a : ℝ) : ℝ × ℝ := (a, 13)
def D (b : ℝ) : ℝ × ℝ := (15, b)

-- We need to prove that a - b = 1 given the conditions
theorem rectangle_equation (a b : ℝ) (h1 : C a = (a, 13)) (h2 : D b = (15, b)) (h3 : 15 - a = 4) (h4 : 13 - b = 3) : 
     a - b = 1 := 
sorry

end rectangle_equation_l718_71862


namespace angela_sleep_difference_l718_71806

theorem angela_sleep_difference :
  let december_sleep_hours := 6.5
  let january_sleep_hours := 8.5
  let december_days := 31
  let january_days := 31
  (january_sleep_hours * january_days) - (december_sleep_hours * december_days) = 62 :=
by
  sorry

end angela_sleep_difference_l718_71806


namespace marble_distribution_l718_71846

theorem marble_distribution (x : ℚ) (total : ℚ) (boy1 : ℚ) (boy2 : ℚ) (boy3 : ℚ) :
  (4 * x + 2) + (2 * x + 1) + (3 * x) = total → total = 62 →
  boy1 = 4 * x + 2 → boy2 = 2 * x + 1 → boy3 = 3 * x →
  boy1 = 254 / 9 ∧ boy2 = 127 / 9 ∧ boy3 = 177 / 9 :=
by
  sorry

end marble_distribution_l718_71846


namespace bert_toy_phones_l718_71899

theorem bert_toy_phones (P : ℕ) (berts_price_per_phone : ℕ) (berts_earning : ℕ)
                        (torys_price_per_gun : ℕ) (torys_earning : ℕ) (tory_guns : ℕ)
                        (earnings_difference : ℕ)
                        (h1 : berts_price_per_phone = 18)
                        (h2 : torys_price_per_gun = 20)
                        (h3 : tory_guns = 7)
                        (h4 : torys_earning = tory_guns * torys_price_per_gun)
                        (h5 : berts_earning = torys_earning + earnings_difference)
                        (h6 : earnings_difference = 4)
                        (h7 : P = berts_earning / berts_price_per_phone) :
  P = 8 := by sorry

end bert_toy_phones_l718_71899


namespace goldfinch_percentage_l718_71881

def number_of_goldfinches := 6
def number_of_sparrows := 9
def number_of_grackles := 5
def total_birds := number_of_goldfinches + number_of_sparrows + number_of_grackles
def goldfinch_fraction := (number_of_goldfinches : ℚ) / total_birds

theorem goldfinch_percentage : goldfinch_fraction * 100 = 30 := 
by
  sorry

end goldfinch_percentage_l718_71881


namespace max_value_of_a_squared_b_squared_c_squared_l718_71891

theorem max_value_of_a_squared_b_squared_c_squared
  (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_constraint : a + 2 * b + 3 * c = 1) : a^2 + b^2 + c^2 ≤ 1 :=
sorry

end max_value_of_a_squared_b_squared_c_squared_l718_71891


namespace simplify_expression_l718_71853

theorem simplify_expression :
  let a := (1/2)^2
  let b := (1/2)^3
  let c := (1/2)^4
  let d := (1/2)^5
  1 / (1/a + 1/b + 1/c + 1/d) = 1/60 :=
by
  sorry

end simplify_expression_l718_71853


namespace part1_monotonicity_part2_intersection_l718_71852

noncomputable def f (a x : ℝ) : ℝ := -x * Real.exp (a * x + 1)

theorem part1_monotonicity (a : ℝ) : 
  ∃ interval : Set ℝ, 
    (∀ x ∈ interval, ∃ interval' : Set ℝ, 
      (∀ x' ∈ interval', f a x' ≤ f a x) ∧ 
      (∀ x' ∈ Set.univ \ interval', f a x' > f a x)) :=
sorry

theorem part2_intersection (a b x_1 x_2 : ℝ) (h1 : a > 0) (h2 : b ≠ 0)
  (h3 : f a x_1 = -b * Real.exp 1) (h4 : f a x_2 = -b * Real.exp 1)
  (h5 : x_1 ≠ x_2) : 
  - (1 / Real.exp 1) < a * b ∧ a * b < 0 ∧ a * (x_1 + x_2) < -2 :=
sorry

end part1_monotonicity_part2_intersection_l718_71852


namespace minimum_n_for_80_intersections_l718_71898

-- Define what an n-sided polygon is and define the intersection condition
def n_sided_polygon (n : ℕ) : Type := sorry -- definition of n-sided polygon

-- Define the condition when boundaries of two polygons intersect at exactly 80 points
def boundaries_intersect_at (P Q : n_sided_polygon n) (k : ℕ) : Prop := sorry -- definition of boundaries intersecting at exactly k points

theorem minimum_n_for_80_intersections (n : ℕ) :
  (∃ (P Q : n_sided_polygon n), boundaries_intersect_at P Q 80) → (n ≥ 10) :=
sorry

end minimum_n_for_80_intersections_l718_71898


namespace nat_numbers_eq_floor_condition_l718_71880

theorem nat_numbers_eq_floor_condition (a b : ℕ):
  (⌊(a ^ 2 : ℚ) / b⌋₊ + ⌊(b ^ 2 : ℚ) / a⌋₊ = ⌊((a ^ 2 + b ^ 2) : ℚ) / (a * b)⌋₊ + a * b) →
  (b = a ^ 2 + 1) ∨ (a = b ^ 2 + 1) :=
by
  sorry

end nat_numbers_eq_floor_condition_l718_71880


namespace set_equivalence_l718_71813

-- Define the given set using the condition.
def given_set : Set ℕ := {x | x ∈ {x | 0 < x} ∧ x - 3 < 2}

-- Define the enumerated set.
def enumerated_set : Set ℕ := {1, 2, 3, 4}

-- Statement of the proof problem.
theorem set_equivalence : given_set = enumerated_set :=
by
  -- The proof is omitted
  sorry

end set_equivalence_l718_71813


namespace isosceles_base_length_l718_71807

theorem isosceles_base_length (b : ℝ) (h1 : 7 + 7 + b = 23) : b = 9 :=
sorry

end isosceles_base_length_l718_71807


namespace odd_function_sin_cos_product_l718_71864

-- Prove that if the function f(x) = sin(x + α) - 2cos(x - α) is an odd function, then sin(α) * cos(α) = 2/5
theorem odd_function_sin_cos_product (α : ℝ)
  (hf : ∀ x, Real.sin (x + α) - 2 * Real.cos (x - α) = -(Real.sin (-x + α) - 2 * Real.cos (-x - α))) :
  Real.sin α * Real.cos α = 2 / 5 :=
  sorry

end odd_function_sin_cos_product_l718_71864


namespace N2O3_weight_l718_71861

-- Definitions from the conditions
def molecularWeightN : Float := 14.01
def molecularWeightO : Float := 16.00
def molecularWeightN2O3 : Float := (2 * molecularWeightN) + (3 * molecularWeightO)
def moles : Float := 4

-- The main proof problem statement
theorem N2O3_weight (h1 : molecularWeightN = 14.01)
                    (h2 : molecularWeightO = 16.00)
                    (h3 : molecularWeightN2O3 = (2 * molecularWeightN) + (3 * molecularWeightO))
                    (h4 : moles = 4) :
                    (moles * molecularWeightN2O3) = 304.08 :=
by
  sorry

end N2O3_weight_l718_71861


namespace graph_passes_through_point_l718_71805

theorem graph_passes_through_point : ∀ (a : ℝ), a > 0 ∧ a ≠ 1 → (∃ x y, (x, y) = (0, 2) ∧ y = a^x + 1) :=
by
  intros a ha
  use 0
  use 2
  obtain ⟨ha1, ha2⟩ := ha
  have h : a^0 = 1 := by simp
  simp [h]
  sorry

end graph_passes_through_point_l718_71805


namespace find_natural_pairs_l718_71883

theorem find_natural_pairs (m n : ℕ) :
  (n * (n - 1) * (n - 2) * (n - 3) = m * (m - 1)) ↔ (n = 1 ∧ m = 1) ∨ (n = 2 ∧ m = 1) ∨ (n = 3 ∧ m = 1) :=
by sorry

end find_natural_pairs_l718_71883


namespace sqrt3_op_sqrt3_l718_71815

def custom_op (x y : ℝ) : ℝ :=
  (x + y)^2 - (x - y)^2

theorem sqrt3_op_sqrt3 : custom_op (Real.sqrt 3) (Real.sqrt 3) = 12 :=
  sorry

end sqrt3_op_sqrt3_l718_71815


namespace marbles_problem_l718_71831

theorem marbles_problem (initial_marble_tyrone : ℕ) (initial_marble_eric : ℕ) (x : ℝ)
  (h1 : initial_marble_tyrone = 125)
  (h2 : initial_marble_eric = 25)
  (h3 : initial_marble_tyrone - x = 3 * (initial_marble_eric + x)) :
  x = 12.5 := 
sorry

end marbles_problem_l718_71831


namespace bin_expected_value_l718_71895

theorem bin_expected_value (m : ℕ) (h : (21 - 4 * m) / (7 + m) = 1) : m = 3 := 
by {
  sorry
}

end bin_expected_value_l718_71895


namespace polynomial_evaluation_l718_71893

theorem polynomial_evaluation (n : ℕ) (p : ℕ → ℝ) 
  (h_poly : ∀ k, k ≤ n → p k = 1 / (Nat.choose (n + 1) k)) :
  p (n + 1) = if n % 2 = 0 then 1 else 0 :=
by
  sorry

end polynomial_evaluation_l718_71893


namespace license_plate_calculation_l718_71867

def license_plate_count : ℕ :=
  let letter_choices := 26^3
  let first_digit_choices := 5
  let remaining_digit_combinations := 5 * 5
  letter_choices * first_digit_choices * remaining_digit_combinations

theorem license_plate_calculation :
  license_plate_count = 455625 :=
by
  sorry

end license_plate_calculation_l718_71867


namespace smallest_term_4_in_c_seq_l718_71875

noncomputable def a_seq (n : ℕ) : ℕ :=
  if n = 0 then 0 else n

noncomputable def b_seq (n : ℕ) : ℕ :=
  if n = 0 then 0 else n * (n - 1) + 15

noncomputable def c_seq (n : ℕ) : ℚ :=
  if n = 0 then 0 else (b_seq n) / (a_seq n)

theorem smallest_term_4_in_c_seq : 
  ∀ n : ℕ, n > 0 → c_seq 4 ≤ c_seq n :=
sorry

end smallest_term_4_in_c_seq_l718_71875


namespace functional_equation_solution_l718_71848

theorem functional_equation_solution (f g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, g (f (x + y)) = f x + 2 * (x + y) * g y) : 
  (∀ x : ℝ, f x = 0) ∧ (∀ x : ℝ, g x = 0) :=
sorry

end functional_equation_solution_l718_71848


namespace volume_of_cylinder_in_pyramid_l718_71855

theorem volume_of_cylinder_in_pyramid
  (a α : ℝ)
  (sin_alpha : ℝ := Real.sin α)
  (tan_alpha : ℝ := Real.tan α)
  (sin_pi_four_alpha : ℝ := Real.sin (Real.pi / 4 + α))
  (sqrt_two : ℝ := Real.sqrt 2) :
  (π * a^3 * sqrt_two * (Real.sin (2 * α))^3) / (128 * sin_pi_four_alpha^3) =
  (π * a^3 * sqrt_two * (Real.sin (2 * α))^3 / (128 * sin_pi_four_alpha^3)) :=
by
  sorry

end volume_of_cylinder_in_pyramid_l718_71855


namespace two_pow_n_minus_one_prime_imp_n_prime_l718_71823

theorem two_pow_n_minus_one_prime_imp_n_prime (n : ℕ) (h : Nat.Prime (2^n - 1)) : Nat.Prime n := 
sorry

end two_pow_n_minus_one_prime_imp_n_prime_l718_71823


namespace geometric_sequence_a5_value_l718_71860

theorem geometric_sequence_a5_value
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n m : ℕ, a n = a 0 * r ^ n)
  (h_condition : a 3 * a 7 = 8) :
  a 5 = 2 * Real.sqrt 2 ∨ a 5 = -2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequence_a5_value_l718_71860


namespace complementary_angle_l718_71885

theorem complementary_angle (angle_deg : ℕ) (angle_min : ℕ) 
  (h1 : angle_deg = 37) (h2 : angle_min = 38) : 
  exists (comp_deg : ℕ) (comp_min : ℕ), comp_deg = 52 ∧ comp_min = 22 :=
by
  sorry

end complementary_angle_l718_71885


namespace domain_shift_l718_71854

noncomputable def domain := { x : ℝ | 1 ≤ x ∧ x ≤ 4 }
noncomputable def shifted_domain := { x : ℝ | 2 ≤ x ∧ x ≤ 5 }

theorem domain_shift (f : ℝ → ℝ) (h : ∀ x, x ∈ domain ↔ (1 ≤ x ∧ x ≤ 4)) :
  ∀ x, x ∈ shifted_domain ↔ ∃ y, (y = x - 1) ∧ y ∈ domain :=
by
  sorry

end domain_shift_l718_71854


namespace point_outside_circle_l718_71870

theorem point_outside_circle
  (radius : ℝ) (distance : ℝ) (h_radius : radius = 8) (h_distance : distance = 10) :
  distance > radius :=
by sorry

end point_outside_circle_l718_71870


namespace triangle_inequality_l718_71888

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  2 < (a + b) / c + (b + c) / a + (c + a) / b - (a^3 + b^3 + c^3) / (a * b * c) ∧
  (a + b) / c + (b + c) / a + (c + a) / b - (a^3 + b^3 + c^3) / (a * b * c) ≤ 3 :=
sorry

end triangle_inequality_l718_71888


namespace situps_difference_l718_71822

def ken_situps : ℕ := 20
def nathan_situps : ℕ := 2 * ken_situps
def bob_situps : ℕ := (ken_situps + nathan_situps) / 2
def emma_situps : ℕ := bob_situps / 3

theorem situps_difference : 
  (nathan_situps + bob_situps + emma_situps) - ken_situps = 60 := by
  sorry

end situps_difference_l718_71822


namespace line_intersects_x_axis_between_A_and_B_l718_71833

theorem line_intersects_x_axis_between_A_and_B (a : ℝ) :
  (∀ x, (x = 1 ∨ x = 3) → (2 * x + (3 - a) = 0)) ↔ 5 ≤ a ∧ a ≤ 9 :=
by
  sorry

end line_intersects_x_axis_between_A_and_B_l718_71833


namespace Ariana_running_time_l718_71874

theorem Ariana_running_time
  (time_Sadie : ℝ)
  (speed_Sadie : ℝ)
  (speed_Ariana : ℝ)
  (speed_Sarah : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (distance_Sadie := speed_Sadie * time_Sadie)
  (time_Ariana_Sarah := total_time - time_Sadie)
  (distance_Ariana_Sarah := total_distance - distance_Sadie) :
  (6 * (time_Ariana_Sarah - (11 - 6 * (time_Ariana_Sarah / (speed_Ariana + (4 / speed_Sarah)))))
  = (0.5 : ℝ)) :=
by
  sorry

end Ariana_running_time_l718_71874


namespace value_of_X_l718_71837

def M : ℕ := 2024 / 4
def N : ℕ := M / 2
def X : ℕ := M + N

theorem value_of_X : X = 759 := by
  sorry

end value_of_X_l718_71837


namespace extreme_values_l718_71816

def f (x : ℝ) : ℝ := x^3 - 4 * x^2 + 5 * x - 4

theorem extreme_values :
  (∃ (x1 x2 : ℝ), x1 = 1 ∧ x2 = 5 / 3 ∧ f x1 = -2 ∧ f x2 = -58 / 27) ∧ 
  (∃ (a b : ℝ), a = 2 ∧ b = f 2 ∧ (∀ (x : ℝ), (a, b) = (x, f x) → (∀ y : ℝ, y = x - 4))) :=
by
  sorry

end extreme_values_l718_71816


namespace distance_between_city_centers_l718_71884

def distance_on_map : ℝ := 45  -- Distance on the map in cm
def scale_factor : ℝ := 20     -- Scale factor (1 cm : 20 km)

theorem distance_between_city_centers : distance_on_map * scale_factor = 900 := by
  sorry

end distance_between_city_centers_l718_71884


namespace fourth_term_geometric_sequence_l718_71824

theorem fourth_term_geometric_sequence :
  let a := (6: ℝ)^(1/2)
  let b := (6: ℝ)^(1/6)
  let c := (6: ℝ)^(1/12)
  b = a * r ∧ c = a * r^2 → (a * r^3) = 1 := 
by
  sorry

end fourth_term_geometric_sequence_l718_71824


namespace multiplication_factor_average_l718_71868

theorem multiplication_factor_average (a : ℕ) (b : ℕ) (c : ℕ) (F : ℝ) 
  (h1 : a = 7) 
  (h2 : b = 26) 
  (h3 : (c : ℝ) = 130) 
  (h4 : (a * b * F : ℝ) = a * c) :
  F = 5 := 
by 
  sorry

end multiplication_factor_average_l718_71868


namespace quadratic_complete_square_l718_71851

theorem quadratic_complete_square (x p q : ℤ) 
  (h_eq : x^2 - 6 * x + 3 = 0) 
  (h_pq_form : x^2 - 6 * x + (p - x)^2 = q) 
  (h_int : ∀ t, t = p + q) : p + q = 3 := sorry

end quadratic_complete_square_l718_71851


namespace no_valid_2011_matrix_l718_71830

def valid_matrix (A : ℕ → ℕ → ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ 2011 →
    (∀ k, 1 ≤ k ∧ k ≤ 4021 →
      (∃ j, 1 ≤ j ∧ j ≤ 2011 ∧ A i j = k) ∨ (∃ j, 1 ≤ j ∧ j ≤ 2011 ∧ A j i = k))

theorem no_valid_2011_matrix :
  ¬ ∃ A : ℕ → ℕ → ℕ, (∀ i j, 1 ≤ i ∧ i ≤ 2011 ∧ 1 ≤ j ∧ j ≤ 2011 → 1 ≤ A i j ∧ A i j ≤ 4021) ∧ valid_matrix A :=
by
  sorry

end no_valid_2011_matrix_l718_71830


namespace opposite_number_113_is_114_l718_71849

theorem opposite_number_113_is_114 :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 200 → 
  (∀ k, (k + 100) % 200 ≠ 113 → 113 + 100 ≤ 200 → n = 113 →
  (k = 114)) :=
by
  intro n hn h_opposite
  sorry

end opposite_number_113_is_114_l718_71849


namespace curve_defined_by_r_eq_4_is_circle_l718_71821

theorem curve_defined_by_r_eq_4_is_circle : ∀ θ : ℝ, ∃ r : ℝ, r = 4 → ∀ θ : ℝ, r = 4 :=
by
  sorry

end curve_defined_by_r_eq_4_is_circle_l718_71821


namespace find_c_l718_71873

open Real

def vector := (ℝ × ℝ)

def a : vector := (1, 2)
def b : vector := (2, -3)

def is_parallel (v1 v2 : vector) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def is_perpendicular (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_c (c : vector) : 
  (is_parallel (c.1 + a.1, c.2 + a.2) b) ∧ (is_perpendicular c (a.1 + b.1, a.2 + b.2)) → 
  c = (-7 / 9, -20 / 9) := 
by
  sorry

end find_c_l718_71873


namespace weighted_average_is_correct_l718_71838

def bag1_pop_kernels := 60
def bag1_total_kernels := 75
def bag2_pop_kernels := 42
def bag2_total_kernels := 50
def bag3_pop_kernels := 25
def bag3_total_kernels := 100
def bag4_pop_kernels := 77
def bag4_total_kernels := 120
def bag5_pop_kernels := 106
def bag5_total_kernels := 150

noncomputable def weighted_average_percentage : ℚ :=
  ((bag1_pop_kernels / bag1_total_kernels * 100 * bag1_total_kernels) +
   (bag2_pop_kernels / bag2_total_kernels * 100 * bag2_total_kernels) +
   (bag3_pop_kernels / bag3_total_kernels * 100 * bag3_total_kernels) +
   (bag4_pop_kernels / bag4_total_kernels * 100 * bag4_total_kernels) +
   (bag5_pop_kernels / bag5_total_kernels * 100 * bag5_total_kernels)) /
  (bag1_total_kernels + bag2_total_kernels + bag3_total_kernels + bag4_total_kernels + bag5_total_kernels)

theorem weighted_average_is_correct : weighted_average_percentage = 60.61 := 
by
  sorry

end weighted_average_is_correct_l718_71838


namespace negation_of_proposition_divisible_by_2_is_not_even_l718_71877

theorem negation_of_proposition_divisible_by_2_is_not_even :
  (¬ ∀ n : ℕ, n % 2 = 0 → (n % 2 = 0 → n % 2 = 0))
  ↔ ∃ n : ℕ, n % 2 = 0 ∧ n % 2 ≠ 0 := 
  by
    sorry

end negation_of_proposition_divisible_by_2_is_not_even_l718_71877


namespace probability_one_and_three_painted_faces_l718_71865

-- Define the conditions of the problem
def side_length := 5
def total_unit_cubes := side_length^3
def painted_faces := 2
def unit_cubes_one_painted_face := 26
def unit_cubes_three_painted_faces := 4

-- Define the probability statement in Lean
theorem probability_one_and_three_painted_faces :
  (unit_cubes_one_painted_face * unit_cubes_three_painted_faces : ℝ) / (total_unit_cubes * (total_unit_cubes - 1) / 2) = 52 / 3875 :=
by
  sorry

end probability_one_and_three_painted_faces_l718_71865


namespace angle_through_point_l718_71812

theorem angle_through_point : 
  (∃ θ : ℝ, ∃ k : ℤ, θ = 2 * k * Real.pi + 5 * Real.pi / 6 ∧ 
                      ∃ x y : ℝ, x = -Real.sqrt 3 / 2 ∧ y = 1 / 2 ∧ 
                                    y / x = Real.tan θ) := 
sorry

end angle_through_point_l718_71812


namespace total_legs_l718_71825

-- Define the number of octopuses
def num_octopuses : ℕ := 5

-- Define the number of legs per octopus
def legs_per_octopus : ℕ := 8

-- The total number of legs should be num_octopuses * legs_per_octopus
theorem total_legs : num_octopuses * legs_per_octopus = 40 :=
by
  -- The proof is omitted
  sorry

end total_legs_l718_71825


namespace find_length_of_brick_l718_71839

-- Definitions given in the problem
def w : ℕ := 4
def h : ℕ := 2
def SA : ℕ := 112
def surface_area (l w h : ℕ) : ℕ := 2 * l * w + 2 * l * h + 2 * w * h

-- Lean 4 statement for the proof problem
theorem find_length_of_brick (l : ℕ) (h w SA : ℕ) (h_w : w = 4) (h_h : h = 2) (h_SA : SA = 112) :
  surface_area l w h = SA → l = 8 := by
  intros H
  simp [surface_area, h_w, h_h, h_SA] at H
  sorry

end find_length_of_brick_l718_71839


namespace squares_sum_l718_71889

theorem squares_sum {r s : ℝ} (h1 : r * s = 16) (h2 : r + s = 8) : r^2 + s^2 = 32 :=
by
  sorry

end squares_sum_l718_71889


namespace find_n_divides_polynomial_l718_71818

theorem find_n_divides_polynomial :
  ∀ (n : ℕ), 0 < n → (n + 2) ∣ (n^3 + 3 * n + 29) ↔ (n = 1 ∨ n = 3 ∨ n = 13) :=
by
  sorry

end find_n_divides_polynomial_l718_71818


namespace solve_remainder_problem_l718_71890

def remainder_problem : Prop :=
  ∃ (n : ℕ), 
    (n % 481 = 179) ∧ 
    (n % 752 = 231) ∧ 
    (n % 1063 = 359) ∧ 
    (((179 + 231 - 359) % 37) = 14)

theorem solve_remainder_problem : remainder_problem :=
by
  sorry

end solve_remainder_problem_l718_71890


namespace find_function_l718_71845

/-- Any function f : ℝ → ℝ satisfying the two given conditions must be of the form f(x) = cx where |c| ≤ 1. -/
theorem find_function (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, x ≠ 0 → x * (f (x + 1) - f x) = f x)
  (h2 : ∀ x y : ℝ, |f x - f y| ≤ |x - y|) :
  ∃ c : ℝ, (∀ x : ℝ, f x = c * x) ∧ |c| ≤ 1 :=
by
  sorry

end find_function_l718_71845


namespace largest_number_A_l718_71847

theorem largest_number_A (A B C : ℕ) (h1: A = 7 * B + C) (h2: B = C) 
  : A ≤ 48 :=
sorry

end largest_number_A_l718_71847


namespace sum_of_any_three_on_line_is_30_l718_71840

/-- Define the list of numbers from 1 to 19 -/
def numbers := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18, 19]

/-- Define the specific sequence found in the solution -/
def arrangement :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 18,
   17, 16, 15, 14, 13, 12, 11]

/-- Define the function to compute the sum of any three numbers on a straight line -/
def sum_on_line (a b c : ℕ) := a + b + c

theorem sum_of_any_three_on_line_is_30 :
  ∀ i j k : ℕ, 
  i ∈ numbers ∧ j ∈ numbers ∧ k ∈ numbers ∧ (i = 10 ∨ j = 10 ∨ k = 10) →
  sum_on_line i j k = 30 :=
by
  sorry

end sum_of_any_three_on_line_is_30_l718_71840


namespace find_k_l718_71804

theorem find_k : ∃ k : ℕ, (2 * (Real.sqrt (225 + k)) = (Real.sqrt (49 + k) + Real.sqrt (441 + k))) → k = 255 :=
by
  sorry

end find_k_l718_71804


namespace find_C_in_terms_of_D_l718_71819

noncomputable def h (C D x : ℝ) : ℝ := C * x - 3 * D ^ 2
noncomputable def k (D x : ℝ) : ℝ := D * x + 1

theorem find_C_in_terms_of_D (C D : ℝ) (h_eq : h C D (k D 2) = 0) (h_def : ∀ x, h C D x = C * x - 3 * D ^ 2) (k_def : ∀ x, k D x = D * x + 1) (D_ne_neg1 : D ≠ -1) : 
C = (3 * D ^ 2) / (2 * D + 1) := 
by 
  sorry

end find_C_in_terms_of_D_l718_71819


namespace intersection_points_l718_71802

def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2
def parabola2 (x : ℝ) : ℝ := 9 * x^2 + 6 * x + 2

theorem intersection_points :
  {p : ℝ × ℝ | parabola1 p.1 = p.2 ∧ parabola2 p.1 = p.2} = 
  {(-5/3, 17), (0, 2)} :=
by
  sorry

end intersection_points_l718_71802


namespace ratio_of_paper_plates_l718_71828

theorem ratio_of_paper_plates (total_pallets : ℕ) (paper_towels : ℕ) (tissues : ℕ) (paper_cups : ℕ) :
  total_pallets = 20 →
  paper_towels = 20 / 2 →
  tissues = 20 / 4 →
  paper_cups = 1 →
  (total_pallets - (paper_towels + tissues + paper_cups)) / total_pallets = 1 / 5 :=
by
  intros h_total h_towels h_tissues h_cups
  sorry

end ratio_of_paper_plates_l718_71828


namespace six_times_six_l718_71869

-- Definitions based on the conditions
def pattern (n : ℕ) : ℕ := n * 6

-- Theorem statement to be proved
theorem six_times_six : pattern 6 = 36 :=
by {
  sorry
}

end six_times_six_l718_71869


namespace fractions_correct_l718_71809
-- Broader import to ensure all necessary libraries are included.

-- Definitions of the conditions
def batman_homes_termite_ridden : ℚ := 1/3
def batman_homes_collapsing : ℚ := 7/10 * batman_homes_termite_ridden
def robin_homes_termite_ridden : ℚ := 3/7
def robin_homes_collapsing : ℚ := 4/5 * robin_homes_termite_ridden
def joker_homes_termite_ridden : ℚ := 1/2
def joker_homes_collapsing : ℚ := 3/8 * joker_homes_termite_ridden

-- Definitions of the fractions of homes that are termite-ridden but not collapsing
def batman_non_collapsing_fraction : ℚ := batman_homes_termite_ridden - batman_homes_collapsing
def robin_non_collapsing_fraction : ℚ := robin_homes_termite_ridden - robin_homes_collapsing
def joker_non_collapsing_fraction : ℚ := joker_homes_termite_ridden - joker_homes_collapsing

-- Proof statement
theorem fractions_correct :
  batman_non_collapsing_fraction = 1/10 ∧
  robin_non_collapsing_fraction = 3/35 ∧
  joker_non_collapsing_fraction = 5/16 :=
sorry

end fractions_correct_l718_71809


namespace min_value_of_y_l718_71803

noncomputable def y (x : ℝ) : ℝ :=
  2 * Real.sin (Real.pi / 3 - x) - Real.cos (Real.pi / 6 + x)

theorem min_value_of_y : ∃ x : ℝ, y x = -1 := by
  sorry

end min_value_of_y_l718_71803


namespace radius_large_circle_l718_71811

-- Definitions for the conditions
def radius_small_circle : ℝ := 2

def is_tangent_externally (r1 r2 : ℝ) : Prop := -- Definition of external tangency
  r1 + r2 = 4

def is_tangent_internally (R r : ℝ) : Prop := -- Definition of internal tangency
  R - r = 4

-- Setting up the property we need to prove: large circle radius
theorem radius_large_circle
  (R r : ℝ)
  (h1 : r = radius_small_circle)
  (h2 : is_tangent_externally r r)
  (h3 : is_tangent_externally r r)
  (h4 : is_tangent_externally r r)
  (h5 : is_tangent_externally r r)
  (h6 : is_tangent_internally R r) :
  R = 4 :=
by sorry

end radius_large_circle_l718_71811


namespace modified_prism_surface_area_l718_71894

theorem modified_prism_surface_area :
  let original_surface_area := 2 * (2 * 4 + 2 * 5 + 4 * 5)
  let modified_surface_area := original_surface_area + 5
  modified_surface_area = original_surface_area + 5 :=
by
  -- set the original dimensions
  let l := 2
  let w := 4
  let h := 5
  -- calculate original surface area
  let SA_original := 2 * (l * w + l * h + w * h)
  -- calculate modified surface area
  let SA_new := SA_original + 5
  -- assert the relationship
  have : SA_new = SA_original + 5 := rfl
  exact this

end modified_prism_surface_area_l718_71894


namespace find_a_plus_b_l718_71836

variable (a : ℝ) (b : ℝ)
def op (x y : ℝ) : ℝ := x + 2 * y + 3

theorem find_a_plus_b (a b : ℝ) (h1 : op (op (a^3) (a^2)) a = b)
    (h2 : op (a^3) (op (a^2) a) = b) : a + b = 21/8 :=
  sorry

end find_a_plus_b_l718_71836


namespace cube_volume_in_pyramid_l718_71872

noncomputable def pyramid_base_side : ℝ := 2
noncomputable def equilateral_triangle_side : ℝ := 2 * Real.sqrt 2
noncomputable def equilateral_triangle_height : ℝ := Real.sqrt 6
noncomputable def cube_side : ℝ := Real.sqrt 6 / 2
noncomputable def cube_volume : ℝ := (Real.sqrt 6 / 2) ^ 3

theorem cube_volume_in_pyramid : cube_volume = 3 * Real.sqrt 6 / 4 :=
by
  sorry

end cube_volume_in_pyramid_l718_71872


namespace find_q_l718_71887

theorem find_q (p q : ℝ) (h : (-2)^3 - 2*(-2)^2 + p*(-2) + q = 0) : 
  q = 16 + 2 * p :=
sorry

end find_q_l718_71887


namespace correct_conclusion_l718_71876

theorem correct_conclusion :
  ¬ (-(-3)^2 = 9) ∧
  ¬ (-6 / 6 * (1 / 6) = -6) ∧
  ((-3)^2 * abs (-1/3) = 3) ∧
  ¬ (3^2 / 2 = 9 / 4) :=
by
  sorry

end correct_conclusion_l718_71876
