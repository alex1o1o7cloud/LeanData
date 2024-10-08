import Mathlib

namespace find_nat_numbers_l37_37472

theorem find_nat_numbers (a b : ℕ) (h : 1 / (a - b) = 3 * (1 / (a * b))) : a = 6 ∧ b = 2 :=
sorry

end find_nat_numbers_l37_37472


namespace joe_anne_bill_difference_l37_37999

theorem joe_anne_bill_difference (m j a : ℝ) 
  (hm : (15 / 100) * m = 3) 
  (hj : (10 / 100) * j = 2) 
  (ha : (20 / 100) * a = 3) : 
  j - a = 5 := 
by {
  sorry
}

end joe_anne_bill_difference_l37_37999


namespace bowler_overs_l37_37321

theorem bowler_overs (x : ℕ) (h1 : ∀ y, y ≤ 3 * x) 
                     (h2 : y = 10) : x = 4 := by
  sorry

end bowler_overs_l37_37321


namespace cos_pi_over_6_minus_a_eq_5_over_12_l37_37678

theorem cos_pi_over_6_minus_a_eq_5_over_12 (a : ℝ) (h : Real.sin (Real.pi / 3 + a) = 5 / 12) :
  Real.cos (Real.pi / 6 - a) = 5 / 12 :=
by
  sorry

end cos_pi_over_6_minus_a_eq_5_over_12_l37_37678


namespace sequence_value_2_l37_37137

/-- 
Given the following sequence:
1 = 6
3 = 18
4 = 24
5 = 30

The sequence follows the pattern that for all n ≠ 6, n is mapped to n * 6.
Prove that the value of the 2nd term in the sequence is 12.
-/

theorem sequence_value_2 (a : ℕ → ℕ) 
  (h1 : a 1 = 6) 
  (h3 : a 3 = 18) 
  (h4 : a 4 = 24) 
  (h5 : a 5 = 30) 
  (h_pattern : ∀ n, n ≠ 6 → a n = n * 6) :
  a 2 = 12 :=
by
  sorry

end sequence_value_2_l37_37137


namespace prod_ge_27_eq_iff_equality_l37_37797

variable (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
          (h4 : a + b + c + 2 = a * b * c)

theorem prod_ge_27 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a + b + c + 2 = a * b * c) : (a + 1) * (b + 1) * (c + 1) ≥ 27 :=
by sorry

theorem eq_iff_equality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a + b + c + 2 = a * b * c) : 
  ((a + 1) * (b + 1) * (c + 1) = 27) ↔ (a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end prod_ge_27_eq_iff_equality_l37_37797


namespace perimeter_of_figure_l37_37302

theorem perimeter_of_figure (a b c d : ℕ) (p : ℕ) (h1 : a = 6) (h2 : b = 3) (h3 : c = 2) (h4 : d = 4) (h5 : p = a * b + c * d) : p = 26 :=
by
  sorry

end perimeter_of_figure_l37_37302


namespace increasing_function_range_l37_37519

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 - a) * x + 1 else a^x

theorem increasing_function_range (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : ∀ x y : ℝ, x < y → f a x ≤ f a y) : 
  1.5 ≤ a ∧ a < 2 :=
sorry

end increasing_function_range_l37_37519


namespace intersect_sets_l37_37001

def A := {x : ℝ | x > -1}
def B := {x : ℝ | x ≤ 5}

theorem intersect_sets : (A ∩ B) = {x : ℝ | -1 < x ∧ x ≤ 5} := 
by 
  sorry

end intersect_sets_l37_37001


namespace domain_width_of_g_l37_37471

theorem domain_width_of_g (h : ℝ → ℝ) (domain_h : ∀ x, -8 ≤ x ∧ x ≤ 8 → h x = h x) :
  let g (x : ℝ) := h (x / 2)
  ∃ a b, (∀ x, a ≤ x ∧ x ≤ b → ∃ y, g x = y) ∧ (b - a = 32) := 
sorry

end domain_width_of_g_l37_37471


namespace correct_calculation_l37_37589

variable {a : ℝ}

theorem correct_calculation : a^2 * a^3 = a^5 :=
by sorry

end correct_calculation_l37_37589


namespace brian_total_commission_l37_37178

theorem brian_total_commission :
  let commission_rate := 0.02
  let house1 := 157000
  let house2 := 499000
  let house3 := 125000
  let total_sales := house1 + house2 + house3
  let total_commission := total_sales * commission_rate
  total_commission = 15620 := by
{
  sorry
}

end brian_total_commission_l37_37178


namespace subtraction_identity_l37_37356

theorem subtraction_identity : 3.57 - 1.14 - 0.23 = 2.20 := sorry

end subtraction_identity_l37_37356


namespace variance_male_greater_than_female_l37_37434

noncomputable def male_scores : List ℝ := [87, 95, 89, 93, 91]
noncomputable def female_scores : List ℝ := [89, 94, 94, 89, 94]

-- Function to calculate the variance of scores
noncomputable def variance (scores : List ℝ) : ℝ :=
  let n := scores.length
  let mean := scores.sum / n
  (scores.map (λ x => (x - mean) ^ 2)).sum / n

-- We assert the problem statement
theorem variance_male_greater_than_female :
  variance male_scores > variance female_scores :=
by
  sorry

end variance_male_greater_than_female_l37_37434


namespace Julie_initial_savings_l37_37875

theorem Julie_initial_savings (P r : ℝ) 
  (h1 : 100 = P * r * 2) 
  (h2 : 105 = P * (1 + r) ^ 2 - P) : 
  2 * P = 1000 :=
by
  sorry

end Julie_initial_savings_l37_37875


namespace part1_max_value_l37_37558

variable (f : ℝ → ℝ)
def is_maximum (y : ℝ) := ∀ x : ℝ, f x ≤ y

theorem part1_max_value (m : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = -x^2 + m*x + 1) :
  m = 0 → (exists y, is_maximum f y ∧ y = 1) := 
sorry

end part1_max_value_l37_37558


namespace solution_set_of_inequality_l37_37076

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≥ 0 → f x = 2^x - 4

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h1 : is_even_function f)
  (h2 : satisfies_condition f) :
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
sorry

end solution_set_of_inequality_l37_37076


namespace minimum_value_expression_l37_37528

theorem minimum_value_expression : ∃ x : ℝ, (3 * x^2 - 18 * x + 2023) = 1996 := sorry

end minimum_value_expression_l37_37528


namespace local_minimum_f_when_k2_l37_37332

noncomputable def f (k : ℕ) (x : ℝ) : ℝ := (Real.exp x - 1) * (x - 1) ^ k

theorem local_minimum_f_when_k2 : ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f 2 x ≥ f 2 1 :=
by
  -- the question asks to prove that the function attains a local minimum at x = 1 when k = 2
  sorry

end local_minimum_f_when_k2_l37_37332


namespace total_dogs_on_farm_l37_37624

-- Definitions based on conditions from part a)
def num_dog_houses : ℕ := 5
def num_dogs_per_house : ℕ := 4

-- Statement to prove
theorem total_dogs_on_farm : num_dog_houses * num_dogs_per_house = 20 :=
by
  sorry

end total_dogs_on_farm_l37_37624


namespace Suresh_meeting_time_l37_37363

theorem Suresh_meeting_time :
  let C := 726
  let v1 := 75
  let v2 := 62.5
  C / (v1 + v2) = 5.28 := by
  sorry

end Suresh_meeting_time_l37_37363


namespace division_value_l37_37418

theorem division_value (a b c : ℝ) 
  (h1 : a / b = 5 / 3) 
  (h2 : b / c = 7 / 2) : 
  c / a = 6 / 35 := 
by
  sorry

end division_value_l37_37418


namespace ion_electronic_structure_l37_37753

theorem ion_electronic_structure (R M Z n m X : ℤ) (h1 : R + X = M - n) (h2 : M - n = Z - m) (h3 : n > m) : M > Z ∧ Z > R := 
by 
  sorry

end ion_electronic_structure_l37_37753


namespace polar_to_rect_l37_37051

open Real 

theorem polar_to_rect (r : ℝ) (θ : ℝ) (h_r : r = 3) (h_θ : θ = 3 * π / 4) : 
  (r * cos θ, r * sin θ) = (-3 / Real.sqrt 2, 3 / Real.sqrt 2) :=
by
  -- Optional step: you can introduce the variables as they have already been proved using the given conditions
  have hr : r = 3 := h_r
  have hθ : θ = 3 * π / 4 := h_θ
  -- Goal changes according to the values of r and θ derived from the conditions
  sorry

end polar_to_rect_l37_37051


namespace min_time_adult_worms_l37_37465

noncomputable def f : ℕ → ℝ
| 1 => 0
| n => (1 - 1 / (2 ^ (n - 1)))

theorem min_time_adult_worms (n : ℕ) (h : n ≥ 1) : 
  ∃ min_time : ℝ, 
  (min_time = 1 - 1 / (2 ^ (n - 1))) ∧ 
  (∀ t : ℝ, (t = 1 - 1 / (2 ^ (n - 1)))) := 
sorry

end min_time_adult_worms_l37_37465


namespace min_capacity_for_raft_l37_37314

-- Define the weights of the animals
def weight_mouse : ℕ := 70
def weight_mole : ℕ := 90
def weight_hamster : ℕ := 120

-- Define the number of each type of animal
def number_mice : ℕ := 5
def number_moles : ℕ := 3
def number_hamsters : ℕ := 4

-- Define the minimum weight capacity for the raft
def min_weight_capacity : ℕ := 140

-- Prove that the minimum weight capacity the raft must have to transport all animals is 140 grams.
theorem min_capacity_for_raft :
  (weight_mouse * 2 ≤ min_weight_capacity) ∧ 
  (∀ trip_weight, trip_weight ≥ min_weight_capacity → 
    (trip_weight = weight_mouse * 2 ∨ trip_weight = weight_mole * 2 ∨ trip_weight = weight_hamster * 2)) :=
by 
  sorry

end min_capacity_for_raft_l37_37314


namespace repeating_decimal_sum_l37_37883

theorem repeating_decimal_sum :
  (0.3333333333 : ℚ) + (0.0404040404 : ℚ) + (0.005005005 : ℚ) + (0.000600060006 : ℚ) = 3793 / 9999 := by
sorry

end repeating_decimal_sum_l37_37883


namespace percentage_girls_l37_37724

theorem percentage_girls (initial_boys : ℕ) (initial_girls : ℕ) (added_boys : ℕ) :
  initial_boys = 11 → initial_girls = 13 → added_boys = 1 → 
  100 * initial_girls / (initial_boys + added_boys + initial_girls) = 52 :=
by
  intros h_boys h_girls h_added
  sorry

end percentage_girls_l37_37724


namespace five_peso_coins_count_l37_37581

theorem five_peso_coins_count (x y : ℕ) (h1 : x + y = 56) (h2 : 10 * x + 5 * y = 440) (h3 : x = 24 ∨ y = 24) : y = 24 :=
by sorry

end five_peso_coins_count_l37_37581


namespace lines_intersect_at_l37_37396

noncomputable def line1 (x : ℚ) : ℚ := (-2 / 3) * x + 2
noncomputable def line2 (x : ℚ) : ℚ := -2 * x + (3 / 2)

theorem lines_intersect_at :
  ∃ (x y : ℚ), line1 x = y ∧ line2 x = y ∧ x = (3 / 8) ∧ y = (7 / 4) :=
sorry

end lines_intersect_at_l37_37396


namespace birds_find_more_than_half_millet_on_thursday_l37_37536

def millet_on_day (n : ℕ) : ℝ :=
  2 - 2 * (0.7 ^ n)

def more_than_half_millet (day : ℕ) : Prop :=
  millet_on_day day > 1

theorem birds_find_more_than_half_millet_on_thursday : more_than_half_millet 4 :=
by
  sorry

end birds_find_more_than_half_millet_on_thursday_l37_37536


namespace sqrt_div_equality_l37_37258

noncomputable def sqrt_div (x y : ℝ) : ℝ := Real.sqrt x / Real.sqrt y

theorem sqrt_div_equality (x y : ℝ)
  (h : ( ( (1/3 : ℝ) ^ 2 + (1/4 : ℝ) ^ 2 ) / ( (1/5 : ℝ) ^ 2 + (1/6 : ℝ) ^ 2 ) = 25 * x / (73 * y) )) :
  sqrt_div x y = 5 / 2 :=
sorry

end sqrt_div_equality_l37_37258


namespace sandwiches_ordered_l37_37228

-- Define the cost per sandwich
def cost_per_sandwich : ℝ := 5

-- Define the delivery fee
def delivery_fee : ℝ := 20

-- Define the tip percentage
def tip_percentage : ℝ := 0.10

-- Define the total amount received
def total_received : ℝ := 121

-- Define the equation representing the total amount received
def total_equation (x : ℝ) : Prop :=
  cost_per_sandwich * x + delivery_fee + (cost_per_sandwich * x + delivery_fee) * tip_percentage = total_received

-- Define the theorem that needs to be proved
theorem sandwiches_ordered (x : ℝ) : total_equation x ↔ x = 18 :=
sorry

end sandwiches_ordered_l37_37228


namespace houses_distance_l37_37585

theorem houses_distance (num_houses : ℕ) (total_length : ℝ) (at_both_ends : Bool) 
  (h1: num_houses = 6) (h2: total_length = 11.5) (h3: at_both_ends = true) : 
  total_length / (num_houses - 1) = 2.3 := 
by
  sorry

end houses_distance_l37_37585


namespace value_of_f_neg1_l37_37014

def f (x : ℝ) : ℝ := x^3 + x^2 - 2 * x

theorem value_of_f_neg1 : f (-1) = 2 := by
  sorry

end value_of_f_neg1_l37_37014


namespace square_area_from_diagonal_l37_37973

theorem square_area_from_diagonal
  (d : ℝ) (h : d = 10) : ∃ (A : ℝ), A = 50 :=
by {
  -- here goes the proof
  sorry
}

end square_area_from_diagonal_l37_37973


namespace polynomial_expansion_sum_is_21_l37_37521

theorem polynomial_expansion_sum_is_21 :
  ∃ (A B C D : ℤ), (∀ (x : ℤ), (x + 2) * (3 * x^2 - x + 5) = A * x^3 + B * x^2 + C * x + D) ∧
  A + B + C + D = 21 :=
by
  sorry

end polynomial_expansion_sum_is_21_l37_37521


namespace jar_weight_percentage_l37_37671

theorem jar_weight_percentage (J B : ℝ) (h : 0.60 * (J + B) = J + 1 / 3 * B) :
  (J / (J + B)) = 0.403 :=
by
  sorry

end jar_weight_percentage_l37_37671


namespace polygon_sides_eq_eight_l37_37386

theorem polygon_sides_eq_eight (n : ℕ) 
  (h_diff : (n - 2) * 180 - 360 = 720) :
  n = 8 := 
by 
  sorry

end polygon_sides_eq_eight_l37_37386


namespace consecutive_integers_sum_l37_37067

theorem consecutive_integers_sum (a b : ℤ) (h1 : a < b) (h2 : b = a + 1) (h3 : a < Real.sqrt 17) (h4 : Real.sqrt 17 < b) : a + b = 9 :=
sorry

end consecutive_integers_sum_l37_37067


namespace increase_in_y_coordinate_l37_37860

theorem increase_in_y_coordinate (m n : ℝ) (h₁ : m = (n / 5) - 2 / 5) : 
  (5 * (m + 3) + 2) - (5 * m + 2) = 15 :=
by
  sorry

end increase_in_y_coordinate_l37_37860


namespace sum_of_possible_values_l37_37374

theorem sum_of_possible_values (A B : ℕ) 
  (hA1 : A < 10) (hA2 : 0 < A) (hB1 : B < 10) (hB2 : 0 < B)
  (h1 : 3 / 12 < A / 12) (h2 : A / 12 < 7 / 12)
  (h3 : 1 / 10 < 1 / B) (h4 : 1 / B < 1 / 3) :
  3 + 6 = 9 :=
by
  sorry

end sum_of_possible_values_l37_37374


namespace B_and_D_know_their_grades_l37_37186

-- Define the students and their respective grades
inductive Grade : Type
| excellent : Grade
| good : Grade

-- Define the students
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student

-- Define the information given in the problem regarding which student sees whose grade
def sees (s1 s2 : Student) : Prop :=
  (s1 = Student.A ∧ (s2 = Student.B ∨ s2 = Student.C)) ∨
  (s1 = Student.B ∧ s2 = Student.C) ∨
  (s1 = Student.D ∧ s2 = Student.A)

-- Define the condition that there are 2 excellent and 2 good grades
def grade_distribution (gA gB gC gD : Grade) : Prop :=
  gA ≠ gB → (gC = gA ∨ gC = gB) ∧ (gD = gA ∨ gD = gB) ∧
  (gA = Grade.excellent ∧ (gB = Grade.good ∨ gC = Grade.good ∨ gD = Grade.good)) ∧
  (gA = Grade.good ∧ (gB = Grade.excellent ∨ gC = Grade.excellent ∨ gD = Grade.excellent))

-- Student A's statement after seeing B and C's grades
def A_statement (gA gB gC : Grade) : Prop :=
  (gB = gA ∨ gC = gA) ∨ (gB ≠ gA ∧ gC ≠ gA)

-- Formal proof goal: Prove that B and D can know their own grades based on the information provided
theorem B_and_D_know_their_grades (gA gB gC gD : Grade)
  (h1 : grade_distribution gA gB gC gD)
  (h2 : A_statement gA gB gC)
  (h3 : sees Student.A Student.B)
  (h4 : sees Student.A Student.C)
  (h5 : sees Student.B Student.C)
  (h6 : sees Student.D Student.A) :
  (gB ≠ Grade.excellent ∨ gB ≠ Grade.good) ∧ (gD ≠ Grade.excellent ∨ gD ≠ Grade.good) :=
by sorry

end B_and_D_know_their_grades_l37_37186


namespace area_of_region_B_l37_37990

-- Given conditions
def region_B (z : ℂ) : Prop :=
  (0 ≤ (z.re / 50) ∧ (z.re / 50) ≤ 1 ∧ 0 ≤ (z.im / 50) ∧ (z.im / 50) ≤ 1)
  ∧
  (0 ≤ (50 * z.re / (z.re^2 + z.im^2)) ∧ (50 * z.re / (z.re^2 + z.im^2)) ≤ 1 ∧ 
  0 ≤ (50 * z.im / (z.re^2 + z.im^2)) ∧ (50 * z.im / (z.re^2 + z.im^2)) ≤ 1)

-- Theorem to be proved
theorem area_of_region_B : 
  (∫ z in {z : ℂ | region_B z}, 1) = 1875 - 312.5 * Real.pi :=
by
  sorry

end area_of_region_B_l37_37990


namespace ticket_sales_revenue_l37_37175

theorem ticket_sales_revenue :
  let student_ticket_price := 4
  let general_admission_ticket_price := 6
  let total_tickets_sold := 525
  let general_admission_tickets_sold := 388
  let student_tickets_sold := total_tickets_sold - general_admission_tickets_sold
  let money_from_student_tickets := student_tickets_sold * student_ticket_price
  let money_from_general_admission_tickets := general_admission_tickets_sold * general_admission_ticket_price
  let total_money_collected := money_from_student_tickets + money_from_general_admission_tickets
  total_money_collected = 2876 :=
by
  sorry

end ticket_sales_revenue_l37_37175


namespace simplify_expression_l37_37672

variable (a : ℝ)

theorem simplify_expression : 
  (a^2 / (a^(1/2) * a^(2/3))) = a^(5/6) :=
by
  sorry

end simplify_expression_l37_37672


namespace suitable_land_acres_l37_37364

theorem suitable_land_acres (new_multiplier : ℝ) (previous_acres : ℝ) (pond_acres : ℝ) :
  new_multiplier = 10 ∧ previous_acres = 2 ∧ pond_acres = 1 → 
  (new_multiplier * previous_acres - pond_acres) = 19 :=
by
  intro h
  sorry

end suitable_land_acres_l37_37364


namespace find_number_added_l37_37861

theorem find_number_added (x n : ℕ) (h : (x + x + 2 + x + 4 + x + n + x + 22) / 5 = x + 7) : n = 7 :=
by
  sorry

end find_number_added_l37_37861


namespace total_payment_mr_benson_made_l37_37134

noncomputable def general_admission_ticket_cost : ℝ := 40
noncomputable def num_general_admission_tickets : ℕ := 10
noncomputable def num_vip_tickets : ℕ := 3
noncomputable def num_premium_tickets : ℕ := 2
noncomputable def vip_ticket_rate_increase : ℝ := 0.20
noncomputable def premium_ticket_rate_increase : ℝ := 0.50
noncomputable def discount_rate : ℝ := 0.05
noncomputable def threshold_tickets : ℕ := 10

noncomputable def vip_ticket_cost : ℝ := general_admission_ticket_cost * (1 + vip_ticket_rate_increase)
noncomputable def premium_ticket_cost : ℝ := general_admission_ticket_cost * (1 + premium_ticket_rate_increase)

noncomputable def total_general_admission_cost : ℝ := num_general_admission_tickets * general_admission_ticket_cost
noncomputable def total_vip_cost : ℝ := num_vip_tickets * vip_ticket_cost
noncomputable def total_premium_cost : ℝ := num_premium_tickets * premium_ticket_cost

noncomputable def total_tickets : ℕ := num_general_admission_tickets + num_vip_tickets + num_premium_tickets
noncomputable def tickets_exceeding_threshold : ℕ := if total_tickets > threshold_tickets then total_tickets - threshold_tickets else 0

noncomputable def discounted_vip_cost : ℝ := vip_ticket_cost * (1 - discount_rate)
noncomputable def discounted_premium_cost : ℝ := premium_ticket_cost * (1 - discount_rate)

noncomputable def total_discounted_vip_cost : ℝ :=  num_vip_tickets * discounted_vip_cost
noncomputable def total_discounted_premium_cost : ℝ := num_premium_tickets * discounted_premium_cost

noncomputable def total_cost_with_discounts : ℝ := total_general_admission_cost + total_discounted_vip_cost + total_discounted_premium_cost

theorem total_payment_mr_benson_made : total_cost_with_discounts = 650.80 :=
by
  -- Proof is omitted
  sorry

end total_payment_mr_benson_made_l37_37134


namespace box_weights_l37_37952

theorem box_weights (a b c : ℕ) (h1 : a + b = 132) (h2 : b + c = 135) (h3 : c + a = 137) (h4 : a > 40) (h5 : b > 40) (h6 : c > 40) : a + b + c = 202 :=
by 
  sorry

end box_weights_l37_37952


namespace geometric_sum_first_8_terms_eq_17_l37_37729

theorem geometric_sum_first_8_terms_eq_17 (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = 2 * a n)
  (h2 : a 0 + a 1 + a 2 + a 3 = 1) : 
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 17 :=
sorry

end geometric_sum_first_8_terms_eq_17_l37_37729


namespace root_properties_of_cubic_l37_37941

theorem root_properties_of_cubic (z1 z2 : ℂ) (h1 : z1^2 + z1 + 1 = 0) (h2 : z2^2 + z2 + 1 = 0) :
  z1 * z2 = 1 ∧ z1^3 = 1 ∧ z2^3 = 1 :=
by
  -- Proof omitted
  sorry

end root_properties_of_cubic_l37_37941


namespace mary_initial_amount_l37_37138

theorem mary_initial_amount (current_amount pie_cost mary_after_pie : ℕ) 
  (h1 : pie_cost = 6) 
  (h2 : mary_after_pie = 52) :
  current_amount = pie_cost + mary_after_pie → 
  current_amount = 58 :=
by
  intro h
  rw [h1, h2] at h
  exact h

end mary_initial_amount_l37_37138


namespace two_digit_number_l37_37088

theorem two_digit_number (x y : ℕ) (hx : 0 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
  (h1 : x^2 + y^2 = 10*x + y + 11) (h2 : 2*x*y = 10*x + y - 5) :
  10*x + y = 95 ∨ 10*x + y = 15 := 
sorry

end two_digit_number_l37_37088


namespace inequality_proof_l37_37653

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (2 * x^2 / (y + z) + 2 * y^2 / (z + x) + 2 * z^2 / (x + y) ≥ x + y + z) :=
by
  sorry

end inequality_proof_l37_37653


namespace fishing_tomorrow_l37_37295

theorem fishing_tomorrow (yesterday_fishers today_fishers : ℕ)
  (every_day_fishers every_other_day_fishers every_three_days_fishers : ℕ)
  (total_population : ℕ):
  yesterday_fishers = 12 → 
  today_fishers = 10 → 
  every_day_fishers = 7 → 
  every_other_day_fishers = 8 → 
  every_three_days_fishers = 3 → 
  total_population = yesterday_fishers + today_fishers + (total_population - (every_day_fishers + every_other_day_fishers + every_three_days_fishers)) →
  ∃ tomorrow_fishers : ℕ, tomorrow_fishers = 15 :=
by {
  -- This is a statement definition, the proof is not required and thus marked as "sorry:"
  sorry
}

end fishing_tomorrow_l37_37295


namespace contrapositive_example_l37_37400

theorem contrapositive_example (x : ℝ) : (x = 1 → x^2 - 3 * x + 2 = 0) ↔ (x^2 - 3 * x + 2 ≠ 0 → x ≠ 1) :=
by
  sorry

end contrapositive_example_l37_37400


namespace ratio_of_girls_to_boys_l37_37501

variable (g b : ℕ)

theorem ratio_of_girls_to_boys (h₁ : g + b = 36)
                               (h₂ : g = b + 6) : g / b = 7 / 5 :=
by sorry

end ratio_of_girls_to_boys_l37_37501


namespace period_of_f_cos_theta_l37_37626

open Real

noncomputable def alpha (x : ℝ) : ℝ × ℝ :=
  (sqrt 3 * sin (2 * x), cos x + sin x)

noncomputable def beta (x : ℝ) : ℝ × ℝ :=
  (1, cos x - sin x)

noncomputable def f (x : ℝ) : ℝ :=
  let (α1, α2) := alpha x
  let (β1, β2) := beta x
  α1 * β1 + α2 * β2

theorem period_of_f :
  (∀ x : ℝ, f (x + π) = f x) ∧ (∀ T : ℝ, (T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) → T = π) :=
sorry

theorem cos_theta :
  ∀ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ f θ = 1 → cos (θ - π / 6) = sqrt 3 / 2 :=
sorry

end period_of_f_cos_theta_l37_37626


namespace money_left_after_shopping_l37_37587

def initial_money : ℕ := 158
def shoe_cost : ℕ := 45
def bag_cost := shoe_cost - 17
def lunch_cost := bag_cost / 4
def total_expenses := shoe_cost + bag_cost + lunch_cost
def remaining_money := initial_money - total_expenses

theorem money_left_after_shopping : remaining_money = 78 := by
  sorry

end money_left_after_shopping_l37_37587


namespace susan_age_in_5_years_l37_37266

variable (J N S X : ℕ)

-- Conditions
axiom h1 : J - 8 = 2 * (N - 8)
axiom h2 : J + X = 37
axiom h3 : S = N - 3

-- Theorem statement
theorem susan_age_in_5_years : S + 5 = N + 2 :=
by sorry

end susan_age_in_5_years_l37_37266


namespace num_of_integers_abs_leq_six_l37_37102

theorem num_of_integers_abs_leq_six (x : ℤ) : 
  (|x - 3| ≤ 6) → ∃ (n : ℕ), n = 13 := 
by 
  sorry

end num_of_integers_abs_leq_six_l37_37102


namespace number_sum_20_eq_30_l37_37588

theorem number_sum_20_eq_30 : ∃ x : ℤ, 20 + x = 30 → x = 10 :=
by {
  sorry
}

end number_sum_20_eq_30_l37_37588


namespace not_sufficient_nor_necessary_l37_37095

theorem not_sufficient_nor_necessary (a b : ℝ) (hb : b ≠ 0) :
  ¬ ((a > b) ↔ (1 / a < 1 / b)) :=
by
  sorry

end not_sufficient_nor_necessary_l37_37095


namespace factor_expression_l37_37708

-- Define variables s and m
variables (s m : ℤ)

-- State the theorem to be proven: If s = 5, then m^2 - sm - 24 can be factored as (m - 8)(m + 3)
theorem factor_expression (hs : s = 5) : m^2 - s * m - 24 = (m - 8) * (m + 3) :=
by {
  sorry
}

end factor_expression_l37_37708


namespace value_of_x_l37_37096

theorem value_of_x (x : ℝ) (h : 0.5 * x - (1 / 3) * x = 110) : x = 660 :=
sorry

end value_of_x_l37_37096


namespace diameter_of_lake_l37_37452

-- Given conditions: the radius of the circular lake
def radius : ℝ := 7

-- The proof problem: proving the diameter of the lake is 14 meters
theorem diameter_of_lake : 2 * radius = 14 :=
by
  sorry

end diameter_of_lake_l37_37452


namespace rate_of_grapes_l37_37975

theorem rate_of_grapes (G : ℝ) 
  (h_grapes : 8 * G + 9 * 60 = 1100) : 
  G = 70 := 
by
  sorry

end rate_of_grapes_l37_37975


namespace find_a_and_b_l37_37554

theorem find_a_and_b (a b : ℝ) (h1 : b ≠ 0) 
  (h2 : (ab = a + b ∨ ab = a - b ∨ ab = a / b) 
  ∧ (a + b = a - b ∨ a + b = a / b) 
  ∧ (a - b = a / b)) : 
  (a = 1 / 2 ∨ a = -1 / 2) ∧ b = -1 := by
  sorry

end find_a_and_b_l37_37554


namespace convert_base10_to_base7_l37_37500

-- Definitions for powers and conditions
def n1 : ℕ := 7
def n2 : ℕ := n1 * n1
def n3 : ℕ := n2 * n1
def n4 : ℕ := n3 * n1

theorem convert_base10_to_base7 (n : ℕ) (h₁ : n = 395) : 
  ∃ a b c d : ℕ, 
    a * n3 + b * n2 + c * n1 + d = 395 ∧
    a < 7 ∧ b < 7 ∧ c < 7 ∧ d < 7 ∧
    a = 1 ∧ b = 1 ∧ c = 0 ∧ d = 3 :=
by { sorry }

end convert_base10_to_base7_l37_37500


namespace parametric_curve_C_line_tangent_to_curve_C_l37_37811

open Real

-- Definitions of the curve C and line l
def curve_C (ρ θ : ℝ) : Prop := ρ^2 - 4 * ρ * cos θ + 1 = 0

def line_l (t α x y : ℝ) : Prop := x = 4 + t * sin α ∧ y = t * cos α ∧ 0 ≤ α ∧ α < π

-- Parametric equation of curve C
theorem parametric_curve_C :
  ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * π →
  ∃ x y : ℝ, (x = 2 + sqrt 3 * cos θ ∧ y = sqrt 3 * sin θ ∧
              curve_C (sqrt (x^2 + y^2)) θ) :=
sorry

-- Tangency condition for line l and curve C
theorem line_tangent_to_curve_C :
  ∀ α : ℝ, 0 ≤ α ∧ α < π →
  (∃ t : ℝ, ∃ x y : ℝ, (line_l t α x y ∧ (x - 2)^2 + y^2 = 3 ∧
                        ((abs (2 * cos α - 4 * cos α) / sqrt (cos α ^ 2 + sin α ^ 2)) = sqrt 3)) →
                       (α = π / 6 ∧ x = 7 / 2 ∧ y = - sqrt 3 / 2)) :=
sorry

end parametric_curve_C_line_tangent_to_curve_C_l37_37811


namespace one_sixths_in_fraction_l37_37215

theorem one_sixths_in_fraction :
  (11 / 3) / (1 / 6) = 22 :=
sorry

end one_sixths_in_fraction_l37_37215


namespace negation_of_proposition_l37_37397

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 := 
sorry

end negation_of_proposition_l37_37397


namespace distance_to_river_l37_37058

theorem distance_to_river (d : ℝ) (h1 : ¬ (d ≥ 8)) (h2 : ¬ (d ≤ 7)) (h3 : ¬ (d ≤ 6)) : 7 < d ∧ d < 8 :=
by
  sorry

end distance_to_river_l37_37058


namespace value_of_f_2011_l37_37160

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_f_2011 (h_even : ∀ x : ℝ, f x = f (-x))
                       (h_sym : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 0 → f (2 + x) = f (2 - x))
                       (h_def : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 0 → f x = 2^x) : 
  f 2011 = 1 / 2 := 
sorry

end value_of_f_2011_l37_37160


namespace sum_d_e_f_equals_23_l37_37623

theorem sum_d_e_f_equals_23
  (d e f : ℤ)
  (h1 : ∀ x : ℝ, x^2 + 9 * x + 20 = (x + d) * (x + e))
  (h2 : ∀ x : ℝ, x^2 + 11 * x - 60 = (x + e) * (x - f)) :
  d + e + f = 23 :=
by
  sorry

end sum_d_e_f_equals_23_l37_37623


namespace satellite_orbit_time_approx_l37_37967

noncomputable def earth_radius_km : ℝ := 6371
noncomputable def satellite_speed_kmph : ℝ := 7000

theorem satellite_orbit_time_approx :
  let circumference := 2 * Real.pi * earth_radius_km 
  let time := circumference / satellite_speed_kmph 
  5.6 < time ∧ time < 5.8 :=
by
  sorry

end satellite_orbit_time_approx_l37_37967


namespace commutative_l37_37392

variable (R : Type) [NonAssocRing R]
variable (star : R → R → R)

axiom assoc : ∀ x y z : R, star (star x y) z = star x (star y z)
axiom comm_left : ∀ x y z : R, star (star x y) z = star (star y z) x
axiom distinct : ∀ {x y : R}, x ≠ y → ∃ z : R, star z x ≠ star z y

theorem commutative (x y : R) : star x y = star y x := sorry

end commutative_l37_37392


namespace sequence_general_term_l37_37474

theorem sequence_general_term (n : ℕ) (hn : 0 < n) : 
  ∃ (a_n : ℕ), a_n = 2 * Int.floor (Real.sqrt (n - 1)) + 1 :=
by
  sorry

end sequence_general_term_l37_37474


namespace stock_value_sale_l37_37781

theorem stock_value_sale
  (X : ℝ)
  (h1 : 0.20 * X * 0.10 - 0.80 * X * 0.05 = -350) :
  X = 17500 := by
  -- Proof goes here
  sorry

end stock_value_sale_l37_37781


namespace S_10_value_l37_37817

noncomputable def S (n : ℕ) (a : ℕ → ℕ) : ℕ := n * (a 1 + a n) / 2

theorem S_10_value (a : ℕ → ℕ) (h1 : a 2 = 3) (h2 : a 9 = 17) (h_arith : ∀ n, a (n + 1) = a n + (a 2 - a 1)) : 
  S 10 a = 100 := 
by
  sorry

end S_10_value_l37_37817


namespace coplanar_points_scalar_eq_l37_37034

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D O : V) (k : ℝ)

theorem coplanar_points_scalar_eq:
  (3 • (A - O) - 2 • (B - O) + 5 • (C - O) + k • (D - O) = (0 : V)) →
  k = -6 :=
by sorry

end coplanar_points_scalar_eq_l37_37034


namespace quadrilateral_area_l37_37725

/-
Proof Statement: For a square with a side length of 8 cm, each of whose sides is divided by a point into two equal segments, 
prove that the area of the quadrilateral formed by connecting these points is 32 cm².
-/

theorem quadrilateral_area (side_len : ℝ) (h : side_len = 8) :
  let quadrilateral_area := (side_len * side_len) / 2
  quadrilateral_area = 32 :=
by
  sorry

end quadrilateral_area_l37_37725


namespace find_circle_radius_l37_37201

-- Definitions based on the given conditions
def circle_eq (x y : ℝ) : Prop := (x^2 - 8*x + y^2 - 10*y + 34 = 0)

-- Problem statement
theorem find_circle_radius (x y : ℝ) : circle_eq x y → ∃ r : ℝ, r = Real.sqrt 7 :=
by
  sorry

end find_circle_radius_l37_37201


namespace john_buys_packs_l37_37462

theorem john_buys_packs :
  let classes := 6
  let students_per_class := 30
  let packs_per_student := 2
  let total_students := classes * students_per_class
  let total_packs := total_students * packs_per_student
  total_packs = 360 :=
by
  let classes := 6
  let students_per_class := 30
  let packs_per_student := 2
  let total_students := classes * students_per_class
  let total_packs := total_students * packs_per_student
  show total_packs = 360
  sorry

end john_buys_packs_l37_37462


namespace man_swim_distance_downstream_l37_37961

noncomputable def DistanceDownstream (Vm : ℝ) (Vupstream : ℝ) (time : ℝ) : ℝ :=
  let Vs := Vm - Vupstream
  let Vdownstream := Vm + Vs
  Vdownstream * time

theorem man_swim_distance_downstream :
  let Vm : ℝ := 3  -- speed of man in still water in km/h
  let time : ℝ := 6 -- time taken in hours
  let d_upstream : ℝ := 12 -- distance swum upstream in km
  let Vupstream : ℝ := d_upstream / time
  DistanceDownstream Vm Vupstream time = 24 := sorry

end man_swim_distance_downstream_l37_37961


namespace no_real_solution_range_of_a_l37_37313

theorem no_real_solution_range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬(|x + 1| + |x - 2| < a)) → a ≤ 3 :=
by
  sorry  -- Proof skipped

end no_real_solution_range_of_a_l37_37313


namespace daily_profit_at_45_selling_price_for_1200_profit_l37_37564

-- Definitions for the conditions
def cost_price (p: ℝ) : Prop := p = 30
def initial_sales (p: ℝ) (s: ℝ) : Prop := p = 40 ∧ s = 80
def sales_decrease_rate (r: ℝ) : Prop := r = 2
def max_selling_price (p: ℝ) : Prop := p ≤ 55

-- Proof for Question 1
theorem daily_profit_at_45 (cost price profit : ℝ) (sales : ℝ) (rate : ℝ) 
  (h_cost : cost_price cost)
  (h_initial_sales : initial_sales price sales) 
  (h_sales_decrease : sales_decrease_rate rate) :
  (price = 45) → profit = 1050 :=
by sorry

-- Proof for Question 2
theorem selling_price_for_1200_profit (cost price profit : ℝ) (sales : ℝ) (rate : ℝ) 
  (h_cost : cost_price cost)
  (h_initial_sales : initial_sales price sales) 
  (h_sales_decrease : sales_decrease_rate rate)
  (h_max_price : ∀ p, max_selling_price p → p ≤ 55) :
  profit = 1200 → price = 50 :=
by sorry

end daily_profit_at_45_selling_price_for_1200_profit_l37_37564


namespace eccentricity_of_ellipse_l37_37603

noncomputable def ellipse (a b c : ℝ) :=
  (a > b) ∧ (b > 0) ∧ (a^2 = b^2 + c^2) ∧ (b = 2 * c)

theorem eccentricity_of_ellipse (a b c : ℝ) (h : ellipse a b c) :
  (c / a = Real.sqrt 5 / 5) :=
by
  sorry

end eccentricity_of_ellipse_l37_37603


namespace area_inside_C_outside_A_B_l37_37348

-- Define the given circles with corresponding radii and positions
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the circles A, B, and C with the specific properties given
def CircleA : Circle := { center := (0, 0), radius := 1 }
def CircleB : Circle := { center := (2, 0), radius := 1 }
def CircleC : Circle := { center := (1, 2), radius := 2 }

-- Given that Circle C is tangent to the midpoint M of the line segment AB
-- Prove the area inside Circle C but outside Circle A and B
theorem area_inside_C_outside_A_B : 
  let area_inside_C := π * CircleC.radius ^ 2
  let overlap_area := (π - 2)
  area_inside_C - overlap_area = 3 * π + 2 := by
  sorry

end area_inside_C_outside_A_B_l37_37348


namespace paco_min_cookies_l37_37539

theorem paco_min_cookies (x : ℕ) (h_initial : 25 - x ≥ 0) : 
  x + (3 + 2) ≥ 5 := by
  sorry

end paco_min_cookies_l37_37539


namespace inclination_angle_l37_37207

theorem inclination_angle (α : ℝ) (t : ℝ) (h : 0 < α ∧ α < π / 2) :
  let x := 1 + t * Real.cos (α + 3 * π / 2)
  let y := 2 + t * Real.sin (α + 3 * π / 2)
  ∃ θ, θ = α + π / 2 := by
  sorry

end inclination_angle_l37_37207


namespace julie_hours_per_week_school_year_l37_37405

-- Defining the assumptions
variable (summer_hours_per_week : ℕ) (summer_weeks : ℕ) (summer_earnings : ℝ)
variable (school_year_weeks : ℕ) (school_year_earnings : ℝ)

-- Assuming the given values
def assumptions : Prop :=
  summer_hours_per_week = 36 ∧ 
  summer_weeks = 10 ∧ 
  summer_earnings = 4500 ∧ 
  school_year_weeks = 45 ∧ 
  school_year_earnings = 4500

-- Proving that Julie must work 8 hours per week during the school year to make another $4500
theorem julie_hours_per_week_school_year : 
  assumptions summer_hours_per_week summer_weeks summer_earnings school_year_weeks school_year_earnings →
  (school_year_earnings / (summer_earnings / (summer_hours_per_week * summer_weeks)) / school_year_weeks = 8) :=
by
  sorry

end julie_hours_per_week_school_year_l37_37405


namespace solve_for_y_l37_37275

theorem solve_for_y :
  ∀ (y : ℝ), (9 * y^2 + 49 * y^2 + 21/2 * y^2 = 1300) → y = 4.34 := 
by sorry

end solve_for_y_l37_37275


namespace coin_flip_probability_l37_37989

theorem coin_flip_probability (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1)
  (h_win : ∑' n, (1 - p) ^ n * p ^ (n + 1) = 1 / 2) :
  p = (3 - Real.sqrt 5) / 2 :=
by
  sorry

end coin_flip_probability_l37_37989


namespace trigonometric_identity_l37_37815

theorem trigonometric_identity (α : ℝ) (h : Real.sin (α + (Real.pi / 3)) = 3 / 5) :
  Real.cos ((Real.pi / 6) - α) = 3 / 5 :=
by
  sorry

end trigonometric_identity_l37_37815


namespace dylan_ice_cubes_l37_37480

-- Definitions based on conditions
def trays := 2
def spaces_per_tray := 12
def total_tray_ice := trays * spaces_per_tray
def pitcher_multiplier := 2

-- The statement to be proven
theorem dylan_ice_cubes (x : ℕ) : x + pitcher_multiplier * x = total_tray_ice → x = 8 :=
by {
  sorry
}

end dylan_ice_cubes_l37_37480


namespace num_even_3digit_nums_lt_700_l37_37764

theorem num_even_3digit_nums_lt_700 
  (digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}) 
  (even_digits : Finset ℕ := {2, 4, 6}) 
  (h1 : ∀ n ∈ digits, n < 10)
  (h2 : 0 ∉ digits) : 
  ∃ n, n = 126 ∧ ∀ d, d ∈ digits → 
  (d < 10) ∧ ∀ u, u ∈ even_digits → 
  (u < 10) 
:=
  sorry

end num_even_3digit_nums_lt_700_l37_37764


namespace initial_time_between_maintenance_checks_l37_37068

theorem initial_time_between_maintenance_checks (x : ℝ) (h1 : 1.20 * x = 30) : x = 25 := by
  sorry

end initial_time_between_maintenance_checks_l37_37068


namespace arithmetic_sequence_first_term_range_l37_37253

theorem arithmetic_sequence_first_term_range (a_1 : ℝ) (d : ℝ) (a_10 : ℝ) (a_11 : ℝ) :
  d = (Real.pi / 8) → 
  (a_1 + 9 * d ≤ 0) → 
  (a_1 + 10 * d ≥ 0) → 
  - (5 * Real.pi / 4) ≤ a_1 ∧ a_1 ≤ - (9 * Real.pi / 8) :=
by
  sorry

end arithmetic_sequence_first_term_range_l37_37253


namespace max_fridays_in_year_l37_37779

theorem max_fridays_in_year (days_in_common_year days_in_leap_year : ℕ) 
  (h_common_year : days_in_common_year = 365)
  (h_leap_year : days_in_leap_year = 366) : 
  ∃ (max_fridays : ℕ), max_fridays = 53 := 
by
  existsi 53
  sorry

end max_fridays_in_year_l37_37779


namespace problem_statement_l37_37494

theorem problem_statement (a b c : ℝ)
  (h : a * b * c = ( Real.sqrt ( (a + 2) * (b + 3) ) ) / (c + 1)) :
  6 * 15 * 7 = 1.5 :=
sorry

end problem_statement_l37_37494


namespace digit_D_eq_9_l37_37507

-- Define digits and the basic operations on 2-digit numbers
def is_digit (n : ℕ) : Prop := n < 10
def tens (n : ℕ) : ℕ := n / 10
def units (n : ℕ) : ℕ := n % 10
def two_digit (a b : ℕ) : ℕ := 10 * a + b

theorem digit_D_eq_9 (A B C D : ℕ):
  is_digit A → is_digit B → is_digit C → is_digit D →
  (two_digit A B) + (two_digit C B) = two_digit D A →
  (two_digit A B) - (two_digit C B) = A →
  D = 9 :=
by sorry

end digit_D_eq_9_l37_37507


namespace sum_of_m_and_n_l37_37003

theorem sum_of_m_and_n :
  ∃ m n : ℝ, (∀ x : ℝ, (x = 2 → m = 6 / x) ∧ (x = -2 → n = 6 / x)) ∧ (m + n = 0) :=
by
  let m := 6 / 2
  let n := 6 / (-2)
  use m, n
  simp
  sorry -- Proof omitted

end sum_of_m_and_n_l37_37003


namespace radish_patch_size_l37_37533

theorem radish_patch_size (R P : ℕ) (h1 : P = 2 * R) (h2 : P / 6 = 5) : R = 15 := by
  sorry

end radish_patch_size_l37_37533


namespace mandy_book_length_l37_37532

theorem mandy_book_length :
  let initial_length := 8
  let initial_age := 6
  let doubled_age := 2 * initial_age
  let length_at_doubled_age := 5 * initial_length
  let later_age := doubled_age + 8
  let length_at_later_age := 3 * length_at_doubled_age
  let final_length := 4 * length_at_later_age
  final_length = 480 :=
by
  sorry

end mandy_book_length_l37_37532


namespace length_of_bridge_l37_37139

theorem length_of_bridge (ship_length : ℝ) (ship_speed_kmh : ℝ) (time : ℝ) (bridge_length : ℝ) :
  ship_length = 450 → ship_speed_kmh = 24 → time = 202.48 → bridge_length = (6.67 * 202.48 - 450) → bridge_length = 900.54 :=
by
  intros h1 h2 h3 h4
  sorry

end length_of_bridge_l37_37139


namespace sides_of_regular_polygon_l37_37264

theorem sides_of_regular_polygon {n : ℕ} (h₁ : n ≥ 3)
  (h₂ : (n * (n - 3)) / 2 + 6 = 2 * n) : n = 4 :=
sorry

end sides_of_regular_polygon_l37_37264


namespace inverse_square_variation_l37_37179

theorem inverse_square_variation (k : ℝ) (y x : ℝ) (h1: x = k / y^2) (h2: 0.25 = k / 36) : 
  x = 1 :=
by
  -- Here, you would provide further Lean code to complete the proof
  -- using the given hypothesis h1 and h2, along with some computation.
  sorry

end inverse_square_variation_l37_37179


namespace approximation_hundred_thousandth_place_l37_37775

theorem approximation_hundred_thousandth_place (n : ℕ) (h : n = 537400000) : 
  ∃ p : ℕ, p = 100000 := 
sorry

end approximation_hundred_thousandth_place_l37_37775


namespace complex_number_properties_l37_37011

open Complex

noncomputable def z : ℂ := (1 - I) / I

theorem complex_number_properties :
  z ^ 2 = 2 * I ∧ Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_number_properties_l37_37011


namespace range_of_a_l37_37751

noncomputable def f (x : ℝ) : ℝ := 4 * x + 3 * Real.sin x

theorem range_of_a (a : ℝ) (h : f (1 - a) + f (1 - a^2) < 0) : 1 < a ∧ a < Real.sqrt 2 := sorry

end range_of_a_l37_37751


namespace amc_inequality_l37_37520

theorem amc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 := 
by 
  sorry

end amc_inequality_l37_37520


namespace solve_system_of_equations_l37_37761

theorem solve_system_of_equations (x y : ℝ) (h1 : x - y = -5) (h2 : 3 * x + 2 * y = 10) : x = 0 ∧ y = 5 := by
  sorry

end solve_system_of_equations_l37_37761


namespace gwen_received_more_money_from_mom_l37_37142

theorem gwen_received_more_money_from_mom :
  let mom_money := 8
  let dad_money := 5
  mom_money - dad_money = 3 :=
by
  sorry

end gwen_received_more_money_from_mom_l37_37142


namespace roots_reciprocal_l37_37660

theorem roots_reciprocal (a b c x1 x2 x3 x4 : ℝ) 
  (h1 : a ≠ 0)
  (h2 : c ≠ 0)
  (hx1 : a * x1^2 + b * x1 + c = 0)
  (hx2 : a * x2^2 + b * x2 + c = 0)
  (hx3 : c * x3^2 + b * x3 + a = 0)
  (hx4 : c * x4^2 + b * x4 + a = 0) :
  (x3 = 1/x1 ∧ x4 = 1/x2) :=
  sorry

end roots_reciprocal_l37_37660


namespace max_stamps_l37_37763

theorem max_stamps (n friends extra total: ℕ) (h1: friends = 15) (h2: extra = 5) (h3: total < 150) : total ≤ 140 :=
by
  sorry

end max_stamps_l37_37763


namespace sum_fraction_equals_two_l37_37212

theorem sum_fraction_equals_two
  (a b c d : ℝ) (h₁ : a ≠ -1) (h₂ : b ≠ -1) (h₃ : c ≠ -1) (h₄ : d ≠ -1)
  (ω : ℂ) (h₅ : ω^4 = 1) (h₆ : ω ≠ 1)
  (h₇ : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = (4 / (ω^2))) 
  (h₈ : a + b + c + d = a * b * c * d)
  (h₉ : a * b + a * c + a * d + b * c + b * d + c * d = a * b * c + a * b * d + a * c * d + b * c * d) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 2 := 
sorry

end sum_fraction_equals_two_l37_37212


namespace clothing_value_is_correct_l37_37041

-- Define the value of the clothing to be C and the correct answer
def value_of_clothing (C : ℝ) : Prop :=
  (C + 2) = (7 / 12) * (C + 10)

-- Statement of the problem
theorem clothing_value_is_correct :
  ∃ (C : ℝ), value_of_clothing C ∧ C = 46 / 5 :=
by {
  sorry
}

end clothing_value_is_correct_l37_37041


namespace sin_C_value_proof_a2_b2_fraction_proof_sides_sum_comparison_l37_37992

variables (A B C a b c S : ℝ)
variables (h_area : S = (a + b) ^ 2 - c ^ 2) (h_sum : a + b = 4)
variables (h_triangle : ∀ (x : ℝ), x = sin C)

open Real

theorem sin_C_value_proof :
  sin C = 8 / 17 :=
sorry

theorem a2_b2_fraction_proof :
  (a ^ 2 - b ^ 2) / c ^ 2 = sin (A - B) / sin C :=
sorry

theorem sides_sum_comparison :
  a ^ 2 + b ^ 2 + c ^ 2 ≥ 4 * sqrt 3 * S :=
sorry

end sin_C_value_proof_a2_b2_fraction_proof_sides_sum_comparison_l37_37992


namespace book_store_sold_total_copies_by_saturday_l37_37099

def copies_sold_on_monday : ℕ := 15
def copies_sold_on_tuesday : ℕ := copies_sold_on_monday * 2
def copies_sold_on_wednesday : ℕ := copies_sold_on_tuesday + (copies_sold_on_tuesday / 2)
def copies_sold_on_thursday : ℕ := copies_sold_on_wednesday + (copies_sold_on_wednesday / 2)
def copies_sold_on_friday_pre_promotion : ℕ := copies_sold_on_thursday + (copies_sold_on_thursday / 2)
def copies_sold_on_friday_post_promotion : ℕ := copies_sold_on_friday_pre_promotion + (copies_sold_on_friday_pre_promotion / 4)
def copies_sold_on_saturday : ℕ := copies_sold_on_friday_pre_promotion * 7 / 10

def total_copies_sold_by_saturday : ℕ :=
  copies_sold_on_monday + copies_sold_on_tuesday + copies_sold_on_wednesday +
  copies_sold_on_thursday + copies_sold_on_friday_post_promotion + copies_sold_on_saturday

theorem book_store_sold_total_copies_by_saturday : total_copies_sold_by_saturday = 357 :=
by
  -- Proof here
  sorry

end book_store_sold_total_copies_by_saturday_l37_37099


namespace like_terms_sum_l37_37620

theorem like_terms_sum (m n : ℕ) (h1 : 2 * m = 4) (h2 : 3 - n = 1) : m + n = 4 :=
by
  sorry

end like_terms_sum_l37_37620


namespace three_equal_of_four_l37_37411

theorem three_equal_of_four (a b c d : ℕ) 
  (h1 : (a + b)^2 ∣ c * d) 
  (h2 : (a + c)^2 ∣ b * d) 
  (h3 : (a + d)^2 ∣ b * c) 
  (h4 : (b + c)^2 ∣ a * d) 
  (h5 : (b + d)^2 ∣ a * c) 
  (h6 : (c + d)^2 ∣ a * b) : 
  (a = b ∧ b = c) ∨ (a = b ∧ b = d) ∨ (a = c ∧ c = d) ∨ (b = c ∧ c = d) := 
sorry

end three_equal_of_four_l37_37411


namespace daughter_current_age_l37_37859

-- Define the conditions
def mother_current_age := 42
def years_later := 9
def mother_age_in_9_years := mother_current_age + years_later
def daughter_age_in_9_years (D : ℕ) := D + years_later

-- Define the statement we need to prove
theorem daughter_current_age : ∃ D : ℕ, mother_age_in_9_years = 3 * daughter_age_in_9_years D ∧ D = 8 :=
by {
  sorry
}

end daughter_current_age_l37_37859


namespace doughnut_machine_completion_time_l37_37333

-- Define the start time and the time when half the job is completed
def start_time := 8 * 60 -- 8:00 AM in minutes
def half_job_time := 10 * 60 + 30 -- 10:30 AM in minutes

-- Given the machine completes half of the day's job by 10:30 AM
-- Prove that the doughnut machine will complete the entire job by 1:00 PM
theorem doughnut_machine_completion_time :
  half_job_time - start_time = 150 → 
  (start_time + 2 * 150) % (24 * 60) = 13 * 60 :=
by
  sorry

end doughnut_machine_completion_time_l37_37333


namespace log_equality_l37_37404

theorem log_equality (x : ℝ) : (8 : ℝ)^x = 16 ↔ x = 4 / 3 :=
by
  sorry

end log_equality_l37_37404


namespace polygon_has_9_diagonals_has_6_sides_l37_37158

theorem polygon_has_9_diagonals_has_6_sides :
  ∀ (n : ℕ), (∃ D : ℕ, D = n * (n - 3) / 2 ∧ D = 9) → n = 6 := 
by
  sorry

end polygon_has_9_diagonals_has_6_sides_l37_37158


namespace student_weight_l37_37538

-- Define the weights of the student and sister
variables (S R : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := S - 5 = 1.25 * R
def condition2 : Prop := S + R = 104

-- The theorem we want to prove
theorem student_weight (h1 : condition1 S R) (h2 : condition2 S R) : S = 60 := 
by
  sorry

end student_weight_l37_37538


namespace range_values_for_a_l37_37604

def p (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0
def q (x a : ℝ) (ha : 0 < a) : Prop := x^2 - 2 * x + 1 - a^2 ≥ 0

theorem range_values_for_a (a : ℝ) : (∃ ha : 0 < a, (∀ x : ℝ, (¬ p x → q x a ha))) → (0 < a ∧ a ≤ 3) :=
by
  sorry

end range_values_for_a_l37_37604


namespace initial_alarm_time_was_l37_37516

def faster_watch_gain (rate : ℝ) (hours : ℝ) : ℝ := hours * rate

def absolute_time_difference (faster_time : ℝ) (correct_time : ℝ) : ℝ := faster_time - correct_time

theorem initial_alarm_time_was :
  ∀ (rate minutes time_difference : ℝ),
  rate = 2 →
  minutes = 12 →
  time_difference = minutes / rate →
  abs (4 - (4 - time_difference)) = 6 →
  (24 - 6) = 22 :=
by
  intros rate minutes time_difference hrate hminutes htime_diff htime
  sorry

end initial_alarm_time_was_l37_37516


namespace two_digit_number_is_27_l37_37126

theorem two_digit_number_is_27 :
  ∃ n : ℕ, (n / 10 < 10) ∧ (n % 10 < 10) ∧ 
  (100*(n) = 37*(10*(n) + 1)) ∧ 
  n = 27 :=
by {
  sorry
}

end two_digit_number_is_27_l37_37126


namespace EricBenJackMoneySum_l37_37097

noncomputable def EricBenJackTotal (E B J : ℕ) :=
  (E + B + J : ℕ)

theorem EricBenJackMoneySum :
  ∀ (E B J : ℕ), (E = B - 10) → (B = J - 9) → (J = 26) → (EricBenJackTotal E B J) = 50 :=
by
  intros E B J
  intro hE hB hJ
  rw [hJ] at hB
  rw [hB] at hE
  sorry

end EricBenJackMoneySum_l37_37097


namespace total_turtles_l37_37281

theorem total_turtles (num_green_turtles : ℕ) (num_hawksbill_turtles : ℕ) 
  (h1 : num_green_turtles = 800)
  (h2 : num_hawksbill_turtles = 2 * 800 + 800) :
  num_green_turtles + num_hawksbill_turtles = 3200 := 
by
  sorry

end total_turtles_l37_37281


namespace new_three_digit_number_l37_37426

theorem new_three_digit_number (t u : ℕ) (h1 : t < 10) (h2 : u < 10) :
  let original := 10 * t + u
  let new_number := (original * 10) + 2
  new_number = 100 * t + 10 * u + 2 :=
by
  sorry

end new_three_digit_number_l37_37426


namespace rex_cards_left_l37_37053

-- Definitions
def nicole_cards : ℕ := 400
def cindy_cards : ℕ := 2 * nicole_cards
def combined_total : ℕ := nicole_cards + cindy_cards
def rex_cards : ℕ := combined_total / 2
def people_count : ℕ := 4
def cards_per_person : ℕ := rex_cards / people_count

-- Proof statement
theorem rex_cards_left : cards_per_person = 150 := by
  sorry

end rex_cards_left_l37_37053


namespace find_side_b_l37_37276

variables {A B C a b c x : ℝ}

theorem find_side_b 
  (cos_A : ℝ) (cos_C : ℝ) (a : ℝ) (hcosA : cos_A = 4/5) 
  (hcosC : cos_C = 5/13) (ha : a = 1) : 
  b = 21/13 :=
by
  sorry

end find_side_b_l37_37276


namespace find_obtuse_angle_l37_37036

-- Define the conditions
def is_obtuse (α : ℝ) : Prop := 90 < α ∧ α < 180

-- Lean statement assuming the needed conditions
theorem find_obtuse_angle (α : ℝ) (h1 : is_obtuse α) (h2 : 4 * α = 360 + α) : α = 120 :=
by sorry

end find_obtuse_angle_l37_37036


namespace max_remaining_area_l37_37320

theorem max_remaining_area (original_area : ℕ) (rec1 : ℕ × ℕ) (rec2 : ℕ × ℕ) (rec3 : ℕ × ℕ)
  (rec4 : ℕ × ℕ) (total_area_cutout : ℕ):
  original_area = 132 →
  rec1 = (1, 4) →
  rec2 = (2, 2) →
  rec3 = (2, 3) →
  rec4 = (2, 3) →
  total_area_cutout = 20 →
  original_area - total_area_cutout = 112 :=
by
  intros
  sorry

end max_remaining_area_l37_37320


namespace jack_and_jill_meet_distance_l37_37284

theorem jack_and_jill_meet_distance :
  ∃ t : ℝ, t = 15 / 60 ∧ 14 * t ≤ 4 ∧ 15 * (t - 15 / 60) ≤ 4 ∧
  ( 14 * t - 4 + 18 * (t - 2 / 7) = 15 * (t - 15 / 60) ∨ 15 * (t - 15 / 60) = 4 - 18 * (t - 2 / 7) ) ∧
  4 - 15 * (t - 15 / 60) = 851 / 154 :=
sorry

end jack_and_jill_meet_distance_l37_37284


namespace B_completes_remaining_work_in_12_days_l37_37439

-- Definitions for conditions.
def work_rate_a := 1/15
def work_rate_b := 1/18
def days_worked_by_a := 5

-- Calculation of work done by A and the remaining work for B
def work_done_by_a := days_worked_by_a * work_rate_a
def remaining_work := 1 - work_done_by_a

-- Proof statement
theorem B_completes_remaining_work_in_12_days : 
  ∀ (work_rate_a work_rate_b : ℚ), 
    work_rate_a = 1/15 → 
    work_rate_b = 1/18 → 
    days_worked_by_a = 5 → 
    work_done_by_a = days_worked_by_a * work_rate_a → 
    remaining_work = 1 - work_done_by_a → 
    (remaining_work / work_rate_b) = 12 :=
by 
  intros 
  sorry

end B_completes_remaining_work_in_12_days_l37_37439


namespace trapezoid_area_l37_37208

variable (A B C D K : Type)
variable [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty K]

-- Define the lengths as given in the conditions
def AK : ℝ := 16
def DK : ℝ := 4
def CD : ℝ := 6

-- Define the property that the trapezoid ABCD has an inscribed circle
axiom trapezoid_with_inscribed_circle (ABCD : Prop) : Prop

-- The Lean theorem statement
theorem trapezoid_area (ABCD : Prop) (AK DK CD : ℝ) 
  (H1 : trapezoid_with_inscribed_circle ABCD)
  (H2 : AK = 16)
  (H3 : DK = 4)
  (H4 : CD = 6) : 
  ∃ (area : ℝ), area = 432 :=
by
  sorry

end trapezoid_area_l37_37208


namespace raft_min_capacity_l37_37300

theorem raft_min_capacity
  (num_mice : ℕ) (weight_mouse : ℕ)
  (num_moles : ℕ) (weight_mole : ℕ)
  (num_hamsters : ℕ) (weight_hamster : ℕ)
  (raft_condition : ∀ (x y : ℕ), x + y ≥ 2 ∧ (x = weight_mouse ∨ x = weight_mole ∨ x = weight_hamster) ∧ (y = weight_mouse ∨ y = weight_mole ∨ y = weight_hamster) → x + y ≥ 140)
  : 140 ≤ ((num_mice*weight_mouse + num_moles*weight_mole + num_hamsters*weight_hamster) / 2) := sorry

end raft_min_capacity_l37_37300


namespace base_b_eq_five_l37_37866

theorem base_b_eq_five (b : ℕ) (h1 : 1225 = b^3 + 2 * b^2 + 2 * b + 5) (h2 : 35 = 3 * b + 5) :
    (3 * b + 5)^2 = b^3 + 2 * b^2 + 2 * b + 5 ↔ b = 5 :=
by
  sorry

end base_b_eq_five_l37_37866


namespace ezekiel_new_shoes_l37_37633

-- condition Ezekiel bought 3 pairs of shoes
def pairs_of_shoes : ℕ := 3

-- condition Each pair consists of 2 shoes
def shoes_per_pair : ℕ := 2

-- proving the number of new shoes Ezekiel has
theorem ezekiel_new_shoes (pairs_of_shoes shoes_per_pair : ℕ) : pairs_of_shoes * shoes_per_pair = 6 :=
by
  sorry

end ezekiel_new_shoes_l37_37633


namespace g_inv_g_inv_14_l37_37547

noncomputable def g (x : ℝ) := 3 * x - 4
noncomputable def g_inv (x : ℝ) := (x + 4) / 3

theorem g_inv_g_inv_14 : g_inv (g_inv 14) = 10 / 3 :=
by sorry

end g_inv_g_inv_14_l37_37547


namespace months_passed_l37_37141

-- Let's define our conditions in mathematical terms
def received_bones (months : ℕ) : ℕ := 10 * months
def buried_bones : ℕ := 42
def available_bones : ℕ := 8
def total_bones (months : ℕ) : Prop := received_bones months = buried_bones + available_bones

-- We need to prove that the number of months (x) satisfies the condition
theorem months_passed (x : ℕ) : total_bones x → x = 5 :=
by
  sorry

end months_passed_l37_37141


namespace union_of_P_and_neg_RQ_l37_37590

noncomputable def R : Set ℝ := Set.univ

noncomputable def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

noncomputable def Q : Set ℝ := {x | -2 < x ∧ x < 2}

noncomputable def neg_RQ : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}

theorem union_of_P_and_neg_RQ : 
  P ∪ neg_RQ = {x | x ≤ -2 ∨ 1 ≤ x ∧ x ≤ 3} :=
by
  sorry

end union_of_P_and_neg_RQ_l37_37590


namespace product_expression_l37_37896

theorem product_expression :
  (3^4 - 1) / (3^4 + 1) * (4^4 - 1) / (4^4 + 1) * (5^4 - 1) / (5^4 + 1) * (6^4 - 1) / (6^4 + 1) * (7^4 - 1) / (7^4 + 1) = 880 / 91 := by
sorry

end product_expression_l37_37896


namespace theta_in_second_quadrant_l37_37880

theorem theta_in_second_quadrant
  (θ : ℝ)
  (h1 : Real.sin θ > 0)
  (h2 : Real.tan θ < 0) :
  (π / 2 < θ) ∧ (θ < π) :=
by
  sorry

end theta_in_second_quadrant_l37_37880


namespace gross_profit_percentage_l37_37485

theorem gross_profit_percentage (sales_price gross_profit : ℝ) (h_sales_price : sales_price = 91) (h_gross_profit : gross_profit = 56) :
  (gross_profit / (sales_price - gross_profit)) * 100 = 160 :=
by
  sorry

end gross_profit_percentage_l37_37485


namespace largest_sum_of_two_3_digit_numbers_l37_37640

theorem largest_sum_of_two_3_digit_numbers : 
  ∃ (a b c d e f : ℕ), 
    (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧ (1 ≤ c ∧ c ≤ 6) ∧
    (1 ≤ d ∧ d ≤ 6) ∧ (1 ≤ e ∧ e ≤ 6) ∧ (1 ≤ f ∧ f ≤ 6) ∧
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
     c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
     d ≠ e ∧ d ≠ f ∧ 
     e ≠ f) ∧ 
    (100 * (a + d) + 10 * (b + e) + (c + f) = 1173) :=
by
  sorry

end largest_sum_of_two_3_digit_numbers_l37_37640


namespace distance_traveled_l37_37738

-- Define constants for speed and time
def speed : ℝ := 60
def time : ℝ := 5

-- Define the expected distance
def expected_distance : ℝ := 300

-- Theorem statement
theorem distance_traveled : speed * time = expected_distance :=
by
  sorry

end distance_traveled_l37_37738


namespace max_value_of_g_l37_37958

noncomputable def g (x : ℝ) : ℝ := min (min (3 * x + 3) ((1 / 3) * x + 1)) (-2 / 3 * x + 8)

theorem max_value_of_g : ∃ x : ℝ, g x = 10 / 3 :=
by
  sorry

end max_value_of_g_l37_37958


namespace Alfred_repair_cost_l37_37543

noncomputable def scooter_price : ℕ := 4700
noncomputable def sale_price : ℕ := 5800
noncomputable def gain_percent : ℚ := 9.433962264150944
noncomputable def gain_value (repair_cost : ℚ) : ℚ := sale_price - (scooter_price + repair_cost)

theorem Alfred_repair_cost : ∃ R : ℚ, gain_percent = (gain_value R / (scooter_price + R)) * 100 ∧ R = 600 :=
by
  sorry

end Alfred_repair_cost_l37_37543


namespace circle_radius_l37_37133

theorem circle_radius 
  (x y : ℝ)
  (h : x^2 + y^2 + 36 = 6 * x + 24 * y) : 
  ∃ (r : ℝ), r = Real.sqrt 117 :=
by 
  sorry

end circle_radius_l37_37133


namespace average_salary_of_technicians_l37_37720

theorem average_salary_of_technicians
  (total_workers : ℕ)
  (avg_salary_all_workers : ℕ)
  (total_technicians : ℕ)
  (avg_salary_non_technicians : ℕ)
  (h1 : total_workers = 18)
  (h2 : avg_salary_all_workers = 8000)
  (h3 : total_technicians = 6)
  (h4 : avg_salary_non_technicians = 6000) :
  (72000 / total_technicians) = 12000 := 
  sorry

end average_salary_of_technicians_l37_37720


namespace water_depth_upright_l37_37976

def tank_is_right_cylindrical := true
def tank_height := 18.0
def tank_diameter := 6.0
def tank_initial_position_is_flat := true
def water_depth_flat := 4.0

theorem water_depth_upright : water_depth_flat = 4.0 :=
by
  sorry

end water_depth_upright_l37_37976


namespace tom_tickets_l37_37063

theorem tom_tickets :
  let tickets_whack_a_mole := 32
  let tickets_skee_ball := 25
  let tickets_spent_on_hat := 7
  let total_tickets := tickets_whack_a_mole + tickets_skee_ball
  let tickets_left := total_tickets - tickets_spent_on_hat
  tickets_left = 50 :=
by
  sorry

end tom_tickets_l37_37063


namespace solution_set_of_inverse_inequality_l37_37270

open Function

variable {f : ℝ → ℝ}

theorem solution_set_of_inverse_inequality 
  (h_decreasing : ∀ x y, x < y → f y < f x)
  (h_A : f (-2) = 2)
  (h_B : f 2 = -2)
  : { x : ℝ | |(invFun f (x + 1))| ≤ 2 } = { x : ℝ | -3 ≤ x ∧ x ≤ 1 } :=
sorry

end solution_set_of_inverse_inequality_l37_37270


namespace solve_x_l37_37983

theorem solve_x (x : ℚ) : (∀ z : ℚ, 10 * x * z - 15 * z + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := 
by
  sorry

end solve_x_l37_37983


namespace average_postcards_collected_per_day_l37_37082

theorem average_postcards_collected_per_day 
    (a : ℕ) (d : ℕ) (n : ℕ) 
    (h_a : a = 10)
    (h_d : d = 12)
    (h_n : n = 7) :
    (a + (a + (n - 1) * d)) / 2 = 46 := by
  sorry

end average_postcards_collected_per_day_l37_37082


namespace fraction_simplification_l37_37606

theorem fraction_simplification (a b : ℚ) (h : b / a = 3 / 5) : (a - b) / a = 2 / 5 :=
by
  sorry

end fraction_simplification_l37_37606


namespace minimize_perimeter_isosceles_l37_37602

noncomputable def inradius (A B C : ℝ) (r : ℝ) : Prop := sorry -- Define inradius

theorem minimize_perimeter_isosceles (A B C : ℝ) (r : ℝ) 
  (h1 : A + B + C = 180) -- Angles sum to 180 degrees
  (h2 : inradius A B C r) -- Given inradius
  (h3 : A = fixed_angle) -- Given fixed angle A
  : B = C :=
by sorry

end minimize_perimeter_isosceles_l37_37602


namespace division_proof_l37_37415

-- Defining the given conditions
def total_books := 1200
def first_div := 3
def second_div := 4
def final_books_per_category := 15

-- Calculating the number of books per each category after each division
def books_per_first_category := total_books / first_div
def books_per_second_group := books_per_first_category / second_div

-- Correcting the third division to ensure each part has 15 books
def third_div := books_per_second_group / final_books_per_category
def rounded_parts := (books_per_second_group : ℕ) / final_books_per_category -- Rounded to the nearest integer

-- The number of final parts must be correct to ensure the total final categories
def final_division := first_div * second_div * rounded_parts

-- Required proof statement
theorem division_proof : final_division = 84 ∧ books_per_second_group = final_books_per_category :=
by 
  sorry

end division_proof_l37_37415


namespace arithmetic_series_first_term_l37_37562

theorem arithmetic_series_first_term (a d : ℚ) 
  (h1 : 15 * (2 * a + 29 * d) = 450) 
  (h2 : 15 * (2 * a + 89 * d) = 1950) : 
  a = -55 / 6 :=
by 
  sorry

end arithmetic_series_first_term_l37_37562


namespace passengers_at_18_max_revenue_l37_37114

noncomputable def P (t : ℝ) : ℝ :=
if 10 ≤ t ∧ t < 20 then 500 - 4 * (20 - t)^2 else
if 20 ≤ t ∧ t ≤ 30 then 500 else 0

noncomputable def Q (t : ℝ) : ℝ :=
if 10 ≤ t ∧ t < 20 then -8 * t - (1800 / t) + 320 else
if 20 ≤ t ∧ t ≤ 30 then 1400 / t else 0

-- 1. Prove P(18) = 484
theorem passengers_at_18 : P 18 = 484 := sorry

-- 2. Prove that Q(t) is maximized at t = 15 with a maximum value of 80
theorem max_revenue : ∃ t, Q t = 80 ∧ t = 15 := sorry

end passengers_at_18_max_revenue_l37_37114


namespace solve_for_x_l37_37887

theorem solve_for_x : 
  ∃ x₁ x₂ : ℝ, abs (x₁ - 0.175) < 1e-3 ∧ abs (x₂ - 18.325) < 1e-3 ∧
    (∀ x : ℝ, (8 * x ^ 2 + 120 * x + 7) / (3 * x + 10) = 4 * x + 2 → x = x₁ ∨ x = x₂) := 
by 
  sorry

end solve_for_x_l37_37887


namespace angle_comparison_l37_37078

theorem angle_comparison :
  let A := 60.4
  let B := 60.24
  let C := 60.24
  A > B ∧ B = C :=
by
  sorry

end angle_comparison_l37_37078


namespace find_certain_number_l37_37891

theorem find_certain_number (x : ℝ) (h : 0.7 * x = 28) : x = 40 := 
by
  sorry

end find_certain_number_l37_37891


namespace negation_of_existential_proposition_l37_37219

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, Real.exp x < 0) = (∀ x : ℝ, Real.exp x ≥ 0) :=
sorry

end negation_of_existential_proposition_l37_37219


namespace initial_tanks_hold_fifteen_fish_l37_37125

theorem initial_tanks_hold_fifteen_fish (t : Nat) (additional_tanks : Nat) (fish_per_additional_tank : Nat) (total_fish : Nat) :
  t = 3 ∧ additional_tanks = 3 ∧ fish_per_additional_tank = 10 ∧ total_fish = 75 → 
  ∀ (F : Nat), (F * t) = 45 → F = 15 :=
by
  sorry

end initial_tanks_hold_fifteen_fish_l37_37125


namespace minimum_value_x_plus_y_l37_37235

theorem minimum_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y^2 = 1) : x + y = 16 :=
sorry

end minimum_value_x_plus_y_l37_37235


namespace measure_of_angle_B_l37_37277

-- Define the conditions and the goal as a theorem
theorem measure_of_angle_B (A B C : ℝ)
  (h1 : C = 2 * B)
  (h2 : A = 3 * B)
  (triangle_angle_sum : A + B + C = 180) : B = 30 :=
by
  -- Substitute the conditions into Lean to express and prove the statement
  sorry

end measure_of_angle_B_l37_37277


namespace parallel_lines_a_values_l37_37390

theorem parallel_lines_a_values (a : Real) : 
  (∃ k : Real, 2 = k * a ∧ -a = k * (-8)) ↔ (a = 4 ∨ a = -4) := sorry

end parallel_lines_a_values_l37_37390


namespace coffee_on_Thursday_coffee_on_Friday_average_coffee_l37_37823

noncomputable def coffee_consumption (k h : ℝ) : ℝ := k / h

theorem coffee_on_Thursday : coffee_consumption 24 4 = 6 :=
by sorry

theorem coffee_on_Friday : coffee_consumption 24 10 = 2.4 :=
by sorry

theorem average_coffee : 
  (coffee_consumption 24 8 + coffee_consumption 24 4 + coffee_consumption 24 10) / 3 = 3.8 :=
by sorry

end coffee_on_Thursday_coffee_on_Friday_average_coffee_l37_37823


namespace main_theorem_l37_37615

-- Define the interval (3π/4, π)
def theta_range (θ : ℝ) : Prop :=
  (3 * Real.pi / 4) < θ ∧ θ < Real.pi

-- Define the condition
def inequality_condition (θ x : ℝ) : Prop :=
  x^2 * Real.sin θ - x * (1 - x) + (1 - x)^2 * Real.cos θ + 2 * x * (1 - x) * Real.sqrt (Real.cos θ * Real.sin θ) > 0

-- The main theorem
theorem main_theorem (θ x : ℝ) (hθ : theta_range θ) (hx : 0 ≤ x ∧ x ≤ 1) : inequality_condition θ x :=
by
  sorry

end main_theorem_l37_37615


namespace vector_simplification_l37_37075

variables (V : Type) [AddCommGroup V]

variables (CE AC DE AD : V)

theorem vector_simplification :
  CE + AC - DE - AD = 0 :=
by
  sorry

end vector_simplification_l37_37075


namespace min_m_plus_inv_m_min_frac_expr_l37_37403

-- Sub-problem (1): Minimum value of m + 1/m for m > 0.
theorem min_m_plus_inv_m (m : ℝ) (h : m > 0) : m + 1/m = 2 :=
sorry

-- Sub-problem (2): Minimum value of (x^2 + x - 5)/(x - 2) for x > 2.
theorem min_frac_expr (x : ℝ) (h : x > 2) : (x^2 + x - 5)/(x - 2) = 7 :=
sorry

end min_m_plus_inv_m_min_frac_expr_l37_37403


namespace probability_of_matching_correctly_l37_37242

-- Define the number of plants and seedlings.
def num_plants : ℕ := 4

-- Define the number of total arrangements.
def total_arrangements : ℕ := Nat.factorial num_plants

-- Define the number of correct arrangements.
def correct_arrangements : ℕ := 1

-- Define the probability of a correct guess.
def probability_of_correct_guess : ℚ := correct_arrangements / total_arrangements

-- The problem requires to prove that the probability of correct guess is 1/24
theorem probability_of_matching_correctly :
  probability_of_correct_guess = 1 / 24 :=
  by
    sorry

end probability_of_matching_correctly_l37_37242


namespace savings_together_vs_separate_l37_37816

def price_per_window : ℕ := 100

def free_windows_per_5_purchased : ℕ := 2

def daves_windows_needed : ℕ := 10

def dougs_windows_needed : ℕ := 11

def total_windows_needed : ℕ := daves_windows_needed + dougs_windows_needed

-- Cost calculation for Dave's windows with the offer
def daves_cost_with_offer : ℕ := 8 * price_per_window

-- Cost calculation for Doug's windows with the offer
def dougs_cost_with_offer : ℕ := 9 * price_per_window

-- Total cost calculation if purchased separately with the offer
def total_cost_separately_with_offer : ℕ := daves_cost_with_offer + dougs_cost_with_offer

-- Total cost calculation if purchased together with the offer
def total_cost_together_with_offer : ℕ := 17 * price_per_window

-- Calculate additional savings if Dave and Doug purchase together rather than separately
def additional_savings_together_vs_separate := 
  total_cost_separately_with_offer - total_cost_together_with_offer = 0

theorem savings_together_vs_separate : additional_savings_together_vs_separate := by
  sorry

end savings_together_vs_separate_l37_37816


namespace percentage_within_one_standard_deviation_l37_37661

-- Define the constants
def m : ℝ := sorry     -- mean
def g : ℝ := sorry     -- standard deviation
def P : ℝ → ℝ := sorry -- cumulative distribution function

-- The condition that 84% of the distribution is less than m + g
def condition1 : Prop := P (m + g) = 0.84

-- The condition that the distribution is symmetric about the mean
def symmetric_distribution (P : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, P (m + (m - x)) = 1 - P x

-- The problem asks to prove that 68% of the distribution lies within one standard deviation of the mean
theorem percentage_within_one_standard_deviation 
  (h₁ : condition1)
  (h₂ : symmetric_distribution P m) : 
  P (m + g) - P (m - g) = 0.68 :=
sorry

end percentage_within_one_standard_deviation_l37_37661


namespace find_tangent_perpendicular_t_l37_37699

noncomputable def y (x : ℝ) : ℝ := x * Real.log x

theorem find_tangent_perpendicular_t (t : ℝ) (ht : 0 < t) (h_perpendicular : (1 : ℝ) * (1 + Real.log t) = -1) :
  t = Real.exp (-2) :=
by
  sorry

end find_tangent_perpendicular_t_l37_37699


namespace area_of_30_60_90_triangle_hypotenuse_6sqrt2_l37_37642

theorem area_of_30_60_90_triangle_hypotenuse_6sqrt2 :
  ∀ (a b c : ℝ),
  a = 3 * Real.sqrt 2 →
  b = 3 * Real.sqrt 6 →
  c = 6 * Real.sqrt 2 →
  c = 2 * a →
  (1 / 2) * a * b = 18 * Real.sqrt 3 :=
by
  intro a b c ha hb hc h2a
  sorry

end area_of_30_60_90_triangle_hypotenuse_6sqrt2_l37_37642


namespace cos_triple_angle_l37_37964

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := by
  sorry

end cos_triple_angle_l37_37964


namespace charles_richard_difference_in_dimes_l37_37644

variable (q : ℕ)

-- Charles' quarters
def charles_quarters : ℕ := 5 * q + 1

-- Richard's quarters
def richard_quarters : ℕ := q + 5

-- Difference in quarters
def diff_quarters : ℕ := charles_quarters q - richard_quarters q

-- Difference in dimes
def diff_dimes : ℕ := (diff_quarters q) * 5 / 2

theorem charles_richard_difference_in_dimes : diff_dimes q = 10 * (q - 1) := by
  sorry

end charles_richard_difference_in_dimes_l37_37644


namespace multiply_expression_l37_37428

-- Definitions of variables
def a (x y : ℝ) := 3 * x^2
def b (x y : ℝ) := 4 * y^3

-- Theorem statement
theorem multiply_expression (x y : ℝ) :
  ((a x y) - (b x y)) * ((a x y)^2 + (a x y) * (b x y) + (b x y)^2) = 27 * x^6 - 64 * y^9 := 
by 
  -- Placeholder for the proof
  sorry

end multiply_expression_l37_37428


namespace range_of_f_l37_37365

noncomputable def f (x : ℕ) : ℤ := x^2 - 3 * x

def domain : Finset ℕ := {1, 2, 3}

def range : Finset ℤ := {-2, 0}

theorem range_of_f :
  Finset.image f domain = range :=
by
  sorry

end range_of_f_l37_37365


namespace sanya_towels_count_l37_37790

-- Defining the conditions based on the problem
def towels_per_hour := 7
def hours_per_day := 2
def days_needed := 7

-- The main statement to prove
theorem sanya_towels_count : 
  (towels_per_hour * hours_per_day * days_needed = 98) :=
by
  sorry

end sanya_towels_count_l37_37790


namespace remainder_with_conditions_l37_37605

theorem remainder_with_conditions (a b c d : ℕ) (h1 : a % 53 = 33) (h2 : b % 53 = 15) (h3 : c % 53 = 27) (h4 : d % 53 = 8) :
  ((a + b + c + d + 10) % 53) = 40 :=
by
  sorry

end remainder_with_conditions_l37_37605


namespace original_denominator_l37_37083

theorem original_denominator (d : ℕ) (h : 3 * (d : ℚ) = 2) : d = 3 := 
by
  sorry

end original_denominator_l37_37083


namespace solve_eq_solve_ineq_l37_37081

-- Proof Problem 1 statement
theorem solve_eq (x : ℝ) : (2 / (x + 3) - (x - 3) / (2 * x + 6) = 1) → (x = 1 / 3) :=
by sorry

-- Proof Problem 2 statement
theorem solve_ineq (x : ℝ) : (2 * x - 1 > 3 * (x - 1)) ∧ ((5 - x) / 2 < x + 4) → (-1 < x ∧ x < 2) :=
by sorry

end solve_eq_solve_ineq_l37_37081


namespace length_minus_width_l37_37487

theorem length_minus_width 
  (area length diff width : ℝ)
  (h_area : area = 171)
  (h_length : length = 19.13)
  (h_diff : diff = length - width)
  (h_area_eq : area = length * width) :
  diff = 10.19 := 
by {
  sorry
}

end length_minus_width_l37_37487


namespace find_number_l37_37567

theorem find_number (x : ℕ) (h : x + 5 * 8 = 340) : x = 300 :=
sorry

end find_number_l37_37567


namespace mary_flour_l37_37218

-- Defining the conditions
def total_flour : ℕ := 11
def total_sugar : ℕ := 7
def flour_difference : ℕ := 2

-- The problem we want to prove
theorem mary_flour (F : ℕ) (C : ℕ) (S : ℕ)
  (h1 : C + 2 = S)
  (h2 : total_flour = F + C)
  (h3 : S = total_sugar) :
  F = 2 :=
by
  sorry

end mary_flour_l37_37218


namespace simplify_sqrt_product_l37_37598

theorem simplify_sqrt_product : (Real.sqrt (3 * 5) * Real.sqrt (3 ^ 5 * 5 ^ 5) = 3375) :=
  sorry

end simplify_sqrt_product_l37_37598


namespace rival_awards_l37_37227

theorem rival_awards (jessie_multiple : ℕ) (scott_awards : ℕ) (rival_multiple : ℕ) 
  (h1 : jessie_multiple = 3) 
  (h2 : scott_awards = 4) 
  (h3 : rival_multiple = 2) 
  : (rival_multiple * (jessie_multiple * scott_awards) = 24) :=
by 
  sorry

end rival_awards_l37_37227


namespace find_c_of_perpendicular_lines_l37_37949

theorem find_c_of_perpendicular_lines (c : ℤ) :
  (∀ x y : ℤ, y = -3 * x + 4 → ∃ y' : ℤ, y' = (c * x + 18) / 9) →
  c = 3 :=
by
  sorry

end find_c_of_perpendicular_lines_l37_37949


namespace rick_books_division_l37_37849

theorem rick_books_division (books_per_group initial_books final_groups : ℕ) 
  (h_initial : initial_books = 400) 
  (h_books_per_group : books_per_group = 25) 
  (h_final_groups : final_groups = 16) : 
  ∃ divisions : ℕ, (divisions = 4) ∧ 
    ∃ f : ℕ → ℕ, 
    (f 0 = initial_books) ∧ 
    (f divisions = books_per_group * final_groups) ∧ 
    (∀ n, 1 ≤ n → n ≤ divisions → f n = f (n - 1) / 2) := 
by 
  sorry

end rick_books_division_l37_37849


namespace trader_cloth_sale_l37_37119

theorem trader_cloth_sale (total_SP : ℕ) (profit_per_meter : ℕ) (cost_per_meter : ℕ) (SP_per_meter : ℕ)
  (h1 : total_SP = 8400) (h2 : profit_per_meter = 12) (h3 : cost_per_meter = 128) (h4 : SP_per_meter = cost_per_meter + profit_per_meter) :
  ∃ (x : ℕ), SP_per_meter * x = total_SP ∧ x = 60 :=
by
  -- We will skip the proof using sorry
  sorry

end trader_cloth_sale_l37_37119


namespace total_cards_beginning_l37_37800

-- Define the initial conditions
def num_boxes_orig : ℕ := 2 + 5  -- Robie originally had 2 + 5 boxes
def cards_per_box : ℕ := 10      -- Each box contains 10 cards
def extra_cards : ℕ := 5         -- 5 cards were not placed in a box

-- Prove the total number of cards Robie had in the beginning
theorem total_cards_beginning : (num_boxes_orig * cards_per_box) + extra_cards = 75 :=
by sorry

end total_cards_beginning_l37_37800


namespace triangle_square_ratio_l37_37862

theorem triangle_square_ratio (s_t s_s : ℝ) (h : 3 * s_t = 4 * s_s) : s_t / s_s = 4 / 3 := by
  sorry

end triangle_square_ratio_l37_37862


namespace slope_of_asymptotes_l37_37398

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  (x - 2)^2 / 144 - (y + 3)^2 / 81 = 1

-- The theorem stating the slope of the asymptotes
theorem slope_of_asymptotes : ∀ x y : ℝ, hyperbola x y → (∃ m : ℝ, m = 3 / 4) :=
by
  sorry

end slope_of_asymptotes_l37_37398


namespace total_cost_of_hotel_stay_l37_37185

-- Define the necessary conditions
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- State the problem
theorem total_cost_of_hotel_stay :
  (cost_per_night_per_person * number_of_people * number_of_nights) = 360 := by
  sorry

end total_cost_of_hotel_stay_l37_37185


namespace least_integer_value_of_x_l37_37087

theorem least_integer_value_of_x (x : ℤ) (h : 3 * |x| + 4 < 19) : x = -4 :=
by sorry

end least_integer_value_of_x_l37_37087


namespace symm_y_axis_l37_37344

noncomputable def f (x : ℝ) : ℝ := abs x

theorem symm_y_axis (x : ℝ) : f (-x) = f (x) := by
  sorry

end symm_y_axis_l37_37344


namespace return_trip_amount_l37_37713

noncomputable def gasoline_expense : ℝ := 8
noncomputable def lunch_expense : ℝ := 15.65
noncomputable def gift_expense_per_person : ℝ := 5
noncomputable def grandma_gift_per_person : ℝ := 10
noncomputable def initial_amount : ℝ := 50

theorem return_trip_amount : 
  let total_expense := gasoline_expense + lunch_expense + (gift_expense_per_person * 2)
  let total_money_gifted := grandma_gift_per_person * 2
  initial_amount - total_expense + total_money_gifted = 36.35 :=
by 
  let total_expense := gasoline_expense + lunch_expense + (gift_expense_per_person * 2)
  let total_money_gifted := grandma_gift_per_person * 2
  sorry

end return_trip_amount_l37_37713


namespace shirt_cost_correct_l37_37985

-- Define the conditions
def pants_cost : ℝ := 9.24
def bill_amount : ℝ := 20
def change_received : ℝ := 2.51

-- Calculate total spent and shirt cost
def total_spent : ℝ := bill_amount - change_received
def shirt_cost : ℝ := total_spent - pants_cost

-- The theorem statement
theorem shirt_cost_correct : shirt_cost = 8.25 := by
  sorry

end shirt_cost_correct_l37_37985


namespace international_postage_surcharge_l37_37367

theorem international_postage_surcharge 
  (n_letters : ℕ) 
  (std_postage_per_letter : ℚ) 
  (n_international : ℕ) 
  (total_cost : ℚ) 
  (cents_per_dollar : ℚ) 
  (std_total_cost : ℚ) 
  : 
  n_letters = 4 →
  std_postage_per_letter = 108 / 100 →
  n_international = 2 →
  total_cost = 460 / 100 →
  cents_per_dollar = 100 →
  std_total_cost = n_letters * std_postage_per_letter →
  (total_cost - std_total_cost) / n_international * cents_per_dollar = 14 := 
sorry

end international_postage_surcharge_l37_37367


namespace unique_zero_of_quadratic_l37_37502

theorem unique_zero_of_quadratic {m : ℝ} (h : ∃ x : ℝ, x^2 + 2*x + m = 0 ∧ (∀ y : ℝ, y^2 + 2*y + m = 0 → y = x)) : m = 1 :=
sorry

end unique_zero_of_quadratic_l37_37502


namespace miles_tankful_highway_l37_37931

variable (miles_tankful_city : ℕ)
variable (mpg_city : ℕ)
variable (mpg_highway : ℕ)

-- Relationship between miles per gallon in city and highway
axiom h_mpg_relation : mpg_highway = mpg_city + 18

-- Given the car travels 336 miles per tankful of gasoline in the city
axiom h_miles_tankful_city : miles_tankful_city = 336

-- Given the car travels 48 miles per gallon in the city
axiom h_mpg_city : mpg_city = 48

-- Prove the car travels 462 miles per tankful of gasoline on the highway
theorem miles_tankful_highway : ∃ (miles_tankful_highway : ℕ), miles_tankful_highway = (mpg_highway * (miles_tankful_city / mpg_city)) := 
by 
  exists (66 * (336 / 48)) -- Since 48 + 18 = 66 and 336 / 48 = 7, 66 * 7 = 462
  sorry

end miles_tankful_highway_l37_37931


namespace max_value_a_plus_b_plus_c_plus_d_eq_34_l37_37170

theorem max_value_a_plus_b_plus_c_plus_d_eq_34 :
  ∃ (a b c d : ℕ), (∀ (x y: ℝ), 0 < x → 0 < y → x^2 - 2 * x * y + 3 * y^2 = 10 → x^2 + 2 * x * y + 3 * y^2 = (a + b * Real.sqrt c) / d) ∧ a + b + c + d = 34 :=
sorry

end max_value_a_plus_b_plus_c_plus_d_eq_34_l37_37170


namespace embankment_construction_l37_37722

theorem embankment_construction :
  (∃ r : ℚ, 0 < r ∧ (1 / 2 = 60 * r * 3)) →
  (∃ t : ℕ, 1 = 45 * 1 / 360 * t) :=
by
  sorry

end embankment_construction_l37_37722


namespace price_diff_is_correct_l37_37621

-- Define initial conditions
def initial_price : ℝ := 30
def flat_discount : ℝ := 5
def percent_discount : ℝ := 0.25
def sales_tax : ℝ := 0.10

def price_after_flat_discount (price : ℝ) : ℝ :=
  price - flat_discount

def price_after_percent_discount (price : ℝ) : ℝ :=
  price * (1 - percent_discount)

def price_after_tax (price : ℝ) : ℝ :=
  price * (1 + sales_tax)

def final_price_method1 : ℝ :=
  price_after_tax (price_after_percent_discount (price_after_flat_discount initial_price))

def final_price_method2 : ℝ :=
  price_after_tax (price_after_flat_discount (price_after_percent_discount initial_price))

def difference_in_cents : ℝ :=
  (final_price_method1 - final_price_method2) * 100

-- Lean statement to prove the final difference in cents
theorem price_diff_is_correct : difference_in_cents = 137.5 :=
  by sorry

end price_diff_is_correct_l37_37621


namespace minimum_a_condition_l37_37769

theorem minimum_a_condition (a : ℝ) (h₀ : 0 < a) 
  (h₁ : ∀ x : ℝ, 1 < x → x + a / (x - 1) ≥ 5) :
  4 ≤ a :=
sorry

end minimum_a_condition_l37_37769


namespace sequence_behavior_l37_37650

theorem sequence_behavior (b : ℕ → ℕ) :
  (∀ n, b n = n) ∨ ∃ N, ∀ n, n ≥ N → b n = b N :=
sorry

end sequence_behavior_l37_37650


namespace distance_between_consecutive_trees_l37_37774

-- Definitions from the problem statement
def yard_length : ℕ := 414
def number_of_trees : ℕ := 24
def number_of_intervals : ℕ := number_of_trees - 1
def distance_between_trees : ℕ := yard_length / number_of_intervals

-- Main theorem we want to prove
theorem distance_between_consecutive_trees :
  distance_between_trees = 18 := by
  -- Proof would go here
  sorry

end distance_between_consecutive_trees_l37_37774


namespace scientific_notation_of_19400000000_l37_37013

theorem scientific_notation_of_19400000000 :
  ∃ a n, 1 ≤ |a| ∧ |a| < 10 ∧ (19400000000 : ℝ) = a * 10^n ∧ a = 1.94 ∧ n = 10 :=
by
  sorry

end scientific_notation_of_19400000000_l37_37013


namespace math_proof_equivalent_l37_37749

theorem math_proof_equivalent :
  (60 + 5 * 12) / (Real.sqrt 180 / 3) ^ 2 = 6 := by
  sorry

end math_proof_equivalent_l37_37749


namespace sum_of_roots_of_quadratic_eq_l37_37308

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

-- Prove that the sum of the roots of the given quadratic equation is 6
theorem sum_of_roots_of_quadratic_eq : 
  ∀ x y : ℝ, quadratic_eq 1 (-6) 8 x → quadratic_eq 1 (-6) 8 y → (x + y) = 6 :=
by
  sorry

end sum_of_roots_of_quadratic_eq_l37_37308


namespace complement_correct_l37_37542

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set A as the set of real numbers such that -1 ≤ x < 2
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

-- Define the complement of A in U
def complement_U_A : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 2}

-- The proof statement: the complement of A in U is the expected set
theorem complement_correct : (U \ A) = complement_U_A := 
by
  sorry

end complement_correct_l37_37542


namespace general_formula_a_general_formula_c_l37_37677

-- Definition of the sequence {a_n}
def S (n : ℕ) : ℕ := n^2 + 2 * n
def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem general_formula_a (n : ℕ) (hn : n > 0) : a n = 2 * n + 1 := sorry

-- Definitions for the second problem
def f (x : ℝ) : ℝ := x^2 + 2 * x
def f' (x : ℝ) : ℝ := 2 * x + 2
def k (n : ℕ) : ℝ := 2 * n + 2

def Q (k : ℝ) : Prop := ∃ (n : ℕ), k = 2 * n + 2
def R (k : ℝ) : Prop := ∃ (n : ℕ), k = 4 * n + 2

def c (n : ℕ) : ℕ := 12 * n - 6

theorem general_formula_c (n : ℕ) (hn1 : 0 < c 10)
    (hn2 : c 10 < 115) : c n = 12 * n - 6 := sorry

end general_formula_a_general_formula_c_l37_37677


namespace volume_of_prism_l37_37760

noncomputable def prismVolume {x y z : ℝ} 
  (h1 : x * y = 20) 
  (h2 : y * z = 12) 
  (h3 : x * z = 8) : ℝ :=
  x * y * z

theorem volume_of_prism (x y z : ℝ)
  (h1 : x * y = 20)
  (h2 : y * z = 12)
  (h3 : x * z = 8) : prismVolume h1 h2 h3 = 8 * Real.sqrt 15 :=
by
  sorry

end volume_of_prism_l37_37760


namespace clock1_runs_10_months_longer_l37_37049

noncomputable def battery_a_charge (C_B : ℝ) := 6 * C_B
noncomputable def clock1_total_charge (C_B : ℝ) := 2 * battery_a_charge C_B
noncomputable def clock2_total_charge (C_B : ℝ) := 2 * C_B
noncomputable def clock2_operating_time := 2
noncomputable def clock1_operating_time (C_B : ℝ) := clock1_total_charge C_B / C_B
noncomputable def operating_time_difference (C_B : ℝ) := clock1_operating_time C_B - clock2_operating_time

theorem clock1_runs_10_months_longer (C_B : ℝ) :
  operating_time_difference C_B = 10 :=
by
  unfold operating_time_difference clock1_operating_time clock2_operating_time clock1_total_charge battery_a_charge
  sorry

end clock1_runs_10_months_longer_l37_37049


namespace minimum_reciprocal_sum_l37_37994

noncomputable def minimum_value_of_reciprocal_sum (x y z : ℝ) : ℝ :=
  if x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 2 then 
    max (1/x + 1/y + 1/z) (9/2)
  else
    0
  
theorem minimum_reciprocal_sum (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2): 
  1/x + 1/y + 1/z ≥ 9/2 :=
sorry

end minimum_reciprocal_sum_l37_37994


namespace least_possible_value_l37_37877

theorem least_possible_value (x : ℚ) (h1 : x > 5 / 3) (h2 : x < 9 / 2) : 
  (9 / 2 - 5 / 3 : ℚ) = 17 / 6 :=
by sorry

end least_possible_value_l37_37877


namespace divisibility_theorem_l37_37387

theorem divisibility_theorem (a b n : ℕ) (h : a^n ∣ b) : a^(n + 1) ∣ (a + 1)^b - 1 :=
by 
sorry

end divisibility_theorem_l37_37387


namespace rational_number_25_units_away_l37_37062

theorem rational_number_25_units_away (x : ℚ) (h : |x| = 2.5) : x = 2.5 ∨ x = -2.5 := 
by
  sorry

end rational_number_25_units_away_l37_37062


namespace ratio_of_cookies_l37_37323

-- Definitions based on the conditions
def initial_cookies : ℕ := 19
def cookies_to_friend : ℕ := 5
def cookies_left : ℕ := 5
def cookies_eaten : ℕ := 2

-- Calculating the number of cookies left after giving cookies to the friend
def cookies_after_giving_to_friend := initial_cookies - cookies_to_friend

-- Maria gave to her family the remaining cookies minus the cookies she has left and she has eaten.
def cookies_given_to_family := cookies_after_giving_to_friend - cookies_eaten - cookies_left

-- The ratio to be proven 1:2, which is mathematically 1/2
theorem ratio_of_cookies : (cookies_given_to_family : ℚ) / (cookies_after_giving_to_friend : ℚ) = 1 / 2 := by
  sorry

end ratio_of_cookies_l37_37323


namespace total_games_played_l37_37346

theorem total_games_played (games_attended games_missed : ℕ) 
  (h_attended : games_attended = 395) 
  (h_missed : games_missed = 469) : 
  games_attended + games_missed = 864 := 
by
  sorry

end total_games_played_l37_37346


namespace heather_total_distance_l37_37921

-- Definitions for distances walked
def distance_car_to_entrance : ℝ := 0.33
def distance_entrance_to_rides : ℝ := 0.33
def distance_rides_to_car : ℝ := 0.08

-- Statement of the problem to be proven
theorem heather_total_distance :
  distance_car_to_entrance + distance_entrance_to_rides + distance_rides_to_car = 0.74 :=
by
  sorry

end heather_total_distance_l37_37921


namespace circle_equation_m_l37_37726
open Real

theorem circle_equation_m (m : ℝ) : (x^2 + y^2 + 4 * x + 2 * y + m = 0 → m < 5) := sorry

end circle_equation_m_l37_37726


namespace unique_integer_solution_l37_37361

theorem unique_integer_solution (x y : ℤ) : 
  x^4 + y^4 = 3 * x^3 * y → x = 0 ∧ y = 0 :=
by
  -- This is where the proof would go
  sorry

end unique_integer_solution_l37_37361


namespace mean_exercise_days_correct_l37_37234

def students_exercise_days : List (Nat × Nat) := 
  [ (2, 0), (4, 1), (5, 2), (7, 3), (5, 4), (3, 5), (1, 6)]

def total_days_exercised : Nat := 
  List.sum (students_exercise_days.map (λ (count, days) => count * days))

def total_students : Nat := 
  List.sum (students_exercise_days.map Prod.fst)

def mean_exercise_days : Float := 
  total_days_exercised.toFloat / total_students.toFloat

theorem mean_exercise_days_correct : Float.round (mean_exercise_days * 100) / 100 = 2.81 :=
by
  sorry -- proof not required

end mean_exercise_days_correct_l37_37234


namespace construct_unit_segment_l37_37355

-- Definitions of the problem
variable (a b : ℝ)

-- Parabola definition
def parabola (x : ℝ) : ℝ := x^2 + a * x + b

-- Statement of the problem in Lean 4
theorem construct_unit_segment
  (h : ∃ x y : ℝ, parabola a b x = y) :
  ∃ (u v : ℝ), abs (u - v) = 1 :=
sorry

end construct_unit_segment_l37_37355


namespace expenditures_ratio_l37_37747

open Real

variables (I1 I2 E1 E2 : ℝ)
variables (x : ℝ)

theorem expenditures_ratio 
  (h1 : I1 = 4500)
  (h2 : I1 / I2 = 5 / 4)
  (h3 : I1 - E1 = 1800)
  (h4 : I2 - E2 = 1800) : 
  E1 / E2 = 3 / 2 :=
by
  have h5 : I1 / 5 = x := by sorry
  have h6 : I2 = 4 * x := by sorry
  have h7 : I2 = 3600 := by sorry
  have h8 : E1 = 2700 := by sorry
  have h9 : E2 = 1800 := by sorry
  exact sorry 

end expenditures_ratio_l37_37747


namespace most_likely_outcome_is_draw_l37_37768

noncomputable def prob_A_win : ℝ := 0.3
noncomputable def prob_A_not_lose : ℝ := 0.7
noncomputable def prob_draw : ℝ := prob_A_not_lose - prob_A_win

theorem most_likely_outcome_is_draw :
  prob_draw = 0.4 ∧ prob_draw > prob_A_win ∧ prob_draw > (1 - prob_A_not_lose) :=
by
  -- proof goes here
  sorry

end most_likely_outcome_is_draw_l37_37768


namespace ratio_surface_area_l37_37920

noncomputable def side_length (a : ℝ) := a
noncomputable def radius (R : ℝ) := R

theorem ratio_surface_area (a R : ℝ) (h : a^3 = (4/3) * Real.pi * R^3) : 
  (6 * a^2) / (4 * Real.pi * R^2) = (3 * (6 / Real.pi)) :=
by sorry

end ratio_surface_area_l37_37920


namespace max_value_of_m_l37_37727

theorem max_value_of_m (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (2 / a) + (1 / b) = 1 / 4) : 2 * a + b ≥ 36 :=
by 
  -- Skipping the proof
  sorry

end max_value_of_m_l37_37727


namespace ramesh_installation_cost_l37_37971

noncomputable def labelled_price (discounted_price : ℝ) (discount_rate : ℝ) : ℝ :=
  discounted_price / (1 - discount_rate)

noncomputable def selling_price (labelled_price : ℝ) (profit_rate : ℝ) : ℝ :=
  labelled_price * (1 + profit_rate)

def ramesh_total_cost (purchase_price transport_cost : ℝ) (installation_cost : ℝ) : ℝ :=
  purchase_price + transport_cost + installation_cost

theorem ramesh_installation_cost :
  ∀ (purchase_price discounted_price transport_cost labelled_price profit_rate selling_price installation_cost : ℝ),
  discounted_price = 12500 → transport_cost = 125 → profit_rate = 0.18 → selling_price = 18880 →
  labelled_price = discounted_price / (1 - 0.20) →
  selling_price = labelled_price * (1 + profit_rate) →
  ramesh_total_cost purchase_price transport_cost installation_cost = selling_price →
  installation_cost = 6255 :=
by
  intros
  sorry

end ramesh_installation_cost_l37_37971


namespace difference_of_numbers_l37_37312

theorem difference_of_numbers (a : ℕ) (h : a + (10 * a + 5) = 30000) : (10 * a + 5) - a = 24548 :=
by
  sorry

end difference_of_numbers_l37_37312


namespace f_of_x_squared_domain_l37_37383

structure FunctionDomain (f : ℝ → ℝ) :=
  (domain : Set ℝ)
  (domain_eq : domain = Set.Icc 0 1)

theorem f_of_x_squared_domain (f : ℝ → ℝ) (h : FunctionDomain f) :
  FunctionDomain (fun x => f (x ^ 2)) :=
{
  domain := Set.Icc (-1) 1,
  domain_eq := sorry
}

end f_of_x_squared_domain_l37_37383


namespace equal_circumradii_l37_37617

-- Define the points and triangles involved
variable (A B C M : Type*) 

-- The circumcircle radius of a triangle is at least R
variable (R R1 R2 R3 : ℝ)

-- Hypotheses: the given conditions
variable (hR1 : R1 ≥ R)
variable (hR2 : R2 ≥ R)
variable (hR3 : R3 ≥ R)

-- The goal: to show that all four radii are equal
theorem equal_circumradii {A B C M : Type*} (R R1 R2 R3 : ℝ) 
    (hR1 : R1 ≥ R) 
    (hR2 : R2 ≥ R) 
    (hR3 : R3 ≥ R): 
    R1 = R ∧ R2 = R ∧ R3 = R := 
by 
  sorry

end equal_circumradii_l37_37617


namespace trigonometric_identity_application_l37_37912

theorem trigonometric_identity_application :
  2 * (Real.sin (35 * Real.pi / 180) * Real.cos (25 * Real.pi / 180) +
       Real.cos (35 * Real.pi / 180) * Real.cos (65 * Real.pi / 180)) = Real.sqrt 3 :=
by sorry

end trigonometric_identity_application_l37_37912


namespace equalChargesAtFour_agencyADecisionWhenTen_l37_37362

-- Define the conditions as constants
def fullPrice : ℕ := 240
def agencyADiscount : ℕ := 50
def agencyBDiscount : ℕ := 60

-- Define the total charge function for both agencies
def totalChargeAgencyA (students: ℕ) : ℕ :=
  fullPrice * students * agencyADiscount / 100 + fullPrice

def totalChargeAgencyB (students: ℕ) : ℕ :=
  fullPrice * (students + 1) * agencyBDiscount / 100

-- Define the equivalence when the number of students is 4
theorem equalChargesAtFour : totalChargeAgencyA 4 = totalChargeAgencyB 4 := by sorry

-- Define the decision when there are 10 students
theorem agencyADecisionWhenTen : totalChargeAgencyA 10 < totalChargeAgencyB 10 := by sorry

end equalChargesAtFour_agencyADecisionWhenTen_l37_37362


namespace simplify_expression_l37_37460

theorem simplify_expression (n : ℕ) :
  (2^(n+5) - 3 * 2^n) / (3 * 2^(n+3)) = 29 / 24 :=
by
  sorry

end simplify_expression_l37_37460


namespace range_of_k_value_of_k_l37_37697

-- Defining the quadratic equation having two real roots condition
def has_real_roots (k : ℝ) : Prop :=
  let Δ := 9 - 4 * (k - 2)
  Δ ≥ 0

-- First part: range of k
theorem range_of_k (k : ℝ) : has_real_roots k ↔ k ≤ 17 / 4 :=
  sorry

-- Second part: specific value of k given additional condition
theorem value_of_k (x1 x2 k : ℝ) (h1 : (x1 + x2) = 3) (h2 : (x1 * x2) = k - 2) (h3 : (x1 + x2 - x1 * x2) = 1) : k = 4 :=
  sorry

end range_of_k_value_of_k_l37_37697


namespace triangle_no_solution_l37_37609

def angleSumOfTriangle : ℝ := 180

def hasNoSolution (a b A : ℝ) : Prop :=
  A >= angleSumOfTriangle

theorem triangle_no_solution {a b A : ℝ} (ha : a = 181) (hb : b = 209) (hA : A = 121) :
  hasNoSolution a b A := sorry

end triangle_no_solution_l37_37609


namespace factor_3x2_minus_3y2_l37_37490

theorem factor_3x2_minus_3y2 (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end factor_3x2_minus_3y2_l37_37490


namespace sum_of_remainders_l37_37188

theorem sum_of_remainders (a b c d : ℕ)
  (ha : a % 17 = 3) (hb : b % 17 = 5) (hc : c % 17 = 7) (hd : d % 17 = 9) :
  (a + b + c + d) % 17 = 7 :=
by
  sorry

end sum_of_remainders_l37_37188


namespace table_area_l37_37924

theorem table_area (A : ℝ) 
  (combined_area : ℝ)
  (coverage_percentage : ℝ)
  (area_two_layers : ℝ)
  (area_three_layers : ℝ)
  (combined_area_eq : combined_area = 220)
  (coverage_percentage_eq : coverage_percentage = 0.80 * A)
  (area_two_layers_eq : area_two_layers = 24)
  (area_three_layers_eq : area_three_layers = 28) :
  A = 275 :=
by
  -- Assumptions and derivations can be filled in.
  sorry

end table_area_l37_37924


namespace side_length_is_36_l37_37177

variable (a : ℝ)

def side_length_of_largest_square (a : ℝ) := 
  2 * (a / 2) ^ 2 + 2 * (a / 4) ^ 2 = 810

theorem side_length_is_36 (h : side_length_of_largest_square a) : a = 36 :=
by
  sorry

end side_length_is_36_l37_37177


namespace cube_root_of_sum_of_powers_l37_37829

theorem cube_root_of_sum_of_powers :
  ∃ (x : ℝ), x = 16 * (4 ^ (1 / 3)) ∧ x = (4^6 + 4^6 + 4^6 + 4^6) ^ (1 / 3) :=
by
  sorry

end cube_root_of_sum_of_powers_l37_37829


namespace part1_part2_part3_l37_37467

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := (1/2) * x^2 - 2 * x
noncomputable def g' (x : ℝ) : ℝ := x - 2
noncomputable def h (x : ℝ) : ℝ := f (x + 1) - g' x

theorem part1 : ∃ x : ℝ, (h x ≤ 2) := sorry

theorem part2 (a b : ℝ) (h1 : 0 < b) (h2 : b < a) : 
  f (a + b) - f (2 * a) < (b - a) / (2 * a) := sorry

theorem part3 (k : ℤ) : (∀ x : ℝ, x > 1 → k * (x - 1) < x * f x + 3 * g' x + 4) ↔ k ≤ 5 := sorry

end part1_part2_part3_l37_37467


namespace symmetry_implies_value_l37_37132

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 3)
noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := Real.cos (2 * x + φ)

theorem symmetry_implies_value :
  (∀ (x : ℝ), ∃ (k : ℤ), ω * x - Real.pi / 3 = k * Real.pi + Real.pi / 2) →
  (∀ (x : ℝ), ∃ (k : ℤ), 2 * x + φ = k * Real.pi) →
  0 < φ → φ < Real.pi →
  ω = 2 →
  φ = Real.pi / 6 →
  g (Real.pi / 3) φ = -Real.sqrt 3 / 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  exact sorry

end symmetry_implies_value_l37_37132


namespace binary_subtraction_to_decimal_l37_37216

theorem binary_subtraction_to_decimal :
  (511 - 63 = 448) :=
by
  sorry

end binary_subtraction_to_decimal_l37_37216


namespace calculate_f_f_f_one_l37_37482

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 1

theorem calculate_f_f_f_one : f (f (f 1)) = 9184 :=
by
  sorry

end calculate_f_f_f_one_l37_37482


namespace problem_statement_l37_37930

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

axiom f_pos (x : ℝ) : x > 0 → f x > 0
axiom f'_less_f (x : ℝ) : f' x < f x
axiom f_has_deriv_at : ∀ x, HasDerivAt f (f' x) x

def a : ℝ := sorry
axiom a_in_range : 0 < a ∧ a < 1

theorem problem_statement : 3 * f 0 > f a ∧ f a > a * f 1 :=
  sorry

end problem_statement_l37_37930


namespace arithmetic_sequence_ratio_l37_37050

variable {a_n : ℕ → ℤ} {S_n : ℕ → ℤ}
variable (d : ℤ)
variable (a1 a3 a4 : ℤ)
variable (h_geom : a3^2 = a1 * a4)
variable (h_seq : ∀ n, a_n (n+1) = a_n n + d)
variable (h_sum : ∀ n, S_n n = (n * (2 * a1 + (n - 1) * d)) / 2)

theorem arithmetic_sequence_ratio :
  (S_n 3 - S_n 2) / (S_n 5 - S_n 3) = 2 :=
by 
  sorry

end arithmetic_sequence_ratio_l37_37050


namespace jack_afternoon_emails_l37_37017

theorem jack_afternoon_emails : 
  ∀ (morning_emails afternoon_emails : ℕ), 
  morning_emails = 6 → 
  afternoon_emails = morning_emails + 2 → 
  afternoon_emails = 8 := 
by
  intros morning_emails afternoon_emails hm ha
  rw [hm] at ha
  exact ha

end jack_afternoon_emails_l37_37017


namespace initial_contribution_l37_37870

theorem initial_contribution (j k l : ℝ)
  (h1 : j + k + l = 1200)
  (h2 : j - 200 + 3 * (k + l) = 1800) :
  j = 800 :=
sorry

end initial_contribution_l37_37870


namespace find_p_q_l37_37222

theorem find_p_q (p q : ℤ) (h : ∀ x : ℤ, (x - 5) * (x + 2) = x^2 + p * x + q) :
  p = -3 ∧ q = -10 :=
by {
  -- The proof would go here, but for now we'll use sorry to indicate it's incomplete.
  sorry
}

end find_p_q_l37_37222


namespace range_of_a_l37_37488

-- Define the propositions p and q
def prop_p (a : ℝ) : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

-- Define the main theorem which combines both propositions and infers the range of a
theorem range_of_a (a : ℝ) : prop_p a ∧ prop_q a → a ≤ -2 := sorry

end range_of_a_l37_37488


namespace find_M_l37_37894

theorem find_M : ∀ M : ℕ, (10 + 11 + 12 : ℕ) / 3 = (2024 + 2025 + 2026 : ℕ) / M → M = 552 :=
by
  intro M
  sorry

end find_M_l37_37894


namespace three_digit_number_count_l37_37867

theorem three_digit_number_count :
  ∃ n : ℕ, n = 15 ∧
  (∀ a b c : ℕ, (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) →
    (100 * a + 10 * b + c = 37 * (a + b + c) → ∃ k : ℕ, k = n)) :=
sorry

end three_digit_number_count_l37_37867


namespace symmetric_slope_angle_l37_37366

-- Define the problem conditions in Lean
def slope_angle (θ : Real) : Prop :=
  0 ≤ θ ∧ θ < Real.pi

-- Statement of the theorem in Lean
theorem symmetric_slope_angle (θ : Real) (h : slope_angle θ) :
  θ = 0 ∨ θ = Real.pi - θ :=
sorry

end symmetric_slope_angle_l37_37366


namespace hall_reunion_attendance_l37_37754

/-- At the Taj Hotel, two family reunions are happening: the Oates reunion and the Hall reunion.
All 150 guests at the hotel attend at least one of the reunions.
70 people attend the Oates reunion.
28 people attend both reunions.
Prove that 108 people attend the Hall reunion. -/
theorem hall_reunion_attendance (total oates both : ℕ) (h_total : total = 150) (h_oates : oates = 70) (h_both : both = 28) :
  ∃ hall : ℕ, total = oates + hall - both ∧ hall = 108 :=
by
  -- Proof will be skipped and not considered for this task
  sorry

end hall_reunion_attendance_l37_37754


namespace midpoint_trajectory_l37_37473

-- Define the data for the problem
variable (P : (ℝ × ℝ)) (Q : ℝ × ℝ)
variable (M : ℝ × ℝ)
variable (x y : ℝ)
variable (hQ : Q = (2*x - 2, 2*y)) -- Definition of point Q based on midpoint M
variable (hC : (Q.1)^2 + (Q.2)^2 = 1) -- Q moves on the circle x^2 + y^2 = 1

-- Define the proof problem
theorem midpoint_trajectory (P : (ℝ × ℝ)) (hP : P = (2, 0)) (M : ℝ × ℝ) (hQ : Q = (2*M.1 - 2, 2*M.2))
  (hC : (Q.1)^2 + (Q.2)^2 = 1) : 4*(M.1 - 1)^2 + 4*(M.2)^2 = 1 := by
  sorry

end midpoint_trajectory_l37_37473


namespace odd_integers_count_between_fractions_l37_37093

theorem odd_integers_count_between_fractions :
  ∃ (count : ℕ), count = 14 ∧
  ∀ (n : ℤ), (25:ℚ)/3 < (n : ℚ) ∧ (n : ℚ) < (73 : ℚ)/2 ∧ (n % 2 = 1) :=
sorry

end odd_integers_count_between_fractions_l37_37093


namespace final_price_is_correct_l37_37630

-- Define the original price and percentages as constants
def original_price : ℝ := 160
def increase_percentage : ℝ := 0.25
def discount_percentage : ℝ := 0.25

-- Calculate increased price
def increased_price : ℝ := original_price * (1 + increase_percentage)
-- Calculate the discount on the increased price
def discount_amount : ℝ := increased_price * discount_percentage
-- Calculate final price after discount
def final_price : ℝ := increased_price - discount_amount

-- Statement of the theorem: prove final price is $150
theorem final_price_is_correct : final_price = 150 :=
by
  -- Proof would go here
  sorry

end final_price_is_correct_l37_37630


namespace tiered_water_pricing_usage_l37_37257

theorem tiered_water_pricing_usage (total_cost : ℤ) (water_used : ℤ) :
  (total_cost = 60) →
  (water_used > 12 ∧ water_used ≤ 18) →
  (3 * 12 + (water_used - 12) * 6 = total_cost) →
  water_used = 16 :=
by
  intros h_cost h_range h_eq
  sorry

end tiered_water_pricing_usage_l37_37257


namespace shelby_rain_time_l37_37248

noncomputable def speedNonRainy : ℚ := 30 / 60
noncomputable def speedRainy : ℚ := 20 / 60
noncomputable def totalDistance : ℚ := 16
noncomputable def totalTime : ℚ := 40

theorem shelby_rain_time : 
  ∃ x : ℚ, (speedNonRainy * (totalTime - x) + speedRainy * x = totalDistance) ∧ x = 24 := 
by
  sorry

end shelby_rain_time_l37_37248


namespace palm_trees_in_forest_l37_37923

variable (F D : ℕ)

theorem palm_trees_in_forest 
  (h1 : D = 2 * F / 5)
  (h2 : D + F = 7000) :
  F = 5000 := by
  sorry

end palm_trees_in_forest_l37_37923


namespace count_even_positive_integers_satisfy_inequality_l37_37231

open Int

noncomputable def countEvenPositiveIntegersInInterval : ℕ :=
  (List.filter (fun n : ℕ => n % 2 = 0) [2, 4, 6, 8, 10, 12]).length

theorem count_even_positive_integers_satisfy_inequality :
  countEvenPositiveIntegersInInterval = 6 := by
  sorry

end count_even_positive_integers_satisfy_inequality_l37_37231


namespace derivative_at_2_l37_37241

noncomputable def f (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)

theorem derivative_at_2 : deriv f 2 = 15 := by
  sorry

end derivative_at_2_l37_37241


namespace line_passes_through_fixed_point_l37_37933

theorem line_passes_through_fixed_point 
  (a b : ℝ) 
  (h : 2 * a + b = 1) : 
  a * 4 + b * 2 = 2 :=
sorry

end line_passes_through_fixed_point_l37_37933


namespace difference_of_squares_l37_37705

theorem difference_of_squares (a b : ℝ) : -4 * a^2 + b^2 = (b + 2 * a) * (b - 2 * a) :=
by
  sorry

end difference_of_squares_l37_37705


namespace queen_middle_school_teachers_l37_37545

theorem queen_middle_school_teachers
  (students : ℕ) 
  (classes_per_student : ℕ) 
  (classes_per_teacher : ℕ)
  (students_per_class : ℕ)
  (h_students : students = 1500)
  (h_classes_per_student : classes_per_student = 6)
  (h_classes_per_teacher : classes_per_teacher = 5)
  (h_students_per_class : students_per_class = 25) : 
  (students * classes_per_student / students_per_class) / classes_per_teacher = 72 :=
by
  sorry

end queen_middle_school_teachers_l37_37545


namespace quadratic_complete_square_r_plus_s_l37_37843

theorem quadratic_complete_square_r_plus_s :
  ∃ r s : ℚ, (∀ x : ℚ, 7 * x^2 - 21 * x - 56 = 0 → (x + r)^2 = s) ∧ r + s = 35 / 4 := sorry

end quadratic_complete_square_r_plus_s_l37_37843


namespace find_a_l37_37710

-- Given function and its condition
def f (a x : ℝ) := a * x ^ 3 + 3 * x ^ 2 + 2
def f' (a x : ℝ) := 3 * a * x ^ 2 + 6 * x

-- Condition and proof that a = -2 given the condition f'(-1) = -12
theorem find_a 
  (a : ℝ)
  (h : f' a (-1) = -12) : 
  a = -2 := 
by 
  sorry

end find_a_l37_37710


namespace negative_solution_condition_l37_37982

variable {a b c x y : ℝ}

theorem negative_solution_condition (h1 : a * x + b * y = c)
    (h2 : b * x + c * y = a)
    (h3 : c * x + a * y = b)
    (hx : x < 0)
    (hy : y < 0) :
    a + b + c = 0 :=
sorry

end negative_solution_condition_l37_37982


namespace travel_time_and_speed_l37_37236

theorem travel_time_and_speed :
  (total_time : ℝ) = 5.5 →
  (bus_whole_journey : ℝ) = 1 →
  (bus_half_journey : ℝ) = bus_whole_journey / 2 →
  (walk_half_journey : ℝ) = total_time - bus_half_journey →
  (walk_whole_journey : ℝ) = 2 * walk_half_journey →
  (bus_speed_factor : ℝ) = walk_whole_journey / bus_whole_journey →
  walk_whole_journey = 10 ∧ bus_speed_factor = 10 := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end travel_time_and_speed_l37_37236


namespace simplify_T_l37_37579

theorem simplify_T (x : ℝ) : 
  (x + 2)^6 + 6 * (x + 2)^5 + 15 * (x + 2)^4 + 20 * (x + 2)^3 + 15 * (x + 2)^2 + 6 * (x + 2) + 1 = (x + 3)^6 :=
by
  sorry

end simplify_T_l37_37579


namespace distance_between_foci_of_ellipse_l37_37913

theorem distance_between_foci_of_ellipse :
  let F1 := (4, -3)
  let F2 := (-6, 9)
  let distance := Real.sqrt ( ((4 - (-6))^2) + ((-3 - 9)^2) )
  distance = 2 * Real.sqrt 61 :=
by
  let F1 := (4, -3)
  let F2 := (-6, 9)
  let distance := Real.sqrt ( ((4 - (-6))^2) + ((-3 - 9)^2) )
  sorry

end distance_between_foci_of_ellipse_l37_37913


namespace B_days_to_complete_work_l37_37476

theorem B_days_to_complete_work 
  (W : ℝ) -- Define the amount of work
  (A_rate : ℝ := W / 15) -- A can complete the work in 15 days
  (B_days : ℝ) -- B can complete the work in B_days days
  (B_rate : ℝ := W / B_days) -- B's rate of work
  (total_days : ℝ := 12) -- Total days to complete the work
  (A_days_after_B_leaves : ℝ := 10) -- Days A works alone after B leaves
  (work_done_together : ℝ := 2 * (A_rate + B_rate)) -- Work done together in 2 days
  (work_done_by_A : ℝ := 10 * A_rate) -- Work done by A alone in 10 days
  (total_work_done : ℝ := work_done_together + work_done_by_A) -- Total work done
  (h_total_work_done : total_work_done = W) -- Total work equals W
  : B_days = 10 :=
sorry

end B_days_to_complete_work_l37_37476


namespace salt_added_correctly_l37_37213

-- Define the problem's conditions and the correct answer in Lean
variable (x : ℝ) (y : ℝ)
variable (S : ℝ := 0.2 * x) -- original salt
variable (E : ℝ := (1 / 4) * x) -- evaporated water
variable (New_volume : ℝ := x - E + 10) -- new volume after adding water

theorem salt_added_correctly :
  x = 150 → y = (1 / 3) * New_volume - S :=
by
  sorry

end salt_added_correctly_l37_37213


namespace min_width_of_garden_l37_37681

theorem min_width_of_garden (w : ℝ) (h : w*(w + 10) ≥ 150) : w ≥ 10 :=
by
  sorry

end min_width_of_garden_l37_37681


namespace cube_surface_area_l37_37839

theorem cube_surface_area (V : ℝ) (hV : V = 125) : ∃ A : ℝ, A = 25 :=
by
  sorry

end cube_surface_area_l37_37839


namespace galya_number_l37_37016

theorem galya_number (N k : ℤ) (h : (k - N + 1 = k - 7729)) : N = 7730 := 
by
  sorry

end galya_number_l37_37016


namespace rhombus_side_length_l37_37784

noncomputable def side_length_rhombus (AB BC AC : ℝ) (condition1 : AB = 12) (condition2 : BC = 12) (condition3 : AC = 6) : ℝ :=
  4

theorem rhombus_side_length (AB BC AC : ℝ) (condition1 : AB = 12) (condition2 : BC = 12) (condition3 : AC = 6) (x : ℝ) :
  side_length_rhombus AB BC AC condition1 condition2 condition3 = x ↔ x = 4 := by
  sorry

end rhombus_side_length_l37_37784


namespace product_ab_l37_37422

noncomputable def median_of_four_numbers (a b : ℕ) := 3
noncomputable def mean_of_four_numbers (a b : ℕ) := 4

theorem product_ab (a b : ℕ)
  (h1 : 1 + 2 + a + b = 4 * 4)
  (h2 : median_of_four_numbers a b = 3)
  (h3 : mean_of_four_numbers a b = 4) : (a * b = 36) :=
by sorry

end product_ab_l37_37422


namespace inequality_true_l37_37274

theorem inequality_true (a b : ℝ) (h : a > b) (x : ℝ) : 
  (a > b) → (x ≥ 0) → (a / ((2^x) + 1) > b / ((2^x) + 1)) :=
by 
  sorry

end inequality_true_l37_37274


namespace infinitely_many_arithmetic_sequences_l37_37943

theorem infinitely_many_arithmetic_sequences (x : ℕ) (hx : 0 < x) :
  ∃ y z : ℕ, y = 5 * x + 2 ∧ z = 7 * x + 3 ∧ x * (x + 1) < y * (y + 1) ∧ y * (y + 1) < z * (z + 1) ∧
  y * (y + 1) - x * (x + 1) = z * (z + 1) - y * (y + 1) :=
by
  sorry

end infinitely_many_arithmetic_sequences_l37_37943


namespace dots_not_visible_l37_37368

def total_dots_on_die : Nat := 21
def number_of_dice : Nat := 4
def total_dots : Nat := number_of_dice * total_dots_on_die
def visible_faces : List Nat := [1, 2, 2, 3, 3, 5, 6]
def sum_visible_faces : Nat := visible_faces.sum

theorem dots_not_visible : total_dots - sum_visible_faces = 62 := by
  sorry

end dots_not_visible_l37_37368


namespace lcm_12_15_18_l37_37656

theorem lcm_12_15_18 : Nat.lcm (Nat.lcm 12 15) 18 = 180 := by 
  sorry

end lcm_12_15_18_l37_37656


namespace planes_contain_at_least_three_midpoints_l37_37414

-- Define the cube structure and edge midpoints
structure Cube where
  edges : Fin 12

def midpoints (c : Cube) : Set (Fin 12) := { e | true }

-- Define the total planes considering the constraints
noncomputable def planes : ℕ := 4 + 18 + 56

-- The proof goal
theorem planes_contain_at_least_three_midpoints :
  planes = 81 := by
  sorry

end planes_contain_at_least_three_midpoints_l37_37414


namespace wider_can_radius_l37_37309

theorem wider_can_radius (h : ℝ) : 
  (∃ r : ℝ, ∀ V : ℝ, V = π * 8^2 * 2 * h → V = π * r^2 * h → r = 8 * Real.sqrt 2) :=
by 
  sorry

end wider_can_radius_l37_37309


namespace total_handshakes_l37_37759

-- Definitions based on conditions
def num_wizards : ℕ := 25
def num_elves : ℕ := 18

-- Each wizard shakes hands with every other wizard
def wizard_handshakes : ℕ := num_wizards * (num_wizards - 1) / 2

-- Each elf shakes hands with every wizard
def elf_wizard_handshakes : ℕ := num_elves * num_wizards

-- Total handshakes is the sum of the above two
theorem total_handshakes : wizard_handshakes + elf_wizard_handshakes = 750 := by
  sorry

end total_handshakes_l37_37759


namespace stripe_area_is_480pi_l37_37337

noncomputable def stripeArea (diameter : ℝ) (height : ℝ) (width : ℝ) (revolutions : ℕ) : ℝ :=
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let stripeLength := circumference * revolutions
  let area := width * stripeLength
  area

theorem stripe_area_is_480pi : stripeArea 40 90 4 3 = 480 * Real.pi :=
  by
    show stripeArea 40 90 4 3 = 480 * Real.pi
    sorry

end stripe_area_is_480pi_l37_37337


namespace triangle_area_l37_37288

theorem triangle_area (base height : ℝ) (h_base : base = 4.5) (h_height : height = 6) :
  (base * height) / 2 = 13.5 := 
by
  rw [h_base, h_height]
  norm_num

-- sorry
-- The later use of sorry statement is commented out because the proof itself has been provided in by block.

end triangle_area_l37_37288


namespace number_of_pieces_of_string_l37_37144

theorem number_of_pieces_of_string (total_length piece_length : ℝ) (h1 : total_length = 60) (h2 : piece_length = 0.6) :
    total_length / piece_length = 100 := by
  sorry

end number_of_pieces_of_string_l37_37144


namespace circle_equation_solution_l37_37229

theorem circle_equation_solution
  (a : ℝ)
  (h1 : a ^ 2 = a + 2)
  (h2 : (2 * a / (a + 2)) ^ 2 - 4 * a / (a + 2) > 0) : 
  a = -1 := 
sorry

end circle_equation_solution_l37_37229


namespace local_minimum_at_2_l37_37245

open Real

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x

def f' (x : ℝ) : ℝ := 3 * x^2 - 12

theorem local_minimum_at_2 :
  (∀ x : ℝ, -2 < x ∧ x < 2 → f' x < 0) →
  (∀ x : ℝ, x > 2 → f' x > 0) →
  (∃ ε > 0, ∀ x : ℝ, abs (x - 2) < ε → f x > f 2) :=
by
  sorry

end local_minimum_at_2_l37_37245


namespace hyperbola_focal_distance_solution_l37_37269

-- Definitions corresponding to the problem conditions
def hyperbola_equation (x y m : ℝ) :=
  x^2 / m - y^2 / 6 = 1

def focal_distance (c : ℝ) := 2 * c

-- Theorem statement to prove m = 3 based on given conditions
theorem hyperbola_focal_distance_solution (m : ℝ) (h_eq : ∀ x y : ℝ, hyperbola_equation x y m) (h_focal : focal_distance 3 = 6) :
  m = 3 :=
by {
  -- sorry is used here as a placeholder for the actual proof steps
  sorry
}

end hyperbola_focal_distance_solution_l37_37269


namespace marc_watch_days_l37_37112

theorem marc_watch_days (bought_episodes : ℕ) (watch_fraction : ℚ) (episodes_per_day : ℚ) (total_days : ℕ) : 
  bought_episodes = 50 → 
  watch_fraction = 1 / 10 → 
  episodes_per_day = (50 : ℚ) * watch_fraction → 
  total_days = (bought_episodes : ℚ) / episodes_per_day →
  total_days = 10 := 
sorry

end marc_watch_days_l37_37112


namespace length_of_ac_l37_37821

theorem length_of_ac (a b c d e : ℝ) (ab bc cd de ae ac : ℝ)
  (h1 : ab = 5)
  (h2 : bc = 2 * cd)
  (h3 : de = 8)
  (h4 : ae = 22)
  (h5 : ae = ab + bc + cd + de)
  (h6 : ac = ab + bc) :
  ac = 11 := by
  sorry

end length_of_ac_l37_37821


namespace candy_bar_multiple_l37_37667

theorem candy_bar_multiple (s m x : ℕ) (h1 : s = m * x + 6) (h2 : x = 24) (h3 : s = 78) : m = 3 :=
by
  sorry

end candy_bar_multiple_l37_37667


namespace principal_amount_is_1200_l37_37592

-- Define the given conditions
def simple_interest (P : ℝ) : ℝ := 0.10 * P
def compound_interest (P : ℝ) : ℝ := 0.1025 * P

-- Define given difference
def interest_difference (P : ℝ) : ℝ := compound_interest P - simple_interest P

-- The main goal is to prove that the principal amount P that satisfies the difference condition is 1200
theorem principal_amount_is_1200 : ∃ P : ℝ, interest_difference P = 3 ∧ P = 1200 :=
by
  sorry -- Proof to be completed

end principal_amount_is_1200_l37_37592


namespace contractor_fine_amount_l37_37402

def total_days := 30
def daily_earning := 25
def total_earnings := 360
def days_absent := 12
def days_worked := total_days - days_absent
def fine_per_absent_day (x : ℝ) : Prop :=
  (daily_earning * days_worked) - (x * days_absent) = total_earnings

theorem contractor_fine_amount : ∃ x : ℝ, fine_per_absent_day x := by
  use 7.5
  sorry

end contractor_fine_amount_l37_37402


namespace highest_price_more_than_lowest_l37_37285

-- Define the highest price and lowest price.
def highest_price : ℕ := 350
def lowest_price : ℕ := 250

-- Define the calculation for the percentage increase.
def percentage_increase (hp lp : ℕ) : ℕ :=
  ((hp - lp) * 100) / lp

-- The theorem to prove the required percentage increase.
theorem highest_price_more_than_lowest : percentage_increase highest_price lowest_price = 40 := 
  by sorry

end highest_price_more_than_lowest_l37_37285


namespace drone_altitude_l37_37524

theorem drone_altitude (h c d : ℝ) (HC HD CD : ℝ)
  (HCO_eq : h^2 + c^2 = HC^2)
  (HDO_eq : h^2 + d^2 = HD^2)
  (CD_eq : c^2 + d^2 = CD^2) 
  (HC_val : HC = 170)
  (HD_val : HD = 160)
  (CD_val : CD = 200) :
  h = 50 * Real.sqrt 29 :=
by
  sorry

end drone_altitude_l37_37524


namespace area_of_triangle_ABC_l37_37070

theorem area_of_triangle_ABC
  (A B C : ℝ)
  (a b c : ℝ)
  (sin_C_eq : Real.sin C = Real.sqrt 3 / 3)
  (sin_CBA_eq : Real.sin C + Real.sin (B - A) = Real.sin (2 * A))
  (a_minus_b_eq : a - b = 3 - Real.sqrt 6)
  (c_eq : c = Real.sqrt 3) :
  1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 2 / 2 := sorry

end area_of_triangle_ABC_l37_37070


namespace union_A_B_l37_37325

noncomputable def A : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }
noncomputable def B : Set ℝ := { x | x^3 = x }

theorem union_A_B : A ∪ B = { -1, 0, 1, 2 } := by
  sorry

end union_A_B_l37_37325


namespace area_of_triangle_l37_37115

theorem area_of_triangle (A : ℝ) (b : ℝ) (a : ℝ) (hA : A = 60) (hb : b = 4) (ha : a = 2 * Real.sqrt 3) : 
  1 / 2 * a * b * Real.sin (60 * Real.pi / 180) = 2 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l37_37115


namespace cost_per_square_inch_l37_37634

def length : ℕ := 9
def width : ℕ := 12
def total_cost : ℕ := 432

theorem cost_per_square_inch :
  total_cost / ((length * width) / 2) = 8 := 
by 
  sorry

end cost_per_square_inch_l37_37634


namespace rationalize_denominator_l37_37037

theorem rationalize_denominator (sqrt_98 : Real) (h : sqrt_98 = 7 * Real.sqrt 2) : 
  7 / sqrt_98 = Real.sqrt 2 / 2 := 
by 
  sorry

end rationalize_denominator_l37_37037


namespace minimum_height_l37_37809

theorem minimum_height (y : ℝ) (h : ℝ) (S : ℝ) (hS : S = 10 * y^2) (hS_min : S ≥ 150) (h_height : h = 2 * y) : h = 2 * Real.sqrt 15 :=
  sorry

end minimum_height_l37_37809


namespace append_five_new_number_l37_37828

theorem append_five_new_number (t u : ℕ) (h1 : t < 10) (h2 : u < 10) : 
  10 * (10 * t + u) + 5 = 100 * t + 10 * u + 5 :=
by sorry

end append_five_new_number_l37_37828


namespace enclosure_largest_side_l37_37336

theorem enclosure_largest_side (l w : ℕ) (h1 : 2 * l + 2 * w = 240) (h2 : l * w = 3600) : l = 60 :=
by
  sorry

end enclosure_largest_side_l37_37336


namespace xy_zero_l37_37004

theorem xy_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 54) : x * y = 0 :=
by
  sorry

end xy_zero_l37_37004


namespace solve_system_of_inequalities_l37_37131

open Set

theorem solve_system_of_inequalities : ∀ x : ℕ, (2 * (x - 1) < x + 3) ∧ ((2 * x + 1) / 3 > x - 1) → x ∈ ({0, 1, 2, 3} : Set ℕ) :=
by
  intro x
  intro h
  sorry

end solve_system_of_inequalities_l37_37131


namespace kanul_cash_percentage_l37_37610

theorem kanul_cash_percentage (raw_materials : ℕ) (machinery : ℕ) (total_amount : ℕ) (cash_percentage : ℕ)
  (H1 : raw_materials = 80000)
  (H2 : machinery = 30000)
  (H3 : total_amount = 137500)
  (H4 : cash_percentage = 20) :
  ((total_amount - (raw_materials + machinery)) * 100 / total_amount) = cash_percentage := by
    sorry

end kanul_cash_percentage_l37_37610


namespace train_speed_in_km_per_hr_l37_37655

def train_length : ℝ := 116.67 -- length of the train in meters
def crossing_time : ℝ := 7 -- time to cross the pole in seconds

theorem train_speed_in_km_per_hr : 
  (train_length / 1000) / (crossing_time / 3600) = 60 := by
  sorry

end train_speed_in_km_per_hr_l37_37655


namespace area_of_triangle_ABC_is_24_l37_37869

-- Define the vertices of the triangle
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (6, 1)
def C : ℝ × ℝ := (10, 6)

-- Define the area calculation
def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  0.5 * |(v.1 * w.2 - v.2 * w.1)|

theorem area_of_triangle_ABC_is_24 :
  triangleArea A B C = 24 := by
  sorry

end area_of_triangle_ABC_is_24_l37_37869


namespace xyz_value_l37_37651

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) : 
  x * y * z = 26 / 3 := 
by
  sorry

end xyz_value_l37_37651


namespace arithmetic_sequence_15th_term_l37_37824

theorem arithmetic_sequence_15th_term (a1 a2 a3 : ℕ) (d : ℕ) (n : ℕ) (h1 : a1 = 3) (h2 : a2 = 14) (h3 : a3 = 25) (h4 : d = a2 - a1) (h5 : a2 - a1 = a3 - a2) (h6 : n = 15) :
  a1 + (n - 1) * d = 157 :=
by
  -- Proof goes here
  sorry

end arithmetic_sequence_15th_term_l37_37824


namespace decrypt_nbui_is_math_l37_37352

-- Define the sets A and B as the 26 English letters
def A := {c : Char | c ≥ 'a' ∧ c ≤ 'z'}
def B := A

-- Define the mapping f from A to B
def f (c : Char) : Char :=
  if c = 'z' then 'a'
  else Char.ofNat (c.toNat + 1)

-- Define the decryption function g (it reverses the mapping f)
def g (c : Char) : Char :=
  if c = 'a' then 'z'
  else Char.ofNat (c.toNat - 1)

-- Define the decryption of the given ciphertext
def decrypt (ciphertext : String) : String :=
  ciphertext.map g

-- Prove that the decryption of "nbui" is "math"
theorem decrypt_nbui_is_math : decrypt "nbui" = "math" :=
  by
  sorry

end decrypt_nbui_is_math_l37_37352


namespace minnie_takes_more_time_l37_37381

def minnie_speed_flat : ℝ := 25
def minnie_speed_downhill : ℝ := 35
def minnie_speed_uphill : ℝ := 10
def penny_speed_flat : ℝ := 35
def penny_speed_downhill : ℝ := 45
def penny_speed_uphill : ℝ := 15

def distance_A_to_B : ℝ := 15
def distance_B_to_D : ℝ := 20
def distance_D_to_C : ℝ := 25

def distance_C_to_B : ℝ := 20
def distance_D_to_A : ℝ := 25

noncomputable def time_minnie : ℝ :=
(distance_A_to_B / minnie_speed_uphill) + 
(distance_B_to_D / minnie_speed_downhill) + 
(distance_D_to_C / minnie_speed_flat)

noncomputable def time_penny : ℝ :=
(distance_C_to_B / penny_speed_uphill) + 
(distance_B_to_D / penny_speed_downhill) + 
(distance_D_to_A / penny_speed_flat)

noncomputable def time_diff : ℝ := (time_minnie - time_penny) * 60

theorem minnie_takes_more_time : time_diff = 10 := by
  sorry

end minnie_takes_more_time_l37_37381


namespace solve_X_l37_37988

def diamond (X Y : ℝ) : ℝ := 4 * X - 3 * Y + 2

theorem solve_X :
  (∃ X : ℝ, diamond X 6 = 35) ↔ (X = 51 / 4) := by
  sorry

end solve_X_l37_37988


namespace quadratic_roots_l37_37632

theorem quadratic_roots (x : ℝ) : (x^2 + 4*x + 3 = 0) ↔ (x = -3 ∨ x = -1) := 
sorry

end quadratic_roots_l37_37632


namespace find_first_number_l37_37391

-- Definitions from conditions
variable (x : ℕ) -- Let the first number be x
variable (y : ℕ) -- Let the second number be y

-- Given conditions in the problem
def condition1 : Prop := y = 43
def condition2 : Prop := x + 2 * y = 124

-- The proof target
theorem find_first_number (h1 : condition1 y) (h2 : condition2 x y) : x = 38 := by
  sorry

end find_first_number_l37_37391


namespace sum_of_numbers_l37_37878

theorem sum_of_numbers (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 241)
  (h2 : ab + bc + ca = 100) :
  a + b + c = 21 :=
sorry

end sum_of_numbers_l37_37878


namespace arithmetic_sequence_max_sum_l37_37395

theorem arithmetic_sequence_max_sum (a : ℕ → ℝ) (d : ℝ) (m : ℕ) (S : ℕ → ℝ):
  (∀ n, a n = a 1 + (n - 1) * d) → 
  3 * a 8 = 5 * a m → 
  a 1 > 0 →
  (∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * d)) →
  (∀ n, S n ≤ S 20) →
  m = 13 := 
by {
  -- State the corresponding solution steps leading to the proof.
  sorry
}

end arithmetic_sequence_max_sum_l37_37395


namespace line_AB_bisects_segment_DE_l37_37382

variables {A B C D E : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
  {trapezoid : A × B × C × D} (AC CD : Prop) (BD_sym : Prop) (intersect_E : Prop)
  (line_AB : Prop) (bisects_DE : Prop)

-- Given a trapezoid ABCD
def is_trapezoid (A B C D : Type) : Prop := sorry

-- Given the diagonal AC is equal to the side CD
def diagonal_eq_leg (AC CD : Prop) : Prop := sorry

-- Given line BD is symmetric with respect to AD intersects AC at point E
def symmetric_line_intersect (BD_sym AD AC E : Prop) : Prop := sorry

-- Prove that line AB bisects segment DE
theorem line_AB_bisects_segment_DE
  (h_trapezoid : is_trapezoid A B C D)
  (h_diagonal_eq_leg : diagonal_eq_leg AC CD)
  (h_symmetric_line_intersect : symmetric_line_intersect BD_sym (sorry : Prop) AC intersect_E)
  (h_line_AB : line_AB) :
  bisects_DE := sorry

end line_AB_bisects_segment_DE_l37_37382


namespace cricketer_average_after_19_innings_l37_37385

theorem cricketer_average_after_19_innings
  (runs_19th_inning : ℕ)
  (increase_in_average : ℤ)
  (initial_average : ℤ)
  (new_average : ℤ)
  (h1 : runs_19th_inning = 95)
  (h2 : increase_in_average = 4)
  (eq1 : 18 * initial_average + 95 = 19 * (initial_average + increase_in_average))
  (eq2 : new_average = initial_average + increase_in_average) :
  new_average = 23 :=
by sorry

end cricketer_average_after_19_innings_l37_37385


namespace cos_inequality_range_l37_37080

theorem cos_inequality_range (x : ℝ) (h₁ : 0 ≤ x) (h₂ : x ≤ 2 * Real.pi) (h₃ : Real.cos x ≤ 1 / 2) :
  x ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 3) := 
sorry

end cos_inequality_range_l37_37080


namespace find_f_expression_l37_37750

theorem find_f_expression (f : ℝ → ℝ) (x : ℝ) (h : f (Real.log x) = 3 * x + 4) : 
  f x = 3 * Real.exp x + 4 := 
by
  sorry

end find_f_expression_l37_37750


namespace lychee_ratio_l37_37513

theorem lychee_ratio (total_lychees : ℕ) (sold_lychees : ℕ) (remaining_home : ℕ) (remaining_after_eat : ℕ) 
    (h1: total_lychees = 500) 
    (h2: sold_lychees = total_lychees / 2) 
    (h3: remaining_home = total_lychees - sold_lychees) 
    (h4: remaining_after_eat = 100)
    (h5: remaining_after_eat + (remaining_home - remaining_after_eat) = remaining_home) : 
    (remaining_home - remaining_after_eat) / remaining_home = 3 / 5 :=
by
    -- Proof is omitted
    sorry

end lychee_ratio_l37_37513


namespace regular_triangular_pyramid_volume_l37_37804

noncomputable def pyramid_volume (a h γ : ℝ) : ℝ :=
  (Real.sqrt 3 * a^2 * h) / 12

theorem regular_triangular_pyramid_volume
  (a h γ : ℝ) (h_nonneg : 0 ≤ h) (γ_nonneg : 0 ≤ γ) :
  pyramid_volume a h γ = (Real.sqrt 3 * a^2 * h) / 12 :=
by
  sorry

end regular_triangular_pyramid_volume_l37_37804


namespace range_of_a_l37_37181

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^2 + a * x + 2)
  (h2 : ∀ y, (∃ x, y = f (f x)) ↔ (∃ x, y = f x)) : a ≥ 4 ∨ a ≤ -2 := 
sorry

end range_of_a_l37_37181


namespace charity_event_assignment_l37_37505

theorem charity_event_assignment (students : Finset ℕ) (h_students : students.card = 5) :
  ∃ (num_ways : ℕ), num_ways = 60 :=
by
  let select_two_for_friday := Nat.choose 5 2
  let remaining_students_after_friday := 5 - 2
  let select_one_for_saturday := Nat.choose remaining_students_after_friday 1
  let remaining_students_after_saturday := remaining_students_after_friday - 1
  let select_one_for_sunday := Nat.choose remaining_students_after_saturday 1
  let total_ways := select_two_for_friday * select_one_for_saturday * select_one_for_sunday
  use total_ways
  sorry

end charity_event_assignment_l37_37505


namespace equal_values_of_means_l37_37601

theorem equal_values_of_means (f : ℤ × ℤ → ℤ) 
  (h_pos : ∀ p, 0 < f p)
  (h_mean : ∀ p, f p = (f (p.1 + 1, p.2) + f (p.1 - 1, p.2) + f (p.1, p.2 + 1) + f (p.1, p.2 - 1)) / 4):
  ∃ m : ℤ, ∀ p, f p = m := sorry

end equal_values_of_means_l37_37601


namespace solution_set_x_l37_37972

theorem solution_set_x (x : ℝ) : 
  (|x^2 - x - 2| + |1 / x| = |x^2 - x - 2 + 1 / x|) ↔ 
  (x ∈ {y : ℝ | -1 ≤ y ∧ y < 0} ∨ x ≥ 2) :=
sorry

end solution_set_x_l37_37972


namespace speed_of_man_l37_37647

noncomputable def train_length : ℝ := 150
noncomputable def time_to_pass : ℝ := 6
noncomputable def train_speed_kmh : ℝ := 83.99280057595394

/-- The speed of the man in km/h -/
theorem speed_of_man (train_length time_to_pass train_speed_kmh : ℝ) (h_train_length : train_length = 150) (h_time_to_pass : time_to_pass = 6) (h_train_speed_kmh : train_speed_kmh = 83.99280057595394) : 
  (train_length / time_to_pass * 3600 / 1000 - train_speed_kmh) * 3600 / 1000 = 6.0072 :=
by
  sorry

end speed_of_man_l37_37647


namespace calculate_base_length_l37_37369

variable (A b h : ℝ)

def is_parallelogram_base_length (A : ℝ) (b : ℝ) (h : ℝ) : Prop :=
  (A = b * h) ∧ (h = 2 * b)

theorem calculate_base_length (H : is_parallelogram_base_length A b h) : b = 15 := by
  -- H gives us the hypothesis that (A = b * h) and (h = 2 * b)
  have H1 : A = b * h := H.1
  have H2 : h = 2 * b := H.2
  -- Use substitution and algebra to solve for b
  sorry

end calculate_base_length_l37_37369


namespace farmer_land_l37_37086

variable (A C G P T : ℝ)
variable (h1 : C = 0.90 * A)
variable (h2 : G = 0.10 * C)
variable (h3 : P = 0.80 * C)
variable (h4 : T = 450)
variable (h5 : C = G + P + T)

theorem farmer_land (A : ℝ) (h1 : C = 0.90 * A) (h2 : G = 0.10 * C) (h3 : P = 0.80 * C) (h4 : T = 450) (h5 : C = G + P + T) : A = 5000 := by
  sorry

end farmer_land_l37_37086


namespace problem_statement_l37_37079

noncomputable def h (y : ℂ) : ℂ := y^5 - y^3 + 1
noncomputable def p (y : ℂ) : ℂ := y^2 - 3

theorem problem_statement (y_1 y_2 y_3 y_4 y_5 : ℂ) (hroots : ∀ y, h y = 0 ↔ y = y_1 ∨ y = y_2 ∨ y = y_3 ∨ y = y_4 ∨ y = y_5) :
  (p y_1) * (p y_2) * (p y_3) * (p y_4) * (p y_5) = 22 :=
by
  sorry

end problem_statement_l37_37079


namespace original_number_is_repeating_decimal_l37_37593

theorem original_number_is_repeating_decimal :
  ∃ N : ℚ, (N * 10 ^ 28) % 10^30 = 15 ∧ N * 5 = 0.7894736842105263 ∧ 
  (N = 3 / 19) :=
sorry

end original_number_is_repeating_decimal_l37_37593


namespace smallest_positive_integer_ends_in_3_divisible_by_11_l37_37927

theorem smallest_positive_integer_ends_in_3_divisible_by_11 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 11 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 10 = 3 ∧ m % 11 = 0) → n ≤ m :=
sorry

end smallest_positive_integer_ends_in_3_divisible_by_11_l37_37927


namespace jungkook_mother_age_four_times_jungkook_age_l37_37202

-- Definitions of conditions
def jungkoo_age : ℕ := 16
def mother_age : ℕ := 46

-- Theorem statement for the problem
theorem jungkook_mother_age_four_times_jungkook_age :
  ∃ (x : ℕ), (mother_age - x = 4 * (jungkoo_age - x)) ∧ x = 6 :=
by
  sorry

end jungkook_mother_age_four_times_jungkook_age_l37_37202


namespace x_can_be_positive_negative_or_zero_l37_37599

noncomputable
def characteristics_of_x (x y z w : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : w ≠ 0) 
  (h5 : (x / y) > (z / w)) (h6 : (y * w) > 0) : Prop :=
  ∃ r : ℝ, r = x

theorem x_can_be_positive_negative_or_zero (x y z w : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (h4 : w ≠ 0) (h5 : (x / y) > (z / w)) (h6 : (y * w) > 0) : 
  (characteristics_of_x x y z w h1 h2 h3 h4 h5 h6) :=
sorry

end x_can_be_positive_negative_or_zero_l37_37599


namespace least_number_of_candles_l37_37622

theorem least_number_of_candles (b : ℕ) :
  (b ≡ 5 [MOD 6]) ∧ (b ≡ 7 [MOD 8]) ∧ (b ≡ 3 [MOD 9]) → b = 119 :=
by
  -- Proof omitted
  sorry

end least_number_of_candles_l37_37622


namespace max_value_of_expression_l37_37707

theorem max_value_of_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 1) ≤ Real.sqrt 29 := 
sorry

end max_value_of_expression_l37_37707


namespace tangent_line_eq_l37_37573

theorem tangent_line_eq (x y : ℝ) (h : y = 2 * x^2 + 1) : 
  (x = -1 ∧ y = 3) → (4 * x + y + 1 = 0) :=
by
  intros
  sorry

end tangent_line_eq_l37_37573


namespace cos_angle_relation_l37_37192

theorem cos_angle_relation (α : ℝ) (h : Real.sin (α + π / 6) = 1 / 3) : Real.cos (2 * α - 2 * π / 3) = -7 / 9 := by 
  sorry

end cos_angle_relation_l37_37192


namespace f_neg_two_l37_37683

def f (a b : ℝ) (x : ℝ) :=
  -a * x^5 - x^3 + b * x - 7

theorem f_neg_two (a b : ℝ) (h : f a b 2 = -9) : f a b (-2) = -5 :=
by sorry

end f_neg_two_l37_37683


namespace find_k_value_l37_37324

theorem find_k_value (a : ℕ → ℕ) (k : ℕ) (S : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 3 = 5) 
  (h₃ : S (k + 2) - S k = 36) : 
  k = 8 := 
by 
  sorry

end find_k_value_l37_37324


namespace perimeter_regular_polygon_l37_37371

-- Condition definitions
def is_regular_polygon (n : ℕ) (s : ℝ) : Prop := 
  n * s > 0

def exterior_angle (E : ℝ) (n : ℕ) : Prop := 
  E = 360 / n

def side_length (s : ℝ) : Prop :=
  s = 6

-- Theorem statement to prove the perimeter is 24 units
theorem perimeter_regular_polygon 
  (n : ℕ) (s E : ℝ)
  (h1 : is_regular_polygon n s)
  (h2 : exterior_angle E n)
  (h3 : side_length s)
  (h4 : E = 90) :
  4 * s = 24 :=
by
  sorry

end perimeter_regular_polygon_l37_37371


namespace coefficient_of_1_div_x_l37_37092

open Nat

noncomputable def binomial_expansion (x : ℝ) (n : ℕ) : ℝ :=
  (1 / Real.sqrt x - 3)^n

theorem coefficient_of_1_div_x (x : ℝ) (n : ℕ) (h1 : n ∈ {m | m > 0}) (h2 : binomial_expansion x n = 16) :
  ∃ c : ℝ, c = 54 :=
by
  sorry

end coefficient_of_1_div_x_l37_37092


namespace walk_to_lake_park_restaurant_is_zero_l37_37842

noncomputable def time_to_hidden_lake : ℕ := 15
noncomputable def time_to_return_from_hidden_lake : ℕ := 7
noncomputable def total_walk_time_dante : ℕ := 22

theorem walk_to_lake_park_restaurant_is_zero :
  ∃ (x : ℕ), (2 * x + time_to_hidden_lake + time_to_return_from_hidden_lake = total_walk_time_dante) → x = 0 :=
by
  use 0
  intros
  sorry

end walk_to_lake_park_restaurant_is_zero_l37_37842


namespace unique_common_element_l37_37148

variable (A B : Set ℝ)
variable (a : ℝ)

theorem unique_common_element :
  A = {1, 3, a} → 
  B = {4, 5} →
  A ∩ B = {4} →
  a = 4 := 
by
  intro hA hB hAB
  sorry

end unique_common_element_l37_37148


namespace UVWXY_perimeter_l37_37879

theorem UVWXY_perimeter (U V W X Y Z : ℝ) 
  (hUV : UV = 5)
  (hVW : VW = 3)
  (hWY : WY = 5)
  (hYX : YX = 9)
  (hXU : XU = 7) :
  UV + VW + WY + YX + XU = 29 :=
by
  sorry

end UVWXY_perimeter_l37_37879


namespace area_triangle_MNR_l37_37006

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

/-- Given the quadrilateral PQRS with the midpoints M and N of PQ and QR 
and specified lengths, prove the calculated area of triangle MNR. -/
theorem area_triangle_MNR : 
  let P : (ℝ × ℝ) := (0, 5)
  let Q : (ℝ × ℝ) := (10, 5)
  let R : (ℝ × ℝ) := (14, 0)
  let S : (ℝ × ℝ) := (7, 0)
  let M : (ℝ × ℝ) := (5, 5)  -- Midpoint of PQ
  let N : (ℝ × ℝ) := (12, 2.5) -- Midpoint of QR
  distance M.fst M.snd N.fst N.snd = 7.435 →
  ((5 - 0 : ℝ) / 2 = 2.5) →
  (1 / 2 * 7.435 * 2.5) = 9.294375 :=
by
  sorry

end area_triangle_MNR_l37_37006


namespace shuttle_speed_l37_37421

theorem shuttle_speed (speed_kps : ℕ) (conversion_factor : ℕ) (speed_kph : ℕ) :
  speed_kps = 2 → conversion_factor = 3600 → speed_kph = speed_kps * conversion_factor → speed_kph = 7200 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end shuttle_speed_l37_37421


namespace max_ab_l37_37162

theorem max_ab (a b c : ℝ) (h1 : 0 < a ∧ a < 1) (h2 : 0 < b ∧ b < 1) (h3 : 0 < c ∧ c < 1) (h4 : 3 * a + 2 * b = 2) :
  ab ≤ 1 / 6 :=
sorry

end max_ab_l37_37162


namespace percent_of_x_l37_37159

-- The mathematical equivalent of the problem statement in Lean.
theorem percent_of_x (x : ℝ) (hx : 0 < x) : (x / 10 + x / 25) = 0.14 * x :=
by
  sorry

end percent_of_x_l37_37159


namespace hiking_supplies_l37_37512

theorem hiking_supplies (hours_per_day : ℕ) (days : ℕ) (rate_mph : ℝ) 
    (supply_per_mile : ℝ) (resupply_rate : ℝ)
    (initial_pack_weight : ℝ) : 
    hours_per_day = 8 → days = 5 → rate_mph = 2.5 → 
    supply_per_mile = 0.5 → resupply_rate = 0.25 → 
    initial_pack_weight = (40 : ℝ) :=
by
  intros hpd hd rm spm rr
  sorry

end hiking_supplies_l37_37512


namespace louie_pie_share_l37_37731

theorem louie_pie_share :
  let leftover := (6 : ℝ) / 7
  let people := 3
  leftover / people = (2 : ℝ) / 7 := 
by
  sorry

end louie_pie_share_l37_37731


namespace solve_equation_l37_37327

theorem solve_equation (x : ℝ) (h1 : x ≠ 2 / 3) :
  (7 * x + 3) / (3 * x ^ 2 + 7 * x - 6) = (3 * x) / (3 * x - 2) ↔
  x = (-1 + Real.sqrt 10) / 3 ∨ x = (-1 - Real.sqrt 10) / 3 :=
by sorry

end solve_equation_l37_37327


namespace sandwiches_prepared_l37_37882

-- Define the conditions as given in the problem.
def ruth_ate_sandwiches : ℕ := 1
def brother_ate_sandwiches : ℕ := 2
def first_cousin_ate_sandwiches : ℕ := 2
def each_other_cousin_ate_sandwiches : ℕ := 1
def number_of_other_cousins : ℕ := 2
def sandwiches_left : ℕ := 3

-- Define the total number of sandwiches eaten.
def total_sandwiches_eaten : ℕ := ruth_ate_sandwiches 
                                  + brother_ate_sandwiches
                                  + first_cousin_ate_sandwiches 
                                  + (each_other_cousin_ate_sandwiches * number_of_other_cousins)

-- Define the number of sandwiches prepared by Ruth.
def sandwiches_prepared_by_ruth : ℕ := total_sandwiches_eaten + sandwiches_left

-- Formulate the theorem to prove.
theorem sandwiches_prepared : sandwiches_prepared_by_ruth = 10 :=
by
  -- Use the solution steps to prove the theorem (proof omitted here).
  sorry

end sandwiches_prepared_l37_37882


namespace arithmetic_sequence_general_formula_l37_37701

theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (h₁ : a 1 = 39) (h₂ : a 1 + a 3 = 74) : 
  ∀ n, a n = 41 - 2 * n :=
sorry

end arithmetic_sequence_general_formula_l37_37701


namespace equidistant_points_quadrants_l37_37577

theorem equidistant_points_quadrants (x y : ℝ)
  (h_line : 4 * x + 7 * y = 28)
  (h_equidistant : abs x = abs y) :
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end equidistant_points_quadrants_l37_37577


namespace final_cost_is_30_l37_37991

-- Define conditions as constants
def cost_of_repair : ℝ := 7
def sales_tax : ℝ := 0.50
def number_of_tires : ℕ := 4

-- Define the cost for one tire repair
def cost_one_tire : ℝ := cost_of_repair + sales_tax

-- Define the cost for all tires
def total_cost : ℝ := cost_one_tire * number_of_tires

-- Theorem stating that the total cost is $30
theorem final_cost_is_30 : total_cost = 30 :=
by
  sorry

end final_cost_is_30_l37_37991


namespace quadratic_inequality_solution_l37_37863

theorem quadratic_inequality_solution (y : ℝ) : 
  (y^2 - 9 * y + 14 ≤ 0) ↔ (2 ≤ y ∧ y ≤ 7) :=
sorry

end quadratic_inequality_solution_l37_37863


namespace find_amplitude_l37_37290

noncomputable def amplitude (a b c d x : ℝ) := a * Real.sin (b * x + c) + d

theorem find_amplitude (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h_range : ∀ x, -1 ≤ amplitude a b c d x ∧ amplitude a b c d x ≤ 7) :
  a = 4 :=
by
  sorry

end find_amplitude_l37_37290


namespace min_value_frac_l37_37945

theorem min_value_frac (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 2 * m + n = 1) : 
  (1 / m + 2 / n) = 8 :=
sorry

end min_value_frac_l37_37945


namespace hyperbola_condition_l37_37427

theorem hyperbola_condition (k : ℝ) : (k > 1) -> ( ∀ x y : ℝ, (k - 1) * (k + 1) > 0 ↔ ( ∃ x y : ℝ, (k > 1) ∧ ((x * x) / (k - 1) - (y * y) / (k + 1)) = 1)) :=
sorry

end hyperbola_condition_l37_37427


namespace julia_total_spend_l37_37339

noncomputable def total_cost_julia_puppy : ℝ :=
  let adoption_fee := 20.00
  let dog_food := 20.00
  let treat_cost := 2.50
  let treat_count := 2
  let treats := treat_cost * treat_count
  let toys := 15.00
  let crate := 20.00
  let bed := 20.00
  let collar_leash := 15.00
  let total_supplies := dog_food + treats + toys + crate + bed + collar_leash
  let discount := 0.20 * total_supplies
  let final_supplies := total_supplies - discount
  final_supplies + adoption_fee

theorem julia_total_spend : total_cost_julia_puppy = 96.00 :=
by
  sorry

end julia_total_spend_l37_37339


namespace geometric_sequence_sum_l37_37039

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) 
  (q : ℝ)
  (h₀ : q > 1)
  (h₁ : ∀ n : ℕ, a (n + 1) = a n * q)
  (h₂ : ∀ x : ℝ, 4 * x^2 - 8 * x + 3 = 0 → (x = a 2005 ∨ x = a 2006)) : 
  a 2007 + a 2008 = 18 := 
sorry

end geometric_sequence_sum_l37_37039


namespace red_grapes_more_than_three_times_green_l37_37375

-- Definitions from conditions
variables (G R B : ℕ)
def condition1 := R = 3 * G + (R - 3 * G)
def condition2 := B = G - 5
def condition3 := R + G + B = 102
def condition4 := R = 67

-- The proof problem
theorem red_grapes_more_than_three_times_green : (R = 67) ∧ (R + G + (G - 5) = 102) ∧ (R = 3 * G + (R - 3 * G)) → R - 3 * G = 7 :=
by sorry

end red_grapes_more_than_three_times_green_l37_37375


namespace child_wants_to_buy_3_toys_l37_37493

/- 
  Problem Statement:
  There are 10 toys, and the number of ways to select a certain number 
  of those toys in any order is 120. We need to find out how many toys 
  were selected.
-/

def comb (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem child_wants_to_buy_3_toys :
  ∃ r : ℕ, r ≤ 10 ∧ comb 10 r = 120 :=
by
  use 3
  -- Here you would write the proof
  sorry

end child_wants_to_buy_3_toys_l37_37493


namespace smallest_number_is_10_l37_37712

/-- Define the set of numbers. -/
def numbers : List Int := [10, 11, 12, 13, 14]

theorem smallest_number_is_10 :
  ∃ n ∈ numbers, (∀ m ∈ numbers, n ≤ m) ∧ n = 10 :=
by
  sorry

end smallest_number_is_10_l37_37712


namespace complete_square_solution_l37_37483

theorem complete_square_solution (x : ℝ) :
  (x^2 + 6 * x - 4 = 0) → ((x + 3)^2 = 13) :=
by
  sorry

end complete_square_solution_l37_37483


namespace area_difference_l37_37850

-- Setting up the relevant conditions and entities
def side_red := 8
def length_yellow := 10
def width_yellow := 5

-- Definition of areas
def area_red := side_red * side_red
def area_yellow := length_yellow * width_yellow

-- The theorem we need to prove
theorem area_difference :
  area_red - area_yellow = 14 :=
by
  -- We skip the proof here due to the instruction
  sorry

end area_difference_l37_37850


namespace base8_digits_sum_l37_37007

-- Define digits and their restrictions
variables {A B C : ℕ}

-- Main theorem
theorem base8_digits_sum (h1 : 0 < A ∧ A < 8)
                         (h2 : 0 < B ∧ B < 8)
                         (h3 : 0 < C ∧ C < 8)
                         (distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
                         (condition : (8^2 * A + 8 * B + C) + (8^2 * B + 8 * C + A) + (8^2 * C + 8 * A + B) = (8^2 + 8 + 1) * 8 * A) :
  A + B + C = 8 := 
sorry

end base8_digits_sum_l37_37007


namespace transistors_in_2002_transistors_in_2010_l37_37508

-- Definitions based on the conditions
def mooresLawDoubling (initial_transistors : ℕ) (years : ℕ) : ℕ :=
  initial_transistors * 2^(years / 2)

-- Conditions
def initial_transistors := 2000000
def year_1992 := 1992
def year_2002 := 2002
def year_2010 := 2010

-- Questions translated into proof targets
theorem transistors_in_2002 : mooresLawDoubling initial_transistors (year_2002 - year_1992) = 64000000 := by
  sorry

theorem transistors_in_2010 : mooresLawDoubling (mooresLawDoubling initial_transistors (year_2002 - year_1992)) (year_2010 - year_2002) = 1024000000 := by
  sorry

end transistors_in_2002_transistors_in_2010_l37_37508


namespace find_first_offset_l37_37230

variable (d y A x : ℝ)

theorem find_first_offset (h_d : d = 40) (h_y : y = 6) (h_A : A = 300) :
    x = 9 :=
by
  sorry

end find_first_offset_l37_37230


namespace cubic_expression_l37_37798

theorem cubic_expression (a b c : ℝ) (h1 : a + b + c = 15) (h2 : ab + ac + bc = 50) : 
  a^3 + b^3 + c^3 - 3 * a * b * c = 1125 :=
sorry

end cubic_expression_l37_37798


namespace point_inside_circle_range_of_a_l37_37893

/- 
  Define the circle and the point P. 
  We would show that ensuring the point lies inside the circle implies |a| < 1/13.
-/

theorem point_inside_circle_range_of_a (a : ℝ) : 
  ((5 * a + 1 - 1) ^ 2 + (12 * a) ^ 2 < 1) -> |a| < 1 / 13 := 
by 
  sorry

end point_inside_circle_range_of_a_l37_37893


namespace sum_gcd_lcm_l37_37297

theorem sum_gcd_lcm (a₁ a₂ : ℕ) (h₁ : a₁ = 36) (h₂ : a₂ = 495) :
  Nat.gcd a₁ a₂ + Nat.lcm a₁ a₂ = 1989 :=
by
  -- Proof can be added here
  sorry

end sum_gcd_lcm_l37_37297


namespace digit_divisible_by_3_l37_37412

theorem digit_divisible_by_3 (d : ℕ) (h : d < 10) : (15780 + d) % 3 = 0 ↔ d = 0 ∨ d = 3 ∨ d = 6 ∨ d = 9 := by
  sorry

end digit_divisible_by_3_l37_37412


namespace hyperbola_equations_l37_37499

def eq1 (x y : ℝ) : Prop := x^2 - 4 * y^2 = (5 + Real.sqrt 6)^2
def eq2 (x y : ℝ) : Prop := 4 * y^2 - x^2 = 4

theorem hyperbola_equations 
  (x y : ℝ)
  (hx1 : x - 2 * y = 0)
  (hx2 : x + 2 * y = 0)
  (dist : Real.sqrt ((x - 5)^2 + y^2) = Real.sqrt 6) :
  eq1 x y ∧ eq2 x y := 
sorry

end hyperbola_equations_l37_37499


namespace problem_solution_l37_37293

open Set

theorem problem_solution (x : ℝ) :
  (x ∈ {y : ℝ | (2 / (y + 2) + 4 / (y + 8) ≥ 1)} ↔ x ∈ Ioo (-8 : ℝ) (-2 : ℝ)) :=
sorry

end problem_solution_l37_37293


namespace measure_of_acute_angle_l37_37639

theorem measure_of_acute_angle (x : ℝ) (h_complement : 90 - x = (1/2) * (180 - x) + 20) (h_acute : 0 < x ∧ x < 90) : x = 40 :=
  sorry

end measure_of_acute_angle_l37_37639


namespace calc_x_l37_37956

theorem calc_x : 484 + 2 * 22 * 7 + 49 = 841 := by
  sorry

end calc_x_l37_37956


namespace value_of_b_pos_sum_for_all_x_l37_37445

noncomputable def f (b : ℝ) (x : ℝ) := 3 * x^2 - 2 * x + b
noncomputable def g (b : ℝ) (x : ℝ) := x^2 + b * x - 1
noncomputable def sum_f_g (b : ℝ) (x : ℝ) := f b x + g b x

theorem value_of_b (b : ℝ) (h : ∀ x : ℝ, (sum_f_g b x = 4 * x^2 + (b - 2) * x + (b - 1))) :
  b = 2 := 
sorry

theorem pos_sum_for_all_x :
  ∀ x : ℝ, 4 * x^2 + 1 > 0 := 
sorry

end value_of_b_pos_sum_for_all_x_l37_37445


namespace spending_difference_l37_37098

-- Define the conditions
def spent_on_chocolate : ℤ := 7
def spent_on_candy_bar : ℤ := 2

-- The theorem to be proven
theorem spending_difference : (spent_on_chocolate - spent_on_candy_bar = 5) :=
by sorry

end spending_difference_l37_37098


namespace percentage_less_than_l37_37691

variable (x y z n : ℝ)
variable (hx : x = 8 * y)
variable (hy : y = 2 * |z - n|)
variable (hz : z = 1.1 * n)

theorem percentage_less_than (hx : x = 8 * y) (hy : y = 2 * |z - n|) (hz : z = 1.1 * n) :
  ((x - y) / x) * 100 = 87.5 := sorry

end percentage_less_than_l37_37691


namespace range_of_m_l37_37511

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x - 3| ≤ 2 → 1 ≤ x ∧ x ≤ 5) → 
  (∀ x : ℝ, (x - m + 1) * (x - m - 1) ≤ 0 → m - 1 ≤ x ∧ x ≤ m + 1) → 
  (∀ x : ℝ, x < 1 ∨ x > 5 → x < m - 1 ∨ x > m + 1) → 
  2 ≤ m ∧ m ≤ 4 := 
by
  sorry

end range_of_m_l37_37511


namespace system_has_three_real_k_with_unique_solution_l37_37072

theorem system_has_three_real_k_with_unique_solution :
  (∃ (k : ℝ), ∀ (x y : ℝ), (x^2 + y^2 = 2 * k^2 ∧ k * x - y = 2 * k) → (x, y) = (0, 0)) → 
  ∃ (k : ℝ), ∃ (x y : ℝ), (x^2 + y^2 = 2 * k^2 ∧ k * x - y = 2 * k) :=
by
  sorry

end system_has_three_real_k_with_unique_solution_l37_37072


namespace distinct_prime_factors_2310_l37_37090

theorem distinct_prime_factors_2310 : 
  ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ (S.card = 5) ∧ (S.prod id = 2310) := by
  sorry

end distinct_prime_factors_2310_l37_37090


namespace am_gm_inequality_l37_37939

theorem am_gm_inequality
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by
  sorry

end am_gm_inequality_l37_37939


namespace RelativelyPrimeProbability_l37_37847

def relatively_prime_probability_42 : Rat :=
  let n := 42
  let total := n
  let rel_prime_count := total - (21 + 14 + 6 - 7 - 3 - 2 + 1)
  let probability := (rel_prime_count : Rat) / total
  probability

theorem RelativelyPrimeProbability : relatively_prime_probability_42 = 2 / 7 :=
sorry

end RelativelyPrimeProbability_l37_37847


namespace sequence_formula_and_sum_l37_37423

def arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a : ℕ → ℝ) :=
  ∀ m n k, m < n → n < k → a n^2 = a m * a k

def Sn (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

theorem sequence_formula_and_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (arithmetic_geometric_sequence a 0 ∧ a 1 = 2 ∧ geometric_sequence a → ∀ n, a n = 2) ∧
  (arithmetic_geometric_sequence a 4 ∧ a 1 = 2 ∧ geometric_sequence a → ∀ n, a n = 4 * n - 2) ∧
  (arithmetic_geometric_sequence a 4 ∧ a 1 = 2 ∧ (∀ n, S n = (n * (4 * n)) / 2) → ∃ n > 0, S n > 60 * n + 800 ∧ n = 41) ∧
  (arithmetic_geometric_sequence a 0 ∧ a 1 = 2 ∧ (∀ n, S n = 2 * n) → ∀ n > 0, ¬ (S n > 60 * n + 800)) :=
by sorry

end sequence_formula_and_sum_l37_37423


namespace rollo_guinea_pigs_food_l37_37899

theorem rollo_guinea_pigs_food :
  let first_food := 2
  let second_food := 2 * first_food
  let third_food := second_food + 3
  first_food + second_food + third_food = 13 :=
by
  sorry

end rollo_guinea_pigs_food_l37_37899


namespace value_of_g_at_five_l37_37898

def g (x : ℕ) : ℕ := x^2 - 2 * x

theorem value_of_g_at_five : g 5 = 15 := by
  sorry

end value_of_g_at_five_l37_37898


namespace S7_value_l37_37785

def arithmetic_seq_sum (n : ℕ) (a_1 d : ℚ) : ℚ :=
  n * a_1 + (n * (n - 1) / 2) * d

def a_n (n : ℕ) (a_1 d : ℚ) : ℚ :=
  a_1 + (n - 1) * d

theorem S7_value (a_1 d : ℚ) (S_n : ℕ → ℚ)
  (hSn_def : ∀ n, S_n n = arithmetic_seq_sum n a_1 d)
  (h_sum_condition : S_n 7 + S_n 5 = 10)
  (h_a3_condition : a_n 3 a_1 d = 5) :
  S_n 7 = -15 :=
by
  sorry

end S7_value_l37_37785


namespace umbrellas_problem_l37_37033

theorem umbrellas_problem :
  ∃ (b r : ℕ), b = 36 ∧ r = 27 ∧ 
  b = (45 + r) / 2 ∧ 
  r = (45 + b) / 3 :=
by sorry

end umbrellas_problem_l37_37033


namespace jerry_age_l37_37514

theorem jerry_age (M J : ℕ) (h1 : M = 4 * J + 10) (h2 : M = 30) : J = 5 := by
  sorry

end jerry_age_l37_37514


namespace find_integer_pairs_l37_37209

theorem find_integer_pairs :
  ∃ (x y : ℤ), (x = 30 ∧ y = 21) ∨ (x = -21 ∧ y = -30) ∧ (x^2 + y^2 + 27 = 456 * Int.sqrt (x - y)) :=
by
  sorry

end find_integer_pairs_l37_37209


namespace octagon_area_difference_is_512_l37_37214

noncomputable def octagon_area_difference (side_length : ℝ) : ℝ :=
  let initial_octagon_area := 2 * (1 + Real.sqrt 2) * side_length^2
  let triangle_area := (1 / 2) * side_length^2
  let total_triangle_area := 8 * triangle_area
  let inner_octagon_area := initial_octagon_area - total_triangle_area
  initial_octagon_area - inner_octagon_area

theorem octagon_area_difference_is_512 :
  octagon_area_difference 16 = 512 :=
by
  -- This is where the proof would be filled in.
  sorry

end octagon_area_difference_is_512_l37_37214


namespace best_discount_sequence_l37_37689

/-- 
The initial price of the book is 30.
Stay focused on two sequences of discounts.
Sequence 1: $5 off, then 10% off, then $2 off if applicable.
Sequence 2: 10% off, then $5 off, then $2 off if applicable.
Compare the final prices obtained from applying these sequences.
-/
noncomputable def initial_price : ℝ := 30
noncomputable def five_off (price : ℝ) : ℝ := price - 5
noncomputable def ten_percent_off (price : ℝ) : ℝ := 0.9 * price
noncomputable def additional_two_off_if_applicable (price : ℝ) : ℝ := 
  if price > 20 then price - 2 else price

noncomputable def sequence1_final_price : ℝ := 
  additional_two_off_if_applicable (ten_percent_off (five_off initial_price))

noncomputable def sequence2_final_price : ℝ := 
  additional_two_off_if_applicable (five_off (ten_percent_off initial_price))

theorem best_discount_sequence : 
  sequence2_final_price = 20 ∧ 
  sequence2_final_price < sequence1_final_price ∧ 
  sequence1_final_price - sequence2_final_price = 0.5 :=
by
  sorry

end best_discount_sequence_l37_37689


namespace find_y_square_divisible_by_three_between_50_and_120_l37_37020

theorem find_y_square_divisible_by_three_between_50_and_120 :
  ∃ (y : ℕ), y = 81 ∧ (∃ (n : ℕ), y = n^2) ∧ (3 ∣ y) ∧ (50 < y) ∧ (y < 120) :=
by
  sorry

end find_y_square_divisible_by_three_between_50_and_120_l37_37020


namespace polar_circle_equation_l37_37565

theorem polar_circle_equation {r : ℝ} {phi : ℝ} {rho theta : ℝ} :
  (r = 2) → (phi = π / 3) → (rho = 4 * Real.cos (theta - π / 3)) :=
by
  intros hr hphi
  sorry

end polar_circle_equation_l37_37565


namespace max_value_S_n_l37_37059

open Nat

noncomputable def a_n (n : ℕ) : ℤ := 20 + (n - 1) * (-2)

noncomputable def S_n (n : ℕ) : ℤ := n * 20 + (n * (n - 1)) * (-2) / 2

theorem max_value_S_n : ∃ n : ℕ, S_n n = 110 :=
by
  sorry

end max_value_S_n_l37_37059


namespace three_seventy_five_as_fraction_l37_37917

theorem three_seventy_five_as_fraction : (15 : ℚ) / 4 = 3.75 := by
  sorry

end three_seventy_five_as_fraction_l37_37917


namespace inequality_solution_set_ab2_bc_ca_a3b_ge_1_4_l37_37974

theorem inequality_solution_set (x : ℝ) : (|x - 1| + |2 * x + 5| < 8) ↔ (-4 < x ∧ x < 4 / 3) :=
by
  sorry

theorem ab2_bc_ca_a3b_ge_1_4 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 1) :
  (a^2 / (b + 3 * c) + b^2 / (c + 3 * a) + c^2 / (a + 3 * b) ≥ 1 / 4) :=
by
  sorry

end inequality_solution_set_ab2_bc_ca_a3b_ge_1_4_l37_37974


namespace parity_of_f_monotonicity_of_f_9_l37_37491

-- Condition: f(x) = x + k / x with k ≠ 0
variable (k : ℝ) (hkn0 : k ≠ 0)
noncomputable def f (x : ℝ) : ℝ := x + k / x

-- 1. Prove the parity of the function is odd
theorem parity_of_f : ∀ x : ℝ, f k (-x) = -f k x := by
  sorry

-- Given condition: f(3) = 6, we derive k = 9
def k_9 : ℝ := 9
noncomputable def f_9 (x : ℝ) : ℝ := x + k_9 / x

-- 2. Prove the monotonicity of the function y = f(x) in the interval (-∞, -3]
theorem monotonicity_of_f_9 : ∀ (x1 x2 : ℝ), x1 < x2 → x1 ≤ -3 → x2 ≤ -3 → f_9 x1 < f_9 x2 := by
  sorry

end parity_of_f_monotonicity_of_f_9_l37_37491


namespace find_number_l37_37818

theorem find_number (X : ℝ) (h : 30 = 0.50 * X + 10) : X = 40 :=
by
  sorry

end find_number_l37_37818


namespace division_problem_l37_37755

theorem division_problem (D d q r : ℕ) 
  (h1 : D + d + q + r = 205)
  (h2 : q = d) :
  D = 174 ∧ d = 13 :=
by {
  sorry
}

end division_problem_l37_37755


namespace largest_int_value_of_m_l37_37122

variable {x y m : ℤ}

theorem largest_int_value_of_m (h1 : x + 2 * y = 2 * m + 1)
                              (h2 : 2 * x + y = m + 2)
                              (h3 : x - y > 2) : m = -2 := 
sorry

end largest_int_value_of_m_l37_37122


namespace parallel_lines_a_perpendicular_lines_a_l37_37377

-- Definitions of the lines
def l1 (a x y : ℝ) := a * x + 2 * y + 6 = 0
def l2 (a x y : ℝ) := x + (a - 1) * y + a^2 - 1 = 0

-- Statement for parallel lines problem
theorem parallel_lines_a (a : ℝ) :
  (∀ x y : ℝ, l1 a x y → l2 a x y) → (a = -1) :=
by
  sorry

-- Statement for perpendicular lines problem
theorem perpendicular_lines_a (a : ℝ) :
  (∀ x y : ℝ, l1 a x y → l2 a x y → (-a / 2) * (1 / (a - 1)) = -1) → (a = 2 / 3) :=
by
  sorry

end parallel_lines_a_perpendicular_lines_a_l37_37377


namespace average_visitors_per_day_correct_l37_37221

-- Define the average number of visitors on Sundays
def avg_visitors_sunday : ℕ := 660

-- Define the average number of visitors on other days
def avg_visitors_other : ℕ := 240

-- Define the number of Sundays in a 30-day month starting with a Sunday
def num_sundays_in_month : ℕ := 5

-- Define the number of other days in a 30-day month starting with a Sunday
def num_other_days_in_month : ℕ := 25

-- Calculate the total number of visitors in the month
def total_visitors_in_month : ℕ :=
  (num_sundays_in_month * avg_visitors_sunday) + (num_other_days_in_month * avg_visitors_other)

-- Define the number of days in the month
def days_in_month : ℕ := 30

-- Define the average number of visitors per day
def avg_visitors_per_day := total_visitors_in_month / days_in_month

-- State the theorem to be proved
theorem average_visitors_per_day_correct :
  avg_visitors_per_day = 310 :=
by
  sorry

end average_visitors_per_day_correct_l37_37221


namespace minimum_value_range_l37_37584

noncomputable def f (a x : ℝ) : ℝ := abs (3 * x - 1) + a * x + 2

theorem minimum_value_range (a : ℝ) :
  (-3 ≤ a ∧ a ≤ 3) ↔ ∃ m, ∀ x, f a x ≥ m := sorry

end minimum_value_range_l37_37584


namespace rectangle_width_decrease_l37_37928

theorem rectangle_width_decrease {L W : ℝ} (A : ℝ) (hA : A = L * W) (h_new_length : A = 1.25 * L * (W * y)) : y = 0.8 :=
by sorry

end rectangle_width_decrease_l37_37928


namespace randy_piggy_bank_final_amount_l37_37791

def initial_amount : ℕ := 200
def spending_per_trip : ℕ := 2
def trips_per_month : ℕ := 4
def months_per_year : ℕ := 12

theorem randy_piggy_bank_final_amount :
  initial_amount - (spending_per_trip * trips_per_month * months_per_year) = 104 :=
by
  -- proof to be filled in
  sorry

end randy_piggy_bank_final_amount_l37_37791


namespace depth_B_is_correct_l37_37438

-- Given: Diver A is at a depth of -55 meters.
def depth_A : ℤ := -55

-- Given: Diver B is 5 meters above diver A.
def offset : ℤ := 5

-- Prove: The depth of diver B
theorem depth_B_is_correct : (depth_A + offset) = -50 :=
by
  sorry

end depth_B_is_correct_l37_37438


namespace binomial_expansion_coefficient_x_l37_37399

theorem binomial_expansion_coefficient_x :
  (∃ (c : ℕ), (x : ℝ) → (x + 1/x^(1/2))^7 = c * x + (rest)) ∧ c = 35 := by
  sorry

end binomial_expansion_coefficient_x_l37_37399


namespace remainder_when_divided_l37_37925

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 + x^3 + 1

-- The statement to be proved
theorem remainder_when_divided (x : ℝ) : (p 2) = 25 :=
by
  sorry

end remainder_when_divided_l37_37925


namespace bill_amount_each_person_shared_l37_37557

noncomputable def total_bill : ℝ := 139.00
noncomputable def tip_percentage : ℝ := 0.10
noncomputable def num_people : ℝ := 7.00

noncomputable def tip : ℝ := tip_percentage * total_bill
noncomputable def total_bill_with_tip : ℝ := total_bill + tip
noncomputable def amount_each_person_pays : ℝ := total_bill_with_tip / num_people

theorem bill_amount_each_person_shared :
  amount_each_person_pays = 21.84 := by
  -- proof goes here
  sorry

end bill_amount_each_person_shared_l37_37557


namespace roots_of_cubic_8th_power_sum_l37_37279

theorem roots_of_cubic_8th_power_sum :
  ∀ a b c : ℂ, 
  (a + b + c = 0) → 
  (a * b + b * c + c * a = -1) → 
  (a * b * c = -1) → 
  (a^8 + b^8 + c^8 = 10) := 
by
  sorry

end roots_of_cubic_8th_power_sum_l37_37279


namespace sum_n_k_l37_37854

theorem sum_n_k (n k : ℕ) (h1 : 3 = n - 2 * k) (h2 : 15 = 5 * n - 8 * k) : n + k = 3 :=
by
  -- Use the conditions to conclude the proof.
  sorry

end sum_n_k_l37_37854


namespace first_team_odd_is_correct_l37_37338

noncomputable def odd_for_first_team : Real := 
  let odd2 := 5.23
  let odd3 := 3.25
  let odd4 := 2.05
  let bet_amount := 5.00
  let expected_win := 223.0072
  let total_odds := expected_win / bet_amount
  let denominator := odd2 * odd3 * odd4
  total_odds / denominator

theorem first_team_odd_is_correct : 
  odd_for_first_team = 1.28 := by 
  sorry

end first_team_odd_is_correct_l37_37338


namespace imons_no_entanglements_l37_37353

-- Define the fundamental structure for imons and their entanglements.
universe u
variable {α : Type u}

-- Define a graph structure to represent imons and their entanglement.
structure Graph (α : Type u) where
  vertices : Finset α
  edges : Finset (α × α)
  edge_sym : ∀ {x y}, (x, y) ∈ edges → (y, x) ∈ edges

-- Define the operations that can be performed on imons.
structure ImonOps (G : Graph α) where
  destroy : {v : α} → G.vertices.card % 2 = 1
  double : Graph α

-- Prove the main theorem
theorem imons_no_entanglements (G : Graph α) (op : ImonOps G) : 
  ∃ seq : List (ImonOps G), ∀ g : Graph α, g ∈ (seq.map (λ h => h.double)) → g.edges = ∅ :=
by
  sorry -- The proof would be constructed here.

end imons_no_entanglements_l37_37353


namespace graph_function_quadrant_l37_37540

theorem graph_function_quadrant (x y : ℝ): 
  (∀ x : ℝ, y = -x + 2 → (x < 0 → y ≠ -3 + - x)) := 
sorry

end graph_function_quadrant_l37_37540


namespace remainder_of_product_divided_by_7_l37_37580

theorem remainder_of_product_divided_by_7 :
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 2 :=
by
  sorry

end remainder_of_product_divided_by_7_l37_37580


namespace tangent_lines_to_curve_l37_37316

-- Define the curve
def curve (x : ℝ) : ℝ := x^3

-- Define the general form of a tangent line
def tangent_line (x : ℝ) (y : ℝ) (m : ℝ) (x0 : ℝ) (y0 : ℝ) : Prop :=
  y - y0 = m * (x - x0)

-- Define the conditions
def condition1 : Prop :=
  tangent_line 1 1 3 1 1

def condition2 : Prop :=
  tangent_line 1 1 (3/4) (-1/2) ((-1/2)^3)

-- Define the equations of the tangent lines
def line1 : Prop :=
  ∀ x y : ℝ, 3 * x - y - 2 = 0

def line2 : Prop :=
  ∀ x y : ℝ, 3 * x - 4 * y + 1 = 0

-- The final theorem statement
theorem tangent_lines_to_curve :
  (condition1 → line1) ∧ (condition2 → line2) :=
  by
    sorry -- Placeholder for proof

end tangent_lines_to_curve_l37_37316


namespace total_cost_correct_l37_37840

def cost_of_cat_toy := 10.22
def cost_of_cage := 11.73
def cost_of_cat_food := 7.50
def cost_of_leash := 5.15
def cost_of_cat_treats := 3.98

theorem total_cost_correct : 
  cost_of_cat_toy + cost_of_cage + cost_of_cat_food + cost_of_leash + cost_of_cat_treats = 38.58 := 
by
  sorry

end total_cost_correct_l37_37840


namespace books_sold_l37_37069

theorem books_sold {total_books sold_fraction left_fraction : ℕ} (h_total : total_books = 9900)
    (h_fraction : left_fraction = 4/6) (h_sold : sold_fraction = 1 - left_fraction) : 
  (sold_fraction * total_books) = 3300 := 
  by 
  sorry

end books_sold_l37_37069


namespace scientific_notation_correct_l37_37429

theorem scientific_notation_correct : 1630000 = 1.63 * 10^6 :=
by sorry

end scientific_notation_correct_l37_37429


namespace vertex_on_xaxis_l37_37343

-- Definition of the parabola equation with vertex on the x-axis
def parabola (x m : ℝ) := x^2 - 8 * x + m

-- The problem statement: show that m = 16 given that the vertex of the parabola is on the x-axis
theorem vertex_on_xaxis (m : ℝ) : ∃ x : ℝ, parabola x m = 0 → m = 16 :=
by
  sorry

end vertex_on_xaxis_l37_37343


namespace infinite_series_sum_eq_one_fourth_l37_37084

theorem infinite_series_sum_eq_one_fourth :
  (∑' n : ℕ, 3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+2))) = 1 / 4 :=
sorry

end infinite_series_sum_eq_one_fourth_l37_37084


namespace solve_for_x_over_z_l37_37965

variables (x y z : ℝ)

theorem solve_for_x_over_z
  (h1 : x + y = 2 * x + z)
  (h2 : x - 2 * y = 4 * z)
  (h3 : x + y + z = 21)
  (h4 : y = 6 * z) :
  x / z = 5 :=
sorry

end solve_for_x_over_z_l37_37965


namespace sum_gcd_lcm_eq_180195_l37_37954

def gcd_60_45045 := Nat.gcd 60 45045
def lcm_60_45045 := Nat.lcm 60 45045

theorem sum_gcd_lcm_eq_180195 : gcd_60_45045 + lcm_60_45045 = 180195 := by
  sorry

end sum_gcd_lcm_eq_180195_l37_37954


namespace hcf_36_84_l37_37743

theorem hcf_36_84 : Nat.gcd 36 84 = 12 := by
  sorry

end hcf_36_84_l37_37743


namespace a4_plus_a5_eq_27_l37_37819

-- Define the geometric sequence conditions
variables (a : ℕ → ℝ) (q : ℝ)
axiom a_pos : ∀ n, a n > 0
axiom a_2 : a 2 = 1 - a 1
axiom a_4 : a 4 = 9 - a 3

-- Define the geometric sequence property
axiom geom_seq : ∀ n, a (n + 1) = a n * q

theorem a4_plus_a5_eq_27 : a 4 + a 5 = 27 := sorry

end a4_plus_a5_eq_27_l37_37819


namespace running_race_total_students_l37_37735

theorem running_race_total_students 
  (number_of_first_grade_students number_of_second_grade_students : ℕ)
  (h1 : number_of_first_grade_students = 8)
  (h2 : number_of_second_grade_students = 5 * number_of_first_grade_students) :
  number_of_first_grade_students + number_of_second_grade_students = 48 := 
by
  -- we will leave the proof empty
  sorry

end running_race_total_students_l37_37735


namespace domain_of_tan_l37_37715

open Real

noncomputable def function_domain : Set ℝ :=
  {x | ∀ k : ℤ, x ≠ k * π + 3 * π / 4}

theorem domain_of_tan : ∀ x : ℝ,
  (∃ k : ℤ, x = k * π + 3 * π / 4) → ¬ (∃ y : ℝ, y = tan (π / 4 - x)) :=
by
  intros x hx
  obtain ⟨k, hk⟩ := hx
  sorry

end domain_of_tan_l37_37715


namespace triangle_perimeter_l37_37164

-- Define the given quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ :=
  x^2 - (5 + m) * x + 5 * m

-- Define the isosceles triangle with sides given by the roots of the equation
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

-- Defining the fact that 2 is a root of the given quadratic equation with an unknown m
lemma two_is_root (m : ℝ) : quadratic_equation m 2 = 0 := sorry

-- Prove that the perimeter of triangle ABC is 12 given the conditions
theorem triangle_perimeter (α β γ : ℝ) (m : ℝ) (h1 : quadratic_equation m α = 0) 
  (h2 : quadratic_equation m β = 0) 
  (h3 : is_isosceles_triangle α β γ) : α + β + γ = 12 := sorry

end triangle_perimeter_l37_37164


namespace molecular_weight_of_acetic_acid_l37_37740

-- Define the molecular weight of 7 moles of acetic acid
def molecular_weight_7_moles_acetic_acid := 420 

-- Define the number of moles of acetic acid
def moles_acetic_acid := 7

-- Define the molecular weight of 1 mole of acetic acid
def molecular_weight_1_mole_acetic_acid := molecular_weight_7_moles_acetic_acid / moles_acetic_acid

-- The theorem stating that given the molecular weight of 7 moles of acetic acid, we have the molecular weight of the acetic acid
theorem molecular_weight_of_acetic_acid : molecular_weight_1_mole_acetic_acid = 60 := by
  -- proof to be solved
  sorry

end molecular_weight_of_acetic_acid_l37_37740


namespace max_ab_plus_2bc_l37_37479

theorem max_ab_plus_2bc (A B C : ℝ) (AB AC BC : ℝ) (hB : B = 60) (hAC : AC = Real.sqrt 3) :
  (AB + 2 * BC) ≤ 2 * Real.sqrt 7 :=
sorry

end max_ab_plus_2bc_l37_37479


namespace bathroom_new_area_l37_37582

theorem bathroom_new_area
  (current_area : ℕ)
  (current_width : ℕ)
  (extension : ℕ)
  (current_area_eq : current_area = 96)
  (current_width_eq : current_width = 8)
  (extension_eq : extension = 2) :
  ∃ new_area : ℕ, new_area = 144 :=
by
  sorry

end bathroom_new_area_l37_37582


namespace area_enclosed_by_curve_l37_37793

theorem area_enclosed_by_curve :
  let arc_length := (3 * Real.pi) / 4
  let side_length := 3
  let radius := arc_length / ((3 * Real.pi) / 4)
  let sector_area := (radius ^ 2 * Real.pi * (3 * Real.pi) / (4 * 2 * Real.pi))
  let total_sector_area := 8 * sector_area
  let octagon_area := 2 * (1 + Real.sqrt 2) * (side_length ^ 2)
  total_sector_area + octagon_area = 54 + 54 * Real.sqrt 2 + 3 * Real.pi
:= sorry

end area_enclosed_by_curve_l37_37793


namespace parallel_statements_l37_37627

-- Definitions for lines and planes
variables {Line Plane : Type}

-- Parallelism between a line and another line or a plane
variables (a b : Line) (α : Plane)

-- Parallel relationship assertions
axiom parallel_lines (l1 l2 : Line) : Prop -- l1 is parallel to l2
axiom line_in_plane (l : Line) (p : Plane) : Prop -- line l is in plane p
axiom parallel_line_plane (l : Line) (p : Plane) : Prop -- line l is parallel to plane p

-- Problem statement
theorem parallel_statements :
  (parallel_lines a b ∧ line_in_plane b α → parallel_line_plane a α) ∧
  (parallel_lines a b ∧ parallel_line_plane a α → parallel_line_plane b α) :=
sorry

end parallel_statements_l37_37627


namespace reciprocal_neg3_l37_37955

-- Define the problem
def reciprocal (x : ℚ) : ℚ := 1 / x

-- The required proof statement
theorem reciprocal_neg3 : reciprocal (-3) = -1 / 3 :=
by
  sorry

end reciprocal_neg3_l37_37955


namespace find_original_class_strength_l37_37745

-- Definitions based on given conditions
def original_average_age : ℝ := 40
def additional_students : ℕ := 12
def new_students_average_age : ℝ := 32
def decrease_in_average : ℝ := 4
def new_average_age : ℝ := original_average_age - decrease_in_average

-- The equation setup
theorem find_original_class_strength (N : ℕ) (T : ℝ) 
  (h1 : T = original_average_age * N) 
  (h2 : T + additional_students * new_students_average_age = new_average_age * (N + additional_students)) : 
  N = 12 := 
sorry

end find_original_class_strength_l37_37745


namespace popularity_order_l37_37904

def chess_popularity := 5 / 16
def drama_popularity := 7 / 24
def music_popularity := 11 / 32
def art_popularity := 13 / 48

theorem popularity_order :
  (31 / 96 < 34 / 96) ∧ (34 / 96 < 35 / 96) ∧ (35 / 96 < 36 / 96) ∧ 
  (chess_popularity < music_popularity) ∧ 
  (drama_popularity < music_popularity) ∧ 
  (music_popularity > art_popularity) ∧ 
  (chess_popularity > drama_popularity) ∧ 
  (drama_popularity > art_popularity) := 
sorry

end popularity_order_l37_37904


namespace polynomial_is_2y2_l37_37259

variables (x y : ℝ)

theorem polynomial_is_2y2 (P : ℝ → ℝ → ℝ) (h : P x y + (x^2 - y^2) = x^2 + y^2) : 
  P x y = 2 * y^2 :=
by
  sorry

end polynomial_is_2y2_l37_37259


namespace number_of_pens_l37_37155

theorem number_of_pens (x y : ℝ) (h1 : 60 * (x + 2 * y) = 50 * (x + 3 * y)) (h2 : x = 3 * y) : 
  (60 * (x + 2 * y)) / x = 100 :=
by
  sorry

end number_of_pens_l37_37155


namespace interest_rate_A_l37_37430

-- Definitions for the conditions
def principal : ℝ := 1000
def rate_C : ℝ := 0.115
def time_period : ℝ := 3
def gain_B : ℝ := 45

-- Main theorem to prove
theorem interest_rate_A {R : ℝ} (h1 : gain_B = (principal * rate_C * time_period - principal * (R / 100) * time_period)) : R = 10 := 
by
  sorry

end interest_rate_A_l37_37430


namespace acute_angle_alpha_range_l37_37977

theorem acute_angle_alpha_range (x : ℝ) (α : ℝ) (h1 : 0 < x) (h2 : x < 90) (h3 : α = 180 - 2 * x) : 0 < α ∧ α < 180 :=
by
  sorry

end acute_angle_alpha_range_l37_37977


namespace point_direction_form_eq_l37_37380

-- Define the conditions
def point := (1, 2)
def direction_vector := (3, -4)

-- Define a function to represent the line equation based on point and direction
def line_equation (x y : ℝ) : Prop :=
  (x - point.1) / direction_vector.1 = (y - point.2) / direction_vector.2

-- State the theorem
theorem point_direction_form_eq (x y : ℝ) :
  (x - 1) / 3 = (y - 2) / -4 →
  line_equation x y :=
sorry

end point_direction_form_eq_l37_37380


namespace find_m_l37_37810

-- Define the conditions with variables a, b, and m.
variable (a b m : ℝ)
variable (ha : 2^a = m)
variable (hb : 5^b = m)
variable (hc : 1/a + 1/b = 2)

-- Define the statement to be proven.
theorem find_m : m = Real.sqrt 10 :=
by
  sorry


end find_m_l37_37810


namespace quadrilateral_is_parallelogram_l37_37852

theorem quadrilateral_is_parallelogram (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 2 * a * b + 2 * c * d) : a = b ∧ c = d :=
by
  sorry

end quadrilateral_is_parallelogram_l37_37852


namespace value_of_b_l37_37995

theorem value_of_b (b : ℝ) :
  (∀ x : ℝ, 3 * (5 + b * x) = 18 * x + 15) → b = 6 :=
by
  intro h
  -- Proving that b = 6
  sorry

end value_of_b_l37_37995


namespace num_adult_tickets_l37_37692

theorem num_adult_tickets (adult_ticket_cost child_ticket_cost total_tickets_sold total_receipts : ℕ) 
  (h1 : adult_ticket_cost = 12) 
  (h2 : child_ticket_cost = 4) 
  (h3 : total_tickets_sold = 130) 
  (h4 : total_receipts = 840) :
  ∃ A C : ℕ, A + C = total_tickets_sold ∧ adult_ticket_cost * A + child_ticket_cost * C = total_receipts ∧ A = 40 :=
by {
  sorry
}

end num_adult_tickets_l37_37692


namespace personBCatchesPersonAAtB_l37_37475

-- Definitions based on the given problem's conditions
def personADepartsTime : ℕ := 8 * 60  -- Person A departs at 8:00 AM, given in minutes
def personBDepartsTime : ℕ := 9 * 60  -- Person B departs at 9:00 AM, given in minutes
def catchUpTime : ℕ := 11 * 60        -- Persons meet at 11:00 AM, given in minutes
def returnMultiplier : ℕ := 2         -- Person B returns at double the speed
def chaseMultiplier : ℕ := 2          -- After returning, Person B doubles their speed again

-- Exact question we want to prove
def meetAtBTime : ℕ := 12 * 60 + 48   -- Time when Person B catches up with Person A at point B

-- Statement to be proven
theorem personBCatchesPersonAAtB :
  ∀ (VA VB : ℕ) (x : ℕ),
    VA = 2 * x ∧ VB = 3 * x →
    ∃ t : ℕ, t = meetAtBTime := by
  sorry

end personBCatchesPersonAAtB_l37_37475


namespace number_of_toys_sold_l37_37029

theorem number_of_toys_sold (n : ℕ) 
  (sell_price : ℕ) (gain_price : ℕ) (cost_price_per_toy : ℕ) :
  sell_price = 27300 → 
  gain_price = 3 * cost_price_per_toy → 
  cost_price_per_toy = 1300 →
  n * cost_price_per_toy + gain_price = sell_price → 
  n = 18 :=
by sorry

end number_of_toys_sold_l37_37029


namespace percentage_of_local_arts_students_is_50_l37_37064

-- Definitions
def total_students_arts := 400
def total_students_science := 100
def total_students_commerce := 120
def percent_local_science := 25 / 100
def percent_local_commerce := 85 / 100
def total_locals := 327

-- Problem statement in Lean
theorem percentage_of_local_arts_students_is_50
  (x : ℕ) -- Percentage of local arts students as a natural number
  (h1 : percent_local_science * total_students_science = 25)
  (h2 : percent_local_commerce * total_students_commerce = 102)
  (h3 : (x / 100 : ℝ) * total_students_arts + 25 + 102 = total_locals) :
  x = 50 :=
sorry

end percentage_of_local_arts_students_is_50_l37_37064


namespace Mark_charged_more_l37_37286

theorem Mark_charged_more (K P M : ℕ) 
  (h1 : P = 2 * K) 
  (h2 : P = M / 3)
  (h3 : K + P + M = 153) : M - K = 85 :=
by
  -- proof to be filled in later
  sorry

end Mark_charged_more_l37_37286


namespace incorrect_conclusion_D_l37_37108

def parabola (x : ℝ) : ℝ := (x - 2) ^ 2 + 1

theorem incorrect_conclusion_D :
  ∀ x : ℝ, x < 2 → ∃ y1 y2 : ℝ, y1 = parabola x ∧ y2 = parabola (x + 1) ∧ y1 > y2 :=
by
  sorry

end incorrect_conclusion_D_l37_37108


namespace simplest_radical_l37_37406

theorem simplest_radical (r1 r2 r3 r4 : ℝ) 
  (h1 : r1 = Real.sqrt 3) 
  (h2 : r2 = Real.sqrt 4)
  (h3 : r3 = Real.sqrt 8)
  (h4 : r4 = Real.sqrt (1 / 2)) : r1 = Real.sqrt 3 :=
  by sorry

end simplest_radical_l37_37406


namespace power_quotient_example_l37_37510

theorem power_quotient_example (a : ℕ) (m n : ℕ) (h : 23^11 / 23^8 = 23^(11 - 8)) : 23^3 = 12167 := by
  sorry

end power_quotient_example_l37_37510


namespace find_diameter_of_hemisphere_l37_37591

theorem find_diameter_of_hemisphere (r a : ℝ) (hr : r = a / 2) (volume : ℝ) (hV : volume = 18 * Real.pi) : 
  2/3 * Real.pi * r ^ 3 = 18 * Real.pi → a = 6 := by
  intro h
  sorry

end find_diameter_of_hemisphere_l37_37591


namespace area_of_quadrilateral_ABFG_l37_37246

/-- 
Given conditions:
1. Rectangle with dimensions AC = 40 and AE = 24.
2. Points B and F are midpoints of sides AC and AE, respectively.
3. G is the midpoint of DE.
Prove that the area of quadrilateral ABFG is 600 square units.
-/
theorem area_of_quadrilateral_ABFG (AC AE : ℝ) (B F G : ℤ) 
  (hAC : AC = 40) (hAE : AE = 24) (hB : B = 1/2 * AC) (hF : F = 1/2 * AE) (hG : G = 1/2 * AE):
  area_of_ABFG = 600 :=
by
  sorry

end area_of_quadrilateral_ABFG_l37_37246


namespace each_charity_gets_45_dollars_l37_37272

def dozens : ℤ := 6
def cookies_per_dozen : ℤ := 12
def total_cookies : ℤ := dozens * cookies_per_dozen
def selling_price_per_cookie : ℚ := 1.5
def cost_per_cookie : ℚ := 0.25
def profit_per_cookie : ℚ := selling_price_per_cookie - cost_per_cookie
def total_profit : ℚ := profit_per_cookie * total_cookies
def charities : ℤ := 2
def amount_per_charity : ℚ := total_profit / charities

theorem each_charity_gets_45_dollars : amount_per_charity = 45 := 
by
  sorry

end each_charity_gets_45_dollars_l37_37272


namespace arithmetic_sequence_b3b7_l37_37663

theorem arithmetic_sequence_b3b7 (b : ℕ → ℤ) (d : ℤ)
  (h_arith_seq : ∀ n, b (n + 1) = b n + d)
  (h_increasing : ∀ n, b n < b (n + 1))
  (h_cond : b 4 * b 6 = 17) : 
  b 3 * b 7 = -175 :=
sorry

end arithmetic_sequence_b3b7_l37_37663


namespace rectangle_to_square_l37_37916

theorem rectangle_to_square (a b : ℝ) (h1 : b / 2 < a) (h2 : a < b) :
  ∃ (r : ℝ), r = Real.sqrt (a * b) ∧ 
    (∃ (cut1 cut2 : ℝ × ℝ), 
      cut1.1 = 0 ∧ cut1.2 = a ∧
      cut2.1 = b - r ∧ cut2.2 = r - a ∧
      ∀ t, t = (a * b) - (r ^ 2)) := sorry

end rectangle_to_square_l37_37916


namespace shaded_area_is_correct_l37_37463

noncomputable def total_shaded_area : ℝ :=
  let s := 10
  let R := s / (2 * Real.sin (Real.pi / 8))
  let A := (1 / 2) * R^2 * Real.sin (2 * Real.pi / 8)
  4 * A

theorem shaded_area_is_correct :
  total_shaded_area = 200 * Real.sqrt 2 / Real.sin (Real.pi / 8)^2 := 
sorry

end shaded_area_is_correct_l37_37463


namespace largest_number_is_a_l37_37111

-- Define the numbers in their respective bases
def a := 8 * 9 + 5
def b := 3 * 5^2 + 0 * 5 + 1 * 5^0
def c := 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0

theorem largest_number_is_a : a > b ∧ a > c :=
by
  -- These are the expected results, rest is the proof steps which we skip using sorry
  have ha : a = 77 := rfl
  have hb : b = 76 := rfl
  have hc : c = 9 := rfl
  sorry

end largest_number_is_a_l37_37111


namespace mutually_exclusive_white_ball_events_l37_37150

-- Definitions of persons and balls
inductive Person | A | B | C
inductive Ball | red | black | white

-- Definitions of events
def eventA (dist : Person → Ball) : Prop := dist Person.A = Ball.white
def eventB (dist : Person → Ball) : Prop := dist Person.B = Ball.white

theorem mutually_exclusive_white_ball_events (dist : Person → Ball) :
  (eventA dist → ¬eventB dist) :=
by
  sorry

end mutually_exclusive_white_ball_events_l37_37150


namespace derivative_at_pi_over_2_l37_37008

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem derivative_at_pi_over_2 : 
  (deriv f (π / 2)) = Real.exp (π / 2) :=
by
  sorry

end derivative_at_pi_over_2_l37_37008


namespace range_of_a_l37_37670

noncomputable def range_a : Set ℝ :=
  {a : ℝ | 0 < a ∧ a ≤ 1/2}

theorem range_of_a (O P : ℝ × ℝ) (Q : ℝ × ℝ)
  (hO : O = (0, 0))
  (hP : P = (a, 0))
  (ha : 0 < a)
  (hQ : ∃ m : ℝ, Q = (m^2, m))
  (hPQ_PO : ∀ Q, Q = (m^2, m) → dist P Q ≥ dist O P) :
  a ∈ range_a :=
sorry

end range_of_a_l37_37670


namespace find_number_of_friends_l37_37909

def dante_balloons : Prop :=
  ∃ F : ℕ, (F > 0 ∧ (250 / F) - 11 = 39) ∧ F = 5

theorem find_number_of_friends : dante_balloons :=
by
  sorry

end find_number_of_friends_l37_37909


namespace jerry_removed_old_figures_l37_37503

-- Let's declare the conditions
variables (initial_count added_count current_count removed_count : ℕ)
variables (h1 : initial_count = 7)
variables (h2 : added_count = 11)
variables (h3 : current_count = 8)

-- The statement to prove
theorem jerry_removed_old_figures : removed_count = initial_count + added_count - current_count :=
by
  -- The proof will go here, but we'll use sorry to skip it
  sorry

end jerry_removed_old_figures_l37_37503


namespace farmer_loss_representative_value_l37_37814

def check_within_loss_range (S L : ℝ) : Prop :=
  (S = 100000) → (20000 ≤ L ∧ L ≤ 25000)

theorem farmer_loss_representative_value : check_within_loss_range 100000 21987.53 :=
by
  intros hs
  sorry

end farmer_loss_representative_value_l37_37814


namespace ac_length_l37_37645

theorem ac_length (AB : ℝ) (H1 : AB = 100)
    (BC AC : ℝ)
    (H2 : AC = (1 + Real.sqrt 5)/2 * BC)
    (H3 : AC + BC = AB) : AC = 75 - 25 * Real.sqrt 5 :=
by
  sorry

end ac_length_l37_37645


namespace exponential_order_l37_37944

theorem exponential_order (x y : ℝ) (a : ℝ) (hx : x > y) (hy : y > 1) (ha1 : 0 < a) (ha2 : a < 1) : a^x < a^y :=
sorry

end exponential_order_l37_37944


namespace intersection_A_B_l37_37252

noncomputable def A : Set ℝ := { y | ∃ x : ℝ, y = Real.sin x }
noncomputable def B : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem intersection_A_B : A ∩ B = { y | 0 ≤ y ∧ y ≤ 1 } :=
by 
  sorry

end intersection_A_B_l37_37252


namespace simplify_fraction_l37_37583

theorem simplify_fraction :
  (6 * x ^ 3 + 13 * x ^ 2 + 15 * x - 25) / (2 * x ^ 3 + 4 * x ^ 2 + 4 * x - 10) =
  (6 * x - 5) / (2 * x - 2) :=
by
  sorry

end simplify_fraction_l37_37583


namespace arithmetic_sequence_product_l37_37571

theorem arithmetic_sequence_product (a d : ℕ) :
  (a + 7 * d = 20) → (d = 2) → ((a + d) * (a + 2 * d) = 80) :=
by
  intros h₁ h₂
  sorry

end arithmetic_sequence_product_l37_37571


namespace root_conditions_l37_37525

-- Given conditions and definitions:
def quadratic_eq (m x : ℝ) : ℝ := x^2 + (m - 3) * x + m

-- The proof problem statement
theorem root_conditions (m : ℝ) (h1 : ∃ x y : ℝ, quadratic_eq m x = 0 ∧ quadratic_eq m y = 0 ∧ x > 1 ∧ y < 1) : m < 1 :=
sorry

end root_conditions_l37_37525


namespace find_C_l37_37737

-- Variables and conditions
variables (A B C : ℝ)

-- Conditions given in the problem
def condition1 : Prop := A + B + C = 1000
def condition2 : Prop := A + C = 700
def condition3 : Prop := B + C = 600

-- The statement to be proved
theorem find_C (h1 : condition1 A B C) (h2 : condition2 A C) (h3 : condition3 B C) : C = 300 :=
sorry

end find_C_l37_37737


namespace dot_product_of_vectors_l37_37319

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (-1, 1) - vector_a

theorem dot_product_of_vectors :
  vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = -4 :=
by
  sorry

end dot_product_of_vectors_l37_37319


namespace n_five_minus_n_divisible_by_30_l37_37447

theorem n_five_minus_n_divisible_by_30 (n : ℤ) : 30 ∣ (n^5 - n) :=
sorry

end n_five_minus_n_divisible_by_30_l37_37447


namespace not_possible_2002_pieces_l37_37206

theorem not_possible_2002_pieces (k : ℤ) : ¬ (1 + 7 * k = 2002) :=
by
  sorry

end not_possible_2002_pieces_l37_37206


namespace fraction_of_sum_l37_37061

theorem fraction_of_sum (l : List ℝ) (hl : l.length = 51)
  (n : ℝ) (hn : n ∈ l)
  (h : n = 7 * (l.erase n).sum / 50) :
  n / l.sum = 7 / 57 := by
  sorry

end fraction_of_sum_l37_37061


namespace cell_phone_bill_l37_37833

-- Definitions
def base_cost : ℝ := 20
def cost_per_text : ℝ := 0.05
def cost_per_extra_minute : ℝ := 0.10
def texts_sent : ℕ := 100
def hours_talked : ℝ := 30.5
def included_hours : ℝ := 30

-- Calculate extra minutes used
def extra_minutes : ℝ := (hours_talked - included_hours) * 60

-- Total cost calculation
def total_cost : ℝ := 
  base_cost + 
  (texts_sent * cost_per_text) + 
  (extra_minutes * cost_per_extra_minute)

-- Proof problem statement
theorem cell_phone_bill : total_cost = 28 := by
  sorry

end cell_phone_bill_l37_37833


namespace solve_fruit_juice_problem_l37_37714

open Real

noncomputable def fruit_juice_problem : Prop :=
  ∃ x, ((0.12 * 3 + x) / (3 + x) = 0.185) ∧ (x = 0.239)

theorem solve_fruit_juice_problem : fruit_juice_problem :=
sorry

end solve_fruit_juice_problem_l37_37714


namespace maximum_students_l37_37388

theorem maximum_students (x : ℕ) (hx : x / 2 + x / 4 + x / 7 + 6 > x) : x ≤ 28 :=
by sorry

end maximum_students_l37_37388


namespace efficiency_ratio_l37_37576

-- Define the work efficiencies
def EA : ℚ := 1 / 12
def EB : ℚ := 1 / 24
def EAB : ℚ := 1 / 8

-- State the theorem
theorem efficiency_ratio (EAB_eq : EAB = EA + EB) : (EA / EB) = 2 := by
  -- Insert proof here
  sorry

end efficiency_ratio_l37_37576


namespace find_k_value_l37_37193

noncomputable def solve_for_k (k : ℚ) : Prop :=
  ∃ x : ℚ, (x = 1) ∧ (3 * x + (2 * k - 1) = x - 6 * (3 * k + 2))

theorem find_k_value : solve_for_k (-13 / 20) :=
  sorry

end find_k_value_l37_37193


namespace profit_is_35_percent_l37_37110

def cost_price (C : ℝ) := C
def initial_selling_price (C : ℝ) := 1.20 * C
def second_selling_price (C : ℝ) := 1.50 * C
def final_selling_price (C : ℝ) := 1.35 * C

theorem profit_is_35_percent (C : ℝ) : 
    final_selling_price C - cost_price C = 0.35 * cost_price C :=
by
    sorry

end profit_is_35_percent_l37_37110


namespace tom_initial_game_count_zero_l37_37143

theorem tom_initial_game_count_zero
  (batman_game_cost superman_game_cost total_expenditure initial_game_count : ℝ)
  (h_batman_cost : batman_game_cost = 13.60)
  (h_superman_cost : superman_game_cost = 5.06)
  (h_total_expenditure : total_expenditure = 18.66)
  (h_initial_game_cost : initial_game_count = total_expenditure - (batman_game_cost + superman_game_cost)) :
  initial_game_count = 0 :=
by
  sorry

end tom_initial_game_count_zero_l37_37143


namespace completion_time_C_l37_37153

theorem completion_time_C (r_A r_B r_C : ℝ) 
  (h1 : r_A + r_B = 1 / 3) 
  (h2 : r_B + r_C = 1 / 3) 
  (h3 : r_A + r_C = 1 / 3) :
  1 / r_C = 6 :=
by
  sorry

end completion_time_C_l37_37153


namespace islander_real_name_l37_37756

-- Definition of types of people on the island
inductive IslanderType
| Knight   -- Always tells the truth
| Liar     -- Always lies
| Normal   -- Can lie or tell the truth

-- The possible names of the islander
inductive Name
| Edwin
| Edward

-- Condition: You met the islander who can be Edwin or Edward
def possible_names : List Name := [Name.Edwin, Name.Edward]

-- Condition: The islander said their name is Edward
def islander_statement : Name := Name.Edward

-- Condition: The islander is a Liar (as per the solution interpretation)
def islander_type : IslanderType := IslanderType.Liar

-- The proof problem: Prove the islander's real name is Edwin
theorem islander_real_name : islander_type = IslanderType.Liar ∧ islander_statement = Name.Edward → ∃ n : Name, n = Name.Edwin :=
by
  sorry

end islander_real_name_l37_37756


namespace fraction_power_evaluation_l37_37789

theorem fraction_power_evaluation (x y : ℚ) (h1 : x = 2 / 3) (h2 : y = 3 / 2) : 
  (3 / 4) * x^8 * y^9 = 9 / 8 := 
by
  sorry

end fraction_power_evaluation_l37_37789


namespace fraction_of_students_who_say_dislike_but_actually_like_l37_37996

-- Define the conditions
def total_students : ℕ := 100
def like_dancing : ℕ := total_students / 2
def dislike_dancing : ℕ := total_students / 2

def like_dancing_honest : ℕ := (7 * like_dancing) / 10
def like_dancing_dishonest : ℕ := (3 * like_dancing) / 10

def dislike_dancing_honest : ℕ := (4 * dislike_dancing) / 5
def dislike_dancing_dishonest : ℕ := dislike_dancing / 5

-- Define the proof objective
theorem fraction_of_students_who_say_dislike_but_actually_like :
  (like_dancing_dishonest : ℚ) / (total_students - like_dancing_honest - dislike_dancing_dishonest) = 3 / 11 :=
by
  sorry

end fraction_of_students_who_say_dislike_but_actually_like_l37_37996


namespace count_triangles_l37_37637

-- Define the conditions for the problem
def P (x1 x2 : ℕ) : Prop := 37 * x1 ≤ 2022 ∧ 37 * x2 ≤ 2022

def valid_points (x y : ℕ) : Prop := 37 * x + y = 2022

def area_multiple_of_3 (x1 x2 : ℕ): Prop :=
  (∃ k : ℤ, 3 * k = x1 - x2) ∧ x1 ≠ x2 ∧ P x1 x2

-- The final theorem to prove the number of such distinct triangles
theorem count_triangles : 
  (∃ (n : ℕ), n = 459 ∧ 
    ∃ x1 x2 : ℕ, area_multiple_of_3 x1 x2 ∧ x1 ≠ x2) :=
by
  sorry

end count_triangles_l37_37637


namespace imaginary_unit_power_l37_37796

def i := Complex.I

theorem imaginary_unit_power :
  ∀ a : ℝ, (2 - i + a * i ^ 2011).im = 0 → i ^ 2011 = i :=
by
  intro a
  intro h
  sorry

end imaginary_unit_power_l37_37796


namespace minimum_bamboo_fencing_length_l37_37709

theorem minimum_bamboo_fencing_length 
  (a b z : ℝ) 
  (h1 : a * b = 50)
  (h2 : a + 2 * b = z) : 
  z ≥ 20 := 
  sorry

end minimum_bamboo_fencing_length_l37_37709


namespace mark_reading_time_l37_37942

-- Definitions based on conditions
def daily_reading_hours : ℕ := 3
def days_in_week : ℕ := 7
def weekly_increase : ℕ := 6

-- Proof statement
theorem mark_reading_time : daily_reading_hours * days_in_week + weekly_increase = 27 := by
  -- placeholder for the proof
  sorry

end mark_reading_time_l37_37942


namespace no_solution_l37_37000

def is_digit (B : ℕ) : Prop := B < 10

def divisible_by (n m : ℕ) : Prop := ∃ k, n = m * k

def satisfies_conditions (B : ℕ) : Prop :=
  is_digit B ∧
  divisible_by (12345670 + B) 2 ∧
  divisible_by (12345670 + B) 5 ∧
  divisible_by (12345670 + B) 11

theorem no_solution (B : ℕ) : ¬ satisfies_conditions B :=
sorry

end no_solution_l37_37000


namespace find_f_2012_l37_37196

-- Given a function f: ℤ → ℤ that satisfies the functional equation:
def functional_equation (f : ℤ → ℤ) := ∀ m n : ℤ, m + f (m + f (n + f m)) = n + f m

-- Given condition:
def f_6_is_6 (f : ℤ → ℤ) := f 6 = 6

-- We need to prove that f 2012 = -2000 under the given conditions.
theorem find_f_2012 (f : ℤ → ℤ) (hf : functional_equation f) (hf6 : f_6_is_6 f) : f 2012 = -2000 := sorry

end find_f_2012_l37_37196


namespace evaluate_expression_at_x_l37_37541

theorem evaluate_expression_at_x (x : ℝ) (h : x = Real.sqrt 2 - 3) : 
  (3 * x / (x^2 - 9)) * (1 - 3 / x) - 2 / (x + 3) = Real.sqrt 2 / 2 := by
  sorry

end evaluate_expression_at_x_l37_37541


namespace distance_AB_l37_37619

theorem distance_AB : 
  let A := -1
  let B := 2020
  |A - B| = 2021 := by
  sorry

end distance_AB_l37_37619


namespace regular_pay_per_hour_l37_37315

theorem regular_pay_per_hour (R : ℝ) (h : 40 * R + 11 * (2 * R) = 186) : R = 3 :=
by
  sorry

end regular_pay_per_hour_l37_37315


namespace andy_solves_16_problems_l37_37957

theorem andy_solves_16_problems :
  ∃ N : ℕ, 
    N = (125 - 78)/3 + 1 ∧
    (78 + (N - 1) * 3 <= 125) ∧
    N = 16 := 
by 
  sorry

end andy_solves_16_problems_l37_37957


namespace rectangle_diagonal_length_l37_37613

theorem rectangle_diagonal_length
    (PQ QR : ℝ) (RT RU ST : ℝ) (Area_RST : ℝ)
    (hPQ : PQ = 8) (hQR : QR = 10)
    (hRT_RU : RT = RU)
    (hArea_RST: Area_RST = (1/5) * (PQ * QR)) :
    ST = 8 :=
by
  sorry

end rectangle_diagonal_length_l37_37613


namespace John_study_time_second_exam_l37_37151

variable (StudyTime Score : ℝ)
variable (k : ℝ) (h1 : k = Score / StudyTime)
variable (study_first : ℝ := 3) (score_first : ℝ := 60)
variable (avg_target : ℝ := 75)
variable (total_tests : ℕ := 2)

theorem John_study_time_second_exam :
  (avg_target * total_tests - score_first) / (score_first / study_first) = 4.5 :=
by
  sorry

end John_study_time_second_exam_l37_37151


namespace Suzanna_bike_distance_l37_37886

theorem Suzanna_bike_distance (ride_rate distance_time total_time : ℕ)
  (constant_rate : ride_rate = 3) (time_interval : distance_time = 10)
  (total_riding_time : total_time = 40) :
  (total_time / distance_time) * ride_rate = 12 :=
by
  -- Assuming the conditions:
  -- ride_rate = 3
  -- distance_time = 10
  -- total_time = 40
  sorry

end Suzanna_bike_distance_l37_37886


namespace quadratic_inequality_cond_l37_37233

theorem quadratic_inequality_cond (a : ℝ) :
  (∀ x : ℝ, ax^2 - ax + 1 > 0) ↔ (0 < a ∧ a < 4) :=
sorry

end quadratic_inequality_cond_l37_37233


namespace last_three_digits_product_l37_37568

theorem last_three_digits_product (a b c : ℕ) 
  (h1 : (a + b) % 10 = c % 10) 
  (h2 : (b + c) % 10 = a % 10) 
  (h3 : (c + a) % 10 = b % 10) :
  (a * b * c) % 1000 = 250 ∨ (a * b * c) % 1000 = 500 ∨ (a * b * c) % 1000 = 750 ∨ (a * b * c) % 1000 = 0 := 
by
  sorry

end last_three_digits_product_l37_37568


namespace polynomial_roots_product_l37_37926

theorem polynomial_roots_product (a b : ℤ)
  (h1 : ∀ (r : ℝ), r^2 - r - 2 = 0 → r^3 - a * r - b = 0) : a * b = 6 := sorry

end polynomial_roots_product_l37_37926


namespace seq_20_eq_5_over_7_l37_37832

theorem seq_20_eq_5_over_7 :
  ∃ (a : ℕ → ℚ), 
    a 1 = 6 / 7 ∧ 
    (∀ n, (0 ≤ a n ∧ a n < 1) → 
      (a (n + 1) = if a n < 1 / 2 then 2 * a n else 2 * a n - 1)) ∧ 
    a 20 = 5 / 7 := 
sorry

end seq_20_eq_5_over_7_l37_37832


namespace kiran_has_105_l37_37071

theorem kiran_has_105 
  (R G K L : ℕ) 
  (ratio_rg : 6 * G = 7 * R)
  (ratio_gk : 6 * K = 15 * G)
  (R_value : R = 36) : 
  K = 105 :=
by
  sorry

end kiran_has_105_l37_37071


namespace inequality_solution_sets_equivalence_l37_37135

theorem inequality_solution_sets_equivalence
  (a b : ℝ)
  (h1 : (∀ x : ℝ, -3 < x ∧ x < 2 ↔ ax^2 - 5 * x + b > 0)) :
  (∀ x : ℝ, x < -1/3 ∨ x > 1/2 ↔ bx^2 - 5 * x + a > 0) :=
  sorry

end inequality_solution_sets_equivalence_l37_37135


namespace ellipse_a_plus_k_l37_37198

theorem ellipse_a_plus_k (f1 f2 p : Real × Real) (a b h k : Real) :
  f1 = (2, 0) →
  f2 = (-2, 0) →
  p = (5, 3) →
  (∀ x y, ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1) →
  a > 0 →
  b > 0 →
  h = 0 →
  k = 0 →
  a = (3 * Real.sqrt 2 + Real.sqrt 58) / 2 →
  a + k = (3 * Real.sqrt 2 + Real.sqrt 58) / 2 :=
by
  intros
  sorry

end ellipse_a_plus_k_l37_37198


namespace range_of_m_l37_37635

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, mx^2 + 2 * m * x + 1 > 0) ↔ (0 ≤ m ∧ m < 1) := by
  sorry

end range_of_m_l37_37635


namespace remainder_x5_3x3_2x2_x_2_div_x_minus_2_l37_37851

def polynomial (x : ℝ) : ℝ := x^5 + 3*x^3 + 2*x^2 + x + 2

theorem remainder_x5_3x3_2x2_x_2_div_x_minus_2 :
  polynomial 2 = 68 := 
by 
  sorry

end remainder_x5_3x3_2x2_x_2_div_x_minus_2_l37_37851


namespace vector_parallel_m_l37_37812

theorem vector_parallel_m {m : ℝ} (h : (2:ℝ) * m - (-1 * -1) = 0) : m = 1 / 2 := 
by
  sorry

end vector_parallel_m_l37_37812


namespace minimum_value_expression_l37_37291

theorem minimum_value_expression {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 1) : 
  (x / y + y) * (y / x + x) ≥ 4 :=
sorry

end minimum_value_expression_l37_37291


namespace initial_tomatoes_count_l37_37220

-- Definitions and conditions
def birds_eat_fraction : ℚ := 1/3
def tomatoes_left : ℚ := 14
def fraction_tomatoes_left : ℚ := 2/3

-- We want to prove the initial number of tomatoes
theorem initial_tomatoes_count (initial_tomatoes : ℚ) 
  (h1 : tomatoes_left = fraction_tomatoes_left * initial_tomatoes) : 
  initial_tomatoes = 21 := 
by
  -- skipping the proof for now
  sorry

end initial_tomatoes_count_l37_37220


namespace sum_of_possible_values_l37_37951

theorem sum_of_possible_values (x : ℝ) (h : |x - 5| + 2 = 4) :
  x = 7 ∨ x = 3 → x = 10 := 
by sorry

end sum_of_possible_values_l37_37951


namespace verify_total_amount_l37_37109

noncomputable def total_withdrawable_amount (a r : ℝ) : ℝ :=
  a / r * ((1 + r) ^ 5 - (1 + r))

theorem verify_total_amount (a r : ℝ) (h_r_nonzero : r ≠ 0) :
  total_withdrawable_amount a r = a / r * ((1 + r)^5 - (1 + r)) :=
by
  sorry

end verify_total_amount_l37_37109


namespace unique_perpendicular_line_through_point_l37_37454

-- Definitions of the geometric entities and their relationships
structure Point := (x : ℝ) (y : ℝ)

structure Line := (m : ℝ) (b : ℝ)

-- A function to check if a point lies on a given line
def point_on_line (P : Point) (l : Line) : Prop := P.y = l.m * P.x + l.b

-- A function to represent that a line is perpendicular to another line at a given point
def perpendicular_lines_at_point (P : Point) (l1 l2 : Line) : Prop :=
  l1.m = -(1 / l2.m) ∧ point_on_line P l1 ∧ point_on_line P l2

-- The statement to be proved
theorem unique_perpendicular_line_through_point (P : Point) (l : Line) (h : point_on_line P l) :
  ∃! l' : Line, perpendicular_lines_at_point P l' l :=
by
  sorry

end unique_perpendicular_line_through_point_l37_37454


namespace garrison_reinforcement_l37_37572

/-- A garrison has initial provisions for 2000 men for 65 days. 
    After 15 days, reinforcement arrives and the remaining provisions last for 20 more days. 
    The size of the reinforcement is 3000 men.  -/
theorem garrison_reinforcement (P : ℕ) (M1 M2 D1 D2 D3 R : ℕ) 
  (h1 : M1 = 2000) (h2 : D1 = 65) (h3 : D2 = 15) (h4 : D3 = 20) 
  (h5 : P = M1 * D1) (h6 : P - M1 * D2 = (M1 + R) * D3) : 
  R = 3000 := 
sorry

end garrison_reinforcement_l37_37572


namespace fifth_house_number_is_13_l37_37918

theorem fifth_house_number_is_13 (n : ℕ) (a₁ : ℕ) (h₀ : n ≥ 5) (h₁ : (a₁ + n - 1) * n = 117) (h₂ : ∀ i, 1 ≤ i ∧ i ≤ n -> (a₁ + 2 * (i - 1)) = 2*(i-1) + a₁) : 
  (a₁ + 2 * (5 - 1)) = 13 :=
by
  sorry

end fifth_house_number_is_13_l37_37918


namespace problem_statement_l37_37935

noncomputable def perimeter_rectangle 
  (a b c w : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (area_triangle : ℝ := (1/2) * a * b) 
  (area_rectangle : ℝ := area_triangle) 
  (l : ℝ := area_rectangle / w) : ℝ :=
2 * (w + l)

theorem problem_statement 
  (a b c w : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h_a : a = 9) 
  (h_b : b = 12) 
  (h_c : c = 15) 
  (h_w : w = 6) : 
  perimeter_rectangle a b c w h1 = 30 :=
by 
  sorry

end problem_statement_l37_37935


namespace baron_munchausen_not_lying_l37_37902

def sum_of_digits (n : Nat) : Nat := sorry

theorem baron_munchausen_not_lying :
  ∃ a b : Nat, a ≠ b ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 ∧ 
  (a < 10^10 ∧ 10^9 ≤ a) ∧ (b < 10^10 ∧ 10^9 ≤ b) ∧ 
  (a + sum_of_digits (a ^ 2) = b + sum_of_digits (b ^ 2)) :=
sorry

end baron_munchausen_not_lying_l37_37902


namespace triangle_altitude_angle_l37_37702

noncomputable def angle_between_altitudes (α : ℝ) : ℝ :=
if α ≤ 90 then α else 180 - α

theorem triangle_altitude_angle (α : ℝ) (hα : 0 < α ∧ α < 180) : 
  (angle_between_altitudes α = α ↔ α ≤ 90) ∧ (angle_between_altitudes α = 180 - α ↔ α > 90) := 
by
  sorry

end triangle_altitude_angle_l37_37702


namespace frac_difference_l37_37420

theorem frac_difference (m n : ℝ) (h : m^2 - n^2 = m * n) : (n / m) - (m / n) = -1 :=
sorry

end frac_difference_l37_37420


namespace calculate_angles_and_side_l37_37477

theorem calculate_angles_and_side (a b B : ℝ) (h_a : a = Real.sqrt 3) (h_b : b = Real.sqrt 2) (h_B : B = 45) :
  ∃ A C c, (A = 60 ∧ C = 75 ∧ c = (Real.sqrt 6 + Real.sqrt 2) / 2) ∨ (A = 120 ∧ C = 15 ∧ c = (Real.sqrt 6 - Real.sqrt 2) / 2) :=
by sorry

end calculate_angles_and_side_l37_37477


namespace cylinder_volume_transformation_l37_37394

-- Define the original volume of the cylinder
def original_volume (V: ℝ) := V = 5

-- Define the transformation of quadrupling the dimensions of the cylinder
def new_volume (V V': ℝ) := V' = 64 * V

-- The goal is to show that under these conditions, the new volume is 320 gallons
theorem cylinder_volume_transformation (V V': ℝ) (h: original_volume V) (h': new_volume V V'):
  V' = 320 :=
by
  -- Proof is left as an exercise
  sorry

end cylinder_volume_transformation_l37_37394


namespace ann_fare_90_miles_l37_37379

-- Define the conditions as given in the problem
def fare (distance : ℕ) : ℕ := 30 + distance * 2

-- Theorem statement
theorem ann_fare_90_miles : fare 90 = 210 := by
  sorry

end ann_fare_90_miles_l37_37379


namespace expand_polynomial_l37_37680

theorem expand_polynomial (t : ℝ) :
  (3 * t^3 - 2 * t^2 + t - 4) * (2 * t^2 - t + 3) = 6 * t^5 - 7 * t^4 + 5 * t^3 - 15 * t^2 + 7 * t - 12 :=
by sorry

end expand_polynomial_l37_37680


namespace compare_a_b_c_l37_37563

noncomputable def a := Real.sin (Real.pi / 5)
noncomputable def b := Real.logb (Real.sqrt 2) (Real.sqrt 3)
noncomputable def c := (1 / 4)^(2 / 3)

theorem compare_a_b_c : c < a ∧ a < b := by
  sorry

end compare_a_b_c_l37_37563


namespace mean_of_xyz_l37_37191

theorem mean_of_xyz (x y z : ℝ) (h1 : 9 * x + 3 * y - 5 * z = -4) (h2 : 5 * x + 2 * y - 2 * z = 13) : 
  (x + y + z) / 3 = 10 := 
sorry

end mean_of_xyz_l37_37191


namespace simplified_expression_l37_37265

def f (x : ℝ) : ℝ := 3 * x + 4
def g (x : ℝ) : ℝ := 2 * x - 1

theorem simplified_expression :
  (f (g (f 3))) / (g (f (g 3))) = 79 / 37 :=
by  sorry

end simplified_expression_l37_37265


namespace four_digit_div_by_14_l37_37444

theorem four_digit_div_by_14 (n : ℕ) (h₁ : 9450 + n < 10000) :
  (∃ k : ℕ, 9450 + n = 14 * k) ↔ (n = 8) := by
  sorry

end four_digit_div_by_14_l37_37444


namespace nina_total_money_l37_37530

def original_cost_widget (C : ℝ) : ℝ := C
def num_widgets_nina_can_buy_original (C : ℝ) : ℝ := 6
def num_widgets_nina_can_buy_reduced (C : ℝ) : ℝ := 8
def cost_reduction : ℝ := 1.5

theorem nina_total_money (C : ℝ) (hc : 6 * C = 8 * (C - cost_reduction)) : 
  6 * C = 36 :=
by
  sorry

end nina_total_money_l37_37530


namespace billy_total_problems_solved_l37_37884

theorem billy_total_problems_solved :
  ∃ (Q : ℕ), (3 * Q = 132) ∧ ((Q) + (2 * Q) + (3 * Q) = 264) :=
by
  sorry

end billy_total_problems_solved_l37_37884


namespace twice_plus_eight_lt_five_times_x_l37_37766

theorem twice_plus_eight_lt_five_times_x (x : ℝ) : 2 * x + 8 < 5 * x := 
sorry

end twice_plus_eight_lt_five_times_x_l37_37766


namespace baker_new_cakes_bought_l37_37116

variable (total_cakes initial_sold sold_more_than_bought : ℕ)

def new_cakes_bought (total_cakes initial_sold sold_more_than_bought : ℕ) : ℕ :=
  total_cakes - (initial_sold + sold_more_than_bought)

theorem baker_new_cakes_bought (total_cakes initial_sold sold_more_than_bought : ℕ) 
  (h1 : total_cakes = 170)
  (h2 : initial_sold = 78)
  (h3 : sold_more_than_bought = 47) :
  new_cakes_bought total_cakes initial_sold sold_more_than_bought = 78 :=
  sorry

end baker_new_cakes_bought_l37_37116


namespace maximize_f_l37_37232

open Nat

-- Define the combination function
def comb (n k : ℕ) : ℕ := choose n k

-- Define the probability function f(n)
def f (n : ℕ) : ℚ := 
  (comb n 2 * comb (100 - n) 8 : ℚ) / comb 100 10

-- Define the theorem to find the value of n that maximizes f(n)
theorem maximize_f : ∃ n : ℕ, 2 ≤ n ∧ n ≤ 92 ∧ (∀ m : ℕ, 2 ≤ m ∧ m ≤ 92 → f n ≥ f m) ∧ n = 20 :=
by
  sorry

end maximize_f_l37_37232


namespace largest_int_less_than_100_with_remainder_5_l37_37184

theorem largest_int_less_than_100_with_remainder_5 (n : ℕ) (h1 : n < 100) (h2 : n % 8 = 5) : n = 93 := 
sorry

end largest_int_less_than_100_with_remainder_5_l37_37184


namespace range_of_a_for_positive_f_l37_37786

-- Let the function \(f(x) = ax^2 - 2x + 2\)
def f (a x : ℝ) := a * x^2 - 2 * x + 2

-- Theorem: The range of the real number \( a \) such that \( f(x) > 0 \) for all \( x \) in \( 1 < x < 4 \) is \((\dfrac{1}{2}, +\infty)\)
theorem range_of_a_for_positive_f :
  { a : ℝ | ∀ x : ℝ, 1 < x ∧ x < 4 → f a x > 0 } = { a : ℝ | a > 1/2 } :=
sorry

end range_of_a_for_positive_f_l37_37786


namespace solve_for_a_l37_37607

def quadratic_has_roots (a x1 x2 : ℝ) : Prop :=
  x1 + x2 = a ∧ x1 * x2 = -6 * a^2

theorem solve_for_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : quadratic_has_roots a x1 x2) (h3 : x2 - x1 = 10) : a = 2 :=
by
  sorry

end solve_for_a_l37_37607


namespace monotonic_increasing_implies_range_l37_37167

open Real

noncomputable def f (x : ℝ) : ℝ := 1 / 2 * x ^ 2 + 2 * x - 2 * log x

theorem monotonic_increasing_implies_range (a : ℝ) :
  (∀ x > (0 : ℝ), deriv f x ≥ 0) → a ≤ 1 :=
  by 
  sorry

end monotonic_increasing_implies_range_l37_37167


namespace money_made_is_40_l37_37981

-- Definitions based on conditions
def BettysStrawberries : ℕ := 16
def MatthewsStrawberries : ℕ := BettysStrawberries + 20
def NataliesStrawberries : ℕ := MatthewsStrawberries / 2
def TotalStrawberries : ℕ := BettysStrawberries + MatthewsStrawberries + NataliesStrawberries
def JarsOfJam : ℕ := TotalStrawberries / 7
def MoneyMade : ℕ := JarsOfJam * 4

-- The theorem to prove
theorem money_made_is_40 : MoneyMade = 40 :=
by
  sorry

end money_made_is_40_l37_37981


namespace balloon_volume_safety_l37_37736

theorem balloon_volume_safety (p V : ℝ) (h_prop : p = 90 / V) (h_burst : p ≤ 150) : 0.6 ≤ V :=
by {
  sorry
}

end balloon_volume_safety_l37_37736


namespace part1_part2_l37_37183

variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h1 : a * sin A * sin B + b * cos A^2 = 4 / 3 * a)
variable (h2 : c^2 = a^2 + (1 / 4) * b^2)

theorem part1 : b = 4 / 3 * a := by sorry

theorem part2 : C = π / 3 := by sorry

end part1_part2_l37_37183


namespace g_of_2_eq_14_l37_37953

theorem g_of_2_eq_14 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 * x - 4) = 4 * x + 6) : g 2 = 14 := 
sorry

end g_of_2_eq_14_l37_37953


namespace distinct_factors_of_product_l37_37335

theorem distinct_factors_of_product (m a b d : ℕ) (hm : m ≥ 1) (ha : m^2 < a ∧ a < m^2 + m)
  (hb : m^2 < b ∧ b < m^2 + m) (hab : a ≠ b) (hd : d ∣ (a * b)) (hd_range: m^2 < d ∧ d < m^2 + m) :
  d = a ∨ d = b :=
sorry

end distinct_factors_of_product_l37_37335


namespace proportion_red_MMs_l37_37844

theorem proportion_red_MMs (R B : ℝ) (h1 : R + B = 1) 
  (h2 : R * (4 / 5) = B * (1 / 6)) :
  R = 5 / 29 :=
by
  sorry

end proportion_red_MMs_l37_37844


namespace inequality_solution_set_l37_37690

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 else -1

theorem inequality_solution_set :
  { x : ℝ | (x+1) * f x > 2 } = { x : ℝ | x < -3 } ∪ { x : ℝ | x > 1 } :=
by
  sorry

end inequality_solution_set_l37_37690


namespace donations_received_l37_37441

def profit : Nat := 960
def half_profit: Nat := profit / 2
def goal: Nat := 610
def extra: Nat := 180
def total_needed: Nat := goal + extra
def donations: Nat := total_needed - half_profit

theorem donations_received :
  donations = 310 := by
  -- Proof omitted
  sorry

end donations_received_l37_37441


namespace largest_value_of_x_l37_37172

theorem largest_value_of_x (x : ℝ) (hx : x / 3 + 1 / (7 * x) = 1 / 2) : 
  x = (21 + Real.sqrt 105) / 28 := 
sorry

end largest_value_of_x_l37_37172


namespace second_character_more_lines_l37_37919

theorem second_character_more_lines
  (C1 : ℕ) (S : ℕ) (T : ℕ) (X : ℕ)
  (h1 : C1 = 20)
  (h2 : C1 = S + 8)
  (h3 : T = 2)
  (h4 : S = 3 * T + X) :
  X = 6 :=
by
  -- proof can be filled in here
  sorry

end second_character_more_lines_l37_37919


namespace countMultiplesOf30Between900And27000_l37_37719

noncomputable def smallestPerfectSquareDivisibleBy30 : ℕ :=
  900

noncomputable def smallestPerfectCubeDivisibleBy30 : ℕ :=
  27000

theorem countMultiplesOf30Between900And27000 :
  let lower_bound := smallestPerfectSquareDivisibleBy30 / 30;
  let upper_bound := smallestPerfectCubeDivisibleBy30 / 30;
  upper_bound - lower_bound + 1 = 871 :=
  by
  let lower_bound := smallestPerfectSquareDivisibleBy30 / 30;
  let upper_bound := smallestPerfectCubeDivisibleBy30 / 30;
  show upper_bound - lower_bound + 1 = 871;
  sorry

end countMultiplesOf30Between900And27000_l37_37719


namespace ronald_next_roll_l37_37889

/-- Ronald's rolls -/
def rolls : List ℕ := [1, 3, 2, 4, 3, 5, 3, 4, 4, 2]

/-- Total number of rolls after the next roll -/
def total_rolls := rolls.length + 1

/-- The desired average of the rolls -/
def desired_average : ℕ := 3

/-- The sum Ronald needs to reach after the next roll to achieve the desired average -/
def required_sum : ℕ := desired_average * total_rolls

/-- Ronald's current sum of rolls -/
def current_sum : ℕ := List.sum rolls

/-- The next roll needed to achieve the desired average -/
def next_roll_needed : ℕ := required_sum - current_sum

theorem ronald_next_roll :
  next_roll_needed = 2 := by
  sorry

end ronald_next_roll_l37_37889


namespace words_per_page_l37_37936

theorem words_per_page (p : ℕ) (h1 : 150 * p ≡ 270 [MOD 221]) (h2 : p ≤ 120) : p = 107 :=
sorry

end words_per_page_l37_37936


namespace hyperbola_asymptote_slope_l37_37496

theorem hyperbola_asymptote_slope (m : ℝ) :
  (∀ x y : ℝ, mx^2 + y^2 = 1) →
  (∀ x y : ℝ, y = 2 * x) →
  m = -4 :=
by
  sorry

end hyperbola_asymptote_slope_l37_37496


namespace satisfy_inequality_l37_37529

theorem satisfy_inequality (x : ℤ) : 
  (3 * x - 5 ≤ 10 - 2 * x) ↔ (x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3) :=
sorry

end satisfy_inequality_l37_37529


namespace solution_interval_l37_37154

theorem solution_interval (x : ℝ) : 
  (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (6 < x ∧ x < 7) ∨ (7 < x) ↔ 
  ((x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0) := sorry

end solution_interval_l37_37154


namespace find_complex_z_modulus_of_z_l37_37419

open Complex

theorem find_complex_z (z : ℂ) (h1 : (z - (0 + 3 * I)).im = 0) (h2 : ((z - (0 + 5 * I)) / (2 - I)).re = 0) : 
    z = -1 + 3 * I := by 
  sorry

theorem modulus_of_z (z : ℂ) (h1 : (z - (0 + 3 * I)).im = 0) (h2 : ((z - (0 + 5 * I)) / (2 - I)).re = 0) : 
    Complex.abs (z / (1 - I)) = Real.sqrt 5 := by 
  sorry

end find_complex_z_modulus_of_z_l37_37419


namespace value_of_y_l37_37721

theorem value_of_y (x y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 24) : y = 96 :=
by
  sorry

end value_of_y_l37_37721


namespace integer_solutions_l37_37289

theorem integer_solutions :
  ∀ (m n : ℤ), (m^3 - n^3 = 2 * m * n + 8 ↔ (m = 2 ∧ n = 0) ∨ (m = 0 ∧ n = -2)) :=
by
  intros m n
  sorry

end integer_solutions_l37_37289


namespace cylinder_surface_area_proof_l37_37091

noncomputable def sphere_volume := (500 * Real.pi) / 3
noncomputable def cylinder_base_diameter := 8
noncomputable def cylinder_surface_area := 80 * Real.pi

theorem cylinder_surface_area_proof :
  ∀ (R : ℝ) (r h : ℝ), 
    (4 * Real.pi / 3) * R^3 = (500 * Real.pi) / 3 → -- sphere volume condition
    2 * r = cylinder_base_diameter →               -- base diameter condition
    r * r + (h / 2)^2 = R^2 →                      -- Pythagorean theorem (half height)
    2 * Real.pi * r * h + 2 * Real.pi * r^2 = cylinder_surface_area := -- surface area formula
by
  intros R r h sphere_vol_cond base_diameter_cond pythagorean_cond
  sorry

end cylinder_surface_area_proof_l37_37091


namespace geom_seq_sum_elems_l37_37010

theorem geom_seq_sum_elems (a : ℕ → ℝ) (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_sum1 : a 1 + a 2 = 40)
  (h_sum2 : a 3 + a 4 = 60) :
  a 7 + a 8 = 135 :=
sorry

end geom_seq_sum_elems_l37_37010


namespace avg_temp_in_october_l37_37717

theorem avg_temp_in_october (a A : ℝ)
  (h1 : 28 = a + A)
  (h2 : 18 = a - A)
  (x := 10)
  (temperature : ℝ := a + A * Real.cos (π / 6 * (x - 6))) :
  temperature = 20.5 :=
by
  sorry

end avg_temp_in_october_l37_37717


namespace negation_universal_proposition_l37_37552

theorem negation_universal_proposition :
  (¬ ∀ x : ℝ, Real.exp x > x) ↔ ∃ x : ℝ, Real.exp x ≤ x := 
by 
  sorry

end negation_universal_proposition_l37_37552


namespace problem1_problem2_l37_37168

noncomputable def problem1_solution1 : ℝ := (2 + Real.sqrt 6) / 2
noncomputable def problem1_solution2 : ℝ := (2 - Real.sqrt 6) / 2

theorem problem1 (x : ℝ) : 
  (2 * x ^ 2 - 4 * x - 1 = 0) ↔ (x = problem1_solution1 ∨ x = problem1_solution2) :=
by
  sorry

theorem problem2 : 
  (4 * (x + 2) ^ 2 - 9 * (x - 3) ^ 2 = 0) ↔ (x = 1 ∨ x = 13) :=
by
  sorry

end problem1_problem2_l37_37168


namespace number_of_laborers_l37_37378

-- Definitions based on conditions in the problem
def hpd := 140   -- Earnings per day for heavy equipment operators
def gpd := 90    -- Earnings per day for general laborers
def totalPeople := 35  -- Total number of people hired
def totalPayroll := 3950  -- Total payroll in dollars

-- Variables H and L for the number of operators and laborers
variables (H L : ℕ)

-- Conditions provided in mathematical problem
axiom equation1 : H + L = totalPeople
axiom equation2 : hpd * H + gpd * L = totalPayroll

-- Theorem statement: we want to prove that L = 19
theorem number_of_laborers : L = 19 :=
sorry

end number_of_laborers_l37_37378


namespace largest_integer_dividing_sum_of_5_consecutive_integers_l37_37969

theorem largest_integer_dividing_sum_of_5_consecutive_integers :
  ∀ (a : ℤ), ∃ (n : ℤ), n = 5 ∧ 5 ∣ ((a - 2) + (a - 1) + a + (a + 1) + (a + 2)) := by
  sorry

end largest_integer_dividing_sum_of_5_consecutive_integers_l37_37969


namespace find_ratio_l37_37340

variable {a : ℕ → ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, a n > 0 ∧ a (n+1) / a n = a 1 / a 0

def forms_arithmetic_sequence (a1 a3_half a2_times_two : ℝ) : Prop :=
  a3_half = (a1 + a2_times_two) / 2

theorem find_ratio (a : ℕ → ℝ) (h_geom : is_geometric_sequence a)
  (h_arith : forms_arithmetic_sequence (a 1) (1/2 * a 3) (2 * a 2)) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 :=
sorry

end find_ratio_l37_37340


namespace area_of_quadrilateral_l37_37130

/-- The area of the quadrilateral defined by the system of inequalities is 15/7. -/
theorem area_of_quadrilateral : 
  (∃ (x y : ℝ), 3 * x + 2 * y ≤ 6 ∧ x + 3 * y ≥ 3 ∧ x ≥ 0 ∧ y ≥ 0) →
  (∃ (area : ℝ), area = 15 / 7) :=
by
  sorry

end area_of_quadrilateral_l37_37130


namespace grandparents_gift_l37_37317

theorem grandparents_gift (june_stickers bonnie_stickers total_stickers : ℕ) (x : ℕ)
  (h₁ : june_stickers = 76)
  (h₂ : bonnie_stickers = 63)
  (h₃ : total_stickers = 189) :
  june_stickers + bonnie_stickers + 2 * x = total_stickers → x = 25 :=
by
  intros
  sorry

end grandparents_gift_l37_37317


namespace least_area_of_triangles_l37_37443

-- Define the points A, B, C, D of the unit square
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (1, 1)
def D : ℝ × ℝ := (0, 1)

-- Define the function s(M, N) as the least area of the triangles having their vertices in the set {A, B, C, D, M, N}
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

noncomputable def s (M N : ℝ × ℝ) : ℝ :=
  min (min (min (min (min (triangle_area A B M) (triangle_area A B N)) (triangle_area A C M)) (triangle_area A C N)) (min (triangle_area A D M) (triangle_area A D N)))
    (min (min (min (triangle_area B C M) (triangle_area B C N)) (triangle_area B D M)) (min (triangle_area B D N) (min (triangle_area C D M) (triangle_area C D N))))

-- Define the statement to prove
theorem least_area_of_triangles (M N : ℝ × ℝ)
  (hM : M.1 > 0 ∧ M.1 < 1 ∧ M.2 > 0 ∧ M.2 < 1)
  (hN : N.1 > 0 ∧ N.1 < 1 ∧ N.2 > 0 ∧ N.2 < 1)
  (hMN : (M ≠ A ∨ N ≠ A) ∧ (M ≠ B ∨ N ≠ B) ∧ (M ≠ C ∨ N ≠ C) ∧ (M ≠ D ∨ N ≠ D))
  : s M N ≤ 1 / 8 := 
sorry

end least_area_of_triangles_l37_37443


namespace loss_per_meter_is_five_l37_37481

def cost_price_per_meter : ℝ := 50
def total_meters_sold : ℝ := 400
def selling_price : ℝ := 18000

noncomputable def total_cost_price : ℝ := cost_price_per_meter * total_meters_sold
noncomputable def total_loss : ℝ := total_cost_price - selling_price
noncomputable def loss_per_meter : ℝ := total_loss / total_meters_sold

theorem loss_per_meter_is_five : loss_per_meter = 5 :=
by sorry

end loss_per_meter_is_five_l37_37481


namespace dream_star_games_l37_37334

theorem dream_star_games (x y : ℕ) 
  (h1 : x + y + 2 = 9)
  (h2 : 3 * x + y = 17) : 
  x = 5 ∧ y = 2 := 
by 
  sorry

end dream_star_games_l37_37334


namespace increasing_function_range_of_a_l37_37431

variable {f : ℝ → ℝ}

theorem increasing_function_range_of_a (a : ℝ) (h : ∀ x : ℝ, 3 * a * x^2 ≥ 0) : a > 0 :=
sorry

end increasing_function_range_of_a_l37_37431


namespace gray_area_l37_37171

-- Given conditions
def rect1_length : ℕ := 8
def rect1_width : ℕ := 10
def rect2_length : ℕ := 12
def rect2_width : ℕ := 9
def black_area : ℕ := 37

-- Define areas based on conditions
def area_rect1 : ℕ := rect1_length * rect1_width
def area_rect2 : ℕ := rect2_length * rect2_width
def white_area : ℕ := area_rect1 - black_area

-- Theorem to prove the area of the gray part
theorem gray_area : area_rect2 - white_area = 65 :=
by
  sorry

end gray_area_l37_37171


namespace Eli_saves_more_with_discount_A_l37_37310

-- Define the prices and discounts
def price_book : ℝ := 25
def discount_A (price : ℝ) : ℝ := price * 0.4
def discount_B : ℝ := 5

-- Define the cost calculations:
def cost_with_discount_A (price : ℝ) : ℝ := price + (price - discount_A price)
def cost_with_discount_B (price : ℝ) : ℝ := price + (price - discount_B)

-- Define the savings calculation:
def savings (cost_B : ℝ) (cost_A : ℝ) : ℝ := cost_B - cost_A

-- The main statement to prove:
theorem Eli_saves_more_with_discount_A :
  savings (cost_with_discount_B price_book) (cost_with_discount_A price_book) = 5 :=
by
  sorry

end Eli_saves_more_with_discount_A_l37_37310


namespace symmetric_line_eq_l37_37107

theorem symmetric_line_eq (x y : ℝ) :
    3 * x - 4 * y + 5 = 0 ↔ 3 * x + 4 * (-y) + 5 = 0 :=
sorry

end symmetric_line_eq_l37_37107


namespace sum_mod_7_remainder_l37_37437

def sum_to (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem sum_mod_7_remainder : (sum_to 140) % 7 = 0 :=
by
  sorry

end sum_mod_7_remainder_l37_37437


namespace arithmetic_mean_six_expressions_l37_37864

theorem arithmetic_mean_six_expressions (x : ℝ)
  (h : (x + 8 + 15 + 2 * x + 13 + 2 * x + 4 + 3 * x + 5) / 6 = 30) : x = 13.5 :=
by
  sorry

end arithmetic_mean_six_expressions_l37_37864


namespace num_integer_values_satisfying_condition_l37_37962

theorem num_integer_values_satisfying_condition : 
  ∃ s : Finset ℤ, (∀ x ∈ s, 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) ∧ s.card = 3 :=
by
  sorry

end num_integer_values_satisfying_condition_l37_37962


namespace line_circle_intersection_l37_37306

theorem line_circle_intersection (x y : ℝ) (h1 : 7 * x + 5 * y = 14) (h2 : x^2 + y^2 = 4) :
  ∃ (p q : ℝ), (7 * p + 5 * q = 14) ∧ (p^2 + q^2 = 4) ∧ (7 * p + 5 * q = 14) ∧ (p ≠ q) :=
sorry

end line_circle_intersection_l37_37306


namespace eight_hash_six_l37_37758

def op (r s : ℝ) : ℝ := sorry

axiom op_r_zero (r : ℝ): op r 0 = r + 1
axiom op_comm (r s : ℝ) : op r s = op s r
axiom op_r_add_one_s (r s : ℝ): op (r + 1) s = (op r s) + s + 2

theorem eight_hash_six : op 8 6 = 69 := 
by sorry

end eight_hash_six_l37_37758


namespace initial_students_count_l37_37025

variable (n T : ℕ)
variables (initial_average remaining_average dropped_score : ℚ)
variables (initial_students remaining_students : ℕ)

theorem initial_students_count :
  initial_average = 62.5 →
  remaining_average = 63 →
  dropped_score = 55 →
  T = initial_average * n →
  T - dropped_score = remaining_average * (n - 1) →
  n = 16 :=
by
  intros h_avg_initial h_avg_remaining h_dropped_score h_total h_total_remaining
  sorry

end initial_students_count_l37_37025


namespace find_multiplier_l37_37648

theorem find_multiplier (n x : ℤ) (h1: n = 12) (h2: 4 * n - 3 = (n - 7) * x) : x = 9 :=
by {
  sorry
}

end find_multiplier_l37_37648


namespace count_prime_numbers_in_sequence_l37_37373

theorem count_prime_numbers_in_sequence : 
  ∀ (k : Nat), (∃ n : Nat, 47 * (10^n * k + (10^(n-1) - 1) / 9) = 47) → k = 0 :=
  sorry

end count_prime_numbers_in_sequence_l37_37373


namespace inequality_sin_cos_l37_37149

theorem inequality_sin_cos 
  (a b : ℝ) (n : ℝ) (x : ℝ) 
  (ha : 0 < a) (hb : 0 < b) : 
  (a / (Real.sin x)^n) + (b / (Real.cos x)^n) ≥ (a^(2/(n+2)) + b^(2/(n+2)))^((n+2)/2) :=
sorry

end inequality_sin_cos_l37_37149


namespace correct_calculation_l37_37442

theorem correct_calculation (x : ℕ) (h : 954 - x = 468) : 954 + x = 1440 := by
  sorry

end correct_calculation_l37_37442


namespace jayson_age_l37_37549

/-- When Jayson is a certain age J, his dad is four times his age,
    and his mom is 2 years younger than his dad. Jayson's mom was
    28 years old when he was born. Prove that Jayson is 10 years old
    when his dad is four times his age. -/
theorem jayson_age {J : ℕ} (h1 : ∀ J, J > 0 → J * 4 < J + 4) 
                   (h2 : ∀ J, (4 * J - 2) = J + 28) 
                   (h3 : J - (4 * J - 28) = 0): 
                   J = 10 :=
by 
  sorry

end jayson_age_l37_37549


namespace classroom_students_l37_37767

theorem classroom_students (n : ℕ) (h1 : 20 < n ∧ n < 30) 
  (h2 : ∃ n_y : ℕ, n = 3 * n_y + 1) 
  (h3 : ∃ n_y' : ℕ, n = (4 * (n - 1)) / 3 + 1) :
  n = 25 := 
by sorry

end classroom_students_l37_37767


namespace integer_roots_sum_abs_eq_94_l37_37575

theorem integer_roots_sum_abs_eq_94 {a b c m : ℤ} :
  (∃ m, (x : ℤ) * (x : ℤ) * (x : ℤ) - 2013 * (x : ℤ) + m = 0 ∧ a + b + c = 0 ∧ ab + bc + ac = -2013) →
  |a| + |b| + |c| = 94 :=
sorry

end integer_roots_sum_abs_eq_94_l37_37575


namespace factor_t_sq_minus_64_l37_37968

def isDifferenceOfSquares (a b : Int) : Prop := a = b^2

theorem factor_t_sq_minus_64 (t : Int) (h : isDifferenceOfSquares 64 8) : (t^2 - 64) = (t - 8) * (t + 8) := by
  sorry

end factor_t_sq_minus_64_l37_37968


namespace max_zeros_consecutive_two_digit_product_l37_37728

theorem max_zeros_consecutive_two_digit_product :
  ∃ a b : ℕ, 10 ≤ a ∧ a < 100 ∧ b = a + 1 ∧ 10 ≤ b ∧ b < 100 ∧
  (∀ c, (c * 10) ∣ a * b → c ≤ 2) := 
  by
    sorry

end max_zeros_consecutive_two_digit_product_l37_37728


namespace convert_scientific_notation_l37_37458

theorem convert_scientific_notation (a : ℝ) (b : ℤ) (h : a = 6.03 ∧ b = 5) : a * 10^b = 603000 := by
  cases h with
  | intro ha hb =>
    rw [ha, hb]
    sorry

end convert_scientific_notation_l37_37458


namespace pascal_triangle_row10_sum_l37_37788

def pascal_triangle_row_sum (n : ℕ) : ℕ :=
  2 ^ n

theorem pascal_triangle_row10_sum : pascal_triangle_row_sum 10 = 1024 :=
by
  -- Proof will demonstrate that 2^10 = 1024
  sorry

end pascal_triangle_row10_sum_l37_37788


namespace prob_heads_even_correct_l37_37492

noncomputable def prob_heads_even (n : Nat) : ℝ :=
  if n = 0 then 1
  else (2 / 3) - (1 / 3) * prob_heads_even (n - 1)

theorem prob_heads_even_correct : 
  prob_heads_even 50 = (1 / 2) * (1 + (1 / 3 ^ 50)) :=
sorry

end prob_heads_even_correct_l37_37492


namespace range_of_m_if_p_range_of_m_if_p_and_q_l37_37156

variable (m : ℝ)

def proposition_p (m : ℝ) : Prop :=
  (3 - m > m - 1) ∧ (m - 1 > 0)

def proposition_q (m : ℝ) : Prop :=
  m^2 - 9 / 4 < 0

theorem range_of_m_if_p (m : ℝ) (hp : proposition_p m) : 1 < m ∧ m < 2 :=
  sorry

theorem range_of_m_if_p_and_q (m : ℝ) (hp : proposition_p m) (hq : proposition_q m) : 1 < m ∧ m < 3 / 2 :=
  sorry

end range_of_m_if_p_range_of_m_if_p_and_q_l37_37156


namespace pie_distribution_l37_37748

theorem pie_distribution (x y : ℕ) (h1 : x + y + 2 * x = 13) (h2 : x < y) (h3 : y < 2 * x) :
  x = 3 ∧ y = 4 ∧ 2 * x = 6 := by
  sorry

end pie_distribution_l37_37748


namespace area_of_annulus_l37_37358

section annulus
variables {R r x : ℝ}
variable (h1 : R > r)
variable (h2 : R^2 - r^2 = x^2)

theorem area_of_annulus (R r x : ℝ) (h1 : R > r) (h2 : R^2 - r^2 = x^2) : 
  π * R^2 - π * r^2 = π * x^2 :=
sorry

end annulus

end area_of_annulus_l37_37358


namespace find_t_given_V_S_l37_37841

variables (g V V0 S S0 a t : ℝ)

theorem find_t_given_V_S :
  (V = g * (t - a) + V0) →
  (S = (1 / 2) * g * (t - a) ^ 2 + V0 * (t - a) + S0) →
  t = a + (V - V0) / g :=
by
  intros h1 h2
  sorry

end find_t_given_V_S_l37_37841


namespace james_hours_to_work_l37_37180

theorem james_hours_to_work :
  let meat_cost := 20 * 5
  let fruits_vegetables_cost := 15 * 4
  let bread_cost := 60 * 1.5
  let janitorial_cost := 10 * (10 * 1.5)
  let total_cost := meat_cost + fruits_vegetables_cost + bread_cost + janitorial_cost
  let hourly_wage := 8
  let hours_to_work := total_cost / hourly_wage
  hours_to_work = 50 :=
by 
  sorry

end james_hours_to_work_l37_37180


namespace count_multiples_4_or_9_but_not_both_l37_37351

theorem count_multiples_4_or_9_but_not_both (n : ℕ) (h : n = 200) :
  let count_multiples (k : ℕ) := (n / k)
  count_multiples 4 + count_multiples 9 - 2 * count_multiples 36 = 62 :=
by
  sorry

end count_multiples_4_or_9_but_not_both_l37_37351


namespace evaluate_f_5_minus_f_neg_5_l37_37986

def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x^3

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 1250 :=
by 
  sorry

end evaluate_f_5_minus_f_neg_5_l37_37986


namespace math_contest_students_l37_37895

theorem math_contest_students (n : ℝ) (h : n / 3 + n / 4 + n / 5 + 26 = n) : n = 120 :=
by {
    sorry
}

end math_contest_students_l37_37895


namespace factorize_expression_l37_37597

variable (a b : ℝ)

theorem factorize_expression : a^2 - 4 * b^2 - 2 * a + 4 * b = (a + 2 * b - 2) * (a - 2 * b) := 
  sorry

end factorize_expression_l37_37597


namespace sum_of_digits_is_base_6_l37_37947

def is_valid_digit (x : ℕ) : Prop := x > 0 ∧ x < 6 
def distinct_3 (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a  

theorem sum_of_digits_is_base_6 :
  ∃ (S H E : ℕ), is_valid_digit S ∧ is_valid_digit H ∧ is_valid_digit E
  ∧ distinct_3 S H E 
  ∧ (E + E) % 6 = S 
  ∧ (S + H) % 6 = E 
  ∧ (S + H + E) % 6 = 11 % 6 :=
by 
  sorry

end sum_of_digits_is_base_6_l37_37947


namespace polynomial_expansion_correct_l37_37830

open Polynomial

noncomputable def poly1 : Polynomial ℤ := X^2 + 3 * X - 4
noncomputable def poly2 : Polynomial ℤ := 2 * X^2 - X + 5
noncomputable def expected : Polynomial ℤ := 2 * X^4 + 5 * X^3 - 6 * X^2 + 19 * X - 20

theorem polynomial_expansion_correct :
  poly1 * poly2 = expected :=
sorry

end polynomial_expansion_correct_l37_37830


namespace marbles_left_l37_37960

theorem marbles_left (red_marble_count blue_marble_count broken_marble_count : ℕ)
  (h1 : red_marble_count = 156)
  (h2 : blue_marble_count = 267)
  (h3 : broken_marble_count = 115) :
  red_marble_count + blue_marble_count - broken_marble_count = 308 :=
by
  sorry

end marbles_left_l37_37960


namespace find_ratio_l37_37937

variables {EF GH EH EG EQ ER ES Q R S : ℝ}
variables (x : ℝ)
variables (E F G H : ℝ)

-- Conditions
def is_parallelogram : Prop := 
  -- Placeholder for parallelogram properties, not relevant for this example
  true

def point_on_segment (Q R : ℝ) (segment_length: ℝ) (ratio: ℝ): Prop := Q = segment_length * ratio ∧ R = segment_length * ratio

def intersect (EG QR : ℝ) (S : ℝ): Prop := 
  -- Placeholder for segment intersection properties, not relevant for this example
  true

-- Question
theorem find_ratio 
  (H_parallelogram: is_parallelogram)
  (H_pointQ: point_on_segment EQ ER EF (1/8))
  (H_pointR: point_on_segment ER ES EH (1/9))
  (H_intersection: intersect EG QR ES):
  (ES / EG) = (1/9) := 
by
  sorry

end find_ratio_l37_37937


namespace fractional_to_decimal_l37_37035

theorem fractional_to_decimal : (7 / 16 : ℚ) = 0.4375 := by
  sorry

end fractional_to_decimal_l37_37035


namespace arithmetic_sequence_diff_l37_37686

theorem arithmetic_sequence_diff (a : ℕ → ℝ)
  (h1 : a 5 * a 7 = 6)
  (h2 : a 2 + a 10 = 5) :
  a 10 - a 6 = 2 ∨ a 10 - a 6 = -2 := by
  sorry

end arithmetic_sequence_diff_l37_37686


namespace cost_of_steel_ingot_l37_37301

theorem cost_of_steel_ingot :
  ∃ P : ℝ, 
    (∃ initial_weight : ℝ, initial_weight = 60) ∧
    (∃ weight_increase_percentage : ℝ, weight_increase_percentage = 0.6) ∧
    (∃ ingot_weight : ℝ, ingot_weight = 2) ∧
    (weight_needed = initial_weight * weight_increase_percentage) ∧
    (number_of_ingots = weight_needed / ingot_weight) ∧
    (number_of_ingots > 10) ∧
    (discount_percentage = 0.2) ∧
    (total_cost = 72) ∧
    (discounted_price_per_ingot = P * (1 - discount_percentage)) ∧
    (total_cost = discounted_price_per_ingot * number_of_ingots) ∧
    P = 5 := 
by
  sorry

end cost_of_steel_ingot_l37_37301


namespace solve_for_x_l37_37934

theorem solve_for_x : 
  ∀ (x : ℝ), (∀ (a b : ℝ), a * b = 4 * a - 2 * b) → (3 * (6 * x) = -2) → (x = 17 / 2) :=
by
  sorry

end solve_for_x_l37_37934


namespace tan_alpha_solution_l37_37838

variable (α : ℝ)
variable (h₀ : 0 < α ∧ α < π)
variable (h₁ : Real.sin α + Real.cos α = 7 / 13)

theorem tan_alpha_solution : Real.tan α = -12 / 5 := 
by
  sorry

end tan_alpha_solution_l37_37838


namespace jackson_volume_discount_l37_37876

-- Given conditions as parameters
def hotTubVolume := 40 -- gallons
def quartsPerGallon := 4 -- quarts per gallon
def bottleVolume := 1 -- quart per bottle
def bottleCost := 50 -- dollars per bottle
def totalSpent := 6400 -- dollars spent by Jackson

-- Calculation related definitions
def totalQuarts := hotTubVolume * quartsPerGallon
def totalBottles := totalQuarts / bottleVolume
def costWithoutDiscount := totalBottles * bottleCost
def discountAmount := costWithoutDiscount - totalSpent
def discountPercentage := (discountAmount / costWithoutDiscount) * 100

-- The proof problem
theorem jackson_volume_discount : discountPercentage = 20 :=
by
  sorry

end jackson_volume_discount_l37_37876


namespace sum_mod_15_l37_37803

theorem sum_mod_15 
  (d e f : ℕ) 
  (hd : d % 15 = 11)
  (he : e % 15 = 12)
  (hf : f % 15 = 13) : 
  (d + e + f) % 15 = 6 :=
by
  sorry

end sum_mod_15_l37_37803


namespace min_squared_sum_l37_37649

theorem min_squared_sum {x y z : ℝ} (h : 2 * x + y + 2 * z = 6) : x^2 + y^2 + z^2 ≥ 4 :=
sorry

end min_squared_sum_l37_37649


namespace amber_max_ounces_l37_37021

theorem amber_max_ounces :
  ∀ (money : ℝ) (candy_cost : ℝ) (candy_ounces : ℝ) (chips_cost : ℝ) (chips_ounces : ℝ),
    money = 7 →
    candy_cost = 1 →
    candy_ounces = 12 →
    chips_cost = 1.4 →
    chips_ounces = 17 →
    max (money / candy_cost * candy_ounces) (money / chips_cost * chips_ounces) = 85 :=
by
  intros money candy_cost candy_ounces chips_cost chips_ounces
  intros h_money h_candy_cost h_candy_ounces h_chips_cost h_chips_ounces
  sorry

end amber_max_ounces_l37_37021


namespace viewing_spot_coordinate_correct_l37_37772

-- Define the coordinates of the landmarks
def first_landmark := 150
def second_landmark := 450

-- The expected coordinate of the viewing spot
def expected_viewing_spot := 350

-- The theorem that formalizes the problem
theorem viewing_spot_coordinate_correct :
  let distance := second_landmark - first_landmark
  let fractional_distance := (2 / 3) * distance
  let viewing_spot := first_landmark + fractional_distance
  viewing_spot = expected_viewing_spot := 
by
  -- This is where the proof would go
  sorry

end viewing_spot_coordinate_correct_l37_37772


namespace directrix_parabola_l37_37560

theorem directrix_parabola (y : ℝ → ℝ) (h : ∀ x, y x = 8 * x^2 + 5) : 
  ∃ c : ℝ, ∀ x, y x = 8 * x^2 + 5 ∧ c = 159 / 32 :=
by
  use 159 / 32
  repeat { sorry }

end directrix_parabola_l37_37560


namespace total_students_after_new_classes_l37_37646

def initial_classes : ℕ := 15
def students_per_class : ℕ := 20
def new_classes : ℕ := 5

theorem total_students_after_new_classes :
  initial_classes * students_per_class + new_classes * students_per_class = 400 :=
by
  sorry

end total_students_after_new_classes_l37_37646


namespace graph_is_hyperbola_l37_37282

theorem graph_is_hyperbola : 
  ∀ x y : ℝ, (x + y)^2 = x^2 + y^2 + 4 ↔ x * y = 2 := 
by
  sorry

end graph_is_hyperbola_l37_37282


namespace problem1_problem2_problem3_problem4_l37_37561

theorem problem1 : 0.175 / 0.25 / 4 = 0.175 := by
  sorry

theorem problem2 : 1.4 * 99 + 1.4 = 140 := by 
  sorry

theorem problem3 : 3.6 / 4 - 1.2 * 6 = -6.3 := by
  sorry

theorem problem4 : (3.2 + 0.16) / 0.8 = 4.2 := by
  sorry

end problem1_problem2_problem3_problem4_l37_37561


namespace arithmetic_sequence_sum_l37_37614

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (h_arith_seq: ∀ n: ℕ, S n = S 0 + n * (S 1 - S 0)) 
  (h5 : S 5 = 10) (h10 : S 10 = 30) : S 15 = 60 :=
by
  sorry

end arithmetic_sequence_sum_l37_37614


namespace unique_solution_eq_l37_37238

theorem unique_solution_eq (x : ℝ) : 
  (x ≠ 0 ∧ x ≠ 5) ∧ (∀ x, (3 * x^3 - 15 * x^2) / (x^2 - 5 * x) = x - 2) 
  → ∃! (x : ℝ), (3 * x ^ 3 - 15 * x ^ 2) / (x^2 - 5 * x) = x - 2 := 
by sorry

end unique_solution_eq_l37_37238


namespace A_investment_amount_l37_37298

theorem A_investment_amount
  (B_investment : ℝ) (C_investment : ℝ) 
  (total_profit : ℝ) (A_profit_share : ℝ)
  (h1 : B_investment = 4200)
  (h2 : C_investment = 10500)
  (h3 : total_profit = 14200)
  (h4 : A_profit_share = 4260) :
  ∃ (A_investment : ℝ), 
    A_profit_share / total_profit = A_investment / (A_investment + B_investment + C_investment) ∧ 
    A_investment = 6600 :=
by {
  sorry  -- Proof not required per instructions
}

end A_investment_amount_l37_37298


namespace Joe_total_time_correct_l37_37890

theorem Joe_total_time_correct :
  ∀ (distance : ℝ) (walk_rate : ℝ) (bike_rate : ℝ) (walk_time bike_time : ℝ),
    (walk_time = 9) →
    (bike_rate = 5 * walk_rate) →
    (walk_rate * walk_time = distance / 3) →
    (bike_rate * bike_time = 2 * distance / 3) →
    (walk_time + bike_time = 12.6) := 
by
  intros distance walk_rate bike_rate walk_time bike_time
  intro walk_time_cond
  intro bike_rate_cond
  intro walk_distance_cond
  intro bike_distance_cond
  sorry

end Joe_total_time_correct_l37_37890


namespace complement_A_inter_B_range_of_a_l37_37806

open Set

-- Define sets A and B based on the conditions
def A : Set ℝ := {x | -4 ≤ x - 6 ∧ x - 6 ≤ 0}
def B : Set ℝ := {x | 2 * x - 6 ≥ 3 - x}

-- Define set C based on the conditions
def C (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Problem 1: Prove the complement of (A ∩ B) in ℝ is the set of x where (x < 3 or x > 6)
theorem complement_A_inter_B :
  compl (A ∩ B) = {x | x < 3} ∪ {x | x > 6} :=
sorry

-- Problem 2: Prove that A ∩ C = A implies a ∈ [6, ∞)
theorem range_of_a {a : ℝ} (hC : A ∩ C a = A) :
  6 ≤ a :=
sorry

end complement_A_inter_B_range_of_a_l37_37806


namespace largest_y_coordinate_l37_37032

theorem largest_y_coordinate (x y : ℝ) : 
  (x^2 / 49 + (y - 3)^2 / 25 = 0) → y = 3 :=
by
  intro h
  -- This is where the proofs steps would go if required.
  sorry

end largest_y_coordinate_l37_37032


namespace max_value_y_l37_37224

theorem max_value_y (x : ℝ) : ∃ y, y = -3 * x^2 + 6 ∧ ∀ z, (∃ x', z = -3 * x'^2 + 6) → z ≤ y :=
by sorry

end max_value_y_l37_37224


namespace minimum_a_value_l37_37595

theorem minimum_a_value (a : ℝ) : 
  (∀ (x y : ℝ), 0 < x → 0 < y → x^2 + 2 * x * y ≤ a * (x^2 + y^2)) ↔ a ≥ (Real.sqrt 5 + 1) / 2 := 
sorry

end minimum_a_value_l37_37595


namespace total_selling_price_l37_37742

theorem total_selling_price (cost1 cost2 cost3 : ℕ) (profit1 profit2 profit3 : ℚ) 
  (h1 : cost1 = 280) (h2 : cost2 = 350) (h3 : cost3 = 500) 
  (h4 : profit1 = 30) (h5 : profit2 = 45) (h6 : profit3 = 25) : 
  (cost1 + (profit1 / 100) * cost1) + (cost2 + (profit2 / 100) * cost2) + (cost3 + (profit3 / 100) * cost3) = 1496.5 := by
  sorry

end total_selling_price_l37_37742


namespace sum_gn_eq_one_third_l37_37173

noncomputable def g (n : ℕ) : ℝ :=
  ∑' i : ℕ, if i ≥ 3 then 1 / (i ^ n) else 0

theorem sum_gn_eq_one_third :
  (∑' n : ℕ, if n ≥ 3 then g n else 0) = 1 / 3 := 
by sorry

end sum_gn_eq_one_third_l37_37173


namespace area_of_circle_l37_37794

theorem area_of_circle (r θ : ℝ) (h : r = 4 * Real.cos θ - 3 * Real.sin θ) :
  ∃ π : ℝ, π * (5/2)^2 = 25 * π / 4 :=
by 
  sorry

end area_of_circle_l37_37794


namespace angle_bisector_inequality_l37_37807

theorem angle_bisector_inequality {a b c fa fb fc : ℝ} 
  (h_triangle_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angle_bisectors : fa > 0 ∧ fb > 0 ∧ fc > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  (1 / fa + 1 / fb + 1 / fc > 1 / a + 1 / b + 1 / c) :=
by
  sorry

end angle_bisector_inequality_l37_37807


namespace expression_eval_l37_37497

noncomputable def a : ℕ := 2001
noncomputable def b : ℕ := 2003

theorem expression_eval : 
  b^3 - a * b^2 - a^2 * b + a^3 = 8 :=
by sorry

end expression_eval_l37_37497


namespace symmetric_function_cannot_be_even_l37_37682

noncomputable def f : ℝ → ℝ := sorry

theorem symmetric_function_cannot_be_even :
  (∀ x, f (f x) = x^2) ∧ (∀ x ≥ 0, f (x^2) = x) → ¬ (∀ x, f x = f (-x)) :=
by 
  intros
  sorry -- Proof is not required

end symmetric_function_cannot_be_even_l37_37682


namespace calculation_l37_37047

noncomputable def seq (n : ℕ) : ℕ → ℚ := sorry

axiom cond1 : ∀ (n : ℕ), seq (n + 1) - 2 * seq n = 0
axiom cond2 : ∀ (n : ℕ), seq n ≠ 0

theorem calculation :
  (2 * seq 1 + seq 2) / (seq 3 + seq 5) = 1 / 5 :=
  sorry

end calculation_l37_37047


namespace cos_value_third_quadrant_l37_37018

theorem cos_value_third_quadrant (x : Real) (h1 : Real.sin x = -1 / 3) (h2 : π < x ∧ x < 3 * π / 2) : 
  Real.cos x = -2 * Real.sqrt 2 / 3 :=
by
  sorry

end cos_value_third_quadrant_l37_37018


namespace abs_quadratic_eq_linear_iff_l37_37675

theorem abs_quadratic_eq_linear_iff (x : ℝ) : 
  (|x^2 - 5*x + 6| = x + 2) ↔ (x = 3 + Real.sqrt 5 ∨ x = 3 - Real.sqrt 5) :=
by
  sorry

end abs_quadratic_eq_linear_iff_l37_37675


namespace find_solns_to_eqn_l37_37998

theorem find_solns_to_eqn (x y z w : ℕ) :
  2^x * 3^y - 5^z * 7^w = 1 ↔ (x, y, z, w) = (1, 0, 0, 0) ∨ 
                                        (x, y, z, w) = (3, 0, 0, 1) ∨ 
                                        (x, y, z, w) = (1, 1, 1, 0) ∨ 
                                        (x, y, z, w) = (2, 2, 1, 1) := 
sorry -- Placeholder for the actual proof

end find_solns_to_eqn_l37_37998


namespace option_B_is_correct_l37_37042

-- Definitions and Conditions
variable {Line : Type} {Plane : Type}
variable (m n : Line) (α β γ : Plane)
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Conditions
axiom m_perp_β : perpendicular m β
axiom m_parallel_α : parallel m α

-- Statement to prove
theorem option_B_is_correct : perpendicular_planes α β :=
by
  sorry

end option_B_is_correct_l37_37042


namespace problem1_problem2_l37_37544

theorem problem1 (x1 x2 : ℝ) (h1 : |x1 - 2| < 1) (h2 : |x2 - 2| < 1) :
  (2 < x1 + x2 ∧ x1 + x2 < 6) ∧ |x1 - x2| < 2 :=
by
  sorry

theorem problem2 (x1 x2 : ℝ) (h1 : |x1 - 2| < 1) (h2 : |x2 - 2| < 1) (f : ℝ → ℝ) 
  (hf : ∀ x, f x = x^2 - x + 1) :
  |x1 - x2| < |f x1 - f x2| ∧ |f x1 - f x2| < 5 * |x1 - x2| :=
by
  sorry

end problem1_problem2_l37_37544


namespace elizabeth_net_profit_l37_37527

noncomputable section

def net_profit : ℝ :=
  let cost_bag_1 := 2.5
  let cost_bag_2 := 3.5
  let total_cost := 10 * cost_bag_1 + 10 * cost_bag_2
  let selling_price := 6.0
  let sold_bags_1_no_discount := 7 * selling_price
  let sold_bags_2_no_discount := 8 * selling_price
  let discount_1 := 0.2
  let discount_2 := 0.3
  let discounted_price_1 := selling_price * (1 - discount_1)
  let discounted_price_2 := selling_price * (1 - discount_2)
  let sold_bags_1_with_discount := 3 * discounted_price_1
  let sold_bags_2_with_discount := 2 * discounted_price_2
  let total_revenue := sold_bags_1_no_discount + sold_bags_2_no_discount + sold_bags_1_with_discount + sold_bags_2_with_discount
  total_revenue - total_cost

theorem elizabeth_net_profit : net_profit = 52.8 := by
  sorry

end elizabeth_net_profit_l37_37527


namespace probability_each_box_2_fruits_l37_37805

noncomputable def totalWaysToDistributePears : ℕ := (Nat.choose 8 4)
noncomputable def totalWaysToDistributeApples : ℕ := 5^6

noncomputable def case1 : ℕ := (Nat.choose 5 2) * (Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2))
noncomputable def case2 : ℕ := (Nat.choose 5 1) * (Nat.choose 4 2) * (Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1 * Nat.factorial 1))
noncomputable def case3 : ℕ := (Nat.choose 5 4) * (Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1))

noncomputable def totalFavorableDistributions : ℕ := case1 + case2 + case3
noncomputable def totalPossibleDistributions : ℕ := totalWaysToDistributePears * totalWaysToDistributeApples

noncomputable def probability : ℚ := (totalFavorableDistributions : ℚ) / totalPossibleDistributions * 100

theorem probability_each_box_2_fruits :
  probability = 0.74 := 
sorry

end probability_each_box_2_fruits_l37_37805


namespace polygon_is_quadrilateral_l37_37117

-- Problem statement in Lean 4
theorem polygon_is_quadrilateral 
  (n : ℕ) 
  (h₁ : (n - 2) * 180 = 360) :
  n = 4 :=
by
  sorry

end polygon_is_quadrilateral_l37_37117


namespace triangle_side_lengths_condition_l37_37892

noncomputable def f (x k : ℝ) : ℝ := (x^2 + k*x + 1) / (x^2 + x + 1)

theorem triangle_side_lengths_condition (k : ℝ) :
  (∀ x1 x2 x3 : ℝ, x1 > 0 → x2 > 0 → x3 > 0 →
    (f x1 k) + (f x2 k) > (f x3 k) ∧ (f x2 k) + (f x3 k) > (f x1 k) ∧ (f x3 k) + (f x1 k) > (f x2 k))
  ↔ (-1/2 ≤ k ∧ k ≤ 4) :=
by
  sorry

end triangle_side_lengths_condition_l37_37892


namespace days_to_finish_job_l37_37121

def work_rate_a_b : ℚ := 1 / 15
def work_rate_c : ℚ := 4 / 15
def combined_work_rate : ℚ := work_rate_a_b + work_rate_c

theorem days_to_finish_job (A B C : ℚ) (h1 : A + B = work_rate_a_b) (h2 : C = work_rate_c) :
  1 / (A + B + C) = 3 :=
by
  sorry

end days_to_finish_job_l37_37121


namespace prove_necessary_but_not_sufficient_l37_37698

noncomputable def necessary_but_not_sufficient_condition (m : ℝ) :=
  (∀ x : ℝ, x^2 + 2*x + m > 0) → (m > 0) ∧ ¬ (∀ x : ℝ, x^2 + 2*x + m > 0 → m <= 1)

theorem prove_necessary_but_not_sufficient
    (m : ℝ) :
    necessary_but_not_sufficient_condition m :=
by
  sorry

end prove_necessary_but_not_sufficient_l37_37698


namespace remainder_when_divided_by_five_l37_37509

theorem remainder_when_divided_by_five :
  let E := 1250 * 1625 * 1830 * 2075 + 245
  E % 5 = 0 := by
  sorry

end remainder_when_divided_by_five_l37_37509


namespace integer_root_of_P_l37_37287

def P (x : ℤ) : ℤ := x^3 - 4 * x^2 - 8 * x + 24 

theorem integer_root_of_P :
  (∃ x : ℤ, P x = 0) ∧ (∀ x : ℤ, P x = 0 → x = 2) :=
sorry

end integer_root_of_P_l37_37287


namespace annual_income_from_investment_l37_37240

theorem annual_income_from_investment
  (I : ℝ) (P : ℝ) (R : ℝ)
  (hI : I = 6800) (hP : P = 136) (hR : R = 0.60) :
  (I / P) * 100 * R = 3000 := by
  sorry

end annual_income_from_investment_l37_37240


namespace fraction_of_students_saying_dislike_actually_like_l37_37835

variables (total_students liking_disliking_students saying_disliking_like_students : ℚ)
          (fraction_like_dislike say_dislike : ℚ)
          (cond1 : 0.7 = liking_disliking_students / total_students) 
          (cond2 : 0.3 = (total_students - liking_disliking_students) / total_students)
          (cond3 : 0.3 * liking_disliking_students = saying_disliking_like_students)
          (cond4 : 0.8 * (total_students - liking_disliking_students) 
                    = say_dislike)

theorem fraction_of_students_saying_dislike_actually_like
    (total_students_eq: total_students = 100) : 
    fraction_like_dislike = 46.67 :=
by
  sorry

end fraction_of_students_saying_dislike_actually_like_l37_37835


namespace exists_divisor_for_all_f_values_l37_37718

theorem exists_divisor_for_all_f_values (f : ℕ → ℕ) (h_f_range : ∀ n, 1 < f n) (h_f_div : ∀ m n, f (m + n) ∣ f m + f n) :
  ∃ c : ℕ, c > 1 ∧ ∀ n, c ∣ f n := 
sorry

end exists_divisor_for_all_f_values_l37_37718


namespace financial_loss_example_l37_37550

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ := 
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := 
  P * (1 + r * t)

theorem financial_loss_example :
  let P := 10000
  let r1 := 0.06
  let r2 := 0.05
  let t := 3
  let n := 4
  let A1 := compound_interest P r1 n t
  let A2 := simple_interest P r2 t
  abs (A1 - A2 - 456.18) < 0.01 := by
    sorry

end financial_loss_example_l37_37550


namespace sum_of_squares_of_medians_l37_37057

-- Define the components of the triangle
variables (a b c : ℝ)

-- Define the medians of the triangle
variables (s_a s_b s_c : ℝ)

-- State the theorem
theorem sum_of_squares_of_medians (h1 : s_a^2 + s_b^2 + s_c^2 = (3 / 4) * (a^2 + b^2 + c^2)) : 
  s_a^2 + s_b^2 + s_c^2 = (3 / 4) * (a^2 + b^2 + c^2) :=
by {
  -- The proof goes here
  sorry
}

end sum_of_squares_of_medians_l37_37057


namespace negation_of_proposition_l37_37299

theorem negation_of_proposition :
  (¬ ∀ x > 0, x^2 + x ≥ 0) ↔ (∃ x > 0, x^2 + x < 0) :=
by 
  sorry

end negation_of_proposition_l37_37299


namespace sum_of_adjacent_cells_multiple_of_4_l37_37484

theorem sum_of_adjacent_cells_multiple_of_4 :
  ∃ (i j : ℕ) (a b : ℕ) (H₁ : i < 22) (H₂ : j < 22),
    let grid (i j : ℕ) : ℕ := -- define the function for grid indexing
      ((i * 22) + j + 1 : ℕ)
    ∃ (i1 j1 : ℕ) (H₁₁ : i1 = i ∨ i1 = i + 1 ∨ i1 = i - 1)
                   (H₁₂ : j1 = j ∨ j1 = j + 1 ∨ j1 = j - 1),
      a = grid i j ∧ b = grid i1 j1 ∧ (a + b) % 4 = 0 := sorry

end sum_of_adjacent_cells_multiple_of_4_l37_37484


namespace smallest_d_for_inverse_l37_37556

def g (x : ℝ) : ℝ := (x - 3) ^ 2 - 7

theorem smallest_d_for_inverse :
  ∃ d, (∀ x₁ x₂, d ≤ x₁ ∧ d ≤ x₂ ∧ g x₁ = g x₂ → x₁ = x₂) ∧ (∀ e, (∀ x₁ x₂, e ≤ x₁ ∧ e ≤ x₂ ∧ g x₁ = g x₂ → x₁ = x₂) → d ≤ e) ∧ d = 3 :=
by
  sorry

end smallest_d_for_inverse_l37_37556


namespace function_properties_l37_37113

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log (x + 3)

theorem function_properties :
  (∃ x : ℝ, f x = -1) = false ∧ 
  (∃ x_0 : ℝ, -1 < x_0 ∧ x_0 < 0 ∧ deriv f x_0 = 0) ∧ 
  (∀ x : ℝ, -3 < x → f x > -1 / 2) ∧ 
  (∃ x_0 : ℝ, -3 < x_0 ∧ ∀ x : ℝ, -3 < x → f x_0 ≤ f x) :=
by
  sorry

end function_properties_l37_37113


namespace part1_part2_l37_37929

open Real

noncomputable def f (x : ℝ) : ℝ := abs ((2 / 3) * x + 1)

theorem part1 (a : ℝ) : (∀ x, f x ≥ -abs x + a) → a ≤ 1 :=
sorry

theorem part2 (x y : ℝ) (h1 : abs (x + y + 1) ≤ 1 / 3) (h2 : abs (y - 1 / 3) ≤ 2 / 3) : 
  f x ≤ 7 / 9 :=
sorry

end part1_part2_l37_37929


namespace solution_of_equation_l37_37152

def solve_equation (x : ℚ) : Prop := 
  (x^2 + 3 * x + 4) / (x + 5) = x + 6

theorem solution_of_equation : solve_equation (-13/4) := 
by
  sorry

end solution_of_equation_l37_37152


namespace molecular_weight_calculation_l37_37523

/-- Define the molecular weight of the compound as 972 grams per mole. -/
def molecular_weight : ℕ := 972

/-- Define the number of moles as 9 moles. -/
def number_of_moles : ℕ := 9

/-- Define the total weight of the compound for the given number of moles. -/
def total_weight : ℕ := number_of_moles * molecular_weight

/-- Prove the total weight is 8748 grams. -/
theorem molecular_weight_calculation : total_weight = 8748 := by
  sorry

end molecular_weight_calculation_l37_37523


namespace suggestions_difference_l37_37435

def mashed_potatoes_suggestions : ℕ := 408
def pasta_suggestions : ℕ := 305
def bacon_suggestions : ℕ := 137
def grilled_vegetables_suggestions : ℕ := 213
def sushi_suggestions : ℕ := 137

theorem suggestions_difference :
  let highest := mashed_potatoes_suggestions
  let lowest := bacon_suggestions
  highest - lowest = 271 :=
by
  sorry

end suggestions_difference_l37_37435


namespace center_of_the_hyperbola_l37_37120

def hyperbola_eq (x y : ℝ) : Prop := 9 * x^2 - 54 * x - 36 * y^2 + 288 * y - 576 = 0

structure Point where
  x : ℝ
  y : ℝ

def center_of_hyperbola_is (p : Point) : Prop :=
  hyperbola_eq (p.x + 3) (p.y + 4)

theorem center_of_the_hyperbola :
  ∀ x y : ℝ, hyperbola_eq x y → center_of_hyperbola_is {x := 3, y := 4} :=
by
  intros x y h
  sorry

end center_of_the_hyperbola_l37_37120


namespace gain_percent_is_correct_l37_37868

noncomputable def gain_percent (CP SP : ℝ) : ℝ :=
  let gain := SP - CP
  (gain / CP) * 100

theorem gain_percent_is_correct :
  gain_percent 930 1210 = 30.11 :=
by
  sorry

end gain_percent_is_correct_l37_37868


namespace inequality_holds_l37_37197

theorem inequality_holds (a b c : ℝ) (h1 : a > b) (h2 : b > c) : (a - b) * |c - b| > 0 :=
sorry

end inequality_holds_l37_37197


namespace bill_score_l37_37330

theorem bill_score
  (J B S : ℕ)
  (h1 : B = J + 20)
  (h2 : B = S / 2)
  (h3 : J + B + S = 160) : 
  B = 45 := 
by 
  sorry

end bill_score_l37_37330


namespace not_difference_of_squares_2021_l37_37684

theorem not_difference_of_squares_2021:
  ¬ ∃ (a b : ℕ), (a > b) ∧ (a^2 - b^2 = 2021) :=
sorry

end not_difference_of_squares_2021_l37_37684


namespace estimate_probability_l37_37897

noncomputable def freq_20 : ℝ := 0.300
noncomputable def freq_50 : ℝ := 0.360
noncomputable def freq_100 : ℝ := 0.350
noncomputable def freq_300 : ℝ := 0.350
noncomputable def freq_500 : ℝ := 0.352
noncomputable def freq_1000 : ℝ := 0.351
noncomputable def freq_5000 : ℝ := 0.351

theorem estimate_probability : (|0.35 - ((freq_20 + freq_50 + freq_100 + freq_300 + freq_500 + freq_1000 + freq_5000) / 7)| < 0.01) :=
by sorry

end estimate_probability_l37_37897


namespace expressions_equal_when_a_plus_b_plus_c_eq_1_l37_37658

theorem expressions_equal_when_a_plus_b_plus_c_eq_1
  (a b c : ℝ) (h : a + b + c = 1) :
  a + b * c = (a + b) * (a + c) :=
sorry

end expressions_equal_when_a_plus_b_plus_c_eq_1_l37_37658


namespace mark_pond_depth_l37_37360

def depth_of_Peter_pond := 5

def depth_of_Mark_pond := 3 * depth_of_Peter_pond + 4

theorem mark_pond_depth : depth_of_Mark_pond = 19 := by
  sorry

end mark_pond_depth_l37_37360


namespace max_xy_l37_37294

theorem max_xy (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 4 * x^2 + 9 * y^2 + 3 * x * y = 30) :
  xy ≤ 2 :=
by
  sorry

end max_xy_l37_37294


namespace winner_is_Junsu_l37_37631

def Younghee_water_intake : ℝ := 1.4
def Jimin_water_intake : ℝ := 1.8
def Junsu_water_intake : ℝ := 2.1

theorem winner_is_Junsu : 
  Junsu_water_intake > Younghee_water_intake ∧ Junsu_water_intake > Jimin_water_intake :=
by sorry

end winner_is_Junsu_l37_37631


namespace trapezoid_perimeter_l37_37578

noncomputable def semiCircularTrapezoidPerimeter (x : ℝ) 
  (hx : 0 < x ∧ x < 8 * Real.sqrt 2) : ℝ :=
-((x^2) / 8) + 2 * x + 32

theorem trapezoid_perimeter 
  (x : ℝ) 
  (hx : 0 < x ∧ x < 8 * Real.sqrt 2)
  (r : ℝ) 
  (h_r : r = 8) 
  (AB : ℝ) 
  (h_AB : AB = 2 * r)
  (CD_on_circumference : true) :
  semiCircularTrapezoidPerimeter x hx = -((x^2) / 8) + 2 * x + 32 :=   
sorry

end trapezoid_perimeter_l37_37578


namespace bridge_length_l37_37165

theorem bridge_length 
  (train_length : ℕ) 
  (speed_km_hr : ℕ) 
  (cross_time_sec : ℕ) 
  (conversion_factor_num : ℕ) 
  (conversion_factor_den : ℕ)
  (expected_length : ℕ) 
  (speed_m_s : ℕ := speed_km_hr * conversion_factor_num / conversion_factor_den)
  (total_distance : ℕ := speed_m_s * cross_time_sec) :
  train_length = 150 →
  speed_km_hr = 45 →
  cross_time_sec = 30 →
  conversion_factor_num = 1000 →
  conversion_factor_den = 3600 →
  expected_length = 225 →
  total_distance - train_length = expected_length :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end bridge_length_l37_37165


namespace inequality_proof_l37_37517

theorem inequality_proof (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  (1 / (1 + a)^2) + (1 / (1 + b)^2) + (1 / (1 + c)^2) + (1 / (1 + d)^2) ≥ 1 :=
by
  sorry

end inequality_proof_l37_37517


namespace total_inflation_time_l37_37963

/-- 
  Assume a soccer ball takes 20 minutes to inflate.
  Alexia inflates 20 soccer balls.
  Ermias inflates 5 more balls than Alexia.
  Prove that the total time in minutes taken to inflate all the balls is 900 minutes.
-/
theorem total_inflation_time 
  (alexia_balls : ℕ) (ermias_balls : ℕ) (each_ball_time : ℕ)
  (h1 : alexia_balls = 20)
  (h2 : ermias_balls = alexia_balls + 5)
  (h3 : each_ball_time = 20) :
  (alexia_balls + ermias_balls) * each_ball_time = 900 :=
by
  sorry

end total_inflation_time_l37_37963


namespace div30k_929260_l37_37526

theorem div30k_929260 (k : ℕ) (h : 30^k ∣ 929260) : 3^k - k^3 = 1 := by
  sorry

end div30k_929260_l37_37526


namespace total_carpet_area_correct_l37_37537

-- Define dimensions of the rooms
def room1_width : ℝ := 12
def room1_length : ℝ := 15
def room2_width : ℝ := 7
def room2_length : ℝ := 9
def room3_width : ℝ := 10
def room3_length : ℝ := 11

-- Define the areas of the rooms
def room1_area : ℝ := room1_width * room1_length
def room2_area : ℝ := room2_width * room2_length
def room3_area : ℝ := room3_width * room3_length

-- Total carpet area
def total_carpet_area : ℝ := room1_area + room2_area + room3_area

-- The theorem to prove
theorem total_carpet_area_correct :
  total_carpet_area = 353 :=
sorry

end total_carpet_area_correct_l37_37537


namespace molecular_weight_X_l37_37413

theorem molecular_weight_X (Ba_weight : ℝ) (total_molecular_weight : ℝ) (X_weight : ℝ) 
  (h1 : Ba_weight = 137) 
  (h2 : total_molecular_weight = 171) 
  (h3 : total_molecular_weight - Ba_weight * 1 = 2 * X_weight) : 
  X_weight = 17 :=
by
  sorry

end molecular_weight_X_l37_37413


namespace perpendicular_lines_m_value_l37_37676

theorem perpendicular_lines_m_value
  (l1 : ∀ (x y : ℝ), x - 2 * y + 1 = 0)
  (l2 : ∀ (x y : ℝ), m * x + y - 3 = 0)
  (perpendicular : ∀ (m : ℝ) (l1_slope l2_slope : ℝ), l1_slope * l2_slope = -1) : 
  m = 2 :=
by
  sorry

end perpendicular_lines_m_value_l37_37676


namespace valid_pairs_l37_37836

theorem valid_pairs
  (x y : ℕ)
  (h_pos_x : x > 0)
  (h_pos_y : y > 0)
  (h_div : ∃ k : ℕ, k > 0 ∧ k * (2 * x + 7 * y) = 7 * x + 2 * y) :
  ∃ a : ℕ, a > 0 ∧ (x = a ∧ y = a ∨ x = 4 * a ∧ y = a ∨ x = 19 * a ∧ y = a) :=
by
  sorry

end valid_pairs_l37_37836


namespace train_speed_l37_37044

theorem train_speed (length : ℤ) (time : ℤ) 
  (h_length : length = 280) (h_time : time = 14) : 
  (length * 3600) / (time * 1000) = 72 := 
by {
  -- The proof would go here, this part is omitted as per instructions
  sorry
}

end train_speed_l37_37044


namespace geom_seq_identity_l37_37695

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∀ n, ∃ r, a (n+1) = r * a n

theorem geom_seq_identity (a : ℕ → ℝ) (r : ℝ) (h1 : geometric_sequence a) (h2 : a 2 + a 4 = 2) :
  a 1 * a 3 + 2 * a 2 * a 4 + a 3 * a 5 = 4 := 
  sorry

end geom_seq_identity_l37_37695


namespace existence_of_special_numbers_l37_37176

theorem existence_of_special_numbers :
  ∃ (N : Finset ℕ), N.card = 1998 ∧ 
  ∀ (a b : ℕ), a ∈ N → b ∈ N → a ≠ b → a * b ∣ (a - b)^2 :=
sorry

end existence_of_special_numbers_l37_37176


namespace Iris_shorts_l37_37448

theorem Iris_shorts :
  ∃ s, (3 * 10) + s * 6 + (4 * 12) = 90 ∧ s = 2 := 
by
  existsi 2
  sorry

end Iris_shorts_l37_37448


namespace ratio_of_square_sides_sum_l37_37979

theorem ratio_of_square_sides_sum 
  (h : (75:ℚ) / 128 = (75:ℚ) / 128) :
  let a := 5
  let b := 6
  let c := 16
  a + b + c = 27 :=
by
  -- Our goal is to show that the sum of a + b + c equals 27
  let a := 5
  let b := 6
  let c := 16
  have h1 : a + b + c = 27 := sorry
  exact h1

end ratio_of_square_sides_sum_l37_37979


namespace number_of_acceptable_ages_l37_37664

theorem number_of_acceptable_ages (avg_age : ℤ) (std_dev : ℤ) (a b : ℤ) (h_avg : avg_age = 10) (h_std : std_dev = 8)
    (h1 : a = avg_age - std_dev) (h2 : b = avg_age + std_dev) :
    b - a + 1 = 17 :=
by {
    sorry
}

end number_of_acceptable_ages_l37_37664


namespace minimum_number_of_tiles_l37_37012

-- Define the measurement conversion and area calculations.
def tile_width := 2
def tile_length := 6
def region_width_feet := 3
def region_length_feet := 4

-- Convert feet to inches.
def region_width_inches := region_width_feet * 12
def region_length_inches := region_length_feet * 12

-- Calculate areas.
def tile_area := tile_width * tile_length
def region_area := region_width_inches * region_length_inches

-- Lean 4 statement to prove the minimum number of tiles required.
theorem minimum_number_of_tiles : region_area / tile_area = 144 := by
  sorry

end minimum_number_of_tiles_l37_37012


namespace decreasing_function_iff_m_eq_2_l37_37813

theorem decreasing_function_iff_m_eq_2 
    (m : ℝ) : 
    (∀ x : ℝ, 0 < x → (m^2 - m - 1) * x^(-5*m - 3) < (m^2 - m - 1) * (x + 1)^(-5*m - 3)) ↔ m = 2 := 
sorry

end decreasing_function_iff_m_eq_2_l37_37813


namespace tom_initial_books_l37_37831

theorem tom_initial_books (B : ℕ) (h1 : B - 4 + 38 = 39) : B = 5 :=
by
  sorry

end tom_initial_books_l37_37831


namespace Ricciana_run_distance_l37_37271

def Ricciana_jump : ℕ := 4

def Margarita_run : ℕ := 18

def Margarita_jump (Ricciana_jump : ℕ) : ℕ := 2 * Ricciana_jump - 1

def Margarita_total_distance (Margarita_run Margarita_jump : ℕ) : ℕ := Margarita_run + Margarita_jump

def Ricciana_total_distance (Ricciana_run Ricciana_jump : ℕ) : ℕ := Ricciana_run + Ricciana_jump

theorem Ricciana_run_distance (R : ℕ) 
  (Ricciana_total : ℕ := R + Ricciana_jump) 
  (Margarita_total : ℕ := Margarita_run + Margarita_jump Ricciana_jump) 
  (h : Margarita_total = Ricciana_total + 1) : 
  R = 20 :=
by
  sorry

end Ricciana_run_distance_l37_37271


namespace quadratic_discriminant_l37_37799

noncomputable def discriminant (a b c : ℚ) : ℚ :=
  b^2 - 4 * a * c

theorem quadratic_discriminant :
  discriminant 6 (6 + 1/6) (1/6) = 1225 / 36 :=
by
  sorry

end quadratic_discriminant_l37_37799


namespace f_1_geq_25_l37_37820

-- Define the function f
def f (x : ℝ) (m : ℝ) : ℝ := 4 * x^2 - m * x + 5

-- State that f is increasing on the interval [-2, +∞)
def is_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x y : ℝ, -2 ≤ x → x ≤ y → f x m ≤ f y m

-- Prove that given the function is increasing on [-2, +∞),
-- then f(1) is at least 25.
theorem f_1_geq_25 (m : ℝ) (h : is_increasing_on_interval m) : f 1 m ≥ 25 :=
  sorry

end f_1_geq_25_l37_37820


namespace minimum_k_conditions_l37_37328

theorem minimum_k_conditions (k : ℝ) :
  (∀ (a b c : ℝ), a ≠ 0 → b ≠ 0 → c ≠ 0 → (|a - b| ≤ k ∨ |1/a - 1/b| ≤ k)) ↔ k = 3/2 :=
sorry

end minimum_k_conditions_l37_37328


namespace smallest_base_to_represent_124_with_three_digits_l37_37534

theorem smallest_base_to_represent_124_with_three_digits : 
  ∃ (b : ℕ), b^2 ≤ 124 ∧ 124 < b^3 ∧ ∀ c, (c^2 ≤ 124 ∧ 124 < c^3) → (5 ≤ c) :=
by
  sorry

end smallest_base_to_represent_124_with_three_digits_l37_37534


namespace num_lists_correct_l37_37688

def num_balls : ℕ := 18
def num_draws : ℕ := 4

theorem num_lists_correct : (num_balls ^ num_draws) = 104976 :=
by
  sorry

end num_lists_correct_l37_37688


namespace min_value_f_l37_37161

noncomputable def f (a b : ℝ) : ℝ := (1 / a^5 + a^5 - 2) * (1 / b^5 + b^5 - 2)

theorem min_value_f :
  ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → f a b ≥ (31^4 / 32^2) :=
by
  intros
  sorry

end min_value_f_l37_37161


namespace reflection_ray_equation_l37_37409

theorem reflection_ray_equation (x y : ℝ) : (y = 2 * x + 1) → (∃ (x' y' : ℝ), y' = x ∧ y = 2 * x' + 1 ∧ x - 2 * y - 1 = 0) :=
by
  intro h
  sorry

end reflection_ray_equation_l37_37409


namespace remainder_5_to_5_to_5_to_5_mod_1000_l37_37376

theorem remainder_5_to_5_to_5_to_5_mod_1000 : (5^(5^(5^5))) % 1000 = 125 :=
by {
  sorry
}

end remainder_5_to_5_to_5_to_5_mod_1000_l37_37376


namespace milk_water_mixture_initial_volume_l37_37932

theorem milk_water_mixture_initial_volume
  (M W : ℝ)
  (h1 : 2 * M = 3 * W)
  (h2 : 4 * M = 3 * (W + 58)) :
  M + W = 145 := by
  sorry

end milk_water_mixture_initial_volume_l37_37932


namespace evaluate_expression_l37_37022

theorem evaluate_expression :
  (18 : ℝ) / (14 * 5.3) = (1.8 : ℝ) / 7.42 :=
by
  sorry

end evaluate_expression_l37_37022


namespace necessary_but_not_sufficient_condition_l37_37305
open Locale

variables {l m : Line} {α β : Plane}

def perp (l : Line) (p : Plane) : Prop := sorry
def subset (l : Line) (p : Plane) : Prop := sorry
def parallel (p₁ p₂ : Plane) : Prop := sorry

theorem necessary_but_not_sufficient_condition (h1 : perp l α) (h2 : subset m β) (h3 : perp l m) :
  ∃ (α : Plane) (β : Plane), parallel α β ∧ (perp l α → perp l β) ∧ (parallel α β → perp l β)  :=
sorry

end necessary_but_not_sufficient_condition_l37_37305


namespace combined_tax_rate_l37_37268

theorem combined_tax_rate
  (Mork_income : ℝ)
  (Mindy_income : ℝ)
  (h1 : Mindy_income = 3 * Mork_income)
  (Mork_tax_rate : ℝ := 0.30)
  (Mindy_tax_rate : ℝ := 0.20) :
  (Mork_tax_rate * Mork_income + Mindy_tax_rate * Mindy_income) / (Mork_income + Mindy_income) * 100 = 22.5 :=
by
  sorry

end combined_tax_rate_l37_37268


namespace warehouse_bins_total_l37_37019

theorem warehouse_bins_total (x : ℕ) (h1 : 12 * 20 + x * 15 = 510) : 12 + x = 30 :=
by
  sorry

end warehouse_bins_total_l37_37019


namespace problem_l37_37871

theorem problem (w : ℝ) (h : 13 = 13 * w / (1 - w)) : w^2 = 1 / 4 :=
by
  sorry

end problem_l37_37871


namespace jigsaw_puzzle_pieces_l37_37254

theorem jigsaw_puzzle_pieces
  (P : ℝ)
  (h1 : ∃ P, P = 0.90 * P + 0.72 * 0.10 * P + 0.504 * 0.08 * P + 504)
  (h2 : 0.504 * P = 504) :
  P = 1000 :=
by
  sorry

end jigsaw_puzzle_pieces_l37_37254


namespace geometric_sequences_l37_37782

theorem geometric_sequences :
  ∃ (a q : ℝ) (a1 a2 a3 : ℕ → ℝ), 
    (∀ n, a1 n = a * (q - 2) ^ n) ∧ 
    (∀ n, a2 n = 2 * a * (q - 1) ^ n) ∧ 
    (∀ n, a3 n = 4 * a * q ^ n) ∧
    a = 1 ∧ q = 4 ∨ a = 192 / 31 ∧ q = 9 / 8 ∧
    (a + 2 * a + 4 * a = 84) ∧
    (a * (q - 2) + 2 * a * (q - 1) + 4 * a * q = 24) :=
sorry

end geometric_sequences_l37_37782


namespace contrapositive_l37_37853

theorem contrapositive (a b : ℝ) :
  (a > b → a^2 > b^2) → (a^2 ≤ b^2 → a ≤ b) :=
by
  intro h
  sorry

end contrapositive_l37_37853


namespace power_sum_ge_three_l37_37778

theorem power_sum_ge_three {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 3) :
  a ^ a + b ^ b + c ^ c ≥ 3 :=
by
  sorry

end power_sum_ge_three_l37_37778


namespace sequence_property_l37_37436

theorem sequence_property (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 9) : 7 * n * 15873 = n * 111111 :=
by sorry

end sequence_property_l37_37436


namespace inverse_89_mod_90_l37_37055

theorem inverse_89_mod_90 : (89 * 89) % 90 = 1 := 
by
  sorry -- proof goes here

end inverse_89_mod_90_l37_37055


namespace original_number_value_l37_37611

theorem original_number_value (x : ℝ) (h : 0 < x) (h_eq : 10^4 * x = 4 / x) : x = 0.02 :=
sorry

end original_number_value_l37_37611


namespace booknote_unique_elements_l37_37331

def booknote_string : String := "booknote"
def booknote_set : Finset Char := { 'b', 'o', 'k', 'n', 't', 'e' }

theorem booknote_unique_elements : booknote_set.card = 6 :=
by
  sorry

end booknote_unique_elements_l37_37331


namespace axis_of_symmetry_of_parabola_l37_37468

-- Definitions (from conditions):
def quadratic_equation (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def is_root_of_quadratic (a b c x : ℝ) : Prop := quadratic_equation a b c x = 0

-- Given conditions
variables {a b c : ℝ}
variable (h_a_nonzero : a ≠ 0)
variable (h_root1 : is_root_of_quadratic a b c 1)
variable (h_root2 : is_root_of_quadratic a b c 5)

-- Problem statement
theorem axis_of_symmetry_of_parabola : (3 : ℝ) = (1 + 5) / 2 :=
by
  -- proof omitted
  sorry

end axis_of_symmetry_of_parabola_l37_37468


namespace train_stoppage_time_l37_37237

theorem train_stoppage_time (speed_excluding_stoppages speed_including_stoppages : ℝ) 
(H1 : speed_excluding_stoppages = 54) 
(H2 : speed_including_stoppages = 36) : (18 / (54 / 60)) = 20 :=
by
  sorry

end train_stoppage_time_l37_37237


namespace Jenny_original_number_l37_37946

theorem Jenny_original_number (y : ℝ) (h : 10 * (y / 2 - 6) = 70) : y = 26 :=
by
  sorry

end Jenny_original_number_l37_37946


namespace ice_cream_tubs_eaten_l37_37028

-- Conditions
def number_of_pans := 2
def pieces_per_pan := 16
def percentage_eaten_second_pan := 0.75
def scoops_per_tub := 8
def scoops_per_guest := 2
def guests_not_eating_ala_mode := 4

-- Questions
def tubs_of_ice_cream_eaten : Nat :=
  sorry

theorem ice_cream_tubs_eaten :
  tubs_of_ice_cream_eaten = 6 := by
  sorry

end ice_cream_tubs_eaten_l37_37028


namespace max_number_of_rectangles_in_square_l37_37741

-- Definitions and conditions
def area_square (n : ℕ) : ℕ := 4 * n^2
def area_rectangle (n : ℕ) : ℕ := n + 1
def max_rectangles (n : ℕ) : ℕ := area_square n / area_rectangle n

-- Lean theorem statement for the proof problem
theorem max_number_of_rectangles_in_square (n : ℕ) (h : n ≥ 4) :
  max_rectangles n = 4 * (n - 1) :=
sorry

end max_number_of_rectangles_in_square_l37_37741


namespace find_a_l37_37673

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0 → 3*x + y + a = 0) →
  a = 1 :=
by
  sorry

end find_a_l37_37673


namespace aliyah_more_phones_l37_37910

theorem aliyah_more_phones (vivi_phones : ℕ) (phone_price : ℕ) (total_money : ℕ) (aliyah_more : ℕ) : 
  vivi_phones = 40 → 
  phone_price = 400 → 
  total_money = 36000 → 
  40 + 40 + aliyah_more = total_money / phone_price → 
  aliyah_more = 10 :=
sorry

end aliyah_more_phones_l37_37910


namespace value_of_x_if_additive_inverses_l37_37129

theorem value_of_x_if_additive_inverses (x : ℝ) 
  (h : 4 * x - 1 + (3 * x - 6) = 0) : x = 1 := by
sorry

end value_of_x_if_additive_inverses_l37_37129


namespace scoops_per_carton_l37_37901

-- Definitions for scoops required by everyone
def ethan_vanilla := 1
def ethan_chocolate := 1
def lucas_danny_connor_chocolate_each := 2
def lucas_danny_connor := 3
def olivia_vanilla := 1
def olivia_strawberry := 1
def shannon_vanilla := 2 * olivia_vanilla
def shannon_strawberry := 2 * olivia_strawberry

-- Definitions for total scoops taken
def total_vanilla_taken := ethan_vanilla + olivia_vanilla + shannon_vanilla
def total_chocolate_taken := ethan_chocolate + (lucas_danny_connor_chocolate_each * lucas_danny_connor)
def total_strawberry_taken := olivia_strawberry + shannon_strawberry
def total_scoops_taken := total_vanilla_taken + total_chocolate_taken + total_strawberry_taken

-- Definitions for remaining scoops and original total scoops
def remaining_scoops := 16
def original_scoops := total_scoops_taken + remaining_scoops

-- Definition for number of cartons
def total_cartons := 3

-- Proof goal: scoops per carton
theorem scoops_per_carton : original_scoops / total_cartons = 10 := 
by
  -- Add your proof steps here
  sorry

end scoops_per_carton_l37_37901


namespace find_inverse_modulo_l37_37410

theorem find_inverse_modulo :
  113 * 113 ≡ 1 [MOD 114] :=
by
  sorry

end find_inverse_modulo_l37_37410


namespace squirrels_more_than_nuts_l37_37140

theorem squirrels_more_than_nuts (squirrels nuts : ℕ) (h1 : squirrels = 4) (h2 : nuts = 2) : squirrels - nuts = 2 := by
  sorry

end squirrels_more_than_nuts_l37_37140


namespace graph_avoid_third_quadrant_l37_37948

theorem graph_avoid_third_quadrant (k : ℝ) : 
  (∀ x y : ℝ, y = (2 * k - 1) * x + k → ¬ (x < 0 ∧ y < 0)) ↔ 0 ≤ k ∧ k < (1 / 2) :=
by sorry

end graph_avoid_third_quadrant_l37_37948


namespace least_b_not_in_range_l37_37966

theorem least_b_not_in_range : ∃ b : ℤ, -10 = b ∧ ∀ x : ℝ, x^2 + b * x + 20 ≠ -10 :=
sorry

end least_b_not_in_range_l37_37966


namespace pets_remaining_is_correct_l37_37783

-- Definitions for the initial conditions and actions taken
def initial_puppies : Nat := 7
def initial_kittens : Nat := 6
def puppies_sold : Nat := 2
def kittens_sold : Nat := 3

-- Definition that calculates the remaining number of pets
def remaining_pets : Nat := initial_puppies + initial_kittens - (puppies_sold + kittens_sold)

-- The theorem to prove
theorem pets_remaining_is_correct : remaining_pets = 8 := by sorry

end pets_remaining_is_correct_l37_37783


namespace solve_inequality_l37_37652

theorem solve_inequality :
  ∀ x : ℝ, (x - 2) * (x + 2) < 5 ↔ -3 < x ∧ x < 3 :=
by
  sorry

end solve_inequality_l37_37652


namespace pizza_cost_l37_37074

theorem pizza_cost
  (P T : ℕ)
  (hT : T = 1)
  (h_total : 3 * P + 4 * T + 5 = 39) :
  P = 10 :=
by
  sorry

end pizza_cost_l37_37074


namespace prime_ge_7_divides_30_l37_37940

theorem prime_ge_7_divides_30 (p : ℕ) (hp : p ≥ 7) (hp_prime : Nat.Prime p) : 30 ∣ (p^2 - 1) := by
  sorry

end prime_ge_7_divides_30_l37_37940


namespace summation_problem_l37_37247

open BigOperators

theorem summation_problem : 
  (∑ i in Finset.range 50, ∑ j in Finset.range 75, 2 * (i + 1) + 3 * (j + 1) + (i + 1) * (j + 1)) = 4275000 :=
by
  sorry

end summation_problem_l37_37247


namespace Shawna_situps_l37_37118

theorem Shawna_situps :
  ∀ (goal_per_day : ℕ) (total_days : ℕ) (tuesday_situps : ℕ) (wednesday_situps : ℕ),
  goal_per_day = 30 →
  total_days = 3 →
  tuesday_situps = 19 →
  wednesday_situps = 59 →
  (goal_per_day * total_days) - (tuesday_situps + wednesday_situps) = 12 :=
by
  intros goal_per_day total_days tuesday_situps wednesday_situps
  sorry

end Shawna_situps_l37_37118


namespace sin_2alpha_pos_of_tan_alpha_pos_l37_37801

theorem sin_2alpha_pos_of_tan_alpha_pos (α : Real) (h : Real.tan α > 0) : Real.sin (2 * α) > 0 :=
sorry

end sin_2alpha_pos_of_tan_alpha_pos_l37_37801


namespace cone_volume_l37_37393

theorem cone_volume (S : ℝ) (h_S : S = 12 * Real.pi) (h_lateral : ∃ r : ℝ, S = 3 * Real.pi * r^2) :
    ∃ V : ℝ, V = (8 * Real.sqrt 3 * Real.pi / 3) :=
by
  sorry

end cone_volume_l37_37393


namespace total_journey_distance_l37_37826

theorem total_journey_distance
  (T : ℝ) (D : ℝ)
  (h1 : T = 20)
  (h2 : (D / 2) / 21 + (D / 2) / 24 = 20) :
  D = 448 :=
by
  sorry

end total_journey_distance_l37_37826


namespace inequality1_inequality2_l37_37174

variable (a b c d : ℝ)

theorem inequality1 : 
  (a + c)^2 * (b + d)^2 ≥ 2 * (a * b^2 * c + b * c^2 * d + c * d^2 * a + d * a^2 * b + 4 * a * b * c * d) :=
  sorry

theorem inequality2 : 
  (a + c)^2 * (b + d)^2 ≥ 4 * b * c * (c * d + d * a + a * b) :=
  sorry

end inequality1_inequality2_l37_37174


namespace nested_sqrt_eq_five_l37_37026

-- Define the infinite nested square root expression
def nested_sqrt : ℝ := sorry -- we assume the definition exists
-- Define the property it satisfies
theorem nested_sqrt_eq_five : nested_sqrt = 5 := by
  sorry

end nested_sqrt_eq_five_l37_37026


namespace terminal_side_quadrant_l37_37515

theorem terminal_side_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) :
  ∃ k : ℤ, (k % 2 = 0 ∧ (k * Real.pi + Real.pi / 4 < α / 2 ∧ α / 2 < k * Real.pi + Real.pi / 2)) ∨
           (k % 2 = 1 ∧ (k * Real.pi + 3 * Real.pi / 4 < α / 2 ∧ α / 2 < k * Real.pi + 5 * Real.pi / 4)) := sorry

end terminal_side_quadrant_l37_37515


namespace total_points_other_members_18_l37_37199

-- Definitions
def total_points (x : ℕ) (S : ℕ) (T : ℕ) (M : ℕ) (y : ℕ) :=
  S + T + M + y = x

def Sam_scored (x S : ℕ) := S = x / 3

def Taylor_scored (x T : ℕ) := T = 3 * x / 8

def Morgan_scored (M : ℕ) := M = 21

def other_members_scored (y : ℕ) := ∃ (a b c d e f g h : ℕ),
  a ≤ 3 ∧ b ≤ 3 ∧ c ≤ 3 ∧ d ≤ 3 ∧ e ≤ 3 ∧ f ≤ 3 ∧ g ≤ 3 ∧ h ≤ 3 ∧
  y = a + b + c + d + e + f + g + h

-- Theorem
theorem total_points_other_members_18 (x y S T M : ℕ) :
  Sam_scored x S → Taylor_scored x T → Morgan_scored M → total_points x S T M y → other_members_scored y → y = 18 :=
by
  intros hSam hTaylor hMorgan hTotal hOther
  sorry

end total_points_other_members_18_l37_37199


namespace no_factors_of_p_l37_37459

open Polynomial

noncomputable def p : Polynomial ℝ := X^4 - 4 * X^2 + 16
noncomputable def optionA : Polynomial ℝ := X^2 + 4
noncomputable def optionB : Polynomial ℝ := X + 2
noncomputable def optionC : Polynomial ℝ := X^2 - 4*X + 4
noncomputable def optionD : Polynomial ℝ := X^2 - 4

theorem no_factors_of_p (h : Polynomial ℝ) : h ≠ p / optionA ∧ h ≠ p / optionB ∧ h ≠ p / optionC ∧ h ≠ p / optionD := by
  sorry

end no_factors_of_p_l37_37459


namespace percent_savings_per_roll_l37_37357

theorem percent_savings_per_roll 
  (cost_case : ℕ := 900) -- In cents, equivalent to $9
  (cost_individual : ℕ := 100) -- In cents, equivalent to $1
  (num_rolls : ℕ := 12) :
  (cost_individual - (cost_case / num_rolls)) * 100 / cost_individual = 25 := 
sorry

end percent_savings_per_roll_l37_37357


namespace trader_marked_price_percentage_above_cost_price_l37_37608

theorem trader_marked_price_percentage_above_cost_price 
  (CP MP SP : ℝ) 
  (discount loss : ℝ)
  (h_discount : discount = 0.07857142857142857)
  (h_loss : loss = 0.01)
  (h_SP_discount : SP = MP * (1 - discount))
  (h_SP_loss : SP = CP * (1 - loss)) :
  (MP / CP - 1) * 100 = 7.4285714285714 := 
sorry

end trader_marked_price_percentage_above_cost_price_l37_37608


namespace max_a_value_l37_37771

theorem max_a_value (a : ℝ)
  (H : ∀ x : ℝ, (x - 1) * x - (a - 2) * (a + 1) ≥ 1) :
  a ≤ 3 / 2 := by
  sorry

end max_a_value_l37_37771


namespace add_fractions_l37_37326

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l37_37326


namespace prod_one_minus_nonneg_reals_ge_half_l37_37260

theorem prod_one_minus_nonneg_reals_ge_half (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : 0 ≤ x3)
  (h_sum : x1 + x2 + x3 ≤ 1/2) : 
  (1 - x1) * (1 - x2) * (1 - x3) ≥ 1 / 2 := 
by
  sorry

end prod_one_minus_nonneg_reals_ge_half_l37_37260


namespace max_frac_sum_l37_37370

theorem max_frac_sum (n a b c d : ℕ) (hn : 1 < n) (hab : 0 < a) (hcd : 0 < c)
    (hfrac : (a / b) + (c / d) < 1) (hsum : a + c ≤ n) :
    (∃ (b_val : ℕ), 2 ≤ b_val ∧ b_val ≤ n ∧ 
    1 - 1 / (b_val * (b_val * (n + 1 - b_val) + 1)) = 
    1 - 1 / ((2 * n / 3 + 7 / 6) * ((2 * n / 3 + 7 / 6) * (n - (2 * n / 3 + 1 / 6)) + 1))) :=
sorry

end max_frac_sum_l37_37370


namespace grill_cost_difference_l37_37204

theorem grill_cost_difference:
  let in_store_price : Float := 129.99
  let payment_per_installment : Float := 32.49
  let number_of_installments : Float := 4
  let shipping_handling : Float := 9.99
  let total_tv_cost : Float := (number_of_installments * payment_per_installment) + shipping_handling
  let cost_difference : Float := in_store_price - total_tv_cost
  cost_difference * 100 = -996 := by
    sorry

end grill_cost_difference_l37_37204


namespace new_pyramid_volume_l37_37128

/-- Given an original pyramid with volume 40 cubic inches, where the length is doubled, 
    the width is tripled, and the height is increased by 50%, 
    prove that the volume of the new pyramid is 360 cubic inches. -/
theorem new_pyramid_volume (V : ℝ) (l w h : ℝ) 
  (h_volume : V = 1 / 3 * l * w * h) 
  (h_original : V = 40) : 
  (2 * l) * (3 * w) * (1.5 * h) / 3 = 360 :=
by
  sorry

end new_pyramid_volume_l37_37128


namespace combined_weight_difference_l37_37048

-- Define the weights of the textbooks
def chemistry_weight : ℝ := 7.125
def geometry_weight : ℝ := 0.625
def calculus_weight : ℝ := 5.25
def biology_weight : ℝ := 3.75

-- Define the problem statement that needs to be proven
theorem combined_weight_difference :
  ((calculus_weight + biology_weight) - (chemistry_weight - geometry_weight)) = 2.5 :=
by
  sorry

end combined_weight_difference_l37_37048


namespace money_bounds_l37_37739

variables (c d : ℝ)

theorem money_bounds :
  (7 * c + d > 84) ∧ (5 * c - d = 35) → (c > 9.92 ∧ d > 14.58) :=
by
  intro h
  sorry

end money_bounds_l37_37739


namespace gnomes_remaining_in_ravenswood_l37_37106

theorem gnomes_remaining_in_ravenswood 
  (westerville_gnomes : ℕ)
  (ravenswood_initial_gnomes : ℕ)
  (taken_gnomes : ℕ)
  (remaining_gnomes : ℕ)
  (h1 : westerville_gnomes = 20)
  (h2 : ravenswood_initial_gnomes = 4 * westerville_gnomes)
  (h3 : taken_gnomes = (40 * ravenswood_initial_gnomes) / 100)
  (h4 : remaining_gnomes = ravenswood_initial_gnomes - taken_gnomes) :
  remaining_gnomes = 48 :=
by
  sorry

end gnomes_remaining_in_ravenswood_l37_37106


namespace find_b_l37_37104

variable (b : ℝ)

theorem find_b 
    (h₁ : 0 < b)
    (h₂ : b < 4)
    (area_ratio : ∃ k : ℝ, k = 4/16 ∧ (4 + b) / -b = 2 * k) :
  b = -4/3 :=
by
  sorry

end find_b_l37_37104


namespace find_polynomial_l37_37027

theorem find_polynomial (P : ℝ → ℝ) (h_poly : ∀ a b c : ℝ, ab + bc + ca = 0 → P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)) :
  ∃ r s : ℝ, ∀ x : ℝ, P x = r * x^4 + s * x^2 :=
sorry

end find_polynomial_l37_37027


namespace avg_page_count_per_essay_l37_37905

-- Definitions based on conditions
def students : ℕ := 15
def first_group_students : ℕ := 5
def second_group_students : ℕ := 5
def third_group_students : ℕ := 5
def pages_first_group : ℕ := 2
def pages_second_group : ℕ := 3
def pages_third_group : ℕ := 1
def total_pages := first_group_students * pages_first_group +
                   second_group_students * pages_second_group +
                   third_group_students * pages_third_group

-- Statement to prove
theorem avg_page_count_per_essay :
  total_pages / students = 2 :=
by
  sorry

end avg_page_count_per_essay_l37_37905


namespace age_of_15th_person_l37_37734

theorem age_of_15th_person (avg_16 : ℝ) (avg_5 : ℝ) (avg_9 : ℝ) (total_16 : ℝ) (total_5 : ℝ) (total_9 : ℝ) :
  avg_16 = 15 ∧ avg_5 = 14 ∧ avg_9 = 16 ∧
  total_16 = 16 * avg_16 ∧ total_5 = 5 * avg_5 ∧ total_9 = 9 * avg_9 →
  (total_16 - total_5 - total_9) = 26 :=
by
  sorry

end age_of_15th_person_l37_37734


namespace C_must_be_2_l37_37551

-- Define the given digits and their sum conditions
variables (A B C D : ℤ)

-- The sum of known digits for the first number
def sum1_known_digits := 7 + 4 + 5 + 2

-- The sum of known digits for the second number
def sum2_known_digits := 3 + 2 + 6 + 5

-- The first number must be divisible by 3
def divisible_by_3 (n : ℤ) : Prop := n % 3 = 0

-- Conditions for the divisibility by 3 of both numbers
def conditions := divisible_by_3 (sum1_known_digits + A + B + D) ∧ 
                  divisible_by_3 (sum2_known_digits + A + B + C)

-- The statement of the theorem
theorem C_must_be_2 (A B D : ℤ) (h : conditions A B 2 D) : C = 2 :=
  sorry

end C_must_be_2_l37_37551


namespace eval_expression_l37_37711

-- We define the expression that needs to be evaluated
def expression := (0.76)^3 - (0.1)^3 / (0.76)^2 + 0.076 + (0.1)^2

-- The statement to prove
theorem eval_expression : expression = 0.5232443982683983 :=
by
  sorry

end eval_expression_l37_37711


namespace floor_ceil_inequality_l37_37278

theorem floor_ceil_inequality 
  (a b c : ℝ)
  (h : ⌈a⌉ + ⌈b⌉ + ⌈c⌉ + ⌊a + b⌋ + ⌊b + c⌋ + ⌊c + a⌋ = 2020) :
  ⌊a⌋ + ⌊b⌋ + ⌊c⌋ + ⌈a + b + c⌉ ≥ 1346 := 
by
  sorry 

end floor_ceil_inequality_l37_37278


namespace spring_compression_l37_37060

theorem spring_compression (s F : ℝ) (h : F = 16 * s^2) (hF : F = 4) : s = 0.5 :=
by
  sorry

end spring_compression_l37_37060


namespace Berry_temperature_on_Sunday_l37_37470

theorem Berry_temperature_on_Sunday :
  let avg_temp := 99.0
  let days_in_week := 7
  let temp_day1 := 98.2
  let temp_day2 := 98.7
  let temp_day3 := 99.3
  let temp_day4 := 99.8
  let temp_day5 := 99.0
  let temp_day6 := 98.9
  let total_temp_week := avg_temp * days_in_week
  let total_temp_six_days := temp_day1 + temp_day2 + temp_day3 + temp_day4 + temp_day5 + temp_day6
  let temp_on_sunday := total_temp_week - total_temp_six_days
  temp_on_sunday = 98.1 :=
by
  -- Proof of the statement goes here
  sorry

end Berry_temperature_on_Sunday_l37_37470


namespace room_width_correct_l37_37518

noncomputable def length_of_room : ℝ := 5
noncomputable def total_cost_of_paving : ℝ := 21375
noncomputable def cost_per_square_meter : ℝ := 900
noncomputable def width_of_room : ℝ := 4.75

theorem room_width_correct :
  (total_cost_of_paving / cost_per_square_meter) = (length_of_room * width_of_room) :=
by
  sorry

end room_width_correct_l37_37518


namespace merchant_profit_after_discount_l37_37280

/-- A merchant marks his goods up by 40% and then offers a discount of 20% 
on the marked price. Prove that the merchant makes a profit of 12%. -/
theorem merchant_profit_after_discount :
  ∀ (CP MP SP : ℝ),
    CP > 0 →
    MP = CP * 1.4 →
    SP = MP * 0.8 →
    ((SP - CP) / CP) * 100 = 12 :=
by
  intros CP MP SP hCP hMP hSP
  sorry

end merchant_profit_after_discount_l37_37280


namespace ratio_of_running_to_swimming_l37_37249

variable (Speed_swimming Time_swimming Distance_total Speed_factor : ℕ)

theorem ratio_of_running_to_swimming :
  let Distance_swimming := Speed_swimming * Time_swimming
  let Distance_running := Distance_total - Distance_swimming
  let Speed_running := Speed_factor * Speed_swimming
  let Time_running := Distance_running / Speed_running
  (Distance_total = 12) ∧
  (Speed_swimming = 2) ∧
  (Time_swimming = 2) ∧
  (Speed_factor = 4) →
  (Time_running : ℕ) / Time_swimming = 1 / 2 :=
by
  intros
  sorry

end ratio_of_running_to_swimming_l37_37249


namespace parallel_line_through_intersection_perpendicular_line_through_intersection_l37_37322

/-- Given two lines l1: x + y - 4 = 0 and l2: x - y + 2 = 0,
the line passing through their intersection point and parallel to the line 2x - y - 1 = 0 
is 2x - y + 1 = 0 --/
theorem parallel_line_through_intersection :
  ∃ (c : ℝ), ∃ (x y : ℝ), (x + y - 4 = 0 ∧ x - y + 2 = 0) ∧ (2 * x - y + c = 0) ∧ c = 1 :=
by
  sorry

/-- Given two lines l1: x + y - 4 = 0 and l2: x - y + 2 = 0,
the line passing through their intersection point and perpendicular to the line 2x - y - 1 = 0
is x + 2y - 7 = 0 --/
theorem perpendicular_line_through_intersection :
  ∃ (d : ℝ), ∃ (x y : ℝ), (x + y - 4 = 0 ∧ x - y + 2 = 0) ∧ (x + 2 * y + d = 0) ∧ d = -7 :=
by
  sorry

end parallel_line_through_intersection_perpendicular_line_through_intersection_l37_37322


namespace new_ratio_first_term_less_than_implied_l37_37773

-- Define the original and new ratios
def original_ratio := (6, 7)
def subtracted_value := 3
def new_ratio := (original_ratio.1 - subtracted_value, original_ratio.2 - subtracted_value)

-- Prove the required property
theorem new_ratio_first_term_less_than_implied {r1 r2 : ℕ} (h : new_ratio = (3, 4))
  (h_less : r1 > 3) :
  new_ratio.1 < r1 := 
sorry

end new_ratio_first_term_less_than_implied_l37_37773


namespace tory_sold_each_toy_gun_for_l37_37546

theorem tory_sold_each_toy_gun_for :
  ∃ (x : ℤ), 8 * 18 = 7 * x + 4 ∧ x = 20 := 
by
  use 20
  constructor
  · sorry
  · sorry

end tory_sold_each_toy_gun_for_l37_37546


namespace set_equality_x_plus_y_l37_37846

theorem set_equality_x_plus_y (x y : ℝ) (A B : Set ℝ) (hA : A = {0, |x|, y}) (hB : B = {x, x * y, Real.sqrt (x - y)}) (h : A = B) : x + y = -2 :=
by
  sorry

end set_equality_x_plus_y_l37_37846


namespace clown_balloon_count_l37_37911

theorem clown_balloon_count (b1 b2 : ℕ) (h1 : b1 = 47) (h2 : b2 = 13) : b1 + b2 = 60 := by
  sorry

end clown_balloon_count_l37_37911


namespace perfect_cubes_count_l37_37190

theorem perfect_cubes_count : 
  Nat.card {n : ℕ | n^3 > 500 ∧ n^3 < 2000} = 5 :=
by
  sorry

end perfect_cubes_count_l37_37190


namespace adam_money_given_l37_37182

theorem adam_money_given (original_money : ℕ) (final_money : ℕ) (money_given : ℕ) :
  original_money = 79 →
  final_money = 92 →
  money_given = final_money - original_money →
  money_given = 13 := by
sorry

end adam_money_given_l37_37182


namespace revenue_percentage_change_l37_37535

theorem revenue_percentage_change (P S : ℝ) (hP : P > 0) (hS : S > 0) :
  let P_new := 1.30 * P
  let S_new := 0.80 * S
  let R := P * S
  let R_new := P_new * S_new
  (R_new - R) / R * 100 = 4 := by
  sorry

end revenue_percentage_change_l37_37535


namespace second_largest_div_second_smallest_l37_37808

theorem second_largest_div_second_smallest : 
  let a := 10
  let b := 11
  let c := 12
  ∃ second_smallest second_largest, 
    second_smallest = b ∧ second_largest = b ∧ second_largest / second_smallest = 1 := 
by
  let a := 10
  let b := 11
  let c := 12
  use b
  use b
  exact ⟨rfl, rfl, rfl⟩

end second_largest_div_second_smallest_l37_37808


namespace correct_system_of_equations_l37_37679

variable (x y : ℕ) -- We assume non-negative numbers for counts of chickens and rabbits

theorem correct_system_of_equations :
  (x + y = 35) ∧ (2 * x + 4 * y = 94) ↔
  (∃ (a b : ℕ), a = x ∧ b = y) :=
by
  sorry

end correct_system_of_equations_l37_37679


namespace portia_high_school_students_l37_37723

theorem portia_high_school_students
  (L P M : ℕ)
  (h1 : P = 4 * L)
  (h2 : M = 2 * L)
  (h3 : P + L + M = 4200) :
  P = 2400 :=
sorry

end portia_high_school_students_l37_37723


namespace hancho_height_l37_37638

theorem hancho_height (Hansol_height : ℝ) (h1 : Hansol_height = 134.5) (ratio : ℝ) (h2 : ratio = 1.06) :
  Hansol_height * ratio = 142.57 := by
  sorry

end hancho_height_l37_37638


namespace roots_sum_of_quadratic_l37_37349

theorem roots_sum_of_quadratic :
  ∀ x1 x2 : ℝ, (Polynomial.eval x1 (Polynomial.X ^ 2 + 2 * Polynomial.X - 1) = 0) →
              (Polynomial.eval x2 (Polynomial.X ^ 2 + 2 * Polynomial.X - 1) = 0) →
              x1 + x2 = -2 :=
by
  intros x1 x2 h1 h2
  sorry

end roots_sum_of_quadratic_l37_37349


namespace ninggao_intercity_project_cost_in_scientific_notation_l37_37659

theorem ninggao_intercity_project_cost_in_scientific_notation :
  let length_kilometers := 55
  let cost_per_kilometer_million := 140
  let total_cost_million := length_kilometers * cost_per_kilometer_million
  let total_cost_scientific := 7.7 * 10^6
  total_cost_million = total_cost_scientific := 
  sorry

end ninggao_intercity_project_cost_in_scientific_notation_l37_37659


namespace supermarket_flour_import_l37_37732

theorem supermarket_flour_import :
  let long_grain_rice := (9 : ℚ) / 20
  let glutinous_rice := (7 : ℚ) / 20
  let combined_rice := long_grain_rice + glutinous_rice
  let less_amount := (3 : ℚ) / 20
  let flour : ℚ := combined_rice - less_amount
  flour = (13 : ℚ) / 20 :=
by
  sorry

end supermarket_flour_import_l37_37732


namespace sequence_an_eq_n_l37_37574

theorem sequence_an_eq_n (a : ℕ → ℝ) (S : ℕ → ℝ) (h₀ : ∀ n, n ≥ 1 → a n > 0) 
  (h₁ : ∀ n, n ≥ 1 → a n + 1 / 2 = Real.sqrt (2 * S n + 1 / 4)) : 
  ∀ n, n ≥ 1 → a n = n := 
by
  sorry

end sequence_an_eq_n_l37_37574


namespace sum_even_integers_eq_930_l37_37311

theorem sum_even_integers_eq_930 :
  let sum_first_30_even := 2 * (30 * (30 + 1) / 2)
  let sum_consecutive_even (n : ℤ) := (n - 8) + (n - 6) + (n - 4) + (n - 2) + n
  ∀ n : ℤ, sum_first_30_even = 930 → sum_consecutive_even n = 930 → n = 190 :=
by
  intros sum_first_30_even sum_consecutive_even n h1 h2
  sorry

end sum_even_integers_eq_930_l37_37311


namespace age_difference_l37_37780

variables (F S M B : ℕ)

theorem age_difference:
  (F - S = 38) → (M - B = 36) → (F - M = 6) → (S - B = 4) :=
by
  intros h1 h2 h3
  -- Use the conditions to derive that S - B = 4
  sorry

end age_difference_l37_37780


namespace average_age_of_cricket_team_l37_37787

theorem average_age_of_cricket_team 
  (num_members : ℕ)
  (avg_age : ℕ)
  (wicket_keeper_age : ℕ)
  (remaining_avg : ℕ)
  (cond1 : num_members = 11)
  (cond2 : avg_age = 29)
  (cond3 : wicket_keeper_age = avg_age + 3)
  (cond4 : remaining_avg = avg_age - 1) : 
  avg_age = 29 := 
by 
  have h1 : num_members = 11 := cond1
  have h2 : avg_age = 29 := cond2
  have h3 : wicket_keeper_age = avg_age + 3 := cond3
  have h4 : remaining_avg = avg_age - 1 := cond4
  -- proof steps will go here
  sorry

end average_age_of_cricket_team_l37_37787


namespace net_price_change_l37_37730

theorem net_price_change (P : ℝ) : 
  let decreased_price := P * (1 - 0.30)
  let increased_price := decreased_price * (1 + 0.20)
  increased_price - P = -0.16 * P :=
by
  -- The proof would go here. We just need the statement as per the prompt.
  sorry

end net_price_change_l37_37730


namespace cos_sin_sequence_rational_l37_37498

variable (α : ℝ) (h₁ : ∃ r : ℚ, r = (Real.sin α + Real.cos α))

theorem cos_sin_sequence_rational :
    (∀ n : ℕ, n > 0 → ∃ r : ℚ, r = (Real.cos α)^n + (Real.sin α)^n) :=
by
  sorry

end cos_sin_sequence_rational_l37_37498


namespace sum_a1_a5_l37_37555

def sequence_sum (S : ℕ → ℕ) := ∀ n : ℕ, S n = n^2 + 1

theorem sum_a1_a5 (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h_sum : sequence_sum S)
  (h_a1 : a 1 = S 1)
  (h_a5 : a 5 = S 5 - S 4) :
  a 1 + a 5 = 11 := by
  sorry

end sum_a1_a5_l37_37555


namespace f_odd_f_monotonic_increasing_intervals_f_no_max_value_f_extreme_points_l37_37665

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem f_odd (x : ℝ) : f (-x) = -f (x) :=
by sorry

theorem f_monotonic_increasing_intervals :
  ∀ x : ℝ, (x < -Real.sqrt 3 / 3 ∨ x > Real.sqrt 3 / 3) → f x' > f x :=
by sorry

theorem f_no_max_value :
  ∀ x : ℝ, ¬(∃ M, f x ≤ M) :=
by sorry

theorem f_extreme_points :
  f (-Real.sqrt 3 / 3) = -2 * Real.sqrt 3 / 9 ∧ f (Real.sqrt 3 / 3) = -2 * Real.sqrt 3 / 9 :=
by sorry

end f_odd_f_monotonic_increasing_intervals_f_no_max_value_f_extreme_points_l37_37665


namespace largest_divisor_of_n4_minus_n_l37_37495

theorem largest_divisor_of_n4_minus_n (n : ℤ) (h : ∃ k : ℤ, n = 4 * k) : 4 ∣ (n^4 - n) :=
by sorry

end largest_divisor_of_n4_minus_n_l37_37495


namespace simplify_expression_l37_37825

theorem simplify_expression (x y : ℝ) :
  (3 * x^2 * y)^3 + (4 * x * y) * y^4 = 27 * x^6 * y^3 + 4 * x * y^5 :=
by 
  sorry

end simplify_expression_l37_37825


namespace find_5b_l37_37225

-- Define variables and conditions
variables (a b : ℝ)
axiom h1 : 6 * a + 3 * b = 0
axiom h2 : a = b - 3

-- State the theorem to prove
theorem find_5b : 5 * b = 10 :=
sorry

end find_5b_l37_37225


namespace minimum_red_chips_l37_37451

variable (w b r : ℕ)

-- Define the conditions
def condition1 : Prop := b ≥ 3 * w / 4
def condition2 : Prop := b ≤ r / 4
def condition3 : Prop := 60 ≤ w + b ∧ w + b ≤ 80

-- Prove the minimum number of red chips r is 108
theorem minimum_red_chips (H1 : condition1 w b) (H2 : condition2 b r) (H3 : condition3 w b) : r ≥ 108 := 
sorry

end minimum_red_chips_l37_37451


namespace largest_n_satisfying_equation_l37_37641

theorem largest_n_satisfying_equation :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ ∀ n : ℕ,
  (n * n = x * x + y * y + z * z + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 12) →
  n ≤ 2 :=
by
  sorry

end largest_n_satisfying_equation_l37_37641


namespace rational_numbers_sum_reciprocal_integer_l37_37522

theorem rational_numbers_sum_reciprocal_integer (p1 q1 p2 q2 : ℤ) (k m : ℤ)
  (h1 : Int.gcd p1 q1 = 1)
  (h2 : Int.gcd p2 q2 = 1)
  (h3 : p1 * q2 + p2 * q1 = k * q1 * q2)
  (h4 : q1 * p2 + q2 * p1 = m * p1 * p2) :
  (p1, q1, p2, q2) = (x, y, -x, y) ∨
  (p1, q1, p2, q2) = (2, 1, 2, 1) ∨
  (p1, q1, p2, q2) = (-2, 1, -2, 1) ∨
  (p1, q1, p2, q2) = (1, 1, 1, 1) ∨
  (p1, q1, p2, q2) = (-1, 1, -1, 1) ∨
  (p1, q1, p2, q2) = (1, 2, 1, 2) ∨
  (p1, q1, p2, q2) = (-1, 2, -1, 2) :=
sorry

end rational_numbers_sum_reciprocal_integer_l37_37522


namespace inequality_solution_l37_37347

open Set

noncomputable def solution_set := { x : ℝ | 5 - x^2 > 4 * x }

theorem inequality_solution :
  solution_set = { x : ℝ | -5 < x ∧ x < 1 } :=
by
  sorry

end inequality_solution_l37_37347


namespace m_gt_p_l37_37464

theorem m_gt_p (p m n : ℕ) (prime_p : Nat.Prime p) (pos_m : 0 < m) (pos_n : 0 < n) (h : p^2 + m^2 = n^2) : m > p :=
sorry

end m_gt_p_l37_37464


namespace lambda_value_l37_37461

-- Definitions provided in the conditions
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (e1 e2 : V) (A B C D : V)
-- Non-collinear vectors e1 and e2
variables (h_non_collinear : ∃ a b : ℝ, a ≠ b ∧ a • e1 + b • e2 ≠ 0)
-- Given vectors AB, BC, CD
variables (AB BC CD : V)
variables (lambda : ℝ)
-- Vector definitions based on given conditions
variables (h1 : AB = 2 • e1 + e2)
variables (h2 : BC = -e1 + 3 • e2)
variables (h3 : CD = lambda • e1 - e2)
-- Collinearity condition of points A, B, D
variables (collinear : ∃ β : ℝ, AB = β • (BC + CD))

-- The proof goal
theorem lambda_value (h1 : AB = 2 • e1 + e2) (h2 : BC = -e1 + 3 • e2) (h3 : CD = lambda • e1 - e2) (collinear : ∃ β : ℝ, AB = β • (BC + CD)) : lambda = 5 := 
sorry

end lambda_value_l37_37461


namespace percentage_reduction_l37_37900

theorem percentage_reduction :
  let original := 243.75
  let reduced := 195
  let percentage := ((original - reduced) / original) * 100
  percentage = 20 :=
by
  sorry

end percentage_reduction_l37_37900


namespace inequality_system_solution_l37_37145

theorem inequality_system_solution (x : ℝ) (h1 : 5 - 2 * x ≤ 1) (h2 : x - 4 < 0) : 2 ≤ x ∧ x < 4 :=
  sorry

end inequality_system_solution_l37_37145


namespace solve_for_n_l37_37384

theorem solve_for_n : ∃ n : ℤ, 3^3 - 5 = 4^2 + n ∧ n = 6 := 
by
  use 6
  sorry

end solve_for_n_l37_37384


namespace sqrt_operation_l37_37040

def operation (x y : ℝ) : ℝ :=
  (x + y)^2 - (x - y)^2

theorem sqrt_operation (sqrt5 : ℝ) (h : sqrt5 = Real.sqrt 5) : 
  operation sqrt5 sqrt5 = 20 := by
  sorry

end sqrt_operation_l37_37040


namespace calendars_ordered_l37_37372

theorem calendars_ordered 
  (C D : ℝ) 
  (h1 : C + D = 500) 
  (h2 : 0.75 * C + 0.50 * D = 300) 
  : C = 200 :=
by
  sorry

end calendars_ordered_l37_37372


namespace fraction_identity_l37_37256

theorem fraction_identity (a b : ℚ) (h1 : 3 * a = 4 * b) (h2 : a ≠ 0 ∧ b ≠ 0) : (a + b) / a = 7 / 4 :=
by
  sorry

end fraction_identity_l37_37256


namespace relationship_between_a_and_b_l37_37706

theorem relationship_between_a_and_b (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b)
    (h₃ : ∀ x : ℝ, |(3 * x + 1) - 4| < a → |x - 1| < b) : a ≥ 3 * b :=
by
  -- Applying the given conditions, we want to demonstrate that a ≥ 3b.
  sorry

end relationship_between_a_and_b_l37_37706


namespace original_length_before_final_cut_l37_37914

-- Defining the initial length of the board
def initial_length : ℕ := 143

-- Defining the length after the first cut
def length_after_first_cut : ℕ := initial_length - 25

-- Defining the length after the final cut
def length_after_final_cut : ℕ := length_after_first_cut - 7

-- Stating the theorem to prove that the original length of the board before cutting the final 7 cm is 125 cm
theorem original_length_before_final_cut : initial_length - 25 + 7 = 125 :=
sorry

end original_length_before_final_cut_l37_37914


namespace garden_area_l37_37881

theorem garden_area (P b l: ℕ) (hP: P = 900) (hb: b = 190) (hl: l = P / 2 - b):
  l * b = 49400 := 
by
  sorry

end garden_area_l37_37881


namespace quadratic_complete_square_l37_37453

theorem quadratic_complete_square (x d e: ℝ) (h : x^2 - 26 * x + 129 = (x + d)^2 + e) : 
d + e = -53 := sorry

end quadratic_complete_square_l37_37453


namespace ratio_depth_to_height_l37_37223

theorem ratio_depth_to_height
  (Dean_height : ℝ := 9)
  (additional_depth : ℝ := 81)
  (water_depth : ℝ := Dean_height + additional_depth) :
  water_depth / Dean_height = 10 :=
by
  -- Dean_height = 9
  -- additional_depth = 81
  -- water_depth = 9 + 81 = 90
  -- water_depth / Dean_height = 90 / 9 = 10
  sorry

end ratio_depth_to_height_l37_37223


namespace hyperbola_equation_l37_37417

-- Fixed points F_1 and F_2
def F1 : ℝ × ℝ := (5, 0)
def F2 : ℝ × ℝ := (-5, 0)

-- Condition: The absolute value of the difference in distances from P to F1 and F2 is 6
def distance_condition (P : ℝ × ℝ) : Prop :=
  abs ((dist P F1) - (dist P F2)) = 6

theorem hyperbola_equation : 
  ∃ (a b : ℝ), a = 3 ∧ b = 4 ∧ ∀ (x y : ℝ), distance_condition (x, y) → 
  (x ^ 2) / (a ^ 2) - (y ^ 2) / (b ^ 2) = 1 :=
by
  -- We state the conditions and result derived from them
  sorry

end hyperbola_equation_l37_37417


namespace laptop_price_reduction_l37_37304

-- Conditions definitions
def initial_price (P : ℝ) : ℝ := P
def seasonal_sale (P : ℝ) : ℝ := 0.7 * P
def special_promotion (seasonal_price : ℝ) : ℝ := 0.8 * seasonal_price
def clearance_event (promotion_price : ℝ) : ℝ := 0.9 * promotion_price

-- Proof statement
theorem laptop_price_reduction (P : ℝ) (h1 : seasonal_sale P = 0.7 * P) 
    (h2 : special_promotion (seasonal_sale P) = 0.8 * (seasonal_sale P)) 
    (h3 : clearance_event (special_promotion (seasonal_sale P)) = 0.9 * (special_promotion (seasonal_sale P))) : 
    (initial_price P - clearance_event (special_promotion (seasonal_sale P))) / (initial_price P) = 0.496 := 
by 
  sorry

end laptop_price_reduction_l37_37304


namespace smallest_C_inequality_l37_37354

theorem smallest_C_inequality (x y z : ℝ) (h : x + y + z = -1) : 
  |x^3 + y^3 + z^3 + 1| ≤ (9/10) * |x^5 + y^5 + z^5 + 1| :=
  sorry

end smallest_C_inequality_l37_37354


namespace population_net_increase_in_one_day_l37_37189

-- Define the problem conditions
def birth_rate : ℕ := 6 / 2  -- births per second
def death_rate : ℕ := 3 / 2  -- deaths per second
def seconds_in_a_day : ℕ := 60 * 60 * 24

-- Define the assertion we want to prove
theorem population_net_increase_in_one_day : 
  ( (birth_rate - death_rate) * seconds_in_a_day ) = 259200 := by
  -- Since 6/2 = 3 and 3/2 = 1.5 is not an integer in Lean, we use ratios directly
  sorry  -- Proof is not required

end population_net_increase_in_one_day_l37_37189


namespace crows_eat_worms_l37_37432

theorem crows_eat_worms (worms_eaten_by_3_crows_in_1_hour : ℕ) 
                        (crows_eating_worms_constant : worms_eaten_by_3_crows_in_1_hour = 30)
                        (number_of_crows : ℕ) 
                        (observation_time_hours : ℕ) :
                        number_of_crows = 5 ∧ observation_time_hours = 2 →
                        (number_of_crows * worms_eaten_by_3_crows_in_1_hour / 3) * observation_time_hours = 100 :=
by
  sorry

end crows_eat_worms_l37_37432


namespace find_triplets_l37_37054

theorem find_triplets (x y z : ℕ) (h1 : x ≤ y) (h2 : x^2 + y^2 = 3 * 2016^z + 77) :
  (x, y, z) = (4, 8, 0) ∨ (x, y, z) = (14, 77, 1) ∨ (x, y, z) = (35, 70, 1) :=
  sorry

end find_triplets_l37_37054


namespace geometric_sequence_sum_63_l37_37855

theorem geometric_sequence_sum_63
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_init : a 1 = 1)
  (h_recurrence : ∀ n, a (n + 2) + 2 * a (n + 1) = 8 * a n) :
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6) = 63 :=
by
  sorry

end geometric_sequence_sum_63_l37_37855


namespace tennis_balls_in_each_container_l37_37559

theorem tennis_balls_in_each_container (initial_balls : ℕ) (half_gone : ℕ) (remaining_balls : ℕ) (containers : ℕ) 
  (h1 : initial_balls = 100) 
  (h2 : half_gone = initial_balls / 2)
  (h3 : remaining_balls = initial_balls - half_gone)
  (h4 : containers = 5) :
  remaining_balls / containers = 10 := 
by
  sorry

end tennis_balls_in_each_container_l37_37559


namespace linen_tablecloth_cost_l37_37056

def num_tables : ℕ := 20
def cost_per_place_setting : ℕ := 10
def num_place_settings_per_table : ℕ := 4
def cost_per_rose : ℕ := 5
def num_roses_per_centerpiece : ℕ := 10
def cost_per_lily : ℕ := 4
def num_lilies_per_centerpiece : ℕ := 15
def total_decoration_cost : ℕ := 3500

theorem linen_tablecloth_cost :
  (total_decoration_cost - (num_tables * num_place_settings_per_table * cost_per_place_setting + num_tables * (num_roses_per_centerpiece * cost_per_rose + num_lilies_per_centerpiece * cost_per_lily))) / num_tables = 25 :=
  sorry

end linen_tablecloth_cost_l37_37056


namespace triangles_with_positive_area_l37_37625

theorem triangles_with_positive_area (x y : ℕ) (h₁ : 1 ≤ x ∧ x ≤ 5) (h₂ : 1 ≤ y ∧ y ≤ 3) : 
    ∃ (n : ℕ), n = 420 := 
sorry

end triangles_with_positive_area_l37_37625


namespace ivans_profit_l37_37030

def price_meat_per_kg : ℕ := 500
def kg_meat_sold : ℕ := 100
def price_eggs_per_dozen : ℕ := 50
def eggs_sold : ℕ := 20000
def annual_expenses : ℕ := 100000

def revenue_meat : ℕ := kg_meat_sold * price_meat_per_kg
def revenue_eggs : ℕ := eggs_sold * (price_eggs_per_dozen / 10)
def total_revenue : ℕ := revenue_meat + revenue_eggs

def profit : ℕ := total_revenue - annual_expenses

theorem ivans_profit : profit = 50000 := by
  sorry

end ivans_profit_l37_37030


namespace value_of_k_l37_37733

theorem value_of_k (k : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : y = (k - 1) * x + k^2 - 1)
  (h2 : ∃ m : ℝ, y = m * x)
  (h3 : k ≠ 1) :
  k = -1 :=
by
  sorry

end value_of_k_l37_37733


namespace sufficient_but_not_necessary_condition_l37_37195

theorem sufficient_but_not_necessary_condition (x : ℝ) (h : x^2 - 3 * x + 2 > 0) : x > 2 ∨ x < -1 :=
by
  sorry

example (x : ℝ) (h : x^2 - 3 * x + 2 > 0) : (x > 2) ∨ (x < -1) := 
by 
  apply sufficient_but_not_necessary_condition; exact h

end sufficient_but_not_necessary_condition_l37_37195


namespace soccer_league_fraction_female_proof_l37_37915

variable (m f : ℝ)

def soccer_league_fraction_female : Prop :=
  let males_last_year := m
  let females_last_year := f
  let males_this_year := 1.05 * m
  let females_this_year := 1.2 * f
  let total_this_year := 1.1 * (m + f)
  (1.05 * m + 1.2 * f = 1.1 * (m + f)) → ((0.6 * m) / (1.65 * m) = 4 / 11)

theorem soccer_league_fraction_female_proof (m f : ℝ) : soccer_league_fraction_female m f :=
by {
  sorry
}

end soccer_league_fraction_female_proof_l37_37915


namespace cases_in_1995_l37_37666

theorem cases_in_1995 (initial_cases cases_2010 : ℕ) (years_total : ℕ) (years_passed : ℕ) (cases_1995 : ℕ)
  (h1 : initial_cases = 700000) 
  (h2 : cases_2010 = 1000) 
  (h3 : years_total = 40) 
  (h4 : years_passed = 25)
  (h5 : cases_1995 = initial_cases - (years_passed * (initial_cases - cases_2010) / years_total)) : 
  cases_1995 = 263125 := 
sorry

end cases_in_1995_l37_37666


namespace steve_speed_back_l37_37865

theorem steve_speed_back :
  ∀ (d v_total : ℕ), d = 10 → v_total = 6 →
  (2 * (15 / 6)) = 5 :=
by
  intros d v_total d_eq v_total_eq
  sorry

end steve_speed_back_l37_37865


namespace find_middle_number_l37_37856

namespace Problem

-- Define the three numbers x, y, z
variables (x y z : ℕ)

-- Given conditions from the problem
def condition1 (h1 : x + y = 18) := x + y = 18
def condition2 (h2 : x + z = 23) := x + z = 23
def condition3 (h3 : y + z = 27) := y + z = 27
def condition4 (h4 : x < y ∧ y < z) := x < y ∧ y < z

-- Statement to prove:
theorem find_middle_number (h1 : x + y = 18) (h2 : x + z = 23) (h3 : y + z = 27) (h4 : x < y ∧ y < z) : 
  y = 11 :=
by
  sorry

end Problem

end find_middle_number_l37_37856


namespace positive_difference_l37_37251

theorem positive_difference
  (x y : ℝ)
  (h1 : x + y = 10)
  (h2 : x^2 - y^2 = 40) : abs (x - y) = 4 :=
sorry

end positive_difference_l37_37251


namespace ben_total_distance_walked_l37_37618

-- Definitions based on conditions
def walking_speed : ℝ := 4  -- 4 miles per hour.
def total_time : ℝ := 2  -- 2 hours.
def break_time : ℝ := 0.25  -- 0.25 hours (15 minutes).

-- Proof goal: Prove that the total distance walked is 7.0 miles.
theorem ben_total_distance_walked : (walking_speed * (total_time - break_time) = 7.0) :=
by
  sorry

end ben_total_distance_walked_l37_37618


namespace solution_to_inequalities_l37_37105

theorem solution_to_inequalities (x : ℝ) : 
  (3 * x + 2 < (x + 2)^2 ∧ (x + 2)^2 < 8 * x + 1) ↔ (1 < x ∧ x < 3) := by
  sorry

end solution_to_inequalities_l37_37105


namespace problem_statement_l37_37187

def g (x : ℕ) : ℕ := x^2 - 4 * x

theorem problem_statement :
  g (g (g (g (g (g 2))))) = L := sorry

end problem_statement_l37_37187


namespace solve_equation_l37_37837

theorem solve_equation :
  {x : ℝ | x * (x - 3)^2 * (5 - x) = 0} = {0, 3, 5} :=
by
  sorry

end solve_equation_l37_37837


namespace isha_original_length_l37_37239

variable (current_length sharpened_off : ℕ)

-- Condition 1: Isha's pencil is now 14 inches long
def isha_current_length : current_length = 14 := sorry

-- Condition 2: She sharpened off 17 inches of her pencil
def isha_sharpened_off : sharpened_off = 17 := sorry

-- Statement to prove:
theorem isha_original_length (current_length sharpened_off : ℕ) 
  (h1 : current_length = 14) (h2 : sharpened_off = 17) :
  current_length + sharpened_off = 31 :=
by
  sorry

end isha_original_length_l37_37239


namespace ratio_of_pieces_l37_37005

def total_length (len: ℕ) := len = 35
def longer_piece (len: ℕ) := len = 20

theorem ratio_of_pieces (shorter len_shorter : ℕ) : 
  total_length 35 →
  longer_piece 20 →
  shorter = 35 - 20 →
  len_shorter = 15 →
  (20:ℚ) / (len_shorter:ℚ) = (4:ℚ) / (3:ℚ) :=
by
  sorry

end ratio_of_pieces_l37_37005


namespace interval_length_t_subset_interval_t_l37_37217

-- Statement (1)
theorem interval_length_t (t : ℝ) (h : (Real.log t / Real.log 2) - 2 = 3) : t = 32 :=
  sorry

-- Statement (2)
theorem subset_interval_t (t : ℝ) (h : 2 ≤ Real.log t / Real.log 2 ∧ Real.log t / Real.log 2 ≤ 5) :
  0 < t ∧ t ≤ 32 :=
  sorry

end interval_length_t_subset_interval_t_l37_37217


namespace value_of_x_squared_plus_one_over_x_squared_l37_37489

noncomputable def x: ℝ := sorry

theorem value_of_x_squared_plus_one_over_x_squared (h : 20 = x^6 + 1 / x^6) : x^2 + 1 / x^2 = 23 :=
sorry

end value_of_x_squared_plus_one_over_x_squared_l37_37489


namespace aria_spent_on_cookies_l37_37694

def aria_spent : ℕ := 2356

theorem aria_spent_on_cookies :
  (let cookies_per_day := 4
  let cost_per_cookie := 19
  let days_in_march := 31
  let total_cookies := days_in_march * cookies_per_day
  let total_cost := total_cookies * cost_per_cookie
  total_cost = aria_spent) :=
  sorry

end aria_spent_on_cookies_l37_37694


namespace longest_side_similar_triangle_l37_37993

noncomputable def internal_angle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem longest_side_similar_triangle (a b c A : ℝ) (h₁ : a = 4) (h₂ : b = 6) (h₃ : c = 7) (h₄ : A = 132) :
  let k := Real.sqrt (132 / internal_angle 4 6 7)
  7 * k = 73.5 :=
by
  sorry

end longest_side_similar_triangle_l37_37993


namespace min_candidates_for_same_score_l37_37566

theorem min_candidates_for_same_score :
  (∃ S : ℕ, S ≥ 25 ∧ (∀ elect : Fin S → Fin 12, ∃ s : Fin 12, ∃ a b c : Fin S, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ elect a = s ∧ elect b = s ∧ elect c = s)) := 
sorry

end min_candidates_for_same_score_l37_37566


namespace vector_division_by_three_l37_37978

def OA : ℝ × ℝ := (2, 8)
def OB : ℝ × ℝ := (-7, 2)
def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
noncomputable def scalar_mult (k : ℝ) (u : ℝ × ℝ) : ℝ × ℝ := (k * u.1, k * u.2)

theorem vector_division_by_three :
  scalar_mult (1 / 3) (vector_sub OB OA) = (-3, -2) :=
sorry

end vector_division_by_three_l37_37978


namespace color_changes_probability_l37_37662

-- Define the durations of the traffic lights
def green_duration := 40
def yellow_duration := 5
def red_duration := 45

-- Define the total cycle duration
def total_cycle_duration := green_duration + yellow_duration + red_duration

-- Define the duration of the interval Mary watches
def watch_duration := 4

-- Define the change windows where the color changes can be witnessed
def change_windows :=
  [green_duration - watch_duration,
   green_duration + yellow_duration - watch_duration,
   green_duration + yellow_duration + red_duration - watch_duration]

-- Define the total change window duration
def total_change_window_duration := watch_duration * (change_windows.length)

-- Calculate the probability of witnessing a change
def probability_witnessing_change := (total_change_window_duration : ℚ) / total_cycle_duration

-- The theorem to prove
theorem color_changes_probability :
  probability_witnessing_change = 2 / 15 := by sorry

end color_changes_probability_l37_37662


namespace factorize_expression_l37_37292

-- Define the variables
variables (a b : ℝ)

-- State the theorem to prove the factorization
theorem factorize_expression : a^2 - 2 * a * b = a * (a - 2 * b) :=
by 
  -- Proof goes here
  sorry

end factorize_expression_l37_37292


namespace cube_volume_l37_37636

theorem cube_volume (a : ℕ) (h1 : 9 * 12 * 3 = 324) (h2 : 108 * a^3 = 324) : a^3 = 27 :=
by {
  sorry
}

end cube_volume_l37_37636


namespace mrs_hilt_bakes_loaves_l37_37147

theorem mrs_hilt_bakes_loaves :
  let total_flour := 5
  let flour_per_loaf := 2.5
  (total_flour / flour_per_loaf) = 2 := 
by
  sorry

end mrs_hilt_bakes_loaves_l37_37147


namespace count_bases_for_last_digit_l37_37687

theorem count_bases_for_last_digit (n : ℕ) : n = 729 → ∃ S : Finset ℕ, S.card = 2 ∧ ∀ b ∈ S, 2 ≤ b ∧ b ≤ 10 ∧ (n - 5) % b = 0 :=
by
  sorry

end count_bases_for_last_digit_l37_37687


namespace CoreyCandies_l37_37211

theorem CoreyCandies (T C : ℕ) (h1 : T + C = 66) (h2 : T = C + 8) : C = 29 :=
by
  sorry

end CoreyCandies_l37_37211


namespace mary_mac_download_time_l37_37777

theorem mary_mac_download_time (x : ℕ) (windows_download : ℕ) (total_glitch : ℕ) (time_without_glitches : ℕ) (total_time : ℕ) :
  windows_download = 3 * x ∧
  total_glitch = 14 ∧
  time_without_glitches = 2 * total_glitch ∧
  total_time = 82 ∧
  x + windows_download + total_glitch + time_without_glitches = total_time →
  x = 10 :=
by 
  sorry

end mary_mac_download_time_l37_37777


namespace sum_of_solutions_eq_l37_37002

theorem sum_of_solutions_eq (x : ℝ) : (5 * x - 7) * (4 * x + 11) = 0 ->
  -((27 : ℝ) / (20 : ℝ)) =
  - ((5 * - 7) * (4 * x + 11)) / ((5 * x - 7) * 4) :=
by
  intro h
  sorry

end sum_of_solutions_eq_l37_37002


namespace calculate_expression_l37_37553

def f (x : ℝ) := x^2 + 3
def g (x : ℝ) := 2 * x + 4

theorem calculate_expression : f (g 2) - g (f 2) = 49 := by
  sorry

end calculate_expression_l37_37553


namespace find_original_one_digit_number_l37_37984

theorem find_original_one_digit_number (x : ℕ) (h1 : x < 10) (h2 : (x + 10) * (x + 10) / x = 72) : x = 2 :=
sorry

end find_original_one_digit_number_l37_37984


namespace simplify_trig_expression_l37_37506

open Real

/-- 
Given that θ is in the interval (π/2, π), simplify the expression 
( sin θ / sqrt (1 - sin^2 θ) ) + ( sqrt (1 - cos^2 θ) / cos θ ) to 0.
-/
theorem simplify_trig_expression (θ : ℝ) (hθ1 : π / 2 < θ) (hθ2 : θ < π) :
  (sin θ / sqrt (1 - sin θ ^ 2)) + (sqrt (1 - cos θ ^ 2) / cos θ) = 0 :=
by 
  sorry

end simplify_trig_expression_l37_37506


namespace femaleRainbowTroutCount_l37_37792

noncomputable def numFemaleRainbowTrout : ℕ := 
  let numSpeckledTrout := 645
  let numFemaleSpeckled := 200
  let numMaleSpeckled := 445
  let numMaleRainbow := 150
  let totalTrout := 1000
  let numRainbowTrout := totalTrout - numSpeckledTrout
  numRainbowTrout - numMaleRainbow

theorem femaleRainbowTroutCount : numFemaleRainbowTrout = 205 := by
  -- Conditions
  let numSpeckledTrout : ℕ := 645
  let numMaleSpeckled := 2 * 200 + 45
  let totalTrout := 645 + 355
  let numRainbowTrout := totalTrout - numSpeckledTrout
  let numFemaleRainbow := numRainbowTrout - 150
  
  -- The proof would proceed here
  sorry

end femaleRainbowTroutCount_l37_37792


namespace maura_seashells_l37_37408

theorem maura_seashells (original_seashells given_seashells remaining_seashells : ℕ)
  (h1 : original_seashells = 75) 
  (h2 : remaining_seashells = 57) 
  (h3 : given_seashells = original_seashells - remaining_seashells) :
  given_seashells = 18 := by
  -- Lean will use 'sorry' as a placeholder for the actual proof
  sorry

end maura_seashells_l37_37408


namespace chocolates_bought_l37_37776

theorem chocolates_bought (C S N : ℕ) (h1 : 4 * C = 7 * (S - C)) (h2 : N * C = 77 * S) :
  N = 121 :=
by
  sorry

end chocolates_bought_l37_37776


namespace solution_set_inequality_l37_37416

-- Statement of the problem
theorem solution_set_inequality :
  {x : ℝ | 1 / x < 1 / 2} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 2} :=
sorry

end solution_set_inequality_l37_37416


namespace ball_attendance_l37_37757

variable (n m : ℕ)

def ball_conditions (n m : ℕ) := 
  n + m < 50 ∧ 
  4 ∣ 3 * n ∧ 
  7 ∣ 5 * m

theorem ball_attendance (n m : ℕ) (h : ball_conditions n m) : 
  n + m = 41 :=
sorry

end ball_attendance_l37_37757


namespace linear_eq_m_val_l37_37094

theorem linear_eq_m_val (m : ℤ) (x : ℝ) : (5 * x ^ (m - 2) + 1 = 0) → (m = 3) :=
by
  sorry

end linear_eq_m_val_l37_37094


namespace proof_problem_l37_37146

variable (a b c : ℝ)

theorem proof_problem (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : ∀ x, abs (x + a) - abs (x - b) + c ≤ 10) :
  a + b + c = 10 ∧ 
  (∀ (h5 : a + b + c = 10), 
    (∃ a' b' c', a' = 11/3 ∧ b' = 8/3 ∧ c' = 11/3 ∧ 
                (∀ a'' b'' c'', a'' = a ∧ b'' = b ∧ c'' = c → 
                (1/4 * (a - 1)^2 + (b - 2)^2 + (c - 3)^2) ≥ 8/3 ∧ 
                (1/4 * (a' - 1)^2 + (b' - 2)^2 + (c' - 3)^2) = 8 / 3 ))) := by
  sorry

end proof_problem_l37_37146


namespace athlete_more_stable_l37_37449

theorem athlete_more_stable (var_A var_B : ℝ) 
                                (h1 : var_A = 0.024) 
                                (h2 : var_B = 0.008) 
                                (h3 : var_A > var_B) : 
  var_B < var_A :=
by
  exact h3

end athlete_more_stable_l37_37449


namespace range_of_f_is_pi_div_four_l37_37024

noncomputable def f (x : ℝ) : ℝ := 
  Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_of_f_is_pi_div_four : ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y = π / 4 :=
sorry

end range_of_f_is_pi_div_four_l37_37024


namespace find_principal_l37_37440

theorem find_principal (R P : ℝ) (h₁ : (P * R * 10) / 100 = P * R * 0.1)
  (h₂ : (P * (R + 3) * 10) / 100 = P * (R + 3) * 0.1)
  (h₃ : P * 0.1 * (R + 3) - P * 0.1 * R = 300) : 
  P = 1000 := 
sorry

end find_principal_l37_37440


namespace tangent_line_at_point_l37_37616

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x - 4 * (x - 1)

theorem tangent_line_at_point (x y : ℝ) (h : f 1 = 0) (h' : deriv f 1 = -2) :
  2 * x + y - 2 = 0 :=
sorry

end tangent_line_at_point_l37_37616


namespace consecutive_integers_satisfy_inequality_l37_37250

theorem consecutive_integers_satisfy_inequality :
  ∀ (n m : ℝ), n + 1 = m ∧ n < Real.sqrt 26 ∧ Real.sqrt 26 < m → m + n = 11 :=
by
  sorry

end consecutive_integers_satisfy_inequality_l37_37250


namespace not_perfect_square_l37_37433

theorem not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, (3^n + 2 * 17^n) = k^2 :=
by
  sorry

end not_perfect_square_l37_37433


namespace cube_side_length_ratio_l37_37873

-- Define the conditions and question
variable (s₁ s₂ : ℝ)
variable (weight₁ weight₂ : ℝ)
variable (V₁ V₂ : ℝ)
variable (same_metal : Prop)

-- Conditions
def condition1 (weight₁ : ℝ) : Prop := weight₁ = 4
def condition2 (weight₂ : ℝ) : Prop := weight₂ = 32
def condition3 (V₁ V₂ : ℝ) (s₁ s₂ : ℝ) : Prop := (V₁ = s₁^3) ∧ (V₂ = s₂^3)
def condition4 (same_metal : Prop) : Prop := same_metal

-- Volume definition based on weights and proportion
noncomputable def volume_definition (weight₁ weight₂ V₁ V₂ : ℝ) : Prop :=
(weight₂ / weight₁) = (V₂ / V₁)

-- Define the proof target
theorem cube_side_length_ratio
    (h1 : condition1 weight₁)
    (h2 : condition2 weight₂)
    (h3 : condition3 V₁ V₂ s₁ s₂)
    (h4 : condition4 same_metal)
    (h5 : volume_definition weight₁ weight₂ V₁ V₂) : 
    (s₂ / s₁) = 2 :=
by
  sorry

end cube_side_length_ratio_l37_37873


namespace train_length_l37_37203
-- Import all necessary libraries from Mathlib

-- Define the given conditions and prove the target
theorem train_length (L_t L_p : ℝ) (h1 : L_t = L_p) (h2 : 54 * (1000 / 3600) * 60 = 2 * L_t) : L_t = 450 :=
by
  -- Proof goes here
  sorry

end train_length_l37_37203


namespace divide_5440_K_l37_37548

theorem divide_5440_K (a b c d : ℕ) 
  (h1 : 5440 = a + b + c + d)
  (h2 : 2 * b = 3 * a)
  (h3 : 3 * c = 5 * b)
  (h4 : 5 * d = 6 * c) : 
  a = 680 ∧ b = 1020 ∧ c = 1700 ∧ d = 2040 :=
by 
  sorry

end divide_5440_K_l37_37548


namespace original_manufacturing_cost_l37_37531

variable (SP OC : ℝ)
variable (ManuCost : ℝ) -- Declaring manufacturing cost

-- Current conditions
axiom profit_percentage_constant : ∀ SP, 0.5 * SP = SP - 50

-- Problem Statement
theorem original_manufacturing_cost : (∃ OC, 0.5 * SP - OC = 0.5 * SP) ∧ ManuCost = 50 → OC = 50 := by
  sorry

end original_manufacturing_cost_l37_37531


namespace distance_between_trees_l37_37065

theorem distance_between_trees (length_yard : ℕ) (num_trees : ℕ) (dist : ℕ) :
  length_yard = 275 → num_trees = 26 → dist = length_yard / (num_trees - 1) → dist = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  assumption

end distance_between_trees_l37_37065


namespace point_inside_circle_l37_37703

theorem point_inside_circle (O A : Type) (r OA : ℝ) (h1 : r = 6) (h2 : OA = 5) :
  OA < r :=
by
  sorry

end point_inside_circle_l37_37703


namespace quadrilateral_diagonals_l37_37570

-- Define the points of the quadrilateral
variables {A B C D P Q R S : ℝ × ℝ}

-- Define the midpoints condition
def is_midpoint (M : ℝ × ℝ) (X Y : ℝ × ℝ) := M = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2)

-- Define the lengths squared condition
def dist_sq (X Y : ℝ × ℝ) := (X.1 - Y.1)^2 + (X.2 - Y.2)^2

-- Main theorem to prove
theorem quadrilateral_diagonals (hP : is_midpoint P A B) (hQ : is_midpoint Q B C)
  (hR : is_midpoint R C D) (hS : is_midpoint S D A) :
  dist_sq A C + dist_sq B D = 2 * (dist_sq P R + dist_sq Q S) :=
by
  sorry

end quadrilateral_diagonals_l37_37570


namespace cos_triple_angle_l37_37359

theorem cos_triple_angle (x θ : ℝ) (h : x = Real.cos θ) : Real.cos (3 * θ) = 4 * x^3 - 3 * x :=
by
  sorry

end cos_triple_angle_l37_37359


namespace part1_part2_l37_37903

theorem part1 (a : ℝ) (x : ℝ) (h : a ≠ 0) :
    (|x - a| + |x + a + (1 / a)|) ≥ 2 * Real.sqrt 2 :=
sorry

theorem part2 (a : ℝ) (h : a ≠ 0) (h₁ : |2 - a| + |2 + a + 1 / a| ≤ 3) :
    a ∈ Set.Icc (-1 : ℝ) (-1/2) ∪ Set.Ico (1/2 : ℝ) 2 :=
sorry

end part1_part2_l37_37903


namespace daisies_sold_on_fourth_day_l37_37628

-- Number of daisies sold on the first day
def first_day_daisies : ℕ := 45

-- Number of daisies sold on the second day
def second_day_daisies : ℕ := first_day_daisies + 20

-- Number of daisies sold on the third day
def third_day_daisies : ℕ := 2 * second_day_daisies - 10

-- Total number of daisies sold in the first three days
def total_first_three_days_daisies : ℕ := first_day_daisies + second_day_daisies + third_day_daisies

-- Total number of daisies sold in four days
def total_four_days_daisies : ℕ := 350

-- Number of daisies sold on the fourth day
def fourth_day_daisies : ℕ := total_four_days_daisies - total_first_three_days_daisies

-- Theorem that states the number of daisies sold on the fourth day is 120
theorem daisies_sold_on_fourth_day : fourth_day_daisies = 120 :=
by sorry

end daisies_sold_on_fourth_day_l37_37628


namespace max_two_terms_eq_one_l37_37569

theorem max_two_terms_eq_one (a b c x y z : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : x ≠ y) (h5 : y ≠ z) (h6 : x ≠ z) :
  ∀ (P : ℕ → ℝ), -- Define P(i) as given expressions
  ((P 1 = a * x + b * y + c * z) ∧
   (P 2 = a * x + b * z + c * y) ∧
   (P 3 = a * y + b * x + c * z) ∧
   (P 4 = a * y + b * z + c * x) ∧
   (P 5 = a * z + b * x + c * y) ∧
   (P 6 = a * z + b * y + c * x)) →
  (P 1 = 1 ∨ P 2 = 1 ∨ P 3 = 1 ∨ P 4 = 1 ∨ P 5 = 1 ∨ P 6 = 1) →
  (∃ i j, i ≠ j ∧ P i = 1 ∧ P j = 1) →
  ¬(∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ P i = 1 ∧ P j = 1 ∧ P k = 1) :=
sorry

end max_two_terms_eq_one_l37_37569


namespace radius_of_circumscribed_sphere_l37_37262

noncomputable def circumscribedSphereRadius (a : ℝ) (α := 60 * Real.pi / 180) : ℝ :=
  5 * a / (4 * Real.sqrt 3)

theorem radius_of_circumscribed_sphere (a : ℝ) :
  circumscribedSphereRadius a = 5 * a / (4 * Real.sqrt 3) := by
  sorry

end radius_of_circumscribed_sphere_l37_37262


namespace find_number_l37_37938

theorem find_number (n : ℕ) (h : 582964 * n = 58293485180) : n = 100000 :=
by
  sorry

end find_number_l37_37938


namespace problem_integer_and_decimal_parts_eq_2_l37_37103

theorem problem_integer_and_decimal_parts_eq_2 :
  let x := 3
  let y := 2 - Real.sqrt 3
  2 * x^3 - (y^3 + 1 / y^3) = 2 :=
by
  sorry

end problem_integer_and_decimal_parts_eq_2_l37_37103


namespace find_k_parallel_find_k_perpendicular_l37_37226

noncomputable def veca : (ℝ × ℝ) := (1, 2)
noncomputable def vecb : (ℝ × ℝ) := (-3, 2)

def is_parallel (u v : (ℝ × ℝ)) : Prop := 
  ∃ k : ℝ, k ≠ 0 ∧ u = (k * v.1, k * v.2)

def is_perpendicular (u v : (ℝ × ℝ)) : Prop := 
  u.1 * v.1 + u.2 * v.2 = 0

def calc_vector (k : ℝ) (a b : (ℝ × ℝ)) : (ℝ × ℝ) :=
  (k * a.1 + b.1, k * a.2 + b.2)

theorem find_k_parallel : 
  ∃ k : ℝ, is_parallel (calc_vector k veca vecb) (calc_vector 1 veca (-2 * vecb)) := sorry

theorem find_k_perpendicular :
  ∃ k : ℝ, k = 25 / 3 ∧ is_perpendicular (calc_vector k veca vecb) (calc_vector 1 veca (-2 * vecb)) := sorry

end find_k_parallel_find_k_perpendicular_l37_37226


namespace quadrilateral_type_l37_37123

theorem quadrilateral_type (m n p q : ℝ) (h : m^2 + n^2 + p^2 + q^2 = 2 * m * n + 2 * p * q) : 
  (m = n ∧ p = q) ∨ (m ≠ n ∧ p ≠ q ∧ ∃ k : ℝ, k^2 * (m^2 + n^2) = p^2 + q^2) := 
sorry

end quadrilateral_type_l37_37123


namespace problem_CorrectOption_l37_37959

def setA : Set ℝ := {y | ∃ x : ℝ, y = |x| - 1}
def setB : Set ℝ := {x | x ≥ 2}

theorem problem_CorrectOption : setA ∩ setB = setB := 
  sorry

end problem_CorrectOption_l37_37959


namespace num_candidates_appeared_each_state_l37_37424

-- Definitions
def candidates_appear : ℕ := 8000
def sel_pct_A : ℚ := 0.06
def sel_pct_B : ℚ := 0.07
def additional_selections_B : ℕ := 80

-- Proof Problem Statement
theorem num_candidates_appeared_each_state (x : ℕ) 
  (h1 : x = candidates_appear) 
  (h2 : sel_pct_A * ↑x = 0.06 * ↑x) 
  (h3 : sel_pct_B * ↑x = 0.07 * ↑x) 
  (h4 : sel_pct_B * ↑x = sel_pct_A * ↑x + additional_selections_B) : 
  x = candidates_appear := sorry

end num_candidates_appeared_each_state_l37_37424


namespace inequality_solution_l37_37345

theorem inequality_solution (x : ℤ) (h : x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1) : x - 1 ≥ 0 ↔ x = 1 :=
by
  sorry

end inequality_solution_l37_37345


namespace circle_radius_five_eq_neg_eight_l37_37073

theorem circle_radius_five_eq_neg_eight (c : ℝ) :
  (∃ x y : ℝ, x^2 + 8*x + y^2 + 2*y + c = 0 ∧ (x + 4)^2 + (y + 1)^2 = 25) → c = -8 :=
by
  sorry

end circle_radius_five_eq_neg_eight_l37_37073


namespace friends_total_sales_l37_37765

theorem friends_total_sales :
  (Ryan Jason Zachary : ℕ) →
  (H1 : Ryan = Jason + 50) →
  (H2 : Jason = Zachary + (3 * Zachary / 10)) →
  (H3 : Zachary = 40 * 5) →
  Ryan + Jason + Zachary = 770 :=
by
  sorry

end friends_total_sales_l37_37765


namespace solve_for_x_l37_37762

theorem solve_for_x (x : ℝ) (h₀ : x > 0) (h₁ : 1 / 2 * x * (3 * x) = 96) : x = 8 :=
sorry

end solve_for_x_l37_37762


namespace probability_no_shaded_rectangle_l37_37693

-- Definitions
def total_rectangles_per_row : ℕ := (2005 * 2004) / 2
def shaded_rectangles_per_row : ℕ := 1002 * 1002

-- Proposition to prove
theorem probability_no_shaded_rectangle : 
  (1 - (shaded_rectangles_per_row : ℝ) / (total_rectangles_per_row : ℝ)) = (0.25 / 1002.25) := 
sorry

end probability_no_shaded_rectangle_l37_37693


namespace focus_of_given_parabola_l37_37848

-- Define the given condition as a parameter
def parabola_eq (x y : ℝ) : Prop :=
  y = - (1/2) * x^2

-- Define the property for the focus of the parabola
def is_focus_of_parabola (focus : ℝ × ℝ) : Prop :=
  focus = (0, -1/2)

-- The theorem stating that the given parabola equation has the specific focus
theorem focus_of_given_parabola : 
  (∀ x y : ℝ, parabola_eq x y) → is_focus_of_parabola (0, -1/2) :=
by
  intro h
  unfold parabola_eq at h
  unfold is_focus_of_parabola
  sorry

end focus_of_given_parabola_l37_37848


namespace biggest_number_l37_37716

theorem biggest_number (A B C D : ℕ) (h1 : A / B = 2 / 3) (h2 : B / C = 3 / 4) (h3 : C / D = 4 / 5) (h4 : A + B + C + D = 1344) : D = 480 := 
sorry

end biggest_number_l37_37716


namespace eight_point_shots_count_is_nine_l37_37770

def num_8_point_shots (x y z : ℕ) := 8 * x + 9 * y + 10 * z = 100 ∧
                                      x + y + z > 11 ∧ 
                                      x + y + z ≤ 12 ∧ 
                                      x > 0 ∧ 
                                      y > 0 ∧ 
                                      z > 0

theorem eight_point_shots_count_is_nine : 
  ∃ x y z : ℕ, num_8_point_shots x y z ∧ x = 9 :=
by
  sorry

end eight_point_shots_count_is_nine_l37_37770


namespace animal_population_l37_37685

theorem animal_population
  (number_of_lions : ℕ)
  (number_of_leopards : ℕ)
  (number_of_elephants : ℕ)
  (h1 : number_of_lions = 200)
  (h2 : number_of_lions = 2 * number_of_leopards)
  (h3 : number_of_elephants = (number_of_lions + number_of_leopards) / 2) :
  number_of_lions + number_of_leopards + number_of_elephants = 450 :=
sorry

end animal_population_l37_37685


namespace increase_150_percent_of_80_l37_37210

theorem increase_150_percent_of_80 : 80 * 1.5 + 80 = 200 :=
by
  sorry

end increase_150_percent_of_80_l37_37210


namespace trigonometric_simplification_l37_37669

noncomputable def tan : ℝ → ℝ := λ x => Real.sin x / Real.cos x
noncomputable def simp_expr : ℝ :=
  (tan (96 * Real.pi / 180) - tan (12 * Real.pi / 180) * (1 + 1 / Real.sin (6 * Real.pi / 180)))
  /
  (1 + tan (96 * Real.pi / 180) * tan (12 * Real.pi / 180) * (1 + 1 / Real.sin (6 * Real.pi / 180)))

theorem trigonometric_simplification : simp_expr = Real.sqrt 3 / 3 :=
by
  sorry

end trigonometric_simplification_l37_37669


namespace avg_goals_per_game_l37_37643

def carter_goals_per_game := 4
def shelby_goals_per_game := carter_goals_per_game / 2
def judah_goals_per_game := (2 * shelby_goals_per_game) - 3
def average_total_goals_team := carter_goals_per_game + shelby_goals_per_game + judah_goals_per_game

theorem avg_goals_per_game : average_total_goals_team = 7 :=
by
  -- Proof would go here
  sorry

end avg_goals_per_game_l37_37643


namespace total_sugar_weight_l37_37205

theorem total_sugar_weight (x y : ℝ) (h1 : y - x = 8) (h2 : x - 1 = 0.6 * (y + 1)) : x + y = 40 := by
  sorry

end total_sugar_weight_l37_37205


namespace square_root_then_square_l37_37100

theorem square_root_then_square (x : ℕ) (hx : x = 49) : (Nat.sqrt x) ^ 2 = 49 := by
  sorry

end square_root_then_square_l37_37100


namespace expression_value_l37_37674

theorem expression_value {a b : ℝ} (h : a * b = -3) : a * Real.sqrt (-b / a) + b * Real.sqrt (-a / b) = 0 :=
by
  sorry

end expression_value_l37_37674


namespace total_drums_l37_37157

theorem total_drums (x y : ℕ) (hx : 30 * x + 20 * y = 160) : x + y = 7 :=
sorry

end total_drums_l37_37157


namespace plus_signs_count_l37_37906

theorem plus_signs_count (p m : ℕ) (h_sum : p + m = 23)
                         (h_max_minus : m ≤ 9) (h_max_plus : p ≤ 14)
                         (h_at_least_one_plus_in_10 : ∀ (s : Finset (Fin 23)), s.card = 10 → ∃ i ∈ s, i < p)
                         (h_at_least_one_minus_in_15 : ∀ (s : Finset (Fin 23)), s.card = 15 → ∃ i ∈ s, m ≤ i) :
  p = 14 :=
by sorry

end plus_signs_count_l37_37906


namespace rowing_speed_downstream_l37_37263

theorem rowing_speed_downstream (V_u V_s V_d : ℝ) (h1 : V_u = 10) (h2 : V_s = 15)
  (h3 : V_s = (V_u + V_d) / 2) : V_d = 20 := by
  sorry

end rowing_speed_downstream_l37_37263


namespace tina_took_away_2_oranges_l37_37194

-- Definition of the problem
def oranges_taken_away (x : ℕ) : Prop :=
  let original_oranges := 5
  let tangerines_left := 17 - 10 
  let oranges_left := original_oranges - x
  tangerines_left = oranges_left + 4 

-- The statement that needs to be proven
theorem tina_took_away_2_oranges : oranges_taken_away 2 :=
  sorry

end tina_took_away_2_oranges_l37_37194


namespace solve_for_product_l37_37031

theorem solve_for_product (a b c d : ℚ) (h1 : 3 * a + 4 * b + 6 * c + 8 * d = 48)
                          (h2 : 4 * (d + c) = b) 
                          (h3 : 4 * b + 2 * c = a) 
                          (h4 : c - 2 = d) : 
                          a * b * c * d = -1032192 / 1874161 := 
by 
  sorry

end solve_for_product_l37_37031


namespace worker_weekly_pay_l37_37486

variable (regular_rate : ℕ) -- Regular rate of Rs. 10 per survey
variable (total_surveys : ℕ) -- Worker completes 100 surveys per week
variable (cellphone_surveys : ℕ) -- 60 surveys involve the use of cellphone
variable (increased_rate : ℕ) -- Increased rate 30% higher than regular rate

-- Defining given values
def reg_rate : ℕ := 10
def total_survey_count : ℕ := 100
def cellphone_survey_count : ℕ := 60
def inc_rate : ℕ := reg_rate + 3

-- Calculating payments
def regular_survey_count : ℕ := total_survey_count - cellphone_survey_count
def regular_pay : ℕ := regular_survey_count * reg_rate
def cellphone_pay : ℕ := cellphone_survey_count * inc_rate

-- Total pay calculation
def total_pay : ℕ := regular_pay + cellphone_pay

-- Theorem to be proved
theorem worker_weekly_pay : total_pay = 1180 := 
by
  -- instantiate variables
  let regular_rate := reg_rate
  let total_surveys := total_survey_count
  let cellphone_surveys := cellphone_survey_count
  let increased_rate := inc_rate
  
  -- skip proof
  sorry

end worker_weekly_pay_l37_37486


namespace jane_total_score_l37_37077

theorem jane_total_score :
  let correct_answers := 17
  let incorrect_answers := 12
  let unanswered_questions := 6
  let total_questions := 35
  let points_per_correct := 1
  let points_per_incorrect := -0.25
  let correct_points := correct_answers * points_per_correct
  let incorrect_points := incorrect_answers * points_per_incorrect
  let total_score := correct_points + incorrect_points
  total_score = 14 :=
by
  sorry

end jane_total_score_l37_37077


namespace marias_workday_ends_at_six_pm_l37_37244

theorem marias_workday_ends_at_six_pm :
  ∀ (start_time : ℕ) (work_hours : ℕ) (lunch_start_time : ℕ) (lunch_duration : ℕ) (afternoon_break_time : ℕ) (afternoon_break_duration : ℕ) (end_time : ℕ),
    start_time = 8 ∧
    work_hours = 8 ∧
    lunch_start_time = 13 ∧
    lunch_duration = 1 ∧
    afternoon_break_time = 15 * 60 + 30 ∧  -- Converting 3:30 P.M. to minutes
    afternoon_break_duration = 15 ∧
    end_time = 18  -- 6:00 P.M. in 24-hour format
    → end_time = 18 :=
by
  -- map 13:00 -> 1:00 P.M.,  15:30 -> 3:30 P.M.; convert 6:00 P.M. back 
  sorry

end marias_workday_ends_at_six_pm_l37_37244


namespace cookies_leftover_l37_37089

def amelia_cookies := 52
def benjamin_cookies := 63
def chloe_cookies := 25
def total_cookies := amelia_cookies + benjamin_cookies + chloe_cookies
def package_size := 15

theorem cookies_leftover :
  total_cookies % package_size = 5 := by
  sorry

end cookies_leftover_l37_37089


namespace product_mod_7_l37_37283

theorem product_mod_7 (a b c : ℕ) (ha : a % 7 = 3) (hb : b % 7 = 4) (hc : c % 7 = 5) : 
  (a * b * c) % 7 = 4 :=
sorry

end product_mod_7_l37_37283


namespace all_tutors_work_together_in_90_days_l37_37038

theorem all_tutors_work_together_in_90_days :
  lcm 5 (lcm 6 (lcm 9 10)) = 90 := by
  sorry

end all_tutors_work_together_in_90_days_l37_37038


namespace board_rook_placement_l37_37401

-- Define the color function for the board
def color (n i j : ℕ) : ℕ :=
  min (i + j - 1) (2 * n - i - j + 1)

-- Conditions: It is possible to place n rooks such that no two attack each other and 
-- no two rooks stand on cells of the same color
def non_attacking_rooks (n : ℕ) (rooks : Fin n → Fin n) : Prop :=
  ∀ i j : Fin n, i ≠ j → rooks i ≠ rooks j ∧ color n i.val (rooks i).val ≠ color n j.val (rooks j).val

-- Main theorem to be proven
theorem board_rook_placement (n : ℕ) :
  (∃ rooks : Fin n → Fin n, non_attacking_rooks n rooks) →
  n % 4 = 0 ∨ n % 4 = 1 :=
by
  intros h
  sorry

end board_rook_placement_l37_37401


namespace log_lt_x_l37_37872

theorem log_lt_x (x : ℝ) (hx : 0 < x) : Real.log (1 + x) < x := 
sorry

end log_lt_x_l37_37872


namespace ice_palace_steps_l37_37389

theorem ice_palace_steps (time_for_20_steps total_time : ℕ) (h1 : time_for_20_steps = 120) (h2 : total_time = 180) : 
  total_time * 20 / time_for_20_steps = 30 := by
  have time_per_step : ℕ := time_for_20_steps / 20
  have total_steps : ℕ := total_time / time_per_step
  sorry

end ice_palace_steps_l37_37389


namespace parallel_lines_a_unique_l37_37066

theorem parallel_lines_a_unique (a : ℝ) :
  (∀ x y : ℝ, x + (a + 1) * y + (a^2 - 1) = 0 → x + 2 * y = 0 → -a / 2 = -1 / (a + 1)) →
  a = -2 :=
by
  sorry

end parallel_lines_a_unique_l37_37066


namespace number_of_tables_l37_37504

theorem number_of_tables (last_year_distance : ℕ) (factor : ℕ) 
  (distance_between_table_1_and_3 : ℕ) (number_of_tables : ℕ) :
  (last_year_distance = 300) ∧ 
  (factor = 4) ∧ 
  (distance_between_table_1_and_3 = 400) ∧
  (number_of_tables = ((factor * last_year_distance) / (distance_between_table_1_and_3 / 2)) + 1) 
  → number_of_tables = 7 :=
by
  intros
  sorry

end number_of_tables_l37_37504


namespace value_of_a_l37_37255

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 2 → x^2 - x + a < 0) → a = -2 :=
by
  intro h
  sorry

end value_of_a_l37_37255


namespace smallest_a_mod_remainders_l37_37596

theorem smallest_a_mod_remainders:
  (∀ d ∈ [2, 3, 4, 5, 6, 7, 8, 9], 2521 % d = 1) ∧
  (∀ n : ℕ, ∃ a : ℕ, a = 2520 * n + 1 ∧ (∀ d ∈ [2, 3, 4, 5, 6, 7, 8, 9], a % d = 1)) :=
by
  sorry

end smallest_a_mod_remainders_l37_37596


namespace contrapositive_true_l37_37970

theorem contrapositive_true (h : ∀ x : ℝ, x < 0 → x^2 > 0) : 
  (∀ x : ℝ, ¬ (x^2 > 0) → ¬ (x < 0)) :=
by 
  sorry

end contrapositive_true_l37_37970


namespace sum_of_squares_l37_37023

open Int

theorem sum_of_squares (p q r s t u : ℤ) (h : ∀ x : ℤ, 343 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) :
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 3506 :=
sorry

end sum_of_squares_l37_37023


namespace find_bettys_balance_l37_37101

-- Define the conditions as hypotheses
def balance_in_bettys_account (B : ℕ) : Prop :=
  -- Gina has two accounts with a combined balance equal to $1,728
  (2 * (B / 4)) = 1728

-- State the theorem to be proven
theorem find_bettys_balance (B : ℕ) (h : balance_in_bettys_account B) : B = 3456 :=
by
  -- The proof is provided here as a "sorry"
  sorry

end find_bettys_balance_l37_37101


namespace increase_by_50_percent_l37_37987

def original : ℕ := 350
def increase_percent : ℕ := 50
def increased_number : ℕ := original * increase_percent / 100
def final_number : ℕ := original + increased_number

theorem increase_by_50_percent : final_number = 525 := 
by
  sorry

end increase_by_50_percent_l37_37987


namespace ratio_of_ages_l37_37045

theorem ratio_of_ages (D R : ℕ) (h1 : D = 3) (h2 : R + 22 = 26) : R / D = 4 / 3 := by
  sorry

end ratio_of_ages_l37_37045


namespace lisa_total_miles_flown_l37_37795

variable (distance_per_trip : ℝ := 256.0)
variable (number_of_trips : ℝ := 32.0)

theorem lisa_total_miles_flown : distance_per_trip * number_of_trips = 8192.0 := by
  sorry

end lisa_total_miles_flown_l37_37795


namespace average_price_of_cow_l37_37696

variable (price_cow price_goat : ℝ)

theorem average_price_of_cow (h1 : 2 * price_cow + 8 * price_goat = 1400)
                             (h2 : price_goat = 60) :
                             price_cow = 460 := 
by
  -- The following line allows the Lean code to compile successfully without providing a proof.
  sorry

end average_price_of_cow_l37_37696


namespace trapezoid_area_l37_37127

-- Define the given conditions in the problem
variables (EF GH h EG FH : ℝ)
variables (EF_parallel_GH : true) -- EF and GH are parallel (not used in the calculation)
variables (EF_eq_70 : EF = 70)
variables (GH_eq_40 : GH = 40)
variables (h_eq_15 : h = 15)
variables (EG_eq_20 : EG = 20)
variables (FH_eq_25 : FH = 25)

-- Define the main theorem to prove
theorem trapezoid_area (EF GH h EG FH : ℝ) 
  (EF_eq_70 : EF = 70) 
  (GH_eq_40 : GH = 40) 
  (h_eq_15 : h = 15) 
  (EG_eq_20 : EG = 20) 
  (FH_eq_25 : FH = 25) : 
  0.5 * (EF + GH) * h = 825 := 
by 
  sorry

end trapezoid_area_l37_37127


namespace find_pointA_coordinates_l37_37885

-- Define point B
def pointB : ℝ × ℝ := (4, -1)

-- Define the symmetry condition with respect to the x-axis
def symmetricWithRespectToXAxis (p₁ p₂ : ℝ × ℝ) : Prop :=
  p₁.1 = p₂.1 ∧ p₁.2 = -p₂.2

-- Theorem statement: Prove the coordinates of point A given the conditions
theorem find_pointA_coordinates :
  ∃ A : ℝ × ℝ, symmetricWithRespectToXAxis pointB A ∧ A = (4, 1) :=
by
  sorry

end find_pointA_coordinates_l37_37885


namespace pos_rel_lines_l37_37303

-- Definition of the lines
def line1 (k : ℝ) (x y : ℝ) : Prop := 2 * x - y + k = 0
def line2 (x y : ℝ) : Prop := 4 * x - 2 * y + 1 = 0

-- Theorem stating the positional relationship between the two lines
theorem pos_rel_lines (k : ℝ) : 
  (∀ x y : ℝ, line1 k x y → line2 x y → 2 * k - 1 = 0) → 
  (∀ x y : ℝ, line1 k x y → ¬ line2 x y → 2 * k - 1 ≠ 0) → 
  (k = 1/2 ∨ k ≠ 1/2) :=
by sorry

end pos_rel_lines_l37_37303


namespace train_speed_l37_37822

theorem train_speed (len_train len_bridge time : ℝ) (h_len_train : len_train = 120)
  (h_len_bridge : len_bridge = 150) (h_time : time = 26.997840172786177) :
  let total_distance := len_train + len_bridge
  let speed_m_s := total_distance / time
  let speed_km_h := speed_m_s * 3.6
  speed_km_h = 36 :=
by
  -- Proof goes here
  sorry

end train_speed_l37_37822


namespace inequality_log_range_of_a_l37_37845

open Real

theorem inequality_log (x : ℝ) (h₀ : 0 < x) : 
  1 - 1 / x ≤ log x ∧ log x ≤ x - 1 := sorry

theorem range_of_a (a : ℝ) (h : ∀ (x : ℝ), 0 < x ∧ x ≤ 1 → a * (1 - x^2) + x^2 * log x ≥ 0) : 
  a ≥ 1/2 := sorry

end inequality_log_range_of_a_l37_37845


namespace rectangle_width_l37_37341

theorem rectangle_width (side_length_square : ℕ) (length_rectangle : ℕ) (area_equal : side_length_square * side_length_square = length_rectangle * w) : w = 4 := by
  sorry

end rectangle_width_l37_37341


namespace proof_parabola_statements_l37_37802

theorem proof_parabola_statements (b c : ℝ)
  (h1 : 1/2 - b + c < 0)
  (h2 : 2 - 2 * b + c < 0) :
  (b^2 > 2 * c) ∧
  (c > 1 → b > 3/2) ∧
  (∀ (m1 m2 : ℝ), m1 < m2 ∧ m2 < b → ∀ (y : ℝ), y = (1/2)*m1^2 - b*m1 + c → ∀ (y2 : ℝ), y2 = (1/2)*m2^2 - b*m2 + c → y > y2) ∧
  (¬(∃ x1 x2 : ℝ, (1/2) * x1^2 - b * x1 + c = 0 ∧ (1/2) * x2^2 - b * x2 + c = 0 ∧ x1 + x2 > 3)) :=
by sorry

end proof_parabola_statements_l37_37802


namespace sara_has_green_marbles_l37_37261

-- Definition of the total number of green marbles and Tom's green marbles
def total_green_marbles : ℕ := 7
def tom_green_marbles : ℕ := 4

-- Definition of Sara's green marbles
def sara_green_marbles : ℕ := total_green_marbles - tom_green_marbles

-- The proof statement
theorem sara_has_green_marbles : sara_green_marbles = 3 :=
by
  -- The proof will be filled in here
  sorry

end sara_has_green_marbles_l37_37261


namespace original_price_of_shoes_l37_37163

theorem original_price_of_shoes (P : ℝ) (h1 : 0.25 * P = 51) : P = 204 := 
by 
  sorry

end original_price_of_shoes_l37_37163


namespace greatest_possible_x_lcm_l37_37307

theorem greatest_possible_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105): x = 105 := 
sorry

end greatest_possible_x_lcm_l37_37307


namespace initial_comparison_discount_comparison_B_based_on_discounted_A_l37_37243

noncomputable section

-- Definitions based on the problem conditions
def A_price (x : ℝ) : ℝ := x
def B_price (x : ℝ) : ℝ := (0.2 * 2 * x + 0.3 * 3 * x + 0.4 * 4 * x) / 3
def A_discount_price (x : ℝ) : ℝ := 0.9 * x

-- Initial comparison
theorem initial_comparison (x : ℝ) (h : 0 < x) : B_price x < A_price x :=
by {
  sorry
}

-- After A's discount comparison
theorem discount_comparison (x : ℝ) (h : 0 < x) : A_discount_price x < B_price x :=
by {
  sorry
}

-- B's price based on A’s discounted price comparison
theorem B_based_on_discounted_A (x : ℝ) (h : 0 < x) : B_price (A_discount_price x) < A_discount_price x :=
by {
  sorry
}

end initial_comparison_discount_comparison_B_based_on_discounted_A_l37_37243


namespace greatest_number_in_consecutive_multiples_l37_37857

theorem greatest_number_in_consecutive_multiples (s : Set ℕ) (h₁ : ∃ m : ℕ, s = {n | ∃ k < 100, n = 8 * (m + k)} ∧ m = 14) :
  (∃ n ∈ s, ∀ x ∈ s, x ≤ n) →
  ∃ n ∈ s, n = 904 :=
by
  sorry

end greatest_number_in_consecutive_multiples_l37_37857


namespace multiple_of_sum_squares_l37_37425

theorem multiple_of_sum_squares (a b c : ℕ) (h1 : a < 2017) (h2 : b < 2017) (h3 : c < 2017) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a)
    (h7 : ∃ k1, a^3 - b^3 = k1 * 2017) (h8 : ∃ k2, b^3 - c^3 = k2 * 2017) (h9 : ∃ k3, c^3 - a^3 = k3 * 2017) :
    ∃ k, a^2 + b^2 + c^2 = k * (a + b + c) :=
by
  sorry

end multiple_of_sum_squares_l37_37425


namespace euler_characteristic_convex_polyhedron_l37_37046

-- Define the context of convex polyhedron with vertices (V), edges (E), and faces (F)
structure ConvexPolyhedron :=
  (V : ℕ) -- number of vertices
  (E : ℕ) -- number of edges
  (F : ℕ) -- number of faces
  (convex : Prop) -- property stating the polyhedron is convex

-- Euler characteristic theorem for convex polyhedra
theorem euler_characteristic_convex_polyhedron (P : ConvexPolyhedron) (h : P.convex) : P.V - P.E + P.F = 2 :=
sorry

end euler_characteristic_convex_polyhedron_l37_37046


namespace income_of_A_l37_37922

theorem income_of_A (A B C : ℝ) 
  (h1 : (A + B) / 2 = 4050) 
  (h2 : (B + C) / 2 = 5250) 
  (h3 : (A + C) / 2 = 4200) : 
  A = 3000 :=
by
  sorry

end income_of_A_l37_37922


namespace point_above_line_range_l37_37478

theorem point_above_line_range (a : ℝ) :
  (2 * a - (-1) + 1 < 0) ↔ a < -1 :=
by
  sorry

end point_above_line_range_l37_37478


namespace middle_aged_participating_l37_37858

-- Definitions of the given conditions
def total_employees : Nat := 1200
def ratio (elderly middle_aged young : Nat) := elderly = 1 ∧ middle_aged = 5 ∧ young = 6
def selected_employees : Nat := 36

-- The stratified sampling condition implies
def stratified_sampling (elderly middle_aged young : Nat) (total : Nat) (selected : Nat) :=
  (elderly + middle_aged + young = total) ∧
  (selected = 36)

-- The proof statement
theorem middle_aged_participating (elderly middle_aged young : Nat) (total : Nat) (selected : Nat) 
  (h_ratio : ratio elderly middle_aged young) 
  (h_total : total = total_employees)
  (h_sampled : stratified_sampling elderly middle_aged young (elderly + middle_aged + young) selected) : 
  selected * middle_aged / (elderly + middle_aged + young) = 15 := 
by sorry

end middle_aged_participating_l37_37858


namespace initial_wage_of_illiterate_l37_37668

-- Definitions from the conditions
def illiterate_employees : ℕ := 20
def literate_employees : ℕ := 10
def total_employees := illiterate_employees + literate_employees

-- Given that the daily average wages of illiterate employees decreased to Rs. 10
def daily_wages_after_decrease : ℝ := 10
-- The total decrease in the average salary of all employees by Rs. 10 per day
def decrease_in_avg_wage : ℝ := 10

-- To be proved: the initial daily average wage of the illiterate employees was Rs. 25.
theorem initial_wage_of_illiterate (I : ℝ) :
  (illiterate_employees * I - illiterate_employees * daily_wages_after_decrease = total_employees * decrease_in_avg_wage) → 
  I = 25 := 
by
  sorry

end initial_wage_of_illiterate_l37_37668


namespace common_factor_l37_37752

-- Definition of the polynomial
def polynomial (x y m n : ℝ) : ℝ := 4 * x * (m - n) + 2 * y * (m - n) ^ 2

-- The theorem statement
theorem common_factor (x y m n : ℝ) : ∃ k : ℝ, k * (m - n) = polynomial x y m n :=
sorry

end common_factor_l37_37752


namespace maximize_quadratic_function_l37_37469

theorem maximize_quadratic_function (x : ℝ) :
  (∀ x, -2 * x ^ 2 - 8 * x + 18 ≤ 26) ∧ (-2 * (-2) ^ 2 - 8 * (-2) + 18 = 26) :=
by (
  sorry
)

end maximize_quadratic_function_l37_37469


namespace part_a_l37_37455

theorem part_a (x : ℝ) (hx : x > 0) :
  ∃ color : ℕ, ∃ p1 p2 : ℝ × ℝ, (p1 = p2 ∨ x = dist p1 p2) :=
sorry

end part_a_l37_37455


namespace olivia_total_payment_l37_37136

theorem olivia_total_payment : 
  (4 / 4 + 12 / 4 = 4) :=
by
  sorry

end olivia_total_payment_l37_37136


namespace total_marbles_l37_37980

theorem total_marbles (r b g y : ℝ)
  (h1 : r = 1.35 * b)
  (h2 : g = 1.5 * r)
  (h3 : y = 2 * b) :
  r + b + g + y = 4.72 * r :=
by
  sorry

end total_marbles_l37_37980


namespace rational_function_value_l37_37600

theorem rational_function_value (g : ℚ → ℚ) (h : ∀ x : ℚ, x ≠ 0 → 4 * g (x⁻¹) + 3 * g x / x = 2 * x^3) : g (-1) = -2 :=
sorry

end rational_function_value_l37_37600


namespace negation_of_exists_leq_l37_37166

theorem negation_of_exists_leq (
  P : ∃ x : ℝ, x^2 - 2 * x + 4 ≤ 0
) : ∀ x : ℝ, x^2 - 2 * x + 4 > 0 :=
sorry

end negation_of_exists_leq_l37_37166


namespace side_length_percentage_error_l37_37997

variable (s s' : Real)
-- Conditions
-- s' = s * 1.06 (measured side length is 6% more than actual side length)
-- (s'^2 - s^2) / s^2 * 100% = 12.36% (percentage error in area)

theorem side_length_percentage_error 
    (h1 : s' = s * 1.06)
    (h2 : (s'^2 - s^2) / s^2 * 100 = 12.36) :
    ((s' - s) / s) * 100 = 6 := 
sorry

end side_length_percentage_error_l37_37997


namespace find_x_l37_37874

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x+1, -x)

def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem find_x (x : ℝ) (h : perpendicular vector_a (vector_b x)) : x = 1 :=
by sorry

end find_x_l37_37874


namespace amelia_wins_l37_37834

noncomputable def amelia_wins_probability : ℚ := 21609 / 64328

theorem amelia_wins (h_am_heads : ℚ) (h_bl_heads : ℚ) (game_starts : Prop) (game_alternates : Prop) (win_condition : Prop) :
  h_am_heads = 3/7 ∧ h_bl_heads = 1/3 ∧ game_starts ∧ game_alternates ∧ win_condition →
  amelia_wins_probability = 21609 / 64328 :=
sorry

end amelia_wins_l37_37834


namespace shea_buys_corn_l37_37657

noncomputable def num_pounds_corn (c b : ℚ) : ℚ :=
  if b + c = 24 ∧ 45 * b + 99 * c = 1809 then c else -1

theorem shea_buys_corn (c b : ℚ) : b + c = 24 ∧ 45 * b + 99 * c = 1809 → c = 13.5 :=
by
  intros h
  sorry

end shea_buys_corn_l37_37657


namespace ellipse_foci_y_axis_iff_l37_37586

theorem ellipse_foci_y_axis_iff (m n : ℝ) (h : m > n ∧ n > 0) :
  (m > n ∧ n > 0) ↔ (∀ (x y : ℝ), m * x^2 + n * y^2 = 1 → ∃ a b : ℝ, a^2 - b^2 = 1 ∧ x^2/b^2 + y^2/a^2 = 1 ∧ a > b) :=
sorry

end ellipse_foci_y_axis_iff_l37_37586


namespace eval_expression_l37_37200

theorem eval_expression :
  Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) - Int.ceil (2 / 3 : ℚ) = -1 := 
by 
  sorry

end eval_expression_l37_37200


namespace vasya_number_l37_37407

theorem vasya_number (a b c d : ℕ) (h1 : a * b = 21) (h2 : b * c = 20) (h3 : ∃ x, x ∈ [4, 7] ∧ a ≠ c ∧ b = 7 ∧ c = 4 ∧ d = 5) : (1000 * a + 100 * b + 10 * c + d) = 3745 :=
sorry

end vasya_number_l37_37407


namespace min_value_expression_l37_37456

noncomputable def f (x y : ℝ) : ℝ := 
  (x + 1 / y) * (x + 1 / y - 2023) + (y + 1 / x) * (y + 1 / x - 2023)

theorem min_value_expression : ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ f x y = -2048113 :=
sorry

end min_value_expression_l37_37456


namespace tablespoons_in_half_cup_l37_37124

theorem tablespoons_in_half_cup
    (grains_per_cup : ℕ)
    (half_cup : ℕ)
    (tbsp_to_tsp : ℕ)
    (grains_per_tsp : ℕ)
    (h1 : grains_per_cup = 480)
    (h2 : half_cup = grains_per_cup / 2)
    (h3 : tbsp_to_tsp = 3)
    (h4 : grains_per_tsp = 10) :
    (half_cup / (tbsp_to_tsp * grains_per_tsp) = 8) :=
by
  sorry

end tablespoons_in_half_cup_l37_37124


namespace greatest_integer_x_l37_37704

theorem greatest_integer_x
    (x : ℤ) : 
    (7 / 9 : ℚ) > (x : ℚ) / 13 → x ≤ 10 :=
by
    sorry

end greatest_integer_x_l37_37704


namespace total_money_at_least_108_l37_37746

-- Definitions for the problem
def tram_ticket_cost : ℕ := 1
def passenger_coins (n : ℕ) : Prop := n = 2 ∨ n = 5

-- Condition that conductor had no change initially
def initial_conductor_money : ℕ := 0

-- Condition that each passenger can pay exactly 1 Ft and receive change
def can_pay_ticket_with_change (coins : List ℕ) : Prop := 
  ∀ c ∈ coins, passenger_coins c → 
    ∃ change : List ℕ, (change.sum = c - tram_ticket_cost) ∧ 
      (∀ x ∈ change, passenger_coins x)

-- Assume we have 20 passengers with only 2 Ft and 5 Ft coins
def passengers_coins : List (List ℕ) :=
  -- Simplified representation
  List.replicate 20 [2, 5]

noncomputable def total_passenger_money : ℕ :=
  (passengers_coins.map List.sum).sum

-- Lean statement for the proof problem
theorem total_money_at_least_108 : total_passenger_money ≥ 108 :=
sorry

end total_money_at_least_108_l37_37746


namespace find_number_l37_37950

theorem find_number (x : ℝ) (h_Pos : x > 0) (h_Eq : x + 17 = 60 * (1/x)) : x = 3 :=
by
  sorry

end find_number_l37_37950


namespace find_b2_l37_37744

theorem find_b2 (b : ℕ → ℝ) (h1 : b 1 = 23) (h10 : b 10 = 123) 
  (h : ∀ n ≥ 3, b n = (b 1 + b 2 + (n - 3) * b 3) / (n - 1)) : b 2 = 223 :=
sorry

end find_b2_l37_37744


namespace tom_age_ratio_l37_37085

theorem tom_age_ratio (T N : ℕ) (h1 : sum_ages = T) (h2 : T - N = 3 * (sum_ages_N_years_ago))
  (h3 : sum_ages = T) (h4 : sum_ages_N_years_ago = T - 4 * N) :
  T / N = 11 / 2 := 
by
  sorry

end tom_age_ratio_l37_37085


namespace min_max_values_f_l37_37015

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_values_f :
  ∃ (a b : ℝ), a = -3 * Real.pi / 2 ∧ b = Real.pi / 2 + 2 ∧ 
                ∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ a ∧ f x ≤ b :=
by
  sorry

end min_max_values_f_l37_37015


namespace quadratic_unique_root_l37_37594

theorem quadratic_unique_root (b c : ℝ)
  (h₁ : b = c^2 + 1)
  (h₂ : (x^2 + b * x + c = 0) → ∃! x : ℝ, x^2 + b * x + c = 0) :
  c = 1 ∨ c = -1 := 
sorry

end quadratic_unique_root_l37_37594


namespace inequality_solution_set_l37_37267

theorem inequality_solution_set (x : ℝ) (h : x ≠ 2) :
  (1 / (x - 2) ≤ 1) ↔ (x < 2 ∨ 3 ≤ x) :=
sorry

end inequality_solution_set_l37_37267


namespace reflection_sum_coordinates_l37_37318

theorem reflection_sum_coordinates :
  ∀ (C D : ℝ × ℝ), 
  C = (5, -3) →
  D = (5, -C.2) →
  (C.1 + C.2 + D.1 + D.2 = 10) :=
by
  intros C D hC hD
  rw [hC, hD]
  simp
  sorry

end reflection_sum_coordinates_l37_37318


namespace function_parallel_l37_37888

theorem function_parallel {x y : ℝ} (h : y = -2 * x + 1) : 
    ∀ {a : ℝ}, y = -2 * a + 3 -> y = -2 * x + 1 := by
    sorry

end function_parallel_l37_37888


namespace max_rectangle_area_l37_37450

theorem max_rectangle_area (l w : ℕ) (h : 2 * (l + w) = 120) : l * w ≤ 900 :=
by 
  sorry

end max_rectangle_area_l37_37450


namespace marcella_pairs_l37_37169

theorem marcella_pairs (pairs_initial : ℕ) (shoes_lost : ℕ) (h1 : pairs_initial = 50) (h2 : shoes_lost = 15) :
  ∃ pairs_left : ℕ, pairs_left = 35 := 
by
  existsi 35
  sorry

end marcella_pairs_l37_37169


namespace john_twice_sam_in_years_l37_37009

noncomputable def current_age_sam : ℕ := 9
noncomputable def current_age_john : ℕ := 27

theorem john_twice_sam_in_years (Y : ℕ) :
  (current_age_john + Y = 2 * (current_age_sam + Y)) → Y = 9 := 
by 
  sorry

end john_twice_sam_in_years_l37_37009


namespace triangle_area_l37_37654

-- Define the line equation as a condition.
def line_equation (x : ℝ) : ℝ :=
  4 * x + 8

-- Define the y-intercept (condition 1).
def y_intercept := line_equation 0

-- Define the x-intercept (condition 2).
def x_intercept := (-8) / 4

-- Define the area of the triangle given the intercepts and prove it equals 8 (question and correct answer).
theorem triangle_area :
  (1 / 2) * abs x_intercept * y_intercept = 8 :=
by
  sorry

end triangle_area_l37_37654


namespace range_f_subset_interval_l37_37457

-- Define the function f on real numbers
def f : ℝ → ℝ := sorry

-- The given condition for all real numbers x and y such that x > y
axiom condition (x y : ℝ) (h : x > y) : (f x)^2 ≤ f y

-- The main theorem that needs to be proven
theorem range_f_subset_interval : ∀ x, 0 ≤ f x ∧ f x ≤ 1 := 
by
  intro x
  apply And.intro
  -- Proof for 0 ≤ f x
  sorry
  -- Proof for f x ≤ 1
  sorry

end range_f_subset_interval_l37_37457


namespace mul_powers_same_base_l37_37296

theorem mul_powers_same_base (a : ℝ) : a^3 * a^4 = a^7 := 
by 
  sorry

end mul_powers_same_base_l37_37296


namespace no_real_roots_of_ffx_or_ggx_l37_37273

noncomputable def is_unitary_quadratic_trinomial (p : ℝ → ℝ) : Prop :=
∃ b c : ℝ, ∀ x : ℝ, p x = x^2 + b*x + c

theorem no_real_roots_of_ffx_or_ggx 
    (f g : ℝ → ℝ) 
    (hf : is_unitary_quadratic_trinomial f) 
    (hg : is_unitary_quadratic_trinomial g)
    (hf_ng : ∀ x : ℝ, f (g x) ≠ 0)
    (hg_nf : ∀ x : ℝ, g (f x) ≠ 0) :
    (∀ x : ℝ, f (f x) ≠ 0) ∨ (∀ x : ℝ, g (g x) ≠ 0) :=
sorry

end no_real_roots_of_ffx_or_ggx_l37_37273


namespace geometric_arithmetic_sequence_l37_37907

theorem geometric_arithmetic_sequence 
  (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (q : ℝ) 
  (h0 : 0 < q) (h1 : q ≠ 1)
  (h2 : ∀ n, a_n n = a_n 1 * q ^ (n - 1)) -- a_n is a geometric sequence
  (h3 : 2 * a_n 3 * a_n 5 = a_n 4 * (a_n 3 + a_n 5)) -- a3, a5, a4 form an arithmetic sequence
  (h4 : ∀ n, S n = a_n 1 * (1 - q^n) / (1 - q)) -- S_n is the sum of the first n terms
  : S 6 / S 3 = 9 / 8 :=
by
  sorry

end geometric_arithmetic_sequence_l37_37907


namespace find_principal_amount_l37_37908

variables (P R : ℝ)

theorem find_principal_amount (h : (4 * P * (R + 2) / 100) - (4 * P * R / 100) = 56) : P = 700 :=
sorry

end find_principal_amount_l37_37908


namespace negation_of_proposition_l37_37612

variables (x : ℝ)

def proposition (x : ℝ) : Prop := x > 0 → (x ≠ 2 → (x^3 / (x - 2) > 0))

theorem negation_of_proposition : ∃ x : ℝ, x > 0 ∧ 0 ≤ x ∧ x ≤ 2 :=
by
  sorry

end negation_of_proposition_l37_37612


namespace train_speed_l37_37446

theorem train_speed (length : ℕ) (time : ℕ) (h1 : length = 1600) (h2 : time = 40) : length / time = 40 := 
by
  -- use the given conditions here
  sorry

end train_speed_l37_37446


namespace min_lcm_value_l37_37342

-- Definitions
def gcd_77 (a b c d : ℕ) : Prop :=
  Nat.gcd (Nat.gcd a b) (Nat.gcd c d) = 77

def lcm_n (a b c d n : ℕ) : Prop :=
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d) = n

-- Problem statement
theorem min_lcm_value :
  (∃ a b c d : ℕ, gcd_77 a b c d ∧ lcm_n a b c d 27720) ∧
  (∀ n : ℕ, (∃ a b c d : ℕ, gcd_77 a b c d ∧ lcm_n a b c d n) → 27720 ≤ n) :=
sorry

end min_lcm_value_l37_37342


namespace trapezoid_geometry_proof_l37_37700

theorem trapezoid_geometry_proof
  (midline_length : ℝ)
  (segment_midpoints : ℝ)
  (angle1 angle2 : ℝ)
  (h_midline : midline_length = 5)
  (h_segment_midpoints : segment_midpoints = 3)
  (h_angle1 : angle1 = 30)
  (h_angle2 : angle2 = 60) :
  ∃ (AD BC AB : ℝ), AD = 8 ∧ BC = 2 ∧ AB = 3 :=
by
  sorry

end trapezoid_geometry_proof_l37_37700


namespace price_reduction_l37_37043

variable (x : ℝ)

theorem price_reduction :
  28 * (1 - x) * (1 - x) = 16 :=
sorry

end price_reduction_l37_37043


namespace find_boys_and_girls_l37_37827

noncomputable def number_of_boys_and_girls (a b c d : Nat) : (Nat × Nat) := sorry

theorem find_boys_and_girls : 
  ∃ m d : Nat,
  (∀ (a b c : Nat), 
    ((a = 15 ∨ b = 18 ∨ c = 13) ∧ 
    (a.mod 4 = 3 ∨ b.mod 4 = 2 ∨ c.mod 4 = 1)) 
    → number_of_boys_and_girls a b c d = (16, 14)) :=
sorry

end find_boys_and_girls_l37_37827


namespace solve_abs_equation_l37_37052

theorem solve_abs_equation (x : ℝ) (h : abs (x - 20) + abs (x - 18) = abs (2 * x - 36)) : x = 19 :=
sorry

end solve_abs_equation_l37_37052


namespace solve_for_m_l37_37350

theorem solve_for_m (m α : ℝ) (h1 : Real.tan α = m / 3) (h2 : Real.tan (α + Real.pi / 4) = 2 / m) :
  m = -6 ∨ m = 1 :=
by
  sorry

end solve_for_m_l37_37350


namespace trig_identity_l37_37466

theorem trig_identity :
  (Real.cos (80 * Real.pi / 180) * Real.cos (35 * Real.pi / 180) + 
   Real.sin (80 * Real.pi / 180) * Real.cos (55 * Real.pi / 180)) = 
  (Real.sqrt 2 / 2) :=
by
  sorry

end trig_identity_l37_37466


namespace major_axis_length_l37_37629

-- Define the problem setup
structure Cylinder :=
  (base_radius : ℝ)
  (height : ℝ)

structure Sphere :=
  (radius : ℝ)

-- Define the conditions
def cylinder : Cylinder :=
  { base_radius := 6, height := 0 }  -- height isn't significant for this problem

def sphere1 : Sphere :=
  { radius := 6 }

def sphere2 : Sphere :=
  { radius := 6 }

def distance_between_centers : ℝ :=
  13

-- Statement of the problem in Lean 4
theorem major_axis_length : 
  cylinder.base_radius = 6 →
  sphere1.radius = 6 →
  sphere2.radius = 6 →
  distance_between_centers = 13 →
  ∃ major_axis_length : ℝ, major_axis_length = 13 :=
by
  intros h1 h2 h3 h4
  existsi 13
  sorry

end major_axis_length_l37_37629


namespace log_identity_l37_37329

noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

theorem log_identity : log 2 5 * log 3 2 * log 5 3 = 1 :=
by sorry

end log_identity_l37_37329
