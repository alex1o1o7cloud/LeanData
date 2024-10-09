import Mathlib

namespace brian_distance_more_miles_l2419_241962

variables (s t d m n : ℝ)
-- Mike's distance
variable (hd : d = s * t)
-- Steve's distance condition
variable (hsteve : d + 90 = (s + 6) * (t + 1.5))
-- Brian's distance
variable (hbrian : m = (s + 12) * (t + 3))

theorem brian_distance_more_miles :
  n = m - d → n = 200 :=
sorry

end brian_distance_more_miles_l2419_241962


namespace ellipse_foci_coordinates_l2419_241986

theorem ellipse_foci_coordinates (x y : ℝ) :
  2 * x^2 + 3 * y^2 = 1 →
  (∃ c : ℝ, (c = (Real.sqrt 6) / 6) ∧ ((x = c ∧ y = 0) ∨ (x = -c ∧ y = 0))) :=
by
  sorry

end ellipse_foci_coordinates_l2419_241986


namespace inequality_for_large_exponent_l2419_241912

theorem inequality_for_large_exponent (u : ℕ → ℕ) (x : ℕ) (k : ℕ) (hk : k = 100) (hu : u x = 2^x) : 
  2^(2^(x : ℕ)) > 2^(k * x) :=
by 
  sorry

end inequality_for_large_exponent_l2419_241912


namespace largest_root_ratio_l2419_241944

-- Define the polynomials f(x) and g(x)
def f (x : ℝ) : ℝ := 1 - x - 4 * x^2 + x^4
def g (x : ℝ) : ℝ := 16 - 8 * x - 16 * x^2 + x^4

-- Define the property that x1 is the largest root of f(x) and x2 is the largest root of g(x)
def is_largest_root (p : ℝ → ℝ) (r : ℝ) : Prop := 
  p r = 0 ∧ ∀ x : ℝ, p x = 0 → x ≤ r

-- The main theorem
theorem largest_root_ratio (x1 x2 : ℝ) 
  (hx1 : is_largest_root f x1) 
  (hx2 : is_largest_root g x2) : x2 = 2 * x1 :=
sorry

end largest_root_ratio_l2419_241944


namespace removed_number_is_34_l2419_241919
open Real

theorem removed_number_is_34 (n : ℕ) (x : ℕ) (h₁ : 946 = (43 * (43 + 1)) / 2) (h₂ : 912 = 43 * (152 / 7)) : x = 34 :=
by
  sorry

end removed_number_is_34_l2419_241919


namespace tv_interest_rate_zero_l2419_241984

theorem tv_interest_rate_zero (price_installment first_installment last_installment : ℕ) 
  (installment_count : ℕ) (total_price : ℕ) : 
  total_price = 60000 ∧  
  price_installment = 1000 ∧ 
  first_installment = price_installment ∧ 
  last_installment = 59000 ∧ 
  installment_count = 20 ∧  
  (20 * price_installment = 20000) ∧
  (total_price - first_installment = 59000) →
  0 = 0 :=
by 
  sorry

end tv_interest_rate_zero_l2419_241984


namespace geometric_sequence_q_cubed_l2419_241974

noncomputable def S (a_1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a_1 else a_1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_q_cubed (a_1 q : ℝ) (h1 : q ≠ 1) (h2 : a_1 ≠ 0)
  (h3 : S a_1 q 3 + S a_1 q 6 = 2 * S a_1 q 9) : q^3 = -1 / 2 :=
by
  sorry

end geometric_sequence_q_cubed_l2419_241974


namespace perm_prime_count_12345_l2419_241987

theorem perm_prime_count_12345 : 
  (∀ x : List ℕ, (x ∈ (List.permutations [1, 2, 3, 4, 5])) → 
    (10^4 * x.head! + 10^3 * x.tail.head! + 10^2 * x.tail.tail.head! + 10 * x.tail.tail.tail.head! + x.tail.tail.tail.tail.head!) % 3 = 0)
  → 
  0 = 0 :=
by
  sorry

end perm_prime_count_12345_l2419_241987


namespace caroline_lassis_l2419_241947

theorem caroline_lassis (c : ℕ → ℕ): c 3 = 13 → c 15 = 65 :=
by
  sorry

end caroline_lassis_l2419_241947


namespace initial_velocity_is_three_l2419_241999

noncomputable def displacement (t : ℝ) : ℝ :=
  3 * t - t^2

theorem initial_velocity_is_three : 
  (deriv displacement 0) = 3 :=
by
  sorry

end initial_velocity_is_three_l2419_241999


namespace probability_at_least_one_girl_l2419_241908

theorem probability_at_least_one_girl (total_students boys girls k : ℕ) (h_total: total_students = 5) (h_boys: boys = 3) (h_girls: girls = 2) (h_k: k = 3) : 
  (1 - ((Nat.choose boys k) / (Nat.choose total_students k))) = 9 / 10 :=
by
  sorry

end probability_at_least_one_girl_l2419_241908


namespace calories_per_person_l2419_241991

theorem calories_per_person 
  (oranges : ℕ)
  (pieces_per_orange : ℕ)
  (people : ℕ)
  (calories_per_orange : ℝ)
  (h_oranges : oranges = 7)
  (h_pieces_per_orange : pieces_per_orange = 12)
  (h_people : people = 6)
  (h_calories_per_orange : calories_per_orange = 80.0) :
  (oranges * pieces_per_orange / people) * (calories_per_orange / pieces_per_orange) = 93.3338 :=
by
  sorry

end calories_per_person_l2419_241991


namespace linear_function_general_form_special_case_linear_function_proof_quadratic_function_general_form_special_case_quadratic_function1_proof_special_case_quadratic_function2_proof_l2419_241988

variable {α : Type*} [Ring α]

def linear_function (a b x : α) : α :=
  a * x + b

def special_case_linear_function (a x : α) : α :=
  a * x

def quadratic_function (a b c x : α) : α :=
  a * x^2 + b * x + c

def special_case_quadratic_function1 (a c x : α) : α :=
  a * x^2 + c

def special_case_quadratic_function2 (a x : α) : α :=
  a * x^2

theorem linear_function_general_form (a b x : α) :
  ∃ y, y = linear_function a b x := by
  sorry

theorem special_case_linear_function_proof (a x : α) :
  ∃ y, y = special_case_linear_function a x := by
  sorry

theorem quadratic_function_general_form (a b c x : α) :
  a ≠ 0 → ∃ y, y = quadratic_function a b c x := by
  sorry

theorem special_case_quadratic_function1_proof (a b c x : α) :
  a ≠ 0 → b = 0 → ∃ y, y = special_case_quadratic_function1 a c x := by
  sorry

theorem special_case_quadratic_function2_proof (a b c x : α) :
  a ≠ 0 → b = 0 → c = 0 → ∃ y, y = special_case_quadratic_function2 a x := by
  sorry

end linear_function_general_form_special_case_linear_function_proof_quadratic_function_general_form_special_case_quadratic_function1_proof_special_case_quadratic_function2_proof_l2419_241988


namespace isosceles_triangle_largest_angle_l2419_241983

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h1 : A = B) (h2 : C = 50) (h3 : A + B + C = 180) : max A C = 80 :=
by 
  -- Define additional facts about the triangle, e.g., A = B = 50, and sum of angles = 180.
  have h4 : A = 50 := sorry
  rw [h4, h2] at h3
  -- Prove the final result using the given conditions.
  sorry

end isosceles_triangle_largest_angle_l2419_241983


namespace arithmetic_problem_l2419_241913

theorem arithmetic_problem : 245 - 57 + 136 + 14 - 38 = 300 := by
  sorry

end arithmetic_problem_l2419_241913


namespace relationship_y1_y2_y3_l2419_241928

theorem relationship_y1_y2_y3 (c y1 y2 y3 : ℝ) :
  (y1 = (-(1^2) + 2 * 1 + c))
  ∧ (y2 = (-(2^2) + 2 * 2 + c))
  ∧ (y3 = (-(5^2) + 2 * 5 + c))
  → (y2 > y1 ∧ y1 > y3) :=
by
  intro h
  sorry

end relationship_y1_y2_y3_l2419_241928


namespace inverse_function_composition_l2419_241977

def g (x : ℝ) : ℝ := 3 * x + 7

noncomputable def g_inv (y : ℝ) : ℝ := (y - 7) / 3

theorem inverse_function_composition : g_inv (g_inv 20) = -8 / 9 := by
  sorry

end inverse_function_composition_l2419_241977


namespace remaining_distance_l2419_241993

-- Definitions of the given conditions
def D : ℕ := 500
def daily_alpha : ℕ := 30
def daily_beta : ℕ := 50
def effective_beta : ℕ := daily_beta / 2

-- Proving the theorem with given conditions
theorem remaining_distance (n : ℕ) (h : n = 25) :
  D - daily_alpha * n = 2 * (D - effective_beta * n) :=
by
  sorry

end remaining_distance_l2419_241993


namespace find_a2_b2_geom_sequences_unique_c_l2419_241911

-- Define the sequences as per the problem statement
def seqs (a b : ℕ → ℝ) :=
  a 1 = 0 ∧ b 1 = 2013 ∧
  ∀ n : ℕ, (1 ≤ n → (2 * a (n+1) = a n + b n)) ∧ (1 ≤ n → (4 * b (n+1) = a n + 3 * b n))

-- (1) Find values of a_2 and b_2
theorem find_a2_b2 {a b : ℕ → ℝ} (h : seqs a b) :
  a 2 = 1006.5 ∧ b 2 = 1509.75 :=
sorry

-- (2) Prove that {a_n - b_n} and {a_n + 2b_n} are geometric sequences
theorem geom_sequences {a b : ℕ → ℝ} (h : seqs a b) :
  ∃ r s : ℝ, (∃ c : ℝ, ∀ n : ℕ, a n - b n = c * r^n) ∧
             (∃ d : ℝ, ∀ n : ℕ, a n + 2 * b n = d * s^n) :=
sorry

-- (3) Prove there is a unique positive integer c such that a_n < c < b_n always holds
theorem unique_c {a b : ℕ → ℝ} (h : seqs a b) :
  ∃! c : ℝ, (0 < c) ∧ (∀ n : ℕ, 1 ≤ n → a n < c ∧ c < b n) :=
sorry

end find_a2_b2_geom_sequences_unique_c_l2419_241911


namespace probability_calculation_correct_l2419_241929

def total_balls : ℕ := 100
def white_balls : ℕ := 50
def green_balls : ℕ := 20
def yellow_balls : ℕ := 10
def red_balls : ℕ := 17
def purple_balls : ℕ := 3

def number_of_non_red_or_purple_balls : ℕ := total_balls - (red_balls + purple_balls)

def probability_of_non_red_or_purple : ℚ := number_of_non_red_or_purple_balls / total_balls

theorem probability_calculation_correct :
  probability_of_non_red_or_purple = 0.8 := 
  by 
    -- proof goes here
    sorry

end probability_calculation_correct_l2419_241929


namespace graph_of_equation_is_two_lines_l2419_241951

theorem graph_of_equation_is_two_lines :
  ∀ (x y : ℝ), (x * y - 2 * x + 3 * y - 6 = 0) ↔ ((x + 3 = 0) ∨ (y - 2 = 0)) := 
by
  intro x y
  sorry

end graph_of_equation_is_two_lines_l2419_241951


namespace like_terms_sum_l2419_241927

theorem like_terms_sum (m n : ℕ) (a b : ℝ) 
  (h₁ : 5 * a^m * b^3 = 5 * a^m * b^3) 
  (h₂ : -4 * a^2 * b^(n-1) = -4 * a^2 * b^(n-1)) 
  (h₃ : m = 2) (h₄ : 3 = n - 1) : m + n = 6 := by
  sorry

end like_terms_sum_l2419_241927


namespace identify_value_of_expression_l2419_241957

theorem identify_value_of_expression (x y z : ℝ)
  (h1 : y / (x - y) = x / (y + z))
  (h2 : z^2 = x * (y + z) - y * (x - y)) :
  (y^2 + z^2 - x^2) / (2 * y * z) = 1 / 2 := 
sorry

end identify_value_of_expression_l2419_241957


namespace value_of_f_at_4_l2419_241902

noncomputable def f (x : ℝ) (c : ℝ) (d : ℝ) : ℝ :=
  c * x ^ 2 + d * x + 3

theorem value_of_f_at_4 :
  (∃ c d : ℝ, f 1 c d = 3 ∧ f 2 c d = 5) → f 4 1 (-1) = 15 :=
by
  sorry

end value_of_f_at_4_l2419_241902


namespace bread_consumption_snacks_per_day_l2419_241916

theorem bread_consumption_snacks_per_day (members : ℕ) (breakfast_slices_per_member : ℕ) (slices_per_loaf : ℕ) (loaves : ℕ) (days : ℕ) (total_slices_breakfast : ℕ) (total_slices_all : ℕ) (snack_slices_per_member_per_day : ℕ) :
  members = 4 →
  breakfast_slices_per_member = 3 →
  slices_per_loaf = 12 →
  loaves = 5 →
  days = 3 →
  total_slices_breakfast = members * breakfast_slices_per_member * days →
  total_slices_all = slices_per_loaf * loaves →
  snack_slices_per_member_per_day = ((total_slices_all - total_slices_breakfast) / members / days) →
  snack_slices_per_member_per_day = 2 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  -- We can insert the proof outline here based on the calculations from the solution steps
  sorry

end bread_consumption_snacks_per_day_l2419_241916


namespace fraction_meaningful_iff_nonzero_l2419_241940

theorem fraction_meaningful_iff_nonzero (x : ℝ) : (∃ y : ℝ, y = 1 / x) ↔ x ≠ 0 :=
by sorry

end fraction_meaningful_iff_nonzero_l2419_241940


namespace calc_fraction_l2419_241934

theorem calc_fraction :
  ((1 / 3 + 1 / 6) * (4 / 7) * (5 / 9) = 10 / 63) :=
by
  sorry

end calc_fraction_l2419_241934


namespace complement_intersection_in_U_l2419_241949

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4, 5}

theorem complement_intersection_in_U : (U \ (A ∩ B)) = {1, 4, 5, 6, 7, 8} :=
by {
  sorry
}

end complement_intersection_in_U_l2419_241949


namespace blue_flowers_percentage_l2419_241942

theorem blue_flowers_percentage :
  let total_flowers := 96
  let green_flowers := 9
  let red_flowers := 3 * green_flowers
  let yellow_flowers := 12
  let accounted_flowers := green_flowers + red_flowers + yellow_flowers
  let blue_flowers := total_flowers - accounted_flowers
  (blue_flowers / total_flowers : ℝ) * 100 = 50 :=
by
  sorry

end blue_flowers_percentage_l2419_241942


namespace choir_students_min_l2419_241971

/-- 
  Prove that the minimum number of students in the choir, where the number 
  of students must be a multiple of 9, 10, and 11, is 990. 
-/
theorem choir_students_min (n : ℕ) :
  (∃ n, n > 0 ∧ n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) ∧ (∀ m, m > 0 ∧ m % 9 = 0 ∧ m % 10 = 0 ∧ m % 11 = 0 → n ≤ m) → n = 990 :=
by
  sorry

end choir_students_min_l2419_241971


namespace average_weight_increase_l2419_241975

theorem average_weight_increase
 (num_persons : ℕ) (weight_increase : ℝ) (replacement_weight : ℝ) (new_weight : ℝ) (weight_difference : ℝ) (avg_weight_increase : ℝ)
 (cond1 : num_persons = 10)
 (cond2 : replacement_weight = 65)
 (cond3 : new_weight = 90)
 (cond4 : weight_difference = new_weight - replacement_weight)
 (cond5 : weight_difference = weight_increase)
 (cond6 : avg_weight_increase = weight_increase / num_persons) :
avg_weight_increase = 2.5 :=
by
  sorry

end average_weight_increase_l2419_241975


namespace coffee_expenses_l2419_241978

-- Define amounts consumed and unit costs for French and Columbian roast
def ounces_per_donut_M := 2
def ounces_per_donut_D := 3
def ounces_per_donut_S := ounces_per_donut_D
def ounces_per_pot_F := 12
def ounces_per_pot_C := 15
def cost_per_pot_F := 3
def cost_per_pot_C := 4

-- Define number of donuts consumed
def donuts_M := 8
def donuts_D := 12
def donuts_S := 16

-- Calculate total ounces needed
def total_ounces_F := donuts_M * ounces_per_donut_M
def total_ounces_C := (donuts_D + donuts_S) * ounces_per_donut_D

-- Calculate pots needed, rounding up since partial pots are not allowed
def pots_needed_F := Nat.ceil (total_ounces_F / ounces_per_pot_F)
def pots_needed_C := Nat.ceil (total_ounces_C / ounces_per_pot_C)

-- Calculate total cost
def total_cost := (pots_needed_F * cost_per_pot_F) + (pots_needed_C * cost_per_pot_C)

-- Theorem statement to assert the proof
theorem coffee_expenses : total_cost = 30 := by
  sorry

end coffee_expenses_l2419_241978


namespace zeros_not_adjacent_probability_l2419_241900

-- Definitions based on the conditions
def total_arrangements : ℕ := Nat.choose 6 2
def non_adjacent_zero_arrangements : ℕ := Nat.choose 5 2

-- The probability that the 2 zeros are not adjacent
def probability_non_adjacent_zero : ℚ :=
  (non_adjacent_zero_arrangements : ℚ) / (total_arrangements : ℚ)

-- The theorem statement
theorem zeros_not_adjacent_probability :
  probability_non_adjacent_zero = 2 / 3 :=
by
  -- The proof would go here
  sorry

end zeros_not_adjacent_probability_l2419_241900


namespace sum_pqrst_is_neg_15_over_2_l2419_241967

variable (p q r s t x : ℝ)
variable (h1 : p + 2 = x)
variable (h2 : q + 3 = x)
variable (h3 : r + 4 = x)
variable (h4 : s + 5 = x)
variable (h5 : t + 6 = x)
variable (h6 : p + q + r + s + t + 10 = x)

theorem sum_pqrst_is_neg_15_over_2 : p + q + r + s + t = -15 / 2 := by
  sorry

end sum_pqrst_is_neg_15_over_2_l2419_241967


namespace sum_of_squares_l2419_241941

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 110) : x^2 + y^2 = 1380 := 
by sorry

end sum_of_squares_l2419_241941


namespace percentage_of_the_stock_l2419_241980

noncomputable def faceValue : ℝ := 100
noncomputable def yield : ℝ := 0.10
noncomputable def quotedPrice : ℝ := 160

theorem percentage_of_the_stock : 
  (yield * faceValue / quotedPrice * 100 = 6.25) :=
by
  sorry

end percentage_of_the_stock_l2419_241980


namespace neg_exists_eq_forall_l2419_241966

theorem neg_exists_eq_forall (p : Prop) :
  (∀ x : ℝ, ¬(x^2 + 2*x = 3)) ↔ ¬(∃ x : ℝ, x^2 + 2*x = 3) := 
by
  sorry

end neg_exists_eq_forall_l2419_241966


namespace gcd_of_g_and_y_l2419_241918

-- Define the function g(y)
def g (y : ℕ) := (3 * y + 4) * (8 * y + 3) * (14 * y + 9) * (y + 14)

-- Define that y is a multiple of 45678
def isMultipleOf (y divisor : ℕ) : Prop := ∃ k, y = k * divisor

-- Define the proof problem
theorem gcd_of_g_and_y (y : ℕ) (h : isMultipleOf y 45678) : Nat.gcd (g y) y = 1512 :=
by
  sorry

end gcd_of_g_and_y_l2419_241918


namespace books_in_june_l2419_241943

-- Definitions
def Book_may : ℕ := 2
def Book_july : ℕ := 10
def Total_books : ℕ := 18

-- Theorem statement
theorem books_in_june : ∃ (Book_june : ℕ), Book_may + Book_june + Book_july = Total_books ∧ Book_june = 6 :=
by
  -- Proof will be here
  sorry

end books_in_june_l2419_241943


namespace sandy_spent_correct_amount_l2419_241994

-- Definitions
def shorts_price : ℝ := 13.99
def shirt_price : ℝ := 12.14
def jacket_price : ℝ := 7.43
def shoes_price : ℝ := 8.50
def accessories_price : ℝ := 10.75
def discount_rate : ℝ := 0.10
def coupon_amount : ℝ := 5.00
def tax_rate : ℝ := 0.075

-- Sum of all items before discounts and coupons
def total_before_discount : ℝ :=
  shorts_price + shirt_price + jacket_price + shoes_price + accessories_price

-- Total after applying the discount
def total_after_discount : ℝ :=
  total_before_discount * (1 - discount_rate)

-- Total after applying the coupon
def total_after_coupon : ℝ :=
  total_after_discount - coupon_amount

-- Total after applying the tax
def total_after_tax : ℝ :=
  total_after_coupon * (1 + tax_rate)

-- Theorem assertion that total amount spent is equal to $45.72
theorem sandy_spent_correct_amount : total_after_tax = 45.72 := by
  sorry

end sandy_spent_correct_amount_l2419_241994


namespace true_proposition_is_A_l2419_241989

-- Define the propositions
def l1 := ∀ (x y : ℝ), x - 2 * y + 3 = 0
def l2 := ∀ (x y : ℝ), 2 * x + y + 3 = 0
def p : Prop := ¬(l1 ∧ l2 ∧ ¬(∃ (x y : ℝ), x - 2 * y + 3 = 0 ∧ 2 * x + y + 3 = 0 ∧ (1 * 2 + (-2) * 1 ≠ 0)))
def q : Prop := ∃ x₀ : ℝ, (0 < x₀) ∧ (x₀ + 2 > Real.exp x₀)

-- The proof problem statement
theorem true_proposition_is_A : (¬p) ∧ q :=
by
  sorry

end true_proposition_is_A_l2419_241989


namespace graph_not_in_second_quadrant_l2419_241921

theorem graph_not_in_second_quadrant (b : ℝ) (h : ∀ x < 0, 2^x + b - 1 < 0) : b ≤ 0 :=
sorry

end graph_not_in_second_quadrant_l2419_241921


namespace required_pumps_l2419_241909

-- Define the conditions in Lean
variables (x a b n : ℝ)

-- Condition 1: x + 40a = 80b
def condition1 : Prop := x + 40 * a = 2 * 40 * b

-- Condition 2: x + 16a = 64b
def condition2 : Prop := x + 16 * a = 4 * 16 * b

-- Main theorem: Given the conditions, prove that n >= 6 satisfies the remaining requirement
theorem required_pumps (h1 : condition1 x a b) (h2 : condition2 x a b) : n >= 6 :=
by
  sorry

end required_pumps_l2419_241909


namespace select_books_from_corner_l2419_241956

def num_ways_to_select_books (n₁ n₂ k : ℕ) : ℕ :=
  if h₁ : k > n₁ ∧ k > n₂ then 0
  else if h₂ : k > n₂ then 1
  else if h₃ : k > n₁ then Nat.choose n₂ k
  else Nat.choose n₁ k + 2 * Nat.choose n₁ (k-1) * Nat.choose n₂ 1 + Nat.choose n₁ k * 0 +
    (Nat.choose n₂ 1 * Nat.choose n₂ (k-1)) + Nat.choose n₂ k * 1

theorem select_books_from_corner :
  num_ways_to_select_books 3 6 3 = 42 :=
by
  sorry

end select_books_from_corner_l2419_241956


namespace smallest_x_l2419_241973

theorem smallest_x (x y : ℕ) (h_pos: x > 0 ∧ y > 0) (h_eq: 8 / 10 = y / (186 + x)) : x = 4 :=
sorry

end smallest_x_l2419_241973


namespace problem_l2419_241914

theorem problem (x : ℕ) (h1 : x > 0) (h2 : ∃ k : ℕ, 7 - x = k^2) : x = 3 ∨ x = 6 ∨ x = 7 :=
by
  sorry

end problem_l2419_241914


namespace only_integer_solution_is_zero_l2419_241992

theorem only_integer_solution_is_zero (x y : ℤ) (h : x^4 + y^4 = 3 * x^3 * y) : x = 0 ∧ y = 0 :=
by {
  -- Here we would provide the proof steps.
  sorry
}

end only_integer_solution_is_zero_l2419_241992


namespace probability_of_at_least_2_girls_equals_specified_value_l2419_241923

def num_combinations (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

noncomputable def probability_at_least_2_girls : ℚ :=
  let total_committees := num_combinations 24 5
  let all_boys := num_combinations 14 5
  let one_girl_four_boys := num_combinations 10 1 * num_combinations 14 4
  let at_least_2_girls := total_committees - (all_boys + one_girl_four_boys)
  at_least_2_girls / total_committees

theorem probability_of_at_least_2_girls_equals_specified_value :
  probability_at_least_2_girls = 2541 / 3542 := 
sorry

end probability_of_at_least_2_girls_equals_specified_value_l2419_241923


namespace wives_identification_l2419_241930

theorem wives_identification (Anna Betty Carol Dorothy MrBrown MrGreen MrWhite MrSmith : ℕ):
  Anna = 2 ∧ Betty = 3 ∧ Carol = 4 ∧ Dorothy = 5 ∧
  (MrBrown = Dorothy ∧ MrGreen = 2 * Carol ∧ MrWhite = 3 * Betty ∧ MrSmith = 4 * Anna) ∧
  (Anna + Betty + Carol + Dorothy + MrBrown + MrGreen + MrWhite + MrSmith = 44) →
  (
    Dorothy = 5 ∧
    Carol = 4 ∧
    Betty = 3 ∧
    Anna = 2 ∧
    MrBrown = 5 ∧
    MrGreen = 8 ∧
    MrWhite = 9 ∧
    MrSmith = 8
  ) :=
by
  intros
  sorry

end wives_identification_l2419_241930


namespace ride_count_l2419_241920

noncomputable def initial_tickets : ℕ := 287
noncomputable def spent_on_games : ℕ := 134
noncomputable def earned_tickets : ℕ := 32
noncomputable def cost_per_ride : ℕ := 17

theorem ride_count (initial_tickets : ℕ) (spent_on_games : ℕ) (earned_tickets : ℕ) (cost_per_ride : ℕ) : 
  initial_tickets = 287 ∧ spent_on_games = 134 ∧ earned_tickets = 32 ∧ cost_per_ride = 17 → (initial_tickets - spent_on_games + earned_tickets) / cost_per_ride = 10 :=
by
  intros
  sorry

end ride_count_l2419_241920


namespace compare_neg_fractions_l2419_241907

theorem compare_neg_fractions : (-5/4 : ℚ) > (-4/3 : ℚ) := 
sorry

end compare_neg_fractions_l2419_241907


namespace difference_students_pets_in_all_classrooms_l2419_241970

-- Definitions of the conditions
def students_per_classroom : ℕ := 24
def rabbits_per_classroom : ℕ := 3
def guinea_pigs_per_classroom : ℕ := 2
def number_of_classrooms : ℕ := 5

-- Proof problem statement
theorem difference_students_pets_in_all_classrooms :
  (students_per_classroom * number_of_classrooms) - 
  ((rabbits_per_classroom + guinea_pigs_per_classroom) * number_of_classrooms) = 95 := by
  sorry

end difference_students_pets_in_all_classrooms_l2419_241970


namespace height_difference_after_3_years_l2419_241976

/-- Conditions for the tree's and boy's growth rates per season. --/
def tree_spring_growth : ℕ := 4
def tree_summer_growth : ℕ := 6
def tree_fall_growth : ℕ := 2
def tree_winter_growth : ℕ := 1

def boy_spring_growth : ℕ := 2
def boy_summer_growth : ℕ := 2
def boy_fall_growth : ℕ := 0
def boy_winter_growth : ℕ := 0

/-- Initial heights. --/
def initial_tree_height : ℕ := 16
def initial_boy_height : ℕ := 24

/-- Length of each season in months. --/
def season_length : ℕ := 3

/-- Time period in years. --/
def years : ℕ := 3

/-- Prove the height difference between the tree and the boy after 3 years is 73 inches. --/
theorem height_difference_after_3_years :
    let tree_annual_growth := tree_spring_growth * season_length +
                             tree_summer_growth * season_length +
                             tree_fall_growth * season_length +
                             tree_winter_growth * season_length
    let tree_final_height := initial_tree_height + tree_annual_growth * years
    let boy_annual_growth := boy_spring_growth * season_length +
                            boy_summer_growth * season_length +
                            boy_fall_growth * season_length +
                            boy_winter_growth * season_length
    let boy_final_height := initial_boy_height + boy_annual_growth * years
    tree_final_height - boy_final_height = 73 :=
by sorry

end height_difference_after_3_years_l2419_241976


namespace regular_polygon_sides_l2419_241990

theorem regular_polygon_sides (C : ℕ) (h : (C - 2) * 180 / C = 144) : C = 10 := 
sorry

end regular_polygon_sides_l2419_241990


namespace max_value_correct_l2419_241950

noncomputable def max_value_ineq (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 3 * x + 2 * y + 6 * z = 1) : Prop :=
  x ^ 4 * y ^ 3 * z ^ 2 ≤ 1 / 372008

theorem max_value_correct (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 3 * x + 2 * y + 6 * z = 1) :
  max_value_ineq x y z h1 h2 h3 h4 :=
sorry

end max_value_correct_l2419_241950


namespace fault_line_movement_year_before_l2419_241924

-- Define the total movement over two years
def total_movement : ℝ := 6.5

-- Define the movement during the past year
def past_year_movement : ℝ := 1.25

-- Define the movement the year before
def year_before_movement : ℝ := total_movement - past_year_movement

-- Prove that the fault line moved 5.25 inches the year before
theorem fault_line_movement_year_before : year_before_movement = 5.25 :=
  by  sorry

end fault_line_movement_year_before_l2419_241924


namespace gray_region_area_l2419_241922

noncomputable def area_gray_region : ℝ :=
  let area_rectangle := (12 - 4) * (12 - 4)
  let radius_c := 4
  let radius_d := 4
  let area_quarter_circle_c := 1/4 * Real.pi * radius_c^2
  let area_quarter_circle_d := 1/4 * Real.pi * radius_d^2
  let overlap_area := area_quarter_circle_c + area_quarter_circle_d
  area_rectangle - overlap_area

theorem gray_region_area :
  area_gray_region = 64 - 8 * Real.pi := by
  sorry

end gray_region_area_l2419_241922


namespace parabola_focus_correct_l2419_241979

-- defining the equation of the parabola as a condition
def parabola (y x : ℝ) : Prop := y^2 = 4 * x

-- defining the focus of the parabola
def focus (x y : ℝ) : Prop := (x, y) = (1, 0)

-- the main theorem statement
theorem parabola_focus_correct (y x : ℝ) (h : parabola y x) : focus 1 0 :=
by
  -- proof steps would go here
  sorry

end parabola_focus_correct_l2419_241979


namespace red_box_position_l2419_241963

theorem red_box_position (n : ℕ) (pos_smallest_to_largest : ℕ) (pos_largest_to_smallest : ℕ) 
  (h1 : n = 45) 
  (h2 : pos_smallest_to_largest = 29) 
  (h3 : pos_largest_to_smallest = n - (pos_smallest_to_largest - 1)) :
  pos_largest_to_smallest = 17 := 
  by
    -- This proof is missing; implementation goes here
    sorry

end red_box_position_l2419_241963


namespace at_least_one_misses_l2419_241965

-- Definitions for the given conditions
variables {p q : Prop}

-- Lean 4 statement proving the equivalence
theorem at_least_one_misses (hp : p → false) (hq : q → false) : (¬p ∨ ¬q) :=
by sorry

end at_least_one_misses_l2419_241965


namespace percent_decrease_is_20_l2419_241998

/-- Define the original price and sale price as constants. -/
def P_original : ℕ := 100
def P_sale : ℕ := 80

/-- Define the formula for percent decrease. -/
def percent_decrease (P_original P_sale : ℕ) : ℕ :=
  ((P_original - P_sale) * 100) / P_original

/-- Prove that the percent decrease is 20%. -/
theorem percent_decrease_is_20 : percent_decrease P_original P_sale = 20 :=
by
  sorry

end percent_decrease_is_20_l2419_241998


namespace passed_percentage_l2419_241938

theorem passed_percentage (A B C AB BC AC ABC: ℝ) 
  (hA : A = 0.25) 
  (hB : B = 0.50) 
  (hC : C = 0.30) 
  (hAB : AB = 0.25) 
  (hBC : BC = 0.15) 
  (hAC : AC = 0.10) 
  (hABC : ABC = 0.05) 
  : 100 - (A + B + C - AB - BC - AC + ABC) = 40 := 
by 
  rw [hA, hB, hC, hAB, hBC, hAC, hABC]
  norm_num
  sorry

end passed_percentage_l2419_241938


namespace simplify_expression_l2419_241959

theorem simplify_expression :
  (3 * (Real.sqrt 3 + Real.sqrt 5)) / (4 * Real.sqrt (3 + Real.sqrt 4))
  = (3 * Real.sqrt 15 + 3 * Real.sqrt 5) / 20 :=
by
  sorry

end simplify_expression_l2419_241959


namespace mila_father_total_pay_l2419_241958

def first_job_pay : ℤ := 2125
def pay_difference : ℤ := 375
def second_job_pay : ℤ := first_job_pay - pay_difference
def total_pay : ℤ := first_job_pay + second_job_pay

theorem mila_father_total_pay :
  total_pay = 3875 := by
  sorry

end mila_father_total_pay_l2419_241958


namespace range_of_a_l2419_241946

def p (a : ℝ) : Prop := a ≤ -4 ∨ a ≥ 4
def q (a : ℝ) : Prop := a ≥ -12
def either_p_or_q_but_not_both (a : ℝ) : Prop := (p a ∧ ¬ q a) ∨ (¬ p a ∧ q a)

theorem range_of_a :
  {a : ℝ | either_p_or_q_but_not_both a} = {a : ℝ | (-4 < a ∧ a < 4) ∨ a < -12} :=
sorry

end range_of_a_l2419_241946


namespace derivative_at_2_l2419_241926

def f (x : ℝ) : ℝ := x^3 + 2

theorem derivative_at_2 : deriv f 2 = 12 := by
  sorry

end derivative_at_2_l2419_241926


namespace distance_between_foci_of_hyperbola_is_correct_l2419_241996

noncomputable def distance_between_foci_of_hyperbola : ℝ := 
  let a_sq := 50
  let b_sq := 8
  let c_sq := a_sq + b_sq
  let c := Real.sqrt c_sq
  2 * c

theorem distance_between_foci_of_hyperbola_is_correct :
  distance_between_foci_of_hyperbola = 2 * Real.sqrt 58 :=
by
  sorry

end distance_between_foci_of_hyperbola_is_correct_l2419_241996


namespace prices_proof_sales_revenue_proof_l2419_241953

-- Definitions for the prices and quantities
def price_peanut_oil := 50
def price_corn_oil := 40

-- Conditions from the problem
def condition1 (x y : ℕ) : Prop := 20 * x + 30 * y = 2200
def condition2 (x y : ℕ) : Prop := 30 * x + 10 * y = 1900
def purchased_peanut_oil := 50
def selling_price_peanut_oil := 60

-- Proof statement for Part 1
theorem prices_proof : ∃ (x y : ℕ), condition1 x y ∧ condition2 x y ∧ x = price_peanut_oil ∧ y = price_corn_oil :=
sorry

-- Proof statement for Part 2
theorem sales_revenue_proof : ∃ (m : ℕ), (selling_price_peanut_oil * m > price_peanut_oil * purchased_peanut_oil) ∧ m = 42 :=
sorry

end prices_proof_sales_revenue_proof_l2419_241953


namespace sequence_a_1000_l2419_241917

theorem sequence_a_1000 (a : ℕ → ℕ)
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 3) 
  (h₃ : ∀ n, a (n + 1) = 3 * a n - 2 * a (n - 1)) : 
  a 1000 = 2^1000 - 1 := 
sorry

end sequence_a_1000_l2419_241917


namespace fantasia_max_capacity_reach_l2419_241982

def acre_per_person := 1
def land_acres := 40000
def base_population := 500
def population_growth_factor := 4
def years_per_growth_period := 20

def maximum_capacity := land_acres / acre_per_person

def population_at_time (years_from_2000 : ℕ) : ℕ :=
  base_population * population_growth_factor^(years_from_2000 / years_per_growth_period)

theorem fantasia_max_capacity_reach :
  ∃ t : ℕ, t = 60 ∧ population_at_time t = maximum_capacity := by sorry

end fantasia_max_capacity_reach_l2419_241982


namespace ratio_of_x_to_y_l2419_241925

variable (x y : ℝ)

theorem ratio_of_x_to_y (h : 3 * x = 0.12 * 250 * y) : x / y = 10 :=
sorry

end ratio_of_x_to_y_l2419_241925


namespace sandy_books_l2419_241933

theorem sandy_books (x : ℕ)
  (h1 : 1080 + 840 = 1920)
  (h2 : 16 = 1920 / (x + 55)) :
  x = 65 :=
by
  -- Theorem proof placeholder
  sorry

end sandy_books_l2419_241933


namespace firing_sequence_hits_submarine_l2419_241937

theorem firing_sequence_hits_submarine (a b : ℕ) (hb : b > 0) : ∃ n : ℕ, (∃ (an bn : ℕ), (an + bn * n) = a + n * b) :=
sorry

end firing_sequence_hits_submarine_l2419_241937


namespace minimum_a3_b3_no_exist_a_b_2a_3b_eq_6_l2419_241948

-- Define the conditions once to reuse them for both proof statements.
variables {a b : ℝ} (ha: a > 0) (hb: b > 0) (h: (1/a) + (1/b) = Real.sqrt (a * b))

-- Problem (I)
theorem minimum_a3_b3 (h : (1/a) + (1/b) = Real.sqrt (a * b)) (ha: a > 0) (hb: b > 0) :
  a^3 + b^3 = 4 * Real.sqrt 2 := 
sorry

-- Problem (II)
theorem no_exist_a_b_2a_3b_eq_6 (h : (1/a) + (1/b) = Real.sqrt (a * b)) (ha: a > 0) (hb: b > 0) :
  ¬ ∃ (a b : ℝ), 2 * a + 3 * b = 6 :=
sorry

end minimum_a3_b3_no_exist_a_b_2a_3b_eq_6_l2419_241948


namespace minimum_m_minus_n_l2419_241955

theorem minimum_m_minus_n (m n : ℕ) (hm : m > n) (h : (9^m) % 100 = (9^n) % 100) : m - n = 10 := 
sorry

end minimum_m_minus_n_l2419_241955


namespace number_to_match_l2419_241939

def twenty_five_percent_less (x: ℕ) : ℕ := 3 * x / 4

def one_third_more (n: ℕ) : ℕ := 4 * n / 3

theorem number_to_match (n : ℕ) (x : ℕ) 
  (h1 : x = 80) 
  (h2 : one_third_more n = twenty_five_percent_less x) : n = 45 :=
by
  -- Proof is skipped as per the instruction
  sorry

end number_to_match_l2419_241939


namespace avg_length_remaining_wires_l2419_241964

theorem avg_length_remaining_wires (N : ℕ) (avg_length : ℕ) 
    (third_wires_count : ℕ) (third_wires_avg_length : ℕ) 
    (total_length : ℕ := N * avg_length) 
    (third_wires_total_length : ℕ := third_wires_count * third_wires_avg_length) 
    (remaining_wires_count : ℕ := N - third_wires_count) 
    (remaining_wires_total_length : ℕ := total_length - third_wires_total_length) :
    N = 6 → 
    avg_length = 80 → 
    third_wires_count = 2 → 
    third_wires_avg_length = 70 → 
    remaining_wires_count = 4 → 
    remaining_wires_total_length / remaining_wires_count = 85 :=
by 
  intros hN hAvg hThirdCount hThirdAvg hRemainingCount
  sorry

end avg_length_remaining_wires_l2419_241964


namespace difference_of_fractions_l2419_241969

theorem difference_of_fractions (a b c : ℝ) (h1 : a = 8000 * (1/2000)) (h2 : b = 8000 * (1/10)) (h3 : c = b - a) : c = 796 := 
sorry

end difference_of_fractions_l2419_241969


namespace identify_infected_person_in_4_tests_l2419_241932

theorem identify_infected_person_in_4_tests :
  (∀ (group : Fin 16 → Bool), ∃ infected : Fin 16, group infected = ff) →
  ∃ (tests_needed : ℕ), tests_needed = 4 :=
by sorry

end identify_infected_person_in_4_tests_l2419_241932


namespace gcd_lcm_condition_implies_divisibility_l2419_241995

theorem gcd_lcm_condition_implies_divisibility
  (a b : ℤ) (h : Int.gcd a b + Int.lcm a b = a + b) : a ∣ b ∨ b ∣ a := 
sorry

end gcd_lcm_condition_implies_divisibility_l2419_241995


namespace geometric_sequence_tenth_term_l2419_241906

theorem geometric_sequence_tenth_term :
  let a := 4
  let r := (12 / 3) / 4
  let nth_term (n : ℕ) := a * r^(n-1)
  nth_term 10 = 4 :=
  by sorry

end geometric_sequence_tenth_term_l2419_241906


namespace initial_dogs_l2419_241915

theorem initial_dogs (D : ℕ) (h : D + 5 + 3 = 10) : D = 2 :=
by sorry

end initial_dogs_l2419_241915


namespace power_func_passes_point_l2419_241981

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem power_func_passes_point (f : ℝ → ℝ) (h : ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α) 
  (h_point : f 9 = 1 / 3) : f 25 = 1 / 5 :=
sorry

end power_func_passes_point_l2419_241981


namespace studios_total_l2419_241960

section

variable (s1 s2 s3 : ℕ)

theorem studios_total (h1 : s1 = 110) (h2 : s2 = 135) (h3 : s3 = 131) : s1 + s2 + s3 = 376 :=
by
  sorry

end

end studios_total_l2419_241960


namespace number_of_SUVs_washed_l2419_241945

theorem number_of_SUVs_washed (charge_car charge_truck charge_SUV total_raised : ℕ) (num_trucks num_cars S : ℕ) :
  charge_car = 5 →
  charge_truck = 6 →
  charge_SUV = 7 →
  total_raised = 100 →
  num_trucks = 5 →
  num_cars = 7 →
  total_raised = num_cars * charge_car + num_trucks * charge_truck + S * charge_SUV →
  S = 5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end number_of_SUVs_washed_l2419_241945


namespace min_pieces_per_orange_l2419_241961

theorem min_pieces_per_orange (oranges : ℕ) (calories_per_orange : ℕ) (people : ℕ) (calories_per_person : ℕ) (pieces_per_orange : ℕ) :
  oranges = 5 →
  calories_per_orange = 80 →
  people = 4 →
  calories_per_person = 100 →
  pieces_per_orange ≥ 4 :=
by
  intro h_oranges h_calories_per_orange h_people h_calories_per_person
  sorry

end min_pieces_per_orange_l2419_241961


namespace polynomial_identity_l2419_241985

variable (x y : ℝ)

theorem polynomial_identity :
    (x + y^2) * (x - y^2) * (x^2 + y^4) = x^4 - y^8 :=
sorry

end polynomial_identity_l2419_241985


namespace problem_sum_value_l2419_241936

def letter_value_pattern : List Int := [2, 3, 2, 1, 0, -1, -2, -3, -2, -1]

def char_value (c : Char) : Int :=
  let pos := c.toNat - 'a'.toNat + 1
  letter_value_pattern.get! ((pos - 1) % 10)

def word_value (w : String) : Int :=
  w.data.map char_value |>.sum

theorem problem_sum_value : word_value "problem" = 5 :=
  by sorry

end problem_sum_value_l2419_241936


namespace toot_has_vertical_symmetry_l2419_241952

def has_vertical_symmetry (letter : Char) : Prop :=
  letter = 'T' ∨ letter = 'O'

def word_has_vertical_symmetry (word : List Char) : Prop :=
  ∀ letter ∈ word, has_vertical_symmetry letter

theorem toot_has_vertical_symmetry : word_has_vertical_symmetry ['T', 'O', 'O', 'T'] :=
  by
    sorry

end toot_has_vertical_symmetry_l2419_241952


namespace total_lateness_l2419_241931

/-
  Conditions:
  Charlize was 20 minutes late.
  Ana was 5 minutes later than Charlize.
  Ben was 15 minutes less late than Charlize.
  Clara was twice as late as Charlize.
  Daniel was 10 minutes earlier than Clara.

  Total time for which all five students were late is 120 minutes.
-/

def charlize := 20
def ana := charlize + 5
def ben := charlize - 15
def clara := charlize * 2
def daniel := clara - 10

def total_time := charlize + ana + ben + clara + daniel

theorem total_lateness : total_time = 120 :=
by
  sorry

end total_lateness_l2419_241931


namespace right_triangle_properties_l2419_241905

theorem right_triangle_properties (a b c : ℝ) (h1 : c = 13) (h2 : a = 5)
  (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 30 ∧ a + b + c = 30 := by
  sorry

end right_triangle_properties_l2419_241905


namespace problem_part1_problem_part2_l2419_241935

-- Problem statements

theorem problem_part1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 2) : 
  (a + b) * (a^5 + b^5) ≥ 4 := 
sorry

theorem problem_part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 2) : 
  a + b ≤ 2 := 
sorry

end problem_part1_problem_part2_l2419_241935


namespace twenty_seven_divides_sum_l2419_241968

theorem twenty_seven_divides_sum (x y z : ℤ) (h : (x - y) * (y - z) * (z - x) = x + y + z) : 27 ∣ x + y + z := sorry

end twenty_seven_divides_sum_l2419_241968


namespace sum_groups_eq_250_l2419_241904

-- Definitions for each sum
def sum1 : ℕ := 3 + 13 + 23 + 33 + 43
def sum2 : ℕ := 7 + 17 + 27 + 37 + 47

-- Theorem statement that the sum of these groups is 250
theorem sum_groups_eq_250 : sum1 + sum2 = 250 :=
by sorry

end sum_groups_eq_250_l2419_241904


namespace polar_coordinates_of_point_l2419_241910

theorem polar_coordinates_of_point :
  let x := 2
  let y := 2 * Real.sqrt 3
  let r := Real.sqrt (x^2 + y^2)
  let theta := Real.arctan (y / x)
  r = 4 ∧ theta = Real.pi / 3 :=
by
  let x := 2
  let y := 2 * Real.sqrt 3
  let r := Real.sqrt (x^2 + y^2)
  let theta := Real.arctan (y / x)
  have h_r : r = 4 := by {
    -- Calculation for r
    sorry
  }
  have h_theta : theta = Real.pi / 3 := by {
    -- Calculation for theta
    sorry
  }
  exact ⟨h_r, h_theta⟩

end polar_coordinates_of_point_l2419_241910


namespace midpoint_lattice_point_exists_l2419_241954

theorem midpoint_lattice_point_exists (S : Finset (ℤ × ℤ)) (hS : S.card = 5) :
  ∃ (p1 p2 : ℤ × ℤ), p1 ∈ S ∧ p2 ∈ S ∧ p1 ≠ p2 ∧
  (∃ (x_mid y_mid : ℤ), 
    (p1.1 + p2.1) = 2 * x_mid ∧
    (p1.2 + p2.2) = 2 * y_mid) :=
by
  sorry

end midpoint_lattice_point_exists_l2419_241954


namespace cost_per_dvd_l2419_241997

theorem cost_per_dvd (total_cost : ℝ) (num_dvds : ℕ) (cost_per_dvd : ℝ) :
  total_cost = 4.80 ∧ num_dvds = 4 → cost_per_dvd = 1.20 :=
by
  intro h
  sorry

end cost_per_dvd_l2419_241997


namespace closest_point_on_ellipse_to_line_l2419_241972

theorem closest_point_on_ellipse_to_line :
  ∃ (x y : ℝ), 
    7 * x^2 + 4 * y^2 = 28 ∧ 3 * x - 2 * y - 16 = 0 ∧ (x, y) = (3 / 2, -7 / 4) :=
by
  sorry

end closest_point_on_ellipse_to_line_l2419_241972


namespace boys_camp_percentage_l2419_241903

theorem boys_camp_percentage (x : ℕ) (total_boys : ℕ) (percent_science : ℕ) (not_science_boys : ℕ) 
    (percent_not_science : ℕ) (h1 : not_science_boys = percent_not_science * (x / 100) * total_boys) 
    (h2 : percent_not_science = 100 - percent_science) (h3 : percent_science = 30) 
    (h4 : not_science_boys = 21) (h5 : total_boys = 150) : x = 20 :=
by 
  sorry

end boys_camp_percentage_l2419_241903


namespace driving_distance_l2419_241901

theorem driving_distance:
  ∀ a b: ℕ, (a + b = 500 ∧ a ≥ 150 ∧ b ≥ 150) → 
  (⌊Real.sqrt (a^2 + b^2)⌋ = 380) :=
by
  intro a b
  intro h
  sorry

end driving_distance_l2419_241901
