import Mathlib

namespace units_digit_17_pow_2023_l2073_207326

theorem units_digit_17_pow_2023 
  (cycle : ℕ → ℕ)
  (h1 : cycle 0 = 7)
  (h2 : cycle 1 = 9)
  (h3 : cycle 2 = 3)
  (h4 : cycle 3 = 1)
  (units_digit : ℕ → ℕ)
  (h_units : ∀ n, units_digit (17^n) = units_digit (7^n))
  (h_units_cycle : ∀ n, units_digit (7^n) = cycle (n % 4)) :
  units_digit (17^2023) = 3 :=
by
  sorry

end units_digit_17_pow_2023_l2073_207326


namespace polygon_six_sides_l2073_207373

theorem polygon_six_sides (n : ℕ) (h1 : (n - 2) * 180 = 2 * 360) : n = 6 := by
  sorry

end polygon_six_sides_l2073_207373


namespace exponentiation_identity_l2073_207333

variable {a : ℝ}

theorem exponentiation_identity : (-a) ^ 2 * a ^ 3 = a ^ 5 := sorry

end exponentiation_identity_l2073_207333


namespace remainder_of_product_modulo_12_l2073_207346

theorem remainder_of_product_modulo_12 : (1625 * 1627 * 1629) % 12 = 3 := by
  sorry

end remainder_of_product_modulo_12_l2073_207346


namespace problem_a_problem_b_problem_c_problem_d_problem_e_l2073_207364

section problem_a
  -- Conditions
  def rainbow_russian_first_letters_sequence := ["к", "о", "ж", "з", "г", "с", "ф"]
  
  -- Theorem (question == answer)
  theorem problem_a : rainbow_russian_first_letters_sequence[4] = "г" ∧
                      rainbow_russian_first_letters_sequence[5] = "с" ∧
                      rainbow_russian_first_letters_sequence[6] = "ф" :=
  by
    -- Skip proof: sorry
    sorry
end problem_a

section problem_b
  -- Conditions
  def russian_alphabet_alternating_sequence := ["а", "в", "г", "ё", "ж", "з", "л", "м", "н", "о", "п", "т", "у"]
 
  -- Theorem (question == answer)
  theorem problem_b : russian_alphabet_alternating_sequence[10] = "п" ∧
                      russian_alphabet_alternating_sequence[11] = "т" ∧
                      russian_alphabet_alternating_sequence[12] = "у" :=
  by
    -- Skip proof: sorry
    sorry
end problem_b

section problem_c
  -- Conditions
  def russian_number_of_letters_sequence := ["один", "четыре", "шесть", "пять", "семь", "восемь"]
  
  -- Theorem (question == answer)
  theorem problem_c : russian_number_of_letters_sequence[4] = "семь" ∧
                      russian_number_of_letters_sequence[5] = "восемь" :=
  by
    -- Skip proof: sorry
    sorry
end problem_c

section problem_d
  -- Conditions
  def approximate_symmetry_letters_sequence := ["Ф", "Х", "Ш", "В"]

  -- Theorem (question == answer)
  theorem problem_d : approximate_symmetry_letters_sequence[3] = "В" :=
  by
    -- Skip proof: sorry
    sorry
end problem_d

section problem_e
  -- Conditions
  def russian_loops_in_digit_sequence := ["0", "д", "т", "ч", "п", "ш", "с", "в", "д"]

  -- Theorem (question == answer)
  theorem problem_e : russian_loops_in_digit_sequence[7] = "в" ∧
                      russian_loops_in_digit_sequence[8] = "д" :=
  by
    -- Skip proof: sorry
    sorry
end problem_e

end problem_a_problem_b_problem_c_problem_d_problem_e_l2073_207364


namespace total_hours_driven_l2073_207358

def total_distance : ℝ := 55.0
def distance_in_one_hour : ℝ := 1.527777778

theorem total_hours_driven : (total_distance / distance_in_one_hour) = 36.00 :=
by
  sorry

end total_hours_driven_l2073_207358


namespace Arrow_velocity_at_impact_l2073_207340

def Edward_initial_distance := 1875 -- \(\text{ft}\)
def Edward_initial_velocity := 0 -- \(\text{ft/s}\)
def Edward_acceleration := 1 -- \(\text{ft/s}^2\)
def Arrow_initial_distance := 0 -- \(\text{ft}\)
def Arrow_initial_velocity := 100 -- \(\text{ft/s}\)
def Arrow_deceleration := -1 -- \(\text{ft/s}^2\)
def time_impact := 25 -- \(\text{s}\)

theorem Arrow_velocity_at_impact : 
  (Arrow_initial_velocity + Arrow_deceleration * time_impact) = 75 := 
by
  sorry

end Arrow_velocity_at_impact_l2073_207340


namespace quadratic_real_roots_iff_l2073_207331

theorem quadratic_real_roots_iff (k : ℝ) : 
  (∃ x : ℝ, (k-1) * x^2 + 3 * x - 1 = 0) ↔ k ≥ -5 / 4 ∧ k ≠ 1 := sorry

end quadratic_real_roots_iff_l2073_207331


namespace triangle_area_not_twice_parallelogram_l2073_207311

theorem triangle_area_not_twice_parallelogram (b h : ℝ) :
  (1 / 2) * b * h ≠ 2 * b * h :=
sorry

end triangle_area_not_twice_parallelogram_l2073_207311


namespace inequality_proof_equality_condition_l2073_207323

theorem inequality_proof (a : ℝ) : (a^2 + 5)^2 + 4 * a * (10 - a) ≥ 8 * a^3  :=
by sorry

theorem equality_condition (a : ℝ) : ((a^2 + 5)^2 + 4 * a * (10 - a) = 8 * a^3) ↔ (a = 5 ∨ a = -1) :=
by sorry

end inequality_proof_equality_condition_l2073_207323


namespace rhombus_diagonal_length_l2073_207316

theorem rhombus_diagonal_length (d1 d2 : ℝ) (area : ℝ) (h1 : area = 600) (h2 : d1 = 30) :
  d2 = 40 :=
by
  sorry

end rhombus_diagonal_length_l2073_207316


namespace shopkeeper_profit_percentage_goal_l2073_207372

-- Definitions for CP, MP and discount percentage
variable (CP : ℝ)
noncomputable def MP : ℝ := CP * 1.32
noncomputable def discount_percentage : ℝ := 0.18939393939393938
noncomputable def SP : ℝ := MP CP - (discount_percentage * MP CP)
noncomputable def profit : ℝ := SP CP - CP
noncomputable def profit_percentage : ℝ := (profit CP / CP) * 100

-- Theorem stating that the profit percentage is approximately 7%
theorem shopkeeper_profit_percentage_goal :
  abs (profit_percentage CP - 7) < 0.01 := sorry

end shopkeeper_profit_percentage_goal_l2073_207372


namespace intersection_of_sets_l2073_207343

/-- Given the definitions of sets A and B, prove that A ∩ B equals {1, 2}. -/
theorem intersection_of_sets :
  let A := {x : ℝ | 0 < x}
  let B := {-2, -1, 1, 2}
  A ∩ B = {1, 2} :=
sorry

end intersection_of_sets_l2073_207343


namespace nine_a_eq_frac_minus_eighty_one_over_eleven_l2073_207336

theorem nine_a_eq_frac_minus_eighty_one_over_eleven (a b : ℚ) 
  (h1 : 8 * a + 3 * b = 0) 
  (h2 : a = b - 3) : 
  9 * a = -81 / 11 := 
sorry

end nine_a_eq_frac_minus_eighty_one_over_eleven_l2073_207336


namespace symmetric_circle_eq_l2073_207342

/-- The definition of the original circle equation. -/
def original_circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

/-- The definition of the line of symmetry equation. -/
def line_eq (x y : ℝ) : Prop := x - y - 2 = 0

/-- The statement that the equation of the circle that is symmetric to the original circle 
    about the given line is (x - 4)^2 + (y + 1)^2 = 1. -/
theorem symmetric_circle_eq : 
  (∃ x y : ℝ, original_circle_eq x y ∧ line_eq x y) →
  (∀ x y : ℝ, (x - 4)^2 + (y + 1)^2 = 1) :=
by sorry

end symmetric_circle_eq_l2073_207342


namespace factorization_a4_plus_4_l2073_207302

theorem factorization_a4_plus_4 (a : ℝ) : a^4 + 4 = (a^2 - 2*a + 2) * (a^2 + 2*a + 2) :=
by sorry

end factorization_a4_plus_4_l2073_207302


namespace eval_expression_l2073_207351

theorem eval_expression : 4^3 - 2 * 4^2 + 2 * 4 - 1 = 39 :=
by 
  -- Here we would write the proof, but according to the instructions we skip it with sorry.
  sorry

end eval_expression_l2073_207351


namespace cubes_and_quartics_sum_l2073_207322

theorem cubes_and_quartics_sum (a b : ℝ) (h1 : a + b = 2) (h2 : a^2 + b^2 = 2) : 
  a^3 + b^3 = 2 ∧ a^4 + b^4 = 2 :=
by 
  sorry

end cubes_and_quartics_sum_l2073_207322


namespace min_value_of_expression_l2073_207338

open Real

theorem min_value_of_expression (x y z : ℝ) (h₁ : x + y + z = 1) (h₂ : x > 0) (h₃ : y > 0) (h₄ : z > 0) :
  (∃ a, (∀ x y z, a ≤ (1 / (x + y) + (x + y) / z)) ∧ a = 3) :=
by
  sorry

end min_value_of_expression_l2073_207338


namespace avg_new_students_l2073_207394

-- Definitions for conditions
def orig_strength : ℕ := 17
def orig_avg_age : ℕ := 40
def new_students_count : ℕ := 17
def decreased_avg_age : ℕ := 36 -- given that average decreases by 4 years, i.e., 40 - 4

-- Definition for the original total age
def total_age_orig : ℕ := orig_strength * orig_avg_age

-- Definition for the total number of students after new students join
def total_students : ℕ := orig_strength + new_students_count

-- Definition for the total age after new students join
def total_age_new : ℕ := total_students * decreased_avg_age

-- Definition for the total age of new students
def total_age_new_students : ℕ := total_age_new - total_age_orig

-- Definition for the average age of new students
def avg_age_new_students : ℕ := total_age_new_students / new_students_count

-- Lean theorem stating the proof problem
theorem avg_new_students : 
  avg_age_new_students = 32 := 
by sorry

end avg_new_students_l2073_207394


namespace total_spectators_l2073_207345

-- Definitions of conditions
def num_men : Nat := 7000
def num_children : Nat := 2500
def num_women := num_children / 5

-- Theorem stating the total number of spectators
theorem total_spectators : (num_men + num_children + num_women) = 10000 := by
  sorry

end total_spectators_l2073_207345


namespace max_weight_of_crates_on_trip_l2073_207301

def max_crates : ℕ := 5
def min_crate_weight : ℕ := 150

theorem max_weight_of_crates_on_trip : max_crates * min_crate_weight = 750 := by
  sorry

end max_weight_of_crates_on_trip_l2073_207301


namespace increasing_interval_f_l2073_207393

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - (Real.pi / 6))

theorem increasing_interval_f : ∃ a b : ℝ, a < b ∧ 
  (∀ x y : ℝ, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ x < y → f x < f y) ∧
  (a = - (Real.pi / 6)) ∧ (b = (Real.pi / 3)) :=
by
  sorry

end increasing_interval_f_l2073_207393


namespace total_girls_in_circle_l2073_207386

theorem total_girls_in_circle (girls : Nat) 
  (h1 : (4 + 7) = girls + 2) : girls = 11 := 
by
  sorry

end total_girls_in_circle_l2073_207386


namespace minimum_cars_with_racing_stripes_l2073_207352

-- Definitions and conditions
variable (numberOfCars : ℕ) (withoutAC : ℕ) (maxWithACWithoutStripes : ℕ)

axiom total_number_of_cars : numberOfCars = 100
axiom cars_without_ac : withoutAC = 49
axiom max_ac_without_stripes : maxWithACWithoutStripes = 49    

-- Proposition
theorem minimum_cars_with_racing_stripes 
  (total_number_of_cars : numberOfCars = 100) 
  (cars_without_ac : withoutAC = 49)
  (max_ac_without_stripes : maxWithACWithoutStripes = 49) :
  ∃ (R : ℕ), R = 2 :=
by
  sorry

end minimum_cars_with_racing_stripes_l2073_207352


namespace range_of_a_l2073_207349

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 < x → (a / x - 4 / x^2 < 1)) → a < 4 := 
by
  sorry

end range_of_a_l2073_207349


namespace first_discount_percentage_l2073_207319

/-
  Prove that under the given conditions:
  1. The price before the first discount is $33.78.
  2. The final price after the first and second discounts is $19.
  3. The second discount is 25%.
-/
theorem first_discount_percentage (x : ℝ) :
  (33.78 * (1 - x / 100) * (1 - 25 / 100) = 19) →
  x = 25 :=
by
  -- Proof steps (to be filled)
  sorry

end first_discount_percentage_l2073_207319


namespace intersection_of_A_and_B_l2073_207375

def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | 2 < x ∧ x < 5}

theorem intersection_of_A_and_B :
  A ∩ B = {3, 4} := 
by 
  sorry

end intersection_of_A_and_B_l2073_207375


namespace tissues_used_l2073_207332

-- Define the conditions
def box_tissues : ℕ := 160
def boxes_bought : ℕ := 3
def tissues_left : ℕ := 270

-- Define the theorem that needs to be proven
theorem tissues_used (total_tissues := boxes_bought * box_tissues) : total_tissues - tissues_left = 210 := by
  sorry

end tissues_used_l2073_207332


namespace evaluate_expression_l2073_207387

theorem evaluate_expression : (1 / (3 - (1 / (3 - (1 / (3 - (1 / 3))))))) = (8 / 21) :=
by
  sorry

end evaluate_expression_l2073_207387


namespace divisibility_2_pow_a_plus_1_l2073_207396

theorem divisibility_2_pow_a_plus_1 (a b : ℕ) (h_b_pos : 0 < b) (h_b_ge_2 : 2 ≤ b) 
  (h_div : (2^a + 1) % (2^b - 1) = 0) : b = 2 := by
  sorry

end divisibility_2_pow_a_plus_1_l2073_207396


namespace paint_quantity_l2073_207369

variable (totalPaint : ℕ) (blueRatio greenRatio whiteRatio : ℕ)

theorem paint_quantity 
  (h_total_paint : totalPaint = 45)
  (h_ratio_blue : blueRatio = 5)
  (h_ratio_green : greenRatio = 3)
  (h_ratio_white : whiteRatio = 7) :
  let totalRatio := blueRatio + greenRatio + whiteRatio
  let partQuantity := totalPaint / totalRatio
  let bluePaint := blueRatio * partQuantity
  let greenPaint := greenRatio * partQuantity
  let whitePaint := whiteRatio * partQuantity
  bluePaint = 15 ∧ greenPaint = 9 ∧ whitePaint = 21 :=
by
  sorry

end paint_quantity_l2073_207369


namespace baker_initial_cakes_cannot_be_determined_l2073_207313

theorem baker_initial_cakes_cannot_be_determined (initial_pastries sold_cakes sold_pastries remaining_pastries : ℕ)
  (h1 : initial_pastries = 148)
  (h2 : sold_cakes = 15)
  (h3 : sold_pastries = 103)
  (h4 : remaining_pastries = 45)
  (h5 : sold_pastries + remaining_pastries = initial_pastries) :
  True :=
by
  sorry

end baker_initial_cakes_cannot_be_determined_l2073_207313


namespace locus_of_D_l2073_207318

theorem locus_of_D 
  (a b : ℝ)
  (hA : 0 ≤ a ∧ a ≤ (2 * Real.sqrt 3 / 3))
  (hB : 0 ≤ b ∧ b ≤ (2 * Real.sqrt 3 / 3))
  (AB_eq : Real.sqrt ((b - 2 * a)^2 + (Real.sqrt 3 * b)^2)  = 2) :
  3 * (b - a / 2)^2 + (Real.sqrt 3 / 2 * (a + b))^2 / 3 = 1 :=
sorry

end locus_of_D_l2073_207318


namespace smallest_lcm_l2073_207370

theorem smallest_lcm (m n : ℕ) (hm : 10000 ≤ m ∧ m < 100000) (hn : 10000 ≤ n ∧ n < 100000) (h : Nat.gcd m n = 5) : Nat.lcm m n = 20030010 :=
sorry

end smallest_lcm_l2073_207370


namespace odd_function_increasing_on_negative_interval_l2073_207363

theorem odd_function_increasing_on_negative_interval {f : ℝ → ℝ}
  (h_odd : ∀ x, f (-x) = -f x)
  (h_increasing : ∀ x y, 3 ≤ x → x ≤ 7 → 3 ≤ y → y ≤ 7 → x < y → f x < f y)
  (h_min_value : f 3 = 1) :
  (∀ x y, -7 ≤ x → x ≤ -3 → -7 ≤ y → y ≤ -3 → x < y → f x < f y) ∧ f (-3) = -1 := 
sorry

end odd_function_increasing_on_negative_interval_l2073_207363


namespace three_distinct_roots_condition_l2073_207304

noncomputable def k_condition (k : ℝ) : Prop :=
  ∀ (x : ℝ), (x / (x - 1) + x / (x - 3)) = k * x → 
    (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3)

theorem three_distinct_roots_condition (k : ℝ) : k ≠ 0 ↔ k_condition k :=
by
  sorry

end three_distinct_roots_condition_l2073_207304


namespace simple_interest_correct_l2073_207356

-- Define the given conditions
def Principal : ℝ := 9005
def Rate : ℝ := 0.09
def Time : ℝ := 5

-- Define the simple interest function
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- State the theorem to prove the total interest earned
theorem simple_interest_correct : simple_interest Principal Rate Time = 4052.25 := sorry

end simple_interest_correct_l2073_207356


namespace no_integer_b_such_that_b_sq_plus_3b_plus_1_is_perfect_square_l2073_207385

theorem no_integer_b_such_that_b_sq_plus_3b_plus_1_is_perfect_square :
  ∀ b : ℤ, ¬ ∃ k : ℤ, b^2 + 3*b + 1 = k^2 :=
by
  sorry

end no_integer_b_such_that_b_sq_plus_3b_plus_1_is_perfect_square_l2073_207385


namespace tax_refund_l2073_207379

-- Definitions based on the problem conditions
def monthly_salary : ℕ := 9000
def treatment_cost : ℕ := 100000
def medication_cost : ℕ := 20000
def tax_rate : ℚ := 0.13

-- Annual salary calculation
def annual_salary := monthly_salary * 12

-- Total spending on treatment and medications
def total_spending := treatment_cost + medication_cost

-- Possible tax refund based on total spending
def possible_tax_refund := total_spending * tax_rate

-- Income tax paid on the annual salary
def income_tax_paid := annual_salary * tax_rate

-- Prove statement that the actual tax refund is equal to income tax paid
theorem tax_refund : income_tax_paid = 14040 := by
  sorry

end tax_refund_l2073_207379


namespace length_of_one_side_nonagon_l2073_207381

def total_perimeter (n : ℕ) (side_length : ℝ) : ℝ := n * side_length

theorem length_of_one_side_nonagon (total_perimeter : ℝ) (n : ℕ) (side_length : ℝ) (h1 : n = 9) (h2 : total_perimeter = 171) : side_length = 19 :=
by
  sorry

end length_of_one_side_nonagon_l2073_207381


namespace initial_storks_count_l2073_207330

-- Definitions based on the conditions provided
def initialBirds : ℕ := 3
def additionalStorks : ℕ := 6
def totalBirdsAndStorks : ℕ := 13

-- The mathematical statement to be proved
theorem initial_storks_count (S : ℕ) (h : initialBirds + S + additionalStorks = totalBirdsAndStorks) : S = 4 :=
by
  sorry

end initial_storks_count_l2073_207330


namespace rectangular_garden_width_l2073_207353

theorem rectangular_garden_width (w : ℕ) (h1 : ∃ l : ℕ, l = 3 * w) (h2 : w * (3 * w) = 507) : w = 13 := 
by 
  sorry

end rectangular_garden_width_l2073_207353


namespace extra_discount_percentage_l2073_207367

theorem extra_discount_percentage 
  (initial_price : ℝ)
  (first_discount : ℝ)
  (new_price : ℝ)
  (final_price : ℝ)
  (extra_discount_amount : ℝ)
  (x : ℝ)
  (discount_formula : x = (extra_discount_amount * 100) / new_price) :
  initial_price = 50 ∧ 
  first_discount = 2.08 ∧ 
  new_price = 47.92 ∧ 
  final_price = 46 ∧ 
  extra_discount_amount = new_price - final_price → 
  x = 4 :=
by
  -- The proof will go here
  sorry

end extra_discount_percentage_l2073_207367


namespace cubic_solution_l2073_207383

theorem cubic_solution (a b c : ℝ) (h_eq : ∀ x, x^3 - 4*x^2 + 7*x + 6 = 34 -> x = a ∨ x = b ∨ x = c)
(h_ge : a ≥ b ∧ b ≥ c) : 2 * a + b = 8 := 
sorry

end cubic_solution_l2073_207383


namespace proof_q_values_proof_q_comparison_l2073_207300

-- Definitions of the conditions given.
def q : ℝ → ℝ := 
  sorry -- The definition is not required to be constructed, as we are only focusing on the conditions given.

-- Conditions
axiom cond1 : q 2 = 5
axiom cond2 : q 1.5 = 3

-- Statements to prove
theorem proof_q_values : (q 2 = 5) ∧ (q 1.5 = 3) := 
  by sorry

theorem proof_q_comparison : q 2 > q 1.5 :=
  by sorry

end proof_q_values_proof_q_comparison_l2073_207300


namespace sqrt_mul_l2073_207350

theorem sqrt_mul (h₁ : Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3) : Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3 :=
by
  sorry

end sqrt_mul_l2073_207350


namespace amy_money_left_l2073_207327

-- Definitions for item prices
def stuffed_toy_price : ℝ := 2
def hot_dog_price : ℝ := 3.5
def candy_apple_price : ℝ := 1.5
def soda_price : ℝ := 1.75
def ferris_wheel_ticket_price : ℝ := 2.5

-- Tax rate
def tax_rate : ℝ := 0.1 

-- Initial amount Amy had
def initial_amount : ℝ := 15

-- Function to calculate price including tax
def price_with_tax (price : ℝ) (tax_rate : ℝ) : ℝ := price * (1 + tax_rate)

-- Prices including tax
def stuffed_toy_price_with_tax := price_with_tax stuffed_toy_price tax_rate
def hot_dog_price_with_tax := price_with_tax hot_dog_price tax_rate
def candy_apple_price_with_tax := price_with_tax candy_apple_price tax_rate
def soda_price_with_tax := price_with_tax soda_price tax_rate
def ferris_wheel_ticket_price_with_tax := price_with_tax ferris_wheel_ticket_price tax_rate

-- Discount rates
def discount_most_expensive : ℝ := 0.5
def discount_second_most_expensive : ℝ := 0.25

-- Applying discounts
def discounted_hot_dog_price := hot_dog_price_with_tax * (1 - discount_most_expensive)
def discounted_ferris_wheel_ticket_price := ferris_wheel_ticket_price_with_tax * (1 - discount_second_most_expensive)

-- Total cost with discounts
def total_cost_with_discounts : ℝ := 
  stuffed_toy_price_with_tax + discounted_hot_dog_price + candy_apple_price_with_tax +
  soda_price_with_tax + discounted_ferris_wheel_ticket_price

-- Amount left after purchases
def amount_left : ℝ := initial_amount - total_cost_with_discounts

theorem amy_money_left : amount_left = 5.23 := by
  -- Here the proof will be provided.
  sorry

end amy_money_left_l2073_207327


namespace total_amount_silver_l2073_207398

theorem total_amount_silver (x y : ℝ) (h₁ : y = 7 * x + 4) (h₂ : y = 9 * x - 8) : y = 46 :=
by {
  sorry
}

end total_amount_silver_l2073_207398


namespace find_number_l2073_207391

theorem find_number (x : ℝ) : 3 * (2 * x + 9) = 57 → x = 5 :=
by
  sorry

end find_number_l2073_207391


namespace graph_does_not_pass_second_quadrant_l2073_207312

theorem graph_does_not_pass_second_quadrant (a b : ℝ) (h₀ : 1 < a) (h₁ : b < -1) : 
∀ x : ℝ, ¬ (y = a^x + b ∧ y > 0 ∧ x < 0) :=
by
  sorry

end graph_does_not_pass_second_quadrant_l2073_207312


namespace quadratic_condition_l2073_207347

theorem quadratic_condition (m : ℤ) (x : ℝ) :
  (m + 1) * x^(m^2 + 1) - 2 * x - 5 = 0 ∧ m^2 + 1 = 2 ∧ m + 1 ≠ 0 ↔ m = 1 := 
by
  sorry

end quadratic_condition_l2073_207347


namespace remaining_books_l2073_207361

def initial_books : Nat := 500
def num_people_donating : Nat := 10
def books_per_person : Nat := 8
def borrowed_books : Nat := 220

theorem remaining_books :
  (initial_books + num_people_donating * books_per_person - borrowed_books) = 360 := 
by 
  -- This will contain the mathematical proof
  sorry

end remaining_books_l2073_207361


namespace solve_investment_problem_l2073_207380

def remaining_rate_proof (A I A1 R1 A2 R2 x : ℚ) : Prop :=
  let income1 := A1 * (R1 / 100)
  let income2 := A2 * (R2 / 100)
  let remaining := A - A1 - A2
  let required_income := I - (income1 + income2)
  let expected_rate_in_float := (required_income / remaining) * 100
  expected_rate_in_float = x

theorem solve_investment_problem :
  remaining_rate_proof 15000 800 5000 3 6000 4.5 9.5 :=
by
  -- proof goes here
  sorry

end solve_investment_problem_l2073_207380


namespace range_k_l2073_207360

theorem range_k (k : ℝ) :
  (∀ x : ℝ, (3/8 - k*x - 2*k*x^2) ≥ 0) ↔ (-3 ≤ k ∧ k ≤ 0) :=
sorry

end range_k_l2073_207360


namespace sqrt_multiplication_l2073_207388

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l2073_207388


namespace gecko_bug_eating_l2073_207378

theorem gecko_bug_eating (G L F T : ℝ) (hL : L = G / 2)
                                      (hF : F = 3 * L)
                                      (hT : T = 1.5 * F)
                                      (hTotal : G + L + F + T = 63) :
  G = 15 :=
by
  sorry

end gecko_bug_eating_l2073_207378


namespace range_of_derivative_max_value_of_a_l2073_207389

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ :=
  a * Real.cos x - (x - Real.pi / 2) * Real.sin x

-- Define the derivative of f
noncomputable def f' (a x : ℝ) : ℝ :=
  -(1 + a) * Real.sin x - (x - Real.pi / 2) * Real.cos x

-- Part (1): Prove the range of the derivative when a = -1 is [0, π/2]
theorem range_of_derivative (x : ℝ) (h0 : 0 ≤ x) (hπ : x ≤ Real.pi / 2) :
  (0 ≤ f' (-1) x) ∧ (f' (-1) x ≤ Real.pi / 2) := 
sorry

-- Part (2): Prove the maximum value of 'a' when f(x) ≤ 0 always holds
theorem max_value_of_a (a : ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f a x ≤ 0) :
  a ≤ -1 := 
sorry

end range_of_derivative_max_value_of_a_l2073_207389


namespace smallest_n_l2073_207328

theorem smallest_n (j c g : ℕ) (n : ℕ) (total_cost : ℕ) 
  (h_condition : total_cost = 10 * j ∧ total_cost = 16 * c ∧ total_cost = 18 * g ∧ total_cost = 24 * n) 
  (h_lcm : Nat.lcm (Nat.lcm 10 16) 18 = 720) : n = 30 :=
by
  sorry

end smallest_n_l2073_207328


namespace find_y_value_l2073_207392

theorem find_y_value (y : ℕ) (h1 : y ≤ 150)
  (h2 : (45 + 76 + 123 + y + y + y) / 6 = 2 * y) :
  y = 27 :=
sorry

end find_y_value_l2073_207392


namespace proof_problem_l2073_207376

theorem proof_problem (x : ℝ) 
    (h1 : (x - 1) * (x + 1) = x^2 - 1)
    (h2 : (x - 1) * (x^2 + x + 1) = x^3 - 1)
    (h3 : (x - 1) * (x^3 + x^2 + x + 1) = x^4 - 1)
    (h4 : (x - 1) * (x^4 + x^3 + x^2 + x + 1) = -2) :
    x^2023 = -1 := 
by 
  sorry -- Proof is omitted

end proof_problem_l2073_207376


namespace crayons_more_than_erasers_l2073_207359

-- Definitions of the conditions
def initial_crayons := 531
def initial_erasers := 38
def final_crayons := 391
def final_erasers := initial_erasers -- no erasers lost

-- Theorem statement
theorem crayons_more_than_erasers :
  final_crayons - final_erasers = 102 :=
by
  -- Placeholder for the proof
  sorry

end crayons_more_than_erasers_l2073_207359


namespace exist_infinitely_many_coprime_pairs_l2073_207325

theorem exist_infinitely_many_coprime_pairs (a b : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : Nat.gcd a b = 1) : 
  ∃ (a b : ℕ), (a + b).mod (a^b + b^a) = 0 :=
sorry

end exist_infinitely_many_coprime_pairs_l2073_207325


namespace solve_congruence_l2073_207317

theorem solve_congruence :
  ∃ n : ℤ, 19 * n ≡ 13 [ZMOD 47] ∧ n ≡ 25 [ZMOD 47] :=
by
  sorry

end solve_congruence_l2073_207317


namespace gcd_72_120_168_l2073_207399

theorem gcd_72_120_168 : Nat.gcd (Nat.gcd 72 120) 168 = 24 := 
by
  -- Each step would be proven individually here.
  sorry

end gcd_72_120_168_l2073_207399


namespace total_workers_in_workshop_l2073_207395

theorem total_workers_in_workshop 
  (W : ℕ)
  (T : ℕ := 5)
  (avg_all : ℕ := 700)
  (avg_technicians : ℕ := 800)
  (avg_rest : ℕ := 650) 
  (total_salary_all : ℕ := W * avg_all)
  (total_salary_technicians : ℕ := T * avg_technicians)
  (total_salary_rest : ℕ := (W - T) * avg_rest) :
  total_salary_all = total_salary_technicians + total_salary_rest →
  W = 15 :=
by
  sorry

end total_workers_in_workshop_l2073_207395


namespace shelves_used_l2073_207339

theorem shelves_used (initial_books : ℕ) (sold_books : ℕ) (books_per_shelf : ℕ) (remaining_books : ℕ) (total_shelves : ℕ) :
  initial_books = 120 → sold_books = 39 → books_per_shelf = 9 → remaining_books = initial_books - sold_books → total_shelves = remaining_books / books_per_shelf → total_shelves = 9 :=
by
  intros h_initial_books h_sold_books h_books_per_shelf h_remaining_books h_total_shelves
  rw [h_initial_books, h_sold_books] at h_remaining_books
  rw [h_books_per_shelf, h_remaining_books] at h_total_shelves
  exact h_total_shelves

end shelves_used_l2073_207339


namespace expected_value_winnings_l2073_207341

def probability_heads : ℚ := 2 / 5
def probability_tails : ℚ := 3 / 5
def win_amount_heads : ℚ := 5
def lose_amount_tails : ℚ := -4

theorem expected_value_winnings : 
  probability_heads * win_amount_heads + probability_tails * lose_amount_tails = -2 / 5 := 
by 
  sorry

end expected_value_winnings_l2073_207341


namespace arithmetic_seq_sum_l2073_207377

theorem arithmetic_seq_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 3 = 9)
  (h3 : a 5 = 5) :
  S 9 / S 5 = 1 :=
by
  sorry

end arithmetic_seq_sum_l2073_207377


namespace right_isosceles_triangle_areas_l2073_207315

theorem right_isosceles_triangle_areas :
  let A := (5 * 5) / 2
  let B := (12 * 12) / 2
  let C := (13 * 13) / 2
  A + B = C :=
by
  let A := (5 * 5) / 2
  let B := (12 * 12) / 2
  let C := (13 * 13) / 2
  sorry

end right_isosceles_triangle_areas_l2073_207315


namespace train_speed_from_clicks_l2073_207324

theorem train_speed_from_clicks (speed_mph : ℝ) (rail_length_ft : ℝ) (clicks_heard : ℝ) :
  rail_length_ft = 40 →
  clicks_heard = 1 →
  (60 * rail_length_ft * clicks_heard * speed_mph / 5280) = 27 :=
by
  intros h1 h2
  sorry

end train_speed_from_clicks_l2073_207324


namespace ball_count_proof_l2073_207309

noncomputable def valid_ball_count : ℕ :=
  150

def is_valid_ball_count (N : ℕ) : Prop :=
  80 < N ∧ N ≤ 200 ∧
  (∃ y b w r : ℕ,
    y = Nat.div (12 * N) 100 ∧
    b = Nat.div (20 * N) 100 ∧
    w = 2 * Nat.div N 3 ∧
    r = N - (y + b + w) ∧
    r.mod N = 0 )

theorem ball_count_proof : is_valid_ball_count valid_ball_count :=
by
  -- The proof would be inserted here.
  sorry

end ball_count_proof_l2073_207309


namespace find_original_number_l2073_207348

theorem find_original_number (a b c : ℕ) (h : 100 * a + 10 * b + c = 390) 
  (N : ℕ) (hN : N = 4326) : a = 3 ∧ b = 9 ∧ c = 0 :=
by 
  sorry

end find_original_number_l2073_207348


namespace original_number_l2073_207310

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def digit_list (n : ℕ) (a b c d e : ℕ) : Prop :=
  n = 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e

def four_digit_variant (N n : ℕ) (a b c d e : ℕ) : Prop :=
  (n = 10^3 * b + 10^2 * c + 10 * d + e) ∨
  (n = 10^4 * a + 10^2 * c + 10 * d + e) ∨
  (n = 10^4 * a + 10^3 * b + 10 * d + e) ∨
  (n = 10^4 * a + 10^3 * b + 10^2 * c + e) ∨
  (n = 10^4 * a + 10^3 * b + 10^2 * c + 10 * d)

theorem original_number (N : ℕ) (a b c d e : ℕ) 
  (h1 : is_five_digit N) 
  (h2 : digit_list N a b c d e)
  (h3 : ∃ n, is_five_digit n ∧ four_digit_variant N n a b c d e ∧ N + n = 54321) :
  N = 49383 := 
sorry

end original_number_l2073_207310


namespace symmetric_point_x_correct_l2073_207384

-- Define the Cartesian coordinate system
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the symmetry with respect to the x-axis
def symmetricPointX (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

-- Given point (-2, 1, 4)
def givenPoint : Point3D := { x := -2, y := 1, z := 4 }

-- Define the expected symmetric point
def expectedSymmetricPoint : Point3D := { x := -2, y := -1, z := -4 }

-- State the theorem to prove the expected symmetric point
theorem symmetric_point_x_correct :
  symmetricPointX givenPoint = expectedSymmetricPoint := by
  -- here the proof would go, but we leave it as sorry
  sorry

end symmetric_point_x_correct_l2073_207384


namespace x_zero_necessary_but_not_sufficient_l2073_207355

-- Definitions based on conditions
def x_eq_zero (x : ℝ) := x = 0
def xsq_plus_ysq_eq_zero (x y : ℝ) := x^2 + y^2 = 0

-- Statement that x = 0 is a necessary but not sufficient condition for x^2 + y^2 = 0
theorem x_zero_necessary_but_not_sufficient (x y : ℝ) : (x = 0 ↔ x^2 + y^2 = 0) → False :=
by sorry

end x_zero_necessary_but_not_sufficient_l2073_207355


namespace factor_polynomial_l2073_207305

def p (x y z : ℝ) : ℝ := x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2)

theorem factor_polynomial (x y z : ℝ) : 
  p x y z = (x - y) * (y - z) * (z - x) * -(x * y + x * z + y * z) :=
by 
  simp [p]
  sorry

end factor_polynomial_l2073_207305


namespace roots_of_polynomial_l2073_207320

theorem roots_of_polynomial :
  ∀ x : ℝ, (x^3 - 3 * x^2 + 2 * x) * (x - 5) = 0 ↔ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 5 :=
by 
  sorry

end roots_of_polynomial_l2073_207320


namespace temperature_decrease_l2073_207314

theorem temperature_decrease (current_temp : ℝ) (future_temp : ℝ) : 
  current_temp = 84 → future_temp = (3 / 4) * current_temp → (current_temp - future_temp) = 21 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end temperature_decrease_l2073_207314


namespace weight_of_b_l2073_207354

theorem weight_of_b (a b c : ℝ) (h1 : (a + b + c) / 3 = 45) (h2 : (a + b) / 2 = 40) (h3 : (b + c) / 2 = 43) : b = 31 :=
by
  sorry

end weight_of_b_l2073_207354


namespace elizabeth_stickers_l2073_207374

def total_stickers (initial_bottles lost_bottles stolen_bottles stickers_per_bottle : ℕ) : ℕ :=
  let remaining_bottles := initial_bottles - lost_bottles - stolen_bottles
  remaining_bottles * stickers_per_bottle

theorem elizabeth_stickers :
  total_stickers 10 2 1 3 = 21 :=
by
  sorry

end elizabeth_stickers_l2073_207374


namespace solve_color_problem_l2073_207390

variables (R B G C : Prop)

def color_problem (R B G C : Prop) : Prop :=
  (C → (R ∨ B)) ∧ (¬C → (¬R ∧ ¬G)) ∧ ((B ∨ G) → C) → C ∧ (R ∨ B)

theorem solve_color_problem (R B G C : Prop) (h : (C → (R ∨ B)) ∧ (¬C → (¬R ∧ ¬G)) ∧ ((B ∨ G) → C)) : C ∧ (R ∨ B) :=
  by {
    sorry
  }

end solve_color_problem_l2073_207390


namespace percent_increase_in_pizza_area_l2073_207362

theorem percent_increase_in_pizza_area (r : ℝ) (h : 0 < r) :
  let r_large := 1.10 * r
  let A_medium := π * r^2
  let A_large := π * r_large^2
  let percent_increase := ((A_large - A_medium) / A_medium) * 100 
  percent_increase = 21 := 
by sorry

end percent_increase_in_pizza_area_l2073_207362


namespace problem_solution_l2073_207357

theorem problem_solution : (324^2 - 300^2) / 24 = 624 :=
by 
  -- The proof will be inserted here.
  sorry

end problem_solution_l2073_207357


namespace salmon_trip_l2073_207397

theorem salmon_trip (male_salmons : ℕ) (female_salmons : ℕ) : male_salmons = 712261 → female_salmons = 259378 → male_salmons + female_salmons = 971639 :=
  sorry

end salmon_trip_l2073_207397


namespace tomatoes_picked_yesterday_l2073_207366

-- Definitions corresponding to the conditions in the problem.
def initial_tomatoes : Nat := 160
def tomatoes_left_after_yesterday : Nat := 104

-- Statement of the problem proving the number of tomatoes picked yesterday.
theorem tomatoes_picked_yesterday : initial_tomatoes - tomatoes_left_after_yesterday = 56 :=
by
  sorry

end tomatoes_picked_yesterday_l2073_207366


namespace existence_of_ab_l2073_207337

theorem existence_of_ab (n : ℕ) (hn : 0 < n) : ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ n ∣ (4 * a^2 + 9 * b^2 - 1) :=
by 
  sorry

end existence_of_ab_l2073_207337


namespace sum_of_numbers_l2073_207306

theorem sum_of_numbers : ∃ (a b : ℕ), (a + b = 21) ∧ (a / b = 3 / 4) ∧ (max a b = 12) :=
by
  sorry

end sum_of_numbers_l2073_207306


namespace calculate_expression_l2073_207329

variables (x y : ℝ)

theorem calculate_expression (hx : 0 ≤ x) (hy : 0 ≤ y) :
  (x - y) / (Real.sqrt x + Real.sqrt y) - (x - 2 * Real.sqrt (x * y) + y) / (Real.sqrt x - Real.sqrt y) = 0 :=
by
  sorry

end calculate_expression_l2073_207329


namespace find_x_l2073_207344

variables (z y x : Int)

def condition1 : Prop := z + 1 = 0
def condition2 : Prop := y - 1 = 1
def condition3 : Prop := x + 2 = -1

theorem find_x (h1 : condition1 z) (h2 : condition2 y) (h3 : condition3 x) : x = -3 :=
by
  sorry

end find_x_l2073_207344


namespace find_m_l2073_207382

/-
Define the ellipse equation
-/
def ellipse_eqn (x y : ℝ) : Prop :=
  (x^2 / 9) + (y^2) = 1

/-
Define the region R
-/
def region_R (x y : ℝ) : Prop :=
  (x ≥ 0) ∧ (y ≥ 0) ∧ (2*y = x) ∧ ellipse_eqn x y

/-
Define the region R'
-/
def region_R' (x y m : ℝ) : Prop :=
  (x ≥ 0) ∧ (y ≥ 0) ∧ (y = m*x) ∧ ellipse_eqn x y

/-
The statement we want to prove
-/
theorem find_m (m : ℝ) : (∃ (x y : ℝ), region_R x y) ∧ (∃ (x y : ℝ), region_R' x y m) →
(m = (2 : ℝ) / 9) := 
sorry

end find_m_l2073_207382


namespace range_of_m_l2073_207335

variable {x m : ℝ}

theorem range_of_m (h1 : x + 2 < 2 * m) (h2 : x - m < 0) (h3 : x < 2 * m - 2) : m ≤ 2 :=
sorry

end range_of_m_l2073_207335


namespace value_of_expr_l2073_207321

noncomputable def verify_inequality (x a b c : ℝ) : Prop :=
  (x - a) * (x - b) / (x - c) ≥ 0

theorem value_of_expr (a b c : ℝ) :
  (∀ x : ℝ, verify_inequality x a b c ↔ (x < -6 ∨ abs (x - 30) ≤ 2)) →
  a < b →
  a = 28 →
  b = 32 →
  c = -6 →
  a + 2 * b + 3 * c = 74 := by
  sorry

end value_of_expr_l2073_207321


namespace large_block_dimension_ratio_l2073_207303

theorem large_block_dimension_ratio
  (V_normal V_large : ℝ) 
  (k : ℝ)
  (h1 : V_normal = 4)
  (h2 : V_large = 32) 
  (h3 : V_large = k^3 * V_normal) :
  k = 2 := by
  sorry

end large_block_dimension_ratio_l2073_207303


namespace find_two_digit_number_l2073_207371

theorem find_two_digit_number :
  ∃ x y : ℕ, 10 * x + y = 78 ∧ 10 * x + y < 100 ∧ y ≠ 0 ∧ (10 * x + y) / y = 9 ∧ (10 * x + y) % y = 6 :=
by
  sorry

end find_two_digit_number_l2073_207371


namespace total_combined_grapes_l2073_207307

theorem total_combined_grapes :
  ∀ (r a y : ℕ), (r = 25) → (a = r + 2) → (y = a + 4) → (r + a + y = 83) :=
by
  intros r a y hr ha hy
  rw [hr, ha, hy]
  sorry

end total_combined_grapes_l2073_207307


namespace find_a2_and_sum_l2073_207365

theorem find_a2_and_sum (a a1 a2 a3 a4 : ℝ) (x : ℝ) (h1 : (1 + 2 * x)^4 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4) :
  a2 = 24 ∧ a + a1 + a2 + a3 + a4 = 81 :=
by
  sorry

end find_a2_and_sum_l2073_207365


namespace find_x0_l2073_207334

/-- Given that the tangent line to the curve y = x^2 - 1 at the point x = x0 is parallel 
to the tangent line to the curve y = 1 - x^3 at the point x = x0, prove that x0 = 0 
or x0 = -2/3. -/
theorem find_x0 (x0 : ℝ) (h : (∃ x0, (2 * x0) = (-3 * x0 ^ 2))) : x0 = 0 ∨ x0 = -2/3 := 
sorry

end find_x0_l2073_207334


namespace Xiaoming_speed_l2073_207368

theorem Xiaoming_speed (x xiaohong_speed_xiaoming_diff : ℝ) :
  (50 * (2 * x + 2) = 600) →
  (xiaohong_speed_xiaoming_diff = 2) →
  x + xiaohong_speed_xiaoming_diff = 7 :=
by
  intros h₁ h₂
  sorry

end Xiaoming_speed_l2073_207368


namespace sequence_property_l2073_207308

-- Conditions as definitions
def seq (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  (a 1 = -(2 / 3)) ∧ (∀ n ≥ 2, S n + (1 / S n) + 2 = a n)

-- The desired property of the sequence
def S_property (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < n → S n = -((n + 1) / (n + 2))

-- The main theorem
theorem sequence_property (a S : ℕ → ℝ) (h_seq : seq a S) : S_property S := sorry

end sequence_property_l2073_207308
