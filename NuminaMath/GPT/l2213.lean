import Mathlib

namespace henri_drove_farther_l2213_221318

theorem henri_drove_farther (gervais_avg_miles_per_day : ℕ) (gervais_days : ℕ) (henri_total_miles : ℕ)
  (h1 : gervais_avg_miles_per_day = 315) (h2 : gervais_days = 3) (h3 : henri_total_miles = 1250) :
  (henri_total_miles - (gervais_avg_miles_per_day * gervais_days) = 305) :=
by
  -- Here we would provide the proof, but we are omitting it as requested
  sorry

end henri_drove_farther_l2213_221318


namespace color_ball_ratios_l2213_221383

theorem color_ball_ratios (white_balls red_balls blue_balls : ℕ)
  (h_white : white_balls = 12)
  (h_red_ratio : 4 * red_balls = 3 * white_balls)
  (h_blue_ratio : 4 * blue_balls = 2 * white_balls) :
  red_balls = 9 ∧ blue_balls = 6 :=
by
  sorry

end color_ball_ratios_l2213_221383


namespace trains_crossing_time_l2213_221339

theorem trains_crossing_time
  (L speed1 speed2 : ℝ)
  (time_same_direction time_opposite_direction : ℝ) 
  (h1 : speed1 = 60)
  (h2 : speed2 = 40)
  (h3 : time_same_direction = 40)
  (h4 : 2 * L = (speed1 - speed2) * 5/18 * time_same_direction) :
  time_opposite_direction = 8 := 
sorry

end trains_crossing_time_l2213_221339


namespace juniper_bones_proof_l2213_221398

-- Define the conditions
def juniper_original_bones : ℕ := 4
def bones_given_by_master : ℕ := juniper_original_bones
def bones_stolen_by_neighbor : ℕ := 2

-- Define the final number of bones Juniper has
def juniper_remaining_bones : ℕ := juniper_original_bones + bones_given_by_master - bones_stolen_by_neighbor

-- State the theorem to prove the given answer
theorem juniper_bones_proof : juniper_remaining_bones = 6 :=
by
  -- Proof omitted
  sorry

end juniper_bones_proof_l2213_221398


namespace max_free_squares_l2213_221388

theorem max_free_squares (n : ℕ) :
  ∀ (initial_positions : ℕ), 
    (∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → initial_positions = 2) →
    (∀ (i j : ℕ) (move1 move2 : ℕ × ℕ),
       1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n →
       move1 = (i + 1, j) ∨ move1 = (i - 1, j) ∨ move1 = (i, j + 1) ∨ move1 = (i, j - 1) →
       move2 = (i + 1, j) ∨ move2 = (i - 1, j) ∨ move2 = (i, j + 1) ∨ move2 = (i, j - 1) →
       move1 ≠ move2) →
    ∃ free_squares : ℕ, free_squares = n^2 :=
by
  sorry

end max_free_squares_l2213_221388


namespace number_of_correct_conclusions_l2213_221349

-- Define the conditions as hypotheses
variable {a_n : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}
variable {a_1 : ℝ}
variable {n : ℕ}

-- Arithmetic sequence definition for a_n
def arithmetic_sequence (a_n : ℕ → ℝ) (a_1 d : ℝ) : Prop :=
  ∀ n, a_n n = a_1 + (n - 1) * d

-- Sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (S : ℕ → ℝ) (a_1 d : ℝ) : Prop :=
  ∀ n, S n = n / 2 * (2 * a_1 + (n - 1) * d)

-- Problem statement
theorem number_of_correct_conclusions 
  (h_seq : arithmetic_sequence a_n a_1 d)
  (h_sum : sum_arithmetic_sequence S a_1 d)
  (h1 : S 5 < S 6)
  (h2 : S 6 = S 7 ∧ S 7 > S 8) :
  ∃ n, n = 3 ∧ 
       (d < 0) ∧ 
       (a_n 7 = 0) ∧ 
       ¬(S 9 = S 5) ∧ 
       (S 6 = S 7 ∧ ∀ m, m > 7 → S m < S 6) := 
sorry

end number_of_correct_conclusions_l2213_221349


namespace laticia_total_pairs_l2213_221334

-- Definitions of the conditions about the pairs of socks knitted each week

-- Number of pairs knitted in the first week
def pairs_week1 : ℕ := 12

-- Number of pairs knitted in the second week
def pairs_week2 : ℕ := pairs_week1 + 4

-- Number of pairs knitted in the third week
def pairs_week3 : ℕ := (pairs_week1 + pairs_week2) / 2

-- Number of pairs knitted in the fourth week
def pairs_week4 : ℕ := pairs_week3 - 3

-- Statement: Sum of pairs over the four weeks
theorem laticia_total_pairs :
  pairs_week1 + pairs_week2 + pairs_week3 + pairs_week4 = 53 := by
  sorry

end laticia_total_pairs_l2213_221334


namespace xiaoliang_steps_l2213_221321

/-- 
  Xiaoping lives on the fifth floor and climbs 80 steps to get home every day.
  Xiaoliang lives on the fourth floor.
  Prove that the number of steps Xiaoliang has to climb is 60.
-/
theorem xiaoliang_steps (steps_per_floor : ℕ) (h_xiaoping : 4 * steps_per_floor = 80) : 3 * steps_per_floor = 60 :=
by {
  -- The proof is intentionally left out
  sorry
}

end xiaoliang_steps_l2213_221321


namespace isosceles_triangle_length_l2213_221304

variable (a b : ℝ)

theorem isosceles_triangle_length (h1 : 2 * a + 3 = 16) (h2 : a != 3) : a = 6.5 :=
sorry

end isosceles_triangle_length_l2213_221304


namespace floor_equiv_l2213_221329

theorem floor_equiv {n : ℤ} (h : n > 2) : 
  Int.floor ((n * (n + 1) : ℚ) / (4 * n - 2 : ℚ)) = Int.floor ((n + 1 : ℚ) / 4) := 
sorry

end floor_equiv_l2213_221329


namespace incorrect_assignment_statement_l2213_221359

theorem incorrect_assignment_statement :
  ∀ (a x y : ℕ), ¬(x * y = a) := by
sorry

end incorrect_assignment_statement_l2213_221359


namespace relationship_between_abc_l2213_221343

noncomputable def a : Real := (2 / 5) ^ (3 / 5)
noncomputable def b : Real := (2 / 5) ^ (2 / 5)
noncomputable def c : Real := (3 / 5) ^ (3 / 5)

theorem relationship_between_abc : a < b ∧ b < c := by
  sorry

end relationship_between_abc_l2213_221343


namespace polynomial_roots_sum_reciprocal_l2213_221319

open Polynomial

theorem polynomial_roots_sum_reciprocal (a b c : ℝ) (h : 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1) :
    (40 * a^3 - 70 * a^2 + 32 * a - 3 = 0) ∧
    (40 * b^3 - 70 * b^2 + 32 * b - 3 = 0) ∧
    (40 * c^3 - 70 * c^2 + 32 * c - 3 = 0) →
    (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) = 3 :=
by
  sorry

end polynomial_roots_sum_reciprocal_l2213_221319


namespace minimize_expression_at_c_l2213_221385

theorem minimize_expression_at_c (c : ℝ) : (c = 7 / 4) → (∀ x : ℝ, 2 * c^2 - 7 * c + 4 ≤ 2 * x^2 - 7 * x + 4) :=
sorry

end minimize_expression_at_c_l2213_221385


namespace abs_neg_2_plus_sqrt3_add_tan60_eq_2_l2213_221355

theorem abs_neg_2_plus_sqrt3_add_tan60_eq_2 :
  abs (-2 + Real.sqrt 3) + Real.tan (Real.pi / 3) = 2 :=
by
  sorry

end abs_neg_2_plus_sqrt3_add_tan60_eq_2_l2213_221355


namespace symmetry_in_mathematics_l2213_221328

-- Define the options
def optionA := "summation of harmonic series from 1 to 100"
def optionB := "general quadratic equation: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0"
def optionC := "Law of Sines: a / sin A = b / sin B = c / sin C"
def optionD := "arithmetic operation: 123456789 * 9 + 10 = 1111111111"

-- Define the symmetry property
def exhibits_symmetry (option: String) : Prop :=
  option = optionC

-- The theorem to prove
theorem symmetry_in_mathematics : ∃ option, exhibits_symmetry option := by
  use optionC
  sorry

end symmetry_in_mathematics_l2213_221328


namespace number_of_divisors_3465_l2213_221309

def prime_factors_3465 : Prop := 3465 = 3^2 * 5 * 7^2

theorem number_of_divisors_3465 (h : prime_factors_3465) : Nat.totient 3465 = 18 :=
  sorry

end number_of_divisors_3465_l2213_221309


namespace sum_coefficients_eq_neg_one_l2213_221330

theorem sum_coefficients_eq_neg_one (a a1 a2 a3 a4 a5 : ℝ) :
  (∀ x y : ℝ, (x - 2 * y)^5 = a * x^5 + a1 * x^4 * y + a2 * x^3 * y^2 + a3 * x^2 * y^3 + a4 * x * y^4 + a5 * y^5) →
  a + a1 + a2 + a3 + a4 + a5 = -1 :=
by
  sorry

end sum_coefficients_eq_neg_one_l2213_221330


namespace solve_equation_l2213_221376

theorem solve_equation (x : ℝ) (h : (x - 60) / 3 = (4 - 3 * x) / 6) : x = 124 / 5 := by
  sorry

end solve_equation_l2213_221376


namespace even_integers_between_sqrt_10_and_sqrt_100_l2213_221356

theorem even_integers_between_sqrt_10_and_sqrt_100 : 
  ∃ (n : ℕ), n = 4 ∧ (∀ (a : ℕ), (∃ k, (2 * k = a ∧ a > Real.sqrt 10 ∧ a < Real.sqrt 100)) ↔ 
  (a = 4 ∨ a = 6 ∨ a = 8 ∨ a = 10)) := 
by 
  sorry

end even_integers_between_sqrt_10_and_sqrt_100_l2213_221356


namespace probability_palindrome_divisible_by_11_is_zero_l2213_221395

-- Define the three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b + a

-- Define the divisibility condition
def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- Prove that the probability is zero
theorem probability_palindrome_divisible_by_11_is_zero :
  (∃ n, is_palindrome n ∧ is_divisible_by_11 n) →
  (0 : ℕ) = 0 := by
  sorry

end probability_palindrome_divisible_by_11_is_zero_l2213_221395


namespace point_B_position_l2213_221311

/-- Given points A and B on the same number line, with A at -2 and B 5 units away from A, prove 
    that B can be either -7 or 3. -/
theorem point_B_position (A B : ℤ) (hA : A = -2) (hB : (B = A + 5) ∨ (B = A - 5)) : 
  B = 3 ∨ B = -7 :=
sorry

end point_B_position_l2213_221311


namespace percentage_error_l2213_221353

theorem percentage_error (x : ℝ) (hx : x ≠ 0) :
  let correct_result := 10 * x
  let incorrect_result := x / 10
  let error := correct_result - incorrect_result
  let percentage_error := (error / correct_result) * 100
  percentage_error = 99 :=
by
  sorry

end percentage_error_l2213_221353


namespace original_cube_volume_l2213_221399

theorem original_cube_volume
  (a : ℝ)
  (h : (a + 2) * (a - 1) * a = a^3 + 14) :
  a^3 = 64 :=
by
  sorry

end original_cube_volume_l2213_221399


namespace ratio_of_a_b_l2213_221333

-- Define the problem
theorem ratio_of_a_b (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it.
  sorry

end ratio_of_a_b_l2213_221333


namespace solve_for_y_l2213_221357

theorem solve_for_y (x y : ℝ) : 3 * x + 5 * y = 10 → y = 2 - (3 / 5) * x :=
by 
  -- proof steps would be filled here
  sorry

end solve_for_y_l2213_221357


namespace negation_proposition_l2213_221367

variable (n : ℕ)
variable (n_positive : n > 0)
variable (f : ℕ → ℕ)
variable (H1 : ∀ n, n > 0 → (f n) > 0 ∧ (f n) ≤ n)

theorem negation_proposition :
  (∃ n_0, n_0 > 0 ∧ ((f n_0) ≤ 0 ∨ (f n_0) > n_0)) ↔ ¬(∀ n, n > 0 → (f n) >0 ∧ (f n) ≤ n) :=
by 
  sorry

end negation_proposition_l2213_221367


namespace distances_equal_l2213_221352

noncomputable def distance_from_point_to_line (x y m : ℝ) : ℝ :=
  |m * x + y + 3| / Real.sqrt (m^2 + 1)

theorem distances_equal (m : ℝ) :
  distance_from_point_to_line 3 2 m = distance_from_point_to_line (-1) 4 m ↔
  (m = 1 / 2 ∨ m = -6) := 
sorry

end distances_equal_l2213_221352


namespace max_ab_l2213_221364

theorem max_ab (a b c : ℝ) (h1 : 3 * a + b = 1) (h2 : 0 ≤ a) (h3 : a < 1) (h4 : 0 ≤ b) 
(h5 : b < 1) (h6 : 0 ≤ c) (h7 : c < 1) (h8 : a + b + c = 1) : 
  ab ≤ 1 / 12 := by
  sorry

end max_ab_l2213_221364


namespace geo_seq_sum_eq_l2213_221347

variable {a : ℕ → ℝ}

-- Conditions
def is_geo_seq (a : ℕ → ℝ) : Prop := ∃ r : ℝ, ∀ n : ℕ, a (n+1) = a n * r
def positive_seq (a : ℕ → ℝ) : Prop := ∀ n : ℕ, a n > 0
def specific_eq (a : ℕ → ℝ) : Prop := a 1 * a 3 + 2 * a 2 * a 4 + a 3 * a 5 = 25

theorem geo_seq_sum_eq (a : ℕ → ℝ) (h_geo : is_geo_seq a) (h_pos : positive_seq a) (h_eq : specific_eq a) : 
  a 2 + a 4 = 5 :=
by
  sorry

end geo_seq_sum_eq_l2213_221347


namespace simplify_and_evaluate_expression_l2213_221302

theorem simplify_and_evaluate_expression (a b : ℤ) (h_a : a = 2) (h_b : b = -1) : 
  2 * (-a^2 + 2 * a * b) - 3 * (a * b - a^2) = 2 :=
by 
  sorry

end simplify_and_evaluate_expression_l2213_221302


namespace condition_for_a_b_complex_l2213_221310

theorem condition_for_a_b_complex (a b : ℂ) (h1 : a ≠ 0) (h2 : 2 * a + b ≠ 0) :
  (2 * a + b) / a = b / (2 * a + b) → 
  (∃ z : ℂ, a = z ∨ b = z) ∨ 
  ((∃ z1 : ℂ, a = z1) ∧ (∃ z2 : ℂ, b = z2)) :=
sorry

end condition_for_a_b_complex_l2213_221310


namespace ways_to_make_change_l2213_221307

theorem ways_to_make_change : ∃ ways : ℕ, ways = 60 ∧ (∀ (p n d q : ℕ), p + 5 * n + 10 * d + 25 * q = 55 → True) := 
by
  -- The proof will go here
  sorry

end ways_to_make_change_l2213_221307


namespace smallest_integer_inequality_l2213_221362

theorem smallest_integer_inequality :
  ∃ n : ℤ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧ 
           (∀ m : ℤ, m < n → ¬∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ m * (x^4 + y^4 + z^4 + w^4)) :=
by
  sorry

end smallest_integer_inequality_l2213_221362


namespace odd_consecutive_nums_divisibility_l2213_221373

theorem odd_consecutive_nums_divisibility (a b : ℕ) (h_consecutive : b = a + 2) (h_odd_a : a % 2 = 1) (h_odd_b : b % 2 = 1) : (a^b + b^a) % (a + b) = 0 := by
  sorry

end odd_consecutive_nums_divisibility_l2213_221373


namespace tom_sawyer_bible_l2213_221375

def blue_tickets_needed (yellow: ℕ) (red: ℕ) (blue: ℕ): ℕ := 
  10 * 10 * 10 * yellow + 10 * 10 * red + blue

theorem tom_sawyer_bible (y r b : ℕ) (hc : y = 8 ∧ r = 3 ∧ b = 7):
  blue_tickets_needed 10 0 0 - blue_tickets_needed y r b = 163 :=
by 
  sorry

end tom_sawyer_bible_l2213_221375


namespace carl_insurance_payment_percentage_l2213_221391

variable (property_damage : ℝ) (medical_bills : ℝ) 
          (total_cost : ℝ) (carl_payment : ℝ) (insurance_payment_percentage : ℝ)

theorem carl_insurance_payment_percentage :
  property_damage = 40000 ∧
  medical_bills = 70000 ∧
  total_cost = property_damage + medical_bills ∧
  carl_payment = 22000 ∧
  carl_payment = 0.20 * total_cost →
  insurance_payment_percentage = 100 - 20 :=
by
  sorry

end carl_insurance_payment_percentage_l2213_221391


namespace todd_numbers_sum_eq_l2213_221337

def sum_of_todd_numbers (n : ℕ) : ℕ :=
  sorry -- This would be the implementation of the sum based on provided problem conditions

theorem todd_numbers_sum_eq :
  sum_of_todd_numbers 5000 = 1250025 :=
sorry

end todd_numbers_sum_eq_l2213_221337


namespace new_ratio_milk_to_water_l2213_221396

def total_volume : ℕ := 100
def initial_milk_ratio : ℚ := 3
def initial_water_ratio : ℚ := 2
def additional_water : ℕ := 48

def new_milk_volume := (initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)) * total_volume
def new_water_volume := (initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) * total_volume + additional_water

theorem new_ratio_milk_to_water :
  new_milk_volume / (new_water_volume : ℚ) = 15 / 22 :=
by
  sorry

end new_ratio_milk_to_water_l2213_221396


namespace find_a_l2213_221389

/-- The random variable ξ takes on all possible values 1, 2, 3, 4, 5,
and P(ξ = k) = a * k for k = 1, 2, 3, 4, 5. Given that the sum 
of probabilities for all possible outcomes of a discrete random
variable equals 1, find the value of a. -/
theorem find_a (a : ℝ) 
  (h : (a * 1) + (a * 2) + (a * 3) + (a * 4) + (a * 5) = 1) : 
  a = 1 / 15 :=
sorry

end find_a_l2213_221389


namespace root_of_equation_l2213_221365

theorem root_of_equation (x : ℝ) : 
  169 * (157 - 77 * x)^2 + 100 * (201 - 100 * x)^2 = 26 * (77 * x - 157) * (1000 * x - 2010) ↔ x = 31 := 
by 
  sorry

end root_of_equation_l2213_221365


namespace ratio_of_pens_to_pencils_l2213_221346

/-
The store ordered pens and pencils:
1. The number of pens was some multiple of the number of pencils plus 300.
2. The cost of a pen was $5.
3. The cost of a pencil was $4.
4. The store ordered 15 boxes, each having 80 pencils.
5. The store paid a total of $18,300 for the stationery.
Prove that the ratio of the number of pens to the number of pencils is 2.25.
-/

variables (e p k : ℕ)
variables (cost_pen : ℕ := 5) (cost_pencil : ℕ := 4) (total_cost : ℕ := 18300)

def number_of_pencils := 15 * 80

def number_of_pens := p -- to be defined in terms of e and k

def total_cost_pens := p * cost_pen
def total_cost_pencils := e * cost_pencil

theorem ratio_of_pens_to_pencils :
  p = k * e + 300 →
  e = 1200 →
  5 * p + 4 * e = 18300 →
  (p : ℚ) / e = 2.25 :=
by
  intros hp he htotal
  sorry

end ratio_of_pens_to_pencils_l2213_221346


namespace total_students_is_37_l2213_221323

-- Let b be the number of blue swim caps 
-- Let r be the number of red swim caps
variables (b r : ℕ)

-- The number of blue swim caps according to the male sports commissioner
def condition1 : Prop := b = 4 * r + 1

-- The number of blue swim caps according to the female sports commissioner
def condition2 : Prop := b = r + 24

-- The total number of students in the 3rd grade
def total_students : ℕ := b + r

theorem total_students_is_37 (h1 : condition1 b r) (h2 : condition2 b r) : total_students b r = 37 :=
by sorry

end total_students_is_37_l2213_221323


namespace range_of_m_l2213_221397

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + m / 2 + 2 ≥ 0) ∨ ((1 / 2) * m > 1) ↔ ((m > 4) ∧ ¬(∀ x : ℝ, x^2 + m * x + m / 2 + 2 ≥ 0)) :=
sorry

end range_of_m_l2213_221397


namespace cos_minus_sin_eq_neg_one_fifth_l2213_221338

theorem cos_minus_sin_eq_neg_one_fifth
  (α : ℝ)
  (h1 : Real.sin (2 * α) = 24 / 25)
  (h2 : π < α ∧ α < 5 * π / 4) :
  Real.cos α - Real.sin α = -1 / 5 := sorry

end cos_minus_sin_eq_neg_one_fifth_l2213_221338


namespace pages_copied_for_15_dollars_l2213_221303

theorem pages_copied_for_15_dollars
  (cost_per_page : ℕ)
  (dollar_to_cents : ℕ)
  (dollars_available : ℕ)
  (convert_to_cents : dollar_to_cents = 100)
  (cost_per_page_eq : cost_per_page = 3)
  (dollars_available_eq : dollars_available = 15) :
  (dollars_available * dollar_to_cents) / cost_per_page = 500 := by
  -- Convert the dollar amount to cents
  -- Calculate the number of pages that can be copied
  sorry

end pages_copied_for_15_dollars_l2213_221303


namespace missing_number_in_proportion_l2213_221369

/-- Given the proportion 2 : 5 = x : 3.333333333333333, prove that the missing number x is 1.3333333333333332 -/
theorem missing_number_in_proportion : ∃ x, (2 / 5 = x / 3.333333333333333) ∧ x = 1.3333333333333332 :=
  sorry

end missing_number_in_proportion_l2213_221369


namespace typist_original_salary_l2213_221306

theorem typist_original_salary (S : ℝ) (h : (1.12 * 0.93 * 1.15 * 0.90 * S = 5204.21)) : S = 5504.00 :=
sorry

end typist_original_salary_l2213_221306


namespace inverse_function_log3_l2213_221382

noncomputable def f (x : ℝ) : ℝ := Real.logb 3 x

theorem inverse_function_log3 :
  ∀ x : ℝ, x > 0 →
  ∃ y : ℝ, f (3 ^ y) = y := 
sorry

end inverse_function_log3_l2213_221382


namespace count_positive_integers_l2213_221301

theorem count_positive_integers (n : ℤ) : 
  (130 * n) ^ 50 > (n : ℤ) ^ 100 ∧ (n : ℤ) ^ 100 > 2 ^ 200 → 
  ∃ k : ℕ, k = 125 := sorry

end count_positive_integers_l2213_221301


namespace count_valid_n_l2213_221327

theorem count_valid_n :
  let n_values := [50, 550, 1050, 1550, 2050]
  ( ∀ n : ℤ, (50 * ((n + 500) / 50) - 500 = n) ∧ (Int.floor (Real.sqrt (2 * n : ℝ)) = (n + 500) / 50) → n ∈ n_values ) ∧
  ((∀ n : ℤ, ∃ k : ℤ, (n = 50 * k - 500) ∧ (k = Int.floor (Real.sqrt (2 * (50 * k - 500) : ℝ))) ∧ 0 < n ) → n_values.length = 5) :=
by
  sorry

end count_valid_n_l2213_221327


namespace simplify_expression_l2213_221336

variable (x y : ℝ)

theorem simplify_expression : 
  (3 * x^2 * y - 2 * x * y^2) - (x * y^2 - 2 * x^2 * y) - 2 * (-3 * x^2 * y - x * y^2) = 26 := by
  -- Given conditions
  let x := -1
  let y := 2
  -- Proof to be provided
  sorry

end simplify_expression_l2213_221336


namespace toothpicks_stage_20_l2213_221366

-- Definition of the toothpick sequence
def toothpicks (n : ℕ) : ℕ :=
  if n = 1 then 3
  else 3 + 3 * (n - 1)

-- Theorem statement
theorem toothpicks_stage_20 : toothpicks 20 = 60 := by
  sorry

end toothpicks_stage_20_l2213_221366


namespace triangle_area_l2213_221361

open Real

-- Define the angles A and C, side a, and state the goal as proving the area
theorem triangle_area (A C : ℝ) (a : ℝ) (hA : A = 30 * (π / 180)) (hC : C = 45 * (π / 180)) (ha : a = 2) : 
  (1 / 2) * ((sqrt 6 + sqrt 2) * (2 * sqrt 2) * sin (30 * (π / 180))) = sqrt 3 + 1 := 
by
  sorry

end triangle_area_l2213_221361


namespace min_cubes_required_l2213_221351

theorem min_cubes_required (length width height volume_cube : ℝ) 
  (h_length : length = 14.5) 
  (h_width : width = 17.8) 
  (h_height : height = 7.2) 
  (h_volume_cube : volume_cube = 3) : 
  ⌈(length * width * height) / volume_cube⌉ = 624 := sorry

end min_cubes_required_l2213_221351


namespace solve_for_x_l2213_221386

theorem solve_for_x : ∃ x : ℝ, 64 = 2 * (16 : ℝ)^(x - 2) ∧ x = 3.25 := by
  sorry

end solve_for_x_l2213_221386


namespace inequality_solution_l2213_221374

theorem inequality_solution (x : ℝ) : 
  (x-20) / (x+16) ≤ 0 ↔ -16 < x ∧ x ≤ 20 := by
  sorry

end inequality_solution_l2213_221374


namespace factorial_fraction_is_integer_l2213_221317

open Nat

theorem factorial_fraction_is_integer (m n : ℕ) : 
  ↑((factorial (2 * m)) * (factorial (2 * n))) % (factorial m * factorial n * factorial (m + n)) = 0 := sorry

end factorial_fraction_is_integer_l2213_221317


namespace alice_wins_l2213_221316

noncomputable def game_condition (r : ℝ) (f : ℕ → ℝ) : Prop :=
∀ n, 0 ≤ f n ∧ f n ≤ 1

theorem alice_wins (r : ℝ) (f : ℕ → ℝ) (hf : game_condition r f) :
  r ≤ 3 → (∃ x : ℕ → ℝ, game_condition 3 x ∧ (abs (x 0 - x 1) + abs (x 2 - x 3) + abs (x 4 - x 5) ≥ r)) :=
by
  sorry

end alice_wins_l2213_221316


namespace pythagorean_triple_divisible_by_60_l2213_221300

theorem pythagorean_triple_divisible_by_60 
  (a b c : ℕ) (h : a * a + b * b = c * c) : 60 ∣ (a * b * c) :=
sorry

end pythagorean_triple_divisible_by_60_l2213_221300


namespace largest_possible_s_l2213_221378

theorem largest_possible_s (r s : ℕ) (h1 : 3 ≤ s) (h2 : s ≤ r) (h3 : s < 122)
    (h4 : ∀ r s, (61 * (s - 2) * r = 60 * (r - 2) * s)) : s ≤ 121 :=
by
  sorry

end largest_possible_s_l2213_221378


namespace rahul_work_days_l2213_221384

variable (R : ℕ)

theorem rahul_work_days
  (rajesh_days : ℕ := 2)
  (total_money : ℕ := 355)
  (rahul_share : ℕ := 142)
  (rajesh_share : ℕ := total_money - rahul_share)
  (payment_ratio : ℕ := rahul_share / rajesh_share)
  (work_rate_ratio : ℕ := rajesh_days / R) :
  payment_ratio = work_rate_ratio → R = 3 :=
by
  sorry

end rahul_work_days_l2213_221384


namespace largest_number_l2213_221379

theorem largest_number 
  (A : ℝ) (B : ℝ) (C : ℝ) (D : ℝ) (E : ℝ)
  (hA : A = 0.986)
  (hB : B = 0.9851)
  (hC : C = 0.9869)
  (hD : D = 0.9807)
  (hE : E = 0.9819)
  : C > A ∧ C > B ∧ C > D ∧ C > E :=
by
  sorry

end largest_number_l2213_221379


namespace equal_probability_among_children_l2213_221332

theorem equal_probability_among_children
    (n : ℕ := 100)
    (p : ℝ := 0.232818)
    (k : ℕ := 18)
    (h_pos : 0 < p)
    (h_lt : p < 1)
    (num_outcomes : ℕ := 2^k) :
  ∃ (dist : Fin n → Fin num_outcomes),
    ∀ i : Fin num_outcomes, ∃ j : Fin n, dist j = i ∧ p ^ k * (1 - p) ^ (num_outcomes - k) = 1 / n :=
by
  sorry

end equal_probability_among_children_l2213_221332


namespace percentage_alcohol_in_first_vessel_is_zero_l2213_221393

theorem percentage_alcohol_in_first_vessel_is_zero (x : ℝ) :
  ∀ (alcohol_first_vessel total_vessel_capacity first_vessel_capacity second_vessel_capacity concentration_mixture : ℝ),
  first_vessel_capacity = 2 →
  (∃ xpercent, alcohol_first_vessel = (first_vessel_capacity * xpercent / 100)) →
  second_vessel_capacity = 6 →
  (∃ ypercent, ypercent = 40 ∧ alcohol_first_vessel + 2.4 = concentration_mixture * (total_vessel_capacity/8) * 8) →
  concentration_mixture = 0.3 →
  0 = x := sorry

end percentage_alcohol_in_first_vessel_is_zero_l2213_221393


namespace total_expenditure_is_108_l2213_221371

-- Define the costs of items and quantities purchased by Robert and Teddy
def cost_pizza := 10   -- cost of one box of pizza
def cost_soft_drink := 2  -- cost of one can of soft drink
def cost_hamburger := 3   -- cost of one hamburger

def qty_pizza_robert := 5     -- quantity of pizza boxes by Robert
def qty_soft_drink_robert := 10 -- quantity of soft drinks by Robert

def qty_hamburger_teddy := 6  -- quantity of hamburgers by Teddy
def qty_soft_drink_teddy := 10 -- quantity of soft drinks by Teddy

-- Calculate total expenditure for Robert and Teddy
def total_cost_robert := (qty_pizza_robert * cost_pizza) + (qty_soft_drink_robert * cost_soft_drink)
def total_cost_teddy := (qty_hamburger_teddy * cost_hamburger) + (qty_soft_drink_teddy * cost_soft_drink)

-- Total expenditure in all
def total_expenditure := total_cost_robert + total_cost_teddy

-- We formulate the theorem to prove that the total expenditure is $108
theorem total_expenditure_is_108 : total_expenditure = 108 :=
by 
  -- Placeholder proof
  sorry

end total_expenditure_is_108_l2213_221371


namespace five_by_five_rectangles_l2213_221381

def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem five_by_five_rectangles : (choose 5 2) * (choose 5 2) = 100 :=
by
  sorry

end five_by_five_rectangles_l2213_221381


namespace purely_imaginary_necessary_not_sufficient_l2213_221392

-- Definition of a purely imaginary number
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem purely_imaginary_necessary_not_sufficient (a b : ℝ) :
  a = 0 → (z : ℂ) = ⟨a, b⟩ → is_purely_imaginary z ↔ (a = 0 ∧ b ≠ 0) :=
by
  sorry

end purely_imaginary_necessary_not_sufficient_l2213_221392


namespace matrix_vector_multiplication_correct_l2213_221345

noncomputable def mat : Matrix (Fin 2) (Fin 2) ℤ := ![![3, -2], ![1, 5]]
noncomputable def vec : Fin 2 → ℤ := ![-1, 2]
noncomputable def result : Fin 2 → ℤ := ![-7, 9]

theorem matrix_vector_multiplication_correct :
  (Matrix.mulVec mat vec) = result :=
by
  sorry

end matrix_vector_multiplication_correct_l2213_221345


namespace video_time_per_week_l2213_221394

-- Define the basic conditions
def short_video_length : ℕ := 2
def multiplier : ℕ := 6
def long_video_length : ℕ := multiplier * short_video_length
def short_videos_per_day : ℕ := 2
def long_videos_per_day : ℕ := 1
def days_in_week : ℕ := 7

-- Calculate daily and weekly video release time
def daily_video_time : ℕ := (short_videos_per_day * short_video_length) + (long_videos_per_day * long_video_length)
def weekly_video_time : ℕ := daily_video_time * days_in_week

-- Main theorem to prove
theorem video_time_per_week : weekly_video_time = 112 := by
    sorry

end video_time_per_week_l2213_221394


namespace sales_tax_difference_l2213_221350

noncomputable def price_before_tax : ℝ := 50
noncomputable def sales_tax_rate_7_5_percent : ℝ := 0.075
noncomputable def sales_tax_rate_8_percent : ℝ := 0.08

theorem sales_tax_difference :
  (price_before_tax * sales_tax_rate_8_percent) - (price_before_tax * sales_tax_rate_7_5_percent) = 0.25 :=
by
  sorry

end sales_tax_difference_l2213_221350


namespace number_of_girls_l2213_221368

-- Definitions from the problem conditions
def ratio_girls_boys (g b : ℕ) : Prop := 4 * b = 3 * g
def total_students (g b : ℕ) : Prop := g + b = 56

-- The proof statement
theorem number_of_girls (g b k : ℕ) (hg : 4 * k = g) (hb : 3 * k = b) (hr : ratio_girls_boys g b) (ht : total_students g b) : g = 32 :=
by sorry

end number_of_girls_l2213_221368


namespace correct_answer_l2213_221360

-- Define the sentence structure and the requirement for a formal object
structure SentenceStructure where
  subject : String := "I"
  verb : String := "like"
  object_placeholder : String := "_"
  clause : String := "when the weather is clear and bright"

-- Correct choices provided
inductive Choice
  | this
  | that
  | it
  | one

-- Problem formulation: Based on SentenceStructure, prove that 'it' is the correct choice
theorem correct_answer {S : SentenceStructure} : Choice.it = Choice.it :=
by
  -- Proof omitted
  sorry

end correct_answer_l2213_221360


namespace right_triangle_hypotenuse_segment_ratio_l2213_221326

theorem right_triangle_hypotenuse_segment_ratio
  (x : ℝ) (AB BC AC BD AD CD : ℝ)
  (h1 : AB = 4 * x) 
  (h2 : BC = 3 * x) 
  (h3 : AC = 5 * x) 
  (h4 : (BD ^ 2) = AD * CD) :
  (CD / AD) = (16 / 9) :=
by
  sorry

end right_triangle_hypotenuse_segment_ratio_l2213_221326


namespace sum_cubes_mod_l2213_221320

theorem sum_cubes_mod (n : ℕ) : (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 + 7^3 + 8^3 + 9^3 + 10^3) % 7 = 1 := by
  sorry

end sum_cubes_mod_l2213_221320


namespace division_of_fractions_l2213_221372

theorem division_of_fractions :
  (5 : ℚ) / 6 / ((2 : ℚ) / 3) = (5 : ℚ) / 4 :=
by
  sorry

end division_of_fractions_l2213_221372


namespace probability_divisible_by_three_l2213_221341

noncomputable def prob_divisible_by_three : ℚ :=
  1 - (4/6)^6

theorem probability_divisible_by_three :
  prob_divisible_by_three = 665 / 729 :=
by
  sorry

end probability_divisible_by_three_l2213_221341


namespace sunflower_height_A_l2213_221348

-- Define the height of sunflowers from Packet B
def height_B : ℝ := 160

-- Define that Packet A sunflowers are 20% taller than Packet B sunflowers
def height_A : ℝ := 1.2 * height_B

-- State the theorem to show that height_A equals 192 inches
theorem sunflower_height_A : height_A = 192 := by
  sorry

end sunflower_height_A_l2213_221348


namespace stack_of_logs_total_l2213_221354

-- Define the given conditions as variables and constants in Lean
def bottom_row : Nat := 15
def top_row : Nat := 4
def rows : Nat := bottom_row - top_row + 1
def sum_arithmetic_series (a l n : Nat) : Nat := n * (a + l) / 2

-- Define the main theorem to prove
theorem stack_of_logs_total : sum_arithmetic_series top_row bottom_row rows = 114 :=
by
  -- Here you will normally provide the proof
  sorry

end stack_of_logs_total_l2213_221354


namespace distance_between_A_and_B_l2213_221305

theorem distance_between_A_and_B 
  (d : ℕ) -- The distance we want to prove
  (ha : ∀ (t : ℕ), d = 700 * t)
  (hb : ∀ (t : ℕ), d + 400 = 2100 * t) :
  d = 1700 := 
by
  sorry

end distance_between_A_and_B_l2213_221305


namespace P_has_common_root_l2213_221312

def P (x : ℝ) (p : ℝ) (q : ℝ) : ℝ := x^2 + p * x + q

theorem P_has_common_root (p q : ℝ) (t : ℝ) (h : P t p q = 0) :
  P 0 p q * P 1 p q = 0 :=
by
  sorry

end P_has_common_root_l2213_221312


namespace factory_produces_11250_products_l2213_221342

noncomputable def total_products (refrigerators_per_hour coolers_per_hour hours_per_day days : ℕ) : ℕ :=
  (refrigerators_per_hour + coolers_per_hour) * (hours_per_day * days)

theorem factory_produces_11250_products :
  total_products 90 (90 + 70) 9 5 = 11250 := by
  sorry

end factory_produces_11250_products_l2213_221342


namespace difference_max_min_y_l2213_221377

-- Define initial and final percentages of responses
def initial_yes : ℝ := 0.30
def initial_no : ℝ := 0.70
def final_yes : ℝ := 0.60
def final_no : ℝ := 0.40

-- Define the problem statement
theorem difference_max_min_y : 
  ∃ y_min y_max : ℝ, (initial_yes + initial_no = 1) ∧ (final_yes + final_no = 1) ∧
  (initial_yes + initial_no = final_yes + final_no) ∧ y_min ≤ y_max ∧ 
  y_max - y_min = 0.30 :=
sorry

end difference_max_min_y_l2213_221377


namespace a_mul_b_value_l2213_221344

theorem a_mul_b_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a + b = 15) (h₃ : a * b = 36) : 
  (a * b = (1/a : ℚ) + (1/b : ℚ)) ∧ (a * b = 15/36) ∧ (15 / 36 = 5 / 12) :=
by
  sorry

end a_mul_b_value_l2213_221344


namespace total_fruit_pieces_correct_l2213_221380

/-
  Define the quantities of each type of fruit.
-/
def red_apples : Nat := 9
def green_apples : Nat := 4
def purple_grapes : Nat := 3
def yellow_bananas : Nat := 6
def orange_oranges : Nat := 2

/-
  The total number of fruit pieces in the basket.
-/
def total_fruit_pieces : Nat := red_apples + green_apples + purple_grapes + yellow_bananas + orange_oranges

/-
  Prove that the total number of fruit pieces is 24.
-/
theorem total_fruit_pieces_correct : total_fruit_pieces = 24 := by
  sorry

end total_fruit_pieces_correct_l2213_221380


namespace ratio_of_shaded_area_l2213_221335

-- Definitions
variable (S : Type) [Field S]
variable (square_area shaded_area : S) -- Areas of the square and the shaded regions.
variable (PX XQ : S) -- Lengths such that PX = 3 * XQ.

-- Conditions
axiom condition1 : PX = 3 * XQ
axiom condition2 : shaded_area / square_area = 0.375

-- Goal
theorem ratio_of_shaded_area (PX XQ square_area shaded_area : S) [Field S] 
  (condition1 : PX = 3 * XQ)
  (condition2 : shaded_area / square_area = 0.375) : shaded_area / square_area = 0.375 := 
  by
  sorry

end ratio_of_shaded_area_l2213_221335


namespace min_xy_when_a_16_min_expr_when_a_0_l2213_221363

-- Problem 1: Minimum value of xy when a = 16
theorem min_xy_when_a_16 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x * y = x + 4 * y + 16) : 16 ≤ x * y :=
    sorry

-- Problem 2: Minimum value of x + y + 2 / x + 1 / (2 * y) when a = 0
theorem min_expr_when_a_0 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x * y = x + 4 * y) : (11 : ℝ) / 2 ≤ x + y + 2 / x + 1 / (2 * y) :=
    sorry

end min_xy_when_a_16_min_expr_when_a_0_l2213_221363


namespace relationship_between_Q_and_t_remaining_power_after_5_hours_distance_with_40_power_l2213_221322

-- Define the relationship between Q and t
def remaining_power (t : ℕ) : ℕ := 80 - 15 * t

-- Question 1: Prove relationship between Q and t
theorem relationship_between_Q_and_t : ∀ t : ℕ, remaining_power t = 80 - 15 * t :=
by sorry

-- Question 2: Prove remaining power after 5 hours
theorem remaining_power_after_5_hours : remaining_power 5 = 5 :=
by sorry

-- Question 3: Prove distance the car can travel with 40 kW·h remaining power
theorem distance_with_40_power 
  (remaining_power : ℕ := (80 - 15 * t)) 
  (t := 8 / 3)
  (speed : ℕ := 90) : (90 * (8 / 3)) = 240 :=
by sorry

end relationship_between_Q_and_t_remaining_power_after_5_hours_distance_with_40_power_l2213_221322


namespace remainder_when_multiplied_and_divided_l2213_221387

theorem remainder_when_multiplied_and_divided (n k : ℤ) (h : n % 28 = 15) : (2 * n) % 14 = 2 := 
by
  sorry

end remainder_when_multiplied_and_divided_l2213_221387


namespace minimum_trucks_on_lot_l2213_221390

variable (total_trucks : ℕ) (rented_percentage : ℝ) (max_rented_trucks : ℕ)
variable (initial_trucks : total_trucks = 24) (returned_percentage : rented_percentage = 0.5) (max_trucks_rented : max_rented_trucks = 24)

theorem minimum_trucks_on_lot (total_trucks : ℕ) (rented_percentage : ℝ) (max_rented_trucks : ℕ)
  (initial_trucks : total_trucks = 24) (returned_percentage : rented_percentage = 0.5) (max_trucks_rented : max_rented_trucks = 24) :
  max_rented_trucks / 2 = 12 :=
by sorry

end minimum_trucks_on_lot_l2213_221390


namespace Diana_friends_count_l2213_221308

theorem Diana_friends_count (totalErasers : ℕ) (erasersPerFriend : ℕ) 
  (h1: totalErasers = 3840) (h2: erasersPerFriend = 80) : 
  totalErasers / erasersPerFriend = 48 := 
by 
  sorry

end Diana_friends_count_l2213_221308


namespace relationship_abc_l2213_221315

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.exp (-Real.pi)
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relationship_abc : b < a ∧ a < c :=
by
  -- proofs would be added here
  sorry

end relationship_abc_l2213_221315


namespace solve_for_y_l2213_221313

theorem solve_for_y (x y : ℝ) (h : 4 * x - y = 3) : y = 4 * x - 3 :=
by sorry

end solve_for_y_l2213_221313


namespace student_passing_percentage_l2213_221358

def student_marks : ℕ := 80
def shortfall_marks : ℕ := 100
def total_marks : ℕ := 600

def passing_percentage (student_marks shortfall_marks total_marks : ℕ) : ℕ :=
  (student_marks + shortfall_marks) * 100 / total_marks

theorem student_passing_percentage :
  passing_percentage student_marks shortfall_marks total_marks = 30 :=
by
  sorry

end student_passing_percentage_l2213_221358


namespace person_y_speed_in_still_water_l2213_221370

theorem person_y_speed_in_still_water 
    (speed_x_in_still_water : ℝ)
    (time_meeting_towards_each_other : ℝ)
    (time_catching_up_same_direction: ℝ)
    (distance_upstream_meeting: ℝ)
    (distance_downstream_meeting: ℝ)
    (total_distance: ℝ) :
    speed_x_in_still_water = 6 →
    time_meeting_towards_each_other = 4 →
    time_catching_up_same_direction = 16 →
    distance_upstream_meeting = 4 * (6 - distance_upstream_meeting) + 4 * (10 + distance_downstream_meeting) →
    distance_downstream_meeting = 4 * (6 + distance_upstream_meeting) →
    total_distance = 4 * (6 + 10) →
    ∃ (speed_y_in_still_water : ℝ), speed_y_in_still_water = 10 :=
by
  intros h_speed_x h_time_meeting h_time_catching h_distance_upstream h_distance_downstream h_total_distance
  sorry

end person_y_speed_in_still_water_l2213_221370


namespace alicia_stickers_l2213_221314

theorem alicia_stickers :
  ∃ S : ℕ, S > 2 ∧
  (S % 5 = 2) ∧ (S % 11 = 2) ∧ (S % 13 = 2) ∧
  S = 717 :=
sorry

end alicia_stickers_l2213_221314


namespace sum_in_range_l2213_221324

open Real

def mix1 := 3 + 3/8
def mix2 := 4 + 2/5
def mix3 := 6 + 1/11
def mixed_sum := mix1 + mix2 + mix3

theorem sum_in_range : mixed_sum > 13 ∧ mixed_sum < 14 :=
by
  -- Since we are just providing the statement, we leave the proof as a placeholder.
  sorry

end sum_in_range_l2213_221324


namespace conditional_probability_second_sci_given_first_sci_l2213_221340

-- Definitions based on the conditions
def total_questions : ℕ := 6
def science_questions : ℕ := 4
def humanities_questions : ℕ := 2
def first_draw_is_science : Prop := true

-- The statement we want to prove
theorem conditional_probability_second_sci_given_first_sci : 
    first_draw_is_science → (science_questions - 1) / (total_questions - 1) = 3 / 5 := 
by
  intro h
  have num_sci_after_first : ℕ := science_questions - 1
  have total_after_first : ℕ := total_questions - 1
  have prob_second_sci := num_sci_after_first / total_after_first
  sorry

end conditional_probability_second_sci_given_first_sci_l2213_221340


namespace int_solutions_l2213_221331

theorem int_solutions (a b : ℤ) (h : a^2 + b = b^2022) : (a, b) = (0, 0) ∨ (a, b) = (0, 1) :=
by {
  sorry
}

end int_solutions_l2213_221331


namespace notebook_cost_l2213_221325

theorem notebook_cost (total_spent ruler_cost pencil_count pencil_cost: ℕ)
  (h1 : total_spent = 74)
  (h2 : ruler_cost = 18)
  (h3 : pencil_count = 3)
  (h4 : pencil_cost = 7) :
  total_spent - (ruler_cost + pencil_count * pencil_cost) = 35 := 
by 
  sorry

end notebook_cost_l2213_221325
