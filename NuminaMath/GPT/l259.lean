import Mathlib

namespace SUCCESS_arrangement_count_l259_259142

theorem SUCCESS_arrangement_count : 
  let n := 7
  let n_S := 3
  let n_C := 2
  let n_U := 1
  let n_E := 1
  ∃ (ways_to_arrange : ℕ), ways_to_arrange = Nat.factorial n / (Nat.factorial n_S * Nat.factorial n_C) := 420 :=
by
  -- Problem Conditions
  let n := 7
  let n_S := 3
  let n_C := 2
  let n_U := 1
  let n_E := 1
  -- The Proof
  existsi 420
  sorry

end SUCCESS_arrangement_count_l259_259142


namespace percent_increase_l259_259296

def initial_price : ℝ := 15
def final_price : ℝ := 16

theorem percent_increase : ((final_price - initial_price) / initial_price) * 100 = 6.67 :=
by
  sorry

end percent_increase_l259_259296


namespace max_k_value_l259_259196

open Finset

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

theorem max_k_value (k : ℕ) (A : Fin k (Finset ℕ)) :
  (∀ i : Fin k, A[i] ⊆ S) →
  (∀ i : Fin k, A[i].card = 5) →
  (∀ i j : Fin k, i ≠ j → (A[i] ∩ A[j]).card ≤ 2) →
  k ≤ 6 :=
sorry -- Proof goes here

-- The statement verifies that for any collection of sets A satisfying the above conditions,
-- the number of such sets k cannot exceed 6.

end max_k_value_l259_259196


namespace females_with_advanced_degrees_l259_259870

theorem females_with_advanced_degrees 
  (total_employees : ℕ) 
  (total_females : ℕ) 
  (total_advanced_degrees : ℕ) 
  (total_college_degrees : ℕ) 
  (males_with_college_degree : ℕ) 
  (h1 : total_employees = 180) 
  (h2 : total_females = 110) 
  (h3 : total_advanced_degrees = 90) 
  (h4 : total_college_degrees = 90) 
  (h5 : males_with_college_degree = 35) : 
  ∃ (females_with_advanced_degrees : ℕ), females_with_advanced_degrees = 55 := 
by {
  sorry
}

end females_with_advanced_degrees_l259_259870


namespace Chris_has_6_Teslas_l259_259309

theorem Chris_has_6_Teslas (x y z : ℕ) (h1 : z = 13) (h2 : z = x + 10) (h3 : x = y / 2):
  y = 6 :=
by
  sorry

end Chris_has_6_Teslas_l259_259309


namespace three_digit_not_multiple_of_3_5_7_l259_259007

theorem three_digit_not_multiple_of_3_5_7 : 
  (900 - (let count_mult_3 := 300 in
           let count_mult_5 := 180 in
           let count_mult_7 := 128 in
           let count_mult_15 := 60 in
           let count_mult_21 := 43 in
           let count_mult_35 := 26 in
           let count_mult_105 := 9 in
           let total_mult_3_5_or_7 := 
             count_mult_3 + count_mult_5 + count_mult_7 - 
             (count_mult_15 + count_mult_21 + count_mult_35) +
             count_mult_105 in
           total_mult_3_5_or_7)) = 412 :=
by {
  -- The mathematical calculations were performed above
  -- The proof is represented by 'sorry' indicating the solution is skipped
  sorry
}

end three_digit_not_multiple_of_3_5_7_l259_259007


namespace number_of_students_without_A_l259_259570

theorem number_of_students_without_A (total_students : ℕ) (A_chemistry : ℕ) (A_physics : ℕ) (A_both : ℕ) (h1 : total_students = 40)
    (h2 : A_chemistry = 10) (h3 : A_physics = 18) (h4 : A_both = 5) :
    total_students - (A_chemistry + A_physics - A_both) = 17 :=
by {
  sorry
}

end number_of_students_without_A_l259_259570


namespace minimum_kinds_of_candies_l259_259865

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
    It turned out that between any two candies of the same kind, there is an even number of candies. 
    What is the minimum number of kinds of candies that could be? -/
theorem minimum_kinds_of_candies (n : ℕ) (h : 91 < 2 * n) : n ≥ 46 :=
sorry

end minimum_kinds_of_candies_l259_259865


namespace exactly_one_true_l259_259356

-- Given conditions
def p (x : ℝ) : Prop := (x^2 - 3 * x + 2 ≠ 0) → (x ≠ 2)

-- Define the contrapositive of p
def contrapositive_p (x : ℝ) : Prop := (x = 2) → (x^2 - 3 * x + 2 = 0)

-- Define the converse of p
def converse_p (x : ℝ) : Prop := (x ≠ 2) → (x^2 - 3 * x + 2 ≠ 0)

-- Define the inverse of p
def inverse_p (x : ℝ) : Prop := (x = 2 → x^2 - 3 * x + 2 = 0)

-- Formalize the problem: Prove that exactly one of the converse, inverse, and contrapositive of p is true.
theorem exactly_one_true :
  (∀ x : ℝ, p x) →
  ((∃ x : ℝ, contrapositive_p x) ∧ ¬(∀ x : ℝ, converse_p x) ∧ ¬(∀ x : ℝ, inverse_p x) ∨
   ¬(∃ x : ℝ, contrapositive_p x) ∧ (∀ x : ℝ, converse_p x) ∧ ¬(∀ x : ℝ, inverse_p x) ∨
   ¬(∃ x : ℝ, contrapositive_p x) ∧ ¬(∀ x : ℝ, converse_p x) ∧ (∀ x : ℝ, inverse_p x)) :=
sorry

end exactly_one_true_l259_259356


namespace chrysler_building_floors_l259_259083

theorem chrysler_building_floors (L : ℕ) (Chrysler : ℕ) (h1 : Chrysler = L + 11)
  (h2 : L + Chrysler = 35) : Chrysler = 23 :=
by {
  --- proof would be here
  sorry
}

end chrysler_building_floors_l259_259083


namespace find_a_b_and_water_usage_l259_259755

noncomputable def water_usage_april (a : ℝ) :=
  (15 * (a + 0.8) = 45)

noncomputable def water_usage_may (a b : ℝ) :=
  (17 * (a + 0.8) + 8 * (b + 0.8) = 91)

noncomputable def water_usage_june (a b x : ℝ) :=
  (17 * (a + 0.8) + 13 * (b + 0.8) + (x - 30) * 6.8 = 150)

theorem find_a_b_and_water_usage :
  ∃ (a b x : ℝ), water_usage_april a ∧ water_usage_may a b ∧ water_usage_june a b x ∧ a = 2.2 ∧ b = 4.2 ∧ x = 35 :=
by {
  sorry
}

end find_a_b_and_water_usage_l259_259755


namespace sum_of_roots_equation_l259_259746

noncomputable def sum_of_roots (a b c : ℝ) : ℝ :=
  (-b) / a

theorem sum_of_roots_equation :
  let a := 3
  let b := -15
  let c := 20
  sum_of_roots a b c = 5 := 
  by {
    sorry
  }

end sum_of_roots_equation_l259_259746


namespace find_fraction_l259_259340

theorem find_fraction
  (a₁ a₂ b₁ b₂ c₁ c₂ x y : ℚ)
  (h₁ : a₁ = 3) (h₂ : a₂ = 7) (h₃ : b₁ = 6) (h₄ : b₂ = 5)
  (h₅ : c₁ = 1) (h₆ : c₂ = 7)
  (h : (a₁ / a₂) / (b₁ / b₂) = (c₁ / c₂) / (x / y)) :
  (x / y) = 2 / 5 := 
by
  sorry

end find_fraction_l259_259340


namespace flower_total_l259_259753

theorem flower_total (H C D : ℕ) (h1 : H = 34) (h2 : H = C - 13) (h3 : C = D + 23) : 
  H + C + D = 105 :=
by 
  sorry  -- Placeholder for the proof

end flower_total_l259_259753


namespace train_start_time_l259_259619

theorem train_start_time (D PQ : ℝ) (S₁ S₂ : ℝ) (T₁ T₂ meet : ℝ) :
  PQ = 110  -- Distance between stations P and Q
  ∧ S₁ = 20  -- Speed of the first train
  ∧ S₂ = 25  -- Speed of the second train
  ∧ T₂ = 8  -- Start time of the second train
  ∧ meet = 10 -- Meeting time
  ∧ T₁ + T₂ = meet → -- Meeting time condition
  T₁ = 7.5 := -- Answer: first train start time
by
sorry

end train_start_time_l259_259619


namespace blueberry_basket_count_l259_259093

noncomputable def number_of_blueberry_baskets 
    (plums_in_basket : ℕ) 
    (plum_baskets : ℕ) 
    (blueberries_in_basket : ℕ) 
    (total_fruits : ℕ) : ℕ := 
  let total_plums := plum_baskets * plums_in_basket
  let total_blueberries := total_fruits - total_plums
  total_blueberries / blueberries_in_basket

theorem blueberry_basket_count
  (plums_in_basket : ℕ) 
  (plum_baskets : ℕ) 
  (blueberries_in_basket : ℕ) 
  (total_fruits : ℕ)
  (h1 : plums_in_basket = 46)
  (h2 : plum_baskets = 19)
  (h3 : blueberries_in_basket = 170)
  (h4 : total_fruits = 1894) : 
  number_of_blueberry_baskets plums_in_basket plum_baskets blueberries_in_basket total_fruits = 6 := by
  sorry

end blueberry_basket_count_l259_259093


namespace nat_numbers_in_segment_l259_259740

theorem nat_numbers_in_segment (a : ℕ → ℕ) (blue_index red_index : Set ℕ)
  (cond1 : ∀ i ∈ blue_index, i ≤ 200 → a (i - 1) = i)
  (cond2 : ∀ i ∈ red_index, i ≤ 200 → a (i - 1) = 201 - i) :
    ∀ i, 1 ≤ i ∧ i ≤ 100 → ∃ j, j < 100 ∧ a j = i := 
by
  sorry

end nat_numbers_in_segment_l259_259740


namespace red_balls_probability_l259_259575

/-- Let Bag A have 3 white balls and 4 red balls, and Bag B have 1 white ball and 2 red balls.
    One ball is randomly taken from Bag A and put into Bag B, and then two balls are 
    randomly taken from Bag B. Prove that the probability that both balls taken out are 
    red is 5/14. -/
theorem red_balls_probability :
  let bagA_white := 3
  let bagA_red := 4
  let bagB_white := 1
  let bagB_red := 2
  let total_A := bagA_white + bagA_red
  let P_A := bagA_white / total_A
  let P_notA := bagA_red / total_A
  let P_B_given_A := (bagB_red.choose 2) / ((bagB_white + 1 + bagB_red).choose 2)
  let P_B_given_notA := ((bagB_red + 1).choose 2) / ((bagB_white + bagB_red + 1).choose 2)
  let P_B := P_A * P_B_given_A + P_notA * P_B_given_notA
  in P_B = 5 / 14 := sorry

end red_balls_probability_l259_259575


namespace angle_bisector_length_of_B_l259_259577

noncomputable def angle_of_triangle : Type := Real

constant A C : angle_of_triangle
constant AC AB : Real
constant bisector_length_of_angle_B : Real

axiom h₁ : A = 20
axiom h₂ : C = 40
axiom h₃ : AC - AB = 5

theorem angle_bisector_length_of_B (A C : angle_of_triangle) (AC AB bisector_length_of_angle_B : Real)
    (h₁ : A = 20) (h₂ : C = 40) (h₃ : AC - AB = 5) :
    bisector_length_of_angle_B = 5 := 
sorry

end angle_bisector_length_of_B_l259_259577


namespace find_pairs_l259_259505

-- Define a function that checks if a pair (n, d) satisfies the required conditions
def satisfies_conditions (n d : ℕ) : Prop :=
  ∀ S : ℤ, ∃! (a : ℕ → ℤ), 
    (∀ i : ℕ, i < n → a i ≤ a (i + 1)) ∧                -- Non-decreasing sequence condition
    ((Finset.range n).sum a = S) ∧                  -- Sum of the sequence equals S
    (a n.succ.pred - a 0 = d)                      -- The difference condition

-- The formal statement of the required proof
theorem find_pairs :
  {p : ℕ × ℕ | satisfies_conditions p.fst p.snd} = {(1, 0), (3, 2)} :=
by
  sorry

end find_pairs_l259_259505


namespace andy_l259_259289

theorem andy's_profit_per_cake :
  (∀ (cakes : ℕ), cakes = 2 → ∀ (ingredient_cost : ℕ), ingredient_cost = 12 →
                  ∀ (packaging_cost_per_cake : ℕ), packaging_cost_per_cake = 1 →
                  ∀ (selling_price_per_cake : ℕ), selling_price_per_cake = 15 →
                  ∀ (profit_per_cake : ℕ), profit_per_cake = selling_price_per_cake - (ingredient_cost / cakes + packaging_cost_per_cake) →
                    profit_per_cake = 8) :=
by
  sorry

end andy_l259_259289


namespace problem_statement_l259_259901

-- Define C and D as specified in the problem conditions.
def C : ℕ := 4500
def D : ℕ := 3000

-- The final statement of the problem to prove C + D = 7500.
theorem problem_statement : C + D = 7500 := by
  -- This proof can be completed by checking arithmetic.
  sorry

end problem_statement_l259_259901


namespace find_certain_number_l259_259778

theorem find_certain_number (x : ℕ) (h : 220025 = (x + 445) * (2 * (x - 445)) + 25) : x = 555 :=
sorry

end find_certain_number_l259_259778


namespace pages_to_read_tomorrow_l259_259893

-- Define the problem setup
def total_pages : ℕ := 100
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5

-- Define the total pages read after two days
def pages_read_in_two_days : ℕ := pages_yesterday + pages_today

-- Define the number of pages left to read
def pages_left_to_read (total_pages read_so_far : ℕ) : ℕ := total_pages - read_so_far

-- Prove that the number of pages to read tomorrow is 35
theorem pages_to_read_tomorrow :
  pages_left_to_read total_pages pages_read_in_two_days = 35 :=
by
  -- Proof is omitted
  sorry

end pages_to_read_tomorrow_l259_259893


namespace profit_percentage_is_30_percent_l259_259469

theorem profit_percentage_is_30_percent (CP SP : ℕ) (h1 : CP = 280) (h2 : SP = 364) :
  ((SP - CP : ℤ) / (CP : ℤ) : ℚ) * 100 = 30 :=
by sorry

end profit_percentage_is_30_percent_l259_259469


namespace election_winner_votes_l259_259943

variable (V : ℝ) (winner_votes : ℝ) (winner_margin : ℝ)
variable (condition1 : V > 0)
variable (condition2 : winner_votes = 0.60 * V)
variable (condition3 : winner_margin = 240)

theorem election_winner_votes (h : winner_votes - 0.40 * V = winner_margin) : winner_votes = 720 := by
  sorry

end election_winner_votes_l259_259943


namespace compare_abc_l259_259055

def a : ℝ := 2 * Real.log 1.01
def b : ℝ := Real.log 1.02
def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b :=
by
  /- Proof goes here -/
  sorry

end compare_abc_l259_259055


namespace units_digit_of_fraction_example_l259_259440

def units_digit_of_fraction (n d : ℕ) : ℕ :=
  (n / d) % 10

theorem units_digit_of_fraction_example :
  units_digit_of_fraction (25 * 26 * 27 * 28 * 29 * 30) 1250 = 2 := by
  sorry

end units_digit_of_fraction_example_l259_259440


namespace find_f_value_l259_259690

-- Condition 1: f is an odd function
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Condition 2: f has a period of 5
def is_periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x

-- Condition 3: f(-3) = -4
def f_value_at_neg3 (f : ℝ → ℝ) := f (-3) = -4

-- Condition 4: cos(α) = 1 / 2
def cos_alpha_value (α : ℝ) := Real.cos α = 1 / 2

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def α : ℝ := sorry

theorem find_f_value (h_odd : is_odd_function f)
                     (h_periodic : is_periodic f 5)
                     (h_f_neg3 : f_value_at_neg3 f)
                     (h_cos_alpha : cos_alpha_value α) :
  f (4 * Real.cos (2 * α)) = 4 := 
sorry

end find_f_value_l259_259690


namespace side_length_square_l259_259084

theorem side_length_square (A : ℝ) (s : ℝ) (h1 : A = 30) (h2 : A = s^2) : 5 < s ∧ s < 6 :=
by
  -- the proof would go here
  sorry

end side_length_square_l259_259084


namespace k_value_opposite_solutions_l259_259243

theorem k_value_opposite_solutions (k x1 x2 : ℝ) 
  (h1 : 3 * (2 * x1 - 1) = 1 - 2 * x1)
  (h2 : 8 - k = 2 * (x2 + 1))
  (opposite : x2 = -x1) :
  k = 7 :=
by sorry

end k_value_opposite_solutions_l259_259243


namespace jimmy_income_l259_259072

theorem jimmy_income (r_income : ℕ) (r_increase : ℕ) (combined_percent : ℚ) (j_income : ℕ) : 
  r_income = 15000 → 
  r_increase = 7000 → 
  combined_percent = 0.55 → 
  (combined_percent * (r_income + r_increase + j_income) = r_income + r_increase) → 
  j_income = 18000 := 
by
  intros h1 h2 h3 h4
  sorry

end jimmy_income_l259_259072


namespace ending_number_is_54_l259_259617

def first_even_after_15 : ℕ := 16
def evens_between (a b : ℕ) : ℕ := (b - first_even_after_15) / 2 + 1

theorem ending_number_is_54 (n : ℕ) (h : evens_between 15 n = 20) : n = 54 :=
by {
  sorry
}

end ending_number_is_54_l259_259617


namespace find_x_l259_259014

theorem find_x (x : ℕ) (h : 2^10 = 32^x) (h32 : 32 = 2^5) : x = 2 :=
sorry

end find_x_l259_259014


namespace negative_solution_iff_sum_zero_l259_259317

theorem negative_solution_iff_sum_zero (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔
  a + b + c = 0 :=
by
  sorry

end negative_solution_iff_sum_zero_l259_259317


namespace largest_integer_n_l259_259306

-- Define the condition for existence of positive integers x, y, z that satisfy the given equation
def condition (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10

-- State that the largest such integer n is 4
theorem largest_integer_n : ∀ (n : ℕ), condition n → n ≤ 4 :=
by {
  sorry
}

end largest_integer_n_l259_259306


namespace compare_a_b_c_l259_259062

noncomputable def a := 2 * Real.log 1.01
noncomputable def b := Real.log 1.02
noncomputable def c := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l259_259062


namespace function_is_one_l259_259250

noncomputable def f : ℝ → ℝ := sorry

theorem function_is_one (f : ℝ → ℝ)
  (h : ∀ x y z : ℝ, f (x*y) + f (x*z) ≥ 1 + f (x) * f (y*z))
  : ∀ x : ℝ, f x = 1 :=
sorry

end function_is_one_l259_259250


namespace Jayden_less_Coraline_l259_259596

variables (M J : ℕ)
def Coraline_number := 80
def total_sum := 180

theorem Jayden_less_Coraline
  (h1 : M = J + 20)
  (h2 : J < Coraline_number)
  (h3 : M + J + Coraline_number = total_sum) :
  Coraline_number - J = 40 := by
  sorry

end Jayden_less_Coraline_l259_259596


namespace jack_flyers_count_l259_259713

-- Definitions based on the given conditions
def total_flyers : ℕ := 1236
def rose_flyers : ℕ := 320
def flyers_left : ℕ := 796

-- Statement to prove
theorem jack_flyers_count : total_flyers - (rose_flyers + flyers_left) = 120 := by
  sorry

end jack_flyers_count_l259_259713


namespace perpendicular_vectors_l259_259235

theorem perpendicular_vectors (b : ℚ) : 
  (4 * b - 15 = 0) → (b = 15 / 4) :=
by
  intro h
  sorry

end perpendicular_vectors_l259_259235


namespace part1_coordinates_on_x_axis_part2_coordinates_parallel_y_axis_part3_distances_equal_second_quadrant_l259_259688

-- Part (1)
theorem part1_coordinates_on_x_axis (a : ℝ) (h : a + 5 = 0) : (2*a - 2, a + 5) = (-12, 0) :=
by sorry

-- Part (2)
theorem part2_coordinates_parallel_y_axis (a : ℝ) (h : 2*a - 2 = 4) : (2*a - 2, a + 5) = (4, 8) :=
by sorry

-- Part (3)
theorem part3_distances_equal_second_quadrant (a : ℝ) 
  (h1 : 2*a-2 < 0) (h2 : a+5 > 0) (h3 : abs (2*a - 2) = abs (a + 5)) : a^(2022 : ℕ) + 2022 = 2023 :=
by sorry

end part1_coordinates_on_x_axis_part2_coordinates_parallel_y_axis_part3_distances_equal_second_quadrant_l259_259688


namespace find_value_of_abs_2_add_z_l259_259827

theorem find_value_of_abs_2_add_z (z : ℂ) (i : ℂ)
  (hi : i^2 = -1)
  (hz : (1 - z) / (1 + z) = (-1 - i) ) :
  |2 + z| = (5 * real.sqrt 2) / 2 :=
by sorry

end find_value_of_abs_2_add_z_l259_259827


namespace point_in_first_quadrant_l259_259692

theorem point_in_first_quadrant (m : ℝ) (h : m < 0) : 
  (-m > 0) ∧ (-m + 1 > 0) :=
by 
  sorry

end point_in_first_quadrant_l259_259692


namespace bisector_length_of_angle_B_l259_259580

theorem bisector_length_of_angle_B 
  (A B C : Type) 
  [has_angle A] [has_angle B] [has_angle C]
  (AC AB BC : ℝ)
  (angle_A_eq : ∠A = 20)
  (angle_C_eq : ∠C = 40)
  (AC_minus_AB_eq : AC - AB = 5) : 
  ∃ BM : ℝ, (BM = 5) := 
sorry

end bisector_length_of_angle_B_l259_259580


namespace peter_remaining_money_l259_259725

theorem peter_remaining_money :
  let initial_money := 500
  let potato_cost := 6 * 2
  let tomato_cost := 9 * 3
  let cucumber_cost := 5 * 4
  let banana_cost := 3 * 5
  let total_cost := potato_cost + tomato_cost + cucumber_cost + banana_cost
  let remaining_money := initial_money - total_cost
  remaining_money = 426 :=
by
  let initial_money := 500
  let potato_cost := 6 * 2
  let tomato_cost := 9 * 3
  let cucumber_cost := 5 * 4
  let banana_cost := 3 * 5
  let total_cost := potato_cost + tomato_cost + cucumber_cost + banana_cost
  let remaining_money := initial_money - total_cost
  show remaining_money = 426 from sorry

end peter_remaining_money_l259_259725


namespace prime_factorization_sum_l259_259393

theorem prime_factorization_sum (w x y z k : ℕ) (h : 2^w * 3^x * 5^y * 7^z * 11^k = 2310) :
  2 * w + 3 * x + 5 * y + 7 * z + 11 * k = 28 :=
sorry

end prime_factorization_sum_l259_259393


namespace min_value_inequality_l259_259194

theorem min_value_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 3) :
  (1 / (a + b)) + (1 / (b + c)) + (1 / (c + a)) ≥ 3 / 2 :=
sorry

end min_value_inequality_l259_259194


namespace different_ways_to_eat_spaghetti_l259_259953

-- Define the conditions
def red_spaghetti := 5
def blue_spaghetti := 5
def total_spaghetti := 6

-- This is the proof statement
theorem different_ways_to_eat_spaghetti : 
  ∃ (ways : ℕ), ways = 62 ∧ 
  (∃ r b : ℕ, r ≤ red_spaghetti ∧ b ≤ blue_spaghetti ∧ r + b = total_spaghetti) := 
sorry

end different_ways_to_eat_spaghetti_l259_259953


namespace exist_irreducible_fractions_prod_one_l259_259494

theorem exist_irreducible_fractions_prod_one (S : List ℚ) :
  (∀ x ∈ S, ∃ (n d : ℤ), n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ x = (n /. d) ∧ Int.gcd n d = 1) ∧
  (∀ i j, i ≠ j → (S.get i).num ≠ (S.get j).num ∧ (S.get i).den ≠ (S.get j).den) →
  S.length = 3 ∧ S.prod = 1 :=
begin
  sorry
end

end exist_irreducible_fractions_prod_one_l259_259494


namespace seconds_in_3_hours_25_minutes_l259_259171

theorem seconds_in_3_hours_25_minutes:
  let hours := 3
  let minutesInAnHour := 60
  let additionalMinutes := 25
  let secondsInAMinute := 60
  (hours * minutesInAnHour + additionalMinutes) * secondsInAMinute = 12300 := 
by
  sorry

end seconds_in_3_hours_25_minutes_l259_259171


namespace total_cost_correct_l259_259427

-- Define the parameters
variables (a : ℕ) -- the number of books
-- Define the constants and the conditions
def unit_price : ℝ := 8
def shipping_fee_percentage : ℝ := 0.10

-- Define the total cost including the shipping fee
def total_cost (a : ℕ) : ℝ := unit_price * (1 + shipping_fee_percentage) * a

-- Prove that the total cost is equal to the expected amount
theorem total_cost_correct : total_cost a = 8 * (1 + 0.10) * a := by
  sorry

end total_cost_correct_l259_259427


namespace distinct_solutions_square_l259_259017

theorem distinct_solutions_square (α β : ℝ) (h₁ : α ≠ β)
    (h₂ : α^2 = 2 * α + 2 ∧ β^2 = 2 * β + 2) : (α - β) ^ 2 = 12 := by
  sorry

end distinct_solutions_square_l259_259017


namespace johnny_yellow_picks_l259_259715

variable (total_picks red_picks blue_picks yellow_picks : ℕ)

theorem johnny_yellow_picks
    (h_total_picks : total_picks = 3 * blue_picks)
    (h_half_red_picks : red_picks = total_picks / 2)
    (h_blue_picks : blue_picks = 12)
    (h_pick_sum : total_picks = red_picks + blue_picks + yellow_picks) :
    yellow_picks = 6 := by
  sorry

end johnny_yellow_picks_l259_259715


namespace sqrt_of_sum_of_fractions_l259_259106

theorem sqrt_of_sum_of_fractions:
  sqrt ((25 / 36) + (16 / 9)) = sqrt 89 / 6 := by
    sorry 

end sqrt_of_sum_of_fractions_l259_259106


namespace factorization_correct_l259_259311

noncomputable def factor_expression (y : ℝ) : ℝ :=
  3 * y * (y - 5) + 4 * (y - 5)

theorem factorization_correct (y : ℝ) : factor_expression y = (3 * y + 4) * (y - 5) :=
by sorry

end factorization_correct_l259_259311


namespace pages_to_read_tomorrow_l259_259889

-- Define the conditions
def total_pages : ℕ := 100
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5

-- Lean statement for the proof problem
theorem pages_to_read_tomorrow : (total_pages - (pages_yesterday + pages_today) = 35) :=
by
  let yesterday := pages_yesterday
  let today := pages_today
  let read_so_far := yesterday + today
  have read_so_far_eq : yesterday + today = 65 := by sorry
  have total_eq : total_pages - read_so_far = 35 := by sorry
  exact total_eq

end pages_to_read_tomorrow_l259_259889


namespace radius_of_circle_l259_259956

-- Circle with area x and circumference y
def circle_area (r : ℝ) : ℝ := π * r^2
def circle_circumference (r : ℝ) : ℝ := 2 * π * r
def circle_equation (r : ℝ) : ℝ := circle_area r + circle_circumference r

-- The given condition
theorem radius_of_circle (r : ℝ) (h : circle_equation r = 100 * π) : r = 10 :=
sorry

end radius_of_circle_l259_259956


namespace parabola_focus_equals_hyperbola_focus_l259_259994

noncomputable def hyperbola_right_focus : (Float × Float) := (2, 0)

noncomputable def parabola_focus (p : Float) : (Float × Float) := (p / 2, 0)

theorem parabola_focus_equals_hyperbola_focus (p : Float) :
  parabola_focus p = hyperbola_right_focus → p = 4 := by
  intro h
  sorry

end parabola_focus_equals_hyperbola_focus_l259_259994


namespace braden_total_money_after_bet_l259_259655

theorem braden_total_money_after_bet (initial_amount bet_multiplier : ℕ) (initial_money : initial_amount = 400) (bet_transition : bet_multiplier = 2) :
  let winning_amount := bet_multiplier * initial_amount in
  let total_amount := winning_amount + initial_amount in
  total_amount = 1200 :=
by
  sorry

end braden_total_money_after_bet_l259_259655


namespace solve_problem_l259_259585

noncomputable def Q (M : ℕ) : ℚ :=
  (Nat.floor (M / 3 : ℚ) + Nat.ceil (2 * M / 3 : ℚ)) / (M + 1 : ℚ)

lemma sum_of_digits {M : ℕ} (hM : M = 390) : 
  ∑ d in (M.digits 10), d = 12 := by
  sorry

theorem solve_problem : 
  ∃ M : ℕ, M % 6 = 0 ∧ Q(M) < 320 / 450 ∧ (M.digits 10).sum = 12 := by
  use 390
  split
  · exact Nat.mod_eq_zero_of_dvd (by norm_num : 6 ∣ 390)
  · norm_num [Q]
    rw [Nat.floor_eq, Nat.ceil_eq, Int.cast_coe_nat, Int.cast_coe_nat]
    norm_num
  · exact sum_of_digits rfl

end solve_problem_l259_259585


namespace tire_circumference_l259_259249

theorem tire_circumference (rpm : ℕ) (speed_kmh : ℕ) (C : ℝ) (h_rpm : rpm = 400) (h_speed_kmh : speed_kmh = 48) :
  (C = 2) :=
by
  -- sorry statement to assume the solution for now
  sorry

end tire_circumference_l259_259249


namespace hiker_distance_l259_259265

variable (s t d : ℝ)
variable (h₁ : (s + 1) * (2 / 3 * t) = d)
variable (h₂ : (s - 1) * (t + 3) = d)

theorem hiker_distance  : d = 6 :=
by
  sorry

end hiker_distance_l259_259265


namespace sufficient_not_necessary_condition_l259_259070

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, abs (x - 1) < 3 → (x + 2) * (x + a) < 0) ∧ 
  (∃ x : ℝ, (x + 2) * (x + a) < 0 ∧ ¬(abs (x - 1) < 3)) →
  a < -4 :=
by
  sorry

end sufficient_not_necessary_condition_l259_259070


namespace two_x_plus_y_equals_7_l259_259176

noncomputable def proof_problem (x y A : ℝ) : ℝ :=
  if (2 * x + y = A ∧ x + 2 * y = 8 ∧ (x + y) / 3 = 1.6666666666666667) then A else 0

theorem two_x_plus_y_equals_7 (x y : ℝ) : 
  (2 * x + y = proof_problem x y 7) ↔
  (2 * x + y = 7 ∧ x + 2 * y = 8 ∧ (x + y) / 3 = 1.6666666666666667) :=
by sorry

end two_x_plus_y_equals_7_l259_259176


namespace kiwi_count_l259_259120

theorem kiwi_count (s b o k : ℕ)
  (h1 : s + b + o + k = 340)
  (h2 : s = 3 * b)
  (h3 : o = 2 * k)
  (h4 : k = 5 * s) :
  k = 104 :=
sorry

end kiwi_count_l259_259120


namespace part1_solution_part2_solution_l259_259828

variable {x a : ℝ}

def f (x a : ℝ) : ℝ := abs (x - a)

theorem part1_solution (h1 : 0 ≤ x) (h2 : x ≤ 4) (h3 : f x a ≤ 2) : a = 2 :=
  sorry

theorem part2_solution (ha : 0 ≤ a) (hb : a ≤ 3) : (f (x + a) a + f (x - a) a ≥ f (a * x) a - a * f x a) :=
  sorry

end part1_solution_part2_solution_l259_259828


namespace members_do_not_play_either_l259_259574

noncomputable def total_members := 30
noncomputable def badminton_players := 16
noncomputable def tennis_players := 19
noncomputable def both_players := 7

theorem members_do_not_play_either : 
  (total_members - (badminton_players + tennis_players - both_players)) = 2 :=
by
  sorry

end members_do_not_play_either_l259_259574


namespace morning_registration_count_l259_259068

variable (M : ℕ) -- Number of students registered for the morning session
variable (MorningAbsentees : ℕ := 3) -- Absentees in the morning session
variable (AfternoonRegistered : ℕ := 24) -- Students registered for the afternoon session
variable (AfternoonAbsentees : ℕ := 4) -- Absentees in the afternoon session

theorem morning_registration_count :
  (M - MorningAbsentees) + (AfternoonRegistered - AfternoonAbsentees) = 42 → M = 25 :=
by
  sorry

end morning_registration_count_l259_259068


namespace count_of_inverses_mod_11_l259_259536

theorem count_of_inverses_mod_11 : (Finset.filter (λ a : ℕ, ∃ b : ℕ, (a * b) % 11 = 1) (Finset.range 11)).card = 10 := 
sorry

end count_of_inverses_mod_11_l259_259536


namespace segment_length_294_l259_259204

theorem segment_length_294
  (A B P Q : ℝ)   -- Define points A, B, P, Q on the real line
  (h1 : P = A + (3 / 8) * (B - A))   -- P divides AB in the ratio 3:5
  (h2 : Q = A + (4 / 11) * (B - A))  -- Q divides AB in the ratio 4:7
  (h3 : Q - P = 3)                   -- The length of PQ is 3
  : B - A = 294 := 
sorry

end segment_length_294_l259_259204


namespace fraction_decomposition_l259_259144

theorem fraction_decomposition (A B : ℚ) :
  (∀ x : ℚ, x ≠ -1 ∧ x ≠ 2  →
    7 * x - 18 = A * (3 * x + 1) + B * (x - 2))
  ↔ (A = -4 / 7 ∧ B = 61 / 7) :=
by
  sorry

end fraction_decomposition_l259_259144


namespace ac_bc_nec_not_suff_l259_259513

theorem ac_bc_nec_not_suff (a b c : ℝ) : 
  (a = b → a * c = b * c) ∧ (¬(a * c = b * c → a = b)) := by
  sorry

end ac_bc_nec_not_suff_l259_259513


namespace num_seven_digit_numbers_l259_259699

theorem num_seven_digit_numbers (a b c d e f g : ℕ)
  (h1 : a * b * c = 30)
  (h2 : c * d * e = 7)
  (h3 : e * f * g = 15) :
  ∃ n : ℕ, n = 4 := 
sorry

end num_seven_digit_numbers_l259_259699


namespace remaining_load_after_three_deliveries_l259_259645

def initial_load : ℝ := 50000
def unload_first_store (load : ℝ) : ℝ := load - 0.10 * load
def unload_second_store (load : ℝ) : ℝ := load - 0.20 * load
def unload_third_store (load : ℝ) : ℝ := load - 0.15 * load

theorem remaining_load_after_three_deliveries : 
  unload_third_store (unload_second_store (unload_first_store initial_load)) = 30600 := 
by
  sorry

end remaining_load_after_three_deliveries_l259_259645


namespace inequality_solution_l259_259210

theorem inequality_solution {x : ℝ} :
  ((x < 1) ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (6 < x)) ↔
  ((x - 2) * (x - 3) * (x - 4) / ((x - 1) * (x - 5) * (x - 6)) > 0) := sorry

end inequality_solution_l259_259210


namespace max_value_ab_l259_259387

theorem max_value_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 5 * a + 8 * b = 80) : ab ≤ 40 := 
  sorry

end max_value_ab_l259_259387


namespace num_integers_with_inverse_mod_11_l259_259537

theorem num_integers_with_inverse_mod_11 : 
  (Finset.card (Finset.filter (λ x : ℕ, ∃ y : ℕ, x * y % 11 = 1) (Finset.range 11))) = 10 := 
by 
  sorry

end num_integers_with_inverse_mod_11_l259_259537


namespace sqrt_two_irrational_l259_259922

theorem sqrt_two_irrational : ¬ ∃ (p q : ℕ), (q ≠ 0) ∧ (Nat.gcd p q = 1) ∧ (p ^ 2 = 2 * q ^ 2) := by
  sorry

end sqrt_two_irrational_l259_259922


namespace value_of_f_3_and_f_neg_7_point_5_l259_259990

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 1) = -f x
axiom definition_f : ∀ x : ℝ, -1 < x → x < 1 → f x = x

theorem value_of_f_3_and_f_neg_7_point_5 :
  f 3 + f (-7.5) = 0.5 :=
sorry

end value_of_f_3_and_f_neg_7_point_5_l259_259990


namespace problem_solution_l259_259967

noncomputable def equilateral_triangle_area_to_perimeter_square_ratio (s : ℝ) (h : s = 10) : ℝ :=
  let altitude := s * Real.sqrt 3 / 2
  let area := 1 / 2 * s * altitude
  let perimeter := 3 * s
  let perimeter_squared := perimeter^2
  area / perimeter_squared

theorem problem_solution :
  equilateral_triangle_area_to_perimeter_square_ratio 10 rfl = Real.sqrt 3 / 36 :=
sorry

end problem_solution_l259_259967


namespace hyperbola_asymptote_l259_259520

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) : 
  (∀ x, - y^2 = - x^2 / a^2 + 1) ∧ 
  (∀ x y, y + 2 * x = 0) → 
  a = 2 :=
by
  sorry

end hyperbola_asymptote_l259_259520


namespace weighted_mean_calculation_l259_259103

/-- Prove the weighted mean of the numbers 16, 28, and 45 with weights 2, 3, and 5 is 34.1 -/
theorem weighted_mean_calculation :
  let numbers := [16, 28, 45]
  let weights := [2, 3, 5]
  let total_weight := (2 + 3 + 5 : ℝ)
  let weighted_sum := ((16 * 2 + 28 * 3 + 45 * 5) : ℝ)
  (weighted_sum / total_weight) = 34.1 :=
by
  -- We only state the theorem without providing the proof
  sorry

end weighted_mean_calculation_l259_259103


namespace find_A_find_B_l259_259177

-- First problem: Prove A = 10 given 100A = 35^2 - 15^2
theorem find_A (A : ℕ) (h₁ : 100 * A = 35 ^ 2 - 15 ^ 2) : A = 10 := by
  sorry

-- Second problem: Prove B = 4 given (A-1)^6 = 27^B and A = 10
theorem find_B (B : ℕ) (A : ℕ) (h₁ : 100 * A = 35 ^ 2 - 15 ^ 2) (h₂ : (A - 1) ^ 6 = 27 ^ B) : B = 4 := by
  have A_is_10 : A = 10 := by
    apply find_A
    assumption
  sorry

end find_A_find_B_l259_259177


namespace meat_needed_for_40_hamburgers_l259_259886

theorem meat_needed_for_40_hamburgers (meat_per_10_hamburgers : ℕ) (hamburgers_needed : ℕ) (meat_per_hamburger : ℚ) (total_meat_needed : ℚ) :
  meat_per_10_hamburgers = 5 ∧ hamburgers_needed = 40 ∧
  meat_per_hamburger = meat_per_10_hamburgers / 10 ∧
  total_meat_needed = meat_per_hamburger * hamburgers_needed → 
  total_meat_needed = 20 := by
  sorry

end meat_needed_for_40_hamburgers_l259_259886


namespace negative_solution_condition_l259_259328

-- Define the system of equations
def system_of_equations (a b c x y : ℝ) :=
  a * x + b * y = c ∧
  b * x + c * y = a ∧
  c * x + a * y = b

-- State the theorem
theorem negative_solution_condition (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ system_of_equations a b c x y) ↔ a + b + c = 0 :=
by
  sorry

end negative_solution_condition_l259_259328


namespace sum_of_powers_divisible_by_30_l259_259426

theorem sum_of_powers_divisible_by_30 {a b c : ℤ} (h : (a + b + c) % 30 = 0) : (a^5 + b^5 + c^5) % 30 = 0 := by
  sorry

end sum_of_powers_divisible_by_30_l259_259426


namespace max_non_overlapping_areas_l259_259458

theorem max_non_overlapping_areas (n : ℕ) : 
  ∃ (max_areas : ℕ), max_areas = 3 * n := by
  sorry

end max_non_overlapping_areas_l259_259458


namespace arithmetic_sequence_sum_l259_259420

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℤ) (d : ℤ),
    (∀ n, a (n + 1) = a n + d) →
    (a 1 + a 4 + a 7 = 45) →
    (a 2 + a_5 + a_8 = 39) →
    (a 3 + a_6 + a_9 = 33) :=
by 
  intros a d h_arith_seq h_cond1 h_cond2
  sorry

end arithmetic_sequence_sum_l259_259420


namespace salary_for_May_l259_259213

variable (J F M A May : ℕ)

axiom condition1 : (J + F + M + A) / 4 = 8000
axiom condition2 : (F + M + A + May) / 4 = 8800
axiom condition3 : J = 3300

theorem salary_for_May : May = 6500 :=
by sorry

end salary_for_May_l259_259213


namespace num_7_digit_integers_correct_l259_259698

-- Define the number of choices for each digit
def first_digit_choices : ℕ := 9
def other_digit_choices : ℕ := 10

-- Define the number of 7-digit positive integers
def num_7_digit_integers : ℕ := first_digit_choices * other_digit_choices^6

-- State the theorem to prove
theorem num_7_digit_integers_correct : num_7_digit_integers = 9000000 :=
by
  sorry

end num_7_digit_integers_correct_l259_259698


namespace solve_inequality_l259_259498

theorem solve_inequality (x : ℝ) : (x + 1) / (x - 2) + (x - 3) / (3 * x) ≥ 4 ↔ x ∈ Set.Ico (-1/4) 0 ∪ Set.Ioc 2 3 := 
sorry

end solve_inequality_l259_259498


namespace determine_gizmos_l259_259119

theorem determine_gizmos (g d : ℝ)
  (h1 : 80 * (g * 160 + d * 240) = 80)
  (h2 : 100 * (3 * g * 900 + 3 * d * 600) = 100)
  (h3 : 70 * (5 * g * n + 5 * d * 1050) = 70 * 5 * (g + d) ) :
  n = 70 := sorry

end determine_gizmos_l259_259119


namespace volvox_pentagons_heptagons_diff_l259_259398

-- Given conditions
variables (V E F f_5 f_6 f_7 : ℕ)

-- Euler's polyhedron formula
axiom euler_formula : V - E + F = 2

-- Each edge is shared by two faces
axiom edge_formula : 2 * E = 5 * f_5 + 6 * f_6 + 7 * f_7

-- Each vertex shared by three faces
axiom vertex_formula : 3 * V = 5 * f_5 + 6 * f_6 + 7 * f_7

-- Total number of faces equals sum of individual face types 
def total_faces : ℕ := f_5 + f_6 + f_7

-- Prove that the number of pentagonal cells exceeds the number of heptagonal cells by 12
theorem volvox_pentagons_heptagons_diff : f_5 - f_7 = 12 := 
sorry

end volvox_pentagons_heptagons_diff_l259_259398


namespace sum_of_number_and_reverse_l259_259216

theorem sum_of_number_and_reverse (a b : Nat) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 := by
  sorry

end sum_of_number_and_reverse_l259_259216


namespace number_of_inverses_mod_11_l259_259547

theorem number_of_inverses_mod_11 : 
  ∃ n, n = 10 ∧ ∀ x ∈ finset.range 11, (gcd x 11 = 1 → ∃ y, (x * y) % 11 = 1) :=
by
  sorry

end number_of_inverses_mod_11_l259_259547


namespace find_C_share_l259_259073

-- Definitions
variable (A B C : ℝ)
variable (H1 : A + B + C = 585)
variable (H2 : 4 * A = 6 * B)
variable (H3 : 6 * B = 3 * C)

-- Problem statement
theorem find_C_share (A B C : ℝ) (H1 : A + B + C = 585) (H2 : 4 * A = 6 * B) (H3 : 6 * B = 3 * C) : C = 260 :=
by
  sorry

end find_C_share_l259_259073


namespace center_of_circle_polar_eq_l259_259878

theorem center_of_circle_polar_eq (ρ θ : ℝ) : 
    (∀ ρ θ, ρ = 2 * Real.cos θ ↔ (ρ * Real.cos θ - 1)^2 + (ρ * Real.sin θ)^2 = 1) → 
    ∃ x y : ℝ, x = 1 ∧ y = 0 :=
by
  sorry

end center_of_circle_polar_eq_l259_259878


namespace base7_to_base10_conversion_l259_259668

theorem base7_to_base10_conversion (n : ℤ) (h : n = 2 * 7^2 + 4 * 7^1 + 6 * 7^0) : n = 132 := by
  sorry

end base7_to_base10_conversion_l259_259668


namespace fraction_zero_implies_x_zero_l259_259180

theorem fraction_zero_implies_x_zero (x : ℝ) (h : (x^2 - x) / (x - 1) = 0) (h₁ : x ≠ 1) : x = 0 := by
  sorry

end fraction_zero_implies_x_zero_l259_259180


namespace midpoint_example_l259_259151

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem midpoint_example :
  midpoint (2, 9) (8, 3) = (5, 6) :=
by
  sorry

end midpoint_example_l259_259151


namespace years_taught_third_grade_l259_259202

def total_years : ℕ := 26
def years_taught_second_grade : ℕ := 8

theorem years_taught_third_grade :
  total_years - years_taught_second_grade = 18 :=
by {
  -- Subtract the years taught second grade from the total years
  -- Exact the result
  sorry
}

end years_taught_third_grade_l259_259202


namespace other_root_eq_six_l259_259354

theorem other_root_eq_six (a : ℝ) (x1 : ℝ) (x2 : ℝ) 
  (h : x1 = -2) 
  (eqn : ∀ x, x^2 - a * x - 3 * a = 0 → (x = x1 ∨ x = x2)) :
  x2 = 6 :=
by
  sorry

end other_root_eq_six_l259_259354


namespace minimize_sum_of_distances_l259_259169

theorem minimize_sum_of_distances (P : ℝ × ℝ) (A : ℝ × ℝ) (F : ℝ × ℝ) 
  (hP_on_parabola : P.2 ^ 2 = 2 * P.1)
  (hA : A = (3, 2)) 
  (hF : F = (1/2, 0)) : 
  |P - A| + |P - F| ≥ |(2, 2) - A| + |(2, 2) - F| :=
by sorry

end minimize_sum_of_distances_l259_259169


namespace count_inverses_mod_11_l259_259528

theorem count_inverses_mod_11 : (∃ n : ℕ, n = 10) :=
  have h : ∀ a ∈ finset.range 11, nat.gcd a 11 = 1 -> a ≠ 0 := by 
    intro a ha h1,
    apply (ne_of_lt (by linarith : a < 11)),
    apply nat.pos_of_ne_zero,
    intro hz,
    rw [hz, nat.gcd_zero_left] at h1,
    exact nat.prime.not_dvd_one (nat.prime_iff.2 ⟨sorry, sorry⟩) 11 h1,
  sorry

end count_inverses_mod_11_l259_259528


namespace MaxCandy_l259_259507

theorem MaxCandy (frankieCandy : ℕ) (extraCandy : ℕ) (maxCandy : ℕ) 
  (h1 : frankieCandy = 74) (h2 : extraCandy = 18) (h3 : maxCandy = frankieCandy + extraCandy) :
  maxCandy = 92 := 
by
  sorry

end MaxCandy_l259_259507


namespace solve_abs_inequality_l259_259078

theorem solve_abs_inequality (x : ℝ) : (|x + 3| + |x - 4| < 8) ↔ (4 ≤ x ∧ x < 4.5) := sorry

end solve_abs_inequality_l259_259078


namespace pupils_like_only_maths_l259_259752

noncomputable def number_pupils_like_only_maths (total: ℕ) (maths_lovers: ℕ) (english_lovers: ℕ) 
(neither_lovers: ℕ) (both_lovers: ℕ) : ℕ :=
maths_lovers - both_lovers

theorem pupils_like_only_maths : 
∀ (total: ℕ) (maths_lovers: ℕ) (english_lovers: ℕ) (neither_lovers: ℕ) (both_lovers: ℕ),
total = 30 →
maths_lovers = 20 →
english_lovers = 18 →
both_lovers = 2 * neither_lovers →
neither_lovers + maths_lovers + english_lovers - both_lovers - both_lovers = total →
number_pupils_like_only_maths total maths_lovers english_lovers neither_lovers both_lovers = 4 :=
by
  intros _ _ _ _ _ _ _ _ _ _
  sorry

end pupils_like_only_maths_l259_259752


namespace num_subsets_without_adjacent_elements_l259_259606

theorem num_subsets_without_adjacent_elements {n k : ℕ} :
  (finset.univ.powerset.filter (λ s : finset ℕ, s.card = k ∧
    ∀ i ∈ s, i + 1 ∉ s)).card = nat.choose (n - k + 1) k := sorry

end num_subsets_without_adjacent_elements_l259_259606


namespace comparison_abc_l259_259053

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem comparison_abc : a > c ∧ c > b :=
by
  sorry

end comparison_abc_l259_259053


namespace solution_set_abs_inequality_l259_259614

theorem solution_set_abs_inequality : {x : ℝ | |x - 1| < 2} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end solution_set_abs_inequality_l259_259614


namespace Exponent_Equality_l259_259561

theorem Exponent_Equality : 2^8 * 2^32 = 256^5 :=
by
  sorry

end Exponent_Equality_l259_259561


namespace comparison_abc_l259_259052

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem comparison_abc : a > c ∧ c > b :=
by
  sorry

end comparison_abc_l259_259052


namespace tommy_paint_cost_l259_259226

def tommy_spends_on_paint : ℕ :=
  let width := 5 in
  let height := 4 in
  let sides := 2 in
  let cost_per_quart := 2 in
  let coverage_per_quart := 4 in
  let area_per_side := width * height in
  let total_area := sides * area_per_side in
  let quarts_needed := total_area / coverage_per_quart in
  let total_cost := quarts_needed * cost_per_quart in
  total_cost

theorem tommy_paint_cost : tommy_spends_on_paint = 20 := by
  sorry

end tommy_paint_cost_l259_259226


namespace triangle_area_difference_l259_259711

-- Definitions per conditions
def right_angle (A B C : Type) (angle_EAB : Prop) : Prop := angle_EAB
def angle_ABC_eq_30 (A B C : Type) (angle_ABC : ℝ) : Prop := angle_ABC = 30
def length_AB_eq_5 (A B : Type) (AB : ℝ) : Prop := AB = 5
def length_BC_eq_7 (B C : Type) (BC : ℝ) : Prop := BC = 7
def length_AE_eq_10 (A E : Type) (AE : ℝ) : Prop := AE = 10
def lines_intersect_at_D (A B C E D : Type) (intersects : Prop) : Prop := intersects

-- Main theorem statement
theorem triangle_area_difference
  (A B C E D : Type)
  (angle_EAB : Prop)
  (right_EAB : right_angle A E B angle_EAB)
  (angle_ABC : ℝ)
  (angle_ABC_is_30 : angle_ABC_eq_30 A B C angle_ABC)
  (AB : ℝ)
  (AB_is_5 : length_AB_eq_5 A B AB)
  (BC : ℝ)
  (BC_is_7 : length_BC_eq_7 B C BC)
  (AE : ℝ)
  (AE_is_10 : length_AE_eq_10 A E AE)
  (intersects : Prop)
  (intersects_at_D : lines_intersect_at_D A B C E D intersects) :
  (area_ADE - area_BDC) = 16.25 := sorry

end triangle_area_difference_l259_259711


namespace remaining_strawberries_l259_259431

-- Define the constants based on conditions
def initial_kg1 : ℕ := 3
def initial_g1 : ℕ := 300
def given_kg1 : ℕ := 1
def given_g1 : ℕ := 900

-- Define the conversion from kilograms to grams
def kg_to_g (kg : ℕ) : ℕ := kg * 1000

-- Calculate initial total grams
def initial_total_g : ℕ := kg_to_g initial_kg1 + initial_g1

-- Calculate given total grams
def given_total_g : ℕ := kg_to_g given_kg1 + given_g1

-- Define the remaining grams
def remaining_g (initial_g : ℕ) (given_g : ℕ) : ℕ := initial_g - given_g

-- Statement to prove
theorem remaining_strawberries : remaining_g initial_total_g given_total_g = 1400 := by
sorry

end remaining_strawberries_l259_259431


namespace base9_subtraction_l259_259437

theorem base9_subtraction (a b : Nat) (h1 : a = 256) (h2 : b = 143) : 
  (a - b) = 113 := 
sorry

end base9_subtraction_l259_259437


namespace pipes_fill_tank_in_10_hours_l259_259644

noncomputable def R_A := 1 / 70
noncomputable def R_B := 2 * R_A
noncomputable def R_C := 2 * R_B
noncomputable def R_total := R_A + R_B + R_C
noncomputable def T := 1 / R_total

theorem pipes_fill_tank_in_10_hours :
  T = 10 := 
sorry

end pipes_fill_tank_in_10_hours_l259_259644


namespace solve_equation_l259_259077

theorem solve_equation (x y : ℝ) (k : ℤ) :
  x^2 - 2 * x * Real.sin (x * y) + 1 = 0 ↔ (x = 1 ∧ y = (Real.pi / 2) + 2 * Real.pi * k) ∨ (x = -1 ∧ y = (Real.pi / 2) + 2 * Real.pi * k) :=
by
  -- Logical content will be filled here, sorry is used because proof steps are not required.
  sorry

end solve_equation_l259_259077


namespace desired_percentage_alcohol_l259_259455

noncomputable def original_volume : ℝ := 6
noncomputable def original_percentage : ℝ := 0.40
noncomputable def added_alcohol : ℝ := 1.2
noncomputable def final_solution_volume : ℝ := original_volume + added_alcohol
noncomputable def final_alcohol_volume : ℝ := (original_percentage * original_volume) + added_alcohol
noncomputable def desired_percentage : ℝ := (final_alcohol_volume / final_solution_volume) * 100

theorem desired_percentage_alcohol :
  desired_percentage = 50 := by
  sorry

end desired_percentage_alcohol_l259_259455


namespace system_has_negative_solution_iff_sum_zero_l259_259334

variables {a b c x y : ℝ}

-- Statement of the problem
theorem system_has_negative_solution_iff_sum_zero :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔ (a + b + c = 0) := by
  sorry

end system_has_negative_solution_iff_sum_zero_l259_259334


namespace find_g_inv_f_8_l259_259566

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom f_inv_g : ∀ x : ℝ, f_inv (g x) = x^2 - x
axiom g_bijective : Function.Bijective g

theorem find_g_inv_f_8 : g_inv (f 8) = (1 + Real.sqrt 33) / 2 :=
by
  -- proof is omitted
  sorry

end find_g_inv_f_8_l259_259566


namespace num_inverses_mod_11_l259_259555

theorem num_inverses_mod_11 : (Finset.filter (λ a, Nat.gcd a 11 = 1) (Finset.range 11)).card = 10 :=
sorry

end num_inverses_mod_11_l259_259555


namespace fair_prize_division_l259_259939

theorem fair_prize_division (eq_chance : ∀ (game : ℕ), 0.5 ≤ 1 ∧ 1 ≤ 0.5)
  (first_to_six : ∀ (p1_wins p2_wins : ℕ), (p1_wins = 6 ∨ p2_wins = 6) → (p1_wins + p2_wins) ≤ 11)
  (current_status : 5 + 3 = 8) :
  (7 : ℝ) / 8 = 7 / (8 : ℝ) :=
by
  sorry

end fair_prize_division_l259_259939


namespace geometric_sequence_178th_term_l259_259022

-- Conditions of the problem as definitions
def first_term : ℤ := 5
def second_term : ℤ := -20
def common_ratio : ℤ := second_term / first_term
def nth_term (a : ℤ) (r : ℤ) (n : ℕ) : ℤ := a * r^(n-1)

-- The translated problem statement in Lean 4
theorem geometric_sequence_178th_term :
  nth_term first_term common_ratio 178 = -5 * 4^177 :=
by
  repeat { sorry }

end geometric_sequence_178th_term_l259_259022


namespace citizens_own_a_cat_l259_259874

theorem citizens_own_a_cat (p d : ℝ) (n : ℕ) (h1 : p = 0.60) (h2 : d = 0.50) (h3 : n = 100) : 
  (p * n - d * p * n) = 30 := 
by 
  sorry

end citizens_own_a_cat_l259_259874


namespace stewart_farm_horse_food_l259_259130

theorem stewart_farm_horse_food 
  (ratio : ℚ) (food_per_horse : ℤ) (num_sheep : ℤ) (num_horses : ℤ)
  (h1 : ratio = 5 / 7)
  (h2 : food_per_horse = 230)
  (h3 : num_sheep = 40)
  (h4 : ratio * num_horses = num_sheep) : 
  (num_horses * food_per_horse = 12880) := 
sorry

end stewart_farm_horse_food_l259_259130


namespace coins_of_each_type_l259_259769

theorem coins_of_each_type (x : ℕ) (h : x + x / 2 + x / 4 = 70) : x = 40 :=
sorry

end coins_of_each_type_l259_259769


namespace find_number_l259_259118

theorem find_number (x : ℕ) (h : x * 12 = 540) : x = 45 :=
by sorry

end find_number_l259_259118


namespace find_N_l259_259518

/-- Given a row: [a, b, c, d, 2, f, g], 
    first column: [15, h, i, 14, j, k, l, 10],
    second column: [N, m, n, o, p, q, r, -21],
    where h=i+4 and i=j+4,
    b=15 and d = (2 - 15) / 3.
    The common difference c_n = -2.5.
    Prove N = -13.5.
-/
theorem find_N (a b c d f g h i j k l m n o p q r : ℝ) (N : ℝ) :
  b = 15 ∧ j = 14 ∧ l = 10 ∧ r = -21 ∧
  h = i + 4 ∧ i = j + 4 ∧
  c = (2 - 15) / 3 ∧
  g = b + 6 * c ∧
  N = g + 1 * (-2.5) →
  N = -13.5 :=
by
  intros h1
  sorry

end find_N_l259_259518


namespace inequality_a_c_b_l259_259059

theorem inequality_a_c_b :
  let a := 2 * Real.log 1.01
  let b := Real.log 1.02
  let c := Real.sqrt 1.04 - 1
  a > c ∧ c > b :=
by
  let a : ℝ := 2 * Real.log 1.01
  let b : ℝ := Real.log 1.02
  let c : ℝ := Real.sqrt 1.04 - 1
  split
  sorry
  sorry

end inequality_a_c_b_l259_259059


namespace hexagon_shaded_area_l259_259294

-- Given conditions
variable (A B C D T : ℝ)
variable (h₁ : A = 2)
variable (h₂ : B = 3)
variable (h₃ : C = 4)
variable (h₄ : T = 20)
variable (h₅ : A + B + C + D = T)

-- The goal is to prove that the area of the shaded region (D) is 11 cm².
theorem hexagon_shaded_area : D = 11 := by
  sorry

end hexagon_shaded_area_l259_259294


namespace probability_of_forming_triangle_is_three_tenths_l259_259921

-- Define the set {1, 2, 3, 4, 5}
def S : Finset ℕ := {1, 2, 3, 4, 5}

-- Define the function that checks the triangle inequality
def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the probability of forming a triangle
def probability_triangle (s : Finset ℕ) (n : ℕ) : ℚ :=
  let triplets := s.powerset.filter (λ x, x.card = 3),
      valid_triplets := triplets.filter (λ x, ∀ t ∈ x.val, t.satisfies_triangle_inequality) in
  valid_triplets.card / triplets.card

-- Statement to prove
theorem probability_of_forming_triangle_is_three_tenths :
  probability_triangle S 3 = 3 / 10 := by
    sorry

end probability_of_forming_triangle_is_three_tenths_l259_259921


namespace profit_per_cake_l259_259292

theorem profit_per_cake (ingredient_cost : ℝ) (packaging_cost : ℝ) (selling_price : ℝ) (cake_count : ℝ)
    (h1 : ingredient_cost = 12) (h2 : packaging_cost = 1) (h3 : selling_price = 15) (h4 : cake_count = 2) :
    selling_price - (ingredient_cost / cake_count + packaging_cost) = 8 := by
  sorry

end profit_per_cake_l259_259292


namespace min_kinds_of_candies_l259_259868

theorem min_kinds_of_candies (candies : ℕ) (even_distance_candies : ∀ i j : ℕ, i ≠ j → i < candies → j < candies → is_even (j - i - 1)) :
  candies = 91 → 46 ≤ candies :=
by
  assume h1 : candies = 91
  sorry

end min_kinds_of_candies_l259_259868


namespace fraction_irreducible_l259_259917

theorem fraction_irreducible (a b c d : ℤ) (h : a * d - b * c = 1) : ∀ m : ℤ, m > 1 → ¬ (m ∣ (a^2 + b^2) ∧ m ∣ (a * c + b * d)) :=
by sorry

end fraction_irreducible_l259_259917


namespace closest_point_on_parabola_l259_259745

/-- The coordinates of the point on the parabola y^2 = x that is closest to the line x - 2y + 4 = 0 are (1,1). -/
theorem closest_point_on_parabola (y : ℝ) (x : ℝ) (h_parabola : y^2 = x) (h_line : x - 2*y + 4 = 0) :
  (x = 1 ∧ y = 1) :=
sorry

end closest_point_on_parabola_l259_259745


namespace transformed_parabola_eq_l259_259185

noncomputable def initial_parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 + 3
def shift_left (h : ℝ) (c : ℝ): ℝ := h - c
def shift_down (k : ℝ) (d : ℝ): ℝ := k - d

theorem transformed_parabola_eq :
  ∃ (x : ℝ), (initial_parabola (shift_left x 2) - 1 = 2 * (x + 1)^2 + 2) :=
sorry

end transformed_parabola_eq_l259_259185


namespace difference_max_min_y_l259_259020

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

end difference_max_min_y_l259_259020


namespace complex_mul_eq_l259_259304

/-- Proof that the product of two complex numbers (1 + i) and (2 + i) is equal to (1 + 3i) -/
theorem complex_mul_eq (i : ℂ) (h_i_squared : i^2 = -1) : (1 + i) * (2 + i) = 1 + 3 * i :=
by
  -- The actual proof logic goes here.
  sorry

end complex_mul_eq_l259_259304


namespace system_has_negative_solution_iff_sum_zero_l259_259330

variables {a b c x y : ℝ}

-- Statement of the problem
theorem system_has_negative_solution_iff_sum_zero :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔ (a + b + c = 0) := by
  sorry

end system_has_negative_solution_iff_sum_zero_l259_259330


namespace borrowed_nickels_l259_259395

def n_original : ℕ := 87
def n_left : ℕ := 12
def n_borrowed : ℕ := n_original - n_left

theorem borrowed_nickels : n_borrowed = 75 := by
  sorry

end borrowed_nickels_l259_259395


namespace number_of_distinct_intersections_l259_259008

theorem number_of_distinct_intersections :
  (∃ x y : ℝ, 9 * x^2 + 16 * y^2 = 16 ∧ 16 * x^2 + 9 * y^2 = 9) →
  (∀ x y₁ y₂ : ℝ, 9 * x^2 + 16 * y₁^2 = 16 ∧ 16 * x^2 + 9 * y₁^2 = 9 ∧
    9 * x^2 + 16 * y₂^2 = 16 ∧ 16 * x^2 + 9 * y₂^2 = 9 → y₁ = y₂) →
  (∃! p : ℝ × ℝ, 9 * p.1^2 + 16 * p.2^2 = 16 ∧ 16 * p.1^2 + 9 * p.2^2 = 9) :=
by
  sorry

end number_of_distinct_intersections_l259_259008


namespace green_folder_stickers_l259_259915

theorem green_folder_stickers (total_stickers red_sheets blue_sheets : ℕ) (red_sticker_per_sheet blue_sticker_per_sheet green_stickers_needed green_sheets : ℕ) :
  total_stickers = 60 →
  red_sticker_per_sheet = 3 →
  blue_sticker_per_sheet = 1 →
  red_sheets = 10 →
  blue_sheets = 10 →
  green_sheets = 10 →
  let red_stickers_total := red_sticker_per_sheet * red_sheets
  let blue_stickers_total := blue_sticker_per_sheet * blue_sheets
  let green_stickers_total := total_stickers - (red_stickers_total + blue_stickers_total)
  green_sticker_per_sheet = green_stickers_total / green_sheets →
  green_sticker_per_sheet = 2 := 
sorry

end green_folder_stickers_l259_259915


namespace average_temperature_week_l259_259214

theorem average_temperature_week :
  let d1 := 40
  let d2 := 40
  let d3 := 40
  let d4 := 80
  let d5 := 80
  let remaining_days_total := 140
  d1 + d2 + d3 + d4 + d5 + remaining_days_total = 420 ∧ 420 / 7 = 60 :=
by sorry

end average_temperature_week_l259_259214


namespace fermat_numbers_coprime_l259_259905

theorem fermat_numbers_coprime (n m : ℕ) (h : n ≠ m) :
  Nat.gcd (2 ^ 2 ^ (n - 1) + 1) (2 ^ 2 ^ (m - 1) + 1) = 1 :=
sorry

end fermat_numbers_coprime_l259_259905


namespace ratio_x_y_l259_259506

theorem ratio_x_y (x y : ℝ) (h1 : x * y = 9) (h2 : 0 < x) (h3 : 0 < y) (h4 : y = 0.5) : x / y = 36 :=
by
  sorry

end ratio_x_y_l259_259506


namespace total_time_is_11_l259_259133

-- Define the times each person spent in the pool
def Jerry_time : Nat := 3
def Elaine_time : Nat := 2 * Jerry_time
def George_time : Nat := Elaine_time / 3
def Kramer_time : Nat := 0

-- Define the total time spent in the pool by all friends
def total_time : Nat := Jerry_time + Elaine_time + George_time + Kramer_time

-- Prove that the total time is 11 minutes
theorem total_time_is_11 : total_time = 11 := sorry

end total_time_is_11_l259_259133


namespace truck_stops_l259_259244

variable (a : ℕ → ℕ)
variable (sum_1 : ℕ)
variable (sum_2 : ℕ)

-- Definition for the first sequence with a common difference of -10
def first_sequence : ℕ → ℕ
| 0       => 40
| (n + 1) => first_sequence n - 10

-- Definition for the second sequence with a common difference of -5
def second_sequence : ℕ → ℕ 
| 0       => 10
| (n + 1) => second_sequence n - 5

-- Summing the first sequence elements before the condition change:
def sum_first_sequence : ℕ → ℕ 
| 0       => 40
| (n + 1) => sum_first_sequence n + first_sequence (n + 1)

-- Summing the second sequence elements after the condition change:
def sum_second_sequence : ℕ → ℕ 
| 0       => second_sequence 0
| (n + 1) => sum_second_sequence n + second_sequence (n + 1)

-- Final sum of distances
def total_distance : ℕ :=
  sum_first_sequence 3 + sum_second_sequence 1

theorem truck_stops (sum_1 sum_2 : ℕ) (h1 : sum_1 = sum_first_sequence 3)
 (h2 : sum_2 = sum_second_sequence 1) : 
  total_distance = 115 := by
  sorry


end truck_stops_l259_259244


namespace candy_problem_l259_259852

-- Definitions of conditions
def total_candies : ℕ := 91
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- The minimum number of kinds of candies function
def min_kinds : ℕ := 46

-- Lean statement for the proof problem
theorem candy_problem : 
  ∀ (kinds : ℕ), 
    (∀ c : ℕ, c < total_candies → c % kinds < 2) → (∃ n : ℕ, kinds = min_kinds) := 
sorry

end candy_problem_l259_259852


namespace solve_system_nat_l259_259608

theorem solve_system_nat (a b c d : ℕ) :
  (a * b = c + d ∧ c * d = a + b) →
  (a = 1 ∧ b = 5 ∧ c = 2 ∧ d = 3) ∨
  (a = 1 ∧ b = 5 ∧ c = 3 ∧ d = 2) ∨
  (a = 5 ∧ b = 1 ∧ c = 2 ∧ d = 3) ∨
  (a = 5 ∧ b = 1 ∧ c = 3 ∧ d = 2) ∨
  (a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2) ∨
  (a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 5) ∨
  (a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 1) ∨
  (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 5) ∨
  (a = 3 ∧ b = 2 ∧ c = 5 ∧ d = 1) :=
sorry

end solve_system_nat_l259_259608


namespace solve_equation_naturals_l259_259111

theorem solve_equation_naturals :
  ∀ (X Y Z : ℕ), X^Y + Y^Z = X * Y * Z ↔ 
    (X = 1 ∧ Y = 1 ∧ Z = 2) ∨ 
    (X = 2 ∧ Y = 2 ∧ Z = 2) ∨ 
    (X = 2 ∧ Y = 2 ∧ Z = 3) ∨ 
    (X = 4 ∧ Y = 2 ∧ Z = 3) ∨ 
    (X = 4 ∧ Y = 2 ∧ Z = 4) := 
by
  sorry

end solve_equation_naturals_l259_259111


namespace new_students_joined_l259_259442

-- Define conditions
def initial_students : ℕ := 160
def end_year_students : ℕ := 120
def fraction_transferred_out : ℚ := 1 / 3
def total_students_at_start := end_year_students * 3 / 2

-- Theorem statement
theorem new_students_joined : (total_students_at_start - initial_students = 20) :=
by
  -- Placeholder for proof
  sorry

end new_students_joined_l259_259442


namespace one_in_B_neg_one_not_in_B_B_roster_l259_259908

open Set Int

def B : Set ℤ := {x | ∃ n : ℕ, 6 = n * (3 - x)}

theorem one_in_B : 1 ∈ B :=
by sorry

theorem neg_one_not_in_B : (-1 ∉ B) :=
by sorry

theorem B_roster : B = {2, 1, 0, -3} :=
by sorry

end one_in_B_neg_one_not_in_B_B_roster_l259_259908


namespace area_of_rectangle_is_correct_l259_259774

-- Given Conditions
def radius : ℝ := 7
def diameter : ℝ := 2 * radius
def width : ℝ := diameter
def length : ℝ := 3 * width

-- Question: Find the area of the rectangle
def area := length * width

-- The theorem to prove
theorem area_of_rectangle_is_correct : area = 588 :=
by
  -- Proof steps can go here.
  sorry

end area_of_rectangle_is_correct_l259_259774


namespace find_a_l259_259576

theorem find_a (a n : ℝ) (p : ℝ) (hp : p = 2 / 3)
  (h₁ : a = 3 * n + 5)
  (h₂ : a + 2 = 3 * (n + p) + 5) : a = 3 * n + 5 :=
by 
  sorry

end find_a_l259_259576


namespace irreducible_fractions_product_one_l259_259485

theorem irreducible_fractions_product_one : ∃ (a b c d e f : ℕ),
  {a, b, c, d, e, f}.Subset {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  {a, b, c, d, e, f}.card = 6 ∧
  ∃ (f1_num f1_den f2_num f2_den f3_num f3_den : ℕ), 
    (f1_num ≠ f1_den ∧ coprime f1_num f1_den ∧ f1_num ∈ {a, b, c, d, e, f} ∧ f1_den ∈ {a, b, c, d, e, f} ∧ 
    f2_num ≠ f2_den ∧ coprime f2_num f2_den ∧ f2_num ∈ {a, b, c, d, e, f} ∧ f2_den ∈ {a, b, c, d, e, f} ∧ 
    f3_num ≠ f3_den ∧ coprime f3_num f3_den ∧ f3_num ∈ {a, b, c, d, e, f} ∧ f3_den ∈ {a, b, c, d, e, f} ∧ 
    (f1_num * f2_num * f3_num) = (f1_den * f2_den * f3_den)) :=
sorry

end irreducible_fractions_product_one_l259_259485


namespace irreducible_fractions_product_one_l259_259487

theorem irreducible_fractions_product_one : ∃ (a b c d e f : ℕ),
  {a, b, c, d, e, f}.Subset {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  {a, b, c, d, e, f}.card = 6 ∧
  ∃ (f1_num f1_den f2_num f2_den f3_num f3_den : ℕ), 
    (f1_num ≠ f1_den ∧ coprime f1_num f1_den ∧ f1_num ∈ {a, b, c, d, e, f} ∧ f1_den ∈ {a, b, c, d, e, f} ∧ 
    f2_num ≠ f2_den ∧ coprime f2_num f2_den ∧ f2_num ∈ {a, b, c, d, e, f} ∧ f2_den ∈ {a, b, c, d, e, f} ∧ 
    f3_num ≠ f3_den ∧ coprime f3_num f3_den ∧ f3_num ∈ {a, b, c, d, e, f} ∧ f3_den ∈ {a, b, c, d, e, f} ∧ 
    (f1_num * f2_num * f3_num) = (f1_den * f2_den * f3_den)) :=
sorry

end irreducible_fractions_product_one_l259_259487


namespace restaurant_total_cost_l259_259965

theorem restaurant_total_cost :
  let vegetarian_cost := 5
  let chicken_cost := 7
  let steak_cost := 10
  let kids_cost := 3
  let tax_rate := 0.10
  let tip_rate := 0.15
  let num_vegetarians := 3
  let num_chicken_lovers := 4
  let num_steak_enthusiasts := 2
  let num_kids_hot_dog := 3
  let subtotal := (num_vegetarians * vegetarian_cost) + (num_chicken_lovers * chicken_cost) + (num_steak_enthusiasts * steak_cost) + (num_kids_hot_dog * kids_cost)
  let tax := subtotal * tax_rate
  let tip := subtotal * tip_rate
  let total_cost := subtotal + tax + tip
  total_cost = 90 :=
by sorry

end restaurant_total_cost_l259_259965


namespace average_age_nine_students_l259_259085

theorem average_age_nine_students (total_age_15_students : ℕ)
                                (total_age_5_students : ℕ)
                                (age_15th_student : ℕ)
                                (h1 : total_age_15_students = 225)
                                (h2 : total_age_5_students = 65)
                                (h3 : age_15th_student = 16) :
                                (total_age_15_students - total_age_5_students - age_15th_student) / 9 = 16 := by
  sorry

end average_age_nine_students_l259_259085


namespace compose_frac_prod_eq_one_l259_259479

open Finset

def irreducible_fraction (n d : ℕ) := gcd n d = 1

theorem compose_frac_prod_eq_one :
  ∃ (a b c d e f : ℕ),
  {a, b, c, d, e, f} ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
   d ≠ e ∧ d ≠ f ∧ 
   e ≠ f) ∧
  irreducible_fraction a b ∧
  irreducible_fraction c d ∧
  irreducible_fraction e f ∧
  (a : ℚ) / b * (c : ℚ) / d * (e : ℚ) / f = 1 :=
begin
  sorry
end

end compose_frac_prod_eq_one_l259_259479


namespace problem_solution_l259_259102

/-- Define the repeating decimal 0.\overline{49} as a rational number. --/
def rep49 := 7 / 9

/-- Define the repeating decimal 0.\overline{4} as a rational number. --/
def rep4 := 4 / 9

/-- The main theorem stating that 99 times the difference between 
    the repeating decimals 0.\overline{49} and 0.\overline{4} equals 5. --/
theorem problem_solution : 99 * (rep49 - rep4) = 5 := by
  sorry

end problem_solution_l259_259102


namespace billy_total_tickets_l259_259297

theorem billy_total_tickets :
  let ferris_wheel_rides := 7
  let bumper_car_rides := 3
  let roller_coaster_rides := 4
  let teacups_rides := 5
  let ferris_wheel_cost := 5
  let bumper_car_cost := 6
  let roller_coaster_cost := 8
  let teacups_cost := 4
  let total_ferris_wheel := ferris_wheel_rides * ferris_wheel_cost
  let total_bumper_cars := bumper_car_rides * bumper_car_cost
  let total_roller_coaster := roller_coaster_rides * roller_coaster_cost
  let total_teacups := teacups_rides * teacups_cost
  let total_tickets := total_ferris_wheel + total_bumper_cars + total_roller_coaster + total_teacups
  total_tickets = 105 := 
sorry

end billy_total_tickets_l259_259297


namespace side_length_square_eq_4_l259_259412

theorem side_length_square_eq_4 (s : ℝ) (h : s^2 - 3 * s = 4) : s = 4 :=
sorry

end side_length_square_eq_4_l259_259412


namespace flight_time_is_approximately_50_hours_l259_259091

noncomputable def flightTime (radius : ℝ) (speed : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  circumference / speed

theorem flight_time_is_approximately_50_hours :
  let radius := 4200
  let speed := 525
  abs (flightTime radius speed - 50) < 1 :=
by
  sorry

end flight_time_is_approximately_50_hours_l259_259091


namespace max_xi_value_prob_xi_max_prob_dist_xi_expected_value_xi_l259_259021

-- Cards labeled 1, 2, 3
def card_labels : Set ℕ := {1, 2, 3}

-- Define the random variable xi
noncomputable def xi (x y : ℕ) : ℕ :=
  |x - 2| + |y - x|

-- Probability function for distribution
def prob_fun (x y : ℕ) : ℚ :=
  1 / 9

-- Prove the maximum value of xi
theorem max_xi_value : ∃ x y, xi x y = 3 :=
by
  refine ⟨_, _, _⟩;
  -- Plug in the values that give xi = 3, e.g., (1, 3) and (3, 1)
  sorry

-- Prove the probability of xi attaining its maximum value
theorem prob_xi_max : prob_fun 1 3 + prob_fun 3 1 = 2 / 9 :=
by
  -- Since (1, 3) and (3, 1) give xi = 3 and each has probability 1/9
  sorry

-- Prove the probability distribution of xi
theorem prob_dist_xi : 
  (prob_fun 2 2 = 1/9) ∧
  (prob_fun 1 1 + prob_fun 2 1 + prob_fun 2 3 + prob_fun 3 3 = 4/9) ∧
  (prob_fun 1 2 + prob_fun 3 2 = 2/9) ∧
  (prob_fun 1 3 + prob_fun 3 1 = 2/9) :=
by
  sorry

-- Prove the expected value of xi
theorem expected_value_xi : 
  ∑ (x y : ℕ) in card_labels, xi x y * prob_fun x y = 14 / 9 :=
by
  -- Calculation of expected value
  sorry

end max_xi_value_prob_xi_max_prob_dist_xi_expected_value_xi_l259_259021


namespace radius_of_cookie_l259_259114

theorem radius_of_cookie (x y : ℝ) : 
  (x^2 + y^2 + x - 5 * y = 10) → 
  ∃ r, (r = Real.sqrt (33 / 2)) :=
by
  sorry

end radius_of_cookie_l259_259114


namespace parabola_vertex_l259_259087

theorem parabola_vertex (x y : ℝ) : y^2 + 6*y + 2*x + 5 = 0 → (x, y) = (2, -3) :=
sorry

end parabola_vertex_l259_259087


namespace negative_solution_exists_l259_259320

theorem negative_solution_exists (a b c x y : ℝ) :
  (a * x + b * y = c ∧ b * x + c * y = a ∧ c * x + a * y = b) ∧ (x < 0 ∧ y < 0) ↔ a + b + c = 0 :=
sorry

end negative_solution_exists_l259_259320


namespace convert_base_7_to_base_10_l259_259666

theorem convert_base_7_to_base_10 : 
  let base_7_number := 2 * 7^2 + 4 * 7^1 + 6 * 7^0 in
  base_7_number = 132 :=
by
  let base_7_number := 2 * 7^2 + 4 * 7^1 + 6 * 7^0
  have : base_7_number = 132 := by
    calc
      2 * 7^2 + 4 * 7^1 + 6 * 7^0 = 2 * 49 + 4 * 7 + 6 * 1 : by refl
                               ... = 98 + 28 + 6         : by simp
                               ... = 132                : by norm_num
  exact this

end convert_base_7_to_base_10_l259_259666


namespace compare_abc_l259_259047

noncomputable theory

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l259_259047


namespace complex_division_l259_259349

theorem complex_division (i : ℂ) (hi : i^2 = -1) : (2 * i) / (1 + i) = 1 + i :=
by
  sorry

end complex_division_l259_259349


namespace total_cost_production_l259_259611

variable (FC MC : ℕ) (n : ℕ)

theorem total_cost_production : FC = 12000 → MC = 200 → n = 20 → (FC + MC * n = 16000) :=
by
  intro hFC hMC hn
  sorry

end total_cost_production_l259_259611


namespace gcd_f_x_x_l259_259516

theorem gcd_f_x_x (x : ℕ) (h : ∃ k : ℕ, x = 35622 * k) :
  Nat.gcd ((3 * x + 4) * (5 * x + 6) * (11 * x + 9) * (x + 7)) x = 378 :=
by
  sorry

end gcd_f_x_x_l259_259516


namespace factorize_difference_of_squares_l259_259975

variable {R : Type} [CommRing R]

theorem factorize_difference_of_squares (a x y : R) : a * x^2 - a * y^2 = a * (x + y) * (x - y) :=
by
  sorry

end factorize_difference_of_squares_l259_259975


namespace gcd_euclidean_algorithm_example_l259_259435

theorem gcd_euclidean_algorithm_example : Nat.gcd 98 63 = 7 :=
by
  sorry

end gcd_euclidean_algorithm_example_l259_259435


namespace rabbit_prob_top_or_bottom_l259_259779

-- Define the probability function for the rabbit to hit the top or bottom border from a given point
noncomputable def prob_reach_top_or_bottom (start : ℕ × ℕ) (board_end : ℕ × ℕ) : ℚ :=
  sorry -- Detailed probability computation based on recursive and symmetry argument

-- The proof statement for the starting point (2, 3) on a rectangular board extending to (6, 5)
theorem rabbit_prob_top_or_bottom : prob_reach_top_or_bottom (2, 3) (6, 5) = 17 / 24 :=
  sorry

end rabbit_prob_top_or_bottom_l259_259779


namespace owen_sleep_hours_l259_259399

-- Define the time spent by Owen in various activities
def hours_work : ℕ := 6
def hours_chores : ℕ := 7
def total_hours_day : ℕ := 24

-- The proposition to be proven
theorem owen_sleep_hours : (total_hours_day - (hours_work + hours_chores) = 11) := by
  sorry

end owen_sleep_hours_l259_259399


namespace car_speed_l259_259258

theorem car_speed (distance time : ℝ) (h₁ : distance = 50) (h₂ : time = 5) : (distance / time) = 10 :=
by
  rw [h₁, h₂]
  norm_num

end car_speed_l259_259258


namespace tree_initial_height_example_l259_259434

-- The height of the tree at the time Tony planted it
def initial_tree_height (growth_rate final_height years : ℕ) : ℕ :=
  final_height - (growth_rate * years)

theorem tree_initial_height_example :
  initial_tree_height 5 29 5 = 4 :=
by
  -- This is where the proof would go, we use 'sorry' to indicate it's omitted.
  sorry

end tree_initial_height_example_l259_259434


namespace Brenda_weight_correct_l259_259301

-- Conditions
def MelWeight : ℕ := 70
def BrendaWeight : ℕ := 3 * MelWeight + 10

-- Proof problem
theorem Brenda_weight_correct : BrendaWeight = 220 := by
  sorry

end Brenda_weight_correct_l259_259301


namespace maximum_ab_is_40_l259_259389

noncomputable def maximum_ab (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : 5 * a + 8 * b = 80) : ℝ :=
  max (a * b) 40

theorem maximum_ab_is_40 {a b : ℝ} (h₀ : a > 0) (h₁ : b > 0) (h₂ : 5 * a + 8 * b = 80) : maximum_ab a b h₀ h₁ h₂ = 40 := 
by 
  sorry

end maximum_ab_is_40_l259_259389


namespace compare_abc_l259_259050

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  unfold a b c
  -- Proof goes here
  sorry

end compare_abc_l259_259050


namespace _l259_259443

lemma triangle_inequality_theorem (a b c : ℝ) : 
  a + b > c ∧ a + c > b ∧ b + c > a ↔ 
  (a > 0 ∧ b > 0 ∧ c > 0) := sorry

lemma no_triangle_1_2_3 : ¬ (1 + 2 > 3 ∧ 1 + 3 > 2 ∧ 2 + 3 > 1) := 
by simp [triangle_inequality_theorem]

lemma no_triangle_3_8_5 : ¬ (3 + 8 > 5 ∧ 3 + 5 > 8 ∧ 8 + 5 > 3) := 
by simp [triangle_inequality_theorem]

lemma no_triangle_4_5_10 : ¬ (4 + 5 > 10 ∧ 4 + 10 > 5 ∧ 5 + 10 > 4) := 
by simp [triangle_inequality_theorem]

lemma triangle_4_5_6 : 4 + 5 > 6 ∧ 4 + 6 > 5 ∧ 5 + 6 > 4 := 
by simp [triangle_inequality_theorem]

end _l259_259443


namespace bus_commutes_three_times_a_week_l259_259756

-- Define the commuting times
def bike_time := 30
def bus_time := bike_time + 10
def friend_time := bike_time * (1 - (2/3))
def total_weekly_time := 160

-- Define the number of times taking the bus as a variable
variable (b : ℕ)

-- The equation for total commuting time
def commuting_time_eq := bike_time + bus_time * b + friend_time = total_weekly_time

-- The proof statement: b should be equal to 3
theorem bus_commutes_three_times_a_week (h : commuting_time_eq b) : b = 3 := sorry

end bus_commutes_three_times_a_week_l259_259756


namespace exist_nat_nums_l259_259672

theorem exist_nat_nums :
  ∃ (a b c d : ℕ), (a / (b : ℚ) + c / (d : ℚ) = 1) ∧ (a / (d : ℚ) + c / (b : ℚ) = 2008) :=
sorry

end exist_nat_nums_l259_259672


namespace andy_profit_per_cake_l259_259290

-- Definitions based on the conditions
def cost_of_ingredients (cakes : ℕ) : ℕ := if cakes = 2 then 12 else 0
def cost_of_packaging_per_cake : ℕ := 1
def selling_price_per_cake : ℕ := 15

-- Theorem stating the profit made per cake
theorem andy_profit_per_cake : ∀ (cakes : ℕ), cakes = 2 → 
(cost_of_ingredients cakes / cakes + cost_of_packaging_per_cake) = 7 →
selling_price_per_cake - (cost_of_ingredients cakes / cakes + cost_of_packaging_per_cake) = 8 :=
by
  intros cakes h_cakes cost_hyp
  have h1 : cost_of_ingredients cakes / cakes = 12 / 2 :=
    by rw [h_cakes]; refl
  have h2 : (12 / 2 + cost_of_packaging_per_cake) = 6 + 1 :=
    by rw [h1]; refl
  have h3 : (6 + 1) = 7 :=
    by refl
  rw [← h3] at cost_hyp
  have h4 : selling_price_per_cake - 7 = 8 :=
    by refl
  exact h4

end andy_profit_per_cake_l259_259290


namespace remaining_money_l259_259727

def potato_cost : ℕ := 6 * 2
def tomato_cost : ℕ := 9 * 3
def cucumber_cost : ℕ := 5 * 4
def banana_cost : ℕ := 3 * 5
def total_cost : ℕ := potato_cost + tomato_cost + cucumber_cost + banana_cost
def initial_money : ℕ := 500

theorem remaining_money : initial_money - total_cost = 426 :=
by
  sorry

end remaining_money_l259_259727


namespace no_integer_n_for_fractions_l259_259074

theorem no_integer_n_for_fractions (n : ℤ) : ¬ (∃ n : ℤ, (n - 6) % 15 = 0 ∧ (n - 5) % 24 = 0) :=
by sorry

end no_integer_n_for_fractions_l259_259074


namespace initial_percentage_filled_l259_259257

theorem initial_percentage_filled {P : ℝ} 
  (h1 : 45 + (P / 100) * 100 = (3 / 4) * 100) : 
  P = 30 := by
  sorry

end initial_percentage_filled_l259_259257


namespace subtraction_correct_l259_259797

theorem subtraction_correct :
  2222222222222 - 1111111111111 = 1111111111111 := by
  sorry

end subtraction_correct_l259_259797


namespace conversion_rates_l259_259757

noncomputable def teamADailyConversionRate (a b : ℝ) := 1.2 * b
noncomputable def teamBDailyConversionRate (a b : ℝ) := b

theorem conversion_rates (total_area : ℝ) (b : ℝ) (h1 : total_area = 1500) (h2 : b = 50) 
    (h3 : teamADailyConversionRate 1500 b * b = 1.2) 
    (h4 : teamBDailyConversionRate 1500 b = b) 
    (h5 : (1500 / teamBDailyConversionRate 1500 b) - 5 = 1500 / teamADailyConversionRate 1500 b) :
  teamADailyConversionRate 1500 b = 60 ∧ teamBDailyConversionRate 1500 b = 50 := 
by
  sorry

end conversion_rates_l259_259757


namespace minimum_candy_kinds_l259_259850

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
     It turned out that between any two candies of the same kind, there is an even number of candies.
     Prove that the minimum number of kinds of candies that could be is 46. -/
theorem minimum_candy_kinds (n : ℕ) (candies : ℕ → ℕ) 
  (h_candies_length : ∀ i, i < 91 → candies i < n)
  (h_even_between : ∀ i j, i < j → candies i = candies j → even (j - i - 1)) :
  n ≥ 46 :=
sorry

end minimum_candy_kinds_l259_259850


namespace unique_cylinder_identical_l259_259964

noncomputable def unique_cylinder (V S : ℝ) : Prop :=
  ∀ r₁ r₂ h₁ h₂ : ℝ,
  (π * r₁^2 * h₁ = V ∧ 2 * π * r₁^2 + 2 * π * r₁ * h₁ = S) ∧
  (π * r₂^2 * h₂ = V ∧ 2 * π * r₂^2 + 2 * π * r₂ * h₂ = S) →
  (r₁ = r₂ ∧ h₁ = h₂)

-- This will state the main theorem
theorem unique_cylinder_identical (V S : ℝ) : unique_cylinder V S := 
  by
    sorry -- Proof goes here; it shows that cylinders with given V and S are identical.

end unique_cylinder_identical_l259_259964


namespace sum_of_squares_base_b_l259_259199

theorem sum_of_squares_base_b (b : ℕ) (h : (b + 4)^2 + (b + 8)^2 + (2 * b)^2 = 2 * b^3 + 8 * b^2 + 5 * b) :
  (4 * b + 12 : ℕ) = 62 :=
by
  sorry

end sum_of_squares_base_b_l259_259199


namespace relation_among_abc_l259_259042

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem relation_among_abc : a > c ∧ c > b := 
by {
    -- proof steps go here, but we use sorry for now
    sorry
}

end relation_among_abc_l259_259042


namespace sabi_share_removed_l259_259767

theorem sabi_share_removed :
  ∀ (N S M x : ℝ), N - 5 = 2 * (S - x) / 8 ∧ S - x = 4 * (6 * (M - 4)) / 16 ∧ M = 102 ∧ N + S + M = 1100 
  → x = 829.67 := by
  sorry

end sabi_share_removed_l259_259767


namespace largest_perimeter_polygons_meeting_at_A_l259_259618

theorem largest_perimeter_polygons_meeting_at_A
  (n : ℕ) 
  (r : ℝ)
  (h1 : n ≥ 3)
  (h2 : 2 * 180 * (n - 2) / n + 60 = 360) :
  2 * n * 2 = 24 := 
by
  sorry

end largest_perimeter_polygons_meeting_at_A_l259_259618


namespace base7_to_base10_conversion_l259_259664

theorem base7_to_base10_conversion :
  let base7_246 := 2 * 7^2 + 4 * 7^1 + 6 * 7^0
  in base7_246 = 132 :=
by
  let base7_246 := 2 * 7^2 + 4 * 7^1 + 6 * 7^0
  show base7_246 = 132
  -- The proof steps would go here
  sorry

end base7_to_base10_conversion_l259_259664


namespace min_value_a_plus_b_l259_259817

open Real

theorem min_value_a_plus_b (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h : 1 / a + 2 / b = 1) :
  a + b = 3 + 2 * sqrt 2 :=
sorry

end min_value_a_plus_b_l259_259817


namespace option_B_coplanar_l259_259285

-- Define the three vectors in Option B.
def a : ℝ × ℝ × ℝ := (1, 2, -3)
def b : ℝ × ℝ × ℝ := (-2, -4, 6)
def c : ℝ × ℝ × ℝ := (1, 0, 5)

-- Define the coplanarity condition for vectors a, b, and c.
def coplanar (a b c : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = k • a

-- Prove that the vectors in Option B are coplanar.
theorem option_B_coplanar : coplanar a b c :=
sorry

end option_B_coplanar_l259_259285


namespace chrysler_building_floors_l259_259082

theorem chrysler_building_floors (L : ℕ) (Chrysler : ℕ) (h1 : Chrysler = L + 11)
  (h2 : L + Chrysler = 35) : Chrysler = 23 :=
by {
  --- proof would be here
  sorry
}

end chrysler_building_floors_l259_259082


namespace conic_section_is_hyperbola_l259_259499

noncomputable def is_hyperbola (x y : ℝ) : Prop :=
  (x - 4) ^ 2 = 9 * (y + 3) ^ 2 + 27

theorem conic_section_is_hyperbola : ∀ x y : ℝ, is_hyperbola x y → "H" = "H" := sorry

end conic_section_is_hyperbola_l259_259499


namespace circle_possible_m_values_l259_259369

theorem circle_possible_m_values (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 + m * x - m * y + 2 = 0) ↔ m > 2 ∨ m < -2 :=
by
  sorry

end circle_possible_m_values_l259_259369


namespace integer_solution_l259_259807

theorem integer_solution (n m : ℤ) (h : (n + 2)^4 - n^4 = m^3) : (n = -1 ∧ m = 0) :=
by
  sorry

end integer_solution_l259_259807


namespace statement_correctness_l259_259307

def correct_statements := [4, 8]
def incorrect_statements := [1, 2, 3, 5, 6, 7]

theorem statement_correctness :
  correct_statements = [4, 8] ∧ incorrect_statements = [1, 2, 3, 5, 6, 7] :=
  by sorry

end statement_correctness_l259_259307


namespace problem_l259_259765

theorem problem (A B : ℝ) (h₀ : 0 < A) (h₁ : 0 < B) (h₂ : B > A) (n c : ℝ) 
  (h₃ : B = A * (1 + n / 100)) (h₄ : A = B * (1 - c / 100)) :
  A * Real.sqrt (100 + n) = B * Real.sqrt (100 - c) :=
by
  sorry

end problem_l259_259765


namespace sum_of_cube_faces_l259_259607

theorem sum_of_cube_faces (a b c d e f : ℕ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f) 
    (h_eq_sum: (a * b * c) + (a * e * c) + (a * b * f) + (a * e * f) + (d * b * c) + (d * e * c) + (d * b * f) + (d * e * f) = 1089) :
    a + b + c + d + e + f = 31 := 
by
  sorry

end sum_of_cube_faces_l259_259607


namespace cathy_remaining_money_l259_259659

noncomputable def remaining_money (initial : ℝ) (dad : ℝ) (book : ℝ) (cab_percentage : ℝ) (food_percentage : ℝ) : ℝ :=
  let money_mom := 2 * dad
  let total_money := initial + dad + money_mom
  let remaining_after_book := total_money - book
  let cab_cost := cab_percentage * remaining_after_book
  let food_budget := food_percentage * total_money
  let dinner_cost := 0.5 * food_budget
  remaining_after_book - cab_cost - dinner_cost

theorem cathy_remaining_money :
  remaining_money 12 25 15 0.03 0.4 = 52.44 :=
by
  sorry

end cathy_remaining_money_l259_259659


namespace simplify_fraction_multiplication_l259_259736

theorem simplify_fraction_multiplication:
  (101 / 5050) * 50 = 1 := by
  sorry

end simplify_fraction_multiplication_l259_259736


namespace surface_area_increase_l259_259960

theorem surface_area_increase :
  let l := 4
  let w := 3
  let h := 2
  let side_cube := 1
  let original_surface := 2 * (l * w + l * h + w * h)
  let additional_surface := 6 * side_cube * side_cube
  let new_surface := original_surface + additional_surface
  new_surface = original_surface + 6 :=
by
  sorry

end surface_area_increase_l259_259960


namespace difference_of_interchanged_digits_l259_259930

theorem difference_of_interchanged_digits (X Y : ℕ) (h : X - Y = 5) : (10 * X + Y) - (10 * Y + X) = 45 :=
by
  sorry

end difference_of_interchanged_digits_l259_259930


namespace count_inverses_mod_11_l259_259544

theorem count_inverses_mod_11 : 
  (Finset.filter (λ x : ℕ, Nat.coprime x 11) (Finset.range 11)).card = 10 := 
by
  sorry

end count_inverses_mod_11_l259_259544


namespace annual_earning_difference_l259_259379

def old_hourly_wage := 16
def old_weekly_hours := 25
def new_hourly_wage := 20
def new_weekly_hours := 40
def weeks_per_year := 52

def old_weekly_earnings := old_hourly_wage * old_weekly_hours
def new_weekly_earnings := new_hourly_wage * new_weekly_hours

def old_annual_earnings := old_weekly_earnings * weeks_per_year
def new_annual_earnings := new_weekly_earnings * weeks_per_year

theorem annual_earning_difference:
  new_annual_earnings - old_annual_earnings = 20800 := by
  sorry

end annual_earning_difference_l259_259379


namespace f_at_4_l259_259591

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f (x-1) = g_inv (x-3)
axiom h2 : ∀ x : ℝ, g_inv (g x) = x
axiom h3 : ∀ x : ℝ, g (g_inv x) = x
axiom h4 : g 5 = 2005

theorem f_at_4 : f 4 = 2008 :=
by
  sorry

end f_at_4_l259_259591


namespace math_proof_problem_l259_259153

theorem math_proof_problem :
  2018^2019^2020 % 11 = 5 :=
  by
    have fermat_lt : ∀ a p : ℕ, Nat.Prime p ∧ (a % p ≠ 0) → a ^ (p - 1) % p = 1 :=
      sorry -- Fermat's Little theorem
    have cond1 : 2018 % 11 = 5 := by norm_num
    have cond2 : 2019 % 10 = 9 := by norm_num  -- Since (-1 ≡ 9 mod 10)
    have exp_simpl : 2019^2020 % 10 = 1 := 
      by
        have p1 : 2019 % 10 = 9  := cond2
        have p2 : 9^2020 % 10 = 1 :=
          by
            sorry -- Use cyclicity of powers mod 10: (-1 for odd exponent, 1 for even)
        exact p2
    have base_simpl : 2018 ^ 2019 ^ 2020 % 11 = 5 :=
      by 
        have s1 : 2018 % 11 = 5 := cond1
        have e1 : 2019^2020 % 10 = 1 := exp_simpl
        have final_result : 5 ^ 1 % 11 = 5 := by norm_num
        exact final_result
    exact base_simpl

end math_proof_problem_l259_259153


namespace raviraj_distance_home_l259_259950

theorem raviraj_distance_home :
  let origin := (0, 0)
  let after_south := (0, -20)
  let after_west := (-10, -20)
  let after_north := (-10, 0)
  let final_pos := (-30, 0)
  Real.sqrt ((final_pos.1 - origin.1)^2 + (final_pos.2 - origin.2)^2) = 30 :=
by
  sorry

end raviraj_distance_home_l259_259950


namespace minimum_candy_kinds_l259_259857

theorem minimum_candy_kinds (n : ℕ) (h_n : n = 91) (even_spacing : ∀ i j : ℕ, i < j → i < n → j < n → (∀ k : ℕ, i < k ∧ k < j → k % 2 = 1)) : 46 ≤ n / 2 :=
by
  rw h_n
  have : 46 ≤ 91 / 2 := nat.le_of_lt (by norm_num)
  exact this

end minimum_candy_kinds_l259_259857


namespace flora_needs_more_daily_l259_259982

-- Definitions based on conditions
def totalMilk : ℕ := 105   -- Total milk requirement in gallons
def weeks : ℕ := 3         -- Total weeks
def daysInWeek : ℕ := 7    -- Days per week
def floraPlan : ℕ := 3     -- Flora's planned gallons per day

-- Proof statement
theorem flora_needs_more_daily : (totalMilk / (weeks * daysInWeek)) - floraPlan = 2 := 
by
  sorry

end flora_needs_more_daily_l259_259982


namespace min_candy_kinds_l259_259848

theorem min_candy_kinds (n : ℕ) (m : ℕ) (h_n : n = 91) 
  (h_even : ∀ i j (h_i : i < j) (h_k : j < m), (i ≠ j) → even (j - i - 1)) : 
  m ≥ 46 :=
sorry

end min_candy_kinds_l259_259848


namespace peter_remaining_money_l259_259723

theorem peter_remaining_money (initial_money : ℕ) 
                             (potato_cost_per_kilo : ℕ) (potato_kilos : ℕ)
                             (tomato_cost_per_kilo : ℕ) (tomato_kilos : ℕ)
                             (cucumber_cost_per_kilo : ℕ) (cucumber_kilos : ℕ)
                             (banana_cost_per_kilo : ℕ) (banana_kilos : ℕ) :
  initial_money = 500 →
  potato_cost_per_kilo = 2 → potato_kilos = 6 →
  tomato_cost_per_kilo = 3 → tomato_kilos = 9 →
  cucumber_cost_per_kilo = 4 → cucumber_kilos = 5 →
  banana_cost_per_kilo = 5 → banana_kilos = 3 →
  initial_money - (potato_cost_per_kilo * potato_kilos + 
                   tomato_cost_per_kilo * tomato_kilos +
                   cucumber_cost_per_kilo * cucumber_kilos +
                   banana_cost_per_kilo * banana_kilos) = 426 := by
  sorry

end peter_remaining_money_l259_259723


namespace unique_subset_empty_set_l259_259701

def discriminant (a : ℝ) : ℝ := 4 - 4 * a^2

theorem unique_subset_empty_set (a : ℝ) :
  (∀ (x : ℝ), ¬(a * x^2 + 2 * x + a = 0)) ↔ (a > 1 ∨ a < -1) :=
by
  sorry

end unique_subset_empty_set_l259_259701


namespace binomial_expansion_judgments_l259_259157

theorem binomial_expansion_judgments :
  (∃ n : ℕ, n > 0 ∧ ∃ r : ℕ, n = 4 * r) ∧
  (∃ n : ℕ, n > 0 ∧ ∃ r : ℕ, n = 4 * r + 3) :=
by
  sorry

end binomial_expansion_judgments_l259_259157


namespace volume_of_pyramid_l259_259918

noncomputable def volume_of_pyramid_QEFGH : ℝ := 
  let EF := 10
  let FG := 3
  let base_area := EF * FG
  let height := 9
  (1/3) * base_area * height

theorem volume_of_pyramid {EF FG : ℝ} (hEF : EF = 10) (hFG : FG = 3)
  (QE_perpendicular_EF : true) (QE_perpendicular_EH : true) (QE_height : QE = 9) :
  volume_of_pyramid_QEFGH = 90 := by
  sorry

end volume_of_pyramid_l259_259918


namespace find_fractions_l259_259490

open Function

-- Define the set and the condition that all numbers must be used precisely once
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define what it means for fractions to multiply to 1 within the set
def fractions_mul_to_one (a b c d e f : ℕ) : Prop :=
  (a * c * e) = (b * d * f)

-- Define irreducibility condition for a fraction a/b
def irreducible_fraction (a b : ℕ) := 
  Nat.gcd a b = 1

-- Final main problem statement
theorem find_fractions :
  ∃ (a b c d e f : ℕ) (h₁ : a ∈ S) (h₂ : b ∈ S) (h₃ : c ∈ S) (h₄ : d ∈ S) (h₅ : e ∈ S) (h₆ : f ∈ S),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
  d ≠ e ∧ d ≠ f ∧ 
  e ≠ f ∧
  irreducible_fraction a b ∧ irreducible_fraction c d ∧ irreducible_fraction e f ∧
  fractions_mul_to_one a b c d e f := 
sorry

end find_fractions_l259_259490


namespace value_of_x_plus_y_l259_259840

theorem value_of_x_plus_y (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 4) (h3 : x * y > 0) : x + y = 7 ∨ x + y = -7 :=
by
  sorry

end value_of_x_plus_y_l259_259840


namespace find_x_l259_259721

theorem find_x (x : ℝ) : 0.20 * x - (1 / 3) * (0.20 * x) = 24 → x = 180 :=
by
  intro h
  sorry

end find_x_l259_259721


namespace reckha_code_count_l259_259912

theorem reckha_code_count :
  let total_codes := 1000
  let codes_with_one_digit_different := 27
  let permutations_of_045 := 2
  let original_code := 1
  total_codes - codes_with_one_digit_different - permutations_of_045 - original_code = 970 :=
by
  let total_codes := 1000
  let codes_with_one_digit_different := 27
  let permutations_of_045 := 2
  let original_code := 1
  show total_codes - codes_with_one_digit_different - permutations_of_045 - original_code = 970
  sorry

end reckha_code_count_l259_259912


namespace count_ordered_triples_l259_259386

def S := Finset.range 20

def succ (a b : ℕ) : Prop := 
  (0 < a - b ∧ a - b ≤ 10) ∨ (b - a > 10)

theorem count_ordered_triples 
  (h : ∃ n : ℕ, (S.card = 20) ∧
                (∀ x y z : ℕ, 
                   x ∈ S → y ∈ S → z ∈ S →
                   (succ x y) → (succ y z) → (succ z x) →
                   n = 1260)) : True := sorry

end count_ordered_triples_l259_259386


namespace pages_to_read_tomorrow_l259_259895

theorem pages_to_read_tomorrow (total_pages : ℕ) 
                              (days : ℕ)
                              (pages_read_yesterday : ℕ)
                              (pages_read_today : ℕ)
                              (yesterday_diff : pages_read_today = pages_read_yesterday - 5)
                              (total_pages_eq : total_pages = 100)
                              (days_eq : days = 3)
                              (yesterday_eq : pages_read_yesterday = 35) : 
                              ∃ pages_read_tomorrow,  pages_read_tomorrow = total_pages - (pages_read_yesterday + pages_read_today) := 
                              by
  use 35
  sorry

end pages_to_read_tomorrow_l259_259895


namespace distance_house_to_market_l259_259649

-- Define the conditions
def distance_house_to_school := 50
def total_distance_walked := 140

-- Define the question as a theorem with the correct answer
theorem distance_house_to_market : 
  ∀ (house_to_school school_to_house total_distance market : ℕ), 
  house_to_school = distance_house_to_school →
  school_to_house = distance_house_to_school →
  total_distance = total_distance_walked →
  house_to_school + school_to_house + market = total_distance →
  market = 40 :=
by
  intros house_to_school school_to_house total_distance market 
  intro h1 h2 h3 h4
  sorry

end distance_house_to_market_l259_259649


namespace max_xy_l259_259587

theorem max_xy (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 5 * x + 6 * y < 90) :
  xy * (90 - 5 * x - 6 * y) ≤ 900 := by
  sorry

end max_xy_l259_259587


namespace number_of_people_l259_259137

def totalCups : ℕ := 10
def cupsPerPerson : ℕ := 2

theorem number_of_people {n : ℕ} (h : n = totalCups / cupsPerPerson) : n = 5 := by
  sorry

end number_of_people_l259_259137


namespace initial_amount_of_money_l259_259200

-- Define the conditions
def spent_on_sweets : ℝ := 35.25
def given_to_each_friend : ℝ := 25.20
def num_friends : ℕ := 2
def amount_left : ℝ := 114.85

-- Define the calculated amount given to friends
def total_given_to_friends : ℝ := given_to_each_friend * num_friends

-- State the theorem to prove the initial amount of money
theorem initial_amount_of_money :
  spent_on_sweets + total_given_to_friends + amount_left = 200.50 :=
by 
  -- proof goes here
  sorry

end initial_amount_of_money_l259_259200


namespace total_yards_thrown_l259_259069

-- Definitions for the conditions
def distance_50_degrees : ℕ := 20
def distance_80_degrees : ℕ := distance_50_degrees * 2

def throws_on_saturday : ℕ := 20
def throws_on_sunday : ℕ := 30

def headwind_penalty : ℕ := 5
def tailwind_bonus : ℕ := 10

-- Theorem for the total yards thrown in two days
theorem total_yards_thrown :
  ((distance_50_degrees - headwind_penalty) * throws_on_saturday) + 
  ((distance_80_degrees + tailwind_bonus) * throws_on_sunday) = 1800 :=
by
  sorry

end total_yards_thrown_l259_259069


namespace linear_condition_l259_259933

theorem linear_condition (m : ℝ) : ¬ (m = 2) ↔ (∃ f : ℝ → ℝ, ∀ x, f x = (m - 2) * x + 2) :=
by
  sorry

end linear_condition_l259_259933


namespace count_inverses_mod_11_l259_259529

theorem count_inverses_mod_11 : (∃ n : ℕ, n = 10) :=
  have h : ∀ a ∈ finset.range 11, nat.gcd a 11 = 1 -> a ≠ 0 := by 
    intro a ha h1,
    apply (ne_of_lt (by linarith : a < 11)),
    apply nat.pos_of_ne_zero,
    intro hz,
    rw [hz, nat.gcd_zero_left] at h1,
    exact nat.prime.not_dvd_one (nat.prime_iff.2 ⟨sorry, sorry⟩) 11 h1,
  sorry

end count_inverses_mod_11_l259_259529


namespace negative_solution_iff_sum_zero_l259_259316

theorem negative_solution_iff_sum_zero (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔
  a + b + c = 0 :=
by
  sorry

end negative_solution_iff_sum_zero_l259_259316


namespace Janet_previous_movie_length_l259_259885

theorem Janet_previous_movie_length (L : ℝ) (H1 : 1.60 * L = 1920 / 100) : L / 60 = 0.20 :=
by
  sorry

end Janet_previous_movie_length_l259_259885


namespace find_original_number_l259_259691

theorem find_original_number (h1 : 268 * 74 = 19732) (h2 : 2.68 * x = 1.9832) : x = 0.74 :=
sorry

end find_original_number_l259_259691


namespace work_problem_l259_259632

theorem work_problem 
  (A_real : ℝ)
  (B_days : ℝ := 16)
  (C_days : ℝ := 16)
  (ABC_days : ℝ := 4)
  (H_b : (1 / B_days) = 1 / 16)
  (H_c : (1 / C_days) = 1 / 16)
  (H_abc : (1 / A_real + 1 / B_days + 1 / C_days) = 1 / ABC_days) : 
  A_real = 8 := 
sorry

end work_problem_l259_259632


namespace num_integers_with_inverse_mod_11_l259_259539

theorem num_integers_with_inverse_mod_11 : 
  (Finset.card (Finset.filter (λ x : ℕ, ∃ y : ℕ, x * y % 11 = 1) (Finset.range 11))) = 10 := 
by 
  sorry

end num_integers_with_inverse_mod_11_l259_259539


namespace calc_expr_value_l259_259657

theorem calc_expr_value : (0.5 ^ 4) / (0.05 ^ 2.5) = 559.06 := 
by 
  sorry

end calc_expr_value_l259_259657


namespace target1_target2_l259_259515

variable (α : ℝ)

-- Define the condition
def tan_alpha := Real.tan α = 2

-- State the first target with the condition considered
theorem target1 (h : tan_alpha α) : 
  (Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1 / 6 := by
  sorry

-- State the second target with the condition considered
theorem target2 (h : tan_alpha α) : 
  4 * Real.sin α ^ 2 - 3 * Real.sin α * Real.cos α - 5 * Real.cos α ^ 2 = 1 := by
  sorry

end target1_target2_l259_259515


namespace irreducible_fractions_product_one_l259_259484

theorem irreducible_fractions_product_one : ∃ (a b c d e f : ℕ),
  {a, b, c, d, e, f}.Subset {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  {a, b, c, d, e, f}.card = 6 ∧
  ∃ (f1_num f1_den f2_num f2_den f3_num f3_den : ℕ), 
    (f1_num ≠ f1_den ∧ coprime f1_num f1_den ∧ f1_num ∈ {a, b, c, d, e, f} ∧ f1_den ∈ {a, b, c, d, e, f} ∧ 
    f2_num ≠ f2_den ∧ coprime f2_num f2_den ∧ f2_num ∈ {a, b, c, d, e, f} ∧ f2_den ∈ {a, b, c, d, e, f} ∧ 
    f3_num ≠ f3_den ∧ coprime f3_num f3_den ∧ f3_num ∈ {a, b, c, d, e, f} ∧ f3_den ∈ {a, b, c, d, e, f} ∧ 
    (f1_num * f2_num * f3_num) = (f1_den * f2_den * f3_den)) :=
sorry

end irreducible_fractions_product_one_l259_259484


namespace min_candy_kinds_l259_259846

theorem min_candy_kinds (n : ℕ) (m : ℕ) (h_n : n = 91) 
  (h_even : ∀ i j (h_i : i < j) (h_k : j < m), (i ≠ j) → even (j - i - 1)) : 
  m ≥ 46 :=
sorry

end min_candy_kinds_l259_259846


namespace class_size_l259_259615

theorem class_size (n : ℕ) (h₁ : 60 - n > 0) (h₂ : (60 - n) / 2 = n) : n = 20 :=
by
  sorry

end class_size_l259_259615


namespace hyperbola_eccentricity_ratio_hyperbola_condition_l259_259761

-- Part (a)
theorem hyperbola_eccentricity_ratio
  (a b c : ℝ) (h1 : c^2 = a^2 + b^2)
  (x0 y0 : ℝ) 
  (P : ℝ × ℝ) (h2 : P = (x0, y0))
  (F : ℝ × ℝ) (h3 : F = (c, 0))
  (D : ℝ) (h4 : D = a^2 / c)
  (d_PF : ℝ) (h5 : d_PF = ( (x0 - c)^2 + y0^2 )^(1/2))
  (d_PD : ℝ) (h6 : d_PD = |x0 - a^2 / c|)
  (e : ℝ) (h7 : e = c / a) :
  d_PF / d_PD = e :=
sorry

-- Part (b)
theorem hyperbola_condition
  (F_l : ℝ × ℝ) (h1 : F_l = (0, k))
  (X_l : ℝ × ℝ) (h2 : X_l = (x, l))
  (d_XF : ℝ) (h3 : d_XF = (x^2 + y^2)^(1/2))
  (d_Xl : ℝ) (h4 : d_Xl = |x - k|)
  (e : ℝ) (h5 : e > 1)
  (h6 : d_XF / d_Xl = e) :
  ∃ a b : ℝ, (x / a)^2 - (y / b)^2 = 1 :=
sorry

end hyperbola_eccentricity_ratio_hyperbola_condition_l259_259761


namespace find_g_neg_2_l259_259824

-- Definitions
variable {R : Type*} [CommRing R] [Inhabited R]
variable (f g : R → R)

-- Conditions
axiom odd_y (x : R) : f (-x) + 2 * x^2 = -(f x + 2 * x^2)
axiom definition_g (x : R) : g x = f x + 1
axiom value_f_2 : f 2 = 2

-- Goal
theorem find_g_neg_2 : g (-2) = -17 :=
by
  sorry

end find_g_neg_2_l259_259824


namespace investment_in_stocks_l259_259136

theorem investment_in_stocks (T b s : ℝ) (h1 : T = 200000) (h2 : s = 5 * b) (h3 : T = b + s) :
  s = 166666.65 :=
by sorry

end investment_in_stocks_l259_259136


namespace ratio_of_perimeters_l259_259098

theorem ratio_of_perimeters (d : ℝ) (s1 s2 P1 P2 : ℝ) (h1 : d^2 = 2 * s1^2)
  (h2 : (3 * d)^2 = 2 * s2^2) (h3 : P1 = 4 * s1) (h4 : P2 = 4 * s2) :
  P2 / P1 = 3 := 
by sorry

end ratio_of_perimeters_l259_259098


namespace count_of_inverses_mod_11_l259_259534

theorem count_of_inverses_mod_11 : (Finset.filter (λ a : ℕ, ∃ b : ℕ, (a * b) % 11 = 1) (Finset.range 11)).card = 10 := 
sorry

end count_of_inverses_mod_11_l259_259534


namespace unique_n_l259_259242

theorem unique_n : ∃ n : ℕ, 0 < n ∧ n^3 % 1000 = n ∧ ∀ m : ℕ, 0 < m ∧ m^3 % 1000 = m → m = n :=
by
  sorry

end unique_n_l259_259242


namespace rosa_bonheur_birth_day_l259_259741

/--
Given that Rosa Bonheur's 210th birthday was celebrated on a Wednesday,
prove that she was born on a Sunday.
-/
theorem rosa_bonheur_birth_day :
  let anniversary_year := 2022
  let birth_year := 1812
  let total_years := anniversary_year - birth_year
  let leap_years := (total_years / 4) - (total_years / 100) + (total_years / 400)
  let regular_years := total_years - leap_years
  let day_shifts := regular_years + 2 * leap_years
  (3 - day_shifts % 7) % 7 = 0 := 
sorry

end rosa_bonheur_birth_day_l259_259741


namespace dice_probability_divisible_by_three_ge_one_fourth_l259_259523

theorem dice_probability_divisible_by_three_ge_one_fourth
  (p q r : ℝ) 
  (h1 : 0 ≤ p) (h2 : 0 ≤ q) (h3 : 0 ≤ r) 
  (h4 : p + q + r = 1) : 
  p^3 + q^3 + r^3 + 6 * p * q * r ≥ 1 / 4 :=
sorry

end dice_probability_divisible_by_three_ge_one_fourth_l259_259523


namespace part_1_solution_part_2_solution_l259_259804

def f (x : ℝ) : ℝ := |x - 1| + |2 * x + 2|

theorem part_1_solution (x : ℝ) : f x < 3 ↔ -4 / 3 < x ∧ x < 0 :=
by
  sorry

theorem part_2_solution (a : ℝ) : (∀ x, ¬ (f x < a)) → a ≤ 2 :=
by
  sorry

end part_1_solution_part_2_solution_l259_259804


namespace double_square_area_l259_259793

theorem double_square_area (a k : ℝ) (h : (k * a) ^ 2 = 2 * a ^ 2) : k = Real.sqrt 2 := 
by 
  -- Our goal is to prove that k = sqrt(2)
  sorry

end double_square_area_l259_259793


namespace find_duplicate_page_l259_259125

theorem find_duplicate_page (n p : ℕ) (h : (n * (n + 1) / 2) + p = 3005) : p = 2 := 
sorry

end find_duplicate_page_l259_259125


namespace jane_current_age_l259_259284

theorem jane_current_age (J : ℕ) (h1 : ∀ t : ℕ, t = 13 → 25 + t = 2 * (J + t)) : J = 6 :=
by {
  sorry
}

end jane_current_age_l259_259284


namespace subtraction_of_negatives_l259_259112

theorem subtraction_of_negatives : (-1) - (-4) = 3 :=
by
  -- Proof goes here.
  sorry

end subtraction_of_negatives_l259_259112


namespace six_units_away_has_two_solutions_l259_259179

-- Define point A and its position on the number line
def A_position : ℤ := -3

-- Define the condition for a point x being 6 units away from point A
def is_6_units_away (x : ℤ) : Prop := abs (x + 3) = 6

-- The theorem stating that if x is 6 units away from -3, then x must be either 3 or -9
theorem six_units_away_has_two_solutions (x : ℤ) (h : is_6_units_away x) : x = 3 ∨ x = -9 := by
  sorry

end six_units_away_has_two_solutions_l259_259179


namespace problem_statement_l259_259693

noncomputable def equation_of_altitude (A B C: (ℝ × ℝ)): (ℝ × ℝ × ℝ) :=
by
  sorry

theorem problem_statement :
  let A := (-1, 4)
  let B := (-2, -1)
  let C := (2, 3)
  equation_of_altitude A B C = (1, 1, -3) ∧
  |1 / 2 * (4 - (-1)) * 4| = 8 :=
by
  sorry

end problem_statement_l259_259693


namespace num_of_laborers_is_24_l259_259413

def average_salary_all (L S : Nat) (avg_salary_ls : Nat) (avg_salary_l : Nat) (avg_salary_s : Nat) : Prop :=
  (avg_salary_l * L + avg_salary_s * S) / (L + S) = avg_salary_ls

def average_salary_supervisors (S : Nat) (avg_salary_s : Nat) : Prop :=
  (avg_salary_s * S) / S = avg_salary_s

theorem num_of_laborers_is_24 :
  ∀ (L S : Nat) (avg_salary_ls avg_salary_l avg_salary_s : Nat),
    average_salary_all L S avg_salary_ls avg_salary_l avg_salary_s →
    average_salary_supervisors S avg_salary_s →
    S = 6 → avg_salary_ls = 1250 → avg_salary_l = 950 → avg_salary_s = 2450 →
    L = 24 :=
by
  intros L S avg_salary_ls avg_salary_l avg_salary_s h1 h2 h3 h4 h5 h6
  sorry

end num_of_laborers_is_24_l259_259413


namespace perpendicular_vectors_l259_259234

theorem perpendicular_vectors (b : ℚ) : 
  (4 * b - 15 = 0) → (b = 15 / 4) :=
by
  intro h
  sorry

end perpendicular_vectors_l259_259234


namespace max_discardable_grapes_l259_259255

theorem max_discardable_grapes (n : ℕ) (k : ℕ) (h : k = 8) : 
  ∃ m : ℕ, m < k ∧ (∀ q : ℕ, q * k + m = n) ∧ m = 7 :=
by
  sorry

end max_discardable_grapes_l259_259255


namespace find_x_l259_259314

theorem find_x (x : ℚ) (h : ⌊x⌋ + x = 15/4) : x = 15/4 := by
  sorry

end find_x_l259_259314


namespace max_f_gt_sqrt2_f_is_periodic_f_pi_shift_pos_l259_259831

noncomputable def f (x : ℝ) := Real.sin (Real.cos x) + Real.cos (Real.sin x)

theorem max_f_gt_sqrt2 : (∃ x : ℝ, f x > Real.sqrt 2) :=
sorry

theorem f_is_periodic : ∀ x : ℝ, f (x - 2 * Real.pi) = f x :=
sorry

theorem f_pi_shift_pos : ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi → f (x + Real.pi) > 0 :=
sorry

end max_f_gt_sqrt2_f_is_periodic_f_pi_shift_pos_l259_259831


namespace find_cost_price_l259_259958

-- Conditions
def initial_cost_price (C : ℝ) : Prop :=
  let SP := 1.07 * C
  let NCP := 0.92 * C
  let NSP := SP - 3
  NSP = 1.0304 * C

-- The problem is to prove the initial cost price C given the conditions
theorem find_cost_price (C : ℝ) (h : initial_cost_price C) : C = 75.7575 := 
  sorry

end find_cost_price_l259_259958


namespace perimeter_of_figure_composed_of_squares_l259_259710

theorem perimeter_of_figure_composed_of_squares
  (n : ℕ)
  (side_length : ℝ)
  (square_perimeter : ℝ := 4 * side_length)
  (total_squares : ℕ := 7)
  (total_perimeter_if_independent : ℝ := square_perimeter * total_squares)
  (meet_at_vertices : ∀ i j : ℕ, i ≠ j → ∀ (s1 s2 : ℝ × ℝ), s1 ≠ s2 → ¬(s1 = s2))
  : total_perimeter_if_independent = 28 :=
by sorry

end perimeter_of_figure_composed_of_squares_l259_259710


namespace intersection_x_val_l259_259750

theorem intersection_x_val (x y : ℝ) (h1 : y = 3 * x - 24) (h2 : 5 * x + 2 * y = 102) : x = 150 / 11 :=
by
  sorry

end intersection_x_val_l259_259750


namespace binom_n_plus_one_n_l259_259096

theorem binom_n_plus_one_n (n : ℕ) (h : 0 < n) : Nat.choose (n + 1) n = n + 1 := 
sorry

end binom_n_plus_one_n_l259_259096


namespace product_of_terms_in_geometric_sequence_l259_259351

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a (n + m) = a n * a m

noncomputable def roots_of_quadratic (a b c : ℝ) (r1 r2 : ℝ) : Prop :=
r1 * r2 = c

theorem product_of_terms_in_geometric_sequence
  (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : roots_of_quadratic 1 (-4) 3 (a 5) (a 7)) :
  a 2 * a 10 = 3 :=
sorry

end product_of_terms_in_geometric_sequence_l259_259351


namespace average_gas_mileage_round_trip_l259_259637

theorem average_gas_mileage_round_trip :
  let distance_to_conference := 150
  let distance_return_trip := 150
  let mpg_sedan := 25
  let mpg_hybrid := 40
  let total_distance := distance_to_conference + distance_return_trip
  let gas_used_sedan := distance_to_conference / mpg_sedan
  let gas_used_hybrid := distance_return_trip / mpg_hybrid
  let total_gas_used := gas_used_sedan + gas_used_hybrid
  let average_gas_mileage := total_distance / total_gas_used
  average_gas_mileage = 31 := by
    sorry

end average_gas_mileage_round_trip_l259_259637


namespace sum_of_largest_and_smallest_l259_259940

theorem sum_of_largest_and_smallest (n : ℕ) (h : 6 * n + 15 = 105) : (n + (n + 5) = 35) :=
by
  sorry

end sum_of_largest_and_smallest_l259_259940


namespace average_score_is_8_9_l259_259275

-- Define the scores and their frequencies
def scores : List ℝ := [7.5, 8.5, 9, 10]
def frequencies : List ℕ := [2, 2, 3, 3]

-- Express the condition that the total number of shots is 10
def total_shots : ℕ := frequencies.sum

-- Calculate the weighted sum of the scores
def weighted_sum (scores : List ℝ) (frequencies : List ℕ) : ℝ :=
  (List.zip scores frequencies).foldl (λ acc (sc, freq) => acc + (sc * freq)) 0

-- Prove that the average score is 8.9
theorem average_score_is_8_9 :
  total_shots = 10 →
  weighted_sum scores frequencies / total_shots = 8.9 :=
by
  intros h_total_shots
  sorry

end average_score_is_8_9_l259_259275


namespace different_books_l259_259231

-- Definitions for the conditions
def tony_books : ℕ := 23
def dean_books : ℕ := 12
def breanna_books : ℕ := 17
def common_books_tony_dean : ℕ := 3
def common_books_all : ℕ := 1

-- Prove the total number of different books they have read is 47
theorem different_books : (tony_books + dean_books - common_books_tony_dean + breanna_books - 2 * common_books_all) = 47 :=
by
  sorry

end different_books_l259_259231


namespace problem1_problem2_problem3_l259_259409

theorem problem1 : (x : ℝ) → ((x + 1)^2 = 9 → (x = -4 ∨ x = 2)) :=
by
  intro x
  sorry

theorem problem2 : (x : ℝ) → (x^2 - 12*x - 4 = 0 → (x = 6 + 2*Real.sqrt 10 ∨ x = 6 - 2*Real.sqrt 10)) :=
by
  intro x
  sorry

theorem problem3 : (x : ℝ) → (3*(x - 2)^2 = x*(x - 2) → (x = 2 ∨ x = 3)) :=
by
  intro x
  sorry

end problem1_problem2_problem3_l259_259409


namespace evaluate_expression_l259_259147

theorem evaluate_expression : 2^(3^2) + 3^(2^3) = 7073 := by
  sorry

end evaluate_expression_l259_259147


namespace find_angle_BEC_l259_259708

-- Constants and assumptions
def angle_A : ℝ := 45
def angle_D : ℝ := 50
def angle_F : ℝ := 55
def E_above_C : Prop := true  -- This is a placeholder to represent the condition that E is directly above C.

-- Definition of the problem
theorem find_angle_BEC (angle_A_eq : angle_A = 45) 
                      (angle_D_eq : angle_D = 50) 
                      (angle_F_eq : angle_F = 55)
                      (triangle_BEC_formed : Prop)
                      (E_directly_above_C : E_above_C) 
                      : ∃ (BEC : ℝ), BEC = 10 :=
by sorry

end find_angle_BEC_l259_259708


namespace car_passing_time_l259_259302

open Real

theorem car_passing_time
  (vX : ℝ) (lX : ℝ)
  (vY : ℝ) (lY : ℝ)
  (t : ℝ)
  (h_vX : vX = 90)
  (h_lX : lX = 5)
  (h_vY : vY = 91)
  (h_lY : lY = 6)
  :
  (t * (vY - vX) / 3600) = 0.011 → t = 39.6 := 
by
  sorry

end car_passing_time_l259_259302


namespace exists_integer_x_l259_259900

theorem exists_integer_x (a b c : ℝ) (h : ∃ (a b : ℝ), (a - b).abs > 1 / (2 * Real.sqrt 2)) :
  ∃ (x : ℤ), (x : ℝ)^2 - 4 * (a + b + c) * x + 12 * (a * b + b * c + c * a) < 0 := 
sorry

end exists_integer_x_l259_259900


namespace rectangular_field_perimeter_l259_259927

variable (length width : ℝ)

theorem rectangular_field_perimeter (h_area : length * width = 50) (h_width : width = 5) : 2 * (length + width) = 30 := by
  sorry

end rectangular_field_perimeter_l259_259927


namespace pentagon_PT_length_l259_259709

theorem pentagon_PT_length (QR RS ST : ℝ) (angle_T right_angle_QRS T : Prop) (length_PT := (fun (a b : ℝ) => a + 3 * Real.sqrt b)) :
  QR = 3 →
  RS = 3 →
  ST = 3 →
  angle_T →
  right_angle_QRS →
  (angle_Q angle_R angle_S : ℝ) →
  angle_Q = 135 →
  angle_R = 135 →
  angle_S = 135 →
  ∃ (a b : ℝ), length_PT a b = 6 * Real.sqrt 2 ∧ a + b = 2 :=
by
  sorry

end pentagon_PT_length_l259_259709


namespace game_no_loser_l259_259203

theorem game_no_loser (x : ℕ) (h_start : x = 2017) :
  ∀ y, (y = x ∨ ∀ n, (n = 2 * y ∨ n = y - 1000) → (n > 1000 ∧ n < 4000)) →
       (y > 1000 ∧ y < 4000) :=
sorry

end game_no_loser_l259_259203


namespace simplify_expression_l259_259076

theorem simplify_expression :
  ((9 * 10^8) * 2^2) / (3 * 2^3 * 10^3) = 150000 := by sorry

end simplify_expression_l259_259076


namespace count_inverses_mod_11_l259_259533

def has_inverse_mod (a n : ℕ) : Prop :=
  ∃ b : ℕ, a * b ≡ 1 [MOD n]

theorem count_inverses_mod_11 : 
  (Finset.filter (λ a, has_inverse_mod a 11) (Finset.range 11)).card = 10 :=
by
  sorry

end count_inverses_mod_11_l259_259533


namespace problem_equivalent_l259_259839

theorem problem_equivalent
  (x : ℚ)
  (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 10) :
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 - Real.sqrt (x^4 - 4)) = 289 / 8 := 
by
  sorry

end problem_equivalent_l259_259839


namespace min_value_of_expression_l259_259163

theorem min_value_of_expression
  (x y : ℝ) 
  (h : x + y = 1) : 
  ∃ (m : ℝ), m = 2 * x^2 + 3 * y^2 ∧ m = 6 / 5 := 
sorry

end min_value_of_expression_l259_259163


namespace remaining_slices_correct_l259_259604

-- Define initial slices of pie and cake
def initial_pie_slices : Nat := 2 * 8
def initial_cake_slices : Nat := 12

-- Define slices eaten on Friday
def friday_pie_slices_eaten : Nat := 2
def friday_cake_slices_eaten : Nat := 2

-- Define slices eaten on Saturday
def saturday_pie_slices_eaten (remaining: Nat) : Nat := remaining / 2 -- 50%
def saturday_cake_slices_eaten (remaining: Nat) : Nat := remaining / 4 -- 25%

-- Define slices eaten on Sunday morning
def sunday_morning_pie_slices_eaten : Nat := 2
def sunday_morning_cake_slices_eaten : Nat := 3

-- Define slices eaten on Sunday evening
def sunday_evening_pie_slices_eaten : Nat := 4
def sunday_evening_cake_slices_eaten : Nat := 1

-- Function to calculate remaining slices
def remaining_slices : Nat × Nat :=
  let after_friday_pies := initial_pie_slices - friday_pie_slices_eaten
  let after_friday_cake := initial_cake_slices - friday_cake_slices_eaten
  let after_saturday_pies := after_friday_pies - saturday_pie_slices_eaten after_friday_pies
  let after_saturday_cake := after_friday_cake - saturday_cake_slices_eaten after_friday_cake
  let after_sunday_morning_pies := after_saturday_pies - sunday_morning_pie_slices_eaten
  let after_sunday_morning_cake := after_saturday_cake - sunday_morning_cake_slices_eaten
  let final_pies := after_sunday_morning_pies - sunday_evening_pie_slices_eaten
  let final_cake := after_sunday_morning_cake - sunday_evening_cake_slices_eaten
  (final_pies, final_cake)

theorem remaining_slices_correct :
  remaining_slices = (1, 4) :=
  by {
    sorry -- Proof is omitted
  }

end remaining_slices_correct_l259_259604


namespace total_ways_to_choose_gifts_l259_259742

/-- The 6 pairs of zodiac signs -/
def zodiac_pairs : Set (Set String) :=
  {{"Rat", "Ox"}, {"Tiger", "Rabbit"}, {"Dragon", "Snake"}, {"Horse", "Sheep"}, {"Monkey", "Rooster"}, {"Dog", "Pig"}}

/-- The preferences of Students A, B, and C -/
def A_likes : Set String := {"Ox", "Horse"}
def B_likes : Set String := {"Ox", "Dog", "Sheep"}
def C_likes : Set String := {"Rat", "Ox", "Tiger", "Rabbit", "Dragon", "Snake", "Horse", "Sheep", "Monkey", "Rooster", "Dog", "Pig"}

theorem total_ways_to_choose_gifts : 
  True := 
by
  -- We prove that the number of ways is 16
  sorry

end total_ways_to_choose_gifts_l259_259742


namespace max_amount_xiao_li_spent_l259_259303

theorem max_amount_xiao_li_spent (a m n : ℕ) :
  33 ≤ m ∧ m < n ∧ n ≤ 37 ∧
  ∃ (x y : ℕ), 
  (25 * (a - x) + m * (a - y) + n * (x + y + a) = 700) ∧ 
  (25 * x + m * y + n * (3*a - x - y) = 1200) ∧
  ( 675 <= 700 - 25) :=
sorry

end max_amount_xiao_li_spent_l259_259303


namespace find_d_squared_l259_259121

noncomputable def g (z : ℂ) (c d : ℝ) : ℂ := (c + d * Complex.I) * z

theorem find_d_squared (c d : ℝ) (z : ℂ) (h1 : ∀ z : ℂ, Complex.abs (g z c d - z) = 2 * Complex.abs (g z c d)) (h2 : Complex.abs (c + d * Complex.I) = 6) : d^2 = 11305 / 4 := 
sorry

end find_d_squared_l259_259121


namespace time_for_D_to_complete_job_l259_259627

-- Definitions for conditions
def A_rate : ℚ := 1 / 6
def combined_rate : ℚ := 1 / 4

-- We need to find D_rate
def D_rate : ℚ := combined_rate - A_rate

-- Now we state the theorem
theorem time_for_D_to_complete_job :
  D_rate = 1 / 12 :=
by
  /-
  We want to show that given the conditions:
  1. A_rate = 1 / 6
  2. A_rate + D_rate = 1 / 4
  it results in D_rate = 1 / 12.
  -/
  sorry

end time_for_D_to_complete_job_l259_259627


namespace two_digit_number_sum_l259_259217

theorem two_digit_number_sum (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : |(10 * a + b) - (10 * b + a)| = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 := by
  sorry

end two_digit_number_sum_l259_259217


namespace max_area_of_inscribed_rectangle_l259_259841

open Real

theorem max_area_of_inscribed_rectangle : 
  ∃ a : ℝ, -π/2 ≤ a ∧ a ≤ π/2 ∧ (2 * a * cos a = 1.1222) := 
by
  sorry

end max_area_of_inscribed_rectangle_l259_259841


namespace negative_solution_condition_l259_259336

variable {a b c x y : ℝ}

theorem negative_solution_condition (h1 : a * x + b * y = c)
    (h2 : b * x + c * y = a)
    (h3 : c * x + a * y = b)
    (hx : x < 0)
    (hy : y < 0) :
    a + b + c = 0 :=
sorry

end negative_solution_condition_l259_259336


namespace tetrahedron_has_six_edges_l259_259468

-- Define what a tetrahedron is
inductive Tetrahedron : Type
| mk : Tetrahedron

-- Define the number of edges of a Tetrahedron
def edges_of_tetrahedron (t : Tetrahedron) : Nat := 6

theorem tetrahedron_has_six_edges (t : Tetrahedron) : edges_of_tetrahedron t = 6 := 
by
  sorry

end tetrahedron_has_six_edges_l259_259468


namespace complex_conjugate_real_part_l259_259365

open Complex

theorem complex_conjugate_real_part (z1 z2 : ℂ) : 
  z1 * conj z2 + conj z1 * z2 ∈ ℝ :=
by
  sorry

end complex_conjugate_real_part_l259_259365


namespace count_inverses_modulo_11_l259_259553

theorem count_inverses_modulo_11 : (Finset.filter (λ a => Nat.gcd a 11 = 1) (Finset.range 11)).card = 10 :=
  by
  sorry

end count_inverses_modulo_11_l259_259553


namespace arithmetic_mean_is_correct_l259_259150

variable (x a : ℝ)
variable (hx : x ≠ 0)

theorem arithmetic_mean_is_correct : 
  (1/2 * ((x + 2 * a) / x - 1 + (x - 3 * a) / x + 1)) = (1 - a / (2 * x)) := 
  sorry

end arithmetic_mean_is_correct_l259_259150


namespace count_inverses_mod_11_l259_259550

theorem count_inverses_mod_11 :
  {a ∈ finset.range 11 | Int.gcd a 11 = 1}.card = 10 :=
by
  sorry

end count_inverses_mod_11_l259_259550


namespace deer_families_initial_count_l259_259754

theorem deer_families_initial_count (stayed moved_out : ℕ) (h_stayed : stayed = 45) (h_moved_out : moved_out = 34) :
  stayed + moved_out = 79 :=
by
  sorry

end deer_families_initial_count_l259_259754


namespace part1_part2_l259_259686

section Problem

open Real

noncomputable def f (x : ℝ) := exp x
noncomputable def g (x : ℝ) := log x - 2

theorem part1 (x : ℝ) (hx : x > 0) : g x ≥ - (exp 1) / x :=
by sorry

theorem part2 (a : ℝ) (h : ∀ x, x ≥ 0 → f x - 1 / (f x) ≥ a * x) : a ≤ 2 :=
by sorry

end Problem

end part1_part2_l259_259686


namespace monotonicity_of_f_odd_function_a_value_l259_259830

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2 ^ x + 1)

-- Part 1: Prove that f(x) is monotonically increasing
theorem monotonicity_of_f (a : ℝ) : 
  ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 := by
  intro x1 x2 hx
  sorry

-- Part 2: If f(x) is an odd function, find the value of a
theorem odd_function_a_value (a : ℝ) (h_odd : ∀ x : ℝ, f a (-x) = -f a x) : 
  f a 0 = 0 → a = 1 / 2 := by
  intro h
  sorry

end monotonicity_of_f_odd_function_a_value_l259_259830


namespace num_inverses_mod_11_l259_259559

theorem num_inverses_mod_11 : (finset.filter (λ x, nat.coprime x 11) (finset.range 11)).card = 10 :=
sorry

end num_inverses_mod_11_l259_259559


namespace num_inverses_mod_11_l259_259556

theorem num_inverses_mod_11 : (Finset.filter (λ a, Nat.gcd a 11 = 1) (Finset.range 11)).card = 10 :=
sorry

end num_inverses_mod_11_l259_259556


namespace simplify_expression_l259_259408

theorem simplify_expression (x y : ℝ) :
  4 * x + 8 * x^2 + y^3 + 6 - (3 - 4 * x - 8 * x^2 - y^3) =
  16 * x^2 + 8 * x + 2 * y^3 + 3 :=
by sorry

end simplify_expression_l259_259408


namespace find_constants_eq_l259_259149

theorem find_constants_eq (P Q R : ℚ)
  (h : ∀ x, (x^2 - 5) = P * (x - 4) * (x - 6) + Q * (x - 1) * (x - 6) + R * (x - 1) * (x - 4)) :
  (P = -4 / 15) ∧ (Q = -11 / 6) ∧ (R = 31 / 10) :=
by
  sorry

end find_constants_eq_l259_259149


namespace probability_XiaoYu_group_A_l259_259898

theorem probability_XiaoYu_group_A :
  ∀ (students : Fin 48) (groups : Fin 4) (groupAssignment : Fin 48 → Fin 4)
    (student : Fin 48) (groupA : Fin 4),
    (∀ (s : Fin 48), ∃ (g : Fin 4), groupAssignment s = g) → 
    (∀ (g : Fin 4), ∃ (count : ℕ), (0 < count ∧ count ≤ 12) ∧
       (∃ (groupMembers : List (Fin 48)), groupMembers.length = count ∧
        (∀ (m : Fin 48), m ∈ groupMembers → groupAssignment m = g))) →
    (groupAssignment student = groupA) →
  ∃ (p : ℚ), p = (1/4) ∧ ∀ (s : Fin 48), groupAssignment s = groupA → p = (1/4) :=
by
  sorry

end probability_XiaoYu_group_A_l259_259898


namespace edward_score_l259_259676

theorem edward_score (total_points : ℕ) (friend_points : ℕ) 
  (h1 : total_points = 13) (h2 : friend_points = 6) : 
  ∃ edward_points : ℕ, edward_points = 7 :=
by
  sorry

end edward_score_l259_259676


namespace prime_mod4_eq3_has_x0_y0_solution_l259_259193

theorem prime_mod4_eq3_has_x0_y0_solution (p x0 y0 : ℕ) (h1 : Nat.Prime p) (h2 : p % 4 = 3)
    (h3 : (p + 2) * x0^2 - (p + 1) * y0^2 + p * x0 + (p + 2) * y0 = 1) :
    p ∣ x0 :=
sorry

end prime_mod4_eq3_has_x0_y0_solution_l259_259193


namespace gcd_2814_1806_l259_259947

def a := 2814
def b := 1806

theorem gcd_2814_1806 : Nat.gcd a b = 42 :=
by
  sorry

end gcd_2814_1806_l259_259947


namespace count_inverses_modulo_11_l259_259541

theorem count_inverses_modulo_11 : (∀ a : ℤ, 0 ≤ a ∧ a ≤ 10 → ∃ b : ℤ, a * b ≡ 1 [MOD 11]) → (finset.range 11).filter (λ a, (nat.gcd a 11 = 1)).card = 10 :=
by
  sorry

end count_inverses_modulo_11_l259_259541


namespace area_of_centroid_path_l259_259909

theorem area_of_centroid_path (A B C O G : ℝ) (r : ℝ) (h1 : A ≠ B) 
  (h2 : 2 * r = 30) (h3 : ∀ C, C ≠ A ∧ C ≠ B ∧ dist O C = r) 
  (h4 : dist O G = r / 3) : 
  (π * (r / 3)^2 = 25 * π) :=
by 
  -- def AB := 2 * r -- given AB is a diameter of the circle
  -- def O := (A + B) / 2 -- center of the circle
  -- def G := (A + B + C) / 3 -- centroid of triangle ABC
  sorry

end area_of_centroid_path_l259_259909


namespace joan_gave_sam_seashells_l259_259383

theorem joan_gave_sam_seashells (original_seashells : ℕ) (left_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : original_seashells = 70) (h2 : left_seashells = 27) : given_seashells = 43 :=
by
  have h3 : given_seashells = original_seashells - left_seashells := sorry
  rw [h1, h2] at h3
  exact h3

end joan_gave_sam_seashells_l259_259383


namespace system_has_negative_solution_iff_sum_zero_l259_259331

variables {a b c x y : ℝ}

-- Statement of the problem
theorem system_has_negative_solution_iff_sum_zero :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔ (a + b + c = 0) := by
  sorry

end system_has_negative_solution_iff_sum_zero_l259_259331


namespace max_value_of_cubes_l259_259391

theorem max_value_of_cubes (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 + ab + ac + ad + bc + bd + cd = 10) :
  a^3 + b^3 + c^3 + d^3 ≤ 4 * Real.sqrt 10 :=
sorry

end max_value_of_cubes_l259_259391


namespace prove_sum_is_12_l259_259589

theorem prove_sum_is_12 (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 := 
by 
  sorry

end prove_sum_is_12_l259_259589


namespace find_p_l259_259219

def delta (a b : ℝ) : ℝ := a * b + a + b

theorem find_p (p : ℝ) (h : delta p 3 = 39) : p = 9 :=
by
  sorry

end find_p_l259_259219


namespace max_f_gt_sqrt2_f_is_periodic_f_pi_shift_pos_l259_259832

noncomputable def f (x : ℝ) := Real.sin (Real.cos x) + Real.cos (Real.sin x)

theorem max_f_gt_sqrt2 : (∃ x : ℝ, f x > Real.sqrt 2) :=
sorry

theorem f_is_periodic : ∀ x : ℝ, f (x - 2 * Real.pi) = f x :=
sorry

theorem f_pi_shift_pos : ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi → f (x + Real.pi) > 0 :=
sorry

end max_f_gt_sqrt2_f_is_periodic_f_pi_shift_pos_l259_259832


namespace couple_slices_each_l259_259776

noncomputable def slices_for_couple (total_slices children_slices people_in_couple : ℕ) : ℕ :=
  (total_slices - children_slices) / people_in_couple

theorem couple_slices_each (people_in_couple children slices_per_pizza num_pizzas : ℕ) (H1 : people_in_couple = 2) (H2 : children = 6) (H3 : slices_per_pizza = 4) (H4 : num_pizzas = 3) :
  slices_for_couple (num_pizzas * slices_per_pizza) (children * 1) people_in_couple = 3 := 
  by
  rw [H1, H2, H3, H4]
  show slices_for_couple (3 * 4) (6 * 1) 2 = 3
  rfl

end couple_slices_each_l259_259776


namespace Mary_cut_10_roses_l259_259432

-- Defining the initial and final number of roses
def initial_roses := 6
def final_roses := 16

-- Calculating the number of roses cut by Mary
def roses_cut := final_roses - initial_roses

-- The proof problem: Prove that the number of roses cut is 10
theorem Mary_cut_10_roses : roses_cut = 10 := by
  sorry

end Mary_cut_10_roses_l259_259432


namespace total_shells_l259_259803

theorem total_shells :
  let initial_shells := 2
  let ed_limpet_shells := 7
  let ed_oyster_shells := 2
  let ed_conch_shells := 4
  let ed_scallop_shells := 3
  let jacob_more_shells := 2
  let marissa_limpet_shells := 5
  let marissa_oyster_shells := 6
  let marissa_conch_shells := 3
  let marissa_scallop_shells := 1
  let ed_shells := ed_limpet_shells + ed_oyster_shells + ed_conch_shells + ed_scallop_shells
  let jacob_shells := ed_shells + jacob_more_shells
  let marissa_shells := marissa_limpet_shells + marissa_oyster_shells + marissa_conch_shells + marissa_scallop_shells
  let shells_at_beach := ed_shells + jacob_shells + marissa_shells
  let total_shells := shells_at_beach + initial_shells
  total_shells = 51 := by
  sorry

end total_shells_l259_259803


namespace smallest_number_divisible_by_15_and_36_l259_259241

theorem smallest_number_divisible_by_15_and_36 : 
  ∃ x, (∀ y, (y % 15 = 0 ∧ y % 36 = 0) → y ≥ x) ∧ x = 180 :=
by
  sorry

end smallest_number_divisible_by_15_and_36_l259_259241


namespace playground_area_l259_259938

noncomputable def calculate_area (w s : ℝ) : ℝ := s * s

theorem playground_area (w s : ℝ) (h1 : s = 3 * w + 10) (h2 : 4 * s = 480) : calculate_area w s = 14400 := by
  sorry

end playground_area_l259_259938


namespace tina_total_income_is_correct_l259_259433

-- Definitions based on the conditions
def hourly_wage : ℝ := 18.0
def regular_hours_per_day : ℝ := 8
def overtime_hours_per_day_weekday : ℝ := 2
def double_overtime_hours_per_day_weekend : ℝ := 2

def overtime_rate : ℝ := hourly_wage + 0.5 * hourly_wage
def double_overtime_rate : ℝ := 2 * hourly_wage

def weekday_hours_per_day : ℝ := 10
def weekend_hours_per_day : ℝ := 12

def regular_pay_per_day : ℝ := hourly_wage * regular_hours_per_day
def overtime_pay_per_day_weekday : ℝ := overtime_rate * overtime_hours_per_day_weekday
def double_overtime_pay_per_day_weekend : ℝ := double_overtime_rate * double_overtime_hours_per_day_weekend

def total_weekday_pay_per_day : ℝ := regular_pay_per_day + overtime_pay_per_day_weekday
def total_weekend_pay_per_day : ℝ := regular_pay_per_day + overtime_pay_per_day_weekday + double_overtime_pay_per_day_weekend

def number_of_weekdays : ℝ := 5
def number_of_weekends : ℝ := 2

def total_weekday_income : ℝ := total_weekday_pay_per_day * number_of_weekdays
def total_weekend_income : ℝ := total_weekend_pay_per_day * number_of_weekends

def total_weekly_income : ℝ := total_weekday_income + total_weekend_income

-- The theorem we need to prove
theorem tina_total_income_is_correct : total_weekly_income = 1530 := by
  sorry

end tina_total_income_is_correct_l259_259433


namespace combined_selling_price_l259_259461

theorem combined_selling_price :
  let cost_cycle := 2300
  let cost_scooter := 12000
  let cost_motorbike := 25000
  let loss_cycle := 0.30
  let profit_scooter := 0.25
  let profit_motorbike := 0.15
  let selling_price_cycle := cost_cycle - (loss_cycle * cost_cycle)
  let selling_price_scooter := cost_scooter + (profit_scooter * cost_scooter)
  let selling_price_motorbike := cost_motorbike + (profit_motorbike * cost_motorbike)
  selling_price_cycle + selling_price_scooter + selling_price_motorbike = 45360 := 
by
  sorry

end combined_selling_price_l259_259461


namespace natasha_time_reach_top_l259_259066

variable (t : ℝ) (d_up d_total T : ℝ)

def time_to_reach_top (T d_up d_total t : ℝ) : Prop :=
  d_total = 2 * d_up ∧
  d_up = 1.5 * t ∧
  T = t + 2 ∧
  2 = d_total / T

theorem natasha_time_reach_top (T : ℝ) (h : time_to_reach_top T (1.5 * 4) (3 * 4) 4) : T = 4 :=
by
  sorry

end natasha_time_reach_top_l259_259066


namespace probability_no_three_consecutive_1s_l259_259467

theorem probability_no_three_consecutive_1s (m n : ℕ) (h_relatively_prime : Nat.gcd m n = 1) (h_eq : 2^12 = 4096) :
  let b₁ := 2
  let b₂ := 4
  let b₃ := 7
  let b₄ := b₃ + b₂ + b₁
  let b₅ := b₄ + b₃ + b₂
  let b₆ := b₅ + b₄ + b₃
  let b₇ := b₆ + b₅ + b₄
  let b₈ := b₇ + b₆ + b₅
  let b₉ := b₈ + b₇ + b₆
  let b₁₀ := b₉ + b₈ + b₇
  let b₁₁ := b₁₀ + b₉ + b₈
  let b₁₂ := b₁₁ + b₁₀ + b₉
  (m = 1705 ∧ n = 4096 ∧ b₁₂ = m) →
  m + n = 5801 := 
by
  intros
  sorry

end probability_no_three_consecutive_1s_l259_259467


namespace sports_club_problem_l259_259372

theorem sports_club_problem (N B T Neither X : ℕ) (hN : N = 42) (hB : B = 20) (hT : T = 23) (hNeither : Neither = 6) :
  (B + T - X + Neither = N) → X = 7 :=
by
  intro h
  sorry

end sports_club_problem_l259_259372


namespace problem1_problem2_l259_259392

open Finset

-- Define the finite set M and its subsets
def M (n : ℕ) : Finset ℝ := (range (n + 1)).image (λ k, 1 / 2^(n + k))

-- Define the function S which returns the sum of elements in a subset of M
def S (n : ℕ) (s : Finset ℝ) : ℝ := s.sum id

-- Prove the problems
theorem problem1 :
  S 2 (\{1 / 2^2, 1 / 2^3, 1 / 2^4\}) = 1 + (1 / 2) + (1 / 4) := by
  sorry

theorem problem2 (n : ℕ) (h : 0 < n) :
  let t := 2 ^ (n + 1)
  in (2 ^ n) * ((range (n + 1)).sum (λ k, 1 / 2^ (n + k))) = 2 - (1 / 2^n) := by
  sorry

end problem1_problem2_l259_259392


namespace condition_two_eqn_l259_259517

def line_through_point_and_perpendicular (x1 y1 : ℝ) (c : ℝ) : Prop :=
  ∀ x y : ℝ, (y - y1) = -1/(x - x1) * (x - x1 + c) → x - y + c = 0

theorem condition_two_eqn :
  line_through_point_and_perpendicular 1 (-2) (-3) :=
sorry

end condition_two_eqn_l259_259517


namespace find_x_given_y_and_ratio_l259_259166

variable (x y k : ℝ)

theorem find_x_given_y_and_ratio :
  (∀ x y, (5 * x - 6) / (2 * y + 20) = k) →
  (5 * 3 - 6) / (2 * 5 + 20) = k →
  y = 15 →
  x = 21 / 5 :=
by 
  intro h1 h2 hy
  -- proof steps would go here
  sorry

end find_x_given_y_and_ratio_l259_259166


namespace rem_frac_l259_259441

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem rem_frac : rem (5/7 : ℚ) (3/4 : ℚ) = (5/7 : ℚ) := 
by 
  sorry

end rem_frac_l259_259441


namespace function_increasing_iff_m_eq_1_l259_259521

theorem function_increasing_iff_m_eq_1 (m : ℝ) : 
  (m^2 - 4 * m + 4 = 1) ∧ (m^2 - 6 * m + 8 > 0) ↔ m = 1 :=
by {
  sorry
}

end function_increasing_iff_m_eq_1_l259_259521


namespace compare_abc_l259_259056

def a : ℝ := 2 * Real.log 1.01
def b : ℝ := Real.log 1.02
def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b :=
by
  /- Proof goes here -/
  sorry

end compare_abc_l259_259056


namespace at_least_fifty_same_leading_coefficient_l259_259821

-- Define what it means for two quadratic polynomials to intersect exactly once
def intersect_once (P Q : Polynomial ℝ) : Prop :=
∃ x, P.eval x = Q.eval x ∧ ∀ y ≠ x, P.eval y ≠ Q.eval y

-- Define the main theorem and its conditions
theorem at_least_fifty_same_leading_coefficient 
  (polynomials : Fin 100 → Polynomial ℝ)
  (h1 : ∀ i j, i ≠ j → intersect_once (polynomials i) (polynomials j))
  (h2 : ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
        ¬∃ x, (polynomials i).eval x = (polynomials j).eval x ∧ (polynomials j).eval x = (polynomials k).eval x) : 
  ∃ (S : Finset (Fin 100)), S.card ≥ 50 ∧ ∃ a, ∀ i ∈ S, (polynomials i).leadingCoeff = a :=
sorry

end at_least_fifty_same_leading_coefficient_l259_259821


namespace polygon_diagonals_l259_259019

theorem polygon_diagonals (n : ℕ) (h : n - 3 ≤ 6) : n = 9 :=
by sorry

end polygon_diagonals_l259_259019


namespace count_ways_to_write_2010_l259_259716

theorem count_ways_to_write_2010 : ∃ N : ℕ, 
  (∀ (a_3 a_2 a_1 a_0 : ℕ), a_0 ≤ 99 ∧ a_1 ≤ 99 ∧ a_2 ≤ 99 ∧ a_3 ≤ 99 → 
    2010 = a_3 * 10^3 + a_2 * 10^2 + a_1 * 10 + a_0) ∧ 
    N = 202 :=
sorry

end count_ways_to_write_2010_l259_259716


namespace number_of_arrangements_of_SUCCESS_l259_259143

-- Definitions based on conditions
def word : String := "SUCCESS"
def total_letters : Nat := 7
def repetitions : List Nat := [3, 2, 1, 1]  -- Corresponding to S, C, U, E

-- Lean statement proving the number of arrangements
theorem number_of_arrangements_of_SUCCESS : 
  (Nat.factorial 7) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1) * (Nat.factorial 1)) = 420 := by
  sorry

end number_of_arrangements_of_SUCCESS_l259_259143


namespace Laura_running_speed_l259_259385

noncomputable def running_speed (x : ℝ) :=
  let biking_time := 30 / (3 * x + 2)
  let running_time := 10 / x
  let total_time := biking_time + running_time
  total_time = 3

theorem Laura_running_speed : ∃ x : ℝ, running_speed x ∧ abs (x - 6.35) < 0.01 :=
sorry

end Laura_running_speed_l259_259385


namespace brothers_selection_probability_l259_259707

theorem brothers_selection_probability :
  let P_A := 1 / 7 : ℚ
  let P_B := 2 / 5 : ℚ
  let P_Xi := 3 / 4 : ℚ
  let P_Xt := 4 / 9 : ℚ
  let P_Yi := 5 / 8 : ℚ
  let P_Yt := 7 / 10 : ℚ
  let P_X := P_A * P_Xi * P_Xt
  let P_Y := P_B * P_Yi * P_Yt
  P_X * P_Y = 7 / 840 :=
  by
    let P_A := 1 / 7 : ℚ
    let P_B := 2 / 5 : ℚ
    let P_Xi := 3 / 4 : ℚ
    let P_Xt := 4 / 9 : ℚ
    let P_Yi := 5 / 8 : ℚ
    let P_Yt := 7 / 10 : ℚ
    let P_X := P_A * P_Xi * P_Xt
    let P_Y := P_B * P_Yi * P_Yt
    have : P_X = 1 / 21 := by
      sorry
    have : P_Y = 7 / 40 := by
      sorry
    show P_X * P_Y = 7 / 840
    sorry

end brothers_selection_probability_l259_259707


namespace compare_a_b_c_l259_259040

noncomputable def a : ℝ := 2 * Real.ln 1.01
noncomputable def b : ℝ := Real.ln 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l259_259040


namespace probability_one_from_harold_and_one_from_marilyn_l259_259999

-- Define the names and the number of letters in each name
def harold_name_length := 6
def marilyn_name_length := 7

-- Total cards
def total_cards := harold_name_length + marilyn_name_length

-- Probability of drawing one card from Harold's name and one from Marilyn's name
theorem probability_one_from_harold_and_one_from_marilyn :
    (harold_name_length : ℚ) / total_cards * marilyn_name_length / (total_cards - 1) +
    marilyn_name_length / total_cards * harold_name_length / (total_cards - 1) 
    = 7 / 13 := 
by
  sorry

end probability_one_from_harold_and_one_from_marilyn_l259_259999


namespace ratio_of_sector_CPD_l259_259397

-- Define the given angles
def angle_AOC : ℝ := 40
def angle_DOB : ℝ := 60
def angle_COP : ℝ := 110

-- Calculate the angle CPD
def angle_CPD : ℝ := angle_COP - angle_AOC - angle_DOB

-- State the theorem to prove the ratio
theorem ratio_of_sector_CPD (hAOC : angle_AOC = 40) (hDOB : angle_DOB = 60)
(hCOP : angle_COP = 110) : 
  angle_CPD / 360 = 1 / 36 := by
  -- Proof will go here
  sorry

end ratio_of_sector_CPD_l259_259397


namespace inequality_a_c_b_l259_259058

theorem inequality_a_c_b :
  let a := 2 * Real.log 1.01
  let b := Real.log 1.02
  let c := Real.sqrt 1.04 - 1
  a > c ∧ c > b :=
by
  let a : ℝ := 2 * Real.log 1.01
  let b : ℝ := Real.log 1.02
  let c : ℝ := Real.sqrt 1.04 - 1
  split
  sorry
  sorry

end inequality_a_c_b_l259_259058


namespace three_digit_numbers_not_multiple_of_3_5_7_l259_259006

theorem three_digit_numbers_not_multiple_of_3_5_7 : 
  let total_three_digit_numbers := 900
  let multiples_of_3 := (999 - 100) / 3 + 1
  let multiples_of_5 := (995 - 100) / 5 + 1
  let multiples_of_7 := (994 - 105) / 7 + 1
  let multiples_of_15 := (990 - 105) / 15 + 1
  let multiples_of_21 := (987 - 105) / 21 + 1
  let multiples_of_35 := (980 - 105) / 35 + 1
  let multiples_of_105 := (945 - 105) / 105 + 1
  let total_multiples := multiples_of_3 + multiples_of_5 + multiples_of_7 - multiples_of_15 - multiples_of_21 - multiples_of_35 + multiples_of_105
  let non_multiples_total := total_three_digit_numbers - total_multiples
  non_multiples_total = 412 :=
by
  sorry

end three_digit_numbers_not_multiple_of_3_5_7_l259_259006


namespace solve_for_x_l259_259164

theorem solve_for_x (x : ℝ) (h₀ : x^2 - 4 * x = 0) (h₁ : x ≠ 0) : x = 4 := 
by
  sorry

end solve_for_x_l259_259164


namespace carol_initial_cupcakes_l259_259341

variable (x : ℕ)

theorem carol_initial_cupcakes (h : (x - 9) + 28 = 49) : x = 30 := 
  sorry

end carol_initial_cupcakes_l259_259341


namespace dhoni_savings_l259_259671

theorem dhoni_savings :
  let earnings := 100
  let rent := 0.25 * earnings
  let dishwasher := rent - (0.10 * rent)
  let utilities := 0.15 * earnings
  let groceries := 0.20 * earnings
  let transportation := 0.12 * earnings
  let total_spent := rent + dishwasher + utilities + groceries + transportation
  earnings - total_spent = 0.055 * earnings :=
by
  sorry

end dhoni_savings_l259_259671


namespace negative_solution_condition_l259_259337

variable {a b c x y : ℝ}

theorem negative_solution_condition (h1 : a * x + b * y = c)
    (h2 : b * x + c * y = a)
    (h3 : c * x + a * y = b)
    (hx : x < 0)
    (hy : y < 0) :
    a + b + c = 0 :=
sorry

end negative_solution_condition_l259_259337


namespace tournament_cycle_exists_l259_259952

theorem tournament_cycle_exists :
  ∃ (A B C : Fin 12), 
  (∃ M : Fin 12 → Fin 12 → Bool, 
    (∀ p : Fin 12, ∃ q : Fin 12, q ≠ p ∧ M p q) ∧
    M A B = true ∧ M B C = true ∧ M C A = true) :=
sorry

end tournament_cycle_exists_l259_259952


namespace initial_population_l259_259025

theorem initial_population (P : ℝ) (h1 : 1.05 * (0.765 * P + 50) = 3213) : P = 3935 :=
by
  have h2 : 1.05 * (0.765 * P + 50) = 3213 := h1
  sorry

end initial_population_l259_259025


namespace problem1_problem2_l259_259775

-- Definitions based on the conditions
def internal_medicine_doctors : ℕ := 12
def surgeons : ℕ := 8
def total_doctors : ℕ := internal_medicine_doctors + surgeons
def team_size : ℕ := 5

-- Problem 1: Both doctor A and B must join the team
theorem problem1 : ∃ (ways : ℕ), ways = 816 :=
  by
    let remaining_doctors := total_doctors - 2
    let choose := remaining_doctors.choose (team_size - 2)
    have h1 : choose = 816 := sorry
    exact ⟨choose, h1⟩

-- Problem 2: At least one of doctors A or B must join the team
theorem problem2 : ∃ (ways : ℕ), ways = 5661 :=
  by
    let remaining_doctors := total_doctors - 1
    let scenario1 := 2 * remaining_doctors.choose (team_size - 1)
    let scenario2 := (total_doctors - 2).choose (team_size - 2)
    let total_ways := scenario1 + scenario2
    have h2 : total_ways = 5661 := sorry
    exact ⟨total_ways, h2⟩

end problem1_problem2_l259_259775


namespace problem_inequality_l259_259904

theorem problem_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^6 - a^2 + 4) * (b^6 - b^2 + 4) * (c^6 - c^2 + 4) * (d^6 - d^2 + 4) ≥ (a + b + c + d)^4 :=
by
  sorry

end problem_inequality_l259_259904


namespace evaluate_sum_of_powers_of_i_l259_259146

-- Definition of the imaginary unit i with property i^2 = -1.
def i : ℂ := Complex.I

lemma i_pow_2 : i^2 = -1 := by
  sorry

lemma i_pow_4n (n : ℤ) : i^(4 * n) = 1 := by
  sorry

-- Problem statement: Evaluate i^13 + i^18 + i^23 + i^28 + i^33 + i^38.
theorem evaluate_sum_of_powers_of_i : 
  i^13 + i^18 + i^23 + i^28 + i^33 + i^38 = 0 := by
  sorry

end evaluate_sum_of_powers_of_i_l259_259146


namespace count_divisors_720_l259_259583

theorem count_divisors_720 : ∑ d in divisors 720, 1 = 30 := by
  sorry

end count_divisors_720_l259_259583


namespace max_marks_set_for_test_l259_259280

-- Define the conditions according to the problem statement
def passing_percentage : ℝ := 0.70
def student_marks : ℝ := 120
def marks_needed_to_pass : ℝ := 150
def passing_threshold (M : ℝ) : ℝ := passing_percentage * M

-- The maximum marks set for the test
theorem max_marks_set_for_test (M : ℝ) : M = 386 :=
by
  -- Given the conditions
  have h : passing_threshold M = student_marks + marks_needed_to_pass := sorry
  -- Solving for M
  sorry

end max_marks_set_for_test_l259_259280


namespace sum_of_quarter_circle_arcs_l259_259260

-- Define the main variables and problem statement.
variable (D : ℝ) -- Diameter of the original circle.
variable (n : ℕ) (hn : 0 < n) -- Number of parts (positive integer).

-- Define a theorem stating that the sum of quarter-circle arcs is greater than D, but less than (pi D / 2) as n tends to infinity.
theorem sum_of_quarter_circle_arcs (hn : 0 < n) :
  D < (π * D) / 4 ∧ (π * D) / 4 < (π * D) / 2 :=
by
  sorry -- Proof of the theorem goes here.

end sum_of_quarter_circle_arcs_l259_259260


namespace cheapest_candle_cost_to_measure_1_minute_l259_259298

-- Definitions

def big_candle_cost := 16 -- cost of a big candle in cents
def big_candle_burn_time := 16 -- burn time of a big candle in minutes
def small_candle_cost := 7 -- cost of a small candle in cents
def small_candle_burn_time := 7 -- burn time of a small candle in minutes

-- Problem statement
theorem cheapest_candle_cost_to_measure_1_minute :
  (∃ (n m : ℕ), n * big_candle_burn_time - m * small_candle_burn_time = 1 ∧
                 n * big_candle_cost + m * small_candle_cost = 97) :=
sorry

end cheapest_candle_cost_to_measure_1_minute_l259_259298


namespace negative_two_squared_l259_259658

theorem negative_two_squared :
  (-2 : ℤ)^2 = 4 := 
sorry

end negative_two_squared_l259_259658


namespace michael_passes_donovan_after_laps_l259_259247

/-- The length of the track in meters -/
def track_length : ℕ := 400

/-- Donovan's lap time in seconds -/
def donovan_lap_time : ℕ := 45

/-- Michael's lap time in seconds -/
def michael_lap_time : ℕ := 36

/-- The number of laps that Michael will have to complete in order to pass Donovan -/
theorem michael_passes_donovan_after_laps : 
  ∃ (laps : ℕ), laps = 5 ∧ (∃ t : ℕ, 400 * t / 36 = 5 ∧ 400 * t / 45 < 5) :=
sorry

end michael_passes_donovan_after_laps_l259_259247


namespace non_deg_ellipse_projection_l259_259934

theorem non_deg_ellipse_projection (m : ℝ) : 
  (3 * x^2 + 9 * y^2 - 12 * x + 18 * y + 6 * z = m → (m > -21)) := 
by
  sorry

end non_deg_ellipse_projection_l259_259934


namespace winning_candidate_percentage_l259_259373

theorem winning_candidate_percentage 
    (votes_winner : ℕ)
    (votes_total : ℕ)
    (votes_majority : ℕ)
    (H1 : votes_total = 900)
    (H2 : votes_majority = 360)
    (H3 : votes_winner - (votes_total - votes_winner) = votes_majority) :
    (votes_winner : ℕ) * 100 / (votes_total : ℕ) = 70 := by
    sorry

end winning_candidate_percentage_l259_259373


namespace phone_call_answered_within_first_four_rings_l259_259023

def P1 := 0.1
def P2 := 0.3
def P3 := 0.4
def P4 := 0.1

theorem phone_call_answered_within_first_four_rings :
  P1 + P2 + P3 + P4 = 0.9 :=
by
  rw [P1, P2, P3, P4]
  norm_num
  sorry -- Proof step skipped

end phone_call_answered_within_first_four_rings_l259_259023


namespace a_range_iff_l259_259702

theorem a_range_iff (a x : ℝ) (h1 : x < 3) (h2 : (a - 1) * x < a + 3) : 
  1 ≤ a ∧ a < 3 := 
by
  sorry

end a_range_iff_l259_259702


namespace find_a_b_extreme_values_l259_259829

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + b * x - (2/3)

theorem find_a_b_extreme_values : 
  ∃ (a b : ℝ), 
    (a = -2) ∧ 
    (b = 3) ∧ 
    (f 1 (-2) 3 = 2/3) ∧ 
    (f 3 (-2) 3 = -2/3) :=
by
  sorry

end find_a_b_extreme_values_l259_259829


namespace find_f_ln_inv_6_l259_259685

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x + 2 / x^3 - 3

theorem find_f_ln_inv_6 (k : ℝ) (h : f k (Real.log 6) = 1) : f k (Real.log (1 / 6)) = -7 :=
by
  sorry

end find_f_ln_inv_6_l259_259685


namespace cannot_determine_right_triangle_l259_259128

-- Definitions of conditions
def condition_A (A B C : ℝ) : Prop := A = B + C
def condition_B (a b c : ℝ) : Prop := a/b = 5/12 ∧ b/c = 12/13
def condition_C (a b c : ℝ) : Prop := a^2 = (b + c) * (b - c)
def condition_D (A B C : ℝ) : Prop := A/B = 3/4 ∧ B/C = 4/5

-- The proof problem
theorem cannot_determine_right_triangle (a b c A B C : ℝ)
  (hD : condition_D A B C) : 
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) := by
  sorry

end cannot_determine_right_triangle_l259_259128


namespace distinct_positive_integers_criteria_l259_259906

theorem distinct_positive_integers_criteria (x y z : ℕ) (hx : x ≠ y) (hy : y ≠ z) (hz : z ≠ x)
  (hxyz_div : x * y * z ∣ (x * y - 1) * (y * z - 1) * (z * x - 1)) :
  (x, y, z) = (2, 3, 5) ∨ (x, y, z) = (2, 5, 3) ∨ (x, y, z) = (3, 2, 5) ∨
  (x, y, z) = (3, 5, 2) ∨ (x, y, z) = (5, 2, 3) ∨ (x, y, z) = (5, 3, 2) :=
by sorry

end distinct_positive_integers_criteria_l259_259906


namespace exactly_one_pit_no_replanting_l259_259751

noncomputable def pit_prob : ℚ := 1/4
noncomputable def no_replanting_prob : ℚ := 1 - pit_prob

theorem exactly_one_pit_no_replanting :
  (3.choose 1) * no_replanting_prob * (pit_prob ^ 2) = 9/64 := by
  sorry

end exactly_one_pit_no_replanting_l259_259751


namespace arithmetic_sequence_sum_l259_259092

theorem arithmetic_sequence_sum (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) (c : ℤ) :
  (∀ n : ℕ, 0 < n → S_n n = n^2 + c) →
  a_n 1 = 1 + c →
  (∀ n, 1 < n → a_n n = S_n n - S_n (n - 1)) →
  (∀ n : ℕ, 0 < n → a_n n = 1 + (n - 1) * 2) →
  c = 0 ∧ (∀ n : ℕ, 0 < n → a_n n = 2 * n - 1) :=
by
  sorry

end arithmetic_sequence_sum_l259_259092


namespace rectangular_solid_surface_area_l259_259675

theorem rectangular_solid_surface_area (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (hneq1 : a ≠ b) (hneq2 : b ≠ c) (hneq3 : a ≠ c) (hvol : a * b * c = 770) : 2 * (a * b + b * c + c * a) = 1098 :=
by
  sorry

end rectangular_solid_surface_area_l259_259675


namespace handshakes_in_octagonal_shape_l259_259254

-- Definitions
def number_of_students : ℕ := 8

def non_adjacent_handshakes_per_student : ℕ := number_of_students - 1 - 2

def total_handshakes : ℕ := (number_of_students * non_adjacent_handshakes_per_student) / 2

-- Theorem to prove
theorem handshakes_in_octagonal_shape : total_handshakes = 20 := 
by
  -- Provide the proof here.
  sorry

end handshakes_in_octagonal_shape_l259_259254


namespace max_in_circle_eqn_l259_259186

theorem max_in_circle_eqn : 
  ∀ (x y : ℝ), (x ≥ 0) → (y ≥ 0) → (4 * x + 3 * y ≤ 12) → (x - 1)^2 + (y - 1)^2 = 1 :=
by
  intros x y hx hy hineq
  sorry

end max_in_circle_eqn_l259_259186


namespace total_weight_apples_l259_259719

variable (Minjae_weight : ℝ) (Father_weight : ℝ)

theorem total_weight_apples (h1 : Minjae_weight = 2.6) (h2 : Father_weight = 5.98) :
  Minjae_weight + Father_weight = 8.58 :=
by 
  sorry

end total_weight_apples_l259_259719


namespace solution_set_f_leq_6_sum_of_squares_geq_16_div_3_l259_259347

-- Problem 1: Solution set for the inequality \( f(x) ≤ 6 \)
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

theorem solution_set_f_leq_6 :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -2 ≤ x ∧ x ≤ 4} :=
sorry

-- Problem 2: Prove \( a^2 + b^2 + c^2 ≥ 16/3 \)
variables (a b c : ℝ)
axiom pos_abc : a > 0 ∧ b > 0 ∧ c > 0
axiom sum_abc : a + b + c = 4

theorem sum_of_squares_geq_16_div_3 :
  a^2 + b^2 + c^2 ≥ 16 / 3 :=
sorry

end solution_set_f_leq_6_sum_of_squares_geq_16_div_3_l259_259347


namespace payment_for_150_books_equal_payment_number_of_books_l259_259445

/-- 
Xinhua Bookstore conditions:
- Both suppliers A and B price each book at 40 yuan. 
- Supplier A offers a 10% discount on all books.
- Supplier B offers a 20% discount on any books purchased exceeding 100 books.
-/

def price_per_book_supplier_A (n : ℕ) : ℝ := 40 * 0.9
def price_per_first_100_books_supplier_B : ℝ := 40
def price_per_excess_books_supplier_B (n : ℕ) : ℝ := 40 * 0.8

-- Prove that the payment amounts for 150 books from suppliers A and B are 5400 yuan and 5600 yuan respectively.
theorem payment_for_150_books :
  price_per_book_supplier_A 150 * 150 = 5400 ∧
  price_per_first_100_books_supplier_B * 100 + price_per_excess_books_supplier_B 50 * (150 - 100) = 5600 :=
  sorry

-- Prove the equal payment equivalence theorem for supplier A and B.
theorem equal_payment_number_of_books (x : ℕ) :
  price_per_book_supplier_A x * x = price_per_first_100_books_supplier_B * 100 + price_per_excess_books_supplier_B (x - 100) * (x - 100) → x = 200 :=
  sorry

end payment_for_150_books_equal_payment_number_of_books_l259_259445


namespace range_of_a_l259_259819

variables {x a : ℝ}

def p (x : ℝ) : Prop := (x - 5) / (x - 3) ≥ 2
def q (x a : ℝ) : Prop := x ^ 2 - a * x ≤ x - a

theorem range_of_a (h : ¬(∃ x, p x) → ¬(∃ x, q x a)) :
  1 ≤ a ∧ a < 3 :=
by 
  sorry

end range_of_a_l259_259819


namespace distribute_books_into_bags_l259_259653

def number_of_ways_to_distribute_books (books : Finset ℕ) (bags : ℕ) : ℕ :=
  if (books.card = 5) ∧ (bags = 3) then 51 else 0

theorem distribute_books_into_bags :
  number_of_ways_to_distribute_books (Finset.range 5) 3 = 51 := by
  sorry

end distribute_books_into_bags_l259_259653


namespace escalator_length_is_120_l259_259728

variable (x : ℝ) -- Speed of escalator in steps/unit time.

constant steps_while_ascending : ℕ := 75
constant steps_while_descending : ℕ := 150
constant speed_ascending : ℝ := 1.0
constant speed_descending : ℝ := 3.0
constant walking_speed_ratio : ℝ := 3.0

theorem escalator_length_is_120 :
  let t_ascending := (steps_while_ascending / (speed_ascending + x))
      t_descending := (steps_while_descending / (speed_descending - x)) * (1 / walking_speed_ratio) in
  steps_while_ascending * (speed_ascending + x) = steps_while_descending * (speed_descending - x) / walking_speed_ratio →
  75 * (1 + 0.6) = 120 :=
by
  intro h
  sorry

end escalator_length_is_120_l259_259728


namespace min_kinds_of_candies_l259_259867

theorem min_kinds_of_candies (candies : ℕ) (even_distance_candies : ∀ i j : ℕ, i ≠ j → i < candies → j < candies → is_even (j - i - 1)) :
  candies = 91 → 46 ≤ candies :=
by
  assume h1 : candies = 91
  sorry

end min_kinds_of_candies_l259_259867


namespace probability_of_specific_combination_l259_259871

def count_all_clothes : ℕ := 6 + 7 + 8 + 3
def choose4_out_of_24 : ℕ := Nat.choose 24 4
def choose1_shirt : ℕ := 6
def choose1_pair_shorts : ℕ := 7
def choose1_pair_socks : ℕ := 8
def choose1_hat : ℕ := 3
def favorable_outcomes : ℕ := choose1_shirt * choose1_pair_shorts * choose1_pair_socks * choose1_hat
def probability_of_combination : ℚ := favorable_outcomes / choose4_out_of_24

theorem probability_of_specific_combination :
  probability_of_combination = 144 / 1815 := by
sorry

end probability_of_specific_combination_l259_259871


namespace chinese_horses_problem_l259_259743

variables (x y : ℕ)

theorem chinese_horses_problem (h1 : x + y = 100) (h2 : 3 * x + (y / 3) = 100) :
  (x + y = 100) ∧ (3 * x + (y / 3) = 100) :=
by
  sorry

end chinese_horses_problem_l259_259743


namespace sin_of_angle_l259_259814

theorem sin_of_angle (θ : ℝ) (h : Real.cos (θ + Real.pi) = -1/3) :
  Real.sin (2*θ + Real.pi/2) = -7/9 :=
by
  sorry

end sin_of_angle_l259_259814


namespace find_AC_l259_259987

def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

theorem find_AC :
  let AB := (2, 3)
  let BC := (1, -4)
  vector_add AB BC = (3, -1) :=
by 
  sorry

end find_AC_l259_259987


namespace three_irreducible_fractions_prod_eq_one_l259_259483

-- Define the set of numbers available for use
def available_numbers : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a structure for an irreducible fraction
structure irreducible_fraction :=
(num : ℕ)
(denom : ℕ)
(h_coprime : Nat.gcd num denom = 1)
(h_in_set : num ∈ available_numbers ∧ denom ∈ available_numbers)

-- Definition of the main proof problem
theorem three_irreducible_fractions_prod_eq_one :
  ∃ (f1 f2 f3 : irreducible_fraction), 
    f1.num * f2.num * f3.num = f1.denom * f2.denom * f3.denom ∧ 
    f1.num ≠ f2.num ∧ f1.num ≠ f3.num ∧ f2.num ≠ f3.num ∧ 
    f1.denom ≠ f2.denom ∧ f1.denom ≠ f3.denom ∧ f2.denom ≠ f3.denom := 
by
  sorry

end three_irreducible_fractions_prod_eq_one_l259_259483


namespace final_selling_price_l259_259772

-- Conditions
variable (x : ℝ)
def original_price : ℝ := x
def first_discount : ℝ := 0.8 * x
def additional_reduction : ℝ := 10

-- Statement of the problem
theorem final_selling_price (x : ℝ) : (0.8 * x) - 10 = 0.8 * x - 10 :=
by sorry

end final_selling_price_l259_259772


namespace combined_weight_loss_l259_259787

theorem combined_weight_loss (a_weekly_loss : ℝ) (a_weeks : ℕ) (x_weekly_loss : ℝ) (x_weeks : ℕ)
  (h1 : a_weekly_loss = 1.5) (h2 : a_weeks = 10) (h3 : x_weekly_loss = 2.5) (h4 : x_weeks = 8) :
  a_weekly_loss * a_weeks + x_weekly_loss * x_weeks = 35 := 
by
  -- We will not provide the proof body; the goal is to ensure the statement compiles.
  sorry

end combined_weight_loss_l259_259787


namespace complementary_set_count_is_correct_l259_259308

inductive Shape
| circle | square | triangle | hexagon

inductive Color
| red | blue | green

inductive Shade
| light | medium | dark

structure Card :=
(shape : Shape)
(color : Color)
(shade : Shade)

def deck : List Card :=
  -- (Note: Explicitly listing all 36 cards would be too verbose, pseudo-defining it for simplicity)
  [(Card.mk Shape.circle Color.red Shade.light),
   (Card.mk Shape.circle Color.red Shade.medium), 
   -- and so on for all 36 unique combinations...
   (Card.mk Shape.hexagon Color.green Shade.dark)]

def is_complementary (c1 c2 c3 : Card) : Prop :=
  ((c1.shape ≠ c2.shape ∧ c2.shape ≠ c3.shape ∧ c1.shape ≠ c3.shape) ∨ (c1.shape = c2.shape ∧ c2.shape = c3.shape)) ∧ 
  ((c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c1.color ≠ c3.color) ∨ (c1.color = c2.color ∧ c2.color = c3.color)) ∧
  ((c1.shade ≠ c2.shade ∧ c2.shade ≠ c3.shade ∧ c1.shade ≠ c3.shade) ∨ (c1.shade = c2.shade ∧ c2.shade = c3.shade))

noncomputable def count_complementary_sets : ℕ :=
  -- (Note: Implementation here is a placeholder. Actual counting logic would be non-trivial.)
  1836 -- placeholding the expected count

theorem complementary_set_count_is_correct :
  count_complementary_sets = 1836 :=
by
  trivial

end complementary_set_count_is_correct_l259_259308


namespace exists_convex_polygon_with_n_axes_of_symmetry_l259_259402

theorem exists_convex_polygon_with_n_axes_of_symmetry (n : ℕ) : 
  ∃ (P : Polygon ℝ), Convex P ∧ P.axes_of_symmetry.count = n := sorry

end exists_convex_polygon_with_n_axes_of_symmetry_l259_259402


namespace range_of_reciprocals_l259_259565

theorem range_of_reciprocals (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h_neq : a ≠ b) (h_sum : a + b = 1) :
  4 < (1 / a + 1 / b) :=
sorry

end range_of_reciprocals_l259_259565


namespace sarahs_monthly_fee_l259_259605

noncomputable def fixed_monthly_fee (x y : ℝ) : Prop :=
  x + 4 * y = 30.72 ∧ 1.1 * x + 8 * y = 54.72

theorem sarahs_monthly_fee : ∃ x y : ℝ, fixed_monthly_fee x y ∧ x = 7.47 :=
by
  sorry

end sarahs_monthly_fee_l259_259605


namespace thief_speed_l259_259784

theorem thief_speed (v : ℝ) (hv : v > 0) : 
  let head_start_duration := (1/2 : ℝ)  -- 30 minutes, converted to hours
  let owner_speed := (75 : ℝ)  -- speed of owner in kmph
  let chase_duration := (2 : ℝ)  -- duration of the chase in hours
  let distance_by_owner := owner_speed * chase_duration  -- distance covered by the owner
  let total_distance_thief := head_start_duration * v + chase_duration * v  -- total distance covered by the thief
  distance_by_owner = 150 ->  -- given that owner covers 150 km
  total_distance_thief = 150  -- and so should the thief
  -> v = 60 := sorry

end thief_speed_l259_259784


namespace find_x_l259_259011

theorem find_x (x : ℝ) (h : 2^10 = 32^x) : x = 2 := 
by 
  {
    sorry
  }

end find_x_l259_259011


namespace chantal_gain_l259_259796

variable (sweaters balls cost_selling cost_yarn total_gain : ℕ)

def chantal_knits_sweaters : Prop :=
  sweaters = 28 ∧
  balls = 4 ∧
  cost_yarn = 6 ∧
  cost_selling = 35 ∧
  total_gain = (sweaters * cost_selling) - (sweaters * balls * cost_yarn)

theorem chantal_gain : chantal_knits_sweaters sweaters balls cost_selling cost_yarn total_gain → total_gain = 308 :=
by sorry

end chantal_gain_l259_259796


namespace sum_max_min_interval_l259_259418

def f (x : ℝ) : ℝ := 2 * x^2 - 6 * x + 1

theorem sum_max_min_interval (a b : ℝ) (h₁ : a = -1) (h₂ : b = 1) :
  let M := max (f a) (f b)
  let m := min (f a) (f b)
  M + m = 6 :=
by
  rw [h₁, h₂]
  let M := max (f (-1)) (f 1)
  let m := min (f (-1)) (f 1)
  sorry

end sum_max_min_interval_l259_259418


namespace solve_for_x_and_n_l259_259948

theorem solve_for_x_and_n (x n : ℕ) : 2^n = x^2 + 1 ↔ (x = 0 ∧ n = 0) ∨ (x = 1 ∧ n = 1) := 
sorry

end solve_for_x_and_n_l259_259948


namespace min_number_of_candy_kinds_l259_259861

theorem min_number_of_candy_kinds (n : ℕ) (h : n = 91)
  (even_distance_condition : ∀ i j : ℕ, i ≠ j → n > i ∧ n > j → 
    (∃k : ℕ, i - j = 2 * k) ∨ (∃k : ℕ, j - i = 2 * k) ) :
  91 / 2 =  46 := by
  sorry

end min_number_of_candy_kinds_l259_259861


namespace compare_abc_l259_259036

def a : ℝ := 2 * log 1.01
def b : ℝ := log 1.02
def c : ℝ := real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l259_259036


namespace midpoint_coordinates_l259_259152

theorem midpoint_coordinates (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 2) (hy1 : y1 = 9) (hx2 : x2 = 8) (hy2 : y2 = 3) :
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  (mx, my) = (5, 6) :=
by
  rw [hx1, hy1, hx2, hy2]
  sorry

end midpoint_coordinates_l259_259152


namespace six_star_three_l259_259364

def binary_op (x y : ℕ) : ℕ := 4 * x + 5 * y - x * y

theorem six_star_three : binary_op 6 3 = 21 := by
  sorry

end six_star_three_l259_259364


namespace negative_solution_iff_sum_zero_l259_259318

theorem negative_solution_iff_sum_zero (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔
  a + b + c = 0 :=
by
  sorry

end negative_solution_iff_sum_zero_l259_259318


namespace solution_set_for_f_gt_0_l259_259995

noncomputable def f (x : ℝ) : ℝ := sorry

theorem solution_set_for_f_gt_0
  (odd_f : ∀ x : ℝ, f (-x) = -f x)
  (f_one_eq_zero : f 1 = 0)
  (ineq_f : ∀ x : ℝ, x > 0 → (x * (deriv^[2] f x) - f x) / x^2 > 0) :
  { x : ℝ | f x > 0 } = { x : ℝ | -1 < x ∧ x < 0 } ∪ { x : ℝ | 1 < x } :=
sorry

end solution_set_for_f_gt_0_l259_259995


namespace total_games_in_conference_l259_259277

-- Definitions based on the conditions
def numTeams := 16
def divisionTeams := 8
def gamesWithinDivisionPerTeam := 21
def gamesAcrossDivisionPerTeam := 16
def totalGamesPerTeam := 37
def totalGameCount := 592
def actualGameCount := 296

-- Proof statement
theorem total_games_in_conference : actualGameCount = (totalGameCount / 2) :=
  by sorry

end total_games_in_conference_l259_259277


namespace convert_base_7_to_base_10_l259_259667

theorem convert_base_7_to_base_10 : 
  let base_7_number := 2 * 7^2 + 4 * 7^1 + 6 * 7^0 in
  base_7_number = 132 :=
by
  let base_7_number := 2 * 7^2 + 4 * 7^1 + 6 * 7^0
  have : base_7_number = 132 := by
    calc
      2 * 7^2 + 4 * 7^1 + 6 * 7^0 = 2 * 49 + 4 * 7 + 6 * 1 : by refl
                               ... = 98 + 28 + 6         : by simp
                               ... = 132                : by norm_num
  exact this

end convert_base_7_to_base_10_l259_259667


namespace find_sum_of_common_ratios_l259_259113

-- Definition of the problem conditions
def is_geometric_sequence (a b c : ℕ) (k : ℕ) (r : ℕ) : Prop :=
  b = k * r ∧ c = k * r * r

-- Main theorem statement
theorem find_sum_of_common_ratios (k p r a_2 a_3 b_2 b_3 : ℕ) 
  (hk : k ≠ 0)
  (hp_neq_r : p ≠ r)
  (hp_seq : is_geometric_sequence k a_2 a_3 k p)
  (hr_seq : is_geometric_sequence k b_2 b_3 k r)
  (h_eq : a_3 - b_3 = 3 * (a_2 - b_2)) :
  p + r = 3 :=
sorry

end find_sum_of_common_ratios_l259_259113


namespace missing_digits_pairs_l259_259436

theorem missing_digits_pairs (x y : ℕ) : (2 + 4 + 6 + x + y + 8) % 9 = 0 ↔ x + y = 7 := by
  sorry

end missing_digits_pairs_l259_259436


namespace area_of_perpendicular_triangle_l259_259588

theorem area_of_perpendicular_triangle 
  (S R d : ℝ) (S' : ℝ) -- defining the variables and constants
  (h1 : S > 0) (h2 : R > 0) (h3 : d ≥ 0) :
  S' = (S / 4) * |1 - (d^2 / R^2)| := 
sorry

end area_of_perpendicular_triangle_l259_259588


namespace ria_number_is_2_l259_259172

theorem ria_number_is_2 
  (R S : ℕ) 
  (consecutive : R = S + 1 ∨ S = R + 1) 
  (R_positive : R > 0) 
  (S_positive : S > 0) 
  (R_not_1 : R ≠ 1) 
  (Sylvie_does_not_know : S ≠ 1) 
  (Ria_knows_after_Sylvie : ∃ (R_known : ℕ), R_known = R) :
  R = 2 :=
sorry

end ria_number_is_2_l259_259172


namespace An_is_integer_for_all_n_l259_259350

noncomputable def sin_theta (a b : ℕ) : ℝ :=
  if h : a^2 + b^2 ≠ 0 then (2 * a * b) / (a^2 + b^2) else 0

theorem An_is_integer_for_all_n (a b : ℕ) (n : ℕ) (h₁ : a > b) (h₂ : 0 < sin_theta a b) (h₃ : sin_theta a b < 1) :
  ∃ k : ℤ, ∀ n : ℕ, ((a^2 + b^2)^n * sin_theta a b) = k :=
sorry

end An_is_integer_for_all_n_l259_259350


namespace stretched_curve_l259_259089

noncomputable def transformed_curve (x : ℝ) : ℝ :=
  2 * Real.sin (x / 3 + Real.pi / 3)

theorem stretched_curve (y x : ℝ) :
  y = 2 * Real.sin (x + Real.pi / 3) → y = transformed_curve x := by
  intro h
  sorry

end stretched_curve_l259_259089


namespace bus_speed_l259_259465

theorem bus_speed (t : ℝ) (d : ℝ) (h : t = 42 / 60) (d_eq : d = 35) : d / t = 50 :=
by
  -- Assume
  sorry

end bus_speed_l259_259465


namespace Jung_age_is_26_l259_259109

-- Define the ages of Li, Zhang, and Jung
def Li : ℕ := 12
def Zhang : ℕ := 2 * Li
def Jung : ℕ := Zhang + 2

-- The goal is to prove Jung's age is 26 years
theorem Jung_age_is_26 : Jung = 26 :=
by
  -- Placeholder for the proof
  sorry

end Jung_age_is_26_l259_259109


namespace sale_price_of_trouser_l259_259714

theorem sale_price_of_trouser (original_price : ℝ) (discount_percentage : ℝ) (sale_price : ℝ) 
  (h1 : original_price = 100) (h2 : discount_percentage = 0.5) : sale_price = 50 :=
by
  sorry

end sale_price_of_trouser_l259_259714


namespace right_triangle_congruence_l259_259246

theorem right_triangle_congruence (A B C D : Prop) :
  (A → true) → (C → true) → (D → true) → (¬ B) → B :=
by
sorry

end right_triangle_congruence_l259_259246


namespace minimum_kinds_of_candies_l259_259866

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
    It turned out that between any two candies of the same kind, there is an even number of candies. 
    What is the minimum number of kinds of candies that could be? -/
theorem minimum_kinds_of_candies (n : ℕ) (h : 91 < 2 * n) : n ≥ 46 :=
sorry

end minimum_kinds_of_candies_l259_259866


namespace min_right_triangles_cover_equilateral_triangle_l259_259240

theorem min_right_triangles_cover_equilateral_triangle :
  let side_length_equilateral := 12
  let legs_right_triangle := 1
  let area_equilateral := (Real.sqrt 3 / 4) * side_length_equilateral ^ 2
  let area_right_triangle := (1 / 2) * legs_right_triangle * legs_right_triangle
  let triangles_needed := area_equilateral / area_right_triangle
  triangles_needed = 72 * Real.sqrt 3 := 
by 
  sorry

end min_right_triangles_cover_equilateral_triangle_l259_259240


namespace lily_cups_in_order_l259_259684

theorem lily_cups_in_order :
  ∀ (rose_rate lily_rate : ℕ) (order_rose_cups total_payment hourly_wage : ℕ),
    rose_rate = 6 →
    lily_rate = 7 →
    order_rose_cups = 6 →
    total_payment = 90 →
    hourly_wage = 30 →
    ∃ lily_cups: ℕ, lily_cups = 14 :=
by
  intros
  sorry

end lily_cups_in_order_l259_259684


namespace total_points_l259_259400

theorem total_points (total_players : ℕ) (paige_points : ℕ) (other_points : ℕ) (points_per_other_player : ℕ) :
  total_players = 5 →
  paige_points = 11 →
  points_per_other_player = 6 →
  other_points = (total_players - 1) * points_per_other_player →
  paige_points + other_points = 35 :=
by
  intro h_total_players h_paige_points h_points_per_other_player h_other_points
  sorry

end total_points_l259_259400


namespace badminton_players_l259_259024

theorem badminton_players (B T N Both Total: ℕ) 
  (h1: Total = 35)
  (h2: T = 18)
  (h3: N = 5)
  (h4: Both = 3)
  : B = 15 :=
by
  -- The proof block is intentionally left out.
  sorry

end badminton_players_l259_259024


namespace min_number_of_candy_kinds_l259_259862

theorem min_number_of_candy_kinds (n : ℕ) (h : n = 91)
  (even_distance_condition : ∀ i j : ℕ, i ≠ j → n > i ∧ n > j → 
    (∃k : ℕ, i - j = 2 * k) ∨ (∃k : ℕ, j - i = 2 * k) ) :
  91 / 2 =  46 := by
  sorry

end min_number_of_candy_kinds_l259_259862


namespace solvable_system_of_inequalities_l259_259811

theorem solvable_system_of_inequalities (n : ℕ) : 
  (∃ x : ℝ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (k < x ^ k ∧ x ^ k < k + 1)) ∧ (1 < x ∧ x < 2)) ↔ (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) :=
by sorry

end solvable_system_of_inequalities_l259_259811


namespace angle_bisector_length_l259_259578

noncomputable def length_angle_bisector (A B C : Type) [metric_space A B C] (angle_A angle_C : real) (diff_AC_AB : real) : 
  real :=
  5

theorem angle_bisector_length 
  (A B C : Type) [metric_space A B C]
  (angle_A : real) (angle_C : real)
  (diff_AC_AB : real) 
  (hA : angle_A = 20) 
  (hC : angle_C = 40) 
  (h_diff : diff_AC_AB = 5) :
  length_angle_bisector A B C angle_A angle_C diff_AC_AB = 5 :=
sorry

end angle_bisector_length_l259_259578


namespace inequality_a_c_b_l259_259057

theorem inequality_a_c_b :
  let a := 2 * Real.log 1.01
  let b := Real.log 1.02
  let c := Real.sqrt 1.04 - 1
  a > c ∧ c > b :=
by
  let a : ℝ := 2 * Real.log 1.01
  let b : ℝ := Real.log 1.02
  let c : ℝ := Real.sqrt 1.04 - 1
  split
  sorry
  sorry

end inequality_a_c_b_l259_259057


namespace sqrt_ac_bd_le_sqrt_ef_l259_259198

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem sqrt_ac_bd_le_sqrt_ef
  (a b c d e f : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e ∧ 0 ≤ f)
  (h1 : a + b ≤ e)
  (h2 : c + d ≤ f) :
  sqrt (a * c) + sqrt (b * d) ≤ sqrt (e * f) :=
by
  sorry

end sqrt_ac_bd_le_sqrt_ef_l259_259198


namespace perpendicular_vectors_l259_259236

theorem perpendicular_vectors (b : ℚ) : 
  (4 * b - 15 = 0) → (b = 15 / 4) :=
by
  intro h
  sorry

end perpendicular_vectors_l259_259236


namespace solve_for_x_l259_259209

theorem solve_for_x (x : ℚ) (h : (x + 10) / (x - 4) = (x + 3) / (x - 6)) : x = 48 / 5 :=
sorry

end solve_for_x_l259_259209


namespace negative_solution_condition_l259_259339

variable {a b c x y : ℝ}

theorem negative_solution_condition (h1 : a * x + b * y = c)
    (h2 : b * x + c * y = a)
    (h3 : c * x + a * y = b)
    (hx : x < 0)
    (hy : y < 0) :
    a + b + c = 0 :=
sorry

end negative_solution_condition_l259_259339


namespace divides_2_pow_26k_plus_2_plus_3_by_19_l259_259923

theorem divides_2_pow_26k_plus_2_plus_3_by_19 (k : ℕ) : 19 ∣ (2^(26*k+2) + 3) := 
by
  sorry

end divides_2_pow_26k_plus_2_plus_3_by_19_l259_259923


namespace smallest_five_digit_divisible_by_53_and_3_l259_259099

/-- The smallest five-digit positive integer divisible by 53 and 3 is 10062 -/
theorem smallest_five_digit_divisible_by_53_and_3 : ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 53 = 0 ∧ n % 3 = 0 ∧ ∀ m : ℕ, 10000 ≤ m ∧ m < 100000 ∧ m % 53 = 0 ∧ m % 3 = 0 → n ≤ m ∧ n = 10062 :=
by
  sorry

end smallest_five_digit_divisible_by_53_and_3_l259_259099


namespace perimeter_of_wheel_K_l259_259256

theorem perimeter_of_wheel_K
  (L_turns_K : 4 / 5 = 1 / (length_of_K / length_of_L))
  (L_turns_M : 6 / 7 = 1 / (length_of_L / length_of_M))
  (M_perimeter : length_of_M = 30) :
  length_of_K = 28 := 
sorry

end perimeter_of_wheel_K_l259_259256


namespace geom_seq_a5_a6_eq_180_l259_259872

theorem geom_seq_a5_a6_eq_180 (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a (n+1) = a n * q)
  (cond1 : a 1 + a 2 = 20)
  (cond2 : a 3 + a 4 = 60) :
  a 5 + a 6 = 180 :=
sorry

end geom_seq_a5_a6_eq_180_l259_259872


namespace martha_cards_l259_259065

theorem martha_cards :
  let initial_cards := 3
  let emily_cards := 25
  let alex_cards := 43
  let jenny_cards := 58
  let sam_cards := 14
  initial_cards + emily_cards + alex_cards + jenny_cards - sam_cards = 115 := 
by
  sorry

end martha_cards_l259_259065


namespace count_inverses_mod_11_l259_259545

theorem count_inverses_mod_11 : 
  (Finset.filter (λ x : ℕ, Nat.coprime x 11) (Finset.range 11)).card = 10 := 
by
  sorry

end count_inverses_mod_11_l259_259545


namespace equal_roots_quadratic_eq_l259_259165

theorem equal_roots_quadratic_eq (m n : ℝ) (h : m^2 - 4 * n = 0) : m = 2 ∧ n = 1 :=
by
  sorry

end equal_roots_quadratic_eq_l259_259165


namespace compare_abc_l259_259038

def a : ℝ := 2 * log 1.01
def b : ℝ := log 1.02
def c : ℝ := real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l259_259038


namespace converse_proposition_inverse_proposition_contrapositive_proposition_l259_259108

theorem converse_proposition (x y : ℝ) : (xy = 0 → x^2 + y^2 = 0) = false :=
sorry

theorem inverse_proposition (x y : ℝ) : (x^2 + y^2 ≠ 0 → xy ≠ 0) = false :=
sorry

theorem contrapositive_proposition (x y : ℝ) : (xy ≠ 0 → x^2 + y^2 ≠ 0) = true :=
sorry

end converse_proposition_inverse_proposition_contrapositive_proposition_l259_259108


namespace pages_to_read_tomorrow_l259_259894

-- Define the problem setup
def total_pages : ℕ := 100
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5

-- Define the total pages read after two days
def pages_read_in_two_days : ℕ := pages_yesterday + pages_today

-- Define the number of pages left to read
def pages_left_to_read (total_pages read_so_far : ℕ) : ℕ := total_pages - read_so_far

-- Prove that the number of pages to read tomorrow is 35
theorem pages_to_read_tomorrow :
  pages_left_to_read total_pages pages_read_in_two_days = 35 :=
by
  -- Proof is omitted
  sorry

end pages_to_read_tomorrow_l259_259894


namespace jessica_flowers_problem_l259_259582

theorem jessica_flowers_problem
(initial_roses initial_daisies : ℕ)
(thrown_roses thrown_daisies : ℕ)
(current_roses current_daisies : ℕ)
(cut_roses cut_daisies : ℕ)
(h_initial_roses : initial_roses = 21)
(h_initial_daisies : initial_daisies = 17)
(h_thrown_roses : thrown_roses = 34)
(h_thrown_daisies : thrown_daisies = 25)
(h_current_roses : current_roses = 15)
(h_current_daisies : current_daisies = 10)
(h_cut_roses : cut_roses = (thrown_roses - initial_roses) + current_roses)
(h_cut_daisies : cut_daisies = (thrown_daisies - initial_daisies) + current_daisies) :
thrown_roses + thrown_daisies - (cut_roses + cut_daisies) = 13 := by
  sorry

end jessica_flowers_problem_l259_259582


namespace sufficient_but_not_necessary_l259_259215

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, (0 < x ∧ x < 2) → (x < 2)) ∧ ¬(∀ x : ℝ, (0 < x ∧ x < 2) ↔ (x < 2)) :=
sorry

end sufficient_but_not_necessary_l259_259215


namespace total_matches_correct_total_points_earthlings_correct_total_players_is_square_l259_259663

-- Definitions
variables (t a : ℕ)

-- Part (a): Total number of matches
def total_matches : ℕ := (t + a) * (t + a - 1) / 2

-- Part (b): Total points of the Earthlings
def total_points_earthlings : ℕ := (t * (t - 1)) / 2 + (a * (a - 1)) / 2

-- Part (c): Total number of players is a perfect square
def is_total_players_square : Prop := ∃ k : ℕ, (t + a) = k * k

-- Lean statements
theorem total_matches_correct : total_matches t a = (t + a) * (t + a - 1) / 2 := 
by sorry

theorem total_points_earthlings_correct : total_points_earthlings t a = (t * (t - 1)) / 2 + (a * (a - 1)) / 2 := 
by sorry

theorem total_players_is_square : is_total_players_square t a := by sorry

end total_matches_correct_total_points_earthlings_correct_total_players_is_square_l259_259663


namespace num_females_math_not_english_is_15_l259_259272

-- Define the conditions
def male_math := 120
def female_math := 80
def female_english := 120
def male_english := 80
def total_students := 260
def both_male := 75

def female_math_not_english : Nat :=
  female_math - (female_english + female_math - (total_students - (male_math + male_english - both_male)))

theorem num_females_math_not_english_is_15 :
  female_math_not_english = 15 :=
by
  -- This is where the proof will be, but for now, we use 'sorry' to skip it.
  sorry

end num_females_math_not_english_is_15_l259_259272


namespace sequence_prob_no_three_consecutive_ones_l259_259466

-- Definitions
def b : ℕ → ℕ
| 0     := 1
| 1     := 2
| 2     := 4
| (n+3) := b n + b (n + 1) + b (n + 2)

-- Theorem statement
theorem sequence_prob_no_three_consecutive_ones : 
  let P := (b 12) / (2^12) in
  ∃ m n : ℕ, nat.coprime m n ∧ P = m / n ∧ m + n = 5801 := 
by sorry

end sequence_prob_no_three_consecutive_ones_l259_259466


namespace simplify_expression_l259_259075

variable (x : ℝ)

theorem simplify_expression :
  (3 * x - 6) * (2 * x + 8) - (x + 6) * (3 * x + 1) = 3 * x^2 - 7 * x - 54 :=
by
  sorry

end simplify_expression_l259_259075


namespace range_of_a_l259_259969

def new_operation (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, new_operation x (x - a) > 1) ↔ (a < -3 ∨ 1 < a) := 
by
  sorry

end range_of_a_l259_259969


namespace find_salary_May_l259_259212

-- Define the salaries for each month as variables
variables (J F M A May : ℝ)

-- Declare the conditions as hypotheses
def avg_salary_Jan_to_Apr := (J + F + M + A) / 4 = 8000
def avg_salary_Feb_to_May := (F + M + A + May) / 4 = 8100
def salary_Jan := J = 6100

-- The theorem stating the salary for the month of May
theorem find_salary_May (h1 : avg_salary_Jan_to_Apr J F M A) (h2 : avg_salary_Feb_to_May F M A May) (h3 : salary_Jan J) :
  May = 6500 :=
  sorry

end find_salary_May_l259_259212


namespace boxes_with_no_items_l259_259139

-- Definitions of each condition as given in the problem
def total_boxes : Nat := 15
def pencil_boxes : Nat := 8
def pen_boxes : Nat := 5
def marker_boxes : Nat := 3
def pen_pencil_boxes : Nat := 4
def all_three_boxes : Nat := 1

-- The theorem to prove
theorem boxes_with_no_items : 
     (total_boxes - ((pen_pencil_boxes - all_three_boxes)
                     + (pencil_boxes - pen_pencil_boxes - all_three_boxes)
                     + (pen_boxes - pen_pencil_boxes - all_three_boxes)
                     + (marker_boxes - all_three_boxes)
                     + all_three_boxes)) = 5 := 
by 
  -- This is where the proof would go, but we'll use sorry to indicate it's skipped.
  sorry

end boxes_with_no_items_l259_259139


namespace total_cost_of_constructing_the_path_l259_259270

open Real

-- Define the conditions
def length_field : ℝ := 75
def width_field : ℝ := 55
def path_width : ℝ := 2.8
def area_path_given : ℝ := 1518.72
def cost_per_sq_m : ℝ := 2

-- Define the total cost to be proven
def total_cost : ℝ := 3037.44

-- The statement to be proven
theorem total_cost_of_constructing_the_path :
  let outer_length := length_field + 2 * path_width
  let outer_width := width_field + 2 * path_width
  let total_area_incl_path := outer_length * outer_width
  let area_field := length_field * width_field
  let computed_area_path := total_area_incl_path - area_field
  let given_cost := area_path_given * cost_per_sq_m
  total_cost = given_cost := by
  sorry

end total_cost_of_constructing_the_path_l259_259270


namespace num_integers_with_inverse_mod_11_l259_259538

theorem num_integers_with_inverse_mod_11 : 
  (Finset.card (Finset.filter (λ x : ℕ, ∃ y : ℕ, x * y % 11 = 1) (Finset.range 11))) = 10 := 
by 
  sorry

end num_integers_with_inverse_mod_11_l259_259538


namespace neg_exists_eq_forall_l259_259170

theorem neg_exists_eq_forall (p : Prop) :
  (∀ x : ℝ, ¬(x^2 + 2*x = 3)) ↔ ¬(∃ x : ℝ, x^2 + 2*x = 3) := 
by
  sorry

end neg_exists_eq_forall_l259_259170


namespace min_n_exceeds_1000_l259_259694
-- Lean 4 statement for the problem:

theorem min_n_exceeds_1000 :
  ∃ n : ℕ, (∀ m < n, 2 ^ (m - Real.log (m + 1)) ≤ 1000) ∧ 2 ^ (n - Real.log (n + 1)) > 1000 :=
sorry

end min_n_exceeds_1000_l259_259694


namespace pages_to_read_tomorrow_l259_259892

-- Define the problem setup
def total_pages : ℕ := 100
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5

-- Define the total pages read after two days
def pages_read_in_two_days : ℕ := pages_yesterday + pages_today

-- Define the number of pages left to read
def pages_left_to_read (total_pages read_so_far : ℕ) : ℕ := total_pages - read_so_far

-- Prove that the number of pages to read tomorrow is 35
theorem pages_to_read_tomorrow :
  pages_left_to_read total_pages pages_read_in_two_days = 35 :=
by
  -- Proof is omitted
  sorry

end pages_to_read_tomorrow_l259_259892


namespace g_at_12_l259_259362

def g (n : ℤ) : ℤ := n^2 + 2*n + 23

theorem g_at_12 : g 12 = 191 := by
  -- proof skipped
  sorry

end g_at_12_l259_259362


namespace find_speed_second_part_l259_259282

noncomputable def speed_second_part (x : ℝ) (v : ℝ) : Prop :=
  let t1 := x / 65       -- Time to cover the first x km at 65 kmph
  let t2 := 2 * x / v    -- Time to cover the second 2x km at v kmph
  let avg_time := 3 * x / 26    -- Average speed of the entire journey
  t1 + t2 = avg_time

theorem find_speed_second_part (x : ℝ) (v : ℝ) (h : speed_second_part x v) : v = 86.67 :=
sorry -- Proof of the claim

end find_speed_second_part_l259_259282


namespace area_of_side_face_l259_259949

theorem area_of_side_face (L W H : ℝ) 
  (h1 : W * H = (1/2) * (L * W))
  (h2 : L * W = 1.5 * (H * L))
  (h3 : L * W * H = 648) :
  H * L = 72 := 
by
  sorry

end area_of_side_face_l259_259949


namespace zan_guo_gets_one_deer_l259_259926

noncomputable def a1 : ℚ := 5 / 3
noncomputable def sum_of_sequence (a1 : ℚ) (d : ℚ) : ℚ := 5 * a1 + (5 * 4 / 2) * d
noncomputable def d : ℚ := -1 / 3
noncomputable def a3 (a1 : ℚ) (d : ℚ) : ℚ := a1 + 2 * d

theorem zan_guo_gets_one_deer :
  a3 a1 d = 1 := by
  sorry

end zan_guo_gets_one_deer_l259_259926


namespace average_speed_of_car_l259_259633

noncomputable def average_speed (total_distance total_time : ℕ) : ℕ :=
  total_distance / total_time

theorem average_speed_of_car :
  let d1 := 80
  let d2 := 40
  let d3 := 60
  let d4 := 50
  let d5 := 90
  let d6 := 100
  let total_distance := d1 + d2 + d3 + d4 + d5 + d6
  let total_time := 6
  average_speed total_distance total_time = 70 :=
by
  let d1 := 80
  let d2 := 40
  let d3 := 60
  let d4 := 50
  let d5 := 90
  let d6 := 100
  let total_distance := d1 + d2 + d3 + d4 + d5 + d6
  let total_time := 6
  exact sorry

end average_speed_of_car_l259_259633


namespace eval_fraction_l259_259805

theorem eval_fraction : (3 : ℚ) / (2 - 5 / 4) = 4 := 
by 
  sorry

end eval_fraction_l259_259805


namespace final_balance_l259_259598

noncomputable def initial_balance : ℕ := 10
noncomputable def charity_donation : ℕ := 4
noncomputable def prize_amount : ℕ := 90
noncomputable def lost_at_first_slot : ℕ := 50
noncomputable def lost_at_second_slot : ℕ := 10
noncomputable def lost_at_last_slot : ℕ := 5
noncomputable def cost_of_water : ℕ := 1
noncomputable def cost_of_lottery_ticket : ℕ := 1
noncomputable def lottery_win : ℕ := 65

theorem final_balance : 
  initial_balance - charity_donation + prize_amount - (lost_at_first_slot + lost_at_second_slot + lost_at_last_slot) - (cost_of_water + cost_of_lottery_ticket) + lottery_win = 94 := 
by 
  -- This is the lean statement, the proof is not required as per instructions.
  sorry

end final_balance_l259_259598


namespace fraction_of_tadpoles_surviving_l259_259263

-- Definitions for the conditions
def frogs := 5
def tadpoles := 3 * frogs
def sustainable_frogs := 8

-- Main goal to prove
theorem fraction_of_tadpoles_surviving : (sustainable_frogs - frogs) / tadpoles = 1 / 5 := 
by
  sorry

end fraction_of_tadpoles_surviving_l259_259263


namespace find_x_l259_259009

theorem find_x (x : ℝ) (h : 2^10 = 32^x) : x = 2 :=
by
  have h1 : 32 = 2^5 := by norm_num
  rw [h1, pow_mul] at h
  have h2 : 2^(10) = 2^(5*x) := by exact h
  have h3 : 10 = 5 * x := by exact (pow_inj h2).2
  linarith

end find_x_l259_259009


namespace problem_statement_l259_259155

-- Definition of sum of digits function
def S (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

-- Definition of the function f₁
def f₁ (k : ℕ) : ℕ :=
  (S k) ^ 2

-- Definition of the function fₙ₊₁
def f : ℕ → ℕ → ℕ
| 0, k => k
| (n+1), k => f₁ (f n k)

-- Theorem stating the proof problem
theorem problem_statement : f 2005 (2 ^ 2006) = 169 :=
  sorry

end problem_statement_l259_259155


namespace amitabh_avg_expenditure_feb_to_jul_l259_259762

variable (expenditure_avg_jan_to_jun expenditure_jan expenditure_jul : ℕ)

theorem amitabh_avg_expenditure_feb_to_jul (h1 : expenditure_avg_jan_to_jun = 4200) 
  (h2 : expenditure_jan = 1200) (h3 : expenditure_jul = 1500) :
  (expenditure_avg_jan_to_jun * 6 - expenditure_jan + expenditure_jul) / 6 = 4250 := by
  -- Using the given conditions
  sorry

end amitabh_avg_expenditure_feb_to_jul_l259_259762


namespace n_equal_three_l259_259184

variable (m n : ℝ)

-- Conditions
def in_second_quadrant (m n : ℝ) : Prop := m < 0 ∧ n > 0
def distance_to_x_axis_eq_three (n : ℝ) : Prop := abs n = 3

-- Proof problem statement
theorem n_equal_three 
  (h1 : in_second_quadrant m n) 
  (h2 : distance_to_x_axis_eq_three n) : 
  n = 3 := 
sorry

end n_equal_three_l259_259184


namespace distance_house_to_market_l259_259648

-- Define the conditions
def distance_house_to_school := 50
def total_distance_walked := 140

-- Define the question as a theorem with the correct answer
theorem distance_house_to_market : 
  ∀ (house_to_school school_to_house total_distance market : ℕ), 
  house_to_school = distance_house_to_school →
  school_to_house = distance_house_to_school →
  total_distance = total_distance_walked →
  house_to_school + school_to_house + market = total_distance →
  market = 40 :=
by
  intros house_to_school school_to_house total_distance market 
  intro h1 h2 h3 h4
  sorry

end distance_house_to_market_l259_259648


namespace number_of_girls_is_4_l259_259371

variable (x : ℕ)

def number_of_boys : ℕ := 12

def average_score_boys : ℕ := 84

def average_score_girls : ℕ := 92

def average_score_class : ℕ := 86

theorem number_of_girls_is_4 
  (h : average_score_class = 
    (average_score_boys * number_of_boys + average_score_girls * x) / (number_of_boys + x))
  : x = 4 := 
sorry

end number_of_girls_is_4_l259_259371


namespace chess_amateurs_play_with_l259_259942

theorem chess_amateurs_play_with :
  ∃ n : ℕ, ∃ total_players : ℕ, total_players = 6 ∧
  (total_players * (total_players - 1)) / 2 = 12 ∧
  (n = total_players - 1 ∧ n = 5) :=
by
  sorry

end chess_amateurs_play_with_l259_259942


namespace total_campers_l259_259630

def campers_morning : ℕ := 36
def campers_afternoon : ℕ := 13
def campers_evening : ℕ := 49

theorem total_campers : campers_morning + campers_afternoon + campers_evening = 98 := by
  sorry

end total_campers_l259_259630


namespace employee_n_salary_l259_259945

theorem employee_n_salary (m n : ℝ) (h1: m + n = 594) (h2: m = 1.2 * n) : n = 270 := by
  sorry

end employee_n_salary_l259_259945


namespace count_inverses_mod_11_l259_259530

theorem count_inverses_mod_11 : (∃ n : ℕ, n = 10) :=
  have h : ∀ a ∈ finset.range 11, nat.gcd a 11 = 1 -> a ≠ 0 := by 
    intro a ha h1,
    apply (ne_of_lt (by linarith : a < 11)),
    apply nat.pos_of_ne_zero,
    intro hz,
    rw [hz, nat.gcd_zero_left] at h1,
    exact nat.prime.not_dvd_one (nat.prime_iff.2 ⟨sorry, sorry⟩) 11 h1,
  sorry

end count_inverses_mod_11_l259_259530


namespace father_has_4_chocolate_bars_left_l259_259911

noncomputable def chocolate_bars_given_to_father (initial_bars : ℕ) (num_people : ℕ) : ℕ :=
  let bars_per_person := initial_bars / num_people
  let bars_given := num_people * (bars_per_person / 2)
  bars_given

noncomputable def chocolate_bars_left_with_father (bars_given : ℕ) (bars_given_away : ℕ) : ℕ :=
  bars_given - bars_given_away

theorem father_has_4_chocolate_bars_left :
  ∀ (initial_bars num_people bars_given_away : ℕ), 
  initial_bars = 40 →
  num_people = 7 →
  bars_given_away = 10 →
  chocolate_bars_left_with_father (chocolate_bars_given_to_father initial_bars num_people) bars_given_away = 4 :=
by
  intros initial_bars num_people bars_given_away h_initial h_num h_given_away
  unfold chocolate_bars_given_to_father chocolate_bars_left_with_father
  rw [h_initial, h_num, h_given_away]
  exact sorry

end father_has_4_chocolate_bars_left_l259_259911


namespace shaded_area_correct_l259_259470

noncomputable def shaded_area (side_large side_small : ℝ) (pi_value : ℝ) : ℝ :=
  let area_large_square := side_large^2
  let area_large_circle := pi_value * (side_large / 2)^2
  let area_large_heart := area_large_square + area_large_circle
  let area_small_square := side_small^2
  let area_small_circle := pi_value * (side_small / 2)^2
  let area_small_heart := area_small_square + area_small_circle
  area_large_heart - area_small_heart

theorem shaded_area_correct : shaded_area 40 20 3.14 = 2142 :=
by
  -- Proof goes here
  sorry

end shaded_area_correct_l259_259470


namespace part1_part2_l259_259697

open Set Real

def M (x : ℝ) : Prop := x^2 - 3 * x - 18 ≤ 0
def N (x : ℝ) (a : ℝ) : Prop := 1 - a ≤ x ∧ x ≤ 2 * a + 1

theorem part1 (a : ℝ) (h : a = 3) : (Icc (-2 : ℝ) 6 = {x | M x ∧ N x a}) ∧ (compl {x | N x a} = Iic (-2) ∪ Ioi 7) :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x, M x ∧ N x a ↔ N x a) → a ≤ 5 / 2 :=
by
  sorry

end part1_part2_l259_259697


namespace jill_trips_to_fill_tank_l259_259881

   -- Defining the conditions
   def tank_capacity : ℕ := 600
   def bucket_capacity : ℕ := 5
   def jack_buckets_per_trip : ℕ := 2
   def jill_buckets_per_trip : ℕ := 1
   def jack_trip_rate : ℕ := 3
   def jill_trip_rate : ℕ := 2

   -- Calculate the amount of water Jack and Jill carry per trip
   def jack_gallons_per_trip : ℕ := jack_buckets_per_trip * bucket_capacity
   def jill_gallons_per_trip : ℕ := jill_buckets_per_trip * bucket_capacity

   -- Grouping the trips in the time it takes for Jill to complete her trips
   def total_gallons_per_group : ℕ := (jack_trip_rate * jack_gallons_per_trip) + (jill_trip_rate * jill_gallons_per_trip)

   -- Calculate the number of groups needed to fill the tank
   def groups_needed : ℕ := tank_capacity / total_gallons_per_group

   -- Calculate the total trips Jill makes
   def jill_total_trips : ℕ := groups_needed * jill_trip_rate

   -- The proof statement
   theorem jill_trips_to_fill_tank : jill_total_trips = 30 :=
   by
     -- Skipping the proof
     sorry
   
end jill_trips_to_fill_tank_l259_259881


namespace pages_to_read_tomorrow_l259_259891

-- Define the conditions
def total_pages : ℕ := 100
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5

-- Lean statement for the proof problem
theorem pages_to_read_tomorrow : (total_pages - (pages_yesterday + pages_today) = 35) :=
by
  let yesterday := pages_yesterday
  let today := pages_today
  let read_so_far := yesterday + today
  have read_so_far_eq : yesterday + today = 65 := by sorry
  have total_eq : total_pages - read_so_far = 35 := by sorry
  exact total_eq

end pages_to_read_tomorrow_l259_259891


namespace area_of_red_region_on_larger_sphere_l259_259962

/-- 
A smooth ball with a radius of 1 cm was dipped in red paint and placed between two 
absolutely smooth concentric spheres with radii of 4 cm and 6 cm, respectively
(the ball is outside the smaller sphere but inside the larger sphere).
As the ball moves and touches both spheres, it leaves a red mark. 
After traveling a closed path, a region outlined in red with an area of 37 square centimeters is formed on the smaller sphere. 
Find the area of the region outlined in red on the larger sphere. 
The answer should be 55.5 square centimeters.
-/
theorem area_of_red_region_on_larger_sphere
  (r1 r2 r3 : ℝ)
  (A_small : ℝ)
  (h_red_small_sphere : 37 = 2 * π * r2 * (A_small / (2 * π * r2)))
  (h_red_large_sphere : 55.5 = 2 * π * r3 * (A_small / (2 * π * r2))) :
  ∃ A_large : ℝ, A_large = 55.5 :=
by
  -- Definitions and conditions
  let r1 := 1  -- radius of small ball (1 cm)
  let r2 := 4  -- radius of smaller sphere (4 cm)
  let r3 := 6  -- radius of larger sphere (6 cm)

  -- Given: A small red area is 37 cm^2 on the smaller sphere.
  let A_small := 37

  -- Proof of the relationship of the spherical caps
  sorry

end area_of_red_region_on_larger_sphere_l259_259962


namespace simplify_fraction_l259_259207

theorem simplify_fraction : 5 * (18 / 7) * (21 / -54) = -5 := by
  sorry

end simplify_fraction_l259_259207


namespace factorization_correct_l259_259312

noncomputable def factor_expression (y : ℝ) : ℝ :=
  3 * y * (y - 5) + 4 * (y - 5)

theorem factorization_correct (y : ℝ) : factor_expression y = (3 * y + 4) * (y - 5) :=
by sorry

end factorization_correct_l259_259312


namespace geo_arith_seq_l259_259027

theorem geo_arith_seq
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : 0 < q)
  (h2 : ∀ n, a (n + 1) = a n * q)
  (h3 : a 1 + a 3 = 2 * a 2) :
  (a 4 + a 5) / (a 3 + a 4) = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end geo_arith_seq_l259_259027


namespace maximum_integer_value_of_fraction_is_12001_l259_259173

open Real

def max_fraction_value_12001 : Prop :=
  ∃ x : ℝ, (1 + 12 / (4 * x^2 + 12 * x + 8) : ℝ) = 12001

theorem maximum_integer_value_of_fraction_is_12001 :
  ∃ x : ℝ, 1 + (12 / (4 * x^2 + 12 * x + 8)) = 12001 :=
by
  -- Here you should provide the proof steps.
  sorry

end maximum_integer_value_of_fraction_is_12001_l259_259173


namespace profit_per_cake_l259_259293

theorem profit_per_cake (ingredient_cost : ℝ) (packaging_cost : ℝ) (selling_price : ℝ) (cake_count : ℝ)
    (h1 : ingredient_cost = 12) (h2 : packaging_cost = 1) (h3 : selling_price = 15) (h4 : cake_count = 2) :
    selling_price - (ingredient_cost / cake_count + packaging_cost) = 8 := by
  sorry

end profit_per_cake_l259_259293


namespace compare_a_b_c_l259_259041

noncomputable def a : ℝ := 2 * Real.ln 1.01
noncomputable def b : ℝ := Real.ln 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l259_259041


namespace range_of_f_l259_259168

noncomputable def f (x : ℝ) : ℝ := - (2 / (x - 1))

theorem range_of_f :
  {y : ℝ | ∃ x : ℝ, (0 ≤ x ∧ x < 1 ∨ 1 < x ∧ x ≤ 2) ∧ f x = y} = 
  {y : ℝ | y ≤ -2 ∨ 2 ≤ y} :=
by
  sorry

end range_of_f_l259_259168


namespace count_inverses_modulo_11_l259_259540

theorem count_inverses_modulo_11 : (∀ a : ℤ, 0 ≤ a ∧ a ≤ 10 → ∃ b : ℤ, a * b ≡ 1 [MOD 11]) → (finset.range 11).filter (λ a, (nat.gcd a 11 = 1)).card = 10 :=
by
  sorry

end count_inverses_modulo_11_l259_259540


namespace house_to_market_distance_l259_259650

-- Definitions of the conditions
def distance_to_school : ℕ := 50
def distance_back_home : ℕ := 50
def total_distance_walked : ℕ := 140

-- Statement of the problem
theorem house_to_market_distance :
  distance_to_market = total_distance_walked - (distance_to_school + distance_back_home) :=
by
  sorry

end house_to_market_distance_l259_259650


namespace annual_raise_l259_259381

-- Definitions based on conditions
def new_hourly_rate := 20
def new_weekly_hours := 40
def old_hourly_rate := 16
def old_weekly_hours := 25
def weeks_in_year := 52

-- Statement of the theorem
theorem annual_raise (new_hourly_rate new_weekly_hours old_hourly_rate old_weekly_hours weeks_in_year : ℕ) : 
  new_hourly_rate * new_weekly_hours * weeks_in_year - old_hourly_rate * old_weekly_hours * weeks_in_year = 20800 := 
  sorry -- Proof is omitted

end annual_raise_l259_259381


namespace sqrt_floor_square_18_l259_259973

-- Condition: the sqrt function and floor function
def sqrt (x : ℝ) : ℝ := Real.sqrt x
def floor (x : ℝ) : ℤ := Int.floor x

-- Mathematically equivalent proof problem
theorem sqrt_floor_square_18 : floor (sqrt 18) ^ 2 = 16 := 
by
  sorry

end sqrt_floor_square_18_l259_259973


namespace outer_circle_radius_l259_259625

theorem outer_circle_radius (C_inner : ℝ) (w : ℝ) (r_outer : ℝ) (h1 : C_inner = 440) (h2 : w = 14) :
  r_outer = (440 / (2 * Real.pi)) + 14 :=
by 
  have h_r_inner : r_outer = (440 / (2 * Real.pi)) + 14 := by sorry
  exact h_r_inner

end outer_circle_radius_l259_259625


namespace negative_solution_exists_l259_259322

theorem negative_solution_exists (a b c x y : ℝ) :
  (a * x + b * y = c ∧ b * x + c * y = a ∧ c * x + a * y = b) ∧ (x < 0 ∧ y < 0) ↔ a + b + c = 0 :=
sorry

end negative_solution_exists_l259_259322


namespace sarah_calculate_profit_l259_259733

noncomputable def sarah_total_profit (hot_day_price : ℚ) (regular_day_price : ℚ) (cost_per_cup : ℚ) (cups_per_day : ℕ) (hot_days : ℕ) (total_days : ℕ) : ℚ := 
  let hot_day_revenue := hot_day_price * cups_per_day * hot_days
  let regular_day_revenue := regular_day_price * cups_per_day * (total_days - hot_days)
  let total_revenue := hot_day_revenue + regular_day_revenue
  let total_cost := cost_per_cup * cups_per_day * total_days
  total_revenue - total_cost

theorem sarah_calculate_profit : 
  let hot_day_price := (20951704545454546 : ℚ) / 10000000000000000
  let regular_day_price := hot_day_price / 1.25
  let cost_per_cup := 75 / 100
  let cups_per_day := 32
  let hot_days := 4
  let total_days := 10
  sarah_total_profit hot_day_price regular_day_price cost_per_cup cups_per_day hot_days total_days = (34935102 : ℚ) / 10000000 :=
by
  sorry

end sarah_calculate_profit_l259_259733


namespace negative_solution_condition_l259_259335

variable {a b c x y : ℝ}

theorem negative_solution_condition (h1 : a * x + b * y = c)
    (h2 : b * x + c * y = a)
    (h3 : c * x + a * y = b)
    (hx : x < 0)
    (hy : y < 0) :
    a + b + c = 0 :=
sorry

end negative_solution_condition_l259_259335


namespace solution_set_of_inequality_l259_259422

theorem solution_set_of_inequality :
  { x : ℝ | (x - 2) * (x + 3) > 0 } = { x : ℝ | x < -3 } ∪ { x : ℝ | x > 2 } :=
sorry

end solution_set_of_inequality_l259_259422


namespace second_group_persons_l259_259453

open Nat

theorem second_group_persons
  (P : ℕ)
  (work_first_group : 39 * 24 * 5 = 4680)
  (work_second_group : P * 26 * 6 = 4680) :
  P = 30 :=
by
  sorry

end second_group_persons_l259_259453


namespace candy_problem_l259_259853

-- Definitions of conditions
def total_candies : ℕ := 91
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- The minimum number of kinds of candies function
def min_kinds : ℕ := 46

-- Lean statement for the proof problem
theorem candy_problem : 
  ∀ (kinds : ℕ), 
    (∀ c : ℕ, c < total_candies → c % kinds < 2) → (∃ n : ℕ, kinds = min_kinds) := 
sorry

end candy_problem_l259_259853


namespace peter_remaining_money_l259_259724

theorem peter_remaining_money :
  let initial_money := 500
  let potato_cost := 6 * 2
  let tomato_cost := 9 * 3
  let cucumber_cost := 5 * 4
  let banana_cost := 3 * 5
  let total_cost := potato_cost + tomato_cost + cucumber_cost + banana_cost
  let remaining_money := initial_money - total_cost
  remaining_money = 426 :=
by
  let initial_money := 500
  let potato_cost := 6 * 2
  let tomato_cost := 9 * 3
  let cucumber_cost := 5 * 4
  let banana_cost := 3 * 5
  let total_cost := potato_cost + tomato_cost + cucumber_cost + banana_cost
  let remaining_money := initial_money - total_cost
  show remaining_money = 426 from sorry

end peter_remaining_money_l259_259724


namespace max_distance_between_circle_and_ellipse_l259_259925

noncomputable def max_distance_PQ : ℝ :=
  1 + (3 * Real.sqrt 6) / 2

theorem max_distance_between_circle_and_ellipse :
  ∀ (P Q : ℝ × ℝ), (P.1^2 + (P.2 - 2)^2 = 1) → 
                   (Q.1^2 / 9 + Q.2^2 = 1) →
                   dist P Q ≤ max_distance_PQ :=
by
  intros P Q hP hQ
  sorry

end max_distance_between_circle_and_ellipse_l259_259925


namespace find_f_a5_a6_l259_259511

-- Define the function properties and initial conditions
variables {f : ℝ → ℝ} {a : ℕ → ℝ} {S : ℕ → ℝ}

-- Conditions for the function f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_period : ∀ x, f (3/2 - x) = f x
axiom f_minus_2 : f (-2) = -3

-- Initial sequence condition and recursive relation
axiom a_1 : a 1 = -1
axiom S_def : ∀ n, S n = 2 * a n + n
axiom seq_recursive : ∀ n ≥ 2, S (n - 1) = 2 * a (n - 1) + (n - 1)

-- Theorem to prove
theorem find_f_a5_a6 : f (a 5) + f (a 6) = 3 := by
  sorry

end find_f_a5_a6_l259_259511


namespace resulting_figure_has_25_sides_l259_259079

/-- Consider a sequential construction starting with an isosceles triangle, adding a rectangle 
    on one side, then a regular hexagon on a non-adjacent side of the rectangle, followed by a
    regular heptagon, another regular hexagon, and finally, a regular nonagon. -/
def sides_sequence : List ℕ := [3, 4, 6, 7, 6, 9]

/-- The number of sides exposed to the outside in the resulting figure. -/
def exposed_sides (sides : List ℕ) : ℕ :=
  let total_sides := sides.sum
  let adjacent_count := 2 + 2 + 2 + 2 + 1
  total_sides - adjacent_count

theorem resulting_figure_has_25_sides :
  exposed_sides sides_sequence = 25 := 
by
  sorry

end resulting_figure_has_25_sides_l259_259079


namespace fraction_evaluation_l259_259974

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem fraction_evaluation :
  (sqrt 2 * (sqrt 3 - sqrt 7)) / (2 * sqrt (3 + sqrt 5)) =
  (30 - 10 * sqrt 5 - 6 * sqrt 21 + 2 * sqrt 105) / 8 :=
by
  sorry

end fraction_evaluation_l259_259974


namespace count_invertible_mod_11_l259_259526

theorem count_invertible_mod_11 :
  ∃ (n : ℕ), n = 10 ∧ (∀ a, 0 ≤ a ∧ a ≤ 10 → ∃ x, (a * x) % 11 = 1 ↔ gcd a 11 = 1) := 
begin
  sorry,
end

end count_invertible_mod_11_l259_259526


namespace pages_to_read_tomorrow_l259_259897

theorem pages_to_read_tomorrow (total_pages : ℕ) 
                              (days : ℕ)
                              (pages_read_yesterday : ℕ)
                              (pages_read_today : ℕ)
                              (yesterday_diff : pages_read_today = pages_read_yesterday - 5)
                              (total_pages_eq : total_pages = 100)
                              (days_eq : days = 3)
                              (yesterday_eq : pages_read_yesterday = 35) : 
                              ∃ pages_read_tomorrow,  pages_read_tomorrow = total_pages - (pages_read_yesterday + pages_read_today) := 
                              by
  use 35
  sorry

end pages_to_read_tomorrow_l259_259897


namespace smallest_number_divisible_by_18_70_100_84_increased_by_3_l259_259100

theorem smallest_number_divisible_by_18_70_100_84_increased_by_3 :
  ∃ n : ℕ, (n + 3) % 18 = 0 ∧ (n + 3) % 70 = 0 ∧ (n + 3) % 100 = 0 ∧ (n + 3) % 84 = 0 ∧ n = 6297 :=
by
  sorry

end smallest_number_divisible_by_18_70_100_84_increased_by_3_l259_259100


namespace sufficient_not_necessary_l259_259818

variable (p q : Prop)

theorem sufficient_not_necessary (h1 : p ∧ q) (h2 : ¬¬p) : ¬¬p :=
by
  sorry

end sufficient_not_necessary_l259_259818


namespace largest_among_five_numbers_l259_259444

theorem largest_among_five_numbers :
  max (max (max (max (12345 + 1 / 3579) 
                       (12345 - 1 / 3579))
                   (12345 ^ (1 / 3579)))
               (12345 / (1 / 3579)))
           12345.3579 = 12345 / (1 / 3579) := sorry

end largest_among_five_numbers_l259_259444


namespace count_of_inverses_mod_11_l259_259535

theorem count_of_inverses_mod_11 : (Finset.filter (λ a : ℕ, ∃ b : ℕ, (a * b) % 11 = 1) (Finset.range 11)).card = 10 := 
sorry

end count_of_inverses_mod_11_l259_259535


namespace necessary_but_not_sufficient_l259_259252

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem necessary_but_not_sufficient (f : ℝ → ℝ) :
  (f 0 = 0) →
  (∀ x : ℝ, f (-x) = -f x) →
  ¬∀ f' : ℝ → ℝ, (f' 0 = 0 → ∀ y : ℝ, f' (-y) = -f' y)
:= by
  sorry

end necessary_but_not_sufficient_l259_259252


namespace inequality_l259_259071

theorem inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (a^3 + b^3 + a * b * c) + 1 / (b^3 + c^3 + a * b * c) + 1 / (c^3 + a^3 + a * b * c) ≤ 1 / (a * b * c) :=
sorry

end inequality_l259_259071


namespace fraction_of_15_smaller_by_20_l259_259238

/-- Define 80% of 40 -/
def eighty_percent_of_40 : ℝ := 0.80 * 40

/-- Define the fraction of 15 that we are looking for -/
def fraction_of_15 (x : ℝ) : ℝ := x * 15

/-- Define the problem statement -/
theorem fraction_of_15_smaller_by_20 : ∃ x : ℝ, fraction_of_15 x = eighty_percent_of_40 - 20 ∧ x = 4 / 5 :=
by
  sorry

end fraction_of_15_smaller_by_20_l259_259238


namespace avg_people_moving_per_hour_l259_259378

theorem avg_people_moving_per_hour (total_people : ℕ) (total_days : ℕ) (hours_per_day : ℕ) (h : total_people = 3000 ∧ total_days = 4 ∧ hours_per_day = 24) : 
  (total_people / (total_days * hours_per_day)).toFloat.round = 31 :=
by
  have h1 : total_people = 3000 := h.1;
  have h2 : total_days = 4 := h.2.1;
  have h3 : hours_per_day = 24 := h.2.2;
  rw [h1, h2, h3];
  sorry

end avg_people_moving_per_hour_l259_259378


namespace num_inverses_mod_11_l259_259560

theorem num_inverses_mod_11 : (finset.filter (λ x, nat.coprime x 11) (finset.range 11)).card = 10 :=
sorry

end num_inverses_mod_11_l259_259560


namespace pages_to_read_tomorrow_l259_259896

theorem pages_to_read_tomorrow (total_pages : ℕ) 
                              (days : ℕ)
                              (pages_read_yesterday : ℕ)
                              (pages_read_today : ℕ)
                              (yesterday_diff : pages_read_today = pages_read_yesterday - 5)
                              (total_pages_eq : total_pages = 100)
                              (days_eq : days = 3)
                              (yesterday_eq : pages_read_yesterday = 35) : 
                              ∃ pages_read_tomorrow,  pages_read_tomorrow = total_pages - (pages_read_yesterday + pages_read_today) := 
                              by
  use 35
  sorry

end pages_to_read_tomorrow_l259_259896


namespace triangle_bisector_length_l259_259579

theorem triangle_bisector_length (A B C : Type) [MetricSpace A] [MetricSpace B]
  [MetricSpace C] (angle_A angle_C : ℝ) (AC AB : ℝ) 
  (hAC : angle_A = 20) (hAngle_C : angle_C = 40) (hAC_minus_AB : AC - AB = 5) : ∃ BM : ℝ, BM = 5 :=
by
  sorry

end triangle_bisector_length_l259_259579


namespace negative_solution_condition_l259_259326

-- Define the system of equations
def system_of_equations (a b c x y : ℝ) :=
  a * x + b * y = c ∧
  b * x + c * y = a ∧
  c * x + a * y = b

-- State the theorem
theorem negative_solution_condition (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ system_of_equations a b c x y) ↔ a + b + c = 0 :=
by
  sorry

end negative_solution_condition_l259_259326


namespace true_discount_double_time_l259_259623

theorem true_discount_double_time (PV FV1 FV2 I1 I2 TD1 TD2 : ℕ) 
  (h1 : FV1 = 110)
  (h2 : TD1 = 10)
  (h3 : FV1 - TD1 = PV)
  (h4 : I1 = FV1 - PV)
  (h5 : FV2 = PV + 2 * I1)
  (h6 : TD2 = FV2 - PV) :
  TD2 = 20 := by
  sorry

end true_discount_double_time_l259_259623


namespace minimum_candy_kinds_l259_259851

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
     It turned out that between any two candies of the same kind, there is an even number of candies.
     Prove that the minimum number of kinds of candies that could be is 46. -/
theorem minimum_candy_kinds (n : ℕ) (candies : ℕ → ℕ) 
  (h_candies_length : ∀ i, i < 91 → candies i < n)
  (h_even_between : ∀ i j, i < j → candies i = candies j → even (j - i - 1)) :
  n ≥ 46 :=
sorry

end minimum_candy_kinds_l259_259851


namespace billy_has_2_cherries_left_l259_259300

-- Define the initial number of cherries
def initialCherries : Nat := 74

-- Define the number of cherries eaten
def eatenCherries : Nat := 72

-- Define the number of remaining cherries
def remainingCherries : Nat := initialCherries - eatenCherries

-- Theorem statement: Prove that remainingCherries is equal to 2
theorem billy_has_2_cherries_left : remainingCherries = 2 := by
  sorry

end billy_has_2_cherries_left_l259_259300


namespace prove_remaining_area_is_24_l259_259639

/-- A rectangular piece of paper with length 12 cm and width 8 cm has four identical isosceles 
right triangles with legs of 6 cm cut from it. Prove that the remaining area is 24 cm². --/
def remaining_area : ℕ := 
  let length := 12
  let width := 8
  let rect_area := length * width
  let triangle_leg := 6
  let triangle_area := (triangle_leg * triangle_leg) / 2
  let total_triangle_area := 4 * triangle_area
  rect_area - total_triangle_area

theorem prove_remaining_area_is_24 : (remaining_area = 24) :=
  by sorry

end prove_remaining_area_is_24_l259_259639


namespace students_per_group_correct_l259_259115

def total_students : ℕ := 850
def number_of_teachers : ℕ := 23
def students_per_group : ℕ := total_students / number_of_teachers

theorem students_per_group_correct : students_per_group = 36 := sorry

end students_per_group_correct_l259_259115


namespace sector_field_area_l259_259376

/-- Given a sector field with a circumference of 30 steps and a diameter of 16 steps, prove that its area is 120 square steps. --/
theorem sector_field_area (C : ℝ) (d : ℝ) (A : ℝ) : 
  C = 30 → d = 16 → A = 120 :=
by
  sorry

end sector_field_area_l259_259376


namespace smallest_log_log_x0_l259_259823

noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

theorem smallest_log_log_x0 (x₀ : ℝ) (h₀ : f x₀ = 0) (h_dom : 2 < x₀ ∧ x₀ < Real.exp 1) :
  min (min (Real.log x₀) (Real.log (Real.sqrt x₀))) (min (Real.log (Real.log x₀)) ((Real.log x₀)^2)) = Real.log (Real.log x₀) :=
sorry

end smallest_log_log_x0_l259_259823


namespace linear_function_points_relation_l259_259251

theorem linear_function_points_relation (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (h1 : y1 = 5 * x1 - 3) 
  (h2 : y2 = 5 * x2 - 3) 
  (h3 : x1 < x2) : 
  y1 < y2 :=
sorry

end linear_function_points_relation_l259_259251


namespace roman_numeral_calculation_l259_259732

def I : ℕ := 1
def V : ℕ := 5
def X : ℕ := 10
def L : ℕ := 50
def C : ℕ := 100
def D : ℕ := 500
def M : ℕ := 1000

theorem roman_numeral_calculation : 2 * M + 5 * L + 7 * X + 9 * I = 2329 := by
  sorry

end roman_numeral_calculation_l259_259732


namespace hyperbola_range_m_l259_259414

-- Define the condition that the equation represents a hyperbola
def isHyperbola (m : ℝ) : Prop := (2 + m) * (m + 1) < 0

-- The theorem stating the range of m given the condition
theorem hyperbola_range_m (m : ℝ) : isHyperbola m → -2 < m ∧ m < -1 := by
  sorry

end hyperbola_range_m_l259_259414


namespace find_a₉_l259_259689

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

axiom S_6_eq : S 6 = 3
axiom S_11_eq : S 11 = 18

noncomputable def a₉ : ℝ := sorry -- Define a₉ here, proof skipped by "sorry"

theorem find_a₉ (a : ℕ → ℝ) (S : ℕ → ℝ) :
  S 6 = 3 →
  S 11 = 18 →
  a₉ = 3 :=
by
  intros S_6_eq S_11_eq
  sorry -- Proof goes here

end find_a₉_l259_259689


namespace nanometers_to_scientific_notation_l259_259764

   theorem nanometers_to_scientific_notation :
     (0.000000001 : Float) = 1 * 10 ^ (-9) :=
   by
     sorry
   
end nanometers_to_scientific_notation_l259_259764


namespace arithmetic_sequence_a10_l259_259345

theorem arithmetic_sequence_a10 (a : ℕ → ℕ) (d : ℕ) 
  (h1 : a 1 = 1) (h3 : a 3 = 5) 
  (h_diff : d = (a 3 - a 1) / (3 - 1)) :
  a 10 = 19 := 
by 
  sorry

end arithmetic_sequence_a10_l259_259345


namespace count_invertible_mod_11_l259_259525

theorem count_invertible_mod_11 :
  ∃ (n : ℕ), n = 10 ∧ (∀ a, 0 ≤ a ∧ a ≤ 10 → ∃ x, (a * x) % 11 = 1 ↔ gcd a 11 = 1) := 
begin
  sorry,
end

end count_invertible_mod_11_l259_259525


namespace polygon_interior_angles_540_implies_pentagon_l259_259352

theorem polygon_interior_angles_540_implies_pentagon
  (n : ℕ) (H: 180 * (n - 2) = 540) : n = 5 :=
sorry

end polygon_interior_angles_540_implies_pentagon_l259_259352


namespace problem1_problem2_l259_259474

-- Statement for Problem 1
theorem problem1 :
  real.sqrt 9 - (-1 : ℝ) ^ 2022 - real.cbrt 27 + abs (1 - real.sqrt 2) = real.sqrt 2 - 2 :=
by
  sorry

-- Statement for Problem 2
theorem problem2 (x : ℝ) (h : 3 * x^3 = -24) : x = -2 :=
by
  sorry

end problem1_problem2_l259_259474


namespace divisibility_of_product_l259_259192

theorem divisibility_of_product (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h1 : a ∣ b^3) (h2 : b ∣ c^3) (h3 : c ∣ a^3) : abc ∣ (a + b + c) ^ 13 := by
  sorry

end divisibility_of_product_l259_259192


namespace total_shoes_tried_on_l259_259003

variable (T : Type)
variable (store1 store2 store3 store4 : T)
variable (pair_of_shoes : T → ℕ)
variable (c1 : pair_of_shoes store1 = 7)
variable (c2 : pair_of_shoes store2 = pair_of_shoes store1 + 2)
variable (c3 : pair_of_shoes store3 = 0)
variable (c4 : pair_of_shoes store4 = 2 * (pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3))

theorem total_shoes_tried_on (store1 store2 store3 store4 : T) (pair_of_shoes : T → ℕ) : 
  pair_of_shoes store1 = 7 →
  pair_of_shoes store2 = pair_of_shoes store1 + 2 →
  pair_of_shoes store3 = 0 →
  pair_of_shoes store4 = 2 * (pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3) →
  pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3 + pair_of_shoes store4 = 48 := by
  intro c1 c2 c3 c4
  sorry

end total_shoes_tried_on_l259_259003


namespace three_irreducible_fractions_prod_eq_one_l259_259480

-- Define the set of numbers available for use
def available_numbers : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a structure for an irreducible fraction
structure irreducible_fraction :=
(num : ℕ)
(denom : ℕ)
(h_coprime : Nat.gcd num denom = 1)
(h_in_set : num ∈ available_numbers ∧ denom ∈ available_numbers)

-- Definition of the main proof problem
theorem three_irreducible_fractions_prod_eq_one :
  ∃ (f1 f2 f3 : irreducible_fraction), 
    f1.num * f2.num * f3.num = f1.denom * f2.denom * f3.denom ∧ 
    f1.num ≠ f2.num ∧ f1.num ≠ f3.num ∧ f2.num ≠ f3.num ∧ 
    f1.denom ≠ f2.denom ∧ f1.denom ≠ f3.denom ∧ f2.denom ≠ f3.denom := 
by
  sorry

end three_irreducible_fractions_prod_eq_one_l259_259480


namespace change_calculation_l259_259187

def cost_of_apple : ℝ := 0.75
def amount_paid : ℝ := 5.00

theorem change_calculation : (amount_paid - cost_of_apple = 4.25) := by
  sorry

end change_calculation_l259_259187


namespace internal_diagonal_cubes_l259_259454

theorem internal_diagonal_cubes :
  let A := (120, 360, 400)
  let gcd_xy := gcd 120 360
  let gcd_yz := gcd 360 400
  let gcd_zx := gcd 400 120
  let gcd_xyz := gcd (gcd 120 360) 400
  let new_cubes := 120 + 360 + 400 - (gcd_xy + gcd_yz + gcd_zx) + gcd_xyz
  new_cubes = 720 :=
by
  -- Definitions
  let A := (120, 360, 400)
  let gcd_xy := Int.gcd 120 360
  let gcd_yz := Int.gcd 360 400
  let gcd_zx := Int.gcd 400 120
  let gcd_xyz := Int.gcd (Int.gcd 120 360) 400
  let new_cubes := 120 + 360 + 400 - (gcd_xy + gcd_yz + gcd_zx) + gcd_xyz

  -- Assertion
  exact Eq.refl new_cubes

end internal_diagonal_cubes_l259_259454


namespace annual_raise_l259_259382

-- Definitions based on conditions
def new_hourly_rate := 20
def new_weekly_hours := 40
def old_hourly_rate := 16
def old_weekly_hours := 25
def weeks_in_year := 52

-- Statement of the theorem
theorem annual_raise (new_hourly_rate new_weekly_hours old_hourly_rate old_weekly_hours weeks_in_year : ℕ) : 
  new_hourly_rate * new_weekly_hours * weeks_in_year - old_hourly_rate * old_weekly_hours * weeks_in_year = 20800 := 
  sorry -- Proof is omitted

end annual_raise_l259_259382


namespace negative_solution_exists_l259_259323

theorem negative_solution_exists (a b c x y : ℝ) :
  (a * x + b * y = c ∧ b * x + c * y = a ∧ c * x + a * y = b) ∧ (x < 0 ∧ y < 0) ↔ a + b + c = 0 :=
sorry

end negative_solution_exists_l259_259323


namespace triangle_is_isosceles_l259_259820

theorem triangle_is_isosceles
  (A B C : ℝ)
  (h_triangle : A + B + C = π)
  (h_condition : 2 * Real.cos B * Real.sin C = Real.sin A) :
  B = C :=
sorry

end triangle_is_isosceles_l259_259820


namespace andy_profit_per_cake_l259_259291

-- Definitions based on the conditions
def cost_of_ingredients (cakes : ℕ) : ℕ := if cakes = 2 then 12 else 0
def cost_of_packaging_per_cake : ℕ := 1
def selling_price_per_cake : ℕ := 15

-- Theorem stating the profit made per cake
theorem andy_profit_per_cake : ∀ (cakes : ℕ), cakes = 2 → 
(cost_of_ingredients cakes / cakes + cost_of_packaging_per_cake) = 7 →
selling_price_per_cake - (cost_of_ingredients cakes / cakes + cost_of_packaging_per_cake) = 8 :=
by
  intros cakes h_cakes cost_hyp
  have h1 : cost_of_ingredients cakes / cakes = 12 / 2 :=
    by rw [h_cakes]; refl
  have h2 : (12 / 2 + cost_of_packaging_per_cake) = 6 + 1 :=
    by rw [h1]; refl
  have h3 : (6 + 1) = 7 :=
    by refl
  rw [← h3] at cost_hyp
  have h4 : selling_price_per_cake - 7 = 8 :=
    by refl
  exact h4

end andy_profit_per_cake_l259_259291


namespace new_deck_card_count_l259_259028

-- Define the conditions
def cards_per_time : ℕ := 30
def times_per_week : ℕ := 3
def weeks : ℕ := 11
def decks : ℕ := 18
def total_cards_tear_per_week : ℕ := cards_per_time * times_per_week
def total_cards_tear : ℕ := total_cards_tear_per_week * weeks
def total_cards_in_decks (cards_per_deck : ℕ) : ℕ := decks * cards_per_deck

-- Define the theorem we need to prove
theorem new_deck_card_count :
  ∃ (x : ℕ), total_cards_in_decks x = total_cards_tear ↔ x = 55 := by
  sorry

end new_deck_card_count_l259_259028


namespace complement_A_in_U_l259_259358

universe u

-- Define the universal set U and set A.
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5, 7}

-- Define the complement of A in U.
def complement (A U: Set ℕ) : Set ℕ :=
  {x ∈ U | x ∉ A}

-- Statement to prove.
theorem complement_A_in_U :
  complement A U = {2, 4, 6} :=
sorry

end complement_A_in_U_l259_259358


namespace count_inverses_mod_11_l259_259532

def has_inverse_mod (a n : ℕ) : Prop :=
  ∃ b : ℕ, a * b ≡ 1 [MOD n]

theorem count_inverses_mod_11 : 
  (Finset.filter (λ a, has_inverse_mod a 11) (Finset.range 11)).card = 10 :=
by
  sorry

end count_inverses_mod_11_l259_259532


namespace initial_number_of_men_l259_259739

theorem initial_number_of_men (M : ℕ) 
  (h1 : M * 8 * 40 = (M + 30) * 6 * 50) 
  : M = 450 :=
by 
  sorry

end initial_number_of_men_l259_259739


namespace find_fractions_l259_259489

open Function

-- Define the set and the condition that all numbers must be used precisely once
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define what it means for fractions to multiply to 1 within the set
def fractions_mul_to_one (a b c d e f : ℕ) : Prop :=
  (a * c * e) = (b * d * f)

-- Define irreducibility condition for a fraction a/b
def irreducible_fraction (a b : ℕ) := 
  Nat.gcd a b = 1

-- Final main problem statement
theorem find_fractions :
  ∃ (a b c d e f : ℕ) (h₁ : a ∈ S) (h₂ : b ∈ S) (h₃ : c ∈ S) (h₄ : d ∈ S) (h₅ : e ∈ S) (h₆ : f ∈ S),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
  d ≠ e ∧ d ≠ f ∧ 
  e ≠ f ∧
  irreducible_fraction a b ∧ irreducible_fraction c d ∧ irreducible_fraction e f ∧
  fractions_mul_to_one a b c d e f := 
sorry

end find_fractions_l259_259489


namespace find_tuesday_temperature_l259_259086

variable (T W Th F : ℝ)

def average_temperature_1 : Prop := (T + W + Th) / 3 = 52
def average_temperature_2 : Prop := (W + Th + F) / 3 = 54
def friday_temperature : Prop := F = 53

theorem find_tuesday_temperature (h1 : average_temperature_1 T W Th) (h2 : average_temperature_2 W Th F) (h3 : friday_temperature F) :
  T = 47 :=
by
  sorry

end find_tuesday_temperature_l259_259086


namespace two_students_exist_l259_259768

theorem two_students_exist (scores : Fin 49 → Fin 8 × Fin 8 × Fin 8) :
  ∃ (i j : Fin 49), i ≠ j ∧ (scores i).1 ≥ (scores j).1 ∧ (scores i).2.1 ≥ (scores j).2.1 ∧ (scores i).2.2 ≥ (scores j).2.2 := 
by
  sorry

end two_students_exist_l259_259768


namespace remaining_tomatoes_to_cucumbers_ratio_l259_259262

theorem remaining_tomatoes_to_cucumbers_ratio 
  (initial_tomatoes initial_cucumbers : ℕ)
  (picked_tomatoes_yesterday picked_tomatoes_today picked_cucumbers : ℕ) :
  initial_tomatoes = 171 →
  initial_cucumbers = 225 →
  picked_tomatoes_yesterday = 134 →
  picked_tomatoes_today = 30 →
  picked_cucumbers = 157 →
  (initial_tomatoes - (picked_tomatoes_yesterday + picked_tomatoes_today)) = 7 →
  (initial_cucumbers - picked_cucumbers) = 68 →
  rat.mk 7 68 = (7 / 68 : ℚ) :=
by
  sorry

end remaining_tomatoes_to_cucumbers_ratio_l259_259262


namespace period_of_repeating_decimal_l259_259116

def is_100_digit_number_with_98_sevens (a : ℕ) : Prop :=
  ∃ (n : ℕ), n = 10^98 ∧ a = 1776 + 1777 * n

theorem period_of_repeating_decimal (a : ℕ) (h : is_100_digit_number_with_98_sevens a) : 
  (1:ℚ) / a == 1 / 99 := 
  sorry

end period_of_repeating_decimal_l259_259116


namespace tangent_line_equation_at_point_l259_259610

-- Defining the function and the point
def f (x : ℝ) : ℝ := x^2 + 2 * x
def point : ℝ × ℝ := (1, 3)

-- Main theorem stating the tangent line equation at the given point
theorem tangent_line_equation_at_point : 
  ∃ m b, (m = (2 * 1 + 2)) ∧ 
         (b = (3 - m * 1)) ∧ 
         (∀ x y, y = f x → y = m * x + b → 4 * x - y - 1 = 0) :=
by
  -- Proof is omitted and can be filled in later
  sorry

end tangent_line_equation_at_point_l259_259610


namespace cos_C_in_acute_triangle_l259_259826

theorem cos_C_in_acute_triangle 
  (a b c : ℝ) (A B C : ℝ) 
  (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (h_sides_angles : a * Real.cos B = 4 * c * Real.sin C - b * Real.cos A) 
  : Real.cos C = Real.sqrt 15 / 4 := 
sorry

end cos_C_in_acute_triangle_l259_259826


namespace no_nat_numbers_satisfy_l259_259731

theorem no_nat_numbers_satisfy (x y z k : ℕ) (hx : x < k) (hy : y < k) : x^k + y^k ≠ z^k := 
sorry

end no_nat_numbers_satisfy_l259_259731


namespace negative_solution_condition_l259_259338

variable {a b c x y : ℝ}

theorem negative_solution_condition (h1 : a * x + b * y = c)
    (h2 : b * x + c * y = a)
    (h3 : c * x + a * y = b)
    (hx : x < 0)
    (hy : y < 0) :
    a + b + c = 0 :=
sorry

end negative_solution_condition_l259_259338


namespace num_parallelograms_4x6_grid_l259_259705

noncomputable def numberOfParallelograms (m n : ℕ) : ℕ :=
  let numberOfRectangles := (Nat.choose (m + 1) 2) * (Nat.choose (n + 1) 2)
  let numberOfSquares := (m * n) + ((m - 1) * (n - 1)) + ((m - 2) * (n - 2)) + ((m - 3) * (n - 3))
  let numberOfRectanglesWithUnequalSides := numberOfRectangles - numberOfSquares
  2 * numberOfRectanglesWithUnequalSides

theorem num_parallelograms_4x6_grid : numberOfParallelograms 4 6 = 320 := by
  sorry

end num_parallelograms_4x6_grid_l259_259705


namespace base8_to_base10_sum_l259_259154

theorem base8_to_base10_sum (a b : ℕ) (h₁ : a = 1 * 8^3 + 4 * 8^2 + 5 * 8^1 + 3 * 8^0)
                            (h₂ : b = 5 * 8^2 + 6 * 8^1 + 7 * 8^0) :
                            ((a + b) = 2 * 8^3 + 1 * 8^2 + 4 * 8^1 + 4 * 8^0) →
                            (2 * 8^3 + 1 * 8^2 + 4 * 8^1 + 4 * 8^0 = 1124) :=
by {
  sorry
}

end base8_to_base10_sum_l259_259154


namespace find_cos_alpha_l259_259160

theorem find_cos_alpha 
  (α : ℝ) 
  (h₁ : Real.tan (π - α) = 3/4) 
  (h₂ : α ∈ Set.Ioo (π/2) π) 
: Real.cos α = -4/5 :=
sorry

end find_cos_alpha_l259_259160


namespace not_buy_either_l259_259941

-- Definitions
variables (n T C B : ℕ)
variables (h_n : n = 15)
variables (h_T : T = 9)
variables (h_C : C = 7)
variables (h_B : B = 3)

-- Theorem statement
theorem not_buy_either (n T C B : ℕ) (h_n : n = 15) (h_T : T = 9) (h_C : C = 7) (h_B : B = 3) :
  n - (T - B) - (C - B) - B = 2 :=
sorry

end not_buy_either_l259_259941


namespace part1_part2_part3_l259_259661

noncomputable def y1 (x : ℝ) : ℝ := 0.1 * x + 15
noncomputable def y2 (x : ℝ) : ℝ := 0.15 * x

-- Prove that the functions are as described
theorem part1 : ∀ x : ℝ, y1 x = 0.1 * x + 15 ∧ y2 x = 0.15 * x :=
by sorry

-- Prove that x = 300 results in equal charges for Packages A and B
theorem part2 : y1 300 = y2 300 :=
by sorry

-- Prove that Package A is more cost-effective when x > 300
theorem part3 : ∀ x : ℝ, x > 300 → y1 x < y2 x :=
by sorry

end part1_part2_part3_l259_259661


namespace pump_fills_tank_without_leak_l259_259269

variable (T : ℝ)
-- Condition: The effective rate with the leak is equal to the rate it takes for both to fill the tank.
def effective_rate_with_leak (T : ℝ) : Prop :=
  1 / T - 1 / 21 = 1 / 3.5

-- Conclude: the time it takes the pump to fill the tank without the leak
theorem pump_fills_tank_without_leak : effective_rate_with_leak T → T = 3 :=
by
  intro h
  sorry

end pump_fills_tank_without_leak_l259_259269


namespace rectangular_solid_surface_area_l259_259802

theorem rectangular_solid_surface_area (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (hvol : a * b * c = 455) : 
  let surface_area := 2 * (a * b + b * c + c * a)
  surface_area = 382 := by
-- proof
sorry

end rectangular_solid_surface_area_l259_259802


namespace lidia_money_left_l259_259592

theorem lidia_money_left 
  (cost_per_app : ℕ := 4) 
  (num_apps : ℕ := 15) 
  (total_money : ℕ := 66) 
  (discount_rate : ℚ := 0.15) :
  total_money - (num_apps * cost_per_app - (num_apps * cost_per_app * discount_rate)) = 15 := by 
  sorry

end lidia_money_left_l259_259592


namespace food_bank_remaining_after_four_weeks_l259_259704

def week1_donated : ℝ := 40
def week1_given_out : ℝ := 0.6 * week1_donated
def week1_remaining : ℝ := week1_donated - week1_given_out

def week2_donated : ℝ := 1.5 * week1_donated
def week2_given_out : ℝ := 0.7 * week2_donated
def week2_remaining : ℝ := week2_donated - week2_given_out
def total_remaining_after_week2 : ℝ := week1_remaining + week2_remaining

def week3_donated : ℝ := 1.25 * week2_donated
def week3_given_out : ℝ := 0.8 * week3_donated
def week3_remaining : ℝ := week3_donated - week3_given_out
def total_remaining_after_week3 : ℝ := total_remaining_after_week2 + week3_remaining

def week4_donated : ℝ := 0.9 * week3_donated
def week4_given_out : ℝ := 0.5 * week4_donated
def week4_remaining : ℝ := week4_donated - week4_given_out
def total_remaining_after_week4 : ℝ := total_remaining_after_week3 + week4_remaining

theorem food_bank_remaining_after_four_weeks : total_remaining_after_week4 = 82.75 := by
  sorry

end food_bank_remaining_after_four_weeks_l259_259704


namespace zombie_count_today_l259_259430

theorem zombie_count_today (Z : ℕ) (h : Z < 50) : 16 * Z = 48 :=
by
  -- Assume Z, h conditions from a)
  -- Proof will go here, for now replaced with sorry
  sorry

end zombie_count_today_l259_259430


namespace count_invertible_mod_11_l259_259527

theorem count_invertible_mod_11 :
  ∃ (n : ℕ), n = 10 ∧ (∀ a, 0 ≤ a ∧ a ≤ 10 → ∃ x, (a * x) % 11 = 1 ↔ gcd a 11 = 1) := 
begin
  sorry,
end

end count_invertible_mod_11_l259_259527


namespace polynomial_product_evaluation_l259_259497

theorem polynomial_product_evaluation :
  let p1 := (2*x^3 - 3*x^2 + 5*x - 1)
  let p2 := (8 - 3*x)
  let product := p1 * p2
  let a := -6
  let b := 25
  let c := -39
  let d := 43
  let e := -8
  (16 * a + 8 * b + 4 * c + 2 * d + e) = 26 :=
by
  sorry

end polynomial_product_evaluation_l259_259497


namespace evaluate_expression_l259_259677

theorem evaluate_expression : -(16 / 4 * 11 - 70 + 5 * 11) = -29 := by
  sorry

end evaluate_expression_l259_259677


namespace smallest_m_plus_n_l259_259567

theorem smallest_m_plus_n (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : 3 * m^3 = 5 * n^5) : m + n = 720 :=
by
  sorry

end smallest_m_plus_n_l259_259567


namespace fred_games_this_year_l259_259343

variable (last_year_games : ℕ)
variable (difference : ℕ)

theorem fred_games_this_year (h1 : last_year_games = 36) (h2 : difference = 11) : 
  last_year_games - difference = 25 := 
by
  sorry

end fred_games_this_year_l259_259343


namespace bill_pays_sales_tax_correct_l259_259299

def take_home_salary : ℝ := 40000
def property_tax : ℝ := 2000
def gross_salary : ℝ := 50000
def income_tax (gs : ℝ) : ℝ := 0.10 * gs
def total_taxes_paid (gs th : ℝ) : ℝ := gs - th
def sales_tax (ttp it pt : ℝ) : ℝ := ttp - it - pt

theorem bill_pays_sales_tax_correct :
  sales_tax
    (total_taxes_paid gross_salary take_home_salary)
    (income_tax gross_salary)
    property_tax = 3000 :=
by sorry

end bill_pays_sales_tax_correct_l259_259299


namespace minimum_candy_kinds_l259_259855

theorem minimum_candy_kinds (n : ℕ) (h_n : n = 91) (even_spacing : ∀ i j : ℕ, i < j → i < n → j < n → (∀ k : ℕ, i < k ∧ k < j → k % 2 = 1)) : 46 ≤ n / 2 :=
by
  rw h_n
  have : 46 ≤ 91 / 2 := nat.le_of_lt (by norm_num)
  exact this

end minimum_candy_kinds_l259_259855


namespace minimum_candy_kinds_l259_259860

theorem minimum_candy_kinds (candy_count : ℕ) (h1 : candy_count = 91)
  (h2 : ∀ (k : ℕ), k ∈ (1 : ℕ) → (λ i j, abs (i - j) % 2 = 0))
  : ∃ (kinds : ℕ), kinds = 46 :=
by
  sorry

end minimum_candy_kinds_l259_259860


namespace prove_jens_suckers_l259_259405
noncomputable def Jen_ate_suckers (Sienna_suckers : ℕ) (Jen_suckers_given_to_Molly : ℕ) : Prop :=
  let Molly_suckers_given_to_Harmony := Jen_suckers_given_to_Molly - 2
  let Harmony_suckers_given_to_Taylor := Molly_suckers_given_to_Harmony + 3
  let Taylor_suckers_given_to_Callie := Harmony_suckers_given_to_Taylor - 1
  Taylor_suckers_given_to_Callie = 5 → (Sienna_suckers/2) = Jen_suckers_given_to_Molly * 2

#eval Jen_ate_suckers 44 11 -- Example usage, you can change 44 and 11 accordingly

def jen_ate_11_suckers : Prop :=
  Jen_ate_suckers Sienna_suckers 11

theorem prove_jens_suckers : jen_ate_11_suckers :=
  sorry

end prove_jens_suckers_l259_259405


namespace complex_number_quadrant_l259_259971

theorem complex_number_quadrant (a b : ℝ) (h1 : (2 + a * (0+1*I)) / (1 + 1*I) = b + 1*I) (h2: a = 4) (h3: b = 3) : 
  0 < a ∧ 0 < b :=
by
  sorry

end complex_number_quadrant_l259_259971


namespace pages_to_read_tomorrow_l259_259890

-- Define the conditions
def total_pages : ℕ := 100
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5

-- Lean statement for the proof problem
theorem pages_to_read_tomorrow : (total_pages - (pages_yesterday + pages_today) = 35) :=
by
  let yesterday := pages_yesterday
  let today := pages_today
  let read_so_far := yesterday + today
  have read_so_far_eq : yesterday + today = 65 := by sorry
  have total_eq : total_pages - read_so_far = 35 := by sorry
  exact total_eq

end pages_to_read_tomorrow_l259_259890


namespace count_inverses_mod_11_l259_259549

theorem count_inverses_mod_11 :
  {a ∈ finset.range 11 | Int.gcd a 11 = 1}.card = 10 :=
by
  sorry

end count_inverses_mod_11_l259_259549


namespace common_real_solution_unique_y_l259_259415

theorem common_real_solution_unique_y (x y : ℝ) 
  (h1 : x^2 + y^2 = 16) 
  (h2 : x^2 - 3 * y + 12 = 0) : 
  y = 4 :=
by
  sorry

end common_real_solution_unique_y_l259_259415


namespace expression_value_l259_259452

def a : ℝ := 0.96
def b : ℝ := 0.1

theorem expression_value : (a^3 - (b^3 / a^2) + 0.096 + b^2) = 0.989651 :=
by
  sorry

end expression_value_l259_259452


namespace correct_statement_3_l259_259357

-- Definitions
def acute_angles (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def angles_less_than_90 (θ : ℝ) : Prop := θ < 90
def angles_in_first_quadrant (θ : ℝ) : Prop := ∃ k : ℤ, k * 360 < θ ∧ θ < k * 360 + 90

-- Sets
def M := {θ | acute_angles θ}
def N := {θ | angles_less_than_90 θ}
def P := {θ | angles_in_first_quadrant θ}

-- Proof statement
theorem correct_statement_3 : M ⊆ P := sorry

end correct_statement_3_l259_259357


namespace pure_imaginary_m_value_l259_259344

theorem pure_imaginary_m_value (m : ℝ) (h₁ : m ^ 2 + m - 2 = 0) (h₂ : m ^ 2 - 1 ≠ 0) : m = -2 := by
  sorry

end pure_imaginary_m_value_l259_259344


namespace single_reduction_equivalent_l259_259626

theorem single_reduction_equivalent (P : ℝ) (h1 : P > 0) :
  let final_price := 0.75 * P - 0.7 * (0.75 * P)
  let single_reduction := (P - final_price) / P
  single_reduction * 100 = 77.5 := 
by
  sorry

end single_reduction_equivalent_l259_259626


namespace peter_remaining_money_l259_259722

theorem peter_remaining_money (initial_money : ℕ) 
                             (potato_cost_per_kilo : ℕ) (potato_kilos : ℕ)
                             (tomato_cost_per_kilo : ℕ) (tomato_kilos : ℕ)
                             (cucumber_cost_per_kilo : ℕ) (cucumber_kilos : ℕ)
                             (banana_cost_per_kilo : ℕ) (banana_kilos : ℕ) :
  initial_money = 500 →
  potato_cost_per_kilo = 2 → potato_kilos = 6 →
  tomato_cost_per_kilo = 3 → tomato_kilos = 9 →
  cucumber_cost_per_kilo = 4 → cucumber_kilos = 5 →
  banana_cost_per_kilo = 5 → banana_kilos = 3 →
  initial_money - (potato_cost_per_kilo * potato_kilos + 
                   tomato_cost_per_kilo * tomato_kilos +
                   cucumber_cost_per_kilo * cucumber_kilos +
                   banana_cost_per_kilo * banana_kilos) = 426 := by
  sorry

end peter_remaining_money_l259_259722


namespace average_of_three_numbers_is_78_l259_259597

theorem average_of_three_numbers_is_78 (x y z : ℕ) (h1 : z = 2 * y) (h2 : y = 4 * x) (h3 : x = 18) :
  (x + y + z) / 3 = 78 :=
by sorry

end average_of_three_numbers_is_78_l259_259597


namespace profit_with_discount_l259_259961

theorem profit_with_discount (CP SP_with_discount SP_no_discount : ℝ) (discount profit_no_discount : ℝ) (H1 : discount = 0.1) (H2 : profit_no_discount = 0.3889) (H3 : SP_no_discount = CP * (1 + profit_no_discount)) (H4 : SP_with_discount = SP_no_discount * (1 - discount)) : (SP_with_discount - CP) / CP * 100 = 25 :=
by
  -- The proof will be filled here
  sorry

end profit_with_discount_l259_259961


namespace textopolis_word_count_l259_259416

theorem textopolis_word_count :
  let alphabet_size := 26
  let total_one_letter := 2 -- only "A" and "B"
  let total_two_letter := alphabet_size^2
  let excl_two_letter := (alphabet_size - 2)^2
  let total_three_letter := alphabet_size^3
  let excl_three_letter := (alphabet_size - 2)^3
  let total_four_letter := alphabet_size^4
  let excl_four_letter := (alphabet_size - 2)^4
  let valid_two_letter := total_two_letter - excl_two_letter
  let valid_three_letter := total_three_letter - excl_three_letter
  let valid_four_letter := total_four_letter - excl_four_letter
  2 + valid_two_letter + valid_three_letter + valid_four_letter = 129054 := by
  -- To be proved
  sorry

end textopolis_word_count_l259_259416


namespace arithmetic_sequence_75th_term_l259_259097

theorem arithmetic_sequence_75th_term (a1 d : ℤ) (n : ℤ) (h1 : a1 = 3) (h2 : d = 5) (h3 : n = 75) :
  a1 + (n - 1) * d = 373 :=
by
  rw [h1, h2, h3]
  -- Here, we arrive at the explicitly stated elements and evaluate:
  -- 3 + (75 - 1) * 5 = 373
  sorry

end arithmetic_sequence_75th_term_l259_259097


namespace greatest_possible_x_l259_259621

theorem greatest_possible_x : ∃ (x : ℕ), (x^2 + 5 < 30) ∧ ∀ (y : ℕ), (y^2 + 5 < 30) → y ≤ x :=
by
  sorry

end greatest_possible_x_l259_259621


namespace quadratic_discriminant_l259_259239

-- Define the quadratic equation coefficients
def a : ℤ := 5
def b : ℤ := -11
def c : ℤ := 2

-- Define the discriminant of the quadratic equation
def discriminant (a b c : ℤ) : ℤ := b^2 - 4 * a * c

-- assert the discriminant for given coefficients
theorem quadratic_discriminant : discriminant a b c = 81 :=
by
  sorry

end quadratic_discriminant_l259_259239


namespace rank_best_buy_LMS_l259_259278

theorem rank_best_buy_LMS (c_S q_S : ℝ) :
  let c_M := 1.75 * c_S
  let q_M := 1.1 * q_S
  let c_L := 1.25 * c_M
  let q_L := 1.5 * q_M
  (c_S / q_S) > (c_M / q_M) ∧ (c_M / q_M) > (c_L / q_L) :=
by
  sorry

end rank_best_buy_LMS_l259_259278


namespace compose_frac_prod_eq_one_l259_259478

open Finset

def irreducible_fraction (n d : ℕ) := gcd n d = 1

theorem compose_frac_prod_eq_one :
  ∃ (a b c d e f : ℕ),
  {a, b, c, d, e, f} ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
   d ≠ e ∧ d ≠ f ∧ 
   e ≠ f) ∧
  irreducible_fraction a b ∧
  irreducible_fraction c d ∧
  irreducible_fraction e f ∧
  (a : ℚ) / b * (c : ℚ) / d * (e : ℚ) / f = 1 :=
begin
  sorry
end

end compose_frac_prod_eq_one_l259_259478


namespace max_of_four_expressions_l259_259286

theorem max_of_four_expressions :
  996 * 996 > 995 * 997 ∧ 996 * 996 > 994 * 998 ∧ 996 * 996 > 993 * 999 :=
by
  sorry

end max_of_four_expressions_l259_259286


namespace magnitude_of_Z_l259_259353

-- Define the complex number Z
def Z : ℂ := 3 - 4 * Complex.I

-- Define the theorem to prove the magnitude of Z
theorem magnitude_of_Z : Complex.abs Z = 5 := by
  sorry

end magnitude_of_Z_l259_259353


namespace find_female_employees_l259_259450

-- Definitions from conditions
def total_employees (E : ℕ) := True
def female_employees (F : ℕ) := True
def male_employees (M : ℕ) := True
def female_managers (F_mgrs : ℕ) := F_mgrs = 280
def fraction_of_managers : ℚ := 2 / 5
def fraction_of_male_managers : ℚ := 2 / 5

-- Statements as conditions in Lean
def managers_total (E M : ℕ) := (fraction_of_managers * E : ℚ) = (fraction_of_male_managers * M : ℚ) + 280
def employees_total (E F M : ℕ) := E = F + M

-- The proof target
theorem find_female_employees (E F M : ℕ) (F_mgrs : ℕ)
    (h1 : female_managers F_mgrs)
    (h2 : managers_total E M)
    (h3 : employees_total E F M) : F = 700 := by
  sorry

end find_female_employees_l259_259450


namespace count_inverses_mod_11_l259_259531

def has_inverse_mod (a n : ℕ) : Prop :=
  ∃ b : ℕ, a * b ≡ 1 [MOD n]

theorem count_inverses_mod_11 : 
  (Finset.filter (λ a, has_inverse_mod a 11) (Finset.range 11)).card = 10 :=
by
  sorry

end count_inverses_mod_11_l259_259531


namespace simplify_expression_l259_259735

theorem simplify_expression :
  8 * (18 / 5) * (-40 / 27) = - (128 / 3) := 
by
  sorry

end simplify_expression_l259_259735


namespace actual_distance_traveled_l259_259178

theorem actual_distance_traveled :
  ∀ (t : ℝ) (d1 d2 : ℝ),
  d1 = 15 * t →
  d2 = 30 * t →
  d2 = d1 + 45 →
  d1 = 45 := by
  intro t d1 d2 h1 h2 h3
  sorry

end actual_distance_traveled_l259_259178


namespace constant_term_in_expansion_l259_259363

theorem constant_term_in_expansion : 
  let n := ∫ x in 0 .. 2, 2 * x 
  (n = 4) →
  (Polynomials.constant_coeff ((X - C (1 / 2) * X⁻¹) ^ n) = 3 / 2) :=
by
  sorry

end constant_term_in_expansion_l259_259363


namespace count_inverses_modulo_11_l259_259554

theorem count_inverses_modulo_11 : (Finset.filter (λ a => Nat.gcd a 11 = 1) (Finset.range 11)).card = 10 :=
  by
  sorry

end count_inverses_modulo_11_l259_259554


namespace sin_2alpha_val_l259_259678

-- Define the conditions and the problem in Lean 4
theorem sin_2alpha_val (α : ℝ) (h1 : π < α ∨ α < 3 * π / 2)
  (h2 : 2 * (Real.tan α) ^ 2 - 7 * Real.tan α + 3 = 0) :
  (π < α ∧ α < 5 * π / 4 → Real.sin (2 * α) = 4 / 5) ∧ 
  (5 * π / 4 < α ∧ α < 3 * π / 2 → Real.sin (2 * α) = 3 / 5) := 
sorry

end sin_2alpha_val_l259_259678


namespace compose_frac_prod_eq_one_l259_259476

open Finset

def irreducible_fraction (n d : ℕ) := gcd n d = 1

theorem compose_frac_prod_eq_one :
  ∃ (a b c d e f : ℕ),
  {a, b, c, d, e, f} ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
   d ≠ e ∧ d ≠ f ∧ 
   e ≠ f) ∧
  irreducible_fraction a b ∧
  irreducible_fraction c d ∧
  irreducible_fraction e f ∧
  (a : ℚ) / b * (c : ℚ) / d * (e : ℚ) / f = 1 :=
begin
  sorry
end

end compose_frac_prod_eq_one_l259_259476


namespace no_integer_solutions_l259_259195

theorem no_integer_solutions (P Q : Polynomial ℤ) (a : ℤ) (hP1 : P.eval a = 0) 
  (hP2 : P.eval (a + 1997) = 0) (hQ : Q.eval 1998 = 2000) : 
  ¬ ∃ x : ℤ, Q.eval (P.eval x) = 1 := 
by
  sorry

end no_integer_solutions_l259_259195


namespace expression_equality_l259_259138

theorem expression_equality :
  (2^1001 + 5^1002)^2 - (2^1001 - 5^1002)^2 = 40 * 10^1001 := 
by
  sorry

end expression_equality_l259_259138


namespace arrangement_of_SUCCESS_l259_259141

theorem arrangement_of_SUCCESS : 
  let total_letters := 7
  let count_S := 3
  let count_C := 2
  let count_U := 1
  let count_E := 1
  (fact total_letters) / (fact count_S * fact count_C * fact count_U * fact count_E) = 420 := 
by
  let total_letters := 7
  let count_S := 3
  let count_C := 2
  let count_U := 1
  let count_E := 1
  exact sorry

end arrangement_of_SUCCESS_l259_259141


namespace escalator_steps_l259_259729

theorem escalator_steps
  (steps_ascending : ℤ)
  (steps_descending : ℤ)
  (ascend_units_time : ℤ)
  (descend_units_time : ℤ)
  (speed_ratio : ℤ)
  (equation : ((steps_ascending : ℚ) / (1 + (ascend_units_time : ℚ))) = ((steps_descending : ℚ) / ((descend_units_time : ℚ) * speed_ratio)) )
  (solution_x : (125 * 0.6 = 75)) : 
  (steps_ascending * (1 + 0.6 : ℚ) = 120) :=
by
  sorry

end escalator_steps_l259_259729


namespace probability_at_least_one_meets_standard_l259_259602

-- Define the probabilities for individuals A, B, and C
def P_A_success: ℝ := 0.8
def P_B_success: ℝ := 0.6
def P_C_success: ℝ := 0.5

-- Define the complement probabilities of individuals failing
def P_A_fail : ℝ := 1 - P_A_success
def P_B_fail : ℝ := 1 - P_B_success
def P_C_fail : ℝ := 1 - P_C_success

-- Define the probability that no one meets the standard
def P_NoOne_success : ℝ := P_A_fail * P_B_fail * P_C_fail

-- The targeted probability that at least one meets the standard
def P_AtLeastOne_success : ℝ := 1 - P_NoOne_success

theorem probability_at_least_one_meets_standard :
  P_AtLeastOne_success = 0.96 :=
by
  suffices : P_A_fail = 0.2 ∧ P_B_fail = 0.4 ∧ P_C_fail = 0.5
  { sorry }
  sorry

end probability_at_least_one_meets_standard_l259_259602


namespace sequence_decreasing_l259_259822

noncomputable def x_n (a b : ℝ) (n : ℕ) : ℝ := 2 ^ n * (b ^ (1 / 2 ^ n) - a ^ (1 / 2 ^ n))

theorem sequence_decreasing (a b : ℝ) (h1 : 1 < a) (h2 : a < b) : ∀ n : ℕ, x_n a b n > x_n a b (n + 1) :=
by
  sorry

end sequence_decreasing_l259_259822


namespace picture_area_l259_259464

theorem picture_area (x y : ℕ) (hx : x > 1) (hy : y > 1) 
  (h : (2 * x + 5) * (y + 4) - x * y = 84) : x * y = 15 :=
by
  sorry

end picture_area_l259_259464


namespace quarters_total_l259_259920

variable (q1 q2 S: Nat)

def original_quarters := 760
def additional_quarters := 418

theorem quarters_total : S = original_quarters + additional_quarters :=
sorry

end quarters_total_l259_259920


namespace sum_of_possible_B_is_zero_l259_259463

theorem sum_of_possible_B_is_zero :
  ∀ B : ℕ, B < 10 → (∃ k : ℤ, 7 * k = 500 + 10 * B + 3) -> B = 0 := sorry

end sum_of_possible_B_is_zero_l259_259463


namespace total_horse_food_l259_259131

theorem total_horse_food (ratio_sh_to_h : ℕ → ℕ → Prop) 
    (sheep : ℕ) 
    (ounce_per_horse : ℕ) 
    (total_ounces_per_day : ℕ) : 
    ratio_sh_to_h 5 7 → sheep = 40 → ounce_per_horse = 230 → total_ounces_per_day = 12880 :=
by
  intros h_ratio h_sheep h_ounce
  sorry

end total_horse_food_l259_259131


namespace parabola_focus_distance_l259_259162

theorem parabola_focus_distance (p : ℝ) (h_pos : p > 0) (A : ℝ × ℝ)
  (h_A_on_parabola : A.2 = 5 ∧ A.1^2 = 2 * p * A.2)
  (h_AF : abs (A.2 - (p / 2)) = 8) : p = 6 :=
by
  sorry

end parabola_focus_distance_l259_259162


namespace count_inverses_mod_11_l259_259551

theorem count_inverses_mod_11 :
  {a ∈ finset.range 11 | Int.gcd a 11 = 1}.card = 10 :=
by
  sorry

end count_inverses_mod_11_l259_259551


namespace find_f_neg2_l259_259063

-- Condition (1): f is an even function on ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Condition (2): f(x) = x^2 + 1 for x > 0
def function_defined_for_positive_x {f : ℝ → ℝ} (h_even : even_function f): Prop :=
  ∀ x : ℝ, x > 0 → f x = x^2 + 1

-- Proof problem: prove that given the conditions, f(-2) = 5
theorem find_f_neg2 (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_pos : function_defined_for_positive_x h_even) : 
  f (-2) = 5 := 
sorry

end find_f_neg2_l259_259063


namespace polynomial_int_coeff_property_l259_259978

open Polynomial

theorem polynomial_int_coeff_property 
  (P : Polynomial ℤ) : 
  (∀ s t : ℝ, P.eval s ∈ ℤ ∧ P.eval t ∈ ℤ → P.eval (s * t) ∈ ℤ) ↔ 
  ∃ (n : ℕ) (k : ℤ), P = X^n + C k ∨ P = -X^n + C k :=
by
  sorry

end polynomial_int_coeff_property_l259_259978


namespace fraction_of_historical_fiction_new_releases_l259_259295

theorem fraction_of_historical_fiction_new_releases
  (total_books : ℕ)
  (historical_fiction_percentage : ℝ := 0.4)
  (historical_fiction_new_releases_percentage : ℝ := 0.4)
  (other_genres_new_releases_percentage : ℝ := 0.7)
  (total_historical_fiction_books := total_books * historical_fiction_percentage)
  (total_other_books := total_books * (1 - historical_fiction_percentage))
  (historical_fiction_new_releases := total_historical_fiction_books * historical_fiction_new_releases_percentage)
  (other_genres_new_releases := total_other_books * other_genres_new_releases_percentage)
  (total_new_releases := historical_fiction_new_releases + other_genres_new_releases) :
  historical_fiction_new_releases / total_new_releases = 8 / 29 := 
by 
  sorry

end fraction_of_historical_fiction_new_releases_l259_259295


namespace circle_equation_l259_259749

theorem circle_equation (x y : ℝ) :
  (∀ (C P : ℝ × ℝ), C = (8, -3) ∧ P = (5, 1) →
    ∃ R : ℝ, (x - 8)^2 + (y + 3)^2 = R^2 ∧ R^2 = 25) :=
sorry

end circle_equation_l259_259749


namespace point_on_line_y_coordinate_l259_259279

variables (m b x : ℝ)

def line_equation := m * x + b

theorem point_on_line_y_coordinate : m = 4 → b = 4 → x = 199 → line_equation m b x = 800 :=
by 
  intros h_m h_b h_x
  unfold line_equation
  rw [h_m, h_b, h_x]
  norm_num
  done

end point_on_line_y_coordinate_l259_259279


namespace problem_from_conditions_l259_259563

theorem problem_from_conditions 
  (x y : ℝ)
  (h1 : 3 * x * (2 * x + y) = 14)
  (h2 : y * (2 * x + y) = 35) :
  (2 * x + y)^2 = 49 := 
by 
  sorry

end problem_from_conditions_l259_259563


namespace floor_sqrt_18_squared_eq_16_l259_259972

theorem floor_sqrt_18_squared_eq_16 : (Int.floor (Real.sqrt 18)) ^ 2 = 16 := 
by 
  sorry

end floor_sqrt_18_squared_eq_16_l259_259972


namespace mass_percentage_Br_HBrO3_l259_259504

theorem mass_percentage_Br_HBrO3 (molar_mass_H : ℝ) (molar_mass_Br : ℝ) (molar_mass_O : ℝ)
  (molar_mass_HBrO3 : ℝ) (mass_percentage_H : ℝ) (mass_percentage_Br : ℝ) :
  molar_mass_H = 1.01 →
  molar_mass_Br = 79.90 →
  molar_mass_O = 16.00 →
  molar_mass_HBrO3 = molar_mass_H + molar_mass_Br + 3 * molar_mass_O →
  mass_percentage_H = 0.78 →
  mass_percentage_Br = (molar_mass_Br / molar_mass_HBrO3) * 100 → 
  mass_percentage_Br = 61.98 :=
sorry

end mass_percentage_Br_HBrO3_l259_259504


namespace sum_bn_2999_l259_259983

def b_n (n : ℕ) : ℕ :=
  if n % 17 = 0 ∧ n % 19 = 0 then 15
  else if n % 19 = 0 ∧ n % 13 = 0 then 18
  else if n % 13 = 0 ∧ n % 17 = 0 then 17
  else 0

theorem sum_bn_2999 : (Finset.range 3000).sum b_n = 572 := by
  sorry

end sum_bn_2999_l259_259983


namespace fraction_expression_l259_259968

theorem fraction_expression :
  (1 / 4 - 1 / 6) / (1 / 3 - 1 / 4) = 1 :=
by
  sorry

end fraction_expression_l259_259968


namespace temperature_difference_l259_259600

def Shanghai_temp : ℤ := 3
def Beijing_temp : ℤ := -5

theorem temperature_difference :
  Shanghai_temp - Beijing_temp = 8 := by
  sorry

end temperature_difference_l259_259600


namespace min_candy_kinds_l259_259847

theorem min_candy_kinds (n : ℕ) (m : ℕ) (h_n : n = 91) 
  (h_even : ∀ i j (h_i : i < j) (h_k : j < m), (i ≠ j) → even (j - i - 1)) : 
  m ≥ 46 :=
sorry

end min_candy_kinds_l259_259847


namespace solve_for_x_l259_259737

theorem solve_for_x : (42 / (7 - 3 / 7) = 147 / 23) :=
by
  sorry

end solve_for_x_l259_259737


namespace age_difference_ratio_l259_259919

theorem age_difference_ratio (R J K : ℕ) 
  (h1 : R = J + 8)
  (h2 : R + 2 = 2 * (J + 2))
  (h3 : (R + 2) * (K + 2) = 192) :
  (R - J) / (R - K) = 2 := by
  sorry

end age_difference_ratio_l259_259919


namespace units_digit_of_fraction_l259_259760

theorem units_digit_of_fraction :
  ((30 * 31 * 32 * 33 * 34) / 400) % 10 = 4 :=
by
  sorry

end units_digit_of_fraction_l259_259760


namespace production_days_l259_259158

theorem production_days (n : ℕ) (P : ℕ) (h1: P = n * 50) 
    (h2: (P + 110) / (n + 1) = 55) : n = 11 :=
by
  sorry

end production_days_l259_259158


namespace expressions_equal_iff_l259_259801

variable (a b c : ℝ)

theorem expressions_equal_iff :
  a^2 + b*c = (a - b)*(a - c) ↔ a = 0 ∨ b + c = 0 :=
by
  sorry

end expressions_equal_iff_l259_259801


namespace largest_angle_of_triangle_l259_259935

theorem largest_angle_of_triangle (x : ℝ) 
  (h1 : 4 * x + 5 * x + 9 * x = 180) 
  (h2 : 4 * x > 40) : 
  9 * x = 90 := 
sorry

end largest_angle_of_triangle_l259_259935


namespace shape_is_cylinder_l259_259156

def positive_constant (c : ℝ) := c > 0

def is_cylinder (r θ z : ℝ) (c : ℝ) : Prop :=
  r = c

theorem shape_is_cylinder (c : ℝ) (r θ z : ℝ) 
  (h_pos : positive_constant c) (h_eq : r = c) :
  is_cylinder r θ z c := by
  sorry

end shape_is_cylinder_l259_259156


namespace handshaking_remainder_l259_259572

noncomputable def num_handshaking_arrangements_modulo (n : ℕ) : ℕ := sorry

theorem handshaking_remainder (N : ℕ) (h : num_handshaking_arrangements_modulo 9 = N) :
  N % 1000 = 16 :=
sorry

end handshaking_remainder_l259_259572


namespace pan_dimensions_l259_259524

theorem pan_dimensions (m n : ℕ) : 
  (∃ m n, m * n = 48 ∧ (m-2) * (n-2) = 2 * (2*m + 2*n - 4) ∧ m > 2 ∧ n > 2) → 
  (m = 4 ∧ n = 12) ∨ (m = 12 ∧ n = 4) ∨ (m = 6 ∧ n = 8) ∨ (m = 8 ∧ n = 6) :=
by
  sorry

end pan_dimensions_l259_259524


namespace exist_irreducible_fractions_prod_one_l259_259493

theorem exist_irreducible_fractions_prod_one (S : List ℚ) :
  (∀ x ∈ S, ∃ (n d : ℤ), n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ x = (n /. d) ∧ Int.gcd n d = 1) ∧
  (∀ i j, i ≠ j → (S.get i).num ≠ (S.get j).num ∧ (S.get i).den ≠ (S.get j).den) →
  S.length = 3 ∧ S.prod = 1 :=
begin
  sorry
end

end exist_irreducible_fractions_prod_one_l259_259493


namespace compare_abc_l259_259033

variables (a b c : ℝ)

-- Given conditions
def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

-- Goal to prove
theorem compare_abc : a > c ∧ c > b := sorry

end compare_abc_l259_259033


namespace combined_weight_loss_l259_259788

theorem combined_weight_loss (a_weekly_loss : ℝ) (a_weeks : ℕ) (x_weekly_loss : ℝ) (x_weeks : ℕ)
  (h1 : a_weekly_loss = 1.5) (h2 : a_weeks = 10) (h3 : x_weekly_loss = 2.5) (h4 : x_weeks = 8) :
  a_weekly_loss * a_weeks + x_weekly_loss * x_weeks = 35 := 
by
  -- We will not provide the proof body; the goal is to ensure the statement compiles.
  sorry

end combined_weight_loss_l259_259788


namespace minimum_candy_kinds_l259_259856

theorem minimum_candy_kinds (n : ℕ) (h_n : n = 91) (even_spacing : ∀ i j : ℕ, i < j → i < n → j < n → (∀ k : ℕ, i < k ∧ k < j → k % 2 = 1)) : 46 ≤ n / 2 :=
by
  rw h_n
  have : 46 ≤ 91 / 2 := nat.le_of_lt (by norm_num)
  exact this

end minimum_candy_kinds_l259_259856


namespace brownies_pieces_l259_259593

theorem brownies_pieces (tray_length tray_width piece_length piece_width : ℕ) 
  (h1 : tray_length = 24) 
  (h2 : tray_width = 16) 
  (h3 : piece_length = 2) 
  (h4 : piece_width = 2) : 
  tray_length * tray_width / (piece_length * piece_width) = 96 :=
by sorry

end brownies_pieces_l259_259593


namespace coefficient_x3y3_term_in_expansion_of_x_plus_y_pow_6_l259_259104

theorem coefficient_x3y3_term_in_expansion_of_x_plus_y_pow_6 : 
  (∑ k in finset.range 7, (nat.choose 6 k) * (x ^ k) * (y ^ (6 - k)))[3] = 20 := 
by sorry

end coefficient_x3y3_term_in_expansion_of_x_plus_y_pow_6_l259_259104


namespace total_horse_food_l259_259132

theorem total_horse_food (ratio_sh_to_h : ℕ → ℕ → Prop) 
    (sheep : ℕ) 
    (ounce_per_horse : ℕ) 
    (total_ounces_per_day : ℕ) : 
    ratio_sh_to_h 5 7 → sheep = 40 → ounce_per_horse = 230 → total_ounces_per_day = 12880 :=
by
  intros h_ratio h_sheep h_ounce
  sorry

end total_horse_food_l259_259132


namespace negative_solution_condition_l259_259329

-- Define the system of equations
def system_of_equations (a b c x y : ℝ) :=
  a * x + b * y = c ∧
  b * x + c * y = a ∧
  c * x + a * y = b

-- State the theorem
theorem negative_solution_condition (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ system_of_equations a b c x y) ↔ a + b + c = 0 :=
by
  sorry

end negative_solution_condition_l259_259329


namespace jill_trips_to_fill_tank_l259_259882

   -- Defining the conditions
   def tank_capacity : ℕ := 600
   def bucket_capacity : ℕ := 5
   def jack_buckets_per_trip : ℕ := 2
   def jill_buckets_per_trip : ℕ := 1
   def jack_trip_rate : ℕ := 3
   def jill_trip_rate : ℕ := 2

   -- Calculate the amount of water Jack and Jill carry per trip
   def jack_gallons_per_trip : ℕ := jack_buckets_per_trip * bucket_capacity
   def jill_gallons_per_trip : ℕ := jill_buckets_per_trip * bucket_capacity

   -- Grouping the trips in the time it takes for Jill to complete her trips
   def total_gallons_per_group : ℕ := (jack_trip_rate * jack_gallons_per_trip) + (jill_trip_rate * jill_gallons_per_trip)

   -- Calculate the number of groups needed to fill the tank
   def groups_needed : ℕ := tank_capacity / total_gallons_per_group

   -- Calculate the total trips Jill makes
   def jill_total_trips : ℕ := groups_needed * jill_trip_rate

   -- The proof statement
   theorem jill_trips_to_fill_tank : jill_total_trips = 30 :=
   by
     -- Skipping the proof
     sorry
   
end jill_trips_to_fill_tank_l259_259882


namespace largest_number_is_89_l259_259342

theorem largest_number_is_89 (a b c d : ℕ) 
  (h1 : a + b + c = 180) 
  (h2 : a + b + d = 197) 
  (h3 : a + c + d = 208) 
  (h4 : b + c + d = 222) : 
  max a (max b (max c d)) = 89 := 
by sorry

end largest_number_is_89_l259_259342


namespace negative_solution_condition_l259_259327

-- Define the system of equations
def system_of_equations (a b c x y : ℝ) :=
  a * x + b * y = c ∧
  b * x + c * y = a ∧
  c * x + a * y = b

-- State the theorem
theorem negative_solution_condition (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ system_of_equations a b c x y) ↔ a + b + c = 0 :=
by
  sorry

end negative_solution_condition_l259_259327


namespace comparison_abc_l259_259051

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem comparison_abc : a > c ∧ c > b :=
by
  sorry

end comparison_abc_l259_259051


namespace uncle_fyodor_sandwiches_count_l259_259946

variable (sandwiches_sharik : ℕ)
variable (sandwiches_matroskin : ℕ := 3 * sandwiches_sharik)
variable (total_sandwiches_eaten : ℕ := sandwiches_sharik + sandwiches_matroskin)
variable (sandwiches_uncle_fyodor : ℕ := 2 * total_sandwiches_eaten)
variable (difference : ℕ := sandwiches_uncle_fyodor - sandwiches_sharik)

theorem uncle_fyodor_sandwiches_count :
  (difference = 21) → sandwiches_uncle_fyodor = 24 := by
  intro h
  sorry

end uncle_fyodor_sandwiches_count_l259_259946


namespace geometric_seq_a5_l259_259223

theorem geometric_seq_a5 : ∃ (a₁ q : ℝ), 0 < q ∧ a₁ + 2 * a₁ * q = 4 ∧ (a₁ * q^3)^2 = 4 * (a₁ * q^2) * (a₁ * q^6) ∧ (a₅ = a₁ * q^4) := 
  by
    sorry

end geometric_seq_a5_l259_259223


namespace polynomial_integer_condition_l259_259977

theorem polynomial_integer_condition (P : ℝ → ℝ) (hP : ∀ x, is_polynomial_with_integer_coefficients P x) :
  (∀ s t : ℝ, (P s ∈ ℤ) → (P t ∈ ℤ) → (P (s * t) ∈ ℤ)) →
  (∃ (n : ℕ) (k : ℤ), P = λ x, x^n + k ∨ P = λ x, -x^n + k) :=
by
  sorry

end polynomial_integer_condition_l259_259977


namespace first_player_winning_strategy_l259_259428

noncomputable def optimal_first_move : ℕ := 45

-- Prove that with 300 matches initially and following the game rules,
-- taking 45 matches on the first turn leaves the opponent in a losing position.

theorem first_player_winning_strategy (n : ℕ) (h₀ : n = 300) :
    ∃ m : ℕ, (m ≤ n / 2 ∧ n - m = 255) :=
by
  exists optimal_first_move
  sorry

end first_player_winning_strategy_l259_259428


namespace sqrt_fraction_sum_as_common_fraction_l259_259107

theorem sqrt_fraction_sum_as_common_fraction (a b c d : ℚ) (ha : a = 25) (hb : b = 36) (hc : c = 16) (hd : d = 9) :
  Real.sqrt ((a / b) + (c / d)) = Real.sqrt 89 / 6 := by
  sorry

end sqrt_fraction_sum_as_common_fraction_l259_259107


namespace roberta_started_with_8_records_l259_259205

variable (R : ℕ)

def received_records := 12
def bought_records := 30
def total_received_and_bought := received_records + bought_records

theorem roberta_started_with_8_records (h : R + total_received_and_bought = 50) : R = 8 :=
by
  sorry

end roberta_started_with_8_records_l259_259205


namespace distance_traveled_on_second_day_l259_259375

theorem distance_traveled_on_second_day 
  (a₁ : ℝ) 
  (h_sum : a₁ + a₁ / 2 + a₁ / 4 + a₁ / 8 + a₁ / 16 + a₁ / 32 = 189) 
  : a₁ / 2 = 48 :=
by
  sorry

end distance_traveled_on_second_day_l259_259375


namespace negative_solution_iff_sum_zero_l259_259319

theorem negative_solution_iff_sum_zero (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔
  a + b + c = 0 :=
by
  sorry

end negative_solution_iff_sum_zero_l259_259319


namespace annual_earning_difference_l259_259380

def old_hourly_wage := 16
def old_weekly_hours := 25
def new_hourly_wage := 20
def new_weekly_hours := 40
def weeks_per_year := 52

def old_weekly_earnings := old_hourly_wage * old_weekly_hours
def new_weekly_earnings := new_hourly_wage * new_weekly_hours

def old_annual_earnings := old_weekly_earnings * weeks_per_year
def new_annual_earnings := new_weekly_earnings * weeks_per_year

theorem annual_earning_difference:
  new_annual_earnings - old_annual_earnings = 20800 := by
  sorry

end annual_earning_difference_l259_259380


namespace fraction_not_collapsing_l259_259624

variable (total_homes : ℕ)
variable (termite_ridden_fraction collapsing_fraction : ℚ)
variable (h : termite_ridden_fraction = 1 / 3)
variable (c : collapsing_fraction = 7 / 10)

theorem fraction_not_collapsing : 
  (termite_ridden_fraction - (termite_ridden_fraction * collapsing_fraction)) = 1 / 10 := 
by 
  rw [h, c]
  sorry

end fraction_not_collapsing_l259_259624


namespace Jen_ate_11_suckers_l259_259406

/-
Sienna gave Bailey half of her suckers.
Jen ate half and gave the rest to Molly.
Molly ate 2 and gave the rest to Harmony.
Harmony kept 3 and passed the remainder to Taylor.
Taylor ate one and gave the last 5 to Callie.
How many suckers did Jen eat?
-/

noncomputable def total_suckers_given_to_Callie := 5
noncomputable def total_suckers_Taylor_had := total_suckers_given_to_Callie + 1
noncomputable def total_suckers_Harmony_had := total_suckers_Taylor_had + 3
noncomputable def total_suckers_Molly_had := total_suckers_Harmony_had + 2
noncomputable def total_suckers_Jen_had := total_suckers_Molly_had * 2
noncomputable def suckers_Jen_ate := total_suckers_Jen_had - total_suckers_Molly_had

theorem Jen_ate_11_suckers : suckers_Jen_ate = 11 :=
by {
  unfold total_suckers_given_to_Callie total_suckers_Taylor_had total_suckers_Harmony_had total_suckers_Molly_had total_suckers_Jen_had suckers_Jen_ate,
  sorry
}

end Jen_ate_11_suckers_l259_259406


namespace three_irreducible_fractions_prod_eq_one_l259_259481

-- Define the set of numbers available for use
def available_numbers : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a structure for an irreducible fraction
structure irreducible_fraction :=
(num : ℕ)
(denom : ℕ)
(h_coprime : Nat.gcd num denom = 1)
(h_in_set : num ∈ available_numbers ∧ denom ∈ available_numbers)

-- Definition of the main proof problem
theorem three_irreducible_fractions_prod_eq_one :
  ∃ (f1 f2 f3 : irreducible_fraction), 
    f1.num * f2.num * f3.num = f1.denom * f2.denom * f3.denom ∧ 
    f1.num ≠ f2.num ∧ f1.num ≠ f3.num ∧ f2.num ≠ f3.num ∧ 
    f1.denom ≠ f2.denom ∧ f1.denom ≠ f3.denom ∧ f2.denom ≠ f3.denom := 
by
  sorry

end three_irreducible_fractions_prod_eq_one_l259_259481


namespace parabola_directrix_l259_259989

variable {F P1 P2 : Point}

def is_on_parabola (F : Point) (P1 : Point) : Prop := 
  -- Definition of a point being on the parabola with focus F and a directrix (to be determined).
  sorry

def construct_circles (F P1 P2 : Point) : Circle × Circle :=
  -- Construct circles centered at P1 and P2 passing through F.
  sorry

def common_external_tangents (k1 k2 : Circle) : Nat :=
  -- Function to find the number of common external tangents between two circles.
  sorry

theorem parabola_directrix (F P1 P2 : Point) (h1 : is_on_parabola F P1) (h2 : is_on_parabola F P2) :
  ∃ (k1 k2 : Circle), construct_circles F P1 P2 = (k1, k2) → 
    common_external_tangents k1 k2 = 2 :=
by
  -- Proof that under these conditions, there are exactly 2 common external tangents.
  sorry

end parabola_directrix_l259_259989


namespace combined_weight_loss_l259_259790

theorem combined_weight_loss :
  let aleesia_loss_per_week := 1.5
  let aleesia_weeks := 10
  let alexei_loss_per_week := 2.5
  let alexei_weeks := 8
  (aleesia_loss_per_week * aleesia_weeks) + (alexei_loss_per_week * alexei_weeks) = 35 := by
sorry

end combined_weight_loss_l259_259790


namespace paint_cost_l259_259228

theorem paint_cost {width height : ℕ} (price_per_quart coverage_area : ℕ) (total_cost : ℕ) :
  width = 5 → height = 4 → price_per_quart = 2 → coverage_area = 4 → total_cost = 20 :=
by
  intros h1 h2 h3 h4
  have area_one_side : ℕ := width * height
  have total_area : ℕ := 2 * area_one_side
  have quarts_needed : ℕ := total_area / coverage_area
  have cost : ℕ := quarts_needed * price_per_quart
  sorry

end paint_cost_l259_259228


namespace students_taking_neither_580_l259_259457

noncomputable def numberOfStudentsTakingNeither (total students_m students_a students_d students_ma students_md students_ad students_mad : ℕ) : ℕ :=
  let total_taking_at_least_one := (students_m + students_a + students_d) 
                                - (students_ma + students_md + students_ad) 
                                + students_mad
  total - total_taking_at_least_one

theorem students_taking_neither_580 :
  let total := 800
  let students_m := 140
  let students_a := 90
  let students_d := 75
  let students_ma := 50
  let students_md := 30
  let students_ad := 25
  let students_mad := 20
  numberOfStudentsTakingNeither total students_m students_a students_d students_ma students_md students_ad students_mad = 580 :=
by
  sorry

end students_taking_neither_580_l259_259457


namespace consecutive_numbers_product_l259_259447

theorem consecutive_numbers_product (a b c d : ℤ) 
  (h1 : b = a + 1) 
  (h2 : c = a + 2) 
  (h3 : d = a + 3) 
  (h4 : a + d = 109) : 
  b * c = 2970 := by
  sorry

end consecutive_numbers_product_l259_259447


namespace min_value_gx2_plus_fx_l259_259359

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
noncomputable def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c

theorem min_value_gx2_plus_fx (a b c : ℝ) (h_a : a ≠ 0)
    (h_min_fx_gx : ∀ x : ℝ, (f a b x)^2 + g a c x ≥ -6) :
    ∃ x : ℝ, (g a c x)^2 + f a b x = 11/2 := sorry

end min_value_gx2_plus_fx_l259_259359


namespace find_angle_A_l259_259581

theorem find_angle_A (A B C a b c : ℝ) 
  (h_triangle: a = Real.sqrt 2)
  (h_sides: b = 2 * Real.sin B + Real.cos B)
  (h_b_eq: b = Real.sqrt 2)
  (h_a_lt_b: a < b)
  : A = Real.pi / 6 := sorry

end find_angle_A_l259_259581


namespace house_to_market_distance_l259_259651

-- Definitions of the conditions
def distance_to_school : ℕ := 50
def distance_back_home : ℕ := 50
def total_distance_walked : ℕ := 140

-- Statement of the problem
theorem house_to_market_distance :
  distance_to_market = total_distance_walked - (distance_to_school + distance_back_home) :=
by
  sorry

end house_to_market_distance_l259_259651


namespace john_sells_20_woodburnings_l259_259189

variable (x : ℕ)

theorem john_sells_20_woodburnings (price_per_woodburning cost profit : ℤ) 
  (h1 : price_per_woodburning = 15) (h2 : cost = 100) (h3 : profit = 200) :
  (profit = price_per_woodburning * x - cost) → 
  x = 20 :=
by
  intros h_profit
  rw [h1, h2, h3] at h_profit
  linarith

end john_sells_20_woodburnings_l259_259189


namespace find_fractions_l259_259491

open Function

-- Define the set and the condition that all numbers must be used precisely once
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define what it means for fractions to multiply to 1 within the set
def fractions_mul_to_one (a b c d e f : ℕ) : Prop :=
  (a * c * e) = (b * d * f)

-- Define irreducibility condition for a fraction a/b
def irreducible_fraction (a b : ℕ) := 
  Nat.gcd a b = 1

-- Final main problem statement
theorem find_fractions :
  ∃ (a b c d e f : ℕ) (h₁ : a ∈ S) (h₂ : b ∈ S) (h₃ : c ∈ S) (h₄ : d ∈ S) (h₅ : e ∈ S) (h₆ : f ∈ S),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
  d ≠ e ∧ d ≠ f ∧ 
  e ≠ f ∧
  irreducible_fraction a b ∧ irreducible_fraction c d ∧ irreducible_fraction e f ∧
  fractions_mul_to_one a b c d e f := 
sorry

end find_fractions_l259_259491


namespace heptagonal_prism_faces_and_vertices_l259_259122

structure HeptagonalPrism where
  heptagonal_basis : ℕ
  lateral_faces : ℕ
  basis_vertices : ℕ

noncomputable def faces (h : HeptagonalPrism) : ℕ :=
  2 + h.lateral_faces

noncomputable def vertices (h : HeptagonalPrism) : ℕ :=
  h.basis_vertices * 2

theorem heptagonal_prism_faces_and_vertices : ∀ h : HeptagonalPrism,
  (h.heptagonal_basis = 2) →
  (h.lateral_faces = 7) →
  (h.basis_vertices = 7) →
  faces h = 9 ∧ vertices h = 14 :=
by
  intros
  simp [faces, vertices]
  sorry

end heptagonal_prism_faces_and_vertices_l259_259122


namespace jackson_investment_ratio_l259_259884

theorem jackson_investment_ratio:
  ∀ (B J: ℝ), B = 0.20 * 500 → J = B + 1900 → (J / 500) = 4 :=
by
  intros B J hB hJ
  sorry

end jackson_investment_ratio_l259_259884


namespace interest_rate_unique_l259_259401

theorem interest_rate_unique (P r : ℝ) (h₁ : P * (1 + 3 * r) = 300) (h₂ : P * (1 + 8 * r) = 400) : r = 1 / 12 :=
by {
  sorry
}

end interest_rate_unique_l259_259401


namespace gcd_2pow_2025_minus_1_2pow_2016_minus_1_l259_259620

theorem gcd_2pow_2025_minus_1_2pow_2016_minus_1 :
  Nat.gcd (2^2025 - 1) (2^2016 - 1) = 511 :=
by sorry

end gcd_2pow_2025_minus_1_2pow_2016_minus_1_l259_259620


namespace linear_function_of_additivity_l259_259734

theorem linear_function_of_additivity (f : ℝ → ℝ) 
  (h_add : ∀ x y : ℝ, f (x + y) = f x + f y) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end linear_function_of_additivity_l259_259734


namespace greatest_N_consecutive_sum_50_l259_259438

theorem greatest_N_consecutive_sum_50 :
  ∃ N a : ℤ, (N > 0) ∧ (N * (2 * a + N - 1) = 100) ∧ (N = 100) :=
by
  sorry

end greatest_N_consecutive_sum_50_l259_259438


namespace hyperbola_real_axis_length_proof_l259_259090

noncomputable def hyperbola_real_axis_length : ℝ :=
  let a := sqrt 15 / 2 in
  2 * a

theorem hyperbola_real_axis_length_proof :
  ∃ C O P T (C_center : point) (O_center : point) (C_asymptote : line)
  (C_equation : equation) (O_equation : equation) (T_equation : equation), 
  C_center = (0, 0) ∧
  O_center = (0, 0) ∧
  O_equation = λ (x y: ℝ), x^2 + y^2 = 5 ∧
  P = (2, -1) ∧
  T_equation = λ (x y: ℝ), 2 * x - y = 5 ∧
  parallel T_equation C_asymptote ∧
  length_real_axis C_equation = sqrt 15 := 
  sorry

end hyperbola_real_axis_length_proof_l259_259090


namespace alcohol_quantity_in_mixture_l259_259419

theorem alcohol_quantity_in_mixture : 
  ∃ (A W : ℕ), (A = 8) ∧ (A * 3 = 4 * W) ∧ (A * 5 = 4 * (W + 4)) :=
by
  sorry -- This is a placeholder; the proof itself is not required.

end alcohol_quantity_in_mixture_l259_259419


namespace new_rope_length_l259_259640

-- Define the given constants and conditions
def rope_length_initial : ℝ := 12
def additional_area : ℝ := 1511.7142857142858
noncomputable def pi_approx : ℝ := Real.pi

-- Define the proof statement
theorem new_rope_length :
  let r2 := Real.sqrt ((additional_area / pi_approx) + rope_length_initial ^ 2)
  r2 = 25 :=
by
  -- Placeholder for the proof
  sorry

end new_rope_length_l259_259640


namespace xy_value_l259_259174

theorem xy_value (x y : ℝ) (h : |x - 5| + |y + 3| = 0) : x * y = -15 := by
  sorry

end xy_value_l259_259174


namespace sequence_equality_l259_259273

theorem sequence_equality (a : ℕ → ℤ) (h : ∀ n, a (n + 2) ^ 2 + a (n + 1) * a n ≤ a (n + 2) * (a (n + 1) + a n)) :
  ∃ N : ℕ, ∀ n ≥ N, a (n + 2) = a n :=
by sorry

end sequence_equality_l259_259273


namespace sequence_general_term_l259_259613

theorem sequence_general_term (a : ℕ → ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, a (n + 1) = (n + 1) * a n - n) :
  ∀ n, a n = n! + 1 :=
by
  sorry

end sequence_general_term_l259_259613


namespace cannot_determine_right_triangle_l259_259126

theorem cannot_determine_right_triangle (A B C : Type) (angle_A angle_B angle_C : A) (a b c : B) 
  (h1 : angle_A = angle_B + angle_C)
  (h2 : a / b = 5 / 12 ∧ b / c = 12 / 13)
  (h3 : a ^ 2 = (b + c) * (b - c)):
  ¬ (angle_A / angle_B = 3 / 4 ∧ angle_B / angle_C = 4 / 5) :=
sorry

end cannot_determine_right_triangle_l259_259126


namespace compare_abc_l259_259037

def a : ℝ := 2 * log 1.01
def b : ℝ := log 1.02
def c : ℝ := real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l259_259037


namespace f_at_47_l259_259988

noncomputable def f : ℝ → ℝ := sorry

axiom f_functional_equation : ∀ x : ℝ, f (x - 1) + f (x + 1) = 0
axiom f_interval_definition : ∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x = Real.log (x + 1) / Real.log 2

theorem f_at_47 : f 47 = -1 := by
  sorry

end f_at_47_l259_259988


namespace alpha_eq_one_l259_259812

-- Definitions based on conditions from the problem statement.
variable (α : ℝ) 
variable (f : ℝ → ℝ)

-- The conditions defined as hypotheses
axiom functional_eq (x y : ℝ) : f (α * (x + y)) = f x + f y
axiom non_constant : ∃ x y : ℝ, f x ≠ 0

-- The statement to prove
theorem alpha_eq_one : (∃ f : ℝ → ℝ, (∀ x y : ℝ, f (α * (x + y)) = f x + f y) ∧ (∃ x y : ℝ, f x ≠ f y)) → α = 1 :=
by
  sorry

end alpha_eq_one_l259_259812


namespace cindy_gives_3_envelopes_per_friend_l259_259662

theorem cindy_gives_3_envelopes_per_friend
  (initial_envelopes : ℕ) 
  (remaining_envelopes : ℕ)
  (friends : ℕ)
  (envelopes_per_friend : ℕ) 
  (h1 : initial_envelopes = 37) 
  (h2 : remaining_envelopes = 22)
  (h3 : friends = 5) 
  (h4 : initial_envelopes - remaining_envelopes = envelopes_per_friend * friends) :
  envelopes_per_friend = 3 :=
by
  sorry

end cindy_gives_3_envelopes_per_friend_l259_259662


namespace integer_solutions_system_ineq_l259_259410

theorem integer_solutions_system_ineq (x : ℤ) :
  (3 * x + 6 > x + 8 ∧ (x : ℚ) / 4 ≥ (x - 1) / 3) ↔ (x = 2 ∨ x = 3 ∨ x = 4) :=
by
  sorry

end integer_solutions_system_ineq_l259_259410


namespace last_person_teeth_removed_l259_259683

-- Define the initial conditions
def total_teeth : ℕ := 32
def total_removed : ℕ := 40
def first_person_removed : ℕ := total_teeth * 1 / 4
def second_person_removed : ℕ := total_teeth * 3 / 8
def third_person_removed : ℕ := total_teeth * 1 / 2

-- Express the problem in Lean
theorem last_person_teeth_removed : 
  first_person_removed + second_person_removed + third_person_removed + last_person_removed = total_removed →
  last_person_removed = 4 := 
by
  sorry

end last_person_teeth_removed_l259_259683


namespace pieces_of_gum_per_cousin_l259_259191

theorem pieces_of_gum_per_cousin (total_gum : ℕ) (num_cousins : ℕ) (h1 : total_gum = 20) (h2 : num_cousins = 4) : total_gum / num_cousins = 5 := by
  sorry

end pieces_of_gum_per_cousin_l259_259191


namespace parabola_tangent_midpoint_l259_259267

theorem parabola_tangent_midpoint (p : ℝ) (h : p > 0) :
    (∃ M : ℝ × ℝ, M = (2, -2*p)) ∧ 
    (∃ A B : ℝ × ℝ, A ≠ B ∧ 
                      (∃ yA yB : ℝ, yA = (A.1^2)/(2*p) ∧ yB = (B.1^2)/(2*p)) ∧ 
                      (0.5 * (A.2 + B.2) = 6)) → p = 1 := by sorry

end parabola_tangent_midpoint_l259_259267


namespace find_n_l259_259679

theorem find_n (n : ℕ) (h : 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n = 3^2012) : n = 1005 :=
sorry

end find_n_l259_259679


namespace trucks_more_than_buses_l259_259429

theorem trucks_more_than_buses (b t : ℕ) (h₁ : b = 9) (h₂ : t = 17) : t - b = 8 :=
by
  sorry

end trucks_more_than_buses_l259_259429


namespace probability_square_tile_die_l259_259976

/-
  Define the event of getting a product that is a perfect square
  when choosing a tile from 1 to 15 and a number from a die (1 to 10).
-/

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def possibleTiles := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
def dieFaces := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def favorableOutcomes :=
  { (t, d) ∈ possibleTiles × dieFaces | isPerfectSquare (t * d)}

def totalOutcomes := possibleTiles × dieFaces

noncomputable def probabilityPerfectSquare : ℚ :=
  (favorableOutcomes.toFinset.card : ℚ) / (totalOutcomes.toFinset.card : ℚ)

theorem probability_square_tile_die : probabilityPerfectSquare = 19 / 150 :=
by
  sorry

end probability_square_tile_die_l259_259976


namespace convert_seven_cubic_yards_l259_259838

-- Define the conversion factor from yards to feet
def yardToFeet : ℝ := 3
-- Define the conversion factor from cubic yards to cubic feet
def cubicYardToCubicFeet : ℝ := yardToFeet ^ 3
-- Define the conversion function from cubic yards to cubic feet
noncomputable def convertVolume (volumeInCubicYards : ℝ) : ℝ :=
  volumeInCubicYards * cubicYardToCubicFeet

-- Statement to prove: 7 cubic yards is equivalent to 189 cubic feet
theorem convert_seven_cubic_yards : convertVolume 7 = 189 := by
  sorry

end convert_seven_cubic_yards_l259_259838


namespace region_Z_probability_l259_259123

variable (P : Type) [Field P]
variable (P_X P_Y P_W P_Z : P)

theorem region_Z_probability :
  P_X = 1 / 3 → P_Y = 1 / 4 → P_W = 1 / 6 → P_X + P_Y + P_Z + P_W = 1 → P_Z = 1 / 4 := by
  sorry

end region_Z_probability_l259_259123


namespace percent_of_l259_259629

theorem percent_of (Part Whole : ℕ) (Percent : ℕ) (hPart : Part = 120) (hWhole : Whole = 40) :
  Percent = (Part * 100) / Whole → Percent = 300 :=
by
  sorry

end percent_of_l259_259629


namespace model_lighthouse_height_l259_259595

theorem model_lighthouse_height (h_actual : ℝ) (V_actual : ℝ) (V_model : ℝ) (h_actual_val : h_actual = 60) (V_actual_val : V_actual = 150000) (V_model_val : V_model = 0.15) :
  (h_actual * (V_model / V_actual)^(1/3)) = 0.6 :=
by
  rw [h_actual_val, V_actual_val, V_model_val]
  sorry

end model_lighthouse_height_l259_259595


namespace different_books_read_l259_259229

theorem different_books_read (t_books d_books b_books td_same all_same : ℕ)
  (h_t_books : t_books = 23)
  (h_d_books : d_books = 12)
  (h_b_books : b_books = 17)
  (h_td_same : td_same = 3)
  (h_all_same : all_same = 1) : 
  t_books + d_books + b_books - (td_same + all_same) = 48 := 
by
  sorry

end different_books_read_l259_259229


namespace part1_part2_l259_259026

-- Lean 4 statement for proving A == 2B
theorem part1 (a b c : ℝ) (A B C : ℝ) (h₁ : 0 < A) (h₂ : A < π / 2) 
    (h₃ : 0 < B) (h₄ : B < π / 2) (h₅ : 0 < C) (h₆ : C < π / 2) (h₇ : A + B + C = π)
    (h₈ : c = 2 * b * Real.cos A + b) : A = 2 * B :=
by sorry

-- Lean 4 statement for finding range of area of ∆ABD
theorem part2 (B : ℝ) (c : ℝ) (h₁ : 0 < B) (h₂ : B < π / 2) 
    (h₃ : A = 2 * B) (h₄ : c = 2) : 
    (Real.tan (π / 6) < (1 / 2) * c * (1 / Real.cos B) * Real.sin B) ∧ 
    ((1 / 2) * c * (1 / Real.cos B) * Real.sin B < 1) :=
by sorry

end part1_part2_l259_259026


namespace find_p_q_l259_259590

variable (p q : ℝ)
def f (x : ℝ) : ℝ := x^2 + p * x + q

theorem find_p_q:
  (p, q) = (-6, 7) →
  ∀ x : ℝ, (1 ≤ x ∧ x ≤ 5) → |f p q x| ≤ 2 :=
by
  sorry

end find_p_q_l259_259590


namespace surface_area_difference_l259_259266

theorem surface_area_difference
  (larger_cube_volume : ℝ)
  (num_smaller_cubes : ℝ)
  (smaller_cube_volume : ℝ)
  (h1 : larger_cube_volume = 125)
  (h2 : num_smaller_cubes = 125)
  (h3 : smaller_cube_volume = 1) :
  (6 * (smaller_cube_volume)^(2/3) * num_smaller_cubes) - (6 * (larger_cube_volume)^(2/3)) = 600 :=
by {
  sorry
}

end surface_area_difference_l259_259266


namespace gcd_and_lcm_of_18_and_24_l259_259809

-- Definitions of gcd and lcm for the problem's context
def my_gcd (a b : ℕ) : ℕ := a.gcd b
def my_lcm (a b : ℕ) : ℕ := a.lcm b

-- Constants given in the problem
def a := 18
def b := 24

-- Proof problem statement
theorem gcd_and_lcm_of_18_and_24 : my_gcd a b = 6 ∧ my_lcm a b = 72 := by
  sorry

end gcd_and_lcm_of_18_and_24_l259_259809


namespace chess_program_ratio_l259_259913

theorem chess_program_ratio {total_students chess_program_absent : ℕ}
  (h_total : total_students = 24)
  (h_absent : chess_program_absent = 4)
  (h_half : chess_program_absent * 2 = chess_program_absent + chess_program_absent) :
  (chess_program_absent * 2 : ℚ) / total_students = 1 / 3 :=
by
  sorry

end chess_program_ratio_l259_259913


namespace students_played_both_l259_259571

theorem students_played_both (C B X total : ℕ) (hC : C = 500) (hB : B = 600) (hTotal : total = 880) (hInclusionExclusion : C + B - X = total) : X = 220 :=
by
  rw [hC, hB, hTotal] at hInclusionExclusion
  sorry

end students_played_both_l259_259571


namespace example_problem_l259_259148

theorem example_problem (a b : ℕ) : a = 1 → a * (a + b) + 1 ∣ (a + b) * (b + 1) - 1 :=
by
  sorry

end example_problem_l259_259148


namespace exponent_multiplication_l259_259562

theorem exponent_multiplication (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (a b : ℤ) (h3 : 3^m = a) (h4 : 3^n = b) : 3^(m + n) = a * b :=
by
  sorry

end exponent_multiplication_l259_259562


namespace length_of_escalator_l259_259287

-- Define the conditions
def escalator_speed : ℝ := 15 -- ft/sec
def person_speed : ℝ := 5 -- ft/sec
def time_taken : ℝ := 10 -- sec

-- Define the length of the escalator
def escalator_length (escalator_speed : ℝ) (person_speed : ℝ) (time : ℝ) : ℝ := 
  (escalator_speed + person_speed) * time

-- Theorem to prove
theorem length_of_escalator : escalator_length escalator_speed person_speed time_taken = 200 := by
  sorry

end length_of_escalator_l259_259287


namespace coeff_x2_in_expansion_l259_259876

theorem coeff_x2_in_expansion : 
  (2 : ℚ) - (1 / x) * ((1 + x)^6)^(2 : ℤ) = (10 : ℚ) :=
by sorry

end coeff_x2_in_expansion_l259_259876


namespace find_f_one_third_l259_259348

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def satisfies_condition (f : ℝ → ℝ) : Prop :=
∀ x, f (2 - x) = f x

noncomputable def f (x : ℝ) : ℝ := if (2 ≤ x ∧ x ≤ 3) then Real.log (x - 1) / Real.log 2 else 0

theorem find_f_one_third (h_odd : is_odd_function f) (h_condition : satisfies_condition f) :
  f (1 / 3) = Real.log 3 / Real.log 2 - 2 :=
by
  sorry

end find_f_one_third_l259_259348


namespace distance_between_circumcenters_of_parallelogram_l259_259374

noncomputable def parallelogram_distance_circumcenters
  (a b : ℝ) (α : ℝ) : ℝ :=
  sqrt(a^2 + b^2 + 2 * a * b * cos α) * |cot α|

theorem distance_between_circumcenters_of_parallelogram
  (a b : ℝ) (α : ℝ) (AB BC : ℝ) (angle_ABC : ℝ) (h1 : AB = a) (h2 : BC = b) (h3 : angle_ABC = α)
  (parallelogram_ABCD : (AB = a) ∧ (BC = b) ∧ (angle_ABC = α)) :
  parallelogram_distance_circumcenters a b α =
  sqrt(a^2 + b^2 + 2 * a * b * cos α) * |cot α| := by
  sorry

end distance_between_circumcenters_of_parallelogram_l259_259374


namespace problem_statement_l259_259687

/-
Definitions of the given conditions:
- Circle P: (x-1)^2 + y^2 = 8, center C.
- Point M(-1,0).
- Line y = kx + m intersects trajectory at points A and B.
- k_{OA} \cdot k_{OB} = -1/2.
-/

noncomputable def Circle_P : Set (ℝ × ℝ) :=
  { p | (p.1 - 1)^2 + p.2^2 = 8 }

def Point_M : (ℝ × ℝ) := (-1, 0)

def Trajectory_C : Set (ℝ × ℝ) :=
  { p | p.1^2 / 2 + p.2^2 = 1 }

def Line_kx_m (k m : ℝ) : Set (ℝ × ℝ) :=
  { p | p.2 = k * p.1 + m }

def k_OA_OB (k_OA k_OB : ℝ) : Prop :=
  k_OA * k_OB = -1/2

/-
Mathematical equivalence proof problem:
- Prove the trajectory of center C is an ellipse with equation x^2/2 + y^2 = 1.
- Prove that if line y=kx+m intersects with the trajectory, the area of the triangle AOB is a fixed value.
-/

theorem problem_statement (k m : ℝ)
    (h_intersects : ∃ A B : ℝ × ℝ, A ∈ (Trajectory_C ∩ Line_kx_m k m) ∧ B ∈ (Trajectory_C ∩ Line_kx_m k m))
    (k_OA k_OB : ℝ) (h_k_OA_k_OB : k_OA_OB k_OA k_OB) :
  ∃ (C_center_trajectory : Trajectory_C),
  ∃ (area_AOB : ℝ), area_AOB = (3 * Real.sqrt 2) / 2 :=
sorry

end problem_statement_l259_259687


namespace inequality_abc_l259_259717

theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
    a^2 + b^2 + c^2 + 3 ≥ (1 / a) + (1 / b) + (1 / c) + a + b + c :=
sorry

end inequality_abc_l259_259717


namespace candy_problem_l259_259854

-- Definitions of conditions
def total_candies : ℕ := 91
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- The minimum number of kinds of candies function
def min_kinds : ℕ := 46

-- Lean statement for the proof problem
theorem candy_problem : 
  ∀ (kinds : ℕ), 
    (∀ c : ℕ, c < total_candies → c % kinds < 2) → (∃ n : ℕ, kinds = min_kinds) := 
sorry

end candy_problem_l259_259854


namespace geometric_seq_a5_value_l259_259996

theorem geometric_seq_a5_value 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_seq : ∀ n : ℕ, a (n+1) = a n * q)
  (h_pos : ∀ n : ℕ, a n > 0) 
  (h1 : a 1 * a 8 = 4 * a 5)
  (h2 : (a 4 + 2 * a 6) / 2 = 18) 
  : a 5 = 16 := 
sorry

end geometric_seq_a5_value_l259_259996


namespace compare_a_b_c_l259_259039

noncomputable def a : ℝ := 2 * Real.ln 1.01
noncomputable def b : ℝ := Real.ln 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l259_259039


namespace final_pressure_of_helium_l259_259954

theorem final_pressure_of_helium
  (p v v' : ℝ) (k : ℝ)
  (h1 : p = 4)
  (h2 : v = 3)
  (h3 : v' = 6)
  (h4 : p * v = k)
  (h5 : ∀ p' : ℝ, p' * v' = k → p' = 2) :
  p' = 2 := by
  sorry

end final_pressure_of_helium_l259_259954


namespace first_of_five_consecutive_sums_60_l259_259423

theorem first_of_five_consecutive_sums_60 (n : ℕ) 
  (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 60) : n = 10 :=
by {
  sorry
}

end first_of_five_consecutive_sums_60_l259_259423


namespace largest_integer_condition_l259_259029

theorem largest_integer_condition (m a b : ℤ) 
  (h1 : m < 150) 
  (h2 : m > 50) 
  (h3 : m = 9 * a - 2) 
  (h4 : m = 6 * b - 4) : 
  m = 106 := 
sorry

end largest_integer_condition_l259_259029


namespace find_x_eq_2_l259_259016

theorem find_x_eq_2 : ∀ x : ℝ, 2^10 = 32^x → x = 2 := 
by 
  intros x h
  sorry

end find_x_eq_2_l259_259016


namespace fraction_difference_l259_259473

theorem fraction_difference : (18 / 42) - (3 / 8) = 3 / 56 := 
by
  sorry

end fraction_difference_l259_259473


namespace problem1_problem2_l259_259628

-- Problem 1: Prove the expression equals the calculated value
theorem problem1 : (-2:ℝ)^0 + (1 / Real.sqrt 2) - Real.sqrt 9 = (Real.sqrt 2) / 2 - 2 :=
by sorry

-- Problem 2: Prove the solution to the system of linear equations
theorem problem2 (x y : ℝ) (h1 : 2 * x - y = 3) (h2 : x + y = -2) :
  x = 1/3 ∧ y = -(7/3) :=
by sorry

end problem1_problem2_l259_259628


namespace each_boy_receives_52_l259_259936

theorem each_boy_receives_52 {boys girls : ℕ} (h_ratio : boys / gcd boys girls = 5 ∧ girls / gcd boys girls = 7) (h_total : boys + girls = 180) (h_share : 3900 ∣ boys) :
  3900 / boys = 52 :=
by
  sorry

end each_boy_receives_52_l259_259936


namespace geometric_sequence_problem_l259_259766

variables {a : ℕ → ℝ} {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a 1 * q ^ n

theorem geometric_sequence_problem (h1 : a 1 + a 1 * q ^ 2 = 10) (h2 : a 1 * q + a 1 * q ^ 3 = 5) (h3 : geometric_sequence a q) :
  a 8 = 1 / 16 := sorry

end geometric_sequence_problem_l259_259766


namespace compare_abc_l259_259048

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  unfold a b c
  -- Proof goes here
  sorry

end compare_abc_l259_259048


namespace mom_chicken_cost_l259_259201

def cost_bananas : ℝ := 2 * 4 -- bananas cost
def cost_pears : ℝ := 2 -- pears cost
def cost_asparagus : ℝ := 6 -- asparagus cost
def total_expenses_other_than_chicken : ℝ := cost_bananas + cost_pears + cost_asparagus -- total cost of other items
def initial_money : ℝ := 55 -- initial amount of money
def remaining_money_after_other_purchases : ℝ := initial_money - total_expenses_other_than_chicken -- money left after covering other items

theorem mom_chicken_cost : 
  (remaining_money_after_other_purchases - 28 = 11) := 
by
  sorry

end mom_chicken_cost_l259_259201


namespace geometric_sequence_product_of_terms_l259_259512

theorem geometric_sequence_product_of_terms 
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_a2 : a 2 = 2)
  (h_a6 : a 6 = 8) : 
  a 3 * a 4 * a 5 = 64 := 
by
  sorry

end geometric_sequence_product_of_terms_l259_259512


namespace kim_spends_time_on_coffee_l259_259584

noncomputable def time_per_employee_status_update : ℕ := 2
noncomputable def time_per_employee_payroll_update : ℕ := 3
noncomputable def number_of_employees : ℕ := 9
noncomputable def total_morning_routine_time : ℕ := 50

theorem kim_spends_time_on_coffee :
  ∃ C : ℕ, C + (time_per_employee_status_update * number_of_employees) + 
  (time_per_employee_payroll_update * number_of_employees) = total_morning_routine_time ∧
  C = 5 :=
by
  sorry

end kim_spends_time_on_coffee_l259_259584


namespace expenditure_record_l259_259018

/-- Lean function to represent the condition and the proof problem -/
theorem expenditure_record (income expenditure : Int) (h_income : income = 500) (h_recorded_income : income = 500) (h_expenditure : expenditure = 200) : expenditure = -200 := 
by
  sorry

end expenditure_record_l259_259018


namespace max_value_ab_l259_259388

theorem max_value_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 5 * a + 8 * b = 80) : ab ≤ 40 := 
  sorry

end max_value_ab_l259_259388


namespace tails_and_die_1_or_2_l259_259396

noncomputable def fairCoinFlipProbability : ℚ := 1 / 2
noncomputable def fairDieRollProbability : ℚ := 1 / 6
noncomputable def combinedProbability : ℚ := fairCoinFlipProbability * (fairDieRollProbability + fairDieRollProbability)

theorem tails_and_die_1_or_2 :
  combinedProbability = 1 / 6 :=
by
  sorry

end tails_and_die_1_or_2_l259_259396


namespace isosceles_triangle_sides_l259_259980

theorem isosceles_triangle_sides (a b : ℝ) (h1 : 2 * a + a = 14 ∨ 2 * a + a = 18)
  (h2 : a + b = 18 ∨ a + b = 14) : 
  (a = 14/3 ∧ b = 40/3 ∨ a = 6 ∧ b = 8) :=
by
  sorry

end isosceles_triangle_sides_l259_259980


namespace solve_for_m_l259_259140

theorem solve_for_m (m : ℝ) (f g : ℝ → ℝ) :
  (∀ x : ℝ, f x = x^2 - 2 * x + m) →
  (∀ x : ℝ, g x = x^2 - 2 * x + 9 * m) →
  f 2 = 2 * g 2 →
  m = 0 :=
  by
    intros hf hg hs
    sorry

end solve_for_m_l259_259140


namespace basketball_team_wins_l259_259631

-- Define the known quantities
def games_won_initial : ℕ := 60
def games_total_initial : ℕ := 80
def games_left : ℕ := 50
def total_games : ℕ := games_total_initial + games_left
def desired_win_fraction : ℚ := 3 / 4

-- The main goal: Prove that the team must win 38 of the remaining 50 games to reach the desired win fraction
theorem basketball_team_wins :
  ∃ x : ℕ, x = 38 ∧ (games_won_initial + x : ℚ) / total_games = desired_win_fraction :=
by
  sorry

end basketball_team_wins_l259_259631


namespace basketball_game_l259_259569

variable (H E : ℕ)

theorem basketball_game (h_eq_sum : H + E = 50) (h_margin : H = E + 6) : E = 22 := by
  sorry

end basketball_game_l259_259569


namespace min_value_y_l259_259510

noncomputable def y (x : ℝ) := (2 - Real.cos x) / Real.sin x

theorem min_value_y (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi) : 
  ∃ c ≥ 0, ∀ x, 0 < x ∧ x < Real.pi → y x ≥ c ∧ c = Real.sqrt 3 := 
sorry

end min_value_y_l259_259510


namespace fraction_addition_l259_259794

theorem fraction_addition : (3 / 5) + (2 / 15) = 11 / 15 := sorry

end fraction_addition_l259_259794


namespace find_c_if_lines_parallel_l259_259101

theorem find_c_if_lines_parallel (c : ℝ) : 
  (∀ x : ℝ, 5 * x - 3 = (3 * c) * x + 1) → 
  c = 5 / 3 :=
by
  intro h
  sorry

end find_c_if_lines_parallel_l259_259101


namespace sum_first_four_terms_l259_259932

theorem sum_first_four_terms (a : ℕ → ℤ) (h5 : a 5 = 5) (h6 : a 6 = 9) (h7 : a 7 = 13) : 
  a 1 + a 2 + a 3 + a 4 = -20 :=
sorry

end sum_first_four_terms_l259_259932


namespace no_symmetry_line_for_exponential_l259_259673

theorem no_symmetry_line_for_exponential : ¬ ∃ l : ℝ → ℝ, ∀ x : ℝ, (2 ^ x) = l (2 ^ (2 * l x - x)) := 
sorry

end no_symmetry_line_for_exponential_l259_259673


namespace intersection_P_Q_l259_259837

def set_P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def set_Q : Set ℝ := {x | (x - 1) ^ 2 ≤ 4}

theorem intersection_P_Q :
  {x | x ∈ set_P ∧ x ∈ set_Q} = {x : ℝ | 1 ≤ x ∧ x ≤ 3} := by
  sorry

end intersection_P_Q_l259_259837


namespace combined_area_rectangle_triangle_l259_259875

/-- 
  Given a rectangle ABCD with vertices A = (10, -30), 
  B = (2010, 170), D = (12, -50), and a right triangle
  ADE with vertex E = (12, -30), prove that the combined
  area of the rectangle and the triangle is 
  40400 + 20√101.
-/
theorem combined_area_rectangle_triangle :
  let A := (10, -30)
  let B := (2010, 170)
  let D := (12, -50)
  let E := (12, -30)
  let length_AB := Real.sqrt ((2010 - 10)^2 + (170 + 30)^2)
  let length_AD := Real.sqrt ((12 - 10)^2 + (-50 + 30)^2)
  let area_rectangle := length_AB * length_AD
  let length_DE := Real.sqrt ((12 - 12)^2 + (-50 + 30)^2)
  let area_triangle := 1/2 * length_DE * length_AD
  area_rectangle + area_triangle = 40400 + 20 * Real.sqrt 101 :=
by
  sorry

end combined_area_rectangle_triangle_l259_259875


namespace john_work_days_l259_259188

theorem john_work_days (J : ℕ) (H1 : 1 / J + 1 / 480 = 1 / 192) : J = 320 :=
sorry

end john_work_days_l259_259188


namespace compare_abc_l259_259034

variables (a b c : ℝ)

-- Given conditions
def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

-- Goal to prove
theorem compare_abc : a > c ∧ c > b := sorry

end compare_abc_l259_259034


namespace domain_real_iff_l259_259355

noncomputable def is_domain_ℝ (m : ℝ) : Prop :=
  ∀ x : ℝ, (m * x^2 + 4 * m * x + 3 ≠ 0)

theorem domain_real_iff (m : ℝ) :
  is_domain_ℝ m ↔ (0 ≤ m ∧ m < 3 / 4) :=
sorry

end domain_real_iff_l259_259355


namespace area_of_support_is_15_l259_259460

-- Define the given conditions
def initial_mass : ℝ := 60
def reduced_mass : ℝ := initial_mass - 10
def area_reduction : ℝ := 5
def mass_per_area_increase : ℝ := 1

-- Define the area of the support and prove that it is 15 dm^2
theorem area_of_support_is_15 (x : ℝ) 
  (initial_mass_eq : initial_mass / x = initial_mass / x) 
  (new_mass_eq : reduced_mass / (x - area_reduction) = initial_mass / x + mass_per_area_increase) : 
  x = 15 :=
  sorry

end area_of_support_is_15_l259_259460


namespace negative_solution_exists_l259_259321

theorem negative_solution_exists (a b c x y : ℝ) :
  (a * x + b * y = c ∧ b * x + c * y = a ∧ c * x + a * y = b) ∧ (x < 0 ∧ y < 0) ↔ a + b + c = 0 :=
sorry

end negative_solution_exists_l259_259321


namespace system_has_negative_solution_iff_sum_zero_l259_259332

variables {a b c x y : ℝ}

-- Statement of the problem
theorem system_has_negative_solution_iff_sum_zero :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔ (a + b + c = 0) := by
  sorry

end system_has_negative_solution_iff_sum_zero_l259_259332


namespace systematic_sampling_remove_l259_259159

theorem systematic_sampling_remove (total_people : ℕ) (sample_size : ℕ) (remove_count : ℕ): 
  total_people = 162 → sample_size = 16 → remove_count = 2 → 
  (total_people - 1) % sample_size = sample_size - 1 :=
by
  sorry

end systematic_sampling_remove_l259_259159


namespace jill_trips_to_fill_tank_l259_259879

def tank_capacity : ℕ := 600
def bucket_volume : ℕ := 5
def jack_buckets_per_trip : ℕ := 2
def jill_buckets_per_trip : ℕ := 1
def jack_to_jill_trip_ratio : ℕ := 3 / 2

theorem jill_trips_to_fill_tank : (tank_capacity / bucket_volume) = 120 → 
                                   ((jack_to_jill_trip_ratio * jack_buckets_per_trip) + 2 * jill_buckets_per_trip) = 8 →
                                   15 * 2 = 30 :=
by
  intros h1 h2
  sorry

end jill_trips_to_fill_tank_l259_259879


namespace asha_money_remaining_l259_259652

-- Given conditions as definitions in Lean
def borrowed_from_brother : ℕ := 20
def borrowed_from_father : ℕ := 40
def borrowed_from_mother : ℕ := 30
def gift_from_granny : ℕ := 70
def initial_savings : ℕ := 100

-- Total amount of money Asha has
def total_money : ℕ := borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + initial_savings

-- Money spent by Asha
def money_spent : ℕ := (3 * total_money) / 4

-- Money remaining with Asha
def money_remaining : ℕ := total_money - money_spent

-- Theorem stating the result
theorem asha_money_remaining : money_remaining = 65 := by
  sorry

end asha_money_remaining_l259_259652


namespace compare_a_b_c_l259_259060

noncomputable def a := 2 * Real.log 1.01
noncomputable def b := Real.log 1.02
noncomputable def c := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l259_259060


namespace continuous_stripe_probability_l259_259780

-- Define the conditions of the tetrahedron and stripe orientations
def tetrahedron_faces : ℕ := 4
def stripe_orientations_per_face : ℕ := 2
def total_stripe_combinations : ℕ := stripe_orientations_per_face ^ tetrahedron_faces
def favorable_stripe_combinations : ℕ := 2 -- Clockwise and Counterclockwise combinations for a continuous stripe

-- Define the probability calculation
def probability_of_continuous_stripe : ℚ :=
  favorable_stripe_combinations / total_stripe_combinations

-- Theorem statement
theorem continuous_stripe_probability : probability_of_continuous_stripe = 1 / 8 :=
by
  -- The proof is omitted for brevity
  sorry

end continuous_stripe_probability_l259_259780


namespace cos_three_theta_l259_259361

theorem cos_three_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (3 * θ) = -11 / 16 := by
  sorry

end cos_three_theta_l259_259361


namespace tan_difference_l259_259564

variable (α β : ℝ)
variable (tan_α : ℝ := 3)
variable (tan_β : ℝ := 4 / 3)

theorem tan_difference (h₁ : Real.tan α = tan_α) (h₂ : Real.tan β = tan_β) : 
  Real.tan (α - β) = (tan_α - tan_β) / (1 + tan_α * tan_β) := by
  sorry

end tan_difference_l259_259564


namespace joseph_total_payment_l259_259190
-- Importing necessary libraries

-- Defining the variables and conditions
variables (W : ℝ) -- The cost for the water heater

-- Conditions
def condition1 := 3 * W -- The cost for the refrigerator
def condition2 := 2 * W = 500 -- The electric oven
def condition3 := 300 -- The cost for the air conditioner
def condition4 := 100 -- The cost for the washing machine

-- Calculate total cost
def total_cost := (3 * W) + W + 500 + 300 + 100

-- The theorem stating the total amount Joseph pays
theorem joseph_total_payment : total_cost = 1900 :=
by 
  have hW := condition2;
  sorry

end joseph_total_payment_l259_259190


namespace john_reams_needed_l259_259888

theorem john_reams_needed 
  (pages_flash_fiction_weekly : ℕ := 20) 
  (pages_short_story_weekly : ℕ := 50) 
  (pages_novel_annual : ℕ := 1500) 
  (weeks_in_year : ℕ := 52) 
  (sheets_per_ream : ℕ := 500) 
  (sheets_flash_fiction_weekly : ℕ := 10)
  (sheets_short_story_weekly : ℕ := 25) :
  let sheets_flash_fiction_annual := sheets_flash_fiction_weekly * weeks_in_year
  let sheets_short_story_annual := sheets_short_story_weekly * weeks_in_year
  let total_sheets_annual := sheets_flash_fiction_annual + sheets_short_story_annual + pages_novel_annual
  let reams_needed := (total_sheets_annual + sheets_per_ream - 1) / sheets_per_ream
  reams_needed = 7 := 
by sorry

end john_reams_needed_l259_259888


namespace exists_nat_number_divisible_by_2019_and_its_digit_sum_also_divisible_by_2019_l259_259603

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_nat_number_divisible_by_2019_and_its_digit_sum_also_divisible_by_2019 :
  ∃ N : ℕ, (N % 2019 = 0) ∧ ((sum_of_digits N) % 2019 = 0) :=
by 
  sorry

end exists_nat_number_divisible_by_2019_and_its_digit_sum_also_divisible_by_2019_l259_259603


namespace non_sophomores_is_75_percent_l259_259183

def students_not_sophomores_percentage (total_students : ℕ) 
                                       (percent_juniors : ℚ)
                                       (num_seniors : ℕ)
                                       (freshmen_more_than_sophomores : ℕ) : ℚ :=
  let num_juniors := total_students * percent_juniors 
  let s := (total_students - num_juniors - num_seniors - freshmen_more_than_sophomores) / 2
  let f := s + freshmen_more_than_sophomores
  let non_sophomores := total_students - s
  (non_sophomores / total_students) * 100

theorem non_sophomores_is_75_percent : students_not_sophomores_percentage 800 0.28 160 16 = 75 := by
  sorry

end non_sophomores_is_75_percent_l259_259183


namespace married_men_fraction_l259_259182

-- define the total number of women
def W : ℕ := 7

-- define the number of single women
def single_women (W : ℕ) : ℕ := 3

-- define the probability of picking a single woman
def P_s : ℚ := single_women W / W

-- define number of married women
def married_women (W : ℕ) : ℕ := W - single_women W

-- define number of married men
def married_men (W : ℕ) : ℕ := married_women W

-- define total number of people
def total_people (W : ℕ) : ℕ := W + married_men W

-- define fraction of married men
def married_men_ratio (W : ℕ) : ℚ := married_men W / total_people W

-- theorem to prove that the ratio is 4/11
theorem married_men_fraction : married_men_ratio W = 4 / 11 := 
by 
  sorry

end married_men_fraction_l259_259182


namespace pos_real_ineq_l259_259825

theorem pos_real_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^a * b^b * c^c ≥ (a * b * c)^((a + b + c)/3) :=
by 
  sorry

end pos_real_ineq_l259_259825


namespace number_of_boys_l259_259573

-- We define the conditions provided in the problem
def child_1_has_3_brothers : Prop := ∃ B G : ℕ, B - 1 = 3 ∧ G = 6
def child_2_has_4_brothers : Prop := ∃ B G : ℕ, B - 1 = 4 ∧ G = 5

theorem number_of_boys (B G : ℕ) (h1 : child_1_has_3_brothers) (h2 : child_2_has_4_brothers) : B = 4 :=
by
  sorry

end number_of_boys_l259_259573


namespace min_number_of_candy_kinds_l259_259863

theorem min_number_of_candy_kinds (n : ℕ) (h : n = 91)
  (even_distance_condition : ∀ i j : ℕ, i ≠ j → n > i ∧ n > j → 
    (∃k : ℕ, i - j = 2 * k) ∨ (∃k : ℕ, j - i = 2 * k) ) :
  91 / 2 =  46 := by
  sorry

end min_number_of_candy_kinds_l259_259863


namespace inequality_solution_set_l259_259695

theorem inequality_solution_set {a : ℝ} (x : ℝ) :
  (∀ x, (x - a) / (x^2 - 3 * x + 2) ≥ 0 ↔ (1 < x ∧ x ≤ a) ∨ (2 < x)) → (1 < a ∧ a < 2) :=
by 
  -- We would fill in the proof here. 
  sorry

end inequality_solution_set_l259_259695


namespace fraction_cal_handled_l259_259914

theorem fraction_cal_handled (Mabel Anthony Cal Jade : ℕ) 
  (h_Mabel : Mabel = 90)
  (h_Anthony : Anthony = Mabel + Mabel / 10)
  (h_Jade : Jade = 80)
  (h_Cal : Cal = Jade - 14) :
  (Cal : ℚ) / (Anthony : ℚ) = 2 / 3 :=
by
  sorry

end fraction_cal_handled_l259_259914


namespace triangle_formation_conditions_l259_259759

theorem triangle_formation_conditions (a b c : ℝ) :
  (a + b > c ∧ |a - b| < c) ↔ (a + b > c ∧ b + c > a ∧ c + a > b ∧ |a - b| < c ∧ |b - c| < a ∧ |c - a| < b) :=
sorry

end triangle_formation_conditions_l259_259759


namespace total_pairs_of_shoes_tried_l259_259000

theorem total_pairs_of_shoes_tried (first_store_pairs second_store_additional third_store_pairs fourth_store_factor : ℕ) 
  (h_first : first_store_pairs = 7)
  (h_second : second_store_additional = 2)
  (h_third : third_store_pairs = 0)
  (h_fourth : fourth_store_factor = 2) :
  first_store_pairs + (first_store_pairs + second_store_additional) + third_store_pairs + 
    (fourth_store_factor * (first_store_pairs + (first_store_pairs + second_store_additional) + third_store_pairs)) = 48 := 
  by 
    sorry

end total_pairs_of_shoes_tried_l259_259000


namespace expected_value_of_X_l259_259959

def is_optimal_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  (n >= 1000 ∧ n < 10000) ∧
  (digits.count 8) % 2 = 0

def optimal_numbers : set ℕ := { n | is_optimal_number n }

noncomputable def E_X : ℚ :=
  let total_numbers := 9000
  let optimal_count := 460
  let p := optimal_count / total_numbers
  10 * p

theorem expected_value_of_X : E_X = 23 / 45 :=
  by
  sorry

end expected_value_of_X_l259_259959


namespace range_of_a_l259_259368

def func (x : ℝ) : ℝ := x^2 - 4 * x

def domain (a : ℝ) := ∀ x, -4 ≤ x ∧ x ≤ a

def range_condition (y : ℝ) := -4 ≤ y ∧ y ≤ 32

theorem range_of_a (a : ℝ)
  (domain_condition : ∀ x, x ∈ set.Icc (-4) a → func x ∈ set.Icc (-4) 32) :
  2 ≤ a ∧ a ≤ 8 :=
sorry

end range_of_a_l259_259368


namespace joanne_main_job_hours_l259_259384

theorem joanne_main_job_hours (h : ℕ) (earn_main_job : ℝ) (earn_part_time : ℝ) (hours_part_time : ℕ) (days_week : ℕ) (total_weekly_earn : ℝ) :
  earn_main_job = 16.00 →
  earn_part_time = 13.50 →
  hours_part_time = 2 →
  days_week = 5 →
  total_weekly_earn = 775 →
  days_week * earn_main_job * h + days_week * earn_part_time * hours_part_time = total_weekly_earn →
  h = 8 :=
by
  sorry

end joanne_main_job_hours_l259_259384


namespace number_of_faces_of_prism_proof_l259_259424

noncomputable def number_of_faces_of_prism (n : ℕ) : ℕ := 2 + n

theorem number_of_faces_of_prism_proof (n : ℕ) (E_p E_py : ℕ) (h1 : E_p + E_py = 30) (h2 : E_p = 3 * n) (h3 : E_py = 2 * n) :
  number_of_faces_of_prism n = 8 :=
by
  sorry

end number_of_faces_of_prism_proof_l259_259424


namespace find_positive_integers_l259_259979

theorem find_positive_integers (n : ℕ) : 
  (∀ a : ℕ, a.gcd n = 1 → 2 * n * n ∣ a ^ n - 1) ↔ (n = 2 ∨ n = 6 ∨ n = 42 ∨ n = 1806) :=
sorry

end find_positive_integers_l259_259979


namespace tangents_quadrilateral_cyclic_l259_259642

variables {A B C D K L O1 O2 : Point}
variable (r : ℝ)
variable (AB_cut_circles : ∀ {A B : Point} {O1 O2 : Point}, is_intersect AB O1 O2)
variable (parallel_AB_O1O2 : is_parallel AB O1O2)
variable (tangents_formed_quadrilateral : is_quadrilateral C D K L)
variable (quadrilateral_contains_circles : contains C D K L O1 O2)

theorem tangents_quadrilateral_cyclic
  (h1: AB_cut_circles)
  (h2: parallel_AB_O1O2) 
  (h3: tangents_formed_quadrilateral)
  (h4: quadrilateral_contains_circles)
  : ∃ O : Circle, is_inscribed O C D K L :=
sorry

end tangents_quadrilateral_cyclic_l259_259642


namespace construct_triangle_l259_259799

theorem construct_triangle (α : ℝ) (a : ℝ) (p q : ℝ) :
  ∃ (b c : ℝ), ∃ (ABC : Triangle), 
    ABC.has_angle α ∧ 
    ABC.opposite_side_of_angle α = a ∧ 
    b / c = p / q ∧ 
    ABC.side_ratio b c = p / q ∧ 
    ABC.is_valid :=
sorry

end construct_triangle_l259_259799


namespace fraction_of_time_at_15_mph_l259_259110

theorem fraction_of_time_at_15_mph
  (t1 t2 : ℝ)
  (h : (5 * t1 + 15 * t2) / (t1 + t2) = 10) :
  t2 / (t1 + t2) = 1 / 2 :=
by
  sorry

end fraction_of_time_at_15_mph_l259_259110


namespace inequality_solution_set_minimum_value_expression_l259_259834

-- Definition of the function f
def f (x : ℝ) : ℝ := |x + 2| + 2 * |x - 1|

-- Inequality solution set for f(x) ≤ 4
theorem inequality_solution_set :
  { x : ℝ | 0 ≤ x ∧ x ≤ 4 / 3 } = { x : ℝ | f x ≤ 4 } := 
sorry

-- Minimum value of the given expression given conditions on a and b
theorem minimum_value_expression (a b : ℝ) (h1 : a > 1) (h2 : b > 0)
  (h3 : a + 2 * b = 3) :
  (1 / (a - 1)) + (2 / b) = 9 / 2 := 
sorry

end inequality_solution_set_minimum_value_expression_l259_259834


namespace different_books_read_l259_259230

theorem different_books_read (t_books d_books b_books td_same all_same : ℕ)
  (h_t_books : t_books = 23)
  (h_d_books : d_books = 12)
  (h_b_books : b_books = 17)
  (h_td_same : td_same = 3)
  (h_all_same : all_same = 1) : 
  t_books + d_books + b_books - (td_same + all_same) = 48 := 
by
  sorry

end different_books_read_l259_259230


namespace min_value_proof_l259_259031

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
  2 / a + 2 / b + 2 / c

theorem min_value_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_abc : a + b + c = 9) : 
  minimum_value a b c ≥ 2 := 
by 
  sorry

end min_value_proof_l259_259031


namespace num_inverses_mod_11_l259_259558

theorem num_inverses_mod_11 : (finset.filter (λ x, nat.coprime x 11) (finset.range 11)).card = 10 :=
sorry

end num_inverses_mod_11_l259_259558


namespace determine_original_price_l259_259792

namespace PriceProblem

variable (x : ℝ)

def final_price (x : ℝ) : ℝ := 0.98175 * x

theorem determine_original_price (h : final_price x = 100) : x = 101.86 :=
by
  sorry

end PriceProblem

end determine_original_price_l259_259792


namespace number_div_mult_l259_259105

theorem number_div_mult (n : ℕ) (h : n = 4) : (n / 6) * 12 = 8 :=
by
  sorry

end number_div_mult_l259_259105


namespace total_time_is_60_l259_259310

def emma_time : ℕ := 20
def fernando_time : ℕ := 2 * emma_time
def total_time : ℕ := emma_time + fernando_time

theorem total_time_is_60 : total_time = 60 := by
  sorry

end total_time_is_60_l259_259310


namespace combined_weight_loss_l259_259789

theorem combined_weight_loss :
  let aleesia_loss_per_week := 1.5
  let aleesia_weeks := 10
  let alexei_loss_per_week := 2.5
  let alexei_weeks := 8
  (aleesia_loss_per_week * aleesia_weeks) + (alexei_loss_per_week * alexei_weeks) = 35 := by
sorry

end combined_weight_loss_l259_259789


namespace common_ratio_of_geometric_sequence_l259_259712

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 3 - 3 * a 2 = 2)
  (h2 : 5 * a 4 = (12 * a 3 + 2 * a 5) / 2) :
  (∃ a1 : ℝ, ∃ q : ℝ,
    (∀ n, a n = a1 * q ^ (n - 1)) ∧ 
    q = 2) := 
by 
  sorry

end common_ratio_of_geometric_sequence_l259_259712


namespace sin_cos_identity_l259_259346

theorem sin_cos_identity (θ : Real) (h1 : 0 < θ ∧ θ < π) (h2 : Real.sin θ * Real.cos θ = - (1/8)) :
  Real.sin (2 * Real.pi + θ) - Real.sin ((Real.pi / 2) - θ) = (Real.sqrt 5) / 2 := by
  sorry

end sin_cos_identity_l259_259346


namespace find_angle_A_l259_259181
open Real

theorem find_angle_A
  (a b : ℝ)
  (A B : ℝ)
  (h1 : b = 2 * a)
  (h2 : B = A + 60) :
  A = 30 :=
by 
  sorry

end find_angle_A_l259_259181


namespace Chrysler_Building_floors_l259_259080

variable (C L : ℕ)

theorem Chrysler_Building_floors :
  (C = L + 11) → (C + L = 35) → (C = 23) :=
by
  intro h1 h2
  sorry

end Chrysler_Building_floors_l259_259080


namespace num_inverses_mod_11_l259_259557

theorem num_inverses_mod_11 : (Finset.filter (λ a, Nat.gcd a 11 = 1) (Finset.range 11)).card = 10 :=
sorry

end num_inverses_mod_11_l259_259557


namespace number_of_inverses_mod_11_l259_259548

theorem number_of_inverses_mod_11 : 
  ∃ n, n = 10 ∧ ∀ x ∈ finset.range 11, (gcd x 11 = 1 → ∃ y, (x * y) % 11 = 1) :=
by
  sorry

end number_of_inverses_mod_11_l259_259548


namespace ellipse_solution_l259_259791

theorem ellipse_solution :
  (∃ (a b : ℝ), a = 4 * Real.sqrt 2 + Real.sqrt 17 ∧ b = Real.sqrt (32 + 16 * Real.sqrt 34) ∧ (∀ (x y : ℝ), (3 * 0 ≤ y ∧ y ≤ 8) → (3 * 0 ≤ x ∧ x ≤ 5) → (Real.sqrt ((x+3)^2 + y^2) + Real.sqrt ((x-3)^2 + y^2) = 2 * a) → 
   (Real.sqrt ((x-0)^2 + (y-8)^2) = b))) :=
sorry

end ellipse_solution_l259_259791


namespace braden_total_money_after_winning_bet_l259_259654

def initial_amount : ℕ := 400
def factor : ℕ := 2
def winnings (initial_amount : ℕ) (factor : ℕ) : ℕ := factor * initial_amount

theorem braden_total_money_after_winning_bet : 
  winnings initial_amount factor + initial_amount = 1200 := 
by
  sorry

end braden_total_money_after_winning_bet_l259_259654


namespace arrival_in_capetown_l259_259475

noncomputable def departure_time_london : Time := Time.of "11:00:00"
def duration_london_ny : ℕ := 18
def tz_diff_london_ny : ℤ := 5
noncomputable def arrival_time_ny : Time := departure_time_london.add_hours(duration_london_ny - tz_diff_london_ny)

noncomputable def departure_time_ny : Time := Time.of "07:00:00"
def duration_ny_capetown : ℕ := 10
def tz_diff_ny_capetown : ℤ := -7
noncomputable def arrival_time_capetown : Time := departure_time_ny.add_hours(duration_ny_capetown + tz_diff_ny_capetown)

theorem arrival_in_capetown : arrival_time_capetown = Time.of "00:00:00" :=
by
  -- Prep the necessary components, time zone adjustments, and time calculations.
  sorry

end arrival_in_capetown_l259_259475


namespace proof_triangle_properties_l259_259568

noncomputable def triangle_b (a b c : ℝ) (C : ℝ) : Prop :=
  a = 2 ∧ b = 2 * c ∧ C = 30 * Real.pi / 180 →
  b = (4 * Real.sqrt 3) / 3

noncomputable def max_area (a b c : ℝ) : Prop :=
  a = 2 ∧ b = 2 * c ∧ (2 * c + c > a) → 
  ∃ S, S = (3 / 4) * Real.sqrt (c^2 - (4 / 9) * (4 - c^2)) ∧ S <= 4 / 3

theorem proof_triangle_properties :
  ∀ (a b c : ℝ) (C : ℝ),
  triangle_b a b c C ∧ max_area a b c :=
by sorry

end proof_triangle_properties_l259_259568


namespace simplify_expression_l259_259206

theorem simplify_expression (y : ℝ) : 
  (3 * y) ^ 3 - 2 * y * y ^ 2 + y ^ 4 = 25 * y ^ 3 + y ^ 4 :=
by
  sorry

end simplify_expression_l259_259206


namespace find_a_l259_259496

-- Define the given conditions
def parabola_eq (a b c y : ℝ) : ℝ := a * y^2 + b * y + c
def vertex : (ℝ × ℝ) := (3, -1)
def point_on_parabola : (ℝ × ℝ) := (7, 3)

-- Define the theorem to be proved
theorem find_a (a b c : ℝ) (h_eqn : ∀ y, parabola_eq a b c y = x)
  (h_vertex : parabola_eq a b c (-vertex.snd) = vertex.fst)
  (h_point : parabola_eq a b c (point_on_parabola.snd) = point_on_parabola.fst) :
  a = 1 / 4 := 
sorry

end find_a_l259_259496


namespace compose_frac_prod_eq_one_l259_259477

open Finset

def irreducible_fraction (n d : ℕ) := gcd n d = 1

theorem compose_frac_prod_eq_one :
  ∃ (a b c d e f : ℕ),
  {a, b, c, d, e, f} ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
   d ≠ e ∧ d ≠ f ∧ 
   e ≠ f) ∧
  irreducible_fraction a b ∧
  irreducible_fraction c d ∧
  irreducible_fraction e f ∧
  (a : ℚ) / b * (c : ℚ) / d * (e : ℚ) / f = 1 :=
begin
  sorry
end

end compose_frac_prod_eq_one_l259_259477


namespace water_bottle_size_l259_259594

-- Define conditions
def glasses_per_day : ℕ := 4
def ounces_per_glass : ℕ := 5
def fills_per_week : ℕ := 4
def days_per_week : ℕ := 7

-- Theorem statement
theorem water_bottle_size :
  (glasses_per_day * ounces_per_glass * days_per_week) / fills_per_week = 35 :=
by
  sorry

end water_bottle_size_l259_259594


namespace compare_abc_l259_259035

variables (a b c : ℝ)

-- Given conditions
def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

-- Goal to prove
theorem compare_abc : a > c ∧ c > b := sorry

end compare_abc_l259_259035


namespace bananas_bought_l259_259134

def cost_per_banana : ℝ := 5.00
def total_cost : ℝ := 20.00

theorem bananas_bought : total_cost / cost_per_banana = 4 :=
by {
   sorry
}

end bananas_bought_l259_259134


namespace subsets_selection_count_l259_259274

open Finset

def T : Finset ℕ := {1, 2, 3, 4, 5, 6} -- Assuming distinct integers represent a, b, c, d, e, f

theorem subsets_selection_count :
  (∃ A B : Finset ℕ, A ∪ B = T ∧ (A ∩ B).card = 3) →
  (nat.choose 6 3 * 3^3) / 2 = 270 :=
by
  sorry

end subsets_selection_count_l259_259274


namespace speed_of_stream_l259_259957

theorem speed_of_stream (b s : ℕ) 
  (h1 : b + s = 42) 
  (h2 : b - s = 24) :
  s = 9 := by sorry

end speed_of_stream_l259_259957


namespace inequality_example_l259_259899

theorem inequality_example (a b c : ℝ) (habc_pos : 0 < a ∧ 0 < b ∧ 0 < c) (habc_sum : a + b + c = 3) :
  18 * ((1 / ((3 - a) * (4 - a))) + (1 / ((3 - b) * (4 - b))) + (1 / ((3 - c) * (4 - c)))) + 2 * (a * b + b * c + c * a) ≥ 15 :=
by
  sorry

end inequality_example_l259_259899


namespace domain_v_l259_259305

noncomputable def v (x : ℝ) : ℝ := 1 / (Real.sqrt x + x - 1)

theorem domain_v :
  {x : ℝ | x >= 0 ∧ Real.sqrt x + x - 1 ≠ 0} = {x : ℝ | x ∈ Set.Ico 0 (Real.sqrt 5 - 1) ∪ Set.Ioi (Real.sqrt 5 - 1)} :=
by
  sorry

end domain_v_l259_259305


namespace compare_abc_l259_259046

noncomputable theory

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l259_259046


namespace minimum_kinds_of_candies_l259_259864

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
    It turned out that between any two candies of the same kind, there is an even number of candies. 
    What is the minimum number of kinds of candies that could be? -/
theorem minimum_kinds_of_candies (n : ℕ) (h : 91 < 2 * n) : n ≥ 46 :=
sorry

end minimum_kinds_of_candies_l259_259864


namespace total_shoes_tried_on_l259_259004

variable (T : Type)
variable (store1 store2 store3 store4 : T)
variable (pair_of_shoes : T → ℕ)
variable (c1 : pair_of_shoes store1 = 7)
variable (c2 : pair_of_shoes store2 = pair_of_shoes store1 + 2)
variable (c3 : pair_of_shoes store3 = 0)
variable (c4 : pair_of_shoes store4 = 2 * (pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3))

theorem total_shoes_tried_on (store1 store2 store3 store4 : T) (pair_of_shoes : T → ℕ) : 
  pair_of_shoes store1 = 7 →
  pair_of_shoes store2 = pair_of_shoes store1 + 2 →
  pair_of_shoes store3 = 0 →
  pair_of_shoes store4 = 2 * (pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3) →
  pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3 + pair_of_shoes store4 = 48 := by
  intro c1 c2 c3 c4
  sorry

end total_shoes_tried_on_l259_259004


namespace count_inverses_modulo_11_l259_259542

theorem count_inverses_modulo_11 : (∀ a : ℤ, 0 ≤ a ∧ a ≤ 10 → ∃ b : ℤ, a * b ≡ 1 [MOD 11]) → (finset.range 11).filter (λ a, (nat.gcd a 11 = 1)).card = 10 :=
by
  sorry

end count_inverses_modulo_11_l259_259542


namespace categorize_numbers_l259_259313

noncomputable def positive_numbers (nums : Set ℝ) : Set ℝ :=
  {x | x ∈ nums ∧ x > 0}

noncomputable def non_neg_integers (nums : Set ℝ) : Set ℝ :=
  {x | x ∈ nums ∧ x ≥ 0 ∧ ∃ n : ℤ, x = n}

noncomputable def negative_fractions (nums : Set ℝ) : Set ℝ :=
  {x | x ∈ nums ∧ x < 0 ∧ ∃ n d : ℤ, d ≠ 0 ∧ (x = n / d)}

def given_set : Set ℝ := {6, -3, 2.4, -3/4, 0, -3.14, 2, -7/2, 2/3}

theorem categorize_numbers :
  positive_numbers given_set = {6, 2.4, 2, 2/3} ∧
  non_neg_integers given_set = {6, 0, 2} ∧
  negative_fractions given_set = {-3/4, -3.14, -7/2} :=
by
  sorry

end categorize_numbers_l259_259313


namespace part1_solution_set_part2_minimum_value_l259_259833

-- Definitions for Part (1)
def f (x : Real) : Real := abs (x + 2) + 2 * abs (x - 1)

-- Problem (1): Prove the solution to the inequality f(x) ≤ 4 is [0, 4/3]
theorem part1_solution_set (x : Real) : 0 ≤ x ∧ x ≤ 4 / 3 ↔ f x ≤ 4 := sorry

-- Definitions for Part (2)
def expression (a b : Real) : Real := 1 / (a - 1) + 2 / b

-- Problem (2): Prove the minimum value of the expression given the constraints is 9/2
theorem part2_minimum_value (a b : Real) (h1 : a + 2 * b = 3) (h2 : a > 1) (h3 : b > 0) : 
  expression a b = 9 / 2 := sorry

end part1_solution_set_part2_minimum_value_l259_259833


namespace max_value_of_function_is_seven_l259_259842

theorem max_value_of_function_is_seven:
  ∃ a: ℕ, (0 < a) ∧ 
  (∃ x: ℝ, (x + Real.sqrt (13 - 2 * a * x)) = 7 ∧
    ∀ y: ℝ, (y = x + Real.sqrt (13 - 2 * a * x)) → y ≤ 7) :=
sorry

end max_value_of_function_is_seven_l259_259842


namespace different_books_l259_259232

-- Definitions for the conditions
def tony_books : ℕ := 23
def dean_books : ℕ := 12
def breanna_books : ℕ := 17
def common_books_tony_dean : ℕ := 3
def common_books_all : ℕ := 1

-- Prove the total number of different books they have read is 47
theorem different_books : (tony_books + dean_books - common_books_tony_dean + breanna_books - 2 * common_books_all) = 47 :=
by
  sorry

end different_books_l259_259232


namespace combined_surface_area_of_cube_and_sphere_l259_259634

theorem combined_surface_area_of_cube_and_sphere (V_cube : ℝ) :
  V_cube = 729 →
  ∃ (A_combined : ℝ), A_combined = 486 + 81 * Real.pi :=
by
  intro V_cube
  sorry

end combined_surface_area_of_cube_and_sphere_l259_259634


namespace cos_alpha_minus_beta_l259_259509

theorem cos_alpha_minus_beta (α β : ℝ) (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π)
  (h_roots : ∀ x : ℝ, x^2 + 3*x + 1 = 0 → (x = Real.tan α ∨ x = Real.tan β)) : Real.cos (α - β) = 2 / 3 := 
by
  sorry

end cos_alpha_minus_beta_l259_259509


namespace polyhedron_volume_l259_259224

/-- Each 12 cm × 12 cm square is cut into two right-angled isosceles triangles by joining the midpoints of two adjacent sides. 
    These six triangles are attached to a regular hexagon to form a polyhedron.
    Prove that the volume of the resulting polyhedron is 864 cubic cm. -/
theorem polyhedron_volume :
  let s : ℝ := 12
  let volume_of_cube := s^3
  let volume_of_polyhedron := volume_of_cube / 2
  volume_of_polyhedron = 864 := 
by
  sorry

end polyhedron_volume_l259_259224


namespace jen_ate_eleven_suckers_l259_259407

/-- Representation of the sucker distribution problem and proving that Jen ate 11 suckers. -/
theorem jen_ate_eleven_suckers 
  (sienna_bailey : ℕ) -- Sienna's number of suckers is twice of what Bailey got.
  (jen_molly : ℕ)     -- Jen's number of suckers is twice of what Molly got plus 11.
  (molly_harmony : ℕ) -- Molly's number of suckers is 2 more than what she gave to Harmony.
  (harmony_taylor : ℕ)-- Harmony's number of suckers is 3 more than what she gave to Taylor.
  (taylor_end : ℕ)    -- Taylor ended with 6 suckers after eating 1 before giving 5 to Callie.
  (jen_start : ℕ)     -- Jen's initial number of suckers before eating half.
  (h1 : taylor_end = 6) 
  (h2 : harmony_taylor = taylor_end + 3) 
  (h3 : molly_harmony = harmony_taylor + 2) 
  (h4 : jen_molly = molly_harmony + 11) 
  (h5 : jen_start = jen_molly * 2) :
  jen_start / 2 = 11 := 
by
  -- given all the conditions, it would simplify to show
  -- that jen_start / 2 = 11
  sorry

end jen_ate_eleven_suckers_l259_259407


namespace max_possible_salary_l259_259845

-- Definition of the conditions
def num_players : ℕ := 25
def min_salary : ℕ := 20000
def total_salary_cap : ℕ := 800000

-- The theorem we want to prove: the maximum possible salary for a single player is $320,000
theorem max_possible_salary (total_salary_cap : ℕ) (num_players : ℕ) (min_salary : ℕ) :
  total_salary_cap - (num_players - 1) * min_salary = 320000 :=
by sorry

end max_possible_salary_l259_259845


namespace fencing_required_l259_259638

theorem fencing_required (L W : ℝ) (hL : L = 20) (hArea : L * W = 60) : (L + 2 * W) = 26 := 
by
  sorry

end fencing_required_l259_259638


namespace first_discount_percentage_l259_259117

theorem first_discount_percentage :
  ∃ x : ℝ, (9649.12 * (1 - x / 100) * 0.9 * 0.95 = 6600) ∧ (19.64 ≤ x ∧ x ≤ 19.66) :=
sorry

end first_discount_percentage_l259_259117


namespace determine_min_guesses_l259_259124

def minimum_guesses (n k : ℕ) (h : n > k) : ℕ :=
  if n = 2 * k then 2 else 1

theorem determine_min_guesses (n k : ℕ) (h : n > k) :
  (if n = 2 * k then 2 else 1) = minimum_guesses n k h := by
  sorry

end determine_min_guesses_l259_259124


namespace negative_solution_condition_l259_259325

-- Define the system of equations
def system_of_equations (a b c x y : ℝ) :=
  a * x + b * y = c ∧
  b * x + c * y = a ∧
  c * x + a * y = b

-- State the theorem
theorem negative_solution_condition (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ system_of_equations a b c x y) ↔ a + b + c = 0 :=
by
  sorry

end negative_solution_condition_l259_259325


namespace coloring_count_l259_259030

theorem coloring_count (n : ℕ) (h : 0 < n) :
  ∃ (num_colorings : ℕ), num_colorings = 2 :=
sorry

end coloring_count_l259_259030


namespace find_usual_time_l259_259451

variable (R T : ℝ)

theorem find_usual_time
  (h_condition :  R * T = (9 / 8) * R * (T - 4)) :
  T = 36 :=
by
  sorry

end find_usual_time_l259_259451


namespace diesel_train_slower_l259_259771

theorem diesel_train_slower
    (t_cattle_speed : ℕ)
    (t_cattle_early_hours : ℕ)
    (t_diesel_hours : ℕ)
    (total_distance : ℕ)
    (diesel_speed : ℕ) :
  t_cattle_speed = 56 →
  t_cattle_early_hours = 6 →
  t_diesel_hours = 12 →
  total_distance = 1284 →
  diesel_speed = 23 →
  t_cattle_speed - diesel_speed = 33 :=
by
  intros H1 H2 H3 H4 H5
  sorry

end diesel_train_slower_l259_259771


namespace part_a_part_b_l259_259929

-- Let p_k represent the probability that at the moment of completing the first collection, the second collection is missing exactly k crocodiles.
def p (k : ℕ) : ℝ := sorry

-- The conditions 
def totalCrocodiles : ℕ := 10
def probabilityEachEgg : ℝ := 0.1

-- Problems to prove:

-- (a) Prove that p_1 = p_2
theorem part_a : p 1 = p 2 := sorry

-- (b) Prove that p_2 > p_3 > p_4 > ... > p_10
theorem part_b : ∀ k, 2 ≤ k ∧ k < totalCrocodiles → p k > p (k + 1) := sorry

end part_a_part_b_l259_259929


namespace retailer_mark_up_l259_259781

theorem retailer_mark_up (R C M S : ℝ) 
  (hC : C = 0.7 * R)
  (hS : S = C / 0.7)
  (hSm : S = 0.9 * M) : 
  M = 1.111 * R :=
by 
  sorry

end retailer_mark_up_l259_259781


namespace remaining_money_l259_259726

def potato_cost : ℕ := 6 * 2
def tomato_cost : ℕ := 9 * 3
def cucumber_cost : ℕ := 5 * 4
def banana_cost : ℕ := 3 * 5
def total_cost : ℕ := potato_cost + tomato_cost + cucumber_cost + banana_cost
def initial_money : ℕ := 500

theorem remaining_money : initial_money - total_cost = 426 :=
by
  sorry

end remaining_money_l259_259726


namespace tan_squared_sum_geq_three_over_eight_l259_259197

theorem tan_squared_sum_geq_three_over_eight 
  (α β γ : ℝ) 
  (hα : 0 ≤ α ∧ α < π / 2) 
  (hβ : 0 ≤ β ∧ β < π / 2) 
  (hγ : 0 ≤ γ ∧ γ < π / 2) 
  (h_sum : Real.sin α + Real.sin β + Real.sin γ = 1) :
  Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan γ ^ 2 ≥ 3 / 8 := 
sorry

end tan_squared_sum_geq_three_over_eight_l259_259197


namespace soccer_substitutions_mod_2000_l259_259643

theorem soccer_substitutions_mod_2000 :
  let a_0 := 1
  let a_1 := 11 * 11
  let a_2 := 11 * 10 * a_1
  let a_3 := 11 * 9 * a_2
  let a_4 := 11 * 8 * a_3
  let n := a_0 + a_1 + a_2 + a_3 + a_4
  n % 2000 = 942 :=
by
  sorry

end soccer_substitutions_mod_2000_l259_259643


namespace minimum_candy_kinds_l259_259859

theorem minimum_candy_kinds (candy_count : ℕ) (h1 : candy_count = 91)
  (h2 : ∀ (k : ℕ), k ∈ (1 : ℕ) → (λ i j, abs (i - j) % 2 = 0))
  : ∃ (kinds : ℕ), kinds = 46 :=
by
  sorry

end minimum_candy_kinds_l259_259859


namespace evaluate_expression_l259_259501

theorem evaluate_expression : (2^3001 * 3^3003) / 6^3002 = 3 / 2 :=
by
  sorry

end evaluate_expression_l259_259501


namespace carnations_percentage_l259_259459

-- Definition of the total number of flowers
def total_flowers (F : ℕ) : Prop := 
  F > 0

-- Definition of the pink roses condition
def pink_roses_condition (F : ℕ) : Prop := 
  (1 / 2) * (3 / 5) * F = (3 / 10) * F

-- Definition of the red carnations condition
def red_carnations_condition (F : ℕ) : Prop := 
  (1 / 3) * (2 / 5) * F = (2 / 15) * F

-- Definition of the total pink flowers
def pink_flowers_condition (F : ℕ) : Prop :=
  (3 / 5) * F > 0

-- Proof that the percentage of the flowers that are carnations is 50%
theorem carnations_percentage (F : ℕ) (h_total : total_flowers F) (h_pink_roses : pink_roses_condition F) (h_red_carnations : red_carnations_condition F) (h_pink_flowers : pink_flowers_condition F) :
  (1 / 2) * 100 = 50 :=
by
  sorry

end carnations_percentage_l259_259459


namespace button_remainders_l259_259609

theorem button_remainders 
  (a : ℤ)
  (h1 : a % 2 = 1)
  (h2 : a % 3 = 1)
  (h3 : a % 4 = 3)
  (h4 : a % 5 = 3) :
  a % 12 = 7 := 
sorry

end button_remainders_l259_259609


namespace ab_range_l259_259816

theorem ab_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b + (1 / a) + (1 / b) = 5) :
  1 ≤ a + b ∧ a + b ≤ 4 :=
by
  sorry

end ab_range_l259_259816


namespace find_constants_l259_259785

noncomputable def f (x : ℕ) (a c : ℕ) : ℝ :=
  if x < a then c / Real.sqrt x else c / Real.sqrt a

theorem find_constants (a c : ℕ) (h₁ : f 4 a c = 30) (h₂ : f a a c = 5) : 
  c = 60 ∧ a = 144 := 
by
  sorry

end find_constants_l259_259785


namespace number_of_squares_in_H_l259_259421

-- Define the set H
def H : Set (ℤ × ℤ) :=
{ p | 2 ≤ abs p.1 ∧ abs p.1 ≤ 10 ∧ 2 ≤ abs p.2 ∧ abs p.2 ≤ 10 }

-- State the problem
theorem number_of_squares_in_H : 
  (∃ S : Finset (ℤ × ℤ), S.card = 20 ∧ 
    ∀ square ∈ S, 
      (∃ a b c d : ℤ × ℤ, 
        a ∈ H ∧ b ∈ H ∧ c ∈ H ∧ d ∈ H ∧ 
        (∃ s : ℤ, s ≥ 8 ∧ 
          (a.1 = b.1 ∧ b.2 = c.2 ∧ c.1 = d.1 ∧ d.2 = a.2 ∧ 
           abs (a.1 - c.1) = s ∧ abs (a.2 - d.2) = s)))) :=
sorry

end number_of_squares_in_H_l259_259421


namespace largest_lucky_number_l259_259276

theorem largest_lucky_number (n : ℕ) (h₀ : n = 160) (h₁ : ∀ k, 160 > k → k > 0) (h₂ : ∀ k, k ≡ 7 [MOD 16] → k ≤ 160) : 
  ∃ k, k = 151 := 
sorry

end largest_lucky_number_l259_259276


namespace quadratic_complete_square_r_plus_s_l259_259718

theorem quadratic_complete_square_r_plus_s :
  ∃ r s : ℚ, (∀ x : ℚ, 7 * x^2 - 21 * x - 56 = 0 → (x + r)^2 = s) ∧ r + s = 35 / 4 := sorry

end quadratic_complete_square_r_plus_s_l259_259718


namespace area_of_triangle_l259_259601

theorem area_of_triangle (side1 : ℝ) (side2 : ℝ) (side3 : ℝ) (angle : ℝ) (h_side_ratio : side2 / side3 = 8 / 5)
  (h_side_opposite : side1 = 14)
  (h_angle_opposite : angle = 60) :
  (1/2 * side2 * side3 * Real.sin (angle * Real.pi / 180)) = 40 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l259_259601


namespace biology_class_grades_l259_259706

theorem biology_class_grades (total_students : ℕ)
  (PA PB PC PD : ℕ)
  (h1 : PA = 12 * PB / 10)
  (h2 : PC = PB)
  (h3 : PD = 5 * PB / 10)
  (h4 : PA + PB + PC + PD = total_students) :
  total_students = 40 → PB = 11 := 
by
  sorry

end biology_class_grades_l259_259706


namespace compute_f5_l259_259360

-- Definitions of the logical operations used in the conditions
axiom x1 : Prop
axiom x2 : Prop
axiom x3 : Prop
axiom x4 : Prop
axiom x5 : Prop

noncomputable def x6 : Prop := x1 ∨ x3
noncomputable def x7 : Prop := x2 ∧ x6
noncomputable def x8 : Prop := x3 ∨ x5
noncomputable def x9 : Prop := x4 ∧ x8
noncomputable def f5 : Prop := x7 ∨ x9

-- Proof statement to be proven
theorem compute_f5 : f5 = (x7 ∨ x9) :=
by sorry

end compute_f5_l259_259360


namespace find_x_l259_259013

theorem find_x (x : ℕ) (h : 2^10 = 32^x) (h32 : 32 = 2^5) : x = 2 :=
sorry

end find_x_l259_259013


namespace find_f_difference_l259_259991

variable {α : Type*}
variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_period : ∀ x, f (x + 5) = f x)
variable (h_value : f (-2) = 2)

theorem find_f_difference : f 2012 - f 2010 = -2 :=
by {
  sorry
}

end find_f_difference_l259_259991


namespace determine_a_values_l259_259970

theorem determine_a_values (a b c : ℤ) :
  (∀ x : ℤ, (x - a) * (x - 5) + 2 = (x + b) * (x + c)) ↔ a = 2 ∨ a = 8 :=
by
  sorry

end determine_a_values_l259_259970


namespace babies_per_batch_l259_259264

def num_kettles : ℕ := 6
def pregnancies_per_kettle : ℕ := 15
def total_pregnancies : ℕ := num_kettles * pregnancies_per_kettle
def loss_rate : ℝ := 0.25
def survival_rate : ℝ := 1 - loss_rate
def expected_babies : ℕ := 270
def total_babies_before_loss : ℕ := (expected_babies : ℝ) / survival_rate

theorem babies_per_batch :
  (total_babies_before_loss / total_pregnancies) = 4 :=
sorry

end babies_per_batch_l259_259264


namespace negative_solution_exists_l259_259324

theorem negative_solution_exists (a b c x y : ℝ) :
  (a * x + b * y = c ∧ b * x + c * y = a ∧ c * x + a * y = b) ∧ (x < 0 ∧ y < 0) ↔ a + b + c = 0 :=
sorry

end negative_solution_exists_l259_259324


namespace tan_frac_eq_one_l259_259700

open Real

-- Conditions given in the problem
def sin_frac_cond (x y : ℝ) : Prop := (sin x / sin y) + (sin y / sin x) = 4
def cos_frac_cond (x y : ℝ) : Prop := (cos x / cos y) + (cos y / cos x) = 3

-- Statement of the theorem to be proved
theorem tan_frac_eq_one (x y : ℝ) (h1 : sin_frac_cond x y) (h2 : cos_frac_cond x y) : (tan x / tan y) + (tan y / tan x) = 1 :=
by
  sorry

end tan_frac_eq_one_l259_259700


namespace problem1_problem2_problem3_l259_259738

-- Problem 1
theorem problem1 (x : ℝ) : (3 * (x - 1)^2 = 12) ↔ (x = 3 ∨ x = -1) :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : (3 * x^2 - 6 * x - 2 = 0) ↔ (x = (3 + Real.sqrt 15) / 3 ∨ x = (3 - Real.sqrt 15) / 3) :=
by
  sorry

-- Problem 3
theorem problem3 (x : ℝ) : (3 * x * (2 * x + 1) = 4 * x + 2) ↔ (x = -1 / 2 ∨ x = 2 / 3) :=
by
  sorry

end problem1_problem2_problem3_l259_259738


namespace total_points_other_members_18_l259_259877

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

end total_points_other_members_18_l259_259877


namespace increment_in_displacement_l259_259088

variable (d : ℝ)

def equation_of_motion (t : ℝ) : ℝ := 2 * t^2

theorem increment_in_displacement:
  let t1 := 2
  let t2 := 2 + d
  let s1 := equation_of_motion t1
  let s2 := equation_of_motion t2
  s2 - s1 = 8 * d + 2 * d^2 := by
  sorry

end increment_in_displacement_l259_259088


namespace ratio_of_width_to_perimeter_l259_259782

-- Condition definitions
def length := 22
def width := 13
def perimeter := 2 * (length + width)

-- Statement of the problem in Lean 4
theorem ratio_of_width_to_perimeter : width = 13 ∧ length = 22 → width * 70 = 13 * perimeter :=
by
  sorry

end ratio_of_width_to_perimeter_l259_259782


namespace jim_needs_more_miles_l259_259763

-- Define the conditions
def totalMiles : ℕ := 1200
def drivenMiles : ℕ := 923

-- Define the question and the correct answer
def remainingMiles : ℕ := totalMiles - drivenMiles

-- The theorem statement
theorem jim_needs_more_miles : remainingMiles = 277 :=
by
  -- This will contain the proof which is to be done later
  sorry

end jim_needs_more_miles_l259_259763


namespace total_shoes_tried_on_l259_259005

variable (T : Type)
variable (store1 store2 store3 store4 : T)
variable (pair_of_shoes : T → ℕ)
variable (c1 : pair_of_shoes store1 = 7)
variable (c2 : pair_of_shoes store2 = pair_of_shoes store1 + 2)
variable (c3 : pair_of_shoes store3 = 0)
variable (c4 : pair_of_shoes store4 = 2 * (pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3))

theorem total_shoes_tried_on (store1 store2 store3 store4 : T) (pair_of_shoes : T → ℕ) : 
  pair_of_shoes store1 = 7 →
  pair_of_shoes store2 = pair_of_shoes store1 + 2 →
  pair_of_shoes store3 = 0 →
  pair_of_shoes store4 = 2 * (pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3) →
  pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3 + pair_of_shoes store4 = 48 := by
  intro c1 c2 c3 c4
  sorry

end total_shoes_tried_on_l259_259005


namespace positive_value_of_A_l259_259586

-- Define the relation
def hash (A B : ℝ) : ℝ := A^2 - B^2

-- State the main theorem
theorem positive_value_of_A (A : ℝ) : hash A 7 = 72 → A = 11 :=
by
  -- Placeholder for the proof
  sorry

end positive_value_of_A_l259_259586


namespace andy_l259_259288

theorem andy's_profit_per_cake :
  (∀ (cakes : ℕ), cakes = 2 → ∀ (ingredient_cost : ℕ), ingredient_cost = 12 →
                  ∀ (packaging_cost_per_cake : ℕ), packaging_cost_per_cake = 1 →
                  ∀ (selling_price_per_cake : ℕ), selling_price_per_cake = 15 →
                  ∀ (profit_per_cake : ℕ), profit_per_cake = selling_price_per_cake - (ingredient_cost / cakes + packaging_cost_per_cake) →
                    profit_per_cake = 8) :=
by
  sorry

end andy_l259_259288


namespace compare_abc_l259_259054

def a : ℝ := 2 * Real.log 1.01
def b : ℝ := Real.log 1.02
def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b :=
by
  /- Proof goes here -/
  sorry

end compare_abc_l259_259054


namespace fourth_child_sweets_l259_259777

theorem fourth_child_sweets (total_sweets : ℕ) (mother_sweets : ℕ) (child_sweets : ℕ) 
  (Y E T F: ℕ) (h1 : total_sweets = 120) (h2 : mother_sweets = total_sweets / 4) 
  (h3 : child_sweets = total_sweets - mother_sweets) 
  (h4 : E = 2 * Y) (h5 : T = F - 8) 
  (h6 : Y = (8 * (T + 6)) / 10) 
  (h7 : Y + E + (T + 6) + (F - 8) + F = child_sweets) : 
  F = 24 :=
by
  sorry

end fourth_child_sweets_l259_259777


namespace remainder_division_l259_259636

theorem remainder_division (N : ℤ) (hN : N % 899 = 63) : N % 29 = 5 := 
by 
  sorry

end remainder_division_l259_259636


namespace red_marbles_difference_l259_259446

theorem red_marbles_difference 
  (x y : ℕ) 
  (h1 : 7 * x + 3 * x = 140) 
  (h2 : 3 * y + 2 * y = 140)
  (h3 : 10 * x = 5 * y) : 
  7 * x - 3 * y = 20 := 
by 
  sorry

end red_marbles_difference_l259_259446


namespace stewart_farm_horse_food_l259_259129

theorem stewart_farm_horse_food 
  (ratio : ℚ) (food_per_horse : ℤ) (num_sheep : ℤ) (num_horses : ℤ)
  (h1 : ratio = 5 / 7)
  (h2 : food_per_horse = 230)
  (h3 : num_sheep = 40)
  (h4 : ratio * num_horses = num_sheep) : 
  (num_horses * food_per_horse = 12880) := 
sorry

end stewart_farm_horse_food_l259_259129


namespace dice_probability_sum_15_l259_259094

def is_valid_combination (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ a + b + c = 15

def count_outcomes : ℕ :=
  6 * 6 * 6

def count_valid_combinations : ℕ :=
  10  -- From the list of valid combinations

def probability (valid_count total_count : ℕ) : ℚ :=
  valid_count / total_count

theorem dice_probability_sum_15 : probability count_valid_combinations count_outcomes = 5 / 108 :=
by
  sorry

end dice_probability_sum_15_l259_259094


namespace find_b_l259_259448

variables (a b : ℕ)

theorem find_b
  (h1 : a = 105)
  (h2 : a ^ 3 = 21 * 25 * 315 * b) :
  b = 7 :=
sorry

end find_b_l259_259448


namespace balloon_arrangement_count_l259_259670

theorem balloon_arrangement_count :
  let total_permutations := (Nat.factorial 7) / (Nat.factorial 2 * Nat.factorial 3)
  let ways_to_arrange_L_and_O := Nat.choose 4 1 * (Nat.factorial 3)
  let valid_arrangements := ways_to_arrange_L_and_O * total_permutations
  valid_arrangements = 10080 :=
by
  sorry

end balloon_arrangement_count_l259_259670


namespace arithmetic_sequence_k_l259_259377

theorem arithmetic_sequence_k (d : ℤ) (h_d : d ≠ 0) (a : ℕ → ℤ) 
  (h_seq : ∀ n, a n = 0 + n * d) (h_k : a 21 = a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6):
  21 = 21 :=
by
  -- This would be the problem setup
  -- The proof would go here
  sorry

end arithmetic_sequence_k_l259_259377


namespace smallest_solution_floor_eq_3_l259_259903

def g (x : ℝ) : ℝ := 2 * Real.sin x + 3 * Real.cos x + 4 * Real.tan x

theorem smallest_solution_floor_eq_3 :
  ∃ s > 0, g s = 0 ∧ Int.floor s = 3 :=
sorry

end smallest_solution_floor_eq_3_l259_259903


namespace spanish_peanuts_l259_259411

variable (x : ℝ)

theorem spanish_peanuts :
  (10 * 3.50 + x * 3.00 = (10 + x) * 3.40) → x = 2.5 :=
by
  intro h
  sorry

end spanish_peanuts_l259_259411


namespace mutually_exclusive_scoring_l259_259783

-- Define conditions as types
def shoots_twice : Prop := true
def scoring_at_least_once : Prop :=
  ∃ (shot1 shot2 : Bool), shot1 || shot2
def not_scoring_both_times : Prop :=
  ∀ (shot1 shot2 : Bool), ¬(shot1 && shot2)

-- Statement of the problem: Prove the events are mutually exclusive.
theorem mutually_exclusive_scoring :
  shoots_twice → (scoring_at_least_once → not_scoring_both_times → false) :=
by
  intro h_shoots_twice
  intro h_scoring_at_least_once
  intro h_not_scoring_both_times
  sorry

end mutually_exclusive_scoring_l259_259783


namespace inequality_negatives_l259_259815

theorem inequality_negatives (a b : ℝ) (h₁ : a < b) (h₂ : b < 0) : a^2 > b^2 :=
sorry

end inequality_negatives_l259_259815


namespace pyramid_surface_area_l259_259271

theorem pyramid_surface_area
  (base_side_length : ℝ)
  (peak_height : ℝ)
  (base_area : ℝ)
  (slant_height : ℝ)
  (triangular_face_area : ℝ)
  (total_surface_area : ℝ)
  (h1 : base_side_length = 10)
  (h2 : peak_height = 12)
  (h3 : base_area = base_side_length ^ 2)
  (h4 : slant_height = Real.sqrt (peak_height ^ 2 + (base_side_length / 2) ^ 2))
  (h5 : triangular_face_area = 0.5 * base_side_length * slant_height)
  (h6 : total_surface_area = base_area + 4 * triangular_face_area)
  : total_surface_area = 360 := 
sorry

end pyramid_surface_area_l259_259271


namespace drank_bottles_of_juice_l259_259647

theorem drank_bottles_of_juice
  (bottles_in_refrigerator : ℕ)
  (bottles_in_pantry : ℕ)
  (bottles_bought : ℕ)
  (bottles_left : ℕ)
  (initial_bottles := bottles_in_refrigerator + bottles_in_pantry)
  (total_bottles := initial_bottles + bottles_bought)
  (bottles_drank := total_bottles - bottles_left) :
  bottles_in_refrigerator = 4 ∧
  bottles_in_pantry = 4 ∧
  bottles_bought = 5 ∧
  bottles_left = 10 →
  bottles_drank = 3 :=
by sorry

end drank_bottles_of_juice_l259_259647


namespace sequence_bounded_l259_259222

open Classical

noncomputable def bounded_sequence (a : ℕ → ℝ) (M : ℝ) :=
  ∀ n : ℕ, n > 0 → a n < M

theorem sequence_bounded {a : ℕ → ℝ} (h0 : 0 ≤ a 1 ∧ a 1 ≤ 2)
  (h : ∀ n : ℕ, n > 0 → a (n + 1) = a n + (a n)^2 / n^3) :
  ∃ M : ℝ, 0 < M ∧ bounded_sequence a M :=
by
  sorry

end sequence_bounded_l259_259222


namespace system_has_negative_solution_iff_sum_zero_l259_259333

variables {a b c x y : ℝ}

-- Statement of the problem
theorem system_has_negative_solution_iff_sum_zero :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔ (a + b + c = 0) := by
  sorry

end system_has_negative_solution_iff_sum_zero_l259_259333


namespace average_words_per_hour_l259_259283

theorem average_words_per_hour
  (total_words : ℕ := 60000)
  (total_hours : ℕ := 150)
  (first_period_hours : ℕ := 50)
  (first_period_words : ℕ := total_words / 2) :
  first_period_words / first_period_hours = 600 ∧ total_words / total_hours = 400 := 
by
  sorry

end average_words_per_hour_l259_259283


namespace minimum_candy_kinds_l259_259849

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
     It turned out that between any two candies of the same kind, there is an even number of candies.
     Prove that the minimum number of kinds of candies that could be is 46. -/
theorem minimum_candy_kinds (n : ℕ) (candies : ℕ → ℕ) 
  (h_candies_length : ∀ i, i < 91 → candies i < n)
  (h_even_between : ∀ i j, i < j → candies i = candies j → even (j - i - 1)) :
  n ≥ 46 :=
sorry

end minimum_candy_kinds_l259_259849


namespace triangle_largest_angle_and_type_l259_259937

theorem triangle_largest_angle_and_type
  (a b c : ℝ) 
  (h1 : a + b + c = 180)
  (h2 : a = 4 * k) 
  (h3 : b = 3 * k) 
  (h4 : c = 2 * k) 
  (h5 : a ≥ b) 
  (h6 : a ≥ c) : 
  a = 80 ∧ a < 90 ∧ b < 90 ∧ c < 90 := 
by
  -- Replace 'by' with 'sorry' to denote that the proof should go here
  sorry

end triangle_largest_angle_and_type_l259_259937


namespace ratio_of_bottles_l259_259883

theorem ratio_of_bottles
  (initial_money : ℤ)
  (initial_bottles : ℕ)
  (cost_per_bottle : ℤ)
  (cost_per_pound_cheese : ℤ)
  (cheese_pounds : ℚ)
  (remaining_money : ℤ) :
  initial_money = 100 →
  initial_bottles = 4 →
  cost_per_bottle = 2 →
  cost_per_pound_cheese = 10 →
  cheese_pounds = 0.5 →
  remaining_money = 71 →
  (2 * initial_bottles) / initial_bottles = 2 :=
by 
  sorry

end ratio_of_bottles_l259_259883


namespace not_perfect_power_l259_259916

theorem not_perfect_power (k : ℕ) (h : k ≥ 2) : ∀ m n : ℕ, m > 1 → n > 1 → 10^k - 1 ≠ m ^ n :=
by 
  sorry

end not_perfect_power_l259_259916


namespace count_inverses_modulo_11_l259_259552

theorem count_inverses_modulo_11 : (Finset.filter (λ a => Nat.gcd a 11 = 1) (Finset.range 11)).card = 10 :=
  by
  sorry

end count_inverses_modulo_11_l259_259552


namespace percentage_enclosed_by_pentagons_l259_259220

-- Define the condition for the large square and smaller squares.
def large_square_area (b : ℝ) : ℝ := (4 * b) ^ 2

-- Define the condition for the number of smaller squares forming pentagons.
def pentagon_small_squares : ℝ := 10

-- Define the total number of smaller squares within a large square.
def total_small_squares : ℝ := 16

-- Prove that the percentage of the plane enclosed by pentagons is 62.5%.
theorem percentage_enclosed_by_pentagons :
  (pentagon_small_squares / total_small_squares) * 100 = 62.5 :=
by 
  -- The proof is left as an exercise.
  sorry

end percentage_enclosed_by_pentagons_l259_259220


namespace total_books_is_10033_l259_259873

variable (P C B M H : ℕ)
variable (x : ℕ) (h_P : P = 3 * x) (h_C : C = 2 * x)
variable (h_B : B = (3 / 2) * x)
variable (h_M : M = (3 / 5) * x)
variable (h_H : H = (4 / 5) * x)
variable (total_books : ℕ)
variable (h_total : total_books = P + C + B + M + H)
variable (h_bound : total_books > 10000)

theorem total_books_is_10033 : total_books = 10033 :=
  sorry

end total_books_is_10033_l259_259873


namespace find_x_l259_259012

theorem find_x (x : ℝ) (h : 2^10 = 32^x) : x = 2 := 
by 
  {
    sorry
  }

end find_x_l259_259012


namespace binary_addition_l259_259786

theorem binary_addition :
  0b1101 + 0b101 + 0b1110 + 0b10111 + 0b11000 = 0b11100010 :=
by
  sorry

end binary_addition_l259_259786


namespace checkerboard_problem_l259_259616

def checkerboard_rectangles : ℕ := 2025
def checkerboard_squares : ℕ := 285

def relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

theorem checkerboard_problem :
  ∃ m n : ℕ, relatively_prime m n ∧ m + n = 154 ∧ (285 : ℚ) / 2025 = m / n :=
by {
  sorry
}

end checkerboard_problem_l259_259616


namespace number_of_ways_to_sum_to_4_l259_259813

-- Definitions deriving from conditions
def cards : List ℕ := [0, 1, 2, 3, 4]

-- Goal to prove
theorem number_of_ways_to_sum_to_4 : 
  let pairs := List.product cards cards
  let valid_pairs := pairs.filter (λ (x, y) => x + y = 4)
  List.length valid_pairs = 5 := 
by
  sorry

end number_of_ways_to_sum_to_4_l259_259813


namespace S_not_eq_T_l259_259508

def S := {x : ℤ | ∃ n : ℤ, x = 2 * n}
def T := {x : ℤ | ∃ k : ℤ, x = 4 * k + 1 ∨ x = 4 * k - 1}

theorem S_not_eq_T : S ≠ T := by
  sorry

end S_not_eq_T_l259_259508


namespace annual_decrease_rate_l259_259221

theorem annual_decrease_rate (r : ℝ) 
  (h1 : 15000 * (1 - r / 100)^2 = 9600) : 
  r = 20 := 
sorry

end annual_decrease_rate_l259_259221


namespace trigonometric_quadrant_l259_259986

theorem trigonometric_quadrant (α : ℝ) (h1 : Real.cos α < 0) (h2 : Real.sin α > 0) : 
  (π / 2 < α) ∧ (α < π) :=
by
  sorry

end trigonometric_quadrant_l259_259986


namespace number_of_inverses_mod_11_l259_259546

theorem number_of_inverses_mod_11 : 
  ∃ n, n = 10 ∧ ∀ x ∈ finset.range 11, (gcd x 11 = 1 → ∃ y, (x * y) % 11 = 1) :=
by
  sorry

end number_of_inverses_mod_11_l259_259546


namespace sum_in_base_8_is_correct_l259_259680

noncomputable section

open Nat

def num1 : ℕ := Nat.ofDigits 8 [5, 2, 7]
def num2 : ℕ := Nat.ofDigits 8 [1, 6, 5]
def num3 : ℕ := Nat.ofDigits 8 [2, 7, 3]
def sum_expected : ℕ := Nat.ofDigits 8 [1, 2, 0, 7]

theorem sum_in_base_8_is_correct :
  num1 + num2 + num3 = sum_expected := by
  sorry

end sum_in_base_8_is_correct_l259_259680


namespace avg_combined_is_2a_plus_3b_l259_259993

variables {x1 x2 x3 y1 y2 y3 a b : ℝ}

-- Given conditions
def avg_x_is_a (x1 x2 x3 a : ℝ) : Prop := (x1 + x2 + x3) / 3 = a
def avg_y_is_b (y1 y2 y3 b : ℝ) : Prop := (y1 + y2 + y3) / 3 = b

-- The statement to be proved
theorem avg_combined_is_2a_plus_3b
    (hx : avg_x_is_a x1 x2 x3 a) 
    (hy : avg_y_is_b y1 y2 y3 b) :
    ((2 * x1 + 3 * y1) + (2 * x2 + 3 * y2) + (2 * x3 + 3 * y3)) / 3 = 2 * a + 3 * b := 
by
  sorry

end avg_combined_is_2a_plus_3b_l259_259993


namespace right_triangle_and_inverse_l259_259660

theorem right_triangle_and_inverse :
  30 * 30 + 272 * 272 = 278 * 278 ∧ (∃ (n : ℕ), 0 ≤ n ∧ n < 4079 ∧ (550 * n) % 4079 = 1) :=
by
  sorry

end right_triangle_and_inverse_l259_259660


namespace total_pairs_of_shoes_tried_l259_259002

theorem total_pairs_of_shoes_tried (first_store_pairs second_store_additional third_store_pairs fourth_store_factor : ℕ) 
  (h_first : first_store_pairs = 7)
  (h_second : second_store_additional = 2)
  (h_third : third_store_pairs = 0)
  (h_fourth : fourth_store_factor = 2) :
  first_store_pairs + (first_store_pairs + second_store_additional) + third_store_pairs + 
    (fourth_store_factor * (first_store_pairs + (first_store_pairs + second_store_additional) + third_store_pairs)) = 48 := 
  by 
    sorry

end total_pairs_of_shoes_tried_l259_259002


namespace volume_of_box_l259_259800

theorem volume_of_box (l w h : ℝ) (hlw : l * w = 36) (hwh : w * h = 18) (hlh : l * h = 8) : 
    l * w * h = 72 := 
by 
    sorry

end volume_of_box_l259_259800


namespace range_of_a_l259_259370

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ((1 - a) * x > 1 - a) → (x < 1)) → (1 < a) :=
by sorry

end range_of_a_l259_259370


namespace find_x_l259_259010

theorem find_x (x : ℝ) (h : 2^10 = 32^x) : x = 2 :=
by
  have h1 : 32 = 2^5 := by norm_num
  rw [h1, pow_mul] at h
  have h2 : 2^(10) = 2^(5*x) := by exact h
  have h3 : 10 = 5 * x := by exact (pow_inj h2).2
  linarith

end find_x_l259_259010


namespace inequality_5positives_l259_259730

variable {x1 x2 x3 x4 x5 : ℝ}

theorem inequality_5positives (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) :
  (x1 + x2 + x3 + x4 + x5)^2 ≥ 4 * (x1 * x2 + x2 * x3 + x3 * x4 + x4 * x5 + x5 * x1) :=
by
  sorry

end inequality_5positives_l259_259730


namespace book_costs_l259_259248

theorem book_costs (C1 C2 : ℝ) (h1 : C1 + C2 = 450) (h2 : 0.85 * C1 = 1.19 * C2) : C1 = 262.5 := 
sorry

end book_costs_l259_259248


namespace four_student_committees_from_six_l259_259985

theorem four_student_committees_from_six : (Nat.choose 6 4) = 15 := 
by 
  sorry

end four_student_committees_from_six_l259_259985


namespace cylinder_height_relationship_l259_259237

theorem cylinder_height_relationship
  (r1 h1 r2 h2 : ℝ)
  (volume_equal : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_relationship : r2 = 1.2 * r1) :
  h1 = 1.44 * h2 :=
by sorry

end cylinder_height_relationship_l259_259237


namespace compare_abc_l259_259045

noncomputable theory

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l259_259045


namespace find_greatest_divisor_l259_259439

def greatest_divisor_leaving_remainders (n₁ n₁_r n₂ n₂_r d : ℕ) : Prop :=
  (n₁ % d = n₁_r) ∧ (n₂ % d = n₂_r) 

theorem find_greatest_divisor :
  greatest_divisor_leaving_remainders 1657 10 2037 7 1 :=
by
  sorry

end find_greatest_divisor_l259_259439


namespace irreducible_fractions_product_one_l259_259486

theorem irreducible_fractions_product_one : ∃ (a b c d e f : ℕ),
  {a, b, c, d, e, f}.Subset {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  {a, b, c, d, e, f}.card = 6 ∧
  ∃ (f1_num f1_den f2_num f2_den f3_num f3_den : ℕ), 
    (f1_num ≠ f1_den ∧ coprime f1_num f1_den ∧ f1_num ∈ {a, b, c, d, e, f} ∧ f1_den ∈ {a, b, c, d, e, f} ∧ 
    f2_num ≠ f2_den ∧ coprime f2_num f2_den ∧ f2_num ∈ {a, b, c, d, e, f} ∧ f2_den ∈ {a, b, c, d, e, f} ∧ 
    f3_num ≠ f3_den ∧ coprime f3_num f3_den ∧ f3_num ∈ {a, b, c, d, e, f} ∧ f3_den ∈ {a, b, c, d, e, f} ∧ 
    (f1_num * f2_num * f3_num) = (f1_den * f2_den * f3_den)) :=
sorry

end irreducible_fractions_product_one_l259_259486


namespace sean_days_played_is_14_l259_259404

def total_minutes_played : Nat := 1512
def indira_minutes_played : Nat := 812
def sean_minutes_per_day : Nat := 50
def sean_total_minutes : Nat := total_minutes_played - indira_minutes_played
def sean_days_played : Nat := sean_total_minutes / sean_minutes_per_day

theorem sean_days_played_is_14 : sean_days_played = 14 :=
by
  sorry

end sean_days_played_is_14_l259_259404


namespace determine_set_A_l259_259522

variable (U : Set ℕ) (A : Set ℕ)

theorem determine_set_A (hU : U = {0, 1, 2, 3}) (hcompl : U \ A = {2}) :
  A = {0, 1, 3} :=
by
  sorry

end determine_set_A_l259_259522


namespace exist_irreducible_fractions_prod_one_l259_259492

theorem exist_irreducible_fractions_prod_one (S : List ℚ) :
  (∀ x ∈ S, ∃ (n d : ℤ), n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ x = (n /. d) ∧ Int.gcd n d = 1) ∧
  (∀ i j, i ≠ j → (S.get i).num ≠ (S.get j).num ∧ (S.get i).den ≠ (S.get j).den) →
  S.length = 3 ∧ S.prod = 1 :=
begin
  sorry
end

end exist_irreducible_fractions_prod_one_l259_259492


namespace final_amount_simple_interest_l259_259281

theorem final_amount_simple_interest (P R T : ℕ) (hP : P = 12500) (hR : R = 6) (hT : T = 4) : 
  P + (P * R * T) / 100 = 13250 :=
by
  rw [hP, hR, hT]
  norm_num
  sorry

end final_amount_simple_interest_l259_259281


namespace day_of_week_after_6_pow_2023_l259_259758

def day_of_week_after_days (start_day : ℕ) (days : ℕ) : ℕ :=
  (start_day + days) % 7

theorem day_of_week_after_6_pow_2023 :
  day_of_week_after_days 4 (6^2023) = 3 :=
by
  sorry

end day_of_week_after_6_pow_2023_l259_259758


namespace max_segment_perimeter_l259_259924

def isosceles_triangle (base height : ℝ) := true -- A realistic definition can define properties of an isosceles triangle

def equal_area_segments (triangle : isosceles_triangle 10 12) (n : ℕ) := true -- A realist definition can define cutting into equal area segments

noncomputable def perimeter_segment (base height : ℝ) (k : ℕ) (n : ℕ) : ℝ :=
  1 + Real.sqrt (height^2 + (base / n * k)^2) + Real.sqrt (height^2 + (base / n * (k + 1))^2)

theorem max_segment_perimeter (base height : ℝ) (n : ℕ) (h_base : base = 10) (h_height : height = 12) (h_segments : n = 10) :
  ∃ k, k ∈ Finset.range n ∧ perimeter_segment base height k n = 31.62 :=
by
  sorry

end max_segment_perimeter_l259_259924


namespace relation_among_abc_l259_259043

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem relation_among_abc : a > c ∧ c > b := 
by {
    -- proof steps go here, but we use sorry for now
    sorry
}

end relation_among_abc_l259_259043


namespace rectangle_width_l259_259225

theorem rectangle_width (w : ℝ)
    (h₁ : 5 > 0) (h₂ : 6 > 0) (h₃ : 3 > 0) 
    (area_relation : w * 5 = 3 * 6 + 2) : w = 4 :=
by
  sorry

end rectangle_width_l259_259225


namespace cannot_determine_right_triangle_l259_259127

theorem cannot_determine_right_triangle (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = 180) →
  (\<^angle A = B + C) →
  (5 * a = 13 * c ∧ 12 * b = 13 * c) →
  (a^2 = (b+c) * (b-c)) →
  (\<^angle A = 3 * x ∧ \<^angle B = 4 * x ∧ \<^angle C = 5 * x) →
  (12 * x = 180) →
  (x ≠ 15 → ∃ (A B C : ℝ), A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90) :=
by sorry

end cannot_determine_right_triangle_l259_259127


namespace kids_stayed_home_l259_259674

open Nat

theorem kids_stayed_home (kids_camp : ℕ) (additional_kids_home : ℕ) (total_kids_home : ℕ) 
  (h1 : kids_camp = 202958) 
  (h2 : additional_kids_home = 574664) 
  (h3 : total_kids_home = kids_camp + additional_kids_home) : 
  total_kids_home = 777622 := 
by 
  rw [h1, h2] at h3
  exact h3

end kids_stayed_home_l259_259674


namespace percentage_of_girls_with_dogs_l259_259641

theorem percentage_of_girls_with_dogs (students total_students : ℕ)
(h_total_students : total_students = 100)
(girls boys : ℕ)
(h_half_students : girls = total_students / 2 ∧ boys = total_students / 2)
(boys_with_dogs : ℕ)
(h_boys_with_dogs : boys_with_dogs = boys / 10)
(total_with_dogs : ℕ)
(h_total_with_dogs : total_with_dogs = 15)
(girls_with_dogs : ℕ)
(h_girls_with_dogs : girls_with_dogs = total_with_dogs - boys_with_dogs)
: (girls_with_dogs * 100 / girls = 20) :=
by
  sorry

end percentage_of_girls_with_dogs_l259_259641


namespace MissAisha_height_l259_259720

theorem MissAisha_height (H : ℝ)
  (legs_length : ℝ := H / 3)
  (head_length : ℝ := H / 4)
  (rest_body_length : ℝ := 25) :
  H = 60 :=
by sorry

end MissAisha_height_l259_259720


namespace chicago_denver_temperature_l259_259471

def temperature_problem (C D : ℝ) (N : ℝ) : Prop :=
  (C = D - N) ∧ (abs ((D - N + 4) - (D - 2)) = 1)

theorem chicago_denver_temperature (C D N : ℝ) (h : temperature_problem C D N) :
  N = 5 ∨ N = 7 → (5 * 7 = 35) :=
by sorry

end chicago_denver_temperature_l259_259471


namespace exist_irreducible_fractions_prod_one_l259_259495

theorem exist_irreducible_fractions_prod_one (S : List ℚ) :
  (∀ x ∈ S, ∃ (n d : ℤ), n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ x = (n /. d) ∧ Int.gcd n d = 1) ∧
  (∀ i j, i ≠ j → (S.get i).num ≠ (S.get j).num ∧ (S.get i).den ≠ (S.get j).den) →
  S.length = 3 ∧ S.prod = 1 :=
begin
  sorry
end

end exist_irreducible_fractions_prod_one_l259_259495


namespace part1_increasing_part2_range_a_l259_259167

noncomputable def f (x a : ℝ) : ℝ := exp x - (1/2) * (x + a)^2

theorem part1_increasing (a : ℝ) (h1 : (exp 0 - (1/2) * (0 + a)^2) = 1) :
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 a < f x2 a :=
sorry

theorem part2_range_a (a : ℝ) : (∀ x : ℝ, x ≥ 0 → f x a ≥ 0) ↔ (-sqrt 2 ≤ a ∧ a ≤ 2 - log 2) :=
sorry

end part1_increasing_part2_range_a_l259_259167


namespace total_pairs_of_shoes_tried_l259_259001

theorem total_pairs_of_shoes_tried (first_store_pairs second_store_additional third_store_pairs fourth_store_factor : ℕ) 
  (h_first : first_store_pairs = 7)
  (h_second : second_store_additional = 2)
  (h_third : third_store_pairs = 0)
  (h_fourth : fourth_store_factor = 2) :
  first_store_pairs + (first_store_pairs + second_store_additional) + third_store_pairs + 
    (fourth_store_factor * (first_store_pairs + (first_store_pairs + second_store_additional) + third_store_pairs)) = 48 := 
  by 
    sorry

end total_pairs_of_shoes_tried_l259_259001


namespace range_of_a_l259_259514

variable {α : Type*}

def in_interval (x : ℝ) (a b : ℝ) : Prop := a < x ∧ x < b

def A (a : ℝ) : Set ℝ := {-1, 0, a}

def B : Set ℝ := {x : ℝ | in_interval x 0 1}

theorem range_of_a (a : ℝ) (hA_B_nonempty : (A a ∩ B).Nonempty) : 0 < a ∧ a < 1 := 
sorry

end range_of_a_l259_259514


namespace find_x_eq_2_l259_259015

theorem find_x_eq_2 : ∀ x : ℝ, 2^10 = 32^x → x = 2 := 
by 
  intros x h
  sorry

end find_x_eq_2_l259_259015


namespace base7_to_base10_conversion_l259_259665

theorem base7_to_base10_conversion :
  let base7_246 := 2 * 7^2 + 4 * 7^1 + 6 * 7^0
  in base7_246 = 132 :=
by
  let base7_246 := 2 * 7^2 + 4 * 7^1 + 6 * 7^0
  show base7_246 = 132
  -- The proof steps would go here
  sorry

end base7_to_base10_conversion_l259_259665


namespace basketball_lineup_count_l259_259770

theorem basketball_lineup_count :
  (∃ (players : Finset ℕ), players.card = 15) → 
  ∃ centers power_forwards small_forwards shooting_guards point_guards sixth_men : ℕ,
  ∃ b : Fin (15) → Fin (15),
  15 * 14 * 13 * 12 * 11 * 10 = 360360 
:= by sorry

end basketball_lineup_count_l259_259770


namespace sum_of_squares_l259_259253

theorem sum_of_squares :
  1000^2 + 1001^2 + 1002^2 + 1003^2 + 1004^2 = 5020030 :=
by
  sorry

end sum_of_squares_l259_259253


namespace find_m_l259_259161

variable {a_n : ℕ → ℤ}
variable {S : ℕ → ℤ}

def isArithmeticSeq (a_n : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a_n (n + 1) = a_n n + d

def sumSeq (S : ℕ → ℤ) (a_n : ℕ → ℤ) : Prop :=
∀ n, S n = (n * (a_n 1 + a_n n)) / 2

theorem find_m
  (d : ℤ)
  (a_1 : ℤ)
  (a_n : ∀ n, ℤ)
  (S : ℕ → ℤ)
  (h_arith : isArithmeticSeq a_n d)
  (h_sum : sumSeq S a_n)
  (h1 : S (m - 1) = -2)
  (h2 : S m = 0)
  (h3 : S (m + 1) = 3) :
  m = 5 :=
sorry

end find_m_l259_259161


namespace range_of_sum_l259_259032

theorem range_of_sum (a b : ℝ) (h : a^2 - a * b + b^2 = a + b) :
  0 ≤ a + b ∧ a + b ≤ 4 :=
by
  sorry

end range_of_sum_l259_259032


namespace safe_zone_inequality_l259_259500

theorem safe_zone_inequality (x : ℝ) (fuse_burn_rate : ℝ) (run_speed : ℝ) (safe_zone_dist : ℝ) (H1: fuse_burn_rate = 0.5) (H2: run_speed = 4) (H3: safe_zone_dist = 150) :
  run_speed * (x / fuse_burn_rate) ≥ safe_zone_dist :=
sorry

end safe_zone_inequality_l259_259500


namespace avg_weight_of_13_children_l259_259744

-- Definitions based on conditions:
def boys_avg_weight := 160
def boys_count := 8
def girls_avg_weight := 130
def girls_count := 5

-- Calculation to determine the total weights
def boys_total_weight := boys_avg_weight * boys_count
def girls_total_weight := girls_avg_weight * girls_count

-- Combined total weight
def total_weight := boys_total_weight + girls_total_weight

-- Average weight calculation
def children_count := boys_count + girls_count
def avg_weight := total_weight / children_count

-- The theorem to prove:
theorem avg_weight_of_13_children : avg_weight = 148 := by
  sorry

end avg_weight_of_13_children_l259_259744


namespace jacks_remaining_capacity_l259_259887

noncomputable def jacks_basket_full_capacity : ℕ := 12
noncomputable def jills_basket_full_capacity : ℕ := 2 * jacks_basket_full_capacity
noncomputable def jacks_current_apples (x : ℕ) : Prop := 3 * x = jills_basket_full_capacity

theorem jacks_remaining_capacity {x : ℕ} (hx : jacks_current_apples x) :
  jacks_basket_full_capacity - x = 4 :=
by sorry

end jacks_remaining_capacity_l259_259887


namespace count_inverses_mod_11_l259_259543

theorem count_inverses_mod_11 : 
  (Finset.filter (λ x : ℕ, Nat.coprime x 11) (Finset.range 11)).card = 10 := 
by
  sorry

end count_inverses_mod_11_l259_259543


namespace odds_burning_out_during_second_period_l259_259984

def odds_burning_out_during_first_period := 1 / 3
def odds_not_burning_out_first_period := 1 - odds_burning_out_during_first_period
def odds_not_burning_out_next_period := odds_not_burning_out_first_period / 2

theorem odds_burning_out_during_second_period :
  (1 - odds_not_burning_out_next_period) = 2 / 3 := by
  sorry

end odds_burning_out_during_second_period_l259_259984


namespace largest_of_5_consecutive_odd_integers_l259_259503

theorem largest_of_5_consecutive_odd_integers (n : ℤ) (h : n + (n + 2) + (n + 4) + (n + 6) + (n + 8) = 235) :
  n + 8 = 51 :=
sorry

end largest_of_5_consecutive_odd_integers_l259_259503


namespace Chrysler_Building_floors_l259_259081

variable (C L : ℕ)

theorem Chrysler_Building_floors :
  (C = L + 11) → (C + L = 35) → (C = 23) :=
by
  intro h1 h2
  sorry

end Chrysler_Building_floors_l259_259081


namespace probability_red_odd_blue_special_l259_259233

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def successful_outcomes_red := {1, 3, 5, 7}
def successful_outcomes_blue := {1, 2, 3, 4, 5, 7, 9}

noncomputable def probability (successful : Nat) (total : Nat) : ℚ :=
  (successful : ℚ) / (total : ℚ)

theorem probability_red_odd_blue_special :
  probability (successful_outcomes_red.card * successful_outcomes_blue.card) (8 * 10) = 7 / 20 := by
  sorry

end probability_red_odd_blue_special_l259_259233


namespace find_b_l259_259425

noncomputable def a (c : ℚ) : ℚ := 10 * c - 10
noncomputable def b (c : ℚ) : ℚ := 10 * c + 10
noncomputable def c_val := (200 : ℚ) / 21

theorem find_b : 
  let a := a c_val
  let b := b c_val
  let c := c_val
  a + b + c = 200 ∧ 
  a + 10 = b - 10 ∧ 
  a + 10 = 10 * c → 
  b = 2210 / 21 :=
by
  intros
  sorry

end find_b_l259_259425


namespace area_triangle_tangent_circles_l259_259773

theorem area_triangle_tangent_circles :
  ∃ (A B C : Type) (radius1 radius2 : ℝ) 
    (tangent1 tangent2 : ℝ → ℝ → Prop)
    (congruent_sides : ℝ → Prop),
    radius1 = 1 ∧ radius2 = 2 ∧
    (∀ x y, tangent1 x y) ∧ (∀ x y, tangent2 x y) ∧
    congruent_sides 1 ∧ congruent_sides 2 ∧
    ∃ (area : ℝ), area = 16 * Real.sqrt 2 :=
by
  -- This is where the proof would be written
  sorry

end area_triangle_tangent_circles_l259_259773


namespace find_x_in_average_l259_259211

theorem find_x_in_average (x : ℝ) :
  (201 + 202 + 204 + 205 + 206 + 209 + 209 + 210 + x) / 9 = 207 → x = 217 :=
by
  intro h
  sorry

end find_x_in_average_l259_259211


namespace sin_sum_leq_3div2_sqrt3_l259_259843

theorem sin_sum_leq_3div2_sqrt3 (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π) :
  Real.sin A + Real.sin B + Real.sin C ≤ (3 / 2) * Real.sqrt 3 :=
by
  sorry

end sin_sum_leq_3div2_sqrt3_l259_259843


namespace symmetry_about_origin_l259_259747

noncomputable def f (x : ℝ) : ℝ := x * Real.log (-x)
noncomputable def g (x : ℝ) : ℝ := x * Real.log x

theorem symmetry_about_origin :
  ∀ x : ℝ, f (-x) = -g (-x) :=
by
  sorry

end symmetry_about_origin_l259_259747


namespace circle_radius_l259_259955

theorem circle_radius (r : ℝ) (x y : ℝ) :
  x = π * r^2 ∧ y = 2 * π * r ∧ x + y = 100 * π → r = 10 := 
  by
  sorry

end circle_radius_l259_259955


namespace inequality_maintained_l259_259519

noncomputable def g (x a : ℝ) := x^2 + Real.log (x + a)

theorem inequality_maintained (x1 x2 a : ℝ) (hx1 : x1 = (-a + Real.sqrt (a^2 - 2))/2)
  (hx2 : x2 = (-a - Real.sqrt (a^2 - 2))/2):
  (a > Real.sqrt 2) → 
  (g x1 a + g x2 a) / 2 > g ((x1 + x2 ) / 2) a :=
by
  sorry

end inequality_maintained_l259_259519


namespace calc_x_l259_259656

theorem calc_x : 484 + 2 * 22 * 7 + 49 = 841 := by
  sorry

end calc_x_l259_259656


namespace prob_three_even_dice_l259_259135

noncomputable def fairTwelveSidedDie := ⟨Set.range (λ n, n + 1), sorry⟩

def isEven (n : ℕ) : Prop := n % 2 = 0

theorem prob_three_even_dice (n : ℕ) (hn : n = 6) (m : ℕ) (hm : m = 12) :
  let p_even : ℝ := 1 / 2 in
  let probability :=
    (Nat.choose 6 3) * (p_even ^ 3) * ((1 - p_even) ^ 3) in
  probability = 5 / 16 :=
by
  have h : @Finset.card ℕ _ (Finset.filter isEven (Finset.range (m))) = m / 2 := sorry
  have p_even_def : p_even = 1 / 2 := rfl
  have hc : Nat.choose 6 3 = 20 := sorry
  have h_power : (p_even ^ 3) = (1 / 2) ^ 3 := by rw [p_even_def]
  have h_power2 : ((1 - p_even) ^ 3) = (1 / 2) ^ 3 := sorry
  let probability := 20 * (1 / 64)
  show probability = 5 / 16
  sorry

end prob_three_even_dice_l259_259135


namespace find_number_l259_259245

theorem find_number (x : ℤ) (h : 7 * x + 37 = 100) : x = 9 :=
by
  sorry

end find_number_l259_259245


namespace movie_ticket_cost_l259_259067

variable (x : ℝ)
variable (h1 : x * 2 + 1.59 + 13.95 = 36.78)

theorem movie_ticket_cost : x = 10.62 :=
by
  sorry

end movie_ticket_cost_l259_259067


namespace horner_method_v1_l259_259795

def polynomial (x : ℝ) : ℝ := 4 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

theorem horner_method_v1 (x : ℝ) (h : x = 5) : 
  ((4 * x + 2) * x + 3.5) = 22 := by
  rw [h]
  norm_num
  sorry

end horner_method_v1_l259_259795


namespace percentage_of_boys_l259_259844

def ratio_boys_girls := 2 / 3
def ratio_teacher_students := 1 / 6
def total_people := 36

theorem percentage_of_boys : ∃ (n_student n_teacher n_boys n_girls : ℕ), 
  n_student + n_teacher = 35 ∧
  n_student * (1 + 1/6) = total_people ∧
  n_boys / n_student = ratio_boys_girls ∧
  n_teacher / n_student = ratio_teacher_students ∧
  ((n_boys : ℚ) / total_people) * 100 = 400 / 7 :=
sorry

end percentage_of_boys_l259_259844


namespace arithmetic_result_l259_259806

theorem arithmetic_result :
  1325 + (572 / 52) - 225 + (2^3) = 1119 :=
by
  sorry

end arithmetic_result_l259_259806


namespace tommy_paint_cost_l259_259227

theorem tommy_paint_cost :
  ∀ (width height : ℕ) (cost_per_quart coverage_per_quart : ℕ),
    width = 5 →
    height = 4 →
    cost_per_quart = 2 →
    coverage_per_quart = 4 →
    2 * width * height / coverage_per_quart * cost_per_quart = 20 :=
by
  intros width height cost_per_quart coverage_per_quart
  intros hwidth hheight hcost hcoverage
  rw [hwidth, hheight, hcost, hcoverage]
  simp
  sorry

end tommy_paint_cost_l259_259227


namespace eval_expr_l259_259502

variable {x y : ℝ}

theorem eval_expr (h : x ≠ 0 ∧ y ≠ 0) :
  ((x^4 + 1) / x^2) * ((y^4 + 1) / y^2) - ((x^4 - 1) / y^2) * ((y^4 - 1) / x^2) = (2 * x^2) / (y^2) + (2 * y^2) / (x^2) := by
  sorry

end eval_expr_l259_259502


namespace intersection_A_B_l259_259907

-- Define sets A and B based on given conditions
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | 1 < x ∧ x < 4}

-- Prove the intersection of A and B equals (2,4)
theorem intersection_A_B : A ∩ B = {x | 2 < x ∧ x < 4} := 
by
  sorry

end intersection_A_B_l259_259907


namespace continuous_function_triples_l259_259808

theorem continuous_function_triples (f g h : ℝ → ℝ) (h₁ : Continuous f) (h₂ : Continuous g) (h₃ : Continuous h)
  (h₄ : ∀ x y : ℝ, f (x + y) = g x + h y) :
  ∃ (c a b : ℝ), (∀ x : ℝ, f x = c * x + a + b) ∧ (∀ x : ℝ, g x = c * x + a) ∧ (∀ x : ℝ, h x = c * x + b) :=
sorry

end continuous_function_triples_l259_259808


namespace cleaned_area_correct_l259_259910

def lizzie_cleaned : ℚ := 3534 + 2/3
def hilltown_team_cleaned : ℚ := 4675 + 5/8
def green_valley_cleaned : ℚ := 2847 + 7/9
def riverbank_cleaned : ℚ := 6301 + 1/3
def meadowlane_cleaned : ℚ := 3467 + 4/5

def total_cleaned : ℚ := lizzie_cleaned + hilltown_team_cleaned + green_valley_cleaned + riverbank_cleaned + meadowlane_cleaned
def total_farmland : ℚ := 28500

def remaining_area_to_clean : ℚ := total_farmland - total_cleaned

theorem cleaned_area_correct : remaining_area_to_clean = 7672.7964 :=
by
  sorry

end cleaned_area_correct_l259_259910


namespace find_intersection_l259_259998

noncomputable def setM : Set ℝ := {x : ℝ | x^2 ≤ 9}
noncomputable def setN : Set ℝ := {x : ℝ | x ≤ 1}
noncomputable def intersection : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 1}

theorem find_intersection (x : ℝ) : (x ∈ setM ∧ x ∈ setN) ↔ (x ∈ intersection) := 
by sorry

end find_intersection_l259_259998


namespace eight_lines_no_parallel_no_concurrent_l259_259145

-- Define the number of regions into which n lines divide the plane
def regions (n : ℕ) : ℕ :=
if n = 0 then 1
else if n = 1 then 2
else n * (n - 1) / 2 + n + 1

theorem eight_lines_no_parallel_no_concurrent :
  regions 8 = 37 :=
by
  sorry

end eight_lines_no_parallel_no_concurrent_l259_259145


namespace base7_to_base10_conversion_l259_259669

theorem base7_to_base10_conversion (n : ℤ) (h : n = 2 * 7^2 + 4 * 7^1 + 6 * 7^0) : n = 132 := by
  sorry

end base7_to_base10_conversion_l259_259669


namespace jill_trips_to_fill_tank_l259_259880

def tank_capacity : ℕ := 600
def bucket_volume : ℕ := 5
def jack_buckets_per_trip : ℕ := 2
def jill_buckets_per_trip : ℕ := 1
def jack_to_jill_trip_ratio : ℕ := 3 / 2

theorem jill_trips_to_fill_tank : (tank_capacity / bucket_volume) = 120 → 
                                   ((jack_to_jill_trip_ratio * jack_buckets_per_trip) + 2 * jill_buckets_per_trip) = 8 →
                                   15 * 2 = 30 :=
by
  intros h1 h2
  sorry

end jill_trips_to_fill_tank_l259_259880


namespace range_of_a_l259_259835

noncomputable def a_n (n : ℕ) (a : ℝ) : ℝ :=
  (-1)^(n + 2018) * a

noncomputable def b_n (n : ℕ) : ℝ :=
  2 + (-1)^(n + 2019) / n

theorem range_of_a (a : ℝ) :
  (∀ n : ℕ, 1 ≤ n → a_n n a < b_n n) ↔ -2 ≤ a ∧ a < 3 / 2 :=
  sorry

end range_of_a_l259_259835


namespace number_of_ways_to_select_co_leaders_l259_259963

theorem number_of_ways_to_select_co_leaders (n k : ℕ) (hn : n = 20) (hk : k = 2) :
  (nat.choose n k) = 190 :=
by {
  rw [hn, hk],
  exact nat.choose_eq 20 2,
  sorry
}

end number_of_ways_to_select_co_leaders_l259_259963


namespace range_of_a_l259_259367

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-4 : ℝ) a, (x^2 - 4 * x) ∈ Set.Icc (-4 : ℝ) 32) →
  2 ≤ a ∧ a ≤ 8 :=
sorry

end range_of_a_l259_259367


namespace min_value_of_a3b2c_l259_259902

theorem min_value_of_a3b2c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 1 / a + 1 / b + 1 / c = 9) : 
  a^3 * b^2 * c ≥ 1 / 2916 :=
by 
  sorry

end min_value_of_a3b2c_l259_259902


namespace relation_among_abc_l259_259044

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem relation_among_abc : a > c ∧ c > b := 
by {
    -- proof steps go here, but we use sorry for now
    sorry
}

end relation_among_abc_l259_259044


namespace min_kinds_of_candies_l259_259869

theorem min_kinds_of_candies (candies : ℕ) (even_distance_candies : ∀ i j : ℕ, i ≠ j → i < candies → j < candies → is_even (j - i - 1)) :
  candies = 91 → 46 ≤ candies :=
by
  assume h1 : candies = 91
  sorry

end min_kinds_of_candies_l259_259869


namespace possible_m_value_l259_259997

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^3 - (1/2)*x - 1
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := x^3 + m / x

theorem possible_m_value :
  ∃ m : ℝ, (m = (1/2) - (1/Real.exp 1)) ∧
    (∀ x1 x2 : ℝ, 
      (f x1 = g (-x1) m) →
      (f x2 = g (-x2) m) →
      x1 ≠ 0 ∧ x2 ≠ 0 ∧
      m = x1 * Real.exp x1 - (1/2) * x1^2 - x1 ∧
      m = x2 * Real.exp x2 - (1/2) * x2^2 - x2) :=
by
  sorry

end possible_m_value_l259_259997


namespace tangent_to_circumcircle_of_CSB_l259_259394

open EuclideanGeometry

variables {A B C D S : Point}
variables {Γ : Circle}

-- Definitions based on the problem conditions
def is_parallelogram (A B C D : Point) : Prop :=
∃ P Q, parallelogram A B C D

def is_circumcircle (Γ : Circle) (A B D : Point) : Prop :=
oncircle Γ A ∧ oncircle Γ B ∧ oncircle Γ D

def second_intersects (AC : Line) (Γ : Circle) (S : Point) : Prop :=
∃ E, E ≠ S ∧ onCircle Γ S ∧ onLine AC S ∧ intersectsLine AC Γ

-- The problem to prove
theorem tangent_to_circumcircle_of_CSB
  (parallelogram_ABCD : is_parallelogram A B C D)
  (circumcircle_ABD : is_circumcircle Γ A B D)
  (second_intersection : second_intersects (line_through A C) Γ S) :
  tangent (circumcircle ⟨point B, S, C⟩) (line_through D B) :=
sorry

end tangent_to_circumcircle_of_CSB_l259_259394


namespace area_two_layers_l259_259944

-- Given conditions
variables (A_total A_covered A_three_layers : ℕ)

-- Conditions from the problem
def condition_1 : Prop := A_total = 204
def condition_2 : Prop := A_covered = 140
def condition_3 : Prop := A_three_layers = 20

-- Mathematical equivalent proof problem
theorem area_two_layers (A_total A_covered A_three_layers : ℕ) 
  (h1 : condition_1 A_total) 
  (h2 : condition_2 A_covered) 
  (h3 : condition_3 A_three_layers) : 
  ∃ A_two_layers : ℕ, A_two_layers = 24 :=
by sorry

end area_two_layers_l259_259944


namespace find_base_b_l259_259366

theorem find_base_b : ∃ b : ℕ, (3 * b + 5) ^ 2 = b ^ 3 + 3 * b ^ 2 + 3 * b + 1 ∧ b = 7 := 
by {
  sorry
}

end find_base_b_l259_259366


namespace minimum_candy_kinds_l259_259858

theorem minimum_candy_kinds (candy_count : ℕ) (h1 : candy_count = 91)
  (h2 : ∀ (k : ℕ), k ∈ (1 : ℕ) → (λ i j, abs (i - j) % 2 = 0))
  : ∃ (kinds : ℕ), kinds = 46 :=
by
  sorry

end minimum_candy_kinds_l259_259858


namespace attendees_receive_all_items_l259_259682

theorem attendees_receive_all_items :
  let capacity := 5000
  let every_poster := 100
  let every_program := 45
  let every_drink := 60
  let lcm := Nat.lcm (Nat.lcm every_poster every_program) every_drink
  capacity / lcm = 5 :=
by
  let capacity := 5000
  let every_poster := 100
  let every_program := 45
  let every_drink := 60
  let lcm := Nat.lcm (Nat.lcm every_poster every_program) every_drink
  have h : Nat.lcm 100 45 = 900 := Nat.lcm_eq 100 45 (by norm_num) (by norm_num)
  have h2 : Nat.lcm 900 60 = 900 := Nat.lcm_eq 900 60 (by norm_num) (by norm_num)
  have h3 : capacity / lcm = 5 := by norm_num
  exact h3

end attendees_receive_all_items_l259_259682


namespace final_balance_l259_259599

noncomputable def initial_balance : ℕ := 10
noncomputable def charity_donation : ℕ := 4
noncomputable def prize_amount : ℕ := 90
noncomputable def lost_at_first_slot : ℕ := 50
noncomputable def lost_at_second_slot : ℕ := 10
noncomputable def lost_at_last_slot : ℕ := 5
noncomputable def cost_of_water : ℕ := 1
noncomputable def cost_of_lottery_ticket : ℕ := 1
noncomputable def lottery_win : ℕ := 65

theorem final_balance : 
  initial_balance - charity_donation + prize_amount - (lost_at_first_slot + lost_at_second_slot + lost_at_last_slot) - (cost_of_water + cost_of_lottery_ticket) + lottery_win = 94 := 
by 
  -- This is the lean statement, the proof is not required as per instructions.
  sorry

end final_balance_l259_259599


namespace calculate_A_share_l259_259622

variable (x : ℝ) (total_gain : ℝ)
variable (h_b_invests : 2 * x)  -- B invests double the amount after 6 months
variable (h_c_invests : 3 * x)  -- C invests thrice the amount after 8 months

/-- Calculate the share of A from the total annual gain -/
theorem calculate_A_share (h_total_gain : total_gain = 18600) :
  let a_investmentMonths := x * 12
  let b_investmentMonths := (2 * x) * 6
  let c_investmentMonths := (3 * x) * 4
  let total_investmentMonths := a_investmentMonths + b_investmentMonths + c_investmentMonths
  let a_share := (a_investmentMonths / total_investmentMonths) * total_gain
  a_share = 6200 :=
by
  sorry

end calculate_A_share_l259_259622


namespace maximum_ab_is_40_l259_259390

noncomputable def maximum_ab (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : 5 * a + 8 * b = 80) : ℝ :=
  max (a * b) 40

theorem maximum_ab_is_40 {a b : ℝ} (h₀ : a > 0) (h₁ : b > 0) (h₂ : 5 * a + 8 * b = 80) : maximum_ab a b h₀ h₁ h₂ = 40 := 
by 
  sorry

end maximum_ab_is_40_l259_259390


namespace field_ratio_l259_259417

theorem field_ratio (l w : ℕ) (h_l : l = 20) (pond_side : ℕ) (h_pond_side : pond_side = 5)
  (h_area_pond : pond_side * pond_side = (1 / 8 : ℚ) * l * w) : l / w = 2 :=
by 
  sorry

end field_ratio_l259_259417


namespace moles_of_Cl2_required_l259_259810

theorem moles_of_Cl2_required (n_C2H6 n_HCl : ℕ) (balance : n_C2H6 = 3) (HCl_needed : n_HCl = 6) :
  ∃ n_Cl2 : ℕ, n_Cl2 = 9 :=
by
  sorry

end moles_of_Cl2_required_l259_259810


namespace salary_of_b_l259_259449

theorem salary_of_b (S_A S_B : ℝ)
  (h1 : S_A + S_B = 14000)
  (h2 : 0.20 * S_A = 0.15 * S_B) :
  S_B = 8000 :=
by
  sorry

end salary_of_b_l259_259449


namespace find_b_minus_d_squared_l259_259175

theorem find_b_minus_d_squared (a b c d : ℝ)
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 3) :
  (b - d) ^ 2 = 25 :=
sorry

end find_b_minus_d_squared_l259_259175


namespace magician_earning_l259_259635

-- Definitions based on conditions
def price_per_deck : ℕ := 2
def initial_decks : ℕ := 5
def remaining_decks : ℕ := 3

-- Theorem statement
theorem magician_earning :
  let sold_decks := initial_decks - remaining_decks
  let earning := sold_decks * price_per_deck
  earning = 4 := by
  sorry

end magician_earning_l259_259635


namespace probability_of_diff_families_is_correct_l259_259208

open Finset

noncomputable def probability_diff_families : ℚ :=
  let total_people : ℕ := 18
  let num_people_per_family : ℕ := 3
  let num_families : ℕ := 6
  let total_ways_choose_3 : ℕ := (total_people.choose 3)
  let ways_choose_3_families : ℕ := (num_families.choose 3)
  let ways_choose_1_person_per_family : ℕ := num_people_per_family ^ 3
  let favorable_outcomes : ℕ := ways_choose_3_families * ways_choose_1_person_per_family
  (favorable_outcomes : ℚ) / (total_ways_choose_3 : ℚ)

theorem probability_of_diff_families_is_correct :
  probability_diff_families = 45 / 68 :=
by sorry

end probability_of_diff_families_is_correct_l259_259208


namespace min_ways_to_open_boxes_ways_with_exactly_two_cycles_l259_259798

-- Statement for the first part of the problem
theorem min_ways_to_open_boxes (n : ℕ) 
  (h_n : n = 10) :
  (∃ p : Equiv.Perm (Fin n), ∃ c : ℕ, c ≥ 2 ∧ p.cycleType.length ≥ 2) → 
  (∃ ways : ℕ, ways = 9 * (n - 1)!) :=
sorry

-- Statement for the second part of the problem
theorem ways_with_exactly_two_cycles (n : ℕ) 
  (h_n : n = 10) :
  (∃ p : Equiv.Perm (Fin n), p.cycleType.length = 2) →
  (∃ ways : ℕ, ways = 1024576) :=
sorry

end min_ways_to_open_boxes_ways_with_exactly_two_cycles_l259_259798


namespace negative_solution_iff_sum_zero_l259_259315

theorem negative_solution_iff_sum_zero (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔
  a + b + c = 0 :=
by
  sorry

end negative_solution_iff_sum_zero_l259_259315


namespace total_cats_in_training_center_l259_259472

-- Definitions corresponding to the given conditions
def cats_can_jump : ℕ := 60
def cats_can_fetch : ℕ := 35
def cats_can_meow : ℕ := 40
def cats_jump_fetch : ℕ := 20
def cats_fetch_meow : ℕ := 15
def cats_jump_meow : ℕ := 25
def cats_all_three : ℕ := 11
def cats_none : ℕ := 10

-- Theorem statement corresponding to proving question == answer given conditions
theorem total_cats_in_training_center
    (cjump : ℕ := cats_can_jump)
    (cfetch : ℕ := cats_can_fetch)
    (cmeow : ℕ := cats_can_meow)
    (cjf : ℕ := cats_jump_fetch)
    (cfm : ℕ := cats_fetch_meow)
    (cjm : ℕ := cats_jump_meow)
    (cat : ℕ := cats_all_three)
    (cno : ℕ := cats_none) :
    cjump
    + cfetch
    + cmeow
    - cjf
    - cfm
    - cjm
    + cat
    + cno
    = 96 := sorry

end total_cats_in_training_center_l259_259472


namespace probability_other_child_girl_l259_259261

theorem probability_other_child_girl 
  (a b : Bool) -- True for boy, False for girl
  (h : a = true ∨ b = true) :
  (cond (a = true) (if b = false then 1 else 0)
  + cond (a = false) (if b = true then 1 else 0))
  / (cond (a = true) (1) + cond (a = false) (if b = true then 1 else 0)) = 2 / 3 :=
sorry

end probability_other_child_girl_l259_259261


namespace find_first_number_l259_259951

variable {A B C D : ℕ}

theorem find_first_number (h1 : A + B + C = 60) (h2 : B + C + D = 45) (h3 : D = 18) : A = 33 := 
  sorry

end find_first_number_l259_259951


namespace problem1_problem2_l259_259703

theorem problem1 (a b : ℝ) (h1 : a = 3) :
  (∀ x : ℝ, 2 * x ^ 2 + (2 - a) * x - a > 0 ↔ x < -1 ∨ x > 3 / 2) :=
by
  sorry

theorem problem2 (a b : ℝ) (h1 : a = 3) :
  (∀ x : ℝ, 3 * x ^ 2 + b * x + 3 ≥ 0) ↔ (-6 ≤ b ∧ b ≤ 6) :=
by
  sorry

end problem1_problem2_l259_259703


namespace domain_of_f_l259_259218

noncomputable def f (x : ℝ) := Real.log (1 - x)

theorem domain_of_f : ∀ x, f x = Real.log (1 - x) → (1 - x > 0) →  x < 1 :=
by
  intro x h₁ h₂
  exact lt_of_sub_pos h₂

end domain_of_f_l259_259218


namespace five_integers_sum_to_first_set_impossible_second_set_sum_l259_259681

theorem five_integers_sum_to_first_set :
  ∃ (a b c d e : ℤ), 
    (a + b = 0) ∧ (a + c = 2) ∧ (b + c = 4) ∧ (a + d = 4) ∧ (b + d = 6) ∧
    (a + e = 8) ∧ (b + e = 9) ∧ (c + d = 11) ∧ (c + e = 13) ∧ (d + e = 15) ∧ 
    (a + b + c + d + e = 18) := 
sorry

theorem impossible_second_set_sum : 
  ¬∃ (a b c d e : ℤ), 
    (a + b = 12) ∧ (a + c = 13) ∧ (a + d = 14) ∧ (a + e = 15) ∧ (b + c = 16) ∧
    (b + d = 16) ∧ (b + e = 17) ∧ (c + d = 17) ∧ (c + e = 18) ∧ (d + e = 20) ∧
    (a + b + c + d + e = 39) :=
sorry

end five_integers_sum_to_first_set_impossible_second_set_sum_l259_259681


namespace find_fractions_l259_259488

open Function

-- Define the set and the condition that all numbers must be used precisely once
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define what it means for fractions to multiply to 1 within the set
def fractions_mul_to_one (a b c d e f : ℕ) : Prop :=
  (a * c * e) = (b * d * f)

-- Define irreducibility condition for a fraction a/b
def irreducible_fraction (a b : ℕ) := 
  Nat.gcd a b = 1

-- Final main problem statement
theorem find_fractions :
  ∃ (a b c d e f : ℕ) (h₁ : a ∈ S) (h₂ : b ∈ S) (h₃ : c ∈ S) (h₄ : d ∈ S) (h₅ : e ∈ S) (h₆ : f ∈ S),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
  d ≠ e ∧ d ≠ f ∧ 
  e ≠ f ∧
  irreducible_fraction a b ∧ irreducible_fraction c d ∧ irreducible_fraction e f ∧
  fractions_mul_to_one a b c d e f := 
sorry

end find_fractions_l259_259488


namespace compare_a_b_c_l259_259061

noncomputable def a := 2 * Real.log 1.01
noncomputable def b := Real.log 1.02
noncomputable def c := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l259_259061


namespace airline_odd_landings_l259_259748

/-- Given 1983 localities with direct service between any two of them and 10 international airlines
providing round-trip flights, prove that at least one of these airlines has a round trip with an odd
number of landings. -/
theorem airline_odd_landings (P : Fin 1983 → Type) (A : Fin 10 → Set (Σ i j, P i × P j))
  (h1 : ∀ (i j : Fin 1983), ∃ (a : Fin 10), ∃ (p : P i × P j), (⟨i, j, p⟩ ∈ A a))
  (h2 : ∀ (a : Fin 10) (i : Fin 1983) (j : Fin 1983) (p : P i × P j), (⟨i, j, p⟩ ∈ A a) ↔ (⟨j, i, p.swap⟩ ∈ A a)) :
  ∃ (a : Fin 10), ¬ bipartite (A a) := 
sorry

end airline_odd_landings_l259_259748


namespace base_b_addition_correct_base_b_l259_259981

theorem base_b_addition (b : ℕ) (hb : b > 5) :
  (2 * b^2 + 4 * b + 3) + (1 * b^2 + 5 * b + 2) = 4 * b^2 + 1 * b + 5 :=
  by
    sorry

theorem correct_base_b : ∃ (b : ℕ), b > 5 ∧ 
  (2 * b^2 + 4 * b + 3) + (1 * b^2 + 5 * b + 2) = 4 * b^2 + 1 * b + 5 ∧
  (4 + 5 = b + 1) ∧
  (2 + 1 + 1 = 4) :=
  ⟨8, 
   by decide,
   base_b_addition 8 (by decide),
   by decide,
   by decide⟩ 

end base_b_addition_correct_base_b_l259_259981


namespace find_d_values_l259_259403

theorem find_d_values (u v : ℝ) (c d : ℝ)
  (hpu : u^3 + c * u + d = 0)
  (hpv : v^3 + c * v + d = 0)
  (hqu : (u + 2)^3 + c * (u + 2) + d - 120 = 0)
  (hqv : (v - 5)^3 + c * (v - 5) + d - 120 = 0) :
  d = 396 ∨ d = 8 :=
by
  -- placeholder for the actual proof
  sorry

end find_d_values_l259_259403


namespace N_is_composite_l259_259836

def N : ℕ := 2011 * 2012 * 2013 * 2014 + 1

theorem N_is_composite : ¬ Prime N := by
  sorry

end N_is_composite_l259_259836


namespace washing_machine_heavy_wash_usage_l259_259646

-- Definition of variables and constants
variables (H : ℕ)                           -- Amount of water used for a heavy wash
def regular_wash : ℕ := 10                   -- Gallons used for a regular wash
def light_wash : ℕ := 2                      -- Gallons used for a light wash
def extra_light_wash : ℕ := light_wash       -- Extra light wash due to bleach

-- Number of each type of wash
def num_heavy_washes : ℕ := 2
def num_regular_washes : ℕ := 3
def num_light_washes : ℕ := 1
def num_bleached_washes : ℕ := 2

-- Total water usage
def total_water_usage : ℕ := 
  num_heavy_washes * H + 
  num_regular_washes * regular_wash + 
  num_light_washes * light_wash + 
  num_bleached_washes * extra_light_wash

-- Given total water usage
def given_total_water_usage : ℕ := 76

-- Lean statement to prove the amount of water used for a heavy wash
theorem washing_machine_heavy_wash_usage : total_water_usage H = given_total_water_usage → H = 20 :=
by
  sorry

end washing_machine_heavy_wash_usage_l259_259646


namespace probability_area_l259_259268

noncomputable def probability_x_y_le_five (x y : ℝ) : ℚ :=
  if 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 8 ∧ x + y ≤ 5 then 1 else 0

theorem probability_area {P : ℚ} :
  (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 8 → P = probability_x_y_le_five x y / (4 * 8)) →
  P = 5 / 16 :=
by
  sorry

end probability_area_l259_259268


namespace three_irreducible_fractions_prod_eq_one_l259_259482

-- Define the set of numbers available for use
def available_numbers : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a structure for an irreducible fraction
structure irreducible_fraction :=
(num : ℕ)
(denom : ℕ)
(h_coprime : Nat.gcd num denom = 1)
(h_in_set : num ∈ available_numbers ∧ denom ∈ available_numbers)

-- Definition of the main proof problem
theorem three_irreducible_fractions_prod_eq_one :
  ∃ (f1 f2 f3 : irreducible_fraction), 
    f1.num * f2.num * f3.num = f1.denom * f2.denom * f3.denom ∧ 
    f1.num ≠ f2.num ∧ f1.num ≠ f3.num ∧ f2.num ≠ f3.num ∧ 
    f1.denom ≠ f2.denom ∧ f1.denom ≠ f3.denom ∧ f2.denom ≠ f3.denom := 
by
  sorry

end three_irreducible_fractions_prod_eq_one_l259_259482


namespace drug_price_reduction_eq_l259_259095

variable (x : ℝ)
variable (initial_price : ℝ := 144)
variable (final_price : ℝ := 81)

theorem drug_price_reduction_eq :
  initial_price * (1 - x)^2 = final_price :=
by
  sorry

end drug_price_reduction_eq_l259_259095


namespace minimum_value_l259_259992

theorem minimum_value (x y z : ℝ) (h : 2 * x - 3 * y + z = 3) :
  ∃ y_min, y_min = -2 / 7 ∧ x = 6 / 7 ∧ (x^2 + (y - 1)^2 + z^2) = 18 / 7 :=
by
  sorry

end minimum_value_l259_259992


namespace photo_counts_correct_l259_259064

open Real

-- Definitions based on the conditions from step a)
def animal_photos : ℕ := 20
def flower_photos : ℕ := 30 -- 1.5 * 20
def total_animal_flower_photos : ℕ := animal_photos + flower_photos
def scenery_abstract_photos_combined : ℕ := (4 / 10) * total_animal_flower_photos -- 40% of total_animal_flower_photos

def x : ℕ := scenery_abstract_photos_combined / 5
def scenery_photos : ℕ := 3 * x
def abstract_photos : ℕ := 2 * x
def total_photos : ℕ := animal_photos + flower_photos + scenery_photos + abstract_photos

-- The statement to prove
theorem photo_counts_correct :
  animal_photos = 20 ∧
  flower_photos = 30 ∧
  total_animal_flower_photos = 50 ∧
  scenery_abstract_photos_combined = 20 ∧
  scenery_photos = 12 ∧
  abstract_photos = 8 ∧
  total_photos = 70 :=
by
  sorry

end photo_counts_correct_l259_259064


namespace mutually_exclusive_A_C_probability_BC_eq_C_l259_259456

open ProbabilityTheory Finset

def samplespace := ({1, 2, 3, 4, 5, 6} : Finset ℕ) -- Representing bottle indices in the case
def selection := samplespace.powerset.filter (λ s, card s = 2) -- All 2-bottle selections

def event_A (s : Finset ℕ) : Prop := 1 ∉ s ∧ 2 ∉ s -- "A did not win a prize"
def event_B (s : Finset ℕ) : Prop := 1 ∈ s ∧ 2 ∉ s -- "A won the first prize"
def event_C (s : Finset ℕ) : Prop := 1 ∈ s ∨ 2 ∈ s -- "A won a prize"

theorem mutually_exclusive_A_C :
  ∀ s ∈ selection, event_A s → ¬ event_C s :=
begin
  intros s hs hA hC,
  rw [event_C, event_A] at *,
  cases hC,
  { exact hA.1 hC },
  { exact hA.2 hC },
end

theorem probability_BC_eq_C :
  P(event_B ∪ event_C) = P(event_C) :=
begin
  sorry -- Placeholder for the actual probability computation proof
end

end mutually_exclusive_A_C_probability_BC_eq_C_l259_259456


namespace value_of_x_l259_259612

theorem value_of_x (x : ℝ) : 
  (x ≤ 0 → x^2 + 1 = 5 → x = -2) ∧ 
  (0 < x → -2 * x = 5 → false) := 
sorry

end value_of_x_l259_259612


namespace arun_deepak_age_ratio_l259_259966

-- Define the current age of Arun based on the condition that after 6 years he will be 26 years old
def Arun_current_age : ℕ := 26 - 6

-- Define Deepak's current age based on the given condition
def Deepak_current_age : ℕ := 15

-- The present ratio between Arun's age and Deepak's age
theorem arun_deepak_age_ratio : Arun_current_age / Nat.gcd Arun_current_age Deepak_current_age = (4 : ℕ) ∧ Deepak_current_age / Nat.gcd Arun_current_age Deepak_current_age = (3 : ℕ) := 
by
  -- Proof omitted
  sorry

end arun_deepak_age_ratio_l259_259966


namespace remaining_quantities_count_l259_259928

theorem remaining_quantities_count 
  (S : ℕ) (S3 : ℕ) (S2 : ℕ) (n : ℕ) 
  (h1 : S / 5 = 10) 
  (h2 : S3 / 3 = 4) 
  (h3 : S = 50) 
  (h4 : S3 = 12) 
  (h5 : S2 = S - S3) 
  (h6 : S2 / n = 19) 
  : n = 2 := 
by 
  sorry

end remaining_quantities_count_l259_259928


namespace existence_of_nonnegative_value_l259_259696

theorem existence_of_nonnegative_value :
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x + 12 ≥ 0 := 
by
  sorry

end existence_of_nonnegative_value_l259_259696


namespace stratified_sampling_example_l259_259259

noncomputable def sample_proportion := 70 / 3500
noncomputable def total_students := 3500 + 1500
noncomputable def sample_size := total_students * sample_proportion

theorem stratified_sampling_example 
  (high_school_students : ℕ := 3500)
  (junior_high_students : ℕ := 1500)
  (sampled_high_school_students : ℕ := 70)
  (proportion_of_sampling : ℝ := sampled_high_school_students / high_school_students)
  (total_number_of_students : ℕ := high_school_students + junior_high_students)
  (calculated_sample_size : ℝ := total_number_of_students * proportion_of_sampling) :
  calculated_sample_size = 100 :=
by
  sorry

end stratified_sampling_example_l259_259259


namespace compare_abc_l259_259049

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  unfold a b c
  -- Proof goes here
  sorry

end compare_abc_l259_259049


namespace perfect_square_l259_259462

variables {n x k ℓ : ℕ}

theorem perfect_square (h1 : x^2 < n) (h2 : n < (x + 1)^2)
  (h3 : k = n - x^2) (h4 : ℓ = (x + 1)^2 - n) :
  ∃ m : ℕ, n - k * ℓ = m^2 :=
by
  sorry

end perfect_square_l259_259462


namespace tangent_line_correct_l259_259931

noncomputable def f : ℝ → ℝ := λ x, 4 * x - x ^ 3

def tangent_line_at_point (x1 y1 : ℝ) (k : ℝ) : ℝ → ℝ := λ x, k * (x - x1) + y1

theorem tangent_line_correct :
  tangent_line_at_point (-1) (-3) 1 = λ x, x - 2 := sorry

end tangent_line_correct_l259_259931
