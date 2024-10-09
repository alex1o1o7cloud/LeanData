import Mathlib

namespace max_triangles_9261_l123_12360

-- Define the problem formally
noncomputable def max_triangles (points : ℕ) (circ_radius : ℝ) (min_side_length : ℝ) : ℕ :=
  -- Function definition for calculating the maximum number of triangles
  sorry

-- State the conditions and the expected maximum number of triangles
theorem max_triangles_9261 :
  max_triangles 63 10 9 = 9261 :=
sorry

end max_triangles_9261_l123_12360


namespace num_three_digit_ints_with_odd_factors_l123_12352

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l123_12352


namespace range_of_m_l123_12303

theorem range_of_m (m : ℝ) (h : 1 < m) :
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 1 → -m ≤ x ∧ x ≤ m - 1) → (3 ≤ m) :=
by
  sorry  -- The proof will be constructed here.

end range_of_m_l123_12303


namespace problem_statement_l123_12311

noncomputable def myFunction (f : ℝ → ℝ) := 
  (∀ x y : ℝ, f (x^3 + y^3) = (x + y) * (f x ^ 2 - f x * f y + f y ^ 2)) 

theorem problem_statement (f : ℝ → ℝ) 
  (h : myFunction f) : 
  ∀ x : ℝ, f (2005 * x) = 2005 * f x :=
sorry

end problem_statement_l123_12311


namespace polar_to_rectangular_l123_12313

theorem polar_to_rectangular (r θ : ℝ) (h₁ : r = 6) (h₂ : θ = Real.pi / 2) :
  (r * Real.cos θ, r * Real.sin θ) = (0, 6) :=
by
  sorry

end polar_to_rectangular_l123_12313


namespace age_difference_is_51_l123_12353

def Milena_age : ℕ := 7
def Grandmother_age : ℕ := 9 * Milena_age
def Grandfather_age : ℕ := Grandmother_age + 2
def Cousin_age : ℕ := 2 * Milena_age
def Age_difference : ℕ := Grandfather_age - Cousin_age

theorem age_difference_is_51 : Age_difference = 51 := by
  sorry

end age_difference_is_51_l123_12353


namespace shelves_of_picture_books_l123_12315

-- Define the conditions
def n_mystery : ℕ := 5
def b_per_shelf : ℕ := 4
def b_total : ℕ := 32

-- State the main theorem to be proven
theorem shelves_of_picture_books :
  (b_total - n_mystery * b_per_shelf) / b_per_shelf = 3 :=
by
  -- The proof is omitted
  sorry

end shelves_of_picture_books_l123_12315


namespace support_percentage_l123_12325

theorem support_percentage (men women : ℕ) (support_men_percentage support_women_percentage : ℝ) 
(men_support women_support total_support : ℕ)
(hmen : men = 150) 
(hwomen : women = 850) 
(hsupport_men_percentage : support_men_percentage = 0.55) 
(hsupport_women_percentage : support_women_percentage = 0.70) 
(hmen_support : men_support = 83) 
(hwomen_support : women_support = 595)
(htotal_support : total_support = men_support + women_support) :
  ((total_support : ℝ) / (men + women) * 100) = 68 :=
by
  -- Insert the proof here to verify each step of the calculation and rounding
  sorry

end support_percentage_l123_12325


namespace sum_of_squares_of_reciprocals_l123_12384

-- Definitions based on the problem's conditions
variables (a b : ℝ) (hab : a + b = 3 * a * b + 1) (h_an : a ≠ 0) (h_bn : b ≠ 0)

-- Statement of the problem to be proved
theorem sum_of_squares_of_reciprocals :
  (1 / a^2) + (1 / b^2) = (4 * a * b + 10) / (a^2 * b^2) :=
sorry

end sum_of_squares_of_reciprocals_l123_12384


namespace tenth_term_l123_12376

-- Define the conditions
variables {a d : ℤ}

-- The conditions of the problem
axiom third_term_condition : a + 2 * d = 10
axiom sixth_term_condition : a + 5 * d = 16

-- The goal is to prove the tenth term
theorem tenth_term : a + 9 * d = 24 :=
by
  sorry

end tenth_term_l123_12376


namespace next_time_10_10_11_15_l123_12343

noncomputable def next_time_angle_x (current_time : ℕ × ℕ) (x : ℕ) : ℕ × ℕ := sorry

theorem next_time_10_10_11_15 :
  ∀ (x : ℕ), next_time_angle_x (10, 10) 115 = (11, 15) := sorry

end next_time_10_10_11_15_l123_12343


namespace regular_tetrahedron_subdivision_l123_12380

theorem regular_tetrahedron_subdivision :
  ∃ (n : ℕ), n ≤ 7 ∧ (∀ (i : ℕ) (h : i ≥ n), (1 / 2^i) < (1 / 100)) :=
by
  sorry

end regular_tetrahedron_subdivision_l123_12380


namespace percentage_of_second_solution_is_16point67_l123_12385

open Real

def percentage_second_solution (x : ℝ) : Prop :=
  let v₁ := 50
  let c₁ := 0.10
  let c₂ := x / 100
  let v₂ := 200 - v₁
  let c_final := 0.15
  let v_final := 200
  (c₁ * v₁) + (c₂ * v₂) = (c_final * v_final)

theorem percentage_of_second_solution_is_16point67 :
  ∃ x, percentage_second_solution x ∧ x = (50/3) :=
sorry

end percentage_of_second_solution_is_16point67_l123_12385


namespace total_amount_shared_l123_12377

theorem total_amount_shared (a b c : ℕ) (h_ratio : a * 5 = b * 3) (h_ben : b = 25) (h_ratio_ben : b * 12 = c * 5) :
  a + b + c = 100 := by
  sorry

end total_amount_shared_l123_12377


namespace cos_double_angle_l123_12356

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 1/3) : Real.cos (2 * α) = 7/9 :=
by
    sorry

end cos_double_angle_l123_12356


namespace inequality_solution_set_l123_12301

theorem inequality_solution_set (a : ℝ) (x : ℝ) (h : (a - 1) * x > 2) : x < 2 / (a - 1) ↔ a < 1 :=
by
  sorry

end inequality_solution_set_l123_12301


namespace compare_abc_l123_12307

noncomputable def a : ℝ := Real.log 0.3 / Real.log 2
noncomputable def b : ℝ := Real.exp (0.3 * Real.log 2)
noncomputable def c : ℝ := Real.exp (0.2 * Real.log 0.3)

theorem compare_abc : a < c ∧ c < b :=
by
  -- The proof will be provided here.
  sorry

end compare_abc_l123_12307


namespace product_of_possible_values_of_b_l123_12379

theorem product_of_possible_values_of_b :
  let y₁ := -1
  let y₂ := 4
  let x₁ := 1
  let side_length := y₂ - y₁ -- Since this is 5 units
  let b₁ := x₁ - side_length -- This should be -4
  let b₂ := x₁ + side_length -- This should be 6
  let product := b₁ * b₂ -- So, (-4) * 6
  product = -24 :=
by
  sorry

end product_of_possible_values_of_b_l123_12379


namespace scatter_plot_can_be_made_l123_12387

theorem scatter_plot_can_be_made
    (data : List (ℝ × ℝ)) :
    ∃ (scatter_plot : List (ℝ × ℝ)), scatter_plot = data :=
by
  sorry

end scatter_plot_can_be_made_l123_12387


namespace weightOfEachPacket_l123_12397

/-- Definition for the number of pounds in one ton --/
def poundsPerTon : ℕ := 2100

/-- Total number of packets filling the 13-ton capacity --/
def numPackets : ℕ := 1680

/-- Capacity of the gunny bag in tons --/
def capacityInTons : ℕ := 13

/-- Total weight of the gunny bag in pounds --/
def totalWeightInPounds : ℕ := capacityInTons * poundsPerTon

/-- Statement that each packet weighs 16.25 pounds --/
theorem weightOfEachPacket : (totalWeightInPounds / numPackets : ℚ) = 16.25 :=
sorry

end weightOfEachPacket_l123_12397


namespace probability_two_red_crayons_l123_12383

def num_crayons : ℕ := 6
def num_red : ℕ := 3
def num_blue : ℕ := 2
def num_green : ℕ := 1
def num_choose (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_two_red_crayons :
  let total_pairs := num_choose num_crayons 2
  let red_pairs := num_choose num_red 2
  (red_pairs : ℚ) / (total_pairs : ℚ) = 1 / 5 :=
by
  sorry

end probability_two_red_crayons_l123_12383


namespace a4_eq_2_or_neg2_l123_12344

variable (a : ℕ → ℝ)
variable (r : ℝ)

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- Given conditions
axiom h1 : is_geometric_sequence a r
axiom h2 : a 2 * a 6 = 4

-- Theorem to prove
theorem a4_eq_2_or_neg2 : a 4 = 2 ∨ a 4 = -2 :=
sorry

end a4_eq_2_or_neg2_l123_12344


namespace negation_of_p_l123_12365

-- Define the original proposition p
def p : Prop := ∀ x : ℝ, 2 * x^2 - 1 > 0

-- State the theorem that the negation of p is equivalent to the given existential statement
theorem negation_of_p :
  ¬p ↔ ∃ x : ℝ, 2 * x^2 - 1 ≤ 0 :=
by
  sorry

end negation_of_p_l123_12365


namespace aarons_brothers_number_l123_12334

-- We are defining the conditions as functions

def number_of_aarons_sisters := 4
def bennetts_brothers := 6
def bennetts_cousins := 3
def twice_aarons_brothers_minus_two (Ba : ℕ) := 2 * Ba - 2
def bennetts_cousins_one_more_than_aarons_sisters (As : ℕ) := As + 1

-- We need to prove that Aaron's number of brothers Ba is 4 under these conditions

theorem aarons_brothers_number : ∃ (Ba : ℕ), 
  bennetts_brothers = twice_aarons_brothers_minus_two Ba ∧ 
  bennetts_cousins = bennetts_cousins_one_more_than_aarons_sisters number_of_aarons_sisters ∧ 
  Ba = 4 :=
by {
  sorry
}

end aarons_brothers_number_l123_12334


namespace fraction_students_say_like_actually_dislike_l123_12309

theorem fraction_students_say_like_actually_dislike :
  let n := 200
  let p_l := 0.70
  let p_d := 0.30
  let p_ll := 0.85
  let p_ld := 0.15
  let p_dd := 0.80
  let p_dl := 0.20
  let num_like := p_l * n
  let num_dislike := p_d * n
  let num_ll := p_ll * num_like
  let num_ld := p_ld * num_like
  let num_dd := p_dd * num_dislike
  let num_dl := p_dl * num_dislike
  let total_say_like := num_ll + num_dl
  (num_dl / total_say_like) = 12 / 131 := 
by
  sorry

end fraction_students_say_like_actually_dislike_l123_12309


namespace round_robin_tournament_points_l123_12331

theorem round_robin_tournament_points :
  ∀ (teams : Finset ℕ), teams.card = 6 →
  ∀ (matches_played : ℕ), matches_played = 12 →
  ∀ (total_points : ℤ), total_points = 32 →
  ∀ (third_highest_points : ℤ), third_highest_points = 7 →
  ∀ (draws : ℕ), draws = 4 →
  ∃ (fifth_highest_points_min fifth_highest_points_max : ℤ),
    fifth_highest_points_min = 1 ∧
    fifth_highest_points_max = 3 :=
by
  sorry

end round_robin_tournament_points_l123_12331


namespace part1_part2_l123_12359

-- Define the triangle with sides a, b, c and the properties given.
variable (a b c : ℝ) (A B C : ℝ)
variable (A_ne_zero : A ≠ 0)
variable (b_cos_C a_cos_A c_cos_B : ℝ)

-- Given conditions
variable (h1 : b_cos_C = b * Real.cos C)
variable (h2 : a_cos_A = a * Real.cos A)
variable (h3 : c_cos_B = c * Real.cos B)
variable (h_seq : b_cos_C + c_cos_B = 2 * a_cos_A)
variable (A_plus_B_plus_C_eq_pi : A + B + C = Real.pi)

-- Part 1
theorem part1 : (A = Real.pi / 3) :=
by sorry

-- Part 2 with additional conditions
variable (h_a : a = 3 * Real.sqrt 2)
variable (h_bc_sum : b + c = 6)

theorem part2 : (|Real.sqrt (b ^ 2 + c ^ 2 - b * c)| = Real.sqrt 30) :=
by sorry

end part1_part2_l123_12359


namespace compare_neg_two_and_neg_one_l123_12312

theorem compare_neg_two_and_neg_one : -2 < -1 :=
by {
  -- Proof is omitted
  sorry
}

end compare_neg_two_and_neg_one_l123_12312


namespace cat_food_per_day_l123_12364

theorem cat_food_per_day
  (bowl_empty_weight : ℕ)
  (bowl_weight_after_eating : ℕ)
  (food_eaten : ℕ)
  (days_per_fill : ℕ)
  (daily_food : ℕ) :
  (bowl_empty_weight = 420) →
  (bowl_weight_after_eating = 586) →
  (food_eaten = 14) →
  (days_per_fill = 3) →
  (bowl_weight_after_eating - bowl_empty_weight + food_eaten = days_per_fill * daily_food) →
  daily_food = 60 :=
by
  sorry

end cat_food_per_day_l123_12364


namespace number_of_whole_numbers_without_1_or_2_l123_12330

/-- There are 439 whole numbers between 1 and 500 that do not contain the digit 1 or 2. -/
theorem number_of_whole_numbers_without_1_or_2 : 
  ∃ n : ℕ, n = 439 ∧ ∀ m, 1 ≤ m ∧ m ≤ 500 → ∀ d ∈ (m.digits 10), d ≠ 1 ∧ d ≠ 2 :=
sorry

end number_of_whole_numbers_without_1_or_2_l123_12330


namespace find_smallest_x_l123_12335

-- Definition of the conditions
def cong1 (x : ℤ) : Prop := x % 5 = 4
def cong2 (x : ℤ) : Prop := x % 7 = 6
def cong3 (x : ℤ) : Prop := x % 8 = 7

-- Statement of the problem
theorem find_smallest_x :
  ∃ (x : ℕ), x > 0 ∧ cong1 x ∧ cong2 x ∧ cong3 x ∧ x = 279 :=
by
  sorry

end find_smallest_x_l123_12335


namespace evaluate_expression_l123_12350

theorem evaluate_expression : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 = -1 := 
  sorry

end evaluate_expression_l123_12350


namespace range_of_lg_x_l123_12355

variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x

theorem range_of_lg_x {f : ℝ → ℝ} (h_even : is_even f)
    (h_decreasing : is_decreasing_on_nonneg f)
    (h_condition : f (Real.log x) > f 1) :
    x ∈ Set.Ioo (1/10 : ℝ) (10 : ℝ) :=
  sorry

end range_of_lg_x_l123_12355


namespace find_other_number_l123_12393

theorem find_other_number (HCF LCM num1 num2 : ℕ) 
    (h_hcf : HCF = 14)
    (h_lcm : LCM = 396)
    (h_num1 : num1 = 36)
    (h_prod : HCF * LCM = num1 * num2)
    : num2 = 154 := by
  sorry

end find_other_number_l123_12393


namespace problem_solution_l123_12327

theorem problem_solution
  (n m k l : ℕ)
  (h1 : n ≠ 1)
  (h2 : 0 < n)
  (h3 : 0 < m)
  (h4 : 0 < k)
  (h5 : 0 < l)
  (h6 : n^k + m * n^l + 1 ∣ n^(k + l) - 1) :
  (m = 1 ∧ l = 2 * k) ∨ (l ∣ k ∧ m = (n^(k - l) - 1) / (n^l - 1)) :=
by
  sorry

end problem_solution_l123_12327


namespace find_k_value_l123_12317

theorem find_k_value (k : ℝ) : 
  (∃ x1 x2 x3 x4 : ℝ,
    x1 ≠ 0 ∧ x2 ≠ 0 ∧ x3 ≠ 0 ∧ x4 ≠ 0 ∧
    (x1^2 - 1) * (x1^2 - 4) = k ∧
    (x2^2 - 1) * (x2^2 - 4) = k ∧
    (x3^2 - 1) * (x3^2 - 4) = k ∧
    (x4^2 - 1) * (x4^2 - 4) = k ∧
    x1 ≠ x2 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧
    x4 - x3 = x3 - x2 ∧ x2 - x1 = x4 - x3) → 
  k = 7/4 := 
by
  sorry

end find_k_value_l123_12317


namespace find_k_value_l123_12300

def line (k : ℝ) (x y : ℝ) : Prop := 3 - 2 * k * x = -4 * y

def on_line (k : ℝ) : Prop := line k 5 (-2)

theorem find_k_value (k : ℝ) : on_line k → k = -0.5 :=
by
  sorry

end find_k_value_l123_12300


namespace shen_winning_probability_sum_l123_12346

/-!
# Shen Winning Probability

Prove that the sum of the numerator and the denominator, m + n, 
of the simplified fraction representing Shen's winning probability is 184.
-/

theorem shen_winning_probability_sum :
  let m := 67
  let n := 117
  m + n = 184 :=
by sorry

end shen_winning_probability_sum_l123_12346


namespace remainder_div_l123_12372

theorem remainder_div (P Q R D Q' R' : ℕ) (h₁ : P = Q * D + R) (h₂ : Q = (D - 1) * Q' + R') (h₃ : D > 1) :
  P % (D * (D - 1)) = D * R' + R := by sorry

end remainder_div_l123_12372


namespace inversely_proportional_y_value_l123_12337

theorem inversely_proportional_y_value (x y k : ℝ)
  (h1 : ∀ x y : ℝ, x * y = k)
  (h2 : ∃ y : ℝ, x = 3 * y ∧ x + y = 36 ∧ x * y = k)
  (h3 : x = -9) : y = -27 := 
by
  sorry

end inversely_proportional_y_value_l123_12337


namespace bc_approx_A_l123_12369

theorem bc_approx_A (A B C D E : ℝ) 
    (hA : 0 < A ∧ A < 1) (hB : 0 < B ∧ B < 1) (hC : 0 < C ∧ C < 1)
    (hD : 0 < D ∧ D < 1) (hE : 1 < E ∧ E < 2)
    (hA_val : A = 0.2) (hB_val : B = 0.4) (hC_val : C = 0.6) (hD_val : D = 0.8) :
    abs (B * C - A) < abs (B * C - B) ∧ abs (B * C - A) < abs (B * C - C) ∧ abs (B * C - A) < abs (B * C - D) := 
by 
  sorry

end bc_approx_A_l123_12369


namespace possible_ages_that_sum_to_a_perfect_square_l123_12390

def two_digit_number (a b : ℕ) := 10 * a + b
def reversed_number (a b : ℕ) := 10 * b + a

def sum_of_number_and_its_reversed (a b : ℕ) : ℕ := 
  two_digit_number a b + reversed_number a b

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem possible_ages_that_sum_to_a_perfect_square :
  ∃ (s : Finset ℕ), s.card = 6 ∧ 
  ∀ x ∈ s, ∃ a b : ℕ, a + b = 11 ∧ s = {two_digit_number a b} ∧ is_perfect_square (sum_of_number_and_its_reversed a b) :=
  sorry

end possible_ages_that_sum_to_a_perfect_square_l123_12390


namespace mul_72516_9999_l123_12358

theorem mul_72516_9999 : 72516 * 9999 = 724787484 :=
by
  sorry

end mul_72516_9999_l123_12358


namespace find_k_and_shifted_function_l123_12375

noncomputable def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x + 1

theorem find_k_and_shifted_function (k : ℝ) (h : k ≠ 0) (h1 : linear_function k 1 = 3) :
  k = 2 ∧ linear_function 2 x + 2 = 2 * x + 3 :=
by
  sorry

end find_k_and_shifted_function_l123_12375


namespace prob_xi_eq_12_l123_12320

noncomputable def prob_of_draws (total_draws red_draws : ℕ) (prob_red prob_white : ℚ) : ℚ :=
    (Nat.choose (total_draws - 1) (red_draws - 1)) * (prob_red ^ (red_draws - 1)) * (prob_white ^ (total_draws - red_draws)) * prob_red

theorem prob_xi_eq_12 :
    prob_of_draws 12 10 (3 / 8) (5 / 8) = 
    (Nat.choose 11 9) * (3 / 8)^9 * (5 / 8)^2 * (3 / 8) :=
by sorry

end prob_xi_eq_12_l123_12320


namespace exists_divisible_by_3_on_circle_l123_12394

theorem exists_divisible_by_3_on_circle :
  ∃ a : ℕ → ℕ, (∀ i, a i ≥ 1) ∧
               (∀ i, i < 99 → (a (i + 1) < 99 → (a (i + 1) - a i = 1 ∨ a (i + 1) - a i = 2 ∨ a (i + 1) = 2 * a i))) ∧
               (∃ i, i < 99 ∧ a i % 3 = 0) := 
sorry

end exists_divisible_by_3_on_circle_l123_12394


namespace rectangle_perimeter_divided_into_six_congruent_l123_12392

theorem rectangle_perimeter_divided_into_six_congruent (l w : ℕ) (h1 : 2 * (w + l / 6) = 40) (h2 : l = 120 - 6 * w) : 
  2 * (l + w) = 280 :=
by
  sorry

end rectangle_perimeter_divided_into_six_congruent_l123_12392


namespace percent_increase_hypotenuse_l123_12339

theorem percent_increase_hypotenuse :
  let l1 := 3
  let l2 := 1.25 * l1
  let l3 := 1.25 * l2
  let l4 := 1.25 * l3
  let h1 := l1 * Real.sqrt 2
  let h4 := l4 * Real.sqrt 2
  ((h4 - h1) / h1) * 100 = 95.3 :=
by
  sorry

end percent_increase_hypotenuse_l123_12339


namespace employee_n_salary_l123_12321

variable (m n : ℝ)

theorem employee_n_salary 
  (h1 : m + n = 605) 
  (h2 : m = 1.20 * n) : 
  n = 275 :=
by
  sorry

end employee_n_salary_l123_12321


namespace find_b_coefficients_l123_12389

theorem find_b_coefficients (x : ℝ) (b₁ b₂ b₃ b₄ : ℝ) :
  x^4 = (x + 1)^4 + b₁ * (x + 1)^3 + b₂ * (x + 1)^2 + b₃ * (x + 1) + b₄ →
  b₁ = -4 ∧ b₂ = 6 ∧ b₃ = -4 ∧ b₄ = 1 := by
  sorry

end find_b_coefficients_l123_12389


namespace george_elaine_ratio_l123_12323

-- Define the conditions
def time_jerry := 3
def time_elaine := 2 * time_jerry
def time_kramer := 0
def total_time := 11

-- Define George's time based on the given total time condition
def time_george := total_time - (time_jerry + time_elaine + time_kramer)

-- Prove the ratio of George's time to Elaine's time is 1:3
theorem george_elaine_ratio : time_george / time_elaine = 1 / 3 :=
by
  -- Lean proof would go here
  sorry

end george_elaine_ratio_l123_12323


namespace correct_conclusions_l123_12357

noncomputable def M : Set ℝ := sorry

axiom non_empty : Nonempty M
axiom mem_2 : (2 : ℝ) ∈ M
axiom closed_under_sub : ∀ {x y : ℝ}, x ∈ M → y ∈ M → (x - y) ∈ M
axiom closed_under_div : ∀ {x : ℝ}, x ∈ M → x ≠ 0 → (1 / x) ∈ M

theorem correct_conclusions :
  (0 : ℝ) ∈ M ∧
  (∀ x y : ℝ, x ∈ M → y ∈ M → (x + y) ∈ M) ∧
  (∀ x y : ℝ, x ∈ M → y ∈ M → (x * y) ∈ M) ∧
  ¬ (1 ∉ M) := sorry

end correct_conclusions_l123_12357


namespace trapezoid_bases_12_and_16_l123_12322

theorem trapezoid_bases_12_and_16 :
  ∀ (h R : ℝ) (a b : ℝ),
    (R = 10) →
    (h = (a + b) / 2) →
    (∀ k m, ((k = 3/7 * h) ∧ (m = 4/7 * h) ∧ (R^2 = k^2 + (a/2)^2) ∧ (R^2 = m^2 + (b/2)^2))) →
    (a = 12) ∧ (b = 16) :=
by
  intros h R a b hR hMid eqns
  sorry

end trapezoid_bases_12_and_16_l123_12322


namespace find_m_l123_12302

theorem find_m (m : ℝ) (h : |m| = |m + 2|) : m = -1 :=
sorry

end find_m_l123_12302


namespace numLinesTangentToCircles_eq_2_l123_12306

noncomputable def lineTangents (A B : Point) (dAB rA rB : ℝ) : ℕ :=
  if dAB < rA + rB then 2 else 0

theorem numLinesTangentToCircles_eq_2
  (A B : Point) (dAB rA rB : ℝ)
  (hAB : dAB = 4) (hA : rA = 3) (hB : rB = 2) :
  lineTangents A B dAB rA rB = 2 := by
  sorry

end numLinesTangentToCircles_eq_2_l123_12306


namespace three_numbers_sum_l123_12354

theorem three_numbers_sum (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 10)
  (h4 : (a + b + c) / 3 = a + 8) (h5 : (a + b + c) / 3 = c - 20) : 
  a + b + c = 66 :=
sorry

end three_numbers_sum_l123_12354


namespace PropositionA_PropositionB_PropositionC_PropositionD_l123_12366

-- Proposition A (Incorrect)
theorem PropositionA : ¬(∀ a b c : ℝ, a > b ∧ b > 0 → a * c^2 > b * c^2) :=
sorry

-- Proposition B (Correct)
theorem PropositionB : ∀ a b : ℝ, -2 < a ∧ a < 3 ∧ 1 < b ∧ b < 2 → -4 < a - b ∧ a - b < 2 :=
sorry

-- Proposition C (Correct)
theorem PropositionC : ∀ a b c : ℝ, a > b ∧ b > 0 ∧ c < 0 → c / (a^2) > c / (b^2) :=
sorry

-- Proposition D (Incorrect)
theorem PropositionD : ¬(∀ a b c : ℝ, c > a ∧ a > b → a / (c - a) > b / (c - b)) :=
sorry

end PropositionA_PropositionB_PropositionC_PropositionD_l123_12366


namespace tessellation_solutions_l123_12396

theorem tessellation_solutions (m n : ℕ) (h : 60 * m + 90 * n = 360) : m = 3 ∧ n = 2 :=
by
  sorry

end tessellation_solutions_l123_12396


namespace determine_roles_l123_12398

/-
We have three inhabitants K, M, R.
One of them is a truth-teller (tt), one is a liar (l), 
and one is a trickster (tr).
K states: "I am a trickster."
M states: "That is true."
R states: "I am not a trickster."
A truth-teller always tells the truth.
A liar always lies.
A trickster sometimes lies and sometimes tells the truth.
-/

inductive Role
| truth_teller | liar | trickster

open Role

def inhabitant_role (K M R : Role) : Prop :=
  ((K = liar) ∧ (M = trickster) ∧ (R = truth_teller)) ∧
  (K = trickster → K ≠ K) ∧
  (M = truth_teller → M = truth_teller) ∧
  (R = trickster → R ≠ R)

theorem determine_roles (K M R : Role) : inhabitant_role K M R :=
sorry

end determine_roles_l123_12398


namespace study_tour_part1_l123_12382

theorem study_tour_part1 (x y : ℕ) 
  (h1 : 45 * y + 15 = x) 
  (h2 : 60 * (y - 3) = x) : 
  x = 600 ∧ y = 13 :=
by sorry

end study_tour_part1_l123_12382


namespace inequality_proof_l123_12333

theorem inequality_proof
  (a b c A α : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < α)
  (h_sum : a + b + c = A)
  (h_A : A ≤ 1) :
  (1 / a - a) ^ α + (1 / b - b) ^ α + (1 / c - c) ^ α ≥ 3 * (3 / A - A / 3) ^ α :=
by
  sorry

end inequality_proof_l123_12333


namespace find_p_q_l123_12319

variable (p q : ℝ)
def f (x : ℝ) : ℝ := x^2 + p * x + q

theorem find_p_q:
  (p, q) = (-6, 7) →
  ∀ x : ℝ, (1 ≤ x ∧ x ≤ 5) → |f p q x| ≤ 2 :=
by
  sorry

end find_p_q_l123_12319


namespace complex_exponentiation_problem_l123_12388

theorem complex_exponentiation_problem (z : ℂ) (h : z^2 + z + 1 = 0) : z^2010 + z^2009 + 1 = 0 :=
sorry

end complex_exponentiation_problem_l123_12388


namespace simplify_expression_l123_12363

theorem simplify_expression (x : ℝ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 :=
by
  sorry

end simplify_expression_l123_12363


namespace scale_drawing_l123_12373

theorem scale_drawing (length_cm : ℝ) (representation : ℝ) : length_cm * representation = 3750 :=
by
  let length_cm := 7.5
  let representation := 500
  sorry

end scale_drawing_l123_12373


namespace total_money_l123_12351

theorem total_money (A B C : ℝ) (h1 : A = 1 / 2 * (B + C))
  (h2 : B = 2 / 3 * (A + C)) (h3 : A = 122) :
  A + B + C = 366 := by
  sorry

end total_money_l123_12351


namespace parts_sampling_l123_12374

theorem parts_sampling (first_grade second_grade third_grade : ℕ)
                       (total_sample drawn_third : ℕ)
                       (h_first_grade : first_grade = 24)
                       (h_second_grade : second_grade = 36)
                       (h_total_sample : total_sample = 20)
                       (h_drawn_third : drawn_third = 10)
                       (h_non_third : third_grade = 60 - (24 + 36))
                       (h_total : 2 * (24 + 36) = 120)
                       (h_proportion : 2 * third_grade = 2 * (24 + 36)) :
    (third_grade = 60 ∧ (second_grade * (total_sample - drawn_third) / (24 + 36) = 6)) := by
    simp [h_first_grade, h_second_grade, h_total_sample, h_drawn_third] at *
    sorry

end parts_sampling_l123_12374


namespace tylenol_tablet_mg_l123_12399

/-- James takes 2 Tylenol tablets every 6 hours and consumes 3000 mg a day.
    Prove the mg of each Tylenol tablet. -/
theorem tylenol_tablet_mg (t : ℕ) (h1 : t = 2) (h2 : 24 / 6 = 4) (h3 : 3000 / (4 * t) = 375) : t * (4 * t) = 3000 :=
by
  sorry

end tylenol_tablet_mg_l123_12399


namespace max_value_f1_solve_inequality_f2_l123_12324

def f_1 (x : ℝ) : ℝ := |x + 1| - |x - 1|

theorem max_value_f1 : ∃ x, f_1 x = 2 :=
sorry

def f_2 (x : ℝ) : ℝ := |2 * x - 1| - |x - 1|

theorem solve_inequality_f2 (x : ℝ) : f_2 x ≥ 1 ↔ x ≤ -1 ∨ x ≥ 1 :=
sorry

end max_value_f1_solve_inequality_f2_l123_12324


namespace problem_l123_12328

open Set

theorem problem (a b : ℝ) :
  (A = {x | x^2 - 2*x - 3 > 0}) →
  (B = {x | x^2 + a*x + b ≤ 0}) →
  (A ∪ B = univ) →
  (A ∩ B = Ioo 3 4) →
  a + b = -7 :=
by
  intros hA hB hUnion hIntersection
  sorry

end problem_l123_12328


namespace nonnegative_integer_solutions_l123_12305

theorem nonnegative_integer_solutions :
  {ab : ℕ × ℕ | 3 * 2^ab.1 + 1 = ab.2^2} = {(0, 2), (3, 5), (4, 7)} :=
by
  sorry

end nonnegative_integer_solutions_l123_12305


namespace cone_cube_volume_ratio_l123_12316

theorem cone_cube_volume_ratio (s : ℝ) (h : ℝ) (r : ℝ) (π : ℝ) 
  (cone_inscribed_in_cube : r = s / 2 ∧ h = s ∧ π > 0) :
  ((1/3) * π * r^2 * h) / (s^3) = π / 12 :=
by
  sorry

end cone_cube_volume_ratio_l123_12316


namespace trains_cross_time_l123_12332

theorem trains_cross_time
  (length : ℝ)
  (time1 time2 : ℝ)
  (h_length : length = 120)
  (h_time1 : time1 = 5)
  (h_time2 : time2 = 15)
  : (2 * length) / ((length / time1) + (length / time2)) = 7.5 := by
  sorry

end trains_cross_time_l123_12332


namespace num_integers_contains_3_and_4_l123_12326

theorem num_integers_contains_3_and_4 
  (n : ℕ) (h1 : 500 ≤ n) (h2 : n < 1000) :
  (∀ a b c : ℕ, n = 100 * a + 10 * b + c → (b = 3 ∧ c = 4) ∨ (b = 4 ∧ c = 3)) → 
  n = 10 :=
sorry

end num_integers_contains_3_and_4_l123_12326


namespace doubled_cost_percent_l123_12361

-- Definitions
variable (t b : ℝ)
def cost (b : ℝ) : ℝ := t * b^4

-- Theorem statement
theorem doubled_cost_percent :
  cost t (2 * b) = 16 * cost t b :=
by
  -- To be proved
  sorry

end doubled_cost_percent_l123_12361


namespace smallest_positive_period_of_f_max_min_values_of_f_l123_12391

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1 / 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos (2 * x))
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

--(I) Prove the smallest positive period of f(x) is π.
theorem smallest_positive_period_of_f : ∀ x : ℝ, f (x + π) = f x :=
sorry

--(II) Prove the maximum and minimum values of f(x) on [0, π / 2] are 1 and -1/2 respectively.
theorem max_min_values_of_f : ∃ max min : ℝ, (max = 1) ∧ (min = -1 / 2) ∧
  (∀ x ∈ Set.Icc 0 (π / 2), f x ≤ max ∧ f x ≥ min) :=
sorry

end smallest_positive_period_of_f_max_min_values_of_f_l123_12391


namespace area_at_stage_8_l123_12370

theorem area_at_stage_8 
  (side_length : ℕ)
  (stage : ℕ)
  (num_squares : ℕ)
  (square_area : ℕ) 
  (total_area : ℕ) 
  (h1 : side_length = 4) 
  (h2 : stage = 8) 
  (h3 : num_squares = stage) 
  (h4 : square_area = side_length * side_length) 
  (h5 : total_area = num_squares * square_area) :
  total_area = 128 :=
sorry

end area_at_stage_8_l123_12370


namespace javier_first_throw_l123_12368

theorem javier_first_throw 
  (second third first : ℝ)
  (h1 : first = 2 * second)
  (h2 : first = (1 / 2) * third)
  (h3 : first + second + third = 1050) :
  first = 300 := 
by sorry

end javier_first_throw_l123_12368


namespace min_value_expression_l123_12349

theorem min_value_expression (a b t : ℝ) (h : a + b = t) : 
  ∃ c : ℝ, c = ((a^2 + 1)^2 + (b^2 + 1)^2) → c = (t^4 + 8 * t^2 + 16) / 8 :=
by
  sorry

end min_value_expression_l123_12349


namespace x_share_for_each_rupee_w_gets_l123_12381

theorem x_share_for_each_rupee_w_gets (w_share : ℝ) (y_per_w : ℝ) (total_amount : ℝ) (a : ℝ) :
  w_share = 10 →
  y_per_w = 0.20 →
  total_amount = 15 →
  (w_share + w_share * a + w_share * y_per_w = total_amount) →
  a = 0.30 :=
by
  intros h_w h_y h_total h_eq
  sorry

end x_share_for_each_rupee_w_gets_l123_12381


namespace zero_is_a_root_of_polynomial_l123_12314

theorem zero_is_a_root_of_polynomial :
  (12 * (0 : ℝ)^4 + 38 * (0)^3 - 51 * (0)^2 + 40 * (0) = 0) :=
by simp

end zero_is_a_root_of_polynomial_l123_12314


namespace necessarily_negative_b_ab_l123_12347

theorem necessarily_negative_b_ab (a b : ℝ) (h1 : 0 < a) (h2 : a < 2) (h3 : -2 < b) (h4 : b < 0) : 
  b + a * b < 0 := by 
  sorry

end necessarily_negative_b_ab_l123_12347


namespace pete_flag_total_circles_squares_l123_12386

def US_flag_stars : ℕ := 50
def US_flag_stripes : ℕ := 13

def circles (stars : ℕ) : ℕ := (stars / 2) - 3
def squares (stripes : ℕ) : ℕ := (2 * stripes) + 6

theorem pete_flag_total_circles_squares : 
  circles US_flag_stars + squares US_flag_stripes = 54 := 
by
  unfold circles squares US_flag_stars US_flag_stripes
  sorry

end pete_flag_total_circles_squares_l123_12386


namespace false_proposition_l123_12310

open Classical

variables (a b : ℝ) (x : ℝ)

def P := ∃ (a b : ℝ), (0 < a) ∧ (0 < b) ∧ (a + b = 1) ∧ ((1 / a) + (1 / b) = 3)
def Q := ∀ (x : ℝ), x^2 - x + 1 ≥ 0

theorem false_proposition :
  (¬ P ∧ ¬ Q) = false → (¬ P ∨ ¬ Q) = true → (¬ P ∨ Q) = true → (¬ P ∧ Q) = true :=
sorry

end false_proposition_l123_12310


namespace part_I_solution_part_II_solution_l123_12329

-- Definition of the function f(x)
def f (x a : ℝ) := |x - a| + |2 * x - 1|

-- Part (I) when a = 1, find the solution set for f(x) ≤ 2
theorem part_I_solution (x : ℝ) : f x 1 ≤ 2 ↔ 0 ≤ x ∧ x ≤ 4 / 3 :=
by sorry

-- Part (II) if the solution set for f(x) ≤ |2x + 1| contains [1/2, 1], find the range of a
theorem part_II_solution (a : ℝ) :
  (∀ x : ℝ, 1 / 2 ≤ x ∧ x ≤ 1 → f x a ≤ |2 * x + 1|) → -1 ≤ a ∧ a ≤ 5 / 2 :=
by sorry

end part_I_solution_part_II_solution_l123_12329


namespace total_birds_distance_l123_12318

def birds_flew_collectively : Prop := 
  let distance_eagle := 15 * 2.5
  let distance_falcon := 46 * 2.5
  let distance_pelican := 33 * 2.5
  let distance_hummingbird := 30 * 2.5
  let distance_hawk := 45 * 3
  let distance_swallow := 25 * 1.5
  let total_distance := distance_eagle + distance_falcon + distance_pelican + distance_hummingbird + distance_hawk + distance_swallow
  total_distance = 482.5

theorem total_birds_distance : birds_flew_collectively := by
  -- proof goes here
  sorry

end total_birds_distance_l123_12318


namespace sin_value_l123_12362

theorem sin_value (α : ℝ) (h : Real.cos (π / 6 - α) = (Real.sqrt 3) / 3) :
    Real.sin (5 * π / 6 - 2 * α) = -1 / 3 :=
by
  sorry

end sin_value_l123_12362


namespace quadratics_roots_l123_12371

theorem quadratics_roots (m n : ℝ) (r₁ r₂ : ℝ) 
  (h₁ : r₁^2 - m * r₁ + n = 0) (h₂ : r₂^2 - m * r₂ + n = 0) 
  (p q : ℝ) (h₃ : (r₁^2 - r₂^2)^2 + p * (r₁^2 - r₂^2) + q = 0) :
  p = 0 ∧ q = -m^4 + 4 * m^2 * n := 
sorry

end quadratics_roots_l123_12371


namespace find_k_l123_12304

variable (c : ℝ) (k : ℝ)
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def geometric_sequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  ∀ n, a (n + 1) = c * a n

def sum_sequence (S : ℕ → ℝ) (k : ℝ) : Prop :=
  ∀ n, S n = 3^n + k

theorem find_k (c_ne_zero : c ≠ 0)
  (h_geo : geometric_sequence a c)
  (h_sum : sum_sequence S k)
  (h_a1 : a 1 = 3 + k)
  (h_a2 : a 2 = S 2 - S 1)
  (h_a3 : a 3 = S 3 - S 2) :
  k = -1 :=
sorry

end find_k_l123_12304


namespace journey_length_l123_12340

/-- Define the speed in the urban area as 55 km/h. -/
def urban_speed : ℕ := 55

/-- Define the speed on the highway as 85 km/h. -/
def highway_speed : ℕ := 85

/-- Define the time spent in each area as 3 hours. -/
def travel_time : ℕ := 3

/-- Define the distance traveled in the urban area as the product of the speed and time. -/
def urban_distance : ℕ := urban_speed * travel_time

/-- Define the distance traveled on the highway as the product of the speed and time. -/
def highway_distance : ℕ := highway_speed * travel_time

/-- Define the total distance of the journey. -/
def total_distance : ℕ := urban_distance + highway_distance

/-- The theorem that the total distance is 420 km. -/
theorem journey_length : total_distance = 420 := by
  -- Prove the equality by calculating the distances and summing them up
  sorry

end journey_length_l123_12340


namespace geometric_means_insertion_l123_12308

noncomputable def is_geometric_progression (s : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ) (r_pos : r > 0), ∀ n, s (n + 1) = s n * r

theorem geometric_means_insertion (s : ℕ → ℝ) (n : ℕ)
  (h : is_geometric_progression s)
  (h_pos : ∀ i, s i > 0) :
  ∃ t : ℕ → ℝ, is_geometric_progression t :=
sorry

end geometric_means_insertion_l123_12308


namespace oranges_after_eating_l123_12338

def initial_oranges : ℝ := 77.0
def eaten_oranges : ℝ := 2.0
def final_oranges : ℝ := 75.0

theorem oranges_after_eating :
  initial_oranges - eaten_oranges = final_oranges := by
  sorry

end oranges_after_eating_l123_12338


namespace find_a_l123_12342

theorem find_a 
  (x y a m n : ℝ)
  (h1 : x - 5 / 2 * y + 1 = 0) 
  (h2 : x = m + a) 
  (h3 : y = n + 1)  -- since k = 1, so we replace k with 1
  (h4 : m + a = m + 1 / 2) : 
  a = 1 / 2 := 
by 
  sorry

end find_a_l123_12342


namespace max_M_is_2_l123_12348

theorem max_M_is_2 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hdisc : b^2 - 4 * a * c ≥ 0) :
    max (min (b + c / a) (min (c + a / b) (a + b / c))) = 2 := by
    sorry

end max_M_is_2_l123_12348


namespace abscissa_of_A_is_3_l123_12378

-- Definitions of the points A, B, line l and conditions
def in_first_quadrant (A : ℝ × ℝ) := (A.1 > 0) ∧ (A.2 > 0)

def on_line_l (A : ℝ × ℝ) := A.2 = 2 * A.1

def point_B : ℝ × ℝ := (5, 0)

def diameter_circle (A B : ℝ × ℝ) (P : ℝ × ℝ) :=
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

-- Vectors AB and CD
def vector_AB (A B : ℝ × ℝ) := (B.1 - A.1, B.2 - A.2)

def vector_CD (C D : ℝ × ℝ) := (D.1 - C.1, D.2 - C.2)

def dot_product_zero (A B C D : ℝ × ℝ) := (vector_AB A B).1 * (vector_CD C D).1 + (vector_AB A B).2 * (vector_CD C D).2 = 0

-- Statement to prove
theorem abscissa_of_A_is_3 (A : ℝ × ℝ) (D : ℝ × ℝ) (a : ℝ) :
  in_first_quadrant A →
  on_line_l A →
  diameter_circle A point_B D →
  dot_product_zero A point_B (a, a) D →
  A.1 = 3 :=
by
  sorry

end abscissa_of_A_is_3_l123_12378


namespace combined_age_71_in_6_years_l123_12341

-- Given conditions
variable (combinedAgeIn15Years : ℕ) (h_condition : combinedAgeIn15Years = 107)

-- Define the question
def combinedAgeIn6Years : ℕ := combinedAgeIn15Years - 4 * (15 - 6)

-- State the theorem to prove the question == answer given conditions
theorem combined_age_71_in_6_years (h_condition : combinedAgeIn15Years = 107) : combinedAgeIn6Years combinedAgeIn15Years = 71 := 
by 
  sorry

end combined_age_71_in_6_years_l123_12341


namespace circle_properties_l123_12395

def circle_center_line (x y : ℝ) : Prop := x + y - 1 = 0

def point_A_on_circle (x y : ℝ) : Prop := (x, y) = (-1, 4)
def point_B_on_circle (x y : ℝ) : Prop := (x, y) = (1, 2)

def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 4

def slope_range_valid (k : ℝ) : Prop :=
  k ≤ 0 ∨ k ≥ 4 / 3

theorem circle_properties
  (x y : ℝ)
  (center_x center_y : ℝ)
  (h_center_line : circle_center_line center_x center_y)
  (h_point_A_on_circle : point_A_on_circle x y)
  (h_point_B_on_circle : point_B_on_circle x y)
  (h_circle_equation : circle_equation x y)
  (k : ℝ) :
  circle_equation center_x center_y ∧ slope_range_valid k :=
sorry

end circle_properties_l123_12395


namespace find_y_coordinate_of_P_l123_12367

-- Define the conditions as Lean definitions
def distance_x_axis_to_P (P : ℝ × ℝ) :=
  abs P.2

def distance_y_axis_to_P (P : ℝ × ℝ) :=
  abs P.1

-- Lean statement of the problem
theorem find_y_coordinate_of_P (P : ℝ × ℝ)
  (h1 : distance_x_axis_to_P P = (1/2) * distance_y_axis_to_P P)
  (h2 : distance_y_axis_to_P P = 10) :
  P.2 = 5 ∨ P.2 = -5 :=
sorry

end find_y_coordinate_of_P_l123_12367


namespace greatest_monthly_drop_in_March_l123_12345

noncomputable def jan_price_change : ℝ := -3.00
noncomputable def feb_price_change : ℝ := 1.50
noncomputable def mar_price_change : ℝ := -4.50
noncomputable def apr_price_change : ℝ := 2.00
noncomputable def may_price_change : ℝ := -1.00
noncomputable def jun_price_change : ℝ := 0.50

theorem greatest_monthly_drop_in_March :
  mar_price_change < jan_price_change ∧
  mar_price_change < feb_price_change ∧
  mar_price_change < apr_price_change ∧
  mar_price_change < may_price_change ∧
  mar_price_change < jun_price_change :=
by {
  sorry
}

end greatest_monthly_drop_in_March_l123_12345


namespace find_shirts_yesterday_l123_12336

def shirts_per_minute : ℕ := 8
def total_minutes : ℕ := 2
def shirts_today : ℕ := 3

def total_shirts : ℕ := shirts_per_minute * total_minutes
def shirts_yesterday : ℕ := total_shirts - shirts_today

theorem find_shirts_yesterday : shirts_yesterday = 13 := by
  sorry

end find_shirts_yesterday_l123_12336
