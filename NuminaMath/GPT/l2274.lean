import Mathlib

namespace evaluate_division_l2274_227410

theorem evaluate_division : 64 / 0.08 = 800 := by
  sorry

end evaluate_division_l2274_227410


namespace telethon_total_revenue_l2274_227408

noncomputable def telethon_revenue (first_period_hours : ℕ) (first_period_rate : ℕ) 
  (additional_percent_increase : ℕ) (second_period_hours : ℕ) : ℕ :=
  let first_revenue := first_period_hours * first_period_rate
  let second_period_rate := first_period_rate + (first_period_rate * additional_percent_increase / 100)
  let second_revenue := second_period_hours * second_period_rate
  first_revenue + second_revenue

theorem telethon_total_revenue : 
  telethon_revenue 12 5000 20 14 = 144000 :=
by 
  rfl -- replace 'rfl' with 'sorry' if the proof is non-trivial and longer

end telethon_total_revenue_l2274_227408


namespace product_of_y_coordinates_l2274_227433

theorem product_of_y_coordinates (k : ℝ) (hk : k > 0) :
    let y1 := 2 + Real.sqrt (k^2 - 64)
    let y2 := 2 - Real.sqrt (k^2 - 64)
    y1 * y2 = 68 - k^2 :=
by 
  sorry

end product_of_y_coordinates_l2274_227433


namespace quadratic_has_two_distinct_real_roots_l2274_227458

theorem quadratic_has_two_distinct_real_roots :
  let a := (1 : ℝ)
  let b := (-5 : ℝ)
  let c := (-1 : ℝ)
  b^2 - 4 * a * c > 0 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l2274_227458


namespace proof_problem_l2274_227444

noncomputable def p : Prop := ∃ (α : ℝ), Real.cos (Real.pi - α) = Real.cos α
def q : Prop := ∀ (x : ℝ), x ^ 2 + 1 > 0

theorem proof_problem : p ∨ q := 
by
  sorry

end proof_problem_l2274_227444


namespace sum_third_three_l2274_227477

variables {a : ℕ → ℤ}

-- Define the properties of the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

-- Given conditions
axiom sum_first_three : a 1 + a 2 + a 3 = 9
axiom sum_second_three : a 4 + a 5 + a 6 = 27
axiom arithmetic_seq : is_arithmetic_sequence a

-- The proof goal
theorem sum_third_three : a 7 + a 8 + a 9 = 45 :=
by
  sorry  -- Proof is omitted here

end sum_third_three_l2274_227477


namespace distinct_terms_in_expansion_l2274_227464

theorem distinct_terms_in_expansion:
  (∀ (x y z u v w: ℝ), (x + y + z) * (u + v + w + x + y) = 0 → false) →
  3 * 5 = 15 := by sorry

end distinct_terms_in_expansion_l2274_227464


namespace problem_l2274_227428

def Y (a b : ℤ) : ℤ := a^2 - 2 * a * b + b^2
def Z (a b : ℤ) : ℤ := a * b + a + b

theorem problem
  : Z (Y 5 3) (Y 2 1) = 9 := by
  sorry

end problem_l2274_227428


namespace probability_of_5_pieces_of_candy_l2274_227466

-- Define the conditions
def total_eggs : ℕ := 100 -- Assume total number of eggs is 100 for simplicity
def blue_eggs : ℕ := 4 * total_eggs / 5
def purple_eggs : ℕ := total_eggs / 5
def blue_eggs_with_5_candies : ℕ := blue_eggs / 4
def purple_eggs_with_5_candies : ℕ := purple_eggs / 2
def total_eggs_with_5_candies : ℕ := blue_eggs_with_5_candies + purple_eggs_with_5_candies

-- The proof problem
theorem probability_of_5_pieces_of_candy : (total_eggs_with_5_candies : ℚ) / (total_eggs : ℚ) = 3 / 10 := 
by
  sorry

end probability_of_5_pieces_of_candy_l2274_227466


namespace find_a_l2274_227430

theorem find_a (a : ℝ) (h : a^2 + a^2 / 4 = 5) : a = 2 ∨ a = -2 := 
sorry

end find_a_l2274_227430


namespace area_ratio_problem_l2274_227481

theorem area_ratio_problem
  (A B C : ℝ) -- Areas of the corresponding regions
  (m n : ℕ)  -- Given ratios
  (PQR_is_right_triangle : true)  -- PQR is a right-angled triangle (placeholder condition)
  (RSTU_is_rectangle : true)  -- RSTU is a rectangle (placeholder condition)
  (ratio_A_B : A / B = m / 2)  -- Ratio condition 1
  (ratio_A_C : A / C = n / 1)  -- Ratio condition 2
  (PTS_sim_TQU_sim_PQR : true)  -- Similar triangles (placeholder condition)
  : n = 9 := 
sorry

end area_ratio_problem_l2274_227481


namespace coin_flip_probability_l2274_227407

theorem coin_flip_probability : 
  ∀ (prob_tails : ℚ) (seq : List (Bool × ℚ)),
    prob_tails = 1/2 →
    seq = [(true, 1/2), (true, 1/2), (false, 1/2), (false, 1/2)] →
    (seq.map Prod.snd).prod = 0.0625 :=
by 
  intros prob_tails seq htails hseq 
  sorry

end coin_flip_probability_l2274_227407


namespace part1_part2_l2274_227402

noncomputable def total_seating_arrangements : ℕ := 840
noncomputable def non_adjacent_4_people_arrangements : ℕ := 24
noncomputable def three_empty_adjacent_arrangements : ℕ := 120

theorem part1 : total_seating_arrangements - non_adjacent_4_people_arrangements = 816 := by
  sorry

theorem part2 : total_seating_arrangements - three_empty_adjacent_arrangements = 720 := by
  sorry

end part1_part2_l2274_227402


namespace probability_of_non_defective_is_seven_ninetyninths_l2274_227414

-- Define the number of total pencils, defective pencils, and the number of pencils selected
def total_pencils : ℕ := 12
def defective_pencils : ℕ := 4
def selected_pencils : ℕ := 5

-- Define the number of ways to choose k elements from n elements (the combination function)
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculate the total number of ways to choose 5 pencils out of 12
def total_ways : ℕ := combination total_pencils selected_pencils

-- Calculate the number of non-defective pencils
def non_defective_pencils : ℕ := total_pencils - defective_pencils

-- Calculate the number of ways to choose 5 non-defective pencils out of 8
def non_defective_ways : ℕ := combination non_defective_pencils selected_pencils

-- Calculate the probability that all 5 chosen pencils are non-defective
def probability_non_defective : ℚ :=
  non_defective_ways / total_ways

-- Prove that this probability equals 7/99
theorem probability_of_non_defective_is_seven_ninetyninths :
  probability_non_defective = 7 / 99 :=
by
  -- The proof is left as an exercise
  sorry

end probability_of_non_defective_is_seven_ninetyninths_l2274_227414


namespace value_of_a_minus_b_l2274_227449

theorem value_of_a_minus_b 
  (a b : ℤ)
  (h1 : 1010 * a + 1014 * b = 1018)
  (h2 : 1012 * a + 1016 * b = 1020) : 
  a - b = -3 :=
sorry

end value_of_a_minus_b_l2274_227449


namespace proportionate_enlargement_l2274_227455

theorem proportionate_enlargement 
  (original_width original_height new_width : ℕ)
  (h_orig_width : original_width = 3)
  (h_orig_height : original_height = 2)
  (h_new_width : new_width = 12) : 
  ∃ (new_height : ℕ), new_height = 8 :=
by
  -- sorry to skip proof
  sorry

end proportionate_enlargement_l2274_227455


namespace exists_increasing_sequences_l2274_227496

theorem exists_increasing_sequences (a : ℕ → ℕ) (b : ℕ → ℕ) :
  (∀ n : ℕ, a n < a (n + 1)) ∧ (∀ n : ℕ, b n < b (n + 1)) ∧
  (∀ n : ℕ, a n * (a n + 1) ∣ b n ^ 2 + 1) :=
sorry

end exists_increasing_sequences_l2274_227496


namespace calculate_expression_l2274_227401

theorem calculate_expression :
  (1.99^2 - 1.98 * 1.99 + 0.99^2 = 1) :=
by
  sorry

end calculate_expression_l2274_227401


namespace least_perimeter_of_triangle_l2274_227483

-- Define the sides of the triangle
def side1 : ℕ := 40
def side2 : ℕ := 48

-- Given condition for the third side
def valid_third_side (x : ℕ) : Prop :=
  8 < x ∧ x < 88

-- The least possible perimeter given the conditions
def least_possible_perimeter : ℕ :=
  side1 + side2 + 9

theorem least_perimeter_of_triangle (x : ℕ) (h : valid_third_side x) (hx : x = 9) : least_possible_perimeter = 97 :=
by
  rw [least_possible_perimeter]
  exact rfl

end least_perimeter_of_triangle_l2274_227483


namespace smallest_angle_range_l2274_227493

theorem smallest_angle_range {A B C : ℝ} (hA : 0 < A) (hABC : A + B + C = 180) (horder : A ≤ B ∧ B ≤ C) :
  0 < A ∧ A ≤ 60 := by
  sorry

end smallest_angle_range_l2274_227493


namespace pot_filling_time_l2274_227415

-- Define the given conditions
def drops_per_minute : ℕ := 3
def volume_per_drop : ℕ := 20 -- in ml
def pot_capacity : ℕ := 3000 -- in ml (3 liters * 1000 ml/liter)

-- Define the calculation for the drip rate
def drip_rate_per_minute : ℕ := drops_per_minute * volume_per_drop

-- Define the goal, i.e., how long it will take to fill the pot
def time_to_fill_pot (capacity : ℕ) (rate : ℕ) : ℕ := capacity / rate

-- Proof statement
theorem pot_filling_time :
  time_to_fill_pot pot_capacity drip_rate_per_minute = 50 := 
sorry

end pot_filling_time_l2274_227415


namespace girls_ratio_correct_l2274_227419

-- Define the number of total attendees
def total_attendees : ℕ := 100

-- Define the percentage of faculty and staff
def faculty_staff_percentage : ℕ := 10

-- Define the number of boys among the students
def number_of_boys : ℕ := 30

-- Define the function to calculate the number of faculty and staff
def faculty_staff (total_attendees faculty_staff_percentage: ℕ) : ℕ :=
  (faculty_staff_percentage * total_attendees) / 100

-- Define the function to calculate the number of students
def number_of_students (total_attendees faculty_staff_percentage: ℕ) : ℕ :=
  total_attendees - faculty_staff total_attendees faculty_staff_percentage

-- Define the function to calculate the number of girls
def number_of_girls (total_attendees faculty_staff_percentage number_of_boys: ℕ) : ℕ :=
  number_of_students total_attendees faculty_staff_percentage - number_of_boys

-- Define the function to calculate the ratio of girls to the remaining attendees
def ratio_girls_to_attendees (total_attendees faculty_staff_percentage number_of_boys: ℕ) : ℚ :=
  (number_of_girls total_attendees faculty_staff_percentage number_of_boys) / 
  (number_of_students total_attendees faculty_staff_percentage)

-- The theorem statement that needs to be proven (no proof required)
theorem girls_ratio_correct : ratio_girls_to_attendees total_attendees faculty_staff_percentage number_of_boys = 2 / 3 := 
by 
  -- The proof is skipped.
  sorry

end girls_ratio_correct_l2274_227419


namespace probability_all_yellow_l2274_227478

-- Definitions and conditions
def total_apples : ℕ := 8
def red_apples : ℕ := 5
def yellow_apples : ℕ := 3
def chosen_apples : ℕ := 3

-- Theorem to prove
theorem probability_all_yellow :
  (yellow_apples.choose chosen_apples : ℚ) / (total_apples.choose chosen_apples) = 1 / 56 := sorry

end probability_all_yellow_l2274_227478


namespace quadratic_inequality_l2274_227403

noncomputable def ax2_plus_bx_c (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |ax2_plus_bx_c a b c x| ≤ 1 / 2) →
  ∀ x : ℝ, |x| ≥ 1 → |ax2_plus_bx_c a b c x| ≤ x^2 - 1 / 2 :=
by
  sorry

end quadratic_inequality_l2274_227403


namespace least_positive_integer_l2274_227459

theorem least_positive_integer (x : ℕ) :
  (∃ k : ℤ, (3 * x + 41) ^ 2 = 53 * k) ↔ x = 4 :=
by
  sorry

end least_positive_integer_l2274_227459


namespace perimeter_of_square_is_64_l2274_227440

noncomputable def side_length_of_square (s : ℝ) :=
  let rect_height := s
  let rect_width := s / 4
  let perimeter_of_rectangle := 2 * (rect_height + rect_width)
  perimeter_of_rectangle = 40

theorem perimeter_of_square_is_64 (s : ℝ) (h1 : side_length_of_square s) : 4 * s = 64 :=
by
  sorry

end perimeter_of_square_is_64_l2274_227440


namespace number_of_people_who_purchased_only_book_A_l2274_227421

theorem number_of_people_who_purchased_only_book_A (x y v : ℕ) 
  (h1 : 2 * x = 500)
  (h2 : y = x + 500)
  (h3 : v = 2 * y) : 
  v = 1500 := 
sorry

end number_of_people_who_purchased_only_book_A_l2274_227421


namespace find_original_number_l2274_227438

theorem find_original_number (x : ℤ) : 4 * (3 * x + 29) = 212 → x = 8 :=
by
  intro h
  sorry

end find_original_number_l2274_227438


namespace range_of_a_l2274_227413

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*x - a
noncomputable def g (x : ℝ) : ℝ := 2*x + 2 * Real.log x
noncomputable def h (x : ℝ) : ℝ := x^2 - 2 * Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x y, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 ∧ (1 / Real.exp 1) ≤ y ∧ y ≤ Real.exp 1 ∧ f x a = g x ∧ f y a = g y → x ≠ y) →
  1 < a ∧ a ≤ (1 / Real.exp 2) + 2 :=
sorry

end range_of_a_l2274_227413


namespace mathieu_plot_area_l2274_227424

def total_area (x y : ℕ) : ℕ := x * x

theorem mathieu_plot_area :
  ∃ (x y : ℕ), (x^2 - y^2 = 464) ∧ (x - y = 8) ∧ (total_area x y = 1089) :=
by sorry

end mathieu_plot_area_l2274_227424


namespace binom_n_n_minus_1_l2274_227465

theorem binom_n_n_minus_1 (n : ℕ) (h : 0 < n) : (Nat.choose n (n-1)) = n :=
  sorry

end binom_n_n_minus_1_l2274_227465


namespace opposite_of_neg23_eq_23_reciprocal_of_neg23_eq_neg_1_div_23_abs_value_of_neg23_eq_23_l2274_227488

theorem opposite_of_neg23_eq_23 : -(-23) = 23 := 
by sorry

theorem reciprocal_of_neg23_eq_neg_1_div_23 : (1 : ℚ) / (-23) = -(1 / 23 : ℚ) :=
by sorry

theorem abs_value_of_neg23_eq_23 : abs (-23) = 23 :=
by sorry

end opposite_of_neg23_eq_23_reciprocal_of_neg23_eq_neg_1_div_23_abs_value_of_neg23_eq_23_l2274_227488


namespace scientific_notation_l2274_227467

theorem scientific_notation : 899000 = 8.99 * 10^5 := 
by {
  -- We start by recognizing that we need to express 899,000 in scientific notation.
  -- Placing the decimal point after the first non-zero digit yields 8.99.
  -- Count the number of places moved (5 places to the left).
  -- Thus, 899,000 in scientific notation is 8.99 * 10^5.
  sorry
}

end scientific_notation_l2274_227467


namespace jerry_sister_increase_temp_l2274_227448

theorem jerry_sister_increase_temp :
  let T0 := 40
  let T1 := 2 * T0
  let T2 := T1 - 30
  let T3 := T2 - 0.3 * T2
  let T4 := 59
  T4 - T3 = 24 := by
  sorry

end jerry_sister_increase_temp_l2274_227448


namespace tan_alpha_frac_simplification_l2274_227480

theorem tan_alpha_frac_simplification (α : ℝ) (h : Real.tan α = -1 / 2) : 
  (2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = 4 / 3 :=
by sorry

end tan_alpha_frac_simplification_l2274_227480


namespace base_b_square_of_integer_l2274_227457

theorem base_b_square_of_integer (b : ℕ) (h : b > 4) : ∃ n : ℕ, (n * n) = b^2 + 4 * b + 4 :=
by 
  sorry

end base_b_square_of_integer_l2274_227457


namespace intersection_points_l2274_227494

-- Define parameters: number of sides for each polygon
def n₆ := 6
def n₇ := 7
def n₈ := 8
def n₉ := 9

-- Condition: polygons are inscribed in the same circle, no shared vertices, no three sides intersect at a common point
def polygons_are_disjoint (n₁ n₂ : ℕ) (n₃ n₄ : ℕ) (n₅ : ℕ) : Prop :=
  true -- Assume this is a primitive condition encapsulating given constraints

-- Prove the number of intersection points is 80
theorem intersection_points : polygons_are_disjoint n₆ n₇ n₈ n₉ n₅ → 
  2 * (n₆ + n₇ + n₇ + n₈) + 2 * (n₇ + n₈) + 2 * n₉ = 80 :=
by  
  sorry

end intersection_points_l2274_227494


namespace max_ab_min_reciprocal_sum_l2274_227497

noncomputable section

-- Definitions for conditions
def is_positive_real (x : ℝ) : Prop := x > 0

def condition (a b : ℝ) : Prop := is_positive_real a ∧ is_positive_real b ∧ (a + 10 * b = 1)

-- Maximum value of ab
theorem max_ab (a b : ℝ) (h : condition a b) : a * b ≤ 1 / 40 :=
sorry

-- Minimum value of 1/a + 1/b
theorem min_reciprocal_sum (a b : ℝ) (h : condition a b) : 1 / a + 1 / b ≥ 11 + 2 * Real.sqrt 10 :=
sorry

end max_ab_min_reciprocal_sum_l2274_227497


namespace tara_spent_more_on_icecream_l2274_227451

def iceCreamCount : ℕ := 19
def yoghurtCount : ℕ := 4
def iceCreamCost : ℕ := 7
def yoghurtCost : ℕ := 1

theorem tara_spent_more_on_icecream :
  (iceCreamCount * iceCreamCost) - (yoghurtCount * yoghurtCost) = 129 := 
  sorry

end tara_spent_more_on_icecream_l2274_227451


namespace no_solutions_for_a3_plus_5b3_eq_2016_l2274_227412

theorem no_solutions_for_a3_plus_5b3_eq_2016 (a b : ℤ) : a^3 + 5 * b^3 ≠ 2016 :=
by sorry

end no_solutions_for_a3_plus_5b3_eq_2016_l2274_227412


namespace area_of_triangle_with_sides_13_12_5_l2274_227462

theorem area_of_triangle_with_sides_13_12_5 :
  let a := 13
  let b := 12
  let c := 5
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area = 30 :=
by
  let a := 13
  let b := 12
  let c := 5
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  sorry

end area_of_triangle_with_sides_13_12_5_l2274_227462


namespace total_weight_correct_l2274_227445

-- Define the weights given in the problem
def dog_weight_kg := 2 -- weight in kilograms
def dog_weight_g := 600 -- additional grams
def cat_weight_g := 3700 -- weight in grams

-- Convert dog's weight to grams
def dog_weight_total_g : ℕ := dog_weight_kg * 1000 + dog_weight_g

-- Define the total weight of the animals (dog + cat)
def total_weight_animals_g : ℕ := dog_weight_total_g + cat_weight_g

-- Theorem stating that the total weight of the animals is 6300 grams
theorem total_weight_correct : total_weight_animals_g = 6300 := by
  sorry

end total_weight_correct_l2274_227445


namespace non_neg_sum_of_squares_l2274_227489

theorem non_neg_sum_of_squares (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (h : a + b + c = 1) :
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 :=
by
  sorry

end non_neg_sum_of_squares_l2274_227489


namespace expected_value_correct_prob_abs_diff_ge_1_correct_l2274_227473

/-- Probability distribution for a single die roll -/
def prob_score (n : ℕ) : ℚ :=
  if n = 1 then 1/2 else if n = 2 then 1/3 else if n = 3 then 1/6 else 0

/-- Expected value based on the given probability distribution -/
def expected_value : ℚ := 
  (1 * prob_score 1) + (2 * prob_score 2) + (3 * prob_score 3)

/-- Proving the expected value calculation -/
theorem expected_value_correct : expected_value = 7/6 :=
  by sorry

/-- Calculate the probability of score difference being at least 1 between two players -/
def prob_abs_diff_ge_1 (x y : ℕ) : ℚ :=
  -- Implementation would involve detailed probability combinations that result in diff >= 1
  sorry

/-- Prove the probability of |x - y| being at least 1 -/
theorem prob_abs_diff_ge_1_correct : 
  ∀ (x y : ℕ), prob_abs_diff_ge_1 x y < 1 :=
  by sorry

end expected_value_correct_prob_abs_diff_ge_1_correct_l2274_227473


namespace additive_inverse_of_half_l2274_227479

theorem additive_inverse_of_half :
  - (1 / 2) = -1 / 2 :=
by
  sorry

end additive_inverse_of_half_l2274_227479


namespace cookies_and_sugar_needed_l2274_227450

-- Definitions derived from the conditions
def initial_cookies : ℕ := 24
def initial_flour : ℕ := 3
def initial_sugar : ℝ := 1.5
def flour_needed : ℕ := 5

-- The proof statement
theorem cookies_and_sugar_needed :
  (initial_cookies / initial_flour) * flour_needed = 40 ∧ (initial_sugar / initial_flour) * flour_needed = 2.5 :=
by
  sorry

end cookies_and_sugar_needed_l2274_227450


namespace total_money_l2274_227435

theorem total_money (A B C : ℕ) (h1 : A + C = 400) (h2 : B + C = 750) (hC : C = 250) :
  A + B + C = 900 :=
sorry

end total_money_l2274_227435


namespace smallest_positive_x_l2274_227436

theorem smallest_positive_x (x : ℕ) (h : 42 * x + 9 ≡ 3 [MOD 15]) : x = 2 :=
sorry

end smallest_positive_x_l2274_227436


namespace graph_of_equation_l2274_227498

theorem graph_of_equation (x y : ℝ) : (x - y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 :=
by sorry

end graph_of_equation_l2274_227498


namespace ratio_A_B_l2274_227425

-- Given conditions as definitions
def P_both : ℕ := 500  -- Number of people who purchased both books A and B

def P_only_B : ℕ := P_both / 2  -- Number of people who purchased only book B

def P_only_A : ℕ := 1000  -- Number of people who purchased only book A

-- Total number of people who purchased books
def P_A : ℕ := P_only_A + P_both  -- Total number of people who purchased book A

def P_B : ℕ := P_only_B + P_both  -- Total number of people who purchased book B

-- The ratio of people who purchased book A to book B
theorem ratio_A_B : P_A / P_B = 2 :=
by
  sorry

end ratio_A_B_l2274_227425


namespace part1_part2_l2274_227429

variable (a b c : ℝ)

-- Conditions
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : c > 0
axiom h4 : a^2 + b^2 + 4*c^2 = 3

-- Part 1: Prove that a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 := sorry

-- Part 2: Given b = 2c, prove that 1/a + 1/c ≥ 3
axiom h5 : b = 2*c
theorem part2 : 1/a + 1/c ≥ 3 := sorry

end part1_part2_l2274_227429


namespace seventeen_power_seven_mod_eleven_l2274_227454

-- Define the conditions
def mod_condition : Prop := 17 % 11 = 6

-- Define the main goal (to prove the correct answer)
theorem seventeen_power_seven_mod_eleven (h : mod_condition) : (17^7) % 11 = 8 := by
  -- Proof goes here
  sorry

end seventeen_power_seven_mod_eleven_l2274_227454


namespace bigger_part_l2274_227484

theorem bigger_part (x y : ℕ) (h1 : x + y = 54) (h2 : 10 * x + 22 * y = 780) : y = 34 :=
sorry

end bigger_part_l2274_227484


namespace triangle_existence_condition_l2274_227472

theorem triangle_existence_condition 
  (a b f_c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : f_c > 0) : 
  (2 * a * b / (a + b)) > f_c :=
sorry

end triangle_existence_condition_l2274_227472


namespace range_of_b_l2274_227416

noncomputable def f (x b : ℝ) : ℝ := -1/2 * (x - 2)^2 + b * Real.log (x + 2)
noncomputable def derivative (x b : ℝ) := -(x - 2) + b / (x + 2)

-- Lean theorem statement
theorem range_of_b (b : ℝ) :
  (∀ x > 1, derivative x b ≤ 0) → b ≤ -3 :=
by
  sorry

end range_of_b_l2274_227416


namespace average_is_correct_l2274_227491

theorem average_is_correct (x : ℝ) : 
  (2 * x + 12 + 3 * x + 3 + 5 * x - 8) / 3 = 3 * x + 2 → x = -1 :=
by
  sorry

end average_is_correct_l2274_227491


namespace trains_pass_each_other_time_l2274_227442

theorem trains_pass_each_other_time :
  ∃ t : ℝ, t = 240 / 191.171 := 
sorry

end trains_pass_each_other_time_l2274_227442


namespace bobby_pays_correct_amount_l2274_227490

noncomputable def bobby_total_cost : ℝ := 
  let mold_cost : ℝ := 250
  let material_original_cost : ℝ := 150
  let material_discount : ℝ := 0.20 * material_original_cost
  let material_cost : ℝ := material_original_cost - material_discount
  let hourly_rate_original : ℝ := 75
  let hourly_rate_increased : ℝ := hourly_rate_original + 10
  let work_hours : ℝ := 8
  let work_cost_original : ℝ := work_hours * hourly_rate_increased
  let work_cost_discount : ℝ := 0.80 * work_cost_original
  let cost_before_tax : ℝ := mold_cost + material_cost + work_cost_discount
  let tax : ℝ := 0.10 * cost_before_tax
  cost_before_tax + tax

theorem bobby_pays_correct_amount : bobby_total_cost = 1005.40 := sorry

end bobby_pays_correct_amount_l2274_227490


namespace non_isosceles_triangle_has_equidistant_incenter_midpoints_l2274_227418

structure Triangle (α : Type*) :=
(a b c : α)
(incenter : α)
(midpoint_a_b : α)
(midpoint_b_c : α)
(midpoint_c_a : α)
(equidistant : Bool)
(non_isosceles : Bool)

-- Define the triangle with the specified properties.
noncomputable def counterexample_triangle : Triangle ℝ :=
{ a := 3,
  b := 4,
  c := 5, 
  incenter := 1, -- incenter length for the right triangle.
  midpoint_a_b := 2.5,
  midpoint_b_c := 2,
  midpoint_c_a := 1.5,
  equidistant := true,    -- midpoints of two sides are equidistant from incenter
  non_isosceles := true } -- the triangle is not isosceles

theorem non_isosceles_triangle_has_equidistant_incenter_midpoints :
  ∃ (T : Triangle ℝ), T.equidistant ∧ T.non_isosceles := by
  use counterexample_triangle
  sorry

end non_isosceles_triangle_has_equidistant_incenter_midpoints_l2274_227418


namespace truck_distance_and_efficiency_l2274_227482

theorem truck_distance_and_efficiency (m d g1 g2 : ℕ) (h1 : d = 300) (h2 : g1 = 10) (h3 : g2 = 15) :
  (d * (g2 / g1) = 450) ∧ (d / g1 = 30) :=
by
  sorry

end truck_distance_and_efficiency_l2274_227482


namespace percentage_increase_of_soda_l2274_227420

variable (C S x : ℝ)

theorem percentage_increase_of_soda
  (h1 : 1.25 * C = 10)
  (h2 : S + x * S = 12)
  (h3 : C + S = 16) :
  x = 0.5 :=
sorry

end percentage_increase_of_soda_l2274_227420


namespace hannah_final_pay_l2274_227474

theorem hannah_final_pay : (30 * 18) - (5 * 3) + (15 * 4) - (((30 * 18) - (5 * 3) + (15 * 4)) * 0.10 + ((30 * 18) - (5 * 3) + (15 * 4)) * 0.05) = 497.25 :=
by
  sorry

end hannah_final_pay_l2274_227474


namespace converse_implication_l2274_227471

theorem converse_implication (a : ℝ) : (a^2 = 1 → a = 1) → (a = 1 → a^2 = 1) :=
sorry

end converse_implication_l2274_227471


namespace triangle_properties_l2274_227463

theorem triangle_properties (b c : ℝ) (C : ℝ)
  (hb : b = 10)
  (hc : c = 5 * Real.sqrt 6)
  (hC : C = Real.pi / 3) :
  let R := c / (2 * Real.sin C)
  let B := Real.arcsin (b * Real.sin C / c)
  R = 5 * Real.sqrt 2 ∧ B = Real.pi / 4 :=
by
  sorry

end triangle_properties_l2274_227463


namespace flashes_in_fraction_of_hour_l2274_227432

-- Definitions for the conditions
def flash_interval : ℕ := 6       -- The light flashes every 6 seconds
def hour_in_seconds : ℕ := 3600 -- There are 3600 seconds in an hour
def fraction_of_hour : ℚ := 3/4 -- ¾ of an hour

-- The translated proof problem statement in Lean
theorem flashes_in_fraction_of_hour (interval : ℕ) (sec_in_hour : ℕ) (fraction : ℚ) :
  interval = flash_interval →
  sec_in_hour = hour_in_seconds →
  fraction = fraction_of_hour →
  (fraction * sec_in_hour) / interval = 450 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end flashes_in_fraction_of_hour_l2274_227432


namespace money_left_correct_l2274_227431

-- Define the initial amount of money John had
def initial_money : ℝ := 10.50

-- Define the amount spent on sweets
def sweets_cost : ℝ := 2.25

-- Define the amount John gave to each friend
def gift_per_friend : ℝ := 2.20

-- Define the total number of friends
def number_of_friends : ℕ := 2

-- Calculate the total gifts given to friends
def total_gifts := gift_per_friend * (number_of_friends : ℝ)

-- Calculate the total amount spent
def total_spent := sweets_cost + total_gifts

-- Define the amount of money left
def money_left := initial_money - total_spent

-- The theorem statement
theorem money_left_correct : money_left = 3.85 := 
by 
  sorry

end money_left_correct_l2274_227431


namespace quadratic_to_vertex_form_l2274_227461

theorem quadratic_to_vertex_form :
  ∀ (x : ℝ), (1/2) * x^2 - 2 * x + 1 = (1/2) * (x - 2)^2 - 1 :=
by
  intro x
  -- full proof omitted
  sorry

end quadratic_to_vertex_form_l2274_227461


namespace billy_buys_bottle_l2274_227460

-- Definitions of costs and volumes
def money : ℝ := 10
def cost1 : ℝ := 1
def volume1 : ℝ := 10
def cost2 : ℝ := 2
def volume2 : ℝ := 16
def cost3 : ℝ := 2.5
def volume3 : ℝ := 25
def cost4 : ℝ := 5
def volume4 : ℝ := 50
def cost5 : ℝ := 10
def volume5 : ℝ := 200

-- Statement of the proof problem
theorem billy_buys_bottle : ∃ b : ℕ, b = 1 ∧ cost5 = money := by 
  sorry

end billy_buys_bottle_l2274_227460


namespace no_valid_pairs_l2274_227492

theorem no_valid_pairs : ∀ (m n : ℕ), m ≥ n → m^2 - n^2 = 150 → false :=
by sorry

end no_valid_pairs_l2274_227492


namespace IntersectionOfAandB_l2274_227452

def setA : Set ℝ := {x | x < 5}
def setB : Set ℝ := {x | -1 < x}

theorem IntersectionOfAandB : setA ∩ setB = {x | -1 < x ∧ x < 5} :=
sorry

end IntersectionOfAandB_l2274_227452


namespace average_weight_l2274_227476

theorem average_weight 
  (n₁ n₂ : ℕ) 
  (avg₁ avg₂ total_avg : ℚ) 
  (h₁ : n₁ = 24) 
  (h₂ : n₂ = 8)
  (h₃ : avg₁ = 50.25)
  (h₄ : avg₂ = 45.15)
  (h₅ : total_avg = 48.975) :
  ( (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = total_avg ) :=
sorry

end average_weight_l2274_227476


namespace aftershave_alcohol_concentration_l2274_227409

def initial_volume : ℝ := 12
def initial_concentration : ℝ := 0.60
def desired_concentration : ℝ := 0.40
def water_added : ℝ := 6
def final_volume : ℝ := initial_volume + water_added

theorem aftershave_alcohol_concentration :
  initial_concentration * initial_volume = desired_concentration * final_volume :=
by
  sorry

end aftershave_alcohol_concentration_l2274_227409


namespace total_floors_l2274_227437

theorem total_floors (P Q R S T X F : ℕ) (h1 : 1 < X) (h2 : X < 50) :
  F = 1 + P - Q + R - S + T + X :=
sorry

end total_floors_l2274_227437


namespace teacher_problems_remaining_l2274_227439

theorem teacher_problems_remaining (problems_per_worksheet : Nat) 
                                   (total_worksheets : Nat) 
                                   (graded_worksheets : Nat) 
                                   (remaining_problems : Nat)
  (h1 : problems_per_worksheet = 4)
  (h2 : total_worksheets = 9)
  (h3 : graded_worksheets = 5)
  (h4 : remaining_problems = total_worksheets * problems_per_worksheet - graded_worksheets * problems_per_worksheet) :
  remaining_problems = 16 :=
sorry

end teacher_problems_remaining_l2274_227439


namespace tangent_y_intercept_range_l2274_227446

theorem tangent_y_intercept_range :
  ∀ (x₀ : ℝ), (∃ y₀ : ℝ, y₀ = Real.exp x₀ ∧ (∃ m : ℝ, m = Real.exp x₀ ∧ ∃ b : ℝ, b = Real.exp x₀ * (1 - x₀) ∧ b < 0)) → x₀ > 1 := by
  sorry

end tangent_y_intercept_range_l2274_227446


namespace find_some_number_l2274_227406

theorem find_some_number (some_number q x y : ℤ) 
  (h1 : x = some_number + 2 * q) 
  (h2 : y = 4 * q + 41) 
  (h3 : q = 7) 
  (h4 : x = y) : 
  some_number = 55 := 
by 
  sorry

end find_some_number_l2274_227406


namespace find_y_intercept_l2274_227426

theorem find_y_intercept (m : ℝ) 
  (h1 : ∀ x y : ℝ, y = 2 * x + m)
  (h2 : ∃ x y : ℝ, x = 1 ∧ y = 1 ∧ y = 2 * x + m) : 
  m = -1 := 
sorry

end find_y_intercept_l2274_227426


namespace distribute_candies_l2274_227427

theorem distribute_candies (n : ℕ) (h : ∃ m : ℕ, n = 2^m) : 
  ∀ k : ℕ, ∃ i : ℕ, (1 / 2) * i * (i + 1) % n = k :=
sorry

end distribute_candies_l2274_227427


namespace harly_dogs_final_count_l2274_227443

theorem harly_dogs_final_count (initial_dogs : ℕ) (adopted_percentage : ℕ) (returned_dogs : ℕ) (adoption_rate : adopted_percentage = 40) (initial_count : initial_dogs = 80) (returned_count : returned_dogs = 5) :
  initial_dogs - (initial_dogs * adopted_percentage / 100) + returned_dogs = 53 :=
by
  sorry

end harly_dogs_final_count_l2274_227443


namespace weight_of_mixture_correct_l2274_227486

-- Defining the fractions of each component in the mixture
def sand_fraction : ℚ := 2 / 9
def water_fraction : ℚ := 5 / 18
def gravel_fraction : ℚ := 1 / 6
def cement_fraction : ℚ := 7 / 36
def limestone_fraction : ℚ := 1 - sand_fraction - water_fraction - gravel_fraction - cement_fraction

-- Given weight of limestone
def limestone_weight : ℚ := 12

-- Total weight of the mixture that we need to prove
def total_mixture_weight : ℚ := 86.4

-- Proof problem statement
theorem weight_of_mixture_correct : (limestone_fraction * total_mixture_weight = limestone_weight) :=
by
  have h_sand := sand_fraction
  have h_water := water_fraction
  have h_gravel := gravel_fraction
  have h_cement := cement_fraction
  have h_limestone := limestone_fraction
  have h_limestone_weight := limestone_weight
  have h_total_weight := total_mixture_weight
  sorry

end weight_of_mixture_correct_l2274_227486


namespace taxi_ride_cost_l2274_227447

def baseFare : ℝ := 1.50
def costPerMile : ℝ := 0.25
def milesTraveled : ℕ := 5
def totalCost := baseFare + (costPerMile * milesTraveled)

/-- The cost of a 5-mile taxi ride is $2.75. -/
theorem taxi_ride_cost : totalCost = 2.75 := by
  sorry

end taxi_ride_cost_l2274_227447


namespace difference_in_circumferences_l2274_227487

def r_inner : ℝ := 25
def r_outer : ℝ := r_inner + 15

theorem difference_in_circumferences : 2 * Real.pi * r_outer - 2 * Real.pi * r_inner = 30 * Real.pi := by
  sorry

end difference_in_circumferences_l2274_227487


namespace telephone_call_duration_l2274_227468

theorem telephone_call_duration (x : ℝ) :
  (0.60 + 0.06 * (x - 4) = 0.08 * x) → x = 18 :=
by
  sorry

end telephone_call_duration_l2274_227468


namespace initial_peanuts_l2274_227417

-- Definitions based on conditions
def peanuts_added := 8
def total_peanuts_now := 12

-- Statement to prove
theorem initial_peanuts (initial_peanuts : ℕ) (h : initial_peanuts + peanuts_added = total_peanuts_now) : initial_peanuts = 4 :=
sorry

end initial_peanuts_l2274_227417


namespace average_study_difference_is_6_l2274_227404

def study_time_differences : List ℤ := [15, -5, 25, -10, 40, -30, 10]

def total_sum (lst : List ℤ) : ℤ := lst.foldr (· + ·) 0

def number_of_days : ℤ := 7

def average_difference : ℤ := total_sum study_time_differences / number_of_days

theorem average_study_difference_is_6 : average_difference = 6 :=
by
  unfold average_difference
  unfold total_sum 
  sorry

end average_study_difference_is_6_l2274_227404


namespace length_of_train_is_correct_l2274_227453

noncomputable def length_of_train (speed : ℕ) (time : ℕ) : ℕ :=
  (speed * (time / 3600) * 1000)

theorem length_of_train_is_correct : length_of_train 70 36 = 700 := by
  sorry

end length_of_train_is_correct_l2274_227453


namespace one_hundred_fifty_sixth_digit_is_five_l2274_227475

def repeated_sequence := [0, 6, 0, 5, 1, 3]
def target_index := 156 - 1
def block_length := repeated_sequence.length

theorem one_hundred_fifty_sixth_digit_is_five :
  repeated_sequence[target_index % block_length] = 5 :=
by
  sorry

end one_hundred_fifty_sixth_digit_is_five_l2274_227475


namespace age_sum_l2274_227411

variable (b : ℕ)
variable (a : ℕ := b + 2)
variable (c : ℕ := b / 2)

theorem age_sum : b = 10 → a + b + c = 27 :=
by
  intros h
  rw [h]
  sorry

end age_sum_l2274_227411


namespace counting_numbers_leave_remainder_6_divide_53_l2274_227434

theorem counting_numbers_leave_remainder_6_divide_53 :
  ∃! n : ℕ, (∃ k : ℕ, 53 = n * k + 6) ∧ n > 6 :=
sorry

end counting_numbers_leave_remainder_6_divide_53_l2274_227434


namespace find_number_l2274_227456

theorem find_number
  (P : ℝ) (R : ℝ) (hP : P = 0.0002) (hR : R = 2.4712) :
  (12356 * P = R) := by
  sorry

end find_number_l2274_227456


namespace max_writers_and_editors_l2274_227495

theorem max_writers_and_editors (T W : ℕ) (E : ℕ) (x : ℕ) (hT : T = 100) (hW : W = 35) (hE : E > 38) (h_comb : W + E + x = T)
    (h_neither : T = W + E + x) : x = 26 := by
  sorry

end max_writers_and_editors_l2274_227495


namespace remainder_div_x_minus_4_l2274_227485

def f (x : ℕ) : ℕ := x^5 - 8 * x^4 + 16 * x^3 + 25 * x^2 - 50 * x + 24

theorem remainder_div_x_minus_4 : 
  (f 4) = 224 := 
by 
  -- Proof goes here
  sorry

end remainder_div_x_minus_4_l2274_227485


namespace min_value_f_l2274_227423

noncomputable def f (x : ℝ) : ℝ := x^2 / (x - 10)

theorem min_value_f (h : ∀ x > 10, f x ≥ 40) : ∀ x > 10, f x = 40 → x = 20 :=
by
  sorry

end min_value_f_l2274_227423


namespace binom_2n_2_eq_n_2n_minus_1_l2274_227499

theorem binom_2n_2_eq_n_2n_minus_1 (n : ℕ) (h : n > 0) : 
  (Nat.choose (2 * n) 2) = n * (2 * n - 1) := 
sorry

end binom_2n_2_eq_n_2n_minus_1_l2274_227499


namespace exists_city_reaching_all_l2274_227400

variables {City : Type} (canReach : City → City → Prop)

-- Conditions from the problem
axiom reach_itself (A : City) : canReach A A
axiom reach_transitive {A B C : City} : canReach A B → canReach B C → canReach A C
axiom reach_any_two {P Q : City} : ∃ R : City, canReach R P ∧ canReach R Q

-- The proof problem
theorem exists_city_reaching_all (cities : City → Prop) :
  (∀ P Q, P ≠ Q → cities P → cities Q → ∃ R, cities R ∧ canReach R P ∧ canReach R Q) →
  ∃ C, ∀ A, cities A → canReach C A :=
by
  intros H
  sorry

end exists_city_reaching_all_l2274_227400


namespace hari_joined_after_5_months_l2274_227470

noncomputable def praveen_investment := 3780 * 12
noncomputable def hari_investment (x : ℕ) := 9720 * (12 - x)

theorem hari_joined_after_5_months :
  ∃ (x : ℕ), (praveen_investment : ℝ) / (hari_investment x) = (2:ℝ) / 3 ∧ x = 5 :=
by {
  sorry
}

end hari_joined_after_5_months_l2274_227470


namespace four_fours_expressions_l2274_227422

theorem four_fours_expressions :
  (4 * 4 + 4) / 4 = 5 ∧
  4 + (4 + 4) / 2 = 6 ∧
  4 + 4 - 4 / 4 = 7 ∧
  4 + 4 + 4 - 4 = 8 ∧
  4 + 4 + 4 / 4 = 9 :=
by
  sorry

end four_fours_expressions_l2274_227422


namespace james_ali_difference_l2274_227405

theorem james_ali_difference (J A T : ℝ) (h1 : J = 145) (h2 : T = 250) (h3 : J + A = T) :
  J - A = 40 :=
by
  sorry

end james_ali_difference_l2274_227405


namespace range_of_m_l2274_227469

theorem range_of_m (m y1 y2 k : ℝ) (h1 : y1 = -2 * (m - 2) ^ 2 + k) (h2 : y2 = -2 * (m - 1) ^ 2 + k) (h3 : y1 > y2) : m > 3 / 2 := 
sorry

end range_of_m_l2274_227469


namespace train_cross_time_l2274_227441

-- Define the conditions
def train_speed_kmhr := 52
def train_length_meters := 130

-- Conversion factor from km/hr to m/s
def kmhr_to_ms (speed_kmhr : ℕ) : ℕ := (speed_kmhr * 1000) / 3600

-- Speed of the train in m/s
def train_speed_ms := kmhr_to_ms train_speed_kmhr

-- Calculate time to cross the pole
def time_to_cross_pole (distance_m : ℕ) (speed_ms : ℕ) : ℕ := distance_m / speed_ms

-- The theorem to prove
theorem train_cross_time : time_to_cross_pole train_length_meters train_speed_ms = 9 := by sorry

end train_cross_time_l2274_227441
