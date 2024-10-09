import Mathlib

namespace quadratic_two_distinct_real_roots_l1538_153810

theorem quadratic_two_distinct_real_roots (k : ℝ) : ∃ x : ℝ, x^2 + 2 * x - k = 0 ∧ 
  (∀ x1 x2: ℝ, x1 ≠ x2 → x1^2 + 2 * x1 - k = 0 ∧ x2^2 + 2 * x2 - k = 0) ↔ k > -1 :=
by
  sorry

end quadratic_two_distinct_real_roots_l1538_153810


namespace bert_phone_price_l1538_153888

theorem bert_phone_price :
  ∃ x : ℕ, x * 8 = 144 := sorry

end bert_phone_price_l1538_153888


namespace sufficient_not_necessary_l1538_153801

theorem sufficient_not_necessary (x : ℝ) : (x > 3) → (abs (x - 3) > 0) ∧ (¬(abs (x - 3) > 0) → (¬(x > 3))) :=
by
  sorry

end sufficient_not_necessary_l1538_153801


namespace fraction_transform_l1538_153853

theorem fraction_transform (x : ℝ) (h : (1/3) * x = 12) : (1/4) * x = 9 :=
by 
  sorry

end fraction_transform_l1538_153853


namespace triangle_perimeter_l1538_153899

-- Define the triangle with sides a, b, c
structure Triangle :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

-- Define the predicate that checks if the triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.a = t.c ∨ t.b = t.c

-- Define the predicate that calculates the perimeter of the triangle
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- State the problem
theorem triangle_perimeter : 
  ∃ (t : Triangle), isIsosceles t ∧ (    (t.a = 6 ∧ t.b = 9 ∧ perimeter t = 24)
                                       ∨ (t.b = 6 ∧ t.a = 9 ∧ perimeter t = 24)
                                       ∨ (t.c = 6 ∧ t.a = 9 ∧ perimeter t = 21)
                                       ∨ (t.a = 6 ∧ t.c = 9 ∧ perimeter t = 21)
                                       ∨ (t.b = 6 ∧ t.c = 9 ∧ perimeter t = 24)
                                       ∨ (t.c = 6 ∧ t.b = 9 ∧ perimeter t = 21)
                                    ) :=
sorry

end triangle_perimeter_l1538_153899


namespace josie_leftover_amount_l1538_153824

-- Define constants and conditions
def initial_amount : ℝ := 20.00
def milk_price : ℝ := 4.00
def bread_price : ℝ := 3.50
def detergent_price : ℝ := 10.25
def bananas_price_per_pound : ℝ := 0.75
def bananas_weight : ℝ := 2.0
def detergent_coupon : ℝ := 1.25
def milk_discount_rate : ℝ := 0.5

-- Define the total cost before any discounts
def total_cost_before_discounts : ℝ := 
  milk_price + bread_price + detergent_price + (bananas_weight * bananas_price_per_pound)

-- Define the discounted prices
def milk_discounted_price : ℝ := milk_price * milk_discount_rate
def detergent_discounted_price : ℝ := detergent_price - detergent_coupon

-- Define the total cost after discounts
def total_cost_after_discounts : ℝ := 
  milk_discounted_price + bread_price + detergent_discounted_price + 
  (bananas_weight * bananas_price_per_pound)

-- Prove the amount left over
theorem josie_leftover_amount : initial_amount - total_cost_after_discounts = 4.00 := by
  simp [total_cost_before_discounts, milk_discounted_price, detergent_discounted_price,
    total_cost_after_discounts, initial_amount, milk_price, bread_price, detergent_price,
    bananas_price_per_pound, bananas_weight, detergent_coupon, milk_discount_rate]
  sorry

end josie_leftover_amount_l1538_153824


namespace ratio_perimeters_not_integer_l1538_153825

theorem ratio_perimeters_not_integer
  (a k l : ℤ) (h_a_pos : a > 0) (h_k_pos : k > 0) (h_l_pos : l > 0)
  (h_area : a^2 = k * l) :
  ¬ ∃ n : ℤ, n = (k + l) / (2 * a) :=
by
  sorry

end ratio_perimeters_not_integer_l1538_153825


namespace range_of_a_l1538_153848

noncomputable def f (x : ℝ) := Real.exp x
noncomputable def g (a x : ℝ) := a * Real.sqrt x
noncomputable def f' (x₀ : ℝ) := Real.exp x₀
noncomputable def g' (a t : ℝ) := a / (2 * Real.sqrt t)

theorem range_of_a (a : ℝ) (x₀ t : ℝ) (hx₀ : x₀ = 1 - t) (ht_pos : t > 0)
  (h1 : f x₀ = Real.exp x₀)
  (h2 : g a t = a * Real.sqrt t)
  (h3 : f x₀ = g' a t)
  (h4 : (Real.exp x₀ - a * Real.sqrt t) / (x₀ - t) = Real.exp x₀) :
    0 < a ∧ a ≤ Real.sqrt (2 * Real.exp 1) :=
sorry

end range_of_a_l1538_153848


namespace find_a3_plus_a5_l1538_153845

variable (a : ℕ → ℝ)
variable (positive_arith_geom_seq : ∀ n : ℕ, 0 < a n)
variable (h1 : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25)

theorem find_a3_plus_a5 (positive_arith_geom_seq : ∀ n : ℕ, 0 < a n) (h1 : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25) :
  a 3 + a 5 = 5 :=
by
  sorry

end find_a3_plus_a5_l1538_153845


namespace no_real_solutions_for_equation_l1538_153862

theorem no_real_solutions_for_equation :
  ¬ (∃ x : ℝ, (2 * x - 3 * x + 7)^2 + 2 = -|2 * x|) :=
by 
-- proof will go here
sorry

end no_real_solutions_for_equation_l1538_153862


namespace sum_of_coefficients_l1538_153870

theorem sum_of_coefficients (x : ℝ) : 
  (1 - 2 * x) ^ 10 = 1 :=
sorry

end sum_of_coefficients_l1538_153870


namespace fraction_one_third_between_l1538_153883

theorem fraction_one_third_between (a b : ℚ) (h1 : a = 1/6) (h2 : b = 1/4) : (1/3 * (b - a) + a = 7/36) :=
by
  -- Conditions
  have ha : a = 1/6 := h1
  have hb : b = 1/4 := h2
  -- Start proof
  sorry

end fraction_one_third_between_l1538_153883


namespace new_years_day_more_frequent_l1538_153890

-- Define conditions
def common_year_days : ℕ := 365
def leap_year_days : ℕ := 366
def century_is_leap_year (year : ℕ) : Prop := (year % 400 = 0)

-- Given: 23 October 1948 was a Saturday
def october_23_1948 : ℕ := 5 -- 5 corresponds to Saturday

-- Define the question proof statement
theorem new_years_day_more_frequent :
  (frequency_Sunday : ℕ) > (frequency_Monday : ℕ) :=
sorry

end new_years_day_more_frequent_l1538_153890


namespace find_fourth_number_l1538_153809

def nat_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)

variable {a : ℕ → ℕ}

theorem find_fourth_number (h_seq : nat_sequence a) (h7 : a 7 = 42) (h9 : a 9 = 110) : a 4 = 10 :=
by
  -- Placeholder for proof steps
  sorry

end find_fourth_number_l1538_153809


namespace sum_of_possible_values_l1538_153894

theorem sum_of_possible_values (a b c d : ℝ) (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) : 
  (a - c) * (b - d) / ((c - d) * (d - a)) = 1 :=
by
  -- Solution omitted
  sorry

end sum_of_possible_values_l1538_153894


namespace factor_poly_find_abs_l1538_153878

theorem factor_poly_find_abs {
  p q : ℤ
} (h1 : 3 * (-2)^3 - p * (-2) + q = 0) 
  (h2 : 3 * (3)^3 - p * (3) + q = 0) :
  |3 * p - 2 * q| = 99 := sorry

end factor_poly_find_abs_l1538_153878


namespace find_z_l1538_153811

def z_value (i : ℂ) (z : ℂ) : Prop := z * (1 - (2 * i)) = 2 + (4 * i)

theorem find_z (i z : ℂ) (hi : i^2 = -1) (h : z_value i z) : z = - (2 / 5) + (8 / 5) * i := by
  sorry

end find_z_l1538_153811


namespace smallest_n_satisfying_conditions_l1538_153877

-- We need variables and statements
variables (n : ℕ)

-- Define the conditions
def condition1 : Prop := n % 6 = 4
def condition2 : Prop := n % 7 = 3
def condition3 : Prop := n > 20

-- The main theorem statement to be proved
theorem smallest_n_satisfying_conditions (h1 : condition1 n) (h2 : condition2 n) (h3 : condition3 n) : n = 52 :=
by 
  sorry

end smallest_n_satisfying_conditions_l1538_153877


namespace paul_coins_difference_l1538_153841

/-- Paul owes Paula 145 cents and has a pocket full of 10-cent coins, 
20-cent coins, and 50-cent coins. Prove that the difference between 
the largest and smallest number of coins he can use to pay her is 9. -/
theorem paul_coins_difference :
  ∃ min_coins max_coins : ℕ, 
    (min_coins = 5 ∧ max_coins = 14) ∧ (max_coins - min_coins = 9) :=
by
  sorry

end paul_coins_difference_l1538_153841


namespace number_of_squares_in_H_l1538_153857

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

end number_of_squares_in_H_l1538_153857


namespace recurrence_relation_l1538_153844

def u (n : ℕ) : ℕ := sorry

theorem recurrence_relation (n : ℕ) : 
  u (n + 1) = (n + 1) * u n - (n * (n - 1)) / 2 * u (n - 2) :=
sorry

end recurrence_relation_l1538_153844


namespace probability_of_different_groups_is_correct_l1538_153861

-- Define the number of total members and groups
def num_groups : ℕ := 6
def members_per_group : ℕ := 3
def total_members : ℕ := num_groups * members_per_group

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability function to select 3 people from different groups
noncomputable def probability_different_groups : ℚ :=
  binom num_groups 3 / binom total_members 3

-- State the theorem we want to prove
theorem probability_of_different_groups_is_correct :
  probability_different_groups = 5 / 204 :=
by
  sorry

end probability_of_different_groups_is_correct_l1538_153861


namespace B_took_18_more_boxes_than_D_l1538_153869

noncomputable def A_boxes : ℕ := sorry
noncomputable def B_boxes : ℕ := A_boxes + 4
noncomputable def C_boxes : ℕ := sorry
noncomputable def D_boxes : ℕ := C_boxes + 8
noncomputable def A_owes_C : ℕ := 112
noncomputable def B_owes_D : ℕ := 72

theorem B_took_18_more_boxes_than_D : (B_boxes - D_boxes) = 18 :=
sorry

end B_took_18_more_boxes_than_D_l1538_153869


namespace A_beats_B_by_seconds_l1538_153889

theorem A_beats_B_by_seconds :
  ∀ (t_A : ℝ) (distance_A distance_B : ℝ),
  t_A = 156.67 →
  distance_A = 1000 →
  distance_B = 940 →
  (distance_A * t_A = 60 * (distance_A / t_A)) →
  t_A ≠ 0 →
  ((60 * t_A / distance_A) = 9.4002) :=
by
  intros t_A distance_A distance_B h1 h2 h3 h4 h5
  sorry

end A_beats_B_by_seconds_l1538_153889


namespace binomial_coefficient_sum_l1538_153852

theorem binomial_coefficient_sum :
  Nat.choose 10 3 + Nat.choose 10 2 = 165 := by
  sorry

end binomial_coefficient_sum_l1538_153852


namespace contrapositive_proposition_l1538_153897

def proposition (x : ℝ) : Prop := x < 0 → x^2 > 0

theorem contrapositive_proposition :
  (∀ x : ℝ, proposition x) → (∀ x : ℝ, x^2 ≤ 0 → x ≥ 0) :=
by
  sorry

end contrapositive_proposition_l1538_153897


namespace school_children_equation_l1538_153815

theorem school_children_equation
  (C B : ℕ)
  (h1 : B = 2 * C)
  (h2 : B = 4 * (C - 350)) :
  C = 700 := by
  sorry

end school_children_equation_l1538_153815


namespace how_many_years_older_l1538_153846

-- Definitions of the conditions
variables (a b c : ℕ)
def b_is_16 : Prop := b = 16
def b_is_twice_c : Prop := b = 2 * c
def sum_is_42 : Prop := a + b + c = 42

-- Statement of the proof problem
theorem how_many_years_older (h1 : b_is_16 b) (h2 : b_is_twice_c b c) (h3 : sum_is_42 a b c) : a - b = 2 :=
by
  sorry

end how_many_years_older_l1538_153846


namespace longest_third_side_of_triangle_l1538_153826

theorem longest_third_side_of_triangle {a b : ℕ} (ha : a = 8) (hb : b = 9) : 
  ∃ c : ℕ, 1 < c ∧ c < 17 ∧ ∀ (d : ℕ), (1 < d ∧ d < 17) → d ≤ c :=
by
  sorry

end longest_third_side_of_triangle_l1538_153826


namespace evaluate_expression_l1538_153803

def expression (x y : ℤ) : ℤ :=
  y * (y - 2 * x) ^ 2

theorem evaluate_expression : 
  expression 4 2 = 72 :=
by
  -- Proof will go here
  sorry

end evaluate_expression_l1538_153803


namespace smallest_positive_period_l1538_153867

noncomputable def f (A ω φ : ℝ) (x : ℝ) := A * Real.sin (ω * x + φ)

theorem smallest_positive_period 
  (A ω φ T : ℝ) 
  (hA : A > 0) 
  (hω : ω > 0)
  (h1 : f A ω φ (π / 2) = f A ω φ (2 * π / 3))
  (h2 : f A ω φ (π / 6) = -f A ω φ (π / 2))
  (h3 : ∀ x1 x2, (π / 6) ≤ x1 → x1 ≤ x2 → x2 ≤ (π / 2) → f A ω φ x1 ≤ f A ω φ x2) :
  T = π :=
sorry

end smallest_positive_period_l1538_153867


namespace max_volume_rectangular_frame_l1538_153839

theorem max_volume_rectangular_frame (L W H : ℝ) (h1 : 2 * W = L) (h2 : 4 * (L + W) + 4 * H = 18) :
  volume = (2 * 1 * 1.5 : ℝ) := 
sorry

end max_volume_rectangular_frame_l1538_153839


namespace percentage_employees_6_years_or_more_is_26_l1538_153871

-- Define the units for different years of service
def units_less_than_2_years : ℕ := 4
def units_2_to_4_years : ℕ := 6
def units_4_to_6_years : ℕ := 7
def units_6_to_8_years : ℕ := 3
def units_8_to_10_years : ℕ := 2
def units_more_than_10_years : ℕ := 1

-- Define the total units
def total_units : ℕ :=
  units_less_than_2_years +
  units_2_to_4_years +
  units_4_to_6_years +
  units_6_to_8_years +
  units_8_to_10_years +
  units_more_than_10_years

-- Define the units representing employees with 6 years or more of service
def units_6_years_or_more : ℕ :=
  units_6_to_8_years +
  units_8_to_10_years +
  units_more_than_10_years

-- The goal is to prove that this percentage is 26%
theorem percentage_employees_6_years_or_more_is_26 :
  (units_6_years_or_more * 100) / total_units = 26 := by
  sorry

end percentage_employees_6_years_or_more_is_26_l1538_153871


namespace sin_600_eq_neg_sqrt3_div2_l1538_153893

theorem sin_600_eq_neg_sqrt3_div2 : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by sorry

end sin_600_eq_neg_sqrt3_div2_l1538_153893


namespace complete_job_days_l1538_153865

-- Variables and Conditions
variables (days_5_8 : ℕ) (days_1 : ℕ)

-- Assume that completing 5/8 of the job takes 10 days
def five_eighths_job_days := 10

-- Find days to complete one job at the same pace. 
-- This is the final statement we need to prove
theorem complete_job_days
  (h : 5 * days_1 = 8 * days_5_8) :
  days_1 = 16 := by
  -- Proof is omitted.
  sorry

end complete_job_days_l1538_153865


namespace girls_more_than_boys_l1538_153843

-- Given conditions
def ratio_boys_girls : ℕ := 3
def ratio_girls_boys : ℕ := 4
def total_students : ℕ := 42

-- Theorem statement
theorem girls_more_than_boys : 
  let x := total_students / (ratio_boys_girls + ratio_girls_boys)
  let boys := ratio_boys_girls * x
  let girls := ratio_girls_boys * x
  girls - boys = 6 := by
  sorry

end girls_more_than_boys_l1538_153843


namespace find_square_length_CD_l1538_153875

noncomputable def parabola (x : ℝ) : ℝ := 3 * x ^ 2 + 6 * x - 2

def is_midpoint (mid C D : (ℝ × ℝ)) : Prop :=
  mid.1 = (C.1 + D.1) / 2 ∧ mid.2 = (C.2 + D.2) / 2

theorem find_square_length_CD (C D : ℝ × ℝ)
  (hC : C.2 = parabola C.1)
  (hD : D.2 = parabola D.1)
  (h_mid : is_midpoint (0,0) C D) :
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = 740 / 3 :=
sorry

end find_square_length_CD_l1538_153875


namespace smallest_number_of_lawyers_l1538_153800

/-- Given that:
- n is the number of delegates, where 220 < n < 254
- m is the number of economists, so the number of lawyers is n - m
- Each participant played with each other participant exactly once.
- A match winner got one point, the loser got none, and in case of a draw, both participants received half a point each.
- By the end of the tournament, each participant gained half of all their points from matches against economists.

Prove that the smallest number of lawyers participating in the tournament is 105. -/
theorem smallest_number_of_lawyers (n m : ℕ) (h1 : 220 < n) (h2 : n < 254)
  (h3 : m * (m - 1) + (n - m) * (n - m - 1) = n * (n - 1))
  (h4 : m * (m - 1) = 2 * (n * (n - 1)) / 4) :
  n - m = 105 :=
sorry

end smallest_number_of_lawyers_l1538_153800


namespace Olivia_score_l1538_153849

theorem Olivia_score 
  (n : ℕ) (m : ℕ) (average20 : ℕ) (average21 : ℕ)
  (h_n : n = 20) (h_m : m = 21) (h_avg20 : average20 = 85) (h_avg21 : average21 = 86)
  : ∃ (scoreOlivia : ℕ), scoreOlivia = m * average21 - n * average20 :=
by
  sorry

end Olivia_score_l1538_153849


namespace commission_percentage_l1538_153863

theorem commission_percentage (commission_earned total_sales : ℝ) (h₀ : commission_earned = 18) (h₁ : total_sales = 720) : 
  ((commission_earned / total_sales) * 100) = 2.5 := by {
  sorry
}

end commission_percentage_l1538_153863


namespace negation_proposition_l1538_153814

theorem negation_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by
  sorry

end negation_proposition_l1538_153814


namespace correct_calculation_is_D_l1538_153828

theorem correct_calculation_is_D 
  (a b x : ℝ) :
  ¬ (5 * a + 2 * b = 7 * a * b) ∧
  ¬ (x ^ 2 - 3 * x ^ 2 = -2) ∧
  ¬ (7 * a - b + (7 * a + b) = 0) ∧
  (4 * a - (-7 * a) = 11 * a) :=
by 
  sorry

end correct_calculation_is_D_l1538_153828


namespace number_of_valid_six_digit_house_numbers_l1538_153882

-- Define the set of two-digit primes less than 60
def two_digit_primes : List ℕ := [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]

-- Define a predicate checking if a number is a two-digit prime less than 60
def is_valid_prime (n : ℕ) : Prop :=
  n ∈ two_digit_primes

-- Define the function to count distinct valid primes forming ABCDEF
def count_valid_house_numbers : ℕ :=
  let primes_count := two_digit_primes.length
  primes_count * (primes_count - 1) * (primes_count - 2)

-- State the main theorem
theorem number_of_valid_six_digit_house_numbers : count_valid_house_numbers = 1716 := by
  -- Showing the count of valid house numbers forms 1716
  sorry

end number_of_valid_six_digit_house_numbers_l1538_153882


namespace isla_capsules_days_l1538_153840

theorem isla_capsules_days (days_in_july : ℕ) (days_forgot : ℕ) (known_days_in_july : days_in_july = 31) (known_days_forgot : days_forgot = 2) : days_in_july - days_forgot = 29 := 
by
  -- Placeholder for proof, not required in the response.
  sorry

end isla_capsules_days_l1538_153840


namespace inequality_solution_l1538_153822

theorem inequality_solution (x : ℝ) :
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1/4) ∧ (x - 2 > 0) → x > 2 :=
by {
  sorry
}

end inequality_solution_l1538_153822


namespace reciprocal_of_neg_one_fifth_l1538_153842

theorem reciprocal_of_neg_one_fifth : (-(1 / 5) : ℚ)⁻¹ = -5 :=
by
  sorry

end reciprocal_of_neg_one_fifth_l1538_153842


namespace solution_to_logarithmic_equation_l1538_153873

noncomputable def log_base (a b : ℝ) := Real.log b / Real.log a

def equation (x : ℝ) := log_base 2 x + 1 / log_base (x + 1) 2 = 1

theorem solution_to_logarithmic_equation :
  ∃ x > 0, equation x ∧ x = 1 :=
by
  sorry

end solution_to_logarithmic_equation_l1538_153873


namespace mod_add_5000_l1538_153834

theorem mod_add_5000 (n : ℤ) (h : n % 6 = 4) : (n + 5000) % 6 = 0 :=
sorry

end mod_add_5000_l1538_153834


namespace find_x_in_sequence_l1538_153854

theorem find_x_in_sequence
  (x d1 d2 : ℤ)
  (h1 : d1 = x - 1370)
  (h2 : d2 = 1070 - x)
  (h3 : -180 - 1070 = -1250)
  (h4 : -6430 - (-180) = -6250)
  (h5 : d2 - d1 = 5000) :
  x = 3720 :=
by
-- Proof omitted
sorry

end find_x_in_sequence_l1538_153854


namespace fedya_deposit_l1538_153818

theorem fedya_deposit (n : ℕ) (h1 : n < 30) (h2 : 847 * 100 % (100 - n) = 0) : 
  (847 * 100 / (100 - n) = 1100) :=
by
  sorry

end fedya_deposit_l1538_153818


namespace peanut_butter_sandwich_days_l1538_153874

theorem peanut_butter_sandwich_days 
  (H : ℕ)
  (total_days : ℕ)
  (probability_ham_and_cake : ℚ)
  (ham_probability : ℚ)
  (cake_probability : ℚ)
  (Ham_days : H = 3)
  (Total_days : total_days = 5)
  (Ham_probability_val : ham_probability = H / 5)
  (Cake_probability_val : cake_probability = 1 / 5)
  (Probability_condition : ham_probability * cake_probability = 0.12) :
  5 - H = 2 :=
by 
  sorry

end peanut_butter_sandwich_days_l1538_153874


namespace correct_option_l1538_153808

theorem correct_option (a b : ℝ) : (ab) ^ 2 = a ^ 2 * b ^ 2 :=
by sorry

end correct_option_l1538_153808


namespace regular_polygons_cover_plane_l1538_153804

theorem regular_polygons_cover_plane (n : ℕ) (h_n_ge_3 : 3 ≤ n)
    (h_angle_eq : ∀ n, (180 * (1 - (2 / n)) : ℝ) = (internal_angle : ℝ))
    (h_summation_eq : ∃ k : ℕ, k * internal_angle = 360) :
    n = 3 ∨ n = 4 ∨ n = 6 := 
sorry

end regular_polygons_cover_plane_l1538_153804


namespace length_segment_MN_l1538_153802

open Real

noncomputable def line (x : ℝ) : ℝ := x + 2

def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 5

theorem length_segment_MN :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
    on_circle x₁ y₁ →
    on_circle x₂ y₂ →
    (line x₁ = y₁ ∧ line x₂ = y₂) →
    dist (x₁, y₁) (x₂, y₂) = 2 * sqrt 3 :=
by
  sorry

end length_segment_MN_l1538_153802


namespace find_a_for_odd_function_l1538_153891

theorem find_a_for_odd_function (f : ℝ → ℝ) (a : ℝ) (h₀ : ∀ x, f (-x) = -f x) (h₁ : ∀ x, x < 0 → f x = x^2 + a * x) (h₂ : f 3 = 6) : a = 5 :=
by
  sorry

end find_a_for_odd_function_l1538_153891


namespace range_of_m_l1538_153831

theorem range_of_m (m x : ℝ) (h1 : (3 * x) / (x - 1) = m / (x - 1) + 2) (h2 : x ≥ 0) (h3 : x ≠ 1) : 
  m ≥ 2 ∧ m ≠ 3 := 
sorry

end range_of_m_l1538_153831


namespace gg1_eq_13_l1538_153858

def g (n : ℕ) : ℕ :=
if n < 3 then n^2 + 1
else if n < 6 then 2 * n + 3
else 4 * n - 2

theorem gg1_eq_13 : g (g (g 1)) = 13 :=
by
  sorry

end gg1_eq_13_l1538_153858


namespace quadratic_function_value_when_x_is_zero_l1538_153864

theorem quadratic_function_value_when_x_is_zero :
  (∃ h : ℝ, (∀ x : ℝ, x < -3 → (-(x + h)^2 < -(x + h + 1)^2)) ∧
            (∀ x : ℝ, x > -3 → (-(x + h)^2 > -(x + h - 1)^2)) ∧
            (y = -(0 + h)^2) → y = -9) := 
sorry

end quadratic_function_value_when_x_is_zero_l1538_153864


namespace artist_painting_time_l1538_153823

theorem artist_painting_time (hours_per_week : ℕ) (weeks : ℕ) (total_paintings : ℕ) :
  hours_per_week = 30 → weeks = 4 → total_paintings = 40 →
  ((hours_per_week * weeks) / total_paintings) = 3 := by
  intros h_hours h_weeks h_paintings
  sorry

end artist_painting_time_l1538_153823


namespace problem_statement_l1538_153847

theorem problem_statement (m n : ℝ) (h : m + n = 1 / 2 * m * n) : (m - 2) * (n - 2) = 4 :=
by sorry

end problem_statement_l1538_153847


namespace cafeteria_can_make_7_pies_l1538_153880

theorem cafeteria_can_make_7_pies (initial_apples handed_out apples_per_pie : ℕ)
  (h1 : initial_apples = 86)
  (h2 : handed_out = 30)
  (h3 : apples_per_pie = 8) :
  ((initial_apples - handed_out) / apples_per_pie) = 7 := 
by
  sorry

end cafeteria_can_make_7_pies_l1538_153880


namespace total_built_up_area_l1538_153833

theorem total_built_up_area
    (A1 A2 A3 A4 : ℕ)
    (hA1 : A1 = 480)
    (hA2 : A2 = 560)
    (hA3 : A3 = 200)
    (hA4 : A4 = 440)
    (total_plot_area : ℕ)
    (hplots : total_plot_area = 4 * (480 + 560 + 200 + 440) / 4)
    : 800 = total_plot_area - (A1 + A2 + A3 + A4) :=
by
  -- This is where the solution will be filled in
  sorry

end total_built_up_area_l1538_153833


namespace total_increase_area_l1538_153876

theorem total_increase_area (increase_broccoli increase_cauliflower increase_cabbage : ℕ)
    (area_broccoli area_cauliflower area_cabbage : ℝ)
    (h1 : increase_broccoli = 79)
    (h2 : increase_cauliflower = 25)
    (h3 : increase_cabbage = 50)
    (h4 : area_broccoli = 1)
    (h5 : area_cauliflower = 2)
    (h6 : area_cabbage = 1.5) :
    increase_broccoli * area_broccoli +
    increase_cauliflower * area_cauliflower +
    increase_cabbage * area_cabbage = 204 := 
by 
    sorry

end total_increase_area_l1538_153876


namespace max_chain_length_in_subdivided_triangle_l1538_153860

-- Define an equilateral triangle subdivision
structure EquilateralTriangleSubdivided (n : ℕ) :=
(n_squares : ℕ)
(n_squares_eq : n_squares = n^2)

-- Define the problem's chain concept
def maximum_chain_length (n : ℕ) : ℕ :=
n^2 - n + 1

-- Main statement
theorem max_chain_length_in_subdivided_triangle
  (n : ℕ) (triangle : EquilateralTriangleSubdivided n) :
  maximum_chain_length n = n^2 - n + 1 :=
by sorry

end max_chain_length_in_subdivided_triangle_l1538_153860


namespace xy_divides_x2_plus_2y_minus_1_l1538_153832

theorem xy_divides_x2_plus_2y_minus_1 (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (x * y ∣ x^2 + 2 * y - 1) ↔ (∃ t : ℕ, t > 0 ∧ ((x = 1 ∧ y = t) ∨ (x = 2 * t - 1 ∧ y = t)
  ∨ (x = 3 ∧ y = 8) ∨ (x = 5 ∧ y = 8))) :=
by
  sorry

end xy_divides_x2_plus_2y_minus_1_l1538_153832


namespace find_x_l1538_153896

theorem find_x (x : ℤ) (h : 9873 + x = 13800) : x = 3927 :=
by {
  sorry
}

end find_x_l1538_153896


namespace sum_of_prime_factors_77_l1538_153837

theorem sum_of_prime_factors_77 : (7 + 11 = 18) := by
  sorry

end sum_of_prime_factors_77_l1538_153837


namespace arithmetic_to_geometric_l1538_153836

theorem arithmetic_to_geometric (a1 a2 a3 a4 d : ℝ)
  (h_arithmetic : a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d)
  (h_d_nonzero : d ≠ 0):
  ((a2^2 = a1 * a3 ∨ a2^2 = a1 * a4 ∨ a3^2 = a1 * a4 ∨ a3^2 = a2 * a4) → (a1 / d = 1 ∨ a1 / d = -4)) :=
by {
  sorry
}

end arithmetic_to_geometric_l1538_153836


namespace coords_P_origin_l1538_153830

variable (x y : Int)
def point_P := (-5, 3)

theorem coords_P_origin : point_P = (-5, 3) := 
by 
  -- Proof to be written here
  sorry

end coords_P_origin_l1538_153830


namespace find_counterfeit_l1538_153816

-- Definitions based on the conditions
structure Coin :=
(weight : ℝ)
(is_genuine : Bool)

def is_counterfeit (coins : List Coin) : Prop :=
  ∃ (c : Coin) (h : c ∈ coins), ¬c.is_genuine

def weigh (c1 c2 : Coin) : ℝ := c1.weight - c2.weight

def identify_counterfeit (coins : List Coin) : Prop :=
  ∀ (a b c d : Coin), 
    coins = [a, b, c, d] →
    (¬a.is_genuine ∨ ¬b.is_genuine ∨ ¬c.is_genuine ∨ ¬d.is_genuine) →
    (weigh a b = 0 ∧ weigh c d ≠ 0 ∨ weigh a c = 0 ∧ weigh b d ≠ 0 ∨ weigh a d = 0 ∧ weigh b c ≠ 0) →
    (∃ (fake_coin : Coin), fake_coin ∈ coins ∧ ¬fake_coin.is_genuine)

-- Proof statement
theorem find_counterfeit (coins : List Coin) :
  (∃ (c : Coin), c ∈ coins ∧ ¬c.is_genuine) →
  identify_counterfeit coins :=
by
  sorry

end find_counterfeit_l1538_153816


namespace combined_meows_l1538_153807

theorem combined_meows (first_cat_freq second_cat_freq third_cat_freq : ℕ) 
  (time : ℕ) 
  (h1 : first_cat_freq = 3)
  (h2 : second_cat_freq = 2 * first_cat_freq)
  (h3 : third_cat_freq = second_cat_freq / 3)
  (h4 : time = 5) : 
  first_cat_freq * time + second_cat_freq * time + third_cat_freq * time = 55 := 
by
  sorry

end combined_meows_l1538_153807


namespace missing_digit_B_divisible_by_3_l1538_153850

theorem missing_digit_B_divisible_by_3 (B : ℕ) (h1 : (2 * 10 + 8 + B) % 3 = 0) :
  B = 2 :=
sorry

end missing_digit_B_divisible_by_3_l1538_153850


namespace factor_polynomial_l1538_153898

theorem factor_polynomial 
(a b c d : ℝ) :
  a^3 * (b^2 - d^2) + b^3 * (c^2 - a^2) + c^3 * (d^2 - b^2) + d^3 * (a^2 - c^2)
  = (a - b) * (b - c) * (c - d) * (d - a) * (a^2 + ab + ac + ad + b^2 + bc + bd + c^2 + cd + d^2) :=
sorry

end factor_polynomial_l1538_153898


namespace sum_of_squares_of_distances_l1538_153884

-- Definitions based on the conditions provided:
variables (A B C D X : Point)
variable (a : ℝ)
variable (h1 h2 h3 h4 : ℝ)

-- Conditions:
axiom square_side_length : a = 5
axiom area_ratios : (1/2 * a * h1) / (1/2 * a * h2) = 1 / 5 ∧ 
                    (1/2 * a * h2) / (1/2 * a * h3) = 5 / 9

-- Problem Statement to Prove:
theorem sum_of_squares_of_distances :
  h1^2 + h2^2 + h3^2 + h4^2 = 33 :=
sorry

end sum_of_squares_of_distances_l1538_153884


namespace product_of_two_smaller_numbers_is_85_l1538_153820

theorem product_of_two_smaller_numbers_is_85
  (A B C : ℝ)
  (h1 : B = 10)
  (h2 : C - B = B - A)
  (h3 : B * C = 115) :
  A * B = 85 :=
by
  sorry

end product_of_two_smaller_numbers_is_85_l1538_153820


namespace intersection_of_M_and_N_l1538_153813

def M : Set ℝ := { x | x ≤ 0 }
def N : Set ℝ := { -2, 0, 1 }

theorem intersection_of_M_and_N : M ∩ N = { -2, 0 } := 
by
  sorry

end intersection_of_M_and_N_l1538_153813


namespace total_amount_correct_l1538_153879

namespace ProofExample

def initial_amount : ℝ := 3

def additional_amount : ℝ := 6.8

def total_amount (initial : ℝ) (additional : ℝ) : ℝ := initial + additional

theorem total_amount_correct : total_amount initial_amount additional_amount = 9.8 :=
by
  sorry

end ProofExample

end total_amount_correct_l1538_153879


namespace sin_negative_300_eq_l1538_153806

theorem sin_negative_300_eq : Real.sin (-(300 * Real.pi / 180)) = Real.sqrt 3 / 2 :=
by
  -- Periodic property of sine function: sin(theta) = sin(theta + 360 * n)
  have periodic_property : ∀ θ n : ℤ, Real.sin θ = Real.sin (θ + n * 2 * Real.pi) :=
    by sorry
  -- Known value: sin(60 degrees) = sqrt(3)/2
  have sin_60 : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  -- Apply periodic_property to transform sin(-300 degrees) to sin(60 degrees)
  sorry

end sin_negative_300_eq_l1538_153806


namespace find_m_and_union_A_B_l1538_153827

variable (m : ℝ)
noncomputable def A := ({3, 4, m^2 - 3 * m - 1} : Set ℝ)
noncomputable def B := ({2 * m, -3} : Set ℝ)

theorem find_m_and_union_A_B (h : A m ∩ B m = ({-3} : Set ℝ)) :
  m = 1 ∧ A m ∪ B m = ({-3, 2, 3, 4} : Set ℝ) :=
sorry

end find_m_and_union_A_B_l1538_153827


namespace range_of_a_plus_b_l1538_153885

theorem range_of_a_plus_b (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : |Real.log a| = |Real.log b|) (h₄ : a ≠ b) :
  2 < a + b :=
by
  sorry

end range_of_a_plus_b_l1538_153885


namespace team_selection_l1538_153872

theorem team_selection (boys girls : ℕ) (choose_boys choose_girls : ℕ) 
  (boy_count girl_count : ℕ) (h1 : boy_count = 10) (h2 : girl_count = 12) 
  (h3 : choose_boys = 5) (h4 : choose_girls = 3) :
    (Nat.choose boy_count choose_boys) * (Nat.choose girl_count choose_girls) = 55440 :=
by
  rw [h1, h2, h3, h4]
  sorry

end team_selection_l1538_153872


namespace solve_quadratic_equation_l1538_153895

theorem solve_quadratic_equation (x : ℝ) :
  (x^2 + 2 * x - 15 = 0) ↔ (x = 3 ∨ x = -5) :=
by
  sorry -- proof omitted

end solve_quadratic_equation_l1538_153895


namespace range_of_a_l1538_153856

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * x + a ≤ 0) → a > 1 :=
by
  sorry

end range_of_a_l1538_153856


namespace simplify_expression_l1538_153887

noncomputable def p (x a b c : ℝ) :=
  (x + 2 * a)^2 / ((a - b) * (a - c)) +
  (x + 2 * b)^2 / ((b - a) * (b - c)) +
  (x + 2 * c)^2 / ((c - a) * (c - b))

theorem simplify_expression (a b c x : ℝ) (h : a ≠ b ∧ a ≠ c ∧ b ≠ c) :
  p x a b c = 4 :=
by
  sorry

end simplify_expression_l1538_153887


namespace movie_replay_count_l1538_153835

def movie_length_hours : ℝ := 1.5
def advertisement_length_minutes : ℝ := 20
def theater_operating_hours : ℝ := 11

theorem movie_replay_count :
  let movie_length_minutes := movie_length_hours * 60
  let total_showing_time_minutes := movie_length_minutes + advertisement_length_minutes
  let operating_time_minutes := theater_operating_hours * 60
  (operating_time_minutes / total_showing_time_minutes) = 6 :=
by
  sorry

end movie_replay_count_l1538_153835


namespace Bob_walked_35_miles_l1538_153812

theorem Bob_walked_35_miles (distance : ℕ) 
  (Yolanda_rate Bob_rate : ℕ) (Bob_start_after : ℕ) (Yolanda_initial_walk : ℕ)
  (h1 : distance = 65) 
  (h2 : Yolanda_rate = 5) 
  (h3 : Bob_rate = 7) 
  (h4 : Bob_start_after = 1)
  (h5 : Yolanda_initial_walk = Yolanda_rate * Bob_start_after) :
  Bob_rate * (distance - Yolanda_initial_walk) / (Yolanda_rate + Bob_rate) = 35 := 
by 
  sorry

end Bob_walked_35_miles_l1538_153812


namespace Ryan_bike_time_l1538_153886

-- Definitions of the conditions
variables (B : ℕ)

-- Conditions
def bike_time := B
def bus_time := B + 10
def friend_time := B / 3
def commuting_time := bike_time B + 3 * bus_time B + friend_time B = 160

-- Goal to prove
theorem Ryan_bike_time : commuting_time B → B = 30 :=
by
  intro h
  sorry

end Ryan_bike_time_l1538_153886


namespace largest_n_S_n_positive_l1538_153892

-- We define the arithmetic sequence a_n.
def arith_seq (a_n : ℕ → ℝ) : Prop := 
  ∃ d : ℝ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

-- Definitions for the conditions provided.
def first_term_positive (a_n : ℕ → ℝ) : Prop := 
  a_n 1 > 0

def term_sum_positive (a_n : ℕ → ℝ) : Prop :=
  a_n 2016 + a_n 2017 > 0

def term_product_negative (a_n : ℕ → ℝ) : Prop :=
  a_n 2016 * a_n 2017 < 0

-- Sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a_n 1 + a_n n) / 2

-- Statement we want to prove in Lean 4.
theorem largest_n_S_n_positive (a_n : ℕ → ℝ) 
  (h_seq : arith_seq a_n) 
  (h1 : first_term_positive a_n) 
  (h2 : term_sum_positive a_n) 
  (h3 : term_product_negative a_n) : 
  ∀ n : ℕ, sum_first_n_terms a_n n > 0 → n ≤ 4032 := 
sorry

end largest_n_S_n_positive_l1538_153892


namespace current_speed_l1538_153821

theorem current_speed (r w : ℝ) 
  (h1 : 21 / (r + w) + 3 = 21 / (r - w))
  (h2 : 21 / (1.5 * r + w) + 0.75 = 21 / (1.5 * r - w)) 
  : w = 9.8 :=
by
  sorry

end current_speed_l1538_153821


namespace consecutive_page_sum_l1538_153819

theorem consecutive_page_sum (n : ℤ) (h : n * (n + 1) = 20412) : n + (n + 1) = 285 := by
  sorry

end consecutive_page_sum_l1538_153819


namespace find_length_l1538_153868

variables (w h A l : ℕ)
variable (A_eq : A = 164)
variable (w_eq : w = 4)
variable (h_eq : h = 3)

theorem find_length : 2 * l * w + 2 * l * h + 2 * w * h = A → l = 10 :=
by
  intros H
  rw [w_eq, h_eq, A_eq] at H
  linarith

end find_length_l1538_153868


namespace prove_trigonometric_identities_l1538_153829

variable {α : ℝ}

theorem prove_trigonometric_identities
  (h1 : 0 < α ∧ α < π)
  (h2 : Real.cos α = -3/5) :
  Real.tan α = -4/3 ∧
  (Real.cos (2 * α) - Real.cos (π / 2 + α) = 13/25) := 
by
  sorry

end prove_trigonometric_identities_l1538_153829


namespace chairs_built_in_10_days_l1538_153805

-- Define the conditions as variables
def hours_per_day : ℕ := 8
def days_worked : ℕ := 10
def hours_per_chair : ℕ := 5

-- State the problem as a conjecture or theorem
theorem chairs_built_in_10_days : (hours_per_day * days_worked) / hours_per_chair = 16 := by
    sorry

end chairs_built_in_10_days_l1538_153805


namespace incorrect_score_modulo_l1538_153838

theorem incorrect_score_modulo (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 9) 
  (hb : 0 ≤ b ∧ b ≤ 9) 
  (hc : 0 ≤ c ∧ c ≤ 9) : 
  ∃ remainder : ℕ, remainder = (90 * a + 9 * b + c) % 9 ∧ 0 ≤ remainder ∧ remainder ≤ 9 := 
by
  sorry

end incorrect_score_modulo_l1538_153838


namespace darij_grinberg_inequality_l1538_153817

theorem darij_grinberg_inequality 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) : 
  a + b + c ≤ (bc / (b + c)) + (ca / (c + a)) + (ab / (a + b)) + (1 / 2 * ((bc / a) + (ca / b) + (ab / c))) := 
by sorry

end darij_grinberg_inequality_l1538_153817


namespace percent_area_covered_by_hexagons_l1538_153859

theorem percent_area_covered_by_hexagons (a : ℝ) (h1 : 0 < a) :
  let large_square_area := 4 * a^2
  let hexagon_contribution := a^2 / 4
  (hexagon_contribution / large_square_area) * 100 = 25 := 
by
  sorry

end percent_area_covered_by_hexagons_l1538_153859


namespace rectangle_area_pairs_l1538_153855

theorem rectangle_area_pairs :
  { p : ℕ × ℕ | p.1 * p.2 = 12 ∧ p.1 > 0 ∧ p.2 > 0 } = { (1, 12), (2, 6), (3, 4), (4, 3), (6, 2), (12, 1) } :=
by {
  sorry
}

end rectangle_area_pairs_l1538_153855


namespace plane_equation_l1538_153866

variable (x y z : ℝ)

/-- Equation of the plane passing through points (0, 2, 3) and (2, 0, 3) and perpendicular to the plane 3x - y + 2z = 7 is 2x - 2y + z - 1 = 0. -/
theorem plane_equation :
  ∃ (A B C D : ℤ), A > 0 ∧ Int.gcd (Int.gcd A B) (Int.gcd C D) = 1 ∧ 
  (∀ (x y z : ℝ), (A * x + B * y + C * z + D = 0 ↔ 
  ((0, 2, 3) = (0, 2, 3) ∨ (2, 0, 3) = (2, 0, 3)) ∧ (3 * x - y + 2 * z = 7))) ∧
  A = 2 ∧ B = -2 ∧ C = 1 ∧ D = -1 :=
by
  sorry

end plane_equation_l1538_153866


namespace mrs_hilt_hot_dogs_l1538_153881

theorem mrs_hilt_hot_dogs (cost_per_hotdog total_cost : ℕ) (h1 : cost_per_hotdog = 50) (h2 : total_cost = 300) :
  total_cost / cost_per_hotdog = 6 := by
  sorry

end mrs_hilt_hot_dogs_l1538_153881


namespace solve_for_x_l1538_153851

theorem solve_for_x (x : ℝ) (h₁: 0.45 * x = 0.15 * (1 + x)) : x = 0.5 :=
by sorry

end solve_for_x_l1538_153851
