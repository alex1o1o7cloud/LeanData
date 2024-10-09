import Mathlib

namespace weight_of_mixture_is_correct_l1035_103547

def weight_of_mixture (weight_a_per_l : ℕ) (weight_b_per_l : ℕ) 
                      (total_volume : ℕ) (ratio_a : ℕ) (ratio_b : ℕ) : ℚ :=
  let volume_a := (ratio_a : ℚ) / (ratio_a + ratio_b) * total_volume
  let volume_b := (ratio_b : ℚ) / (ratio_a + ratio_b) * total_volume
  let weight_a := volume_a * weight_a_per_l
  let weight_b := volume_b * weight_b_per_l
  (weight_a + weight_b) / 1000

theorem weight_of_mixture_is_correct :
  weight_of_mixture 800 850 3 3 2 = 2.46 :=
by
  sorry

end weight_of_mixture_is_correct_l1035_103547


namespace smallest_m_for_reflection_l1035_103513

noncomputable def theta : Real := Real.arctan (1 / 3)
noncomputable def pi_8 : Real := Real.pi / 8
noncomputable def pi_12 : Real := Real.pi / 12
noncomputable def pi_4 : Real := Real.pi / 4
noncomputable def pi_6 : Real := Real.pi / 6

/-- The smallest positive integer m such that R^(m)(l) = l
where the transformation R(l) is described as:
l is reflected in l1 (angle pi/8), then the resulting line is
reflected in l2 (angle pi/12) -/
theorem smallest_m_for_reflection :
  ∃ (m : ℕ), m > 0 ∧ ∀ (k : ℤ), m = 12 * k + 12 := by
sorry

end smallest_m_for_reflection_l1035_103513


namespace percentage_caught_sampling_candy_l1035_103562

theorem percentage_caught_sampling_candy
  (S : ℝ) (C : ℝ)
  (h1 : 0.1 * S = 0.1 * 24.444444444444443) -- 10% of the customers who sample the candy are not caught
  (h2 : S = 24.444444444444443)  -- The total percent of all customers who sample candy is 24.444444444444443%
  :
  C = 0.9 * 24.444444444444443 := -- Equivalent \( C \approx 22 \% \)
by
  sorry

end percentage_caught_sampling_candy_l1035_103562


namespace ratio_of_radii_l1035_103548

theorem ratio_of_radii (r R : ℝ) (k : ℝ) (h1 : R > r) (h2 : π * R^2 - π * r^2 = k * π * r^2) :
  R / r = Real.sqrt (k + 1) :=
sorry

end ratio_of_radii_l1035_103548


namespace ax5_by5_eq_28616_l1035_103582

variables (a b x y : ℝ)

theorem ax5_by5_eq_28616
  (h1 : a * x + b * y = 1)
  (h2 : a * x^2 + b * y^2 = 9)
  (h3 : a * x^3 + b * y^3 = 28)
  (h4 : a * x^4 + b * y^4 = 96) :
  a * x^5 + b * y^5 = 28616 :=
sorry

end ax5_by5_eq_28616_l1035_103582


namespace remainder_a52_div_52_l1035_103535

def a_n (n : ℕ) : ℕ := 
  (List.range (n + 1)).foldl (λ acc x => acc * 10 ^ (Nat.digits 10 x).length + x) 0

theorem remainder_a52_div_52 : (a_n 52) % 52 = 28 := 
  by
  sorry

end remainder_a52_div_52_l1035_103535


namespace max_x_lcm_max_x_lcm_value_l1035_103549

theorem max_x_lcm (x : ℕ) (h1 : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 :=
  sorry

theorem max_x_lcm_value (x : ℕ) (h1 : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
  sorry

end max_x_lcm_max_x_lcm_value_l1035_103549


namespace binomial_coefficient_divisible_by_prime_binomial_coefficient_extreme_cases_l1035_103512

-- Definitions and lemma statement
theorem binomial_coefficient_divisible_by_prime
  {p k : ℕ} (hp : Prime p) (hk : 0 < k) (hkp : k < p) :
  p ∣ Nat.choose p k := 
sorry

-- Theorem for k = 0 and k = p cases
theorem binomial_coefficient_extreme_cases {p : ℕ} (hp : Prime p) :
  Nat.choose p 0 = 1 ∧ Nat.choose p p = 1 :=
sorry

end binomial_coefficient_divisible_by_prime_binomial_coefficient_extreme_cases_l1035_103512


namespace g_five_eq_one_l1035_103565

noncomputable def g : ℝ → ℝ := sorry

axiom g_mul (x y : ℝ) : g (x * y) = g x * g y
axiom g_zero_ne_zero : g 0 ≠ 0

theorem g_five_eq_one : g 5 = 1 := by
  sorry

end g_five_eq_one_l1035_103565


namespace two_distinct_nonzero_complex_numbers_l1035_103543

noncomputable def count_distinct_nonzero_complex_numbers_satisfying_conditions : ℕ :=
sorry

theorem two_distinct_nonzero_complex_numbers :
  count_distinct_nonzero_complex_numbers_satisfying_conditions = 2 :=
sorry

end two_distinct_nonzero_complex_numbers_l1035_103543


namespace composite_product_quotient_l1035_103537

def first_seven_composite := [4, 6, 8, 9, 10, 12, 14]
def next_eight_composite := [15, 16, 18, 20, 21, 22, 24, 25]

noncomputable def product {α : Type*} [Monoid α] (l : List α) : α :=
  l.foldl (· * ·) 1

theorem composite_product_quotient : 
  (product first_seven_composite : ℚ) / (product next_eight_composite : ℚ) = 1 / 2475 := 
by 
  sorry

end composite_product_quotient_l1035_103537


namespace unique_solution_for_k_l1035_103506

theorem unique_solution_for_k : 
  ∃! k : ℚ, k ≠ 0 ∧ (∀ x : ℚ, (x + 3) / (k * x - 2) = x ↔ x = -2) :=
by
  sorry

end unique_solution_for_k_l1035_103506


namespace product_of_abc_l1035_103556

-- Define the constants and conditions
variables (a b c m : ℝ)
axiom h1 : a + b + c = 180
axiom h2 : 5 * a = m
axiom h3 : b = m + 12
axiom h4 : c = m - 6

-- Prove that the product of a, b, and c is 42184
theorem product_of_abc : a * b * c = 42184 :=
by {
  sorry
}

end product_of_abc_l1035_103556


namespace final_sale_price_is_correct_l1035_103514

-- Define the required conditions
def original_price : ℝ := 1200.00
def first_discount_rate : ℝ := 0.10
def second_discount_rate : ℝ := 0.20
def final_discount_rate : ℝ := 0.05

-- Define the expression to calculate the sale price after the discounts
def first_discount_price := original_price * (1 - first_discount_rate)
def second_discount_price := first_discount_price * (1 - second_discount_rate)
def final_sale_price := second_discount_price * (1 - final_discount_rate)

-- Prove that the final sale price equals $820.80
theorem final_sale_price_is_correct : final_sale_price = 820.80 := by
  sorry

end final_sale_price_is_correct_l1035_103514


namespace product_of_five_consecutive_integers_not_square_l1035_103502

theorem product_of_five_consecutive_integers_not_square (a : ℕ) :
  ¬ ∃ b c d e : ℕ, b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ ∃ k : ℕ, (a * b * c * d * e) = k^2 :=
by
  sorry

end product_of_five_consecutive_integers_not_square_l1035_103502


namespace adam_spent_money_on_ferris_wheel_l1035_103526

def tickets_bought : ℕ := 13
def tickets_left : ℕ := 4
def ticket_cost : ℕ := 9
def tickets_used : ℕ := tickets_bought - tickets_left

theorem adam_spent_money_on_ferris_wheel :
  tickets_used * ticket_cost = 81 :=
by
  sorry

end adam_spent_money_on_ferris_wheel_l1035_103526


namespace sqrt_mixed_number_eq_l1035_103590

noncomputable def mixed_number : ℝ := 8 + 1 / 9

theorem sqrt_mixed_number_eq : Real.sqrt (8 + 1 / 9) = Real.sqrt 73 / 3 := by
  sorry

end sqrt_mixed_number_eq_l1035_103590


namespace simplest_common_denominator_l1035_103507

theorem simplest_common_denominator (x y : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (d : ℤ), d = x^2 * y^2 ∧ ∀ (a b : ℤ), 
    (∃ (k : ℤ), a = k * (x^2 * y)) ∧ (∃ (m : ℤ), b = m * (x * y^2)) → d = lcm a b :=
by
  sorry

end simplest_common_denominator_l1035_103507


namespace sequence_an_l1035_103576

theorem sequence_an (a : ℕ → ℝ) (h0 : a 1 = 1)
  (h1 : ∀ n, 4 * a n * a (n + 1) = (a n + a (n + 1) - 1)^2)
  (h2 : ∀ n > 1, a n > a (n - 1)) :
  ∀ n, a n = n^2 := 
sorry

end sequence_an_l1035_103576


namespace find_k_l1035_103546

noncomputable def f (x : ℝ) : ℝ := 7 * x^3 - 1 / x + 5
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := x^3 - k

theorem find_k (k : ℝ) : 
  f 3 - g 3 k = 5 → k = - 485 / 3 :=
by
  sorry

end find_k_l1035_103546


namespace shares_distribution_correct_l1035_103530

def shares_distributed (a b c d e : ℕ) : Prop :=
  a = 50 ∧ b = 100 ∧ c = 300 ∧ d = 150 ∧ e = 600

theorem shares_distribution_correct (a b c d e : ℕ) :
  (a = (1/2 : ℚ) * b)
  ∧ (b = (1/3 : ℚ) * c)
  ∧ (c = 2 * d)
  ∧ (d = (1/4 : ℚ) * e)
  ∧ (a + b + c + d + e = 1200) → shares_distributed a b c d e :=
sorry

end shares_distribution_correct_l1035_103530


namespace number_153_satisfies_l1035_103586

noncomputable def sumOfCubes (n : ℕ) : ℕ :=
  (n % 10)^3 + ((n / 10) % 10)^3 + ((n / 100) % 10)^3

theorem number_153_satisfies :
  (sumOfCubes 153) = 153 ∧ 
  (153 % 10 ≠ 0) ∧ ((153 / 10) % 10 ≠ 0) ∧ ((153 / 100) % 10 ≠ 0) ∧ 
  153 ≠ 1 :=
by {
  sorry
}

end number_153_satisfies_l1035_103586


namespace debby_jogged_total_l1035_103585

theorem debby_jogged_total :
  let monday_distance := 2
  let tuesday_distance := 5
  let wednesday_distance := 9
  monday_distance + tuesday_distance + wednesday_distance = 16 :=
by
  sorry

end debby_jogged_total_l1035_103585


namespace AgOH_moles_formed_l1035_103593

noncomputable def number_of_moles_of_AgOH (n_AgNO3 n_NaOH : ℕ) : ℕ :=
  if n_AgNO3 = n_NaOH then n_AgNO3 else 0

theorem AgOH_moles_formed :
  number_of_moles_of_AgOH 3 3 = 3 := by
  sorry

end AgOH_moles_formed_l1035_103593


namespace grace_earnings_september_l1035_103540

def charge_small_lawn_per_hour := 6
def charge_large_lawn_per_hour := 10
def charge_pull_small_weeds_per_hour := 11
def charge_pull_large_weeds_per_hour := 15
def charge_small_mulch_per_hour := 9
def charge_large_mulch_per_hour := 13

def hours_small_lawn := 20
def hours_large_lawn := 43
def hours_small_weeds := 4
def hours_large_weeds := 5
def hours_small_mulch := 6
def hours_large_mulch := 4

def earnings_small_lawn := hours_small_lawn * charge_small_lawn_per_hour
def earnings_large_lawn := hours_large_lawn * charge_large_lawn_per_hour
def earnings_small_weeds := hours_small_weeds * charge_pull_small_weeds_per_hour
def earnings_large_weeds := hours_large_weeds * charge_pull_large_weeds_per_hour
def earnings_small_mulch := hours_small_mulch * charge_small_mulch_per_hour
def earnings_large_mulch := hours_large_mulch * charge_large_mulch_per_hour

def total_earnings : ℕ :=
  earnings_small_lawn + earnings_large_lawn + earnings_small_weeds + earnings_large_weeds +
  earnings_small_mulch + earnings_large_mulch

theorem grace_earnings_september : total_earnings = 775 :=
by
  sorry

end grace_earnings_september_l1035_103540


namespace total_cats_in_center_l1035_103566

def cats_training_center : ℕ := 45
def cats_can_fetch : ℕ := 25
def cats_can_meow : ℕ := 40
def cats_jump_and_fetch : ℕ := 15
def cats_fetch_and_meow : ℕ := 20
def cats_jump_and_meow : ℕ := 23
def cats_all_three : ℕ := 10
def cats_none : ℕ := 5

theorem total_cats_in_center :
  (cats_training_center - (cats_jump_and_fetch + cats_jump_and_meow - cats_all_three)) +
  (cats_all_three) +
  (cats_fetch_and_meow - cats_all_three) +
  (cats_jump_and_fetch - cats_all_three) +
  (cats_jump_and_meow - cats_all_three) +
  cats_none = 67 := by
  sorry

end total_cats_in_center_l1035_103566


namespace math_problem_l1035_103577

noncomputable def proof_problem (a b c d : ℝ) : Prop :=
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48

theorem math_problem (a b c d : ℝ)
  (h1 : a + b + c + d = 6)
  (h2 : a^2 + b^2 + c^2 + d^2 = 12) :
  proof_problem a b c d :=
by
  sorry

end math_problem_l1035_103577


namespace jeans_sold_l1035_103551

-- Definitions based on conditions
def price_per_jean : ℤ := 11
def price_per_tee : ℤ := 8
def tees_sold : ℤ := 7
def total_money : ℤ := 100

-- Proof statement
theorem jeans_sold (J : ℤ)
  (h1 : price_per_jean = 11)
  (h2 : price_per_tee = 8)
  (h3 : tees_sold = 7)
  (h4 : total_money = 100) :
  J = 4 :=
by
  sorry

end jeans_sold_l1035_103551


namespace pyramid_top_row_missing_number_l1035_103591

theorem pyramid_top_row_missing_number (a b c d e f g : ℕ)
  (h₁ : b * c = 720)
  (h₂ : a * b = 240)
  (h₃ : c * d = 1440)
  (h₄ : c = 6)
  : a = 120 :=
by
  sorry

end pyramid_top_row_missing_number_l1035_103591


namespace arithmetic_seq_formula_l1035_103521

variable (a : ℕ → ℤ)

-- Given conditions
axiom h1 : a 1 + a 2 + a 3 = 0
axiom h2 : a 4 + a 5 + a 6 = 18

-- Goal: general formula for the arithmetic sequence
theorem arithmetic_seq_formula (n : ℕ) : a n = 2 * n - 4 := by
  sorry

end arithmetic_seq_formula_l1035_103521


namespace expression_simplifies_to_62_l1035_103553

theorem expression_simplifies_to_62 (a b c : ℕ) (h1 : a = 14) (h2 : b = 19) (h3 : c = 29) :
  (a^2 * (1 / b - 1 / c) + b^2 * (1 / c - 1 / a) + c^2 * (1 / a - 1 / b)) / 
  (a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)) = 62 := by {
  sorry -- Proof goes here
}

end expression_simplifies_to_62_l1035_103553


namespace smallest_non_multiple_of_5_abundant_l1035_103561

def proper_divisors (n : ℕ) : List ℕ := List.filter (fun d => d ∣ n ∧ d < n) (List.range (n + 1))

def is_abundant (n : ℕ) : Prop := (proper_divisors n).sum > n

def is_not_multiple_of_5 (n : ℕ) : Prop := ¬ (5 ∣ n)

theorem smallest_non_multiple_of_5_abundant : ∃ n, is_abundant n ∧ is_not_multiple_of_5 n ∧ 
  ∀ m, is_abundant m ∧ is_not_multiple_of_5 m → n ≤ m :=
  sorry

end smallest_non_multiple_of_5_abundant_l1035_103561


namespace percentage_of_x_eq_y_l1035_103536

theorem percentage_of_x_eq_y
  (x y : ℝ) 
  (h : 0.60 * (x - y) = 0.20 * (x + y)) :
  y = 0.50 * x := 
sorry

end percentage_of_x_eq_y_l1035_103536


namespace chord_line_eq_l1035_103573

open Real

def ellipse (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 9 = 1

def bisecting_point (a b : ℝ) : Prop :=
  a = 4 ∧ b = 2

theorem chord_line_eq :
  (∃ (k : ℝ), ∀ (x y : ℝ), ellipse x y → bisecting_point ((x + y) / 2) ((x + y) / 2) → y - 2 = k * (x - 4)) →
  (∃ (x y : ℝ), ellipse x y ∧ x + 2 * y - 8 = 0) :=
by
  sorry

end chord_line_eq_l1035_103573


namespace triangle_area_l1035_103594

noncomputable def s (a b c : ℝ) : ℝ := (a + b + c) / 2
noncomputable def area (a b c : ℝ) : ℝ := Real.sqrt (s a b c * (s a b c - a) * (s a b c - b) * (s a b c - c))

theorem triangle_area (a b c : ℝ) (ha : a = 13) (hb : b = 12) (hc : c = 5) : area a b c = 30 := by
  rw [ha, hb, hc]
  show area 13 12 5 = 30
  -- manually calculate and reduce the expression to verify the theorem
  sorry

end triangle_area_l1035_103594


namespace combination_sum_l1035_103567

theorem combination_sum :
  (Nat.choose 7 4) + (Nat.choose 7 3) = 70 := by
  sorry

end combination_sum_l1035_103567


namespace F_atoms_in_compound_l1035_103584

-- Given conditions
def atomic_weight_Al : Real := 26.98
def atomic_weight_F : Real := 19.00
def molecular_weight : Real := 84

-- Defining the assertion: number of F atoms in the compound
def number_of_F_atoms (n : Real) : Prop :=
  molecular_weight = atomic_weight_Al + n * atomic_weight_F

-- Proving the assertion that the number of F atoms is approximately 3
theorem F_atoms_in_compound : number_of_F_atoms 3 :=
  by
  sorry

end F_atoms_in_compound_l1035_103584


namespace largest_number_is_C_l1035_103515

theorem largest_number_is_C (A B C D E : ℝ) 
  (hA : A = 0.989) 
  (hB : B = 0.9098) 
  (hC : C = 0.9899) 
  (hD : D = 0.9009) 
  (hE : E = 0.9809) : 
  C > A ∧ C > B ∧ C > D ∧ C > E := 
by 
  sorry

end largest_number_is_C_l1035_103515


namespace quotient_of_501_div_0_point_5_l1035_103554

theorem quotient_of_501_div_0_point_5 : 501 / 0.5 = 1002 := by
  sorry

end quotient_of_501_div_0_point_5_l1035_103554


namespace total_cans_collected_l1035_103520

theorem total_cans_collected :
  let cans_in_first_bag := 5
  let cans_in_second_bag := 7
  let cans_in_third_bag := 12
  let cans_in_fourth_bag := 4
  let cans_in_fifth_bag := 8
  let cans_in_sixth_bag := 10
  let cans_in_seventh_bag := 15
  let cans_in_eighth_bag := 6
  let cans_in_ninth_bag := 5
  let cans_in_tenth_bag := 13
  let total_cans := cans_in_first_bag + cans_in_second_bag + cans_in_third_bag + cans_in_fourth_bag + cans_in_fifth_bag + cans_in_sixth_bag + cans_in_seventh_bag + cans_in_eighth_bag + cans_in_ninth_bag + cans_in_tenth_bag
  total_cans = 85 :=
by
  sorry

end total_cans_collected_l1035_103520


namespace hyperbola_asymptotes_l1035_103500

theorem hyperbola_asymptotes (x y : ℝ) (h : y^2 / 16 - x^2 / 9 = (1 : ℝ)) :
  ∃ (m : ℝ), (m = 4 / 3) ∨ (m = -4 / 3) :=
sorry

end hyperbola_asymptotes_l1035_103500


namespace mostSuitableForComprehensiveSurvey_l1035_103523

-- Definitions of conditions
def optionA := "Understanding the sleep time of middle school students nationwide"
def optionB := "Understanding the water quality of a river"
def optionC := "Surveying the vision of all classmates"
def optionD := "Surveying the number of fish in a pond"

-- Define the notion of being the most suitable option for a comprehensive survey
def isSuitableForComprehensiveSurvey (option : String) := option = optionC

-- The theorem statement
theorem mostSuitableForComprehensiveSurvey : isSuitableForComprehensiveSurvey optionC := by
  -- This is the Lean 4 statement where we accept the hypotheses
  -- and conclude the theorem. Proof is omitted with "sorry".
  sorry

end mostSuitableForComprehensiveSurvey_l1035_103523


namespace oranges_less_per_student_l1035_103595

def total_students : ℕ := 12
def total_oranges : ℕ := 108
def bad_oranges : ℕ := 36

theorem oranges_less_per_student :
  (total_oranges / total_students) - ((total_oranges - bad_oranges) / total_students) = 3 :=
by
  sorry

end oranges_less_per_student_l1035_103595


namespace multiple_of_denominator_l1035_103599

def denominator := 5
def numerator := denominator + 4

theorem multiple_of_denominator:
  (numerator + 6) = 3 * denominator :=
by
  -- Proof steps go here
  sorry

end multiple_of_denominator_l1035_103599


namespace rick_bought_30_guppies_l1035_103533

theorem rick_bought_30_guppies (G : ℕ) (T C : ℕ) 
  (h1 : T = 4 * C) 
  (h2 : C = 2 * G) 
  (h3 : G + C + T = 330) : 
  G = 30 := 
by 
  sorry

end rick_bought_30_guppies_l1035_103533


namespace intersection_complement_l1035_103559

-- Defining the sets A and B
def setA : Set ℝ := { x | -3 < x ∧ x < 3 }
def setB : Set ℝ := { x | x < -2 }
def complementB : Set ℝ := { x | x ≥ -2 }

-- The theorem to be proved
theorem intersection_complement :
  setA ∩ complementB = { x | -2 ≤ x ∧ x < 3 } :=
by
  sorry

end intersection_complement_l1035_103559


namespace Masha_initial_ball_count_l1035_103564

theorem Masha_initial_ball_count (r w n p : ℕ) (h1 : r + n * w = 101) (h2 : p * r + w = 103) (hn : n ≠ 0) :
  r + w = 51 ∨ r + w = 68 :=
  sorry

end Masha_initial_ball_count_l1035_103564


namespace find_m_n_l1035_103516

theorem find_m_n (x m n : ℝ) : (x + 4) * (x - 2) = x^2 + m * x + n → m = 2 ∧ n = -8 := 
by
  intro h
  -- Steps to prove the theorem would be here
  sorry

end find_m_n_l1035_103516


namespace cube_volume_l1035_103539

theorem cube_volume (s : ℝ) (h : s ^ 2 = 64) : s ^ 3 = 512 :=
sorry

end cube_volume_l1035_103539


namespace triangle_side_lengths_l1035_103579

-- Define the problem
variables {r: ℝ} (h_a h_b h_c a b c : ℝ)
variable (sum_of_heights : h_a + h_b + h_c = 13)
variable (r_value : r = 4 / 3)
variable (height_relation : 1/h_a + 1/h_b + 1/h_c = 3/4)

-- Define the theorem to be proven
theorem triangle_side_lengths (h_a h_b h_c : ℝ)
  (sum_of_heights : h_a + h_b + h_c = 13) 
  (r_value : r = 4 / 3)
  (height_relation : 1/h_a + 1/h_b + 1/h_c = 3/4) :
  (a, b, c) = (32 / Real.sqrt 15, 24 / Real.sqrt 15, 16 / Real.sqrt 15) := 
sorry

end triangle_side_lengths_l1035_103579


namespace supplement_of_supplement_of_58_l1035_103501

theorem supplement_of_supplement_of_58 (α : ℝ) (h : α = 58) : 180 - (180 - α) = 58 :=
by
  sorry

end supplement_of_supplement_of_58_l1035_103501


namespace sequence_monotonically_increasing_l1035_103544

noncomputable def a (n : ℕ) : ℝ := (n - 1 : ℝ) / (n + 1 : ℝ)

theorem sequence_monotonically_increasing : ∀ n : ℕ, a (n + 1) > a n :=
by
  sorry

end sequence_monotonically_increasing_l1035_103544


namespace function_decreasing_iff_a_neg_l1035_103568

variable (a : ℝ)

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 ≥ f x2

theorem function_decreasing_iff_a_neg (h : ∀ x : ℝ, (7 * a * x ^ 6) ≤ 0) : a < 0 :=
by
  sorry

end function_decreasing_iff_a_neg_l1035_103568


namespace simplify_logarithmic_expression_l1035_103534

theorem simplify_logarithmic_expression :
  (1 / (Real.logb 12 3 + 1) + 1 / (Real.logb 8 2 + 1) + 1 / (Real.logb 18 9 + 1) = 1) :=
sorry

end simplify_logarithmic_expression_l1035_103534


namespace triangular_number_is_perfect_square_l1035_103545

def is_triangular_number (T : ℕ) : Prop :=
∃ n : ℕ, T = n * (n + 1) / 2

def is_perfect_square (T : ℕ) : Prop :=
∃ y : ℕ, T = y * y

theorem triangular_number_is_perfect_square:
  ∀ (x_k : ℕ), 
    ((∃ n y : ℕ, (2 * n + 1)^2 - 8 * y^2 = 1 ∧ T_n = n * (n + 1) / 2 ∧ T_n = x_k^2 - 1 / 8) →
    (is_triangular_number T_n → is_perfect_square T_n)) :=
by
  sorry

end triangular_number_is_perfect_square_l1035_103545


namespace employees_in_january_l1035_103518

theorem employees_in_january (E : ℝ) (h : 500 = 1.15 * E) : E = 500 / 1.15 :=
by
  sorry

end employees_in_january_l1035_103518


namespace ten_times_hundred_eq_thousand_ten_times_thousand_eq_ten_thousand_hundreds_in_ten_thousand_tens_in_one_thousand_l1035_103592

theorem ten_times_hundred_eq_thousand : 10 * 100 = 1000 := 
by sorry

theorem ten_times_thousand_eq_ten_thousand : 10 * 1000 = 10000 := 
by sorry

theorem hundreds_in_ten_thousand : 10000 / 100 = 100 := 
by sorry

theorem tens_in_one_thousand : 1000 / 10 = 100 := 
by sorry

end ten_times_hundred_eq_thousand_ten_times_thousand_eq_ten_thousand_hundreds_in_ten_thousand_tens_in_one_thousand_l1035_103592


namespace shaded_area_l1035_103504

theorem shaded_area (r : ℝ) (α : ℝ) (β : ℝ) (h1 : r = 4) (h2 : α = 1/2) :
  β = 64 - 16 * Real.pi := by sorry

end shaded_area_l1035_103504


namespace find_k_l1035_103583

-- Definition of the vertices and conditions
variables {t k : ℝ}
def A : (ℝ × ℝ) := (0, 3)
def B : (ℝ × ℝ) := (0, k)
def C : (ℝ × ℝ) := (t, 10)
def D : (ℝ × ℝ) := (t, 0)

-- Condition that the area of the quadrilateral is 50 square units
def area_cond (height base1 base2 : ℝ) : Prop :=
  50 = (1 / 2) * height * (base1 + base2)

-- Stating the problem in Lean
theorem find_k
  (ht : t = 5)
  (hk : k > 3) 
  (t_pos : t > 0)
  (area : area_cond t (k - 3) 10) :
  k = 13 :=
  sorry

end find_k_l1035_103583


namespace train_length_1080_l1035_103527

def length_of_train (speed time : ℕ) : ℕ := speed * time

theorem train_length_1080 (speed time : ℕ) (h1 : speed = 108) (h2 : time = 10) : length_of_train speed time = 1080 := by
  sorry

end train_length_1080_l1035_103527


namespace actual_revenue_percentage_of_projected_l1035_103541

theorem actual_revenue_percentage_of_projected (R : ℝ) (hR : R > 0) :
  (0.75 * R) / (1.2 * R) * 100 = 62.5 := 
by
  sorry

end actual_revenue_percentage_of_projected_l1035_103541


namespace paint_fraction_used_l1035_103597

theorem paint_fraction_used (initial_paint: ℕ) (first_week_fraction: ℚ) (total_paint_used: ℕ) (remaining_paint_after_first_week: ℕ) :
  initial_paint = 360 →
  first_week_fraction = 1/3 →
  total_paint_used = 168 →
  remaining_paint_after_first_week = initial_paint - initial_paint * first_week_fraction →
  (total_paint_used - initial_paint * first_week_fraction) / remaining_paint_after_first_week = 1/5 := 
by
  sorry

end paint_fraction_used_l1035_103597


namespace nine_op_ten_l1035_103510

def op (A B : ℕ) : ℚ := (1 : ℚ) / (A * B) + (1 : ℚ) / ((A + 1) * (B + 2))

theorem nine_op_ten : op 9 10 = 7 / 360 := by
  sorry

end nine_op_ten_l1035_103510


namespace range_of_x_l1035_103532

theorem range_of_x (x : ℝ) (h : |2 * x + 1| + |2 * x - 5| = 6) : -1 / 2 ≤ x ∧ x ≤ 5 / 2 := by
  sorry

end range_of_x_l1035_103532


namespace quadratic_trinomial_negative_value_l1035_103596

theorem quadratic_trinomial_negative_value
  (a b c : ℝ)
  (h1 : b^2 ≥ 4 * c)
  (h2 : 1 ≥ 4 * a * c)
  (h3 : b^2 ≥ 4 * a) :
  ∃ x : ℝ, a * x^2 + b * x + c < 0 :=
by
  sorry

end quadratic_trinomial_negative_value_l1035_103596


namespace no_real_roots_other_than_zero_l1035_103524

theorem no_real_roots_other_than_zero (k : ℝ) (h : k ≠ 0):
  ¬(∃ x : ℝ, x^2 + 2 * k * x + 3 * k^2 = 0) :=
by
  sorry

end no_real_roots_other_than_zero_l1035_103524


namespace parabola_intersection_sum_zero_l1035_103552

theorem parabola_intersection_sum_zero
  (x_1 x_2 x_3 x_4 y_1 y_2 y_3 y_4 : ℝ)
  (h1 : ∀ x, ∃ y, y = (x - 2)^2 + 1)
  (h2 : ∀ y, ∃ x, x - 1 = (y + 2)^2)
  (h_intersect : (∃ x y, (y = (x - 2)^2 + 1) ∧ (x - 1 = (y + 2)^2))) :
  x_1 + x_2 + x_3 + x_4 + y_1 + y_2 + y_3 + y_4 = 0 :=
sorry

end parabola_intersection_sum_zero_l1035_103552


namespace height_relationship_l1035_103525

theorem height_relationship (r1 r2 h1 h2 : ℝ) (h_radii : r2 = 1.2 * r1) (h_volumes : π * r1^2 * h1 = π * r2^2 * h2) : h1 = 1.44 * h2 :=
by
  sorry

end height_relationship_l1035_103525


namespace greatest_value_of_sum_l1035_103531

variable (x y : ℝ)

-- Conditions
axiom sum_of_squares : x^2 + y^2 = 130
axiom product : x * y = 36

-- Statement to prove
theorem greatest_value_of_sum : x + y ≤ Real.sqrt 202 := sorry

end greatest_value_of_sum_l1035_103531


namespace A_eq_B_l1035_103575

noncomputable def A := Real.sqrt 5 + Real.sqrt (22 + 2 * Real.sqrt 5)
noncomputable def B := Real.sqrt (11 + 2 * Real.sqrt 29) 
                      + Real.sqrt (16 - 2 * Real.sqrt 29 
                                   + 2 * Real.sqrt (55 - 10 * Real.sqrt 29))

theorem A_eq_B : A = B := 
  sorry

end A_eq_B_l1035_103575


namespace petya_time_comparison_l1035_103588

variables (D V : ℝ) (hD_pos : D > 0) (hV_pos : V > 0)

theorem petya_time_comparison (hD_pos : D > 0) (hV_pos : V > 0) :
  (41 * D / (40 * V)) > (D / V) :=
by
  sorry

end petya_time_comparison_l1035_103588


namespace previous_salary_l1035_103598

theorem previous_salary (P : ℝ) (h : 1.05 * P = 2100) : P = 2000 :=
by
  sorry

end previous_salary_l1035_103598


namespace evaluate_expression_l1035_103570

theorem evaluate_expression : (1:ℤ)^10 + (-1:ℤ)^8 + (-1:ℤ)^7 + (1:ℤ)^5 = 2 := by
  sorry

end evaluate_expression_l1035_103570


namespace percent_of_12356_equals_1_2356_l1035_103569

theorem percent_of_12356_equals_1_2356 (p : ℝ) (h : p * 12356 = 1.2356) : p = 0.0001 := sorry

end percent_of_12356_equals_1_2356_l1035_103569


namespace distinct_lines_isosceles_not_equilateral_l1035_103503

-- Define a structure for an isosceles triangle that is not equilateral
structure IsoscelesButNotEquilateralTriangle :=
  (a b c : ℕ)    -- sides of the triangle
  (h₁ : a = b)   -- two equal sides
  (h₂ : a ≠ c)   -- not equilateral (not all three sides are equal)

-- Define that the number of distinct lines representing altitudes, medians, and interior angle bisectors is 5
theorem distinct_lines_isosceles_not_equilateral (T : IsoscelesButNotEquilateralTriangle) : 
  ∃ (n : ℕ), n = 5 :=
by
  sorry

end distinct_lines_isosceles_not_equilateral_l1035_103503


namespace Z_is_divisible_by_10001_l1035_103517

theorem Z_is_divisible_by_10001
    (Z : ℕ) (a b c d : ℕ) (ha : a ≠ 0)
    (hZ : Z = 1000 * 10001 * a + 100 * 10001 * b + 10 * 10001 * c + 10001 * d)
    : 10001 ∣ Z :=
by {
    -- Proof omitted
    sorry
}

end Z_is_divisible_by_10001_l1035_103517


namespace simplify_tan_expression_simplify_complex_expression_l1035_103519

-- Problem 1
theorem simplify_tan_expression (α : ℝ) (hα : 0 < α ∧ α < 2 * π ∧ α > 3 / 2 * π) : 
  (Real.tan α + Real.sqrt ((1 / (Real.cos α)^2) - 1) + 2 * (Real.sin α)^2 + 2 * (Real.cos α)^2 = 2) :=
sorry

-- Problem 2
theorem simplify_complex_expression (α : ℝ) (hα : 0 < α ∧ α < 2 * π ∧ α > 3 / 2 * π) : 
  (Real.sin (α + π) * Real.tan (π - α) * Real.cos (2 * π - α) / (Real.sin (π - α) * Real.sin (π / 2 + α)) + Real.cos (5 * π / 2) = - Real.cos α) :=
sorry

end simplify_tan_expression_simplify_complex_expression_l1035_103519


namespace vanAubel_theorem_l1035_103529

variables (A B C O A1 B1 C1 : Type)
variables (CA1 A1B CB1 B1A CO OC1 : ℝ)

-- Given Conditions
axiom condition1 : CB1 / B1A = 1
axiom condition2 : CO / OC1 = 2

-- Van Aubel's theorem statement
theorem vanAubel_theorem : (CO / OC1) = (CA1 / A1B) + (CB1 / B1A) := by
  sorry

end vanAubel_theorem_l1035_103529


namespace day_of_week_nminus1_l1035_103557

theorem day_of_week_nminus1 (N : ℕ) 
  (h1 : (250 % 7 = 3 ∧ (250 / 7 * 7 + 3 = 250)) ∧ (150 % 7 = 3 ∧ (150 / 7 * 7 + 3 = 150))) :
  (50 % 7 = 0 ∧ (50 / 7 * 7 = 50)) := 
sorry

end day_of_week_nminus1_l1035_103557


namespace find_p_q_l1035_103580

theorem find_p_q 
  (p q: ℚ)
  (a : ℚ × ℚ × ℚ × ℚ := (4, p, -2, 1))
  (b : ℚ × ℚ × ℚ × ℚ := (3, 2, q, -1))
  (orthogonal : (4 * 3 + p * 2 + (-2) * q + 1 * (-1) = 0))
  (equal_magnitudes : (4^2 + p^2 + (-2)^2 + 1^2 = 3^2 + 2^2 + q^2 + (-1)^2))
  : p = -93/44 ∧ q = 149/44 := 
  by 
    sorry

end find_p_q_l1035_103580


namespace sign_of_k_l1035_103572

variable (k x y : ℝ)
variable (A B : ℝ × ℝ)
variable (y₁ y₂ : ℝ)
variable (h₁ : A = (-2, y₁))
variable (h₂ : B = (5, y₂))
variable (h₃ : y₁ = k / -2)
variable (h₄ : y₂ = k / 5)
variable (h₅ : y₁ > y₂)
variable (h₀ : k ≠ 0)

-- We need to prove that k < 0
theorem sign_of_k (A B : ℝ × ℝ) (y₁ y₂ k : ℝ) 
  (h₁ : A = (-2, y₁)) 
  (h₂ : B = (5, y₂)) 
  (h₃ : y₁ = k / -2) 
  (h₄ : y₂ = k / 5) 
  (h₅ : y₁ > y₂) 
  (h₀ : k ≠ 0) : k < 0 := 
by
  sorry

end sign_of_k_l1035_103572


namespace average_daily_low_temperature_l1035_103550

theorem average_daily_low_temperature (temps : List ℕ) (h_len : temps.length = 5) 
  (h_vals : temps = [40, 47, 45, 41, 39]) : 
  (temps.sum / 5 : ℝ) = 42.4 := 
by
  sorry

end average_daily_low_temperature_l1035_103550


namespace geometric_sequence_sum_l1035_103589

-- Definition of the sum of the first n terms of a geometric sequence
variable (S : ℕ → ℝ)

-- Conditions given in the problem
def S_n_given (n : ℕ) : Prop := S n = 36
def S_2n_given (n : ℕ) : Prop := S (2 * n) = 42

-- Theorem to prove
theorem geometric_sequence_sum (n : ℕ) (S : ℕ → ℝ) 
    (h1 : S n = 36) (h2 : S (2 * n) = 42) : S (3 * n) = 48 := sorry

end geometric_sequence_sum_l1035_103589


namespace geometric_sequence_sum_four_l1035_103578

theorem geometric_sequence_sum_four (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a (n + 1) = a n * q)
  (h2 : q ≠ 1)
  (h3 : -3 * a 0 = -2 * a 1 - a 2)
  (h4 : a 0 = 1) : 
  S 4 = -20 :=
sorry

end geometric_sequence_sum_four_l1035_103578


namespace correct_division_incorrect_addition_incorrect_multiplication_incorrect_squaring_only_correct_operation_l1035_103560

theorem correct_division (x : ℝ) : x^6 / x^3 = x^3 := by 
  sorry

theorem incorrect_addition (x : ℝ) : ¬(x^2 + x^3 = 2 * x^5) := by 
  sorry

theorem incorrect_multiplication (x : ℝ) : ¬(x^2 * x^3 = x^6) := by 
  sorry

theorem incorrect_squaring (x : ℝ) : ¬((-x^3) ^ 2 = -x^6) := by 
  sorry

theorem only_correct_operation (x : ℝ) : 
  (x^6 / x^3 = x^3) ∧ ¬(x^2 + x^3 = 2 * x^5) ∧ ¬(x^2 * x^3 = x^6) ∧ ¬((-x^3) ^ 2 = -x^6) := 
  by
    exact ⟨correct_division x, incorrect_addition x, incorrect_multiplication x,
           incorrect_squaring x⟩

end correct_division_incorrect_addition_incorrect_multiplication_incorrect_squaring_only_correct_operation_l1035_103560


namespace range_of_a_l1035_103542

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 4 → x^2 - 2 * x + 1 - a^2 < 0) ↔ (a < -3 ∨ a > 3) :=
sorry

end range_of_a_l1035_103542


namespace pipes_fill_tank_in_7_minutes_l1035_103563

theorem pipes_fill_tank_in_7_minutes (T : ℕ) (R_A R_B R_combined : ℚ) 
  (h1 : R_A = 1 / 56) 
  (h2 : R_B = 7 * R_A)
  (h3 : R_combined = R_A + R_B)
  (h4 : T = 1 / R_combined) : 
  T = 7 := by 
  sorry

end pipes_fill_tank_in_7_minutes_l1035_103563


namespace sum_m_n_l1035_103509

-- Declare the namespaces and definitions for the problem
namespace DelegateProblem

-- Condition: total number of delegates
def total_delegates : Nat := 12

-- Condition: number of delegates from each country
def delegates_per_country : Nat := 4

-- Computation of m and n such that their sum is 452
-- This follows from the problem statement and the solution provided
def m : Nat := 221
def n : Nat := 231

-- Theorem statement in Lean for proving m + n = 452
theorem sum_m_n : m + n = 452 := by
  -- Algebraic proof omitted
  sorry

end DelegateProblem

end sum_m_n_l1035_103509


namespace range_of_x_satisfying_inequality_l1035_103508

noncomputable def f : ℝ → ℝ := sorry -- f is some even and monotonically increasing function

theorem range_of_x_satisfying_inequality :
  (∀ x, f (-x) = f x) ∧ (∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y) → {x : ℝ | f x < f 1} = {x : ℝ | -1 < x ∧ x < 1} :=
by
  intro h
  sorry

end range_of_x_satisfying_inequality_l1035_103508


namespace determinant_scaling_l1035_103505

variable (p q r s : ℝ)

theorem determinant_scaling 
  (h : Matrix.det ![![p, q], ![r, s]] = 3) : 
  Matrix.det ![![2 * p, 2 * p + 5 * q], ![2 * r, 2 * r + 5 * s]] = 30 :=
by 
  sorry

end determinant_scaling_l1035_103505


namespace geometric_sequence_problem_l1035_103581

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∀ n, a n = a 0 * (1 / 2) ^ n

theorem geometric_sequence_problem 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : ∀ n, S n = (a 0 * (1 - (1 / 2 : ℝ) ^ n)) / (1 - (1 / 2)))
  (h3 : a 0 + a 2 = 5 / 2)
  (h4 : a 1 + a 3 = 5 / 4) :
  ∀ n, S n / a n = 2 ^ n - 1 :=
by
  sorry

end geometric_sequence_problem_l1035_103581


namespace find_principal_l1035_103538

-- Conditions as definitions
def amount : ℝ := 1120
def rate : ℝ := 0.05
def time : ℝ := 2

-- Required to add noncomputable due to the use of division and real numbers
noncomputable def principal : ℝ := amount / (1 + rate * time)

-- The main theorem statement which needs to be proved
theorem find_principal :
  principal = 1018.18 :=
sorry  -- Proof is not required; it is left as sorry

end find_principal_l1035_103538


namespace expression_rewrite_l1035_103555

theorem expression_rewrite :
  ∃ (d r s : ℚ), (∀ k : ℚ, 8*k^2 - 6*k + 16 = d*(k + r)^2 + s) ∧ s / r = -118 / 3 :=
by sorry

end expression_rewrite_l1035_103555


namespace area_of_BEIH_l1035_103511

def calculate_area_of_quadrilateral (A B C D E F I H : (ℝ × ℝ)) : ℝ := 
  sorry

theorem area_of_BEIH : 
  let A := (0, 3)
  let B := (0, 0)
  let C := (3, 0)
  let D := (3, 3)
  let E := (0, 1.5)
  let F := (1, 0)
  let I := (3 / 5, 9 / 5)
  let H := (3 / 4, 3 / 4)
  calculate_area_of_quadrilateral A B C D E F I H = 27 / 40 :=
sorry

end area_of_BEIH_l1035_103511


namespace frog_hops_ratio_l1035_103571

theorem frog_hops_ratio (S T F : ℕ) (h1 : S = 2 * T) (h2 : S = 18) (h3 : F + S + T = 99) :
  F / S = 4 / 1 :=
by
  sorry

end frog_hops_ratio_l1035_103571


namespace change_received_after_discounts_and_taxes_l1035_103574

theorem change_received_after_discounts_and_taxes :
  let price_wooden_toy : ℝ := 20
  let price_hat : ℝ := 10
  let tax_rate : ℝ := 0.08
  let discount_wooden_toys : ℝ := 0.15
  let discount_hats : ℝ := 0.10
  let quantity_wooden_toys : ℝ := 3
  let quantity_hats : ℝ := 4
  let amount_paid : ℝ := 200
  let cost_wooden_toys := quantity_wooden_toys * price_wooden_toy
  let discounted_cost_wooden_toys := cost_wooden_toys - (discount_wooden_toys * cost_wooden_toys)
  let cost_hats := quantity_hats * price_hat
  let discounted_cost_hats := cost_hats - (discount_hats * cost_hats)
  let total_cost_before_tax := discounted_cost_wooden_toys + discounted_cost_hats
  let tax := tax_rate * total_cost_before_tax
  let total_cost_after_tax := total_cost_before_tax + tax
  let change_received := amount_paid - total_cost_after_tax
  change_received = 106.04 := by
  -- All the conditions and intermediary steps are defined above, from problem to solution.
  sorry

end change_received_after_discounts_and_taxes_l1035_103574


namespace angle_bisector_length_l1035_103528

variable (a b : ℝ) (α l : ℝ)

theorem angle_bisector_length (ha : 0 < a) (hb : 0 < b) (hα : 0 < α) (hl : l = (2 * a * b * Real.cos (α / 2)) / (a + b)) :
  l = (2 * a * b * Real.cos (α / 2)) / (a + b) := by
  -- problem assumptions
  have h1 : a > 0 := ha
  have h2 : b > 0 := hb
  have h3 : α > 0 := hα
  -- conclusion
  exact hl

end angle_bisector_length_l1035_103528


namespace fraction_of_money_left_l1035_103558

theorem fraction_of_money_left (m c : ℝ) 
   (h1 : (1/5) * m = (1/3) * c) :
   (m - ((3/5) * m) = (2/5) * m) := by
  sorry

end fraction_of_money_left_l1035_103558


namespace units_digit_of_7_pow_6_cubed_l1035_103522

-- Define the repeating cycle of unit digits for powers of 7
def unit_digit_of_power_of_7 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 0 -- This case is actually unreachable given the modulus operation

-- Define the main problem statement
theorem units_digit_of_7_pow_6_cubed : unit_digit_of_power_of_7 (6 ^ 3) = 1 :=
by
  sorry

end units_digit_of_7_pow_6_cubed_l1035_103522


namespace quadratic_eq_solution_trig_expression_calc_l1035_103587

-- Part 1: Proof for the quadratic equation solution
theorem quadratic_eq_solution : ∀ (x : ℝ), x^2 - 4 * x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 :=
by
  sorry

-- Part 2: Proof for trigonometric expression calculation
theorem trig_expression_calc : (-1 : ℝ) ^ 2 + 2 * Real.sin (Real.pi / 3) - Real.tan (Real.pi / 4) = Real.sqrt 3 :=
by
  sorry

end quadratic_eq_solution_trig_expression_calc_l1035_103587
