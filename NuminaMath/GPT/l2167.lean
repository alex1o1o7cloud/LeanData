import Mathlib

namespace hilary_stalks_l2167_216776

-- Define the given conditions
def ears_per_stalk : ℕ := 4
def kernels_per_ear_first_half : ℕ := 500
def kernels_per_ear_second_half : ℕ := 600
def total_kernels : ℕ := 237600

-- Average number of kernels per ear
def average_kernels_per_ear : ℕ := (kernels_per_ear_first_half + kernels_per_ear_second_half) / 2

-- Total number of ears based on total kernels
noncomputable def total_ears : ℕ := total_kernels / average_kernels_per_ear

-- Total number of stalks based on total ears
noncomputable def total_stalks : ℕ := total_ears / ears_per_stalk

-- The main theorem to prove
theorem hilary_stalks : total_stalks = 108 :=
by
  sorry

end hilary_stalks_l2167_216776


namespace equation_transformation_correct_l2167_216782

theorem equation_transformation_correct :
  ∀ (x : ℝ), 
  6 * ((x - 1) / 2 - 1) = 6 * ((3 * x + 1) / 3) → 
  (3 * (x - 1) - 6 = 2 * (3 * x + 1)) :=
by
  intro x
  intro h
  sorry

end equation_transformation_correct_l2167_216782


namespace total_birds_in_marsh_l2167_216746

-- Given conditions
def initial_geese := 58
def doubled_geese := initial_geese * 2
def ducks := 37
def swans := 15
def herons := 22

-- Prove that the total number of birds is 190
theorem total_birds_in_marsh : 
  doubled_geese + ducks + swans + herons = 190 := 
by
  sorry

end total_birds_in_marsh_l2167_216746


namespace prob_both_hit_prob_exactly_one_hit_prob_at_least_one_hit_l2167_216713

-- Define events and their probabilities.
def prob_A : ℝ := 0.8
def prob_B : ℝ := 0.8

-- Given P(A and B) = P(A) * P(B)
def prob_AB : ℝ := prob_A * prob_B

-- Statements to prove
theorem prob_both_hit : prob_AB = 0.64 :=
by
  -- P(A and B) = 0.8 * 0.8 = 0.64
  exact sorry

theorem prob_exactly_one_hit : (prob_A * (1 - prob_B) + (1 - prob_A) * prob_B) = 0.32 :=
by
  -- P(A and not B) + P(not A and B) = 0.8 * 0.2 + 0.2 * 0.8 = 0.32
  exact sorry

theorem prob_at_least_one_hit : (1 - (1 - prob_A) * (1 - prob_B)) = 0.96 :=
by
  -- 1 - P(not A and not B) = 1 - 0.04 = 0.96
  exact sorry

end prob_both_hit_prob_exactly_one_hit_prob_at_least_one_hit_l2167_216713


namespace non_right_triangle_option_l2167_216708

-- Definitions based on conditions
def optionA (A B C : ℝ) : Prop := A + B = C
def optionB (A B C : ℝ) : Prop := A - B = C
def optionC (A B C : ℝ) : Prop := A / B = 1 / 2 ∧ B / C = 2 / 3
def optionD (A B C : ℝ) : Prop := A = B ∧ A = 3 * C

-- Given conditions for a right triangle
def is_right_triangle (A B C : ℝ) : Prop := ∃(θ : ℝ), θ = 90 ∧ (A = θ ∨ B = θ ∨ C = θ)

-- The proof problem
theorem non_right_triangle_option (A B C : ℝ) :
  optionD A B C ∧ ¬(is_right_triangle A B C) := sorry

end non_right_triangle_option_l2167_216708


namespace age_of_15th_student_l2167_216735

theorem age_of_15th_student : 
  let average_age_all_students := 15
  let number_of_students := 15
  let average_age_first_group := 13
  let number_of_students_first_group := 5
  let average_age_second_group := 16
  let number_of_students_second_group := 9
  let total_age_all_students := number_of_students * average_age_all_students
  let total_age_first_group := number_of_students_first_group * average_age_first_group
  let total_age_second_group := number_of_students_second_group * average_age_second_group
  total_age_all_students - (total_age_first_group + total_age_second_group) = 16 :=
by
  let average_age_all_students := 15
  let number_of_students := 15
  let average_age_first_group := 13
  let number_of_students_first_group := 5
  let average_age_second_group := 16
  let number_of_students_second_group := 9
  let total_age_all_students := number_of_students * average_age_all_students
  let total_age_first_group := number_of_students_first_group * average_age_first_group
  let total_age_second_group := number_of_students_second_group * average_age_second_group
  sorry

end age_of_15th_student_l2167_216735


namespace quad_intersects_x_axis_l2167_216723

theorem quad_intersects_x_axis (k : ℝ) :
  (∃ x : ℝ, k * x ^ 2 - 7 * x - 7 = 0) ↔ (k ≥ -7 / 4 ∧ k ≠ 0) :=
by sorry

end quad_intersects_x_axis_l2167_216723


namespace smallest_multiple_of_6_8_12_l2167_216700

theorem smallest_multiple_of_6_8_12 : ∃ n : ℕ, n > 0 ∧ n % 6 = 0 ∧ n % 8 = 0 ∧ n % 12 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 6 = 0 ∧ m % 8 = 0 ∧ m % 12 = 0) → n ≤ m := 
sorry

end smallest_multiple_of_6_8_12_l2167_216700


namespace roots_polynomial_equation_l2167_216768

noncomputable def rootsEquation (x y : ℝ) := x + y = 10 ∧ |x - y| = 12

theorem roots_polynomial_equation : ∃ (x y : ℝ), rootsEquation x y ∧ (x^2 - 10 * x - 11 = 0) := sorry

end roots_polynomial_equation_l2167_216768


namespace part1_part2_l2167_216770

def traditional_chinese_paintings : ℕ := 6
def oil_paintings : ℕ := 4
def watercolor_paintings : ℕ := 5

theorem part1 :
  traditional_chinese_paintings * oil_paintings * watercolor_paintings = 120 :=
by
  sorry

theorem part2 :
  (traditional_chinese_paintings * oil_paintings) + 
  (traditional_chinese_paintings * watercolor_paintings) + 
  (oil_paintings * watercolor_paintings) = 74 :=
by
  sorry

end part1_part2_l2167_216770


namespace distance_between_cars_l2167_216739

theorem distance_between_cars (t : ℝ) (v_kmh : ℝ) (v_ms : ℝ) :
  t = 1 ∧ v_kmh = 180 ∧ v_ms = v_kmh * 1000 / 3600 → 
  v_ms * t = 50 := 
by 
  sorry

end distance_between_cars_l2167_216739


namespace sum_xyz_eq_two_l2167_216744

-- Define the variables x, y, and z to be real numbers
variables (x y z : ℝ)

-- Given condition
def condition : Prop :=
  x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0

-- The theorem to prove
theorem sum_xyz_eq_two (h : condition x y z) : x + y + z = 2 :=
sorry

end sum_xyz_eq_two_l2167_216744


namespace problem_l2167_216743

theorem problem (a b : ℤ)
  (h1 : -2022 = -a)
  (h2 : -1 = -b) :
  a + b = 2023 :=
sorry

end problem_l2167_216743


namespace a_2_is_minus_1_l2167_216788
open Nat

variable (a S : ℕ → ℤ)

-- Conditions
axiom sum_first_n (n : ℕ) (hn : n > 0) : 2 * S n - n * a n = n
axiom S_20 : S 20 = -360

-- The problem statement to prove
theorem a_2_is_minus_1 : a 2 = -1 :=
by 
  sorry

end a_2_is_minus_1_l2167_216788


namespace larger_integer_is_21_l2167_216761

-- Setting up the conditions
def quotient_condition (a b : ℕ) : Prop := a / b = 7 / 3
def product_condition (a b : ℕ) : Prop := a * b = 189

-- Assertion: Prove larger of the two integers is 21
theorem larger_integer_is_21 (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) : max a b = 21 :=
by sorry

end larger_integer_is_21_l2167_216761


namespace cube_surface_area_increase_l2167_216747

theorem cube_surface_area_increase (s : ℝ) : 
  let original_surface_area := 6 * s^2
  let new_edge := 1.3 * s
  let new_surface_area := 6 * (new_edge)^2
  let percentage_increase := ((new_surface_area - original_surface_area) / original_surface_area) * 100
  percentage_increase = 69 := 
by
  sorry

end cube_surface_area_increase_l2167_216747


namespace inequality_positive_real_l2167_216797

theorem inequality_positive_real (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 :=
sorry

end inequality_positive_real_l2167_216797


namespace unique_n_for_50_percent_mark_l2167_216757

def exam_conditions (n : ℕ) : Prop :=
  let correct_first_20 : ℕ := 15
  let remaining : ℕ := n - 20
  let correct_remaining : ℕ := remaining / 3
  let total_correct : ℕ := correct_first_20 + correct_remaining
  total_correct * 2 = n

theorem unique_n_for_50_percent_mark : ∃! (n : ℕ), exam_conditions n := sorry

end unique_n_for_50_percent_mark_l2167_216757


namespace sum_T_mod_1000_l2167_216750

open Nat

def T (a b : ℕ) : ℕ :=
  if h : a + b ≤ 6 then Nat.choose 6 a * Nat.choose 6 b * Nat.choose 6 (a + b) else 0

def sum_T : ℕ :=
  (Finset.range 7).sum (λ a => (Finset.range (7 - a)).sum (λ b => T a b))

theorem sum_T_mod_1000 : sum_T % 1000 = 564 := by
  sorry

end sum_T_mod_1000_l2167_216750


namespace monotonic_subsequence_exists_l2167_216775

theorem monotonic_subsequence_exists (n : ℕ) (a : Fin ((2^n : ℕ) + 1) → ℕ)
  (h : ∀ k : Fin (2^n + 1), a k ≤ k.val) : 
  ∃ (b : Fin (n + 2) → Fin (2^n + 1)),
    (∀ i j : Fin (n + 2), i ≤ j → b i ≤ b j) ∧
    (∀ i j : Fin (n + 2), i < j → a (b i) ≤ a ( b j)) :=
by
  sorry

end monotonic_subsequence_exists_l2167_216775


namespace xy_cubed_identity_l2167_216758

namespace ProofProblem

theorem xy_cubed_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by
  sorry

end ProofProblem

end xy_cubed_identity_l2167_216758


namespace ethan_days_worked_per_week_l2167_216731

-- Define the conditions
def hourly_wage : ℕ := 18
def hours_per_day : ℕ := 8
def total_earnings : ℕ := 3600
def weeks_worked : ℕ := 5

-- Compute derived values
def daily_earnings : ℕ := hourly_wage * hours_per_day
def weekly_earnings : ℕ := total_earnings / weeks_worked

-- Define the proposition to be proved
theorem ethan_days_worked_per_week : ∃ d: ℕ, d * daily_earnings = weekly_earnings ∧ d = 5 :=
by
  use 5
  simp [daily_earnings, weekly_earnings]
  sorry

end ethan_days_worked_per_week_l2167_216731


namespace remainder_seven_pow_two_thousand_mod_thirteen_l2167_216759

theorem remainder_seven_pow_two_thousand_mod_thirteen :
  7^2000 % 13 = 1 := by
  sorry

end remainder_seven_pow_two_thousand_mod_thirteen_l2167_216759


namespace cubic_sum_of_roots_l2167_216765

theorem cubic_sum_of_roots :
  ∀ (r s : ℝ), (r + s = 5) → (r * s = 6) → (r^3 + s^3 = 35) :=
by
  intros r s h₁ h₂
  sorry

end cubic_sum_of_roots_l2167_216765


namespace betty_watermelons_l2167_216777

theorem betty_watermelons :
  ∃ b : ℕ, 
  (b + (b + 10) + (b + 20) + (b + 30) + (b + 40) = 200) ∧
  (b + 40 = 60) :=
by
  sorry

end betty_watermelons_l2167_216777


namespace grocery_store_more_expensive_l2167_216719

def bulk_price_per_can (total_price : ℚ) (num_cans : ℕ) : ℚ := total_price / num_cans

def grocery_price_per_can (total_price : ℚ) (num_cans : ℕ) : ℚ := total_price / num_cans

def price_difference_in_cents (price1 : ℚ) (price2 : ℚ) : ℚ := (price2 - price1) * 100

theorem grocery_store_more_expensive
  (bulk_total_price : ℚ)
  (bulk_cans : ℕ)
  (grocery_total_price : ℚ)
  (grocery_cans : ℕ)
  (difference_in_cents : ℚ) :
  bulk_total_price = 12.00 →
  bulk_cans = 48 →
  grocery_total_price = 6.00 →
  grocery_cans = 12 →
  difference_in_cents = 25 →
  price_difference_in_cents (bulk_price_per_can bulk_total_price bulk_cans) 
                            (grocery_price_per_can grocery_total_price grocery_cans) = difference_in_cents := by
  sorry

end grocery_store_more_expensive_l2167_216719


namespace universal_quantifiers_and_propositions_l2167_216772

-- Definitions based on conditions
def universal_quantifiers_phrases := ["for all", "for any"]
def universal_quantifier_symbol := "∀"
def universal_proposition := "Universal Proposition"
def universal_proposition_representation := "∀ x ∈ M, p(x)"

-- Main theorem
theorem universal_quantifiers_and_propositions :
  universal_quantifiers_phrases = ["for all", "for any"]
  ∧ universal_quantifier_symbol = "∀"
  ∧ universal_proposition = "Universal Proposition"
  ∧ universal_proposition_representation = "∀ x ∈ M, p(x)" :=
by
  sorry

end universal_quantifiers_and_propositions_l2167_216772


namespace sequence_increasing_l2167_216760

theorem sequence_increasing (a : ℕ → ℝ) 
  (h : ∀ n, a (n + 1) = a n + 3) : ∀ n, a (n + 1) > a n := 
by 
  sorry

end sequence_increasing_l2167_216760


namespace relationship_among_a_b_c_l2167_216752

noncomputable def a : ℝ := (Real.sqrt 2) / 2 * (Real.sin (17 * Real.pi / 180) + Real.cos (17 * Real.pi / 180))
noncomputable def b : ℝ := 2 * (Real.cos (13 * Real.pi / 180))^2 - 1
noncomputable def c : ℝ := (Real.sqrt 3) / 2

theorem relationship_among_a_b_c : c < a ∧ a < b :=
by
  sorry

end relationship_among_a_b_c_l2167_216752


namespace team_with_at_least_one_girl_l2167_216799

noncomputable def choose (n m : ℕ) := Nat.factorial n / (Nat.factorial m * Nat.factorial (n - m))

theorem team_with_at_least_one_girl (total_boys total_girls select : ℕ) (h_boys : total_boys = 5) (h_girls : total_girls = 5) (h_select : select = 3) :
  (choose (total_boys + total_girls) select) - (choose total_boys select) = 110 := 
by
  sorry

end team_with_at_least_one_girl_l2167_216799


namespace greatest_integer_le_x_squared_div_50_l2167_216796

-- Define the conditions as given in the problem
def trapezoid (b h : ℝ) (x : ℝ) : Prop :=
  let baseDifference := 50
  let longerBase := b + baseDifference
  let midline := (b + longerBase) / 2
  let heightRatioFactor := 2
  let xSquared := 6875
  let regionAreaRatio := 2 / 1 -- represented as 2
  (let areaRatio := (b + midline) / (b + baseDifference / 2)
   areaRatio = regionAreaRatio) ∧
  (x = Real.sqrt xSquared) ∧
  (b = 50)

-- Define the theorem that captures the question
theorem greatest_integer_le_x_squared_div_50 (b h x : ℝ) (h_trapezoid : trapezoid b h x) :
  ⌊ (x^2) / 50 ⌋ = 137 :=
by sorry

end greatest_integer_le_x_squared_div_50_l2167_216796


namespace right_triangle_area_l2167_216715

theorem right_triangle_area (a b : ℝ) (H₁ : a = 3) (H₂ : b = 5) : 
  1 / 2 * a * b = 7.5 := by
  rw [H₁, H₂]
  norm_num

end right_triangle_area_l2167_216715


namespace arithmetic_sequence_s10_l2167_216740

noncomputable def arithmetic_sequence_sum (n : ℕ) (a d : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_s10 (a : ℤ) (d : ℤ)
  (h1 : a + (a + 8 * d) = 18)
  (h4 : a + 3 * d = 7) :
  arithmetic_sequence_sum 10 a d = 100 :=
by sorry

end arithmetic_sequence_s10_l2167_216740


namespace parallelogram_side_length_l2167_216720

-- We need trigonometric functions and operations with real numbers.
open Real

theorem parallelogram_side_length (s : ℝ) 
  (h_side_lengths : s > 0 ∧ 3 * s > 0) 
  (h_angle : sin (30 / 180 * π) = 1 / 2) 
  (h_area : 3 * s * (s * sin (30 / 180 * π)) = 9 * sqrt 3) :
  s = 3 * sqrt 2 :=
by
  sorry

end parallelogram_side_length_l2167_216720


namespace pure_imaginary_number_l2167_216794

theorem pure_imaginary_number (a : ℝ) (ha : (1 + a) / (1 + a^2) = 0) : a = -1 :=
sorry

end pure_imaginary_number_l2167_216794


namespace product_sum_abcd_e_l2167_216753

-- Define the individual numbers
def a : ℕ := 12
def b : ℕ := 25
def c : ℕ := 52
def d : ℕ := 21
def e : ℕ := 32

-- Define the sum of the numbers a, b, c, and d
def sum_abcd : ℕ := a + b + c + d

-- Prove that multiplying the sum by e equals 3520
theorem product_sum_abcd_e : sum_abcd * e = 3520 := by
  sorry

end product_sum_abcd_e_l2167_216753


namespace eval_expr_at_values_l2167_216726

variable (x y : ℝ)

def expr := 2 * (3 * x^2 + x * y^2)- 3 * (2 * x * y^2 - x^2) - 10 * x^2

theorem eval_expr_at_values : x = -1 → y = 0.5 → expr x y = 0 :=
by
  intros hx hy
  rw [hx, hy]
  sorry

end eval_expr_at_values_l2167_216726


namespace box_cost_is_550_l2167_216783

noncomputable def cost_of_dryer_sheets (loads_per_week : ℕ) (sheets_per_load : ℕ)
                                        (sheets_per_box : ℕ) (annual_savings : ℝ) : ℝ :=
  let sheets_per_week := loads_per_week * sheets_per_load
  let sheets_per_year := sheets_per_week * 52
  let boxes_per_year := sheets_per_year / sheets_per_box
  annual_savings / boxes_per_year

theorem box_cost_is_550 (h1 : 4 = 4)
                        (h2 : 1 = 1)
                        (h3 : 104 = 104)
                        (h4 : 11 = 11) :
  cost_of_dryer_sheets 4 1 104 11 = 5.50 :=
by
  sorry

end box_cost_is_550_l2167_216783


namespace kittens_and_mice_count_l2167_216734

theorem kittens_and_mice_count :
  let children := 12
  let baskets_per_child := 3
  let cats_per_basket := 1
  let kittens_per_cat := 12
  let mice_per_kitten := 4
  let total_kittens := children * baskets_per_child * cats_per_basket * kittens_per_cat
  let total_mice := total_kittens * mice_per_kitten
  total_kittens + total_mice = 2160 :=
by
  sorry

end kittens_and_mice_count_l2167_216734


namespace multiplication_333_111_l2167_216748

theorem multiplication_333_111: 333 * 111 = 36963 := 
by 
sorry

end multiplication_333_111_l2167_216748


namespace expression_result_l2167_216751

theorem expression_result :
  ( (9 + (1 / 2)) + (7 + (1 / 6)) + (5 + (1 / 12)) + (3 + (1 / 20)) + (1 + (1 / 30)) ) * 12 = 310 := by
  sorry

end expression_result_l2167_216751


namespace sum_of_cubes_divisible_by_nine_l2167_216711

theorem sum_of_cubes_divisible_by_nine (n : ℕ) (h : 0 < n) : 9 ∣ (n^3 + (n + 1)^3 + (n + 2)^3) :=
by sorry

end sum_of_cubes_divisible_by_nine_l2167_216711


namespace spectators_count_l2167_216785

theorem spectators_count (total_wristbands : ℕ) (wristbands_per_person : ℕ) (h1 : total_wristbands = 250) (h2 : wristbands_per_person = 2) : (total_wristbands / wristbands_per_person = 125) :=
by
  sorry

end spectators_count_l2167_216785


namespace smaller_angle_is_85_l2167_216784

-- Conditions
def isParallelogram (α β : ℝ) : Prop :=
  α + β = 180

def angleExceedsBy10 (α β : ℝ) : Prop :=
  β = α + 10

-- Proof Problem
theorem smaller_angle_is_85 (α β : ℝ)
  (h1 : isParallelogram α β)
  (h2 : angleExceedsBy10 α β) :
  α = 85 :=
by
  sorry

end smaller_angle_is_85_l2167_216784


namespace max_cookies_andy_can_eat_l2167_216704

theorem max_cookies_andy_can_eat (A B C : ℕ) (hB_pos : B > 0) (hC_pos : C > 0) (hB : B ∣ A) (hC : C ∣ A) (h_sum : A + B + C = 36) :
  A ≤ 30 := by
  sorry

end max_cookies_andy_can_eat_l2167_216704


namespace triangle_hypotenuse_segments_l2167_216730

theorem triangle_hypotenuse_segments :
  ∀ (x : ℝ) (BC AC : ℝ),
  BC / AC = 3 / 7 →
  ∃ (h : ℝ) (BD AD : ℝ),
    h = 42 ∧
    BD * AD = h^2 ∧
    BD / AD = 9 / 49 ∧
    BD = 18 ∧
    AD = 98 :=
by
  sorry

end triangle_hypotenuse_segments_l2167_216730


namespace primes_between_4900_8100_l2167_216798

theorem primes_between_4900_8100 :
  ∃ (count : ℕ),
  count = 5 ∧ ∀ n : ℤ, 70 < n ∧ n < 90 ∧ (n * n > 4900 ∧ n * n < 8100 ∧ Prime n) → count = 5 :=
by
  sorry

end primes_between_4900_8100_l2167_216798


namespace feed_mixture_hay_calculation_l2167_216795

theorem feed_mixture_hay_calculation
  (hay_Stepan_percent oats_Pavel_percent corn_mixture_percent : ℝ)
  (hay_Stepan_mass_Stepan hay_Pavel_mass_Pavel total_mixture_mass : ℝ):
  hay_Stepan_percent = 0.4 ∧
  oats_Pavel_percent = 0.26 ∧
  (∃ (x : ℝ), 
  x > 0 ∧ 
  hay_Pavel_percent =  0.74 - x ∧ 
  0.15 * x + 0.25 * x = 0.3 * total_mixture_mass ∧
  hay_Stepan_mass_Stepan = 0.40 * 150 ∧
  hay_Pavel_mass_Pavel = (0.74 - x) * 250 ∧ 
  total_mixture_mass = 150 + 250) → 
  hay_Stepan_mass_Stepan + hay_Pavel_mass_Pavel = 170 := 
by
  intro h
  obtain ⟨h1, h2, ⟨x, hx1, hx2, hx3, hx4, hx5, hx6⟩⟩ := h
  /- proof -/
  sorry

end feed_mixture_hay_calculation_l2167_216795


namespace smaller_rectangle_perimeter_l2167_216709

def problem_conditions (a b : ℝ) : Prop :=
  2 * (a + b) = 96 ∧ 
  8 * b + 11 * a = 342 ∧
  a + b = 48 ∧ 
  (a * (b - 1) <= 0 ∧ b * (a - 1) <= 0 ∧ a > 0 ∧ b > 0)

theorem smaller_rectangle_perimeter (a b : ℝ) (hab : problem_conditions a b) :
  2 * (a / 12 + b / 9) = 9 :=
  sorry

end smaller_rectangle_perimeter_l2167_216709


namespace triangle_area_is_correct_l2167_216722

-- Defining the vertices of the triangle
def vertexA : ℝ × ℝ := (0, 0)
def vertexB : ℝ × ℝ := (0, 6)
def vertexC : ℝ × ℝ := (8, 10)

-- Define a function to calculate the area of a triangle given three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Statement to prove
theorem triangle_area_is_correct : triangle_area vertexA vertexB vertexC = 24.0 := 
by
  sorry

end triangle_area_is_correct_l2167_216722


namespace sum_of_roots_eq_neg_five_l2167_216712

theorem sum_of_roots_eq_neg_five (x₁ x₂ : ℝ) (h₁ : x₁^2 + 5 * x₁ - 2 = 0) (h₂ : x₂^2 + 5 * x₂ - 2 = 0) (h_distinct : x₁ ≠ x₂) :
  x₁ + x₂ = -5 := sorry

end sum_of_roots_eq_neg_five_l2167_216712


namespace solve_for_x_l2167_216729

theorem solve_for_x (x : ℚ) :  (1/2) * (12 * x + 3) = 3 * x + 2 → x = 1/6 := by
  intro h
  sorry

end solve_for_x_l2167_216729


namespace correct_choice_l2167_216741

theorem correct_choice (x y m : ℝ) (h1 : x > y) (h2 : m > 0) : x - y > 0 := by
  sorry

end correct_choice_l2167_216741


namespace correct_calculation_l2167_216738

theorem correct_calculation :
  (∀ a : ℝ, a^3 + a^2 ≠ a^5) ∧
  (∀ a : ℝ, a^3 / a^2 = a) ∧
  (∀ a : ℝ, 3 * a^3 * 2 * a^2 ≠ 6 * a^6) ∧
  (∀ a : ℝ, (a - 2)^2 ≠ a^2 - 4) :=
by
  sorry

end correct_calculation_l2167_216738


namespace negation_of_no_slow_learners_attend_school_l2167_216737

variable {α : Type}
variable (SlowLearner : α → Prop) (AttendsSchool : α → Prop)

-- The original statement
def original_statement : Prop := ∀ x, SlowLearner x → ¬ AttendsSchool x

-- The corresponding negation
def negation_statement : Prop := ∃ x, SlowLearner x ∧ AttendsSchool x

-- The proof problem statement
theorem negation_of_no_slow_learners_attend_school : 
  ¬ original_statement SlowLearner AttendsSchool ↔ negation_statement SlowLearner AttendsSchool := by
  sorry

end negation_of_no_slow_learners_attend_school_l2167_216737


namespace which_point_is_in_fourth_quadrant_l2167_216766

def point (x: ℝ) (y: ℝ) : Prop := x > 0 ∧ y < 0

theorem which_point_is_in_fourth_quadrant :
  point 5 (-4) :=
by {
  -- proofs for each condition can be added,
  sorry
}

end which_point_is_in_fourth_quadrant_l2167_216766


namespace eval_p_nested_l2167_216754

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then 2 * x + 3 * y
  else if x < 0 ∧ y < 0 then x ^ 2 - y
  else 4 * x + 2 * y

theorem eval_p_nested :
  p (p 2 (-3)) (p (-4) (-3)) = 61 :=
by
  sorry

end eval_p_nested_l2167_216754


namespace scientific_notation_of_1_300_000_l2167_216789

-- Define the condition: 1.3 million equals 1,300,000
def one_point_three_million : ℝ := 1300000

-- The theorem statement for the question
theorem scientific_notation_of_1_300_000 :
  one_point_three_million = 1.3 * 10^6 :=
sorry

end scientific_notation_of_1_300_000_l2167_216789


namespace Bella_bought_38_stamps_l2167_216749

def stamps (n t r : ℕ) : ℕ :=
  n + t + r

theorem Bella_bought_38_stamps :
  ∃ (n t r : ℕ),
    n = 11 ∧
    t = n + 9 ∧
    r = t - 13 ∧
    stamps n t r = 38 := 
  by
  sorry

end Bella_bought_38_stamps_l2167_216749


namespace factorization_of_polynomial_l2167_216716

theorem factorization_of_polynomial : ∀ x : ℝ, x^2 - x - 42 = (x + 6) * (x - 7) :=
by
  sorry

end factorization_of_polynomial_l2167_216716


namespace range_of_m_l2167_216763

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + m * x + 2 * m - 3 ≥ 0) ↔ 2 ≤ m ∧ m ≤ 6 := 
by
  sorry

end range_of_m_l2167_216763


namespace substitution_not_sufficient_for_identity_proof_l2167_216721

theorem substitution_not_sufficient_for_identity_proof {α : Type} (f g : α → α) :
  (∀ x : α, f x = g x) ↔ ¬ (∀ x, f x = g x ↔ (∃ (c : α), f c ≠ g c)) := by
  sorry

end substitution_not_sufficient_for_identity_proof_l2167_216721


namespace one_over_m_add_one_over_n_l2167_216781

theorem one_over_m_add_one_over_n (m n : ℕ) (h_sum : m + n = 80) (h_hcf : Nat.gcd m n = 6) (h_lcm : Nat.lcm m n = 210) : 
  1 / (m:ℚ) + 1 / (n:ℚ) = 1 / 15.75 :=
by
  sorry

end one_over_m_add_one_over_n_l2167_216781


namespace fractional_eq_range_m_l2167_216717

theorem fractional_eq_range_m (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) → m ≤ 2 ∧ m ≠ -2 :=
by
  sorry

end fractional_eq_range_m_l2167_216717


namespace range_of_independent_variable_l2167_216703

theorem range_of_independent_variable
  (x : ℝ) 
  (h1 : 2 - 3*x ≥ 0) 
  (h2 : x ≠ 0) 
  : x ≤ 2/3 ∧ x ≠ 0 :=
by 
  sorry

end range_of_independent_variable_l2167_216703


namespace part1_option1_payment_part1_option2_payment_part2_cost_effective_part3_more_cost_effective_l2167_216769

def cost_option1 (x : ℕ) : ℕ :=
  20 * x + 1200

def cost_option2 (x : ℕ) : ℕ :=
  18 * x + 1440

theorem part1_option1_payment (x : ℕ) (h : x > 20) : cost_option1 x = 20 * x + 1200 :=
  by sorry

theorem part1_option2_payment (x : ℕ) (h : x > 20) : cost_option2 x = 18 * x + 1440 :=
  by sorry

theorem part2_cost_effective (x : ℕ) (h : x = 30) : cost_option1 x < cost_option2 x :=
  by sorry

theorem part3_more_cost_effective (x : ℕ) (h : x = 30) : 20 * 80 + 20 * 10 * 9 / 10 = 1780 :=
  by sorry

end part1_option1_payment_part1_option2_payment_part2_cost_effective_part3_more_cost_effective_l2167_216769


namespace sufficient_but_not_necessary_condition_l2167_216724

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x * (x - 1) < 0 → x < 1) ∧ ¬(x < 1 → x * (x - 1) < 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l2167_216724


namespace age_ratio_l2167_216728

noncomputable def ratio_of_ages (A M : ℕ) : ℕ × ℕ :=
if A = 30 ∧ (A + 15 + (M + 15)) / 2 = 50 then
  (A / Nat.gcd A M, M / Nat.gcd A M)
else
  (0, 0)

theorem age_ratio :
  (45 + (40 + 15)) / 2 = 50 → 30 = 3 * 10 ∧ 40 = 4 * 10 →
  ratio_of_ages 30 40 = (3, 4) :=
by
  sorry

end age_ratio_l2167_216728


namespace priyas_speed_is_30_l2167_216767

noncomputable def find_priyas_speed (v : ℝ) : Prop :=
  let riya_speed := 20
  let time := 0.5  -- in hours
  let distance_apart := 25
  (riya_speed + v) * time = distance_apart

theorem priyas_speed_is_30 : ∃ v : ℝ, find_priyas_speed v ∧ v = 30 :=
by
  sorry

end priyas_speed_is_30_l2167_216767


namespace number_of_seats_in_nth_row_l2167_216790

theorem number_of_seats_in_nth_row (n : ℕ) :
    ∃ m : ℕ, m = 3 * n + 15 :=
by
  sorry

end number_of_seats_in_nth_row_l2167_216790


namespace find_x_values_l2167_216725

def f (x : ℝ) : ℝ := 3 * x^2 - 8

noncomputable def f_inv (y : ℝ) : ℝ := sorry  -- Placeholder for the inverse function

theorem find_x_values:
  ∃ x : ℝ, (f x = f_inv x) ↔ (x = (1 + Real.sqrt 97) / 6 ∨ x = (1 - Real.sqrt 97) / 6) := sorry

end find_x_values_l2167_216725


namespace two_pow_p_plus_three_pow_p_not_nth_power_l2167_216710

theorem two_pow_p_plus_three_pow_p_not_nth_power (p n : ℕ) (prime_p : Nat.Prime p) (one_lt_n : 1 < n) :
  ¬ ∃ k : ℕ, 2 ^ p + 3 ^ p = k ^ n :=
sorry

end two_pow_p_plus_three_pow_p_not_nth_power_l2167_216710


namespace min_value_expression_l2167_216793

open Real

theorem min_value_expression : ∀ x y : ℝ, x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 :=
by
  intro x y
  sorry

end min_value_expression_l2167_216793


namespace zoo_initial_animals_l2167_216707

theorem zoo_initial_animals (X : ℕ) :
  X - 6 + 1 + 3 + 8 + 16 = 90 → X = 68 :=
by
  intro h
  sorry

end zoo_initial_animals_l2167_216707


namespace square_octagon_can_cover_ground_l2167_216771

def square_interior_angle := 90
def octagon_interior_angle := 135

theorem square_octagon_can_cover_ground :
  square_interior_angle + 2 * octagon_interior_angle = 360 :=
by
  -- Proof skipped with sorry
  sorry

end square_octagon_can_cover_ground_l2167_216771


namespace work_days_together_l2167_216705

-- Conditions
variable {W : ℝ} (h_a_alone : ∀ (W : ℝ), W / a_work_time = W / 16)
variable {a_work_time : ℝ} (h_work_time_a : a_work_time = 16)

-- Question translated to proof problem
theorem work_days_together (D : ℝ) :
  (10 * (W / D) + 12 * (W / 16) = W) → D = 40 :=
by
  intros h
  have eq1 : 10 * (W / D) + 12 * (W / 16) = W := h
  sorry

end work_days_together_l2167_216705


namespace arithmetic_sequence_tenth_term_l2167_216778

noncomputable def sum_of_arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a1 + (n - 1) * d) / 2

def nth_term (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_tenth_term
  (a1 d : ℝ)
  (h1 : a1 + (a1 + d) + (a1 + 2 * d) = (a1 + 3 * d) + (a1 + 4 * d))
  (h2 : sum_of_arithmetic_sequence a1 d 5 = 60) :
  nth_term a1 d 10 = 26 :=
sorry

end arithmetic_sequence_tenth_term_l2167_216778


namespace james_final_weight_l2167_216706

noncomputable def initial_weight : ℝ := 120
noncomputable def muscle_gain : ℝ := 0.20 * initial_weight
noncomputable def fat_gain : ℝ := muscle_gain / 4
noncomputable def final_weight (initial_weight muscle_gain fat_gain : ℝ) : ℝ :=
  initial_weight + muscle_gain + fat_gain

theorem james_final_weight :
  final_weight initial_weight muscle_gain fat_gain = 150 :=
by
  sorry

end james_final_weight_l2167_216706


namespace quadratic_equal_real_roots_l2167_216791

theorem quadratic_equal_real_roots (a : ℝ) :
  (∃ x : ℝ, x^2 - a * x + 1 = 0 ∧ (x = a*x / 2)) ↔ a = 2 ∨ a = -2 :=
by sorry

end quadratic_equal_real_roots_l2167_216791


namespace base_edge_length_l2167_216756

theorem base_edge_length (x : ℕ) :
  (∃ (x : ℕ), 
    (∀ (sum_edges : ℕ), sum_edges = 6 * x + 48 → sum_edges = 120) →
    x = 12) := 
sorry

end base_edge_length_l2167_216756


namespace calc_30_exp_l2167_216755

theorem calc_30_exp :
  30 * 30 ^ 10 = 30 ^ 11 :=
by sorry

end calc_30_exp_l2167_216755


namespace combined_room_size_l2167_216792

theorem combined_room_size (M J S : ℝ) 
  (h1 : M + J + S = 800) 
  (h2 : J = M + 100) 
  (h3 : S = M - 50) : 
  J + S = 550 := 
by
  sorry

end combined_room_size_l2167_216792


namespace largest_value_x_l2167_216780

-- Definition of the conditions
def equation (x : ℚ) : Prop :=
  (16 * x^2 - 40 * x + 15) / (4 * x - 3) + 7 * x = 8 * x - 2

-- Statement of the proof 
theorem largest_value_x : ∀ x : ℚ, equation x → x ≤ 9 / 4 := sorry

end largest_value_x_l2167_216780


namespace fixed_point_coordinates_l2167_216773

noncomputable def fixed_point (A : Real × Real) : Prop :=
∀ (k : Real), ∃ (x y : Real), A = (x, y) ∧ (3 + k) * x + (1 - 2 * k) * y + 1 + 5 * k = 0

theorem fixed_point_coordinates :
  fixed_point (-1, 2) :=
by
  sorry

end fixed_point_coordinates_l2167_216773


namespace f_at_5_l2167_216742

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function definition

axiom odd_function (f: ℝ → ℝ) : ∀ x : ℝ, f (-x) = -f x
axiom functional_equation (f: ℝ → ℝ) : ∀ x : ℝ, f (x + 1) + f x = 0

theorem f_at_5 : f 5 = 0 :=
by {
  -- Proof to be provided here
  sorry
}

end f_at_5_l2167_216742


namespace min_quotient_l2167_216762

def digits_distinct (a b c : ℕ) : Prop := 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

def quotient (a b c : ℕ) : ℚ := 
  (100 * a + 10 * b + c : ℚ) / (a + b + c : ℚ)

theorem min_quotient (a b c : ℕ) (h1 : b > 3) (h2 : c ≠ b) (h3: digits_distinct a b c) : 
  quotient a b c ≥ 19.62 :=
sorry

end min_quotient_l2167_216762


namespace rays_total_grocery_bill_l2167_216702

-- Conditions
def hamburger_meat_cost : ℝ := 5.0
def crackers_cost : ℝ := 3.50
def frozen_veg_cost_per_bag : ℝ := 2.0
def frozen_veg_bags : ℕ := 4
def cheese_cost : ℝ := 3.50
def discount_rate : ℝ := 0.10

-- Total cost before discount
def total_cost_before_discount : ℝ :=
  hamburger_meat_cost + crackers_cost + (frozen_veg_cost_per_bag * frozen_veg_bags) + cheese_cost

-- Discount amount
def discount_amount : ℝ := discount_rate * total_cost_before_discount

-- Total cost after discount
def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount

-- Theorem: Ray's total grocery bill
theorem rays_total_grocery_bill : total_cost_after_discount = 18.0 :=
  by
    sorry

end rays_total_grocery_bill_l2167_216702


namespace simplify_fraction_expression_l2167_216718

variable (d : ℤ)

theorem simplify_fraction_expression : (6 + 4 * d) / 9 + 3 = (33 + 4 * d) / 9 := by
  sorry

end simplify_fraction_expression_l2167_216718


namespace inequality_a3_b3_c3_l2167_216736

theorem inequality_a3_b3_c3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 + b^3 + c^3 + 3 * a * b * c > a * b * (a + b) + b * c * (b + c) + a * c * (a + c) :=
by
  sorry

end inequality_a3_b3_c3_l2167_216736


namespace B_share_is_correct_l2167_216786

open Real

noncomputable def total_money : ℝ := 10800
noncomputable def ratio_A : ℝ := 0.5
noncomputable def ratio_B : ℝ := 1.5
noncomputable def ratio_C : ℝ := 2.25
noncomputable def ratio_D : ℝ := 3.5
noncomputable def ratio_E : ℝ := 4.25
noncomputable def total_ratio : ℝ := ratio_A + ratio_B + ratio_C + ratio_D + ratio_E
noncomputable def value_per_part : ℝ := total_money / total_ratio
noncomputable def B_share : ℝ := ratio_B * value_per_part

theorem B_share_is_correct : B_share = 1350 := by 
  sorry

end B_share_is_correct_l2167_216786


namespace inequality_and_equality_equality_condition_l2167_216779

theorem inequality_and_equality (a b : ℕ) (ha : a > 1) (hb : b > 2) : a^b + 1 ≥ b * (a + 1) :=
by sorry

theorem equality_condition (a b : ℕ) : a = 2 ∧ b = 3 → a^b + 1 = b * (a + 1) :=
by
  intro h
  cases h
  sorry

end inequality_and_equality_equality_condition_l2167_216779


namespace max_value_k_l2167_216733

theorem max_value_k (x y k : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < k)
(h4 : 4 = k^2 * (x^2 / y^2 + 2 + y^2 / x^2) + k^3 * (x / y + y / x)) : 
k ≤ 4 * (Real.sqrt 2) - 4 :=
by sorry

end max_value_k_l2167_216733


namespace complex_roots_sum_condition_l2167_216727

theorem complex_roots_sum_condition 
  (z1 z2 : ℂ) 
  (h1 : ∀ z, z ^ 2 + z + 1 = 0) 
  (h2 : z1 ^ 2 + z1 + 1 = 0)
  (h3 : z2 ^ 2 + z2 + 1 = 0) : 
  (z2 / (z1 + 1)) + (z1 / (z2 + 1)) = -2 := 
 sorry

end complex_roots_sum_condition_l2167_216727


namespace polynomial_root_sum_l2167_216774

theorem polynomial_root_sum :
  ∃ a b c : ℝ,
    (∀ x : ℝ, Polynomial.eval x (Polynomial.X ^ 3 - 10 * Polynomial.X ^ 2 + 16 * Polynomial.X - 2) = 0) →
    a + b + c = 10 → ab + ac + bc = 16 → abc = 2 →
    (a / (bc + 2) + b / (ac + 2) + c / (ab + 2) = 4) := sorry

end polynomial_root_sum_l2167_216774


namespace arithmetic_seq_common_difference_l2167_216701

theorem arithmetic_seq_common_difference (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 7 * a 11 = 6) (h2 : a 4 + a (14) = 5) : 
  d = 1 / 4 ∨ d = -1 / 4 :=
sorry

end arithmetic_seq_common_difference_l2167_216701


namespace animal_population_l2167_216787

def total_population (L P E : ℕ) : ℕ :=
L + P + E

theorem animal_population 
    (L P E : ℕ) 
    (h1 : L = 2 * P) 
    (h2 : E = (L + P) / 2) 
    (h3 : L = 200) : 
  total_population L P E = 450 := 
  by 
    sorry

end animal_population_l2167_216787


namespace distance_traveled_on_foot_l2167_216764

theorem distance_traveled_on_foot (x y : ℝ) : x + y = 61 ∧ (x / 4 + y / 9 = 9) → x = 16 :=
by {
  sorry
}

end distance_traveled_on_foot_l2167_216764


namespace shaded_area_l2167_216714

def radius (R : ℝ) : Prop := R > 0
def angle (α : ℝ) : Prop := α = 20 * (Real.pi / 180)

theorem shaded_area (R : ℝ) (hR : radius R) (hα : angle (20 * (Real.pi / 180))) :
  let S0 := Real.pi * R^2 / 2
  let sector_radius := 2 * R
  let sector_angle := 20 * (Real.pi / 180)
  (2 * sector_radius * sector_radius * sector_angle / 2) / sector_angle = 2 * Real.pi * R^2 / 9 :=
by
  sorry

end shaded_area_l2167_216714


namespace no_intersection_at_roots_l2167_216745

theorem no_intersection_at_roots {f g : ℝ → ℝ} (h : ∀ x, f x = x ∧ g x = x - 3) :
  ¬ (∃ x, (x = 0 ∨ x = 3) ∧ (f x = g x)) :=
by
  intros 
  sorry

end no_intersection_at_roots_l2167_216745


namespace surface_area_ratio_l2167_216732

theorem surface_area_ratio (x : ℝ) (hx : x > 0) :
  let SA1 := 6 * (4 * x) ^ 2
  let SA2 := 6 * x ^ 2
  (SA1 / SA2) = 16 := by
  sorry

end surface_area_ratio_l2167_216732
