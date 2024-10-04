import Mathlib

namespace probability_of_four_of_a_kind_is_correct_l211_211616

noncomputable def probability_four_of_a_kind: ℚ :=
  let total_ways := Nat.choose 52 5
  let successful_ways := 13 * 1 * 12 * 4
  (successful_ways: ℚ) / (total_ways: ℚ)

theorem probability_of_four_of_a_kind_is_correct :
  probability_four_of_a_kind = 13 / 54145 := 
by
  -- sorry is used because we are only writing the statement, no proof required
  sorry

end probability_of_four_of_a_kind_is_correct_l211_211616


namespace edmund_earning_l211_211808

-- Definitions based on the conditions
def daily_chores := 4
def days_in_week := 7
def weeks := 2
def normal_weekly_chores := 12
def pay_per_extra_chore := 2

-- Theorem to be proven
theorem edmund_earning :
  let total_chores := daily_chores * days_in_week * weeks
      normal_chores := normal_weekly_chores * weeks
      extra_chores := total_chores - normal_chores
      earnings := extra_chores * pay_per_extra_chore
  in earnings = 64 := 
by
  sorry

end edmund_earning_l211_211808


namespace map_distance_l211_211373

theorem map_distance (fifteen_cm_in_km : ℤ) (cm_to_km : ℕ): 
  fifteen_cm_in_km = 90 ∧ cm_to_km = 6 → 20 * cm_to_km = 120 := 
by 
  sorry

end map_distance_l211_211373


namespace complement_intersection_in_U_l211_211833

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4, 5}

theorem complement_intersection_in_U : (U \ (A ∩ B)) = {1, 4, 5, 6, 7, 8} :=
by {
  sorry
}

end complement_intersection_in_U_l211_211833


namespace map_length_represents_distance_l211_211416

theorem map_length_represents_distance :
  ∀ (s : ℝ), (15 * s = 90) → (20 * s = 120) :=
begin
  intros s h,
  sorry
end

end map_length_represents_distance_l211_211416


namespace diagonal_of_rectangle_l211_211046

noncomputable def L : ℝ := 40 * Real.sqrt 3
noncomputable def W : ℝ := 30 * Real.sqrt 3
noncomputable def d : ℝ := Real.sqrt (L^2 + W^2)

theorem diagonal_of_rectangle :
  d = 50 * Real.sqrt 3 :=
by sorry

end diagonal_of_rectangle_l211_211046


namespace islander_parity_l211_211806

-- Define the concept of knights and liars
def is_knight (x : ℕ) : Prop := x % 2 = 0 -- Knight count is even
def is_liar (x : ℕ) : Prop := ¬(x % 2 = 1) -- Liar count being odd is false, so even

-- Define the total inhabitants on the island and conditions
theorem islander_parity (K L : ℕ) (h₁ : is_knight K) (h₂ : is_liar L) (h₃ : K + L = 2021) : false := sorry

end islander_parity_l211_211806


namespace range_of_m_correct_l211_211510

noncomputable def range_of_m (x : ℝ) (m : ℝ) : Prop :=
  (x + m) / (x - 2) - (2 * m) / (x - 2) = 3 ∧ x > 0 ∧ x ≠ 2

theorem range_of_m_correct (m : ℝ) : 
  (∃ x : ℝ, range_of_m x m) ↔ m < 6 ∧ m ≠ 2 :=
sorry

end range_of_m_correct_l211_211510


namespace table_length_is_77_l211_211287

theorem table_length_is_77 :
  ∀ (x : ℕ), (∀ (sheets: ℕ), sheets = 72 → x = (5 + sheets)) → x = 77 :=
by {
  sorry
}

end table_length_is_77_l211_211287


namespace weight_of_7th_person_l211_211060

-- Defining the constants and conditions
def num_people_initial : ℕ := 6
def avg_weight_initial : ℝ := 152
def num_people_total : ℕ := 7
def avg_weight_total : ℝ := 151

-- Calculating the total weights from the given average weights
def total_weight_initial := num_people_initial * avg_weight_initial
def total_weight_total := num_people_total * avg_weight_total

-- Theorem stating the weight of the 7th person
theorem weight_of_7th_person : total_weight_total - total_weight_initial = 145 := 
sorry

end weight_of_7th_person_l211_211060


namespace find_m_for_one_solution_l211_211181

theorem find_m_for_one_solution (m : ℚ) :
  (∀ x : ℝ, 3*x^2 - 7*x + m = 0 → (∃! y : ℝ, 3*y^2 - 7*y + m = 0)) → m = 49/12 := by
  sorry

end find_m_for_one_solution_l211_211181


namespace part1_part2_l211_211528

def f (x a : ℝ) : ℝ := abs (x - a) + 2 * x

theorem part1 (x : ℝ) : f x (-1) ≤ 0 ↔ x ≤ -1/3 :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, x ≥ -1 → f x a ≥ 0) ↔ (a ≤ -3 ∨ a ≥ 1) :=
by
  sorry

end part1_part2_l211_211528


namespace map_length_scale_l211_211390

theorem map_length_scale (len1 len2 : ℕ) (dist1 dist2 : ℕ) (h1 : len1 = 15) (h2 : dist1 = 90) (h3 : len2 = 20) :
  dist2 = 120 :=
by
  sorry

end map_length_scale_l211_211390


namespace b_not_six_iff_neg_two_not_in_range_l211_211948

def g (x b : ℝ) := x^3 + x^2 + b*x + 2

theorem b_not_six_iff_neg_two_not_in_range (b : ℝ) : 
  (∀ x : ℝ, g x b ≠ -2) ↔ b ≠ 6 :=
by
  sorry

end b_not_six_iff_neg_two_not_in_range_l211_211948


namespace number_of_shirts_l211_211352

theorem number_of_shirts (ratio_pants_shirts: ℕ) (num_pants: ℕ) (S: ℕ) : 
  ratio_pants_shirts = 7 ∧ num_pants = 14 → S = 20 :=
by
  sorry

end number_of_shirts_l211_211352


namespace least_five_digit_congruent_to_six_mod_seventeen_l211_211463

theorem least_five_digit_congruent_to_six_mod_seventeen : ∃ x : ℕ, x ≥ 10000 ∧ x < 100000 ∧ x % 17 = 6 ∧ ∀ y : ℕ, y ≥ 10000 ∧ y < 100000 ∧ y % 17 = 6 → x ≤ y :=
  by
    sorry

end least_five_digit_congruent_to_six_mod_seventeen_l211_211463


namespace area_of_annulus_l211_211300

variables (R r x : ℝ) (hRr : R > r) (h : R^2 - r^2 = x^2)

theorem area_of_annulus : π * R^2 - π * r^2 = π * x^2 :=
by
  sorry

end area_of_annulus_l211_211300


namespace hyperbola_eccentricity_l211_211695

theorem hyperbola_eccentricity (a : ℝ) (h : a > 0) (h_asymptote : Real.tan (Real.pi / 6) = 1 / a) :
  let c := Real.sqrt (a^2 + 1)
  let e := c / a
  e = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end hyperbola_eccentricity_l211_211695


namespace solve_for_q_l211_211324

noncomputable def is_arithmetic_SUM_seq (a₁ q: ℝ) (n: ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem solve_for_q (a₁ q S3 S6 S9: ℝ) (hq: q ≠ 1) (hS3: S3 = is_arithmetic_SUM_seq a₁ q 3) 
(hS6: S6 = is_arithmetic_SUM_seq a₁ q 6) (hS9: S9 = is_arithmetic_SUM_seq a₁ q 9) 
(h_arith: 2 * S9 = S3 + S6) : q^3 = 3 / 2 :=
sorry

end solve_for_q_l211_211324


namespace jenny_eggs_in_each_basket_l211_211991

theorem jenny_eggs_in_each_basket (n : ℕ) (h1 : 30 % n = 0) (h2 : 45 % n = 0) (h3 : n ≥ 5) : n = 15 :=
sorry

end jenny_eggs_in_each_basket_l211_211991


namespace mul_3_6_0_5_l211_211666

theorem mul_3_6_0_5 : 3.6 * 0.5 = 1.8 :=
by
  sorry

end mul_3_6_0_5_l211_211666


namespace intersection_of_A_and_B_l211_211698

namespace ProofProblem

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {1, 2, 4, 5}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 5} := by
  sorry

end ProofProblem

end intersection_of_A_and_B_l211_211698


namespace sin_of_cos_of_angle_l211_211984

-- We need to assume that A is an angle of a triangle, hence A is in the range (0, π).
theorem sin_of_cos_of_angle (A : ℝ) (hA : 0 < A ∧ A < π) (h_cos : Real.cos A = -3/5) : Real.sin A = 4/5 := by
  sorry

end sin_of_cos_of_angle_l211_211984


namespace eccentricity_of_ellipse_l211_211671

theorem eccentricity_of_ellipse :
  let a : ℝ := 4
  let b : ℝ := 3
  let c : ℝ := Real.sqrt (a^2 - b^2)
  let e : ℝ := c / a
  e = Real.sqrt 7 / 4 :=
by
  sorry

end eccentricity_of_ellipse_l211_211671


namespace steve_speed_on_way_back_l211_211769

-- Let's define the variables and constants used in the problem.
def distance_to_work : ℝ := 30 -- in km
def total_time_on_road : ℝ := 6 -- in hours
def back_speed_ratio : ℝ := 2 -- Steve drives twice as fast on the way back

theorem steve_speed_on_way_back :
  ∃ v : ℝ, v > 0 ∧ (30 / v + 15 / v = 6) ∧ (2 * v = 15) := by
  sorry

end steve_speed_on_way_back_l211_211769


namespace increasing_function_range_l211_211520

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 - a) * x + 1 else a^x

theorem increasing_function_range (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) (h₃ : ∀ x y : ℝ, x < y → f a x < f a y) : 
  3 / 2 ≤ a ∧ a < 2 := by
  sorry

end increasing_function_range_l211_211520


namespace sum_of_two_numbers_l211_211581

theorem sum_of_two_numbers (x y : ℕ) (h1 : y = 2 * x - 3) (h2 : x = 14) : x + y = 39 :=
by
  sorry

end sum_of_two_numbers_l211_211581


namespace least_five_digit_congruent_to_six_mod_seventeen_l211_211465

theorem least_five_digit_congruent_to_six_mod_seventeen : ∃ x : ℕ, x ≥ 10000 ∧ x < 100000 ∧ x % 17 = 6 ∧ ∀ y : ℕ, y ≥ 10000 ∧ y < 100000 ∧ y % 17 = 6 → x ≤ y :=
  by
    sorry

end least_five_digit_congruent_to_six_mod_seventeen_l211_211465


namespace greatest_possible_m_exists_greatest_m_l211_211641

def reverse_digits (m : ℕ) : ℕ :=
  -- Function to reverse digits of a four-digit number
  sorry 

theorem greatest_possible_m (m : ℕ) : 
  (1000 ≤ m ∧ m < 10000) ∧
  (let n := reverse_digits m in 1000 ≤ n ∧ n < 10000) ∧
  (m % 63 = 0) ∧
  (m % 11 = 0) →
  m ≤ 9696 :=
by
  sorry

theorem exists_greatest_m (m : ℕ) : 
  (1000 ≤ m ∧ m < 10000) ∧
  (let n := reverse_digits m in 1000 ≤ n ∧ n < 10000) ∧
  (m % 63 = 0) ∧
  (m % 11 = 0) →
  ∃ m, m = 9696 :=
by
  sorry

end greatest_possible_m_exists_greatest_m_l211_211641


namespace city_mileage_per_tankful_l211_211280

theorem city_mileage_per_tankful :
  ∀ (T : ℝ), 
  ∃ (city_miles : ℝ),
    (462 = T * (32 + 12)) ∧
    (city_miles = 32 * T) ∧
    (city_miles = 336) :=
by
  sorry

end city_mileage_per_tankful_l211_211280


namespace problem_statement_l211_211330

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x^2 - 3) * Real.exp x

-- Conditions and conclusions
theorem problem_statement :
  (∃ x, is_local_max f x ∧ ∀ y, f y < f x) ∧
  (∀ b, (∀ x, f x = b → ∃! x (h : f x = b), (∃ x1 x2 x3, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3)) → 0 < b ∧ b < 6 * Real.exp (-3)) :=
by
  sorry

end problem_statement_l211_211330


namespace sum_Q_inv_eq_l211_211532

noncomputable def sum_Q_inv (n : ℕ) : ℝ :=
  let nums := {i : ℕ | ∃ k : ℕ, k < n ∧ i = 2^k }
  let all_perms := equiv.Perm.ofFinset (nums.toFinset)
  Finset.sum all_perms.toFinset (λ σ =>
    (Finset.range n).prod (λ k => ∑(i : ℕ) in (Finset.range (k+1)).map (σ.to_fun), (i : ℝ))⁻¹
  )

theorem sum_Q_inv_eq (n : ℕ) : sum_Q_inv n = 2^(-(n * (n-1)) / 2) :=
  sorry

end sum_Q_inv_eq_l211_211532


namespace problem1_extr_vals_l211_211276

-- Definitions from conditions
def f (x : ℝ) := x^3 - 2 * x^2 + x
def g (x : ℝ) := f x + 1

theorem problem1_extr_vals :
  ∃ a b : ℝ, a = g (1/3) ∧ b = g 1 ∧ a = 31/27 ∧ b = 1 :=
by
  sorry

end problem1_extr_vals_l211_211276


namespace pure_imaginary_number_implies_x_eq_1_l211_211480

theorem pure_imaginary_number_implies_x_eq_1 (x : ℝ)
  (h1 : x^2 - 1 = 0)
  (h2 : x + 1 ≠ 0) : x = 1 :=
sorry

end pure_imaginary_number_implies_x_eq_1_l211_211480


namespace quadratic_inequality_solution_set_l211_211440

theorem quadratic_inequality_solution_set (a b c : ℝ) (h₁ : a < 0) (h₂ : b^2 - 4 * a * c < 0) :
  ∀ x : ℝ, a * x^2 + b * x + c < 0 :=
sorry

end quadratic_inequality_solution_set_l211_211440


namespace six_digit_number_consecutive_evens_l211_211940

theorem six_digit_number_consecutive_evens :
  ∃ n : ℕ,
    287232 = (2 * n - 2) * (2 * n) * (2 * n + 2) ∧
    287232 / 100000 = 2 ∧
    287232 % 10 = 2 :=
by
  sorry

end six_digit_number_consecutive_evens_l211_211940


namespace martha_blue_butterflies_l211_211011

-- Definitions based on conditions
variables (total_butterflies : ℕ) (black_butterflies : ℕ)
variables (yellow_butterflies : ℕ) (blue_butterflies : ℕ)

def total_is_11 : Prop := total_butterflies = 11
def black_is_5 : Prop := black_butterflies = 5
def blue_is_twice_yellow : Prop := blue_butterflies = 2 * yellow_butterflies
def remaining_is_blue_and_yellow : Prop := total_butterflies - black_butterflies = blue_butterflies + yellow_butterflies

-- The statement we want to prove
theorem martha_blue_butterflies (h1 : total_is_11) (h2 : black_is_5) 
    (h3 : blue_is_twice_yellow) (h4 : remaining_is_blue_and_yellow) :
    blue_butterflies = 4 := 
begin
  sorry
end

end martha_blue_butterflies_l211_211011


namespace coordinates_of_D_l211_211320
-- Importing the necessary library

-- Defining the conditions as given in the problem
def AB : ℝ × ℝ := (5, 3)
def C : ℝ × ℝ := (-1, 3)
def CD : ℝ × ℝ := (2 * 5, 2 * 3)

-- The target proof statement
theorem coordinates_of_D :
  ∃ D : ℝ × ℝ, CD = D - C ∧ D = (9, -3) :=
by
  sorry

end coordinates_of_D_l211_211320


namespace millet_percentage_in_mix_l211_211905

def contribution_millet_brandA (percA mixA : ℝ) := percA * mixA
def contribution_millet_brandB (percB mixB : ℝ) := percB * mixB

theorem millet_percentage_in_mix
  (percA : ℝ) (percB : ℝ) (mixA : ℝ) (mixB : ℝ)
  (h1 : percA = 0.40) (h2 : percB = 0.65) (h3 : mixA = 0.60) (h4 : mixB = 0.40) :
  (contribution_millet_brandA percA mixA + contribution_millet_brandB percB mixB = 0.50) :=
by
  sorry

end millet_percentage_in_mix_l211_211905


namespace number_of_unit_squares_in_50th_ring_l211_211658

def nth_ring_unit_squares (n : ℕ) : ℕ :=
  8 * n

-- Statement to prove
theorem number_of_unit_squares_in_50th_ring : nth_ring_unit_squares 50 = 400 :=
by
  -- Proof steps (skip with sorry)
  sorry

end number_of_unit_squares_in_50th_ring_l211_211658


namespace value_of_8b_l211_211518

theorem value_of_8b (a b : ℝ) (h1 : 6 * a + 3 * b = 3) (h2 : b = 2 * a - 3) : 8 * b = -8 := by
  sorry

end value_of_8b_l211_211518


namespace value_of_a_for_perfect_square_trinomial_l211_211545

theorem value_of_a_for_perfect_square_trinomial (a : ℝ) : (∃ b : ℝ, (x : ℝ) → x^2 + 2 * a * x + 9 = (x + b)^2) → (a = 3 ∨ a = -3) :=
by
  sorry

end value_of_a_for_perfect_square_trinomial_l211_211545


namespace roots_quadratic_reciprocal_l211_211017

theorem roots_quadratic_reciprocal (x1 x2 : ℝ) (h1 : x1 + x2 = -8) (h2 : x1 * x2 = 4) :
  (1 / x1) + (1 / x2) = -2 :=
sorry

end roots_quadratic_reciprocal_l211_211017


namespace acute_angle_between_CD_AF_l211_211583

noncomputable def midpoint (A B C : Type*) [Euclidean_space V P] (B : V) := midpoint ℝ A C = B

noncomputable def square (A B D E : Type*) [Euclidean_space V P] := 
  IsSquare (convex_hull ℝ {A, B, D, E})

noncomputable def equilateral_triangle (B C F : Type*) [Euclidean_space V P] := 
  IsEquilateralTriangle (convex_hull ℝ {B, C, F})

theorem acute_angle_between_CD_AF
  (A B C D E F : Type*) [Euclidean_space V P] 
  (h1 : midpoint A B C)
  (h2 : square A B D E)
  (h3 : equilateral_triangle B C F) : 
  acute_angle_between_lines D C A F = 75 :=
sorry

end acute_angle_between_CD_AF_l211_211583


namespace fourth_hexagon_dots_l211_211612

def dots_in_hexagon (n : ℕ) : ℕ :=
  if n = 0 then 1
  else 1 + (12 * (n * (n + 1) / 2))

theorem fourth_hexagon_dots : dots_in_hexagon 4 = 85 :=
by
  unfold dots_in_hexagon
  norm_num
  sorry

end fourth_hexagon_dots_l211_211612


namespace no_n_for_equal_sums_l211_211163

theorem no_n_for_equal_sums (n : ℕ) (h : n ≠ 0) :
  let s1 := (3 * n^2 + 7 * n) / 2
  let s2 := (3 * n^2 + 37 * n) / 2
  s1 ≠ s2 :=
by
  let s1 := (3 * n^2 + 7 * n) / 2
  let s2 := (3 * n^2 + 37 * n) / 2
  sorry

end no_n_for_equal_sums_l211_211163


namespace range_alpha_div_three_l211_211688

open Real

theorem range_alpha_div_three (α : ℝ) (k : ℤ) :
  sin α > 0 → cos α < 0 → sin (α / 3) > cos (α / 3) →
  ∃ k : ℤ,
    (2 * k * π + π / 4 < α / 3 ∧ α / 3 < 2 * k * π + π / 3) ∨
    (2 * k * π + 5 * π / 6 < α / 3 ∧ α / 3 < 2 * k * π + π) :=
by
  intros
  sorry

end range_alpha_div_three_l211_211688


namespace two_digit_numbers_div_quotient_remainder_l211_211925

theorem two_digit_numbers_div_quotient_remainder (x y : ℕ) (N : ℕ) (h1 : N = 10 * x + y) (h2 : N = 7 * (x + y) + 6) (hx_range : 1 ≤ x ∧ x ≤ 9) (hy_range : 0 ≤ y ∧ y ≤ 9) :
  N = 62 ∨ N = 83 := sorry

end two_digit_numbers_div_quotient_remainder_l211_211925


namespace count_ordered_triples_l211_211340

theorem count_ordered_triples (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a^2 = b^2 + c^2) (h5 : b^2 = a^2 + c^2) (h6 : c^2 = a^2 + b^2) : 
  (a = b ∧ b = c ∧ a ≠ 0) ∨ (a = -b ∧ b = c ∧ a ≠ 0) ∨ (a = b ∧ b = -c ∧ a ≠ 0) ∨ (a = -b ∧ b = -c ∧ a ≠ 0) :=
sorry

end count_ordered_triples_l211_211340


namespace other_train_length_l211_211900

noncomputable def length_of_other_train
  (l1 : ℝ) (v1_kmph : ℝ) (v2_kmph : ℝ) (t : ℝ) : ℝ :=
  let v1 := (v1_kmph * 1000) / 3600
  let v2 := (v2_kmph * 1000) / 3600
  let relative_speed := v1 + v2
  let total_distance := relative_speed * t
  total_distance - l1

theorem other_train_length
  (l1 : ℝ) (v1_kmph : ℝ) (v2_kmph : ℝ) (t : ℝ)
  (hl1 : l1 = 230)
  (hv1 : v1_kmph = 120)
  (hv2 : v2_kmph = 80)
  (ht : t = 9) :
  length_of_other_train l1 v1_kmph v2_kmph t = 269.95 :=
by
  rw [hl1, hv1, hv2, ht]
  -- Proof steps skipped
  sorry

end other_train_length_l211_211900


namespace scatter_plot_convention_l211_211763

def explanatory_variable := "x-axis"
def predictor_variable := "y-axis"

theorem scatter_plot_convention :
  explanatory_variable = "x-axis" ∧ predictor_variable = "y-axis" :=
by sorry

end scatter_plot_convention_l211_211763


namespace train_speed_is_60_kmph_l211_211071

-- Define the conditions
def length_of_train : ℕ := 300 -- in meters
def time_to_cross_pole : ℕ := 18 -- in seconds

-- Define the conversions
def meters_to_kilometers (m : ℕ) : ℝ := m / 1000.0
def seconds_to_hours (s : ℕ) : ℝ := s / 3600.0

-- Define the speed calculation
def speed_km_per_hr (distance_km : ℝ) (time_hr : ℝ) : ℝ := distance_km / time_hr

-- Prove that the speed of the train is 60 km/hr
theorem train_speed_is_60_kmph :
  speed_km_per_hr (meters_to_kilometers length_of_train) (seconds_to_hours time_to_cross_pole) = 60 := 
  by
    sorry

end train_speed_is_60_kmph_l211_211071


namespace a_plus_c_eq_neg800_l211_211854

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
noncomputable def g (x : ℝ) (c d : ℝ) : ℝ := x^2 + c * x + d

theorem a_plus_c_eq_neg800 (a b c d : ℝ) (h1 : g (-a / 2) c d = 0)
  (h2 : f (-c / 2) a b = 0) (h3 : ∀ x, f x a b ≥ f (-a / 2) a b)
  (h4 : ∀ x, g x c d ≥ g (-c / 2) c d) (h5 : f (-a / 2) a b = g (-c / 2) c d)
  (h6 : f 200 a b = -200) (h7 : g 200 c d = -200) :
  a + c = -800 := sorry

end a_plus_c_eq_neg800_l211_211854


namespace ivan_petrovich_lessons_daily_l211_211061

def donations_per_month (L k : ℕ) : ℕ := 21 * (k / 3) * 1000

theorem ivan_petrovich_lessons_daily (L k : ℕ) (h1 : 24 = 8 + L + 2 * L + k) (h2 : k = 16 - 3 * L)
    (income_from_lessons : 21 * (3 * L) * 1000)
    (rent_income : 14000)
    (monthly_expenses : 70000)
    (charity_donations : donations_per_month L k) :
  L = 2 ∧ charity_donations = 70000 := 
begin
  sorry
end

end ivan_petrovich_lessons_daily_l211_211061


namespace map_length_scale_l211_211388

theorem map_length_scale (len1 len2 : ℕ) (dist1 dist2 : ℕ) (h1 : len1 = 15) (h2 : dist1 = 90) (h3 : len2 = 20) :
  dist2 = 120 :=
by
  sorry

end map_length_scale_l211_211388


namespace median_of_consecutive_integers_l211_211607

theorem median_of_consecutive_integers (n : ℕ) (S : ℤ) (h1 : n = 35) (h2 : S = 1225) : 
  n % 2 = 1 → S / n = 35 := 
sorry

end median_of_consecutive_integers_l211_211607


namespace calculate_selling_price_l211_211283

noncomputable def originalPrice : ℝ := 120
noncomputable def firstDiscountRate : ℝ := 0.30
noncomputable def secondDiscountRate : ℝ := 0.15
noncomputable def taxRate : ℝ := 0.08

def discountedPrice1 (originalPrice firstDiscountRate : ℝ) : ℝ :=
  originalPrice * (1 - firstDiscountRate)

def discountedPrice2 (discountedPrice1 secondDiscountRate : ℝ) : ℝ :=
  discountedPrice1 * (1 - secondDiscountRate)

def finalPrice (discountedPrice2 taxRate : ℝ) : ℝ :=
  discountedPrice2 * (1 + taxRate)

theorem calculate_selling_price : 
  finalPrice (discountedPrice2 (discountedPrice1 originalPrice firstDiscountRate) secondDiscountRate) taxRate = 77.112 := 
sorry

end calculate_selling_price_l211_211283


namespace value_of_m_l211_211109

variable (a m : ℝ)
variable (h1 : a > 0)
variable (h2 : -a*m^2 + 2*a*m + 3 = 3)
variable (h3 : m ≠ 0)

theorem value_of_m : m = 2 :=
by
  sorry

end value_of_m_l211_211109


namespace greatest_prime_factor_341_l211_211231

theorem greatest_prime_factor_341 : ∃ (p : ℕ), prime p ∧ p ∣ 341 ∧ ∀ q, prime q ∧ q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_341_l211_211231


namespace smallest_12_digit_proof_l211_211316

def is_12_digit_number (n : ℕ) : Prop :=
  n >= 10^11 ∧ n < 10^12

def contains_each_digit_0_to_9 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] → d ∈ n.digits 10

def is_divisible_by_36 (n : ℕ) : Prop :=
  n % 36 = 0

noncomputable def smallest_12_digit_divisible_by_36_and_contains_each_digit : ℕ :=
  100023457896

theorem smallest_12_digit_proof :
  is_12_digit_number smallest_12_digit_divisible_by_36_and_contains_each_digit ∧
  contains_each_digit_0_to_9 smallest_12_digit_divisible_by_36_and_contains_each_digit ∧
  is_divisible_by_36 smallest_12_digit_divisible_by_36_and_contains_each_digit ∧
  ∀ m : ℕ, is_12_digit_number m ∧ contains_each_digit_0_to_9 m ∧ is_divisible_by_36 m →
  m >= smallest_12_digit_divisible_by_36_and_contains_each_digit :=
by
  sorry

end smallest_12_digit_proof_l211_211316


namespace range_of_x_sq_add_y_sq_l211_211118

theorem range_of_x_sq_add_y_sq (x y : ℝ) (h : x^2 + y^2 = 4 * x) : 
  ∃ (a b : ℝ), a ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ b ∧ a = 0 ∧ b = 16 :=
by
  sorry

end range_of_x_sq_add_y_sq_l211_211118


namespace add_base6_numbers_l211_211295

def base6_to_base10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6^1 + c * 6^0

def base10_to_base6 (n : ℕ) : (ℕ × ℕ × ℕ) := 
  (n / 6^2, (n % 6^2) / 6^1, (n % 6^2) % 6^1)

theorem add_base6_numbers : 
  let n1 := 3 * 6^1 + 5 * 6^0
  let n2 := 2 * 6^1 + 5 * 6^0
  let sum := n1 + n2
  base10_to_base6 sum = (1, 0, 4) :=
by
  -- Proof steps would go here
  sorry

end add_base6_numbers_l211_211295


namespace clark_discount_l211_211081

theorem clark_discount (price_per_part : ℕ) (number_of_parts : ℕ) (amount_paid : ℕ)
  (h1 : price_per_part = 80)
  (h2 : number_of_parts = 7)
  (h3 : amount_paid = 439) : 
  (number_of_parts * price_per_part) - amount_paid = 121 := by
  sorry

end clark_discount_l211_211081


namespace unit_digit_product_l211_211770

-- Definition of unit digit function
def unit_digit (n : Nat) : Nat := n % 10

-- Conditions about unit digits of given powers
lemma unit_digit_3_pow_68 : unit_digit (3 ^ 68) = 1 := by sorry
lemma unit_digit_6_pow_59 : unit_digit (6 ^ 59) = 6 := by sorry
lemma unit_digit_7_pow_71 : unit_digit (7 ^ 71) = 3 := by sorry

-- Main statement
theorem unit_digit_product : unit_digit (3 ^ 68 * 6 ^ 59 * 7 ^ 71) = 8 := by
  have h3 := unit_digit_3_pow_68
  have h6 := unit_digit_6_pow_59
  have h7 := unit_digit_7_pow_71
  sorry

end unit_digit_product_l211_211770


namespace initial_number_of_trees_l211_211444

theorem initial_number_of_trees (trees_removed remaining_trees initial_trees : ℕ) 
  (h1 : trees_removed = 4) 
  (h2 : remaining_trees = 2) 
  (h3 : remaining_trees + trees_removed = initial_trees) : 
  initial_trees = 6 :=
by
  sorry

end initial_number_of_trees_l211_211444


namespace son_present_age_l211_211765

variable (S F : ℕ)

-- Define the conditions
def fatherAgeCondition := F = S + 35
def twoYearsCondition := F + 2 = 2 * (S + 2)

-- The proof theorem
theorem son_present_age : 
  fatherAgeCondition S F → 
  twoYearsCondition S F → 
  S = 33 :=
by
  intros h1 h2
  sorry

end son_present_age_l211_211765


namespace cone_base_circumference_l211_211290

theorem cone_base_circumference (r : ℝ) (θ : ℝ) (circ_res : ℝ) :
  r = 4 → θ = 270 → circ_res = 6 * Real.pi :=
by 
  sorry

end cone_base_circumference_l211_211290


namespace largest_four_digit_negative_congruent_3_mod_29_l211_211456

theorem largest_four_digit_negative_congruent_3_mod_29 : 
  ∃ (n : ℤ), n < 0 ∧ n ≥ -9999 ∧ (n % 29 = 3) ∧ n = -1012 :=
sorry

end largest_four_digit_negative_congruent_3_mod_29_l211_211456


namespace baseball_games_in_season_l211_211756

def games_per_month : ℕ := 7
def months_in_season : ℕ := 2
def total_games_in_season : ℕ := games_per_month * months_in_season

theorem baseball_games_in_season : total_games_in_season = 14 := by
  sorry

end baseball_games_in_season_l211_211756


namespace arithmetic_sequence_geometric_sequence_l211_211157

-- Arithmetic sequence proof problem
theorem arithmetic_sequence (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, n ≥ 2 → a n - a (n - 1) = 2) :
  ∀ n, a n = 2 * n - 1 :=
by 
  sorry

-- Geometric sequence proof problem
theorem geometric_sequence (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, n ≥ 2 → a n / a (n - 1) = 2) :
  ∀ n, a n = 2 ^ (n - 1) :=
by 
  sorry

end arithmetic_sequence_geometric_sequence_l211_211157


namespace percentage_discount_l211_211582

def cost_per_ball : ℝ := 0.1
def number_of_balls : ℕ := 10000
def amount_paid : ℝ := 700

theorem percentage_discount : (number_of_balls * cost_per_ball - amount_paid) / (number_of_balls * cost_per_ball) * 100 = 30 :=
by
  sorry

end percentage_discount_l211_211582


namespace least_five_digit_congruent_6_mod_17_l211_211457

theorem least_five_digit_congruent_6_mod_17 : ∃ n: ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 6 ∧ ∀ m: ℕ, 10000 ≤ m ∧ m < 100000 ∧ m % 17 = 6 → n ≤ m :=
sorry

end least_five_digit_congruent_6_mod_17_l211_211457


namespace sum_of_integers_equals_75_l211_211027

theorem sum_of_integers_equals_75 
  (n m : ℤ) 
  (h1 : n * (n + 1) * (n + 2) = 924) 
  (h2 : m * (m + 1) * (m + 2) * (m + 3) = 924) 
  (sum_seven_integers : ℤ := n + (n + 1) + (n + 2) + m + (m + 1) + (m + 2) + (m + 3)) :
  sum_seven_integers = 75 := 
  sorry

end sum_of_integers_equals_75_l211_211027


namespace ribbon_initial_amount_l211_211355

theorem ribbon_initial_amount (x : ℕ) (gift_count : ℕ) (ribbon_per_gift : ℕ) (ribbon_left : ℕ)
  (H1 : ribbon_per_gift = 2) (H2 : gift_count = 6) (H3 : ribbon_left = 6)
  (H4 : x = gift_count * ribbon_per_gift + ribbon_left) : x = 18 :=
by
  rw [H1, H2, H3] at H4
  exact H4

end ribbon_initial_amount_l211_211355


namespace range_of_a_l211_211876

noncomputable def f (a x : ℝ) : ℝ :=
  x^2 + (a^2 + 1) * x + a - 2

theorem range_of_a (a : ℝ) :
  (f a 1 < 0) ∧ (f a (-1) < 0) → -1 < a ∧ a < 0 :=
by
  sorry

end range_of_a_l211_211876


namespace problem1_problem2_problem3_l211_211100

-- Statement for the first problem
theorem problem1 (x : ℚ) : 
  16 * (6 * x - 1) * (2 * x - 1) * (3 * x + 1) * (x - 1) + 25 = (24 * x^2 - 16 * x - 3)^2 :=
by sorry

-- Statement for the second problem
theorem problem2 (x : ℚ) : 
  (6 * x - 1) * (2 * x - 1) * (3 * x - 1) * (x - 1) + x^2 = (6 * x^2 - 6 * x + 1)^2 :=
by sorry

-- Statement for the third problem
theorem problem3 (x : ℚ) : 
  (6 * x - 1) * (4 * x - 1) * (3 * x - 1) * (x - 1) + 9 * x^4 = (9 * x^2 - 7 * x + 1)^2 :=
by sorry

end problem1_problem2_problem3_l211_211100


namespace function_is_convex_l211_211337

variable (a k : ℝ) (ha : 0 < a) (hk : 0 < k)

def f (x : ℝ) := 1 / x ^ k + a

theorem function_is_convex (x : ℝ) (hx : 0 < x) : ConvexOn ℝ Set.Ioi { x : ℝ | 0 < x } f := 
sorry

end function_is_convex_l211_211337


namespace dragon_2023_first_reappearance_l211_211874

theorem dragon_2023_first_reappearance :
  let cycle_letters := 6
  let cycle_digits := 4
  Nat.lcm cycle_letters cycle_digits = 12 :=
by
  rfl -- since LCM of 6 and 4 directly calculates to 12

end dragon_2023_first_reappearance_l211_211874


namespace sequence_periodicity_l211_211516

theorem sequence_periodicity (a : ℕ → ℤ) 
  (h1 : a 1 = 3) 
  (h2 : a 2 = 6) 
  (h_rec : ∀ n, a (n + 2) = a (n + 1) - a n): 
  a 2015 = -6 := 
sorry

end sequence_periodicity_l211_211516


namespace b_2030_is_5_l211_211998

def seq (b : ℕ → ℚ) : Prop :=
  b 1 = 4 ∧ b 2 = 5 ∧ ∀ n ≥ 3, b (n + 1) = b n / b (n - 1)

theorem b_2030_is_5 (b : ℕ → ℚ) (h : seq b) : 
  b 2030 = 5 :=
sorry

end b_2030_is_5_l211_211998


namespace percentage_greater_than_88_l211_211147

theorem percentage_greater_than_88 (x : ℝ) (percentage : ℝ) (h1 : x = 110) (h2 : x = 88 + (percentage * 88)) : percentage = 0.25 :=
by
  sorry

end percentage_greater_than_88_l211_211147


namespace greatest_prime_factor_341_l211_211242

theorem greatest_prime_factor_341 : ∃ p : ℕ, prime p ∧ p.factor 341 ∧ (∀ q : ℕ, prime q ∧ q.factor 341 → q ≤ p) ∧ p = 17 :=
by sorry

end greatest_prime_factor_341_l211_211242


namespace combined_error_percentage_l211_211301

theorem combined_error_percentage 
  (S : ℝ) 
  (error_side : ℝ) 
  (error_area : ℝ) 
  (h1 : error_side = 0.20) 
  (h2 : error_area = 0.04) :
  (1.04 * ((1 + error_side) * S) ^ 2 - S ^ 2) / S ^ 2 * 100 = 49.76 := 
by
  sorry

end combined_error_percentage_l211_211301


namespace fraction_identity_l211_211709

theorem fraction_identity (x y : ℚ) (h : x / y = 5 / 3) : y / (x - y) = 3 / 2 :=
by { sorry }

end fraction_identity_l211_211709


namespace bill_earnings_l211_211794

theorem bill_earnings
  (milk_total : ℕ)
  (fraction : ℚ)
  (milk_to_butter_ratio : ℕ)
  (milk_to_sour_cream_ratio : ℕ)
  (butter_price_per_gallon : ℚ)
  (sour_cream_price_per_gallon : ℚ)
  (whole_milk_price_per_gallon : ℚ)
  (milk_for_butter : ℚ)
  (milk_for_sour_cream : ℚ)
  (remaining_milk : ℚ)
  (total_earnings : ℚ) :
  milk_total = 16 →
  fraction = 1/4 →
  milk_to_butter_ratio = 4 →
  milk_to_sour_cream_ratio = 2 →
  butter_price_per_gallon = 5 →
  sour_cream_price_per_gallon = 6 →
  whole_milk_price_per_gallon = 3 →
  milk_for_butter = milk_total * fraction / milk_to_butter_ratio →
  milk_for_sour_cream = milk_total * fraction / milk_to_sour_cream_ratio →
  remaining_milk = milk_total - 2 * (milk_total * fraction) →
  total_earnings = (remaining_milk * whole_milk_price_per_gallon) + 
                   (milk_for_sour_cream * sour_cream_price_per_gallon) + 
                   (milk_for_butter * butter_price_per_gallon) →
  total_earnings = 41 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end bill_earnings_l211_211794


namespace Raven_age_l211_211713

-- Define the conditions
def Phoebe_age_current : Nat := 10
def Phoebe_age_in_5_years : Nat := Phoebe_age_current + 5

-- Define the hypothesis that in 5 years Raven will be 4 times as old as Phoebe
def Raven_in_5_years (R : Nat) : Prop := R + 5 = 4 * Phoebe_age_in_5_years

-- State the theorem to be proved
theorem Raven_age : ∃ R : Nat, Raven_in_5_years R ∧ R = 55 :=
by
  sorry

end Raven_age_l211_211713


namespace convex_f_l211_211336

variable {α : Type*}
variable [LinearOrder α] [OrderedAddCommGroup α] [Module ℝ α] [OrderedSMul ℝ α] [OrderedAddCommMonoid α]

noncomputable def f (x : ℝ) (a k : ℝ) : ℝ := (1 / x^k) + a

theorem convex_f (a k : ℝ) (ha : 0 < a) (hk : 0 < k) :
  ConvexOn ℝ (Ioi 0) (λ x => (1 / x^k) + a) :=
  sorry

end convex_f_l211_211336


namespace lattice_point_count_l211_211846

theorem lattice_point_count :
  (∃ (S : Finset (ℤ × ℤ)), S.card = 16 ∧ ∀ (p : ℤ × ℤ), p ∈ S → (|p.1| - 1) ^ 2 + (|p.2| - 1) ^ 2 < 2) :=
sorry

end lattice_point_count_l211_211846


namespace gcd_153_119_l211_211455

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  have h1 : 153 = 119 * 1 + 34 := by rfl
  have h2 : 119 = 34 * 3 + 17 := by rfl
  have h3 : 34 = 17 * 2 := by rfl
  sorry

end gcd_153_119_l211_211455


namespace area_of_triangle_ABF_l211_211956

theorem area_of_triangle_ABF :
  let C : Set (ℝ × ℝ) := {p | (p.1 ^ 2 / 4) + (p.2 ^ 2 / 3) = 1}
  let line : Set (ℝ × ℝ) := {p | p.1 - p.2 - 1 = 0}
  let F : ℝ × ℝ := (-1, 0)
  let AB := C ∩ line
  ∃ A B : ℝ × ℝ, A ∈ AB ∧ B ∈ AB ∧ A ≠ B ∧ 
  (1/2) * (2 : ℝ) * (12 * Real.sqrt (2 : ℝ) / 7) = (12 * Real.sqrt (2 : ℝ) / 7) :=
sorry

end area_of_triangle_ABF_l211_211956


namespace foot_slide_distance_l211_211634

def ladder_foot_slide (l h_initial h_new x_initial d y: ℝ) : Prop :=
  l = 30 ∧ x_initial = 6 ∧ d = 6 ∧
  h_initial = Real.sqrt (l^2 - x_initial^2) ∧
  h_new = h_initial - d ∧
  (l^2 = h_new^2 + (x_initial + y) ^ 2) → y = 18

theorem foot_slide_distance :
  ladder_foot_slide 30 (Real.sqrt (30^2 - 6^2)) ((Real.sqrt (30^2 - 6^2)) - 6) 6 6 18 :=
by
  sorry

end foot_slide_distance_l211_211634


namespace helen_gas_needed_l211_211959

-- Defining constants for the problem
def largeLawnGasPerUsage (n : ℕ) : ℕ := (n / 3) * 2
def smallLawnGasPerUsage (n : ℕ) : ℕ := (n / 2) * 1

def monthsSpringFall : ℕ := 4
def monthsSummer : ℕ := 4

def largeLawnCutsSpringFall : ℕ := 1
def largeLawnCutsSummer : ℕ := 3

def smallLawnCutsSpringFall : ℕ := 2
def smallLawnCutsSummer : ℕ := 2

-- Number of times Helen cuts large lawn in March-April and September-October
def largeLawnSpringFallCuts : ℕ := monthsSpringFall * largeLawnCutsSpringFall

-- Number of times Helen cuts large lawn in May-August
def largeLawnSummerCuts : ℕ := monthsSummer * largeLawnCutsSummer

-- Total cuts for large lawn
def totalLargeLawnCuts : ℕ := largeLawnSpringFallCuts + largeLawnSummerCuts

-- Number of times Helen cuts small lawn in March-April and September-October
def smallLawnSpringFallCuts : ℕ := monthsSpringFall * smallLawnCutsSpringFall

-- Number of times Helen cuts small lawn in May-August
def smallLawnSummerCuts : ℕ := monthsSummer * smallLawnCutsSummer

-- Total cuts for small lawn
def totalSmallLawnCuts : ℕ := smallLawnSpringFallCuts + smallLawnSummerCuts

-- Total gas needed for both lawns
def totalGasNeeded : ℕ :=
  largeLawnGasPerUsage totalLargeLawnCuts + smallLawnGasPerUsage totalSmallLawnCuts

-- The statement to prove
theorem helen_gas_needed : totalGasNeeded = 18 := sorry

end helen_gas_needed_l211_211959


namespace TimPrankCombinations_l211_211494

-- Definitions of the conditions in the problem
def MondayChoices : ℕ := 3
def TuesdayChoices : ℕ := 1
def WednesdayChoices : ℕ := 6
def ThursdayChoices : ℕ := 4
def FridayChoices : ℕ := 2

-- The main theorem to prove the total combinations
theorem TimPrankCombinations : 
  MondayChoices * TuesdayChoices * WednesdayChoices * ThursdayChoices * FridayChoices = 144 := 
by
  sorry

end TimPrankCombinations_l211_211494


namespace company_KW_price_l211_211306

theorem company_KW_price (A B : ℝ) (x : ℝ) (h1 : P = x * A) (h2 : P = 2 * B) (h3 : P = (6 / 7) * (A + B)) : x = 1.666666666666667 := 
sorry

end company_KW_price_l211_211306


namespace bicycle_weight_l211_211587

theorem bicycle_weight (b s : ℕ) (h1 : 10 * b = 5 * s) (h2 : 5 * s = 200) : b = 20 := 
by 
  sorry

end bicycle_weight_l211_211587


namespace base_conversion_l211_211870

theorem base_conversion (b : ℝ) (h : 2 * b^2 + 3 = 51) : b = 2 * Real.sqrt 6 :=
by
  sorry

end base_conversion_l211_211870


namespace least_five_digit_congruent_6_mod_17_l211_211458

theorem least_five_digit_congruent_6_mod_17 : ∃ n: ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 6 ∧ ∀ m: ℕ, 10000 ≤ m ∧ m < 100000 ∧ m % 17 = 6 → n ≤ m :=
sorry

end least_five_digit_congruent_6_mod_17_l211_211458


namespace map_representation_l211_211384

theorem map_representation (length_1_cm length_2_cm : ℕ) (distance_1_km : ℕ) :
  length_1_cm = 15 → distance_1_km = 90 →
  length_2_cm = 20 →
  distance_2_km = length_2_cm * (distance_1_km / length_1_cm) →
  distance_2_km = 120 :=
by
  intros h_len1 h_distance1 h_len2 h_calculate 
  rw [h_len1, h_distance1, h_len2] at h_calculate
  have scale := 90 / 15
  have distance_2_km := 20 * scale
  have : scale = 6 := by norm_num
  rw [this] at h_calculate
  norm_num at distance_2_km
  assumption
  sorry

end map_representation_l211_211384


namespace square_floor_tile_count_l211_211069

theorem square_floor_tile_count (n : ℕ) (h : 2 * n - 1 = 49) : n^2 = 625 := by
  sorry

end square_floor_tile_count_l211_211069


namespace percentage_decrease_is_24_l211_211030

-- Define the given constants Rs. 820 and Rs. 1078.95
def current_price : ℝ := 820
def original_price : ℝ := 1078.95

-- Define the percentage decrease P
def percentage_decrease (P : ℝ) : Prop :=
  original_price - (P / 100) * original_price = current_price

-- Prove that percentage decrease P is approximately 24
theorem percentage_decrease_is_24 : percentage_decrease 24 :=
by
  unfold percentage_decrease
  sorry

end percentage_decrease_is_24_l211_211030


namespace x_power_12_l211_211689

theorem x_power_12 (x : ℝ) (h : x + 1 / x = 2) : x^12 = 1 :=
by sorry

end x_power_12_l211_211689


namespace average_speed_increased_pace_l211_211903

theorem average_speed_increased_pace 
  (speed_constant : ℝ) (time_constant : ℝ) (distance_increased : ℝ) (total_time : ℝ) 
  (h1 : speed_constant = 15) 
  (h2 : time_constant = 3) 
  (h3 : distance_increased = 190) 
  (h4 : total_time = 13) :
  (distance_increased / (total_time - time_constant)) = 19 :=
by
  sorry

end average_speed_increased_pace_l211_211903


namespace not_pass_first_quadrant_l211_211710

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  (1/5)^(x + 1) + m

theorem not_pass_first_quadrant (m : ℝ) : 
  (∀ x : ℝ, (1/5)^(x+1) + m ≤ 0) ↔ m ≤ -(1/5) :=
  by
  sorry

end not_pass_first_quadrant_l211_211710


namespace map_scale_l211_211394

theorem map_scale (represents_15cm_as_km : 15 * 6 = 90) :
  20 * 6 = 120 :=
by
  sorry

end map_scale_l211_211394


namespace line_from_complex_condition_l211_211837

theorem line_from_complex_condition (z : ℂ) (h : ∃ x y : ℝ, z = x + y * I ∧ (3 * y + 4 * x = 0)) : 
  ∃ (a b : ℝ), (∀ (x y : ℝ), z = x + y * I → 3 * y + 4 * x = 0 → z = a + b * I ∧ 4 * x + 3 * y = 0) := 
sorry

end line_from_complex_condition_l211_211837


namespace greatest_prime_factor_341_l211_211241

theorem greatest_prime_factor_341 : ∃ p, prime p ∧ p ∣ 341 ∧ p = 17 := 
by {
  use 17,
  split,
  { exact prime_of_factor 17, -- 17 is a prime number
  },
  {
    split,
    {
      norm_num,
      -- proof that 17 divides 341
      sorry,
    },
    -- p == 17
    refl,
  },
}

end greatest_prime_factor_341_l211_211241


namespace map_length_representation_l211_211411

theorem map_length_representation :
  ∀ (length1 length2 km1 : ℝ), 
  length1 = 15 ∧ km1 = 90 ∧ length2 = 20 →
  (length2 * (km1 / length1) = 120) :=
by
  intros length1 length2 km1
  intro h
  cases h with hl1 h
  cases h with hkm1 hl2
  rw [hl1, hkm1, hl2]
  simp
  sorry

end map_length_representation_l211_211411


namespace integral_P_zero_l211_211308

open polynomial

noncomputable def T_k (n k : ℕ) (x : ℝ) : ℝ :=
∏ i in finset.range (n + 1), if i = k then 1 else x - i

noncomputable def P (n : ℕ) : polynomial ℝ :=
∑ k in finset.range (n + 1), polynomial.C (T_k n k k)

theorem integral_P_zero (n : ℕ) (s t : ℕ) (h1 : 1 ≤ s) (h2 : s ≤ n) (h3 : 1 ≤ t) (h4 : t ≤ n) : 
  ∫ x in s.to_real..t.to_real, (P n).eval x = 0 :=
by sorry

end integral_P_zero_l211_211308


namespace part_one_part_two_range_l211_211533

/-
Definitions based on conditions from the problem:
- Given vectors ax = (\cos x, \sin x), bx = (3, - sqrt(3))
- Domain for x is [0, π]
--
- Prove if a + b is parallel to b, then x = 5π / 6
- Definition of function f(x), and g(x) based on problem requirements.
- Prove the range of g(x) is [-3, sqrt(3)]
-/

/-
Part (1):
Given ax + bx = (cos x + 3, sin x - sqrt(3)) is parallel to bx =  (3, - sqrt(3));
Prove that x = 5π / 6 under x ∈ [0, π].
-/
noncomputable def vector_ax (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def vector_bx : ℝ × ℝ := (3, - Real.sqrt 3)

theorem part_one (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) 
  (h_parallel : (vector_ax x).1 + vector_bx.1 = (vector_ax x).2 + vector_bx.2) :
  x = 5 * Real.pi / 6 :=
  sorry

/-
Part (2):
Let f(x) = 3 cos x - sqrt(3) sin x.
The function g(x) = -2 sqrt(3) sin(1/2 x - 2π/3) is defined by shifting f(x) right by π/3 and doubling the horizontal coordinate.
Prove the range of g(x) is [-3, sqrt(3)].
-/
noncomputable def f (x : ℝ) := 3 * Real.cos x - Real.sqrt 3 * Real.sin x
noncomputable def g (x : ℝ) := -2 * Real.sqrt 3 * Real.sin (0.5 * x - 2 * Real.pi / 3)

theorem part_two_range (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) : 
  -3 ≤ g x ∧ g x ≤ Real.sqrt 3 :=
  sorry

end part_one_part_two_range_l211_211533


namespace intersecting_graphs_l211_211024

theorem intersecting_graphs (a b c d : ℝ) (h₁ : (3, 6) = (3, -|3 - a| + b))
  (h₂ : (9, 2) = (9, -|9 - a| + b))
  (h₃ : (3, 6) = (3, |3 - c| + d))
  (h₄ : (9, 2) = (9, |9 - c| + d)) : 
  a + c = 12 := 
sorry

end intersecting_graphs_l211_211024


namespace elastic_collision_inelastic_collision_l211_211450

-- Given conditions for Case A and Case B
variables (L V : ℝ) (m : ℝ) -- L is length of the rods, V is the speed, m is mass of each sphere

-- Prove Case A: The dumbbells separate maintaining their initial velocities
theorem elastic_collision (h1 : L > 0) (h2 : V > 0) (h3 : m > 0) :
  -- After a perfectly elastic collision, the dumbbells separate maintaining their initial velocities
  true := sorry

-- Prove Case B: The dumbbells start rotating around the collision point with angular velocity V / (2 * L)
theorem inelastic_collision (h1 : L > 0) (h2 : V > 0) (h3 : m > 0) :
  -- After a perfectly inelastic collision, the dumbbells start rotating around the collision point with angular velocity V / (2 * L)
  true := sorry

end elastic_collision_inelastic_collision_l211_211450


namespace num_convex_quadrilateral_angles_arith_prog_l211_211961

theorem num_convex_quadrilateral_angles_arith_prog :
  ∃ (S : Finset (Finset ℤ)), S.card = 29 ∧
    ∀ {a b c d : ℤ}, {a, b, c, d} ∈ S →
      a + b + c + d = 360 ∧
      a < b ∧ b < c ∧ c < d ∧
      ∃ (m d_diff : ℤ), 
        m - d_diff = a ∧
        m = b ∧
        m + d_diff = c ∧
        m + 2 * d_diff = d ∧
        a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 :=
sorry

end num_convex_quadrilateral_angles_arith_prog_l211_211961


namespace jackson_final_grade_l211_211004

def jackson_hours_playing_video_games : ℕ := 9

def ratio_study_to_play : ℚ := 1 / 3

def time_spent_studying (hours_playing : ℕ) (ratio : ℚ) : ℚ := hours_playing * ratio

def points_per_hour_studying : ℕ := 15

def jackson_grade (time_studied : ℚ) (points_per_hour : ℕ) : ℚ := time_studied * points_per_hour

theorem jackson_final_grade :
  jackson_grade
    (time_spent_studying jackson_hours_playing_video_games ratio_study_to_play)
    points_per_hour_studying = 45 :=
by
  sorry

end jackson_final_grade_l211_211004


namespace sum_of_digits_7_pow_11_l211_211468

/-
  Let's define the problem with the given conditions and expected answer.
  We need to prove that the sum of the tens digit and the ones digit of 7^11 is 7.
-/

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_7_pow_11 : tens_digit (7 ^ 11) + ones_digit (7 ^ 11) = 7 :=
sorry

end sum_of_digits_7_pow_11_l211_211468


namespace equal_division_of_cookie_l211_211764

theorem equal_division_of_cookie (total_area : ℝ) (friends : ℕ) (area_per_person : ℝ) 
  (h1 : total_area = 81.12) 
  (h2 : friends = 6) 
  (h3 : area_per_person = total_area / friends) : 
  area_per_person = 13.52 :=
by 
  sorry

end equal_division_of_cookie_l211_211764


namespace solution_set_inequality_l211_211299

theorem solution_set_inequality (x : ℝ) : (-2 * x + 3 < 0) ↔ (x > 3 / 2) := by 
  sorry

end solution_set_inequality_l211_211299


namespace heidi_zoe_paint_fraction_l211_211969

theorem heidi_zoe_paint_fraction (H_period : ℝ) (HZ_period : ℝ) :
  (H_period = 60 → HZ_period = 40 → (8 / 40) = (1 / 5)) :=
by intros H_period_eq HZ_period_eq
   sorry

end heidi_zoe_paint_fraction_l211_211969


namespace tom_sleep_hours_l211_211215

-- Define initial sleep hours and increase fraction
def initial_sleep_hours : ℕ := 6
def increase_fraction : ℚ := 1 / 3

-- Define the function to calculate increased sleep
def increased_sleep_hours (initial : ℕ) (fraction : ℚ) : ℚ :=
  initial * fraction

-- Define the function to calculate total sleep hours
def total_sleep_hours (initial : ℕ) (increased : ℚ) : ℚ :=
  initial + increased

-- Theorem stating Tom's total sleep hours per night after the increase
theorem tom_sleep_hours (initial : ℕ) (fraction : ℚ) (increased : ℚ) (total : ℚ) :
  initial = initial_sleep_hours →
  fraction = increase_fraction →
  increased = increased_sleep_hours initial fraction →
  total = total_sleep_hours initial increased →
  total = 8 :=
by
  intros h_init h_frac h_incr h_total
  rw [h_init, h_frac] at h_incr
  rw [h_init, h_incr] at h_total
  sorry

end tom_sleep_hours_l211_211215


namespace number_of_distinct_m_values_l211_211364

theorem number_of_distinct_m_values (m : ℤ) :
  (∃ x1 x2 : ℤ, x1 * x2 = 36 ∧ x1 + x2 = m) →
  set.card {m | ∃ x1 x2 : ℤ, x1 * x2 = 36 ∧ x1 + x2 = m} = 10 :=
by
  sorry

end number_of_distinct_m_values_l211_211364


namespace find_f2_l211_211108

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem find_f2 (h : f a b (-2) = 2) : f a b 2 = 0 := by
  sorry

end find_f2_l211_211108


namespace uncle_jerry_total_tomatoes_l211_211758

def tomatoes_reaped_yesterday : ℕ := 120
def tomatoes_reaped_more_today : ℕ := 50

theorem uncle_jerry_total_tomatoes : 
  tomatoes_reaped_yesterday + (tomatoes_reaped_yesterday + tomatoes_reaped_more_today) = 290 :=
by 
  sorry

end uncle_jerry_total_tomatoes_l211_211758


namespace arithmetic_progression_general_formula_geometric_progression_condition_l211_211112

-- Arithmetic progression problem
theorem arithmetic_progression_general_formula :
  ∀ (a : ℕ → ℚ) (S : ℕ → ℚ),
  (∀ n, a n > 0) →
  S 10 = 70 →
  a 1 = 1 →
  (∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2) →
  (∀ n, a n = 1 + (n - 1) * (4/3)) :=
by
  sorry

-- Geometric progression problem
theorem geometric_progression_condition :
  ∀ (a : ℕ → ℚ) (S : ℕ → ℚ),
  (∀ n, a n > 0) →
  a 1 = 1 →
  a 4 = 1/8 →
  (∀ n, S n = a 1 * (1 - (1/2)^n) / (1 - 1/2)) →
  (∃ n, S n > 100 * a n ∧ ∀ m < n, S m ≤ 100 * a m) :=
by
  use 7
  sorry

end arithmetic_progression_general_formula_geometric_progression_condition_l211_211112


namespace bella_eats_six_apples_a_day_l211_211497

variable (A : ℕ) -- Number of apples Bella eats per day
variable (G : ℕ) -- Total number of apples Grace picks in 6 weeks
variable (B : ℕ) -- Total number of apples Bella eats in 6 weeks

-- Definitions for the conditions 
def condition1 := B = 42 * A
def condition2 := B = (1 / 3) * G
def condition3 := (2 / 3) * G = 504

-- Final statement that needs to be proved
theorem bella_eats_six_apples_a_day (A G B : ℕ) 
  (h1 : condition1 A B) 
  (h2 : condition2 G B) 
  (h3 : condition3 G) 
  : A = 6 := by sorry

end bella_eats_six_apples_a_day_l211_211497


namespace sum_of_digits_7_pow_11_l211_211473

theorem sum_of_digits_7_pow_11 : 
  let n := 7 in
  let power := 11 in
  let last_two_digits := (n ^ power) % 100 in
  let tens_digit := last_two_digits / 10 in
  let ones_digit := last_two_digits % 10 in
  tens_digit + ones_digit = 7 :=
by {
  sorry
}

end sum_of_digits_7_pow_11_l211_211473


namespace count_integer_solutions_l211_211540

theorem count_integer_solutions :
  {x : ℤ | (x - 2)^2 ≤ 4}.card = 5 :=
by
  sorry

end count_integer_solutions_l211_211540


namespace peter_hunts_3_times_more_than_mark_l211_211557

theorem peter_hunts_3_times_more_than_mark : 
  ∀ (Sam Rob Mark Peter : ℕ),
  Sam = 6 →
  Rob = Sam / 2 →
  Mark = (Sam + Rob) / 3 →
  Sam + Rob + Mark + Peter = 21 →
  Peter = 3 * Mark :=
by
  intros Sam Rob Mark Peter h1 h2 h3 h4
  sorry

end peter_hunts_3_times_more_than_mark_l211_211557


namespace gf_three_l211_211855

def f (x : ℕ) : ℕ := x^3 - 4 * x + 5
def g (x : ℕ) : ℕ := 3 * x^2 + x + 2

theorem gf_three : g (f 3) = 1222 :=
by {
  -- We would need to prove the given mathematical statement here.
  sorry
}

end gf_three_l211_211855


namespace worker_followed_instructions_l211_211845

def initial_trees (grid_size : ℕ) : ℕ := grid_size * grid_size

noncomputable def rows_of_trees (rows left each_row : ℕ) : ℕ := rows * each_row

theorem worker_followed_instructions :
  initial_trees 7 = 49 →
  rows_of_trees 5 20 4 = 20 →
  rows_of_trees 5 10 4 = 39 →
  (∃ T : Finset (Fin 7 × Fin 7), T.card = 10) :=
by
  sorry

end worker_followed_instructions_l211_211845


namespace total_team_cost_l211_211207

-- Define the costs of individual items and the number of players
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.2
def socks_cost : ℝ := 6.8
def num_players : ℕ := 16

-- Define the total cost for equipment for one player
def player_cost : ℝ :=
  jersey_cost + shorts_cost + socks_cost

-- The main theorem stating the total cost for all players
theorem total_team_cost : num_players * player_cost = 752 := by
  sorry

end total_team_cost_l211_211207


namespace find_side_length_S2_l211_211013

-- Define the variables and conditions
variables (r s : ℕ)
def is_solution (r s : ℕ) : Prop :=
  2 * r + s = 2160 ∧ 2 * r + 3 * s = 3450

-- Define the problem statement
theorem find_side_length_S2 (r s : ℕ) (h : is_solution r s) : s = 645 :=
sorry

end find_side_length_S2_l211_211013


namespace reciprocal_of_2023_l211_211191

theorem reciprocal_of_2023 :
  1 / 2023 = 1 / (2023 : ℝ) :=
by
  sorry

end reciprocal_of_2023_l211_211191


namespace triangle_congruence_example_l211_211816

variable {A B C : Type}
variable (A' B' C' : Type)

def triangle (A B C : Type) : Prop := true

def congruent (t1 t2 : Prop) : Prop := true

variable (P : ℕ)

def perimeter (t : Prop) (p : ℕ) : Prop := true

def length (a b : Type) (l : ℕ) : Prop := true

theorem triangle_congruence_example :
  ∀ (A B C A' B' C' : Type) (h_cong : congruent (triangle A B C) (triangle A' B' C'))
    (h_perimeter : perimeter (triangle A B C) 20)
    (h_AB : length A B 8)
    (h_BC : length B C 5),
    length A C 7 :=
by sorry

end triangle_congruence_example_l211_211816


namespace largest_term_l211_211007

-- Given conditions
def U : ℕ := 2 * (2010 ^ 2011)
def V : ℕ := 2010 ^ 2011
def W : ℕ := 2009 * (2010 ^ 2010)
def X : ℕ := 2 * (2010 ^ 2010)
def Y : ℕ := 2010 ^ 2010
def Z : ℕ := 2010 ^ 2009

-- Proposition to prove
theorem largest_term : 
  (U - V) > (V - W) ∧ 
  (U - V) > (W - X + 100) ∧ 
  (U - V) > (X - Y) ∧ 
  (U - V) > (Y - Z) := 
by 
  sorry

end largest_term_l211_211007


namespace four_digit_numbers_gt_3000_l211_211339

theorem four_digit_numbers_gt_3000 (d1 d2 d3 d4 : ℕ) (h_digits : (d1, d2, d3, d4) = (2, 0, 5, 5)) (h_distinct_4digit : (d1 * 1000 + d2 * 100 + d3 * 10 + d4) > 3000) :
  ∃ count, count = 3 := sorry

end four_digit_numbers_gt_3000_l211_211339


namespace scientific_notation_correct_l211_211848

def n : ℝ := 12910000

theorem scientific_notation_correct : n = 1.291 * 10^7 := 
by
  sorry

end scientific_notation_correct_l211_211848


namespace all_non_positive_l211_211325

theorem all_non_positive (n : ℕ) (a : ℕ → ℤ) 
  (h₀ : a 0 = 0) 
  (hₙ : a n = 0) 
  (ineq : ∀ k, 1 ≤ k ∧ k ≤ n - 1 → a (k - 1) - 2 * a k + a (k + 1) ≥ 0) : ∀ k, a k ≤ 0 :=
by 
  sorry

end all_non_positive_l211_211325


namespace map_length_scale_l211_211392

theorem map_length_scale (len1 len2 : ℕ) (dist1 dist2 : ℕ) (h1 : len1 = 15) (h2 : dist1 = 90) (h3 : len2 = 20) :
  dist2 = 120 :=
by
  sorry

end map_length_scale_l211_211392


namespace divisors_of_30_l211_211704

theorem divisors_of_30 : ∃ (n : ℕ), n = 16 ∧ (∀ d : ℤ, d ∣ 30 → (d ≤ 30 ∧ d ≥ -30)) :=
by
  sorry

end divisors_of_30_l211_211704


namespace factorize_expression_l211_211476

variable {a b x y : ℝ}

theorem factorize_expression :
  (x^2 - y^2) * a^2 - (x^2 - y^2) * b^2 = (x + y) * (x - y) * (a + b) * (a - b) :=
by
  sorry

end factorize_expression_l211_211476


namespace identify_conic_section_l211_211926

theorem identify_conic_section (x y : ℝ) :
  (x + 7)^2 = (5 * y - 6)^2 + 125 →
  ∃ a b c d e f : ℝ, a * x^2 + b * y^2 + c * x + d * y + e * x * y + f = 0 ∧
  (a > 0) ∧ (b < 0) := sorry

end identify_conic_section_l211_211926


namespace marks_in_physics_l211_211910

section
variables (P C M B CS : ℕ)

-- Given conditions
def condition_1 : Prop := P + C + M + B + CS = 375
def condition_2 : Prop := P + M + B = 255
def condition_3 : Prop := P + C + CS = 210

-- Prove that P = 90
theorem marks_in_physics : condition_1 P C M B CS → condition_2 P M B → condition_3 P C CS → P = 90 :=
by sorry
end

end marks_in_physics_l211_211910


namespace find_b_l211_211502

def has_exactly_one_real_solution (f : ℝ → ℝ) : Prop :=
  ∃! x : ℝ, f x = 0

theorem find_b (b : ℝ) :
  (∃! (x : ℝ), x^3 - b * x^2 - 3 * b * x + b^2 - 4 = 0) ↔ b < 2 :=
by
  sorry

end find_b_l211_211502


namespace divisors_of_30_l211_211700

theorem divisors_of_30 : 
  ∃ n : ℕ, (∀ d : ℤ, d ∣ 30 → d > 0 → d ≤ 30) ∧ (∀ d : ℤ, d ∣ 30 → -d ∣ 30) ∧ (∀ d : ℤ, d ∣ 30 → -d = d → d = 0) ∧ 2 * (Finset.card (Finset.filter (λ d, 0 < d) (Finset.filter (λ d, d ∣ 30) (Finset.Icc 1 30).toFinset))) = 16 :=
by {
   sorry
}

end divisors_of_30_l211_211700


namespace parallel_vectors_sum_coords_l211_211129

theorem parallel_vectors_sum_coords
  (x y : ℝ)
  (a b : ℝ × ℝ × ℝ)
  (h_a : a = (2, x, 3))
  (h_b : b = (-4, 2, y))
  (h_parallel : ∃ k : ℝ, a = k • b) :
  x + y = -7 :=
sorry

end parallel_vectors_sum_coords_l211_211129


namespace sum_of_prime_factors_210630_l211_211261

theorem sum_of_prime_factors_210630 : (2 + 3 + 5 + 7 + 17 + 59) = 93 := by
  -- Proof to be provided
  sorry

end sum_of_prime_factors_210630_l211_211261


namespace reciprocal_of_subtraction_l211_211618

-- Defining the conditions
def x : ℚ := 1 / 9
def y : ℚ := 2 / 3

-- Defining the main theorem statement
theorem reciprocal_of_subtraction : (1 / (y - x)) = 9 / 5 :=
by
  sorry

end reciprocal_of_subtraction_l211_211618


namespace sum_proof_l211_211143

theorem sum_proof (X Y : ℝ) (hX : 0.45 * X = 270) (hY : 0.35 * Y = 210) : 
  (0.75 * X) + (0.55 * Y) = 780 := by
  sorry

end sum_proof_l211_211143


namespace greatest_prime_factor_of_341_l211_211257

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem greatest_prime_factor_of_341 : ∃ p, is_prime p ∧ p ∣ 341 ∧ ∀ q, is_prime q → q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_of_341_l211_211257


namespace percent_decrease_l211_211272

variable (OriginalPrice : ℝ) (SalePrice : ℝ)

theorem percent_decrease : 
  OriginalPrice = 100 → 
  SalePrice = 30 → 
  ((OriginalPrice - SalePrice) / OriginalPrice) * 100 = 70 :=
by
  intros h1 h2
  sorry

end percent_decrease_l211_211272


namespace tod_north_distance_l211_211213

-- Given conditions as variables
def speed : ℕ := 25  -- speed in miles per hour
def time : ℕ := 6    -- time in hours
def west_distance : ℕ := 95  -- distance to the west in miles

-- Prove the distance to the north given conditions
theorem tod_north_distance : time * speed - west_distance = 55 := by
  sorry

end tod_north_distance_l211_211213


namespace right_triangle_legs_sum_squares_area_l211_211102

theorem right_triangle_legs_sum_squares_area:
  ∀ (a b c : ℝ), 
  (0 < a) → (0 < b) → (0 < c) → 
  (a^2 + b^2 = c^2) → 
  (1 / 2 * a * b = 24) → 
  (a^2 + b^2 = 48) → 
  (a = 2 * Real.sqrt 6 ∧ b = 2 * Real.sqrt 6 ∧ c = 4 * Real.sqrt 3) := 
by
  sorry

end right_triangle_legs_sum_squares_area_l211_211102


namespace number_of_sides_is_15_l211_211347

variable {n : ℕ} -- n is the number of sides

-- Define the conditions
def sum_of_all_but_one_angle (n : ℕ) : Prop :=
  180 * (n - 2) - 2190 > 0 ∧ 180 * (n - 2) - 2190 < 180

-- State the theorem to be proven
theorem number_of_sides_is_15 (n : ℕ) (h : sum_of_all_but_one_angle n) : n = 15 :=
sorry

end number_of_sides_is_15_l211_211347


namespace exists_root_between_roots_l211_211565

theorem exists_root_between_roots 
  (a b c : ℝ) 
  (h_a : a ≠ 0) 
  (x₁ x₂ : ℝ) 
  (h₁ : a * x₁^2 + b * x₁ + c = 0) 
  (h₂ : -a * x₂^2 + b * x₂ + c = 0) 
  (hx : x₁ < x₂) :
  ∃ x₃ : ℝ, x₁ < x₃ ∧ x₃ < x₂ ∧ (a / 2) * x₃^2 + b * x₃ + c = 0 :=
by 
  sorry

end exists_root_between_roots_l211_211565


namespace findFirstCarSpeed_l211_211613

noncomputable def firstCarSpeed (v : ℝ) (blackCarSpeed : ℝ) (initialGap : ℝ) (timeToCatchUp : ℝ) : Prop :=
  blackCarSpeed * timeToCatchUp = initialGap + v * timeToCatchUp → v = 30

theorem findFirstCarSpeed :
  firstCarSpeed 30 50 20 1 :=
by
  sorry

end findFirstCarSpeed_l211_211613


namespace family_e_initial_members_l211_211630

theorem family_e_initial_members 
(a b c d f E : ℕ) 
(h_a : a = 7) 
(h_b : b = 8) 
(h_c : c = 10) 
(h_d : d = 13) 
(h_f : f = 10)
(h_avg : (a - 1 + b - 1 + c - 1 + d - 1 + E - 1 + f - 1) / 6 = 8) : 
E = 6 := 
by 
  sorry

end family_e_initial_members_l211_211630


namespace Q_evaluation_at_2_l211_211034

noncomputable def Q : Polynomial ℚ := 
  (Polynomial.X^2 + Polynomial.C 4)^2

theorem Q_evaluation_at_2 : 
  Q.eval 2 = 64 :=
by 
  -- We'll skip the proof as per the instructions.
  sorry

end Q_evaluation_at_2_l211_211034


namespace final_pen_count_l211_211059

theorem final_pen_count
  (initial_pens : ℕ := 7) 
  (mike_given_pens : ℕ := 22) 
  (doubled_pens : ℕ := 2)
  (sharon_given_pens : ℕ := 19) :
  let total_after_mike := initial_pens + mike_given_pens
  let total_after_cindy := total_after_mike * doubled_pens
  let final_count := total_after_cindy - sharon_given_pens
  final_count = 39 :=
by
  sorry

end final_pen_count_l211_211059


namespace distinct_m_count_l211_211361

noncomputable def countDistinctMValues : Nat :=
  let pairs := [(1, 36), (2, 18), (3, 12), (4, 9), (6, 6), 
                (-1, -36), (-2, -18), (-3, -12), (-4, -9), (-6, -6)]
  let ms := pairs.map (λ p => p.1 + p.2)
  ms.eraseDups.length

theorem distinct_m_count :
  countDistinctMValues = 10 := sorry

end distinct_m_count_l211_211361


namespace greatest_prime_factor_of_341_l211_211259

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem greatest_prime_factor_of_341 : ∃ p, is_prime p ∧ p ∣ 341 ∧ ∀ q, is_prime q → q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_of_341_l211_211259


namespace mushrooms_gigi_cut_l211_211678

-- Definitions based on conditions in part a)
def pieces_per_mushroom := 4
def kenny_sprinkled := 38
def karla_sprinkled := 42
def pieces_remaining := 8

-- The total number of pieces is the sum of Kenny's, Karla's, and the remaining pieces.
def total_pieces := kenny_sprinkled + karla_sprinkled + pieces_remaining

-- The number of mushrooms GiGi cut up at the beginning is total_pieces divided by pieces_per_mushroom.
def mushrooms_cut := total_pieces / pieces_per_mushroom

-- The theorem to be proved.
theorem mushrooms_gigi_cut (h1 : pieces_per_mushroom = 4)
                           (h2 : kenny_sprinkled = 38)
                           (h3 : karla_sprinkled = 42)
                           (h4 : pieces_remaining = 8)
                           (h5 : total_pieces = kenny_sprinkled + karla_sprinkled + pieces_remaining)
                           (h6 : mushrooms_cut = total_pieces / pieces_per_mushroom) :
  mushrooms_cut = 22 :=
by
  sorry

end mushrooms_gigi_cut_l211_211678


namespace greatest_prime_factor_341_l211_211240

theorem greatest_prime_factor_341 : ∃ p, prime p ∧ p ∣ 341 ∧ p = 17 := 
by {
  use 17,
  split,
  { exact prime_of_factor 17, -- 17 is a prime number
  },
  {
    split,
    {
      norm_num,
      -- proof that 17 divides 341
      sorry,
    },
    -- p == 17
    refl,
  },
}

end greatest_prime_factor_341_l211_211240


namespace solve_for_x_l211_211262

theorem solve_for_x :
  ∃ x : ℝ, x ≠ 0 ∧ (9 * x) ^ 18 = (27 * x) ^ 9 + 81 * x ∧ x = 1 / 3 :=
by
  sorry

end solve_for_x_l211_211262


namespace solution_set_f_l211_211693

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 2^(x - 1) - 2 else 2^(1 - x) - 2

theorem solution_set_f (x : ℝ) : 
  (1 ≤ x ∧ x ≤ 3) ↔ (f (x - 1) ≤ 0) :=
sorry

end solution_set_f_l211_211693


namespace value_of_m_l211_211110

variable (a m : ℝ)
variable (h1 : a > 0)
variable (h2 : -a*m^2 + 2*a*m + 3 = 3)
variable (h3 : m ≠ 0)

theorem value_of_m : m = 2 :=
by
  sorry

end value_of_m_l211_211110


namespace map_distance_l211_211169

theorem map_distance (scale_cm : ℝ) (scale_km : ℝ) (actual_distance_km : ℝ) 
  (h1 : scale_cm = 0.4) (h2 : scale_km = 5.3) (h3 : actual_distance_km = 848) :
  actual_distance_km / (scale_km / scale_cm) = 64 :=
by
  rw [h1, h2, h3]
  -- Further steps would follow here, but to ensure code compiles
  -- and there is no assumption directly from solution steps, we use sorry.
  sorry

end map_distance_l211_211169


namespace direct_proportion_inequality_l211_211691

theorem direct_proportion_inequality (k x1 x2 y1 y2 : ℝ) (h_k : k < 0) (h_y1 : y1 = k * x1) (h_y2 : y2 = k * x2) (h_x : x1 < x2) : y1 > y2 :=
by
  -- The proof will be written here, currently leaving it as sorry
  sorry

end direct_proportion_inequality_l211_211691


namespace Q_at_2_l211_211035

-- Define the polynomial Q(x)
def Q (x : ℚ) : ℚ := x^4 - 8 * x^2 + 16

-- Define the properties of Q(x)
def is_special_polynomial (P : (ℚ → ℚ)) : Prop := 
  degree P = 4 ∧ leading_coeff P = 1 ∧ P.is_root(√3 + √7)

-- Problem: Prove Q(2) = 0 given the polynomial Q meets the conditions.
theorem Q_at_2 (Q : ℚ → ℚ) (h1 : degree Q = 4) (h2 : leading_coeff Q = 1) (h3 : is_root Q (√3 + √7)) : 
  Q 2 = 0 := by
  sorry

end Q_at_2_l211_211035


namespace gcd_largest_of_forms_l211_211479

theorem gcd_largest_of_forms (a b : ℕ) (h1 : a ≠ b) (h2 : a < 10) (h3 : b < 10) :
  Nat.gcd (100 * a + 11 * b) (101 * b + 10 * a) = 45 :=
by
  sorry

end gcd_largest_of_forms_l211_211479


namespace clark_discount_l211_211082

theorem clark_discount (price_per_part : ℕ) (number_of_parts : ℕ) (amount_paid : ℕ)
  (h1 : price_per_part = 80)
  (h2 : number_of_parts = 7)
  (h3 : amount_paid = 439) : 
  (number_of_parts * price_per_part) - amount_paid = 121 := by
  sorry

end clark_discount_l211_211082


namespace mushrooms_gigi_cut_l211_211677

-- Definitions based on conditions in part a)
def pieces_per_mushroom := 4
def kenny_sprinkled := 38
def karla_sprinkled := 42
def pieces_remaining := 8

-- The total number of pieces is the sum of Kenny's, Karla's, and the remaining pieces.
def total_pieces := kenny_sprinkled + karla_sprinkled + pieces_remaining

-- The number of mushrooms GiGi cut up at the beginning is total_pieces divided by pieces_per_mushroom.
def mushrooms_cut := total_pieces / pieces_per_mushroom

-- The theorem to be proved.
theorem mushrooms_gigi_cut (h1 : pieces_per_mushroom = 4)
                           (h2 : kenny_sprinkled = 38)
                           (h3 : karla_sprinkled = 42)
                           (h4 : pieces_remaining = 8)
                           (h5 : total_pieces = kenny_sprinkled + karla_sprinkled + pieces_remaining)
                           (h6 : mushrooms_cut = total_pieces / pieces_per_mushroom) :
  mushrooms_cut = 22 :=
by
  sorry

end mushrooms_gigi_cut_l211_211677


namespace prime_roots_eq_l211_211762

theorem prime_roots_eq (n : ℕ) (hn : 0 < n) :
  (∃ (x1 x2 : ℕ), Prime x1 ∧ Prime x2 ∧ 2*x1^2 - 8*n*x1 + 10*x1 - n^2 + 35*n - 76 = 0 ∧ 
                    2*x2^2 - 8*n*x2 + 10*x2 - n^2 + 35*n - 76 = 0 ∧ x1 ≠ x2 ∧ x1 < x2) →
  n = 3 ∧ ∃ x1 x2 : ℕ, x1 = 2 ∧ x2 = 5 ∧ Prime x1 ∧ Prime x2 ∧
    2*x1^2 - 8*n*x1 + 10*x1 - n^2 + 35*n - 76 = 0 ∧
    2*x2^2 - 8*n*x2 + 10*x2 - n^2 + 35*n - 76 = 0 := 
by
  sorry

end prime_roots_eq_l211_211762


namespace number_of_arrangements_l211_211442

theorem number_of_arrangements (boys girls : ℕ) (adjacent : Bool) (not_ends : Bool) : 
  (boys = 5) → (girls = 2) → (adjacent = true) → (not_ends = true) → 
  (∃ n, n = 960) := by
  intros hboys hgirls hadjacent hnot_ends
  exists 960
  sorry

end number_of_arrangements_l211_211442


namespace negation_is_false_l211_211601

-- Define the proposition and its negation
def proposition (x y : ℝ) : Prop := (x > 2 ∧ y > 3) → (x + y > 5)
def negation_proposition (x y : ℝ) : Prop := ¬ proposition x y

-- The proposition and its negation
theorem negation_is_false : ∀ (x y : ℝ), negation_proposition x y = false :=
by sorry

end negation_is_false_l211_211601


namespace determinant_zero_implies_sum_neg_nine_l211_211856

theorem determinant_zero_implies_sum_neg_nine
  (x y : ℝ)
  (h1 : x ≠ y)
  (h2 : x * y = 1)
  (h3 : (Matrix.det ![
    ![1, 5, 8], 
    ![3, x, y], 
    ![3, y, x]
  ]) = 0) : 
  x + y = -9 := 
sorry

end determinant_zero_implies_sum_neg_nine_l211_211856


namespace coloring_count_l211_211706

theorem coloring_count : 
  ∀ (n : ℕ), n = 2021 → 
  ∃ (ways : ℕ), ways = 3 * 2 ^ 2020 :=
by
  intros n hn
  existsi 3 * 2 ^ 2020
  sorry

end coloring_count_l211_211706


namespace rectangle_dimensions_l211_211784

theorem rectangle_dimensions (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h_area : x * y = 36) (h_perimeter : 2 * x + 2 * y = 30) : 
  (x = 12 ∧ y = 3) ∨ (x = 3 ∧ y = 12) :=
by
  sorry

end rectangle_dimensions_l211_211784


namespace motel_total_rent_l211_211768

theorem motel_total_rent (R40 R60 : ℕ) (total_rent : ℕ) 
  (h1 : total_rent = 40 * R40 + 60 * R60) 
  (h2 : 40 * (R40 + 10) + 60 * (R60 - 10) = total_rent - total_rent / 10) 
  (h3 : total_rent / 10 = 200) : 
  total_rent = 2000 := 
sorry

end motel_total_rent_l211_211768


namespace simplify_expression_l211_211270

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ( ((a ^ (4 / 3 / 5)) ^ (3 / 2)) / ((a ^ (4 / 1 / 5)) ^ 3) ) /
  ( ((a * (a ^ (2 / 3) * b ^ (1 / 3))) ^ (1 / 2)) ^ 4) * 
  (a ^ (1 / 4) * b ^ (1 / 8)) ^ 6 = 1 / ((a ^ (2 / 12)) * (b ^ (1 / 12))) :=
by
  sorry

end simplify_expression_l211_211270


namespace integral_solution_l211_211313

noncomputable def integral_expression : Real → Real :=
  fun x => (1 + (x ^ (3 / 4))) ^ (4 / 5) / (x ^ (47 / 20))

theorem integral_solution :
  ∫ (x : Real), integral_expression x = - (20 / 27) * ((1 + (x ^ (3 / 4)) / (x ^ (3 / 4))) ^ (9 / 5)) + C := 
by 
  sorry

end integral_solution_l211_211313


namespace negation_of_universal_l211_211513

open Classical

theorem negation_of_universal (P : ∀ x : ℤ, x^3 < 1) : ∃ x : ℤ, x^3 ≥ 1 :=
by sorry

end negation_of_universal_l211_211513


namespace solutions_are__l211_211924

def satisfies_system (x y z : ℝ) : Prop :=
  x^2 * y + y^2 * z = 1040 ∧
  x^2 * z + z^2 * y = 260 ∧
  (x - y) * (y - z) * (z - x) = -540

theorem solutions_are_ (x y z : ℝ) :
  satisfies_system x y z ↔ (x = 16 ∧ y = 4 ∧ z = 1) ∨ (x = 1 ∧ y = 16 ∧ z = 4) :=
by
  sorry

end solutions_are__l211_211924


namespace valid_configuration_exists_l211_211982

noncomputable def unique_digits (digits: List ℕ) := (digits.length = List.length (List.eraseDup digits)) ∧ ∀ (d : ℕ), d ∈ digits ↔ d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

theorem valid_configuration_exists :
  ∃ a b c d e f g h i j : ℕ,
  unique_digits [a, b, c, d, e, f, g, h, i, j] ∧
  a * (100 * b + 10 * c + d) * (100 * e + 10 * f + g) = 1000 * h + 100 * i + 10 * 9 + 71 := 
by
  sorry

end valid_configuration_exists_l211_211982


namespace gcd_153_119_l211_211454

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  have h1 : 153 = 119 * 1 + 34 := by rfl
  have h2 : 119 = 34 * 3 + 17 := by rfl
  have h3 : 34 = 17 * 2 := by rfl
  sorry

end gcd_153_119_l211_211454


namespace log_50_between_integers_l211_211754

open Real

-- Declaration of the proof problem
theorem log_50_between_integers (a b : ℤ) (h1 : log 10 = 1) (h2 : log 100 = 2) (h3 : 10 < 50) (h4 : 50 < 100) :
  a + b = 3 :=
by
  sorry

end log_50_between_integers_l211_211754


namespace Andy_and_Carlos_tie_for_first_l211_211084

def AndyLawnArea (A : ℕ) := 3 * A
def CarlosLawnArea (A : ℕ) := A / 4
def BethMowingRate := 90
def CarlosMowingRate := BethMowingRate / 3
def AndyMowingRate := BethMowingRate * 4

theorem Andy_and_Carlos_tie_for_first (A : ℕ) (hA_nonzero : 0 < A) :
  (AndyLawnArea A / AndyMowingRate) = (CarlosLawnArea A / CarlosMowingRate) ∧
  (AndyLawnArea A / AndyMowingRate) < (A / BethMowingRate) :=
by
  unfold AndyLawnArea CarlosLawnArea BethMowingRate CarlosMowingRate AndyMowingRate
  sorry

end Andy_and_Carlos_tie_for_first_l211_211084


namespace chessboard_queen_placements_l211_211168

theorem chessboard_queen_placements :
  ∃ (n : ℕ), n = 864 ∧
  (∀ (qpos : Finset (Fin 8 × Fin 8)), 
    qpos.card = 3 ∧
    (∀ (q1 q2 q3 : Fin 8 × Fin 8), 
      q1 ∈ qpos ∧ q2 ∈ qpos ∧ q3 ∈ qpos ∧ q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3 → 
      (q1.1 = q2.1 ∨ q1.2 = q2.2 ∨ abs (q1.1 - q2.1) = abs (q1.2 - q2.2)) ∧ 
      (q1.1 = q3.1 ∨ q1.2 = q3.2 ∨ abs (q1.1 - q3.1) = abs (q1.2 - q3.2)) ∧ 
      (q2.1 = q3.1 ∨ q2.2 = q3.2 ∨ abs (q2.1 - q3.1) = abs (q2.2 - q3.2)))) ↔ n = 864
:=
by
  sorry

end chessboard_queen_placements_l211_211168


namespace greatest_prime_factor_of_341_l211_211249

theorem greatest_prime_factor_of_341 : ∃ p, Prime p ∧ p ∣ 341 ∧ ∀ q, Prime q ∧ q ∣ 341 → q ≤ p :=
by
  let factors := [3, 7, 17]
  have h : 341 = 17 * 3 * 7 := by sorry
  exists 17
  split
  · exact PrimeNat.prime_17
  split
  · exact dvd.intro (3 * 7) rfl
  intro q hpq hq
  have H : q ∈ factors := by sorry
  exact List.mem_le_of_mod_le (PrimeNat.factors_unique H)

end greatest_prime_factor_of_341_l211_211249


namespace xyz_plus_54_l211_211343

theorem xyz_plus_54 (x y z : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : x * y + z = 53) (h2 : y * z + x = 53) (h3 : z * x + y = 53) : 
  x + y + z = 54 := by
  sorry

end xyz_plus_54_l211_211343


namespace greatest_price_book_l211_211088

theorem greatest_price_book (p : ℕ) (B : ℕ) (D : ℕ) (F : ℕ) (T : ℚ) 
  (h1 : B = 20) 
  (h2 : D = 200) 
  (h3 : F = 5)
  (h4 : T = 0.07) 
  (h5 : ∀ p, 20 * p * (1 + T) ≤ (D - F)) : 
  p ≤ 9 :=
by
  sorry

end greatest_price_book_l211_211088


namespace probability_particle_at_2_3_after_5_moves_l211_211782

noncomputable def binom (n k : ℕ) : ℕ :=
if h : k ≤ n then Nat.choose n k
else 0

theorem probability_particle_at_2_3_after_5_moves :
  let p_right := 1 / 2
  let p_up := 1 / 2
  let moves := 5
  let right_moves := 2
  let up_moves := 3
  let total_probability := (binom moves right_moves) * (p_right ^ right_moves) * (p_up ^ up_moves)
  total_probability = binom moves right_moves * (1/2)^5 := by sorry

end probability_particle_at_2_3_after_5_moves_l211_211782


namespace percentage_divisible_by_6_l211_211263

-- Defining the sets S and T using Lean
def S := {n : ℕ | 1 ≤ n ∧ n ≤ 120}
def T := {n : ℕ | n ∈ S ∧ 6 ∣ n}

-- Proving the percentage of elements in T with respect to S is 16.67%
theorem percentage_divisible_by_6 : 
  (↑(T.card) : ℚ) / (S.card) * 100 = 16.67 := sorry

end percentage_divisible_by_6_l211_211263


namespace parabola_focus_l211_211105

theorem parabola_focus (a b c : ℝ) (h_eq : ∀ x : ℝ, 2 * x^2 + 8 * x - 1 = a * (x + b)^2 + c) :
  ∃ focus : ℝ × ℝ, focus = (-2, -71 / 8) :=
sorry

end parabola_focus_l211_211105


namespace g_is_even_l211_211986

noncomputable def g (x : ℝ) : ℝ := 4 / (3 * x^8 - 7)

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  sorry

end g_is_even_l211_211986


namespace greatest_prime_factor_of_341_l211_211246

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, is_prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, is_prime q ∧ q ∣ 341 → q ≤ p :=
begin
  -- Our goal is to show that p = 31 meets these conditions.
  sorry
end

end greatest_prime_factor_of_341_l211_211246


namespace ellipse_eq_from_hyperbola_l211_211530

noncomputable def hyperbola_eq : Prop :=
  ∀ x y : ℝ, (x^2 / 4 - y^2 / 12 = -1) →
  (x^2 / 4 + y^2 / 16 = 1)

theorem ellipse_eq_from_hyperbola :
  hyperbola_eq :=
by
  sorry

end ellipse_eq_from_hyperbola_l211_211530


namespace count_random_events_l211_211078

-- Definitions based on conditions in the problem
def total_products : ℕ := 100
def genuine_products : ℕ := 95
def defective_products : ℕ := 5
def drawn_products : ℕ := 6

-- Events definitions
def event_1 := drawn_products > defective_products  -- at least 1 genuine product
def event_2 := drawn_products ≥ 3  -- at least 3 defective products
def event_3 := drawn_products = defective_products  -- all 6 are defective
def event_4 := drawn_products - 2 = 4  -- 2 defective and 4 genuine products

-- Dummy definition for random event counter state in the problem context
def random_events : ℕ := 2

-- Main theorem statement
theorem count_random_events :
  (event_1 → true) ∧ 
  (event_2 ∧ ¬ event_3 ∧ event_4) →
  random_events = 2 :=
by
  sorry

end count_random_events_l211_211078


namespace change_in_total_berries_l211_211617

theorem change_in_total_berries (B S : ℕ) (hB : B = 20) (hS : S + B = 50) : (S - B) = 10 := by
  sorry

end change_in_total_berries_l211_211617


namespace defective_units_percentage_l211_211983

variables (D : ℝ)

-- 4% of the defective units are shipped for sale
def percent_defective_shipped : ℝ := 0.04

-- 0.24% of the units produced are defective units that are shipped for sale
def percent_total_defective_shipped : ℝ := 0.0024

-- The theorem to prove: the percentage of the units produced that are defective is 0.06
theorem defective_units_percentage (h : percent_defective_shipped * D = percent_total_defective_shipped) : D = 0.06 :=
sorry

end defective_units_percentage_l211_211983


namespace adults_not_wearing_blue_l211_211302

-- Conditions
def children : ℕ := 45
def adults : ℕ := children / 3
def adults_wearing_blue : ℕ := adults / 3

-- Theorem Statement
theorem adults_not_wearing_blue :
  adults - adults_wearing_blue = 10 :=
sorry

end adults_not_wearing_blue_l211_211302


namespace value_of_n_l211_211053

theorem value_of_n (n : ℝ) (h : n + (n + 1) + (n + 2) + (n + 3) = 20) : n = 3.5 :=
by sorry

end value_of_n_l211_211053


namespace fraction_simplification_l211_211930

theorem fraction_simplification (a b d : ℝ) (h : a^2 + d^2 - b^2 + 2 * a * d ≠ 0) :
  (a^2 + b^2 + d^2 + 2 * b * d) / (a^2 + d^2 - b^2 + 2 * a * d) = (a^2 + (b + d)^2) / ((a + d)^2 + a^2 - b^2) :=
sorry

end fraction_simplification_l211_211930


namespace roots_are_prime_then_a_is_five_l211_211825

theorem roots_are_prime_then_a_is_five (x1 x2 a : ℕ) (h_prime_x1 : Prime x1) (h_prime_x2 : Prime x2)
  (h_eq : x1 + x2 = a) (h_eq_mul : x1 * x2 = a + 1) : a = 5 :=
sorry

end roots_are_prime_then_a_is_five_l211_211825


namespace problem_f_17_l211_211792

/-- Assume that f(1) = 0 and f(m + n) = f(m) + f(n) + 4 * (9 * m * n - 1) for all natural numbers m and n.
    Prove that f(17) = 4832.
-/
theorem problem_f_17 (f : ℕ → ℤ) 
  (h1 : f 1 = 0) 
  (h_func : ∀ m n : ℕ, f (m + n) = f m + f n + 4 * (9 * m * n - 1)) 
  : f 17 = 4832 := 
sorry

end problem_f_17_l211_211792


namespace sin_B_value_l211_211349

variable {A B C : Real}
variable {a b c : Real}
variable {sin_A sin_B sin_C : Real}

-- Given conditions as hypotheses
axiom h1 : c = 2 * a
axiom h2 : b * sin_B - a * sin_A = (1 / 2) * a * sin_C

-- The statement to prove
theorem sin_B_value : sin_B = Real.sqrt 7 / 4 :=
by
  -- Proof omitted
  sorry

end sin_B_value_l211_211349


namespace train_speed_l211_211070

def distance := 300 -- meters
def time := 18 -- seconds

noncomputable def speed_kmh := 
  let speed_ms := distance / time -- speed in meters per second
  speed_ms * 3.6 -- convert to kilometers per hour

theorem train_speed : speed_kmh = 60 := 
  by
    -- The proof steps are omitted
    sorry

end train_speed_l211_211070


namespace gena_hits_target_l211_211952

-- Definitions from the problem conditions
def initial_shots : ℕ := 5
def total_shots : ℕ := 17
def shots_per_hit : ℕ := 2

-- Mathematical equivalent proof statement
theorem gena_hits_target (G : ℕ) (H : G * shots_per_hit + initial_shots = total_shots) : G = 6 :=
by
  sorry

end gena_hits_target_l211_211952


namespace determine_m_for_unique_solution_l211_211184

-- Define the quadratic equation and the condition for a unique solution
def quadratic_eq_has_one_solution (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c = 0

-- Define the specific quadratic equation and its discriminant
def specific_quadratic_eq (m : ℝ) : Prop :=
  quadratic_eq_has_one_solution 3 (-7) m

-- State the main theorem to prove the value of m
theorem determine_m_for_unique_solution :
  specific_quadratic_eq (49 / 12) :=
by
  unfold specific_quadratic_eq quadratic_eq_has_one_solution
  sorry

end determine_m_for_unique_solution_l211_211184


namespace events_A_and_B_independent_l211_211580

-- Definitions of the events and sample space
def sample_space : Set (Set Bool) := 
  {{true, true, true}, {true, true, false}, {true, false, true}, {false, true, true},
   {true, false, false}, {false, true, false}, {false, false, true}, {false, false, false}}

-- Event that there are both boys (true) and girls (false)
def event_A : Set (Set Bool) :=
  {{true, true, false}, {true, false, true}, {false, true, true}, 
   {true, false, false}, {false, true, false}, {false, false, true}}

-- Event that there is at most one boy
def event_B : Set (Set Bool) :=
  {{true, false, false}, {false, true, false}, {false, false, true}, {false, false, false}}

-- Probability of an event given uniform probability distribution on the sample space
def probability (e : Set (Set Bool)) : ℝ :=
  (e.card : ℝ) / (sample_space.card : ℝ)

-- Independence of two events
def independent (e1 e2 : Set (Set Bool)) : Prop :=
  probability (e1 ∩ e2) = probability e1 * probability e2

theorem events_A_and_B_independent : independent event_A event_B := by
  sorry

end events_A_and_B_independent_l211_211580


namespace go_stones_perimeter_l211_211543

-- Define the conditions for the problem
def stones_wide : ℕ := 4
def stones_tall : ℕ := 8

-- Define what we want to prove based on the conditions
theorem go_stones_perimeter : 2 * stones_wide + 2 * stones_tall - 4 = 20 :=
by
  -- Proof would normally go here
  sorry

end go_stones_perimeter_l211_211543


namespace map_length_representation_l211_211420

theorem map_length_representation (a b : ℕ) (h : a = 15 ∧ b = 90) : b * (20 / a) = 120 :=
by
  sorry

end map_length_representation_l211_211420


namespace percent_divisible_by_six_up_to_120_l211_211267

theorem percent_divisible_by_six_up_to_120 : 
  let total_numbers := 120
  let divisible_by_six := total_numbers / 6
  let percentage := (divisible_by_six * 100) / total_numbers
  percentage = 50 / 3 := sorry

end percent_divisible_by_six_up_to_120_l211_211267


namespace smallest_k_correct_l211_211047

noncomputable def smallest_k (n m : ℕ) (hn : 0 < n) (hm : 0 < m ∧ m ≤ 5) : ℕ :=
    6

theorem smallest_k_correct (n : ℕ) (m : ℕ) (hn : 0 < n) (hm : 0 < m ∧ m ≤ 5) :
  64 ^ smallest_k n m hn hm + 32 ^ m > 4 ^ (16 + n) :=
sorry

end smallest_k_correct_l211_211047


namespace tile_area_l211_211031

-- Define the properties and conditions of the tile

structure Tile where
  sides : Fin 9 → ℝ 
  six_of_length_1 : ∀ i : Fin 6, sides i = 1 
  congruent_quadrilaterals : Fin 3 → Quadrilateral

structure Quadrilateral where
  length : ℝ
  width : ℝ

-- Given the tile structure, calculate the area
noncomputable def area_of_tile (t: Tile) : ℝ := sorry

-- Statement: Prove the area of the tile given the conditions
theorem tile_area (t : Tile) : area_of_tile t = (4 * Real.sqrt 3 / 3) :=
  sorry

end tile_area_l211_211031


namespace evaluate_fraction_sum_l211_211116

theorem evaluate_fraction_sum (a b c : ℝ) (h : a ≠ 40) (h_a : b ≠ 75) (h_b : c ≠ 85)
  (h_cond : (a / (40 - a)) + (b / (75 - b)) + (c / (85 - c)) = 8) :
  (8 / (40 - a)) + (15 / (75 - b)) + (17 / (85 - c)) = 40 := 
sorry

end evaluate_fraction_sum_l211_211116


namespace fraction_product_l211_211920

theorem fraction_product :
  (7 / 4) * (8 / 14) * (28 / 16) * (24 / 36) * (49 / 35) * (40 / 25) * (63 / 42) * (32 / 48) = 56 / 25 :=
by sorry

end fraction_product_l211_211920


namespace complement_of_beta_l211_211953

theorem complement_of_beta (α β : ℝ) (h₀ : α + β = 180) (h₁ : α > β) : 
  90 - β = 1/2 * (α - β) :=
by
  sorry

end complement_of_beta_l211_211953


namespace f_1991_eq_1988_l211_211752

def f (n : ℕ) : ℕ := sorry

theorem f_1991_eq_1988 : f 1991 = 1988 :=
by sorry

end f_1991_eq_1988_l211_211752


namespace inequality_solution_l211_211506

open Set

noncomputable def solution_set := { x : ℝ | 5 - x^2 > 4 * x }

theorem inequality_solution :
  solution_set = { x : ℝ | -5 < x ∧ x < 1 } :=
by
  sorry

end inequality_solution_l211_211506


namespace coefficient_x2_in_expansion_l211_211156

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Statement to prove the coefficient of the x^2 term in (x + 1)^42 is 861
theorem coefficient_x2_in_expansion :
  (binomial 42 2) = 861 := by
  sorry

end coefficient_x2_in_expansion_l211_211156


namespace map_scale_l211_211400

namespace MapProblem

variables {c1 c2 k1 k2 : ℕ}

def representing_kilometers (c1 c2 k1 k2 : ℕ) : Prop :=
  (c1 * k2 = c2 * k1)

theorem map_scale :
  representing_kilometers 15 20 90 120 :=
by
  unfold representing_kilometers
  calc
    15 * 120 = 1800 := by norm_num
    20 * 90 = 1800 := by norm_num
  sorry

end MapProblem

end map_scale_l211_211400


namespace determine_g_l211_211731

def real_function (g : ℝ → ℝ) :=
  ∀ c d : ℝ, g (c + d) + g (c - d) = g (c) * g (d) + g (d)

def non_zero_function (g : ℝ → ℝ) :=
  ∃ x : ℝ, g x ≠ 0

theorem determine_g (g : ℝ → ℝ) (h1 : real_function g) (h2 : non_zero_function g) : g 0 = 1 ∧ ∀ x : ℝ, g (-x) = g x := 
sorry

end determine_g_l211_211731


namespace percentage_difference_l211_211542

theorem percentage_difference :
  (0.50 * 56 - 0.30 * 50) = 13 := 
by
  -- sorry is used to skip the actual proof steps
  sorry 

end percentage_difference_l211_211542


namespace clark_discount_l211_211080

noncomputable def price_per_part : ℕ := 80
noncomputable def num_parts : ℕ := 7
noncomputable def total_paid : ℕ := 439

theorem clark_discount : (price_per_part * num_parts - total_paid) = 121 :=
by
  -- proof goes here
  sorry

end clark_discount_l211_211080


namespace solve_gcd_problem_l211_211452

def gcd_problem : Prop :=
  gcd 153 119 = 17

theorem solve_gcd_problem : gcd_problem :=
  by
    sorry

end solve_gcd_problem_l211_211452


namespace gravitational_force_solution_l211_211438

noncomputable def gravitational_force_proportionality (d d' : ℕ) (f f' k : ℝ) : Prop :=
  (f * (d:ℝ)^2 = k) ∧
  d = 6000 ∧
  f = 800 ∧
  d' = 36000 ∧
  f' * (d':ℝ)^2 = k

theorem gravitational_force_solution : ∃ k, gravitational_force_proportionality 6000 36000 800 (1/45) k :=
by
  sorry

end gravitational_force_solution_l211_211438


namespace integer_solution_count_l211_211535

theorem integer_solution_count : 
  let condition := ∀ x : ℤ, (x - 2) ^ 2 ≤ 4
  ∃ count : ℕ, count = 5 ∧ (∀ x : ℤ, condition → (0 ≤ x ∧ x ≤ 4)) := 
sorry

end integer_solution_count_l211_211535


namespace geo_seq_fifth_term_l211_211328

theorem geo_seq_fifth_term (a : ℕ → ℝ) (q : ℝ) (h1 : q = 2) (h2 : a 3 = 3) :
  a 5 = 12 := 
sorry

end geo_seq_fifth_term_l211_211328


namespace determine_m_for_unique_solution_l211_211183

-- Define the quadratic equation and the condition for a unique solution
def quadratic_eq_has_one_solution (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c = 0

-- Define the specific quadratic equation and its discriminant
def specific_quadratic_eq (m : ℝ) : Prop :=
  quadratic_eq_has_one_solution 3 (-7) m

-- State the main theorem to prove the value of m
theorem determine_m_for_unique_solution :
  specific_quadratic_eq (49 / 12) :=
by
  unfold specific_quadratic_eq quadratic_eq_has_one_solution
  sorry

end determine_m_for_unique_solution_l211_211183


namespace common_root_for_permutations_of_coeffs_l211_211547

theorem common_root_for_permutations_of_coeffs :
  ∀ (a b c d : ℤ), (a = -7 ∨ a = 4 ∨ a = -3 ∨ a = 6) ∧ 
                   (b = -7 ∨ b = 4 ∨ b = -3 ∨ b = 6) ∧
                   (c = -7 ∨ c = 4 ∨ c = -3 ∨ c = 6) ∧
                   (d = -7 ∨ d = 4 ∨ d = -3 ∨ d = 6) ∧
                   (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
  (a * 1^3 + b * 1^2 + c * 1 + d = 0) :=
by
  intros a b c d h
  sorry

end common_root_for_permutations_of_coeffs_l211_211547


namespace dasha_meeting_sasha_l211_211676

def stripes_on_zebra : ℕ := 360

variables {v : ℝ} -- speed of Masha
def dasha_speed (v : ℝ) : ℝ := 2 * v -- speed of Dasha (twice Masha's speed)

def masha_distance_before_meeting_sasha : ℕ := 180
def total_stripes_met : ℕ := stripes_on_zebra
def relative_speed_masha_sasha (v : ℝ) : ℝ := v + v -- combined speed of Masha and Sasha
def relative_speed_dasha_sasha (v : ℝ) : ℝ := 3 * v -- combined speed of Dasha and Sasha

theorem dasha_meeting_sasha (v : ℝ) (hv : 0 < v) :
  ∃ t' t'', 
  (t'' = 120 / v) ∧ (dasha_speed v * t' = 240) :=
by {
  sorry
}

end dasha_meeting_sasha_l211_211676


namespace mark_donates_1800_cans_l211_211576

variable (number_of_shelters people_per_shelter cans_per_person : ℕ)
variable (total_people total_cans_of_soup : ℕ)

-- Given conditions
def number_of_shelters := 6
def people_per_shelter := 30
def cans_per_person := 10

-- Calculations based on conditions
def total_people := number_of_shelters * people_per_shelter
def total_cans_of_soup := total_people * cans_per_person

-- Proof statement
theorem mark_donates_1800_cans : total_cans_of_soup = 1800 := by
  -- stretch sorry proof placeholder for the proof
  sorry

end mark_donates_1800_cans_l211_211576


namespace two_digit_numbers_tens_greater_ones_l211_211962

theorem two_digit_numbers_tens_greater_ones : 
  ∃ (count : ℕ), count = 45 ∧ ∀ (n : ℕ), 10 ≤ n ∧ n < 100 → 
    let tens := n / 10;
    let ones := n % 10;
    tens > ones → count = 45 :=
by {
  sorry
}

end two_digit_numbers_tens_greater_ones_l211_211962


namespace geometric_sequence_sum_l211_211326

variables (a : ℕ → ℤ) (q : ℤ)

-- assumption that the sequence is geometric
def geometric_sequence (a : ℕ → ℤ) (q : ℤ) : Prop := 
  ∀ n, a (n + 1) = a n * q

noncomputable def a2 := a 2
noncomputable def a3 := a 3
noncomputable def a4 := a 4
noncomputable def a5 := a 5
noncomputable def a6 := a 6
noncomputable def a7 := a 7

theorem geometric_sequence_sum
  (h_geom : geometric_sequence a q)
  (h1 : a2 + a3 = 1)
  (h2 : a3 + a4 = -2) :
  a5 + a6 + a7 = 24 :=
sorry

end geometric_sequence_sum_l211_211326


namespace map_representation_l211_211381

theorem map_representation (d1 d2 l1 l2 : ℕ)
  (h1 : l1 = 15)
  (h2 : d1 = 90)
  (h3 : l2 = 20) :
  d2 = 120 :=
by
  sorry

end map_representation_l211_211381


namespace rational_neither_positive_nor_fraction_l211_211909

def is_rational (q : ℚ) : Prop :=
  q.floor = q

def is_integer (q : ℚ) : Prop :=
  ∃ n : ℤ, q = n

def is_fraction (q : ℚ) : Prop :=
  ∃ p q : ℤ, q ≠ 0 ∧ q = p / q

def is_positive (q : ℚ) : Prop :=
  q > 0

theorem rational_neither_positive_nor_fraction (q : ℚ) :
  (is_rational q) ∧ ¬(is_positive q) ∧ ¬(is_fraction q) ↔
  (is_integer q ∧ q ≤ 0) :=
sorry

end rational_neither_positive_nor_fraction_l211_211909


namespace angle_B_measure_l211_211000

theorem angle_B_measure (a b : ℝ) (A B : ℝ) (h₁ : a = 4) (h₂ : b = 4 * Real.sqrt 3) (h₃ : A = Real.pi / 6) : 
  B = Real.pi / 3 ∨ B = 2 * Real.pi / 3 :=
by
  sorry

end angle_B_measure_l211_211000


namespace diagonals_in_decagon_l211_211534

def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem diagonals_in_decagon : number_of_diagonals 10 = 35 := by
  sorry

end diagonals_in_decagon_l211_211534


namespace number_of_terms_before_4_appears_l211_211140

-- Define the parameters of the arithmetic sequence
def first_term : ℤ := 100
def common_difference : ℤ := -4
def nth_term (n : ℕ) : ℤ := first_term + common_difference * (n - 1)

-- Problem: Prove that the number of terms before the number 4 appears in this sequence is 24.
theorem number_of_terms_before_4_appears :
  ∃ n : ℕ, nth_term n = 4 ∧ n - 1 = 24 := 
by
  sorry

end number_of_terms_before_4_appears_l211_211140


namespace problem1_problem2_problem3_l211_211654

-- First problem
theorem problem1 : 24 - |(-2)| + (-16) - 8 = -2 := by
  sorry

-- Second problem
theorem problem2 : (-2) * (3 / 2) / (-3 / 4) * 4 = 4 := by
  sorry

-- Third problem
theorem problem3 : -1^2016 - (1 - 0.5) / 3 * (2 - (-3)^2) = 1 / 6 := by
  sorry

end problem1_problem2_problem3_l211_211654


namespace sum_of_ages_l211_211857

theorem sum_of_ages {l t : ℕ} (h1 : t > l) (h2 : t * t * l = 72) : t + t + l = 14 :=
sorry

end sum_of_ages_l211_211857


namespace sin_cos_eq_values_l211_211096

theorem sin_cos_eq_values (θ : ℝ) (hθ : 0 < θ ∧ θ ≤ 2 * Real.pi) :
  (∃ t : ℝ, 
    0 < t ∧ 
    t ≤ 2 * Real.pi ∧ 
    (2 + 4 * Real.sin t - 3 * Real.cos (2 * t) = 0)) ↔ (∃ n : ℕ, n = 4) :=
by 
  sorry

end sin_cos_eq_values_l211_211096


namespace find_x_intercept_l211_211507

-- Define the equation of the line
def line_eq (x y : ℝ) : Prop := 4 * x + 7 * y = 28

-- Define the x-intercept point when y = 0
def x_intercept (x : ℝ) : Prop := line_eq x 0

-- Prove that for the x-intercept, when y = 0, x = 7
theorem find_x_intercept : x_intercept 7 :=
by
  -- proof would go here
  sorry

end find_x_intercept_l211_211507


namespace geometric_series_has_value_a_l211_211684

theorem geometric_series_has_value_a (a : ℝ) (S : ℕ → ℝ)
  (h : ∀ n, S (n + 1) = a * (1 / 4) ^ n + 6) :
  a = -3 / 2 :=
sorry

end geometric_series_has_value_a_l211_211684


namespace find_y_intercept_l211_211194

theorem find_y_intercept (m : ℝ) (x_intercept : ℝ × ℝ) (hx : x_intercept = (4, 0)) (hm : m = -3) : ∃ y_intercept : ℝ × ℝ, y_intercept = (0, 12) := 
by
  sorry

end find_y_intercept_l211_211194


namespace range_of_mu_l211_211817

noncomputable def problem_statement (a b μ : ℝ) : Prop :=
  (0 < a) ∧ (0 < b) ∧ (0 < μ) ∧ (1 / a + 9 / b = 1) → (0 < μ ∧ μ ≤ 16)

theorem range_of_mu (a b μ : ℝ) : problem_statement a b μ :=
  sorry

end range_of_mu_l211_211817


namespace recurring_decimal_sum_is_13_over_33_l211_211310

noncomputable def recurring_decimal_sum : ℚ :=
  let x := 1/3 -- 0.\overline{3}
  let y := 2/33 -- 0.\overline{06}
  x + y

theorem recurring_decimal_sum_is_13_over_33 : recurring_decimal_sum = 13/33 := by
  sorry

end recurring_decimal_sum_is_13_over_33_l211_211310


namespace gcd_m_n_is_one_l211_211887

-- Definitions of m and n
def m : ℕ := 101^2 + 203^2 + 307^2
def n : ℕ := 100^2 + 202^2 + 308^2

-- The main theorem stating the gcd of m and n
theorem gcd_m_n_is_one : Nat.gcd m n = 1 := by
  sorry

end gcd_m_n_is_one_l211_211887


namespace divisors_of_30_l211_211701

theorem divisors_of_30 : 
  let positive_divisors := {1, 2, 3, 5, 6, 10, 15, 30}
  in 2 * positive_divisors.card = 16 :=
by
  let positive_divisors := {1, 2, 3, 5, 6, 10, 15, 30}
  have h1 : positive_divisors.card = 8 := sorry
  have h2 : 2 * 8 = 16 := sorry
  exact h2

end divisors_of_30_l211_211701


namespace real_solution_2015x_equation_l211_211660

theorem real_solution_2015x_equation (k : ℝ) :
  (∃ x : ℝ, (4 * 2015^x - 2015^(-x)) / (2015^x - 3 * 2015^(-x)) = k) ↔ (k < 1/3 ∨ k > 4) := 
by sorry

end real_solution_2015x_equation_l211_211660


namespace probability_n_n_plus_1_divisible_13_and_17_l211_211790

theorem probability_n_n_plus_1_divisible_13_and_17 :
  let p : ℚ := 1 / 250 in
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 1000 → Probability (n*(n+1) % 13 = 0 ∧ n*(n+1) % 17 = 0) = p :=
by
  sorry -- proof to be completed

end probability_n_n_plus_1_divisible_13_and_17_l211_211790


namespace UVWXY_perimeter_l211_211720

theorem UVWXY_perimeter (U V W X Y Z : ℝ) 
  (hUV : UV = 5)
  (hVW : VW = 3)
  (hWY : WY = 5)
  (hYX : YX = 9)
  (hXU : XU = 7) :
  UV + VW + WY + YX + XU = 29 :=
by
  sorry

end UVWXY_perimeter_l211_211720


namespace average_of_first_5_multiples_of_5_l211_211886

theorem average_of_first_5_multiples_of_5 : 
  (5 + 10 + 15 + 20 + 25) / 5 = 15 :=
by
  sorry

end average_of_first_5_multiples_of_5_l211_211886


namespace system_of_equations_solution_non_negative_system_of_equations_solution_positive_sum_l211_211811

theorem system_of_equations_solution_non_negative (x y : ℝ) (h1 : x^3 + y^3 + 3 * x * y = 1) (h2 : x^2 - y^2 = 1) (h3 : x ≥ 0) (h4 : y ≥ 0) : x = 1 ∧ y = 0 :=
sorry

theorem system_of_equations_solution_positive_sum (x y : ℝ) (h1 : x^3 + y^3 + 3 * x * y = 1) (h2 : x^2 - y^2 = 1) (h3 : x + y > 0) : x = 1 ∧ y = 0 :=
sorry

end system_of_equations_solution_non_negative_system_of_equations_solution_positive_sum_l211_211811


namespace first_three_digits_of_quotient_are_239_l211_211863

noncomputable def a : ℝ := 0.12345678910114748495051
noncomputable def b_lower_bound : ℝ := 0.515
noncomputable def b_upper_bound : ℝ := 0.516

theorem first_three_digits_of_quotient_are_239 (b : ℝ) (hb : b_lower_bound < b ∧ b < b_upper_bound) :
    0.239 * b < a ∧ a < 0.24 * b := 
sorry

end first_three_digits_of_quotient_are_239_l211_211863


namespace number_of_elements_in_A_is_power_of_2_l211_211357

variables {k : ℕ} {a : Fin k → ℕ} (h : ∀i, a i ∈ {0, 1, 2, 3})

def p (z : ℕ) : ℕ := ∑ i in Finset.range k, a i * (4 ^ i)

def is_base4_expansion (x : ℕ) (l : List ℕ) (k : ℕ) : Prop :=
  ∃ (x' : ℕ), (x' < 4^k) ∧ (x'.digits 4 = l)

def A : Finset ℕ :=
  (Finset.range (4 ^ k)).filter (λ z, p z = z)

theorem number_of_elements_in_A_is_power_of_2 :
  ∃ (n : ℕ), A.card = 2 ^ n :=
sorry

end number_of_elements_in_A_is_power_of_2_l211_211357


namespace total_weight_correct_l211_211304

-- Definitions for the problem conditions
def jug1_capacity : ℝ := 2
def jug2_capacity : ℝ := 3
def jug3_capacity : ℝ := 4

def fill1 : ℝ := 0.7
def fill2 : ℝ := 0.6
def fill3 : ℝ := 0.5

def density1 : ℝ := 5
def density2 : ℝ := 4
def density3 : ℝ := 3

-- The weights of the sand in each jug
def weight1 : ℝ := fill1 * jug1_capacity * density1
def weight2 : ℝ := fill2 * jug2_capacity * density2
def weight3 : ℝ := fill3 * jug3_capacity * density3

-- The total weight of the sand in all jugs
def total_weight : ℝ := weight1 + weight2 + weight3

-- The proof statement
theorem total_weight_correct : total_weight = 20.2 := by
  sorry

end total_weight_correct_l211_211304


namespace sum_of_x_y_l211_211432

theorem sum_of_x_y (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 48) : x + y = 2 :=
sorry

end sum_of_x_y_l211_211432


namespace leo_trousers_count_l211_211995

theorem leo_trousers_count (S T : ℕ) (h1 : 5 * S + 9 * T = 140) (h2 : S = 10) : T = 10 :=
by
  sorry

end leo_trousers_count_l211_211995


namespace probability_all_black_after_rotation_l211_211279

-- Define the conditions
def num_unit_squares : ℕ := 16
def num_colors : ℕ := 3
def prob_per_color : ℚ := 1 / 3

-- Define the type for probabilities
def prob_black_grid : ℚ := (1 / 81) * (11 / 27) ^ 12

-- The statement to be proven
theorem probability_all_black_after_rotation :
  (prob_black_grid =
    ((1 / 3) ^ 4) * ((11 / 27) ^ 12)) :=
sorry

end probability_all_black_after_rotation_l211_211279


namespace screws_per_pile_l211_211858

-- Definitions based on the given conditions
def initial_screws : ℕ := 8
def multiplier : ℕ := 2
def sections : ℕ := 4

-- Derived values based on the conditions
def additional_screws : ℕ := initial_screws * multiplier
def total_screws : ℕ := initial_screws + additional_screws

-- Proposition statement
theorem screws_per_pile : total_screws / sections = 6 := by
  sorry

end screws_per_pile_l211_211858


namespace fraction_of_girls_is_one_third_l211_211211

-- Define the number of children and number of boys
def total_children : Nat := 45
def boys : Nat := 30

-- Calculate the number of girls
def girls : Nat := total_children - boys

-- Calculate the fraction of girls
def fraction_of_girls : Rat := (girls : Rat) / (total_children : Rat)

theorem fraction_of_girls_is_one_third : fraction_of_girls = 1 / 3 :=
by
  sorry -- Proof is not required

end fraction_of_girls_is_one_third_l211_211211


namespace equipment_total_cost_l211_211203

-- Definition of costs for each item of equipment
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.20
def socks_cost : ℝ := 6.80

-- Number of players
def num_players : ℕ := 16

-- Total cost for one player
def total_cost_one_player : ℝ := jersey_cost + shorts_cost + socks_cost

-- Total cost for all players
def total_cost_all_players : ℝ := total_cost_one_player * num_players

-- Theorem to prove
theorem equipment_total_cost : total_cost_all_players = 752 := by
  sorry

end equipment_total_cost_l211_211203


namespace map_length_representation_l211_211407

theorem map_length_representation :
  ∀ (length1 length2 km1 : ℝ), 
  length1 = 15 ∧ km1 = 90 ∧ length2 = 20 →
  (length2 * (km1 / length1) = 120) :=
by
  intros length1 length2 km1
  intro h
  cases h with hl1 h
  cases h with hkm1 hl2
  rw [hl1, hkm1, hl2]
  simp
  sorry

end map_length_representation_l211_211407


namespace smallest_Y_74_l211_211359

def isDigitBin (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d = 0 ∨ d = 1

def smallest_Y (Y : ℕ) : Prop :=
  ∃ T : ℕ, T > 0 ∧ isDigitBin T ∧ T % 15 = 0 ∧ Y = T / 15

theorem smallest_Y_74 : smallest_Y 74 := by
  sorry

end smallest_Y_74_l211_211359


namespace map_distance_to_actual_distance_l211_211592

theorem map_distance_to_actual_distance :
  ∀ (d_map : ℝ) (scale_inch : ℝ) (scale_mile : ℝ), 
    d_map = 15 → scale_inch = 0.25 → scale_mile = 3 →
    (d_map / scale_inch) * scale_mile = 180 :=
by
  intros d_map scale_inch scale_mile h1 h2 h3
  rw [h1, h2, h3]
  sorry

end map_distance_to_actual_distance_l211_211592


namespace rational_numbers_sum_reciprocal_integer_l211_211881

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

end rational_numbers_sum_reciprocal_integer_l211_211881


namespace length_PR_l211_211750

variable (P Q R : Type) [Inhabited P] [Inhabited Q] [Inhabited R]
variable {xPR xQR xsinR : ℝ}
variable (hypotenuse_opposite_ratio : xsinR = (3/5))
variable (sideQR : xQR = 9)
variable (rightAngle : ∀ (P Q R : Type), P ≠ Q → Q ∈ line_through Q R)

theorem length_PR : (∃ xPR : ℝ, xPR = 15) :=
by
  sorry

end length_PR_l211_211750


namespace raven_current_age_l211_211714

variable (R P : ℕ) -- Raven's current age, Phoebe's current age
variable (h₁ : P = 10) -- Phoebe is currently 10 years old
variable (h₂ : R + 5 = 4 * (P + 5)) -- In 5 years, Raven will be 4 times as old as Phoebe

theorem raven_current_age : R = 55 := 
by
  -- h2: R + 5 = 4 * (P + 5)
  -- h1: P = 10
  sorry

end raven_current_age_l211_211714


namespace translation_correct_l211_211827

def vector_a : ℝ × ℝ := (1, 1)

def translate_right (v : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (v.1 + d, v.2)
def translate_down (v : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (v.1, v.2 - d)

def vector_b := translate_down (translate_right vector_a 2) 1

theorem translation_correct :
  vector_b = (3, 0) :=
by
  -- proof steps will go here
  sorry

end translation_correct_l211_211827


namespace sum_of_fifth_powers_divisibility_l211_211987

theorem sum_of_fifth_powers_divisibility (a b c d e : ℤ) :
  (a^5 + b^5 + c^5 + d^5 + e^5) % 25 = 0 → (a % 5 = 0) ∨ (b % 5 = 0) ∨ (c % 5 = 0) ∨ (d % 5 = 0) ∨ (e % 5 = 0) :=
by
  sorry

end sum_of_fifth_powers_divisibility_l211_211987


namespace greatest_prime_factor_of_341_l211_211256

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, nat.prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 341 → q ≤ p :=
begin
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { exact dvd.intro 17 rfl },
  { intros q hq,
    rcases hq with ⟨hq_prime, hq_dvd⟩,
    rw nat.dvd_prime at hq_dvd,
    { cases hq_dvd,
      { exact le_rfl },
      { linarith [hq_prime.ne_zero, nat.prime.ne_one nat.prime_19] } },
    { exact nat.prime_19 },
    { intro h,
      linarith [hq_prime.ne_zero, ne.symm h] } }
end

end greatest_prime_factor_of_341_l211_211256


namespace greatest_prime_factor_of_341_l211_211247

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, is_prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, is_prime q ∧ q ∣ 341 → q ≤ p :=
begin
  -- Our goal is to show that p = 31 meets these conditions.
  sorry
end

end greatest_prime_factor_of_341_l211_211247


namespace min_value_of_function_l211_211189

theorem min_value_of_function : ∃ x : ℝ, ∀ x : ℝ, x * (x + 1) * (x + 2) * (x + 3) ≥ -1 :=
by
  sorry

end min_value_of_function_l211_211189


namespace x_eq_zero_sufficient_not_necessary_l211_211327

theorem x_eq_zero_sufficient_not_necessary (x : ℝ) : 
  (x = 0 → x^2 - 2 * x = 0) ∧ (x^2 - 2 * x = 0 → x = 0 ∨ x = 2) :=
by
  sorry

end x_eq_zero_sufficient_not_necessary_l211_211327


namespace greatest_prime_factor_of_341_is_17_l211_211235

theorem greatest_prime_factor_of_341_is_17 : ∃ p : ℕ, prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, q ∣ 341 → prime q → q ≤ p :=
sorry

end greatest_prime_factor_of_341_is_17_l211_211235


namespace complete_pairs_of_socks_l211_211090

def initial_pairs_blue : ℕ := 20
def initial_pairs_green : ℕ := 15
def initial_pairs_red : ℕ := 15

def lost_socks_blue : ℕ := 3
def lost_socks_green : ℕ := 2
def lost_socks_red : ℕ := 2

def donated_socks_blue : ℕ := 10
def donated_socks_green : ℕ := 15
def donated_socks_red : ℕ := 10

def purchased_pairs_blue : ℕ := 5
def purchased_pairs_green : ℕ := 3
def purchased_pairs_red : ℕ := 2

def gifted_pairs_blue : ℕ := 2
def gifted_pairs_green : ℕ := 1

theorem complete_pairs_of_socks : 
  (initial_pairs_blue - 1 - (donated_socks_blue / 2) + purchased_pairs_blue + gifted_pairs_blue) +
  (initial_pairs_green - 1 - (donated_socks_green / 2) + purchased_pairs_green + gifted_pairs_green) +
  (initial_pairs_red - 1 - (donated_socks_red / 2) + purchased_pairs_red) = 43 := by
  sorry

end complete_pairs_of_socks_l211_211090


namespace ratio_of_linear_combination_l211_211802

theorem ratio_of_linear_combination (a b x y : ℝ) (hb : b ≠ 0) 
  (h1 : 4 * x - 2 * y = a) (h2 : 5 * y - 10 * x = b) :
  a / b = -2 / 5 :=
by {
  sorry
}

end ratio_of_linear_combination_l211_211802


namespace same_speed_4_l211_211851

theorem same_speed_4 {x : ℝ} (hx : x ≠ -7)
  (H1 : ∀ (x : ℝ), (x^2 - 7*x - 60)/(x + 7) = x - 12) 
  (H2 : ∀ (x : ℝ), x^3 - 5*x^2 - 14*x + 104 = x - 12) :
  ∃ (speed : ℝ), speed = 4 :=
by
  sorry

end same_speed_4_l211_211851


namespace SquareArea_l211_211043

theorem SquareArea (s : ℝ) (θ : ℝ) (h1 : s = 3) (h2 : θ = π / 4) : s * s = 9 := 
by 
  sorry

end SquareArea_l211_211043


namespace arithmetic_sequence_solution_l211_211114

theorem arithmetic_sequence_solution (a : ℕ → ℝ) (q : ℝ) (S_n : ℕ → ℝ)
    (h1 : q > 0)
    (h2 : 2 * a 3 = a 5 - 3 * a 4) 
    (h3 : a 2 * a 4 * a 6 = 64) 
    (h4 : ∀ n, S_n n = (1 - q^n) / (1 - q) * a 1) :
    q = 2 ∧ (∀ n, S_n n = (2^n - 1) / 2) := 
  by
  sorry

end arithmetic_sequence_solution_l211_211114


namespace intersection_eq_l211_211115

/-
Define the sets A and B
-/
def setA : Set ℝ := {-1, 0, 1, 2}
def setB : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}

/-
Lean statement to prove the intersection A ∩ B equals {1, 2}
-/
theorem intersection_eq :
  setA ∩ setB = {1, 2} :=
by
  sorry

end intersection_eq_l211_211115


namespace find_a100_l211_211826

noncomputable def S (k : ℝ) (n : ℤ) : ℝ := k * (n ^ 2) + n
noncomputable def a (k : ℝ) (n : ℤ) : ℝ := S k n - S k (n - 1)

theorem find_a100 (k : ℝ) 
  (h1 : a k 10 = 39) :
  a k 100 = 399 :=
sorry

end find_a100_l211_211826


namespace clark_discount_l211_211079

noncomputable def price_per_part : ℕ := 80
noncomputable def num_parts : ℕ := 7
noncomputable def total_paid : ℕ := 439

theorem clark_discount : (price_per_part * num_parts - total_paid) = 121 :=
by
  -- proof goes here
  sorry

end clark_discount_l211_211079


namespace numbers_less_than_reciprocal_l211_211622

theorem numbers_less_than_reciprocal :
  (1 / 3 < 3) ∧ (1 / 2 < 2) ∧ ¬(1 < 1) ∧ ¬(2 < 1 / 2) ∧ ¬(3 < 1 / 3) :=
by
  sorry

end numbers_less_than_reciprocal_l211_211622


namespace sin_2B_sin_A_sin_C_eq_neg_7_over_8_l211_211552

theorem sin_2B_sin_A_sin_C_eq_neg_7_over_8
    (A B C : ℝ)
    (a b c : ℝ)
    (h1 : (2 * a + c) * Real.cos B + b * Real.cos C = 0)
    (h2 : 1/2 * a * c * Real.sin B = 15 * Real.sqrt 3)
    (h3 : a + b + c = 30) :
    (2 * Real.sin B * Real.cos B) / (Real.sin A + Real.sin C) = -7/8 := 
sorry

end sin_2B_sin_A_sin_C_eq_neg_7_over_8_l211_211552


namespace shaded_area_correct_l211_211653

-- Conditions
def side_length_square := 40
def triangle1_base := 15
def triangle1_height := 15
def triangle2_base := 15
def triangle2_height := 15

-- Calculation
def square_area := side_length_square * side_length_square
def triangle1_area := 1 / 2 * triangle1_base * triangle1_height
def triangle2_area := 1 / 2 * triangle2_base * triangle2_height
def total_triangle_area := triangle1_area + triangle2_area
def shaded_region_area := square_area - total_triangle_area

-- Theorem to prove
theorem shaded_area_correct : shaded_region_area = 1375 := by
  sorry

end shaded_area_correct_l211_211653


namespace five_b_value_l211_211344

theorem five_b_value (a b : ℚ) (h1 : 3 * a + 4 * b = 0) (h2 : a = b - 3) : 5 * b = 45 / 7 :=
by
  sorry

end five_b_value_l211_211344


namespace CitadelSchoolEarnings_l211_211793

theorem CitadelSchoolEarnings :
  let apex_students : Nat := 9
  let apex_days : Nat := 5
  let beacon_students : Nat := 3
  let beacon_days : Nat := 4
  let citadel_students : Nat := 6
  let citadel_days : Nat := 7
  let total_payment : ℕ := 864
  let total_student_days : ℕ := (apex_students * apex_days) + (beacon_students * beacon_days) + (citadel_students * citadel_days)
  let daily_wage_per_student : ℚ := total_payment / total_student_days
  let citadel_student_days : ℕ := citadel_students * citadel_days
  let citadel_earnings : ℚ := daily_wage_per_student * citadel_student_days
  citadel_earnings = 366.55 := by
  sorry

end CitadelSchoolEarnings_l211_211793


namespace value_of_n_l211_211055

theorem value_of_n (n : ℝ) (h : n + (n + 1) + (n + 2) + (n + 3) = 20) : n = 3.5 :=
by sorry

end value_of_n_l211_211055


namespace janet_clarinet_hours_l211_211725

theorem janet_clarinet_hours 
  (C : ℕ)  -- number of clarinet lessons hours per week
  (clarinet_cost_per_hour : ℕ := 40)
  (piano_cost_per_hour : ℕ := 28)
  (hours_of_piano_per_week : ℕ := 5)
  (annual_extra_piano_cost : ℕ := 1040) :
  52 * (piano_cost_per_hour * hours_of_piano_per_week - clarinet_cost_per_hour * C) = annual_extra_piano_cost → 
  C = 3 :=
by
  sorry

end janet_clarinet_hours_l211_211725


namespace greatest_prime_factor_341_l211_211232

theorem greatest_prime_factor_341 : ∃ (p : ℕ), prime p ∧ p ∣ 341 ∧ ∀ q, prime q ∧ q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_341_l211_211232


namespace total_savings_calculation_l211_211188

theorem total_savings_calculation
  (income : ℕ)
  (ratio_income_to_expenditure : ℕ)
  (ratio_expenditure_to_income : ℕ)
  (tax_rate : ℚ)
  (investment_rate : ℚ)
  (expenditure : ℕ)
  (taxes : ℚ)
  (investments : ℚ)
  (total_savings : ℚ)
  (h_income : income = 17000)
  (h_ratio : ratio_income_to_expenditure / ratio_expenditure_to_income = 5 / 4)
  (h_tax_rate : tax_rate = 0.15)
  (h_investment_rate : investment_rate = 0.1)
  (h_expenditure : expenditure = (income / 5) * 4)
  (h_taxes : taxes = 0.15 * income)
  (h_investments : investments = 0.1 * income)
  (h_total_savings : total_savings = income - (expenditure + taxes + investments)) :
  total_savings = 900 :=
by
  sorry

end total_savings_calculation_l211_211188


namespace total_people_in_house_l211_211626

-- Definitions of initial condition in the bedroom and living room
def charlie_susan_in_bedroom : ℕ := 2
def sarah_and_friends_in_bedroom : ℕ := 5
def people_in_living_room : ℕ := 8

-- Prove the total number of people in the house is 15
theorem total_people_in_house : charlie_susan_in_bedroom + sarah_and_friends_in_bedroom + people_in_living_room = 15 :=
by
  -- sum the people in the bedroom (Charlie, Susan, Sarah, 4 friends)
  have bedroom_total : charlie_susan_in_bedroom + sarah_and_friends_in_bedroom = 7 := by sorry
  -- sum the people in the house (bedroom + living room)
  show bedroom_total + people_in_living_room = 15 from sorry

end total_people_in_house_l211_211626


namespace percent_divisible_by_6_l211_211264

theorem percent_divisible_by_6 (N : ℕ) (hN : N = 120) :
  (∃ M, M = (finset.univ.filter (λ n : ℕ, n ≤ N ∧ n % 6 = 0)).card ∧ M * 6 = N) →
  (M.to_real / N.to_real) * 100 = 16.66666667 :=
by
  intros h
  sorry

end percent_divisible_by_6_l211_211264


namespace map_scale_l211_211405

namespace MapProblem

variables {c1 c2 k1 k2 : ℕ}

def representing_kilometers (c1 c2 k1 k2 : ℕ) : Prop :=
  (c1 * k2 = c2 * k1)

theorem map_scale :
  representing_kilometers 15 20 90 120 :=
by
  unfold representing_kilometers
  calc
    15 * 120 = 1800 := by norm_num
    20 * 90 = 1800 := by norm_num
  sorry

end MapProblem

end map_scale_l211_211405


namespace functions_equal_l211_211789

noncomputable def f (x : ℝ) : ℝ := x^0
noncomputable def g (x : ℝ) : ℝ := x / x

theorem functions_equal (x : ℝ) (hx : x ≠ 0) : f x = g x :=
by
  unfold f g
  sorry

end functions_equal_l211_211789


namespace gigi_mushrooms_l211_211680

-- Define the conditions
def pieces_per_mushroom := 4
def kenny_pieces := 38
def karla_pieces := 42
def remaining_pieces := 8

-- Main theorem
theorem gigi_mushrooms : (kenny_pieces + karla_pieces + remaining_pieces) / pieces_per_mushroom = 22 :=
by
  sorry

end gigi_mushrooms_l211_211680


namespace anne_distance_l211_211840
  
theorem anne_distance (S T : ℕ) (H1 : S = 2) (H2 : T = 3) : S * T = 6 := by
  -- Given that speed S = 2 miles/hour and time T = 3 hours, we need to show the distance S * T = 6 miles.
  sorry

end anne_distance_l211_211840


namespace divide_80_into_two_parts_l211_211928

theorem divide_80_into_two_parts :
  ∃ a b : ℕ, a + b = 80 ∧ b / 2 = a + 10 ∧ a = 20 ∧ b = 60 :=
by
  sorry

end divide_80_into_two_parts_l211_211928


namespace simultaneous_messengers_l211_211023

theorem simultaneous_messengers (m n : ℕ) (h : m * n = 2010) : 
  m ≠ n → ((m, n) = (1, 2010) ∨ (m, n) = (2, 1005) ∨ (m, n) = (3, 670) ∨ 
          (m, n) = (5, 402) ∨ (m, n) = (6, 335) ∨ (m, n) = (10, 201) ∨ 
          (m, n) = (15, 134) ∨ (m, n) = (30, 67)) :=
sorry

end simultaneous_messengers_l211_211023


namespace map_representation_l211_211382

theorem map_representation (length_1_cm length_2_cm : ℕ) (distance_1_km : ℕ) :
  length_1_cm = 15 → distance_1_km = 90 →
  length_2_cm = 20 →
  distance_2_km = length_2_cm * (distance_1_km / length_1_cm) →
  distance_2_km = 120 :=
by
  intros h_len1 h_distance1 h_len2 h_calculate 
  rw [h_len1, h_distance1, h_len2] at h_calculate
  have scale := 90 / 15
  have distance_2_km := 20 * scale
  have : scale = 6 := by norm_num
  rw [this] at h_calculate
  norm_num at distance_2_km
  assumption
  sorry

end map_representation_l211_211382


namespace map_length_represents_distance_l211_211415

theorem map_length_represents_distance :
  ∀ (s : ℝ), (15 * s = 90) → (20 * s = 120) :=
begin
  intros s h,
  sorry
end

end map_length_represents_distance_l211_211415


namespace equipment_total_cost_l211_211204

-- Definition of costs for each item of equipment
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.20
def socks_cost : ℝ := 6.80

-- Number of players
def num_players : ℕ := 16

-- Total cost for one player
def total_cost_one_player : ℝ := jersey_cost + shorts_cost + socks_cost

-- Total cost for all players
def total_cost_all_players : ℝ := total_cost_one_player * num_players

-- Theorem to prove
theorem equipment_total_cost : total_cost_all_players = 752 := by
  sorry

end equipment_total_cost_l211_211204


namespace sqrt2_minus1_mul_sqrt2_plus1_eq1_l211_211655

theorem sqrt2_minus1_mul_sqrt2_plus1_eq1 : (Real.sqrt 2 - 1) * (Real.sqrt 2 + 1) = 1 :=
  sorry

end sqrt2_minus1_mul_sqrt2_plus1_eq1_l211_211655


namespace course_choice_gender_related_l211_211447
open scoped Real

theorem course_choice_gender_related :
  let a := 40 -- Males choosing Calligraphy
  let b := 10 -- Males choosing Paper Cutting
  let c := 30 -- Females choosing Calligraphy
  let d := 20 -- Females choosing Paper Cutting
  let n := a + b + c + d -- Total number of students
  let χ_squared := (n * (a*d - b*c)^2) / ((a+b) * (c+d) * (a+c) * (b+d))
  χ_squared > 3.841 := 
by
  sorry

end course_choice_gender_related_l211_211447


namespace unique_function_solution_l211_211501

theorem unique_function_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 + f y) = y + f x^2) → (∀ x : ℝ, f x = x) :=
by
  sorry

end unique_function_solution_l211_211501


namespace distinct_real_roots_l211_211525

open Real

theorem distinct_real_roots (n : ℕ) (hn : n > 0) (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (2 * n - 1 < x1 ∧ x1 ≤ 2 * n + 1) ∧ 
  (2 * n - 1 < x2 ∧ x2 ≤ 2 * n + 1) ∧ |x1 - 2 * n| = k ∧ |x2 - 2 * n| = k) ↔ (0 < k ∧ k ≤ 1) :=
by
  sorry

end distinct_real_roots_l211_211525


namespace value_of_v3_at_neg4_l211_211615

def poly (x : ℤ) : ℤ := (((((2 * x + 5) * x + 6) * x + 23) * x - 8) * x + 10) * x - 3

theorem value_of_v3_at_neg4 : poly (-4) = -49 := 
by
  sorry

end value_of_v3_at_neg4_l211_211615


namespace average_age_combined_l211_211021

-- Definitions of the given conditions
def avg_age_fifth_graders := 10
def number_fifth_graders := 40
def avg_age_parents := 40
def number_parents := 60

-- The theorem we need to prove
theorem average_age_combined : 
  (avg_age_fifth_graders * number_fifth_graders + avg_age_parents * number_parents) / (number_fifth_graders + number_parents) = 28 := 
by
  sorry

end average_age_combined_l211_211021


namespace lottery_winning_situations_l211_211717

theorem lottery_winning_situations :
  let num_tickets := 8
  let first_prize := 1
  let second_prize := 1
  let third_prize := 1
  let non_winning := 5
  let customers := 4
  let tickets_per_customer := 2
  let total_ways := 24 + 36
  total_ways = 60 :=
by
  let num_tickets := 8
  let first_prize := 1
  let second_prize := 1
  let third_prize := 1
  let non_winning := 5
  let customers := 4
  let tickets_per_customer := 2
  let total_ways := 24 + 36

  -- Skipping proof steps
  sorry

end lottery_winning_situations_l211_211717


namespace divisors_of_30_count_l211_211702

theorem divisors_of_30_count : 
  ∃ S : Set ℤ, (∀ d ∈ S, 30 % d = 0) ∧ S.card = 16 :=
by
  sorry

end divisors_of_30_count_l211_211702


namespace map_length_representation_l211_211421

theorem map_length_representation (a b : ℕ) (h : a = 15 ∧ b = 90) : b * (20 / a) = 120 :=
by
  sorry

end map_length_representation_l211_211421


namespace triangle_lengths_relationship_l211_211574

-- Given data
variables {a b c f_a f_b f_c t_a t_b t_c : ℝ}
-- Conditions/assumptions
variables (h1 : f_a * t_a = b * c)
variables (h2 : f_b * t_b = a * c)
variables (h3 : f_c * t_c = a * b)

-- Theorem to prove
theorem triangle_lengths_relationship :
  a^2 * b^2 * c^2 = f_a * f_b * f_c * t_a * t_b * t_c :=
by sorry

end triangle_lengths_relationship_l211_211574


namespace verify_value_of_2a10_minus_a12_l211_211718

-- Define the arithmetic sequence and the sum condition
variable {a : ℕ → ℝ}  -- arithmetic sequence
variable {a1 : ℝ}     -- the first term of the sequence
variable {d : ℝ}      -- the common difference of the sequence

-- Assume that the sequence is arithmetic
def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n, a n = a1 + n * d

-- Assume the sum condition
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

-- The goal is to prove that 2 * a 10 - a 12 = 24
theorem verify_value_of_2a10_minus_a12 (h_arith : arithmetic_sequence a a1 d) (h_sum : sum_condition a) :
  2 * a 10 - a 12 = 24 :=
  sorry

end verify_value_of_2a10_minus_a12_l211_211718


namespace jo_age_l211_211433

theorem jo_age (j d g : ℕ) (even_j : 2 * j = j * 2) (even_d : 2 * d = d * 2) (even_g : 2 * g = g * 2)
    (h : 8 * j * d * g = 2024) : 2 * j = 46 :=
sorry

end jo_age_l211_211433


namespace minimum_additional_marbles_l211_211448

-- Definitions corresponding to the conditions
def friends := 12
def initial_marbles := 40

-- Sum of the first n natural numbers definition
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Prove the necessary number of additional marbles
theorem minimum_additional_marbles (h1 : friends = 12) (h2 : initial_marbles = 40) : 
  ∃ additional_marbles, additional_marbles = sum_first_n friends - initial_marbles := by
  sorry

end minimum_additional_marbles_l211_211448


namespace trig_sum_identity_l211_211945

theorem trig_sum_identity :
  Real.sin (20 * Real.pi / 180) + Real.sin (40 * Real.pi / 180) +
  Real.sin (60 * Real.pi / 180) - Real.sin (80 * Real.pi / 180) = Real.sqrt 3 / 2 := 
sorry

end trig_sum_identity_l211_211945


namespace map_scale_l211_211396

theorem map_scale (represents_15cm_as_km : 15 * 6 = 90) :
  20 * 6 = 120 :=
by
  sorry

end map_scale_l211_211396


namespace max_abc_l211_211368

theorem max_abc : ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  (a * b + b * c = 518) ∧ 
  (a * b - a * c = 360) ∧ 
  (a * b * c = 1008) := 
by {
  -- Definitions of a, b, c satisfying the given conditions.
  -- Proof of the maximum value will be placed here (not required as per instructions).
  sorry
}

end max_abc_l211_211368


namespace map_scale_l211_211404

namespace MapProblem

variables {c1 c2 k1 k2 : ℕ}

def representing_kilometers (c1 c2 k1 k2 : ℕ) : Prop :=
  (c1 * k2 = c2 * k1)

theorem map_scale :
  representing_kilometers 15 20 90 120 :=
by
  unfold representing_kilometers
  calc
    15 * 120 = 1800 := by norm_num
    20 * 90 = 1800 := by norm_num
  sorry

end MapProblem

end map_scale_l211_211404


namespace map_length_represents_distance_l211_211413

theorem map_length_represents_distance :
  ∀ (s : ℝ), (15 * s = 90) → (20 * s = 120) :=
begin
  intros s h,
  sorry
end

end map_length_represents_distance_l211_211413


namespace prob_max_atleast_twice_chloe_l211_211921

section
variables {Ω : Type*} [ProbabilitySpace Ω]

noncomputable def probability_favor : ℝ :=
  let A := set.prod (set.Icc (0 : ℝ) 3000) (set.Icc (0 : ℝ) 4500) in
  let B := {xy : ℝ × ℝ | xy.2 >= 2 * xy.1} in
  (Prob (A ∩ B) / Prob A)

theorem prob_max_atleast_twice_chloe : probability_favor = (3 / 8) :=
sorry
end

end prob_max_atleast_twice_chloe_l211_211921


namespace unique_integer_m_l211_211508

theorem unique_integer_m :
  ∃! (m : ℤ), m - ⌊m / (2005 : ℝ)⌋ = 2005 :=
by
  --- Here belongs the proof part, but we leave it with a sorry
  sorry

end unique_integer_m_l211_211508


namespace greatest_diff_l211_211271

theorem greatest_diff (x y : ℤ) (hx1 : 6 < x) (hx2 : x < 10) (hy1 : 10 < y) (hy2 : y < 17) : y - x = 7 :=
sorry

end greatest_diff_l211_211271


namespace clothing_store_earnings_l211_211638

-- Defining the conditions
def num_shirts : ℕ := 20
def num_jeans : ℕ := 10
def shirt_cost : ℕ := 10
def jeans_cost : ℕ := 2 * shirt_cost

-- Statement of the problem
theorem clothing_store_earnings :
  num_shirts * shirt_cost + num_jeans * jeans_cost = 400 :=
by
  sorry

end clothing_store_earnings_l211_211638


namespace cameron_gold_tokens_l211_211498

/-- Cameron starts with 90 red tokens and 60 blue tokens. 
  Booth 1 exchange: 3 red tokens for 1 gold token and 2 blue tokens.
  Booth 2 exchange: 2 blue tokens for 1 gold token and 1 red token.
  Cameron stops when fewer than 3 red tokens or 2 blue tokens remain.
  Prove that the number of gold tokens Cameron ends up with is 148.
-/
theorem cameron_gold_tokens :
  ∃ (x y : ℕ), 
    90 - 3 * x + y < 3 ∧
    60 + 2 * x - 2 * y < 2 ∧
    (x + y = 148) :=
  sorry

end cameron_gold_tokens_l211_211498


namespace harry_terry_difference_l211_211834

theorem harry_terry_difference :
  let H := 8 - (2 + 5)
  let T := 8 - 2 + 5
  H - T = -10 :=
by 
  sorry

end harry_terry_difference_l211_211834


namespace am_gm_equality_l211_211570

theorem am_gm_equality (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end am_gm_equality_l211_211570


namespace income_left_at_end_of_year_l211_211065

variable (I : ℝ) -- Monthly income at the beginning of the year
variable (food_expense : ℝ := 0.35 * I) 
variable (education_expense : ℝ := 0.25 * I)
variable (transportation_expense : ℝ := 0.15 * I)
variable (medical_expense : ℝ := 0.10 * I)
variable (initial_expenses : ℝ := food_expense + education_expense + transportation_expense + medical_expense)
variable (remaining_income : ℝ := I - initial_expenses)
variable (house_rent : ℝ := 0.80 * remaining_income)

variable (annual_income : ℝ := 12 * I)
variable (annual_expenses : ℝ := 12 * (initial_expenses + house_rent))

variable (increased_food_expense : ℝ := food_expense * 1.05)
variable (increased_education_expense : ℝ := education_expense * 1.05)
variable (increased_transportation_expense : ℝ := transportation_expense * 1.05)
variable (increased_medical_expense : ℝ := medical_expense * 1.05)
variable (total_increased_expenses : ℝ := increased_food_expense + increased_education_expense + increased_transportation_expense + increased_medical_expense)

variable (new_income : ℝ := 1.10 * I)
variable (new_remaining_income : ℝ := new_income - total_increased_expenses)

variable (new_house_rent : ℝ := 0.80 * new_remaining_income)

variable (final_remaining_income : ℝ := new_income - (total_increased_expenses + new_house_rent))

theorem income_left_at_end_of_year : 
  final_remaining_income / new_income * 100 = 2.15 := 
  sorry

end income_left_at_end_of_year_l211_211065


namespace nonnegative_solution_count_nonnegative_solution_count_unique_l211_211134

theorem nonnegative_solution_count (x : ℝ) :
  (x^2 = -6 * x) → (x ≥ 0) → x = 0 :=
by
  sorry

-- Prove that the number of nonnegative solutions is 1
theorem nonnegative_solution_count_unique :
  {x : ℝ // x ≥ 0 ∧ x^2 = -6 * x}.to_finset.card = 1 :=
by
  sorry

end nonnegative_solution_count_nonnegative_solution_count_unique_l211_211134


namespace no_valid_n_l211_211946

theorem no_valid_n : ¬ ∃ (n : ℕ), (n > 0) ∧ (100 ≤ n / 4) ∧ (n / 4 ≤ 999) ∧ (100 ≤ 4 * n) ∧ (4 * n ≤ 999) :=
by {
  sorry
}

end no_valid_n_l211_211946


namespace expected_value_red_balls_draws_l211_211019

/-- Let there be a bag containing 4 red balls and 2 white balls,
all of the same size and texture. 
If balls are drawn one after another with replacement from the bag,
and the number of times a red ball is drawn in 6 draws is denoted by 
ξ, then the expected value E(ξ) is 4. -/
theorem expected_value_red_balls_draws :
  let p := 2 / 3 in
  let n := 6 in
  let ξ : ℕ → ℕ := λ k, k  in
  n * p = 4 := 
by
  -- Proof steps would go here
  sorry

end expected_value_red_balls_draws_l211_211019


namespace rational_ordering_l211_211657

theorem rational_ordering :
  (-3:ℚ)^2 < -1/3 ∧ (-1/3 < ((-3):ℚ)^2 ∧ ((-3:ℚ)^2 = |((-3:ℚ))^2|)) := 
by 
  sorry

end rational_ordering_l211_211657


namespace base_six_equals_base_b_l211_211434

theorem base_six_equals_base_b (b : ℕ) (h1 : 3 * 6 ^ 1 + 4 * 6 ^ 0 = 22)
  (h2 : b ^ 2 + 2 * b + 1 = 22) : b = 3 :=
sorry

end base_six_equals_base_b_l211_211434


namespace mingi_initial_tomatoes_l211_211736

theorem mingi_initial_tomatoes (n m r : ℕ) (h1 : n = 15) (h2 : m = 20) (h3 : r = 6) : n * m + r = 306 := by
  sorry

end mingi_initial_tomatoes_l211_211736


namespace g_is_even_function_l211_211985

def g (x : ℝ) : ℝ := 4 / (3 * x^8 - 7)

theorem g_is_even_function : ∀ x : ℝ, g (-x) = g x :=
by
  intro x
  rw [g, g]
  sorry

end g_is_even_function_l211_211985


namespace fraction_of_yard_occupied_l211_211487

noncomputable def area_triangle_flower_bed : ℝ := 
  2 * (0.5 * (10:ℝ) * (10:ℝ))

noncomputable def area_circular_flower_bed : ℝ := 
  Real.pi * (2:ℝ)^2

noncomputable def total_area_flower_beds : ℝ := 
  area_triangle_flower_bed + area_circular_flower_bed

noncomputable def area_yard : ℝ := 
  (40:ℝ) * (10:ℝ)

noncomputable def fraction_occupied := 
  total_area_flower_beds / area_yard

theorem fraction_of_yard_occupied : 
  fraction_occupied = 0.2814 := 
sorry

end fraction_of_yard_occupied_l211_211487


namespace total_equipment_cost_l211_211202

-- Define the cost of each piece of equipment
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.2
def socks_cost : ℝ := 6.8

-- Define the number of players
def players : ℕ := 16

-- Define the total cost of equipment for one player
def equipment_cost_per_player : ℝ := jersey_cost + shorts_cost + socks_cost

-- Define the total cost for all players
def total_cost : ℝ := players * equipment_cost_per_player

-- The proof problem to be stated:
theorem total_equipment_cost (jc sc k p : ℝ) (n : ℕ) :
  jc = 25 ∧ sc = 15.2 ∧ k = 6.8 ∧ p = 16 →
  total_cost = 752 :=
by
  intro h
  rcases h with ⟨hc1, hc2, hc3, hc4⟩
  simp [total_cost, equipment_cost_per_player, hc1, hc2, hc3, hc4]
  exact sorry

end total_equipment_cost_l211_211202


namespace solve_for_n_l211_211052

theorem solve_for_n (n : ℚ) (h : n + (n + 1) + (n + 2) + (n + 3) = 20) : 
    n = 3.5 :=
  sorry

end solve_for_n_l211_211052


namespace range_of_a_for_increasing_l211_211334

noncomputable def f (a : ℝ) : (ℝ → ℝ) := λ x => x^3 + a * x^2 + 3 * x

theorem range_of_a_for_increasing (a : ℝ) : 
  (∀ x : ℝ, (3 * x^2 + 2 * a * x + 3) ≥ 0) ↔ (-3 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_of_a_for_increasing_l211_211334


namespace map_scale_l211_211403

namespace MapProblem

variables {c1 c2 k1 k2 : ℕ}

def representing_kilometers (c1 c2 k1 k2 : ℕ) : Prop :=
  (c1 * k2 = c2 * k1)

theorem map_scale :
  representing_kilometers 15 20 90 120 :=
by
  unfold representing_kilometers
  calc
    15 * 120 = 1800 := by norm_num
    20 * 90 = 1800 := by norm_num
  sorry

end MapProblem

end map_scale_l211_211403


namespace bottle_caps_remaining_l211_211012

-- Define the problem using the conditions and the desired proof.
theorem bottle_caps_remaining (original_count removed_count remaining_count : ℕ) 
    (h_original : original_count = 87) 
    (h_removed : removed_count = 47)
    (h_remaining : remaining_count = original_count - removed_count) :
    remaining_count = 40 :=
by 
  rw [h_original, h_removed] at h_remaining 
  exact h_remaining

end bottle_caps_remaining_l211_211012


namespace simplify_expression_l211_211748

theorem simplify_expression (x : ℝ) (h : x^2 - x - 1 = 0) :
  ( ( (x - 1) / x - (x - 2) / (x + 1) ) / ( (2 * x^2 - x) / (x^2 + 2 * x + 1) ) ) = 1 := 
by
  sorry

end simplify_expression_l211_211748


namespace map_distance_l211_211370

theorem map_distance (fifteen_cm_in_km : ℤ) (cm_to_km : ℕ): 
  fifteen_cm_in_km = 90 ∧ cm_to_km = 6 → 20 * cm_to_km = 120 := 
by 
  sorry

end map_distance_l211_211370


namespace probability_LM_accurate_l211_211753

noncomputable def defect_probability {K L M : Type} 
  (def_k : ℝ) (def_l : ℝ) (def_m : ℝ) 
  (two_defective : ℝ)
  (defective_lm : ℝ) : Prop :=
  def_k = 0.1 ∧ def_l = 0.2 ∧ def_m = 0.15 ∧ 
  two_defective = 0.056 ∧ 
  defective_lm = 0.4821

theorem probability_LM_accurate : 
  ∃ (K L M : Type), ∀ (def_k def_l def_m two_defective defective_lm : ℝ),
    defect_probability def_k def_l def_m two_defective defective_lm → 
    defective_lm ≈ 0.4821 :=
by 
  sorry

end probability_LM_accurate_l211_211753


namespace hyperbola_foci_distance_l211_211504

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := x^2 - 4 * x - 9 * y^2 - 18 * y = 56

-- Define the distance between the foci of the hyperbola
def distance_between_foci (d : ℝ) : Prop :=
  d = 2 * Real.sqrt (170 / 3)

-- The theorem stating that the distance between the foci of the given hyperbola
theorem hyperbola_foci_distance :
  ∃ d, hyperbola_eq x y → distance_between_foci d :=
by { sorry }

end hyperbola_foci_distance_l211_211504


namespace jack_should_leave_300_in_till_l211_211850

-- Defining the amounts of each type of bill
def num_100_bills := 2
def num_50_bills := 1
def num_20_bills := 5
def num_10_bills := 3
def num_5_bills := 7
def num_1_bills := 27

-- The amount he needs to hand in
def amount_to_hand_in := 142

-- Calculating the total amount in notes
def total_in_notes := 
  (num_100_bills * 100) + 
  (num_50_bills * 50) + 
  (num_20_bills * 20) + 
  (num_10_bills * 10) + 
  (num_5_bills * 5) + 
  (num_1_bills * 1)

-- Calculating the amount to leave in the till
def amount_to_leave := total_in_notes - amount_to_hand_in

-- Proof statement
theorem jack_should_leave_300_in_till :
  amount_to_leave = 300 :=
by sorry

end jack_should_leave_300_in_till_l211_211850


namespace consecutive_even_product_6digit_l211_211936

theorem consecutive_even_product_6digit :
  ∃ (a b c : ℕ), 
  (a % 2 = 0) ∧ (b = a + 2) ∧ (c = a + 4) ∧ 
  (Nat.digits 10 (a * b * c)).length = 6 ∧ 
  (Nat.digits 10 (a * b * c)).head! = 2 ∧ 
  (Nat.digits 10 (a * b * c)).getLast! = 2 ∧ 
  (a * b * c = 287232) :=
by
  sorry

end consecutive_even_product_6digit_l211_211936


namespace range_of_inverse_proportion_l211_211696

theorem range_of_inverse_proportion (x : ℝ) (h : 3 < x) :
    -1 < -3 / x ∧ -3 / x < 0 :=
by
  sorry

end range_of_inverse_proportion_l211_211696


namespace exponent_identity_l211_211633

theorem exponent_identity (m : ℕ) : 5 ^ m = 5 * (25 ^ 4) * (625 ^ 3) ↔ m = 21 := by
  sorry

end exponent_identity_l211_211633


namespace correct_calculation_l211_211057

theorem correct_calculation :
  (3 * Real.sqrt 2) * (2 * Real.sqrt 3) = 6 * Real.sqrt 6 :=
by sorry

end correct_calculation_l211_211057


namespace simplify_trig_identity_l211_211180

theorem simplify_trig_identity (α β : ℝ) : 
  (Real.cos (α + β) * Real.cos β + Real.sin (α + β) * Real.sin β) = Real.cos α :=
by
  sorry

end simplify_trig_identity_l211_211180


namespace ratio_M_N_l211_211341

theorem ratio_M_N (P Q M N : ℝ) (h1 : M = 0.30 * Q) (h2 : Q = 0.20 * P) (h3 : N = 0.50 * P) (hP_nonzero : P ≠ 0) :
  M / N = 3 / 25 := 
by 
  sorry

end ratio_M_N_l211_211341


namespace find_a_l211_211335

noncomputable def f (x : ℝ) : ℝ := x + 1 + Real.log x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

theorem find_a (a : ℝ) : 
  (∃ x : ℝ, g a x = 2 * x) ∧ (deriv f 1 = 2) ∧ f 1 = 2 → a = 4 :=
by
  -- Math proof goes here
  sorry

end find_a_l211_211335


namespace hidden_dots_sum_l211_211028

-- Lean 4 equivalent proof problem definition
theorem hidden_dots_sum (d1 d2 d3 d4 : ℕ)
    (h1 : d1 ≠ d2 ∧ d1 + d2 = 7)
    (h2 : d3 ≠ d4 ∧ d3 + d4 = 7)
    (h3 : d1 = 2 ∨ d1 = 4 ∨ d2 = 2 ∨ d2 = 4)
    (h4 : d3 + 4 = 7) :
    d1 + 7 + 7 + d3 = 24 :=
sorry

end hidden_dots_sum_l211_211028


namespace quadrilateral_possible_with_2_2_2_l211_211058

theorem quadrilateral_possible_with_2_2_2 :
  ∀ (s1 s2 s3 s4 : ℕ), (s1 = 2) → (s2 = 2) → (s3 = 2) → (s4 = 5) →
  s1 + s2 + s3 > s4 :=
by
  intros s1 s2 s3 s4 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- Proof omitted
  sorry

end quadrilateral_possible_with_2_2_2_l211_211058


namespace f_2015_2016_l211_211117

theorem f_2015_2016 (f : ℤ → ℤ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_periodic : ∀ x, f (x + 2) = -f x)
  (h_f1 : f 1 = 2) :
  f 2015 + f 2016 = -2 :=
sorry

end f_2015_2016_l211_211117


namespace negation_of_implication_l211_211439

theorem negation_of_implication (x : ℝ) :
  ¬ (x ≠ 3 ∧ x ≠ 2 → x^2 - 5 * x + 6 ≠ 0) ↔ (x = 3 ∨ x = 2 → x^2 - 5 * x + 6 = 0) := 
by {
  sorry
}

end negation_of_implication_l211_211439


namespace nonnegative_solutions_eq1_l211_211136

theorem nonnegative_solutions_eq1 : (∃ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x) ∧ (∀ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x → x = 0) := by
  sorry

end nonnegative_solutions_eq1_l211_211136


namespace total_people_in_house_l211_211627

-- Define the number of people in various locations based on the given conditions.
def charlie_and_susan := 2
def sarah_and_friends := 5
def people_in_bedroom := charlie_and_susan + sarah_and_friends
def people_in_living_room := 8

-- Prove the total number of people in the house is 14.
theorem total_people_in_house : people_in_bedroom + people_in_living_room = 14 := by
  -- Here we can use Lean's proof system, but we skip with 'sorry'
  sorry

end total_people_in_house_l211_211627


namespace degenerate_ellipse_single_point_c_l211_211185

theorem degenerate_ellipse_single_point_c (c : ℝ) :
  (∀ x y : ℝ, 2 * x^2 + y^2 + 8 * x - 10 * y + c = 0 → x = -2 ∧ y = 5) →
  c = 33 :=
by
  intros h
  sorry

end degenerate_ellipse_single_point_c_l211_211185


namespace max_gcd_value_l211_211916

theorem max_gcd_value (n : ℕ) (hn : 0 < n) : ∃ k, k = gcd (13 * n + 4) (8 * n + 3) ∧ k <= 7 := sorry

end max_gcd_value_l211_211916


namespace new_average_weight_is_27_3_l211_211590

-- Define the given conditions as variables/constants in Lean
noncomputable def original_students : ℕ := 29
noncomputable def original_average_weight : ℝ := 28
noncomputable def new_student_weight : ℝ := 7

-- The total weight of the original students
noncomputable def original_total_weight : ℝ := original_students * original_average_weight
-- The new total number of students
noncomputable def new_total_students : ℕ := original_students + 1
-- The new total weight after new student is added
noncomputable def new_total_weight : ℝ := original_total_weight + new_student_weight

-- The theorem to prove that the new average weight is 27.3 kg
theorem new_average_weight_is_27_3 : (new_total_weight / new_total_students) = 27.3 := 
by
  sorry -- The proof will be provided here

end new_average_weight_is_27_3_l211_211590


namespace sandy_savings_percentage_l211_211356

theorem sandy_savings_percentage
  (S : ℝ) -- Sandy's salary last year
  (H1 : 0.10 * S = saved_last_year) -- Last year, Sandy saved 10% of her salary.
  (H2 : 1.10 * S = salary_this_year) -- This year, Sandy made 10% more than last year.
  (H3 : 0.15 * salary_this_year = saved_this_year) -- This year, Sandy saved 15% of her salary.
  : (saved_this_year / saved_last_year) * 100 = 165 := 
by 
  sorry

end sandy_savings_percentage_l211_211356


namespace mul_example_l211_211668

theorem mul_example : (3.6 * 0.5 = 1.8) := by
  sorry

end mul_example_l211_211668


namespace lesser_of_two_numbers_l211_211880

theorem lesser_of_two_numbers (x y : ℝ) (h1 : x + y = 70) (h2 : x * y = 1050) : min x y = 30 :=
sorry

end lesser_of_two_numbers_l211_211880


namespace range_of_k_l211_211970

noncomputable def quadratic_inequality (k : ℝ) := 
  ∀ x : ℝ, 2 * k * x^2 + k * x - (3 / 8) < 0

theorem range_of_k (k : ℝ) :
  (quadratic_inequality k) → -3 < k ∧ k < 0 := sorry

end range_of_k_l211_211970


namespace find_number_l211_211221

theorem find_number (x : ℝ) (h : 20 / x = 0.8) : x = 25 := 
by
  sorry

end find_number_l211_211221


namespace tank_empty_time_l211_211907

def tank_capacity : ℝ := 6480
def leak_time : ℝ := 6
def inlet_rate_per_minute : ℝ := 4.5
def inlet_rate_per_hour : ℝ := inlet_rate_per_minute * 60

theorem tank_empty_time : tank_capacity / (tank_capacity / leak_time - inlet_rate_per_hour) = 8 := 
by
  sorry

end tank_empty_time_l211_211907


namespace units_digit_of_square_l211_211996

theorem units_digit_of_square (n : ℤ) (h : (n^2 / 10) % 10 = 7) : (n^2 % 10) = 6 := 
by 
  sorry

end units_digit_of_square_l211_211996


namespace map_scale_representation_l211_211428

theorem map_scale_representation :
  (∀ len_km_1 len_cm_1 len_cm_2 : ℕ, len_km_1 = 90 → len_cm_1 = 15 → len_cm_2 = 20 →
    let scale := len_km_1 / len_cm_1 in
    len_cm_2 * scale = 120)
  := sorry

end map_scale_representation_l211_211428


namespace trig_fraction_identity_l211_211743

noncomputable def cos_63 := Real.cos (Real.pi * 63 / 180)
noncomputable def cos_3 := Real.cos (Real.pi * 3 / 180)
noncomputable def cos_87 := Real.cos (Real.pi * 87 / 180)
noncomputable def cos_27 := Real.cos (Real.pi * 27 / 180)
noncomputable def cos_132 := Real.cos (Real.pi * 132 / 180)
noncomputable def cos_72 := Real.cos (Real.pi * 72 / 180)
noncomputable def cos_42 := Real.cos (Real.pi * 42 / 180)
noncomputable def cos_18 := Real.cos (Real.pi * 18 / 180)
noncomputable def tan_24 := Real.tan (Real.pi * 24 / 180)

theorem trig_fraction_identity :
  (cos_63 * cos_3 - cos_87 * cos_27) / 
  (cos_132 * cos_72 - cos_42 * cos_18) = 
  -tan_24 := 
by
  sorry

end trig_fraction_identity_l211_211743


namespace origami_papers_per_cousin_l211_211891

theorem origami_papers_per_cousin (total_papers : ℕ) (num_cousins : ℕ) (same_papers_each : ℕ) 
  (h1 : total_papers = 48) 
  (h2 : num_cousins = 6) 
  (h3 : same_papers_each = total_papers / num_cousins) : 
  same_papers_each = 8 := 
by 
  sorry

end origami_papers_per_cousin_l211_211891


namespace daily_evaporation_l211_211902

variable (initial_water : ℝ) (percentage_evaporated : ℝ) (days : ℕ)
variable (evaporation_amount : ℝ)

-- Given conditions
def conditions_met : Prop :=
  initial_water = 10 ∧ percentage_evaporated = 0.4 ∧ days = 50

-- Question: Prove the amount of water evaporated each day is 0.08
theorem daily_evaporation (h : conditions_met initial_water percentage_evaporated days) :
  evaporation_amount = (initial_water * percentage_evaporated) / days :=
sorry

end daily_evaporation_l211_211902


namespace nonneg_solutions_count_l211_211133

theorem nonneg_solutions_count :
  {x : ℝ | x^2 + 6 * x = 0 ∧ 0 ≤ x}.card = 1 :=
by
  sorry

end nonneg_solutions_count_l211_211133


namespace beavers_fraction_l211_211904

theorem beavers_fraction (total_beavers : ℕ) (swim_percentage : ℕ) (work_percentage : ℕ) (fraction_working : ℕ) : 
total_beavers = 4 → 
swim_percentage = 75 → 
work_percentage = 100 - swim_percentage → 
fraction_working = 1 →
(work_percentage * total_beavers) / 100 = fraction_working → 
fraction_working / total_beavers = 1 / 4 :=
by 
  intros h1 h2 h3 h4 h5 
  sorry

end beavers_fraction_l211_211904


namespace sum_of_interior_diagonals_of_box_l211_211286

theorem sum_of_interior_diagonals_of_box (a b c : ℝ) 
  (h_edges : 4 * (a + b + c) = 60)
  (h_surface_area : 2 * (a * b + b * c + c * a) = 150) : 
  4 * Real.sqrt (a^2 + b^2 + c^2) = 20 * Real.sqrt 3 := 
by
  sorry

end sum_of_interior_diagonals_of_box_l211_211286


namespace greatest_prime_factor_341_l211_211236

theorem greatest_prime_factor_341 : ∃ p : ℕ, Nat.Prime p ∧ p = 17 ∧ p = Nat.gcd 341 (Nat.gcd 341 (Nat.gcd 341 341)) :=
by
  sorry

end greatest_prime_factor_341_l211_211236


namespace sum_first_7_l211_211119

variable {α : Type*} [LinearOrderedField α]

-- Definitions for the arithmetic sequence
noncomputable def arithmetic_sequence (a d : α) (n : ℕ) : α :=
  a + d * (n - 1)

noncomputable def sum_of_first_n_terms (a d : α) (n : ℕ) : α :=
  n * (2 * a + (n - 1) * d) / 2

-- Conditions
variable {a d : α} -- Initial term and common difference of the arithmetic sequence
variable (h : arithmetic_sequence a d 2 + arithmetic_sequence a d 4 + arithmetic_sequence a d 6 = 12)

-- Proof statement
theorem sum_first_7 (a d : α) (h : arithmetic_sequence a d 2 + arithmetic_sequence a d 4 + arithmetic_sequence a d 6 = 12) : 
  sum_of_first_n_terms a d 7 = 28 := 
by 
  sorry

end sum_first_7_l211_211119


namespace find_x_l211_211546

theorem find_x (x : ℝ) (h : 9 / (x + 4) = 1) : x = 5 :=
sorry

end find_x_l211_211546


namespace six_digit_product_of_consecutive_even_integers_l211_211943

theorem six_digit_product_of_consecutive_even_integers :
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ (a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0) ∧ a * b * c = 287232 :=
sorry

end six_digit_product_of_consecutive_even_integers_l211_211943


namespace standard_eq_of_parabola_l211_211606

-- Conditions:
-- The point (1, -2) lies on the parabola.
def point_on_parabola : Prop := ∃ p : ℝ, (1, -2).2^2 = 2 * p * (1, -2).1 ∨ (1, -2).1^2 = 2 * p * (1, -2).2

-- Question to be proved:
-- The standard equation of the parabola passing through the point (1, -2) is y^2 = 4x or x^2 = - (1/2) y.
theorem standard_eq_of_parabola : point_on_parabola → (y^2 = 4*x ∨ x^2 = -(1/(2:ℝ)) * y) :=
by
  sorry -- proof to be provided

end standard_eq_of_parabola_l211_211606


namespace necessary_but_not_sufficient_l211_211954

theorem necessary_but_not_sufficient (a b : ℝ) : 
  (b < -1 → |a| + |b| > 1) ∧ (∃ a b : ℝ, |a| + |b| > 1 ∧ b >= -1) :=
by
  sorry

end necessary_but_not_sufficient_l211_211954


namespace police_arrangements_l211_211500

theorem police_arrangements (officers : Fin 5) (A B : Fin 5) (intersections : Fin 3) :
  A ≠ B →
  (∃ arrangement : Fin 5 → Fin 3, (∀ i j : Fin 3, i ≠ j → ∃ off : Fin 5, arrangement off = i ∧ arrangement off = j) ∧
    arrangement A = arrangement B) →
  ∃ arrangements_count : Nat, arrangements_count = 36 :=
by
  sorry

end police_arrangements_l211_211500


namespace monotonically_increasing_interval_l211_211830

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem monotonically_increasing_interval :
  ∀ x, 0 < x ∧ x ≤ π / 6 → ∀ y, x ≤ y ∧ y < π / 2 → f x ≤ f y :=
by
  intro x hx y hy
  sorry

end monotonically_increasing_interval_l211_211830


namespace shopkeeper_marked_price_l211_211488

theorem shopkeeper_marked_price 
  (L C M S : ℝ)
  (h1 : C = 0.75 * L)
  (h2 : C = 0.75 * S)
  (h3 : S = 0.85 * M) :
  M = 1.17647 * L :=
sorry

end shopkeeper_marked_price_l211_211488


namespace average_of_remaining_two_numbers_l211_211751

theorem average_of_remaining_two_numbers (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) / 6 = 3.95)
  (h2 : (a + b) / 2 = 3.4)
  (h3 : (c + d) / 2 = 3.85) :
  (e + f) / 2 = 4.6 :=
sorry

end average_of_remaining_two_numbers_l211_211751


namespace subtracting_is_adding_opposite_l211_211771

theorem subtracting_is_adding_opposite (a b : ℚ) : a - b = a + (-b) :=
by sorry

end subtracting_is_adding_opposite_l211_211771


namespace greatest_prime_factor_of_341_is_17_l211_211234

theorem greatest_prime_factor_of_341_is_17 : ∃ p : ℕ, prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, q ∣ 341 → prime q → q ≤ p :=
sorry

end greatest_prime_factor_of_341_is_17_l211_211234


namespace base6_addition_correct_l211_211293

-- Define a function to convert a base 6 digit to its base 10 equivalent
def base6_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 4
  | 5 => 5
  | d => 0 -- for illegal digits, fallback to 0

-- Define a function to convert a number in base 6 to base 10
def convert_base6_to_base10 (n : Nat) : Nat :=
  let units := base6_to_base10 (n % 10)
  let tens := base6_to_base10 ((n / 10) % 10)
  let hundreds := base6_to_base10 ((n / 100) % 10)
  units + 6 * tens + 6 * 6 * hundreds

-- Define a function to convert a base 10 number to a base 6 number
def base10_to_base6 (n : Nat) : Nat :=
  (n % 6) + 10 * ((n / 6) % 6) + 100 * ((n / (6 * 6)) % 6)

theorem base6_addition_correct : base10_to_base6 (convert_base6_to_base10 35 + convert_base6_to_base10 25) = 104 := by
  sorry

end base6_addition_correct_l211_211293


namespace distance_between_parallel_lines_l211_211041

theorem distance_between_parallel_lines (r d : ℝ) 
  (h1 : ∃ p1 p2 p3 : ℝ, p1 = 40 ∧ p2 = 40 ∧ p3 = 36) 
  (h2 : ∀ θ : ℝ, ∃ A B C D : ℝ → ℝ, 
    (A θ - B θ) = 40 ∧ (C θ - D θ) = 36) : d = 6 :=
sorry

end distance_between_parallel_lines_l211_211041


namespace tangent_line_x_squared_at_one_one_l211_211505

open Real

theorem tangent_line_x_squared_at_one_one :
  ∀ (x y : ℝ), y = x^2 → (x, y) = (1, 1) → (2 * x - y - 1 = 0) :=
by
  intros x y h_curve h_point
  sorry

end tangent_line_x_squared_at_one_one_l211_211505


namespace zack_group_size_l211_211629

theorem zack_group_size (total_students : Nat) (groups : Nat) (group_size : Nat)
  (H1 : total_students = 70)
  (H2 : groups = 7)
  (H3 : total_students = group_size * groups) :
  group_size = 10 := by
  sorry

end zack_group_size_l211_211629


namespace x_sq_add_x_add_1_divides_x_pow_3p_add_x_pow_3m1_add_x_pow_3n2_l211_211744

theorem x_sq_add_x_add_1_divides_x_pow_3p_add_x_pow_3m1_add_x_pow_3n2 
  (x : ℤ) (p m n : ℕ) (hp : 0 < p) (hm : 0 < m) (hn : 0 < n) :
  (x^2 + x + 1) ∣ (x^(3 * p) + x^(3 * m + 1) + x^(3 * n + 2)) :=
by
  sorry

end x_sq_add_x_add_1_divides_x_pow_3p_add_x_pow_3m1_add_x_pow_3n2_l211_211744


namespace function_is_zero_l211_211935

theorem function_is_zero (f : ℝ → ℝ)
  (H : ∀ x y : ℝ, f (f x + x + y) = f (x + y) + y * f y) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end function_is_zero_l211_211935


namespace opposite_of_one_over_2023_l211_211593

def one_over_2023 : ℚ := 1 / 2023

theorem opposite_of_one_over_2023 : -one_over_2023 = -1 / 2023 :=
by
  sorry

end opposite_of_one_over_2023_l211_211593


namespace polynomial_division_l211_211162

open Polynomial

-- Define the theorem statement
theorem polynomial_division (f g : ℤ[X])
  (h : ∀ n : ℤ, f.eval n ∣ g.eval n) :
  ∃ (h : ℤ[X]), g = f * h :=
sorry

end polynomial_division_l211_211162


namespace find_other_number_l211_211767

theorem find_other_number (A : ℕ) (hcf_cond : Nat.gcd A 48 = 12) (lcm_cond : Nat.lcm A 48 = 396) : A = 99 := by
    sorry

end find_other_number_l211_211767


namespace max_profit_thousand_rubles_l211_211075

theorem max_profit_thousand_rubles :
  ∃ x y : ℕ, 
    (80 * x + 100 * y = 2180) ∧ 
    (10 * x + 70 * y ≤ 700) ∧ 
    (23 * x + 40 * y ≤ 642) := 
by
  -- proof goes here
  sorry

end max_profit_thousand_rubles_l211_211075


namespace map_representation_l211_211387

theorem map_representation (length_1_cm length_2_cm : ℕ) (distance_1_km : ℕ) :
  length_1_cm = 15 → distance_1_km = 90 →
  length_2_cm = 20 →
  distance_2_km = length_2_cm * (distance_1_km / length_1_cm) →
  distance_2_km = 120 :=
by
  intros h_len1 h_distance1 h_len2 h_calculate 
  rw [h_len1, h_distance1, h_len2] at h_calculate
  have scale := 90 / 15
  have distance_2_km := 20 * scale
  have : scale = 6 := by norm_num
  rw [this] at h_calculate
  norm_num at distance_2_km
  assumption
  sorry

end map_representation_l211_211387


namespace chord_line_eq_l211_211871

theorem chord_line_eq (x y : ℝ) (h : x^2 + 4 * y^2 = 36) (midpoint : x = 4 ∧ y = 2) :
  x + 2 * y - 8 = 0 := 
sorry

end chord_line_eq_l211_211871


namespace jackson_grade_l211_211003

open Function

theorem jackson_grade :
  ∃ (grade : ℕ), 
  ∀ (hours_playing hours_studying : ℕ), 
    (hours_playing = 9) ∧ 
    (hours_studying = hours_playing / 3) ∧ 
    (grade = hours_studying * 15) →
    grade = 45 := 
by {
  sorry
}

end jackson_grade_l211_211003


namespace quadratic_inequality_solution_l211_211604

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 5*x + 6 ≤ 0) : 
  28 ≤ x^2 + 7*x + 10 ∧ x^2 + 7*x + 10 ≤ 40 :=
sorry

end quadratic_inequality_solution_l211_211604


namespace shaded_triangle_area_l211_211632

theorem shaded_triangle_area (b h : ℝ) (hb : b = 2) (hh : h = 3) : 
  (1 / 2 * b * h) = 3 := 
by
  rw [hb, hh]
  norm_num

end shaded_triangle_area_l211_211632


namespace total_cost_of_office_supplies_l211_211066

-- Define the conditions
def cost_of_pencil : ℝ := 0.5
def cost_of_folder : ℝ := 0.9
def count_of_pencils : ℕ := 24
def count_of_folders : ℕ := 20

-- Define the theorem to prove
theorem total_cost_of_office_supplies
  (cop : ℝ := cost_of_pencil)
  (cof : ℝ := cost_of_folder)
  (ncp : ℕ := count_of_pencils)
  (ncg : ℕ := count_of_folders) :
  cop * ncp + cof * ncg = 30 :=
sorry

end total_cost_of_office_supplies_l211_211066


namespace total_equipment_cost_l211_211200

-- Define the cost of each piece of equipment
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.2
def socks_cost : ℝ := 6.8

-- Define the number of players
def players : ℕ := 16

-- Define the total cost of equipment for one player
def equipment_cost_per_player : ℝ := jersey_cost + shorts_cost + socks_cost

-- Define the total cost for all players
def total_cost : ℝ := players * equipment_cost_per_player

-- The proof problem to be stated:
theorem total_equipment_cost (jc sc k p : ℝ) (n : ℕ) :
  jc = 25 ∧ sc = 15.2 ∧ k = 6.8 ∧ p = 16 →
  total_cost = 752 :=
by
  intro h
  rcases h with ⟨hc1, hc2, hc3, hc4⟩
  simp [total_cost, equipment_cost_per_player, hc1, hc2, hc3, hc4]
  exact sorry

end total_equipment_cost_l211_211200


namespace average_students_l211_211032

def ClassGiraffe : ℕ := 225

def ClassElephant (giraffe: ℕ) : ℕ := giraffe + 48

def ClassRabbit (giraffe: ℕ) : ℕ := giraffe - 24

theorem average_students (giraffe : ℕ) (elephant : ℕ) (rabbit : ℕ) :
  giraffe = 225 → elephant = giraffe + 48 → rabbit = giraffe - 24 →
  (giraffe + elephant + rabbit) / 3 = 233 := by
  sorry

end average_students_l211_211032


namespace BE_eq_FD_l211_211979

variable {A B C D E F : Point}
variable {O : Circle}
variable [hTriangle : Triangle ABC]
variable [hIsosceles : IsoscelesTriangle ABC]
variable [hCircumscribed : CircumscribedCircle ABC O]
variable [hBisector : Bisector CD]
variable [hPerpendicular : PerpendicularToBisectorThroughCenter CD O E]
variable [hParallel : ParallelLineThrough E CD AB F]

-- The conjecture to prove BE = FD
theorem BE_eq_FD :
    IsoscelesTriangle ABC ∧
    Bisector CD ∧
    PerpendicularToBisectorThroughCenter CD O E ∧
    ParallelLineThrough E CD AB F → 
    SegmentLength BE = SegmentLength FD := 
sorry

end BE_eq_FD_l211_211979


namespace total_equipment_cost_l211_211201

-- Define the cost of each piece of equipment
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.2
def socks_cost : ℝ := 6.8

-- Define the number of players
def players : ℕ := 16

-- Define the total cost of equipment for one player
def equipment_cost_per_player : ℝ := jersey_cost + shorts_cost + socks_cost

-- Define the total cost for all players
def total_cost : ℝ := players * equipment_cost_per_player

-- The proof problem to be stated:
theorem total_equipment_cost (jc sc k p : ℝ) (n : ℕ) :
  jc = 25 ∧ sc = 15.2 ∧ k = 6.8 ∧ p = 16 →
  total_cost = 752 :=
by
  intro h
  rcases h with ⟨hc1, hc2, hc3, hc4⟩
  simp [total_cost, equipment_cost_per_player, hc1, hc2, hc3, hc4]
  exact sorry

end total_equipment_cost_l211_211201


namespace julia_played_more_kids_on_monday_l211_211563

def n_monday : ℕ := 6
def n_tuesday : ℕ := 5

theorem julia_played_more_kids_on_monday : n_monday - n_tuesday = 1 := by
  -- Proof goes here
  sorry

end julia_played_more_kids_on_monday_l211_211563


namespace mark_donates_cans_of_soup_l211_211577

theorem mark_donates_cans_of_soup:
  let n_shelters := 6
  let p_per_shelter := 30
  let c_per_person := 10
  let total_people := n_shelters * p_per_shelter
  let total_cans := total_people * c_per_person
  total_cans = 1800 :=
by sorry

end mark_donates_cans_of_soup_l211_211577


namespace solve_inequality_l211_211095

noncomputable def f : ℝ → ℝ := sorry

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_nonnegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

def f_at_one_third (f : ℝ → ℝ) : Prop :=
  f (1/3) = 0

theorem solve_inequality (f : ℝ → ℝ) (x : ℝ) :
  even_function f →
  increasing_on_nonnegatives f →
  f_at_one_third f →
  (0 < x ∧ x < 1/2) ∨ (x > 2) ↔ f (Real.logb (1/8) x) > 0 :=
by
  -- the proof will be filled in here
  sorry

end solve_inequality_l211_211095


namespace rationalize_denominator_l211_211176

theorem rationalize_denominator :
  let expr := (2 + Real.sqrt 5) / (3 - Real.sqrt 5),
      A := 11 / 4,
      B := 5 / 4,
      C := 5
  in expr = (A + B * Real.sqrt C) → (A * B * C) = 275 / 16 :=
by
  intros
  sorry

end rationalize_denominator_l211_211176


namespace value_of_n_l211_211054

theorem value_of_n (n : ℝ) (h : n + (n + 1) + (n + 2) + (n + 3) = 20) : n = 3.5 :=
by sorry

end value_of_n_l211_211054


namespace trigonometric_identity_l211_211275

theorem trigonometric_identity : sin (20 * real.pi / 180) * cos (10 * real.pi / 180) - cos (160 * real.pi / 180) * sin (10 * real.pi / 180) = 1 / 2 :=
by
  sorry

end trigonometric_identity_l211_211275


namespace find_number_l211_211146

-- Define the condition that k is a non-negative integer
def is_nonnegative_int (k : ℕ) : Prop := k ≥ 0

-- Define the condition that 18^k is a divisor of the number n
def is_divisor (n k : ℕ) : Prop := 18^k ∣ n

-- The main theorem statement
theorem find_number (n k : ℕ) (h_nonneg : is_nonnegative_int k) (h_eq : 6^k - k^6 = 1) (h_div : is_divisor n k) : n = 1 :=
  sorry

end find_number_l211_211146


namespace monotonicity_of_f_extremum_of_f_on_interval_l211_211829

noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x + 1)

theorem monotonicity_of_f : ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ < x₂ → 1 ≤ x₂ → f x₁ < f x₂ := by
  sorry

theorem extremum_of_f_on_interval : 
  f 1 = 3 / 2 ∧ f 4 = 9 / 5 := by
  sorry

end monotonicity_of_f_extremum_of_f_on_interval_l211_211829


namespace prime_power_of_n_l211_211015

theorem prime_power_of_n (n : ℕ) (h : Nat.Prime (4^n + 2^n + 1)) : ∃ k : ℕ, n = 3^k := 
sorry

end prime_power_of_n_l211_211015


namespace Q_evaluation_at_2_l211_211033

noncomputable def Q : Polynomial ℚ := 
  (Polynomial.X^2 + Polynomial.C 4)^2

theorem Q_evaluation_at_2 : 
  Q.eval 2 = 64 :=
by 
  -- We'll skip the proof as per the instructions.
  sorry

end Q_evaluation_at_2_l211_211033


namespace unique_prime_solution_l211_211705

theorem unique_prime_solution :
  ∃! (p : ℕ), Prime p ∧ (∃ (k : ℤ), 2 * (p ^ 4) - 7 * (p ^ 2) + 1 = k ^ 2) := 
sorry

end unique_prime_solution_l211_211705


namespace fran_speed_calculation_l211_211562

noncomputable def fran_speed (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) : ℝ :=
  joann_speed * joann_time / fran_time

theorem fran_speed_calculation : 
  fran_speed 15 3 2.5 = 18 := 
by
  -- Remember to write down the proof steps if needed, currently we use sorry as placeholder
  sorry

end fran_speed_calculation_l211_211562


namespace triangle_XOY_hypotenuse_l211_211728

theorem triangle_XOY_hypotenuse (a b : ℝ) (h1 : (a/2)^2 + b^2 = 22^2) (h2 : a^2 + (b/2)^2 = 19^2) :
  Real.sqrt (a^2 + b^2) = 26 :=
sorry

end triangle_XOY_hypotenuse_l211_211728


namespace each_child_plays_for_90_minutes_l211_211843

-- Definitions based on the conditions
def total_playing_time : ℕ := 180
def children_playing_at_a_time : ℕ := 3
def total_children : ℕ := 6

-- The proof problem statement
theorem each_child_plays_for_90_minutes :
  (children_playing_at_a_time * total_playing_time) / total_children = 90 := by
  sorry

end each_child_plays_for_90_minutes_l211_211843


namespace sum_tens_ones_digit_of_7_pow_11_l211_211467

/--
The sum of the tens digit and the ones digit of (3+4)^{11} is 7.
-/
theorem sum_tens_ones_digit_of_7_pow_11 : 
  let number := (3 + 4)^11
  let tens_digit := (number / 10) % 10
  let ones_digit := number % 10
  tens_digit + ones_digit = 7 :=
by
  sorry

end sum_tens_ones_digit_of_7_pow_11_l211_211467


namespace greatest_prime_factor_341_l211_211252

theorem greatest_prime_factor_341 : ∃ p, Nat.Prime p ∧ p ≥ 17 ∧ (∀ q, Nat.Prime q ∧ q ∣ 341 → q ≤ p) ∧ p = 19 := by
  sorry

end greatest_prime_factor_341_l211_211252


namespace functional_equation_solution_l211_211669

theorem functional_equation_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f x * f y = f (x - y)) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) :=
sorry

end functional_equation_solution_l211_211669


namespace percentage_of_copper_first_alloy_l211_211085

theorem percentage_of_copper_first_alloy :
  ∃ x : ℝ, 
  (66 * x / 100) + (55 * 21 / 100) = 121 * 15 / 100 ∧
  x = 10 := 
sorry

end percentage_of_copper_first_alloy_l211_211085


namespace unique_poly_degree_4_l211_211037

theorem unique_poly_degree_4 
  (Q : ℚ[X]) 
  (h_deg : Q.degree = 4) 
  (h_lead_coeff : Q.leading_coeff = 1)
  (h_root : Q.eval (sqrt 3 + sqrt 7) = 0) : 
  Q = X^4 - 12 * X^2 + 16 ∧ Q.eval 2 = 0 :=
by sorry

end unique_poly_degree_4_l211_211037


namespace arithmetic_sequence_S12_l211_211559

theorem arithmetic_sequence_S12 (S : ℕ → ℝ) (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (hS4 : S 4 = 25) (hS8 : S 8 = 100) : S 12 = 225 :=
by
  sorry

end arithmetic_sequence_S12_l211_211559


namespace map_scale_representation_l211_211424

theorem map_scale_representation :
  (∀ len_km_1 len_cm_1 len_cm_2 : ℕ, len_km_1 = 90 → len_cm_1 = 15 → len_cm_2 = 20 →
    let scale := len_km_1 / len_cm_1 in
    len_cm_2 * scale = 120)
  := sorry

end map_scale_representation_l211_211424


namespace find_value_of_m_l211_211269

variables (x y m : ℝ)

theorem find_value_of_m (h1 : y ≥ x) (h2 : x + 3 * y ≤ 4) (h3 : x ≥ m) (hz_max : ∀ z, (z = x - 3 * y) → z ≤ 8) :
  m = -4 :=
sorry

end find_value_of_m_l211_211269


namespace six_digit_product_of_consecutive_even_integers_l211_211942

theorem six_digit_product_of_consecutive_even_integers :
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ (a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0) ∧ a * b * c = 287232 :=
sorry

end six_digit_product_of_consecutive_even_integers_l211_211942


namespace elastic_collision_inelastic_collision_l211_211451

-- Definition of conditions
variables {m L V : ℝ}
variables (w1 w2 : ℝ → Prop)

-- Proof problem for Elastic Collision
theorem elastic_collision (w1_at_collision : w1 L) (w2_at_collision : w2 L)  :
  let v1_after := V
      v2_after := -V in
  w1 L ∧ w2 L := sorry

-- Proof problem for Inelastic Collision
theorem inelastic_collision (w1_at_collision : w1 L) (w2_at_collision : w2 L)  :
  let omega := V / (2 * L) in
  w1 L ∧ w2 L := sorry

end elastic_collision_inelastic_collision_l211_211451


namespace missing_score_and_variance_l211_211974

theorem missing_score_and_variance (score_A score_B score_D score_E : ℕ) (avg_score : ℕ)
  (h_scores : score_A = 81 ∧ score_B = 79 ∧ score_D = 80 ∧ score_E = 82)
  (h_avg : avg_score = 80):
  ∃ (score_C variance : ℕ), score_C = 78 ∧ variance = 2 := by
  sorry

end missing_score_and_variance_l211_211974


namespace max_intersections_five_points_l211_211976

noncomputable def max_perpendicular_intersections (n : ℕ) : ℕ :=
  let num_lines := (Nat.choose n 2)
  let num_perpendiculars := n * (Nat.choose (n - 1) 2)
  let max_intersections := Nat.choose num_perpendiculars 2
  let adjust_perpendiculars := n * (Nat.choose (n - 1) 2)
  let adjust_triangles := (Nat.choose n 3) * (Nat.choose (n - 1) 2 - 1)
  let adjust_points := n * (Nat.choose (Nat.choose (n - 1) 2) 2)
  (max_intersections - adjust_perpendiculars - adjust_triangles - adjust_points)

theorem max_intersections_five_points : max_perpendicular_intersections 5 = 310 := by
  sorry

end max_intersections_five_points_l211_211976


namespace interval_for_f_l211_211346

noncomputable def f (x : ℝ) : ℝ :=
-0.5 * x ^ 2 + 13 / 2

theorem interval_for_f (a b : ℝ) :
  f a = 2 * b ∧ f b = 2 * a ∧ (a ≤ 0 ∨ 0 ≤ b) → 
  ([a, b] = [1, 3] ∨ [a, b] = [-2 - Real.sqrt 17, 13 / 4]) :=
by sorry

end interval_for_f_l211_211346


namespace builder_windows_installed_l211_211781

theorem builder_windows_installed (total_windows : ℕ) (hours_per_window : ℕ) (total_hours_left : ℕ) :
  total_windows = 14 → hours_per_window = 4 → total_hours_left = 36 → (total_windows - total_hours_left / hours_per_window) = 5 :=
by
  intros
  sorry

end builder_windows_installed_l211_211781


namespace total_miles_driven_l211_211746

-- Given constants and conditions
def city_mpg : ℝ := 30
def highway_mpg : ℝ := 37
def total_gallons : ℝ := 11
def highway_extra_miles : ℕ := 5

-- Variable for the number of city miles
variable (x : ℝ)

-- Conditions encapsulated in a theorem statement
theorem total_miles_driven:
  (x / city_mpg) + ((x + highway_extra_miles) / highway_mpg) = total_gallons →
  x + (x + highway_extra_miles) = 365 :=
by
  sorry

end total_miles_driven_l211_211746


namespace distribute_pencils_l211_211353

theorem distribute_pencils (number_of_pencils : ℕ) (number_of_people : ℕ)
  (h_pencils : number_of_pencils = 2) (h_people : number_of_people = 5) :
  number_of_distributions = 15 := by
  sorry

end distribute_pencils_l211_211353


namespace y_intercept_of_line_l211_211196

theorem y_intercept_of_line (m x1 y1 : ℝ) (x_intercept : x1 = 4) (y_intercept_at_x1_zero : y1 = 0) (m_value : m = -3) :
  ∃ b : ℝ, (∀ x y : ℝ, y = m * x + b ∧ x = 0 → y = b) ∧ b = 12 :=
by
  sorry

end y_intercept_of_line_l211_211196


namespace find_y_rotation_l211_211089

def rotate_counterclockwise (A : Point) (B : Point) (θ : ℝ) : Point := sorry
def rotate_clockwise (A : Point) (B : Point) (θ : ℝ) : Point := sorry

variable {A B C : Point}
variable {y : ℝ}

theorem find_y_rotation
  (h1 : rotate_counterclockwise A B 450 = C)
  (h2 : rotate_clockwise A B y = C)
  (h3 : y < 360) :
  y = 270 :=
sorry

end find_y_rotation_l211_211089


namespace matchstick_ratio_is_one_half_l211_211735

def matchsticks_used (houses : ℕ) (matchsticks_per_house : ℕ) : ℕ :=
  houses * matchsticks_per_house

def ratio (a b : ℕ) : ℚ := a / b

def michael_original_matchsticks : ℕ := 600
def michael_houses : ℕ := 30
def matchsticks_per_house : ℕ := 10
def michael_used_matchsticks : ℕ := matchsticks_used michael_houses matchsticks_per_house

theorem matchstick_ratio_is_one_half :
  ratio michael_used_matchsticks michael_original_matchsticks = 1 / 2 :=
by
  sorry

end matchstick_ratio_is_one_half_l211_211735


namespace nonnegative_solutions_eq_1_l211_211135

variable (x : ℝ)

theorem nonnegative_solutions_eq_1 : (x^2 = -6 * x) → (∃ x₀ : ℝ, x₀ ≥ 0 ∧ x = x₀) → (x ∈ {0}) :=
by
  assume (h1: x^2 = -6 * x)
  assume (h2: ∃ x₀ : ℝ, x₀ ≥ 0 ∧ x = x₀)
  sorry

end nonnegative_solutions_eq_1_l211_211135


namespace find_product_of_M1_M2_l211_211566

theorem find_product_of_M1_M2 (x M1 M2 : ℝ) 
  (h : (27 * x - 19) / (x^2 - 5 * x + 6) = M1 / (x - 2) + M2 / (x - 3)) : 
  M1 * M2 = -2170 := 
sorry

end find_product_of_M1_M2_l211_211566


namespace largest_divisor_of_expression_of_even_x_l211_211838

theorem largest_divisor_of_expression_of_even_x (x : ℤ) (h_even : ∃ k : ℤ, x = 2 * k) :
  ∃ (d : ℤ), d = 240 ∧ d ∣ ((8 * x + 2) * (8 * x + 4) * (4 * x + 2)) :=
by
  sorry

end largest_divisor_of_expression_of_even_x_l211_211838


namespace total_revenue_correct_l211_211894

-- Defining the basic parameters
def ticket_price : ℝ := 20
def first_discount_percentage : ℝ := 0.40
def next_discount_percentage : ℝ := 0.15
def first_people : ℕ := 10
def next_people : ℕ := 20
def total_people : ℕ := 48

-- Calculate the discounted prices based on the given percentages
def discounted_price_first : ℝ := ticket_price * (1 - first_discount_percentage)
def discounted_price_next : ℝ := ticket_price * (1 - next_discount_percentage)

-- Calculate the total revenue
def revenue_first : ℝ := first_people * discounted_price_first
def revenue_next : ℝ := next_people * discounted_price_next
def remaining_people : ℕ := total_people - first_people - next_people
def revenue_remaining : ℝ := remaining_people * ticket_price

def total_revenue : ℝ := revenue_first + revenue_next + revenue_remaining

-- The statement to be proved
theorem total_revenue_correct : total_revenue = 820 :=
by
  -- The proof will go here
  sorry

end total_revenue_correct_l211_211894


namespace scientific_notation_826M_l211_211988

theorem scientific_notation_826M : 826000000 = 8.26 * 10^8 :=
by
  sorry

end scientific_notation_826M_l211_211988


namespace knights_statements_l211_211740

theorem knights_statements (r ℓ : Nat) (hr : r ≥ 2) (hℓ : ℓ ≥ 2)
  (h : 2 * r * ℓ = 230) :
  (r + ℓ) * (r + ℓ - 1) - 230 = 526 :=
by
  sorry

end knights_statements_l211_211740


namespace line_does_not_pass_through_third_quadrant_l211_211025

theorem line_does_not_pass_through_third_quadrant (x y : ℝ) (h : y = -x + 1) :
  ¬(x < 0 ∧ y < 0) :=
sorry

end line_does_not_pass_through_third_quadrant_l211_211025


namespace find_middle_number_l211_211575

theorem find_middle_number (x y z : ℕ) (h1 : x < y) (h2 : y < z)
  (h3 : x + y = 22) (h4 : x + z = 29) (h5 : y + z = 31) (h6 : x = 10) :
  y = 12 :=
sorry

end find_middle_number_l211_211575


namespace sum_of_digits_divisible_by_9_l211_211485

theorem sum_of_digits_divisible_by_9 (D E : ℕ) (hD : D < 10) (hE : E < 10) : 
  (D + E + 37) % 9 = 0 → ((D + E = 8) ∨ (D + E = 17)) →
  (8 + 17 = 25) := 
by
  intro h1 h2
  sorry

end sum_of_digits_divisible_by_9_l211_211485


namespace initial_roses_count_l211_211964

theorem initial_roses_count 
  (roses_to_mother : ℕ)
  (roses_to_grandmother : ℕ)
  (roses_to_sister : ℕ)
  (roses_kept : ℕ)
  (initial_roses : ℕ)
  (h_mother : roses_to_mother = 6)
  (h_grandmother : roses_to_grandmother = 9)
  (h_sister : roses_to_sister = 4)
  (h_kept : roses_kept = 1)
  (h_initial : initial_roses = roses_to_mother + roses_to_grandmother + roses_to_sister + roses_kept) :
  initial_roses = 20 :=
by
  rw [h_mother, h_grandmother, h_sister, h_kept] at h_initial
  exact h_initial

end initial_roses_count_l211_211964


namespace emilia_donut_holes_count_l211_211303

noncomputable def surface_area (r : ℕ) : ℕ := 4 * r^2

def lcm (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

def donut_holes := 5103

theorem emilia_donut_holes_count :
  ∀ (S1 S2 S3 : ℕ), 
  S1 = surface_area 5 → 
  S2 = surface_area 7 → 
  S3 = surface_area 9 → 
  donut_holes = lcm S1 S2 S3 / S1 :=
by
  intros S1 S2 S3 hS1 hS2 hS3
  sorry

end emilia_donut_holes_count_l211_211303


namespace maura_seashells_l211_211579

theorem maura_seashells (original_seashells given_seashells remaining_seashells : ℕ)
  (h1 : original_seashells = 75) 
  (h2 : remaining_seashells = 57) 
  (h3 : given_seashells = original_seashells - remaining_seashells) :
  given_seashells = 18 := by
  -- Lean will use 'sorry' as a placeholder for the actual proof
  sorry

end maura_seashells_l211_211579


namespace gold_coins_percent_l211_211083

variable (total_objects beads papers coins silver_gold total_gold : ℝ)
variable (h1 : total_objects = 100)
variable (h2 : beads = 15)
variable (h3 : papers = 10)
variable (h4 : silver_gold = 30)
variable (h5 : total_gold = 52.5)

theorem gold_coins_percent : (total_objects - beads - papers) * (100 - silver_gold) / 100 = total_gold :=
by 
  -- Insert proof here
  sorry

end gold_coins_percent_l211_211083


namespace map_scale_l211_211399

theorem map_scale (represents_15cm_as_km : 15 * 6 = 90) :
  20 * 6 = 120 :=
by
  sorry

end map_scale_l211_211399


namespace original_number_l211_211484

theorem original_number (x : ℝ) (h : 1.35 * x = 935) : x = 693 := by
  sorry

end original_number_l211_211484


namespace lineup_count_l211_211558

-- Define five distinct people
inductive Person 
| youngest : Person 
| oldest : Person 
| person1 : Person 
| person2 : Person 
| person3 : Person 

-- Define the total number of people
def numberOfPeople : ℕ := 5

-- Define a function to calculate the number of ways to line up five people with constraints
def lineupWays : ℕ := 3 * 4 * 3 * 2 * 1

-- State the theorem
theorem lineup_count (h₁ : numberOfPeople = 5) (h₂ : ¬ ∃ (p : Person), p = Person.youngest ∨ p = Person.oldest → p = Person.youngest) :
  lineupWays = 72 :=
by
  sorry

end lineup_count_l211_211558


namespace distance_to_place_l211_211067

theorem distance_to_place (rowing_speed still_water : ℝ) (downstream_speed : ℝ)
                         (upstream_speed : ℝ) (total_time : ℝ) (distance : ℝ) :
  rowing_speed = 10 → downstream_speed = 2 → upstream_speed = 3 →
  total_time = 10 → distance = 44.21 → 
  (distance / (rowing_speed + downstream_speed) + distance / (rowing_speed - upstream_speed)) = 10 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3]
  field_simp
  sorry

end distance_to_place_l211_211067


namespace rectangle_proof_right_triangle_proof_l211_211835

-- Definition of rectangle condition
def rectangle_condition (a b : ℕ) : Prop :=
  a * b = 2 * (a + b)

-- Definition of right triangle condition
def right_triangle_condition (a b : ℕ) : Prop :=
  a + b + Int.natAbs (Int.sqrt (a^2 + b^2)) = a * b / 2 ∧
  (∃ c : ℕ, c = Int.natAbs (Int.sqrt (a^2 + b^2)))

-- Recangle proof
theorem rectangle_proof : ∃! p : ℕ × ℕ, rectangle_condition p.1 p.2 := sorry

-- Right triangle proof
theorem right_triangle_proof : ∃! t : ℕ × ℕ, right_triangle_condition t.1 t.2 := sorry

end rectangle_proof_right_triangle_proof_l211_211835


namespace cupcakes_total_l211_211317

theorem cupcakes_total (initially_made : ℕ) (sold : ℕ) (newly_made : ℕ) (initially_made_eq : initially_made = 42) (sold_eq : sold = 22) (newly_made_eq : newly_made = 39) : initially_made - sold + newly_made = 59 :=
by
  sorry

end cupcakes_total_l211_211317


namespace initial_concentration_l211_211281

theorem initial_concentration (C : ℝ) 
  (hC : (C * 0.2222222222222221) + (0.25 * 0.7777777777777779) = 0.35) :
  C = 0.7 :=
sorry

end initial_concentration_l211_211281


namespace parabola_equation_l211_211186

theorem parabola_equation (h_axis : ∃ p > 0, x = p / 2) :
  ∃ p > 0, y^2 = -2 * p * x :=
by 
  -- proof steps will be added here
  sorry

end parabola_equation_l211_211186


namespace quadratic_inequality_sufficient_necessary_l211_211292

theorem quadratic_inequality_sufficient_necessary (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + a > 0) ↔ 0 < a ∧ a < 4 :=
by
  -- proof skipped
  sorry

end quadratic_inequality_sufficient_necessary_l211_211292


namespace tan_sub_pi_over_4_l211_211681

-- Define the conditions and the problem statement
variable (α : ℝ) (h : Real.tan α = 2)

-- State the problem as a theorem
theorem tan_sub_pi_over_4 : Real.tan (α - Real.pi / 4) = 1 / 3 :=
by
  sorry

end tan_sub_pi_over_4_l211_211681


namespace equation_has_one_negative_and_one_zero_root_l211_211865

theorem equation_has_one_negative_and_one_zero_root :
  ∃ x y : ℝ, x < 0 ∧ y = 0 ∧ 3^x + x^2 + 2 * x - 1 = 0 ∧ 3^y + y^2 + 2 * y - 1 = 0 :=
sorry

end equation_has_one_negative_and_one_zero_root_l211_211865


namespace find_diameter_of_hemisphere_l211_211609

theorem find_diameter_of_hemisphere (r a : ℝ) (hr : r = a / 2) (volume : ℝ) (hV : volume = 18 * Real.pi) : 
  2/3 * Real.pi * r ^ 3 = 18 * Real.pi → a = 6 := by
  intro h
  sorry

end find_diameter_of_hemisphere_l211_211609


namespace map_length_representation_l211_211406

theorem map_length_representation :
  ∀ (length1 length2 km1 : ℝ), 
  length1 = 15 ∧ km1 = 90 ∧ length2 = 20 →
  (length2 * (km1 / length1) = 120) :=
by
  intros length1 length2 km1
  intro h
  cases h with hl1 h
  cases h with hkm1 hl2
  rw [hl1, hkm1, hl2]
  simp
  sorry

end map_length_representation_l211_211406


namespace max_sum_ac_bc_l211_211551

noncomputable def triangle_ab_bc_sum_max (AB : ℝ) (C : ℝ) : ℝ :=
  if AB = Real.sqrt 6 - Real.sqrt 2 ∧ C = Real.pi / 6 then 4 else 0

theorem max_sum_ac_bc {A B C : ℝ} (h1 : AB = Real.sqrt 6 - Real.sqrt 2) (h2 : C = Real.pi / 6) :
  triangle_ab_bc_sum_max AB C = 4 :=
by {
  sorry
}

end max_sum_ac_bc_l211_211551


namespace sign_of_x_minus_y_l211_211839

theorem sign_of_x_minus_y (x y a : ℝ) (h1 : x + y > 0) (h2 : a < 0) (h3 : a * y > 0) : x - y > 0 := 
by 
  sorry

end sign_of_x_minus_y_l211_211839


namespace consecutive_even_product_6digit_l211_211938

theorem consecutive_even_product_6digit :
  ∃ (a b c : ℕ), 
  (a % 2 = 0) ∧ (b = a + 2) ∧ (c = a + 4) ∧ 
  (Nat.digits 10 (a * b * c)).length = 6 ∧ 
  (Nat.digits 10 (a * b * c)).head! = 2 ∧ 
  (Nat.digits 10 (a * b * c)).getLast! = 2 ∧ 
  (a * b * c = 287232) :=
by
  sorry

end consecutive_even_product_6digit_l211_211938


namespace no_solution_fermat_like_l211_211277

theorem no_solution_fermat_like (x y z k : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hk : k > 0) 
  (hxk : x < k) (hyk : y < k) (hxk_eq : x ^ k + y ^ k = z ^ k) : false :=
sorry

end no_solution_fermat_like_l211_211277


namespace find_missing_number_l211_211585

theorem find_missing_number (x : ℤ) (h : (4 + 3) + (8 - x - 1) = 11) : x = 3 :=
sorry

end find_missing_number_l211_211585


namespace find_a_div_b_l211_211732

theorem find_a_div_b (a b : ℝ) (h_distinct : a ≠ b) 
  (h_eq : a / b + (a + 5 * b) / (b + 5 * a) = 2) : a / b = 0.6 :=
by
  sorry

end find_a_div_b_l211_211732


namespace clea_ride_escalator_time_l211_211724

theorem clea_ride_escalator_time (x y k : ℝ) (h1 : 80 * x = y) (h2 : 30 * (x + k) = y) : (y / k) + 5 = 53 :=
by {
  sorry
}

end clea_ride_escalator_time_l211_211724


namespace initial_investments_l211_211773

theorem initial_investments (x y : ℝ) : 
  -- Conditions
  5000 = y + (5000 - y) ∧
  (y * (1 + x / 100) = 2100) ∧
  ((5000 - y) * (1 + (x + 1) / 100) = 3180) →
  -- Conclusion
  y = 2000 ∧ (5000 - y) = 3000 := 
by 
  sorry

end initial_investments_l211_211773


namespace volume_frustum_as_fraction_of_original_l211_211644

theorem volume_frustum_as_fraction_of_original :
  let original_base_edge := 40
  let original_altitude := 20
  let smaller_altitude := original_altitude / 3
  let smaller_base_edge := original_base_edge / 3
  let volume_original := (1 / 3) * (original_base_edge * original_base_edge) * original_altitude
  let volume_smaller := (1 / 3) * (smaller_base_edge * smaller_base_edge) * smaller_altitude
  let volume_frustum := volume_original - volume_smaller
  (volume_frustum / volume_original) = (87 / 96) :=
by
  let original_base_edge := 40
  let original_altitude := 20
  let smaller_altitude := original_altitude / 3
  let smaller_base_edge := original_base_edge / 3
  let volume_original := (1 / 3) * (original_base_edge * original_base_edge) * original_altitude
  let volume_smaller := (1 / 3) * (smaller_base_edge * smaller_base_edge) * smaller_altitude
  let volume_frustum := volume_original - volume_smaller
  have h : volume_frustum / volume_original = 87 / 96 := sorry
  exact h

end volume_frustum_as_fraction_of_original_l211_211644


namespace find_rate_of_current_l211_211878

noncomputable def rate_of_current (speed_boat : ℝ) (distance_downstream : ℝ) (time_downstream_min : ℝ) : ℝ :=
 speed_boat + 3

theorem find_rate_of_current : ∀ (speed_boat distance_downstream time_downstream_min : ℝ), 
  speed_boat = 15 → 
  distance_downstream = 3.6 → 
  time_downstream_min = 12 → 
  rate_of_current speed_boat distance_downstream time_downstream_min = 18 - speed_boat :=
by
  intros speed_boat distance_downstream time_downstream_min h_speed h_distance h_time
  rw [h_speed, h_distance, h_time]
  have time_hours : ℝ := (12 / 60 : ℝ)
  calc
    (rate_of_current 15 3.6 12) = speed_boat + 3 : rfl
    ... = 15 + 3 : by rw [h_speed]
    ... = 18 - speed_boat : by rw [h_speed]

end find_rate_of_current_l211_211878


namespace sqrt_14_range_l211_211929

theorem sqrt_14_range : 3 < Real.sqrt 14 ∧ Real.sqrt 14 < 4 :=
by
  -- We know that 9 < 14 < 16, so we can take the square root of all parts to get 3 < sqrt(14) < 4.
  sorry

end sqrt_14_range_l211_211929


namespace count_integer_length_chords_l211_211861

/-- Point P is 9 units from the center of a circle with radius 15. -/
def point_distance_from_center : ℝ := 9

def circle_radius : ℝ := 15

/-- Correct answer to the number of different chords that contain P and have integer lengths. -/
def correct_answer : ℕ := 7

/-- Proving the number of chords containing P with integer lengths given the conditions. -/
theorem count_integer_length_chords : 
  ∀ (r_P : ℝ) (r_circle : ℝ), r_P = point_distance_from_center → r_circle = circle_radius → 
  (∃ n : ℕ, n = correct_answer) :=
by 
  intros r_P r_circle h1 h2
  use 7 
  sorry

end count_integer_length_chords_l211_211861


namespace ratio_of_chickens_in_run_to_coop_l211_211610

def chickens_in_coop : ℕ := 14
def free_ranging_chickens : ℕ := 52
def run_condition (R : ℕ) : Prop := 2 * R - 4 = 52

theorem ratio_of_chickens_in_run_to_coop (R : ℕ) (hR : run_condition R) :
  R / chickens_in_coop = 2 :=
by
  sorry

end ratio_of_chickens_in_run_to_coop_l211_211610


namespace twenty_is_80_percent_of_what_number_l211_211220

theorem twenty_is_80_percent_of_what_number : ∃ y : ℕ, (20 : ℚ) / y = 4 / 5 ∧ y = 25 := by
  sorry

end twenty_is_80_percent_of_what_number_l211_211220


namespace triangle_angle_sum_l211_211719

theorem triangle_angle_sum (a b : ℝ) (ha : a = 40) (hb : b = 60) : ∃ x : ℝ, x = 180 - (a + b) :=
by
  use 80
  sorry

end triangle_angle_sum_l211_211719


namespace algebraic_simplification_l211_211967

theorem algebraic_simplification (m x : ℝ) (h₀ : 0 < m) (h₁ : m < 10) (h₂ : m ≤ x) (h₃ : x ≤ 10) : 
  |x - m| + |x - 10| + |x - m - 10| = 20 - x :=
by
  sorry

end algebraic_simplification_l211_211967


namespace tangent_line_equation_l211_211104
-- Import the necessary Lean library

-- Define the function and the point
def curve (x : ℝ) : ℝ := 4 / (Real.exp x + 1)

theorem tangent_line_equation : 
  let x0 := 0
  let y0 := 2
  let tangent_slope := -1
  let tangent_line := λ x y : ℝ, x + y - 2
  ∀ x y, 
  HasDerivAt curve tangent_slope x0 ∧
  curve x0 = y0 →
  tangent_line (x0 + 1) (y0 - 1) = 0 := 
by
  sorry

end tangent_line_equation_l211_211104


namespace num_integers_satisfying_inequality_l211_211539

theorem num_integers_satisfying_inequality : 
  {x : ℤ | (x - 2)^2 ≤ 4}.finite.to_finset.card = 5 :=
by {
  sorry
}

end num_integers_satisfying_inequality_l211_211539


namespace intersection_M_N_l211_211125

def M (x : ℝ) : Prop := (x - 3) / (x + 1) > 0
def N (x : ℝ) : Prop := 3 * x + 2 > 0

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 3 < x} :=
by
  sorry

end intersection_M_N_l211_211125


namespace nonnegative_solutions_count_l211_211138

theorem nonnegative_solutions_count :
  {x : ℝ | x^2 = -6 * x ∧ 0 ≤ x}.finite.to_finset.card = 1 :=
by
  sorry

end nonnegative_solutions_count_l211_211138


namespace standard_deviation_of_applicants_ages_l211_211022

noncomputable def average_age : ℝ := 30
noncomputable def max_different_ages : ℝ := 15

theorem standard_deviation_of_applicants_ages 
  (σ : ℝ)
  (h : max_different_ages = 2 * σ) 
  : σ = 7.5 :=
by
  sorry

end standard_deviation_of_applicants_ages_l211_211022


namespace mul_example_l211_211667

theorem mul_example : (3.6 * 0.5 = 1.8) := by
  sorry

end mul_example_l211_211667


namespace find_b_l211_211842

theorem find_b (a b : ℝ) (h : ∀ x, 2 * x^2 - a * x + 4 < 0 ↔ 1 < x ∧ x < b) : b = 2 :=
sorry

end find_b_l211_211842


namespace find_number_l211_211223

theorem find_number (x : ℝ) (h : 20 / x = 0.8) : x = 25 := 
by
  sorry

end find_number_l211_211223


namespace square_inequality_l211_211142

theorem square_inequality (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) : a^2 > b^2 :=
by
  sorry

end square_inequality_l211_211142


namespace integer_solutions_to_inequality_l211_211537

theorem integer_solutions_to_inequality :
  ∃ n : ℕ, n = 5 ∧ (∀ x : ℤ, (x - 2) ^ 2 ≤ 4 ↔ x ∈ {0, 1, 2, 3, 4}) :=
by
  sorry

end integer_solutions_to_inequality_l211_211537


namespace range_of_k_l211_211526

noncomputable def quadratic_has_real_roots (k : ℝ) :=
  ∃ (x : ℝ), (k - 3) * x^2 - 4 * x + 2 = 0

theorem range_of_k (k : ℝ) : quadratic_has_real_roots k ↔ k ≤ 5 := 
  sorry

end range_of_k_l211_211526


namespace find_number_l211_211892

theorem find_number (n : ℕ) : (n / 2) + 5 = 15 → n = 20 :=
by
  intro h
  sorry

end find_number_l211_211892


namespace skating_speeds_ratio_l211_211776

theorem skating_speeds_ratio (v_s v_f : ℝ) (h1 : v_f > v_s) (h2 : |v_f + v_s| / |v_f - v_s| = 5) :
  v_f / v_s = 3 / 2 :=
by
  sorry

end skating_speeds_ratio_l211_211776


namespace local_maximum_no_global_maximum_equation_root_condition_l211_211331

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3) * Real.exp x
noncomputable def f' (x : ℝ) : ℝ := (x^2 + 2*x - 3) * Real.exp x

theorem local_maximum_no_global_maximum : (∃ x0 : ℝ, f' x0 = 0 ∧ (∀ x < x0, f x < f x0) ∧ (∀ x > x0, f x < f x0))
∧ (f 1 = -2 * Real.exp 1) 
∧ (∀ x : ℝ, ∃ b : ℝ, f x = b ∧ b > 6 * Real.exp (-3) → ¬(f x = f 1))
:= sorry

theorem equation_root_condition (b : ℝ) : (∃ x1 x2 x3 : ℝ, f x1 = b ∧ f x2 = b ∧ f x3 = b ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) 
→ (0 < b ∧ b < 6 * Real.exp (-3))
:= sorry

end local_maximum_no_global_maximum_equation_root_condition_l211_211331


namespace people_in_house_l211_211624

theorem people_in_house 
  (charlie_and_susan : ℕ := 2)
  (sarah_and_friends : ℕ := 5)
  (living_room_people : ℕ := 8) :
  (charlie_and_susan + sarah_and_friends) + living_room_people = 15 := 
by
  sorry

end people_in_house_l211_211624


namespace necessary_but_not_sufficient_l211_211321

theorem necessary_but_not_sufficient (a : ℝ) : 
  (a > 2 → a^2 > 2 * a) ∧ (a^2 > 2 * a → (a > 2 ∨ a < 0)) :=
by
  sorry

end necessary_but_not_sufficient_l211_211321


namespace least_multiple_of_36_with_digit_product_multiple_of_9_l211_211260

def is_multiple_of_36 (n : ℕ) : Prop :=
  n % 36 = 0

def product_of_digits_multiple_of_9 (n : ℕ) : Prop :=
  ∃ d : List ℕ, (n = List.foldl (λ x y => x * 10 + y) 0 d) ∧ (List.foldl (λ x y => x * y) 1 d) % 9 = 0

theorem least_multiple_of_36_with_digit_product_multiple_of_9 : ∃ n : ℕ, is_multiple_of_36 n ∧ product_of_digits_multiple_of_9 n ∧ n = 36 :=
by
  sorry

end least_multiple_of_36_with_digit_product_multiple_of_9_l211_211260


namespace Margie_can_drive_200_miles_l211_211166

/--
  Margie's car can go 40 miles per gallon of gas, and the price of gas is $5 per gallon.
  Prove that Margie can drive 200 miles with $25 worth of gas.
-/
theorem Margie_can_drive_200_miles (miles_per_gallon price_per_gallon money_available : ℕ) 
  (h1 : miles_per_gallon = 40) (h2 : price_per_gallon = 5) (h3 : money_available = 25) : 
  (money_available / price_per_gallon) * miles_per_gallon = 200 :=
by 
  /- The proof goes here -/
  sorry

end Margie_can_drive_200_miles_l211_211166


namespace log_eighteen_fifteen_l211_211161

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_eighteen_fifteen (a b : ℝ) (h1 : Real.log 2 = a) (h2 : Real.log 3 = b) :
  log_base 18 15 = (b - a + 1) / (a + 2 * b) :=
by sorry

end log_eighteen_fifteen_l211_211161


namespace find_number_l211_211222

theorem find_number (x : ℝ) (h : 20 / x = 0.8) : x = 25 := 
by
  sorry

end find_number_l211_211222


namespace rationalize_denominator_ABC_value_l211_211178

def A := 11 / 4
def B := 5 / 4
def C := 5

theorem rationalize_denominator : 
  (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = A + B * Real.sqrt C :=
sorry

theorem ABC_value :
  A * B * C = 275 :=
sorry

end rationalize_denominator_ABC_value_l211_211178


namespace probability_red_or_white_l211_211063

noncomputable def total_marbles : ℕ := 50
noncomputable def blue_marbles : ℕ := 5
noncomputable def red_marbles : ℕ := 9
noncomputable def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

theorem probability_red_or_white : 
  (red_marbles + white_marbles) / total_marbles = 9 / 10 :=
by sorry

end probability_red_or_white_l211_211063


namespace figure_100_squares_l211_211934

theorem figure_100_squares : (∃ f : ℕ → ℕ, f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 ∧ f 100 = 30301) :=
  sorry

end figure_100_squares_l211_211934


namespace cone_volume_l211_211783

theorem cone_volume (d : ℝ) (h : ℝ) (π : ℝ) (volume : ℝ) 
  (hd : d = 10) (hh : h = 0.8 * d) (hπ : π = Real.pi) : 
  volume = (200 / 3) * π :=
by
  sorry

end cone_volume_l211_211783


namespace base_10_to_base_5_l211_211229

noncomputable def base_five_equivalent (n : ℕ) : ℕ :=
  let (d1, r1) := div_mod n (5 * 5 * 5) in
  let (d2, r2) := div_mod r1 (5 * 5) in
  let (d3, r3) := div_mod r2 5 in
  let (d4, r4) := div_mod r3 1 in
  d1 * 1000 + d2 * 100 + d3 * 10 + d4

theorem base_10_to_base_5 : base_five_equivalent 156 = 1111 :=
by
  -- Include the proof here
  sorry

end base_10_to_base_5_l211_211229


namespace polynomial_coefficient_l211_211056

theorem polynomial_coefficient :
  ∀ d : ℝ, (2 * (2 : ℝ)^4 + 3 * (2 : ℝ)^3 + d * (2 : ℝ)^2 - 4 * (2 : ℝ) + 15 = 0) ↔ (d = -15.75) :=
by
  sorry

end polynomial_coefficient_l211_211056


namespace total_team_cost_l211_211208

-- Define the costs of individual items and the number of players
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.2
def socks_cost : ℝ := 6.8
def num_players : ℕ := 16

-- Define the total cost for equipment for one player
def player_cost : ℝ :=
  jersey_cost + shorts_cost + socks_cost

-- The main theorem stating the total cost for all players
theorem total_team_cost : num_players * player_cost = 752 := by
  sorry

end total_team_cost_l211_211208


namespace workdays_ride_l211_211212

-- Define the conditions
def work_distance : ℕ := 20
def weekend_ride : ℕ := 200
def speed : ℕ := 25
def hours_per_week : ℕ := 16

-- Define the question
def total_distance : ℕ := speed * hours_per_week
def distance_during_workdays : ℕ := total_distance - weekend_ride
def round_trip_distance : ℕ := 2 * work_distance

theorem workdays_ride : 
  (distance_during_workdays / round_trip_distance) = 5 :=
by
  sorry

end workdays_ride_l211_211212


namespace prime_factorization_2020_prime_factorization_2021_l211_211319

theorem prime_factorization_2020 : 2020 = 2^2 * 5 * 101 := by
  sorry

theorem prime_factorization_2021 : 2021 = 43 * 47 := by
  sorry

end prime_factorization_2020_prime_factorization_2021_l211_211319


namespace expression_to_diophantine_l211_211174

theorem expression_to_diophantine (x : ℝ) (y : ℝ) (n : ℕ) :
  (∃ (A B : ℤ), (x - y) ^ (2 * n + 1) = (A * x - B * y) ∧ (1969 : ℤ) * A^2 - (1968 : ℤ) * B^2 = 1) :=
sorry

end expression_to_diophantine_l211_211174


namespace trajectory_curve_point_F_exists_l211_211957

noncomputable def curve_C := { p : ℝ × ℝ | (p.1 - 1/2)^2 + (p.2 - 1/2)^2 = 4 }

theorem trajectory_curve (M : ℝ × ℝ) (p : ℝ × ℝ) (q : ℝ × ℝ) :
    M = ((p.1 + q.1) / 2, (p.2 + q.2) / 2) → 
    p.1^2 + p.2^2 = 9 → 
    q.1^2 + q.2^2 = 9 →
    (p.1 - 1)^2 + (p.2 - 1)^2 > 0 → 
    (q.1 - 1)^2 + (q.2 - 1)^2 > 0 → 
    ((p.1 - 1) * (q.1 - 1) + (p.2 - 1) * (q.2 - 1) = 0) →
    (M.1 - 1/2)^2 + (M.2 - 1/2)^2 = 4 :=
sorry

theorem point_F_exists (E D : ℝ × ℝ) (F : ℝ × ℝ) (H : ℝ × ℝ) :
    E = (9/2, 1/2) → D = (1/2, 1/2) → F.2 = 1/2 → 
    (∃ t : ℝ, t ≠ 9/2 ∧ F.1 = t) →
    (H ∈ curve_C) →
    ((H.1 - 9/2)^2 + (H.2 - 1/2)^2) / ((H.1 - F.1)^2 + (H.2 - 1/2)^2) = 24 * (15 - 8 * H.1) / ((t^2 + 15/4) * (24)) :=
sorry

end trajectory_curve_point_F_exists_l211_211957


namespace boy_age_proof_l211_211951

theorem boy_age_proof (P X : ℕ) (hP : P = 16) (hcond : P - X = (P + 4) / 2) : X = 6 :=
by
  sorry

end boy_age_proof_l211_211951


namespace total_tagged_numbers_l211_211918

theorem total_tagged_numbers:
  let W := 200
  let X := W / 2
  let Y := X + W
  let Z := 400
  W + X + Y + Z = 1000 := by 
    sorry

end total_tagged_numbers_l211_211918


namespace train_pass_platform_in_correct_time_l211_211772

def length_of_train : ℝ := 2500
def time_to_cross_tree : ℝ := 90
def length_of_platform : ℝ := 1500

noncomputable def speed_of_train : ℝ := length_of_train / time_to_cross_tree
noncomputable def total_distance_to_cover : ℝ := length_of_train + length_of_platform
noncomputable def time_to_pass_platform : ℝ := total_distance_to_cover / speed_of_train

theorem train_pass_platform_in_correct_time :
  abs (time_to_pass_platform - 143.88) < 0.01 :=
sorry

end train_pass_platform_in_correct_time_l211_211772


namespace num_divisors_of_30_l211_211703

-- Define what it means to be a divisor of 30
def is_divisor (n : ℤ) : Prop :=
  n ≠ 0 ∧ 30 % n = 0

-- Define a function that counts the number of divisors of 30
def count_divisors (d : ℤ) : ℕ :=
  ((Multiset.filter is_divisor ((List.range' (-30) 61).map (Int.ofNat))).attach.to_finset.card)

-- State the theorem
theorem num_divisors_of_30 : count_divisors 30 = 16 := sorry

end num_divisors_of_30_l211_211703


namespace least_number_to_add_l211_211760

-- Definition of LCM for given primes
def lcm_of_primes : ℕ := 5 * 7 * 11 * 13 * 17 * 19

theorem least_number_to_add (n : ℕ) : 
  (5432 + n) % 5 = 0 ∧ 
  (5432 + n) % 7 = 0 ∧ 
  (5432 + n) % 11 = 0 ∧ 
  (5432 + n) % 13 = 0 ∧ 
  (5432 + n) % 17 = 0 ∧ 
  (5432 + n) % 19 = 0 ↔ 
  n = 1611183 :=
by sorry

end least_number_to_add_l211_211760


namespace painting_time_l211_211564

theorem painting_time (karl_time leo_time : ℝ) (t : ℝ) (break_time : ℝ) : 
  karl_time = 6 → leo_time = 8 → break_time = 0.5 → 
  (1 / karl_time + 1 / leo_time) * (t - break_time) = 1 :=
by
  intros h_karl h_leo h_break
  rw [h_karl, h_leo, h_break]
  -- sorry to skip the proof
  sorry

end painting_time_l211_211564


namespace map_representation_l211_211380

theorem map_representation (d1 d2 l1 l2 : ℕ)
  (h1 : l1 = 15)
  (h2 : d1 = 90)
  (h3 : l2 = 20) :
  d2 = 120 :=
by
  sorry

end map_representation_l211_211380


namespace total_arrangements_l211_211927

-- Given conditions
def teachers := 2
def students := 6
def group_size := 4  -- because each group consists of 1 teacher and 3 students

-- Required to prove
theorem total_arrangements : ((Finset.card (Finset.powersetLen 1 (Finset.range teachers))).card * 
                              (Finset.card (Finset.powersetLen 3 (Finset.range students))).card * 
                              1) = 40 := by
  sorry

end total_arrangements_l211_211927


namespace nonnegative_solutions_count_l211_211137

theorem nonnegative_solutions_count :
  {x : ℝ | x^2 = -6 * x ∧ 0 ≤ x}.card = 1 := 
sorry

end nonnegative_solutions_count_l211_211137


namespace juice_fraction_left_l211_211159

theorem juice_fraction_left (initial_juice : ℝ) (given_juice : ℝ) (remaining_juice : ℝ) : 
  initial_juice = 5 → given_juice = 18/4 → remaining_juice = initial_juice - given_juice → remaining_juice = 1/2 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  rw [h3]
  sorry

end juice_fraction_left_l211_211159


namespace intersect_A_B_when_a_1_subset_A_B_range_a_l211_211165

def poly_eqn (x : ℝ) : Prop := -x ^ 2 - 2 * x + 8 = 0

def sol_set_A : Set ℝ := {x | poly_eqn x}

def inequality (a x : ℝ) : Prop := a * x - 1 ≤ 0

def sol_set_B (a : ℝ) : Set ℝ := {x | inequality a x}

theorem intersect_A_B_when_a_1 :
  sol_set_A ∩ sol_set_B 1 = { -4 } :=
sorry

theorem subset_A_B_range_a (a : ℝ) :
  sol_set_A ⊆ sol_set_B a ↔ (-1 / 4 : ℝ) ≤ a ∧ a ≤ 1 / 2 :=
sorry
 
end intersect_A_B_when_a_1_subset_A_B_range_a_l211_211165


namespace average_salary_l211_211273

theorem average_salary (a b c d e : ℕ) (h1 : a = 8000) (h2 : b = 5000) (h3 : c = 16000) (h4 : d = 7000) (h5 : e = 9000) :
  (a + b + c + d + e) / 5 = 9000 :=
by
  sorry

end average_salary_l211_211273


namespace number_of_classes_l211_211726

theorem number_of_classes
  (p : ℕ) (s : ℕ) (t : ℕ) (c : ℕ)
  (hp : p = 2) (hs : s = 30) (ht : t = 360) :
  c = t / (p * s) :=
by
  simp [hp, hs, ht]
  sorry

end number_of_classes_l211_211726


namespace find_x_from_w_condition_l211_211474

theorem find_x_from_w_condition :
  ∀ (x u y z w : ℕ), 
  (x = u + 7) → 
  (u = y + 5) → 
  (y = z + 12) → 
  (z = w + 25) → 
  (w = 100) → 
  x = 149 :=
by intros x u y z w h1 h2 h3 h4 h5
   sorry

end find_x_from_w_condition_l211_211474


namespace sum_of_undefined_fractions_l211_211619

theorem sum_of_undefined_fractions (x₁ x₂ : ℝ) (h₁ : x₁^2 - 7*x₁ + 12 = 0) (h₂ : x₂^2 - 7*x₂ + 12 = 0) :
  x₁ + x₂ = 7 :=
sorry

end sum_of_undefined_fractions_l211_211619


namespace map_representation_l211_211377

theorem map_representation (d1 d2 l1 l2 : ℕ)
  (h1 : l1 = 15)
  (h2 : d1 = 90)
  (h3 : l2 = 20) :
  d2 = 120 :=
by
  sorry

end map_representation_l211_211377


namespace raven_current_age_l211_211715

variable (R P : ℕ) -- Raven's current age, Phoebe's current age
variable (h₁ : P = 10) -- Phoebe is currently 10 years old
variable (h₂ : R + 5 = 4 * (P + 5)) -- In 5 years, Raven will be 4 times as old as Phoebe

theorem raven_current_age : R = 55 := 
by
  -- h2: R + 5 = 4 * (P + 5)
  -- h1: P = 10
  sorry

end raven_current_age_l211_211715


namespace rabbits_initially_bought_l211_211481

theorem rabbits_initially_bought (R : ℕ) (h : ∃ (k : ℕ), R + 6 = 17 * k) : R = 28 :=
sorry

end rabbits_initially_bought_l211_211481


namespace Benny_and_Tim_have_47_books_together_l211_211651

/-
  Definitions and conditions:
  1. Benny_has_24_books : Benny has 24 books.
  2. Benny_gave_10_books_to_Sandy : Benny gave Sandy 10 books.
  3. Tim_has_33_books : Tim has 33 books.
  
  Goal:
  Prove that together Benny and Tim have 47 books.
-/

def Benny_has_24_books : ℕ := 24
def Benny_gave_10_books_to_Sandy : ℕ := 10
def Tim_has_33_books : ℕ := 33

def Benny_remaining_books : ℕ := Benny_has_24_books - Benny_gave_10_books_to_Sandy

def Benny_and_Tim_together : ℕ := Benny_remaining_books + Tim_has_33_books

theorem Benny_and_Tim_have_47_books_together :
  Benny_and_Tim_together = 47 := by
  sorry

end Benny_and_Tim_have_47_books_together_l211_211651


namespace geometric_series_first_term_l211_211648

theorem geometric_series_first_term (r : ℝ) (S : ℝ) (a : ℝ) (h_r : r = 1/4) (h_S : S = 80)
  (h_sum : S = a / (1 - r)) : a = 60 :=
by
  -- proof steps
  sorry

end geometric_series_first_term_l211_211648


namespace inequality_solution_set_l211_211820

theorem inequality_solution_set (a b : ℝ) (h1 : a = -2) (h2 : b = 1) :
  {x : ℝ | |2 * x + a| + |x - b| < 6} = {x : ℝ | -1 < x ∧ x < 3} :=
by
  sorry

end inequality_solution_set_l211_211820


namespace remainder_abc_mod9_l211_211853

open Nat

-- Define the conditions for the problem
variables (a b c : ℕ)

-- Assume conditions: a, b, c are non-negative and less than 9, and the given congruences
theorem remainder_abc_mod9 (h1 : a < 9) (h2 : b < 9) (h3 : c < 9)
  (h4 : (a + 3 * b + 2 * c) % 9 = 3)
  (h5 : (2 * a + 2 * b + 3 * c) % 9 = 6)
  (h6 : (3 * a + b + 2 * c) % 9 = 1) :
  (a * b * c) % 9 = 4 :=
sorry

end remainder_abc_mod9_l211_211853


namespace george_reels_per_day_l211_211972

theorem george_reels_per_day
  (days : ℕ := 5)
  (jackson_per_day : ℕ := 6)
  (jonah_per_day : ℕ := 4)
  (total_fishes : ℕ := 90) :
  (∃ george_per_day : ℕ, george_per_day = 8) :=
by
  -- Calculation steps are skipped here; they would need to be filled in for a complete proof.
  sorry

end george_reels_per_day_l211_211972


namespace linear_function_change_l211_211360

-- Define a linear function g
variable (g : ℝ → ℝ)

-- Define and assume the conditions
def linear_function (g : ℝ → ℝ) : Prop := ∀ x y, g (x + y) = g x + g y ∧ g (x - y) = g x - g y
def condition_g_at_points : Prop := g 3 - g (-1) = 20

-- Prove that g(10) - g(2) = 40
theorem linear_function_change (g : ℝ → ℝ) 
  (linear_g : linear_function g) 
  (cond_g : condition_g_at_points g) : 
  g 10 - g 2 = 40 :=
sorry

end linear_function_change_l211_211360


namespace min_x_plus_y_l211_211682

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) : x + y ≥ 16 :=
by
  sorry

end min_x_plus_y_l211_211682


namespace least_five_digit_congruent_to_six_mod_seventeen_l211_211464

theorem least_five_digit_congruent_to_six_mod_seventeen : ∃ x : ℕ, x ≥ 10000 ∧ x < 100000 ∧ x % 17 = 6 ∧ ∀ y : ℕ, y ≥ 10000 ∧ y < 100000 ∧ y % 17 = 6 → x ≤ y :=
  by
    sorry

end least_five_digit_congruent_to_six_mod_seventeen_l211_211464


namespace central_angle_of_sector_l211_211111

theorem central_angle_of_sector (r l : ℝ) (h1 : l + 2 * r = 4) (h2 : (1 / 2) * l * r = 1) : l / r = 2 :=
by
  -- The proof should be provided here
  sorry

end central_angle_of_sector_l211_211111


namespace map_scale_representation_l211_211425

theorem map_scale_representation :
  (∀ len_km_1 len_cm_1 len_cm_2 : ℕ, len_km_1 = 90 → len_cm_1 = 15 → len_cm_2 = 20 →
    let scale := len_km_1 / len_cm_1 in
    len_cm_2 * scale = 120)
  := sorry

end map_scale_representation_l211_211425


namespace converse_proposition_l211_211591

-- Define the propositions p and q
variables (p q : Prop)

-- State the problem as a theorem
theorem converse_proposition (p q : Prop) : (q → p) ↔ ¬p → ¬q ∧ ¬q → ¬p ∧ (p → q) := 
by 
  sorry

end converse_proposition_l211_211591


namespace goods_train_speed_l211_211642

def train_speed_km_per_hr (length_of_train length_of_platform time_to_cross : ℕ) : ℕ :=
  let total_distance := length_of_train + length_of_platform
  let speed_m_s := total_distance / time_to_cross
  speed_m_s * 36 / 10

-- Define the conditions given in the problem
def length_of_train : ℕ := 310
def length_of_platform : ℕ := 210
def time_to_cross : ℕ := 26

-- Define the target speed
def target_speed : ℕ := 72

-- The theorem proving the conclusion
theorem goods_train_speed :
  train_speed_km_per_hr length_of_train length_of_platform time_to_cross = target_speed := by
  sorry

end goods_train_speed_l211_211642


namespace percentage_of_girls_l211_211199

theorem percentage_of_girls (B G : ℕ) (h1 : B + G = 400) (h2 : B = 80) :
  (G * 100) / (B + G) = 80 :=
by sorry

end percentage_of_girls_l211_211199


namespace cost_price_l211_211492

theorem cost_price (MP SP C : ℝ) (h1 : MP = 112.5) (h2 : SP = 0.95 * MP) (h3 : SP = 1.25 * C) : 
  C = 85.5 :=
by
  sorry

end cost_price_l211_211492


namespace scientific_notation_l211_211847

theorem scientific_notation (n : ℤ) (hn : n = 12910000) : ∃ a : ℝ, (a = 1.291 ∧ hn = (a * 10^7).to_int) :=
by
  sorry

end scientific_notation_l211_211847


namespace smallest_percentage_owning_90_percent_money_l211_211560

theorem smallest_percentage_owning_90_percent_money
  (P M : ℝ)
  (h1 : 0.2 * P = 0.8 * M) :
  (∃ x : ℝ, x = 0.6 * P ∧ 0.9 * M <= (0.2 * P + (x - 0.2 * P))) :=
sorry

end smallest_percentage_owning_90_percent_money_l211_211560


namespace ratio_gold_to_green_horses_l211_211435

theorem ratio_gold_to_green_horses (blue_horses purple_horses green_horses gold_horses : ℕ)
    (h1 : blue_horses = 3)
    (h2 : purple_horses = 3 * blue_horses)
    (h3 : green_horses = 2 * purple_horses)
    (h4 : blue_horses + purple_horses + green_horses + gold_horses = 33) :
  gold_horses / gcd gold_horses green_horses = 1 / 6 :=
by
  sorry

end ratio_gold_to_green_horses_l211_211435


namespace mul_3_6_0_5_l211_211665

theorem mul_3_6_0_5 : 3.6 * 0.5 = 1.8 :=
by
  sorry

end mul_3_6_0_5_l211_211665


namespace fraction_of_paint_first_week_l211_211992

-- Definitions based on conditions
def total_paint := 360
def fraction_first_week (f : ℚ) : ℚ := f * total_paint
def paint_remaining_first_week (f : ℚ) : ℚ := total_paint - fraction_first_week f
def fraction_second_week (f : ℚ) : ℚ := (1 / 5) * paint_remaining_first_week f
def total_paint_used (f : ℚ) : ℚ := fraction_first_week f + fraction_second_week f
def total_paint_used_value := 104

-- Proof problem statement
theorem fraction_of_paint_first_week (f : ℚ) (h : total_paint_used f = total_paint_used_value) : f = 1 / 9 := 
sorry

end fraction_of_paint_first_week_l211_211992


namespace map_scale_representation_l211_211426

theorem map_scale_representation :
  (∀ len_km_1 len_cm_1 len_cm_2 : ℕ, len_km_1 = 90 → len_cm_1 = 15 → len_cm_2 = 20 →
    let scale := len_km_1 / len_cm_1 in
    len_cm_2 * scale = 120)
  := sorry

end map_scale_representation_l211_211426


namespace bag_ratio_l211_211749

noncomputable def ratio_of_costs : ℚ := 1 / 2

theorem bag_ratio :
  ∃ (shirt_cost shoes_cost total_cost bag_cost : ℚ),
    shirt_cost = 7 ∧
    shoes_cost = shirt_cost + 3 ∧
    total_cost = 2 * shirt_cost + shoes_cost ∧
    bag_cost = 36 - total_cost ∧
    bag_cost / total_cost = ratio_of_costs :=
sorry

end bag_ratio_l211_211749


namespace clothing_store_earnings_l211_211639

-- Definitions for the given conditions
def num_shirts : ℕ := 20
def num_jeans : ℕ := 10
def cost_per_shirt : ℕ := 10
def cost_per_jeans : ℕ := 2 * cost_per_shirt

-- Theorem statement
theorem clothing_store_earnings : 
  (num_shirts * cost_per_shirt + num_jeans * cost_per_jeans = 400) := 
sorry

end clothing_store_earnings_l211_211639


namespace ellipse_properties_l211_211674

theorem ellipse_properties :
  (∃ a e : ℝ, (∃ b c : ℝ, a^2 = 25 ∧ b^2 = 9 ∧ c^2 = a^2 - b^2 ∧ c = 4 ∧ e = c / a) ∧ a = 5 ∧ e = 4 / 5) :=
sorry

end ellipse_properties_l211_211674


namespace polygon_area_is_400_l211_211803

def Point : Type := (ℤ × ℤ)

def area_of_polygon (vertices : List Point) : ℤ := 
  -- Formula to calculate polygon area would go here
  -- As a placeholder, for now we return 400 since proof details aren't required
  400

theorem polygon_area_is_400 :
  area_of_polygon [(0,0), (20,0), (30,10), (20,20), (0,20), (10,10), (0,0)] = 400 := by
  -- Proof would go here
  sorry

end polygon_area_is_400_l211_211803


namespace sum_of_tens_and_ones_digits_of_seven_eleven_l211_211470

theorem sum_of_tens_and_ones_digits_of_seven_eleven :
  let n := (3 + 4) ^ 11 in 
  (let ones := n % 10 in
   let tens := (n / 10) % 10 in
   ones + tens = 7) := 
by sorry

end sum_of_tens_and_ones_digits_of_seven_eleven_l211_211470


namespace consecutive_even_product_6digit_l211_211937

theorem consecutive_even_product_6digit :
  ∃ (a b c : ℕ), 
  (a % 2 = 0) ∧ (b = a + 2) ∧ (c = a + 4) ∧ 
  (Nat.digits 10 (a * b * c)).length = 6 ∧ 
  (Nat.digits 10 (a * b * c)).head! = 2 ∧ 
  (Nat.digits 10 (a * b * c)).getLast! = 2 ∧ 
  (a * b * c = 287232) :=
by
  sorry

end consecutive_even_product_6digit_l211_211937


namespace knights_statements_count_l211_211739

-- Definitions for the conditions
variables (r l : ℕ) -- r is the number of knights, l is the number of liars
variables (hr : r ≥ 2) (hl : l ≥ 2)
variables (total_liar_statements : 2 * r * l = 230)

-- Theorem statement: Prove that the number of times "You are a knight!" was said is 526
theorem knights_statements_count : 
  let total_islanders := r + l in
  let total_statements := total_islanders * (total_islanders - 1) in
  let knight_statements := total_statements - 230 in
  knight_statements = 526 :=
by
  sorry

end knights_statements_count_l211_211739


namespace suzhou_metro_scientific_notation_l211_211020

theorem suzhou_metro_scientific_notation : 
  (∃(a : ℝ) (n : ℤ), 
    1 ≤ abs a ∧ abs a < 10 ∧ 15.6 * 10^9 = a * 10^n) → 
    (a = 1.56 ∧ n = 9) := 
by
  sorry

end suzhou_metro_scientific_notation_l211_211020


namespace non_empty_subsets_satisfying_criteria_l211_211541

theorem non_empty_subsets_satisfying_criteria :
  (∑ m in Finset.range 11 \ Finset.singleton 0, ((20 - 2 * (m - 1)) choose m)) = 2163 :=
by
  sorry

end non_empty_subsets_satisfying_criteria_l211_211541


namespace range_of_a_l211_211697

variable (a : ℝ)

def p (a : ℝ) : Prop := 3/2 < a ∧ a < 5/2
def q (a : ℝ) : Prop := 2 ≤ a ∧ a ≤ 4

theorem range_of_a (h₁ : ¬(p a ∧ q a)) (h₂ : p a ∨ q a) : (3/2 < a ∧ a < 2) ∨ (5/2 ≤ a ∧ a ≤ 4) :=
sorry

end range_of_a_l211_211697


namespace frank_initial_boxes_l211_211813

theorem frank_initial_boxes (filled left : ℕ) (h_filled : filled = 8) (h_left : left = 5) : 
  filled + left = 13 := by
  sorry

end frank_initial_boxes_l211_211813


namespace Edmund_earns_64_dollars_l211_211809

-- Conditions
def chores_per_week : Nat := 12
def pay_per_extra_chore : Nat := 2
def chores_per_day : Nat := 4
def weeks : Nat := 2
def days_per_week : Nat := 7

-- Goal
theorem Edmund_earns_64_dollars :
  let total_chores_without_extra := chores_per_week * weeks
  let total_chores_with_extra := chores_per_day * (days_per_week * weeks)
  let extra_chores := total_chores_with_extra - total_chores_without_extra
  let earnings := pay_per_extra_chore * extra_chores
  earnings = 64 :=
by
  sorry

end Edmund_earns_64_dollars_l211_211809


namespace gcd_of_36_and_60_is_12_l211_211888

theorem gcd_of_36_and_60_is_12 :
  Nat.gcd 36 60 = 12 :=
sorry

end gcd_of_36_and_60_is_12_l211_211888


namespace find_specific_M_in_S_l211_211727

section MatrixProgression

variable {R : Type*} [CommRing R]

-- Definition of arithmetic progression in a 2x2 matrix.
def is_arithmetic_progression (a b c d : R) : Prop :=
  ∃ r : R, b = a + r ∧ c = a + 2 * r ∧ d = a + 3 * r

-- Definition of set S.
def S : Set (Matrix (Fin 2) (Fin 2) R) :=
  { M | ∃ a b c d : R, M = ![![a, b], ![c, d]] ∧ is_arithmetic_progression a b c d }

-- Main problem statement
theorem find_specific_M_in_S (M : Matrix (Fin 2) (Fin 2) ℝ) (k : ℕ) :
  k > 1 → M ∈ S → ∃ (α : ℝ), (M = α • ![![1, 1], ![1, 1]] ∨ (M = α • ![![ -3, -1], ![1, 3]] ∧ Odd k)) :=
by
  sorry

end MatrixProgression

end find_specific_M_in_S_l211_211727


namespace data_set_properties_l211_211822

open Finset
open Real

theorem data_set_properties :
  let data := {4, 5, 12, 7, 11, 9, 8} in
  let sorted_data := sort data in
  let n := sorted_data.card in
  let mean := (sorted_data.sum : ℝ) / n in
  let median := sorted_data.val[n / 2] in
  let variance := (sorted_data.sum (λ x, (x - mean) ^ 2) / n) in
  mean = 8 ∧ median = 8 ∧ variance = 52 / 7 := 
by {
  sorry
}

end data_set_properties_l211_211822


namespace sign_of_c_l211_211690

/-
Define the context and conditions as Lean axioms.
-/

variables (a b c : ℝ)

-- Axiom: The sum of coefficients is less than zero
axiom h1 : a + b + c < 0

-- Axiom: The quadratic equation has no real roots, thus the discriminant is less than zero
axiom h2 : (b^2 - 4*a*c) < 0

/-
Formal statement of the proof problem:
-/

theorem sign_of_c : c < 0 :=
by
  -- We state that the proof of c < 0 follows from the given axioms
  sorry

end sign_of_c_l211_211690


namespace calculate_new_volume_l211_211291

noncomputable def volume_of_sphere_with_increased_radius
  (initial_surface_area : ℝ) (radius_increase : ℝ) : ℝ :=
  (4 / 3) * Real.pi * ((Real.sqrt (initial_surface_area / (4 * Real.pi)) + radius_increase) ^ 3)

theorem calculate_new_volume :
  volume_of_sphere_with_increased_radius 400 (2) = 2304 * Real.pi :=
by
  sorry

end calculate_new_volume_l211_211291


namespace total_votes_l211_211978

variable (T S R F V : ℝ)

-- Conditions
axiom h1 : T = S + 0.15 * V
axiom h2 : S = R + 0.05 * V
axiom h3 : R = F + 0.07 * V
axiom h4 : T + S + R + F = V
axiom h5 : T - 2500 - 2000 = S + 2500
axiom h6 : S + 2500 = R + 2000 + 0.05 * V

theorem total_votes : V = 30000 :=
sorry

end total_votes_l211_211978


namespace total_bottles_capped_in_10_minutes_l211_211755

-- Define the capacities per minute for the three machines
def machine_a_capacity : ℕ := 12
def machine_b_capacity : ℕ := machine_a_capacity - 2
def machine_c_capacity : ℕ := machine_b_capacity + 5

-- Define the total capping capacity for 10 minutes
def total_capacity_in_10_minutes (a b c : ℕ) : ℕ := a * 10 + b * 10 + c * 10

-- The theorem we aim to prove
theorem total_bottles_capped_in_10_minutes :
  total_capacity_in_10_minutes machine_a_capacity machine_b_capacity machine_c_capacity = 370 :=
by
  -- Directly use the capacities defined above
  sorry

end total_bottles_capped_in_10_minutes_l211_211755


namespace inequality_solution_l211_211197

def solution_set_inequality : Set ℝ := {x | x < -1/3 ∨ x > 1/2}

theorem inequality_solution (x : ℝ) : 
  (2 * x - 1) / (3 * x + 1) > 0 ↔ x ∈ solution_set_inequality :=
by 
  sorry

end inequality_solution_l211_211197


namespace students_taking_either_geometry_or_history_but_not_both_l211_211210

theorem students_taking_either_geometry_or_history_but_not_both
    (students_in_both : ℕ)
    (students_in_geometry : ℕ)
    (students_only_in_history : ℕ)
    (students_in_both_cond : students_in_both = 15)
    (students_in_geometry_cond : students_in_geometry = 35)
    (students_only_in_history_cond : students_only_in_history = 18) :
    (students_in_geometry - students_in_both + students_only_in_history = 38) :=
by
  sorry

end students_taking_either_geometry_or_history_but_not_both_l211_211210


namespace find_n_plus_c_l211_211224

variables (n c : ℝ)

-- Conditions from the problem
def line1 := ∀ (x y : ℝ), (x = 4) → (y = 11) → (y = n * x + 3)
def line2 := ∀ (x y : ℝ), (x = 4) → (y = 11) → (y = 5 * x + c)

theorem find_n_plus_c (h1 : line1 n)
                      (h2 : line2 c) :
  n + c = -7 := by
  sorry

end find_n_plus_c_l211_211224


namespace greatest_prime_factor_of_341_is_17_l211_211233

theorem greatest_prime_factor_of_341_is_17 : ∃ p : ℕ, prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, q ∣ 341 → prime q → q ≤ p :=
sorry

end greatest_prime_factor_of_341_is_17_l211_211233


namespace monotonicity_x_pow_2_over_3_l211_211026

noncomputable def x_pow_2_over_3 (x : ℝ) : ℝ := x^(2/3)

theorem monotonicity_x_pow_2_over_3 : ∀ x y : ℝ, 0 < x → x < y → x_pow_2_over_3 x < x_pow_2_over_3 y :=
by
  intros x y hx hxy
  sorry

end monotonicity_x_pow_2_over_3_l211_211026


namespace least_rice_l211_211042

variable (o r : ℝ)

-- Conditions
def condition_1 : Prop := o ≥ 8 + r / 2
def condition_2 : Prop := o ≤ 3 * r

-- The main theorem we want to prove
theorem least_rice (h1 : condition_1 o r) (h2 : condition_2 o r) : r ≥ 4 :=
sorry

end least_rice_l211_211042


namespace friend_gain_is_20_percent_l211_211064

noncomputable def original_cost : ℝ := 52325.58
noncomputable def loss_percentage : ℝ := 0.14
noncomputable def friend_selling_price : ℝ := 54000
noncomputable def friend_percentage_gain : ℝ :=
  ((friend_selling_price - (original_cost * (1 - loss_percentage))) / (original_cost * (1 - loss_percentage))) * 100

theorem friend_gain_is_20_percent :
  friend_percentage_gain = 20 := by
  sorry

end friend_gain_is_20_percent_l211_211064


namespace monoticity_f_max_min_f_l211_211797

noncomputable def f (x : ℝ) := Real.log (2 * x + 3) + x^2

theorem monoticity_f :
  (∀ x ∈ Ioo (-(3:ℝ)/2) (-1), f x > f (-1)) ∧ 
  (∀ x ∈ Ioo (-(1:ℝ)/2) (1/0), f x > f (1/0)) ∧ 
  (∀ x ∈ Ioo (-1) (-(1:ℝ)/2), f x < f (-1/2)) :=
sorry

theorem max_min_f :
  ∀ x ∈ Icc (0 : ℝ) (1), 
  f(1) = Real.log 5 + 1 ∧ 
  f(0) = Real.log 3 :=
sorry

end monoticity_f_max_min_f_l211_211797


namespace distance_between_intersections_l211_211597

theorem distance_between_intersections :
  let a := 3
  let b := 2
  let c := -7
  let x1 := (-1 + Real.sqrt 22) / 3
  let x2 := (-1 - Real.sqrt 22) / 3
  let distance := abs (x1 - x2)
  let p := 88  -- 2^2 * 22 = 88
  let q := 9   -- 3^2 = 9
  distance = 2 * Real.sqrt 22 / 3 →
  p - q = 79 :=
by
  sorry

end distance_between_intersections_l211_211597


namespace tom_sleep_hours_l211_211214

-- Define initial sleep hours and increase fraction
def initial_sleep_hours : ℕ := 6
def increase_fraction : ℚ := 1 / 3

-- Define the function to calculate increased sleep
def increased_sleep_hours (initial : ℕ) (fraction : ℚ) : ℚ :=
  initial * fraction

-- Define the function to calculate total sleep hours
def total_sleep_hours (initial : ℕ) (increased : ℚ) : ℚ :=
  initial + increased

-- Theorem stating Tom's total sleep hours per night after the increase
theorem tom_sleep_hours (initial : ℕ) (fraction : ℚ) (increased : ℚ) (total : ℚ) :
  initial = initial_sleep_hours →
  fraction = increase_fraction →
  increased = increased_sleep_hours initial fraction →
  total = total_sleep_hours initial increased →
  total = 8 :=
by
  intros h_init h_frac h_incr h_total
  rw [h_init, h_frac] at h_incr
  rw [h_init, h_incr] at h_total
  sorry

end tom_sleep_hours_l211_211214


namespace base6_divisibility_13_l211_211947

theorem base6_divisibility_13 (d : ℕ) (h : 0 ≤ d ∧ d ≤ 5) : (435 + 42 * d) % 13 = 0 ↔ d = 5 :=
by sorry

end base6_divisibility_13_l211_211947


namespace nonnegative_solutions_eq_one_l211_211132

theorem nonnegative_solutions_eq_one : 
  (finset.univ.filter (λ x : ℝ, x^2 = -6 * x ∧ 0 <= x)).card = 1 := 
by 
  sorry

end nonnegative_solutions_eq_one_l211_211132


namespace hash_four_times_l211_211800

noncomputable def hash (N : ℝ) : ℝ := 0.6 * N + 2

theorem hash_four_times (N : ℝ) : hash (hash (hash (hash N))) = 11.8688 :=
  sorry

end hash_four_times_l211_211800


namespace total_shirts_sold_l211_211288

theorem total_shirts_sold (p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 : ℕ) (h1 : p1 = 20) (h2 : p2 = 22) (h3 : p3 = 25)
(h4 : p4 + p5 + p6 + p7 + p8 + p9 + p10 = 133) (h5 : ((p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10) / 10) > 20)
: p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10 = 200 ∧ 10 = 10 := sorry

end total_shirts_sold_l211_211288


namespace trapezoid_EC_length_l211_211896

-- Define a trapezoid and its properties.
structure Trapezoid (A B C D : Type) :=
(base1 : ℝ) -- AB
(base2 : ℝ) -- CD
(diagonal_AC : ℝ) -- AC
(AB_eq_3CD : base1 = 3 * base2)
(AC_length : diagonal_AC = 15)
(E : Type) -- point of intersection of diagonals

-- Proof statement that length of EC is 15/4
theorem trapezoid_EC_length
  {A B C D E : Type}
  (t : Trapezoid A B C D)
  (E : Type)
  (intersection_E : E) :
  ∃ (EC : ℝ), EC = 15 / 4 :=
by
  have h1 : t.base1 = 3 * t.base2 := t.AB_eq_3CD
  have h2 : t.diagonal_AC = 15 := t.AC_length
  -- Use the given conditions to derive the length of EC
  sorry

end trapezoid_EC_length_l211_211896


namespace exists_directed_triangle_l211_211441

structure Tournament (V : Type) :=
  (edges : V → V → Prop)
  (complete : ∀ x y, x ≠ y → edges x y ∨ edges y x)
  (outdegree_at_least_one : ∀ x, ∃ y, edges x y)

theorem exists_directed_triangle {V : Type} [Fintype V] (T : Tournament V) :
  ∃ (a b c : V), T.edges a b ∧ T.edges b c ∧ T.edges c a := by
sorry

end exists_directed_triangle_l211_211441


namespace subset_implies_value_l211_211358

theorem subset_implies_value (a : ℝ) : (∀ x ∈ ({0, -a} : Set ℝ), x ∈ ({1, -1, 2 * a - 2} : Set ℝ)) → a = 1 := by
  sorry

end subset_implies_value_l211_211358


namespace map_length_representation_l211_211419

theorem map_length_representation (a b : ℕ) (h : a = 15 ∧ b = 90) : b * (20 / a) = 120 :=
by
  sorry

end map_length_representation_l211_211419


namespace john_needs_392_tanks_l211_211993

/- Variables representing the conditions -/
def small_balloons : ℕ := 5000
def medium_balloons : ℕ := 5000
def large_balloons : ℕ := 5000

def small_balloon_volume : ℕ := 20
def medium_balloon_volume : ℕ := 30
def large_balloon_volume : ℕ := 50

def helium_tank_capacity : ℕ := 1000
def hydrogen_tank_capacity : ℕ := 1200
def mixture_tank_capacity : ℕ := 1500

/- Mathematical calculations -/
def helium_volume : ℕ := small_balloons * small_balloon_volume
def hydrogen_volume : ℕ := medium_balloons * medium_balloon_volume
def mixture_volume : ℕ := large_balloons * large_balloon_volume

def helium_tanks : ℕ := (helium_volume + helium_tank_capacity - 1) / helium_tank_capacity
def hydrogen_tanks : ℕ := (hydrogen_volume + hydrogen_tank_capacity - 1) / hydrogen_tank_capacity
def mixture_tanks : ℕ := (mixture_volume + mixture_tank_capacity - 1) / mixture_tank_capacity

def total_tanks : ℕ := helium_tanks + hydrogen_tanks + mixture_tanks

theorem john_needs_392_tanks : total_tanks = 392 :=
by {
  -- calculation proof goes here
  sorry
}

end john_needs_392_tanks_l211_211993


namespace complementary_angle_l211_211828

theorem complementary_angle (α : ℝ) (h : α = 35 + 30 / 60) : 90 - α = 54 + 30 / 60 :=
by
  sorry

end complementary_angle_l211_211828


namespace sqrt_factorial_l211_211049

theorem sqrt_factorial : Real.sqrt (Real.ofNat (Nat.factorial 5) * Real.ofNat (Nat.factorial 5)) = 120 := 
by 
  sorry

end sqrt_factorial_l211_211049


namespace younger_person_age_l211_211868

theorem younger_person_age 
  (y e : ℕ)
  (h1 : e = y + 20)
  (h2 : e - 4 = 5 * (y - 4)) : 
  y = 9 := 
sorry

end younger_person_age_l211_211868


namespace evaluate_Q_at_2_l211_211040

-- Define the polynomial Q(x)
noncomputable def Q (x : ℚ) : ℚ := x^4 - 20 * x^2 + 16

-- Define the root condition of sqrt(3) + sqrt(7)
def root_condition (x : ℚ) : Prop := (x = ℚ.sqrt(3) + ℚ.sqrt(7))

-- Theorem statement
theorem evaluate_Q_at_2 : Q 2 = -32 :=
by
  -- Given conditions
  have h1 : degree Q = 4 := sorry,
  have h2 : leading_coeff Q = 1 := sorry,
  have h3 : root_condition (ℚ.sqrt(3) + ℚ.sqrt(7)) := sorry,
  -- Goal
  exact sorry

end evaluate_Q_at_2_l211_211040


namespace faye_homework_problems_left_l211_211311

-- Defining the problem conditions
def M : ℕ := 46
def S : ℕ := 9
def A : ℕ := 40

-- The statement to prove
theorem faye_homework_problems_left : M + S - A = 15 := by
  sorry

end faye_homework_problems_left_l211_211311


namespace min_value_of_a_l211_211531

theorem min_value_of_a (x y : ℝ) 
  (h1 : 0 < x) (h2 : 0 < y) 
  (h : ∀ x y, 0 < x → 0 < y → (x + y) * (1 / x + a / y) ≥ 9) :
  4 ≤ a :=
sorry

end min_value_of_a_l211_211531


namespace map_scale_representation_l211_211427

theorem map_scale_representation :
  (∀ len_km_1 len_cm_1 len_cm_2 : ℕ, len_km_1 = 90 → len_cm_1 = 15 → len_cm_2 = 20 →
    let scale := len_km_1 / len_cm_1 in
    len_cm_2 * scale = 120)
  := sorry

end map_scale_representation_l211_211427


namespace find_lesser_number_l211_211879

theorem find_lesser_number (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 8) : y = 26 :=
by sorry

end find_lesser_number_l211_211879


namespace map_length_representation_l211_211410

theorem map_length_representation :
  ∀ (length1 length2 km1 : ℝ), 
  length1 = 15 ∧ km1 = 90 ∧ length2 = 20 →
  (length2 * (km1 / length1) = 120) :=
by
  intros length1 length2 km1
  intro h
  cases h with hl1 h
  cases h with hkm1 hl2
  rw [hl1, hkm1, hl2]
  simp
  sorry

end map_length_representation_l211_211410


namespace distribution_problem_l211_211647

theorem distribution_problem (cards friends : ℕ) (h1 : cards = 7) (h2 : friends = 9) :
  (Nat.choose friends cards) * (Nat.factorial cards) = 181440 :=
by
  -- According to the combination formula and factorial definition
  -- We can insert specific values and calculations here, but as per the task requirements, 
  -- we are skipping the actual proof.
  sorry

end distribution_problem_l211_211647


namespace solve_gcd_problem_l211_211453

def gcd_problem : Prop :=
  gcd 153 119 = 17

theorem solve_gcd_problem : gcd_problem :=
  by
    sorry

end solve_gcd_problem_l211_211453


namespace probability_of_other_girl_l211_211149

theorem probability_of_other_girl (A B : Prop) (P : Prop → ℝ) 
    (hA : P A = 3 / 4) 
    (hAB : P (A ∧ B) = 1 / 4) : 
    P (B ∧ A) / P A = 1 / 3 := by 
  -- The proof is skipped using the sorry keyword.
  sorry

end probability_of_other_girl_l211_211149


namespace median_computation_l211_211980

noncomputable def length_of_median (A B C A1 P Q R : ℝ) : Prop :=
  let AB := 10
  let AC := 6
  let BC := Real.sqrt (AB^2 - AC^2)
  let A1C := 24 / 7
  let A1B := 32 / 7
  let QR := Real.sqrt (A1B^2 - A1C^2)
  let median_length := QR / 2
  median_length = 4 * Real.sqrt 7 / 7

theorem median_computation (A B C A1 P Q R : ℝ) :
  length_of_median A B C A1 P Q R := by
  sorry

end median_computation_l211_211980


namespace values_of_a_for_equation_l211_211318

theorem values_of_a_for_equation :
  ∃ S : Finset ℤ, (∀ a ∈ S, |3 * a + 7| + |3 * a - 5| = 12) ∧ S.card = 4 :=
by
  sorry

end values_of_a_for_equation_l211_211318


namespace unique_poly_degree_4_l211_211038

theorem unique_poly_degree_4 
  (Q : ℚ[X]) 
  (h_deg : Q.degree = 4) 
  (h_lead_coeff : Q.leading_coeff = 1)
  (h_root : Q.eval (sqrt 3 + sqrt 7) = 0) : 
  Q = X^4 - 12 * X^2 + 16 ∧ Q.eval 2 = 0 :=
by sorry

end unique_poly_degree_4_l211_211038


namespace minimum_k_l211_211608

theorem minimum_k (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℝ) :
  a 1 = (1/2) ∧ (∀ n, 2 * a (n + 1) + S n = 0) ∧ (∀ n, S n ≤ k) → k = (1/2) :=
sorry

end minimum_k_l211_211608


namespace ellipse_focus_m_eq_3_l211_211686

theorem ellipse_focus_m_eq_3 (m : ℝ) (h : m > 0) : 
  (∃ a c : ℝ, a = 5 ∧ c = 4 ∧ c^2 = a^2 - m^2)
  → m = 3 :=
by
  sorry

end ellipse_focus_m_eq_3_l211_211686


namespace right_triangle_x_value_l211_211152

variable (BM MA BC CA x h d : ℝ)

theorem right_triangle_x_value (BM MA BC CA x h d : ℝ)
  (h4 : BM + MA = BC + CA)
  (h5 : BM = x)
  (h6 : BC = h)
  (h7 : CA = d) :
  x = h * d / (2 * h + d) := 
sorry

end right_triangle_x_value_l211_211152


namespace find_A_and_B_l211_211860

theorem find_A_and_B (A : ℕ) (B : ℕ) (x y : ℕ) 
  (h1 : 1000 ≤ A ∧ A ≤ 9999) 
  (h2 : B = 10^5 * x + 10 * A + y) 
  (h3 : B = 21 * A)
  (h4 : x < 10) 
  (h5 : y < 10) : 
  A = 9091 ∧ B = 190911 :=
sorry

end find_A_and_B_l211_211860


namespace planes_count_l211_211172

-- Define the conditions as given in the problem.
def total_wings : ℕ := 90
def wings_per_plane : ℕ := 2

-- Define the number of planes calculation based on conditions.
def number_of_planes : ℕ := total_wings / wings_per_plane

-- Prove that the number of planes is 45.
theorem planes_count : number_of_planes = 45 :=
by 
  -- The proof steps are omitted as specified.
  sorry

end planes_count_l211_211172


namespace f_value_plus_deriv_l211_211529

noncomputable def f : ℝ → ℝ := sorry

-- Define the function f and its derivative at x = 1
axiom f_deriv_at_1 : deriv f 1 = 1 / 2

-- Define the value of the function f at x = 1
axiom f_value_at_1 : f 1 = 5 / 2

-- Prove that f(1) + f'(1) = 3
theorem f_value_plus_deriv : f 1 + deriv f 1 = 3 :=
by
  rw [f_value_at_1, f_deriv_at_1]
  norm_num

end f_value_plus_deriv_l211_211529


namespace nonnegative_solution_count_l211_211131

theorem nonnegative_solution_count : 
  ∃! x : ℝ, x^2 = -6 * x ∧ 0 ≤ x := 
begin
  sorry
end

end nonnegative_solution_count_l211_211131


namespace decrease_in_average_age_l211_211869

theorem decrease_in_average_age (original_avg_age : ℕ) (new_students_avg_age : ℕ) 
    (original_strength : ℕ) (new_students_strength : ℕ) 
    (h1 : original_avg_age = 40) (h2 : new_students_avg_age = 32) 
    (h3 : original_strength = 8) (h4 : new_students_strength = 8) : 
    (original_avg_age - ((original_strength * original_avg_age + new_students_strength * new_students_avg_age) / (original_strength + new_students_strength))) = 4 :=
by 
  sorry

end decrease_in_average_age_l211_211869


namespace intersection_distance_eq_l211_211594

theorem intersection_distance_eq (p q : ℕ) (h1 : p = 88) (h2 : q = 9) :
  p - q = 79 :=
by
  sorry

end intersection_distance_eq_l211_211594


namespace projected_revenue_increase_is_20_percent_l211_211578

noncomputable def projected_percentage_increase_of_revenue (R : ℝ) (actual_revenue : ℝ) (projected_revenue : ℝ) : ℝ :=
  (projected_revenue / R - 1) * 100

theorem projected_revenue_increase_is_20_percent (R : ℝ) (actual_revenue : ℝ) :
  actual_revenue = R * 0.75 →
  actual_revenue = (R * (1 + 20 / 100)) * 0.625 →
  projected_percentage_increase_of_revenue R ((R * (1 + 20 / 100))) = 20 :=
by
  intros h1 h2
  sorry

end projected_revenue_increase_is_20_percent_l211_211578


namespace steak_and_egg_meal_cost_is_16_l211_211561

noncomputable def steak_and_egg_cost (x : ℝ) := 
  (x + 14) / 2 + 0.20 * (x + 14) = 21

theorem steak_and_egg_meal_cost_is_16 (x : ℝ) (h : steak_and_egg_cost x) : x = 16 := 
by 
  sorry

end steak_and_egg_meal_cost_is_16_l211_211561


namespace washing_time_per_cycle_l211_211913

theorem washing_time_per_cycle
    (shirts pants sweaters jeans : ℕ)
    (items_per_cycle total_hours : ℕ)
    (h1 : shirts = 18)
    (h2 : pants = 12)
    (h3 : sweaters = 17)
    (h4 : jeans = 13)
    (h5 : items_per_cycle = 15)
    (h6 : total_hours = 3) :
    ((shirts + pants + sweaters + jeans) / items_per_cycle) * (total_hours * 60) / ((shirts + pants + sweaters + jeans) / items_per_cycle) = 45 := 
by
  sorry

end washing_time_per_cycle_l211_211913


namespace divisibility_56786730_polynomial_inequality_l211_211766

theorem divisibility_56786730 (m n : ℤ) : 56786730 ∣ m * n * (m^60 - n^60) :=
sorry

theorem polynomial_inequality (m n : ℤ) : m^5 + 3 * m^4 * n - 5 * m^3 * n^2 - 15 * m^2 * n^3 + 4 * m * n^4 + 12 * n^5 ≠ 33 :=
sorry

end divisibility_56786730_polynomial_inequality_l211_211766


namespace possible_rectangular_arrays_l211_211556

theorem possible_rectangular_arrays (n : ℕ) (h : n = 48) :
  ∃ (m k : ℕ), m * k = n ∧ 2 ≤ m ∧ 2 ≤ k :=
sorry

end possible_rectangular_arrays_l211_211556


namespace always_positive_inequality_l211_211873

theorem always_positive_inequality (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) :=
sorry

end always_positive_inequality_l211_211873


namespace map_length_represents_distance_l211_211414

theorem map_length_represents_distance :
  ∀ (s : ℝ), (15 * s = 90) → (20 * s = 120) :=
begin
  intros s h,
  sorry
end

end map_length_represents_distance_l211_211414


namespace fraction_is_square_l211_211517

theorem fraction_is_square (a b : ℕ) (hpos_a : a > 0) (hpos_b : b > 0) 
  (hdiv : (ab + 1) ∣ (a^2 + b^2)) :
  ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
sorry

end fraction_is_square_l211_211517


namespace workEfficiencyRatioProof_is_2_1_l211_211477

noncomputable def workEfficiencyRatioProof : Prop :=
  ∃ (A B : ℝ), 
  (1 / B = 21) ∧ 
  (1 / (A + B) = 7) ∧
  (A / B = 2)

theorem workEfficiencyRatioProof_is_2_1 : workEfficiencyRatioProof :=
  sorry

end workEfficiencyRatioProof_is_2_1_l211_211477


namespace eight_sided_die_divisible_by_48_l211_211914

/--
An eight-sided die is numbered from 1 to 8. When it is rolled, the product \( P \) of the seven numbers that are visible is always divisible by \( 48 \).
-/
theorem eight_sided_die_divisible_by_48 (f : Fin 8 → ℕ)
  (h : ∀ i, f i = i + 1) : 
  ∃ (P : ℕ), (∀ n, P = (∏ i in (Finset.univ.filter (λ j, j ≠ n)), f i)) ∧ (48 ∣ P) := 
by
  sorry

end eight_sided_die_divisible_by_48_l211_211914


namespace divisor_of_polynomial_l211_211812

theorem divisor_of_polynomial (a : ℤ) (h : ∀ x : ℤ, (x^2 - x + a) ∣ (x^13 + x + 180)) : a = 1 :=
sorry

end divisor_of_polynomial_l211_211812


namespace parallel_lines_m_condition_l211_211572

theorem parallel_lines_m_condition (m : ℝ) : 
  (∀ (x y : ℝ), (2 * x - m * y - 1 = 0) ↔ ((m - 1) * x - y + 1 = 0)) → m = 2 :=
by
  sorry

end parallel_lines_m_condition_l211_211572


namespace sum_of_areas_of_circles_l211_211877

theorem sum_of_areas_of_circles :
  (∑' n : ℕ, π * (9 / 16) ^ n) = π * (16 / 7) :=
by
  sorry

end sum_of_areas_of_circles_l211_211877


namespace inverse_proportional_l211_211016

theorem inverse_proportional (p q : ℝ) (k : ℝ) 
  (h1 : ∀ (p q : ℝ), p * q = k)
  (h2 : p = 25)
  (h3 : q = 6) 
  (h4 : q = 15) : 
  p = 10 := 
by
  sorry

end inverse_proportional_l211_211016


namespace sqrt_factorial_mul_self_eq_sqrt_factorial_mul_self_value_l211_211048

theorem sqrt_factorial_mul_self_eq :
  sqrt ((5!) * (5!)) = 5! :=
by sorry

theorem sqrt_factorial_mul_self_value :
  sqrt ((5!) * (5!)) = 120 :=
by {
  rw sqrt_factorial_mul_self_eq,
  norm_num,
  exact rfl,
  sorry
}

end sqrt_factorial_mul_self_eq_sqrt_factorial_mul_self_value_l211_211048


namespace matrix_and_line_l211_211523

open Matrix

noncomputable def eigen_matrix := λ (M : Matrix (Fin 2) (Fin 2) ℝ), 
  eigenvalues M = [8] ∧ has_eigenvector M 8 ![1, 1] 
  ∧ M.mul_vec ![-1, 2] = ![-2, 4]

noncomputable def mapped_line_eq := 
  ∀ (M : Matrix (Fin 2) (Fin 2) ℝ), eigen_matrix M →
  (λ (x y x' y' : ℝ), 
    M.mul_vec ![x, y] = ![x', y'] → (x' - 2 * y' = 4 ↔ x + 3 * y + 2 = 0))

theorem matrix_and_line (M : Matrix (Fin 2) (Fin 2) ℝ):
  eigen_matrix M →
  M = ![![6, 2], ![4, 4]] ∧ mapped_line_eq M :=
by
  sorry

end matrix_and_line_l211_211523


namespace some_number_value_l211_211550

theorem some_number_value (a : ℤ) (x1 x2 : ℤ)
  (h1 : x1 + a = 10) (h2 : x2 + a = -10) (h_sum : x1 + x2 = 20) : a = -10 :=
by
  sorry

end some_number_value_l211_211550


namespace local_maximum_no_global_maximum_equation_root_condition_l211_211332

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3) * Real.exp x
noncomputable def f' (x : ℝ) : ℝ := (x^2 + 2*x - 3) * Real.exp x

theorem local_maximum_no_global_maximum : (∃ x0 : ℝ, f' x0 = 0 ∧ (∀ x < x0, f x < f x0) ∧ (∀ x > x0, f x < f x0))
∧ (f 1 = -2 * Real.exp 1) 
∧ (∀ x : ℝ, ∃ b : ℝ, f x = b ∧ b > 6 * Real.exp (-3) → ¬(f x = f 1))
:= sorry

theorem equation_root_condition (b : ℝ) : (∃ x1 x2 x3 : ℝ, f x1 = b ∧ f x2 = b ∧ f x3 = b ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) 
→ (0 < b ∧ b < 6 * Real.exp (-3))
:= sorry

end local_maximum_no_global_maximum_equation_root_condition_l211_211332


namespace smallest_number_groups_l211_211761

theorem smallest_number_groups :
  ∃ x : ℕ, (∀ y : ℕ, (y % 12 = 0 ∧ y % 20 = 0 ∧ y % 6 = 0) → y ≥ x) ∧ 
           (x % 12 = 0 ∧ x % 20 = 0 ∧ x % 6 = 0) ∧ x = 60 :=
by
  sorry

end smallest_number_groups_l211_211761


namespace ordered_pairs_bound_l211_211729

variable (m n : ℕ) (a b : ℕ → ℝ)

theorem ordered_pairs_bound
  (h_m : m ≥ n)
  (h_n : n ≥ 2022)
  : (∃ (pairs : Finset (ℕ × ℕ)), 
      (∀ i j, (i, j) ∈ pairs → 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ |a i + b j - (i * j)| ≤ m) ∧
      pairs.card ≤ 3 * n * Real.sqrt (m * Real.log (n))) := 
  sorry

end ordered_pairs_bound_l211_211729


namespace twenty_is_80_percent_of_what_number_l211_211219

theorem twenty_is_80_percent_of_what_number : ∃ y : ℕ, (20 : ℚ) / y = 4 / 5 ∧ y = 25 := by
  sorry

end twenty_is_80_percent_of_what_number_l211_211219


namespace decreasing_function_condition_l211_211123

theorem decreasing_function_condition :
  (∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → x1 ≠ x2 → (x1 - x2) * ((1 / x1 - x1) - (1 / x2 - x2)) < 0) :=
by
  -- Proof outline goes here
  sorry

end decreasing_function_condition_l211_211123


namespace factor_expression_l211_211099

theorem factor_expression (b : ℚ) : 
  294 * b^3 + 63 * b^2 - 21 * b = 21 * b * (14 * b^2 + 3 * b - 1) :=
by 
  sorry

end factor_expression_l211_211099


namespace percentage_divisible_by_6_l211_211265

theorem percentage_divisible_by_6 : 
  let numbers_less_than_or_equal_to_120 := (list.range 120).map (λ x, x + 1) in
  let divisible_by_6 := numbers_less_than_or_equal_to_120.filter (λ x, x % 6 = 0) in
  let percent := (divisible_by_6.length : ℚ) / 120 * 100 in
  percent = 16.67 :=
by 
  sorry

end percentage_divisible_by_6_l211_211265


namespace sara_picked_peaches_l211_211747

def peaches_original : ℕ := 24
def peaches_now : ℕ := 61
def peaches_picked (p_o p_n : ℕ) : ℕ := p_n - p_o

theorem sara_picked_peaches : peaches_picked peaches_original peaches_now = 37 :=
by
  sorry

end sara_picked_peaches_l211_211747


namespace value_of_sum_plus_five_l211_211170

theorem value_of_sum_plus_five (a b : ℕ) (h : 4 * a^2 + 4 * b^2 + 8 * a * b = 100) :
  (a + b) + 5 = 10 :=
sorry

end value_of_sum_plus_five_l211_211170


namespace mike_chocolate_squares_l211_211990

theorem mike_chocolate_squares (M : ℕ) (h1 : 65 = 3 * M + 5) : M = 20 :=
by {
  -- proof of the theorem (not included as per instructions)
  sorry
}

end mike_chocolate_squares_l211_211990


namespace Carolina_Winning_Probability_Beto_Winning_Probability_Ana_Winning_Probability_l211_211649

section
  -- Define the types of participants and the colors
  inductive Participant
  | Ana | Beto | Carolina

  inductive Color
  | blue | green

  -- Define the strategies for each participant
  inductive Strategy
  | guessBlue | guessGreen | pass

  -- Probability calculations for each strategy
  def carolinaStrategyProbability : ℚ := 1 / 8
  def betoStrategyProbability : ℚ := 1 / 2
  def anaStrategyProbability : ℚ := 3 / 4

  -- Statements to prove the probabilities
  theorem Carolina_Winning_Probability :
    carolinaStrategyProbability = 1 / 8 :=
  sorry

  theorem Beto_Winning_Probability :
    betoStrategyProbability = 1 / 2 :=
  sorry

  theorem Ana_Winning_Probability :
    anaStrategyProbability = 3 / 4 :=
  sorry
end

end Carolina_Winning_Probability_Beto_Winning_Probability_Ana_Winning_Probability_l211_211649


namespace split_enthusiasts_into_100_sections_l211_211917

theorem split_enthusiasts_into_100_sections :
  ∃ (sections : Fin 100 → Set ℕ),
    (∀ i, sections i ≠ ∅) ∧
    (∀ i j, i ≠ j → sections i ∩ sections j = ∅) ∧
    (⋃ i, sections i) = {n : ℕ | n < 5000} :=
sorry

end split_enthusiasts_into_100_sections_l211_211917


namespace find_m_l211_211687

-- Definitions for the sets
def setA (x : ℝ) : Prop := -2 < x ∧ x < 8
def setB (m : ℝ) (x : ℝ) : Prop := 2 * m - 1 < x ∧ x < m + 3

-- Condition on the intersection
def intersection (m : ℝ) (a b : ℝ) (x : ℝ) : Prop := 2 * m - 1 < x ∧ x < m + 3 ∧ -2 < x ∧ x < 8

-- Theorem statement
theorem find_m (m a b : ℝ) (h₀ : b - a = 3) (h₁ : ∀ x, intersection m a b x ↔ (a < x ∧ x < b)) : m = -2 ∨ m = 1 :=
sorry

end find_m_l211_211687


namespace initial_number_of_fruits_l211_211786

theorem initial_number_of_fruits (oranges apples limes : ℕ) (h_oranges : oranges = 50)
  (h_apples : apples = 72) (h_oranges_limes : oranges = 2 * limes) (h_apples_limes : apples = 3 * limes) :
  (oranges + apples + limes) * 2 = 288 :=
by
  sorry

end initial_number_of_fruits_l211_211786


namespace problem_statement_l211_211329

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x^2 - 3) * Real.exp x

-- Conditions and conclusions
theorem problem_statement :
  (∃ x, is_local_max f x ∧ ∀ y, f y < f x) ∧
  (∀ b, (∀ x, f x = b → ∃! x (h : f x = b), (∃ x1 x2 x3, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3)) → 0 < b ∧ b < 6 * Real.exp (-3)) :=
by
  sorry

end problem_statement_l211_211329


namespace angle_measure_of_E_l211_211150

theorem angle_measure_of_E (E F G H : ℝ) 
  (h1 : E = 3 * F) 
  (h2 : E = 4 * G) 
  (h3 : E = 6 * H) 
  (h_sum : E + F + G + H = 360) : 
  E = 206 := 
by 
  sorry

end angle_measure_of_E_l211_211150


namespace course_choice_related_to_gender_l211_211446

def contingency_table (a b c d n : ℕ) : Prop :=
  n = a + b + c + d ∧
  a + b = 50 ∧
  c + d = 50 ∧
  a + c = 70 ∧
  b + d = 30

def chi_square_test (a b c d n : ℕ) : ℕ := 
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem course_choice_related_to_gender (a b c d n : ℕ) :
  contingency_table 40 10 30 20 100 →
  chi_square_test 40 10 30 20 100 > 3.841 :=
by
  intros h_table
  sorry

end course_choice_related_to_gender_l211_211446


namespace similar_triangles_l211_211289

theorem similar_triangles (y : ℝ) 
  (h₁ : 12 / y = 9 / 6) : y = 8 :=
by {
  -- solution here
  -- currently, we just provide the theorem statement as requested
  sorry
}

end similar_triangles_l211_211289


namespace solution_set_of_xf_l211_211955

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

theorem solution_set_of_xf (f : ℝ → ℝ) (hf_odd : is_odd_function f) (hf_one : f 1 = 0)
    (h_derivative : ∀ x > 0, (x * (deriv f x) - f x) / (x^2) > 0) :
    {x : ℝ | x * f x > 0} = {x : ℝ | x < -1 ∨ x > 1} :=
by
  sorry

end solution_set_of_xf_l211_211955


namespace greatest_sum_of_int_pairs_squared_eq_64_l211_211443

theorem greatest_sum_of_int_pairs_squared_eq_64 :
  ∃ (x y : ℤ), x^2 + y^2 = 64 ∧ (∀ (a b : ℤ), a^2 + b^2 = 64 → a + b ≤ 8) ∧ x + y = 8 :=
by 
  sorry

end greatest_sum_of_int_pairs_squared_eq_64_l211_211443


namespace greatest_prime_factor_of_341_l211_211255

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, nat.prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 341 → q ≤ p :=
begin
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { exact dvd.intro 17 rfl },
  { intros q hq,
    rcases hq with ⟨hq_prime, hq_dvd⟩,
    rw nat.dvd_prime at hq_dvd,
    { cases hq_dvd,
      { exact le_rfl },
      { linarith [hq_prime.ne_zero, nat.prime.ne_one nat.prime_19] } },
    { exact nat.prime_19 },
    { intro h,
      linarith [hq_prime.ne_zero, ne.symm h] } }
end

end greatest_prime_factor_of_341_l211_211255


namespace seedlings_planted_l211_211014

theorem seedlings_planted (x : ℕ) (h1 : 2 * x + x = 1200) : x = 400 :=
by {
  sorry
}

end seedlings_planted_l211_211014


namespace total_marbles_l211_211973

-- There are only red, blue, and yellow marbles
universe u
variable {α : Type u}

-- The ratio of red marbles to blue marbles to yellow marbles is \(2:3:4\)
variables {r b y T : ℕ}
variable (ratio_cond : 2 * y = 4 * r ∧ 3 * y = 4 * b)

-- There are 40 yellow marbles in the container
variable (yellow_cond : y = 40)

-- Prove the total number of marbles in the container is 90
theorem total_marbles (ratio_cond : 2 * y = 4 * r ∧ 3 * y = 4 * b) (yellow_cond : y = 40) :
  T = r + b + y → T = 90 :=
sorry

end total_marbles_l211_211973


namespace line_single_point_not_necessarily_tangent_l211_211778

-- Define a curve
def curve : Type := ℝ → ℝ

-- Define a line
def line (m b : ℝ) : curve := λ x => m * x + b

-- Define a point of intersection
def intersects_at (l : curve) (c : curve) (x : ℝ) : Prop :=
  l x = c x

-- Define the property of having exactly one common point
def has_single_intersection (l : curve) (c : curve) : Prop :=
  ∃ x, ∀ y ≠ x, l y ≠ c y

-- Define the tangent line property
def is_tangent (l : curve) (c : curve) (x : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ h, abs h < δ → abs ((c (x + h) - c x) / h - (l (x + h) - l x) / h) < ε

-- The proof statement: There exists a curve c and a line l such that l has exactly one intersection point with c, but l is not necessarily a tangent to c.
theorem line_single_point_not_necessarily_tangent :
  ∃ c : curve, ∃ l : curve, has_single_intersection l c ∧ ∃ x, ¬ is_tangent l c x :=
sorry

end line_single_point_not_necessarily_tangent_l211_211778


namespace prove_correct_statement_l211_211298

-- Define the conditions; we use the negation of incorrect statements
def condition1 (a b : ℝ) : Prop := a ≠ b → ¬((a - b > 0) → (a > 0 ∧ b > 0))
def condition2 (x : ℝ) : Prop := ¬(|x| > 0)
def condition4 (x : ℝ) : Prop := x ≠ 0 → (¬(∃ y, y = 1 / x))

-- Define the statement we want to prove as the correct one
def correct_statement (q : ℚ) : Prop := 0 - q = -q

-- The main theorem that combines conditions and proves the correct statement
theorem prove_correct_statement (a b : ℝ) (q : ℚ) :
  condition1 a b →
  condition2 a →
  condition4 a →
  correct_statement q :=
  by
  intros h1 h2 h4
  unfold correct_statement
  -- Proof goes here
  sorry

end prove_correct_statement_l211_211298


namespace circle_equation_through_intersections_l211_211103

theorem circle_equation_through_intersections 
  (h₁ : ∀ x y : ℝ, x^2 + y^2 + 6 * x - 4 = 0 ↔ x^2 + y^2 + 6 * y - 28 = 0)
  (h₂ : ∀ x y : ℝ, x - y - 4 = 0) : 
  ∃ x y : ℝ, (x - 1/2) ^ 2 + (y + 7 / 2) ^ 2 = 89 / 2 :=
by sorry

end circle_equation_through_intersections_l211_211103


namespace equipment_total_cost_l211_211205

-- Definition of costs for each item of equipment
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.20
def socks_cost : ℝ := 6.80

-- Number of players
def num_players : ℕ := 16

-- Total cost for one player
def total_cost_one_player : ℝ := jersey_cost + shorts_cost + socks_cost

-- Total cost for all players
def total_cost_all_players : ℝ := total_cost_one_player * num_players

-- Theorem to prove
theorem equipment_total_cost : total_cost_all_players = 752 := by
  sorry

end equipment_total_cost_l211_211205


namespace scientific_notation_equivalent_l211_211931

theorem scientific_notation_equivalent : ∃ a n, (3120000 : ℝ) = a * 10^n ∧ a = 3.12 ∧ n = 6 :=
by
  exists 3.12
  exists 6
  sorry

end scientific_notation_equivalent_l211_211931


namespace minimum_rows_required_l211_211777

theorem minimum_rows_required (n : ℕ) : (3 * n * (n + 1)) / 2 ≥ 150 ↔ n ≥ 10 := 
by
  sorry

end minimum_rows_required_l211_211777


namespace distribute_ways_l211_211141

/-- There are 5 distinguishable balls and 4 distinguishable boxes.
The total number of ways to distribute these balls into the boxes is 1024. -/
theorem distribute_ways : (4 : ℕ) ^ (5 : ℕ) = 1024 := by
  sorry

end distribute_ways_l211_211141


namespace map_length_representation_l211_211422

theorem map_length_representation (a b : ℕ) (h : a = 15 ∧ b = 90) : b * (20 / a) = 120 :=
by
  sorry

end map_length_representation_l211_211422


namespace max_range_f_plus_2g_l211_211586

noncomputable def max_val_of_f_plus_2g (f g : ℝ → ℝ) (hf : ∀ x, -3 ≤ f x ∧ f x ≤ 5) (hg : ∀ x, -4 ≤ g x ∧ g x ≤ 2) : ℝ :=
  9

theorem max_range_f_plus_2g (f g : ℝ → ℝ) (hf : ∀ x, -3 ≤ f x ∧ f x ≤ 5) (hg : ∀ x, -4 ≤ g x ∧ g x ≤ 2) :
  ∃ (a b : ℝ), (-3 ≤ a ∧ a ≤ 5) ∧ (-8 ≤ b ∧ b ≤ 4) ∧ b = 9 := 
sorry

end max_range_f_plus_2g_l211_211586


namespace max_abc_l211_211367

theorem max_abc : ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  (a * b + b * c = 518) ∧ 
  (a * b - a * c = 360) ∧ 
  (a * b * c = 1008) := 
by {
  -- Definitions of a, b, c satisfying the given conditions.
  -- Proof of the maximum value will be placed here (not required as per instructions).
  sorry
}

end max_abc_l211_211367


namespace angle_A_is_30_degrees_l211_211711

theorem angle_A_is_30_degrees
    (a b : ℝ)
    (B A : ℝ)
    (a_eq_4 : a = 4)
    (b_eq_4_sqrt2 : b = 4 * Real.sqrt 2)
    (B_eq_45 : B = Real.pi / 4) : 
    A = Real.pi / 6 := 
by 
    sorry

end angle_A_is_30_degrees_l211_211711


namespace spending_difference_l211_211798

-- Define the conditions
def spent_on_chocolate : ℤ := 7
def spent_on_candy_bar : ℤ := 2

-- The theorem to be proven
theorem spending_difference : (spent_on_chocolate - spent_on_candy_bar = 5) :=
by sorry

end spending_difference_l211_211798


namespace total_games_l211_211077

/-- Definition of the number of games Alyssa went to this year -/
def games_this_year : Nat := 11

/-- Definition of the number of games Alyssa went to last year -/
def games_last_year : Nat := 13

/-- Definition of the number of games Alyssa plans to go to next year -/
def games_next_year : Nat := 15

/-- Statement to prove the total number of games Alyssa will go to in all -/
theorem total_games : games_this_year + games_last_year + games_next_year = 39 := by
  -- A sorry placeholder to skip the proof
  sorry

end total_games_l211_211077


namespace map_scale_l211_211397

theorem map_scale (represents_15cm_as_km : 15 * 6 = 90) :
  20 * 6 = 120 :=
by
  sorry

end map_scale_l211_211397


namespace option_C_true_l211_211511

variable {a b : ℝ}

theorem option_C_true (h : a < b) : a / 3 < b / 3 := sorry

end option_C_true_l211_211511


namespace map_representation_l211_211376

theorem map_representation (d1 d2 l1 l2 : ℕ)
  (h1 : l1 = 15)
  (h2 : d1 = 90)
  (h3 : l2 = 20) :
  d2 = 120 :=
by
  sorry

end map_representation_l211_211376


namespace club_members_remainder_l211_211599

theorem club_members_remainder (N : ℕ) (h1 : 50 < N) (h2 : N < 80)
  (h3 : N % 5 = 0) (h4 : N % 8 = 0 ∨ N % 7 = 0) :
  N % 9 = 6 ∨ N % 9 = 7 := by
  sorry

end club_members_remainder_l211_211599


namespace min_value_le_one_l211_211694

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x
noncomputable def g (a : ℝ) : ℝ := a - a * Real.log a

theorem min_value_le_one (a : ℝ) (ha : a > 0) :
  (∀ x : ℝ, f x a ≥ g a) ∧ g a ≤ 1 := sorry

end min_value_le_one_l211_211694


namespace gigi_mushrooms_l211_211679

-- Define the conditions
def pieces_per_mushroom := 4
def kenny_pieces := 38
def karla_pieces := 42
def remaining_pieces := 8

-- Main theorem
theorem gigi_mushrooms : (kenny_pieces + karla_pieces + remaining_pieces) / pieces_per_mushroom = 22 :=
by
  sorry

end gigi_mushrooms_l211_211679


namespace probability_A_not_losing_l211_211173

variable (P_A_wins : ℝ)
variable (P_draw : ℝ)
variable (P_A_not_losing : ℝ)

theorem probability_A_not_losing 
  (h1 : P_A_wins = 0.3) 
  (h2 : P_draw = 0.5) 
  (h3 : P_A_not_losing = P_A_wins + P_draw) :
  P_A_not_losing = 0.8 :=
sorry

end probability_A_not_losing_l211_211173


namespace v2_correct_at_2_l211_211333

def poly (x : ℕ) : ℕ := x^5 + x^4 + 2 * x^3 + 3 * x^2 + 4 * x + 1

def horner_v2 (x : ℕ) : ℕ :=
  let v0 := 1
  let v1 := v0 * x + 4
  let v2 := v1 * x + 3
  v2

theorem v2_correct_at_2 : horner_v2 2 = 15 := by
  sorry

end v2_correct_at_2_l211_211333


namespace train_speed_l211_211072

theorem train_speed (length_m : ℕ) (time_s : ℕ) (length_km : ℝ) (time_hr : ℝ) 
(length_conversion : length_km = (length_m : ℝ) / 1000)
(time_conversion : time_hr = (time_s : ℝ) / 3600)
(speed : ℝ) (speed_formula : speed = length_km / time_hr) :
  length_m = 300 → time_s = 18 → speed = 60 :=
by
  intros h1 h2
  rw [h1, h2] at *
  simp [length_conversion, time_conversion, speed_formula]
  norm_num
  sorry

end train_speed_l211_211072


namespace map_distance_l211_211372

theorem map_distance (fifteen_cm_in_km : ℤ) (cm_to_km : ℕ): 
  fifteen_cm_in_km = 90 ∧ cm_to_km = 6 → 20 * cm_to_km = 120 := 
by 
  sorry

end map_distance_l211_211372


namespace greatest_prime_factor_341_l211_211244

theorem greatest_prime_factor_341 : ∃ p : ℕ, prime p ∧ p.factor 341 ∧ (∀ q : ℕ, prime q ∧ q.factor 341 → q ≤ p) ∧ p = 17 :=
by sorry

end greatest_prime_factor_341_l211_211244


namespace joan_remaining_oranges_l211_211160

def total_oranges_joan_picked : ℕ := 37
def oranges_sara_sold : ℕ := 10

theorem joan_remaining_oranges : total_oranges_joan_picked - oranges_sara_sold = 27 := by
  sorry

end joan_remaining_oranges_l211_211160


namespace arithmetic_progression_sum_geometric_progression_sum_l211_211113

-- Arithmetic Progression Problem
theorem arithmetic_progression_sum (d : ℚ) :
  let S_n (n : ℕ) := n * (1 + (n - 1) / 2 * d) in
  S_n 10 = 70 → 
  ∀ n, S_n n = n * (1 + (n - 1) / 2 * (4 / 3)) := 
by
  sorry

-- Geometric Progression Problem
theorem geometric_progression_sum (q : ℚ) :
  let a_n (n : ℕ) := 1 * q ^ (n - 1) in
  let S_n (n : ℕ) := (1 * (1 - q ^ n)) / (1 - q) in
  a_n 4 = 1 / 8 → 
  S_n n > 100 * a_n n → 
  n ≥ 7 :=
by
  sorry

end arithmetic_progression_sum_geometric_progression_sum_l211_211113


namespace find_x_such_that_custom_op_neg3_eq_one_l211_211348

def custom_op (x y : Int) : Int := x * y - 2 * (x + y)

theorem find_x_such_that_custom_op_neg3_eq_one :
  ∃ x : Int, custom_op x (-3) = 1 ∧ x = 1 :=
by
  use 1
  sorry

end find_x_such_that_custom_op_neg3_eq_one_l211_211348


namespace more_candidates_selected_l211_211351

theorem more_candidates_selected (total_a total_b selected_a selected_b : ℕ)
  (h1 : total_a = 8000)
  (h2 : total_b = 8000)
  (h3 : selected_a = 6 * total_a / 100)
  (h4 : selected_b = 7 * total_b / 100) :
  selected_b - selected_a = 80 :=
  sorry

end more_candidates_selected_l211_211351


namespace harold_catches_up_at_12_miles_l211_211430

/-- 
Proof Problem: Given that Adrienne starts walking from X to Y at 3 miles per hour and one hour later Harold starts walking from X to Y at 4 miles per hour, prove that Harold covers 12 miles when he catches up to Adrienne.
-/
theorem harold_catches_up_at_12_miles :
  (∀ (T : ℕ), (ad_distance : ℕ) = 3 * (T + 1) → (ha_distance : ℕ) = 4 * T → ad_distance = ha_distance) →
  (∃ T : ℕ, ha_distance = 12) :=
by
  sorry

end harold_catches_up_at_12_miles_l211_211430


namespace f_positive_l211_211342

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
noncomputable def f'' : ℝ → ℝ := sorry

axiom f_monotonically_decreasing : ∀ x y : ℝ, x < y → f x > f y
axiom inequality_condition : ∀ x : ℝ, (f x) / (f'' x) + x < 1

theorem f_positive : ∀ x : ℝ, f x > 0 :=
by sorry

end f_positive_l211_211342


namespace number_of_integers_satisfying_inequality_l211_211536

theorem number_of_integers_satisfying_inequality : 
  (finset.filter (λ x, (x - 2)^2 ≤ 4) (finset.Icc 0 4)).card = 5 := 
  sorry

end number_of_integers_satisfying_inequality_l211_211536


namespace smallest_12_digit_divisible_by_36_with_all_digits_l211_211315

/-- We want to prove that the smallest 12-digit natural number that is divisible by 36 
    and contains each digit from 0 to 9 at least once is 100023457896. -/
theorem smallest_12_digit_divisible_by_36_with_all_digits :
  ∃ n : ℕ, n = 100023457896 ∧ 
    (nat.digits 10 n).length = 12 ∧ 
    (∀ d ∈ (finset.range 10).val, d ∈ (nat.digits 10 n).val) ∧ 
    n % 36 = 0 :=
begin
  sorry
end

end smallest_12_digit_divisible_by_36_with_all_digits_l211_211315


namespace largest_vertex_sum_l211_211509

def parabola_vertex_sum (a T : ℤ) (hT : T ≠ 0) : ℤ :=
  let x_vertex := T
  let y_vertex := a * T^2 - 2 * a * T^2
  x_vertex + y_vertex

theorem largest_vertex_sum (a T : ℤ) (hT : T ≠ 0)
  (hA : 0 = a * 0^2 + 0 * 0 + 0)
  (hB : 0 = a * (2 * T)^2 + (2 * T) * (2 * -T))
  (hC : 36 = a * (2 * T + 1)^2 + (2 * T - 2 * T * (2 * T + 1)))
  : parabola_vertex_sum a T hT ≤ -14 :=
sorry

end largest_vertex_sum_l211_211509


namespace map_scale_l211_211401

namespace MapProblem

variables {c1 c2 k1 k2 : ℕ}

def representing_kilometers (c1 c2 k1 k2 : ℕ) : Prop :=
  (c1 * k2 = c2 * k1)

theorem map_scale :
  representing_kilometers 15 20 90 120 :=
by
  unfold representing_kilometers
  calc
    15 * 120 = 1800 := by norm_num
    20 * 90 = 1800 := by norm_num
  sorry

end MapProblem

end map_scale_l211_211401


namespace total_distance_covered_l211_211774

theorem total_distance_covered (h : ℝ) : (h > 0) → 
  ∑' n : ℕ, (h * (0.8 : ℝ) ^ n + h * (0.8 : ℝ) ^ (n + 1)) = 5 * h :=
  by
  sorry

end total_distance_covered_l211_211774


namespace time_to_run_100_meters_no_wind_l211_211646

-- Definitions based on the conditions
variables (v w : ℝ)
axiom speed_with_wind : v + w = 9
axiom speed_against_wind : v - w = 7

-- The theorem statement to prove
theorem time_to_run_100_meters_no_wind : (100 / v) = 12.5 :=
by 
  sorry

end time_to_run_100_meters_no_wind_l211_211646


namespace length_to_width_ratio_is_three_l211_211950

def rectangle_ratio (x : ℝ) : Prop :=
  let side_length_large_square := 4 * x
  let length_rectangle := 4 * x
  let width_rectangle := x
  length_rectangle / width_rectangle = 3

-- We state the theorem to be proved
theorem length_to_width_ratio_is_three (x : ℝ) (h : 0 < x) :
  rectangle_ratio x :=
sorry

end length_to_width_ratio_is_three_l211_211950


namespace two_colonies_reach_limit_l211_211636

noncomputable def bacteria_growth (n : ℕ) : ℕ := 2^n

theorem two_colonies_reach_limit (days : ℕ) (h : bacteria_growth days = (2^20)) : 
  bacteria_growth days = bacteria_growth 20 := 
by sorry

end two_colonies_reach_limit_l211_211636


namespace money_made_from_milk_sales_l211_211795

namespace BillMilkProblem

def total_gallons_milk : ℕ := 16
def fraction_for_sour_cream : ℚ := 1 / 4
def fraction_for_butter : ℚ := 1 / 4
def milk_to_sour_cream_ratio : ℕ := 2
def milk_to_butter_ratio : ℕ := 4
def price_per_gallon_butter : ℕ := 5
def price_per_gallon_sour_cream : ℕ := 6
def price_per_gallon_whole_milk : ℕ := 3

theorem money_made_from_milk_sales : ℕ :=
  let milk_for_sour_cream := (fraction_for_sour_cream * total_gallons_milk).toNat
  let milk_for_butter := (fraction_for_butter * total_gallons_milk).toNat
  let sour_cream_gallons := milk_for_sour_cream / milk_to_sour_cream_ratio
  let butter_gallons := milk_for_butter / milk_to_butter_ratio
  let milk_remaining := total_gallons_milk - milk_for_sour_cream - milk_for_butter
  let money_from_butter := butter_gallons * price_per_gallon_butter
  let money_from_sour_cream := sour_cream_gallons * price_per_gallon_sour_cream
  let money_from_whole_milk := milk_remaining * price_per_gallon_whole_milk
  money_from_butter + money_from_sour_cream + money_from_whole_milk = 41 :=
by
  sorry 

end BillMilkProblem

end money_made_from_milk_sales_l211_211795


namespace intersections_line_segment_l211_211496

def intersects_count (a b : ℕ) (x y : ℕ) : ℕ :=
  let steps := gcd x y
  2 * (steps + 1)

theorem intersections_line_segment (x y : ℕ) (h_x : x = 501) (h_y : y = 201) :
  intersects_count 1 1 x y = 336 := by
  sorry

end intersections_line_segment_l211_211496


namespace hillary_descending_rate_correct_l211_211960

-- Define the conditions in Lean
def base_to_summit := 5000 -- height from base camp to the summit
def departure_time := 6 -- departure time in hours after midnight (6:00)
def summit_time_hillary := 5 -- time taken by Hillary to reach 1000 ft short of the summit
def passing_time := 12 -- time when Hillary and Eddy pass each other (12:00)
def climb_rate_hillary := 800 -- Hillary's climbing rate in ft/hr
def climb_rate_eddy := 500 -- Eddy's climbing rate in ft/hr
def stop_short := 1000 -- distance short of the summit Hillary stops at

-- Define the correct answer based on the conditions
def descending_rate_hillary := 1000 -- Hillary's descending rate in ft/hr

-- Create the theorem to prove Hillary's descending rate
theorem hillary_descending_rate_correct (base_to_summit departure_time summit_time_hillary passing_time climb_rate_hillary climb_rate_eddy stop_short descending_rate_hillary : ℕ) :
  (descending_rate_hillary = 1000) :=
sorry

end hillary_descending_rate_correct_l211_211960


namespace map_length_representation_l211_211418

theorem map_length_representation (a b : ℕ) (h : a = 15 ∧ b = 90) : b * (20 / a) = 120 :=
by
  sorry

end map_length_representation_l211_211418


namespace determine_a_if_roots_are_prime_l211_211824
open Nat

theorem determine_a_if_roots_are_prime (a x1 x2 : ℕ) (h1 : Prime x1) (h2 : Prime x2) 
  (h_eq : x1^2 - x1 * a + a + 1 = 0) :
  a = 5 :=
by
  -- Placeholder for the proof
  sorry

end determine_a_if_roots_are_prime_l211_211824


namespace greatest_prime_factor_341_l211_211239

theorem greatest_prime_factor_341 : ∃ p, prime p ∧ p ∣ 341 ∧ p = 17 := 
by {
  use 17,
  split,
  { exact prime_of_factor 17, -- 17 is a prime number
  },
  {
    split,
    {
      norm_num,
      -- proof that 17 divides 341
      sorry,
    },
    -- p == 17
    refl,
  },
}

end greatest_prime_factor_341_l211_211239


namespace Ian_kept_1_rose_l211_211965

def initial_roses : ℕ := 20
def roses_given_to_mother : ℕ := 6
def roses_given_to_grandmother : ℕ := 9
def roses_given_to_sister : ℕ := 4
def total_roses_given : ℕ := roses_given_to_mother + roses_given_to_grandmother + roses_given_to_sister
def roses_kept (initial: ℕ) (given: ℕ) : ℕ := initial - given

theorem Ian_kept_1_rose :
  roses_kept initial_roses total_roses_given = 1 :=
by
  sorry

end Ian_kept_1_rose_l211_211965


namespace map_representation_l211_211383

theorem map_representation (length_1_cm length_2_cm : ℕ) (distance_1_km : ℕ) :
  length_1_cm = 15 → distance_1_km = 90 →
  length_2_cm = 20 →
  distance_2_km = length_2_cm * (distance_1_km / length_1_cm) →
  distance_2_km = 120 :=
by
  intros h_len1 h_distance1 h_len2 h_calculate 
  rw [h_len1, h_distance1, h_len2] at h_calculate
  have scale := 90 / 15
  have distance_2_km := 20 * scale
  have : scale = 6 := by norm_num
  rw [this] at h_calculate
  norm_num at distance_2_km
  assumption
  sorry

end map_representation_l211_211383


namespace map_length_representation_l211_211423

theorem map_length_representation (a b : ℕ) (h : a = 15 ∧ b = 90) : b * (20 / a) = 120 :=
by
  sorry

end map_length_representation_l211_211423


namespace minimum_sides_of_polygon_l211_211883

theorem minimum_sides_of_polygon (θ : ℝ) (hθ : θ = 25.5) : ∃ n : ℕ, n = 240 ∧ ∀ k : ℕ, (k * θ) % 360 = 0 → k = n := 
by
  -- The proof goes here
  sorry

end minimum_sides_of_polygon_l211_211883


namespace map_scale_l211_211402

namespace MapProblem

variables {c1 c2 k1 k2 : ℕ}

def representing_kilometers (c1 c2 k1 k2 : ℕ) : Prop :=
  (c1 * k2 = c2 * k1)

theorem map_scale :
  representing_kilometers 15 20 90 120 :=
by
  unfold representing_kilometers
  calc
    15 * 120 = 1800 := by norm_num
    20 * 90 = 1800 := by norm_num
  sorry

end MapProblem

end map_scale_l211_211402


namespace IvanPetrovich_daily_lessons_and_charity_l211_211062

def IvanPetrovichConditions (L k : ℕ) : Prop :=
  24 = 8 + 3*L + k ∧
  3000 * L * 21 + 14000 = 70000 + (7000 * k / 3)

theorem IvanPetrovich_daily_lessons_and_charity
  (L k : ℕ) (h : IvanPetrovichConditions L k) :
  L = 2 ∧ 7000 * k / 3 = 70000 := 
by
  sorry

end IvanPetrovich_daily_lessons_and_charity_l211_211062


namespace f_2015_l211_211122

noncomputable def f : ℝ → ℝ := sorry
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_periodic : ∀ x : ℝ, f (x - 2) = -f x
axiom f_initial_segment : ∀ x : ℝ, (-1 ≤ x ∧ x ≤ 0) → f x = 2^x

theorem f_2015 : f 2015 = 1 / 2 :=
by
  -- Proof goes here
  sorry

end f_2015_l211_211122


namespace no_integer_pairs_satisfy_equation_l211_211309

theorem no_integer_pairs_satisfy_equation :
  ∀ (a b : ℤ), a^3 + 3 * a^2 + 2 * a ≠ 125 * b^3 + 75 * b^2 + 15 * b + 2 :=
by
  intro a b
  sorry

end no_integer_pairs_satisfy_equation_l211_211309


namespace largest_divisor_of_product_l211_211915

-- Definition of factorial
def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

-- Definition of P, the product of the visible numbers when an 8-sided die is rolled
def P (excluded: ℕ) : ℕ :=
  factorial 8 / excluded

-- The main theorem to prove
theorem largest_divisor_of_product (excluded: ℕ) (h₁: 1 ≤ excluded) (h₂: excluded ≤ 8): 
  ∃ n, n = 192 ∧ ∀ k, k > 192 → ¬k ∣ P excluded :=
sorry

end largest_divisor_of_product_l211_211915


namespace sin_cos_fraction_l211_211323

theorem sin_cos_fraction (α : ℝ) (h1 : Real.sin α - Real.cos α = 1 / 5) (h2 : α ∈ Set.Ioo (-Real.pi / 2) (Real.pi / 2)) :
    Real.sin α * Real.cos α / (Real.sin α + Real.cos α) = 12 / 35 :=
by
  sorry

end sin_cos_fraction_l211_211323


namespace abs_diff_61st_term_l211_211044

-- Define sequences C and D
def seqC (n : ℕ) : ℤ := 20 + 15 * (n - 1)
def seqD (n : ℕ) : ℤ := 20 - 15 * (n - 1)

-- Prove the absolute value of the difference between the 61st terms is 1800
theorem abs_diff_61st_term : (abs (seqC 61 - seqD 61) = 1800) :=
by
  sorry

end abs_diff_61st_term_l211_211044


namespace sum_of_digits_7_pow_11_l211_211469

/-
  Let's define the problem with the given conditions and expected answer.
  We need to prove that the sum of the tens digit and the ones digit of 7^11 is 7.
-/

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_7_pow_11 : tens_digit (7 ^ 11) + ones_digit (7 ^ 11) = 7 :=
sorry

end sum_of_digits_7_pow_11_l211_211469


namespace alpha_beta_expression_l211_211815

noncomputable theory

variables (α β : ℝ)

-- Conditions
axiom root_equation : ∀ x, (x = α ∨ x = β) → x^2 - x - 1 = 0
axiom alpha_square : α^2 = α + 1
axiom alpha_beta_sum : α + β = 1

-- The statement
theorem alpha_beta_expression : α^4 + 3 * β = 5 :=
sorry

end alpha_beta_expression_l211_211815


namespace winning_percentage_l211_211757

noncomputable def total_votes (votes_winner votes_margin : ℕ) : ℕ :=
  votes_winner + (votes_winner - votes_margin)

noncomputable def percentage_votes (votes_winner total_votes : ℕ) : ℝ :=
  (votes_winner : ℝ) / (total_votes : ℝ) * 100

theorem winning_percentage
  (votes_winner : ℕ)
  (votes_margin : ℕ)
  (h_winner : votes_winner = 775)
  (h_margin : votes_margin = 300) :
  percentage_votes votes_winner (total_votes votes_winner votes_margin) = 62 :=
sorry

end winning_percentage_l211_211757


namespace solution_of_inequality_system_l211_211198

theorem solution_of_inequality_system (x : ℝ) : 
  (x - 1 < 0) ∧ (x + 1 > 0) ↔ (-1 < x) ∧ (x < 1) := 
by sorry

end solution_of_inequality_system_l211_211198


namespace greatest_prime_factor_of_341_l211_211254

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, nat.prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 341 → q ≤ p :=
begin
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { exact dvd.intro 17 rfl },
  { intros q hq,
    rcases hq with ⟨hq_prime, hq_dvd⟩,
    rw nat.dvd_prime at hq_dvd,
    { cases hq_dvd,
      { exact le_rfl },
      { linarith [hq_prime.ne_zero, nat.prime.ne_one nat.prime_19] } },
    { exact nat.prime_19 },
    { intro h,
      linarith [hq_prime.ne_zero, ne.symm h] } }
end

end greatest_prime_factor_of_341_l211_211254


namespace measure_85_liters_l211_211490

theorem measure_85_liters (C1 C2 C3 : ℕ) (capacity : ℕ) : 
  (C1 = 0 ∧ C2 = 0 ∧ C3 = 1 ∧ capacity = 85) → 
  (∃ weighings : ℕ, weighings ≤ 8 ∧ C1 = 85 ∨ C2 = 85 ∨ C3 = 85) :=
by 
  sorry

end measure_85_liters_l211_211490


namespace dodecagon_diagonals_l211_211775

def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem dodecagon_diagonals : num_diagonals 12 = 54 :=
by
  -- by sorry means we skip the actual proof
  sorry

end dodecagon_diagonals_l211_211775


namespace map_length_represents_distance_l211_211412

theorem map_length_represents_distance :
  ∀ (s : ℝ), (15 * s = 90) → (20 * s = 120) :=
begin
  intros s h,
  sorry
end

end map_length_represents_distance_l211_211412


namespace domain_of_log_l211_211436

def log_domain := {x : ℝ | x > 1}

theorem domain_of_log : {x : ℝ | ∃ y, y = log_domain} = {x : ℝ | x > 1} :=
by
  sorry

end domain_of_log_l211_211436


namespace solve_for_n_l211_211050

theorem solve_for_n (n : ℚ) (h : n + (n + 1) + (n + 2) + (n + 3) = 20) : 
    n = 3.5 :=
  sorry

end solve_for_n_l211_211050


namespace ramon_twice_loui_age_in_future_l211_211155

theorem ramon_twice_loui_age_in_future : 
  ∀ (x : ℕ), 
  (∀ t : ℕ, t = 23 → 
            t * 2 = 46 → 
            ∀ r : ℕ, r = 26 → 
                      26 + x = 46 → 
                      x = 20) := 
by sorry

end ramon_twice_loui_age_in_future_l211_211155


namespace inequality_solution_l211_211503

theorem inequality_solution :
  {x : ℝ | (x - 3) * (x + 2) ≠ 0 ∧ (x^2 + 1) / ((x - 3) * (x + 2)) ≥ 0} = 
  {x : ℝ | x ≤ -2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end inequality_solution_l211_211503


namespace base_five_of_156_is_1111_l211_211227

def base_five_equivalent (n : ℕ) : ℕ := sorry

theorem base_five_of_156_is_1111 :
  base_five_equivalent 156 = 1111 :=
sorry

end base_five_of_156_is_1111_l211_211227


namespace range_of_m_l211_211527

noncomputable def f (m x : ℝ) : ℝ := x^3 + m * x^2 + (m + 6) * x + 1

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, ∃ c : ℝ, (f m x) ≤ c ∧ (f m y) ≥ (f m x) ∧ ∀ z : ℝ, f m z ≥ f m x ∧ f m z ≤ c) ↔ (m < -3 ∨ m > 6) :=
by
  sorry

end range_of_m_l211_211527


namespace exact_time_now_l211_211354

/-- Given that it is between 9:00 and 10:00 o'clock,
and nine minutes from now, the minute hand of a watch
will be exactly opposite the place where the hour hand
was six minutes ago, show that the exact time now is 9:06
-/
theorem exact_time_now 
  (t : ℕ)
  (h1 : t < 60)
  (h2 : ∃ t, 6 * (t + 9) - (270 + 0.5 * (t - 6)) = 180 ∨ 6 * (t + 9) - (270 + 0.5 * (t - 6)) = -180) :
  t = 6 := 
sorry

end exact_time_now_l211_211354


namespace remainder_250_div_k_l211_211106

theorem remainder_250_div_k {k : ℕ} (h1 : 0 < k) (h2 : 180 % (k * k) = 12) : 250 % k = 10 := by
  sorry

end remainder_250_div_k_l211_211106


namespace mouse_jump_vs_grasshopper_l211_211437

-- Definitions for jumps
def grasshopper_jump : ℕ := 14
def frog_jump : ℕ := grasshopper_jump + 37
def mouse_jump : ℕ := frog_jump - 16

-- Theorem stating the result
theorem mouse_jump_vs_grasshopper : mouse_jump - grasshopper_jump = 21 :=
by
  -- Skip the proof
  sorry

end mouse_jump_vs_grasshopper_l211_211437


namespace percentage_of_integers_divisible_by_6_up_to_120_l211_211266

theorem percentage_of_integers_divisible_by_6_up_to_120 : 
  let total := 120
      divisible_by_6 := λ n, n % 6 = 0
      count := (list.range (total + 1)).countp divisible_by_6
      percentage := (count.toFloat / total.toFloat) * 100
  in percentage = 16.67 :=
by
  sorry

end percentage_of_integers_divisible_by_6_up_to_120_l211_211266


namespace zoo_feeding_ways_l211_211076

-- Noncomputable is used for definitions that are not algorithmically computable
noncomputable def numFeedingWays : Nat :=
  4 * 3 * 3 * 2 * 2

theorem zoo_feeding_ways :
  ∀ (pairs : Fin 4 → (String × String)), -- Representing pairs of animals
  numFeedingWays = 144 :=
by
  sorry

end zoo_feeding_ways_l211_211076


namespace greatest_prime_factor_341_l211_211238

theorem greatest_prime_factor_341 : ∃ p : ℕ, Nat.Prime p ∧ p = 17 ∧ p = Nat.gcd 341 (Nat.gcd 341 (Nat.gcd 341 341)) :=
by
  sorry

end greatest_prime_factor_341_l211_211238


namespace tom_sleep_increase_l211_211217

theorem tom_sleep_increase :
  ∀ (initial_sleep : ℕ) (increase_by : ℚ), 
  initial_sleep = 6 → 
  increase_by = 1/3 → 
  initial_sleep + increase_by * initial_sleep = 8 :=
by 
  intro initial_sleep increase_by h1 h2
  simp [*, add_mul, mul_comm]
  sorry

end tom_sleep_increase_l211_211217


namespace vertex_on_x_axis_l211_211209

theorem vertex_on_x_axis (c : ℝ) : (∃ (h : ℝ), (h, 0) = ((-(-8) / (2 * 1)), c - (-8)^2 / (4 * 1))) → c = 16 :=
by
  sorry

end vertex_on_x_axis_l211_211209


namespace map_representation_l211_211386

theorem map_representation (length_1_cm length_2_cm : ℕ) (distance_1_km : ℕ) :
  length_1_cm = 15 → distance_1_km = 90 →
  length_2_cm = 20 →
  distance_2_km = length_2_cm * (distance_1_km / length_1_cm) →
  distance_2_km = 120 :=
by
  intros h_len1 h_distance1 h_len2 h_calculate 
  rw [h_len1, h_distance1, h_len2] at h_calculate
  have scale := 90 / 15
  have distance_2_km := 20 * scale
  have : scale = 6 := by norm_num
  rw [this] at h_calculate
  norm_num at distance_2_km
  assumption
  sorry

end map_representation_l211_211386


namespace ratio_not_necessarily_constant_l211_211603

theorem ratio_not_necessarily_constant (x y : ℝ) : ¬ (∃ k : ℝ, ∀ x y, x / y = k) :=
by
  sorry

end ratio_not_necessarily_constant_l211_211603


namespace mean_of_two_equals_mean_of_three_l211_211598

theorem mean_of_two_equals_mean_of_three (z : ℚ) : 
  (5 + 10 + 20) / 3 = (15 + z) / 2 → 
  z = 25 / 3 := 
by 
  sorry

end mean_of_two_equals_mean_of_three_l211_211598


namespace six_digit_product_of_consecutive_even_integers_l211_211944

theorem six_digit_product_of_consecutive_even_integers :
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ (a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0) ∧ a * b * c = 287232 :=
sorry

end six_digit_product_of_consecutive_even_integers_l211_211944


namespace emily_workers_needed_l211_211499

noncomputable def least_workers_needed
  (total_days : ℕ) (initial_days : ℕ) (total_workers : ℕ) (work_done : ℕ) (remaining_work : ℕ) (remaining_days : ℕ) :
  ℕ :=
  (remaining_work / remaining_days) / (work_done / initial_days / total_workers) * total_workers

theorem emily_workers_needed 
  (total_days : ℕ) (initial_days : ℕ) (total_workers : ℕ) (work_done : ℕ) (remaining_work : ℕ) (remaining_days : ℕ)
  (h1 : total_days = 40)
  (h2 : initial_days = 10)
  (h3 : total_workers = 12)
  (h4 : work_done = 40)
  (h5 : remaining_work = 60)
  (h6 : remaining_days = 30) :
  least_workers_needed total_days initial_days total_workers work_done remaining_work remaining_days = 6 := 
sorry

end emily_workers_needed_l211_211499


namespace no_2021_residents_possible_l211_211804

-- Definition: Each islander is either a knight or a liar
def is_knight_or_liar (i : ℕ) : Prop := true -- Placeholder definition for either being a knight or a liar

-- Definition: Knights always tell the truth
def knight_tells_truth (i : ℕ) : Prop := true -- Placeholder definition for knights telling the truth

-- Definition: Liars always lie
def liar_always_lies (i : ℕ) : Prop := true -- Placeholder definition for liars always lying

-- Definition: Even number of knights claimed by some islanders
def even_number_of_knights : Prop := true -- Placeholder definition for the claim of even number of knights

-- Definition: Odd number of liars claimed by remaining islanders
def odd_number_of_liars : Prop := true -- Placeholder definition for the claim of odd number of liars

-- Question and proof problem
theorem no_2021_residents_possible (K L : ℕ) (h1 : K + L = 2021) (h2 : ∀ i, is_knight_or_liar i) 
(h3 : ∀ k, knight_tells_truth k → even_number_of_knights) 
(h4 : ∀ l, liar_always_lies l → odd_number_of_liars) : 
  false := sorry

end no_2021_residents_possible_l211_211804


namespace letters_with_dot_no_straight_line_theorem_l211_211148

variables {D S : ℕ}
variables (letters_with_dot_and_straight_line : ℕ)
variables (letters_with_straight_line_no_dot : ℕ)
variables (total_letters : ℕ)

-- Conditions from the problem
def condition1 := letters_with_dot_and_straight_line
def condition2 := letters_with_straight_line_no_dot
def condition3 := total_letters

-- Define the quantity of interest
def letters_with_dot_no_straight_line : ℕ := D - letters_with_dot_and_straight_line

-- Theorem statement that needs to be proved
theorem letters_with_dot_no_straight_line_theorem
  (h1 : letters_with_dot_and_straight_line = 16)
  (h2 : letters_with_straight_line_no_dot = 30)
  (h3 : total_letters = 50) :
  letters_with_dot_no_straight_line = 4 :=
by { sorry }

end letters_with_dot_no_straight_line_theorem_l211_211148


namespace min_distance_PS_l211_211862

-- Definitions of the distances given in the problem
def PQ : ℝ := 12
def QR : ℝ := 7
def RS : ℝ := 5

-- Hypotheses for the problem
axiom h1 : PQ = 12
axiom h2 : QR = 7
axiom h3 : RS = 5

-- The goal is to prove that the minimum distance between P and S is 0.
theorem min_distance_PS : ∃ PS : ℝ, PS = 0 :=
by
  -- The proof is omitted
  sorry

end min_distance_PS_l211_211862


namespace map_distance_l211_211375

theorem map_distance (fifteen_cm_in_km : ℤ) (cm_to_km : ℕ): 
  fifteen_cm_in_km = 90 ∧ cm_to_km = 6 → 20 * cm_to_km = 120 := 
by 
  sorry

end map_distance_l211_211375


namespace fraction_simplest_form_iff_n_odd_l211_211314

theorem fraction_simplest_form_iff_n_odd (n : ℤ) :
  (Nat.gcd (3 * n + 10) (5 * n + 16) = 1) ↔ (n % 2 ≠ 0) :=
by sorry

end fraction_simplest_form_iff_n_odd_l211_211314


namespace find_x_l211_211548

theorem find_x (x : ℝ) : (1 + (1 / (1 + x)) = 2 * (1 / (1 + x))) → x = 0 :=
by
  intro h
  sorry

end find_x_l211_211548


namespace martha_blue_butterflies_l211_211010

variables (B Y : Nat)

theorem martha_blue_butterflies (h_total : B + Y + 5 = 11) (h_twice : B = 2 * Y) : B = 4 :=
by
  sorry

end martha_blue_butterflies_l211_211010


namespace Q_at_2_l211_211036

-- Define the polynomial Q(x)
def Q (x : ℚ) : ℚ := x^4 - 8 * x^2 + 16

-- Define the properties of Q(x)
def is_special_polynomial (P : (ℚ → ℚ)) : Prop := 
  degree P = 4 ∧ leading_coeff P = 1 ∧ P.is_root(√3 + √7)

-- Problem: Prove Q(2) = 0 given the polynomial Q meets the conditions.
theorem Q_at_2 (Q : ℚ → ℚ) (h1 : degree Q = 4) (h2 : leading_coeff Q = 1) (h3 : is_root Q (√3 + √7)) : 
  Q 2 = 0 := by
  sorry

end Q_at_2_l211_211036


namespace value_of_f_l211_211801

def B : Set ℚ := {x | x ≠ 0 ∧ x ≠ 1}

def f (x : ℚ) : ℝ := sorry

noncomputable def h (x : ℚ) : ℚ :=
  1 / (1 - x)

lemma cyclic_of_h :
  ∀ x ∈ B, h (h (h x)) = x :=
sorry

lemma functional_property (x : ℚ) (hx : x ∈ B) :
  f x + f (h x) = 2 * Real.log (|x|) :=
sorry

theorem value_of_f :
  f 2023 = Real.log 2023 :=
sorry

end value_of_f_l211_211801


namespace odd_function_of_power_l211_211692

noncomputable def f (a b x : ℝ) : ℝ := (a - 1) * x ^ b

theorem odd_function_of_power (a b : ℝ) (h : f a b a = 1/2) : 
  ∀ x : ℝ, f a b (-x) = -f a b x := 
by
  sorry

end odd_function_of_power_l211_211692


namespace find_y_intercept_l211_211193

theorem find_y_intercept (m : ℝ) (x_intercept : ℝ × ℝ) (hx : x_intercept = (4, 0)) (hm : m = -3) : ∃ y_intercept : ℝ × ℝ, y_intercept = (0, 12) := 
by
  sorry

end find_y_intercept_l211_211193


namespace magnitude_angle_between_vectors_l211_211127

def a : ℝ × ℝ := (1, real.sqrt 3)
def b : ℝ × ℝ := (real.sqrt 3, 1)

theorem magnitude_angle_between_vectors :
  let θ := real.arccos ((a.1 * b.1 + a.2 * b.2) / (real.sqrt (a.1^2 + a.2^2) * real.sqrt (b.1^2 + b.2^2))) in
  θ = real.pi / 6 :=
by 
  sorry

end magnitude_angle_between_vectors_l211_211127


namespace six_digit_number_consecutive_evens_l211_211939

theorem six_digit_number_consecutive_evens :
  ∃ n : ℕ,
    287232 = (2 * n - 2) * (2 * n) * (2 * n + 2) ∧
    287232 / 100000 = 2 ∧
    287232 % 10 = 2 :=
by
  sorry

end six_digit_number_consecutive_evens_l211_211939


namespace fish_remaining_l211_211092

theorem fish_remaining
  (initial_guppies : ℕ)
  (initial_angelfish : ℕ)
  (initial_tiger_sharks : ℕ)
  (initial_oscar_fish : ℕ)
  (sold_guppies : ℕ)
  (sold_angelfish : ℕ)
  (sold_tiger_sharks : ℕ)
  (sold_oscar_fish : ℕ)
  (initial_total : ℕ := initial_guppies + initial_angelfish + initial_tiger_sharks + initial_oscar_fish)
  (sold_total : ℕ := sold_guppies + sold_angelfish + sold_tiger_sharks + sold_oscar_fish)
  (remaining : ℕ := initial_total - sold_total) :
  initial_guppies = 94 →
  initial_angelfish = 76 →
  initial_tiger_sharks = 89 →
  initial_oscar_fish = 58 →
  sold_guppies = 30 →
  sold_angelfish = 48 →
  sold_tiger_sharks = 17 →
  sold_oscar_fish = 24 →
  remaining = 198 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  rw [h1, h2, h3, h4, h5, h6, h7, h8]
  norm_num
  sorry

end fish_remaining_l211_211092


namespace greatest_divisor_of_28_l211_211673

theorem greatest_divisor_of_28 : ∀ d : ℕ, d ∣ 28 → d ≤ 28 :=
by
  sorry

end greatest_divisor_of_28_l211_211673


namespace age_of_teacher_l211_211588

theorem age_of_teacher
    (n_students : ℕ)
    (avg_age_students : ℕ)
    (new_avg_age : ℕ)
    (n_total : ℕ)
    (H1 : n_students = 22)
    (H2 : avg_age_students = 21)
    (H3 : new_avg_age = avg_age_students + 1)
    (H4 : n_total = n_students + 1) :
    ((new_avg_age * n_total) - (avg_age_students * n_students) = 44) :=
by
    sorry

end age_of_teacher_l211_211588


namespace percentage_of_girls_taking_lunch_l211_211350

theorem percentage_of_girls_taking_lunch 
  (total_students : ℕ)
  (boys_ratio girls_ratio : ℕ)
  (boys_to_girls_ratio : boys_ratio + girls_ratio = 10)
  (boys : ℕ)
  (girls : ℕ)
  (boys_calc : boys = (boys_ratio * total_students) / 10)
  (girls_calc : girls = (girls_ratio * total_students) / 10)
  (boys_lunch_percentage : ℕ)
  (boys_lunch : ℕ)
  (boys_lunch_calc : boys_lunch = (boys_lunch_percentage * boys) / 100)
  (total_lunch_percentage : ℕ)
  (total_lunch : ℕ)
  (total_lunch_calc : total_lunch = (total_lunch_percentage * total_students) / 100)
  (girls_lunch : ℕ)
  (girls_lunch_calc : girls_lunch = total_lunch - boys_lunch) :
  ((girls_lunch * 100) / girls) = 40 :=
by 
  -- The proof can be filled in here
  sorry

end percentage_of_girls_taking_lunch_l211_211350


namespace person_dining_minutes_l211_211086

theorem person_dining_minutes
  (initial_angle : ℕ)
  (final_angle : ℕ)
  (time_spent : ℕ)
  (minute_angle_per_minute : ℕ)
  (hour_angle_per_minute : ℕ)
  (h1 : initial_angle = 110)
  (h2 : final_angle = 110)
  (h3 : minute_angle_per_minute = 6)
  (h4 : hour_angle_per_minute = minute_angle_per_minute / 12)
  (h5 : time_spent = (final_angle - initial_angle) / (minute_angle_per_minute / (minute_angle_per_minute / 12) - hour_angle_per_minute)) :
  time_spent = 40 := sorry

end person_dining_minutes_l211_211086


namespace least_five_digit_congruent_to_6_mod_17_l211_211461

theorem least_five_digit_congruent_to_6_mod_17 :
  ∃ (x : ℕ), 10000 ≤ x ∧ x ≤ 99999 ∧ x % 17 = 6 ∧
  ∀ (y : ℕ), 10000 ≤ y ∧ y ≤ 99999 ∧ y % 17 = 6 → x ≤ y := 
sorry

end least_five_digit_congruent_to_6_mod_17_l211_211461


namespace find_EC_l211_211895

variable (A B C D E: Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variable (AB: ℝ) (CD: ℝ) (AC: ℝ) (EC: ℝ)
variable [Parallel : ∀ A B, Prop] [Measure : ∀ A B, Real]

def is_trapezoid (AB: ℝ) (CD: ℝ) := AB = 3 * CD

theorem find_EC 
  (h1 : is_trapezoid AB CD)
  (h2 : AC = 15)
  : EC = 15 / 4 :=
by
  sorry

end find_EC_l211_211895


namespace a²_minus_b²_l211_211145

theorem a²_minus_b² (a b : ℝ) (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 := 
by
  sorry

end a²_minus_b²_l211_211145


namespace initial_inventory_correct_l211_211923

-- Define the conditions as given in the problem
def bottles_sold_monday : ℕ := 2445
def bottles_sold_tuesday : ℕ := 900
def bottles_sold_per_day_wed_to_sun : ℕ := 50
def days_wed_to_sun : ℕ := 5
def bottles_delivered_saturday : ℕ := 650
def final_inventory : ℕ := 1555

-- Define the total number of bottles sold during the week
def total_bottles_sold : ℕ :=
  bottles_sold_monday + bottles_sold_tuesday + (bottles_sold_per_day_wed_to_sun * days_wed_to_sun)

-- Define the initial inventory calculation
def initial_inventory : ℕ :=
  final_inventory + total_bottles_sold - bottles_delivered_saturday

-- The theorem we want to prove
theorem initial_inventory_correct :
  initial_inventory = 4500 :=
by
  sorry

end initial_inventory_correct_l211_211923


namespace abs_add_lt_abs_sub_l211_211008

-- Define the conditions
variables {a b : ℝ} (h1 : a * b < 0)

-- Prove the statement
theorem abs_add_lt_abs_sub (h1 : a * b < 0) : |a + b| < |a - b| := sorry

end abs_add_lt_abs_sub_l211_211008


namespace integer_values_of_a_l211_211101

theorem integer_values_of_a (x : ℤ) (a : ℤ)
  (h : x^3 + 3*x^2 + a*x + 11 = 0) :
  a = -155 ∨ a = -15 ∨ a = 13 ∨ a = 87 :=
sorry

end integer_values_of_a_l211_211101


namespace problem1_problem2_l211_211897

theorem problem1 (x : ℚ) (h : x - 2/11 = -1/3) : x = -5/33 :=
sorry

theorem problem2 : -2 - (-1/3 + 1/2) = -13/6 :=
sorry

end problem1_problem2_l211_211897


namespace sum_of_tens_and_ones_digits_of_seven_eleven_l211_211471

theorem sum_of_tens_and_ones_digits_of_seven_eleven :
  let n := (3 + 4) ^ 11 in 
  (let ones := n % 10 in
   let tens := (n / 10) % 10 in
   ones + tens = 7) := 
by sorry

end sum_of_tens_and_ones_digits_of_seven_eleven_l211_211471


namespace meeting_lamppost_l211_211274

-- Define the initial conditions of the problem
def lampposts : ℕ := 400
def start_alla : ℕ := 1
def start_boris : ℕ := 400
def meet_alla : ℕ := 55
def meet_boris : ℕ := 321

-- Define a theorem that we need to prove: Alla and Boris will meet at the 163rd lamppost
theorem meeting_lamppost : ∃ (n : ℕ), n = 163 := 
by {
  sorry -- Proof goes here
}

end meeting_lamppost_l211_211274


namespace difference_in_lengths_l211_211614

def speed_of_first_train := 60 -- in km/hr
def time_to_cross_pole_first_train := 3 -- in seconds
def speed_of_second_train := 90 -- in km/hr
def time_to_cross_pole_second_train := 2 -- in seconds

noncomputable def length_of_first_train : ℝ := (speed_of_first_train * (5 / 18)) * time_to_cross_pole_first_train
noncomputable def length_of_second_train : ℝ := (speed_of_second_train * (5 / 18)) * time_to_cross_pole_second_train

theorem difference_in_lengths : abs (length_of_second_train - length_of_first_train) = 0.01 :=
by
  -- The full proof would be placed here.
  sorry

end difference_in_lengths_l211_211614


namespace clothing_store_earnings_l211_211637

-- Defining the conditions
def num_shirts : ℕ := 20
def num_jeans : ℕ := 10
def shirt_cost : ℕ := 10
def jeans_cost : ℕ := 2 * shirt_cost

-- Statement of the problem
theorem clothing_store_earnings :
  num_shirts * shirt_cost + num_jeans * jeans_cost = 400 :=
by
  sorry

end clothing_store_earnings_l211_211637


namespace discount_per_bear_l211_211885

/-- Suppose the price of the first bear is $4.00 and Wally pays $354.00 for 101 bears.
 Prove that the discount per bear after the first bear is $0.50. -/
theorem discount_per_bear 
  (price_first : ℝ) (total_bears : ℕ) (total_paid : ℝ) (price_rest_bears : ℝ )
  (h1 : price_first = 4.0) (h2 : total_bears = 101) (h3 : total_paid = 354.0) : 
  (price_first + (total_bears - 1) * price_rest_bears - total_paid) / (total_bears - 1) = 0.50 :=
sorry

end discount_per_bear_l211_211885


namespace train_second_speed_20_l211_211645

variable (x v: ℕ)

theorem train_second_speed_20 
  (h1 : (x / 40) + (2 * x / v) = (6 * x / 48)) : 
  v = 20 := by 
  sorry

end train_second_speed_20_l211_211645


namespace boats_distance_one_minute_before_collision_l211_211449

theorem boats_distance_one_minute_before_collision :
  let speedA := 5  -- miles/hr
  let speedB := 21 -- miles/hr
  let initial_distance := 20 -- miles
  let combined_speed := speedA + speedB -- combined speed in miles/hr
  let speed_per_minute := combined_speed / 60 -- convert to miles/minute
  let time_to_collision := initial_distance / speed_per_minute -- time in minutes until collision
  initial_distance - (time_to_collision - 1) * speed_per_minute = 0.4333 :=
by
  sorry

end boats_distance_one_minute_before_collision_l211_211449


namespace percentage_of_600_eq_half_of_900_l211_211345

theorem percentage_of_600_eq_half_of_900 : 
  ∃ P : ℝ, (P / 100) * 600 = 0.5 * 900 ∧ P = 75 := by
  -- Proof goes here
  sorry

end percentage_of_600_eq_half_of_900_l211_211345


namespace simplify_complex_fraction_l211_211584

theorem simplify_complex_fraction : 
  ∀ (i : ℂ), 
  i^2 = -1 → 
  (2 - 2 * i) / (3 + 4 * i) = -(2 / 25 : ℝ) - (14 / 25) * i :=
by
  intros
  sorry

end simplify_complex_fraction_l211_211584


namespace least_five_digit_congruent_6_mod_17_l211_211459

theorem least_five_digit_congruent_6_mod_17 : ∃ n: ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 6 ∧ ∀ m: ℕ, 10000 ≤ m ∧ m < 100000 ∧ m % 17 = 6 → n ≤ m :=
sorry

end least_five_digit_congruent_6_mod_17_l211_211459


namespace alicia_tax_deduction_is_50_cents_l211_211787

def alicia_hourly_wage_dollars : ℝ := 25
def deduction_rate : ℝ := 0.02

def alicia_hourly_wage_cents : ℝ := alicia_hourly_wage_dollars * 100
def tax_deduction_cents : ℝ := alicia_hourly_wage_cents * deduction_rate

theorem alicia_tax_deduction_is_50_cents : tax_deduction_cents = 50 := by
  sorry

end alicia_tax_deduction_is_50_cents_l211_211787


namespace sqrt_12_minus_sqrt_27_l211_211664

theorem sqrt_12_minus_sqrt_27 :
  (Real.sqrt 12 - Real.sqrt 27 = -Real.sqrt 3) := by
  sorry

end sqrt_12_minus_sqrt_27_l211_211664


namespace largest_reciprocal_l211_211621

theorem largest_reciprocal :
  let a := -1/2
  let b := 1/4
  let c := 0.5
  let d := 3
  let e := 10
  (1 / b) > (1 / a) ∧ (1 / b) > (1 / c) ∧ (1 / b) > (1 / d) ∧ (1 / b) > (1 / e) :=
by
  let a := -1/2
  let b := 1/4
  let c := 0.5
  let d := 3
  let e := 10
  sorry

end largest_reciprocal_l211_211621


namespace problem_statement_l211_211009

open Complex

theorem problem_statement (x y : ℂ) (h : (x + y) / (x - y) - (3 * (x - y)) / (x + y) = 2) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 8320 / 4095 := 
by 
  sorry

end problem_statement_l211_211009


namespace sum_of_angles_l211_211722

theorem sum_of_angles (ABC ABD : ℝ) (n_octagon n_triangle : ℕ) 
(h1 : n_octagon = 8) 
(h2 : n_triangle = 3) 
(h3 : ABC = 180 * (n_octagon - 2) / n_octagon)
(h4 : ABD = 180 * (n_triangle - 2) / n_triangle) : 
ABC + ABD = 195 :=
by {
  sorry
}

end sum_of_angles_l211_211722


namespace find_f2_l211_211322

def f (x : ℝ) (a b : ℝ) := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 3) : f 2 a b = -19 :=
by sorry

end find_f2_l211_211322


namespace people_in_house_l211_211623

theorem people_in_house 
  (charlie_and_susan : ℕ := 2)
  (sarah_and_friends : ℕ := 5)
  (living_room_people : ℕ := 8) :
  (charlie_and_susan + sarah_and_friends) + living_room_people = 15 := 
by
  sorry

end people_in_house_l211_211623


namespace skipping_times_eq_l211_211192

theorem skipping_times_eq (x : ℝ) (h : x > 0) :
  180 / x = 240 / (x + 5) :=
sorry

end skipping_times_eq_l211_211192


namespace intersection_distance_eq_l211_211595

theorem intersection_distance_eq (p q : ℕ) (h1 : p = 88) (h2 : q = 9) :
  p - q = 79 :=
by
  sorry

end intersection_distance_eq_l211_211595


namespace map_length_scale_l211_211389

theorem map_length_scale (len1 len2 : ℕ) (dist1 dist2 : ℕ) (h1 : len1 = 15) (h2 : dist1 = 90) (h3 : len2 = 20) :
  dist2 = 120 :=
by
  sorry

end map_length_scale_l211_211389


namespace total_team_cost_l211_211206

-- Define the costs of individual items and the number of players
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.2
def socks_cost : ℝ := 6.8
def num_players : ℕ := 16

-- Define the total cost for equipment for one player
def player_cost : ℝ :=
  jersey_cost + shorts_cost + socks_cost

-- The main theorem stating the total cost for all players
theorem total_team_cost : num_players * player_cost = 752 := by
  sorry

end total_team_cost_l211_211206


namespace existence_of_b_l211_211571

theorem existence_of_b's (n m : ℕ) (h1 : 1 < n) (h2 : 1 < m) 
  (a : Fin m → ℕ) (h3 : ∀ i, 0 < a i ∧ a i ≤ n^m) :
  ∃ b : Fin m → ℕ, (∀ i, 0 < b i ∧ b i ≤ n) ∧ (∀ i, a i + b i < n) :=
by
  sorry

end existence_of_b_l211_211571


namespace problem_solution_l211_211707

theorem problem_solution (x y z : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 20) (h3 : 0.6 * y = z) : 
  z = 60 := by
  sorry

end problem_solution_l211_211707


namespace find_x_l211_211968

theorem find_x (x : ℕ) (h1 : 8^x = 2^9) (h2 : 8 = 2^3) : x = 3 := by
  sorry

end find_x_l211_211968


namespace problem_1_exists_problem_2_exists_l211_211121

-- Given definition: Center of circle lies on line x - 2y = 0
def lies_on_line (x y : ℝ) : Prop := x - 2 * y = 0

-- Given definition: Chord intercepted on x-axis has a length of 2sqrt(3)
def chord_length (r : ℝ) : Prop := 2 * sqrt 3 = 2 * sqrt (r^2 - (sqrt 3)^2)

-- Given definition for circle equation
def circle_eq (x y cx cy r : ℝ) : Prop := (x - cx)^2 + (y - cy)^2 = r^2

-- Problem 1: Prove standard equation of circle
theorem problem_1_exists (a : ℝ) : 
  (a > 0 ∧ lies_on_line (2 * a) a ∧ chord_length (2 * a)) → 
  ∃ (x y : ℝ), circle_eq x y 2 1 2 := sorry

-- Given definition for line equation
def line_eq (x y b : ℝ) : Prop := y = -2 * x + b

-- Given definition: The discriminant condition for intersection
def disc_condition (b : ℝ) : Prop := b^2 - 10 * b + 5 < 0

-- Problem 2: Prove value of b
theorem problem_2_exists (b : ℝ) : 
  (∃ a, a > 0 ∧ lies_on_line (2 * a) a ∧ chord_length (2 * a) ∧ ∀ x y, line_eq x y b → circle_eq x y 2 1 2) →
  disc_condition b → 
  b = (5 + sqrt 15) / 2 ∨ b = (5 - sqrt 15) / 2 := sorry

end problem_1_exists_problem_2_exists_l211_211121


namespace rectangle_perimeter_l211_211785

-- Definitions based on conditions
def length (w : ℝ) : ℝ := 2 * w
def width (w : ℝ) : ℝ := w
def area (w : ℝ) : ℝ := length w * width w
def perimeter (w : ℝ) : ℝ := 2 * (length w + width w)

-- Problem statement: Prove that the perimeter is 120 cm given area is 800 cm² and length is twice the width
theorem rectangle_perimeter (w : ℝ) (h : area w = 800) : perimeter w = 120 := by
  sorry

end rectangle_perimeter_l211_211785


namespace sum_of_powers_2017_l211_211818

theorem sum_of_powers_2017 (n : ℕ) (x : Fin n → ℤ) (h : ∀ i, x i = 0 ∨ x i = 1 ∨ x i = -1) (h_sum : (Finset.univ : Finset (Fin n)).sum x = 1000) :
  (Finset.univ : Finset (Fin n)).sum (λ i => (x i)^2017) = 1000 :=
by
  sorry

end sum_of_powers_2017_l211_211818


namespace tom_sleep_increase_l211_211216

theorem tom_sleep_increase :
  ∀ (initial_sleep : ℕ) (increase_by : ℚ), 
  initial_sleep = 6 → 
  increase_by = 1/3 → 
  initial_sleep + increase_by * initial_sleep = 8 :=
by 
  intro initial_sleep increase_by h1 h2
  simp [*, add_mul, mul_comm]
  sorry

end tom_sleep_increase_l211_211216


namespace polyhedron_volume_correct_l211_211975

-- Definitions of geometric shapes and their properties
def is_isosceles_right_triangle (A : Type) (a b c : ℝ) := 
  a = b ∧ c = a * Real.sqrt 2

def is_square (B : Type) (side : ℝ) := 
  side = 2

def is_equilateral_triangle (G : Type) (side : ℝ) := 
  side = Real.sqrt 8

noncomputable def polyhedron_volume (A E F B C D G : Type) (a b c d e f g : ℝ) := 
  let cube_volume := 8
  let tetrahedron_volume := 2 * Real.sqrt 2 / 3
  cube_volume - tetrahedron_volume

theorem polyhedron_volume_correct (A E F B C D G : Type) (a b c d e f g : ℝ) :
  (is_isosceles_right_triangle A a b c) →
  (is_isosceles_right_triangle E a b c) →
  (is_isosceles_right_triangle F a b c) →
  (is_square B d) →
  (is_square C e) →
  (is_square D f) →
  (is_equilateral_triangle G g) →
  a = 2 → d = 2 → e = 2 → f = 2 → g = Real.sqrt 8 →
  polyhedron_volume A E F B C D G a b c d e f g =
    8 - (2 * Real.sqrt 2 / 3) :=
by
  intros hA hE hF hB hC hD hG ha hd he hf hg
  sorry

end polyhedron_volume_correct_l211_211975


namespace translated_graph_symmetric_l211_211872

noncomputable def f (x : ℝ) : ℝ := sorry

theorem translated_graph_symmetric (f : ℝ → ℝ)
  (h_translate : ∀ x, f (x - 1) = e^x)
  (h_symmetric : ∀ x, f x = f (-x)) :
  ∀ x, f x = e^(-x - 1) :=
by
  sorry

end translated_graph_symmetric_l211_211872


namespace expression_value_l211_211661

theorem expression_value (x : ℝ) (h : x = 3) : x^4 - 4 * x^2 = 45 := by
  sorry

end expression_value_l211_211661


namespace twenty_is_80_percent_of_what_number_l211_211218

theorem twenty_is_80_percent_of_what_number : ∃ y : ℕ, (20 : ℚ) / y = 4 / 5 ∧ y = 25 := by
  sorry

end twenty_is_80_percent_of_what_number_l211_211218


namespace max_area_100_max_fence_length_l211_211282

noncomputable def maximum_allowable_area (x y : ℝ) : Prop :=
  40 * x + 2 * 45 * y + 20 * x * y ≤ 3200

theorem max_area_100 (x y S : ℝ) (h : maximum_allowable_area x y) :
  S <= 100 :=
sorry

theorem max_fence_length (x y : ℝ) (h : maximum_allowable_area x y) (h1 : x * y = 100) :
  x = 15 :=
sorry

end max_area_100_max_fence_length_l211_211282


namespace acute_triangle_condition_l211_211268

theorem acute_triangle_condition (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A > 0) (h3 : B > 0) (h4 : C > 0)
    (h5 : A + B > 90) (h6 : B + C > 90) (h7 : C + A > 90) : A < 90 ∧ B < 90 ∧ C < 90 :=
sorry

end acute_triangle_condition_l211_211268


namespace factorial_square_gt_power_l211_211745

theorem factorial_square_gt_power (n : ℕ) (h : n > 2) : (n!)^2 > n^n := by
  sorry

end factorial_square_gt_power_l211_211745


namespace original_integer_is_21_l211_211107

theorem original_integer_is_21 (a b c d : ℕ) 
  (h1 : (a + b + c) / 3 + d = 29) 
  (h2 : (a + b + d) / 3 + c = 23) 
  (h3 : (a + c + d) / 3 + b = 21) 
  (h4 : (b + c + d) / 3 + a = 17) : 
  d = 21 :=
sorry

end original_integer_is_21_l211_211107


namespace largest_integral_solution_l211_211810

theorem largest_integral_solution (x : ℤ) : (1 / 4 : ℝ) < (x / 7 : ℝ) ∧ (x / 7 : ℝ) < (3 / 5 : ℝ) → x = 4 :=
by {
  sorry
}

end largest_integral_solution_l211_211810


namespace no_2021_residents_possible_l211_211805

-- Definition: Each islander is either a knight or a liar
def is_knight_or_liar (i : ℕ) : Prop := true -- Placeholder definition for either being a knight or a liar

-- Definition: Knights always tell the truth
def knight_tells_truth (i : ℕ) : Prop := true -- Placeholder definition for knights telling the truth

-- Definition: Liars always lie
def liar_always_lies (i : ℕ) : Prop := true -- Placeholder definition for liars always lying

-- Definition: Even number of knights claimed by some islanders
def even_number_of_knights : Prop := true -- Placeholder definition for the claim of even number of knights

-- Definition: Odd number of liars claimed by remaining islanders
def odd_number_of_liars : Prop := true -- Placeholder definition for the claim of odd number of liars

-- Question and proof problem
theorem no_2021_residents_possible (K L : ℕ) (h1 : K + L = 2021) (h2 : ∀ i, is_knight_or_liar i) 
(h3 : ∀ k, knight_tells_truth k → even_number_of_knights) 
(h4 : ∀ l, liar_always_lies l → odd_number_of_liars) : 
  false := sorry

end no_2021_residents_possible_l211_211805


namespace distance_between_intersections_l211_211596

theorem distance_between_intersections :
  let a := 3
  let b := 2
  let c := -7
  let x1 := (-1 + Real.sqrt 22) / 3
  let x2 := (-1 - Real.sqrt 22) / 3
  let distance := abs (x1 - x2)
  let p := 88  -- 2^2 * 22 = 88
  let q := 9   -- 3^2 = 9
  distance = 2 * Real.sqrt 22 / 3 →
  p - q = 79 :=
by
  sorry

end distance_between_intersections_l211_211596


namespace least_five_digit_congruent_to_6_mod_17_l211_211460

theorem least_five_digit_congruent_to_6_mod_17 :
  ∃ (x : ℕ), 10000 ≤ x ∧ x ≤ 99999 ∧ x % 17 = 6 ∧
  ∀ (y : ℕ), 10000 ≤ y ∧ y ≤ 99999 ∧ y % 17 = 6 → x ≤ y := 
sorry

end least_five_digit_congruent_to_6_mod_17_l211_211460


namespace measure_of_angle_B_l211_211849

-- Define the conditions and the goal as a theorem
theorem measure_of_angle_B (A B C : ℝ)
  (h1 : C = 2 * B)
  (h2 : A = 3 * B)
  (triangle_angle_sum : A + B + C = 180) : B = 30 :=
by
  -- Substitute the conditions into Lean to express and prove the statement
  sorry

end measure_of_angle_B_l211_211849


namespace find_m_l211_211958

theorem find_m (m : ℝ) (h : ∀ x : ℝ, m - |x| ≥ 0 ↔ -1 ≤ x ∧ x ≤ 1) : m = 1 :=
sorry

end find_m_l211_211958


namespace jackson_final_grade_l211_211005

def jackson_hours_playing_video_games : ℕ := 9

def ratio_study_to_play : ℚ := 1 / 3

def time_spent_studying (hours_playing : ℕ) (ratio : ℚ) : ℚ := hours_playing * ratio

def points_per_hour_studying : ℕ := 15

def jackson_grade (time_studied : ℚ) (points_per_hour : ℕ) : ℚ := time_studied * points_per_hour

theorem jackson_final_grade :
  jackson_grade
    (time_spent_studying jackson_hours_playing_video_games ratio_study_to_play)
    points_per_hour_studying = 45 :=
by
  sorry

end jackson_final_grade_l211_211005


namespace max_a_plus_b_l211_211999

theorem max_a_plus_b (a b : ℝ) (h1 : 4 * a + 3 * b ≤ 10) (h2 : 3 * a + 6 * b ≤ 12) : a + b ≤ 14 / 5 := 
sorry

end max_a_plus_b_l211_211999


namespace map_length_represents_distance_l211_211417

theorem map_length_represents_distance :
  ∀ (s : ℝ), (15 * s = 90) → (20 * s = 120) :=
begin
  intros s h,
  sorry
end

end map_length_represents_distance_l211_211417


namespace find_desired_expression_l211_211966

variable (y : ℝ)

theorem find_desired_expression
  (h : y + Real.sqrt (y^2 - 4) + (1 / (y - Real.sqrt (y^2 - 4))) = 12) :
  y^2 + Real.sqrt (y^4 - 4) + (1 / (y^2 - Real.sqrt (y^4 - 4))) = 200 / 9 :=
sorry

end find_desired_expression_l211_211966


namespace minimum_photos_needed_l211_211844

theorem minimum_photos_needed 
  (total_photos : ℕ) 
  (photos_IV : ℕ)
  (photos_V : ℕ) 
  (photos_VI : ℕ) 
  (photos_VII : ℕ) 
  (photos_I_III : ℕ) 
  (H : total_photos = 130)
  (H_IV : photos_IV = 35)
  (H_V : photos_V = 30)
  (H_VI : photos_VI = 25)
  (H_VII : photos_VII = 20)
  (H_I_III : photos_I_III = total_photos - (photos_IV + photos_V + photos_VI + photos_VII)) :
  77 = 77 :=
by
  sorry

end minimum_photos_needed_l211_211844


namespace map_scale_representation_l211_211429

theorem map_scale_representation :
  (∀ len_km_1 len_cm_1 len_cm_2 : ℕ, len_km_1 = 90 → len_cm_1 = 15 → len_cm_2 = 20 →
    let scale := len_km_1 / len_cm_1 in
    len_cm_2 * scale = 120)
  := sorry

end map_scale_representation_l211_211429


namespace total_goals_in_five_matches_is_4_l211_211483

theorem total_goals_in_five_matches_is_4
    (A : ℚ) -- defining the average number of goals before the fifth match as rational
    (h1 : A * 4 + 2 = (A + 0.3) * 5) : -- condition representing total goals equation
    4 = (4 * A + 2) := -- statement that the total number of goals in 5 matches is 4
by
  sorry

end total_goals_in_five_matches_is_4_l211_211483


namespace sum_of_digits_7_pow_11_l211_211472

theorem sum_of_digits_7_pow_11 : 
  let n := 7 in
  let power := 11 in
  let last_two_digits := (n ^ power) % 100 in
  let tens_digit := last_two_digits / 10 in
  let ones_digit := last_two_digits % 10 in
  tens_digit + ones_digit = 7 :=
by {
  sorry
}

end sum_of_digits_7_pow_11_l211_211472


namespace inequality_AM_GM_l211_211683

theorem inequality_AM_GM (a b t : ℝ) (h₁ : 1 < a) (h₂ : 1 < b) (h₃ : 0 < t) : 
  (a^2 / (b^t - 1) + b^(2 * t) / (a^t - 1)) ≥ 8 :=
by
  sorry

end inequality_AM_GM_l211_211683


namespace base_7_perfect_square_ab2c_l211_211841

-- Define the necessary conditions
def is_base_7_representation_of (n : ℕ) (a b c : ℕ) : Prop :=
  n = a * 7^3 + b * 7^2 + 2 * 7 + c

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Lean statement for the problem
theorem base_7_perfect_square_ab2c (n a b c : ℕ) (h1 : a ≠ 0) (h2 : is_base_7_representation_of n a b c) (h3 : is_perfect_square n) :
  c = 2 ∨ c = 3 ∨ c = 6 :=
  sorry

end base_7_perfect_square_ab2c_l211_211841


namespace map_length_representation_l211_211409

theorem map_length_representation :
  ∀ (length1 length2 km1 : ℝ), 
  length1 = 15 ∧ km1 = 90 ∧ length2 = 20 →
  (length2 * (km1 / length1) = 120) :=
by
  intros length1 length2 km1
  intro h
  cases h with hl1 h
  cases h with hkm1 hl2
  rw [hl1, hkm1, hl2]
  simp
  sorry

end map_length_representation_l211_211409


namespace g_at_1001_l211_211018

open Function

variable (g : ℝ → ℝ)

axiom g_property : ∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x
axiom g_at_1 : g 1 = 3

theorem g_at_1001 : g 1001 = -997 :=
by
  sorry

end g_at_1001_l211_211018


namespace second_machine_time_l211_211779

/-- Given:
1. A first machine can address 600 envelopes in 10 minutes.
2. Both machines together can address 600 envelopes in 4 minutes.
We aim to prove that the second machine alone would take 20/3 minutes to address 600 envelopes. -/
theorem second_machine_time (x : ℝ) 
  (first_machine_rate : ℝ := 600 / 10)
  (combined_rate_needed : ℝ := 600 / 4)
  (second_machine_rate : ℝ := combined_rate_needed - first_machine_rate) 
  (secs_envelope_rate : ℝ := second_machine_rate) 
  (envelopes : ℝ := 600) : 
  x = envelopes / secs_envelope_rate :=
sorry

end second_machine_time_l211_211779


namespace percent_of_x_is_y_l211_211893

variable (x y : ℝ)

theorem percent_of_x_is_y
  (h : 0.5 * (x - y) = 0.2 * (x + y)) :
  y = 0.4286 * x := by
  sorry

end percent_of_x_is_y_l211_211893


namespace coal_extraction_in_four_months_l211_211631

theorem coal_extraction_in_four_months
  (x1 x2 x3 x4 : ℝ)
  (h1 : 4 * x1 + x2 + 2 * x3 + 5 * x4 = 10)
  (h2 : 2 * x1 + 3 * x2 + 2 * x3 + x4 = 7)
  (h3 : 5 * x1 + 2 * x2 + x3 + 4 * x4 = 14) :
  4 * (x1 + x2 + x3 + x4) = 12 :=
by
  sorry

end coal_extraction_in_four_months_l211_211631


namespace compatible_polynomial_count_l211_211908

theorem compatible_polynomial_count (n : ℕ) : 
  ∃ num_polynomials : ℕ, num_polynomials = (n / 2) + 1 :=
by
  sorry

end compatible_polynomial_count_l211_211908


namespace smallest_positive_integer_x_l211_211890

-- Definitions based on the conditions given
def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

-- Statement of the problem
theorem smallest_positive_integer_x (x : ℕ) :
  (is_multiple (900 * x) 640) → x = 32 :=
sorry

end smallest_positive_integer_x_l211_211890


namespace number_of_taxis_l211_211875

-- Define the conditions explicitly
def number_of_cars : ℕ := 3
def people_per_car : ℕ := 4
def number_of_vans : ℕ := 2
def people_per_van : ℕ := 5
def people_per_taxi : ℕ := 6
def total_people : ℕ := 58

-- Define the number of people in cars and vans
def people_in_cars := number_of_cars * people_per_car
def people_in_vans := number_of_vans * people_per_van
def people_in_taxis := total_people - (people_in_cars + people_in_vans)

-- The theorem we need to prove
theorem number_of_taxis : people_in_taxis / people_per_taxi = 6 := by
  sorry

end number_of_taxis_l211_211875


namespace boys_count_l211_211153

-- Define the number of girls
def girls : ℕ := 635

-- Define the number of boys as being 510 more than the number of girls
def boys : ℕ := girls + 510

-- Prove that the number of boys in the school is 1145
theorem boys_count : boys = 1145 := by
  sorry

end boys_count_l211_211153


namespace trig_inequality_l211_211997

noncomputable def a : ℝ := Real.sin (31 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (58 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (32 * Real.pi / 180)

theorem trig_inequality : c > b ∧ b > a := by
  sorry

end trig_inequality_l211_211997


namespace seq_ratio_l211_211515

noncomputable def arith_seq (a d : ℝ) (n : ℕ) : ℝ := a + n * d

theorem seq_ratio (a d : ℝ) (h₁ : d ≠ 0) (h₂ : (arith_seq a d 2)^2 = (arith_seq a d 0) * (arith_seq a d 8)) :
  (arith_seq a d 0 + arith_seq a d 2 + arith_seq a d 4) / (arith_seq a d 1 + arith_seq a d 3 + arith_seq a d 5) = 3 / 4 :=
by
  sorry

end seq_ratio_l211_211515


namespace train_length_is_correct_l211_211911

noncomputable def length_of_train (time_in_seconds : ℝ) (relative_speed : ℝ) : ℝ :=
  relative_speed * time_in_seconds

noncomputable def relative_speed_in_mps (speed_of_train_kmph : ℝ) (speed_of_man_kmph : ℝ) : ℝ :=
  (speed_of_train_kmph + speed_of_man_kmph) * (1000 / 3600)

theorem train_length_is_correct :
  let speed_of_train_kmph := 65.99424046076315
  let speed_of_man_kmph := 6
  let time_in_seconds := 6
  length_of_train time_in_seconds (relative_speed_in_mps speed_of_train_kmph speed_of_man_kmph) = 119.9904 := by
  sorry

end train_length_is_correct_l211_211911


namespace parabola_focus_line_slope_intersect_l211_211338

theorem parabola_focus (p : ℝ) (hp : 0 < p) 
  (focus : (1/2 : ℝ) = p/2) : p = 1 :=
by sorry

theorem line_slope_intersect (t : ℝ)
  (intersects_parabola : ∃ A B : ℝ × ℝ, A ≠ (0, 0) ∧ B ≠ (0, 0) ∧
    A ≠ B ∧ A.2 = 2 * A.1 + t ∧ B.2 = 2 * B.1 + t ∧ 
    A.2^2 = 2 * p * A.1 ∧ B.2^2 = 2 * p * B.1 ∧ 
    A.1 * B.1 + A.2 * B.2 = 0) : 
  t = -4 :=
by sorry

end parabola_focus_line_slope_intersect_l211_211338


namespace probability_of_first_three_red_cards_l211_211780

theorem probability_of_first_three_red_cards :
  let total_cards := 60
  let red_cards := 36
  let black_cards := total_cards - red_cards
  let total_ways := total_cards * (total_cards - 1) * (total_cards - 2)
  let red_ways := red_cards * (red_cards - 1) * (red_cards - 2)
  (red_ways / total_ways) = 140 / 673 :=
by
  sorry

end probability_of_first_three_red_cards_l211_211780


namespace greatest_prime_factor_341_l211_211253

theorem greatest_prime_factor_341 : ∃ p, Nat.Prime p ∧ p ≥ 17 ∧ (∀ q, Nat.Prime q ∧ q ∣ 341 → q ≤ p) ∧ p = 19 := by
  sorry

end greatest_prime_factor_341_l211_211253


namespace actual_cost_of_article_l211_211491

noncomputable def article_actual_cost (x : ℝ) : Prop :=
  (0.58 * x = 1050) → x = 1810.34

theorem actual_cost_of_article : ∃ x : ℝ, article_actual_cost x :=
by
  use 1810.34
  sorry

end actual_cost_of_article_l211_211491


namespace rate_of_current_is_8_5_l211_211284

-- Define the constants for the problem
def downstream_speed : ℝ := 24
def upstream_speed : ℝ := 7
def rate_still_water : ℝ := 15.5

-- Define the rate of the current calculation
def rate_of_current : ℝ := downstream_speed - rate_still_water

-- Define the rate of the current proof statement
theorem rate_of_current_is_8_5 :
  rate_of_current = 8.5 :=
by
  -- This skip the actual proof
  sorry

end rate_of_current_is_8_5_l211_211284


namespace calculate_expression_l211_211796

theorem calculate_expression : 15 * 30 + 45 * 15 + 90 = 1215 := 
by 
  sorry

end calculate_expression_l211_211796


namespace integer_triplet_solution_l211_211733

def circ (a b : ℤ) : ℤ := a + b - a * b

theorem integer_triplet_solution (x y z : ℤ) :
  circ (circ x y) z + circ (circ y z) x + circ (circ z x) y = 0 ↔
  (x = 0 ∧ y = 0 ∧ z = 2) ∨ (x = 0 ∧ y = 2 ∧ z = 0) ∨ (x = 2 ∧ y = 0 ∧ z = 0) :=
by
  sorry

end integer_triplet_solution_l211_211733


namespace combined_points_kjm_l211_211171

theorem combined_points_kjm {P B K J M H C E: ℕ} 
  (total_points : P + B + K + J + M = 81)
  (paige_points : P = 21)
  (brian_points : B = 20)
  (karen_jennifer_michael_sum : K + J + M = 40)
  (karen_scores : ∀ p, K = 2 * p + 5 * (H - p))
  (jennifer_scores : ∀ p, J = 2 * p + 5 * (C - p))
  (michael_scores : ∀ p, M = 2 * p + 5 * (E - p)) :
  K + J + M = 40 :=
by sorry

end combined_points_kjm_l211_211171


namespace total_presents_l211_211799

variables (ChristmasPresents BirthdayPresents EasterPresents HalloweenPresents : ℕ)

-- Given conditions
def condition1 : ChristmasPresents = 60 := sorry
def condition2 : BirthdayPresents = 3 * EasterPresents := sorry
def condition3 : EasterPresents = (ChristmasPresents / 2) - 10 := sorry
def condition4 : HalloweenPresents = BirthdayPresents - EasterPresents := sorry

-- Proof statement
theorem total_presents (h1 : ChristmasPresents = 60)
    (h2 : BirthdayPresents = 3 * EasterPresents)
    (h3 : EasterPresents = (ChristmasPresents / 2) - 10)
    (h4 : HalloweenPresents = BirthdayPresents - EasterPresents) :
    ChristmasPresents + BirthdayPresents + EasterPresents + HalloweenPresents = 180 :=
sorry

end total_presents_l211_211799


namespace roots_equation_value_l211_211814

theorem roots_equation_value (α β : ℝ) (h1 : α^2 - α - 1 = 0) (h2 : β^2 - β - 1 = 0) (h3 : α + β = 1) :
    α^4 + 3 * β = 5 := by
sorry

end roots_equation_value_l211_211814


namespace rationalize_denominator_l211_211179

theorem rationalize_denominator :
  let A := 5
  let B := 2
  let C := 1
  let D := 4
  A + B + C + D = 12 :=
by
  sorry

end rationalize_denominator_l211_211179


namespace least_five_digit_congruent_to_6_mod_17_l211_211462

theorem least_five_digit_congruent_to_6_mod_17 :
  ∃ (x : ℕ), 10000 ≤ x ∧ x ≤ 99999 ∧ x % 17 = 6 ∧
  ∀ (y : ℕ), 10000 ≤ y ∧ y ≤ 99999 ∧ y % 17 = 6 → x ≤ y := 
sorry

end least_five_digit_congruent_to_6_mod_17_l211_211462


namespace map_representation_l211_211379

theorem map_representation (d1 d2 l1 l2 : ℕ)
  (h1 : l1 = 15)
  (h2 : d1 = 90)
  (h3 : l2 = 20) :
  d2 = 120 :=
by
  sorry

end map_representation_l211_211379


namespace marble_probability_l211_211901

theorem marble_probability :
  let total_marbles := 13
  let red_marbles := 5
  let white_marbles := 8
  let first_red_prob := (red_marbles:ℚ) / total_marbles
  let second_white_given_first_red_prob := (white_marbles:ℚ) / (total_marbles - 1)
  let third_red_given_first_red_and_second_white_prob := (red_marbles - 1:ℚ) / (total_marbles - 2)
  first_red_prob * second_white_given_first_red_prob * third_red_given_first_red_and_second_white_prob = (40 : ℚ) / 429 :=
by
  let total_marbles := 13
  let red_marbles := 5
  let white_marbles := 8
  let first_red_prob := (red_marbles:ℚ) / total_marbles
  let second_white_given_first_red_prob := (white_marbles:ℚ) / (total_marbles - 1)
  let third_red_given_first_red_and_second_white_prob := (red_marbles - 1:ℚ) / (total_marbles - 2)
  -- Adding sorry to skip the proof
  sorry

end marble_probability_l211_211901


namespace three_monotonic_intervals_l211_211549

open Real

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := - (4 / 3) * x ^ 3 + (b - 1) * x

noncomputable def f' (b : ℝ) (x : ℝ) : ℝ := -4 * x ^ 2 + (b - 1)

theorem three_monotonic_intervals (b : ℝ) (h : (b - 1) > 0) : b > 1 := 
by
  have discriminant : 16 * (b - 1) > 0 := sorry
  sorry

end three_monotonic_intervals_l211_211549


namespace islander_parity_l211_211807

-- Define the concept of knights and liars
def is_knight (x : ℕ) : Prop := x % 2 = 0 -- Knight count is even
def is_liar (x : ℕ) : Prop := ¬(x % 2 = 1) -- Liar count being odd is false, so even

-- Define the total inhabitants on the island and conditions
theorem islander_parity (K L : ℕ) (h₁ : is_knight K) (h₂ : is_liar L) (h₃ : K + L = 2021) : false := sorry

end islander_parity_l211_211807


namespace father_age_three_times_xiaojun_after_years_l211_211611

theorem father_age_three_times_xiaojun_after_years (years_passed : ℕ) (xiaojun_current_age : ℕ) (father_current_age : ℕ) 
  (h1 : xiaojun_current_age = 5) (h2 : father_current_age = 31) (h3 : years_passed = 8) :
  father_current_age + years_passed = 3 * (xiaojun_current_age + years_passed) := by
  sorry

end father_age_three_times_xiaojun_after_years_l211_211611


namespace odd_indexed_terms_geometric_sequence_l211_211158

open Nat

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (2 * n + 3) = r * a (2 * n + 1)

theorem odd_indexed_terms_geometric_sequence (b : ℕ → ℝ) (h : ∀ n, b n * b (n + 1) = 3 ^ n) :
  is_geometric_sequence b 3 :=
by
  sorry

end odd_indexed_terms_geometric_sequence_l211_211158


namespace probability_of_event_correct_l211_211175

def within_interval (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ Real.pi

def tan_in_range (x : ℝ) : Prop :=
  -1 ≤ Real.tan x ∧ Real.tan x ≤ Real.sqrt 3

def valid_subintervals (x : ℝ) : Prop :=
  within_interval x ∧ tan_in_range x

def interval_length (a b : ℝ) : ℝ :=
  b - a

noncomputable def probability_of_event : ℝ :=
  (interval_length 0 (Real.pi / 3) + interval_length (3 * Real.pi / 4) Real.pi) / Real.pi

theorem probability_of_event_correct :
  probability_of_event = 7 / 12 := sorry

end probability_of_event_correct_l211_211175


namespace jackson_grade_l211_211002

open Function

theorem jackson_grade :
  ∃ (grade : ℕ), 
  ∀ (hours_playing hours_studying : ℕ), 
    (hours_playing = 9) ∧ 
    (hours_studying = hours_playing / 3) ∧ 
    (grade = hours_studying * 15) →
    grade = 45 := 
by {
  sorry
}

end jackson_grade_l211_211002


namespace max_abc_value_l211_211365

theorem max_abc_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * b + b * c = 518) (h2 : a * b - a * c = 360) : 
  a * b * c ≤ 1008 :=
sorry

end max_abc_value_l211_211365


namespace yards_after_8_marathons_l211_211285

-- Define the constants and conditions
def marathon_miles := 26
def marathon_yards := 395
def yards_per_mile := 1760

-- Definition for total distance covered after 8 marathons
def total_miles := marathon_miles * 8
def total_yards := marathon_yards * 8

-- Convert the total yards into miles with remainder
def extra_miles := total_yards / yards_per_mile
def remainder_yards := total_yards % yards_per_mile

-- Prove the remainder yards is 1400
theorem yards_after_8_marathons : remainder_yards = 1400 := by
  -- Proof steps would go here
  sorry

end yards_after_8_marathons_l211_211285


namespace fixed_point_is_one_three_l211_211522

noncomputable def fixed_point_of_function (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) : ℝ × ℝ :=
  (1, 3)

theorem fixed_point_is_one_three {a : ℝ} (h_pos : 0 < a) (h_ne_one : a ≠ 1) :
  fixed_point_of_function a h_pos h_ne_one = (1, 3) :=
  sorry

end fixed_point_is_one_three_l211_211522


namespace map_distance_l211_211374

theorem map_distance (fifteen_cm_in_km : ℤ) (cm_to_km : ℕ): 
  fifteen_cm_in_km = 90 ∧ cm_to_km = 6 → 20 * cm_to_km = 120 := 
by 
  sorry

end map_distance_l211_211374


namespace base_five_equivalent_of_156_is_1111_l211_211228

theorem base_five_equivalent_of_156_is_1111 : nat_to_base 5 156 = [1, 1, 1, 1] := 
sorry

end base_five_equivalent_of_156_is_1111_l211_211228


namespace seventh_place_is_unspecified_l211_211553

noncomputable def charlie_position : ℕ := 5
noncomputable def emily_position : ℕ := charlie_position + 5
noncomputable def dana_position : ℕ := 10
noncomputable def bob_position : ℕ := dana_position - 2
noncomputable def alice_position : ℕ := emily_position + 3

theorem seventh_place_is_unspecified :
  ∀ (x : ℕ), 1 ≤ x ∧ x ≤ 15 ∧ x ≠ charlie_position ∧ x ≠ emily_position ∧
  x ≠ dana_position ∧ x ≠ bob_position ∧ x ≠ alice_position →
  x = 7 → false := 
by
  sorry

end seventh_place_is_unspecified_l211_211553


namespace a_18_value_l211_211659

-- Define the concept of an "Equally Summed Sequence"
def equallySummedSequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  ∀ n, a n + a (n + 1) = c

-- Define the specific conditions for a_1 and the common sum
def specific_sequence (a : ℕ → ℝ) : Prop :=
  equallySummedSequence a 5 ∧ a 1 = 2

-- The theorem we want to prove
theorem a_18_value (a : ℕ → ℝ) (h : specific_sequence a) : a 18 = 3 :=
sorry

end a_18_value_l211_211659


namespace pipe_filling_time_l211_211742

theorem pipe_filling_time 
  (rate_A : ℚ := 1/8) 
  (rate_L : ℚ := 1/24) :
  (1 / (rate_A - rate_L) = 12) :=
by
  sorry

end pipe_filling_time_l211_211742


namespace operation_correct_l211_211971

def operation (x y : ℝ) := x^2 + y^2 + 12

theorem operation_correct :
  operation (Real.sqrt 6) (Real.sqrt 6) = 23.999999999999996 :=
by
  -- proof omitted
  sorry

end operation_correct_l211_211971


namespace correct_answer_is_ln_abs_l211_211297

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_monotonically_increasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, (0 < x ∧ x < y) → f x ≤ f y

theorem correct_answer_is_ln_abs :
  is_even_function (fun x => Real.log (abs x)) ∧ is_monotonically_increasing_on_pos (fun x => Real.log (abs x)) ∧
  ¬ is_even_function (fun x => x^3) ∧
  ¬ is_monotonically_increasing_on_pos (fun x => Real.cos x) :=
by
  sorry

end correct_answer_is_ln_abs_l211_211297


namespace general_term_sequence_x_l211_211001

-- Definitions used in Lean statement corresponding to the conditions.
noncomputable def sequence_a (n : ℕ) : ℝ := sorry

noncomputable def sequence_x (n : ℕ) : ℝ := sorry

axiom condition_1 : ∀ n : ℕ, 
  ((sequence_a (n + 2))⁻¹ = ((sequence_a n)⁻¹ + (sequence_a (n + 1))⁻¹) / 2)

axiom condition_2 {n : ℕ} : sequence_x n > 0

axiom condition_3 : sequence_x 1 = 3

axiom condition_4 : sequence_x 1 + sequence_x 2 + sequence_x 3 = 39

axiom condition_5 (n : ℕ) : (sequence_x n)^(sequence_a n) = 
  (sequence_x (n + 1))^(sequence_a (n + 1)) ∧ 
  (sequence_x (n + 1))^(sequence_a (n + 1)) = 
  (sequence_x (n + 2))^(sequence_a (n + 2))

-- Theorem stating that the general term of sequence {x_n} is 3^n.
theorem general_term_sequence_x : ∀ n : ℕ, sequence_x n = 3^n :=
by
  sorry

end general_term_sequence_x_l211_211001


namespace number_of_nonnegative_solutions_l211_211139

-- Define the equation x^2 = -6x
def equation (x : ℝ) : Prop := x^2 = -6 * x

-- Define the condition for a nonnegative solution
def nonnegative (x : ℝ) : Prop := x ≥ 0

-- Define the main theorem to prove the number of nonnegative solutions
theorem number_of_nonnegative_solutions : 
  {x : ℝ // equation x ∧ nonnegative x}.to_finset.card = 1 :=
sorry

end number_of_nonnegative_solutions_l211_211139


namespace main_theorem_l211_211819

variable (x y z : ℝ) (h₀ : 0 < x ∧ 0 < y ∧ 0 < z) (h₁ : x^3 + y^3 + z^3 = 1)

theorem main_theorem (h₀ : 0 < x ∧ 0 < y ∧ 0 < z) (h₁ : x^3 + y^3 + z^3 = 1):
  (x^2 / (1 - x^2)) + (y^2 / (1 - y^2)) + (z^2 / (1 - z^2)) ≥ (3 * Real.sqrt 3) / 2 := 
by
  sorry

end main_theorem_l211_211819


namespace number_of_cards_per_page_l211_211864

variable (packs : ℕ) (cards_per_pack : ℕ) (total_pages : ℕ)

def number_of_cards (packs cards_per_pack : ℕ) : ℕ :=
  packs * cards_per_pack

def cards_per_page (total_cards total_pages : ℕ) : ℕ :=
  total_cards / total_pages

theorem number_of_cards_per_page
  (packs := 60) (cards_per_pack := 7) (total_pages := 42)
  (total_cards := number_of_cards packs cards_per_pack)
    : cards_per_page total_cards total_pages = 10 :=
by {
  sorry
}

end number_of_cards_per_page_l211_211864


namespace bishop_safe_squares_l211_211852

def chessboard_size : ℕ := 64
def total_squares_removed_king : ℕ := chessboard_size - 1
def threat_squares : ℕ := 7

theorem bishop_safe_squares : total_squares_removed_king - threat_squares = 30 :=
by
  sorry

end bishop_safe_squares_l211_211852


namespace system_of_two_linear_equations_l211_211475

theorem system_of_two_linear_equations :
  ((∃ x y z, x + z = 5 ∧ x - 2 * y = 6) → False) ∧
  ((∃ x y, x * y = 5 ∧ x - 4 * y = 2) → False) ∧
  ((∃ x y, x + y = 5 ∧ 3 * x - 4 * y = 12) → True) ∧
  ((∃ x y, x^2 + y = 2 ∧ x - y = 9) → False) :=
by {
  sorry
}

end system_of_two_linear_equations_l211_211475


namespace Mrs_Hilt_bought_two_cones_l211_211737

def ice_cream_cone_cost : ℕ := 99
def total_spent : ℕ := 198

theorem Mrs_Hilt_bought_two_cones : total_spent / ice_cream_cone_cost = 2 :=
by
  sorry

end Mrs_Hilt_bought_two_cones_l211_211737


namespace problem_statement_l211_211519

theorem problem_statement (a b c : ℝ) (h₀ : 4 * a - 4 * b + c > 0) (h₁ : a + 2 * b + c < 0) : b^2 > a * c :=
sorry

end problem_statement_l211_211519


namespace right_triangle_acute_angle_le_45_l211_211759

theorem right_triangle_acute_angle_le_45
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hright : a^2 + b^2 = c^2):
  ∃ θ φ : ℝ, θ + φ = 90 ∧ (θ ≤ 45 ∨ φ ≤ 45) :=
by
  sorry

end right_triangle_acute_angle_le_45_l211_211759


namespace number_of_distinct_m_values_l211_211362

theorem number_of_distinct_m_values :
  let roots (x_1 x_2 : ℤ) := x_1 * x_2 = 36 ∧ x_2 = x_2
  let m_values := {m : ℤ | ∃ (x_1 x_2 : ℤ), x_1 * x_2 = 36 ∧ m = x_1 + x_2}
  m_values.card = 10 :=
sorry

end number_of_distinct_m_values_l211_211362


namespace greatest_prime_factor_341_l211_211243

theorem greatest_prime_factor_341 : ∃ p : ℕ, prime p ∧ p.factor 341 ∧ (∀ q : ℕ, prime q ∧ q.factor 341 → q ≤ p) ∧ p = 17 :=
by sorry

end greatest_prime_factor_341_l211_211243


namespace intersection_M_N_l211_211126

-- Definitions of the sets M and N based on the conditions
def M (x : ℝ) : Prop := ∃ (y : ℝ), y = Real.log (x^2 - 3*x - 4)
def N (y : ℝ) : Prop := ∃ (x : ℝ), y = 2^(x - 1)

-- The proof statement
theorem intersection_M_N : { x : ℝ | M x } ∩ { x : ℝ | ∃ y : ℝ, N y ∧ y = Real.log (x^2 - 3*x - 4) } = { x : ℝ | x > 4 } :=
by
  sorry

end intersection_M_N_l211_211126


namespace water_usage_l211_211555

noncomputable def litres_per_household_per_month (total_litres : ℕ) (number_of_households : ℕ) : ℕ :=
  total_litres / number_of_households

theorem water_usage : litres_per_household_per_month 2000 10 = 200 :=
by
  sorry

end water_usage_l211_211555


namespace geom_seq_m_equals_11_l211_211981

theorem geom_seq_m_equals_11 
  (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 = 1) 
  (h2 : |q| ≠ 1) 
  (h3 : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h4 : a m = (a 1) * (a 2) * (a 3) * (a 4) * (a 5)) : 
  m = 11 :=
sorry

end geom_seq_m_equals_11_l211_211981


namespace finish_11th_l211_211151

noncomputable def place_in_race (place: Fin 15) := ℕ

variables (Dana Ethan Alice Bob Chris Flora : Fin 15)

def conditions := 
  Dana.val + 3 = Ethan.val ∧
  Alice.val = Bob.val - 2 ∧
  Chris.val = Flora.val - 5 ∧
  Flora.val = Dana.val + 2 ∧
  Ethan.val = Alice.val - 3 ∧
  Bob.val = 6

theorem finish_11th (h : conditions Dana Ethan Alice Bob Chris Flora) : Flora.val = 10 :=
  by sorry

end finish_11th_l211_211151


namespace max_abc_value_l211_211366

theorem max_abc_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * b + b * c = 518) (h2 : a * b - a * c = 360) : 
  a * b * c ≤ 1008 :=
sorry

end max_abc_value_l211_211366


namespace rationalize_result_l211_211177

noncomputable def rationalize_denominator (x y : ℚ) (sqrt_c : ℚ) : ℚ :=
  let numerator := x + sqrt_c
  let denominator := y - sqrt_c
  (numerator * (y + sqrt_c)) / (denominator * (y + sqrt_c))

theorem rationalize_result :
  let sqrt_5 := Real.sqrt 5
  let expr := rationalize_denominator 2 3 sqrt_5
  let A := 11 / 4
  let B := 5 / 4
  let C := 5
  expr = A + B * sqrt_5 ∧ A * B * C = 275 / 16 := 
sorry

end rationalize_result_l211_211177


namespace simplify_expression_l211_211866

variable (a b : ℤ)

theorem simplify_expression : 
  (15 * a + 45 * b) + (21 * a + 32 * b) - (12 * a + 40 * b) = 24 * a + 37 * b := 
    by sorry

end simplify_expression_l211_211866


namespace find_number_l211_211278

theorem find_number (x : ℝ) : 61 + x * 12 / (180 / 3) = 62 → x = 5 :=
by
  sorry

end find_number_l211_211278


namespace arrange_numbers_l211_211493

theorem arrange_numbers (x y z : ℝ) (h1 : x = 20.8) (h2 : y = 0.82) (h3 : z = Real.log 20.8) : z < y ∧ y < x :=
by
  sorry

end arrange_numbers_l211_211493


namespace find_m_for_one_solution_l211_211182

theorem find_m_for_one_solution (m : ℚ) :
  (∀ x : ℝ, 3*x^2 - 7*x + m = 0 → (∃! y : ℝ, 3*y^2 - 7*y + m = 0)) → m = 49/12 := by
  sorry

end find_m_for_one_solution_l211_211182


namespace incircle_incenters_perpendicular_l211_211495

theorem incircle_incenters_perpendicular (A B C D E F : Point) (incircle_touch_BC_at_D : incircle_touches_side BC A B C D) 
  (E_is_incenter_ABD : incenter_of_tri A B D E) (F_is_incenter_ACD : incenter_of_tri A C D F) : 
  perp EF AD := 
sorry

end incircle_incenters_perpendicular_l211_211495


namespace num_of_distinct_m_values_l211_211363

theorem num_of_distinct_m_values : 
  (∃ (x1 x2 : ℤ), x1 * x2 = 36 ∧ m = x1 + x2) → 
  (finset.card (finset.image (λ (p : ℤ × ℤ), p.1 + p.2) 
    {p | p.1 * p.2 = 36})) = 10 :=
sorry

end num_of_distinct_m_values_l211_211363


namespace find_f_60_l211_211187

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function definition.

axiom functional_eq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x / y
axiom f_48 : f 48 = 36

theorem find_f_60 : f 60 = 28.8 := by 
  sorry

end find_f_60_l211_211187


namespace sum_tens_ones_digit_of_7_pow_11_l211_211466

/--
The sum of the tens digit and the ones digit of (3+4)^{11} is 7.
-/
theorem sum_tens_ones_digit_of_7_pow_11 : 
  let number := (3 + 4)^11
  let tens_digit := (number / 10) % 10
  let ones_digit := number % 10
  tens_digit + ones_digit = 7 :=
by
  sorry

end sum_tens_ones_digit_of_7_pow_11_l211_211466


namespace problem_proof_l211_211521

-- Define the given conditions and the target statement
theorem problem_proof (a b : ℝ) (h1 : a - b = 2) (h2 : a * b = 10.5) : a^2 + b^2 = 25 := 
by sorry

end problem_proof_l211_211521


namespace evaluate_Q_at_2_l211_211039

-- Define the polynomial Q(x)
noncomputable def Q (x : ℚ) : ℚ := x^4 - 20 * x^2 + 16

-- Define the root condition of sqrt(3) + sqrt(7)
def root_condition (x : ℚ) : Prop := (x = ℚ.sqrt(3) + ℚ.sqrt(7))

-- Theorem statement
theorem evaluate_Q_at_2 : Q 2 = -32 :=
by
  -- Given conditions
  have h1 : degree Q = 4 := sorry,
  have h2 : leading_coeff Q = 1 := sorry,
  have h3 : root_condition (ℚ.sqrt(3) + ℚ.sqrt(7)) := sorry,
  -- Goal
  exact sorry

end evaluate_Q_at_2_l211_211039


namespace map_scale_l211_211398

theorem map_scale (represents_15cm_as_km : 15 * 6 = 90) :
  20 * 6 = 120 :=
by
  sorry

end map_scale_l211_211398


namespace simplify_expression_l211_211867

theorem simplify_expression (x : ℝ) : 
  x^2 * (4 * x^3 - 3 * x + 1) - 6 * (x^3 - 3 * x^2 + 4 * x - 5) = 
  4 * x^5 - 9 * x^3 + 19 * x^2 - 24 * x + 30 := by
  sorry

end simplify_expression_l211_211867


namespace sequence_sum_l211_211685

theorem sequence_sum (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
    (h1 : a 1 = 1)
    (h_rec : ∀ n, a (n + 2) = 1 / (a n + 1))
    (h6_2 : a 6 = a 2) :
    a 2016 + a 3 = (Real.sqrt 5) / 2 :=
by
  sorry

end sequence_sum_l211_211685


namespace smallest_value_n_l211_211663

theorem smallest_value_n :
  ∃ (n : ℕ), n * 25 = Nat.lcm (Nat.lcm 10 18) 20 ∧ (∀ m, m * 25 = Nat.lcm (Nat.lcm 10 18) 20 → n ≤ m) := 
sorry

end smallest_value_n_l211_211663


namespace greatest_prime_factor_341_l211_211230

theorem greatest_prime_factor_341 : ∃ (p : ℕ), prime p ∧ p ∣ 341 ∧ ∀ q, prime q ∧ q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_341_l211_211230


namespace concentric_circles_ratio_l211_211045

theorem concentric_circles_ratio (R r k : ℝ) (hr : r > 0) (hRr : R > r) (hk : k > 0)
  (area_condition : π * (R^2 - r^2) = k * π * r^2) :
  R / r = Real.sqrt (k + 1) :=
by
  sorry

end concentric_circles_ratio_l211_211045


namespace base6_addition_correct_l211_211294

-- Define a function to convert a base 6 digit to its base 10 equivalent
def base6_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 4
  | 5 => 5
  | d => 0 -- for illegal digits, fallback to 0

-- Define a function to convert a number in base 6 to base 10
def convert_base6_to_base10 (n : Nat) : Nat :=
  let units := base6_to_base10 (n % 10)
  let tens := base6_to_base10 ((n / 10) % 10)
  let hundreds := base6_to_base10 ((n / 100) % 10)
  units + 6 * tens + 6 * 6 * hundreds

-- Define a function to convert a base 10 number to a base 6 number
def base10_to_base6 (n : Nat) : Nat :=
  (n % 6) + 10 * ((n / 6) % 6) + 100 * ((n / (6 * 6)) % 6)

theorem base6_addition_correct : base10_to_base6 (convert_base6_to_base10 35 + convert_base6_to_base10 25) = 104 := by
  sorry

end base6_addition_correct_l211_211294


namespace children_group_size_l211_211087

theorem children_group_size (x : ℕ) (h1 : 255 % 17 = 0) (h2: ∃ n : ℕ, n * 17 = 255) 
                            (h3 : ∀ a c, a = c → a = 255 → c = 255 → x = 17) : 
                            (255 / x = 15) → x = 17 :=
by
  sorry

end children_group_size_l211_211087


namespace jimin_initial_candies_l211_211167

theorem jimin_initial_candies : 
  let candies_given_to_yuna := 25
  let candies_given_to_sister := 13
  candies_given_to_yuna + candies_given_to_sister = 38 := 
  by 
    sorry

end jimin_initial_candies_l211_211167


namespace rectangle_diagonal_opposite_vertex_l211_211544

theorem rectangle_diagonal_opposite_vertex :
  ∀ (x y : ℝ),
    (∃ (x1 y1 x2 y2 x3 y3 : ℝ),
      (x1, y1) = (5, 10) ∧ (x2, y2) = (15, -6) ∧ (x3, y3) = (11, 2) ∧
      (∃ (mx my : ℝ), mx = (x1 + x2) / 2 ∧ my = (y1 + y2) / 2 ∧
        mx = (x + x3) / 2 ∧ my = (y + y3) / 2) ∧
      x = 9 ∧ y = 2) :=
by
  sorry

end rectangle_diagonal_opposite_vertex_l211_211544


namespace doubled_cylinder_volume_l211_211482

theorem doubled_cylinder_volume (r h : ℝ) (V : ℝ) (original_volume : V = π * r^2 * h) (V' : ℝ) : (2 * 2 * π * r^2 * h = 40) := 
by 
  have original_volume := 5
  sorry

end doubled_cylinder_volume_l211_211482


namespace B_coordinates_when_A_is_origin_l211_211859

-- Definitions based on the conditions
def A_coordinates_when_B_is_origin := (2, 5)

-- Theorem to prove the coordinates of B when A is the origin
theorem B_coordinates_when_A_is_origin (x y : ℤ) :
    A_coordinates_when_B_is_origin = (2, 5) →
    (x, y) = (-2, -5) :=
by
  intro h
  -- skipping the proof steps
  sorry

end B_coordinates_when_A_is_origin_l211_211859


namespace average_salary_l211_211589

theorem average_salary (avg_officer_salary avg_nonofficer_salary num_officers num_nonofficers : ℕ) (total_salary total_employees : ℕ) : 
  avg_officer_salary = 430 → 
  avg_nonofficer_salary = 110 → 
  num_officers = 15 → 
  num_nonofficers = 465 → 
  total_salary = avg_officer_salary * num_officers + avg_nonofficer_salary * num_nonofficers → 
  total_employees = num_officers + num_nonofficers → 
  total_salary / total_employees = 120 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end average_salary_l211_211589


namespace greatest_prime_factor_341_l211_211237

theorem greatest_prime_factor_341 : ∃ p : ℕ, Nat.Prime p ∧ p = 17 ∧ p = Nat.gcd 341 (Nat.gcd 341 (Nat.gcd 341 341)) :=
by
  sorry

end greatest_prime_factor_341_l211_211237


namespace max_value_fraction_l211_211831

theorem max_value_fraction (e a b : ℝ) (h : ∀ x : ℝ, (e - a) * Real.exp x + x + b + 1 ≤ 0) : 
  (b + 1) / a ≤ 1 / e :=
sorry

end max_value_fraction_l211_211831


namespace shaded_area_l211_211226

theorem shaded_area (r1 r2 : ℝ) (h1 : r2 = 3 * r1) (h2 : r1 = 2) : 
  π * (r2 ^ 2) - π * (r1 ^ 2) = 32 * π :=
by
  sorry

end shaded_area_l211_211226


namespace snow_volume_l211_211994

-- Define the dimensions of the sidewalk and the snow depth
def length : ℝ := 20
def width : ℝ := 2
def depth : ℝ := 0.5

-- Define the volume calculation
def volume (l w d : ℝ) : ℝ := l * w * d

-- The theorem to prove
theorem snow_volume : volume length width depth = 20 := 
by
  sorry

end snow_volume_l211_211994


namespace percent_problem_l211_211478

theorem percent_problem (x : ℝ) (hx : 0.60 * 600 = 0.50 * x) : x = 720 :=
by
  sorry

end percent_problem_l211_211478


namespace number_of_dogs_l211_211091

def legs_in_pool : ℕ := 24
def human_legs : ℕ := 4
def legs_per_dog : ℕ := 4

theorem number_of_dogs : (legs_in_pool - human_legs) / legs_per_dog = 5 :=
by
  sorry

end number_of_dogs_l211_211091


namespace greatest_prime_factor_341_l211_211251

theorem greatest_prime_factor_341 : ∃ p, Nat.Prime p ∧ p ≥ 17 ∧ (∀ q, Nat.Prime q ∧ q ∣ 341 → q ≤ p) ∧ p = 19 := by
  sorry

end greatest_prime_factor_341_l211_211251


namespace ca1_l211_211144

theorem ca1 {
  a b : ℝ
} (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 := 
by
  sorry

end ca1_l211_211144


namespace find_quadratic_eq_with_given_roots_l211_211602

theorem find_quadratic_eq_with_given_roots (A z x1 x2 : ℝ) 
  (h1 : A * z * x1^2 + x1 * x1 + x2 = 0) 
  (h2 : A * z * x2^2 + x1 * x2 + x2 = 0) : 
  (A * z * x^2 + x1 * x - x2 = 0) :=
by
  sorry

end find_quadratic_eq_with_given_roots_l211_211602


namespace probability_of_scoring_l211_211486

theorem probability_of_scoring :
  ∀ (p : ℝ), (p + (1 / 3) * p = 1) → (p = 3 / 4) → (p * (1 - p) = 3 / 16) :=
by
  intros p h1 h2
  sorry

end probability_of_scoring_l211_211486


namespace powers_of_i_cyclic_l211_211097

theorem powers_of_i_cyclic {i : ℂ} (h_i_squared : i^2 = -1) :
  i^(66) + i^(103) = -1 - i :=
by {
  -- Providing the proof steps as sorry.
  -- This is a placeholder for the actual proof.
  sorry
}

end powers_of_i_cyclic_l211_211097


namespace driving_speed_ratio_l211_211098

theorem driving_speed_ratio
  (x : ℝ) (y : ℝ)
  (h1 : y = 2 * x) :
  y / x = 2 := by
  sorry

end driving_speed_ratio_l211_211098


namespace total_people_in_house_l211_211628

-- Define the number of people in various locations based on the given conditions.
def charlie_and_susan := 2
def sarah_and_friends := 5
def people_in_bedroom := charlie_and_susan + sarah_and_friends
def people_in_living_room := 8

-- Prove the total number of people in the house is 14.
theorem total_people_in_house : people_in_bedroom + people_in_living_room = 14 := by
  -- Here we can use Lean's proof system, but we skip with 'sorry'
  sorry

end total_people_in_house_l211_211628


namespace james_pays_per_episode_l211_211989

-- Conditions
def minor_characters : ℕ := 4
def major_characters : ℕ := 5
def pay_per_minor_character : ℕ := 15000
def multiplier_major_payment : ℕ := 3

-- Theorems and Definitions needed
def pay_per_major_character : ℕ := pay_per_minor_character * multiplier_major_payment
def total_pay_minor : ℕ := minor_characters * pay_per_minor_character
def total_pay_major : ℕ := major_characters * pay_per_major_character
def total_pay_per_episode : ℕ := total_pay_minor + total_pay_major

-- Main statement to prove
theorem james_pays_per_episode : total_pay_per_episode = 285000 := by
  sorry

end james_pays_per_episode_l211_211989


namespace clothing_store_earnings_l211_211640

-- Definitions for the given conditions
def num_shirts : ℕ := 20
def num_jeans : ℕ := 10
def cost_per_shirt : ℕ := 10
def cost_per_jeans : ℕ := 2 * cost_per_shirt

-- Theorem statement
theorem clothing_store_earnings : 
  (num_shirts * cost_per_shirt + num_jeans * cost_per_jeans = 400) := 
sorry

end clothing_store_earnings_l211_211640


namespace relation_between_abc_l211_211512

theorem relation_between_abc (a b c : ℕ) (h₁ : a = 3 ^ 44) (h₂ : b = 4 ^ 33) (h₃ : c = 5 ^ 22) : a > b ∧ b > c :=
by
  -- Proof goes here
  sorry

end relation_between_abc_l211_211512


namespace equilateral_triangle_sum_l211_211922

noncomputable def equilateral_triangle (a b c : Complex) (s : ℝ) : Prop :=
  Complex.abs (a - b) = s ∧ Complex.abs (b - c) = s ∧ Complex.abs (c - a) = s

theorem equilateral_triangle_sum (a b c : Complex):
  equilateral_triangle a b c 18 →
  Complex.abs (a + b + c) = 36 →
  Complex.abs (b * c + c * a + a * b) = 432 := by
  intros h_triangle h_sum
  sorry

end equilateral_triangle_sum_l211_211922


namespace ordered_triple_unique_l211_211734

variable (a b c : ℝ)

theorem ordered_triple_unique
  (h_pos_a : a > 4)
  (h_pos_b : b > 4)
  (h_pos_c : c > 4)
  (h_eq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) :
  (a, b, c) = (12, 10, 8) := 
sorry

end ordered_triple_unique_l211_211734


namespace cost_of_camel_l211_211898

theorem cost_of_camel
  (C H O E : ℝ)
  (h1 : 10 * C = 24 * H)
  (h2 : 16 * H = 4 * O)
  (h3 : 6 * O = 4 * E)
  (h4 : 10 * E = 140000) :
  C = 5600 :=
by
  -- Skipping the proof steps
  sorry

end cost_of_camel_l211_211898


namespace distance_of_each_race_l211_211884

theorem distance_of_each_race (d : ℝ) : 
  (∃ (d : ℝ), 
    let lake_speed := 3 
    let ocean_speed := 2.5 
    let num_races := 10 
    let total_time := 11
    let num_lake_races := num_races / 2
    let num_ocean_races := num_races / 2
    (num_lake_races * (d / lake_speed) + num_ocean_races * (d / ocean_speed) = total_time)) →
  d = 3 :=
sorry

end distance_of_each_race_l211_211884


namespace sequence_mono_iff_b_gt_neg3_l211_211573

theorem sequence_mono_iff_b_gt_neg3 (b : ℝ) : 
  (∀ n : ℕ, 1 ≤ n → (n + 1) ^ 2 + b * (n + 1) > n ^ 2 + b * n) → b > -3 := 
by
  sorry

end sequence_mono_iff_b_gt_neg3_l211_211573


namespace factorize_l211_211933

theorem factorize (m : ℝ) : m^3 - 4 * m = m * (m + 2) * (m - 2) :=
by
  sorry

end factorize_l211_211933


namespace factorize_l211_211932

theorem factorize (m : ℝ) : m^3 - 4 * m = m * (m + 2) * (m - 2) :=
by
  sorry

end factorize_l211_211932


namespace analogical_reasoning_correct_l211_211620

variable (a b c : Real)

theorem analogical_reasoning_correct (h : c ≠ 0) (h_eq : (a + b) * c = a * c + b * c) : 
  (a + b) / c = a / c + b / c :=
  sorry

end analogical_reasoning_correct_l211_211620


namespace at_least_one_not_less_than_two_l211_211569

theorem at_least_one_not_less_than_two (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x + 1/y) ≥ 2 ∨ (y + 1/z) ≥ 2 ∨ (z + 1/x) ≥ 2 :=
by
  sorry

end at_least_one_not_less_than_two_l211_211569


namespace map_representation_l211_211378

theorem map_representation (d1 d2 l1 l2 : ℕ)
  (h1 : l1 = 15)
  (h2 : d1 = 90)
  (h3 : l2 = 20) :
  d2 = 120 :=
by
  sorry

end map_representation_l211_211378


namespace problem_l211_211567

variables {a b c d : ℝ}

theorem problem (h1 : c + d = 14 * a) (h2 : c * d = 15 * b) (h3 : a + b = 14 * c) (h4 : a * b = 15 * d) (distinct : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a) :
  a + b + c + d = 3150 := sorry

end problem_l211_211567


namespace minimum_dot_product_l211_211128

noncomputable def ellipse (x y : ℝ) : Prop := (x^2 / 36) + (y^2 / 9) = 1

def K : (ℝ × ℝ) := (2, 0)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

theorem minimum_dot_product (M N : ℝ × ℝ) (hM : ellipse M.1 M.2) (hN : ellipse N.1 N.2) (h : dot_product (vector_sub M K) (vector_sub N K) = 0) :
  ∃ α β : ℝ, 0 ≤ α ∧ α < 2 * Real.pi ∧ 0 ≤ β ∧ β < 2 * Real.pi ∧ M = (6 * Real.cos α, 3 * Real.sin α) ∧ N = (6 * Real.cos β, 3 * Real.sin β) ∧
  (∃ C : ℝ, C = 23 / 3 ∧ ∀ M N, ellipse M.1 M.2 → ellipse N.1 N.2 → dot_product (vector_sub M K) (vector_sub N K) = 0 → dot_product (vector_sub M K) (vector_sub (vector_sub M N) K) >= C) :=
sorry

end minimum_dot_product_l211_211128


namespace line_equation_l211_211823

theorem line_equation (a b : ℝ) (h_intercept_eq : a = b) (h_pass_through : 3 * a + 2 * b = 2 * a + 5) : (3 + 2 = 5) ↔ (a = 5 ∧ b = 5) :=
sorry

end line_equation_l211_211823


namespace total_people_in_house_l211_211625

-- Definitions of initial condition in the bedroom and living room
def charlie_susan_in_bedroom : ℕ := 2
def sarah_and_friends_in_bedroom : ℕ := 5
def people_in_living_room : ℕ := 8

-- Prove the total number of people in the house is 15
theorem total_people_in_house : charlie_susan_in_bedroom + sarah_and_friends_in_bedroom + people_in_living_room = 15 :=
by
  -- sum the people in the bedroom (Charlie, Susan, Sarah, 4 friends)
  have bedroom_total : charlie_susan_in_bedroom + sarah_and_friends_in_bedroom = 7 := by sorry
  -- sum the people in the house (bedroom + living room)
  show bedroom_total + people_in_living_room = 15 from sorry

end total_people_in_house_l211_211625


namespace union_of_sets_l211_211154

theorem union_of_sets (A B : Set α) : A ∪ B = { x | x ∈ A ∨ x ∈ B } :=
by
  sorry

end union_of_sets_l211_211154


namespace greatest_prime_factor_of_341_l211_211250

theorem greatest_prime_factor_of_341 : ∃ p, Prime p ∧ p ∣ 341 ∧ ∀ q, Prime q ∧ q ∣ 341 → q ≤ p :=
by
  let factors := [3, 7, 17]
  have h : 341 = 17 * 3 * 7 := by sorry
  exists 17
  split
  · exact PrimeNat.prime_17
  split
  · exact dvd.intro (3 * 7) rfl
  intro q hpq hq
  have H : q ∈ factors := by sorry
  exact List.mem_le_of_mod_le (PrimeNat.factors_unique H)

end greatest_prime_factor_of_341_l211_211250


namespace probability_same_length_l211_211730

/-- Defining the set of all sides and diagonals of a regular hexagon. -/
def T : Finset ℚ := sorry

/-- There are exactly 6 sides in the set T. -/
def sides_count : ℕ := 6

/-- There are exactly 9 diagonals in the set T. -/
def diagonals_count : ℕ := 9

/-- The total number of segments in the set T. -/
def total_segments : ℕ := sides_count + diagonals_count

theorem probability_same_length :
  let prob_side := (6 : ℚ) / total_segments * (5 / (total_segments - 1))
  let prob_diagonal := (9 : ℚ) / total_segments * (4 / (total_segments - 1))
  prob_side + prob_diagonal = 17 / 35 := 
by
  admit

end probability_same_length_l211_211730


namespace letters_with_dot_not_line_l211_211716

-- Definitions from conditions
def D_inter_S : ℕ := 23
def S : ℕ := 42
def Total_letters : ℕ := 70

-- Problem statement
theorem letters_with_dot_not_line : (Total_letters - S - D_inter_S) = 5 :=
by sorry

end letters_with_dot_not_line_l211_211716


namespace combined_alloy_force_l211_211788

-- Define the masses and forces exerted by Alloy A and Alloy B
def mass_A : ℝ := 6
def force_A : ℝ := 30
def mass_B : ℝ := 3
def force_B : ℝ := 10

-- Define the combined mass and force
def combined_mass : ℝ := mass_A + mass_B
def combined_force : ℝ := force_A + force_B

-- Theorem statement
theorem combined_alloy_force :
  combined_force = 40 :=
by
  -- The proof is omitted.
  sorry

end combined_alloy_force_l211_211788


namespace sum_of_other_endpoint_coordinates_l211_211029

theorem sum_of_other_endpoint_coordinates
  (x₁ y₁ x₂ y₂ : ℝ)
  (hx : (x₁ + x₂) / 2 = 5)
  (hy : (y₁ + y₂) / 2 = -8)
  (endpt1 : x₁ = 7)
  (endpt2 : y₁ = -2) :
  x₂ + y₂ = -11 :=
sorry

end sum_of_other_endpoint_coordinates_l211_211029


namespace y_n_sq_eq_3_x_n_sq_add_1_l211_211120

def x : ℕ → ℤ
| 0       => 0
| 1       => 1
| (n + 1) => 4 * x n - x (n - 1)

def y : ℕ → ℤ
| 0       => 1
| 1       => 2
| (n + 1) => 4 * y n - y (n - 1)

theorem y_n_sq_eq_3_x_n_sq_add_1 (n : ℕ) : y n ^ 2 = 3 * (x n) ^ 2 + 1 :=
sorry

end y_n_sq_eq_3_x_n_sq_add_1_l211_211120


namespace combined_sale_price_correct_l211_211675

-- Define constants for purchase costs of items A, B, and C.
def purchase_cost_A : ℝ := 650
def purchase_cost_B : ℝ := 350
def purchase_cost_C : ℝ := 400

-- Define profit percentages for items A, B, and C.
def profit_percentage_A : ℝ := 0.40
def profit_percentage_B : ℝ := 0.25
def profit_percentage_C : ℝ := 0.30

-- Define the desired sale prices for items A, B, and C based on profit margins.
def sale_price_A : ℝ := purchase_cost_A * (1 + profit_percentage_A)
def sale_price_B : ℝ := purchase_cost_B * (1 + profit_percentage_B)
def sale_price_C : ℝ := purchase_cost_C * (1 + profit_percentage_C)

-- Calculate the combined sale price for all three items.
def combined_sale_price : ℝ := sale_price_A + sale_price_B + sale_price_C

-- The theorem stating that the combined sale price for all three items is $1867.50.
theorem combined_sale_price_correct :
  combined_sale_price = 1867.50 := 
sorry

end combined_sale_price_correct_l211_211675


namespace thomas_score_l211_211738

def average (scores : List ℕ) : ℚ := scores.sum / scores.length

variable (scores : List ℕ)

theorem thomas_score (h_length : scores.length = 19)
                     (h_avg_before : average scores = 78)
                     (h_avg_after : average ((98 :: scores)) = 79) :
  let thomas_score := 98
  thomas_score = 98 := sorry

end thomas_score_l211_211738


namespace opposite_of_negative_2020_is_2020_l211_211600

theorem opposite_of_negative_2020_is_2020 :
  ∃ x : ℤ, -2020 + x = 0 :=
by
  use 2020
  sorry

end opposite_of_negative_2020_is_2020_l211_211600


namespace map_distance_l211_211371

theorem map_distance (fifteen_cm_in_km : ℤ) (cm_to_km : ℕ): 
  fifteen_cm_in_km = 90 ∧ cm_to_km = 6 → 20 * cm_to_km = 120 := 
by 
  sorry

end map_distance_l211_211371


namespace interior_points_in_divided_square_l211_211489

theorem interior_points_in_divided_square :
  ∀ (n : ℕ), 
  (n = 2016) →
  ∃ (k : ℕ), 
  (∀ (t : ℕ), t = 180 * n) → 
  k = 1007 :=
by
  intros n hn
  use 1007
  sorry

end interior_points_in_divided_square_l211_211489


namespace map_representation_l211_211385

theorem map_representation (length_1_cm length_2_cm : ℕ) (distance_1_km : ℕ) :
  length_1_cm = 15 → distance_1_km = 90 →
  length_2_cm = 20 →
  distance_2_km = length_2_cm * (distance_1_km / length_1_cm) →
  distance_2_km = 120 :=
by
  intros h_len1 h_distance1 h_len2 h_calculate 
  rw [h_len1, h_distance1, h_len2] at h_calculate
  have scale := 90 / 15
  have distance_2_km := 20 * scale
  have : scale = 6 := by norm_num
  rw [this] at h_calculate
  norm_num at distance_2_km
  assumption
  sorry

end map_representation_l211_211385


namespace entree_cost_difference_l211_211130

theorem entree_cost_difference 
  (total_cost : ℕ)
  (entree_cost : ℕ)
  (dessert_cost : ℕ)
  (h1 : total_cost = 23)
  (h2 : entree_cost = 14)
  (h3 : total_cost = entree_cost + dessert_cost) :
  entree_cost - dessert_cost = 5 :=
by
  sorry

end entree_cost_difference_l211_211130


namespace votes_switched_l211_211721

theorem votes_switched (x : ℕ) (total_votes : ℕ) (half_votes : ℕ) 
  (votes_first_round : ℕ) (votes_second_round_winner : ℕ) (votes_second_round_loser : ℕ)
  (cond1 : total_votes = 48000)
  (cond2 : half_votes = total_votes / 2)
  (cond3 : votes_first_round = half_votes)
  (cond4 : votes_second_round_winner = half_votes + x)
  (cond5 : votes_second_round_loser = half_votes - x)
  (cond6 : votes_second_round_winner = 5 * votes_second_round_loser) :
  x = 16000 := by
  -- Proof will go here
  sorry

end votes_switched_l211_211721


namespace count_integers_satisfying_inequality_l211_211538

theorem count_integers_satisfying_inequality : ∃ (count : ℕ), count = 5 ∧ 
  ∀ x : ℤ, (0 ≤ x ∧ x ≤ 4) → ((x - 2) * (x - 2) ≤ 4) :=
begin
  existsi 5, 
  split,
  { refl },
  { intros x hx,
    cases hx with h1 h2,
    have : 0 ≤ (x - 2) ^ 2, from pow_two_nonneg (x - 2),
    exact le_antisymm this (show (x - 2) ^ 2 ≤ 4, by sorry) }
end

end count_integers_satisfying_inequality_l211_211538


namespace greatest_prime_factor_of_341_l211_211258

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem greatest_prime_factor_of_341 : ∃ p, is_prime p ∧ p ∣ 341 ∧ ∀ q, is_prime q → q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_of_341_l211_211258


namespace summation_indices_equal_l211_211821

theorem summation_indices_equal
  (a : ℕ → ℕ)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_bound : ∀ i, a i ≤ 100)
  (h_length : ∀ i, i < 16) :
  ∃ i j k l, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ a i + a j = a k + a l := 
by {
  sorry
}

end summation_indices_equal_l211_211821


namespace add_base6_numbers_l211_211296

def base6_to_base10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6^1 + c * 6^0

def base10_to_base6 (n : ℕ) : (ℕ × ℕ × ℕ) := 
  (n / 6^2, (n % 6^2) / 6^1, (n % 6^2) % 6^1)

theorem add_base6_numbers : 
  let n1 := 3 * 6^1 + 5 * 6^0
  let n2 := 2 * 6^1 + 5 * 6^0
  let sum := n1 + n2
  base10_to_base6 sum = (1, 0, 4) :=
by
  -- Proof steps would go here
  sorry

end add_base6_numbers_l211_211296


namespace polynomial_q_value_l211_211307

theorem polynomial_q_value :
  ∀ (p q d : ℝ),
    (d = 6) →
    (-p / 3 = -d) →
    (1 + p + q + d = - d) →
    q = -31 :=
by sorry

end polynomial_q_value_l211_211307


namespace uruguayan_goals_conceded_l211_211836

theorem uruguayan_goals_conceded (x : ℕ) (h : 14 = 9 + x) : x = 5 := by
  sorry

end uruguayan_goals_conceded_l211_211836


namespace intersection_A_B_l211_211124

open Set Real -- Opens necessary namespaces for sets and real numbers

-- Definitions for the sets A and B
def A : Set ℝ := {x | 1 / x < 1}
def B : Set ℝ := {x | x > -1}

-- The proof statement for the intersection of sets A and B
theorem intersection_A_B : A ∩ B = (Ioo (-1 : ℝ) 0) ∪ (Ioi 1) :=
by
  sorry -- Proof not included

end intersection_A_B_l211_211124


namespace map_length_scale_l211_211391

theorem map_length_scale (len1 len2 : ℕ) (dist1 dist2 : ℕ) (h1 : len1 = 15) (h2 : dist1 = 90) (h3 : len2 = 20) :
  dist2 = 120 :=
by
  sorry

end map_length_scale_l211_211391


namespace trigonometric_identity_l211_211305

theorem trigonometric_identity :
  (Real.cos (Real.pi / 3)) - (Real.tan (Real.pi / 4)) + (3 / 4) * (Real.tan (Real.pi / 6))^2 - (Real.sin (Real.pi / 6)) + (Real.cos (Real.pi / 6))^2 = 0 :=
by
  sorry

end trigonometric_identity_l211_211305


namespace train_speed_correct_l211_211073

-- Define the problem conditions
def length_of_train : ℝ := 300  -- length in meters
def time_to_cross_pole : ℝ := 18  -- time in seconds

-- Conversion factors
def meters_to_kilometers : ℝ := 0.001
def seconds_to_hours : ℝ := 1 / 3600

-- Define the conversions
def distance_in_kilometers := length_of_train * meters_to_kilometers
def time_in_hours := time_to_cross_pole * seconds_to_hours

-- Define the speed calculation
def speed_of_train := distance_in_kilometers / time_in_hours

-- The theorem to prove
theorem train_speed_correct : speed_of_train = 60 := 
by
  sorry

end train_speed_correct_l211_211073


namespace solution_set_of_inequality_l211_211568

variable {R : Type*} [LinearOrderedField R]

def odd_function (f : R → R) : Prop :=
  ∀ x : R, f (-x) = -f x

theorem solution_set_of_inequality
  (f : R → R)
  (odd_f : odd_function f)
  (h1 : f (-2) = 0)
  (h2 : ∀ (x1 x2 : R), x1 ≠ x2 ∧ 0 < x1 ∧ 0 < x2 → (x2 * f x1 - x1 * f x2) / (x1 - x2) < 0) :
  { x : R | (f x) / x < 0 } = { x : R | x < -2 } ∪ { x : R | x > 2 } := 
sorry

end solution_set_of_inequality_l211_211568


namespace find_q_l211_211699

open Real

noncomputable def q := (9 + 3 * Real.sqrt 5) / 2

theorem find_q (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 9) : q = (9 + 3 * Real.sqrt 5) / 2 :=
by
  sorry

end find_q_l211_211699


namespace exist_positive_integers_for_perfect_squares_l211_211662

theorem exist_positive_integers_for_perfect_squares :
  ∃ (x y : ℕ), (0 < x ∧ 0 < y) ∧ (∃ a b c : ℕ, x + y = a^2 ∧ x^2 + y^2 = b^2 ∧ x^3 + y^3 = c^2) :=
by
  sorry

end exist_positive_integers_for_perfect_squares_l211_211662


namespace find_complex_number_l211_211524

open Complex

theorem find_complex_number (z : ℂ) (h : z * (1 - I) = 2) : z = 1 + I :=
sorry

end find_complex_number_l211_211524


namespace uki_cupcakes_per_day_l211_211225

-- Define the conditions
def price_cupcake : ℝ := 1.50
def price_cookie : ℝ := 2
def price_biscuit : ℝ := 1
def daily_cookies : ℝ := 10
def daily_biscuits : ℝ := 20
def total_earnings : ℝ := 350
def days : ℝ := 5

-- Define the number of cupcakes baked per day
def cupcakes_per_day (x : ℝ) : Prop :=
  let earnings_cupcakes := price_cupcake * x * days
  let earnings_cookies := price_cookie * daily_cookies * days
  let earnings_biscuits := price_biscuit * daily_biscuits * days
  earnings_cupcakes + earnings_cookies + earnings_biscuits = total_earnings

-- The statement to be proven
theorem uki_cupcakes_per_day : cupcakes_per_day 20 :=
by 
  sorry

end uki_cupcakes_per_day_l211_211225


namespace find_m_p_pairs_l211_211670

theorem find_m_p_pairs (m p : ℕ) (h_prime : Nat.Prime p) (h_eq : ∃ (x : ℕ), 2^m * p^2 + 27 = x^3) :
  (m, p) = (1, 7) :=
sorry

end find_m_p_pairs_l211_211670


namespace wall_height_l211_211445

noncomputable def brick_volume : ℝ := 25 * 11.25 * 6

noncomputable def total_brick_volume : ℝ := brick_volume * 6400

noncomputable def wall_length : ℝ := 800

noncomputable def wall_width : ℝ := 600

theorem wall_height :
  ∀ (wall_volume : ℝ), 
  wall_volume = total_brick_volume → 
  wall_volume = wall_length * wall_width * 22.48 :=
by
  sorry

end wall_height_l211_211445


namespace lunch_special_cost_l211_211650

theorem lunch_special_cost (total_bill : ℕ) (num_people : ℕ) (cost_per_lunch_special : ℕ)
  (h1 : total_bill = 24) 
  (h2 : num_people = 3) 
  (h3 : cost_per_lunch_special = total_bill / num_people) : 
  cost_per_lunch_special = 8 := 
by
  sorry

end lunch_special_cost_l211_211650


namespace cosine_values_count_l211_211963

theorem cosine_values_count (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 360) (h3 : Real.cos x = -0.65) : 
  ∃ (n : ℕ), n = 2 := by
  sorry

end cosine_values_count_l211_211963


namespace greatest_prime_factor_of_341_l211_211245

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, is_prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, is_prime q ∧ q ∣ 341 → q ≤ p :=
begin
  -- Our goal is to show that p = 31 meets these conditions.
  sorry
end

end greatest_prime_factor_of_341_l211_211245


namespace solve_for_n_l211_211051

theorem solve_for_n (n : ℚ) (h : n + (n + 1) + (n + 2) + (n + 3) = 20) : 
    n = 3.5 :=
  sorry

end solve_for_n_l211_211051


namespace y_intercept_of_line_l211_211195

theorem y_intercept_of_line (m x1 y1 : ℝ) (x_intercept : x1 = 4) (y_intercept_at_x1_zero : y1 = 0) (m_value : m = -3) :
  ∃ b : ℝ, (∀ x y : ℝ, y = m * x + b ∧ x = 0 → y = b) ∧ b = 12 :=
by
  sorry

end y_intercept_of_line_l211_211195


namespace laundry_loads_l211_211899

-- Definitions based on conditions
def num_families : ℕ := 3
def people_per_family : ℕ := 4
def num_people : ℕ := num_families * people_per_family

def days : ℕ := 7
def towels_per_person_per_day : ℕ := 1
def total_towels : ℕ := num_people * days * towels_per_person_per_day

def washing_machine_capacity : ℕ := 14

-- Statement to prove
theorem laundry_loads : total_towels / washing_machine_capacity = 6 := 
by
  sorry

end laundry_loads_l211_211899


namespace gcd_105_90_l211_211672

theorem gcd_105_90 : Nat.gcd 105 90 = 15 :=
by
  sorry

end gcd_105_90_l211_211672


namespace percentage_increase_l211_211741

theorem percentage_increase (L : ℕ) (h : L + 60 = 240) : 
  ((60:ℝ) / (L:ℝ)) * 100 = 33.33 := 
by
  sorry

end percentage_increase_l211_211741


namespace rhombus_area_fraction_l211_211369

theorem rhombus_area_fraction :
  let grid_area := 36
  let vertices := [(2, 2), (4, 2), (3, 3), (3, 1)]
  let rhombus_area := 2
  rhombus_area / grid_area = 1 / 18 :=
by
  sorry

end rhombus_area_fraction_l211_211369


namespace melanie_balloons_l211_211006

theorem melanie_balloons (joan_balloons melanie_balloons total_balloons : ℕ)
  (h_joan : joan_balloons = 40)
  (h_total : total_balloons = 81) :
  melanie_balloons = total_balloons - joan_balloons :=
by
  sorry

end melanie_balloons_l211_211006


namespace probability_no_practice_l211_211190

def prob_has_practice : ℚ := 5 / 8

theorem probability_no_practice : 
  1 - prob_has_practice = 3 / 8 := 
by
  sorry

end probability_no_practice_l211_211190


namespace six_digit_number_consecutive_evens_l211_211941

theorem six_digit_number_consecutive_evens :
  ∃ n : ℕ,
    287232 = (2 * n - 2) * (2 * n) * (2 * n + 2) ∧
    287232 / 100000 = 2 ∧
    287232 % 10 = 2 :=
by
  sorry

end six_digit_number_consecutive_evens_l211_211941


namespace puppies_per_cage_l211_211068

theorem puppies_per_cage (initial_puppies sold_puppies cages remaining_puppies puppies_per_cage : ℕ)
  (h1 : initial_puppies = 18)
  (h2 : sold_puppies = 3)
  (h3 : cages = 3)
  (h4 : remaining_puppies = initial_puppies - sold_puppies)
  (h5 : puppies_per_cage = remaining_puppies / cages) :
  puppies_per_cage = 5 := by
  sorry

end puppies_per_cage_l211_211068


namespace Raven_age_l211_211712

-- Define the conditions
def Phoebe_age_current : Nat := 10
def Phoebe_age_in_5_years : Nat := Phoebe_age_current + 5

-- Define the hypothesis that in 5 years Raven will be 4 times as old as Phoebe
def Raven_in_5_years (R : Nat) : Prop := R + 5 = 4 * Phoebe_age_in_5_years

-- State the theorem to be proved
theorem Raven_age : ∃ R : Nat, Raven_in_5_years R ∧ R = 55 :=
by
  sorry

end Raven_age_l211_211712


namespace ellipse_standard_equation_l211_211605

theorem ellipse_standard_equation :
  ∃ (a b c : ℝ),
    2 * a = 10 ∧
    c / a = 3 / 5 ∧
    b^2 = a^2 - c^2 ∧
    (∀ x y : ℝ, (x^2 / 16) + (y^2 / 25) = 1) :=
by
  sorry

end ellipse_standard_equation_l211_211605


namespace probability_at_least_one_black_ball_l211_211554

def total_balls : ℕ := 10
def red_balls : ℕ := 6
def black_balls : ℕ := 4
def selected_balls : ℕ := 4

theorem probability_at_least_one_black_ball :
  (∃ (p : ℚ), p = 13 / 14 ∧ 
  (number_of_ways_to_choose4_balls_has_at_least_1_black / number_of_ways_to_choose4_balls) = p) :=
by
  sorry

end probability_at_least_one_black_ball_l211_211554


namespace range_of_a_l211_211832

noncomputable def is_decreasing (a : ℝ) : Prop :=
∀ n : ℕ, 0 < n → n ≤ 6 → (1 - 3 * a) * n + 10 * a > (1 - 3 * a) * (n + 1) + 10 * a ∧ 0 < a ∧ a < 1 ∧ ((1 - 3 * a) * 6 + 10 * a > 1)

theorem range_of_a (a : ℝ) : is_decreasing a ↔ (1/3 < a ∧ a < 5/8) :=
sorry

end range_of_a_l211_211832


namespace general_term_an_sum_bn_Tn_l211_211514

-- Problem 1
theorem general_term_an (a_n S_n : ℕ → ℝ) (h_a1 : a_n 1 = 1/2)
  (h_cond : ∀ n, a_n n + S_n n = 1) :
  ∀ n, a_n n = 1 / 2^n :=
sorry

-- Problem 2
theorem sum_bn_Tn (b_n : ℕ → ℝ) (T_n : ℕ → ℝ)
  (h_b_n_def : ∀ n, b_n n = n / 2^n)
  (h_T_n_def : ∀ n, T_n n = ∑ k in finset.range n, b_n (k + 1)) :
  ∀ n, T_n n = 2 - ((n + 2) / 2^n) :=
sorry

end general_term_an_sum_bn_Tn_l211_211514


namespace map_length_scale_l211_211393

theorem map_length_scale (len1 len2 : ℕ) (dist1 dist2 : ℕ) (h1 : len1 = 15) (h2 : dist1 = 90) (h3 : len2 = 20) :
  dist2 = 120 :=
by
  sorry

end map_length_scale_l211_211393


namespace simplify_expression_l211_211656

theorem simplify_expression : 
  (Real.sqrt 12) + (Real.sqrt 4) * ((Real.sqrt 5 - Real.pi) ^ 0) - (abs (-2 * Real.sqrt 3)) = 2 := 
by 
  sorry

end simplify_expression_l211_211656


namespace altitude_segment_product_eq_half_side_diff_square_l211_211164

noncomputable def altitude_product (a b c t m m_1: ℝ) :=
  m * m_1 = (b^2 + c^2 - a^2) / 2

theorem altitude_segment_product_eq_half_side_diff_square {a b c t m m_1: ℝ}
  (hm : m = 2 * t / a)
  (hm_1 : m_1 = a * (b^2 + c^2 - a^2) / (4 * t)) :
  altitude_product a b c t m m_1 :=
by sorry

end altitude_segment_product_eq_half_side_diff_square_l211_211164


namespace find_number_l211_211906

theorem find_number (x : ℝ) (h : 97 * x - 89 * x = 4926) : x = 615.75 :=
by
  sorry

end find_number_l211_211906


namespace pie_eating_contest_l211_211977

def pies_eaten (Adam Bill Sierra Taylor: ℕ) : ℕ :=
  Adam + Bill + Sierra + Taylor

theorem pie_eating_contest (Bill : ℕ) 
  (Adam_eq_Bill_plus_3 : ∀ B: ℕ, Adam = B + 3)
  (Sierra_eq_2times_Bill : ∀ B: ℕ, Sierra = 2 * B)
  (Sierra_eq_12 : Sierra = 12)
  (Taylor_eq_avg : ∀ A B S: ℕ, Taylor = (A + B + S) / 3)
  : pies_eaten Adam Bill Sierra Taylor = 36 := sorry

end pie_eating_contest_l211_211977


namespace sum_of_square_areas_l211_211791

variable (WX XZ : ℝ)

theorem sum_of_square_areas (hW : WX = 15) (hX : XZ = 20) : WX^2 + XZ^2 = 625 := by
  sorry

end sum_of_square_areas_l211_211791


namespace numberOfWaysToChooseLeadershipStructure_correct_l211_211912

noncomputable def numberOfWaysToChooseLeadershipStructure : ℕ :=
  12 * 11 * 10 * Nat.choose 9 3 * Nat.choose 6 3

theorem numberOfWaysToChooseLeadershipStructure_correct :
  numberOfWaysToChooseLeadershipStructure = 221760 :=
by
  simp [numberOfWaysToChooseLeadershipStructure]
  -- Add detailed simplification/proof steps here if required
  sorry

end numberOfWaysToChooseLeadershipStructure_correct_l211_211912


namespace football_team_lineup_count_l211_211431

theorem football_team_lineup_count :
  let team_members := 12
  let offensive_lineman_choices := 5
  let remaining_choices := team_members - 1
  let quarterback_choices := remaining_choices
  let running_back_choices := remaining_choices - 1
  let wide_receiver_choices := remaining_choices - 2
  let tight_end_choices := remaining_choices - 3

  offensive_lineman_choices * quarterback_choices * running_back_choices * wide_receiver_choices * tight_end_choices = 39600 :=
by
  let team_members := 12
  let offensive_lineman_choices := 5
  let remaining_choices := team_members - 1
  let quarterback_choices := remaining_choices
  let running_back_choices := remaining_choices - 1
  let wide_receiver_choices := remaining_choices - 2
  let tight_end_choices := remaining_choices - 3
  
  exact sorry

end football_team_lineup_count_l211_211431


namespace find_y_l211_211708

theorem find_y (x y : ℝ) (h1 : 2 * (x - y) = 12) (h2 : x + y = 14) : y = 4 := 
by
  sorry

end find_y_l211_211708


namespace gcd_36_60_eq_12_l211_211889

theorem gcd_36_60_eq_12 :
  ∃ (g : ℕ), g = Nat.gcd 36 60 ∧ g = 12 := by
  -- Defining the conditions:
  let a := 36
  let b := 60
  have fact_a : a = 2^2 * 3^2 := rfl
  have fact_b : b = 2^2 * 3 * 5 := rfl
  
  -- The statement to prove:
  sorry

end gcd_36_60_eq_12_l211_211889


namespace van_capacity_l211_211094

theorem van_capacity (s a v : ℕ) (h1 : s = 2) (h2 : a = 6) (h3 : v = 2) : (s + a) / v = 4 := by
  sorry

end van_capacity_l211_211094


namespace team_won_five_games_l211_211635
-- Import the entire Mathlib library

-- Number of games played (given as a constant)
def numberOfGamesPlayed : ℕ := 10

-- Number of losses definition based on the ratio condition
def numberOfLosses : ℕ := numberOfGamesPlayed / 2

-- The number of wins is defined as the total games played minus the number of losses
def numberOfWins : ℕ := numberOfGamesPlayed - numberOfLosses

-- Proof statement: The number of wins is 5
theorem team_won_five_games :
  numberOfWins = 5 := by
  sorry

end team_won_five_games_l211_211635


namespace fixed_numbers_in_diagram_has_six_solutions_l211_211723

-- Define the problem setup and constraints
def is_divisor (m n : ℕ) : Prop := ∃ k, n = k * m

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Formulating the main proof statement
theorem fixed_numbers_in_diagram_has_six_solutions : 
  ∃ (a b c k : ℕ),
    (14 * 4 * a = 14 * 6 * c) ∧
    (4 * a = 6 * c) ∧
    (2 * a = 3 * c) ∧
    (∃ k, c = 2 * k ∧ a = 3 * k) ∧
    (14 * 4 * 3 * k = 3 * k * b * 2 * k) ∧
    (∃ k, 56 * k = 6 * k^2 * b) ∧
    (b = 28 / k) ∧
    ((is_divisor k 28) ∧
     (k = 1 ∨ k = 2 ∨ k = 4 ∨ k = 7 ∨ k = 14 ∨ k = 28)) ∧
    (6 = 6) := sorry

end fixed_numbers_in_diagram_has_six_solutions_l211_211723


namespace remaining_fish_count_l211_211093

def initial_fish_counts : Type := (ℕ, ℕ, ℕ, ℕ)
def sold_fish_counts : Type := (ℕ, ℕ, ℕ, ℕ)

-- Define the initial number of fish
def initial_counts : initial_fish_counts := (94, 76, 89, 58)

-- Define the number of fish sold
def sold_counts : sold_fish_counts := (30, 48, 17, 24)

theorem remaining_fish_count : 
  let (guppies, angelfish, tiger_sharks, oscar_fish) := initial_counts in
  let (sold_guppies, sold_angelfish, sold_tiger_sharks, sold_oscar_fish) := sold_counts in
  guppies - sold_guppies + (angelfish - sold_angelfish) + (tiger_sharks - sold_tiger_sharks) + (oscar_fish - sold_oscar_fish) = 198 :=
by
  sorry

end remaining_fish_count_l211_211093


namespace area_of_triangle_with_sides_13_12_5_l211_211652

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

end area_of_triangle_with_sides_13_12_5_l211_211652


namespace map_scale_l211_211395

theorem map_scale (represents_15cm_as_km : 15 * 6 = 90) :
  20 * 6 = 120 :=
by
  sorry

end map_scale_l211_211395


namespace values_of_k_for_exactly_one_real_solution_l211_211949

variable {k : ℝ}

def quadratic_eq (k : ℝ) : Prop := 3 * k^2 + 42 * k - 573 = 0

theorem values_of_k_for_exactly_one_real_solution :
  quadratic_eq k ↔ k = 8 ∨ k = -22 := by
  sorry

end values_of_k_for_exactly_one_real_solution_l211_211949


namespace return_trip_time_l211_211643

-- conditions 
variables (d p w : ℝ) (h1 : d = 90 * (p - w)) (h2 : ∀ t : ℝ, t = d / p → d / (p + w) = t - 15)

--  statement
theorem return_trip_time :
  ∃ t : ℝ, t = 30 ∨ t = 45 :=
by
  -- placeholder proof 
  sorry

end return_trip_time_l211_211643


namespace map_length_representation_l211_211408

theorem map_length_representation :
  ∀ (length1 length2 km1 : ℝ), 
  length1 = 15 ∧ km1 = 90 ∧ length2 = 20 →
  (length2 * (km1 / length1) = 120) :=
by
  intros length1 length2 km1
  intro h
  cases h with hl1 h
  cases h with hkm1 hl2
  rw [hl1, hkm1, hl2]
  simp
  sorry

end map_length_representation_l211_211408


namespace white_pieces_total_l211_211882

theorem white_pieces_total (B W : ℕ) 
  (h_total_pieces : B + W = 300) 
  (h_total_piles : 100 * 3 = B + W) 
  (h_piles_1_white : {n : ℕ | n = 27}) 
  (h_piles_2_3_black : {m : ℕ | m = 42}) 
  (h_piles_3_black_3_white : 15 = 15) :
  W = 158 :=
by
  sorry

end white_pieces_total_l211_211882


namespace truck_travel_distance_l211_211074

variable (d1 d2 g1 g2 : ℝ)
variable (rate : ℝ)

-- Define the conditions
axiom condition1 : d1 = 300
axiom condition2 : g1 = 10
axiom condition3 : rate = d1 / g1
axiom condition4 : g2 = 15

-- Define the goal
theorem truck_travel_distance : d2 = rate * g2 := by
  -- axiom assumption placeholder
  exact sorry

end truck_travel_distance_l211_211074


namespace no_possible_values_of_k_l211_211919

theorem no_possible_values_of_k :
  ¬(∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p + q = 65) :=
by
  sorry

end no_possible_values_of_k_l211_211919


namespace greatest_prime_factor_of_341_l211_211248

theorem greatest_prime_factor_of_341 : ∃ p, Prime p ∧ p ∣ 341 ∧ ∀ q, Prime q ∧ q ∣ 341 → q ≤ p :=
by
  let factors := [3, 7, 17]
  have h : 341 = 17 * 3 * 7 := by sorry
  exists 17
  split
  · exact PrimeNat.prime_17
  split
  · exact dvd.intro (3 * 7) rfl
  intro q hpq hq
  have H : q ∈ factors := by sorry
  exact List.mem_le_of_mod_le (PrimeNat.factors_unique H)

end greatest_prime_factor_of_341_l211_211248


namespace find_triples_l211_211312

theorem find_triples (a b c : ℕ) (h₁ : a ≥ b) (h₂ : b ≥ c) (h₃ : a^3 + 9 * b^2 + 9 * c + 7 = 1997) :
  (a = 10 ∧ b = 10 ∧ c = 10) :=
by sorry

end find_triples_l211_211312
