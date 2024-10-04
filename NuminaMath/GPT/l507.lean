import Mathlib

namespace inequality_sqrt_sum_l507_507026

theorem inequality_sqrt_sum (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
by
  sorry

end inequality_sqrt_sum_l507_507026


namespace polynomial_sum_l507_507649

theorem polynomial_sum (x : ℂ) (h1 : x ≠ 1) (h2 : x^2017 - 3*x + 2 = 0) : 
  x^2016 + x^2015 + ... + x + 1 = 3 :=
sorry

end polynomial_sum_l507_507649


namespace simplify_expression_l507_507046

theorem simplify_expression : 
  let a := 7^4 + 4^5 
  let b := (2^3 - (-2)^2)^2 
  a * b = 54800 :=
by 
  let a := 7^4 + 4^5
  let b := (2^3 - (-2)^2)^2 
  show a * b = 54800 from sorry

end simplify_expression_l507_507046


namespace chloe_recycled_pounds_l507_507644

noncomputable def pounds_recycled_per_point : ℕ := 6
noncomputable def points_earned : ℕ := 5
noncomputable def friends_recycled : ℕ := 2

theorem chloe_recycled_pounds : 
    let total_pounds := pounds_recycled_per_point * points_earned in
    let chloe_recycled := total_pounds - friends_recycled in 
    chloe_recycled = 28 :=
by
  let total_pounds := pounds_recycled_per_point * points_earned
  let chloe_recycled := total_pounds - friends_recycled
  have h : chloe_recycled = 28 := by sorry
  exact h

end chloe_recycled_pounds_l507_507644


namespace triangle_perimeter_l507_507024

theorem triangle_perimeter (A B C D E F : Point) (h_square : square A B C D) (h_E : E ∈ LineSegment B C) (h_F : F ∈ LineSegment C D) (h_angle : ∠ E A F = 45) :
  perimeter (Triangle.mk C E F) = 2 :=
sorry

end triangle_perimeter_l507_507024


namespace joanie_popcorn_l507_507579

def popcornKernelsToCupsRatio : ℚ := 4 / 2 -- 2 tbsp make 4 cups means 4/2 cups per tbsp

def mitchellCups : ℚ := 4
def milesAndDavisCups : ℚ := 6
def cliffCups : ℚ := 3

def totalKernels : ℚ := 8

def totalOtherCups : ℚ := mitchellCups + milesAndDavisCups + cliffCups

def totalPopcornCups : ℚ := totalKernels * popcornKernelsToCupsRatio

theorem joanie_popcorn (cups : ℚ) : cups = 3 :=
by
  have totalOtherCups := totalOtherCups
  have totalPopcornCups := totalPopcornCups
  exact (totalPopcornCups - totalOtherCups) = cups
  sorry

end joanie_popcorn_l507_507579


namespace count_valid_three_digit_numbers_l507_507740

theorem count_valid_three_digit_numbers : 
  let total_numbers := 900
  let excluded_numbers := 9 * 10 * 9
  let valid_numbers := total_numbers - excluded_numbers
  valid_numbers = 90 :=
by
  let total_numbers := 900
  let excluded_numbers := 9 * 10 * 9
  let valid_numbers := total_numbers - excluded_numbers
  have h1 : valid_numbers = 900 - 810 := by rfl
  have h2 : 900 - 810 = 90 := by norm_num
  exact h1.trans h2

end count_valid_three_digit_numbers_l507_507740


namespace calc_subtract_l507_507205

-- Define the repeating decimal
def repeating_decimal := (11 : ℚ) / 9

-- Define the problem statement
theorem calc_subtract : 3 - repeating_decimal = (16 : ℚ) / 9 := by
  sorry

end calc_subtract_l507_507205


namespace arithmetic_mean_of_fractions_l507_507911

theorem arithmetic_mean_of_fractions : 
  (3 : ℚ) / 8 + (5 : ℚ) / 9 = 2 * (67 : ℚ) / 144 :=
by {
  rw [(show (3 : ℚ) / 8 + (5 : ℚ) / 9 = (27 : ℚ) / 72 + (40 : ℚ) / 72, by sorry),
      (show (27 : ℚ) / 72 + (40 : ℚ) / 72 = (67 : ℚ) / 72, by sorry),
      (show (2 : ℚ) * (67 : ℚ) / 144 = (67 : ℚ) / 72, by sorry)]
}

end arithmetic_mean_of_fractions_l507_507911


namespace value_in_box_l507_507955

theorem value_in_box (x : ℤ) (h : 5 + x = 10 + 20) : x = 25 := by
  sorry

end value_in_box_l507_507955


namespace arithmetic_mean_eq_l507_507900

theorem arithmetic_mean_eq :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = (67 : ℚ) / 144 :=
by
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  have h₁ : a = 3 / 8 := by rfl
  have h₂ : b = 5 / 9 := by rfl
  sorry

end arithmetic_mean_eq_l507_507900


namespace sequence_term_formula_l507_507756

theorem sequence_term_formula (S : ℕ → ℤ) (a : ℕ → ℤ) (h_sum : ∀ n, S n = 2 * n^2 - 3 * n) :
  (∀ n, a n = if n = 1 then -1 else 4 * n - 5) :=
by
  sorry

end sequence_term_formula_l507_507756


namespace polar_line_equation_l507_507236

/-- A line that passes through a given point in polar coordinates and is parallel to the polar axis
    has a specific polar coordinate equation. -/
theorem polar_line_equation (r : ℝ) (θ : ℝ) (h : r = 6 ∧ θ = π / 6) : θ = π / 6 :=
by
  /- We are given that the line passes through the point \(C(6, \frac{\pi}{6})\) which means
     \(r = 6\) and \(θ = \frac{\pi}{6}\). Since the line is parallel to the polar axis, 
     the angle \(θ\) remains the same. Therefore, the polar coordinate equation of the line 
     is simply \(θ = \frac{\pi}{6}\). -/
  sorry

end polar_line_equation_l507_507236


namespace solve_congruence_13n_15_mod_47_l507_507437

theorem solve_congruence_13n_15_mod_47 :
  ∃ n : ℤ, (0 ≤ n ∧ n < 47) ∧ 13 * n ≡ 15 [MOD 47] ∧ n = 24 :=
by
  use 24
  split
  · split
    · norm_num -- shows 0 ≤ 24
    · norm_num -- shows 24 < 47
  · split
    · norm_num
    rw [← int_mod_eq_mod, ← int.modeq_iff_dvd' (by norm_num : 47 ≠ 0), mul_comm 13 24]
    norm_num
    exact (by norm_num : (312 - 15) % 47 = 0)
  · refl

end solve_congruence_13n_15_mod_47_l507_507437


namespace arithmetic_mean_eq_l507_507907

theorem arithmetic_mean_eq :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = (67 : ℚ) / 144 :=
by
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  have h₁ : a = 3 / 8 := by rfl
  have h₂ : b = 5 / 9 := by rfl
  sorry

end arithmetic_mean_eq_l507_507907


namespace arithmetic_mean_of_fractions_l507_507909

theorem arithmetic_mean_of_fractions : 
  (3 : ℚ) / 8 + (5 : ℚ) / 9 = 2 * (67 : ℚ) / 144 :=
by {
  rw [(show (3 : ℚ) / 8 + (5 : ℚ) / 9 = (27 : ℚ) / 72 + (40 : ℚ) / 72, by sorry),
      (show (27 : ℚ) / 72 + (40 : ℚ) / 72 = (67 : ℚ) / 72, by sorry),
      (show (2 : ℚ) * (67 : ℚ) / 144 = (67 : ℚ) / 72, by sorry)]
}

end arithmetic_mean_of_fractions_l507_507909


namespace total_digits_first_2500_even_integers_l507_507118

theorem total_digits_first_2500_even_integers :
  let even_nums := List.range' 2 5000 (λ n, 2*n)  -- List of the first 2500 even integers
  let one_digit_nums := even_nums.filter (λ n, n < 10)
  let two_digit_nums := even_nums.filter (λ n, 10 ≤ n ∧ n < 100)
  let three_digit_nums := even_nums.filter (λ n, 100 ≤ n ∧ n < 1000)
  let four_digit_nums := even_nums.filter (λ n, 1000 ≤ n ∧ n ≤ 5000)
  let sum_digits := one_digit_nums.length * 1 + two_digit_nums.length * 2 + three_digit_nums.length * 3 + four_digit_nums.length * 4
  in sum_digits = 9448 := by sorry

end total_digits_first_2500_even_integers_l507_507118


namespace smallest_base_10_integer_is_19_l507_507951

noncomputable def smallest_base_10_integer : ℕ :=
  if h : ∃ (A C : ℕ), A < 8 ∧ C < 6 ∧ 9 * A = 7 * C 
  then
    let ⟨A, C, h1, h2, h3⟩ := h
    in 9 * A
  else 
    0

theorem smallest_base_10_integer_is_19 : smallest_base_10_integer = 19 := 
  by 
    -- proof placeholder
    sorry

end smallest_base_10_integer_is_19_l507_507951


namespace distance_relation_possible_l507_507770

-- Define a structure representing points in 2D space
structure Point where
  x : ℤ
  y : ℤ

-- Define the artificial geometry distance function (Euclidean distance)
def varrho (p1 p2 : Point) : ℝ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2).sqrt

-- Define the non-collinearity condition for points A, B, and C
def non_collinear (A B C : Point) : Prop :=
  ¬(A.x = B.x ∧ B.x = C.x) ∧ ¬(A.y = B.y ∧ B.y = C.y)

theorem distance_relation_possible :
  ∃ (A B C : Point), non_collinear A B C ∧ varrho A C ^ 2 + varrho B C ^ 2 = varrho A B ^ 2 :=
by
  sorry

end distance_relation_possible_l507_507770


namespace divisors_of_power_sum_of_primes_l507_507008

theorem divisors_of_power_sum_of_primes (n : ℕ) (p : ℕ → ℕ) (h_distinct : ∀ i j, i ≠ j → p i ≠ p j)
  (h_prime : ∀ i < n, Nat.prime (p i)) (h_gt_3 : ∀ i < n, p i > 3) :
  ∃ (k : ℕ), k ≥ 4^n ∧ (2^(p 0 * ... * p (n-1)) + 1) = k :=
sorry

end divisors_of_power_sum_of_primes_l507_507008


namespace compare_a_b_c_l507_507710

def a : ℝ := 2^(1/2)
def b : ℝ := 3^(1/3)
def c : ℝ := 5^(1/5)

theorem compare_a_b_c : b > a ∧ a > c :=
  by
  sorry

end compare_a_b_c_l507_507710


namespace sin_240_eq_neg_sqrt3_div_2_l507_507660

theorem sin_240_eq_neg_sqrt3_div_2 :
  (∀ α : ℝ, sin (180 * real.pi / 180 + α) = -sin α) →
  sin (60 * real.pi / 180) = (real.sqrt 3) / 2 →
  sin (240 * real.pi / 180) = -(real.sqrt 3) / 2 :=
by
  intro h1 h2
  sorry

end sin_240_eq_neg_sqrt3_div_2_l507_507660


namespace smallest_integer_with_eight_factors_l507_507553

theorem smallest_integer_with_eight_factors :
  ∃ n : ℕ, (∀ m : ℕ, (∃ k : ℕ, m has_factors k ∧ k = 8) → m ≥ n) ∧ n = 24 :=
by
  sorry

end smallest_integer_with_eight_factors_l507_507553


namespace advertisements_shown_l507_507661

theorem advertisements_shown (advertisement_duration : ℕ) (cost_per_minute : ℕ) (total_cost : ℕ) :
  advertisement_duration = 3 →
  cost_per_minute = 4000 →
  total_cost = 60000 →
  total_cost / (advertisement_duration * cost_per_minute) = 5 :=
by
  sorry

end advertisements_shown_l507_507661


namespace repeating_decimal_sum_l507_507856

/--
The number 3.17171717... can be written as a reduced fraction x/y where x = 314 and y = 99.
We aim to prove that the sum of x and y is 413.
-/
theorem repeating_decimal_sum : 
  let x := 314
  let y := 99
  (x + y) = 413 := 
by
  sorry

end repeating_decimal_sum_l507_507856


namespace figure_102_unit_squares_l507_507231

theorem figure_102_unit_squares : 
  ∃ g : ℕ → ℕ, (∀ n, g n = 2 * n^2 - 2 * n + 1) ∧ g 1 = 1 ∧ g 2 = 5 ∧ g 3 = 13 ∧ g 4 = 25 ∧ g 102 = 20605 :=
by {
  use (λ n, 2 * n^2 - 2 * n + 1),
  split,
  { intros n,
    refl,
  },
  split, { refl },
  split, { refl },
  split, { refl },
  split, { refl },
  refl
}


end figure_102_unit_squares_l507_507231


namespace total_number_of_digits_l507_507121

-- Definitions based on identified conditions
def first2500EvenIntegers := {n : ℕ | n % 2 = 0 ∧ 1 ≤ n ∧ n ≤ 5000}

-- Theorem statement based on the equivalent proof problem
theorem total_number_of_digits : 
  (first2500EvenIntegers.count_digits = 9448) :=
sorry

end total_number_of_digits_l507_507121


namespace jamesons_sword_length_l507_507214

theorem jamesons_sword_length (c j j' : ℕ) (hC: c = 15) 
  (hJ: j = c + 23) (hJJ: j' = j - 5) : 
  j' = 2 * c + 3 := by 
  sorry

end jamesons_sword_length_l507_507214


namespace smallest_integer_with_eight_factors_l507_507514

theorem smallest_integer_with_eight_factors:
  ∃ n : ℕ, ∀ (d : ℕ), d > 0 → d ∣ n → 8 = (divisor_count n) 
  ∧ (∀ m : ℕ, m > 0 → (∀ (d : ℕ), d > 0 → d ∣ m → 8 = (divisor_count m)) → n ≤ m) → n = 24 :=
begin
  sorry
end

end smallest_integer_with_eight_factors_l507_507514


namespace unmeasurable_weights_set_l507_507500

-- Define the set of weights used in the problem.
def weights : Set ℕ := {1, 2, 3, 8, 16, 32}

-- Define the set of all measurable weights using these weights.
def measurable_weights : Set ℕ :=
  {w | ∃ (a b c d e f : ℕ), a ∈ {0, 1} ∧ b ∈ {0, 1} ∧ c ∈ {0, 1} ∧ d ∈ {0, 1} ∧ e ∈ {0, 1} ∧ f ∈ {0, 1} ∧
                              a * 1 + b * 2 + c * 3 + d * 8 + e * 16 + f * 32 = w ∧ w ≤ 60}

-- Define the set of weights that cannot be measured.
def unmeasurable_weights : Set ℕ := {w | w ≤ 60} \ measurable_weights

theorem unmeasurable_weights_set :
  unmeasurable_weights = {7, 15, 23, 31, 39, 47, 55} :=
by
  sorry

end unmeasurable_weights_set_l507_507500


namespace y_intercepts_of_parabola_number_of_y_intercepts_l507_507736

theorem y_intercepts_of_parabola : ∀ y : ℝ, (∃ y : ℝ, 0 = 3 * y ^ 2 - 6 * y + 3) ↔ (∃ y : ℝ, (y - 1) ^ 2 = 0) :=
by
  sorry

theorem number_of_y_intercepts : 1 = finset.card (finset.filter (fun y => ∃ y : ℝ, 0 = 3 * y ^ 2 - 6 * y + 3) {y | true}) :=
by 
  sorry

end y_intercepts_of_parabola_number_of_y_intercepts_l507_507736


namespace total_payment_correct_l507_507413

-- Define the conditions for each singer
def firstSingerPayment : ℝ := 2 * 25
def secondSingerPayment : ℝ := 3 * 35
def thirdSingerPayment : ℝ := 4 * 20
def fourthSingerPayment : ℝ := 2.5 * 30

def firstSingerTip : ℝ := 0.15 * firstSingerPayment
def secondSingerTip : ℝ := 0.20 * secondSingerPayment
def thirdSingerTip : ℝ := 0.25 * thirdSingerPayment
def fourthSingerTip : ℝ := 0.18 * fourthSingerPayment

def firstSingerTotal : ℝ := firstSingerPayment + firstSingerTip
def secondSingerTotal : ℝ := secondSingerPayment + secondSingerTip
def thirdSingerTotal : ℝ := thirdSingerPayment + thirdSingerTip
def fourthSingerTotal : ℝ := fourthSingerPayment + fourthSingerTip

-- Define the total amount paid
def totalPayment : ℝ := firstSingerTotal + secondSingerTotal + thirdSingerTotal + fourthSingerTotal

-- The proof problem: Prove the total amount paid
theorem total_payment_correct : totalPayment = 372 := by
  sorry

end total_payment_correct_l507_507413


namespace arithmetic_mean_of_fractions_l507_507916

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 := 
by 
  sorry

end arithmetic_mean_of_fractions_l507_507916


namespace work_rate_solution_l507_507991

theorem work_rate_solution (y : ℕ) (hy : y > 0) : 
  ∃ z : ℕ, z = (y^2 + 3 * y) / (2 * y + 3) :=
by
  sorry

end work_rate_solution_l507_507991


namespace base_number_whole_l507_507318

-- Define the problem with the given conditions
def problem_statement (x : ℝ) : Prop :=
  let digitsAfterDecimal (n : ℝ) : ℕ := sorry in
  (digitsAfterDecimal ((x^4 * 3.456789)^11) = 22) ∧ digitsAfterDecimal 3.456789 = 6

-- The theorem to be proved
theorem base_number_whole (x : ℝ) (h : problem_statement x) : x ∈ ℤ :=
sorry

end base_number_whole_l507_507318


namespace inequality_sqrt_sum_l507_507025

theorem inequality_sqrt_sum (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
by
  sorry

end inequality_sqrt_sum_l507_507025


namespace james_monthly_earnings_l507_507784

theorem james_monthly_earnings :
  let initial_subscribers := 150
  let gifted_subscribers := 50
  let rate_per_subscriber := 9
  let total_subscribers := initial_subscribers + gifted_subscribers
  let total_earnings := total_subscribers * rate_per_subscriber
  total_earnings = 1800 := by
  sorry

end james_monthly_earnings_l507_507784


namespace smallest_number_with_eight_factors_l507_507535

-- Definition of a function to count the number of distinct positive factors
def count_distinct_factors (n : ℕ) : ℕ := (List.range n).filter (fun d => d > 0 ∧ n % d = 0).length

-- Statement to prove the main problem
theorem smallest_number_with_eight_factors (n : ℕ) :
  count_distinct_factors n = 8 → n = 24 :=
by
  sorry

end smallest_number_with_eight_factors_l507_507535


namespace cone_volume_with_same_radius_and_height_l507_507084

theorem cone_volume_with_same_radius_and_height (r h : ℝ) 
  (Vcylinder : ℝ) (Vcone : ℝ) (h1 : Vcylinder = 54 * Real.pi) 
  (h2 : Vcone = (1 / 3) * Vcylinder) : Vcone = 18 * Real.pi :=
by sorry

end cone_volume_with_same_radius_and_height_l507_507084


namespace hyperbola_eccentricity_correct_l507_507320

noncomputable def hyperbola_eccentricity (a b : ℝ) (asymptote : ℝ) : ℝ :=
  if a > 0 ∧ asymptote = real.sqrt 3 ∧ a^2 = 6 then
    let e := real.sqrt (1 + (b^2 / a^2)) in
    e
  else
    0

theorem hyperbola_eccentricity_correct :
  ∀ (b : ℝ),
  hyperbola_eccentricity (real.sqrt 6) b (real.sqrt 3) = (2 * real.sqrt 3) / 3 :=
by
  sorry

end hyperbola_eccentricity_correct_l507_507320


namespace arithmetic_mean_eq_l507_507903

theorem arithmetic_mean_eq :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = (67 : ℚ) / 144 :=
by
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  have h₁ : a = 3 / 8 := by rfl
  have h₂ : b = 5 / 9 := by rfl
  sorry

end arithmetic_mean_eq_l507_507903


namespace probability_log3_integer_is_correct_l507_507177

-- Define a four-digit number N
def is_four_digit (N : ℕ) : Prop := 1000 ≤ N ∧ N ≤ 9999

-- Define the predicate that log base 3 of N is an integer
def log3_is_integer (N : ℕ) : Prop := ∃ k : ℕ, N = 3^k

-- The main theorem: the probability that log base 3 of a randomly chosen four-digit number N is an integer
theorem probability_log3_integer_is_correct :
  (∑' (N : ℕ) in finset.filter is_four_digit (finset.range 10000), if log3_is_integer N then 1 else 0) / 9000 = 1 / 4500 := 
sorry

end probability_log3_integer_is_correct_l507_507177


namespace number_of_digits_of_2500_even_integers_l507_507124

theorem number_of_digits_of_2500_even_integers : 
  let even_integers := List.range (5000 : Nat) in
  let first_2500_even := List.filter (fun n => n % 2 = 0) even_integers in
  List.length (List.join (first_2500_even.map (fun n => n.toDigits Nat))) = 9448 :=
by
  sorry

end number_of_digits_of_2500_even_integers_l507_507124


namespace solution_set_a_range_m_l507_507294

theorem solution_set_a (a : ℝ) :
  (∀ x : ℝ, |x - a| ≤ 3 ↔ -6 ≤ x ∧ x ≤ 0) ↔ a = -3 :=
by
  sorry

theorem range_m (m : ℝ) :
  (∀ x : ℝ, |x + 3| + |x + 8| ≥ 2 * m) ↔ m ≤ 5 / 2 :=
by
  sorry

end solution_set_a_range_m_l507_507294


namespace sin_alpha_minus_pi_over_3_l507_507692

theorem sin_alpha_minus_pi_over_3 (α : ℝ) (h : cos (α + π / 6) = -1 / 3) :
  sin (α - π / 3) = 1 / 3 :=
sorry

end sin_alpha_minus_pi_over_3_l507_507692


namespace problem_statement_l507_507080

noncomputable def range_of_a (a : ℝ) : Prop :=
  a ∈ set.Iic 1 ∪ set.Ici 3

theorem problem_statement (a : ℝ) :
  (λ (x : ℝ), (x^2 + (2 * a^2 + 2) * x - a^2 + 4 * a - 7) /
              (x^2 + (a^2 + 4 * a - 5) * x - a^2 + 4 * a - 7)) < 0 ∧
  ((x : ℝ), x ∈ { x | ((x^2 + (a^2 + 4 * a - 5) * x - a^2 + 4 * a - 7) < 0)}.sum < 4) → 
  range_of_a a :=
sorry

end problem_statement_l507_507080


namespace find_m_l507_507305

theorem find_m (m : ℝ) : 
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (-2, m)
  let c : ℝ × ℝ := (fst a + fst b, snd a + snd b)
  (fst a) * (fst c) + (snd a) * (snd c) = 0 → 
  m = -8 / 3 :=
by 
  sorry

end find_m_l507_507305


namespace slope_angle_of_vertical_line_l507_507478

theorem slope_angle_of_vertical_line :
  ∀ x : ℝ, x = -1 → ∃ angle : ℝ, angle = 90 := 
by intro x hx
   use 90
   sorry -- this is where the actual proof goes

end slope_angle_of_vertical_line_l507_507478


namespace problem_1_l507_507984

theorem problem_1 : 
  let a := 0.1111111111 + 0.0222222222 + 0.0033333333 + 0.0004444444 + 0.0000555555 + 0.0000066666 + 0.00000077777 + 0.00000008888 + 0.00000000999 in
  a = 0.13717421 := 
sorry

end problem_1_l507_507984


namespace sum_of_intersections_l507_507700

theorem sum_of_intersections (A : Finset ℕ) (hA : A.card = n) :
  let S := (∑ i in Finset.range(n+1), i * (Nat.choose n i) * (3^(n-i) - 1)) / 2 in
  S = n * (2^(2*n-3) - 2^(n-2)) := by
  sorry

end sum_of_intersections_l507_507700


namespace probability_sum_equals_6_l507_507563

theorem probability_sum_equals_6 : 
  let possible_outcomes := 36
  let favorable_outcomes := 5
  (favorable_outcomes / possible_outcomes : ℚ) = 5 / 36 := 
by 
  sorry

end probability_sum_equals_6_l507_507563


namespace part_I_part_II_part_III_l507_507720

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x + 1)) / x

theorem part_I (h0 : ∀ x > 0, f' x < 0) : ∀ x > 0, f x > f (x + 1) := 
sorry

theorem part_II (H : ∀ x > 0, f x > k / (x + 1)) : k ≤ 3 :=
sorry

theorem part_III (n : ℕ) (H2 : ∀ x > 0, Real.log (x + 1) > 2 - 3 / (x + 1)) :
  (List.prod (List.map (λ i => 1 + i * (i + 1)) (List.range n))) > Real.exp (2 * n - 3) :=
sorry

-- f' is the derivative of the function f
def f' (x : ℝ) : ℝ := - (1 / x^2) * (1 / (x + 1) + Real.log (x + 1))

end part_I_part_II_part_III_l507_507720


namespace equation_of_line_intercepts_l507_507963

noncomputable def line_equation (x y : ℝ) : Prop :=
  x / 3 - y / 5 = 1

theorem equation_of_line_intercepts (a b : ℝ) (h₁ : a = 3) (h₂ : b = -5) :
    ∀ (x y : ℝ), x / a + y / b = 1 ↔ line_equation x y :=
by
  intros x y
  rw [h₁, h₂]
  simp [line_equation]
  split
  { intro h
    rw [mul_div_comm, mul_div_comm] at h
    assumption }
  { intro h
    rw [mul_div_comm, mul_div_comm]
    assumption }

end equation_of_line_intercepts_l507_507963


namespace margo_total_distance_l507_507808

theorem margo_total_distance (time_to_friend : ℝ) (time_back_home : ℝ) (average_rate : ℝ)
  (total_time_hours : ℝ) (total_miles : ℝ) :
  time_to_friend = 12 / 60 ∧
  time_back_home = 24 / 60 ∧
  total_time_hours = (12 / 60) + (24 / 60) ∧
  average_rate = 3 ∧
  total_miles = average_rate * total_time_hours →
  total_miles = 1.8 :=
by
  sorry

end margo_total_distance_l507_507808


namespace combined_original_price_l507_507017

def original_price_shoes (discount_price : ℚ) (discount_rate : ℚ) : ℚ := discount_price / (1 - discount_rate)

def original_price_dress (discount_price : ℚ) (discount_rate : ℚ) : ℚ := discount_price / (1 - discount_rate)

theorem combined_original_price (shoes_price : ℚ) (shoes_discount : ℚ) (dress_price : ℚ) (dress_discount : ℚ) 
  (h_shoes : shoes_discount = 0.20 ∧ shoes_price = 480) 
  (h_dress : dress_discount = 0.30 ∧ dress_price = 350) : 
  original_price_shoes shoes_price shoes_discount + original_price_dress dress_price dress_discount = 1100 := by
  sorry

end combined_original_price_l507_507017


namespace sum_of_first_50_odd_numbers_eq_2500_sum_of_first_n_odd_numbers_eq_n_squared_l507_507018

theorem sum_of_first_50_odd_numbers_eq_2500 : ∑ i in finset.range 50, (2 * i + 1) = 2500 :=
by sorry

theorem sum_of_first_n_odd_numbers_eq_n_squared (n : ℕ) : ∑ i in finset.range n, (2 * i + 1) = n^2 :=
by sorry

end sum_of_first_50_odd_numbers_eq_2500_sum_of_first_n_odd_numbers_eq_n_squared_l507_507018


namespace find_larger_number_l507_507020

-- Define the problem conditions
variable (x y : ℕ)
hypothesis h1 : y = x + 10
hypothesis h2 : x + y = 34

-- Formalize the goal to prove
theorem find_larger_number : y = 22 := by
  -- placeholders in lean statement to skip the proof
  sorry

end find_larger_number_l507_507020


namespace smallest_number_with_eight_factors_l507_507534

-- Definition of a function to count the number of distinct positive factors
def count_distinct_factors (n : ℕ) : ℕ := (List.range n).filter (fun d => d > 0 ∧ n % d = 0).length

-- Statement to prove the main problem
theorem smallest_number_with_eight_factors (n : ℕ) :
  count_distinct_factors n = 8 → n = 24 :=
by
  sorry

end smallest_number_with_eight_factors_l507_507534


namespace tetrahedron_area_projection_l507_507329

variables (P A B C : Type) [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace C]

noncomputable def S := sorry -- Placeholder for area function on triangles

-- Given sides opposite to angles in triangle ABC
variables (a b c : ℝ)
variables (AngleA AngleB AngleC : ℝ)

-- Given condition in the triangle ABC
axiom sides_condition : a = b * Real.cos AngleC + c * Real.cos AngleB

-- Areas of triangles
variables (S_ABC S_PAB S_PBC S_PAC : ℝ)

-- Dihedral angles in the tetrahedron P-ABC
variables (alpha beta gamma : ℝ)

-- Required proof
theorem tetrahedron_area_projection :
  S_ABC = S_PAB * Real.cos alpha + S_PBC * Real.cos beta + S_PAC * Real.cos gamma :=
sorry

end tetrahedron_area_projection_l507_507329


namespace value_of_x_l507_507135

theorem value_of_x (u w z y x : ℤ) (h1 : u = 95) (h2 : w = u + 10) (h3 : z = w + 25) (h4 : y = z + 15) (h5 : x = y + 12) : x = 157 := by
  sorry

end value_of_x_l507_507135


namespace complex_ab_value_l507_507316

theorem complex_ab_value (a b : ℝ) (i : ℂ) (h : i = complex.I) 
    (h1 : a + b * i = 5 / (1 + 2 * complex.I)) : a * b = -2 :=
sorry

end complex_ab_value_l507_507316


namespace water_channel_depth_l507_507461

-- Definitions of the given conditions
def top_width := 14
def bottom_width := 8
def area := 880

-- Definition of the depth (height)
def depth (a b A : ℝ) : ℝ := 2 * A / (a + b)

-- Goal statement
theorem water_channel_depth : depth top_width bottom_width area = 80 := 
by
  -- To be completed as a proof
  sorry

end water_channel_depth_l507_507461


namespace total_spent_on_video_games_l507_507787

theorem total_spent_on_video_games (cost_basketball cost_racing : ℝ) (h_ball : cost_basketball = 5.20) (h_race : cost_racing = 4.23) : 
  cost_basketball + cost_racing = 9.43 :=
by
  sorry

end total_spent_on_video_games_l507_507787


namespace isosceles_triangle_smallest_angle_l507_507625

-- Given conditions:
-- 1. The triangle is isosceles
-- 2. One angle is 40% larger than the measure of a right angle

theorem isosceles_triangle_smallest_angle :
  ∃ (A B C : ℝ), 
  A + B + C = 180 ∧ 
  (A = B ∨ A = C ∨ B = C) ∧ 
  (∃ (large_angle : ℝ), large_angle = 90 + 0.4 * 90 ∧ (A = large_angle ∨ B = large_angle ∨ C = large_angle)) →
  (A = 27 ∨ B = 27 ∨ C = 27) := sorry

end isosceles_triangle_smallest_angle_l507_507625


namespace car_can_crash_into_house_l507_507416

-- Definitions of the problem constraints
def car_moves (n : ℕ) (grid : Fin n × Fin n) : Prop := sorry

def can_exit (n : ℕ) (grid : Fin n × Fin n) : Prop := 
  ∀ cell : (Fin n × Fin n), ∃ path : List (Fin n × Fin n), 
    path.head = cell ∧ path.last ∈ { (x, y) | x = 0 ∨ y = 0 } ∧ 
    ∀ p ∈ path, car_moves n p

theorem car_can_crash_into_house (n : ℕ) (house : Fin n × Fin n) : 
  ∀ (signs : Fin n × Fin n → ℤ), ∃ (path : List (Fin n × Fin n)), 
  path.head ∈ { (0, y) | y ∈ Fin n } ∨ path.head ∈ { (x, 0) | x ∈ Fin n } ∧ 
  path.last = house ∧ 
  ∀ p ∈ path, car_moves n p :=
sorry

end car_can_crash_into_house_l507_507416


namespace infinite_sum_converges_to_3_l507_507217

theorem infinite_sum_converges_to_3 :
  (∑' k : ℕ, (7 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))) = 3 :=
by
  sorry

end infinite_sum_converges_to_3_l507_507217


namespace find_a_range_l507_507249

noncomputable def point_in_second_quadrant (z1 z2 : ℂ) : Prop :=
  (z1 / z2).re < 0 ∧ (z1 / z2).im > 0

theorem find_a_range (a : ℝ) (h : a + 2 * Complex.I = z1) (h1 : z1 ^ 2 = 3 - Complex.I) (h2 : z1 = a + 2 * Complex.I) (h3 : z2 = 3 - Complex.I) :
  point_in_second_quadrant z1 z2 ↔ a ∈ set.Ioo (-1/2) (-1/3) :=
sorry

end find_a_range_l507_507249


namespace find_corresponding_element_l507_507699

variable {A B : Type}
variables f : A → B
variables {x y : ℝ}

def f_map (x y : ℝ) : ℝ × ℝ := (x - y, x + y)
def A_set : set (ℝ × ℝ) := {p | ∃ x y : ℝ, p = (x, y)}
def B_set : set (ℝ × ℝ) := {p | ∃ x y : ℝ, p = (x - y, x + y)}

theorem find_corresponding_element :
  ∃ (x y : ℝ), f_map x y = (-2, 4) ∧ (x, y) = (1, 3) :=
by
  use 1, 3
  split
  · exact rfl
  · exact rfl

end find_corresponding_element_l507_507699


namespace perimeter_of_square_with_area_625_cm2_l507_507607

noncomputable def side_length (a : ℝ) : ℝ := 
  real.sqrt a

noncomputable def perimeter (s : ℝ) : ℝ :=
  4 * s

theorem perimeter_of_square_with_area_625_cm2 :
  perimeter (side_length 625) = 100 :=
by
  sorry

end perimeter_of_square_with_area_625_cm2_l507_507607


namespace equal_clubs_and_students_l507_507345

theorem equal_clubs_and_students (S C : ℕ) 
  (h1 : ∀ c : ℕ, c < C → ∃ (m : ℕ → Prop), (∃ p, m p ∧ p = 3))
  (h2 : ∀ s : ℕ, s < S → ∃ (n : ℕ → Prop), (∃ p, n p ∧ p = 3)) :
  S = C := 
by
  sorry

end equal_clubs_and_students_l507_507345


namespace parabola_equation_correct_line_AB_fixed_point_l507_507355

-- Define the conditions and problem
def hyperbola_center := (0, 0)
def hyperbola_focus := (0, 2)
def parabola_focus := hyperbola_focus
def parabola_vertex := hyperbola_center
def parabola_equation (x y : ℝ) := x^2 = 8 * y

-- Define the fixed point given the conditions of the problem
def fixed_point := (-4, 10)

-- Define the given point P on the parabola
def point_P (t : ℝ) (h : t > 0) := (4, 2)

-- Define generic points A and B on the parabola
def point_A (x₁ : ℝ) := (x₁, (x₁^2) / 8)
def point_B (x₂ : ℝ) := (x₂, (x₂^2) / 8)

-- The main statement of the problem
theorem parabola_equation_correct :
  ∀ x y : ℝ, (x, y) ∈ { p : ℝ × ℝ | parabola_equation p.1 p.2 } ↔ x^2 = 8 * y := by
  sorry

theorem line_AB_fixed_point :
  ∀ (x₁ x₂ : ℝ), (point_P 4 (by norm_num), point_A x₁, point_B x₂),
  (¬(point_A x₁ = point_P _ (by norm_num)) ∧ 
  ¬(point_B x₂ = point_P _ (by norm_num))) ∧
  ((x₁ - 4) * (x₂ - 4) + ((x₁^2 / 8) - 2) * ((x₂^2 / 8) - 2) = 0) →
  ∃ (p : ℝ × ℝ), p = fixed_point := by
  sorry

end parabola_equation_correct_line_AB_fixed_point_l507_507355


namespace union_of_sets_l507_507726

def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {x | x ∈ {0, 3, 9}}

theorem union_of_sets : M ∪ N = ({0, 1, 3, 9} : Set ℕ) :=
by
  sorry

end union_of_sets_l507_507726


namespace postage_cost_l507_507685

theorem postage_cost (W : ℝ) : 
    let floor := λ x : ℝ, ⌊x⌋
    let ceil := λ x : ℝ, ⌈x⌉
    let cost := 6 * ceil W
    cost = -6 * floor (-W) := 
by
  -- The proof goes here
  sorry

end postage_cost_l507_507685


namespace points_ABD_collinear_l507_507730

variables (a b : Vector ℝ) (A B C D : Point)
variables (h_not_collinear : ¬ collinear a b)
variables (h_AB : vector_from A B = a + 2 • b)
variables (h_BC : vector_from B C = -3 • a + 7 • b)
variables (h_CD : vector_from C D = 4 • a - 5 • b)

theorem points_ABD_collinear : collinear A B D :=
sorry

end points_ABD_collinear_l507_507730


namespace arithmetic_sequence_z_l507_507102

-- Define the arithmetic sequence and value of z
theorem arithmetic_sequence_z (z : ℤ) (arith_seq : 9 + 27 = 2 * z) : z = 18 := 
by 
  sorry

end arithmetic_sequence_z_l507_507102


namespace yanna_kept_l507_507567

theorem yanna_kept (Y_0 : ℕ) (H1 : Y_0 = 60) : 
  let Z := Y_0 * 40 / 100 in
  let Y_1 := Y_0 - Z in
  let A := Y_1 * 25 / 100 in
  let Y_2 := Y_1 - A in
  let F := Y_2 / 2 in
  Y_2 - F = 15 := 
by
  sorry

end yanna_kept_l507_507567


namespace good_horse_catches_up_l507_507351

noncomputable def catch_up_days : ℕ := sorry

theorem good_horse_catches_up (x : ℕ) :
  (∀ (good_horse_speed slow_horse_speed head_start_duration : ℕ),
    good_horse_speed = 200 →
    slow_horse_speed = 120 →
    head_start_duration = 10 →
    200 * x = 120 * x + 120 * 10) →
  catch_up_days = x :=
by
  intro h
  have := h 200 120 10 rfl rfl rfl
  sorry

end good_horse_catches_up_l507_507351


namespace benjamin_walks_95_miles_in_a_week_l507_507202

def distance_to_work_one_way : ℕ := 6
def days_to_work : ℕ := 5
def distance_to_dog_walk_one_way : ℕ := 2
def dog_walks_per_day : ℕ := 2
def days_for_dog_walk : ℕ := 7
def distance_to_best_friend_one_way : ℕ := 1
def times_to_best_friend : ℕ := 1
def distance_to_convenience_store_one_way : ℕ := 3
def times_to_convenience_store : ℕ := 2

def total_distance_in_a_week : ℕ :=
  (days_to_work * 2 * distance_to_work_one_way) +
  (days_for_dog_walk * dog_walks_per_day * distance_to_dog_walk_one_way) +
  (times_to_convenience_store * 2 * distance_to_convenience_store_one_way) +
  (times_to_best_friend * 2 * distance_to_best_friend_one_way)

theorem benjamin_walks_95_miles_in_a_week :
  total_distance_in_a_week = 95 :=
by {
  simp [distance_to_work_one_way, days_to_work,
        distance_to_dog_walk_one_way, dog_walks_per_day,
        days_for_dog_walk, distance_to_best_friend_one_way,
        times_to_best_friend, distance_to_convenience_store_one_way,
        times_to_convenience_store, total_distance_in_a_week],
  sorry
}

end benjamin_walks_95_miles_in_a_week_l507_507202


namespace vector_b_magnitude_l507_507501

variables (a b : ℝ) (theta : ℝ)
noncomputable def magnitude_a := 2
noncomputable def perpendicular1 := (magnitude_a + b) * (a * cos theta) = 0
noncomputable def perpendicular2 := (2 * magnitude_a + b) * (b * cos theta) = 0

theorem vector_b_magnitude :
  (magnitude_a = 2) ∧
  (perpendicular1 = 0) ∧
  (perpendicular2 = 0) →
  (b = 2 * real.sqrt 2) :=
begin
  sorry
end

end vector_b_magnitude_l507_507501


namespace moes_mowing_time_l507_507415

-- Define the given conditions
def lawn_length : ℝ := 120 -- feet
def lawn_width : ℝ := 200 -- feet
def swath_width : ℝ := 30 / 12 -- inches to feet
def overlap : ℝ := 6 / 12 -- inches to feet
def effective_swath_width : ℝ := swath_width - overlap
def moes_speed : ℝ := 4000 -- feet per hour

-- Define the goal to prove
theorem moes_mowing_time :
  let strips_needed := lawn_width / effective_swath_width in
  let total_distance := strips_needed * lawn_length in
  let time_required := total_distance / moes_speed in
  time_required = 3 := by
  sorry

end moes_mowing_time_l507_507415


namespace total_digits_first_2500_even_integers_l507_507114

theorem total_digits_first_2500_even_integers :
  let even_nums := List.range' 2 5000 (λ n, 2*n)  -- List of the first 2500 even integers
  let one_digit_nums := even_nums.filter (λ n, n < 10)
  let two_digit_nums := even_nums.filter (λ n, 10 ≤ n ∧ n < 100)
  let three_digit_nums := even_nums.filter (λ n, 100 ≤ n ∧ n < 1000)
  let four_digit_nums := even_nums.filter (λ n, 1000 ≤ n ∧ n ≤ 5000)
  let sum_digits := one_digit_nums.length * 1 + two_digit_nums.length * 2 + three_digit_nums.length * 3 + four_digit_nums.length * 4
  in sum_digits = 9448 := by sorry

end total_digits_first_2500_even_integers_l507_507114


namespace distance_moved_from_B_l507_507185

theorem distance_moved_from_B (A B : Point) (s : ℝ) : 
  (s = sqrt 18) ∧ 
  (visible_red_area = visible_blue_area) → 
  distance A B = 2 * sqrt 6 := by
  sorry

end distance_moved_from_B_l507_507185


namespace simplify_expression_l507_507833

variable {R : Type} [AddCommGroup R] [Module ℤ R]

theorem simplify_expression (a b : R) :
  (25 • a + 70 • b) + (15 • a + 34 • b) - (12 • a + 55 • b) = 28 • a + 49 • b :=
by sorry

end simplify_expression_l507_507833


namespace smallest_positive_root_l507_507240

noncomputable def alpha := Real.arctan (14 / 3)
noncomputable def beta := Real.arctan (13 / 6)

theorem smallest_positive_root:
  ∀ (x : ℝ), (14 * Real.sin(3 * x) - 3 * Real.cos(3 * x) = 13 * Real.sin(2 * x) - 6 * Real.cos(2 * x)) →
  (0 < x) →
  x = (2 * Real.pi - alpha - beta) / 5 :=
by
  sorry

end smallest_positive_root_l507_507240


namespace clubs_students_equal_l507_507342

theorem clubs_students_equal
  (C E : ℕ)
  (h1 : ∃ N, N = 3 * C)
  (h2 : ∃ N, N = 3 * E) :
  C = E :=
by
  sorry

end clubs_students_equal_l507_507342


namespace circle_area_is_323pi_l507_507023

-- Define points A and B
def A : ℝ × ℝ := (2, 9)
def B : ℝ × ℝ := (14, 7)

-- Define that points A and B lie on circle ω
def on_circle_omega (A B C : ℝ × ℝ) (r : ℝ) : Prop :=
  (A.1 - C.1) ^ 2 + (A.2 - C.2) ^ 2 = r ^ 2 ∧
  (B.1 - C.1) ^ 2 + (B.2 - C.2) ^ 2 = r ^ 2

-- Define the tangent lines intersect at a point on the x-axis
def tangents_intersect_on_x_axis (A B : ℝ × ℝ) (C : ℝ × ℝ) (ω : (ℝ × ℝ) → ℝ): Prop := 
  ∃ x : ℝ, (A.1 - C.1) ^ 2 + (A.2 - C.2) ^ 2 = (B.1 - C.1) ^ 2 + (B.2 - C.2) ^ 2 ∧
             C.2 = 0

-- Problem statement to prove
theorem circle_area_is_323pi (C : ℝ × ℝ) (radius : ℝ) (on_circle_omega : on_circle_omega A B C radius)
  (tangents_intersect_on_x_axis : tangents_intersect_on_x_axis A B C omega) :
  π * radius ^ 2 = 323 * π :=
sorry

end circle_area_is_323pi_l507_507023


namespace fixed_point_parabola_l507_507794

theorem fixed_point_parabola (t : ℝ) : 4 * 3^2 + t * 3 - t^2 - 3 * t = 36 := by
  sorry

end fixed_point_parabola_l507_507794


namespace inverse_function_passing_point_l507_507279

theorem inverse_function_passing_point
  (f : ℝ → ℝ) (hf : Function.Bijective f) (h1 : f(4) = 3) : f⁻¹ 3 = 4 :=
by
  -- to prove
  sorry

end inverse_function_passing_point_l507_507279


namespace min_max_sum_square_l507_507402

theorem min_max_sum_square (n : ℕ) (a : Fin n → ℝ) (h1 : 2 ≤ n)
  (h2 : ∑ i, |a i| + |∑ i, a i| = 1) :
  (1 / (4 * n) : ℝ) ≤ ∑ i, (a i)^2 ∧ ∑ i, (a i)^2 ≤ 1 / 2 :=
sorry

end min_max_sum_square_l507_507402


namespace calculate_expression_l507_507208

theorem calculate_expression : 
  (-1)^(2023) - Real.sqrt 9 + abs (1 - Real.sqrt 2) - Real.cbrt (-8) = Real.sqrt 2 - 3 := 
sorry

end calculate_expression_l507_507208


namespace simplify_expression_l507_507239

theorem simplify_expression (b : ℝ) (h : b ≠ 1 / 2) : 1 - (2 / (1 + (b / (1 - 2 * b)))) = (3 * b - 1) / (1 - b) :=
by
    sorry

end simplify_expression_l507_507239


namespace min_x2_y2_z2_l507_507405

open Real

theorem min_x2_y2_z2 (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : x^2 + y^2 + z^2 ≥ 4 :=
sorry

end min_x2_y2_z2_l507_507405


namespace jia_winning_strategy_l507_507339

def regular_polygon (n : ℕ) := 
  ∃ (vertices : finset ℝ), vertices.card = n

def turn (player : Type) :=
  player

def draw_segment (vertices : finset ℝ) 
  (p1 p2 : ℝ) (segments : finset (ℝ × ℝ)) : Prop :=
  (p1 ∈ vertices ∧ p2 ∈ vertices ∧
  ∀ (s ∈ segments), ¬ intersects (p1, p2) s)

def rule (vertices : finset ℝ) (segments : finset (ℝ × ℝ)) :=
  ∀ p1 p2, (p1 ≠ p2) → 
  (p1 ∈ vertices ∧ p2 ∈ vertices) → 
  (∃ s, s ∈ segments → intersects (p1, p2) s)

noncomputable def initial_polygon : finset ℝ :=
  sorry

theorem jia_winning_strategy : 
  ∃ strategy : (vertices : finset ℝ) → 
  (segments : finset (ℝ × ℝ)) → 
  turn player, player = A → 
  (¬ rule (initial_polygon) (∅ : finset (ℝ × ℝ))) :=
sorry

end jia_winning_strategy_l507_507339


namespace smallest_integer_with_eight_factors_l507_507537

theorem smallest_integer_with_eight_factors : ∃ N : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) ∧ ∀ M : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) → N ≤ M :=
sorry -- Proof not provided.

end smallest_integer_with_eight_factors_l507_507537


namespace exists_natural_number_satisfying_conditions_l507_507467

theorem exists_natural_number_satisfying_conditions :
  ∃ N : ℕ, (∃ n : ℤ, N = 995 * (2 * n + 1989)) ∧
           (card { k : ℕ | ∃ m : ℤ, N = (k+1) * (2 * m + k) / 2 } = 1990) :=
sorry

end exists_natural_number_satisfying_conditions_l507_507467


namespace smallest_positive_period_centers_of_symmetry_maximum_value_minimum_value_l507_507282

noncomputable def f (x : ℝ) : ℝ := -2 * (Real.sin x)^2 + 2 * (Real.sin x) * (Real.cos x) + 1

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem centers_of_symmetry :
  ∀ k : ℤ, ∃ x, x = -Real.pi / 4 + k * Real.pi ∧ f (-x) = f x := sorry

theorem maximum_value :
  ∀ x : ℝ, f x ≤ 2 := sorry

theorem minimum_value :
  ∀ x : ℝ, f x ≥ -1 := sorry

end smallest_positive_period_centers_of_symmetry_maximum_value_minimum_value_l507_507282


namespace number_of_lines_through_point_in_triangle_l507_507312

-- Definitions
variable {P : Type} [Point P]
variable {A B C : P}

-- Given conditions: Point P is inside triangle ABC
def point_in_triangle (P : P) (A B C : P) : Prop := 
  -- (actual condition definition that Point P is inside ∆ABC would be defined here)
  sorry

-- Theorem statement
theorem number_of_lines_through_point_in_triangle (P : P) (A B C : P) (h : point_in_triangle P A B C) : 
  ∃ n : ℕ, n = 6 :=
  sorry

end number_of_lines_through_point_in_triangle_l507_507312


namespace common_ratio_is_one_third_l507_507480

-- Definitions
def is_arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum a

-- Hypotheses
variables {a : ℕ → ℝ} (q : ℝ)
hypothesis h1 : a 1 = a 0 * q
hypothesis h2 : a 2 = a 1 * q
hypothesis h3 : a 3 = a 2 * q

-- Given conditions
hypothesis h4 : is_arithmetic_sequence (sum_first_n_terms a 1) 
  (2 * sum_first_n_terms a 2) 
  (3 * sum_first_n_terms a 3)

-- Prove the common ratio of the sequence
theorem common_ratio_is_one_third : q = 1/3 := by
  sorry

end common_ratio_is_one_third_l507_507480


namespace sum_of_numbers_grouped_by_nearest_square_l507_507472

theorem sum_of_numbers_grouped_by_nearest_square (n : ℕ)
  (h : n ≤ 999999) :
  let odd_group := { x ∈ finset.range (999999 + 1) | ∃ k, x = k^2 ∧ odd k }
      even_group := { x ∈ finset.range (999999 + 1) | ∃ k, x = k^2 ∧ even k } in
  finset.sum odd_group id = finset.sum even_group id :=
sorry

end sum_of_numbers_grouped_by_nearest_square_l507_507472


namespace clubs_students_equal_l507_507341

theorem clubs_students_equal
  (C E : ℕ)
  (h1 : ∃ N, N = 3 * C)
  (h2 : ∃ N, N = 3 * E) :
  C = E :=
by
  sorry

end clubs_students_equal_l507_507341


namespace machines_work_together_l507_507412

theorem machines_work_together (a_time b_time : ℕ) (ha : a_time = 4) (hb : b_time = 12) : 
  let combined_time := 3 in
  combined_time = (a_time * b_time) / (a_time + b_time) :=
by
  have h1 : (a_time * b_time) / (a_time + b_time) = 12 / (4 + 12) := sorry
  have h2 : 12 / 16 = 3 := sorry
  calc 
    combined_time = 3 : by sorry  -- as proven with h1 and h2

end machines_work_together_l507_507412


namespace domain_of_function_l507_507066

theorem domain_of_function :
  ∀ x, (2 * x - 1 ≥ 0) ∧ (x^2 ≠ 1) → (x ≥ 1/2 ∧ x < 1) ∨ (x > 1) := 
sorry

end domain_of_function_l507_507066


namespace calculate_expression_l507_507212

theorem calculate_expression : 
  (1 - Real.sqrt 2)^0 + |(2 - Real.sqrt 5)| + (-1)^2022 - (1/3) * Real.sqrt 45 = 0 :=
by
  sorry

end calculate_expression_l507_507212


namespace isosceles_triangle_angle_l507_507628

theorem isosceles_triangle_angle (A B C : ℝ) (h_iso : A = C)
  (h_obtuse : B = 1.4 * 90) (h_sum : A + B + C = 180) :
  A = 27 :=
by
  have h1 : B = 126 from h_obtuse
  have h2 : A + C = 54 := by linarith [h1, h_sum]
  have h3 : 2 * A = 54 := by linarith [h_iso, h2]
  exact eq_div_of_mul_eq two_ne_zero h3

end isosceles_triangle_angle_l507_507628


namespace cups_of_flour_per_pound_of_pasta_l507_507811

-- Definitions from conditions
def pounds_of_pasta_per_rack : ℕ := 3
def racks_owned : ℕ := 3
def additional_rack_needed : ℕ := 1
def cups_per_bag : ℕ := 8
def bags_used : ℕ := 3

-- Derived definitions from above conditions
def total_cups_of_flour : ℕ := bags_used * cups_per_bag  -- 24 cups
def total_racks_needed : ℕ := racks_owned + additional_rack_needed  -- 4 racks
def total_pounds_of_pasta : ℕ := total_racks_needed * pounds_of_pasta_per_rack  -- 12 pounds

theorem cups_of_flour_per_pound_of_pasta (x : ℕ) :
  (total_cups_of_flour / total_pounds_of_pasta) = x → x = 2 :=
by
  intro h
  sorry

end cups_of_flour_per_pound_of_pasta_l507_507811


namespace partial_fraction_decomposition_l507_507666

theorem partial_fraction_decomposition :
  ∃ x y z : ℕ, 77 * x + 55 * y + 35 * z = 674 ∧ x + y + z = 14 :=
begin
  sorry
end

end partial_fraction_decomposition_l507_507666


namespace total_digits_used_l507_507109

theorem total_digits_used (n : ℕ) (h : n = 2500) : 
  let first_n_even := (finset.range (2 * n + 1)).filter (λ x, x % 2 = 0)
  let count_digits := λ x, if x < 10 then 1 else if x < 100 then 2 else if x < 1000 then 3 else 4
  let total_digits := first_n_even.sum (λ x, count_digits x)
  total_digits = 9444 :=
by sorry

end total_digits_used_l507_507109


namespace Daria_financial_result_l507_507982

noncomputable def convert_rubles_to_usd (rubles : ℝ) (selling_rate : ℝ) : ℝ :=
  rubles / selling_rate

noncomputable def calculate_usd_after_deposit (usd : ℝ) (annual_interest_rate : ℝ) (months : ℕ) : ℝ :=
  usd * (1 + (annual_interest_rate / 12) * months / 100)

noncomputable def convert_usd_to_rubles (usd : ℝ) (buying_rate : ℝ) : ℝ :=
  usd * buying_rate

def final_financial_result (initial_rubles : ℝ) (initial_selling_rate : ℝ) (annual_interest_rate : ℝ)
 (months : ℕ) (final_buying_rate : ℝ) : ℝ :=
  let usd := convert_rubles_to_usd initial_rubles initial_selling_rate
  let final_usd := calculate_usd_after_deposit usd annual_interest_rate months
  let final_rubles := convert_usd_to_rubles final_usd final_buying_rate
  initial_rubles - final_rubles

theorem Daria_financial_result :
  final_financial_result 60000 59.65 1.5 6 55.95 ≈ -3309 :=
by
  sorry

end Daria_financial_result_l507_507982


namespace find_a5_l507_507757

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Condition: The sum of the first n terms of the sequence {a_n} is represented by S_n = 2a_n - 1 (n ∈ ℕ)
axiom sum_of_terms (n : ℕ) : S n = 2 * (a n) - 1

-- Prove that a_5 = 16
theorem find_a5 : a 5 = 16 :=
  sorry

end find_a5_l507_507757


namespace proof_problem_l507_507715

variables (PA PB PC PE PF BC BF AC : ℝ)
variables [InnerProductSpace ℝ V]

-- Conditions
def condition1 : Prop :=
  inner PA PB = 1/2 ∧ inner PA PC = 1/2 ∧ inner PC PB = 1/2

def condition2 : Prop := PA = 2 • PE

def condition3 : Prop := BC = 2 • BF

-- Question
theorem proof_problem :
  condition1 PA PB PC ∧ condition2 PA PE ∧ condition3 BC BF →
  inner PA (BC + AC) = -1/2 :=
sorry

end proof_problem_l507_507715


namespace profit_without_discount_l507_507182

theorem profit_without_discount (CP : ℝ) (discount_percent : ℝ) (profit_percent : ℝ) 
  (h1 : CP = 100) (h2 : discount_percent = 0.05) (h3 : profit_percent = 0.2065) :
  let SP_without_discount := CP * (1 + profit_percent) in
  ((SP_without_discount - CP) / CP) * 100 = 20.65 :=
by
  let SP_discounted := CP * (1 - discount_percent)
  let actual_profit := profit_percent * CP
  let SP_actual := CP + actual_profit
  let expected_profit_percent := ((SP_actual - CP) / CP) * 100
  have h4 : SP_discounted = 95 := sorry
  have h5 : actual_profit = 20.65 := sorry
  have h6 : SP_actual = 120.65 := sorry
  exact eq.trans (eq.symm h5) (by { simp [expected_profit_percent, h2, h3]; ring })

end profit_without_discount_l507_507182


namespace exist_congruent_polygons_with_red_and_black_l507_507379

-- Definitions for the conditions
def is_regular_ngon (n : ℕ) (points : set (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ), radius > 0 ∧
  ∀ p ∈ points, let θ := atan2 (p.2 - center.2) (p.1 - center.1) in
  ∃ k : ℕ, k < n ∧ 
  (p = (center.1 + radius * cos (2 * k * Real.pi / n), center.2 + radius * sin (2 * k * Real.pi / n)))

def colored_ngon (n : ℕ) (p : ℕ) (points : set (ℝ × ℝ)) (red_points : set (ℝ × ℝ)) : Prop :=
  is_regular_ngon n points ∧
  |red_points| = p ∧
  (∀ point ∈ points, point ∈ red_points ∨ point ∉ red_points)

theorem exist_congruent_polygons_with_red_and_black 
  (n : ℕ) (p : ℕ) (points red_points : set (ℝ × ℝ)):
  n ≥ 6 →
  3 ≤ p →
  p < n - p →
  colored_ngon n p points red_points →
  ∃ (red_ngon black_ngon : set (ℝ × ℝ)),
    is_regular_ngon (nat.floor (p / 2) + 1) red_ngon ∧
    is_regular_ngon (nat.floor (p / 2) + 1) black_ngon ∧
    red_ngon ⊆ red_points ∧
    black_ngon ⊆ (points \ red_points) :=
by sorry

end exist_congruent_polygons_with_red_and_black_l507_507379


namespace smallest_integer_with_eight_factors_l507_507513

theorem smallest_integer_with_eight_factors:
  ∃ n : ℕ, ∀ (d : ℕ), d > 0 → d ∣ n → 8 = (divisor_count n) 
  ∧ (∀ m : ℕ, m > 0 → (∀ (d : ℕ), d > 0 → d ∣ m → 8 = (divisor_count m)) → n ≤ m) → n = 24 :=
begin
  sorry
end

end smallest_integer_with_eight_factors_l507_507513


namespace stuffed_animals_total_stuffed_animals_average_stuffed_animals_percentage_l507_507809

def McKenna : ℕ := 34
def Kenley : ℕ := 2 * McKenna
def Tenly : ℕ := Kenley + 5
def Total : ℕ := McKenna + Kenley + Tenly
def Average : ℕ := Total / 3
def Percentage : ℝ := (McKenna : ℝ) / (Total : ℝ) * 100

theorem stuffed_animals_total :
  Total = 175 := by
  sorry

theorem stuffed_animals_average :
  Average = 58 := by -- as Lean integers can't represent non-terminating decimals directly
  sorry

theorem stuffed_animals_percentage :
  Percentage ≈ 19.43 := by
  sorry

end stuffed_animals_total_stuffed_animals_average_stuffed_animals_percentage_l507_507809


namespace AD_bisects_angle_MDN_l507_507771

-- Definitions and conditions
variables {A B C D M N : Point}
variables {O : Circle}

-- Triangle ABC is acute-angled and inscribed in circle O
axiom inside_circle : ∀ {A B C}, InscribedTriangle O A B C

-- Tangents at B and C intersecting the tangent at A at points M and N respectively
axiom tangents : TangentAt O B ∩ TangentAt O C ∩ TangentAt O A = {M, N}

-- AD is the altitude on side BC
axiom altitude : Altitude A D B C

-- Prove AD bisects angle MDN
theorem AD_bisects_angle_MDN 
  (triangle_ABC : InscribedTriangle O A B C)
  (tangent_points : TangentAt O B ∩ TangentAt O C ∩ TangentAt O A = {M, N})
  (altitude_AD : Altitude A D B C) :
  Bisects (Angle A D M) (Angle A D N) :=
sorry

end AD_bisects_angle_MDN_l507_507771


namespace find_removed_number_l507_507557

theorem find_removed_number (numbers : List ℕ) (avg_remain : ℝ) (h_list : numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13]) (h_avg : avg_remain = 7.5) :
  ∃ x, x ∈ numbers ∧ 
       (let numbers_removed := numbers.erase x in 
        (numbers.sum - x) / (numbers.length - 1) = avg_remain) := 
by
  sorry

end find_removed_number_l507_507557


namespace arithmetic_mean_of_fractions_l507_507908

theorem arithmetic_mean_of_fractions : 
  (3 : ℚ) / 8 + (5 : ℚ) / 9 = 2 * (67 : ℚ) / 144 :=
by {
  rw [(show (3 : ℚ) / 8 + (5 : ℚ) / 9 = (27 : ℚ) / 72 + (40 : ℚ) / 72, by sorry),
      (show (27 : ℚ) / 72 + (40 : ℚ) / 72 = (67 : ℚ) / 72, by sorry),
      (show (2 : ℚ) * (67 : ℚ) / 144 = (67 : ℚ) / 72, by sorry)]
}

end arithmetic_mean_of_fractions_l507_507908


namespace john_spent_amount_l507_507373

-- Definitions based on the conditions in the problem.
def hours_played : ℕ := 3
def cost_per_6_minutes : ℚ := 0.50
def minutes_per_6_minutes_interval : ℕ := 6
def total_minutes_played (hours : ℕ) : ℕ := hours * 60

-- The theorem statement.
theorem john_spent_amount 
  (h : hours_played = 3) 
  (c : cost_per_6_minutes = 0.50)
  (m_interval : minutes_per_6_minutes_interval = 6) :
  let intervals := (total_minutes_played hours_played) / minutes_per_6_minutes_interval in
  intervals * cost_per_6_minutes = 15 := 
by
  sorry

end john_spent_amount_l507_507373


namespace arithmetic_mean_of_fractions_l507_507928

theorem arithmetic_mean_of_fractions (a b : ℚ) (h1 : a = 3 / 8) (h2 : b = 5 / 9) :
  (a + b) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_fractions_l507_507928


namespace number_of_eggs_left_l507_507638

theorem number_of_eggs_left (initial_eggs : ℕ) (eggs_eaten_morning : ℕ) (eggs_eaten_afternoon : ℕ) (eggs_left : ℕ) :
    initial_eggs = 20 → eggs_eaten_morning = 4 → eggs_eaten_afternoon = 3 → eggs_left = initial_eggs - (eggs_eaten_morning + eggs_eaten_afternoon) → eggs_left = 13 :=
by
  intros h_initial h_morning h_afternoon h_calc
  rw [h_initial, h_morning, h_afternoon] at h_calc
  norm_num at h_calc
  exact h_calc

end number_of_eggs_left_l507_507638


namespace minimum_layovers_l507_507767

theorem minimum_layovers (n m : ℕ) (h1 : n = 20) (h2 : m = 4) :
  ∃ k : ℕ, (k = 2) ∧ (∀ a b : ℕ, (a ≠ b) → reachable_within_k_layovers n m a b k) := 
sorry

-- Additional definitions that will be needed
def reachable_within_k_layovers (n m a b k : ℕ) : Prop := sorry

end minimum_layovers_l507_507767


namespace Thabo_books_l507_507452

theorem Thabo_books :
  ∃ (H : ℕ), ∃ (P : ℕ), ∃ (F : ℕ), 
  (H + P + F = 220) ∧ 
  (P = H + 20) ∧ 
  (F = 2 * P) ∧ 
  (H = 40) :=
by
  -- Here will be the formal proof, which is not required for this task.
  sorry

end Thabo_books_l507_507452


namespace total_digits_first_2500_even_integers_l507_507117

theorem total_digits_first_2500_even_integers :
  let even_nums := List.range' 2 5000 (λ n, 2*n)  -- List of the first 2500 even integers
  let one_digit_nums := even_nums.filter (λ n, n < 10)
  let two_digit_nums := even_nums.filter (λ n, 10 ≤ n ∧ n < 100)
  let three_digit_nums := even_nums.filter (λ n, 100 ≤ n ∧ n < 1000)
  let four_digit_nums := even_nums.filter (λ n, 1000 ≤ n ∧ n ≤ 5000)
  let sum_digits := one_digit_nums.length * 1 + two_digit_nums.length * 2 + three_digit_nums.length * 3 + four_digit_nums.length * 4
  in sum_digits = 9448 := by sorry

end total_digits_first_2500_even_integers_l507_507117


namespace smallest_integer_with_eight_factors_l507_507554

theorem smallest_integer_with_eight_factors :
  ∃ n : ℕ, (∀ m : ℕ, (∃ k : ℕ, m has_factors k ∧ k = 8) → m ≥ n) ∧ n = 24 :=
by
  sorry

end smallest_integer_with_eight_factors_l507_507554


namespace sum_of_a_values_for_one_solution_l507_507225

theorem sum_of_a_values_for_one_solution :
  (∀ a : ℝ, (∀ x : ℝ, 3 * x ^ 2 + (a + 6) * x + 7 = 0 → (a + 6) ^ 2 - 4 * 3 * 7 = 0) → 
  (a = -6 + 2 * Real.sqrt 21 ∨ a = -6 - 2 * Real.sqrt 21)) → 
  ((-6 + 2 * Real.sqrt 21) + (-6 - 2 * Real.sqrt 21) = -12) :=
by
  intro h
  simp at h
  sorry

end sum_of_a_values_for_one_solution_l507_507225


namespace irrational_roots_of_odd_coeff_quad_l507_507039

theorem irrational_roots_of_odd_coeff_quad (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) :
  ¬ ∃ r : ℚ, a * r^2 + b * r + c = 0 := 
sorry

end irrational_roots_of_odd_coeff_quad_l507_507039


namespace solve_cubic_eq_l507_507244

theorem solve_cubic_eq (z : ℂ) : (z^3 + 27 = 0) ↔ (z = -3 ∨ z = (3 / 2) + (3 * complex.I * real.sqrt 3 / 2) ∨ z = (3 / 2) - (3 * complex.I * real.sqrt 3 / 2)) :=
by
  sorry

end solve_cubic_eq_l507_507244


namespace f_2019_value_l507_507380

noncomputable def B : Set ℚ := {q : ℚ | q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1}

noncomputable def g (x : ℚ) (h : x ∈ B) : ℚ :=
  1 - (2 / x)

noncomputable def f (x : ℚ) (h : x ∈ B) : ℝ :=
  sorry

theorem f_2019_value (h2019 : 2019 ∈ B) :
  f 2019 h2019 = Real.log ((2019 - 0.5) ^ 2 / 2018.5) :=
sorry

end f_2019_value_l507_507380


namespace total_bills_inserted_l507_507163

theorem total_bills_inserted (x y : ℕ) (h1 : x = 175) (h2 : x + 5 * y = 300) : 
  x + y = 200 :=
by {
  -- Since we focus strictly on the statement per instruction, the proof is omitted
  sorry 
}

end total_bills_inserted_l507_507163


namespace largest_value_among_given_numbers_l507_507193

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem largest_value_among_given_numbers :
  let a := Real.log (Real.sqrt 2)
  let b := 1 / Real.exp 1
  let c := (Real.log Real.pi) / Real.pi
  let d := (Real.sqrt 10 * Real.log 10) / 20 
  b > a ∧ b > c ∧ b > d :=
by
  let a := Real.log (Real.sqrt 2)
  let b := 1 / Real.exp 1
  let c := (Real.log Real.pi) / Real.pi
  let d := (Real.sqrt 10 * Real.log 10) / 20
  -- Add the necessary steps to show that b is the largest value
  sorry

end largest_value_among_given_numbers_l507_507193


namespace arithmetic_mean_of_fractions_l507_507898

theorem arithmetic_mean_of_fractions :
  (3/8 + 5/9) / 2 = 67 / 144 :=
by
  sorry

end arithmetic_mean_of_fractions_l507_507898


namespace total_number_of_digits_l507_507122

-- Definitions based on identified conditions
def first2500EvenIntegers := {n : ℕ | n % 2 = 0 ∧ 1 ≤ n ∧ n ≤ 5000}

-- Theorem statement based on the equivalent proof problem
theorem total_number_of_digits : 
  (first2500EvenIntegers.count_digits = 9448) :=
sorry

end total_number_of_digits_l507_507122


namespace smallest_integer_with_eight_factors_l507_507525

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, n = 24 ∧
  (∀ d : ℕ, d ∣ n → d > 0) ∧
  ((∃ p : ℕ, prime p ∧ n = p^7) ∨
   (∃ p q : ℕ, prime p ∧ prime q ∧ n = p^3 * q) ∨
   (∃ p q r : ℕ, prime p ∧ prime q ∧ prime r ∧ n = p * q * r)) ∧
  (∀ m : ℕ, (∀ d : ℕ, d ∣ m → d > 0) → 
           ((∃ p : ℕ, prime p ∧ m = p^7 ∨ m = p^3 * q ∨ m = p * q * r) → 
            m ≥ n)) := by
  sorry

end smallest_integer_with_eight_factors_l507_507525


namespace parabola_sum_of_distances_l507_507315

variable (n : ℕ) (x : ℕ → ℝ)
variable (C : ∀ i, x i ^ 2 / 4 = C)
variable (F : ℝ)
variable (y : ∀ i, y i ^ 2 = 4 (x i))
variable (sum_x_eq_10 : ∑ i in finset.range n, x i = 10)

theorem parabola_sum_of_distances (hx : ∀ i, y i ^ 2 = 4 * (x i))
  : ∑ i in finset.range n, (abs (x i + 1)) = n + 10 :=
by {
  sorry
}

end parabola_sum_of_distances_l507_507315


namespace situps_together_l507_507308

theorem situps_together (hani_rate diana_rate : ℕ) (diana_situps diana_time hani_situps total_situps : ℕ)
  (h1 : hani_rate = diana_rate + 3)
  (h2 : diana_rate = 4)
  (h3 : diana_situps = 40)
  (h4 : diana_time = diana_situps / diana_rate)
  (h5 : hani_situps = hani_rate * diana_time)
  (h6 : total_situps = diana_situps + hani_situps) : 
  total_situps = 110 :=
sorry

end situps_together_l507_507308


namespace remove_one_to_get_average_of_75_l507_507560

theorem remove_one_to_get_average_of_75 : 
  ∃ l : List ℕ, l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] ∧ 
  (∃ m : ℕ, List.erase l m = ([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] : List ℕ) ∧ 
  (12 : ℕ) = List.length (List.erase l m) ∧
  7.5 = ((List.sum (List.erase l m) : ℚ) / 12)) :=
sorry

end remove_one_to_get_average_of_75_l507_507560


namespace evaluate_expression_l507_507229

theorem evaluate_expression : 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 221 :=
by
  sorry

end evaluate_expression_l507_507229


namespace smallest_positive_integer_with_eight_factors_l507_507520

theorem smallest_positive_integer_with_eight_factors :
  ∃ n : ℕ, (∀ m : ℕ, (m < n ∧ (∀ d : ℕ, d | m → d = 1 ∨ d = m) → (∃ a b : ℕ, distinct_factors_count m a b ∧ a = 8)) → n = 24) :=
by
  sorry

def distinct_factors_count (n : ℕ) (a b : ℕ) : Prop :=
  ∃ (p q : ℕ), prime p ∧ prime q ∧ n = p^a * q^b ∧ (a + 1) * (b + 1) = 8

end smallest_positive_integer_with_eight_factors_l507_507520


namespace total_boys_in_camp_l507_507766

noncomputable def numberOfBoys (T : ℕ) : Prop :=
  let boysFromSchoolA := 0.20 * T
  let nonScientificBoysFromSchoolA := 0.70 * boysFromSchoolA
  nonScientificBoysFromSchoolA = 77 ∧ T = 550

theorem total_boys_in_camp : ∃ T : ℕ, numberOfBoys T :=
by
  use 550
  have h1 : 0.20 * (550:ℝ) = 110 := by sorry
  have h2 : 0.70 * 110 = 77 := by sorry
  show numberOfBoys 550, by
    split
    · exact h2
    · refl

end total_boys_in_camp_l507_507766


namespace arithmetic_mean_eq_l507_507901

theorem arithmetic_mean_eq :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = (67 : ℚ) / 144 :=
by
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  have h₁ : a = 3 / 8 := by rfl
  have h₂ : b = 5 / 9 := by rfl
  sorry

end arithmetic_mean_eq_l507_507901


namespace trigonometric_identity_l507_507709

noncomputable def trigonometric_expression (x : ℝ) : ℝ :=
  sin x ^ 2 + 3 * sin x * cos x - 1

theorem trigonometric_identity (x : ℝ) (h : tan x = -1/2) : trigonometric_expression x = -2 :=
by
  sorry

end trigonometric_identity_l507_507709


namespace solve_equation_l507_507438

def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4
def g (x : ℝ) : ℝ := x^4 + 2*x^3 + x^2 + 11*x + 11
def h (x : ℝ) : ℝ := x + 1

theorem solve_equation :
  ∀ x : ℝ, f⁻¹(g(x)) = h(x) → (x = 1 ∨ x = (-3 + real.sqrt 5) / 2 ∨ x = (-3 - real.sqrt 5) / 2) :=
by
  sorry

end solve_equation_l507_507438


namespace triangle_BD_length_l507_507332

/-- In a triangle ABC with AC = BC = 10 and AB = 8, point D lies on the line AB
such that B is between A and D and CD = 12. Prove that BD = 2 * sqrt 15. -/
theorem triangle_BD_length (A B C D : Type)
  [euclidean_geometry A B C D] -- Assume it's in a Euclidean space
  (hAC : dist A C = 10)
  (hBC : dist B C = 10)
  (hAB : dist A B = 8)
  (hCD : dist C D = 12)
  (hB_on_AB : B ∈ segment A D) :
  dist B D = 2 * Real.sqrt 15 :=
sorry

end triangle_BD_length_l507_507332


namespace ratio_difference_l507_507988

theorem ratio_difference (x : ℕ) (h_largest : 7 * x = 70) : 70 - 3 * x = 40 := by
  sorry

end ratio_difference_l507_507988


namespace product_mn_l507_507463

-- Λet θ1 be the angle L1 makes with the positive x-axis.
-- Λet θ2 be the angle L2 makes with the positive x-axis.
-- Given that θ1 = 3 * θ2 and m = 6 * n.
-- Using the tangent triple angle formula: tan(3θ) = (3 * tan(θ) - tan^3(θ)) / (1 - 3 * tan^2(θ))
-- We need to prove mn = 9/17.

noncomputable def mn_product_condition (θ1 θ2 : ℝ) (m n : ℝ) : Prop :=
θ1 = 3 * θ2 ∧ m = 6 * n ∧ m = Real.tan θ1 ∧ n = Real.tan θ2

theorem product_mn (θ1 θ2 : ℝ) (m n : ℝ) (h : mn_product_condition θ1 θ2 m n) :
  m * n = 9 / 17 :=
sorry

end product_mn_l507_507463


namespace sum_of_digits_of_n_l507_507492

theorem sum_of_digits_of_n :
  ∃ n : ℕ, (log 2 (log 16 n) = log 4 (log 4 n)) ∧ (sum_digits n = 13) :=
by 
  sorry

-- Auxiliary function to calculate the sum of digits, assume it's implemented correctly
def sum_digits (n : ℕ) : ℕ := 
  sorry

end sum_of_digits_of_n_l507_507492


namespace age_problem_l507_507502

theorem age_problem (g v k x : ℕ) (h_g_lt_6 : g < 6) 
    (h_v_eq_xg : v = x * g) (h_k_eq_xv : k = x * v)
    (h_v_plus_k : v + k = 112) : 
    g = 2 ∧ v = 14 ∧ k = 98 :=
begin
    sorry
end

end age_problem_l507_507502


namespace sum_of_diagonals_l507_507875

theorem sum_of_diagonals (n : ℕ) (h : n ≥ 4)
  (M : matrix (fin n) (fin n) ℝ)
  (rows_are_arith_seq : ∀ i : fin n, ∃ d : ℝ, ∀ j : fin n, M i j = M i 0 + (j.val : ℝ) * d)
  (cols_are_geom_seq : ∃ q : ℝ, ∀ j : fin n, ∀ i : fin n, M i j = M 0 j * q^(i.val : ℝ))
  (common_ratio_cols : ∀ j k : fin n, j ≠ k → cols_are_geom_seq.some = cols_are_geom_seq.some)
  (a_24 : M ⟨1, by simp⟩ ⟨3, by simp⟩ = 1)
  (a_42 : M ⟨3, by simp⟩ ⟨1, by simp⟩ = 1 / 8)
  (a_43 : M ⟨3, by simp⟩ ⟨2, by simp⟩ = 3 / 16) :
  (finset.univ.filter (λ x : fin n, x.val < n)).sum (λ k, M k k) = 2 - 1 / 2^(n-1) - n / 2^n :=
by {
  have h_geom := cols_are_geom_seq.some_spec,
  have rows_geom_eq := cols_are_geom_seq_some_spec,
  sorry
}

end sum_of_diagonals_l507_507875


namespace round_to_nearest_hundredth_l507_507429

-- Definition of the repeating decimal
def repeating_decimal : ℚ := 67 + 673 / 999

-- Theorem stating that rounding this repeating decimal to the nearest hundredth gives 67.67
theorem round_to_nearest_hundredth : repeating_decimal.round(2) = 67.67 :=
by
  sorry

end round_to_nearest_hundredth_l507_507429


namespace arithmetic_mean_of_38_and_59_is_67_over_144_l507_507939

theorem arithmetic_mean_of_38_and_59_is_67_over_144 :
  (3 / 8 + 5 / 9) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_38_and_59_is_67_over_144_l507_507939


namespace purely_imaginary_solution_l507_507273

noncomputable def complex_number_is_purely_imaginary (m : ℝ) : Prop :=
  (m^2 - 2 * m - 3 = 0) ∧ (m + 1 ≠ 0)

theorem purely_imaginary_solution (m : ℝ) (h : complex_number_is_purely_imaginary m) : m = 3 := by
  sorry

end purely_imaginary_solution_l507_507273


namespace louie_monthly_payment_l507_507411

noncomputable def monthly_payment (P r : ℝ) (n : ℕ) : ℝ :=
  let A := P * (1 + r) ^ n
  A / n

theorem louie_monthly_payment :
  monthly_payment 5000 0.15 6 ≈ 1676 :=
by sorry

#check louie_monthly_payment

end louie_monthly_payment_l507_507411


namespace solve_problem_l507_507655

-- Define the distinct digits A, B, C, D
def is_digit (n : ℕ) := n ≥ 0 ∧ n ≤ 9

-- Define the main condition to satisfy
def valid_tuple (A B C D : ℕ) : Prop :=
  is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧
  (A ≠ 0) ∧ (B ≠ 0) ∧ (C ≠ 0) ∧ (D ≠ 0) ∧
  (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧
  (B ≠ C) ∧ (B ≠ D) ∧ (C ≠ D) ∧
  (100 * A + 10 * B + C) * D = (100 * B + 10 * A + D) * C

-- Prove the problem statement
theorem solve_problem :
  ∃ A B C D : ℕ, valid_tuple A B C D ∧
  (A, B, C, D) ∈ {(2,1,7,4), (1,2,4,7), (8,1,9,2), (1,8,2,9), (7,2,8,3), (2,7,3,8), (6,3,7,4), (3,6,4,7)} :=
sorry

end solve_problem_l507_507655


namespace carol_optimal_strategy_l507_507191

-- Definitions of the random variables
def uniform_A (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1
def uniform_B (b : ℝ) : Prop := 0.25 ≤ b ∧ b ≤ 0.75
def winning_condition (a b c : ℝ) : Prop := (a < c ∧ c < b) ∨ (b < c ∧ c < a)

-- Carol's optimal strategy stated as a theorem
theorem carol_optimal_strategy : ∀ (a b c : ℝ), 
  uniform_A a → uniform_B b → (c = 7 / 12) → 
  winning_condition a b c → 
  ∀ (c' : ℝ), uniform_A c' → c' ≠ c → ¬(winning_condition a b c') :=
by
  sorry

end carol_optimal_strategy_l507_507191


namespace smallest_integer_with_eight_factors_l507_507552

theorem smallest_integer_with_eight_factors :
  ∃ n : ℕ, (∀ m : ℕ, (∃ k : ℕ, m has_factors k ∧ k = 8) → m ≥ n) ∧ n = 24 :=
by
  sorry

end smallest_integer_with_eight_factors_l507_507552


namespace chord_bisected_by_point_eq_line_l507_507586

theorem chord_bisected_by_point_eq_line {P : ℝ × ℝ} (hP : P = (2, -3)) 
 (h_eq : ∀ (A B : ℝ × ℝ), A ≠ B → let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in
   M = P → ∃ m b : ℝ, ∀ (x y : ℝ), (x - y - 5 = 0)) 
  : ∃ m b : ℝ, ∀ (x y : ℝ), (x - y - 5 = 0) :=
by
  sorry

end chord_bisected_by_point_eq_line_l507_507586


namespace matrix_equation_solution_l507_507383

theorem matrix_equation_solution :
  ∃ r s : ℝ, 
  let B := !![0, 1; 4, -2] in
  let I := !![1, 0; 0, 1] in
  let B4 := B^4 in
  let rB := r • B in
  let sI := s • I in
  B4 = rB + sI ∧ r = -12 ∧ s = 16 :=
sorry

end matrix_equation_solution_l507_507383


namespace compare_negatives_l507_507215

theorem compare_negatives : (- (3 : ℝ) / 5) > (- (5 : ℝ) / 7) :=
by
  sorry

end compare_negatives_l507_507215


namespace collinear_vectors_l507_507304

theorem collinear_vectors (x y : ℝ) (λ : ℝ) 
  (h1 : 2 * x = λ * 1) 
  (h2 : 1 = λ * (-2 * y)) 
  (h3 : 3 = λ * 9) : 
  x + y = -4 / 3 := 
by 
  sorry

end collinear_vectors_l507_507304


namespace sqrt_inequality_l507_507030

theorem sqrt_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b + b * c + c * a = 1) :
  sqrt (a + 1 / a) + sqrt (b + 1 / b) + sqrt (c + 1 / c) ≥ 2 * (sqrt a + sqrt b + sqrt c) :=
sorry

end sqrt_inequality_l507_507030


namespace perpendicular_distance_correct_l507_507677

open Real EuclideanGeometry

noncomputable def perpendicular_distance (D A B C : Point) : Real :=
  let AB := dist A B
  let AC := dist A C
  let BC := dist B C
  let s := (AB + AC + BC) / 2
  let Δ := sqrt (s * (s - AB) * (s - AC) * (s - BC))
  let V := (1 / 3) * 5 * 6 * 4
  (3 * V) / Δ

theorem perpendicular_distance_correct : 
  perpendicular_distance (Point.mk 0 0 0) (Point.mk 5 0 0) (Point.mk 0 6 0) (Point.mk 0 0 4) = 4.2 :=
by
  sorry

end perpendicular_distance_correct_l507_507677


namespace redistribute_six_chests_evenly_l507_507061

def six_chests_even (C : Fin 6 → Nat) : Prop :=
  ∀ (i j : Fin 6), i ≠ j → (C i + C j) % 2 = 0

def six_chests_div_by_three (C : Fin 6 → Nat) : Prop :=
  ∀ (i j k : Fin 6), i ≠ j → j ≠ k → k ≠ i → (C i + C j + C k) % 3 = 0

def six_chests_div_by_four_or_five (C : Fin 6 → Nat) : Prop :=
  ∀ (n : Fin (n.choose 4 + n.choose 5)), (i : Fin 5) → C i % 2 = 0

theorem redistribute_six_chests_evenly (C : Fin 6 → Nat) :
  six_chests_even C ∧ six_chests_div_by_three C ∧ six_chests_div_by_four_or_five C → 
  (∑ i, C i) % 6 = 0 := 
sorry

end redistribute_six_chests_evenly_l507_507061


namespace place_mat_length_l507_507181

-- Define the radius of the round table
def radius : ℝ := 5

-- Define the width of each place mat
def width : ℝ := 1

-- Define and state the objective value x
def x : ℝ := (3 * Real.sqrt 11 + 1) / 2

-- State the theorem saying the length of each place mat is x given the conditions
theorem place_mat_length:
    ∃ x : ℝ, 
    let r := radius in 
    let w := width in 
    (∀ (i : Fin 8),
     let θ := (2 * Real.pi / 8 : ℝ) in 
     let mat_start := r * Complex.exp(Complex.I * (i * θ)) in
     let mat_end := r * Complex.exp(Complex.I * ((i + 1) % 8 * θ)) in 
     let chord_length := Complex.abs (mat_end - mat_start) in 
     x = chord_length ∧ 
     (r^2 = (w/2)^2 + (x - w/2)^2)) → 
    x = (3 * Real.sqrt 11 + 1) / 2 :=
sorry

end place_mat_length_l507_507181


namespace carrots_picked_next_day_l507_507645

-- Definitions based on conditions
def initial_carrots : Nat := 48
def carrots_thrown_away : Nat := 45
def total_carrots_next_day : Nat := 45

-- The proof problem statement
theorem carrots_picked_next_day : 
  (initial_carrots - carrots_thrown_away + x = total_carrots_next_day) → (x = 42) :=
by 
  sorry

end carrots_picked_next_day_l507_507645


namespace roots_of_unity_l507_507409

open Complex

theorem roots_of_unity (p q r s t m : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0)
  (h1 : p * m^4 + q * m^3 + r * m^2 + s * m + t = 0)
  (h2 : q * m^4 + r * m^3 + s * m^2 + t * m + p = 0) :
  ∃ k ∈ {0, 1, 2, 3, 4}, m = Complex.exp (2 * Real.pi * Complex.I * k / 5) :=
by
  sorry

end roots_of_unity_l507_507409


namespace intersection_of_A_and_B_l507_507299

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := 
by
  sorry

end intersection_of_A_and_B_l507_507299


namespace smallest_integer_with_eight_factors_l507_507548

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, 
  ∀ m : ℕ, (∀ p : ℕ, ∃ k : ℕ, m = p^k → (k + 1) * (p + 1) = 8) → (n ≤ m) ∧ 
  (∀ d : ℕ, d ∣ n → d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 24) :=
sorry

end smallest_integer_with_eight_factors_l507_507548


namespace part1_part2_l507_507258

/-- Given a triangle ABC with sides opposite to angles A, B, C being a, b, c respectively,
and a sin A sin B + b cos^2 A = 5/3 a,
prove that (1) b / a = 5/3. -/
theorem part1 (a b : ℝ) (A B : ℝ) (h₁ : a * (Real.sin A) * (Real.sin B) + b * (Real.cos A) ^ 2 = (5 / 3) * a) :
  b / a = 5 / 3 :=
sorry

/-- Given the previous result b / a = 5/3 and the condition c^2 = a^2 + 8/5 b^2,
prove that (2) angle C = 2π / 3. -/
theorem part2 (a b c : ℝ) (A B C : ℝ)
  (h₁ : a * (Real.sin A) * (Real.sin B) + b * (Real.cos A) ^ 2 = (5 / 3) * a)
  (h₂ : c^2 = a^2 + (8 / 5) * b^2)
  (h₃ : b / a = 5 / 3) :
  C = 2 * Real.pi / 3 :=
sorry

end part1_part2_l507_507258


namespace probability_x_plus_y_less_than_2_5_in_square_l507_507602

theorem probability_x_plus_y_less_than_2_5_in_square :
  (let area_of_triangle := (1/2) * 2.5 * 2.5 in
   let area_of_square := 3 * 3 in
   area_of_triangle / area_of_square = 125 / 360) :=
by
  sorry

end probability_x_plus_y_less_than_2_5_in_square_l507_507602


namespace randy_gave_sally_l507_507823

-- Define the given conditions
def initial_amount_randy : ℕ := 3000
def smith_contribution : ℕ := 200
def amount_kept_by_randy : ℕ := 2000

-- The total amount Randy had after Smith's contribution
def total_amount_randy : ℕ := initial_amount_randy + smith_contribution

-- The amount of money Randy gave to Sally
def amount_given_to_sally : ℕ := total_amount_randy - amount_kept_by_randy

-- The theorem statement: Given the conditions, prove that Randy gave Sally $1,200
theorem randy_gave_sally : amount_given_to_sally = 1200 :=
by
  sorry

end randy_gave_sally_l507_507823


namespace focus_of_parabola_with_vertex_and_directrix_l507_507302

theorem focus_of_parabola_with_vertex_and_directrix
  (vertex : ℝ × ℝ) (directrix : ℝ) 
  (hv : vertex = (2, 0))
  (hd : directrix = -1) :
  let focus := (5, 0) in
  focus = (5, 0) :=
by
  sorry

end focus_of_parabola_with_vertex_and_directrix_l507_507302


namespace area_AOB_l507_507354

open Real

namespace Geometry

-- Define the parametric equation of curve C₁
def curveC1_parametric (t: ℝ) : ℝ × ℝ := (t^2, 2 * t)

-- Define the polar equation of curve C₂
def curveC2_polar (theta: ℝ) : ℝ := 5 * cos theta

-- Define the Cartesian equation of curve C₂
def curveC2_cartesian (x y: ℝ) : Prop := x^2 + y^2 = 5 * x

-- The condition for point A being in the first quadrant intersection of curves C₁ and C₂
def pointA_in_first_quadrant (x y: ℝ) : Prop := (y^2 = 4 * x ∧ x^2 + y^2 = 5 * x ∧ x > 0 ∧ y > 0)

-- The condition for point B being on C₁ and OA⊥OB
def pointB_condition (xA yA xB yB: ℝ) : Prop :=
  (yB^2 = 4 * xB ∧ yA * (xB - xA) = -xA * (yB - yA))

-- Define the area of triangle using determinant formula
def area_of_triangle (Ax Ay Bx By: ℝ) : ℝ :=
  (1/2) * abs (Ax * By - Ay * Bx)

-- Main theorem statement
theorem area_AOB (tA tB: ℝ) :
  let A := curveC1_parametric tA,
  let B := curveC1_parametric tB,
  let xA := A.1, let yA := A.2,
  let xB := B.1, let yB := B.2
in pointA_in_first_quadrant xA yA ∧ pointB_condition xA yA xB yB →
  area_of_triangle xA yA xB yB = 20 :=
by {
  sorry
}

end Geometry

end area_AOB_l507_507354


namespace isosceles_triangle_smallest_angle_l507_507626

-- Given conditions:
-- 1. The triangle is isosceles
-- 2. One angle is 40% larger than the measure of a right angle

theorem isosceles_triangle_smallest_angle :
  ∃ (A B C : ℝ), 
  A + B + C = 180 ∧ 
  (A = B ∨ A = C ∨ B = C) ∧ 
  (∃ (large_angle : ℝ), large_angle = 90 + 0.4 * 90 ∧ (A = large_angle ∨ B = large_angle ∨ C = large_angle)) →
  (A = 27 ∨ B = 27 ∨ C = 27) := sorry

end isosceles_triangle_smallest_angle_l507_507626


namespace min_omega_value_l507_507832

noncomputable def min_omega (omega : ℝ) : ℝ :=
by
  let f := λ x : ℝ, 2 * sin (omega * x + pi / 3)
  let f_left := λ x : ℝ, f (x + pi / 3)
  let f_right := λ x : ℝ, f (x - pi / 3)
  let axes_symmetry_coincide := (∀ x : ℝ, f_left x = f_right x)
  exact if axes_symmetry_coincide then omega else 0

theorem min_omega_value (h ω_pos : ω > 0) : min_omega ω = 3 / 2 := 
by
  sorry

end min_omega_value_l507_507832


namespace diagonal_lengths_l507_507247

theorem diagonal_lengths (x : ℕ) : 
  (4 < x ∧ x < 20) → (set.Ico 5 20).card = 15 :=
begin
  intros h,
  sorry
end

end diagonal_lengths_l507_507247


namespace circumcircle_triangle_ADF_tangent_line_AC_l507_507822

variables {A B C D F: Type}
variables [is_circumscribed_quadrilateral A B C D]
variables [on_extension BD D F]
variables [parallel AF BC]

theorem circumcircle_triangle_ADF_tangent_line_AC :
  tangent (circumcircle_triangle A D F) (line AC) :=
sorry

end circumcircle_triangle_ADF_tangent_line_AC_l507_507822


namespace vector_at_t_neg1_l507_507171

open Matrix

theorem vector_at_t_neg1 (a b : Matrix (Fin 3) (Fin 1) ℝ) (t : ℝ)
  (h0 : a = colVec 2 6 16)
  (h1 : b = colVec 1 1 8)
  (ht : t = -1) :
  a + t * (b - a) = colVec 3 11 24 :=
by
  -- Proof omitted
  sorry

def colVec (x y z : ℝ) : Matrix (Fin 3) (Fin 1) ℝ :=
  ![[x], [y], [z]]

end vector_at_t_neg1_l507_507171


namespace intersection_A_complement_B_eq_interval_l507_507300

-- We define universal set U as ℝ
def U := Set ℝ

-- Definitions provided in the problem
def A : Set ℝ := { x | x > 1 }
def B : Set ℝ := { y | y >= 2 }

-- Complement of B in U
def C_U_B : Set ℝ := { y | y < 2 }

-- Now we state the theorem
theorem intersection_A_complement_B_eq_interval :
  A ∩ C_U_B = { x | 1 < x ∧ x < 2 } :=
by 
  sorry

end intersection_A_complement_B_eq_interval_l507_507300


namespace math_problem_l507_507210

theorem math_problem :
  (-1) ^ 2023 - Real.sqrt 9 + abs (1 - Real.sqrt 2) - Real.cbrt (-8) = Real.sqrt 2 - 3 :=
by 
  sorry

end math_problem_l507_507210


namespace kaleb_non_working_games_l507_507375

theorem kaleb_non_working_games (total_games working_game_price earning : ℕ) (h1 : total_games = 10) (h2 : working_game_price = 6) (h3 : earning = 12) :
  total_games - (earning / working_game_price) = 8 :=
by
  sorry

end kaleb_non_working_games_l507_507375


namespace inequality_sqrt_l507_507032

theorem inequality_sqrt (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
by
  sorry

end inequality_sqrt_l507_507032


namespace weight_of_b_l507_507456

variable {A B C : ℤ}

def condition1 (A B C : ℤ) : Prop := (A + B + C) / 3 = 45
def condition2 (A B : ℤ) : Prop := (A + B) / 2 = 42
def condition3 (B C : ℤ) : Prop := (B + C) / 2 = 43

theorem weight_of_b (A B C : ℤ) 
  (h1 : condition1 A B C) 
  (h2 : condition2 A B) 
  (h3 : condition3 B C) : 
  B = 35 := 
by
  sorry

end weight_of_b_l507_507456


namespace find_z_l507_507574

noncomputable def k : ℝ := 192

theorem find_z (z x : ℝ) (h₁ : 3 * z = k / x^3) (hx2 : x = 2) (hz8 : z = 8) :
    (z = 1 → x = 4) :=
by
  have h_k : k = 192 := sorry
  have h2 : z = 1 := sorry
  exact h2

end find_z_l507_507574


namespace remainder_eq_four_l507_507956

theorem remainder_eq_four {x : ℤ} (h : x % 61 = 24) : x % 5 = 4 :=
sorry

end remainder_eq_four_l507_507956


namespace cos_product_inequality_l507_507821

theorem cos_product_inequality : (1 / 8 : ℝ) < (Real.cos (20 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) * Real.cos (70 * Real.pi / 180)) ∧
    (Real.cos (20 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) * Real.cos (70 * Real.pi / 180)) < (1 / 4 : ℝ) :=
by
  sorry

end cos_product_inequality_l507_507821


namespace combined_average_correct_l507_507640

def springfield_percentage : ℕ → ℝ 
| 1990 := 12
| 2000 := 18
| 2010 := 25
| 2020 := 40
| _ := 0

def shelbyville_percentage : ℕ → ℝ 
| 1990 := 10
| 2000 := 15
| 2010 := 23
| 2020 := 35
| _ := 0

def combined_average (year : ℕ) : ℝ :=
  (springfield_percentage year + shelbyville_percentage year) / 2

theorem combined_average_correct :
  combined_average 1990 = 11 ∧
  combined_average 2000 = 16.5 ∧
  combined_average 2010 = 24 ∧
  combined_average 2020 = 37.5 :=
by
  sorry

end combined_average_correct_l507_507640


namespace batteries_manufactured_l507_507593

theorem batteries_manufactured (gather_time create_time : Nat) (robots : Nat) (hours : Nat) (total_batteries : Nat) :
  gather_time = 6 →
  create_time = 9 →
  robots = 10 →
  hours = 5 →
  total_batteries = (hours * 60 / (gather_time + create_time)) * robots →
  total_batteries = 200 :=
by
  intros h_gather h_create h_robots h_hours h_batteries
  simp [h_gather, h_create, h_robots, h_hours] at h_batteries
  exact h_batteries

end batteries_manufactured_l507_507593


namespace perimeter_of_square_l507_507611

-- Defining the square with area
structure Square where
  side_length : ℝ
  area : ℝ

-- Defining a constant square with given area 625
def givenSquare : Square := 
  { side_length := 25, -- will square root the area of 625
    area := 625 }

-- Defining the function to calculate the perimeter of the square
noncomputable def perimeter (s : Square) : ℝ :=
  4 * s.side_length

-- The theorem stating that the perimeter of the given square with area 625 is 100
theorem perimeter_of_square : perimeter givenSquare = 100 := 
sorry

end perimeter_of_square_l507_507611


namespace part_a_part_b_part_c_l507_507889

-- Define the grid and the valid moves based on distance condition
structure Board (m n : ℕ) :=
(valid_move : Π {x1 y1 x2 y2 : ℕ}, (x1 < m) → (y1 < n) → (x2 < m) → (y2 < n) →
              (x1 - x2) ^ 2 + (y1 - y2) ^ 2 = r → Prop)

-- Define the task impossibility conditions
def task_impossible_for_even_or_third (r : ℕ) [Fact (r % 2 = 0 ∨ r % 3 = 0)] : Prop :=
  ∀ (board : Board 20 12), ¬ ∃ seq : list (ℕ × ℕ),
    (seq.head = (0, 0) ∧ seq.last = (19, 0)) ∧ (∀ xy in seq, board.valid_move xy)

-- Define the task possibility condition for r = 73
def task_possible_for_73 (board : Board 20 12) [Fact (r = 73)] : Prop :=
  ∃ seq : list (ℕ × ℕ),
    (seq.head = (0, 0) ∧ seq.last == (19, 0)) ∧ (∀ xy in seq, board.valid_move xy)

-- Define the task impossibility condition for r = 97
def task_impossible_for_97 (board : Board 20 12) [Fact (r = 97)] : Prop :=
  ¬ ∃ seq : list (ℕ × ℕ),
    (seq.head = (0, 0) ∧ seq.last = (19, 0)) ∧ (∀ xy in seq, board.valid_move xy)

---

-- The conditions from the problem statement
parameter (r : ℕ)
parameter (board : Board 20 12)

-- The Lean statements for each part of the problem
theorem part_a : Fact (r % 2 = 0 ∨ r % 3 = 0) → task_impossible_for_even_or_third r :=
  by sorry

theorem part_b : Fact (r = 73) → task_possible_for_73 board :=
  by sorry

theorem part_c : Fact (r = 97) → task_impossible_for_97 board :=
  by sorry

end part_a_part_b_part_c_l507_507889


namespace combined_daily_wages_b_d_f_l507_507972

-- Daily wages denoted as A, B, C, D, E, F for workers a, b, c, d, e, f respectively
variables (A B C D E F : ℝ)
-- Working days for each worker
variables (wA wB wC wD wE wF : ℝ)
-- Total earnings condition
variable (totalEarnings : ℝ)

-- Ratios of daily wages
axiom ratio_AB : A / B = 3 / 4
axiom ratio_BC : B / C = 4 / 5
axiom ratio_CD : C / D = 5 / 6
axiom ratio_DE : D / E = 6 / 7
axiom ratio_EF : E / F = 7 / 8

-- Working days
axiom work_days : wA = 6 ∧ wB = 9 ∧ wC = 4 ∧ wD = 12 ∧ wE = 8 ∧ wF = 5

-- Total earnings of all six workers
axiom total_earnings : wA * A + wB * B + wC * C + wD * D + wE * E + wF * F = totalEarnings

-- Given total earnings
constant given_total_earnings : totalEarnings = 2970

-- Find combined daily wages of b, d, and f
theorem combined_daily_wages_b_d_f : B + D + F = 661.98 := sorry

end combined_daily_wages_b_d_f_l507_507972


namespace magnitude_vector_sum_l507_507250

-- Definitions for the given conditions
variable (a b : ℝ^3) -- Vectors in three-dimensional space
variable (ha : ‖a‖ = 1) -- Magnitude of vector a
variable (hb : ‖b‖ = 2) -- Magnitude of vector b
variable (h_angle : real.angle.cos (real.angle_between a b) = 1/2) -- Cosine of the angle between a and b is 1/2

-- The statement to be proven
theorem magnitude_vector_sum : ‖a + b‖ = real.sqrt 7 :=
sorry

end magnitude_vector_sum_l507_507250


namespace possible_values_of_x_and_factors_l507_507016

theorem possible_values_of_x_and_factors (p : ℕ) (h_prime : Nat.Prime p) :
  ∃ (x : ℕ), x = p^5 ∧ (∀ (d : ℕ), d ∣ x → d = p^0 ∨ d = p^1 ∨ d = p^2 ∨ d = p^3 ∨ d = p^4 ∨ d = p^5) ∧ Nat.divisors x ≠ ∅ ∧ (Nat.divisors x).card = 6 := 
  by 
    sorry

end possible_values_of_x_and_factors_l507_507016


namespace prime_numbers_satisfying_condition_l507_507232

theorem prime_numbers_satisfying_condition (p : ℕ) (hp : Nat.Prime p) :
  (∃ x : ℕ, 1 + p * 2^p = x^2) ↔ p = 2 ∨ p = 3 :=
by
  sorry

end prime_numbers_satisfying_condition_l507_507232


namespace find_k_l507_507724

-- Definitions and conditions from the given problem
def seq (n : ℕ) : ℚ :=
  if n = 1 then 15
  else if n = 2 then 43 / 3
  else if n >= 2 then (seq (n - 1) + seq (n - 3)) / 2
  else 0

-- Condition: seq 1 = 15
lemma seq_1 : seq 1 = 15 :=
by { unfold seq, simp }

-- Condition: seq 2 = 43 / 3
lemma seq_2 : seq 2 = 43 / 3 :=
by { unfold seq, simp }

-- Condition: 2 * seq (n + 1) = seq n + seq (n + 2)
lemma recursive_formula (n : ℕ) (hn : n > 0)  : 
  2 * seq (n + 1) = seq n + seq (n + 2) :=
sorry

-- Theorem: If seq k * seq (k + 1) < 0, then k = 23
theorem find_k (k : ℕ) : (seq k) * (seq (k + 1)) < 0 → k = 23 :=
sorry

end find_k_l507_507724


namespace smallest_a_b_sum_l507_507009

theorem smallest_a_b_sum : 
  ∃ (a b : ℝ), 
    (0 < a) ∧ (0 < b) ∧ 
    (a^2 >= 3 * b) ∧ 
    (b^2 >= (8 / 9) * a) ∧ 
    (a + b = 5) :=
begin
  sorry
end

end smallest_a_b_sum_l507_507009


namespace intersection_P_Q_l507_507407

def P : Set ℝ := { x : ℝ | 2 ≤ x ∧ x < 4 }
def Q : Set ℝ := { x : ℝ | 3 ≤ x }

theorem intersection_P_Q :
  P ∩ Q = { x : ℝ | 3 ≤ x ∧ x < 4 } :=
by
  sorry  -- Proof step will be provided here

end intersection_P_Q_l507_507407


namespace probability_odd_3_in_6_rolls_l507_507100

-- Definitions based on problem conditions
def probability_of_odd (outcome: ℕ) : ℚ := if outcome % 2 = 1 then 1/2 else 0 

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ := 
  ((Nat.choose n k : ℚ) * (p^k) * ((1 - p)^(n - k)))

-- Given problem
theorem probability_odd_3_in_6_rolls : 
  binomial_probability 6 3 (1/2) = 5 / 16 :=
by
  sorry

end probability_odd_3_in_6_rolls_l507_507100


namespace painted_faces_l507_507865

theorem painted_faces (n : ℕ) (h : n = 4) : 
  let total_corners := 8 in
  (let one_inch_cubes := n ^ 3 in 
  ∃ count_three_faces : ℕ, 
  count_three_faces = total_corners ∧ 
  count_three_faces = 8) :=
by
  sorry

end painted_faces_l507_507865


namespace smallest_integer_with_eight_factors_l507_507540

theorem smallest_integer_with_eight_factors : ∃ N : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) ∧ ∀ M : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) → N ≤ M :=
sorry -- Proof not provided.

end smallest_integer_with_eight_factors_l507_507540


namespace find_y_l507_507957

theorem find_y (x y : ℕ) (h1 : x % y = 7) (h2 : (x : ℚ) / y = 86.1) (h3 : Nat.Prime (x + y)) : y = 70 :=
sorry

end find_y_l507_507957


namespace molecular_weight_one_mole_n2o_l507_507106

noncomputable def molecular_weight_n2o (n : ℕ) [h : n = 8] : ℝ := 352

theorem molecular_weight_one_mole_n2o :
  (molecular_weight_n2o 1) = 44 :=
by
  sorry

end molecular_weight_one_mole_n2o_l507_507106


namespace total_digits_used_l507_507112

theorem total_digits_used (n : ℕ) (h : n = 2500) : 
  let first_n_even := (finset.range (2 * n + 1)).filter (λ x, x % 2 = 0)
  let count_digits := λ x, if x < 10 then 1 else if x < 100 then 2 else if x < 1000 then 3 else 4
  let total_digits := first_n_even.sum (λ x, count_digits x)
  total_digits = 9444 :=
by sorry

end total_digits_used_l507_507112


namespace triangle_side_ratio_l507_507763

theorem triangle_side_ratio (A B C : ℝ)
  (a b c : ℝ)
  (h : b * real.sin A * real.sin B + a * real.cos B ^ 2 = 2 * c) :
  a / c = 2 :=
sorry

end triangle_side_ratio_l507_507763


namespace benjamin_walks_95_miles_in_a_week_l507_507200

def distance_to_work_one_way : ℕ := 6
def days_to_work : ℕ := 5
def distance_to_dog_walk_one_way : ℕ := 2
def dog_walks_per_day : ℕ := 2
def days_for_dog_walk : ℕ := 7
def distance_to_best_friend_one_way : ℕ := 1
def times_to_best_friend : ℕ := 1
def distance_to_convenience_store_one_way : ℕ := 3
def times_to_convenience_store : ℕ := 2

def total_distance_in_a_week : ℕ :=
  (days_to_work * 2 * distance_to_work_one_way) +
  (days_for_dog_walk * dog_walks_per_day * distance_to_dog_walk_one_way) +
  (times_to_convenience_store * 2 * distance_to_convenience_store_one_way) +
  (times_to_best_friend * 2 * distance_to_best_friend_one_way)

theorem benjamin_walks_95_miles_in_a_week :
  total_distance_in_a_week = 95 :=
by {
  simp [distance_to_work_one_way, days_to_work,
        distance_to_dog_walk_one_way, dog_walks_per_day,
        days_for_dog_walk, distance_to_best_friend_one_way,
        times_to_best_friend, distance_to_convenience_store_one_way,
        times_to_convenience_store, total_distance_in_a_week],
  sorry
}

end benjamin_walks_95_miles_in_a_week_l507_507200


namespace fraction_to_zero_power_l507_507101

theorem fraction_to_zero_power :
  756321948 ≠ 0 ∧ -3958672103 ≠ 0 →
  (756321948 / -3958672103 : ℝ) ^ 0 = 1 :=
by
  intro h
  have numerator_nonzero : 756321948 ≠ 0 := h.left
  have denominator_nonzero : -3958672103 ≠ 0 := h.right
  -- Skipping the rest of the proof.
  sorry

end fraction_to_zero_power_l507_507101


namespace smallest_integer_with_eight_factors_l507_507547

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, 
  ∀ m : ℕ, (∀ p : ℕ, ∃ k : ℕ, m = p^k → (k + 1) * (p + 1) = 8) → (n ≤ m) ∧ 
  (∀ d : ℕ, d ∣ n → d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 24) :=
sorry

end smallest_integer_with_eight_factors_l507_507547


namespace tower_combinations_l507_507585

-- Definitions based on problem conditions
def total_cubes : ℕ := 10
def red_cubes : ℕ := 3
def blue_cubes : ℕ := 4
def green_cubes : ℕ := 3
def tower_height : ℕ := 6

-- Constraint definitions
def valid_tower (r b g : ℕ) : Prop :=
  r + b + g = tower_height ∧
  ((r = red_cubes ∨ b = blue_cubes ∨ g = green_cubes) ∧ ¬(r = red_cubes ∧ b = blue_cubes ∧ g = green_cubes))

-- Main theorem statement
theorem tower_combinations : Σ (r b g : ℕ), valid_tower r b g = 162 :=
sorry

end tower_combinations_l507_507585


namespace max_value_and_period_of_g_l507_507291

theorem max_value_and_period_of_g (a b : ℝ) (f g : ℝ → ℝ) (max_f min_f : ℝ) :
  (∀ x, f x = a - b * real.cos x) →
  max_f = 5 / 2 →
  min_f = -1 / 2 →
  (∀ x, g x = -4 * a * real.sin (b * x)) →
  ∃ (max_g period_g : ℝ), max_g = 4 ∧ period_g = 4 * real.pi / 3 :=
by
  sorry

end max_value_and_period_of_g_l507_507291


namespace travel_cost_interval_l507_507595

-- Define the cost function
def cost (a b : ℝ) : ℝ := b^3 - a * b^2

-- Define what it means to travel from 0 to 1 with a certain cost
def canTravelWithCost (c : ℝ) : Prop :=
  ∃ (n : ℕ) (b : Fin n → ℝ), 
    0 = b 0 ∧
    (∀ i : Fin (n-1), b i < b (i+1)) ∧ 
    b (n-1) = 1 ∧ 
    ∑ i in Finset.range n, cost (b i) (b (i+1)) = c

-- Prove that you can travel from 0 to 1 with cost "c" if and only if "c" is in the interval (1/3, 1]
theorem travel_cost_interval (c : ℝ) : canTravelWithCost c ↔ c ∈ Set.Icc (1 / 3) 1 :=
sorry

end travel_cost_interval_l507_507595


namespace six_and_zero_not_perfect_square_no_perfect_square_with_1_to_9_ending_in_5_l507_507642

-- Part (a)
theorem six_and_zero_not_perfect_square 
  (n : ℕ) (dec_rep: String) 
  (h1: ∀ (c : Char), c ∈ dec_rep → c = '0' ∨ c = '6')
  (h2: dec_rep.getLast = '6') 
  (h3: n = dec_rep.toNat) : 
  ¬ ∃ k : ℕ, n = k * k := 
sorry

-- Part (b)
theorem no_perfect_square_with_1_to_9_ending_in_5 
  (m : ℕ) (dec_rep: String) 
  (h1: ∀ (c : Char), c ∈ dec_rep.init → c ≠ '0' ∧ (c.toNat - '0'.toNat + 1) ∈ (Finset.range 9))
  (h2: dec_rep.getLast = '5') 
  (h3: dec_rep.length = 9) 
  (h4: m = dec_rep.toNat) : 
  ¬ ∃ k : ℕ, m = k * k := 
sorry

end six_and_zero_not_perfect_square_no_perfect_square_with_1_to_9_ending_in_5_l507_507642


namespace interval_increase_max_value_f_l507_507295

noncomputable def f (a x : ℝ) : ℝ := a^x + x^2 - x * Real.log a

noncomputable def f_prime (a x : ℝ) : ℝ := a^x * Real.log a + 2 * x - Real.log a

theorem interval_increase (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x, (0 < x → 0 < f_prime a x) ∧ (x < 0 → f_prime a x < 0) :=
sorry

theorem max_value_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (1 < a → f a 2 = a^2 + 4 - 2 * Real.log a) ∧ (0 < a ∧ a < 1 → f a (-2) = a^(-2) + 4 + 2 * Real.log a) :=
sorry

end interval_increase_max_value_f_l507_507295


namespace bisect_chord_of_circumcircle_l507_507819

theorem bisect_chord_of_circumcircle {A B C D P Q M : Type*} [cyclic A B C D] 
  (hM : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (hM_diag : ∃ (MA : (A : Set) ∩ (B : Set)), has_inter.inter (HasInter.inter_inter A B) MA = M)
  (hPQ : ∃ (PQ : (P : Set) ∩ (Q : Set)), M.itns PQ PQ = Some (MP = MQ))
  (hCirc : {Circ : Set M}) :
  ∃ chord (e : P Q) (circ e.M) := 
begin
  sorry
end

end bisect_chord_of_circumcircle_l507_507819


namespace extreme_value_interval_l507_507284

theorem extreme_value_interval (a : ℝ) :
  (∃ x : ℝ, 2 < x ∧ x < 3 ∧ f_derivative_on_interval x) → (5/4 < a ∧ a < 5/3) :=
by
  sorry

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * a * x^2 + 3 * x + 1

noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - 6 * a * x + 3

def f_derivative_on_interval (x : ℝ) : Prop :=
  f'(x) = 0

end extreme_value_interval_l507_507284


namespace locus_of_P_is_circle_l507_507253

noncomputable theory
open_locale classical

-- Definitions based on problem conditions
variables {θ : ℝ} (hθ : 0 < θ ∧ θ < π / 2)
variables {O₁ O₂ A B M N P : Type*}
variables [fact (circle O₁ A)] [fact (circle O₂ A)]
variables (line_l : line O₁ O₂ A B)

-- Geometry assumptions given in the problem
variables (major_arc_MB : major_arc O₁ A B)
variables [intersection_segment : ∀ M ∈ major_arc_MB, point (segment O₀ M) ∈ circle O₂ A]
variables (P : ∀ M ∈ major_arc_MB, ray (segment O₁ B) P)
variables [angle_theta : ∀ M ∈ major_arc_MB, angle (spoke O₁) (P M) = θ]

-- Statement
theorem locus_of_P_is_circle 
  (h_fixed_angle : 0 < θ ∧ θ < π / 2)
  (tangency_point : O₁ ∈ circle O₂ A) 
  (line_intersect : l ∩ circle O₁ = {A, B})
  (point_on_arc : ∀ M, M ∈ major_arc O₁ − {A, B})
  (intersection_point : ∀ M, N ∈ segment A M ∩ circle O₂)
  (angle_condition : ∀ M, angle M P N = θ)
  : ∃ S, ∀ M, P M ∈ circle A :=
sorry

end locus_of_P_is_circle_l507_507253


namespace gold_weight_l507_507622

theorem gold_weight:
  ∀ (G C A : ℕ), 
  C = 9 → 
  (A = (4 * G + C) / 5) → 
  A = 17 → 
  G = 19 :=
by
  intros G C A hc ha h17
  sorry

end gold_weight_l507_507622


namespace find_abs_z_find_complex_z_l507_507716

noncomputable def z_satisfies_condition1 (z : ℂ) : Prop :=
  abs (2 * z + 5) = abs (z + 10)

noncomputable def z_on_bisector (z : ℂ) : Prop :=
  let w := (1 - 2 * complex.I) * z in
  w.re = w.im

theorem find_abs_z (z : ℂ) (hz1 : z_satisfies_condition1 z) : abs z = 5 :=
sorry

theorem find_complex_z (z : ℂ) (hz1 : z_satisfies_condition1 z) (hz2 : z_on_bisector z) :
  z = (sqrt 10 / 2) - (3 * sqrt 10 / 2) * complex.I ∨
  z = -(sqrt 10 / 2) + (3 * sqrt 10 / 2) * complex.I :=
sorry

end find_abs_z_find_complex_z_l507_507716


namespace experiment_implies_101_sq_1_equals_10200_l507_507815

theorem experiment_implies_101_sq_1_equals_10200 :
    (5^2 - 1 = 24) →
    (7^2 - 1 = 48) →
    (11^2 - 1 = 120) →
    (13^2 - 1 = 168) →
    (101^2 - 1 = 10200) :=
by
  repeat { intro }
  sorry

end experiment_implies_101_sq_1_equals_10200_l507_507815


namespace fluid_motion_through_circle_l507_507469

noncomputable def complexPotential (z : ℂ) : ℂ :=
  Complex.log (Complex.sin (Real.pi * z))

def flowRate (f : ℂ → ℂ) (circle : ℂ → Prop) : ℝ :=
  let derivatives := deriv f
  let integrand (z : ℂ) := derivatives z
  let integral := ∮ z in circle, integrand z
  (integral.imaginary)

def circulation (f : ℂ → ℂ) (circle : ℂ → Prop) : ℝ :=
  let derivatives := deriv f
  let integrand (z : ℂ) := derivatives z
  let integral := ∮ z in circle, integrand z
  (integral.real)

theorem fluid_motion_through_circle :
  (circle : ℂ → Prop :=
  function z, Complex.abs z = 3 / 2)
  flowRate complexPotential circle = 6 * Real.pi ^ 2 ∧
  circulation complexPotential circle = 0 :=
by
  sorry

end fluid_motion_through_circle_l507_507469


namespace arithmetic_mean_of_fractions_l507_507894

theorem arithmetic_mean_of_fractions :
  (3/8 + 5/9) / 2 = 67 / 144 :=
by
  sorry

end arithmetic_mean_of_fractions_l507_507894


namespace solution_set_of_inequality_l507_507479

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 - 3x + 4 > 0} = set.Ioo (-4) 1 :=
by
  sorry

end solution_set_of_inequality_l507_507479


namespace tomatoes_for_5_liters_l507_507881

theorem tomatoes_for_5_liters (kg_per_3_liters : ℝ) (liters_needed : ℝ) :
  (kg_per_3_liters = 69 / 3) → (liters_needed = 5) → (kg_per_3_liters * liters_needed = 115) := 
by
  intros h1 h2
  sorry

end tomatoes_for_5_liters_l507_507881


namespace Mikaela_tutored_more_hours_l507_507812

theorem Mikaela_tutored_more_hours :
  ∀ (rate hours1 saved : ℕ),
  rate = 10 ∧ hours1 = 35 ∧ saved = 150 →
  let total_earnings := 5 * saved in
  let earnings1 := hours1 * rate in
  ∃ (x : ℕ), earnings1 + (hours1 + x) * rate = total_earnings ∧ x = 5 :=
by
  intros rate hours1 saved h
  let total_earnings := 5 * saved
  let earnings1 := hours1 * rate
  use 5
  sorry

end Mikaela_tutored_more_hours_l507_507812


namespace problem_equiv_proof_l507_507257

variables {a b : ℝ} {a_n : ℕ → ℝ}

def S (n : ℕ) := a_n n ^ 2 + b * n

theorem problem_equiv_proof 
  (h1 : S 25 = 100) 
  (h2 : a_{12} + a_{14} = a_{1} + a_{25}) :
  a_{12} + a_{14} = 8 := 
sorry

end problem_equiv_proof_l507_507257


namespace find_m_value_l507_507728

variable (U A : Set ℝ)
variable (m : ℝ)

def universal_set : Set ℝ := {4, m^2 + 2*m - 3, 19}
def set_A : Set ℝ := {5}
def complement_A_relative_U : Set ℝ := {|4*m - 3|, 4}

theorem find_m_value (hU : U = universal_set m) (hA : A = set_A) (hC : compl A U = complement_A_relative_U m) :
  m = -4 := by
  sorry

end find_m_value_l507_507728


namespace right_side_longer_l507_507083

/-- The sum of the three sides of a triangle is 50. 
    The right side of the triangle is a certain length longer than the left side, which has a value of 12 cm. 
    The triangle base has a value of 24 cm. 
    Prove that the right side is 2 cm longer than the left side. -/
theorem right_side_longer (L R B : ℝ) (hL : L = 12) (hB : B = 24) (hSum : L + B + R = 50) : R = L + 2 :=
by
  sorry

end right_side_longer_l507_507083


namespace extreme_value_m_eq_1_monotonicity_of_f_min_int_value_m_l507_507287

-- Define the function f(x)
def f (x m : ℝ) : ℝ := Real.log x - m * x ^ 2 + (1 - 2 * m) * x + 1

-- Part 1: Prove the extreme value when m = 1
theorem extreme_value_m_eq_1 : 
  ∀ x > 0, f x 1 ≤ f (0.5) 1 := sorry

-- Part 2: Discussing the monotonicity of f(x)
theorem monotonicity_of_f :
  ∀ m x, (m ≤ 0 → f' x m > 0) ∧ 
         (m > 0 → ((x < 1 / 2 / m) → (f' x m > 0)) ∧ ((x > 1 / 2 / m) → (f' x m < 0))) := sorry

-- Part 3: Minimum integer value of m for f(x) ≤ 0 for all x > 0
theorem min_int_value_m (x : ℝ) (h : x > 0) : 
  ∀ x > 0, f x 1 ≤ 0 := sorry

end extreme_value_m_eq_1_monotonicity_of_f_min_int_value_m_l507_507287


namespace arithmetic_mean_of_fractions_l507_507915

theorem arithmetic_mean_of_fractions : 
  (3 : ℚ) / 8 + (5 : ℚ) / 9 = 2 * (67 : ℚ) / 144 :=
by {
  rw [(show (3 : ℚ) / 8 + (5 : ℚ) / 9 = (27 : ℚ) / 72 + (40 : ℚ) / 72, by sorry),
      (show (27 : ℚ) / 72 + (40 : ℚ) / 72 = (67 : ℚ) / 72, by sorry),
      (show (2 : ℚ) * (67 : ℚ) / 144 = (67 : ℚ) / 72, by sorry)]
}

end arithmetic_mean_of_fractions_l507_507915


namespace cos_sub_eq_frac_five_over_eight_l507_507708

theorem cos_sub_eq_frac_five_over_eight (A B : ℝ)
  (h1 : sin A + sin B = 3 / 2)
  (h2 : cos A + cos B = 1) :
  cos (A - B) = 5 / 8 :=
sorry

end cos_sub_eq_frac_five_over_eight_l507_507708


namespace deductive_reasoning_example_l507_507137

theorem deductive_reasoning_example :
  (∀ (L : Type) (parallel : L → L → Prop) (C : L → L → Prop) (supp : L → ℝ) (A B : L),
    parallel A B → C A B →
    ∠A + ∠B = 180°) → 
  (Option_A_is_deductive_reasoning : Prop) := 
by
  intros
  sorry

end deductive_reasoning_example_l507_507137


namespace smallest_integer_with_eight_factors_l507_507555

theorem smallest_integer_with_eight_factors :
  ∃ n : ℕ, (∀ m : ℕ, (∃ k : ℕ, m has_factors k ∧ k = 8) → m ≥ n) ∧ n = 24 :=
by
  sorry

end smallest_integer_with_eight_factors_l507_507555


namespace people_on_trolley_final_l507_507617

-- Defining the conditions from part a)
def people_first_stop : ℕ := 10
def people_off_second_stop : ℕ := 3
def people_on_second_stop : ℕ := 2 * people_first_stop
def people_off_third_stop : ℕ := 18
def people_on_third_stop : ℕ := 2

-- Proving the theorem that the number of people on the trolley after the third stop is 12.
theorem people_on_trolley_final :
  let initial_people := 1 in -- the trolley driver alone initially
  let after_first_stop := initial_people + people_first_stop in
  let after_second_stop := after_first_stop - people_off_second_stop + people_on_second_stop in
  let final_count := after_second_stop - people_off_third_stop + people_on_third_stop in
  final_count = 12 :=
by 
  -- Proof omitted
  sorry

end people_on_trolley_final_l507_507617


namespace total_fireworks_correct_l507_507186

variable (fireworks_num fireworks_reg)
variable (fireworks_H fireworks_E fireworks_L fireworks_O)
variable (fireworks_square fireworks_triangle fireworks_circle)
variable (boxes fireworks_per_box : ℕ)

-- Given Conditions
def fireworks_years_2021_2023 : ℕ := 6 * 4 * 3
def fireworks_HAPPY_NEW_YEAR : ℕ := 5 * 11 + 6
def fireworks_geometric_shapes : ℕ := 4 + 3 + 12
def fireworks_HELLO : ℕ := 8 + 7 + 6 * 2 + 9
def fireworks_additional_boxes : ℕ := 100 * 10

-- Total Fireworks
def total_fireworks : ℕ :=
  fireworks_years_2021_2023 + 
  fireworks_HAPPY_NEW_YEAR + 
  fireworks_geometric_shapes + 
  fireworks_HELLO + 
  fireworks_additional_boxes

theorem total_fireworks_correct : 
  total_fireworks = 1188 :=
  by
  -- The proof is omitted.
  sorry

end total_fireworks_correct_l507_507186


namespace ratio_of_areas_l507_507428

-- Define the side lengths and areas
def side_length (perimeter : ℕ) : ℕ := perimeter / 4
def area (side : ℕ) : ℕ := side * side

-- Given conditions
def perimeter_A := 16
def perimeter_B := 32

-- Calculate side lengths
def side_A := side_length perimeter_A
def side_B := side_length perimeter_B
def side_C := 2 * side_B -- since side_C is double the side_B

-- Calculate areas
def area_B := area side_B
def area_C := area side_C

-- The theorem we need to prove
theorem ratio_of_areas : area_B.toRat / area_C.toRat = 1 / 4 :=
by
  -- Sorry used to skip the actual proof
  sorry

end ratio_of_areas_l507_507428


namespace consecutive_even_sum_l507_507082

theorem consecutive_even_sum : 
  ∃ n : ℕ, 
  (∃ x : ℕ, (∀ i : ℕ, i < n → (2 * i + x = 14 → i = 2) → 
  2 * x + (n - 1) * n = 52) ∧ n = 4) :=
by
  sorry

end consecutive_even_sum_l507_507082


namespace ratio_of_red_to_blue_marbles_l507_507874

theorem ratio_of_red_to_blue_marbles:
  ∀ (R B : ℕ), 
    R + B = 30 →
    2 * (20 - B) = 10 →
    B = 15 → 
    R = 15 →
    R / B = 1 :=
by intros R B h₁ h₂ h₃ h₄
   sorry

end ratio_of_red_to_blue_marbles_l507_507874


namespace triangle_angle_A_is_pi_over_3_l507_507259

variable (a b c S : ℝ)
variable (A B C : ℝ)
variable [fact (0 < A)] [fact (A < π)]

-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively
-- Given the area S of the triangle
-- Given that (b + c)² - a² = 4√3 * S
-- Prove that the measure of angle A is π/3

theorem triangle_angle_A_is_pi_over_3
  (h1 : (b + c)^2 - a^2 = 4 * (Real.sqrt 3) * S)
  (h2 : S = 0.5 * b * c * Real.sin A)
  (h3 : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) :
  A = π / 3 :=
sorry

end triangle_angle_A_is_pi_over_3_l507_507259


namespace tangent_line_y_intercept_at_P_1_12_is_9_l507_507482

noncomputable def curve (x : ℝ) : ℝ := x^3 + 11

noncomputable def tangent_slope_at (x : ℝ) : ℝ := 3 * x^2

noncomputable def tangent_line_y_intercept : ℝ :=
  let P : ℝ × ℝ := (1, curve 1)
  let slope := tangent_slope_at 1
  P.snd - slope * P.fst

theorem tangent_line_y_intercept_at_P_1_12_is_9 :
  tangent_line_y_intercept = 9 :=
sorry

end tangent_line_y_intercept_at_P_1_12_is_9_l507_507482


namespace square_of_binomial_l507_507223

theorem square_of_binomial (c : ℝ) (h : c = 3600) :
  ∃ a : ℝ, (x : ℝ) → (x + a)^2 = x^2 + 120 * x + c := by
  sorry

end square_of_binomial_l507_507223


namespace range_of_a_l507_507262

theorem range_of_a (x y a : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 / x + 1 / y = 1) : 
  (x + y + a > 0) ↔ (a > -3 - 2 * Real.sqrt 2) :=
sorry

end range_of_a_l507_507262


namespace least_number_to_add_l507_507146

theorem least_number_to_add (n : ℕ) (d : ℕ) (r : ℕ) (k : ℕ) (l : ℕ) (h₁ : n = 1077) (h₂ : d = 23) (h₃ : n % d = r) (h₄ : d - r = k) (h₅ : r = 19) (h₆ : k = l) : l = 4 :=
by
  sorry

end least_number_to_add_l507_507146


namespace exponential_function_a_eq_3_l507_507752

theorem exponential_function_a_eq_3 (a : ℝ) (f : ℝ → ℝ) (h1 : f = λ x, (a-2) * a^x) :
  a - 2 = 1 ∧ a > 0 ∧ a ≠ 1 → a = 3 :=
by
  intro h
  sorry

end exponential_function_a_eq_3_l507_507752


namespace number_of_ways_to_choose_routes_l507_507162

theorem number_of_ways_to_choose_routes (classes routes : ℕ) (h_classes : classes = 38) (h_routes : routes = 6) : 
  (routes ^ classes) = 6 ^ 38 :=
by
  rw [h_classes, h_routes]
  sorry

end number_of_ways_to_choose_routes_l507_507162


namespace range_of_m_l507_507723

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.exp x - m * x + 1

def f_prime (x : ℝ) (m : ℝ) : ℝ := Real.exp x - m

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, f_prime x m = -1 / Real.exp 1) ↔ m ∈ Set.Ioi (1 / Real.exp 1) :=
by
  sorry

end range_of_m_l507_507723


namespace arithmetic_mean_of_fractions_l507_507897

theorem arithmetic_mean_of_fractions :
  (3/8 + 5/9) / 2 = 67 / 144 :=
by
  sorry

end arithmetic_mean_of_fractions_l507_507897


namespace largest_quotient_of_set_l507_507103

theorem largest_quotient_of_set :
  let S := {-30, -5, -3, 1, 3, 10, 15}
  ∃ a b ∈ S, a ≠ b ∧ (∀ x y ∈ S, x ≠ y → abs ((a: ℤ) / b) ≥ abs ((x: ℤ) / y)) ∧ abs ((a: ℤ) / b) = 15 := by
sorry

end largest_quotient_of_set_l507_507103


namespace min_side_length_l507_507448

noncomputable def side_length_min : ℝ := 30

theorem min_side_length (s r : ℝ) (hs₁ : s^2 ≥ 900) (hr₁ : π * r^2 ≥ 100) (hr₂ : 2 * r ≤ s) :
  s ≥ side_length_min :=
by
  sorry

end min_side_length_l507_507448


namespace staircase_steps_eq_twelve_l507_507830

theorem staircase_steps_eq_twelve (n : ℕ) :
  (3 * n * (n + 1) / 2 = 270) → (n = 12) :=
by
  intro h
  sorry

end staircase_steps_eq_twelve_l507_507830


namespace arithmetic_mean_of_fractions_l507_507925

theorem arithmetic_mean_of_fractions (a b : ℚ) (h1 : a = 3 / 8) (h2 : b = 5 / 9) :
  (a + b) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_fractions_l507_507925


namespace radius_of_circle_on_sphere_touches_all_three_l507_507089

variable (r a : ℝ)

-- Conditions
def three_circles_on_sphere (a r : ℝ) : Prop :=
  ∀ (α β γ : ℝ), 0 < α ∧ α < π / 2 ∧ α = β ∧ β = γ ∧ 
  ∃ O1 O2 O3 : ℝ → ℝ³, 
    dist O1 O2 = a ∧
    dist O2 O3 = a ∧
    dist O3 O1 = a

def circle_on_sphere_touches_all_three (a r : ℝ) : ℝ :=
  r * sin (arcsin (a / r) - arcsin (a * sqrt 3 / (3 * r)))

-- Proof problem statement
theorem radius_of_circle_on_sphere_touches_all_three (a r : ℝ) :
  three_circles_on_sphere a r → 
  ∃ R : ℝ, R = circle_on_sphere_touches_all_three a r := by
  sorry

end radius_of_circle_on_sphere_touches_all_three_l507_507089


namespace average_age_of_team_l507_507457

theorem average_age_of_team (A : ℕ) (captain_age : ℕ) (keeper_age : ℕ) (team_size : ℕ) (remaining_players : ℕ)
  (condition1 : team_size = 11)
  (condition2 : captain_age = 28)
  (condition3 : keeper_age = captain_age + 3)
  (condition4 : remaining_players = team_size - 2)
  (condition5 : ∑ i in range remaining_players, (A - 1) = (team_size - 2) * (A - 1))
  (condition6 : ∑ i in range team_size, A = 59 + ∑ i in range remaining_players, (A - 1)) :
  A = 25 := 
  sorry

end average_age_of_team_l507_507457


namespace arithmetic_mean_of_38_and_59_is_67_over_144_l507_507938

theorem arithmetic_mean_of_38_and_59_is_67_over_144 :
  (3 / 8 + 5 / 9) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_38_and_59_is_67_over_144_l507_507938


namespace exists_x_iff_q_congruent_1_mod_p_l507_507792

theorem exists_x_iff_q_congruent_1_mod_p
  (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hq_odd : q % 2 = 1) :
  (∃ x : ℤ, q ∣ (x + 1)^p - x^p) ↔ (q ≡ 1 [MOD p]) :=
by
  sorry

end exists_x_iff_q_congruent_1_mod_p_l507_507792


namespace sum_s_r_values_l507_507005

def r_values : List ℤ := [-2, -1, 0, 1, 3]
def r_range : List ℤ := [-1, 0, 1, 3, 5]

def s (x : ℤ) : ℤ := if 1 ≤ x then 2 * x + 1 else 0

theorem sum_s_r_values :
  (s 1) + (s 3) + (s 5) = 21 :=
by
  sorry

end sum_s_r_values_l507_507005


namespace total_digits_2500_is_9449_l507_507130

def nth_even (n : ℕ) : ℕ := 2 * n

def count_digits_in_range (start : ℕ) (stop : ℕ) : ℕ :=
  (stop - start) / 2 + 1

def total_digits (n : ℕ) : ℕ :=
  let one_digit := 4
  let two_digit := count_digits_in_range 10 98
  let three_digit := count_digits_in_range 100 998
  let four_digit := count_digits_in_range 1000 4998
  let five_digit := 1
  one_digit * 1 +
  two_digit * 2 +
  (three_digit * 3) +
  (four_digit * 4) +
  (five_digit * 5)

theorem total_digits_2500_is_9449 : total_digits 2500 = 9449 := by
  sorry

end total_digits_2500_is_9449_l507_507130


namespace primes_dividing_finite_implies_eq_l507_507489

theorem primes_dividing_finite_implies_eq (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h_finite_primes : ∃ P : finset ℕ, ∀ n : ℕ, n ≥ 1 → ∀ p : ℕ, nat.prime p → p ∣ a * b^n + c * d^n → p ∈ P) : b = d :=
by
  sorry

end primes_dividing_finite_implies_eq_l507_507489


namespace inequality_solution_set_l507_507866

theorem inequality_solution_set :
  {x : ℝ | (x / (x ^ 2 - 8 * x + 15) ≥ 2) ∧ (x ^ 2 - 8 * x + 15 ≠ 0)} =
  {x : ℝ | (5 / 2 ≤ x ∧ x < 3) ∨ (5 < x ∧ x ≤ 6)} :=
by
  -- The proof is omitted
  sorry

end inequality_solution_set_l507_507866


namespace space_diagonal_of_cube_l507_507081

theorem space_diagonal_of_cube (s d : ℝ) (h1 : s = real.sqrt(12.5)) (h2 : d = real.sqrt(3) * s) : d ≈ 6.13 :=
by
-- we need to show that d approximately equals 6.13 under given conditions.
sorry

end space_diagonal_of_cube_l507_507081


namespace concurrency_of_lines_l507_507777

variable {A B C : Type*} [AffineGeometry A B C]

def triangle_ABC (A B C : A) : Prop := 
  ∠CAB = 45 ∧ ∠ABC = 22.5 

theorem concurrency_of_lines (h_triangle : triangle_ABC A B C) :
  let angle_bisector_A := line_segment A (midpoint A C)
  let median_B := line_segment B (midpoint A C)
  let altitude_C := perpendicular C (line AC)
  ∃ D : A, D ∈ (angle_bisector_A ∩ median_B ∩ altitude_C) :=
sorry

end concurrency_of_lines_l507_507777


namespace twin_primes_property_l507_507759

noncomputable def S (p : ℕ) : ℚ :=
  let k := ⌊(2 * p - 1) / 3⌋ in
  (Finset.range k).sum (λ i, 1 / (i + 1) * Nat.choose (p - 1) i)

theorem twin_primes_property (p : ℕ) (hp : Nat.Prime (p - 2) ∧ Nat.Prime p) (hk : k = ⌊(2 * p - 1) / 3⌋)
  (S : ℕ → ℚ) : (1 + (Finset.range k).sum (λ i, 1 / (i + 1) * Nat.choose (p - 1) i)).denominator = 1 ∧
  p ∣ (1 + (Finset.range k).sum (λ i, 1 / (i + 1) * Nat.choose (p - 1) i)).numerator := by
  sorry

end twin_primes_property_l507_507759


namespace arithmetic_mean_eq_l507_507904

theorem arithmetic_mean_eq :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = (67 : ℚ) / 144 :=
by
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  have h₁ : a = 3 / 8 := by rfl
  have h₂ : b = 5 / 9 := by rfl
  sorry

end arithmetic_mean_eq_l507_507904


namespace equilateral_sum_of_squares_l507_507406

theorem equilateral_sum_of_squares (A B C D₁ D₂ E₁ E₂ E₃ : Type*)
  (side_length : ℝ) (congruent : ∀ (Δ₁ Δ₂ Δ₃ : Type*), Δ₁ = Δ₂ ∧ Δ₂ = Δ₃)
  (BD₁ BD₂ : ℝ) (BD_equal : BD₁ = BD₂)
  (BD₁_value : BD₁ = Real.sqrt 21)
  (ABC_equilateral : ∀ {Δ : Type*} (side : ℝ), Δ = ABC ∧ (side = 7 → Δ := equilateral))
  (triangle_equiv : ∀ {Δ : Type*} {s : ℝ} (A B C D E : Type*),
    Δ = AD₁E₁ ∧ Δ = AD₁E₂ ∧ Δ = AD₂E₃ ∧ side_length = s →
      congruence Δ ABC ∧ side_length = 7)
  (CE₁ CE₂ CE₃ : ℝ)
  (calculation : CE₁^2 + CE₂^2 + CE₃^2 = 294) :
  CE₁^2 + CE₂^2 + CE₃^2 = 294 := by
  sorry

end equilateral_sum_of_squares_l507_507406


namespace side_length_of_S2_l507_507826

-- Define our context and the statements we need to work with
theorem side_length_of_S2
  (r s : ℕ)
  (h1 : 2 * r + s = 2450)
  (h2 : 2 * r + 3 * s = 4000) : 
  s = 775 :=
sorry

end side_length_of_S2_l507_507826


namespace find_original_radius_l507_507361

noncomputable def original_radius (x : ℝ) (r : ℝ) : Prop :=
  let V_orig := (4 : ℝ) * π * r^2
  let V_radius_inc := (4 : ℝ) * π * (r + 4)^2
  let V_height_inc := (12 : ℝ) * π * r^2
  (V_radius_inc - V_orig = x) ∧ (V_height_inc - V_orig = x) 

theorem find_original_radius (x : ℝ) (r : ℝ) : original_radius x r → r = 2 + 2 * real.sqrt 3 :=
by
  -- proof omitted
  sorry

end find_original_radius_l507_507361


namespace smallest_number_with_eight_factors_l507_507531

-- Definition of a function to count the number of distinct positive factors
def count_distinct_factors (n : ℕ) : ℕ := (List.range n).filter (fun d => d > 0 ∧ n % d = 0).length

-- Statement to prove the main problem
theorem smallest_number_with_eight_factors (n : ℕ) :
  count_distinct_factors n = 8 → n = 24 :=
by
  sorry

end smallest_number_with_eight_factors_l507_507531


namespace range_of_a_l507_507322

def is_monotonically_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f(x) ≥ f(y)

theorem range_of_a (a : ℝ) :
  is_monotonically_decreasing_on (λ x : ℝ, x^2 + 2*x + a * Real.log x) {x | 0 < x ∧ x < 1} ↔ a ≤ -4 :=
by
  sorry

end range_of_a_l507_507322


namespace arithmetic_mean_of_fractions_l507_507921

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 := 
by 
  sorry

end arithmetic_mean_of_fractions_l507_507921


namespace symmetry_y_axis_l507_507349

theorem symmetry_y_axis (P : ℝ × ℝ × ℝ) (hP : P = (3, -2, 1)) : 
  ∃ Q : ℝ × ℝ × ℝ, Q = (-3, -2, -1) ∧ Q = (-P.1, P.2, -P.3) :=
by
  -- Given conditions
  let P : ℝ × ℝ × ℝ := (3, -2, 1)
  have hQ : (-P.1, P.2, -P.3) = (-3, -2, -1) := sorry
  exact ⟨(-P.1, P.2, -P.3), hQ, rfl⟩

end symmetry_y_axis_l507_507349


namespace cosine_set_solution_l507_507863

theorem cosine_set_solution :
    {x | x ∈ set.Icc 0 real.pi ∧ real.cos (real.pi * real.cos x) = 0} = 
    {real.pi / 3, 2 * real.pi / 3} :=
by
  sorry

end cosine_set_solution_l507_507863


namespace weight_of_six_moles_BaF2_l507_507105

variable (atomic_weight_Ba : ℝ := 137.33) -- Atomic weight of Barium in g/mol
variable (atomic_weight_F : ℝ := 19.00) -- Atomic weight of Fluorine in g/mol
variable (moles_BaF2 : ℝ := 6) -- Number of moles of BaF2

theorem weight_of_six_moles_BaF2 :
  moles_BaF2 * (atomic_weight_Ba + 2 * atomic_weight_F) = 1051.98 :=
by sorry

end weight_of_six_moles_BaF2_l507_507105


namespace arithmetic_mean_of_fractions_l507_507917

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 := 
by 
  sorry

end arithmetic_mean_of_fractions_l507_507917


namespace value_of_a_minus_b_l507_507754

theorem value_of_a_minus_b (a b : ℝ) :
  (∀ x, - (1 / 2 : ℝ) < x ∧ x < (1 / 3 : ℝ) → ax^2 + bx + 2 > 0) → a - b = -10 := by
sorry

end value_of_a_minus_b_l507_507754


namespace archer_score_below_8_probability_l507_507858

theorem archer_score_below_8_probability :
  ∀ (p10 p9 p8 : ℝ), p10 = 0.2 → p9 = 0.3 → p8 = 0.3 → 
  (1 - (p10 + p9 + p8) = 0.2) :=
by
  intros p10 p9 p8 hp10 hp9 hp8
  rw [hp10, hp9, hp8]
  sorry

end archer_score_below_8_probability_l507_507858


namespace arithmetic_mean_of_fractions_l507_507920

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 := 
by 
  sorry

end arithmetic_mean_of_fractions_l507_507920


namespace product_of_common_divisors_of_180_and_45_l507_507237

def divisors (n : ℤ) : Set ℤ := { d | d ∣ n }

def common_divisors_product (a b : ℤ) : ℤ :=
  (divisors a ∩ divisors b).prod id

theorem product_of_common_divisors_of_180_and_45 : 
  common_divisors_product 180 45 = 8300625625 := 
by 
  sorry

end product_of_common_divisors_of_180_and_45_l507_507237


namespace range_of_m_condition_l507_507703

theorem range_of_m_condition :
  (∀ x : ℝ, mx^2 - (3-m)x + 1 > 0) ∨ (∀ x : ℝ, mx > 0) ↔ (1/9 < m ∧ m < 1) :=
by
  sorry

end range_of_m_condition_l507_507703


namespace arithmetic_mean_of_fractions_l507_507924

theorem arithmetic_mean_of_fractions (a b : ℚ) (h1 : a = 3 / 8) (h2 : b = 5 / 9) :
  (a + b) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_fractions_l507_507924


namespace smallest_integer_with_eight_factors_l507_507508

theorem smallest_integer_with_eight_factors:
  ∃ n : ℕ, ∀ (d : ℕ), d > 0 → d ∣ n → 8 = (divisor_count n) 
  ∧ (∀ m : ℕ, m > 0 → (∀ (d : ℕ), d > 0 → d ∣ m → 8 = (divisor_count m)) → n ≤ m) → n = 24 :=
begin
  sorry
end

end smallest_integer_with_eight_factors_l507_507508


namespace minimum_lambda_l507_507706

theorem minimum_lambda (λ : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * sqrt (2 * x * y) ≤ λ * (x + y)) → λ ≥ 2 :=
sorry

end minimum_lambda_l507_507706


namespace arithmetic_seq_geometric_iff_rational_l507_507350

noncomputable def seq_contains_geometric (a b : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) : Prop :=
∃ (k : ℝ) (n1 n2 n3 : ℕ), 1 ≤ n1 ∧ n1 < n2 ∧ n2 < n3 ∧ 
  (a + n1 * b) / (a + n2 * b) = k ∧ (a + n2 * b) / (a + n3 * b) = k

theorem arithmetic_seq_geometric_iff_rational (a b : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) :
  seq_contains_geometric a b h_a h_b ↔ (a / b) ∈ ℚ :=
begin
  sorry
end

end arithmetic_seq_geometric_iff_rational_l507_507350


namespace x_pow_2048_minus_inv_pow_2048_eq_zero_l507_507712

theorem x_pow_2048_minus_inv_pow_2048_eq_zero (x : ℂ) (h : x - 1/x = 2 * complex.I) : x^2048 - 1/x^2048 = 0 :=
sorry

end x_pow_2048_minus_inv_pow_2048_eq_zero_l507_507712


namespace inverse_function_passing_point_l507_507278

theorem inverse_function_passing_point
  (f : ℝ → ℝ) (hf : Function.Bijective f) (h1 : f(4) = 3) : f⁻¹ 3 = 4 :=
by
  -- to prove
  sorry

end inverse_function_passing_point_l507_507278


namespace arithmetic_mean_eq_l507_507906

theorem arithmetic_mean_eq :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = (67 : ℚ) / 144 :=
by
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  have h₁ : a = 3 / 8 := by rfl
  have h₂ : b = 5 / 9 := by rfl
  sorry

end arithmetic_mean_eq_l507_507906


namespace candy_bar_cost_l507_507639

-- Definitions of conditions
def soft_drink_cost : ℕ := 4
def num_soft_drinks : ℕ := 2
def num_candy_bars : ℕ := 5
def total_cost : ℕ := 28

-- Proof Statement
theorem candy_bar_cost : (total_cost - num_soft_drinks * soft_drink_cost) / num_candy_bars = 4 := by
  sorry

end candy_bar_cost_l507_507639


namespace part_I_part_II_l507_507721

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + (1 - a) * x

theorem part_I (a : ℝ) (h_a : a ≠ 0) :
  (∃ x : ℝ, (x * (f a (1/x))) = 4 * x - 3 ∧ ∀ y, x = y → (x * (f a (1/x))) = 4 * x - 3) →
  a = 2 :=
sorry

noncomputable def f2 (x : ℝ) : ℝ := 2 / x - x

theorem part_II : ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f2 x1 > f2 x2 :=
sorry

end part_I_part_II_l507_507721


namespace slower_train_speed_is_36_l507_507888

-- Define the given conditions as Lean definitions
def speed_faster_train := 45 -- km/hr
def length_each_train := 45 -- meters
def time_to_overtake := 36 -- seconds

-- Speed of the slower train
def speed_slower_train (V : ℝ) : Prop :=
  let relative_speed := (speed_faster_train - V) * 5 / 18 in
  let distance_covered := 2 * length_each_train in
  relative_speed = distance_covered / time_to_overtake

-- The theorem to be proven
theorem slower_train_speed_is_36 : ∃ V, speed_slower_train V ∧ V = 36 :=
by
  -- We can use "choose" to extract the value from ∃ and then prove this value is 36.
  sorry

end slower_train_speed_is_36_l507_507888


namespace sin_pi_over_six_eq_half_l507_507485

theorem sin_pi_over_six_eq_half : Real.sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_pi_over_six_eq_half_l507_507485


namespace parity_of_f_find_a_l507_507696

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.exp x + a * Real.exp (-x)

theorem parity_of_f (a : ℝ) :
  (∀ x : ℝ, f (-x) a = f x a ↔ a = 1 ∨ a = -1) ∧
  (∀ x : ℝ, f (-x) a = -f x a ↔ a = -1) ∧
  (∀ x : ℝ, ¬(f (-x) a = f x a) ∧ ¬(f (-x) a = -f x a) ↔ ¬(a = 1 ∨ a = -1)) :=
by
  sorry

theorem find_a (h : ∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f x a ≥ f 0 a) : 
  a = 1 :=
by
  sorry

end parity_of_f_find_a_l507_507696


namespace linear_dependency_k_l507_507076

theorem linear_dependency_k (k : ℝ) :
  (∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧
    (c1 * 1 + c2 * 4 = 0) ∧
    (c1 * 2 + c2 * k = 0) ∧
    (c1 * 3 + c2 * 6 = 0)) ↔ k = 8 :=
by
  sorry

end linear_dependency_k_l507_507076


namespace integer_root_multiplicity_l507_507460

theorem integer_root_multiplicity 
  (b c d e f : ℤ)
  (p : polynomial ℤ := polynomial.C 1 * polynomial.X ^ 5 + 
                        polynomial.C b * polynomial.X ^ 4 + 
                        polynomial.C c * polynomial.X ^ 3 + 
                        polynomial.C d * polynomial.X ^ 2 + 
                        polynomial.C e * polynomial.X + 
                        polynomial.C f) :
  ∃ (n : ℕ), n ∈ {0, 1, 2, 5} ∧ ∃ (roots : multiset ℤ), roots.card = n ∧ ∀ x ∈ roots, polynomial.eval x p = 0 := 
begin
  sorry
end

end integer_root_multiplicity_l507_507460


namespace realNumbersGreaterThan8IsSet_l507_507194

-- Definitions based on conditions:
def verySmallNumbers : Type := {x : ℝ // sorry} -- Need to define what very small numbers would be
def interestingBooks : Type := sorry -- Need to define what interesting books would be
def realNumbersGreaterThan8 : Set ℝ := { x : ℝ | x > 8 }
def tallPeople : Type := sorry -- Need to define what tall people would be

-- Main theorem: Real numbers greater than 8 can form a set
theorem realNumbersGreaterThan8IsSet : Set ℝ :=
  realNumbersGreaterThan8

end realNumbersGreaterThan8IsSet_l507_507194


namespace inequality_proof_l507_507398

theorem inequality_proof (a b : ℝ) (n : ℕ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (h : (1/a) + (1/b) = 1) 
: (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end inequality_proof_l507_507398


namespace total_slices_per_pie_l507_507180

variable (apple_pie : ℕ → ℕ → ℕ → ℕ)
variable (blueberry_pie : ℕ → ℕ → ℕ → ℕ)
variable (cherry_pie : ℕ → ℕ → ℕ → ℕ)

variable (monday : ℕ)
variable (tuesday : ℕ)
variable (wednesday : ℕ)

theorem total_slices_per_pie :
  (apple_pie monday 3 7 4 + apple_pie tuesday 4 9 5 + apple_pie wednesday 5 6 8 = 51) ∧
  (blueberry_pie monday 2 5 6 + blueberry_pie tuesday 3 3 7 + blueberry_pie wednesday 1 4 2 = 33) ∧
  (cherry_pie monday 1 0 3 + cherry_pie tuesday 2 1 4 + cherry_pie wednesday 3 2 5 = 21) :=
by {
  sorry
}

end total_slices_per_pie_l507_507180


namespace a_share_correct_l507_507971

-- Investment periods for each individual in months
def investment_a := 12
def investment_b := 6
def investment_c := 4
def investment_d := 9
def investment_e := 7
def investment_f := 5

-- Investment multiplier for each individual
def multiplier_b := 2
def multiplier_c := 3
def multiplier_d := 4
def multiplier_e := 5
def multiplier_f := 6

-- Total annual gain
def total_gain := 38400

-- Calculate individual shares
def share_a (x : ℝ) := x * investment_a
def share_b (x : ℝ) := multiplier_b * x * investment_b
def share_c (x : ℝ) := multiplier_c * x * investment_c
def share_d (x : ℝ) := multiplier_d * x * investment_d
def share_e (x : ℝ) := multiplier_e * x * investment_e
def share_f (x : ℝ) := multiplier_f * x * investment_f

-- Calculate total investment
def total_investment (x : ℝ) :=
  share_a x + share_b x + share_c x + share_d x + share_e x + share_f x

-- Prove that a's share of the annual gain is Rs. 3360
theorem a_share_correct : 
  ∃ x : ℝ, (12 * x / total_investment x) * total_gain = 3360 := 
sorry

end a_share_correct_l507_507971


namespace min_num_cuboids_l507_507838

/-
Definitions based on the conditions:
- Dimensions of the cuboid are given as 3 cm, 4 cm, and 5 cm.
- We need to find the Least Common Multiple (LCM) of these dimensions.
- Calculate the volume of the smallest cube.
- Calculate the volume of the given cuboid.
- Find the number of such cuboids needed to form the cube.
-/
def cuboid_length : ℤ := 3
def cuboid_width : ℤ := 4
def cuboid_height : ℤ := 5

noncomputable def lcm_3_4_5 : ℤ := Int.lcm (Int.lcm cuboid_length cuboid_width) cuboid_height

noncomputable def cube_side_length : ℤ := lcm_3_4_5
noncomputable def cube_volume : ℤ := cube_side_length * cube_side_length * cube_side_length
noncomputable def cuboid_volume : ℤ := cuboid_length * cuboid_width * cuboid_height

noncomputable def num_cuboids : ℤ := cube_volume / cuboid_volume

theorem min_num_cuboids :
  num_cuboids = 3600 := by
  sorry

end min_num_cuboids_l507_507838


namespace problem_statement_l507_507288

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^6 + (Real.cos x)^6

theorem problem_statement :
  (∀ x, f x = (5 / 8) + (3 / 8) * Real.cos (4 * x)) ∧
  (∀ z : ℤ, ∃ k : ℤ, 4 * k * Real.pi = z * Real.pi → x = k * Real.pi / 4) ∧
  (∀ k : ℤ, ∀ x, f ((Real.pi / 8) + k * Real.pi / 4) = 5 / 8) :=
by
  -- Proof part (omitted)
  sorry

end problem_statement_l507_507288


namespace broadcast_arrangements_l507_507454

/-- 
The TV station plans to select 5 programs from 5 recorded news reports and 4 personality interview programs 
to broadcast one each day from October 1st to October 5th. If the number of news report programs cannot 
be less than 3, there are 9720 different broadcasting arrangements. 
-/
theorem broadcast_arrangements : 
  let news_reports := 5
  let interview_programs := 4
  let total_programs := 5 -∗ find_sequences news_reports interview_programs total_programs = 9720
  sorry

end broadcast_arrangements_l507_507454


namespace total_number_of_digits_l507_507119

-- Definitions based on identified conditions
def first2500EvenIntegers := {n : ℕ | n % 2 = 0 ∧ 1 ≤ n ∧ n ≤ 5000}

-- Theorem statement based on the equivalent proof problem
theorem total_number_of_digits : 
  (first2500EvenIntegers.count_digits = 9448) :=
sorry

end total_number_of_digits_l507_507119


namespace equifacial_iff_conditions_l507_507503

-- Definitions for conditions
def equifacial_tetrahedron (T : Tetrahedron) : Prop := 
  (∀ F1 F2 F3 : Face, F1 ≠ F2 → F2 ≠ F3 → F3 ≠ F1 → (T.face_eq F1 F2 ∧ T.face_eq F2 F3 ∧ T.face_eq F1 F3))

def condition_a (T : Tetrahedron) : Prop :=
  (∀ v1 v2 v3 : Vertex, T.sum_plane_angles_at v1 + T.sum_plane_angles_at v2 + T.sum_plane_angles_at v3 = 180)

def condition_b (T : Tetrahedron) : Prop :=
  (∃ v1 v2 : Vertex, T.sum_plane_angles_at v1 + T.sum_plane_angles_at v2 = 180 ∧ ∃ e1 e2 : Edge, T.opposite_edges_eq e1 e2)

def condition_c (T : Tetrahedron) : Prop :=
  (∃ v : Vertex, T.sum_plane_angles_at v = 180 ∧ ∃ e1 e2 e3 e4 : Edge, T.opposite_edges_eq_pairs e1 e2 e3 e4)

def condition_d (T : Tetrahedron) : Prop :=
  T.equal_four_angles

def condition_e (T : Tetrahedron) : Prop :=
  T.all_faces_equal_area

def condition_f (T : Tetrahedron) : Prop :=
  T.inscribed_and_circumscribed_centers_coincide

def condition_g (T : Tetrahedron) : Prop :=
  T.midpoints_opposite_edges_perpendicular

def condition_h (T : Tetrahedron) : Prop :=
  T.centroid_and_circumscribed_center_coincide

def condition_i (T : Tetrahedron) : Prop :=
  T.centroid_and_inscribed_center_coincide

-- Theorem statement
theorem equifacial_iff_conditions (T : Tetrahedron) :
  equifacial_tetrahedron T ↔
  condition_a T ∨
  condition_b T ∨
  condition_c T ∨
  condition_d T ∨
  condition_e T ∨
  condition_f T ∨
  condition_g T ∨
  condition_h T ∨
  condition_i T := sorry

end equifacial_iff_conditions_l507_507503


namespace product_of_roots_l507_507238

theorem product_of_roots :
  let a := 25
  let b := 60
  let c := -675
  let prod_roots := c / a
  prod_roots = -27 :=
by
  let a := 25
  let c := -675
  let prod_roots := c / a
  show prod_roots = -27 from
  sorry

end product_of_roots_l507_507238


namespace inequality_for_M_cap_N_l507_507014

def f (x : ℝ) := 2 * |x - 1| + x - 1
def g (x : ℝ) := 16 * x^2 - 8 * x + 1

def M := {x : ℝ | 0 ≤ x ∧ x ≤ 4 / 3}
def N := {x : ℝ | -1 / 4 ≤ x ∧ x ≤ 3 / 4}
def M_cap_N := {x : ℝ | 0 ≤ x ∧ x ≤ 3 / 4}

theorem inequality_for_M_cap_N (x : ℝ) (hx : x ∈ M_cap_N) : x^2 * f x + x * (f x)^2 ≤ 1 / 4 := 
by 
  sorry

end inequality_for_M_cap_N_l507_507014


namespace perp_line_eq_l507_507069

theorem perp_line_eq (x y : ℝ) (c : ℝ) (hx : x = 1) (hy : y = 2) (hline : 2 * x + y - 5 = 0) :
  x - 2 * y + c = 0 ↔ c = 3 := 
by
  sorry

end perp_line_eq_l507_507069


namespace number_of_digits_of_2500_even_integers_l507_507126

theorem number_of_digits_of_2500_even_integers : 
  let even_integers := List.range (5000 : Nat) in
  let first_2500_even := List.filter (fun n => n % 2 = 0) even_integers in
  List.length (List.join (first_2500_even.map (fun n => n.toDigits Nat))) = 9448 :=
by
  sorry

end number_of_digits_of_2500_even_integers_l507_507126


namespace smallest_integer_with_eight_factors_l507_507539

theorem smallest_integer_with_eight_factors : ∃ N : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) ∧ ∀ M : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) → N ≤ M :=
sorry -- Proof not provided.

end smallest_integer_with_eight_factors_l507_507539


namespace dice_inequality_probability_l507_507998

/-- A fair 8-sided die is rolled twice. The probability that the first number rolled is greater than or
equal to the second number rolled is 9/16. -/
theorem dice_inequality_probability :
  let num_faces := 8
  in let total_outcomes := num_faces * num_faces
  in let favorable_outcomes := 36
  in (favorable_outcomes : ℚ) / total_outcomes = 9 / 16 :=
by
  sorry

end dice_inequality_probability_l507_507998


namespace remaining_number_larger_than_4_l507_507453

theorem remaining_number_larger_than_4 (m : ℕ) (h : 2 ≤ m) (a : ℚ) (b : ℚ) (h_sum_inv : (1 : ℚ) - 1 / (2 * m + 1 : ℚ) = 3 / 4 + 1 / b) :
  b > 4 :=
by sorry

end remaining_number_larger_than_4_l507_507453


namespace yi_reads_more_than_jia_by_9_pages_l507_507175

-- Define the number of pages in the book
def total_pages : ℕ := 120

-- Define number of pages read per day by Jia and Yi
def pages_per_day_jia : ℕ := 8
def pages_per_day_yi : ℕ := 13

-- Define the number of days in the period
def total_days : ℕ := 7

-- Calculate total pages read by Jia in the given period
def pages_read_by_jia : ℕ := total_days * pages_per_day_jia

-- Calculate the number of reading days by Yi in the given period
def reading_days_yi : ℕ := (total_days / 3) * 2 + (total_days % 3).min 2

-- Calculate total pages read by Yi in the given period
def pages_read_by_yi : ℕ := reading_days_yi * pages_per_day_yi

-- Given all conditions, prove that Yi reads 9 pages more than Jia over the 7-day period
theorem yi_reads_more_than_jia_by_9_pages :
  pages_read_by_yi - pages_read_by_jia = 9 :=
by
  sorry

end yi_reads_more_than_jia_by_9_pages_l507_507175


namespace total_earnings_correct_l507_507785

-- Define the earnings of Terrence
def TerrenceEarnings : ℕ := 30

-- Define the difference in earnings between Jermaine and Terrence
def JermaineEarningsDifference : ℕ := 5

-- Define the earnings of Jermaine
def JermaineEarnings : ℕ := TerrenceEarnings + JermaineEarningsDifference

-- Define the earnings of Emilee
def EmileeEarnings : ℕ := 25

-- Define the total earnings
def TotalEarnings : ℕ := TerrenceEarnings + JermaineEarnings + EmileeEarnings

theorem total_earnings_correct : TotalEarnings = 90 := by
  sorry

end total_earnings_correct_l507_507785


namespace find_2013_group_l507_507814

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def find_group (number : ℕ) : ℕ :=
  let rec loop n :=
    if sum_first_n n < number then loop (n + 1) else n
  loop 1

theorem find_2013_group : find_group 1007 = 45 := by
  sorry

end find_2013_group_l507_507814


namespace find_angle_C_max_area_l507_507331

-- Define the conditions as hypotheses
variable (a b c : ℝ) (A B C : ℝ)
variable (h1 : c = 2 * Real.sqrt 3)
variable (h2 : c * Real.cos B + (b - 2 * a) * Real.cos C = 0)

-- Problem (1): Prove that angle C is π/3
theorem find_angle_C : C = Real.pi / 3 :=
by
  sorry

-- Problem (2): Prove that the maximum area of triangle ABC is 3√3
theorem max_area : (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 :=
by
  sorry

end find_angle_C_max_area_l507_507331


namespace expressions_divisible_by_17_l507_507433

theorem expressions_divisible_by_17 (a b : ℤ) : 
  let x := 3 * b - 5 * a
  let y := 9 * a - 2 * b
  (∃ k : ℤ, (2 * x + 3 * y) = 17 * k) ∧ (∃ k : ℤ, (9 * x + 5 * y) = 17 * k) :=
by
  exact ⟨⟨a, by sorry⟩, ⟨b, by sorry⟩⟩

end expressions_divisible_by_17_l507_507433


namespace sum_of_other_endpoint_is_negative_fifteen_l507_507857

theorem sum_of_other_endpoint_is_negative_fifteen
  (M N P: ℝ × ℝ)
  (hM : M = (10, -5))
  (hN : N = (15, 10))
  (midpoint : M = ((fst P + fst N) / 2, (snd P + snd N) / 2)) :
  (fst P + snd P) = -15 := by
  sorry

end sum_of_other_endpoint_is_negative_fifteen_l507_507857


namespace max_roses_l507_507974

noncomputable def individual_cost : ℝ := 6.30
noncomputable def dozen_cost : ℝ := 36
noncomputable def two_dozen_cost : ℝ := 50
noncomputable def budget : ℝ := 680

theorem max_roses (individual_cost dozen_cost two_dozen_cost budget: ℝ)
  (H1 : individual_cost = 6.30)
  (H2 : dozen_cost = 36)
  (H3 : two_dozen_cost = 50)
  (H4 : budget = 680) :
  (budget / two_dozen_cost) * 24 + (budget - (floor (budget / two_dozen_cost) * two_dozen_cost)) / individual_cost
  ≤ 316 :=
by
  sorry

end max_roses_l507_507974


namespace arc_VUW_eq_120_degrees_l507_507775

variable (P Q R U V W O : Type)
variables [IsoscelesTriangle P Q R] (QR : Segment P Q) (PQ PR: Segment P Q)
variable (r : ℝ) (h : ℝ)

def circle_radius (r h : ℝ) : Prop := r = 0.5 * h

def is_tangent (circle_tangent : Segment U QR) : Prop

def intersects_lines (circle_intersections: Segment V PQ) (circle_intersections': Segment W PR) : Prop

theorem arc_VUW_eq_120_degrees
  (H1 : PQ = PR)
  (H2 : IsoscelesTriangle P Q R)
  (H3 : circle_radius r (sqrt (PQ^2 - (QR / 2)^2)))
  (H4 : is_tangent U QR)
  (H5 : intersects_lines V PQ)
  (H6 : intersects_lines' W PR) :
  arc V U W = 120 :=
sorry

end arc_VUW_eq_120_degrees_l507_507775


namespace number_of_digits_of_2500_even_integers_l507_507127

theorem number_of_digits_of_2500_even_integers : 
  let even_integers := List.range (5000 : Nat) in
  let first_2500_even := List.filter (fun n => n % 2 = 0) even_integers in
  List.length (List.join (first_2500_even.map (fun n => n.toDigits Nat))) = 9448 :=
by
  sorry

end number_of_digits_of_2500_even_integers_l507_507127


namespace talia_drives_total_distance_l507_507449

-- Define the distances for each leg of the trip
def distance_house_to_park : ℕ := 5
def distance_park_to_store : ℕ := 3
def distance_store_to_friend : ℕ := 6
def distance_friend_to_house : ℕ := 4

-- Define the total distance Talia drives
def total_distance := distance_house_to_park + distance_park_to_store + distance_store_to_friend + distance_friend_to_house

-- Prove that the total distance is 18 miles
theorem talia_drives_total_distance : total_distance = 18 := by
  sorry

end talia_drives_total_distance_l507_507449


namespace triangle_probability_l507_507690

theorem triangle_probability (hexagon : set (set ℝ)) (h_hexagon : regular_hexagon hexagon) :
  let triangles := {tri | ∃ a b c ∈ hexagon.vertices, a ≠ b ∧ b ≠ c ∧ c ≠ a}
  let favorable := {tri ∈ triangles | ∃ ab ∈ hexagon.sides, ab ⊆ tri}
  (|favorable| : ℝ) / (|triangles| : ℝ) = 9 / 10 := 
sorry

end triangle_probability_l507_507690


namespace sufficient_not_necessary_l507_507392

theorem sufficient_not_necessary (x : ℝ) :
  (|x - 1| < 2 → x^2 - 4 * x - 5 < 0) ∧ ¬(x^2 - 4 * x - 5 < 0 → |x - 1| < 2) :=
by
  sorry

end sufficient_not_necessary_l507_507392


namespace find_pairs_l507_507807

theorem find_pairs (a b : ℕ) (h : a + b + a * b = 1000) : 
  (a = 6 ∧ b = 142) ∨ (a = 142 ∧ b = 6) ∨
  (a = 10 ∧ b = 90) ∨ (a = 90 ∧ b = 10) ∨
  (a = 12 ∧ b = 76) ∨ (a = 76 ∧ b = 12) :=
by sorry

end find_pairs_l507_507807


namespace even_3_digit_numbers_with_digit_sum_27_l507_507065

theorem even_3_digit_numbers_with_digit_sum_27 : 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ (n % 2 = 0) ∧ (sum_of_digits n = 27) → false :=
by
  sorry

-- Definition to compute sum of digits
def sum_of_digits (n : ℕ) : ℕ := 
  let d1 := n / 100 % 10
  let d2 := n / 10 % 10
  let d3 := n % 10
  d1 + d2 + d3

end even_3_digit_numbers_with_digit_sum_27_l507_507065


namespace sufficient_condition_for_having_skin_l507_507842

theorem sufficient_condition_for_having_skin (H_no_skin_no_hair : ¬skin → ¬hair) :
  (hair → skin) :=
sorry

end sufficient_condition_for_having_skin_l507_507842


namespace frozen_yogurt_combinations_l507_507594

theorem frozen_yogurt_combinations :
  (5 * (nat.choose 7 3) = 175) :=
by
  have x := nat.choose 7 3,
  calc
    5 * x = 5 * 35 : by rw nat.choose_eq_factorial_div_factorial
    ...    = 175   : by norm_num

end frozen_yogurt_combinations_l507_507594


namespace line_shift_right_by_one_l507_507853

-- Definitions of the original and transformed lines
def original_line (x : ℝ) : ℝ := 2 * x + 1
def transformed_line (x : ℝ) : ℝ := 2 * x - 1

-- Proposition that the shift is by 1 unit to the right
theorem line_shift_right_by_one : 
  ∃ a : ℝ, (∀ x : ℝ, transformed_line x = original_line (x + a)) ∧ a = 1 :=
by
  exists -1
  intros x
  dsimp [transformed_line, original_line]
  linarith
  sorry

end line_shift_right_by_one_l507_507853


namespace find_f_neg_a_l507_507267

noncomputable def f (x : ℝ) : ℝ := x^3 * (Real.exp x + Real.exp (-x)) + 2

theorem find_f_neg_a (a : ℝ) (h : f a = 4) : f (-a) = 0 :=
by
  sorry

end find_f_neg_a_l507_507267


namespace complement_of_union_l507_507015

open Set

namespace Proof

-- Define the universal set U, set A, and set B
def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- The complement of the union of sets A and B with respect to U
theorem complement_of_union (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4, 5}) 
  (hA : A = {1, 3}) (hB : B = {3, 5}) : 
  U \ (A ∪ B) = {0, 2, 4} :=
by {
  sorry
}

end Proof

end complement_of_union_l507_507015


namespace power_function_increasing_l507_507473

noncomputable def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

def power_function (m : ℝ) (x : ℝ) : ℝ :=
  (m^2 - m - 1) * x^(5 * m + 3)

theorem power_function_increasing {m : ℝ} (h1 : 0 < m)
  (h2 : m ≠ -1) (h3 : m = 2 → is_increasing_on (power_function m) 0 ∞) :
  m = 2 :=
  sorry

end power_function_increasing_l507_507473


namespace concrete_used_for_built_anchor_l507_507374

theorem concrete_used_for_built_anchor 
    (total_concrete : ℕ) 
    (deck_concrete : ℕ) 
    (pillars_concrete : ℕ) 
    (equal_anchors : Bool) 
    (one_anchor_built : Bool) 
    (total_concrete = 4800) 
    (deck_concrete = 1600) 
    (pillars_concrete = 1800) 
    (equal_anchors = true) 
    (one_anchor_built = true) : 
    ∃ (anchor1_concrete : ℕ), anchor1_concrete = 700 :=
by {
  sorry
}

end concrete_used_for_built_anchor_l507_507374


namespace math_problem_l507_507393

theorem math_problem (x y n : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100)
  (hy_reverse : ∃ a b, x = 10 * a + b ∧ y = 10 * b + a) 
  (h_xy_square_sum : x^2 + y^2 = n^2) : x + y + n = 132 :=
sorry

end math_problem_l507_507393


namespace smallest_positive_integer_with_eight_factors_l507_507518

theorem smallest_positive_integer_with_eight_factors :
  ∃ n : ℕ, (∀ m : ℕ, (m < n ∧ (∀ d : ℕ, d | m → d = 1 ∨ d = m) → (∃ a b : ℕ, distinct_factors_count m a b ∧ a = 8)) → n = 24) :=
by
  sorry

def distinct_factors_count (n : ℕ) (a b : ℕ) : Prop :=
  ∃ (p q : ℕ), prime p ∧ prime q ∧ n = p^a * q^b ∧ (a + 1) * (b + 1) = 8

end smallest_positive_integer_with_eight_factors_l507_507518


namespace quadratic_inequality_solutions_l507_507054

theorem quadratic_inequality_solutions (a x : ℝ) :
  (x^2 - (2+a)*x + 2*a > 0) → (
    (a < 2  → (x < a ∨ x > 2)) ∧
    (a = 2  → (x ≠ 2)) ∧
    (a > 2  → (x < 2 ∨ x > a))
  ) :=
by sorry

end quadratic_inequality_solutions_l507_507054


namespace arithmetic_mean_eq_l507_507905

theorem arithmetic_mean_eq :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = (67 : ℚ) / 144 :=
by
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  have h₁ : a = 3 / 8 := by rfl
  have h₂ : b = 5 / 9 := by rfl
  sorry

end arithmetic_mean_eq_l507_507905


namespace smallest_integer_with_eight_factors_l507_507522

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, n = 24 ∧
  (∀ d : ℕ, d ∣ n → d > 0) ∧
  ((∃ p : ℕ, prime p ∧ n = p^7) ∨
   (∃ p q : ℕ, prime p ∧ prime q ∧ n = p^3 * q) ∨
   (∃ p q r : ℕ, prime p ∧ prime q ∧ prime r ∧ n = p * q * r)) ∧
  (∀ m : ℕ, (∀ d : ℕ, d ∣ m → d > 0) → 
           ((∃ p : ℕ, prime p ∧ m = p^7 ∨ m = p^3 * q ∨ m = p * q * r) → 
            m ≥ n)) := by
  sorry

end smallest_integer_with_eight_factors_l507_507522


namespace polynomial_evaluation_l507_507389

def f (x : ℝ) : ℝ := sorry

theorem polynomial_evaluation (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x^2 + 1) = x^4 + 6 * x^2 + 2) :
  f (x^2 - 3) = x^4 - 2 * x^2 - 7 :=
sorry

end polynomial_evaluation_l507_507389


namespace fixed_monthly_fee_l507_507643

theorem fixed_monthly_fee (x y z : ℝ) 
  (h1 : x + y = 18.50) 
  (h2 : x + y + 3 * z = 23.45) : 
  x = 7.42 := 
by 
  sorry

end fixed_monthly_fee_l507_507643


namespace max_area_of_equilateral_triangle_in_rectangle_l507_507804

noncomputable def maxEquilateralTriangleArea (a b : ℝ) : ℝ :=
  if h : a ≤ b then
    (a^2 * Real.sqrt 3) / 4
  else
    (b^2 * Real.sqrt 3) / 4

theorem max_area_of_equilateral_triangle_in_rectangle :
  maxEquilateralTriangleArea 12 14 = 36 * Real.sqrt 3 :=
by
  sorry

end max_area_of_equilateral_triangle_in_rectangle_l507_507804


namespace total_digits_used_l507_507113

theorem total_digits_used (n : ℕ) (h : n = 2500) : 
  let first_n_even := (finset.range (2 * n + 1)).filter (λ x, x % 2 = 0)
  let count_digits := λ x, if x < 10 then 1 else if x < 100 then 2 else if x < 1000 then 3 else 4
  let total_digits := first_n_even.sum (λ x, count_digits x)
  total_digits = 9444 :=
by sorry

end total_digits_used_l507_507113


namespace geometric_sequence_sum_value_l507_507867

variable (n : ℕ) (t : ℝ)
def S_n (n : ℕ) := 3^n + t

theorem geometric_sequence_sum_value (h1 : S_n 1 = 3 + t) (h2 : S_n 2 - S_n 1 = 6) (h3 : S_n 3 - S_n 2 = 18) :
  t + (S_n 3 - S_n 2) = 17 := by
  sorry

-- here h1, h2, h3 represent conditions from the problem
-- we need to prove that under these conditions, the result t + a_3 = 17 holds

end geometric_sequence_sum_value_l507_507867


namespace fraction_of_phone_numbers_l507_507632

theorem fraction_of_phone_numbers (b a : ℕ) (h_total : b = 7 * 10^6) (h_a : a = 10^5) :
  (a : ℚ) / b = 1 / 70 :=
by
  rw [h_total, h_a],
  norm_cast,
  field_simp [show (70 : ℕ) ≠ 0, by norm_num],
  sorry

end fraction_of_phone_numbers_l507_507632


namespace remainder_of_euclidean_division_l507_507394

variable {R : Type*} [CommRing R]

theorem remainder_of_euclidean_division (P : Polynomial R) (a : R) :
  ∃ Q R : Polynomial R, P = (Polynomial.X - Polynomial.C a)^2 * Q + R ∧ 
    R.degree < 2 ∧ 
    R.eval a = P.eval a ∧ 
    R.derivative.eval a = P.derivative.eval a :=
by
  let Q := (P - (Polynomial.C (P.eval a) + Polynomial.C (P.derivative.eval a) * (Polynomial.X - Polynomial.C a))) / (Polynomial.X - Polynomial.C a)^2
  let R := Polynomial.C (P.eval a) + Polynomial.C (P.derivative.eval a) * (Polynomial.X - Polynomial.C a)
  use [Q, R]
  split
  sorry
  split
  sorry
  split
  { show R.eval a = P.eval a, sorry }
  { show R.derivative.eval a = P.derivative.eval a, sorry }

end remainder_of_euclidean_division_l507_507394


namespace total_digits_2500_is_9449_l507_507132

def nth_even (n : ℕ) : ℕ := 2 * n

def count_digits_in_range (start : ℕ) (stop : ℕ) : ℕ :=
  (stop - start) / 2 + 1

def total_digits (n : ℕ) : ℕ :=
  let one_digit := 4
  let two_digit := count_digits_in_range 10 98
  let three_digit := count_digits_in_range 100 998
  let four_digit := count_digits_in_range 1000 4998
  let five_digit := 1
  one_digit * 1 +
  two_digit * 2 +
  (three_digit * 3) +
  (four_digit * 4) +
  (five_digit * 5)

theorem total_digits_2500_is_9449 : total_digits 2500 = 9449 := by
  sorry

end total_digits_2500_is_9449_l507_507132


namespace smallest_positive_integer_with_eight_factors_l507_507521

theorem smallest_positive_integer_with_eight_factors :
  ∃ n : ℕ, (∀ m : ℕ, (m < n ∧ (∀ d : ℕ, d | m → d = 1 ∨ d = m) → (∃ a b : ℕ, distinct_factors_count m a b ∧ a = 8)) → n = 24) :=
by
  sorry

def distinct_factors_count (n : ℕ) (a b : ℕ) : Prop :=
  ∃ (p q : ℕ), prime p ∧ prime q ∧ n = p^a * q^b ∧ (a + 1) * (b + 1) = 8

end smallest_positive_integer_with_eight_factors_l507_507521


namespace part_a_part_b_l507_507150

-- Part (a): Proof statement.
theorem part_a (weights: Fin 10 → ℕ) :
  (∀ n, weights n ∈ {1, 1, 2, 3, 5, 8, 13, 21, 34, 55}) →
  (∀ k ∈ Finset.range 56, ∃ A : Finset (Fin 10), A.card = 9 → k = (A.sum weights)) :=
begin
  sorry
end

-- Part (b): Proof statement.
theorem part_b (weights: Fin 12 → ℕ) :
  (∀ n, weights n ∈ {1, 1, 2, 3, 4, 6, 9, 13, 19, 28, 41}) →
  (∀ k ∈ Finset.range 56, ∃ A : Finset (Fin 12), A.card = 10 → k = (A.sum weights)) :=
begin
  sorry
end

end part_a_part_b_l507_507150


namespace find_C_l507_507621

theorem find_C (A B C : ℝ) 
  (h1 : A + B + C = 600) 
  (h2 : A + C = 250) 
  (h3 : B + C = 450) : 
  C = 100 :=
sorry

end find_C_l507_507621


namespace smallest_six_consecutive_number_exists_max_value_N_perfect_square_l507_507167

-- Definition of 'six-consecutive numbers'
def is_six_consecutive (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧
  b ≠ d ∧ c ≠ d ∧ (a + b) * (c + d) = 60

-- Definition of the function F
def F (a b c d : ℕ) : ℤ :=
  let p := (10 * a + c) - (10 * b + d)
  let q := (10 * a + d) - (10 * b + c)
  q - p

-- Exists statement for the smallest six-consecutive number
theorem smallest_six_consecutive_number_exists :
  ∃ (a b c d : ℕ), is_six_consecutive a b c d ∧ (1000 * a + 100 * b + 10 * c + d) = 1369 := 
sorry

-- Exists statement for the maximum N such that F(N) is perfect square
theorem max_value_N_perfect_square :
  ∃ (a b c d : ℕ), is_six_consecutive a b c d ∧ 
  (1000 * a + 100 * b + 10 * c + d) = 9613 ∧
  ∃ (k : ℤ), F a b c d = k ^ 2 := 
sorry

end smallest_six_consecutive_number_exists_max_value_N_perfect_square_l507_507167


namespace monotonicity_of_f_existence_of_a_l507_507698

noncomputable def f : ℝ → ℝ := sorry
 
axiom condition1 {x y : ℝ} : f(x) + f(y) = f(x + y) + 3
axiom condition2 : f(3) = 6
axiom condition3 (x : ℝ) : x > 0 → f(x) > 3

theorem monotonicity_of_f :
  monotone f :=
begin
  sorry
end

theorem existence_of_a :
  ∃ a : ℝ, f(a^2 - a - 5) < 4 ∧ -2 < a ∧ a < 3 :=
begin
  sorry
end

end monotonicity_of_f_existence_of_a_l507_507698


namespace sin_cubed_identity_l507_507656

theorem sin_cubed_identity (c d : ℝ) :
  (∀ θ : ℝ, sin θ ^ 3 = c * sin (3 * θ) + d * sin θ) ↔ (c = -1 / 4 ∧ d = 3 / 4) := by
  sorry

end sin_cubed_identity_l507_507656


namespace failed_in_english_is_48_percent_l507_507769

-- Definitions
def failedHindi : ℝ := 25 / 100
def failedBoth : ℝ := 27 / 100
def passedBoth : ℝ := 54 / 100
def passedAtLeastOneSubject := 1 - passedBoth

-- Proof statement
theorem failed_in_english_is_48_percent :
  (failedHindi + (E / 100) - failedBoth) + passedBoth = 1 →
  (E / 100) = 48 / 100 :=
by
  intro h
  sorry

end failed_in_english_is_48_percent_l507_507769


namespace smallest_integer_with_eight_factors_l507_507524

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, n = 24 ∧
  (∀ d : ℕ, d ∣ n → d > 0) ∧
  ((∃ p : ℕ, prime p ∧ n = p^7) ∨
   (∃ p q : ℕ, prime p ∧ prime q ∧ n = p^3 * q) ∨
   (∃ p q r : ℕ, prime p ∧ prime q ∧ prime r ∧ n = p * q * r)) ∧
  (∀ m : ℕ, (∀ d : ℕ, d ∣ m → d > 0) → 
           ((∃ p : ℕ, prime p ∧ m = p^7 ∨ m = p^3 * q ∨ m = p * q * r) → 
            m ≥ n)) := by
  sorry

end smallest_integer_with_eight_factors_l507_507524


namespace number_of_handshakes_l507_507680

theorem number_of_handshakes (n : ℕ) (h1 : n = 5) : (nat.choose n 2) = 10 := by
  rw h1
  norm_num

end number_of_handshakes_l507_507680


namespace tallest_building_height_l507_507868

theorem tallest_building_height :
  ∃ H : ℝ, H + (1/2) * H + (1/4) * H + (1/20) * H = 180 ∧ H = 100 := by
  sorry

end tallest_building_height_l507_507868


namespace max_value_of_f_l507_507854

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) + 6 * cos (π / 2 - x)

theorem max_value_of_f : ∃ y ∈ Icc (-1 : ℝ) 1, f y = 5 :=
by
  sorry

end max_value_of_f_l507_507854


namespace difference_of_squares_l507_507377

theorem difference_of_squares (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∃ x y : ℤ, a = x^2 - y^2) ∨ 
  (∃ x y : ℤ, b = x^2 - y^2) ∨ 
  (∃ x y : ℤ, a + b = x^2 - y^2) :=
by
  sorry

end difference_of_squares_l507_507377


namespace path_of_sum_l507_507073

-- Define chessboard and adjacency
def is_adjacent (a b : ℕ × ℕ) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 + 1 = b.2)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 + 1 = b.1))

-- Define condition of placing integers on the chessboard
def valid_chessboard (n : ℕ) (f : ℕ × ℕ → ℕ) : Prop :=
  (∀ (x : ℕ × ℕ), 1 ≤ f x ∧ f x ≤ n^2) ∧
  (∀ i j i' j', (f (i, j)) ≠ (f (i', j')))

-- Main theorem statement
theorem path_of_sum (n : ℕ) (f : ℕ × ℕ → ℕ)
  (h : valid_chessboard n f) :
  ∃ (path : list (ℕ × ℕ)),
  path.length = 2 * n - 1 ∧
  (∀ i, i < path.length - 1 → is_adjacent (path.nth_le i sorry) (path.nth_le (i+1) sorry)) ∧
  path.head = (0, 0) ∧
  path.last sorry = (n - 1, n - 1) ∧
  ((path.map f).sum ≥ (⌊n^3 / 2⌋ + n^2 - n + 1)) :=
sorry

end path_of_sum_l507_507073


namespace no_primes_in_sequence_l507_507568

def P : ℕ := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47 * 53 * 59 * 61

theorem no_primes_in_sequence :
  ∀ n : ℕ, 2 ≤ n ∧ n ≤ 59 → ¬ Nat.Prime (P + n) :=
by
  sorry

end no_primes_in_sequence_l507_507568


namespace smallest_number_with_eight_factors_l507_507529

-- Definition of a function to count the number of distinct positive factors
def count_distinct_factors (n : ℕ) : ℕ := (List.range n).filter (fun d => d > 0 ∧ n % d = 0).length

-- Statement to prove the main problem
theorem smallest_number_with_eight_factors (n : ℕ) :
  count_distinct_factors n = 8 → n = 24 :=
by
  sorry

end smallest_number_with_eight_factors_l507_507529


namespace arithmetic_mean_of_fractions_l507_507918

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 := 
by 
  sorry

end arithmetic_mean_of_fractions_l507_507918


namespace unique_lines_for_p_l507_507828

-- Definition of the problem statement and conditions
def intersecting_lines (p : ℕ) : ℕ :=
  if p = 2 then 3 else 0

-- Theorem stating the equivalent math proof problem
theorem unique_lines_for_p {p : ℕ} (h : p = 2) : intersecting_lines p = 3 :=
by {
  rw h,
  simp [intersecting_lines],
  sorry
}

end unique_lines_for_p_l507_507828


namespace permutations_of_six_attractions_is_720_l507_507364

-- Define the number of attractions
def num_attractions : ℕ := 6

-- State the theorem to be proven
theorem permutations_of_six_attractions_is_720 : (num_attractions.factorial = 720) :=
by {
  sorry
}

end permutations_of_six_attractions_is_720_l507_507364


namespace math_problem_l507_507211

theorem math_problem :
  (-1) ^ 2023 - Real.sqrt 9 + abs (1 - Real.sqrt 2) - Real.cbrt (-8) = Real.sqrt 2 - 3 :=
by 
  sorry

end math_problem_l507_507211


namespace triple_layer_area_l507_507494

theorem triple_layer_area (A B C X Y : ℕ) 
  (h1 : A + B + C = 204) 
  (h2 : 140 = (A + B + C) - X - 2 * Y + X + Y)
  (h3 : X = 24) : 
  Y = 64 := by
  sorry

end triple_layer_area_l507_507494


namespace sqrt_inequality_l507_507029

theorem sqrt_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b + b * c + c * a = 1) :
  sqrt (a + 1 / a) + sqrt (b + 1 / b) + sqrt (c + 1 / c) ≥ 2 * (sqrt a + sqrt b + sqrt c) :=
sorry

end sqrt_inequality_l507_507029


namespace arithmetic_mean_of_fractions_l507_507927

theorem arithmetic_mean_of_fractions (a b : ℚ) (h1 : a = 3 / 8) (h2 : b = 5 / 9) :
  (a + b) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_fractions_l507_507927


namespace heather_walked_distance_l507_507440

theorem heather_walked_distance :
  ∀ (D H S : ℝ) (t : ℝ),
    D = 30 ∧ H = 5 ∧ S = 6 ∧ (H * t + S * (t + 0.4)) = D →
    H * t = 12.55 :=
begin
  intros D H S t h,
  sorry
end

end heather_walked_distance_l507_507440


namespace sum_of_numbers_odd_probability_l507_507561

namespace ProbabilityProblem

/-- 
  Given a biased die where the probability of rolling an even number is 
  twice the probability of rolling an odd number, and rolling the die three times,
  the probability that the sum of the numbers rolled is odd is 13/27.
-/
theorem sum_of_numbers_odd_probability :
  let p_odd := 1 / 3
  let p_even := 2 / 3
  let prob_all_odd := (p_odd) ^ 3
  let prob_one_odd_two_even := 3 * (p_odd) * (p_even) ^ 2
  prob_all_odd + prob_one_odd_two_even = 13 / 27 :=
by
  sorry

end sum_of_numbers_odd_probability_l507_507561


namespace total_distance_walked_l507_507199

-- Define the given conditions
def walks_to_work_days := 5
def walks_dog_days := 7
def walks_to_friend_days := 1
def walks_to_store_days := 2

def distance_to_work := 6
def distance_dog_walk := 2
def distance_to_friend := 1
def distance_to_store := 3

-- The proof statement
theorem total_distance_walked :
  (walks_to_work_days * (distance_to_work * 2)) +
  (walks_dog_days * (distance_dog_walk * 2)) +
  (walks_to_friend_days * distance_to_friend) +
  (walks_to_store_days * distance_to_store) = 95 := 
sorry

end total_distance_walked_l507_507199


namespace find_t_value_l507_507087

theorem find_t_value (x y z t : ℕ) (hx : x = 1) (hy : y = 2) (hz : z = 3) (hpos_x : 0 < x) (hpos_y : 0 < y) (hpos_z : 0 < z) :
  x + y + z + t = 10 → t = 4 :=
by
  -- Proof goes here
  sorry

end find_t_value_l507_507087


namespace inscribed_sphere_volume_l507_507487

-- Definitions
def edge_length : ℝ := 4
def radius : ℝ := edge_length / 2

-- Statement
theorem inscribed_sphere_volume : (4 / 3) * Real.pi * radius^3 = (32 / 3) * Real.pi :=
by
  sorry

end inscribed_sphere_volume_l507_507487


namespace sum_of_fractions_l507_507207

theorem sum_of_fractions : 
  (2/100) + (5/1000) + (5/10000) + 3 * (4/1000) = 0.0375 := 
by 
  sorry

end sum_of_fractions_l507_507207


namespace total_batteries_produced_l507_507591

def time_to_gather_materials : ℕ := 6 -- in minutes
def time_to_create_battery : ℕ := 9   -- in minutes
def num_robots : ℕ := 10
def total_time : ℕ := 5 * 60 -- in minutes (5 hours * 60 minutes/hour)

theorem total_batteries_produced :
  total_time / (time_to_gather_materials + time_to_create_battery) * num_robots = 200 :=
by
  -- Placeholder for the proof steps
  sorry

end total_batteries_produced_l507_507591


namespace sum_sin_squared_l507_507397

/-- 
Let S be the set of all real values of x for 0 < x < π / 4 such that sin x, cos x, 
and tan x form the side lengths (in some order) of a right triangle. 
Prove that the sum of all values of sin^2 x over all x in S is 1/2.
-/
theorem sum_sin_squared (S : Set ℝ) (hS : ∀ x ∈ S, 0 < x ∧ x < π / 4 ∧ 
  (∃ (a b c : ℝ), (Set.Perm (λ y, y = a ∨ y = b ∨ y = c) [sin x, cos x, tan x]) ∧ (a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2 ∨ c^2 = a^2 + b^2))) : 
  ∑ x in S, sin^2 x = 1/2 := 
by 
  sorry

end sum_sin_squared_l507_507397


namespace quadratic_solution_l507_507470

-- Definitions come from the conditions of the problem
def satisfies_equation (y : ℝ) : Prop := 6 * y^2 + 2 = 4 * y + 12

-- Statement of the proof
theorem quadratic_solution (y : ℝ) (hy : satisfies_equation y) : (12 * y - 2)^2 = 324 ∨ (12 * y - 2)^2 = 196 := 
sorry

end quadratic_solution_l507_507470


namespace frog_jump_probability_l507_507999

noncomputable def jump_length : ℝ := 1
noncomputable def num_jumps : ℕ := 5
noncomputable def max_distance : ℝ := Real.sqrt 2

-- Definition to model the frog's jumps
noncomputable def is_within_distance (pos : ℝ × ℝ) : Prop :=
  Real.sqrt (pos.1 ^ 2 + pos.2 ^ 2) ≤ max_distance

-- The Main Theorem stating the problem
theorem frog_jump_probability : 
  let jumps := List.replicate num_jumps jump_length in
  (IndependentlyRandomJumps.jumps_probability jumps is_within_distance) = (2 / 5) :=
sorry

end frog_jump_probability_l507_507999


namespace geometric_progression_solution_l507_507860

theorem geometric_progression_solution 
  (b₁ q : ℝ)
  (h₁ : b₁^3 * q^3 = 1728)
  (h₂ : b₁ * (1 + q + q^2) = 63) :
  (b₁ = 3 ∧ q = 4) ∨ (b₁ = 48 ∧ q = 1/4) :=
  sorry

end geometric_progression_solution_l507_507860


namespace PQ_div_AQ_eq_sqrt2_l507_507707

theorem PQ_div_AQ_eq_sqrt2
  (A B C D P Q : Type)
  (circle_Q : circle Q)
  (AB CD : line)
  (AB_diameter : circle_Q.is_diameter AB)
  (CD_diameter : circle_Q.is_diameter CD)
  (perp : AB ⊥ CD)
  (P_on_ext_AQ : ∃ R : line, P ∈ R ∧ AQ.extends_to R)
  (angle_CPQ_45 : ∡ CPQ = 45) :
  PQ.length / AQ.length = sqrt 2 :=
sorry

end PQ_div_AQ_eq_sqrt2_l507_507707


namespace find_original_six_digit_number_l507_507589

theorem find_original_six_digit_number (N x y : ℕ) (h1 : N = 10 * x + y) (h2 : N - x = 654321) (h3 : 0 ≤ y ∧ y ≤ 9) :
  N = 727023 :=
sorry

end find_original_six_digit_number_l507_507589


namespace car_speed_50_l507_507634

-- Define the conditions in Lean
def car (gallons_per_mile : ℝ) (full_tank_gallons : ℝ) : Type :=
{ gasoline_per_mile := gallons_per_mile,
  full_tank := full_tank_gallons }

def travel (car : car 1/30 20) (hours : ℝ) (fraction_tank_used : ℝ) : Prop :=
∀ speed : ℝ, hours = 5 ∧ fraction_tank_used = 0.4166666666666667 → speed = 50

-- Problem statement in Lean
theorem car_speed_50 : travel (car 1/30 20) 5 0.4166666666666667 := by
  sorry

end car_speed_50_l507_507634


namespace cube_faces_sum_l507_507218

theorem cube_faces_sum (n : ℤ) (h_even : n % 2 = 0) :
  let nums := [n, n + 1, n + 2, n + 3, n + 4, n + 5] in
  nums.sum = 27 := by
  sorry

end cube_faces_sum_l507_507218


namespace arithmetic_mean_of_fractions_l507_507940

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 3/8
  let b : ℚ := 5/9
  (a + b) / 2 = 67 / 144 := 
by
  sorry

end arithmetic_mean_of_fractions_l507_507940


namespace sum_of_digits_of_largest_five_digit_number_with_product_120_l507_507154

theorem sum_of_digits_of_largest_five_digit_number_with_product_120 
  (a b c d e : ℕ)
  (h_digit_a : 0 ≤ a ∧ a ≤ 9)
  (h_digit_b : 0 ≤ b ∧ b ≤ 9)
  (h_digit_c : 0 ≤ c ∧ c ≤ 9)
  (h_digit_d : 0 ≤ d ∧ d ≤ 9)
  (h_digit_e : 0 ≤ e ∧ e ≤ 9)
  (h_product : a * b * c * d * e = 120)
  (h_largest : ∀ f g h i j : ℕ, 
                0 ≤ f ∧ f ≤ 9 → 
                0 ≤ g ∧ g ≤ 9 → 
                0 ≤ h ∧ h ≤ 9 → 
                0 ≤ i ∧ i ≤ 9 → 
                0 ≤ j ∧ j ≤ 9 → 
                f * g * h * i * j = 120 → 
                f * 10000 + g * 1000 + h * 100 + i * 10 + j ≤ a * 10000 + b * 1000 + c * 100 + d * 10 + e) :
  a + b + c + d + e = 18 :=
by sorry

end sum_of_digits_of_largest_five_digit_number_with_product_120_l507_507154


namespace same_color_triangle_in_painted_pyramid_l507_507870

theorem same_color_triangle_in_painted_pyramid :
  ∀ (V : Type) (A_1 A_2 A_3 A_4 A_5 A_6 A_7 A_8 A_9 : V),
    (∀ i j, i ≠ j → {0, 1}) →
    (∀ i, {0, 1}) →
    ∃ T : (V × V × V), 
      ((color_of_edge T.1.1 T.1.2 = color_of_edge T.1.2 T.2) ∧ (color_of_edge T.1.2 T.2 = color_of_edge T.2 T.1.1)) :=
begin
  sorry
end

end same_color_triangle_in_painted_pyramid_l507_507870


namespace isosceles_triangle_angle_l507_507627

theorem isosceles_triangle_angle (A B C : ℝ) (h_iso : A = C)
  (h_obtuse : B = 1.4 * 90) (h_sum : A + B + C = 180) :
  A = 27 :=
by
  have h1 : B = 126 from h_obtuse
  have h2 : A + C = 54 := by linarith [h1, h_sum]
  have h3 : 2 * A = 54 := by linarith [h_iso, h2]
  exact eq_div_of_mul_eq two_ne_zero h3

end isosceles_triangle_angle_l507_507627


namespace part1_part2_l507_507762

/-- Given that in triangle ABC, sides opposite to angles A, B, and C are a, b, and c respectively
     if 2a sin B = sqrt(3) b and A is an acute angle, then A = 60 degrees. -/
theorem part1 {a b : ℝ} {A B : ℝ} (h1 : 2 * a * Real.sin B = Real.sqrt 3 * b)
  (h2 : 0 < A ∧ A < Real.pi / 2) : A = Real.pi / 3 :=
sorry

/-- Given that in triangle ABC, sides opposite to angles A, B, and C are a, b, and c respectively
     if b = 5, c = sqrt(5), and cos C = 9 / 10, then a = 4 or a = 5. -/
theorem part2 {a b c : ℝ} {C : ℝ} (h1 : b = 5) (h2 : c = Real.sqrt 5) 
  (h3 : Real.cos C = 9 / 10) : a = 4 ∨ a = 5 :=
sorry

end part1_part2_l507_507762


namespace sum_log_exp_eval_l507_507665

theorem sum_log_exp_eval :
  (∑ k in Finset.range 15 + 1, Real.logb (7 ^ k) (2 ^ (k ^ 2)) * ∑ k in Finset.range 80 + 1, Real.logb (16 ^ k) (49 ^ k) = 2400 :=
sorry

end sum_log_exp_eval_l507_507665


namespace perimeter_of_square_l507_507612

-- Defining the square with area
structure Square where
  side_length : ℝ
  area : ℝ

-- Defining a constant square with given area 625
def givenSquare : Square := 
  { side_length := 25, -- will square root the area of 625
    area := 625 }

-- Defining the function to calculate the perimeter of the square
noncomputable def perimeter (s : Square) : ℝ :=
  4 * s.side_length

-- The theorem stating that the perimeter of the given square with area 625 is 100
theorem perimeter_of_square : perimeter givenSquare = 100 := 
sorry

end perimeter_of_square_l507_507612


namespace smallest_integer_with_eight_factors_l507_507536

theorem smallest_integer_with_eight_factors : ∃ N : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) ∧ ∀ M : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) → N ≤ M :=
sorry -- Proof not provided.

end smallest_integer_with_eight_factors_l507_507536


namespace conjugate_modulus_converse_conjugate_modulus_inverse_conjugate_modulus_contrapositive_conjugate_modulus_l507_507071

open Complex

theorem conjugate_modulus (z1 z2 : ℂ) (h : z2 = conj z1) : |z1| = |z2| :=
begin
  rw [abs_conj, h]
end

theorem converse_conjugate_modulus (z1 z2 : ℂ) (h : |z1| = |z2|) : z1 = conj z2 :=
sorry -- Here we add sorry since the converse is actually false as per the solution.

theorem inverse_conjugate_modulus (z1 z2 : ℂ) (h : |z1| ≠ |z2|) : z2 ≠ conj z1 :=
sorry -- Here we add sorry since the inverse is actually false as per the solution.

theorem contrapositive_conjugate_modulus (z1 z2 : ℂ) (h : |z1| ≠ |z2|) : z2 ≠ conj z1 :=
begin
  rw [←abs_conj z1, h],
  assume hc : z2 = conj z1,
  rw [abs_conj, hc] at h,
  exact h rfl,
end

end conjugate_modulus_converse_conjugate_modulus_inverse_conjugate_modulus_contrapositive_conjugate_modulus_l507_507071


namespace probability_all_male_l507_507689

theorem probability_all_male (m n : ℕ) (h : (m + n) ≥ 3) 
  (h_prob : 4 / 5 = (1 - (m choose 3 / (m + n) choose 3))): 
  ((m choose 3) / ((m + n) choose 3)) = 1 / 5 :=
by
  sorry

end probability_all_male_l507_507689


namespace group_cost_possible_ticket_prices_l507_507880

variables {h m c : ℕ}

-- Conditions
def condition1 := 2 * h + 4 * m + 2 * c = 226
def condition2 := 3 * h + 3 * m + c = 207

-- Problem statements
theorem group_cost : condition1 → condition2 → 8 * h + 10 * m + 4 * c = 640 :=
sorry

theorem possible_ticket_prices : condition1 → condition2 → {n : ℕ | 25 < n ∧ n < 47}.card = 21 :=
sorry

end group_cost_possible_ticket_prices_l507_507880


namespace trig_inequality_probability_l507_507426

theorem trig_inequality_probability :
  let event_probability := ∫ x in set.Icc (0 : ℝ) real.pi, if (real.sin x + real.cos x > real.sqrt 6 / 2) then 1 else 0,
  in event_probability / (real.pi - 0) = 1 / 3 := sorry

end trig_inequality_probability_l507_507426


namespace f_f_4_eq_half_l507_507068

def f : ℝ → ℝ :=
λ x, if |x| ≤ 1 then real.sqrt x else 1 / x

theorem f_f_4_eq_half : f (f 4) = 1 / 2 := 
by sorry

end f_f_4_eq_half_l507_507068


namespace loom_weaving_rate_l507_507992

-- Define the time taken to weave 15 meters of cloth
def time_to_weave := 116.27906976744185

-- Define the total cloth woven in meters
def total_cloth := 15

-- Given the above definitions, we aim to show the rate of weaving in meters per second
theorem loom_weaving_rate :
  (total_cloth / time_to_weave) ≈ 0.129 :=
by
  sorry

end loom_weaving_rate_l507_507992


namespace union_of_sets_l507_507985

theorem union_of_sets (A B : Set ℕ) (hA : A = {1, 3}) (hB : B = {0, 3}) : 
  A ∪ B = {0, 1, 3} := 
by
  rw [hA, hB]
  rfl

end union_of_sets_l507_507985


namespace smallest_number_with_eight_factors_l507_507530

-- Definition of a function to count the number of distinct positive factors
def count_distinct_factors (n : ℕ) : ℕ := (List.range n).filter (fun d => d > 0 ∧ n % d = 0).length

-- Statement to prove the main problem
theorem smallest_number_with_eight_factors (n : ℕ) :
  count_distinct_factors n = 8 → n = 24 :=
by
  sorry

end smallest_number_with_eight_factors_l507_507530


namespace problem1_l507_507151

/-- Problem 1: Given the formula \( S = vt + \frac{1}{2}at^2 \) and the conditions
  when \( t=1, S=4 \) and \( t=2, S=10 \), prove that when \( t=3 \), \( S=18 \). -/
theorem problem1 (v a t S: ℝ) 
  (h₁ : t = 1 → S = 4 → S = v * t + 1 / 2 * a * t^2)
  (h₂ : t = 2 → S = 10 → S = v * t + 1 / 2 * a * t^2):
  t = 3 → S = v * t + 1 / 2 * a * t^2 → S = 18 := by
  sorry

end problem1_l507_507151


namespace not_all_roots_in_pair_C_l507_507834

theorem not_all_roots_in_pair_C : 
  let roots := { x | x^2 - 3 * x = 0 }, 
  let pairC_roots := { x | x = 2 * x } in
  {0, 3} = roots → ¬(roots ⊆ pairC_roots) :=
by
  let roots := { x | x^2 - 3 * x = 0 },
  let pairC_roots := { x | x = 2 * x },
  sorry

end not_all_roots_in_pair_C_l507_507834


namespace break_even_l507_507447

-- Define the initial investment, cost per T-shirt, and selling price per T-shirt.
def initial_investment : ℝ := 1500
def cost_per_tshirt : ℝ := 3
def selling_price_per_tshirt : ℝ := 20

-- Define the profit per T-shirt.
def profit_per_tshirt : ℝ := selling_price_per_tshirt - cost_per_tshirt

-- Define the break-even point calculation.
def break_even_tshirts (n : ℕ) : Prop := profit_per_tshirt * n ≥ initial_investment

-- State the theorem.
theorem break_even : ∃ n : ℕ, break_even_tshirts n ∧ ∀ m : ℕ, m < n → ¬ break_even_tshirts m :=
begin
  -- The proof can be given here, but we are required to skip it.
  sorry
end

end break_even_l507_507447


namespace find_center_numbers_l507_507192

def isConsecutive (a b : ℕ) : Bool :=
  abs (a - b) = 1

noncomputable def four_by_four_grid : List (List ℕ) := [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

theorem find_center_numbers :
  ∃ grid : List (List ℕ),
    -- all numbers from 1 to 16 are used without repetition
    (∀ x ∈ grid.flatten, 1 ≤ x ∧ x ≤ 16) ∧
    (∀ k, (count (grid.flatten) k = 1)) ∧
    -- consecutive numbers occupy adjacent squares
    (∀ i j : ℕ, i < 4 → j < 4 →
     (i < 3 → isConsecutive (grid.nthLe i j sorry) (grid.nthLe (i+1) j sorry) = true ∨
      j < 3 → isConsecutive (grid.nthLe i j sorry) (grid.nthLe i (j+1) sorry) = true)) ∧
    -- sum of corner numbers equals 34
    (grid.nthLe 0 0 sorry + grid.nthLe 0 3 sorry + grid.nthLe 3 0 sorry + grid.nthLe 3 3 sorry = 34) ∧
    -- center numbers are one of 9, 10, 11, or 12
    ∃ center_set : Set ℕ, center_set = {grid.nthLe 1 1 sorry, grid.nthLe 1 2 sorry, grid.nthLe 2 1 sorry,
      grid.nthLe 2 2 sorry} ∧ center_set ⊆ {9, 10, 11, 12} :=
sorry

end find_center_numbers_l507_507192


namespace y_intercepts_of_parabola_number_of_y_intercepts_l507_507735

theorem y_intercepts_of_parabola : ∀ y : ℝ, (∃ y : ℝ, 0 = 3 * y ^ 2 - 6 * y + 3) ↔ (∃ y : ℝ, (y - 1) ^ 2 = 0) :=
by
  sorry

theorem number_of_y_intercepts : 1 = finset.card (finset.filter (fun y => ∃ y : ℝ, 0 = 3 * y ^ 2 - 6 * y + 3) {y | true}) :=
by 
  sorry

end y_intercepts_of_parabola_number_of_y_intercepts_l507_507735


namespace wiper_draws_sector_l507_507488

theorem wiper_draws_sector :
  ∀ (wiper : Type) (draws_sector : wiper → Prop),
    (∃ (windshield : Type) (sector : windshield → Prop),
      sector windshield) →
    draws_sector wiper → 
    (∃ (line : Type) (surface : line → Prop),
      surface line) :=
by sorry

end wiper_draws_sector_l507_507488


namespace distance_between_planes_l507_507672

-- Define the two planes as functions
def plane1 : ℝ × ℝ × ℝ → ℝ := λ (x y z : ℝ), x + 2 * y - 2 * z + 1
def plane2 : ℝ × ℝ × ℝ → ℝ := λ (x y z : ℝ), 2 * x + 5 * y - 4 * z + 5

-- Define a function to calculate the distance from a point to a plane
noncomputable def point_to_plane_distance (a b c d : ℝ) (x₀ y₀ z₀ : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c * z₀ + d) / sqrt (a ^ 2 + b ^ 2 + c ^ 2)

-- A point on the first plane
def point_on_plane1 : ℝ × ℝ × ℝ := (-1, 0, 0)

-- Calculate the distance between the point and the second plane
def distance_point_plane2 : ℝ :=
  point_to_plane_distance 2 5 (-4) 5 (-1) 0 0

-- Prove the distance between the planes is 1/√5
theorem distance_between_planes : distance_point_plane2 = 1 / sqrt 5 :=
by {
  sorry
}

end distance_between_planes_l507_507672


namespace nonzero_terms_count_l507_507224

open Polynomial

noncomputable def poly1 := (C 2) * X^3 - C 4
noncomputable def poly2 := (C 3) * X^2 + (C 5) * X - C 7
noncomputable def poly3 := (C 5) * (X^4 - (C 3) * X^3 + (C 2) * X^2)

theorem nonzero_terms_count :
  let expanded_poly := poly1 * poly2 + poly3 in
  (expanded_poly.support.card = 6) :=
sorry

end nonzero_terms_count_l507_507224


namespace linear_bound_x_alpha_y_beta_l507_507683

def is_linearly_bounded (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (K : ℝ), 0 < K ∧ ∀ x y : ℝ, 0 < x → 0 < y → |f x y| < K * (x + y)

theorem linear_bound_x_alpha_y_beta (α β : ℝ) : 
  is_linearly_bounded (λ x y, x^α * y^β) ↔ (α + β = 1 ∧ 0 ≤ α ∧ 0 ≤ β) :=
sorry

end linear_bound_x_alpha_y_beta_l507_507683


namespace total_cost_of_shirts_l507_507681

theorem total_cost_of_shirts :
  let cost_shirt1 := 15
  let discount1 := 0.10
  let num_shirt1 := 3
  let cost_shirt2 := 20
  let tax2 := 0.05
  let num_shirt2 := 2
  let discounted_price1 := cost_shirt1 * (1 - discount1)
  let total_cost1 := discounted_price1 * num_shirt1
  let total_price2 := cost_shirt2 * (1 + tax2)
  let total_cost2 := total_price2 * num_shirt2
  total_cost1 + total_cost2 = 82.50 := by
  let cost_shirt1 := 15
  let discount1 := 0.10
  let num_shirt1 := 3
  let cost_shirt2 := 20
  let tax2 := 0.05
  let num_shirt2 := 2
  let discounted_price1 := cost_shirt1 * (1 - discount1)
  let total_cost1 := discounted_price1 * num_shirt1
  let total_price2 := cost_shirt2 * (1 + tax2)
  let total_cost2 := total_price2 * num_shirt2
  sorry

end total_cost_of_shirts_l507_507681


namespace six_people_mutual_known_l507_507572

/-!
Proof Problem:
Given:
1. There are 512 people at the meeting.
2. Under every six people, there is always at least two who know each other.
Prove:
There must be six people at this gathering who all mutually know each other.
-/

theorem six_people_mutual_known (n : ℕ) (h : n = 512)
  (H : ∀ s : Finset (Fin n), s.card = 6 → ∃ x y : Fin n, x ≠ y ∧ x ∈ s ∧ y ∈ s ∧ x.adj y) :
  ∃ t : Finset (Fin n), t.card = 6 ∧ ∀ x y : Fin n, x ∈ t ∧ y ∈ t → x.adj y := by
  sorry

end six_people_mutual_known_l507_507572


namespace Z_real_iff_Z_pure_imag_iff_Z_first_quadrant_iff_l507_507013

-- Define the complex number Z based on m
def Z (m : ℝ) : ℂ := complex.log (m^2 - 2*m - 2) + (m^2 + 3*m + 2 : ℝ) * complex.I

-- 1. Define the condition for Z being a real number
def is_real (Z : ℂ) : Prop := Z.im = 0

-- 2. Define the condition for Z being a pure imaginary number
def is_pure_imaginary (Z : ℂ) : Prop := Z.re = 0 ∧ Z.im ≠ 0

-- 3. Define the condition for Z being in the first quadrant of the complex plane
def is_first_quadrant (Z : ℂ) : Prop := Z.re > 0 ∧ Z.im > 0

-- Problem (1): Prove that Z is a real number iff m = -1 or m = -2
theorem Z_real_iff (m : ℝ) : is_real (Z m) ↔ m = -1 ∨ m = -2 := sorry

-- Problem (2): Prove that Z is a pure imaginary number iff m = 3 or m = -1
theorem Z_pure_imag_iff (m : ℝ) : is_pure_imaginary (Z m) ↔ m = 3 ∨ m = -1 := sorry

-- Problem (3): Prove that Z is in the first quadrant iff m < -2 or m > 3
theorem Z_first_quadrant_iff (m : ℝ) : is_first_quadrant (Z m) ↔ m < -2 ∨ m > 3 := sorry

end Z_real_iff_Z_pure_imag_iff_Z_first_quadrant_iff_l507_507013


namespace smallest_integer_with_eight_factors_l507_507523

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, n = 24 ∧
  (∀ d : ℕ, d ∣ n → d > 0) ∧
  ((∃ p : ℕ, prime p ∧ n = p^7) ∨
   (∃ p q : ℕ, prime p ∧ prime q ∧ n = p^3 * q) ∨
   (∃ p q r : ℕ, prime p ∧ prime q ∧ prime r ∧ n = p * q * r)) ∧
  (∀ m : ℕ, (∀ d : ℕ, d ∣ m → d > 0) → 
           ((∃ p : ℕ, prime p ∧ m = p^7 ∨ m = p^3 * q ∨ m = p * q * r) → 
            m ≥ n)) := by
  sorry

end smallest_integer_with_eight_factors_l507_507523


namespace unique_increasing_seq_l507_507849

noncomputable def unique_seq (a : ℕ → ℕ) (r : ℝ) : Prop :=
∀ (b : ℕ → ℕ), (∀ n, b n = 3 * n - 2 → ∑' n, r ^ (b n) = 1 / 2 ) → (∀ n, a n = b n)

theorem unique_increasing_seq {r : ℝ} 
  (hr : 0.4 < r ∧ r < 0.5) 
  (hc : r^3 + 2*r = 1):
  ∃ a : ℕ → ℕ, (∀ n, a n = 3 * n - 2) ∧ (∑'(n), r^(a n) = 1/2) ∧ unique_seq a r :=
by
  sorry

end unique_increasing_seq_l507_507849


namespace magnitude_of_vector_b_l507_507731

def vector_a : ℝ × ℝ × ℝ := (-1, 2, 1)

noncomputable def vector_b (x y : ℝ) : ℝ × ℝ × ℝ := (3, x, y)

def are_parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2, k * a.3)

theorem magnitude_of_vector_b (x y : ℝ) 
  (h_parallel : are_parallel vector_a (vector_b x y)) :
  ‖(3, x, y)‖ = 3 * real.sqrt 6 :=
sorry

end magnitude_of_vector_b_l507_507731


namespace infinitely_many_terms_odd_l507_507800

theorem infinitely_many_terms_odd (n : ℕ) (h : n > 1) : ∃^∞ k, k > 1 ∧ odd (⌊ (n^k : ℝ) / k ⌋₊) :=
by
  sorry

end infinitely_many_terms_odd_l507_507800


namespace rachel_problems_solved_each_minute_l507_507425

-- Definitions and conditions
def problems_solved_each_minute (x : ℕ) : Prop :=
  let problems_before_bed := 12 * x
  let problems_at_lunch := 16
  let total_problems := problems_before_bed + problems_at_lunch
  total_problems = 76

-- Theorem to be proved
theorem rachel_problems_solved_each_minute : ∃ x : ℕ, problems_solved_each_minute x ∧ x = 5 :=
by
  sorry

end rachel_problems_solved_each_minute_l507_507425


namespace arithmetic_mean_eq_l507_507902

theorem arithmetic_mean_eq :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = (67 : ℚ) / 144 :=
by
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  have h₁ : a = 3 / 8 := by rfl
  have h₂ : b = 5 / 9 := by rfl
  sorry

end arithmetic_mean_eq_l507_507902


namespace arithmetic_mean_of_fractions_l507_507926

theorem arithmetic_mean_of_fractions (a b : ℚ) (h1 : a = 3 / 8) (h2 : b = 5 / 9) :
  (a + b) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_fractions_l507_507926


namespace platform_length_l507_507615

theorem platform_length (train_length : ℕ) (pole_time : ℕ) (platform_time : ℕ) (V : ℕ) (L : ℕ)
  (h_train_length : train_length = 500)
  (h_pole_time : pole_time = 50)
  (h_platform_time : platform_time = 100)
  (h_speed : V = train_length / pole_time)
  (h_platform_distance : V * platform_time = train_length + L) : 
  L = 500 := 
sorry

end platform_length_l507_507615


namespace power_mod_result_l507_507949

-- Define the modulus and base
def mod : ℕ := 8
def base : ℕ := 7
def exponent : ℕ := 202

-- State the theorem
theorem power_mod_result :
  (base ^ exponent) % mod = 1 :=
by
  sorry

end power_mod_result_l507_507949


namespace sqrt_inequality_l507_507028

theorem sqrt_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b + b * c + c * a = 1) :
  sqrt (a + 1 / a) + sqrt (b + 1 / b) + sqrt (c + 1 / c) ≥ 2 * (sqrt a + sqrt b + sqrt c) :=
sorry

end sqrt_inequality_l507_507028


namespace convex_ngon_in_square_l507_507165

noncomputable def area_triangle (a b c : (ℝ × ℝ)) : ℝ :=
  0.5 * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))

theorem convex_ngon_in_square
  (n : ℕ) 
  (n_pos : 3 ≤ n)
  (vertices : fin n → ℝ × ℝ)
  (inside_square : ∀ i, 0 ≤ (vertices i).1 ∧ (vertices i).1 ≤ 1 ∧ 0 ≤ (vertices i).2 ∧ (vertices i).2 ≤ 1)
  (convex : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → 
    let a := vertices i, b := vertices j, c := vertices k in 
    (b.1 - a.1) * (c.2 - a.2) ≠ (c.1 - a.1) * (b.2 - a.2) ) :
  (∃ (i j k : fin n), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ area_triangle (vertices i) (vertices j) (vertices k) ≤ 8 / (n * n)) ∧
  (∃ (i j k : fin n), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ area_triangle (vertices i) (vertices j) (vertices k) ≤ 16 * real.pi / (n * n * n)) :=
sorry

end convex_ngon_in_square_l507_507165


namespace smallest_integer_with_eight_factors_l507_507510

theorem smallest_integer_with_eight_factors:
  ∃ n : ℕ, ∀ (d : ℕ), d > 0 → d ∣ n → 8 = (divisor_count n) 
  ∧ (∀ m : ℕ, m > 0 → (∀ (d : ℕ), d > 0 → d ∣ m → 8 = (divisor_count m)) → n ≤ m) → n = 24 :=
begin
  sorry
end

end smallest_integer_with_eight_factors_l507_507510


namespace graph_crosses_x_axis_at_origin_l507_507650

-- Let g(x) be a quadratic function defined as ax^2 + bx
def g (a b x : ℝ) : ℝ := a * x^2 + b * x

-- Define the conditions a ≠ 0 and b ≠ 0
axiom a_ne_0 (a : ℝ) : a ≠ 0
axiom b_ne_0 (b : ℝ) : b ≠ 0

-- The problem statement
theorem graph_crosses_x_axis_at_origin (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  ∃ x : ℝ, g a b x = 0 ∧ ∀ x', g a b x' = 0 → x' = 0 ∨ x' = -b / a :=
sorry

end graph_crosses_x_axis_at_origin_l507_507650


namespace typist_present_salary_l507_507476

def original_salary := 2000
def raise_percentage := 0.10
def reduction_percentage := 0.05

theorem typist_present_salary :
  let raised_salary := original_salary * (1 + raise_percentage)
  let reduced_salary := raised_salary * (1 - reduction_percentage)
  reduced_salary = 2090 :=
by
  let raised_salary := original_salary * (1 + raise_percentage)
  let reduced_salary := raised_salary * (1 - reduction_percentage)
  show reduced_salary = 2090 from sorry

end typist_present_salary_l507_507476


namespace maximum_uncovered_sections_l507_507022

theorem maximum_uncovered_sections :
  ∀ (length_corridor tiles_total_length tile_coverage_per_tile number_of_tiles : ℕ), 
  length_corridor = 100 →
  tiles_total_length = 1000 →
  tile_coverage_per_tile = tiles_total_length / number_of_tiles →
  number_of_tiles = 20 →
  tile_coverage_per_tile = 50 →
  ∃ uncovered_sections : ℕ, uncovered_sections = 11 :=
begin
  intros length_corridor tiles_total_length tile_coverage_per_tile number_of_tiles 
          H_length_corridor
          H_tiles_total_length
          H_tile_coverage_per_tile
          H_number_of_tiles
          H_share_per_tile,
  use 11,
  sorry
end

end maximum_uncovered_sections_l507_507022


namespace total_digits_used_l507_507110

theorem total_digits_used (n : ℕ) (h : n = 2500) : 
  let first_n_even := (finset.range (2 * n + 1)).filter (λ x, x % 2 = 0)
  let count_digits := λ x, if x < 10 then 1 else if x < 100 then 2 else if x < 1000 then 3 else 4
  let total_digits := first_n_even.sum (λ x, count_digits x)
  total_digits = 9444 :=
by sorry

end total_digits_used_l507_507110


namespace johns_total_payment_l507_507369

theorem johns_total_payment :
  let silverware_cost := 20
  let dinner_plate_cost := 0.5 * silverware_cost
  let total_cost := dinner_plate_cost + silverware_cost
  total_cost = 30 := sorry

end johns_total_payment_l507_507369


namespace number_of_digits_of_2500_even_integers_l507_507128

theorem number_of_digits_of_2500_even_integers : 
  let even_integers := List.range (5000 : Nat) in
  let first_2500_even := List.filter (fun n => n % 2 = 0) even_integers in
  List.length (List.join (first_2500_even.map (fun n => n.toDigits Nat))) = 9448 :=
by
  sorry

end number_of_digits_of_2500_even_integers_l507_507128


namespace correct_statement_l507_507138

theorem correct_statement :
  1. (∃ x : ℝ, x = √(√(25)) ∧ x = 5) = false ∧ -- Arithmetic square root of √25 is not 5
  2. (∃ y : ℝ, y^2 = 9 ∧ y = 3) ∧            -- 3 is a square root of 9
  3. (∀ z : ℝ, z < 0 → ∃ w : ℝ, w^3 = z) ∧   -- Negative numbers have cube roots
  4. (∀ t : ℝ, t^3 = t ↔ t ∈ ({0, 1, -1} : set ℝ)) →  -- Cube root equals themselves for 0, 1, and -1
    true := 
by
  sorry

end correct_statement_l507_507138


namespace solution_to_functional_equation_l507_507653

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f(x) * f(y) * f(x - y) = x^2 * f(y) - y^2 * f(x)

theorem solution_to_functional_equation :
  (∀ x, f(x) = x) ∨
  (∀ x, f(x) = -x) ∨
  (∀ x, f(x) = 0) :=
sorry

end solution_to_functional_equation_l507_507653


namespace arithmetic_mean_of_fractions_l507_507943

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 3/8
  let b : ℚ := 5/9
  (a + b) / 2 = 67 / 144 := 
by
  sorry

end arithmetic_mean_of_fractions_l507_507943


namespace problem1_problem2_problem3_l507_507697

variables {a b : EuclideanSpace ℝ (Fin 3)}

-- Conditions
axiom mag_a : ‖a‖ = 4
axiom mag_b : ‖b‖ = 2
axiom angle_ab : real.angle.to_deg (real.inner_product_geometry.angle a b) = 120

-- Questions (converted to Lean 4 proofs)
theorem problem1 : ((a - 2 • b) ⬝ (a + b)) = 12 :=
by sorry

theorem problem2 : ‖(2 • a - b)‖ = 2 * real.sqrt 21 :=
by sorry

theorem problem3 : real.angle.to_deg (real.inner_product_geometry.angle a (a + b)) = 30 :=
by sorry

end problem1_problem2_problem3_l507_507697


namespace problem1_problem2_problem3_problem4_l507_507573

-- Problem 1
theorem problem1 : ∃ n : ℕ, n = 3^4 ∧ n = 81 :=
by
  sorry

-- Problem 2
theorem problem2 : ∃ n : ℕ, n = (Nat.choose 4 2) * 6 ∧ n = 36 :=
by
  sorry

-- Problem 3
theorem problem3 : ∃ n : ℕ, n = Nat.choose 4 2 ∧ n = 6 :=
by
  sorry

-- Problem 4
theorem problem4 : ∃ n : ℕ, n = 1 + (Nat.choose 4 1 + Nat.choose 4 2 / 2) + 6 ∧ n = 14 :=
by
  sorry

end problem1_problem2_problem3_problem4_l507_507573


namespace toothpick_count_l507_507497

theorem toothpick_count (height width : ℕ) (h_height : height = 20) (h_width : width = 10) : 
  (21 * width + 11 * height) = 430 :=
by
  sorry

end toothpick_count_l507_507497


namespace cardinals_in_park_l507_507682

theorem cardinals_in_park (total_birds : ℕ) (fraction_sparrows : ℚ) (fraction_robins_and_cardinals : ℚ) (sparrows_partition : fraction_sparrows = 5/6) (total_birds_in_park : total_birds = 120) : total_birds * (1 / 12) = 10 :=
by
  have fraction_not_sparrows : ℚ := 1 - fraction_sparrows
  have fraction_cardinals : ℚ := fraction_not_sparrows / 2
  have cardinals : ℚ := total_birds * fraction_cardinals
  rw [sparrows_partition, total_birds_in_park]
  simp [fraction_not_sparrows, fraction_cardinals, cardinals]
  norm_num
  sorry

end cardinals_in_park_l507_507682


namespace symmetric_lines_l507_507751

theorem symmetric_lines (l : ℝ → ℝ → Prop) :
  (∀ x y : ℝ, l x y ↔ x = y - 2) ↔
  (∀ x y : ℝ, ((x^2 + y^2 = 1) ∧ (x^2 + y^2 + 4*x - 4*y + 7 = 0)) →
    symmetric (x^2 + y^2) (x^2 + y^2 + 4*x - 4*y + 7) l) :=
by
  sorry

end symmetric_lines_l507_507751


namespace inequality_sqrt_l507_507033

theorem inequality_sqrt (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
by
  sorry

end inequality_sqrt_l507_507033


namespace ratio_of_perimeter_to_b_l507_507613

theorem ratio_of_perimeter_to_b (b : ℝ) (hb : b ≠ 0) :
  let p1 := (-2*b, -2*b)
  let p2 := (2*b, -2*b)
  let p3 := (2*b, 2*b)
  let p4 := (-2*b, 2*b)
  let l := (y = b * x)
  let d1 := 4*b
  let d2 := 4*b
  let d3 := 4*b
  let d4 := 4*b*Real.sqrt 2
  let perimeter := d1 + d2 + d3 + d4
  let ratio := perimeter / b
  ratio = 12 + 4 * Real.sqrt 2 := by
  -- Placeholder for proof
  sorry

end ratio_of_perimeter_to_b_l507_507613


namespace white_given_popped_l507_507155

-- Define the conditions
def white_kernels : ℚ := 1 / 2
def yellow_kernels : ℚ := 1 / 3
def blue_kernels : ℚ := 1 / 6

def white_kernels_pop : ℚ := 3 / 4
def yellow_kernels_pop : ℚ := 1 / 2
def blue_kernels_pop : ℚ := 1 / 3

def probability_white_popped : ℚ := white_kernels * white_kernels_pop
def probability_yellow_popped : ℚ := yellow_kernels * yellow_kernels_pop
def probability_blue_popped : ℚ := blue_kernels * blue_kernels_pop

def probability_popped : ℚ := probability_white_popped + probability_yellow_popped + probability_blue_popped

-- The theorem to be proved
theorem white_given_popped : (probability_white_popped / probability_popped) = (27 / 43) := 
by sorry

end white_given_popped_l507_507155


namespace forest_volume_estimation_l507_507190

-- Definitions for conditions
def n : ℕ := 10
def x : Fin n → ℝ := ![0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
def y : Fin n → ℝ := ![0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]

noncomputable def sum_x := 0.6
noncomputable def sum_y := 3.9
noncomputable def sum_x_sq := 0.038
noncomputable def sum_y_sq := 1.6158
noncomputable def sum_xy := 0.2474

-- Proof problem statement
theorem forest_volume_estimation :
  (∑ i, x i = sum_x) →
  (∑ i, y i = sum_y) →
  (∑ i, (x i) ^ 2 = sum_x_sq) →
  (∑ i, (y i) ^ 2 = sum_y_sq) →
  (∑ i, (x i) * (y i) = sum_xy) →
  (real.sqrt 1.896 = 1.377) →
  let x̄ := sum_x / n,
      ȳ := sum_y / n,
      r : ℝ := (sum_xy - n * x̄ * ȳ) / (real.sqrt ((sum_x_sq - n * x̄ ^ 2) * (sum_y_sq - n * ȳ ^ 2))),
      total_area := 186,
      total_volume := total_area * ȳ / x̄
  in x̄ = 0.06 ∧ ȳ = 0.39 ∧ |r - 0.97| < 0.01 ∧ total_volume = 1209 :=
by sorry

end forest_volume_estimation_l507_507190


namespace max_shaded_squares_no_adjacent_l507_507358

-- Define a grid consisting of 9 squares arranged in a 3x3 pattern
structure Grid :=
  (squares : Fin 9 → ℕ) 
  (adjacent : (Fin 9 → ℕ) → (Fin 9 → ℕ) → Bool)

-- Define a property on shading such that no two adjacent squares are shaded
def no_adjacent_shaded (g : Grid) : Prop :=
  ∀ i j, g.squares i = 1 → g.adjacent g.squares i g.squares j → g.squares j = 0

-- The main theorem that needs to be proved
theorem max_shaded_squares_no_adjacent : 
  ∀ g : Grid, no_adjacent_shaded g → (∑ i, g.squares i) ≤ 6 :=
by sorry

end max_shaded_squares_no_adjacent_l507_507358


namespace side_error_percentage_l507_507624

theorem side_error_percentage (S S' : ℝ) (h1: S' = S * Real.sqrt 1.0609) : 
  (S' / S - 1) * 100 = 3 :=
by
  sorry

end side_error_percentage_l507_507624


namespace sarah_wide_reflections_l507_507227

variables (tall_mirrors_sarah : ℕ) (tall_mirrors_ellie : ℕ) 
          (wide_mirrors_ellie : ℕ) (tall_count : ℕ) (wide_count : ℕ)
          (total_reflections : ℕ) (S : ℕ)

def reflections_in_tall_mirrors_sarah := 10 * tall_count
def reflections_in_tall_mirrors_ellie := 6 * tall_count
def reflections_in_wide_mirrors_ellie := 3 * wide_count
def total_reflections_no_wide_sarah := reflections_in_tall_mirrors_sarah + reflections_in_tall_mirrors_ellie + reflections_in_wide_mirrors_ellie

theorem sarah_wide_reflections :
  reflections_in_tall_mirrors_sarah = 30 →
  reflections_in_tall_mirrors_ellie = 18 →
  reflections_in_wide_mirrors_ellie = 15 →
  tall_count = 3 →
  wide_count = 5 →
  total_reflections = 88 →
  total_reflections = total_reflections_no_wide_sarah + 5 * S →
  S = 5 :=
sorry

end sarah_wide_reflections_l507_507227


namespace arithmetic_geometric_sequences_l507_507869

open function

def is_arithmetic_sequence (a_n : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a_n n = n * d + 3

def is_geometric_sequence (b_n : ℕ → ℕ) : Prop :=
  ∃ q, ∀ n, b_n n = 2 * q^n

def sum_first_n_terms (seq : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in range(n + 1), seq i

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 1
noncomputable def b_n (n : ℕ) : ℕ := 2^n

def S_n (n : ℕ) : ℕ :=
  sum_first_n_terms a_n n

def T_n (n : ℕ) : ℕ :=
  sum_first_n_terms (λ n, a_n n * b_n n) n

theorem arithmetic_geometric_sequences :
  (S_n 2) * (b_n 2) = 32 ∧
  (S_n 3) * (b_n 3) = 120 ∧
  ∀ n, (sum_first_n_terms (λ n, a_n n * b_n n) n) = (2 * n - 1) * 2^(n + 1) + 2 :=
begin
  sorry
end

end arithmetic_geometric_sequences_l507_507869


namespace functional_equation_solution_l507_507835

theorem functional_equation_solution {f : ℝ → ℝ} (h : ∀ (x y : ℝ), 0 < x → 0 < y → f(x * y + f(x)) = x * f(y) + 2):
  ∀ x : ℝ, 0 < x → f(x) = x + 1 :=
by
  sorry

end functional_equation_solution_l507_507835


namespace total_situps_l507_507306

theorem total_situps (hani_rate_increase : ℕ) (diana_situps : ℕ) (diana_rate : ℕ) : 
  hani_rate_increase = 3 →
  diana_situps = 40 → 
  diana_rate = 4 →
  let diana_time := diana_situps / diana_rate in
  let hani_rate := diana_rate + hani_rate_increase in
  let hani_situps := hani_rate * diana_time in
  diana_situps + hani_situps = 110 := 
by
  intro hani_rate_increase_is_three diana_situps_is_forty diana_rate_is_four
  let diana_time := diana_situps / diana_rate
  let hani_rate := diana_rate + hani_rate_increase
  let hani_situps := hani_rate * diana_time
  sorry

end total_situps_l507_507306


namespace percent_area_occupied_is_741_l507_507184

-- Define the conditions as mathematical facts
def square_side (s : ℝ) : Prop := s > 0
def rectangle_width (w : ℝ) (s : ℝ) : Prop := w = 3 * s
def rectangle_length (l : ℝ) (w : ℝ) : Prop := l = 1.5 * (3 * s) -- l = 4.5 * s

-- Given conditions
axiom width_length_ratio (s : ℝ) (w : ℝ) (l : ℝ) : square_side s → rectangle_width w s → rectangle_length l w → w / s = 3 ∧ l / w = 1.5

-- The total area occupied by the square in percentage
def percent_area_occupied (s w l : ℝ) : ℝ := (s * s) / (l * w) * 100

theorem percent_area_occupied_is_741 (s w l : ℝ) (h1 : square_side s) (h2 : rectangle_width w s) (h3 : rectangle_length l w) :
  width_length_ratio s w l h1 h2 h3 → percent_area_occupied s w l = 7.41 := 
by 
  -- Proof to be filled in
  sorry

end percent_area_occupied_is_741_l507_507184


namespace area_triangle_l507_507778

theorem area_triangle (A B C: ℝ) (AB AC : ℝ) (h1 : Real.sin A = 4 / 5) (h2 : AB * AC * Real.cos A = 6) :
  (1 / 2) * AB * AC * Real.sin A = 4 :=
by
  sorry

end area_triangle_l507_507778


namespace integral_exp_plus_x_l507_507230

theorem integral_exp_plus_x :
  ∫ x in 0..1, (Real.exp x + x) = Real.exp 1 - 1 / 2 :=
by
  sorry

end integral_exp_plus_x_l507_507230


namespace total_distance_walked_l507_507198

-- Define the given conditions
def walks_to_work_days := 5
def walks_dog_days := 7
def walks_to_friend_days := 1
def walks_to_store_days := 2

def distance_to_work := 6
def distance_dog_walk := 2
def distance_to_friend := 1
def distance_to_store := 3

-- The proof statement
theorem total_distance_walked :
  (walks_to_work_days * (distance_to_work * 2)) +
  (walks_dog_days * (distance_dog_walk * 2)) +
  (walks_to_friend_days * distance_to_friend) +
  (walks_to_store_days * distance_to_store) = 95 := 
sorry

end total_distance_walked_l507_507198


namespace dive_point_value_correct_l507_507336

-- Definitions
def degree_of_difficulty : ℝ := 3.2
def scores : List ℝ := [7.5, 8.8, 9.0, 6.0, 8.5]

-- The point value calculation
def point_value_of_dive (dd: ℝ) (scores: List ℝ) : ℝ :=
let sorted_scores := List.sort scores
let trimmed_scores := List.drop 1 (List.reverse (List.drop 1 sorted_scores))
let total := trimmed_scores.foldl (· + ·) 0
total * dd

-- Goal
theorem dive_point_value_correct : point_value_of_dive degree_of_difficulty scores = 79.36 := 
by
  sorry

end dive_point_value_correct_l507_507336


namespace remove_one_to_get_average_of_75_l507_507559

theorem remove_one_to_get_average_of_75 : 
  ∃ l : List ℕ, l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] ∧ 
  (∃ m : ℕ, List.erase l m = ([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] : List ℕ) ∧ 
  (12 : ℕ) = List.length (List.erase l m) ∧
  7.5 = ((List.sum (List.erase l m) : ℚ) / 12)) :=
sorry

end remove_one_to_get_average_of_75_l507_507559


namespace general_terms_and_sums_l507_507266

noncomputable def a_n (n : ℕ) := -2 * n + 21

noncomputable def S_n (n : ℕ) := 20 * n - n^2

noncomputable def b_n (n : ℕ) := -2 * n + 21 + 3^(n - 1)

noncomputable def T_n (n : ℕ) := -n^2 + 20 * n + (3^n - 1)/2

theorem general_terms_and_sums (n : ℕ) :
  (a_n n = -2 * n + 21) ∧
  (S_n n = 20 * n - n^2) ∧
  (b_n n = -2 * n + 21 + 3^(n - 1)) ∧
  (T_n n = -n^2 + 20 * n + (3^n - 1)/2) :=
by
  sorry

end general_terms_and_sums_l507_507266


namespace inverse_passes_through_3_4_l507_507276

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- Given that f(x) has an inverse
def has_inverse := ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- Given that y = f(x+1) passes through the point (3,3)
def condition := f (3 + 1) = 3

theorem inverse_passes_through_3_4 
  (h1 : has_inverse f) 
  (h2 : condition f) : 
  f⁻¹ 3 = 4 :=
sorry

end inverse_passes_through_3_4_l507_507276


namespace total_situps_l507_507307

theorem total_situps (hani_rate_increase : ℕ) (diana_situps : ℕ) (diana_rate : ℕ) : 
  hani_rate_increase = 3 →
  diana_situps = 40 → 
  diana_rate = 4 →
  let diana_time := diana_situps / diana_rate in
  let hani_rate := diana_rate + hani_rate_increase in
  let hani_situps := hani_rate * diana_time in
  diana_situps + hani_situps = 110 := 
by
  intro hani_rate_increase_is_three diana_situps_is_forty diana_rate_is_four
  let diana_time := diana_situps / diana_rate
  let hani_rate := diana_rate + hani_rate_increase
  let hani_situps := hani_rate * diana_time
  sorry

end total_situps_l507_507307


namespace max_free_vertices_l507_507352

theorem max_free_vertices (grid : matrix (fin 5) (fin 5) (fin 2)) :
  (∀ i j, grid i j < 2) → 
  (∀ v : fin 6 × fin 6, (∃ i j, (v = (i, j) ∨ v = (i.succ, j.succ)) ∧ grid i j = 1) 
    → 36 - (finset.card (finset.univ.filter (λ v', ∃ i j, (v' = (i, j) ∨ v' = (i.succ, j.succ)) ∧ grid i j = 1))) = 18 :=
  sorry

end max_free_vertices_l507_507352


namespace total_money_spent_l507_507370

def time_in_minutes_at_arcade : ℕ := 3 * 60
def cost_per_interval : ℕ := 50 -- in cents
def interval_duration : ℕ := 6 -- in minutes
def total_intervals : ℕ := time_in_minutes_at_arcade / interval_duration

theorem total_money_spent :
  ((total_intervals * cost_per_interval) = 1500) := 
by
  sorry

end total_money_spent_l507_507370


namespace smallest_integer_with_eight_factors_l507_507512

theorem smallest_integer_with_eight_factors:
  ∃ n : ℕ, ∀ (d : ℕ), d > 0 → d ∣ n → 8 = (divisor_count n) 
  ∧ (∀ m : ℕ, m > 0 → (∀ (d : ℕ), d > 0 → d ∣ m → 8 = (divisor_count m)) → n ≤ m) → n = 24 :=
begin
  sorry
end

end smallest_integer_with_eight_factors_l507_507512


namespace find_larger_number_l507_507019

-- Define the problem conditions
variable (x y : ℕ)
hypothesis h1 : y = x + 10
hypothesis h2 : x + y = 34

-- Formalize the goal to prove
theorem find_larger_number : y = 22 := by
  -- placeholders in lean statement to skip the proof
  sorry

end find_larger_number_l507_507019


namespace loss_percentage_correct_l507_507172

-- Mathematical definitions from the given problem
def cost_price : ℝ := 1600
def selling_price : ℝ := 1280

-- Definition of loss and percentage loss
def loss (CP SP : ℝ) : ℝ := CP - SP
def percentage_loss (CP loss_amt : ℝ) : ℝ := (loss_amt / CP) * 100

-- The theorem to be proved
theorem loss_percentage_correct : percentage_loss cost_price (loss cost_price selling_price) = 20 :=
by
  -- skipping the proof here
  sorry

end loss_percentage_correct_l507_507172


namespace primes_remainder_mod_6_sum_primes_less_than_166338_adjusted_sum_primes_less_than_166000_l507_507571

theorem primes_remainder_mod_6 (p : Nat) (hprime : Prime p) (hgt3 : p > 3) : p % 6 = 1 ∨ p % 6 = 5 :=
by
  sorry

theorem sum_primes_less_than_166338 : ∑ p in (Finset.filter Prime (Finset.range 1000)), if p > 1 then p else 0 < 166338 :=
by
  sorry

theorem adjusted_sum_primes_less_than_166000 : 
  ∑ p in (Finset.filter Prime (Finset.range 1000)), if p > 1 ∧ ¬(p = 25 ∨ p = 35 ∨ p = 55 ∨ p = 65 ∨ p = 85 ∨ p = 95) then p else 0 < 166000 
:= by 
  sorry

end primes_remainder_mod_6_sum_primes_less_than_166338_adjusted_sum_primes_less_than_166000_l507_507571


namespace problem_proof_l507_507722

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
  if x ≤ 1 then (1 / 3) ^ x else log x / log (1 / 2)

-- Prove that f(f(sqrt 2)) = sqrt 3
theorem problem_proof : f (f (Real.sqrt 2)) = Real.sqrt 3 :=
by
  sorry

end problem_proof_l507_507722


namespace linear_system_solution_l507_507140

theorem linear_system_solution (x y : ℝ) (h1 : x = 2) (h2 : y = -3) : x + y = -1 :=
by
  sorry

end linear_system_solution_l507_507140


namespace angle_BOD_correct_l507_507357

-- Define the conditions given in the problem
def perpendiculary (u v : Type) [inner_product_space ℝ u] := ∀ a b : u, ⟪a, b⟫ = 0

variables {O A B C D : Type} [euclidean_space ℝ O] [euclidean_space ℝ A] 
  [euclidean_space ℝ B] [euclidean_space ℝ C] [euclidean_space ℝ D]

variable (y : ℝ) -- the angle AOC in degrees
variable condition1 : perpendiculary O A
variable condition2 : perpendiculary O B
variable condition3 : perpendiculary O C
variable condition4 : perpendiculary O D
variable angle_AOB : 90
variable angle_COD : 90
variable eq1 : 4 * y = (y + 180 - 2 * y)

-- The angle BOD's measure based on the given conditions
def angle_BOD := 4 * y

-- Prove the equivalence
theorem angle_BOD_correct (h : y = 36) : angle_BOD y = 144 :=
by
  rw [angle_BOD, h],
  norm_num,
  sorry

end angle_BOD_correct_l507_507357


namespace lcm_six_ten_fifteen_is_30_l507_507675

-- Define the numbers and their prime factorizations
def six := 6
def ten := 10
def fifteen := 15

noncomputable def lcm_six_ten_fifteen : ℕ :=
  Nat.lcm (Nat.lcm six ten) fifteen

-- The theorem to prove the LCM
theorem lcm_six_ten_fifteen_is_30 : lcm_six_ten_fifteen = 30 :=
  sorry

end lcm_six_ten_fifteen_is_30_l507_507675


namespace correct_article_choice_l507_507977

def specific_keyboard : Prop := ∃ (keyboard : Set), keyboard = "the one in both speakers' minds"
def general_computer : Prop := ∃ (computer : Set), computer = "any computer in general"

theorem correct_article_choice (h1 : specific_keyboard) (h2 : general_computer) : 
  ("the", "a") = ("the", "a") :=
by 
  sorry

end correct_article_choice_l507_507977


namespace monthly_interest_payment_l507_507195

theorem monthly_interest_payment (principal : ℝ) (annual_rate : ℝ) (months_in_year : ℝ) : 
  principal = 31200 → 
  annual_rate = 0.09 → 
  months_in_year = 12 → 
  (principal * annual_rate) / months_in_year = 234 := 
by 
  intros h_principal h_rate h_months
  rw [h_principal, h_rate, h_months]
  sorry

end monthly_interest_payment_l507_507195


namespace smallest_number_with_eight_factors_l507_507533

-- Definition of a function to count the number of distinct positive factors
def count_distinct_factors (n : ℕ) : ℕ := (List.range n).filter (fun d => d > 0 ∧ n % d = 0).length

-- Statement to prove the main problem
theorem smallest_number_with_eight_factors (n : ℕ) :
  count_distinct_factors n = 8 → n = 24 :=
by
  sorry

end smallest_number_with_eight_factors_l507_507533


namespace calculate_expression_l507_507213

theorem calculate_expression :
  |(-1 : ℝ)| + Real.sqrt 9 - (1 - Real.sqrt 3)^0 - (1/2)^(-1 : ℝ) = 1 :=
by
  sorry

end calculate_expression_l507_507213


namespace area_of_triangle_ADF_l507_507604

-- Define the regular hexagon with sides of length 2
structure Hexagon :=
  (side_length : ℝ)
  (is_regular : ∀ (A B C D E F : ℝ), 
    A = side_length ∧ B = side_length ∧ C = side_length ∧ D = side_length ∧ E = side_length ∧ F = side_length)

-- Assume hexagon ABCDEF where all sides are equal to 2
def ABCDEF := Hexagon.mk 2 (by sorry)

-- Define distance function (placeholder for actual distance calculation)
def distance (a b c d : ℝ) : ℝ := 4 

-- Define the angle function (placeholder for actual angle calculation)
def angle (sides : ℕ) : ℝ := 120

-- Define sine of the angle
def sine_120 : ℝ := let pi := Real.pi in Real.sin(2*pi/3)

-- Define function to find the area of triangle given side lengths and included angle
def triangle_area (a b angle: ℝ) : ℝ :=
  0.5 * a * b * Real.sin(angle)

-- Theorem that the area of triangle ADF is 4√3
theorem area_of_triangle_ADF : triangle_area (distance 2 2 2 2) (distance 2 2 2 2) (angle 2) = 4 * Real.sqrt(3) :=
  sorry

end area_of_triangle_ADF_l507_507604


namespace solve_quadratic_l507_507051

theorem solve_quadratic :
  ∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ (x = 2 ∨ x = -1) :=
by
  sorry

end solve_quadratic_l507_507051


namespace Daria_financial_result_l507_507981

noncomputable def convert_rubles_to_usd (rubles : ℝ) (selling_rate : ℝ) : ℝ :=
  rubles / selling_rate

noncomputable def calculate_usd_after_deposit (usd : ℝ) (annual_interest_rate : ℝ) (months : ℕ) : ℝ :=
  usd * (1 + (annual_interest_rate / 12) * months / 100)

noncomputable def convert_usd_to_rubles (usd : ℝ) (buying_rate : ℝ) : ℝ :=
  usd * buying_rate

def final_financial_result (initial_rubles : ℝ) (initial_selling_rate : ℝ) (annual_interest_rate : ℝ)
 (months : ℕ) (final_buying_rate : ℝ) : ℝ :=
  let usd := convert_rubles_to_usd initial_rubles initial_selling_rate
  let final_usd := calculate_usd_after_deposit usd annual_interest_rate months
  let final_rubles := convert_usd_to_rubles final_usd final_buying_rate
  initial_rubles - final_rubles

theorem Daria_financial_result :
  final_financial_result 60000 59.65 1.5 6 55.95 ≈ -3309 :=
by
  sorry

end Daria_financial_result_l507_507981


namespace f_range_and_boundedness_g_odd_and_bounded_g_upper_bound_l507_507169

noncomputable def f (x : ℝ) : ℝ := (1 / 4) ^ x + (1 / 2) ^ x - 1
noncomputable def g (x m : ℝ) : ℝ := (1 - m * 2 ^ x) / (1 + m * 2 ^ x)

theorem f_range_and_boundedness :
  ∀ x : ℝ, x < 0 → 1 < f x ∧ ¬(∃ M : ℝ, ∀ x : ℝ, x < 0 → |f x| ≤ M) :=
by sorry

theorem g_odd_and_bounded (x : ℝ) :
  g x 1 = -g (-x) 1 ∧ |g x 1| < 1 :=
by sorry

theorem g_upper_bound (m : ℝ) (hm : 0 < m ∧ m < 1 / 2) :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g x m ≤ (1 - m) / (1 + m) :=
by sorry

end f_range_and_boundedness_g_odd_and_bounded_g_upper_bound_l507_507169


namespace arithmetic_mean_of_fractions_l507_507899

theorem arithmetic_mean_of_fractions :
  (3/8 + 5/9) / 2 = 67 / 144 :=
by
  sorry

end arithmetic_mean_of_fractions_l507_507899


namespace min_translation_l507_507221

def det (a₁ a₂ b₁ b₂ : ℝ) : ℝ :=
  a₁ * b₂ - a₂ * b₁

def f (x : ℝ) : ℝ :=
  det (real.sqrt 3) (real.sin x) 1 (real.cos x)

theorem min_translation (t : ℝ) (ht : t > 0) :
  (∀ x, f (x + t) = f (-(x + t))) ↔ t = 5 * real.pi / 6 :=
by
  sorry

end min_translation_l507_507221


namespace sequence_a_general_term_l507_507256

noncomputable def sequence_a : ℕ → ℕ
| 0     := 0
| (n+1) := if h : n = 0 then 1 else sequence_a n + 2 * n + 1

theorem sequence_a_general_term (n : ℕ) : sequence_a (n+1) = (n+1) ^ 2 := by
  sorry

end sequence_a_general_term_l507_507256


namespace smallest_integer_with_eight_factors_l507_507551

theorem smallest_integer_with_eight_factors :
  ∃ n : ℕ, (∀ m : ℕ, (∃ k : ℕ, m has_factors k ∧ k = 8) → m ≥ n) ∧ n = 24 :=
by
  sorry

end smallest_integer_with_eight_factors_l507_507551


namespace distinct_prime_factors_bound_l507_507441

open Int

theorem distinct_prime_factors_bound
  (a b : ℕ)
  (h_gcd : Nat.prime_factors (gcd a b).natAbs.card = 8)
  (h_lcm : Nat.prime_factors (natAbs ((a * b) / gcd a b)).card = 32)
  (h_less : Nat.prime_factors a.card < Nat.prime_factors b.card) :
  Nat.prime_factors a.card ≤ 19 :=
by
  sorry

end distinct_prime_factors_bound_l507_507441


namespace magnitude_of_z_l507_507228

open Complex

noncomputable def z : ℂ := (4 / 5 : ℝ) - (3 : ℝ) * I

theorem magnitude_of_z : complex.abs z = real.sqrt 241 / 5 := 
by
  -- The proof would go here, but it's not required as per the instructions
  sorry

end magnitude_of_z_l507_507228


namespace total_digits_first_2500_even_integers_l507_507115

theorem total_digits_first_2500_even_integers :
  let even_nums := List.range' 2 5000 (λ n, 2*n)  -- List of the first 2500 even integers
  let one_digit_nums := even_nums.filter (λ n, n < 10)
  let two_digit_nums := even_nums.filter (λ n, 10 ≤ n ∧ n < 100)
  let three_digit_nums := even_nums.filter (λ n, 100 ≤ n ∧ n < 1000)
  let four_digit_nums := even_nums.filter (λ n, 1000 ≤ n ∧ n ≤ 5000)
  let sum_digits := one_digit_nums.length * 1 + two_digit_nums.length * 2 + three_digit_nums.length * 3 + four_digit_nums.length * 4
  in sum_digits = 9448 := by sorry

end total_digits_first_2500_even_integers_l507_507115


namespace part_a_part_b_part_c_l507_507141

-- Part a
theorem part_a (h : ∃ (L₁ L₂ L₃ L₄ : set ℕ), L₁.count = 3 ∧ L₂.count = 3 ∧ L₃.count = 3 ∧ L₄.count = 3): 
  ∃ ways, ways = 24 := 
sorry

-- Part b
theorem part_b (h : ∀ (P : set ℕ), P.count = 9 ∧  ∀ pt ∈ P, ∃ (L₁ L₂ L₃ : set ℕ), 
  L₁.count = 3 ∧ L₂.count = 3 ∧ L₃.count = 3 ∧ (P ∈ L₁ ∧ P ∈ L₂ ∧ P ∈ L₃)): 
∃ configs, configs = 108 :=
sorry

-- Part c
theorem part_c (h : ∃ (L : set (set ℕ)), ∀ l ∈ L, l.count = 3 ∧ L.count = 10 ∧ 
  ∀ (p : set ℕ), p.count = 10 ∧ (∀ (l : set ℕ), l ∈ L → p⊆ l)):
  ∃ configs, configs = 120 :=
sorry

end part_a_part_b_part_c_l507_507141


namespace trajectory_equation_of_P_minimal_area_of_quadrilateral_l507_507006

-- Defining the general conditions
def parabola_f: Type := { p : ℝ × ℝ // p.2^2 = 4 * p.1 }
def directrix_intersection: ℝ × ℝ := (-1, 0)
def focus: ℝ × ℝ := (1, 0)

-- The moving point P satisfies the vector equation
noncomputable def moving_point (A B: ℝ × ℝ) (E P: ℝ × ℝ): Prop :=
  (P.1 - E.1, P.2 - E.2) = ((B.1 - E.1) + (A.1 - E.1), (B.2 - E.2) + (A.2 - E.2))

-- Q1: Prove the trajectory equation of point P is y² = 4x - 12
theorem trajectory_equation_of_P (A B P E: ℝ × ℝ) (hA: parabola_f A) (hB: parabola_f B) (hE: E = directrix_intersection) (hFocus: focus = (1, 0)) (hP: moving_point A B E P) : P.2^2 = 4 * P.1 - 12 := 
sorry

-- Q2: Prove that the line l when the area of quadrilateral EAPB is minimal has the equation x - 1 = 0
noncomputable def line_through_focus_min_area (m: ℝ): Prop := 
  ∃ l: ℝ × ℝ → Prop, l = λ p, p.1 - m * p.2 - 1 = 0

theorem minimal_area_of_quadrilateral (m: ℝ) (A B P E: ℝ × ℝ) (hA: parabola_f A) (hB: parabola_f B) (hE: E = directrix_intersection) (hFocus: focus = (1, 0)) (hP: moving_point A B E P) : line_through_focus_min_area m → m = 0 ∧ (λ p : ℝ × ℝ, p.1 - 1 = 0) :=
sorry

end trajectory_equation_of_P_minimal_area_of_quadrilateral_l507_507006


namespace max_y_eq_arctan_sqrt6_div_12_l507_507805

noncomputable def z (θ : ℝ) : ℂ := 3 * complex.cos θ + 2 * complex.I * complex.sin θ

theorem max_y_eq_arctan_sqrt6_div_12 :
  ∀ θ ∈ Ioo 0 (π / 2),
  let y := θ - complex.arg (z θ) in
  y ≤ arctan (sqrt 6 / 12) := 
sorry

end max_y_eq_arctan_sqrt6_div_12_l507_507805


namespace percent_value_in_quarters_l507_507965

theorem percent_value_in_quarters (dimes quarters : ℕ) (dime_value quarter_value : ℕ) (dime_count quarter_count : ℕ) :
  dimes = 50 →
  quarters = 20 →
  dime_value = 10 →
  quarter_value = 25 →
  dime_count = dimes * dime_value →
  quarter_count = quarters * quarter_value →
  (quarter_count : ℚ) / (dime_count + quarter_count) * 100 = 50 :=
by
  intros
  sorry

end percent_value_in_quarters_l507_507965


namespace percentage_reduction_in_price_l507_507813

noncomputable def original_price_per_mango : ℝ := 416.67 / 125

noncomputable def original_num_mangoes : ℝ := 360 / original_price_per_mango

def additional_mangoes : ℝ := 12

noncomputable def new_num_mangoes : ℝ := original_num_mangoes + additional_mangoes

noncomputable def new_price_per_mango : ℝ := 360 / new_num_mangoes

noncomputable def percentage_reduction : ℝ := (original_price_per_mango - new_price_per_mango) / original_price_per_mango * 100

theorem percentage_reduction_in_price : percentage_reduction = 10 := by
  sorry

end percentage_reduction_in_price_l507_507813


namespace ordered_pair_sqrt_l507_507235

/-- Problem statement: Given positive integers a and b such that a < b, prove that:
sqrt (1 + sqrt (40 + 24 * sqrt 5)) = sqrt a + sqrt b, if (a, b) = (1, 6). -/
theorem ordered_pair_sqrt (a b : ℕ) (h1 : a = 1) (h2 : b = 6) (h3 : a < b) :
  Real.sqrt (1 + Real.sqrt (40 + 24 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b :=
by
  sorry -- The proof is not required in this task.

end ordered_pair_sqrt_l507_507235


namespace area_A1B1C1_eq_7_l507_507077

-- Definitions of the given problem
variables {A B C A1 B1 C1 : Type}
variable  (B_mid_AA1 : B = midpoint A A1)
variable  (C_mid_BB1 : C = midpoint B B1)
variable  (A_mid_CC1 : A = midpoint C C1)
variable  (area_ABC_eq_1 : area A B C = 1)

-- Proof problem to solve
theorem area_A1B1C1_eq_7
    (B_mid_AA1 : B = midpoint A A1)
    (C_mid_BB1 : C = midpoint B B1)
    (A_mid_CC1 : A = midpoint C C1)
    (area_ABC_eq_1 : area A B C = 1) :
  area A1 B1 C1 = 7 :=
sorry

end area_A1B1C1_eq_7_l507_507077


namespace findAlphaWhenBeta36_l507_507057

variable (α β k : ℝ)

-- Condition: alpha squared is inversely proportional to beta.
def invProp (α β : ℝ) := α^2 * β = k

-- Condition: α = 4 when β = 9
def initialCondition : Prop := invProp (4 : ℝ) (9 : ℝ) = (16 * 9 : ℝ)

-- Theorem: Find α when β = 36
theorem findAlphaWhenBeta36 (h : invProp α β) (hf : initialCondition k) : 
  invProp α 36 → (α = 2) ∨ (α = -2) := 
by
  sorry

end findAlphaWhenBeta36_l507_507057


namespace total_digits_2500_is_9449_l507_507129

def nth_even (n : ℕ) : ℕ := 2 * n

def count_digits_in_range (start : ℕ) (stop : ℕ) : ℕ :=
  (stop - start) / 2 + 1

def total_digits (n : ℕ) : ℕ :=
  let one_digit := 4
  let two_digit := count_digits_in_range 10 98
  let three_digit := count_digits_in_range 100 998
  let four_digit := count_digits_in_range 1000 4998
  let five_digit := 1
  one_digit * 1 +
  two_digit * 2 +
  (three_digit * 3) +
  (four_digit * 4) +
  (five_digit * 5)

theorem total_digits_2500_is_9449 : total_digits 2500 = 9449 := by
  sorry

end total_digits_2500_is_9449_l507_507129


namespace total_number_of_digits_l507_507123

-- Definitions based on identified conditions
def first2500EvenIntegers := {n : ℕ | n % 2 = 0 ∧ 1 ≤ n ∧ n ≤ 5000}

-- Theorem statement based on the equivalent proof problem
theorem total_number_of_digits : 
  (first2500EvenIntegers.count_digits = 9448) :=
sorry

end total_number_of_digits_l507_507123


namespace latus_rectum_of_parabola_l507_507673

theorem latus_rectum_of_parabola :
  (∃ p : ℝ, ∀ x y : ℝ, y = - (1 / 6) * x^2 → y = p ∧ p = 3 / 2) :=
sorry

end latus_rectum_of_parabola_l507_507673


namespace arithmetic_mean_of_fractions_l507_507923

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 := 
by 
  sorry

end arithmetic_mean_of_fractions_l507_507923


namespace calculate_expression_l507_507209

theorem calculate_expression : 
  (-1)^(2023) - Real.sqrt 9 + abs (1 - Real.sqrt 2) - Real.cbrt (-8) = Real.sqrt 2 - 3 := 
sorry

end calculate_expression_l507_507209


namespace equal_clubs_and_students_l507_507344

theorem equal_clubs_and_students (S C : ℕ) 
  (h1 : ∀ c : ℕ, c < C → ∃ (m : ℕ → Prop), (∃ p, m p ∧ p = 3))
  (h2 : ∀ s : ℕ, s < S → ∃ (n : ℕ → Prop), (∃ p, n p ∧ p = 3)) :
  S = C := 
by
  sorry

end equal_clubs_and_students_l507_507344


namespace first_year_digits_sum_to_seven_after_2010_l507_507505

theorem first_year_digits_sum_to_seven_after_2010 :
  ∃ (y : ℕ), y > 2010 ∧ (∀ n, 2010 < n ∧ n < y → (∑ d in (nat.digits 10 n), d) ≠ 7) ∧ (∑ d in (nat.digits 10 y), d) = 7 ∧ y = 2014 :=
sorry

end first_year_digits_sum_to_seven_after_2010_l507_507505


namespace quadratic_sum_l507_507475

theorem quadratic_sum (b c : ℝ) : 
  (∀ x : ℝ, x^2 - 24 * x + 50 = (x + b)^2 + c) → b + c = -106 :=
by
  intro h
  sorry

end quadratic_sum_l507_507475


namespace no_perfect_square_in_seq_l507_507376

noncomputable def seq : ℕ → ℕ
| 0       => 2
| 1       => 7
| (n + 2) => 4 * seq (n + 1) - seq n

theorem no_perfect_square_in_seq :
  ¬ ∃ (n : ℕ), ∃ (k : ℕ), (seq n) = k * k :=
sorry

end no_perfect_square_in_seq_l507_507376


namespace S_10_value_l507_507481

noncomputable def S (n : ℕ) (a : ℕ → ℕ) : ℕ := n * (a 1 + a n) / 2

theorem S_10_value (a : ℕ → ℕ) (h1 : a 2 = 3) (h2 : a 9 = 17) (h_arith : ∀ n, a (n + 1) = a n + (a 2 - a 1)) : 
  S 10 a = 100 := 
by
  sorry

end S_10_value_l507_507481


namespace Jane_remaining_time_l507_507962

noncomputable def JaneRate : ℚ := 1 / 4
noncomputable def RoyRate : ℚ := 1 / 5
noncomputable def workingTime : ℚ := 2
noncomputable def cakeFractionCompletedTogether : ℚ := (JaneRate + RoyRate) * workingTime
noncomputable def remainingCakeFraction : ℚ := 1 - cakeFractionCompletedTogether
noncomputable def timeForJaneToCompleteRemainingCake : ℚ := remainingCakeFraction / JaneRate

theorem Jane_remaining_time :
  timeForJaneToCompleteRemainingCake = 2 / 5 :=
by
  sorry

end Jane_remaining_time_l507_507962


namespace equal_clubs_and_students_l507_507343

theorem equal_clubs_and_students (S C : ℕ) 
  (h1 : ∀ c : ℕ, c < C → ∃ (m : ℕ → Prop), (∃ p, m p ∧ p = 3))
  (h2 : ∀ s : ℕ, s < S → ∃ (n : ℕ → Prop), (∃ p, n p ∧ p = 3)) :
  S = C := 
by
  sorry

end equal_clubs_and_students_l507_507343


namespace polynomial_quotient_sum_l507_507396

def polynomial_quotient_is_integer (n : ℤ) : Prop := 
  (8 * n^3 - 96 * n^2 + 360 * n - 400) % (2 * n - 7) = 0

noncomputable def sum_of_absolute_values_of_n : ℤ := 
  ∑ n in { n : ℤ | polynomial_quotient_is_integer n }.toFinset, |n|

theorem polynomial_quotient_sum (S : set ℤ) 
  (hS : ∀ n ∈ S, polynomial_quotient_is_integer n) : 
  ∑ n in S.toFinset, |n| = 50 :=
sorry

end polynomial_quotient_sum_l507_507396


namespace smallest_integer_with_eight_factors_l507_507527

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, n = 24 ∧
  (∀ d : ℕ, d ∣ n → d > 0) ∧
  ((∃ p : ℕ, prime p ∧ n = p^7) ∨
   (∃ p q : ℕ, prime p ∧ prime q ∧ n = p^3 * q) ∨
   (∃ p q r : ℕ, prime p ∧ prime q ∧ prime r ∧ n = p * q * r)) ∧
  (∀ m : ℕ, (∀ d : ℕ, d ∣ m → d > 0) → 
           ((∃ p : ℕ, prime p ∧ m = p^7 ∨ m = p^3 * q ∨ m = p * q * r) → 
            m ≥ n)) := by
  sorry

end smallest_integer_with_eight_factors_l507_507527


namespace cube_container_unoccupied_volume_l507_507788

theorem cube_container_unoccupied_volume :
  let side_length_container := 12 in
  let side_length_ice := 3 in
  
  let volume_container := side_length_container ^ 3 in
  let volume_water := (1 / 3) * volume_container in
  let volume_ice_cube := side_length_ice ^ 3 in
  let num_ice_cubes := 20 in
  let total_volume_ice := num_ice_cubes * volume_ice_cube in
  
  let volume_unoccupied := volume_container - (volume_water + total_volume_ice) in
  volume_unoccupied = 612 :=
by
  sorry

end cube_container_unoccupied_volume_l507_507788


namespace pages_read_each_night_l507_507964

def pages_per_night (total_pages : ℝ) (total_nights : ℝ) : ℝ := total_pages / total_nights

theorem pages_read_each_night : pages_per_night 1200 10 = 120 := 
by 
  unfold pages_per_night
  show 1200 / 10 = 120
  sorry

end pages_read_each_night_l507_507964


namespace minimum_positive_period_of_f_l507_507468

def f (x : ℝ) : ℝ :=
  let m11 := sin x + cos x
  let m12 := cos (π - x)
  let m21 := 2 * sin x
  let m22 := cos x - sin x
  m11 * m22 - m12 * m21

theorem minimum_positive_period_of_f :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ π) := 
sorry

end minimum_positive_period_of_f_l507_507468


namespace reciprocal_twice_l507_507362

theorem reciprocal_twice (x : ℝ) (h : x = 81) : (1 / (1 / x)) = 81 :=
by 
  rw h
  simp
  simp

end reciprocal_twice_l507_507362


namespace sum_of_radii_of_tangent_circle_l507_507994

def is_tangent (cx cy r : ℝ) (px py : ℝ) := 
  (cx - px)^2 + (cy - py)^2 = (r + 2)^2

theorem sum_of_radii_of_tangent_circle :
  ∃ r1 r2 : ℝ, 
  (r1 + r2 = 14) ∧
  ∀ r : ℝ, (r = r1 ∨ r = r2) →
  is_tangent r r r 5 0 ∧ r > 0 :=
begin
  sorry
end

end sum_of_radii_of_tangent_circle_l507_507994


namespace exists_good_number_with_2013_distinct_prime_factors_l507_507378

def is_good_number (n : ℕ) :=
  ∃ (m : ℕ) (a : Fin m → ℕ) (b : Fin m → ℕ), (∀ i, a i ∣ n ∧ (∃ j, a j = i + 1)) ∧ n = (Finset.univ.sum (λ i, (-1)^(b i) * (a i)))

theorem exists_good_number_with_2013_distinct_prime_factors :
  ∃ n, n.factorization.support.card = 2013 ∧ is_good_number n := 
sorry

end exists_good_number_with_2013_distinct_prime_factors_l507_507378


namespace total_number_of_digits_l507_507120

-- Definitions based on identified conditions
def first2500EvenIntegers := {n : ℕ | n % 2 = 0 ∧ 1 ≤ n ∧ n ≤ 5000}

-- Theorem statement based on the equivalent proof problem
theorem total_number_of_digits : 
  (first2500EvenIntegers.count_digits = 9448) :=
sorry

end total_number_of_digits_l507_507120


namespace max_consecutive_zhonghuan_l507_507324

-- Definition of a Zhonghuan number
def isZhonghuan (n : ℕ) : Prop :=
  ∃ m : ℕ, (∃ oddDivisors : ℕ, 2 ^ m = oddDivisors) ∧ countOddDivisors n = oddDivisors

-- Placeholder for the count of odd divisors function
noncomputable def countOddDivisors (n : ℕ) : ℕ := sorry

-- Proving the maximum value of n for consecutive Zhonghuan numbers is 17
theorem max_consecutive_zhonghuan : ∃ n : ℕ, (∀ k : ℕ, (∃ m : ℕ, k ≤ 17 ∧ k ≥ 1 → isZhonghuan k)) :=
begin
  sorry
end

end max_consecutive_zhonghuan_l507_507324


namespace pagoda_lanterns_l507_507353

-- Definitions
def top_layer_lanterns (a₁ : ℕ) : ℕ := a₁
def bottom_layer_lanterns (a₁ : ℕ) : ℕ := a₁ * 2^6
def sum_of_lanterns (a₁ : ℕ) : ℕ := (a₁ * (1 - 2^7)) / (1 - 2)
def total_lanterns : ℕ := 381
def layers : ℕ := 7
def common_ratio : ℕ := 2

-- Problem Statement
theorem pagoda_lanterns (a₁ : ℕ) (h : sum_of_lanterns a₁ = total_lanterns) : 
  top_layer_lanterns a₁ + bottom_layer_lanterns a₁ = 195 := sorry

end pagoda_lanterns_l507_507353


namespace solve_quadratic_l507_507052

theorem solve_quadratic :
  ∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ (x = 2 ∨ x = -1) :=
by
  sorry

end solve_quadratic_l507_507052


namespace profit_in_may_highest_monthly_profit_and_max_value_l507_507164

def f (x : ℕ) : ℕ :=
  if 1 ≤ x ∧ x ≤ 6 then 12 * x + 28 else 200 - 14 * x

theorem profit_in_may :
  f 5 = 88 :=
by sorry

theorem highest_monthly_profit_and_max_value :
  ∃ x, 1 ≤ x ∧ x ≤ 12 ∧ f x = 102 :=
by sorry

end profit_in_may_highest_monthly_profit_and_max_value_l507_507164


namespace smallest_period_f_value_phi_sin_2alpha_l507_507289

def f (x φ : ℝ) : ℝ := sin x * cos φ + cos x * sin φ

theorem smallest_period_f (φ : ℝ) (hφ : 0 < φ ∧ φ < π) :
  ∃ T > 0, ∀ x, f (x + T) φ = f x φ :=
by
  use 2 * π
  intros
  simp [f]
  sorry

theorem value_phi (h : ∀ x, f (2 * x + π / 4) φ = f (2 * (π / 6) + π / 4) φ) :
  φ = 11 * π / 12 :=
by
  sorry

theorem sin_2alpha (α : ℝ)
  (h : f (α - 2 * π / 3) (11 * π / 12) = sqrt 2 / 4) :
  sin (2 * α) = -3 / 4 :=
by
  have : f α (11 * π / 12) = (sqrt 2 / 2) * (sin α + cos α),
  { simp [f] },
  rw this at h,
  sorry

end smallest_period_f_value_phi_sin_2alpha_l507_507289


namespace find_b_num_days_worked_l507_507970

noncomputable def a_num_days_worked := 6
noncomputable def b_num_days_worked := 9  -- This is what we want to verify
noncomputable def c_num_days_worked := 4

noncomputable def c_daily_wage := 105
noncomputable def wage_ratio_a := 3
noncomputable def wage_ratio_b := 4
noncomputable def wage_ratio_c := 5

-- Helper to find daily wages for a and b given the ratio and c's wage
noncomputable def x := c_daily_wage / wage_ratio_c
noncomputable def a_daily_wage := wage_ratio_a * x
noncomputable def b_daily_wage := wage_ratio_b * x

-- Calculate total earnings
noncomputable def a_total_earning := a_num_days_worked * a_daily_wage
noncomputable def c_total_earning := c_num_days_worked * c_daily_wage
noncomputable def total_earning := 1554
noncomputable def b_total_earning := b_num_days_worked * b_daily_wage

theorem find_b_num_days_worked : total_earning = a_total_earning + b_total_earning + c_total_earning → b_num_days_worked = 9 := by
  sorry

end find_b_num_days_worked_l507_507970


namespace problem_solution_l507_507575

variables (A B C D E : Point) (circle : Circle)

-- Defining the conditions
def conditions : Prop :=
  Inscribed pentagon A B C D E circle ∧
  Length A B = 4 ∧ Length B C = 4 ∧ Length C D = 4 ∧ Length D E = 4 ∧ Length A E = 3

-- The claim (to be proven)
def claim : Prop :=
  (1 - cos (angle B)) * (1 - cos (angle A C E)) = 9 / 1024

-- The final theorem statement
theorem problem_solution (h : conditions) : claim :=
by
  sorry

end problem_solution_l507_507575


namespace problem_l507_507458

variables {A B C O P Q : Point}
variables {AB AC BC PB CQ : Length}
variables (triangle_isosceles : AB = AC)
variables (circle_tangent_AB_AC : S.Center = O ∧ tangent S AB ∧ tangent S AC)
variables (points_on_sides : P ∈ AB ∧ Q ∈ AC)
variables (segment_tangent_to_circle : tangent S PQ)

theorem problem (h1 : triangle_isosceles)
                 (h2 : circle_tangent_AB_AC)
                 (h3 : points_on_sides)
                 (h4 : segment_tangent_to_circle) :
  4 * PB * CQ = BC^2 :=
by sorry

end problem_l507_507458


namespace P_over_P_neg_one_eq_l507_507072

noncomputable def f (x : ℂ) : ℂ := x^2007 + 17*x^2006 + 1

-- Assuming distinct roots of f
def distinct_roots (r : Fin 2007 → ℂ) : Prop :=
  ∀ i j, i ≠ j → r i ≠ r j ∧ f (r i) = 0

-- Defining the polynomial P and its properties
def P (r : Fin 2007 → ℂ) (x : ℂ) : ℂ :=
  ∏ i, (x - (r i + 1 / r i))

theorem P_over_P_neg_one_eq {r : Fin 2007 → ℂ} (h_distinct : distinct_roots r) :
  P r 1 / P r (-1) = 289 / 259 :=
sorry

end P_over_P_neg_one_eq_l507_507072


namespace second_marble_orange_probability_l507_507636

noncomputable def prob_second_orange (BagX_red : ℕ) (BagX_green : ℕ)
                                     (BagY_orange : ℕ) (BagY_purple : ℕ)
                                     (BagZ_orange : ℕ) (BagZ_purple : ℕ) : ℚ :=
  let P_red_from_X := BagX_red / (BagX_red + BagX_green : ℚ)
  let P_green_from_X := BagX_green / (BagX_red + BagX_green : ℚ)
  let P_orange_from_Y := BagY_orange / (BagY_orange + BagY_purple : ℚ)
  let P_orange_from_Z := BagZ_orange / (BagZ_orange + BagZ_purple : ℚ)
  (P_red_from_X * P_orange_from_Y) + (P_green_from_X * P_orange_from_Z)

theorem second_marble_orange_probability : 
  prob_second_orange 5 3 7 5 4 6 = 247 / 480 := 
by sorry

end second_marble_orange_probability_l507_507636


namespace min_buses_needed_l507_507605

theorem min_buses_needed (n : ℕ) : 325 / 45 ≤ n ∧ n < 325 / 45 + 1 ↔ n = 8 :=
by
  sorry

end min_buses_needed_l507_507605


namespace solution_set_of_abs_x_gt_1_l507_507079

theorem solution_set_of_abs_x_gt_1 (x : ℝ) : |x| > 1 ↔ x > 1 ∨ x < -1 := 
sorry

end solution_set_of_abs_x_gt_1_l507_507079


namespace sector_area_proof_l507_507062

open Real

-- Set up the conditions
def theta1 := π / 3
def theta2 := 2 * π / 3
def rho := 4
def sector_area (r θ : ℝ) := 1 / 2 * r ^ 2 * θ

-- Define the theorem to prove
theorem sector_area_proof : sector_area rho (theta2 - theta1) = 8 / 3 * π := 
by sorry

end sector_area_proof_l507_507062


namespace race_problem_l507_507582

theorem race_problem (a_speed b_speed : ℕ) (A B : ℕ) (finish_dist : ℕ)
  (h1 : finish_dist = 3000)
  (h2 : A = finish_dist - 500)
  (h3 : B = finish_dist - 600)
  (h4 : A / a_speed = B / b_speed)
  (h5 : a_speed / b_speed = 25 / 24) :
  B - ((500 * b_speed) / a_speed) = 120 :=
by
  sorry

end race_problem_l507_507582


namespace sqrt_half_eq_sqrt2_div_2_l507_507845

theorem sqrt_half_eq_sqrt2_div_2 : real.sqrt 0.5 = real.sqrt 2 / 2 :=
by
  sorry

end sqrt_half_eq_sqrt2_div_2_l507_507845


namespace vertical_asymptotes_single_valued_l507_507687

theorem vertical_asymptotes_single_valued (k : ℝ) :
  (∀ x, x ≠ 6 ∨ x ≠ -3 → g x = (λ x, (x^2 + 3*x + k) / (x^2 - 3*x - 18)))
  → ( (k = -54 ∧ ¬ (k = 0)) ∨ (k = 0 ∧ ¬ (k = -54)) ) :=
sorry

def g (x k : ℝ) : ℝ := (x^2 + 3*x + k) / (x^2 - 3*x - 18)

end vertical_asymptotes_single_valued_l507_507687


namespace quadratic_root_ratio_eq_l507_507134

theorem quadratic_root_ratio_eq (k : ℝ) :
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ (x = 3 * y ∨ y = 3 * x) ∧ x + y = -10 ∧ x * y = k) → k = 18.75 := by
  sorry

end quadratic_root_ratio_eq_l507_507134


namespace x_squared_eq_y_squared_iff_x_eq_y_or_neg_y_x_squared_eq_y_squared_necessary_but_not_sufficient_l507_507847

theorem x_squared_eq_y_squared_iff_x_eq_y_or_neg_y (x y : ℝ) : 
  (x^2 = y^2) ↔ (x = y ∨ x = -y) := by
  sorry

theorem x_squared_eq_y_squared_necessary_but_not_sufficient (x y : ℝ) :
  (x^2 = y^2 → x = y) ↔ false := by
  sorry

end x_squared_eq_y_squared_iff_x_eq_y_or_neg_y_x_squared_eq_y_squared_necessary_but_not_sufficient_l507_507847


namespace problem_seat_benches_l507_507989

noncomputable def smallest_N := 18

theorem problem_seat_benches (N : ℕ) (h1 : ∑ i in finset.range N, 7 = 77 ∧ ∑ i in finset.range N, 11 = 77): N = smallest_N :=
by
  -- problem solution is omitted
  sorry

end problem_seat_benches_l507_507989


namespace digital_earth_correct_purposes_l507_507474

def Purpose : Type := String

def P1 : Purpose := "To deal with natural and social issues of the entire Earth using digital means."
def P2 : Purpose := "To maximize the utilization of natural resources."
def P3 : Purpose := "To conveniently obtain information about the Earth."
def P4 : Purpose := "To provide precise locations, directions of movement, and speeds of moving objects."

def correct_purposes : Set Purpose := {P1, P2, P3}

theorem digital_earth_correct_purposes :
  {P1, P2, P3} = correct_purposes :=
by 
  sorry

end digital_earth_correct_purposes_l507_507474


namespace amount_earned_from_each_family_l507_507862

theorem amount_earned_from_each_family
  (goal : ℕ) (earn_from_fifteen_families : ℕ) (additional_needed : ℕ) (three_families : ℕ) 
  (earn_from_three_families_total : ℕ) (per_family_earn : ℕ) :
  goal = 150 →
  earn_from_fifteen_families = 75 →
  additional_needed = 45 →
  three_families = 3 →
  earn_from_three_families_total = (goal - additional_needed) - earn_from_fifteen_families →
  per_family_earn = earn_from_three_families_total / three_families →
  per_family_earn = 10 :=
by
  sorry

end amount_earned_from_each_family_l507_507862


namespace arithmetic_mean_of_fractions_l507_507910

theorem arithmetic_mean_of_fractions : 
  (3 : ℚ) / 8 + (5 : ℚ) / 9 = 2 * (67 : ℚ) / 144 :=
by {
  rw [(show (3 : ℚ) / 8 + (5 : ℚ) / 9 = (27 : ℚ) / 72 + (40 : ℚ) / 72, by sorry),
      (show (27 : ℚ) / 72 + (40 : ℚ) / 72 = (67 : ℚ) / 72, by sorry),
      (show (2 : ℚ) * (67 : ℚ) / 144 = (67 : ℚ) / 72, by sorry)]
}

end arithmetic_mean_of_fractions_l507_507910


namespace find_angle_y_l507_507773

open Classical

variables {α : Type*} [linear_ordered_ring α]

-- Assume parallel lines m and n:
variables {m n t : ℝ} (hmn : m ∥ n)

-- Angles formed by transversal
variables {a b : ℝ} (ha : ∠a = 40) (hb : ∠b = 50)

-- Given the angle relationships across the transversal:
variable [add_comm_group α]

-- The Lean theorem statement
theorem find_angle_y
  (m n : α) (t : α) 
  (parallel_lines : m ∥ n)
  (angle_40 : ∠a = 40)
  (angle_50 : ∠b = 50)
  :  ∃ y, y = 40 := 
begin
  sorry -- proof goes here
end

end find_angle_y_l507_507773


namespace y_intercepts_parabola_l507_507738

theorem y_intercepts_parabola : 
  let f : ℝ → ℝ := λ y, 3 * y^2 - 6 * y + 3
  in (∀ y : ℝ, f y = 0 → y = 1) → f 1 = 0 := 
begin
sorry
end

end y_intercepts_parabola_l507_507738


namespace john_recreation_percent_l507_507790

def wages_last_week : ℝ
def percent_recreation_last_week : ℝ := 30 / 100
def percent_wages_less_this_week : ℝ := 25 / 100
def percent_recreation_less_this_week : ℝ := 50 / 100

theorem john_recreation_percent :
  let W := wages_last_week in
  let wages_this_week := W - percent_wages_less_this_week * W in
  let recreation_this_week := percent_recreation_less_this_week * (percent_recreation_last_week * W) in
  let expected_percent := 20 / 100 in
  recreation_this_week = expected_percent * wages_this_week := 
sorry

end john_recreation_percent_l507_507790


namespace prob_two_hits_is_correct_l507_507765

section ring_toss_game

def prob_hit_M : ℝ := 3 / 4
def prob_hit_N : ℝ := 2 / 3
def prob_miss_M : ℝ := 1 - prob_hit_M
def prob_miss_N : ℝ := 1 - prob_hit_N

def scenario1_prob : ℝ := prob_hit_M * (2 * prob_hit_N * prob_miss_N)
def scenario2_prob : ℝ := prob_miss_M * (prob_hit_N * prob_hit_N)

def prob_hit_two_times : ℝ := scenario1_prob + scenario2_prob

theorem prob_two_hits_is_correct : prob_hit_two_times = 4 / 9 := by
  sorry

end ring_toss_game

end prob_two_hits_is_correct_l507_507765


namespace arithmetic_mean_of_38_and_59_is_67_over_144_l507_507933

theorem arithmetic_mean_of_38_and_59_is_67_over_144 :
  (3 / 8 + 5 / 9) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_38_and_59_is_67_over_144_l507_507933


namespace number_of_green_hats_l507_507099

variables (B G : ℕ)

-- Given conditions as definitions
def totalHats : Prop := B + G = 85
def totalCost : Prop := 6 * B + 7 * G = 530

-- The statement we need to prove
theorem number_of_green_hats (h1 : totalHats B G) (h2 : totalCost B G) : G = 20 :=
sorry

end number_of_green_hats_l507_507099


namespace triangle_area_l507_507097

/-- The conditions of the lines forming the triangle. -/
def line1 (x : ℝ) : ℝ := -1 / 3 * x + 4
def line2 (x : ℝ) : ℝ := 3 * x - 6
def line3 (x : ℝ) (y : ℝ) : Prop := x + y = 8

/-- The points of intersection of the lines. -/
def A : ℝ × ℝ := (3, 3)
def B : ℝ × ℝ := (3.5, 4.5)
def C : ℝ × ℝ := (6, 2)

/-- The area of the triangle formed by the points A, B, and C is 2.5. -/
theorem triangle_area : 
  let x1 := A.1, y1 := A.2, x2 := B.1, y2 := B.2, x3 := C.1, y3 := C.2 in 
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) = 2.5 := by
  sorry

end triangle_area_l507_507097


namespace hexagon_segments_divide_into_regions_l507_507841

open Finset

/-- Given a regular hexagon, and drawing all its sides and diagonals,
the total number of distinct regions formed is 24. -/
theorem hexagon_segments_divide_into_regions :
  let vertices := fin 6
  let hexagon : SimpleGraph vertices := {
    adj := λ i j, i ≠ j ∧ (abs (i - j) ≠ 1 ∧ abs (i - j) ≠ 5)
  }
  let sides := hexagon.edge_set.card = 6
  let diagonals := (vertices.card.choose 2) - sides = 9
  in sides ∧ diagonals → num_regions hexagon = 24 :=
sorry

end hexagon_segments_divide_into_regions_l507_507841


namespace identity_proof_l507_507659

theorem identity_proof :
  ∀ (x : ℝ), 
    x ≠ 2 →
    (x^2 + x + 1) ≠ 0 →
    ((x + 3) ^ 2 / ((x - 2) * (x^2 + x + 1)) = 
     (25 / 7) / (x - 2) + (-18 / 7 * x - 19 / 7) / (x^2 + x + 1)) :=
by
  intro x
  intros hx1 hx2
  -- proof goes here
  sorry

end identity_proof_l507_507659


namespace caterer_ordered_ice_cream_bars_l507_507160

theorem caterer_ordered_ice_cream_bars :
  ∀ (x : ℕ), (0.60 * x + 1.2 * x = 225) → x = 125 := 
by
  intros x h
  have h_eq : 1.8 * x = 225 := by linarith
  have h_sol : x = 125 := by
    apply eq_of_mul_eq_mul_right _ h_eq
    linarith
  exact h_sol

end caterer_ordered_ice_cream_bars_l507_507160


namespace find_b_when_a_is_6_l507_507486

-- Declare variables
variables (a b C : ℝ) (h_inv_prop : a * b = C) (h_sum : a + b = 30) (h_diff : a - b = 8)

-- Provide the leading proof
theorem find_b_when_a_is_6 : b = 209 / 6 :=
by
  -- Since 'a' and 'b' are inversely proportional, and 'a' equals to 6,
  -- we derive 'b' from the equation a * b = C.
  have inv_prop := h_inv_prop,
  have c_value : C = 209 := sorry,
  have a_value : a = 6 := sorry,
  show b = 209 / 6,
  sorry

end find_b_when_a_is_6_l507_507486


namespace solutions_to_cubic_eq_l507_507242

theorem solutions_to_cubic_eq (z : ℂ) :
  (z = -3 ∨ z = (3 + 3 * complex.I * real.sqrt 3) / 2 ∨ z = (3 - 3 * complex.I * real.sqrt 3) / 2)
  ↔ z^3 = -27 :=
by
  sorry

end solutions_to_cubic_eq_l507_507242


namespace sum_of_squares_le_sum_of_squares_sum_of_cubes_le_weighted_sum_of_squares_l507_507010

theorem sum_of_squares_le_sum_of_squares {n : ℕ} {a b : ℕ → ℝ}
  (h1 : ∀ i j, 1 ≤ i → i ≤ j → j ≤ n → a i ≥ a j ∧ a n > 0)
  (h2 : ∀ k, 1 ≤ k → k ≤ n → ∑ i in Finset.range k, a i ≤ ∑ i in Finset.range k, b i) :
  ∑ i in Finset.range n, (a i)^2 ≤ ∑ i in Finset.range n, (b i)^2 :=
sorry

theorem sum_of_cubes_le_weighted_sum_of_squares {n : ℕ} {a b : ℕ → ℝ}
  (h1 : ∀ i j, 1 ≤ i → i ≤ j → j ≤ n → a i ≥ a j ∧ a n > 0)
  (h2 : ∀ k, 1 ≤ k → k ≤ n → ∑ i in Finset.range k, a i ≤ ∑ i in Finset.range k, b i) :
  ∑ i in Finset.range n, (a i)^3 ≤ ∑ i in Finset.range n, a i * (b i)^2 :=
sorry

end sum_of_squares_le_sum_of_squares_sum_of_cubes_le_weighted_sum_of_squares_l507_507010


namespace volume_PABCD_l507_507825

noncomputable def volume_of_pyramid (AB BC : ℝ) (PA : ℝ) : ℝ :=
  (1 / 3) * (AB * BC) * PA

theorem volume_PABCD (AB BC : ℝ) (h_AB : AB = 10) (h_BC : BC = 5)
  (PA : ℝ) (h_PA : PA = 2 * BC) :
  volume_of_pyramid AB BC PA = 500 / 3 :=
by
  subst h_AB
  subst h_BC
  subst h_PA
  -- At this point, we assert that everything simplifies correctly.
  -- This fill in the details for the correct expressions.
  sorry

end volume_PABCD_l507_507825


namespace max_distance_between_bus_stops_l507_507158

noncomputable def speed_ratio := 1 / 3
noncomputable def visual_distance := 2  -- in km

theorem max_distance_between_bus_stops : ∃ d, d = 1.5 :=
by 
  have boy_speed := v
  have bus_speed := 3 * v
  have max_distance_seen := visual_distance -- by the boy
  use (3 / 2)
  exact sorry -- the detailed proof steps would go here

end max_distance_between_bus_stops_l507_507158


namespace intersection_complement_l507_507725

-- Declare variables for sets
variable (I A B : Set ℤ)

-- Define the universal set I
def universal_set : Set ℤ := { x | -3 < x ∧ x < 3 }

-- Define sets A and B
def set_A : Set ℤ := { -2, 0, 1 }
def set_B : Set ℤ := { -1, 0, 1, 2 }

-- Main theorem statement
theorem intersection_complement
  (hI : I = universal_set)
  (hA : A = set_A)
  (hB : B = set_B) :
  B ∩ (I \ A) = { -1, 2 } :=
sorry

end intersection_complement_l507_507725


namespace henry_geography_math_score_l507_507310

variable (G M : ℕ)

theorem henry_geography_math_score (E : ℕ) (H : ℕ) (total_score : ℕ) 
  (hE : E = 66) 
  (hH : H = (G + M + E) / 3)
  (hTotal : G + M + E + H = total_score) 
  (htotal_score : total_score = 248) :
  G + M = 120 := 
by
  sorry

end henry_geography_math_score_l507_507310


namespace simplify_expression_l507_507434

theorem simplify_expression (x : ℤ) : 120 * x - 55 * x = 65 * x := by
  sorry

end simplify_expression_l507_507434


namespace inequality_proof_l507_507404

variables {n : ℕ} (a : Fin n → ℝ)
-- Conditions
hypothesis (hpos : ∀ i, 0 < a i)
hypothesis (hsum : (Finset.univ.sum a) < 1)

-- Theorem statement
theorem inequality_proof : (Finset.univ.prod a * (1 - Finset.univ.sum a)) / 
  ((Finset.univ.sum a) * (Finset.range n).prod (λ i => 1 - a i)) ≤ 1 / n^(n+1) :=
sorry

end inequality_proof_l507_507404


namespace points_B_O_I_H_C_cocylic_AH_AO_R_OI_IH_PO_HQ_OH_lengths_incenter_relation_l507_507328

-- Given conditions
variables {A B C : Point}
variables (O I H P Q F : Point) -- points
variables (R : ℝ) -- radii
variables (angle_A_eq_60 : ∠ A = 60)
variables (circumcenter_O : is_circumcenter O A B C)
variables (incenter_I : is_incenter I A B C)
variables (orthocenter_H : is_orthocenter H A B C)
variables (line_OH_intersects_AB_at_P : lies_on (O,H) P)
variables (line_OH_intersects_AC_at_Q : lies_on (O,H) Q)
variables (line_BI_intersects_AC_at_F : lies_on (B,I) F)

-- (1) Prove that points B, O, I, H, C are concyclic
theorem points_B_O_I_H_C_cocylic 
  (circumcenter_cond : is_circumcenter O A B C)
  (incenter_cond : is_incenter I A B C)
  (orthocenter_cond : is_orthocenter H A B C) 
  (angle_A_60 : ∠ A = 60) :
  cocyclic {B, O, I, H, C} := sorry

-- (2) Prove AH = AO = R, OI = IH, and PO = HQ
theorem AH_AO_R_OI_IH_PO_HQ 
  (AH_eq_AO : AH = AO)
  (AO_eq_R : AO = R)
  (OI_eq_IH : OI = IH)
  (PO_eq_HQ : PO = HQ) :
  AH = AO ∧ AO = R ∧ OI = IH ∧ PO = HQ := sorry

-- (3) Prove OH = |AB - AC| and OH = |HB - HC| / sqrt(3)
theorem OH_lengths 
  (AB_neq_AC : AB ≠ AC) :
  OH = abs (AB - AC) ∧ OH = abs (HB - HC) / √(3) := sorry

-- (4) Prove (1/IB) + (1/IC) = (1/IF)
theorem incenter_relation :
  1 / IB + 1 / IC = 1 / IF := sorry

end points_B_O_I_H_C_cocylic_AH_AO_R_OI_IH_PO_HQ_OH_lengths_incenter_relation_l507_507328


namespace sum_of_possible_b_values_l507_507445

theorem sum_of_possible_b_values (a b c : ℤ) 
  (h : ∀ x : ℝ, (x - a) * (x - 6) + 3 = (x + b) * (x + c)) :
  b ∈ {-3, -5, -7, -9} → -3 + -5 + -7 + -9 = -24 :=
by sorry

end sum_of_possible_b_values_l507_507445


namespace inverse_passes_through_3_4_l507_507277

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- Given that f(x) has an inverse
def has_inverse := ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- Given that y = f(x+1) passes through the point (3,3)
def condition := f (3 + 1) = 3

theorem inverse_passes_through_3_4 
  (h1 : has_inverse f) 
  (h2 : condition f) : 
  f⁻¹ 3 = 4 :=
sorry

end inverse_passes_through_3_4_l507_507277


namespace cosine_sum_identity_l507_507424

theorem cosine_sum_identity (n : ℕ) (x : ℝ) :
  (1 / 2 + Finset.sum (Finset.range n) (λ k, Real.cos ((k + 1) * x)) = Real.sin ((n + 1 / 2) * x) / (2 * Real.sin (1 / 2 * x))) :=
by
  sorry

end cosine_sum_identity_l507_507424


namespace six_digit_number_property_l507_507078

theorem six_digit_number_property :
  ∃ N : ℕ, N = 285714 ∧ (∃ x : ℕ, N = 2 * 10^5 + x ∧ M = 10 * x + 2 ∧ M = 3 * N) :=
by
  sorry

end six_digit_number_property_l507_507078


namespace functional_eq_unique_l507_507676

theorem functional_eq_unique (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(x - f(y)) = x - y) ↔ f = id :=
by
  apply Function.funext
  intro x
  sorry

end functional_eq_unique_l507_507676


namespace solve_system_1_solve_system_2_l507_507837

theorem solve_system_1 (x y : ℤ) (h1 : y = 2 * x - 3) (h2 : 3 * x + 2 * y = 8) : x = 2 ∧ y = 1 :=
by {
  sorry
}

theorem solve_system_2 (x y : ℤ) (h1 : 2 * x + 3 * y = 7) (h2 : 3 * x - 2 * y = 4) : x = 2 ∧ y = 1 :=
by {
  sorry
}

end solve_system_1_solve_system_2_l507_507837


namespace walk_to_bus_stop_l507_507976

variables (S T : ℝ)
variables (h_speed : S > 0)
variables (h_time : T > 0)
variables (h_condition : S * T = (4 / 5) * S * (T + 8))

theorem walk_to_bus_stop :
  T = 32 := 
by {
  have h1 : T - (4 / 5) * T = (4 / 5) * 8,
  { calc
      T - (4 / 5) * T
          = T * (1 - 4 / 5) : by ring
      ... = T * (1 / 5) : by norm_num },
  have h2 : (1 / 5) * T = (4 / 5) * 8,
  { rw h1 },
  have h3 : T = 32,
  { calc
      T = (4 / 5) * 8 / (1 / 5) : by noncomm_ring
      ... = 32 : by norm_num },
  exact h3,
}

end walk_to_bus_stop_l507_507976


namespace Cody_initial_money_l507_507646

-- Define the conditions
def initial_money (x : ℕ) : Prop :=
  x + 9 - 19 = 35

-- Define the theorem we need to prove
theorem Cody_initial_money : initial_money 45 :=
by
  -- Add a placeholder for the proof
  sorry

end Cody_initial_money_l507_507646


namespace area_difference_l507_507359

theorem area_difference 
  (AB AE BC : ℝ) (H1 : AB = 5) (H2 : AE = 10) (H3 : BC = 8) 
  (angle_EAB angle_ABC : ℝ) 
  (H4 : angle_EAB = π / 2) (H5 : angle_ABC = π / 2) 
  (x y z area_ADE area_BDC area_ABD : ℝ)
  (H6 : area_ADE = 1/2 * AB * AE) 
  (H7 : area_BDC = 1/2 * AB * BC) 
  (H8 : area_ABD = 1/2 * AB * AE = 25) 
  (H9 : area_ABD = 1/2 * AB * BC = 20) 
  : (area_ADE - area_BDC = 5) :=
sorry

end area_difference_l507_507359


namespace smallest_integer_with_eight_factors_l507_507556

theorem smallest_integer_with_eight_factors :
  ∃ n : ℕ, (∀ m : ℕ, (∃ k : ℕ, m has_factors k ∧ k = 8) → m ≥ n) ∧ n = 24 :=
by
  sorry

end smallest_integer_with_eight_factors_l507_507556


namespace perp_vectors_find_angles_l507_507303

variables {α β : ℝ}

def vector_a (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
def vector_b (β : ℝ) : ℝ × ℝ := (Real.cos β, Real.sin β)
def vector_c : ℝ × ℝ := (0, 1)

theorem perp_vectors (h1 : 0 < β) (h2 : β < α) (h3 : α < Real.pi) (h4 : ∥vector_a α - vector_b β∥ = Real.sqrt 2) : 
  ∃ θ, vector_a α ⬝ vector_b β = θ ∧ θ = 0 := 
sorry

theorem find_angles (h5 : vector_a α + vector_b β = vector_c) :
  α = (5 * Real.pi) / 6 ∧ β = Real.pi / 6 :=
sorry

end perp_vectors_find_angles_l507_507303


namespace sum_cubes_of_roots_l507_507050

noncomputable def cube_root_sum_cubes (α β γ : ℝ) : ℝ :=
  α^3 + β^3 + γ^3
  
theorem sum_cubes_of_roots : 
  (cube_root_sum_cubes (Real.rpow 27 (1/3)) (Real.rpow 64 (1/3)) (Real.rpow 125 (1/3))) - 3 * ((Real.rpow 27 (1/3)) * (Real.rpow 64 (1/3)) * (Real.rpow 125 (1/3)) + 4/3) = 36 
  ∧
  ((Real.rpow 27 (1/3) + Real.rpow 64 (1/3) + Real.rpow 125 (1/3)) * ((Real.rpow 27 (1/3) + Real.rpow 64 (1/3) + Real.rpow 125 (1/3))^2 - 3 * ((Real.rpow 27 (1/3)) * (Real.rpow 64 (1/3)) + (Real.rpow 64 (1/3)) * (Real.rpow 125 (1/3)) + (Real.rpow 125 (1/3)) * (Real.rpow 27 (1/3)))) = 36) 
  → 
  cube_root_sum_cubes (Real.rpow 27 (1/3)) (Real.rpow 64 (1/3)) (Real.rpow 125 (1/3)) = 220 := 
sorry

end sum_cubes_of_roots_l507_507050


namespace clarinet_count_l507_507831

-- Define the number of people in each section
def orchestra_size : ℕ := 21
def percussion : ℕ := 1
def trombone : ℕ := 4
def trumpet : ℕ := 2
def french_horn : ℕ := 1
def violin : ℕ := 3
def cello : ℕ := 1
def contrabass : ℕ := 1
def flute : ℕ := 4
def maestro : ℕ := 1

-- The total of known members
def known_members : ℕ := percussion + trombone + trumpet + french_horn +
                          violin + cello + contrabass + flute + maestro

-- Theorem to prove the number of clarinet players
theorem clarinet_count : (orchestra_size - known_members) = 3 :=
by
  exact nil -- skip the proof

end clarinet_count_l507_507831


namespace min_N_of_block_viewed_l507_507606

theorem min_N_of_block_viewed (x y z N : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_factor : (x - 1) * (y - 1) * (z - 1) = 231) : 
  N = x * y * z ∧ N = 384 :=
by {
  sorry 
}

end min_N_of_block_viewed_l507_507606


namespace arithmetic_mean_of_fractions_l507_507896

theorem arithmetic_mean_of_fractions :
  (3/8 + 5/9) / 2 = 67 / 144 :=
by
  sorry

end arithmetic_mean_of_fractions_l507_507896


namespace clubs_students_equal_l507_507340

theorem clubs_students_equal
  (C E : ℕ)
  (h1 : ∃ N, N = 3 * C)
  (h2 : ∃ N, N = 3 * E) :
  C = E :=
by
  sorry

end clubs_students_equal_l507_507340


namespace hyperbola_asymptotes_l507_507219

variable {a b : ℝ}
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (he : (Real.sqrt (a^2 + b^2))/a = 3)

theorem hyperbola_asymptotes :
  (x : ℝ)(y : ℝ) (h : x * y = 0) → (x ± 2 * Real.sqrt 2 * y = 0) :=
by
  sorry

end hyperbola_asymptotes_l507_507219


namespace financial_loss_correct_l507_507980

noncomputable def financial_result : ℝ :=
  let initial_rubles : ℝ := 60000
  let initial_dollar_selling_rate : ℝ := 59.65
  let initial_dollar_buying_rate : ℝ := 56.65
  let annual_interest_rate : ℝ := 0.015
  let duration_in_months : ℝ := 6
  let final_dollar_selling_rate : ℝ := 58.95
  let final_dollar_buying_rate : ℝ := 55.95

  -- Step 1: Convert Rubles to Dollars
  let amount_in_usd : ℝ := initial_rubles / initial_dollar_selling_rate

  -- Step 2: Calculate the amount of USD after 6 months with interest
  let effective_interest_rate : ℝ := annual_interest_rate / 2
  let amount_in_usd_after_6_months : ℝ := amount_in_usd * (1 + effective_interest_rate)

  -- Step 3: Convert USD back to Rubles
  let final_amount_in_rubles : ℝ := amount_in_usd_after_6_months * final_dollar_buying_rate

  -- Step 4: Calculate the financial result
  initial_rubles - final_amount_in_rubles

theorem financial_loss_correct : financial_result ≈ 3309 :=
by
  sorry

end financial_loss_correct_l507_507980


namespace correct_sunset_time_l507_507419

-- Define time and duration using hours and minutes.
structure Time where
  hour : Nat
  minute : Nat

-- Define the conditions given in the problem.
def sunrise : Time := ⟨6, 32⟩
def length_of_daylight : Time := ⟨11, 35⟩
def incorrect_sunset : Time := ⟨19, 18⟩ -- 7:18 PM in 24-hour format

-- Prove the correct sunset time.
theorem correct_sunset_time :
  let correct_sunset := Time.mk 18 7 in
  Time.mk ((sunrise.hour + length_of_daylight.hour) % 24 + (sunrise.minute + length_of_daylight.minute) / 60)
    ((sunrise.minute + length_of_daylight.minute) % 60) = correct_sunset :=
by
  -- define the correct_sunset to be 6:07 PM in 24-hour notation
  have correct_sunset := Time.mk 18 7
  -- The hours and minutes addition logic as derived in the solution
  have hr := (sunrise.hour + length_of_daylight.hour) % 24 + (sunrise.minute + length_of_daylight.minute) / 60
  have min := (sunrise.minute + length_of_daylight.minute) % 60
  -- Complete the proof
  sorry

end correct_sunset_time_l507_507419


namespace friends_same_group_probability_l507_507060

def total_students : ℕ := 800
def groups : ℕ := 4
def students_per_group : ℕ := total_students / groups

-- Probability function that defines the probability of n friends ending up in the same group
noncomputable def probability_same_group (n : ℕ) (total_students groups : ℕ) : ℚ :=
  let students_per_group := total_students / groups in
  (students_per_group / total_students)^(n - 1)

theorem friends_same_group_probability : probability_same_group 3 total_students groups = 1 / 16 := by
  sorry

end friends_same_group_probability_l507_507060


namespace aarti_work_multiple_l507_507189

-- Aarti can do a piece of work in 5 days
def days_per_unit_work := 5

-- It takes her 15 days to complete the certain multiple of work
def days_for_multiple_work := 15

-- Prove the ratio of the days for multiple work to the days per unit work equals 3
theorem aarti_work_multiple :
  days_for_multiple_work / days_per_unit_work = 3 :=
sorry

end aarti_work_multiple_l507_507189


namespace construct_region_D_l507_507978

theorem construct_region_D :
  ∀ (x y : ℝ), (2 ≤ x ∧ x ≤ 6) ∧ (1 ≤ y ∧ y ≤ 3) ∧ 
               (x^2 / 9 + y^2 / 4 < 1) ∧ (4 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 9) ∧ (0 < y ∧ y < x) →
               ∃ D : set (ℝ × ℝ), D = { p : ℝ × ℝ | 
                 (2 ≤ p.1 ∧ p.1 ≤ 6) ∧ (1 ≤ p.2 ∧ p.2 ≤ 3) ∧
                 (p.1^2 / 9 + p.2^2 / 4 < 1) ∧ (4 ≤ p.1^2 + p.2^2 ∧ p.1^2 + p.2^2 ≤ 9) ∧
                 (0 < p.2 ∧ p.2 < p.1)} :=
by
  intros x y h
  let D := { p : ℝ × ℝ |
    (2 ≤ p.1 ∧ p.1 ≤ 6) ∧ (1 ≤ p.2 ∧ p.2 ≤ 3) ∧
    (p.1^2 / 9 + p.2^2 / 4 < 1) ∧ (4 ≤ p.1^2 + p.2^2 ∧ p.1^2 + p.2^2 ≤ 9) ∧
    (0 < p.2 ∧ p.2 < p.1) }
  use D
  sorry

end construct_region_D_l507_507978


namespace total_digits_used_l507_507111

theorem total_digits_used (n : ℕ) (h : n = 2500) : 
  let first_n_even := (finset.range (2 * n + 1)).filter (λ x, x % 2 = 0)
  let count_digits := λ x, if x < 10 then 1 else if x < 100 then 2 else if x < 1000 then 3 else 4
  let total_digits := first_n_even.sum (λ x, count_digits x)
  total_digits = 9444 :=
by sorry

end total_digits_used_l507_507111


namespace smallest_result_set_l507_507952

theorem smallest_result_set :
  ∃ (a b c : ℕ), 
    a ∈ {4, 6, 8, 12, 14, 16} ∧ b ∈ {4, 6, 8, 12, 14, 16} ∧ c ∈ {4, 6, 8, 12, 14, 16} ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (a > 10 ∨ b > 10 ∨ c > 10) ∧ 
    ∀ (x y z : ℕ), 
      x ∈ {4, 6, 8, 12, 14, 16} ∧ y ∈ {4, 6, 8, 12, 14, 16} ∧ z ∈ {4, 6, 8, 12, 14, 16} ∧
      x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
      (x > 10 ∨ y > 10 ∨ z > 10) →
      (let result := [x + y, x + z, y + z].map (λ s, s * [x, y, z].erase s).min
      in result = 120) := sorry

end smallest_result_set_l507_507952


namespace largest_n_divisible_l507_507234

theorem largest_n_divisible (n : ℕ) : ∃ n = 11, 2^n ∣ (3^512 - 1) :=
by
  use 11
  sorry

end largest_n_divisible_l507_507234


namespace second_candidate_marks_l507_507159

variable (T : ℝ) (pass_mark : ℝ := 160)

-- Conditions
def condition1 : Prop := 0.20 * T + 40 = pass_mark
def condition2 : Prop := 0.30 * T - pass_mark > 0 

-- The statement we want to prove
theorem second_candidate_marks (h1 : condition1 T) (h2 : condition2 T) : 
  (0.30 * T - pass_mark = 20) :=
by 
  -- Skipping proof steps as per the guidelines
  sorry

end second_candidate_marks_l507_507159


namespace cafeteria_orders_green_apples_l507_507477

theorem cafeteria_orders_green_apples (G : ℕ) (h1 : 6 + G = 5 + 16) : G = 15 :=
by
  sorry

end cafeteria_orders_green_apples_l507_507477


namespace slopes_product_l507_507297

variables {a b c x0 y0 alpha beta : ℝ}
variables {P Q : ℝ × ℝ}
variables (M : ℝ × ℝ) (kPQ kOM : ℝ)

-- Conditions: a, b are positive real numbers
axiom a_pos : a > 0
axiom b_pos : b > 0

-- Condition: b^2 = a c
axiom b_squared_eq_a_mul_c : b^2 = a * c

-- Condition: P and Q lie on the hyperbola
axiom P_on_hyperbola : (P.1^2 / a^2) - (P.2^2 / b^2) = 1
axiom Q_on_hyperbola : (Q.1^2 / a^2) - (Q.2^2 / b^2) = 1

-- Condition: M is the midpoint of P and Q
axiom M_is_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Condition: Slopes kPQ and kOM exist
axiom kOM_def : kOM = y0 / x0
axiom kPQ_def : kPQ = beta / alpha

-- Theorem: Value of the product of the slopes
theorem slopes_product : kPQ * kOM = (1 + Real.sqrt 5) / 2 :=
sorry

end slopes_product_l507_507297


namespace total_bill_equal_180_l507_507884

theorem total_bill_equal_180 (M : ℝ) (N : ℝ := 12) (daisy_leo_didnt_pay : ℝ := 2) (extra_paid_per_remaining_friend : ℝ := 3) :
  (let total_amount_paid_by_remaining_friends := 10 * ((M / 12) + 3) in
  total_amount_paid_by_remaining_friends = M) → M = 180 :=
by
  sorry

end total_bill_equal_180_l507_507884


namespace min_weights_proof_l507_507254

open Real

noncomputable def min_weights (n : ℕ) : ℕ :=
  if ∃ k : ℕ, 3^k = 2*n + 1 then 
    log 3 (2*n + 1)
  else 
    (log 3 (2*n + 1)).ceil + 1

theorem min_weights_proof (n : ℕ) : 
  (∀ s : ℕ, 
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → ∃ ε : Finset (Fin s) → Int, k = (ε.sum (λ i, if ε i = 0 then 0 else if ε i = 1 then 3^(i : ℕ) else -3^(i : ℕ))))
  → s ≥ min_weights n) := 
sorry

end min_weights_proof_l507_507254


namespace polynomial_solution_l507_507802

noncomputable def roots (a b c : ℤ) : Set ℝ :=
  { x : ℝ | a * x ^ 2 + b * x + c = 0 }

theorem polynomial_solution :
  let x1 := (1 + Real.sqrt 13) / 2
  let x2 := (1 - Real.sqrt 13) / 2
  x1 ∈ roots 1 (-1) (-3) → x2 ∈ roots 1 (-1) (-3) →
  ((x1^5 - 20) * (3*x2^4 - 2*x2 - 35) = -1063) :=
by
  sorry

end polynomial_solution_l507_507802


namespace smallest_number_with_eight_factors_l507_507532

-- Definition of a function to count the number of distinct positive factors
def count_distinct_factors (n : ℕ) : ℕ := (List.range n).filter (fun d => d > 0 ∧ n % d = 0).length

-- Statement to prove the main problem
theorem smallest_number_with_eight_factors (n : ℕ) :
  count_distinct_factors n = 8 → n = 24 :=
by
  sorry

end smallest_number_with_eight_factors_l507_507532


namespace reaction_rate_l507_507148

-- Define the specific values for the reaction calculations
def delta_v : ℝ := 4 - 1.5
def volume : ℝ := 50
def delta_t : ℝ := 10 - 0

-- Define the rate of the reaction formula
def v_reaction (Δv V Δt : ℝ) : ℝ := Δv / (V * Δt)

-- The proof statement of the reaction rate
theorem reaction_rate :
  v_reaction delta_v volume delta_t = 0.005 :=
sorry

end reaction_rate_l507_507148


namespace power_inequality_l507_507817

theorem power_inequality 
( a b : ℝ )
( h1 : 0 < a )
( h2 : 0 < b )
( h3 : a ^ 1999 + b ^ 2000 ≥ a ^ 2000 + b ^ 2001 ) :
  a ^ 2000 + b ^ 2000 ≤ 2 :=
sorry

end power_inequality_l507_507817


namespace find_f2_plus_fneg2_l507_507742

def f (x a: ℝ) := (x + a)^3

theorem find_f2_plus_fneg2 (a : ℝ)
  (h_cond : ∀ x : ℝ, f (1 + x) a = -f (1 - x) a) :
  f 2 (-1) + f (-2) (-1) = -26 :=
by
  sorry

end find_f2_plus_fneg2_l507_507742


namespace proof_of_problem_l507_507002

variable (a b : ℝ)

def condition1 : Prop := (sin a / cos b) + (sin b / cos a) = 2
def condition2 : Prop := (cos a / sin b) + (cos b / sin a) = 4
def target : Prop := (tan a / tan b) + (tan b / tan a) = 2.5

theorem proof_of_problem (h1 : condition1 a b) (h2 : condition2 a b) : target a b := sorry

end proof_of_problem_l507_507002


namespace general_term_sequence_l507_507465

theorem general_term_sequence (n : ℕ) : 
  let a_n := n + (n^2 / (n^2 + 1)) in 
  a_n = n + (n^2 / (n^2 + 1)) :=
by sorry

end general_term_sequence_l507_507465


namespace sum_of_integers_satisfying_inequality_l507_507245

theorem sum_of_integers_satisfying_inequality :
  (∑ n in Finset.filter (λ n : ℕ, 1.5 * n - 6.5 < 7) (Finset.range 9) id) = 36 :=
by
  -- the proof will go here 
  sorry

end sum_of_integers_satisfying_inequality_l507_507245


namespace smallest_positive_integer_square_ends_in_644_l507_507679

theorem smallest_positive_integer_square_ends_in_644 :
  ∃ n : ℕ, 0 < n ∧ (n^2 % 1000 = 644) ∧ ∀ m : ℕ, 0 < m ∧ (m^2 % 1000 = 644) → n ≤ m :=
begin
  use 194,
  split,
  { linarith, },
  split,
  { norm_num, },
  {
    intros m hm,
    sorry
  }
end

end smallest_positive_integer_square_ends_in_644_l507_507679


namespace perimeter_square_III_is_8_l507_507887

-- Define the perimeters of square I and II
def perimeter_square_I := 20
def perimeter_square_II := 28

-- Define the side lengths of square I and II
def side_length_I := perimeter_square_I / 4
def side_length_II := perimeter_square_II / 4

-- Define the side length of square III
def side_length_III := abs (side_length_I - side_length_II)

-- Define the perimeter of square III
def perimeter_square_III := 4 * side_length_III

-- The theorem we want to prove
theorem perimeter_square_III_is_8 : perimeter_square_III = 8 :=
by
  sorry

end perimeter_square_III_is_8_l507_507887


namespace hemisphere_surface_area_l507_507975

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (area_base : ℝ) (surface_area_sphere : ℝ) (Q : ℝ) : 
  area_base = 3 ∧ surface_area_sphere = 4 * π * r^2 → Q = 9 :=
by
  sorry

end hemisphere_surface_area_l507_507975


namespace incircle_reflection_meet_median_l507_507852

-- Definitions: triangle ABC, right angle at B, incircle touches, reflections
variables {A B C A_1 B_1 C_1 A_2 C_2 M : Type}

variables (triangle_ABC : ∀ {a b c : Type}, ∃ (angle_right : ∠ B = 90), true)
variables (incircle_touches : ∃ (C1 A1 B1 : Type), B1 touches AB BC CA at C1 A1)
variables (reflections_A2_C2 : ∃ (A2 C2 : Type), A2 reflects B1 across BC, C2 reflects B1 across AB)

-- Problem statement
theorem incircle_reflection_meet_median
  (h1 : triangle_ABC)
  (h2 : incircle_touches)
  (h3 : reflections_A2_C2)
  : ∃ M, ∃ X, A1 A2 meet C1 C2 on median AM :=
sorry

end incircle_reflection_meet_median_l507_507852


namespace cards_distribution_l507_507996

theorem cards_distribution : ∀ (cards people : ℕ), cards = 48 ∧ people = 9 → 
  (∃ n : ℕ, n = 6 ∧ ∀ m : ℕ, m < people → if (cards / people) + (if m < cards % people then 1 else 0) < 6 then m < n) :=
by
  intros cards people h
  rw h.1 at *
  rw h.2 at *
  use 6
  split
  sorry -- to fill in the proof

end cards_distribution_l507_507996


namespace radius_of_larger_circle_l507_507688

theorem radius_of_larger_circle (r : ℝ) (R : ℝ) (h : r = 2) (externally_tangent : ∀ i j, i ≠ j → dist (C i) (C j) = 2 * r) (internally_tangent : ∀ i, dist (O) (C i) = R - r) : 
  R = 4 * real.sqrt 2 + 2 := 
by
  sorry

end radius_of_larger_circle_l507_507688


namespace prove_box_problem_l507_507157

noncomputable def boxProblem : Prop :=
  let height1 := 2
  let width1 := 4
  let length1 := 6
  let clay1 := 48
  let height2 := 3 * height1
  let width2 := 2 * width1
  let length2 := 1.5 * length1
  let volume1 := height1 * width1 * length1
  let volume2 := height2 * width2 * length2
  let n := (volume2 / volume1) * clay1
  n = 432

theorem prove_box_problem : boxProblem := by
  sorry

end prove_box_problem_l507_507157


namespace temperature_decrease_l507_507483

theorem temperature_decrease (initial : ℤ) (decrease : ℤ) : initial = -3 → decrease = 6 → initial - decrease = -9 :=
by
  intros
  sorry

end temperature_decrease_l507_507483


namespace smallest_positive_integer_with_eight_factors_l507_507519

theorem smallest_positive_integer_with_eight_factors :
  ∃ n : ℕ, (∀ m : ℕ, (m < n ∧ (∀ d : ℕ, d | m → d = 1 ∨ d = m) → (∃ a b : ℕ, distinct_factors_count m a b ∧ a = 8)) → n = 24) :=
by
  sorry

def distinct_factors_count (n : ℕ) (a b : ℕ) : Prop :=
  ∃ (p q : ℕ), prime p ∧ prime q ∧ n = p^a * q^b ∧ (a + 1) * (b + 1) = 8

end smallest_positive_integer_with_eight_factors_l507_507519


namespace power_of_integer_l507_507319

theorem power_of_integer (a : ℕ) (h : a = 14) : 
  ∃ (b : ℕ), (3150 * a) = b^2 := by
  have h1 : PrimeFactors (3150 * 14) = {2, 3, 5, 7}
  sorry

end power_of_integer_l507_507319


namespace billy_ate_2_apples_on_monday_l507_507203

theorem billy_ate_2_apples_on_monday :
  ∃ (m t w th f : ℕ),
    m + t + w + th + f = 20 ∧         -- Condition 1
    (m = 2 ∨ t = 2 ∨ w = 2 ∨ th = 2 ∨ f = 2) ∧ -- Condition 2
    t = 2 * m ∧                      -- Condition 3
    w = 9 ∧                          -- Condition 4
    th = 4 * f ∧                     -- Condition 5
    f = m / 2 ∧                      -- Condition 6
    m = 2 :=                         -- Conclusion
begin
  sorry
end

end billy_ate_2_apples_on_monday_l507_507203


namespace smallest_integer_with_eight_factors_l507_507528

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, n = 24 ∧
  (∀ d : ℕ, d ∣ n → d > 0) ∧
  ((∃ p : ℕ, prime p ∧ n = p^7) ∨
   (∃ p q : ℕ, prime p ∧ prime q ∧ n = p^3 * q) ∨
   (∃ p q r : ℕ, prime p ∧ prime q ∧ prime r ∧ n = p * q * r)) ∧
  (∀ m : ℕ, (∀ d : ℕ, d ∣ m → d > 0) → 
           ((∃ p : ℕ, prime p ∧ m = p^7 ∨ m = p^3 * q ∨ m = p * q * r) → 
            m ≥ n)) := by
  sorry

end smallest_integer_with_eight_factors_l507_507528


namespace arithmetic_mean_of_fractions_l507_507947

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 3/8
  let b : ℚ := 5/9
  (a + b) / 2 = 67 / 144 := 
by
  sorry

end arithmetic_mean_of_fractions_l507_507947


namespace remainder_of_a_sq_plus_five_mod_seven_l507_507603

theorem remainder_of_a_sq_plus_five_mod_seven (a : ℕ) (h : a % 7 = 4) : (a^2 + 5) % 7 = 0 := 
by 
  sorry

end remainder_of_a_sq_plus_five_mod_seven_l507_507603


namespace spherical_to_rectangular_coordinates_l507_507651

theorem spherical_to_rectangular_coordinates :
  let ρ := 5
  let θ := Real.pi / 4
  let φ := Real.pi / 3
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  x = (5 * Real.sqrt 6) / 4 ∧ y = (5 * Real.sqrt 6) / 4 ∧ z = 5 / 2
:= by
  let ρ := 5
  let θ := Real.pi / 4
  let φ := Real.pi / 3
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  have hx : x = (5 * Real.sqrt 6) / 4 := sorry
  have hy : y = (5 * Real.sqrt 6) / 4 := sorry
  have hz : z = 5 / 2 := sorry
  exact ⟨hx, hy, hz⟩

end spherical_to_rectangular_coordinates_l507_507651


namespace arithmetic_mean_of_fractions_l507_507895

theorem arithmetic_mean_of_fractions :
  (3/8 + 5/9) / 2 = 67 / 144 :=
by
  sorry

end arithmetic_mean_of_fractions_l507_507895


namespace solve_prime_equation_l507_507836

theorem solve_prime_equation (p q r : ℕ) (hp : p.Prime) (hq : q.Prime) (hr : r.Prime) 
(h_eq : p^3 - q^3 = 5 * r) : p = 7 ∧ q = 2 ∧ r = 67 := 
sorry

end solve_prime_equation_l507_507836


namespace find_diagonal_length_l507_507233

noncomputable def diagonal_length (area : ℝ) (offset1 offset2 : ℝ) : ℝ :=
let d := (2 * area) / (offset1 + offset2) in d

theorem find_diagonal_length :
  ∀ (area offsets1 offsets2 : ℝ), area = 150 → offset1 = 9 → offset2 = 6 → diagonal_length area offset1 offset2 = 20 :=
by
  intros area offset1 offset2 h_area h_offset1 h_offset2
  rw [h_area, h_offset1, h_offset2]
  have h : diagonal_length 150 9 6 = 20 := sorry
  exact h

end find_diagonal_length_l507_507233


namespace domain_F_domain_G_l507_507274

variable {f : ℝ → ℝ}
variable {a : ℝ}

-- Given condition
def domain_f : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

-- Part 1: Domain of F(x) = f(x^2)
def F (x : ℝ) : ℝ := f (x^2)

theorem domain_F :
  Set.Icc (-1 : ℝ) (1 : ℝ) = {x | F x ∈ domain_f} :=
sorry

-- Part 2: Domain of G(x) = f(x+a) + f(x-a)
def G (x : ℝ) : ℝ := f (x + a) + f (x - a)

theorem domain_G (a : ℝ) :
  ((a < -1/2) ∧ (∀ x, ¬ (0 ≤ x + a ∧ x + a ≤ 1 ∧ 0 ≤ x - a ∧ x - a ≤ 1))) ∨
  ((-1/2 ≤ a ∧ a ≤ 0) ∧ (∀ x, (-a ≤ x ∧ x ≤ 1 + a) ↔ (0 ≤ x + a ∧ x + a ≤ 1 ∧ 0 ≤ x - a ∧ x - a ≤ 1))) ∨
  ((0 < a ∧ a ≤ 1/2) ∧ (∀ x, (a ≤ x ∧ x ≤ 1 - a) ↔ (0 ≤ x + a ∧ x + a ≤ 1 ∧ 0 ≤ x - a ∧ x - a ≤ 1))) ∨
  ((a > 1/2) ∧ (∀ x, ¬ (0 ≤ x + a ∧ x + a ≤ 1 ∧ 0 ≤ x - a ∧ x - a ≤ 1))) :=
sorry

end domain_F_domain_G_l507_507274


namespace equation_of_ellipse_midpoint_trajectory_l507_507701

-- Definition of the ellipse and conditions
def ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b) : Prop :=
  ∀ (x y : ℝ), (x / a) ^ 2 + (y / b) ^ 2 = 1

-- Given points and conditions
def left_focus (F : ℝ × ℝ) : Prop := F = (-1, 0)

def point_on_ellipse (G : ℝ × ℝ) (a b : ℝ) (h : ellipse a b (by positivity) (by positivity) (by positivity)) : Prop :=
  G = (1, sqrt 2 / 2)

-- Line passing through focus intersects ellipse
def line_through_focus_intersects_ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b) (F : ℝ × ℝ) (hF : left_focus F) : Prop :=
  ∃ (A B : ℝ × ℝ), 
    ∃ line, 
      line F (by exact A) (by exact B) ∧ 
      ellipse a b a_pos b_pos a_gt_b (A.1) (A.2) ∧ 
      ellipse a b a_pos b_pos a_gt_b (B.1) (B.2)

-- Proof of ellipse equation
theorem equation_of_ellipse (a b : ℝ) (hp : a^2 - b^2 = 1)
  (hg : (1:ℝ) / a ^ 2 + (sqrt 2 / 2) ^ 2 / b ^ 2 = 1) :
  a = sqrt 2 ∧ b = 1 :=
sorry

-- Proof of the trajectory of midpoint M
theorem midpoint_trajectory (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b) (G : ℝ × ℝ)
  (h : equation_of_ellipse a b (by positivity) (by positivity))
  (F : ℝ × ℝ) (hF : left_focus F) 
  (intersects : line_through_focus_intersects_ellipse a b a_pos b_pos a_gt_b F hF) :
  ∀ (M : ℝ × ℝ), ∃ x y : ℝ, M = (x, y) ∧ x^2 + 2 * y^2 + x = 0 :=
sorry

end equation_of_ellipse_midpoint_trajectory_l507_507701


namespace arithmetic_mean_of_38_and_59_is_67_over_144_l507_507934

theorem arithmetic_mean_of_38_and_59_is_67_over_144 :
  (3 / 8 + 5 / 9) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_38_and_59_is_67_over_144_l507_507934


namespace correct_equation_l507_507564

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Definitions for each equation in the conditions
def equationA := (x : ℝ) → (x - 1) * (x + 1) = 0
def equationB := (x : ℝ) → (x - 1) * (x - 1) = 0
def equationC := (x : ℝ) → (x - 1)^2 = 4
def equationD := (x : ℝ) → x * (x - 1) = 0

-- Matching each equation to its general quadratic form
def coefficientsA := (1, 0, -1)
def coefficientsB := (1, -2, 1)
def coefficientsC := (1, -2, -3)
def coefficientsD := (1, -1, 0)

-- Check the discriminant for each equation
def discriminantA := discriminant 1 0 -1
def discriminantB := discriminant 1 -2 1
def discriminantC := discriminant 1 -2 -3
def discriminantD := discriminant 1 -1 0

-- The statement to prove
theorem correct_equation : discriminant 1 -2 1 = 0 := by
  sorry

end correct_equation_l507_507564


namespace clubs_equal_students_l507_507348

-- Define the concepts of Club and Student
variable (Club Student : Type)

-- Define the membership relations
variable (Members : Club → Finset Student)
variable (Clubs : Student → Finset Club)

-- Define the conditions
axiom club_membership (c : Club) : (Members c).card = 3
axiom student_club_membership (s : Student) : (Clubs s).card = 3

-- The goal is to prove that the number of clubs is equal to the number of students
theorem clubs_equal_students [Fintype Club] [Fintype Student] : Fintype.card Club = Fintype.card Student := by
  sorry

end clubs_equal_students_l507_507348


namespace chocolate_bars_in_box_l507_507662

theorem chocolate_bars_in_box (x : ℕ) 
  (h_cost : ∀ n, n * 3 = 9 → x = n + 4) : x = 7 :=
by
  have h_sold := h_cost (x - 4)
  specialize h_sold
  have h_equation : 3 * (x - 4) = 9, from sorry,
  obtain ⟨h_x⟩ := h_sold h_equation,
  exact h_x

end chocolate_bars_in_box_l507_507662


namespace find_mistaken_number_l507_507789

theorem find_mistaken_number : 
  ∃! x : ℕ, (x ∈ {n : ℕ | n ≥ 10 ∧ n < 100 ∧ (n % 10 = 5 ∨ n % 10 = 0)} ∧ 
  (10 + 15 + 20 + 25 + 30 + 35 + 40 + 45 + 50 + 55 + 60 + 65 + 70 + 75 + 80 + 85 + 90 + 95) + 2 * x = 1035) :=
sorry

end find_mistaken_number_l507_507789


namespace hamburger_combinations_l507_507734

theorem hamburger_combinations : 
  let condiments := 9 in
  let patty_types := 4 in
  2^condiments * patty_types = 2048 := by 
begin
  have H1 : 2^9 = 512 := by norm_num,
  have H2 : 4 * 512 = 2048 := by norm_num,
  rw [H1, H2]
end

end hamburger_combinations_l507_507734


namespace sum_a_1_to_100_l507_507399

noncomputable def a : ℕ → ℕ
| 1 := 1
| 2 := 1
| 3 := 1
| 4 := 1
| (n+1) := 
  let p := (a n) + (a (n-3)) in
  let q := (a (n-1)) * (a (n-2)) in
  if (p/2)^2 - q < 0 then 0
  else if (p/2)^2 - q = 0 then
    if p = 0 then 1 else 2
  else if q = 0 then 3
  else 4

theorem sum_a_1_to_100 : (Finset.range 100).sum (fun n => a (n+1)) = S := 
by
  sorry

end sum_a_1_to_100_l507_507399


namespace number_of_eggs_left_l507_507637

theorem number_of_eggs_left (initial_eggs : ℕ) (eggs_eaten_morning : ℕ) (eggs_eaten_afternoon : ℕ) (eggs_left : ℕ) :
    initial_eggs = 20 → eggs_eaten_morning = 4 → eggs_eaten_afternoon = 3 → eggs_left = initial_eggs - (eggs_eaten_morning + eggs_eaten_afternoon) → eggs_left = 13 :=
by
  intros h_initial h_morning h_afternoon h_calc
  rw [h_initial, h_morning, h_afternoon] at h_calc
  norm_num at h_calc
  exact h_calc

end number_of_eggs_left_l507_507637


namespace smallest_positive_integer_with_eight_factors_l507_507517

theorem smallest_positive_integer_with_eight_factors :
  ∃ n : ℕ, (∀ m : ℕ, (m < n ∧ (∀ d : ℕ, d | m → d = 1 ∨ d = m) → (∃ a b : ℕ, distinct_factors_count m a b ∧ a = 8)) → n = 24) :=
by
  sorry

def distinct_factors_count (n : ℕ) (a b : ℕ) : Prop :=
  ∃ (p q : ℕ), prime p ∧ prime q ∧ n = p^a * q^b ∧ (a + 1) * (b + 1) = 8

end smallest_positive_integer_with_eight_factors_l507_507517


namespace smallest_integer_with_eight_factors_l507_507546

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, 
  ∀ m : ℕ, (∀ p : ℕ, ∃ k : ℕ, m = p^k → (k + 1) * (p + 1) = 8) → (n ≤ m) ∧ 
  (∀ d : ℕ, d ∣ n → d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 24) :=
sorry

end smallest_integer_with_eight_factors_l507_507546


namespace value_of_expression_l507_507136

theorem value_of_expression (x y z : ℤ) (h1 : x = -3) (h2 : y = 5) (h3 : z = -4) :
  x^2 + y^2 - z^2 + 2*x*y = -12 :=
by
  -- proof goes here
  sorry

end value_of_expression_l507_507136


namespace coefficient_x3y9_in_expansion_l507_507948

theorem coefficient_x3y9_in_expansion : 
  coefficient_of_term_in_expansion (x^3 * y^9) ((2 / 3) * x - (1 / 3) * y)^12 = -1760 / 531441 := 
by
  sorry

end coefficient_x3y9_in_expansion_l507_507948


namespace total_miles_walked_l507_507422

-- Given conditions
def steps_per_reset : ℕ := 100000
def reset_count : ℕ := 50
def final_day_steps : ℕ := 25000
def steps_per_mile : ℕ := 1500

-- Problem statement
lemma miles_walked : reset_count * steps_per_reset + final_day_steps = 5025000 :=
by
  calc
    reset_count * steps_per_reset + final_day_steps
        = 50 * 100000 + 25000 : by rfl
    ... = 5000000 + 25000 : by rfl
    ... = 5025000 : by rfl

-- Correct answer
theorem total_miles_walked : (reset_count * steps_per_reset + final_day_steps) / steps_per_mile = 3350 :=
by
  calc
    (reset_count * steps_per_reset + final_day_steps) / steps_per_mile
        = 5025000 / 1500 : by rw miles_walked
    ... = 3350 : by norm_num


end total_miles_walked_l507_507422


namespace inscribed_square_sum_c_d_eq_200689_l507_507188

theorem inscribed_square_sum_c_d_eq_200689 :
  ∃ (c d : ℕ), Nat.gcd c d = 1 ∧ (∃ x : ℚ, x = (c : ℚ) / (d : ℚ) ∧ 
    let a := 48
    let b := 55
    let longest_side := 73
    let s := (a + b + longest_side) / 2
    let area := Real.sqrt (s * (s - a) * (s - b) * (s - longest_side))
    area = 1320 ∧ x = 192720 / 7969 ∧ c + d = 200689) :=
sorry

end inscribed_square_sum_c_d_eq_200689_l507_507188


namespace batteries_manufactured_l507_507592

theorem batteries_manufactured (gather_time create_time : Nat) (robots : Nat) (hours : Nat) (total_batteries : Nat) :
  gather_time = 6 →
  create_time = 9 →
  robots = 10 →
  hours = 5 →
  total_batteries = (hours * 60 / (gather_time + create_time)) * robots →
  total_batteries = 200 :=
by
  intros h_gather h_create h_robots h_hours h_batteries
  simp [h_gather, h_create, h_robots, h_hours] at h_batteries
  exact h_batteries

end batteries_manufactured_l507_507592


namespace simplify_and_evaluate_expression_l507_507048

def x := 2 + 3 * Real.sqrt 3
def y := 2 - 3 * Real.sqrt 3

theorem simplify_and_evaluate_expression :
  (x^2 / (x - y) - y^2 / (x - y)) = 4 := by
  sorry

end simplify_and_evaluate_expression_l507_507048


namespace at_least_one_square_leq_2M_div_n_n_minus_1_l507_507797

theorem at_least_one_square_leq_2M_div_n_n_minus_1
  (n : ℕ) (a : ℕ → ℝ)
  (h_nonneg : ∀ i, 1 ≤ i → i ≤ n → 0 ≤ a i)
  (M_def : M = ∑ (i j : ℕ) in finset.filter (λ ij, ij.1 < ij.2) (finset.range n).product (finset.range n), (a i.1) * (a i.2))
  (M : ℝ)
  : ∃ i, 1 ≤ i ∧ i ≤ n ∧ (a i) ^ 2 ≤ 2 * M / (n * (n - 1)) :=
sorry

end at_least_one_square_leq_2M_div_n_n_minus_1_l507_507797


namespace divide_fractions_l507_507739

theorem divide_fractions : (3 / 8) / (1 / 4) = 3 / 2 :=
by sorry

end divide_fractions_l507_507739


namespace find_A_find_b_and_c_l507_507781

open Real

variable {a b c A B C : ℝ}

-- Conditions for the problem
axiom triangle_sides : ∀ {A B C : ℝ}, a > 0
axiom sine_law_condition : b * sin B + c * sin C - sqrt 2 * b * sin C = a * sin A
axiom degrees_60 : B = π / 3
axiom side_a : a = 2

theorem find_A : A = π / 4 :=
by sorry

theorem find_b_and_c (h : A = π / 4) (hB : B = π / 3) (ha : a = 2) : b = sqrt 6 ∧ c = 1 + sqrt 3 :=
by sorry

end find_A_find_b_and_c_l507_507781


namespace vectors_coplanar_l507_507630

/-
Problem: Are the vectors a, b, and c coplanar?

Given:
  a = {4, 3, 1}
  b = {6, 7, 4}
  c = {2, 0, -1}

Solution: Verify whether the determinant of the matrix formed by these vectors is zero.
-/
def a : ℝ × ℝ × ℝ := (4, 3, 1)
def b : ℝ × ℝ × ℝ := (6, 7, 4)
def c : ℝ × ℝ × ℝ := (2, 0, -1)

theorem vectors_coplanar : 
  det ![
    ![4, 3, 1],
    ![6, 7, 4],
    ![2, 0, -1]
  ] = 0 := by
  sorry

end vectors_coplanar_l507_507630


namespace remove_max_preserves_perfect_l507_507791

-- Let \( n \) be a natural number and suppose that \( w_1, w_2, \ldots, w_n \) are \( n \) weights.
variable (n : ℕ)
variable (w : ℕ → ℕ)
variable (W : ℕ := ∑ i in finset.range n, w i)

-- Define a Perfect Set
def isPerfectSet (S : finset ℕ) : Prop :=
  (1 ∈ S) ∧ (∀ k ∈ S, ∃ t : finset ℕ, t ⊆ S ∧ k = t.sum ∧ k ≤ 1 + (t.erase k).sum)

-- Given condition that \( \{ w_1, w_2, \ldots , w_n \} \) is a Perfect Set
variable (hPerfect : isPerfectSet (finset.univ.image w))

-- Prove that if we delete the maximum weight, the other weights make again a Perfect Set.
theorem remove_max_preserves_perfect :
  ∀ (k : ℕ) (h : k < n), isPerfectSet (finset.univ.image (λ i, if i < k then w i else w (i + 1))) :=
sorry

end remove_max_preserves_perfect_l507_507791


namespace exact_one_solves_l507_507153

variables (p1 p2 : ℝ)

/-- The probability that exactly one of two persons solves the problem
    when their respective probabilities are p1 and p2. -/
theorem exact_one_solves (h1 : 0 ≤ p1) (h2 : p1 ≤ 1) (h3 : 0 ≤ p2) (h4 : p2 ≤ 1) :
  (p1 * (1 - p2) + p2 * (1 - p1)) = (p1 + p2 - 2 * p1 * p2) := 
sorry

end exact_one_solves_l507_507153


namespace clubs_equal_students_l507_507346

-- Define the concepts of Club and Student
variable (Club Student : Type)

-- Define the membership relations
variable (Members : Club → Finset Student)
variable (Clubs : Student → Finset Club)

-- Define the conditions
axiom club_membership (c : Club) : (Members c).card = 3
axiom student_club_membership (s : Student) : (Clubs s).card = 3

-- The goal is to prove that the number of clubs is equal to the number of students
theorem clubs_equal_students [Fintype Club] [Fintype Student] : Fintype.card Club = Fintype.card Student := by
  sorry

end clubs_equal_students_l507_507346


namespace black_to_white_area_ratio_l507_507246

theorem black_to_white_area_ratio {r1 r2 r3 r4 r5 : ℕ}
  (h1 : r1 = 2)
  (h2 : r2 = 4)
  (h3 : r3 = 6)
  (h4 : r4 = 8)
  (h5 : r5 = 10) :
  let a1 := ℕ.pi * r1^2
      a2 := ℕ.pi * r2^2
      a3 := ℕ.pi * r3^2
      a4 := ℕ.pi * r4^2
      a5 := ℕ.pi * r5^2
      white1 := a1
      black1 := a2 - a1
      white2 := a3 - a2
      black2 := a4 - a3
      white3 := a5 - a4
      total_black := black1 + black2
      total_white := white1 + white2 + white3
  in total_black / total_white = 2 / 3 :=
sorry

end black_to_white_area_ratio_l507_507246


namespace probability_red_even_green_gt_5_l507_507095

def red_die := fin 6
def green_die := fin 8

def is_even (n : red_die) : Prop :=
  n = 1 ∨ n = 3 ∨ n = 5

def greater_than_5 (n : green_die) : Prop :=
  n = 5 ∨ n = 6 ∨ n = 7

theorem probability_red_even_green_gt_5 : 
  ∃ (p : ℚ), p = 3 / 16 :=
by
  sorry

end probability_red_even_green_gt_5_l507_507095


namespace at_most_p_minus_1_multiples_of_p_in_S_l507_507801

open Int

theorem at_most_p_minus_1_multiples_of_p_in_S (p : ℕ) (hp : prime p) (hp_cond : p > 2) (p_mod : (p - 2) % 3 = 0) :
  let S := { z : ℤ | ∃ x y : ℤ, 0 <= x ∧ x <= p - 1 ∧ 0 <= y ∧ y <= p - 1 ∧ z = y^2 - x^3 - 1 }
  ∃ t, t.card = p - 1 ∧ t ⊆ { z | z ≠ 0 → z % p = 0 } := 
sorry

end at_most_p_minus_1_multiples_of_p_in_S_l507_507801


namespace integral_eq_two_l507_507693

theorem integral_eq_two (a : ℝ) : 
  (∫ x in 0..(Real.pi / 2), (Real.sin x - a * Real.cos x)) = 2 → a = -1 := 
by 
  sorry

end integral_eq_two_l507_507693


namespace eccentricity_range_of_hyperbola_l507_507272

theorem eccentricity_range_of_hyperbola :
  ∀ (A B C F D P : Point) (hyp : Locus P = Hyperbola)
    (h1 : collinear A B C) (h2 : collinear A B F)
    (h3 : dist A B = 12) (h4 : dist A C = 6)
    (h5 : dist A D = 6) (h6 : bisects FD AD P),
    1 < eccentricity (Locus P) ∧ eccentricity (Locus P) ≤ 2 :=
sorry

end eccentricity_range_of_hyperbola_l507_507272


namespace total_boys_in_class_l507_507338

theorem total_boys_in_class (n : ℕ)
  (h1 : 19 + 19 - 1 = n) :
  n = 37 :=
  sorry

end total_boys_in_class_l507_507338


namespace largest_odd_distinct_digits_persistence_1_largest_even_distinct_nonzero_digits_persistence_1_smallest_nat_with_persistence_3_l507_507859

-- Definition of persistence
def digit_product (n : ℕ) := (n.digits 10).prod

def persistence (n : ℕ) : ℕ :=
  if n < 10 then 0 else
    let rec aux p ct :=
      if p < 10 then ct else aux (digit_product p) (ct + 1)
    aux (digit_product n) 1

-- 1. Prove that the largest odd number with distinct digits and persistence 1 is 9876543201
theorem largest_odd_distinct_digits_persistence_1 :
  ∃ n, (n = 9876543201) ∧ (odd n) ∧ (∀ (d : ℕ), d ∈ n.digits 10 → (n.digits 10).count d = 1) ∧ (persistence n = 1) :=
by
  sorry

-- 2. Prove that the largest even number with distinct nonzero digits and persistence 1 is 412
theorem largest_even_distinct_nonzero_digits_persistence_1 :
  ∃ n, (n = 412) ∧ (even n) ∧ (∀ (d : ℕ), d ∈ n.digits 10 → d ≠ 0) ∧ (∀ (d : ℕ), d ∈ n.digits 10 → (n.digits 10).count d = 1) ∧ (persistence n = 1) :=
by
  sorry

-- 3. Prove that the smallest natural number with persistence 3 is 39
theorem smallest_nat_with_persistence_3 :
  ∃ n, (n = 39) ∧ (persistence n = 3) :=
by
  sorry

end largest_odd_distinct_digits_persistence_1_largest_even_distinct_nonzero_digits_persistence_1_smallest_nat_with_persistence_3_l507_507859


namespace ellipse_equation_correct_l507_507623

-- Definitions for Lean proof
theorem ellipse_equation_correct :
  ∀ (x y : ℝ) (F1 F2 : ℝ × ℝ) (P : ℝ × ℝ),
  P = (2, real.sqrt 3) ∧
  F1.2 = 0 ∧ F2.2 = 0 ∧ -- F1 and F2 lie on the x-axis
  (∀ x1 y1 x2 y2 : ℝ, dist (x1, y1) (x2, y2) = real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)) ∧
  let PF1 := dist P F1,
  PF2 := dist P F2,
  F1F2 := dist F1 F2 in
  (PF1, F1F2, PF2) form an arithmetic sequence ∧
  (∀ a b c : ℝ, 2*F1F2 = PF2 + PF1) →
  (a = 2*c) ∧
  (a^2 = b^2 + c^2) ∧
  (4/(a^2) + 3/(b^2) = 1) →
  (a = 2 * real.sqrt 2) ∧
  (c = real.sqrt 2) ∧
  (b^2 = 6) →
  ((d / 8) + (e / 6) = 1) :=
sorry

end ellipse_equation_correct_l507_507623


namespace find_h_at_1_l507_507799

-- Definitions for conditions
variables (a b c : ℕ)
-- Assume conditions on a, b, and c.
variables (p q r : ℝ) (h f : ℝ → ℝ)
-- Assume a, b, c are positive and a < b < c.
variables (ha : a > 0) (hb : b > 0) (hc : c > 0) (hab : a < b) (hbc : b < c)

-- Polynomial f(x) with roots p, q, r.
def f (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Roots of polynomial f.
axiom roots_f : ∀ (x : ℝ), f(x) = (x - p) * (x - q) * (x - r)

-- Polynomial h(x) with transformed roots.
def h (x : ℝ) : ℝ := (x - 1/(p^2+1)) * (x - 1/(q^2+1)) * (x - 1/(r^2+1))

-- Main theorem statement
theorem find_h_at_1 : h 1 = 1 :=
sorry

end find_h_at_1_l507_507799


namespace johns_total_payment_l507_507368

theorem johns_total_payment :
  let silverware_cost := 20
  let dinner_plate_cost := 0.5 * silverware_cost
  let total_cost := dinner_plate_cost + silverware_cost
  total_cost = 30 := sorry

end johns_total_payment_l507_507368


namespace total_money_spent_l507_507145

noncomputable def calculate_total_expenditure (n: ℕ) (eighths_expenditure: ℕ) (additional_expenditure: ℝ) (average_expenditure: ℝ) : ℝ :=
  let total_expenditure_of_eight := (n - 1) * eighths_expenditure
  let ninth_person_expenditure := average_expenditure + additional_expenditure
  total_expenditure_of_eight + ninth_person_expenditure

theorem total_money_spent
  (n : ℕ := 9)
  (eighths_expenditure : ℕ := 30)
  (additional_expenditure : ℝ := 20)
  (total_expenditure : ℝ := 292.5) :
  let A := (8 * 30 + 20) / 8
  let total := calculate_total_expenditure n eighths_expenditure additional_expenditure A in
  total = total_expenditure :=
begin
  sorry
end

end total_money_spent_l507_507145


namespace smallest_integer_with_eight_factors_l507_507509

theorem smallest_integer_with_eight_factors:
  ∃ n : ℕ, ∀ (d : ℕ), d > 0 → d ∣ n → 8 = (divisor_count n) 
  ∧ (∀ m : ℕ, m > 0 → (∀ (d : ℕ), d > 0 → d ∣ m → 8 = (divisor_count m)) → n ≤ m) → n = 24 :=
begin
  sorry
end

end smallest_integer_with_eight_factors_l507_507509


namespace max_abs_x_minus_2y_plus_1_l507_507686

theorem max_abs_x_minus_2y_plus_1 (x y : ℝ) (h1 : |x - 1| ≤ 1) (h2 : |y - 2| ≤ 1) :
  |x - 2 * y + 1| ≤ 5 :=
sorry

end max_abs_x_minus_2y_plus_1_l507_507686


namespace smallest_integer_with_eight_factors_l507_507542

theorem smallest_integer_with_eight_factors : ∃ N : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) ∧ ∀ M : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) → N ≤ M :=
sorry -- Proof not provided.

end smallest_integer_with_eight_factors_l507_507542


namespace compute_expression_l507_507216

theorem compute_expression : (125 / 27) ^ (-1 / 3) + Math.log 10 (1 / 4) - Math.log 10 25 = -7 / 5 :=
by sorry

end compute_expression_l507_507216


namespace cylinder_cone_same_base_height_l507_507166

noncomputable def volume_difference_cone_cylinder (V_sum : ℝ) : ℝ := 
  let V_cone := V_sum / 4
  let V_cylinder := V_cone * 3
  V_cylinder - V_cone

theorem cylinder_cone_same_base_height (V_sum : ℝ) (h1 : V_sum = 196) : 
  volume_difference_cone_cylinder V_sum = 98 :=
by
  rw [volume_difference_cone_cylinder, h1]
  have V_cylinder : 196 / 4 * 3 = 147
  {
    rw div_eq_mul_inv
    norm_num
  }
  have V_cone : 196 / 4 = 49
  {
    rw div_eq_mul_inv
    norm_num
  }
  calc 
    147 - 49 = 98 : by norm_num

end cylinder_cone_same_base_height_l507_507166


namespace inequality_must_hold_l507_507744

theorem inequality_must_hold (m n : ℝ) (h1 : m < n) (h2 : n < 0) : m + 2 < n + 2 ∧ ¬(-2m < -2n) ∧ ¬(m / 2 > n / 2) ∧ ¬(1 / m < 1 / n) := 
by
  sorry

end inequality_must_hold_l507_507744


namespace opening_length_correct_l507_507170

def radius : ℝ := 7
def fence_length : ℝ := 33
def pi_approx : ℝ := 3.14159

noncomputable def length_of_opening : ℝ :=
  pi_approx * radius + 2 * radius - fence_length

theorem opening_length_correct :
  length_of_opening ≈ 2.99113 :=
begin
  sorry
end

end opening_length_correct_l507_507170


namespace some_employee_not_team_leader_l507_507633

variables (Employee : Type) (isTeamLeader : Employee → Prop) (meetsDeadline : Employee → Prop)

-- Conditions
axiom some_employee_not_meets_deadlines : ∃ e : Employee, ¬ meetsDeadline e
axiom all_team_leaders_meet_deadlines : ∀ e : Employee, isTeamLeader e → meetsDeadline e

-- Theorem to prove
theorem some_employee_not_team_leader : ∃ e : Employee, ¬ isTeamLeader e :=
sorry

end some_employee_not_team_leader_l507_507633


namespace incorrect_mode_l507_507684

theorem incorrect_mode (s : List ℕ) (h : s = [3, 4, 2, 2, 4]) :
  ¬(mode s = [4]) :=
by
  sorry

end incorrect_mode_l507_507684


namespace divide_space_identical_tetrahedra_divide_space_identical_isohedral_tetrahedra_l507_507432

-- Problem (a): Prove that space can be divided into identical tetrahedra with volume 1/12
theorem divide_space_identical_tetrahedra :
  ∀ space : Type, ∃ identical_tetrahedra : Set space, 
    (∀ tetrahedron ∈ identical_tetrahedra, volume tetrahedron = 1 / 12) ∧ 
    (space = ⋃₀ identical_tetrahedra) :=
sorry

-- Problem (b): Prove that space can be divided into identical isohedral tetrahedra
theorem divide_space_identical_isohedral_tetrahedra :
  ∀ space : Type, ∃ isohedral_tetrahedra : Set space, 
    (∀ tetrahedron ∈ isohedral_tetrahedra, is_isohedral tetrahedron) ∧ 
    (space = ⋃₀ isohedral_tetrahedra) :=
sorry

end divide_space_identical_tetrahedra_divide_space_identical_isohedral_tetrahedra_l507_507432


namespace find_m_l507_507292

-- Define the function f(x)
def f (x m : ℝ) := x^2 - 2 * x + m

-- Define the condition that the probability of f(x) being negative over the interval is 2/3
def prob_f_neg (m : ℝ) : Prop :=
  let X : Set ℝ := { x | f x m < 0 }
  (Set.Icc (-2 : ℝ) 4).measure (X) = 2 / 3 * (Set.Icc (-2 : ℝ) 4).measure

-- Define the main theorem stating that the value of m must be -3 given the condition
theorem find_m (m : ℝ) (h : prob_f_neg m)  : m = -3 := sorry

end find_m_l507_507292


namespace prove_by_contradiction_l507_507036

-- Statement: To prove "a > b" by contradiction, assuming the negation "a ≤ b".
theorem prove_by_contradiction (a b : ℝ) (h : a ≤ b) : false := sorry

end prove_by_contradiction_l507_507036


namespace square_perimeter_l507_507844

-- We define a structure for a square with an area as a condition.
structure Square (s : ℝ) :=
(area_eq : s ^ 2 = 400)

-- The theorem states that given the area of the square is 400 square meters,
-- the perimeter of the square is 80 meters.
theorem square_perimeter (s : ℝ) (sq : Square s) : 4 * s = 80 :=
by
  -- proof omitted
  sorry

end square_perimeter_l507_507844


namespace find_angle_A_find_min_magnitude_l507_507360

-- Definitions of the problem conditions
def is_in_triangle_A_B_C (A B C a b c : ℝ) : Prop := 
  -- In triangle ABC, the sides opposite to angles A, B, C are a, b, c respectively
  1 + (Real.tan A) / (Real.tan B) = 2 * c / b

def vectors_defined (B C : ℝ) : Prop := 
  let m := (0, -1 : ℝ × ℝ) in
  let n := (Real.cos B, 2 * Real.cos (C / 2)^2 : ℝ × ℝ) in
  true

-- Statements of the proof problems
theorem find_angle_A (A B C a b c : ℝ) (h1 : is_in_triangle_A_B_C A B C a b c) :
  A = Real.pi / 3 := 
sorry

theorem find_min_magnitude (B C : ℝ) (h2 : vectors_defined B C) :
  let m := (0, -1 : ℝ × ℝ) in
  let n := (Real.cos B, 2 * Real.cos (C / 2)^2 : ℝ × ℝ) in
  (m + n).norm = Real.sqrt 2 / 2 := 
sorry

end find_angle_A_find_min_magnitude_l507_507360


namespace smallest_integer_with_eight_factors_l507_507541

theorem smallest_integer_with_eight_factors : ∃ N : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) ∧ ∀ M : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) → N ≤ M :=
sorry -- Proof not provided.

end smallest_integer_with_eight_factors_l507_507541


namespace possible_denominators_of_repeating_decimal_l507_507444

theorem possible_denominators_of_repeating_decimal :
  ∃ (D : Nat), D = 6 ∧ ∀ (a b : Fin 10),
  ¬(a = 0 ∧ b = 0) →
  let frac := (a.val * 10 + b.val) / 99 in 
  let denom := (frac.num.gcd frac.denom) in
  denom.count_divisors = D := 
sorry

end possible_denominators_of_repeating_decimal_l507_507444


namespace bowling_ball_weight_l507_507059

variables (b c : ℝ)

def condition1 := (10 * b = 5 * c)
def condition2 := (3 * c = 102)

theorem bowling_ball_weight : condition1 b c → condition2 c → b = 17 :=
by {
  intros h_cond1 h_cond2,
  -- proof unfolds here, using the conditions h_cond1 and h_cond2
  sorry
}

end bowling_ball_weight_l507_507059


namespace people_on_trolley_final_l507_507616

-- Defining the conditions from part a)
def people_first_stop : ℕ := 10
def people_off_second_stop : ℕ := 3
def people_on_second_stop : ℕ := 2 * people_first_stop
def people_off_third_stop : ℕ := 18
def people_on_third_stop : ℕ := 2

-- Proving the theorem that the number of people on the trolley after the third stop is 12.
theorem people_on_trolley_final :
  let initial_people := 1 in -- the trolley driver alone initially
  let after_first_stop := initial_people + people_first_stop in
  let after_second_stop := after_first_stop - people_off_second_stop + people_on_second_stop in
  let final_count := after_second_stop - people_off_third_stop + people_on_third_stop in
  final_count = 12 :=
by 
  -- Proof omitted
  sorry

end people_on_trolley_final_l507_507616


namespace count_divisors_31824_l507_507311

def is_divisor (n : ℕ) (d : ℕ) : Prop :=
  d ∣ n

theorem count_divisors_31824 :
  (Finset.filter (λ d => is_divisor 31824 d) (Finset.range 10)).card = 7 :=
by {
  sorry
}

end count_divisors_31824_l507_507311


namespace _l507_507705

noncomputable def point_line_plane_theorem (A : Type) 
    (a α : Set Type) 
    (h₁ : ∀ {A : A}, A ∈ a → a ⊄ α → A ∉ α) 
    (h₂ : ∀ {A : A}, A ∈ a → a ∈ α → A ∈ α)
    (h₃ : ∀ {A : A}, A ∉ a → a ⊂ α → A ∉ α)
    (h₄ : ∀ {A : A}, A ∈ a → a ⊂ α → A ⊆ α) : 
    ∃ n : ℕ, n = 0 := 
by
    exists 0
    sorry

end _l507_507705


namespace find_n_l507_507464

theorem find_n {x n : ℕ} (h1 : 3 * x - 4 = 8) (h2 : 7 * x - 15 = 13) (h3 : 4 * x + 2 = 18) 
  (h4 : n = 803) : 8 + (n - 1) * 5 = 4018 := by
  sorry

end find_n_l507_507464


namespace parabola_vertex_l507_507226

theorem parabola_vertex (x m n : ℝ) :
  (∀ x : ℝ, 2 * x^2 + 16 * x + 50 = 2 * (x + 4)^2 + 18) →
  m = -4 →
  n = 18 →
  ∃ (m n : ℝ), vertex(λ x, 2 * x^2 + 16 * x + 50) = (m, n) :=
by
  intro h_eq h_m h_n
  existsi (-4 : ℝ)
  existsi (18 : ℝ)
  rw [vertex]
  rw [vertex_form_of_parabola h_eq]
  exact ⟨h_m, h_n⟩
  sorry

end parabola_vertex_l507_507226


namespace perimeter_problem_l507_507774

theorem perimeter_problem
  {t k : ℝ}
  (h1 : 2 * k + t = 3 * t)
  (h2 : 6k = 18): 
    k = 6 := by
  -- Given the relations and values,
  -- we need to prove that k = 6.
  sorry

end perimeter_problem_l507_507774


namespace quarters_addition_l507_507431

def original_quarters : ℝ := 783.0
def added_quarters : ℝ := 271.0
def total_quarters : ℝ := 1054.0

theorem quarters_addition :
  original_quarters + added_quarters = total_quarters :=
by
  sorry

end quarters_addition_l507_507431


namespace tenth_permutation_is_3581_l507_507491

theorem tenth_permutation_is_3581 :
  (List.permutations [1, 3, 5, 8]).nth 9 = some [3, 5, 8, 1] :=
by
  sorry

end tenth_permutation_is_3581_l507_507491


namespace max_f_l507_507855

noncomputable def f (x : ℝ) : ℝ := (2 * Math.sin x * Math.cos x) / (1 + Math.sin x + Math.cos x)

theorem max_f : ∃ x : ℝ, f x = sqrt 2 - 1 := sorry

end max_f_l507_507855


namespace problem1_problem2_l507_507043

-- Problem 1: Prove that a^2 + b^2 + c^2 ≥ ab + ac + bc for real numbers a, b, and c.
theorem problem1 (a b c : ℝ) : a^2 + b^2 + c^2 ≥ ab + ac + bc := 
sorry

-- Problem 2: Prove that √6 + √7 > 2√2 + √5.
theorem problem2 : sqrt 6 + sqrt 7 > 2 * sqrt 2 + sqrt 5 := 
sorry

end problem1_problem2_l507_507043


namespace number_of_digits_of_2500_even_integers_l507_507125

theorem number_of_digits_of_2500_even_integers : 
  let even_integers := List.range (5000 : Nat) in
  let first_2500_even := List.filter (fun n => n % 2 = 0) even_integers in
  List.length (List.join (first_2500_even.map (fun n => n.toDigits Nat))) = 9448 :=
by
  sorry

end number_of_digits_of_2500_even_integers_l507_507125


namespace arithmetic_mean_of_fractions_l507_507941

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 3/8
  let b : ℚ := 5/9
  (a + b) / 2 = 67 / 144 := 
by
  sorry

end arithmetic_mean_of_fractions_l507_507941


namespace license_plate_count_l507_507313

-- Formalize the conditions
def is_letter (c : Char) : Prop := 'a' ≤ c ∧ c ≤ 'z'
def is_digit (c : Char) : Prop := '0' ≤ c ∧ c ≤ '9'

-- Define the main proof problem
theorem license_plate_count :
  (26 * (25 + 9) * 26 * 10 = 236600) :=
by sorry

end license_plate_count_l507_507313


namespace generate_one_fifth_from_zero_point_one_generate_rationals_between_zero_and_one_l507_507876

variable {n m : ℚ} 

/-- Starting from the number set containing 0.1, prove we can generate the number 1/5 using an averaging process. -/
theorem generate_one_fifth_from_zero_point_one :
  (∀ a b ∈ {0.1}, (a + b) / 2 ∉ {0.1}) →
  (∀ s : set ℚ, s = {0.1} ∨ ∀ p q ∈ s, (p + q) / 2 ∈ s) →
  (∃ s : set ℚ, 1/5 ∈ s ∧ {0.1} ⊆ s ∧ ∀ p q ∈ s, (p + q) / 2 ∈ s) :=
sorry

/-- Starting from the number set containing 0.1, prove we can generate any rational number between 0 and 1 using an averaging process. -/
theorem generate_rationals_between_zero_and_one :
  (∀ a b ∈ {0.1}, (a + b) / 2 ∉ {0.1}) →
  (∀ s : set ℚ, s = {0.1} ∨ ∀ p q ∈ s, (p + q) / 2 ∈ s) →
  (∀ r : ℚ, 0 < r ∧ r < 1 → ∃ s : set ℚ, r ∈ s ∧ {0.1} ⊆ s ∧ ∀ p q ∈ s, (p + q) / 2 ∈ s) :=
sorry

end generate_one_fifth_from_zero_point_one_generate_rationals_between_zero_and_one_l507_507876


namespace value_of_x_l507_507619

theorem value_of_x (a b : ℝ) : let x := (a^2 - b^2) / (2 * a) in x^2 + b^2 = (a - x)^2 := 
by
  let x := (a^2 - b^2) / (2 * a)
  have h : x^2 + b^2 = (a - x)^2 := sorry
  exact h

end value_of_x_l507_507619


namespace solution_to_system_l507_507053

theorem solution_to_system :
  ∃ (x y : ℝ), (x = 12) ∧ (y = 16) ∧ 
  (real.rpow (2 * x - y) (2 / x) = 2) ∧ 
  ((2 * x - y) * real.rpow 5 (x / 4) = 1000) :=
by {
  use [12, 16],
  split,
  { refl },
  split,
  { refl },
  split,
  {
    simp,
    sorry,
  },
  {
    simp,
    sorry,
  }
}

end solution_to_system_l507_507053


namespace eighth_term_of_geometric_sequence_l507_507648

theorem eighth_term_of_geometric_sequence :
  let a1 := (3 : ℚ)
  let r := (3 / 2 : ℚ) in
  (a1 * r^(8 - 1) = 6561 / 128) :=
by
  sorry

end eighth_term_of_geometric_sequence_l507_507648


namespace symmetric_point_polar_coordinates_l507_507776

noncomputable def symmetric_polar_point (A : ℝ × ℝ) (l : ℝ → Prop) : ℝ × ℝ :=
  (2 * real.sqrt 2, real.pi / 4)

theorem symmetric_point_polar_coordinates :
  let A := (2, real.pi / 2)
  let l  := λ (ρ θ : ℝ), ρ * real.cos θ = 1
  (symmetric_polar_point A l) = (2 * real.sqrt 2, real.pi / 4) :=
by
  sorry

end symmetric_point_polar_coordinates_l507_507776


namespace find_actual_number_of_children_l507_507144

theorem find_actual_number_of_children (B C : ℕ) (h1 : B = 2 * C) (h2 : B = 4 * (C - 420)) : C = 840 := 
by
  sorry

end find_actual_number_of_children_l507_507144


namespace average_pairs_of_consecutive_integers_l507_507504

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem average_pairs_of_consecutive_integers :
  let Ω := (binom 20 4)
  let E1 := (3 * binom 16 3)
  let E2 := (6 * binom 16 2)
  (1 / Ω : ℚ) * (E1 + E2) = 80 / 161 := by
  let Ω := (binom 20 4)
  let E1 := (3 * binom 16 3)
  let E2 := (6 * binom 16 2)
  have h1 : (1 / Ω : ℚ) = (1 : ℚ) / ↑Ω := rfl
  have h2 : (E1 + E2 : ℚ) = ↑(E1 + E2) := by norm_cast
  rw [h1, h2]
  have h3 : (1 : ℚ) / ↑Ω * (↑E1 + ↑E2) = (E1 + E2) / Ω := by
    rw [div_eq_mul_inv, ← mul_assoc, mul_inv_cancel, one_mul]
    exact Nat.cast_ne_zero.2 (Nat.choose_pos 4 20).ne'
  rw [h3]
  have h4 : (binom 20 4 : ℚ) = 4845 := by sorry
  have h5 : (3 * binom 16 3 + 6 * binom 16 2 : ℚ) = 2400 := by sorry
  rw [h4, h5]
  norm_num
  apply div_eq_of_eq_mul
  norm_num
  sorry

end average_pairs_of_consecutive_integers_l507_507504


namespace arithmetic_mean_of_fractions_l507_507945

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 3/8
  let b : ℚ := 5/9
  (a + b) / 2 = 67 / 144 := 
by
  sorry

end arithmetic_mean_of_fractions_l507_507945


namespace find_reflection_line_l507_507092

def Triangle := {P Q R : ℝ × ℝ}
def Reflection := {P' Q' R' : ℝ × ℝ}

theorem find_reflection_line (P Q R : ℝ × ℝ) (P' Q' R' : ℝ × ℝ)
  (hP : P = (-2, 3)) (hQ : Q = (3, 7)) (hR : R = (5, 1))
  (hP' : P' = (-6, 3)) (hQ' : Q' = (-9, 7)) (hR' : R' = (-11, 1)) :
  ∃ L : ℝ, L = -3 :=
by
  sorry

end find_reflection_line_l507_507092


namespace range_of_b_l507_507012

/-- Let A = {x | -1 < x < 1} and B = {x | b - 1 < x < b + 1}.
    We need to show that if A ∩ B ≠ ∅, then b is within the interval (-2, 2). -/
theorem range_of_b (b : ℝ) :
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ b - 1 < x ∧ x < b + 1) →
  -2 < b ∧ b < 2 :=
sorry

end range_of_b_l507_507012


namespace triangle_segment_lengths_l507_507183

theorem triangle_segment_lengths {base : ℝ} (h_base : base = 20) :
  ∃ seg1 seg2 seg3 seg4 : ℝ,
  seg1 = base * (1/5) ∧
  seg2 = base * (2/5) ∧
  seg3 = base * (3/5) ∧
  seg4 = base * (4/5) ∧
  seg1 = 4 ∧ seg2 = 8 ∧ seg3 = 12 ∧ seg4 = 16 :=
by
  use base * (1/5), base * (2/5), base * (3/5), base * (4/5)
  split; [refl, split; [refl, split; [refl, split; refl]]]
  exact congr_arg (λ x, base * x) (by norm_num : 1 / 5 = 0.2 : ℝ)
  exact congr_arg (λ x, base * x) (by norm_num : 2 / 5 = 0.4 : ℝ)
  exact congr_arg (λ x, base * x) (by norm_num : 3 / 5 = 0.6 : ℝ)
  exact congr_arg (λ x, base * x) (by norm_num : 4 / 5 = 0.8 : ℝ)
  sorry

end triangle_segment_lengths_l507_507183


namespace cyclic_sequence_u_16_eq_a_l507_507995

-- Sequence definition and recurrence relation
def cyclic_sequence (u : ℕ → ℝ) (a : ℝ) : Prop :=
  u 1 = a ∧ ∀ n : ℕ, u (n + 1) = -1 / (u n + 1)

-- Proof that u_{16} = a under given conditions
theorem cyclic_sequence_u_16_eq_a (a : ℝ) (h : 0 < a) : ∃ (u : ℕ → ℝ), cyclic_sequence u a ∧ u 16 = a :=
by
  sorry

end cyclic_sequence_u_16_eq_a_l507_507995


namespace total_distance_walked_l507_507197

-- Define the given conditions
def walks_to_work_days := 5
def walks_dog_days := 7
def walks_to_friend_days := 1
def walks_to_store_days := 2

def distance_to_work := 6
def distance_dog_walk := 2
def distance_to_friend := 1
def distance_to_store := 3

-- The proof statement
theorem total_distance_walked :
  (walks_to_work_days * (distance_to_work * 2)) +
  (walks_dog_days * (distance_dog_walk * 2)) +
  (walks_to_friend_days * distance_to_friend) +
  (walks_to_store_days * distance_to_store) = 95 := 
sorry

end total_distance_walked_l507_507197


namespace arithmetic_mean_of_fractions_l507_507922

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 := 
by 
  sorry

end arithmetic_mean_of_fractions_l507_507922


namespace problem_statement_l507_507390

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then log 2 (1 - x) else 4^x

theorem problem_statement : f (-3) + f (log 2 3) = 11 :=
by 
  have h1 : f (-3) = log 2 (1 - (-3)), from if_pos (by norm_num),
  have h2 : log 2 4 = 2, from by sorry,
  have h3 : f (log 2 3) = 4 ^ (log 2 3), from if_neg (by norm_num),
  have h4 : (4 : ℝ) = 2^2, from by norm_num,
  have h5 : (4 : ℝ) ^ (log 2 3) = (2^2) ^ (log 2 3), from congr_arg (^ (log 2 3)) (by sorry),
  have h6 : (2^2) ^ (log 2 3) = 2^(2 * log 2 3), from by sorry,
  have h7 : 2^(2 * log 2 3) = 2^(log 2 (3^2)), from by sorry,
  have h8 : 2^(log 2 (3^2)) = 3^2, from by sorry,
  have h9 : 3^2 = 9, from by norm_num,
  show f (-3) + f (log 2 3) = 11, from by sorry

end problem_statement_l507_507390


namespace geometric_sequence_third_sixth_term_l507_507484

theorem geometric_sequence_third_sixth_term (a r : ℝ) 
  (h3 : a * r^2 = 18) 
  (h6 : a * r^5 = 162) : 
  a = 2 ∧ r = 3 := 
sorry

end geometric_sequence_third_sixth_term_l507_507484


namespace sum_of_coefficients_y_l507_507562

theorem sum_of_coefficients_y (x y : ℕ) : 
  let expr := (2 * x + 3 * y + 4) * (3 * x + 5 * y + 6)
  in 
  let expanded_expr := (6 * x^2 + 19 * x * y + 24 * x + 15 * y^2 + 38 * y + 24)
  in 
  (expanded_expr.terms.filter (λ term, term.contains y)).sum_coeffs = 72 := 
by
  sorry

end sum_of_coefficients_y_l507_507562


namespace problem_quadrant_l507_507717

def complex_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem problem_quadrant (z : ℂ) (hz : z = 2 * Complex.i / (1 - Complex.i)) :
  complex_in_second_quadrant z :=
by
  unfold complex_in_second_quadrant
  rw [hz]
  sorry

end problem_quadrant_l507_507717


namespace constant_term_expansion_l507_507670

theorem constant_term_expansion :
  let expr := (λ (x : ℂ), (sqrt x + 2) * (1 / sqrt x - 1) ^ 5)
  constant_term_in_expansion expr = 3 :=
by
  sorry

end constant_term_expansion_l507_507670


namespace exists_directed_cycle_face_l507_507816

structure Polyhedron where
  vertices : Set
  edges : Set (vertices × vertices)
  faces : Set (Set (vertices × vertices))
  convex : Bool

def directed_edges (edges : Set (α × α)) : Prop :=
  ∀ v ∈ vertices, ∃ u ∈ vertices, (u, v) ∈ edges ∧ ∃ w ∈ vertices, (v, w) ∈ edges

theorem exists_directed_cycle_face (P : Polyhedron) (h_convex : P.convex = true) (h_dir : directed_edges P.edges) :
  ∃ face ∈ P.faces, ∃ cycle : list (P.vertices × P.vertices), (∀ (e ∈ cycle), e ∈ face) ∧ is_cycle cycle :=
  by
  sorry

end exists_directed_cycle_face_l507_507816


namespace perimeter_triangle_MUV_l507_507495

open Real

theorem perimeter_triangle_MUV (M D E F U V G : Point) (circle : Circle) (r : ℝ) :
  TangentFrom M D circle → TangentFrom M E circle → TangentFrom M F circle →
  TangentIntersection ME U circle → TangentIntersection MF V circle →
  segment M D = 25 ∧ segment M E = 25 ∧ segment M F = 30 →
  perimeter (Triangle.mk M U V) = 55 :=
by
  sorry

end perimeter_triangle_MUV_l507_507495


namespace probability_third_term_is_three_l507_507382

noncomputable def permutations_filtered (n : ℕ) (excluded : list ℕ) : Finset (Fin n → Fin n) :=
  (Finset.univ.filter (λ x : Fin n → Fin n, (x 0) ∉ excluded))

def favorable_perm_count (n : ℕ) (excluded : list ℕ) : ℕ :=
  permutations_filtered n excluded.filter (λ x, x 2 = 2).card

theorem probability_third_term_is_three (n : ℕ) (excluded : list ℕ) :
  let total_perm := (n.factorial - 2 * (n-1).factorial)
  let favorable_perm := favorable_perm_count n excluded
  ∃ (c d : ℕ), (favorable_perm / total_perm = c / d) ∧ c + d = 23 :=
sorry

end probability_third_term_is_three_l507_507382


namespace smallest_integer_with_eight_factors_l507_507550

theorem smallest_integer_with_eight_factors :
  ∃ n : ℕ, (∀ m : ℕ, (∃ k : ℕ, m has_factors k ∧ k = 8) → m ≥ n) ∧ n = 24 :=
by
  sorry

end smallest_integer_with_eight_factors_l507_507550


namespace circle_equation_center_at_1_2_passing_through_origin_l507_507067

theorem circle_equation_center_at_1_2_passing_through_origin :
  ∃ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = 5 ∧
                (0 - 1)^2 + (0 - 2)^2 = 5 :=
by
  sorry

end circle_equation_center_at_1_2_passing_through_origin_l507_507067


namespace problem_inequality_l507_507798

theorem problem_inequality (n : ℕ) 
  (a : Fin n → ℝ) 
  (h1 : 2 ≤ n) 
  (h2 : ∀ i, 0 ≤ a i ∧ a i ≤ (Real.pi / 2)) : 
  (1/n * ∑ i in (Finset.range n), 1 / (1 + Real.sin (a i))) * 
  (1 + ∏ i in (Finset.range n), (Real.sin (a i))^(1/n.toNat)) ≤ 1 := 
sorry

end problem_inequality_l507_507798


namespace inequality_proof_l507_507034

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a^3 + b^3 = 2) :
  (1 / a) + (1 / b) ≥ 2 * (a^2 - a + 1) * (b^2 - b + 1) := 
by
  sorry

end inequality_proof_l507_507034


namespace concurrency_AN_FP_EM_l507_507967

variables {A B C D E F M N P : Type} [EquilateralTriangle A B C]
variables (midpoint_D : is_midpoint D B C)
variables (midpoint_E : is_midpoint E C A)
variables (midpoint_F : is_midpoint F A B)
variables (midpoint_M : is_midpoint M B F)
variables (midpoint_N : is_midpoint N D F)
variables (midpoint_P : is_midpoint P D C)

theorem concurrency_AN_FP_EM : are_concurrent AN FP EM := sorry

end concurrency_AN_FP_EM_l507_507967


namespace reduce_to_one_l507_507421

theorem reduce_to_one (n : ℕ) : ∃ k, (k = 1) :=
by
  sorry

end reduce_to_one_l507_507421


namespace roots_sum_of_quadratic_l507_507861

theorem roots_sum_of_quadratic :
  ∀ x1 x2 : ℝ, (Polynomial.eval x1 (Polynomial.X ^ 2 + 2 * Polynomial.X - 1) = 0) →
              (Polynomial.eval x2 (Polynomial.X ^ 2 + 2 * Polynomial.X - 1) = 0) →
              x1 + x2 = -2 :=
by
  intros x1 x2 h1 h2
  sorry

end roots_sum_of_quadratic_l507_507861


namespace perfect_score_l507_507600

theorem perfect_score (P : ℕ) (h : 3 * P = 63) : P = 21 :=
by
  -- Proof to be provided
  sorry

end perfect_score_l507_507600


namespace books_ratio_l507_507810

/-- Define the total number of books read by each individual, and show that the ratio of the number of books Kelcie has read to the number of books Megan has read is 1:4. -/
theorem books_ratio :
  ∃ (K : ℕ), let Greg_books := 2 * K + 9 in
  let total_books := 32 + K + Greg_books in 
  total_books = 65 ∧ (K : ℚ) / 32 = 1 / 4 :=
begin
  sorry
end

end books_ratio_l507_507810


namespace problem_f_f_inv_e_l507_507281

noncomputable def f : ℝ → ℝ :=
  λ x, if h : x > 0 then Real.log x else Real.exp (x + 1) - 2

theorem problem_f_f_inv_e : f (f (1 / Real.exp 1)) = -1 := by
  sorry

end problem_f_f_inv_e_l507_507281


namespace slopes_product_hyperbola_constant_l507_507783

theorem slopes_product_hyperbola_constant (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∀ (M N P : ℝ × ℝ),
    (M.1^2 / a^2 - M.2^2 / b^2 = 1) →
    (N.1^2 / a^2 - N.2^2 / b^2 = 1) →
    (P.1^2 / a^2 - P.2^2 / b^2 = 1) →
    (N = (-M.1, -M.2)) →
    let kPM := (P.2 - M.2) / (P.1 - M.1)
    let kPN := (P.2 + N.2) / (P.1 + N.1)
    in kPM * kPN = (b^2 / a^2) :=
by
  intros M N P hM hN hP hSymm
  let kPM := (P.2 - M.2) / (P.1 - M.1)
  let kPN := (P.2 + N.2) / (P.1 + N.1)
  have res : kPM * kPN = (b^2 / a^2) := sorry
  exact res

end slopes_product_hyperbola_constant_l507_507783


namespace angle_between_vectors_eq_90_l507_507987

variables {a b : ℝ}

theorem angle_between_vectors_eq_90 (a b : ℝ) (h : |a + b| = |a - b|) : (angle_between a b) = 90 := 
sorry

end angle_between_vectors_eq_90_l507_507987


namespace frustum_volume_correct_l507_507620

-- Definitions of pyramids and their properties
structure Pyramid :=
  (base_edge : ℕ)
  (altitude : ℕ)
  (volume : ℚ)

-- Definition of the original pyramid and smaller pyramid
def original_pyramid : Pyramid := {
  base_edge := 20,
  altitude := 10,
  volume := (1 / 3 : ℚ) * (20 ^ 2) * 10
}

def smaller_pyramid : Pyramid := {
  base_edge := 8,
  altitude := 5,
  volume := (1 / 3 : ℚ) * (8 ^ 2) * 5
}

-- Definition and calculation of the volume of the frustum 
def volume_frustum (p1 p2 : Pyramid) : ℚ :=
  p1.volume - p2.volume

-- Main theorem to be proved
theorem frustum_volume_correct :
  volume_frustum original_pyramid smaller_pyramid = 992 := by
  sorry

end frustum_volume_correct_l507_507620


namespace least_value_expression_l507_507507

open Real

theorem least_value_expression : ∃ x : ℝ, (λ x, (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2024) x = 2008 :=
sorry

end least_value_expression_l507_507507


namespace solve_system_l507_507055

noncomputable def proof_problem (x y : ℝ) : Prop :=
  (x = 0) ∧ (∃ k : ℤ, y = (π / 2) + π * k) →
  (x^2 + 4 * sin y ^ 2 - 4 = 0) ∧ (cos x - 2 * cos y ^ 2 - 1 = 0)

-- Theorem statement
theorem solve_system : proof_problem 0 ((π / 2) + π * k) := 
by
  intro x y,
  intro h,
  cases h with hx hy,
  rw [hx, hy],
  split,
  { -- Prove x^2 + 4 * sin y ^ 2 - 4 = 0
    sorry },
  { -- Prove cos x - 2 * cos y ^ 2 - 1 = 0
    sorry }

end solve_system_l507_507055


namespace find_angle_C_l507_507779

theorem find_angle_C (a b c : ℝ) (h : a ^ 2 + b ^ 2 - c ^ 2 + a * b = 0) : 
  C = 2 * pi / 3 := 
sorry

end find_angle_C_l507_507779


namespace problem_solution_l507_507280

section
  variables {q : ℝ} (h_q_ne_one : q ≠ 1)
             {a b : ℕ → ℝ}
             (h_a1 : a 1 = 1)
             (h_b1 : a 2 = b 1)
             (h_arith_geom_diff : a 3 - a 1 = b 2 - b 1)
             (h_prod_eq : a 2 * b 2 = b 3)
             (h_an : ∀ n : ℕ, a n = n)
             (h_bn : ∀ n : ℕ, b n = 2^n)
             
  noncomputable def S (n : ℕ) : ℝ :=
    ∑ i in finset.range n, a (i + 1) * b (n - i)

  theorem problem_solution (n : ℕ) :
    S n = 2^(n+2) - 2 * n - 4 :=
  sorry
end

end problem_solution_l507_507280


namespace sum_of_roots_l507_507108

theorem sum_of_roots : 
  ( ∀ x : ℝ, x^2 - 7*x + 10 = 0 → x = 2 ∨ x = 5 ) → 
  ( 2 + 5 = 7 ) := 
by
  sorry

end sum_of_roots_l507_507108


namespace cosecant_pi_over_six_l507_507206

/-- The mathematical statement -/
theorem cosecant_pi_over_six : Real.csc (Real.pi / 6) = 2 := by
  sorry

end cosecant_pi_over_six_l507_507206


namespace rectangle_y_value_l507_507455

theorem rectangle_y_value
  (E : (ℝ × ℝ)) (F : (ℝ × ℝ)) (G : (ℝ × ℝ)) (H : (ℝ × ℝ))
  (hE : E = (0, 0)) (hF : F = (0, 5)) (hG : ∃ y : ℝ, G = (y, 5))
  (hH : ∃ y : ℝ, H = (y, 0)) (area : ℝ) (h_area : area = 35)
  (hy_pos : ∃ y : ℝ, y > 0)
  : ∃ y : ℝ, y = 7 :=
by
  sorry

end rectangle_y_value_l507_507455


namespace first_player_winning_strategy_l507_507337

-- Define the game state and rules
def Game : Type :=
  ℕ -- Number of sticks / oranges left in the basket

def initial_state : Game := 100

-- Player action: take between 1 and 5 sticks
def valid_action (n : ℕ) (a : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 5 ∧ a ≤ n

-- Define the strategy for the first player
def first_player_strategy (opponent_action : ℕ) (n : ℕ) : ℕ :=
  if n = initial_state then 4 else (6 - opponent_action)

-- Winning condition: the player who takes the last stick wins
def winning_condition (n : ℕ) : Prop :=
  n = 0

-- Proof statement
theorem first_player_winning_strategy :
  ∃ strategy : ℕ → ℕ → ℕ, (∀ n opponent_action, valid_action n opponent_action → 
  (valid_action n (strategy opponent_action n)) ∧ 
  (winning_condition (n - (strategy opponent_action n)))) :=
by
  sorry

end first_player_winning_strategy_l507_507337


namespace percentage_in_range_80_to_100_l507_507161

def dataset : List ℕ := [50, 77, 83, 91, 93, 101, 87, 102, 111, 63, 117, 89, 121, 130, 133, 146, 88, 158, 177, 188]

def inRangeCount (data : List ℕ) (lower upper : ℕ) : ℕ :=
  List.length (List.filter (λ x => lower ≤ x ∧ x ≤ upper) data)

def dataCount (data : List ℕ) : ℕ := List.length data

theorem percentage_in_range_80_to_100 :
  let in_range := inRangeCount dataset 80 100
  let total := dataCount dataset
  in (in_range.toFloat / total.toFloat) * 100 = 30 :=
by
  let in_range := inRangeCount dataset 80 100
  let total := dataCount dataset
  have : in_range = 6 := sorry
  have : total = 20 := sorry
  have : (6.0 / 20.0) * 100 = 30 := by norm_num
  exact this

end percentage_in_range_80_to_100_l507_507161


namespace tv_price_net_change_l507_507325

theorem tv_price_net_change (P : ℝ) : 
  let P1 := P * 0.825,
      P2 := P1 * 1.25,
      P3 := P2 * 0.97,
      P4 := P3 * 1.427 in
  ((P4 - P) / P) * 100 = 42.7 :=
by
  unfold P1 P2 P3 P4
  -- Simplify expressions and show the equivalence
  unravel_le P4
  nlinarith

end tv_price_net_change_l507_507325


namespace DEF_area_l507_507091

-- Definitions of the triangle and its properties
def is_isosceles_right_triangle (DEF : Triangle) : Prop :=
  (DEF.angleD = 90) ∧ (DEF.sideDE = DEF.sideDF)

-- Definitions of the lengths and the area formula
def length_DE (DEF : Triangle) : ℝ := 12
def area_isosceles_right (a : ℝ) : ℝ := 0.5 * a * a

-- The statement to prove
theorem DEF_area (DEF : Triangle)
  (h1 : is_isosceles_right_triangle DEF)
  (h2 : length_DE DEF = 12) :
  Triangle.area DEF = 72 :=
by
  sorry

end DEF_area_l507_507091


namespace final_position_east_6_average_speed_30_l507_507058

-- Define the distances driven for the eight batches
def distances : List ℤ := [+8, -6, +3, -4, +8, -4, +4, -3]

-- Define the times in the form of start and end times
def start_time := (8, 0)  -- 8:00 AM
def end_time := (9, 20)  -- 9:20 AM

-- Prove that after the last batch Li is 6 kilometers to the east of the starting point
theorem final_position_east_6 :
  distances.foldl (λ acc x => acc + x) 0 = 6 :=
sorry

-- Define the total distance traveled (absolute value)
def total_distance := List.sum (distances.map Int.natAbs)

-- Define the total time in hours (converted from 1 hour and 20 minutes)
def total_time_hours : Rat := 4 / 3

-- Prove that Li's average speed is 30 kilometers per hour
theorem average_speed_30 :
  (total_distance : Rat) / total_time_hours = 30 :=
sorry

end final_position_east_6_average_speed_30_l507_507058


namespace parallelogram_area_ratio_l507_507598

theorem parallelogram_area_ratio (
  AB CD BC AD AP CQ BP DQ: ℝ)
  (h1 : AB = 13)
  (h2 : CD = 13)
  (h3 : BC = 15)
  (h4 : AD = 15)
  (h5 : AP = 10 / 3)
  (h6 : CQ = 10 / 3)
  (h7 : BP = 29 / 3)
  (h8 : DQ = 29 / 3)
  : ((area_APDQ / area_BPCQ) = 19) :=
sorry

end parallelogram_area_ratio_l507_507598


namespace perimeter_of_square_l507_507610

-- Defining the square with area
structure Square where
  side_length : ℝ
  area : ℝ

-- Defining a constant square with given area 625
def givenSquare : Square := 
  { side_length := 25, -- will square root the area of 625
    area := 625 }

-- Defining the function to calculate the perimeter of the square
noncomputable def perimeter (s : Square) : ℝ :=
  4 * s.side_length

-- The theorem stating that the perimeter of the given square with area 625 is 100
theorem perimeter_of_square : perimeter givenSquare = 100 := 
sorry

end perimeter_of_square_l507_507610


namespace arithmetic_mean_of_38_and_59_is_67_over_144_l507_507935

theorem arithmetic_mean_of_38_and_59_is_67_over_144 :
  (3 / 8 + 5 / 9) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_38_and_59_is_67_over_144_l507_507935


namespace equilateral_triangle_side_length_l507_507439

-- Geometry definitions to describe the problem
variables {Point : Type*} [metric_space Point]

structure Square (A B C D : Point) :=
  (side_length : ℝ)
  (isosceles_right_triangle : ∀ (P Q R : Point), P ≠ Q → Q ≠ R → R ≠ P → dist P Q = side_length → dist Q R = side_length)

structure PointOnLine (P Q : Point) :=
  (E F : Point)
  (on_PQ : ∃ (t : ℝ), t ∈ (Icc 0 1) ∧ E = t • Q + (1 - t) • P)

structure RightTriangle (A E F : Point) :=
  (angle_A : real.angle)
  (angle_E : real.angle)
  (angle_F : real.angle)
  (angle_A_def : angle_A = real.angle.pi/6)
  (angle_E_def : angle_E = real.angle.pi/2)
  (angle_F_def : angle_F = real.angle.pi/3)

def equilateral_triangle_side {A B C D E F : Point} (s : ℝ) [RightTriangle A E F] [Square A B C D] [PointOnLine B C] [PointOnLine C D] :=
  s = (real.sqrt 3) / 2

/-- Given the initial conditions and the property of the triangle, this function checks 
whether the side length of the equilateral triangle is correct -/
theorem equilateral_triangle_side_length {A B C D E F : Point} [metric_space Point]
  [Square A B C D]
  [RightTriangle A E F]
  [PointOnLine B C]
  [PointOnLine C D] 
  : equilateral_triangle_side (real.sqrt 3 / 2) :=
begin
  sorry
end

end equilateral_triangle_side_length_l507_507439


namespace total_sales_l507_507829

-- Define sales of Robyn and Lucy
def Robyn_sales : Nat := 47
def Lucy_sales : Nat := 29

-- Prove total sales
theorem total_sales : Robyn_sales + Lucy_sales = 76 :=
by
  sorry

end total_sales_l507_507829


namespace PQRS_all_acute_l507_507587

theorem PQRS_all_acute 
  (A B C D P Q R S : Point)
  (hABCD: ConvexQuadrilateral A B C D)
  (hincircle: InscribedCircle A B C D P Q R S) :
  AllAcuteQuadrilateral P Q R S := 
sorry

end PQRS_all_acute_l507_507587


namespace min_cos_C_l507_507711

theorem min_cos_C (a b c : ℝ) (A B C : ℝ) (h1 : a^2 + b^2 = (5 / 2) * c^2) 
  (h2 : ∃ (A B C : ℝ), a ≠ b ∧ 
    c = (a ^ 2 + b ^ 2 - 2 * a * b * (Real.cos C))) : 
  ∃ (C : ℝ), Real.cos C = 3 / 5 :=
by
  sorry

end min_cos_C_l507_507711


namespace arithmetic_mean_of_fractions_l507_507942

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 3/8
  let b : ℚ := 5/9
  (a + b) / 2 = 67 / 144 := 
by
  sorry

end arithmetic_mean_of_fractions_l507_507942


namespace inequality_solution_set_empty_l507_507326

theorem inequality_solution_set_empty (a : ℝ) :
  (∀ x : ℝ, ¬ (|x - 2| + |x + 3| < a)) → a ≤ 5 :=
by sorry

end inequality_solution_set_empty_l507_507326


namespace total_digits_2500_is_9449_l507_507131

def nth_even (n : ℕ) : ℕ := 2 * n

def count_digits_in_range (start : ℕ) (stop : ℕ) : ℕ :=
  (stop - start) / 2 + 1

def total_digits (n : ℕ) : ℕ :=
  let one_digit := 4
  let two_digit := count_digits_in_range 10 98
  let three_digit := count_digits_in_range 100 998
  let four_digit := count_digits_in_range 1000 4998
  let five_digit := 1
  one_digit * 1 +
  two_digit * 2 +
  (three_digit * 3) +
  (four_digit * 4) +
  (five_digit * 5)

theorem total_digits_2500_is_9449 : total_digits 2500 = 9449 := by
  sorry

end total_digits_2500_is_9449_l507_507131


namespace expected_value_winnings_l507_507618

def probability_heads : ℚ := 2 / 5
def probability_tails : ℚ := 3 / 5
def winnings_heads : ℚ := 4
def loss_tails : ℚ := -3

theorem expected_value_winnings : 
  (probability_heads * winnings_heads + probability_tails * loss_tails) = -1 / 5 := 
by
  -- calculation steps and proof would go here
  sorry

end expected_value_winnings_l507_507618


namespace arithmetic_mean_of_fractions_l507_507919

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 := 
by 
  sorry

end arithmetic_mean_of_fractions_l507_507919


namespace car_speed_l507_507583

-- Define the given conditions
def distance := 1280.0
def time_in_hours := 46.0 / 13.0
def expected_speed := 361.54

-- State the theorem to prove the speed of the car
theorem car_speed :
  abs (distance / time_in_hours - expected_speed) < 0.01 :=
by
  sorry

end car_speed_l507_507583


namespace total_students_in_school_l507_507075

-- given conditions
def num_buses := 95
def bus_capacity := 118
def bus_efficiency := 0.9
def attendance_rate := 0.8

-- proving the number of students in the school
theorem total_students_in_school : 
  let total_students := (95 * (bus_capacity * bus_efficiency : Nat) : Nat) / attendance_rate
  total_students = 12588 := by
  sorry

end total_students_in_school_l507_507075


namespace probability_grid_sums_odd_exceed_15_l507_507471

theorem probability_grid_sums_odd_exceed_15 
  (numbers : Finset ℕ) 
  (h : numbers = {2, 3, 4, 5, 6, 7, 8, 9, 10}) 
  (grid : Matrix (Fin 3) (Fin 3) ℕ)
  (h_disjoint : ∀ i j, grid i j ∈ numbers) 
  (h_unique : ∀ i j k l, grid i j = grid k l → i = k ∧ j = l) :
  probability 
    (∀ i, odd (∑ j, grid i j) ∧ (∑ j, grid i j > 15) ∧
          ∀ j, odd (∑ i, grid i j) ∧ (∑ i, grid i j > 15) ∧
          odd (∑ i, grid i i) ∧ (∑ i, grid i i > 15) ∧
          odd (∑ i, grid i (2 - i)) ∧ (∑ i, grid i (2 - i) > 15)) = 1 / 210 := 
sorry

end probability_grid_sums_odd_exceed_15_l507_507471


namespace find_a_for_perfect_square_trinomial_l507_507713

theorem find_a_for_perfect_square_trinomial (a : ℝ) :
  (∃ b : ℝ, x^2 - 8*x + a = (x - b)^2) ↔ a = 16 :=
by sorry

end find_a_for_perfect_square_trinomial_l507_507713


namespace abs_equality_l507_507566

theorem abs_equality : abs 20 = 2 * abs 5 := 
  sorry

end abs_equality_l507_507566


namespace monotonicity_of_f_range_of_a_for_real_roots_of_equation_l507_507384

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 1 + log a (x - 3)

theorem monotonicity_of_f (a : ℝ) (h_pos : a > 0) (h_not_one : a ≠ 1) :
  (∀ x1 x2, x1 < x2 ∧ x2 < -5 → f(a) x1 < f(a) x2 ↔ a > 1) ∧
  (∀ x1 x2, x1 < x2 ∧ x2 < -5 → f(a) x1 > f(a) x2 ↔ 0 < a ∧ a < 1) :=
sorry

theorem range_of_a_for_real_roots_of_equation (a : ℝ) (h_pos : a > 0) (h_not_one : a ≠ 1) :
  (∀ x, f(a) x = g(a) x → 0 < a ∧ a ≤ 16) :=
sorry

end monotonicity_of_f_range_of_a_for_real_roots_of_equation_l507_507384


namespace total_digits_first_2500_even_integers_l507_507116

theorem total_digits_first_2500_even_integers :
  let even_nums := List.range' 2 5000 (λ n, 2*n)  -- List of the first 2500 even integers
  let one_digit_nums := even_nums.filter (λ n, n < 10)
  let two_digit_nums := even_nums.filter (λ n, 10 ≤ n ∧ n < 100)
  let three_digit_nums := even_nums.filter (λ n, 100 ≤ n ∧ n < 1000)
  let four_digit_nums := even_nums.filter (λ n, 1000 ≤ n ∧ n ≤ 5000)
  let sum_digits := one_digit_nums.length * 1 + two_digit_nums.length * 2 + three_digit_nums.length * 3 + four_digit_nums.length * 4
  in sum_digits = 9448 := by sorry

end total_digits_first_2500_even_integers_l507_507116


namespace total_triangles_in_triangular_grid_l507_507741

theorem total_triangles_in_triangular_grid : 
  let small_triangles := 10
  let medium_triangles := 5
  let large_triangles := 1
  small_triangles + medium_triangles + large_triangles = 16 := 
by
  have small_triangles := 4 + 3 + 2 + 1
  have medium_triangles := 3 + 2
  have large_triangles := 1
  simp [small_triangles, medium_triangles, large_triangles]
  sorry

end total_triangles_in_triangular_grid_l507_507741


namespace trig_ratios_l507_507443

-- Define the problem
theorem trig_ratios (x : ℝ) (p q : ℕ) 
  (hsec : Real.sec x + Real.tan x = 15 / 4) 
  (hcsc : Real.csc x + Real.cot x = p / q) 
  (coprime : Nat.coprime p q) 
  : p + q = 73 :=
sorry

end trig_ratios_l507_507443


namespace complex_point_in_fourth_quadrant_l507_507743

def i := Complex.I

def z : Complex := (1 + i) / (1 + 2 * i)

theorem complex_point_in_fourth_quadrant : 
  Re z > 0 ∧ Im z < 0 :=
by
  sorry

end complex_point_in_fourth_quadrant_l507_507743


namespace train_speed_is_72_kmh_l507_507614

noncomputable def train_length : ℝ := 110
noncomputable def bridge_length : ℝ := 175
noncomputable def crossing_time : ℝ := 14.248860091192705

theorem train_speed_is_72_kmh :
  (train_length + bridge_length) / crossing_time * 3.6 = 72 := by
  sorry

end train_speed_is_72_kmh_l507_507614


namespace simplify_expression_l507_507047

theorem simplify_expression (m : ℝ) (h : m ≠ 0) : ((1 / (3 * m)) ^ (-3)) * ((-m) ^ 4) = 27 * (m ^ 7) := 
by
  sorry

end simplify_expression_l507_507047


namespace xy_system_sol_l507_507314

theorem xy_system_sol (x y : ℝ) (h1 : x * y = 8) (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^3 + y^3 = 416000 / 729 :=
by
  sorry

end xy_system_sol_l507_507314


namespace linda_original_amount_l507_507747

theorem linda_original_amount (L L2 : ℕ) 
  (h1 : L = 20) 
  (h2 : L - 5 = L2) : 
  L2 + 5 = 15 := 
sorry

end linda_original_amount_l507_507747


namespace smallest_integer_with_eight_factors_l507_507543

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, 
  ∀ m : ℕ, (∀ p : ℕ, ∃ k : ℕ, m = p^k → (k + 1) * (p + 1) = 8) → (n ≤ m) ∧ 
  (∀ d : ℕ, d ∣ n → d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 24) :=
sorry

end smallest_integer_with_eight_factors_l507_507543


namespace dot_product_collinear_A_Q_N_l507_507149

noncomputable theory

-- Definition of the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Given any point P on the ellipse
variables {x₀ y₀ : ℝ}
axiom point_on_ellipse : ellipse x₀ y₀

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Non-points A and B condition
axiom not_A_or_B : (x₀, y₀) ≠ A ∧ (x₀, y₀) ≠ B

-- Intersection points M and N
def M : ℝ × ℝ := (3, 5 * y₀ / (x₀ + 2))
def N : ℝ × ℝ := (3, y₀ / (x₀ - 2))

-- Vectors FM and FN with respect to point F(1,0)
def F : ℝ × ℝ := (1, 0)
def FM : ℝ × ℝ := (M.1 - F.1, M.2 - F.2)
def FN : ℝ × ℝ := (N.1 - F.1, N.2 - F.2)

-- Q is the intersection of the line MB with the ellipse
def intersects_ellipse (x y : ℝ) : Prop := ellipse x y

axiom Q_on_ellipse : ∃ (s t : ℝ), intersects_ellipse s t ∧ (s, t) ≠ A ∧ (s, t) ≠ B
def Q := classical.some Q_on_ellipse

-- Part (1): Proof that the dot product is 1/4
theorem dot_product (FM FN : ℝ × ℝ) : (FM.1 * FN.1 + FM.2 * FN.2) = 1/4 :=
sorry

-- Part (2): Proof of collinearity of points A, Q, and N
theorem collinear_A_Q_N : collinear ℝ {A, Q, N} :=
sorry

end dot_product_collinear_A_Q_N_l507_507149


namespace chip_draw_probability_l507_507334

theorem chip_draw_probability :
  let total_chips := 7 + 8 + 6 + 3 in
  let p_blue_then_diff := (7 / total_chips) * (17 / total_chips) in
  let p_red_then_diff := (8 / total_chips) * (16 / total_chips) in
  let p_yellow_then_diff := (6 / total_chips) * (18 / total_chips) in
  let p_green_then_diff := (3 / total_chips) * (21 / total_chips) in
  p_blue_then_diff + p_red_then_diff + p_yellow_then_diff + p_green_then_diff = 537 / 576 :=
by
  sorry

end chip_draw_probability_l507_507334


namespace arithmetic_mean_of_fractions_l507_507912

theorem arithmetic_mean_of_fractions : 
  (3 : ℚ) / 8 + (5 : ℚ) / 9 = 2 * (67 : ℚ) / 144 :=
by {
  rw [(show (3 : ℚ) / 8 + (5 : ℚ) / 9 = (27 : ℚ) / 72 + (40 : ℚ) / 72, by sorry),
      (show (27 : ℚ) / 72 + (40 : ℚ) / 72 = (67 : ℚ) / 72, by sorry),
      (show (2 : ℚ) * (67 : ℚ) / 144 = (67 : ℚ) / 72, by sorry)]
}

end arithmetic_mean_of_fractions_l507_507912


namespace evaluate_expression_l507_507954

theorem evaluate_expression : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 = -1 := 
  sorry

end evaluate_expression_l507_507954


namespace greatest_product_of_slopes_l507_507885

theorem greatest_product_of_slopes (θ : ℝ) (hθ : θ = real.pi / 6) (m m₁ m₂ : ℝ) 
(h₁ : m₁ = m) (h₂ : m₂ = 4 * m) :
  |(θ.tan)| = |(4 * m - m) / (1 + 4 * m * m)| → 
  (m₁ * m₂) = (4 * (m^2)) := 
begin
  sorry
end

end greatest_product_of_slopes_l507_507885


namespace num_chairs_l507_507414

variable (C : Nat)
variable (tables_sticks : Nat := 6 * 9)
variable (stools_sticks : Nat := 4 * 2)
variable (total_sticks_needed : Nat := 34 * 5)
variable (total_sticks_chairs : Nat := 6 * C)

theorem num_chairs (h : total_sticks_chairs + tables_sticks + stools_sticks = total_sticks_needed) : C = 18 := 
by sorry

end num_chairs_l507_507414


namespace range_of_a_l507_507283

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then (a-2) * x else (1/2)^x - 1

theorem range_of_a (a : ℝ) :
  (∀ (x1 x2 : ℝ), x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) →
  a ≤ 13 / 8 :=
sorry

end range_of_a_l507_507283


namespace situps_together_l507_507309

theorem situps_together (hani_rate diana_rate : ℕ) (diana_situps diana_time hani_situps total_situps : ℕ)
  (h1 : hani_rate = diana_rate + 3)
  (h2 : diana_rate = 4)
  (h3 : diana_situps = 40)
  (h4 : diana_time = diana_situps / diana_rate)
  (h5 : hani_situps = hani_rate * diana_time)
  (h6 : total_situps = diana_situps + hani_situps) : 
  total_situps = 110 :=
sorry

end situps_together_l507_507309


namespace tangent_line_at_x_2_range_of_m_for_three_roots_l507_507286

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

/-
Part 1: Proving the tangent line equation at x = 2
-/
theorem tangent_line_at_x_2 : ∃ k b, (k = 12) ∧ (b = -17) ∧ 
  (∀ x, 12 * x - f 2 - 17 = 0) :=
by
  sorry

/-
Part 2: Proving the range of m for three distinct real roots
-/
theorem range_of_m_for_three_roots (m : ℝ) :
  (∃ x1 x2 x3, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 + m = 0 ∧ f x2 + m = 0 ∧ f x3 + m = 0) ↔ 
  -3 < m ∧ m < -2 :=
by
  sorry

end tangent_line_at_x_2_range_of_m_for_three_roots_l507_507286


namespace inequality_sqrt_l507_507031

theorem inequality_sqrt (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
by
  sorry

end inequality_sqrt_l507_507031


namespace perpendicular_lines_l507_507269

theorem perpendicular_lines (a : ℝ) :
  (∃ x y : ℝ, ax + 2 * y + 6 = 0) ∧ (∃ x y : ℝ, x + (a - 1) * y + a^2 - 1 = 0) ∧ (∀ m1 m2 : ℝ, m1 * m2 = -1) →
  a = 2/3 :=
by
  sorry

end perpendicular_lines_l507_507269


namespace exists_prime_factor_above_20_l507_507403

theorem exists_prime_factor_above_20 
  (d : ℕ → ℤ)
  (distinct_d : ∀ i j : ℕ, i ≠ j → d i ≠ d j)
  (n : ℕ)
  (P : ℤ → ℤ := λ x, ∏ i in finset.range n, (x + d i)) :
  ∃ N : ℤ, ∀ x : ℤ, x ≥ N → ∃ p : ℕ, p.prime ∧ p > 20 ∧ p ∣ P x :=
by
  sorry

end exists_prime_factor_above_20_l507_507403


namespace rate_of_mixed_oil_l507_507142

/-- If 10 litres of an oil at Rs. 50 per litre is mixed with 5 litres of another oil at Rs. 67 per litre,
    then the rate of the mixed oil per litre is Rs. 55.67. --/
theorem rate_of_mixed_oil : 
  let volume1 := 10
  let price1 := 50
  let volume2 := 5
  let price2 := 67
  let total_cost := (volume1 * price1) + (volume2 * price2)
  let total_volume := volume1 + volume2
  (total_cost / total_volume : ℝ) = 55.67 :=
by
  sorry

end rate_of_mixed_oil_l507_507142


namespace acute_angle_of_rhombus_l507_507877

-- Define the conditions
variables {R : ℝ} {α : ℝ}
-- Define the statement to be proven
theorem acute_angle_of_rhombus (h : R > 0) (angle : α > 0 ∧ α < π/2)
  (circumscribed : true)
  (height_prism : 2 * R > 0)
  (angle_diag : true):
  ∃ β : ℝ, β = 2 * arcsin (tan α) :=
sorry

end acute_angle_of_rhombus_l507_507877


namespace teams_equation_l507_507872

theorem teams_equation (x : ℕ) (h1 : 100 = x + 4*x - 10) : 4 * x + x - 10 = 100 :=
by
  sorry

end teams_equation_l507_507872


namespace intersection_M_N_l507_507264

def M := { x : ℝ | ∃ y : ℝ, y = real.sqrt (1 - x) }
def N := { x : ℝ | 0 < x ∧ x < 2 }
def intersection := { x : ℝ | 0 < x ∧ x ≤ 1 }

theorem intersection_M_N :
  { x : ℝ | x ∈ M ∧ x ∈ N } = intersection := sorry

end intersection_M_N_l507_507264


namespace arithmetic_mean_of_38_and_59_is_67_over_144_l507_507937

theorem arithmetic_mean_of_38_and_59_is_67_over_144 :
  (3 / 8 + 5 / 9) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_38_and_59_is_67_over_144_l507_507937


namespace am_hm_inequality_l507_507038

theorem am_hm_inequality (n : ℕ) (x : Fin n → ℝ) (h : ∀ i, 0 < x i) :
  (Finset.univ.sum x) * (Finset.univ.sum (λ i, (x i)⁻¹)) ≥ n^2 := by
sorry

end am_hm_inequality_l507_507038


namespace triangular_figure_area_ratio_l507_507993

theorem triangular_figure_area_ratio (r : ℝ) : 
  (r = 3) → 
  (let A_circle := Real.pi * r^2 in 
   let C := 2 * Real.pi * r in 
   let arc_length := C / 3 in 
   let h := r in 
   let A_triangle_sector := (1 / 2) * arc_length * h in 
   let A_triangular_figure := 3 * A_triangle_sector in 
   (A_triangular_figure / A_circle) = 1) :=
begin
  intros,
  sorry
end

end triangular_figure_area_ratio_l507_507993


namespace arithmetic_mean_of_fractions_l507_507944

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 3/8
  let b : ℚ := 5/9
  (a + b) / 2 = 67 / 144 := 
by
  sorry

end arithmetic_mean_of_fractions_l507_507944


namespace math_problem_l507_507664

theorem math_problem :
  (let ceil35_23 := 2 in
   let ceil9_23_35 := 6 in
   let num := ⌈(23 / 9) - ceil35_23⌉ in
   let denom := ⌈(35 / 9) + ceil9_23_35⌉ in
   num / denom = 1 / 10) :=
by
  sorry

end math_problem_l507_507664


namespace dilution_problem_l507_507657

-- Definitions based on conditions
def initial_volume := 12 -- initial mixture volume in ounces
def initial_concentration := 0.5 -- initial alcohol concentration (50%)
def desired_concentration := 0.25 -- desired alcohol concentration (25%)

-- Main theorem statement
theorem dilution_problem :
  ∃ w : ℝ, ((initial_concentration * initial_volume) = desired_concentration * (initial_volume + w) ∧ w = 12) :=
sorry

end dilution_problem_l507_507657


namespace a_can_be_any_value_l507_507694

section

variables {a b c d : ℤ}

theorem a_can_be_any_value (h1 : (a : ℚ) / b > c / d) (h2 : b > 0) (h3 : d < 0) : 
  ∃ m : ℤ, m = a ∧ (m > (c * b / d) ↔ m > (c * b / d)) :=
by
  intro h1 h2 h3
  use a
  split
  rfl
  sorry

end

end a_can_be_any_value_l507_507694


namespace Michael_pizza_fraction_l507_507882

theorem Michael_pizza_fraction (T : ℚ) (L : ℚ) (total : ℚ) (M : ℚ) 
  (hT : T = 1 / 2) (hL : L = 1 / 6) (htotal : total = 1) (hM : total - (T + L) = M) :
  M = 1 / 3 := 
sorry

end Michael_pizza_fraction_l507_507882


namespace solutions_to_cubic_eq_l507_507241

theorem solutions_to_cubic_eq (z : ℂ) :
  (z = -3 ∨ z = (3 + 3 * complex.I * real.sqrt 3) / 2 ∨ z = (3 - 3 * complex.I * real.sqrt 3) / 2)
  ↔ z^3 = -27 :=
by
  sorry

end solutions_to_cubic_eq_l507_507241


namespace least_integer_with_int_area_l507_507104

-- Definitions
def triangle_sides (a : ℕ) : Prop := a > 14
def integer_area (a : ℕ) : ℕ := 
  let s := (a + (a - 1) + (a + 1)) / 2 in
  s * (s - (a - 1)) * (s - a) * (s - (a + 1))

-- Theorem to be proved
theorem least_integer_with_int_area : ∃ a : ℕ, triangle_sides a ∧ (∃ x, integer_area a = x^2) ∧ a = 52 :=
by {
  sorry
}

end least_integer_with_int_area_l507_507104


namespace point_A_symmetric_to_B_about_l_l507_507270

variables {A B : ℝ × ℝ} {l : ℝ → ℝ → Prop}

-- define point B
def point_B := (1, 2)

-- define the line equation x + y + 3 = 0 as a property
def line_l (x y : ℝ) := x + y + 3 = 0

-- define that A is symmetric to B about line l
def symmetric_about (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop) :=
  (∀ x y : ℝ, l x y → ((A.1 + B.1) / 2 + (A.2 + B.2) / 2 = -(x + y)))
  ∧ ((A.2 - B.2) / (A.1 - B.1) * -1 = -1)

theorem point_A_symmetric_to_B_about_l :
  A = (-5, -4) →
  symmetric_about A B line_l →
  A = (-5, -4) := by
  intros _ sym
  sorry

end point_A_symmetric_to_B_about_l_l507_507270


namespace frog_probability_ends_on_vertical_side_l507_507168

/-
A frog starts at the point (1, 3) and jumps in sequences where each jump is
2 units long in one of the four cardinal directions: up, down, right, or left.
Each direction for a jump is chosen independently at random. The sequence of jumps
ends when the frog reaches the boundary of the square defined by the vertices
(0,0), (0,6), (6,6), (6,0). Prove that the probability that the sequence of jumps ends
on a vertical side of this square is 2/3.
-/

def position := (ℕ × ℕ)

def initial_position : position := (1, 3)

def is_boundary : position → Prop
| (x, y) := x = 0 ∨ x = 6 ∨ y = 0 ∨ y = 6

def is_vertical_boundary : position → Prop
| (x, y) := x = 0 ∨ x = 6

noncomputable def P : position → ℚ
| (x, y) :=
  if is_boundary (x, y) then (if is_vertical_boundary (x, y) then 1 else 0)
  else 1 / 4 * (P (x + 2, y) + P (x - 2, y) + P (x, y + 2) + P (x, y - 2))

theorem frog_probability_ends_on_vertical_side : P initial_position = 2 / 3 :=
sorry

end frog_probability_ends_on_vertical_side_l507_507168


namespace joe_total_paint_used_l507_507973

-- Conditions
def initial_paint : ℕ := 360
def paint_first_week : ℕ := initial_paint * 1 / 4
def remaining_paint_after_first_week : ℕ := initial_paint - paint_first_week
def paint_second_week : ℕ := remaining_paint_after_first_week * 1 / 6

-- Theorem statement
theorem joe_total_paint_used : paint_first_week + paint_second_week = 135 := by
  sorry

end joe_total_paint_used_l507_507973


namespace total_digits_2500_is_9449_l507_507133

def nth_even (n : ℕ) : ℕ := 2 * n

def count_digits_in_range (start : ℕ) (stop : ℕ) : ℕ :=
  (stop - start) / 2 + 1

def total_digits (n : ℕ) : ℕ :=
  let one_digit := 4
  let two_digit := count_digits_in_range 10 98
  let three_digit := count_digits_in_range 100 998
  let four_digit := count_digits_in_range 1000 4998
  let five_digit := 1
  one_digit * 1 +
  two_digit * 2 +
  (three_digit * 3) +
  (four_digit * 4) +
  (five_digit * 5)

theorem total_digits_2500_is_9449 : total_digits 2500 = 9449 := by
  sorry

end total_digits_2500_is_9449_l507_507133


namespace min_tip_percentage_l507_507173

namespace TipCalculation

def mealCost : Float := 35.50
def totalPaid : Float := 37.275
def maxTipPercent : Float := 0.08

theorem min_tip_percentage : ∃ (P : Float), (P / 100 * mealCost = (totalPaid - mealCost)) ∧ (P < maxTipPercent * 100) ∧ (P = 5) := by
  sorry

end TipCalculation

end min_tip_percentage_l507_507173


namespace number_of_real_solutions_l507_507042

noncomputable def system_of_equations (n : ℕ) (a b c : ℝ) (x : Fin n → ℝ) : Prop :=
∀ i : Fin n, a * (x i) ^ 2 + b * (x i) + c = x (⟨(i + 1) % n, sorry⟩)

theorem number_of_real_solutions
  (a b c : ℝ)
  (h : a ≠ 0)
  (n : ℕ)
  (x : Fin n → ℝ) :
  (b - 1) ^ 2 - 4 * a * c < 0 → ¬(∃ x : Fin n → ℝ, system_of_equations n a b c x) ∧
  (b - 1) ^ 2 - 4 * a * c = 0 → ∃! x : Fin n → ℝ, system_of_equations n a b c x ∧
  (b - 1) ^ 2 - 4 * a * c > 0 → ∃ x : Fin n → ℝ, ∃ y : Fin n → ℝ, x ≠ y ∧ system_of_equations n a b c x ∧ system_of_equations n a b c y := 
sorry

end number_of_real_solutions_l507_507042


namespace partition_exists_l507_507890

-- Defining the set S and its elements
variable {S : Finset ℂ} (hS : S.card = 1993) (hnz : ∀ z ∈ S, z ≠ 0)

-- Defining the conditions
def T (p : Finset ℂ) : ℂ := p.sum id
def angle_leq_90 (z T : ℂ) : Prop := Complex.arg z - Complex.arg T ≤ π/2 ∧ Complex.arg z - Complex.arg T ≥ -π/2

-- Main theorem statement
theorem partition_exists (S : Finset ℂ) (hS : S.card = 1993) (hnz : ∀ z ∈ S, z ≠ 0) :
  ∃ (P : List (Finset ℂ)), (∀ p ∈ P, p ⊆ S ∧ p ≠ ∅ ) ∧ (S = P.foldr (· ∪ ·) ∅) ∧
  (∀ p ∈ P, ∀ z ∈ p, angle_leq_90 z (T p)) ∧
  (∀ p q ∈ P, p ≠ q → Complex.arg (T p) - Complex.arg (T q) > π/2 ∨ Complex.arg (T p) - Complex.arg (T q) < -π/2) :=
sorry

end partition_exists_l507_507890


namespace greatest_divisor_of_420_and_90_l507_507506

-- Define divisibility
def divides (a b : ℕ) : Prop := ∃ k, b = k * a

-- Main problem statement
theorem greatest_divisor_of_420_and_90 {d : ℕ} :
  (divides d 420) ∧ (d < 60) ∧ (divides d 90) → d ≤ 30 := 
sorry

end greatest_divisor_of_420_and_90_l507_507506


namespace problem_solution_l507_507565

theorem problem_solution :
  (tan 25 * pi / 180 + tan 35 * pi / 180 + sqrt 3 * tan 25 * pi / 180 * tan 35 * pi / 180 = sqrt 3) ∧
  (tan (22.5 * pi / 180) / (1 - (tan (22.5 * pi / 180)) ^ 2) ≠ 1) ∧
  (cos (pi / 8) ^ 2 - sin (pi / 8) ^ 2 ≠ 1 / 2) ∧
  (1 / sin (10 * pi / 180) - sqrt 3 / cos (10 * pi / 180) = 4) :=
by
  sorry

end problem_solution_l507_507565


namespace syllogism_error_l507_507654

-- Definitions based on conditions from a)
def major_premise (a: ℝ) : Prop := a^2 > 0

def minor_premise (a: ℝ) : Prop := true

-- Theorem stating that the conclusion does not necessarily follow
theorem syllogism_error (a : ℝ) (h_minor : minor_premise a) : ¬major_premise 0 :=
by
  sorry

end syllogism_error_l507_507654


namespace part_a_part_b_l507_507436

-- Define sum conditions for consecutive odd integers
def consecutive_odd_sum (N : ℕ) : Prop :=
  ∃ (n k : ℕ), n ≥ 2 ∧ N = n * (2 * k + n)

-- Part (a): Prove 2005 can be written as sum of consecutive odd positive integers
theorem part_a : consecutive_odd_sum 2005 :=
by
  sorry

-- Part (b): Prove 2006 cannot be written as sum of consecutive odd positive integers
theorem part_b : ¬consecutive_odd_sum 2006 :=
by
  sorry

end part_a_part_b_l507_507436


namespace triangle_at_most_one_obtuse_l507_507044

-- Define the notion of a triangle and obtuse angle
def isTriangle (A B C: ℝ) : Prop := (A + B > C) ∧ (A + C > B) ∧ (B + C > A)
def isObtuseAngle (theta: ℝ) : Prop := 90 < theta ∧ theta < 180

-- A theorem to prove that a triangle cannot have more than one obtuse angle 
theorem triangle_at_most_one_obtuse (A B C: ℝ) (angleA angleB angleC : ℝ) 
    (h1 : isTriangle A B C)
    (h2 : isObtuseAngle angleA)
    (h3 : isObtuseAngle angleB)
    (h4 : angleA + angleB + angleC = 180):
    false :=
by
  sorry

end triangle_at_most_one_obtuse_l507_507044


namespace inequality_holds_l507_507668

variable (a t1 t2 t3 t4 : ℝ)

theorem inequality_holds
  (a_pos : 0 < a)
  (h_a_le : a ≤ 7/9)
  (t1_pos : 0 < t1)
  (t2_pos : 0 < t2)
  (t3_pos : 0 < t3)
  (t4_pos : 0 < t4)
  (h_prod : t1 * t2 * t3 * t4 = a^4) :
  (1 / Real.sqrt (1 + t1) + 1 / Real.sqrt (1 + t2) + 1 / Real.sqrt (1 + t3) + 1 / Real.sqrt (1 + t4)) ≤ (4 / Real.sqrt (1 + a)) :=
by
  sorry 

end inequality_holds_l507_507668


namespace range_of_x_l507_507678

theorem range_of_x (x : ℝ) : (x^2 - 9*x + 14 < 0) ∧ (2*x + 3 > 0) ↔ (2 < x) ∧ (x < 7) := 
by 
  sorry

end range_of_x_l507_507678


namespace length_major_axis_l507_507601

noncomputable def cylinder_radius : ℝ := 1

def plane_intersects_cylinder_form_ellipse (radius : ℝ) : Prop :=
∃ minor_axis major_axis : ℝ, 
  minor_axis = 2 * radius ∧
  major_axis = 1.5 * minor_axis

theorem length_major_axis (h : plane_intersects_cylinder_form_ellipse cylinder_radius) :
  ∃ major_axis, major_axis = 3 :=
by {
  cases h with minor_axis h_minor_major,
  cases h_minor_major with h_minor h_major,
  use (1.5 * minor_axis),
  rw h_minor,
  rw mul_comm 1.5,
  norm_num,
  exact (0.5 : ℝ) + 1 + 1 }

end length_major_axis_l507_507601


namespace inequality_sqrt_sum_l507_507027

theorem inequality_sqrt_sum (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
by
  sorry

end inequality_sqrt_sum_l507_507027


namespace oxen_count_l507_507580

theorem oxen_count (B C O : ℕ) (H1 : 3 * B = 4 * C) (H2 : 3 * B = 2 * O) (H3 : 15 * B + 24 * C + O * O = 33 * B + (3 / 2) * O * B) (H4 : 24 * B = 48) (H5 : 60 * C + 30 * B + 18 * (O * (3 / 2) * B) = 108 * B + (3 / 2) * O * B * 18)
: O = 8 :=
by 
  sorry

end oxen_count_l507_507580


namespace number_of_valid_sequences_l507_507793

-- Define the conditions of the problem
def first_7_odd_pos_integers : List ℕ := [1, 3, 5, 7, 9, 11, 13]

-- Define a predicate to check if a given list satisfies the constraints
def valid_sequence (seq : List ℕ) : Prop :=
  seq.length = 7 ∧
  (∀ i, 2 ≤ i ∧ i ≤ 7 → (seq[i-1] + 2 ∈ seq.take (i-1) ∨ seq[i-1] - 2 ∈ seq.take (i-1)))

-- Define the main theorem to prove the number of valid sequences
theorem number_of_valid_sequences : 
  {l : List ℕ // (valid_sequence l)}.toFinset.card = 64 :=
sorry

end number_of_valid_sequences_l507_507793


namespace construct_triangle_from_medians_l507_507098

theorem construct_triangle_from_medians (AA1 BB1 CC1 : ℝ) 
  (h1 : AA1 > 0) (h2 : BB1 > 0) (h3 : CC1 > 0) : 
  ∃ (A B C A1 B1 C1 M : Point),
    is_centroid_of A B C M ∧
    dist A A1 = AA1 ∧ dist B B1 = BB1 ∧ dist C C1 = CC1 ∧
    dist M A1 = 1/3 * AA1 ∧ dist M B1 = 1/3 * BB1 ∧ dist M C1 = 1/3 * CC1 :=
by
  sorry

end construct_triangle_from_medians_l507_507098


namespace algebraic_expression_value_l507_507435

variable (x y A B : ℤ)
variable (x_val : x = -1)
variable (y_val : y = 2)
variable (A_def : A = 2*x + y)
variable (B_def : B = 2*x - y)

theorem algebraic_expression_value : 
  (A^2 - B^2) * (x - 2*y) = 80 := 
by
  rw [x_val, y_val, A_def, B_def]
  sorry

end algebraic_expression_value_l507_507435


namespace problem_l507_507401

theorem problem (n : ℕ) (h₁ : 2 ≤ n) (a : Fin n → ℝ) 
  (h₂ : ∑ i, a i = 0) 
  (A : Finset (Fin n × Fin n)) 
  (h₃ : ∀ i j, (i, j) ∈ A ↔ (1 ≤ i) ∧ (i < j) ∧ (j ≤ n) ∧ abs (a i - a j) ≥ 1)
  (h₄ : A.nonempty) :
  (∑ (ij : Fin n × Fin n) in A, a ij.1 * a ij.2) < 0 := 
sorry

end problem_l507_507401


namespace problem_statement_l507_507391

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then log 2 (1 - x) else 4^x

theorem problem_statement : f (-3) + f (log 2 3) = 11 :=
by 
  have h1 : f (-3) = log 2 (1 - (-3)), from if_pos (by norm_num),
  have h2 : log 2 4 = 2, from by sorry,
  have h3 : f (log 2 3) = 4 ^ (log 2 3), from if_neg (by norm_num),
  have h4 : (4 : ℝ) = 2^2, from by norm_num,
  have h5 : (4 : ℝ) ^ (log 2 3) = (2^2) ^ (log 2 3), from congr_arg (^ (log 2 3)) (by sorry),
  have h6 : (2^2) ^ (log 2 3) = 2^(2 * log 2 3), from by sorry,
  have h7 : 2^(2 * log 2 3) = 2^(log 2 (3^2)), from by sorry,
  have h8 : 2^(log 2 (3^2)) = 3^2, from by sorry,
  have h9 : 3^2 = 9, from by norm_num,
  show f (-3) + f (log 2 3) = 11, from by sorry

end problem_statement_l507_507391


namespace horizontal_distance_is_0_65_l507_507599

def parabola (x : ℝ) : ℝ := 2 * x^2 - 3 * x - 4

-- Calculate the horizontal distance between two points on the parabola given their y-coordinates and prove it equals to 0.65
theorem horizontal_distance_is_0_65 :
  ∃ (x1 x2 : ℝ), 
    parabola x1 = 10 ∧ parabola x2 = 0 ∧ abs (x1 - x2) = 0.65 :=
sorry

end horizontal_distance_is_0_65_l507_507599


namespace max_value_expression_achieves_max_value_l507_507803

noncomputable def f (y : ℝ) : ℝ := (y^2 + 4 - (y^4 + 16).sqrt) / y

theorem max_value_expression (y : ℝ) (hy : 0 < y) : f y ≤ 2 * real.sqrt 2 - 2 :=
sorry

theorem achieves_max_value : f 2 = 2 * real.sqrt 2 - 2 :=
sorry

end max_value_expression_achieves_max_value_l507_507803


namespace arithmetic_mean_of_fractions_l507_507914

theorem arithmetic_mean_of_fractions : 
  (3 : ℚ) / 8 + (5 : ℚ) / 9 = 2 * (67 : ℚ) / 144 :=
by {
  rw [(show (3 : ℚ) / 8 + (5 : ℚ) / 9 = (27 : ℚ) / 72 + (40 : ℚ) / 72, by sorry),
      (show (27 : ℚ) / 72 + (40 : ℚ) / 72 = (67 : ℚ) / 72, by sorry),
      (show (2 : ℚ) * (67 : ℚ) / 144 = (67 : ℚ) / 72, by sorry)]
}

end arithmetic_mean_of_fractions_l507_507914


namespace john_spent_amount_l507_507372

-- Definitions based on the conditions in the problem.
def hours_played : ℕ := 3
def cost_per_6_minutes : ℚ := 0.50
def minutes_per_6_minutes_interval : ℕ := 6
def total_minutes_played (hours : ℕ) : ℕ := hours * 60

-- The theorem statement.
theorem john_spent_amount 
  (h : hours_played = 3) 
  (c : cost_per_6_minutes = 0.50)
  (m_interval : minutes_per_6_minutes_interval = 6) :
  let intervals := (total_minutes_played hours_played) / minutes_per_6_minutes_interval in
  intervals * cost_per_6_minutes = 15 := 
by
  sorry

end john_spent_amount_l507_507372


namespace find_point_P_coordinates_l507_507718

theorem find_point_P_coordinates :
  ∃ P : ℝ × ℝ, 
    (∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ π ∧ P.1 = 3 * cos θ ∧ P.2 = 4 * sin θ) ∧
    (P.2 / P.1 = 1) ∧
    P = (12 / 5, 12 / 5) := by
  sorry

end find_point_P_coordinates_l507_507718


namespace equilateral_triangle_ellipse_eccentricity_l507_507323

noncomputable def eccentricity_of_ellipse {b : ℝ} (hb : b ≠ 0) : ℝ :=
  let a := 2 * b in
  let c := Real.sqrt (a^2 - b^2) in
  c / a

theorem equilateral_triangle_ellipse_eccentricity (b : ℝ) (hb : b ≠ 0) :
  eccentricity_of_ellipse hb = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ellipse_eccentricity_l507_507323


namespace fourth_term_of_geometric_sequence_l507_507850

theorem fourth_term_of_geometric_sequence (a₁ : ℝ) (a₆ : ℝ) (a₄ : ℝ) (r : ℝ)
  (h₁ : a₁ = 1000)
  (h₂ : a₆ = a₁ * r^5)
  (h₃ : a₆ = 125)
  (h₄ : a₄ = a₁ * r^3) : 
  a₄ = 125 :=
sorry

end fourth_term_of_geometric_sequence_l507_507850


namespace equal_division_l507_507997

theorem equal_division (total_boxes stops : ℕ) (h1 : total_boxes = 27) (h2 : stops = 3) : 
  ∃ (boxes_per_stop : ℕ), boxes_per_stop = total_boxes / stops ∧ boxes_per_stop = 9 :=
by
  use 9
  have h3 : total_boxes / stops = 9 := by sorry
  exact ⟨h3, rfl⟩

end equal_division_l507_507997


namespace factorial_division_l507_507647

theorem factorial_division (N : Nat) (h : N ≥ 2) : 
  (Nat.factorial (2 * N)) / ((Nat.factorial (N + 2)) * (Nat.factorial (N - 2))) = 
  (List.prod (List.range' (N + 3) (2 * N - (N + 2) + 1))) / (Nat.factorial (N - 1)) :=
sorry

end factorial_division_l507_507647


namespace correct_propositions_is_4_l507_507490

theorem correct_propositions_is_4 :
  (∀ α β : ℝ, (0 < α ∧ α < π/2) → (0 < β ∧ β < π/2) → α > β → ¬(sin α > sin β)) ∧
  (∀ a : ℝ, (∃ T : ℝ, T = 4 * π ∧ T = (2 * π) / |a|) → a ≠ 1/2 ∧ a ≠ -1/2) ∧
  (¬∀ x : ℝ, (sin (2 * x) - sin x) / (sin x - 1) = -(sin (2 * -x) - sin (-x)) / (sin (-x) - 1)) ∧
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ π) → -cos x ≤ -cos x → -(-cos x) ≤ -(-cos x))
  → true := sorry

end correct_propositions_is_4_l507_507490


namespace tangent_line_at_one_min_value_f_l507_507385

noncomputable def f (x a : ℝ) : ℝ :=
  x^2 + a * |Real.log x - 1|

theorem tangent_line_at_one (a : ℝ) (h1 : a = 1) : 
  ∃ (m b : ℝ), (∀ x : ℝ, f x a = m * x + b) ∧ m = 1 ∧ b = 1 ∧ (x - y + 1 = 0) := 
sorry

theorem min_value_f (a : ℝ) (h1 : 0 < a) : 
  (1 ≤ x ∧ x < e)  →  (x - f x a <= 0) ∨  (∀ (x : ℝ), 
  (f x a = if 0 < a ∧ a ≤ 2 then 1 + a 
          else if 2 < a ∧ a ≤ 2 * Real.exp (2) then 3 * (a / 2)^2 - (a / 2)^2 * Real.log (a / 2) else 
          Real.exp 2) 
   ) := 
sorry

end tangent_line_at_one_min_value_f_l507_507385


namespace percent_of_percent_is_five_l507_507147

def percent_to_decimal (p : ℝ) : ℝ := p / 100

theorem percent_of_percent_is_five (p1 p2 : ℝ) (h_p1 : p1 = 20) (h_p2 : p2 = 25) :
    (percent_to_decimal p1) * (percent_to_decimal p2) * 100 = 5 :=
by
  sorry

end percent_of_percent_is_five_l507_507147


namespace problem_solution_l507_507007

def f (n : ℕ) : ℕ :=
  -- This function counts the number of 0's in the decimal representation of n.
  -- Placeholder implementation (the actual implementation should correctly count 0's).
  sorry

def M : ℕ :=
  -- This function calculates the given expression for M.
  @Finset.sum (ℕ) ℕ ⟨0⟩ (Finset.range 99999) (λ n, f n * (2 ^ f n))

theorem problem_solution : M - 100000 = 2780 :=
by
  -- Proof placeholder
  sorry

end problem_solution_l507_507007


namespace find_first_term_and_common_difference_of_arithmetic_sequence_l507_507001

theorem find_first_term_and_common_difference_of_arithmetic_sequence {a b : ℕ → ℝ} 
  (h_arith : ∀ n, a (n + 1) = a n + (a 1 - a 0))
  (h_geom : ∀ n, b (n + 1) = b n * (b 1 / b 0))
  (h_b1 : b 0 = (a 0) ^ 2)
  (h_b2 : b 1 = (a 1) ^ 2)
  (h_b3 : b 2 = (a 2) ^ 2)
  (h_a1_lt_a2 : a 0 < a 1)
  (h_limit : ∀ ε > 0, ∃ N, ∀ n ≥ N, abs ((∑ i in finset.range (n + 1), b i) - (√2 + 1)) < ε) :
  a 0 = -√2 ∧ (a 1 - a 0) = 2 * √2 - 2 :=
by
  sorry

end find_first_term_and_common_difference_of_arithmetic_sequence_l507_507001


namespace prime_range_for_integer_roots_l507_507317

theorem prime_range_for_integer_roots (p : ℕ) (h_prime : Prime p) 
  (h_int_roots : ∃ (a b : ℤ), a + b = -p ∧ a * b = -300 * p) : 
  1 < p ∧ p ≤ 11 :=
sorry

end prime_range_for_integer_roots_l507_507317


namespace arithmetic_mean_of_38_and_59_is_67_over_144_l507_507932

theorem arithmetic_mean_of_38_and_59_is_67_over_144 :
  (3 / 8 + 5 / 9) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_38_and_59_is_67_over_144_l507_507932


namespace find_distance_between_stations_l507_507187

noncomputable def distance_between_stations (D T : ℝ) : Prop :=
  D = 100 * T ∧
  D = 50 * (T + 15 / 60) ∧
  D = 70 * (T + 7 / 60)

theorem find_distance_between_stations :
  ∃ D T : ℝ, distance_between_stations D T ∧ D = 25 :=
by
  sorry

end find_distance_between_stations_l507_507187


namespace linda_original_amount_l507_507746

-- Define the original amount of money Lucy and Linda have
variables (L : ℕ) (lucy_initial : ℕ := 20)

-- Condition: If Lucy gives Linda $5, they have the same amount of money.
def condition := (lucy_initial - 5) = (L + 5)

-- Theorem: The original amount of money that Linda had
theorem linda_original_amount (h : condition L) : L = 10 := 
sorry

end linda_original_amount_l507_507746


namespace total_amount_paid_each_person_after_year_l507_507966

-- All conditions as definitions

def monthly_cost_before_tax_first_six_months := 14 + (2 * 4)
def tax_first_six_months := 0.10 * monthly_cost_before_tax_first_six_months
def monthly_cost_after_tax_first_six_months := monthly_cost_before_tax_first_six_months + tax_first_six_months
def each_person_share_first_six_months := monthly_cost_after_tax_first_six_months / 4
def total_first_six_months := each_person_share_first_six_months * 6

def monthly_cost_before_tax_last_six_months := 18 + (2.5 * 5)
def tax_last_six_months := 0.15 * monthly_cost_before_tax_last_six_months
def monthly_cost_after_tax_last_six_months := monthly_cost_before_tax_last_six_months + tax_last_six_months
def each_person_share_last_six_months := monthly_cost_after_tax_last_six_months / 5
def total_last_six_months := each_person_share_last_six_months * 6

def total_paid_by_each_person := total_first_six_months + total_last_six_months

-- Problem statement
theorem total_amount_paid_each_person_after_year : 
  total_paid_by_each_person = 78.39 := 
by
  sorry

end total_amount_paid_each_person_after_year_l507_507966


namespace variance_probability_binomial_l507_507090

open ProbabilityTheory

noncomputable def variance_of_binomial (n : ℕ) (ξ : ℕ → ℕ) : ℚ :=
  n * (1 / 2 : ℚ) * (1 - 1 / 2 : ℚ)

theorem variance_probability_binomial :
  ∀ (n : ℕ) (ξ : ℕ → ℕ),
  3 ≤ n ∧ n ≤ 8 ∧
  (∀ k : ℕ, Probability.(binomial n (1 / 2 : ℚ)).pmf k = choose n k * (1 / 2 : ℚ) ^ k * (1 / 2 : ℚ) ^ (n - k)) ∧
  Probability.(binomial n (1 / 2 : ℚ)).pmf 1 = 3 / 32
  → variance_of_binomial n ξ = 3 / 2 :=
by
  sorry

end variance_probability_binomial_l507_507090


namespace melanie_apples_l507_507786

theorem melanie_apples (apples_initial apples_total : ℕ) (h1 : apples_initial = 43) (h2 : apples_total = 70) : apples_total - apples_initial = 27 :=
by
  rw [h1, h2]
  norm_num

end melanie_apples_l507_507786


namespace player_B_wins_in_least_steps_l507_507886

noncomputable def least_steps_to_win (n : ℕ) : ℕ :=
  n

theorem player_B_wins_in_least_steps (n : ℕ) (h_n : n > 0) :
  ∃ k, k = least_steps_to_win n ∧ k = n := by
  sorry

end player_B_wins_in_least_steps_l507_507886


namespace trip_parts_distance_l507_507333

theorem trip_parts_distance 
    (total_distance : ℝ) (first_part_distance : ℝ) 
    (first_part_speed : ℝ) (avg_speed_whole_trip : ℝ) 
    (avg_speed_last_part : ℝ) : ℝ :=
    let first_part_time := first_part_distance / first_part_speed in
    let total_time := total_distance / avg_speed_whole_trip in
    let last_part_time := total_time - first_part_time in
    let last_part_distance := avg_speed_last_part * last_part_time in
    last_part_distance

example : trip_parts_distance 100 30 60 40 35 = 70 := by
  sorry

end trip_parts_distance_l507_507333


namespace problem_I_l507_507577

theorem problem_I (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) : 
  (1 + 1 / a) * (1 + 1 / b) ≥ 9 := 
by
  sorry

end problem_I_l507_507577


namespace product_discount_rate_l507_507584

theorem product_discount_rate (cost_price marked_price : ℝ) (desired_profit_rate : ℝ) :
  cost_price = 200 → marked_price = 300 → desired_profit_rate = 0.2 →
  (∃ discount_rate : ℝ, discount_rate = 0.8 ∧ marked_price * discount_rate = cost_price * (1 + desired_profit_rate)) :=
by
  intros
  sorry

end product_discount_rate_l507_507584


namespace farthest_point_from_origin_l507_507960

theorem farthest_point_from_origin :
  let points := [(0, 6), (2, 1), (4, -3), (0, 7), (-2, -1)] in
  ∃ p ∈ points, ∀ q ∈ points, (sqrt (p.1 ^ 2 + p.2 ^ 2)) ≥ (sqrt (q.1 ^ 2 + q.2 ^ 2)) :=
sorry

end farthest_point_from_origin_l507_507960


namespace polygon_sides_16_l507_507070

def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

noncomputable def arithmetic_sequence_sum (a1 an : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (a1 + an) / 2

theorem polygon_sides_16 (n : ℕ) (a1 an : ℝ) (d : ℝ) 
  (h1 : d = 5) (h2 : an = 160) (h3 : a1 = 160 - 5 * (n - 1))
  (h4 : arithmetic_sequence_sum a1 an d n = sum_of_interior_angles n)
  : n = 16 :=
sorry

end polygon_sides_16_l507_507070


namespace only_other_list_with_same_product_l507_507663

-- Assigning values to letters
def letter_value (ch : Char) : ℕ :=
  match ch with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5 | 'F' => 6 | 'G' => 7 | 'H' => 8
  | 'I' => 9 | 'J' => 10| 'K' => 11| 'L' => 12| 'M' => 13| 'N' => 14| 'O' => 15| 'P' => 16
  | 'Q' => 17| 'R' => 18| 'S' => 19| 'T' => 20| 'U' => 21| 'V' => 22| 'W' => 23| 'X' => 24
  | 'Y' => 25| 'Z' => 26| _ => 0

-- Define the product function for a list of 4 letters
def product_of_list (lst : List Char) : ℕ :=
  lst.map letter_value |> List.prod

-- Define the specific lists
def BDFH : List Char := ['B', 'D', 'F', 'H']
def BCDH : List Char := ['B', 'C', 'D', 'H']

-- The main statement to prove
theorem only_other_list_with_same_product : 
  product_of_list BCDH = product_of_list BDFH :=
by
  -- Sorry is a placeholder for the proof
  sorry

end only_other_list_with_same_product_l507_507663


namespace num_men_employed_l507_507588

noncomputable def original_number_of_men (M : ℕ) : Prop :=
  let total_work_original := M * 5
  let total_work_actual := (M - 8) * 15
  total_work_original = total_work_actual

theorem num_men_employed (M : ℕ) (h : original_number_of_men M) : M = 12 :=
by sorry

end num_men_employed_l507_507588


namespace sum_not_perfect_cube_l507_507040

theorem sum_not_perfect_cube :
  ∀ (s : List ℕ), s = List.range 1967.succ → ∑ i in s, i ≠ k^3 for any k : ℕ :=
by
  intros s hs
  sorry

end sum_not_perfect_cube_l507_507040


namespace ctg_to_tg_inequality_l507_507969

theorem ctg_to_tg_inequality (α β γ : ℝ) (a b c S : ℝ)
  (h_ctg_sum : Real.cot α + Real.cot β + Real.cot γ ≥ Real.sqrt 3)
  (h_sides : a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S) :
  Real.tan (α / 2) + Real.tan (β / 2) + Real.tan (γ / 2) ≥ Real.sqrt 3 := sorry

end ctg_to_tg_inequality_l507_507969


namespace p₁_line_p₂_line_base_case_inductive_step_point_on_line_l507_507298

-- Definitions
variable {a b : ℕ → ℝ}

-- Initial condition
def P₁ : ∀ n, P₁(n) = (1, -1)

-- Recurrence relations
def a_rec (n : ℕ) := a(n+1) = a n * b (n+1)
def b_rec (n : ℕ) := b(n+1) = b n / (1 - 4 * (a n)^2)

-- Equation of the line
def line_eq (x y : ℝ) := 2 * x + y = 1

-- Proof statements
theorem p₁_line : line_eq 1 (-1) := by sorry
theorem p₂_line : line_eq (1/3) (1/3) := by sorry

-- Induction base case
theorem base_case (n : ℕ) : 2 * a 1 + b 1 = 1 := by sorry

-- Induction step
theorem inductive_step (k : ℕ) (h : 2 * a k + b k = 1) : 2 * a (k+1) + b (k+1) = 1 := by sorry

-- Final proof by induction
theorem point_on_line (n : ℕ) : line_eq (a n) (b n) := by sorry

end p₁_line_p₂_line_base_case_inductive_step_point_on_line_l507_507298


namespace length_of_QR_l507_507760

-- Define a triangle ABC with given side lengths
structure Triangle (α : Type) [Field α] :=
  (A B C : α × α)
  (AB AC BC : α)
  (h_AB : AB = dist A B)
  (h_AC : AC = dist A C)
  (h_BC : BC = dist B C)

-- Define the problem conditions
def given_triangle : Triangle ℝ :=
{ 
  A := (0, 0),
  B := (13, 0),
  C := (0, 12),
  AB := 13,
  AC := 12,
  BC := 5,
  h_AB := by sorry,
  h_AC := by sorry,
  h_BC := by sorry
}

-- Define the circle P conditions
def circleP_radius := (2.70833 : ℝ)

-- Define points of intersection Q and R
def QR_length := (5.42 : ℝ)

-- The theorem to prove, given the conditions
theorem length_of_QR (ABC : Triangle ℝ) (radius_P : ℝ) : ABC.AB = 13 → ABC.AC = 12 → ABC.BC = 5 → radius_P = 2.70833 → QR_length = 5.42 :=
by
  intros h1 h2 h3 h4
  exact sorry

end length_of_QR_l507_507760


namespace arithmetic_mean_of_fractions_l507_507931

theorem arithmetic_mean_of_fractions (a b : ℚ) (h1 : a = 3 / 8) (h2 : b = 5 / 9) :
  (a + b) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_fractions_l507_507931


namespace sin_ratio_zero_l507_507003

theorem sin_ratio_zero (c : ℝ) (h : c = Real.pi / 12) : 
  (sin (4 * c) * sin (8 * c) * sin (12 * c)) / (sin (2 * c) * sin (4 * c) * sin (6 * c)) = 0 := 
by
  sorry

end sin_ratio_zero_l507_507003


namespace total_workers_proof_l507_507846

noncomputable def total_workers (W : ℕ) : Prop :=
  let avg_all_workers := 9500
  let avg_technicians := 12000
  let num_technicians := 7
  let avg_non_technicians := 6000 in
  (W * avg_all_workers) = (num_technicians * avg_technicians + (W - num_technicians) * avg_non_technicians)

theorem total_workers_proof : total_workers 12 :=
by
  unfold total_workers
  have h : 9500 * 12 = 7 * 12000 + (12 - 7) * 6000 := by norm_num
  exact h

end total_workers_proof_l507_507846


namespace log2_8192_eq_13_l507_507871

theorem log2_8192_eq_13 : ∃ c d : ℤ, (log 2 8192 = 13) ∧ (c = 13) ∧ (d = 14) ∧ (c + d = 27) := by
  sorry

end log2_8192_eq_13_l507_507871


namespace distance_A_to_B_is_64_yards_l507_507248

theorem distance_A_to_B_is_64_yards :
  let south1 := 50
  let west := 80
  let north := 20
  let east := 30
  let south2 := 10
  let net_south := south1 + south2 - north
  let net_west := west - east
  let distance := Real.sqrt ((net_south ^ 2) + (net_west ^ 2))
  distance = 64 :=
  by
  let south1 := 50
  let west := 80
  let north := 20
  let east := 30
  let south2 := 10
  let net_south := south1 + south2 - north
  let net_west := west - east
  let distance := Real.sqrt ((net_south ^ 2) + (net_west ^ 2))
  sorry

end distance_A_to_B_is_64_yards_l507_507248


namespace ellipse_properties_l507_507719

variables {a b c : ℝ} (x y : ℝ)

-- Condition: Defining the ellipse equation, eccentricity and chord length
def is_ellipse_C (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def eccentricity_condition : Prop := a > b ∧ b > 0 ∧ (c = a * (sqrt 2 / 2))
def chord_condition : Prop := ∃ x y : ℝ, (x - y - sqrt 2 = 0 ∧ x^2 + y^2 = a^2 ∧ 2 = 2 * sqrt(a^2 - 1^2))

-- Equation I: Standard form of the ellipse
def standard_equation : Prop :=  a = sqrt 2 ∧ b = 1 ∧ (a^2 = 2 ∧ (x^2 / 2 + y^2 = 1))

-- Equation II: Fixed point on x-axis for dot product being constant
def fixed_point_condition (M : ℝ × ℝ) : Prop :=
  ∃ k x1 x2 y1 y2 : ℝ, y1 = k * (x1 - 1) ∧ y2 = k * (x2 - 1) ∧ (1 + 2*k^2)*x^2 - 4*k^2*x + 2*k^2 - 2 = 0 ∧
  x1 + x2 = (4*k^2 / (1 + 2*k^2)) ∧ x1 * x2 = (2*k^2 - 2)/(1 + 2*k^2) ∧
  M = (5/4, 0) ∧ (1 + k^2)*(2*k^2 - 2)/(1 + 2*k^2) - (5/4 + k^2)*(4*k^2/(1 + 2*k^2)) + (5/4)^2 + k^2 = -7/16

-- Proof statement for the entire problem
theorem ellipse_properties (x y : ℝ) :
  (is_ellipse_C x y ∧ eccentricity_condition ∧ chord_condition)
  → (standard_equation x y ∧ ∃ M : ℝ × ℝ, fixed_point_condition M) :=
by {
  sorry,
}

end ellipse_properties_l507_507719


namespace find_b_value_l507_507265

theorem find_b_value (b : ℝ) : IsRoot (fun x => x^2 + b * x - 36) (-9) → b = 5 :=
by
  intro h
  sorry

end find_b_value_l507_507265


namespace remove_2_rows_and_2_columns_cross_all_stars_l507_507356

theorem remove_2_rows_and_2_columns_cross_all_stars (grid : Fin 4 → Fin 4 → Prop) :
  (∃ (stars : Fin 6 → (Fin 4 × Fin 4)),
    (∀ i, grid (stars i).fst (stars i).snd) ∧
    (∀ (i j : Fin 6), i ≠ j → stars i ≠ stars j)) →
  ∃ (r1 r2 : Fin 4) (c1 c2 : Fin 4),
    (∀ i, grid r1 (stars i).snd ∨ grid r2 (stars i).snd ∨
         grid (stars i).fst c1 ∨ grid (stars i).fst c2) :=
sorry

end remove_2_rows_and_2_columns_cross_all_stars_l507_507356


namespace problem1_problem2_l507_507576

variable {a b : ℝ}

theorem problem1 (h : a ≠ b) : 
  ((b / (a - b)) - (a / (a - b))) = -1 := 
by
  sorry

theorem problem2 (h1 : a ≠ b) (h2 : a ≠ 0) (h3 : b ≠ 0) : 
  ((a^2 - a * b)/(a^2) / ((a / b) - (b / a))) = (b / (a + b)) := 
by
  sorry

end problem1_problem2_l507_507576


namespace length_of_angle_bisector_l507_507864

theorem length_of_angle_bisector (AB AC : ℝ) (angleBAC : ℝ) (AD : ℝ) :
  AB = 6 → AC = 3 → angleBAC = 60 → AD = 2 * Real.sqrt 3 :=
by
  intro hAB hAC hAngleBAC
  -- Consider adding proof steps here in the future
  sorry

end length_of_angle_bisector_l507_507864


namespace ball_bounce_height_lt_one_l507_507156

theorem ball_bounce_height_lt_one :
  ∃ (k : ℕ), 15 * (1/3:ℝ)^k < 1 ∧ k = 3 := 
sorry

end ball_bounce_height_lt_one_l507_507156


namespace smallest_positive_integer_with_eight_factors_l507_507515

theorem smallest_positive_integer_with_eight_factors :
  ∃ n : ℕ, (∀ m : ℕ, (m < n ∧ (∀ d : ℕ, d | m → d = 1 ∨ d = m) → (∃ a b : ℕ, distinct_factors_count m a b ∧ a = 8)) → n = 24) :=
by
  sorry

def distinct_factors_count (n : ℕ) (a b : ℕ) : Prop :=
  ∃ (p q : ℕ), prime p ∧ prime q ∧ n = p^a * q^b ∧ (a + 1) * (b + 1) = 8

end smallest_positive_integer_with_eight_factors_l507_507515


namespace projection_is_five_over_two_l507_507000

variables {ℝ : Type*} [inner_product_space ℝ]

noncomputable def projection_of_vector (e1 e2 a b : ℝ) :=
  let cos_theta := real.cos (real.pi / 3) in
  let dot_e1_e2 := 1 * 1 * cos_theta in
  let dot_a_b := (e1 + 3 * e2) * (2 * e1) in
  let magnitude_b := 2 * e1 in
  dot_a_b / magnitude_b

theorem projection_is_five_over_two
  (e1 e2 : ℝ)
  (he1_unit : ∥e1∥ = 1)
  (he2_unit : ∥e2∥ = 1)
  (angle_e1_e2 : real.angle e1 e2 = real.pi / 3) :
  projection_of_vector e1 e2 (e1 + 3 * e2) (2 * e1) = 5 / 2 := sorry

end projection_is_five_over_two_l507_507000


namespace find_removed_number_l507_507558

theorem find_removed_number (numbers : List ℕ) (avg_remain : ℝ) (h_list : numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13]) (h_avg : avg_remain = 7.5) :
  ∃ x, x ∈ numbers ∧ 
       (let numbers_removed := numbers.erase x in 
        (numbers.sum - x) / (numbers.length - 1) = avg_remain) := 
by
  sorry

end find_removed_number_l507_507558


namespace infinite_seq_condition_l507_507222

theorem infinite_seq_condition (x : ℕ → ℕ) (n m : ℕ) : 
  (∀ i, x i = 0 → x (i + m) = 1) → 
  (∀ i, x i = 1 → x (i + n) = 0) → 
  ∃ d p q : ℕ, n = 2^d * p ∧ m = 2^d * q ∧ p % 2 = 1 ∧ q % 2 = 1  :=
by 
  intros h1 h2 
  sorry

end infinite_seq_condition_l507_507222


namespace team_A_champion_probability_l507_507450

/-- Teams A and B are playing a volleyball match.
Team A needs to win one more game to become the champion, while Team B needs to win two more games to become the champion.
The probability of each team winning each game is 0.5. -/
theorem team_A_champion_probability :
  let p_win := (0.5 : ℝ)
  let prob_A_champion := 1 - p_win * p_win
  prob_A_champion = 0.75 := by
  sorry

end team_A_champion_probability_l507_507450


namespace quadratic_value_at_two_l507_507268

open Real

-- Define the conditions
variables (a b : ℝ)

def f (x : ℝ) : ℝ := x^2 + a * x + b

-- State the proof problem
theorem quadratic_value_at_two (h₀ : f a b (f a b 0) = 0) (h₁ : f a b (f a b 1) = 0) (h₂ : f a b 0 ≠ f a b 1) :
  f a b 2 = 2 := 
sorry

end quadratic_value_at_two_l507_507268


namespace arithmetic_mean_of_fractions_l507_507893

theorem arithmetic_mean_of_fractions :
  (3/8 + 5/9) / 2 = 67 / 144 :=
by
  sorry

end arithmetic_mean_of_fractions_l507_507893


namespace inequality_proof_l507_507386

open Real

-- Define the conditions
def conditions (a b c : ℝ) := (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a * b * c = 1)

-- Express the inequality we need to prove
def inequality (a b c : ℝ) :=
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1

-- Statement of the theorem
theorem inequality_proof (a b c : ℝ) (h : conditions a b c) : inequality a b c :=
by {
  sorry
}

end inequality_proof_l507_507386


namespace invalid_domain_for_mapping_l507_507004

noncomputable def f (x : ℝ) : ℝ := x^2

theorem invalid_domain_for_mapping (M N : Set ℝ) (hN : N = {1, 2}) (hM : M = {1, Real.sqrt 2, 2}) : ¬(∀ x ∈ M, f x ∈ N) :=
by {
  rw [hN, hM],
  intro h,
  have h2 : 2 ∈ M := by simp,
  specialize h 2 h2,
  have : f 2 = 4 := by norm_num,
  rw this at h,
  simp at h,
}

end invalid_domain_for_mapping_l507_507004


namespace curve_is_limaçon_l507_507671

def r (θ : ℝ) : ℝ := 1 + 2 * Real.cos θ

theorem curve_is_limaçon : 
  (∃ θ : ℝ, r θ = 1 + 2 * Real.cos θ) → 
  (∃ C : Type, C = "Limaçon") :=
by
  intro h
  use "Limaçon"
  sorry

end curve_is_limaçon_l507_507671


namespace ball_hits_ground_time_approximation_l507_507462

theorem ball_hits_ground_time_approximation :
  let y (t : ℝ) := -16 * t^2 - 12 * t + 72 in
  ∃ t : ℝ, y t = 0 ∧ abs (t - 1.78) < 0.005 :=
by
  sorry

end ball_hits_ground_time_approximation_l507_507462


namespace total_money_spent_l507_507371

def time_in_minutes_at_arcade : ℕ := 3 * 60
def cost_per_interval : ℕ := 50 -- in cents
def interval_duration : ℕ := 6 -- in minutes
def total_intervals : ℕ := time_in_minutes_at_arcade / interval_duration

theorem total_money_spent :
  ((total_intervals * cost_per_interval) = 1500) := 
by
  sorry

end total_money_spent_l507_507371


namespace ellipse_equation_k_range_l507_507260

variables (a b : ℝ) (e : ℝ) (k m : ℝ) (G : ℝ × ℝ)

-- Conditions
def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ (e = 1 / 2) ∧ (x, y) = (1, 3 / 2)

-- 1. Proving the equation of the ellipse
theorem ellipse_equation (h : is_ellipse a b 1 (3 / 2)) :
  ∀ x y : ℝ, (x^2 / 4) + (y^2 / 3) = 1 :=
sorry

-- 2. Proving the range of k
theorem k_range (h : is_ellipse a b 1 (3 / 2)) 
  (h_int : ∃ (Mx My Nx Ny : ℝ), ∀ k m : ℝ, k ≠ 0 → 
            (G = (1 / 8, 0)) → 
            (Mx, My) ≠ (Nx, Ny) ∧
            ∃ Qx Qy : ℝ, (Qx, Qy) = (- (4 * k * m / (3 + 4 * k^2)), 3 * m / (3 + 4 * k^2)) ∧
            (4 * k^2 + 8 * k * m + 3 = 0) ∧
            (Qy = -(1 / k) * (Qx - 1 / 8))) :
  ∀ k : ℝ, k^2 > 1 / 20 → k > sqrt 5 / 10 ∨ k < -sqrt 5 / 10 :=
sorry

end ellipse_equation_k_range_l507_507260


namespace find_number_of_white_balls_l507_507400

-- Define the conditions
variables (n k : ℕ)
axiom k_ge_2 : k ≥ 2
axiom prob_white_black : (n * k) / ((n + k) * (n + k - 1)) = n / 100

-- State the theorem
theorem find_number_of_white_balls (n k : ℕ) (k_ge_2 : k ≥ 2) (prob_white_black : (n * k) / ((n + k) * (n + k - 1)) = n / 100) : n = 19 :=
sorry

end find_number_of_white_balls_l507_507400


namespace solution_set_l507_507275

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (h_deriv : ∀ x, 2 * f' x - f x < 0)
variable (h_val : f (Real.log 2) = 2)

theorem solution_set (h_f : ∀ x, f' x = derivative f x) : 
  ∀ x : ℝ, (0 < x ∧ x < 2) → (f (Real.log x) - Real.sqrt (2 * x) > 0) := 
by 
  sorry

end solution_set_l507_507275


namespace bobby_blocks_l507_507204

theorem bobby_blocks 
  (initial_blocks : ℕ)
  (given_blocks : ℕ)
  (initial_blocks_eq : initial_blocks = 2)
  (given_blocks_eq : given_blocks = 6) :
  initial_blocks + given_blocks = 8 :=
by
  rw [initial_blocks_eq, given_blocks_eq]
  sorry -- Proof is skipped

end bobby_blocks_l507_507204


namespace financial_loss_correct_l507_507979

noncomputable def financial_result : ℝ :=
  let initial_rubles : ℝ := 60000
  let initial_dollar_selling_rate : ℝ := 59.65
  let initial_dollar_buying_rate : ℝ := 56.65
  let annual_interest_rate : ℝ := 0.015
  let duration_in_months : ℝ := 6
  let final_dollar_selling_rate : ℝ := 58.95
  let final_dollar_buying_rate : ℝ := 55.95

  -- Step 1: Convert Rubles to Dollars
  let amount_in_usd : ℝ := initial_rubles / initial_dollar_selling_rate

  -- Step 2: Calculate the amount of USD after 6 months with interest
  let effective_interest_rate : ℝ := annual_interest_rate / 2
  let amount_in_usd_after_6_months : ℝ := amount_in_usd * (1 + effective_interest_rate)

  -- Step 3: Convert USD back to Rubles
  let final_amount_in_rubles : ℝ := amount_in_usd_after_6_months * final_dollar_buying_rate

  -- Step 4: Calculate the financial result
  initial_rubles - final_amount_in_rubles

theorem financial_loss_correct : financial_result ≈ 3309 :=
by
  sorry

end financial_loss_correct_l507_507979


namespace arithmetic_mean_of_fractions_l507_507913

theorem arithmetic_mean_of_fractions : 
  (3 : ℚ) / 8 + (5 : ℚ) / 9 = 2 * (67 : ℚ) / 144 :=
by {
  rw [(show (3 : ℚ) / 8 + (5 : ℚ) / 9 = (27 : ℚ) / 72 + (40 : ℚ) / 72, by sorry),
      (show (27 : ℚ) / 72 + (40 : ℚ) / 72 = (67 : ℚ) / 72, by sorry),
      (show (2 : ℚ) * (67 : ℚ) / 144 = (67 : ℚ) / 72, by sorry)]
}

end arithmetic_mean_of_fractions_l507_507913


namespace central_cell_value_l507_507764

noncomputable def common_sum (nums : List ℕ) : ℕ :=
  let S := (nums.sum / 3)
  S

theorem central_cell_value :
  ∀ (nums : List ℕ),
    nums.length = 9 ∧ 
    nums.sum = 2019 ∧ 
    (∀ r c d1 d2 : List ℕ, r.sum = c.sum ∧ c.sum = d1.sum ∧ d1.sum = d2.sum) →
    nums.nth 4 = some (673 / 3) := 
  by
    intros nums h
    have h1 : nums.sum = 2019 := h.2.1
    have h2 : ∀ r c d1 d2 : List ℕ, r.sum = c.sum ∧ c.sum = d1.sum ∧ d1.sum = d2.sum := h.2.2
    let S := common_sum nums 
    let c := S * 1 / 3
    have c_value_correct : c = 673 / 3 := by sorry
    rw ← c_value_correct
    sorry

end central_cell_value_l507_507764


namespace hyperbola_equation_l507_507750

theorem hyperbola_equation :
  ∃ (a b c : ℝ), 
    ( (4 * x^2 + y^2 = 64) ∧ 
      (e1 e2 : ℝ), (e1 = sqrt (c^2 - a^2) / a ∧ e2 = 1 / e1) ∧ 
      (c = 4 * sqrt 3) ∧ 
      (e2 = 2 / sqrt 3) ∧ 
      (a = 6) ∧ 
      (b = 2 * sqrt 3) ∧ 
      (c^2 = a^2 + b^2) ∧ 
      ( a > 0 ∧ b > 0)
    ) →
    (y^2 - 3 * x^2 = 36) :=
by
  -- The proof is to be provided here
  sorry

end hyperbola_equation_l507_507750


namespace job_adjustment_count_l507_507873

theorem job_adjustment_count :
  let n := 5 in
  let total_permutations := nat.factorial n in
  total_permutations - 1 = 119 :=
by
  sorry

end job_adjustment_count_l507_507873


namespace find_divisor_l507_507596

theorem find_divisor :
  ∃ d : ℕ, (d = 859560) ∧ ∃ n : ℕ, (n + 859622) % d = 0 ∧ n = 859560 :=
by
  sorry

end find_divisor_l507_507596


namespace list_price_formula_l507_507990

theorem list_price_formula (C L : ℝ) (h : L = C * 1.7073) :
  0.82 * L = 1.40 * C :=
by
  have h1 : 1.7073 = 1.40 / 0.82 := by norm_num
  have h2 : L = C * (1.40 / 0.82) := by rw [h, h1]
  have h3 : 0.82 * L = 0.82 * (C * (1.40 / 0.82)) := by rw h2
  have h4 : 0.82 * L = C * 1.40 := by ring
  exact h4

end list_price_formula_l507_507990


namespace find_trajectory_l507_507252

noncomputable def trajectory_equation (x y : ℝ) : Prop :=
  (y - 1) * (y + 1) / ((x + 1) * (x - 1)) = -1 / 3

theorem find_trajectory (x y : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  trajectory_equation x y → x^2 + 3 * y^2 = 4 :=
by
  sorry

end find_trajectory_l507_507252


namespace incenter_touch_l507_507466

theorem incenter_touch (a b c a_1 b_1 c_1 s : ℝ)
  (s_def : s = (a + b + c) / 2)
  (A1_def : a_1 = 2 * ((exactly_defined\partial (a, b)) \cos (exactly_defined\partial (angle α))"
  : 
  (a_1 * a_1)/(a * (s - a)) = (b_1 * b_1)/(b * (s - b)) (c_1 * c_1)/(c * (s - c)) :=
begin
  sorry,
end

end incenter_touch_l507_507466


namespace not_covered_by_homothetic_polygons_l507_507818

structure Polygon :=
  (vertices : Set (ℝ × ℝ))

def homothetic (M : Polygon) (k : ℝ) (O : ℝ × ℝ) : Polygon :=
  {
    vertices := {p | ∃ (q : ℝ × ℝ) (hq : q ∈ M.vertices), p = (O.1 + k * (q.1 - O.1), O.2 + k * (q.2 - O.2))}
  }

theorem not_covered_by_homothetic_polygons (M : Polygon) (k : ℝ) (h : 0 < k ∧ k < 1)
  (O1 O2 : ℝ × ℝ) :
  ¬ (∀ p ∈ M.vertices, p ∈ (homothetic M k O1).vertices ∨ p ∈ (homothetic M k O2).vertices) := by
  sorry

end not_covered_by_homothetic_polygons_l507_507818


namespace trig_expression_evaluation_l507_507271

theorem trig_expression_evaluation
  (α : ℝ)
  (h : Real.tan α = 2) :
  (6 * Real.sin α + 8 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 5 := 
by 
  sorry

end trig_expression_evaluation_l507_507271


namespace polar_eq_of_line_segment_l507_507753

-- Define the Cartesian equation of the line segment
def cartesian_eq (x y : ℝ) : Prop := y = 1 - x

-- Define the interval for x
def x_interval (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1

-- Define the polar coordinate transformations
def polar_transform_x (ρ θ : ℝ) : ℝ := ρ * cos θ
def polar_transform_y (ρ θ : ℝ) : ℝ := ρ * sin θ

-- Define the polar equation of the line segment and the range for θ
theorem polar_eq_of_line_segment : 
  (∀ x y : ℝ, cartesian_eq x y → 
              ∃ ρ θ : ℝ, ρ = 1 / (cos θ + sin θ) ∧ 
              0 ≤ θ ∧ θ ≤ π / 2 ∧
              x = polar_transform_x ρ θ ∧ y = polar_transform_y ρ θ) :=
by
  sorry

end polar_eq_of_line_segment_l507_507753


namespace ratio_of_falls_l507_507839

variable (SteveFalls : ℕ) (StephFalls : ℕ) (SonyaFalls : ℕ)
variable (H1 : SteveFalls = 3)
variable (H2 : StephFalls = SteveFalls + 13)
variable (H3 : SonyaFalls = 6)

theorem ratio_of_falls : SonyaFalls / (StephFalls / 2) = 3 / 4 := by
  sorry

end ratio_of_falls_l507_507839


namespace circle_problem_l507_507499

theorem circle_problem (P : ℝ × ℝ) (QR : ℝ) (S : ℝ × ℝ) (k : ℝ)
  (h1 : P = (5, 12))
  (h2 : QR = 5)
  (h3 : S = (0, k))
  (h4 : dist (0,0) P = 13) -- OP = 13 from the origin to point P
  (h5 : dist (0,0) S = 8) -- OQ = 8 from the origin to point S
: k = 8 ∨ k = -8 :=
by sorry

end circle_problem_l507_507499


namespace problem_l507_507045

noncomputable def midpoint (p q : Point) : Point :=
{ x := (p.x + q.x) / 2, y := (p.y + q.y) / 2 }

def lies_between (A O B : Point) : Prop :=
 collinear {A, O, B} ∧ dist O A < dist O B

def line_through_midpoints (A B C D O M N : Point) : Prop :=
  let P := midpoint A D in
  let Q := midpoint B C in
  lies_on_line O M P ∧ lies_on_line O N Q

theorem problem (A B C D O M N : Point) 
  (h1 : lies_between A O B)
  (h2 : lies_between C O D)
  (h3 : line_through_midpoints A B C D O M N) : 
  dist O M / dist O N = dist A B / dist C D := 
sorry

end problem_l507_507045


namespace sequence_b5_eq_142_l507_507220

theorem sequence_b5_eq_142 :
  ∃ (b : ℕ → ℕ), 
    b 1 = 2 ∧ b 2 = 5 ∧ 
    (∀ n, n ≥ 3 → b n = 2 * b (n - 1) + 3 * b (n - 2)) ∧ 
    b 5 = 142 :=
begin
  sorry
end

end sequence_b5_eq_142_l507_507220


namespace hair_radius_in_scientific_notation_l507_507578

theorem hair_radius_in_scientific_notation :
  let nanometer_to_meter := 10^(-9)
  let hair_diameter_nm := 60000
  let hair_radius_meter := (hair_diameter_nm / 2) * nanometer_to_meter
  hair_radius_meter = 3 * 10^(-5)
:= by
  sorry

end hair_radius_in_scientific_notation_l507_507578


namespace problem_part1_problem_part2_l507_507285

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + 2 * Real.sqrt 3 * (Real.sin x) * (Real.cos x) + a

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (4 * x - Real.pi / 6) + 3

theorem problem_part1 (h : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x 2 ≥ 2) :
    ∃ a : ℝ, a = 2 ∧ 
    ∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6), 
    ∃ m : ℤ, x = (m * Real.pi / 2 + Real.pi / 12) ∨ x = (m * Real.pi / 2 + Real.pi / 4) := sorry

theorem problem_part2 :
    ∀ x ∈ Set.Icc 0 (Real.pi / 2), g x = 4 → 
    ∃ s : ℝ, s = Real.pi / 3 := sorry

end problem_part1_problem_part2_l507_507285


namespace max_volume_pyramid_l507_507255

-- The vertices of the pyramid
variables (O A B C : ℝ³)
variables (r : ℕ) (angleAOB : ℕ)

-- The conditions given:
-- 1. 'A', 'B', 'C' are on the surface of a sphere with radius 3.
def is_on_sphere (P : ℝ³) (center : ℝ³) (radius : ℝ) : Prop :=
  dist center P = radius

-- 2. 'O' is the center of the sphere.
def is_center (O : ℝ³) (center : ℝ³) : Prop :=
  O = center

-- 3. ∠AOB = 150 degrees.
def angle_AOB_is_150 (A B O : ℝ³) : Prop :=
  ∠(A - O) (B - O) = (5 * π) / 6

-- The Lean statement to prove:
theorem max_volume_pyramid (O A B C : ℝ³) (h_sphere_A : is_on_sphere A O 3)
  (h_sphere_B : is_on_sphere B O 3) (h_sphere_C : is_on_sphere C O 3)
  (h_center : is_center O O) (h_angle : angle_AOB_is_150 A B O) :
  volume_pyramid O A B C = 9 * sqrt 3 / 2 :=
sorry

end max_volume_pyramid_l507_507255


namespace remainder_b100_mod_81_l507_507388

def b (n : ℕ) := 7^n + 9^n

theorem remainder_b100_mod_81 : (b 100) % 81 = 38 := by
  sorry

end remainder_b100_mod_81_l507_507388


namespace find_x4_l507_507179

theorem find_x4 (x : ℝ) (h : sqrt (1 - x^2) + sqrt (1 + x^2) = sqrt 2) : x^4 = 1 :=
sorry

end find_x4_l507_507179


namespace product_of_roots_l507_507074

theorem product_of_roots : (∛(27) * (16)^(1/4) = 6) :=
by sorry

end product_of_roots_l507_507074


namespace no_meet_at_different_vertex_l507_507768

variable {A : Type} [Fintype A] [DecidableEq A]

-- Define the nonagon as a cycle
def regular_nonagon (n : ℕ) := Zmod n

-- Define the cars and their properties
variables (v : regular_nonagon 9 → ℕ) (T : ℕ) (initial_position : regular_nonagon 9)

-- Assumptions
axiom regular_nonagon_exists : ∃ (v : regular_nonagon 9 → ℕ) (T : ℕ),
  (∀ i, (v i) * T % 9 = 0) ∧
  (∃ t, v (regular_nonagon 9.castAdd (initial_position t)) * T % 9 = initial_position 0) 

-- Prove that cars can't meet at different vertex
theorem no_meet_at_different_vertex :
  ¬ ∃ (k : regular_nonagon 9), k ≠ initial_position 0 ∧
  (∀ i, v i * T % 9 = k) :=
sorry

end no_meet_at_different_vertex_l507_507768


namespace termite_ridden_but_not_collapsing_fraction_l507_507570

def homesOnGothamStreet (total_homes : ℕ) (termite_ridden_fraction : ℚ) (collapsing_fraction : ℚ) :=
  termite_ridden_fraction = 1 / 3 ∧ collapsing_fraction = 1 / 4

theorem termite_ridden_but_not_collapsing_fraction (total_homes : ℕ) (termite_ridden_fraction : ℚ) (collapsing_fraction : ℚ) :
  homesOnGothamStreet total_homes termite_ridden_fraction collapsing_fraction →
  (termite_ridden_fraction - (termite_ridden_fraction * collapsing_fraction)) = 1 / 4 :=
by
  intro h
  cases h
  rw [h_left, h_right]
  norm_num
  sorry

end termite_ridden_but_not_collapsing_fraction_l507_507570


namespace sqrt_two_minus_one_pow_zero_l507_507152

theorem sqrt_two_minus_one_pow_zero : (Real.sqrt 2 - 1)^0 = 1 := by
  sorry

end sqrt_two_minus_one_pow_zero_l507_507152


namespace markers_multiple_of_4_l507_507410

-- Definitions corresponding to conditions
def Lisa_has_12_coloring_books := 12
def Lisa_has_36_crayons := 36
def greatest_number_baskets := 4

-- Theorem statement
theorem markers_multiple_of_4
    (h1 : Lisa_has_12_coloring_books = 12)
    (h2 : Lisa_has_36_crayons = 36)
    (h3 : greatest_number_baskets = 4) :
    ∃ (M : ℕ), M % 4 = 0 :=
by
  sorry

end markers_multiple_of_4_l507_507410


namespace floor_S_sq_eq_4036079_l507_507011

noncomputable def S : ℝ :=
  ∑ i in Finset.range 2008, Real.sqrt (1 + (2 : ℝ) / (i + 1)^2 + (2 : ℝ) / (i + 2)^2)

theorem floor_S_sq_eq_4036079 : ⌊S^2⌋ = 4036079 := by
  sorry

end floor_S_sq_eq_4036079_l507_507011


namespace area_ratio_of_region_A_and_C_l507_507827

theorem area_ratio_of_region_A_and_C
  (pA : ℕ) (pC : ℕ) 
  (hA : pA = 16)
  (hC : pC = 24) :
  let sA := pA / 4
  let sC := pC / 6
  let areaA := sA * sA
  let areaC := (3 * Real.sqrt 3 / 2) * sC * sC
  (areaA / areaC) = (2 * Real.sqrt 3 / 9) :=
by
  sorry

end area_ratio_of_region_A_and_C_l507_507827


namespace number_of_men_in_second_group_l507_507879

-- Definitions based on conditions
def man_hours_1km (men days hours : ℕ) : ℕ := men * days * hours
def man_hours_2km (man_hours_1km : ℕ) : ℕ := 2 * man_hours_1km
def men_needed (total_man_hours days hours : ℕ) : ℕ := total_man_hours / (days * hours)

-- Given values
def first_group_men := 30
def first_group_days := 12
def first_group_hours := 8
def second_group_days := 19.2
def second_group_hours := 15

-- Correct answer
def correct_men := 20

-- Problem statement in Lean
theorem number_of_men_in_second_group :
  men_needed (man_hours_2km (man_hours_1km first_group_men first_group_days first_group_hours)) second_group_days second_group_hours = correct_men :=
by
  sorry

end number_of_men_in_second_group_l507_507879


namespace jeff_prime_probability_l507_507365

def prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_prime (n : ℕ) : bool :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes : List ℕ := [2, 3, 5, 7, 11]

def card_numbers : List ℕ := List.range (12 + 1)
def initial_prob (card : ℕ) : ℚ := if card ∈ primes then 5 / 12 else 7 / 12

def move (pos : ℕ) (instruction : ℤ) : ℕ :=
  if pos + instruction < 1 then 1 else (if pos + instruction > 12 then 12 else pos + instruction)

def spinner_result (spinner : ℚ) (pos : ℕ) : List ℕ := [
  move pos (-1),
  move pos (-1),
  move pos 2,
  move pos 2
  ]

def path_prob (pos : ℕ) : ℚ :=
  let possible_ends := List.join (spinner_result 1 / 2 pos).map (spinner_result 1 / 2)
  let prime_ends := possible_ends.filter is_prime
  prime_ends.length / possible_ends.length

def final_prob : ℚ :=
  card_numbers.foldl
    (λ acc card, acc + (initial_prob card * path_prob card))
    0

theorem jeff_prime_probability : final_prob = 5 / 16 := by
  sorry

end jeff_prime_probability_l507_507365


namespace sum_first_30_gareth_seq_l507_507581

def gareth_seq : ℕ → ℕ
| 0       := 10
| 1       := 8
| (n + 2) := abs (gareth_seq n - gareth_seq (n + 1))

noncomputable def sum_gareth_seq (n : ℕ) : ℕ :=
  (List.range n).map gareth_seq |>.sum

theorem sum_first_30_gareth_seq : sum_gareth_seq 30 = 64 :=
by
  sorry -- Proof omitted, only the statement is required.

end sum_first_30_gareth_seq_l507_507581


namespace johns_total_cost_l507_507367

variable (C_s C_d : ℝ)

theorem johns_total_cost (h_s : C_s = 20) (h_d : C_d = 0.5 * C_s) : C_s + C_d = 30 := by
  sorry

end johns_total_cost_l507_507367


namespace letters_with_dot_only_is_8_l507_507335

-- Define the conditions
variables (letters_with_both : Nat)
variables (letters_with_straight_line_only : Nat)
variables (total_letters : Nat)
variable (letters_with_dot_only : Nat)

-- Assume the given conditions
axiom h1 : letters_with_both = 8
axiom h2 : letters_with_straight_line_only = 24
axiom h3 : total_letters = 40

-- Define the equation based on the conditions
def equation := total_letters = letters_with_both + letters_with_straight_line_only + letters_with_dot_only

-- State the main theorem we want to prove
theorem letters_with_dot_only_is_8 : letters_with_dot_only = 8 :=
by
  rw [equation, h1, h2, h3]
  show 40 = 8 + 24 + letters_with_dot_only
  sorry

end letters_with_dot_only_is_8_l507_507335


namespace monochromatic_triangle_l507_507417

def R₃ (n : ℕ) : ℕ := sorry

theorem monochromatic_triangle {n : ℕ} (h1 : R₃ 2 = 6)
  (h2 : ∀ n, R₃ (n + 1) ≤ (n + 1) * R₃ n - n + 1) :
  R₃ n ≤ 3 * Nat.factorial n :=
by
  induction n with
  | zero => sorry -- base case proof
  | succ n ih => sorry -- inductive step proof

end monochromatic_triangle_l507_507417


namespace factor_expression_l507_507667

theorem factor_expression (x : ℝ) : 100 * x ^ 23 + 225 * x ^ 46 = 25 * x ^ 23 * (4 + 9 * x ^ 23) :=
by
  -- Proof steps will go here
  sorry

end factor_expression_l507_507667


namespace choose_objects_l507_507840

theorem choose_objects :
  let total = 4960
  let adjacent_pairs = 32 * 28
  let diametric_pairs = 16 * 28
  let over_counted = 32
  total - adjacent_pairs - diametric_pairs + over_counted = 3648 :=
by
  -- Let the total number of ways to choose 3 objects from 32 be:
  let total : ℕ := 4960
  -- Let the number of ways in which two of the chosen objects are adjacent be:
  let adjacent_pairs : ℕ := 32 * 28
  -- Let the number of ways in which two of the chosen objects are diametrically opposite be:
  let diametric_pairs : ℕ := 16 * 28
  -- Let the over-counted cases where the chosen objects are both adjacent and diametrically opposite be:
  let over_counted : ℕ := 32
  -- Prove the given equation:
  exact
  calc
    total - adjacent_pairs - diametric_pairs + over_counted
    = 4960 - (32 * 28) - (16 * 28) + 32 : by sorry
    ... = 3648 : by sorry

end choose_objects_l507_507840


namespace tulip_gift_count_l507_507418

theorem tulip_gift_count :
  let max_tulips := 11 in
  let odd_tulip_combinations := ∑ k in (Finset.range (max_tulips + 1)), if k % 2 = 1 then Nat.choose max_tulips k else 0 in
  odd_tulip_combinations = 1024 :=
by
  sorry

end tulip_gift_count_l507_507418


namespace finite_monic_poly_unit_circle_roots_roots_roots_of_unity_l507_507986

noncomputable section

-- Part (a)
theorem finite_monic_poly_unit_circle_roots (n : ℕ) (h : 0 < n) : 
  {P : Polynomial ℤ // P.monic ∧ P.degree = n ∧ ∀ x, P.isRoot x → ∥x∥ = 1}.finite :=
  sorry

-- Part (b)
theorem roots_roots_of_unity (P : Polynomial ℤ) (hmonic : P.monic) (hroots : ∀ x, P.isRoot x → ∥x∥ = 1) :
  ∃ m : ℕ, 0 < m ∧ ∀ x, P.isRoot x → x^m = 1 :=
  sorry

end finite_monic_poly_unit_circle_roots_roots_roots_of_unity_l507_507986


namespace proof_problem_l507_507658

def sixtyPercentLess (n : ℝ) : ℝ := n - 0.605 * n

def twentySevenPercentLess (n : ℝ) : ℝ := n - 0.273 * n

def initialNumber : ℝ := 4500.78

def intermediateResult : ℝ := sixtyPercentLess initialNumber

def finalResult : ℝ := twentySevenPercentLess intermediateResult

theorem proof_problem : finalResult ≈ 1292.29 :=
by sorry

end proof_problem_l507_507658


namespace smallest_integer_with_eight_factors_l507_507538

theorem smallest_integer_with_eight_factors : ∃ N : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) ∧ ∀ M : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) → N ≤ M :=
sorry -- Proof not provided.

end smallest_integer_with_eight_factors_l507_507538


namespace solve_for_x_l507_507597

theorem solve_for_x (x : ℝ) (h₁ : sqrt (7 * x / 5) = x) (h₂ : x ≠ 0) : x = 7 / 5 :=
by sorry

end solve_for_x_l507_507597


namespace smallest_base10_integer_l507_507950

theorem smallest_base10_integer :
  ∃ a b : ℕ, a > 3 ∧ b > 3 ∧ (2 * a + 2 = 3 * b + 3) ∧ (2 * a + 2 = 18) :=
by
  existsi 8 -- assign specific solutions to a
  existsi 5 -- assign specific solutions to b
  exact sorry -- follows from the validations done above

end smallest_base10_integer_l507_507950


namespace max_f_value_l507_507695

-- Definition of the function f(x, y)
def f (x y : ℝ) : ℝ :=
  (x - y)^2 + (4 + real.sqrt (1 - x^2) + real.sqrt (1 - y^2 / 9))^2

-- Main statement: Prove that the maximum value of f(x,y) is 28 + 6 * sqrt(3)
theorem max_f_value : 
  ∃ x y : ℝ, f x y = 28 + 6 * real.sqrt 3 :=
by
  sorry

end max_f_value_l507_507695


namespace clubs_equal_students_l507_507347

-- Define the concepts of Club and Student
variable (Club Student : Type)

-- Define the membership relations
variable (Members : Club → Finset Student)
variable (Clubs : Student → Finset Club)

-- Define the conditions
axiom club_membership (c : Club) : (Members c).card = 3
axiom student_club_membership (s : Student) : (Clubs s).card = 3

-- The goal is to prove that the number of clubs is equal to the number of students
theorem clubs_equal_students [Fintype Club] [Fintype Student] : Fintype.card Club = Fintype.card Student := by
  sorry

end clubs_equal_students_l507_507347


namespace arithmetic_mean_of_fractions_l507_507946

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 3/8
  let b : ℚ := 5/9
  (a + b) / 2 = 67 / 144 := 
by
  sorry

end arithmetic_mean_of_fractions_l507_507946


namespace sum_of_primitive_roots_mod_11_l507_507953

noncomputable def isPrimitiveRoot (g : ℕ) (n : ℕ) : Prop :=
  ∀ k : ℕ, (1 ≤ k ∧ k < n) → ∃ m : ℕ, (g ^ m) % n = k

theorem sum_of_primitive_roots_mod_11 :
  let s := {n : ℕ | n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} ∧ isPrimitiveRoot n 11} in
  ∑ x in s, x = 23 :=
by
  sorry

end sum_of_primitive_roots_mod_11_l507_507953


namespace planar_vector_magnitude_l507_507729

variables {a b : EuclideanSpace ℝ (Fin 2)}

theorem planar_vector_magnitude
  (h₁ : ∥a∥ = 1)
  (h₂ : ∥a - 2 • b∥ = Real.sqrt 21)
  (h₃ : real_inner a b = (1 : ℝ) * ∥b∥ * Real.cos (120 * Real.pi / 180)) : 
  ∥b∥ = 2 :=
sorry

end planar_vector_magnitude_l507_507729


namespace ratio_of_x_intercepts_l507_507096

theorem ratio_of_x_intercepts (b s t : ℝ) (h_b : b ≠ 0)
  (h1 : 0 = 8 * s + b)
  (h2 : 0 = 4 * t + b) :
  s / t = 1 / 2 :=
by
  sorry

end ratio_of_x_intercepts_l507_507096


namespace red_eggs_distribution_probability_l507_507021

noncomputable def probability_red_eggs_in_both_boxes (total_eggs : ℕ) (red_eggs : ℕ) (larger_box_eggs : ℕ) (smaller_box_eggs : ℕ) : ℚ :=
(let total_combinations := Nat.choose total_eggs smaller_box_eggs in
 let favorable_combinations := Nat.choose red_eggs 1 * Nat.choose (total_eggs - red_eggs) (smaller_box_eggs - 1) +
                              Nat.choose red_eggs 2 * Nat.choose (total_eggs - red_eggs) (smaller_box_eggs - 2) in
 favorable_combinations / total_combinations)

theorem red_eggs_distribution_probability :
  probability_red_eggs_in_both_boxes 16 3 10 6 = 3 / 4 :=
by
  sorry

end red_eggs_distribution_probability_l507_507021


namespace average_speed_l507_507174

theorem average_speed :
  let v1 := 5  -- speed in m/s
  let t1 := 20 / 60.0  -- time in hours
  let d1 := v1 * (20 * 60) / 1000  -- distance in km
  let v2 := 4  -- speed in km/h
  let t2 := 1.5  -- time in hours
  let d2 := v2 * t2  -- distance in km
  let d_total := d1 + d2  -- total distance
  let t_total := t1 + t2  -- total time
  let v_average := d_total / t_total -- average speed in km/h
  in v_average = 72 / 11 := 
by
  sorry

end average_speed_l507_507174


namespace f_fraction_neg_1987_1988_l507_507851

-- Define the function f and its properties
def f : ℚ → ℝ := sorry

axiom functional_eq (x y : ℚ) : f (x + y) = f x * f y - f (x * y) + 1
axiom not_equal_f : f 1988 ≠ f 1987

-- Prove the desired equality
theorem f_fraction_neg_1987_1988 : f (-1987 / 1988) = 1 / 1988 :=
by
  sorry

end f_fraction_neg_1987_1988_l507_507851


namespace arithmetic_mean_of_fractions_l507_507930

theorem arithmetic_mean_of_fractions (a b : ℚ) (h1 : a = 3 / 8) (h2 : b = 5 / 9) :
  (a + b) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_fractions_l507_507930


namespace smallest_positive_integer_with_eight_factors_l507_507516

theorem smallest_positive_integer_with_eight_factors :
  ∃ n : ℕ, (∀ m : ℕ, (m < n ∧ (∀ d : ℕ, d | m → d = 1 ∨ d = m) → (∃ a b : ℕ, distinct_factors_count m a b ∧ a = 8)) → n = 24) :=
by
  sorry

def distinct_factors_count (n : ℕ) (a b : ℕ) : Prop :=
  ∃ (p q : ℕ), prime p ∧ prime q ∧ n = p^a * q^b ∧ (a + 1) * (b + 1) = 8

end smallest_positive_integer_with_eight_factors_l507_507516


namespace part_a_l507_507569

variables (m_a m_b m_c R : ℝ)
variables (A B C G O : Point)

-- Conditions
axiom centroid (G : Point) (A B C : Point) : G = centroid A B C
axiom circumcenter (O : Point) (A B C : Point) : O = circumcenter A B C
axiom circumradius (A B C O : Point) (R : ℝ) : AO^2 + BO^2 + CO^2 = 3 * R^2
axiom medians (A B C G : Point) : 
  (m_a = AG ∧ m_b = BG ∧ m_c = CG)

theorem part_a (h : AO^2 + BO^2 + CO^2 = 3 * R^2) : 
  m_a ^ 2 + m_b ^ 2 + m_c ^ 2 ≤ 27 / 4 * R ^ 2 := 
sorry

end part_a_l507_507569


namespace linda_original_amount_l507_507745

-- Define the original amount of money Lucy and Linda have
variables (L : ℕ) (lucy_initial : ℕ := 20)

-- Condition: If Lucy gives Linda $5, they have the same amount of money.
def condition := (lucy_initial - 5) = (L + 5)

-- Theorem: The original amount of money that Linda had
theorem linda_original_amount (h : condition L) : L = 10 := 
sorry

end linda_original_amount_l507_507745


namespace intersection_complement_l507_507301

open Set

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)
variable [hU : U = univ] [hA : A = {x | -2 ≤ x ∧ x ≤ 3}] [hB : B = {x | x < -1 ∨ x > 4}]

theorem intersection_complement :
  A ∩ (U \ B) = {x | -1 ≤ x ∧ x ≤ 3} :=
by
  rw [hU, hA, hB]
  ext x
  simp
  sorry

end intersection_complement_l507_507301


namespace triangle_symmetric_vertex_in_boundary_or_inside_l507_507093

-- Let T be a triangle contained within a centrally symmetric polygon M
-- Let T' be the symmetric image of T with respect to a point P inside T
-- Prove that T' has at least one vertex located inside polygon M or on its boundary

theorem triangle_symmetric_vertex_in_boundary_or_inside (T M : Set ℝ) (P : ℝ) (T_is_triangle : is_triangle T) (M_symmetry : is_centrally_symmetric M) (T_in_polygon_M : T ⊆ M) (P_in_T : P ∈ T) :
  ∃ v, v ∈ T' ∧ (v ∈ M ∨ ∃ e ∈ (boundary M), v ∈ e) :=
sorry

end triangle_symmetric_vertex_in_boundary_or_inside_l507_507093


namespace proof_equivalence_l507_507704

open Real

-- Definition of functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := a * x * exp x
def g (a : ℝ) (x : ℝ) : ℝ := (2 * log x + x + a) / x

-- Main theorem stating the desired properties
theorem proof_equivalence (a : ℝ) :
  (∃ x : ℝ, (differentiable ℝ (f a)) ∧ ¬ (f a has_local_min x)) ∧
  (∃ x : ℝ, (differentiable ℝ (g a)) ∧ (g a has_local_max x)) ∧
  (∀ x : ℝ, (f a x ≥ g a x) → a = 1) ∧
  (∀ x : ℝ, @roots ℝ (λ x => f a x - g a x) x ≤ 2) :=
by
  sorry

end proof_equivalence_l507_507704


namespace smallest_three_digit_multiple_of_13_l507_507107

theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, (100 ≤ n) ∧ (n < 1000) ∧ (n % 13 = 0) ∧ (∀ k : ℕ, (100 ≤ k) ∧ (k < 1000) ∧ (k % 13 = 0) → n ≤ k) → n = 104 :=
by
  sorry

end smallest_three_digit_multiple_of_13_l507_507107


namespace solve_for_n_l507_507049

theorem solve_for_n :
  ∀ n : ℝ, (n-5)^4 = (1 / 16)^(-1) → n = 7 :=
by
  intro n h
  sorry

end solve_for_n_l507_507049


namespace graph_symmetry_about_line_l507_507290

def f (x : ℝ) : ℝ := (Real.sin x) * (Real.cos x) - (Real.sqrt 3) / 2 * (Real.cos (2 * x))

theorem graph_symmetry_about_line :
  (∀ x : ℝ, f (2 * x - π / 3) = f (2 * (x - π / 6))) :=
by
  intros x
  sorry

end graph_symmetry_about_line_l507_507290


namespace value_of_y_l507_507094

noncomputable def triangle ::= {a b c : ℝ}

def is_acute (T : triangle) : Prop :=
  ∀ ⦃A B C : ℝ⦄ (hA : A ∠ B < π / 2) (hB : B ∠ C < π / 2) (hC : C ∠ A < π / 2), 
  T.a < π / 2 ∧ T.b < π / 2 ∧ T.c < π / 2

def altitudes_divide (T : triangle) (s1 s2 s3 y : ℝ) : Prop :=
  -- Representing the side segments as given lengths
  (s1 = 7) ∧ (s2 = 4) ∧ (s3 = 3) ∧ (∀ x : ℝ, x = y → y = 121 / 3)

theorem value_of_y (T : triangle) (s1 s2 s3 y : ℝ) :
  is_acute T → altitudes_divide T s1 s2 s3 y → y = 121 / 3 :=
by
  intro h1 h2
  sorry

end value_of_y_l507_507094


namespace complex_number_in_first_quadrant_l507_507063

open Complex

theorem complex_number_in_first_quadrant :
  let z := (2 + Complex.I) * (1 + Complex.I)
  ∃ q : ℕ, (q = 1 ∧ z.re > 0 ∧ z.im > 0) :=
by
  let z : ℂ := (2 + Complex.I) * (1 + Complex.I)
  have h1 : z = 1 + 3 * Complex.I := by sorry
  have h2 : z.re = 1 := by sorry
  have h3 : z.im = 3 := by sorry
  use 1
  split
  · exact rfl
  · split
    · calc
        z.re = 1 := by exact h2
        _ > 0 := by norm_num
    · calc
        z.im = 3 := by exact h3
        _ > 0 := by norm_num

end complex_number_in_first_quadrant_l507_507063


namespace triangle_problem_l507_507780

theorem triangle_problem
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : (√3 * real.sin C - 2 * real.cos A) * real.sin B = (2 * real.sin A - real.sin C) * real.cos B)
  (h2 : a^2 + c^2 = 4 + √3)
  (area : real := (1 / 2) * a * c * real.sin B)
  (h3 : area = (3 + √3) / 4) :
  B = π / 3 ∧ a + b + c = (√6 + 2 * √3 + 3 * √2) / 2 :=
by
  sorry

end triangle_problem_l507_507780


namespace triangle_sine_relation_l507_507761

theorem triangle_sine_relation
  (A B C : ℝ)
  (a b c : ℝ)
  (h_triangle_abc : ∀ x : ℝ, ∃ R : ℝ, x^2 = a^2 + b^2 = 2c^2)
  (h_law_of_sines: ∀ (a b c : ℝ) (A B C : ℝ), a / sin A = b / sin B ∧ b / sin B = c / sin C)
  : (sin A)^2 + (sin B)^2 = 2 * (sin C)^2 :=
begin
  sorry -- Proof omitted
end

end triangle_sine_relation_l507_507761


namespace distance_between_centers_l507_507420

-- Declare radii of the circles and the shortest distance between points on the circles
def R := 28
def r := 12
def d := 10

-- Define the problem to prove the distance between the centers
theorem distance_between_centers (R r d : ℝ) (hR : R = 28) (hr : r = 12) (hd : d = 10) : 
  ∀ OO1 : ℝ, OO1 = 6 :=
by sorry

end distance_between_centers_l507_507420


namespace volume_of_annular_region_l507_507088

-- Define the radii of the two spheres
def smaller_radius : ℝ := 3
def larger_radius : ℝ := 6

-- Volume of a sphere of radius r
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Volumes of the smaller and larger spheres
def smaller_volume : ℝ := volume_of_sphere smaller_radius
def larger_volume : ℝ := volume_of_sphere larger_radius

-- The volume of the region within the larger sphere and not within the smaller sphere
def volume_difference : ℝ := larger_volume - smaller_volume

-- The theorem to be proven
theorem volume_of_annular_region : volume_difference = 252 * Real.pi :=
by
  sorry

end volume_of_annular_region_l507_507088


namespace johns_total_cost_l507_507366

variable (C_s C_d : ℝ)

theorem johns_total_cost (h_s : C_s = 20) (h_d : C_d = 0.5 * C_s) : C_s + C_d = 30 := by
  sorry

end johns_total_cost_l507_507366


namespace find_sum_of_coordinates_l507_507446

variable {α : Type*}

-- Definitions we derive from the problem conditions
def f (x : α) : α
def f_inv (y : α) : α := classical.some (classical.some_spec $ classical.some_spec $ exists_inverse f)

-- Assuming that point (3,4) is on the graph of y = f(x) / 3
axiom f_at_3_is_12 : f 3 = 12

-- The main theorem we want to prove: the sum of the coordinates of the point that 
-- must be on the graph of y = f_inv(x) / 4, given the above axiom.
theorem find_sum_of_coordinates : 
  let point := (12 : α, f_inv (12) / 4) in
  ∃ (x y : α), f_inv (12) = 3 ∧ x = 12 ∧ y = 3 / 4 ∧ x + y = (12 + 3 / 4) :=
by
  sorry

end find_sum_of_coordinates_l507_507446


namespace sum_arithmetic_series_l507_507641

theorem sum_arithmetic_series (n : ℕ) :
  (∑ k in Finset.range (n + 1), 3 * k + 2) = (n + 1) * (3 * n + 1) / 2 :=
by
  sorry

end sum_arithmetic_series_l507_507641


namespace smallest_integer_with_eight_factors_l507_507544

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, 
  ∀ m : ℕ, (∀ p : ℕ, ∃ k : ℕ, m = p^k → (k + 1) * (p + 1) = 8) → (n ≤ m) ∧ 
  (∀ d : ℕ, d ∣ n → d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 24) :=
sorry

end smallest_integer_with_eight_factors_l507_507544


namespace handshakes_at_conference_l507_507085

/-- 
There are 5 representatives from each of 5 companies at a conference. 
At the start of the conference, every person shakes hands once with 
every person from other companies except the members of company C.
Prove that the total number of handshakes is 150. 
-/
theorem handshakes_at_conference :
  let num_companies := 5
  let rep_per_company := 5
  let total_persons := num_companies * rep_per_company
  let num_handshakes_per_person := (total_persons - rep_per_company) - rep_per_company -- 15 handshakes each for A, B, D, E representatives
  let participating_persons := 4 * rep_per_company -- 20 persons from A, B, D, E
  let total_handshakes := (participating_persons * num_handshakes_per_person) / 2
  total_handshakes = 150 :=
by
  dsimp
  sorry

end handshakes_at_conference_l507_507085


namespace length_of_AC_l507_507427

theorem length_of_AC {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] (a b c : ℝ) :
  a = 4 → c = 2 → ∠A = 60 → b = sqrt(13) + 1 :=
by
  assume h1 h2 h3,
  sorry

end length_of_AC_l507_507427


namespace question_l507_507035

section

variable (x : ℝ)
variable (p q : Prop)

-- Define proposition p: ∀ x in [0,1], e^x ≥ 1
def Proposition_p : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → Real.exp x ≥ 1

-- Define proposition q: ∃ x in ℝ such that x^2 + x + 1 < 0
def Proposition_q : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

-- The problem to prove: p ∨ q
theorem question (p q : Prop) (hp : Proposition_p) (hq : ¬ Proposition_q) : p ∨ q := by
  sorry

end

end question_l507_507035


namespace part1_part2_part3_l507_507056

-- Definitions of conditions
def sum_even (n : ℕ) : ℕ := n * (n + 1)
def sum_even_between (a b : ℕ) : ℕ := sum_even b - sum_even a

-- Problem 1: Prove that for n = 8, S = 72
theorem part1 (n : ℕ) (h : n = 8) : sum_even n = 72 := by
  rw [h]
  exact rfl

-- Problem 2: Prove the general formula for the sum of the first n consecutive even numbers
theorem part2 (n : ℕ) : sum_even n = n * (n + 1) := by
  exact rfl

-- Problem 3: Prove the sum of 102 to 212 is 8792 using the formula
theorem part3 : sum_even_between 50 106 = 8792 := by
  sorry

end part1_part2_part3_l507_507056


namespace smallest_integer_with_eight_factors_l507_507511

theorem smallest_integer_with_eight_factors:
  ∃ n : ℕ, ∀ (d : ℕ), d > 0 → d ∣ n → 8 = (divisor_count n) 
  ∧ (∀ m : ℕ, m > 0 → (∀ (d : ℕ), d > 0 → d ∣ m → 8 = (divisor_count m)) → n ≤ m) → n = 24 :=
begin
  sorry
end

end smallest_integer_with_eight_factors_l507_507511


namespace boys_and_girls_arrangement_l507_507714

def num_boys : Nat := 4
def num_girls : Nat := 3
def total_arrangements : Nat := 2304

theorem boys_and_girls_arrangement 
  (boys_le1_adj_girl : ∀ (b₁ b₂ b₃ b₄ g₁ g₂ g₃ : Prop), 
                        (b₁ (adjacent g₁) ∧ b₂ (adjacent g₂) ∧ b₃ (adjacent g₃) → 
                        ∀ b, b (adjacent g₁) ∨ b (adjacent g₂) ∨ b (adjacent g₃))): 
  total_arrangements = 2304 :=
sorry

end boys_and_girls_arrangement_l507_507714


namespace angles_and_distances_l507_507321

-- We start by defining the properties and equations of the problem
variables (p λ : ℝ)
hypothesis (h_p_pos : p > 0)
hypothesis (h_λ_gt_one : λ > 1)

-- Representing points and lines in the Cartesian plane
structure Point :=
(x : ℝ)
(y : ℝ)

def F : Point := ⟨p / 2, 0⟩
def A : Point := ⟨-p / 2, 0⟩

-- Define the parabola y^2 = 2px
def on_parabola (P : Point) : Prop :=
P.y ^ 2 = 2 * p * P.x

variables (M N E : Point)

-- Points M and N on the parabola
hypothesis (h_M_on_parabola : on_parabola p M)
hypothesis (h_N_on_parabola : on_parabola p N)

-- Conditions on vectors MF and FN
hypothesis (h_vector_relation : 
  M.x - F.x = λ * (N.x - F.x) ∧ 
  M.y = -λ * N.y)

-- Defining the line passing through A and M
def line_AM (x : ℝ) : ℝ := M.y / (M.x - A.x) * (x - A.x)

-- Point E is the second intersection of line AM with the parabola
hypothesis (h_E_intersect : E.x ≠ M.x ∧ 
                             on_parabola p E ∧ 
                             E.y = line_AM p E.x)

-- Conclusions to be proved
theorem angles_and_distances :
  (∠MAF = ∠NAF) ∧ 
  (dist A E = dist A N)
:= 
begin
  sorry
end

end angles_and_distances_l507_507321


namespace f_for_minus_one_to_zero_l507_507652

-- Define the function f with the given properties
def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then x * (1 - x)
else if x + 1 ≤ 1 ∧ x + 1 ≥ 0 then -1 / 2 * x * (x + 1)
else 0 -- This placeholder, assuming the function is only defined in the given segments

-- The theorem to prove
theorem f_for_minus_one_to_zero (x : ℝ) (h : -1 ≤ x ∧ x ≤ 0) : 
  f x = -1 / 2 * x * (x + 1) :=
sorry

end f_for_minus_one_to_zero_l507_507652


namespace symmetric_point_origin_l507_507848

def Point := (ℝ × ℝ × ℝ)

def symmetric_point (P : Point) (O : Point) : Point :=
  let (x, y, z) := P
  let (ox, oy, oz) := O
  (2 * ox - x, 2 * oy - y, 2 * oz - z)

theorem symmetric_point_origin :
  symmetric_point (1, 3, 5) (0, 0, 0) = (-1, -3, -5) :=
by sorry

end symmetric_point_origin_l507_507848


namespace regular_octagon_centrally_symmetric_not_centrally_symmetric_for_6_or_12_l507_507968

/-- A regular polygon with n sides where each side is an integer. -/
structure RegularPolygon (n : ℕ) :=
  (side_length : ℕ)
  (is_regular : ∀ i j, i < n → j < n → side_length = side_length) -- Regular implies equal side lengths

/-- A regular octagon with integer side lengths is centrally symmetric. -/
theorem regular_octagon_centrally_symmetric (n : ℕ) (h1 : n = 8) 
  (polygon : RegularPolygon n) : 
  ∃ center, ∀ vertex, dist vertex center = dist (symm_vertex vertex) center :=
sorry

/-- The statement does not hold for polygons with 6 or 12 sides. -/
theorem not_centrally_symmetric_for_6_or_12 (n : ℕ) (h : n = 6 ∨ n = 12) 
  (polygon : RegularPolygon n) :
  ¬ ∃ center, ∀ vertex, dist vertex center = dist (symm_vertex vertex) center := 
sorry

end regular_octagon_centrally_symmetric_not_centrally_symmetric_for_6_or_12_l507_507968


namespace benjamin_collects_6_dozen_eggs_l507_507196

theorem benjamin_collects_6_dozen_eggs (B : ℕ) (h : B + 3 * B + (B - 4) = 26) : B = 6 :=
by sorry

end benjamin_collects_6_dozen_eggs_l507_507196


namespace S2_eq_S3_l507_507395

noncomputable def S_0 : set ℝ³ := sorry -- initial set of points in space
noncomputable def S_n (n : ℕ) : set ℝ³ :=
  match n with
  | 0     => S_0
  | (n+1) => {P | ∃ A B ∈ S_n n, P ∈ segment ℝ³ A B}

theorem S2_eq_S3 : S_n 2 = S_n 3 :=
begin
  sorry
end

end S2_eq_S3_l507_507395


namespace distance_probability_l507_507772

-- Given conditions
def roads : ℕ := 8
def speed : ℕ := 5
def time : ℕ := 1

-- Defining the function to calculate distance given the angle
noncomputable def distance (angle : ℝ) (speed : ℝ) : ℝ :=
  real.sqrt (speed^2 + speed^2 - 2 * speed * speed * real.cos angle)

-- Total possible combinations
def total_combinations : ℕ := roads * roads

-- Favorable outcomes where distance > 8 km
def favorable_outcomes : ℕ := 8 * 3  -- Each road has 3 favorable options (two apart)

-- Probability
noncomputable def probability : ℝ := favorable_outcomes / total_combinations

-- Theorem to prove
theorem distance_probability : probability = 0.375 :=
by
  -- Skipping the proof
  sorry

end distance_probability_l507_507772


namespace cost_of_23_days_l507_507143

-- Define the daily charges
def first_week_charge : ℕ → ℝ := λ days, 18 * days
def additional_week_charge : ℕ → ℝ := λ days, 13 * days

-- Define the total cost based on the problem conditions
noncomputable def total_cost_23_days : ℝ :=
  let first_week_days := 7 in
  let remaining_days := 23 - first_week_days in
  let full_additional_weeks := remaining_days / 7 in
  let full_additional_days := full_additional_weeks * 7 in
  let extra_days := remaining_days % 7 in
  first_week_charge first_week_days +
  additional_week_charge full_additional_days +
  additional_week_charge extra_days

theorem cost_of_23_days : total_cost_23_days = 334.00 :=
  sorry

end cost_of_23_days_l507_507143


namespace perimeter_of_square_with_area_625_cm2_l507_507609

noncomputable def side_length (a : ℝ) : ℝ := 
  real.sqrt a

noncomputable def perimeter (s : ℝ) : ℝ :=
  4 * s

theorem perimeter_of_square_with_area_625_cm2 :
  perimeter (side_length 625) = 100 :=
by
  sorry

end perimeter_of_square_with_area_625_cm2_l507_507609


namespace min_odds_among_seven_l507_507878
open Nat

theorem min_odds_among_seven :
  ∀ (a b c d e f g : ℤ),
  (a + b + c = 30) →
  (a + b + c + d + e = 50) →
  (a + b + c + d + e + f + g = 75) →
  (∃ m : ℕ, m = 1 ∧ ∃ n : ℕ, n = 7 ∧ ∃ odd_cnt : ℕ, odd_cnt = Finset.filter (λ x, x % 2 ≠ 0) (Finset.range n).card ∧ odd_cnt = m) :=
by
  intros a b c d e f g h₁ h₂ h₃
  sorry

end min_odds_among_seven_l507_507878


namespace monotonic_intervals_and_extreme_values_minimum_value_on_interval_1_minimum_value_on_interval_2_minimum_value_on_interval_3_l507_507293

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 4*x + (2 - a) * log x
def a1 : ℝ := 8
def a2_1 : ℝ := 2 * (Real.exp 1)^2 - 1)^2
def a2_2 : ℝ := 2 * (Real.exp 1 - 1)^2
def e : ℝ := Real.exp 1

theorem monotonic_intervals_and_extreme_values :
  monotonic (λ x, f x a1) ∧ (∃ x, is_minimizer f x) :=
by {
    sorry
}

theorem minimum_value_on_interval_1 (a : ℝ) (h : a ≥ a2_1) : 
  ∀ x ∈ set.Icc e (e^2), f x a ≥ f (e^2) a :=
by {
    sorry
}

theorem minimum_value_on_interval_2 (a : ℝ) (h1 : a2_2 < a) (h2 : a < a2_1) :
  ∀ x ∈ set.Icc e (e^2), f x a ≥ f (1 + sqrt (2 * a) / 2) a :=
by {
    sorry
}

theorem minimum_value_on_interval_3 (a : ℝ) (h : a ≤ 0 ∨ a ≤ a2_2) :
  ∀ x ∈ set.Icc e (e^2), f x a ≥ f e a :=
by {
    sorry
}

end monotonic_intervals_and_extreme_values_minimum_value_on_interval_1_minimum_value_on_interval_2_minimum_value_on_interval_3_l507_507293


namespace measure_of_B_l507_507330

theorem measure_of_B (a b : ℝ) (A B : ℝ) (angleA_nonneg : 0 < A ∧ A < 180) (angleB_nonneg : 0 < B ∧ B < 180)
    (a_eq : a = 1) (b_eq : b = Real.sqrt 3) (A_eq : A = 30) :
    B = 60 :=
by
  sorry

end measure_of_B_l507_507330


namespace monotonic_intervals_g_minimum_value_MN_l507_507296

noncomputable def f (x : ℝ) : ℝ := 1 / x
noncomputable def g (x : ℝ) : ℝ := Real.logb 2 (2 - Real.abs (x + 1))

theorem monotonic_intervals_g :
  (∀ x ∈ Ioo (-3:ℝ) (-1:ℝ), monotone_increasing_on g (Ioo (-3) (-1))) ∧ 
  (∀ x ∈ Ioo (-1:ℝ) (1:ℝ), monotone_decreasing_on g (Ioo (-1) (1))) := sorry

theorem minimum_value_MN :
  let M : ℝ × ℝ := (-1, 1)
  ∃ x₀ > 0, let N : ℝ × ℝ := (x₀, f x₀) in
    (∀ x₀, (|N.1 + 1, N.2 - 1|) ≥ 3) ∧ (|N.1 + 1, N.2 - 1| = 3 ↔ x₀ = ((Real.sqrt 5 - 1) / 2)) := sorry

end monotonic_intervals_g_minimum_value_MN_l507_507296


namespace distinct_ratios_exist_l507_507408

theorem distinct_ratios_exist (n : ℕ) (a : Fin n → ℝ) (h_distinct : Function.Injective a) :
  ∃ (A : Fin n → ℝ), ∀ (x : ℝ), (1 / (List.prod (List.map (λ i => x + a i) (List.finRange n)))) = 
  ∑ i, A i / (x + a i) :=
begin
  sorry
end

end distinct_ratios_exist_l507_507408


namespace sqrt_range_l507_507755

theorem sqrt_range (x : ℝ) (h : 5 - x ≥ 0) : x ≤ 5 :=
sorry

end sqrt_range_l507_507755


namespace pentagon_area_l507_507064

/-- Given a convex pentagon ABCDE with specific angle and side length conditions, 
prove that the area of the pentagon is approximately equal to (9 * sqrt(3) / 4 + 6.8832). -/
theorem pentagon_area (A B C D E : Point) 
    (h_convex : ConvexPentagon A B C D E)
    (h_angleA : ∠ A = 110)
    (h_angleB : ∠ B = 110)
    (h_sideEA : dist E A = 3)
    (h_sideAB : dist A B = 3)
    (h_sideBC : dist B C = 3)
    (h_sideCD : dist C D = 5)
    (h_sideDE : dist D E = 5) :
    approximate_area A B C D E = 9 * Real.sqrt(3) / 4 + 6.8832 :=
by
  sorry

end pentagon_area_l507_507064


namespace area_of_regular_hexagon_l507_507381

theorem area_of_regular_hexagon (JKL : Type*) [regular_hexagon JKL] (M N O : JKL) 
  (hM : is_midpoint M (side JKL JK))
  (hN : is_midpoint N (side JKL KL))
  (hO : is_midpoint O (side JKL LM))
  (h_area_mno : area (triangle M N O) = 100) :
  area (hexagon JKL) = 2400 / 9 := 
sorry

end area_of_regular_hexagon_l507_507381


namespace round_3967149_8587234_to_nearest_tenth_l507_507430

def round_to_nearest_tenth (x : ℝ) : ℝ :=
  (Real.floor (10 * x) / 10) + 
  if x - (Real.floor (10 * x) / 10) < 0.05 then 0 else 0.1

theorem round_3967149_8587234_to_nearest_tenth :
  round_to_nearest_tenth 3967149.8587234 = 3967149.9 :=
by
  sorry

end round_3967149_8587234_to_nearest_tenth_l507_507430


namespace smallest_integer_with_eight_factors_l507_507526

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, n = 24 ∧
  (∀ d : ℕ, d ∣ n → d > 0) ∧
  ((∃ p : ℕ, prime p ∧ n = p^7) ∨
   (∃ p q : ℕ, prime p ∧ prime q ∧ n = p^3 * q) ∨
   (∃ p q r : ℕ, prime p ∧ prime q ∧ prime r ∧ n = p * q * r)) ∧
  (∀ m : ℕ, (∀ d : ℕ, d ∣ m → d > 0) → 
           ((∃ p : ℕ, prime p ∧ m = p^7 ∨ m = p^3 * q ∨ m = p * q * r) → 
            m ≥ n)) := by
  sorry

end smallest_integer_with_eight_factors_l507_507526


namespace smith_family_seating_problem_l507_507843

theorem smith_family_seating_problem :
  let total_children := 8
  let boys := 4
  let girls := 4
  (total_children.factorial - (boys.factorial * girls.factorial)) = 39744 :=
by
  sorry

end smith_family_seating_problem_l507_507843


namespace terry_daily_income_l507_507451

theorem terry_daily_income (T : ℕ) (h1 : ∀ j : ℕ, j = 30) (h2 : 7 * 30 = 210) (h3 : 7 * T - 210 = 42) : T = 36 := 
by
  sorry

end terry_daily_income_l507_507451


namespace must_be_divisor_of_a_l507_507387

theorem must_be_divisor_of_a
    (a b c d : ℕ)
    (h1 : Nat.gcd a b = 40)
    (h2 : Nat.gcd b c = 45)
    (h3 : Nat.gcd c d = 75)
    (h4 : 120 < Nat.gcd d a ∧ Nat.gcd d a < 150) :
    5 ∣ a :=
sorry

end must_be_divisor_of_a_l507_507387


namespace problem_part_1_problem_part_2_l507_507261

-- Define the ellipse E
def ellipse (x y : ℝ) : Prop :=
  (x^2) / 9 + (y^2) / 4 = 1

-- Define points A and P
def A : ℝ × ℝ := (0, 2)
def P : ℝ × ℝ := (-3, 2)

-- Define the line l passing through point P
def line_l (k x : ℝ) : ℝ :=
  k * (x + 3) + 2

-- Given conditions
theorem problem_part_1 (k : ℝ) (B C : ℝ × ℝ) (M N : ℝ × ℝ)
  (h1 : ellipse B.1 B.2)
  (h2 : ellipse C.1 C.2)
  (h3 : B.2 = line_l k B.1)
  (h4 : C.2 = line_l k C.1)
  (h5 : B ≠ C)
  (h6 : M.2 = 0)
  (h7 : N.2 = 0)
  (h8 : M.1 = - (2 * B.1) / (B.2 - 2))
  (h9 : N.1 = - (2 * C.1) / (C.2 - 2))
  (h10 : |M.1 - N.1| = 6) :
  line_l (-4/3) = λ x, -4/3 * x - 2 := sorry

-- Define the circumcenter of triangle PMN
noncomputable def circumcenter (P M N : ℝ × ℝ) : ℝ × ℝ :=
  let xM := (P.1 + M.1) / 2
  let yM := (P.2 + M.2) / 2
  let xN := (P.1 + N.1) / 2
  let yN := (P.2 + N.2) / 2
  let x := (xM + xN) / 2
  let y := (yM + yN) / 2
  (x, y)

-- Theorem for the locus of the circumcenter
theorem problem_part_2 (M N : ℝ × ℝ)
  (h1 : circumcenter P M N = (x, y))
  (h2 : y < 1) :
  x = -3 := sorry

end problem_part_1_problem_part_2_l507_507261


namespace no_positive_integers_k_m_exist_l507_507796

def binary_ones_count (n : ℕ) : ℕ :=
  (nat.bits n).count tt

def f (n : ℕ) : ℕ :=
  if binary_ones_count n % 2 = 1 then 1 else 0

theorem no_positive_integers_k_m_exist :
  ∀ k m : ℕ, (k > 0) → (m > 0) → (∀ j < m, f(k + j) = f(k + m + j) ∧ f(k + m + j) = f(k + 2 * m + j)) → false :=
by
  intros k m hk hm h
  sorry

end no_positive_integers_k_m_exist_l507_507796


namespace find_m_l507_507442

theorem find_m (a b j : ℤ) 
    (h1 : 0 ≤ a ∧ a ≤ 9)
    (h2 : 0 ≤ b ∧ b ≤ 9)
    (h3 : 10 * a + b = j * (a^2 + b^2))
    (h4 : 10 * b + a = m * (a^2 + b^2)) :
    m = j := 
begin
    sorry
end

end find_m_l507_507442


namespace grover_total_profit_is_15_l507_507732

theorem grover_total_profit_is_15 
  (boxes : ℕ) 
  (masks_per_box : ℕ) 
  (price_per_mask : ℝ) 
  (cost_of_boxes : ℝ) 
  (total_profit : ℝ)
  (hb : boxes = 3)
  (hm : masks_per_box = 20)
  (hp : price_per_mask = 0.5)
  (hc : cost_of_boxes = 15)
  (htotal : total_profit = (boxes * masks_per_box) * price_per_mask - cost_of_boxes) :
  total_profit = 15 :=
sorry

end grover_total_profit_is_15_l507_507732


namespace problem_statement_l507_507733

noncomputable def distance_home_to_gym := 3
noncomputable def ratio := 2 / 3

noncomputable def a₀ := 0
noncomputable def b₀ := (2 : ℝ)

noncomputable def a (n : ℕ) : ℝ :=
  if n = 0 then a₀ else b (n - 1) - ratio * (distance_home_to_gym - b (n - 1))

noncomputable def b (n : ℕ) : ℝ :=
  if n = 0 then b₀ else a (n - 1) + ratio * (distance_home_to_gym - a (n - 1))

noncomputable def A := (20 / 7 : ℝ)
noncomputable def B := (26 / 7 : ℝ)

theorem problem_statement : |A - B| = (6 / 7 : ℝ) :=
by
  sorry

end problem_statement_l507_507733


namespace find_ab_value_l507_507263

variable (a b : ℝ)

def condition1 : Prop := exp (2 - a) = a
def condition2 : Prop := b * (log b - 1) = exp (3)

theorem find_ab_value (h1 : condition1 a) (h2 : condition2 b) : a * b = exp (3) :=
  sorry

end find_ab_value_l507_507263


namespace coefficient_of_x_in_binomial_expansion_coefficient_of_x_in_x_sub_2_exp_5_l507_507459

theorem coefficient_of_x_in_binomial_expansion :
  (binom 5 4 * (-2)^4) = 80 :=
by lint exactly -- This part includes lint checking for matching precisely.

theorem coefficient_of_x_in_x_sub_2_exp_5 :
  coefficient_of_x_in_binomial_expansion := -- Using the previous theorem directly.
begin
  sorry -- Here we skip the actual detailed proof because it's mentioned we don't need to consider solution steps.
end

end coefficient_of_x_in_binomial_expansion_coefficient_of_x_in_x_sub_2_exp_5_l507_507459


namespace linda_original_amount_l507_507748

theorem linda_original_amount (L L2 : ℕ) 
  (h1 : L = 20) 
  (h2 : L - 5 = L2) : 
  L2 + 5 = 15 := 
sorry

end linda_original_amount_l507_507748


namespace ammonium_iodide_requirement_l507_507669

theorem ammonium_iodide_requirement :
  ∀ (NH4I KOH NH3 KI H2O : ℕ),
  (NH4I + KOH = NH3 + KI + H2O) → 
  (NH4I = 3) →
  (KOH = 3) →
  (NH3 = 3) →
  (KI = 3) →
  (H2O = 3) →
  NH4I = 3 :=
by
  intros NH4I KOH NH3 KI H2O reaction_balanced NH4I_req KOH_req NH3_prod KI_prod H2O_prod
  exact NH4I_req

end ammonium_iodide_requirement_l507_507669


namespace brick_height_correct_l507_507496

-- Definitions
def wall_length : ℝ := 8
def wall_height : ℝ := 6
def wall_thickness : ℝ := 0.02 -- converted from 2 cm to meters
def brick_length : ℝ := 0.05 -- converted from 5 cm to meters
def brick_width : ℝ := 0.11 -- converted from 11 cm to meters
def brick_height : ℝ := 0.06 -- converted from 6 cm to meters
def number_of_bricks : ℝ := 2909.090909090909

-- Statement to prove
theorem brick_height_correct : brick_height = 0.06 := by
  sorry

end brick_height_correct_l507_507496


namespace solve_cubic_eq_l507_507243

theorem solve_cubic_eq (z : ℂ) : (z^3 + 27 = 0) ↔ (z = -3 ∨ z = (3 / 2) + (3 * complex.I * real.sqrt 3 / 2) ∨ z = (3 / 2) - (3 * complex.I * real.sqrt 3 / 2)) :=
by
  sorry

end solve_cubic_eq_l507_507243


namespace max_area_large_rectangle_l507_507631

theorem max_area_large_rectangle
  (perimeter_A : ℕ := 26)
  (perimeter_B : ℕ := 28)
  (perimeter_C : ℕ := 30)
  (perimeter_D : ℕ := 32)
  (perimeter_E : ℕ := 34) :
  ∃ (max_area : ℕ), max_area = 512 :=
begin
  sorry
end

end max_area_large_rectangle_l507_507631


namespace exchange_ways_10_dollar_l507_507327

theorem exchange_ways_10_dollar (p q : ℕ) (H : 2 * p + 5 * q = 200) : 
  ∃ (n : ℕ), n = 20 :=
by {
  sorry
}

end exchange_ways_10_dollar_l507_507327


namespace unique_parallel_plane_l507_507423

theorem unique_parallel_plane (A : Point) (P : Plane) :
  (∀ M N : Plane, (is_parallel M P ∧ is_parallel N P ∧ M ≠ N → M ∩ N ≠ ∅) → False) := by
sorry

end unique_parallel_plane_l507_507423


namespace F_double_prime_coordinates_correct_l507_507498

structure Point where
  x : Int
  y : Int

def reflect_over_y_axis (p : Point) : Point :=
  { x := -p.x, y := p.y }

def reflect_over_x_axis (p : Point) : Point :=
  { x := p.x, y := -p.y }

def F : Point := { x := 6, y := -4 }

def F' : Point := reflect_over_y_axis F

def F'' : Point := reflect_over_x_axis F'

theorem F_double_prime_coordinates_correct : F'' = { x := -6, y := 4 } :=
  sorry

end F_double_prime_coordinates_correct_l507_507498


namespace y_intercepts_parabola_l507_507737

theorem y_intercepts_parabola : 
  let f : ℝ → ℝ := λ y, 3 * y^2 - 6 * y + 3
  in (∀ y : ℝ, f y = 0 → y = 1) → f 1 = 0 := 
begin
sorry
end

end y_intercepts_parabola_l507_507737


namespace number_of_elements_in_complement_l507_507806

open Set

variable (A B U : Set ℕ)

def complement_size_proof : Prop :=
  (A = {1, 2, 3, 4}) ∧ 
  (B = {0, 1, 2, 4, 5}) ∧ 
  (U = A ∪ B) ∧ 
  #(U \ (A ∩ B)) = 3

theorem number_of_elements_in_complement : complement_size_proof := 
  by
  sorry

end number_of_elements_in_complement_l507_507806


namespace set_intersection_l507_507727

open Set

def M : Set ℕ := {0, 1, 2, 3, 4}
def N : Set ℕ := {1, 3, 5}
def intersection : Set ℕ := {1, 3}

theorem set_intersection : M ∩ N = intersection := by
  sorry

end set_intersection_l507_507727


namespace parallelogram_height_l507_507674

theorem parallelogram_height
  (A b : ℝ)
  (h : ℝ)
  (h_area : A = 120)
  (h_base : b = 12)
  (h_formula : A = b * h) : h = 10 :=
by 
  sorry

end parallelogram_height_l507_507674


namespace max_min_cubic_sums_l507_507795

theorem max_min_cubic_sums
  (n : ℕ)
  (x : Fin n → ℤ)
  (hx1 : ∀ i, -1 ≤ x i ∧ x i ≤ 2)
  (hx2 : ∑ i, x i = 19)
  (hx3 : ∑ i, (x i)^2 = 99) :
  19 ≤ ∑ i, (x i)^3 ∧ ∑ i, (x i)^3 ≤ 133 :=
sorry

end max_min_cubic_sums_l507_507795


namespace smallest_integer_with_eight_factors_l507_507545

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, 
  ∀ m : ℕ, (∀ p : ℕ, ∃ k : ℕ, m = p^k → (k + 1) * (p + 1) = 8) → (n ≤ m) ∧ 
  (∀ d : ℕ, d ∣ n → d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 24) :=
sorry

end smallest_integer_with_eight_factors_l507_507545


namespace perpendicular_OI_AK_l507_507782

noncomputable def Triangle (A B C : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] := sorry

noncomputable def circumcenter (A B C : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] := sorry

noncomputable def incenter (A B C : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] := sorry

noncomputable def angle_bisector (A B C : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] := sorry

theorem perpendicular_OI_AK
  {A B C K O I : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace K] [MetricSpace O] [MetricSpace I]
  (hAB : dist A B = 4)
  (hAC : dist A C = 6)
  (hBC : dist B C = 5)
  (circumcenter : circumcenter A B C = O)
  (incenter : incenter A B C = I)
  (angle_bisector : ∃ D, angle_bisector A B C = D ∧ Circle.circumcircle_intersects_at D K)
  (perp : OI ⟂ AK) :
  OI ⟂ AK := sorry

end perpendicular_OI_AK_l507_507782


namespace largest_prime_factor_of_expris_17_l507_507961

/-- Define the expression we are working with. -/
def expr : ℤ := 17^4 + 2 * 17^3 + 17^2 - 16^4

/-- Prove that the largest prime factor of the expression is 17. -/
theorem largest_prime_factor_of_expris_17 : 
  ∃ p : ℤ, p.prime ∧ p ∣ expr ∧ ∀ q : ℤ, q.prime ∧ q ∣ expr → q ≤ p :=
sorry

end largest_prime_factor_of_expris_17_l507_507961


namespace largest_number_l507_507959

noncomputable def a : ℝ := 8.12331
noncomputable def b : ℝ := 8.123 + 3 / 10000 * ∑' n, 1 / (10 : ℝ)^n
noncomputable def c : ℝ := 8.12 + 331 / 100000 * ∑' n, 1 / (1000 : ℝ)^n
noncomputable def d : ℝ := 8.1 + 2331 / 1000000 * ∑' n, 1 / (10000 : ℝ)^n
noncomputable def e : ℝ := 8 + 12331 / 100000 * ∑' n, 1 / (10000 : ℝ)^n

theorem largest_number : (b > a) ∧ (b > c) ∧ (b > d) ∧ (b > e) := by
  sorry

end largest_number_l507_507959


namespace tree_last_tree_height_difference_l507_507363

noncomputable def treeHeightDifference : ℝ :=
  let t1 := 1000
  let t2 := 500
  let t3 := 500
  let avgHeight := 800
  let lastTreeHeight := 4 * avgHeight - (t1 + t2 + t3)
  lastTreeHeight - t1

theorem tree_last_tree_height_difference :
  treeHeightDifference = 200 := sorry

end tree_last_tree_height_difference_l507_507363


namespace arithmetic_mean_of_38_and_59_is_67_over_144_l507_507936

theorem arithmetic_mean_of_38_and_59_is_67_over_144 :
  (3 / 8 + 5 / 9) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_38_and_59_is_67_over_144_l507_507936


namespace arrangement_ways_l507_507086

theorem arrangement_ways (people : Finset (Fin 6)) (A B C : Fin 6) :
    (∀ x ∈ people, x ≠ A ∧ x ≠ B ∧ x ≠ C) →
    ((∀ (p₁ p₂ : List (Fin 6)), p₁ ≠ p₂ ∧ (A ∈ p₁) ∧ (B ∈ p₁) ∧ (C ∈ p₁) → 
    List.neighbors p₁ p₂ → False) →
    people.card = 6 →
    ∃ l : List (Fin 6), l.permutations.card = 144 :=
by
  sorry

end arrangement_ways_l507_507086


namespace initial_amount_in_cookie_jar_l507_507493

theorem initial_amount_in_cookie_jar (doris_spent : ℕ) (martha_spent : ℕ) (amount_left : ℕ) (spent_eq_martha : martha_spent = doris_spent / 2) (amount_left_eq : amount_left = 12) (doris_spent_eq : doris_spent = 6) : (doris_spent + martha_spent + amount_left = 21) :=
by
  sorry

end initial_amount_in_cookie_jar_l507_507493


namespace range_of_sum_l507_507251

theorem range_of_sum {x y z : ℝ} (h1 : x + 2 * y + 3 * z = 1) (h2 : y * z + z * x + x * y = -1) :
  let t := x + y + z in
  (3 - 3 * Real.sqrt 3) / 4 ≤ t ∧ t ≤ (3 + 3 * Real.sqrt 3) / 4 :=
by {
  sorry
}

end range_of_sum_l507_507251


namespace total_batteries_produced_l507_507590

def time_to_gather_materials : ℕ := 6 -- in minutes
def time_to_create_battery : ℕ := 9   -- in minutes
def num_robots : ℕ := 10
def total_time : ℕ := 5 * 60 -- in minutes (5 hours * 60 minutes/hour)

theorem total_batteries_produced :
  total_time / (time_to_gather_materials + time_to_create_battery) * num_robots = 200 :=
by
  -- Placeholder for the proof steps
  sorry

end total_batteries_produced_l507_507590


namespace find_subtracted_number_l507_507629

theorem find_subtracted_number 
  (a : ℕ) (b : ℕ) (g : ℕ) (n : ℕ) 
  (h1 : a = 2) 
  (h2 : b = 3 * a) 
  (h3 : g = 2 * b - n) 
  (h4 : g = 8) : n = 4 :=
by 
  sorry

end find_subtracted_number_l507_507629


namespace unit_digit_of_expression_l507_507749

theorem unit_digit_of_expression (Q : ℤ) : 
  let R := (8 ^ Q + 7 ^ (10 * Q) + 6 ^ (100 * Q) + 5 ^ (1000 * Q)) % 10 
  in R = 8 := 
by 
  sorry

end unit_digit_of_expression_l507_507749


namespace hyperbola_definition_l507_507820

section
variables {x y x1 y1 x2 y2 a : ℝ}
def F1 := (x1, y1)
def F2 := (x2, y2)
def d1 (x y : ℝ) := real.sqrt ((x - x1)^2 + (y - y1)^2)
def d2 (x y : ℝ) := real.sqrt ((x - x2)^2 + (y - y2)^2)
def is_hyperbola (x y : ℝ) : Prop := abs (d1 x y - d2 x y) = 2 * a

theorem hyperbola_definition : 0 < a → ∀ (x y : ℝ), is_hyperbola x y → is_hyperbola x y :=
by
  intros h_a x y h
  sorry
end

end hyperbola_definition_l507_507820


namespace sin_cos_product_l507_507691

variable (α : ℝ)

theorem sin_cos_product (h : cos α - sin α = 1 / 2) : sin α * cos α = 3 / 8 := by
  sorry

end sin_cos_product_l507_507691


namespace perimeter_of_square_with_area_625_cm2_l507_507608

noncomputable def side_length (a : ℝ) : ℝ := 
  real.sqrt a

noncomputable def perimeter (s : ℝ) : ℝ :=
  4 * s

theorem perimeter_of_square_with_area_625_cm2 :
  perimeter (side_length 625) = 100 :=
by
  sorry

end perimeter_of_square_with_area_625_cm2_l507_507608


namespace find_five_digit_palindromes_l507_507891

def is_palindrome (n : ℕ) : Prop :=
  let digits := nat.digits 10 n
  digits = digits.reverse

def contains_42 (n : ℕ) : Prop :=
  let digits := nat.digits 10 n
  list.is_infix [4, 2] digits

def divisible_by_12 (n : ℕ) : Prop :=
  n % 12 = 0

def valid_numbers := [42024, 42324, 42624, 42924, 44244, 82428]

theorem find_five_digit_palindromes :
  ∀ n : ℕ, n ∈ valid_numbers ↔ (10000 ≤ n ∧ n < 100000 ∧ is_palindrome n ∧ contains_42 n ∧ divisible_by_12 n) :=
by
  sorry

end find_five_digit_palindromes_l507_507891


namespace arithmetic_mean_of_fractions_l507_507892

theorem arithmetic_mean_of_fractions :
  (3/8 + 5/9) / 2 = 67 / 144 :=
by
  sorry

end arithmetic_mean_of_fractions_l507_507892


namespace smallest_integer_with_eight_factors_l507_507549

theorem smallest_integer_with_eight_factors : ∃ n : ℕ, 
  ∀ m : ℕ, (∀ p : ℕ, ∃ k : ℕ, m = p^k → (k + 1) * (p + 1) = 8) → (n ≤ m) ∧ 
  (∀ d : ℕ, d ∣ n → d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 24) :=
sorry

end smallest_integer_with_eight_factors_l507_507549


namespace fraction_with_variable_denominator_l507_507958

theorem fraction_with_variable_denominator (m n a : ℝ) (ha : a ≠ 0) :
  (∃ x ∈ { (m+n)/3, -sqrt 2/2, (2*a)/π, 1/a }, x = 1/a) :=
by
  sorry

end fraction_with_variable_denominator_l507_507958


namespace log_two_three_irrational_log_sqrt2_three_irrational_log_five_plus_three_sqrt2_irrational_l507_507041

-- Define irrational numbers in Lean
def irrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ p / q

-- Prove that log base 2 of 3 is irrational
theorem log_two_three_irrational : irrational (Real.log 3 / Real.log 2) := 
sorry

-- Prove that log base sqrt(2) of 3 is irrational
theorem log_sqrt2_three_irrational : 
  irrational (Real.log 3 / (1/2 * Real.log 2)) := 
sorry

-- Prove that log base (5 + 3sqrt(2)) of (3 + 5sqrt(2)) is irrational
theorem log_five_plus_three_sqrt2_irrational :
  irrational (Real.log (3 + 5 * Real.sqrt 2) / Real.log (5 + 3 * Real.sqrt 2)) := 
sorry

end log_two_three_irrational_log_sqrt2_three_irrational_log_five_plus_three_sqrt2_irrational_l507_507041


namespace benjamin_walks_95_miles_in_a_week_l507_507201

def distance_to_work_one_way : ℕ := 6
def days_to_work : ℕ := 5
def distance_to_dog_walk_one_way : ℕ := 2
def dog_walks_per_day : ℕ := 2
def days_for_dog_walk : ℕ := 7
def distance_to_best_friend_one_way : ℕ := 1
def times_to_best_friend : ℕ := 1
def distance_to_convenience_store_one_way : ℕ := 3
def times_to_convenience_store : ℕ := 2

def total_distance_in_a_week : ℕ :=
  (days_to_work * 2 * distance_to_work_one_way) +
  (days_for_dog_walk * dog_walks_per_day * distance_to_dog_walk_one_way) +
  (times_to_convenience_store * 2 * distance_to_convenience_store_one_way) +
  (times_to_best_friend * 2 * distance_to_best_friend_one_way)

theorem benjamin_walks_95_miles_in_a_week :
  total_distance_in_a_week = 95 :=
by {
  simp [distance_to_work_one_way, days_to_work,
        distance_to_dog_walk_one_way, dog_walks_per_day,
        days_for_dog_walk, distance_to_best_friend_one_way,
        times_to_best_friend, distance_to_convenience_store_one_way,
        times_to_convenience_store, total_distance_in_a_week],
  sorry
}

end benjamin_walks_95_miles_in_a_week_l507_507201


namespace ray_initial_cents_l507_507824

theorem ray_initial_cents (cents_to_peter : ℕ) (cents_to_randi : ℕ) (nickels_left : ℕ) :
  cents_to_peter = 25 ∧ cents_to_randi = 2 * cents_to_peter ∧ nickels_left = 4 →
  let total_given := cents_to_peter + cents_to_randi in
  let total_left := nickels_left * 5 in
  total_given + total_left = 95 := sorry

end ray_initial_cents_l507_507824


namespace prob_B2_geometric_sequence_maximize_P_X_l507_507139

noncomputable theory

-- Definitions for given conditions
def P_B1 := 1 / 3
def P_notB1 := 2 / 3
def P_B2_given_B1 := 1 / 2
def P_B2_given_notB1 := 3 / 4
def P_B_next_given_A := 3 / 4
def P_B_next_given_B := 1 / 2

-- Inference of the probability of choosing meal B on the second day
theorem prob_B2 : ∀ A B : Type, (P_B1 * P_B2_given_B1 + P_notB1 * P_B2_given_notB1 = 2 / 3) :=
by sorry

-- Inference of the geometric sequence property
theorem geometric_sequence : ∀ {P_n : ℕ → ℝ},
  (∀ (n : ℕ), P_{n+1} =  (1/2) * P_n + (3/4) * (1 - P_n)) →
  (∀ (n : ℕ), (P_n - 3/5) = (λ x, (-1/4)^n * x) (P_1 - 3/5)) :=
by sorry

-- Proof that k = 33 maximizes the binomial probability function
theorem maximize_P_X : ∃ k : ℕ, k = 33 ∧ ∀ j : ℕ, P(X = k) > P(X = j) :=
by sorry

end prob_B2_geometric_sequence_maximize_P_X_l507_507139


namespace part1_part2_l507_507178

-- Part 1: Showing x range for increasing actual processing fee
theorem part1 (x : ℝ) : (x ≤ 99.5) ↔ (∀ y, 0 < y → y ≤ x → (1/2) * Real.log (2 * y + 1) - y / 200 ≤ (1/2) * Real.log (2 * (y + 0.1) + 1) - (y + 0.1) / 200) :=
sorry

-- Part 2: Showing m range for no losses in processing production
theorem part2 (m x : ℝ) (hx : x ∈ Set.Icc 10 20) : 
  (m ≤ (Real.log 41 - 2) / 40) ↔ ((1/2) * Real.log (2 * x + 1) - m * x ≥ (1/20) * x) :=
sorry

end part1_part2_l507_507178


namespace num_adults_in_group_l507_507635

theorem num_adults_in_group (A C : ℕ) 
  (h1 : A + C = 7) 
  (h2 : 9.50 * A + 6.50 * C = 54.50) : 
  A = 3 :=
sorry

end num_adults_in_group_l507_507635


namespace length_of_BD_l507_507883

-- Given conditions
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (AB BC: ℝ) (right_angle_at_B : ∠ B = π / 2) (AB_eq_3 : AB = 3) (BC_eq_4 : BC = 4)
variable (D_bisects_∠BAC : is_angle_bisector A B C D)

-- The triangle sides satisfy the Pythagorean theorem
axiom AC_eq_5 : EuclideanGeometry.pythagorean_theorem AB BC 5

-- The Angle Bisector Theorem relates the segments BD and DC
axiom angle_bisector_theorem : ∀ {x y z w : ℝ}, is_angle_bisector y z w -> (x / y = w / z) -> (∀ (a : ℝ), a = x -> (a / (AB - a) = 3 / 4))

-- We need to prove the length of BD
theorem length_of_BD : BD = 12 / 7 :=
by { sorry }

end length_of_BD_l507_507883


namespace pet_store_cats_sale_l507_507176

theorem pet_store_cats_sale (initial_siamese : ℕ) (initial_house : ℕ) (cats_left : ℕ) :
  initial_siamese = 38 → initial_house = 25 → cats_left = 18 →
  initial_siamese + initial_house - cats_left = 45 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end pet_store_cats_sale_l507_507176


namespace convex_polyhedron_faces_same_number_of_sides_l507_507037

-- Definitions used in the Lean 4 statement
def convex_polyhedron (P : Type) : Prop :=
  sorry -- Add the formal definition of a convex polyhedron

def faces (P : Type) : Type :=
  sorry -- Add the formal definition of faces of the polyhedron

def number_of_sides (f : faces P) : ℕ :=
  sorry -- Add the formal definition of the number of sides of a face

-- The statement of the proof problem
theorem convex_polyhedron_faces_same_number_of_sides (P : Type) [convex_polyhedron P] :
  ∃ f₁ f₂ : faces P, f₁ ≠ f₂ ∧ number_of_sides f₁ = number_of_sides f₂ :=
by
  sorry

end convex_polyhedron_faces_same_number_of_sides_l507_507037


namespace S6_equals_63_l507_507702

variable {S : ℕ → ℕ}

-- Define conditions
axiom S_n_geometric_sequence (a : ℕ → ℕ) (n : ℕ) : n ≥ 1 → S n = (a 0) * ((a 1)^(n) -1) / (a 1 - 1)
axiom S_2_eq_3 : S 2 = 3
axiom S_4_eq_15 : S 4 = 15

-- State theorem
theorem S6_equals_63 : S 6 = 63 := by
  sorry

end S6_equals_63_l507_507702


namespace unique_values_symbol_permuted_l507_507758

section
variable {A B C D : ℝ}

def symbol (A B C D : ℝ) := (A * C) / (B * C) / ((A * D) / (B * D))

theorem unique_values_symbol_permuted (A B C D : ℝ) (k_1 k_2 k_3 : ℝ) :
  (symbol A B C D = k_1) →
  (symbol A B D C = 1 / k_1) →
  (symbol A C B D = k_2) →
  (symbol A C D B = 1 / k_2) →
  (symbol A D B C = k_3) →
  (symbol A D C B = 1 / k_3) →
  (symbol B A C D = 1 / k_1) →
  (symbol B A D C = k_1) →
  (symbol B C A D = k_3) →
  (symbol B C D A = 1 / k_3) →
  (symbol B D A C = k_2) →
  (symbol B D C A = 1 / k_2) →
  (symbol C A B D = 1 / k_2) →
  (symbol C A D B = k_2) →
  (symbol C B A D = 1 / k_3) →
  (symbol C B D A = k_3) →
  (symbol C D A B = k_1) →
  (symbol D A B C = 1 / k_3) →
  (symbol D A C B = k_3) →
  (symbol D B A C = 1 / k_2) →
  (symbol D B C A = k_2) →
  (symbol D C A B = 1 / k_1) →
  { x : ℝ | ∃ (A' B' C' D' : ℝ), x = symbol A' B' C' D' } = 
    { k_1, 1 / k_1, 1 - k_1, 1 / (1 - k_1), k_1 / (k_1 - 1), (k_1 - 1) / k_1 } :=
sorry
end

end unique_values_symbol_permuted_l507_507758


namespace find_M_l507_507983

open Real

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 27 = 1
def F1 : ℝ × ℝ := (-6, 0)
def F2 : ℝ × ℝ := (6, 0)
def A : ℝ × ℝ := (9 / 2, sqrt 135 / 2)

theorem find_M :
  ∃ M : ℝ × ℝ, M.2 = 0 ∧
  (let x := M.1 in
   |x + 6| / |x - 6| = 2 ∧ M = (2, 0)) :=
begin
  use (2, 0),
  split,
  { refl },
  split,
  { refl },
  sorry -- Proof steps omitted
end

end find_M_l507_507983


namespace arithmetic_mean_of_fractions_l507_507929

theorem arithmetic_mean_of_fractions (a b : ℚ) (h1 : a = 3 / 8) (h2 : b = 5 / 9) :
  (a + b) / 2 = 67 / 144 := by
  sorry

end arithmetic_mean_of_fractions_l507_507929
