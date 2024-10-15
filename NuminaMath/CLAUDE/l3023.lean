import Mathlib

namespace NUMINAMATH_CALUDE_consecutive_odd_squares_difference_divisible_by_8_l3023_302389

theorem consecutive_odd_squares_difference_divisible_by_8 (n : ℤ) : 
  ∃ k : ℤ, (2*n + 3)^2 - (2*n + 1)^2 = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_squares_difference_divisible_by_8_l3023_302389


namespace NUMINAMATH_CALUDE_polly_tweets_l3023_302399

/-- Represents the tweet rate (tweets per minute) for different states of Polly --/
structure TweetRate where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Represents the duration (in minutes) Polly spends in each state --/
structure Duration where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Calculates the total number of tweets given tweet rates and durations --/
def totalTweets (rate : TweetRate) (duration : Duration) : ℕ :=
  rate.happy * duration.happy + rate.hungry * duration.hungry + rate.mirror * duration.mirror

/-- Theorem stating that given the specific conditions, Polly tweets 1340 times --/
theorem polly_tweets : 
  ∀ (rate : TweetRate) (duration : Duration),
  rate.happy = 18 ∧ rate.hungry = 4 ∧ rate.mirror = 45 ∧
  duration.happy = 20 ∧ duration.hungry = 20 ∧ duration.mirror = 20 →
  totalTweets rate duration = 1340 := by
  sorry


end NUMINAMATH_CALUDE_polly_tweets_l3023_302399


namespace NUMINAMATH_CALUDE_u_v_sum_of_squares_l3023_302340

theorem u_v_sum_of_squares (u v : ℝ) (hu : u > 1) (hv : v > 1)
  (h : (Real.log u / Real.log 3)^4 + (Real.log v / Real.log 7)^4 = 10 * (Real.log u / Real.log 3) * (Real.log v / Real.log 7)) :
  u^2 + v^2 = 3^Real.sqrt 5 + 7^Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_u_v_sum_of_squares_l3023_302340


namespace NUMINAMATH_CALUDE_new_students_average_age_l3023_302311

theorem new_students_average_age
  (original_strength : ℕ)
  (original_average_age : ℝ)
  (new_students : ℕ)
  (new_average_age : ℝ) :
  original_strength = 10 →
  original_average_age = 40 →
  new_students = 10 →
  new_average_age = 36 →
  let total_original_age := original_strength * original_average_age
  let total_new_age := (original_strength + new_students) * new_average_age
  let new_students_total_age := total_new_age - total_original_age
  new_students_total_age / new_students = 32 := by
sorry

end NUMINAMATH_CALUDE_new_students_average_age_l3023_302311


namespace NUMINAMATH_CALUDE_cannot_find_fourth_vertex_l3023_302336

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Symmetric point operation -/
def symmetricPoint (a b : Point) : Point :=
  { x := 2 * b.x - a.x, y := 2 * b.y - a.y }

/-- Represents a square -/
structure Square where
  v1 : Point
  v2 : Point
  v3 : Point

/-- Checks if a point is a valid fourth vertex of a square -/
def isValidFourthVertex (s : Square) (p : Point) : Prop := sorry

theorem cannot_find_fourth_vertex (s : Square) :
  ¬ ∃ (p : Point), (∃ (a b : Point), p = symmetricPoint a b) ∧ isValidFourthVertex s p := by
  sorry

end NUMINAMATH_CALUDE_cannot_find_fourth_vertex_l3023_302336


namespace NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l3023_302397

-- Define the complex polynomial z^5 - z^3 + z
def f (z : ℂ) : ℂ := z^5 - z^3 + z

-- Define the property of being an nth root of unity
def is_nth_root_of_unity (z : ℂ) (n : ℕ) : Prop := z^n = 1

-- State the theorem
theorem smallest_n_for_roots_of_unity : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (z : ℂ), f z = 0 → is_nth_root_of_unity z n) ∧
  (∀ (m : ℕ), m > 0 → m < n → 
    ∃ (w : ℂ), f w = 0 ∧ ¬(is_nth_root_of_unity w m)) ∧
  n = 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l3023_302397


namespace NUMINAMATH_CALUDE_second_number_in_sequence_l3023_302382

/-- The second number in the sequence of numbers that, when divided by 7, 9, and 11,
    always leaves a remainder of 5, given that 1398 - 22 = 1376 is the first such number. -/
theorem second_number_in_sequence (first_number : ℕ) (h1 : first_number = 1376) :
  ∃ (second_number : ℕ),
    second_number > first_number ∧
    second_number % 7 = 5 ∧
    second_number % 9 = 5 ∧
    second_number % 11 = 5 ∧
    ∀ (n : ℕ), first_number < n ∧ n < second_number →
      (n % 7 ≠ 5 ∨ n % 9 ≠ 5 ∨ n % 11 ≠ 5) :=
by sorry

end NUMINAMATH_CALUDE_second_number_in_sequence_l3023_302382


namespace NUMINAMATH_CALUDE_min_operations_to_300_l3023_302318

def Calculator (n : ℕ) : Set ℕ :=
  { m | ∃ (ops : List (ℕ → ℕ)), 
    (∀ op ∈ ops, op = (· + 1) ∨ op = (· * 2)) ∧
    ops.foldl (λ acc f => f acc) 1 = m ∧
    ops.length = n }

theorem min_operations_to_300 :
  (∀ n < 11, 300 ∉ Calculator n) ∧ 300 ∈ Calculator 11 :=
sorry

end NUMINAMATH_CALUDE_min_operations_to_300_l3023_302318


namespace NUMINAMATH_CALUDE_f_positive_iff_f_plus_abs_lower_bound_f_plus_abs_equality_exists_l3023_302348

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

-- Theorem for part (1)
theorem f_positive_iff (x : ℝ) : f x > 0 ↔ x > 1 ∨ x < -5 := by sorry

-- Theorem for part (2)
theorem f_plus_abs_lower_bound (x : ℝ) : f x + 3*|x - 4| ≥ 9 := by sorry

-- Theorem for the existence of equality in part (2)
theorem f_plus_abs_equality_exists : ∃ x : ℝ, f x + 3*|x - 4| = 9 := by sorry

end NUMINAMATH_CALUDE_f_positive_iff_f_plus_abs_lower_bound_f_plus_abs_equality_exists_l3023_302348


namespace NUMINAMATH_CALUDE_jordans_weight_loss_l3023_302335

/-- Calculates Jordan's final weight after 13 weeks of an exercise program --/
theorem jordans_weight_loss (initial_weight : ℕ) : 
  initial_weight = 250 →
  (initial_weight 
    - (3 * 4)  -- Weeks 1-4
    - 5        -- Week 5
    - (2 * 4)  -- Weeks 6-9
    + 3        -- Week 10
    - (4 * 3)) -- Weeks 11-13
  = 216 := by
  sorry

#check jordans_weight_loss

end NUMINAMATH_CALUDE_jordans_weight_loss_l3023_302335


namespace NUMINAMATH_CALUDE_id_number_permutations_l3023_302379

/-- The number of permutations of n distinct elements -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The problem statement -/
theorem id_number_permutations :
  permutations 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_id_number_permutations_l3023_302379


namespace NUMINAMATH_CALUDE_kates_retirement_fund_l3023_302310

/-- The initial value of Kate's retirement fund, given the current value and the decrease amount. -/
def initial_value (current_value decrease : ℕ) : ℕ := current_value + decrease

/-- Theorem stating that Kate's initial retirement fund value was $1472. -/
theorem kates_retirement_fund : initial_value 1460 12 = 1472 := by
  sorry

end NUMINAMATH_CALUDE_kates_retirement_fund_l3023_302310


namespace NUMINAMATH_CALUDE_rectangle_in_square_l3023_302324

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents the arrangement of rectangles in the square -/
def arrangement (r : Rectangle) : ℝ := 2 * r.length + 2 * r.width

/-- The theorem stating the properties of the rectangles in the square -/
theorem rectangle_in_square (r : Rectangle) : 
  arrangement r = 18 ∧ 3 * r.length = 18 → r.length = 6 ∧ r.width = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_in_square_l3023_302324


namespace NUMINAMATH_CALUDE_total_time_circling_island_l3023_302364

-- Define the problem parameters
def time_per_round : ℕ := 30
def saturday_rounds : ℕ := 11
def sunday_rounds : ℕ := 15

-- State the theorem
theorem total_time_circling_island : 
  (saturday_rounds + sunday_rounds) * time_per_round = 780 := by
  sorry

end NUMINAMATH_CALUDE_total_time_circling_island_l3023_302364


namespace NUMINAMATH_CALUDE_stratified_sample_female_count_l3023_302344

/-- Represents the number of female students in a stratified sample -/
def female_students_in_sample (total_male : ℕ) (total_female : ℕ) (sample_size : ℕ) : ℕ :=
  (total_female * sample_size) / (total_male + total_female)

/-- Theorem: In a stratified sampling by gender with 500 male students, 400 female students, 
    and a sample size of 45, the number of female students in the sample is 20 -/
theorem stratified_sample_female_count :
  female_students_in_sample 500 400 45 = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_female_count_l3023_302344


namespace NUMINAMATH_CALUDE_solve_for_a_l3023_302351

theorem solve_for_a (x : ℝ) (a : ℝ) (h1 : 2 * x - a - 5 = 0) (h2 : x = 3) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3023_302351


namespace NUMINAMATH_CALUDE_product_minus_quotient_l3023_302395

theorem product_minus_quotient : 11 * 13 * 17 - 33 / 3 = 2420 := by
  sorry

end NUMINAMATH_CALUDE_product_minus_quotient_l3023_302395


namespace NUMINAMATH_CALUDE_factorization_problem_l3023_302309

theorem factorization_problem (a b c : ℤ) : 
  (∀ x, x^2 + 7*x + 12 = (x + a) * (x + b)) →
  (∀ x, x^2 - 8*x - 20 = (x - b) * (x - c)) →
  a - b + c = -9 :=
by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_l3023_302309


namespace NUMINAMATH_CALUDE_factor_implies_c_equals_three_l3023_302321

theorem factor_implies_c_equals_three (c : ℝ) : 
  (∀ x : ℝ, (x + 7) ∣ (c * x^3 + 19 * x^2 - 4 * c * x + 20)) → c = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_c_equals_three_l3023_302321


namespace NUMINAMATH_CALUDE_marble_ratio_l3023_302378

/-- Proves that the ratio of marbles Lori gave to marbles Hilton lost is 2:1 --/
theorem marble_ratio (initial : ℕ) (found : ℕ) (lost : ℕ) (final : ℕ) 
  (h_initial : initial = 26)
  (h_found : found = 6)
  (h_lost : lost = 10)
  (h_final : final = 42) :
  (final - (initial + found - lost)) / lost = 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_l3023_302378


namespace NUMINAMATH_CALUDE_subset_complement_of_intersection_eq_l3023_302338

universe u

theorem subset_complement_of_intersection_eq {U : Type u} [TopologicalSpace U] (M N : Set U) 
  (h : M ∩ N = N) : (Mᶜ : Set U) ⊆ Nᶜ := by
  sorry

end NUMINAMATH_CALUDE_subset_complement_of_intersection_eq_l3023_302338


namespace NUMINAMATH_CALUDE_salary_comparison_l3023_302302

theorem salary_comparison (a b : ℝ) (h : a = 0.8 * b) :
  (b - a) / a * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_salary_comparison_l3023_302302


namespace NUMINAMATH_CALUDE_right_triangle_condition_l3023_302370

/-- If sin γ - cos α = cos β in a triangle, then the triangle is right-angled -/
theorem right_triangle_condition (α β γ : ℝ) (h_triangle : α + β + γ = Real.pi) 
  (h_condition : Real.sin γ - Real.cos α = Real.cos β) : 
  α = Real.pi / 2 ∨ β = Real.pi / 2 ∨ γ = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_condition_l3023_302370


namespace NUMINAMATH_CALUDE_prob_both_three_eq_one_forty_second_l3023_302337

/-- A fair die with n sides -/
def FairDie (n : ℕ) := Fin n

/-- The probability of rolling a specific number on a fair die with n sides -/
def prob_specific_roll (n : ℕ) : ℚ := 1 / n

/-- The probability of rolling a 3 on a 6-sided die and a 7-sided die simultaneously -/
def prob_both_three : ℚ := (prob_specific_roll 6) * (prob_specific_roll 7)

theorem prob_both_three_eq_one_forty_second :
  prob_both_three = 1 / 42 := by sorry

end NUMINAMATH_CALUDE_prob_both_three_eq_one_forty_second_l3023_302337


namespace NUMINAMATH_CALUDE_no_hexagon_for_19_and_20_l3023_302342

theorem no_hexagon_for_19_and_20 : 
  (¬ ∃ (ℓ : ℤ), 19 = 2 * ℓ^2 + ℓ) ∧ (¬ ∃ (ℓ : ℤ), 20 = 2 * ℓ^2 + ℓ) := by
  sorry

end NUMINAMATH_CALUDE_no_hexagon_for_19_and_20_l3023_302342


namespace NUMINAMATH_CALUDE_equation_solution_l3023_302357

theorem equation_solution : 
  ∃! x : ℝ, (1 / (x + 3) + 3 * x / (x + 3) - 4 / (x + 3) = 4) ∧ x = -15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3023_302357


namespace NUMINAMATH_CALUDE_baguettes_left_l3023_302300

/-- The number of batches of baguettes made per day -/
def batches_per_day : ℕ := 3

/-- The number of baguettes in each batch -/
def baguettes_per_batch : ℕ := 48

/-- The number of baguettes sold after the first batch -/
def sold_after_first : ℕ := 37

/-- The number of baguettes sold after the second batch -/
def sold_after_second : ℕ := 52

/-- The number of baguettes sold after the third batch -/
def sold_after_third : ℕ := 49

/-- Theorem stating that the number of baguettes left is 6 -/
theorem baguettes_left : 
  batches_per_day * baguettes_per_batch - (sold_after_first + sold_after_second + sold_after_third) = 6 := by
  sorry

end NUMINAMATH_CALUDE_baguettes_left_l3023_302300


namespace NUMINAMATH_CALUDE_factorization_equality_l3023_302387

theorem factorization_equality (x : ℝ) : 12 * x^2 + 18 * x - 24 = 6 * (2 * x - 1) * (x + 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3023_302387


namespace NUMINAMATH_CALUDE_complex_power_sum_l3023_302368

theorem complex_power_sum (i : ℂ) (h1 : i^2 = -1) (h2 : i^4 = 1) :
  i^15 + i^22 + i^29 + i^36 + i^43 = -i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l3023_302368


namespace NUMINAMATH_CALUDE_largest_number_l3023_302317

theorem largest_number (a b c d e : ℝ) : 
  a = 12345 + 1/5678 →
  b = 12345 - 1/5678 →
  c = 12345 * 1/5678 →
  d = 12345 / (1/5678) →
  e = 12345.5678 →
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l3023_302317


namespace NUMINAMATH_CALUDE_remaining_cents_l3023_302345

-- Define the number of quarters Winston has
def initial_quarters : ℕ := 14

-- Define the value of a quarter in cents
def cents_per_quarter : ℕ := 25

-- Define the amount spent in cents (half a dollar)
def amount_spent : ℕ := 50

-- Theorem to prove
theorem remaining_cents :
  initial_quarters * cents_per_quarter - amount_spent = 300 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cents_l3023_302345


namespace NUMINAMATH_CALUDE_intersection_condition_l3023_302314

/-- The set M in ℝ² -/
def M : Set (ℝ × ℝ) := {p | p.2 ≥ p.1^2}

/-- The set N in ℝ² parameterized by a -/
def N (a : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - a)^2 ≤ 1}

/-- The necessary and sufficient condition for M ∩ N = N -/
theorem intersection_condition (a : ℝ) : M ∩ N a = N a ↔ a ≥ 5/4 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_l3023_302314


namespace NUMINAMATH_CALUDE_kiki_scarf_problem_l3023_302332

/-- Kiki's scarf and hat buying problem -/
theorem kiki_scarf_problem (total_money : ℝ) (scarf_price : ℝ) :
  total_money = 90 →
  scarf_price = 2 →
  ∃ (num_scarves num_hats : ℕ) (hat_price : ℝ),
    num_hats = 2 * num_scarves ∧
    hat_price * num_hats = 0.6 * total_money ∧
    scarf_price * num_scarves = 0.4 * total_money ∧
    num_scarves = 18 := by
  sorry


end NUMINAMATH_CALUDE_kiki_scarf_problem_l3023_302332


namespace NUMINAMATH_CALUDE_binary_110011_is_51_l3023_302350

def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_110011_is_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_is_51_l3023_302350


namespace NUMINAMATH_CALUDE_quadratic_roots_l3023_302326

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), x₁ = Real.sqrt 3 ∧ x₂ = -Real.sqrt 3 ∧ x₁^2 - 3 = 0 ∧ x₂^2 - 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3023_302326


namespace NUMINAMATH_CALUDE_sum_a_b_equals_nine_l3023_302365

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (a b : ℝ) : Prop :=
  i * (a - i) = b - (2 * i) ^ 3

-- Theorem statement
theorem sum_a_b_equals_nine (a b : ℝ) (h : equation a b) : a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_nine_l3023_302365


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3023_302331

theorem constant_term_binomial_expansion :
  ∃ (n : ℕ), n = 11 ∧ 
  (∀ (r : ℕ), (15 : ℝ) - (3 / 2 : ℝ) * r = 0 → r = 10) ∧
  (∀ (k : ℕ), k ≠ n - 1 → 
    ∃ (c : ℝ), c ≠ 0 ∧ 
    (Nat.choose 15 k * (6 : ℝ)^(15 - k) * (-1 : ℝ)^k) * (0 : ℝ)^(15 - (3 * k) / 2) = c) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3023_302331


namespace NUMINAMATH_CALUDE_distance_is_95_over_17_l3023_302383

def point : ℝ × ℝ × ℝ := (2, 4, 5)
def line_point : ℝ × ℝ × ℝ := (5, 8, 9)
def line_direction : ℝ × ℝ × ℝ := (4, 3, -3)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_is_95_over_17 : 
  distance_to_line point line_point line_direction = 95 / 17 := by
  sorry

end NUMINAMATH_CALUDE_distance_is_95_over_17_l3023_302383


namespace NUMINAMATH_CALUDE_candy_necklaces_per_pack_l3023_302316

theorem candy_necklaces_per_pack (total_packs : ℕ) (opened_packs : ℕ) (leftover_necklaces : ℕ) 
  (h1 : total_packs = 9)
  (h2 : opened_packs = 4)
  (h3 : leftover_necklaces = 40) :
  leftover_necklaces / (total_packs - opened_packs) = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_necklaces_per_pack_l3023_302316


namespace NUMINAMATH_CALUDE_rhombus_diagonal_sum_max_l3023_302319

theorem rhombus_diagonal_sum_max (s x y : ℝ) : 
  s = 5 → 
  x^2 + y^2 = 4 * s^2 →
  x ≥ 6 →
  y ≤ 6 →
  x + y ≤ 14 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_sum_max_l3023_302319


namespace NUMINAMATH_CALUDE_johns_donation_l3023_302330

theorem johns_donation (n : ℕ) (new_avg : ℚ) (increase_percent : ℚ) :
  n = 1 →
  new_avg = 75 →
  increase_percent = 50 / 100 →
  let old_avg := new_avg / (1 + increase_percent)
  let total_before := old_avg * n
  let total_after := new_avg * (n + 1)
  total_after - total_before = 100 := by
sorry

end NUMINAMATH_CALUDE_johns_donation_l3023_302330


namespace NUMINAMATH_CALUDE_solve_equation_l3023_302371

theorem solve_equation (C D : ℚ) 
  (eq1 : 5 * C + 3 * D - 4 = 47) 
  (eq2 : C = D + 2) : 
  C = 57 / 8 ∧ D = 41 / 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3023_302371


namespace NUMINAMATH_CALUDE_sentence_B_is_correct_l3023_302369

/-- Represents a sentence in English --/
structure Sentence where
  text : String

/-- Checks if a sentence is grammatically correct --/
def is_grammatically_correct (s : Sentence) : Prop := sorry

/-- The four sentences given in the problem --/
def sentence_A : Sentence := { text := "The \"Criminal Law Amendment (IX)\", which was officially implemented on November 1, 2015, criminalizes exam cheating for the first time, showing the government's strong determination to combat exam cheating, and may become the \"magic weapon\" to govern the chaos of exams." }

def sentence_B : Sentence := { text := "The APEC Business Leaders Summit is held during the annual APEC Leaders' Informal Meeting. It is an important platform for dialogue and exchange between leaders of economies and the business community, and it is also the most influential business event in the Asia-Pacific region." }

def sentence_C : Sentence := { text := "Since the implementation of the comprehensive two-child policy, many Chinese families have chosen not to have a second child. It is said that it's not because they don't want to, but because they can't afford it, as the cost of raising a child in China is too high." }

def sentence_D : Sentence := { text := "Although it ended up being a futile effort, having fought for a dream, cried, and laughed, we are without regrets. For us, such experiences are treasures in themselves." }

/-- Theorem stating that sentence B is grammatically correct --/
theorem sentence_B_is_correct :
  is_grammatically_correct sentence_B ∧
  ¬is_grammatically_correct sentence_A ∧
  ¬is_grammatically_correct sentence_C ∧
  ¬is_grammatically_correct sentence_D :=
by
  sorry

end NUMINAMATH_CALUDE_sentence_B_is_correct_l3023_302369


namespace NUMINAMATH_CALUDE_x_minus_y_value_l3023_302390

theorem x_minus_y_value (x y : ℝ) 
  (h1 : |x| = 3) 
  (h2 : y^2 = 1/4) 
  (h3 : x + y < 0) : 
  x - y = -7/2 ∨ x - y = -5/2 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l3023_302390


namespace NUMINAMATH_CALUDE_complex_division_simplification_l3023_302304

theorem complex_division_simplification :
  (1 - 2 * Complex.I) / (1 + Complex.I) = -1/2 - 3/2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l3023_302304


namespace NUMINAMATH_CALUDE_fruit_stand_problem_l3023_302320

/-- Represents the number of fruits Mary selects -/
structure FruitSelection where
  apples : ℕ
  oranges : ℕ
  bananas : ℕ

/-- Calculates the total cost of fruits in cents -/
def totalCost (s : FruitSelection) : ℕ :=
  40 * s.apples + 60 * s.oranges + 80 * s.bananas

/-- Calculates the average cost of fruits in cents -/
def averageCost (s : FruitSelection) : ℚ :=
  (totalCost s : ℚ) / (s.apples + s.oranges + s.bananas : ℚ)

theorem fruit_stand_problem (s : FruitSelection) 
  (total_fruits : s.apples + s.oranges + s.bananas = 12)
  (initial_avg : averageCost s = 55) :
  let new_selection := FruitSelection.mk s.apples (s.oranges - 6) s.bananas
  averageCost new_selection = 50 := by
  sorry

end NUMINAMATH_CALUDE_fruit_stand_problem_l3023_302320


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3023_302392

theorem fraction_subtraction : (4 : ℚ) / 5 - (1 : ℚ) / 5 = (3 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3023_302392


namespace NUMINAMATH_CALUDE_condo_has_23_floors_l3023_302346

/-- Represents a condo development with regular and penthouse floors -/
structure CondoDevelopment where
  total_units : ℕ
  regular_units_per_floor : ℕ
  penthouse_units_per_floor : ℕ
  penthouse_floors : ℕ

/-- Calculates the total number of floors in a condo development -/
def total_floors (condo : CondoDevelopment) : ℕ :=
  let regular_floors := (condo.total_units - condo.penthouse_floors * condo.penthouse_units_per_floor) / condo.regular_units_per_floor
  regular_floors + condo.penthouse_floors

/-- Theorem stating that a condo development with the given specifications has 23 floors -/
theorem condo_has_23_floors :
  let condo := CondoDevelopment.mk 256 12 2 2
  total_floors condo = 23 := by
  sorry

end NUMINAMATH_CALUDE_condo_has_23_floors_l3023_302346


namespace NUMINAMATH_CALUDE_translated_circle_equation_l3023_302375

-- Define the points A and B
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (5, 8)

-- Define the translation vector u
def u : ℝ × ℝ := (2, -1)

-- Define the theorem
theorem translated_circle_equation :
  let diameter := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let radius := diameter / 2
  let center := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let new_center := (center.1 + u.1, center.2 + u.2)
  ∀ x y : ℝ, (x - new_center.1)^2 + (y - new_center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_translated_circle_equation_l3023_302375


namespace NUMINAMATH_CALUDE_sqrt_52_rational_l3023_302307

theorem sqrt_52_rational : 
  (((52 : ℝ).sqrt + 5) ^ (1/3 : ℝ)) - (((52 : ℝ).sqrt - 5) ^ (1/3 : ℝ)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_52_rational_l3023_302307


namespace NUMINAMATH_CALUDE_max_product_constrained_l3023_302376

theorem max_product_constrained (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_constraint : 3*x + 2*y = 12) :
  x * y ≤ 6 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3*x₀ + 2*y₀ = 12 ∧ x₀ * y₀ = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_l3023_302376


namespace NUMINAMATH_CALUDE_cut_out_pieces_border_l3023_302329

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a piece that can be cut out from the grid -/
inductive Piece
  | UnitSquare
  | LShape

/-- Represents the configuration of cut-out pieces -/
structure CutOutConfig :=
  (grid : Grid)
  (unitSquares : ℕ)
  (lShapes : ℕ)

/-- Predicate to check if two pieces border each other -/
def border (p1 p2 : Piece) : Prop := sorry

theorem cut_out_pieces_border
  (config : CutOutConfig)
  (h1 : config.grid.size = 55)
  (h2 : config.unitSquares = 500)
  (h3 : config.lShapes = 400) :
  ∃ (p1 p2 : Piece), p1 ≠ p2 ∧ border p1 p2 :=
sorry

end NUMINAMATH_CALUDE_cut_out_pieces_border_l3023_302329


namespace NUMINAMATH_CALUDE_middle_number_proof_l3023_302355

theorem middle_number_proof (x y : ℝ) : 
  (3*x)^2 + (2*x)^2 + (5*x)^2 = 1862 →
  3*x + 2*x + 5*x + 4*y + 7*y = 155 →
  2*x = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l3023_302355


namespace NUMINAMATH_CALUDE_second_number_calculation_l3023_302313

theorem second_number_calculation (A B : ℝ) (h1 : A = 680) (h2 : 0.2 * A = 0.4 * B + 80) : B = 140 := by
  sorry

end NUMINAMATH_CALUDE_second_number_calculation_l3023_302313


namespace NUMINAMATH_CALUDE_sector_area_l3023_302352

theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (h1 : perimeter = 8) (h2 : central_angle = 2) :
  let radius := (perimeter - central_angle * (perimeter / (2 + central_angle))) / 2
  let arc_length := central_angle * radius
  (1 / 2) * radius * arc_length = 4 := by sorry

end NUMINAMATH_CALUDE_sector_area_l3023_302352


namespace NUMINAMATH_CALUDE_max_intersection_points_l3023_302377

-- Define a circle on a plane
def Circle : Type := Unit

-- Define a line on a plane
def Line : Type := Unit

-- Function to count intersection points between a circle and a line
def circleLineIntersections (c : Circle) (l : Line) : ℕ := 2

-- Function to count intersection points between two lines
def lineLineIntersections (l1 l2 : Line) : ℕ := 1

-- Theorem stating the maximum number of intersection points
theorem max_intersection_points (c : Circle) (l1 l2 l3 : Line) :
  ∃ (n : ℕ), n ≤ 9 ∧ 
  (∀ (m : ℕ), m ≤ circleLineIntersections c l1 + 
               circleLineIntersections c l2 + 
               circleLineIntersections c l3 + 
               lineLineIntersections l1 l2 + 
               lineLineIntersections l1 l3 + 
               lineLineIntersections l2 l3 → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_max_intersection_points_l3023_302377


namespace NUMINAMATH_CALUDE_eight_divisors_l3023_302361

theorem eight_divisors (n : ℕ) : (Finset.card (Nat.divisors n) = 8) ↔ 
  (∃ p : ℕ, Nat.Prime p ∧ n = p^7) ∨ 
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p * q^3) ∨ 
  (∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ n = p * q * r) :=
by sorry

end NUMINAMATH_CALUDE_eight_divisors_l3023_302361


namespace NUMINAMATH_CALUDE_exactly_one_head_and_two_heads_mutually_exclusive_but_not_complementary_l3023_302380

/-- Represents the outcome of a single coin toss -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the outcome of tossing two coins -/
def TwoCoinsOutcome := CoinOutcome × CoinOutcome

/-- The event of exactly one head facing up -/
def exactlyOneHead (outcome : TwoCoinsOutcome) : Prop :=
  (outcome.1 = CoinOutcome.Heads ∧ outcome.2 = CoinOutcome.Tails) ∨
  (outcome.1 = CoinOutcome.Tails ∧ outcome.2 = CoinOutcome.Heads)

/-- The event of exactly two heads facing up -/
def exactlyTwoHeads (outcome : TwoCoinsOutcome) : Prop :=
  outcome.1 = CoinOutcome.Heads ∧ outcome.2 = CoinOutcome.Heads

/-- The sample space of all possible outcomes when tossing two coins -/
def sampleSpace : Set TwoCoinsOutcome :=
  {(CoinOutcome.Heads, CoinOutcome.Heads),
   (CoinOutcome.Heads, CoinOutcome.Tails),
   (CoinOutcome.Tails, CoinOutcome.Heads),
   (CoinOutcome.Tails, CoinOutcome.Tails)}

theorem exactly_one_head_and_two_heads_mutually_exclusive_but_not_complementary :
  (∀ (outcome : TwoCoinsOutcome), ¬(exactlyOneHead outcome ∧ exactlyTwoHeads outcome)) ∧
  (∃ (outcome : TwoCoinsOutcome), ¬exactlyOneHead outcome ∧ ¬exactlyTwoHeads outcome) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_head_and_two_heads_mutually_exclusive_but_not_complementary_l3023_302380


namespace NUMINAMATH_CALUDE_highest_divisible_digit_l3023_302385

theorem highest_divisible_digit : 
  ∃ (a : ℕ), a ≤ 9 ∧ 
  (365 * 1000 + a * 100 + 16) % 8 = 0 ∧
  ∀ (b : ℕ), b ≤ 9 → b > a → (365 * 1000 + b * 100 + 16) % 8 ≠ 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_highest_divisible_digit_l3023_302385


namespace NUMINAMATH_CALUDE_valid_grid_exists_l3023_302394

/-- Represents a 6x6 grid of integers -/
def Grid := Matrix (Fin 6) (Fin 6) ℕ

/-- Checks if a given row in the grid contains distinct numbers from 1 to 6 -/
def validRow (g : Grid) (row : Fin 6) : Prop :=
  (Set.range fun col => g row col) = {1, 2, 3, 4, 5, 6}

/-- Checks if a given column in the grid contains distinct numbers from 1 to 6 -/
def validColumn (g : Grid) (col : Fin 6) : Prop :=
  (Set.range fun row => g row col) = {1, 2, 3, 4, 5, 6}

/-- Checks if the grid satisfies all row and column constraints -/
def validGrid (g : Grid) : Prop :=
  (∀ row, validRow g row) ∧ (∀ col, validColumn g col)

/-- The main theorem stating the existence of a valid grid with the given properties -/
theorem valid_grid_exists : ∃ (g : Grid), 
  validGrid g ∧ 
  g 1 1 = 5 ∧
  g 2 3 = 6 ∧
  g 5 0 = 4 ∧
  g 5 1 = 6 ∧
  g 5 2 = 1 ∧
  g 5 3 = 2 ∧
  g 5 4 = 3 :=
sorry


end NUMINAMATH_CALUDE_valid_grid_exists_l3023_302394


namespace NUMINAMATH_CALUDE_water_bottle_consumption_l3023_302359

theorem water_bottle_consumption (total_bottles : ℕ) (days : ℕ) (bottles_per_day : ℕ) : 
  total_bottles = 153 → 
  days = 17 → 
  total_bottles = bottles_per_day * days → 
  bottles_per_day = 9 := by
  sorry

end NUMINAMATH_CALUDE_water_bottle_consumption_l3023_302359


namespace NUMINAMATH_CALUDE_probability_two_girls_l3023_302305

theorem probability_two_girls (total_members : ℕ) (girl_members : ℕ) : 
  total_members = 15 → girl_members = 6 → 
  (Nat.choose girl_members 2 : ℚ) / (Nat.choose total_members 2 : ℚ) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_girls_l3023_302305


namespace NUMINAMATH_CALUDE_relationship_a_x_l3023_302372

theorem relationship_a_x (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 + b^3 = 14*x^3) 
  (h3 : a + b = x) : 
  a = (Real.sqrt 165 - 3) / 6 * x ∨ a = -(Real.sqrt 165 + 3) / 6 * x := by
  sorry

end NUMINAMATH_CALUDE_relationship_a_x_l3023_302372


namespace NUMINAMATH_CALUDE_ceiling_product_equation_l3023_302349

theorem ceiling_product_equation (x : ℝ) : 
  ⌈x⌉ * x = 198 ↔ x = 13.2 := by sorry

end NUMINAMATH_CALUDE_ceiling_product_equation_l3023_302349


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l3023_302312

theorem fraction_zero_implies_x_equals_three (x : ℝ) :
  (x^2 - 9) / (x + 3) = 0 → x = 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l3023_302312


namespace NUMINAMATH_CALUDE_direct_inverse_variation_l3023_302384

theorem direct_inverse_variation (k : ℝ) : 
  (∃ (R S T : ℝ), R = k * S / T ∧ R = 2 ∧ S = 6 ∧ T = 3) →
  (∀ (R S T : ℝ), R = k * S / T → R = 8 ∧ T = 2 → S = 16) :=
by sorry

end NUMINAMATH_CALUDE_direct_inverse_variation_l3023_302384


namespace NUMINAMATH_CALUDE_unique_perpendicular_line_l3023_302339

/-- A line in a plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Predicate to check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Two lines are perpendicular if their slopes are negative reciprocals of each other -/
def Line.isPerpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- Main theorem: There exists exactly one perpendicular line through a point on a given line -/
theorem unique_perpendicular_line (l : Line) (p : Point) (h : p.liesOn l) :
  ∃! l_perp : Line, l_perp.isPerpendicular l ∧ p.liesOn l_perp :=
sorry

end NUMINAMATH_CALUDE_unique_perpendicular_line_l3023_302339


namespace NUMINAMATH_CALUDE_deriv_zero_necessary_not_sufficient_l3023_302325

-- Define a differentiable function f from ℝ to ℝ
variable (f : ℝ → ℝ) (hf : Differentiable ℝ f)

-- Define what it means for a point to be an extremum
def IsExtremum (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ f x ≥ f x₀

-- State the theorem
theorem deriv_zero_necessary_not_sufficient :
  (∀ x₀, IsExtremum f x₀ → deriv f x₀ = 0) ∧
  ¬(∀ x₀, deriv f x₀ = 0 → IsExtremum f x₀) :=
sorry

end NUMINAMATH_CALUDE_deriv_zero_necessary_not_sufficient_l3023_302325


namespace NUMINAMATH_CALUDE_jasons_books_l3023_302306

theorem jasons_books (books_per_shelf : ℕ) (num_shelves : ℕ) (h1 : books_per_shelf = 45) (h2 : num_shelves = 7) :
  books_per_shelf * num_shelves = 315 := by
  sorry

end NUMINAMATH_CALUDE_jasons_books_l3023_302306


namespace NUMINAMATH_CALUDE_integral_x_minus_one_l3023_302356

theorem integral_x_minus_one : ∫ x in (0 : ℝ)..2, (x - 1) = 0 := by sorry

end NUMINAMATH_CALUDE_integral_x_minus_one_l3023_302356


namespace NUMINAMATH_CALUDE_paving_rate_per_square_meter_l3023_302328

/-- Given a room with length 5.5 m and width 3.75 m, and a total paving cost of Rs. 16500,
    the rate of paving per square meter is Rs. 800. -/
theorem paving_rate_per_square_meter
  (length : ℝ)
  (width : ℝ)
  (total_cost : ℝ)
  (h_length : length = 5.5)
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 16500) :
  total_cost / (length * width) = 800 := by
sorry

end NUMINAMATH_CALUDE_paving_rate_per_square_meter_l3023_302328


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3023_302373

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x < 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3023_302373


namespace NUMINAMATH_CALUDE_min_dot_product_OA_OP_l3023_302386

/-- The minimum dot product of OA and OP -/
theorem min_dot_product_OA_OP : ∃ (min : ℝ),
  (∀ x y : ℝ, x > 0 → y = 9 / x → (1 * x + 1 * y) ≥ min) ∧
  (∃ x y : ℝ, x > 0 ∧ y = 9 / x ∧ 1 * x + 1 * y = min) ∧
  min = 6 := by sorry

end NUMINAMATH_CALUDE_min_dot_product_OA_OP_l3023_302386


namespace NUMINAMATH_CALUDE_average_age_problem_l3023_302353

theorem average_age_problem (a b c : ℕ) : 
  (a + b + c) / 3 = 29 →
  b = 23 →
  (a + c) / 2 = 32 := by
sorry

end NUMINAMATH_CALUDE_average_age_problem_l3023_302353


namespace NUMINAMATH_CALUDE_range_of_a_l3023_302398

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | |x - a| < 4}
def B : Set ℝ := {x | (x - 2) * (3 - x) > 0}

-- Define the proposition p and q
def p (a : ℝ) (x : ℝ) : Prop := x ∈ A a
def q (x : ℝ) : Prop := x ∈ B

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, ¬(p a x) → ¬(q x)) →
  a ∈ Set.Icc (-1 : ℝ) 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3023_302398


namespace NUMINAMATH_CALUDE_fraction_division_problem_solution_l3023_302334

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem problem_solution : (5 / 3) / (8 / 15) = 25 / 8 :=
by
  -- Apply the fraction division theorem
  have h1 : (5 / 3) / (8 / 15) = (5 * 15) / (3 * 8) := by sorry
  
  -- Simplify the numerator and denominator
  have h2 : (5 * 15) / (3 * 8) = 75 / 24 := by sorry
  
  -- Further simplify the fraction
  have h3 : 75 / 24 = 25 / 8 := by sorry
  
  -- Combine the steps
  sorry

end NUMINAMATH_CALUDE_fraction_division_problem_solution_l3023_302334


namespace NUMINAMATH_CALUDE_min_value_sum_product_l3023_302366

theorem min_value_sum_product (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d)) ≥ 36 / 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l3023_302366


namespace NUMINAMATH_CALUDE_circle_m_equation_l3023_302358

/-- A circle M passing through two points with its center on a given line -/
structure CircleM where
  -- Circle M passes through these two points
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  -- The center of circle M lies on this line
  center_line : ℝ → ℝ → ℝ
  -- Conditions from the problem
  h1 : point1 = (0, 2)
  h2 : point2 = (0, 4)
  h3 : ∀ x y, center_line x y = 2*x - y - 1

/-- The equation of circle M -/
def circle_equation (c : CircleM) (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 5

/-- Theorem stating that the given conditions imply the circle equation -/
theorem circle_m_equation (c : CircleM) :
  ∀ x y, circle_equation c x y :=
sorry

end NUMINAMATH_CALUDE_circle_m_equation_l3023_302358


namespace NUMINAMATH_CALUDE_male_listeners_count_l3023_302396

/-- Represents the survey data for Radio Wave XFM --/
structure SurveyData where
  total_listeners : ℕ
  total_non_listeners : ℕ
  female_listeners : ℕ
  male_non_listeners : ℕ

/-- Calculates the number of male listeners given the survey data --/
def male_listeners (data : SurveyData) : ℕ :=
  data.total_listeners - data.female_listeners

/-- Theorem stating that the number of male listeners is 75 --/
theorem male_listeners_count (data : SurveyData)
  (h1 : data.total_listeners = 150)
  (h2 : data.total_non_listeners = 180)
  (h3 : data.female_listeners = 75)
  (h4 : data.male_non_listeners = 84) :
  male_listeners data = 75 := by
  sorry

#eval male_listeners { total_listeners := 150, total_non_listeners := 180, female_listeners := 75, male_non_listeners := 84 }

end NUMINAMATH_CALUDE_male_listeners_count_l3023_302396


namespace NUMINAMATH_CALUDE_abs_ratio_greater_than_one_l3023_302374

theorem abs_ratio_greater_than_one (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  |a| / |b| > 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_ratio_greater_than_one_l3023_302374


namespace NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l3023_302308

theorem points_in_quadrants_I_and_II (x y : ℝ) :
  y > 3 * x → y > -2 * x + 3 → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l3023_302308


namespace NUMINAMATH_CALUDE_product_of_roots_l3023_302393

theorem product_of_roots (x : ℝ) : (x - 1) * (x + 4) = 22 → ∃ y : ℝ, (x - 1) * (x + 4) = 22 ∧ (y - 1) * (y + 4) = 22 ∧ x * y = -26 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3023_302393


namespace NUMINAMATH_CALUDE_kyle_track_laps_l3023_302388

/-- The number of laps Kyle jogged in P.E. class -/
def pe_laps : ℝ := 1.12

/-- The total number of laps Kyle jogged -/
def total_laps : ℝ := 3.25

/-- The number of laps Kyle jogged during track practice -/
def track_laps : ℝ := total_laps - pe_laps

theorem kyle_track_laps : track_laps = 2.13 := by
  sorry

end NUMINAMATH_CALUDE_kyle_track_laps_l3023_302388


namespace NUMINAMATH_CALUDE_inequality_solution_l3023_302303

theorem inequality_solution (x : ℝ) : 
  (10 * x^2 + 20 * x - 60) / ((3 * x - 5) * (x + 6)) < 4 ↔ 
  (x > -6 ∧ x < 5/3) ∨ x > 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3023_302303


namespace NUMINAMATH_CALUDE_race_finish_orders_l3023_302362

theorem race_finish_orders (n : ℕ) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end NUMINAMATH_CALUDE_race_finish_orders_l3023_302362


namespace NUMINAMATH_CALUDE_interchange_digits_sum_product_l3023_302301

/-- Given a two-digit number n and a constant k, prove that if n is (k+1) times the sum of its digits,
    then the number formed by interchanging its digits is (10-k) times the sum of its digits. -/
theorem interchange_digits_sum_product (a b k : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : b ≤ 9) :
  (10 * a + b = (k + 1) * (a + b)) →
  (10 * b + a = (10 - k) * (a + b)) :=
by sorry

end NUMINAMATH_CALUDE_interchange_digits_sum_product_l3023_302301


namespace NUMINAMATH_CALUDE_anthony_transaction_percentage_l3023_302363

/-- Proves that Anthony handled 10% more transactions than Mabel given the conditions in the problem. -/
theorem anthony_transaction_percentage (mabel_transactions cal_transactions jade_transactions : ℕ) 
  (anthony_transactions : ℕ) (anthony_percentage : ℚ) :
  mabel_transactions = 90 →
  cal_transactions = (2 : ℚ) / 3 * anthony_transactions →
  jade_transactions = cal_transactions + 19 →
  jade_transactions = 85 →
  anthony_transactions = mabel_transactions * (1 + anthony_percentage / 100) →
  anthony_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_anthony_transaction_percentage_l3023_302363


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l3023_302327

theorem pizza_toppings_combinations : Nat.choose 7 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l3023_302327


namespace NUMINAMATH_CALUDE_new_person_weight_l3023_302354

/-- Given a group of 8 persons where one person weighing 65 kg is replaced by a new person,
    and the average weight of the group increases by 2.5 kg,
    prove that the weight of the new person is 85 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 85 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l3023_302354


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l3023_302347

theorem largest_prime_divisor_of_factorial_sum : 
  ∃ p : ℕ, Prime p ∧ p ∣ (Nat.factorial 12 + Nat.factorial 13) ∧ 
  ∀ q : ℕ, Prime q → q ∣ (Nat.factorial 12 + Nat.factorial 13) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l3023_302347


namespace NUMINAMATH_CALUDE_select_subjects_with_distinct_grades_l3023_302391

/-- Represents a grade for a single subject -/
def Grade : Type := ℕ

/-- Represents the grades of a student for all subjects -/
def StudentGrades : Type := Fin 12 → Grade

/-- The number of students -/
def numStudents : ℕ := 7

/-- The number of subjects -/
def numSubjects : ℕ := 12

/-- The number of subjects to be selected -/
def numSelectedSubjects : ℕ := 6

theorem select_subjects_with_distinct_grades 
  (grades : Fin numStudents → StudentGrades)
  (h : ∀ i j, i ≠ j → ∃ k, grades i k ≠ grades j k) :
  ∃ (selected : Fin numSelectedSubjects → Fin numSubjects),
    (∀ i j, i ≠ j → ∃ k, grades i (selected k) ≠ grades j (selected k)) :=
sorry

end NUMINAMATH_CALUDE_select_subjects_with_distinct_grades_l3023_302391


namespace NUMINAMATH_CALUDE_roles_assignment_count_l3023_302323

/-- The number of ways to assign n distinct roles to n different people. -/
def assignRoles (n : ℕ) : ℕ := Nat.factorial n

/-- There are four team members. -/
def numTeamMembers : ℕ := 4

/-- There are four different roles. -/
def numRoles : ℕ := 4

/-- Each person can only take one role. -/
axiom one_role_per_person : numTeamMembers = numRoles

theorem roles_assignment_count :
  assignRoles numTeamMembers = 24 :=
sorry

end NUMINAMATH_CALUDE_roles_assignment_count_l3023_302323


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l3023_302343

/-- Calculates the perimeter of a rectangular field given its width and length ratio. --/
def field_perimeter (width : ℝ) (length_ratio : ℝ) : ℝ :=
  2 * (width + length_ratio * width)

/-- Theorem: The perimeter of a rectangular field with width 50 meters and length 7/5 times its width is 240 meters. --/
theorem rectangular_field_perimeter :
  field_perimeter 50 (7/5) = 240 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l3023_302343


namespace NUMINAMATH_CALUDE_parabola_symmetry_transform_l3023_302315

/-- Given a parabola with equation y = -2(x+1)^2 + 3, prove that its transformation
    by symmetry about the line y = 1 results in the equation y = 2(x+1)^2 - 1. -/
theorem parabola_symmetry_transform (x y : ℝ) :
  (y = -2 * (x + 1)^2 + 3) →
  (∃ (y' : ℝ), y' = 2 * (x + 1)^2 - 1 ∧ 
    (∀ (p q : ℝ × ℝ), (p.2 = -2 * (p.1 + 1)^2 + 3 ∧ q.2 = y') → 
      (p.1 = q.1 ∧ p.2 + q.2 = 2))) :=
by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_transform_l3023_302315


namespace NUMINAMATH_CALUDE_texts_sent_per_month_l3023_302367

/-- Represents the number of texts sent per month -/
def T : ℕ := sorry

/-- Represents the cost of the current plan in dollars -/
def current_plan_cost : ℕ := 12

/-- Represents the cost per 30 texts in dollars -/
def text_cost_per_30 : ℕ := 1

/-- Represents the cost per 20 minutes of calls in dollars -/
def call_cost_per_20_min : ℕ := 3

/-- Represents the number of minutes spent on calls per month -/
def call_minutes : ℕ := 60

/-- Represents the cost difference between current and alternative plans in dollars -/
def cost_difference : ℕ := 1

theorem texts_sent_per_month :
  T = 60 ∧
  (T / 30 : ℚ) * text_cost_per_30 + 
  (call_minutes / 20 : ℚ) * call_cost_per_20_min = 
  current_plan_cost - cost_difference :=
by sorry

end NUMINAMATH_CALUDE_texts_sent_per_month_l3023_302367


namespace NUMINAMATH_CALUDE_unique_factorial_solution_l3023_302333

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem unique_factorial_solution :
  ∃! (n : ℕ), n > 0 ∧ factorial (n + 1) + factorial (n + 4) = factorial n * 3480 :=
sorry

end NUMINAMATH_CALUDE_unique_factorial_solution_l3023_302333


namespace NUMINAMATH_CALUDE_basketball_team_score_lower_bound_l3023_302381

theorem basketball_team_score_lower_bound (n : ℕ) (player_scores : Fin n → ℕ) 
  (h1 : n = 12) 
  (h2 : ∀ i, player_scores i ≥ 7) 
  (h3 : ∀ i, player_scores i ≤ 23) : 
  (Finset.sum Finset.univ player_scores) ≥ 84 := by
  sorry

#check basketball_team_score_lower_bound

end NUMINAMATH_CALUDE_basketball_team_score_lower_bound_l3023_302381


namespace NUMINAMATH_CALUDE_no_real_solutions_l3023_302360

theorem no_real_solutions : ¬∃ x : ℝ, (x - 5*x + 12)^2 + 1 = -abs x := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3023_302360


namespace NUMINAMATH_CALUDE_combinations_equal_twenty_l3023_302322

/-- The number of available paint colors. -/
def num_colors : ℕ := 5

/-- The number of available painting methods. -/
def num_methods : ℕ := 4

/-- The total number of combinations of paint colors and painting methods. -/
def total_combinations : ℕ := num_colors * num_methods

/-- Theorem stating that the total number of combinations is 20. -/
theorem combinations_equal_twenty : total_combinations = 20 := by
  sorry

end NUMINAMATH_CALUDE_combinations_equal_twenty_l3023_302322


namespace NUMINAMATH_CALUDE_negative_difference_l3023_302341

theorem negative_difference (a b : ℝ) : -(a - b) = -a + b := by
  sorry

end NUMINAMATH_CALUDE_negative_difference_l3023_302341
