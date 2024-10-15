import Mathlib

namespace NUMINAMATH_GPT_inequality_proof_l431_43108

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * a * c) + c / Real.sqrt (c^2 + 8 * a * b) >= 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l431_43108


namespace NUMINAMATH_GPT_multiple_of_669_l431_43159

theorem multiple_of_669 (k : ℕ) (h : ∃ a : ℤ, 2007 ∣ (a + k : ℤ)^3 - a^3) : 669 ∣ k :=
sorry

end NUMINAMATH_GPT_multiple_of_669_l431_43159


namespace NUMINAMATH_GPT_nth_equation_l431_43177

theorem nth_equation (n : ℕ) (h : n > 0) : 9 * (n - 1) + n = 10 * (n - 1) + 1 :=
by
  sorry

end NUMINAMATH_GPT_nth_equation_l431_43177


namespace NUMINAMATH_GPT_rectangle_square_area_ratio_eq_one_l431_43193

theorem rectangle_square_area_ratio_eq_one (r l w s: ℝ) (h1: l = 2 * w) (h2: r ^ 2 = (l / 2) ^ 2 + w ^ 2) (h3: s ^ 2 = 2 * r ^ 2) : 
  (l * w) / (s ^ 2) = 1 :=
by
sorry

end NUMINAMATH_GPT_rectangle_square_area_ratio_eq_one_l431_43193


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l431_43170

theorem sufficient_but_not_necessary (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) : 
  (a > 1 ∧ b > 1 → a * b > 1) ∧ ¬(a * b > 1 → a > 1 ∧ b > 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l431_43170


namespace NUMINAMATH_GPT_largest_three_digit_number_divisible_by_six_l431_43175

theorem largest_three_digit_number_divisible_by_six : ∃ n : ℕ, (∃ m < 1000, m ≥ 100 ∧ m % 6 = 0 ∧ m = n) ∧ (∀ k < 1000, k ≥ 100 ∧ k % 6 = 0 → k ≤ n) ∧ n = 996 :=
by sorry

end NUMINAMATH_GPT_largest_three_digit_number_divisible_by_six_l431_43175


namespace NUMINAMATH_GPT_power_inequality_l431_43106

variable {a b : ℝ}

theorem power_inequality (ha : 0 < a) (hb : 0 < b) : a^a * b^b ≥ a^b * b^a := 
by sorry

end NUMINAMATH_GPT_power_inequality_l431_43106


namespace NUMINAMATH_GPT_ratio_eval_l431_43198

universe u

def a : ℕ := 121
def b : ℕ := 123
def c : ℕ := 122

theorem ratio_eval : (2 ^ a * 3 ^ b) / (6 ^ c) = (3 / 2) := by
  sorry

end NUMINAMATH_GPT_ratio_eval_l431_43198


namespace NUMINAMATH_GPT_factorials_sum_of_two_squares_l431_43168

-- Define what it means for a number to be a sum of two squares.
def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2

theorem factorials_sum_of_two_squares :
  {n : ℕ | n < 14 ∧ is_sum_of_two_squares (n!)} = {2, 6} :=
by
  sorry

end NUMINAMATH_GPT_factorials_sum_of_two_squares_l431_43168


namespace NUMINAMATH_GPT_robot_distance_covered_l431_43100

theorem robot_distance_covered :
  let start1 := -3
  let end1 := -8
  let end2 := 6
  let distance1 := abs (end1 - start1)
  let distance2 := abs (end2 - end1)
  distance1 + distance2 = 19 := by
  sorry

end NUMINAMATH_GPT_robot_distance_covered_l431_43100


namespace NUMINAMATH_GPT_miles_to_friends_house_l431_43162

-- Define the conditions as constants
def miles_per_gallon : ℕ := 19
def gallons : ℕ := 2
def miles_to_school : ℕ := 15
def miles_to_softball_park : ℕ := 6
def miles_to_burger_restaurant : ℕ := 2
def miles_home : ℕ := 11

-- Define the total miles driven
def total_miles_driven (miles_to_friend : ℕ) :=
  miles_to_school + miles_to_softball_park + miles_to_burger_restaurant + miles_to_friend + miles_home

-- Define the total miles possible with given gallons of gas
def total_miles_possible : ℕ :=
  miles_per_gallon * gallons

-- Prove that the miles driven to the friend's house is 4
theorem miles_to_friends_house : 
  ∃ miles_to_friend, total_miles_driven miles_to_friend = total_miles_possible ∧ miles_to_friend = 4 :=
by
  sorry

end NUMINAMATH_GPT_miles_to_friends_house_l431_43162


namespace NUMINAMATH_GPT_linear_equation_m_equals_neg_3_l431_43143

theorem linear_equation_m_equals_neg_3 
  (m : ℤ)
  (h1 : |m| - 2 = 1)
  (h2 : m - 3 ≠ 0) :
  m = -3 :=
sorry

end NUMINAMATH_GPT_linear_equation_m_equals_neg_3_l431_43143


namespace NUMINAMATH_GPT_sam_pam_ratio_is_2_l431_43157

-- Definition of given conditions
def min_assigned_pages : ℕ := 25
def harrison_extra_read : ℕ := 10
def pam_extra_read : ℕ := 15
def sam_read : ℕ := 100

-- Calculations based on the given conditions
def harrison_read : ℕ := min_assigned_pages + harrison_extra_read
def pam_read : ℕ := harrison_read + pam_extra_read

-- Prove the ratio of the number of pages Sam read to the number of pages Pam read is 2
theorem sam_pam_ratio_is_2 : sam_read / pam_read = 2 := 
by
  sorry

end NUMINAMATH_GPT_sam_pam_ratio_is_2_l431_43157


namespace NUMINAMATH_GPT_divisible_by_square_of_k_l431_43144

theorem divisible_by_square_of_k (a b l : ℕ) (k : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : a % 2 = 1) (h4 : b % 2 = 1) (h5 : a + b = 2 ^ l) : k = 1 ↔ k^2 ∣ a^k + b^k := 
sorry

end NUMINAMATH_GPT_divisible_by_square_of_k_l431_43144


namespace NUMINAMATH_GPT_totalPizzaEaten_l431_43124

-- Define the conditions
def rachelAte : ℕ := 598
def bellaAte : ℕ := 354

-- State the theorem
theorem totalPizzaEaten : rachelAte + bellaAte = 952 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_totalPizzaEaten_l431_43124


namespace NUMINAMATH_GPT_largest_n_with_100_trailing_zeros_l431_43132

def trailing_zeros_factorial (n : ℕ) : ℕ :=
  if n = 0 then 0 else n / 5 + trailing_zeros_factorial (n / 5)

theorem largest_n_with_100_trailing_zeros :
  ∃ (n : ℕ), trailing_zeros_factorial n = 100 ∧ ∀ (m : ℕ), (trailing_zeros_factorial m = 100 → m ≤ 409) :=
by
  sorry

end NUMINAMATH_GPT_largest_n_with_100_trailing_zeros_l431_43132


namespace NUMINAMATH_GPT_number_of_digits_in_sum_l431_43145

def is_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 9

theorem number_of_digits_in_sum (C D : ℕ) (hC : is_digit C) (hD : is_digit D) :
  let n1 := 98765
  let n2 := C * 1000 + 433
  let n3 := D * 100 + 22
  let s := n1 + n2 + n3
  100000 ≤ s ∧ s < 1000000 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_digits_in_sum_l431_43145


namespace NUMINAMATH_GPT_perfect_square_trinomial_l431_43146

theorem perfect_square_trinomial (c : ℝ) : (∃ a : ℝ, (x : ℝ) → x^2 + 150 * x + c = (x + a)^2) → c = 5625 :=
sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l431_43146


namespace NUMINAMATH_GPT_range_of_t_l431_43126

theorem range_of_t (a b : ℝ) (h : a^2 + a*b + b^2 = 1) :
  ∃ t : ℝ, (t = a^2 - a*b + b^2) ∧ (1/3 ≤ t ∧ t ≤ 3) :=
sorry

end NUMINAMATH_GPT_range_of_t_l431_43126


namespace NUMINAMATH_GPT_initial_number_of_women_l431_43171

variable (W : ℕ)

def work_done_by_women_per_day (W : ℕ) : ℚ := 1 / (8 * W)
def work_done_by_children_per_day (W : ℕ) : ℚ := 1 / (12 * W)

theorem initial_number_of_women :
  (6 * work_done_by_women_per_day W + 3 * work_done_by_children_per_day W = 1 / 10) → W = 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_women_l431_43171


namespace NUMINAMATH_GPT_range_of_a_l431_43142

noncomputable def f (x : ℝ) : ℝ := sorry -- The actual definition of the function f is not given
def g (a x : ℝ) : ℝ := a * x - 1

theorem range_of_a (a : ℝ) :
  (∀ x₁ : ℝ, x₁ ∈ Set.Icc (-2 : ℝ) 2 → ∃ x₀ : ℝ, x₀ ∈ Set.Icc (-2 : ℝ) 2 ∧ g a x₀ = f x₁) ↔
  a ≤ -1/2 ∨ 5/2 ≤ a :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l431_43142


namespace NUMINAMATH_GPT_cone_sphere_ratio_l431_43161

-- Defining the conditions and proof goals
theorem cone_sphere_ratio (r h : ℝ) (h_cone_sphere_radius : r ≠ 0) 
  (h_cone_volume : (1 / 3) * π * r^2 * h = (1 / 3) * (4 / 3) * π * r^3) : 
  h / r = 4 / 3 :=
by
  -- All the assumptions / conditions given in the problem
  sorry -- Proof omitted

end NUMINAMATH_GPT_cone_sphere_ratio_l431_43161


namespace NUMINAMATH_GPT_initial_treasure_amount_l431_43140

theorem initial_treasure_amount 
  (T : ℚ)
  (h₁ : T * (1 - 1/13) * (1 - 1/17) = 150) : 
  T = 172 + 21/32 :=
sorry

end NUMINAMATH_GPT_initial_treasure_amount_l431_43140


namespace NUMINAMATH_GPT_amount_after_two_years_l431_43102

-- Definition of initial amount and the rate of increase
def initial_value : ℝ := 32000
def rate_of_increase : ℝ := 0.125
def time_period : ℕ := 2

-- The compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- The proof problem: Prove that after 2 years the amount is 40500
theorem amount_after_two_years : compound_interest initial_value rate_of_increase time_period = 40500 :=
sorry

end NUMINAMATH_GPT_amount_after_two_years_l431_43102


namespace NUMINAMATH_GPT_fraction_meaningful_range_l431_43156

-- Define the condition
def meaningful_fraction_condition (x : ℝ) : Prop := (x - 2023) ≠ 0

-- Define the conclusion that we need to prove
def meaningful_fraction_range (x : ℝ) : Prop := x ≠ 2023

theorem fraction_meaningful_range (x : ℝ) : meaningful_fraction_condition x → meaningful_fraction_range x :=
by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_fraction_meaningful_range_l431_43156


namespace NUMINAMATH_GPT_merchant_product_quantities_l431_43187

theorem merchant_product_quantities
  (x p1 : ℝ)
  (h1 : 4000 = x * p1)
  (h2 : 8800 = 2 * x * (p1 + 4))
  (h3 : (8800 / (2 * x)) - (4000 / x) = 4):
  x = 100 ∧ 2 * x = 200 :=
by sorry

end NUMINAMATH_GPT_merchant_product_quantities_l431_43187


namespace NUMINAMATH_GPT_quarters_addition_l431_43120

def original_quarters : ℝ := 783.0
def added_quarters : ℝ := 271.0
def total_quarters : ℝ := 1054.0

theorem quarters_addition :
  original_quarters + added_quarters = total_quarters :=
by
  sorry

end NUMINAMATH_GPT_quarters_addition_l431_43120


namespace NUMINAMATH_GPT_ap_number_of_terms_is_six_l431_43151

noncomputable def arithmetic_progression_number_of_terms (a d : ℕ) (n : ℕ) : Prop :=
  let odd_sum := (n / 2) * (2 * a + (n - 2) * d)
  let even_sum := (n / 2) * (2 * a + n * d)
  let last_term_condition := (n - 1) * d = 15
  n % 2 = 0 ∧ odd_sum = 30 ∧ even_sum = 36 ∧ last_term_condition

theorem ap_number_of_terms_is_six (a d n : ℕ) (h : arithmetic_progression_number_of_terms a d n) :
  n = 6 :=
by sorry

end NUMINAMATH_GPT_ap_number_of_terms_is_six_l431_43151


namespace NUMINAMATH_GPT_sum_of_integers_is_19_l431_43183

theorem sum_of_integers_is_19
  (a b : ℕ) 
  (h1 : a > b) 
  (h2 : a - b = 5) 
  (h3 : a * b = 84) : 
  a + b = 19 :=
sorry

end NUMINAMATH_GPT_sum_of_integers_is_19_l431_43183


namespace NUMINAMATH_GPT_cos_180_degree_l431_43179

theorem cos_180_degree : Real.cos (180 * Real.pi / 180) = -1 := by
  -- sorry can be replaced by the actual proof in a complete solution
  sorry

end NUMINAMATH_GPT_cos_180_degree_l431_43179


namespace NUMINAMATH_GPT_find_m_l431_43141

theorem find_m (x m : ℝ) (h_eq : (x + m) / (x - 2) + 1 / (2 - x) = 3) (h_root : x = 2) : m = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l431_43141


namespace NUMINAMATH_GPT_seq_sum_difference_l431_43148

-- Define the sequences
def seq1 : List ℕ := List.range 93 |> List.map (λ n => 2001 + n)
def seq2 : List ℕ := List.range 93 |> List.map (λ n => 301 + n)

-- Define the sum of the sequences
def sum_seq1 : ℕ := seq1.sum
def sum_seq2 : ℕ := seq2.sum

-- Define the difference between the sums of the sequences
def diff_seq_sum : ℕ := sum_seq1 - sum_seq2

-- Lean statement to prove the difference equals 158100
theorem seq_sum_difference : diff_seq_sum = 158100 := by
  sorry

end NUMINAMATH_GPT_seq_sum_difference_l431_43148


namespace NUMINAMATH_GPT_highest_student_id_in_sample_l431_43131

theorem highest_student_id_in_sample
    (total_students : ℕ)
    (sample_size : ℕ)
    (included_student_id : ℕ)
    (interval : ℕ)
    (first_id in_sample : ℕ)
    (k : ℕ)
    (highest_id : ℕ)
    (total_students_eq : total_students = 63)
    (sample_size_eq : sample_size = 7)
    (included_student_id_eq : included_student_id = 11)
    (k_def : k = total_students / sample_size)
    (included_student_id_in_second_pos : included_student_id = first_id + k)
    (interval_eq : interval = first_id - k)
    (in_sample_eq : in_sample = interval)
    (highest_id_eq : highest_id = in_sample + k * (sample_size - 1)) :
  highest_id = 56 := sorry

end NUMINAMATH_GPT_highest_student_id_in_sample_l431_43131


namespace NUMINAMATH_GPT_problem_solution_l431_43153

theorem problem_solution (x : ℝ) (h1 : x = 12) (h2 : 5 + 7 / x = some_number - 5 / x) : some_number = 6 := 
by
  sorry

end NUMINAMATH_GPT_problem_solution_l431_43153


namespace NUMINAMATH_GPT_sum_of_first_four_terms_of_sequence_l431_43150

-- Define the sequence, its common difference, and the given initial condition
def a_sequence (a : ℕ → ℤ) : Prop :=
  (∀ n : ℕ, a (n + 1) - a n = 2) ∧ (a 2 = 5)

-- Define the sum of the first four terms
def sum_first_four_terms (a : ℕ → ℤ) : ℤ :=
  a 0 + a 1 + a 2 + a 3

theorem sum_of_first_four_terms_of_sequence :
  ∀ (a : ℕ → ℤ), a_sequence a → sum_first_four_terms a = 24 :=
by
  intro a h
  rw [a_sequence] at h
  obtain ⟨h_diff, h_a2⟩ := h
  sorry

end NUMINAMATH_GPT_sum_of_first_four_terms_of_sequence_l431_43150


namespace NUMINAMATH_GPT_length_of_route_l431_43127

theorem length_of_route 
  (D vA vB : ℝ)
  (h_vA : vA = D / 10)
  (h_vB : vB = D / 6)
  (t : ℝ)
  (h_va_t : vA * t = 75)
  (h_vb_t : vB * t = D - 75) :
  D = 200 :=
by
  sorry

end NUMINAMATH_GPT_length_of_route_l431_43127


namespace NUMINAMATH_GPT_eleven_million_scientific_notation_l431_43139

-- Definition of the scientific notation condition and question
def scientific_notation (a n : ℝ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ ∃ k : ℤ, n = 10 ^ k

-- The main theorem stating that 11 million can be expressed as 1.1 * 10^7
theorem eleven_million_scientific_notation : scientific_notation 1.1 (10 ^ 7) :=
by 
  -- Adding sorry to skip the proof
  sorry

end NUMINAMATH_GPT_eleven_million_scientific_notation_l431_43139


namespace NUMINAMATH_GPT_point_on_circle_l431_43118

noncomputable def x_value_on_circle : ℝ :=
  let a := (-3 : ℝ)
  let b := (21 : ℝ)
  let Cx := (a + b) / 2
  let Cy := 0
  let radius := (b - a) / 2
  let y := 12
  Cx

theorem point_on_circle (x y : ℝ) (a b : ℝ) (ha : a = -3) (hb : b = 21) (hy : y = 12) :
  let Cx := (a + b) / 2
  let Cy := 0
  let radius := (b - a) / 2
  (x - Cx) ^ 2 + y ^ 2 = radius ^ 2 → x = x_value_on_circle :=
by
  intros
  sorry

end NUMINAMATH_GPT_point_on_circle_l431_43118


namespace NUMINAMATH_GPT_calculate_truck_loads_of_dirt_l431_43154

noncomputable def truck_loads_sand: ℚ := 0.16666666666666666
noncomputable def truck_loads_cement: ℚ := 0.16666666666666666
noncomputable def total_truck_loads_material: ℚ := 0.6666666666666666
noncomputable def truck_loads_dirt: ℚ := total_truck_loads_material - (truck_loads_sand + truck_loads_cement)

theorem calculate_truck_loads_of_dirt :
  truck_loads_dirt = 0.3333333333333333 := 
by
  sorry

end NUMINAMATH_GPT_calculate_truck_loads_of_dirt_l431_43154


namespace NUMINAMATH_GPT_sector_area_l431_43116

theorem sector_area (theta : ℝ) (L : ℝ) (h_theta : theta = π / 3) (h_L : L = 4) :
  ∃ r : ℝ, (L = r * theta ∧ ∃ A : ℝ, A = 1/2 * r^2 * theta ∧ A = 24 / π) := by
  sorry

end NUMINAMATH_GPT_sector_area_l431_43116


namespace NUMINAMATH_GPT_line_equation_l431_43173

theorem line_equation
  (t : ℝ)
  (x : ℝ) (y : ℝ)
  (h1 : x = 3 * t + 6)
  (h2 : y = 5 * t - 10) :
  y = (5 / 3) * x - 20 :=
sorry

end NUMINAMATH_GPT_line_equation_l431_43173


namespace NUMINAMATH_GPT_equilateral_triangle_of_arith_geo_seq_l431_43101

def triangle (A B C : ℝ) (a b c : ℝ) (α β γ : ℝ) :=
  (α + β + γ = Real.pi) ∧
  (2 * β = α + γ) ∧
  (b^2 = a * c)

theorem equilateral_triangle_of_arith_geo_seq
  (A B C : ℝ) (a b c α β γ : ℝ)
  (h1 : triangle A B C a b c α β γ)
  : (a = c) ∧ (A = B) ∧ (B = C) ∧ (a = b) :=
  sorry

end NUMINAMATH_GPT_equilateral_triangle_of_arith_geo_seq_l431_43101


namespace NUMINAMATH_GPT_gcd_hcf_of_36_and_84_l431_43199

theorem gcd_hcf_of_36_and_84 : Nat.gcd 36 84 = 12 := sorry

end NUMINAMATH_GPT_gcd_hcf_of_36_and_84_l431_43199


namespace NUMINAMATH_GPT_pencils_per_row_l431_43122

-- Definitions of conditions.
def num_pencils : ℕ := 35
def num_rows : ℕ := 7

-- Hypothesis: given the conditions, prove the number of pencils per row.
theorem pencils_per_row : num_pencils / num_rows = 5 := 
  by 
  -- Proof steps go here, but are replaced by sorry.
  sorry

end NUMINAMATH_GPT_pencils_per_row_l431_43122


namespace NUMINAMATH_GPT_shifting_parabola_l431_43133

def original_function (x : ℝ) : ℝ := (x + 1)^2 + 3

def shifted_function (x : ℝ) : ℝ := (x - 1)^2 + 2

theorem shifting_parabola : ∀ x : ℝ, shifted_function x = original_function (x + 2) - 1 := 
by 
  sorry

end NUMINAMATH_GPT_shifting_parabola_l431_43133


namespace NUMINAMATH_GPT_nina_walking_distance_l431_43191

def distance_walked_by_john : ℝ := 0.7
def distance_john_further_than_nina : ℝ := 0.3

def distance_walked_by_nina : ℝ := distance_walked_by_john - distance_john_further_than_nina

theorem nina_walking_distance :
  distance_walked_by_nina = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_nina_walking_distance_l431_43191


namespace NUMINAMATH_GPT_money_collected_l431_43128

theorem money_collected
  (households_per_day : ℕ)
  (days : ℕ)
  (half_give_money : ℕ → ℕ)
  (total_money_collected : ℕ)
  (households_give_money : ℕ) :
  households_per_day = 20 →  
  days = 5 →
  total_money_collected = 2000 →
  half_give_money (households_per_day * days) = (households_per_day * days) / 2 →
  households_give_money = (households_per_day * days) / 2 →
  total_money_collected / households_give_money = 40
:= sorry

end NUMINAMATH_GPT_money_collected_l431_43128


namespace NUMINAMATH_GPT_product_of_largest_two_and_four_digit_primes_l431_43160

theorem product_of_largest_two_and_four_digit_primes :
  let largest_two_digit_prime := 97
  let largest_four_digit_prime := 9973
  largest_two_digit_prime * largest_four_digit_prime = 967781 := by
  sorry

end NUMINAMATH_GPT_product_of_largest_two_and_four_digit_primes_l431_43160


namespace NUMINAMATH_GPT_suzanna_bike_distance_l431_43123

variable (constant_rate : ℝ) (time_minutes : ℝ) (interval : ℝ) (distance_per_interval : ℝ)

theorem suzanna_bike_distance :
  (constant_rate = 1 / interval) ∧ (interval = 5) ∧ (distance_per_interval = constant_rate * interval) ∧ (time_minutes = 30) →
  ((time_minutes / interval) * distance_per_interval = 6) :=
by
  intros
  sorry

end NUMINAMATH_GPT_suzanna_bike_distance_l431_43123


namespace NUMINAMATH_GPT_inequality_proof_l431_43158

theorem inequality_proof (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 2) : 
  (1 / a + 1 / b) ≥ 2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l431_43158


namespace NUMINAMATH_GPT_probability_at_least_one_coordinate_greater_l431_43104

theorem probability_at_least_one_coordinate_greater (p : ℝ) :
  (∃ (x y : ℝ), (0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ (x > p ∨ y > p))) ↔ p = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_coordinate_greater_l431_43104


namespace NUMINAMATH_GPT_length_of_side_b_max_area_of_triangle_l431_43152

variable {A B C a b c : ℝ}
variable {triangle_ABC : a + c = 6}
variable {eq1 : (3 - Real.cos A) * Real.sin B = Real.sin A * (1 + Real.cos B)}

-- Theorem for part (1) length of side b
theorem length_of_side_b :
  b = 2 :=
sorry

-- Theorem for part (2) maximum area of the triangle
theorem max_area_of_triangle :
  ∃ (S : ℝ), S = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_length_of_side_b_max_area_of_triangle_l431_43152


namespace NUMINAMATH_GPT_circle_radius_k_l431_43129

theorem circle_radius_k (k : ℝ) : (∃ x y : ℝ, (x^2 + 14*x + y^2 + 8*y - k = 0) ∧ ((x + 7)^2 + (y + 4)^2 = 100)) → k = 35 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_k_l431_43129


namespace NUMINAMATH_GPT_min_mod_z_l431_43165

open Complex

theorem min_mod_z (z : ℂ) (hz : abs (z - 2 * I) + abs (z - 5) = 7) : abs z = 10 / 7 :=
sorry

end NUMINAMATH_GPT_min_mod_z_l431_43165


namespace NUMINAMATH_GPT_point_in_third_quadrant_l431_43147

section quadrant_problem

variables (a b : ℝ)

-- Given: Point (a, b) is in the fourth quadrant
def in_fourth_quadrant (a b : ℝ) : Prop :=
  a > 0 ∧ b < 0

-- To prove: Point (a / b, 2 * b - a) is in the third quadrant
def in_third_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

-- The theorem stating that if (a, b) is in the fourth quadrant,
-- then (a / b, 2 * b - a) is in the third quadrant
theorem point_in_third_quadrant (a b : ℝ) (h : in_fourth_quadrant a b) :
  in_third_quadrant (a / b) (2 * b - a) :=
  sorry

end quadrant_problem

end NUMINAMATH_GPT_point_in_third_quadrant_l431_43147


namespace NUMINAMATH_GPT_sum_of_numbers_l431_43136

theorem sum_of_numbers :
  36 + 17 + 32 + 54 + 28 + 3 = 170 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l431_43136


namespace NUMINAMATH_GPT_goose_eggs_l431_43117

theorem goose_eggs (E : ℝ) :
  (E / 2 * 3 / 4 * 2 / 5 + (1 / 3 * (E / 2)) * 2 / 3 * 3 / 4 + (1 / 6 * (E / 2 + E / 6)) * 1 / 2 * 2 / 3 = 150) →
  E = 375 :=
by
  sorry

end NUMINAMATH_GPT_goose_eggs_l431_43117


namespace NUMINAMATH_GPT_coins_to_rubles_l431_43163

theorem coins_to_rubles (a1 a2 a3 a4 a5 a6 a7 k m : ℕ)
  (h1 : a1 + 2 * a2 + 5 * a3 + 10 * a4 + 20 * a5 + 50 * a6 + 100 * a7 = m)
  (h2 : a1 + a2 + a3 + a4 + a5 + a6 + a7 = k) :
  m * 100 = k :=
by sorry

end NUMINAMATH_GPT_coins_to_rubles_l431_43163


namespace NUMINAMATH_GPT_initial_money_equals_26_l431_43182

def cost_jumper : ℕ := 9
def cost_tshirt : ℕ := 4
def cost_heels : ℕ := 5
def money_left : ℕ := 8

def total_cost_items : ℕ := cost_jumper + cost_tshirt + cost_heels

theorem initial_money_equals_26 : total_cost_items + money_left = 26 := by
  sorry

end NUMINAMATH_GPT_initial_money_equals_26_l431_43182


namespace NUMINAMATH_GPT_seashells_remainder_l431_43167

theorem seashells_remainder :
  let derek := 58
  let emily := 73
  let fiona := 31 
  let total_seashells := derek + emily + fiona
  total_seashells % 10 = 2 :=
by
  sorry

end NUMINAMATH_GPT_seashells_remainder_l431_43167


namespace NUMINAMATH_GPT_complex_number_on_imaginary_axis_l431_43121

theorem complex_number_on_imaginary_axis (a : ℝ) 
(h : ∃ z : ℂ, z = (a^2 - 2 * a) + (a^2 - a - 2) * Complex.I ∧ z.re = 0) : 
a = 0 ∨ a = 2 :=
by
  sorry

end NUMINAMATH_GPT_complex_number_on_imaginary_axis_l431_43121


namespace NUMINAMATH_GPT_cannot_be_square_of_difference_formula_l431_43188

theorem cannot_be_square_of_difference_formula (x y c d a b m n : ℝ) :
  ¬ ((m - n) * (-m + n) = (x^2 - y^2) ∨ 
       (m - n) * (-m + n) = (c^2 - d^2) ∨ 
       (m - n) * (-m + n) = (a^2 - b^2)) :=
by sorry

end NUMINAMATH_GPT_cannot_be_square_of_difference_formula_l431_43188


namespace NUMINAMATH_GPT_find_a_b_k_l431_43149

noncomputable def a (k : ℕ) : ℕ := if h : k = 9 then 243 else sorry
noncomputable def b (k : ℕ) : ℕ := if h : k = 9 then 3 else sorry

theorem find_a_b_k (a b k : ℕ) (hb : b = 3) (ha : a = 243) (hk : k = 9)
  (h1 : a * b = k^3) (h2 : a / b = k^2) (h3 : 100 ≤ a * b ∧ a * b < 1000) :
  a = 243 ∧ b = 3 ∧ k = 9 :=
by 
  sorry

end NUMINAMATH_GPT_find_a_b_k_l431_43149


namespace NUMINAMATH_GPT_pyramid_coloring_ways_l431_43194

theorem pyramid_coloring_ways (colors : Fin 5) 
  (coloring_condition : ∀ (a b : Fin 5), a ≠ b) :
  ∃ (ways: Nat), ways = 420 :=
by
  -- Given:
  -- 1. There are 5 available colors
  -- 2. Each vertex of the pyramid is colored differently from the vertices connected by an edge
  -- Prove:
  -- There are 420 ways to color the pyramid's vertices
  sorry

end NUMINAMATH_GPT_pyramid_coloring_ways_l431_43194


namespace NUMINAMATH_GPT_Clever_not_Green_l431_43176

variables {Lizard : Type}
variables [DecidableEq Lizard] (Clever Green CanJump CanSwim : Lizard → Prop)

theorem Clever_not_Green (h1 : ∀ x, Clever x → CanJump x)
                        (h2 : ∀ x, Green x → ¬ CanSwim x)
                        (h3 : ∀ x, ¬ CanSwim x → ¬ CanJump x) :
  ∀ x, Clever x → ¬ Green x :=
by
  intro x hClever hGreen
  apply h3 x
  apply h2 x hGreen
  exact h1 x hClever

end NUMINAMATH_GPT_Clever_not_Green_l431_43176


namespace NUMINAMATH_GPT_polynomial_has_three_real_roots_l431_43174

theorem polynomial_has_three_real_roots (a b c : ℝ) (h1 : b < 0) (h2 : a * b = 9 * c) :
  ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    (x1^3 + a * x1^2 + b * x1 + c = 0) ∧ 
    (x2^3 + a * x2^2 + b * x2 + c = 0) ∧ 
    (x3^3 + a * x3^2 + b * x3 + c = 0) := sorry

end NUMINAMATH_GPT_polynomial_has_three_real_roots_l431_43174


namespace NUMINAMATH_GPT_gravel_cost_calculation_l431_43189

def cubicYardToCubicFoot : ℕ := 27
def costPerCubicFoot : ℕ := 8
def volumeInCubicYards : ℕ := 8

theorem gravel_cost_calculation : 
  (volumeInCubicYards * cubicYardToCubicFoot * costPerCubicFoot) = 1728 := 
by
  -- This is just a placeholder to ensure the statement is syntactically correct.
  sorry

end NUMINAMATH_GPT_gravel_cost_calculation_l431_43189


namespace NUMINAMATH_GPT_num_digits_expr_l431_43135

noncomputable def num_digits (n : ℕ) : ℕ :=
  (Int.ofNat n).natAbs.digits 10 |>.length

def expr : ℕ := 2^15 * 5^10 * 12

theorem num_digits_expr : num_digits expr = 13 := by
  sorry

end NUMINAMATH_GPT_num_digits_expr_l431_43135


namespace NUMINAMATH_GPT_mitchell_pizzas_l431_43196

def pizzas_bought (slices_per_goal goals_per_game games slices_per_pizza : ℕ) : ℕ :=
  (slices_per_goal * goals_per_game * games) / slices_per_pizza

theorem mitchell_pizzas : pizzas_bought 1 9 8 12 = 6 := by
  sorry

end NUMINAMATH_GPT_mitchell_pizzas_l431_43196


namespace NUMINAMATH_GPT_largest_cube_surface_area_l431_43195

theorem largest_cube_surface_area (width length height: ℕ) (h_w: width = 12) (h_l: length = 16) (h_h: height = 14) :
  (6 * (min width (min length height))^2) = 864 := by
  sorry

end NUMINAMATH_GPT_largest_cube_surface_area_l431_43195


namespace NUMINAMATH_GPT_possible_values_count_l431_43134

theorem possible_values_count {x y z : ℤ} (h₁ : x = 5) (h₂ : y = -3) (h₃ : z = -1) :
  ∃ v, v = x - y - z ∧ (v = 7 ∨ v = 8 ∨ v = 9) :=
by
  sorry

end NUMINAMATH_GPT_possible_values_count_l431_43134


namespace NUMINAMATH_GPT_julia_change_l431_43192

-- Definitions based on the problem conditions
def price_of_snickers : ℝ := 1.5
def price_of_mms : ℝ := 2 * price_of_snickers
def total_cost_of_snickers (num_snickers : ℕ) : ℝ := num_snickers * price_of_snickers
def total_cost_of_mms (num_mms : ℕ) : ℝ := num_mms * price_of_mms
def total_purchase (num_snickers num_mms : ℕ) : ℝ := total_cost_of_snickers num_snickers + total_cost_of_mms num_mms
def amount_given : ℝ := 2 * 10

-- Prove the change is $8
theorem julia_change : total_purchase 2 3 = 12 ∧ (amount_given - total_purchase 2 3) = 8 :=
by
  sorry

end NUMINAMATH_GPT_julia_change_l431_43192


namespace NUMINAMATH_GPT_at_least_one_gt_one_l431_43166

theorem at_least_one_gt_one (a b : ℝ) (h : a + b > 2) : a > 1 ∨ b > 1 :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_gt_one_l431_43166


namespace NUMINAMATH_GPT_last_two_digits_of_product_squared_l431_43197

def mod_100 (n : ℕ) : ℕ := n % 100

theorem last_two_digits_of_product_squared :
  mod_100 ((301 * 402 * 503 * 604 * 646 * 547 * 448 * 349) ^ 2) = 76 := 
by
  sorry

end NUMINAMATH_GPT_last_two_digits_of_product_squared_l431_43197


namespace NUMINAMATH_GPT_quadratic_solution_l431_43184

theorem quadratic_solution
  (a c : ℝ) (h : a ≠ 0) (h_passes_through : ∃ b, b = c - 9 * a) :
  ∀ (x : ℝ), (ax^2 - 2 * a * x + c = 0) ↔ (x = -1) ∨ (x = 3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_l431_43184


namespace NUMINAMATH_GPT_range_of_function_l431_43109

theorem range_of_function : ∀ x : ℝ, 1 ≤ abs (Real.sin x) + 2 * abs (Real.cos x) ∧ abs (Real.sin x) + 2 * abs (Real.cos x) ≤ Real.sqrt 5 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_range_of_function_l431_43109


namespace NUMINAMATH_GPT_correct_operation_l431_43138

variable {x y : ℝ}

theorem correct_operation :
  (2 * x^2 + 4 * x^2 = 6 * x^2) → 
  (x * x^3 = x^4) → 
  ((x^3)^2 = x^6) →
  ((xy)^5 = x^5 * y^5) →
  ((x^3)^2 = x^6) := 
by 
  intros h1 h2 h3 h4
  exact h3

end NUMINAMATH_GPT_correct_operation_l431_43138


namespace NUMINAMATH_GPT_books_on_shelf_l431_43178

theorem books_on_shelf (total_books : ℕ) (sold_books : ℕ) (shelves : ℕ) (remaining_books : ℕ) (books_per_shelf : ℕ) :
  total_books = 27 → sold_books = 6 → shelves = 3 → remaining_books = total_books - sold_books → books_per_shelf = remaining_books / shelves → books_per_shelf = 7 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_books_on_shelf_l431_43178


namespace NUMINAMATH_GPT_period_six_l431_43107

variable {R : Type} [LinearOrderedField R]

def symmetric1 (f : R → R) : Prop := ∀ x : R, f (2 + x) = f (2 - x)
def symmetric2 (f : R → R) : Prop := ∀ x : R, f (5 + x) = f (5 - x)

theorem period_six (f : R → R) (h1 : symmetric1 f) (h2 : symmetric2 f) : ∀ x : R, f (x + 6) = f x :=
sorry

end NUMINAMATH_GPT_period_six_l431_43107


namespace NUMINAMATH_GPT_equality_of_areas_l431_43172

theorem equality_of_areas (d : ℝ) :
  (∀ d : ℝ, (1/2) * d * 3 = 9 / 2 → d = 3) ↔ d = 3 :=
by
  sorry

end NUMINAMATH_GPT_equality_of_areas_l431_43172


namespace NUMINAMATH_GPT_injective_g_restricted_to_interval_l431_43181

def g (x : ℝ) : ℝ := (x + 3) ^ 2 - 10

theorem injective_g_restricted_to_interval :
  (∀ x1 x2 : ℝ, x1 ∈ Set.Ici (-3) → x2 ∈ Set.Ici (-3) → g x1 = g x2 → x1 = x2) :=
sorry

end NUMINAMATH_GPT_injective_g_restricted_to_interval_l431_43181


namespace NUMINAMATH_GPT_part1_part2_l431_43115

variable (a b c x : ℝ)

-- Condition: lengths of the sides of the triangle
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

-- Quadratic equation
def quadratic_eq (x : ℝ) : ℝ := (a + c) * x^2 - 2 * b * x + (a - c)

-- Proof problem 1: If x = 1 is a root, then triangle ABC is isosceles
theorem part1 (h : quadratic_eq a b c 1 = 0) : a = b :=
by
  sorry

-- Proof problem 2: If triangle ABC is equilateral, then roots of the quadratic equation are 0 and 1
theorem part2 (h_eq : a = b ∧ b = c) :
  (quadratic_eq a a a 0 = 0) ∧ (quadratic_eq a a a 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l431_43115


namespace NUMINAMATH_GPT_lcm_condition_proof_l431_43180

theorem lcm_condition_proof (n : ℕ) (a : ℕ → ℕ)
  (h1 : ∀ i, 1 ≤ i → i ≤ n → 0 < a i)
  (h2 : ∀ i j, 1 ≤ i → i < j → j ≤ n → a i < a j)
  (h3 : ∀ i, 1 ≤ i → i ≤ n → a i ≤ 2 * n)
  (h4 : ∀ i j, 1 ≤ i → i ≤ n → 1 ≤ j → j ≤ n → i ≠ j → Nat.lcm (a i) (a j) > 2 * n) :
  a 1 > n * 2 / 3 := 
sorry

end NUMINAMATH_GPT_lcm_condition_proof_l431_43180


namespace NUMINAMATH_GPT_prove_N_value_l431_43114

theorem prove_N_value (x y N : ℝ) 
  (h1 : N = 4 * x + y) 
  (h2 : 3 * x - 4 * y = 5) 
  (h3 : 7 * x - 3 * y = 23) : 
  N = 86 / 3 := by
  sorry

end NUMINAMATH_GPT_prove_N_value_l431_43114


namespace NUMINAMATH_GPT_find_t_l431_43169

variables (V V₀ g a S t : ℝ)

-- Conditions
axiom eq1 : V = 3 * g * t + V₀
axiom eq2 : S = (3 / 2) * g * t^2 + V₀ * t + (1 / 2) * a * t^2

-- Theorem to prove
theorem find_t : t = (9 * g * S) / (2 * (V - V₀)^2 + 3 * V₀ * (V - V₀)) :=
by
  sorry

end NUMINAMATH_GPT_find_t_l431_43169


namespace NUMINAMATH_GPT_grassy_plot_width_l431_43119

noncomputable def gravel_cost (L w p : ℝ) : ℝ :=
  0.80 * ((L + 2 * p) * (w + 2 * p) - L * w)

theorem grassy_plot_width
  (L : ℝ) 
  (p : ℝ) 
  (cost : ℝ) 
  (hL : L = 110) 
  (hp : p = 2.5) 
  (hcost : cost = 680) :
  ∃ w : ℝ, gravel_cost L w p = cost ∧ w = 97.5 :=
by
  sorry

end NUMINAMATH_GPT_grassy_plot_width_l431_43119


namespace NUMINAMATH_GPT_pencils_in_each_box_l431_43111

open Nat

theorem pencils_in_each_box (boxes pencils_given_to_Lauren pencils_left pencils_each_box more_pencils : ℕ)
  (h1 : boxes = 2)
  (h2 : pencils_given_to_Lauren = 6)
  (h3 : pencils_left = 9)
  (h4 : more_pencils = 3)
  (h5 : pencils_given_to_Matt = pencils_given_to_Lauren + more_pencils)
  (h6 : pencils_each_box = (pencils_given_to_Lauren + pencils_given_to_Matt + pencils_left) / boxes) :
  pencils_each_box = 12 := by
  sorry

end NUMINAMATH_GPT_pencils_in_each_box_l431_43111


namespace NUMINAMATH_GPT_sin_fourth_plus_cos_fourth_l431_43155

theorem sin_fourth_plus_cos_fourth (α : ℝ) (h : Real.cos (2 * α) = 3 / 5) : 
  Real.sin α ^ 4 + Real.cos α ^ 4 = 17 / 25 := 
by
  sorry

end NUMINAMATH_GPT_sin_fourth_plus_cos_fourth_l431_43155


namespace NUMINAMATH_GPT_woman_weaves_amount_on_20th_day_l431_43190

theorem woman_weaves_amount_on_20th_day
  (a d : ℚ)
  (a2 : a + d = 17) -- second-day weaving in inches
  (S15 : 15 * a + 105 * d = 720) -- total for the first 15 days in inches
  : a + 19 * d = 108 := -- weaving on the twentieth day in inches (9 feet)
by
  sorry

end NUMINAMATH_GPT_woman_weaves_amount_on_20th_day_l431_43190


namespace NUMINAMATH_GPT_inequality_abc_l431_43103

theorem inequality_abc (a b c : ℝ) (h1 : a ∈ Set.Icc (-1 : ℝ) 2) (h2 : b ∈ Set.Icc (-1 : ℝ) 2) (h3 : c ∈ Set.Icc (-1 : ℝ) 2) : 
  a * b * c + 4 ≥ a * b + b * c + c * a := 
sorry

end NUMINAMATH_GPT_inequality_abc_l431_43103


namespace NUMINAMATH_GPT_abs_eq_case_l431_43110

theorem abs_eq_case (x : ℝ) (h : |x - 3| = |x + 2|) : x = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_abs_eq_case_l431_43110


namespace NUMINAMATH_GPT_greatest_value_of_sum_l431_43130

theorem greatest_value_of_sum (x : ℝ) (h : 13 = x^2 + (1/x)^2) : x + 1/x ≤ Real.sqrt 15 :=
sorry

end NUMINAMATH_GPT_greatest_value_of_sum_l431_43130


namespace NUMINAMATH_GPT_wedge_volume_calculation_l431_43105

theorem wedge_volume_calculation :
  let r := 5 
  let h := 8 
  let V := (1 / 3) * (Real.pi * r^2 * h) 
  V = (200 * Real.pi) / 3 :=
by
  let r := 5
  let h := 8
  let V := (1 / 3) * (Real.pi * r^2 * h)
  -- Prove the equality step is omitted as per the prompt
  sorry

end NUMINAMATH_GPT_wedge_volume_calculation_l431_43105


namespace NUMINAMATH_GPT_tensor_identity_l431_43113

def tensor (a b : ℝ) : ℝ := a^3 - b

theorem tensor_identity (a : ℝ) : tensor a (tensor a (tensor a a)) = a^3 - a :=
by
  sorry

end NUMINAMATH_GPT_tensor_identity_l431_43113


namespace NUMINAMATH_GPT_sum_of_common_ratios_l431_43164

variable {k p r : ℝ}

theorem sum_of_common_ratios (h1 : k ≠ 0)
                             (h2 : p ≠ r)
                             (h3 : k * p^2 - k * r^2 = 5 * (k * p - k * r)) :
                             p + r = 5 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_common_ratios_l431_43164


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l431_43185

theorem eccentricity_of_ellipse (m n : ℝ) (h1 : 1 / m + 2 / n = 1) (h2 : 0 < m) (h3 : 0 < n) (h4 : m * n = 8) :
  let a := n
  let b := m
  let c := Real.sqrt (a^2 - b^2)
  let e := c / a
  e = Real.sqrt 3 / 2 := 
sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l431_43185


namespace NUMINAMATH_GPT_initial_pounds_of_coffee_l431_43186

variable (x : ℝ) (h1 : 0.25 * x = d₀) (h2 : 0.60 * 100 = d₁) 
          (h3 : (d₀ + d₁) / (x + 100) = 0.32)

theorem initial_pounds_of_coffee (d₀ d₁ : ℝ) : 
  x = 400 :=
by
  -- Given conditions
  have h1 : d₀ = 0.25 * x := sorry
  have h2 : d₁ = 0.60 * 100 := sorry
  have h3 : 0.32 = (d₀ + d₁) / (x + 100) := sorry
  
  -- Additional steps to solve for x
  sorry

end NUMINAMATH_GPT_initial_pounds_of_coffee_l431_43186


namespace NUMINAMATH_GPT_parabola_focus_directrix_distance_l431_43137

theorem parabola_focus_directrix_distance 
  (p : ℝ) 
  (hp : 3 = p * (1:ℝ)^2) 
  (hparabola : ∀ x : ℝ, y = p * x^2 → x^2 = (1/3:ℝ) * y)
  : (distance_focus_directrix : ℝ) = (1 / 6:ℝ) :=
  sorry

end NUMINAMATH_GPT_parabola_focus_directrix_distance_l431_43137


namespace NUMINAMATH_GPT_lucy_snowballs_eq_19_l431_43112

-- Define the conditions
def charlie_snowballs : ℕ := 50
def difference_charlie_lucy : ℕ := 31

-- Define what we want to prove, i.e., Lucy has 19 snowballs
theorem lucy_snowballs_eq_19 : (charlie_snowballs - difference_charlie_lucy = 19) :=
by
  -- We would provide the proof here, but it's not required for this prompt
  sorry

end NUMINAMATH_GPT_lucy_snowballs_eq_19_l431_43112


namespace NUMINAMATH_GPT_odd_function_evaluation_l431_43125

theorem odd_function_evaluation
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_def : ∀ x, x ≤ 0 → f x = 2 * x^2 - x) :
  f 1 = -3 :=
by {
  sorry
}

end NUMINAMATH_GPT_odd_function_evaluation_l431_43125
