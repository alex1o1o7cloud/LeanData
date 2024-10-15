import Mathlib

namespace NUMINAMATH_GPT_simplify_expr_l1650_165084

theorem simplify_expr : 3 * (4 - 2 * Complex.I) - 2 * Complex.I * (3 - 2 * Complex.I) = 8 - 12 * Complex.I :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr_l1650_165084


namespace NUMINAMATH_GPT_number_of_penny_piles_l1650_165049

theorem number_of_penny_piles
    (piles_of_quarters : ℕ := 4) 
    (piles_of_dimes : ℕ := 6)
    (piles_of_nickels : ℕ := 9)
    (total_value_in_dollars : ℝ := 21)
    (coins_per_pile : ℕ := 10)
    (quarter_value : ℝ := 0.25)
    (dime_value : ℝ := 0.10)
    (nickel_value : ℝ := 0.05)
    (penny_value : ℝ := 0.01) :
    (total_value_in_dollars - ((piles_of_quarters * coins_per_pile * quarter_value) +
                               (piles_of_dimes * coins_per_pile * dime_value) +
                               (piles_of_nickels * coins_per_pile * nickel_value))) /
                               (coins_per_pile * penny_value) = 5 := 
by
  sorry

end NUMINAMATH_GPT_number_of_penny_piles_l1650_165049


namespace NUMINAMATH_GPT_dartboard_odd_sum_probability_l1650_165098

theorem dartboard_odd_sum_probability :
  let innerR := 4
  let outerR := 8
  let inner_points := [3, 1, 1]
  let outer_points := [2, 3, 3]
  let total_area := π * outerR^2
  let inner_area := π * innerR^2
  let outer_area := total_area - inner_area
  let each_inner_area := inner_area / 3
  let each_outer_area := outer_area / 3
  let odd_area := 2 * each_inner_area + 2 * each_outer_area
  let even_area := each_inner_area + each_outer_area
  let P_odd := odd_area / total_area
  let P_even := even_area / total_area
  let odd_sum_prob := 2 * (P_odd * P_even)
  odd_sum_prob = 4 / 9 := by
    sorry

end NUMINAMATH_GPT_dartboard_odd_sum_probability_l1650_165098


namespace NUMINAMATH_GPT_division_example_l1650_165075

theorem division_example :
  100 / 0.25 = 400 :=
by sorry

end NUMINAMATH_GPT_division_example_l1650_165075


namespace NUMINAMATH_GPT_find_second_group_of_men_l1650_165027

noncomputable def work_rate_of_man := ℝ
noncomputable def work_rate_of_woman := ℝ

variables (m w : ℝ)

-- Condition 1: 3 men and 8 women complete the task in the same time as x men and 2 women.
axiom condition1 (x : ℝ) : 3 * m + 8 * w = x * m + 2 * w

-- Condition 2: 2 men and 3 women complete half the task in the same time as 3 men and 8 women completing the whole task.
axiom condition2 : 2 * m + 3 * w = 0.5 * (3 * m + 8 * w)

theorem find_second_group_of_men (x : ℝ) (m w : ℝ) (h1 : 0.5 * m = w)
  (h2 : 3 * m + 8 * w = x * m + 2 * w) : x = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_second_group_of_men_l1650_165027


namespace NUMINAMATH_GPT_sqrt_t6_plus_t4_eq_t2_sqrt_t2_plus_1_l1650_165019

theorem sqrt_t6_plus_t4_eq_t2_sqrt_t2_plus_1 (t : ℝ) : 
  Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1) :=
sorry

end NUMINAMATH_GPT_sqrt_t6_plus_t4_eq_t2_sqrt_t2_plus_1_l1650_165019


namespace NUMINAMATH_GPT_arithmetic_progression_common_difference_l1650_165081

theorem arithmetic_progression_common_difference :
  ∀ (A1 An n d : ℕ), A1 = 3 → An = 103 → n = 21 → An = A1 + (n - 1) * d → d = 5 :=
by
  intros A1 An n d h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_arithmetic_progression_common_difference_l1650_165081


namespace NUMINAMATH_GPT_no_integer_solutions_l1650_165024

theorem no_integer_solutions :
   ¬ ∃ (x y z : ℤ), x^4 + y^4 + z^4 = 2 * x^2 * y^2 + 2 * y^2 * z^2 + 2 * z^2 * x^2 + 24 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l1650_165024


namespace NUMINAMATH_GPT_rooms_equation_l1650_165058

theorem rooms_equation (x : ℕ) (h₁ : ∃ n, n = 6 * (x - 1)) (h₂ : ∃ m, m = 5 * x + 4) :
  6 * (x - 1) = 5 * x + 4 :=
sorry

end NUMINAMATH_GPT_rooms_equation_l1650_165058


namespace NUMINAMATH_GPT_hcf_of_two_numbers_l1650_165046

theorem hcf_of_two_numbers 
  (x y : ℕ) 
  (h1 : x + y = 45)
  (h2 : Nat.lcm x y = 120)
  (h3 : (1/x : ℚ) + (1/y : ℚ) = 11/120) : 
  Nat.gcd x y = 1 := 
sorry

end NUMINAMATH_GPT_hcf_of_two_numbers_l1650_165046


namespace NUMINAMATH_GPT_pascal_28_25_eq_2925_l1650_165009

-- Define the Pascal's triangle nth-row function
def pascal (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the theorem to prove that the 25th element in the 28 element row is 2925
theorem pascal_28_25_eq_2925 :
  pascal 27 24 = 2925 :=
by
  sorry

end NUMINAMATH_GPT_pascal_28_25_eq_2925_l1650_165009


namespace NUMINAMATH_GPT_rachelle_meat_needed_l1650_165005

-- Define the ratio of meat per hamburger
def meat_per_hamburger (pounds : ℕ) (hamburgers : ℕ) : ℚ :=
  pounds / hamburgers

-- Define the total meat needed for a given number of hamburgers
def total_meat (meat_per_hamburger : ℚ) (hamburgers : ℕ) : ℚ :=
  meat_per_hamburger * hamburgers

-- Prove that Rachelle needs 15 pounds of meat to make 36 hamburgers
theorem rachelle_meat_needed : total_meat (meat_per_hamburger 5 12) 36 = 15 := by
  sorry

end NUMINAMATH_GPT_rachelle_meat_needed_l1650_165005


namespace NUMINAMATH_GPT_shifted_function_expression_l1650_165022

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.sin (ω * x + Real.pi / 3)

theorem shifted_function_expression (ω : ℝ) (h : ℝ) (x : ℝ) (h_positive : ω > 0) (h_period : Real.pi = 2 * Real.pi / ω) :
  f ω (x + h) = Real.cos (2 * x) :=
by
  -- We assume h = π/12, ω = 2
  have ω_val : ω = 2 := by sorry
  have h_val : h = Real.pi / 12 := by sorry
  rw [ω_val, h_val]
  sorry

end NUMINAMATH_GPT_shifted_function_expression_l1650_165022


namespace NUMINAMATH_GPT_calculate_expression_l1650_165000

theorem calculate_expression :
  ((12 ^ 12 / 12 ^ 11) ^ 2 * 4 ^ 2) / 2 ^ 4 = 144 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1650_165000


namespace NUMINAMATH_GPT_find_x_l1650_165073

theorem find_x (x y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1650_165073


namespace NUMINAMATH_GPT_polynomial_in_y_l1650_165043

theorem polynomial_in_y {x y : ℝ} (h₁ : x^3 - 6 * x^2 + 11 * x - 6 = 0) (h₂ : y = x + 1/x) :
  x^2 * (y^2 + y - 6) = 0 :=
sorry

end NUMINAMATH_GPT_polynomial_in_y_l1650_165043


namespace NUMINAMATH_GPT_infinite_rational_solutions_x3_y3_9_l1650_165069

theorem infinite_rational_solutions_x3_y3_9 :
  ∃ (S : Set (ℚ × ℚ)), S.Infinite ∧ (∀ (x y : ℚ), (x, y) ∈ S → x^3 + y^3 = 9) :=
sorry

end NUMINAMATH_GPT_infinite_rational_solutions_x3_y3_9_l1650_165069


namespace NUMINAMATH_GPT_jason_cutting_hours_l1650_165031

-- Definitions derived from conditions
def time_to_cut_one_lawn : ℕ := 30  -- minutes
def lawns_per_day := 8 -- number of lawns Jason cuts each day
def days := 2 -- number of days (Saturday and Sunday)
def minutes_in_an_hour := 60 -- conversion factor from minutes to hours

-- The proof problem
theorem jason_cutting_hours : 
  (time_to_cut_one_lawn * lawns_per_day * days) / minutes_in_an_hour = 8 := sorry

end NUMINAMATH_GPT_jason_cutting_hours_l1650_165031


namespace NUMINAMATH_GPT_math_problem_solution_l1650_165079

open Real

noncomputable def math_problem (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : a + b + c + d = 4) : Prop :=
  (b / sqrt (a + 2 * c) + c / sqrt (b + 2 * d) + d / sqrt (c + 2 * a) + a / sqrt (d + 2 * b)) ≥ (4 * sqrt 3) / 3

theorem math_problem_solution (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a + b + c + d = 4) :
  math_problem a b c d ha hb hc hd h := by sorry

end NUMINAMATH_GPT_math_problem_solution_l1650_165079


namespace NUMINAMATH_GPT_conference_min_duration_l1650_165039

theorem conference_min_duration : Nat.gcd 9 11 = 1 ∧ Nat.gcd 9 12 = 3 ∧ Nat.gcd 11 12 = 1 ∧ Nat.lcm 9 (Nat.lcm 11 12) = 396 := by
  sorry

end NUMINAMATH_GPT_conference_min_duration_l1650_165039


namespace NUMINAMATH_GPT_consecutive_integers_divisor_l1650_165016

theorem consecutive_integers_divisor {m n : ℕ} (hm : m < n) (a : ℕ) :
  ∃ i j : ℕ, i ≠ j ∧ (a + i) * (a + j) % (m * n) = 0 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_integers_divisor_l1650_165016


namespace NUMINAMATH_GPT_ferris_wheel_rides_l1650_165062

theorem ferris_wheel_rides :
  let people_per_20_minutes := 70
  let operation_duration_hours := 6
  let minutes_per_hour := 60
  let operation_duration_minutes := operation_duration_hours * minutes_per_hour
  let times_per_hour := minutes_per_hour / 20
  let total_people_per_hour := times_per_hour * people_per_20_minutes
  let total_people := total_people_per_hour * operation_duration_hours
  total_people = 1260 :=
by
  let people_per_20_minutes := 70
  let operation_duration_hours := 6
  let minutes_per_hour := 60
  let operation_duration_minutes := operation_duration_hours * minutes_per_hour
  let times_per_hour := minutes_per_hour / 20
  let total_people_per_hour := times_per_hour * people_per_20_minutes
  let total_people := total_people_per_hour * operation_duration_hours
  have : total_people = 1260 := by sorry
  exact this

end NUMINAMATH_GPT_ferris_wheel_rides_l1650_165062


namespace NUMINAMATH_GPT_real_roots_of_system_l1650_165090

theorem real_roots_of_system :
  { (x, y) : ℝ × ℝ | (x + y)^4 = 6 * x^2 * y^2 - 215 ∧ x * y * (x^2 + y^2) = -78 } =
  { (3, -2), (-2, 3), (-3, 2), (2, -3) } :=
by 
  sorry

end NUMINAMATH_GPT_real_roots_of_system_l1650_165090


namespace NUMINAMATH_GPT_solution_set_inequality_k_l1650_165050

theorem solution_set_inequality_k (k : ℚ) :
  (∀ x : ℚ, 3 * x - (2 * k - 3) < 4 * x + 3 * k + 6 ↔ x > 1) → k = -4/5 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_k_l1650_165050


namespace NUMINAMATH_GPT_ratio_of_triangle_to_square_l1650_165033

theorem ratio_of_triangle_to_square (s : ℝ) (hs : 0 < s) :
  let A_square := s^2
  let A_triangle := (1/2) * s * (s/2)
  A_triangle / A_square = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_triangle_to_square_l1650_165033


namespace NUMINAMATH_GPT_fractions_not_equal_to_seven_over_five_l1650_165037

theorem fractions_not_equal_to_seven_over_five :
  (7 / 5 ≠ 1 + (4 / 20)) ∧ (7 / 5 ≠ 1 + (3 / 15)) ∧ (7 / 5 ≠ 1 + (2 / 6)) :=
by
  sorry

end NUMINAMATH_GPT_fractions_not_equal_to_seven_over_five_l1650_165037


namespace NUMINAMATH_GPT_inequality_iff_l1650_165083

theorem inequality_iff (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : (a > b) ↔ (1/a < 1/b) = false :=
by
  sorry

end NUMINAMATH_GPT_inequality_iff_l1650_165083


namespace NUMINAMATH_GPT_trees_after_planting_l1650_165080

variable (x : ℕ)

theorem trees_after_planting (x : ℕ) : 
  let additional_trees := 12
  let days := 12 / 2
  let trees_removed := days * 3
  x + additional_trees - trees_removed = x - 6 :=
by
  let additional_trees := 12
  let days := 12 / 2
  let trees_removed := days * 3
  sorry

end NUMINAMATH_GPT_trees_after_planting_l1650_165080


namespace NUMINAMATH_GPT_geom_series_correct_sum_l1650_165012

-- Define the geometric series sum
noncomputable def geom_series_sum (a r : ℚ) (n : ℕ) :=
  a * (1 - r ^ n) / (1 - r)

-- Given conditions
def a := (1 : ℚ) / 4
def r := (1 : ℚ) / 4
def n := 8

-- Correct answer sum
def correct_sum := (65535 : ℚ) / 196608

-- Proof problem statement
theorem geom_series_correct_sum : geom_series_sum a r n = correct_sum := 
  sorry

end NUMINAMATH_GPT_geom_series_correct_sum_l1650_165012


namespace NUMINAMATH_GPT_day_crew_fraction_l1650_165089

theorem day_crew_fraction (D W : ℝ) (h1 : D > 0) (h2 : W > 0) :
  (D * W / (D * W + (3 / 4 * D * 1 / 2 * W)) = 8 / 11) :=
by
  sorry

end NUMINAMATH_GPT_day_crew_fraction_l1650_165089


namespace NUMINAMATH_GPT_gcd_55555555_111111111_l1650_165021

/-- Let \( m = 55555555 \) and \( n = 111111111 \).
We want to prove that the greatest common divisor (gcd) of \( m \) and \( n \) is 1. -/
theorem gcd_55555555_111111111 :
  let m := 55555555
  let n := 111111111
  Nat.gcd m n = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_55555555_111111111_l1650_165021


namespace NUMINAMATH_GPT_original_price_of_article_l1650_165061

theorem original_price_of_article
  (P S : ℝ) 
  (h1 : S = 1.4 * P) 
  (h2 : S - P = 560) 
  : P = 1400 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_article_l1650_165061


namespace NUMINAMATH_GPT_percentage_spent_on_household_items_l1650_165074

theorem percentage_spent_on_household_items (monthly_income : ℝ) (savings : ℝ) (clothes_percentage : ℝ) (medicines_percentage : ℝ) (household_spent : ℝ) : 
  monthly_income = 40000 ∧ 
  savings = 9000 ∧ 
  clothes_percentage = 0.25 ∧ 
  medicines_percentage = 0.075 ∧ 
  household_spent = monthly_income - (clothes_percentage * monthly_income + medicines_percentage * monthly_income + savings)
  → (household_spent / monthly_income) * 100 = 45 :=
by
  intro h
  cases' h with h1 h_rest
  cases' h_rest with h2 h_rest
  cases' h_rest with h3 h_rest
  cases' h_rest with h4 h5
  have h_clothes := h3
  have h_medicines := h4
  have h_savings := h2
  have h_income := h1
  have h_household := h5
  sorry

end NUMINAMATH_GPT_percentage_spent_on_household_items_l1650_165074


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1650_165059

open Set

variable {α : Type} [PartialOrder α]

noncomputable def A := { x : ℝ | -1 < x ∧ x < 1 }
noncomputable def B := { x : ℝ | 0 < x }

theorem intersection_of_A_and_B :
  A ∩ B = { x : ℝ | 0 < x ∧ x < 1 } :=
by sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1650_165059


namespace NUMINAMATH_GPT_primes_divisible_by_3_percentage_is_12_5_l1650_165018

-- Definition of the primes less than 20
def primes_less_than_20 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

-- Definition of the prime numbers from the list that are divisible by 3
def primes_divisible_by_3 : List Nat := primes_less_than_20.filter (λ p => p % 3 = 0)

-- Total number of primes less than 20
def total_primes_less_than_20 : Nat := primes_less_than_20.length

-- Total number of primes less than 20 that are divisible by 3
def total_primes_divisible_by_3 : Nat := primes_divisible_by_3.length

-- The percentage of prime numbers less than 20 that are divisible by 3
noncomputable def percentage_primes_divisible_by_3 : Float := 
  (total_primes_divisible_by_3.toFloat / total_primes_less_than_20.toFloat) * 100

theorem primes_divisible_by_3_percentage_is_12_5 :
  percentage_primes_divisible_by_3 = 12.5 := by
  sorry

end NUMINAMATH_GPT_primes_divisible_by_3_percentage_is_12_5_l1650_165018


namespace NUMINAMATH_GPT_complement_intersection_l1650_165086

open Set -- Open namespace for set operations

-- Define the universal set I
def I : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set ℕ := {1, 2, 3, 4}

-- Define set B
def B : Set ℕ := {3, 4, 5, 6}

-- Define the intersection A ∩ B
def A_inter_B : Set ℕ := A ∩ B

-- Define the complement C_I(S) as I \ S, where S is a subset of I
def complement (S : Set ℕ) : Set ℕ := I \ S

-- Prove that the complement of A ∩ B in I is {1, 2, 5, 6}
theorem complement_intersection : complement A_inter_B = {1, 2, 5, 6} :=
by
  sorry -- Proof to be provided

end NUMINAMATH_GPT_complement_intersection_l1650_165086


namespace NUMINAMATH_GPT_min_value_expression_l1650_165092

open Real

theorem min_value_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 7)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 36 := by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1650_165092


namespace NUMINAMATH_GPT_probability_two_draws_l1650_165091

def probability_first_red_second_kd (total_cards : ℕ) (red_cards : ℕ) (king_of_diamonds : ℕ) : ℚ :=
  (red_cards / total_cards) * (king_of_diamonds / (total_cards - 1))

theorem probability_two_draws :
  let total_cards := 52
  let red_cards := 26
  let king_of_diamonds := 1
  probability_first_red_second_kd total_cards red_cards king_of_diamonds = 1 / 102 :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_two_draws_l1650_165091


namespace NUMINAMATH_GPT_minimize_transfers_l1650_165095

-- Define the initial number of pieces in each supermarket
def pieces_in_A := 15
def pieces_in_B := 7
def pieces_in_C := 11
def pieces_in_D := 3
def pieces_in_E := 14

-- Define the target number of pieces in each supermarket after transfers
def target_pieces := 10

-- Define a function to compute the total number of pieces
def total_pieces := pieces_in_A + pieces_in_B + pieces_in_C + pieces_in_D + pieces_in_E

-- Define the minimum number of transfers needed
def min_transfers := 12

-- The main theorem: proving that the minimum number of transfers is 12
theorem minimize_transfers : 
  total_pieces = 5 * target_pieces → 
  ∃ (transfers : ℕ), transfers = min_transfers :=
by
  -- This represents the proof section, we leave it as sorry
  sorry

end NUMINAMATH_GPT_minimize_transfers_l1650_165095


namespace NUMINAMATH_GPT_average_all_results_l1650_165002

theorem average_all_results (s₁ s₂ : ℤ) (n₁ n₂ : ℤ) (h₁ : n₁ = 60) (h₂ : n₂ = 40) (avg₁ : s₁ / n₁ = 40) (avg₂ : s₂ / n₂ = 60) : 
  ((s₁ + s₂) / (n₁ + n₂) = 48) :=
sorry

end NUMINAMATH_GPT_average_all_results_l1650_165002


namespace NUMINAMATH_GPT_arthur_spent_on_second_day_l1650_165085

variable (H D : ℝ)
variable (a1 : 3 * H + 4 * D = 10)
variable (a2 : D = 1)

theorem arthur_spent_on_second_day :
  2 * H + 3 * D = 7 :=
by
  sorry

end NUMINAMATH_GPT_arthur_spent_on_second_day_l1650_165085


namespace NUMINAMATH_GPT_directrix_of_parabola_l1650_165036

theorem directrix_of_parabola :
  (∀ x y : ℝ, x^2 = 4 * y → ∃ y₀ : ℝ, y₀ = -1 ∧ ∀ y' : ℝ, y' = y₀) :=
by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l1650_165036


namespace NUMINAMATH_GPT_root_equation_l1650_165099

variable (m : ℝ)
theorem root_equation (h : m^2 - 2 * m - 3 = 0) : m^2 - 2 * m + 2020 = 2023 := by
  sorry

end NUMINAMATH_GPT_root_equation_l1650_165099


namespace NUMINAMATH_GPT_total_points_l1650_165097

theorem total_points (darius_score marius_score matt_score total_points : ℕ) 
    (h1 : darius_score = 10) 
    (h2 : marius_score = darius_score + 3) 
    (h3 : matt_score = darius_score + 5) 
    (h4 : total_points = darius_score + marius_score + matt_score) : 
    total_points = 38 :=
by sorry

end NUMINAMATH_GPT_total_points_l1650_165097


namespace NUMINAMATH_GPT_quadratic_inequality_sum_l1650_165094

theorem quadratic_inequality_sum (a b : ℝ) (h1 : 1 < 2) 
 (h2 : ∀ x : ℝ, 1 < x ∧ x < 2 → x^2 - a * x + b < 0) 
 (h3 : 1 + 2 = a)  (h4 : 1 * 2 = b) : 
 a + b = 5 := 
by 
sorry

end NUMINAMATH_GPT_quadratic_inequality_sum_l1650_165094


namespace NUMINAMATH_GPT_six_star_three_l1650_165025

-- Define the mathematical operation.
def operation (r t : ℝ) : ℝ := sorry

axiom condition_1 (r : ℝ) : operation r 0 = r^2
axiom condition_2 (r t : ℝ) : operation r t = operation t r
axiom condition_3 (r t : ℝ) : operation (r + 1) t = operation r t + 2 * t + 1

-- Prove that 6 * 3 = 75 given the conditions.
theorem six_star_three : operation 6 3 = 75 := by
  sorry

end NUMINAMATH_GPT_six_star_three_l1650_165025


namespace NUMINAMATH_GPT_complex_number_in_second_quadrant_l1650_165067

theorem complex_number_in_second_quadrant :
  let z := (2 + 4 * Complex.I) / (1 + Complex.I) 
  ∃ (im : ℂ), z = im ∧ im.re < 0 ∧ 0 < im.im := by
  sorry

end NUMINAMATH_GPT_complex_number_in_second_quadrant_l1650_165067


namespace NUMINAMATH_GPT_find_precy_age_l1650_165011

-- Defining the given conditions as Lean definitions
def alex_current_age : ℕ := 15
def alex_age_in_3_years : ℕ := alex_current_age + 3
def alex_age_a_year_ago : ℕ := alex_current_age - 1
axiom precy_current_age : ℕ
axiom in_3_years : alex_age_in_3_years = 3 * (precy_current_age + 3)
axiom a_year_ago : alex_age_a_year_ago = 7 * (precy_current_age - 1)

-- Stating the equivalent proof problem
theorem find_precy_age : precy_current_age = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_precy_age_l1650_165011


namespace NUMINAMATH_GPT_total_feet_l1650_165044

theorem total_feet (heads hens : ℕ) (h1 : heads = 46) (h2 : hens = 22) : 
  ∃ feet : ℕ, feet = 140 := 
by 
  sorry

end NUMINAMATH_GPT_total_feet_l1650_165044


namespace NUMINAMATH_GPT_inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l1650_165056

theorem inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
sorry

end NUMINAMATH_GPT_inequality_1_le_x3y3_over_x2y2_squared_le_9_over_8_l1650_165056


namespace NUMINAMATH_GPT_least_three_digit_multiple_of_3_4_9_is_108_l1650_165014

theorem least_three_digit_multiple_of_3_4_9_is_108 :
  ∃ (n : ℕ), (100 ≤ n) ∧ (n % 3 = 0) ∧ (n % 4 = 0) ∧ (n % 9 = 0) ∧ (n = 108) :=
by
  sorry

end NUMINAMATH_GPT_least_three_digit_multiple_of_3_4_9_is_108_l1650_165014


namespace NUMINAMATH_GPT_Rebecca_group_count_l1650_165034

def groupEggs (total_eggs number_of_eggs_per_group total_groups : Nat) : Prop :=
  total_groups = total_eggs / number_of_eggs_per_group

theorem Rebecca_group_count :
  groupEggs 8 2 4 :=
by
  sorry

end NUMINAMATH_GPT_Rebecca_group_count_l1650_165034


namespace NUMINAMATH_GPT_deformable_to_triangle_l1650_165038

-- We define a planar polygon with n rods connected by hinges
structure PlanarPolygon (n : ℕ) :=
  (rods : Fin n → ℝ)
  (connections : Fin n → Fin n → Prop)

-- Define the conditions for the rods being rigid and connections (hinges)
def rigid_rod (n : ℕ) : PlanarPolygon n → Prop := λ poly => 
  ∀ i j, poly.connections i j → poly.rods i = poly.rods j

-- Defining the theorem for deformation into a triangle
theorem deformable_to_triangle (n : ℕ) (p : PlanarPolygon n) : 
  (n > 4) ↔ ∃ q : PlanarPolygon 3, true :=
by
  sorry

end NUMINAMATH_GPT_deformable_to_triangle_l1650_165038


namespace NUMINAMATH_GPT_shaded_triangle_area_l1650_165015

theorem shaded_triangle_area (b h : ℝ) (hb : b = 2) (hh : h = 3) : 
  (1 / 2 * b * h) = 3 := 
by
  rw [hb, hh]
  norm_num

end NUMINAMATH_GPT_shaded_triangle_area_l1650_165015


namespace NUMINAMATH_GPT_problem1_problem2_l1650_165026

theorem problem1 (n : ℕ) (hn : 0 < n) : (3^(2*n+1) + 2^(n+2)) % 7 = 0 := 
sorry

theorem problem2 (n : ℕ) (hn : 0 < n) : (3^(2*n+2) + 2^(6*n+1)) % 11 = 0 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l1650_165026


namespace NUMINAMATH_GPT_vertical_coordinate_intersection_l1650_165054

def original_function (x : ℝ) := x^2 + 2 * x + 1

def shifted_function (x : ℝ) := (x + 3)^2 + 3

theorem vertical_coordinate_intersection :
  shifted_function 0 = 12 :=
by
  sorry

end NUMINAMATH_GPT_vertical_coordinate_intersection_l1650_165054


namespace NUMINAMATH_GPT_objects_meeting_time_l1650_165042

theorem objects_meeting_time 
  (initial_velocity : ℝ) (g : ℝ) (t_delay : ℕ) (t_meet : ℝ) 
  (hv : initial_velocity = 120) 
  (hg : g = 9.8) 
  (ht : t_delay = 5)
  : t_meet = 14.74 :=
sorry

end NUMINAMATH_GPT_objects_meeting_time_l1650_165042


namespace NUMINAMATH_GPT_reciprocal_pair_c_l1650_165063

def is_reciprocal (a b : ℝ) : Prop :=
  a * b = 1

theorem reciprocal_pair_c :
  is_reciprocal (-2) (-1/2) :=
by sorry

end NUMINAMATH_GPT_reciprocal_pair_c_l1650_165063


namespace NUMINAMATH_GPT_length_of_bridge_l1650_165045

theorem length_of_bridge (L_train : ℕ) (v_km_hr : ℕ) (t : ℕ) 
  (h_L_train : L_train = 150)
  (h_v_km_hr : v_km_hr = 45)
  (h_t : t = 30) : 
  ∃ L_bridge : ℕ, L_bridge = 225 :=
by 
  sorry

end NUMINAMATH_GPT_length_of_bridge_l1650_165045


namespace NUMINAMATH_GPT_part1_part2_l1650_165003

variable (a b : ℝ)

-- Conditions
axiom abs_a_eq_4 : |a| = 4
axiom abs_b_eq_6 : |b| = 6

-- Part 1: If ab > 0, find the value of a - b
theorem part1 (h : a * b > 0) : a - b = 2 ∨ a - b = -2 := 
by
  -- Proof will go here
  sorry

-- Part 2: If |a + b| = -(a + b), find the value of a + b
theorem part2 (h : |a + b| = -(a + b)) : a + b = -10 ∨ a + b = -2 := 
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_part1_part2_l1650_165003


namespace NUMINAMATH_GPT_probability_mass_range_l1650_165066

/-- Let ξ be a random variable representing the mass of a badminton product. 
    Suppose P(ξ < 4.8) = 0.3 and P(ξ ≥ 4.85) = 0.32. 
    We want to prove that the probability that the mass is in the range [4.8, 4.85) is 0.38. -/
theorem probability_mass_range (P : ℝ → ℝ) (h1 : P (4.8) = 0.3) (h2 : P (4.85) = 0.32) :
  P (4.8) - P (4.85) = 0.38 :=
by 
  sorry

end NUMINAMATH_GPT_probability_mass_range_l1650_165066


namespace NUMINAMATH_GPT_Math_Proof_Problem_l1650_165060

noncomputable def problem : ℝ := (1005^3) / (1003 * 1004) - (1003^3) / (1004 * 1005)

theorem Math_Proof_Problem : ⌊ problem ⌋ = 8 :=
by
  sorry

end NUMINAMATH_GPT_Math_Proof_Problem_l1650_165060


namespace NUMINAMATH_GPT_exponential_ordering_l1650_165055

noncomputable def a := (0.4:ℝ)^(0.3:ℝ)
noncomputable def b := (0.3:ℝ)^(0.4:ℝ)
noncomputable def c := (0.3:ℝ)^(-0.2:ℝ)

theorem exponential_ordering : b < a ∧ a < c := by
  sorry

end NUMINAMATH_GPT_exponential_ordering_l1650_165055


namespace NUMINAMATH_GPT_quadratic_function_points_l1650_165051

theorem quadratic_function_points:
  (∀ x y, (y = x^2 + x - 1) → ((x = -2 → y = 1) ∧ (x = 0 → y = -1) ∧ (x = 2 → y = 5))) →
  (-1 < 1 ∧ 1 < 5) :=
by
  intro h
  have h1 := h (-2) 1 (by ring)
  have h2 := h 0 (-1) (by ring)
  have h3 := h 2 5 (by ring)
  exact And.intro (by linarith) (by linarith)

end NUMINAMATH_GPT_quadratic_function_points_l1650_165051


namespace NUMINAMATH_GPT_f_2017_equals_neg_one_fourth_l1650_165093

noncomputable def f : ℝ → ℝ := sorry -- Original definition will be derived from the conditions

axiom symmetry_about_y_axis : ∀ (x : ℝ), f (-x) = f x
axiom periodicity : ∀ (x : ℝ), f (x + 3) = -f x
axiom specific_interval : ∀ (x : ℝ), (3/2 < x ∧ x < 5/2) → f x = (1/2)^x

theorem f_2017_equals_neg_one_fourth : f 2017 = -1/4 :=
by sorry

end NUMINAMATH_GPT_f_2017_equals_neg_one_fourth_l1650_165093


namespace NUMINAMATH_GPT_ratio_and_equation_imp_value_of_a_l1650_165001

theorem ratio_and_equation_imp_value_of_a (a b : ℚ) (h1 : b / a = 4) (h2 : b = 20 - 7 * a) :
  a = 20 / 11 :=
by
  sorry

end NUMINAMATH_GPT_ratio_and_equation_imp_value_of_a_l1650_165001


namespace NUMINAMATH_GPT_molecular_weight_of_9_moles_l1650_165028

theorem molecular_weight_of_9_moles (molecular_weight : ℕ) (moles : ℕ) (h₁ : molecular_weight = 1098) (h₂ : moles = 9) :
  molecular_weight * moles = 9882 :=
by {
  sorry
}

end NUMINAMATH_GPT_molecular_weight_of_9_moles_l1650_165028


namespace NUMINAMATH_GPT_find_a2015_l1650_165030

variable (a : ℕ → ℝ)

-- Conditions
axiom h1 : a 1 = 1
axiom h2 : a 2 = 3
axiom h3 : ∀ n : ℕ, n > 0 → a (n + 1) - a n ≤ 2 ^ n
axiom h4 : ∀ n : ℕ, n > 0 → a (n + 2) - a n ≥ 3 * 2 ^ n

-- Theorem stating the solution
theorem find_a2015 : a 2015 = 2 ^ 2015 - 1 :=
by sorry

end NUMINAMATH_GPT_find_a2015_l1650_165030


namespace NUMINAMATH_GPT_find_x_average_is_3_l1650_165047

theorem find_x_average_is_3 (x : ℝ) (h : (2 + 4 + 1 + 3 + x) / 5 = 3) : x = 5 :=
sorry

end NUMINAMATH_GPT_find_x_average_is_3_l1650_165047


namespace NUMINAMATH_GPT_mark_final_buttons_l1650_165006

def mark_initial_buttons : ℕ := 14
def shane_factor : ℚ := 3.5
def lent_to_anna : ℕ := 7
def lost_fraction : ℚ := 0.5
def sam_fraction : ℚ := 2 / 3

theorem mark_final_buttons : 
  let shane_buttons := mark_initial_buttons * shane_factor
  let before_anna := mark_initial_buttons + shane_buttons
  let after_lending_anna := before_anna - lent_to_anna
  let anna_returned := lent_to_anna * (1 - lost_fraction)
  let after_anna_return := after_lending_anna + anna_returned
  let after_sam := after_anna_return - (after_anna_return * sam_fraction)
  round after_sam = 20 := 
by
  sorry

end NUMINAMATH_GPT_mark_final_buttons_l1650_165006


namespace NUMINAMATH_GPT_equilateral_triangle_of_equal_heights_and_inradius_l1650_165010

theorem equilateral_triangle_of_equal_heights_and_inradius 
  {a b c h1 h2 h3 r : ℝ} (h1_eq : h1 = 2 * r * (a * b * c) / a) 
  (h2_eq : h2 = 2 * r * (a * b * c) / b) 
  (h3_eq : h3 = 2 * r * (a * b * c) / c) 
  (sum_heights_eq : h1 + h2 + h3 = 9 * r) : a = b ∧ b = c ∧ c = a :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_of_equal_heights_and_inradius_l1650_165010


namespace NUMINAMATH_GPT_seq_solution_l1650_165023

-- Definitions: Define the sequence {a_n} according to the given conditions
def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧ ∀ n ≥ 2, a n - 2 * a (n - 1) = n ^ 2 - 3

-- Main statement: Prove that for all n, the sequence satisfies the derived formula
theorem seq_solution (a : ℕ → ℤ) (h : seq a) : ∀ n, a n = 2 ^ (n + 2) - n ^ 2 - 4 * n - 3 :=
sorry

end NUMINAMATH_GPT_seq_solution_l1650_165023


namespace NUMINAMATH_GPT_sarah_stamp_collection_value_l1650_165096

theorem sarah_stamp_collection_value :
  ∀ (stamps_owned total_value_for_4_stamps : ℝ) (num_stamps_single_series : ℕ), 
  stamps_owned = 20 → 
  total_value_for_4_stamps = 10 → 
  num_stamps_single_series = 4 → 
  (stamps_owned / num_stamps_single_series) * (total_value_for_4_stamps / num_stamps_single_series) = 50 :=
by
  intros stamps_owned total_value_for_4_stamps num_stamps_single_series 
  intro h_stamps_owned
  intro h_total_value_for_4_stamps
  intro h_num_stamps_single_series
  rw [h_stamps_owned, h_total_value_for_4_stamps, h_num_stamps_single_series]
  sorry

end NUMINAMATH_GPT_sarah_stamp_collection_value_l1650_165096


namespace NUMINAMATH_GPT_oranges_to_pears_l1650_165041

-- Define the equivalence relation between oranges and pears
def equivalent_weight (orange pear : ℕ) : Prop := 4 * pear = 3 * orange

-- Given:
-- 1. 4 oranges weigh the same as 3 pears
-- 2. Jimmy has 36 oranges
-- Prove that 27 pears are required to balance the weight of 36 oranges
theorem oranges_to_pears (orange pear : ℕ) (h : equivalent_weight 1 1) :
  (4 * pear = 3 * orange) → equivalent_weight 36 27 :=
by
  sorry

end NUMINAMATH_GPT_oranges_to_pears_l1650_165041


namespace NUMINAMATH_GPT_supplement_of_angle_with_given_complement_l1650_165032

theorem supplement_of_angle_with_given_complement (θ : ℝ) (h : 90 - θ = 50) : 180 - θ = 140 :=
by sorry

end NUMINAMATH_GPT_supplement_of_angle_with_given_complement_l1650_165032


namespace NUMINAMATH_GPT_hannah_practice_hours_l1650_165004

theorem hannah_practice_hours (weekend_hours : ℕ) (total_weekly_hours : ℕ) (more_weekday_hours : ℕ)
  (h1 : weekend_hours = 8)
  (h2 : total_weekly_hours = 33)
  (h3 : more_weekday_hours = 17) :
  (total_weekly_hours - weekend_hours) - weekend_hours = more_weekday_hours :=
by
  sorry

end NUMINAMATH_GPT_hannah_practice_hours_l1650_165004


namespace NUMINAMATH_GPT_carol_spending_l1650_165077

noncomputable def savings (S : ℝ) : Prop :=
∃ (X : ℝ) (stereo_spending television_spending : ℝ), 
  stereo_spending = (1 / 4) * S ∧
  television_spending = X * S ∧
  stereo_spending + television_spending = 0.25 * S ∧
  (stereo_spending - television_spending) / S = 0.25

theorem carol_spending (S : ℝ) : savings S :=
sorry

end NUMINAMATH_GPT_carol_spending_l1650_165077


namespace NUMINAMATH_GPT_L_shaped_figure_area_l1650_165072

noncomputable def area_rectangle (length : ℕ) (width : ℕ) : ℕ :=
  length * width

theorem L_shaped_figure_area :
  let large_rect_length := 10
  let large_rect_width := 7
  let small_rect_length := 4
  let small_rect_width := 3
  area_rectangle large_rect_length large_rect_width - area_rectangle small_rect_length small_rect_width = 58 :=
by
  sorry

end NUMINAMATH_GPT_L_shaped_figure_area_l1650_165072


namespace NUMINAMATH_GPT_cone_base_circumference_l1650_165013

theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) (C : ℝ) 
  (hV : V = 18 * Real.pi)
  (hh : h = 6) 
  (hV_cone : V = (1/3) * Real.pi * r^2 * h) :
  C = 2 * Real.pi * r → C = 6 * Real.pi :=
by 
  -- We assume as conditions are only mentioned
  sorry

end NUMINAMATH_GPT_cone_base_circumference_l1650_165013


namespace NUMINAMATH_GPT_circle_center_coordinates_l1650_165017

theorem circle_center_coordinates :
  ∃ c : ℝ × ℝ, (c = (1, -2)) ∧ 
  (∀ x y : ℝ, (x^2 + y^2 - 2*x + 4*y - 4 = 0 ↔ (x - 1)^2 + (y + 2)^2 = 9)) :=
by
  sorry

end NUMINAMATH_GPT_circle_center_coordinates_l1650_165017


namespace NUMINAMATH_GPT_chord_ratio_l1650_165052

theorem chord_ratio (A B C D P : Type) (AP BP CP DP : ℝ)
  (h1 : AP = 4) (h2 : CP = 9)
  (h3 : AP * BP = CP * DP) : BP / DP = 9 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_chord_ratio_l1650_165052


namespace NUMINAMATH_GPT_G_is_even_l1650_165071

noncomputable def G (F : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := 
  F x * (1 / (a^x - 1) + 1 / 2)

theorem G_is_even (a : ℝ) (F : ℝ → ℝ) 
  (h₀ : a > 0) 
  (h₁ : a ≠ 1)
  (hF : ∀ x : ℝ, F (-x) = - F x) : 
  ∀ x : ℝ, G F a (-x) = G F a x :=
by 
  sorry

end NUMINAMATH_GPT_G_is_even_l1650_165071


namespace NUMINAMATH_GPT_unique_triangle_determination_l1650_165082

-- Definitions for each type of triangle and their respective conditions
def isosceles_triangle (base_angle : ℝ) (altitude : ℝ) : Type := sorry
def vertex_base_isosceles_triangle (vertex_angle : ℝ) (base : ℝ) : Type := sorry
def circ_radius_side_equilateral_triangle (radius : ℝ) (side : ℝ) : Type := sorry
def leg_radius_right_triangle (leg : ℝ) (radius : ℝ) : Type := sorry
def angles_side_scalene_triangle (angle1 : ℝ) (angle2 : ℝ) (opp_side : ℝ) : Type := sorry

-- Condition: Option A does not uniquely determine a triangle
def option_A_does_not_uniquely_determine : Prop :=
  ∀ (base_angle altitude : ℝ), 
    (∃ t1 t2 : isosceles_triangle base_angle altitude, t1 ≠ t2)

-- Condition: Options B through E uniquely determine the triangle
def options_B_to_E_uniquely_determine : Prop :=
  (∀ (vertex_angle base : ℝ), ∃! t : vertex_base_isosceles_triangle vertex_angle base, true) ∧
  (∀ (radius side : ℝ), ∃! t : circ_radius_side_equilateral_triangle radius side, true) ∧
  (∀ (leg radius : ℝ), ∃! t : leg_radius_right_triangle leg radius, true) ∧
  (∀ (angle1 angle2 opp_side : ℝ), ∃! t : angles_side_scalene_triangle angle1 angle2 opp_side, true)

-- Main theorem combining both conditions
theorem unique_triangle_determination :
  option_A_does_not_uniquely_determine ∧ options_B_to_E_uniquely_determine :=
  sorry

end NUMINAMATH_GPT_unique_triangle_determination_l1650_165082


namespace NUMINAMATH_GPT_area_of_inscribed_triangle_l1650_165070

theorem area_of_inscribed_triangle 
  (x : ℝ) 
  (h1 : (2:ℝ) * x ≤ (3:ℝ) * x ∧ (3:ℝ) * x ≤ (4:ℝ) * x) 
  (h2 : (4:ℝ) * x = 2 * 4) :
  ∃ (area : ℝ), area = 12.00 :=
by
  sorry

end NUMINAMATH_GPT_area_of_inscribed_triangle_l1650_165070


namespace NUMINAMATH_GPT_regular_polygon_sides_l1650_165078

theorem regular_polygon_sides (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C]
  (angle_A angle_B angle_C : ℝ)
  (is_circle_inscribed_triangle : angle_B = 3 * angle_A ∧ angle_C = 3 * angle_A ∧ angle_B + angle_C + angle_A = 180)
  (n : ℕ)
  (is_regular_polygon : B = C ∧ angle_B = 3 * angle_A ∧ angle_C = 3 * angle_A) :
  n = 9 := sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1650_165078


namespace NUMINAMATH_GPT_count_base_8_digits_5_or_6_l1650_165064

-- Define the conditions in Lean
def is_digit_5_or_6 (d : ℕ) : Prop := d = 5 ∨ d = 6

def count_digits_5_or_6 := 
  let total_base_8 := 512
  let total_without_5_6 := 6 * 6 * 6 -- since we exclude 2 out of 8 digits
  total_base_8 - total_without_5_6

-- The statement of the proof problem
theorem count_base_8_digits_5_or_6 : count_digits_5_or_6 = 296 :=
by {
  sorry
}

end NUMINAMATH_GPT_count_base_8_digits_5_or_6_l1650_165064


namespace NUMINAMATH_GPT_walnut_trees_planted_l1650_165008

theorem walnut_trees_planted (initial_trees : ℕ) (final_trees : ℕ) (num_trees_planted : ℕ) : initial_trees = 107 → final_trees = 211 → num_trees_planted = final_trees - initial_trees → num_trees_planted = 104 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_walnut_trees_planted_l1650_165008


namespace NUMINAMATH_GPT_math_proof_problem_l1650_165068

def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def complement_R (s : Set ℝ) := {x : ℝ | x ∉ s}

theorem math_proof_problem :
  (complement_R A ∩ B) = {x | 2 < x ∧ x ≤ 3} :=
sorry

end NUMINAMATH_GPT_math_proof_problem_l1650_165068


namespace NUMINAMATH_GPT_lines_intersect_ellipse_at_2_or_4_points_l1650_165035

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 9 = 1

def line_intersects_ellipse (line : ℝ → ℝ → Prop) (x y : ℝ) : Prop :=
  ellipse_eq x y ∧ line x y

def number_of_intersections (line1 line2 : ℝ → ℝ → Prop) (n : ℕ) : Prop :=
  ∃ pts : Finset (ℝ × ℝ), (∀ pt ∈ pts, (line_intersects_ellipse line1 pt.1 pt.2 ∨
                                        line_intersects_ellipse line2 pt.1 pt.2)) ∧
                           pts.card = n ∧ 
                           (∀ pt ∈ pts, line1 pt.1 pt.2 ∨ line2 pt.1 pt.2) ∧
                           (∀ (pt1 pt2 : ℝ × ℝ), pt1 ∈ pts → pt2 ∈ pts → pt1 ≠ pt2 → pt1 ≠ pt2)

theorem lines_intersect_ellipse_at_2_or_4_points 
  (line1 line2 : ℝ → ℝ → Prop)
  (h1 : ∃ x1 y1, line1 x1 y1 ∧ ellipse_eq x1 y1)
  (h2 : ∃ x2 y2, line2 x2 y2 ∧ ellipse_eq x2 y2)
  (h3: ¬ ∀ x y, line1 x y ∧ ellipse_eq x y → false)
  (h4: ¬ ∀ x y, line2 x y ∧ ellipse_eq x y → false) :
  ∃ n : ℕ, (n = 2 ∨ n = 4) ∧ number_of_intersections line1 line2 n := sorry

end NUMINAMATH_GPT_lines_intersect_ellipse_at_2_or_4_points_l1650_165035


namespace NUMINAMATH_GPT_solve_pair_N_n_l1650_165057

def is_solution_pair (N n : ℕ) : Prop :=
  N ^ 2 = 1 + n * (N + n)

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

theorem solve_pair_N_n (N n : ℕ) (i : ℕ) :
  is_solution_pair N n ↔ N = fibonacci (i + 1) ∧ n = fibonacci i := sorry

end NUMINAMATH_GPT_solve_pair_N_n_l1650_165057


namespace NUMINAMATH_GPT_find_a_and_b_l1650_165007

-- Define the two numbers a and b and the given conditions
variables (a b : ℕ)
variables (h1 : a - b = 831) (h2 : a = 21 * b + 11)

-- State the theorem to find the values of a and b
theorem find_a_and_b (a b : ℕ) (h1 : a - b = 831) (h2 : a = 21 * b + 11) : a = 872 ∧ b = 41 :=
by
  sorry

end NUMINAMATH_GPT_find_a_and_b_l1650_165007


namespace NUMINAMATH_GPT_petes_average_speed_is_correct_l1650_165087

-- Definition of the necessary constants
def map_distance := 5.0 -- inches
def scale := 0.023809523809523808 -- inches per mile
def travel_time := 3.5 -- hours

-- The real distance calculation based on the given map scale
def real_distance := map_distance / scale -- miles

-- Proving the average speed calculation
def average_speed := real_distance / travel_time -- miles per hour

-- Theorem statement: Pete's average speed calculation is correct
theorem petes_average_speed_is_correct : average_speed = 60 :=
by
  -- Proof outline
  -- The real distance is 5 / 0.023809523809523808 ≈ 210
  -- The average speed is 210 / 3.5 ≈ 60
  sorry

end NUMINAMATH_GPT_petes_average_speed_is_correct_l1650_165087


namespace NUMINAMATH_GPT_janet_initial_number_l1650_165048

-- Define the conditions using Lean definitions
def janetProcess (x : ℕ) : ℕ :=
  (2 * (x + 7)) - 4

-- The theorem that expresses the statement of the problem: If the final result of the process is 28, then x = 9
theorem janet_initial_number (x : ℕ) (h : janetProcess x = 28) : x = 9 :=
sorry

end NUMINAMATH_GPT_janet_initial_number_l1650_165048


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1650_165088

theorem solution_set_of_inequality (a : ℝ) (h : a < 0) :
  {x : ℝ | (x - 1) * (a * x - 4) < 0} = {x : ℝ | x > 1 ∨ x < 4 / a} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1650_165088


namespace NUMINAMATH_GPT_expr_value_at_neg2_l1650_165029

variable (a b : ℝ)

def expr (x : ℝ) : ℝ := a * x^3 + b * x - 7

theorem expr_value_at_neg2 :
  (expr a b 2 = -19) → (expr a b (-2) = 5) :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_expr_value_at_neg2_l1650_165029


namespace NUMINAMATH_GPT_megan_markers_l1650_165076

theorem megan_markers (initial_markers : ℕ) (new_markers : ℕ) (total_markers : ℕ) :
  initial_markers = 217 →
  new_markers = 109 →
  total_markers = 326 →
  initial_markers + new_markers = 326 :=
by
  sorry

end NUMINAMATH_GPT_megan_markers_l1650_165076


namespace NUMINAMATH_GPT_quadratic_roots_difference_l1650_165040

theorem quadratic_roots_difference (a b : ℝ) :
  (5 * a^2 - 30 * a + 45 = 0) ∧ (5 * b^2 - 30 * b + 45 = 0) → (a - b)^2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_difference_l1650_165040


namespace NUMINAMATH_GPT_paul_books_left_l1650_165053
-- Add the necessary imports

-- Define the initial conditions
def initial_books : ℕ := 115
def books_sold : ℕ := 78

-- Statement of the problem as a theorem
theorem paul_books_left : (initial_books - books_sold) = 37 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_paul_books_left_l1650_165053


namespace NUMINAMATH_GPT_find_n_l1650_165020

theorem find_n (n : ℕ) (M : ℕ) (A : ℕ) 
  (hM : M = n - 11) 
  (hA : A = n - 2) 
  (hM_ge_one : M ≥ 1) 
  (hA_ge_one : A ≥ 1) 
  (hM_plus_A_lt_n : M + A < n) : 
  n = 12 := 
by 
  sorry

end NUMINAMATH_GPT_find_n_l1650_165020


namespace NUMINAMATH_GPT_gerald_added_crayons_l1650_165065

namespace Proof

variable (original_crayons : ℕ) (total_crayons : ℕ)

theorem gerald_added_crayons (h1 : original_crayons = 7) (h2 : total_crayons = 13) : 
  total_crayons - original_crayons = 6 := by
  sorry

end Proof

end NUMINAMATH_GPT_gerald_added_crayons_l1650_165065
