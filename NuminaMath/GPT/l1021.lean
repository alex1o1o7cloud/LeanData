import Mathlib

namespace NUMINAMATH_GPT_dagger_example_l1021_102140

def dagger (m n p q : ℚ) : ℚ := 2 * m * p * (q / n)

theorem dagger_example : dagger 5 8 3 4 = 15 := by
  sorry

end NUMINAMATH_GPT_dagger_example_l1021_102140


namespace NUMINAMATH_GPT_determine_a_l1021_102142

theorem determine_a (a : ℝ): (∃ b : ℝ, 27 * x^3 + 9 * x^2 + 36 * x + a = (3 * x + b)^3) → a = 8 := 
by
  sorry

end NUMINAMATH_GPT_determine_a_l1021_102142


namespace NUMINAMATH_GPT_solution1_solution2_solution3_l1021_102165

noncomputable def problem1 : Nat :=
  (1) * (2 - 1) * (2 + 1)

theorem solution1 : problem1 = 3 := by
  sorry

noncomputable def problem2 : Nat :=
  (2) * (2 + 1) * (2^2 + 1)

theorem solution2 : problem2 = 15 := by
  sorry

noncomputable def problem3 : Nat :=
  (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)

theorem solution3 : problem3 = 2^64 - 1 := by
  sorry

end NUMINAMATH_GPT_solution1_solution2_solution3_l1021_102165


namespace NUMINAMATH_GPT_sum_of_digits_base_2_315_l1021_102172

theorem sum_of_digits_base_2_315 :
  let binary_representation := "100111011"
  let digits_sum := binary_representation.toList.map (λ c => c.toNat - '0'.toNat)
  let sum_of_digits := digits_sum.sum
  sum_of_digits = 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_base_2_315_l1021_102172


namespace NUMINAMATH_GPT_sum_of_squares_l1021_102148

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 20) (h2 : a * b + b * c + c * a = 131) : 
  a^2 + b^2 + c^2 = 138 := 
sorry

end NUMINAMATH_GPT_sum_of_squares_l1021_102148


namespace NUMINAMATH_GPT_min_triangle_area_l1021_102150

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 2 = 1
noncomputable def circle_with_diameter_passing_origin (A B : ℝ × ℝ) : Prop :=
  let O := (0, 0)
  let d := (A.1 - B.1)^2 + (A.2 - B.2)^2
  let center := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  center.1^2 + center.2^2 = d / 4

theorem min_triangle_area (A B : ℝ × ℝ)
    (hA : hyperbola A.1 A.2)
    (hB : hyperbola B.1 B.2)
    (hc : circle_with_diameter_passing_origin A B) : 
    ∃ (S : ℝ), S = 2 :=
sorry

end NUMINAMATH_GPT_min_triangle_area_l1021_102150


namespace NUMINAMATH_GPT_largest_whole_x_l1021_102162

theorem largest_whole_x (x : ℕ) (h : 11 * x < 150) : x ≤ 13 :=
sorry

end NUMINAMATH_GPT_largest_whole_x_l1021_102162


namespace NUMINAMATH_GPT_pet_snake_cost_l1021_102198

theorem pet_snake_cost (original_amount left_amount snake_cost : ℕ) 
  (h1 : original_amount = 73) 
  (h2 : left_amount = 18)
  (h3 : snake_cost = original_amount - left_amount) : 
  snake_cost = 55 := 
by 
  sorry

end NUMINAMATH_GPT_pet_snake_cost_l1021_102198


namespace NUMINAMATH_GPT_quadratic_equation_statements_l1021_102196

theorem quadratic_equation_statements (a b c : ℝ) (h₀ : a ≠ 0) :
  (if -4 * a * c > 0 then (b^2 - 4 * a * c) > 0 else false) ∧
  ¬((b^2 - 4 * a * c > 0) → (b^2 - 4 * c * a > 0)) ∧
  ¬((c^2 * a + c * b + c = 0) → (a * c + b + 1 = 0)) ∧
  ¬(∀ (x₀ : ℝ), (a * x₀^2 + b * x₀ + c = 0) → (b^2 - 4 * a * c = (2 * a * x₀ - b)^2)) :=
by
    sorry

end NUMINAMATH_GPT_quadratic_equation_statements_l1021_102196


namespace NUMINAMATH_GPT_m_value_if_linear_l1021_102134

theorem m_value_if_linear (m : ℝ) (x : ℝ) (h : (m + 2) * x^(|m| - 1) + 8 = 0) (linear : |m| - 1 = 1) : m = 2 :=
sorry

end NUMINAMATH_GPT_m_value_if_linear_l1021_102134


namespace NUMINAMATH_GPT_width_of_carton_is_25_l1021_102160

-- Definitions for the given problem
def carton_width := 25
def carton_length := 60
def width_or_height := min carton_width carton_length

theorem width_of_carton_is_25 : width_or_height = 25 := by
  sorry

end NUMINAMATH_GPT_width_of_carton_is_25_l1021_102160


namespace NUMINAMATH_GPT_determine_k_l1021_102159

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + 2 * k * x + 1

-- State the problem
theorem determine_k (k : ℝ) :
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, f k x ≤ 4) ∧ (∃ x ∈ Set.Icc (-3 : ℝ) 2, f k x = 4)
  ↔ (k = 3 / 8 ∨ k = -3) :=
by
  sorry

end NUMINAMATH_GPT_determine_k_l1021_102159


namespace NUMINAMATH_GPT_range_of_f_on_nonneg_reals_l1021_102112

theorem range_of_f_on_nonneg_reals (k : ℕ) (h_even : k % 2 = 0) (h_pos : 0 < k) :
    ∀ y : ℝ, 0 ≤ y ↔ ∃ x : ℝ, 0 ≤ x ∧ x^k = y :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_on_nonneg_reals_l1021_102112


namespace NUMINAMATH_GPT_geometric_common_ratio_eq_three_l1021_102186

theorem geometric_common_ratio_eq_three 
  (a : ℕ → ℤ) 
  (d : ℤ) 
  (h_arithmetic_seq : ∀ n, a (n + 1) = a n + d)
  (h_nonzero_d : d ≠ 0) 
  (h_geom_seq : (a 2 + 2 * d) ^ 2 = (a 2 + d) * (a 2 + 5 * d)) : 
  (a 3) / (a 2) = 3 :=
by 
  sorry

end NUMINAMATH_GPT_geometric_common_ratio_eq_three_l1021_102186


namespace NUMINAMATH_GPT_area_quadrilateral_is_60_l1021_102102

-- Definitions of the lengths of the quadrilateral sides and the ratio condition
def AB : ℝ := 8
def BC : ℝ := 5
def CD : ℝ := 17
def DA : ℝ := 10

-- Function representing the area of the quadrilateral ABCD
def area_ABCD (AB BC CD DA : ℝ) (ratio: ℝ) : ℝ :=
  -- Here we define the function to calculate the area, incorporating the given ratio
  sorry

-- The theorem to show that the area of quadrilateral ABCD is 60
theorem area_quadrilateral_is_60 : 
  area_ABCD AB BC CD DA (1/2) = 60 :=
by
  sorry

end NUMINAMATH_GPT_area_quadrilateral_is_60_l1021_102102


namespace NUMINAMATH_GPT_concert_tickets_full_price_revenue_l1021_102157

theorem concert_tickets_full_price_revenue :
  ∃ (f p d : ℕ), f + d = 200 ∧ f * p + d * (p / 3) = 2688 ∧ f * p = 2128 :=
by
  -- We need to find the solution steps are correct to establish the existence
  sorry

end NUMINAMATH_GPT_concert_tickets_full_price_revenue_l1021_102157


namespace NUMINAMATH_GPT_find_z_l1021_102141

-- Given conditions as Lean definitions
def consecutive (x y z : ℕ) : Prop := x = z + 2 ∧ y = z + 1 ∧ x > y ∧ y > z
def equation (x y z : ℕ) : Prop := 2 * x + 3 * y + 3 * z = 5 * y + 11

-- The statement to be proven
theorem find_z (x y z : ℕ) (h1 : consecutive x y z) (h2 : equation x y z) : z = 3 :=
sorry

end NUMINAMATH_GPT_find_z_l1021_102141


namespace NUMINAMATH_GPT_range_of_x_in_sqrt_x_plus_3_l1021_102151

theorem range_of_x_in_sqrt_x_plus_3 (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_in_sqrt_x_plus_3_l1021_102151


namespace NUMINAMATH_GPT_max_prime_area_of_rectangle_with_perimeter_40_is_19_l1021_102189

-- Predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := sorry

-- Given conditions: perimeter of 40 units; perimeter condition and area as prime number.
def max_prime_area_of_rectangle_with_perimeter_40 : Prop :=
  ∃ (l w : ℕ), l + w = 20 ∧ is_prime (l * (20 - l)) ∧
  ∀ (l' w' : ℕ), l' + w' = 20 → is_prime (l' * (20 - l')) → (l * (20 - l)) ≥ (l' * (20 - l'))

theorem max_prime_area_of_rectangle_with_perimeter_40_is_19 :
  max_prime_area_of_rectangle_with_perimeter_40 :=
sorry

end NUMINAMATH_GPT_max_prime_area_of_rectangle_with_perimeter_40_is_19_l1021_102189


namespace NUMINAMATH_GPT_simplify_fraction_multiplication_l1021_102197

theorem simplify_fraction_multiplication :
  8 * (15 / 4) * (-40 / 45) = -64 / 9 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_multiplication_l1021_102197


namespace NUMINAMATH_GPT_arithmetic_sequence_root_sum_l1021_102193

theorem arithmetic_sequence_root_sum (a : ℕ → ℝ) (h_arith : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) 
    (h_roots : (a 3) * (a 8) + 3 * (a 3) + 3 * (a 8) - 18 = 0) : a 5 + a 6 = 3 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_root_sum_l1021_102193


namespace NUMINAMATH_GPT_smallest_x_solution_l1021_102161

def smallest_x_condition (x : ℝ) : Prop :=
  (x^2 - 5 * x - 84 = (x - 12) * (x + 7)) ∧
  (x ≠ 9) ∧
  (x ≠ -7) ∧
  ((x^2 - 5 * x - 84) / (x - 9) = 4 / (x + 7))

theorem smallest_x_solution :
  ∃ x : ℝ, smallest_x_condition x ∧ ∀ y : ℝ, smallest_x_condition y → x ≤ y :=
sorry

end NUMINAMATH_GPT_smallest_x_solution_l1021_102161


namespace NUMINAMATH_GPT_distance_Q_to_EH_l1021_102185

noncomputable def N : ℝ × ℝ := (3, 0)
noncomputable def E : ℝ × ℝ := (0, 6)
noncomputable def circle1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 16
noncomputable def circle2 (x y : ℝ) : Prop := x^2 + (y - 6)^2 = 9
noncomputable def EH_line (y : ℝ) : Prop := y = 6

theorem distance_Q_to_EH :
  ∃ (Q : ℝ × ℝ), circle1 Q.1 Q.2 ∧ circle2 Q.1 Q.2 ∧ Q ≠ (0, 0) ∧ abs (Q.2 - 6) = 19 / 3 := sorry

end NUMINAMATH_GPT_distance_Q_to_EH_l1021_102185


namespace NUMINAMATH_GPT_min_goals_in_previous_three_matches_l1021_102109

theorem min_goals_in_previous_three_matches 
  (score1 score2 score3 score4 : ℕ)
  (total_after_seven_matches : ℕ)
  (previous_three_goal_sum : ℕ) :
  score1 = 18 →
  score2 = 12 →
  score3 = 15 →
  score4 = 14 →
  total_after_seven_matches ≥ 100 →
  previous_three_goal_sum = total_after_seven_matches - (score1 + score2 + score3 + score4) →
  (previous_three_goal_sum / 3 : ℝ) < ((score1 + score2 + score3 + score4) / 4 : ℝ) →
  previous_three_goal_sum ≥ 41 :=
by
  sorry

end NUMINAMATH_GPT_min_goals_in_previous_three_matches_l1021_102109


namespace NUMINAMATH_GPT_find_digit_l1021_102154

theorem find_digit {x : ℕ} (hx : x = 7) : (10 * (x - 3) + x) = 47 :=
by
  sorry

end NUMINAMATH_GPT_find_digit_l1021_102154


namespace NUMINAMATH_GPT_taco_truck_revenue_l1021_102101

-- Conditions
def price_of_soft_taco : ℕ := 2
def price_of_hard_taco : ℕ := 5
def family_soft_tacos : ℕ := 3
def family_hard_tacos : ℕ := 4
def other_customers : ℕ := 10
def soft_tacos_per_other_customer : ℕ := 2

-- Calculation
def total_soft_tacos : ℕ := family_soft_tacos + other_customers * soft_tacos_per_other_customer
def revenue_from_soft_tacos : ℕ := total_soft_tacos * price_of_soft_taco
def revenue_from_hard_tacos : ℕ := family_hard_tacos * price_of_hard_taco
def total_revenue : ℕ := revenue_from_soft_tacos + revenue_from_hard_tacos

-- The proof problem
theorem taco_truck_revenue : total_revenue = 66 := 
by 
-- The proof should go here
sorry

end NUMINAMATH_GPT_taco_truck_revenue_l1021_102101


namespace NUMINAMATH_GPT_total_canoes_built_by_End_of_May_l1021_102137

noncomputable def total_canoes_built (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem total_canoes_built_by_End_of_May :
  total_canoes_built 7 2 5 = 217 :=
by
  -- The proof would go here.
  sorry

end NUMINAMATH_GPT_total_canoes_built_by_End_of_May_l1021_102137


namespace NUMINAMATH_GPT_equal_roots_h_l1021_102132

theorem equal_roots_h (h : ℝ) : (∀ x : ℝ, 3 * x^2 - 4 * x + h / 3 = 0) ↔ h = 4 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_equal_roots_h_l1021_102132


namespace NUMINAMATH_GPT_ratio_playground_landscape_l1021_102116

-- Defining the conditions
def breadth := 420
def length := breadth / 6
def playground_area := 4200
def landscape_area := length * breadth

-- Stating the theorem to prove the ratio is 1:7
theorem ratio_playground_landscape :
  (playground_area.toFloat / landscape_area.toFloat) = (1.0 / 7.0) :=
by
  sorry

end NUMINAMATH_GPT_ratio_playground_landscape_l1021_102116


namespace NUMINAMATH_GPT_tailor_charges_30_per_hour_l1021_102139

noncomputable def tailor_hourly_rate (shirts pants : ℕ) (shirt_hours pant_hours total_cost : ℝ) :=
  total_cost / (shirts * shirt_hours + pants * pant_hours)

theorem tailor_charges_30_per_hour :
  tailor_hourly_rate 10 12 1.5 3 1530 = 30 := by
  sorry

end NUMINAMATH_GPT_tailor_charges_30_per_hour_l1021_102139


namespace NUMINAMATH_GPT_known_number_is_24_l1021_102153

noncomputable def HCF (a b : ℕ) : ℕ := sorry
noncomputable def LCM (a b : ℕ) : ℕ := sorry

theorem known_number_is_24 (A B : ℕ) (h1 : B = 182)
  (h2 : HCF A B = 14)
  (h3 : LCM A B = 312) : A = 24 := by
  sorry

end NUMINAMATH_GPT_known_number_is_24_l1021_102153


namespace NUMINAMATH_GPT_sum_zero_opposites_l1021_102156

theorem sum_zero_opposites {a b : ℝ} (h : a + b = 0) : a = -b :=
by sorry

end NUMINAMATH_GPT_sum_zero_opposites_l1021_102156


namespace NUMINAMATH_GPT_range_of_m_l1021_102136

theorem range_of_m {m : ℝ} (h : ∀ x : ℝ, (3 * m - 1) ^ x = (3 * m - 1) ^ x ∧ (3 * m - 1) > 0 ∧ (3 * m - 1) < 1) :
  1 / 3 < m ∧ m < 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1021_102136


namespace NUMINAMATH_GPT_train_speed_l1021_102171

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 630) (h_time : time = 36) :
  (length / 1000) / (time / 3600) = 63 :=
by
  rw [h_length, h_time]
  sorry

end NUMINAMATH_GPT_train_speed_l1021_102171


namespace NUMINAMATH_GPT_tan_beta_identity_l1021_102190

theorem tan_beta_identity (α β : ℝ) (h1 : Real.tan α = 1/3) (h2 : Real.tan (α + β) = 1/2) :
  Real.tan β = 1/7 :=
sorry

end NUMINAMATH_GPT_tan_beta_identity_l1021_102190


namespace NUMINAMATH_GPT_unique_integer_sequence_l1021_102174

theorem unique_integer_sequence (a : ℕ → ℤ) :
  a 1 = 1 ∧ a 2 > 1 ∧ (∀ n ≥ 1, a (n + 1)^3 + 1 = a n * a (n + 2)) →
  ∃! (a : ℕ → ℤ), a 1 = 1 ∧ a 2 > 1 ∧ (∀ n ≥ 1, a (n + 1)^3 + 1 = a n * a (n + 2)) :=
sorry

end NUMINAMATH_GPT_unique_integer_sequence_l1021_102174


namespace NUMINAMATH_GPT_parallelogram_side_lengths_l1021_102194

theorem parallelogram_side_lengths (x y : ℚ) 
  (h1 : 12 * x - 2 = 10) 
  (h2 : 5 * y + 5 = 4) : 
  x + y = 4 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_parallelogram_side_lengths_l1021_102194


namespace NUMINAMATH_GPT_movie_theater_total_revenue_l1021_102108

noncomputable def revenue_from_matinee_tickets : ℕ := 20 * 5 * 1 / 2 + 180 * 5
noncomputable def revenue_from_evening_tickets : ℕ := 150 * 12 * 9 / 10 + 75 * 12 * 75 / 100 + 75 * 12
noncomputable def revenue_from_3d_tickets : ℕ := 60 * 23 + 25 * 20 * 85 / 100 + 15 * 20
noncomputable def revenue_from_late_night_tickets : ℕ := 30 * 10 * 12 / 10 + 20 * 10

noncomputable def total_revenue : ℕ :=
  revenue_from_matinee_tickets + revenue_from_evening_tickets +
  revenue_from_3d_tickets + revenue_from_late_night_tickets

theorem movie_theater_total_revenue : total_revenue = 6810 := by
  sorry

end NUMINAMATH_GPT_movie_theater_total_revenue_l1021_102108


namespace NUMINAMATH_GPT_question_condition_l1021_102100

def sufficient_but_not_necessary_condition (x : ℝ) : Prop :=
  (1 - 2 * x) * (x + 1) < 0 → x > 1 / 2 ∨ x < -1

theorem question_condition
(x : ℝ) : sufficient_but_not_necessary_condition x := sorry

end NUMINAMATH_GPT_question_condition_l1021_102100


namespace NUMINAMATH_GPT_problem_statement_l1021_102146

noncomputable def theta (h1 : 2 * Real.cos θ + Real.sin θ = 0) (h2 : 0 < θ ∧ θ < Real.pi) : Real :=
θ

noncomputable def varphi (h4 : Real.pi / 2 < φ ∧ φ < Real.pi) : Real :=
φ

theorem problem_statement
  (θ : Real) (φ : Real)
  (h1 : 2 * Real.cos θ + Real.sin θ = 0)
  (h2 : 0 < θ ∧ θ < Real.pi)
  (h3 : Real.sin (θ - φ) = Real.sqrt 10 / 10)
  (h4 : Real.pi / 2 < φ ∧ φ < Real.pi) :
  Real.tan θ = -2 ∧
  Real.sin θ = (2 * Real.sqrt 5) / 5 ∧
  Real.cos θ = -Real.sqrt 5 / 5 ∧
  Real.cos φ = -Real.sqrt 2 / 10 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1021_102146


namespace NUMINAMATH_GPT_part1_part2_l1021_102175

-- Part 1: Number of k-tuples of ordered subsets with empty intersection
theorem part1 (S : Finset α) (n k : ℕ) (h : S.card = n) (hk : 0 < k) :
  (∃ (f : Fin (n) → Fin (2^k - 1)), true) :=
sorry

-- Part 2: Number of k-tuples of subsets with chain condition
theorem part2 (S : Finset α) (n k : ℕ) (h : S.card = n) (hk : 0 < k) :
  (S.card = (k + 1)^n) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1021_102175


namespace NUMINAMATH_GPT_true_propositions_l1021_102167

theorem true_propositions :
  (∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 + 2*x - m = 0) ∧            -- Condition 1
  ((∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ∧                    -- Condition 2
   (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1) ) ∧
  (∀ x y : ℝ, (x * y ≠ 0) → (x ≠ 0 ∧ y ≠ 0)) ∧              -- Condition 3
  ¬ ( (∀ p q : Prop, ¬p → ¬ (p ∧ q)) ∧ (¬ ¬p → p ∧ q) ) ∧   -- Condition 4
  (∃ x : ℝ, x^2 + x + 3 ≤ 0)                                 -- Condition 5
:= by {
  sorry
}

end NUMINAMATH_GPT_true_propositions_l1021_102167


namespace NUMINAMATH_GPT_max_distinct_values_is_two_l1021_102145

-- Definitions of non-negative numbers and conditions
variable (a b c d : ℝ)
variable (ha : 0 ≤ a)
variable (hb : 0 ≤ b)
variable (hc : 0 ≤ c)
variable (hd : 0 ≤ d)
variable (h1 : Real.sqrt (a + b) + Real.sqrt (c + d) = Real.sqrt (a + c) + Real.sqrt (b + d))
variable (h2 : Real.sqrt (a + c) + Real.sqrt (b + d) = Real.sqrt (a + d) + Real.sqrt (b + c))

-- Theorem stating that the maximum number of distinct values among a, b, c, d is 2.
theorem max_distinct_values_is_two : 
  ∃ (u v : ℝ), 0 ≤ u ∧ 0 ≤ v ∧ (u = a ∨ u = b ∨ u = c ∨ u = d) ∧ (v = a ∨ v = b ∨ v = c ∨ v = d) ∧ 
  ∀ (x y : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d) → (y = a ∨ y = b ∨ y = c ∨ y = d) → x = y ∨ x = u ∨ x = v :=
sorry

end NUMINAMATH_GPT_max_distinct_values_is_two_l1021_102145


namespace NUMINAMATH_GPT_fraction_of_males_l1021_102155

theorem fraction_of_males (M F : ℝ) (h1 : M + F = 1) (h2 : (7/8 * M + 9/10 * (1 - M)) = 0.885) :
  M = 0.6 :=
sorry

end NUMINAMATH_GPT_fraction_of_males_l1021_102155


namespace NUMINAMATH_GPT_millie_initial_bracelets_l1021_102110

theorem millie_initial_bracelets (n : ℕ) (h1 : n - 2 = 7) : n = 9 :=
sorry

end NUMINAMATH_GPT_millie_initial_bracelets_l1021_102110


namespace NUMINAMATH_GPT_range_of_p_l1021_102178

theorem range_of_p 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n : ℕ, S n = (-1 : ℝ)^n * a n + 1/(2^n) + n - 3)
  (h2 : ∀ n : ℕ, (a (n + 1) - p) * (a n - p) < 0) :
  -3/4 < p ∧ p < 11/4 :=
sorry

end NUMINAMATH_GPT_range_of_p_l1021_102178


namespace NUMINAMATH_GPT_sandbox_area_l1021_102184

def length : ℕ := 312
def width : ℕ := 146
def area : ℕ := 45552

theorem sandbox_area : length * width = area := by
  sorry

end NUMINAMATH_GPT_sandbox_area_l1021_102184


namespace NUMINAMATH_GPT_ratio_of_volumes_cone_cylinder_l1021_102177

theorem ratio_of_volumes_cone_cylinder (r h_cylinder : ℝ) (h_cone : ℝ) (h_radius : r = 4) (h_height_cylinder : h_cylinder = 12) (h_height_cone : h_cone = h_cylinder / 2) :
  ((1/3) * (π * r^2 * h_cone)) / (π * r^2 * h_cylinder) = 1 / 6 :=
by
  -- Definitions and assumptions are directly included from the conditions.
  sorry

end NUMINAMATH_GPT_ratio_of_volumes_cone_cylinder_l1021_102177


namespace NUMINAMATH_GPT_length_of_AB_l1021_102169

open Real

noncomputable def line (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, (sqrt 3 / 2) * t)
noncomputable def curve (θ : ℝ) : ℝ × ℝ := (cos θ, sin θ)

theorem length_of_AB :
  ∃ A B : ℝ × ℝ, (∃ t : ℝ, line t = A) ∧ (∃ θ : ℝ, curve θ = A) ∧
                 (∃ t : ℝ, line t = B) ∧ (∃ θ : ℝ, curve θ = B) ∧
                 dist A B = 1 :=
by
  sorry

end NUMINAMATH_GPT_length_of_AB_l1021_102169


namespace NUMINAMATH_GPT_tank_dimension_l1021_102199

theorem tank_dimension (cost_per_sf : ℝ) (total_cost : ℝ) (length1 length3 : ℝ) (surface_area : ℝ) (dimension : ℝ) :
  cost_per_sf = 20 ∧ total_cost = 1520 ∧ 
  length1 = 4 ∧ length3 = 2 ∧ 
  surface_area = total_cost / cost_per_sf ∧
  12 * dimension + 16 = surface_area → dimension = 5 :=
by
  intro h
  obtain ⟨hcps, htac, hl1, hl3, hsa, heq⟩ := h
  sorry

end NUMINAMATH_GPT_tank_dimension_l1021_102199


namespace NUMINAMATH_GPT_min_value_2a_b_c_l1021_102158

theorem min_value_2a_b_c (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * (a + b + c) + b * c = 4) : 
  2 * a + b + c ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_2a_b_c_l1021_102158


namespace NUMINAMATH_GPT_amount_of_money_l1021_102182

theorem amount_of_money (x y : ℝ) 
  (h1 : x + 1/2 * y = 50) 
  (h2 : 2/3 * x + y = 50) : 
  (x + 1/2 * y = 50) ∧ (2/3 * x + y = 50) :=
by
  exact ⟨h1, h2⟩ 

end NUMINAMATH_GPT_amount_of_money_l1021_102182


namespace NUMINAMATH_GPT_total_number_of_employees_l1021_102133
  
def part_time_employees : ℕ := 2041
def full_time_employees : ℕ := 63093
def total_employees : ℕ := part_time_employees + full_time_employees

theorem total_number_of_employees : total_employees = 65134 := by
  sorry

end NUMINAMATH_GPT_total_number_of_employees_l1021_102133


namespace NUMINAMATH_GPT_inequality_b_2pow_a_a_2pow_neg_b_l1021_102170

theorem inequality_b_2pow_a_a_2pow_neg_b (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : 
  b * 2^a + a * 2^(-b) ≥ a + b :=
sorry

end NUMINAMATH_GPT_inequality_b_2pow_a_a_2pow_neg_b_l1021_102170


namespace NUMINAMATH_GPT_emily_page_production_difference_l1021_102111

variables (p h : ℕ)

def first_day_pages (p h : ℕ) : ℕ := p * h
def second_day_pages (p h : ℕ) : ℕ := (p - 3) * (h + 3)
def page_difference (p h : ℕ) : ℕ := second_day_pages p h - first_day_pages p h

theorem emily_page_production_difference (h : ℕ) (p_eq_3h : p = 3 * h) :
  page_difference p h = 6 * h - 9 :=
by sorry

end NUMINAMATH_GPT_emily_page_production_difference_l1021_102111


namespace NUMINAMATH_GPT_unknown_cube_edge_length_l1021_102131

theorem unknown_cube_edge_length (a b c x : ℕ) (h_a : a = 6) (h_b : b = 10) (h_c : c = 12) : a^3 + b^3 + x^3 = c^3 → x = 8 :=
by
  sorry

end NUMINAMATH_GPT_unknown_cube_edge_length_l1021_102131


namespace NUMINAMATH_GPT_solve_system_l1021_102119

theorem solve_system (a b c x y z : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (eq1 : a * y + b * x = c)
  (eq2 : c * x + a * z = b)
  (eq3 : b * z + c * y = a) :
  x = (b^2 + c^2 - a^2) / (2 * b * c) ∧ 
  y = (a^2 + c^2 - b^2) / (2 * a * c) ∧ 
  z = (a^2 + b^2 - c^2) / (2 * a * b) :=
sorry

end NUMINAMATH_GPT_solve_system_l1021_102119


namespace NUMINAMATH_GPT_g_f_3_eq_1476_l1021_102180

def f (x : ℝ) : ℝ := x^3 - 2 * x + 1
def g (x : ℝ) : ℝ := 3 * x^2 + x + 2

theorem g_f_3_eq_1476 : g (f 3) = 1476 :=
by
  sorry

end NUMINAMATH_GPT_g_f_3_eq_1476_l1021_102180


namespace NUMINAMATH_GPT_new_person_weight_l1021_102188

theorem new_person_weight (W : ℝ) (N : ℝ) (old_weight : ℝ) (average_increase : ℝ) (num_people : ℕ)
  (h1 : num_people = 8)
  (h2 : old_weight = 45)
  (h3 : average_increase = 6)
  (h4 : (W - old_weight + N) / num_people = W / num_people + average_increase) :
  N = 93 :=
by
  sorry

end NUMINAMATH_GPT_new_person_weight_l1021_102188


namespace NUMINAMATH_GPT_distance_geologists_probability_l1021_102168

theorem distance_geologists_probability :
  let speed := 4 -- km/h
  let n_roads := 6
  let travel_time := 1 -- hour
  let distance_traveled := speed * travel_time -- km
  let distance_threshold := 6 -- km
  let n_outcomes := n_roads * n_roads
  let favorable_outcomes := 18 -- determined from the solution steps
  let probability := favorable_outcomes / n_outcomes
  probability = 0.5 := by
  sorry

end NUMINAMATH_GPT_distance_geologists_probability_l1021_102168


namespace NUMINAMATH_GPT_find_k_l1021_102127

noncomputable def S (n : ℕ) : ℤ := n^2 - 8 * n
noncomputable def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem find_k (k : ℕ) (h : a k = 5) : k = 7 := by
  sorry

end NUMINAMATH_GPT_find_k_l1021_102127


namespace NUMINAMATH_GPT_value_of_fraction_l1021_102115

theorem value_of_fraction (x y : ℝ) (h : x / y = 7 / 3) : (x + y) / (x - y) = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_fraction_l1021_102115


namespace NUMINAMATH_GPT_compare_xyz_l1021_102191

open Real

noncomputable def x : ℝ := 6 * log 3 / log 64
noncomputable def y : ℝ := (1 / 3) * log 64 / log 3
noncomputable def z : ℝ := (3 / 2) * log 3 / log 8

theorem compare_xyz : x > y ∧ y > z := 
by {
  sorry
}

end NUMINAMATH_GPT_compare_xyz_l1021_102191


namespace NUMINAMATH_GPT_trishul_invested_percentage_less_than_raghu_l1021_102195

variable {T V R : ℝ}

def vishal_invested_more (T V : ℝ) : Prop :=
  V = 1.10 * T

def total_sum_of_investments (T V : ℝ) : Prop :=
  T + V + 2300 = 6647

def raghu_investment : ℝ := 2300

theorem trishul_invested_percentage_less_than_raghu
  (h1 : vishal_invested_more T V)
  (h2 : total_sum_of_investments T V) :
  ((raghu_investment - T) / raghu_investment) * 100 = 10 :=
  sorry

end NUMINAMATH_GPT_trishul_invested_percentage_less_than_raghu_l1021_102195


namespace NUMINAMATH_GPT_larger_number_hcf_lcm_l1021_102120

theorem larger_number_hcf_lcm (a b : ℕ) (h1 : Nat.gcd a b = 84) (h2 : Nat.lcm a b = 21) (h3 : a = b / 4) : max a b = 84 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_hcf_lcm_l1021_102120


namespace NUMINAMATH_GPT_parametric_to_ellipse_parametric_to_line_l1021_102121

-- Define the conditions and the corresponding parametric equations
variable (φ t : ℝ) (x y : ℝ)

-- The first parametric equation converted to the ordinary form
theorem parametric_to_ellipse (h1 : x = 5 * Real.cos φ) (h2 : y = 4 * Real.sin φ) :
  (x ^ 2 / 25) + (y ^ 2 / 16) = 1 := sorry

-- The second parametric equation converted to the ordinary form
theorem parametric_to_line (h3 : x = 1 - 3 * t) (h4 : y = 4 * t) :
  4 * x + 3 * y - 4 = 0 := sorry

end NUMINAMATH_GPT_parametric_to_ellipse_parametric_to_line_l1021_102121


namespace NUMINAMATH_GPT_square_projection_exists_l1021_102147

structure Point :=
(x y : Real)

structure Line :=
(a b c : Real) -- Line equation ax + by + c = 0

def is_on_line (P : Point) (l : Line) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

theorem square_projection_exists (P : Point) (l : Line) :
  ∃ (A B C D : Point), 
  is_on_line A l ∧ 
  is_on_line B l ∧
  (A.x + B.x) / 2 = P.x ∧ 
  (A.y + B.y) / 2 = P.y ∧ 
  (A.x = B.x ∨ A.y = B.y) ∧ -- assuming one of the sides lies along the line
  (C.x + D.x) / 2 = P.x ∧ 
  (C.y + D.y) / 2 = P.y ∧ 
  C ≠ A ∧ C ≠ B ∧ D ≠ A ∧ D ≠ B :=
sorry

end NUMINAMATH_GPT_square_projection_exists_l1021_102147


namespace NUMINAMATH_GPT_right_angled_triangle_lines_l1021_102181

theorem right_angled_triangle_lines (m : ℝ) :
  (∀ x y : ℝ, 2 * x - y + 4 = 0 → x - 2 * y + 5 = 0 → m * x - 3 * y + 12 = 0 → 
    (exists x₁ y₁ : ℝ, 2 * x₁ - 1 * y₁ + 4 = 0 ∧ (x₁ - 5) ^ 2 / 4 + y₁ / (4) = (2^(1/2))^2) ∨ 
    (exists x₂ y₂ : ℝ, 1/2 * x₂ * y₂ - y₂ / 3 + 1 / 6 = 0 ∧ (x₂ + 5) ^ 2 / 9 + y₂ / 4 = small)) → 
    (m = -3 / 2 ∨ m = -6) :=
sorry

end NUMINAMATH_GPT_right_angled_triangle_lines_l1021_102181


namespace NUMINAMATH_GPT_num_natural_a_l1021_102138

theorem num_natural_a (a b : ℕ) : 
  (a^2 + a + 100 = b^2) → ∃ n : ℕ, n = 4 := sorry

end NUMINAMATH_GPT_num_natural_a_l1021_102138


namespace NUMINAMATH_GPT_solution_set_even_function_l1021_102114

/-- Let f be an even function, and for x in [0, ∞), f(x) = x - 1. Determine the solution set for the inequality f(x) > 1.
We prove that the solution set is {x | x < -2 or x > 2}. -/
theorem solution_set_even_function (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x))
  (h_def : ∀ x, 0 ≤ x → f x = x - 1) :
  {x : ℝ | f x > 1} = {x | x < -2 ∨ x > 2} :=
by
  sorry  -- Proof steps go here.

end NUMINAMATH_GPT_solution_set_even_function_l1021_102114


namespace NUMINAMATH_GPT_subset_problem_l1021_102113

theorem subset_problem (a : ℝ) (P S : Set ℝ) :
  P = { x | x^2 - 2 * x - 3 = 0 } →
  S = { x | a * x + 2 = 0 } →
  (S ⊆ P) →
  (a = 0 ∨ a = 2 ∨ a = -2 / 3) :=
by
  intro hP hS hSubset
  sorry

end NUMINAMATH_GPT_subset_problem_l1021_102113


namespace NUMINAMATH_GPT_binom_20_10_l1021_102179

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k + 1 => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

theorem binom_20_10 :
  binom 18 8 = 43758 →
  binom 18 9 = 48620 →
  binom 18 10 = 43758 →
  binom 20 10 = 184756 :=
by
  intros h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_binom_20_10_l1021_102179


namespace NUMINAMATH_GPT_domain_of_sqrt_l1021_102124

theorem domain_of_sqrt (x : ℝ) : (x - 1 ≥ 0) → (x ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_sqrt_l1021_102124


namespace NUMINAMATH_GPT_find_point_B_l1021_102105

structure Point where
  x : ℝ
  y : ℝ

def vec_scalar_mult (c : ℝ) (v : Point) : Point :=
  ⟨c * v.x, c * v.y⟩

def vec_add (p : Point) (v : Point) : Point :=
  ⟨p.x + v.x, p.y + v.y⟩

theorem find_point_B :
  let A := Point.mk 1 (-3)
  let a := Point.mk 3 4
  let B := vec_add A (vec_scalar_mult 2 a)
  B = Point.mk 7 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_point_B_l1021_102105


namespace NUMINAMATH_GPT_proof_stmt_l1021_102187

variable (a x y : ℝ)
variable (ha : a > 0) (hneq : a ≠ 1)

noncomputable def S (x : ℝ) := a^x - a^(-x)
noncomputable def C (x : ℝ) := a^x + a^(-x)

theorem proof_stmt :
  2 * S a (x + y) = S a x * C a y + C a x * S a y ∧
  2 * S a (x - y) = S a x * C a y - C a x * S a y :=
by sorry

end NUMINAMATH_GPT_proof_stmt_l1021_102187


namespace NUMINAMATH_GPT_cleaning_times_l1021_102143

theorem cleaning_times (A B C : ℕ) (hA : A = 40) (hB : B = A / 4) (hC : C = 2 * B) : 
  B = 10 ∧ C = 20 := by
  sorry

end NUMINAMATH_GPT_cleaning_times_l1021_102143


namespace NUMINAMATH_GPT_A_pow_five_eq_rA_add_sI_l1021_102106

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  !![2, 1; 4, 3]

def I : Matrix (Fin 2) (Fin 2) ℚ :=
  1

theorem A_pow_five_eq_rA_add_sI :
  ∃ (r s : ℚ), (A^5) = r • A + s • I :=
sorry

end NUMINAMATH_GPT_A_pow_five_eq_rA_add_sI_l1021_102106


namespace NUMINAMATH_GPT_parabola_ellipse_focus_l1021_102192

theorem parabola_ellipse_focus (p : ℝ) :
  (∃ (x y : ℝ), x^2 = 2 * p * y ∧ y = -1 ∧ x = 0) →
  p = -2 :=
by
  sorry

end NUMINAMATH_GPT_parabola_ellipse_focus_l1021_102192


namespace NUMINAMATH_GPT_product_in_M_l1021_102128

def M : Set ℤ := {x | ∃ (a b : ℤ), x = a^2 - b^2}

theorem product_in_M (p q : ℤ) (hp : p ∈ M) (hq : q ∈ M) : p * q ∈ M :=
by
  sorry

end NUMINAMATH_GPT_product_in_M_l1021_102128


namespace NUMINAMATH_GPT_ambika_candles_count_l1021_102126

-- Definitions
def Aniyah_candles (A : ℕ) : ℕ := 6 * A
def combined_candles (A : ℕ) : ℕ := A + Aniyah_candles A

-- Problem Statement:
theorem ambika_candles_count : ∃ A : ℕ, combined_candles A = 28 ∧ A = 4 :=
by
  sorry

end NUMINAMATH_GPT_ambika_candles_count_l1021_102126


namespace NUMINAMATH_GPT_cos_fourth_minus_sin_fourth_l1021_102125

theorem cos_fourth_minus_sin_fourth (α : ℝ) (h : Real.sin α = (Real.sqrt 5) / 5) :
  Real.cos α ^ 4 - Real.sin α ^ 4 = 3 / 5 := 
sorry

end NUMINAMATH_GPT_cos_fourth_minus_sin_fourth_l1021_102125


namespace NUMINAMATH_GPT_apples_in_box_ratio_mixed_fruits_to_total_l1021_102118

variable (total_fruits : Nat) (oranges : Nat) (peaches : Nat) (apples : Nat) (mixed_fruits : Nat)
variable (one_fourth_of_box_contains_oranges : oranges = total_fruits / 4)
variable (half_as_many_peaches_as_oranges : peaches = oranges / 2)
variable (five_times_as_many_apples_as_peaches : apples = 5 * peaches)
variable (mixed_fruits_double_peaches : mixed_fruits = 2 * peaches)
variable (total_fruits_56 : total_fruits = 56)

theorem apples_in_box : apples = 35 := by
  sorry

theorem ratio_mixed_fruits_to_total : mixed_fruits / total_fruits = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_apples_in_box_ratio_mixed_fruits_to_total_l1021_102118


namespace NUMINAMATH_GPT_mouse_lives_count_l1021_102107

-- Define the basic conditions
def catLives : ℕ := 9
def dogLives : ℕ := catLives - 3
def mouseLives : ℕ := dogLives + 7

-- The main theorem to prove
theorem mouse_lives_count : mouseLives = 13 :=
by
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_mouse_lives_count_l1021_102107


namespace NUMINAMATH_GPT_two_false_propositions_l1021_102152

theorem two_false_propositions (a : ℝ) :
  (¬((a > -3) → (a > -6))) ∧ (¬((a > -6) → (a > -3))) → (¬(¬(a > -3) → ¬(a > -6))) :=
by
  sorry

end NUMINAMATH_GPT_two_false_propositions_l1021_102152


namespace NUMINAMATH_GPT_exp_function_not_increasing_l1021_102103

open Real

theorem exp_function_not_increasing (a : ℝ) (x : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (h₃ : a < 1) :
  ¬(∀ x₁ x₂ : ℝ, x₁ < x₂ → a^x₁ < a^x₂) := by
  sorry

end NUMINAMATH_GPT_exp_function_not_increasing_l1021_102103


namespace NUMINAMATH_GPT_ratio_of_hexagon_areas_l1021_102123

open Real

-- Define the given conditions about the hexagon and the midpoints
structure Hexagon :=
  (s : ℝ)
  (regular : True)
  (midpoints : True)

theorem ratio_of_hexagon_areas (h : Hexagon) : 
  let s := 2
  ∃ (area_ratio : ℝ), area_ratio = 4 / 7 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_hexagon_areas_l1021_102123


namespace NUMINAMATH_GPT_triangle_inequality_equivalence_l1021_102173

theorem triangle_inequality_equivalence
    (a b c : ℝ) :
  (a < b + c ∧ b < a + c ∧ c < a + b) ↔
  (|b - c| < a ∧ a < b + c ∧ |a - c| < b ∧ b < a + c ∧ |a - b| < c ∧ c < a + b) ∧
  (max a (max b c) < b + c ∧ max a (max b c) < a + c ∧ max a (max b c) < a + b) :=
by sorry

end NUMINAMATH_GPT_triangle_inequality_equivalence_l1021_102173


namespace NUMINAMATH_GPT_cost_price_of_computer_table_l1021_102144

theorem cost_price_of_computer_table (SP : ℝ) (CP : ℝ) (h : SP = CP * 1.24) (h_SP : SP = 8215) : CP = 6625 :=
by
  -- Start the proof block
  sorry -- Proof is not required as per the instructions

end NUMINAMATH_GPT_cost_price_of_computer_table_l1021_102144


namespace NUMINAMATH_GPT_runner_distance_l1021_102163

theorem runner_distance (track_length race_length : ℕ) (A_speed B_speed C_speed : ℚ)
  (h1 : track_length = 400) (h2 : race_length = 800)
  (h3 : A_speed = 1) (h4 : B_speed = 8 / 7) (h5 : C_speed = 6 / 7) :
  ∃ distance_from_finish : ℚ, distance_from_finish = 200 :=
by {
  -- We are not required to provide the actual proof steps, just setting up the definitions and initial statements for the proof.
  sorry
}

end NUMINAMATH_GPT_runner_distance_l1021_102163


namespace NUMINAMATH_GPT_niko_total_profit_l1021_102129

noncomputable def calculate_total_profit : ℝ :=
  let pairs := 9
  let price_per_pair := 2
  let discount_rate := 0.10
  let shipping_cost := 5
  let profit_4_pairs := 0.25
  let profit_5_pairs := 0.20
  let tax_rate := 0.05
  let cost_socks := pairs * price_per_pair
  let discount := discount_rate * cost_socks
  let cost_after_discount := cost_socks - discount
  let total_cost := cost_after_discount + shipping_cost
  let resell_price_4_pairs := (price_per_pair * (1 + profit_4_pairs)) * 4
  let resell_price_5_pairs := (price_per_pair * (1 + profit_5_pairs)) * 5
  let total_resell_price := resell_price_4_pairs + resell_price_5_pairs
  let sales_tax := tax_rate * total_resell_price
  let total_resell_price_after_tax := total_resell_price + sales_tax
  let total_profit := total_resell_price_after_tax - total_cost
  total_profit

theorem niko_total_profit : calculate_total_profit = 0.85 :=
by
  sorry

end NUMINAMATH_GPT_niko_total_profit_l1021_102129


namespace NUMINAMATH_GPT_total_votes_l1021_102164

-- Define the conditions
variable (V : ℝ) -- total number of votes polled
variable (w : ℝ) -- votes won by the winning candidate
variable (l : ℝ) -- votes won by the losing candidate
variable (majority : ℝ) -- majority votes

-- Define the specific values for the problem
def candidate_win_percentage (V : ℝ) : ℝ := 0.70 * V
def candidate_lose_percentage (V : ℝ) : ℝ := 0.30 * V

-- Define the majority condition
def majority_condition (V : ℝ) : Prop := (candidate_win_percentage V - candidate_lose_percentage V) = 240

-- The proof statement
theorem total_votes (V : ℝ) (h : majority_condition V) : V = 600 := by
  sorry

end NUMINAMATH_GPT_total_votes_l1021_102164


namespace NUMINAMATH_GPT_cost_of_book_l1021_102166

-- Definitions based on the conditions
def cost_pen : ℕ := 4
def cost_ruler : ℕ := 1
def fifty_dollar_bill : ℕ := 50
def change_received : ℕ := 20
def total_spent : ℕ := fifty_dollar_bill - change_received

-- Problem Statement: Prove the cost of the book
theorem cost_of_book : ∀ (cost_pen cost_ruler total_spent : ℕ), 
  total_spent = 50 - 20 → cost_pen = 4 → cost_ruler = 1 →
  (total_spent - (cost_pen + cost_ruler) = 25) :=
by
  intros cost_pen cost_ruler total_spent h1 h2 h3
  sorry

end NUMINAMATH_GPT_cost_of_book_l1021_102166


namespace NUMINAMATH_GPT_line_pass_through_point_l1021_102117

theorem line_pass_through_point (k b : ℝ) (x1 x2 : ℝ) (h1: b ≠ 0) (h2: x1^2 - k*x1 - b = 0) (h3: x2^2 - k*x2 - b = 0)
(h4: x1 + x2 = k) (h5: x1 * x2 = -b) 
(h6: (k^2 * (-b) + k * b * k + b^2 = b^2) = true) : 
  ∃ (x y : ℝ), (y = k * x + 1) ∧ (x, y) = (0, 1) :=
by
  sorry

end NUMINAMATH_GPT_line_pass_through_point_l1021_102117


namespace NUMINAMATH_GPT_find_value_of_m_l1021_102149

theorem find_value_of_m (x m : ℤ) (h₁ : x = 2) (h₂ : y = m) (h₃ : 3 * x + 2 * y = 10) : m = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_value_of_m_l1021_102149


namespace NUMINAMATH_GPT_no_such_integers_l1021_102176

theorem no_such_integers :
  ¬ (∃ a b c d : ℤ, a * 19^3 + b * 19^2 + c * 19 + d = 1 ∧ a * 62^3 + b * 62^2 + c * 62 + d = 2) :=
by
  sorry

end NUMINAMATH_GPT_no_such_integers_l1021_102176


namespace NUMINAMATH_GPT_valid_votes_for_candidate_a_l1021_102130

theorem valid_votes_for_candidate_a (total_votes : ℕ) (invalid_percentage : ℝ) (candidate_a_percentage : ℝ) (valid_votes_a : ℝ) :
  total_votes = 560000 ∧ invalid_percentage = 0.15 ∧ candidate_a_percentage = 0.80 →
  valid_votes_a = (candidate_a_percentage * (1 - invalid_percentage) * total_votes) := 
sorry

end NUMINAMATH_GPT_valid_votes_for_candidate_a_l1021_102130


namespace NUMINAMATH_GPT_starWars_earnings_correct_l1021_102122

-- Define the given conditions
def lionKing_cost : ℕ := 10
def lionKing_earnings : ℕ := 200
def starWars_cost : ℕ := 25
def lionKing_profit : ℕ := lionKing_earnings - lionKing_cost
def starWars_profit : ℕ := lionKing_profit * 2
def starWars_earnings : ℕ := starWars_profit + starWars_cost

-- The theorem which states that the Star Wars earnings are indeed 405 million
theorem starWars_earnings_correct : starWars_earnings = 405 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_starWars_earnings_correct_l1021_102122


namespace NUMINAMATH_GPT_base_area_functional_relationship_base_area_when_height_4_8_l1021_102183

noncomputable def cylinder_base_area (h : ℝ) : ℝ := 24 / h

theorem base_area_functional_relationship (h : ℝ) (H : h ≠ 0) :
  cylinder_base_area h = 24 / h := by
  unfold cylinder_base_area
  rfl

theorem base_area_when_height_4_8 :
  cylinder_base_area 4.8 = 5 := by
  unfold cylinder_base_area
  norm_num

end NUMINAMATH_GPT_base_area_functional_relationship_base_area_when_height_4_8_l1021_102183


namespace NUMINAMATH_GPT_martin_family_ice_cream_cost_l1021_102135

theorem martin_family_ice_cream_cost (R : ℤ)
  (kiddie_scoop_cost : ℤ) (double_scoop_cost : ℤ)
  (total_cost : ℤ) :
  kiddie_scoop_cost = 3 → 
  double_scoop_cost = 6 → 
  total_cost = 32 →
  2 * R + 2 * kiddie_scoop_cost + 3 * double_scoop_cost = total_cost →
  R = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_martin_family_ice_cream_cost_l1021_102135


namespace NUMINAMATH_GPT_mr_brown_selling_price_l1021_102104

noncomputable def initial_price : ℝ := 100000
noncomputable def profit_percentage : ℝ := 0.10
noncomputable def loss_percentage : ℝ := 0.10

def selling_price_mr_brown (initial_price profit_percentage : ℝ) : ℝ :=
  initial_price * (1 + profit_percentage)

def selling_price_to_friend (selling_price_mr_brown loss_percentage : ℝ) : ℝ :=
  selling_price_mr_brown * (1 - loss_percentage)

theorem mr_brown_selling_price :
  selling_price_to_friend (selling_price_mr_brown initial_price profit_percentage) loss_percentage = 99000 :=
by
  sorry

end NUMINAMATH_GPT_mr_brown_selling_price_l1021_102104
