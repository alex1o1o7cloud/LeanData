import Mathlib

namespace NUMINAMATH_GPT_sum_of_numbers_l1431_143199

-- Define the given conditions.
def S : ℕ := 30
def F : ℕ := 2 * S
def T : ℕ := F / 3

-- State the proof problem.
theorem sum_of_numbers : F + S + T = 110 :=
by
  -- Assume the proof here.
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l1431_143199


namespace NUMINAMATH_GPT_simplify_expression_1_combine_terms_l1431_143125

variable (a b : ℝ)

-- Problem 1: Simplification
theorem simplify_expression_1 : 2 * (2 * a^2 + 9 * b) + (-3 * a^2 - 4 * b) = a^2 + 14 * b := by 
  sorry

-- Problem 2: Combine like terms
theorem combine_terms : 3 * a^2 * b + 2 * a * b^2 - 5 - 3 * a^2 * b - 5 * a * b^2 + 2 = -3 * a * b^2 - 3 := by 
  sorry

end NUMINAMATH_GPT_simplify_expression_1_combine_terms_l1431_143125


namespace NUMINAMATH_GPT_price_returns_to_initial_l1431_143102

theorem price_returns_to_initial {P₀ P₁ P₂ P₃ P₄ : ℝ} (y : ℝ) (h₁ : P₀ = 100)
  (h₂ : P₁ = P₀ * 1.30) (h₃ : P₂ = P₁ * 0.70) (h₄ : P₃ = P₂ * 1.40) 
  (h₅ : P₄ = P₃ * (1 - y / 100)) : P₄ = P₀ → y = 22 :=
by
  sorry

end NUMINAMATH_GPT_price_returns_to_initial_l1431_143102


namespace NUMINAMATH_GPT_find_angle_F_l1431_143109

-- Declaring the necessary angles
variables (E F G H : ℝ) -- Angles are real numbers

-- Declaring the conditions
axiom parallel_lines : E = 3 * H
axiom angle_relation1 : G = 2 * F
axiom supplementary_angles : F + G = 180

-- The theorem statement
theorem find_angle_F (h1 : E = 3 * H) (h2 : G = 2 * F) (h3 : F + G = 180) : F = 60 :=
  sorry

end NUMINAMATH_GPT_find_angle_F_l1431_143109


namespace NUMINAMATH_GPT_minimum_buses_needed_l1431_143134

theorem minimum_buses_needed (bus_capacity : ℕ) (students : ℕ) (h : bus_capacity = 38 ∧ students = 411) :
  ∃ n : ℕ, 38 * n ≥ students ∧ ∀ m : ℕ, 38 * m ≥ students → n ≤ m :=
by sorry

end NUMINAMATH_GPT_minimum_buses_needed_l1431_143134


namespace NUMINAMATH_GPT_production_rate_equation_l1431_143149

theorem production_rate_equation (x : ℝ) (h : x > 0) :
  3000 / x - 3000 / (2 * x) = 5 :=
sorry

end NUMINAMATH_GPT_production_rate_equation_l1431_143149


namespace NUMINAMATH_GPT_find_a_l1431_143145

theorem find_a : (a : ℕ) = 103 * 97 * 10009 → a = 99999919 := by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l1431_143145


namespace NUMINAMATH_GPT_find_m_and_f_max_l1431_143129

noncomputable def f (x m : ℝ) : ℝ :=
  (Real.sqrt 3) * Real.sin (2 * x) + 2 * (Real.cos x)^2 + m

theorem find_m_and_f_max (m a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x m ≥ 3) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), ∃ y, f y m = 3) →
  (∀ x ∈ Set.Icc a (a + Real.pi), ∃ y, f y m = 6) →
  m = 3 ∧ ∀ x ∈ Set.Icc a (a + Real.pi), f x 3 ≤ 6 :=
sorry

end NUMINAMATH_GPT_find_m_and_f_max_l1431_143129


namespace NUMINAMATH_GPT_domain_of_tan_l1431_143178

theorem domain_of_tan :
    ∀ k : ℤ, ∀ x : ℝ,
    (x > (k * π / 2 - π / 8) ∧ x < (k * π / 2 + 3 * π / 8)) ↔
    2 * x - π / 4 ≠ k * π + π / 2 :=
by
  intro k x
  sorry

end NUMINAMATH_GPT_domain_of_tan_l1431_143178


namespace NUMINAMATH_GPT_mike_total_spending_is_correct_l1431_143175

-- Definitions for the costs of the items
def cost_marbles : ℝ := 9.05
def cost_football : ℝ := 4.95
def cost_baseball : ℝ := 6.52
def cost_toy_car : ℝ := 3.75
def cost_puzzle : ℝ := 8.99
def cost_stickers : ℝ := 1.25

-- Definitions for the discounts
def discount_puzzle : ℝ := 0.15
def discount_toy_car : ℝ := 0.10

-- Definition for the coupon
def coupon_amount : ℝ := 5.00

-- Total spent by Mike on toys
def total_spent : ℝ :=
  cost_marbles + 
  cost_football + 
  cost_baseball + 
  (cost_toy_car - cost_toy_car * discount_toy_car) + 
  (cost_puzzle - cost_puzzle * discount_puzzle) + 
  cost_stickers - 
  coupon_amount

-- Proof statement
theorem mike_total_spending_is_correct : 
  total_spent = 27.7865 :=
by
  sorry

end NUMINAMATH_GPT_mike_total_spending_is_correct_l1431_143175


namespace NUMINAMATH_GPT_greatest_common_divisor_456_108_lt_60_l1431_143138

theorem greatest_common_divisor_456_108_lt_60 : 
  let divisors_456 := {d : ℕ | d ∣ 456}
  let divisors_108 := {d : ℕ | d ∣ 108}
  let common_divisors := divisors_456 ∩ divisors_108
  let common_divisors_lt_60 := {d ∈ common_divisors | d < 60}
  ∃ d, d ∈ common_divisors_lt_60 ∧ ∀ e ∈ common_divisors_lt_60, e ≤ d ∧ d = 12 := by {
    sorry
  }

end NUMINAMATH_GPT_greatest_common_divisor_456_108_lt_60_l1431_143138


namespace NUMINAMATH_GPT_integer_divisibility_l1431_143188

theorem integer_divisibility
  (x y z : ℤ)
  (h : 11 ∣ (7 * x + 2 * y - 5 * z)) :
  11 ∣ (3 * x - 7 * y + 12 * z) :=
sorry

end NUMINAMATH_GPT_integer_divisibility_l1431_143188


namespace NUMINAMATH_GPT_xy_value_l1431_143111

theorem xy_value (x y : ℝ) (h : (x + y)^2 - (x - y)^2 = 20) : x * y = 5 :=
by
  sorry

end NUMINAMATH_GPT_xy_value_l1431_143111


namespace NUMINAMATH_GPT_ratio_of_A_to_B_l1431_143155

theorem ratio_of_A_to_B (A B C : ℝ) (hB : B = 270) (hBC : B = (1 / 4) * C) (hSum : A + B + C = 1440) : A / B = 1 / 3 :=
by
  -- The proof is omitted for this example
  sorry

end NUMINAMATH_GPT_ratio_of_A_to_B_l1431_143155


namespace NUMINAMATH_GPT_complement_A_inter_B_l1431_143169

def U : Set ℤ := { x | -1 ≤ x ∧ x ≤ 2 }
def A : Set ℤ := { x | x * (x - 1) = 0 }
def B : Set ℤ := { x | -1 < x ∧ x < 2 }

theorem complement_A_inter_B {U A B : Set ℤ} :
  A ⊆ U → B ⊆ U → 
  (A ∩ B) ⊆ (U ∩ A ∩ B) → 
  (U \ (A ∩ B)) = { -1, 2 } :=
by 
  sorry

end NUMINAMATH_GPT_complement_A_inter_B_l1431_143169


namespace NUMINAMATH_GPT_infinitely_many_n_squared_plus_one_no_special_divisor_l1431_143159

theorem infinitely_many_n_squared_plus_one_no_special_divisor :
  ∃ (f : ℕ → ℕ), (∀ n, f n ≠ 0) ∧ ∀ n, ∀ k, f n^2 + 1 ≠ k^2 + 1 ∨ k^2 + 1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_infinitely_many_n_squared_plus_one_no_special_divisor_l1431_143159


namespace NUMINAMATH_GPT_monotonically_increasing_intervals_exists_a_decreasing_l1431_143100

noncomputable def f (x : ℝ) (a : ℝ) := Real.exp x - a * x - 1

theorem monotonically_increasing_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x : ℝ, 0 ≤ Real.exp x - a) ∧
  (a > 0 → ∀ x : ℝ, x ≥ Real.log a → 0 ≤ Real.exp x - a) :=
by sorry

theorem exists_a_decreasing (a : ℝ) :
  (a ≥ Real.exp 3) ↔ ∀ x : ℝ, -2 < x ∧ x < 3 → Real.exp x - a ≤ 0 :=
by sorry

end NUMINAMATH_GPT_monotonically_increasing_intervals_exists_a_decreasing_l1431_143100


namespace NUMINAMATH_GPT_smallest_omega_l1431_143186

theorem smallest_omega (ω : ℝ) (hω_pos : ω > 0) :
  (∃ k : ℤ, (2 / 3) * ω = 2 * k) -> ω = 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_omega_l1431_143186


namespace NUMINAMATH_GPT_prob_sum_divisible_by_4_l1431_143168

-- Defining the set and its properties
def set : Finset ℕ := {1, 2, 3, 4, 5}

def isDivBy4 (n : ℕ) : Prop := n % 4 = 0

-- Defining a function to calculate combinations
def combinations (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Defining the successful outcomes and the total combinations
def successfulOutcomes : ℕ := 3
def totalOutcomes : ℕ := combinations 5 3

-- Defining the probability
def probability : ℚ := successfulOutcomes / ↑totalOutcomes

-- The proof problem
theorem prob_sum_divisible_by_4 : probability = 3 / 10 := by
  sorry

end NUMINAMATH_GPT_prob_sum_divisible_by_4_l1431_143168


namespace NUMINAMATH_GPT_rectangle_area_unchanged_l1431_143161

theorem rectangle_area_unchanged
  (x y : ℝ)
  (h1 : x * y = (x + 3) * (y - 1))
  (h2 : x * y = (x - 3) * (y + 1.5)) :
  x * y = 31.5 :=
sorry

end NUMINAMATH_GPT_rectangle_area_unchanged_l1431_143161


namespace NUMINAMATH_GPT_min_value_f_l1431_143112

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) + (1 / (x^2 + (1 / x)))

theorem min_value_f : ∃ x > 0, ∀ y > 0, f y ≥ f x ∧ f x = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_f_l1431_143112


namespace NUMINAMATH_GPT_common_divisors_sum_diff_l1431_143114

theorem common_divisors_sum_diff (A B : ℤ) (h : Int.gcd A B = 1) : 
  {d : ℤ | d ∣ A + B ∧ d ∣ A - B} = {1, 2} :=
sorry

end NUMINAMATH_GPT_common_divisors_sum_diff_l1431_143114


namespace NUMINAMATH_GPT_find_M_l1431_143185

variable (M : ℕ)

theorem find_M (h : (5 + 6 + 7) / 3 = (2005 + 2006 + 2007) / M) : M = 1003 :=
sorry

end NUMINAMATH_GPT_find_M_l1431_143185


namespace NUMINAMATH_GPT_tip_percentage_is_20_l1431_143179

noncomputable def total_bill : ℕ := 16 + 14
noncomputable def james_share : ℕ := total_bill / 2
noncomputable def james_paid : ℕ := 21
noncomputable def tip_amount : ℕ := james_paid - james_share
noncomputable def tip_percentage : ℕ := (tip_amount * 100) / total_bill 

theorem tip_percentage_is_20 :
  tip_percentage = 20 :=
by
  sorry

end NUMINAMATH_GPT_tip_percentage_is_20_l1431_143179


namespace NUMINAMATH_GPT_pies_sold_each_day_l1431_143165

theorem pies_sold_each_day (total_pies: ℕ) (days_in_week: ℕ) 
  (h1: total_pies = 56) (h2: days_in_week = 7) : 
  total_pies / days_in_week = 8 :=
by
  sorry

end NUMINAMATH_GPT_pies_sold_each_day_l1431_143165


namespace NUMINAMATH_GPT_ellipse_major_minor_axes_product_l1431_143198

-- Definitions based on conditions
def OF : ℝ := 8
def inradius_triangle_OCF : ℝ := 2  -- diameter / 2

-- Define a and b based on the ellipse properties and conditions
def a : ℝ := 10  -- Solved from the given conditions and steps
def b : ℝ := 6   -- Solved from the given conditions and steps

-- Defining the axes of the ellipse in terms of a and b
def AB : ℝ := 2 * a
def CD : ℝ := 2 * b

-- The product (AB)(CD) we are interested in
def product_AB_CD := AB * CD

-- The main proof statement
theorem ellipse_major_minor_axes_product : product_AB_CD = 240 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_major_minor_axes_product_l1431_143198


namespace NUMINAMATH_GPT_quadratic_equation_has_real_root_l1431_143184

theorem quadratic_equation_has_real_root
  (a c m n : ℝ) :
  ∃ x : ℝ, c * x^2 + m * x - a = 0 ∨ ∃ y : ℝ, a * y^2 + n * y + c = 0 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_quadratic_equation_has_real_root_l1431_143184


namespace NUMINAMATH_GPT_count_special_four_digit_integers_is_100_l1431_143120

def count_special_four_digit_integers : Nat := sorry

theorem count_special_four_digit_integers_is_100 :
  count_special_four_digit_integers = 100 :=
sorry

end NUMINAMATH_GPT_count_special_four_digit_integers_is_100_l1431_143120


namespace NUMINAMATH_GPT_f_neg_1_l1431_143157

-- Define the functions
variable (f : ℝ → ℝ) -- f is a real-valued function
variable (g : ℝ → ℝ) -- g is a real-valued function

-- Given conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_def : ∀ x, g x = f x + 4
axiom g_at_1 : g 1 = 2

-- Define the theorem to prove
theorem f_neg_1 : f (-1) = 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_f_neg_1_l1431_143157


namespace NUMINAMATH_GPT_solve_inequality_l1431_143105

noncomputable def f (x : ℝ) : ℝ :=
  x^3 + x + 2^x - 2^(-x)

theorem solve_inequality (x : ℝ) : 
  f (Real.exp x - x) ≤ 7/2 ↔ x = 0 := 
sorry

end NUMINAMATH_GPT_solve_inequality_l1431_143105


namespace NUMINAMATH_GPT_circle_diameter_l1431_143108

theorem circle_diameter (r d : ℝ) (h1 : π * r^2 = 4 * π) (h2 : d = 2 * r) : d = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_circle_diameter_l1431_143108


namespace NUMINAMATH_GPT_num_rows_seat_9_people_l1431_143103

-- Define the premises of the problem.
def seating_arrangement (x y : ℕ) : Prop := (9 * x + 7 * y = 58)

-- The theorem stating the number of rows seating exactly 9 people.
theorem num_rows_seat_9_people
  (x y : ℕ)
  (h : seating_arrangement x y) :
  x = 1 :=
by
  -- Proof is not required as per the instruction
  sorry

end NUMINAMATH_GPT_num_rows_seat_9_people_l1431_143103


namespace NUMINAMATH_GPT_inequality_solution_l1431_143162

theorem inequality_solution (x : ℝ) : 3 * x^2 - 8 * x + 3 < 0 ↔ (1 / 3 < x ∧ x < 3) := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1431_143162


namespace NUMINAMATH_GPT_amount_of_salmon_sold_first_week_l1431_143195

-- Define the conditions
def fish_sold_in_two_weeks (x : ℝ) := x + 3 * x = 200

-- Define the theorem we want to prove
theorem amount_of_salmon_sold_first_week (x : ℝ) (h : fish_sold_in_two_weeks x) : x = 50 :=
by
  sorry

end NUMINAMATH_GPT_amount_of_salmon_sold_first_week_l1431_143195


namespace NUMINAMATH_GPT_part1_part2_l1431_143142

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) + abs (x - 2 * a + 3)

theorem part1 (x : ℝ) : f x 2 ≤ 9 ↔ -2 ≤ x ∧ x ≤ 4 :=
by sorry

theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 4) : a ∈ Set.Iic (-2 / 3) ∪ Set.Ici (14 / 3) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1431_143142


namespace NUMINAMATH_GPT_inequality_solution_set_l1431_143189

variable {f : ℝ → ℝ}

-- Conditions
def neg_domain : Set ℝ := {x | x < 0}
def pos_domain : Set ℝ := {x | x > 0}
def f_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def f_property_P (f : ℝ → ℝ) := ∀ (x1 x2 : ℝ), (0 < x1) → (0 < x2) → (x1 ≠ x2) → ((x2 * f x1 - x1 * f x2) / (x1 - x2) < 0)

-- Translate question and correct answer into a proposition in Lean
theorem inequality_solution_set (h1 : ∀ x, f (-x) = -f x)
                                (h2 : ∀ x1 x2, (0 < x1) → (0 < x2) → (x1 ≠ x1) → ((x2 * f x1 - x1 * f x2) / (x1 - x2) < 0)) :
  {x | f (x - 2) < f (x^2 - 4) / (x + 2)} = {x | x < -3} ∪ {x | -1 < x ∧ x < 2} := 
sorry

end NUMINAMATH_GPT_inequality_solution_set_l1431_143189


namespace NUMINAMATH_GPT_carlos_local_tax_deduction_l1431_143126

theorem carlos_local_tax_deduction :
  let hourly_wage_dollars := 25
  let hourly_wage_cents := hourly_wage_dollars * 100
  let tax_rate := 2.5 / 100
  hourly_wage_cents * tax_rate = 62.5 :=
by
  sorry

end NUMINAMATH_GPT_carlos_local_tax_deduction_l1431_143126


namespace NUMINAMATH_GPT_compare_numbers_l1431_143180

theorem compare_numbers :
  3 * 10^5 < 2 * 10^6 ∧ -2 - 1 / 3 > -3 - 1 / 2 := by
  sorry

end NUMINAMATH_GPT_compare_numbers_l1431_143180


namespace NUMINAMATH_GPT_putnam_inequality_l1431_143181

variable (a x : ℝ)

theorem putnam_inequality (h1 : 0 < x) (h2 : x < a) :
  (a - x)^6 - 3 * a * (a - x)^5 +
  5 / 2 * a^2 * (a - x)^4 -
  1 / 2 * a^4 * (a - x)^2 < 0 :=
by
  sorry

end NUMINAMATH_GPT_putnam_inequality_l1431_143181


namespace NUMINAMATH_GPT_positive_difference_of_two_numbers_l1431_143107

theorem positive_difference_of_two_numbers 
  (x y : ℝ) 
  (h1 : x + y = 10) 
  (h2 : x^2 - y^2 = 24) : 
  |x - y| = 2.4 := 
sorry

end NUMINAMATH_GPT_positive_difference_of_two_numbers_l1431_143107


namespace NUMINAMATH_GPT_weavers_in_first_group_l1431_143123

theorem weavers_in_first_group :
  (∃ W : ℕ, (W * 4 = 4) ∧ (12 * 12 = 36) ∧ (4 / (W * 4) = 36 / (12 * 12))) -> (W = 4) :=
by
  sorry

end NUMINAMATH_GPT_weavers_in_first_group_l1431_143123


namespace NUMINAMATH_GPT_find_cost_price_per_meter_l1431_143144

/-- Given that a shopkeeper sells 200 meters of cloth for Rs. 12000 at a loss of Rs. 6 per meter,
we want to find the cost price per meter of cloth. Specifically, we need to prove that the
cost price per meter is Rs. 66. -/
theorem find_cost_price_per_meter
  (total_meters : ℕ := 200)
  (selling_price : ℕ := 12000)
  (loss_per_meter : ℕ := 6) :
  (selling_price + total_meters * loss_per_meter) / total_meters = 66 :=
sorry

end NUMINAMATH_GPT_find_cost_price_per_meter_l1431_143144


namespace NUMINAMATH_GPT_circle_center_polar_coords_l1431_143115

noncomputable def polar_center (ρ θ : ℝ) : (ℝ × ℝ) :=
  (-1, 0)

theorem circle_center_polar_coords : 
  ∀ ρ θ : ℝ, ρ = -2 * Real.cos θ → polar_center ρ θ = (1, π) :=
by
  intro ρ θ h
  sorry

end NUMINAMATH_GPT_circle_center_polar_coords_l1431_143115


namespace NUMINAMATH_GPT_simplify_polynomial_expression_l1431_143193

theorem simplify_polynomial_expression (r : ℝ) :
  (2 * r^3 + 5 * r^2 + 6 * r - 4) - (r^3 + 9 * r^2 + 4 * r - 7) = r^3 - 4 * r^2 + 2 * r + 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_expression_l1431_143193


namespace NUMINAMATH_GPT_correct_equation_l1431_143143

theorem correct_equation :
  ¬ (7^3 * 7^3 = 7^9) ∧ 
  (-3^7 / 3^2 = -3^5) ∧ 
  ¬ (2^6 + (-2)^6 = 0) ∧ 
  ¬ ((-3)^5 / (-3)^3 = -3^2) :=
by 
  sorry

end NUMINAMATH_GPT_correct_equation_l1431_143143


namespace NUMINAMATH_GPT_factorial_simplification_l1431_143130

theorem factorial_simplification :
  Nat.factorial 16 / (Nat.factorial 6 * Nat.factorial 10) = 728 := 
sorry

end NUMINAMATH_GPT_factorial_simplification_l1431_143130


namespace NUMINAMATH_GPT_marbles_left_l1431_143133

theorem marbles_left (initial_marbles : ℕ) (given_marbles : ℕ) (remaining_marbles : ℕ) :
  initial_marbles = 64 → given_marbles = 14 → remaining_marbles = (initial_marbles - given_marbles) → remaining_marbles = 50 :=
by
  intros h_initial h_given h_calculation
  rw [h_initial, h_given] at h_calculation
  exact h_calculation

end NUMINAMATH_GPT_marbles_left_l1431_143133


namespace NUMINAMATH_GPT_volume_ratio_l1431_143116

theorem volume_ratio (A B C : ℚ) (h1 : (3/4) * A = (2/3) * B) (h2 : (2/3) * B = (1/2) * C) :
  A / C = 2 / 3 :=
sorry

end NUMINAMATH_GPT_volume_ratio_l1431_143116


namespace NUMINAMATH_GPT_part_I_part_II_l1431_143171

noncomputable def f (x : ℝ) : ℝ :=
  |x - (1/2)| + |x + (1/2)|

def solutionSetM : Set ℝ :=
  { x : ℝ | -1 < x ∧ x < 1 }

theorem part_I :
  { x : ℝ | f x < 2 } = solutionSetM := 
sorry

theorem part_II (a b : ℝ) (ha : a ∈ solutionSetM) (hb : b ∈ solutionSetM) :
  |a + b| < |1 + a * b| :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l1431_143171


namespace NUMINAMATH_GPT_problem_equivalence_l1431_143154

theorem problem_equivalence :
  (∃ a a1 a2 a3 a4 a5 : ℝ, ((1 - x)^5 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5)) → 
  ∀ (a a1 a2 a3 a4 a5 : ℝ), (1 - 1)^5 = a + a1 + a2 + a3 + a4 + a5 →
  (1 + 1)^5 = a - a1 + a2 - a3 + a4 - a5 →
  (a + a2 + a4) * (a1 + a3 + a5) = -256 := 
by
  intros h a a1 a2 a3 a4 a5 e1 e2
  sorry

end NUMINAMATH_GPT_problem_equivalence_l1431_143154


namespace NUMINAMATH_GPT_P_and_Q_equivalent_l1431_143148

def P (x : ℝ) : Prop := 3 * x - x^2 ≤ 0
def Q (x : ℝ) : Prop := |x| ≤ 2
def P_intersection_Q (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 0

theorem P_and_Q_equivalent : ∀ x, (P x ∧ Q x) ↔ P_intersection_Q x :=
by {
  sorry
}

end NUMINAMATH_GPT_P_and_Q_equivalent_l1431_143148


namespace NUMINAMATH_GPT_product_of_numbers_l1431_143167

theorem product_of_numbers (x y z : ℤ) 
  (h1 : x + y + z = 30) 
  (h2 : x = 3 * ((y + z) - 2))
  (h3 : y = 4 * z - 1) : 
  x * y * z = 294 := 
  sorry

end NUMINAMATH_GPT_product_of_numbers_l1431_143167


namespace NUMINAMATH_GPT_yoongi_age_l1431_143174

theorem yoongi_age (Y H : ℕ) (h1 : Y + H = 16) (h2 : Y = H + 2) : Y = 9 :=
by
  sorry

end NUMINAMATH_GPT_yoongi_age_l1431_143174


namespace NUMINAMATH_GPT_derivative_at_neg_one_l1431_143196

noncomputable def f (a b c x : ℝ) : ℝ := a * x^4 + b * x^2 + c

theorem derivative_at_neg_one (a b c : ℝ) (h : (4 * a * 1^3 + 2 * b * 1) = 2) : 
  (4 * a * (-1)^3 + 2 * b * (-1)) = -2 := 
sorry

end NUMINAMATH_GPT_derivative_at_neg_one_l1431_143196


namespace NUMINAMATH_GPT_fraction_of_n_is_80_l1431_143150

-- Definitions from conditions
def n := (5 / 6) * 240

-- The theorem we want to prove
theorem fraction_of_n_is_80 : (2 / 5) * n = 80 :=
by
  -- This is just a placeholder to complete the statement, 
  -- actual proof logic is not included based on the prompt instructions
  sorry

end NUMINAMATH_GPT_fraction_of_n_is_80_l1431_143150


namespace NUMINAMATH_GPT_hyperbola_vertex_distance_l1431_143187

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  16 * x^2 + 64 * x - 4 * y^2 + 8 * y + 36 = 0

-- Statement: The distance between the vertices of the hyperbola is 1
theorem hyperbola_vertex_distance :
  ∀ x y : ℝ, hyperbola_eq x y → 2 * (1 / 2) = 1 :=
by
  intros x y H
  sorry

end NUMINAMATH_GPT_hyperbola_vertex_distance_l1431_143187


namespace NUMINAMATH_GPT_brother_to_madeline_ratio_l1431_143172

theorem brother_to_madeline_ratio (M B T : ℕ) (hM : M = 48) (hT : T = 72) (hSum : M + B = T) : B / M = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_brother_to_madeline_ratio_l1431_143172


namespace NUMINAMATH_GPT_common_ratio_geom_series_l1431_143151

theorem common_ratio_geom_series 
  (a₁ a₂ : ℚ) 
  (h₁ : a₁ = 4 / 7) 
  (h₂ : a₂ = 20 / 21) :
  ∃ r : ℚ, r = 5 / 3 ∧ a₂ / a₁ = r := 
sorry

end NUMINAMATH_GPT_common_ratio_geom_series_l1431_143151


namespace NUMINAMATH_GPT_age_of_b_l1431_143152

variable {a b c d Y : ℝ}

-- Conditions
def condition1 (a b : ℝ) := a = b + 2
def condition2 (b c : ℝ) := b = 2 * c
def condition3 (a d : ℝ) := d = a / 2
def condition4 (a b c d Y : ℝ) := a + b + c + d = Y

-- Theorem to prove
theorem age_of_b (a b c d Y : ℝ) 
  (h1 : condition1 a b) 
  (h2 : condition2 b c) 
  (h3 : condition3 a d) 
  (h4 : condition4 a b c d Y) : 
  b = Y / 3 - 1 := 
sorry

end NUMINAMATH_GPT_age_of_b_l1431_143152


namespace NUMINAMATH_GPT_cone_cube_volume_ratio_l1431_143191

noncomputable def volumeRatio (s : ℝ) : ℝ :=
  let r := s / 2
  let h := s
  let volume_cone := (1 / 3) * Real.pi * r^2 * h
  let volume_cube := s^3
  volume_cone / volume_cube

theorem cone_cube_volume_ratio (s : ℝ) (h_cube_eq_s : s > 0) :
  volumeRatio s = Real.pi / 12 :=
by
  sorry

end NUMINAMATH_GPT_cone_cube_volume_ratio_l1431_143191


namespace NUMINAMATH_GPT_relatively_prime_2n_plus_1_4n2_plus_1_l1431_143146

theorem relatively_prime_2n_plus_1_4n2_plus_1 (n : ℕ) (h : n > 0) : 
  Nat.gcd (2 * n + 1) (4 * n^2 + 1) = 1 := 
by
  sorry

end NUMINAMATH_GPT_relatively_prime_2n_plus_1_4n2_plus_1_l1431_143146


namespace NUMINAMATH_GPT_problem_part1_and_part2_l1431_143117

noncomputable def g (x a b : ℝ) : ℝ := a * Real.log x + 0.5 * x ^ 2 + (1 - b) * x

-- Given: the function definition and conditions
variables (a b : ℝ)
variables (x1 x2 : ℝ)
variables (hx1 : x1 ∈ Set.Ioi 0) (hx2 : x2 ∈ Set.Ioi 0)
variables (h_tangent : 8 * 1 - 2 * g 1 a b - 3 = 0)
variables (h_extremes : b = a + 1)

-- Prove the values of a and b as well as the inequality
theorem problem_part1_and_part2 :
  (a = 1 ∧ b = -1) ∧ (g x1 a b + g x2 a b < -4) :=
sorry

end NUMINAMATH_GPT_problem_part1_and_part2_l1431_143117


namespace NUMINAMATH_GPT_square_side_length_l1431_143131

noncomputable def diagonal_in_inches : ℝ := 2 * Real.sqrt 2
noncomputable def inches_to_feet : ℝ := 1 / 12
noncomputable def diagonal_in_feet := diagonal_in_inches * inches_to_feet
noncomputable def factor_sqrt_2 : ℝ := 1 / Real.sqrt 2

theorem square_side_length :
  let diagonal_feet := diagonal_in_feet 
  let side_length_feet := diagonal_feet * factor_sqrt_2
  side_length_feet = 1 / 6 :=
sorry

end NUMINAMATH_GPT_square_side_length_l1431_143131


namespace NUMINAMATH_GPT_abs_value_product_l1431_143156

theorem abs_value_product (x : ℝ) (h : |x - 5| - 4 = 0) : ∃ y z, (y - 5 = 4 ∨ y - 5 = -4) ∧ (z - 5 = 4 ∨ z - 5 = -4) ∧ y * z = 9 :=
by 
  sorry

end NUMINAMATH_GPT_abs_value_product_l1431_143156


namespace NUMINAMATH_GPT_problem_statement_l1431_143197

/-!
The problem states:
If |a-2| and |m+n+3| are opposite numbers, then a + m + n = -1.
-/

theorem problem_statement (a m n : ℤ) (h : |a - 2| = -|m + n + 3|) : a + m + n = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_statement_l1431_143197


namespace NUMINAMATH_GPT_distinct_nonzero_digits_sum_l1431_143194

theorem distinct_nonzero_digits_sum
  (x y z w : Nat)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0)
  (hw : w ≠ 0)
  (hxy : x ≠ y)
  (hxz : x ≠ z)
  (hxw : x ≠ w)
  (hyz : y ≠ z)
  (hyw : y ≠ w)
  (hzw : z ≠ w)
  (h1 : w + x = 10)
  (h2 : y + w = 9)
  (h3 : z + x = 9) :
  x + y + z + w = 18 :=
sorry

end NUMINAMATH_GPT_distinct_nonzero_digits_sum_l1431_143194


namespace NUMINAMATH_GPT_cost_per_pack_is_correct_l1431_143158

def total_amount_spent : ℝ := 120
def num_packs_bought : ℕ := 6
def expected_cost_per_pack : ℝ := 20

theorem cost_per_pack_is_correct :
  total_amount_spent / num_packs_bought = expected_cost_per_pack :=
  by 
    -- here would be the proof
    sorry

end NUMINAMATH_GPT_cost_per_pack_is_correct_l1431_143158


namespace NUMINAMATH_GPT_tim_total_expenditure_l1431_143170

def apple_price : ℕ := 1
def milk_price : ℕ := 3
def pineapple_price : ℕ := 4
def flour_price : ℕ := 6
def chocolate_price : ℕ := 10

def apple_quantity : ℕ := 8
def milk_quantity : ℕ := 4
def pineapple_quantity : ℕ := 3
def flour_quantity : ℕ := 3
def chocolate_quantity : ℕ := 1

def discounted_pineapple_price : ℕ := pineapple_price / 2
def discounted_milk_price : ℕ := milk_price - 1
def coupon_discount : ℕ := 10
def discount_threshold : ℕ := 50

def total_cost_before_coupon : ℕ :=
  (apple_quantity * apple_price) +
  (milk_quantity * discounted_milk_price) +
  (pineapple_quantity * discounted_pineapple_price) +
  (flour_quantity * flour_price) +
  chocolate_price

def final_price : ℕ :=
  if total_cost_before_coupon >= discount_threshold
  then total_cost_before_coupon - coupon_discount
  else total_cost_before_coupon

theorem tim_total_expenditure : final_price = 40 := by
  sorry

end NUMINAMATH_GPT_tim_total_expenditure_l1431_143170


namespace NUMINAMATH_GPT_new_average_score_l1431_143139

theorem new_average_score (avg_score : ℝ) (num_students : ℕ) (dropped_score : ℝ) (new_num_students : ℕ) :
  num_students = 16 →
  avg_score = 61.5 →
  dropped_score = 24 →
  new_num_students = num_students - 1 →
  (avg_score * num_students - dropped_score) / new_num_students = 64 :=
by
  sorry

end NUMINAMATH_GPT_new_average_score_l1431_143139


namespace NUMINAMATH_GPT_minimum_discount_l1431_143182

open Real

theorem minimum_discount (CP MP SP_min : ℝ) (profit_margin : ℝ) (discount : ℝ) :
  CP = 800 ∧ MP = 1200 ∧ SP_min = 960 ∧ profit_margin = 0.20 ∧
  MP * (1 - discount / 100) ≥ SP_min → discount = 20 :=
by
  intros h
  rcases h with ⟨h_cp, h_mp, h_sp_min, h_profit_margin, h_selling_price⟩
  simp [h_cp, h_mp, h_sp_min, h_profit_margin, sub_eq_self, div_eq_self] at *
  sorry

end NUMINAMATH_GPT_minimum_discount_l1431_143182


namespace NUMINAMATH_GPT_ellen_total_legos_l1431_143135

-- Conditions
def ellen_original_legos : ℝ := 2080.0
def ellen_winning_legos : ℝ := 17.0

-- Theorem statement
theorem ellen_total_legos : ellen_original_legos + ellen_winning_legos = 2097.0 :=
by
  -- The proof would go here, but we will use sorry to indicate it is skipped.
  sorry

end NUMINAMATH_GPT_ellen_total_legos_l1431_143135


namespace NUMINAMATH_GPT_opposite_of_neg_two_l1431_143124

-- Define what it means for 'b' to be the opposite of 'a'
def is_opposite (a b : Int) : Prop := a + b = 0

-- The theorem to be proved
theorem opposite_of_neg_two : is_opposite (-2) 2 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_two_l1431_143124


namespace NUMINAMATH_GPT_sqrt6_op_sqrt6_l1431_143113

variable (x y : ℝ)

noncomputable def op (x y : ℝ) := (x + y)^2 - (x - y)^2

theorem sqrt6_op_sqrt6 : ∀ (x y : ℝ), op (Real.sqrt 6) (Real.sqrt 6) = 24 := by
  sorry

end NUMINAMATH_GPT_sqrt6_op_sqrt6_l1431_143113


namespace NUMINAMATH_GPT_annual_interest_rate_last_year_l1431_143127

-- Define the conditions
def increased_by_ten_percent (r : ℝ) : ℝ := 1.10 * r

-- Statement of the problem
theorem annual_interest_rate_last_year (r : ℝ) (h : increased_by_ten_percent r = 0.11) : r = 0.10 :=
sorry

end NUMINAMATH_GPT_annual_interest_rate_last_year_l1431_143127


namespace NUMINAMATH_GPT_min_value_eq_9_l1431_143163

-- Defining the conditions
variable (a b : ℝ)
variable (ha : a > 0) (hb : b > 0)
variable (h_eq : a - 2 * b = 0)

-- The goal is to prove the minimum value of (1/a) + (4/b) is 9
theorem min_value_eq_9 (ha : a > 0) (hb : b > 0) (h_eq : a - 2 * b = 0) 
  : ∃ (m : ℝ), m = 9 ∧ (∀ x, x = 1/a + 4/b → x ≥ m) :=
sorry

end NUMINAMATH_GPT_min_value_eq_9_l1431_143163


namespace NUMINAMATH_GPT_problem1_problem2_l1431_143137

-- Problem 1
theorem problem1 (m n : ℚ) (h : m ≠ n) : 
  (m / (m - n)) + (n / (n - m)) = 1 := 
by
  -- Proof steps would go here
  sorry

-- Problem 2
theorem problem2 (x : ℚ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) : 
  (2 / (x^2 - 1)) / (1 / (x + 1)) = 2 / (x - 1) := 
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1431_143137


namespace NUMINAMATH_GPT_slope_of_line_l1431_143166

noncomputable def line_equation (x y : ℝ) : Prop := 4 * y + 2 * x = 10

theorem slope_of_line (x y : ℝ) (h : line_equation x y) : -1 / 2 = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_l1431_143166


namespace NUMINAMATH_GPT_total_paintable_area_correct_l1431_143121

namespace BarnPainting

-- Define the dimensions of the barn
def barn_width : ℕ := 12
def barn_length : ℕ := 15
def barn_height : ℕ := 6

-- Define the dimensions of the windows
def window_width : ℕ := 2
def window_height : ℕ := 3
def num_windows : ℕ := 2

-- Calculate the total number of square yards to be painted
def total_paintable_area : ℕ :=
  let wall1_area := barn_height * barn_width
  let wall2_area := barn_height * barn_length
  let wall_area := 2 * wall1_area + 2 * wall2_area
  let window_area := num_windows * (window_width * window_height)
  let painted_walls_area := wall_area - window_area
  let ceiling_area := barn_width * barn_length
  let total_area := 2 * painted_walls_area + ceiling_area
  total_area

theorem total_paintable_area_correct : total_paintable_area = 780 :=
  by sorry

end BarnPainting

end NUMINAMATH_GPT_total_paintable_area_correct_l1431_143121


namespace NUMINAMATH_GPT_initial_amount_of_milk_l1431_143153

theorem initial_amount_of_milk (M : ℝ) (h : 0 < M) (h2 : 0.10 * M = 0.05 * (M + 20)) : M = 20 := 
sorry

end NUMINAMATH_GPT_initial_amount_of_milk_l1431_143153


namespace NUMINAMATH_GPT_total_hunts_is_21_l1431_143136

-- Define the initial conditions
def Sam_hunts : Nat := 6
def Rob_hunts : Nat := Sam_hunts / 2
def Rob_Sam_total_hunt : Nat := Sam_hunts + Rob_hunts
def Mark_hunts : Nat := Rob_Sam_total_hunt / 3
def Peter_hunts : Nat := Mark_hunts * 3

-- The main theorem to prove
theorem total_hunts_is_21 : Sam_hunts + Rob_hunts + Mark_hunts + Peter_hunts = 21 :=
by
  sorry

end NUMINAMATH_GPT_total_hunts_is_21_l1431_143136


namespace NUMINAMATH_GPT_part1_part2_l1431_143183

def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Part (1) 
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ a) → -2 ≤ a ∧ a ≤ 1 := by
  sorry

-- Part (2)
theorem part2 (a : ℝ) : 
  (∀ x : ℝ, x ≥ -1 → f a x ≥ a) → -3 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1431_143183


namespace NUMINAMATH_GPT_quadratic_range_l1431_143119

open Real

def quadratic (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 5

theorem quadratic_range :
  ∀ x, -2 ≤ x ∧ x ≤ 2 → -8 ≤ quadratic x ∧ quadratic x ≤ 19 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_quadratic_range_l1431_143119


namespace NUMINAMATH_GPT_exists_long_segment_between_parabolas_l1431_143128

def parabola1 (x : ℝ) : ℝ :=
  x ^ 2

def parabola2 (x : ℝ) : ℝ :=
  x ^ 2 - 1

def in_between_parabolas (x y : ℝ) : Prop :=
  (parabola2 x) ≤ y ∧ y ≤ (parabola1 x)

theorem exists_long_segment_between_parabolas :
  ∃ (M1 M2: ℝ × ℝ), in_between_parabolas M1.1 M1.2 ∧ in_between_parabolas M2.1 M2.2 ∧ dist M1 M2 > 10^6 :=
sorry

end NUMINAMATH_GPT_exists_long_segment_between_parabolas_l1431_143128


namespace NUMINAMATH_GPT_acute_angle_range_l1431_143164

theorem acute_angle_range (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : Real.sin α < Real.cos α) : 0 < α ∧ α < π / 4 :=
sorry

end NUMINAMATH_GPT_acute_angle_range_l1431_143164


namespace NUMINAMATH_GPT_pure_imaginary_m_value_l1431_143160

theorem pure_imaginary_m_value (m : ℝ) (h₁ : m ^ 2 + m - 2 = 0) (h₂ : m ^ 2 - 1 ≠ 0) : m = -2 := by
  sorry

end NUMINAMATH_GPT_pure_imaginary_m_value_l1431_143160


namespace NUMINAMATH_GPT_arithmetic_avg_salary_technicians_l1431_143118

noncomputable def avg_salary_technicians_problem : Prop :=
  let average_salary_all := 8000
  let total_workers := 21
  let average_salary_rest := 6000
  let technician_count := 7
  let total_salary_all := average_salary_all * total_workers
  let total_salary_rest := average_salary_rest * (total_workers - technician_count)
  let total_salary_technicians := total_salary_all - total_salary_rest
  let average_salary_technicians := total_salary_technicians / technician_count
  average_salary_technicians = 12000

theorem arithmetic_avg_salary_technicians :
  avg_salary_technicians_problem :=
by {
  sorry -- Proof is omitted as per instructions.
}

end NUMINAMATH_GPT_arithmetic_avg_salary_technicians_l1431_143118


namespace NUMINAMATH_GPT_positive_slope_asymptote_l1431_143177

-- Define the foci points A and B and the given equation of the hyperbola
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (-3, 1)
def hyperbola_eqn (x y : ℝ) : Prop :=
  Real.sqrt ((x - 3)^2 + (y - 1)^2) - Real.sqrt ((x + 3)^2 + (y - 1)^2) = 4

-- State the theorem about the positive slope of the asymptote
theorem positive_slope_asymptote (x y : ℝ) (h : hyperbola_eqn x y) : 
  ∃ b a : ℝ, b = Real.sqrt 5 ∧ a = 2 ∧ (b / a) = Real.sqrt 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_positive_slope_asymptote_l1431_143177


namespace NUMINAMATH_GPT_triangle_sides_inequality_l1431_143192

theorem triangle_sides_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) (h4 : a + b + c ≤ 2) :
  -3 < (a^3 / b + b^3 / c + c^3 / a - a^3 / c - b^3 / a - c^3 / b) ∧ 
  (a^3 / b + b^3 / c + c^3 / a - a^3 / c - b^3 / a - c^3 / b) < 3 :=
by sorry

end NUMINAMATH_GPT_triangle_sides_inequality_l1431_143192


namespace NUMINAMATH_GPT_exists_n0_find_N_l1431_143190

noncomputable def f (x : ℝ) : ℝ := 1 / (2 - x)

-- Definition of the sequence {a_n}
def seq (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, a (n + 1) = f (a n)

-- Problem (1): Existence of n0
theorem exists_n0 (a : ℕ → ℝ) (h_seq : seq a) (h_a1 : a 1 = 3) : 
  ∃ n0 : ℕ, ∀ n ≥ n0, a (n + 1) > a n :=
  sorry

-- Problem (2): Smallest N
theorem find_N (a : ℕ → ℝ) (h_seq : seq a) (m : ℕ) (h_m : m > 1) 
  (h_a1 : 1 + 1 / (m : ℝ) < a 1 ∧ a 1 < m / (m - 1)) : 
  ∃ N : ℕ, ∀ n ≥ N, 0 < a n ∧ a n < 1 :=
  sorry

end NUMINAMATH_GPT_exists_n0_find_N_l1431_143190


namespace NUMINAMATH_GPT_science_club_election_l1431_143173

theorem science_club_election :
  let total_candidates := 20
  let past_officers := 10
  let non_past_officers := total_candidates - past_officers
  let positions := 6
  let total_ways := Nat.choose total_candidates positions
  let no_past_officer_ways := Nat.choose non_past_officers positions
  let exactly_one_past_officer_ways := past_officers * Nat.choose non_past_officers (positions - 1)
  total_ways - no_past_officer_ways - exactly_one_past_officer_ways = 36030 := by
    sorry

end NUMINAMATH_GPT_science_club_election_l1431_143173


namespace NUMINAMATH_GPT_percentage_of_students_owning_birds_l1431_143147

theorem percentage_of_students_owning_birds
    (total_students : ℕ) 
    (students_owning_birds : ℕ) 
    (h_total_students : total_students = 500) 
    (h_students_owning_birds : students_owning_birds = 75) : 
    (students_owning_birds * 100) / total_students = 15 := 
by 
    sorry

end NUMINAMATH_GPT_percentage_of_students_owning_birds_l1431_143147


namespace NUMINAMATH_GPT_trig_problem_l1431_143141

-- Translate the conditions and problems into Lean 4:
theorem trig_problem (α : ℝ) (h1 : Real.tan α = 2) :
    (2 * Real.sin α - 2 * Real.cos α) / (4 * Real.sin α - 9 * Real.cos α) = -2 := by
  sorry

end NUMINAMATH_GPT_trig_problem_l1431_143141


namespace NUMINAMATH_GPT_hyperbola_asymptote_focal_length_l1431_143101

theorem hyperbola_asymptote_focal_length (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : c = 2 * Real.sqrt 5) (h4 : b / a = 2) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_focal_length_l1431_143101


namespace NUMINAMATH_GPT_min_value_of_expression_l1431_143104

variable (a b c : ℝ)
variable (h1 : a + b + c = 1)
variable (h2 : 0 < a ∧ a < 1)
variable (h3 : 0 < b ∧ b < 1)
variable (h4 : 0 < c ∧ c < 1)
variable (h5 : 3 * a + 2 * b = 2)

theorem min_value_of_expression : (2 / a + 1 / (3 * b)) ≥ 16 / 3 := 
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l1431_143104


namespace NUMINAMATH_GPT_unattainable_y_l1431_143110

theorem unattainable_y (x : ℚ) (y : ℚ) (h : y = (1 - 2 * x) / (3 * x + 4)) (hx : x ≠ -4 / 3) : y ≠ -2 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_unattainable_y_l1431_143110


namespace NUMINAMATH_GPT_largest_consecutive_odd_sum_l1431_143176

theorem largest_consecutive_odd_sum (x : ℤ) (h : 20 * (x + 19) = 8000) : x + 38 = 419 := 
by
  sorry

end NUMINAMATH_GPT_largest_consecutive_odd_sum_l1431_143176


namespace NUMINAMATH_GPT_school_population_l1431_143140

variable (b g t a : ℕ)

theorem school_population (h1 : b = 2 * g) (h2 : g = 4 * t) (h3 : a = t / 2) : 
  b + g + t + a = 27 * b / 16 := by
  sorry

end NUMINAMATH_GPT_school_population_l1431_143140


namespace NUMINAMATH_GPT_weekly_exercise_time_l1431_143122

def milesWalked := 3
def walkingSpeed := 3 -- in miles per hour
def milesRan := 10
def runningSpeed := 5 -- in miles per hour
def daysInWeek := 7

theorem weekly_exercise_time : (milesWalked / walkingSpeed + milesRan / runningSpeed) * daysInWeek = 21 := 
by
  -- The actual proof part is intentionally omitted as per the instruction
  sorry

end NUMINAMATH_GPT_weekly_exercise_time_l1431_143122


namespace NUMINAMATH_GPT_frank_money_remaining_l1431_143106

-- Define the conditions
def cost_cheapest_lamp : ℕ := 20
def factor_most_expensive : ℕ := 3
def initial_money : ℕ := 90

-- Define the cost of the most expensive lamp
def cost_most_expensive_lamp : ℕ := cost_cheapest_lamp * factor_most_expensive

-- Define the money remaining after purchase
def money_remaining : ℕ := initial_money - cost_most_expensive_lamp

-- The theorem we need to prove
theorem frank_money_remaining : money_remaining = 30 := by
  sorry

end NUMINAMATH_GPT_frank_money_remaining_l1431_143106


namespace NUMINAMATH_GPT_rope_for_second_post_l1431_143132

theorem rope_for_second_post 
(r1 r2 r3 r4 : ℕ) 
(h_total : r1 + r2 + r3 + r4 = 70)
(h_r1 : r1 = 24)
(h_r3 : r3 = 14)
(h_r4 : r4 = 12) 
: r2 = 20 := 
by 
  sorry

end NUMINAMATH_GPT_rope_for_second_post_l1431_143132
