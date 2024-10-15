import Mathlib

namespace NUMINAMATH_GPT_find_compounding_frequency_l1861_186127

-- Lean statement defining the problem conditions and the correct answer

theorem find_compounding_frequency (P A : ℝ) (r t : ℝ) (hP : P = 12000) (hA : A = 13230) 
(hri : r = 0.10) (ht : t = 1) 
: ∃ (n : ℕ), A = P * (1 + r / n) ^ (n * t) ∧ n = 2 := 
by
  -- Definitions from the conditions
  have hP := hP
  have hA := hA
  have hr := hri
  have ht := ht
  
  -- Substitute known values
  use 2
  -- Show that the statement holds with n = 2
  sorry

end NUMINAMATH_GPT_find_compounding_frequency_l1861_186127


namespace NUMINAMATH_GPT_trench_dig_time_l1861_186103

theorem trench_dig_time (a b c d : ℝ) (h1 : a + b + c + d = 1/6)
  (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
  (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4) :
  a + b + c = 1 / 6 := sorry

end NUMINAMATH_GPT_trench_dig_time_l1861_186103


namespace NUMINAMATH_GPT_kaleb_balance_l1861_186126

theorem kaleb_balance (springEarnings : ℕ) (summerEarnings : ℕ) (suppliesCost : ℕ) (totalBalance : ℕ)
  (h1 : springEarnings = 4)
  (h2 : summerEarnings = 50)
  (h3 : suppliesCost = 4)
  (h4 : totalBalance = (springEarnings + summerEarnings) - suppliesCost) : totalBalance = 50 := by
  sorry

end NUMINAMATH_GPT_kaleb_balance_l1861_186126


namespace NUMINAMATH_GPT_units_digit_of_sum_of_cubes_l1861_186123

theorem units_digit_of_sum_of_cubes : 
  (24^3 + 42^3) % 10 = 2 := by
sorry

end NUMINAMATH_GPT_units_digit_of_sum_of_cubes_l1861_186123


namespace NUMINAMATH_GPT_find_m_l1861_186163

theorem find_m (x p q m : ℝ) 
    (h1 : 4 * p^2 + 9 * q^2 = 2) 
    (h2 : (1/2) * x + 3 * p * q = 1) 
    (h3 : ∀ x, x^2 + 2 * m * x - 3 * m + 1 ≥ 1) :
    m = -3 ∨ m = 1 :=
sorry

end NUMINAMATH_GPT_find_m_l1861_186163


namespace NUMINAMATH_GPT_positive_root_of_equation_l1861_186124

theorem positive_root_of_equation :
  ∃ a b : ℤ, (a + b * Real.sqrt 3)^3 - 5 * (a + b * Real.sqrt 3)^2 + 2 * (a + b * Real.sqrt 3) - Real.sqrt 3 = 0 ∧
    a + b * Real.sqrt 3 > 0 ∧
    (a + b * Real.sqrt 3) = 3 + Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_positive_root_of_equation_l1861_186124


namespace NUMINAMATH_GPT_smallest_z_in_arithmetic_and_geometric_progression_l1861_186137

theorem smallest_z_in_arithmetic_and_geometric_progression :
  ∃ x y z : ℤ, x < y ∧ y < z ∧ (2 * y = x + z) ∧ (z^2 = x * y) ∧ z = -2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_z_in_arithmetic_and_geometric_progression_l1861_186137


namespace NUMINAMATH_GPT_evaluate_expression_l1861_186157

theorem evaluate_expression (x y z : ℤ) (h1 : x = -2) (h2 : y = -4) (h3 : z = 3) :
  (5 * (x - y)^2 - x * z^2) / (z - y) = 38 / 7 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1861_186157


namespace NUMINAMATH_GPT_solve_system_l1861_186167

theorem solve_system : ∃ s t : ℝ, (11 * s + 7 * t = 240) ∧ (s = 1 / 2 * t + 3) ∧ (t = 414 / 25) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l1861_186167


namespace NUMINAMATH_GPT_microorganism_half_filled_time_l1861_186169

theorem microorganism_half_filled_time :
  (∀ x, 2^x = 2^9 ↔ x = 9) :=
by
  sorry

end NUMINAMATH_GPT_microorganism_half_filled_time_l1861_186169


namespace NUMINAMATH_GPT_neg_prop_l1861_186173

-- Definition of the proposition to be negated
def prop (x : ℝ) : Prop := x^2 + 2 * x + 5 = 0

-- Negation of the proposition
theorem neg_prop : ¬ (∃ x : ℝ, prop x) ↔ ∀ x : ℝ, ¬ prop x :=
by
  sorry

end NUMINAMATH_GPT_neg_prop_l1861_186173


namespace NUMINAMATH_GPT_sphere_radius_equals_4_l1861_186152

noncomputable def radius_of_sphere
  (sun_parallel : true)
  (meter_stick_height : ℝ)
  (meter_stick_shadow : ℝ)
  (sphere_shadow_distance : ℝ) : ℝ :=
if h : meter_stick_height / meter_stick_shadow = sphere_shadow_distance / 16 then
  4
else
  sorry

theorem sphere_radius_equals_4 
  (sun_parallel : true = true)
  (meter_stick_height : ℝ := 1)
  (meter_stick_shadow : ℝ := 4)
  (sphere_shadow_distance : ℝ := 16) : 
  radius_of_sphere sun_parallel meter_stick_height meter_stick_shadow sphere_shadow_distance = 4 :=
by
  simp [radius_of_sphere]
  sorry

end NUMINAMATH_GPT_sphere_radius_equals_4_l1861_186152


namespace NUMINAMATH_GPT_line_intersects_y_axis_at_origin_l1861_186121

theorem line_intersects_y_axis_at_origin 
  (x₁ y₁ x₂ y₂ : ℤ) 
  (h₁ : (x₁, y₁) = (3, 9)) 
  (h₂ : (x₂, y₂) = (-7, -21)) 
  : 
  ∃ y : ℤ, (0, y) = (0, 0) := by
  sorry

end NUMINAMATH_GPT_line_intersects_y_axis_at_origin_l1861_186121


namespace NUMINAMATH_GPT_carrots_chloe_l1861_186181

theorem carrots_chloe (c_i c_t c_p : ℕ) (H1 : c_i = 48) (H2 : c_t = 45) (H3 : c_p = 42) : 
  c_i - c_t + c_p = 45 := by
  sorry

end NUMINAMATH_GPT_carrots_chloe_l1861_186181


namespace NUMINAMATH_GPT_bus_ride_cost_l1861_186142

variable (cost_bus cost_train : ℝ)

-- Condition 1: cost_train = cost_bus + 2.35
#check (cost_train = cost_bus + 2.35)

-- Condition 2: cost_bus + cost_train = 9.85
#check (cost_bus + cost_train = 9.85)

theorem bus_ride_cost :
  (∃ (cost_bus cost_train : ℝ),
    cost_train = cost_bus + 2.35 ∧
    cost_bus + cost_train = 9.85) →
  cost_bus = 3.75 :=
sorry

end NUMINAMATH_GPT_bus_ride_cost_l1861_186142


namespace NUMINAMATH_GPT_not_solvable_det_three_times_l1861_186146

theorem not_solvable_det_three_times (a b c d : ℝ) (h : a * d - b * c = 5) :
  ¬∃ (x : ℝ), (3 * a + 1) * (3 * d + 1) - (3 * b + 1) * (3 * c + 1) = x :=
by {
  -- This is where the proof would go, but the problem states that it's not solvable with the given information.
  sorry
}

end NUMINAMATH_GPT_not_solvable_det_three_times_l1861_186146


namespace NUMINAMATH_GPT_greatest_value_x_l1861_186131

theorem greatest_value_x (x: ℤ) : 
  (∃ k: ℤ, (x^2 - 5 * x + 14) = k * (x - 4)) → x ≤ 14 :=
sorry

end NUMINAMATH_GPT_greatest_value_x_l1861_186131


namespace NUMINAMATH_GPT_pyarelal_loss_l1861_186165

theorem pyarelal_loss (P : ℝ) (total_loss : ℝ) (ashok_ratio pyarelal_ratio : ℝ)
  (h1 : ashok_ratio = 1/9) (h2 : pyarelal_ratio = 1)
  (h3 : total_loss = 2000) : (pyarelal_ratio / (ashok_ratio + pyarelal_ratio)) * total_loss = 1800 :=
by
  sorry

end NUMINAMATH_GPT_pyarelal_loss_l1861_186165


namespace NUMINAMATH_GPT_golf_balls_count_l1861_186198

theorem golf_balls_count (dozen_count : ℕ) (balls_per_dozen : ℕ) (total_balls : ℕ) 
  (h1 : dozen_count = 13) 
  (h2 : balls_per_dozen = 12) 
  (h3 : total_balls = dozen_count * balls_per_dozen) : 
  total_balls = 156 := 
sorry

end NUMINAMATH_GPT_golf_balls_count_l1861_186198


namespace NUMINAMATH_GPT_total_amount_paid_l1861_186192

def price_grapes (kg: ℕ) (rate: ℕ) : ℕ := kg * rate
def price_mangoes (kg: ℕ) (rate: ℕ) : ℕ := kg * rate
def price_pineapple (kg: ℕ) (rate: ℕ) : ℕ := kg * rate
def price_kiwi (kg: ℕ) (rate: ℕ) : ℕ := kg * rate

theorem total_amount_paid :
  price_grapes 14 54 + price_mangoes 10 62 + price_pineapple 8 40 + price_kiwi 5 30 = 1846 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l1861_186192


namespace NUMINAMATH_GPT_base_conversion_subtraction_l1861_186187

def base6_to_base10 (n : ℕ) : ℕ :=
3 * (6^2) + 2 * (6^1) + 5 * (6^0)

def base5_to_base10 (m : ℕ) : ℕ :=
2 * (5^2) + 3 * (5^1) + 1 * (5^0)

theorem base_conversion_subtraction : 
  base6_to_base10 325 - base5_to_base10 231 = 59 :=
by
  sorry

end NUMINAMATH_GPT_base_conversion_subtraction_l1861_186187


namespace NUMINAMATH_GPT_smallest_positive_x_for_maximum_l1861_186101

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 4) + Real.cos (x / 9)

theorem smallest_positive_x_for_maximum (x : ℝ) :
  (∀ k m : ℤ, x = 360 * (1 + k) ∧ x = 3600 * m ∧ 0 < x → x = 3600) :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_x_for_maximum_l1861_186101


namespace NUMINAMATH_GPT_suraj_next_innings_runs_l1861_186170

variable (A R : ℕ)

def suraj_average_eq (A : ℕ) : Prop :=
  A + 8 = 128

def total_runs_eq (A R : ℕ) : Prop :=
  9 * A + R = 10 * 128

theorem suraj_next_innings_runs :
  ∃ A : ℕ, suraj_average_eq A ∧ ∃ R : ℕ, total_runs_eq A R ∧ R = 200 := 
by
  sorry

end NUMINAMATH_GPT_suraj_next_innings_runs_l1861_186170


namespace NUMINAMATH_GPT_question1_question2_l1861_186172

-- Define the sets A and B as given in the problem
def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}
def B : Set ℝ := {x | x < -2 ∨ x > 5}

-- Lean statement for (1)
theorem question1 (m : ℝ) : 
  (A m ⊆ B) ↔ (m < 2 ∨ m > 4) :=
by
  sorry

-- Lean statement for (2)
theorem question2 (m : ℝ) : 
  (A m ∩ B = ∅) ↔ (m ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_question1_question2_l1861_186172


namespace NUMINAMATH_GPT_intersecting_lines_b_plus_m_l1861_186122

theorem intersecting_lines_b_plus_m :
  ∃ (m b : ℚ), (∀ x y : ℚ, y = m * x + 5 → y = 4 * x + b → (x, y) = (8, 14)) →
               b + m = -63 / 4 :=
by
  sorry

end NUMINAMATH_GPT_intersecting_lines_b_plus_m_l1861_186122


namespace NUMINAMATH_GPT_total_weight_CaBr2_l1861_186108

-- Definitions derived from conditions
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_Br : ℝ := 79.904
def mol_weight_CaBr2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_Br
def moles_CaBr2 : ℝ := 4

-- Theorem statement based on the problem and correct answer
theorem total_weight_CaBr2 : moles_CaBr2 * mol_weight_CaBr2 = 799.552 :=
by
  -- Prove the theorem step-by-step
  -- substitute the definition of mol_weight_CaBr2
  -- show lhs = rhs
  sorry

end NUMINAMATH_GPT_total_weight_CaBr2_l1861_186108


namespace NUMINAMATH_GPT_average_age_of_first_and_fifth_fastest_dogs_l1861_186100

-- Definitions based on the conditions
def first_dog_age := 10
def second_dog_age := first_dog_age - 2
def third_dog_age := second_dog_age + 4
def fourth_dog_age := third_dog_age / 2
def fifth_dog_age := fourth_dog_age + 20

-- Statement to prove
theorem average_age_of_first_and_fifth_fastest_dogs : 
  (first_dog_age + fifth_dog_age) / 2 = 18 := by
  -- Add your proof here
  sorry

end NUMINAMATH_GPT_average_age_of_first_and_fifth_fastest_dogs_l1861_186100


namespace NUMINAMATH_GPT_sum_of_digits_3n_l1861_186195

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem sum_of_digits_3n (n : ℕ) (hn1 : digit_sum n = 100) (hn2 : digit_sum (44 * n) = 800) : digit_sum (3 * n) = 300 := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_3n_l1861_186195


namespace NUMINAMATH_GPT_sector_area_l1861_186102

theorem sector_area (r l : ℝ) (h1 : l + 2 * r = 8) (h2 : l = 2 * r) : 
  (1 / 2) * l * r = 4 := 
by sorry

end NUMINAMATH_GPT_sector_area_l1861_186102


namespace NUMINAMATH_GPT_number_of_distinct_triangles_l1861_186125

-- Definition of the grid
def grid_points : List (ℕ × ℕ) := 
  [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1)]

-- Definition involving combination logic
def binomial (n k : ℕ) : ℕ := n.choose k

-- Count all possible combinations of 3 points
def total_combinations : ℕ := binomial 8 3

-- Count the degenerate cases (collinear points) in the grid
def degenerate_cases : ℕ := 2 * binomial 4 3

-- The required value of distinct triangles
def distinct_triangles : ℕ := total_combinations - degenerate_cases

theorem number_of_distinct_triangles :
  distinct_triangles = 48 :=
by
  sorry

end NUMINAMATH_GPT_number_of_distinct_triangles_l1861_186125


namespace NUMINAMATH_GPT_equations_not_equivalent_l1861_186140

theorem equations_not_equivalent :
  (∀ x, (2 * (x - 10) / (x^2 - 13 * x + 30) = 1 ↔ x = 5)) ∧ 
  (∃ x, x ≠ 5 ∧ (x^2 - 15 * x + 50 = 0)) :=
sorry

end NUMINAMATH_GPT_equations_not_equivalent_l1861_186140


namespace NUMINAMATH_GPT_max_oranges_to_teachers_l1861_186183

theorem max_oranges_to_teachers {n r : ℕ} (h1 : n % 8 = r) (h2 : r < 8) : r = 7 :=
sorry

end NUMINAMATH_GPT_max_oranges_to_teachers_l1861_186183


namespace NUMINAMATH_GPT_number_of_boys_is_90_l1861_186106

-- Define the conditions
variables (B G : ℕ)
axiom sum_condition : B + G = 150
axiom percentage_condition : G = (B / 150) * 100

-- State the theorem
theorem number_of_boys_is_90 : B = 90 :=
by
  -- We can skip the proof for now using sorry
  sorry

end NUMINAMATH_GPT_number_of_boys_is_90_l1861_186106


namespace NUMINAMATH_GPT_q_investment_l1861_186162

theorem q_investment (p_investment : ℕ) (ratio_pq : ℕ × ℕ) (profit_ratio : ℕ × ℕ) (hp : p_investment = 12000) (hpr : ratio_pq = (3, 5)) : 
  (∃ q_investment, q_investment = 20000) :=
  sorry

end NUMINAMATH_GPT_q_investment_l1861_186162


namespace NUMINAMATH_GPT_cow_problem_l1861_186184

noncomputable def problem_statement : Prop :=
  ∃ (F M : ℕ), F + M = 300 ∧
               (∃ S H : ℕ, S = 1/2 * F ∧ H = 1/2 * M ∧ S = H + 50) ∧
               F = 2 * M

theorem cow_problem : problem_statement :=
sorry

end NUMINAMATH_GPT_cow_problem_l1861_186184


namespace NUMINAMATH_GPT_sum_of_reciprocal_transformed_roots_l1861_186158

-- Define the polynomial f
def f (x : ℝ) : ℝ := 15 * x^3 - 35 * x^2 + 20 * x - 2

-- Define the condition that the roots are distinct real numbers between 0 and 1
def is_root (f : ℝ → ℝ) (x : ℝ) : Prop := f x = 0
def roots_between_0_and_1 (a b c : ℝ) : Prop := 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  0 < a ∧ a < 1 ∧ 
  0 < b ∧ b < 1 ∧ 
  0 < c ∧ c < 1 ∧
  is_root f a ∧ is_root f b ∧ is_root f c

-- The theorem representing the proof problem
theorem sum_of_reciprocal_transformed_roots (a b c : ℝ) 
  (h : roots_between_0_and_1 a b c) :
  (1/(1-a)) + (1/(1-b)) + (1/(1-c)) = 2/3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocal_transformed_roots_l1861_186158


namespace NUMINAMATH_GPT_neg_p_true_l1861_186178

theorem neg_p_true :
  (∃ x : ℝ, x^2 ≤ 0) :=
sorry

end NUMINAMATH_GPT_neg_p_true_l1861_186178


namespace NUMINAMATH_GPT_chemistry_more_than_physics_l1861_186143

noncomputable def M : ℕ := sorry
noncomputable def P : ℕ := sorry
noncomputable def C : ℕ := sorry
noncomputable def x : ℕ := sorry

theorem chemistry_more_than_physics :
  M + P = 20 ∧ C = P + x ∧ (M + C) / 2 = 20 → x = 20 :=
by
  sorry

end NUMINAMATH_GPT_chemistry_more_than_physics_l1861_186143


namespace NUMINAMATH_GPT_multiple_of_four_l1861_186161

open BigOperators

theorem multiple_of_four (n : ℕ) (x y z : Fin n → ℤ)
  (hx : ∀ i, x i = 1 ∨ x i = -1)
  (hy : ∀ i, y i = 1 ∨ y i = -1)
  (hz : ∀ i, z i = 1 ∨ z i = -1)
  (hxy : ∑ i, x i * y i = 0)
  (hxz : ∑ i, x i * z i = 0)
  (hyz : ∑ i, y i * z i = 0) :
  (n % 4 = 0) :=
sorry

end NUMINAMATH_GPT_multiple_of_four_l1861_186161


namespace NUMINAMATH_GPT_find_a_l1861_186185

theorem find_a
  (f : ℝ → ℝ)
  (h₁ : ∀ x, f x = 3 * Real.sin (2 * x - Real.pi / 3))
  (a : ℝ)
  (h₂ : 0 < a)
  (h₃ : a < Real.pi / 2)
  (h₄ : ∀ x, f (x + a) = f (-x + a)) :
  a = 5 * Real.pi / 12 :=
sorry

end NUMINAMATH_GPT_find_a_l1861_186185


namespace NUMINAMATH_GPT_probability_Q_within_2_of_origin_eq_pi_div_9_l1861_186119

noncomputable def probability_within_circle (π : ℝ) : ℝ :=
  let area_of_square := (2 * 3)^2
  let area_of_circle := π * 2^2
  area_of_circle / area_of_square

theorem probability_Q_within_2_of_origin_eq_pi_div_9 :
  probability_within_circle Real.pi = Real.pi / 9 :=
by
  sorry

end NUMINAMATH_GPT_probability_Q_within_2_of_origin_eq_pi_div_9_l1861_186119


namespace NUMINAMATH_GPT_percentage_of_a_is_4b_l1861_186135

variable (a b : ℝ)

theorem percentage_of_a_is_4b (h : a = 1.2 * b) : 4 * b = (10 / 3) * a := 
by 
    sorry

end NUMINAMATH_GPT_percentage_of_a_is_4b_l1861_186135


namespace NUMINAMATH_GPT_solve_xy_l1861_186159

theorem solve_xy (x y : ℕ) :
  (x^2 + (x + y)^2 = (x + 9)^2) ↔ (x = 0 ∧ y = 9) ∨ (x = 8 ∧ y = 7) ∨ (x = 20 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_xy_l1861_186159


namespace NUMINAMATH_GPT_smallest_period_sum_l1861_186134

noncomputable def smallest_positive_period (f : ℝ → ℝ) (g : ℝ → ℝ): ℝ → ℝ :=
λ x => f x + g x

theorem smallest_period_sum
  (f g : ℝ → ℝ)
  (m n : ℕ)
  (hf : ∀ x, f (x + m) = f x)
  (hg : ∀ x, g (x + n) = g x)
  (hm : m > 1)
  (hn : n > 1)
  (hgcd : Nat.gcd m n = 1)
  : ∃ T, T > 0 ∧ (∀ x, smallest_positive_period f g (x + T) = smallest_positive_period f g x) ∧ T = m * n := by
  sorry

end NUMINAMATH_GPT_smallest_period_sum_l1861_186134


namespace NUMINAMATH_GPT_initial_people_in_gym_l1861_186174

variables (W A S : ℕ)

theorem initial_people_in_gym (h1 : (W - 3 + 2 - 3 + 4 - 2 + 1 = W + 1))
                              (h2 : (A + 2 - 1 + 3 - 3 + 1 = A + 2))
                              (h3 : (S + 1 - 2 + 1 + 3 - 2 + 2 = S + 3))
                              (final_total : (W + 1) + (A + 2) + (S + 3) + 2 = 30) :
  W + A + S = 22 :=
by 
  sorry

end NUMINAMATH_GPT_initial_people_in_gym_l1861_186174


namespace NUMINAMATH_GPT_problem_statement_l1861_186120

-- Define the arithmetic sequence and the conditions
noncomputable def a : ℕ → ℝ := sorry
axiom a_arith_seq : ∃ d : ℝ, ∀ n m : ℕ, a (n + m) = a n + m • d
axiom condition : a 4 + a 10 + a 16 = 30

-- State the theorem
theorem problem_statement : a 18 - 2 * a 14 = -10 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1861_186120


namespace NUMINAMATH_GPT_earl_stuff_rate_l1861_186118

variable (E L : ℕ)

-- Conditions
def ellen_rate : Prop := L = (2 * E) / 3
def combined_rate : Prop := E + L = 60

-- Main statement
theorem earl_stuff_rate (h1 : ellen_rate E L) (h2 : combined_rate E L) : E = 36 := by
  sorry

end NUMINAMATH_GPT_earl_stuff_rate_l1861_186118


namespace NUMINAMATH_GPT_correct_investment_allocation_l1861_186110

noncomputable def investment_division (x : ℤ) : Prop :=
  let s := 2000
  let w := 500
  let rogers_investment := 2500
  let total_initial_capital := (5 / 2 : ℚ) * x
  let new_total_capital := total_initial_capital + rogers_investment
  let equal_share := new_total_capital / 3
  s + w = rogers_investment ∧ 
  (3 / 2 : ℚ) * x + s = equal_share ∧ 
  x + w = equal_share

theorem correct_investment_allocation (x : ℤ) (hx : 3 * x % 2 = 0) :
  x > 0 ∧ investment_division x :=
by
  sorry

end NUMINAMATH_GPT_correct_investment_allocation_l1861_186110


namespace NUMINAMATH_GPT_polynomial_has_real_root_l1861_186141

theorem polynomial_has_real_root (b : ℝ) : ∃ x : ℝ, (x^4 + b * x^3 + 2 * x^2 + b * x - 2 = 0) := sorry

end NUMINAMATH_GPT_polynomial_has_real_root_l1861_186141


namespace NUMINAMATH_GPT_smaller_solution_of_quadratic_l1861_186113

theorem smaller_solution_of_quadratic :
  (∃ x y : ℝ, x ≠ y ∧ (x^2 - 13 * x + 36 = 0) ∧ (y^2 - 13 * y + 36 = 0) ∧ min x y = 4) :=
sorry

end NUMINAMATH_GPT_smaller_solution_of_quadratic_l1861_186113


namespace NUMINAMATH_GPT_vector_calculation_l1861_186114

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (1, -1)

def vector_operation (a b : ℝ × ℝ) : ℝ × ℝ :=
(3 * a.1 - 2 * b.1, 3 * a.2 - 2 * b.2)

theorem vector_calculation : vector_operation vector_a vector_b = (1, 5) :=
by sorry

end NUMINAMATH_GPT_vector_calculation_l1861_186114


namespace NUMINAMATH_GPT_problem1_problem2_l1861_186109

-- Proof problem 1
theorem problem1 (x : ℝ) : (x - 1)^2 + x * (3 - x) = x + 1 := sorry

-- Proof problem 2
theorem problem2 (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ -2) : (a - 2) / (a - 1) / (a + 1 - 3 / (a - 1)) = 1 / (a + 2) := sorry

end NUMINAMATH_GPT_problem1_problem2_l1861_186109


namespace NUMINAMATH_GPT_julia_played_more_kids_l1861_186191

variable (kidsPlayedMonday : Nat) (kidsPlayedTuesday : Nat)

theorem julia_played_more_kids :
  kidsPlayedMonday = 11 →
  kidsPlayedTuesday = 12 →
  kidsPlayedTuesday - kidsPlayedMonday = 1 :=
by
  intros hMonday hTuesday
  sorry

end NUMINAMATH_GPT_julia_played_more_kids_l1861_186191


namespace NUMINAMATH_GPT_problem_composite_for_n_geq_9_l1861_186176

theorem problem_composite_for_n_geq_9 (n : ℤ) (h : n ≥ 9) : ∃ k m : ℤ, (2 ≤ k ∧ 2 ≤ m ∧ n + 7 = k * m) :=
by
  sorry

end NUMINAMATH_GPT_problem_composite_for_n_geq_9_l1861_186176


namespace NUMINAMATH_GPT_dogs_in_pet_shop_l1861_186160

variable (D C B x : ℕ)

theorem dogs_in_pet_shop 
  (h1 : D = 7 * x) 
  (h2 : B = 8 * x)
  (h3 : D + B = 330) : 
  D = 154 :=
by
  sorry

end NUMINAMATH_GPT_dogs_in_pet_shop_l1861_186160


namespace NUMINAMATH_GPT_total_population_l1861_186171

theorem total_population (b g t : ℕ) (h₁ : b = 6 * g) (h₂ : g = 5 * t) :
  b + g + t = 36 * t :=
by
  sorry

end NUMINAMATH_GPT_total_population_l1861_186171


namespace NUMINAMATH_GPT_Billy_Reads_3_Books_l1861_186168

theorem Billy_Reads_3_Books 
    (weekend_days : ℕ) 
    (hours_per_day : ℕ) 
    (reading_percentage : ℕ) 
    (pages_per_hour : ℕ) 
    (pages_per_book : ℕ) : 
    (weekend_days = 2) ∧ 
    (hours_per_day = 8) ∧ 
    (reading_percentage = 25) ∧ 
    (pages_per_hour = 60) ∧ 
    (pages_per_book = 80) → 
    ((weekend_days * hours_per_day * reading_percentage / 100 * pages_per_hour) / pages_per_book = 3) :=
by
  intros
  sorry

end NUMINAMATH_GPT_Billy_Reads_3_Books_l1861_186168


namespace NUMINAMATH_GPT_absolute_value_and_power_sum_l1861_186147

theorem absolute_value_and_power_sum :
  |(-4 : ℤ)| + (3 - Real.pi)^0 = 5 := by
  sorry

end NUMINAMATH_GPT_absolute_value_and_power_sum_l1861_186147


namespace NUMINAMATH_GPT_symmetry_probability_l1861_186155

-- Define the setting of the problem
def grid_points : ℕ := 121
def grid_size : ℕ := 11
def center_point : (ℕ × ℕ) := (6, 6)
def total_points : ℕ := grid_points - 1
def symmetric_lines : ℕ := 4
def points_per_line : ℕ := 10
def total_symmetric_points : ℕ := symmetric_lines * points_per_line
def probability : ℚ := total_symmetric_points / total_points

-- Theorem statement
theorem symmetry_probability 
  (hp: grid_points = 121) 
  (hs: grid_size = 11) 
  (hc: center_point = (6, 6))
  (htp: total_points = 120)
  (hsl: symmetric_lines = 4)
  (hpl: points_per_line = 10)
  (htsp: total_symmetric_points = 40)
  (hp: probability = 1 / 3) : 
  probability = 1 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_symmetry_probability_l1861_186155


namespace NUMINAMATH_GPT_monomial_properties_l1861_186150

theorem monomial_properties (a b : ℕ) (h : a = 2 ∧ b = 1) : 
  (2 * a ^ 2 * b = 2 * (a ^ 2) * b) ∧ (2 = 2) ∧ ((2 + 1) = 3) :=
by
  sorry

end NUMINAMATH_GPT_monomial_properties_l1861_186150


namespace NUMINAMATH_GPT_angle_hyperbola_l1861_186133

theorem angle_hyperbola (a b : ℝ) (e : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (hyperbola_eq : ∀ (x y : ℝ), ((x^2)/(a^2) - (y^2)/(b^2) = 1)) 
  (eccentricity_eq : e = 2 + Real.sqrt 6 - Real.sqrt 3 - Real.sqrt 2) :
  ∃ α : ℝ, α = 15 :=
by
  sorry

end NUMINAMATH_GPT_angle_hyperbola_l1861_186133


namespace NUMINAMATH_GPT_check_double_root_statements_l1861_186148

-- Condition Definitions
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ r : ℝ, a * r^2 + b * r + c = 0 ∧ a * (2 * r)^2 + b * (2 * r) + c = 0

-- Statement ①
def statement_1 : Prop := ¬is_double_root_equation 1 2 (-8)

-- Statement ②
def statement_2 : Prop := is_double_root_equation 1 (-3) 2

-- Statement ③
def statement_3 (m n : ℝ) : Prop := 
  (∃ r : ℝ, (r - 2) * (m * r + n) = 0 ∧ (m * (2 * r) + n = 0) ∧ r = 2) → 4 * m^2 + 5 * m * n + n^2 = 0

-- Statement ④
def statement_4 (p q : ℝ) : Prop := 
  (p * q = 2 → is_double_root_equation p 3 q)

-- Main proof problem statement
theorem check_double_root_statements (m n p q : ℝ) : 
  statement_1 ∧ statement_2 ∧ statement_3 m n ∧ statement_4 p q :=
by
  sorry

end NUMINAMATH_GPT_check_double_root_statements_l1861_186148


namespace NUMINAMATH_GPT_range_of_m_l1861_186179

theorem range_of_m (m : ℝ) (x : ℝ) 
  (h1 : ∀ x : ℝ, -1 < x ∧ x < 4 → x > 2 * m^2 - 3)
  (h2 : ¬ (∀ x : ℝ, x > 2 * m^2 - 3 → -1 < x ∧ x < 4))
  :
  -1 ≤ m ∧ m ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1861_186179


namespace NUMINAMATH_GPT_smallest_value_expression_l1861_186196

theorem smallest_value_expression (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 5) :
  ∃ m, m = y ∧ m = 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_expression_l1861_186196


namespace NUMINAMATH_GPT_masking_tape_needed_l1861_186180

def wall1_width : ℝ := 4
def wall1_count : ℕ := 2
def wall2_width : ℝ := 6
def wall2_count : ℕ := 2
def door_width : ℝ := 2
def door_count : ℕ := 1
def window_width : ℝ := 1.5
def window_count : ℕ := 2

def total_width_of_walls : ℝ := (wall1_count * wall1_width) + (wall2_count * wall2_width)
def total_width_of_door_and_windows : ℝ := (door_count * door_width) + (window_count * window_width)

theorem masking_tape_needed : total_width_of_walls - total_width_of_door_and_windows = 15 := by
  sorry

end NUMINAMATH_GPT_masking_tape_needed_l1861_186180


namespace NUMINAMATH_GPT_find_digit_A_l1861_186151

def sum_of_digits_divisible_by_3 (A : ℕ) : Prop :=
  (2 + A + 3) % 3 = 0

theorem find_digit_A (A : ℕ) (hA : sum_of_digits_divisible_by_3 A) : A = 1 ∨ A = 4 :=
  sorry

end NUMINAMATH_GPT_find_digit_A_l1861_186151


namespace NUMINAMATH_GPT_percentage_is_36_point_4_l1861_186156

def part : ℝ := 318.65
def whole : ℝ := 875.3

theorem percentage_is_36_point_4 : (part / whole) * 100 = 36.4 := 
by sorry

end NUMINAMATH_GPT_percentage_is_36_point_4_l1861_186156


namespace NUMINAMATH_GPT_necessary_and_sufficient_l1861_186105

theorem necessary_and_sufficient (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ¬ ((a > 0 ∧ b > 0 → ab < (a + b) / 2 ^ 2) 
  ∧ (ab < (a + b) / 2 ^ 2 → a > 0 ∧ b > 0)) := 
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_l1861_186105


namespace NUMINAMATH_GPT_clothing_price_l1861_186197

theorem clothing_price
  (total_spent : ℕ)
  (num_pieces : ℕ)
  (price_piece_1 : ℕ)
  (price_piece_2 : ℕ)
  (num_remaining_pieces : ℕ)
  (total_remaining_pieces_price : ℕ)
  (price_remaining_piece : ℕ) :
  total_spent = 610 →
  num_pieces = 7 →
  price_piece_1 = 49 →
  price_piece_2 = 81 →
  num_remaining_pieces = 5 →
  total_spent = price_piece_1 + price_piece_2 + total_remaining_pieces_price →
  total_remaining_pieces_price = price_remaining_piece * num_remaining_pieces →
  price_remaining_piece = 96 :=
by
  intros h_total_spent h_num_pieces h_price_piece_1 h_price_piece_2 h_num_remaining_pieces h_total_remaining_price h_price_remaining_piece
  sorry

end NUMINAMATH_GPT_clothing_price_l1861_186197


namespace NUMINAMATH_GPT_find_factor_l1861_186111

theorem find_factor {n f : ℝ} (h1 : n = 10) (h2 : f * (2 * n + 8) = 84) : f = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_factor_l1861_186111


namespace NUMINAMATH_GPT_vector_a_properties_l1861_186115

-- Definitions of the points in space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Definition of vector subtraction to find the vector between two points
def vector_sub (p1 p2 : Point3D) : Point3D :=
  { x := p2.x - p1.x, y := p2.y - p1.y, z := p2.z - p1.z }

-- Definition of dot product for vectors
def dot_product (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

-- Definition of vector magnitude squared for vectors
def magnitude_squared (v : Point3D) : ℝ :=
  v.x * v.x + v.y * v.y + v.z * v.z

-- Main theorem statement
theorem vector_a_properties :
  let A := {x := 0, y := 2, z := 3}
  let B := {x := -2, y := 1, z := 6}
  let C := {x := 1, y := -1, z := 5}
  let AB := vector_sub A B
  let AC := vector_sub A C
  ∀ (a : Point3D), 
    (magnitude_squared a = 3) → 
    (dot_product a AB = 0) → 
    (dot_product a AC = 0) → 
    (a = {x := 1, y := 1, z := 1} ∨ a = {x := -1, y := -1, z := -1}) := 
by
  intros A B C AB AC a ha_magnitude ha_perpendicular_AB ha_perpendicular_AC
  sorry

end NUMINAMATH_GPT_vector_a_properties_l1861_186115


namespace NUMINAMATH_GPT_grid_midpoint_exists_l1861_186166

theorem grid_midpoint_exists (points : Fin 5 → ℤ × ℤ) :
  ∃ i j : Fin 5, i ≠ j ∧ (points i).fst % 2 = (points j).fst % 2 ∧ (points i).snd % 2 = (points j).snd % 2 :=
by 
  sorry

end NUMINAMATH_GPT_grid_midpoint_exists_l1861_186166


namespace NUMINAMATH_GPT_find_a_l1861_186149

variable (a : ℕ) (N : ℕ)
variable (h1 : Nat.gcd (2 * a + 1) (2 * a + 2) = 1) 
variable (h2 : Nat.gcd (2 * a + 1) (2 * a + 3) = 1)
variable (h3 : Nat.gcd (2 * a + 2) (2 * a + 3) = 2)
variable (hN : N = Nat.lcm (2 * a + 1) (Nat.lcm (2 * a + 2) (2 * a + 3)))
variable (hDiv : (2 * a + 4) ∣ N)

theorem find_a (h_pos : a > 0) : a = 1 :=
by
  -- Lean proof code will go here
  sorry

end NUMINAMATH_GPT_find_a_l1861_186149


namespace NUMINAMATH_GPT_sides_of_right_triangle_l1861_186190

theorem sides_of_right_triangle (r : ℝ) (a b c : ℝ) 
  (h_area : (2 / (2 / r)) * 2 = 2 * r) 
  (h_right : a^2 + b^2 = c^2) :
  (a = r ∧ b = (4 / 3) * r ∧ c = (5 / 3) * r) ∨
  (b = r ∧ a = (4 / 3) * r ∧ c = (5 / 3) * r) :=
sorry

end NUMINAMATH_GPT_sides_of_right_triangle_l1861_186190


namespace NUMINAMATH_GPT_ladybugs_without_spots_l1861_186112

-- Defining the conditions given in the problem
def total_ladybugs : ℕ := 67082
def ladybugs_with_spots : ℕ := 12170

-- Proving the number of ladybugs without spots
theorem ladybugs_without_spots : total_ladybugs - ladybugs_with_spots = 54912 := by
  sorry

end NUMINAMATH_GPT_ladybugs_without_spots_l1861_186112


namespace NUMINAMATH_GPT_probability_not_snowing_l1861_186136

theorem probability_not_snowing (P_snowing : ℚ) (h : P_snowing = 2/7) :
  (1 - P_snowing) = 5/7 :=
sorry

end NUMINAMATH_GPT_probability_not_snowing_l1861_186136


namespace NUMINAMATH_GPT_chord_line_equation_l1861_186175

/-- 
  Given the parabola y^2 = 4x and a chord AB 
  that exactly bisects at point P(1,1), prove 
  that the equation of the line on which chord AB lies is 2x - y - 1 = 0.
-/
theorem chord_line_equation (x y : ℝ) 
  (hx : y^2 = 4 * x)
  (bisect : ∃ A B : ℝ × ℝ, 
             (A.1^2 = 4 * A.2) ∧ (B.1^2 = 4 * B.2) ∧
             (A.1 + B.1 = 2 * 1) ∧ (A.2 + B.2 = 2 * 1)) :
  2 * x - y - 1 = 0 := sorry

end NUMINAMATH_GPT_chord_line_equation_l1861_186175


namespace NUMINAMATH_GPT_lunch_special_cost_l1861_186144

theorem lunch_special_cost (total_bill : ℕ) (num_people : ℕ) (cost_per_lunch_special : ℕ)
  (h1 : total_bill = 24) 
  (h2 : num_people = 3) 
  (h3 : cost_per_lunch_special = total_bill / num_people) : 
  cost_per_lunch_special = 8 := 
by
  sorry

end NUMINAMATH_GPT_lunch_special_cost_l1861_186144


namespace NUMINAMATH_GPT_conditional_probability_age_30_40_female_l1861_186164

noncomputable def total_people : ℕ := 350
noncomputable def total_females : ℕ := 180
noncomputable def females_30_40 : ℕ := 50

theorem conditional_probability_age_30_40_female :
  (females_30_40 : ℚ) / total_females = 5 / 18 :=
by
  sorry

end NUMINAMATH_GPT_conditional_probability_age_30_40_female_l1861_186164


namespace NUMINAMATH_GPT_ivan_ivanovich_increase_l1861_186154

variable (p v s i : ℝ)
variable (k : ℝ)

-- Conditions
def initial_shares_sum := p + v + s + i = 1
def petya_doubles := 2 * p + v + s + i = 1.3
def vanya_doubles := p + 2 * v + s + i = 1.4
def sergey_triples := p + v + 3 * s + i = 1.2

-- Target statement to be proved
theorem ivan_ivanovich_increase (hp : p = 0.3) (hv : v = 0.4) (hs : s = 0.1)
  (hi : i = 0.2) (k : ℝ) : k * i > 0.75 → k > 3.75 :=
sorry

end NUMINAMATH_GPT_ivan_ivanovich_increase_l1861_186154


namespace NUMINAMATH_GPT_ratio_a_b_l1861_186128

theorem ratio_a_b (a b c : ℝ) (h1 : a * (-1) ^ 2 + b * (-1) + c = 1) (h2 : a * 3 ^ 2 + b * 3 + c = 1) : 
  a / b = -2 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_a_b_l1861_186128


namespace NUMINAMATH_GPT_unique_positive_integer_divisibility_l1861_186189

theorem unique_positive_integer_divisibility (n : ℕ) (h : n > 0) : 
  (5^(n-1) + 3^(n-1)) ∣ (5^n + 3^n) ↔ n = 1 :=
by
  sorry

end NUMINAMATH_GPT_unique_positive_integer_divisibility_l1861_186189


namespace NUMINAMATH_GPT_equivalent_proof_l1861_186139

theorem equivalent_proof :
  let a := 4
  let b := Real.sqrt 17 - a
  b^2020 * (a + Real.sqrt 17)^2021 = Real.sqrt 17 + 4 :=
by
  let a := 4
  let b := Real.sqrt 17 - a
  sorry

end NUMINAMATH_GPT_equivalent_proof_l1861_186139


namespace NUMINAMATH_GPT_num_integer_solutions_l1861_186132

def circle_center := (3, 3)
def circle_radius := 10

theorem num_integer_solutions :
  (∃ f : ℕ, f = 15) :=
sorry

end NUMINAMATH_GPT_num_integer_solutions_l1861_186132


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l1861_186199

def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1)
theorem part1_solution_set (x : ℝ) :
  ∀ x, f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 := sorry

-- Part (2)
theorem part2_range_of_a (a : ℝ) :
  (∀ x, f x a ≥ 4) ↔ (a ≤ -1 ∨ a ≥ 3) := sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l1861_186199


namespace NUMINAMATH_GPT_initially_calculated_average_height_l1861_186153

/-- Suppose the average height of 20 students was initially calculated incorrectly. Later, it was found that one student's height 
was incorrectly recorded as 151 cm instead of 136 cm. Given the actual average height of the students is 174.25 cm, prove that the 
initially calculated average height was 173.5 cm. -/
theorem initially_calculated_average_height
  (initial_avg actual_avg : ℝ)
  (num_students : ℕ)
  (incorrect_height correct_height : ℝ)
  (h_avg : actual_avg = 174.25)
  (h_students : num_students = 20)
  (h_incorrect : incorrect_height = 151)
  (h_correct : correct_height = 136)
  (h_total_actual : num_students * actual_avg = num_students * initial_avg + incorrect_height - correct_height) :
  initial_avg = 173.5 :=
by
  sorry

end NUMINAMATH_GPT_initially_calculated_average_height_l1861_186153


namespace NUMINAMATH_GPT_final_retail_price_l1861_186177

theorem final_retail_price (wholesale_price markup_percentage discount_percentage desired_profit_percentage : ℝ)
  (h_wholesale : wholesale_price = 90)
  (h_markup : markup_percentage = 1)
  (h_discount : discount_percentage = 0.2)
  (h_desired_profit : desired_profit_percentage = 0.6) :
  let initial_retail_price := wholesale_price + (wholesale_price * markup_percentage)
  let discount_amount := initial_retail_price * discount_percentage
  let final_retail_price := initial_retail_price - discount_amount
  final_retail_price = 144 ∧ final_retail_price = wholesale_price + (wholesale_price * desired_profit_percentage) := by
 sorry

end NUMINAMATH_GPT_final_retail_price_l1861_186177


namespace NUMINAMATH_GPT_f_satisfies_equation_l1861_186182

noncomputable def f (x : ℝ) : ℝ := (20 / 3) * x * (Real.sqrt (1 - x^2))

theorem f_satisfies_equation (f : ℝ → ℝ) :
  (∀ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), 2 * f (Real.sin x * -1) + 3 * f (Real.sin x) = 4 * Real.sin x * Real.cos x) →
  (∀ x ∈ Set.Icc (-Real.sqrt 2 / 2) (Real.sqrt 2 / 2), f x = (20 / 3) * x * (Real.sqrt (1 - x^2))) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_f_satisfies_equation_l1861_186182


namespace NUMINAMATH_GPT_conic_is_pair_of_lines_l1861_186117

-- Define the specific conic section equation
def conic_eq (x y : ℝ) : Prop := 9 * x^2 - 36 * y^2 = 0

-- State the theorem
theorem conic_is_pair_of_lines : ∀ x y : ℝ, conic_eq x y ↔ (x = 2 * y ∨ x = -2 * y) :=
by
  -- Sorry is placed to denote that proof steps are omitted in this statement
  sorry

end NUMINAMATH_GPT_conic_is_pair_of_lines_l1861_186117


namespace NUMINAMATH_GPT_find_positive_integer_k_l1861_186188

theorem find_positive_integer_k (p : ℕ) (hp : Prime p) (hp2 : Odd p) : 
  ∃ k : ℕ, k > 0 ∧ ∃ n : ℕ, n * n = k - p * k ∧ k = ((p + 1) * (p + 1)) / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_positive_integer_k_l1861_186188


namespace NUMINAMATH_GPT_train_speed_l1861_186104

theorem train_speed (train_length platform_length total_time : ℕ) 
  (h_train_length : train_length = 150) 
  (h_platform_length : platform_length = 250) 
  (h_total_time : total_time = 8) : 
  (train_length + platform_length) / total_time = 50 := 
by
  -- Proof goes here
  -- Given: train_length = 150, platform_length = 250, total_time = 8
  -- We need to prove: (train_length + platform_length) / total_time = 50
  -- So we calculate
  --  (150 + 250)/8 = 400/8 = 50
  sorry

end NUMINAMATH_GPT_train_speed_l1861_186104


namespace NUMINAMATH_GPT_bus_probability_l1861_186130

/-- A bus arrives randomly between 3:00 and 4:00, waits for 15 minutes, and then leaves. 
Sarah also arrives randomly between 3:00 and 4:00. Prove the probability that the bus 
will be there when Sarah arrives is 4275/7200. -/
theorem bus_probability : (4275 : ℚ) / 7200 = (4275 / 7200) :=
by 
  sorry

end NUMINAMATH_GPT_bus_probability_l1861_186130


namespace NUMINAMATH_GPT_prob_at_least_one_hit_correct_prob_exactly_two_hits_correct_l1861_186138

-- Define the probabilities that A, B, and C hit the target
def prob_A := 0.7
def prob_B := 0.6
def prob_C := 0.5

-- Define the probabilities that A, B, and C miss the target
def miss_A := 1 - prob_A
def miss_B := 1 - prob_B
def miss_C := 1 - prob_C

-- Probability that no one hits the target
def prob_no_hits := miss_A * miss_B * miss_C

-- Probability that at least one person hits the target
def prob_at_least_one_hit := 1 - prob_no_hits

-- Probabilities for the cases where exactly two people hit the target:
def prob_A_B_hits := prob_A * prob_B * miss_C
def prob_A_C_hits := prob_A * miss_B * prob_C
def prob_B_C_hits := miss_A * prob_B * prob_C

-- Probability that exactly two people hit the target
def prob_exactly_two_hits := prob_A_B_hits + prob_A_C_hits + prob_B_C_hits

-- Theorem statement to prove the probabilities match given conditions
theorem prob_at_least_one_hit_correct : prob_at_least_one_hit = 0.94 := by
  sorry

theorem prob_exactly_two_hits_correct : prob_exactly_two_hits = 0.44 := by
  sorry

end NUMINAMATH_GPT_prob_at_least_one_hit_correct_prob_exactly_two_hits_correct_l1861_186138


namespace NUMINAMATH_GPT_enclosed_area_abs_x_abs_3y_eq_12_l1861_186116

theorem enclosed_area_abs_x_abs_3y_eq_12 : 
  let f (x y : ℝ) := |x| + |3 * y|
  ∃ (A : ℝ), ∀ (x y : ℝ), f x y = 12 → A = 96 := 
sorry

end NUMINAMATH_GPT_enclosed_area_abs_x_abs_3y_eq_12_l1861_186116


namespace NUMINAMATH_GPT_xy_yx_eq_zy_yz_eq_xz_zx_l1861_186145

theorem xy_yx_eq_zy_yz_eq_xz_zx 
  (x y z : ℝ) 
  (h : x * (y + z - x) / x = y * (z + x - y) / y ∧ y * (z + x - y) / y = z * (x + y - z) / z): 
  x ^ y * y ^ x = z ^ y * y ^ z ∧ z ^ y * y ^ z = x ^ z * z ^ x :=
by
  sorry

end NUMINAMATH_GPT_xy_yx_eq_zy_yz_eq_xz_zx_l1861_186145


namespace NUMINAMATH_GPT_value_of_k_l1861_186194

theorem value_of_k (k : ℕ) : (∃ b : ℕ, x^2 - 20 * x + k = (x + b)^2) → k = 100 := by
  sorry

end NUMINAMATH_GPT_value_of_k_l1861_186194


namespace NUMINAMATH_GPT_ben_less_than_jack_l1861_186186

def jack_amount := 26
def total_amount := 50
def eric_ben_difference := 10

theorem ben_less_than_jack (E B J : ℕ) (h1 : E = B - eric_ben_difference) (h2 : J = jack_amount) (h3 : E + B + J = total_amount) :
  J - B = 9 :=
by sorry

end NUMINAMATH_GPT_ben_less_than_jack_l1861_186186


namespace NUMINAMATH_GPT_hyperbola_equation_l1861_186193

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (asymptote_eq : ∀ x y, 3*x + 4*y = 0 → y = (-3/4) * x)
  (focus_eq : (0, 5) = (0, 5)) :
  ∃ a b : ℝ, a = 3 ∧ b = 4 ∧ (∀ y x, (y^2 / 9 - x^2 / 16 = 1)) :=
sorry

end NUMINAMATH_GPT_hyperbola_equation_l1861_186193


namespace NUMINAMATH_GPT_trapezoid_area_l1861_186107

theorem trapezoid_area 
  (h : ℝ) (BM CM : ℝ) 
  (height_cond : h = 12) 
  (BM_cond : BM = 15) 
  (CM_cond : CM = 13) 
  (angle_bisectors_intersect : ∃ M : ℝ, (BM^2 - h^2) = 9^2 ∧ (CM^2 - h^2) = 5^2) : 
  ∃ (S : ℝ), S = 260.4 :=
by
  -- Skipping the proof part by using sorry
  sorry

end NUMINAMATH_GPT_trapezoid_area_l1861_186107


namespace NUMINAMATH_GPT_find_last_number_2_l1861_186129

theorem find_last_number_2 (A B C D : ℤ) 
  (h1 : A + B + C = 18)
  (h2 : B + C + D = 9)
  (h3 : A + D = 13) : 
  D = 2 := 
sorry

end NUMINAMATH_GPT_find_last_number_2_l1861_186129
