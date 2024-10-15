import Mathlib

namespace NUMINAMATH_GPT_slope_of_line_through_points_l239_23906

theorem slope_of_line_through_points :
  let x1 := 1
  let y1 := 3
  let x2 := 5
  let y2 := 7
  let m := (y2 - y1) / (x2 - x1)
  m = 1 := by
  sorry

end NUMINAMATH_GPT_slope_of_line_through_points_l239_23906


namespace NUMINAMATH_GPT_least_subtraction_for_divisibility_l239_23945

/-- 
  Theorem: The least number that must be subtracted from 9857621 so that 
  the result is divisible by 17 is 8.
-/
theorem least_subtraction_for_divisibility :
  ∃ k : ℕ, 9857621 % 17 = k ∧ k = 8 :=
by
  sorry

end NUMINAMATH_GPT_least_subtraction_for_divisibility_l239_23945


namespace NUMINAMATH_GPT_machines_complete_order_l239_23987

theorem machines_complete_order (h1 : ℝ) (h2 : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time : ℝ)
  (h1_def : h1 = 9)
  (h2_def : h2 = 8)
  (rate1_def : rate1 = 1 / h1)
  (rate2_def : rate2 = 1 / h2)
  (combined_rate : ℝ := rate1 + rate2) :
  time = 72 / 17 :=
by
  sorry

end NUMINAMATH_GPT_machines_complete_order_l239_23987


namespace NUMINAMATH_GPT_jenny_money_l239_23942

theorem jenny_money (x : ℝ) (h : (4 / 7) * x = 24) : (x / 2) = 21 := 
sorry

end NUMINAMATH_GPT_jenny_money_l239_23942


namespace NUMINAMATH_GPT_quadratic_factorization_sum_l239_23983

theorem quadratic_factorization_sum (d e f : ℤ) (h1 : ∀ x, x^2 + 18 * x + 80 = (x + d) * (x + e)) 
                                     (h2 : ∀ x, x^2 - 20 * x + 96 = (x - e) * (x - f)) : 
                                     d + e + f = 30 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_factorization_sum_l239_23983


namespace NUMINAMATH_GPT_lines_with_equal_intercepts_l239_23909

theorem lines_with_equal_intercepts (A : ℝ × ℝ) (hA : A = (1, 2)) :
  ∃ (n : ℕ), n = 3 ∧ (∀ l : ℝ → ℝ, (l 1 = 2) → ((l 0 = l (-0)) ∨ (l (-0) = l 0))) :=
by
  sorry

end NUMINAMATH_GPT_lines_with_equal_intercepts_l239_23909


namespace NUMINAMATH_GPT_sequence_may_or_may_not_be_arithmetic_l239_23976

theorem sequence_may_or_may_not_be_arithmetic (a : ℕ → ℕ) 
  (h1 : a 0 = 1) (h2 : a 1 = 2) (h3 : a 2 = 3) 
  (h4 : a 3 = 4) (h5 : a 4 = 5) : 
  ¬(∀ n, a (n + 1) - a n = 1) → 
  (∀ n, a (n + 1) - a n = 1) ∨ ¬(∀ n, a (n + 1) - a n = 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_may_or_may_not_be_arithmetic_l239_23976


namespace NUMINAMATH_GPT_minimum_value_l239_23988

theorem minimum_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  ∃ (y : ℝ), y = (c / (a + b)) + (b / c) ∧ y ≥ (Real.sqrt 2) - (1 / 2) :=
sorry

end NUMINAMATH_GPT_minimum_value_l239_23988


namespace NUMINAMATH_GPT_find_incorrect_value_l239_23907

theorem find_incorrect_value (n : ℕ) (mean_initial mean_correct : ℕ) (wrongly_copied correct_value incorrect_value : ℕ) 
  (h1 : n = 30) 
  (h2 : mean_initial = 150) 
  (h3 : mean_correct = 151) 
  (h4 : correct_value = 165) 
  (h5 : n * mean_initial = 4500) 
  (h6 : n * mean_correct = 4530) 
  (h7 : n * mean_correct - n * mean_initial = 30) 
  (h8 : correct_value - (n * mean_correct - n * mean_initial) = incorrect_value) : 
  incorrect_value = 135 :=
by
  sorry

end NUMINAMATH_GPT_find_incorrect_value_l239_23907


namespace NUMINAMATH_GPT_regression_decrease_by_5_l239_23931

theorem regression_decrease_by_5 (x y : ℝ) (h : y = 2 - 2.5 * x) :
  y = 2 - 2.5 * (x + 2) → y ≠ 2 - 2.5 * x - 5 :=
by sorry

end NUMINAMATH_GPT_regression_decrease_by_5_l239_23931


namespace NUMINAMATH_GPT_g_value_at_2_l239_23908

theorem g_value_at_2 (g : ℝ → ℝ) 
  (h : ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = x^2 - 2) : g 2 = 11 / 28 :=
sorry

end NUMINAMATH_GPT_g_value_at_2_l239_23908


namespace NUMINAMATH_GPT_bottle_capacity_l239_23978

theorem bottle_capacity
  (num_boxes : ℕ)
  (bottles_per_box : ℕ)
  (fill_fraction : ℚ)
  (total_volume : ℚ)
  (total_bottles : ℕ)
  (filled_volume : ℚ) :
  num_boxes = 10 →
  bottles_per_box = 50 →
  fill_fraction = 3 / 4 →
  total_volume = 4500 →
  total_bottles = num_boxes * bottles_per_box →
  filled_volume = (total_bottles : ℚ) * (fill_fraction * (12 : ℚ)) →
  12 = 4500 / (total_bottles * fill_fraction) := 
by 
  intros h1 h2 h3 h4 h5 h6
  simp [h1, h2, h3, h4, h5, h6]
  sorry

end NUMINAMATH_GPT_bottle_capacity_l239_23978


namespace NUMINAMATH_GPT_average_weight_of_all_girls_l239_23952

theorem average_weight_of_all_girls 
    (avg_weight_group1 : ℝ) (avg_weight_group2 : ℝ) 
    (num_girls_group1 : ℕ) (num_girls_group2 : ℕ) 
    (h1 : avg_weight_group1 = 50.25) 
    (h2 : avg_weight_group2 = 45.15) 
    (h3 : num_girls_group1 = 16) 
    (h4 : num_girls_group2 = 8) : 
    (avg_weight_group1 * num_girls_group1 + avg_weight_group2 * num_girls_group2) / (num_girls_group1 + num_girls_group2) = 48.55 := 
by 
    sorry

end NUMINAMATH_GPT_average_weight_of_all_girls_l239_23952


namespace NUMINAMATH_GPT_possible_values_y_l239_23961

theorem possible_values_y (x : ℝ) (h : x^2 + 4 * (x / (x - 2))^2 = 45) : 
  ∃ y : ℝ, y = 2 ∨ y = 16 :=
sorry

end NUMINAMATH_GPT_possible_values_y_l239_23961


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_l239_23946

theorem sum_arithmetic_sequence :
  let n := 21
  let a := 100
  let l := 120
  (n / 2) * (a + l) = 2310 :=
by
  -- define n, a, and l based on the conditions
  let n := 21
  let a := 100
  let l := 120
  -- state the goal
  have h : (n / 2) * (a + l) = 2310 := sorry
  exact h

end NUMINAMATH_GPT_sum_arithmetic_sequence_l239_23946


namespace NUMINAMATH_GPT_calc_root_diff_l239_23901

theorem calc_root_diff : 81^(1/4) - 16^(1/2) = -1 := by
  sorry

end NUMINAMATH_GPT_calc_root_diff_l239_23901


namespace NUMINAMATH_GPT_derivative_of_odd_function_is_even_l239_23910

-- Define an odd function f
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Define the main theorem
theorem derivative_of_odd_function_is_even (f g : ℝ → ℝ) 
  (h1 : is_odd_function f) 
  (h2 : ∀ x, g x = deriv f x) :
  ∀ x, g (-x) = g x :=
by
  sorry

end NUMINAMATH_GPT_derivative_of_odd_function_is_even_l239_23910


namespace NUMINAMATH_GPT_iron_ii_sulfate_moles_l239_23985

/-- Given the balanced chemical equation for the reaction between iron (Fe) and sulfuric acid (H2SO4)
    to form Iron (II) sulfate (FeSO4) and hydrogen gas (H2) and the 1:1 molar ratio between iron and
    sulfuric acid, determine the number of moles of Iron (II) sulfate formed when 3 moles of Iron and
    2 moles of Sulfuric acid are combined. This is a limiting reactant problem with the final 
    product being 2 moles of Iron (II) sulfate (FeSO4). -/
theorem iron_ii_sulfate_moles (Fe moles_H2SO4 : Nat) (reaction_ratio : Nat) (FeSO4 moles_formed : Nat) :
  Fe = 3 → moles_H2SO4 = 2 → reaction_ratio = 1 → moles_formed = 2 :=
by
  intros hFe hH2SO4 hRatio
  apply sorry

end NUMINAMATH_GPT_iron_ii_sulfate_moles_l239_23985


namespace NUMINAMATH_GPT_factor_theorem_l239_23995

theorem factor_theorem (m : ℝ) : (∀ x : ℝ, x + 5 = 0 → x ^ 2 - m * x - 40 = 0) → m = 3 :=
by
  sorry

end NUMINAMATH_GPT_factor_theorem_l239_23995


namespace NUMINAMATH_GPT_distinct_points_count_l239_23926

theorem distinct_points_count :
  ∃ (P : Finset (ℝ × ℝ)), 
    (∀ p ∈ P, p.1^2 + p.2^2 = 1 ∧ p.1^2 + 9 * p.2^2 = 9) ∧ P.card = 2 :=
by
  sorry

end NUMINAMATH_GPT_distinct_points_count_l239_23926


namespace NUMINAMATH_GPT_infinite_primes_l239_23933

theorem infinite_primes : ∀ (p : ℕ), Prime p → ¬ (∃ q : ℕ, Prime q ∧ q > p) := sorry

end NUMINAMATH_GPT_infinite_primes_l239_23933


namespace NUMINAMATH_GPT_monica_study_ratio_l239_23954

theorem monica_study_ratio :
  let wednesday := 2
  let thursday := 3 * wednesday
  let friday := thursday / 2
  let weekday_total := wednesday + thursday + friday
  let total := 22
  let weekend := total - weekday_total
  weekend = wednesday + thursday + friday :=
by
  let wednesday := 2
  let thursday := 3 * wednesday
  let friday := thursday / 2
  let weekday_total := wednesday + thursday + friday
  let total := 22
  let weekend := total - weekday_total
  sorry

end NUMINAMATH_GPT_monica_study_ratio_l239_23954


namespace NUMINAMATH_GPT_part_I_part_II_part_III_l239_23950

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 2) - (1 / (2^x + 1))

theorem part_I :
  ∃ a : ℝ, ∀ x : ℝ, f x = a - (1 / (2^x + 1)) → a = (1 / 2) :=
by sorry

theorem part_II :
  ∀ y : ℝ, y = f x → (-1 / 2) < y ∧ y < (1 / 2) :=
by sorry

theorem part_III :
  ∀ m n : ℝ, m + n ≠ 0 → (f m + f n) / (m^3 + n^3) > f 0 :=
by sorry

end NUMINAMATH_GPT_part_I_part_II_part_III_l239_23950


namespace NUMINAMATH_GPT_remainder_24_2377_mod_15_l239_23923

theorem remainder_24_2377_mod_15 :
  24^2377 % 15 = 9 :=
sorry

end NUMINAMATH_GPT_remainder_24_2377_mod_15_l239_23923


namespace NUMINAMATH_GPT_dave_apps_left_l239_23925

theorem dave_apps_left (A : ℕ) 
  (h1 : 24 = A + 22) : A = 2 :=
by
  sorry

end NUMINAMATH_GPT_dave_apps_left_l239_23925


namespace NUMINAMATH_GPT_triangle_inequality_equality_condition_l239_23913

variables {A B C a b c : ℝ}

theorem triangle_inequality (A a B b C c : ℝ) :
  A * a + B * b + C * c ≥ 1 / 2 * (A * b + B * a + A * c + C * a + B * c + C * b) :=
sorry

theorem equality_condition (A B C a b c : ℝ) :
  (A * a + B * b + C * c = 1 / 2 * (A * b + B * a + A * c + C * a + B * c + C * b)) ↔ (a = b ∧ b = c ∧ A = B ∧ B = C) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_equality_condition_l239_23913


namespace NUMINAMATH_GPT_mean_of_set_l239_23903

theorem mean_of_set {m : ℝ} 
  (median_condition : (m + 8 + m + 11) / 2 = 19) : 
  (m + (m + 6) + (m + 8) + (m + 11) + (m + 18) + (m + 20)) / 6 = 20 := 
by 
  sorry

end NUMINAMATH_GPT_mean_of_set_l239_23903


namespace NUMINAMATH_GPT_find_x_plus_one_over_x_l239_23960

open Real

theorem find_x_plus_one_over_x (x : ℝ) (h : x ^ 3 + 1 / x ^ 3 = 110) : x + 1 / x = 5 :=
sorry

end NUMINAMATH_GPT_find_x_plus_one_over_x_l239_23960


namespace NUMINAMATH_GPT_isosceles_obtuse_triangle_smallest_angle_l239_23927

theorem isosceles_obtuse_triangle_smallest_angle :
  ∀ (α β : ℝ), 0 < α ∧ α = 1.5 * 90 ∧ α + 2 * β = 180 ∧ β = 22.5 := by
  sorry

end NUMINAMATH_GPT_isosceles_obtuse_triangle_smallest_angle_l239_23927


namespace NUMINAMATH_GPT_quad_relation_l239_23967

theorem quad_relation
  (α AI BI CI DI : ℝ)
  (h1 : AB = α * (AI / CI + BI / DI))
  (h2 : BC = α * (BI / DI + CI / AI))
  (h3 : CD = α * (CI / AI + DI / BI))
  (h4 : DA = α * (DI / BI + AI / CI)) :
  AB + CD = AD + BC := by
  sorry

end NUMINAMATH_GPT_quad_relation_l239_23967


namespace NUMINAMATH_GPT_fibonacci_invariant_abs_difference_l239_23965

-- Given the sequence defined by the recurrence relation
def mArithmetical_fibonacci (u_n : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, u_n n = u_n (n - 2) + u_n (n - 1)

theorem fibonacci_invariant_abs_difference (u : ℕ → ℤ) 
  (h : mArithmetical_fibonacci u) :
  ∃ c : ℤ, ∀ n : ℕ, |u (n - 1) * u (n + 2) - u n * u (n + 1)| = c := 
sorry

end NUMINAMATH_GPT_fibonacci_invariant_abs_difference_l239_23965


namespace NUMINAMATH_GPT_expression_divisible_by_x_minus_1_squared_l239_23902

theorem expression_divisible_by_x_minus_1_squared :
  ∀ (n : ℕ) (x : ℝ), x ≠ 1 →
  (n * x^(n + 1) * (1 - 1 / x) - x^n * (1 - 1 / x^n)) / (x - 1)^2 = 
  (n * x^(n + 1) - n * x^n - x^n + 1) / (x - 1)^2 :=
by
  intro n x hx_ne_1
  sorry

end NUMINAMATH_GPT_expression_divisible_by_x_minus_1_squared_l239_23902


namespace NUMINAMATH_GPT_count_special_integers_l239_23935

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def base7 (n : ℕ) : ℕ := 
  let c := n / 343
  let rem1 := n % 343
  let d := rem1 / 49
  let rem2 := rem1 % 49
  let e := rem2 / 7
  let f := rem2 % 7
  343 * c + 49 * d + 7 * e + f

def base8 (n : ℕ) : ℕ := 
  let g := n / 512
  let rem1 := n % 512
  let h := rem1 / 64
  let rem2 := rem1 % 64
  let i := rem2 / 8
  let j := rem2 % 8
  512 * g + 64 * h + 8 * i + j

def matches_last_two_digits (n t : ℕ) : Prop := (t % 100) = (3 * (n % 100))

theorem count_special_integers : 
  ∃! (N : ℕ), is_three_digit N ∧ 
    matches_last_two_digits N (base7 N + base8 N) :=
sorry

end NUMINAMATH_GPT_count_special_integers_l239_23935


namespace NUMINAMATH_GPT_sum_sequence_a_b_eq_1033_l239_23937

def a (n : ℕ) : ℕ := n + 1
def b (n : ℕ) : ℕ := 2^(n-1)

theorem sum_sequence_a_b_eq_1033 : 
  (a (b 1)) + (a (b 2)) + (a (b 3)) + (a (b 4)) + (a (b 5)) + 
  (a (b 6)) + (a (b 7)) + (a (b 8)) + (a (b 9)) + (a (b 10)) = 1033 := by
  sorry

end NUMINAMATH_GPT_sum_sequence_a_b_eq_1033_l239_23937


namespace NUMINAMATH_GPT_net_profit_calculation_l239_23953

def original_purchase_price : ℝ := 80000
def annual_property_tax_rate : ℝ := 0.012
def annual_maintenance_cost : ℝ := 1500
def annual_mortgage_interest_rate : ℝ := 0.04
def selling_profit_rate : ℝ := 0.20
def broker_commission_rate : ℝ := 0.05
def years_of_ownership : ℕ := 5

noncomputable def net_profit : ℝ :=
  let selling_price := original_purchase_price * (1 + selling_profit_rate)
  let brokers_commission := original_purchase_price * broker_commission_rate
  let total_property_tax := original_purchase_price * annual_property_tax_rate * years_of_ownership
  let total_maintenance_cost := annual_maintenance_cost * years_of_ownership
  let total_mortgage_interest := original_purchase_price * annual_mortgage_interest_rate * years_of_ownership
  let total_costs := brokers_commission + total_property_tax + total_maintenance_cost + total_mortgage_interest
  (selling_price - original_purchase_price) - total_costs

theorem net_profit_calculation : net_profit = -16300 := by
  sorry

end NUMINAMATH_GPT_net_profit_calculation_l239_23953


namespace NUMINAMATH_GPT_coplanar_lines_k_values_l239_23999

theorem coplanar_lines_k_values (k : ℝ) :
  (∃ t u : ℝ, 
    (1 + t = 2 + u) ∧ 
    (2 + 2 * t = 5 + k * u) ∧ 
    (3 - k * t = 6 + u)) ↔ 
  (k = -2 + Real.sqrt 6 ∨ k = -2 - Real.sqrt 6) :=
sorry

end NUMINAMATH_GPT_coplanar_lines_k_values_l239_23999


namespace NUMINAMATH_GPT_number_of_n_l239_23957

theorem number_of_n (n : ℕ) (h1 : n ≤ 1000) (h2 : ∃ k : ℕ, 18 * n = k^2) : 
  ∃ K : ℕ, K = 7 :=
sorry

end NUMINAMATH_GPT_number_of_n_l239_23957


namespace NUMINAMATH_GPT_phosphorus_atoms_l239_23948

theorem phosphorus_atoms (x : ℝ) : 122 = 26.98 + 30.97 * x + 64 → x = 1 := by
sorry

end NUMINAMATH_GPT_phosphorus_atoms_l239_23948


namespace NUMINAMATH_GPT_find_OH_squared_l239_23924

variables {O H : Type} {a b c R : ℝ}

-- Given conditions
def is_circumcenter (O : Type) (ABC : Type) := true -- Placeholder definition
def is_orthocenter (H : Type) (ABC : Type) := true -- Placeholder definition
def circumradius (O : Type) (R : ℝ) := true -- Placeholder definition
def sides_squared_sum (a b c : ℝ) := a^2 + b^2 + c^2

-- The theorem to be proven
theorem find_OH_squared (O H : Type) (a b c : ℝ) (R : ℝ) 
  (circ : is_circumcenter O ABC) 
  (orth: is_orthocenter H ABC) 
  (radius : circumradius O R) 
  (terms_sum : sides_squared_sum a b c = 50)
  (R_val : R = 10) 
  : OH^2 = 850 := 
sorry

end NUMINAMATH_GPT_find_OH_squared_l239_23924


namespace NUMINAMATH_GPT_complete_square_solution_l239_23900

theorem complete_square_solution (a b c : ℤ) (h1 : a^2 = 25) (h2 : 10 * b = 30) (h3 : (a * x + b)^2 = 25 * x^2 + 30 * x + c) :
  a + b + c = -58 :=
by
  sorry

end NUMINAMATH_GPT_complete_square_solution_l239_23900


namespace NUMINAMATH_GPT_colin_speed_l239_23928

variable (B T Br C : ℝ)

def Bruce := B = 1
def Tony := T = 2 * B
def Brandon := Br = T / 3
def Colin := C = 6 * Br

theorem colin_speed : Bruce B → Tony B T → Brandon T Br → Colin Br C → C = 4 := by
  sorry

end NUMINAMATH_GPT_colin_speed_l239_23928


namespace NUMINAMATH_GPT_solve_problem_l239_23990

def spadesuit (a b : ℤ) : ℤ := abs (a - b)

theorem solve_problem : spadesuit 3 (spadesuit 5 (spadesuit 8 11)) = 1 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_solve_problem_l239_23990


namespace NUMINAMATH_GPT_point_on_inverse_proportion_l239_23998

theorem point_on_inverse_proportion (k : ℝ) (hk : k ≠ 0) :
  (2 * 3 = k) → (1 * 6 = k) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_point_on_inverse_proportion_l239_23998


namespace NUMINAMATH_GPT_right_triangle_set_C_l239_23929

theorem right_triangle_set_C :
  ∃ (a b c : ℕ), a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_set_C_l239_23929


namespace NUMINAMATH_GPT_three_digit_solutions_exist_l239_23982

theorem three_digit_solutions_exist :
  ∃ (x y z : ℤ), 100 ≤ x ∧ x ≤ 999 ∧ 
                 100 ≤ y ∧ y ≤ 999 ∧
                 100 ≤ z ∧ z ≤ 999 ∧
                 17 * x + 15 * y - 28 * z = 61 ∧
                 19 * x - 25 * y + 12 * z = 31 :=
by
    sorry

end NUMINAMATH_GPT_three_digit_solutions_exist_l239_23982


namespace NUMINAMATH_GPT_weight_of_second_piece_l239_23969

-- Define the uniform density of the metal.
def density : ℝ := 0.5  -- ounces per square inch

-- Define the side lengths of the two pieces of metal.
def side_length1 : ℝ := 4  -- inches
def side_length2 : ℝ := 7  -- inches

-- Define the weights of the first piece of metal.
def weight1 : ℝ := 8  -- ounces

-- Define the areas of the pieces of metal.
def area1 : ℝ := side_length1^2  -- square inches
def area2 : ℝ := side_length2^2  -- square inches

-- The theorem to prove: the weight of the second piece of metal.
theorem weight_of_second_piece : (area2 * density) = 24.5 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_second_piece_l239_23969


namespace NUMINAMATH_GPT_A_serves_on_50th_week_is_Friday_l239_23963

-- Define the people involved in the rotation
inductive Person
| A | B | C | D | E | F

open Person

-- Define the function that computes the day A serves on given the number of weeks
def day_A_serves (weeks : ℕ) : ℕ :=
  let days := weeks * 7
  (days % 6 + 0) % 7 -- 0 is the offset for the initial day when A serves (Sunday)

theorem A_serves_on_50th_week_is_Friday :
  day_A_serves 50 = 5 :=
by
  -- We provide the proof here
  sorry

end NUMINAMATH_GPT_A_serves_on_50th_week_is_Friday_l239_23963


namespace NUMINAMATH_GPT_boxes_left_to_sell_l239_23941

def sales_goal : ℕ := 150
def first_customer : ℕ := 5
def second_customer : ℕ := 4 * first_customer
def third_customer : ℕ := second_customer / 2
def fourth_customer : ℕ := 3 * third_customer
def fifth_customer : ℕ := 10
def total_sold : ℕ := first_customer + second_customer + third_customer + fourth_customer + fifth_customer

theorem boxes_left_to_sell : sales_goal - total_sold = 75 := by
  sorry

end NUMINAMATH_GPT_boxes_left_to_sell_l239_23941


namespace NUMINAMATH_GPT_baron_munchausen_claim_l239_23971

-- Given conditions and question:
def weight_partition_problem (weights : Finset ℕ) (h_card : weights.card = 50) (h_distinct : ∀ w ∈ weights,  1 ≤ w ∧ w ≤ 100) (h_sum_even : weights.sum id % 2 = 0) : Prop :=
  ¬(∃ (s1 s2 : Finset ℕ), s1 ∪ s2 = weights ∧ s1 ∩ s2 = ∅ ∧ s1.sum id = s2.sum id)

-- We need to prove that the above statement is true.
theorem baron_munchausen_claim :
  ∀ (weights : Finset ℕ), weights.card = 50 ∧ (∀ w ∈ weights, 1 ≤ w ∧ w ≤ 100) ∧ weights.sum id % 2 = 0 → weight_partition_problem weights (by sorry) (by sorry) (by sorry) :=
sorry

end NUMINAMATH_GPT_baron_munchausen_claim_l239_23971


namespace NUMINAMATH_GPT_min_nS_n_eq_neg32_l239_23989

variable (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (d : ℤ) (a_1 : ℤ)

-- Conditions
axiom arithmetic_sequence_def : ∀ n : ℕ, a n = a_1 + (n - 1) * d
axiom sum_first_n_def : ∀ n : ℕ, S n = n * a_1 + (n * (n - 1) / 2) * d

axiom a5_eq_3 : a 5 = 3
axiom S10_eq_40 : S 10 = 40

theorem min_nS_n_eq_neg32 : ∃ n : ℕ, n * S n = -32 :=
sorry

end NUMINAMATH_GPT_min_nS_n_eq_neg32_l239_23989


namespace NUMINAMATH_GPT_evaluate_expression_at_3_l239_23994

theorem evaluate_expression_at_3 :
  (1 / (3 + 1 / (3 + 1 / (3 - 1 / 3)))) = 0.30337078651685395 :=
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_3_l239_23994


namespace NUMINAMATH_GPT_sasha_took_right_triangle_l239_23930

-- Define types of triangles
inductive Triangle
| acute
| right
| obtuse

open Triangle

-- Define the function that determines if Borya can form a triangle identical to Sasha's
def can_form_identical_triangle (t1 t2 t3: Triangle) : Bool :=
match t1, t2, t3 with
| right, acute, obtuse => true
| _ , _ , _ => false

-- Define the main theorem
theorem sasha_took_right_triangle : 
  ∀ (sasha_takes borya_takes1 borya_takes2 : Triangle),
  (sasha_takes ≠ borya_takes1 ∧ sasha_takes ≠ borya_takes2 ∧ borya_takes1 ≠ borya_takes2) →
  can_form_identical_triangle sasha_takes borya_takes1 borya_takes2 →
  sasha_takes = right :=
by sorry

end NUMINAMATH_GPT_sasha_took_right_triangle_l239_23930


namespace NUMINAMATH_GPT_largest_n_divides_1005_fact_l239_23938

theorem largest_n_divides_1005_fact (n : ℕ) : (∃ n, 10^n ∣ (Nat.factorial 1005)) ↔ n = 250 :=
by
  sorry

end NUMINAMATH_GPT_largest_n_divides_1005_fact_l239_23938


namespace NUMINAMATH_GPT_more_movies_than_books_l239_23956

-- Conditions
def books_read := 15
def movies_watched := 29

-- Question: How many more movies than books have you watched?
theorem more_movies_than_books : (movies_watched - books_read) = 14 := sorry

end NUMINAMATH_GPT_more_movies_than_books_l239_23956


namespace NUMINAMATH_GPT_maximum_value_l239_23912

noncomputable def maxValue (x y : ℝ) (h : x + y = 5) : ℝ :=
  x^5 * y + x^4 * y^2 + x^3 * y^3 + x^2 * y^4 + x * y^5

theorem maximum_value (x y : ℝ) (h : x + y = 5) : maxValue x y h ≤ 625 / 4 :=
sorry

end NUMINAMATH_GPT_maximum_value_l239_23912


namespace NUMINAMATH_GPT_unique_solution_of_inequality_l239_23920

open Real

theorem unique_solution_of_inequality (b : ℝ) : 
  (∃! x : ℝ, |x^2 + 2 * b * x + 2 * b| ≤ 1) ↔ b = 1 := 
by exact sorry

end NUMINAMATH_GPT_unique_solution_of_inequality_l239_23920


namespace NUMINAMATH_GPT_math_problem_l239_23916

theorem math_problem :
  18 * 35 + 45 * 18 - 18 * 10 = 1260 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l239_23916


namespace NUMINAMATH_GPT_smallest_z_value_l239_23955

theorem smallest_z_value :
  ∀ w x y z : ℤ, (∃ k : ℤ, w = 2 * k - 1 ∧ x = 2 * k + 1 ∧ y = 2 * k + 3 ∧ z = 2 * k + 5) ∧
    w^3 + x^3 + y^3 = z^3 →
    z = 9 :=
sorry

end NUMINAMATH_GPT_smallest_z_value_l239_23955


namespace NUMINAMATH_GPT_macy_hit_ball_50_times_l239_23979

-- Definitions and conditions
def token_pitches : ℕ := 15
def macy_tokens : ℕ := 11
def piper_tokens : ℕ := 17
def piper_hits : ℕ := 55
def missed_pitches : ℕ := 315

-- Calculation based on conditions
def total_pitches : ℕ := (macy_tokens + piper_tokens) * token_pitches
def total_hits : ℕ := total_pitches - missed_pitches
def macy_hits : ℕ := total_hits - piper_hits

-- Prove that Macy hit 50 times
theorem macy_hit_ball_50_times : macy_hits = 50 := 
by
  sorry

end NUMINAMATH_GPT_macy_hit_ball_50_times_l239_23979


namespace NUMINAMATH_GPT_relationship_between_a_b_c_l239_23932

noncomputable def a := 33
noncomputable def b := 5 * 6^1 + 2 * 6^0
noncomputable def c := 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem relationship_between_a_b_c : a > b ∧ b > c := by
  sorry

end NUMINAMATH_GPT_relationship_between_a_b_c_l239_23932


namespace NUMINAMATH_GPT_angle_between_hands_at_3_40_l239_23936

def degrees_per_minute_minute_hand := 360 / 60
def minutes_passed := 40
def degrees_minute_hand := degrees_per_minute_minute_hand * minutes_passed -- 240 degrees

def degrees_per_hour_hour_hand := 360 / 12
def hours_passed := 3
def degrees_hour_hand_at_hour := degrees_per_hour_hour_hand * hours_passed -- 90 degrees

def degrees_per_minute_hour_hand := degrees_per_hour_hour_hand / 60
def degrees_hour_hand_additional := degrees_per_minute_hour_hand * minutes_passed -- 20 degrees

def total_degrees_hour_hand := degrees_hour_hand_at_hour + degrees_hour_hand_additional -- 110 degrees

def expected_angle_between_hands := 130

theorem angle_between_hands_at_3_40
  (h1: degrees_minute_hand = 240)
  (h2: total_degrees_hour_hand = 110):
  (degrees_minute_hand - total_degrees_hour_hand = expected_angle_between_hands) :=
by
  sorry

end NUMINAMATH_GPT_angle_between_hands_at_3_40_l239_23936


namespace NUMINAMATH_GPT_quadrilateral_angle_B_l239_23904

/-- In quadrilateral ABCD,
given that angle A + angle C = 150 degrees,
prove that angle B = 105 degrees. -/
theorem quadrilateral_angle_B (A C : ℝ) (B : ℝ) (h1 : A + C = 150) (h2 : A + B = 180) : B = 105 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_angle_B_l239_23904


namespace NUMINAMATH_GPT_hannah_strawberries_l239_23996

theorem hannah_strawberries (days give_away stolen remaining_strawberries x : ℕ) 
  (h1 : days = 30) 
  (h2 : give_away = 20) 
  (h3 : stolen = 30) 
  (h4 : remaining_strawberries = 100) 
  (hx : x = (remaining_strawberries + give_away + stolen) / days) : 
  x = 5 := 
by 
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_hannah_strawberries_l239_23996


namespace NUMINAMATH_GPT_workers_not_worked_days_l239_23974

theorem workers_not_worked_days (W N : ℤ) (h1 : W + N = 30) (h2 : 100 * W - 25 * N = 0) : N = 24 := 
by
  sorry

end NUMINAMATH_GPT_workers_not_worked_days_l239_23974


namespace NUMINAMATH_GPT_g_of_f_of_3_eq_1902_l239_23918

def f (x : ℕ) := x^3 - 2
def g (x : ℕ) := 3 * x^2 + x + 2

theorem g_of_f_of_3_eq_1902 : g (f 3) = 1902 := by
  sorry

end NUMINAMATH_GPT_g_of_f_of_3_eq_1902_l239_23918


namespace NUMINAMATH_GPT_vertices_form_vertical_line_l239_23986

theorem vertices_form_vertical_line (a b k d : ℝ) (ha : 0 < a) (hk : 0 < k) :
  ∃ x, ∀ t : ℝ, ∃ y, (x = -b / (2 * a) ∧ y = - (b^2) / (4 * a) + k * t + d) :=
sorry

end NUMINAMATH_GPT_vertices_form_vertical_line_l239_23986


namespace NUMINAMATH_GPT_solveNumberOfWaysToChooseSeats_l239_23977

/--
Define the problem of professors choosing their seats among 9 chairs with specific constraints.
-/
noncomputable def numberOfWaysToChooseSeats : ℕ :=
  let totalChairs := 9
  let endChairChoices := 2 * (7 * (7 - 2))  -- (2 end chairs, 7 for 2nd prof, 5 for 3rd prof)
  let middleChairChoices := 7 * (6 * (6 - 2))  -- (7 non-end chairs, 6 for 2nd prof, 4 for 3rd prof)
  endChairChoices + middleChairChoices

/--
The final result should be 238
-/
theorem solveNumberOfWaysToChooseSeats : numberOfWaysToChooseSeats = 238 := by
  sorry

end NUMINAMATH_GPT_solveNumberOfWaysToChooseSeats_l239_23977


namespace NUMINAMATH_GPT_ratio_equation_solution_l239_23934

theorem ratio_equation_solution (x : ℝ) :
  (4 + 2 * x) / (6 + 3 * x) = (2 + x) / (3 + 2 * x) → (x = 0 ∨ x = 4) :=
by
  -- the proof steps would go here
  sorry

end NUMINAMATH_GPT_ratio_equation_solution_l239_23934


namespace NUMINAMATH_GPT_fourth_vertex_of_regular_tetrahedron_exists_and_is_unique_l239_23919

theorem fourth_vertex_of_regular_tetrahedron_exists_and_is_unique :
  ∃ (x y z : ℤ),
    (x, y, z) ≠ (1, 2, 3) ∧ (x, y, z) ≠ (5, 3, 2) ∧ (x, y, z) ≠ (4, 2, 6) ∧
    (x - 1)^2 + (y - 2)^2 + (z - 3)^2 = 18 ∧
    (x - 5)^2 + (y - 3)^2 + (z - 2)^2 = 18 ∧
    (x - 4)^2 + (y - 2)^2 + (z - 6)^2 = 18 ∧
    (x, y, z) = (2, 3, 5) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_fourth_vertex_of_regular_tetrahedron_exists_and_is_unique_l239_23919


namespace NUMINAMATH_GPT_total_kids_attended_camp_l239_23921

theorem total_kids_attended_camp :
  let n1 := 34044
  let n2 := 424944
  n1 + n2 = 458988 := 
by {
  sorry
}

end NUMINAMATH_GPT_total_kids_attended_camp_l239_23921


namespace NUMINAMATH_GPT_difference_of_squares_l239_23981

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 8) : x^2 - y^2 = 80 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l239_23981


namespace NUMINAMATH_GPT_circle_radius_l239_23964

theorem circle_radius (x y d : ℝ) (h₁ : x = π * r^2) (h₂ : y = 2 * π * r) (h₃ : d = 2 * r) (h₄ : x + y + d = 164 * π) : r = 10 :=
by sorry

end NUMINAMATH_GPT_circle_radius_l239_23964


namespace NUMINAMATH_GPT_product_of_solutions_l239_23975

theorem product_of_solutions :
  let a := 2
  let b := 4
  let c := -6
  let discriminant := b^2 - 4*a*c
  ∃ (x₁ x₂ : ℝ), 2*x₁^2 + 4*x₁ - 6 = 0 ∧ 2*x₂^2 + 4*x₂ - 6 = 0 ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -3 :=
sorry

end NUMINAMATH_GPT_product_of_solutions_l239_23975


namespace NUMINAMATH_GPT_cost_price_of_book_l239_23958

theorem cost_price_of_book 
  (C : ℝ) 
  (h1 : 1.10 * C = sp10) 
  (h2 : 1.15 * C = sp15)
  (h3 : sp15 - sp10 = 90) : 
  C = 1800 := 
sorry

end NUMINAMATH_GPT_cost_price_of_book_l239_23958


namespace NUMINAMATH_GPT_ladder_slip_l239_23966

theorem ladder_slip 
  (ladder_length : ℝ) 
  (initial_base : ℝ) 
  (slip_height : ℝ) 
  (h_length : ladder_length = 30) 
  (h_base : initial_base = 11) 
  (h_slip : slip_height = 6) 
  : ∃ (slide_distance : ℝ), abs (slide_distance - 9.49) < 0.01 :=
by
  let initial_height := Real.sqrt (ladder_length^2 - initial_base^2)
  let new_height := initial_height - slip_height
  let new_base := Real.sqrt (ladder_length^2 - new_height^2)
  let slide_distance := new_base - initial_base
  use slide_distance
  have h_approx : abs (slide_distance - 9.49) < 0.01 := sorry
  exact h_approx

end NUMINAMATH_GPT_ladder_slip_l239_23966


namespace NUMINAMATH_GPT_range_of_m_l239_23915

noncomputable def set_M (m : ℝ) : Set ℝ := {x | x < m}
noncomputable def set_N : Set ℝ := {y | ∃ (x : ℝ), y = Real.log x / Real.log 2 - 1 ∧ 4 ≤ x}

theorem range_of_m (m : ℝ) : set_M m ∩ set_N = ∅ → m < 1 
:= by
  sorry

end NUMINAMATH_GPT_range_of_m_l239_23915


namespace NUMINAMATH_GPT_num_partition_sets_correct_l239_23991

noncomputable def num_partition_sets (n : ℕ) : ℕ :=
  2^(n-1) - 1

theorem num_partition_sets_correct (n : ℕ) (hn : n ≥ 2) : 
  num_partition_sets n = 2^(n-1) - 1 := 
by sorry

end NUMINAMATH_GPT_num_partition_sets_correct_l239_23991


namespace NUMINAMATH_GPT_avg_weight_b_c_l239_23973

variables (A B C : ℝ)

-- Given Conditions
def condition1 := (A + B + C) / 3 = 45
def condition2 := (A + B) / 2 = 40
def condition3 := B = 37

-- Statement to prove
theorem avg_weight_b_c 
  (h1 : condition1 A B C)
  (h2 : condition2 A B)
  (h3 : condition3 B) : 
  (B + C) / 2 = 46 :=
sorry

end NUMINAMATH_GPT_avg_weight_b_c_l239_23973


namespace NUMINAMATH_GPT_num_tickets_bought_l239_23968

-- Defining the cost and discount conditions
def ticket_cost : ℝ := 40
def discount_rate : ℝ := 0.05
def total_paid : ℝ := 476
def base_tickets : ℕ := 10

-- Definition to calculate the cost of the first 10 tickets
def cost_first_10_tickets : ℝ := base_tickets * ticket_cost
-- Definition of the discounted price for tickets exceeding 10
def discounted_ticket_cost : ℝ := ticket_cost * (1 - discount_rate)
-- Definition of the total cost for the tickets exceeding 10
def cost_discounted_tickets (num_tickets_exceeding_10 : ℕ) : ℝ := num_tickets_exceeding_10 * discounted_ticket_cost
-- Total amount spent on the tickets exceeding 10
def amount_spent_on_discounted_tickets : ℝ := total_paid - cost_first_10_tickets

-- Main theorem statement proving the total number of tickets Mr. Benson bought
theorem num_tickets_bought : ∃ x : ℕ, x = base_tickets + (amount_spent_on_discounted_tickets / discounted_ticket_cost) ∧ x = 12 := 
by
  sorry

end NUMINAMATH_GPT_num_tickets_bought_l239_23968


namespace NUMINAMATH_GPT_a_minus_3d_eq_zero_l239_23992

noncomputable def f (a b c d x : ℝ) : ℝ := (2 * a * x + b) / (c * x - 3 * d)

theorem a_minus_3d_eq_zero (a b c d : ℝ) (h : f a b c d ≠ 0)
  (h1 : ∀ x, f a b c d x = x) :
  a - 3 * d = 0 :=
sorry

end NUMINAMATH_GPT_a_minus_3d_eq_zero_l239_23992


namespace NUMINAMATH_GPT_quadratic_has_two_real_roots_find_m_for_roots_difference_4_l239_23972

-- Define the function representing the quadratic equation
def quadratic_eq (m x : ℝ) := x^2 + (2 - m) * x + 1 - m

-- Part 1
theorem quadratic_has_two_real_roots (m : ℝ) : 
  ∃ (x1 x2 : ℝ), quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0 :=
sorry

-- Part 2
theorem find_m_for_roots_difference_4 (m : ℝ) (H : m < 0) :
  (∃ (x1 x2 : ℝ), quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0 ∧ x1 - x2 = 4) → m = -4 :=
sorry

end NUMINAMATH_GPT_quadratic_has_two_real_roots_find_m_for_roots_difference_4_l239_23972


namespace NUMINAMATH_GPT_brand_tangyuan_purchase_l239_23905

theorem brand_tangyuan_purchase (x y : ℕ) 
  (h1 : x + y = 1000) 
  (h2 : x = 2 * y + 20) : 
  x = 670 ∧ y = 330 := 
sorry

end NUMINAMATH_GPT_brand_tangyuan_purchase_l239_23905


namespace NUMINAMATH_GPT_calculate_value_l239_23944

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

variable (f : ℝ → ℝ)

axiom h : odd_function f
axiom h1 : increasing_on_interval f 3 7
axiom h2 : f 3 = -1
axiom h3 : f 6 = 8

theorem calculate_value : 2 * f (-6) + f (-3) = -15 := by
  sorry

end NUMINAMATH_GPT_calculate_value_l239_23944


namespace NUMINAMATH_GPT_work_done_by_force_l239_23940

noncomputable def displacement (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem work_done_by_force :
  let F := (5, 2)
  let A := (-1, 3)
  let B := (2, 6)
  let AB := displacement A B
  dot_product F AB = 21 := by
  sorry

end NUMINAMATH_GPT_work_done_by_force_l239_23940


namespace NUMINAMATH_GPT_cylinder_sphere_ratio_l239_23939

theorem cylinder_sphere_ratio (r R : ℝ) (h : 8 * r^2 = 4 * R^2) : R / r = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_sphere_ratio_l239_23939


namespace NUMINAMATH_GPT_derivative_at_neg_one_l239_23914

def f (x : ℝ) : ℝ := List.prod (List.map (λ k => (x^3 + k)) (List.range' 1 100))

theorem derivative_at_neg_one : deriv f (-1) = 3 * Nat.factorial 99 := by
  sorry

end NUMINAMATH_GPT_derivative_at_neg_one_l239_23914


namespace NUMINAMATH_GPT_hot_dog_cost_l239_23949

theorem hot_dog_cost : 
  ∃ h d : ℝ, (3 * h + 4 * d = 10) ∧ (2 * h + 3 * d = 7) ∧ (d = 1) := 
by 
  sorry

end NUMINAMATH_GPT_hot_dog_cost_l239_23949


namespace NUMINAMATH_GPT_number_of_possible_m_values_l239_23947

theorem number_of_possible_m_values :
  ∃ m_set : Finset ℤ, (∀ x1 x2 : ℤ, x1 * x2 = 40 → (x1 + x2) ∈ m_set) ∧ m_set.card = 8 :=
sorry

end NUMINAMATH_GPT_number_of_possible_m_values_l239_23947


namespace NUMINAMATH_GPT_minimum_value_f_l239_23943

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * Real.log x

theorem minimum_value_f :
  ∃ x > 0, (∀ y > 0, f x ≤ f y) ∧ f x = 1 :=
sorry

end NUMINAMATH_GPT_minimum_value_f_l239_23943


namespace NUMINAMATH_GPT_percent_of_whole_is_fifty_l239_23970

theorem percent_of_whole_is_fifty (part whole : ℝ) (h1 : part = 180) (h2 : whole = 360) : 
  ((part / whole) * 100) = 50 := 
by 
  rw [h1, h2] 
  sorry

end NUMINAMATH_GPT_percent_of_whole_is_fifty_l239_23970


namespace NUMINAMATH_GPT_average_age_of_second_group_is_16_l239_23911

theorem average_age_of_second_group_is_16
  (total_age_15_students : ℕ := 225)
  (total_age_first_group_7_students : ℕ := 98)
  (age_15th_student : ℕ := 15) :
  (total_age_15_students - total_age_first_group_7_students - age_15th_student) / 7 = 16 := 
by
  sorry

end NUMINAMATH_GPT_average_age_of_second_group_is_16_l239_23911


namespace NUMINAMATH_GPT_johnny_marbles_l239_23962

noncomputable def choose_at_least_one_red : ℕ :=
  let total_marbles := 8
  let red_marbles := 1
  let other_marbles := 7
  let choose_4_out_of_8 := Nat.choose total_marbles 4
  let choose_3_out_of_7 := Nat.choose other_marbles 3
  let choose_4_with_at_least_1_red := choose_3_out_of_7
  choose_4_with_at_least_1_red

theorem johnny_marbles : choose_at_least_one_red = 35 :=
by
  -- Sorry, proof is omitted
  sorry

end NUMINAMATH_GPT_johnny_marbles_l239_23962


namespace NUMINAMATH_GPT_yola_past_weight_l239_23993

-- Definitions based on the conditions
def current_weight_yola : ℕ := 220
def weight_difference_current (D : ℕ) : ℕ := 30
def weight_difference_past (D : ℕ) : ℕ := D

-- Main statement
theorem yola_past_weight (D : ℕ) :
  (250 - D) = (current_weight_yola + weight_difference_current D - weight_difference_past D) :=
by
  sorry

end NUMINAMATH_GPT_yola_past_weight_l239_23993


namespace NUMINAMATH_GPT_dabbies_turkey_cost_l239_23984

noncomputable def first_turkey_weight : ℕ := 6
noncomputable def second_turkey_weight : ℕ := 9
noncomputable def third_turkey_weight : ℕ := 2 * second_turkey_weight
noncomputable def cost_per_kg : ℕ := 2

noncomputable def total_cost : ℕ :=
  first_turkey_weight * cost_per_kg +
  second_turkey_weight * cost_per_kg +
  third_turkey_weight * cost_per_kg

theorem dabbies_turkey_cost : total_cost = 66 :=
by
  sorry

end NUMINAMATH_GPT_dabbies_turkey_cost_l239_23984


namespace NUMINAMATH_GPT_quadratic_inequality_condition_l239_23922

theorem quadratic_inequality_condition (x : ℝ) : x^2 - 2*x - 3 < 0 ↔ x ∈ Set.Ioo (-1) 3 := 
sorry

end NUMINAMATH_GPT_quadratic_inequality_condition_l239_23922


namespace NUMINAMATH_GPT_find_y_l239_23980

theorem find_y (x y : ℤ) (h1 : x^2 = y - 3) (h2 : x = -5) : y = 28 := by
  sorry

end NUMINAMATH_GPT_find_y_l239_23980


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l239_23997

-- Proof problem for the first expression
theorem simplify_expr1 (x y : ℤ) : (2 - x + 3 * y + 8 * x - 5 * y - 6) = (7 * x - 2 * y -4) := 
by 
   -- Proving steps would go here
   sorry

-- Proof problem for the second expression
theorem simplify_expr2 (a b : ℤ) : (15 * a^2 * b - 12 * a * b^2 + 12 - 4 * a^2 * b - 18 + 8 * a * b^2) = (11 * a^2 * b - 4 * a * b^2 - 6) := 
by 
   -- Proving steps would go here
   sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l239_23997


namespace NUMINAMATH_GPT_sum_circumferences_of_small_circles_l239_23959

theorem sum_circumferences_of_small_circles (R : ℝ) (n : ℕ) (hR : R > 0) (hn : n > 0) :
  let original_circumference := 2 * Real.pi * R
  let part_length := original_circumference / n
  let small_circle_radius := part_length / Real.pi
  let small_circle_circumference := 2 * Real.pi * small_circle_radius
  let total_circumference := n * small_circle_circumference
  total_circumference = 2 * Real.pi ^ 2 * R :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_circumferences_of_small_circles_l239_23959


namespace NUMINAMATH_GPT_a_lt_1_sufficient_but_not_necessary_l239_23917

noncomputable def represents_circle (a : ℝ) : Prop :=
  a^2 - 10 * a + 9 > 0

theorem a_lt_1_sufficient_but_not_necessary (a : ℝ) :
  represents_circle a → ((a < 1) ∨ (a > 9)) :=
sorry

end NUMINAMATH_GPT_a_lt_1_sufficient_but_not_necessary_l239_23917


namespace NUMINAMATH_GPT_greatest_value_q_minus_r_l239_23951

theorem greatest_value_q_minus_r : ∃ q r : ℕ, 1043 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ (q - r = 37) :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_value_q_minus_r_l239_23951
