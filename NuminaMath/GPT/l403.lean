import Mathlib

namespace NUMINAMATH_GPT_coordinates_with_respect_to_origin_l403_40373

theorem coordinates_with_respect_to_origin :
  ∀ (point : ℝ × ℝ), point = (3, -2) → point = (3, -2) := by
  intro point h
  exact h

end NUMINAMATH_GPT_coordinates_with_respect_to_origin_l403_40373


namespace NUMINAMATH_GPT_problem_statement_l403_40321

theorem problem_statement (x y : ℝ) (h1 : |x| + x - y = 16) (h2 : x - |y| + y = -8) : x + y = -8 := sorry

end NUMINAMATH_GPT_problem_statement_l403_40321


namespace NUMINAMATH_GPT_line_through_ellipse_and_midpoint_l403_40341

theorem line_through_ellipse_and_midpoint :
  ∃ l : ℝ → ℝ → Prop,
    (∀ (x y : ℝ), l x y ↔ (x + y) = 0) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      (x₁ + x₂ = 2 ∧ y₁ + y₂ = 1) ∧
      (x₁^2 / 2 + y₁^2 = 1 ∧ x₂^2 / 2 + y₂^2 = 1) ∧
      l x₁ y₁ ∧ l x₂ y₂ ∧
      ∀ (mx my : ℝ), (mx, my) = (1, 0.5) → (mx = (x₁ + x₂) / 2 ∧ my = (y₁ + y₂) / 2))
  := sorry

end NUMINAMATH_GPT_line_through_ellipse_and_midpoint_l403_40341


namespace NUMINAMATH_GPT_length_of_each_brick_l403_40363

theorem length_of_each_brick (wall_length wall_height wall_thickness : ℝ) (brick_length brick_width brick_height : ℝ) (num_bricks_used : ℝ) 
  (h1 : wall_length = 8) 
  (h2 : wall_height = 6) 
  (h3 : wall_thickness = 0.02) 
  (h4 : brick_length = 0.11) 
  (h5 : brick_width = 0.05) 
  (h6 : brick_height = 0.06) 
  (h7 : num_bricks_used = 2909.090909090909) : 
  brick_length = 0.11 :=
by
  -- variables and assumptions
  have vol_wall : ℝ := wall_length * wall_height * wall_thickness
  have vol_brick : ℝ := brick_length * brick_width * brick_height
  have calc_bricks : ℝ := vol_wall / vol_brick
  -- skipping proof
  sorry

end NUMINAMATH_GPT_length_of_each_brick_l403_40363


namespace NUMINAMATH_GPT_solve_system_of_equations_l403_40391

theorem solve_system_of_equations:
  ∃ (x y : ℚ), 3 * x + 4 * y = 16 ∧ 5 * x - 6 * y = 33 ∧ x = 6 ∧ y = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l403_40391


namespace NUMINAMATH_GPT_sqrt_450_eq_15_sqrt_2_l403_40334

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_450_eq_15_sqrt_2_l403_40334


namespace NUMINAMATH_GPT_david_marks_in_physics_l403_40398

theorem david_marks_in_physics 
  (english_marks mathematics_marks chemistry_marks biology_marks : ℕ)
  (num_subjects : ℕ)
  (average_marks : ℕ)
  (h1 : english_marks = 81)
  (h2 : mathematics_marks = 65)
  (h3 : chemistry_marks = 67)
  (h4 : biology_marks = 85)
  (h5 : num_subjects = 5)
  (h6 : average_marks = 76) :
  ∃ physics_marks : ℕ, physics_marks = 82 :=
by
  sorry

end NUMINAMATH_GPT_david_marks_in_physics_l403_40398


namespace NUMINAMATH_GPT_midpoint_product_zero_l403_40390

theorem midpoint_product_zero (x y : ℝ) :
  let A := (2, 6)
  let B := (x, y)
  let C := (4, 3)
  (C = ((2 + x) / 2, (6 + y) / 2)) → (x * y = 0) := by
  intros
  sorry

end NUMINAMATH_GPT_midpoint_product_zero_l403_40390


namespace NUMINAMATH_GPT_nth_monomial_is_correct_l403_40307

-- conditions
def coefficient (n : ℕ) : ℕ := 2 * n - 1
def exponent (n : ℕ) : ℕ := n
def monomial (n : ℕ) : ℕ × ℕ := (coefficient n, exponent n)

-- theorem to prove the nth monomial
theorem nth_monomial_is_correct (n : ℕ) : monomial n = (2 * n - 1, n) := 
by 
    sorry

end NUMINAMATH_GPT_nth_monomial_is_correct_l403_40307


namespace NUMINAMATH_GPT_polynomial_identity_l403_40389

theorem polynomial_identity (x : ℝ) :
  (x - 2)^5 + 5 * (x - 2)^4 + 10 * (x - 2)^3 + 10 * (x - 2)^2 + 5 * (x - 2) + 1 = (x - 1)^5 := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_identity_l403_40389


namespace NUMINAMATH_GPT_flora_needs_more_daily_l403_40335

-- Definitions based on conditions
def totalMilk : ℕ := 105   -- Total milk requirement in gallons
def weeks : ℕ := 3         -- Total weeks
def daysInWeek : ℕ := 7    -- Days per week
def floraPlan : ℕ := 3     -- Flora's planned gallons per day

-- Proof statement
theorem flora_needs_more_daily : (totalMilk / (weeks * daysInWeek)) - floraPlan = 2 := 
by
  sorry

end NUMINAMATH_GPT_flora_needs_more_daily_l403_40335


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l403_40327

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := { y | ∃ x ∈ A, y = x + 1 }

theorem intersection_of_A_and_B :
  A ∩ B = {2, 3, 4} :=
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l403_40327


namespace NUMINAMATH_GPT_find_line_equation_l403_40369

theorem find_line_equation (k : ℝ) (x y : ℝ) :
  (∀ k, (∃ x y, y = k * x + 1 ∧ x^2 + y^2 - 2 * x - 3 = 0) ↔ x - y + 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_find_line_equation_l403_40369


namespace NUMINAMATH_GPT_age_difference_ratio_l403_40371

theorem age_difference_ratio (R J K : ℕ) 
  (h1 : R = J + 8)
  (h2 : R + 2 = 2 * (J + 2))
  (h3 : (R + 2) * (K + 2) = 192) :
  (R - J) / (R - K) = 2 := by
  sorry

end NUMINAMATH_GPT_age_difference_ratio_l403_40371


namespace NUMINAMATH_GPT_prime_divisibility_l403_40323

theorem prime_divisibility
  (a b : ℕ) (p q : ℕ) 
  (hp : Nat.Prime p) 
  (hq : Nat.Prime q) 
  (hm1 : ¬ p ∣ q - 1)
  (hm2 : q ∣ a ^ p - b ^ p) : q ∣ a - b :=
sorry

end NUMINAMATH_GPT_prime_divisibility_l403_40323


namespace NUMINAMATH_GPT_triangle_largest_angle_l403_40353

theorem triangle_largest_angle (A B C : ℚ) (sinA sinB sinC : ℚ) 
(h_ratio : sinA / sinB = 3 / 5)
(h_ratio2 : sinB / sinC = 5 / 7)
(h_sum : A + B + C = 180) : C = 120 := 
sorry

end NUMINAMATH_GPT_triangle_largest_angle_l403_40353


namespace NUMINAMATH_GPT_a_5_eq_31_l403_40302

def seq (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ (∀ n, a (n + 1) = 2 * a n + 1)

theorem a_5_eq_31 (a : ℕ → ℕ) (h : seq a) : a 5 = 31 :=
by
  sorry
 
end NUMINAMATH_GPT_a_5_eq_31_l403_40302


namespace NUMINAMATH_GPT_cars_sold_proof_l403_40348

noncomputable def total_cars_sold : Nat := 300
noncomputable def perc_audi : ℝ := 0.10
noncomputable def perc_toyota : ℝ := 0.15
noncomputable def perc_acura : ℝ := 0.20
noncomputable def perc_honda : ℝ := 0.18

theorem cars_sold_proof : total_cars_sold * (1 - (perc_audi + perc_toyota + perc_acura + perc_honda)) = 111 := by
  sorry

end NUMINAMATH_GPT_cars_sold_proof_l403_40348


namespace NUMINAMATH_GPT_length_of_bridge_l403_40344

/-- A train that is 357 meters long is running at a speed of 42 km/hour. 
    It takes 42.34285714285714 seconds to pass a bridge. 
    Prove that the length of the bridge is 136.7142857142857 meters. -/
theorem length_of_bridge : 
  let train_length := 357 -- meters
  let speed_kmh := 42 -- km/hour
  let passing_time := 42.34285714285714 -- seconds
  let speed_mps := 42 * (1000 / 3600) -- meters/second
  let total_distance := speed_mps * passing_time -- meters
  let bridge_length := total_distance - train_length -- meters
  bridge_length = 136.7142857142857 :=
by
  sorry

end NUMINAMATH_GPT_length_of_bridge_l403_40344


namespace NUMINAMATH_GPT_intersection_A_B_l403_40346

def setA : Set ℝ := { x | |x| < 2 }
def setB : Set ℝ := { x | x^2 - 4 * x + 3 < 0 }
def setC : Set ℝ := { x | 1 < x ∧ x < 2 }

theorem intersection_A_B : setA ∩ setB = setC := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l403_40346


namespace NUMINAMATH_GPT_Jungkook_has_bigger_number_l403_40392

theorem Jungkook_has_bigger_number : (3 + 6) > 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_Jungkook_has_bigger_number_l403_40392


namespace NUMINAMATH_GPT_solve_quadratics_l403_40325

theorem solve_quadratics (x : ℝ) :
  (x^2 - 7 * x - 18 = 0 → x = 9 ∨ x = -2) ∧
  (4 * x^2 + 1 = 4 * x → x = 1/2) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratics_l403_40325


namespace NUMINAMATH_GPT_union_sets_l403_40354

def S : Set ℕ := {0, 1}
def T : Set ℕ := {0, 3}

theorem union_sets : S ∪ T = {0, 1, 3} :=
by
  sorry

end NUMINAMATH_GPT_union_sets_l403_40354


namespace NUMINAMATH_GPT_total_stairs_climbed_l403_40331

theorem total_stairs_climbed (samir_stairs veronica_stairs ravi_stairs total_stairs_climbed : ℕ) 
  (h_samir : samir_stairs = 318)
  (h_veronica : veronica_stairs = (318 / 2) + 18)
  (h_ravi : ravi_stairs = (3 * veronica_stairs) / 2) :
  samir_stairs + veronica_stairs + ravi_stairs = total_stairs_climbed ->
  total_stairs_climbed = 761 :=
by
  sorry

end NUMINAMATH_GPT_total_stairs_climbed_l403_40331


namespace NUMINAMATH_GPT_total_initial_yield_l403_40311

variable (x y z : ℝ)

theorem total_initial_yield (h1 : 0.4 * x + 0.2 * y = 5) 
                           (h2 : 0.4 * y + 0.2 * z = 10) 
                           (h3 : 0.4 * z + 0.2 * x = 9) 
                           : x + y + z = 40 := 
sorry

end NUMINAMATH_GPT_total_initial_yield_l403_40311


namespace NUMINAMATH_GPT_minimum_value_of_m_plus_n_l403_40320

noncomputable def m (a b : ℝ) : ℝ := b + (1 / a)
noncomputable def n (a b : ℝ) : ℝ := a + (1 / b)

theorem minimum_value_of_m_plus_n (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 1) :
  m a b + n a b = 4 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_m_plus_n_l403_40320


namespace NUMINAMATH_GPT_polynomial_problem_l403_40343

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := 2 * x + 4

theorem polynomial_problem (f_nonzero : ∀ x, f x ≠ 0) 
  (h1 : ∀ x, f (g x) = f x * g x)
  (h2 : g 3 = 10)
  (h3 : ∃ a b, g x = a * x + b) :
  g x = 2 * x + 4 :=
sorry

end NUMINAMATH_GPT_polynomial_problem_l403_40343


namespace NUMINAMATH_GPT_perpendicular_bisector_eqn_l403_40312

-- Definitions based on given conditions
def C₁ (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ
def C₂ (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

theorem perpendicular_bisector_eqn {ρ θ : ℝ} :
  (∃ A B : ℝ × ℝ,
    A ∈ {p : ℝ × ℝ | ∃ ρ θ, p = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ C₁ ρ θ} ∧
    B ∈ {p : ℝ × ℝ | ∃ ρ θ, p = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ C₂ ρ θ}) →
  ρ * Real.sin θ + ρ * Real.cos θ = 1 :=
sorry

end NUMINAMATH_GPT_perpendicular_bisector_eqn_l403_40312


namespace NUMINAMATH_GPT_rupert_jumps_more_l403_40352

theorem rupert_jumps_more (Ronald_jumps Rupert_jumps total_jumps : ℕ)
  (h1 : Ronald_jumps = 157)
  (h2 : total_jumps = 243)
  (h3 : Rupert_jumps + Ronald_jumps = total_jumps) :
  Rupert_jumps - Ronald_jumps = 86 :=
by
  sorry

end NUMINAMATH_GPT_rupert_jumps_more_l403_40352


namespace NUMINAMATH_GPT_product_of_c_values_l403_40376

theorem product_of_c_values :
  ∃ (c1 c2 : ℕ), (c1 > 0 ∧ c2 > 0) ∧
  (∃ (x1 x2 : ℚ), (7 * x1^2 + 15 * x1 + c1 = 0) ∧ (7 * x2^2 + 15 * x2 + c2 = 0)) ∧
  (c1 * c2 = 16) :=
sorry

end NUMINAMATH_GPT_product_of_c_values_l403_40376


namespace NUMINAMATH_GPT_simplify_and_evaluate_l403_40384

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sin (Real.pi / 6)) :
  (1 - 2 / (x - 1)) / ((x - 3) / (x^2 - 1)) = 3 / 2 :=
by
  -- simplify and evaluate the expression given the condition on x
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l403_40384


namespace NUMINAMATH_GPT_intersection_of_A_B_find_a_b_l403_40386

-- Lean 4 definitions based on the given conditions
def setA (x : ℝ) : Prop := 4 - x^2 > 0
def setB (x : ℝ) (y : ℝ) : Prop := y = Real.log (-x^2 + 2*x + 3) ∧ -x^2 + 2*x + 3 > 0

-- Prove the intersection of sets A and B
theorem intersection_of_A_B :
  {x : ℝ | setA x} ∩ {x : ℝ | ∃ y : ℝ, setB x y} = {x : ℝ | -2 < x ∧ x < 1} :=
by
  sorry

-- On the roots of the quadratic equation and solution interval of inequality
theorem find_a_b (a b : ℝ) :
  (∀ x : ℝ, 2 * x^2 + a * x + b < 0 ↔ -3 < x ∧ x < 1) →
  a = 4 ∧ b = -6 :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_B_find_a_b_l403_40386


namespace NUMINAMATH_GPT_seven_distinct_numbers_with_reversed_digits_l403_40328

theorem seven_distinct_numbers_with_reversed_digits (x y : ℕ) :
  (∃ a b c d e f g : ℕ, 
  (10 * a + b + 18 = 10 * b + a) ∧ (10 * c + d + 18 = 10 * d + c) ∧ 
  (10 * e + f + 18 = 10 * f + e) ∧ (10 * g + y + 18 = 10 * y + g) ∧ 
  a ≠ c ∧ a ≠ e ∧ a ≠ g ∧ 
  c ≠ e ∧ c ≠ g ∧ 
  e ≠ g ∧ 
  (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧
  (1 ≤ c ∧ c ≤ 9) ∧ (1 ≤ d ∧ d ≤ 9) ∧
  (1 ≤ e ∧ e <= 9) ∧ (1 ≤ f ∧ f <= 9) ∧
  (1 ≤ g ∧ g <= 9) ∧ (1 ≤ y ∧ y <= 9)) :=
sorry

end NUMINAMATH_GPT_seven_distinct_numbers_with_reversed_digits_l403_40328


namespace NUMINAMATH_GPT_max_min_diff_eq_l403_40378

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 2*x + 2) - Real.sqrt (x^2 - 3*x + 3)

theorem max_min_diff_eq : 
  (∀ x : ℝ, ∃ max min : ℝ, max = Real.sqrt (8 - Real.sqrt 3) ∧ min = -Real.sqrt (8 - Real.sqrt 3) ∧ 
  (max - min = 2 * Real.sqrt (8 - Real.sqrt 3))) :=
sorry

end NUMINAMATH_GPT_max_min_diff_eq_l403_40378


namespace NUMINAMATH_GPT_ladder_base_distance_l403_40375

noncomputable def length_of_ladder : ℝ := 8.5
noncomputable def height_on_wall : ℝ := 7.5

theorem ladder_base_distance (x : ℝ) (h : x ^ 2 + height_on_wall ^ 2 = length_of_ladder ^ 2) :
  x = 4 :=
by sorry

end NUMINAMATH_GPT_ladder_base_distance_l403_40375


namespace NUMINAMATH_GPT_coffee_mix_price_l403_40360

theorem coffee_mix_price 
  (P : ℝ)
  (pound_2nd : ℝ := 2.45)
  (total_pounds : ℝ := 18)
  (final_price_per_pound : ℝ := 2.30)
  (pounds_each_kind : ℝ := 9) :
  9 * P + 9 * pound_2nd = total_pounds * final_price_per_pound →
  P = 2.15 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_coffee_mix_price_l403_40360


namespace NUMINAMATH_GPT_second_number_is_30_l403_40374

-- Definitions from the conditions
def second_number (x : ℕ) := x
def first_number (x : ℕ) := 2 * x
def third_number (x : ℕ) := (2 * x) / 3
def sum_of_numbers (x : ℕ) := first_number x + second_number x + third_number x

-- Lean statement
theorem second_number_is_30 (x : ℕ) (h1 : sum_of_numbers x = 110) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_second_number_is_30_l403_40374


namespace NUMINAMATH_GPT_find_all_n_l403_40394

theorem find_all_n (n : ℕ) : 
  (∀ k : ℤ, ∃ a : ℤ, (a^3 + a - k) % n = 0) ↔ (∃ j : ℕ, n = 3^j) :=
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_find_all_n_l403_40394


namespace NUMINAMATH_GPT_miranda_monthly_savings_l403_40385

noncomputable def total_cost := 260
noncomputable def sister_contribution := 50
noncomputable def months := 3

theorem miranda_monthly_savings : 
  (total_cost - sister_contribution) / months = 70 := 
by
  sorry

end NUMINAMATH_GPT_miranda_monthly_savings_l403_40385


namespace NUMINAMATH_GPT_find_hourly_rate_l403_40349

-- Definitions of conditions in a)
def hourly_rate : ℝ := sorry  -- This is what we will find.
def hours_worked : ℝ := 3
def tip_percentage : ℝ := 0.2
def total_paid : ℝ := 54

-- Functions based on the conditions
def cost_without_tip (rate : ℝ) : ℝ := hours_worked * rate
def tip_amount (rate : ℝ) : ℝ := tip_percentage * (cost_without_tip rate)
def total_cost (rate : ℝ) : ℝ := (cost_without_tip rate) + (tip_amount rate)

-- The goal is to prove that the rate is 15
theorem find_hourly_rate : total_cost 15 = total_paid :=
by
  sorry

end NUMINAMATH_GPT_find_hourly_rate_l403_40349


namespace NUMINAMATH_GPT_salt_cups_l403_40355

theorem salt_cups (S : ℕ) (h1 : 8 = S + 1) : S = 7 := by
  -- Problem conditions
  -- 1. The recipe calls for 8 cups of sugar.
  -- 2. Mary needs to add 1 more cup of sugar than cups of salt.
  -- This corresponds to h1.

  -- Prove S = 7
  sorry

end NUMINAMATH_GPT_salt_cups_l403_40355


namespace NUMINAMATH_GPT_find_k_no_solution_l403_40308

-- Conditions
def vector1 : ℝ × ℝ := (1, 3)
def direction1 : ℝ × ℝ := (5, -8)
def vector2 : ℝ × ℝ := (0, -1)
def direction2 (k : ℝ) : ℝ × ℝ := (-2, k)

-- Statement
theorem find_k_no_solution (k : ℝ) : 
  (∀ t s : ℝ, vector1 + t • direction1 ≠ vector2 + s • direction2 k) ↔ k = 16 / 5 :=
sorry

end NUMINAMATH_GPT_find_k_no_solution_l403_40308


namespace NUMINAMATH_GPT_remainder_when_divided_by_5_l403_40383

theorem remainder_when_divided_by_5 
  (n : ℕ) 
  (h : n % 10 = 7) : 
  n % 5 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_5_l403_40383


namespace NUMINAMATH_GPT_find_y_l403_40357

theorem find_y : ∀ (x y : ℤ), x > 0 ∧ y > 0 ∧ x % y = 9 ∧ (x:ℝ) / (y:ℝ) = 96.15 → y = 60 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_find_y_l403_40357


namespace NUMINAMATH_GPT_inequality_always_true_l403_40350

theorem inequality_always_true (a : ℝ) :
  (∀ x : ℝ, ax^2 - x + 1 > 0) ↔ a > 1/4 :=
sorry

end NUMINAMATH_GPT_inequality_always_true_l403_40350


namespace NUMINAMATH_GPT_find_number_l403_40377

theorem find_number (x : ℕ) (h : (x + 720) / 125 = 7392 / 462) : x = 1280 :=
sorry

end NUMINAMATH_GPT_find_number_l403_40377


namespace NUMINAMATH_GPT_dishonest_shopkeeper_weight_l403_40359

noncomputable def weight_used (gain_percent : ℝ) (correct_weight : ℝ) : ℝ :=
  correct_weight / (1 + gain_percent / 100)

theorem dishonest_shopkeeper_weight :
  weight_used 5.263157894736836 1000 = 950 := 
by
  sorry

end NUMINAMATH_GPT_dishonest_shopkeeper_weight_l403_40359


namespace NUMINAMATH_GPT_find_c_plus_d_l403_40309

theorem find_c_plus_d (a b c d : ℝ) (h1 : a + b = 12) (h2 : b + c = 9) (h3 : a + d = 6) : 
  c + d = 3 := 
sorry

end NUMINAMATH_GPT_find_c_plus_d_l403_40309


namespace NUMINAMATH_GPT_questions_two_and_four_equiv_questions_three_and_seven_equiv_l403_40356

-- Definitions representing conditions about students in classes A and B:
def ClassA (student : Student) : Prop := sorry
def ClassB (student : Student) : Prop := sorry
def taller (x y : Student) : Prop := sorry
def shorter (x y : Student) : Prop := sorry
def tallest (students : Set Student) : Student := sorry
def shortest (students : Set Student) : Student := sorry
def averageHeight (students : Set Student) : ℝ := sorry
def totalHeight (students : Set Student) : ℝ := sorry
def medianHeight (students : Set Student) : ℝ := sorry

-- Equivalence of question 2 and question 4:
theorem questions_two_and_four_equiv (students_A students_B : Set Student) :
  (∀ a ∈ students_A, ∃ b ∈ students_B, taller a b) ↔ 
  (∀ b ∈ students_B, ∃ a ∈ students_A, taller a b) :=
sorry

-- Equivalence of question 3 and question 7:
theorem questions_three_and_seven_equiv (students_A students_B : Set Student) :
  (∀ a ∈ students_A, ∃ b ∈ students_B, shorter b a) ↔ 
  (shorter (shortest students_B) (shortest students_A)) :=
sorry

end NUMINAMATH_GPT_questions_two_and_four_equiv_questions_three_and_seven_equiv_l403_40356


namespace NUMINAMATH_GPT_complex_addition_result_l403_40368

theorem complex_addition_result (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1)
  (h2 : a + b * i = (1 - i) * (2 + i)) : a + b = 2 :=
sorry

end NUMINAMATH_GPT_complex_addition_result_l403_40368


namespace NUMINAMATH_GPT_find_k_l403_40340

noncomputable section

open Polynomial

-- Define the conditions
variables (h k : Polynomial ℚ)
variables (C : k.eval (-1) = 15) (H : h.comp k = h * k) (nonzero_h : h ≠ 0)

-- The goal is to prove k(x) = x^2 + 21x - 35
theorem find_k : k = X^2 + 21 * X - 35 :=
  by sorry

end NUMINAMATH_GPT_find_k_l403_40340


namespace NUMINAMATH_GPT_geometric_sequence_value_of_m_l403_40318

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_value_of_m (r : ℝ) (hr : r ≠ 1) 
    (h1 : is_geometric_sequence a r)
    (h2 : a 5 * a 6 + a 4 * a 7 = 18) 
    (h3 : a 1 * a m = 9) :
  m = 10 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_value_of_m_l403_40318


namespace NUMINAMATH_GPT_mike_typing_time_l403_40365

-- Definitions based on the given conditions
def original_speed : ℕ := 65
def speed_reduction : ℕ := 20
def document_words : ℕ := 810
def reduced_speed : ℕ := original_speed - speed_reduction

-- The statement to prove
theorem mike_typing_time : (document_words / reduced_speed) = 18 :=
  by
    sorry

end NUMINAMATH_GPT_mike_typing_time_l403_40365


namespace NUMINAMATH_GPT_value_of_a_l403_40342

theorem value_of_a (a : ℝ) (k : ℝ) (hA : -5 = k * 3) (hB : a = k * (-6)) : a = 10 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l403_40342


namespace NUMINAMATH_GPT_number_properties_l403_40304

theorem number_properties (a b x : ℝ) 
  (h1 : a + b = 40) 
  (h2 : a * b = 375) 
  (h3 : a - b = x) : 
  (a = 25 ∧ b = 15 ∧ x = 10) ∨ (a = 15 ∧ b = 25 ∧ x = 10) :=
by
  sorry

end NUMINAMATH_GPT_number_properties_l403_40304


namespace NUMINAMATH_GPT_magic_square_x_value_l403_40324

theorem magic_square_x_value 
  (a b c d e f g h : ℤ) 
  (h1 : x + b + c = d + e + c)
  (h2 : x + f + e = a + b + d)
  (h3 : x + e + c = a + g + 19)
  (h4 : b + f + e = a + g + 96) 
  (h5 : 19 = b)
  (h6 : 96 = c)
  (h7 : 1 = f)
  (h8 : a + d + x = b + c + f) : 
    x = 200 :=
by
  sorry

end NUMINAMATH_GPT_magic_square_x_value_l403_40324


namespace NUMINAMATH_GPT_bag_contains_fifteen_balls_l403_40387

theorem bag_contains_fifteen_balls 
  (r b : ℕ) 
  (h1 : r + b = 15) 
  (h2 : (r * (r - 1)) / 210 = 1 / 21) 
  : r = 4 := 
sorry

end NUMINAMATH_GPT_bag_contains_fifteen_balls_l403_40387


namespace NUMINAMATH_GPT_tangerines_in_basket_l403_40303

/-- Let n be the initial number of tangerines in the basket. -/
theorem tangerines_in_basket
  (n : ℕ)
  (c1 : ∃ m : ℕ, m = 10) -- Minyoung ate 10 tangerines from the basket initially
  (c2 : ∃ k : ℕ, k = 6)  -- An hour later, Minyoung ate 6 more tangerines
  (c3 : n = 10 + 6)      -- The basket was empty after these were eaten
  : n = 16 := sorry

end NUMINAMATH_GPT_tangerines_in_basket_l403_40303


namespace NUMINAMATH_GPT_number_of_unit_squares_in_50th_ring_l403_40397

def nth_ring_unit_squares (n : ℕ) : ℕ :=
  8 * n

-- Statement to prove
theorem number_of_unit_squares_in_50th_ring : nth_ring_unit_squares 50 = 400 :=
by
  -- Proof steps (skip with sorry)
  sorry

end NUMINAMATH_GPT_number_of_unit_squares_in_50th_ring_l403_40397


namespace NUMINAMATH_GPT_trigonometric_identity_l403_40316

open Real

theorem trigonometric_identity
  (x : ℝ)
  (h1 : sin x * cos x = 1 / 8)
  (h2 : π / 4 < x)
  (h3 : x < π / 2) :
  cos x - sin x = - (sqrt 3 / 2) :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l403_40316


namespace NUMINAMATH_GPT_minute_hand_rotation_l403_40326

theorem minute_hand_rotation (h : ℕ) (radians_per_rotation : ℝ) : h = 5 → radians_per_rotation = 2 * Real.pi → - (h * radians_per_rotation) = -10 * Real.pi :=
by
  intros h_eq rp_eq
  rw [h_eq, rp_eq]
  sorry

end NUMINAMATH_GPT_minute_hand_rotation_l403_40326


namespace NUMINAMATH_GPT_arithmetic_seq_a8_l403_40314

theorem arithmetic_seq_a8
  (a : ℕ → ℤ)
  (h1 : a 5 = 10)
  (h2 : a 1 + a 2 + a 3 = 3) :
  a 8 = 19 := sorry

end NUMINAMATH_GPT_arithmetic_seq_a8_l403_40314


namespace NUMINAMATH_GPT_minimum_ab_condition_l403_40379

open Int

theorem minimum_ab_condition 
  (a b : ℕ) 
  (h_pos : 0 < a ∧ 0 < b)
  (h_div7_ab_sum : ab * (a + b) % 7 ≠ 0) 
  (h_div7_expansion : ((a + b) ^ 7 - a ^ 7 - b ^ 7) % 7 = 0) : 
  ab = 18 :=
sorry

end NUMINAMATH_GPT_minimum_ab_condition_l403_40379


namespace NUMINAMATH_GPT_tiles_walked_on_l403_40393

/-- 
A park has a rectangular shape with a width of 13 feet and a length of 19 feet.
Square-shaped tiles of dimension 1 foot by 1 foot cover the entire area.
The gardener walks in a straight line from one corner of the rectangle to the opposite corner.
One specific tile in the path is not to be stepped on. 
Prove that the number of tiles the gardener walks on is 30.
-/
theorem tiles_walked_on (width length gcd_width_length tiles_to_avoid : ℕ)
  (h_width : width = 13)
  (h_length : length = 19)
  (h_gcd : gcd width length = 1)
  (h_tiles_to_avoid : tiles_to_avoid = 1) : 
  (width + length - gcd_width_length - tiles_to_avoid = 30) := 
by
  sorry

end NUMINAMATH_GPT_tiles_walked_on_l403_40393


namespace NUMINAMATH_GPT_polar_coordinates_of_2_neg2_l403_40310

noncomputable def rect_to_polar_coord (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let theta := if y < 0 
                then 2 * Real.pi - Real.arctan (x / (-y)) 
                else Real.arctan (y / x)
  (r, theta)

theorem polar_coordinates_of_2_neg2 :
  rect_to_polar_coord 2 (-2) = (2 * Real.sqrt 2, 7 * Real.pi / 4) :=
by 
  sorry

end NUMINAMATH_GPT_polar_coordinates_of_2_neg2_l403_40310


namespace NUMINAMATH_GPT_interval_after_speed_limit_l403_40315

noncomputable def car_speed_before : ℝ := 80 -- speed before the sign in km/h
noncomputable def car_speed_after : ℝ := 60 -- speed after the sign in km/h
noncomputable def initial_interval : ℕ := 10 -- interval between the cars in meters

-- Convert speeds from km/h to m/s
noncomputable def v : ℝ := car_speed_before * 1000 / 3600
noncomputable def u : ℝ := car_speed_after * 1000 / 3600

-- Given the initial interval and speed before the sign, calculate the time it takes for the second car to reach the sign
noncomputable def delta_t : ℝ := initial_interval / v

-- Given u and delta_t, calculate the new interval after slowing down
noncomputable def new_interval : ℝ := u * delta_t

-- Theorem statement in Lean
theorem interval_after_speed_limit : new_interval = 7.5 :=
sorry

end NUMINAMATH_GPT_interval_after_speed_limit_l403_40315


namespace NUMINAMATH_GPT_find_x_l403_40380

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : ⌈x⌉ * x = 220) : x = 14.67 :=
sorry

end NUMINAMATH_GPT_find_x_l403_40380


namespace NUMINAMATH_GPT_largest_in_set_average_11_l403_40336

theorem largest_in_set_average_11 :
  ∃ (a_1 a_2 a_3 a_4 a_5 : ℕ), (a_1 < a_2) ∧ (a_2 < a_3) ∧ (a_3 < a_4) ∧ (a_4 < a_5) ∧
  (1 ≤ a_1 ∧ 1 ≤ a_2 ∧ 1 ≤ a_3 ∧ 1 ≤ a_4 ∧ 1 ≤ a_5) ∧
  (a_1 + a_2 + a_3 + a_4 + a_5 = 55) ∧
  (a_5 = 45) := 
sorry

end NUMINAMATH_GPT_largest_in_set_average_11_l403_40336


namespace NUMINAMATH_GPT_polygon_triangle_even_l403_40337

theorem polygon_triangle_even (n m : ℕ) (h : (3 * m - n) % 2 = 0) : (m + n) % 2 = 0 :=
sorry

noncomputable def number_of_distinct_interior_sides (n m : ℕ) : ℕ :=
(3 * m - n) / 2

noncomputable def number_of_distinct_interior_vertices (n m : ℕ) : ℕ :=
(m - n + 2) / 2

end NUMINAMATH_GPT_polygon_triangle_even_l403_40337


namespace NUMINAMATH_GPT_smallest_of_powers_l403_40388

theorem smallest_of_powers :
  (2:ℤ)^(55) < (3:ℤ)^(44) ∧ (2:ℤ)^(55) < (5:ℤ)^(33) ∧ (2:ℤ)^(55) < (6:ℤ)^(22) := by
  sorry

end NUMINAMATH_GPT_smallest_of_powers_l403_40388


namespace NUMINAMATH_GPT_donovan_correct_answers_l403_40364

variable (C : ℝ)
variable (incorrectAnswers : ℝ := 13)
variable (percentageCorrect : ℝ := 0.7292)

theorem donovan_correct_answers :
  (C / (C + incorrectAnswers)) = percentageCorrect → C = 35 := by
  sorry

end NUMINAMATH_GPT_donovan_correct_answers_l403_40364


namespace NUMINAMATH_GPT_general_term_formula_l403_40361

-- Define the given sequence as a function
def seq (n : ℕ) : ℤ :=
  match n with
  | 0 => 3
  | n + 1 => if (n % 2 = 0) then 4 * (n + 1) - 1 else -(4 * (n + 1) - 1)

-- Define the proposed general term formula
def a_n (n : ℕ) : ℤ :=
  (-1)^(n+1) * (4 * n - 1)

-- State the theorem that general term of the sequence equals the proposed formula
theorem general_term_formula : ∀ n : ℕ, seq n = a_n n := 
by
  sorry

end NUMINAMATH_GPT_general_term_formula_l403_40361


namespace NUMINAMATH_GPT_molecular_weight_constant_l403_40339

-- Define the molecular weight of bleach
def molecular_weight_bleach (num_moles : Nat) : Nat := 222

-- Theorem stating the molecular weight of any amount of bleach is 222 g/mol
theorem molecular_weight_constant (n : Nat) : molecular_weight_bleach n = 222 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_constant_l403_40339


namespace NUMINAMATH_GPT_heart_ratio_correct_l403_40332

def heart (n m : ℕ) : ℕ := n^3 + m^2

theorem heart_ratio_correct : (heart 3 5 : ℚ) / (heart 5 3) = 26 / 67 :=
by
  sorry

end NUMINAMATH_GPT_heart_ratio_correct_l403_40332


namespace NUMINAMATH_GPT_proof_problem_l403_40362

-- Definitions needed for conditions
def a := -7 / 4
def b := -2 / 3
def m : ℚ := 1  -- m can be any rational number
def n : ℚ := -m  -- since m and n are opposite numbers

-- Lean statement to prove the given problem
theorem proof_problem : 4 * a / b + 3 * (m + n) = 21 / 2 := by
  -- Definitions ensuring a, b, m, n meet the conditions
  have habs : |a| = 7 / 4 := by sorry
  have brecip : 1 / b = -3 / 2 := by sorry
  have moppos : m + n = 0 := by sorry
  sorry

end NUMINAMATH_GPT_proof_problem_l403_40362


namespace NUMINAMATH_GPT_find_divisor_l403_40382

theorem find_divisor (x : ℕ) (h : 144 = (x * 13) + 1) : x = 11 := by
  sorry

end NUMINAMATH_GPT_find_divisor_l403_40382


namespace NUMINAMATH_GPT_barrel_tank_ratio_l403_40351

theorem barrel_tank_ratio
  (B T : ℝ)
  (h1 : (3 / 4) * B = (5 / 8) * T) :
  B / T = 5 / 6 :=
sorry

end NUMINAMATH_GPT_barrel_tank_ratio_l403_40351


namespace NUMINAMATH_GPT_general_term_a_n_l403_40395

open BigOperators

variable {a : ℕ → ℝ}  -- The sequence a_n
variable {S : ℕ → ℝ}  -- The sequence sum S_n

-- Define the sum of the first n terms:
def seq_sum (a : ℕ → ℝ) (n : ℕ) := ∑ k in Finset.range (n + 1), a k

theorem general_term_a_n (h : ∀ n : ℕ, S n = 2 ^ n - 1) (n : ℕ) : a n = 2 ^ (n - 1) :=
by
  sorry

end NUMINAMATH_GPT_general_term_a_n_l403_40395


namespace NUMINAMATH_GPT_fewest_number_of_students_l403_40347

def satisfiesCongruences (n : ℕ) : Prop :=
  n % 6 = 3 ∧
  n % 7 = 4 ∧
  n % 8 = 5 ∧
  n % 9 = 2

theorem fewest_number_of_students : ∃ n : ℕ, satisfiesCongruences n ∧ n = 765 :=
by
  have h_ex : ∃ n : ℕ, satisfiesCongruences n := sorry
  obtain ⟨n, hn⟩ := h_ex
  use 765
  have h_correct : satisfiesCongruences 765 := sorry
  exact ⟨h_correct, rfl⟩

end NUMINAMATH_GPT_fewest_number_of_students_l403_40347


namespace NUMINAMATH_GPT_Tom_age_problem_l403_40300

theorem Tom_age_problem 
  (T : ℝ) 
  (h1 : T = T1 + T2 + T3 + T4) 
  (h2 : T - 3 = 3 * (T - 3 - 3 - 3 - 3)) : 
  T / 3 = 5.5 :=
by 
  -- sorry here to skip the proof
  sorry

end NUMINAMATH_GPT_Tom_age_problem_l403_40300


namespace NUMINAMATH_GPT_petrol_price_increase_l403_40345

variable (P C : ℝ)

/- The original price of petrol is P per unit, and the user consumes C units of petrol.
   The new consumption after a 28.57142857142857% reduction is (5/7) * C units.
   The expenditure remains constant, i.e., P * C = P' * (5/7) * C.
-/
theorem petrol_price_increase (h : P * C = (P * (7/5)) * (5/7) * C) :
  (P * (7/5) - P) / P * 100 = 40 :=
by
  sorry

end NUMINAMATH_GPT_petrol_price_increase_l403_40345


namespace NUMINAMATH_GPT_total_flowers_l403_40330

def number_of_pots : ℕ := 141
def flowers_per_pot : ℕ := 71

theorem total_flowers : number_of_pots * flowers_per_pot = 10011 :=
by
  -- formal proof goes here
  sorry

end NUMINAMATH_GPT_total_flowers_l403_40330


namespace NUMINAMATH_GPT_smallest_prime_perf_sqr_minus_eight_l403_40399

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def is_perf_sqr_minus_eight (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 8

theorem smallest_prime_perf_sqr_minus_eight :
  ∃ (n : ℕ), is_prime n ∧ is_perf_sqr_minus_eight n ∧ (∀ m : ℕ, is_prime m ∧ is_perf_sqr_minus_eight m → n ≤ m) :=
sorry

end NUMINAMATH_GPT_smallest_prime_perf_sqr_minus_eight_l403_40399


namespace NUMINAMATH_GPT_mike_picked_12_pears_l403_40338

theorem mike_picked_12_pears (k_picked k_gave_away k_m_together k_left m_left : ℕ) 
  (hkp : k_picked = 47) 
  (hkg : k_gave_away = 46) 
  (hkt : k_m_together = 13)
  (hkl : k_left = k_picked - k_gave_away) 
  (hlt : k_m_left = k_left + m_left) : 
  m_left = 12 := by
  sorry

end NUMINAMATH_GPT_mike_picked_12_pears_l403_40338


namespace NUMINAMATH_GPT_sum_black_cells_even_l403_40333

-- Define a rectangular board with cells colored in a chess manner.

structure ChessBoard (m n : ℕ) :=
  (cells : Fin m → Fin n → Int)
  (row_sums_even : ∀ i : Fin m, (Finset.univ.sum (λ j => cells i j)) % 2 = 0)
  (column_sums_even : ∀ j : Fin n, (Finset.univ.sum (λ i => cells i j)) % 2 = 0)

def is_black_cell (i j : ℕ) : Bool :=
  (i + j) % 2 = 0

theorem sum_black_cells_even {m n : ℕ} (B : ChessBoard m n) :
    (Finset.univ.sum (λ (i : Fin m) =>
         Finset.univ.sum (λ (j : Fin n) =>
            if (is_black_cell i.val j.val) then B.cells i j else 0))) % 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_black_cells_even_l403_40333


namespace NUMINAMATH_GPT_same_percentage_loss_as_profit_l403_40301

theorem same_percentage_loss_as_profit (CP SP L : ℝ) (h_prof : SP = 1720)
  (h_loss : L = CP - (14.67 / 100) * CP)
  (h_25_prof : 1.25 * CP = 1875) :
  L = 1280 := 
  sorry

end NUMINAMATH_GPT_same_percentage_loss_as_profit_l403_40301


namespace NUMINAMATH_GPT_find_third_number_l403_40381

theorem find_third_number (A B C : ℝ) (h1 : (A + B + C) / 3 = 48) (h2 : (A + B) / 2 = 56) : C = 32 :=
by sorry

end NUMINAMATH_GPT_find_third_number_l403_40381


namespace NUMINAMATH_GPT_heather_oranges_l403_40372

theorem heather_oranges (initial_oranges additional_oranges : ℝ) (h1 : initial_oranges = 60.5) (h2 : additional_oranges = 35.8) :
  initial_oranges + additional_oranges = 96.3 :=
by
  -- sorry is used to indicate the proof is omitted
  sorry

end NUMINAMATH_GPT_heather_oranges_l403_40372


namespace NUMINAMATH_GPT_trains_at_starting_positions_after_2016_minutes_l403_40396

-- Definitions corresponding to conditions
def round_trip_minutes (line: String) : Nat :=
  if line = "red" then 14
  else if line = "blue" then 16
  else if line = "green" then 18
  else 0

def is_multiple_of (n m : Nat) : Prop :=
  n % m = 0

-- Formalize the statement to be proven
theorem trains_at_starting_positions_after_2016_minutes :
  ∀ (line: String), 
  line = "red" ∨ line = "blue" ∨ line = "green" →
  is_multiple_of 2016 (round_trip_minutes line) :=
by
  intro line h
  cases h with
  | inl red =>
    sorry
  | inr hb =>
    cases hb with
    | inl blue =>
      sorry
    | inr green =>
      sorry

end NUMINAMATH_GPT_trains_at_starting_positions_after_2016_minutes_l403_40396


namespace NUMINAMATH_GPT_fraction_to_decimal_l403_40306

theorem fraction_to_decimal : (7 / 12 : ℝ) = 0.5833 := by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l403_40306


namespace NUMINAMATH_GPT_diminish_to_divisible_l403_40358

-- Definitions based on conditions
def LCM (a b : ℕ) : ℕ := Nat.lcm a b
def numbers : List ℕ := [12, 16, 18, 21, 28]
def lcm_numbers : ℕ := List.foldr LCM 1 numbers
def n : ℕ := 1011
def x : ℕ := 3

-- The proof problem statement
theorem diminish_to_divisible :
  ∃ x : ℕ, n - x = lcm_numbers := sorry

end NUMINAMATH_GPT_diminish_to_divisible_l403_40358


namespace NUMINAMATH_GPT_message_spread_in_24_hours_l403_40322

theorem message_spread_in_24_hours : ∃ T : ℕ, (T = (2^25 - 1)) :=
by 
  let T := 2^24 - 1
  use T
  sorry

end NUMINAMATH_GPT_message_spread_in_24_hours_l403_40322


namespace NUMINAMATH_GPT_lines_perpendicular_l403_40319

theorem lines_perpendicular 
  (a b : ℝ) (θ : ℝ)
  (L1 : ∀ x y : ℝ, x * Real.cos θ + y * Real.sin θ + a = 0)
  (L2 : ∀ x y : ℝ, x * Real.sin θ - y * Real.cos θ + b = 0)
  : ∀ m1 m2 : ℝ, m1 = -(Real.cos θ) / (Real.sin θ) → m2 = (Real.sin θ) / (Real.cos θ) → m1 * m2 = -1 :=
by 
  intros m1 m2 h1 h2
  sorry

end NUMINAMATH_GPT_lines_perpendicular_l403_40319


namespace NUMINAMATH_GPT_peanuts_in_box_l403_40370

variable (original_peanuts : Nat)
variable (additional_peanuts : Nat)

theorem peanuts_in_box (h1 : original_peanuts = 4) (h2 : additional_peanuts = 4) :
  original_peanuts + additional_peanuts = 8 := 
by
  sorry

end NUMINAMATH_GPT_peanuts_in_box_l403_40370


namespace NUMINAMATH_GPT_total_candies_third_set_l403_40366

-- Variables for the quantities of candies
variables (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)

-- Conditions as stated in the problem
axiom equal_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3
axiom first_set_chocolates_gummy : S1 = M1
axiom first_set_hard_candies : L1 = S1 + 7
axiom second_set_hard_chocolates : L2 = S2
axiom second_set_gummy_candies : M2 = L2 - 15
axiom third_set_no_hard_candies : L3 = 0

-- Theorem to be proven: the number of candies in the third set
theorem total_candies_third_set : 
  L3 + S3 + M3 = 29 :=
by
  sorry

end NUMINAMATH_GPT_total_candies_third_set_l403_40366


namespace NUMINAMATH_GPT_problem_statement_l403_40317

def A := {x : ℝ | x * (x - 1) < 0}
def B := {y : ℝ | ∃ x : ℝ, y = x^2}

theorem problem_statement : A ⊆ {y : ℝ | y ≥ 0} :=
sorry

end NUMINAMATH_GPT_problem_statement_l403_40317


namespace NUMINAMATH_GPT_compare_M_N_l403_40367

variables (a : ℝ)

-- Definitions based on given conditions
def M : ℝ := 2 * a * (a - 2) + 3
def N : ℝ := (a - 1) * (a - 3)

theorem compare_M_N : M a ≥ N a := 
by {
  sorry
}

end NUMINAMATH_GPT_compare_M_N_l403_40367


namespace NUMINAMATH_GPT_complement_B_in_A_l403_40329

noncomputable def A : Set ℝ := {x | x < 2}
noncomputable def B : Set ℝ := {x | 1 < x ∧ x < 2}

theorem complement_B_in_A : {x | x ∈ A ∧ x ∉ B} = {x | x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_complement_B_in_A_l403_40329


namespace NUMINAMATH_GPT_probability_exactly_one_solves_problem_l403_40305

-- Define the context in which A and B solve the problem with given probabilities.
variables (p1 p2 : ℝ)

-- Define the constraint that the probabilities are between 0 and 1
axiom prob_A_nonneg : 0 ≤ p1
axiom prob_A_le_one : p1 ≤ 1
axiom prob_B_nonneg : 0 ≤ p2
axiom prob_B_le_one : p2 ≤ 1

-- Define the context that A and B solve the problem independently.
axiom A_and_B_independent : true

-- The theorem statement to prove the desired probability of exactly one solving the problem.
theorem probability_exactly_one_solves_problem : (p1 * (1 - p2) + p2 * (1 - p1)) =  p1 * (1 - p2) + p2 * (1 - p1) :=
by
  sorry

end NUMINAMATH_GPT_probability_exactly_one_solves_problem_l403_40305


namespace NUMINAMATH_GPT_find_t_l403_40313

theorem find_t (s t : ℝ) (h1 : 15 * s + 7 * t = 236) (h2 : t = 2 * s + 1) : t = 16.793 :=
by
  sorry

end NUMINAMATH_GPT_find_t_l403_40313
