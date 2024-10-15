import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_equation_l835_83511

/-- Represents a hyperbola with foci on the x-axis -/
structure Hyperbola where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  c : ℝ  -- focal distance
  e : ℝ  -- eccentricity

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- Theorem: For a hyperbola with given properties, prove its standard equation -/
theorem hyperbola_equation (h : Hyperbola) 
  (h_b : h.b = 12)
  (h_e : h.e = 5/4)
  (h_foci : h.c^2 = h.a^2 + h.b^2)
  (x y : ℝ) :
  standard_equation h x y ↔ x^2 / 64 - y^2 / 36 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l835_83511


namespace NUMINAMATH_CALUDE_zhou_yu_age_theorem_l835_83585

/-- Represents the equation for Zhou Yu's age at death -/
def zhou_yu_age_equation (x : ℕ) : Prop :=
  x^2 = 10 * (x - 3) + x

/-- Theorem stating the conditions and the equation for Zhou Yu's age at death -/
theorem zhou_yu_age_theorem (x : ℕ) :
  (x ≥ 10 ∧ x < 100) →  -- Two-digit number
  (x / 10 = x % 10 - 3) →  -- Tens digit is 3 less than units digit
  (x^2 = 10 * (x - 3) + x) →  -- Square of units digit equals the age
  zhou_yu_age_equation x :=
by
  sorry

#check zhou_yu_age_theorem

end NUMINAMATH_CALUDE_zhou_yu_age_theorem_l835_83585


namespace NUMINAMATH_CALUDE_fill_675_cans_in_36_minutes_l835_83515

/-- A machine that fills cans of paint -/
structure PaintMachine where
  cans_per_batch : ℕ
  minutes_per_batch : ℕ

/-- Calculate the time needed to fill a given number of cans -/
def time_to_fill (machine : PaintMachine) (total_cans : ℕ) : ℕ :=
  (total_cans * machine.minutes_per_batch + machine.cans_per_batch - 1) / machine.cans_per_batch

/-- Theorem stating that it takes 36 minutes to fill 675 cans -/
theorem fill_675_cans_in_36_minutes :
  let machine : PaintMachine := { cans_per_batch := 150, minutes_per_batch := 8 }
  time_to_fill machine 675 = 36 := by
  sorry

end NUMINAMATH_CALUDE_fill_675_cans_in_36_minutes_l835_83515


namespace NUMINAMATH_CALUDE_compote_level_reduction_l835_83510

theorem compote_level_reduction (V : ℝ) (h : V > 0) :
  let initial_level := V
  let level_after_third := 3/4 * V
  let volume_of_remaining_peaches := 1/6 * V
  let final_level := level_after_third - volume_of_remaining_peaches
  (level_after_third - final_level) / level_after_third = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_compote_level_reduction_l835_83510


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_f_upper_bound_implies_a_range_l835_83583

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + Real.log x

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2 * a * x + 1 / x

theorem tangent_slope_implies_a (a : ℝ) :
  f_deriv a 1 = -1 → a = -1 := by sorry

theorem f_upper_bound_implies_a_range (a : ℝ) :
  a < 0 →
  (∀ x > 0, f a x ≤ -1/2) →
  a ≤ -1/2 := by sorry

end

end NUMINAMATH_CALUDE_tangent_slope_implies_a_f_upper_bound_implies_a_range_l835_83583


namespace NUMINAMATH_CALUDE_right_triangle_m_values_l835_83565

/-- A right triangle in a 2D Cartesian coordinate system -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∨
             (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0 ∨
             (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

/-- The theorem to be proved -/
theorem right_triangle_m_values (t : RightTriangle) 
    (h1 : t.B.1 - t.A.1 = 1 ∧ t.B.2 - t.A.2 = 1)
    (h2 : t.C.1 - t.A.1 = 2 ∧ ∃ m : ℝ, t.C.2 - t.A.2 = m) :
  ∃ m : ℝ, (t.C.2 - t.A.2 = m ∧ (m = -2 ∨ m = 0)) := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_m_values_l835_83565


namespace NUMINAMATH_CALUDE_complement_of_union_l835_83529

def U : Set ℕ := {x | x > 0 ∧ x < 9}
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {5, 6, 7}

theorem complement_of_union : 
  (U \ (M ∪ N)) = {2, 4, 8} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l835_83529


namespace NUMINAMATH_CALUDE_function_fits_data_l835_83501

/-- The set of data points representing the relationship between x and y -/
def data_points : List (ℚ × ℚ) := [(0, 200), (2, 160), (4, 80), (6, 0), (8, -120)]

/-- The proposed quadratic function -/
def f (x : ℚ) : ℚ := -10 * x^2 + 200

/-- Theorem stating that the proposed function fits all data points -/
theorem function_fits_data : ∀ (point : ℚ × ℚ), point ∈ data_points → f point.1 = point.2 := by
  sorry

end NUMINAMATH_CALUDE_function_fits_data_l835_83501


namespace NUMINAMATH_CALUDE_area_bounded_by_cos_sin_squared_l835_83539

theorem area_bounded_by_cos_sin_squared (f : ℝ → ℝ) (h : ∀ x, f x = Real.cos x * Real.sin x ^ 2) :
  ∫ x in (0)..(Real.pi / 2), f x = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_area_bounded_by_cos_sin_squared_l835_83539


namespace NUMINAMATH_CALUDE_consecutive_odd_sum_fourth_power_l835_83554

theorem consecutive_odd_sum_fourth_power (a b c : ℕ) : 
  (∃ n : ℕ, n < 10 ∧ a + b + c = n^4) ∧ 
  (Odd a ∧ Odd b ∧ Odd c) ∧
  (b = a + 2 ∧ c = b + 2) →
  ((a, b, c) = (25, 27, 29) ∨ (a, b, c) = (2185, 2187, 2189)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_sum_fourth_power_l835_83554


namespace NUMINAMATH_CALUDE_crackers_per_friend_l835_83548

theorem crackers_per_friend (initial_crackers : ℕ) (friends : ℕ) (remaining_crackers : ℕ) 
  (h1 : initial_crackers = 15)
  (h2 : friends = 5)
  (h3 : remaining_crackers = 10) :
  (initial_crackers - remaining_crackers) / friends = 1 := by
  sorry

end NUMINAMATH_CALUDE_crackers_per_friend_l835_83548


namespace NUMINAMATH_CALUDE_most_likely_top_quality_count_l835_83580

/-- The proportion of top-quality products -/
def p : ℝ := 0.31

/-- The number of products in the batch -/
def n : ℕ := 75

/-- The most likely number of top-quality products in the batch -/
def most_likely_count : ℕ := 23

/-- Theorem stating that the most likely number of top-quality products in the batch is 23 -/
theorem most_likely_top_quality_count :
  ⌊n * p⌋ = most_likely_count ∧
  (n * p - (1 - p) ≤ most_likely_count) ∧
  (most_likely_count ≤ n * p + p) :=
sorry

end NUMINAMATH_CALUDE_most_likely_top_quality_count_l835_83580


namespace NUMINAMATH_CALUDE_absolute_difference_of_roots_l835_83575

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 7*x + 12 = 0

-- Define the roots of the equation
noncomputable def r₁ : ℝ := sorry
noncomputable def r₂ : ℝ := sorry

-- State the theorem
theorem absolute_difference_of_roots : 
  quadratic_equation r₁ ∧ quadratic_equation r₂ → |r₁ - r₂| = 1 := by sorry

end NUMINAMATH_CALUDE_absolute_difference_of_roots_l835_83575


namespace NUMINAMATH_CALUDE_sheila_hourly_rate_l835_83505

/-- Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_long_day : ℕ
  days_long : ℕ
  hours_short_day : ℕ
  days_short : ℕ
  weekly_earnings : ℕ

/-- Calculate hourly rate given a work schedule --/
def hourly_rate (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (schedule.hours_long_day * schedule.days_long + 
                              schedule.hours_short_day * schedule.days_short)

/-- Sheila's specific work schedule --/
def sheila_schedule : WorkSchedule := {
  hours_long_day := 8,
  days_long := 3,
  hours_short_day := 6,
  days_short := 2,
  weekly_earnings := 252
}

/-- Theorem: Sheila's hourly rate is $7 --/
theorem sheila_hourly_rate : hourly_rate sheila_schedule = 7 := by
  sorry


end NUMINAMATH_CALUDE_sheila_hourly_rate_l835_83505


namespace NUMINAMATH_CALUDE_circle_equation_through_ABC_circle_equation_center_y_2_l835_83591

-- Define the circle P
def CircleP : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (c : ℝ × ℝ) (r : ℝ), (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2}

-- Define the points A, B, and C
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (4, 0)
def C : ℝ × ℝ := (6, -2)

-- Theorem 1
theorem circle_equation_through_ABC :
  A ∈ CircleP ∧ B ∈ CircleP ∧ C ∈ CircleP →
  ∃ (D E F : ℝ), ∀ (x y : ℝ), (x, y) ∈ CircleP ↔ x^2 + y^2 + D*x + E*y + F = 0 :=
sorry

-- Theorem 2
theorem circle_equation_center_y_2 :
  A ∈ CircleP ∧ B ∈ CircleP ∧ (∃ (c : ℝ × ℝ), c ∈ CircleP ∧ c.2 = 2) →
  ∃ (c : ℝ × ℝ) (r : ℝ), c = (5/2, 2) ∧ r = 5/2 ∧
    ∀ (x y : ℝ), (x, y) ∈ CircleP ↔ (x - c.1)^2 + (y - c.2)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_through_ABC_circle_equation_center_y_2_l835_83591


namespace NUMINAMATH_CALUDE_albert_and_allison_marbles_albert_and_allison_marbles_proof_l835_83503

/-- Proves that Albert and Allison have a total of 136 marbles given the conditions of the problem -/
theorem albert_and_allison_marbles : ℕ → ℕ → ℕ → Prop :=
  fun allison_marbles angela_marbles albert_marbles =>
    allison_marbles = 28 ∧
    angela_marbles = allison_marbles + 8 ∧
    albert_marbles = 3 * angela_marbles →
    albert_marbles + allison_marbles = 136

/-- Proof of the theorem -/
theorem albert_and_allison_marbles_proof :
  ∃ (allison_marbles angela_marbles albert_marbles : ℕ),
    albert_and_allison_marbles allison_marbles angela_marbles albert_marbles :=
by
  sorry

end NUMINAMATH_CALUDE_albert_and_allison_marbles_albert_and_allison_marbles_proof_l835_83503


namespace NUMINAMATH_CALUDE_profit_increase_l835_83514

theorem profit_increase (profit_1995 : ℝ) : 
  let profit_1996 := profit_1995 * 1.1
  let profit_1997 := profit_1995 * 1.3200000000000001
  (profit_1997 / profit_1996 - 1) * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_profit_increase_l835_83514


namespace NUMINAMATH_CALUDE_total_animals_l835_83577

/-- Given a field with cows, sheep, and goats, calculate the total number of animals -/
theorem total_animals (cows sheep goats : ℕ) 
  (h_cows : cows = 40)
  (h_sheep : sheep = 56)
  (h_goats : goats = 104) :
  cows + sheep + goats = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_l835_83577


namespace NUMINAMATH_CALUDE_wrench_force_calculation_l835_83573

/-- Given two wrenches with different handle lengths, calculate the force required for the second wrench -/
theorem wrench_force_calculation (l₁ l₂ f₁ : ℝ) (h₁ : l₁ > 0) (h₂ : l₂ > 0) (h₃ : f₁ > 0) :
  let f₂ := (l₁ * f₁) / l₂
  l₁ = 12 ∧ f₁ = 450 ∧ l₂ = 18 → f₂ = 300 := by
  sorry

#check wrench_force_calculation

end NUMINAMATH_CALUDE_wrench_force_calculation_l835_83573


namespace NUMINAMATH_CALUDE_cube_difference_l835_83513

theorem cube_difference (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m^2 - n^2 = 43) :
  m^3 - n^3 = 1387 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l835_83513


namespace NUMINAMATH_CALUDE_b_47_mod_49_l835_83557

/-- Definition of the sequence b_n -/
def b (n : ℕ) : ℕ := 7^n + 9^n

/-- The remainder of b_47 when divided by 49 is 14 -/
theorem b_47_mod_49 : b 47 % 49 = 14 := by
  sorry

end NUMINAMATH_CALUDE_b_47_mod_49_l835_83557


namespace NUMINAMATH_CALUDE_solve_for_p_l835_83551

theorem solve_for_p (n m p : ℚ) 
  (h1 : (3 : ℚ) / 4 = n / 48)
  (h2 : (3 : ℚ) / 4 = (m + n) / 96)
  (h3 : (3 : ℚ) / 4 = (p - m) / 160) : 
  p = 156 := by sorry

end NUMINAMATH_CALUDE_solve_for_p_l835_83551


namespace NUMINAMATH_CALUDE_parallelogram_area_l835_83579

/-- The area of a parallelogram with base 20 meters and height 4 meters is 80 square meters. -/
theorem parallelogram_area :
  let base : ℝ := 20
  let height : ℝ := 4
  let area : ℝ := base * height
  area = 80 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l835_83579


namespace NUMINAMATH_CALUDE_base7_digit_sum_l835_83593

/-- Converts a base-7 number to base-10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a base-7 number --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The main theorem --/
theorem base7_digit_sum :
  let a := toBase10 45
  let b := toBase10 16
  let c := toBase10 12
  let result := toBase7 ((a * b) + c)
  sumOfDigitsBase7 result = 17 := by sorry

end NUMINAMATH_CALUDE_base7_digit_sum_l835_83593


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l835_83572

/-- Parabola represented by the equation y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Line represented by the equation y = kx - 1 -/
def line (k x y : ℝ) : Prop := y = k*x - 1

/-- Focus of the parabola y² = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- The line passes through the focus of the parabola -/
def line_passes_through_focus (k : ℝ) : Prop :=
  line k (focus.1) (focus.2)

/-- The line intersects the parabola at two points -/
def line_intersects_parabola (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧ 
    line k x₁ y₁ ∧ line k x₂ y₂

theorem parabola_line_intersection 
  (h1 : line_passes_through_focus k)
  (h2 : line_intersects_parabola k) :
  k = 1 ∧ ∃ (x₁ x₂ : ℝ), x₁ + x₂ + 2 = 8 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l835_83572


namespace NUMINAMATH_CALUDE_area_of_square_II_l835_83532

/-- Given a square I with diagonal √3ab, prove that the area of a square II 
    with three times the area of square I is 9(ab)²/2 -/
theorem area_of_square_II (a b : ℝ) (h : a > 0 ∧ b > 0) : 
  let diagonal_I := Real.sqrt 3 * a * b
  let area_I := (diagonal_I ^ 2) / 2
  let area_II := 3 * area_I
  area_II = 9 * (a * b) ^ 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_area_of_square_II_l835_83532


namespace NUMINAMATH_CALUDE_expression_evaluation_l835_83564

theorem expression_evaluation :
  let x : ℚ := -2
  (x - 2)^2 + (2 + x)*(x - 2) - 2*x*(2*x - 1) = -4 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l835_83564


namespace NUMINAMATH_CALUDE_max_value_problem_l835_83568

theorem max_value_problem (a b c d : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  ∀ x, x = (a * b * c * d) ^ (1/4) + ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ^ (1/2) → x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_problem_l835_83568


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l835_83566

theorem right_triangle_third_side (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  ((a = Real.sqrt 2 ∧ b = Real.sqrt 3) ∨ (a = Real.sqrt 3 ∧ b = Real.sqrt 2)) →
  c = Real.sqrt 5 ∨ c = 1 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l835_83566


namespace NUMINAMATH_CALUDE_birthday_pigeonhole_l835_83525

theorem birthday_pigeonhole (n : ℕ) (h : n = 50) :
  ∃ (m : ℕ) (S : Finset (Fin n)), S.card ≥ 5 ∧ (∀ i ∈ S, (i : ℕ) % 12 + 1 = m) :=
sorry

end NUMINAMATH_CALUDE_birthday_pigeonhole_l835_83525


namespace NUMINAMATH_CALUDE_power_of_product_l835_83569

theorem power_of_product (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l835_83569


namespace NUMINAMATH_CALUDE_functional_equation_implies_g_50_eq_0_l835_83589

/-- A function satisfying the given functional equation for all positive real numbers -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → x * g y - y * g x = g (x / y) + g (x + y)

/-- The main theorem stating that any function satisfying the functional equation must have g(50) = 0 -/
theorem functional_equation_implies_g_50_eq_0 (g : ℝ → ℝ) (h : FunctionalEquation g) : g 50 = 0 := by
  sorry

#check functional_equation_implies_g_50_eq_0

end NUMINAMATH_CALUDE_functional_equation_implies_g_50_eq_0_l835_83589


namespace NUMINAMATH_CALUDE_least_odd_prime_factor_of_2100_8_plus_1_l835_83535

theorem least_odd_prime_factor_of_2100_8_plus_1 :
  (Nat.minFac (2100^8 + 1)) = 193 := by
  sorry

end NUMINAMATH_CALUDE_least_odd_prime_factor_of_2100_8_plus_1_l835_83535


namespace NUMINAMATH_CALUDE_binomial_expansion_102_l835_83552

theorem binomial_expansion_102 : 
  102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 104060401 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_102_l835_83552


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l835_83578

theorem smallest_n_square_and_cube : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (j : ℕ), 5 * n = j^3) ∧
  (∀ (m : ℕ), m > 0 → 
    (∃ (k : ℕ), 4 * m = k^2) → 
    (∃ (j : ℕ), 5 * m = j^3) → 
    m ≥ n) ∧
  n = 125 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l835_83578


namespace NUMINAMATH_CALUDE_trig_identity_l835_83509

/-- For any angle α, sin²(α) + cos²(30° + α) + sin(α)cos(30° + α) = 3/4 -/
theorem trig_identity (α : Real) : 
  (Real.sin α)^2 + (Real.cos (π/6 + α))^2 + (Real.sin α) * (Real.cos (π/6 + α)) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l835_83509


namespace NUMINAMATH_CALUDE_equation_solution_exists_l835_83581

theorem equation_solution_exists : ∃ (MA TE TI KA : ℕ),
  MA < 10 ∧ TE < 10 ∧ TI < 10 ∧ KA < 10 ∧
  MA ≠ TE ∧ MA ≠ TI ∧ MA ≠ KA ∧ TE ≠ TI ∧ TE ≠ KA ∧ TI ≠ KA ∧
  MA * TE * MA * TI * KA = 2016000 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l835_83581


namespace NUMINAMATH_CALUDE_even_quadratic_implies_k_eq_one_l835_83500

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The quadratic function f(x) = kx^2 + (k-1)x + 2 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

/-- If f(x) = kx^2 + (k-1)x + 2 is an even function, then k = 1 -/
theorem even_quadratic_implies_k_eq_one (k : ℝ) : IsEven (f k) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_quadratic_implies_k_eq_one_l835_83500


namespace NUMINAMATH_CALUDE_total_distance_to_fountain_l835_83598

/-- The distance from Mrs. Hilt's desk to the water fountain in feet -/
def distance_to_fountain : ℕ := 30

/-- The number of trips Mrs. Hilt makes to the water fountain -/
def number_of_trips : ℕ := 4

/-- Theorem: The total distance Mrs. Hilt walks to the water fountain is 120 feet -/
theorem total_distance_to_fountain :
  distance_to_fountain * number_of_trips = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_to_fountain_l835_83598


namespace NUMINAMATH_CALUDE_investment_pays_off_after_9_months_l835_83560

/-- Cumulative net income function for the first 5 months after improvement -/
def g (n : ℕ) : ℚ :=
  if n ≤ 5 then n^2 + 100*n else 109*n - 20

/-- Monthly income without improvement (in 10,000 yuan) -/
def monthly_income : ℚ := 70

/-- Fine function without improvement (in 10,000 yuan) -/
def fine (n : ℕ) : ℚ := n^2 + 2*n

/-- Initial investment (in 10,000 yuan) -/
def investment : ℚ := 500

/-- One-time reward after improvement (in 10,000 yuan) -/
def reward : ℚ := 100

/-- Cumulative net income with improvement (in 10,000 yuan) -/
def income_with_improvement (n : ℕ) : ℚ :=
  g n - investment + reward

/-- Cumulative net income without improvement (in 10,000 yuan) -/
def income_without_improvement (n : ℕ) : ℚ :=
  n * monthly_income - fine n

theorem investment_pays_off_after_9_months :
  ∀ n : ℕ, n ≥ 9 → income_with_improvement n > income_without_improvement n :=
sorry

end NUMINAMATH_CALUDE_investment_pays_off_after_9_months_l835_83560


namespace NUMINAMATH_CALUDE_simplify_2A_minus_3B_value_2A_minus_3B_specific_value_2A_minus_3B_independent_l835_83521

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 3 * x^2 - x + 2 * y - 4 * x * y
def B (x y : ℝ) : ℝ := 2 * x^2 - 3 * x - y + x * y

-- Theorem 1: Simplification of 2A - 3B
theorem simplify_2A_minus_3B (x y : ℝ) :
  2 * A x y - 3 * B x y = 7 * x + 7 * y - 11 * x * y :=
by sorry

-- Theorem 2: Value of 2A - 3B under specific conditions
theorem value_2A_minus_3B_specific (x y : ℝ) 
  (h1 : x + y = 6/7) (h2 : x * y = -1) :
  2 * A x y - 3 * B x y = 17 :=
by sorry

-- Theorem 3: Value of 2A - 3B when independent of y
theorem value_2A_minus_3B_independent (x : ℝ) 
  (h : ∀ y : ℝ, 2 * A x y - 3 * B x y = 2 * A x 0 - 3 * B x 0) :
  2 * A x 0 - 3 * B x 0 = 49/11 :=
by sorry

end NUMINAMATH_CALUDE_simplify_2A_minus_3B_value_2A_minus_3B_specific_value_2A_minus_3B_independent_l835_83521


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l835_83530

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 → x^2 - 2*k*x + 2*k - 1 > 0) ↔ k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l835_83530


namespace NUMINAMATH_CALUDE_xy_value_l835_83561

theorem xy_value (x y : ℝ) 
  (h1 : (8:ℝ)^x / 2^(x+y) = 16)
  (h2 : (16:ℝ)^(x+y) / 4^(5*y) = 1024) : 
  x * y = 7/8 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l835_83561


namespace NUMINAMATH_CALUDE_smaller_partner_profit_theorem_l835_83586

/-- Represents a partnership between two individuals -/
structure Partnership where
  investment_ratio : ℚ  -- Ratio of investments (larger / smaller)
  time_ratio : ℚ        -- Ratio of investment periods (longer / shorter)
  total_profit : ℕ      -- Total profit in rupees

/-- Calculates the profit of the partner with the smaller investment -/
def smaller_partner_profit (p : Partnership) : ℚ :=
  p.total_profit * (1 / (1 + p.investment_ratio * p.time_ratio))

/-- Theorem stating the profit of the partner with smaller investment -/
theorem smaller_partner_profit_theorem (p : Partnership) 
  (h1 : p.investment_ratio = 3)
  (h2 : p.time_ratio = 2)
  (h3 : p.total_profit = 35000) :
  ⌊smaller_partner_profit p⌋ = 5000 := by
  sorry

#eval ⌊smaller_partner_profit ⟨3, 2, 35000⟩⌋

end NUMINAMATH_CALUDE_smaller_partner_profit_theorem_l835_83586


namespace NUMINAMATH_CALUDE_wire_ratio_theorem_l835_83534

theorem wire_ratio_theorem (B C : ℝ) (h1 : B > 0) (h2 : C > 0) (h3 : B + C = 80) : 
  ∃ (r : ℝ → ℝ → ℝ → Prop), r 16 B C ∧ 
  (∀ (x y z : ℝ), r x y z ↔ ∃ (k : ℝ), k > 0 ∧ x = 16 * k ∧ y = B * k ∧ z = C * k) :=
sorry

end NUMINAMATH_CALUDE_wire_ratio_theorem_l835_83534


namespace NUMINAMATH_CALUDE_trig_fraction_equals_four_fifths_l835_83519

theorem trig_fraction_equals_four_fifths (θ : ℝ) (h : Real.tan θ = 2) :
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_equals_four_fifths_l835_83519


namespace NUMINAMATH_CALUDE_median_and_mode_of_scores_l835_83528

/-- Represents the score distribution of students in the competition -/
def score_distribution : List (Nat × Nat) :=
  [(85, 1), (88, 7), (90, 11), (93, 10), (94, 13), (97, 7), (99, 1)]

/-- The total number of students -/
def total_students : Nat := 50

/-- Calculates the median of the given score distribution -/
def median (dist : List (Nat × Nat)) (total : Nat) : Nat :=
  sorry

/-- Calculates the mode of the given score distribution -/
def mode (dist : List (Nat × Nat)) : Nat :=
  sorry

/-- Theorem stating that the median is 93 and the mode is 94 for the given distribution -/
theorem median_and_mode_of_scores :
  median score_distribution total_students = 93 ∧
  mode score_distribution = 94 :=
sorry

end NUMINAMATH_CALUDE_median_and_mode_of_scores_l835_83528


namespace NUMINAMATH_CALUDE_train_distance_l835_83517

/-- Proves that a train traveling at 3 m/s for 9 seconds covers 27 meters -/
theorem train_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 3 → time = 9 → distance = speed * time → distance = 27 :=
by sorry

end NUMINAMATH_CALUDE_train_distance_l835_83517


namespace NUMINAMATH_CALUDE_rectangle_areas_sum_l835_83518

theorem rectangle_areas_sum : 
  let width := 3
  let lengths := [1, 8, 27, 64, 125, 216]
  let areas := lengths.map (λ l => width * l)
  areas.sum = 1323 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_areas_sum_l835_83518


namespace NUMINAMATH_CALUDE_digit_250_of_13_over_17_is_8_l835_83570

/-- The 250th decimal digit of 13/17 -/
def digit_250_of_13_over_17 : ℕ :=
  let decimal_expansion := (13 : ℚ) / 17
  let period := 16
  let position_in_period := 250 % period
  8

/-- Theorem: The 250th decimal digit in the decimal representation of 13/17 is 8 -/
theorem digit_250_of_13_over_17_is_8 :
  digit_250_of_13_over_17 = 8 := by
  sorry

end NUMINAMATH_CALUDE_digit_250_of_13_over_17_is_8_l835_83570


namespace NUMINAMATH_CALUDE_max_factors_is_231_l835_83544

/-- The number of positive factors of b^n, where b and n are positive integers -/
def num_factors (b n : ℕ+) : ℕ := sorry

/-- The maximum number of positive factors for b^n given constraints -/
def max_num_factors : ℕ := sorry

theorem max_factors_is_231 :
  ∀ b n : ℕ+, b ≤ 20 → n ≤ 10 → num_factors b n ≤ max_num_factors ∧ max_num_factors = 231 := by sorry

end NUMINAMATH_CALUDE_max_factors_is_231_l835_83544


namespace NUMINAMATH_CALUDE_no_distinct_complex_numbers_satisfying_equations_l835_83562

theorem no_distinct_complex_numbers_satisfying_equations :
  ∀ (a b c d : ℂ),
  (a^3 - b*c*d = b^3 - c*d*a) ∧
  (b^3 - c*d*a = c^3 - d*a*b) ∧
  (c^3 - d*a*b = d^3 - a*b*c) →
  ¬(a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :=
by sorry

end NUMINAMATH_CALUDE_no_distinct_complex_numbers_satisfying_equations_l835_83562


namespace NUMINAMATH_CALUDE_y_intercept_of_specific_line_l835_83524

/-- A line is defined by its slope and a point it passes through -/
structure Line where
  slope : ℚ
  point : ℚ × ℚ

/-- The y-intercept of a line is the y-coordinate where the line crosses the y-axis -/
def y_intercept (l : Line) : ℚ := 
  l.point.2 - l.slope * l.point.1

theorem y_intercept_of_specific_line : 
  let l : Line := { slope := -3/2, point := (4, 0) }
  y_intercept l = 6 := by
  sorry

#check y_intercept_of_specific_line

end NUMINAMATH_CALUDE_y_intercept_of_specific_line_l835_83524


namespace NUMINAMATH_CALUDE_lemonade_third_intermission_l835_83504

theorem lemonade_third_intermission 
  (total : ℝ) 
  (first : ℝ) 
  (second : ℝ) 
  (h1 : total = 0.9166666666666666) 
  (h2 : first = 0.25) 
  (h3 : second = 0.4166666666666667) :
  total - (first + second) = 0.25 := by
sorry

end NUMINAMATH_CALUDE_lemonade_third_intermission_l835_83504


namespace NUMINAMATH_CALUDE_variance_invariant_under_translation_mutually_exclusive_events_l835_83506

-- Define a dataset as a list of real numbers
def Dataset := List Real

-- Define variance function
noncomputable def variance (data : Dataset) : Real := sorry

-- Define a function to add a constant to each element of a dataset
def addConstant (data : Dataset) (c : Real) : Dataset := sorry

-- Statement 1: Variance remains unchanged after adding a constant
theorem variance_invariant_under_translation (data : Dataset) (c : Real) :
  variance (addConstant data c) = variance data := by sorry

-- Define a type for students
inductive Student
| Boy
| Girl

-- Define a function to create a group of students
def createGroup (numBoys numGirls : Nat) : List Student := sorry

-- Define a function to select n students from a group
def selectStudents (group : List Student) (n : Nat) : List (List Student) := sorry

-- Define predicates for the events
def atLeastOneGirl (selection : List Student) : Prop := sorry
def allBoys (selection : List Student) : Prop := sorry

-- Statement 2: "At least 1 girl" and "all boys" are mutually exclusive when selecting 2 from 3 boys and 2 girls
theorem mutually_exclusive_events :
  let group := createGroup 3 2
  let selections := selectStudents group 2
  ∀ selection ∈ selections, ¬(atLeastOneGirl selection ∧ allBoys selection) := by sorry

end NUMINAMATH_CALUDE_variance_invariant_under_translation_mutually_exclusive_events_l835_83506


namespace NUMINAMATH_CALUDE_notebook_savings_correct_l835_83588

def notebook_savings (quantity : ℕ) (original_price : ℝ) (individual_discount_rate : ℝ) (bulk_discount_rate : ℝ) (bulk_discount_threshold : ℕ) : ℝ :=
  let discounted_price := original_price * (1 - individual_discount_rate)
  let total_without_discount := quantity * original_price
  let total_with_individual_discount := quantity * discounted_price
  let final_total := if quantity > bulk_discount_threshold
                     then total_with_individual_discount * (1 - bulk_discount_rate)
                     else total_with_individual_discount
  total_without_discount - final_total

theorem notebook_savings_correct :
  notebook_savings 8 3 0.1 0.05 6 = 3.48 :=
sorry

end NUMINAMATH_CALUDE_notebook_savings_correct_l835_83588


namespace NUMINAMATH_CALUDE_percentage_of_male_students_l835_83516

theorem percentage_of_male_students (male_percentage : ℝ) 
  (h1 : 0 ≤ male_percentage ∧ male_percentage ≤ 100)
  (h2 : 50 = 100 * (1 - (male_percentage / 100 * 0.5 + (100 - male_percentage) / 100 * 0.6)))
  : male_percentage = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_male_students_l835_83516


namespace NUMINAMATH_CALUDE_decimal_to_binary_21_l835_83558

theorem decimal_to_binary_21 : 
  (21 : ℕ) = (1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) :=
by sorry

end NUMINAMATH_CALUDE_decimal_to_binary_21_l835_83558


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l835_83584

/-- Given a line passing through two points, prove that the sum of its slope and y-intercept is -5/2 --/
theorem line_slope_intercept_sum (m b : ℚ) : 
  ((-1 : ℚ) = m * (1/2) + b) → 
  (2 = m * (-1/2) + b) → 
  m + b = -5/2 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l835_83584


namespace NUMINAMATH_CALUDE_quadratic_factorization_l835_83595

theorem quadratic_factorization (m : ℝ) : 2 * m^2 - 12 * m + 18 = 2 * (m - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l835_83595


namespace NUMINAMATH_CALUDE_polynomial_factorization_l835_83556

theorem polynomial_factorization :
  ∀ x : ℝ, (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 7) = 
           (x^2 + 7*x + 2) * (x^2 + 5*x + 19) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l835_83556


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l835_83541

theorem inequality_holds_iff (n : ℕ) :
  (∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 2 →
    (1 / a^n + 1 / b^n ≥ a^m + b^m)) ↔ (m = n ∨ m = n + 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l835_83541


namespace NUMINAMATH_CALUDE_problem_solution_l835_83594

theorem problem_solution (a b c d : ℕ+) 
  (h1 : a^6 = b^5) 
  (h2 : c^4 = d^3) 
  (h3 : c - a = 31) : 
  c - b = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l835_83594


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l835_83576

theorem sufficient_but_not_necessary (a b : ℝ) : 
  (((0 ≤ a) ∧ (a ≤ 1) ∧ (0 ≤ b) ∧ (b ≤ 1)) → (0 ≤ a * b) ∧ (a * b ≤ 1)) ∧ 
  (∃ (a b : ℝ), ((0 ≤ a * b) ∧ (a * b ≤ 1)) ∧ ¬((0 ≤ a) ∧ (a ≤ 1) ∧ (0 ≤ b) ∧ (b ≤ 1))) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l835_83576


namespace NUMINAMATH_CALUDE_major_axis_length_l835_83527

/-- Rectangle PQRS with ellipse passing through P and R, foci at Q and S -/
structure EllipseInRectangle where
  /-- Area of the rectangle PQRS -/
  rect_area : ℝ
  /-- Area of the ellipse -/
  ellipse_area : ℝ
  /-- The ellipse passes through P and R, and has foci at Q and S -/
  ellipse_through_PR_foci_QS : Bool

/-- Given the specific rectangle and ellipse, prove the length of the major axis -/
theorem major_axis_length (e : EllipseInRectangle) 
  (h1 : e.rect_area = 4050)
  (h2 : e.ellipse_area = 3240 * Real.pi)
  (h3 : e.ellipse_through_PR_foci_QS = true) : 
  ∃ (major_axis : ℝ), major_axis = 144 := by
  sorry

end NUMINAMATH_CALUDE_major_axis_length_l835_83527


namespace NUMINAMATH_CALUDE_min_ones_23x23_l835_83536

/-- Represents a tiling of a square grid --/
structure Tiling (n : ℕ) :=
  (ones : ℕ)
  (twos : ℕ)
  (threes : ℕ)
  (valid : ones + 4 * twos + 9 * threes = n^2)

/-- The minimum number of 1x1 squares in a valid 23x23 tiling --/
def min_ones : ℕ := 1

theorem min_ones_23x23 :
  ∀ (t : Tiling 23), t.ones ≥ min_ones :=
sorry

end NUMINAMATH_CALUDE_min_ones_23x23_l835_83536


namespace NUMINAMATH_CALUDE_f_is_quadratic_l835_83553

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² - 2x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l835_83553


namespace NUMINAMATH_CALUDE_door_height_is_eight_l835_83522

/-- Represents the dimensions of a rectangular door and a pole. -/
structure DoorPole where
  pole_length : ℝ
  door_width : ℝ
  door_height : ℝ
  door_diagonal : ℝ

/-- The conditions of the door and pole problem. -/
def door_pole_conditions (d : DoorPole) : Prop :=
  d.pole_length = d.door_width + 4 ∧
  d.pole_length = d.door_height + 2 ∧
  d.pole_length = d.door_diagonal ∧
  d.door_diagonal^2 = d.door_width^2 + d.door_height^2

/-- The theorem stating that under the given conditions, the door height is 8 feet. -/
theorem door_height_is_eight (d : DoorPole) 
  (h : door_pole_conditions d) : d.door_height = 8 := by
  sorry

end NUMINAMATH_CALUDE_door_height_is_eight_l835_83522


namespace NUMINAMATH_CALUDE_range_of_a_l835_83597

-- Define the propositions
def P (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0
def Q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (-(5-2*a))^x > (-(5-2*a))^y

-- State the theorem
theorem range_of_a :
  (∃ a : ℝ, (¬(P a) ∧ Q a) ∨ (P a ∧ ¬(Q a))) →
  (∃ a : ℝ, a ≤ -2 ∧ ∀ b : ℝ, b ≤ -2 → (¬(P b) ∧ Q b) ∨ (P b ∧ ¬(Q b))) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l835_83597


namespace NUMINAMATH_CALUDE_class_composition_l835_83571

theorem class_composition (boys_avg : ℝ) (girls_avg : ℝ) (class_avg : ℝ) :
  boys_avg = 4 →
  girls_avg = 3.25 →
  class_avg = 3.6 →
  ∃ (boys girls : ℕ),
    boys + girls > 30 ∧
    boys + girls < 50 ∧
    (boys_avg * boys + girls_avg * girls) / (boys + girls) = class_avg ∧
    boys = 21 ∧
    girls = 24 := by
  sorry

end NUMINAMATH_CALUDE_class_composition_l835_83571


namespace NUMINAMATH_CALUDE_greatest_integer_third_side_l835_83543

theorem greatest_integer_third_side (a b : ℝ) (ha : a = 7) (hb : b = 11) :
  ∃ (c : ℕ), c = 17 ∧ 
  (∀ (x : ℕ), x > c → ¬(a + b > x ∧ a + x > b ∧ b + x > a)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_third_side_l835_83543


namespace NUMINAMATH_CALUDE_exists_function_satisfying_condition_l835_83520

theorem exists_function_satisfying_condition : ∃ f : ℕ → ℕ, 
  (∀ m n : ℕ, f (m + n) ≥ f m + f (f n) - 1) ∧ 
  f 2019 = 2019 := by
  sorry

end NUMINAMATH_CALUDE_exists_function_satisfying_condition_l835_83520


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l835_83559

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other -/
def symmetric_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

/-- Given that point A(m, 1) is symmetric to point B(2, n) with respect to the x-axis,
    prove that m + n = 1 -/
theorem symmetric_points_sum (m n : ℝ) :
  symmetric_x_axis (m, 1) (2, n) → m + n = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l835_83559


namespace NUMINAMATH_CALUDE_remainder_theorem_l835_83590

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 40 * k - 1) :
  (n^2 - 3*n + 5) % 40 = 9 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l835_83590


namespace NUMINAMATH_CALUDE_factor_expression_l835_83596

theorem factor_expression (x : ℝ) : 72 * x^5 - 162 * x^9 = -18 * x^5 * (9 * x^4 - 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l835_83596


namespace NUMINAMATH_CALUDE_student_number_problem_l835_83592

theorem student_number_problem (x : ℝ) : (3/2 : ℝ) * x + 53.4 = -78.9 → x = -88.2 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l835_83592


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_two_l835_83512

def a : Fin 2 → ℝ := ![1, 1]
def b (x : ℝ) : Fin 2 → ℝ := ![2, x]

theorem parallel_vectors_imply_x_equals_two :
  ∀ x : ℝ, (∃ k : ℝ, k ≠ 0 ∧ (a + b x) = k • (4 • b x - 2 • a)) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_two_l835_83512


namespace NUMINAMATH_CALUDE_f_of_one_equals_two_l835_83537

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + x

-- State the theorem
theorem f_of_one_equals_two : f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_one_equals_two_l835_83537


namespace NUMINAMATH_CALUDE_quadratic_function_passes_through_points_l835_83549

/-- The quadratic function f(x) = x² + 2x - 3 passes through the points (0, -3), (1, 0), and (-3, 0). -/
theorem quadratic_function_passes_through_points :
  let f : ℝ → ℝ := λ x ↦ x^2 + 2*x - 3
  f 0 = -3 ∧ f 1 = 0 ∧ f (-3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_passes_through_points_l835_83549


namespace NUMINAMATH_CALUDE_unique_intersection_point_l835_83574

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 5*x^2 + 10*x + 20

-- State the theorem
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = g p.2 ∧ p.2 = g p.1 ∧ p = (-4, -4) :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l835_83574


namespace NUMINAMATH_CALUDE_evenBlueFaceCubesFor642Block_l835_83531

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : Nat
  width : Nat
  height : Nat

/-- Counts the number of cubes with an even number of blue faces in a painted block -/
def evenBlueFaceCubes (b : Block) : Nat :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that a 6x4x2 inch block has 20 cubes with an even number of blue faces -/
theorem evenBlueFaceCubesFor642Block :
  evenBlueFaceCubes { length := 6, width := 4, height := 2 } = 20 := by
  sorry

end NUMINAMATH_CALUDE_evenBlueFaceCubesFor642Block_l835_83531


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l835_83507

/-- The imaginary part of (1 - i)(2 + 4i) is 2, where i is the imaginary unit. -/
theorem imaginary_part_of_product : Complex.im ((1 - Complex.I) * (2 + 4 * Complex.I)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l835_83507


namespace NUMINAMATH_CALUDE_guitar_ratio_proof_l835_83582

/-- Proves that the ratio of Barbeck's guitars to Steve's guitars is 2:1 given the problem conditions -/
theorem guitar_ratio_proof (total_guitars : ℕ) (davey_guitars : ℕ) (barbeck_guitars : ℕ) (steve_guitars : ℕ) : 
  total_guitars = 27 →
  davey_guitars = 18 →
  barbeck_guitars = steve_guitars →
  davey_guitars = 3 * barbeck_guitars →
  total_guitars = davey_guitars + barbeck_guitars + steve_guitars →
  (barbeck_guitars : ℚ) / steve_guitars = 2 / 1 :=
by sorry


end NUMINAMATH_CALUDE_guitar_ratio_proof_l835_83582


namespace NUMINAMATH_CALUDE_quadratic_sufficient_not_necessary_l835_83538

theorem quadratic_sufficient_not_necessary (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0) ↔ 
  ((a > 0 ∧ b^2 - 4*a*c < 0) ∨ 
   ∃ a' b' c' : ℝ, (∀ x : ℝ, a' * x^2 + b' * x + c' > 0) ∧ ¬(a' > 0 ∧ b'^2 - 4*a'*c' < 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sufficient_not_necessary_l835_83538


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l835_83526

theorem sum_of_two_numbers (x y : ℝ) (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 23 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l835_83526


namespace NUMINAMATH_CALUDE_festival_allowance_rate_l835_83546

/-- The daily rate for a festival allowance given the number of staff members,
    number of days, and total amount. -/
def daily_rate (staff_members : ℕ) (days : ℕ) (total_amount : ℕ) : ℚ :=
  total_amount / (staff_members * days)

/-- Theorem stating that the daily rate for the festival allowance is 110
    given the problem conditions. -/
theorem festival_allowance_rate : 
  daily_rate 20 30 66000 = 110 := by sorry

end NUMINAMATH_CALUDE_festival_allowance_rate_l835_83546


namespace NUMINAMATH_CALUDE_allyns_june_expenses_l835_83599

/-- Calculates the total monthly electricity expenses for a given number of bulbs --/
def calculate_monthly_expenses (
  bulb_wattage : ℕ)  -- Wattage of each bulb
  (num_bulbs : ℕ)    -- Number of bulbs
  (days_in_month : ℕ) -- Number of days in the month
  (cost_per_watt : ℚ) -- Cost per watt in dollars
  : ℚ :=
  (bulb_wattage * num_bulbs * days_in_month : ℚ) * cost_per_watt

/-- Theorem stating that Allyn's monthly electricity expenses for June are $14400 --/
theorem allyns_june_expenses :
  calculate_monthly_expenses 60 40 30 (20 / 100) = 14400 := by
  sorry

end NUMINAMATH_CALUDE_allyns_june_expenses_l835_83599


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l835_83545

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -9 ∧ x₂ = 1 ∧ x₁^2 + 8*x₁ - 9 = 0 ∧ x₂^2 + 8*x₂ - 9 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = -3 ∧ y₂ = 1 ∧ y₁*(y₁-1) + 3*(y₁-1) = 0 ∧ y₂*(y₂-1) + 3*(y₂-1) = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l835_83545


namespace NUMINAMATH_CALUDE_sum_of_ages_in_three_years_l835_83550

-- Define the current ages
def jeremy_current_age : ℕ := 40
def sebastian_current_age : ℕ := jeremy_current_age + 4
def sophia_future_age : ℕ := 60

-- Define the ages in three years
def jeremy_future_age : ℕ := jeremy_current_age + 3
def sebastian_future_age : ℕ := sebastian_current_age + 3
def sophia_current_age : ℕ := sophia_future_age - 3

-- Theorem to prove
theorem sum_of_ages_in_three_years :
  jeremy_future_age + sebastian_future_age + sophia_future_age = 150 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_in_three_years_l835_83550


namespace NUMINAMATH_CALUDE_abc_fraction_value_l835_83523

theorem abc_fraction_value (a b c : ℝ) 
  (h1 : a * b / (a + b) = 1 / 3)
  (h2 : b * c / (b + c) = 1 / 4)
  (h3 : c * a / (c + a) = 1 / 5) :
  a * b * c / (a * b + b * c + c * a) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_abc_fraction_value_l835_83523


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l835_83547

/-- A trinomial ax^2 + bx + c is a perfect square if there exist p and q such that
    ax^2 + bx + c = (px + q)^2 for all x. -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

/-- If 4x^2 + mx + 25 is a perfect square trinomial, then m = 20. -/
theorem perfect_square_trinomial_condition (m : ℝ) :
  is_perfect_square_trinomial 4 m 25 → m = 20 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l835_83547


namespace NUMINAMATH_CALUDE_safari_park_acrobats_l835_83587

theorem safari_park_acrobats :
  ∀ (acrobats giraffes : ℕ),
    2 * acrobats + 4 * giraffes = 32 →
    acrobats + giraffes = 10 →
    acrobats = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_safari_park_acrobats_l835_83587


namespace NUMINAMATH_CALUDE_trig_identity_proof_l835_83508

theorem trig_identity_proof : 
  Real.sin (15 * π / 180) * Real.cos (45 * π / 180) + 
  Real.sin (75 * π / 180) * Real.sin (135 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l835_83508


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l835_83540

theorem quadratic_equation_condition (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, a ≠ 0 ∧ (m - 2) * x^2 + (2*m + 1) * x - m = a * x^2 + b * x + c) →
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l835_83540


namespace NUMINAMATH_CALUDE_male_average_tickets_l835_83563

/-- Proves that the average number of tickets sold by male members is 58,
    given the overall average, female average, and male-to-female ratio. -/
theorem male_average_tickets (total_members : ℕ) (male_members : ℕ) (female_members : ℕ) :
  male_members > 0 →
  female_members = 2 * male_members →
  (male_members * q + female_members * 70) / total_members = 66 →
  total_members = male_members + female_members →
  q = 58 :=
by sorry

end NUMINAMATH_CALUDE_male_average_tickets_l835_83563


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l835_83502

-- Problem 1
theorem problem_1 : |(-12)| - (-6) + 5 - 10 = 13 := by sorry

-- Problem 2
theorem problem_2 : 64.83 - 5 * (18/19) + 35.17 - 44 * (1/19) = 50 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l835_83502


namespace NUMINAMATH_CALUDE_initial_workers_correct_l835_83533

/-- The number of workers initially working on the job -/
def initial_workers : ℕ := 6

/-- The number of days to finish the job initially -/
def initial_days : ℕ := 8

/-- The number of days worked before new workers join -/
def days_before_join : ℕ := 3

/-- The number of new workers that join -/
def new_workers : ℕ := 4

/-- The number of additional days needed to finish the job after new workers join -/
def additional_days : ℕ := 3

/-- Theorem stating that the initial number of workers is correct -/
theorem initial_workers_correct : 
  initial_workers * initial_days = 
  initial_workers * days_before_join + 
  (initial_workers + new_workers) * additional_days := by
  sorry

#check initial_workers_correct

end NUMINAMATH_CALUDE_initial_workers_correct_l835_83533


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l835_83567

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (-1, m)
  are_parallel a b → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l835_83567


namespace NUMINAMATH_CALUDE_positive_A_value_l835_83555

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value :
  ∃ A : ℝ, A > 0 ∧ hash A 3 = 145 ∧ A = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l835_83555


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l835_83542

theorem consecutive_odd_integers_sum (x y : ℤ) : 
  x = 63 → 
  y = x + 2 → 
  Odd x → 
  Odd y → 
  x + y = 128 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l835_83542
