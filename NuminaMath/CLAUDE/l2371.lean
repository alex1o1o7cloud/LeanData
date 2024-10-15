import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2371_237121

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - 3*a*x + 9 < 0) ↔ -2 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2371_237121


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l2371_237163

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 30)
  (area2 : w * h = 18)
  (area3 : l * h = 15) :
  l * w * h = 90 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l2371_237163


namespace NUMINAMATH_CALUDE_inequality_proof_l2371_237199

theorem inequality_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (a + 1/b)^2 + (b + 1/b)^2 ≥ 25/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2371_237199


namespace NUMINAMATH_CALUDE_average_permutation_sum_l2371_237104

def permutation_sum (b : Fin 8 → Fin 8) : ℕ :=
  |b 0 - b 1| + |b 2 - b 3| + |b 4 - b 5| + |b 6 - b 7|

def all_permutations : Finset (Fin 8 → Fin 8) :=
  Finset.univ.filter (fun f => Function.Injective f)

theorem average_permutation_sum :
  (Finset.sum all_permutations permutation_sum) / all_permutations.card = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_permutation_sum_l2371_237104


namespace NUMINAMATH_CALUDE_twelfth_term_of_specific_sequence_l2371_237185

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence (α : Type*) [Field α] where
  first_term : α
  common_difference : α

/-- The nth term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence ℚ) (n : ℕ) : ℚ :=
  seq.first_term + (n - 1 : ℚ) * seq.common_difference

theorem twelfth_term_of_specific_sequence :
  let seq := ArithmeticSequence.mk (1/2 : ℚ) ((5/6 - 1/2) : ℚ)
  nth_term seq 2 = 5/6 → nth_term seq 3 = 7/6 → nth_term seq 12 = 25/6 := by
  sorry


end NUMINAMATH_CALUDE_twelfth_term_of_specific_sequence_l2371_237185


namespace NUMINAMATH_CALUDE_power_mean_inequality_l2371_237153

theorem power_mean_inequality (a b : ℝ) (n : ℕ) 
  (ha : a > 0) (hb : b > 0) (hn : n > 0) : 
  (a^n + b^n) / 2 ≥ ((a + b) / 2)^n := by
  sorry

end NUMINAMATH_CALUDE_power_mean_inequality_l2371_237153


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2371_237188

theorem roots_of_polynomial (x : ℝ) :
  (x^2 - 5*x + 6)*(x - 3)*(2*x - 8) = 0 ↔ x = 2 ∨ x = 3 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2371_237188


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2371_237150

theorem quadratic_roots_relation (c d : ℚ) : 
  (∃ r s : ℚ, r + s = 3/5 ∧ r * s = -8/5) →
  (∃ p q : ℚ, p + q = -c ∧ p * q = d ∧ p = r - 3 ∧ q = s - 3) →
  d = 28/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2371_237150


namespace NUMINAMATH_CALUDE_count_valid_numbers_l2371_237195

/-- A function that generates all valid four-digit even numbers greater than 2000
    using digits 0, 1, 2, 3, 4, 5 without repetition -/
def validNumbers : Finset Nat := sorry

/-- The cardinality of the set of valid numbers -/
theorem count_valid_numbers : Finset.card validNumbers = 120 := by sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l2371_237195


namespace NUMINAMATH_CALUDE_sunflower_cost_l2371_237137

theorem sunflower_cost
  (num_roses : ℕ)
  (num_sunflowers : ℕ)
  (cost_per_rose : ℚ)
  (total_cost : ℚ)
  (h1 : num_roses = 24)
  (h2 : num_sunflowers = 3)
  (h3 : cost_per_rose = 3/2)
  (h4 : total_cost = 45) :
  (total_cost - num_roses * cost_per_rose) / num_sunflowers = 3 := by
sorry

end NUMINAMATH_CALUDE_sunflower_cost_l2371_237137


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_zero_l2371_237182

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x + 1

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_line_at_point_one_zero :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = x - 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_zero_l2371_237182


namespace NUMINAMATH_CALUDE_gcd_168_486_l2371_237174

theorem gcd_168_486 : Nat.gcd 168 486 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_168_486_l2371_237174


namespace NUMINAMATH_CALUDE_cosine_period_proof_l2371_237127

/-- Given a cosine function y = a cos(bx + c) + d where a, b, c, and d are positive constants,
    and the graph covers three periods from 0 to 3π, prove that b = 2. -/
theorem cosine_period_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_period : (3 : ℝ) * (2 * π / b) = 3 * π) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_period_proof_l2371_237127


namespace NUMINAMATH_CALUDE_shifted_quadratic_roots_l2371_237148

theorem shifted_quadratic_roots
  (b c : ℝ)
  (h1 : ∃ x1 x2 : ℝ, x1 = 2 ∧ x2 = -3 ∧ ∀ x, x^2 + b*x + c = 0 ↔ x = x1 ∨ x = x2) :
  ∃ y1 y2 : ℝ, y1 = 6 ∧ y2 = 1 ∧ ∀ x, (x-4)^2 + b*(x-4) + c = 0 ↔ x = y1 ∨ x = y2 :=
sorry

end NUMINAMATH_CALUDE_shifted_quadratic_roots_l2371_237148


namespace NUMINAMATH_CALUDE_puppies_adopted_theorem_l2371_237169

/-- The number of puppies adopted each day from a shelter -/
def puppies_adopted_per_day (initial_puppies additional_puppies adoption_days : ℕ) : ℕ :=
  (initial_puppies + additional_puppies) / adoption_days

/-- Theorem stating the number of puppies adopted each day -/
theorem puppies_adopted_theorem (initial_puppies additional_puppies adoption_days : ℕ) 
  (h1 : initial_puppies = 5)
  (h2 : additional_puppies = 35)
  (h3 : adoption_days = 5) :
  puppies_adopted_per_day initial_puppies additional_puppies adoption_days = 8 := by
  sorry

end NUMINAMATH_CALUDE_puppies_adopted_theorem_l2371_237169


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2371_237133

/-- A geometric sequence with its sum sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  geom : ∀ n : ℕ, a (n + 1) = a n * (a 1 / a 0)  -- Geometric property
  sum : ∀ n : ℕ, S n = (a 0 * (1 - (a 1 / a 0)^n)) / (1 - (a 1 / a 0))  -- Sum formula

/-- Theorem: If S_4 / S_2 = 3 for a geometric sequence, then 2a_2 - a_4 = 0 -/
theorem geometric_sequence_property (seq : GeometricSequence) 
  (h : seq.S 4 / seq.S 2 = 3) : 2 * seq.a 2 - seq.a 4 = 0 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l2371_237133


namespace NUMINAMATH_CALUDE_triangle_vector_sum_l2371_237100

/-- Given a triangle ABC with points E and F, prove that x + y = -1/6 -/
theorem triangle_vector_sum (A B C E F : ℝ × ℝ) (x y : ℝ) : 
  (E - A : ℝ × ℝ) = (1/2 : ℝ) • (B - A) →
  (C - F : ℝ × ℝ) = 2 • (F - A) →
  (F - E : ℝ × ℝ) = x • (B - A) + y • (C - A) →
  x + y = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_vector_sum_l2371_237100


namespace NUMINAMATH_CALUDE_price_adjustment_l2371_237165

theorem price_adjustment (original_price : ℝ) (original_price_pos : 0 < original_price) :
  let increased_price := original_price * (1 + 30 / 100)
  let decrease_factor := 3 / 13
  increased_price * (1 - decrease_factor) = original_price :=
by sorry

end NUMINAMATH_CALUDE_price_adjustment_l2371_237165


namespace NUMINAMATH_CALUDE_gcf_4320_2550_l2371_237128

theorem gcf_4320_2550 : Nat.gcd 4320 2550 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_4320_2550_l2371_237128


namespace NUMINAMATH_CALUDE_price_per_shirt_is_35_l2371_237192

/-- Calculates the price per shirt given the following parameters:
    * num_employees: number of employees
    * shirts_per_employee: number of shirts made per employee per day
    * hours_per_shift: number of hours in a shift
    * hourly_wage: hourly wage per employee
    * per_shirt_wage: additional wage per shirt made
    * nonemployee_expenses: daily nonemployee expenses
    * daily_profit: target daily profit
-/
def price_per_shirt (
  num_employees : ℕ
) (shirts_per_employee : ℕ
) (hours_per_shift : ℕ
) (hourly_wage : ℚ
) (per_shirt_wage : ℚ
) (nonemployee_expenses : ℚ
) (daily_profit : ℚ
) : ℚ :=
  let total_shirts := num_employees * shirts_per_employee
  let total_wages := num_employees * hours_per_shift * hourly_wage + total_shirts * per_shirt_wage
  let total_expenses := total_wages + nonemployee_expenses
  let total_revenue := daily_profit + total_expenses
  total_revenue / total_shirts

theorem price_per_shirt_is_35 :
  price_per_shirt 20 20 8 12 5 1000 9080 = 35 := by
  sorry

#eval price_per_shirt 20 20 8 12 5 1000 9080

end NUMINAMATH_CALUDE_price_per_shirt_is_35_l2371_237192


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l2371_237194

theorem solution_set_abs_inequality (x : ℝ) :
  (|1 - 2*x| < 3) ↔ (-1 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l2371_237194


namespace NUMINAMATH_CALUDE_magazine_cost_lynne_magazine_cost_l2371_237151

/-- The cost of each magazine given Lynne's purchase details -/
theorem magazine_cost (cat_books : ℕ) (solar_books : ℕ) (magazines : ℕ) 
  (book_price : ℕ) (total_spent : ℕ) : ℕ :=
  let total_books := cat_books + solar_books
  let book_cost := total_books * book_price
  let magazine_total_cost := total_spent - book_cost
  magazine_total_cost / magazines

/-- Proof that each magazine costs $4 given Lynne's purchase details -/
theorem lynne_magazine_cost : 
  magazine_cost 7 2 3 7 75 = 4 := by
  sorry

end NUMINAMATH_CALUDE_magazine_cost_lynne_magazine_cost_l2371_237151


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2371_237143

theorem trigonometric_identities :
  (Real.tan (25 * π / 180) + Real.tan (20 * π / 180) + Real.tan (25 * π / 180) * Real.tan (20 * π / 180) = 1) ∧
  (1 / Real.sin (10 * π / 180) - Real.sqrt 3 / Real.cos (10 * π / 180) = 4) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2371_237143


namespace NUMINAMATH_CALUDE_max_value_abc_l2371_237105

theorem max_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b * c * (a + b + c)) / ((a + b)^2 * (b + c)^3) ≤ 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_max_value_abc_l2371_237105


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l2371_237186

theorem largest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ (n + 10) ∣ (n^3 + 100) ∧ ∀ (m : ℕ), m > n → ¬((m + 10) ∣ (m^3 + 100)) :=
by
  use 890
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l2371_237186


namespace NUMINAMATH_CALUDE_difference_of_squares_301_297_l2371_237102

theorem difference_of_squares_301_297 : 301^2 - 297^2 = 2392 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_301_297_l2371_237102


namespace NUMINAMATH_CALUDE_pair_conditions_l2371_237159

def satisfies_conditions (a b : ℚ) : Prop :=
  a * b = 24 ∧ a + b > 0

theorem pair_conditions :
  ¬(satisfies_conditions (-6) (-4)) ∧
  (satisfies_conditions 3 8) ∧
  ¬(satisfies_conditions (-3/2) (-16)) ∧
  (satisfies_conditions 2 12) ∧
  (satisfies_conditions (4/3) 18) :=
by sorry

end NUMINAMATH_CALUDE_pair_conditions_l2371_237159


namespace NUMINAMATH_CALUDE_find_k_l2371_237107

theorem find_k (k : ℚ) (h : 64 / k = 4) : k = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l2371_237107


namespace NUMINAMATH_CALUDE_sum_of_max_min_a_is_zero_l2371_237116

-- Define the quadratic function
def f (a x : ℝ) : ℝ := x^2 - a*x - 20*a^2

-- Define the condition that the difference between any two solutions does not exceed 9
def solution_difference_condition (a : ℝ) : Prop :=
  ∀ x y : ℝ, f a x < 0 → f a y < 0 → |x - y| ≤ 9

-- Define the set of valid 'a' values
def valid_a_set : Set ℝ :=
  {a : ℝ | solution_difference_condition a}

-- State the theorem
theorem sum_of_max_min_a_is_zero :
  ∃ (a_min a_max : ℝ), 
    a_min ∈ valid_a_set ∧ 
    a_max ∈ valid_a_set ∧ 
    (∀ a ∈ valid_a_set, a_min ≤ a ∧ a ≤ a_max) ∧
    a_min + a_max = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_a_is_zero_l2371_237116


namespace NUMINAMATH_CALUDE_unique_solution_g_equals_g_inv_l2371_237191

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x - 5

-- Define the inverse function of g
noncomputable def g_inv (x : ℝ) : ℝ := (x + 5) / 4

-- Theorem statement
theorem unique_solution_g_equals_g_inv :
  ∃! x : ℝ, g x = g_inv x :=
sorry

end NUMINAMATH_CALUDE_unique_solution_g_equals_g_inv_l2371_237191


namespace NUMINAMATH_CALUDE_problem_solution_l2371_237139

-- Custom operation
def star (x y : ℕ) : ℕ := x * y + 1

-- Prime number function
def nth_prime (n : ℕ) : ℕ := sorry

-- Product function
def product_to_n (n : ℕ) : ℚ := sorry

-- Area of inscribed square
def inscribed_square_area (r : ℝ) : ℝ := sorry

theorem problem_solution :
  (star (star 2 4) 2 = 19) ∧
  (nth_prime 8 = 19) ∧
  (product_to_n 50 = 1 / 50) ∧
  (inscribed_square_area 10 = 200) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2371_237139


namespace NUMINAMATH_CALUDE_fraction_addition_l2371_237178

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2371_237178


namespace NUMINAMATH_CALUDE_path_area_is_775_l2371_237196

/-- Represents the dimensions and cost of a rectangular field with a surrounding path. -/
structure FieldWithPath where
  fieldLength : ℝ
  fieldWidth : ℝ
  pathWidth : ℝ
  pathCostPerSqm : ℝ
  totalPathCost : ℝ

/-- Calculates the area of the path surrounding a rectangular field. -/
def pathArea (f : FieldWithPath) : ℝ :=
  let totalLength := f.fieldLength + 2 * f.pathWidth
  let totalWidth := f.fieldWidth + 2 * f.pathWidth
  totalLength * totalWidth - f.fieldLength * f.fieldWidth

/-- Theorem stating that the area of the path is 775 sq m for the given field dimensions. -/
theorem path_area_is_775 (f : FieldWithPath)
  (h1 : f.fieldLength = 95)
  (h2 : f.fieldWidth = 55)
  (h3 : f.pathWidth = 2.5)
  (h4 : f.pathCostPerSqm = 2)
  (h5 : f.totalPathCost = 1550) :
  pathArea f = 775 := by
  sorry

end NUMINAMATH_CALUDE_path_area_is_775_l2371_237196


namespace NUMINAMATH_CALUDE_max_value_and_minimum_l2371_237168

noncomputable def f (x a b c : ℝ) : ℝ := |x + a| - |x - b| + c

theorem max_value_and_minimum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hmax : ∀ x, f x a b c ≤ 10) 
  (hmax_exists : ∃ x, f x a b c = 10) : 
  (a + b + c = 10) ∧ 
  (∀ a' b' c', a' > 0 → b' > 0 → c' > 0 → a' + b' + c' = 10 → 
    1/4 * (a - 1)^2 + (b - 2)^2 + (c - 3)^2 ≤ 1/4 * (a' - 1)^2 + (b' - 2)^2 + (c' - 3)^2) ∧
  (1/4 * (a - 1)^2 + (b - 2)^2 + (c - 3)^2 = 8/3) ∧
  (a = 11/3 ∧ b = 8/3 ∧ c = 11/3) :=
by sorry

end NUMINAMATH_CALUDE_max_value_and_minimum_l2371_237168


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l2371_237156

/-- The distance between the foci of the ellipse 9x^2 + 36y^2 = 1296 is 12√3 -/
theorem ellipse_foci_distance (x y : ℝ) :
  (9 * x^2 + 36 * y^2 = 1296) →
  (∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = (12 * Real.sqrt 3)^2 ∧
    ∀ (p : ℝ × ℝ), 9 * p.1^2 + 36 * p.2^2 = 1296 →
      Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) +
      Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) =
      2 * Real.sqrt (144)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l2371_237156


namespace NUMINAMATH_CALUDE_circle_center_l2371_237108

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 2*y - 15 = 0

/-- The center of a circle given by its coordinates -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Theorem stating that the center of the circle with the given equation is (3, -1) -/
theorem circle_center : 
  ∃ (center : CircleCenter), center.x = 3 ∧ center.y = -1 ∧
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - center.x)^2 + (y - center.y)^2 = 25 :=
sorry

end NUMINAMATH_CALUDE_circle_center_l2371_237108


namespace NUMINAMATH_CALUDE_roots_equation_t_value_l2371_237167

theorem roots_equation_t_value (n s : ℝ) (u v : ℝ) : 
  u^2 - n*u + 6 = 0 →
  v^2 - n*v + 6 = 0 →
  (u + 2/v)^2 - s*(u + 2/v) + t = 0 →
  (v + 2/u)^2 - s*(v + 2/u) + t = 0 →
  t = 32/3 := by
sorry

end NUMINAMATH_CALUDE_roots_equation_t_value_l2371_237167


namespace NUMINAMATH_CALUDE_standard_equation_min_area_OPQ_l2371_237118

-- Define the ellipse C
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the right focus F
def right_focus (a b : ℝ) : Prop := a^2 - b^2 = 1

-- Define the perpendicular condition
def perpendicular_condition (a b : ℝ) : Prop := b = 1

-- Theorem for the standard equation of the ellipse
theorem standard_equation (a b : ℝ) 
  (h1 : ellipse x y a b) 
  (h2 : right_focus a b) 
  (h3 : perpendicular_condition a b) : 
  x^2 / 2 + y^2 = 1 := by sorry

-- Define the triangle OPQ
def triangle_OPQ (x y m : ℝ) : Prop := 
  x^2 / 2 + y^2 = 1 ∧ 
  ∃ (P : ℝ × ℝ), P.2 = 2 ∧ 
  (P.1 * y = 2 * x ∨ (P.1 = 0 ∧ x = Real.sqrt 2))

-- Theorem for the minimum area of triangle OPQ
theorem min_area_OPQ (x y m : ℝ) 
  (h : triangle_OPQ x y m) : 
  ∃ (S : ℝ), S ≥ 1 ∧ 
  (∀ (S' : ℝ), triangle_OPQ x y m → S' ≥ S) := by sorry

end NUMINAMATH_CALUDE_standard_equation_min_area_OPQ_l2371_237118


namespace NUMINAMATH_CALUDE_vacation_duration_l2371_237147

theorem vacation_duration (plane_cost hotel_cost_per_day total_cost : ℕ) 
  (h1 : plane_cost = 48)
  (h2 : hotel_cost_per_day = 24)
  (h3 : total_cost = 120) :
  ∃ d : ℕ, d = 3 ∧ plane_cost + hotel_cost_per_day * d = total_cost := by
  sorry

end NUMINAMATH_CALUDE_vacation_duration_l2371_237147


namespace NUMINAMATH_CALUDE_grid_filling_exists_l2371_237131

/-- A function representing the grid filling -/
def GridFilling (n : ℕ) := Fin n → Fin n → Fin (2*n - 1)

/-- Predicate to check if a number is a power of 2 -/
def IsPowerOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

/-- Predicate to check if the grid filling is valid -/
def IsValidFilling (n : ℕ) (f : GridFilling n) : Prop :=
  (∀ k : Fin n, ∀ i j : Fin n, i ≠ j → f k i ≠ f k j) ∧
  (∀ k : Fin n, ∀ i j : Fin n, i ≠ j → f i k ≠ f j k)

theorem grid_filling_exists (n : ℕ) (h : IsPowerOfTwo n) :
  ∃ f : GridFilling n, IsValidFilling n f :=
sorry

end NUMINAMATH_CALUDE_grid_filling_exists_l2371_237131


namespace NUMINAMATH_CALUDE_remainder_theorem_l2371_237193

theorem remainder_theorem : ∃ q : ℕ, 
  2^206 + 206 = q * (2^103 + 2^53 + 1) + 205 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2371_237193


namespace NUMINAMATH_CALUDE_books_minus_figures_equals_two_l2371_237171

/-- The number of books on Jerry's shelf -/
def initial_books : ℕ := 7

/-- The initial number of action figures on Jerry's shelf -/
def initial_action_figures : ℕ := 3

/-- The number of action figures Jerry added later -/
def added_action_figures : ℕ := 2

/-- The total number of action figures after addition -/
def total_action_figures : ℕ := initial_action_figures + added_action_figures

theorem books_minus_figures_equals_two :
  initial_books - total_action_figures = 2 := by
  sorry

end NUMINAMATH_CALUDE_books_minus_figures_equals_two_l2371_237171


namespace NUMINAMATH_CALUDE_inequality_solution_l2371_237180

theorem inequality_solution (x : ℝ) (h : x ≠ 5) :
  (x * (x^2 + x + 1)) / ((x - 5)^2) ≥ 15 ↔ x ∈ Set.Iio 5 ∪ Set.Ioi 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2371_237180


namespace NUMINAMATH_CALUDE_scale_division_l2371_237197

/-- Given a scale of length 198 inches divided into 8 equal parts, 
    prove that the length of each part is 24.75 inches. -/
theorem scale_division (total_length : ℝ) (num_parts : ℕ) 
  (h1 : total_length = 198) 
  (h2 : num_parts = 8) :
  total_length / num_parts = 24.75 := by
  sorry

end NUMINAMATH_CALUDE_scale_division_l2371_237197


namespace NUMINAMATH_CALUDE_euro_equation_solution_l2371_237136

-- Define the € operation
def euro (x y : ℝ) : ℝ := 2 * x * y

-- State the theorem
theorem euro_equation_solution :
  ∀ y : ℝ, euro y (euro 7 5) = 560 → y = 4 := by
  sorry

end NUMINAMATH_CALUDE_euro_equation_solution_l2371_237136


namespace NUMINAMATH_CALUDE_cos_alpha_minus_pi_fourth_l2371_237145

theorem cos_alpha_minus_pi_fourth (α : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : Real.tan (α + π/4) = -3) : 
  Real.cos (α - π/4) = 3 * Real.sqrt 10 / 10 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_pi_fourth_l2371_237145


namespace NUMINAMATH_CALUDE_rational_difference_l2371_237114

theorem rational_difference (x y : ℚ) (h : (1 + y) / (x - y) = x) : y = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_rational_difference_l2371_237114


namespace NUMINAMATH_CALUDE_photo_arrangement_probability_photo_arrangement_probability_is_one_twentieth_l2371_237173

/-- The probability that in a group of six students with distinct heights,
    arranged in two rows of three each, every student in the back row
    is taller than every student in the front row. -/
theorem photo_arrangement_probability : ℚ :=
  let n_students : ℕ := 6
  let n_per_row : ℕ := 3
  let total_arrangements : ℕ := n_students.factorial
  let favorable_arrangements : ℕ := (n_per_row.factorial) * (n_per_row.factorial)
  favorable_arrangements / total_arrangements

/-- Proof that the probability is 1/20 -/
theorem photo_arrangement_probability_is_one_twentieth :
  photo_arrangement_probability = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangement_probability_photo_arrangement_probability_is_one_twentieth_l2371_237173


namespace NUMINAMATH_CALUDE_angle_ratio_l2371_237113

-- Define the points
variable (A B C P Q M : Point)

-- Define the angles
def angle (X Y Z : Point) : ℝ := sorry

-- BP and BQ trisect ∠ABC
axiom trisect : angle A B P = angle B P Q ∧ angle B P Q = angle P B Q

-- BM bisects ∠ABP
axiom bisect : angle A B M = (1/2) * angle A B P

-- Theorem statement
theorem angle_ratio : 
  (angle M B Q) / (angle A B Q) = 3/4 := by sorry

end NUMINAMATH_CALUDE_angle_ratio_l2371_237113


namespace NUMINAMATH_CALUDE_collinear_dots_probability_l2371_237144

/-- The number of dots in each row or column of the grid -/
def grid_size : ℕ := 5

/-- The total number of dots in the grid -/
def total_dots : ℕ := grid_size * grid_size

/-- The number of dots to be selected -/
def selected_dots : ℕ := 4

/-- The number of sets of collinear dots -/
def collinear_sets : ℕ := 14

/-- The total number of ways to choose 4 dots out of 25 -/
def total_combinations : ℕ := Nat.choose total_dots selected_dots

/-- The probability of selecting 4 collinear dots -/
def collinear_probability : ℚ := collinear_sets / total_combinations

theorem collinear_dots_probability :
  collinear_probability = 7 / 6325 := by sorry

end NUMINAMATH_CALUDE_collinear_dots_probability_l2371_237144


namespace NUMINAMATH_CALUDE_game_theory_proof_l2371_237184

theorem game_theory_proof (x y : ℝ) : 
  (x + y + (24 - x - y) = 24) →
  (2*x - 24 = 2) →
  (4*y - 24 = 4) →
  (∀ (a b c : ℝ), (a + b + c = 24) → (a = 8 ∧ b = 8 ∧ c = 8)) →
  (x = 13 ∧ y = 7 ∧ 24 - x - y = 4) :=
by sorry

end NUMINAMATH_CALUDE_game_theory_proof_l2371_237184


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l2371_237138

theorem profit_percentage_calculation (selling_price cost_price : ℝ) : 
  selling_price = 600 → 
  cost_price = 480 → 
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l2371_237138


namespace NUMINAMATH_CALUDE_additional_cars_needed_l2371_237130

def current_cars : ℕ := 37
def cars_per_row : ℕ := 8

theorem additional_cars_needed : 
  ∃ (n : ℕ), 
    (n > 0) ∧ 
    (current_cars + n) % cars_per_row = 0 ∧
    ∀ (m : ℕ), m < n → (current_cars + m) % cars_per_row ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_additional_cars_needed_l2371_237130


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2371_237129

theorem hyperbola_equation (f1 f2 : ℝ × ℝ) (p : ℝ × ℝ) :
  f1 = (0, 5) →
  f2 = (0, -5) →
  p = (2, 3 * Real.sqrt 5 / 2) →
  ∃ (a b : ℝ),
    a^2 = 9 ∧
    b^2 = 16 ∧
    ∀ (x y : ℝ),
      (y^2 / a^2) - (x^2 / b^2) = 1 ↔
      (x - f1.1)^2 + (y - f1.2)^2 - ((x - f2.1)^2 + (y - f2.2)^2) = 4 * a^2 ∧
      (p.1 - f1.1)^2 + (p.2 - f1.2)^2 - ((p.1 - f2.1)^2 + (p.2 - f2.2)^2) = 4 * a^2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2371_237129


namespace NUMINAMATH_CALUDE_impossible_to_use_all_parts_l2371_237155

theorem impossible_to_use_all_parts (p q r : ℕ) : 
  ¬∃ (x y z : ℕ), 
    (2 * x + 2 * z = 2 * p + 2 * r + 2) ∧ 
    (2 * x + y = 2 * p + q + 1) ∧ 
    (y + z = q + r) :=
by sorry

end NUMINAMATH_CALUDE_impossible_to_use_all_parts_l2371_237155


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2371_237166

/-- A quadratic function that intersects the x-axis at (0,0) and (-2,0) and has a minimum value of -1 -/
def f (x : ℝ) : ℝ := x^2 + 2*x

theorem quadratic_function_properties :
  (f 0 = 0) ∧
  (f (-2) = 0) ∧
  (∃ x₀, ∀ x, f x ≥ f x₀) ∧
  (∃ x₀, f x₀ = -1) ∧
  (∀ x, f x = x^2 + 2*x) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2371_237166


namespace NUMINAMATH_CALUDE_triangle_formation_l2371_237154

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem triangle_formation :
  can_form_triangle 3 4 5 ∧
  ¬can_form_triangle 1 1 2 ∧
  ¬can_form_triangle 1 4 6 ∧
  ¬can_form_triangle 2 3 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_formation_l2371_237154


namespace NUMINAMATH_CALUDE_prob_three_odd_dice_l2371_237164

def num_dice : ℕ := 5
def num_odd : ℕ := 3

theorem prob_three_odd_dice :
  (num_dice.choose num_odd : ℚ) * (1 / 2) ^ num_dice = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_odd_dice_l2371_237164


namespace NUMINAMATH_CALUDE_inequality_proof_l2371_237111

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h_sum : a + b + c + d = 1) : 
  b * c * d / (1 - a)^2 + c * d * a / (1 - b)^2 + 
  d * a * b / (1 - c)^2 + a * b * c / (1 - d)^2 ≤ 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2371_237111


namespace NUMINAMATH_CALUDE_problem_solution_l2371_237112

theorem problem_solution :
  (∀ a : ℝ, 2*a + 3*a - 4*a = a) ∧
  (-1^2022 + 27/4 * (-1/3 - 1) / (-3)^2 + |-1| = -1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2371_237112


namespace NUMINAMATH_CALUDE_total_balls_theorem_l2371_237170

/-- The number of balls of wool used for a single item -/
def balls_per_item : String → ℕ
  | "scarf" => 3
  | "sweater" => 4
  | "hat" => 2
  | "mittens" => 1
  | _ => 0

/-- The number of items made by Aaron -/
def aaron_items : String → ℕ
  | "scarf" => 10
  | "sweater" => 5
  | "hat" => 6
  | _ => 0

/-- The number of items made by Enid -/
def enid_items : String → ℕ
  | "sweater" => 8
  | "hat" => 12
  | "mittens" => 4
  | _ => 0

/-- The total number of balls of wool used by both Enid and Aaron -/
def total_balls_used : ℕ := 
  (aaron_items "scarf" * balls_per_item "scarf") +
  (aaron_items "sweater" * balls_per_item "sweater") +
  (aaron_items "hat" * balls_per_item "hat") +
  (enid_items "sweater" * balls_per_item "sweater") +
  (enid_items "hat" * balls_per_item "hat") +
  (enid_items "mittens" * balls_per_item "mittens")

theorem total_balls_theorem : total_balls_used = 122 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_theorem_l2371_237170


namespace NUMINAMATH_CALUDE_tracys_candies_l2371_237120

theorem tracys_candies (x : ℕ) : 
  (x % 3 = 0) →  -- x is divisible by 3
  (x % 2 = 0) →  -- x is divisible by 2
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 5) →  -- Tracy's brother took between 1 and 5 candies
  (x / 2 - 30 - k = 3) →  -- Tracy was left with 3 candies after all events
  x = 72 :=
by sorry

#check tracys_candies

end NUMINAMATH_CALUDE_tracys_candies_l2371_237120


namespace NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l2371_237117

/-- 
For a regular polygon where each exterior angle is 40°, 
the sum of the interior angles is 1260°.
-/
theorem sum_interior_angles_regular_polygon (n : ℕ) : 
  (360 / n = 40) → (n - 2) * 180 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l2371_237117


namespace NUMINAMATH_CALUDE_line_satisfies_conditions_l2371_237177

/-- A line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line --/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a point bisects a line segment --/
def bisectsSegment (p : Point) (l : Line) : Prop :=
  ∃ (p1 p2 : Point), pointOnLine p1 l ∧ pointOnLine p2 l ∧ 
    p.x = (p1.x + p2.x) / 2 ∧ p.y = (p1.y + p2.y) / 2

/-- Check if a line lies between two other lines --/
def linesBetween (l : Line) (l1 l2 : Line) : Prop :=
  ∀ (p : Point), pointOnLine p l → 
    (l1.a * p.x + l1.b * p.y + l1.c) * (l2.a * p.x + l2.b * p.y + l2.c) ≤ 0

theorem line_satisfies_conditions : 
  let P : Point := ⟨3, 0⟩
  let L : Line := ⟨8, -1, -24⟩
  let L1 : Line := ⟨2, -1, -2⟩
  let L2 : Line := ⟨1, 1, 3⟩
  pointOnLine P L ∧ 
  bisectsSegment P L ∧
  linesBetween L L1 L2 :=
by sorry

end NUMINAMATH_CALUDE_line_satisfies_conditions_l2371_237177


namespace NUMINAMATH_CALUDE_structure_surface_area_270_l2371_237134

def surface_area_cube (side_length : ℝ) : ℝ := 6 * side_length^2

def structure_surface_area (large_side : ℝ) (medium_side : ℝ) (small_side : ℝ) : ℝ :=
  surface_area_cube large_side +
  4 * surface_area_cube medium_side +
  4 * surface_area_cube small_side

theorem structure_surface_area_270 :
  structure_surface_area 5 2 1 = 270 := by
  sorry

end NUMINAMATH_CALUDE_structure_surface_area_270_l2371_237134


namespace NUMINAMATH_CALUDE_prob_valid_sequence_equals_377_4096_sum_numerator_denominator_l2371_237110

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Number of valid sequences of length n -/
def valid_sequences (n : ℕ) : ℕ := fib (n + 2)

/-- Total number of possible sequences of length n -/
def total_sequences (n : ℕ) : ℕ := 2^n

/-- The probability of a valid sequence of length 12 -/
def prob_valid_sequence : ℚ := (valid_sequences 12 : ℚ) / (total_sequences 12 : ℚ)

theorem prob_valid_sequence_equals_377_4096 :
  prob_valid_sequence = 377 / 4096 :=
sorry

theorem sum_numerator_denominator :
  377 + 4096 = 4473 :=
sorry

end NUMINAMATH_CALUDE_prob_valid_sequence_equals_377_4096_sum_numerator_denominator_l2371_237110


namespace NUMINAMATH_CALUDE_polygon_25_sides_diagonals_l2371_237181

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 25 sides has 275 diagonals -/
theorem polygon_25_sides_diagonals : num_diagonals 25 = 275 := by
  sorry

end NUMINAMATH_CALUDE_polygon_25_sides_diagonals_l2371_237181


namespace NUMINAMATH_CALUDE_probability_two_non_defective_pens_l2371_237187

/-- The probability of selecting two non-defective pens from a box with defective pens -/
theorem probability_two_non_defective_pens 
  (total_pens : ℕ) 
  (defective_pens : ℕ) 
  (selected_pens : ℕ) 
  (h1 : total_pens = 10) 
  (h2 : defective_pens = 2) 
  (h3 : selected_pens = 2) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 28 / 45 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_non_defective_pens_l2371_237187


namespace NUMINAMATH_CALUDE_not_well_placed_2_pow_2011_l2371_237135

/-- Represents the first number in a row of the triangular table -/
def first_in_row (row : ℕ) : ℕ := (row - 1)^2 + 1

/-- Represents the first number in a column of the triangular table -/
def first_in_column (col : ℕ) : ℕ := (col - 1)^2 + 1

/-- A number is well-placed if it equals the sum of the first number in its row and the first number in its column -/
def is_well_placed (n : ℕ) : Prop :=
  ∃ (row col : ℕ), n = first_in_row row + first_in_column col

theorem not_well_placed_2_pow_2011 : ¬ is_well_placed (2^2011) := by
  sorry

end NUMINAMATH_CALUDE_not_well_placed_2_pow_2011_l2371_237135


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_theorem_l2371_237106

/-- Represents an isosceles trapezoid with inscribed and circumscribed circles. -/
structure IsoscelesTrapezoid where
  r : ℝ  -- radius of inscribed circle
  R : ℝ  -- radius of circumscribed circle
  k : ℝ  -- ratio of R to r
  h_k_def : k = R / r
  h_k_pos : k > 0

/-- The angles and permissible k values for an isosceles trapezoid. -/
def trapezoid_properties (t : IsoscelesTrapezoid) : Prop :=
  let angle := Real.arcsin (1 / t.k * Real.sqrt ((1 + Real.sqrt (1 + 4 * t.k ^ 2)) / 2))
  (∀ θ, θ = angle ∨ θ = Real.pi - angle → 
    θ.cos * t.r = t.r ∧ θ.sin * t.R = t.R / 2) ∧ 
  t.k > Real.sqrt 2

/-- Main theorem about isosceles trapezoid properties. -/
theorem isosceles_trapezoid_theorem (t : IsoscelesTrapezoid) : 
  trapezoid_properties t := by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_theorem_l2371_237106


namespace NUMINAMATH_CALUDE_original_number_is_seventeen_l2371_237198

theorem original_number_is_seventeen : 
  ∀ x : ℕ, 
  (∀ y : ℕ, y < 6 → ¬(23 ∣ (x + y))) → 
  (23 ∣ (x + 6)) → 
  x = 17 := by
sorry

end NUMINAMATH_CALUDE_original_number_is_seventeen_l2371_237198


namespace NUMINAMATH_CALUDE_functional_equation_implies_constant_l2371_237125

/-- A function from ℤ² to [0,1] satisfying the given functional equation -/
def FunctionalEquation (f : ℤ × ℤ → ℝ) : Prop :=
  ∀ x y : ℤ, 0 ≤ f (x, y) ∧ f (x, y) ≤ 1 ∧ 
  f (x, y) = (f (x - 1, y) + f (x, y - 1)) / 2

/-- Theorem stating that any function satisfying the functional equation must be constant -/
theorem functional_equation_implies_constant 
  (f : ℤ × ℤ → ℝ) 
  (h : FunctionalEquation f) : 
  ∃ c : ℝ, c ∈ Set.Icc 0 1 ∧ ∀ x y : ℤ, f (x, y) = c :=
sorry

end NUMINAMATH_CALUDE_functional_equation_implies_constant_l2371_237125


namespace NUMINAMATH_CALUDE_linear_equation_solution_l2371_237132

theorem linear_equation_solution :
  ∃! x : ℝ, 8 * x = 2 * x - 6 :=
by
  use -1
  constructor
  · -- Prove that -1 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check linear_equation_solution

end NUMINAMATH_CALUDE_linear_equation_solution_l2371_237132


namespace NUMINAMATH_CALUDE_tan_3x_increasing_interval_l2371_237158

theorem tan_3x_increasing_interval (m : ℝ) : 
  (∀ x₁ x₂, m < x₁ ∧ x₁ < x₂ ∧ x₂ < π/6 → Real.tan (3*x₁) < Real.tan (3*x₂)) → 
  m ∈ Set.Icc (-π/6) (π/6) := by
sorry

end NUMINAMATH_CALUDE_tan_3x_increasing_interval_l2371_237158


namespace NUMINAMATH_CALUDE_factor_expression_l2371_237190

theorem factor_expression (b : ℝ) : 221 * b^2 + 17 * b = 17 * b * (13 * b + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2371_237190


namespace NUMINAMATH_CALUDE_trapezoid_area_l2371_237126

theorem trapezoid_area (large_triangle_area small_triangle_area : ℝ)
  (num_trapezoids : ℕ) (h1 : large_triangle_area = 36)
  (h2 : small_triangle_area = 4) (h3 : num_trapezoids = 4) :
  (large_triangle_area - small_triangle_area) / num_trapezoids = 8 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2371_237126


namespace NUMINAMATH_CALUDE_unique_intersection_points_l2371_237124

/-- The set of values k for which |z - 2| = 3|z + 2| intersects |z| = k in exactly one point -/
def intersection_points : Set ℝ :=
  {1.5, 4.5, 5.5}

/-- Predicate to check if a complex number satisfies |z - 2| = 3|z + 2| -/
def satisfies_equation (z : ℂ) : Prop :=
  Complex.abs (z - 2) = 3 * Complex.abs (z + 2)

/-- Predicate to check if a complex number has magnitude k -/
def has_magnitude (z : ℂ) (k : ℝ) : Prop :=
  Complex.abs z = k

/-- The theorem stating that the intersection_points set contains all values of k
    for which |z - 2| = 3|z + 2| intersects |z| = k in exactly one point -/
theorem unique_intersection_points :
  ∀ k : ℝ, (∃! z : ℂ, satisfies_equation z ∧ has_magnitude z k) ↔ k ∈ intersection_points :=
by sorry

end NUMINAMATH_CALUDE_unique_intersection_points_l2371_237124


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2371_237176

noncomputable def f (x : ℝ) : ℝ := x - Real.cos x

theorem tangent_line_equation :
  let p : ℝ × ℝ := (π / 2, π / 2)
  let m : ℝ := 1 + Real.sin (π / 2)
  let tangent_eq (x y : ℝ) : Prop := 2 * x - y - π / 2 = 0
  tangent_eq (p.1) (p.2) ∧
  ∀ x y : ℝ, tangent_eq x y ↔ y - p.2 = m * (x - p.1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2371_237176


namespace NUMINAMATH_CALUDE_retail_price_increase_l2371_237175

theorem retail_price_increase (manufacturing_cost : ℝ) (retailer_price : ℝ) (customer_price : ℝ)
  (h1 : customer_price = retailer_price * 1.3)
  (h2 : customer_price = manufacturing_cost * 1.82) :
  (retailer_price - manufacturing_cost) / manufacturing_cost = 0.4 := by
sorry

end NUMINAMATH_CALUDE_retail_price_increase_l2371_237175


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2371_237172

-- Define a positive geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ (n : ℕ), a (n + 1) = a n * r

-- State the theorem
theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_condition1 : a 1 * a 8 = 4 * a 5)
  (h_condition2 : (a 4 + 2 * a 6) / 2 = 18) :
  a 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2371_237172


namespace NUMINAMATH_CALUDE_parabola_equation_l2371_237157

theorem parabola_equation (p : ℝ) (h_p : p > 0) :
  (∃ x y : ℝ, y^2 = 2*p*x ∧ 
    (x + p/2)^2 + y^2 = 100 ∧ 
    y^2 = 36) → 
  p = 2 ∨ p = 18 := by
sorry

end NUMINAMATH_CALUDE_parabola_equation_l2371_237157


namespace NUMINAMATH_CALUDE_steve_initial_berries_l2371_237189

/-- Proves that Steve started with 21 berries given the conditions of the problem -/
theorem steve_initial_berries :
  ∀ (stacy_initial steve_initial : ℕ),
    stacy_initial = 32 →
    steve_initial + 4 = stacy_initial - 7 →
    steve_initial = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_steve_initial_berries_l2371_237189


namespace NUMINAMATH_CALUDE_find_m_value_l2371_237161

theorem find_m_value (α : Real) (m : Real) :
  let P : Real × Real := (-8 * m, -6 * Real.sin (30 * π / 180))
  (∃ (r : Real), r > 0 ∧ P.1 = r * Real.cos α ∧ P.2 = r * Real.sin α) →
  Real.cos α = -4/5 →
  m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_find_m_value_l2371_237161


namespace NUMINAMATH_CALUDE_card_count_proof_l2371_237109

/-- The number of cards Sasha added to the box -/
def cards_added : ℕ := 48

/-- The fraction of cards Karen removed from what Sasha added -/
def removal_fraction : ℚ := 1 / 6

/-- The number of cards in the box after Sasha's and Karen's actions -/
def final_card_count : ℕ := 83

/-- The original number of cards in the box -/
def original_card_count : ℕ := 75

theorem card_count_proof :
  (cards_added : ℚ) - removal_fraction * cards_added + original_card_count = final_card_count :=
sorry

end NUMINAMATH_CALUDE_card_count_proof_l2371_237109


namespace NUMINAMATH_CALUDE_halfway_distance_theorem_l2371_237149

def errand_distances : List ℕ := [10, 15, 5]

theorem halfway_distance_theorem (distances : List ℕ) (h : distances = errand_distances) :
  (distances.sum / 2 : ℕ) = 15 := by sorry

end NUMINAMATH_CALUDE_halfway_distance_theorem_l2371_237149


namespace NUMINAMATH_CALUDE_compute_expression_l2371_237162

theorem compute_expression : 3 * 3^4 + 9^60 / 9^59 - 27^3 = -19431 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2371_237162


namespace NUMINAMATH_CALUDE_two_digit_integers_count_l2371_237122

def digits : List ℕ := [2, 3, 4, 7]
def tens_digits : List ℕ := [2, 3]
def units_digits : List ℕ := [4, 7]

theorem two_digit_integers_count : 
  (List.length tens_digits) * (List.length units_digits) = 4 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_integers_count_l2371_237122


namespace NUMINAMATH_CALUDE_stream_speed_l2371_237141

/-- Proves that given a boat with a speed of 22 km/hr in still water,
    traveling 189 km downstream in 7 hours, the speed of the stream is 5 km/hr. -/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 22 →
  downstream_distance = 189 →
  downstream_time = 7 →
  ∃ stream_speed : ℝ,
    stream_speed = 5 ∧
    downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by
  sorry


end NUMINAMATH_CALUDE_stream_speed_l2371_237141


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2371_237142

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 3)) is equal to 9/4. -/
theorem infinite_series_sum : 
  (∑' n : ℕ, (3*n - 2) / (n * (n + 1) * (n + 3))) = 9/4 := by
  sorry


end NUMINAMATH_CALUDE_infinite_series_sum_l2371_237142


namespace NUMINAMATH_CALUDE_smallest_candy_count_l2371_237119

theorem smallest_candy_count : ∃ (n : ℕ), 
  (n ≥ 100 ∧ n ≤ 999) ∧ 
  (n + 7) % 9 = 0 ∧ 
  (n - 9) % 7 = 0 ∧
  n = 110 ∧
  ∀ (m : ℕ), (m ≥ 100 ∧ m ≤ 999) → 
    (m + 7) % 9 = 0 → (m - 9) % 7 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l2371_237119


namespace NUMINAMATH_CALUDE_table_length_is_77_l2371_237146

/-- Represents the dimensions and placement of sheets on a table. -/
structure TableSetup where
  tableWidth : ℕ
  tableLength : ℕ
  sheetWidth : ℕ
  sheetHeight : ℕ
  sheetCount : ℕ

/-- Checks if the given setup satisfies the conditions of the problem. -/
def isValidSetup (setup : TableSetup) : Prop :=
  setup.tableWidth = 80 ∧
  setup.sheetWidth = 8 ∧
  setup.sheetHeight = 5 ∧
  setup.sheetWidth + setup.sheetCount = setup.tableWidth ∧
  setup.sheetHeight + setup.sheetCount = setup.tableLength

/-- The main theorem stating that if the setup is valid, the table length must be 77. -/
theorem table_length_is_77 (setup : TableSetup) :
  isValidSetup setup → setup.tableLength = 77 := by
  sorry

#check table_length_is_77

end NUMINAMATH_CALUDE_table_length_is_77_l2371_237146


namespace NUMINAMATH_CALUDE_circle_equation_l2371_237179

theorem circle_equation (r : ℝ) (h1 : r = 6) :
  ∃ (a b : ℝ),
    (∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = r^2) ∧
    (b = r) ∧
    (∃ (x y : ℝ), x^2 + y^2 - 6*y + 8 = 0 ∧ (x - a)^2 + (y - b)^2 = (r - 1)^2) →
    ((a = 4 ∨ a = -4) ∧ b = 6) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2371_237179


namespace NUMINAMATH_CALUDE_min_value_greater_than_nine_l2371_237101

theorem min_value_greater_than_nine (a : ℝ) (h : a = 6) :
  ∀ x > a, x + 4 / (x - a) > 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_greater_than_nine_l2371_237101


namespace NUMINAMATH_CALUDE_train_lengths_l2371_237140

/-- Theorem: Train Lengths
Given:
- A bridge of length 800 meters
- Train A takes 45 seconds to cross the bridge
- Train B takes 40 seconds to cross the bridge
- Train A takes 15 seconds to pass a lamp post
- Train B takes 10 seconds to pass a lamp post

Prove that the length of Train A is 400 meters and the length of Train B is 800/3 meters.
-/
theorem train_lengths (bridge_length : ℝ) (time_A_bridge time_B_bridge time_A_post time_B_post : ℝ)
  (h1 : bridge_length = 800)
  (h2 : time_A_bridge = 45)
  (h3 : time_B_bridge = 40)
  (h4 : time_A_post = 15)
  (h5 : time_B_post = 10) :
  ∃ (length_A length_B : ℝ),
    length_A = 400 ∧ length_B = 800 / 3 ∧
    length_A + bridge_length = (length_A / time_A_post) * time_A_bridge ∧
    length_B + bridge_length = (length_B / time_B_post) * time_B_bridge :=
by
  sorry

end NUMINAMATH_CALUDE_train_lengths_l2371_237140


namespace NUMINAMATH_CALUDE_dresses_total_l2371_237123

/-- The total number of dresses for Emily, Melissa, and Debora -/
def total_dresses (emily_dresses melissa_dresses debora_dresses : ℕ) : ℕ :=
  emily_dresses + melissa_dresses + debora_dresses

/-- Theorem stating the total number of dresses given the conditions -/
theorem dresses_total (emily_dresses : ℕ) 
  (h1 : emily_dresses = 16)
  (h2 : ∃ (melissa_dresses : ℕ), melissa_dresses = emily_dresses / 2)
  (h3 : ∃ (debora_dresses : ℕ), debora_dresses = emily_dresses / 2 + 12) :
  ∃ (total : ℕ), total = total_dresses emily_dresses (emily_dresses / 2) (emily_dresses / 2 + 12) ∧ total = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_dresses_total_l2371_237123


namespace NUMINAMATH_CALUDE_line_l_passes_through_fixed_point_chord_length_y_axis_shortest_chord_equation_l2371_237152

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 16

-- Define the line l
def line_l (x y a : ℝ) : Prop := x - a * y + 3 * a - 2 = 0

-- Statement A
theorem line_l_passes_through_fixed_point (a : ℝ) :
  ∃ x y, line_l x y a ∧ x = 2 ∧ y = 3 :=
sorry

-- Statement B
theorem chord_length_y_axis :
  ∃ y₁ y₂, circle_C 0 y₁ ∧ circle_C 0 y₂ ∧ y₂ - y₁ = 2 * Real.sqrt 15 :=
sorry

-- Statement D
theorem shortest_chord_equation (a : ℝ) :
  (∀ x y, line_l x y a → circle_C x y → 
    ∀ x' y', line_l x' y' a → circle_C x' y' → 
      (x - x')^2 + (y - y')^2 ≤ (x - (-1))^2 + (y - 1)^2) →
  ∃ k, a = -3/2 ∧ k * (3 * x + 2 * y - 12) = x - a * y + 3 * a - 2 :=
sorry

end NUMINAMATH_CALUDE_line_l_passes_through_fixed_point_chord_length_y_axis_shortest_chord_equation_l2371_237152


namespace NUMINAMATH_CALUDE_second_green_probability_l2371_237103

-- Define the contents of each bag
def bag1 : Finset ℕ := {0, 0, 0, 1}  -- 0 represents green, 1 represents red
def bag2 : Finset ℕ := {0, 0, 1, 1}
def bag3 : Finset ℕ := {0, 1, 1, 1}

-- Define the probability of selecting each bag
def bagProb : ℕ → ℚ
  | 1 => 1/3
  | 2 => 1/3
  | 3 => 1/3
  | _ => 0

-- Define the probability of selecting a green candy from a bag
def greenProb : Finset ℕ → ℚ
  | s => (s.filter (· = 0)).card / s.card

-- Define the probability of selecting a red candy from a bag
def redProb : Finset ℕ → ℚ
  | s => (s.filter (· = 1)).card / s.card

-- Define the probability of selecting a green candy as the second candy
def secondGreenProb : ℚ := sorry

theorem second_green_probability : secondGreenProb = 73/144 := by sorry

end NUMINAMATH_CALUDE_second_green_probability_l2371_237103


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2371_237183

theorem arithmetic_calculations :
  (-9 + 5 * (-6) - 18 / (-3) = -33) ∧
  ((-3/4 - 5/8 + 9/12) * (-24) + (-8) / (2/3) = 3) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2371_237183


namespace NUMINAMATH_CALUDE_expression_factorization_l2371_237160

theorem expression_factorization (x : ℝ) : 
  (16 * x^7 + 36 * x^4 - 9) - (4 * x^7 - 6 * x^4 - 9) = 6 * x^4 * (2 * x^3 + 7) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2371_237160


namespace NUMINAMATH_CALUDE_valid_arrangement_exists_l2371_237115

/-- A chessboard is represented as a function from (Fin 8 × Fin 8) to Option (Fin 2),
    where Some 0 represents a white piece, Some 1 represents a black piece,
    and None represents an empty square. -/
def Chessboard := Fin 8 → Fin 8 → Option (Fin 2)

/-- Count the number of neighbors of a given color for a piece at position (i, j) -/
def countNeighbors (board : Chessboard) (i j : Fin 8) (color : Fin 2) : Nat :=
  sorry

/-- Check if a given arrangement satisfies the condition that each piece
    has an equal number of white and black neighbors -/
def isValidArrangement (board : Chessboard) : Prop :=
  sorry

/-- Count the total number of pieces of a given color on the board -/
def countPieces (board : Chessboard) (color : Fin 2) : Nat :=
  sorry

/-- The main theorem stating that a valid arrangement exists -/
theorem valid_arrangement_exists : ∃ (board : Chessboard),
  (countPieces board 0 = 16) ∧
  (countPieces board 1 = 16) ∧
  isValidArrangement board :=
sorry

end NUMINAMATH_CALUDE_valid_arrangement_exists_l2371_237115
