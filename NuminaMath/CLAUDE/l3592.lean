import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_coefficient_sum_l3592_359223

/-- A quadratic function passing through specific points with a given vertex -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_coefficient_sum (a b c : ℝ) :
  (QuadraticFunction a b c 1 = 4) →
  (QuadraticFunction a b c (-2) = -1) →
  (∀ x, QuadraticFunction a b c x ≥ QuadraticFunction a b c (-1)) →
  (QuadraticFunction a b c (-1) = -2) →
  a + b + c = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_sum_l3592_359223


namespace NUMINAMATH_CALUDE_at_least_one_equation_has_two_distinct_roots_l3592_359242

theorem at_least_one_equation_has_two_distinct_roots 
  (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∃ x y : ℝ, 
    (x ≠ y ∧ 
      ((a * x^2 + 2*b*x + c = 0 ∧ a * y^2 + 2*b*y + c = 0) ∨
       (b * x^2 + 2*c*x + a = 0 ∧ b * y^2 + 2*c*y + a = 0) ∨
       (c * x^2 + 2*a*x + c = 0 ∧ c * y^2 + 2*a*y + c = 0))) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_equation_has_two_distinct_roots_l3592_359242


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3592_359271

theorem sum_of_roots_quadratic (a b : ℝ) 
  (ha : a^2 - a - 6 = 0) 
  (hb : b^2 - b - 6 = 0) 
  (hab : a ≠ b) : 
  a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3592_359271


namespace NUMINAMATH_CALUDE_trail_mix_weight_l3592_359240

theorem trail_mix_weight : 
  let peanuts : ℚ := 0.16666666666666666
  let chocolate_chips : ℚ := 0.16666666666666666
  let raisins : ℚ := 0.08333333333333333
  let almonds : ℚ := 0.14583333333333331
  let cashews : ℚ := 1/8
  let dried_cranberries : ℚ := 3/32
  peanuts + chocolate_chips + raisins + almonds + cashews + dried_cranberries = 0.78125 := by
  sorry

end NUMINAMATH_CALUDE_trail_mix_weight_l3592_359240


namespace NUMINAMATH_CALUDE_work_completion_time_l3592_359266

theorem work_completion_time 
  (john_time : ℝ) 
  (rose_time : ℝ) 
  (h1 : john_time = 320) 
  (h2 : rose_time = 480) : 
  1 / (1 / john_time + 1 / rose_time) = 192 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3592_359266


namespace NUMINAMATH_CALUDE_exponent_product_simplification_l3592_359229

theorem exponent_product_simplification :
  (5 ^ 0.4) * (5 ^ 0.1) * (5 ^ 0.5) * (5 ^ 0.3) * (5 ^ 0.7) = 25 := by
  sorry

end NUMINAMATH_CALUDE_exponent_product_simplification_l3592_359229


namespace NUMINAMATH_CALUDE_exists_positive_solution_l3592_359272

/-- Definition of the star operation -/
def star (a b : ℝ) : ℝ := a * b^2 + 3 * b - a

/-- Theorem stating the existence of a positive solution -/
theorem exists_positive_solution :
  ∃ x : ℝ, x > 0 ∧ star 5 x = 100 := by
  sorry

end NUMINAMATH_CALUDE_exists_positive_solution_l3592_359272


namespace NUMINAMATH_CALUDE_area_inscribed_circle_l3592_359244

/-- The area of an inscribed circle in a triangle with given side lengths -/
theorem area_inscribed_circle (a b c : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) :
  let s := (a + b + c) / 2
  let area_triangle := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := area_triangle / s
  π * r^2 = (3136 / 81) * π := by sorry

end NUMINAMATH_CALUDE_area_inscribed_circle_l3592_359244


namespace NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l3592_359257

/-- The ratio of the area of an inscribed circle to the area of an equilateral triangle -/
theorem inscribed_circle_area_ratio (s r : ℝ) (h1 : s > 0) (h2 : r > 0) 
  (h3 : r = (Real.sqrt 3 / 6) * s) : 
  (π * r^2) / ((Real.sqrt 3 / 4) * s^2) = π / (3 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l3592_359257


namespace NUMINAMATH_CALUDE_least_tablets_extracted_l3592_359224

theorem least_tablets_extracted (tablets_a tablets_b : ℕ) 
  (ha : tablets_a = 10) (hb : tablets_b = 16) :
  ∃ (n : ℕ), n ≤ tablets_a + tablets_b ∧ 
  (∀ (k : ℕ), k < n → 
    (k < tablets_a + 2 → ∃ (x y : ℕ), x + y = k ∧ (x < 2 ∨ y < 2)) ∧
    (k ≥ tablets_a + 2 → ∃ (x y : ℕ), x + y = k ∧ x ≥ 2 ∧ y ≥ 2)) ∧
  n = 12 :=
sorry

end NUMINAMATH_CALUDE_least_tablets_extracted_l3592_359224


namespace NUMINAMATH_CALUDE_no_real_roots_geometric_sequence_l3592_359274

/-- If a, b, and c form a geometric sequence, then ax^2 + bx + c = 0 has no real solutions -/
theorem no_real_roots_geometric_sequence (a b c : ℝ) (h1 : a ≠ 0) (h2 : b^2 = a*c) (h3 : a*c > 0) :
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_geometric_sequence_l3592_359274


namespace NUMINAMATH_CALUDE_calculation_proof_l3592_359234

theorem calculation_proof : (3.242 * 15) / 100 = 0.4863 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3592_359234


namespace NUMINAMATH_CALUDE_order_of_trig_powers_l3592_359283

theorem order_of_trig_powers (α : Real) (h : π/4 < α ∧ α < π/2) :
  (Real.cos α) ^ (Real.sin α) < (Real.cos α) ^ (Real.cos α) ∧
  (Real.cos α) ^ (Real.cos α) < (Real.sin α) ^ (Real.cos α) := by
  sorry

end NUMINAMATH_CALUDE_order_of_trig_powers_l3592_359283


namespace NUMINAMATH_CALUDE_original_price_of_discounted_shoes_l3592_359201

/-- Given a pair of shoes sold at a 20% discount for $480, prove that its original price was $600. -/
theorem original_price_of_discounted_shoes (discount_rate : ℝ) (discounted_price : ℝ) : 
  discount_rate = 0.20 → discounted_price = 480 → (1 - discount_rate) * 600 = discounted_price := by
  sorry

end NUMINAMATH_CALUDE_original_price_of_discounted_shoes_l3592_359201


namespace NUMINAMATH_CALUDE_time_for_one_smoothie_l3592_359245

/-- The time it takes to make a certain number of smoothies -/
def time_to_make_smoothies (n : ℕ) : ℕ := 55

/-- The number of smoothies made in the given time -/
def number_of_smoothies : ℕ := 5

/-- Proves that the time to make one smoothie is 11 minutes -/
theorem time_for_one_smoothie :
  time_to_make_smoothies number_of_smoothies / number_of_smoothies = 11 :=
sorry

end NUMINAMATH_CALUDE_time_for_one_smoothie_l3592_359245


namespace NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_implies_a_gt_b_l3592_359236

theorem ac_squared_gt_bc_squared_implies_a_gt_b (a b c : ℝ) (hc : c ≠ 0) :
  a * c^2 > b * c^2 → a > b := by
  sorry

end NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_implies_a_gt_b_l3592_359236


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l3592_359255

theorem framed_painting_ratio : 
  let painting_width : ℝ := 20
  let painting_height : ℝ := 30
  let frame_side_width : ℝ := 2  -- This is the solution, but we don't use it in the statement
  let frame_top_bottom_width := 3 * frame_side_width
  let framed_width := painting_width + 2 * frame_side_width
  let framed_height := painting_height + 2 * frame_top_bottom_width
  (framed_width * framed_height - painting_width * painting_height = painting_width * painting_height) →
  (min framed_width framed_height) / (max framed_width framed_height) = 4 / 7 := by
sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l3592_359255


namespace NUMINAMATH_CALUDE_min_product_value_l3592_359281

def S (n : ℕ+) : ℚ := n / (n + 1)

def b (n : ℕ+) : ℤ := n - 8

def product (n : ℕ+) : ℚ := (b n : ℚ) * S n

theorem min_product_value :
  ∃ (m : ℕ+), ∀ (n : ℕ+), product m ≤ product n ∧ product m = -4 :=
sorry

end NUMINAMATH_CALUDE_min_product_value_l3592_359281


namespace NUMINAMATH_CALUDE_smallest_three_digit_with_equal_digit_sums_l3592_359258

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number satisfies the condition -/
def satisfiesCondition (n : ℕ) : Prop :=
  ∀ k : ℕ, k ≤ n → sumOfDigits n = sumOfDigits (k * n)

/-- Theorem statement -/
theorem smallest_three_digit_with_equal_digit_sums :
  ∃ n : ℕ, n = 999 ∧ 
    (∀ m : ℕ, 100 ≤ m ∧ m < 999 → ¬satisfiesCondition m) ∧
    satisfiesCondition n :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_with_equal_digit_sums_l3592_359258


namespace NUMINAMATH_CALUDE_inequality_proof_l3592_359230

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = a * b) :
  (a / (b^2 + 4)) + (b / (a^2 + 4)) ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3592_359230


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3592_359218

/-- A quadratic trinomial x^2 + mx + 1 is a perfect square if and only if m = ±2 -/
theorem perfect_square_trinomial (m : ℝ) :
  (∀ x, ∃ a, x^2 + m*x + 1 = (x + a)^2) ↔ (m = 2 ∨ m = -2) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3592_359218


namespace NUMINAMATH_CALUDE_dice_roll_circle_probability_l3592_359294

theorem dice_roll_circle_probability (r : ℕ) (h1 : 3 ≤ r) (h2 : r ≤ 18) :
  2 * Real.pi * r ≤ 2 * Real.pi * r^2 := by sorry

end NUMINAMATH_CALUDE_dice_roll_circle_probability_l3592_359294


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3592_359214

theorem sin_alpha_value (α : Real) (h1 : 0 < α ∧ α < π/2) 
  (h2 : Real.sin (α + π/2) = 3/5) : Real.sin α = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3592_359214


namespace NUMINAMATH_CALUDE_missing_number_solution_l3592_359243

theorem missing_number_solution : ∃ x : ℤ, (476 + 424) * x - 4 * 476 * 424 = 2704 ∧ x = 904 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_solution_l3592_359243


namespace NUMINAMATH_CALUDE_ellipse_k_value_l3592_359215

/-- An ellipse with equation 5x^2 + ky^2 = 5 and one focus at (0, 2) has k = 1 -/
theorem ellipse_k_value (k : ℝ) : 
  (∃ (x y : ℝ), 5 * x^2 + k * y^2 = 5) →  -- Equation of the ellipse
  (∃ (c : ℝ), c^2 = 5/k - 1) →            -- Property of ellipse: c^2 = a^2 - b^2
  (2 : ℝ)^2 = 5/k - 1 →                   -- Focus at (0, 2)
  k = 1 := by
sorry


end NUMINAMATH_CALUDE_ellipse_k_value_l3592_359215


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l3592_359217

theorem logarithm_expression_equality : 
  (Real.log 243 / Real.log 3) / (Real.log 81 / Real.log 3) - 
  (Real.log 729 / Real.log 3) / (Real.log 27 / Real.log 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l3592_359217


namespace NUMINAMATH_CALUDE_similar_quadrilateral_longest_side_l3592_359264

/-- Given a quadrilateral Q1 with side lengths a, b, c, d, and a similar quadrilateral Q2
    where the minimum side length of Q2 is equal to twice the minimum side length of Q1,
    prove that the longest side of Q2 is twice the longest side of Q1. -/
theorem similar_quadrilateral_longest_side
  (a b c d : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hmin : a ≤ b ∧ a ≤ c ∧ a ≤ d)
  (hmax : b ≤ d ∧ c ≤ d)
  (h_similar : ∃ (k : ℝ), k > 0 ∧ k * a = 2 * a) :
  ∃ (l : ℝ), l = 2 * d ∧ l = max (k * a) (max (k * b) (max (k * c) (k * d))) :=
sorry

end NUMINAMATH_CALUDE_similar_quadrilateral_longest_side_l3592_359264


namespace NUMINAMATH_CALUDE_initial_number_relation_l3592_359273

/-- The game sequence for Professor Célia's number game -/
def game_sequence (n : ℤ) : Vector ℤ 4 :=
  let c := 2 * (n + 1)
  let m := 3 * (c - 1)
  let a := 4 * (m + 1)
  ⟨[n, c, m, a], rfl⟩

/-- Theorem stating the relationship between the initial number and Ademar's number -/
theorem initial_number_relation (n x : ℤ) : 
  (game_sequence n).get 3 = x → n = (x - 16) / 24 :=
sorry

end NUMINAMATH_CALUDE_initial_number_relation_l3592_359273


namespace NUMINAMATH_CALUDE_roof_area_calculation_l3592_359288

def roof_area (width : ℝ) (length : ℝ) : ℝ :=
  width * length

theorem roof_area_calculation :
  ∀ w l : ℝ,
  l = 4 * w →
  l - w = 36 →
  roof_area w l = 576 :=
by
  sorry

end NUMINAMATH_CALUDE_roof_area_calculation_l3592_359288


namespace NUMINAMATH_CALUDE_tree_age_difference_l3592_359222

/-- The number of rings in one group -/
def rings_per_group : ℕ := 6

/-- The number of ring groups in the first tree -/
def first_tree_groups : ℕ := 70

/-- The number of ring groups in the second tree -/
def second_tree_groups : ℕ := 40

/-- Each ring represents one year of growth -/
axiom ring_year_correspondence : ∀ (n : ℕ), n.succ.pred = n

theorem tree_age_difference : 
  (first_tree_groups * rings_per_group) - (second_tree_groups * rings_per_group) = 180 := by
  sorry

end NUMINAMATH_CALUDE_tree_age_difference_l3592_359222


namespace NUMINAMATH_CALUDE_mikes_toy_expenses_l3592_359287

/-- The total amount Mike spent on toys -/
def total_spent (marbles_cost football_cost baseball_cost : ℚ) : ℚ :=
  marbles_cost + football_cost + baseball_cost

/-- Theorem stating the total amount Mike spent on toys -/
theorem mikes_toy_expenses :
  total_spent 9.05 4.95 6.52 = 20.52 := by sorry

end NUMINAMATH_CALUDE_mikes_toy_expenses_l3592_359287


namespace NUMINAMATH_CALUDE_chocolate_leftover_l3592_359206

/-- Calculates the amount of chocolate left over when making cookies -/
theorem chocolate_leftover (dough : ℝ) (total_chocolate : ℝ) (chocolate_percentage : ℝ) : 
  dough = 36 → 
  total_chocolate = 13 → 
  chocolate_percentage = 0.20 → 
  (total_chocolate - (chocolate_percentage * (dough + (chocolate_percentage * (dough + total_chocolate) / (1 - chocolate_percentage))))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_leftover_l3592_359206


namespace NUMINAMATH_CALUDE_parabola_triangle_area_l3592_359253

/-- Given a parabola y = x^2 - 20x + c (c ≠ 0) that intersects the x-axis at points A and B
    and the y-axis at point C, where A and C are symmetrical with respect to the line y = -x,
    the area of triangle ABC is 231. -/
theorem parabola_triangle_area (c : ℝ) (hc : c ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ x^2 - 20*x + c
  let A := (21 : ℝ)
  let B := (-1 : ℝ)
  let C := (0, c)
  (∀ x, f x = 0 → x = A ∨ x = B) →
  (f 0 = c) →
  (A, 0) = (-C.2, 0) →
  (1/2 : ℝ) * (A - B) * (-C.2) = 231 :=
by sorry

end NUMINAMATH_CALUDE_parabola_triangle_area_l3592_359253


namespace NUMINAMATH_CALUDE_cosine_sine_sum_equality_l3592_359260

theorem cosine_sine_sum_equality : 
  Real.cos (42 * π / 180) * Real.cos (78 * π / 180) + 
  Real.sin (42 * π / 180) * Real.cos (168 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_sum_equality_l3592_359260


namespace NUMINAMATH_CALUDE_cubic_function_property_l3592_359250

/-- Given a cubic function f(x) = ax³ - bx + 5 where a and b are real numbers,
    if f(-3) = -1, then f(3) = 11. -/
theorem cubic_function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - b * x + 5
  f (-3) = -1 → f 3 = 11 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l3592_359250


namespace NUMINAMATH_CALUDE_sin_120_degrees_l3592_359247

theorem sin_120_degrees : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l3592_359247


namespace NUMINAMATH_CALUDE_unique_solution_system_l3592_359204

theorem unique_solution_system :
  ∃! (x y z : ℝ),
    x^2 - 22*y - 69*z + 703 = 0 ∧
    y^2 + 23*x + 23*z - 1473 = 0 ∧
    z^2 - 63*x + 66*y + 2183 = 0 ∧
    x = 20 ∧ y = -22 ∧ z = 23 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3592_359204


namespace NUMINAMATH_CALUDE_solve_system_l3592_359275

theorem solve_system (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0)
  (eq1 : x = 2 + 1/z) (eq2 : z = 3 + 1/x) :
  z = (3 + Real.sqrt 15) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3592_359275


namespace NUMINAMATH_CALUDE_series_convergence_implies_scaled_convergence_l3592_359263

theorem series_convergence_implies_scaled_convergence 
  (a : ℕ → ℝ) (h : Summable a) : Summable (fun n => a n / n) := by
  sorry

end NUMINAMATH_CALUDE_series_convergence_implies_scaled_convergence_l3592_359263


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3592_359216

theorem sufficient_not_necessary (a b : ℝ) :
  ((a - b) * a^2 < 0 → a < b) ∧
  ¬(a < b → (a - b) * a^2 < 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3592_359216


namespace NUMINAMATH_CALUDE_number_and_square_sum_l3592_359212

theorem number_and_square_sum (x : ℝ) : x + x^2 = 132 → x = 11 ∨ x = -12 := by
  sorry

end NUMINAMATH_CALUDE_number_and_square_sum_l3592_359212


namespace NUMINAMATH_CALUDE_gcf_of_48_160_120_l3592_359213

theorem gcf_of_48_160_120 : Nat.gcd 48 (Nat.gcd 160 120) = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_48_160_120_l3592_359213


namespace NUMINAMATH_CALUDE_inequality_proof_l3592_359221

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + 2*b + 3*c = 9) : 1/a + 1/b + 1/c ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3592_359221


namespace NUMINAMATH_CALUDE_distance_to_origin_of_fourth_point_on_circle_l3592_359279

/-- Given four points on a circle, prove that the distance from the fourth point to the origin is √13 -/
theorem distance_to_origin_of_fourth_point_on_circle 
  (A B C D : ℝ × ℝ) 
  (hA : A = (-2, 1)) 
  (hB : B = (-1, 0)) 
  (hC : C = (2, 3)) 
  (hD : D.2 = 3) 
  (h_circle : ∃ (center : ℝ × ℝ) (radius : ℝ), 
    (center.1 - A.1)^2 + (center.2 - A.2)^2 = radius^2 ∧
    (center.1 - B.1)^2 + (center.2 - B.2)^2 = radius^2 ∧
    (center.1 - C.1)^2 + (center.2 - C.2)^2 = radius^2 ∧
    (center.1 - D.1)^2 + (center.2 - D.2)^2 = radius^2) :
  Real.sqrt (D.1^2 + D.2^2) = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_distance_to_origin_of_fourth_point_on_circle_l3592_359279


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3592_359295

theorem quadratic_equation_solutions :
  let eq1 : ℂ → Prop := λ x ↦ x^2 - 6*x + 13 = 0
  let eq2 : ℂ → Prop := λ x ↦ 9*x^2 + 12*x + 29 = 0
  let sol1 : Set ℂ := {3 - 2*I, 3 + 2*I}
  let sol2 : Set ℂ := {-2/3 - 5/3*I, -2/3 + 5/3*I}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ x, eq1 x → x ∈ sol1) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ x, eq2 x → x ∈ sol2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3592_359295


namespace NUMINAMATH_CALUDE_triangle_problem_l3592_359282

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.b * Real.cos t.C + t.c * Real.cos t.B = 2 * t.a * Real.cos t.A)
  (h2 : t.b * t.c * Real.cos t.A = Real.sqrt 3) :
  t.A = π / 3 ∧ (1 / 2) * t.b * t.c * Real.sin t.A = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3592_359282


namespace NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l3592_359269

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} :=
sorry

-- Part 2: Range of a when f(x) ≥ 4
theorem range_of_a :
  {a : ℝ | ∃ x, f x a ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l3592_359269


namespace NUMINAMATH_CALUDE_remainder_theorem_l3592_359299

theorem remainder_theorem (d r : ℤ) : 
  d > 1 → 
  1059 % d = r →
  1417 % d = r →
  2312 % d = r →
  d - r = 15 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3592_359299


namespace NUMINAMATH_CALUDE_power_of_power_equals_power_of_product_three_squared_to_fourth_power_l3592_359246

theorem power_of_power_equals_power_of_product (a m n : ℕ) :
  (a^m)^n = a^(m*n) :=
sorry

theorem three_squared_to_fourth_power :
  (3^2)^4 = 3^8 ∧ 3^8 = 6561 :=
sorry

end NUMINAMATH_CALUDE_power_of_power_equals_power_of_product_three_squared_to_fourth_power_l3592_359246


namespace NUMINAMATH_CALUDE_parabola_distance_difference_l3592_359232

/-- Parabola type representing y² = 4x -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Function to check if a point is on the parabola -/
def on_parabola (p : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 4 * pt.x

/-- Function to check if a point is on a line -/
def on_line (l : Line) (pt : Point) : Prop :=
  pt.y = l.slope * pt.x + l.intercept

/-- Function to check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- Main theorem -/
theorem parabola_distance_difference 
  (p : Parabola)
  (F N A B : Point)
  (l : Line) :
  p.focus = (1, 0) →
  p.directrix = -1 →
  N.x = -1 ∧ N.y = 0 →
  on_parabola p A →
  on_parabola p B →
  on_line l A →
  on_line l B →
  on_line l F →
  perpendicular (Line.mk (B.y / (B.x - N.x)) 0) l →
  |A.x - F.x| - |B.x - F.x| = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_distance_difference_l3592_359232


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3592_359237

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 7 + 1
  let expr := (x^2 / (x - 3) - 2 * x / (x - 3)) / (x / (x - 3))
  expr = x - 2 ∧ expr = Real.sqrt 7 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3592_359237


namespace NUMINAMATH_CALUDE_cards_given_to_jeff_l3592_359290

/-- The number of cards Nell initially had -/
def initial_cards : ℕ := 455

/-- The number of cards Nell has left -/
def remaining_cards : ℕ := 154

/-- The number of cards Nell gave to Jeff -/
def cards_given : ℕ := initial_cards - remaining_cards

theorem cards_given_to_jeff : cards_given = 301 := by
  sorry

end NUMINAMATH_CALUDE_cards_given_to_jeff_l3592_359290


namespace NUMINAMATH_CALUDE_joe_fruit_probability_l3592_359261

/-- The number of fruit types Joe can choose from -/
def num_fruit_types : ℕ := 4

/-- The number of meals Joe has in a day -/
def num_meals : ℕ := 4

/-- The probability of choosing a specific fruit for one meal -/
def prob_one_fruit : ℚ := 1 / num_fruit_types

/-- The probability of eating the same fruit for all meals -/
def prob_same_fruit : ℚ := num_fruit_types * (prob_one_fruit ^ num_meals)

/-- The probability of eating at least two different kinds of fruit in one day -/
def prob_different_fruits : ℚ := 1 - prob_same_fruit

theorem joe_fruit_probability : prob_different_fruits = 63 / 64 := by
  sorry

end NUMINAMATH_CALUDE_joe_fruit_probability_l3592_359261


namespace NUMINAMATH_CALUDE_infinitely_many_primes_4k_minus_1_l3592_359219

theorem infinitely_many_primes_4k_minus_1 : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ k : ℕ, p = 4 * k - 1} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_4k_minus_1_l3592_359219


namespace NUMINAMATH_CALUDE_table_size_lower_bound_l3592_359207

/-- A table with 10 columns and n rows, where each cell contains a digit -/
structure Table (n : ℕ) :=
  (cells : Fin n → Fin 10 → Fin 10)

/-- The property that for each row and any two columns, there exists a row
    that differs from it in exactly these two columns -/
def has_differing_rows (t : Table n) : Prop :=
  ∀ (row : Fin n) (col1 col2 : Fin 10),
    col1 ≠ col2 →
    ∃ (diff_row : Fin n),
      (∀ (col : Fin 10), col ≠ col1 ∧ col ≠ col2 → t.cells diff_row col = t.cells row col) ∧
      t.cells diff_row col1 ≠ t.cells row col1 ∧
      t.cells diff_row col2 ≠ t.cells row col2

theorem table_size_lower_bound {n : ℕ} (t : Table n) (h : has_differing_rows t) :
  n ≥ 512 :=
sorry

end NUMINAMATH_CALUDE_table_size_lower_bound_l3592_359207


namespace NUMINAMATH_CALUDE_petes_total_distance_l3592_359296

/-- Represents the distance Pete traveled in blocks for each leg of his journey -/
structure Journey where
  house_to_garage : ℕ
  garage_to_post_office : ℕ
  post_office_to_friend : ℕ

/-- Calculates the total distance traveled for a round trip -/
def total_distance (j : Journey) : ℕ :=
  2 * (j.house_to_garage + j.garage_to_post_office + j.post_office_to_friend)

/-- Pete's actual journey -/
def petes_journey : Journey :=
  { house_to_garage := 5
  , garage_to_post_office := 20
  , post_office_to_friend := 10 }

/-- Theorem stating that Pete traveled 70 blocks in total -/
theorem petes_total_distance : total_distance petes_journey = 70 := by
  sorry

end NUMINAMATH_CALUDE_petes_total_distance_l3592_359296


namespace NUMINAMATH_CALUDE_unique_solution_sum_l3592_359249

-- Define the equation
def satisfies_equation (x y : ℕ+) : Prop :=
  (x : ℝ)^2 + 84 * (x : ℝ) + 2008 = (y : ℝ)^2

-- State the theorem
theorem unique_solution_sum :
  ∃! (x y : ℕ+), satisfies_equation x y ∧ x + y = 80 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_sum_l3592_359249


namespace NUMINAMATH_CALUDE_intersection_M_N_l3592_359289

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {x | x^2 - 4*x < 0}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3592_359289


namespace NUMINAMATH_CALUDE_cookie_boxes_problem_l3592_359200

theorem cookie_boxes_problem (n : ℕ) : 
  n - 7 ≥ 1 → 
  n - 2 ≥ 1 → 
  (n - 7) + (n - 2) < n → 
  n = 8 :=
by sorry

end NUMINAMATH_CALUDE_cookie_boxes_problem_l3592_359200


namespace NUMINAMATH_CALUDE_tates_education_years_l3592_359277

/-- The total years Tate spent in high school and college -/
def total_education_years (normal_hs_duration : ℕ) (hs_reduction : ℕ) (college_multiplier : ℕ) : ℕ :=
  let hs_duration := normal_hs_duration - hs_reduction
  let college_duration := hs_duration * college_multiplier
  hs_duration + college_duration

/-- Theorem stating that Tate's total education years is 12 -/
theorem tates_education_years :
  total_education_years 4 1 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_tates_education_years_l3592_359277


namespace NUMINAMATH_CALUDE_factorization_a_squared_minus_one_l3592_359248

theorem factorization_a_squared_minus_one (a : ℝ) : a^2 - 1 = (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_squared_minus_one_l3592_359248


namespace NUMINAMATH_CALUDE_asymptotes_of_hyperbola_l3592_359205

theorem asymptotes_of_hyperbola (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let e1 := Real.sqrt (a^2 - b^2) / a
  let e2 := Real.sqrt (a^2 + b^2) / a
  let C1 := fun (x y : ℝ) ↦ x^2 / a^2 + y^2 / b^2 = 1
  let C2 := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  e1 * e2 = Real.sqrt 15 / 4 →
  (∀ x y, C2 x y → (x + 2*y = 0 ∨ x - 2*y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_asymptotes_of_hyperbola_l3592_359205


namespace NUMINAMATH_CALUDE_club_leader_selection_l3592_359225

/-- Represents a club with members of two genders, some wearing glasses -/
structure Club where
  total_members : Nat
  boys : Nat
  girls : Nat
  boys_with_glasses : Nat
  girls_with_glasses : Nat

/-- Calculates the number of ways to choose a president and vice-president -/
def ways_to_choose_leaders (c : Club) : Nat :=
  (c.boys_with_glasses * (c.boys_with_glasses - 1)) +
  (c.girls_with_glasses * (c.girls_with_glasses - 1))

/-- The main theorem to prove -/
theorem club_leader_selection (c : Club) 
  (h1 : c.total_members = 24)
  (h2 : c.boys = 12)
  (h3 : c.girls = 12)
  (h4 : c.boys_with_glasses = 6)
  (h5 : c.girls_with_glasses = 6) :
  ways_to_choose_leaders c = 60 := by
  sorry

#eval ways_to_choose_leaders { total_members := 24, boys := 12, girls := 12, boys_with_glasses := 6, girls_with_glasses := 6 }

end NUMINAMATH_CALUDE_club_leader_selection_l3592_359225


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3592_359278

theorem fraction_sum_equality : 
  (2 / 20 : ℚ) + (3 / 50 : ℚ) * (5 / 100 : ℚ) + (4 / 1000 : ℚ) + (6 / 10000 : ℚ) = 1076 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3592_359278


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_relation_l3592_359291

/-- Given two cubic polynomials h and j, where the roots of j are one less than the roots of h,
    prove that the coefficients of j are (1, 2, 1) -/
theorem cubic_polynomial_root_relation (x : ℝ) :
  let h := fun (x : ℝ) => x^3 - 2*x^2 + 3*x - 1
  let j := fun (x : ℝ) => x^3 + b*x^2 + c*x + d
  (∀ s, h s = 0 → j (s - 1) = 0) →
  (b, c, d) = (1, 2, 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_relation_l3592_359291


namespace NUMINAMATH_CALUDE_geometric_sequence_special_case_l3592_359226

/-- A geometric sequence with first term 1 and nth term equal to the product of the first 5 terms has n = 11 -/
theorem geometric_sequence_special_case (a : ℕ → ℝ) (n : ℕ) : 
  (∀ k, a (k + 1) / a k = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 →                            -- first term is 1
  a n = a 1 * a 2 * a 3 * a 4 * a 5 →   -- nth term equals product of first 5 terms
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_case_l3592_359226


namespace NUMINAMATH_CALUDE_system_inequalities_solution_set_l3592_359211

theorem system_inequalities_solution_set (m : ℝ) :
  (∀ x : ℝ, x > 4 ∧ x > m ↔ x > 4) ↔ m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_system_inequalities_solution_set_l3592_359211


namespace NUMINAMATH_CALUDE_cistern_emptying_time_l3592_359251

theorem cistern_emptying_time (fill_time : ℝ) (combined_fill_time : ℝ) (empty_time : ℝ) : 
  fill_time = 2 → 
  combined_fill_time = 2.571428571428571 →
  (1 / fill_time) - (1 / empty_time) = (1 / combined_fill_time) →
  empty_time = 9 := by
  sorry

end NUMINAMATH_CALUDE_cistern_emptying_time_l3592_359251


namespace NUMINAMATH_CALUDE_table_length_proof_l3592_359298

/-- Proves that the length of the table is 77 cm given the conditions of the paper placement problem. -/
theorem table_length_proof (table_width : ℕ) (sheet_width sheet_height : ℕ) (x : ℕ) :
  table_width = 80 ∧
  sheet_width = 8 ∧
  sheet_height = 5 ∧
  (x - sheet_height : ℤ) = (table_width - sheet_width : ℤ) →
  x = 77 := by
  sorry

end NUMINAMATH_CALUDE_table_length_proof_l3592_359298


namespace NUMINAMATH_CALUDE_dilation_problem_l3592_359231

/-- Dilation of a complex number -/
def dilation (z : ℂ) (center : ℂ) (scale : ℝ) : ℂ :=
  center + scale * (z - center)

/-- The problem statement -/
theorem dilation_problem : dilation (-2 + I) (1 - 3*I) 3 = -8 + 9*I := by
  sorry

end NUMINAMATH_CALUDE_dilation_problem_l3592_359231


namespace NUMINAMATH_CALUDE_arithmetic_sequence_and_equation_l3592_359285

-- Define arithmetic sequence
def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

-- Define the equation from proposition B
def satisfies_equation (a b c : ℝ) : Prop :=
  b ≠ 0 ∧ a / b + c / b = 2

-- Theorem statement
theorem arithmetic_sequence_and_equation :
  (∀ a b c : ℝ, satisfies_equation a b c → is_arithmetic_sequence a b c) ∧
  (∃ a b c : ℝ, is_arithmetic_sequence a b c ∧ ¬satisfies_equation a b c) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_and_equation_l3592_359285


namespace NUMINAMATH_CALUDE_triangle_inequality_l3592_359268

open Real

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b^2 + c^2) / a + (c^2 + a^2) / b + (a^2 + b^2) / c ≥ 2 * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3592_359268


namespace NUMINAMATH_CALUDE_prime_sum_squares_l3592_359280

theorem prime_sum_squares (p q m : ℕ) : 
  p.Prime → q.Prime → p ≠ q →
  p^2 - 2001*p + m = 0 → q^2 - 2001*q + m = 0 →
  p^2 + q^2 = 3996005 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_squares_l3592_359280


namespace NUMINAMATH_CALUDE_school_distance_l3592_359293

/-- The distance between a child's home and school, given two walking scenarios. -/
theorem school_distance (v₁ v₂ : ℝ) (t₁ t₂ : ℝ) (D : ℝ) : 
  v₁ = 5 →  -- First walking speed in m/min
  v₂ = 7 →  -- Second walking speed in m/min
  t₁ = 6 →  -- Late time in minutes for first scenario
  t₂ = 30 → -- Early time in minutes for second scenario
  v₁ * (D / v₁ + t₁) = D →  -- Equation for first scenario
  v₂ * (D / v₂ - t₂) = D →  -- Equation for second scenario
  D = 630 := by
sorry

end NUMINAMATH_CALUDE_school_distance_l3592_359293


namespace NUMINAMATH_CALUDE_concatenated_numbers_remainder_l3592_359208

-- Define a function to concatenate numbers from 1 to n
def concatenateNumbers (n : ℕ) : ℕ := sorry

-- Define a function to calculate the remainder when a number is divided by 9
def remainderMod9 (n : ℕ) : ℕ := n % 9

-- Theorem statement
theorem concatenated_numbers_remainder (n : ℕ) (h : n = 2001) :
  remainderMod9 (concatenateNumbers n) = 6 := by sorry

end NUMINAMATH_CALUDE_concatenated_numbers_remainder_l3592_359208


namespace NUMINAMATH_CALUDE_mode_and_median_of_data_set_l3592_359262

def data_set : List ℕ := [9, 16, 18, 23, 32, 23, 48, 23]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem mode_and_median_of_data_set :
  mode data_set = 23 ∧ median data_set = 23 := by sorry

end NUMINAMATH_CALUDE_mode_and_median_of_data_set_l3592_359262


namespace NUMINAMATH_CALUDE_egyptian_fraction_identity_l3592_359209

theorem egyptian_fraction_identity (n : ℕ+) :
  (2 : ℚ) / (2 * n + 1) = 1 / (n + 1) + 1 / ((n + 1) * (2 * n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_egyptian_fraction_identity_l3592_359209


namespace NUMINAMATH_CALUDE_equation_describes_hyperbola_l3592_359210

/-- The equation (x-y)^2 = x^2 + y^2 - 2 describes a hyperbola -/
theorem equation_describes_hyperbola :
  ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (∀ (x y : ℝ), (x - y)^2 = x^2 + y^2 - 2 ↔ (x * y = 1)) :=
sorry

end NUMINAMATH_CALUDE_equation_describes_hyperbola_l3592_359210


namespace NUMINAMATH_CALUDE_g_of_4_equals_18_l3592_359267

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x - 2

-- Theorem statement
theorem g_of_4_equals_18 : g 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_g_of_4_equals_18_l3592_359267


namespace NUMINAMATH_CALUDE_g_max_min_sum_l3592_359252

def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2*x - 8| + x

theorem g_max_min_sum :
  ∃ (max min : ℝ), 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 10 → g x ≤ max) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 10 ∧ g x = max) ∧
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 10 → min ≤ g x) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 10 ∧ g x = min) ∧
    max + min = 7 :=
by sorry

end NUMINAMATH_CALUDE_g_max_min_sum_l3592_359252


namespace NUMINAMATH_CALUDE_penguins_to_feed_l3592_359203

theorem penguins_to_feed (total_penguins : ℕ) (fed_penguins : ℕ) 
  (h1 : total_penguins = 36) 
  (h2 : fed_penguins = 19) : 
  total_penguins - fed_penguins = 17 := by
  sorry

end NUMINAMATH_CALUDE_penguins_to_feed_l3592_359203


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3592_359202

def A : Set ℝ := {x | 2 * x ≤ 1}
def B : Set ℝ := {-1, 0, 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3592_359202


namespace NUMINAMATH_CALUDE_parabola_equation_proof_l3592_359239

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 13 - y^2 / 12 = 1

/-- The right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (5, 0)

/-- The vertex of the parabola -/
def parabola_vertex : ℝ × ℝ := (0, 0)

/-- The focus of the parabola -/
def parabola_focus : ℝ × ℝ := right_focus

/-- The equation of the parabola -/
def parabola_equation (x y : ℝ) : Prop := y^2 = 20 * x

theorem parabola_equation_proof :
  ∀ x y : ℝ, parabola_equation x y ↔ 
  (parabola_vertex = (0, 0) ∧ parabola_focus = right_focus) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_proof_l3592_359239


namespace NUMINAMATH_CALUDE_alice_class_size_l3592_359284

/-- The number of students in Alice's white water rafting class -/
def num_students : ℕ := 40

/-- The number of instructors, including Alice -/
def num_instructors : ℕ := 10

/-- The number of life vests Alice has on hand -/
def vests_on_hand : ℕ := 20

/-- The percentage of students bringing their own life vests -/
def percent_students_with_vests : ℚ := 1/5

/-- The additional number of life vests Alice needs to get -/
def additional_vests_needed : ℕ := 22

theorem alice_class_size :
  num_students = 40 ∧
  (num_students + num_instructors) * (1 - percent_students_with_vests) =
    vests_on_hand + additional_vests_needed :=
by sorry

end NUMINAMATH_CALUDE_alice_class_size_l3592_359284


namespace NUMINAMATH_CALUDE_coins_after_fifth_hour_l3592_359228

def coins_in_jar (hour1 : ℕ) (hour2_3 : ℕ) (hour4 : ℕ) (taken_out : ℕ) : ℕ :=
  hour1 + 2 * hour2_3 + hour4 - taken_out

theorem coins_after_fifth_hour :
  coins_in_jar 20 30 40 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_coins_after_fifth_hour_l3592_359228


namespace NUMINAMATH_CALUDE_simplify_expression_l3592_359265

theorem simplify_expression (m : ℝ) : 150*m - 72*m + 3*(5*m) = 93*m := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3592_359265


namespace NUMINAMATH_CALUDE_division_remainder_proof_l3592_359238

theorem division_remainder_proof (dividend : Nat) (divisor : Nat) (quotient : Nat) (h1 : dividend = 109) (h2 : divisor = 12) (h3 : quotient = 9) :
  dividend % divisor = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l3592_359238


namespace NUMINAMATH_CALUDE_stratified_sample_correct_l3592_359286

/-- Represents the number of students in each year and the sample size -/
structure SchoolData where
  total_students : ℕ
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ
  sample_size : ℕ

/-- Represents the number of students to be sampled from each year -/
structure SampleAllocation where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- Calculates the correct sample allocation for stratified sampling -/
def stratifiedSample (data : SchoolData) : SampleAllocation :=
  { first_year := data.sample_size * data.first_year / data.total_students,
    second_year := data.sample_size * data.second_year / data.total_students,
    third_year := data.sample_size * data.third_year / data.total_students }

/-- Theorem stating that the stratified sampling allocation is correct -/
theorem stratified_sample_correct (data : SchoolData)
  (h1 : data.total_students = 2700)
  (h2 : data.first_year = 900)
  (h3 : data.second_year = 1200)
  (h4 : data.third_year = 600)
  (h5 : data.sample_size = 135) :
  stratifiedSample data = { first_year := 45, second_year := 60, third_year := 30 } :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_correct_l3592_359286


namespace NUMINAMATH_CALUDE_books_left_to_read_l3592_359241

theorem books_left_to_read (total_books read_books : ℕ) : 
  total_books = 14 → read_books = 8 → total_books - read_books = 6 := by
sorry

end NUMINAMATH_CALUDE_books_left_to_read_l3592_359241


namespace NUMINAMATH_CALUDE_emily_cookies_l3592_359254

theorem emily_cookies (e : ℚ) 
  (total : ℚ) 
  (h1 : total = e + 3*e + 2*(3*e) + 4*(2*(3*e))) 
  (h2 : total = 90) : e = 45/17 := by
  sorry

end NUMINAMATH_CALUDE_emily_cookies_l3592_359254


namespace NUMINAMATH_CALUDE_smaller_number_puzzle_l3592_359297

theorem smaller_number_puzzle (x y : ℝ) (h_sum : x + y = 18) (h_product : x * y = 80) :
  min x y = 8 := by sorry

end NUMINAMATH_CALUDE_smaller_number_puzzle_l3592_359297


namespace NUMINAMATH_CALUDE_simplify_expression_l3592_359259

theorem simplify_expression (x : ℝ) : 
  Real.sqrt (1 + ((x^6 - 1) / (3 * x^3))^2) = (Real.sqrt (x^12 + 7*x^6 + 1)) / (3 * x^3) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3592_359259


namespace NUMINAMATH_CALUDE_expression_simplification_l3592_359220

theorem expression_simplification (m n : ℝ) 
  (h : Real.sqrt (m - 1/2) + (n + 2)^2 = 0) : 
  ((3*m + n) * (m + n) - (2*m - n)^2 + (m + 2*n) * (m - 2*n)) / (2*n) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3592_359220


namespace NUMINAMATH_CALUDE_integral_inequality_l3592_359270

theorem integral_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a ≤ b) :
  (2 / Real.sqrt 3) * Real.arctan ((2 * (b^2 - a^2)) / ((a^2 + 2) * (b^2 + 2))) ≤
  (∫ (x : ℝ) in a..b, ((x^2 + 1) * (x^2 + x + 1)) / ((x^3 + x^2 + 1) * (x^3 + x + 1))) ∧
  (∫ (x : ℝ) in a..b, ((x^2 + 1) * (x^2 + x + 1)) / ((x^3 + x^2 + 1) * (x^3 + x + 1))) ≤
  (4 / Real.sqrt 3) * Real.arctan (((b - a) * Real.sqrt 3) / (a + b + 2 * (1 + a * b))) :=
by sorry

end NUMINAMATH_CALUDE_integral_inequality_l3592_359270


namespace NUMINAMATH_CALUDE_eat_chips_in_ten_days_l3592_359235

/-- The number of days it takes to eat all chips in a bag -/
def days_to_eat_chips (total_chips : ℕ) (first_day_chips : ℕ) (daily_chips : ℕ) : ℕ :=
  1 + (total_chips - first_day_chips) / daily_chips

/-- Theorem: It takes 10 days to eat a bag of 100 chips -/
theorem eat_chips_in_ten_days :
  days_to_eat_chips 100 10 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_eat_chips_in_ten_days_l3592_359235


namespace NUMINAMATH_CALUDE_five_by_seven_double_covered_cells_l3592_359227

/-- Represents a rectangular grid with fold lines -/
structure FoldableGrid :=
  (rows : ℕ)
  (cols : ℕ)
  (foldLines : List (ℕ × ℕ × ℕ × ℕ))  -- List of start and end points of fold lines

/-- Counts the number of cells covered exactly twice after folding -/
def countDoubleCoveredCells (grid : FoldableGrid) : ℕ :=
  sorry

/-- The main theorem stating that a 5x7 grid with specific fold lines has 9 double-covered cells -/
theorem five_by_seven_double_covered_cells :
  ∃ (foldLines : List (ℕ × ℕ × ℕ × ℕ)),
    let grid := FoldableGrid.mk 5 7 foldLines
    countDoubleCoveredCells grid = 9 :=
  sorry

end NUMINAMATH_CALUDE_five_by_seven_double_covered_cells_l3592_359227


namespace NUMINAMATH_CALUDE_total_trees_count_l3592_359292

/-- Represents the number of Douglas fir trees -/
def D : ℕ := 350

/-- Represents the number of ponderosa pine trees -/
def P : ℕ := 500

/-- The cost of a Douglas fir tree -/
def douglas_cost : ℕ := 300

/-- The cost of a ponderosa pine tree -/
def ponderosa_cost : ℕ := 225

/-- The total cost paid for all trees -/
def total_cost : ℕ := 217500

/-- Theorem stating that given the conditions, the total number of trees is 850 -/
theorem total_trees_count : D + P = 850 ∧ 
  douglas_cost * D + ponderosa_cost * P = total_cost ∧
  (D = 350 ∨ P = 350) := by
  sorry

#check total_trees_count

end NUMINAMATH_CALUDE_total_trees_count_l3592_359292


namespace NUMINAMATH_CALUDE_second_train_speed_l3592_359256

/-- Proves that the speed of the second train is 16 km/hr given the problem conditions -/
theorem second_train_speed (speed1 : ℝ) (total_distance : ℝ) (distance_difference : ℝ) :
  speed1 = 20 →
  total_distance = 450 →
  distance_difference = 50 →
  ∃ (speed2 : ℝ) (time : ℝ),
    speed2 > 0 ∧
    time > 0 ∧
    speed1 * time = speed2 * time + distance_difference ∧
    speed1 * time + speed2 * time = total_distance ∧
    speed2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_second_train_speed_l3592_359256


namespace NUMINAMATH_CALUDE_train_passing_time_l3592_359276

/-- The time it takes for a faster train to completely pass a slower train -/
theorem train_passing_time (v_fast v_slow : ℝ) (length : ℝ) (h_fast : v_fast = 50) (h_slow : v_slow = 32) (h_length : length = 75) :
  (length / ((v_fast - v_slow) * (1000 / 3600))) = 15 :=
sorry

end NUMINAMATH_CALUDE_train_passing_time_l3592_359276


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3592_359233

theorem sufficient_not_necessary (a b x : ℝ) :
  (∀ x, x > a^2 + b^2 → x > 2*a*b) ∧
  (∃ a b x, x > 2*a*b ∧ x ≤ a^2 + b^2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3592_359233
