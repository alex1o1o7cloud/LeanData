import Mathlib

namespace NUMINAMATH_CALUDE_flagpole_break_height_l1514_151484

theorem flagpole_break_height (h : ℝ) (b : ℝ) (x : ℝ) 
  (hypotenuse : h = 6)
  (base : b = 2)
  (right_triangle : x^2 + b^2 = h^2) :
  x = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_flagpole_break_height_l1514_151484


namespace NUMINAMATH_CALUDE_inscribed_circle_area_l1514_151485

/-- The area of the circle inscribed in a right triangle with perimeter 2p and hypotenuse c is π(p - c)². -/
theorem inscribed_circle_area (p c : ℝ) (h1 : 0 < p) (h2 : 0 < c) (h3 : c < 2 * p) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y + c = 2 * p ∧
  ∃ (r : ℝ), r > 0 ∧ r = p - c ∧
  ∃ (S : ℝ), S = π * r^2 ∧ S = π * (p - c)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_area_l1514_151485


namespace NUMINAMATH_CALUDE_expression_evaluation_l1514_151433

theorem expression_evaluation :
  let x : ℝ := 2 * Real.sqrt 3
  (x - Real.sqrt 2) * (x + Real.sqrt 2) + x * (x - 1) = 22 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1514_151433


namespace NUMINAMATH_CALUDE_rectangular_prism_dimensions_l1514_151477

/-- Proves that a rectangular prism with given conditions has length 9 and width 3 -/
theorem rectangular_prism_dimensions :
  ∀ l w h : ℝ,
  l = 3 * w →
  h = 12 →
  Real.sqrt (l^2 + w^2 + h^2) = 15 →
  l = 9 ∧ w = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_dimensions_l1514_151477


namespace NUMINAMATH_CALUDE_puzzle_sum_l1514_151466

theorem puzzle_sum (a b c : ℤ) 
  (h1 : a + b = 31) 
  (h2 : b + c = 48) 
  (h3 : c + a = 59) 
  (h4 : a ≠ b ∧ b ≠ c ∧ c ≠ a) : 
  a + b + c = 69 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_sum_l1514_151466


namespace NUMINAMATH_CALUDE_fourth_root_16_times_cube_root_8_times_sqrt_4_eq_8_l1514_151450

theorem fourth_root_16_times_cube_root_8_times_sqrt_4_eq_8 :
  (16 : ℝ) ^ (1/4) * (8 : ℝ) ^ (1/3) * (4 : ℝ) ^ (1/2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_16_times_cube_root_8_times_sqrt_4_eq_8_l1514_151450


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1514_151496

theorem cubic_roots_sum (m : ℤ) (p q r : ℤ) : 
  (∀ x, x^3 - 2023*x + m = (x - p) * (x - q) * (x - r)) →
  |p| + |q| + |r| = 100 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1514_151496


namespace NUMINAMATH_CALUDE_loggers_required_is_eight_l1514_151463

/-- Represents the number of loggers required to cut down all trees in a forest under specific conditions. -/
def number_of_loggers (forest_length : ℕ) (forest_width : ℕ) (trees_per_square_mile : ℕ) 
  (trees_per_day : ℕ) (days_per_month : ℕ) (months_to_complete : ℕ) : ℕ :=
  (forest_length * forest_width * trees_per_square_mile) / 
  (trees_per_day * days_per_month * months_to_complete)

/-- Theorem stating that the number of loggers required under the given conditions is 8. -/
theorem loggers_required_is_eight :
  number_of_loggers 4 6 600 6 30 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_loggers_required_is_eight_l1514_151463


namespace NUMINAMATH_CALUDE_line_satisfies_conditions_l1514_151492

-- Define the lines
def line1 (x y : ℝ) : Prop := x - 2*y + 4 = 0
def line2 (x y : ℝ) : Prop := x + y - 2 = 0
def line3 (x y : ℝ) : Prop := 3*x - 4*y + 7 = 0

-- Define the result line
def result_line (x y : ℝ) : Prop := 10*x + 13*y - 26 = 0

-- Theorem statement
theorem line_satisfies_conditions :
  -- The result line passes through the intersection of line1 and line2
  (∃ x y : ℝ, line1 x y ∧ line2 x y ∧ result_line x y) ∧
  -- The result line passes through the point (3, -2)
  (result_line 3 (-2)) ∧
  -- The result line is perpendicular to line3
  (∃ m1 m2 : ℝ, 
    (∀ x y : ℝ, line3 x y → y = m1 * x + (7 / 4)) ∧
    (∀ x y : ℝ, result_line x y → y = m2 * x + (26 / 10)) ∧
    m1 * m2 = -1) :=
by sorry

end NUMINAMATH_CALUDE_line_satisfies_conditions_l1514_151492


namespace NUMINAMATH_CALUDE_cubic_polynomial_special_roots_l1514_151416

/-- A polynomial of degree 3 with real coefficients -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The roots of a cubic polynomial -/
structure CubicRoots where
  r : ℝ
  s : ℝ
  t : ℝ

/-- Proposition: For a cubic polynomial x^3 - 5x^2 + 2bx - c with real and positive roots,
    where one root is twice another and four times the third, c = 1000/343 -/
theorem cubic_polynomial_special_roots (p : CubicPolynomial) (roots : CubicRoots) :
  p.a = 1 ∧ p.b = -5 ∧  -- Coefficients of the polynomial
  (∀ x, x^3 - 5*x^2 + 2*p.b*x - p.c = 0 ↔ x = roots.r ∨ x = roots.s ∨ x = roots.t) ∧  -- Roots definition
  roots.r > 0 ∧ roots.s > 0 ∧ roots.t > 0 ∧  -- Roots are positive
  roots.s = 2 * roots.t ∧ roots.r = 4 * roots.t  -- Root relationships
  →
  p.c = 1000 / 343 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_special_roots_l1514_151416


namespace NUMINAMATH_CALUDE_davids_math_marks_l1514_151425

theorem davids_math_marks
  (english_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℚ)
  (num_subjects : ℕ)
  (h1 : english_marks = 70)
  (h2 : physics_marks = 80)
  (h3 : chemistry_marks = 63)
  (h4 : biology_marks = 65)
  (h5 : average_marks = 68.2)
  (h6 : num_subjects = 5) :
  ∃ math_marks : ℕ,
    math_marks = 63 ∧
    (english_marks + physics_marks + chemistry_marks + biology_marks + math_marks : ℚ) / num_subjects = average_marks :=
by sorry

end NUMINAMATH_CALUDE_davids_math_marks_l1514_151425


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l1514_151418

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m = 0 ∧ y^2 - 2*y + m = 0) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l1514_151418


namespace NUMINAMATH_CALUDE_average_income_P_Q_l1514_151498

/-- Given the monthly incomes of three people P, Q, and R, prove that the average monthly income of P and Q is 5050, given certain conditions. -/
theorem average_income_P_Q (P Q R : ℕ) : 
  (Q + R) / 2 = 6250 →  -- Average income of Q and R
  (P + R) / 2 = 5200 →  -- Average income of P and R
  P = 4000 →            -- Income of P
  (P + Q) / 2 = 5050 :=  -- Average income of P and Q
by sorry

end NUMINAMATH_CALUDE_average_income_P_Q_l1514_151498


namespace NUMINAMATH_CALUDE_correct_rounding_sum_l1514_151454

def round_to_nearest_hundred (n : ℤ) : ℤ :=
  (n + 50) / 100 * 100

theorem correct_rounding_sum : round_to_nearest_hundred (125 + 96) = 200 := by
  sorry

end NUMINAMATH_CALUDE_correct_rounding_sum_l1514_151454


namespace NUMINAMATH_CALUDE_total_fish_l1514_151456

def micah_fish : ℕ := 7

def kenneth_fish (m : ℕ) : ℕ := 3 * m

def matthias_fish (k : ℕ) : ℕ := k - 15

theorem total_fish :
  micah_fish + kenneth_fish micah_fish + matthias_fish (kenneth_fish micah_fish) = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_l1514_151456


namespace NUMINAMATH_CALUDE_bisector_sum_ratio_bound_bisector_sum_ratio_bound_tight_l1514_151404

/-- A triangle with sides a, b, c and angle bisectors l_a, l_b -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  l_a : ℝ
  l_b : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  bisector_formula_a : l_a = (2 * b * c * Real.sqrt ((1 + (b^2 + c^2 - a^2) / (2 * b * c)) / 2)) / (b + c)
  bisector_formula_b : l_b = (2 * a * c * Real.sqrt ((1 + (a^2 + c^2 - b^2) / (2 * a * c)) / 2)) / (a + c)

/-- The main theorem: the ratio of sum of bisectors to sum of sides is at most 4/3 -/
theorem bisector_sum_ratio_bound (t : Triangle) : (t.l_a + t.l_b) / (t.a + t.b) ≤ 4/3 := by
  sorry

/-- The bound 4/3 is tight -/
theorem bisector_sum_ratio_bound_tight : 
  ∀ ε > 0, ∃ t : Triangle, (t.l_a + t.l_b) / (t.a + t.b) > 4/3 - ε := by
  sorry

end NUMINAMATH_CALUDE_bisector_sum_ratio_bound_bisector_sum_ratio_bound_tight_l1514_151404


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1514_151405

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 6 → x ≠ -3 →
  (4 * x - 3) / (x^2 - 3 * x - 18) = (7 / 3) / (x - 6) + (5 / 3) / (x + 3) := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1514_151405


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l1514_151474

/-- Given vectors a, b, and c in ℝ², prove that if a - 2b is perpendicular to c, 
    then the magnitude of b is 3√5. -/
theorem vector_magnitude_problem (a b c : ℝ × ℝ) : 
  a = (-2, 1) → 
  b.1 = k ∧ b.2 = -3 → 
  c = (1, 2) → 
  (a.1 - 2 * b.1, a.2 - 2 * b.2) • c = 0 → 
  Real.sqrt (b.1^2 + b.2^2) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l1514_151474


namespace NUMINAMATH_CALUDE_fraction_addition_l1514_151435

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1514_151435


namespace NUMINAMATH_CALUDE_C_younger_than_A_l1514_151483

-- Define variables for ages
variable (A B C : ℕ)

-- Define the condition from the problem
def age_condition (A B C : ℕ) : Prop := A + B = B + C + 12

-- Theorem to prove
theorem C_younger_than_A (h : age_condition A B C) : A = C + 12 := by
  sorry

end NUMINAMATH_CALUDE_C_younger_than_A_l1514_151483


namespace NUMINAMATH_CALUDE_min_socks_for_pair_l1514_151488

theorem min_socks_for_pair (n : ℕ) (h : n = 2019) : ∃ m : ℕ, m = n + 1 ∧ 
  (∀ k : ℕ, k < m → ∃ f : Fin k → Fin n, Function.Injective f) ∧
  (∀ g : Fin m → Fin n, ¬Function.Injective g) :=
by
  sorry

end NUMINAMATH_CALUDE_min_socks_for_pair_l1514_151488


namespace NUMINAMATH_CALUDE_bakery_storage_l1514_151479

theorem bakery_storage (sugar flour baking_soda : ℕ) : 
  sugar = flour ∧ 
  flour = 10 * baking_soda ∧ 
  flour = 8 * (baking_soda + 60) → 
  sugar = 2400 := by
sorry

end NUMINAMATH_CALUDE_bakery_storage_l1514_151479


namespace NUMINAMATH_CALUDE_largest_fraction_of_consecutive_evens_l1514_151486

theorem largest_fraction_of_consecutive_evens (a b c d : ℕ) : 
  2 < a → a < b → b < c → c < d → 
  Even a → Even b → Even c → Even d →
  (b = a + 2) → (c = b + 2) → (d = c + 2) →
  (c + d) / (b + a) > (b + c) / (a + d) ∧
  (c + d) / (b + a) > (a + d) / (c + b) ∧
  (c + d) / (b + a) > (a + c) / (b + d) ∧
  (c + d) / (b + a) > (b + d) / (c + a) := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_of_consecutive_evens_l1514_151486


namespace NUMINAMATH_CALUDE_sum_of_ten_consecutive_squares_not_perfect_square_l1514_151482

theorem sum_of_ten_consecutive_squares_not_perfect_square (n : ℕ) (h : n > 4) :
  ¬ ∃ m : ℕ, 10 * n^2 + 10 * n + 85 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ten_consecutive_squares_not_perfect_square_l1514_151482


namespace NUMINAMATH_CALUDE_other_number_proof_l1514_151473

theorem other_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 14) → 
  (Nat.lcm a b = 396) → 
  (a = 36) → 
  (b = 154) := by
sorry

end NUMINAMATH_CALUDE_other_number_proof_l1514_151473


namespace NUMINAMATH_CALUDE_parabola_point_ordinate_l1514_151460

theorem parabola_point_ordinate (x y : ℝ) : 
  y^2 = 8*x →                  -- Point M(x, y) is on the parabola y^2 = 8x
  (x - 2)^2 + y^2 = 4^2 →      -- Distance from M to focus (2, 0) is 4
  y = 4 ∨ y = -4 :=            -- The ordinate of M is either 4 or -4
by sorry

end NUMINAMATH_CALUDE_parabola_point_ordinate_l1514_151460


namespace NUMINAMATH_CALUDE_alicia_final_collection_l1514_151469

def egyptian_mask_collection (initial : ℕ) : ℕ :=
  let after_guggenheim := initial - 51
  let after_metropolitan := after_guggenheim - (after_guggenheim / 3)
  let after_louvre := after_metropolitan - (after_metropolitan / 4)
  let after_damage := after_louvre - 30
  let after_british := after_damage - (after_damage * 2 / 5)
  after_british - (after_british / 8)

theorem alicia_final_collection :
  egyptian_mask_collection 600 = 129 := by sorry

end NUMINAMATH_CALUDE_alicia_final_collection_l1514_151469


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l1514_151475

theorem smallest_four_digit_divisible_by_35 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 → n ≥ 1006 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l1514_151475


namespace NUMINAMATH_CALUDE_max_k_value_l1514_151493

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 5 = k^2 * (x^2/y^2 + y^2/x^2) + k * (x/y + y/x)) :
  k ≤ (-1 + Real.sqrt 17) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l1514_151493


namespace NUMINAMATH_CALUDE_carrot_usage_l1514_151449

theorem carrot_usage (total_carrots : ℕ) (unused_carrots : ℕ) 
  (h1 : total_carrots = 300)
  (h2 : unused_carrots = 72) : 
  ∃ (x : ℚ), 
    x * total_carrots + (3/5 : ℚ) * (total_carrots - x * total_carrots) = total_carrots - unused_carrots ∧ 
    x = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_carrot_usage_l1514_151449


namespace NUMINAMATH_CALUDE_max_area_rectangular_garden_l1514_151426

/-- The maximum area of a rectangular garden enclosed by a fence of length 36m is 81 m² -/
theorem max_area_rectangular_garden : 
  ∀ x y : ℝ, x > 0 → y > 0 → 2*(x + y) = 36 → x*y ≤ 81 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*(x + y) = 36 ∧ x*y = 81 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangular_garden_l1514_151426


namespace NUMINAMATH_CALUDE_tangent_line_at_2_f_greater_than_2x_minus_ln_l1514_151436

noncomputable section

def f (x : ℝ) : ℝ := Real.exp x / x

/-- The equation of the tangent line to y = f(x) at x = 2 is e^2x - 4y = 0 -/
theorem tangent_line_at_2 :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    y = m * (x - 2) + f 2 ↔ Real.exp 2 * x - 4 * y = 0 :=
sorry

/-- For all x > 0, f(x) > 2(x - ln x) -/
theorem f_greater_than_2x_minus_ln :
  ∀ x > 0, f x > 2 * (x - Real.log x) :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_at_2_f_greater_than_2x_minus_ln_l1514_151436


namespace NUMINAMATH_CALUDE_total_profit_is_36000_l1514_151478

/-- Represents the profit sharing problem of Tom and Jose's shop -/
def ProfitSharing (tom_investment : ℕ) (tom_months : ℕ) (jose_investment : ℕ) (jose_months : ℕ) (jose_profit : ℕ) : Prop :=
  let tom_total_investment := tom_investment * tom_months
  let jose_total_investment := jose_investment * jose_months
  let total_investment := tom_total_investment + jose_total_investment
  let profit_ratio := tom_total_investment / jose_total_investment
  let tom_profit := (profit_ratio * jose_profit) / (profit_ratio + 1)
  let total_profit := tom_profit + jose_profit
  total_profit = 36000

/-- The main theorem stating that given the investments and Jose's profit, the total profit is 36000 -/
theorem total_profit_is_36000 :
  ProfitSharing 30000 12 45000 10 20000 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_36000_l1514_151478


namespace NUMINAMATH_CALUDE_range_of_a_l1514_151443

/-- Given propositions p and q, and the condition that ¬p is a sufficient but not necessary condition for ¬q, prove that the range of real number a is [-1, 2]. -/
theorem range_of_a (a : ℝ) : 
  (∀ x, (x^2 - (2*a+4)*x + a^2 + 4*a < 0) ↔ (a < x ∧ x < a+4)) →
  (∀ x, ((x-2)*(x-3) < 0) ↔ (2 < x ∧ x < 3)) →
  (∀ x, ¬(a < x ∧ x < a+4) → ¬(2 < x ∧ x < 3)) →
  (∃ x, (2 < x ∧ x < 3) ∧ ¬(a < x ∧ x < a+4)) →
  -1 ≤ a ∧ a ≤ 2 :=
by sorry


end NUMINAMATH_CALUDE_range_of_a_l1514_151443


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l1514_151462

/-- The y-intercept of the tangent line to f(x) = ax - ln x at x = 1 is 1 -/
theorem tangent_line_y_intercept (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x - Real.log x
  let f' : ℝ → ℝ := λ x ↦ a - 1 / x
  let tangent_slope : ℝ := f' 1
  let tangent_point : ℝ × ℝ := (1, f 1)
  let tangent_line : ℝ → ℝ := λ x ↦ tangent_slope * (x - tangent_point.1) + tangent_point.2
  tangent_line 0 = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l1514_151462


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l1514_151495

theorem difference_of_squares_special_case : (527 : ℤ) * 527 - 526 * 528 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l1514_151495


namespace NUMINAMATH_CALUDE_certain_number_proof_l1514_151448

theorem certain_number_proof (x : ℤ) : x - 82 = 17 → x = 99 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1514_151448


namespace NUMINAMATH_CALUDE_intersection_M_N_l1514_151415

def M : Set ℕ := {1, 2, 3, 5, 7}

def N : Set ℕ := {x | ∃ k ∈ M, x = 2 * k - 1}

theorem intersection_M_N : M ∩ N = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1514_151415


namespace NUMINAMATH_CALUDE_inequality_proof_l1514_151464

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  1 / (x^3 * y) + 1 / (y^3 * z) + 1 / (z^3 * x) ≥ x * y + y * z + z * x := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1514_151464


namespace NUMINAMATH_CALUDE_no_valid_coloring_l1514_151424

def Color := Fin 3

theorem no_valid_coloring :
  ¬∃ f : ℕ+ → Color,
    (∀ c : Color, ∃ n : ℕ+, f n = c) ∧
    (∀ a b : ℕ+, f a ≠ f b → f (a * b) ≠ f a ∧ f (a * b) ≠ f b) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_coloring_l1514_151424


namespace NUMINAMATH_CALUDE_school_time_problem_l1514_151430

/-- Given a boy who reaches school 6 minutes early when walking at 7/6 of his usual rate,
    his usual time to reach the school is 42 minutes. -/
theorem school_time_problem (usual_time : ℝ) (usual_rate : ℝ) : 
  (usual_rate / usual_time = (7/6 * usual_rate) / (usual_time - 6)) → 
  usual_time = 42 := by
  sorry

end NUMINAMATH_CALUDE_school_time_problem_l1514_151430


namespace NUMINAMATH_CALUDE_numbers_not_sum_of_two_elements_l1514_151476

def A : Finset ℕ := {1, 2, 3, 5, 8, 13, 21, 34, 55}

def range_start : ℕ := 3
def range_end : ℕ := 89

def sums_of_two_elements (S : Finset ℕ) : Finset ℕ :=
  (S.product S).image (λ (x : ℕ × ℕ) => x.1 + x.2)

def numbers_in_range : Finset ℕ :=
  Finset.Icc range_start range_end

theorem numbers_not_sum_of_two_elements : 
  (numbers_in_range.card - (numbers_in_range ∩ sums_of_two_elements A).card) = 51 := by
  sorry

end NUMINAMATH_CALUDE_numbers_not_sum_of_two_elements_l1514_151476


namespace NUMINAMATH_CALUDE_ratio_characterization_l1514_151434

/-- Given points A, B, and M on a line, where M ≠ B, this theorem characterizes the position of M based on the ratio AM:BM -/
theorem ratio_characterization (A B M M1 M2 : ℝ) : 
  (M ≠ B) →
  (A < B) →
  (A < M1) → (M1 < B) →
  (A < M2) → (B < M2) →
  (A - M1 = 2 * (M1 - B)) →
  (M2 - A = 2 * (B - A)) →
  (((M - A) > 2 * (B - M) ↔ (M1 < M ∧ M < M2 ∧ M ≠ B)) ∧
   ((M - A) < 2 * (B - M) ↔ (M < M1 ∨ M2 < M))) :=
by sorry

end NUMINAMATH_CALUDE_ratio_characterization_l1514_151434


namespace NUMINAMATH_CALUDE_discount_calculation_l1514_151481

/-- Given a cost price, prove that if the marked price is 150% of the cost price
    and the selling price results in a 1% loss on the cost price, then the discount
    (difference between marked price and selling price) is 51% of the cost price. -/
theorem discount_calculation (CP : ℝ) (CP_pos : CP > 0) : 
  let MP := 1.5 * CP
  let SP := 0.99 * CP
  MP - SP = 0.51 * CP := by sorry

end NUMINAMATH_CALUDE_discount_calculation_l1514_151481


namespace NUMINAMATH_CALUDE_smaller_two_digit_number_l1514_151428

theorem smaller_two_digit_number 
  (x y : ℕ) 
  (h1 : x < y) 
  (h2 : x ≥ 10 ∧ x < 100) 
  (h3 : y ≥ 10 ∧ y < 100) 
  (h4 : x + y = 88) 
  (h5 : (100 * y + x) - (100 * x + y) = 3564) : 
  x = 26 := by
sorry

end NUMINAMATH_CALUDE_smaller_two_digit_number_l1514_151428


namespace NUMINAMATH_CALUDE_ribbon_division_l1514_151453

theorem ribbon_division (total_ribbon : ℚ) (num_boxes : ℕ) : 
  total_ribbon = 5 / 12 → num_boxes = 5 → total_ribbon / num_boxes = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_division_l1514_151453


namespace NUMINAMATH_CALUDE_club_members_count_l1514_151499

/-- The number of female members in the club -/
def female_members : ℕ := 12

/-- The number of male members in the club -/
def male_members : ℕ := female_members / 2

/-- The total number of members in the club -/
def total_members : ℕ := female_members + male_members

/-- Proof that the total number of members in the club is 18 -/
theorem club_members_count : total_members = 18 := by
  sorry

end NUMINAMATH_CALUDE_club_members_count_l1514_151499


namespace NUMINAMATH_CALUDE_max_take_home_pay_l1514_151471

/-- The take-home pay function for income y (in thousand dollars) -/
def P (y : ℝ) : ℝ := -10 * (y - 5)^2 + 1000

/-- The income that yields the greatest take-home pay -/
def max_income : ℝ := 5

theorem max_take_home_pay :
  ∀ y : ℝ, y ≥ 0 → P y ≤ P max_income :=
sorry

end NUMINAMATH_CALUDE_max_take_home_pay_l1514_151471


namespace NUMINAMATH_CALUDE_not_one_zero_pronounced_l1514_151419

def number_of_pronounced_zeros (n : Nat) : Nat :=
  sorry -- Implementation of counting pronounced zeros

theorem not_one_zero_pronounced (n : Nat) (h : n = 83721000) : 
  number_of_pronounced_zeros n ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_not_one_zero_pronounced_l1514_151419


namespace NUMINAMATH_CALUDE_power_multiplication_l1514_151409

theorem power_multiplication (x : ℝ) : x^5 * x^3 = x^8 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1514_151409


namespace NUMINAMATH_CALUDE_robins_gum_packages_robins_gum_packages_solution_l1514_151461

theorem robins_gum_packages (candy_packages : ℕ) (pieces_per_package : ℕ) (additional_pieces : ℕ) : ℕ :=
  let total_pieces := candy_packages * pieces_per_package + additional_pieces
  total_pieces / pieces_per_package

theorem robins_gum_packages_solution :
  robins_gum_packages 14 6 7 = 15 := by sorry

end NUMINAMATH_CALUDE_robins_gum_packages_robins_gum_packages_solution_l1514_151461


namespace NUMINAMATH_CALUDE_l_shapes_on_8x8_chessboard_l1514_151455

/-- Represents a chessboard --/
structure Chessboard where
  size : ℕ
  size_pos : size > 0

/-- Represents an L-shaped pattern on a chessboard --/
structure LShape where
  board : Chessboard

/-- Count of L-shaped patterns on a given chessboard --/
def count_l_shapes (board : Chessboard) : ℕ :=
  sorry

theorem l_shapes_on_8x8_chessboard :
  ∃ (board : Chessboard), board.size = 8 ∧ count_l_shapes board = 196 :=
sorry

end NUMINAMATH_CALUDE_l_shapes_on_8x8_chessboard_l1514_151455


namespace NUMINAMATH_CALUDE_clock_sale_second_price_l1514_151410

/-- Represents the sale and resale of a clock in a shop. -/
def ClockSale (original_cost : ℝ) : Prop :=
  let first_sale_price := 1.2 * original_cost
  let buy_back_price := 0.6 * original_cost
  let second_sale_price := 1.08 * original_cost
  (original_cost - buy_back_price = 100) ∧
  (second_sale_price = 270)

/-- Proves that the shop's second selling price of the clock is $270 given the conditions. -/
theorem clock_sale_second_price :
  ∃ (original_cost : ℝ), ClockSale original_cost :=
sorry

end NUMINAMATH_CALUDE_clock_sale_second_price_l1514_151410


namespace NUMINAMATH_CALUDE_square_area_error_l1514_151458

theorem square_area_error (edge : ℝ) (edge_error : ℝ) (area_error : ℝ) : 
  edge_error = 0.02 → 
  area_error = (((1 + edge_error) * edge)^2 - edge^2) / edge^2 * 100 → 
  area_error = 4.04 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l1514_151458


namespace NUMINAMATH_CALUDE_arctan_two_tan_75_minus_three_tan_15_l1514_151467

theorem arctan_two_tan_75_minus_three_tan_15 :
  Real.arctan (2 * Real.tan (75 * π / 180) - 3 * Real.tan (15 * π / 180)) = 30 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_arctan_two_tan_75_minus_three_tan_15_l1514_151467


namespace NUMINAMATH_CALUDE_matrix_sum_theorem_l1514_151459

def matrix_not_invertible (a b c : ℝ) : Prop :=
  ∀ k : ℝ, Matrix.det
    !![a + k, b + k, c + k;
       b + k, c + k, a + k;
       c + k, a + k, b + k] = 0

theorem matrix_sum_theorem (a b c : ℝ) :
  matrix_not_invertible a b c →
  (a / (b + c) + b / (a + c) + c / (a + b) = -3 ∨
   a / (b + c) + b / (a + c) + c / (a + b) = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_matrix_sum_theorem_l1514_151459


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l1514_151491

/-- Represents a 2D point --/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a line in parametric form --/
structure ParametricLine where
  origin : Point
  direction : Point

/-- The first line --/
def line1 : ParametricLine :=
  { origin := { x := 2, y := 3 },
    direction := { x := 3, y := 4 } }

/-- The second line --/
def line2 : ParametricLine :=
  { origin := { x := 6, y := 1 },
    direction := { x := 5, y := -1 } }

/-- The proposed intersection point --/
def intersectionPoint : Point :=
  { x := 20/23, y := 27/23 }

/-- Function to get a point on a parametric line given a parameter --/
def pointOnLine (line : ParametricLine) (t : ℚ) : Point :=
  { x := line.origin.x + t * line.direction.x,
    y := line.origin.y + t * line.direction.y }

/-- Theorem stating that the given point is the unique intersection of the two lines --/
theorem intersection_point_is_unique :
  ∃! t u, pointOnLine line1 t = intersectionPoint ∧ pointOnLine line2 u = intersectionPoint :=
sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l1514_151491


namespace NUMINAMATH_CALUDE_unique_solution_l1514_151408

def equation1 (x y z : ℝ) : Prop := x^2 - 22*y - 69*z + 703 = 0
def equation2 (x y z : ℝ) : Prop := y^2 + 23*x + 23*z - 1473 = 0
def equation3 (x y z : ℝ) : Prop := z^2 - 63*x + 66*y + 2183 = 0

theorem unique_solution :
  ∃! (x y z : ℝ), equation1 x y z ∧ equation2 x y z ∧ equation3 x y z ∧ x = 20 ∧ y = -22 ∧ z = 23 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1514_151408


namespace NUMINAMATH_CALUDE_flowers_after_one_month_l1514_151411

/-- Represents the number of flowers in Mark's garden -/
structure GardenFlowers where
  yellow : ℕ
  purple : ℕ
  green : ℕ
  red : ℕ

/-- Calculates the number of flowers after one month -/
def flowersAfterOneMonth (initial : GardenFlowers) : ℕ :=
  let yellowAfter := initial.yellow + (initial.yellow / 2)
  let purpleAfter := initial.purple * 2
  let greenAfter := initial.green - (initial.green / 5)
  let redAfter := initial.red + (initial.red * 4 / 5)
  yellowAfter + purpleAfter + greenAfter + redAfter

/-- Theorem stating the number of flowers after one month -/
theorem flowers_after_one_month :
  ∃ (initial : GardenFlowers),
    initial.yellow = 10 ∧
    initial.purple = initial.yellow + (initial.yellow * 4 / 5) ∧
    initial.green = (initial.yellow + initial.purple) / 4 ∧
    initial.red = ((initial.yellow + initial.purple + initial.green) * 35) / 100 ∧
    flowersAfterOneMonth initial = 77 :=
  sorry

end NUMINAMATH_CALUDE_flowers_after_one_month_l1514_151411


namespace NUMINAMATH_CALUDE_cubic_root_fraction_equality_l1514_151451

theorem cubic_root_fraction_equality (x : ℝ) (h : x^3 + x - 1 = 0) :
  (x^4 - 2*x^3 + x^2 - 3*x + 5) / (x^5 - x^2 - x + 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_fraction_equality_l1514_151451


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1514_151438

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k^2*x₁^2 - (2*k+1)*x₁ + 1 = 0 ∧ k^2*x₂^2 - (2*k+1)*x₂ + 1 = 0) ↔ 
  (k ≥ -1/4 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1514_151438


namespace NUMINAMATH_CALUDE_intersection_empty_range_l1514_151413

theorem intersection_empty_range (a : ℝ) : 
  let A := {x : ℝ | |x - a| < 1}
  let B := {x : ℝ | 1 < x ∧ x < 5}
  (A ∩ B = ∅) ↔ (a ≤ 0 ∨ a ≥ 6) := by sorry

end NUMINAMATH_CALUDE_intersection_empty_range_l1514_151413


namespace NUMINAMATH_CALUDE_suresh_job_time_l1514_151439

/-- The time it takes Ashutosh to complete the job alone (in hours) -/
def ashutosh_time : ℝ := 15

/-- The time Suresh works on the job (in hours) -/
def suresh_work_time : ℝ := 9

/-- The time Ashutosh works to complete the remaining job (in hours) -/
def ashutosh_remaining_time : ℝ := 6

/-- The time it takes Suresh to complete the job alone (in hours) -/
def suresh_time : ℝ := 15

theorem suresh_job_time :
  suresh_time * (1 / ashutosh_time * ashutosh_remaining_time + 1 / suresh_time * suresh_work_time) = suresh_time := by
  sorry

#check suresh_job_time

end NUMINAMATH_CALUDE_suresh_job_time_l1514_151439


namespace NUMINAMATH_CALUDE_fathers_age_multiple_l1514_151402

theorem fathers_age_multiple (sons_age : ℕ) (multiple : ℕ) : 
  (44 = multiple * sons_age + 4) →
  (44 + 4 = 2 * (sons_age + 4) + 20) →
  multiple = 4 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_multiple_l1514_151402


namespace NUMINAMATH_CALUDE_houses_with_neither_feature_l1514_151480

theorem houses_with_neither_feature (total : ℕ) (garage : ℕ) (pool : ℕ) (both : ℕ) :
  total = 70 →
  garage = 50 →
  pool = 40 →
  both = 35 →
  total - (garage + pool - both) = 15 :=
by sorry

end NUMINAMATH_CALUDE_houses_with_neither_feature_l1514_151480


namespace NUMINAMATH_CALUDE_right_triangle_power_equation_l1514_151446

theorem right_triangle_power_equation (a b c n : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  n > 2 →
  a^2 + b^2 = c^2 →
  (a^n + b^n + c^n)^2 = 2*(a^(2*n) + b^(2*n) + c^(2*n)) →
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_power_equation_l1514_151446


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_50_l1514_151440

theorem largest_four_digit_divisible_by_50 : 
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 50 = 0 → n ≤ 9950 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_50_l1514_151440


namespace NUMINAMATH_CALUDE_product_of_decimals_l1514_151444

theorem product_of_decimals : (0.7 : ℝ) * 0.8 = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_product_of_decimals_l1514_151444


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1514_151465

theorem trigonometric_equation_solution (x : ℝ) :
  (4 * Real.sin (π / 6 + x) * Real.sin (5 * π / 6 + x) / (Real.cos x)^2 + 2 * Real.tan x = 0) ∧ (Real.cos x ≠ 0) →
  (∃ k : ℤ, x = -Real.arctan (1 / 3) + k * π) ∨ (∃ n : ℤ, x = π / 4 + n * π) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1514_151465


namespace NUMINAMATH_CALUDE_notebook_cost_l1514_151472

theorem notebook_cost (notebook_cost pencil_cost : ℝ) 
  (total_cost : notebook_cost + pencil_cost = 3.40)
  (price_difference : notebook_cost = pencil_cost + 2) : 
  notebook_cost = 2.70 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_l1514_151472


namespace NUMINAMATH_CALUDE_joans_games_l1514_151470

theorem joans_games (football_this_year basketball_this_year total_both_years : ℕ)
  (h1 : football_this_year = 4)
  (h2 : basketball_this_year = 3)
  (h3 : total_both_years = 9) :
  total_both_years - (football_this_year + basketball_this_year) = 2 := by
sorry

end NUMINAMATH_CALUDE_joans_games_l1514_151470


namespace NUMINAMATH_CALUDE_point_outside_circle_iff_m_in_range_l1514_151427

/-- A circle in the x-y plane defined by the equation x^2 + y^2 + 2x - m = 0 -/
def Circle (m : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 - m = 0}

/-- The point P with coordinates (1,1) -/
def P : ℝ × ℝ := (1, 1)

/-- Predicate to check if a point is outside a circle -/
def IsOutside (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : Prop :=
  ∀ q ∈ c, (p.1 - q.1)^2 + (p.2 - q.2)^2 > 0

theorem point_outside_circle_iff_m_in_range :
  ∀ m : ℝ, IsOutside P (Circle m) ↔ -1 < m ∧ m < 4 :=
sorry

end NUMINAMATH_CALUDE_point_outside_circle_iff_m_in_range_l1514_151427


namespace NUMINAMATH_CALUDE_fraction_not_on_time_is_one_eighth_l1514_151431

/-- Represents the fraction of attendees who did not arrive on time at a monthly meeting -/
def fraction_not_on_time (total : ℕ) (male : ℕ) (male_on_time : ℕ) (female_on_time : ℕ) : ℚ :=
  1 - (male_on_time + female_on_time : ℚ) / total

/-- Theorem stating the fraction of attendees who did not arrive on time -/
theorem fraction_not_on_time_is_one_eighth
  (total : ℕ) (male : ℕ) (male_on_time : ℕ) (female_on_time : ℕ)
  (h_total_pos : 0 < total)
  (h_male_ratio : male = (3 * total) / 5)
  (h_male_on_time : male_on_time = (7 * male) / 8)
  (h_female_on_time : female_on_time = (9 * (total - male)) / 10) :
  fraction_not_on_time total male male_on_time female_on_time = 1/8 := by
  sorry

#check fraction_not_on_time_is_one_eighth

end NUMINAMATH_CALUDE_fraction_not_on_time_is_one_eighth_l1514_151431


namespace NUMINAMATH_CALUDE_cube_product_three_six_l1514_151412

theorem cube_product_three_six : 3^3 * 6^3 = 5832 := by
  sorry

end NUMINAMATH_CALUDE_cube_product_three_six_l1514_151412


namespace NUMINAMATH_CALUDE_bug_traversal_12_25_l1514_151452

/-- The number of tiles a bug traverses when walking diagonally across a rectangular floor -/
def bugTraversal (width length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

theorem bug_traversal_12_25 :
  bugTraversal 12 25 = 36 := by
  sorry

end NUMINAMATH_CALUDE_bug_traversal_12_25_l1514_151452


namespace NUMINAMATH_CALUDE_paco_cookies_eaten_l1514_151457

/-- Given that Paco had 17 cookies, gave 13 to his friend, and ate 1 more than he gave away,
    prove that Paco ate 14 cookies. -/
theorem paco_cookies_eaten (initial : ℕ) (given : ℕ) (eaten : ℕ) 
    (h1 : initial = 17)
    (h2 : given = 13)
    (h3 : eaten = given + 1) : 
  eaten = 14 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_eaten_l1514_151457


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1514_151445

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (-2, -3),
    prove that the line L2 with equation y = -2x - 7 is perpendicular to L1
    and passes through P. -/
theorem perpendicular_line_through_point 
  (L1 : Real → Real → Prop) 
  (P : Real × Real) 
  (L2 : Real → Real → Prop) : 
  (∀ x y, L1 x y ↔ 3 * x - 6 * y = 9) →
  P = (-2, -3) →
  (∀ x y, L2 x y ↔ y = -2 * x - 7) →
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    (x₂ - x₁) * (P.1 - x₁) + (y₂ - y₁) * (P.2 - y₁) = 0) →
  L2 P.1 P.2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1514_151445


namespace NUMINAMATH_CALUDE_tip_calculation_correct_l1514_151441

/-- Calculates the tip for a restaurant check with given conditions. -/
def calculate_tip (check_amount : ℚ) (tax_rate : ℚ) (senior_discount : ℚ) (dine_in_surcharge : ℚ) (payment : ℚ) : ℚ :=
  let total_with_tax := check_amount * (1 + tax_rate)
  let discount_amount := check_amount * senior_discount
  let surcharge_amount := check_amount * dine_in_surcharge
  let final_total := total_with_tax - discount_amount + surcharge_amount
  payment - final_total

/-- Theorem stating that the tip calculation for the given conditions results in $2.75. -/
theorem tip_calculation_correct :
  calculate_tip 15 (20/100) (10/100) (5/100) 20 = 275/100 := by
  sorry

end NUMINAMATH_CALUDE_tip_calculation_correct_l1514_151441


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l1514_151490

/-- Calculate the simple interest rate given the principal and annual interest -/
theorem simple_interest_rate_calculation
  (principal : ℝ) 
  (annual_interest : ℝ) 
  (h1 : principal = 9000)
  (h2 : annual_interest = 810) :
  (annual_interest / principal) * 100 = 9 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l1514_151490


namespace NUMINAMATH_CALUDE_room_population_l1514_151400

theorem room_population (P M : ℕ) : 
  (P : ℚ) * (2 / 100) = 1 →  -- 2% of painters are musicians
  (M : ℚ) * (5 / 100) = 1 →  -- 5% of musicians are painters
  P + M - 1 = 69             -- Total people in the room
  := by sorry

end NUMINAMATH_CALUDE_room_population_l1514_151400


namespace NUMINAMATH_CALUDE_dice_probability_l1514_151432

def red_die : Finset Nat := {4, 6}
def yellow_die : Finset Nat := {1, 2, 3, 4, 5, 6}

def total_outcomes : Finset (Nat × Nat) :=
  red_die.product yellow_die

def favorable_outcomes : Finset (Nat × Nat) :=
  total_outcomes.filter (fun p => p.1 * p.2 > 20)

theorem dice_probability :
  (favorable_outcomes.card : ℚ) / total_outcomes.card = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l1514_151432


namespace NUMINAMATH_CALUDE_largest_number_in_sample_l1514_151422

/-- Represents a systematic sampling process -/
structure SystematicSample where
  population_size : ℕ
  start : ℕ
  interval : ℕ

/-- Calculates the largest number in a systematic sample -/
def largest_sample_number (s : SystematicSample) : ℕ :=
  s.start + s.interval * ((s.population_size - s.start) / s.interval)

/-- Theorem: The largest number in the given systematic sample is 1468 -/
theorem largest_number_in_sample :
  let s : SystematicSample := ⟨1500, 18, 50⟩
  largest_sample_number s = 1468 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_in_sample_l1514_151422


namespace NUMINAMATH_CALUDE_equation_solution_l1514_151447

theorem equation_solution (n : ℤ) : n + (n + 1) + (n + 2) = 15 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1514_151447


namespace NUMINAMATH_CALUDE_circle_equations_correct_l1514_151494

-- Define the points A, B, and D
def A : ℝ × ℝ := (-1, 5)
def B : ℝ × ℝ := (5, 5)
def D : ℝ × ℝ := (5, -1)

-- Define circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 18

-- Define circle M
def circle_M (x y : ℝ) : Prop :=
  (x - 6)^2 + (y - 6)^2 = 2

-- Theorem statement
theorem circle_equations_correct :
  (circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ circle_C D.1 D.2) ∧
  (∃ (t : ℝ), circle_C (B.1 + t) (B.2 + t) ∧ circle_M (B.1 + t) (B.2 + t)) ∧
  (∀ (x y : ℝ), circle_M x y → (x - B.1)^2 + (y - B.2)^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equations_correct_l1514_151494


namespace NUMINAMATH_CALUDE_exact_one_common_point_chord_length_when_m_4_l1514_151487

-- Define the curve C
def curve_C (t : ℝ) : ℝ × ℝ := (4 * t^2, 4 * t)

-- Define the line l in polar form
def line_l (m : ℝ) (ρ θ : ℝ) : Prop := ρ * (4 * Real.cos θ + 3 * Real.sin θ) - m = 0

-- Theorem 1: Value of m for exactly one common point
theorem exact_one_common_point :
  ∃ (m : ℝ), m = -9/4 ∧
  (∃! (t : ℝ), ∃ (ρ θ : ℝ), curve_C t = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ line_l m ρ θ) :=
sorry

-- Theorem 2: Length of chord when m = 4
theorem chord_length_when_m_4 :
  let m := 4
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧
  (∃ (ρ₁ θ₁ ρ₂ θ₂ : ℝ), 
    curve_C t₁ = (ρ₁ * Real.cos θ₁, ρ₁ * Real.sin θ₁) ∧ 
    curve_C t₂ = (ρ₂ * Real.cos θ₂, ρ₂ * Real.sin θ₂) ∧
    line_l m ρ₁ θ₁ ∧ line_l m ρ₂ θ₂) ∧
  let (x₁, y₁) := curve_C t₁
  let (x₂, y₂) := curve_C t₂
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 25/4 :=
sorry

end NUMINAMATH_CALUDE_exact_one_common_point_chord_length_when_m_4_l1514_151487


namespace NUMINAMATH_CALUDE_conference_handshakes_l1514_151406

/-- The number of handshakes in a conference with multiple companies --/
def num_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_people := num_companies * reps_per_company
  let handshakes_per_person := total_people - reps_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem stating that the number of handshakes for the given scenario is 75 --/
theorem conference_handshakes :
  num_handshakes 3 5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1514_151406


namespace NUMINAMATH_CALUDE_unique_solution_for_prime_equation_l1514_151423

theorem unique_solution_for_prime_equation (p q r t n : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  p^2 + q*t = (p + t)^n → 
  p^2 + q*r = t^4 → 
  (p = 2 ∧ q = 7 ∧ r = 11 ∧ t = 3 ∧ n = 2) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_for_prime_equation_l1514_151423


namespace NUMINAMATH_CALUDE_max_value_implies_a_l1514_151442

/-- The function f(x) = -x^2 + 2ax + 1 - a has a maximum value of 2 in the interval [0, 1] -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 1 - a

/-- The maximum value of f(x) in the interval [0, 1] is 2 -/
def max_value (a : ℝ) : Prop := ∀ x, x ∈ Set.Icc 0 1 → f a x ≤ 2

/-- The theorem stating that if f(x) has a maximum value of 2 in [0, 1], then a = -1 or a = 2 -/
theorem max_value_implies_a (a : ℝ) : max_value a → (a = -1 ∨ a = 2) := by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l1514_151442


namespace NUMINAMATH_CALUDE_correct_observation_value_l1514_151420

theorem correct_observation_value (n : ℕ) (original_mean corrected_mean wrong_value : ℚ) 
  (h1 : n = 50)
  (h2 : original_mean = 41)
  (h3 : corrected_mean = 41.5)
  (h4 : wrong_value = 23) :
  let original_sum := n * original_mean
  let correct_sum := n * corrected_mean
  let correct_value := correct_sum - (original_sum - wrong_value)
  correct_value = 48 := by sorry

end NUMINAMATH_CALUDE_correct_observation_value_l1514_151420


namespace NUMINAMATH_CALUDE_f_inequality_implies_a_range_l1514_151417

open Set Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (log x + 2) / x + a * (x - 1) - 2

def domain : Set ℝ := {x | x ∈ (Set.Ioo 0 1) ∪ (Set.Ioi 1)}

theorem f_inequality_implies_a_range (a : ℝ) :
  (∀ x ∈ domain, (f a x) / (1 - x) < a / x) → a ≥ (1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_implies_a_range_l1514_151417


namespace NUMINAMATH_CALUDE_max_height_triangle_def_l1514_151403

/-- Triangle DEF with sides a, b, c -/
structure Triangle (a b c : ℝ) where
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The maximum possible height of a table constructed from a triangle -/
def max_table_height (t : Triangle a b c) : ℝ :=
  sorry

theorem max_height_triangle_def (t : Triangle 20 29 35) :
  max_table_height t = 84 * Real.sqrt 2002 / 64 := by
  sorry

end NUMINAMATH_CALUDE_max_height_triangle_def_l1514_151403


namespace NUMINAMATH_CALUDE_min_m_value_l1514_151421

/-- The function f(x) = x^2 - 3x --/
def f (x : ℝ) : ℝ := x^2 - 3*x

/-- The interval [-3, 2] --/
def I : Set ℝ := Set.Icc (-3) 2

/-- The theorem statement --/
theorem min_m_value :
  ∃ (m : ℝ), m = 81/4 ∧ 
  (∀ (x₁ x₂ : ℝ), x₁ ∈ I → x₂ ∈ I → |f x₁ - f x₂| ≤ m) ∧
  (∀ (m' : ℝ), (∀ (x₁ x₂ : ℝ), x₁ ∈ I → x₂ ∈ I → |f x₁ - f x₂| ≤ m') → m ≤ m') :=
sorry

end NUMINAMATH_CALUDE_min_m_value_l1514_151421


namespace NUMINAMATH_CALUDE_trigonometric_simplification_logarithmic_simplification_l1514_151497

theorem trigonometric_simplification (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin (θ + π/2) * Real.cos (π/2 - θ) - Real.cos (π - θ)^2) / (1 + Real.sin θ^2) = 1/3 := by
  sorry

theorem logarithmic_simplification (x : Real) :
  Real.log (Real.sqrt (x^2 + 1) + x) + Real.log (Real.sqrt (x^2 + 1) - x) +
  (Real.log 2 / Real.log 10)^2 + (1 + Real.log 2 / Real.log 10) * (Real.log 5 / Real.log 10) -
  2 * Real.sin (30 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_logarithmic_simplification_l1514_151497


namespace NUMINAMATH_CALUDE_negation_equivalence_l1514_151407

theorem negation_equivalence : 
  (¬ ∀ x : ℝ, x^2 + 3*x + 2 < 0) ↔ (∀ x : ℝ, x^2 + 3*x + 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1514_151407


namespace NUMINAMATH_CALUDE_certain_number_is_six_l1514_151414

theorem certain_number_is_six : ∃ x : ℝ, (7 * x - 6 - 12 = 4 * x) ∧ (x = 6) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_six_l1514_151414


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l1514_151437

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 133 → 
  (∀ a b : ℕ, a^2 - b^2 = 133 → x^2 + y^2 ≤ a^2 + b^2) → 
  x^2 + y^2 = 205 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l1514_151437


namespace NUMINAMATH_CALUDE_money_division_l1514_151489

theorem money_division (a b c : ℕ) (h1 : a = b / 2) (h2 : b = c / 2) (h3 : c = 400) :
  a + b + c = 700 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l1514_151489


namespace NUMINAMATH_CALUDE_divisibility_17_and_289_l1514_151429

theorem divisibility_17_and_289 (n : ℤ) :
  (∃ k : ℤ, n^2 - n - 4 = 17 * k) ↔ (∃ m : ℤ, n = 17 * m - 8) ∧
  ¬(∃ l : ℤ, n^2 - n - 4 = 289 * l) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_17_and_289_l1514_151429


namespace NUMINAMATH_CALUDE_shoe_repair_cost_l1514_151401

theorem shoe_repair_cost (new_shoe_cost : ℝ) (new_shoe_lifespan : ℝ) (repaired_shoe_lifespan : ℝ) (cost_difference_percentage : ℝ) :
  new_shoe_cost = 30 →
  new_shoe_lifespan = 2 →
  repaired_shoe_lifespan = 1 →
  cost_difference_percentage = 42.857142857142854 →
  ∃ repair_cost : ℝ,
    repair_cost = 10.5 ∧
    (new_shoe_cost / new_shoe_lifespan) = repair_cost * (1 + cost_difference_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_shoe_repair_cost_l1514_151401


namespace NUMINAMATH_CALUDE_composite_s_l1514_151468

theorem composite_s (s : ℕ) (h1 : s ≥ 4) :
  (∃ a b c d : ℕ+, (a:ℕ) + b + c + d = s ∧ 
    (s ∣ a * b * c + a * b * d + a * c * d + b * c * d)) →
  ¬(Nat.Prime s) :=
by sorry

end NUMINAMATH_CALUDE_composite_s_l1514_151468
