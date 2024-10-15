import Mathlib

namespace NUMINAMATH_CALUDE_percentage_calculation_l2954_295471

theorem percentage_calculation : 
  (2 * (1/4 * (4/100))) + (3 * (15/100)) - (1/2 * (10/100)) = 0.42 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2954_295471


namespace NUMINAMATH_CALUDE_min_coach_handshakes_l2954_295463

def total_handshakes (na nb : ℕ) : ℕ :=
  (na + nb) * (na + nb - 1) / 2 + na + nb

def is_valid_configuration (na nb : ℕ) : Prop :=
  na < nb ∧ total_handshakes na nb = 465

theorem min_coach_handshakes :
  ∃ (na nb : ℕ), is_valid_configuration na nb ∧
  ∀ (ma mb : ℕ), is_valid_configuration ma mb → na ≤ ma :=
by sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_l2954_295463


namespace NUMINAMATH_CALUDE_power_of_seven_mod_hundred_l2954_295491

theorem power_of_seven_mod_hundred : ∃ (n : ℕ), n > 0 ∧ 7^n % 100 = 1 ∧ ∀ (k : ℕ), 0 < k → k < n → 7^k % 100 ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_hundred_l2954_295491


namespace NUMINAMATH_CALUDE_arcade_tickets_l2954_295416

theorem arcade_tickets (initial_tickets spent_tickets additional_tickets : ℕ) :
  initial_tickets ≥ spent_tickets →
  (initial_tickets - spent_tickets + additional_tickets) = 
    initial_tickets - spent_tickets + additional_tickets :=
by
  sorry

end NUMINAMATH_CALUDE_arcade_tickets_l2954_295416


namespace NUMINAMATH_CALUDE_evaluate_expression_l2954_295427

theorem evaluate_expression : (800^2 : ℚ) / (300^2 - 296^2) = 640000 / 2384 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2954_295427


namespace NUMINAMATH_CALUDE_power_of_fraction_three_fourths_five_l2954_295423

theorem power_of_fraction_three_fourths_five :
  (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by sorry

end NUMINAMATH_CALUDE_power_of_fraction_three_fourths_five_l2954_295423


namespace NUMINAMATH_CALUDE_klinker_age_proof_l2954_295446

/-- Mr. Klinker's current age -/
def klinker_age : ℕ := 35

/-- Mr. Klinker's daughter's current age -/
def daughter_age : ℕ := 10

/-- Years into the future when the age relation holds -/
def years_future : ℕ := 15

theorem klinker_age_proof :
  klinker_age = 35 ∧
  daughter_age = 10 ∧
  klinker_age + years_future = 2 * (daughter_age + years_future) := by
  sorry

#check klinker_age_proof

end NUMINAMATH_CALUDE_klinker_age_proof_l2954_295446


namespace NUMINAMATH_CALUDE_stock_purchase_problem_l2954_295495

/-- Mr. Wise's stock purchase problem -/
theorem stock_purchase_problem (total_value : ℝ) (price_type1 : ℝ) (total_shares : ℕ) (shares_type1 : ℕ) :
  total_value = 1950 →
  price_type1 = 3 →
  total_shares = 450 →
  shares_type1 = 400 →
  ∃ (price_type2 : ℝ),
    price_type2 * (total_shares - shares_type1) + price_type1 * shares_type1 = total_value ∧
    price_type2 = 15 :=
by sorry

end NUMINAMATH_CALUDE_stock_purchase_problem_l2954_295495


namespace NUMINAMATH_CALUDE_matrix_power_sum_l2954_295452

def A (a : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![1, 3, a],
    ![0, 1, 5],
    ![0, 0, 1]]

theorem matrix_power_sum (a : ℝ) (n : ℕ) :
  (A a)^n = ![![1, 27, 2883],
              ![0,  1,   45],
              ![0,  0,    1]] →
  a + n = 264 := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_sum_l2954_295452


namespace NUMINAMATH_CALUDE_train_crossing_time_l2954_295459

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) :
  train_length = 50 ∧ train_speed_kmh = 60 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2954_295459


namespace NUMINAMATH_CALUDE_jame_annual_earnings_difference_l2954_295448

/-- Calculates the difference in annual earnings between Jame's new job and old job -/
def annual_earnings_difference (
  new_hourly_rate : ℕ) 
  (new_weekly_hours : ℕ)
  (old_hourly_rate : ℕ)
  (old_weekly_hours : ℕ)
  (weeks_per_year : ℕ) : ℕ :=
  ((new_hourly_rate * new_weekly_hours) - (old_hourly_rate * old_weekly_hours)) * weeks_per_year

/-- Proves that the difference in annual earnings between Jame's new job and old job is $20,800 -/
theorem jame_annual_earnings_difference :
  annual_earnings_difference 20 40 16 25 52 = 20800 := by
  sorry

end NUMINAMATH_CALUDE_jame_annual_earnings_difference_l2954_295448


namespace NUMINAMATH_CALUDE_distinct_triangles_count_l2954_295430

/-- Represents a 2x4 grid of points -/
def Grid := Fin 2 × Fin 4

/-- Represents a triangle formed by three points on the grid -/
def Triangle := Fin 3 → Grid

/-- Checks if three points are collinear -/
def collinear (p q r : Grid) : Prop := sorry

/-- Counts the number of distinct triangles in a 2x4 grid -/
def count_distinct_triangles : ℕ := sorry

/-- Theorem stating that the number of distinct triangles in a 2x4 grid is 44 -/
theorem distinct_triangles_count : count_distinct_triangles = 44 := by sorry

end NUMINAMATH_CALUDE_distinct_triangles_count_l2954_295430


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l2954_295499

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 6 * Real.sin β - 10)^2 + (3 * Real.sin α + 6 * Real.cos β - 18)^2 ≥ 121 ∧
  ∃ α₀ β₀ : ℝ, (3 * Real.cos α₀ + 6 * Real.sin β₀ - 10)^2 + (3 * Real.sin α₀ + 6 * Real.cos β₀ - 18)^2 = 121 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l2954_295499


namespace NUMINAMATH_CALUDE_egg_problem_solution_l2954_295408

/-- Represents the number of eggs of each type --/
structure EggCounts where
  newLaid : ℕ
  fresh : ℕ
  ordinary : ℕ

/-- Checks if the given egg counts satisfy all problem constraints --/
def satisfiesConstraints (counts : EggCounts) : Prop :=
  counts.newLaid + counts.fresh + counts.ordinary = 100 ∧
  5 * counts.newLaid + counts.fresh + (counts.ordinary / 2) = 100 ∧
  (counts.newLaid = counts.fresh ∨ counts.newLaid = counts.ordinary ∨ counts.fresh = counts.ordinary)

/-- The unique solution to the egg problem --/
def eggSolution : EggCounts :=
  { newLaid := 10, fresh := 10, ordinary := 80 }

/-- Theorem stating that the egg solution is unique and satisfies all constraints --/
theorem egg_problem_solution :
  satisfiesConstraints eggSolution ∧
  ∀ counts : EggCounts, satisfiesConstraints counts → counts = eggSolution := by
  sorry


end NUMINAMATH_CALUDE_egg_problem_solution_l2954_295408


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l2954_295415

theorem rectangle_area_diagonal (d : ℝ) (h : d > 0) : ∃ (l w : ℝ),
  l > 0 ∧ w > 0 ∧ l / w = 5 / 2 ∧ l ^ 2 + w ^ 2 = d ^ 2 ∧ l * w = (10 / 29) * d ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l2954_295415


namespace NUMINAMATH_CALUDE_cube_root_simplification_l2954_295406

theorem cube_root_simplification : Real.rpow (2^9 * 5^3 * 7^3) (1/3) = 280 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l2954_295406


namespace NUMINAMATH_CALUDE_smallest_n_property_ratio_is_sqrt_three_l2954_295400

/-- The smallest positive integer n for which there exist positive real numbers a and b
    such that (a + bi)^n = -(a - bi)^n -/
def smallest_n : ℕ := 4

theorem smallest_n_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Complex.I : ℂ)^smallest_n * (a + b * Complex.I)^smallest_n = -(a - b * Complex.I)^smallest_n :=
sorry

theorem ratio_is_sqrt_three (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : (Complex.I : ℂ)^smallest_n * (a + b * Complex.I)^smallest_n = -(a - b * Complex.I)^smallest_n) :
  a / b = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_property_ratio_is_sqrt_three_l2954_295400


namespace NUMINAMATH_CALUDE_quadratic_root_implies_c_value_l2954_295432

theorem quadratic_root_implies_c_value (c : ℝ) :
  (∀ x : ℝ, (3/2) * x^2 + 11*x + c = 0 ↔ x = (-11 + Real.sqrt 7) / 3 ∨ x = (-11 - Real.sqrt 7) / 3) →
  c = 19 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_c_value_l2954_295432


namespace NUMINAMATH_CALUDE_jessica_chocolate_bar_cost_l2954_295466

/-- Represents Jessica's purchase --/
structure Purchase where
  total_cost : ℕ
  gummy_bear_packs : ℕ
  chocolate_chip_bags : ℕ
  gummy_bear_cost : ℕ
  chocolate_chip_cost : ℕ

/-- Calculates the cost of chocolate bars in Jessica's purchase --/
def chocolate_bar_cost (p : Purchase) : ℕ :=
  p.total_cost - (p.gummy_bear_packs * p.gummy_bear_cost + p.chocolate_chip_bags * p.chocolate_chip_cost)

/-- Theorem stating that the cost of chocolate bars in Jessica's purchase is $30 --/
theorem jessica_chocolate_bar_cost :
  let p : Purchase := {
    total_cost := 150,
    gummy_bear_packs := 10,
    chocolate_chip_bags := 20,
    gummy_bear_cost := 2,
    chocolate_chip_cost := 5
  }
  chocolate_bar_cost p = 30 := by
  sorry


end NUMINAMATH_CALUDE_jessica_chocolate_bar_cost_l2954_295466


namespace NUMINAMATH_CALUDE_exists_nonnegative_coeff_multiplier_l2954_295493

/-- A polynomial with real coefficients that is positive for all nonnegative real numbers. -/
structure PositivePolynomial where
  P : Polynomial ℝ
  pos : ∀ x : ℝ, x ≥ 0 → P.eval x > 0

/-- The theorem stating that for any positive polynomial, there exists a positive integer n
    such that (1 + x)^n * P(x) has nonnegative coefficients. -/
theorem exists_nonnegative_coeff_multiplier (p : PositivePolynomial) :
  ∃ n : ℕ+, ∀ i : ℕ, ((1 + X : Polynomial ℝ)^(n : ℕ) * p.P).coeff i ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_nonnegative_coeff_multiplier_l2954_295493


namespace NUMINAMATH_CALUDE_max_subjects_per_teacher_l2954_295449

theorem max_subjects_per_teacher 
  (total_subjects : Nat) 
  (min_teachers : Nat) 
  (maths_teachers : Nat) 
  (physics_teachers : Nat) 
  (chemistry_teachers : Nat) 
  (h1 : total_subjects = maths_teachers + physics_teachers + chemistry_teachers)
  (h2 : maths_teachers = 4)
  (h3 : physics_teachers = 3)
  (h4 : chemistry_teachers = 3)
  (h5 : min_teachers = 5)
  : (total_subjects / min_teachers : Nat) = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_subjects_per_teacher_l2954_295449


namespace NUMINAMATH_CALUDE_inequality_solution_and_function_property_l2954_295467

def f (x : ℝ) := |x - 1|

theorem inequality_solution_and_function_property :
  (∃ (S : Set ℝ), S = {x : ℝ | x ≤ -10/3 ∨ x ≥ 2} ∧
    ∀ x, x ∈ S ↔ f (2*x) + f (x + 4) ≥ 8) ∧
  (∀ a b : ℝ, |a| < 1 → |b| < 1 → a ≠ 0 → f (a*b) / |a| > f (b/a)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_and_function_property_l2954_295467


namespace NUMINAMATH_CALUDE_coefficient_sum_l2954_295433

theorem coefficient_sum (b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x, (2*x + 3)^5 = b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 3125 := by
sorry

end NUMINAMATH_CALUDE_coefficient_sum_l2954_295433


namespace NUMINAMATH_CALUDE_min_x_plus_y_l2954_295457

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x + 2

-- State the theorem
theorem min_x_plus_y (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, y ≥ f a (|x|)) →
  (∃ x₀ y₀ : ℝ, y₀ ≥ f a (|x₀|) ∧ x₀ + y₀ = -a - 1/a) ∧
  (∀ x y : ℝ, y ≥ f a (|x|) → x + y ≥ -a - 1/a) :=
by sorry

end NUMINAMATH_CALUDE_min_x_plus_y_l2954_295457


namespace NUMINAMATH_CALUDE_cosine_sine_inequality_l2954_295490

theorem cosine_sine_inequality (x : ℝ) : 
  (1 / 4 : ℝ) ≤ (Real.cos x)^6 + (Real.sin x)^6 ∧ (Real.cos x)^6 + (Real.sin x)^6 ≤ 1 :=
by
  sorry

#check cosine_sine_inequality

end NUMINAMATH_CALUDE_cosine_sine_inequality_l2954_295490


namespace NUMINAMATH_CALUDE_exam_time_on_type_A_l2954_295401

/-- Represents the time spent on type A problems in an exam -/
def time_on_type_A (total_time : ℚ) (total_questions : ℕ) (type_A_questions : ℕ) : ℚ :=
  let type_B_questions := total_questions - type_A_questions
  let time_ratio := (2 * type_A_questions + type_B_questions) / total_questions
  (total_time * time_ratio * 2 * type_A_questions) / (2 * type_A_questions + type_B_questions)

/-- Theorem stating the time spent on type A problems in the given exam conditions -/
theorem exam_time_on_type_A :
  time_on_type_A (5/2) 200 10 = 100/7 :=
by sorry

end NUMINAMATH_CALUDE_exam_time_on_type_A_l2954_295401


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2954_295442

theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → 2*a + 2*b = 2 → (1/a + 1/b ≥ 4) ∧ (∃ a b, 1/a + 1/b = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2954_295442


namespace NUMINAMATH_CALUDE_egg_weight_probability_l2954_295434

theorem egg_weight_probability (p_less_than_30 p_30_to_40 : ℝ) 
  (h1 : p_less_than_30 = 0.30)
  (h2 : p_30_to_40 = 0.50) :
  1 - p_less_than_30 = 0.70 :=
by sorry

end NUMINAMATH_CALUDE_egg_weight_probability_l2954_295434


namespace NUMINAMATH_CALUDE_rhombus_area_l2954_295479

/-- The area of a rhombus with specific properties -/
theorem rhombus_area (s : ℝ) (d₁ d₂ : ℝ) (h_side : s = Real.sqrt 130) 
  (h_diag_diff : d₂ = d₁ + 4) (h_perp : d₁ * d₂ = 4 * s^2) : d₁ * d₂ / 2 = 126 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l2954_295479


namespace NUMINAMATH_CALUDE_gh_length_is_60_over_77_l2954_295478

/-- Represents a right triangle with squares inscribed -/
structure RightTriangleWithSquares where
  -- Right triangle ABC
  AC : ℝ
  BC : ℝ
  -- Square DEFG
  DE : ℝ
  -- Square GHIJ
  GH : ℝ
  -- Condition that E lies on AC and I lies on BC
  E_on_AC : ℝ
  I_on_BC : ℝ
  -- J is the midpoint of DG
  DJ : ℝ

/-- The length of GH in the inscribed square configuration -/
def ghLength (t : RightTriangleWithSquares) : ℝ := t.GH

/-- Theorem stating the length of GH in the given configuration -/
theorem gh_length_is_60_over_77 (t : RightTriangleWithSquares) 
  (h1 : t.AC = 4) 
  (h2 : t.BC = 3) 
  (h3 : t.DE = 2 * t.GH) 
  (h4 : t.DJ = t.GH) 
  (h5 : t.E_on_AC + t.DE + t.GH = t.AC) 
  (h6 : t.I_on_BC + t.GH = t.BC) :
  ghLength t = 60 / 77 := by
  sorry


end NUMINAMATH_CALUDE_gh_length_is_60_over_77_l2954_295478


namespace NUMINAMATH_CALUDE_sodas_drunk_equals_three_l2954_295497

/-- The number of sodas Robin bought -/
def total_sodas : ℕ := 11

/-- The number of sodas left after drinking -/
def extras : ℕ := 8

/-- The number of sodas drunk -/
def sodas_drunk : ℕ := total_sodas - extras

theorem sodas_drunk_equals_three : sodas_drunk = 3 := by
  sorry

end NUMINAMATH_CALUDE_sodas_drunk_equals_three_l2954_295497


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_23_20_l2954_295443

theorem sum_of_solutions_eq_23_20 : 
  let f : ℝ → ℝ := λ x => (5*x + 3) * (4*x - 7)
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0 ∧ x + y = 23/20) := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_23_20_l2954_295443


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l2954_295476

theorem profit_percentage_calculation (cost_price selling_price : ℚ) : 
  cost_price = 500 → selling_price = 750 → 
  (selling_price - cost_price) / cost_price * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l2954_295476


namespace NUMINAMATH_CALUDE_lea_notebooks_l2954_295405

/-- The number of notebooks Léa bought -/
def notebooks : ℕ := sorry

/-- The cost of the book Léa bought -/
def book_cost : ℕ := 16

/-- The number of binders Léa bought -/
def num_binders : ℕ := 3

/-- The cost of each binder -/
def binder_cost : ℕ := 2

/-- The cost of each notebook -/
def notebook_cost : ℕ := 1

/-- The total cost of Léa's purchases -/
def total_cost : ℕ := 28

theorem lea_notebooks : 
  notebooks = 6 ∧
  book_cost + num_binders * binder_cost + notebooks * notebook_cost = total_cost :=
sorry

end NUMINAMATH_CALUDE_lea_notebooks_l2954_295405


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l2954_295404

theorem multiply_and_simplify (x : ℝ) (h : x ≠ 0) :
  (18 * x^3) * (4 * x^2) * (1 / (2*x)^3) = 9 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l2954_295404


namespace NUMINAMATH_CALUDE_bilingual_point_part1_bilingual_points_part2_bilingual_point_part3_l2954_295456

/-- Definition of a bilingual point -/
def is_bilingual_point (x y : ℝ) : Prop := y = 2 * x

/-- Part 1: Bilingual point of y = 3x + 1 -/
theorem bilingual_point_part1 : 
  ∃ x y : ℝ, is_bilingual_point x y ∧ y = 3 * x + 1 ∧ x = -1 ∧ y = -2 := by sorry

/-- Part 2: Bilingual points of y = k/x -/
theorem bilingual_points_part2 (k : ℝ) (h : k ≠ 0) :
  (∃ x y : ℝ, is_bilingual_point x y ∧ y = k / x) ↔ k > 0 := by sorry

/-- Part 3: Conditions for the function y = 1/4 * x^2 + (n-k-1)x + m+k+2 -/
theorem bilingual_point_part3 (n m k : ℝ) :
  (∃! x y : ℝ, is_bilingual_point x y ∧ 
    y = 1/4 * x^2 + (n - k - 1) * x + m + k + 2) ∧
  1 ≤ n ∧ n ≤ 3 ∧
  (∀ m' : ℝ, m' ≥ m → 
    ∃! x y : ℝ, is_bilingual_point x y ∧ 
      y = 1/4 * x^2 + (n - k - 1) * x + m' + k + 2) →
  k = 1 + Real.sqrt 3 ∨ k = -1 := by sorry

end NUMINAMATH_CALUDE_bilingual_point_part1_bilingual_points_part2_bilingual_point_part3_l2954_295456


namespace NUMINAMATH_CALUDE_probability_VIP_ticket_specific_l2954_295481

/-- The probability of drawing a VIP ticket from a set of tickets -/
def probability_VIP_ticket (num_VIP : ℕ) (num_regular : ℕ) : ℚ :=
  num_VIP / (num_VIP + num_regular)

/-- Theorem: The probability of drawing a VIP ticket from a set of 1 VIP ticket and 2 regular tickets is 1/3 -/
theorem probability_VIP_ticket_specific : probability_VIP_ticket 1 2 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_VIP_ticket_specific_l2954_295481


namespace NUMINAMATH_CALUDE_tan_60_plus_inverse_sqrt_3_l2954_295403

theorem tan_60_plus_inverse_sqrt_3 :
  let tan_60 := Real.sqrt 3
  tan_60 + (Real.sqrt 3)⁻¹ = (4 * Real.sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_tan_60_plus_inverse_sqrt_3_l2954_295403


namespace NUMINAMATH_CALUDE_pat_calculation_l2954_295409

theorem pat_calculation (x : ℝ) : (x / 8) * 2 - 12 = 40 → x * 8 + 2 * x + 12 > 1000 := by
  sorry

end NUMINAMATH_CALUDE_pat_calculation_l2954_295409


namespace NUMINAMATH_CALUDE_min_draw_correct_l2954_295440

/-- The total number of balls in the bag -/
def total_balls : ℕ := 70

/-- The number of red balls in the bag -/
def red_balls : ℕ := 20

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 20

/-- The number of yellow balls in the bag -/
def yellow_balls : ℕ := 20

/-- The minimum number of balls that must be drawn to ensure at least 10 balls of one color -/
def min_draw : ℕ := 38

theorem min_draw_correct : 
  ∀ (draw : ℕ), draw ≥ min_draw → 
  ∃ (color : ℕ), color ≥ 10 ∧ 
  (color ≤ red_balls ∨ color ≤ blue_balls ∨ color ≤ yellow_balls ∨ color ≤ total_balls - red_balls - blue_balls - yellow_balls) :=
by sorry

end NUMINAMATH_CALUDE_min_draw_correct_l2954_295440


namespace NUMINAMATH_CALUDE_solution_range_l2954_295462

-- Define the solution set A
def A : Set ℝ := {x | x^2 ≤ 5*x - 4}

-- Define the solution set M as a function of a
def M (a : ℝ) : Set ℝ := {x | (x - a) * (x - 2) ≤ 0}

-- State the theorem
theorem solution_range : 
  {a : ℝ | M a ⊆ A} = {a : ℝ | 1 ≤ a ∧ a ≤ 4} := by sorry

end NUMINAMATH_CALUDE_solution_range_l2954_295462


namespace NUMINAMATH_CALUDE_complement_of_A_l2954_295429

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | (x + 2) / x < 0}

theorem complement_of_A : Set.compl A = {x : ℝ | x ≥ 0 ∨ x ≤ -2} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l2954_295429


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l2954_295464

theorem fraction_product_simplification :
  let fractions : List Rat := 
    (7 / 3) :: 
    (List.range 124).map (fun n => ((8 * (n + 1) + 7) : ℚ) / (8 * (n + 1) - 1))
  (fractions.prod : ℚ) = 333 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l2954_295464


namespace NUMINAMATH_CALUDE_intersection_count_is_four_l2954_295441

/-- The number of intersection points between two curves -/
def intersection_count (C₁ C₂ : ℝ → ℝ → Prop) : ℕ :=
  sorry

/-- First curve: x² - y² + 4y - 3 = 0 -/
def C₁ (x y : ℝ) : Prop :=
  x^2 - y^2 + 4*y - 3 = 0

/-- Second curve: y = ax², where a > 0 -/
def C₂ (a : ℝ) (x y : ℝ) : Prop :=
  y = a * x^2

/-- Theorem stating that the number of intersection points is 4 -/
theorem intersection_count_is_four (a : ℝ) (h : a > 0) :
  intersection_count C₁ (C₂ a) = 4 :=
sorry

end NUMINAMATH_CALUDE_intersection_count_is_four_l2954_295441


namespace NUMINAMATH_CALUDE_max_gcd_of_sequence_l2954_295454

theorem max_gcd_of_sequence (n : ℕ+) :
  let a : ℕ+ → ℕ := fun k => 120 + k^2
  let d : ℕ+ → ℕ := fun k => Nat.gcd (a k) (a (k + 1))
  ∃ k : ℕ+, d k = 121 ∧ ∀ m : ℕ+, d m ≤ 121 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_of_sequence_l2954_295454


namespace NUMINAMATH_CALUDE_percentage_difference_l2954_295470

theorem percentage_difference (p t j : ℝ) : 
  t = 0.9375 * p →  -- t is 6.25% less than p
  j = 0.8 * t →     -- j is 20% less than t
  j = 0.75 * p :=   -- j is 25% less than p
by sorry

end NUMINAMATH_CALUDE_percentage_difference_l2954_295470


namespace NUMINAMATH_CALUDE_sin_pi_plus_2alpha_l2954_295460

theorem sin_pi_plus_2alpha (α : ℝ) (h : Real.sin (α - π/4) = 3/5) :
  Real.sin (π + 2*α) = -7/25 := by sorry

end NUMINAMATH_CALUDE_sin_pi_plus_2alpha_l2954_295460


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l2954_295414

/-- A geometric sequence with positive common ratio -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_a7 (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q →
  a 4 * a 8 = 2 * (a 5)^2 →
  a 3 = 1 →
  a 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l2954_295414


namespace NUMINAMATH_CALUDE_min_overlap_cells_l2954_295417

/-- Given positive integers m and n where m < n, in an n × n board filled with integers from 1 to n^2, 
    if the m largest numbers in each row are colored red and the m largest numbers in each column are colored blue, 
    then the minimum number of cells that are both red and blue is m^2. -/
theorem min_overlap_cells (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : m < n) : 
  (∀ (board : Fin n → Fin n → ℕ), 
    (∀ i j, board i j ∈ Finset.range (n^2 + 1)) →
    (∀ i, ∃ (red : Finset (Fin n)), red.card = m ∧ ∀ j ∈ red, ∀ k ∉ red, board i j ≥ board i k) →
    (∀ j, ∃ (blue : Finset (Fin n)), blue.card = m ∧ ∀ i ∈ blue, ∀ k ∉ blue, board i j ≥ board k j) →
    ∃ (overlap : Finset (Fin n × Fin n)), 
      overlap.card = m^2 ∧ 
      (∀ (i j), (i, j) ∈ overlap ↔ (∃ (red blue : Finset (Fin n)), 
        red.card = m ∧ blue.card = m ∧
        (∀ k ∉ red, board i j ≥ board i k) ∧
        (∀ k ∉ blue, board i j ≥ board k j) ∧
        i ∈ red ∧ j ∈ blue))) :=
by sorry

end NUMINAMATH_CALUDE_min_overlap_cells_l2954_295417


namespace NUMINAMATH_CALUDE_prob_at_least_one_box_match_l2954_295411

/-- Represents the probability of a single block matching the previous one -/
def match_probability : ℚ := 1/2

/-- Represents the number of people -/
def num_people : ℕ := 3

/-- Represents the number of boxes -/
def num_boxes : ℕ := 3

/-- Represents the number of colors -/
def num_colors : ℕ := 3

/-- Calculates the probability of all three blocks in a single box being the same color -/
def prob_single_box_match : ℚ := match_probability * match_probability

/-- Calculates the probability of at least one box having all three blocks of the same color -/
theorem prob_at_least_one_box_match : 
  (1 : ℚ) - (1 - prob_single_box_match) ^ num_boxes = 37/64 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_box_match_l2954_295411


namespace NUMINAMATH_CALUDE_quadratic_real_roots_m_range_l2954_295444

/-- 
Given a quadratic equation x^2 - 2x + m = 0, if it has real roots, 
then m ≤ 1.
-/
theorem quadratic_real_roots_m_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 0) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_m_range_l2954_295444


namespace NUMINAMATH_CALUDE_work_completion_proof_l2954_295474

/-- The number of days it takes p to complete the work alone -/
def p_days : ℕ := 80

/-- The number of days it takes q to complete the work alone -/
def q_days : ℕ := 48

/-- The total number of days the work lasted -/
def total_days : ℕ := 35

/-- The number of days after which q joined p -/
def q_join_day : ℕ := 8

/-- The work rate of p per day -/
def p_rate : ℚ := 1 / p_days

/-- The work rate of q per day -/
def q_rate : ℚ := 1 / q_days

/-- The total work completed is 1 (representing 100%) -/
def total_work : ℚ := 1

theorem work_completion_proof :
  p_rate * q_join_day + (p_rate + q_rate) * (total_days - q_join_day) = total_work :=
sorry

end NUMINAMATH_CALUDE_work_completion_proof_l2954_295474


namespace NUMINAMATH_CALUDE_magical_stack_size_is_470_l2954_295496

/-- A stack of cards is magical if it satisfies certain conditions --/
structure MagicalStack :=
  (n : ℕ)
  (total_cards : ℕ := 2 * n)
  (retains_position : ℕ := 157)
  (is_magical : Prop)

/-- The number of cards in a magical stack where card 157 retains its position --/
def magical_stack_size (stack : MagicalStack) : ℕ := stack.total_cards

/-- Theorem stating the size of the magical stack --/
theorem magical_stack_size_is_470 (stack : MagicalStack) : 
  stack.retains_position = 157 → magical_stack_size stack = 470 := by
  sorry

#check magical_stack_size_is_470

end NUMINAMATH_CALUDE_magical_stack_size_is_470_l2954_295496


namespace NUMINAMATH_CALUDE_product_of_smaller_numbers_l2954_295455

theorem product_of_smaller_numbers (A B C : ℝ) : 
  B = 10 → 
  C - B = B - A → 
  B * C = 115 → 
  A * B = 85 := by
sorry

end NUMINAMATH_CALUDE_product_of_smaller_numbers_l2954_295455


namespace NUMINAMATH_CALUDE_expression_simplification_l2954_295477

theorem expression_simplification (a : ℝ) 
  (h1 : a ≠ 2) (h2 : a ≠ -2) (h3 : a ≠ 3) :
  ((a + 3) / (a^2 - 4) - a / (a^2 - a - 6)) / ((2*a - 9) / (5*a - 10)) = 
  5 / (a^2 - a - 6) := by
  sorry

-- Verifying the result for a = 5
example : 
  let a : ℝ := 5
  ((a + 3) / (a^2 - 4) - a / (a^2 - a - 6)) / ((2*a - 9) / (5*a - 10)) = 
  5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2954_295477


namespace NUMINAMATH_CALUDE_sin_product_equals_one_eighth_l2954_295450

theorem sin_product_equals_one_eighth : 
  Real.sin (12 * π / 180) * Real.sin (48 * π / 180) * 
  Real.sin (72 * π / 180) * Real.sin (84 * π / 180) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_eighth_l2954_295450


namespace NUMINAMATH_CALUDE_remainder_sum_mod_l2954_295487

theorem remainder_sum_mod (x y : ℤ) (hx : x ≠ y) 
  (hx_mod : x % 124 = 13) (hy_mod : y % 186 = 17) : 
  (x + y + 19) % 62 = 49 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_l2954_295487


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l2954_295447

theorem smallest_sum_of_squares (x₁ x₂ x₃ : ℝ) (h_pos₁ : 0 < x₁) (h_pos₂ : 0 < x₂) (h_pos₃ : 0 < x₃)
  (h_sum : 2 * x₁ + 3 * x₂ + 4 * x₃ = 120) :
  ∃ (min : ℝ), min = 14400 / 29 ∧ x₁^2 + x₂^2 + x₃^2 ≥ min ∧
  ∃ (y₁ y₂ y₃ : ℝ), 0 < y₁ ∧ 0 < y₂ ∧ 0 < y₃ ∧
    2 * y₁ + 3 * y₂ + 4 * y₃ = 120 ∧ y₁^2 + y₂^2 + y₃^2 = min := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l2954_295447


namespace NUMINAMATH_CALUDE_amy_tips_calculation_l2954_295435

/-- Calculates the amount of tips earned by Amy given her hourly wage, hours worked, and total earnings. -/
theorem amy_tips_calculation (hourly_wage : ℝ) (hours_worked : ℝ) (total_earnings : ℝ) : 
  hourly_wage = 2 → hours_worked = 7 → total_earnings = 23 → 
  total_earnings - (hourly_wage * hours_worked) = 9 := by
  sorry

end NUMINAMATH_CALUDE_amy_tips_calculation_l2954_295435


namespace NUMINAMATH_CALUDE_parallel_line_equation_l2954_295483

/-- A line passing through point (-2, 0) and parallel to 3x - y + 1 = 0 has equation y = 3x + 6 -/
theorem parallel_line_equation :
  let point : ℝ × ℝ := (-2, 0)
  let parallel_line (x y : ℝ) := 3 * x - y + 1 = 0
  let proposed_line (x y : ℝ) := y = 3 * x + 6
  (∀ x y, parallel_line x y ↔ y = 3 * x - 1) →
  (proposed_line point.1 point.2) ∧
  (∀ x₁ y₁ x₂ y₂, parallel_line x₁ y₁ → proposed_line x₂ y₂ →
    y₂ - y₁ = 3 * (x₂ - x₁)) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l2954_295483


namespace NUMINAMATH_CALUDE_workshop_selection_count_l2954_295498

/-- The number of photography enthusiasts --/
def total_students : ℕ := 4

/-- The number of sessions in the workshop --/
def num_sessions : ℕ := 3

/-- The number of students who cannot participate in the first session --/
def restricted_students : ℕ := 2

/-- The number of different ways to select students for the workshop --/
def selection_methods : ℕ := (total_students - restricted_students) * (total_students - 1) * (total_students - 2)

theorem workshop_selection_count :
  selection_methods = 12 :=
sorry

end NUMINAMATH_CALUDE_workshop_selection_count_l2954_295498


namespace NUMINAMATH_CALUDE_tank_capacity_l2954_295439

theorem tank_capacity (initial_fraction : Rat) (added_amount : Rat) (final_fraction : Rat) :
  initial_fraction = 3/4 →
  added_amount = 8 →
  final_fraction = 7/8 →
  ∃ (total_capacity : Rat),
    initial_fraction * total_capacity + added_amount = final_fraction * total_capacity ∧
    total_capacity = 64 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l2954_295439


namespace NUMINAMATH_CALUDE_manuscript_fee_proof_l2954_295461

/-- Calculates the tax payable for manuscript income not exceeding 4000 yuan -/
def tax_payable (income : ℝ) : ℝ := (income - 800) * 0.2 * 0.7

/-- The manuscript fee before tax deduction -/
def manuscript_fee : ℝ := 2800

theorem manuscript_fee_proof :
  manuscript_fee ≤ 4000 ∧
  tax_payable manuscript_fee = 280 :=
sorry

end NUMINAMATH_CALUDE_manuscript_fee_proof_l2954_295461


namespace NUMINAMATH_CALUDE_grandpas_tomatoes_l2954_295436

/-- The number of tomatoes that grew in Grandpa's absence -/
def tomatoesGrown (initialCount : ℕ) (growthFactor : ℕ) : ℕ :=
  initialCount * growthFactor - initialCount

theorem grandpas_tomatoes :
  tomatoesGrown 36 100 = 3564 := by
  sorry

end NUMINAMATH_CALUDE_grandpas_tomatoes_l2954_295436


namespace NUMINAMATH_CALUDE_monitoring_system_odd_agents_l2954_295494

/-- Represents a cyclic monitoring system of agents -/
structure MonitoringSystem (n : ℕ) where
  -- The number of agents is positive
  agents_exist : 0 < n
  -- The monitoring function
  monitor : Fin n → Fin n
  -- The monitoring is cyclic
  cyclic : ∀ i : Fin n, monitor (monitor i) = i.succ

/-- Theorem: In a cyclic monitoring system, the number of agents is odd -/
theorem monitoring_system_odd_agents (n : ℕ) (sys : MonitoringSystem n) : 
  Odd n := by
  sorry


end NUMINAMATH_CALUDE_monitoring_system_odd_agents_l2954_295494


namespace NUMINAMATH_CALUDE_max_value_constraint_l2954_295453

theorem max_value_constraint (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  (6 * a + 3 * b + 10 * c) ≤ Real.sqrt 41 / 2 ∧
  ∃ a₀ b₀ c₀ : ℝ, 9 * a₀^2 + 4 * b₀^2 + 25 * c₀^2 = 1 ∧ 
    6 * a₀ + 3 * b₀ + 10 * c₀ = Real.sqrt 41 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l2954_295453


namespace NUMINAMATH_CALUDE_blue_paper_side_length_l2954_295451

theorem blue_paper_side_length (red_side : ℝ) (blue_side1 : ℝ) (blue_side2 : ℝ) : 
  red_side = 5 →
  blue_side1 = 4 →
  red_side * red_side = blue_side1 * blue_side2 →
  blue_side2 = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_blue_paper_side_length_l2954_295451


namespace NUMINAMATH_CALUDE_greatest_divisible_power_of_three_l2954_295428

theorem greatest_divisible_power_of_three (m : ℕ+) : 
  (∃ (k : ℕ), k = 2 ∧ (3^k : ℕ) ∣ (2^(3^m.val) + 1)) ∧
  (∀ (k : ℕ), k > 2 → ¬((3^k : ℕ) ∣ (2^(3^m.val) + 1))) :=
sorry

end NUMINAMATH_CALUDE_greatest_divisible_power_of_three_l2954_295428


namespace NUMINAMATH_CALUDE_bees_second_day_l2954_295484

def bees_first_day : ℕ := 144
def multiplier : ℕ := 3

theorem bees_second_day : bees_first_day * multiplier = 432 := by
  sorry

end NUMINAMATH_CALUDE_bees_second_day_l2954_295484


namespace NUMINAMATH_CALUDE_number_ratio_and_sum_of_squares_l2954_295488

theorem number_ratio_and_sum_of_squares (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  x / y = 2 / (3/2) → x^2 + y^2 = 400 → x = 16 ∧ y = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_and_sum_of_squares_l2954_295488


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2954_295407

theorem sufficient_not_necessary (x : ℝ) :
  (x > 1/2 → (1 - 2*x) * (x + 1) < 0) ∧
  ¬(∀ x : ℝ, (1 - 2*x) * (x + 1) < 0 → x > 1/2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2954_295407


namespace NUMINAMATH_CALUDE_nine_digit_repeat_gcd_l2954_295425

theorem nine_digit_repeat_gcd : 
  ∃ (d : ℕ), ∀ (n : ℕ), 100 ≤ n → n < 1000 → 
  Nat.gcd d (1001001 * n) = 1001001 ∧
  ∀ (m : ℕ), 100 ≤ m → m < 1000 → Nat.gcd d (1001001 * m) ∣ 1001001 :=
by sorry

end NUMINAMATH_CALUDE_nine_digit_repeat_gcd_l2954_295425


namespace NUMINAMATH_CALUDE_exam_fail_percentage_l2954_295485

theorem exam_fail_percentage 
  (total_candidates : ℕ) 
  (girls : ℕ) 
  (pass_rate : ℚ) 
  (h1 : total_candidates = 2000)
  (h2 : girls = 900)
  (h3 : pass_rate = 32/100) :
  let boys := total_candidates - girls
  let passed_candidates := (boys * pass_rate).floor + (girls * pass_rate).floor
  let failed_candidates := total_candidates - passed_candidates
  let fail_percentage := (failed_candidates : ℚ) / total_candidates * 100
  fail_percentage = 68 := by sorry

end NUMINAMATH_CALUDE_exam_fail_percentage_l2954_295485


namespace NUMINAMATH_CALUDE_min_distance_to_circle_l2954_295412

theorem min_distance_to_circle (x y : ℝ) : 
  (x - 2)^2 + (y - 1)^2 = 1 → x^2 + y^2 ≥ 6 - 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_circle_l2954_295412


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l2954_295431

/-- A parabola defined by y = 2x^2 -/
def parabola (x y : ℝ) : Prop := y = 2 * x^2

/-- The point A on the parabola -/
def point_A : ℝ × ℝ := (-1, 2)

/-- The equation of line l -/
def line_l (x y : ℝ) : Prop := 4 * x + y + 2 = 0

/-- Theorem stating that line l is tangent to the parabola at point A -/
theorem line_tangent_to_parabola :
  parabola (point_A.1) (point_A.2) ∧
  line_l (point_A.1) (point_A.2) ∧
  ∀ x y : ℝ, parabola x y ∧ line_l x y → (x, y) = point_A :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l2954_295431


namespace NUMINAMATH_CALUDE_min_value_reciprocal_plus_x_l2954_295469

theorem min_value_reciprocal_plus_x (x : ℝ) (h : x > 0) : 
  4 / x + x ≥ 4 ∧ (4 / x + x = 4 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_plus_x_l2954_295469


namespace NUMINAMATH_CALUDE_expression_simplification_l2954_295458

theorem expression_simplification :
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2954_295458


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2954_295445

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 - 2*x = 0 ↔ x = 0 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2954_295445


namespace NUMINAMATH_CALUDE_total_miles_driven_l2954_295421

theorem total_miles_driven (darius_miles julia_miles : ℕ) 
  (h1 : darius_miles = 679) 
  (h2 : julia_miles = 998) : 
  darius_miles + julia_miles = 1677 := by
  sorry

end NUMINAMATH_CALUDE_total_miles_driven_l2954_295421


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2954_295420

/-- The line equation passing through a fixed point -/
def line_equation (a x y : ℝ) : Prop :=
  a * y = (3 * a - 1) * x - 1

/-- Theorem stating that the line passes through (-1, -3) for all a -/
theorem line_passes_through_fixed_point :
  ∀ (a : ℝ), line_equation a (-1) (-3) :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2954_295420


namespace NUMINAMATH_CALUDE_equal_passengers_after_changes_l2954_295480

/-- Represents the number of passengers in a bus --/
structure BusPassengers where
  men : ℕ
  women : ℕ

/-- Calculates the total number of passengers --/
def BusPassengers.total (p : BusPassengers) : ℕ := p.men + p.women

/-- Represents the changes in passengers at a city --/
structure PassengerChanges where
  menLeaving : ℕ
  womenEntering : ℕ

/-- Applies changes to the passenger count --/
def applyChanges (p : BusPassengers) (c : PassengerChanges) : BusPassengers :=
  { men := p.men - c.menLeaving,
    women := p.women + c.womenEntering }

theorem equal_passengers_after_changes 
  (initialPassengers : BusPassengers)
  (changes : PassengerChanges) :
  initialPassengers.total = 72 →
  initialPassengers.women = initialPassengers.men / 2 →
  changes.menLeaving = 16 →
  changes.womenEntering = 8 →
  let finalPassengers := applyChanges initialPassengers changes
  finalPassengers.men = finalPassengers.women :=
by sorry

end NUMINAMATH_CALUDE_equal_passengers_after_changes_l2954_295480


namespace NUMINAMATH_CALUDE_bus_problem_l2954_295492

/-- The number of students remaining on a bus after a given number of stops,
    where half the students get off at each stop. -/
def studentsRemaining (initial : ℕ) (stops : ℕ) : ℚ :=
  initial / (2 ^ stops)

/-- Theorem stating that if a bus starts with 48 students and half of the remaining
    students get off at each of three stops, then 6 students will remain after the third stop. -/
theorem bus_problem : studentsRemaining 48 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l2954_295492


namespace NUMINAMATH_CALUDE_jeds_cards_after_four_weeks_l2954_295472

/-- Calculates the number of cards Jed has after a given number of weeks -/
def cards_after_weeks (initial_cards : ℕ) (cards_per_week : ℕ) (cards_given_away : ℕ) (weeks : ℕ) : ℕ :=
  initial_cards + cards_per_week * weeks - cards_given_away * (weeks / 2)

/-- Proves that Jed will have 40 cards after 4 weeks -/
theorem jeds_cards_after_four_weeks :
  ∃ (weeks : ℕ), cards_after_weeks 20 6 2 weeks = 40 ∧ weeks = 4 :=
by
  sorry

#check jeds_cards_after_four_weeks

end NUMINAMATH_CALUDE_jeds_cards_after_four_weeks_l2954_295472


namespace NUMINAMATH_CALUDE_problem_1_l2954_295438

theorem problem_1 : 4 * Real.sin (π / 3) + (1 / 3)⁻¹ + |-2| - Real.sqrt 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2954_295438


namespace NUMINAMATH_CALUDE_triangle_area_l2954_295413

/-- Given a triangle MNP where:
  * MN is the side opposite to the 60° angle
  * MP is the hypotenuse with length 40
  * Angle N is 90°
  Prove that the area of triangle MNP is 200√3 -/
theorem triangle_area (M N P : ℝ × ℝ) : 
  let MN := Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)
  let MP := Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2)
  let NP := Real.sqrt ((N.1 - P.1)^2 + (N.2 - P.2)^2)
  (∃ θ : Real, θ = π/3 ∧ MN = MP * Real.sin θ) →  -- MN is opposite to 60° angle
  MP = 40 →  -- MP is the hypotenuse with length 40
  (N.1 - M.1) * (P.1 - M.1) + (N.2 - M.2) * (P.2 - M.2) = 0 →  -- Angle N is 90°
  (1/2) * MN * NP = 200 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l2954_295413


namespace NUMINAMATH_CALUDE_train_b_length_l2954_295437

/-- Calculates the length of Train B given the conditions of the problem -/
theorem train_b_length : 
  let train_a_speed : ℝ := 10  -- Initial speed of Train A in m/s
  let train_b_speed : ℝ := 12.5  -- Initial speed of Train B in m/s
  let train_a_accel : ℝ := 1  -- Acceleration of Train A in m/s²
  let train_b_decel : ℝ := 0.5  -- Deceleration of Train B in m/s²
  let passing_time : ℝ := 10  -- Time to pass each other in seconds
  
  let train_a_final_speed := train_a_speed + train_a_accel * passing_time
  let train_b_final_speed := train_b_speed - train_b_decel * passing_time
  let relative_speed := train_a_final_speed + train_b_final_speed
  
  relative_speed * passing_time = 275 := by
  sorry

#check train_b_length

end NUMINAMATH_CALUDE_train_b_length_l2954_295437


namespace NUMINAMATH_CALUDE_bologna_sandwiches_l2954_295418

/-- Given a ratio of cheese, bologna, and peanut butter sandwiches as 1:7:8 and a total of 80 sandwiches,
    prove that the number of bologna sandwiches is 35. -/
theorem bologna_sandwiches (total : ℕ) (cheese : ℕ) (bologna : ℕ) (peanut_butter : ℕ)
  (h_total : total = 80)
  (h_ratio : cheese + bologna + peanut_butter = 16)
  (h_cheese : cheese = 1)
  (h_bologna : bologna = 7)
  (h_peanut_butter : peanut_butter = 8) :
  (total / (cheese + bologna + peanut_butter)) * bologna = 35 :=
by sorry

end NUMINAMATH_CALUDE_bologna_sandwiches_l2954_295418


namespace NUMINAMATH_CALUDE_mary_flour_calculation_l2954_295468

/-- The number of cups of flour required by the recipe -/
def total_flour : ℕ := 7

/-- The number of cups of flour Mary has already added -/
def added_flour : ℕ := 2

/-- The number of cups of flour Mary needs to add -/
def flour_to_add : ℕ := total_flour - added_flour

theorem mary_flour_calculation :
  flour_to_add = 5 := by sorry

end NUMINAMATH_CALUDE_mary_flour_calculation_l2954_295468


namespace NUMINAMATH_CALUDE_calculator_sale_loss_l2954_295486

theorem calculator_sale_loss :
  ∀ (x y : ℝ),
    x * (1 + 0.2) = 60 →
    y * (1 - 0.2) = 60 →
    60 + 60 - (x + y) = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_calculator_sale_loss_l2954_295486


namespace NUMINAMATH_CALUDE_mary_balloons_l2954_295410

/-- Given that Nancy has 7 black balloons and Mary has 4 times more black balloons than Nancy,
    prove that Mary has 28 black balloons. -/
theorem mary_balloons (nancy_balloons : ℕ) (mary_multiplier : ℕ) 
    (h1 : nancy_balloons = 7)
    (h2 : mary_multiplier = 4) : 
  nancy_balloons * mary_multiplier = 28 := by
  sorry

end NUMINAMATH_CALUDE_mary_balloons_l2954_295410


namespace NUMINAMATH_CALUDE_vector_sum_equality_l2954_295475

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- For any three points A, B, and C in a vector space, 
    the sum of vectors AB, BC, and BA equals vector BC. -/
theorem vector_sum_equality (A B C : V) : 
  (B - A) + (C - B) + (A - B) = C - B := by sorry

end NUMINAMATH_CALUDE_vector_sum_equality_l2954_295475


namespace NUMINAMATH_CALUDE_regular_tetrahedron_face_center_volume_ratio_l2954_295482

/-- The ratio of the volume of a tetrahedron formed by the centers of the faces of a regular tetrahedron to the volume of the original tetrahedron -/
def face_center_tetrahedron_volume_ratio : ℚ :=
  8 / 27

/-- Theorem stating that in a regular tetrahedron, the ratio of the volume of the tetrahedron 
    formed by the centers of the faces to the volume of the original tetrahedron is 8/27 -/
theorem regular_tetrahedron_face_center_volume_ratio :
  face_center_tetrahedron_volume_ratio = 8 / 27 := by
  sorry

#eval Nat.gcd 8 27  -- To verify that 8 and 27 are coprime

#eval 8 + 27  -- To compute the final answer

end NUMINAMATH_CALUDE_regular_tetrahedron_face_center_volume_ratio_l2954_295482


namespace NUMINAMATH_CALUDE_walking_rate_ratio_l2954_295402

/-- The ratio of a boy's faster walking rate to his usual walking rate, given his usual time and early arrival time. -/
theorem walking_rate_ratio (usual_time early_time : ℕ) : 
  usual_time = 42 → early_time = 6 → (usual_time : ℚ) / (usual_time - early_time) = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_walking_rate_ratio_l2954_295402


namespace NUMINAMATH_CALUDE_min_distance_complex_circles_l2954_295424

theorem min_distance_complex_circles (z w : ℂ) 
  (hz : Complex.abs (z - (2 - 4 * Complex.I)) = 2)
  (hw : Complex.abs (w - (6 - 5 * Complex.I)) = 4) :
  ∃ (min_dist : ℝ), 
    (∀ z' w' : ℂ, 
      Complex.abs (z' - (2 - 4 * Complex.I)) = 2 → 
      Complex.abs (w' - (6 - 5 * Complex.I)) = 4 → 
      Complex.abs (z' - w') ≥ min_dist) ∧
    Complex.abs (z - w) ≥ min_dist ∧
    min_dist = Real.sqrt 17 - 6 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_complex_circles_l2954_295424


namespace NUMINAMATH_CALUDE_rectangle_triangle_equal_area_l2954_295489

/-- Given a rectangle with perimeter 60 and a triangle with height 60, 
    if their areas are equal, then the base of the triangle is 20/3 -/
theorem rectangle_triangle_equal_area (rect_width rect_height tri_base : ℝ) : 
  rect_width > 0 → 
  rect_height > 0 → 
  tri_base > 0 → 
  rect_width + rect_height = 30 → 
  rect_width * rect_height = 30 * tri_base → 
  tri_base = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_triangle_equal_area_l2954_295489


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2954_295426

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_equation_solution (z : ℂ) (a : ℝ) (h1 : is_pure_imaginary z) 
  (h2 : (2 - Complex.I) * z = a + Complex.I) : a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2954_295426


namespace NUMINAMATH_CALUDE_point_coordinates_l2954_295473

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the Cartesian coordinate system -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The distance of a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- The distance of a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: A point in the fourth quadrant with distances 3 and 5 to x-axis and y-axis respectively has coordinates (5, -3) -/
theorem point_coordinates (p : Point) 
  (h1 : fourth_quadrant p) 
  (h2 : distance_to_x_axis p = 3) 
  (h3 : distance_to_y_axis p = 5) : 
  p = Point.mk 5 (-3) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l2954_295473


namespace NUMINAMATH_CALUDE_difference_number_and_three_fifths_l2954_295419

theorem difference_number_and_three_fifths (n : ℚ) : n = 140 → n - (3 / 5 * n) = 56 := by
  sorry

end NUMINAMATH_CALUDE_difference_number_and_three_fifths_l2954_295419


namespace NUMINAMATH_CALUDE_alien_resource_conversion_l2954_295422

/-- Converts a base-5 number represented as a list of digits to base 10 -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

theorem alien_resource_conversion :
  base5ToBase10 [3, 6, 2] = 83 := by
  sorry

end NUMINAMATH_CALUDE_alien_resource_conversion_l2954_295422


namespace NUMINAMATH_CALUDE_euclid_middle_school_contest_l2954_295465

/-- The number of distinct students preparing for the math contest at Euclid Middle School -/
def total_students (euler_students fibonacci_students gauss_students overlap : ℕ) : ℕ :=
  euler_students + fibonacci_students + gauss_students - overlap

theorem euclid_middle_school_contest :
  let euler_students := 12
  let fibonacci_students := 10
  let gauss_students := 11
  let overlap := 3
  total_students euler_students fibonacci_students gauss_students overlap = 27 := by
  sorry

#eval total_students 12 10 11 3

end NUMINAMATH_CALUDE_euclid_middle_school_contest_l2954_295465
