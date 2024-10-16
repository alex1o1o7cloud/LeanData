import Mathlib

namespace NUMINAMATH_CALUDE_xy_ratio_values_l2745_274513

theorem xy_ratio_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : 2 * x^2 + 2 * y^2 = 5 * x * y) : 
  (x + y) / (x - y) = 3 ∨ (x + y) / (x - y) = -3 := by
  sorry

end NUMINAMATH_CALUDE_xy_ratio_values_l2745_274513


namespace NUMINAMATH_CALUDE_tadd_500th_number_l2745_274562

def tadd_sequence (n : ℕ) : ℕ := (3 * n - 2) ^ 2

theorem tadd_500th_number : tadd_sequence 500 = 2244004 := by
  sorry

end NUMINAMATH_CALUDE_tadd_500th_number_l2745_274562


namespace NUMINAMATH_CALUDE_beach_trip_time_l2745_274585

/-- Calculates the total trip time given the one-way drive time and the ratio of destination time to total drive time -/
def totalTripTime (oneWayDriveTime : ℝ) (destinationTimeRatio : ℝ) : ℝ :=
  let totalDriveTime := 2 * oneWayDriveTime
  let destinationTime := destinationTimeRatio * totalDriveTime
  totalDriveTime + destinationTime

/-- Proves that for a trip with 2 hours one-way drive time and 2.5 ratio of destination time to total drive time, the total trip time is 14 hours -/
theorem beach_trip_time : totalTripTime 2 2.5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_beach_trip_time_l2745_274585


namespace NUMINAMATH_CALUDE_quadratic_radical_rule_l2745_274523

theorem quadratic_radical_rule (n : ℕ+) : 
  Real.sqrt (n + 1 / (n + 2)) = (n + 1) * Real.sqrt (1 / (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_rule_l2745_274523


namespace NUMINAMATH_CALUDE_lindas_savings_l2745_274580

theorem lindas_savings (furniture_fraction : Real) (tv_cost : Real) 
  (refrigerator_percent : Real) (furniture_discount : Real) (tv_tax : Real) :
  furniture_fraction = 3/4 →
  tv_cost = 210 →
  refrigerator_percent = 20/100 →
  furniture_discount = 7/100 →
  tv_tax = 6/100 →
  ∃ (savings : Real),
    savings = 1898.40 ∧
    (furniture_fraction * savings * (1 - furniture_discount) + 
     tv_cost * (1 + tv_tax) + 
     tv_cost * (1 + refrigerator_percent)) = savings :=
by
  sorry


end NUMINAMATH_CALUDE_lindas_savings_l2745_274580


namespace NUMINAMATH_CALUDE_integer_remainder_properties_l2745_274549

theorem integer_remainder_properties (n : ℤ) (h : n % 20 = 13) :
  (n % 4 + n % 5 = 4) ∧ n % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_integer_remainder_properties_l2745_274549


namespace NUMINAMATH_CALUDE_sequence_eventually_zero_l2745_274570

/-- The function f defined on the interval [0, 1] -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≤ k then 0
  else 1 - (Real.sqrt (k * x) + Real.sqrt ((1 - k) * (1 - x)))^2

/-- The theorem stating that the sequence eventually becomes zero -/
theorem sequence_eventually_zero (k : ℝ) (hk : 0 < k ∧ k < 1) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → (f k)^[n] 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_sequence_eventually_zero_l2745_274570


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_ab_value_l2745_274516

theorem ellipse_hyperbola_ab_value (a b : ℝ) : 
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → (x = 0 ∧ y = 5) ∨ (x = 0 ∧ y = -5)) →
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → (x = 7 ∧ y = 0) ∨ (x = -7 ∧ y = 0)) →
  |a*b| = 2 * Real.sqrt 111 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_ab_value_l2745_274516


namespace NUMINAMATH_CALUDE_book_sale_loss_l2745_274538

/-- Given that the cost price of 15 books equals the selling price of 20 books,
    prove that there is a 25% loss. -/
theorem book_sale_loss (C S : ℝ) (h : 15 * C = 20 * S) :
  (C - S) / C * 100 = 25 :=
sorry

end NUMINAMATH_CALUDE_book_sale_loss_l2745_274538


namespace NUMINAMATH_CALUDE_locus_and_slope_theorem_l2745_274587

noncomputable def A : ℝ × ℝ := (0, 4/3)
noncomputable def B : ℝ × ℝ := (-1, 0)
noncomputable def C : ℝ × ℝ := (1, 0)

def distance_to_line (P : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ := sorry

def line_BC : ℝ × ℝ → Prop := sorry
def line_AB : ℝ × ℝ → Prop := sorry
def line_AC : ℝ × ℝ → Prop := sorry

def locus_equation_1 (P : ℝ × ℝ) : Prop :=
  (P.1^2 + P.2^2 + 3/2 * P.2 - 1 = 0)

def locus_equation_2 (P : ℝ × ℝ) : Prop :=
  (8 * P.1^2 - 17 * P.2^2 + 12 * P.2 - 8 = 0)

def incenter (triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ × ℝ := sorry

def line_intersects_locus_at_3_points (l : ℝ → ℝ) : Prop := sorry

def slope_set : Set ℝ := {0, 1/2, -1/2, 2 * Real.sqrt 34 / 17, -2 * Real.sqrt 34 / 17, Real.sqrt 2 / 2, -Real.sqrt 2 / 2}

theorem locus_and_slope_theorem :
  ∀ P : ℝ × ℝ,
  (distance_to_line P line_BC)^2 = (distance_to_line P line_AB) * (distance_to_line P line_AC) →
  (locus_equation_1 P ∨ locus_equation_2 P) ∧
  ∀ l : ℝ → ℝ,
  (∃ x : ℝ, l x = (incenter (A, B, C)).2) →
  line_intersects_locus_at_3_points l →
  ∃ k : ℝ, k ∈ slope_set ∧ ∀ x : ℝ, l x = k * x + (incenter (A, B, C)).2 :=
sorry

end NUMINAMATH_CALUDE_locus_and_slope_theorem_l2745_274587


namespace NUMINAMATH_CALUDE_sum_of_roots_l2745_274574

theorem sum_of_roots (α β : ℝ) : 
  α^3 - 3*α^2 + 5*α - 4 = 0 → 
  β^3 - 3*β^2 + 5*β - 2 = 0 → 
  α + β = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2745_274574


namespace NUMINAMATH_CALUDE_certain_number_existence_l2745_274569

theorem certain_number_existence : ∃ x : ℕ, x > 1 ∧ (x - 1) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_existence_l2745_274569


namespace NUMINAMATH_CALUDE_a_bounds_l2745_274545

theorem a_bounds (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 3)
  (sum_squares : a^2 + 2*b^2 + 3*c^2 + 6*d^2 = 5) :
  1 ≤ a ∧ a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_a_bounds_l2745_274545


namespace NUMINAMATH_CALUDE_expression_evaluation_l2745_274572

theorem expression_evaluation (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (x^2 / y^3) / (y / x) = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2745_274572


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_base_6_l2745_274519

def base_6_to_10 (n : ℕ) : ℕ := n

def base_10_to_6 (n : ℕ) : ℕ := n

def arithmetic_sum (a₁ aₙ n : ℕ) : ℕ := n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum_base_6 (a₁ aₙ d : ℕ) 
  (h₁ : a₁ = base_6_to_10 5)
  (h₂ : aₙ = base_6_to_10 31)
  (h₃ : d = 2)
  (h₄ : aₙ = a₁ + (n - 1) * d) :
  base_10_to_6 (arithmetic_sum a₁ aₙ ((aₙ - a₁) / d + 1)) = 240 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_base_6_l2745_274519


namespace NUMINAMATH_CALUDE_flour_for_cookies_l2745_274541

/-- Given a recipe where 20 cookies require 3 cups of flour,
    calculate the number of cups of flour needed for 100 cookies. -/
theorem flour_for_cookies (original_cookies : ℕ) (original_flour : ℕ) (target_cookies : ℕ) :
  original_cookies = 20 →
  original_flour = 3 →
  target_cookies = 100 →
  (target_cookies * original_flour) / original_cookies = 15 :=
by sorry

end NUMINAMATH_CALUDE_flour_for_cookies_l2745_274541


namespace NUMINAMATH_CALUDE_ball_probability_l2745_274515

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h_total : total = 100)
  (h_white : white = 20)
  (h_green : green = 30)
  (h_yellow : yellow = 10)
  (h_red : red = 37)
  (h_purple : purple = 3)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l2745_274515


namespace NUMINAMATH_CALUDE_quadratic_roots_in_unit_interval_l2745_274564

theorem quadratic_roots_in_unit_interval (a b c : ℤ) (ha : a > 0) 
  (h_roots : ∃ x y : ℝ, x ≠ y ∧ 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ 
    (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ) = 0 ∧
    (a : ℝ) * y^2 + (b : ℝ) * y + (c : ℝ) = 0) : 
  a ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_in_unit_interval_l2745_274564


namespace NUMINAMATH_CALUDE_negation_equivalence_l2745_274558

theorem negation_equivalence (a b c : ℝ) :
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c = 3 ∧ a^2 + b^2 + c^2 < 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2745_274558


namespace NUMINAMATH_CALUDE_range_of_m_l2745_274508

def p (m : ℝ) : Prop := ∃ x y : ℝ, x + y - m = 0 ∧ (x - 1)^2 + y^2 = 1

def q (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 - 2 * x + 1 = 0

theorem range_of_m (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬¬(q m)) : m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2745_274508


namespace NUMINAMATH_CALUDE_sqrt_product_equals_product_l2745_274536

theorem sqrt_product_equals_product : Real.sqrt (9 * 16) = 3 * 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_product_l2745_274536


namespace NUMINAMATH_CALUDE_parallel_lines_minimum_value_l2745_274501

-- Define the linear functions f and g
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c

-- Define the theorem
theorem parallel_lines_minimum_value 
  (a b c : ℝ) 
  (h1 : a ≠ 0)  -- Ensure lines are not parallel to coordinate axes
  (h2 : ∃ (x : ℝ), (f a b x)^2 + g a c x = 4)  -- Minimum value of (f(x))^2 + g(x) is 4
  : ∃ (x : ℝ), (g a c x)^2 + f a b x = -9/2 :=  -- Minimum value of (g(x))^2 + f(x) is -9/2
by sorry

end NUMINAMATH_CALUDE_parallel_lines_minimum_value_l2745_274501


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2745_274507

/-- Given a line l passing through points (a-2, -1) and (-a-2, 1), perpendicular to a line
    passing through (-2, 1) with slope -2/3, prove that a = -2/3 --/
theorem perpendicular_lines_a_value (a : ℝ) : 
  let l : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (t * (-a-2) + (1-t) * (a-2), t * 1 + (1-t) * (-1))}
  let other_line : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (t + (-2), -2/3 * t + 1)}
  (∀ p ∈ l, ∀ q ∈ other_line, (p.1 - q.1) * (-2/3) + (p.2 - q.2) * 1 = 0) →
  a = -2/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2745_274507


namespace NUMINAMATH_CALUDE_unique_element_condition_l2745_274533

def M (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + 1 = 0}

theorem unique_element_condition (a : ℝ) : 
  (∃! x, x ∈ M a) ↔ (a = 0 ∨ a = 1) := by
sorry

end NUMINAMATH_CALUDE_unique_element_condition_l2745_274533


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l2745_274526

theorem quadratic_solution_sum (x p q : ℝ) : 
  (5 * x^2 + 7 = 4 * x - 12) →
  (∃ (i : ℂ), x = p + q * i ∨ x = p - q * i) →
  p + q^2 = 101 / 25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l2745_274526


namespace NUMINAMATH_CALUDE_linear_function_k_value_l2745_274591

theorem linear_function_k_value : ∀ k : ℝ, 
  (∀ x y : ℝ, y = k * x + 3) →  -- Linear function condition
  (2 = k * 1 + 3) →             -- Passes through (1, 2)
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_linear_function_k_value_l2745_274591


namespace NUMINAMATH_CALUDE_min_value_theorem_l2745_274556

theorem min_value_theorem (x y : ℝ) 
  (h : ∀ (n : ℕ), n > 0 → n * x + (1 / n) * y ≥ 1) :
  (∀ (a b : ℝ), (∀ (n : ℕ), n > 0 → n * a + (1 / n) * b ≥ 1) → 41 * x + 2 * y ≤ 41 * a + 2 * b) ∧ 
  (∃ (x₀ y₀ : ℝ), (∀ (n : ℕ), n > 0 → n * x₀ + (1 / n) * y₀ ≥ 1) ∧ 41 * x₀ + 2 * y₀ = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2745_274556


namespace NUMINAMATH_CALUDE_remainder_problem_l2745_274559

theorem remainder_problem (n : ℤ) (h : n % 7 = 5) : (3 * n + 2) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2745_274559


namespace NUMINAMATH_CALUDE_arccos_cos_three_l2745_274598

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_arccos_cos_three_l2745_274598


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2745_274522

/-- Given two arithmetic sequences, this theorem proves that if the ratio of their sums
    follows a specific pattern, then the ratio of their 7th terms is 13/20. -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) :
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Sum formula for arithmetic sequence a
  (∀ n, T n = (n * (b 1 + b n)) / 2) →  -- Sum formula for arithmetic sequence b
  (∀ n, S n / T n = n / (n + 7)) →      -- Given condition
  a 7 / b 7 = 13 / 20 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2745_274522


namespace NUMINAMATH_CALUDE_tan_alpha_equals_one_l2745_274529

theorem tan_alpha_equals_one (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 2 * Real.sin (α - 15 * π / 180) - 1 = 0) : Real.tan α = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_equals_one_l2745_274529


namespace NUMINAMATH_CALUDE_simplify_expression_l2745_274592

theorem simplify_expression (b : ℝ) : 3*b*(3*b^3 + 2*b) - 2*b^2 = 9*b^4 + 4*b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2745_274592


namespace NUMINAMATH_CALUDE_age_difference_l2745_274509

theorem age_difference (A B C : ℤ) (h1 : C = A - 12) : A + B - (B + C) = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2745_274509


namespace NUMINAMATH_CALUDE_laura_garden_area_l2745_274503

/-- Represents a rectangular garden with fence posts --/
structure Garden where
  total_posts : ℕ
  gap : ℕ
  longer_side_post_ratio : ℕ

/-- Calculates the area of the garden given its specifications --/
def garden_area (g : Garden) : ℕ :=
  let shorter_side_posts := (g.total_posts + 4) / (1 + g.longer_side_post_ratio)
  let longer_side_posts := shorter_side_posts * g.longer_side_post_ratio
  let shorter_side_length := (shorter_side_posts - 1) * g.gap
  let longer_side_length := (longer_side_posts - 1) * g.gap
  shorter_side_length * longer_side_length

theorem laura_garden_area :
  let g : Garden := { total_posts := 24, gap := 5, longer_side_post_ratio := 3 }
  garden_area g = 3000 := by
  sorry


end NUMINAMATH_CALUDE_laura_garden_area_l2745_274503


namespace NUMINAMATH_CALUDE_last_digit_of_repeated_seven_exponentiation_l2745_274506

def repeated_exponentiation (base : ℕ) (times : ℕ) : ℕ :=
  match times with
  | 0 => base
  | n + 1 => repeated_exponentiation (base^base) n

theorem last_digit_of_repeated_seven_exponentiation :
  repeated_exponentiation 7 1000 % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_repeated_seven_exponentiation_l2745_274506


namespace NUMINAMATH_CALUDE_big_n_conference_teams_l2745_274520

/-- The number of games in a round-robin tournament with n teams -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of teams in the BIG N conference -/
theorem big_n_conference_teams : ∃ (n : ℕ), n > 0 ∧ num_games n = 21 :=
  sorry

end NUMINAMATH_CALUDE_big_n_conference_teams_l2745_274520


namespace NUMINAMATH_CALUDE_third_vertex_coordinates_l2745_274544

/-- Given a triangle with vertices (2, 3), (0, 0), and (x, 0) where x < 0,
    if the area of the triangle is 12 square units, then x = -8. -/
theorem third_vertex_coordinates (x : ℝ) (h1 : x < 0) :
  (1/2 : ℝ) * |3 * x| = 12 → x = -8 := by sorry

end NUMINAMATH_CALUDE_third_vertex_coordinates_l2745_274544


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l2745_274535

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_one :
  (∀ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l2745_274535


namespace NUMINAMATH_CALUDE_data_entry_team_size_l2745_274566

theorem data_entry_team_size :
  let rudy_speed := 64
  let joyce_speed := 76
  let gladys_speed := 91
  let lisa_speed := 80
  let mike_speed := 89
  let team_average := 80
  let total_speed := rudy_speed + joyce_speed + gladys_speed + lisa_speed + mike_speed
  (total_speed / team_average : ℚ) = 5 := by sorry

end NUMINAMATH_CALUDE_data_entry_team_size_l2745_274566


namespace NUMINAMATH_CALUDE_cos_four_arccos_two_fifths_l2745_274537

theorem cos_four_arccos_two_fifths :
  Real.cos (4 * Real.arccos (2 / 5)) = -47 / 625 := by
  sorry

end NUMINAMATH_CALUDE_cos_four_arccos_two_fifths_l2745_274537


namespace NUMINAMATH_CALUDE_volleyball_teams_l2745_274514

theorem volleyball_teams (managers : ℕ) (employees : ℕ) (team_size : ℕ) : 
  managers = 23 → employees = 7 → team_size = 5 → 
  (managers + employees) / team_size = 6 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_teams_l2745_274514


namespace NUMINAMATH_CALUDE_fly_path_length_l2745_274582

theorem fly_path_length (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c = 5) 
  (h4 : a^2 + b^2 = c^2) : ∃ (path_length : ℝ), path_length > 10 ∧ 
  path_length = 5 * c := by sorry

end NUMINAMATH_CALUDE_fly_path_length_l2745_274582


namespace NUMINAMATH_CALUDE_max_y_coordinate_l2745_274581

theorem max_y_coordinate (x y : ℝ) : 
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_y_coordinate_l2745_274581


namespace NUMINAMATH_CALUDE_solve_equations_l2745_274553

theorem solve_equations :
  (∃ x : ℝ, 2 * (x + 8) = 3 * (x - 1) ∧ x = 19) ∧
  (∃ y : ℝ, (3 * y - 1) / 4 - 1 = (5 * y - 7) / 6 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l2745_274553


namespace NUMINAMATH_CALUDE_inequalities_proof_l2745_274521

theorem inequalities_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  2/(a-1) + 1/(b-2) ≥ 2 ∧ 2*a + b ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l2745_274521


namespace NUMINAMATH_CALUDE_speed_AB_is_60_l2745_274568

-- Define the distances and speeds
def distance_BC : ℝ := 1  -- We can use any positive real number as the base distance
def distance_AB : ℝ := 2 * distance_BC
def speed_BC : ℝ := 20
def average_speed : ℝ := 36

-- Define the speed from A to B as a variable we want to solve for
def speed_AB : ℝ := sorry

-- Theorem statement
theorem speed_AB_is_60 : speed_AB = 60 := by
  sorry

end NUMINAMATH_CALUDE_speed_AB_is_60_l2745_274568


namespace NUMINAMATH_CALUDE_f_values_l2745_274590

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 3 * x + 4 else 4 * x - 1

theorem f_values : f (-3) = -5 ∧ f 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_f_values_l2745_274590


namespace NUMINAMATH_CALUDE_cordelia_bleaching_time_l2745_274557

/-- Represents the time in hours for a hair coloring process -/
structure HairColoringTime where
  bleaching : ℝ
  dyeing : ℝ

/-- The properties of Cordelia's hair coloring process -/
def cordelias_hair_coloring (t : HairColoringTime) : Prop :=
  t.bleaching + t.dyeing = 9 ∧ t.dyeing = 2 * t.bleaching

theorem cordelia_bleaching_time :
  ∀ t : HairColoringTime, cordelias_hair_coloring t → t.bleaching = 3 := by
  sorry

end NUMINAMATH_CALUDE_cordelia_bleaching_time_l2745_274557


namespace NUMINAMATH_CALUDE_printer_time_calculation_l2745_274599

/-- Given a printer that prints 23 pages per minute, prove that it takes 15 minutes to print 345 pages. -/
theorem printer_time_calculation (print_rate : ℕ) (total_pages : ℕ) (time : ℕ) : 
  print_rate = 23 → total_pages = 345 → time = total_pages / print_rate → time = 15 := by
  sorry

end NUMINAMATH_CALUDE_printer_time_calculation_l2745_274599


namespace NUMINAMATH_CALUDE_reflection_line_sum_l2745_274567

/-- Given a line y = mx + b, if the reflection of point (2,3) across this line is (10,6),
    then m + b = 107/6 -/
theorem reflection_line_sum (m b : ℚ) : 
  (∃ (x y : ℚ), 
    -- Midpoint of original and reflected point lies on the line
    (x + 2)/2 = 6 ∧ (y + 3)/2 = 9/2 ∧ y = m*x + b ∧
    -- Perpendicular slope condition
    (x - 2)*(10 - 2) + (y - 3)*(6 - 3) = 0 ∧
    -- Distance equality condition
    (x - 2)^2 + (y - 3)^2 = (10 - x)^2 + (6 - y)^2) →
  m + b = 107/6 := by
sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l2745_274567


namespace NUMINAMATH_CALUDE_tangent_ellipse_major_axis_length_l2745_274511

/-- An ellipse with foci at (3, -4 + 2√3) and (3, -4 - 2√3), tangent to both x and y axes -/
structure TangentEllipse where
  /-- The ellipse is tangent to the x-axis -/
  tangent_x : Bool
  /-- The ellipse is tangent to the y-axis -/
  tangent_y : Bool
  /-- The first focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- The second focus of the ellipse -/
  focus2 : ℝ × ℝ
  /-- Ensure the foci are at the specified points -/
  foci_constraint : focus1 = (3, -4 + 2 * Real.sqrt 3) ∧ focus2 = (3, -4 - 2 * Real.sqrt 3)
  /-- Ensure the ellipse is tangent to both axes -/
  tangent_constraint : tangent_x ∧ tangent_y

/-- The length of the major axis of the ellipse -/
def majorAxisLength (e : TangentEllipse) : ℝ := 8

/-- Theorem stating that the major axis length of the specified ellipse is 8 -/
theorem tangent_ellipse_major_axis_length (e : TangentEllipse) : 
  majorAxisLength e = 8 := by sorry

end NUMINAMATH_CALUDE_tangent_ellipse_major_axis_length_l2745_274511


namespace NUMINAMATH_CALUDE_abc_positive_l2745_274594

/-- A quadratic function y = ax^2 + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  opens_upwards : a > 0
  has_two_real_roots : b^2 > 4*a*c
  right_root_larger : ∃ (r₁ r₂ : ℝ), r₁ < 0 ∧ r₂ > 0 ∧ |r₂| > |r₁| ∧
    a*r₁^2 + b*r₁ + c = 0 ∧ a*r₂^2 + b*r₂ + c = 0

/-- Theorem: For a quadratic function with the given properties, abc > 0 -/
theorem abc_positive (f : QuadraticFunction) : f.a * f.b * f.c > 0 := by
  sorry

end NUMINAMATH_CALUDE_abc_positive_l2745_274594


namespace NUMINAMATH_CALUDE_sin_negative_1020_degrees_l2745_274565

theorem sin_negative_1020_degrees : Real.sin ((-1020 : ℝ) * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_1020_degrees_l2745_274565


namespace NUMINAMATH_CALUDE_martinez_chiquita_height_difference_l2745_274543

/-- The height difference between Mr. Martinez and Chiquita -/
theorem martinez_chiquita_height_difference :
  ∀ (martinez_height chiquita_height : ℝ),
  chiquita_height = 5 →
  martinez_height + chiquita_height = 12 →
  martinez_height - chiquita_height = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_martinez_chiquita_height_difference_l2745_274543


namespace NUMINAMATH_CALUDE_john_booking_l2745_274563

/-- Calculates the number of nights booked given the nightly rate, discount, and total paid -/
def nights_booked (nightly_rate : ℕ) (discount : ℕ) (total_paid : ℕ) : ℕ :=
  (total_paid + discount) / nightly_rate

theorem john_booking :
  nights_booked 250 100 650 = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_booking_l2745_274563


namespace NUMINAMATH_CALUDE_factorization_equality_l2745_274510

theorem factorization_equality (x y : ℝ) : 
  y^2 + x*y - 3*x - y - 6 = (y - 3) * (y + 2 + x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2745_274510


namespace NUMINAMATH_CALUDE_paint_liters_needed_l2745_274518

def cost_of_brushes : ℝ := 20
def cost_of_canvas : ℝ := 3 * cost_of_brushes
def cost_of_paint_per_liter : ℝ := 8
def selling_price : ℝ := 200
def profit : ℝ := 80

theorem paint_liters_needed : 
  ∃ (liters : ℝ), 
    liters > 0 ∧ 
    cost_of_brushes + cost_of_canvas + cost_of_paint_per_liter * liters = selling_price - profit ∧
    liters = 5 := by
  sorry

end NUMINAMATH_CALUDE_paint_liters_needed_l2745_274518


namespace NUMINAMATH_CALUDE_guitar_strings_problem_l2745_274571

theorem guitar_strings_problem (total_strings : ℕ) 
  (num_basses : ℕ) (strings_per_bass : ℕ) 
  (strings_per_normal_guitar : ℕ) (strings_difference : ℕ) :
  let num_normal_guitars := 2 * num_basses
  let strings_for_basses := num_basses * strings_per_bass
  let strings_for_normal_guitars := num_normal_guitars * strings_per_normal_guitar
  let remaining_strings := total_strings - strings_for_basses - strings_for_normal_guitars
  let strings_per_fewer_guitar := strings_per_normal_guitar - strings_difference
  total_strings = 72 ∧ 
  num_basses = 3 ∧ 
  strings_per_bass = 4 ∧ 
  strings_per_normal_guitar = 6 ∧ 
  strings_difference = 3 →
  strings_per_fewer_guitar = 3 :=
by sorry

end NUMINAMATH_CALUDE_guitar_strings_problem_l2745_274571


namespace NUMINAMATH_CALUDE_path_area_l2745_274539

/-- The area of a ring-shaped path around a circular lawn -/
theorem path_area (r : ℝ) (w : ℝ) (h1 : r = 35) (h2 : w = 7) :
  (π * (r + w)^2 - π * r^2) = 539 * π :=
sorry

end NUMINAMATH_CALUDE_path_area_l2745_274539


namespace NUMINAMATH_CALUDE_initial_amount_proof_l2745_274500

/-- Proves that if an amount increases by 1/8th of itself each year for two years
    and becomes 64800, then the initial amount was 51200. -/
theorem initial_amount_proof (initial_amount : ℚ) : 
  (initial_amount * (9/8) * (9/8) = 64800) → initial_amount = 51200 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l2745_274500


namespace NUMINAMATH_CALUDE_pages_per_notebook_l2745_274560

/-- Given that James buys 2 notebooks, pays $5 in total, and each page costs 5 cents,
    prove that the number of pages in each notebook is 50. -/
theorem pages_per_notebook :
  let notebooks : ℕ := 2
  let total_cost : ℕ := 500  -- in cents
  let cost_per_page : ℕ := 5 -- in cents
  let total_pages : ℕ := total_cost / cost_per_page
  let pages_per_notebook : ℕ := total_pages / notebooks
  pages_per_notebook = 50 := by
sorry

end NUMINAMATH_CALUDE_pages_per_notebook_l2745_274560


namespace NUMINAMATH_CALUDE_fraction_equality_l2745_274517

theorem fraction_equality (x y : ℚ) (hx : x = 4/6) (hy : y = 5/8) :
  (6*x + 8*y) / (48*x*y) = 9/20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2745_274517


namespace NUMINAMATH_CALUDE_rice_distribution_l2745_274548

/-- Given 33/4 pounds of rice divided equally into 4 containers, 
    and 1 pound equals 16 ounces, prove that each container contains 15 ounces of rice. -/
theorem rice_distribution (total_weight : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) :
  total_weight = 33 / 4 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_weight * ounces_per_pound) / num_containers = 15 :=
by sorry

end NUMINAMATH_CALUDE_rice_distribution_l2745_274548


namespace NUMINAMATH_CALUDE_dore_change_l2745_274540

/-- The amount of change Mr. Doré receives after his purchase -/
def change (pants_cost shirt_cost tie_cost payment : ℕ) : ℕ :=
  payment - (pants_cost + shirt_cost + tie_cost)

/-- Theorem stating that Mr. Doré receives $2 in change -/
theorem dore_change : change 140 43 15 200 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dore_change_l2745_274540


namespace NUMINAMATH_CALUDE_chess_club_problem_l2745_274502

theorem chess_club_problem (total_members chess_players checkers_players both_players : ℕ) 
  (h1 : total_members = 70)
  (h2 : chess_players = 45)
  (h3 : checkers_players = 38)
  (h4 : both_players = 25) :
  total_members - (chess_players + checkers_players - both_players) = 12 := by
  sorry

end NUMINAMATH_CALUDE_chess_club_problem_l2745_274502


namespace NUMINAMATH_CALUDE_orange_trees_count_l2745_274504

theorem orange_trees_count (total_trees apple_trees : ℕ) 
  (h1 : total_trees = 74) 
  (h2 : apple_trees = 47) : 
  total_trees - apple_trees = 27 := by
  sorry

end NUMINAMATH_CALUDE_orange_trees_count_l2745_274504


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l2745_274534

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  tiles : ℕ
  perimeter : ℕ

/-- Represents the process of adding tiles to a configuration -/
def add_tiles (initial : TileConfiguration) (added : ℕ) : TileConfiguration :=
  { tiles := initial.tiles + added, perimeter := initial.perimeter + 1 }

theorem perimeter_after_adding_tiles 
  (initial : TileConfiguration) 
  (h1 : initial.tiles = 9)
  (h2 : initial.perimeter = 16) :
  ∃ (final : TileConfiguration), 
    final = add_tiles initial 3 ∧ 
    final.perimeter = 17 :=
sorry

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l2745_274534


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l2745_274575

theorem quadratic_roots_expression (a b : ℝ) : 
  (3 * a^2 + 2 * a - 2 = 0) →
  (3 * b^2 + 2 * b - 2 = 0) →
  (2 * a / (a^2 - b^2) - 1 / (a - b) = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l2745_274575


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2745_274573

def A : Set ℤ := {-1, 0, 1, 2, 3, 4, 5}

def B : Set ℤ := {b | ∃ n : ℤ, b = n^2 - 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2745_274573


namespace NUMINAMATH_CALUDE_f_composition_one_ninth_l2745_274551

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 2^x

theorem f_composition_one_ninth : f (f (1/9)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_one_ninth_l2745_274551


namespace NUMINAMATH_CALUDE_exists_x_exp_geq_one_plus_x_l2745_274595

theorem exists_x_exp_geq_one_plus_x : ∃ x : ℝ, x ≠ 0 ∧ Real.exp x ≥ 1 + x := by
  sorry

end NUMINAMATH_CALUDE_exists_x_exp_geq_one_plus_x_l2745_274595


namespace NUMINAMATH_CALUDE_max_sum_of_roots_l2745_274524

theorem max_sum_of_roots (x y z : ℝ) 
  (sum_eq : x + y + z = 0)
  (x_ge : x ≥ -1/2)
  (y_ge : y ≥ -1)
  (z_ge : z ≥ -3/2) :
  (∀ a b c : ℝ, a + b + c = 0 → a ≥ -1/2 → b ≥ -1 → c ≥ -3/2 → 
    Real.sqrt (4*a + 2) + Real.sqrt (4*b + 4) + Real.sqrt (4*c + 6) ≤ 
    Real.sqrt (4*x + 2) + Real.sqrt (4*y + 4) + Real.sqrt (4*z + 6)) ∧
  Real.sqrt (4*x + 2) + Real.sqrt (4*y + 4) + Real.sqrt (4*z + 6) = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_roots_l2745_274524


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l2745_274593

theorem quadratic_equation_root (m : ℝ) : 
  ((-1 : ℝ)^2 + m * (-1) - 4 = 0) → 
  ∃ (x : ℝ), x ≠ -1 ∧ x^2 + m*x - 4 = 0 ∧ x = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l2745_274593


namespace NUMINAMATH_CALUDE_least_integer_with_12_factors_and_consecutive_primes_l2745_274530

/-- A function that returns the number of positive factors of a given natural number -/
def num_factors (n : ℕ) : ℕ := sorry

/-- A function that checks if two prime numbers are consecutive -/
def are_consecutive_primes (p q : ℕ) : Prop := sorry

/-- The theorem stating that 36 is the least positive integer with exactly 12 factors
    and consecutive prime factors -/
theorem least_integer_with_12_factors_and_consecutive_primes :
  ∀ n : ℕ, n > 0 → num_factors n = 12 →
  (∃ p q : ℕ, n = p^2 * q^2 ∧ are_consecutive_primes p q) →
  n ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_with_12_factors_and_consecutive_primes_l2745_274530


namespace NUMINAMATH_CALUDE_symmetric_complex_product_l2745_274547

theorem symmetric_complex_product : 
  ∀ (z₁ z₂ : ℂ), 
  z₁ = 3 + 2*I → 
  (z₂.re = z₁.im ∧ z₂.im = z₁.re) → 
  z₁ * z₂ = 13 * I :=
by sorry

end NUMINAMATH_CALUDE_symmetric_complex_product_l2745_274547


namespace NUMINAMATH_CALUDE_sequence_length_l2745_274596

/-- Given a sequence of real numbers b₀, b₁, ..., bₙ with the following properties:
    - n is a positive integer
    - b₀ = 40
    - b₁ = 75
    - bₙ = 0
    - bₖ₊₁ = bₖ₋₁ - 4/bₖ for k = 1, 2, ..., n-1
    Then n = 751 -/
theorem sequence_length (b : ℕ → ℝ) (n : ℕ+) :
  b 0 = 40 →
  b 1 = 75 →
  b n = 0 →
  (∀ k : ℕ, 1 ≤ k → k < n → b (k + 1) = b (k - 1) - 4 / b k) →
  n = 751 := by
sorry

end NUMINAMATH_CALUDE_sequence_length_l2745_274596


namespace NUMINAMATH_CALUDE_zero_in_interval_implies_alpha_range_l2745_274589

theorem zero_in_interval_implies_alpha_range (α : ℝ) :
  (∃ x ∈ Set.Icc 0 1, x^2 + 2*α*x + 1 = 0) → α ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_implies_alpha_range_l2745_274589


namespace NUMINAMATH_CALUDE_second_customer_regular_hours_l2745_274554

/-- Represents the hourly rates and customer data for an online service -/
structure OnlineService where
  regularRate : ℝ
  premiumRate : ℝ
  customer1PremiumHours : ℝ
  customer1RegularHours : ℝ
  customer1TotalCharge : ℝ
  customer2PremiumHours : ℝ
  customer2TotalCharge : ℝ

/-- Calculates the number of regular hours for the second customer -/
def calculateCustomer2RegularHours (service : OnlineService) : ℝ :=
  -- Implementation not required for the statement
  sorry

/-- Theorem stating that the second customer spent 3 regular hours -/
theorem second_customer_regular_hours (service : OnlineService) 
  (h1 : service.customer1PremiumHours = 2)
  (h2 : service.customer1RegularHours = 9)
  (h3 : service.customer1TotalCharge = 28)
  (h4 : service.customer2PremiumHours = 3)
  (h5 : service.customer2TotalCharge = 27) :
  calculateCustomer2RegularHours service = 3 := by
  sorry

#eval "Lean 4 statement generated successfully."

end NUMINAMATH_CALUDE_second_customer_regular_hours_l2745_274554


namespace NUMINAMATH_CALUDE_smallest_two_digit_factor_of_5280_l2745_274597

theorem smallest_two_digit_factor_of_5280 :
  ∃ (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    a * b = 5280 ∧
    (∀ (x y : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ x * y = 5280 → min x y ≥ 66) :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_factor_of_5280_l2745_274597


namespace NUMINAMATH_CALUDE_art_kit_student_ratio_is_two_to_one_l2745_274578

/-- Represents the art class scenario --/
structure ArtClass where
  students : ℕ
  art_kits : ℕ
  total_artworks : ℕ

/-- Calculates the ratio of art kits to students --/
def art_kit_student_ratio (ac : ArtClass) : Rat :=
  ac.art_kits / ac.students

/-- Theorem stating the ratio of art kits to students is 2:1 --/
theorem art_kit_student_ratio_is_two_to_one (ac : ArtClass) 
  (h1 : ac.students = 10)
  (h2 : ac.art_kits = 20)
  (h3 : ac.total_artworks = 35)
  (h4 : ∃ (n : ℕ), 2 * n = ac.students ∧ 
       n * 3 + n * 4 = ac.total_artworks) : 
  art_kit_student_ratio ac = 2 := by
  sorry

end NUMINAMATH_CALUDE_art_kit_student_ratio_is_two_to_one_l2745_274578


namespace NUMINAMATH_CALUDE_problem_solution_l2745_274555

theorem problem_solution : (42 / (9 - 2 + 3)) * 7 = 29.4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2745_274555


namespace NUMINAMATH_CALUDE_triangle_condition_equivalent_to_m_gt_2_l2745_274577

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

-- Define the interval [0,2]
def I : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- Define the triangle inequality condition
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem triangle_condition_equivalent_to_m_gt_2 :
  (∀ m : ℝ, (∀ a b c : ℝ, a ∈ I → b ∈ I → c ∈ I → a ≠ b → b ≠ c → a ≠ c →
    triangle_inequality (f m a) (f m b) (f m c)) ↔ m > 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_condition_equivalent_to_m_gt_2_l2745_274577


namespace NUMINAMATH_CALUDE_investment_percentage_proof_l2745_274579

theorem investment_percentage_proof (total_investment : ℝ) (first_investment : ℝ) 
  (second_investment : ℝ) (second_rate : ℝ) (third_rate : ℝ) (yearly_income : ℝ) :
  total_investment = 10000 ∧ 
  first_investment = 4000 ∧ 
  second_investment = 3500 ∧ 
  second_rate = 0.04 ∧ 
  third_rate = 0.064 ∧ 
  yearly_income = 500 →
  ∃ x : ℝ, 
    x = 5 ∧ 
    first_investment * (x / 100) + second_investment * second_rate + 
    (total_investment - first_investment - second_investment) * third_rate = yearly_income :=
by sorry

end NUMINAMATH_CALUDE_investment_percentage_proof_l2745_274579


namespace NUMINAMATH_CALUDE_right_triangle_arm_square_l2745_274588

/-- In a right triangle with hypotenuse c and arms a and b, where c = a + 2,
    the square of b is equal to 4a + 4. -/
theorem right_triangle_arm_square (a c : ℝ) (h1 : c = a + 2) :
  ∃ b : ℝ, b^2 = 4*a + 4 ∧ a^2 + b^2 = c^2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_arm_square_l2745_274588


namespace NUMINAMATH_CALUDE_problem_solution_l2745_274505

theorem problem_solution (a : ℝ) : 3 ∈ ({a + 3, 2 * a + 1, a^2 + a + 1} : Set ℝ) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2745_274505


namespace NUMINAMATH_CALUDE_book_purchase_total_price_l2745_274552

/-- Given a total of 80 books, with 32 math books costing $4 each and the rest being history books costing $5 each, prove that the total price is $368. -/
theorem book_purchase_total_price :
  let total_books : ℕ := 80
  let math_books : ℕ := 32
  let math_book_price : ℕ := 4
  let history_book_price : ℕ := 5
  let history_books : ℕ := total_books - math_books
  let total_price : ℕ := math_books * math_book_price + history_books * history_book_price
  total_price = 368 := by
  sorry

end NUMINAMATH_CALUDE_book_purchase_total_price_l2745_274552


namespace NUMINAMATH_CALUDE_shortest_time_5x6_checkerboard_l2745_274550

/-- Represents a checkerboard with alternating black and white squares. -/
structure Checkerboard where
  rows : Nat
  cols : Nat
  squareSize : ℝ
  normalSpeed : ℝ
  slowSpeed : ℝ

/-- Calculates the shortest time to travel from bottom-left to top-right corner of the checkerboard. -/
def shortestTravelTime (board : Checkerboard) : ℝ :=
  sorry

/-- The theorem stating the shortest travel time for the specific checkerboard. -/
theorem shortest_time_5x6_checkerboard :
  let board : Checkerboard := {
    rows := 5
    cols := 6
    squareSize := 1
    normalSpeed := 2
    slowSpeed := 1
  }
  shortestTravelTime board = (1 + 5 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_shortest_time_5x6_checkerboard_l2745_274550


namespace NUMINAMATH_CALUDE_fred_balloons_l2745_274586

theorem fred_balloons (total sam mary : ℕ) (h1 : total = 18) (h2 : sam = 6) (h3 : mary = 7) :
  total - (sam + mary) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fred_balloons_l2745_274586


namespace NUMINAMATH_CALUDE_value_of_y_l2745_274542

theorem value_of_y : ∃ y : ℚ, (3 * y) / 7 = 14 ∧ y = 98 / 3 := by sorry

end NUMINAMATH_CALUDE_value_of_y_l2745_274542


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l2745_274531

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  swimmer : ℝ  -- Speed of the swimmer in still water
  stream : ℝ   -- Speed of the stream

/-- Calculates the effective speed given swimmer's speed and stream speed. -/
def effectiveSpeed (s : SwimmerSpeed) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Theorem: Given the conditions, the swimmer's speed in still water is 4 km/h. -/
theorem swimmer_speed_in_still_water :
  ∀ (s : SwimmerSpeed),
  (effectiveSpeed s true * 6 = 30) →   -- Downstream condition
  (effectiveSpeed s false * 6 = 18) →  -- Upstream condition
  s.swimmer = 4 := by
sorry

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l2745_274531


namespace NUMINAMATH_CALUDE_sticker_distribution_l2745_274561

/-- The number of ways to distribute n identical objects into k distinct bins -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The problem statement -/
theorem sticker_distribution :
  stars_and_bars 10 4 = Nat.choose 13 3 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l2745_274561


namespace NUMINAMATH_CALUDE_function_properties_l2745_274532

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.cos (ω * x)

theorem function_properties (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_eq : f ω (π/8) = f ω (5*π/8)) :
  (∃! (min max : ℝ), min ∈ Set.Ioo (π/8) (5*π/8) ∧ 
    max ∈ Set.Ioo (π/8) (5*π/8) ∧ 
    (∀ x ∈ Set.Ioo (π/8) (5*π/8), f ω x ≥ f ω min ∧ f ω x ≤ f ω max) →
    ω = 4) ∧
  (∃! (z₁ z₂ : ℝ), z₁ ∈ Set.Ioo (π/8) (5*π/8) ∧ 
    z₂ ∈ Set.Ioo (π/8) (5*π/8) ∧ 
    f ω z₁ = 0 ∧ f ω z₂ = 0 ∧ 
    (∀ x ∈ Set.Ioo (π/8) (5*π/8), f ω x = 0 → x = z₁ ∨ x = z₂) →
    ω = 10/3 ∨ ω = 4 ∨ ω = 6) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2745_274532


namespace NUMINAMATH_CALUDE_glorias_turtle_finish_time_l2745_274584

/-- The finish time of Gloria's turtle in the Key West Turtle Race -/
def glorias_turtle_time (gretas_time georges_time : ℕ) : ℕ :=
  2 * georges_time

/-- Theorem stating that Gloria's turtle finish time is 8 minutes -/
theorem glorias_turtle_finish_time :
  ∃ (gretas_time georges_time : ℕ),
    gretas_time = 6 ∧
    georges_time = gretas_time - 2 ∧
    glorias_turtle_time gretas_time georges_time = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_glorias_turtle_finish_time_l2745_274584


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2745_274525

theorem complex_fraction_simplification :
  let z : ℂ := (1 + I) / (3 - 4*I)
  z = -(1/25) + (7/25)*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2745_274525


namespace NUMINAMATH_CALUDE_sprint_jog_difference_value_l2745_274576

/-- The difference between Darnel's total sprinting distance and total jogging distance -/
def sprint_jog_difference : ℝ :=
  let sprint1 := 0.8932
  let sprint2 := 0.9821
  let sprint3 := 1.2534
  let jog1 := 0.7683
  let jog2 := 0.4356
  let jog3 := 0.6549
  (sprint1 + sprint2 + sprint3) - (jog1 + jog2 + jog3)

/-- Theorem stating that the difference between Darnel's total sprinting distance and total jogging distance is 1.2699 laps -/
theorem sprint_jog_difference_value : sprint_jog_difference = 1.2699 := by
  sorry

end NUMINAMATH_CALUDE_sprint_jog_difference_value_l2745_274576


namespace NUMINAMATH_CALUDE_max_value_sum_products_l2745_274528

theorem max_value_sum_products (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_constraint : a + b + c + d = 200) : 
  ∃ (max : ℝ), max = 10000 ∧ ∀ (x y z w : ℝ), 
    0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ 0 ≤ w ∧ x + y + z + w = 200 →
    x * y + y * z + z * w ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_products_l2745_274528


namespace NUMINAMATH_CALUDE_meal_payment_proof_l2745_274546

theorem meal_payment_proof :
  ∃! (p n d : ℕ),
    p + n + d = 50 ∧
    p + 5 * n + 10 * d = 240 ∧
    d = 10 :=
by sorry

end NUMINAMATH_CALUDE_meal_payment_proof_l2745_274546


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2745_274512

theorem inequality_system_solution (x : ℝ) :
  (x + 7) / 3 ≤ x + 3 ∧ 2 * (x + 1) < x + 3 → -1 ≤ x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2745_274512


namespace NUMINAMATH_CALUDE_arithmetic_equation_l2745_274527

theorem arithmetic_equation : (26.3 * 12 * 20) / 3 + 125 = 2229 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l2745_274527


namespace NUMINAMATH_CALUDE_correct_product_with_decimals_l2745_274583

theorem correct_product_with_decimals :
  let x : ℚ := 0.85
  let y : ℚ := 3.25
  let product_without_decimals : ℕ := 27625
  x * y = 2.7625 :=
by sorry

end NUMINAMATH_CALUDE_correct_product_with_decimals_l2745_274583
