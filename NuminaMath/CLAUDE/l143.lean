import Mathlib

namespace no_valid_A_l143_14395

theorem no_valid_A : ¬∃ (A : ℕ), A ≤ 9 ∧ 45 % A = 0 ∧ (456204 + A * 10) % 8 = 0 := by
  sorry

end no_valid_A_l143_14395


namespace parabola_position_l143_14376

/-- Represents a quadratic function of the form ax^2 + bx + c --/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem stating properties of a specific type of parabola --/
theorem parabola_position (f : QuadraticFunction) 
  (ha : f.a > 0) (hb : f.b > 0) (hc : f.c < 0) : 
  f.c < 0 ∧ -f.b / (2 * f.a) < 0 := by
  sorry

#check parabola_position

end parabola_position_l143_14376


namespace point_on_x_axis_point_in_second_quadrant_l143_14302

-- Define point P
def P (a : ℝ) : ℝ × ℝ := (2*a - 2, a + 5)

-- Part 1
theorem point_on_x_axis (a : ℝ) :
  P a = (-12, 0) ↔ (P a).2 = 0 :=
sorry

-- Part 2
theorem point_in_second_quadrant (a : ℝ) :
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ |(P a).1| = |(P a).2| →
  a^2023 + 2023 = 2022 :=
sorry

end point_on_x_axis_point_in_second_quadrant_l143_14302


namespace min_victory_points_l143_14347

/-- Represents the point system for a football competition --/
structure PointSystem where
  victory_points : ℕ
  draw_points : ℕ
  defeat_points : ℕ

/-- Represents the state of a team's performance --/
structure TeamPerformance where
  total_matches : ℕ
  played_matches : ℕ
  current_points : ℕ
  target_points : ℕ
  min_victories_needed : ℕ

/-- The theorem to prove --/
theorem min_victory_points (ps : PointSystem) (tp : TeamPerformance) : 
  ps.draw_points = 1 ∧ 
  ps.defeat_points = 0 ∧
  tp.total_matches = 20 ∧ 
  tp.played_matches = 5 ∧
  tp.current_points = 14 ∧
  tp.target_points = 40 ∧
  tp.min_victories_needed = 6 →
  ps.victory_points ≥ 3 :=
by sorry

end min_victory_points_l143_14347


namespace inequality_system_solution_l143_14384

theorem inequality_system_solution (x : ℝ) :
  (3 * x - (x - 2) ≥ 6) ∧ (x + 1 > (4 * x - 1) / 3) → 2 ≤ x ∧ x < 4 := by
  sorry

end inequality_system_solution_l143_14384


namespace geometric_sequence_sum_l143_14385

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 1 = 3 →  -- First term is 3
  a 1 + a 2 + a 3 = 21 →  -- Sum of first three terms is 21
  a 3 + a 4 + a 5 = 84 := by
  sorry

end geometric_sequence_sum_l143_14385


namespace geometric_sequence_common_ratio_l143_14392

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (h_geometric : is_geometric_sequence a) 
  (h_condition : 8 * a 2 + a 5 = 0) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n) ∧ q = -2 :=
sorry

end geometric_sequence_common_ratio_l143_14392


namespace bcm_hens_count_l143_14397

/-- Given a farm with chickens, calculate the number of Black Copper Marans (BCM) hens -/
theorem bcm_hens_count (total_chickens : ℕ) (bcm_percentage : ℚ) (bcm_hen_percentage : ℚ) : 
  total_chickens = 100 →
  bcm_percentage = 1/5 →
  bcm_hen_percentage = 4/5 →
  (total_chickens : ℚ) * bcm_percentage * bcm_hen_percentage = 16 := by
sorry

end bcm_hens_count_l143_14397


namespace line_equation_through_point_with_angle_45_l143_14348

/-- The equation of a line passing through (-4, 3) with a slope angle of 45° -/
theorem line_equation_through_point_with_angle_45 :
  ∃ (f : ℝ → ℝ),
    (∀ x y, f x = y ↔ x - y + 7 = 0) ∧
    f (-4) = 3 ∧
    (∀ x, (f x - f (-4)) / (x - (-4)) = 1) :=
by sorry

end line_equation_through_point_with_angle_45_l143_14348


namespace multiplication_closed_in_P_l143_14326

def P : Set ℕ := {n : ℕ | ∃ m : ℕ, m > 0 ∧ n = m^2}

theorem multiplication_closed_in_P : 
  ∀ a b : ℕ, a ∈ P → b ∈ P → (a * b) ∈ P := by
  sorry

end multiplication_closed_in_P_l143_14326


namespace range_of_a_l143_14387

-- Define propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x a : ℝ) : Prop := x > a

-- Define the property that ¬q is sufficient but not necessary for ¬p
def not_q_sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, ¬(q x a) → ¬(p x)) ∧ ¬(∀ x, ¬(p x) → ¬(q x a))

-- Theorem statement
theorem range_of_a (a : ℝ) (h : not_q_sufficient_not_necessary a) : a ≥ 1 :=
sorry

end range_of_a_l143_14387


namespace dividend_remainder_proof_l143_14311

theorem dividend_remainder_proof (D d q r : ℕ) : 
  D = 18972 → d = 526 → q = 36 → D = d * q + r → r = 36 := by
  sorry

end dividend_remainder_proof_l143_14311


namespace ratio_BL_LC_l143_14325

/-- A square with side length 5 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 0) ∧ B = (5, 0) ∧ C = (5, 5) ∧ D = (0, 5))

/-- A point K on side AB of the square -/
def K : ℝ × ℝ := (3, 0)

/-- A point L on side BC of the square -/
def L (y : ℝ) : ℝ × ℝ := (5, y)

/-- Distance function between a point and a line -/
def distance_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ := sorry

/-- The theorem to be proved -/
theorem ratio_BL_LC (ABCD : Square) :
  ∃ y : ℝ, 0 ≤ y ∧ y ≤ 5 ∧
  distance_to_line K (fun p => p.2 = (y - 5) / 5 * p.1 + 5) = 3 →
  (5 - y) / y = 8 / 7 := by sorry

end ratio_BL_LC_l143_14325


namespace f_ratio_range_l143_14362

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the derivative of f
noncomputable def f' : ℝ → ℝ := sorry

-- State the theorem
theorem f_ratio_range :
  (∀ x : ℝ, f' x - f x = 2 * x * Real.exp x) →
  f 0 = 1 →
  ∀ x : ℝ, x > 0 → 1 < (f' x) / (f x) ∧ (f' x) / (f x) ≤ 2 :=
by sorry

end f_ratio_range_l143_14362


namespace b_remaining_work_days_l143_14371

-- Define the work rates and time periods
def a_rate : ℚ := 1 / 4
def b_rate : ℚ := 1 / 14
def initial_work_days : ℕ := 2

-- Theorem statement
theorem b_remaining_work_days :
  let total_work : ℚ := 1
  let combined_rate : ℚ := a_rate + b_rate
  let work_done_together : ℚ := combined_rate * initial_work_days
  let remaining_work : ℚ := total_work - work_done_together
  (remaining_work / b_rate : ℚ) = 5 := by
  sorry

end b_remaining_work_days_l143_14371


namespace book_price_proof_l143_14349

theorem book_price_proof (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 63)
  (h2 : profit_percentage = 5) :
  ∃ original_price : ℝ, 
    original_price * (1 + profit_percentage / 100) = selling_price ∧ 
    original_price = 60 := by
  sorry

end book_price_proof_l143_14349


namespace range_of_m_l143_14314

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -a * log x + x + (1 - a) / x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := exp x + m * x^2 - 2 * exp 2 - 3

theorem range_of_m :
  ∀ m : ℝ, (∃ x₂ : ℝ, x₂ ≥ 1 ∧ ∀ x₁ : ℝ, x₁ ≥ 1 → g m x₂ ≤ f (exp 2 + 1) x₁) ↔ m ≤ exp 2 - exp 1 :=
by sorry

end range_of_m_l143_14314


namespace like_terms_exponent_sum_l143_14300

/-- Given two algebraic terms are like terms, prove that the sum of their exponents is 5 -/
theorem like_terms_exponent_sum (a b : ℝ) (m n : ℕ) : 
  (∃ (k : ℝ), k * a^(2*m) * b^3 = 5 * a^6 * b^(n+1)) → m + n = 5 := by
  sorry

end like_terms_exponent_sum_l143_14300


namespace number_count_l143_14365

theorem number_count (n : ℕ) 
  (h1 : (n : ℝ) * 30 = (4 : ℝ) * 25 + (3 : ℝ) * 35 - 25)
  (h2 : n > 4) : n = 6 := by
  sorry

end number_count_l143_14365


namespace smallest_non_odd_unit_proof_l143_14377

/-- The set of possible units digits for odd numbers -/
def odd_units : Set Nat := {1, 3, 5, 7, 9}

/-- A number is odd if and only if its units digit is in the odd_units set -/
def is_odd (n : Nat) : Prop := n % 10 ∈ odd_units

/-- The smallest digit not in the units place of an odd number -/
def smallest_non_odd_unit : Nat := 0

theorem smallest_non_odd_unit_proof :
  (∀ n : Nat, is_odd n → smallest_non_odd_unit ≠ n % 10) ∧
  (∀ d : Nat, d < smallest_non_odd_unit → ∃ n : Nat, is_odd n ∧ d = n % 10) :=
sorry

end smallest_non_odd_unit_proof_l143_14377


namespace divisibility_of_3_105_plus_4_105_l143_14333

theorem divisibility_of_3_105_plus_4_105 :
  let n : ℕ := 3^105 + 4^105
  (∃ k : ℕ, n = 13 * k) ∧
  (∃ k : ℕ, n = 49 * k) ∧
  (∃ k : ℕ, n = 181 * k) ∧
  (∃ k : ℕ, n = 379 * k) ∧
  (∀ k : ℕ, n ≠ 5 * k) ∧
  (∀ k : ℕ, n ≠ 11 * k) := by
  sorry

end divisibility_of_3_105_plus_4_105_l143_14333


namespace largest_valid_pair_l143_14343

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ b ∧ b ≤ 100 ∧ is_integer ((a + b) * (a + b + 1) / (a * b : ℚ))

theorem largest_valid_pair :
  ∀ a b : ℕ, valid_pair a b →
    b ≤ 90 ∧
    (b = 90 → a ≤ 35) ∧
    valid_pair 35 90
  := by sorry

end largest_valid_pair_l143_14343


namespace x_positive_sufficient_not_necessary_for_x_squared_positive_l143_14396

theorem x_positive_sufficient_not_necessary_for_x_squared_positive :
  (∃ x : ℝ, x > 0 → x^2 > 0) ∧ 
  (∃ x : ℝ, x^2 > 0 ∧ ¬(x > 0)) := by
  sorry

end x_positive_sufficient_not_necessary_for_x_squared_positive_l143_14396


namespace three_subset_M_l143_14386

def M : Set ℤ := {x | ∃ n : ℤ, x = 4 * n - 1}

theorem three_subset_M : {3} ⊆ M := by
  sorry

end three_subset_M_l143_14386


namespace jays_family_female_guests_l143_14346

theorem jays_family_female_guests 
  (total_guests : ℕ) 
  (female_percentage : ℚ) 
  (jays_family_percentage : ℚ) 
  (h1 : total_guests = 240)
  (h2 : female_percentage = 60 / 100)
  (h3 : jays_family_percentage = 50 / 100) :
  ↑total_guests * female_percentage * jays_family_percentage = 72 := by
  sorry

end jays_family_female_guests_l143_14346


namespace g_of_three_equals_five_l143_14344

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define g in terms of f
def g (x : ℝ) : ℝ := f (x - 2)

-- Theorem to prove
theorem g_of_three_equals_five : g 3 = 5 := by
  sorry

end g_of_three_equals_five_l143_14344


namespace cyclist_problem_l143_14309

theorem cyclist_problem (v t : ℝ) 
  (h1 : (v + 3) * (t - 1) = v * t)
  (h2 : (v - 2) * (t + 1) = v * t) :
  v * t = 60 ∧ v = 12 ∧ t = 5 :=
by sorry

end cyclist_problem_l143_14309


namespace relationship_abc_l143_14356

theorem relationship_abc (a b c : ℕ) : 
  a = 2^555 → b = 3^444 → c = 6^222 → a < c ∧ c < b := by
  sorry

end relationship_abc_l143_14356


namespace rational_equation_proof_l143_14366

theorem rational_equation_proof (m n : ℚ) 
  (h1 : 3 * m + 2 * n = 0) 
  (h2 : m * n ≠ 0) : 
  m / n - n / m = 5 / 6 := by
sorry

end rational_equation_proof_l143_14366


namespace ellipse_cartesian_eq_l143_14324

def ellipse_eq (x y : ℝ) : Prop :=
  ∃ t : ℝ, x = (3 * (Real.sin t - 2)) / (3 - Real.cos t) ∧
            y = (4 * (Real.cos t - 6)) / (3 - Real.cos t)

theorem ellipse_cartesian_eq :
  ∀ x y : ℝ, ellipse_eq x y ↔ 9*x^2 + 36*x*y + 9*y^2 + 216*x + 432*y + 1440 = 0 :=
by sorry

end ellipse_cartesian_eq_l143_14324


namespace max_vertices_1000_triangles_l143_14335

/-- The maximum number of distinct points that can be vertices of 1000 triangles in a quadrilateral -/
def max_distinct_vertices (num_triangles : ℕ) (quadrilateral_angle_sum : ℕ) : ℕ :=
  let triangle_angle_sum := 180
  let total_angle_sum := num_triangles * triangle_angle_sum
  let excess_angle_sum := total_angle_sum - quadrilateral_angle_sum
  let side_vertices := excess_angle_sum / triangle_angle_sum
  let original_vertices := 4
  side_vertices + original_vertices

/-- Theorem stating that the maximum number of distinct vertices is 1002 -/
theorem max_vertices_1000_triangles :
  max_distinct_vertices 1000 360 = 1002 := by
  sorry

end max_vertices_1000_triangles_l143_14335


namespace inequality_proof_l143_14316

theorem inequality_proof (a₁ a₂ a₃ : ℝ) 
  (h₁ : 0 ≤ a₁) (h₂ : 0 ≤ a₂) (h₃ : 0 ≤ a₃) 
  (h_sum : a₁ + a₂ + a₃ = 1) : 
  a₁ * Real.sqrt a₂ + a₂ * Real.sqrt a₃ + a₃ * Real.sqrt a₁ ≤ 1 / Real.sqrt 3 := by
sorry

end inequality_proof_l143_14316


namespace geometric_sequence_formula_l143_14368

theorem geometric_sequence_formula (a : ℕ → ℝ) (n : ℕ) :
  (∀ k, a (k + 1) = 3 * a k) →  -- Geometric sequence with common ratio 3
  a 1 = 4 →                     -- First term is 4
  a n = 4 * 3^(n - 1) :=        -- General formula
by
  sorry

end geometric_sequence_formula_l143_14368


namespace ethans_candles_weight_l143_14329

/-- The combined weight of Ethan's candles -/
def combined_weight (total_candles : ℕ) (beeswax_per_candle : ℕ) (coconut_oil_per_candle : ℕ) : ℕ :=
  total_candles * (beeswax_per_candle + coconut_oil_per_candle)

/-- Theorem: The combined weight of Ethan's candles is 63 ounces -/
theorem ethans_candles_weight :
  combined_weight (10 - 3) 8 1 = 63 := by
  sorry

end ethans_candles_weight_l143_14329


namespace intersection_angle_l143_14388

theorem intersection_angle (φ : Real) 
  (h1 : 0 ≤ φ) (h2 : φ < π)
  (h3 : 2 * Real.cos (π/3) = 2 * Real.sin (2 * (π/3) + φ)) :
  φ = π/6 := by sorry

end intersection_angle_l143_14388


namespace simplify_expression_l143_14345

theorem simplify_expression (a : ℝ) : (a^2)^3 + 3*a^4*a^2 - a^8/a^2 = 3*a^6 := by
  sorry

end simplify_expression_l143_14345


namespace product_expansion_l143_14354

theorem product_expansion (x : ℝ) : (x^2 - 3*x + 3) * (x^2 + 3*x + 1) = x^4 - 5*x^2 + 6*x + 3 := by
  sorry

end product_expansion_l143_14354


namespace table_tennis_matches_l143_14361

theorem table_tennis_matches (player1_matches player2_matches : ℕ) 
  (h1 : player1_matches = 10) 
  (h2 : player2_matches = 21) : ℕ := by
  -- The number of matches the third player played
  sorry

#check table_tennis_matches

end table_tennis_matches_l143_14361


namespace tangent_line_condition_function_upper_bound_inequality_for_reciprocal_product_l143_14369

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := x / Real.exp x - a * x * Real.log x

theorem tangent_line_condition (a : ℝ) : 
  (deriv (f a)) 1 = -1 → a = 1 := by sorry

theorem function_upper_bound (x : ℝ) (hx : x > 0) : 
  x / Real.exp x - x * Real.log x < 2 / Real.exp 1 := by sorry

theorem inequality_for_reciprocal_product (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m * n = 1) :
  1 / Real.exp m + 1 / Real.exp n < 2 * (m + n) := by sorry

end tangent_line_condition_function_upper_bound_inequality_for_reciprocal_product_l143_14369


namespace square_ratio_side_lengths_l143_14327

theorem square_ratio_side_lengths : 
  ∃ (a b c : ℕ), 
    (a * a * b : ℚ) / (c * c) = 50 / 98 ∧ 
    a = 5 ∧ 
    b = 1 ∧ 
    c = 7 := by
  sorry

end square_ratio_side_lengths_l143_14327


namespace perpendicular_line_through_point_l143_14383

/-- Given a line L1 with equation 2x - 3y + 4 = 0 and a point P (-1, 2),
    prove that the line L2 with equation 3x + 2y - 1 = 0 passes through P
    and is perpendicular to L1. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 2 * x - 3 * y + 4 = 0
  let P : ℝ × ℝ := (-1, 2)
  let L2 : ℝ → ℝ → Prop := λ x y => 3 * x + 2 * y - 1 = 0
  (L2 P.1 P.2) ∧ 
  (∀ (x1 y1 x2 y2 : ℝ), L1 x1 y1 → L1 x2 y2 → x1 ≠ x2 → 
    (x2 - x1) * (P.1 - x1) + (y2 - y1) * (P.2 - y1) = 0) := by
  sorry

end perpendicular_line_through_point_l143_14383


namespace sum_of_three_numbers_l143_14380

theorem sum_of_three_numbers : ∀ (n₁ n₂ n₃ : ℕ),
  n₂ = 72 →
  n₁ = 2 * n₂ →
  n₃ = n₁ / 3 →
  n₁ + n₂ + n₃ = 264 := by
  sorry

end sum_of_three_numbers_l143_14380


namespace power_equality_implies_exponent_l143_14351

theorem power_equality_implies_exponent (m : ℝ) : (81 : ℝ) ^ (1/4 : ℝ) = 3^m → m = 1 := by
  sorry

end power_equality_implies_exponent_l143_14351


namespace toms_balloons_l143_14310

theorem toms_balloons (sara_balloons : ℕ) (total_balloons : ℕ) 
  (h1 : sara_balloons = 8)
  (h2 : total_balloons = 17) :
  total_balloons - sara_balloons = 9 := by
  sorry

end toms_balloons_l143_14310


namespace joans_attendance_l143_14389

/-- The number of football games Joan attended -/
structure FootballAttendance where
  total : ℕ
  lastYear : ℕ
  thisYear : ℕ

/-- Theorem stating that Joan's attendance this year is 4 games -/
theorem joans_attendance (joan : FootballAttendance) 
  (h1 : joan.total = 13)
  (h2 : joan.lastYear = 9)
  (h3 : joan.total = joan.lastYear + joan.thisYear) :
  joan.thisYear = 4 := by
  sorry

end joans_attendance_l143_14389


namespace smallest_equal_burgers_and_buns_l143_14303

theorem smallest_equal_burgers_and_buns :
  ∃ n : ℕ+, (∀ k : ℕ+, (∃ m : ℕ+, 5 * k = 7 * m) → n ≤ k) ∧ (∃ m : ℕ+, 5 * n = 7 * m) :=
by sorry

end smallest_equal_burgers_and_buns_l143_14303


namespace complement_A_intersect_B_l143_14331

-- Define the sets A and B
def A : Set ℝ := {x | x > -1}
def B : Set ℝ := {-2, -1, 0, 1}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {-2, -1} := by sorry

end complement_A_intersect_B_l143_14331


namespace equation_solutions_l143_14336

theorem equation_solutions :
  ∀ x : ℝ, 
    Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6 ↔ 
    x = 2 ∨ x = -2 := by
  sorry

end equation_solutions_l143_14336


namespace largest_root_of_f_cubed_l143_14319

/-- The function f(x) = x^2 + 12x + 30 -/
def f (x : ℝ) : ℝ := x^2 + 12*x + 30

/-- The composition of f with itself three times -/
def f_cubed (x : ℝ) : ℝ := f (f (f x))

/-- The largest real root of f(f(f(x))) = 0 -/
noncomputable def largest_root : ℝ := -6 + (6 : ℝ)^(1/8)

theorem largest_root_of_f_cubed :
  (f_cubed largest_root = 0) ∧
  (∀ x : ℝ, f_cubed x = 0 → x ≤ largest_root) :=
by sorry

end largest_root_of_f_cubed_l143_14319


namespace point_satisfies_conditions_l143_14332

/-- A point on a line with equal distance to coordinate axes -/
def point_on_line_equal_distance (x y : ℝ) : Prop :=
  y = -2 * x + 2 ∧ (x = y ∨ x = -y)

/-- The point (2/3, 2/3) satisfies the conditions -/
theorem point_satisfies_conditions : point_on_line_equal_distance (2/3) (2/3) := by
  sorry

end point_satisfies_conditions_l143_14332


namespace probability_two_girls_l143_14381

theorem probability_two_girls (total : Nat) (girls : Nat) (selected : Nat) : 
  total = 6 → girls = 4 → selected = 2 →
  (Nat.choose girls selected : Rat) / (Nat.choose total selected : Rat) = 2/5 := by
sorry

end probability_two_girls_l143_14381


namespace sam_distance_l143_14322

/-- Given that Marguerite drove 150 miles in 3 hours, and Sam drove for 4 hours at the same average rate as Marguerite, prove that Sam drove 200 miles. -/
theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ)
  (h1 : marguerite_distance = 150)
  (h2 : marguerite_time = 3)
  (h3 : sam_time = 4) :
  (marguerite_distance / marguerite_time) * sam_time = 200 :=
by sorry

end sam_distance_l143_14322


namespace base_conversion_sum_l143_14398

/-- Converts a number from base 8 to base 10 -/
def base8To10 (n : Nat) : Nat := sorry

/-- Converts a number from base 13 to base 10 -/
def base13To10 (n : Nat) : Nat := sorry

/-- Represents the value of C in base 13 -/
def C : Nat := 12

/-- Represents the value of D in base 13 (adjusted to 0) -/
def D : Nat := 0

theorem base_conversion_sum :
  base8To10 367 + base13To10 (4 * 13^2 + C * 13 + D) = 1079 := by sorry

end base_conversion_sum_l143_14398


namespace infinite_series_sum_l143_14301

open Real

noncomputable def series_sum (n : ℕ) : ℝ := 3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1))

theorem infinite_series_sum :
  (∑' n, series_sum n) = 1/4 := by sorry

end infinite_series_sum_l143_14301


namespace prob_blue_twelve_sided_die_l143_14394

/-- A die with a specified number of sides and blue faces -/
structure Die where
  sides : ℕ
  blue_faces : ℕ
  blue_faces_le_sides : blue_faces ≤ sides

/-- The probability of rolling a blue face on a given die -/
def prob_blue (d : Die) : ℚ :=
  d.blue_faces / d.sides

/-- Theorem: The probability of rolling a blue face on a 12-sided die with 4 blue faces is 1/3 -/
theorem prob_blue_twelve_sided_die :
  ∃ d : Die, d.sides = 12 ∧ d.blue_faces = 4 ∧ prob_blue d = 1/3 := by
  sorry

end prob_blue_twelve_sided_die_l143_14394


namespace inequality_solution_implies_m_value_l143_14338

theorem inequality_solution_implies_m_value (m : ℝ) :
  (∀ x : ℝ, (0 < x ∧ x < 2) ↔ (-1/2 * x^2 + 2*x > m*x)) →
  m = 1 := by
sorry

end inequality_solution_implies_m_value_l143_14338


namespace rider_pedestrian_problem_l143_14373

/-- A problem about a rider and a pedestrian traveling between two points. -/
theorem rider_pedestrian_problem
  (total_time : ℝ) -- Total time for the rider's journey
  (time_difference : ℝ) -- Time difference between rider and pedestrian arriving at B
  (meeting_distance : ℝ) -- Distance from B where rider meets pedestrian on return
  (h_total_time : total_time = 100 / 60) -- Total time is 1 hour 40 minutes (100 minutes)
  (h_time_difference : time_difference = 50 / 60) -- Rider arrives 50 minutes earlier
  (h_meeting_distance : meeting_distance = 2) -- They meet 2 km from B
  : ∃ (distance speed_rider speed_pedestrian : ℝ),
    distance = 6 ∧ 
    speed_rider = 7.2 ∧ 
    speed_pedestrian = 3.6 :=
by sorry

end rider_pedestrian_problem_l143_14373


namespace problem_statement_l143_14390

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (∀ x y, x > 0 ∧ y > 0 ∧ x + y = 1 → x * y ≤ a * b) ∧
  (a^2 + b^2 ≥ 1/2) ∧
  (4/a + 1/b ≥ 9) := by
sorry

end problem_statement_l143_14390


namespace problem_solution_l143_14315

-- Define the set of integers
def Z : Set ℝ := {x : ℝ | ∃ n : ℤ, x = n}

-- Define the set of x satisfying the conditions
def S : Set ℝ := {x : ℝ | (|x - 1| < 2 ∨ x ∉ Z) ∧ x ∈ Z}

-- Define the target set
def T : Set ℝ := {0, 1, 2}

-- Theorem statement
theorem problem_solution : S = T := by
  sorry

end problem_solution_l143_14315


namespace sum_in_base4_l143_14306

-- Define a function to convert from base 10 to base 4
def toBase4 (n : ℕ) : List ℕ :=
  sorry

-- Define a function to convert from base 4 to base 10
def fromBase4 (l : List ℕ) : ℕ :=
  sorry

theorem sum_in_base4 : 
  toBase4 (195 + 61) = [1, 0, 0, 0, 0] :=
sorry

end sum_in_base4_l143_14306


namespace smallest_integer_satisfying_inequality_l143_14342

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, (7 - 4 * x > 25) → x ≥ -5 ∧ (7 - 4 * (-5) > 25) :=
by sorry

end smallest_integer_satisfying_inequality_l143_14342


namespace arithmetic_sequence_mean_median_l143_14393

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) :=
  b * b = a * c

theorem arithmetic_sequence_mean_median
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_d_nonzero : d ≠ 0)
  (h_a3 : a 3 = 8)
  (h_geom : geometric_sequence (a 1) (a 3) (a 7)) :
  let mean := (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10) / 10
  let median := (a 5 + a 6) / 2
  mean = 13 ∧ median = 13 := by
  sorry

end arithmetic_sequence_mean_median_l143_14393


namespace solution_set_theorem_range_of_m_l143_14320

-- Define the function f(x) = |x - 2|
def f (x : ℝ) : ℝ := |x - 2|

-- Theorem for the solution set of f(x) + f(2x + 1) ≥ 6
theorem solution_set_theorem (x : ℝ) :
  f x + f (2 * x + 1) ≥ 6 ↔ x ∈ Set.Iic (-1) ∪ Set.Ici 3 :=
sorry

-- Theorem for the range of m
theorem range_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x : ℝ, f (x - m) - (-x) ≤ 4/a + 1/b) →
  -13 ≤ m ∧ m ≤ 5 :=
sorry

end solution_set_theorem_range_of_m_l143_14320


namespace central_angle_common_chord_l143_14399

/-- The central angle corresponding to the common chord of two circles -/
theorem central_angle_common_chord (x y : ℝ) : 
  let circle1 := {(x, y) | (x - 2)^2 + y^2 = 4}
  let circle2 := {(x, y) | x^2 + (y - 2)^2 = 4}
  let center1 := (2, 0)
  let center2 := (0, 2)
  let radius := 2
  let center_distance := Real.sqrt ((2 - 0)^2 + (0 - 2)^2)
  let chord_distance := center_distance / 2
  let cos_half_angle := chord_distance / radius
  let central_angle := 2 * Real.arccos cos_half_angle
  central_angle = π / 2 := by
  sorry

end central_angle_common_chord_l143_14399


namespace prime_representation_mod_24_l143_14337

theorem prime_representation_mod_24 (p : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  (∃ x y : ℤ, (p : ℤ) = 2 * x^2 + 3 * y^2) ↔ (p % 24 = 5 ∨ p % 24 = 11) :=
by sorry

end prime_representation_mod_24_l143_14337


namespace permutations_of_seven_distinct_objects_l143_14359

theorem permutations_of_seven_distinct_objects : Nat.factorial 7 = 5040 := by
  sorry

end permutations_of_seven_distinct_objects_l143_14359


namespace evaluate_expression_l143_14374

theorem evaluate_expression : 3000 * (3000 ^ 3000) = 3000 ^ 3001 := by
  sorry

end evaluate_expression_l143_14374


namespace ray_walks_11_blocks_home_l143_14352

/-- Represents Ray's dog walking routine -/
structure DogWalk where
  trips_per_day : ℕ
  total_blocks_per_day : ℕ
  blocks_to_park : ℕ
  blocks_to_school : ℕ

/-- Calculates the number of blocks Ray walks to get back home -/
def blocks_to_home (dw : DogWalk) : ℕ :=
  (dw.total_blocks_per_day / dw.trips_per_day) - (dw.blocks_to_park + dw.blocks_to_school)

/-- Theorem stating that Ray walks 11 blocks to get back home -/
theorem ray_walks_11_blocks_home :
  ∃ (dw : DogWalk),
    dw.trips_per_day = 3 ∧
    dw.total_blocks_per_day = 66 ∧
    dw.blocks_to_park = 4 ∧
    dw.blocks_to_school = 7 ∧
    blocks_to_home dw = 11 := by
  sorry

end ray_walks_11_blocks_home_l143_14352


namespace liya_number_preference_l143_14355

theorem liya_number_preference (n : ℕ) : 
  (n % 3 = 0) ∧ (n % 10 = 0) → n % 10 = 0 := by
sorry

end liya_number_preference_l143_14355


namespace pencils_theorem_l143_14370

def pencils_problem (monday tuesday wednesday thursday friday : ℕ) : Prop :=
  let total_tuesday := monday + tuesday
  let total_wednesday := total_tuesday + 3 * tuesday - 20
  let total_thursday := total_wednesday + wednesday / 2
  let total_friday := total_thursday + 2 * monday
  let final_total := total_friday - 50
  (monday = 35) ∧
  (tuesday = 42) ∧
  (wednesday = 3 * tuesday) ∧
  (thursday = wednesday / 2) ∧
  (friday = 2 * monday) ∧
  (final_total = 266)

theorem pencils_theorem :
  ∃ (monday tuesday wednesday thursday friday : ℕ),
    pencils_problem monday tuesday wednesday thursday friday :=
by
  sorry

end pencils_theorem_l143_14370


namespace point_on_x_axis_point_neg_two_zero_on_x_axis_l143_14372

/-- A point lies on the x-axis if and only if its y-coordinate is 0 -/
theorem point_on_x_axis (x y : ℝ) : 
  (x, y) ∈ {p : ℝ × ℝ | p.2 = 0} ↔ y = 0 := by sorry

/-- The point (-2, 0) lies on the x-axis -/
theorem point_neg_two_zero_on_x_axis : 
  (-2, 0) ∈ {p : ℝ × ℝ | p.2 = 0} := by sorry

end point_on_x_axis_point_neg_two_zero_on_x_axis_l143_14372


namespace light_bulb_configurations_l143_14353

/-- The number of light bulbs -/
def num_bulbs : ℕ := 5

/-- The number of states each bulb can have (on or off) -/
def states_per_bulb : ℕ := 2

/-- The total number of possible lighting configurations -/
def total_configurations : ℕ := states_per_bulb ^ num_bulbs

theorem light_bulb_configurations :
  total_configurations = 32 :=
by sorry

end light_bulb_configurations_l143_14353


namespace sin_cos_sum_l143_14360

theorem sin_cos_sum (θ : ℝ) (h : Real.sin θ ^ 3 + Real.cos θ ^ 3 = 11/16) : 
  Real.sin θ + Real.cos θ = 1/2 := by
sorry

end sin_cos_sum_l143_14360


namespace soda_per_syrup_box_l143_14379

/-- Given a convenience store that sells soda and buys syrup boxes, this theorem proves
    the number of gallons of soda that can be made from one box of syrup. -/
theorem soda_per_syrup_box 
  (total_soda : ℝ) 
  (box_cost : ℝ) 
  (total_syrup_cost : ℝ) 
  (h1 : total_soda = 180) 
  (h2 : box_cost = 40) 
  (h3 : total_syrup_cost = 240) : 
  total_soda / (total_syrup_cost / box_cost) = 30 := by
sorry

end soda_per_syrup_box_l143_14379


namespace sushi_eating_orders_l143_14318

/-- Represents a 2 × 3 grid of sushi pieces -/
structure SushiGrid :=
  (pieces : Fin 6 → Bool)

/-- Checks if a piece is adjacent to at most two other pieces -/
def isEatable (grid : SushiGrid) (pos : Fin 6) : Bool :=
  sorry

/-- Generates all valid eating orders for a given SushiGrid -/
def validEatingOrders (grid : SushiGrid) : List (List (Fin 6)) :=
  sorry

/-- The number of valid eating orders for a 2 × 3 sushi grid -/
def numValidOrders : Nat :=
  sorry

theorem sushi_eating_orders :
  numValidOrders = 360 :=
sorry

end sushi_eating_orders_l143_14318


namespace pumpkins_left_l143_14304

theorem pumpkins_left (grown : ℕ) (eaten : ℕ) (h1 : grown = 43) (h2 : eaten = 23) :
  grown - eaten = 20 := by
  sorry

end pumpkins_left_l143_14304


namespace cost_per_sqm_intersecting_roads_l143_14364

/-- The cost per square meter for traveling two intersecting roads on a rectangular lawn. -/
theorem cost_per_sqm_intersecting_roads 
  (lawn_length : ℝ) 
  (lawn_width : ℝ) 
  (road_width : ℝ) 
  (total_cost : ℝ) : 
  lawn_length = 80 ∧ 
  lawn_width = 40 ∧ 
  road_width = 10 ∧ 
  total_cost = 3300 → 
  (total_cost / ((lawn_length * road_width + lawn_width * road_width) - road_width * road_width)) = 3 := by
sorry

end cost_per_sqm_intersecting_roads_l143_14364


namespace inequality_proof_l143_14391

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  2 * (a + b + c) + 9 / ((a * b + b * c + c * a) ^ 2) ≥ 7 := by
  sorry

end inequality_proof_l143_14391


namespace smallest_solution_quadratic_equation_l143_14313

theorem smallest_solution_quadratic_equation :
  let f : ℝ → ℝ := λ y => 6 * y^2 - 29 * y + 24
  ∃ y : ℝ, f y = 0 ∧ ∀ z : ℝ, f z = 0 → y ≤ z ∧ y = 4/3 :=
by sorry

end smallest_solution_quadratic_equation_l143_14313


namespace fraction_problem_l143_14339

theorem fraction_problem (n : ℕ) : 
  (n : ℚ) / (3 * n - 7 : ℚ) = 2 / 5 → n = 14 := by sorry

end fraction_problem_l143_14339


namespace person_age_in_1900_l143_14341

theorem person_age_in_1900 (birth_year : ℕ) (death_year : ℕ) (age_at_death : ℕ) :
  (age_at_death = birth_year / 29) →
  (birth_year < 1900) →
  (1901 ≤ death_year) →
  (death_year ≤ 1930) →
  (death_year = birth_year + age_at_death) →
  (1900 - birth_year = 44) :=
by sorry

end person_age_in_1900_l143_14341


namespace edward_garage_sale_games_l143_14367

/-- The number of games Edward bought at the garage sale -/
def garage_sale_games : ℕ := 14

/-- The number of games Edward bought from a friend -/
def friend_games : ℕ := 41

/-- The number of games that didn't work -/
def bad_games : ℕ := 31

/-- The number of good games Edward ended up with -/
def good_games : ℕ := 24

theorem edward_garage_sale_games :
  garage_sale_games = (good_games + bad_games) - friend_games :=
by sorry

end edward_garage_sale_games_l143_14367


namespace managers_salary_proof_l143_14312

def prove_managers_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) : Prop :=
  let total_salary := num_employees * avg_salary
  let new_avg := avg_salary + avg_increase
  let new_total := (num_employees + 1) * new_avg
  new_total - total_salary = 3800

theorem managers_salary_proof :
  prove_managers_salary 20 1700 100 := by
  sorry

end managers_salary_proof_l143_14312


namespace multiplication_simplification_l143_14350

theorem multiplication_simplification :
  2000 * 2992 * 0.2992 * 20 = 4 * 2992^2 := by
  sorry

end multiplication_simplification_l143_14350


namespace mans_speed_against_current_l143_14363

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem stating that given the specific conditions, 
    the man's speed against the current is 16 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 21 2.5 = 16 := by
  sorry

#eval speed_against_current 21 2.5

end mans_speed_against_current_l143_14363


namespace least_number_to_add_for_divisibility_l143_14357

def is_two_digit_prime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

theorem least_number_to_add_for_divisibility :
  ∃ (p : ℕ) (h : is_two_digit_prime p), ∀ (k : ℕ), k < 1 → ¬(∃ (q : ℕ), is_two_digit_prime q ∧ (54321 + k) % q = 0) :=
sorry

end least_number_to_add_for_divisibility_l143_14357


namespace sqrt_difference_equality_l143_14334

theorem sqrt_difference_equality : 3 * Real.sqrt 5 - Real.sqrt 20 = Real.sqrt 5 := by
  sorry

end sqrt_difference_equality_l143_14334


namespace patel_family_concert_cost_l143_14375

theorem patel_family_concert_cost : 
  let regular_ticket_price : ℚ := 7.50 / (1 - 0.20)
  let children_ticket_price : ℚ := regular_ticket_price * (1 - 0.60)
  let senior_ticket_price : ℚ := 7.50
  let num_tickets_per_generation : ℕ := 2
  let handling_fee : ℚ := 5

  (num_tickets_per_generation * senior_ticket_price + 
   num_tickets_per_generation * regular_ticket_price + 
   num_tickets_per_generation * children_ticket_price + 
   handling_fee) = 46.25 := by
sorry


end patel_family_concert_cost_l143_14375


namespace log_equation_implies_sum_l143_14358

theorem log_equation_implies_sum (u v : ℝ) 
  (hu : u > 1) (hv : v > 1)
  (h : (Real.log u / Real.log 3)^3 + (Real.log v / Real.log 5)^3 + 6 = 
       6 * (Real.log u / Real.log 3) * (Real.log v / Real.log 5)) :
  u^Real.sqrt 3 + v^Real.sqrt 3 = 152 := by
  sorry

end log_equation_implies_sum_l143_14358


namespace buffalo_count_l143_14307

/-- A group of animals consisting of buffaloes and ducks -/
structure AnimalGroup where
  buffaloes : ℕ
  ducks : ℕ

/-- The total number of legs in the group -/
def total_legs (group : AnimalGroup) : ℕ := 4 * group.buffaloes + 2 * group.ducks

/-- The total number of heads in the group -/
def total_heads (group : AnimalGroup) : ℕ := group.buffaloes + group.ducks

/-- The main theorem: there are 12 buffaloes in the group -/
theorem buffalo_count (group : AnimalGroup) : 
  (total_legs group = 2 * total_heads group + 24) → group.buffaloes = 12 := by
  sorry

end buffalo_count_l143_14307


namespace intersection_equality_condition_l143_14323

def A : Set ℝ := {x | 1 < x ∧ x ≤ 2}

def B (a : ℝ) : Set ℝ := {x | (2 : ℝ)^(2*a*x) < (2 : ℝ)^(a+x)}

theorem intersection_equality_condition (a : ℝ) :
  A ∩ B a = A ↔ a < 2/3 := by sorry

end intersection_equality_condition_l143_14323


namespace divide_by_reciprocal_twelve_divided_by_one_twelfth_l143_14317

theorem divide_by_reciprocal (x y : ℚ) (h : y ≠ 0) : x / y = x * (1 / y) := by sorry

theorem twelve_divided_by_one_twelfth : 12 / (1 / 12) = 144 := by sorry

end divide_by_reciprocal_twelve_divided_by_one_twelfth_l143_14317


namespace smallest_number_above_50_with_conditions_fifty_one_satisfies_conditions_fifty_one_is_answer_l143_14308

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def count_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem smallest_number_above_50_with_conditions : 
  ∀ n : ℕ, n > 50 → n < 51 → 
  (¬ is_perfect_square n ∨ count_factors n % 2 = 1 ∨ n % 3 ≠ 0) :=
by sorry

theorem fifty_one_satisfies_conditions : 
  ¬ is_perfect_square 51 ∧ count_factors 51 % 2 = 0 ∧ 51 % 3 = 0 :=
by sorry

theorem fifty_one_is_answer : 
  ∀ n : ℕ, n > 50 → n < 51 → 
  (¬ is_perfect_square n ∨ count_factors n % 2 = 1 ∨ n % 3 ≠ 0) ∧
  (¬ is_perfect_square 51 ∧ count_factors 51 % 2 = 0 ∧ 51 % 3 = 0) :=
by sorry

end smallest_number_above_50_with_conditions_fifty_one_satisfies_conditions_fifty_one_is_answer_l143_14308


namespace unique_quadruple_solution_l143_14330

theorem unique_quadruple_solution :
  ∃! (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧
    a^2 + b^2 + c^2 + d^2 = 9 ∧
    (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 81 ∧
    a + b + c + d = 6 ∧
    a = 1.5 ∧ b = 1.5 ∧ c = 1.5 ∧ d = 1.5 :=
by sorry

end unique_quadruple_solution_l143_14330


namespace odd_prob_greater_than_even_prob_l143_14340

/-- Represents the number of beads in the bottle -/
def num_beads : ℕ := 4

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the total number of possible outcomes when pouring beads -/
def total_outcomes : ℕ := choose num_beads 1 + choose num_beads 2 + choose num_beads 3 + choose num_beads 4

/-- Calculates the number of outcomes resulting in an odd number of beads -/
def odd_outcomes : ℕ := choose num_beads 1 + choose num_beads 3

/-- Calculates the number of outcomes resulting in an even number of beads -/
def even_outcomes : ℕ := choose num_beads 2 + choose num_beads 4

/-- Theorem stating that the probability of pouring out an odd number of beads
    is greater than the probability of pouring out an even number of beads -/
theorem odd_prob_greater_than_even_prob :
  (odd_outcomes : ℚ) / total_outcomes > (even_outcomes : ℚ) / total_outcomes :=
sorry

end odd_prob_greater_than_even_prob_l143_14340


namespace triangle_similarity_l143_14328

-- Define the points in the plane
variable (A B C A' B' C' S M N : ℝ × ℝ)

-- Define the properties of the triangles and points
def is_equilateral (X Y Z : ℝ × ℝ) : Prop := sorry
def is_center (S X Y Z : ℝ × ℝ) : Prop := sorry
def is_midpoint (M X Y : ℝ × ℝ) : Prop := sorry
def are_similar (T1 T2 T3 U1 U2 U3 : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem triangle_similarity 
  (h1 : is_equilateral A B C)
  (h2 : is_equilateral A' B' C')
  (h3 : is_center S A B C)
  (h4 : A' ≠ S)
  (h5 : B' ≠ S)
  (h6 : is_midpoint M A' B)
  (h7 : is_midpoint N A B') :
  are_similar S B' M S A' N := by sorry

end triangle_similarity_l143_14328


namespace purely_imaginary_complex_number_l143_14382

theorem purely_imaginary_complex_number (m : ℝ) : 
  (Complex.I * (m + 1) : ℂ).re = 0 ∧ (Complex.I * (m + 1) : ℂ).im ≠ 0 → m = 1 := by
  sorry

end purely_imaginary_complex_number_l143_14382


namespace sarah_test_result_l143_14321

/-- Represents a math test with a number of problems and a score percentage -/
structure MathTest where
  problems : ℕ
  score : ℚ
  score_valid : 0 ≤ score ∧ score ≤ 1

/-- Calculates the number of correctly answered problems in a test -/
def correctProblems (test : MathTest) : ℚ :=
  test.problems * test.score

/-- Calculates the overall percentage of correctly answered problems across multiple tests -/
def overallPercentage (tests : List MathTest) : ℚ :=
  let totalCorrect := (tests.map correctProblems).sum
  let totalProblems := (tests.map (·.problems)).sum
  totalCorrect / totalProblems

theorem sarah_test_result : 
  let test1 : MathTest := { problems := 30, score := 85/100, score_valid := by norm_num }
  let test2 : MathTest := { problems := 50, score := 75/100, score_valid := by norm_num }
  let test3 : MathTest := { problems := 20, score := 80/100, score_valid := by norm_num }
  let tests := [test1, test2, test3]
  overallPercentage tests = 78/100 := by
  sorry

end sarah_test_result_l143_14321


namespace danny_travel_time_l143_14305

/-- The time it takes Danny to reach Steve's house -/
def danny_time : ℝ := 31

/-- The time it takes Steve to reach Danny's house -/
def steve_time (t : ℝ) : ℝ := 2 * t

/-- The time difference between Steve and Danny reaching the halfway point -/
def halfway_time_difference : ℝ := 15.5

theorem danny_travel_time :
  ∀ t : ℝ,
  (steve_time t / 2 - t / 2 = halfway_time_difference) →
  t = danny_time :=
by
  sorry


end danny_travel_time_l143_14305


namespace mascot_problem_solution_l143_14378

/-- Represents the sales data for a week -/
structure WeekSales where
  bing : ℕ
  shuey : ℕ
  revenue : ℕ

/-- Solves for mascot prices and maximum purchase given sales data and budget -/
def solve_mascot_problem (week1 week2 : WeekSales) (total_budget total_mascots : ℕ) :
  (ℕ × ℕ × ℕ) :=
sorry

/-- Theorem stating the correctness of the solution -/
theorem mascot_problem_solution :
  let week1 : WeekSales := ⟨3, 5, 1800⟩
  let week2 : WeekSales := ⟨4, 10, 3100⟩
  let (bing_price, shuey_price, max_bing) := solve_mascot_problem week1 week2 6700 30
  bing_price = 250 ∧ shuey_price = 210 ∧ max_bing = 10 :=
sorry

end mascot_problem_solution_l143_14378
