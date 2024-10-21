import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hvac_cost_calculation_l1329_132971

noncomputable def hvac_cost_per_vent (base_cost : ℝ) (zone1_vents : ℕ) (zone1_cost : ℝ) 
                       (zone2_vents : ℕ) (zone2_cost : ℝ) 
                       (installation_fee_rate : ℝ) (discount_rate : ℝ) : ℝ :=
  let total_vent_cost := zone1_vents * zone1_cost + zone2_vents * zone2_cost
  let installation_fee := installation_fee_rate * total_vent_cost
  let discount := discount_rate * base_cost
  let discounted_base_cost := base_cost - discount
  let overall_cost := discounted_base_cost + total_vent_cost + installation_fee
  let total_vents := zone1_vents + zone2_vents
  overall_cost / (total_vents : ℝ)

theorem hvac_cost_calculation :
  hvac_cost_per_vent 20000 5 300 7 400 0.1 0.05 = 1977.50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hvac_cost_calculation_l1329_132971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_two_l1329_132945

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 1)

-- Define the tangent line at x = 2
def tangent_line (y : ℝ) : Prop := y = f 2

-- Define the vertical line x = 1
def vertical_line (x : ℝ) : Prop := x = 1

-- Define the line y = x
def diagonal_line (x y : ℝ) : Prop := y = x

-- Theorem statement
theorem triangle_area_is_two :
  ∃ (A B C : ℝ × ℝ),
    vertical_line A.1 ∧
    diagonal_line A.1 A.2 ∧
    vertical_line B.1 ∧
    tangent_line B.2 ∧
    diagonal_line C.1 C.2 ∧
    tangent_line C.2 ∧
    (1/2 * |B.1 - A.1| * |C.2 - A.2| = 2) := by
  sorry

#check triangle_area_is_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_two_l1329_132945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_from_parametric_l1329_132955

def plane_equation (x y z : ℝ) : ℝ := 4 * x - 3 * y - 2 * z + 3

def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ := (2 + 2*s - t, 1 - 2*s, 4 + s + t)

theorem plane_equation_from_parametric :
  ∀ (s t : ℝ), 
    let (x, y, z) := parametric_plane s t
    plane_equation x y z = 0 ∧
    (4 : ℤ) > 0 ∧
    Int.gcd (Int.natAbs 4) (Int.natAbs 3) = 
      Int.gcd (Int.gcd (Int.natAbs 4) (Int.natAbs 3)) (Int.gcd (Int.natAbs 2) (Int.natAbs 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_from_parametric_l1329_132955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_series_properties_l1329_132966

/-- A basketball game series with 6 games and 1/3 probability of winning each game -/
structure BasketballSeries where
  num_games : ℕ := 6
  win_prob : ℚ := 1/3
  independent : Bool := true

/-- The probability of losing the first two games and then winning the third -/
def prob_lose_two_win_third (series : BasketballSeries) : ℚ :=
  (1 - series.win_prob)^2 * series.win_prob

/-- The probability of winning exactly 3 out of 6 games -/
def prob_win_three_of_six (series : BasketballSeries) : ℚ :=
  (Nat.choose series.num_games 3) * (series.win_prob^3) * ((1 - series.win_prob)^3)

/-- The mean number of games won in the series -/
def mean_games_won (series : BasketballSeries) : ℚ :=
  series.num_games * series.win_prob

/-- The variance of the number of games won in the series -/
def variance_games_won (series : BasketballSeries) : ℚ :=
  series.num_games * series.win_prob * (1 - series.win_prob)

theorem basketball_series_properties (series : BasketballSeries) :
  prob_lose_two_win_third series = 4/27 ∧
  prob_win_three_of_six series = 160/729 ∧
  mean_games_won series = 2 ∧
  variance_games_won series = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_series_properties_l1329_132966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_for_diophantine_equation_l1329_132954

theorem solution_for_diophantine_equation (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3) :
  ∃ (a b : ℤ), a = -↑p ∧ b = 0 ∧ a^2 + 3*a*b + 2*↑p*(a + b) + ↑p^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_for_diophantine_equation_l1329_132954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l1329_132924

noncomputable def f (x : ℝ) := (1/3) * x^3 - 4*x + 4

theorem f_extrema :
  (∃ x : ℝ, f x = 28/3 ∧ ∀ y : ℝ, f y ≤ f x) ∧
  (∃ x : ℝ, f x = -4/3 ∧ ∀ y : ℝ, f y ≥ f x) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 3 → f x ≤ 4) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 3 → f x ≥ -4/3) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 3 ∧ f x = 4) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 3 ∧ f x = -4/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l1329_132924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_iff_a_in_range_l1329_132951

-- Define the function f(x) as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - (1/3) * Real.sin (2*x) + a * Real.sin x

-- State the theorem
theorem monotonic_increasing_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Icc (-1/3) (1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_iff_a_in_range_l1329_132951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_great_wall_length_proof_l1329_132969

/-- The length of the Great Wall given the number of soldiers and tower spacing -/
def great_wall_length 
  (tower_spacing : ℕ) 
  (soldiers_per_tower : ℕ) 
  (total_soldiers : ℕ) : ℕ :=
  let num_towers := total_soldiers / soldiers_per_tower
  (num_towers - 1) * tower_spacing

/-- Proof of the Great Wall's length -/
theorem great_wall_length_proof 
  (h1 : tower_spacing = 5)
  (h2 : soldiers_per_tower = 2)
  (h3 : total_soldiers = 2920) :
  great_wall_length tower_spacing soldiers_per_tower total_soldiers = 7295 := by
  sorry

#eval great_wall_length 5 2 2920

end NUMINAMATH_CALUDE_ERRORFEEDBACK_great_wall_length_proof_l1329_132969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_integer_product_l1329_132959

theorem four_integer_product (m n p q : ℕ) : 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q →
  m > 0 ∧ n > 0 ∧ p > 0 ∧ q > 0 →
  (7 - m : ℤ) * (7 - n : ℤ) * (7 - p : ℤ) * (7 - q : ℤ) = 4 →
  m + n + p + q = 28 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_integer_product_l1329_132959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l1329_132933

/-- The cubic polynomial q(x) that satisfies specific conditions -/
noncomputable def q (x : ℝ) : ℝ := (8/3) * x^3 - 12 * x^2 + 76 * x - 200/3

/-- Theorem stating that q(x) satisfies the required conditions -/
theorem q_satisfies_conditions :
  q 1 = 0 ∧ q 2 = 8 ∧ q 3 = 24 ∧ q 4 = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l1329_132933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_better_fit_as_r_squared_approaches_one_l1329_132939

/-- The coefficient of determination in regression analysis -/
def coefficient_of_determination : ℝ → ℝ := sorry

/-- A measure of the fit of a regression model -/
def model_fit : ℝ → ℝ := sorry

/-- As the coefficient of determination approaches 1, the model fit improves -/
theorem better_fit_as_r_squared_approaches_one :
  ∀ ε > 0, ∃ δ > 0, ∀ r : ℝ, 
  1 - δ < r → r < 1 → 
  model_fit r > model_fit (1 - ε) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_better_fit_as_r_squared_approaches_one_l1329_132939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_215_bound_sum_l1329_132983

theorem log_215_bound_sum : ∃ (c d : ℤ), c + 1 = d ∧ 
  (c : ℝ) < Real.log 215 / Real.log 10 ∧ Real.log 215 / Real.log 10 < (d : ℝ) ∧ c + d = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_215_bound_sum_l1329_132983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_solution_l1329_132985

/-- Represents the investment scenario described in the problem -/
structure InvestmentScenario where
  totalInvestment : ℚ
  fundXRate : ℚ
  fundYRate : ℚ
  interestDifference : ℚ

/-- The solution to the investment problem -/
def solveFundXInvestment (scenario : InvestmentScenario) : ℚ :=
  scenario.totalInvestment * (scenario.fundYRate - scenario.interestDifference / scenario.totalInvestment) / 
    (scenario.fundXRate + scenario.fundYRate)

/-- Theorem stating the solution to the specific investment problem -/
theorem investment_problem_solution :
  let scenario : InvestmentScenario := {
    totalInvestment := 100000,
    fundXRate := 23/100,
    fundYRate := 17/100,
    interestDifference := 200
  }
  solveFundXInvestment scenario = 42000 := by
  sorry

#eval solveFundXInvestment {
  totalInvestment := 100000,
  fundXRate := 23/100,
  fundYRate := 17/100,
  interestDifference := 200
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_solution_l1329_132985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_8_ball_probability_l1329_132922

def num_questions : ℕ := 6
def num_positive : ℕ := 3
def prob_positive : ℚ := 1/3

theorem magic_8_ball_probability :
  (Nat.choose num_questions num_positive : ℚ) * prob_positive^num_positive * (1 - prob_positive)^(num_questions - num_positive) = 160/729 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_8_ball_probability_l1329_132922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_cover_modified_chessboard_l1329_132989

/-- Represents a chessboard with two opposite corners removed -/
structure ModifiedChessboard where
  size : Nat
  removed_corners : (Nat × Nat) × (Nat × Nat)

/-- Represents a domino piece -/
structure Domino where
  length : Nat
  width : Nat

/-- Function to check if a board can be covered by dominoes -/
def can_cover_board (board : ModifiedChessboard) (domino : Domino) (num_dominoes : Nat) : Prop :=
  ∃ (arrangement : List (Nat × Nat)),
    arrangement.length = num_dominoes ∧
    (∀ pos ∈ arrangement, pos.1 ≤ board.size ∧ pos.2 ≤ board.size) ∧
    (∀ pos ∈ arrangement, pos ≠ board.removed_corners.1 ∧ pos ≠ board.removed_corners.2) ∧
    (∀ i j, i < arrangement.length → j < arrangement.length → i ≠ j → arrangement[i]? ≠ arrangement[j]?)

/-- Theorem stating that it's impossible to cover the modified 8x8 board with 31 dominoes -/
theorem impossible_cover_modified_chessboard :
  ¬ can_cover_board
    { size := 8, removed_corners := ((1, 1), (8, 8)) }
    { length := 2, width := 1 }
    31 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_cover_modified_chessboard_l1329_132989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_angle_relation_l1329_132928

theorem triangle_side_angle_relation (α β γ a b c : Real) : 
  0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = Real.pi ∧  -- acute triangle conditions
  Real.sin α = 3/5 ∧ Real.cos β = 5/13 ∧  -- given angle conditions
  Real.sin α = a / (2 * Real.sqrt (a * b * Real.sin γ / 2)) ∧  -- law of sines
  Real.sin β = b / (2 * Real.sqrt (a * b * Real.sin γ / 2)) ∧
  Real.sin γ = c / (2 * Real.sqrt (a * b * Real.sin γ / 2)) →
  (a^2 + b^2 - c^2) / (a * b) = 32/65 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_angle_relation_l1329_132928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slope_is_one_l1329_132990

/-- Represents a circle in the 2D plane -/
structure Circle where
  a : ℝ  -- coefficient of x^2
  b : ℝ  -- coefficient of y^2
  c : ℝ  -- coefficient of x
  d : ℝ  -- coefficient of y
  e : ℝ  -- constant term

/-- The equation of a circle: ax^2 + by^2 + cx + dy + e = 0 -/
def Circle.equation (circle : Circle) (x y : ℝ) : Prop :=
  circle.a * x^2 + circle.b * y^2 + circle.c * x + circle.d * y + circle.e = 0

/-- Two circles intersect if there exist points satisfying both circle equations -/
def intersect (c1 c2 : Circle) : Prop :=
  ∃ x y, c1.equation x y ∧ c2.equation x y

/-- The slope of a line given two points (x1, y1) and (x2, y2) -/
noncomputable def lineslope (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y2 - y1) / (x2 - x1)

/-- Main theorem: The slope of the line connecting intersection points of the given circles is 1 -/
theorem intersection_slope_is_one :
  let c1 : Circle := { a := 1, b := 1, c := -6, d := 4, e := -20 }
  let c2 : Circle := { a := 1, b := 1, c := -10, d := 8, e := 40 }
  intersect c1 c2 →
  ∃ x1 y1 x2 y2 : ℝ,
    c1.equation x1 y1 ∧ c2.equation x1 y1 ∧
    c1.equation x2 y2 ∧ c2.equation x2 y2 ∧
    x1 ≠ x2 ∧
    lineslope x1 y1 x2 y2 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slope_is_one_l1329_132990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_ring_area_l1329_132900

/-- The area of a circular ring with outer radius r and inner radius r/2 -/
noncomputable def circularRingArea (r : ℝ) : ℝ := Real.pi * r^2 - Real.pi * (r/2)^2

theorem circular_ring_area (r : ℝ) (h : r > 0) : 
  circularRingArea r = 3/4 * Real.pi * r^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_ring_area_l1329_132900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_pairs_l1329_132921

theorem harmonic_mean_pairs : 
  let harmonic_mean (x y : ℕ) := (2 * x * y : ℝ) / (x + y)
  let valid_pair (p : ℕ × ℕ) := p.1 < p.2 ∧ harmonic_mean p.1 p.2 = (4 : ℝ)^15
  (Finset.filter valid_pair (Finset.range (2^30 + 1) ×ˢ Finset.range (2^30 + 1))).card = 29 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_pairs_l1329_132921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_count_l1329_132999

theorem book_arrangement_count : ℕ := by
  let total_books : ℕ := 7
  let identical_mystery : ℕ := 3
  let identical_scifi : ℕ := 2
  let different_books : ℕ := total_books - identical_mystery - identical_scifi

  have h : (Nat.factorial total_books) / (Nat.factorial identical_mystery * Nat.factorial identical_scifi) = 420 := by
    sorry

  exact 420

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_count_l1329_132999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_real_imag_parts_of_complex_product_l1329_132913

theorem equal_real_imag_parts_of_complex_product (a : ℝ) : 
  (((2 : ℂ) + a * Complex.I) * (1 + Complex.I)).re = 
  (((2 : ℂ) + a * Complex.I) * (1 + Complex.I)).im → 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_real_imag_parts_of_complex_product_l1329_132913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_comparison_l1329_132914

/-- Represents the percentage of salary saved last year -/
noncomputable def last_year_savings_rate : ℝ := 0.075

/-- Represents the percentage increase in salary this year -/
noncomputable def salary_increase_rate : ℝ := 0.125

/-- Represents the percentage of salary saved this year -/
noncomputable def this_year_savings_rate : ℝ := 0.092

/-- Represents the ratio of this year's savings to last year's savings -/
noncomputable def savings_ratio : ℝ := 
  (this_year_savings_rate * (1 + salary_increase_rate)) / last_year_savings_rate

theorem savings_comparison : savings_ratio = 1.38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_comparison_l1329_132914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_n_properties_l1329_132916

noncomputable def T_n (n : ℕ) (x : ℝ) : ℝ := Real.cos (n * Real.arccos x)

theorem T_n_properties (n : ℕ) :
  ∃ (p : Polynomial ℝ),
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, T_n n x = p.eval x) ∧
    p.degree = n ∧
    p.leadingCoeff = 2^(n-1) ∧
    (∀ k ∈ Finset.range n, p.eval (Real.cos ((2 * k.succ - 1 : ℝ) * π / (2 * n))) = 0) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, T_n n x ≤ 1) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, T_n n x ≥ -1) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 1, T_n n x = 1) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 1, T_n n x = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_n_properties_l1329_132916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_two_subsets_l1329_132905

-- Define the set A
def A (k : ℝ) : Set ℝ := {x : ℝ | (k + 1) * x^2 + x - k = 0}

-- State the theorem
theorem set_A_two_subsets (k : ℝ) : 
  (∃ s : Finset ℝ, s.card = 1 ∧ A k = s) → (k = -1 ∨ k = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_two_subsets_l1329_132905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1329_132949

-- Define the quadratic equation
def quadratic_equation (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 + x + 1 = 0

-- Define the function f(x) = cos^2(ax) - sin^2(ax)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (Real.cos (a * x))^2 - (Real.sin (a * x))^2

-- Define the function g(x) = 2^x - x^2
noncomputable def g (x : ℝ) : ℝ :=
  2^x - x^2

-- Define the power function
noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ :=
  x^α

theorem problem_solution :
  (¬ (∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ quadratic_equation a x ∧ quadratic_equation a y) → a ≤ 1/4)) ∧
  (∀ a : ℝ, (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f a (x + T) = f a x ∧ ∀ S : ℝ, S > 0 ∧ S < T → ∃ y : ℝ, f a (y + S) ≠ f a y) → T = π) →
    (a = 1 ∨ a = -1) ∧
  (∃ a : ℝ, a ≠ 1 ∧ a ≠ -1 ∧ 
    (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f a (x + T) = f a x ∧ ∀ S : ℝ, S > 0 ∧ S < T → ∃ y : ℝ, f a (y + S) ≠ f a y) ∧ T = π) ∧
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ g x₁ = 0 ∧ g x₂ = 0 ∧ g x₃ = 0) ∧
  (∃ α : ℝ, ∀ ε : ℝ, ε > 0 → ∃ x : ℝ, 0 < |x| ∧ |x| < ε ∧ power_function α x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1329_132949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_parabola_intersection_distance_asymptote_parabola_intersection_distance_proof_l1329_132926

/-- The distance between the two non-origin intersection points of y = ±x and y^2 = 4x is 8 -/
theorem asymptote_parabola_intersection_distance : ℝ :=
  let hyperbola := fun x y : ℝ => x^2 - y^2 = 1
  let parabola := fun x y : ℝ => y^2 = 4*x
  let asymptote_pos := fun x y : ℝ => y = x
  let asymptote_neg := fun x y : ℝ => y = -x
  let origin : ℝ × ℝ := (0, 0)
  let intersections := {p : ℝ × ℝ | (asymptote_pos p.1 p.2 ∨ asymptote_neg p.1 p.2) ∧ parabola p.1 p.2 ∧ p ≠ origin}
  8

theorem asymptote_parabola_intersection_distance_proof : asymptote_parabola_intersection_distance = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_parabola_intersection_distance_asymptote_parabola_intersection_distance_proof_l1329_132926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_to_asymptote_distance_l1329_132912

/-- The distance from the focus of a hyperbola to its asymptote -/
noncomputable def distance_focus_to_asymptote (a b : ℝ) : ℝ :=
  |b| / a

/-- Theorem stating that the distance from the focus of the hyperbola to its asymptote is 2 -/
theorem hyperbola_focus_to_asymptote_distance :
  ∃ b : ℝ,
  (∀ x y : ℝ, x^2 / 5 - y^2 / b^2 = 1 → -- Hyperbola equation
    ∃ c : ℝ, y^2 = 12 * x) ∧           -- Parabola equation
  (∃ x₀ : ℝ, x₀ = 3) ∧                 -- Right focus of hyperbola coincides with parabola focus
  distance_focus_to_asymptote 5 b = 2  -- Distance from focus to asymptote is 2
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_to_asymptote_distance_l1329_132912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_l₂_tangent_to_C_and_l₁_equation_l1329_132977

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 8 + y^2 / 2 = 1

-- Define line l₂
def l₂ (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Define line l₁
def l₁ (k x y : ℝ) : Prop := y = k*x + 2

-- Define the intersection points A and B
def intersect_l₁_C (k : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂ : ℝ, 
  C x₁ y₁ ∧ C x₂ y₂ ∧ l₁ k x₁ y₁ ∧ l₁ k x₂ y₂ ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define point M
def point_M (k : ℝ) : Prop := ∃ x y : ℝ, l₁ k x y ∧ l₂ x y

-- Define midpoint N
def midpoint_N (k : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂ x_N y_N : ℝ,
  C x₁ y₁ ∧ C x₂ y₂ ∧ l₁ k x₁ y₁ ∧ l₁ k x₂ y₂ ∧
  x_N = (x₁ + x₂) / 2 ∧ y_N = (y₁ + y₂) / 2

-- Define the condition |AB| = |MN|
def AB_equals_MN (k : ℝ) : Prop := ∃ x_A y_A x_B y_B x_M y_M x_N y_N : ℝ,
  C x_A y_A ∧ C x_B y_B ∧ l₁ k x_A y_A ∧ l₁ k x_B y_B ∧
  l₂ x_M y_M ∧ l₁ k x_M y_M ∧
  x_N = (x_A + x_B) / 2 ∧ y_N = (y_A + y_B) / 2 ∧
  (x_A - x_B)^2 + (y_A - y_B)^2 = (x_M - x_N)^2 + (y_M - y_N)^2

theorem l₂_tangent_to_C_and_l₁_equation :
  ∀ k : ℝ, intersect_l₁_C k → point_M k → midpoint_N k → AB_equals_MN k →
  (∀ x y : ℝ, l₂ x y → C x y → ∀ ε > 0, ∃ x' y' : ℝ, 
    C x' y' ∧ (x' - x)^2 + (y' - y)^2 < ε^2) ∧
  (k = Real.sqrt 2 / 2 ∨ k = -Real.sqrt 2 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_l₂_tangent_to_C_and_l₁_equation_l1329_132977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l1329_132957

def a : Fin 3 → ℝ := ![(-4 : ℝ), 2, 4]
def b : Fin 3 → ℝ := ![(-6 : ℝ), 3, -2]

theorem vector_properties :
  (Real.sqrt ((a 0)^2 + (a 1)^2 + (a 2)^2) = 6) ∧
  ((a 0 * b 0 + a 1 * b 1 + a 2 * b 2) / 
   (Real.sqrt ((a 0)^2 + (a 1)^2 + (a 2)^2) * 
    Real.sqrt ((b 0)^2 + (b 1)^2 + (b 2)^2)) = 11/21) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l1329_132957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l1329_132964

noncomputable def k (x : ℝ) : ℝ := (2 * x + 7) / (x - 3)

theorem range_of_k :
  Set.range k = {y : ℝ | y ≠ 2} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l1329_132964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_pages_count_l1329_132940

def book_pages : Fin 20 → ℕ
  | ⟨0, h⟩ => 120
  | ⟨1, h⟩ => 150
  | ⟨2, h⟩ => 80
  | ⟨3, h⟩ => 200
  | ⟨4, h⟩ => 90
  | ⟨5, h⟩ => 180
  | ⟨6, h⟩ => 75
  | ⟨7, h⟩ => 190
  | ⟨8, h⟩ => 110
  | ⟨9, h⟩ => 160
  | ⟨10, h⟩ => 130
  | ⟨11, h⟩ => 170
  | ⟨12, h⟩ => 100
  | ⟨13, h⟩ => 140
  | ⟨14, h⟩ => 210
  | ⟨15, h⟩ => 185
  | ⟨16, h⟩ => 220
  | ⟨17, h⟩ => 135
  | ⟨18, h⟩ => 145
  | ⟨19, h⟩ => 205
  | ⟨n, h⟩ => 0 -- Default case for n ≥ 20

def missing_books : Finset (Fin 20) :=
  {⟨1, by norm_num⟩, ⟨6, by norm_num⟩, ⟨10, by norm_num⟩, ⟨13, by norm_num⟩, ⟨15, by norm_num⟩, ⟨19, by norm_num⟩}

theorem remaining_pages_count :
  (Finset.univ.sum book_pages) - (missing_books.sum book_pages) = 2110 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_pages_count_l1329_132940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_road_network_l1329_132935

/-- A square in 2D space -/
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Angle between three points in 2D space -/
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- The optimal point for minimum road network -/
noncomputable def optimalPoint (s : Square) : ℝ × ℝ := sorry

theorem optimal_road_network (s : Square) :
  let P := optimalPoint s
  (∀ Q : ℝ × ℝ,
    distance P s.A + distance P s.B + distance P s.C + distance P s.D ≤
    distance Q s.A + distance Q s.B + distance Q s.C + distance Q s.D) ∧
  (∀ X Y Z, (X = s.A ∨ X = s.B ∨ X = s.C ∨ X = s.D) →
            (Y = s.A ∨ Y = s.B ∨ Y = s.C ∨ Y = s.D) →
            (Z = s.A ∨ Z = s.B ∨ Z = s.C ∨ Z = s.D) →
            X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z →
    angle X P Y = 2 * Real.pi / 3 ∧
    angle Y P Z = 2 * Real.pi / 3 ∧
    angle Z P X = 2 * Real.pi / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_road_network_l1329_132935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_properties_l1329_132910

noncomputable def secant_length (r d : ℝ) : ℝ :=
  6 * Real.sqrt 6

noncomputable def secant_angle (r d : ℝ) : ℝ :=
  Real.arccos (6 * Real.sqrt 6 / 11)

theorem secant_properties (r d : ℝ) (hr : r = 7) (hd : d = 11) :
  secant_length r d = 6 * Real.sqrt 6 ∧
  secant_angle r d = Real.arccos (6 * Real.sqrt 6 / 11) := by
  sorry

#check secant_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_properties_l1329_132910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_l1329_132908

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the transformation from C₁ to C₂
noncomputable def transform (x y : ℝ) : ℝ × ℝ := (Real.sqrt 5 * x, y)

-- Define the curve C₂
def C₂ (x y : ℝ) : Prop := ∃ (x' y' : ℝ), C₁ x' y' ∧ transform x' y' = (x, y)

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-4 + Real.sqrt 2 * t, Real.sqrt 2 * t)

-- Define point F
def point_F : ℝ × ℝ := (-4, 0)

-- Statement to prove
theorem distance_sum : 
  ∃ (t₁ t₂ : ℝ), 
    let A := line_l t₁
    let B := line_l t₂
    C₂ A.1 A.2 ∧ C₂ B.1 B.2 ∧ t₁ ≠ t₂ ∧
    Real.sqrt ((A.1 - point_F.1)^2 + (A.2 - point_F.2)^2) + 
    Real.sqrt ((B.1 - point_F.1)^2 + (B.2 - point_F.2)^2) = 
    (4 * Real.sqrt 5) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_l1329_132908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_y_coordinate_l1329_132936

/-- The parabola function y = 4x^2 -/
def parabola (x : ℝ) : ℝ := 4 * x^2

/-- The derivative of the parabola function -/
def parabola_derivative (x : ℝ) : ℝ := 8 * x

/-- Point A on the parabola -/
def A : ℝ × ℝ := (0, parabola 0)

/-- Point B on the parabola -/
noncomputable def B : ℝ × ℝ := (0, parabola 0)

/-- The tangent line at point A is horizontal -/
axiom tangent_A_horizontal : parabola_derivative A.1 = 0

/-- The tangent line at point B is perpendicular to the tangent at A -/
axiom tangent_B_perpendicular : parabola_derivative B.1 * parabola_derivative A.1 = -1

/-- The y-coordinate of the intersection point P -/
def P_y_coordinate : ℝ := 0

/-- Theorem: The y-coordinate of the intersection point P is 0 -/
theorem intersection_point_y_coordinate : P_y_coordinate = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_y_coordinate_l1329_132936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_condition_l1329_132930

theorem complex_number_condition (a b : ℝ) (h1 : b ≠ 0) :
  let z : ℂ := Complex.ofReal a + Complex.I * b
  (z^2 - 4*b*z).im = 0 → a = 2*b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_condition_l1329_132930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_interval_length_l1329_132901

theorem max_interval_length (m n : ℕ) : 
  m < n → 
  (∃ k : ℕ, k * 2021 ∈ Set.Icc m (n - 1) ∧ 
    (∀ j : ℕ, j * 2000 ∈ Set.Icc m (n - 1) → j < k)) → 
  n - m ≤ 1999 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_interval_length_l1329_132901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1329_132904

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (|2*x - 1| + |x + 1| - a)

theorem function_properties :
  (∃ (k : ℝ), k = 3/2 ∧ ∀ (a : ℝ), ∀ (x : ℝ), f a x ∈ Set.univ → a ≤ k) ∧
  (∀ (m n : ℝ), m > 0 → n > 0 → m + n = 3 → 1/m + 4/n ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1329_132904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_pass_time_l1329_132947

/-- Calculates the time for a train to pass a platform given the train's length, 
    initial speed, speed increase percentage, and platform length. -/
noncomputable def train_pass_time (train_length : ℝ) (initial_speed : ℝ) 
                    (speed_increase_percent : ℝ) (platform_length : ℝ) : ℝ :=
  let new_speed := initial_speed * (1 + speed_increase_percent / 100)
  let total_distance := train_length + platform_length
  total_distance / new_speed

/-- Theorem stating that under given conditions, the time for the train to pass 
    the platform is approximately 160.064 seconds. -/
theorem train_platform_pass_time :
  let train_length := (1500 : ℝ)
  let initial_speed := (8.33 : ℝ)
  let speed_increase_percent := (50 : ℝ)
  let platform_length := (500 : ℝ)
  abs (train_pass_time train_length initial_speed speed_increase_percent platform_length - 160.064) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_pass_time_l1329_132947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_side_length_l1329_132965

-- Define the quadrilateral
structure Quadrilateral :=
  (a b c d : ℝ)
  (radius : ℝ)
  (inscribed : Bool)
  (area : ℝ)

-- Define our specific quadrilateral
noncomputable def our_quad : Quadrilateral :=
  { a := 100
  , b := 100
  , c := 150
  , d := 0  -- This is the unknown side we're solving for
  , radius := 100
  , inscribed := true
  , area := 7500 * Real.sqrt 3
  }

-- Theorem statement
theorem fourth_side_length (q : Quadrilateral) 
  (h1 : q.a = 100)
  (h2 : q.b = 100)
  (h3 : q.c = 150)
  (h4 : q.radius = 100)
  (h5 : q.inscribed = true)
  (h6 : q.area = 7500 * Real.sqrt 3) :
  q.d = 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_side_length_l1329_132965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_problem_l1329_132967

/-- Line in 2D represented by ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflect a point across a line -/
def reflect (p : Point) (l : Line) : Point :=
  sorry

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

theorem reflection_problem :
  let ℓ₁ : Line := ⟨2, -1, 7⟩
  let ℓ₂ : Line := ⟨5, 1, 42⟩
  let ℓ₃ : Line := ⟨1, 1, 14⟩
  
  let f₁ := (fun p => reflect p ℓ₁)
  let f₂ := (fun p => reflect p ℓ₂)
  let f₃ := (fun p => reflect p ℓ₃)
  
  ∀ X Y : Point,
    X.y = 0 →  -- X is on x-axis
    Y.x = 0 →  -- Y is on y-axis
    f₁ (f₂ (f₃ X)) = Y →
    (distance X Y)^2 = 500245 / 81 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_problem_l1329_132967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l1329_132974

theorem tan_alpha_plus_pi_fourth (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : Real.sin α = Real.sqrt 5 / 5) : 
  Real.tan (α + π/4) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l1329_132974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_max_tan_sum_l1329_132929

theorem right_triangle_max_tan_sum (A B C : ℝ) : 
  0 < A → 0 < B → 0 < C →  -- Angles are positive
  A + B + C = π →  -- Sum of angles in a triangle
  C = π / 2 →  -- Right angle at C
  Real.tan A + Real.tan B * Real.tan C ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_max_tan_sum_l1329_132929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1329_132918

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

/-- The ellipse equation -/
def ellipse_eq (x y : ℝ) : Prop := 4*x^2 + y^2 = 4

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop := y^2/3 - x^2/9 = 1

/-- The asymptote equation -/
def asymptote_eq (x y : ℝ) : Prop := y = (Real.sqrt 3/3) * x ∨ y = -(Real.sqrt 3/3) * x

/-- Foci of the ellipse -/
def ellipse_foci (x y : ℝ) : Prop := (x = 0 ∧ y = Real.sqrt 3) ∨ (x = 0 ∧ y = -Real.sqrt 3)

theorem hyperbola_equation :
  ∀ x y : ℝ,
  (∀ a b : ℝ, asymptote_eq a b → (a = 0 ∧ b = 0) ∨ (circle_eq a b)) →
  (∀ a b : ℝ, asymptote_eq a b → ∃ t : ℝ, a = t ∧ b = (Real.sqrt 3/3) * t) →
  (∀ a b : ℝ, ellipse_foci a b → hyperbola_eq a b) →
  hyperbola_eq x y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1329_132918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequalities_l1329_132991

theorem count_integers_satisfying_inequalities :
  ∃! n : ℕ, ∃ S : Finset ℤ,
    (∀ x ∈ S, (-4:ℤ) * x ≥ x + 9 ∧ (-3:ℤ) * x ≤ 15 ∧ (-5:ℤ) * x ≥ 3 * x + 21) ∧
    (∀ x : ℤ, (-4:ℤ) * x ≥ x + 9 ∧ (-3:ℤ) * x ≤ 15 ∧ (-5:ℤ) * x ≥ 3 * x + 21 → x ∈ S) ∧
    S.card = n ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequalities_l1329_132991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l1329_132944

noncomputable def P (x a : ℝ) : ℝ := (x + a / x) * (2 * x - 1 / x)^5

noncomputable def sum_of_coefficients (a : ℝ) : ℝ := P 1 a

-- Theorem statement
theorem constant_term_of_expansion (a : ℝ) 
  (h : sum_of_coefficients a = 2) : 
  ∃ (c : ℝ), c = 40 ∧ 
  (∀ (x : ℝ), x ≠ 0 → P x a = c + x * (P x a - c) / x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l1329_132944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_148_l1329_132975

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  height : ℝ
  top_base : ℝ
  bottom_base : ℝ
  perimeter_diff : ℝ
  area_ratio : ℝ

/-- The area of a trapezoid with specific properties -/
noncomputable def trapezoid_area (t : Trapezoid) : ℝ :=
  (t.top_base + t.bottom_base) * t.height / 2

/-- Theorem stating the area of the trapezoid with given conditions -/
theorem trapezoid_area_is_148 (t : Trapezoid) 
    (h1 : t.height = 2)
    (h2 : t.perimeter_diff = 24)
    (h3 : t.area_ratio = 20/17)
    (h4 : t.bottom_base - t.top_base = t.perimeter_diff)
    (h5 : (3 * t.bottom_base + t.top_base) / (3 * t.top_base + t.bottom_base) = t.area_ratio) :
  trapezoid_area t = 148 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_148_l1329_132975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_post_circumference_l1329_132997

/-- A squirrel running up a cylindrical post in a spiral path -/
structure SpiralPath where
  postHeight : ℝ
  circuitHeight : ℝ
  travelDistance : ℝ

/-- The circumference of the post given a spiral path -/
noncomputable def postCircumference (path : SpiralPath) : ℝ :=
  path.travelDistance / (path.postHeight / path.circuitHeight)

/-- Theorem stating the circumference of the post is 4 feet -/
theorem squirrel_post_circumference :
  let path : SpiralPath := {
    postHeight := 16,
    circuitHeight := 4,
    travelDistance := 8
  }
  postCircumference path = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_post_circumference_l1329_132997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_congruent_partition_of_disk_l1329_132970

-- Define a disk
def isDisk (D : Set (ℝ × ℝ)) : Prop := 
  ∃ (c : ℝ × ℝ) (r : ℝ), D = {p : ℝ × ℝ | dist p c ≤ r}

-- Define congruence for sets
def areCongruent (A B : Set (ℝ × ℝ)) : Prop :=
  ∃ (f : ℝ × ℝ → ℝ × ℝ), Isometry f ∧ f '' A = B

theorem no_congruent_partition_of_disk (D : Set (ℝ × ℝ)) :
  isDisk D →
  ¬∃ (A B : Set (ℝ × ℝ)), 
    areCongruent A B ∧ 
    A ∩ B = ∅ ∧ 
    A ∪ B = D :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_congruent_partition_of_disk_l1329_132970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l1329_132953

-- Define the circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the point P
def P (x0 y0 : ℝ) : Prop := unit_circle x0 y0

-- Define the point M
def M (x y : ℝ) : Prop := ∃ x0 y0 : ℝ, P x0 y0 ∧ x = 2*x0 ∧ y = y0

-- Define an ellipse with foci on the x-axis
def ellipse_x_axis (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Theorem statement
theorem trajectory_of_M :
  ∀ x y : ℝ, M x y → ellipse_x_axis x y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l1329_132953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l1329_132937

-- Define the constants
noncomputable def a : ℝ := 2^(3/10 : ℝ)
def b : ℝ := 3^2
noncomputable def c : ℝ := 2^(-(3/10) : ℝ)

-- State the theorem
theorem size_relationship : c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l1329_132937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1329_132907

-- Define the function (marked as noncomputable due to Real.log)
noncomputable def f (x : ℝ) : ℝ := (Real.log (abs (x - 2))) / (Real.log 0.3) / (x^2 - 4*x)

-- Define the solution set
def solution_set : Set ℝ := 
  {x | x < 0 ∨ (1 < x ∧ x < 2) ∨ (2 < x ∧ x < 3) ∨ x > 4}

-- State the theorem
theorem inequality_solution (x : ℝ) : 
  f x < 0 ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1329_132907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l1329_132919

/-- The time taken for two trains to cross each other -/
noncomputable def time_to_cross (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / ((speed1 + speed2) * 1000 / 3600)

/-- Theorem: Two trains with given lengths and speeds take 10.8 seconds to cross each other -/
theorem trains_crossing_time :
  let length1 : ℝ := 140
  let length2 : ℝ := 160
  let speed1 : ℝ := 60
  let speed2 : ℝ := 40
  time_to_cross length1 length2 speed1 speed2 = 10.8 := by
  sorry

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check time_to_cross 140 160 60 40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l1329_132919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_13_l1329_132946

-- Define the circle
def circle_center (x : ℝ) : ℝ × ℝ := (x, 0)

-- Define the two points on the circle
def point1 : ℝ × ℝ := (2, 6)
def point2 : ℝ × ℝ := (3, 2)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem circle_radius_is_13 :
  ∃ x : ℝ, x > 3 ∧
    distance (circle_center x) point1 = distance (circle_center x) point2 ∧
    distance (circle_center x) point1 = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_13_l1329_132946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_bottle_volume_l1329_132979

/-- The number of liters in each bottle of soda -/
def liters_per_bottle : ℝ := sorry

/-- The number of bottles of orange soda Julio has -/
def julio_orange : ℕ := 4

/-- The number of bottles of grape soda Julio has -/
def julio_grape : ℕ := 7

/-- The number of bottles of orange soda Mateo has -/
def mateo_orange : ℕ := 1

/-- The number of bottles of grape soda Mateo has -/
def mateo_grape : ℕ := 3

/-- The difference in liters of soda between Julio and Mateo -/
def difference : ℕ := 14

theorem soda_bottle_volume :
  (julio_orange + julio_grape) * liters_per_bottle - 
  (mateo_orange + mateo_grape) * liters_per_bottle = difference →
  liters_per_bottle = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_bottle_volume_l1329_132979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_tessellation_l1329_132942

/-- A tile in the tessellation -/
inductive Tile
  | Hexagon
  | Triangle

/-- The tessellation structure -/
structure Tessellation where
  tiles : Set Tile
  adjacent : Tile → Tile → Prop

/-- A coloring of the tessellation -/
def Coloring (τ : Tessellation) := Tile → Fin 3

/-- A valid coloring where adjacent tiles have different colors -/
def ValidColoring (τ : Tessellation) (c : Coloring τ) : Prop :=
  ∀ t1 t2, τ.adjacent t1 t2 → c t1 ≠ c t2

/-- The property that hexagons are surrounded by triangles -/
def HexagonsSurroundedByTriangles (τ : Tessellation) : Prop :=
  ∀ h, h ∈ τ.tiles → h = Tile.Hexagon →
    ∃ t, t ∈ τ.tiles ∧ t = Tile.Triangle ∧ τ.adjacent h t

/-- The main theorem: minimum 3 colors are needed for a valid coloring -/
theorem min_colors_tessellation (τ : Tessellation)
    (surround : HexagonsSurroundedByTriangles τ) :
    (∃ c : Coloring τ, ValidColoring τ c) ∧
    ∀ c : Coloring τ, ValidColoring τ c → Fintype.card (Fin 3) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_tessellation_l1329_132942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1329_132998

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- Predicate to check if a point is a focus of an ellipse -/
def IsFocus (e : Ellipse) (F : Point) : Prop :=
  sorry

/-- Predicate to check if PF₁ is perpendicular to PF₂ -/
def IsPerp (P F₁ F₂ : Point) : Prop :=
  sorry

/-- Function to calculate the area of a triangle given three points -/
def TriangleArea (P F₁ F₂ : Point) : ℝ :=
  sorry

/-- Theorem about an ellipse with specific properties -/
theorem ellipse_properties (e : Ellipse) (P F₁ F₂ : Point) :
  -- P is on the ellipse
  (P.x^2 / e.a^2) + (P.y^2 / e.b^2) = 1 →
  -- P has coordinates (3,4)
  P.x = 3 ∧ P.y = 4 →
  -- F₁ and F₂ are foci of the ellipse
  IsFocus e F₁ ∧ IsFocus e F₂ →
  -- PF₁ is perpendicular to PF₂
  IsPerp P F₁ F₂ →
  -- The equation of the ellipse is x^2/25 + y^2/20 = 1
  e.a^2 = 25 ∧ e.b^2 = 20 ∧
  -- The area of triangle PF₁F₂ is 20
  TriangleArea P F₁ F₂ = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1329_132998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_4_l1329_132994

/-- The function f(x) = 3√x + 15/√x + x -/
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sqrt x + 15 / Real.sqrt x + x

/-- The function g(x) = 2x^2 - 2x - 3 -/
def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x - 3

/-- Theorem stating that f(g(4)) = (78 + 441)/√21 -/
theorem f_of_g_4 : f (g 4) = (78 + 441) / Real.sqrt 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_4_l1329_132994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1329_132903

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x + 3)

theorem domain_of_f : Set.univ = {x : ℝ | ∃ y, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1329_132903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1329_132981

noncomputable def f (x m : ℝ) := Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 + m

theorem f_properties (m : ℝ) :
  ∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f x m = f (x + T) m ∧
  (∀ (x : ℝ), f x m ≤ 1 → m = -Real.sqrt 2 ∧ ∃ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) ∧ f y m = -Real.sqrt 2) ∧
  (f (3 * Real.pi / 8) m = 0 → m = -1 ∧ ∃ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) ∧ f y m = -1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1329_132981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_factors_12_factorial_l1329_132961

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem min_sum_of_factors_12_factorial (p q r s : ℕ+) 
  (h : (p : ℕ) * q * r * s = factorial 12) : 
  ∀ (a b c d : ℕ+), (a : ℕ) * b * c * d = factorial 12 → p + q + r + s ≤ a + b + c + d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_factors_12_factorial_l1329_132961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_equals_zero_l1329_132962

theorem sin_sum_equals_zero (x y z : ℝ) 
  (h1 : Real.sin x = Real.tan y) 
  (h2 : Real.sin y = Real.tan z) 
  (h3 : Real.sin z = Real.tan x) : 
  Real.sin x + Real.sin y + Real.sin z = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_equals_zero_l1329_132962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_problem_l1329_132934

open Set

theorem complement_union_problem (U A B : Set ℕ) : 
  U = {0, 1, 2, 3, 4} →
  A = {1, 2, 3} →
  B = {2, 4} →
  (U \ A) ∪ B = {0, 2, 4} := by
  intros hU hA hB
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_problem_l1329_132934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_l1329_132980

/-- The total distance of Maria's trip -/
def D : ℝ := sorry

/-- The remaining distance after the first stop -/
noncomputable def remaining_after_first_stop : ℝ := D / 2

/-- The remaining distance after the second stop -/
noncomputable def remaining_after_second_stop : ℝ := 3 * D / 8

/-- The remaining distance after the second stop is 210 miles -/
axiom remaining_distance : remaining_after_second_stop = 210

theorem total_distance : D = 560 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_l1329_132980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_exponents_is_7_l1329_132943

-- Define 10!
def factorial_10 : ℕ := Nat.factorial 10

-- Define the largest perfect square that divides 10!
def largest_perfect_square (n : ℕ) : ℕ :=
  sorry

-- Define the square root of the largest perfect square
def sqrt_largest_perfect_square (n : ℕ) : ℕ :=
  sorry

-- Define a function to get the prime factors and their exponents
def prime_factors_with_exponents (n : ℕ) : List (ℕ × ℕ) :=
  sorry

-- Define a function to sum the exponents
def sum_of_exponents (factors : List (ℕ × ℕ)) : ℕ :=
  sorry

-- The main theorem
theorem sum_of_exponents_is_7 :
  sum_of_exponents (prime_factors_with_exponents (sqrt_largest_perfect_square factorial_10)) = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_exponents_is_7_l1329_132943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_fourth_l1329_132963

theorem sin_alpha_plus_pi_fourth (α : Real) 
  (h1 : Real.cos α = -4/5)
  (h2 : π < α ∧ α < 3*π/2) : 
  Real.sin (α + π/4) = -7*Real.sqrt 2/10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_fourth_l1329_132963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_distance_theorem_l1329_132931

-- Define the given conditions
def travel_time_minutes : ℚ := 42
def speed_mph : ℚ := 50

-- Define the function to calculate distance
def calculate_distance (time_minutes : ℚ) (speed : ℚ) : ℚ :=
  (time_minutes / 60) * speed

-- Theorem statement
theorem bus_distance_theorem : 
  calculate_distance travel_time_minutes speed_mph = 35 := by
  -- Unfold the definition of calculate_distance
  unfold calculate_distance
  -- Perform the calculation
  norm_num
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_distance_theorem_l1329_132931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_five_x_approx_l1329_132909

-- Define x as given in the problem
noncomputable def x : ℝ := (Real.log 2 / Real.log 4) ^ (Real.log 16 / Real.log 2)

-- State the theorem
theorem log_five_x_approx : 
  ∃ ε > 0, |Real.log x / Real.log 5 + 1.7228| < ε := by
  sorry

-- Note: We use an epsilon-delta definition for approximation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_five_x_approx_l1329_132909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_breaking_height_l1329_132968

/-- The height where a flagpole breaks, given its initial height and the distance from its base to where the fallen part touches the ground. -/
noncomputable def breakingHeight (initialHeight : ℝ) (fallenDistance : ℝ) : ℝ :=
  Real.sqrt ((initialHeight ^ 2 + fallenDistance ^ 2) / 2)

/-- Theorem stating the breaking height of a 10-meter flagpole that falls 3 meters from its base. -/
theorem flagpole_breaking_height :
  breakingHeight 10 3 = Real.sqrt (109 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_breaking_height_l1329_132968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_product_l1329_132972

noncomputable section

/-- The ellipse with equation x^2/2 + y^2 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | p.1^2 / 2 + p.2^2 = 1}

/-- The left focus of the ellipse -/
def F : ℝ × ℝ := (-1, 0)

/-- Point M -/
def M : ℝ × ℝ := (-5/4, 0)

/-- A line passing through the left focus F -/
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = k * (p.1 + 1)}

/-- The dot product of two 2D vectors -/
def dotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Vector from point p1 to point p2 -/
def vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2)

theorem ellipse_constant_product (k : ℝ) :
  ∀ A B : ℝ × ℝ,
  A ∈ Ellipse → B ∈ Ellipse →
  A ∈ Line k → B ∈ Line k →
  A ≠ B →
  dotProduct (vector M A) (vector M B) = -7/16 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_product_l1329_132972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_equals_zero_l1329_132987

theorem max_min_sum_equals_zero :
  ∃ (M m : ℝ), 
    (∀ (x y z : ℝ), 5 * (x + y + z) = x^2 + y^2 + z^2 → x*y + x*z + y*z ≤ M) ∧
    (∀ (x y z : ℝ), 5 * (x + y + z) = x^2 + y^2 + z^2 → m ≤ x*y + x*z + y*z) ∧
    M + 10 * m = 0 :=
by
  -- We'll use existence introduction (∃) to claim that M = 50 and m = -5
  use 50, -5
  -- Now we need to prove the three conjuncts
  constructor
  · -- Prove the upper bound
    sorry
  constructor
  · -- Prove the lower bound
    sorry
  · -- Prove that M + 10m = 0
    norm_num  -- This should evaluate 50 + 10 * (-5) to 0


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_equals_zero_l1329_132987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_ratios_on_line_l1329_132993

/-- Given an angle α whose terminal side lies on the line 5x - 12y = 0,
    prove the trigonometric ratios. -/
theorem trig_ratios_on_line (α : ℝ) 
  (h : ∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ 5 * x - 12 * y = 0 ∧ 
       x = Real.cos α * |Real.cos α| ∧ 
       y = Real.sin α * |Real.sin α|) : 
  |Real.sin α| = 5/13 ∧ 
  |Real.cos α| = 12/13 ∧ 
  Real.tan α = 5/12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_ratios_on_line_l1329_132993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_belfried_industries_tax_l1329_132923

/-- Calculates the special municipal payroll tax for a given payroll amount -/
noncomputable def specialMunicipalTax (payroll : ℝ) : ℝ :=
  if payroll ≤ 200000 then 0
  else (payroll - 200000) * 0.002

/-- Theorem: Belfried Industries, with a payroll of $400,000, pays $400 in special municipal tax -/
theorem belfried_industries_tax : specialMunicipalTax 400000 = 400 := by
  -- Unfold the definition of specialMunicipalTax
  unfold specialMunicipalTax
  -- Simplify the if-then-else expression
  simp
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_belfried_industries_tax_l1329_132923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l1329_132996

-- Define the plane
def plane (x y z : ℝ) : Prop := 3 * x - y + 2 * z = 18

-- Define the point we claim is closest to the origin
noncomputable def closest_point : ℝ × ℝ × ℝ := (27/7, -9/7, 18/7)

-- Theorem statement
theorem closest_point_on_plane :
  -- The point lies on the plane
  plane closest_point.1 closest_point.2.1 closest_point.2.2 ∧
  -- This point is closer to the origin than any other point on the plane
  ∀ (p : ℝ × ℝ × ℝ), plane p.1 p.2.1 p.2.2 →
    (closest_point.1^2 + closest_point.2.1^2 + closest_point.2.2^2 ≤ p.1^2 + p.2.1^2 + p.2.2^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l1329_132996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1329_132988

/-- The function we want to maximize -/
noncomputable def f (t : ℝ) : ℝ := (3^t - 4*t)*t / 9^t

/-- The theorem stating the maximum value of the function -/
theorem max_value_of_f :
  ∃ (M : ℝ), M = 1/16 ∧ ∀ (t : ℝ), f t ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1329_132988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equality_l1329_132948

theorem polynomial_equality (g : ℝ → ℝ) 
  (h : ∀ x, 7 * x^4 - 4 * x^2 + 2 + g x = -5 * x^3 + 2 * x^2 - 7) : 
  g = λ x ↦ -7 * x^4 - 5 * x^3 + 6 * x^2 - 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equality_l1329_132948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_condition_l1329_132960

open Real

-- Define the constants a and b with given conditions
variable (a b : ℝ) 
variable (ha : a > 1)
variable (hb : b > 0)
variable (hab : a > b)

-- Define the function f(x)
noncomputable def f (x : ℝ) := log (a^x - b^x)

-- State the theorem
theorem solution_set_condition :
  (∀ x : ℝ, f a b x > 0 ↔ x > 1) ↔ a - b = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_condition_l1329_132960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_and_inverse_l1329_132925

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 1 else -x + 1

theorem f_composition_and_inverse : 
  (f (f (-1)) = -1) ∧ 
  (f 0 = -1) ∧ 
  (f 2 = -1) ∧ 
  (∀ x : ℝ, f x = -1 → x = 0 ∨ x = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_and_inverse_l1329_132925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l1329_132915

noncomputable def f (k : ℤ) (x y : ℝ) : ℝ := (x^k + y^k + (-1)^k * (x + y)^k) / k

def valid_pair (m n : ℤ) : Prop :=
  m ≠ 0 ∧ n ≠ 0 ∧ m ≤ n ∧ m + n ≠ 0 ∧
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x + y ≠ 0 →
    f m x y * f n x y = f (m + n) x y

theorem function_identity :
  ∀ m n : ℤ, valid_pair m n ↔ (m = 2 ∧ n = 3) ∨ (m = 2 ∧ n = 5) ∨ (m = -1 ∧ n = 3) :=
by sorry

#check function_identity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l1329_132915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1329_132911

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y + 7 = 0

-- Theorem statement
theorem chord_length :
  ∀ P Q : ℝ × ℝ,
  line P.1 P.2 → line Q.1 Q.2 →
  circle_eq P.1 P.2 → circle_eq Q.1 Q.2 →
  P ≠ Q →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1329_132911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1329_132917

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem function_properties (ω φ : ℝ) (h1 : ω > 0) (h2 : -π/2 < φ ∧ φ < 0)
  (h3 : Real.tan φ = -Real.sqrt 3)
  (h4 : ∃ x₁ x₂, |f ω φ x₁ - f ω φ x₂| = 4 ∧ |x₁ - x₂| = π/3 ∧
    ∀ y₁ y₂, |f ω φ y₁ - f ω φ y₂| = 4 → |y₁ - y₂| ≥ π/3) :
  (∀ x, f ω φ x = 2 * Real.sin (3*x - π/3)) ∧
  (∀ k : ℤ, ∀ x, (x ≥ -π/18 + 2*k*π/3 ∧ x ≤ 5*π/18 + 2*k*π/3) →
    (∀ y, y ∈ Set.Icc (-π/18 + 2*k*π/3) (5*π/18 + 2*k*π/3) → f ω φ x ≤ f ω φ y → x ≤ y)) ∧
  (∀ m : ℝ, (∀ x, x ∈ Set.Icc 0 (π/6) → m * f ω φ x + 2*m ≥ f ω φ x) → m ≥ 1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1329_132917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_regular_polygon_sum_l1329_132956

/-- A function that checks if complex numbers form a regular polygon on the unit circle -/
def is_regular_polygon_on_unit_circle {n : ℕ} (z : Fin n → ℂ) : Prop :=
  ∀ i : Fin n, Complex.abs (z i) = 1 ∧ 
  ∃ θ : ℝ, ∀ i : Fin n, z i = Complex.exp (Complex.I * (θ * ↑i + 2 * Real.pi / ↑n))

/-- The main theorem statement -/
theorem unique_regular_polygon_sum : 
  ∃! (n : ℕ), n ≥ 2 ∧ 
  ∀ (z : Fin n → ℂ), 
    (∀ i : Fin n, Complex.abs (z i) = 1) → 
    (Finset.sum Finset.univ (λ i => z i) = n) → 
    is_regular_polygon_on_unit_circle z :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_regular_polygon_sum_l1329_132956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spectator_rearrangement_l1329_132995

/-- Represents a seating arrangement of spectators -/
def SeatingArrangement (n : ℕ+) := Fin n → Fin n

/-- A derangement is a seating arrangement where no spectator is in their correct seat -/
def IsDerangement {n : ℕ+} (arrangement : SeatingArrangement n) : Prop :=
  ∀ i : Fin n, arrangement i ≠ i

/-- Represents a swap of two adjacent spectators -/
inductive AdjacentSwap {n : ℕ+} : SeatingArrangement n → SeatingArrangement n → Prop
  | swap (arrangement : SeatingArrangement n) (i : Fin n) (h : i.val + 1 < n) :
    AdjacentSwap arrangement (Function.update (Function.update arrangement i (arrangement ⟨i.val + 1, h⟩)) ⟨i.val + 1, h⟩ (arrangement i))

/-- A valid swap is one where at least one of the swapped spectators is not in their correct seat -/
def IsValidSwap {n : ℕ+} (arrangement : SeatingArrangement n) (i : Fin n) (h : i.val + 1 < n) : Prop :=
  arrangement i ≠ i ∨ arrangement ⟨i.val + 1, h⟩ ≠ ⟨i.val + 1, h⟩

/-- The main theorem: any derangement can be transformed into the correct arrangement through valid swaps -/
theorem spectator_rearrangement {n : ℕ+} (initial : SeatingArrangement n) :
  IsDerangement initial →
  ∃ (final : SeatingArrangement n),
    (∀ i : Fin n, final i = i) ∧
    (∃ (swaps : List (Σ' (i : Fin n), i.val + 1 < n)),
      (∀ swap ∈ swaps, IsValidSwap (List.foldl (λ arr s => Function.update (Function.update arr s.1 (arr ⟨s.1.val + 1, s.2⟩)) ⟨s.1.val + 1, s.2⟩ (arr s.1)) initial (swaps.take (swaps.indexOf swap))) swap.1 swap.2) ∧
      List.foldl (λ arr s => Function.update (Function.update arr s.1 (arr ⟨s.1.val + 1, s.2⟩)) ⟨s.1.val + 1, s.2⟩ (arr s.1)) initial swaps = final) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spectator_rearrangement_l1329_132995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_family_age_theorem_l1329_132932

/-- Calculates the average age of a family at the time of the youngest member's birth. -/
noncomputable def family_average_age_at_youngest_birth (current_average_age : ℝ) (family_size : ℕ) 
  (youngest_age : ℝ) (second_youngest_age : ℝ) : ℝ :=
  let total_age := current_average_age * (family_size : ℝ)
  let remaining_members_age := total_age - youngest_age - second_youngest_age
  let remaining_members_age_at_birth := remaining_members_age - (youngest_age * ((family_size : ℝ) - 2))
  remaining_members_age_at_birth / ((family_size : ℝ) - 1)

/-- Theorem stating the average age of the family at the youngest member's birth. -/
theorem family_age_theorem :
  let current_average_age : ℝ := 25
  let family_size : ℕ := 7
  let youngest_age : ℝ := 3
  let second_youngest_age : ℝ := 8
  abs (family_average_age_at_youngest_birth current_average_age family_size youngest_age second_youngest_age - 24.83) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not necessary for building and may cause issues
-- #eval family_average_age_at_youngest_birth 25 7 3 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_family_age_theorem_l1329_132932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_bound_l1329_132950

/-- Sequence defined by the given recurrence relation -/
noncomputable def a (a₀ : ℝ) : ℕ → ℝ
  | 0 => 1
  | 1 => a₀
  | n + 2 => ((a a₀ n)^2 / (a a₀ (n + 1))^2 - 2) * (a a₀ (n + 1))

/-- Sum of reciprocals of sequence elements up to index k -/
noncomputable def sum_reciprocals (a₀ : ℝ) (k : ℕ) : ℝ :=
  (Finset.range (k + 1)).sum (λ i => 1 / (a a₀ i))

/-- Main theorem statement -/
theorem sum_reciprocals_bound (a₀ : ℝ) (h : a₀ > 2) :
  ∀ k : ℕ, sum_reciprocals a₀ k < (a₀ + 2 - Real.sqrt (a₀^2 - 4)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_bound_l1329_132950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1329_132973

theorem triangle_inequality (A B C : ℝ) (k : ℝ) : 
  A + B + C = π → k ≥ 0 → 
  Real.cos B * Real.cos C * (Real.sin (A/2))^k + 
  Real.cos C * Real.cos A * (Real.sin (B/2))^k + 
  Real.cos A * Real.cos B * (Real.sin (C/2))^k < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1329_132973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stick_markings_l1329_132938

/-- Represents a marking on the stick --/
structure Marking where
  position : ℚ
  deriving Repr

/-- Represents a stick with markings --/
structure MarkedStick where
  length : ℚ
  markings : List Marking
  deriving Repr

/-- Creates markings for a given denominator --/
def createMarkings (denom : ℕ) : List Marking :=
  sorry

/-- Combines two lists of markings, removing duplicates --/
def combineMarkings (m1 m2 : List Marking) : List Marking :=
  sorry

/-- Theorem: The number of unique markings on a 1-foot stick marked in 1/3 and 1/5 portions is 9 --/
theorem stick_markings :
  let stick : MarkedStick :=
    { length := 1
    , markings := combineMarkings (createMarkings 3) (createMarkings 5) }
  stick.markings.length = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stick_markings_l1329_132938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_satisfying_conditions_l1329_132952

/-- Represents a three-digit number in base 10 -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its decimal value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Converts a natural number to its base 9 representation -/
def toBase9 (n : Nat) : List Nat :=
  sorry

theorem unique_number_satisfying_conditions :
  ∃! x : ThreeDigitNumber,
    (x.ones = 6) ∧
    (let base9 := toBase9 x.toNat
     base9.length = 3 ∧
     base9.get? 1 = some 4 ∧
     base9.get? 0 = base9.get? 2) ∧
    x.toNat = 446 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_satisfying_conditions_l1329_132952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lila_average_speed_l1329_132906

/-- Calculates the average speed given distance and time -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- Represents Lila's ride -/
structure Ride where
  distance1 : ℝ
  speed1 : ℝ
  distance2 : ℝ
  speed2 : ℝ
  break_time : ℝ

theorem lila_average_speed (r : Ride) 
  (h1 : r.distance1 = 50)
  (h2 : r.speed1 = 20)
  (h3 : r.distance2 = 20)
  (h4 : r.speed2 = 40)
  (h5 : r.break_time = 0.5) :
  average_speed (r.distance1 + r.distance2) 
    (r.distance1 / r.speed1 + r.distance2 / r.speed2 + r.break_time) = 20 := by
  sorry

#eval "Lila's average speed theorem is defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lila_average_speed_l1329_132906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pat_initial_stickers_l1329_132958

/-- The number of stickers Pat had at the beginning of the week. -/
def initial_stickers : ℕ := sorry

/-- The number of stickers Pat earned during the week. -/
def earned_stickers : ℕ := 22

/-- The total number of stickers Pat had at the end of the week. -/
def final_stickers : ℕ := 61

/-- Theorem stating that Pat had 39 stickers at the beginning of the week. -/
theorem pat_initial_stickers :
  initial_stickers = 39 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pat_initial_stickers_l1329_132958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jet_flow_pump_time_l1329_132941

/-- The rate at which the JetFlow pump operates in gallons per hour -/
noncomputable def pump_rate : ℚ := 600

/-- The volume of water to be pumped in gallons -/
noncomputable def water_volume : ℚ := 900

/-- The time required to pump the given volume of water -/
noncomputable def pump_time : ℚ := water_volume / pump_rate

theorem jet_flow_pump_time : pump_time = 3/2 := by
  -- Unfold definitions
  unfold pump_time water_volume pump_rate
  -- Simplify the fraction
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jet_flow_pump_time_l1329_132941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l1329_132920

noncomputable def h (x : ℝ) : ℝ := (5*x - 2) / (2*x - 10)

theorem domain_of_h :
  Set.Iio 5 ∪ Set.Ioi 5 = {x | ∃ y, h x = y} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l1329_132920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sin_minus_x_negative_l1329_132902

open Real

theorem exists_sin_minus_x_negative :
  ∃ x : ℝ, x ∈ Set.Ioo 0 (π / 2) ∧ sin x - x < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sin_minus_x_negative_l1329_132902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1329_132986

/-- An ellipse with center at origin and eccentricity 1/2 -/
structure Ellipse where
  eccentricity : ℝ
  center_x : ℝ
  center_y : ℝ
  eccentricity_eq : eccentricity = 1/2
  center_origin : center_x = 0 ∧ center_y = 0

/-- A parabola with equation y² = 8x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  eq : ∀ x y, equation x y ↔ y^2 = 8*x

/-- The focus of a parabola -/
def parabola_focus : ℝ × ℝ := (2, 0)

/-- The directrix of a parabola -/
def parabola_directrix : ℝ → Prop := fun x ↦ x = -2

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The theorem to be proved -/
theorem intersection_distance (e : Ellipse) (p : Parabola) : 
  ∃ A B : ℝ × ℝ, 
    (parabola_directrix A.1 ∧ 
     parabola_directrix B.1 ∧ 
     A ≠ B ∧ 
     distance A B = 6) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1329_132986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suitcase_ratio_change_l1329_132976

/-- Given a suitcase with books, clothes, and electronics, prove the new ratio of books to clothes after removing some clothing. -/
theorem suitcase_ratio_change (initial_ratio_books : ℚ) (initial_ratio_clothes : ℚ) (initial_ratio_electronics : ℚ)
  (electronics_weight : ℚ) (removed_clothes_weight : ℚ) :
  initial_ratio_books = 5 →
  initial_ratio_clothes = 4 →
  initial_ratio_electronics = 2 →
  electronics_weight = 9 →
  removed_clothes_weight = 9 →
  let total_ratio := initial_ratio_books + initial_ratio_clothes + initial_ratio_electronics
  let books_weight := (initial_ratio_books / initial_ratio_electronics) * electronics_weight
  let initial_clothes_weight := (initial_ratio_clothes / initial_ratio_electronics) * electronics_weight
  let new_clothes_weight := initial_clothes_weight - removed_clothes_weight
  (books_weight / new_clothes_weight) = 5/2 := by
  intro h1 h2 h3 h4 h5
  sorry

#check suitcase_ratio_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suitcase_ratio_change_l1329_132976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_permutations_l1329_132927

theorem banana_permutations : 
  (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_permutations_l1329_132927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1329_132984

/-- A hyperbola with foci F₁ and F₂ -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (C : Hyperbola) : ℝ := sorry

/-- A point on a hyperbola -/
def point_on_hyperbola (C : Hyperbola) (P : ℝ × ℝ) : Prop := sorry

/-- The angle between two points and a vertex -/
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

/-- The distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem hyperbola_eccentricity (C : Hyperbola) (P : ℝ × ℝ) :
  point_on_hyperbola C P →
  angle C.F₁ P C.F₂ = π / 3 →
  distance C.F₁ P = 3 * distance C.F₂ P →
  eccentricity C = Real.sqrt 7 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1329_132984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l1329_132978

theorem circle_area_ratio (C D : ℝ) (hC : C > 0) (hD : D > 0) :
  (60 / 360) * (2 * Real.pi * C) = (40 / 360) * (2 * Real.pi * D) →
  (C^2 / D^2 : ℝ) = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l1329_132978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_range_l1329_132992

/-- An ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis -/
structure Ellipse (k : ℝ) where
  eq : ∀ (x y : ℝ), x^2 + k*y^2 = 2
  foci_on_y : ∀ (f : ℝ × ℝ), f.1 = 0 -- Removed IsFocus as it's not defined

/-- The range of k for which the Ellipse structure is valid -/
def valid_k_range : Set ℝ :=
  {k : ℝ | ∃ e : Ellipse k, True}

/-- The theorem stating the range of k for a valid ellipse -/
theorem ellipse_k_range : valid_k_range = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_range_l1329_132992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_condition_longest_chord_l1329_132982

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 1

-- Define the line equation
def line (x y m : ℝ) : Prop := y = x + m

-- Theorem for the intersection condition
theorem intersection_condition (m : ℝ) :
  (∃ x y : ℝ, ellipse x y ∧ line x y m) ↔ -Real.sqrt 5 / 2 ≤ m ∧ m ≤ Real.sqrt 5 / 2 :=
sorry

-- Define chord length
noncomputable def chord_length (m : ℝ) : ℝ := 
  (2 * Real.sqrt 2 * Real.sqrt (5 - 4 * m^2)) / 5

-- Theorem for the longest chord
theorem longest_chord :
  ∀ m : ℝ, chord_length 0 ≥ chord_length m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_condition_longest_chord_l1329_132982
