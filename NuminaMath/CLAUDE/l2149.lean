import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l2149_214935

theorem equation_solution : 
  ∃ x : ℚ, (x + 10) / (x - 4) = (x - 3) / (x + 6) ↔ x = -48 / 23 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2149_214935


namespace NUMINAMATH_CALUDE_inequality_range_l2149_214937

theorem inequality_range (a : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π / 2 → 
    Real.sin (2 * θ) - (2 * Real.sqrt 2 + Real.sqrt 2 * a) * Real.sin (θ + π / 4) - 
    (2 * Real.sqrt 2 / Real.cos (θ - π / 4)) > -3 - 2 * a) → 
  a > 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l2149_214937


namespace NUMINAMATH_CALUDE_cube_volume_from_diagonal_l2149_214977

/-- Given a cube with space diagonal 5√3, prove its volume is 125 -/
theorem cube_volume_from_diagonal (s : ℝ) : 
  s * Real.sqrt 3 = 5 * Real.sqrt 3 → s^3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_diagonal_l2149_214977


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2149_214927

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.sqrt (18 * x) * Real.sqrt (2 * x) * Real.sqrt (25 * x) * Real.sqrt (5 * x) = 50 ∧ 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ |x - 0.8632| < ε := by
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2149_214927


namespace NUMINAMATH_CALUDE_limit_x_plus_sin_x_power_sin_x_plus_x_l2149_214994

/-- The limit of (x + sin x)^(sin x + x) as x approaches π is π^π. -/
theorem limit_x_plus_sin_x_power_sin_x_plus_x (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x : ℝ, 0 < |x - π| ∧ |x - π| < δ →
    |(x + Real.sin x)^(Real.sin x + x) - π^π| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_x_plus_sin_x_power_sin_x_plus_x_l2149_214994


namespace NUMINAMATH_CALUDE_number_of_girls_in_class_l2149_214957

/-- Given a class where there are 3 more girls than boys and the total number of students is 41,
    prove that the number of girls in the class is 22. -/
theorem number_of_girls_in_class (boys girls : ℕ) : 
  girls = boys + 3 → 
  boys + girls = 41 → 
  girls = 22 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_in_class_l2149_214957


namespace NUMINAMATH_CALUDE_quadratic_discriminant_nonnegative_l2149_214951

/-- 
Given a quadratic equation ax^2 + 4bx + c = 0 where a, b, and c form an arithmetic progression,
prove that the discriminant Δ is always non-negative.
-/
theorem quadratic_discriminant_nonnegative 
  (a b c : ℝ) 
  (h_progression : ∃ d : ℝ, b = a + d ∧ c = a + 2*d) : 
  (4*b)^2 - 4*a*c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_nonnegative_l2149_214951


namespace NUMINAMATH_CALUDE_ellipse_foci_l2149_214936

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 64 + y^2 / 100 = 1

/-- The coordinates of a focus of the ellipse -/
def focus_coordinate : ℝ × ℝ := (0, 6)

/-- Theorem stating that the given coordinates are the foci of the ellipse -/
theorem ellipse_foci :
  (ellipse_equation (focus_coordinate.1) (focus_coordinate.2) ∧
   ellipse_equation (focus_coordinate.1) (-focus_coordinate.2)) ∧
  (∀ x y : ℝ, ellipse_equation x y →
    (x^2 + y^2 < focus_coordinate.1^2 + focus_coordinate.2^2 ∨
     x^2 + y^2 = focus_coordinate.1^2 + focus_coordinate.2^2)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_l2149_214936


namespace NUMINAMATH_CALUDE_inequality_solution_existence_l2149_214906

theorem inequality_solution_existence (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_existence_l2149_214906


namespace NUMINAMATH_CALUDE_f_properties_l2149_214912

noncomputable def f (x : ℝ) := 2 * Real.cos x * (Real.cos x + Real.sqrt 3 * Real.sin x)

theorem f_properties :
  ∃ (T : ℝ),
    (∀ x, f (x + T) = f x) ∧
    T = π ∧
    (∀ k : ℤ, StrictMonoOn f (Set.Ioo (↑k * π - π / 3) (↑k * π + π / 6))) ∧
    (∀ x ∈ Set.Icc 0 (π / 2), f x ≤ 3) ∧
    (∃ x ∈ Set.Icc 0 (π / 2), f x = 3) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2149_214912


namespace NUMINAMATH_CALUDE_difference_of_cubes_divisible_by_27_l2149_214975

theorem difference_of_cubes_divisible_by_27 (a b : ℤ) :
  ∃ k : ℤ, (3 * a + 2)^3 - (3 * b + 2)^3 = 27 * k := by sorry

end NUMINAMATH_CALUDE_difference_of_cubes_divisible_by_27_l2149_214975


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2149_214996

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 4}
def B : Set ℝ := {0, 2, 4, 6}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2149_214996


namespace NUMINAMATH_CALUDE_rebecca_marbles_l2149_214991

theorem rebecca_marbles (group_size : ℕ) (num_groups : ℕ) (total_marbles : ℕ) : 
  group_size = 4 → num_groups = 5 → total_marbles = group_size * num_groups → total_marbles = 20 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_marbles_l2149_214991


namespace NUMINAMATH_CALUDE_M_union_N_eq_M_l2149_214968

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | |p.1 * p.2| = 1 ∧ p.1 > 0}
def N : Set (ℝ × ℝ) := {p | Real.arctan p.1 + Real.arctan p.2 = Real.pi}

-- Theorem statement
theorem M_union_N_eq_M : M ∪ N = M := by
  sorry

end NUMINAMATH_CALUDE_M_union_N_eq_M_l2149_214968


namespace NUMINAMATH_CALUDE_apple_percentage_l2149_214954

theorem apple_percentage (initial_apples initial_oranges added_oranges : ℕ) :
  initial_apples = 10 →
  initial_oranges = 5 →
  added_oranges = 5 →
  (initial_apples : ℚ) / (initial_apples + initial_oranges + added_oranges : ℚ) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_apple_percentage_l2149_214954


namespace NUMINAMATH_CALUDE_loan_interest_rate_l2149_214911

theorem loan_interest_rate (principal : ℝ) (total_paid : ℝ) (time : ℝ) : 
  principal = 150 → 
  total_paid = 159 → 
  time = 1 → 
  (total_paid - principal) / (principal * time) = 0.06 := by
sorry

end NUMINAMATH_CALUDE_loan_interest_rate_l2149_214911


namespace NUMINAMATH_CALUDE_total_games_is_506_l2149_214901

/-- Represents the structure of a soccer league --/
structure SoccerLeague where
  total_teams : Nat
  divisions : Nat
  teams_per_division : Nat
  regular_season_intra_division_matches : Nat
  regular_season_cross_division_matches : Nat
  mid_season_tournament_matches_per_team : Nat
  mid_season_tournament_intra_division_matches : Nat
  playoff_teams_per_division : Nat
  playoff_stages : Nat

/-- Calculates the total number of games in the soccer league season --/
def total_games (league : SoccerLeague) : Nat :=
  -- Regular season games
  (league.teams_per_division * (league.teams_per_division - 1) * league.regular_season_intra_division_matches * league.divisions) / 2 +
  (league.total_teams * league.regular_season_cross_division_matches) / 2 +
  -- Mid-season tournament games
  (league.total_teams * league.mid_season_tournament_matches_per_team) / 2 +
  -- Playoff games
  (league.playoff_teams_per_division * 2 - 1) * 2 * league.divisions

/-- Theorem stating that the total number of games in the given league structure is 506 --/
theorem total_games_is_506 (league : SoccerLeague) 
  (h1 : league.total_teams = 24)
  (h2 : league.divisions = 2)
  (h3 : league.teams_per_division = 12)
  (h4 : league.regular_season_intra_division_matches = 2)
  (h5 : league.regular_season_cross_division_matches = 4)
  (h6 : league.mid_season_tournament_matches_per_team = 5)
  (h7 : league.mid_season_tournament_intra_division_matches = 3)
  (h8 : league.playoff_teams_per_division = 4)
  (h9 : league.playoff_stages = 3) :
  total_games league = 506 := by
  sorry

end NUMINAMATH_CALUDE_total_games_is_506_l2149_214901


namespace NUMINAMATH_CALUDE_cubic_root_product_l2149_214976

theorem cubic_root_product (x : ℝ) : 
  (∃ p q r : ℝ, x^3 - 15*x^2 + 60*x - 36 = (x - p) * (x - q) * (x - r)) → 
  (∃ p q r : ℝ, x^3 - 15*x^2 + 60*x - 36 = (x - p) * (x - q) * (x - r) ∧ p * q * r = 36) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_product_l2149_214976


namespace NUMINAMATH_CALUDE_no_consecutive_power_l2149_214925

theorem no_consecutive_power (n : ℕ) : ¬ ∃ (m k : ℕ), k ≥ 2 ∧ n * (n + 1) = m ^ k := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_power_l2149_214925


namespace NUMINAMATH_CALUDE_triangle_construction_theorem_l2149_214997

-- Define the necessary structures
structure Line where
  -- Add necessary fields for a line

structure Point where
  -- Add necessary fields for a point

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the concept of a point lying on a line
def Point.liesOn (p : Point) (l : Line) : Prop := sorry

-- Define the concept of a point being the foot of an altitude
def isAltitudeFoot (foot : Point) (vertex : Point) (base1 : Point) (base2 : Point) : Prop := sorry

-- Main theorem
theorem triangle_construction_theorem 
  (l : Line) (A₁ : Point) (B₁ : Point) : 
  ∃ (t : Triangle), 
    (t.A.liesOn l) ∧ 
    (t.B.liesOn l) ∧ 
    (isAltitudeFoot A₁ t.A t.B t.C) ∧ 
    (isAltitudeFoot B₁ t.B t.A t.C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_construction_theorem_l2149_214997


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2149_214969

theorem rectangle_perimeter (a b c w : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (h4 : w = 6) : 
  let triangle_area := (1/2) * a * b
  let rectangle_length := triangle_area / w
  2 * (rectangle_length + w) = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2149_214969


namespace NUMINAMATH_CALUDE_cos_beta_value_l2149_214923

theorem cos_beta_value (α β : Real) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.tan α = 2) (h4 : Real.sin (α + β) = Real.sqrt 2 / 2) :
  Real.cos β = Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_beta_value_l2149_214923


namespace NUMINAMATH_CALUDE_ice_cream_shop_sales_l2149_214910

/-- Given a ratio of sugar cones to waffle cones and the number of waffle cones sold,
    calculate the number of sugar cones sold. -/
def sugar_cones_sold (sugar_ratio : ℕ) (waffle_ratio : ℕ) (waffle_cones : ℕ) : ℕ :=
  (sugar_ratio * waffle_cones) / waffle_ratio

/-- Theorem stating that given the specific ratio and number of waffle cones,
    the number of sugar cones sold is 45. -/
theorem ice_cream_shop_sales : sugar_cones_sold 5 4 36 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_shop_sales_l2149_214910


namespace NUMINAMATH_CALUDE_composite_solid_volume_l2149_214988

/-- The volume of a composite solid consisting of a rectangular prism and a cylinder -/
theorem composite_solid_volume :
  ∀ (prism_length prism_width prism_height cylinder_radius cylinder_height overlap_volume : ℝ),
  prism_length = 2 →
  prism_width = 2 →
  prism_height = 1 →
  cylinder_radius = 1 →
  cylinder_height = 3 →
  overlap_volume = π / 2 →
  prism_length * prism_width * prism_height + π * cylinder_radius^2 * cylinder_height - overlap_volume = 4 + 5 * π / 2 :=
by sorry

end NUMINAMATH_CALUDE_composite_solid_volume_l2149_214988


namespace NUMINAMATH_CALUDE_arithmetic_trapezoid_area_l2149_214966

/-- Represents a trapezoid with bases and altitude in arithmetic progression -/
structure ArithmeticTrapezoid where
  b : ℝ  -- altitude
  d : ℝ  -- common difference

/-- The area of an arithmetic trapezoid is b^2 -/
theorem arithmetic_trapezoid_area (t : ArithmeticTrapezoid) : 
  (1 / 2 : ℝ) * ((t.b + t.d) + (t.b - t.d)) * t.b = t.b^2 := by
  sorry

#check arithmetic_trapezoid_area

end NUMINAMATH_CALUDE_arithmetic_trapezoid_area_l2149_214966


namespace NUMINAMATH_CALUDE_final_savings_calculation_l2149_214934

/-- Calculate final savings after a period of time given initial savings, monthly income, and monthly expenses. -/
def calculate_final_savings (initial_savings : ℕ) (monthly_income : ℕ) (monthly_expenses : ℕ) (months : ℕ) : ℕ :=
  initial_savings + months * monthly_income - months * monthly_expenses

/-- Theorem: Given the specific financial conditions, the final savings will be 1106900 rubles. -/
theorem final_savings_calculation :
  let initial_savings : ℕ := 849400
  let monthly_income : ℕ := 45000 + 35000 + 7000 + 10000 + 13000
  let monthly_expenses : ℕ := 30000 + 10000 + 5000 + 4500 + 9000
  let months : ℕ := 5
  calculate_final_savings initial_savings monthly_income monthly_expenses months = 1106900 := by
  sorry

end NUMINAMATH_CALUDE_final_savings_calculation_l2149_214934


namespace NUMINAMATH_CALUDE_q_polynomial_form_l2149_214900

/-- Given a function q(x) satisfying the equation 
    q(x) + (2x^6 + 4x^4 + 12x^2) = (10x^4 + 36x^3 + 37x^2 + 5),
    prove that q(x) = -2x^6 + 6x^4 + 36x^3 + 25x^2 + 5 -/
theorem q_polynomial_form (x : ℝ) (q : ℝ → ℝ) 
    (h : ∀ x, q x + (2*x^6 + 4*x^4 + 12*x^2) = 10*x^4 + 36*x^3 + 37*x^2 + 5) :
  q x = -2*x^6 + 6*x^4 + 36*x^3 + 25*x^2 + 5 := by
  sorry

end NUMINAMATH_CALUDE_q_polynomial_form_l2149_214900


namespace NUMINAMATH_CALUDE_parabola_equation_from_axis_and_focus_l2149_214924

/-- A parabola with given axis of symmetry and focus -/
structure Parabola where
  axis_of_symmetry : ℝ
  focus : ℝ × ℝ

/-- The equation of a parabola given its parameters -/
def parabola_equation (p : Parabola) : ℝ → ℝ → Prop :=
  fun x y => y^2 = -4 * x

/-- Theorem: For a parabola with axis of symmetry x = 1 and focus at (-1, 0), its equation is y² = -4x -/
theorem parabola_equation_from_axis_and_focus :
  ∀ (p : Parabola), p.axis_of_symmetry = 1 ∧ p.focus = (-1, 0) →
  parabola_equation p = fun x y => y^2 = -4 * x :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_from_axis_and_focus_l2149_214924


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l2149_214931

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem reflection_across_x_axis :
  let A : Point := { x := 2, y := 3 }
  reflectAcrossXAxis A = { x := 2, y := -3 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l2149_214931


namespace NUMINAMATH_CALUDE_log_equation_solution_l2149_214995

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 8 + Real.log (x^3) / Real.log 4 = 9 →
  x = 2^(54/11) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2149_214995


namespace NUMINAMATH_CALUDE_order_of_abc_l2149_214985

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define a, b, and c as real numbers
variable (a b c : ℝ)

-- State the theorem
theorem order_of_abc (hf : Monotone f) (ha : a = f 2 ∧ a < 0) 
  (hb : f b = 2) (hc : f c = 0) : b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l2149_214985


namespace NUMINAMATH_CALUDE_cassie_daily_water_consumption_l2149_214945

/-- Calculates the number of cups of water consumed daily given the water bottle capacity, ounces per cup, and number of refills. -/
def cups_of_water_daily (bottle_capacity : ℕ) (ounces_per_cup : ℕ) (refills : ℕ) : ℕ :=
  (bottle_capacity / ounces_per_cup) * refills

/-- Theorem stating that given specific conditions, Cassie drinks 12 cups of water daily. -/
theorem cassie_daily_water_consumption :
  cups_of_water_daily 16 8 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_cassie_daily_water_consumption_l2149_214945


namespace NUMINAMATH_CALUDE_triangle_theorem_l2149_214959

theorem triangle_theorem (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → -- Triangle condition
  a > 0 ∧ b > 0 ∧ c > 0 → -- Positive side lengths
  a * (Real.sin A - Real.sin B) + b * Real.sin B = c * Real.sin C → -- Line condition
  2 * (Real.cos (A / 2))^2 - 2 * (Real.sin (B / 2))^2 = Real.sqrt 3 / 2 → -- Given equation
  A < B → -- Given inequality
  C = π / 3 ∧ c / a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2149_214959


namespace NUMINAMATH_CALUDE_inequality_proof_l2149_214938

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^4 ≤ x^2 + y^3) : x^3 + y^3 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2149_214938


namespace NUMINAMATH_CALUDE_line_through_points_specific_line_equation_l2149_214929

/-- A line passing through two given points has a specific equation -/
theorem line_through_points (x₁ y₁ x₂ y₂ : ℝ) (h : x₁ ≠ x₂) :
  ∃ k b : ℝ, ∀ x y : ℝ, y = k * x + b ↔ (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) :=
sorry

/-- The line passing through (2, 5) and (1, 1) has the equation y = 4x - 3 -/
theorem specific_line_equation :
  ∃ k b : ℝ, (k = 4 ∧ b = -3) ∧
    (∀ x y : ℝ, y = k * x + b ↔ (x = 2 ∧ y = 5) ∨ (x = 1 ∧ y = 1)) :=
sorry

end NUMINAMATH_CALUDE_line_through_points_specific_line_equation_l2149_214929


namespace NUMINAMATH_CALUDE_fencing_length_l2149_214984

/-- Calculates the required fencing length for a rectangular field -/
theorem fencing_length (area : ℝ) (uncovered_side : ℝ) : area = 680 ∧ uncovered_side = 40 →
  2 * (area / uncovered_side) + uncovered_side = 74 := by
  sorry


end NUMINAMATH_CALUDE_fencing_length_l2149_214984


namespace NUMINAMATH_CALUDE_profit_at_4_max_profit_price_l2149_214918

noncomputable section

-- Define the sales volume function
def sales_volume (x : ℝ) : ℝ := 10 / (x - 2) + 4 * (x - 6)^2

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - 2) * sales_volume x

-- Theorem for part (1)
theorem profit_at_4 : profit 4 = 42 := by sorry

-- Theorem for part (2)
theorem max_profit_price : 
  ∃ (x : ℝ), 2 < x ∧ x < 6 ∧ 
  (∀ (y : ℝ), 2 < y ∧ y < 6 → profit y ≤ profit x) ∧
  x = 10/3 := by sorry

end

end NUMINAMATH_CALUDE_profit_at_4_max_profit_price_l2149_214918


namespace NUMINAMATH_CALUDE_farmer_corn_rows_l2149_214916

/-- Given a farmer's crop scenario, prove the number of corn stalk rows. -/
theorem farmer_corn_rows (C : ℕ) : 
  (C * 9 + 5 * 30 = 240) → C = 10 := by
  sorry

end NUMINAMATH_CALUDE_farmer_corn_rows_l2149_214916


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l2149_214907

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/4
  let a₂ : ℚ := 28/9
  let a₃ : ℚ := 112/27
  let r : ℚ := a₂ / a₁
  r = 16/9 := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l2149_214907


namespace NUMINAMATH_CALUDE_vanessa_savings_time_l2149_214940

/-- Calculates the number of weeks needed to save for a dress -/
def weeks_to_save (dress_cost : ℕ) (initial_savings : ℕ) (weekly_allowance : ℕ) (weekly_spending : ℕ) : ℕ :=
  let additional_needed := dress_cost - initial_savings
  let weekly_savings := weekly_allowance - weekly_spending
  (additional_needed + weekly_savings - 1) / weekly_savings

/-- Proof that Vanessa needs exactly 3 weeks to save for the dress -/
theorem vanessa_savings_time : 
  weeks_to_save 80 20 30 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_savings_time_l2149_214940


namespace NUMINAMATH_CALUDE_initial_puppies_count_l2149_214922

/-- The number of puppies Sandy's dog initially had -/
def initial_puppies : ℕ := sorry

/-- The number of puppies Sandy gave away -/
def puppies_given_away : ℕ := 4

/-- The number of puppies Sandy has left -/
def puppies_left : ℕ := 4

/-- Theorem stating that the initial number of puppies is 8 -/
theorem initial_puppies_count : initial_puppies = 8 := by sorry

end NUMINAMATH_CALUDE_initial_puppies_count_l2149_214922


namespace NUMINAMATH_CALUDE_quadratic_roots_l2149_214909

theorem quadratic_roots (a b c : ℝ) (ha : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + c = 0 ∧ a * y^2 + c = 0) →
  (∃ u v : ℝ, u ≠ v ∧ a * u^2 + b * u + c = 0 ∧ a * v^2 + b * v + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2149_214909


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l2149_214980

/-- The longest segment in a cylinder -/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 12) :
  Real.sqrt ((2 * r) ^ 2 + h ^ 2) = Real.sqrt 244 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l2149_214980


namespace NUMINAMATH_CALUDE_constant_b_value_l2149_214953

theorem constant_b_value (x y b : ℝ) 
  (h1 : (7 * x + b * y) / (x - 2 * y) = 25)
  (h2 : x / (2 * y) = 3 / 2) : 
  b = 4 := by sorry

end NUMINAMATH_CALUDE_constant_b_value_l2149_214953


namespace NUMINAMATH_CALUDE_equal_chance_in_all_methods_l2149_214974

/-- Represents a sampling method -/
structure SamplingMethod where
  name : String
  equal_chance : Bool

/-- Simple random sampling -/
def simple_random_sampling : SamplingMethod :=
  { name := "Simple Random Sampling", equal_chance := true }

/-- Systematic sampling -/
def systematic_sampling : SamplingMethod :=
  { name := "Systematic Sampling", equal_chance := true }

/-- Stratified sampling -/
def stratified_sampling : SamplingMethod :=
  { name := "Stratified Sampling", equal_chance := true }

/-- Theorem: All three sampling methods have equal chance of selection for each individual -/
theorem equal_chance_in_all_methods :
  simple_random_sampling.equal_chance ∧
  systematic_sampling.equal_chance ∧
  stratified_sampling.equal_chance :=
by sorry

end NUMINAMATH_CALUDE_equal_chance_in_all_methods_l2149_214974


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l2149_214963

theorem right_triangle_arctan_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  a^2 = b^2 + c^2 → Real.arctan (b / (c + a)) + Real.arctan (c / (b + a)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l2149_214963


namespace NUMINAMATH_CALUDE_storage_unit_paint_area_l2149_214915

/-- Represents a rectangular storage unit with windows --/
structure StorageUnit where
  length : ℝ
  width : ℝ
  height : ℝ
  windowCount : ℕ
  windowLength : ℝ
  windowWidth : ℝ

/-- Calculates the total area to be painted in the storage unit --/
def totalPaintArea (unit : StorageUnit) : ℝ :=
  let wallArea := 2 * (unit.length * unit.height + unit.width * unit.height)
  let ceilingArea := unit.length * unit.width
  let windowArea := unit.windowCount * (unit.windowLength * unit.windowWidth)
  wallArea + ceilingArea - windowArea

/-- Theorem stating that the total paint area for the given storage unit is 1020 square yards --/
theorem storage_unit_paint_area :
  let unit : StorageUnit := {
    length := 15,
    width := 12,
    height := 8,
    windowCount := 2,
    windowLength := 3,
    windowWidth := 4
  }
  totalPaintArea unit = 1020 := by sorry

end NUMINAMATH_CALUDE_storage_unit_paint_area_l2149_214915


namespace NUMINAMATH_CALUDE_cost_in_usd_l2149_214908

/-- The cost of coffee and snack in USD given their prices in yen and the exchange rate -/
theorem cost_in_usd (coffee_yen : ℕ) (snack_yen : ℕ) (exchange_rate : ℚ) : 
  coffee_yen = 250 → snack_yen = 150 → exchange_rate = 1 / 100 →
  (coffee_yen + snack_yen : ℚ) * exchange_rate = 4 := by
  sorry

end NUMINAMATH_CALUDE_cost_in_usd_l2149_214908


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l2149_214939

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l2149_214939


namespace NUMINAMATH_CALUDE_monica_books_l2149_214967

/-- The number of books Monica read last year -/
def books_last_year : ℕ := sorry

/-- The number of books Monica read this year -/
def books_this_year : ℕ := 2 * books_last_year

/-- The number of books Monica will read next year -/
def books_next_year : ℕ := 2 * books_this_year + 5

theorem monica_books : books_last_year = 16 ∧ books_next_year = 69 := by
  sorry

end NUMINAMATH_CALUDE_monica_books_l2149_214967


namespace NUMINAMATH_CALUDE_factor_quadratic_l2149_214946

theorem factor_quadratic (t : ℚ) : 
  (∃ k : ℚ, ∀ x : ℚ, 10 * x^2 + 21 * x - 10 = k * (x - t) * (10 * x + 5 * t + 5)) ↔ 
  (t = 2/5 ∨ t = -5/2) := by
sorry

end NUMINAMATH_CALUDE_factor_quadratic_l2149_214946


namespace NUMINAMATH_CALUDE_ellipse_with_foci_on_y_axis_l2149_214943

theorem ellipse_with_foci_on_y_axis (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
  ∀ (x y : ℝ), m * x^2 + n * y^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_with_foci_on_y_axis_l2149_214943


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2149_214926

theorem polynomial_simplification (x : ℝ) :
  (2 * x^10 + 8 * x^9 + 3 * x^8) + (5 * x^12 - x^10 + 2 * x^9 - 5 * x^8 + 4 * x^5 + 6) =
  5 * x^12 + x^10 + 10 * x^9 - 2 * x^8 + 4 * x^5 + 6 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2149_214926


namespace NUMINAMATH_CALUDE_largest_three_digit_sum_l2149_214998

-- Define the sum function
def sum (A B : Nat) : Nat :=
  (100 * A + 10 * A + B) + (10 * B + A) + B

-- Theorem statement
theorem largest_three_digit_sum :
  ∃ (A B : Nat),
    A ≠ B ∧
    A < 10 ∧
    B < 10 ∧
    sum A B ≤ 999 ∧
    ∀ (X Y : Nat),
      X ≠ Y →
      X < 10 →
      Y < 10 →
      sum X Y ≤ 999 →
      sum X Y ≤ sum A B :=
by
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_sum_l2149_214998


namespace NUMINAMATH_CALUDE_unit_circle_sector_arc_length_l2149_214902

theorem unit_circle_sector_arc_length (θ : Real) :
  (1/2 * θ = 1) → (θ = 2) := by
  sorry

end NUMINAMATH_CALUDE_unit_circle_sector_arc_length_l2149_214902


namespace NUMINAMATH_CALUDE_cone_volume_l2149_214973

/-- Given a cone whose lateral surface is an arc of a sector with radius 2 and arc length 2π,
    prove that its volume is (√3 * π) / 3 -/
theorem cone_volume (r : Real) (h : Real) :
  (r = 1) →
  (h^2 + r^2 = 2^2) →
  (1/3 * π * r^2 * h = (Real.sqrt 3 * π) / 3) := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l2149_214973


namespace NUMINAMATH_CALUDE_angle_relation_l2149_214930

theorem angle_relation (A B C : Real) (h1 : 0 < A ∧ A < π/2) (h2 : 0 < B ∧ B < π/2) (h3 : 0 < C ∧ C < π/2) (h4 : A + B + C = π) :
  (Real.cos (A/2))^2 = (Real.cos (B/2))^2 + (Real.cos (C/2))^2 - 2 * Real.cos (B/2) * Real.cos (C/2) * Real.sin (A/2) := by
  sorry

end NUMINAMATH_CALUDE_angle_relation_l2149_214930


namespace NUMINAMATH_CALUDE_john_unique_performance_l2149_214986

/-- Represents the Australian Senior Mathematics Competition (ASMC) -/
structure ASMC where
  total_questions : ℕ
  score_formula : ℕ → ℕ → ℕ
  score_uniqueness : ℕ → Prop

/-- John's performance in the ASMC -/
structure JohnPerformance where
  asmc : ASMC
  correct : ℕ
  wrong : ℕ

/-- Theorem stating that John's performance is unique given his score -/
theorem john_unique_performance (asmc : ASMC) (h : asmc.total_questions = 25) 
    (h_formula : asmc.score_formula = fun c w => 25 + 5 * c - 2 * w)
    (h_uniqueness : asmc.score_uniqueness = fun s => 
      ∀ c₁ w₁ c₂ w₂, s = asmc.score_formula c₁ w₁ → s = asmc.score_formula c₂ w₂ → 
      c₁ + w₁ ≤ asmc.total_questions → c₂ + w₂ ≤ asmc.total_questions → c₁ = c₂ ∧ w₁ = w₂) :
  ∃! (jp : JohnPerformance), 
    jp.asmc = asmc ∧ 
    jp.correct + jp.wrong ≤ asmc.total_questions ∧
    asmc.score_formula jp.correct jp.wrong = 100 ∧
    jp.correct = 19 ∧ jp.wrong = 10 := by
  sorry


end NUMINAMATH_CALUDE_john_unique_performance_l2149_214986


namespace NUMINAMATH_CALUDE_two_white_balls_probability_l2149_214970

/-- The probability of drawing two white balls without replacement from a box containing 
    8 white balls and 10 black balls is 28/153. -/
theorem two_white_balls_probability (white_balls black_balls : ℕ) 
    (h1 : white_balls = 8) (h2 : black_balls = 10) :
  let total_balls := white_balls + black_balls
  let prob_first_white := white_balls / total_balls
  let prob_second_white := (white_balls - 1) / (total_balls - 1)
  prob_first_white * prob_second_white = 28 / 153 := by
  sorry

end NUMINAMATH_CALUDE_two_white_balls_probability_l2149_214970


namespace NUMINAMATH_CALUDE_equation_solutions_l2149_214941

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = 2/3 ∧ ∀ x : ℝ, 3*x*(x-2) = 2*(x-2) ↔ x = x₁ ∨ x = x₂) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 7/2 ∧ y₂ = -2 ∧ ∀ x : ℝ, 2*x^2 - 3*x - 14 = 0 ↔ x = y₁ ∨ x = y₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2149_214941


namespace NUMINAMATH_CALUDE_logical_reasoning_methods_correct_answer_is_C_l2149_214990

-- Define the reasoning methods
inductive ReasoningMethod
| SphereFromCircle
| TriangleAngleSum
| ClassPerformance
| PolygonAngleSum

-- Define a predicate for logical reasoning
def isLogical : ReasoningMethod → Prop
| ReasoningMethod.SphereFromCircle => True
| ReasoningMethod.TriangleAngleSum => True
| ReasoningMethod.ClassPerformance => False
| ReasoningMethod.PolygonAngleSum => True

-- Theorem stating which reasoning methods are logical
theorem logical_reasoning_methods :
  (isLogical ReasoningMethod.SphereFromCircle) ∧
  (isLogical ReasoningMethod.TriangleAngleSum) ∧
  (¬isLogical ReasoningMethod.ClassPerformance) ∧
  (isLogical ReasoningMethod.PolygonAngleSum) :=
by sorry

-- Define the answer options
inductive AnswerOption
| A
| B
| C
| D

-- Define the correct answer
def correctAnswer : AnswerOption := AnswerOption.C

-- Theorem stating that C is the correct answer
theorem correct_answer_is_C :
  correctAnswer = AnswerOption.C :=
by sorry

end NUMINAMATH_CALUDE_logical_reasoning_methods_correct_answer_is_C_l2149_214990


namespace NUMINAMATH_CALUDE_hulk_jump_distance_l2149_214987

def jump_distance (n : ℕ) : ℝ := 3 * (2 ^ (n - 1))

theorem hulk_jump_distance :
  (∀ k < 11, jump_distance k ≤ 3000) ∧ jump_distance 11 > 3000 := by
  sorry

end NUMINAMATH_CALUDE_hulk_jump_distance_l2149_214987


namespace NUMINAMATH_CALUDE_rosy_fish_count_l2149_214955

/-- The number of fish Lilly has -/
def lillys_fish : ℕ := 10

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := 18

/-- The number of fish Rosy has -/
def rosys_fish : ℕ := total_fish - lillys_fish

theorem rosy_fish_count : rosys_fish = 8 := by sorry

end NUMINAMATH_CALUDE_rosy_fish_count_l2149_214955


namespace NUMINAMATH_CALUDE_profit_calculation_l2149_214979

-- Define package prices
def basic_price : ℕ := 5
def deluxe_price : ℕ := 10
def premium_price : ℕ := 15

-- Define weekday car wash numbers
def basic_cars : ℕ := 50
def deluxe_cars : ℕ := 40
def premium_cars : ℕ := 20

-- Define employee wages
def employee_a_wage : ℕ := 110
def employee_b_wage : ℕ := 90
def employee_c_wage : ℕ := 100
def employee_d_wage : ℕ := 80

-- Define operating expenses
def weekday_expenses : ℕ := 200

-- Define the number of weekdays
def weekdays : ℕ := 5

-- Define the function to calculate total profit
def total_profit : ℕ :=
  let daily_revenue := basic_price * basic_cars + deluxe_price * deluxe_cars + premium_price * premium_cars
  let total_revenue := daily_revenue * weekdays
  let employee_expenses := employee_a_wage * 5 + employee_b_wage * 2 + employee_c_wage * 3 + employee_d_wage * 2
  let total_expenses := employee_expenses + weekday_expenses * weekdays
  total_revenue - total_expenses

-- Theorem statement
theorem profit_calculation : total_profit = 2560 := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l2149_214979


namespace NUMINAMATH_CALUDE_total_profit_is_30000_l2149_214971

/-- Represents the profit distribution problem -/
structure ProfitProblem where
  total_subscription : ℕ
  a_more_than_b : ℕ
  b_more_than_c : ℕ
  b_profit : ℕ

/-- Calculate the total profit given the problem parameters -/
def calculate_total_profit (p : ProfitProblem) : ℕ :=
  sorry

/-- Theorem stating that the total profit is 30000 given the specific problem parameters -/
theorem total_profit_is_30000 :
  let p : ProfitProblem := {
    total_subscription := 50000,
    a_more_than_b := 4000,
    b_more_than_c := 5000,
    b_profit := 10200
  }
  calculate_total_profit p = 30000 := by sorry

end NUMINAMATH_CALUDE_total_profit_is_30000_l2149_214971


namespace NUMINAMATH_CALUDE_isabellas_haircuts_l2149_214948

/-- The total length of hair cut off in two haircuts -/
def total_hair_cut (initial_length first_cut_length second_cut_length : ℝ) : ℝ :=
  (initial_length - first_cut_length) + (first_cut_length - second_cut_length)

/-- Theorem: The total length of hair cut off in Isabella's two haircuts is 9 inches -/
theorem isabellas_haircuts :
  total_hair_cut 18 14 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_haircuts_l2149_214948


namespace NUMINAMATH_CALUDE_probability_ratio_equals_ways_ratio_l2149_214914

def number_of_balls : ℕ := 20
def number_of_bins : ℕ := 5

def distribution_p : List ℕ := [3, 6, 4, 4, 4]
def distribution_q : List ℕ := [4, 4, 4, 4, 4]

def ways_to_distribute (dist : List ℕ) : ℕ :=
  sorry

theorem probability_ratio_equals_ways_ratio :
  let p := (ways_to_distribute distribution_p : ℚ) / number_of_balls ^ number_of_bins
  let q := (ways_to_distribute distribution_q : ℚ) / number_of_balls ^ number_of_bins
  p / q = (ways_to_distribute distribution_p : ℚ) / (ways_to_distribute distribution_q) :=
by sorry

end NUMINAMATH_CALUDE_probability_ratio_equals_ways_ratio_l2149_214914


namespace NUMINAMATH_CALUDE_third_circle_radius_l2149_214956

/-- Given two externally tangent circles and a third circle tangent to both and their common external tangent, prove the radius of the third circle is 5. -/
theorem third_circle_radius (P Q R : ℝ × ℝ) (r : ℝ) : 
  let d := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  (d = 8) →  -- Distance between centers P and Q
  (∀ X : ℝ × ℝ, (X.1 - P.1)^2 + (X.2 - P.2)^2 = 3^2 → 
    (X.1 - Q.1)^2 + (X.2 - Q.2)^2 = 8^2) →  -- Circles are externally tangent
  (∀ Y : ℝ × ℝ, (Y.1 - P.1)^2 + (Y.2 - P.2)^2 = (3 + r)^2 ∧ 
    (Y.1 - Q.1)^2 + (Y.2 - Q.2)^2 = (5 - r)^2) →  -- Third circle is tangent to both circles
  (∃ Z : ℝ × ℝ, (Z.1 - R.1)^2 + (Z.2 - R.2)^2 = r^2 ∧ 
    ((Z.1 - P.1) * (Q.2 - P.2) = (Z.2 - P.2) * (Q.1 - P.1))) →  -- Third circle is tangent to common external tangent
  r = 5 := by
sorry

end NUMINAMATH_CALUDE_third_circle_radius_l2149_214956


namespace NUMINAMATH_CALUDE_employee_hire_year_l2149_214920

/-- Rule of 70 provision: An employee can retire when their age plus years of employment is at least 70 -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year an employee was hired -/
def hire_year : ℕ := 1968

/-- The age at which the employee was hired -/
def hire_age : ℕ := 32

/-- The year the employee becomes eligible to retire -/
def retirement_year : ℕ := 2006

theorem employee_hire_year :
  rule_of_70 (hire_age + (retirement_year - hire_year)) hire_age ∧
  ∀ y, y > hire_year → ¬rule_of_70 (hire_age + (y - hire_year)) hire_age :=
by sorry

end NUMINAMATH_CALUDE_employee_hire_year_l2149_214920


namespace NUMINAMATH_CALUDE_root_minus_one_quadratic_equation_l2149_214944

theorem root_minus_one_quadratic_equation (p : ℚ) :
  (∀ x, (2*p - 1) * x^2 + 2*(1 - p) * x + 3*p = 0 ↔ x = -1) ↔ p = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_root_minus_one_quadratic_equation_l2149_214944


namespace NUMINAMATH_CALUDE_unique_solution_l2149_214981

/-- A three-digit number represented as a tuple of its digits -/
def ThreeDigitNumber := (ℕ × ℕ × ℕ)

/-- Convert a ThreeDigitNumber to its integer representation -/
def to_int (n : ThreeDigitNumber) : ℕ :=
  100 * n.1 + 10 * n.2.1 + n.2.2

/-- Check if a ThreeDigitNumber satisfies the condition abc = (a + b + c)^3 -/
def satisfies_condition (n : ThreeDigitNumber) : Prop :=
  to_int n = (n.1 + n.2.1 + n.2.2) ^ 3

/-- The theorem stating that 512 is the only solution -/
theorem unique_solution :
  ∃! (n : ThreeDigitNumber), 
    100 ≤ to_int n ∧ 
    to_int n ≤ 999 ∧ 
    satisfies_condition n ∧
    to_int n = 512 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l2149_214981


namespace NUMINAMATH_CALUDE_mascot_sales_equation_l2149_214983

/-- Represents the sales growth of a mascot over two months -/
def sales_growth (initial_sales : ℝ) (final_sales : ℝ) (growth_rate : ℝ) : Prop :=
  initial_sales * (1 + growth_rate)^2 = final_sales

/-- Theorem stating the correct equation for the given sales scenario -/
theorem mascot_sales_equation :
  ∀ (x : ℝ), x > 0 →
  sales_growth 10 11.5 x :=
by
  sorry

end NUMINAMATH_CALUDE_mascot_sales_equation_l2149_214983


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2149_214978

theorem fraction_evaluation (a b : ℝ) (h : a ≠ b) : (a^4 - b^4) / (a^2 - b^2) = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2149_214978


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l2149_214958

theorem number_exceeding_percentage (x : ℝ) : x = 0.2 * x + 40 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l2149_214958


namespace NUMINAMATH_CALUDE_shelter_cats_count_l2149_214960

theorem shelter_cats_count (total animals : ℕ) (cats dogs : ℕ) : 
  total = 60 →
  cats = dogs + 20 →
  cats + dogs = total →
  cats = 40 :=
by sorry

end NUMINAMATH_CALUDE_shelter_cats_count_l2149_214960


namespace NUMINAMATH_CALUDE_divisibility_relation_l2149_214947

theorem divisibility_relation (a b c : ℕ+) : 
  (a ∣ (b * c - 1)) ∧ (b ∣ (c * a - 1)) ∧ (c ∣ (a * b - 1)) →
  ((a = 2 ∧ b = 3 ∧ c = 5) ∨ 
   (a = 2 ∧ b = 5 ∧ c = 3) ∨ 
   (a = 3 ∧ b = 2 ∧ c = 5) ∨ 
   (a = 3 ∧ b = 5 ∧ c = 2) ∨ 
   (a = 5 ∧ b = 2 ∧ c = 3) ∨ 
   (a = 5 ∧ b = 3 ∧ c = 2) ∨ 
   (a = 1 ∧ b = 1 ∧ c ≥ 1) ∨ 
   (a = 1 ∧ c = 1 ∧ b ≥ 1) ∨ 
   (b = 1 ∧ c = 1 ∧ a ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_relation_l2149_214947


namespace NUMINAMATH_CALUDE_factorization_equality_l2149_214999

theorem factorization_equality (a : ℝ) : 2 * a^2 - 4 * a + 2 = 2 * (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2149_214999


namespace NUMINAMATH_CALUDE_total_birds_in_pet_store_l2149_214919

/-- Represents the number of birds in a cage -/
structure CageBirds where
  parrots : Nat
  finches : Nat
  canaries : Nat
  parakeets : Nat

/-- The pet store's bird inventory -/
def petStore : List CageBirds := [
  { parrots := 9, finches := 4, canaries := 7, parakeets := 0 },
  { parrots := 5, finches := 10, canaries := 0, parakeets := 8 },
  { parrots := 0, finches := 7, canaries := 3, parakeets := 15 },
  { parrots := 10, finches := 12, canaries := 0, parakeets := 5 }
]

/-- Calculates the total number of birds in a cage -/
def totalBirdsInCage (cage : CageBirds) : Nat :=
  cage.parrots + cage.finches + cage.canaries + cage.parakeets

/-- Theorem: The total number of birds in the pet store is 95 -/
theorem total_birds_in_pet_store :
  (petStore.map totalBirdsInCage).sum = 95 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_in_pet_store_l2149_214919


namespace NUMINAMATH_CALUDE_bobs_family_children_l2149_214921

/-- Given the following conditions about Bob's family and apple consumption:
  * Bob picked 450 apples in total
  * There are 40 adults in the family
  * Each adult ate 3 apples
  * Each child ate 10 apples
  This theorem proves that there are 33 children in Bob's family. -/
theorem bobs_family_children (total_apples : ℕ) (num_adults : ℕ) (apples_per_adult : ℕ) (apples_per_child : ℕ) :
  total_apples = 450 →
  num_adults = 40 →
  apples_per_adult = 3 →
  apples_per_child = 10 →
  (total_apples - num_adults * apples_per_adult) / apples_per_child = 33 :=
by sorry

end NUMINAMATH_CALUDE_bobs_family_children_l2149_214921


namespace NUMINAMATH_CALUDE_fraction_to_percentage_l2149_214942

/-- Represents a mixed repeating decimal number -/
structure MixedRepeatingDecimal where
  integerPart : ℕ
  nonRepeatingPart : ℚ
  repeatingPart : ℚ

/-- Converts a rational number to a MixedRepeatingDecimal -/
def toMixedRepeatingDecimal (q : ℚ) : MixedRepeatingDecimal :=
  sorry

/-- Converts a MixedRepeatingDecimal to a percentage string -/
def toPercentageString (m : MixedRepeatingDecimal) : String :=
  sorry

theorem fraction_to_percentage (n d : ℕ) (h : d ≠ 0) :
  toPercentageString (toMixedRepeatingDecimal (n / d)) = "8.(923076)%" :=
sorry

end NUMINAMATH_CALUDE_fraction_to_percentage_l2149_214942


namespace NUMINAMATH_CALUDE_symmetric_line_passes_through_point_l2149_214952

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Function to check if two lines are symmetric about a point
def symmetric_lines (l₁ l₂ : Line) (p : Point) : Prop :=
  ∀ (x y : ℝ), point_on_line ⟨x, y⟩ l₁ ↔ 
    point_on_line ⟨2*p.x - x, 2*p.y - y⟩ l₂

-- Theorem statement
theorem symmetric_line_passes_through_point :
  ∀ (k : ℝ),
  let l₁ : Line := ⟨k, -4*k⟩
  let l₂ : Line := sorry
  let p : Point := ⟨2, 1⟩
  symmetric_lines l₁ l₂ p →
  point_on_line ⟨0, 2⟩ l₂ := by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_passes_through_point_l2149_214952


namespace NUMINAMATH_CALUDE_greatest_common_multiple_15_20_under_125_l2149_214913

theorem greatest_common_multiple_15_20_under_125 : 
  ∃ n : ℕ, n = 120 ∧ 
  (∀ m : ℕ, m < 125 ∧ 15 ∣ m ∧ 20 ∣ m → m ≤ n) ∧
  15 ∣ n ∧ 20 ∣ n ∧ n < 125 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_15_20_under_125_l2149_214913


namespace NUMINAMATH_CALUDE_distribute_eight_to_two_groups_l2149_214993

/-- The number of ways to distribute n distinct objects into 2 non-empty groups -/
def distribute_to_two_groups (n : ℕ) : ℕ :=
  2^n - 2

/-- The theorem stating that distributing 8 distinct objects into 2 non-empty groups results in 254 possibilities -/
theorem distribute_eight_to_two_groups :
  distribute_to_two_groups 8 = 254 := by
  sorry

#eval distribute_to_two_groups 8

end NUMINAMATH_CALUDE_distribute_eight_to_two_groups_l2149_214993


namespace NUMINAMATH_CALUDE_kenneth_initial_money_l2149_214965

/-- The amount of money Kenneth had initially -/
def initial_money : ℕ := 50

/-- The number of baguettes Kenneth bought -/
def num_baguettes : ℕ := 2

/-- The cost of each baguette in dollars -/
def cost_baguette : ℕ := 2

/-- The number of water bottles Kenneth bought -/
def num_water : ℕ := 2

/-- The cost of each water bottle in dollars -/
def cost_water : ℕ := 1

/-- The amount of money Kenneth has left after the purchase -/
def money_left : ℕ := 44

/-- Theorem stating that Kenneth's initial money equals $50 -/
theorem kenneth_initial_money :
  initial_money = 
    num_baguettes * cost_baguette + 
    num_water * cost_water + 
    money_left := by sorry

end NUMINAMATH_CALUDE_kenneth_initial_money_l2149_214965


namespace NUMINAMATH_CALUDE_no_integer_tangent_length_l2149_214961

theorem no_integer_tangent_length (t₁ m : ℕ) : 
  (∃ (m : ℕ), m % 2 = 1 ∧ m < 24 ∧ t₁^2 = m * (24 - m)) → False :=
sorry

end NUMINAMATH_CALUDE_no_integer_tangent_length_l2149_214961


namespace NUMINAMATH_CALUDE_revenue_change_l2149_214989

theorem revenue_change (R : ℝ) (x : ℝ) (h : R > 0) :
  R * (1 + x / 100) * (1 - x / 100) = R * 0.96 →
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_l2149_214989


namespace NUMINAMATH_CALUDE_system_solution_l2149_214904

theorem system_solution : ∃! (x y : ℝ), x - y = 2 ∧ 2*x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2149_214904


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l2149_214992

/-- 
Given an arithmetic sequence where the sum of the third and fifth terms is 10,
prove that the fourth term is 5.
-/
theorem arithmetic_sequence_fourth_term 
  (a : ℕ → ℝ) -- a is the arithmetic sequence
  (h : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) -- definition of arithmetic sequence
  (sum_condition : a 3 + a 5 = 10) -- sum of third and fifth terms is 10
  : a 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l2149_214992


namespace NUMINAMATH_CALUDE_apollo_wheel_replacement_ratio_l2149_214905

/-- Represents the chariot wheel replacement scenario -/
structure WheelReplacement where
  initial_rate : ℕ  -- Initial rate in golden apples
  months : ℕ        -- Total number of months
  half_year : ℕ     -- Number of months before rate change
  total_payment : ℕ -- Total payment for the year

/-- Calculates the ratio of new rate to old rate -/
def rate_ratio (w : WheelReplacement) : ℚ :=
  let first_half_payment := w.initial_rate * w.half_year
  let second_half_payment := w.total_payment - first_half_payment
  (second_half_payment : ℚ) / (w.initial_rate * (w.months - w.half_year))

/-- Theorem stating that the rate ratio is 2 for the given scenario -/
theorem apollo_wheel_replacement_ratio :
  let w : WheelReplacement := ⟨3, 12, 6, 54⟩
  rate_ratio w = 2 := by
  sorry

end NUMINAMATH_CALUDE_apollo_wheel_replacement_ratio_l2149_214905


namespace NUMINAMATH_CALUDE_prime_square_sum_l2149_214933

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(2,2,5), (2,5,2), (3,2,3), (3,3,2)} ∪ {(p,q,r) | p = 2 ∧ q = r ∧ q ≥ 3 ∧ Nat.Prime q}

theorem prime_square_sum (p q r : ℕ) :
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ is_perfect_square (p^q + p^r) ↔ (p,q,r) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_prime_square_sum_l2149_214933


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l2149_214917

def is_first_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi < α ∧ α < (Real.pi / 2) + 2 * k * Real.pi

def is_first_or_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, (2 * k * Real.pi < α ∧ α < (Real.pi / 2) + 2 * k * Real.pi) ∨
           ((Real.pi + 2 * k * Real.pi < α) ∧ (α < (3 * Real.pi / 2) + 2 * k * Real.pi))

theorem half_angle_quadrant (α : Real) :
  is_first_quadrant α → is_first_or_third_quadrant (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l2149_214917


namespace NUMINAMATH_CALUDE_tan_neg_alpha_implies_expression_l2149_214972

theorem tan_neg_alpha_implies_expression (α : Real) 
  (h : Real.tan (-α) = 3) : 
  (Real.sin α)^2 - Real.sin (2 * α) = (-15/8) * Real.cos (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_tan_neg_alpha_implies_expression_l2149_214972


namespace NUMINAMATH_CALUDE_investment_plans_count_l2149_214982

/-- The number of ways to distribute projects across cities -/
def distribute_projects (num_projects : ℕ) (num_cities : ℕ) (max_per_city : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of investment plans -/
theorem investment_plans_count : distribute_projects 3 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_investment_plans_count_l2149_214982


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l2149_214932

/-- Given a train of length 2000 m that crosses a tree in 200 sec,
    the time it takes to pass a platform of length 2500 m is 450 sec. -/
theorem train_platform_crossing_time :
  ∀ (train_length platform_length tree_crossing_time : ℝ),
    train_length = 2000 →
    platform_length = 2500 →
    tree_crossing_time = 200 →
    (train_length + platform_length) / (train_length / tree_crossing_time) = 450 := by
  sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l2149_214932


namespace NUMINAMATH_CALUDE_barbell_to_rack_ratio_is_one_to_ten_l2149_214928

/-- Given a squat rack cost and total cost, calculates the ratio of barbell cost to squat rack cost -/
def barbellToRackRatio (rackCost totalCost : ℚ) : ℚ × ℚ :=
  let barbellCost := totalCost - rackCost
  (barbellCost, rackCost)

/-- Theorem: The ratio of barbell cost to squat rack cost is 1:10 for given costs -/
theorem barbell_to_rack_ratio_is_one_to_ten :
  barbellToRackRatio 2500 2750 = (1, 10) := by
  sorry

#eval barbellToRackRatio 2500 2750

end NUMINAMATH_CALUDE_barbell_to_rack_ratio_is_one_to_ten_l2149_214928


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2149_214949

theorem max_value_quadratic (x y : ℝ) : 
  4 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 10 ≤ -13 :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2149_214949


namespace NUMINAMATH_CALUDE_wall_length_is_850_l2149_214950

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (w : WallDimensions) : ℝ :=
  w.length * w.width * w.height

/-- The main theorem stating that under given conditions, the wall length is 850 cm -/
theorem wall_length_is_850 (brick : BrickDimensions)
    (wall : WallDimensions) (num_bricks : ℕ) :
    brick.length = 25 →
    brick.width = 11.25 →
    brick.height = 6 →
    wall.width = 600 →
    wall.height = 22.5 →
    num_bricks = 6800 →
    brickVolume brick * num_bricks = wallVolume wall →
    wall.length = 850 := by
  sorry


end NUMINAMATH_CALUDE_wall_length_is_850_l2149_214950


namespace NUMINAMATH_CALUDE_mixed_doubles_pairing_methods_l2149_214903

-- Define the number of male and female players
def num_male_players : ℕ := 5
def num_female_players : ℕ := 4

-- Define the number of players to be selected for each gender
def male_players_to_select : ℕ := 2
def female_players_to_select : ℕ := 2

-- Define the total number of pairing methods
def total_pairing_methods : ℕ := 120

-- Theorem statement
theorem mixed_doubles_pairing_methods :
  (Nat.choose num_male_players male_players_to_select) *
  (Nat.choose num_female_players female_players_to_select) * 2 =
  total_pairing_methods := by
  sorry


end NUMINAMATH_CALUDE_mixed_doubles_pairing_methods_l2149_214903


namespace NUMINAMATH_CALUDE_correct_num_technicians_l2149_214964

/-- Represents the workshop scenario with workers and salaries -/
structure Workshop where
  total_workers : ℕ
  avg_salary : ℚ
  technician_salary : ℚ
  other_salary : ℚ

/-- The number of technicians in the workshop -/
def num_technicians (w : Workshop) : ℕ :=
  7  -- We'll prove this is correct

/-- The given workshop scenario -/
def given_workshop : Workshop :=
  { total_workers := 56
    avg_salary := 6750
    technician_salary := 12000
    other_salary := 6000 }

/-- Theorem stating that the number of technicians in the given workshop is correct -/
theorem correct_num_technicians :
    let n := num_technicians given_workshop
    let m := given_workshop.total_workers - n
    n + m = given_workshop.total_workers ∧
    (n * given_workshop.technician_salary + m * given_workshop.other_salary) / given_workshop.total_workers = given_workshop.avg_salary :=
  sorry


end NUMINAMATH_CALUDE_correct_num_technicians_l2149_214964


namespace NUMINAMATH_CALUDE_problem_solution_l2149_214962

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

theorem problem_solution :
  ∀ a b c : ℝ,
  (∀ x : ℝ, f a b c (-x) = -(f a b c x)) →
  f a b c 1 = 3 →
  f a b c 2 = 12 →
  (a = 1 ∧ b = 0 ∧ c = 2) ∧
  (∀ x y : ℝ, x < y → f a b c x < f a b c y) ∧
  (∀ m n : ℝ, m^3 - 3*m^2 + 5*m = 5 → n^3 - 3*n^2 + 5*n = 1 → m + n = 2) ∧
  (∀ k : ℝ, (∀ x : ℝ, 0 < x ∧ x < 1 → f a b c (x^2 - 4) + f a b c (k*x + 2*k) < 0) → k ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2149_214962
