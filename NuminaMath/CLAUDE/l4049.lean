import Mathlib

namespace NUMINAMATH_CALUDE_hex_B1E_equals_2846_l4049_404957

/-- Converts a hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | 'B' => 11
  | '1' => 1
  | 'E' => 14
  | _ => 0

/-- Converts a hexadecimal string to its decimal value -/
def hex_string_to_dec (s : String) : ℕ :=
  s.foldr (fun c acc => 16 * acc + hex_to_dec c) 0

/-- The hexadecimal number B1E is equal to 2846 in decimal -/
theorem hex_B1E_equals_2846 : hex_string_to_dec "B1E" = 2846 := by
  sorry

end NUMINAMATH_CALUDE_hex_B1E_equals_2846_l4049_404957


namespace NUMINAMATH_CALUDE_max_value_theorem_l4049_404933

theorem max_value_theorem (x y : ℝ) (h : x * y > 0) :
  ∃ (max : ℝ), max = 4 - 2 * Real.sqrt 2 ∧
  ∀ (z : ℝ), z = x / (x + y) + 2 * y / (x + 2 * y) → z ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l4049_404933


namespace NUMINAMATH_CALUDE_polygon_angle_sum_l4049_404988

theorem polygon_angle_sum (n : ℕ) : n ≥ 3 →
  (n - 2) * 180 + (180 - ((n - 2) * 180) / n) = 1350 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_angle_sum_l4049_404988


namespace NUMINAMATH_CALUDE_initial_student_count_l4049_404962

/-- Proves that the initial number of students is 29 given the conditions of the problem -/
theorem initial_student_count (initial_avg : ℝ) (new_avg : ℝ) (new_student_weight : ℝ) :
  initial_avg = 28 →
  new_avg = 27.5 →
  new_student_weight = 13 →
  ∃ n : ℕ, n = 29 ∧
    (n : ℝ) * initial_avg + new_student_weight = (n + 1 : ℝ) * new_avg :=
by
  sorry


end NUMINAMATH_CALUDE_initial_student_count_l4049_404962


namespace NUMINAMATH_CALUDE_partnership_profit_difference_l4049_404926

/-- Given a partnership scenario with specific investments and profit-sharing rules, 
    calculate the difference in profit shares between two partners. -/
theorem partnership_profit_difference 
  (john_investment mike_investment : ℚ)
  (total_profit : ℚ)
  (effort_share investment_share : ℚ)
  (h1 : john_investment = 700)
  (h2 : mike_investment = 300)
  (h3 : total_profit = 3000.0000000000005)
  (h4 : effort_share = 1/3)
  (h5 : investment_share = 2/3)
  (h6 : effort_share + investment_share = 1) :
  let total_investment := john_investment + mike_investment
  let john_investment_ratio := john_investment / total_investment
  let mike_investment_ratio := mike_investment / total_investment
  let john_share := (effort_share * total_profit / 2) + 
                    (investment_share * total_profit * john_investment_ratio)
  let mike_share := (effort_share * total_profit / 2) + 
                    (investment_share * total_profit * mike_investment_ratio)
  john_share - mike_share = 800.0000000000001 := by
  sorry


end NUMINAMATH_CALUDE_partnership_profit_difference_l4049_404926


namespace NUMINAMATH_CALUDE_smallest_natural_with_remainders_l4049_404996

theorem smallest_natural_with_remainders : ∃ N : ℕ,
  (N % 9 = 8) ∧
  (N % 8 = 7) ∧
  (N % 7 = 6) ∧
  (N % 6 = 5) ∧
  (N % 5 = 4) ∧
  (N % 4 = 3) ∧
  (N % 3 = 2) ∧
  (N % 2 = 1) ∧
  (∀ M : ℕ, M < N →
    ¬((M % 9 = 8) ∧
      (M % 8 = 7) ∧
      (M % 7 = 6) ∧
      (M % 6 = 5) ∧
      (M % 5 = 4) ∧
      (M % 4 = 3) ∧
      (M % 3 = 2) ∧
      (M % 2 = 1))) ∧
  N = 2519 :=
by sorry

end NUMINAMATH_CALUDE_smallest_natural_with_remainders_l4049_404996


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l4049_404943

theorem arithmetic_calculations : 
  (23 - 17 - (-6) + (-16) = -4) ∧ 
  (0 - 32 / ((-2)^3 - (-4)) = 8) := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l4049_404943


namespace NUMINAMATH_CALUDE_parabola_equation_from_hyperbola_focus_l4049_404913

/-- Given a hyperbola with equation x²/4 - y²/5 = 1, prove that the standard equation
    of a parabola with its focus at the left focus of the hyperbola is y² = -12x. -/
theorem parabola_equation_from_hyperbola_focus (x y : ℝ) :
  (x^2 / 4 - y^2 / 5 = 1) →
  ∃ (x₀ y₀ : ℝ), (x₀ = -3 ∧ y₀ = 0) ∧
    (∀ (x' y' : ℝ), y'^2 = -12 * x' ↔ 
      ((x' - x₀)^2 + (y' - y₀)^2 = (x' - (x₀ + 3/4))^2 + y'^2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_from_hyperbola_focus_l4049_404913


namespace NUMINAMATH_CALUDE_scale_drawing_conversion_l4049_404989

/-- Converts a length in inches to miles, given a scale where 1 inch represents 1000 feet --/
def inches_to_miles (inches : ℚ) : ℚ :=
  inches * 1000 / 5280

/-- Theorem stating that 7.5 inches on the given scale represents 125/88 miles --/
theorem scale_drawing_conversion :
  inches_to_miles (7.5) = 125 / 88 := by
  sorry

end NUMINAMATH_CALUDE_scale_drawing_conversion_l4049_404989


namespace NUMINAMATH_CALUDE_min_m_value_l4049_404934

open Real

-- Define the inequality function
def f (x m : ℝ) : Prop := x + m * log x + exp (-x) ≥ x^m

-- State the theorem
theorem min_m_value :
  (∀ x > 1, ∀ m : ℝ, f x m) →
  (∃ m₀ : ℝ, m₀ = -exp 1 ∧ (∀ m : ℝ, (∀ x > 1, f x m) → m ≥ m₀)) :=
sorry

end NUMINAMATH_CALUDE_min_m_value_l4049_404934


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l4049_404940

theorem quadratic_inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x^2 - 8 * x - 4 - a > 0) ↔ 
  (-16 < a ∧ a < -8) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l4049_404940


namespace NUMINAMATH_CALUDE_sector_area_from_arc_length_l4049_404919

/-- Given a circle where the arc length corresponding to a central angle of 2 radians is 4 cm,
    the area of the sector enclosed by this central angle is 4 cm². -/
theorem sector_area_from_arc_length (r : ℝ) (h : 2 * r = 4) :
  (1 / 2) * 4 * r = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_from_arc_length_l4049_404919


namespace NUMINAMATH_CALUDE_justin_pencils_l4049_404976

theorem justin_pencils (total_pencils sabrina_pencils : ℕ) : 
  total_pencils = 50 →
  sabrina_pencils = 14 →
  total_pencils - sabrina_pencils > 2 * sabrina_pencils →
  (total_pencils - sabrina_pencils) - 2 * sabrina_pencils = 8 := by
  sorry

end NUMINAMATH_CALUDE_justin_pencils_l4049_404976


namespace NUMINAMATH_CALUDE_alcohol_dilution_l4049_404909

/-- Given an initial solution and added water, calculate the new alcohol percentage -/
theorem alcohol_dilution (initial_volume : ℝ) (initial_percentage : ℝ) (added_water : ℝ) :
  initial_volume = 15 →
  initial_percentage = 26 →
  added_water = 5 →
  let initial_alcohol := initial_volume * (initial_percentage / 100)
  let total_volume := initial_volume + added_water
  let new_percentage := (initial_alcohol / total_volume) * 100
  new_percentage = 19.5 := by
  sorry

#check alcohol_dilution

end NUMINAMATH_CALUDE_alcohol_dilution_l4049_404909


namespace NUMINAMATH_CALUDE_youngest_sibling_age_problem_l4049_404985

/-- The age of the youngest sibling given the conditions of the problem -/
def youngest_sibling_age (n : ℕ) (age_differences : List ℕ) (average_age : ℚ) : ℚ :=
  (n * average_age - (age_differences.sum)) / n

/-- Theorem stating the age of the youngest sibling under the given conditions -/
theorem youngest_sibling_age_problem : 
  let n : ℕ := 4
  let age_differences : List ℕ := [2, 7, 11]
  let average_age : ℚ := 25
  youngest_sibling_age n age_differences average_age = 20 := by
sorry

#eval youngest_sibling_age 4 [2, 7, 11] 25

end NUMINAMATH_CALUDE_youngest_sibling_age_problem_l4049_404985


namespace NUMINAMATH_CALUDE_stamp_collection_problem_l4049_404948

theorem stamp_collection_problem (tom_initial : ℕ) (tom_final : ℕ) (harry_extra : ℕ) :
  tom_initial = 3000 →
  tom_final = 3061 →
  harry_extra = 10 →
  ∃ (mike : ℕ),
    mike = 17 ∧
    tom_final = tom_initial + mike + (2 * mike + harry_extra) :=
by sorry

end NUMINAMATH_CALUDE_stamp_collection_problem_l4049_404948


namespace NUMINAMATH_CALUDE_max_days_same_shift_l4049_404975

/-- The number of nurses in the ward -/
def num_nurses : ℕ := 15

/-- The number of shifts per day -/
def shifts_per_day : ℕ := 3

/-- Calculates the number of possible nurse pair combinations -/
def nurse_pair_combinations (n : ℕ) : ℕ := n.choose 2

/-- Theorem: Maximum days for two specific nurses to work the same shift again -/
theorem max_days_same_shift : 
  nurse_pair_combinations num_nurses / shifts_per_day = 35 := by
  sorry

end NUMINAMATH_CALUDE_max_days_same_shift_l4049_404975


namespace NUMINAMATH_CALUDE_triple_tangent_identity_l4049_404968

theorem triple_tangent_identity (x y z : ℝ) 
  (hx : |x| ≠ 1 / Real.sqrt 3) 
  (hy : |y| ≠ 1 / Real.sqrt 3) 
  (hz : |z| ≠ 1 / Real.sqrt 3) 
  (h_sum : x + y + z = x * y * z) : 
  (3 * x - x^3) / (1 - 3 * x^2) + (3 * y - y^3) / (1 - 3 * y^2) + (3 * z - z^3) / (1 - 3 * z^2) = 
  (3 * x - x^3) / (1 - 3 * x^2) * (3 * y - y^3) / (1 - 3 * y^2) * (3 * z - z^3) / (1 - 3 * z^2) := by
  sorry

end NUMINAMATH_CALUDE_triple_tangent_identity_l4049_404968


namespace NUMINAMATH_CALUDE_tan_half_difference_l4049_404936

theorem tan_half_difference (a b : Real) 
  (h1 : Real.cos a + Real.cos b = 3/5) 
  (h2 : Real.sin a + Real.sin b = 2/5) : 
  Real.tan ((a - b)/2) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_difference_l4049_404936


namespace NUMINAMATH_CALUDE_min_voters_for_tall_to_win_l4049_404911

/-- Represents the voting structure and rules of the giraffe beauty contest --/
structure GiraffeContest where
  total_voters : Nat
  num_districts : Nat
  sections_per_district : Nat
  voters_per_section : Nat
  (total_voters_eq : total_voters = num_districts * sections_per_district * voters_per_section)
  (num_districts_pos : num_districts > 0)
  (sections_per_district_pos : sections_per_district > 0)
  (voters_per_section_pos : voters_per_section > 0)

/-- Calculates the minimum number of voters required to win the contest --/
def minVotersToWin (contest : GiraffeContest) : Nat :=
  let districts_to_win := contest.num_districts / 2 + 1
  let sections_to_win := contest.sections_per_district / 2 + 1
  let voters_to_win_section := contest.voters_per_section / 2 + 1
  districts_to_win * sections_to_win * voters_to_win_section

/-- The main theorem stating the minimum number of voters required for Tall to win --/
theorem min_voters_for_tall_to_win (contest : GiraffeContest)
  (h_total : contest.total_voters = 105)
  (h_districts : contest.num_districts = 5)
  (h_sections : contest.sections_per_district = 7)
  (h_voters : contest.voters_per_section = 3) :
  minVotersToWin contest = 24 := by
  sorry


end NUMINAMATH_CALUDE_min_voters_for_tall_to_win_l4049_404911


namespace NUMINAMATH_CALUDE_circle_radius_range_l4049_404925

/-- Given points P and C in a 2D Cartesian coordinate system, 
    if there exist two distinct points A and B on the circle centered at C with radius r, 
    such that PA - 2AB = 0, then r is in the range [1, 5). -/
theorem circle_radius_range (P C A B : ℝ × ℝ) (r : ℝ) : 
  P = (2, 2) →
  C = (5, 6) →
  A ≠ B →
  (∃ (A B : ℝ × ℝ), 
    (A.1 - C.1)^2 + (A.2 - C.2)^2 = r^2 ∧ 
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = r^2 ∧
    (A.1 - P.1, A.2 - P.2) = 2 • (B.1 - A.1, B.2 - A.2)) →
  r ∈ Set.Icc 1 5 ∧ r ≠ 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_range_l4049_404925


namespace NUMINAMATH_CALUDE_smallest_angle_for_tan_equation_l4049_404903

theorem smallest_angle_for_tan_equation :
  ∃ x : ℝ, x > 0 ∧ x < 2 * Real.pi ∧
  Real.tan (6 * x) = (Real.sin x - Real.cos x) / (Real.sin x + Real.cos x) ∧
  x = 45 * Real.pi / (7 * 180) ∧
  ∀ y : ℝ, y > 0 → y < 2 * Real.pi →
    Real.tan (6 * y) = (Real.sin y - Real.cos y) / (Real.sin y + Real.cos y) →
    x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_for_tan_equation_l4049_404903


namespace NUMINAMATH_CALUDE_profit_share_ratio_l4049_404982

/-- The ratio of profit shares for two investors is proportional to their investments -/
theorem profit_share_ratio (p_investment q_investment : ℕ) 
  (hp : p_investment = 52000)
  (hq : q_investment = 65000) :
  ∃ (k : ℕ), k ≠ 0 ∧ 
    p_investment * 5 = q_investment * 4 * k ∧ 
    q_investment * 4 = p_investment * 4 * k :=
sorry

end NUMINAMATH_CALUDE_profit_share_ratio_l4049_404982


namespace NUMINAMATH_CALUDE_workers_gone_home_is_120_l4049_404946

/-- Represents the problem of workers leaving a factory for Chinese New Year --/
structure WorkerProblem where
  total_days : Nat
  weekend_days : Nat
  remaining_workers : Nat
  total_worker_days : Nat

/-- The specific instance of the worker problem --/
def factory_problem : WorkerProblem := {
  total_days := 15
  weekend_days := 4
  remaining_workers := 121
  total_worker_days := 2011
}

/-- Calculates the number of workers who have gone home --/
def workers_gone_home (p : WorkerProblem) : Nat :=
  sorry

/-- Theorem stating that 120 workers have gone home --/
theorem workers_gone_home_is_120 : 
  workers_gone_home factory_problem = 120 := by
  sorry

end NUMINAMATH_CALUDE_workers_gone_home_is_120_l4049_404946


namespace NUMINAMATH_CALUDE_multivariable_jensen_inequality_l4049_404924

/-- A function F: ℝⁿ → ℝ is convex if for any two points x and y in ℝⁿ and weights q₁, q₂ ≥ 0 with q₁ + q₂ = 1,
    F(q₁x + q₂y) ≤ q₁F(x) + q₂F(y) -/
def IsConvex (n : ℕ) (F : (Fin n → ℝ) → ℝ) : Prop :=
  ∀ (x y : Fin n → ℝ) (q₁ q₂ : ℝ), q₁ ≥ 0 → q₂ ≥ 0 → q₁ + q₂ = 1 →
    F (fun i => q₁ * x i + q₂ * y i) ≤ q₁ * F x + q₂ * F y

/-- Jensen's inequality for multivariable convex functions -/
theorem multivariable_jensen_inequality {n : ℕ} (F : (Fin n → ℝ) → ℝ) (h_convex : IsConvex n F)
    (x y : Fin n → ℝ) (q₁ q₂ : ℝ) (hq₁ : q₁ ≥ 0) (hq₂ : q₂ ≥ 0) (hsum : q₁ + q₂ = 1) :
    F (fun i => q₁ * x i + q₂ * y i) ≤ q₁ * F x + q₂ * F y := by
  sorry

end NUMINAMATH_CALUDE_multivariable_jensen_inequality_l4049_404924


namespace NUMINAMATH_CALUDE_george_red_marbles_l4049_404966

/-- The number of red marbles in George's collection --/
def red_marbles (total yellow white green : ℕ) : ℕ :=
  total - (yellow + white + green)

/-- Theorem stating the number of red marbles in George's collection --/
theorem george_red_marbles :
  ∀ (total yellow white green : ℕ),
    total = 50 →
    yellow = 12 →
    white = total / 2 →
    green = yellow / 2 →
    red_marbles total yellow white green = 7 := by
  sorry

end NUMINAMATH_CALUDE_george_red_marbles_l4049_404966


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_system_l4049_404912

theorem unique_solution_quadratic_system :
  ∃! y : ℚ, (9 * y^2 + 8 * y - 3 = 0) ∧ (27 * y^2 + 35 * y - 12 = 0) ∧ (y = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_system_l4049_404912


namespace NUMINAMATH_CALUDE_log_inequality_l4049_404977

theorem log_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  Real.log ((a + b) / 2) + Real.log ((b + c) / 2) + Real.log ((c + a) / 2) >
  Real.log a + Real.log b + Real.log c :=
sorry

end NUMINAMATH_CALUDE_log_inequality_l4049_404977


namespace NUMINAMATH_CALUDE_power_difference_over_sum_l4049_404906

theorem power_difference_over_sum : (3^2016 - 3^2014) / (3^2016 + 3^2014) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_over_sum_l4049_404906


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l4049_404951

theorem vector_sum_magnitude (a b : ℝ × ℝ) : 
  ‖a - 2 • b‖ = 1 → a • b = 1 → ‖a + 2 • b‖ = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l4049_404951


namespace NUMINAMATH_CALUDE_min_value_function_l4049_404983

theorem min_value_function (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.sqrt (x^2 - x*y + y^2) + Real.sqrt (x^2 - 9*x + 27) + Real.sqrt (y^2 - 15*y + 75) ≥ 7 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_function_l4049_404983


namespace NUMINAMATH_CALUDE_both_games_count_l4049_404967

/-- The number of people who play both kabadi and kho kho -/
def both_games : ℕ := sorry

/-- The total number of players -/
def total_players : ℕ := 45

/-- The number of people who play kabadi (including those who play both) -/
def kabadi_players : ℕ := 10

/-- The number of people who play only kho kho -/
def only_kho_kho : ℕ := 35

theorem both_games_count : both_games = 10 := by sorry

end NUMINAMATH_CALUDE_both_games_count_l4049_404967


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l4049_404997

/-- Represents an ellipse with equation x²/(2m) + y²/m = 1, where m > 0 -/
structure Ellipse (m : ℝ) where
  equation : ∀ (x y : ℝ), x^2 / (2*m) + y^2 / m = 1
  m_pos : m > 0

/-- Represents a point on the ellipse -/
structure EllipsePoint (m : ℝ) where
  x : ℝ
  y : ℝ
  on_ellipse : x^2 / (2*m) + y^2 / m = 1

/-- The theorem stating that if an ellipse with equation x²/(2m) + y²/m = 1 (m > 0)
    is intersected by the line x = √m at two points with distance 2 between them,
    then the length of the major axis of the ellipse is 4 -/
theorem ellipse_major_axis_length 
  (m : ℝ) 
  (e : Ellipse m) 
  (A B : EllipsePoint m) 
  (h1 : A.x = Real.sqrt m) 
  (h2 : B.x = Real.sqrt m) 
  (h3 : (A.y - B.y)^2 = 4) : 
  ∃ (a : ℝ), a = 2 ∧ 2*a = 4 := by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l4049_404997


namespace NUMINAMATH_CALUDE_highest_score_can_be_less_than_16_l4049_404970

/-- Represents a team in the tournament -/
structure Team :=
  (id : Nat)
  (score : Nat)

/-- Represents the tournament -/
structure Tournament :=
  (teams : Finset Team)
  (num_teams : Nat)
  (games_played : Nat)
  (total_points : Nat)

/-- The tournament satisfies the given conditions -/
def valid_tournament (t : Tournament) : Prop :=
  t.num_teams = 16 ∧
  t.games_played = (t.num_teams * (t.num_teams - 1)) / 2 ∧
  t.total_points = 2 * t.games_played

/-- The highest score in the tournament -/
def highest_score (t : Tournament) : Nat :=
  Finset.sup t.teams (fun team => team.score)

/-- Theorem stating that it's possible for the highest score to be less than 16 -/
theorem highest_score_can_be_less_than_16 (t : Tournament) :
  valid_tournament t → ∃ (score : Nat), highest_score t < 16 :=
by
  sorry

end NUMINAMATH_CALUDE_highest_score_can_be_less_than_16_l4049_404970


namespace NUMINAMATH_CALUDE_water_conservation_l4049_404931

/-- Represents the amount of water in tons, where negative values indicate waste and positive values indicate savings. -/
def WaterAmount : Type := ℤ

/-- Records the water amount given the number of tons wasted or saved. -/
def recordWaterAmount (tons : ℤ) : WaterAmount := tons

theorem water_conservation (waste : WaterAmount) (save : ℤ) :
  waste = recordWaterAmount (-10) →
  recordWaterAmount save = recordWaterAmount 30 :=
by sorry

end NUMINAMATH_CALUDE_water_conservation_l4049_404931


namespace NUMINAMATH_CALUDE_extremum_value_monotonicity_intervals_a_range_l4049_404981

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x + a^2 / x
def g (x : ℝ) : ℝ := x + Real.log x

-- Define h as the sum of f and g
def h (a : ℝ) (x : ℝ) : ℝ := f a x + g x

-- Theorem 1: When x=1 is an extremum of h(x), a = √3
theorem extremum_value (a : ℝ) (ha : a ≠ 0) :
  (∀ x, x > 0 → (deriv (h a)) x = 0 → x = 1) → a = Real.sqrt 3 := by sorry

-- Theorem 2: Monotonicity intervals of h(x)
theorem monotonicity_intervals (a : ℝ) (ha : a = Real.sqrt 3) :
  (∀ x, 0 < x → x < 1 → (deriv (h a)) x < 0) ∧
  (∀ x, x > 1 → (deriv (h a)) x > 0) := by sorry

-- Theorem 3: Range of a when f(x₁) ≥ g(x₂) for any x₁, x₂ ∈ [1,2] and -2 < a < 0
theorem a_range :
  (∀ a, -2 < a → a < 0 → ∀ x₁ x₂, 1 ≤ x₁ → x₁ ≤ 2 → 1 ≤ x₂ → x₂ ≤ 2 → f a x₁ ≥ g x₂) →
  ∃ a, -2 < a ∧ a < -1 - 1/2 * Real.log 2 := by sorry

end

end NUMINAMATH_CALUDE_extremum_value_monotonicity_intervals_a_range_l4049_404981


namespace NUMINAMATH_CALUDE_mobile_profit_percentage_l4049_404944

-- Define the given values
def grinder_cost : ℝ := 15000
def mobile_cost : ℝ := 8000
def grinder_loss_percentage : ℝ := 0.02
def overall_profit : ℝ := 500

-- Define the theorem
theorem mobile_profit_percentage :
  let grinder_selling_price := grinder_cost * (1 - grinder_loss_percentage)
  let total_cost := grinder_cost + mobile_cost
  let total_selling_price := total_cost + overall_profit
  let mobile_selling_price := total_selling_price - grinder_selling_price
  let mobile_profit := mobile_selling_price - mobile_cost
  (mobile_profit / mobile_cost) * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_mobile_profit_percentage_l4049_404944


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l4049_404923

theorem average_of_three_numbers (N : ℝ) : 
  9 ≤ N ∧ N ≤ 17 →
  ∃ k : ℕ, (6 + 10 + N) / 3 = 2 * k →
  (6 + 10 + N) / 3 = 10 := by
sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l4049_404923


namespace NUMINAMATH_CALUDE_history_homework_time_l4049_404947

/-- Represents the time in minutes for each homework subject and the total available time. -/
structure HomeworkTime where
  total : Nat
  math : Nat
  english : Nat
  science : Nat
  special_project : Nat

/-- Calculates the time remaining for history homework given the times for other subjects. -/
def history_time (hw : HomeworkTime) : Nat :=
  hw.total - (hw.math + hw.english + hw.science + hw.special_project)

/-- Proves that given the specified homework times, the remaining time for history is 25 minutes. -/
theorem history_homework_time :
  let hw : HomeworkTime := {
    total := 180,  -- 3 hours in minutes
    math := 45,
    english := 30,
    science := 50,
    special_project := 30
  }
  history_time hw = 25 := by sorry

end NUMINAMATH_CALUDE_history_homework_time_l4049_404947


namespace NUMINAMATH_CALUDE_divisible_by_five_l4049_404969

theorem divisible_by_five (n : ℕ) : 
  (∃ B : ℕ, B < 10 ∧ n = 5270 + B) → (n % 5 = 0 ↔ B = 0 ∨ B = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_l4049_404969


namespace NUMINAMATH_CALUDE_area_of_bounded_region_l4049_404937

/-- The equation of the boundary curve -/
def boundary_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4 * |x - y| + 4 * |x + y|

/-- The region bounded by the curve -/
def bounded_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | boundary_equation p.1 p.2}

/-- The area of the bounded region -/
noncomputable def area : ℝ := sorry

theorem area_of_bounded_region :
  area = 64 + 32 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_area_of_bounded_region_l4049_404937


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_l4049_404964

-- Define the total number of students
def total_students : ℕ := 5

-- Define the number of male students
def male_students : ℕ := 3

-- Define the number of female students
def female_students : ℕ := 2

-- Define the number of students to be selected
def selected_students : ℕ := 2

-- Define the event "at least one male student is selected"
def at_least_one_male (selected : Finset (Fin total_students)) : Prop :=
  ∃ s ∈ selected, s.val < male_students

-- Define the event "all female students are selected"
def all_females (selected : Finset (Fin total_students)) : Prop :=
  ∀ s ∈ selected, s.val ≥ male_students

-- Theorem statement
theorem events_mutually_exclusive :
  ∀ selected : Finset (Fin total_students),
  selected.card = selected_students →
  ¬(at_least_one_male selected ∧ all_females selected) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_l4049_404964


namespace NUMINAMATH_CALUDE_binary_multiplication_subtraction_l4049_404938

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binary_to_nat (bits : List Bool) : Nat :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a natural number to a binary representation as a list of bits. -/
def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

theorem binary_multiplication_subtraction :
  let a := binary_to_nat [true, false, true, true]  -- 1101₂
  let b := binary_to_nat [true, true, true]         -- 111₂
  let c := binary_to_nat [true, false, true]        -- 101₂
  nat_to_binary ((a * b) - c) = [false, false, false, true, false, false, true] -- 1001000₂
:= by sorry

end NUMINAMATH_CALUDE_binary_multiplication_subtraction_l4049_404938


namespace NUMINAMATH_CALUDE_polynomial_expansion_l4049_404960

theorem polynomial_expansion (x : ℝ) :
  (7 * x - 3) * (2 * x^3 + 5 * x^2 - 4) = 14 * x^4 + 29 * x^3 - 15 * x^2 - 28 * x + 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l4049_404960


namespace NUMINAMATH_CALUDE_egg_marble_distribution_unique_l4049_404991

/-- Represents the distribution of eggs and marbles among three groups. -/
structure EggMarbleDistribution where
  eggs_a : ℕ
  eggs_b : ℕ
  eggs_c : ℕ
  marbles_a : ℕ
  marbles_b : ℕ
  marbles_c : ℕ

/-- Checks if the given distribution satisfies all conditions. -/
def is_valid_distribution (d : EggMarbleDistribution) : Prop :=
  d.eggs_a + d.eggs_b + d.eggs_c = 15 ∧
  d.marbles_a + d.marbles_b + d.marbles_c = 4 ∧
  d.eggs_a ≠ d.eggs_b ∧ d.eggs_b ≠ d.eggs_c ∧ d.eggs_a ≠ d.eggs_c ∧
  d.eggs_b = d.marbles_b - d.marbles_a ∧
  d.eggs_c = d.marbles_c - d.marbles_b

theorem egg_marble_distribution_unique :
  ∃! d : EggMarbleDistribution, is_valid_distribution d ∧
    d.eggs_a = 12 ∧ d.eggs_b = 1 ∧ d.eggs_c = 2 :=
sorry

end NUMINAMATH_CALUDE_egg_marble_distribution_unique_l4049_404991


namespace NUMINAMATH_CALUDE_all_triangles_present_l4049_404953

/-- Represents a permissible triangle with angles (i/p)180°, (j/p)180°, (k/p)180° --/
structure PermissibleTriangle (p : ℕ) where
  i : ℕ
  j : ℕ
  k : ℕ
  sum_eq_p : i + j + k = p

/-- The set of all permissible triangles for a given prime p --/
def AllPermissibleTriangles (p : ℕ) : Set (PermissibleTriangle p) :=
  {t : PermissibleTriangle p | True}

/-- The set of triangles obtained after the division process stops --/
def FinalTriangleSet (p : ℕ) : Set (PermissibleTriangle p) :=
  sorry

/-- Theorem stating that the final set of triangles includes all permissible triangles --/
theorem all_triangles_present (p : ℕ) (h : Prime p) :
  FinalTriangleSet p = AllPermissibleTriangles p :=
sorry

end NUMINAMATH_CALUDE_all_triangles_present_l4049_404953


namespace NUMINAMATH_CALUDE_grid_sequence_problem_l4049_404973

theorem grid_sequence_problem (row : List ℤ) (d_col : ℤ) (last_col : ℤ) (M : ℤ) :
  row = [15, 11, 7] →
  d_col = -5 →
  last_col = -4 →
  M = last_col - 4 * d_col →
  M = 6 := by
  sorry

end NUMINAMATH_CALUDE_grid_sequence_problem_l4049_404973


namespace NUMINAMATH_CALUDE_min_moves_to_no_moves_l4049_404941

/-- Represents a chessboard configuration -/
structure ChessBoard (n : ℕ) where
  pieces : Fin n → Fin n → Bool

/-- A move on the chessboard -/
inductive Move (n : ℕ)
  | jump : Fin n → Fin n → Fin n → Fin n → Move n

/-- Predicate to check if a move is valid -/
def is_valid_move (n : ℕ) (board : ChessBoard n) (move : Move n) : Prop :=
  match move with
  | Move.jump from_x from_y to_x to_y =>
    -- Implement the logic for a valid move
    sorry

/-- Predicate to check if no further moves are possible -/
def no_moves_possible (n : ℕ) (board : ChessBoard n) : Prop :=
  ∀ (move : Move n), ¬(is_valid_move n board move)

/-- The main theorem -/
theorem min_moves_to_no_moves (n : ℕ) :
  ∀ (move_sequence : List (Move n)),
    (∃ (final_board : ChessBoard n),
      no_moves_possible n final_board ∧
      -- final_board is the result of applying move_sequence to the initial board
      sorry) →
    move_sequence.length ≥ ⌈(n^2 : ℚ) / 3⌉ :=
  sorry

end NUMINAMATH_CALUDE_min_moves_to_no_moves_l4049_404941


namespace NUMINAMATH_CALUDE_remaining_black_cards_l4049_404918

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total_cards : Nat)
  (black_cards : Nat)
  (black_face_cards : Nat)
  (black_number_cards : Nat)

/-- The number of decks in the mix -/
def num_decks : Nat := 5

/-- A standard deck of 52 cards -/
def standard_deck : Deck :=
  { total_cards := 52,
    black_cards := 26,
    black_face_cards := 6,
    black_number_cards := 20 }

/-- The number of black face cards removed -/
def removed_black_face_cards : Nat := 7

/-- The number of black number cards removed -/
def removed_black_number_cards : Nat := 12

/-- Theorem: The number of remaining black cards in the mix of 5 decks
    after removing 7 black face cards and 12 black number cards is 111 -/
theorem remaining_black_cards :
  (num_decks * standard_deck.black_cards) - (removed_black_face_cards + removed_black_number_cards) = 111 :=
by
  sorry


end NUMINAMATH_CALUDE_remaining_black_cards_l4049_404918


namespace NUMINAMATH_CALUDE_equation_holds_iff_sum_ten_l4049_404935

theorem equation_holds_iff_sum_ten (a b c : ℕ) 
  (ha : 0 < a ∧ a < 10) 
  (hb : 0 < b ∧ b < 10) 
  (hc : 0 < c ∧ c < 10) : 
  (10 * a + b) * (10 * a + c) = 100 * a * (a + 1) + b * c ↔ b + c = 10 := by
sorry

end NUMINAMATH_CALUDE_equation_holds_iff_sum_ten_l4049_404935


namespace NUMINAMATH_CALUDE_michelle_savings_l4049_404974

/-- Represents the number of $100 bills Michelle has after exchanging her savings -/
def number_of_bills : ℕ := 8

/-- Represents the value of each bill in dollars -/
def bill_value : ℕ := 100

/-- Theorem stating that Michelle's total savings equal $800 -/
theorem michelle_savings : number_of_bills * bill_value = 800 := by
  sorry

end NUMINAMATH_CALUDE_michelle_savings_l4049_404974


namespace NUMINAMATH_CALUDE_total_candy_collected_l4049_404901

/-- The number of candy pieces collected by Travis and his brother -/
def total_candy : ℕ := 68

/-- The number of people who collected candy -/
def num_people : ℕ := 2

/-- The number of candy pieces each person ate -/
def candy_eaten_per_person : ℕ := 4

/-- The number of candy pieces left after eating -/
def candy_left : ℕ := 60

/-- Theorem stating that the total candy collected equals 68 -/
theorem total_candy_collected :
  total_candy = candy_left + (num_people * candy_eaten_per_person) :=
by sorry

end NUMINAMATH_CALUDE_total_candy_collected_l4049_404901


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l4049_404930

/-- Given that (-4, y₁) and (2, y₂) both lie on the line y = -2x + 3, prove that y₁ > y₂ -/
theorem y1_greater_than_y2 (y₁ y₂ : ℝ) : 
  y₁ = -2 * (-4) + 3 → y₂ = -2 * 2 + 3 → y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l4049_404930


namespace NUMINAMATH_CALUDE_division_problem_l4049_404949

/-- Given a division with quotient 20, divisor 66, and remainder 55, the dividend is 1375. -/
theorem division_problem :
  ∀ (dividend quotient divisor remainder : ℕ),
    quotient = 20 →
    divisor = 66 →
    remainder = 55 →
    dividend = divisor * quotient + remainder →
    dividend = 1375 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4049_404949


namespace NUMINAMATH_CALUDE_distance_product_sum_bound_l4049_404907

/-- Given an equilateral triangle with side length 1 and a point P inside it,
    let a, b, c be the distances from P to the three sides of the triangle. -/
def DistancesFromPoint (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = Real.sqrt 3 / 2

/-- The sum of products of distances from a point inside an equilateral triangle
    to its sides is bounded. -/
theorem distance_product_sum_bound {a b c : ℝ} (h : DistancesFromPoint a b c) :
  0 < a * b + b * c + c * a ∧ a * b + b * c + c * a ≤ 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_distance_product_sum_bound_l4049_404907


namespace NUMINAMATH_CALUDE_number_of_schools_l4049_404958

theorem number_of_schools (n : ℕ) : n = 22 :=
  -- Define the total number of students
  let total_students := 4 * n
  -- Define Alex's rank
  let alex_rank := 2 * n
  -- Define the ranks of Alex's teammates
  let jordan_rank := 45
  let kim_rank := 73
  let lee_rank := 98
  -- State the conditions
  have h1 : alex_rank < jordan_rank := by sorry
  have h2 : alex_rank < kim_rank := by sorry
  have h3 : alex_rank < lee_rank := by sorry
  have h4 : total_students = 2 * alex_rank - 1 := by sorry
  have h5 : alex_rank ≤ 49 := by sorry
  -- Prove that n = 22
  sorry

#check number_of_schools

end NUMINAMATH_CALUDE_number_of_schools_l4049_404958


namespace NUMINAMATH_CALUDE_no_unique_solution_l4049_404990

/-- 
Given a system of two linear equations:
  3(3x + 4y) = 36
  kx + cy = 30
where k = 9, prove that the system does not have a unique solution when c = 12.
-/
theorem no_unique_solution (x y : ℝ) : 
  (3 * (3 * x + 4 * y) = 36) →
  (9 * x + 12 * y = 30) →
  ¬ (∃! (x y : ℝ), 3 * (3 * x + 4 * y) = 36 ∧ 9 * x + 12 * y = 30) :=
by sorry

end NUMINAMATH_CALUDE_no_unique_solution_l4049_404990


namespace NUMINAMATH_CALUDE_initial_column_size_l4049_404914

theorem initial_column_size (total_people : ℕ) (initial_columns : ℕ) (people_per_column : ℕ) : 
  total_people = initial_columns * people_per_column →
  total_people = 40 * 12 →
  initial_columns = 16 →
  people_per_column = 30 := by
sorry

end NUMINAMATH_CALUDE_initial_column_size_l4049_404914


namespace NUMINAMATH_CALUDE_like_terms_ratio_l4049_404904

theorem like_terms_ratio (m n : ℕ) : 
  (∃ (x y : ℝ), 2 * x^(m-2) * y^3 = -1/2 * x^2 * y^(2*n-1)) → 
  m / n = 2 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_ratio_l4049_404904


namespace NUMINAMATH_CALUDE_well_depth_l4049_404994

/-- Proves that a circular well with diameter 4 meters and volume 301.59289474462014 cubic meters has a depth of 24 meters. -/
theorem well_depth (diameter : Real) (volume : Real) (depth : Real) :
  diameter = 4 →
  volume = 301.59289474462014 →
  depth = volume / (π * (diameter / 2)^2) →
  depth = 24 := by
  sorry

end NUMINAMATH_CALUDE_well_depth_l4049_404994


namespace NUMINAMATH_CALUDE_sum_of_products_zero_l4049_404955

theorem sum_of_products_zero 
  (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 108)
  (eq2 : y^2 + y*z + z^2 = 9)
  (eq3 : z^2 + x*z + x^2 = 117) :
  x*y + y*z + x*z = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_zero_l4049_404955


namespace NUMINAMATH_CALUDE_three_digit_power_ending_l4049_404952

theorem three_digit_power_ending (N : ℕ) : 
  (100 ≤ N ∧ N < 1000) → 
  (∀ k : ℕ, k > 0 → N^k ≡ N [ZMOD 1000]) → 
  (N = 625 ∨ N = 376) :=
sorry

end NUMINAMATH_CALUDE_three_digit_power_ending_l4049_404952


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l4049_404984

theorem diophantine_equation_solution :
  ∃ (x y : ℕ), 1984 * x - 1983 * y = 1985 ∧ x = 27764 ∧ y = 27777 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l4049_404984


namespace NUMINAMATH_CALUDE_total_applications_eq_600_l4049_404905

def in_state_applications : ℕ := 200

def out_state_applications : ℕ := 2 * in_state_applications

def total_applications : ℕ := in_state_applications + out_state_applications

theorem total_applications_eq_600 : total_applications = 600 := by
  sorry

end NUMINAMATH_CALUDE_total_applications_eq_600_l4049_404905


namespace NUMINAMATH_CALUDE_fifteen_percent_of_600_is_90_l4049_404986

theorem fifteen_percent_of_600_is_90 : (15 / 100) * 600 = 90 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_of_600_is_90_l4049_404986


namespace NUMINAMATH_CALUDE_max_value_quadratic_l4049_404965

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 9) : 
  x^2 + 2*x*y + 3*y^2 ≤ (117 + 36*Real.sqrt 3) / 11 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l4049_404965


namespace NUMINAMATH_CALUDE_count_closest_to_two_sevenths_l4049_404995

def is_closest_to_two_sevenths (r : ℚ) : Prop :=
  ∀ n d : ℕ, n ≤ 2 → d > 0 → |r - 2/7| ≤ |r - (n : ℚ)/d|

def is_four_place_decimal (r : ℚ) : Prop :=
  ∃ a b c d : ℕ, a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    r = (a * 1000 + b * 100 + c * 10 + d) / 10000

theorem count_closest_to_two_sevenths :
  ∃! (s : Finset ℚ), 
    (∀ r ∈ s, is_four_place_decimal r ∧ is_closest_to_two_sevenths r) ∧
    s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_count_closest_to_two_sevenths_l4049_404995


namespace NUMINAMATH_CALUDE_apple_cost_price_l4049_404917

theorem apple_cost_price (selling_price : ℚ) (loss_fraction : ℚ) : 
  selling_price = 15 →
  loss_fraction = 1/6 →
  ∃ cost_price : ℚ, 
    selling_price = cost_price - loss_fraction * cost_price ∧
    cost_price = 18 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_price_l4049_404917


namespace NUMINAMATH_CALUDE_school_trip_theorem_l4049_404998

/-- The number of school buses -/
def num_buses : ℕ := 95

/-- The number of seats on each bus -/
def seats_per_bus : ℕ := 118

/-- All buses are fully filled -/
axiom buses_full : True

/-- The total number of students in the school -/
def total_students : ℕ := num_buses * seats_per_bus

theorem school_trip_theorem : total_students = 11210 := by
  sorry

end NUMINAMATH_CALUDE_school_trip_theorem_l4049_404998


namespace NUMINAMATH_CALUDE_intersection_sum_l4049_404956

theorem intersection_sum (a b : ℚ) : 
  (3 = (1/3) * 4 + a) → 
  (4 = (1/3) * 3 + b) → 
  a + b = 14/3 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l4049_404956


namespace NUMINAMATH_CALUDE_tan_alpha_value_l4049_404929

theorem tan_alpha_value (α : ℝ) (h : (Real.sin α - 2 * Real.cos α) / (Real.sin α + Real.cos α) = -1) :
  Real.tan α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l4049_404929


namespace NUMINAMATH_CALUDE_F_zeros_and_reciprocal_sum_l4049_404902

noncomputable def F (x : ℝ) : ℝ := 1 / (2 * x) + Real.log (x / 2)

theorem F_zeros_and_reciprocal_sum :
  (∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ F x₁ = 0 ∧ F x₂ = 0 ∧
    (∀ (x : ℝ), x > 0 ∧ F x = 0 → x = x₁ ∨ x = x₂)) ∧
  (∀ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ F x₁ = 0 ∧ F x₂ = 0 →
    1 / x₁ + 1 / x₂ > 4) :=
by sorry

end NUMINAMATH_CALUDE_F_zeros_and_reciprocal_sum_l4049_404902


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_l4049_404928

theorem arithmetic_geometric_progression (a b : ℝ) : 
  (1 = (a + b) / 2) →  -- arithmetic progression condition
  (1 = |a * b|) →      -- geometric progression condition
  ((a = 1 ∧ b = 1) ∨ 
   (a = 1 + Real.sqrt 2 ∧ b = 1 - Real.sqrt 2) ∨ 
   (a = 1 - Real.sqrt 2 ∧ b = 1 + Real.sqrt 2)) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_l4049_404928


namespace NUMINAMATH_CALUDE_choose_three_from_eight_l4049_404999

theorem choose_three_from_eight : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_eight_l4049_404999


namespace NUMINAMATH_CALUDE_prob_shortest_diagonal_21_sided_l4049_404959

/-- The number of sides in the regular polygon -/
def n : ℕ := 21

/-- The number of shortest diagonals in a regular n-sided polygon -/
def num_shortest_diagonals (n : ℕ) : ℕ := n / 2

/-- The total number of diagonals in a regular n-sided polygon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The probability of randomly selecting one of the shortest diagonals
    from all the diagonals of a regular n-sided polygon -/
def prob_shortest_diagonal (n : ℕ) : ℚ :=
  (num_shortest_diagonals n : ℚ) / (total_diagonals n : ℚ)

theorem prob_shortest_diagonal_21_sided :
  prob_shortest_diagonal n = 10 / 189 := by
  sorry

end NUMINAMATH_CALUDE_prob_shortest_diagonal_21_sided_l4049_404959


namespace NUMINAMATH_CALUDE_existence_of_mn_l4049_404916

theorem existence_of_mn : ∃ (m n : ℕ), ∀ (a b : ℝ), 
  ((-2 * a^n * b^n)^m + (3 * a^m * b^m)^n) = a^6 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_mn_l4049_404916


namespace NUMINAMATH_CALUDE_equation_solution_l4049_404942

theorem equation_solution (a x : ℚ) : 
  (2 * (x - 2 * (x - a / 4)) = 3 * x) ∧ 
  ((x + a) / 9 - (1 - 3 * x) / 12 = 1) →
  a = 65 / 11 ∧ x = 13 / 11 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l4049_404942


namespace NUMINAMATH_CALUDE_circumscribed_square_area_l4049_404910

/-- Given a circle with an inscribed square of perimeter p, 
    the area of the square that circumscribes the circle is p²/8 -/
theorem circumscribed_square_area (p : ℝ) (p_pos : p > 0) : 
  let inscribed_square_perimeter := p
  let circumscribed_square_area := p^2 / 8
  inscribed_square_perimeter = p → circumscribed_square_area = p^2 / 8 := by
sorry

end NUMINAMATH_CALUDE_circumscribed_square_area_l4049_404910


namespace NUMINAMATH_CALUDE_simplify_polynomial_l4049_404922

theorem simplify_polynomial (y : ℝ) : 
  3 * y^3 - 7 * y^2 + 12 * y + 5 - (2 * y^3 - 4 + 3 * y^2 - 9 * y) = y^3 - 10 * y^2 + 21 * y + 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l4049_404922


namespace NUMINAMATH_CALUDE_slope_constraint_implies_a_value_l4049_404920

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x - x^4

-- Define the theorem
theorem slope_constraint_implies_a_value :
  ∀ a : ℝ,
  (∀ x y : ℝ, 
    1/2 ≤ x ∧ x < y ∧ y ≤ 1 →
    1/2 ≤ (f a y - f a x) / (y - x) ∧ (f a y - f a x) / (y - x) ≤ 4) →
  a = 9/2 :=
sorry

end NUMINAMATH_CALUDE_slope_constraint_implies_a_value_l4049_404920


namespace NUMINAMATH_CALUDE_triangle_side_range_l4049_404987

theorem triangle_side_range (a b : ℝ) (A B C : ℝ) :
  b = 2 →
  B = π / 4 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  (∃ (A' : ℝ), A' ≠ A ∧ 0 < A' ∧ A' < π ∧ a / Real.sin A' = b / Real.sin B) →
  2 < a ∧ a < 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l4049_404987


namespace NUMINAMATH_CALUDE_cars_produced_in_europe_l4049_404992

theorem cars_produced_in_europe (total_cars : ℕ) (north_america_cars : ℕ) (europe_cars : ℕ) :
  total_cars = 6755 →
  north_america_cars = 3884 →
  total_cars = north_america_cars + europe_cars →
  europe_cars = 2871 :=
by sorry

end NUMINAMATH_CALUDE_cars_produced_in_europe_l4049_404992


namespace NUMINAMATH_CALUDE_exists_n_power_half_eq_twenty_l4049_404950

theorem exists_n_power_half_eq_twenty :
  ∃ n : ℝ, n > 0 ∧ n^(n/2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_power_half_eq_twenty_l4049_404950


namespace NUMINAMATH_CALUDE_remainder_theorem_l4049_404993

def polynomial (x : ℝ) : ℝ := 5*x^6 - 3*x^5 + 6*x^4 - x^3 + x^2 - 15

def divisor (x : ℝ) : ℝ := 3*x - 9

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ), 
    polynomial x = (divisor x) * q x + polynomial 3 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4049_404993


namespace NUMINAMATH_CALUDE_product_xyz_l4049_404932

theorem product_xyz (x y z : ℚ) 
  (eq1 : 3 * x + 4 * y = 60)
  (eq2 : 6 * x - 4 * y = 12)
  (eq3 : 2 * x - 3 * z = 38) :
  x * y * z = -1584 / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_l4049_404932


namespace NUMINAMATH_CALUDE_inequality_solution_l4049_404961

theorem inequality_solution :
  {n : ℕ+ | 25 - 5 * n.val < 15} = {n : ℕ+ | n.val > 2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4049_404961


namespace NUMINAMATH_CALUDE_nathan_ate_twenty_packages_l4049_404972

/-- The number of gumballs in each package -/
def gumballs_per_package : ℕ := 5

/-- The total number of gumballs Nathan ate -/
def total_gumballs_eaten : ℕ := 100

/-- The number of packages Nathan ate -/
def packages_eaten : ℕ := total_gumballs_eaten / gumballs_per_package

theorem nathan_ate_twenty_packages : packages_eaten = 20 := by
  sorry

end NUMINAMATH_CALUDE_nathan_ate_twenty_packages_l4049_404972


namespace NUMINAMATH_CALUDE_sum_first_five_multiples_of_twelve_l4049_404945

theorem sum_first_five_multiples_of_twelve : 
  (Finset.range 5).sum (fun i => 12 * (i + 1)) = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_five_multiples_of_twelve_l4049_404945


namespace NUMINAMATH_CALUDE_reverse_divisibility_implies_divides_99_l4049_404939

-- Define a function to reverse the digits of a natural number
def reverse_digits (n : ℕ) : ℕ := sorry

-- Define the property of k
def has_reverse_divisibility_property (k : ℕ) : Prop :=
  ∀ n : ℕ, k ∣ n → k ∣ reverse_digits n

-- Theorem statement
theorem reverse_divisibility_implies_divides_99 (k : ℕ) :
  has_reverse_divisibility_property k → k ∣ 99 := by sorry

end NUMINAMATH_CALUDE_reverse_divisibility_implies_divides_99_l4049_404939


namespace NUMINAMATH_CALUDE_monotonicity_intervals_when_a_2_range_of_a_with_extreme_point_l4049_404927

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x^2 + 3*x + 1

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 6*a*x + 3

-- Part I: Monotonicity intervals when a = 2
theorem monotonicity_intervals_when_a_2 :
  let a := 2
  ∀ x : ℝ, 
    (x ≤ 2 - Real.sqrt 3 ∨ x ≥ 2 + Real.sqrt 3 → f' a x > 0) ∧
    (2 - Real.sqrt 3 < x ∧ x < 2 + Real.sqrt 3 → f' a x < 0) :=
sorry

-- Part II: Range of a when f(x) has at least one extreme value point in (2,3)
theorem range_of_a_with_extreme_point :
  ∀ a : ℝ,
    (∃ x : ℝ, 2 < x ∧ x < 3 ∧ f' a x = 0) →
    (5/4 < a ∧ a < 5/3) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_intervals_when_a_2_range_of_a_with_extreme_point_l4049_404927


namespace NUMINAMATH_CALUDE_cyclists_distance_l4049_404980

theorem cyclists_distance (a b : ℝ) : 
  (a = b^2) ∧ (a - 1 = 3 * (b - 1)) → (a - b = 0 ∨ a - b = 2) :=
by sorry

end NUMINAMATH_CALUDE_cyclists_distance_l4049_404980


namespace NUMINAMATH_CALUDE_min_blue_chips_correct_l4049_404971

/-- Represents the number of chips of each color in the box -/
structure ChipCounts where
  white : ℕ
  blue : ℕ
  red : ℕ

/-- Checks if the chip counts satisfy the given conditions -/
def satisfiesConditions (counts : ChipCounts) : Prop :=
  counts.blue ≥ counts.white / 3 ∧
  counts.blue ≤ counts.red / 4 ∧
  counts.white + counts.blue ≥ 75

/-- The minimum number of blue chips that satisfies the conditions -/
def minBlueChips : ℕ := 19

theorem min_blue_chips_correct :
  (∀ counts : ChipCounts, satisfiesConditions counts → counts.blue ≥ minBlueChips) ∧
  (∃ counts : ChipCounts, satisfiesConditions counts ∧ counts.blue = minBlueChips) := by
  sorry

end NUMINAMATH_CALUDE_min_blue_chips_correct_l4049_404971


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l4049_404978

theorem simplify_sqrt_expression : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l4049_404978


namespace NUMINAMATH_CALUDE_circle_radius_secant_l4049_404900

theorem circle_radius_secant (center P Q R : ℝ × ℝ) : 
  let distance := λ (a b : ℝ × ℝ) => Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
  let radius := distance center Q
  distance center P = 15 ∧ 
  distance P Q = 10 ∧ 
  distance Q R = 8 ∧
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ R = (1 - t) • P + t • Q) →
  radius = 3 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_secant_l4049_404900


namespace NUMINAMATH_CALUDE_fraction_subtraction_l4049_404908

theorem fraction_subtraction : (8 : ℚ) / 19 - (5 : ℚ) / 57 = (1 : ℚ) / 3 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l4049_404908


namespace NUMINAMATH_CALUDE_rabbits_after_four_springs_l4049_404963

/-- Calculates the total number of rabbits after four breeding seasons --/
def totalRabbitsAfterFourSprings (initialBreedingRabbits : ℕ) 
  (spring1KittensPerRabbit spring1AdoptionRate : ℚ) (spring1Returns : ℕ)
  (spring2Kittens : ℕ) (spring2AdoptionRate : ℚ) (spring2Returns : ℕ)
  (spring3BreedingRabbits : ℕ) (spring3KittensPerRabbit : ℕ) (spring3AdoptionRate : ℚ) (spring3Returns : ℕ)
  (spring4BreedingRabbits : ℕ) (spring4KittensPerRabbit : ℕ) (spring4AdoptionRate : ℚ) (spring4Returns : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the total number of rabbits after four springs is 242 --/
theorem rabbits_after_four_springs : 
  totalRabbitsAfterFourSprings 10 10 (1/2) 5 60 (2/5) 10 12 8 (3/10) 3 12 6 (1/5) 2 = 242 :=
by sorry

end NUMINAMATH_CALUDE_rabbits_after_four_springs_l4049_404963


namespace NUMINAMATH_CALUDE_expression_simplification_l4049_404915

theorem expression_simplification (a : ℝ) (h : a = 2 * Real.sqrt 3 + 3) :
  (1 - 1 / (a - 2)) / ((a^2 - 6*a + 9) / (2*a - 4)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4049_404915


namespace NUMINAMATH_CALUDE_vasya_gift_choices_l4049_404921

theorem vasya_gift_choices (n_cars : ℕ) (n_sets : ℕ) : 
  n_cars = 7 → n_sets = 5 → (n_cars.choose 2) + (n_sets.choose 2) + n_cars * n_sets = 66 :=
by
  sorry

end NUMINAMATH_CALUDE_vasya_gift_choices_l4049_404921


namespace NUMINAMATH_CALUDE_raw_materials_cost_l4049_404954

def total_amount : ℝ := 93750
def machinery_cost : ℝ := 40000
def cash_percentage : ℝ := 0.20

theorem raw_materials_cost (raw_materials : ℝ) : raw_materials = 35000 :=
  by
    have cash : ℝ := total_amount * cash_percentage
    have total_equation : raw_materials + machinery_cost + cash = total_amount := by sorry
    sorry

end NUMINAMATH_CALUDE_raw_materials_cost_l4049_404954


namespace NUMINAMATH_CALUDE_quadratic_properties_l4049_404979

/-- A quadratic function of the form y = -x^2 + bx + c -/
def quadratic_function (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

theorem quadratic_properties :
  ∀ (b c : ℝ),
  (b = 4 ∧ c = 3 →
    (∃ (x y : ℝ), (x = 2 ∧ y = 7) ∧ 
      ∀ (t : ℝ), -1 ≤ t ∧ t ≤ 3 → 
        -2 ≤ quadratic_function b c t ∧ quadratic_function b c t ≤ 7)) ∧
  ((∀ (x : ℝ), x ≤ 0 → quadratic_function b c x ≤ 2) ∧
   (∀ (x : ℝ), x > 0 → quadratic_function b c x ≤ 3) ∧
   (∃ (x₁ x₂ : ℝ), x₁ ≤ 0 ∧ x₂ > 0 ∧ 
     quadratic_function b c x₁ = 2 ∧ quadratic_function b c x₂ = 3) →
    b = 2 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l4049_404979
