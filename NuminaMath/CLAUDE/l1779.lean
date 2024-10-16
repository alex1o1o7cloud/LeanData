import Mathlib

namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1779_177934

def M : Set ℝ := {x | Real.sqrt x < 2}
def N : Set ℝ := {x | 3 * x ≥ 1}

theorem intersection_of_M_and_N :
  M ∩ N = {x | 1/3 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1779_177934


namespace NUMINAMATH_CALUDE_product_evaluation_l1779_177940

theorem product_evaluation : 
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * 
  (2^32 + 3^32) * (2^64 + 3^64) * (2 + 1) = 3^129 - 3 * 2^128 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l1779_177940


namespace NUMINAMATH_CALUDE_largest_divisor_n4_minus_n_l1779_177981

/-- A positive integer greater than 1 is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ k, 1 < k ∧ k < n ∧ n % k = 0

theorem largest_divisor_n4_minus_n (n : ℕ) (h : IsComposite n) :
  (∀ d : ℕ, d > 6 → ¬(d ∣ (n^4 - n))) ∧
  (6 ∣ (n^4 - n)) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_n4_minus_n_l1779_177981


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1779_177983

/-- Given a geometric sequence {a_n} where a_4 = 4, prove that a_2 * a_6 = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence property
  a 4 = 4 →                                            -- given condition
  a 2 * a 6 = 16 :=                                    -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1779_177983


namespace NUMINAMATH_CALUDE_canvas_area_l1779_177993

/-- The area of a rectangular canvas inside a decorative border -/
theorem canvas_area (outer_width outer_height border_width : ℝ) : 
  outer_width = 100 →
  outer_height = 140 →
  border_width = 15 →
  (outer_width - 2 * border_width) * (outer_height - 2 * border_width) = 7700 := by
sorry

end NUMINAMATH_CALUDE_canvas_area_l1779_177993


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l1779_177980

def team_size : ℕ := 12
def quadruplets_size : ℕ := 4
def starters_size : ℕ := 5
def max_quadruplets_in_lineup : ℕ := 2

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem basketball_lineup_combinations :
  (choose (team_size - quadruplets_size) starters_size) +
  (choose quadruplets_size 1 * choose (team_size - quadruplets_size) (starters_size - 1)) +
  (choose quadruplets_size 2 * choose (team_size - quadruplets_size) (starters_size - 2)) = 672 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l1779_177980


namespace NUMINAMATH_CALUDE_rain_probability_l1779_177994

theorem rain_probability (p : ℝ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 4) :
  1 - (1 - p)^n = 255/256 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l1779_177994


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1779_177924

/-- A line y = 3x + c is tangent to the parabola y^2 = 12x if and only if c = 3 -/
theorem line_tangent_to_parabola (c : ℝ) : 
  (∃ x y : ℝ, y = 3*x + c ∧ y^2 = 12*x ∧ 
   ∀ x' y' : ℝ, y' = 3*x' + c → y'^2 = 12*x' → (x', y') = (x, y)) ↔ 
  c = 3 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1779_177924


namespace NUMINAMATH_CALUDE_video_game_marathon_points_l1779_177906

theorem video_game_marathon_points : 
  ∀ (jack_points alex_bella_points : ℕ),
    jack_points = 8972 →
    alex_bella_points = 21955 →
    jack_points + alex_bella_points = 30927 := by
  sorry

end NUMINAMATH_CALUDE_video_game_marathon_points_l1779_177906


namespace NUMINAMATH_CALUDE_expression_value_l1779_177907

theorem expression_value (a b c : ℤ) (ha : a = 8) (hb : b = 10) (hc : c = 3) :
  (2 * a - (b - 2 * c)) - ((2 * a - b) - 2 * c) + 3 * (a - c) = 27 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1779_177907


namespace NUMINAMATH_CALUDE_min_value_cubic_fraction_l1779_177912

theorem min_value_cubic_fraction (x : ℝ) (h : x > 9) :
  x^3 / (x - 9) ≥ 325 ∧ ∃ y > 9, y^3 / (y - 9) = 325 := by
  sorry

end NUMINAMATH_CALUDE_min_value_cubic_fraction_l1779_177912


namespace NUMINAMATH_CALUDE_vidyas_age_l1779_177960

theorem vidyas_age (vidya_age : ℕ) (mother_age : ℕ) : 
  mother_age = 3 * vidya_age + 5 →
  mother_age = 44 →
  vidya_age = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_vidyas_age_l1779_177960


namespace NUMINAMATH_CALUDE_expression_evaluation_l1779_177954

theorem expression_evaluation : 7 ^ 8 - 6 / 2 + 9 ^ 3 + 3 + 12 = 5765542 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1779_177954


namespace NUMINAMATH_CALUDE_number_equality_l1779_177943

theorem number_equality (x : ℝ) (h1 : x > 0) (h2 : (2/3) * x = (49/216) * (1/x)) : x = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l1779_177943


namespace NUMINAMATH_CALUDE_week_cycling_distance_l1779_177989

/-- Represents the cycling data for a single day -/
structure DailyRide where
  base_distance : Float
  speed_bonus : Float

/-- Calculates the effective distance for a single day -/
def effective_distance (ride : DailyRide) : Float :=
  ride.base_distance * (1 + ride.speed_bonus)

/-- Calculates the total effective distance for the week -/
def total_effective_distance (rides : List DailyRide) : Float :=
  rides.map effective_distance |> List.sum

/-- The main theorem: proves that the total effective distance is 367.05 km -/
theorem week_cycling_distance : 
  let monday : DailyRide := { base_distance := 40, speed_bonus := 0.05 }
  let tuesday : DailyRide := { base_distance := 50, speed_bonus := 0.03 }
  let wednesday : DailyRide := { base_distance := 25, speed_bonus := 0.07 }
  let thursday : DailyRide := { base_distance := 65, speed_bonus := 0.04 }
  let friday : DailyRide := { base_distance := 78, speed_bonus := 0.06 }
  let saturday : DailyRide := { base_distance := 58.5, speed_bonus := 0.02 }
  let sunday : DailyRide := { base_distance := 33.5, speed_bonus := 0.10 }
  let week_rides : List DailyRide := [monday, tuesday, wednesday, thursday, friday, saturday, sunday]
  total_effective_distance week_rides = 367.05 := by
  sorry


end NUMINAMATH_CALUDE_week_cycling_distance_l1779_177989


namespace NUMINAMATH_CALUDE_tan_315_degrees_l1779_177990

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l1779_177990


namespace NUMINAMATH_CALUDE_ratio_sum_theorem_l1779_177964

theorem ratio_sum_theorem (a b : ℕ) (h1 : a * 4 = b * 3) (h2 : a = 180) : a + b = 420 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_theorem_l1779_177964


namespace NUMINAMATH_CALUDE_choose_15_3_l1779_177913

theorem choose_15_3 : Nat.choose 15 3 = 455 := by sorry

end NUMINAMATH_CALUDE_choose_15_3_l1779_177913


namespace NUMINAMATH_CALUDE_binomial_inequality_l1779_177985

theorem binomial_inequality (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : k < n) :
  (1 : ℝ) / (n + 1 : ℝ) * (n^n : ℝ) / ((k^k : ℝ) * ((n - k)^(n - k) : ℝ)) <
  (n.factorial : ℝ) / ((k.factorial : ℝ) * ((n - k).factorial : ℝ)) ∧
  (n.factorial : ℝ) / ((k.factorial : ℝ) * ((n - k).factorial : ℝ)) <
  (n^n : ℝ) / ((k^k : ℝ) * ((n - k)^(n - k) : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_binomial_inequality_l1779_177985


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1779_177930

/-- Given a geometric sequence {a_n} with a_1 = 8 and a_4 = 64, the common ratio q is 2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 8 →                     -- first term condition
  a 4 = 64 →                    -- fourth term condition
  q = 2 :=                      -- conclusion: common ratio is 2
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1779_177930


namespace NUMINAMATH_CALUDE_set_partition_l1779_177908

def S : Set ℝ := {-5/6, 0, -3.5, 1.2, 6}

def N : Set ℝ := {x ∈ S | x < 0}

def NN : Set ℝ := {x ∈ S | x ≥ 0}

theorem set_partition :
  N = {-5/6, -3.5} ∧ NN = {0, 1.2, 6} := by sorry

end NUMINAMATH_CALUDE_set_partition_l1779_177908


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l1779_177949

-- Define the hyperbola equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / 9 - y^2 / m = 1

-- Define the focus point
def focus : ℝ × ℝ := (5, 0)

-- Theorem statement
theorem hyperbola_m_value :
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), hyperbola_equation x y m) ∧ 
    (focus.1 = 5 ∧ focus.2 = 0) →
    m = 16 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l1779_177949


namespace NUMINAMATH_CALUDE_five_digit_number_formation_l1779_177917

theorem five_digit_number_formation (m n : ℕ) : 
  (100 ≤ m) ∧ (m < 1000) ∧ (10 ≤ n) ∧ (n < 100) → 
  (m * 100 + n = 100 * m + n) := by
  sorry

end NUMINAMATH_CALUDE_five_digit_number_formation_l1779_177917


namespace NUMINAMATH_CALUDE_solution_set_part_i_range_of_m_part_ii_l1779_177971

-- Define the function f
def f (x m : ℝ) : ℝ := |2*x| + |2*x + 3| + m

-- Part I
theorem solution_set_part_i : 
  {x : ℝ | f x (-2) ≤ 3} = {x : ℝ | -2 ≤ x ∧ x ≤ 1/2} := by sorry

-- Part II
theorem range_of_m_part_ii :
  ∀ m : ℝ, (∀ x < 0, f x m ≥ x + 2/x) → m ≥ -3 - 2*Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_part_i_range_of_m_part_ii_l1779_177971


namespace NUMINAMATH_CALUDE_zara_brixton_height_l1779_177919

/-- The heights of four people satisfying certain conditions -/
structure Heights where
  itzayana : ℝ
  zora : ℝ
  brixton : ℝ
  zara : ℝ
  itzayana_taller : itzayana = zora + 4
  zora_shorter : zora = brixton - 8
  zara_equal : zara = brixton
  average_height : (itzayana + zora + brixton + zara) / 4 = 61

/-- Theorem stating that Zara and Brixton's height is 64 inches -/
theorem zara_brixton_height (h : Heights) : h.zara = 64 ∧ h.brixton = 64 := by
  sorry

end NUMINAMATH_CALUDE_zara_brixton_height_l1779_177919


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l1779_177904

theorem function_inequality_implies_a_bound (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 1, ∃ x₂ ∈ Set.Icc 2 3, 
    x₁ + 4/x₁ ≥ 2^x₂ + a) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l1779_177904


namespace NUMINAMATH_CALUDE_cookie_distribution_l1779_177992

theorem cookie_distribution (boxes : ℕ) (classes : ℕ) 
  (h1 : boxes = 3) (h2 : classes = 4) :
  (boxes : ℚ) / classes = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_cookie_distribution_l1779_177992


namespace NUMINAMATH_CALUDE_cube_folding_l1779_177986

/-- Represents the squares on the flat sheet --/
inductive Square
| A | B | C | D | E | F

/-- Represents the adjacency of squares on the flat sheet --/
def adjacent : Square → Square → Prop :=
  sorry

/-- Represents the opposite faces of the cube after folding --/
def opposite : Square → Square → Prop :=
  sorry

/-- The theorem to be proved --/
theorem cube_folding (h1 : adjacent Square.B Square.A)
                     (h2 : adjacent Square.C Square.B)
                     (h3 : adjacent Square.C Square.A)
                     (h4 : adjacent Square.D Square.C)
                     (h5 : adjacent Square.E Square.A)
                     (h6 : adjacent Square.F Square.D)
                     (h7 : adjacent Square.F Square.E) :
  opposite Square.A Square.D :=
sorry

end NUMINAMATH_CALUDE_cube_folding_l1779_177986


namespace NUMINAMATH_CALUDE_P_not_factorable_l1779_177997

/-- The polynomial P(x,y) = x^n + xy + y^n -/
def P (n : ℕ) (x y : ℝ) : ℝ := x^n + x*y + y^n

/-- Theorem stating that P(x,y) cannot be factored into two non-constant real polynomials -/
theorem P_not_factorable (n : ℕ) :
  ¬∃ (G H : ℝ → ℝ → ℝ), 
    (∀ x y, P n x y = G x y * H x y) ∧ 
    (∃ a b c d, G a b ≠ G c d) ∧ 
    (∃ a b c d, H a b ≠ H c d) :=
sorry

end NUMINAMATH_CALUDE_P_not_factorable_l1779_177997


namespace NUMINAMATH_CALUDE_inequality_holds_in_intervals_l1779_177927

theorem inequality_holds_in_intervals (a b : ℝ) : 
  (((0 ≤ a ∧ a < b ∧ b ≤ π/2) ∨ (π ≤ a ∧ a < b ∧ b ≤ 3*π/2)) → 
   (a - Real.sin a < b - Real.sin b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_in_intervals_l1779_177927


namespace NUMINAMATH_CALUDE_triangle_problem_l1779_177978

open Real

theorem triangle_problem (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * sin (A - C) = sin B) :
  sin A = (3 * sqrt 10) / 10 ∧
  (∀ (AB : ℝ), AB = 5 → ∃ (h : ℝ), h = 6 ∧ h * AB / 2 = sin C * (AB * sin A / sin C) * (AB * sin B / sin C) / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_problem_l1779_177978


namespace NUMINAMATH_CALUDE_tan_pi_plus_alpha_eq_two_implies_fraction_eq_three_l1779_177952

theorem tan_pi_plus_alpha_eq_two_implies_fraction_eq_three (α : Real) 
  (h : Real.tan (π + α) = 2) : 
  (Real.sin (α - π) + Real.cos (π - α)) / (Real.sin (π + α) - Real.cos (π - α)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_plus_alpha_eq_two_implies_fraction_eq_three_l1779_177952


namespace NUMINAMATH_CALUDE_handshakes_at_event_l1779_177969

/-- Represents the number of married couples at the event -/
def num_couples : ℕ := 15

/-- Calculates the total number of handshakes at the event -/
def total_handshakes (n : ℕ) : ℕ :=
  let num_men := n
  let num_women := n
  let handshakes_among_men := n * (n - 1) / 2
  let handshakes_men_women := n * (n - 1)
  handshakes_among_men + handshakes_men_women

/-- Theorem stating that the total number of handshakes is 315 -/
theorem handshakes_at_event : 
  total_handshakes num_couples = 315 := by
  sorry

#eval total_handshakes num_couples

end NUMINAMATH_CALUDE_handshakes_at_event_l1779_177969


namespace NUMINAMATH_CALUDE_visibility_time_proof_l1779_177998

/-- The time when Jenny and Kenny become visible to each other after being blocked by a circular building -/
def visibilityTime (buildingRadius : ℝ) (pathDistance : ℝ) (jennySpeed : ℝ) (kennySpeed : ℝ) : ℝ :=
  120

theorem visibility_time_proof (buildingRadius : ℝ) (pathDistance : ℝ) (jennySpeed : ℝ) (kennySpeed : ℝ) 
    (h1 : buildingRadius = 60)
    (h2 : pathDistance = 240)
    (h3 : jennySpeed = 4)
    (h4 : kennySpeed = 2) :
  visibilityTime buildingRadius pathDistance jennySpeed kennySpeed = 120 :=
by
  sorry

#check visibility_time_proof

end NUMINAMATH_CALUDE_visibility_time_proof_l1779_177998


namespace NUMINAMATH_CALUDE_layla_and_alan_apples_l1779_177946

def maggie_apples : ℕ := 40
def kelsey_apples : ℕ := 28
def total_people : ℕ := 4
def average_apples : ℕ := 30

theorem layla_and_alan_apples :
  ∃ (layla_apples alan_apples : ℕ),
    maggie_apples + kelsey_apples + layla_apples + alan_apples = total_people * average_apples ∧
    layla_apples + alan_apples = 52 :=
by sorry

end NUMINAMATH_CALUDE_layla_and_alan_apples_l1779_177946


namespace NUMINAMATH_CALUDE_problem_statement_l1779_177915

def f (x : ℝ) := x^3 - x^2

theorem problem_statement :
  (∀ m n : ℝ, m > 0 → n > 0 → m * n > 1 → max (f m) (f n) ≥ 0) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a ≠ b → f a = f b → a + b > 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1779_177915


namespace NUMINAMATH_CALUDE_fraction_above_line_is_seven_tenths_l1779_177968

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : Prod ℝ ℝ
  topRight : Prod ℝ ℝ

/-- A line in the 2D plane defined by two points --/
structure Line where
  point1 : Prod ℝ ℝ
  point2 : Prod ℝ ℝ

/-- Calculate the area of a square --/
def squareArea (s : Square) : ℝ :=
  (s.topRight.1 - s.bottomLeft.1) * (s.topRight.2 - s.bottomLeft.2)

/-- Calculate the fraction of a square's area above a line --/
noncomputable def fractionAboveLine (s : Square) (l : Line) : ℝ :=
  sorry

/-- Theorem stating the fraction of the square's area above the given line --/
theorem fraction_above_line_is_seven_tenths (s : Square) (l : Line) :
  s.bottomLeft = (2, 0) ∧ s.topRight = (7, 5) ∧
  l.point1 = (2, 1) ∧ l.point2 = (7, 3) →
  fractionAboveLine s l = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_above_line_is_seven_tenths_l1779_177968


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1779_177999

theorem infinite_series_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let series : ℕ → ℝ := fun n =>
    1 / ((n - 1) * a - (n - 3) * b) / (n * a - (2 * n - 3) * b)
  ∑' n, series n = 1 / ((a - b) * b) :=
by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1779_177999


namespace NUMINAMATH_CALUDE_project_budget_l1779_177941

theorem project_budget (total_spent : ℕ) (over_budget : ℕ) : 
  total_spent = 6580 →
  over_budget = 280 →
  ∃ (monthly_allocation : ℕ),
    monthly_allocation * 6 = total_spent - over_budget ∧
    monthly_allocation * 12 = 12600 := by
  sorry

end NUMINAMATH_CALUDE_project_budget_l1779_177941


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1779_177936

theorem sqrt_equation_solution : ∃ (a b c : ℕ+), 
  (2 * Real.sqrt (Real.sqrt 4 - Real.sqrt 3) = Real.sqrt a.val - Real.sqrt b.val + Real.sqrt c.val) ∧
  (a.val + b.val + c.val = 22) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1779_177936


namespace NUMINAMATH_CALUDE_parallel_lines_k_equals_two_l1779_177953

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The slope-intercept form of the line 2x - y + 2 = 0 -/
def line1_slope_intercept (x y : ℝ) : Prop := y = 2 * x + 2

/-- The slope-intercept form of the line y = kx + 1 -/
def line2_slope_intercept (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

theorem parallel_lines_k_equals_two :
  (∀ x y : ℝ, 2 * x - y + 2 = 0 ↔ y = k * x + 1) → k = 2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_k_equals_two_l1779_177953


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1779_177918

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (2 - i) / (1 + i) = (1 : ℂ) / 2 - (3 : ℂ) / 2 * i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1779_177918


namespace NUMINAMATH_CALUDE_connie_calculation_l1779_177929

theorem connie_calculation (x : ℝ) : 4 * x = 200 → x / 4 + 10 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_connie_calculation_l1779_177929


namespace NUMINAMATH_CALUDE_exactly_21_numbers_reach_one_in_8_steps_l1779_177921

def operation (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n + 1

def reachesOneIn (steps : ℕ) (n : ℕ) : Prop :=
  ∃ (sequence : Fin (steps + 1) → ℕ),
    sequence 0 = n ∧
    sequence steps = 1 ∧
    ∀ i : Fin steps, sequence (i + 1) = operation (sequence i)

theorem exactly_21_numbers_reach_one_in_8_steps :
  ∃! (s : Finset ℕ), s.card = 21 ∧ ∀ n, n ∈ s ↔ reachesOneIn 8 n :=
sorry

end NUMINAMATH_CALUDE_exactly_21_numbers_reach_one_in_8_steps_l1779_177921


namespace NUMINAMATH_CALUDE_fraction_simplification_l1779_177956

theorem fraction_simplification (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  (1 / y) / (1 / x) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1779_177956


namespace NUMINAMATH_CALUDE_expression_evaluation_l1779_177916

theorem expression_evaluation : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1779_177916


namespace NUMINAMATH_CALUDE_expression_evaluation_l1779_177903

theorem expression_evaluation (x y : ℚ) (hx : x = 4/3) (hy : y = 5/8) :
  (6*x + 8*y) / (48*x*y) = 13/40 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1779_177903


namespace NUMINAMATH_CALUDE_fertilizer_transport_l1779_177931

theorem fertilizer_transport (x y t : ℕ) : 
  (x * t = (x - 4) * (t + 10)) →
  (y * t = (y - 3) * (t + 10)) →
  (x * t - y * t = 60) →
  (x - 4 = 8) ∧ (y - 3 = 6) ∧ (t + 10 = 30) :=
by sorry

end NUMINAMATH_CALUDE_fertilizer_transport_l1779_177931


namespace NUMINAMATH_CALUDE_horner_v3_value_l1779_177972

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^6 - 5x^5 + 6x^4 + x^2 + 3x + 2 -/
def f : List ℤ := [2, 3, 1, 0, 6, -5, 1]

/-- Theorem: Horner's method for f(x) at x = -2 gives v₃ = -40 -/
theorem horner_v3_value :
  let coeffs := f.take 4
  horner coeffs (-2) = -40 := by sorry

end NUMINAMATH_CALUDE_horner_v3_value_l1779_177972


namespace NUMINAMATH_CALUDE_dollar_sum_squared_zero_l1779_177950

/-- Definition of the $ operation for real numbers -/
def dollar (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem: For real numbers x and y, (x + y)^2 $ (y + x)^2 = 0 -/
theorem dollar_sum_squared_zero (x y : ℝ) : dollar ((x + y)^2) ((y + x)^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_dollar_sum_squared_zero_l1779_177950


namespace NUMINAMATH_CALUDE_chord_intersection_tangent_circle_l1779_177905

/-- Given a point A and a circle S with center O and radius R, 
    prove that the line through A intersecting S in a chord PQ of length d 
    is tangent to a circle with center O and radius sqrt(R^2 - d^2/4) -/
theorem chord_intersection_tangent_circle 
  (A O : ℝ × ℝ) (R d : ℝ) (S : Set (ℝ × ℝ)) :
  let circle_S := {p : ℝ × ℝ | dist p O = R}
  let chord_length (l : Set (ℝ × ℝ)) := ∃ P Q : ℝ × ℝ, P ∈ l ∩ S ∧ Q ∈ l ∩ S ∧ dist P Q = d
  let tangent_circle := {p : ℝ × ℝ | dist p O = Real.sqrt (R^2 - d^2/4)}
  ∀ l : Set (ℝ × ℝ), A ∈ l → S = circle_S → chord_length l → 
    ∃ p : ℝ × ℝ, p ∈ l ∩ tangent_circle :=
by sorry

end NUMINAMATH_CALUDE_chord_intersection_tangent_circle_l1779_177905


namespace NUMINAMATH_CALUDE_system_solution_l1779_177937

theorem system_solution : 
  ∀ x y z : ℝ, 
    (y * z = 3 * y + 2 * z - 8) ∧ 
    (z * x = 4 * z + 3 * x - 8) ∧ 
    (x * y = 2 * x + y - 1) → 
    ((x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 5/2 ∧ z = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1779_177937


namespace NUMINAMATH_CALUDE_employee_pay_l1779_177966

/-- Proves that employee y is paid 268.18 per week given the conditions -/
theorem employee_pay (x y : ℝ) (h1 : x + y = 590) (h2 : x = 1.2 * y) : y = 268.18 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_l1779_177966


namespace NUMINAMATH_CALUDE_y_value_l1779_177920

theorem y_value (x y : ℝ) (h1 : x^2 = y - 7) (h2 : x = 7) : y = 56 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l1779_177920


namespace NUMINAMATH_CALUDE_odd_function_value_l1779_177910

/-- Given a function f(x) = sin(x + φ) + √3 cos(x + φ) where 0 ≤ φ ≤ π,
    if f(x) is an odd function, then f(π/6) = -1 -/
theorem odd_function_value (φ : Real) (h1 : 0 ≤ φ) (h2 : φ ≤ π) :
  let f : Real → Real := λ x => Real.sin (x + φ) + Real.sqrt 3 * Real.cos (x + φ)
  (∀ x, f (-x) = -f x) →
  f (π / 6) = -1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_value_l1779_177910


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1779_177944

def M : Set ℕ := {1, 2, 3, 6, 7}
def N : Set ℕ := {1, 2, 4, 5}

theorem intersection_of_M_and_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1779_177944


namespace NUMINAMATH_CALUDE_intersection_range_l1779_177987

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the property of having three distinct intersection points
def has_three_distinct_intersections (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
  f x₁ = a ∧ f x₂ = a ∧ f x₃ = a

-- Theorem statement
theorem intersection_range (a : ℝ) :
  has_three_distinct_intersections a → -2 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l1779_177987


namespace NUMINAMATH_CALUDE_polynomial_factor_l1779_177996

-- Define the polynomials
def P (c : ℝ) (x : ℝ) : ℝ := 3 * x^3 + c * x + 12
def Q (q : ℝ) (x : ℝ) : ℝ := x^2 + q * x + 2

-- Theorem statement
theorem polynomial_factor (c : ℝ) :
  (∃ q r : ℝ, ∀ x : ℝ, P c x = Q q x * (r * x + (12 / r))) → c = 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_l1779_177996


namespace NUMINAMATH_CALUDE_cylindrical_to_cartesian_l1779_177982

/-- Given a point P in cylindrical coordinates (r, θ, z) = (√2, π/4, 1),
    prove that its Cartesian coordinates (x, y, z) are (1, 1, 1). -/
theorem cylindrical_to_cartesian :
  let r : ℝ := Real.sqrt 2
  let θ : ℝ := π / 4
  let z : ℝ := 1
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y, z) = (1, 1, 1) := by sorry

end NUMINAMATH_CALUDE_cylindrical_to_cartesian_l1779_177982


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1779_177961

theorem quadratic_coefficient (a b c : ℤ) :
  (∀ x : ℝ, a * x^2 + b * x + c = a * (x - 2)^2 + 3) →
  a * 1^2 + b * 1 + c = 5 →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1779_177961


namespace NUMINAMATH_CALUDE_total_tickets_sold_l1779_177973

/-- Represents the number of tickets sold at different prices and times -/
structure TicketSales where
  reducedFirstWeek : ℕ
  reducedRemainingWeeks : ℕ
  fullPrice : ℕ

/-- Calculates the total number of tickets sold -/
def totalTicketsSold (sales : TicketSales) : ℕ :=
  sales.reducedFirstWeek + sales.reducedRemainingWeeks + sales.fullPrice

/-- Theorem stating the total number of tickets sold given the conditions -/
theorem total_tickets_sold :
  ∀ (sales : TicketSales),
    sales.reducedFirstWeek = 5400 →
    sales.fullPrice = 16500 →
    sales.fullPrice = 5 * sales.reducedRemainingWeeks →
    totalTicketsSold sales = 25200 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l1779_177973


namespace NUMINAMATH_CALUDE_club_truncator_season_probability_l1779_177967

/-- Represents the possible outcomes of a soccer match -/
inductive MatchResult
| Win
| Lose
| Tie

/-- Represents the season results for Club Truncator -/
structure SeasonResult :=
  (wins : ℕ)
  (losses : ℕ)
  (ties : ℕ)

/-- The number of teams in the league -/
def numTeams : ℕ := 8

/-- The number of matches Club Truncator plays -/
def numMatches : ℕ := 7

/-- The probability of winning a single match -/
def winProb : ℚ := 2/5

/-- The probability of losing a single match -/
def loseProb : ℚ := 1/5

/-- The probability of tying a single match -/
def tieProb : ℚ := 2/5

/-- Checks if a season result has more wins than losses -/
def moreWinsThanLosses (result : SeasonResult) : Prop :=
  result.wins > result.losses

/-- The probability of Club Truncator finishing with more wins than losses -/
def probMoreWinsThanLosses : ℚ := 897/2187

theorem club_truncator_season_probability :
  probMoreWinsThanLosses = 897/2187 := by sorry

end NUMINAMATH_CALUDE_club_truncator_season_probability_l1779_177967


namespace NUMINAMATH_CALUDE_salary_difference_l1779_177911

def initial_salary : ℝ := 30000
def hansel_raise_percent : ℝ := 0.10
def gretel_raise_percent : ℝ := 0.15

def hansel_new_salary : ℝ := initial_salary * (1 + hansel_raise_percent)
def gretel_new_salary : ℝ := initial_salary * (1 + gretel_raise_percent)

theorem salary_difference :
  gretel_new_salary - hansel_new_salary = 1500 := by
  sorry

end NUMINAMATH_CALUDE_salary_difference_l1779_177911


namespace NUMINAMATH_CALUDE_total_rice_weight_l1779_177902

-- Define the number of containers
def num_containers : ℕ := 4

-- Define the weight of rice in each container (in ounces)
def rice_per_container : ℝ := 29

-- Define the conversion rate from ounces to pounds
def ounces_per_pound : ℝ := 16

-- State the theorem
theorem total_rice_weight :
  (num_containers : ℝ) * rice_per_container / ounces_per_pound = 7.25 := by
  sorry

end NUMINAMATH_CALUDE_total_rice_weight_l1779_177902


namespace NUMINAMATH_CALUDE_f_shifted_l1779_177965

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_shifted (x : ℝ) : f (x + 1) = x^2 + 2*x → f (x - 1) = x^2 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_f_shifted_l1779_177965


namespace NUMINAMATH_CALUDE_journey_speed_proof_l1779_177977

/-- Proves that given a journey of 120 miles in 90 minutes, where the average speed
    for the first 30 minutes was 70 mph and for the second 30 minutes was 75 mph,
    the average speed for the last 30 minutes must be 95 mph. -/
theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) 
    (speed1 : ℝ) (speed2 : ℝ) (time_segment : ℝ) :
    total_distance = 120 →
    total_time = 1.5 →
    speed1 = 70 →
    speed2 = 75 →
    time_segment = 0.5 →
    speed1 * time_segment + speed2 * time_segment + 
    ((total_distance - (speed1 * time_segment + speed2 * time_segment)) / time_segment) = 
    total_distance / total_time :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_proof_l1779_177977


namespace NUMINAMATH_CALUDE_net_profit_calculation_l1779_177938

/-- Calculates the net profit percentage given a markup percentage and discount percentage -/
def netProfitPercentage (markup : ℝ) (discount : ℝ) : ℝ :=
  let markedPrice := 1 + markup
  let sellingPrice := markedPrice * (1 - discount)
  sellingPrice - 1

/-- Theorem stating that a 20% markup followed by a 15% discount results in a 2% net profit -/
theorem net_profit_calculation :
  netProfitPercentage 0.2 0.15 = 0.02 := by
  sorry

#eval netProfitPercentage 0.2 0.15

end NUMINAMATH_CALUDE_net_profit_calculation_l1779_177938


namespace NUMINAMATH_CALUDE_courtyard_width_main_theorem_l1779_177991

/-- Proves that the width of a rectangular courtyard is 16 meters -/
theorem courtyard_width : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun (length width brick_length brick_width : ℝ) =>
    length = 30 ∧
    brick_length = 0.2 ∧
    brick_width = 0.1 ∧
    (length * width) / (brick_length * brick_width) = 24000 →
    width = 16

/-- Main theorem proof -/
theorem main_theorem : courtyard_width 30 16 0.2 0.1 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_width_main_theorem_l1779_177991


namespace NUMINAMATH_CALUDE_remainder_sum_l1779_177926

theorem remainder_sum (a b : ℤ) 
  (ha : a % 60 = 53) 
  (hb : b % 45 = 22) : 
  (a + b) % 30 = 15 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l1779_177926


namespace NUMINAMATH_CALUDE_bcm_percentage_is_twenty_percent_l1779_177942

/-- The percentage of Black Copper Marans (BCM) in a flock of chickens -/
def bcm_percentage (total_chickens : ℕ) (bcm_hen_percentage : ℚ) (bcm_hens : ℕ) : ℚ :=
  (bcm_hens : ℚ) / (bcm_hen_percentage * total_chickens)

/-- Theorem stating that the percentage of BCM in a flock of 100 chickens is 20%,
    given that 80% of BCM are hens and there are 16 BCM hens -/
theorem bcm_percentage_is_twenty_percent :
  bcm_percentage 100 (4/5) 16 = 1/5 := by
  sorry

#eval bcm_percentage 100 (4/5) 16

end NUMINAMATH_CALUDE_bcm_percentage_is_twenty_percent_l1779_177942


namespace NUMINAMATH_CALUDE_license_plate_theorem_l1779_177984

/-- The number of distinct digits available for the license plate -/
def num_digits : ℕ := 10

/-- The number of distinct letters available for the license plate -/
def num_letters : ℕ := 26

/-- The number of digits in the license plate -/
def digits_count : ℕ := 5

/-- The number of letters in the license plate -/
def letters_count : ℕ := 3

/-- The number of slots available for letter placement -/
def letter_slots : ℕ := digits_count + 1

/-- Calculates the number of distinct license plates -/
def license_plate_count : ℕ :=
  num_digits ^ digits_count *
  num_letters ^ letters_count *
  Nat.choose letter_slots letters_count

theorem license_plate_theorem :
  license_plate_count = 35152000000 := by sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l1779_177984


namespace NUMINAMATH_CALUDE_childrens_buffet_price_l1779_177928

def adult_price : ℚ := 30
def senior_discount : ℚ := 1/10
def num_adults : ℕ := 2
def num_seniors : ℕ := 2
def num_children : ℕ := 3
def total_spent : ℚ := 159

theorem childrens_buffet_price :
  ∃ (child_price : ℚ),
    child_price * num_children +
    adult_price * num_adults +
    adult_price * (1 - senior_discount) * num_seniors = total_spent ∧
    child_price = 15 := by
  sorry

end NUMINAMATH_CALUDE_childrens_buffet_price_l1779_177928


namespace NUMINAMATH_CALUDE_average_income_l1779_177951

/-- The average monthly income problem -/
theorem average_income (A B C : ℕ) : 
  (B + C) / 2 = 5250 →
  (A + C) / 2 = 4200 →
  A = 3000 →
  (A + B) / 2 = 4050 := by
sorry

end NUMINAMATH_CALUDE_average_income_l1779_177951


namespace NUMINAMATH_CALUDE_roots_triangle_condition_l1779_177947

/-- A cubic equation with coefficients p, q, and r -/
structure CubicEquation where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The roots of a cubic equation form a triangle -/
def roots_form_triangle (eq : CubicEquation) : Prop :=
  ∃ (u v w : ℝ), u > 0 ∧ v > 0 ∧ w > 0 ∧
    u^3 + eq.p * u^2 + eq.q * u + eq.r = 0 ∧
    v^3 + eq.p * v^2 + eq.q * v + eq.r = 0 ∧
    w^3 + eq.p * w^2 + eq.q * w + eq.r = 0 ∧
    u + v > w ∧ u + w > v ∧ v + w > u

/-- The theorem stating the condition for roots to form a triangle -/
theorem roots_triangle_condition (eq : CubicEquation) :
  roots_form_triangle eq ↔ eq.p^3 - 4 * eq.p * eq.q + 8 * eq.r > 0 :=
sorry

end NUMINAMATH_CALUDE_roots_triangle_condition_l1779_177947


namespace NUMINAMATH_CALUDE_max_red_points_l1779_177935

/-- Represents a point on the circle -/
structure Point where
  color : Bool  -- True for red, False for blue
  connections : Nat

/-- Represents the circle with its points -/
structure Circle where
  points : Finset Point
  total_points : Nat
  red_points : Nat
  blue_points : Nat
  valid_connections : Bool

/-- The main theorem statement -/
theorem max_red_points (c : Circle) : 
  c.total_points = 25 ∧ 
  c.red_points + c.blue_points = c.total_points ∧
  c.valid_connections ∧
  (∀ p q : Point, p ∈ c.points → q ∈ c.points → 
    p.color = true → q.color = true → p ≠ q → p.connections ≠ q.connections) →
  c.red_points ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_max_red_points_l1779_177935


namespace NUMINAMATH_CALUDE_parallel_intersections_l1779_177955

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the intersection of a plane with another plane resulting in a line
variable (intersect : Plane → Plane → Line)

-- Define the parallel relation for lines
variable (parallel_lines : Line → Line → Prop)

-- Theorem statement
theorem parallel_intersections
  (P1 P2 P3 : Plane) (l1 l2 : Line)
  (h1 : parallel_planes P1 P2)
  (h2 : l1 = intersect P3 P1)
  (h3 : l2 = intersect P3 P2) :
  parallel_lines l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_intersections_l1779_177955


namespace NUMINAMATH_CALUDE_six_last_digit_to_appear_l1779_177959

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to check if a digit has appeared in the sequence up to n
def digitAppeared (d : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ n ∧ unitsDigit (fib k) = d

-- Theorem statement
theorem six_last_digit_to_appear :
  ∀ d : ℕ, d < 10 → d ≠ 6 →
    ∃ n : ℕ, digitAppeared d n ∧ ¬digitAppeared 6 n :=
by sorry

end NUMINAMATH_CALUDE_six_last_digit_to_appear_l1779_177959


namespace NUMINAMATH_CALUDE_dart_second_session_score_l1779_177933

/-- Represents the points scored in each dart-throwing session -/
structure DartScores where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Checks if the given DartScores satisfy the problem conditions -/
def validScores (scores : DartScores) : Prop :=
  scores.second = 2 * scores.first ∧
  scores.third = 3 * scores.first ∧
  scores.first ≥ 8

theorem dart_second_session_score (scores : DartScores) :
  validScores scores → scores.second = 48 := by
  sorry

#check dart_second_session_score

end NUMINAMATH_CALUDE_dart_second_session_score_l1779_177933


namespace NUMINAMATH_CALUDE_sean_patch_profit_l1779_177976

/-- Calculates the net profit for Sean's patch business -/
theorem sean_patch_profit :
  let order_quantity : ℕ := 100
  let cost_per_patch : ℚ := 125/100
  let sell_price_per_patch : ℚ := 12
  let total_cost : ℚ := order_quantity * cost_per_patch
  let total_revenue : ℚ := order_quantity * sell_price_per_patch
  let net_profit : ℚ := total_revenue - total_cost
  net_profit = 1075 := by sorry

end NUMINAMATH_CALUDE_sean_patch_profit_l1779_177976


namespace NUMINAMATH_CALUDE_at_most_one_lattice_point_on_circle_l1779_177948

/-- A point in a 2D lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- The squared distance between two points -/
def squaredDistance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem at_most_one_lattice_point_on_circle 
  (center : ℝ × ℝ) 
  (h_center : center = (Real.sqrt 2, Real.sqrt 3)) 
  (p q : LatticePoint) 
  (r : ℝ) 
  (h_p : squaredDistance (p.x, p.y) center = r^2) 
  (h_q : squaredDistance (q.x, q.y) center = r^2) : 
  p = q :=
sorry

end NUMINAMATH_CALUDE_at_most_one_lattice_point_on_circle_l1779_177948


namespace NUMINAMATH_CALUDE_days_worked_by_c_l1779_177914

/-- Represents the number of days worked by person a -/
def days_a : ℕ := 6

/-- Represents the number of days worked by person b -/
def days_b : ℕ := 9

/-- Represents the daily wage of person c -/
def wage_c : ℕ := 100

/-- Represents the total earnings of all three people -/
def total_earnings : ℕ := 1480

/-- Represents the ratio of daily wages for a, b, and c -/
def wage_ratio : Fin 3 → ℕ 
  | 0 => 3
  | 1 => 4
  | 2 => 5

/-- 
Proves that given the conditions, the number of days worked by person c is 4
-/
theorem days_worked_by_c : 
  ∃ (days_c : ℕ), 
    days_c * wage_c + 
    days_a * (wage_ratio 0 * wage_c / wage_ratio 2) + 
    days_b * (wage_ratio 1 * wage_c / wage_ratio 2) = 
    total_earnings ∧ days_c = 4 := by
  sorry

end NUMINAMATH_CALUDE_days_worked_by_c_l1779_177914


namespace NUMINAMATH_CALUDE_katya_magic_pen_problem_l1779_177962

theorem katya_magic_pen_problem (katya_prob : ℚ) (pen_prob : ℚ) (total_problems : ℕ) (min_correct : ℕ) :
  katya_prob = 4/5 →
  pen_prob = 1/2 →
  total_problems = 20 →
  min_correct = 13 →
  ∃ x : ℕ, x ≥ 10 ∧
    (x : ℚ) * katya_prob + (total_problems - x : ℚ) * pen_prob ≥ min_correct ∧
    ∀ y : ℕ, y < 10 →
      (y : ℚ) * katya_prob + (total_problems - y : ℚ) * pen_prob < min_correct :=
by sorry

end NUMINAMATH_CALUDE_katya_magic_pen_problem_l1779_177962


namespace NUMINAMATH_CALUDE_cost_calculation_l1779_177995

/-- The cost of items given their quantities and price ratios -/
def cost_of_items (pen_price pencil_price eraser_price : ℚ) : ℚ :=
  4 * pen_price + 6 * pencil_price + 2 * eraser_price

/-- The cost of a dozen pens and half a dozen erasers -/
def cost_of_dozen_pens_and_half_dozen_erasers (pen_price eraser_price : ℚ) : ℚ :=
  12 * pen_price + 6 * eraser_price

theorem cost_calculation :
  ∀ (x : ℚ),
    cost_of_items (4*x) (2*x) x = 360 →
    cost_of_dozen_pens_and_half_dozen_erasers (4*x) x = 648 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_calculation_l1779_177995


namespace NUMINAMATH_CALUDE_can_make_all_white_l1779_177963

/-- Represents the color of a number -/
inductive Color
| Black
| White

/-- Represents a move in the repainting process -/
structure Move where
  number : Nat
  deriving Repr

/-- The state of all numbers from 1 to 1,000,000 -/
def State := Fin 1000000 → Color

/-- Apply a move to a state -/
def applyMove (s : State) (m : Move) : State :=
  sorry

/-- Check if all numbers in the state are white -/
def allWhite (s : State) : Prop :=
  sorry

/-- The initial state where all numbers are black -/
def initialState : State :=
  sorry

/-- Theorem stating that it's possible to make all numbers white -/
theorem can_make_all_white : ∃ (moves : List Move), allWhite (moves.foldl applyMove initialState) := by
  sorry

end NUMINAMATH_CALUDE_can_make_all_white_l1779_177963


namespace NUMINAMATH_CALUDE_equal_area_rectangles_intersection_l1779_177974

/-- A rectangle represented by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents a horizontal line segment -/
structure HorizontalSegment where
  y : ℝ
  x1 : ℝ
  x2 : ℝ

theorem equal_area_rectangles_intersection 
  (r1 r2 : Rectangle) 
  (h_equal_area : r1.area = r2.area) :
  ∃ (f : ℝ × ℝ → ℝ × ℝ) (g : ℝ × ℝ → ℝ × ℝ),
    (∀ x y, f (x, y) = g (x, y) → 
      ∃ (s1 s2 : HorizontalSegment), 
        s1.y = s2.y ∧ 
        s1.x2 - s1.x1 = s2.x2 - s2.x1 ∧
        (s1.x1, s1.y) ∈ Set.range f ∧
        (s1.x2, s1.y) ∈ Set.range f ∧
        (s2.x1, s2.y) ∈ Set.range g ∧
        (s2.x2, s2.y) ∈ Set.range g) := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_intersection_l1779_177974


namespace NUMINAMATH_CALUDE_remainder_problem_l1779_177932

theorem remainder_problem (N : ℤ) : 
  N % 37 = 1 → N % 296 = 260 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1779_177932


namespace NUMINAMATH_CALUDE_mixture_weight_l1779_177988

/-- The weight of the mixture of two brands of vegetable ghee -/
theorem mixture_weight (brand_a_weight : ℝ) (brand_b_weight : ℝ) 
  (mix_ratio_a : ℝ) (mix_ratio_b : ℝ) (total_volume : ℝ) : 
  brand_a_weight = 900 →
  brand_b_weight = 750 →
  mix_ratio_a = 3 →
  mix_ratio_b = 2 →
  total_volume = 4 →
  (mix_ratio_a * total_volume * brand_a_weight + mix_ratio_b * total_volume * brand_b_weight) / 
  ((mix_ratio_a + mix_ratio_b) * 1000) = 3.36 := by
  sorry

#check mixture_weight

end NUMINAMATH_CALUDE_mixture_weight_l1779_177988


namespace NUMINAMATH_CALUDE_milk_delivery_proof_l1779_177970

/-- The amount of milk in liters delivered to Minjeong's house in a week -/
def milk_per_week (bottles_per_day : ℕ) (liters_per_bottle : ℚ) (days_in_week : ℕ) : ℚ :=
  (bottles_per_day : ℚ) * liters_per_bottle * (days_in_week : ℚ)

/-- Proof that 4.2 liters of milk are delivered to Minjeong's house in a week -/
theorem milk_delivery_proof :
  milk_per_week 3 (2/10) 7 = 21/5 := by
  sorry

end NUMINAMATH_CALUDE_milk_delivery_proof_l1779_177970


namespace NUMINAMATH_CALUDE_more_difficult_than_easy_l1779_177922

/-- Represents the number of problems solved by a specific number of people -/
structure ProblemCounts where
  total : ℕ
  solvedByOne : ℕ
  solvedByTwo : ℕ
  solvedByThree : ℕ

/-- The total number of problems solved by each person -/
def problemsPerPerson : ℕ := 60

theorem more_difficult_than_easy (p : ProblemCounts) :
  p.total = 100 →
  p.solvedByOne + p.solvedByTwo + p.solvedByThree = p.total →
  p.solvedByOne + 3 * p.solvedByThree + 2 * p.solvedByTwo = 3 * problemsPerPerson →
  p.solvedByOne = p.solvedByThree + 20 :=
by
  sorry

#check more_difficult_than_easy

end NUMINAMATH_CALUDE_more_difficult_than_easy_l1779_177922


namespace NUMINAMATH_CALUDE_range_of_z_l1779_177909

theorem range_of_z (x y : ℝ) (h1 : x + 2 ≥ y) (h2 : x + 2*y ≥ 4) (h3 : y ≤ 5 - 2*x) :
  let z := (2*x + y - 1) / (x + 1)
  ∃ (z_min z_max : ℝ), z_min = 1 ∧ z_max = 2 ∧ ∀ z', z' = z → z_min ≤ z' ∧ z' ≤ z_max :=
by sorry

end NUMINAMATH_CALUDE_range_of_z_l1779_177909


namespace NUMINAMATH_CALUDE_inequality_proof_l1779_177925

theorem inequality_proof (x : ℝ) (h : x ≠ 1) :
  Real.sqrt (x^2 - 2*x + 2) ≥ -Real.sqrt 5 * x ↔ (-1 ≤ x ∧ x < 1) ∨ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1779_177925


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_15_l1779_177958

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_sum_factorials_15 :
  last_two_digits (sum_factorials 15) = 13 :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_15_l1779_177958


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l1779_177957

noncomputable def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  2 * Real.sqrt 3 * (Real.sin ((A + B) / 2))^2 - Real.sin C = Real.sqrt 3 ∧
  c = Real.sqrt 3 ∧
  a = Real.sqrt 2

theorem triangle_ABC_properties {A B C a b c : ℝ} 
  (h : triangle_ABC A B C a b c) : 
  C = π / 3 ∧ 
  (1/2 * a * b * Real.sin C) = (Real.sqrt 3 + 3) / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l1779_177957


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l1779_177901

theorem consecutive_odd_integers_sum (n : ℤ) : 
  (n + (n + 4) = 130) → (n + (n + 2) + (n + 4) = 195) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l1779_177901


namespace NUMINAMATH_CALUDE_right_triangle_among_sets_l1779_177939

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_among_sets : 
  (¬ is_right_triangle 2 3 4) ∧
  (is_right_triangle 3 4 5) ∧
  (¬ is_right_triangle 4 5 6) ∧
  (¬ is_right_triangle 6 8 9) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_among_sets_l1779_177939


namespace NUMINAMATH_CALUDE_unique_solution_abc_l1779_177900

theorem unique_solution_abc (a b c : ℕ+) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : a * b + b * c + c * a = a * b * c) : 
  a = 2 ∧ b = 3 ∧ c = 6 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_abc_l1779_177900


namespace NUMINAMATH_CALUDE_candy_distribution_l1779_177975

/-- 
Given a group of students where each student receives a fixed number of candy pieces,
this theorem proves that the total number of candy pieces given away is equal to
the product of the number of students and the number of pieces per student.
-/
theorem candy_distribution (num_students : ℕ) (pieces_per_student : ℕ) 
  (h1 : num_students = 9) 
  (h2 : pieces_per_student = 2) : 
  num_students * pieces_per_student = 18 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1779_177975


namespace NUMINAMATH_CALUDE_smallest_irreducible_n_l1779_177979

def is_irreducible (n k : ℕ) : Prop :=
  Nat.gcd k (n + k + 2) = 1

def all_irreducible (n : ℕ) : Prop :=
  ∀ k : ℕ, 68 ≤ k → k ≤ 133 → is_irreducible n k

theorem smallest_irreducible_n :
  (all_irreducible 65 ∧
   all_irreducible 135 ∧
   (∀ n : ℕ, n < 65 → ¬all_irreducible n) ∧
   (∀ n : ℕ, 65 < n → n < 135 → ¬all_irreducible n)) :=
sorry

end NUMINAMATH_CALUDE_smallest_irreducible_n_l1779_177979


namespace NUMINAMATH_CALUDE_expression_multiple_of_six_l1779_177923

theorem expression_multiple_of_six (n : ℕ) (h : n ≥ 10) :
  ∃ k : ℤ, ((n + 3).factorial - (n + 1).factorial) / n.factorial = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_expression_multiple_of_six_l1779_177923


namespace NUMINAMATH_CALUDE_sequence_property_l1779_177945

theorem sequence_property (n : ℕ) (x : ℕ → ℚ) (h_n : n ≥ 7) 
  (h_def : ∀ k > 1, x k = 1 / (1 - x (k-1)))
  (h_x2 : x 2 = 5) : 
  x 7 = 4/5 := by
sorry

end NUMINAMATH_CALUDE_sequence_property_l1779_177945
