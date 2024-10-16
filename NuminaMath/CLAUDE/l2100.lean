import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l2100_210016

theorem range_of_a (a : ℝ) :
  (a + 1)^(-1/2 : ℝ) < (3 - 2*a)^(-1/2 : ℝ) → 2/3 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2100_210016


namespace NUMINAMATH_CALUDE_largest_multiple_of_eight_less_than_neg_63_l2100_210061

theorem largest_multiple_of_eight_less_than_neg_63 :
  ∀ n : ℤ, n * 8 < -63 → n * 8 ≤ -64 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_eight_less_than_neg_63_l2100_210061


namespace NUMINAMATH_CALUDE_no_solution_to_inequality_system_l2100_210047

theorem no_solution_to_inequality_system :
  ¬∃ x : ℝ, (x / 6 + 7 / 2 > (3 * x + 29) / 5) ∧
            (x + 9 / 2 > x / 8) ∧
            (11 / 3 - x / 6 < (34 - 3 * x) / 5) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_inequality_system_l2100_210047


namespace NUMINAMATH_CALUDE_reciprocal_equality_implies_equality_l2100_210063

theorem reciprocal_equality_implies_equality (x y : ℝ) (h : x ≠ 0) (k : y ≠ 0) : 
  1 / x = 1 / y → x = y := by
sorry

end NUMINAMATH_CALUDE_reciprocal_equality_implies_equality_l2100_210063


namespace NUMINAMATH_CALUDE_counterexample_exists_l2100_210054

theorem counterexample_exists : ∃ n : ℕ, 
  (Even n) ∧ (¬ Prime n) ∧ (¬ Prime (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2100_210054


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2100_210037

/-- Given a line segment from (2, 2) to (x, 6) with length 10 and x > 0, prove x = 2 + 2√21 -/
theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  (x - 2)^2 + 4^2 = 10^2 → 
  x = 2 + 2 * Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2100_210037


namespace NUMINAMATH_CALUDE_machines_completion_time_l2100_210029

theorem machines_completion_time 
  (time_A time_B time_C time_D time_E : ℝ) 
  (h_A : time_A = 4)
  (h_B : time_B = 12)
  (h_C : time_C = 6)
  (h_D : time_D = 8)
  (h_E : time_E = 18) :
  (1 / (1/time_A + 1/time_B + 1/time_C + 1/time_D + 1/time_E)) = 72/49 := by
  sorry

end NUMINAMATH_CALUDE_machines_completion_time_l2100_210029


namespace NUMINAMATH_CALUDE_sum_not_zero_l2100_210045

theorem sum_not_zero (a b c d : ℝ) 
  (eq1 : a * b * c - d = 1)
  (eq2 : b * c * d - a = 2)
  (eq3 : c * d * a - b = 3)
  (eq4 : d * a * b - c = -6) : 
  a + b + c + d ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_not_zero_l2100_210045


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2100_210018

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = q * a n) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2100_210018


namespace NUMINAMATH_CALUDE_unique_consecutive_triangle_with_double_angle_l2100_210096

/-- Represents a triangle with side lengths (a, a+1, a+2) -/
structure ConsecutiveTriangle where
  a : ℕ
  a_pos : a > 0

/-- Calculates the cosine of an angle in a ConsecutiveTriangle using the law of cosines -/
def cos_angle (t : ConsecutiveTriangle) (side : Fin 3) : ℚ :=
  match side with
  | 0 => (t.a^2 + 6*t.a + 5) / (2*t.a^2 + 6*t.a + 4)
  | 1 => ((t.a + 1) * (t.a + 3)) / (2*t.a*(t.a + 2))
  | 2 => ((t.a - 1) * (t.a - 3)) / (2*t.a*(t.a + 1))

/-- Checks if one angle is twice another in a ConsecutiveTriangle -/
def has_double_angle (t : ConsecutiveTriangle) : Prop :=
  ∃ (i j : Fin 3), i ≠ j ∧ cos_angle t j = 2 * (cos_angle t i)^2 - 1

/-- The main theorem stating that there's a unique ConsecutiveTriangle with a double angle -/
theorem unique_consecutive_triangle_with_double_angle :
  ∃! (t : ConsecutiveTriangle), has_double_angle t :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_triangle_with_double_angle_l2100_210096


namespace NUMINAMATH_CALUDE_triangle_side_constraint_l2100_210044

theorem triangle_side_constraint (a : ℝ) : 
  (0 < a) → (0 < 2) → (0 < 6) → 
  (2 + 6 > a) → (6 + a > 2) → (2 + a > 6) → 
  (4 < a ∧ a < 8) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_constraint_l2100_210044


namespace NUMINAMATH_CALUDE_basketball_league_games_l2100_210013

theorem basketball_league_games (total_games : ℕ) : 
  (5 : ℚ) / 7 * total_games - (2 : ℚ) / 3 * total_games = 5 →
  (2 : ℚ) / 7 * total_games - (1 : ℚ) / 3 * total_games = 5 →
  total_games = 105 := by
  sorry

end NUMINAMATH_CALUDE_basketball_league_games_l2100_210013


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l2100_210091

theorem profit_percent_calculation (selling_price cost_price : ℝ) 
  (h : cost_price = 0.9 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = 100 / 9 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l2100_210091


namespace NUMINAMATH_CALUDE_roots_in_arithmetic_progression_l2100_210084

theorem roots_in_arithmetic_progression (m : ℝ) : 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ x : ℝ, x^4 - (3*m + 2)*x^2 + m^2 = 0 ↔ x = a ∨ x = b ∨ x = -b ∨ x = -a) ∧
    (b - (-b) = -b - (-a) ∧ a - b = b - (-b)))
  ↔ m = 6 ∨ m = -6/19 := by sorry

end NUMINAMATH_CALUDE_roots_in_arithmetic_progression_l2100_210084


namespace NUMINAMATH_CALUDE_percentage_problem_l2100_210083

theorem percentage_problem (p : ℝ) : p = 60 ↔ 180 * (1/3) - (p * 180 * (1/3) / 100) = 24 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2100_210083


namespace NUMINAMATH_CALUDE_five_pages_thirty_lines_each_l2100_210026

/-- Given a page capacity and number of pages, calculates the total lines of information. -/
def total_lines (lines_per_page : ℕ) (num_pages : ℕ) : ℕ :=
  lines_per_page * num_pages

/-- Theorem stating that 5 pages with 30 lines each result in 150 total lines. -/
theorem five_pages_thirty_lines_each :
  total_lines 30 5 = 150 := by
  sorry

end NUMINAMATH_CALUDE_five_pages_thirty_lines_each_l2100_210026


namespace NUMINAMATH_CALUDE_expected_white_balls_after_transfer_l2100_210006

/-- Represents a bag of colored balls -/
structure Bag where
  red : ℕ
  white : ℕ

/-- Represents the process of transferring balls between bags -/
def transfer (a b : Bag) : ℝ → Bag × Bag
  | p => sorry

/-- Calculates the expected number of white balls in the first bag after transfers -/
noncomputable def expected_white_balls (a b : Bag) : ℝ :=
  sorry

theorem expected_white_balls_after_transfer :
  let a : Bag := { red := 2, white := 3 }
  let b : Bag := { red := 3, white := 3 }
  expected_white_balls a b = 102 / 35 := by sorry

end NUMINAMATH_CALUDE_expected_white_balls_after_transfer_l2100_210006


namespace NUMINAMATH_CALUDE_exist_non_adjacent_non_sharing_l2100_210056

/-- A simple graph with 17 vertices where each vertex has degree 4. -/
structure Graph17Deg4 where
  vertices : Finset (Fin 17)
  edges : Finset (Fin 17 × Fin 17)
  vertex_count : vertices.card = 17
  degree_4 : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 4

/-- Two vertices are adjacent if there's an edge between them. -/
def adjacent (G : Graph17Deg4) (u v : Fin 17) : Prop :=
  (u, v) ∈ G.edges ∨ (v, u) ∈ G.edges

/-- Two vertices share a common neighbor if there exists a vertex adjacent to both. -/
def share_neighbor (G : Graph17Deg4) (u v : Fin 17) : Prop :=
  ∃ w : Fin 17, w ∈ G.vertices ∧ adjacent G u w ∧ adjacent G v w

/-- There exist two vertices that are neither adjacent nor share a common neighbor. -/
theorem exist_non_adjacent_non_sharing (G : Graph17Deg4) :
  ∃ u v : Fin 17, u ∈ G.vertices ∧ v ∈ G.vertices ∧ u ≠ v ∧
    ¬(adjacent G u v) ∧ ¬(share_neighbor G u v) := by
  sorry

end NUMINAMATH_CALUDE_exist_non_adjacent_non_sharing_l2100_210056


namespace NUMINAMATH_CALUDE_not_lucky_1994_l2100_210017

/-- Represents a date with month, day, and year --/
structure Date where
  month : Nat
  day : Nat
  year : Nat

/-- Checks if a given date is valid --/
def isValidDate (d : Date) : Prop :=
  d.month ≥ 1 ∧ d.month ≤ 12 ∧ d.day ≥ 1 ∧ d.day ≤ 31

/-- Checks if a year is lucky --/
def isLuckyYear (year : Nat) : Prop :=
  ∃ (d : Date), isValidDate d ∧ d.year = year ∧ d.month * d.day = year % 100

/-- Theorem stating that 1994 is not a lucky year --/
theorem not_lucky_1994 : ¬ isLuckyYear 1994 := by
  sorry


end NUMINAMATH_CALUDE_not_lucky_1994_l2100_210017


namespace NUMINAMATH_CALUDE_experienced_sailors_monthly_earnings_l2100_210074

theorem experienced_sailors_monthly_earnings :
  let total_sailors : ℕ := 17
  let inexperienced_sailors : ℕ := 5
  let experienced_sailors : ℕ := total_sailors - inexperienced_sailors
  let inexperienced_hourly_wage : ℚ := 10
  let wage_increase_ratio : ℚ := 1 / 5
  let experienced_hourly_wage : ℚ := inexperienced_hourly_wage * (1 + wage_increase_ratio)
  let weekly_hours : ℕ := 60
  let weeks_per_month : ℕ := 4
  
  experienced_sailors * experienced_hourly_wage * weekly_hours * weeks_per_month = 34560 :=
by sorry

end NUMINAMATH_CALUDE_experienced_sailors_monthly_earnings_l2100_210074


namespace NUMINAMATH_CALUDE_suresh_job_completion_time_l2100_210009

/-- The time it takes Suresh to complete the job alone -/
def suresh_time : ℝ := 15

/-- The time it takes Ashutosh to complete the job alone -/
def ashutosh_time : ℝ := 20

/-- The time Suresh works on the job -/
def suresh_work_time : ℝ := 9

/-- The time Ashutosh works to complete the remaining job -/
def ashutosh_completion_time : ℝ := 8

theorem suresh_job_completion_time : 
  (suresh_work_time / suresh_time) + (ashutosh_completion_time / ashutosh_time) = 1 ∧ 
  suresh_time = 15 := by
  sorry


end NUMINAMATH_CALUDE_suresh_job_completion_time_l2100_210009


namespace NUMINAMATH_CALUDE_octal_to_decimal_fraction_l2100_210065

theorem octal_to_decimal_fraction (c d : ℕ) : 
  (543 : ℕ) = 5 * 8^2 + 4 * 8^1 + 3 * 8^0 →
  (2 * 10 + c) * 10 + d = 5 * 8^2 + 4 * 8^1 + 3 * 8^0 →
  0 ≤ c ∧ c ≤ 9 →
  0 ≤ d ∧ d ≤ 9 →
  (c * d : ℚ) / 12 = 5 / 4 :=
by sorry

end NUMINAMATH_CALUDE_octal_to_decimal_fraction_l2100_210065


namespace NUMINAMATH_CALUDE_coefficient_a3_value_l2100_210025

/-- Given a polynomial expansion and sum of coefficients condition, prove a₃ = -5 -/
theorem coefficient_a3_value (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 + x) * (a - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0) →
  a₃ = -5 := by sorry

end NUMINAMATH_CALUDE_coefficient_a3_value_l2100_210025


namespace NUMINAMATH_CALUDE_student_speed_ratio_l2100_210001

theorem student_speed_ratio :
  ∀ (distance_A distance_B time_A time_B : ℚ),
    distance_A = (6 / 5) * distance_B →
    time_B = (10 / 11) * time_A →
    (distance_A / time_A) / (distance_B / time_B) = 12 / 11 :=
by
  sorry

end NUMINAMATH_CALUDE_student_speed_ratio_l2100_210001


namespace NUMINAMATH_CALUDE_joans_kittens_l2100_210038

theorem joans_kittens (initial_kittens given_away_kittens : ℕ) 
  (h1 : initial_kittens = 15)
  (h2 : given_away_kittens = 7) :
  initial_kittens - given_away_kittens = 8 := by
  sorry

end NUMINAMATH_CALUDE_joans_kittens_l2100_210038


namespace NUMINAMATH_CALUDE_base9_4318_equals_3176_l2100_210023

/-- Converts a base-9 digit to its decimal (base-10) value. -/
def base9ToDecimal (digit : ℕ) : ℕ := digit

/-- Converts a base-9 number to its decimal (base-10) equivalent. -/
def convertBase9ToDecimal (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ (digits.length - 1 - i))) 0

theorem base9_4318_equals_3176 :
  convertBase9ToDecimal [4, 3, 1, 8] = 3176 := by
  sorry

#eval convertBase9ToDecimal [4, 3, 1, 8]

end NUMINAMATH_CALUDE_base9_4318_equals_3176_l2100_210023


namespace NUMINAMATH_CALUDE_expression_factorization_l2100_210097

theorem expression_factorization (a : ℝ) :
  (6 * a^3 + 92 * a^2 - 7) - (-7 * a^3 + a^2 - 7) = 13 * a^2 * (a + 7) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2100_210097


namespace NUMINAMATH_CALUDE_specific_ellipse_area_l2100_210080

/-- An ellipse with given properties --/
structure Ellipse where
  major_axis_endpoint1 : ℝ × ℝ
  major_axis_endpoint2 : ℝ × ℝ
  point_on_ellipse : ℝ × ℝ

/-- The area of an ellipse --/
def ellipse_area (e : Ellipse) : ℝ := sorry

/-- Theorem stating the area of the specific ellipse --/
theorem specific_ellipse_area :
  let e : Ellipse := {
    major_axis_endpoint1 := (-8, 3),
    major_axis_endpoint2 := (12, 3),
    point_on_ellipse := (10, 6)
  }
  ellipse_area e = 50 * Real.pi := by sorry

end NUMINAMATH_CALUDE_specific_ellipse_area_l2100_210080


namespace NUMINAMATH_CALUDE_box_volume_increase_l2100_210064

theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5000)
  (surface_area : 2 * (l * w + w * h + h * l) = 1800)
  (edge_sum : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7048 := by
sorry

end NUMINAMATH_CALUDE_box_volume_increase_l2100_210064


namespace NUMINAMATH_CALUDE_quadratic_properties_l2100_210031

/-- Represents a quadratic function ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate that checks if x is in the solution set (2, 3) -/
def inSolutionSet (x : ℝ) : Prop := 2 < x ∧ x < 3

/-- The quadratic function is positive in the interval (2, 3) -/
def isPositiveInInterval (f : QuadraticFunction) : Prop :=
  ∀ x, inSolutionSet x → f.a * x^2 + f.b * x + f.c > 0

theorem quadratic_properties (f : QuadraticFunction) 
  (h : isPositiveInInterval f) : 
  f.a < 0 ∧ f.b * f.c < 0 ∧ f.b + f.c = f.a ∧ f.a - f.b + f.c ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2100_210031


namespace NUMINAMATH_CALUDE_range_of_a_l2100_210058

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + (2 + a) * x + 1 = 0) → 
  a ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2100_210058


namespace NUMINAMATH_CALUDE_divisibility_of_power_difference_l2100_210059

theorem divisibility_of_power_difference (a b c k q : ℕ) (n : ℤ) :
  a ≥ 1 →
  b ≥ 1 →
  c ≥ 1 →
  k ≥ 1 →
  n = a^(c^k) - b^(c^k) →
  (∃ (p : List ℕ), (∀ x ∈ p, Nat.Prime x) ∧ p.length ≥ q ∧ (∀ x ∈ p, c % x = 0)) →
  ∃ (r : List ℕ), (∀ x ∈ r, Nat.Prime x) ∧ r.length ≥ q * k ∧ (∀ x ∈ r, n % x = 0) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_power_difference_l2100_210059


namespace NUMINAMATH_CALUDE_g_three_properties_l2100_210068

/-- A function satisfying the given condition for all real x and y -/
def special_function (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g x * g y - g (x * y) = x + y + 1

/-- The theorem stating the properties of g(3) -/
theorem g_three_properties (g : ℝ → ℝ) (h : special_function g) :
  (∃ a b : ℝ, (∀ x : ℝ, g 3 = x → (x = a ∨ x = b)) ∧ a + b = 0) :=
sorry

end NUMINAMATH_CALUDE_g_three_properties_l2100_210068


namespace NUMINAMATH_CALUDE_max_x_over_y_l2100_210089

theorem max_x_over_y (x y a b : ℝ) (h1 : x ≥ y) (h2 : y > 0)
  (h3 : 0 ≤ a) (h4 : a ≤ x) (h5 : 0 ≤ b) (h6 : b ≤ y)
  (h7 : (x - a)^2 + (y - b)^2 = x^2 + b^2)
  (h8 : x^2 + b^2 = y^2 + a^2) :
  ∃ (x' y' : ℝ), x' ≥ y' ∧ y' > 0 ∧
  ∃ (a' b' : ℝ), 0 ≤ a' ∧ a' ≤ x' ∧ 0 ≤ b' ∧ b' ≤ y' ∧
  (x' - a')^2 + (y' - b')^2 = x'^2 + b'^2 ∧ x'^2 + b'^2 = y'^2 + a'^2 ∧
  x' / y' = 2 * Real.sqrt 3 / 3 ∧
  ∀ (x'' y'' : ℝ), x'' ≥ y'' → y'' > 0 →
  ∃ (a'' b'' : ℝ), 0 ≤ a'' ∧ a'' ≤ x'' ∧ 0 ≤ b'' ∧ b'' ≤ y'' ∧
  (x'' - a'')^2 + (y'' - b'')^2 = x''^2 + b''^2 ∧ x''^2 + b''^2 = y''^2 + a''^2 →
  x'' / y'' ≤ 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_x_over_y_l2100_210089


namespace NUMINAMATH_CALUDE_fixed_OC_length_l2100_210051

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point P inside the circle
def P (c : Circle) : ℝ × ℝ := sorry

-- Distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the chord AB
def chord (c : Circle) (p : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Define point C
def pointC (c : Circle) (p : ℝ × ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem fixed_OC_length (c : Circle) : 
  let o := c.center
  let r := c.radius
  let p := P c
  let d := distance o p
  let oc_length := distance o (pointC c p)
  oc_length = Real.sqrt (2 * r^2 - d^2) := by sorry

end NUMINAMATH_CALUDE_fixed_OC_length_l2100_210051


namespace NUMINAMATH_CALUDE_binomial_12_choose_6_l2100_210078

theorem binomial_12_choose_6 : Nat.choose 12 6 = 924 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_6_l2100_210078


namespace NUMINAMATH_CALUDE_division_problem_l2100_210081

theorem division_problem : ∃ x : ℝ, (x / 1.33 = 48) ↔ (x = 63.84) := by sorry

end NUMINAMATH_CALUDE_division_problem_l2100_210081


namespace NUMINAMATH_CALUDE_square_side_properties_l2100_210099

theorem square_side_properties (a : ℝ) (h : a > 0) (area_eq : a^2 = 10) :
  a = Real.sqrt 10 ∧ a^2 - 10 = 0 ∧ 3 < a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_properties_l2100_210099


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l2100_210040

theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter : ℝ := outer_cube_edge
  let inner_cube_diagonal : ℝ := sphere_diameter
  let inner_cube_edge : ℝ := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_volume : ℝ := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l2100_210040


namespace NUMINAMATH_CALUDE_polynomial_factor_l2100_210079

-- Define the polynomial
def f (b : ℝ) (x : ℝ) : ℝ := 3 * x^3 + b * x + 12

-- Define the quadratic factor
def g (p : ℝ) (x : ℝ) : ℝ := x^2 + p * x + 2

-- Theorem statement
theorem polynomial_factor (b : ℝ) :
  (∃ p : ℝ, ∀ x : ℝ, ∃ k : ℝ, f b x = g p x * (3 * x + 6)) →
  b = -6 := by sorry

end NUMINAMATH_CALUDE_polynomial_factor_l2100_210079


namespace NUMINAMATH_CALUDE_max_a_value_l2100_210032

theorem max_a_value (a b c d : ℕ+) 
  (h1 : a < 3 * b)
  (h2 : b < 3 * c)
  (h3 : c < 4 * d)
  (h4 : b + d = 200) :
  a ≤ 449 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 449 ∧ 
    a' < 3 * b' ∧ 
    b' < 3 * c' ∧ 
    c' < 4 * d' ∧ 
    b' + d' = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l2100_210032


namespace NUMINAMATH_CALUDE_geometric_mean_of_1_and_9_l2100_210033

def geometric_mean (a b : ℝ) : Set ℝ :=
  {x | x ^ 2 = a * b}

theorem geometric_mean_of_1_and_9 :
  geometric_mean 1 9 = {3, -3} := by sorry

end NUMINAMATH_CALUDE_geometric_mean_of_1_and_9_l2100_210033


namespace NUMINAMATH_CALUDE_sphere_surface_area_with_inscribed_parallelepiped_l2100_210090

theorem sphere_surface_area_with_inscribed_parallelepiped (a b c : ℝ) (S : ℝ) :
  a = 1 →
  b = 2 →
  c = 2 →
  S = 4 * Real.pi * ((a^2 + b^2 + c^2) / 4) →
  S = 9 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_with_inscribed_parallelepiped_l2100_210090


namespace NUMINAMATH_CALUDE_root_and_c_value_l2100_210008

theorem root_and_c_value (x : ℝ) (c : ℝ) : 
  (2 + Real.sqrt 3)^2 - 4*(2 + Real.sqrt 3) + c = 0 →
  (∃ y : ℝ, y ≠ 2 + Real.sqrt 3 ∧ y^2 - 4*y + c = 0 ∧ y = 2 - Real.sqrt 3) ∧
  c = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_and_c_value_l2100_210008


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l2100_210034

theorem mod_equivalence_unique_solution : 
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -3402 [ZMOD 10] ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l2100_210034


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2100_210094

def polynomial (x : ℤ) : ℤ := x^3 - 2*x^2 + 3*x - 17

def is_root (x : ℤ) : Prop := polynomial x = 0

theorem integer_roots_of_polynomial :
  {x : ℤ | is_root x} = {-17, -1, 1, 17} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2100_210094


namespace NUMINAMATH_CALUDE_abc_inequality_l2100_210072

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 / (1 + a^2) + b^2 / (1 + b^2) + c^2 / (1 + c^2) = 1) :
  a * b * c ≤ Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2100_210072


namespace NUMINAMATH_CALUDE_no_distinct_integers_divisibility_l2100_210007

theorem no_distinct_integers_divisibility : ¬∃ (a : Fin 2001 → ℕ+), 
  (∀ (i j : Fin 2001), i ≠ j → (a i).val * (a j).val ∣ 
    ((a i).val ^ 2000 - (a i).val ^ 1000 + 1) * 
    ((a j).val ^ 2000 - (a j).val ^ 1000 + 1)) ∧ 
  (∀ (i j : Fin 2001), i ≠ j → a i ≠ a j) :=
by sorry

end NUMINAMATH_CALUDE_no_distinct_integers_divisibility_l2100_210007


namespace NUMINAMATH_CALUDE_law_firm_associates_tenure_l2100_210000

/-- 
Given a law firm where:
- 30% of associates are second-year associates
- 60% of associates are not first-year associates

This theorem proves that 30% of associates have been at the firm for more than two years.
-/
theorem law_firm_associates_tenure (total : ℝ) (second_year : ℝ) (not_first_year : ℝ) 
  (h1 : second_year = 0.3 * total) 
  (h2 : not_first_year = 0.6 * total) : 
  total - (second_year + (total - not_first_year)) = 0.3 * total := by
  sorry

end NUMINAMATH_CALUDE_law_firm_associates_tenure_l2100_210000


namespace NUMINAMATH_CALUDE_eliminate_x_l2100_210062

/-- Represents a linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The system of equations -/
def system : (LinearEquation × LinearEquation) :=
  ({ a := 6, b := 2, c := 4 },
   { a := 3, b := -3, c := -6 })

/-- Operation that combines two equations -/
def combineEquations (eq1 eq2 : LinearEquation) (k : ℝ) : LinearEquation :=
  { a := eq1.a - k * eq2.a,
    b := eq1.b - k * eq2.b,
    c := eq1.c - k * eq2.c }

/-- Theorem stating that the specified operation eliminates x -/
theorem eliminate_x :
  let (eq1, eq2) := system
  let result := combineEquations eq1 eq2 2
  result.a = 0 := by sorry

end NUMINAMATH_CALUDE_eliminate_x_l2100_210062


namespace NUMINAMATH_CALUDE_geometric_series_sum_8_terms_l2100_210066

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_8_terms :
  geometric_series_sum (1/4) (1/4) 8 = 65535/196608 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_8_terms_l2100_210066


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2100_210092

-- Define the inequality
def inequality (x : ℝ) : Prop := (1 - x) * (x - 3) < 0

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | x < 1 ∨ x > 3}

-- Theorem stating that the solution set is correct
theorem inequality_solution_set : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2100_210092


namespace NUMINAMATH_CALUDE_circle_points_l2100_210043

theorem circle_points (π : ℝ) (h : π > 0) : 
  let radii : List ℝ := [1.5, 2, 3.5, 4.5, 5.5]
  let circumference (r : ℝ) := 2 * π * r
  let area (r : ℝ) := π * r^2
  let points := radii.map (λ r => (circumference r, area r))
  points = [(3*π, 2.25*π), (4*π, 4*π), (7*π, 12.25*π), (9*π, 20.25*π), (11*π, 30.25*π)] := by
  sorry

end NUMINAMATH_CALUDE_circle_points_l2100_210043


namespace NUMINAMATH_CALUDE_speed_difference_l2100_210049

/-- The speed difference between a cyclist and a car -/
theorem speed_difference (cyclist_distance car_distance : ℝ) (time : ℝ) 
  (h_cyclist : cyclist_distance = 88)
  (h_car : car_distance = 48)
  (h_time : time = 8)
  (h_time_pos : time > 0) :
  cyclist_distance / time - car_distance / time = 5 := by
sorry

end NUMINAMATH_CALUDE_speed_difference_l2100_210049


namespace NUMINAMATH_CALUDE_golden_ratio_trigonometric_identity_l2100_210046

theorem golden_ratio_trigonometric_identity :
  let m := 2 * Real.sin (18 * π / 180)
  (Real.sin (42 * π / 180) + m) / Real.cos (42 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_trigonometric_identity_l2100_210046


namespace NUMINAMATH_CALUDE_job_productivity_solution_l2100_210069

/-- Represents the productivity of workers on a job -/
structure JobProductivity where
  workers : ℕ
  hours_per_day : ℕ
  days_to_complete : ℕ

/-- The job productivity satisfies the given conditions -/
def satisfies_conditions (jp : JobProductivity) : Prop :=
  ∃ (t : ℝ),
    -- Condition 1: Initial setup
    jp.workers * jp.hours_per_day * jp.days_to_complete * t = jp.workers * jp.hours_per_day * 14 * t ∧
    -- Condition 2: 4 more workers, 1 hour longer, 10 days
    jp.workers * jp.hours_per_day * 14 * t = (jp.workers + 4) * (jp.hours_per_day + 1) * 10 * t ∧
    -- Condition 3: 10 more workers, 2 hours longer, 7 days
    jp.workers * jp.hours_per_day * 14 * t = (jp.workers + 10) * (jp.hours_per_day + 2) * 7 * t

/-- The theorem to be proved -/
theorem job_productivity_solution :
  ∃ (jp : JobProductivity),
    satisfies_conditions jp ∧ jp.workers = 20 ∧ jp.hours_per_day = 6 :=
sorry

end NUMINAMATH_CALUDE_job_productivity_solution_l2100_210069


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l2100_210014

/-- A regular decagon is a 10-sided polygon with all sides and angles equal -/
def RegularDecagon : Type := Unit

/-- The number of diagonals in a regular decagon, excluding the sides -/
def num_diagonals : ℕ := 35

/-- The number of ways to choose 2 diagonals from the total number of diagonals -/
def num_diagonal_pairs : ℕ := 595

/-- The number of sets of intersecting diagonals in a regular decagon -/
def num_intersecting_diagonals : ℕ := 210

/-- The probability that two randomly chosen diagonals in a regular decagon intersect inside the decagon -/
theorem decagon_diagonal_intersection_probability (d : RegularDecagon) : 
  (num_intersecting_diagonals : ℚ) / num_diagonal_pairs = 42 / 119 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l2100_210014


namespace NUMINAMATH_CALUDE_sin_810_degrees_l2100_210085

theorem sin_810_degrees : Real.sin (810 * π / 180) = 1 := by sorry

end NUMINAMATH_CALUDE_sin_810_degrees_l2100_210085


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2100_210057

variable (i : ℂ)
variable (z : ℂ)

theorem complex_equation_solution (hi : i * i = -1) (hz : (1 + i) / z = 1 - i) : z = i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2100_210057


namespace NUMINAMATH_CALUDE_total_revenue_is_1168_l2100_210035

/-- Calculates the total revenue from apple and orange sales given the following conditions:
  * 50 boxes of apples and 30 boxes of oranges on Saturday
  * 25 boxes of apples and 15 boxes of oranges on Sunday
  * 10 apples in each apple box
  * 8 oranges in each orange box
  * Each apple sold for $1.20
  * Each orange sold for $0.80
  * Total of 720 apples and 380 oranges sold on Saturday and Sunday -/
def total_revenue : ℝ :=
  let apple_boxes_saturday : ℕ := 50
  let orange_boxes_saturday : ℕ := 30
  let apple_boxes_sunday : ℕ := 25
  let orange_boxes_sunday : ℕ := 15
  let apples_per_box : ℕ := 10
  let oranges_per_box : ℕ := 8
  let apple_price : ℝ := 1.20
  let orange_price : ℝ := 0.80
  let total_apples_sold : ℕ := 720
  let total_oranges_sold : ℕ := 380
  let apple_revenue : ℝ := (total_apples_sold : ℝ) * apple_price
  let orange_revenue : ℝ := (total_oranges_sold : ℝ) * orange_price
  apple_revenue + orange_revenue

/-- Theorem stating that the total revenue is $1168 -/
theorem total_revenue_is_1168 : total_revenue = 1168 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_is_1168_l2100_210035


namespace NUMINAMATH_CALUDE_basketball_winning_percentage_l2100_210024

theorem basketball_winning_percentage (total_games season_games remaining_games first_wins : ℕ)
  (h1 : total_games = season_games + remaining_games)
  (h2 : season_games = 75)
  (h3 : remaining_games = 45)
  (h4 : first_wins = 60)
  (h5 : total_games = 120) :
  (∃ x : ℕ, x = 36 ∧ (first_wins + x : ℚ) / total_games = 4/5) :=
sorry

end NUMINAMATH_CALUDE_basketball_winning_percentage_l2100_210024


namespace NUMINAMATH_CALUDE_function_inequality_l2100_210073

-- Define the function f on the non-zero real numbers
variable (f : ℝ → ℝ)

-- Define the condition that f is twice differentiable
variable (hf : TwiceDifferentiable ℝ f)

-- Define the condition that f''(x) - f(x)/x > 0 for all non-zero x
variable (h : ∀ x : ℝ, x ≠ 0 → (deriv^[2] f) x - f x / x > 0)

-- State the theorem
theorem function_inequality : 3 * f 4 > 4 * f 3 := by sorry

end NUMINAMATH_CALUDE_function_inequality_l2100_210073


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2100_210098

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 12 + 23 + 17 + y) / 5 = 15 → y = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2100_210098


namespace NUMINAMATH_CALUDE_binomial_eight_five_l2100_210052

theorem binomial_eight_five : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_eight_five_l2100_210052


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2100_210075

theorem expression_simplification_and_evaluation :
  ∀ x y : ℝ, (x - 2)^2 + |y + 1| = 0 →
  3 * x^2 * y - (2 * x^2 * y - 3 * (2 * x * y - x^2 * y) + 5 * x * y) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2100_210075


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l2100_210012

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l2100_210012


namespace NUMINAMATH_CALUDE_odot_equation_solution_l2100_210041

-- Define the operation ⊙
noncomputable def odot (a b : ℝ) : ℝ :=
  a^2 + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem odot_equation_solution (g : ℝ) (h1 : g ≥ 0) (h2 : odot 4 g = 20) : g = 12 := by
  sorry

end NUMINAMATH_CALUDE_odot_equation_solution_l2100_210041


namespace NUMINAMATH_CALUDE_parabola_reflection_theorem_l2100_210086

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line passing through two points -/
def Line (P Q : Point) : Set Point :=
  {R : Point | ∃ t : ℝ, R.x = P.x + t * (Q.x - P.x) ∧ R.y = P.y + t * (Q.y - P.y)}

/-- Check if a point is on the parabola -/
def onParabola (par : Parabola) (P : Point) : Prop :=
  P.y^2 = 2 * par.p * P.x

/-- Check if a point is on the axis of symmetry -/
def onAxisOfSymmetry (P : Point) : Prop :=
  P.y = 0

/-- Reflection of a point about y-axis -/
def reflectAboutYAxis (P : Point) : Point :=
  ⟨-P.x, P.y⟩

/-- Angle between three points -/
noncomputable def angle (A B C : Point) : ℝ := sorry

/-- Main theorem -/
theorem parabola_reflection_theorem (par : Parabola) (A : Point)
  (h_A_on_axis : onAxisOfSymmetry A)
  (h_A_inside : onParabola par A → False) :
  let B := reflectAboutYAxis A
  (∀ P Q : Point, onParabola par P → onParabola par Q →
    P.x * Q.x > 0 → P ∈ Line A Q → Q ∈ Line A P →
    angle P B A = angle Q B A) ∧
  (∀ P Q : Point, onParabola par P → onParabola par Q →
    P.x * Q.x > 0 → P ∈ Line B Q → Q ∈ Line B P →
    angle P A B + angle Q A B = 180) :=
by sorry

end NUMINAMATH_CALUDE_parabola_reflection_theorem_l2100_210086


namespace NUMINAMATH_CALUDE_smallest_number_of_ducks_l2100_210050

/-- Represents the number of birds in a flock for each type --/
structure FlockSize where
  duck : ℕ
  crane : ℕ
  heron : ℕ

/-- Represents the number of flocks for each type of bird --/
structure FlockCount where
  duck : ℕ
  crane : ℕ
  heron : ℕ

/-- The main theorem stating the smallest number of ducks observed --/
theorem smallest_number_of_ducks 
  (flock_size : FlockSize)
  (flock_count : FlockCount)
  (h1 : flock_size.duck = 13)
  (h2 : flock_size.crane = 17)
  (h3 : flock_size.heron = 11)
  (h4 : flock_size.duck * flock_count.duck = flock_size.crane * flock_count.crane)
  (h5 : 6 * (flock_size.duck * flock_count.duck) = 5 * (flock_size.heron * flock_count.heron))
  (h6 : 3 * (flock_size.crane * flock_count.crane) = 8 * (flock_size.heron * flock_count.heron))
  (h7 : ∀ c : FlockCount, 
    (c.duck < flock_count.duck ∨ c.crane < flock_count.crane ∨ c.heron < flock_count.heron) →
    (flock_size.duck * c.duck ≠ flock_size.crane * c.crane ∨
     6 * (flock_size.duck * c.duck) ≠ 5 * (flock_size.heron * c.heron) ∨
     3 * (flock_size.crane * c.crane) ≠ 8 * (flock_size.heron * c.heron))) :
  flock_size.duck * flock_count.duck = 520 := by
  sorry


end NUMINAMATH_CALUDE_smallest_number_of_ducks_l2100_210050


namespace NUMINAMATH_CALUDE_gerald_chores_solution_l2100_210002

/-- Represents the problem of calculating the number of chores Gerald needs to do per month --/
def gerald_chores_problem (monthly_expense : ℕ) (season_length : ℕ) (chore_price : ℕ) (months_in_year : ℕ) : Prop :=
  let total_expense := monthly_expense * season_length
  let off_season_months := months_in_year - season_length
  let monthly_savings_needed := total_expense / off_season_months
  monthly_savings_needed / chore_price = 5

/-- Theorem stating that Gerald needs to average 5 chores per month to save for his baseball supplies --/
theorem gerald_chores_solution :
  gerald_chores_problem 100 4 10 12 := by
  sorry

#check gerald_chores_solution

end NUMINAMATH_CALUDE_gerald_chores_solution_l2100_210002


namespace NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l2100_210076

/-- A truncated cone with a tangent sphere -/
structure TruncatedConeWithSphere where
  bottom_radius : ℝ
  top_radius : ℝ
  sphere_radius : ℝ
  tangent_to_top : Bool
  tangent_to_bottom : Bool
  tangent_to_lateral : Bool

/-- The theorem stating the radius of the sphere in a specific truncated cone configuration -/
theorem sphere_radius_in_truncated_cone
  (cone : TruncatedConeWithSphere)
  (h1 : cone.bottom_radius = 12)
  (h2 : cone.top_radius = 3)
  (h3 : cone.tangent_to_top = true)
  (h4 : cone.tangent_to_bottom = true)
  (h5 : cone.tangent_to_lateral = true) :
  cone.sphere_radius = 6 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l2100_210076


namespace NUMINAMATH_CALUDE_tangent_line_sum_l2100_210067

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_line_sum (h : ∀ y, y = f 2 → y = 2 * 2 + 3) : f 2 + (deriv f) 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l2100_210067


namespace NUMINAMATH_CALUDE_speed_ratio_l2100_210082

/-- Two perpendicular lines intersecting at O with points A and B moving along them -/
structure PointMovement where
  O : ℝ × ℝ
  speedA : ℝ
  speedB : ℝ
  initialDistB : ℝ
  time1 : ℝ
  time2 : ℝ

/-- The conditions of the problem -/
def problem_conditions (pm : PointMovement) : Prop :=
  pm.O = (0, 0) ∧
  pm.speedA > 0 ∧
  pm.speedB > 0 ∧
  pm.initialDistB = 500 ∧
  pm.time1 = 2 ∧
  pm.time2 = 10 ∧
  pm.speedA * pm.time1 = pm.initialDistB - pm.speedB * pm.time1 ∧
  pm.speedA * pm.time2 = pm.speedB * pm.time2 - pm.initialDistB

/-- The theorem to be proved -/
theorem speed_ratio (pm : PointMovement) :
  problem_conditions pm → pm.speedA / pm.speedB = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_speed_ratio_l2100_210082


namespace NUMINAMATH_CALUDE_unique_pin_l2100_210011

def is_valid_pin (pin : Nat) : Prop :=
  pin ≥ 1000 ∧ pin < 10000 ∧
  let first_digit := pin / 1000
  let last_three_digits := pin % 1000
  10 * last_three_digits + first_digit = 3 * pin - 6

theorem unique_pin : ∃! pin, is_valid_pin pin ∧ pin = 2856 := by
  sorry

end NUMINAMATH_CALUDE_unique_pin_l2100_210011


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2100_210003

theorem simplify_trig_expression :
  (Real.sin (30 * π / 180) + Real.sin (60 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (60 * π / 180)) =
  Real.tan (45 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2100_210003


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2100_210015

theorem geometric_sequence_common_ratio 
  (a₁ : ℝ) (r : ℝ) 
  (h₁ : a₁ ≠ 0) 
  (h₂ : r > 0) 
  (h₃ : ∀ n m : ℕ, n ≠ m → a₁ * r^n ≠ a₁ * r^m) 
  (h₄ : ∃ d : ℝ, a₁ * r^3 - a₁ * r = d ∧ a₁ * r^4 - a₁ * r^3 = d) : 
  r = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2100_210015


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_t_value_l2100_210022

theorem sqrt_equality_implies_t_value :
  ∀ t : ℝ, (Real.sqrt (3 * Real.sqrt (t - 3)) = (8 - t) ^ (1/4)) → t = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_t_value_l2100_210022


namespace NUMINAMATH_CALUDE_farmer_plots_allocation_l2100_210036

theorem farmer_plots_allocation (x y : ℕ) (h : x ≠ y) : ∃ (a b : ℕ), a^2 + b^2 = 2 * (x^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_farmer_plots_allocation_l2100_210036


namespace NUMINAMATH_CALUDE_parallel_segments_length_l2100_210071

/-- Represents a line segment with a length -/
structure Segment where
  length : ℝ

/-- Represents three parallel line segments -/
structure ParallelSegments where
  ef : Segment
  gh : Segment
  ij : Segment

/-- Theorem: Given three parallel line segments EF, GH, and IJ,
    where IJ = 120 cm and EF = 180 cm, the length of GH is 72 cm -/
theorem parallel_segments_length 
  (segments : ParallelSegments) 
  (h1 : segments.ij.length = 120) 
  (h2 : segments.ef.length = 180) : 
  segments.gh.length = 72 := by
  sorry

end NUMINAMATH_CALUDE_parallel_segments_length_l2100_210071


namespace NUMINAMATH_CALUDE_simple_interest_rate_equivalence_l2100_210004

/-- The simple interest rate that yields $10 less than a 10% interest rate compounded semi-annually for a $5,000 investment over 1 year is 10.05% -/
theorem simple_interest_rate_equivalence :
  let initial_investment : ℝ := 5000
  let compound_rate : ℝ := 0.10
  let compounding_frequency : ℕ := 2
  let time : ℝ := 1
  let compound_interest := initial_investment * (1 + compound_rate / compounding_frequency) ^ (compounding_frequency * time) - initial_investment
  let simple_interest := compound_interest - 10
  let simple_rate := simple_interest / (initial_investment * time)
  simple_rate = 0.1005 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_equivalence_l2100_210004


namespace NUMINAMATH_CALUDE_locus_proof_methods_correctness_l2100_210048

-- Define a type for points in a geometric space
variable {Point : Type}

-- Define a predicate for points satisfying the locus conditions
variable (satisfiesConditions : Point → Prop)

-- Define a predicate for points being on the locus
variable (onLocus : Point → Prop)

-- Define the correctness of each statement
def statementA : Prop :=
  (∀ p : Point, onLocus p → satisfiesConditions p) ∧
  (∀ p : Point, ¬onLocus p → ¬satisfiesConditions p)

def statementB : Prop :=
  (∀ p : Point, ¬satisfiesConditions p → onLocus p) ∧
  (∀ p : Point, onLocus p → satisfiesConditions p)

def statementC : Prop :=
  (∀ p : Point, satisfiesConditions p → onLocus p) ∧
  (∀ p : Point, ¬onLocus p → satisfiesConditions p)

def statementD : Prop :=
  (∀ p : Point, ¬onLocus p → ¬satisfiesConditions p) ∧
  (∀ p : Point, ¬satisfiesConditions p → ¬onLocus p)

def statementE : Prop :=
  (∀ p : Point, satisfiesConditions p → onLocus p) ∧
  (∀ p : Point, ¬satisfiesConditions p → ¬onLocus p)

-- Theorem stating which methods are correct and which are incorrect
theorem locus_proof_methods_correctness :
  (statementA satisfiesConditions onLocus) ∧
  (¬statementB satisfiesConditions onLocus) ∧
  (¬statementC satisfiesConditions onLocus) ∧
  (statementD satisfiesConditions onLocus) ∧
  (statementE satisfiesConditions onLocus) :=
sorry

end NUMINAMATH_CALUDE_locus_proof_methods_correctness_l2100_210048


namespace NUMINAMATH_CALUDE_polynomial_multiplication_identity_l2100_210077

theorem polynomial_multiplication_identity (x y : ℝ) :
  (3 * x^4 - 7 * y^3) * (9 * x^8 + 21 * x^4 * y^3 + 49 * y^6) = 27 * x^12 - 343 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_identity_l2100_210077


namespace NUMINAMATH_CALUDE_linear_function_decreases_iff_positive_slope_l2100_210042

/-- A linear function represented by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- The value of a linear function at a given x -/
def LinearFunction.value (f : LinearFunction) (x : ℝ) : ℝ :=
  f.slope * x + f.intercept

/-- A linear function decreases as x decreases iff its slope is positive -/
theorem linear_function_decreases_iff_positive_slope (f : LinearFunction) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f.value x₁ < f.value x₂) ↔ f.slope > 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_decreases_iff_positive_slope_l2100_210042


namespace NUMINAMATH_CALUDE_boat_round_trip_time_l2100_210028

/-- Calculates the total time for a round trip boat journey given the boat's speed in standing water, 
    the stream's speed, and the distance to the destination. -/
theorem boat_round_trip_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed = 8) 
  (h2 : stream_speed = 6) 
  (h3 : distance = 210) : 
  distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed) = 120 := by
  sorry

#check boat_round_trip_time

end NUMINAMATH_CALUDE_boat_round_trip_time_l2100_210028


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2100_210053

theorem polynomial_remainder_theorem (x : ℝ) : 
  (4 * x^3 - 10 * x^2 + 15 * x - 17) % (4 * x - 8) = 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2100_210053


namespace NUMINAMATH_CALUDE_min_product_of_three_distinct_l2100_210095

def S : Finset Int := {-10, -5, -3, 0, 4, 6, 9}

theorem min_product_of_three_distinct (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∀ x y z, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z → 
  a * b * c ≤ x * y * z :=
by sorry

end NUMINAMATH_CALUDE_min_product_of_three_distinct_l2100_210095


namespace NUMINAMATH_CALUDE_basketball_scores_theorem_l2100_210088

/-- Represents the scores of a basketball player in a series of games -/
structure BasketballScores where
  total_games : Nat
  sixth_game_score : Nat
  seventh_game_score : Nat
  eighth_game_score : Nat
  ninth_game_score : Nat
  first_five_avg : ℝ
  first_nine_avg : ℝ

/-- Theorem about basketball scores -/
theorem basketball_scores_theorem (scores : BasketballScores) 
  (h1 : scores.total_games = 10)
  (h2 : scores.sixth_game_score = 22)
  (h3 : scores.seventh_game_score = 15)
  (h4 : scores.eighth_game_score = 12)
  (h5 : scores.ninth_game_score = 19) :
  (scores.first_nine_avg = (5 * scores.first_five_avg + 68) / 9) ∧
  (∃ (min_y : ℝ), min_y = 12 ∧ ∀ y, y = scores.first_nine_avg → y ≥ min_y) ∧
  (scores.first_nine_avg > scores.first_five_avg → 
    ∃ (max_score : ℕ), max_score = 84 ∧ 
    ∀ s, s = (5 : ℝ) * scores.first_five_avg → s ≤ max_score) := by
  sorry

end NUMINAMATH_CALUDE_basketball_scores_theorem_l2100_210088


namespace NUMINAMATH_CALUDE_sqrt_five_squared_l2100_210021

theorem sqrt_five_squared : (Real.sqrt 5) ^ 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_squared_l2100_210021


namespace NUMINAMATH_CALUDE_tim_pencil_count_l2100_210055

/-- Given that Tyrah has six times as many pencils as Sarah, Tim has eight times as many pencils as Sarah, and Tyrah has 12 pencils, prove that Tim has 16 pencils. -/
theorem tim_pencil_count (sarah_pencils : ℕ) 
  (h1 : 6 * sarah_pencils = 12)  -- Tyrah has six times as many pencils as Sarah and has 12 pencils
  (h2 : 8 * sarah_pencils = tim_pencils) : -- Tim has eight times as many pencils as Sarah
  tim_pencils = 16 := by
  sorry

end NUMINAMATH_CALUDE_tim_pencil_count_l2100_210055


namespace NUMINAMATH_CALUDE_jeffrey_steps_l2100_210027

-- Define Jeffrey's walking pattern
def forward_steps : ℕ := 3
def backward_steps : ℕ := 2

-- Define the distance between house and mailbox
def distance : ℕ := 66

-- Define the function to calculate total steps
def total_steps (fwd : ℕ) (bwd : ℕ) (dist : ℕ) : ℕ :=
  dist * (fwd + bwd)

-- Theorem statement
theorem jeffrey_steps :
  total_steps forward_steps backward_steps distance = 330 := by
  sorry

end NUMINAMATH_CALUDE_jeffrey_steps_l2100_210027


namespace NUMINAMATH_CALUDE_function_inequality_l2100_210019

/-- Given a real-valued function f(x) = e^x / x, prove that for all real x ≠ 0, 
    1 / (x * f(x)) > 1 - x -/
theorem function_inequality (x : ℝ) (hx : x ≠ 0) : 
  let f : ℝ → ℝ := fun x => Real.exp x / x
  1 / (x * f x) > 1 - x := by sorry

end NUMINAMATH_CALUDE_function_inequality_l2100_210019


namespace NUMINAMATH_CALUDE_stating_chess_tournament_players_l2100_210060

/-- The number of players in the chess tournament. -/
def num_players : ℕ := 11

/-- The total number of games played in the tournament. -/
def total_games : ℕ := 132

/-- 
Theorem stating that the number of players in the chess tournament is 11,
given the conditions of the problem.
-/
theorem chess_tournament_players :
  (∀ n : ℕ, n > 0 → 2 * n * (n - 1) = total_games) → num_players = 11 :=
by sorry

end NUMINAMATH_CALUDE_stating_chess_tournament_players_l2100_210060


namespace NUMINAMATH_CALUDE_axiom_1_l2100_210070

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the membership relations
variable (pointOnLine : Point → Line → Prop)
variable (pointInPlane : Point → Plane → Prop)
variable (lineInPlane : Line → Plane → Prop)

-- State the theorem
theorem axiom_1 (A B : Point) (l : Line) (α : Plane) :
  pointOnLine A l → pointOnLine B l → pointInPlane A α → pointInPlane B α →
  lineInPlane l α := by
  sorry

end NUMINAMATH_CALUDE_axiom_1_l2100_210070


namespace NUMINAMATH_CALUDE_max_product_sum_l2100_210010

theorem max_product_sum (a b c d : ℕ) : 
  a ∈ ({2, 3, 4, 5} : Set ℕ) → 
  b ∈ ({2, 3, 4, 5} : Set ℕ) → 
  c ∈ ({2, 3, 4, 5} : Set ℕ) → 
  d ∈ ({2, 3, 4, 5} : Set ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d → 
  (a * b + b * c + c * d + d * a) ≤ 49 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_l2100_210010


namespace NUMINAMATH_CALUDE_new_average_income_l2100_210039

/-- Given a family's initial average income, number of earning members, and the income of a deceased member,
    calculate the new average income after the member's death. -/
theorem new_average_income
  (initial_average : ℚ)
  (initial_members : ℕ)
  (deceased_income : ℚ)
  (new_members : ℕ)
  (h1 : initial_average = 735)
  (h2 : initial_members = 4)
  (h3 : deceased_income = 1170)
  (h4 : new_members = initial_members - 1) :
  let initial_total := initial_average * initial_members
  let new_total := initial_total - deceased_income
  new_total / new_members = 590 := by
sorry


end NUMINAMATH_CALUDE_new_average_income_l2100_210039


namespace NUMINAMATH_CALUDE_min_f_at_75_l2100_210087

/-- The function representing the total time needed for production -/
def f (x : ℕ) : ℚ := 9000 / x + 1000 / (100 - x)

/-- The theorem stating that f(x) reaches its minimum when x = 75 -/
theorem min_f_at_75 :
  ∀ x : ℕ, 1 ≤ x → x ≤ 99 → f 75 ≤ f x :=
sorry

end NUMINAMATH_CALUDE_min_f_at_75_l2100_210087


namespace NUMINAMATH_CALUDE_tangent_implies_m_six_or_twelve_l2100_210005

/-- An ellipse defined by x^2 + 9y^2 = 9 -/
def ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

/-- A hyperbola defined by x^2 - m(y-1)^2 = 4 -/
def hyperbola (x y m : ℝ) : Prop := x^2 - m*(y-1)^2 = 4

/-- The condition for the ellipse and hyperbola to be tangent -/
def are_tangent (m : ℝ) : Prop :=
  ∃ x y, ellipse x y ∧ hyperbola x y m ∧
  ∀ x' y', ellipse x' y' ∧ hyperbola x' y' m → (x', y') = (x, y)

/-- The theorem stating that if the ellipse and hyperbola are tangent, then m must be 6 or 12 -/
theorem tangent_implies_m_six_or_twelve :
  ∀ m, are_tangent m → m = 6 ∨ m = 12 :=
sorry

end NUMINAMATH_CALUDE_tangent_implies_m_six_or_twelve_l2100_210005


namespace NUMINAMATH_CALUDE_right_triangle_sin_d_l2100_210030

theorem right_triangle_sin_d (D E F : Real) (h1 : 4 * Real.sin D = 5 * Real.cos D) :
  Real.sin D = 5 * Real.sqrt 41 / 41 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_d_l2100_210030


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2100_210020

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (40 - a) + b / (75 - b) + c / (85 - c) = 8) :
  8 / (40 - a) + 15 / (75 - b) + 17 / (85 - c) = 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2100_210020


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2100_210093

theorem complex_modulus_problem (z : ℂ) :
  (2017 * z - 25) / (z - 2017) = (3 : ℂ) + 4 * I →
  Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2100_210093
