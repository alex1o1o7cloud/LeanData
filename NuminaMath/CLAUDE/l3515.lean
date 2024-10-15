import Mathlib

namespace NUMINAMATH_CALUDE_biathlon_average_speed_l3515_351518

def cycling_speed : ℝ := 18
def running_speed : ℝ := 8

theorem biathlon_average_speed :
  let harmonic_mean := 2 / (1 / cycling_speed + 1 / running_speed)
  harmonic_mean = 144 / 13 := by
  sorry

end NUMINAMATH_CALUDE_biathlon_average_speed_l3515_351518


namespace NUMINAMATH_CALUDE_sum_in_base_b_l3515_351594

/-- Given a base b, this function converts a number from base b to base 10 --/
def toBase10 (b : ℕ) (x : ℕ) : ℕ := sorry

/-- Given a base b, this function converts a number from base 10 to base b --/
def fromBase10 (b : ℕ) (x : ℕ) : ℕ := sorry

/-- The product of 12, 15, and 16 in base b --/
def product (b : ℕ) : ℕ := toBase10 b 12 * toBase10 b 15 * toBase10 b 16

/-- The sum of 12, 15, and 16 in base b --/
def sum (b : ℕ) : ℕ := toBase10 b 12 + toBase10 b 15 + toBase10 b 16

theorem sum_in_base_b (b : ℕ) :
  (product b = toBase10 b 3146) → (fromBase10 b (sum b) = 44) := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base_b_l3515_351594


namespace NUMINAMATH_CALUDE_dot_path_length_on_rotating_cube_l3515_351511

/-- The path length of a dot on a rotating cube -/
theorem dot_path_length_on_rotating_cube (cube_edge : ℝ) (h_edge : cube_edge = 2) :
  let dot_radius : ℝ := cube_edge / 2
  let path_length : ℝ := 2 * Real.pi * dot_radius
  path_length = 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_dot_path_length_on_rotating_cube_l3515_351511


namespace NUMINAMATH_CALUDE_car_wash_earnings_l3515_351555

def weekly_allowance : ℝ := 8
def final_amount : ℝ := 12

theorem car_wash_earnings :
  final_amount - (weekly_allowance / 2) = 8 := by sorry

end NUMINAMATH_CALUDE_car_wash_earnings_l3515_351555


namespace NUMINAMATH_CALUDE_alexis_alyssa_age_multiple_l3515_351523

theorem alexis_alyssa_age_multiple :
  ∀ (alexis_age alyssa_age : ℝ),
    alexis_age = 45 →
    alyssa_age = 45 →
    ∃ k : ℝ, alexis_age = k * alyssa_age - 162 →
    k = 4.6 :=
by
  sorry

end NUMINAMATH_CALUDE_alexis_alyssa_age_multiple_l3515_351523


namespace NUMINAMATH_CALUDE_units_digit_of_seven_to_six_to_five_l3515_351585

theorem units_digit_of_seven_to_six_to_five (n : ℕ) :
  7^(6^5) ≡ 1 [MOD 10] :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_to_six_to_five_l3515_351585


namespace NUMINAMATH_CALUDE_complex_number_property_l3515_351596

theorem complex_number_property (b : ℝ) : 
  let z : ℂ := (2 - b * Complex.I) / (1 + 2 * Complex.I)
  (z.re = -z.im) → b = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_property_l3515_351596


namespace NUMINAMATH_CALUDE_area_is_60_l3515_351550

/-- Two perpendicular lines intersecting at point A(6,8) with y-intercepts P and Q -/
structure PerpendicularLines where
  A : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  perpendicular : True  -- Represents that the lines are perpendicular
  intersect_at_A : True -- Represents that the lines intersect at A
  A_coords : A = (6, 8)
  P_is_y_intercept : P.1 = 0
  Q_is_y_intercept : Q.1 = 0
  sum_of_y_intercepts_zero : P.2 + Q.2 = 0

/-- The area of triangle APQ -/
def triangle_area (lines : PerpendicularLines) : ℝ := sorry

/-- Theorem stating that the area of triangle APQ is 60 -/
theorem area_is_60 (lines : PerpendicularLines) : triangle_area lines = 60 := by
  sorry

end NUMINAMATH_CALUDE_area_is_60_l3515_351550


namespace NUMINAMATH_CALUDE_horner_rule_v4_horner_rule_correct_l3515_351551

def horner_polynomial (x : ℝ) : ℝ := 3*x^6 + 5*x^5 + 6*x^4 + 20*x^3 - 8*x^2 + 35*x + 12

def horner_v4 (x : ℝ) : ℝ :=
  let v0 := 3
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  let v3 := v2 * x + 20
  v3 * x - 8

theorem horner_rule_v4 :
  horner_v4 (-2) = -16 :=
by sorry

theorem horner_rule_correct :
  horner_v4 (-2) = horner_polynomial (-2) :=
by sorry

end NUMINAMATH_CALUDE_horner_rule_v4_horner_rule_correct_l3515_351551


namespace NUMINAMATH_CALUDE_num_tangent_circles_bounds_l3515_351593

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a circle in 2D space --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The number of solutions for circles tangent to a line and another circle --/
def num_tangent_circles (r : ℝ) (L : Line) (C : Circle) : ℕ :=
  sorry

/-- Theorem stating the bounds on the number of tangent circles --/
theorem num_tangent_circles_bounds (r : ℝ) (L : Line) (C : Circle) :
  0 ≤ num_tangent_circles r L C ∧ num_tangent_circles r L C ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_num_tangent_circles_bounds_l3515_351593


namespace NUMINAMATH_CALUDE_xy_and_x2y_2xy2_values_l3515_351549

theorem xy_and_x2y_2xy2_values (x y : ℝ) 
  (h1 : x - 2*y = 3) 
  (h2 : x^2 - 2*x*y + 4*y^2 = 11) : 
  x * y = 1 ∧ x^2 * y - 2 * x * y^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_xy_and_x2y_2xy2_values_l3515_351549


namespace NUMINAMATH_CALUDE_hexagon_quadrilateral_areas_l3515_351569

/-- The area of a regular hexagon -/
def hexagon_area : ℝ := 156

/-- The number of distinct quadrilateral shapes possible -/
def num_distinct_quadrilaterals : ℕ := 3

/-- The areas of the distinct quadrilaterals -/
def quadrilateral_areas : Set ℝ := {78, 104}

/-- Theorem: Given a regular hexagon with area 156 cm², the areas of all possible
    distinct quadrilaterals formed by its vertices are 78 cm² and 104 cm² -/
theorem hexagon_quadrilateral_areas :
  ∀ (area : ℝ), area ∈ quadrilateral_areas →
  ∃ (vertices : Finset (Fin 6)), vertices.card = 4 ∧
  (area = hexagon_area / 2 ∨ area = hexagon_area * 2 / 3) :=
sorry

end NUMINAMATH_CALUDE_hexagon_quadrilateral_areas_l3515_351569


namespace NUMINAMATH_CALUDE_min_value_expression_l3515_351532

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 2) (hab : a + b = 2) :
  (a * c / b) + (c / (a * b)) - (c / 2) + (Real.sqrt 5 / (c - 2)) ≥ Real.sqrt 10 + Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3515_351532


namespace NUMINAMATH_CALUDE_min_socks_for_given_problem_l3515_351525

/-- The minimum number of socks to pull out to guarantee at least one of each color -/
def min_socks_to_pull (red blue green khaki : ℕ) : ℕ :=
  (red + blue + green + khaki) - min red (min blue (min green khaki)) + 1

/-- Theorem stating the minimum number of socks to pull out for the given problem -/
theorem min_socks_for_given_problem :
  min_socks_to_pull 10 20 30 40 = 91 := by
  sorry

end NUMINAMATH_CALUDE_min_socks_for_given_problem_l3515_351525


namespace NUMINAMATH_CALUDE_no_savings_on_joint_purchase_l3515_351542

/-- Calculates the number of paid windows given the total number of windows needed -/
def paidWindows (total : ℕ) : ℕ :=
  total - (total / 3)

/-- Calculates the cost of windows before any flat discount -/
def windowCost (paid : ℕ) : ℕ :=
  paid * 150

/-- Applies the flat discount if the cost is over 1000 -/
def applyDiscount (cost : ℕ) : ℕ :=
  if cost > 1000 then cost - 200 else cost

theorem no_savings_on_joint_purchase (dave_windows doug_windows : ℕ) 
  (h_dave : dave_windows = 9) (h_doug : doug_windows = 10) :
  let dave_cost := applyDiscount (windowCost (paidWindows dave_windows))
  let doug_cost := applyDiscount (windowCost (paidWindows doug_windows))
  let separate_cost := dave_cost + doug_cost
  let joint_windows := dave_windows + doug_windows
  let joint_cost := applyDiscount (windowCost (paidWindows joint_windows))
  separate_cost = joint_cost := by
  sorry

end NUMINAMATH_CALUDE_no_savings_on_joint_purchase_l3515_351542


namespace NUMINAMATH_CALUDE_gamblers_initial_win_rate_l3515_351539

theorem gamblers_initial_win_rate 
  (initial_games : ℕ) 
  (additional_games : ℕ) 
  (new_win_rate : ℚ) 
  (final_win_rate : ℚ) :
  initial_games = 30 →
  additional_games = 30 →
  new_win_rate = 4/5 →
  final_win_rate = 3/5 →
  ∃ (initial_win_rate : ℚ),
    initial_win_rate = 2/5 ∧
    (initial_win_rate * initial_games + new_win_rate * additional_games) / (initial_games + additional_games) = final_win_rate :=
by sorry

end NUMINAMATH_CALUDE_gamblers_initial_win_rate_l3515_351539


namespace NUMINAMATH_CALUDE_convex_polygon_27_diagonals_has_9_sides_l3515_351533

/-- The number of diagonals in a convex n-gon --/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 27 diagonals has 9 sides --/
theorem convex_polygon_27_diagonals_has_9_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 27 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_27_diagonals_has_9_sides_l3515_351533


namespace NUMINAMATH_CALUDE_jamie_oliver_vacation_cost_l3515_351567

def vacation_cost (num_people : ℕ) (num_days : ℕ) (ticket_cost : ℕ) (hotel_cost_per_day : ℕ) : ℕ :=
  num_people * ticket_cost + num_people * hotel_cost_per_day * num_days

theorem jamie_oliver_vacation_cost :
  vacation_cost 2 3 24 12 = 120 := by
  sorry

end NUMINAMATH_CALUDE_jamie_oliver_vacation_cost_l3515_351567


namespace NUMINAMATH_CALUDE_polar_line_properties_l3515_351583

/-- A line in polar coordinates passing through (2, π/3) and parallel to the polar axis -/
def polar_line (r θ : ℝ) : Prop :=
  r * Real.sin θ = Real.sqrt 3

theorem polar_line_properties :
  ∀ (r θ : ℝ),
    polar_line r θ →
    (r = 2 ∧ θ = π/3 → polar_line 2 (π/3)) ∧
    (∀ (r' : ℝ), polar_line r' θ → r' * Real.sin θ = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_polar_line_properties_l3515_351583


namespace NUMINAMATH_CALUDE_sqrt_floor_problem_l3515_351548

theorem sqrt_floor_problem (a b c : ℝ) : 
  (abs a = 4) → 
  (b^2 = 9) → 
  (c^3 = -8) → 
  (a > c) → 
  (c > b) → 
  Int.floor (Real.sqrt (a - b - 2*c)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_floor_problem_l3515_351548


namespace NUMINAMATH_CALUDE_cube_diagonal_length_l3515_351559

theorem cube_diagonal_length (s : ℝ) (h : s = 15) :
  let diagonal := Real.sqrt (3 * s^2)
  diagonal = 15 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_diagonal_length_l3515_351559


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3515_351558

-- Define the propositions
def p : Prop := (m : ℝ) → m = -1

def q (m : ℝ) : Prop := 
  let line1 : ℝ → ℝ → Prop := λ x y => x - y = 0
  let line2 : ℝ → ℝ → Prop := λ x y => x + m^2 * y = 0
  ∀ x1 y1 x2 y2, line1 x1 y1 → line2 x2 y2 → 
    (x2 - x1) * (y2 - y1) + (x2 - x1) * (x2 - x1) = 0

-- Theorem statement
theorem p_sufficient_not_necessary_for_q :
  (∃ m : ℝ, p → q m) ∧ (∃ m : ℝ, q m ∧ ¬p) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3515_351558


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l3515_351529

theorem power_fraction_simplification :
  (16 : ℕ) ^ 24 / (64 : ℕ) ^ 8 = (16 : ℕ) ^ 12 := by sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l3515_351529


namespace NUMINAMATH_CALUDE_i_power_sum_l3515_351513

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the property that i^2 = -1
axiom i_squared : i^2 = -1

-- Define the property that powers of i repeat every 4 powers
axiom i_power_cycle (n : ℤ) : i^n = i^(n % 4)

-- State the theorem
theorem i_power_sum : i^17 + i^2023 = 0 := by
  sorry

end NUMINAMATH_CALUDE_i_power_sum_l3515_351513


namespace NUMINAMATH_CALUDE_clara_stickers_l3515_351574

theorem clara_stickers (initial_stickers : ℕ) (stickers_to_boy : ℕ) (final_stickers : ℕ) : 
  initial_stickers = 100 →
  final_stickers = 45 →
  final_stickers = (initial_stickers - stickers_to_boy) / 2 →
  stickers_to_boy = 10 := by
  sorry

end NUMINAMATH_CALUDE_clara_stickers_l3515_351574


namespace NUMINAMATH_CALUDE_sequence_sum_l3515_351592

theorem sequence_sum (A B C D E F G H I : ℝ) : 
  D = 8 →
  A + B + C + D = 50 →
  B + C + D + E = 50 →
  C + D + E + F = 50 →
  D + E + F + G = 50 →
  E + F + G + H = 50 →
  F + G + H + I = 50 →
  A + I = 92 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l3515_351592


namespace NUMINAMATH_CALUDE_min_value_absolute_sum_l3515_351504

theorem min_value_absolute_sum (x y : ℝ) : 
  |x - 1| + |x| + |y - 1| + |y + 1| ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_absolute_sum_l3515_351504


namespace NUMINAMATH_CALUDE_circle_transformation_l3515_351571

theorem circle_transformation (x₀ y₀ x y : ℝ) :
  x₀^2 + y₀^2 = 9 → x = x₀ → y = 4*y₀ → x^2/9 + y^2/144 = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_transformation_l3515_351571


namespace NUMINAMATH_CALUDE_equation_to_lines_l3515_351524

theorem equation_to_lines :
  ∀ x y : ℝ,
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔
  (y = -x - 2 ∨ y = -2 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_to_lines_l3515_351524


namespace NUMINAMATH_CALUDE_triangle_inequality_l3515_351560

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  2 * (a^2 * b^2 + b^2 * c^2 + a^2 * c^2) > a^4 + b^4 + c^4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3515_351560


namespace NUMINAMATH_CALUDE_abc_sum_product_bound_l3515_351589

theorem abc_sum_product_bound (a b c : ℝ) (h : a + b + c = 3) :
  ∃ (M : ℝ), ∀ (x : ℝ), x ≤ M ∧ (∃ (a' b' c' : ℝ), a' + b' + c' = 3 ∧ a' * b' + a' * c' + b' * c' = x) :=
sorry

end NUMINAMATH_CALUDE_abc_sum_product_bound_l3515_351589


namespace NUMINAMATH_CALUDE_max_intersections_l3515_351541

/-- A circle in a plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane, represented by its slope and y-intercept. -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The configuration of figures on the plane. -/
structure Configuration where
  circle : Circle
  lines : Fin 3 → Line

/-- The number of intersection points between a circle and a line. -/
def circleLineIntersections (c : Circle) (l : Line) : ℕ := sorry

/-- The number of intersection points between two lines. -/
def lineLineIntersections (l1 l2 : Line) : ℕ := sorry

/-- The total number of intersection points in a configuration. -/
def totalIntersections (config : Configuration) : ℕ := sorry

/-- The theorem stating that the maximum number of intersections is 9. -/
theorem max_intersections :
  ∃ (config : Configuration), totalIntersections config = 9 ∧
  ∀ (other : Configuration), totalIntersections other ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_l3515_351541


namespace NUMINAMATH_CALUDE_volume_maximized_when_perpendicular_l3515_351528

/-- A tetrahedron with edge lengths u, v, and w. -/
structure Tetrahedron (u v w : ℝ) where
  edge_u : ℝ := u
  edge_v : ℝ := v
  edge_w : ℝ := w

/-- The volume of a tetrahedron. -/
noncomputable def volume (t : Tetrahedron u v w) : ℝ :=
  sorry

/-- Mutually perpendicular edges of a tetrahedron. -/
def mutually_perpendicular (t : Tetrahedron u v w) : Prop :=
  sorry

/-- Theorem: The volume of a tetrahedron is maximized when its edges are mutually perpendicular. -/
theorem volume_maximized_when_perpendicular (u v w : ℝ) (t : Tetrahedron u v w) :
  mutually_perpendicular t ↔ ∀ (t' : Tetrahedron u v w), volume t ≥ volume t' :=
sorry

end NUMINAMATH_CALUDE_volume_maximized_when_perpendicular_l3515_351528


namespace NUMINAMATH_CALUDE_book_difference_l3515_351570

theorem book_difference (total : ℕ) (fiction : ℕ) (picture : ℕ)
  (h_total : total = 35)
  (h_fiction : fiction = 5)
  (h_picture : picture = 11)
  (h_autobio : ∃ autobio : ℕ, autobio = 2 * fiction) :
  ∃ nonfiction : ℕ, nonfiction - fiction = 4 :=
by sorry

end NUMINAMATH_CALUDE_book_difference_l3515_351570


namespace NUMINAMATH_CALUDE_sculpture_cost_in_yuan_l3515_351508

theorem sculpture_cost_in_yuan 
  (usd_to_nam : ℝ) -- Exchange rate from USD to Namibian dollars
  (usd_to_cny : ℝ) -- Exchange rate from USD to Chinese yuan
  (cost_nam : ℝ) -- Cost of the sculpture in Namibian dollars
  (h1 : usd_to_nam = 8) -- 1 USD = 8 Namibian dollars
  (h2 : usd_to_cny = 5) -- 1 USD = 5 Chinese yuan
  (h3 : cost_nam = 160) -- The sculpture costs 160 Namibian dollars
  : cost_nam / usd_to_nam * usd_to_cny = 100 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_in_yuan_l3515_351508


namespace NUMINAMATH_CALUDE_price_relationship_total_cost_max_toy_A_l3515_351564

/- Define the unit prices of toys A and B -/
def price_A : ℕ := 50
def price_B : ℕ := 75

/- Define the relationship between prices -/
theorem price_relationship : price_B = price_A + 25 := by sorry

/- Define the total cost of 2B and 1A -/
theorem total_cost : 2 * price_B + price_A = 200 := by sorry

/- Define the function for total cost given number of A -/
def total_cost_function (num_A : ℕ) : ℕ := price_A * num_A + price_B * (2 * num_A)

/- Define the maximum budget -/
def max_budget : ℕ := 20000

/- Theorem to prove the maximum number of toy A that can be purchased -/
theorem max_toy_A : 
  (∀ n : ℕ, total_cost_function n ≤ max_budget → n ≤ 100) ∧ 
  total_cost_function 100 ≤ max_budget := by sorry

end NUMINAMATH_CALUDE_price_relationship_total_cost_max_toy_A_l3515_351564


namespace NUMINAMATH_CALUDE_extra_people_on_train_l3515_351576

theorem extra_people_on_train (current : ℕ) (initial : ℕ) (got_off : ℕ)
  (h1 : current = 63)
  (h2 : initial = 78)
  (h3 : got_off = 27) :
  current - (initial - got_off) = 12 :=
by sorry

end NUMINAMATH_CALUDE_extra_people_on_train_l3515_351576


namespace NUMINAMATH_CALUDE_serezha_puts_more_berries_l3515_351556

/-- Represents the berry picking scenario -/
structure BerryPicking where
  total_berries : ℕ
  serezha_rate : ℕ → ℕ  -- Function representing Serezha's picking pattern
  dima_rate : ℕ → ℕ     -- Function representing Dima's picking pattern
  serezha_speed : ℕ
  dima_speed : ℕ

/-- The specific berry picking scenario from the problem -/
def berry_scenario : BerryPicking :=
  { total_berries := 450
  , serezha_rate := λ n => n / 2  -- 1 out of every 2
  , dima_rate := λ n => 2 * n / 3 -- 2 out of every 3
  , serezha_speed := 2
  , dima_speed := 1 }

/-- Theorem stating the difference in berries put in basket -/
theorem serezha_puts_more_berries (bp : BerryPicking) (h : bp = berry_scenario) :
  ∃ (s d : ℕ), s = bp.serezha_rate (bp.serezha_speed * bp.total_berries / (bp.serezha_speed + bp.dima_speed)) ∧
                d = bp.dima_rate (bp.dima_speed * bp.total_berries / (bp.serezha_speed + bp.dima_speed)) ∧
                s - d = 50 := by
  sorry


end NUMINAMATH_CALUDE_serezha_puts_more_berries_l3515_351556


namespace NUMINAMATH_CALUDE_optimal_config_is_minimum_l3515_351568

/-- Represents the types of vans available --/
inductive VanType
  | A
  | B
  | C

/-- Capacity of each van type --/
def vanCapacity : VanType → ℕ
  | VanType.A => 7
  | VanType.B => 9
  | VanType.C => 12

/-- Available number of each van type --/
def availableVans : VanType → ℕ
  | VanType.A => 3
  | VanType.B => 4
  | VanType.C => 2

/-- Total number of people to transport --/
def totalPeople : ℕ := 40 + 14

/-- A configuration of vans --/
structure VanConfiguration where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Calculate the total capacity of a given van configuration --/
def totalCapacity (config : VanConfiguration) : ℕ :=
  config.typeA * vanCapacity VanType.A +
  config.typeB * vanCapacity VanType.B +
  config.typeC * vanCapacity VanType.C

/-- Check if a van configuration is valid (within available vans) --/
def isValidConfiguration (config : VanConfiguration) : Prop :=
  config.typeA ≤ availableVans VanType.A ∧
  config.typeB ≤ availableVans VanType.B ∧
  config.typeC ≤ availableVans VanType.C

/-- The optimal van configuration --/
def optimalConfig : VanConfiguration :=
  { typeA := 0, typeB := 4, typeC := 2 }

/-- Theorem stating that the optimal configuration is the minimum number of vans needed --/
theorem optimal_config_is_minimum :
  isValidConfiguration optimalConfig ∧
  totalCapacity optimalConfig ≥ totalPeople ∧
  ∀ (config : VanConfiguration),
    isValidConfiguration config →
    totalCapacity config ≥ totalPeople →
    config.typeA + config.typeB + config.typeC ≥
    optimalConfig.typeA + optimalConfig.typeB + optimalConfig.typeC :=
by
  sorry


end NUMINAMATH_CALUDE_optimal_config_is_minimum_l3515_351568


namespace NUMINAMATH_CALUDE_circle_radius_l3515_351503

theorem circle_radius (x y : ℝ) : 
  (16 * x^2 - 32 * x + 16 * y^2 + 64 * y + 64 = 0) → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_l3515_351503


namespace NUMINAMATH_CALUDE_three_numbers_problem_l3515_351522

theorem three_numbers_problem (a b c : ℝ) :
  ((a + 1) * (b + 1) * (c + 1) = a * b * c + 1) →
  ((a + 2) * (b + 2) * (c + 2) = a * b * c + 2) →
  (a = -1 ∧ b = -1 ∧ c = -1) :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_problem_l3515_351522


namespace NUMINAMATH_CALUDE_hotel_loss_calculation_l3515_351562

/-- Calculates the loss incurred by a hotel given its operations expenses and client payments --/
def hotel_loss (expenses : ℝ) (client_payment_ratio : ℝ) : ℝ :=
  expenses - (client_payment_ratio * expenses)

/-- Theorem: A hotel with $100 expenses and client payments of 3/4 of expenses incurs a $25 loss --/
theorem hotel_loss_calculation :
  hotel_loss 100 (3/4) = 25 := by
  sorry

end NUMINAMATH_CALUDE_hotel_loss_calculation_l3515_351562


namespace NUMINAMATH_CALUDE_disjoint_chords_with_equal_sum_endpoints_l3515_351510

/-- Given 2^500 points numbered 1 to 2^500 arranged on a circle,
    there exist 100 disjoint chords such that the sum of the endpoints
    is the same for each chord. -/
theorem disjoint_chords_with_equal_sum_endpoints :
  ∃ (chords : Finset (Fin (2^500) × Fin (2^500))) (s : ℕ),
    chords.card = 100 ∧
    (∀ (c₁ c₂ : Fin (2^500) × Fin (2^500)), c₁ ∈ chords → c₂ ∈ chords → c₁ ≠ c₂ →
      (c₁.1 ≠ c₂.1 ∧ c₁.1 ≠ c₂.2 ∧ c₁.2 ≠ c₂.1 ∧ c₁.2 ≠ c₂.2)) ∧
    (∀ (c : Fin (2^500) × Fin (2^500)), c ∈ chords →
      c.1.val + c.2.val = s) :=
by sorry

end NUMINAMATH_CALUDE_disjoint_chords_with_equal_sum_endpoints_l3515_351510


namespace NUMINAMATH_CALUDE_souvenir_cost_in_usd_l3515_351512

/-- Calculates the cost in USD given the cost in yen and the exchange rate -/
def cost_in_usd (cost_yen : ℚ) (exchange_rate : ℚ) : ℚ :=
  cost_yen / exchange_rate

theorem souvenir_cost_in_usd :
  let cost_yen : ℚ := 500
  let exchange_rate : ℚ := 120
  cost_in_usd cost_yen exchange_rate = 25 / 6 := by sorry

end NUMINAMATH_CALUDE_souvenir_cost_in_usd_l3515_351512


namespace NUMINAMATH_CALUDE_divisibility_condition_l3515_351587

theorem divisibility_condition (x y : ℕ+) :
  (∃ k : ℤ, (2 * x * y^2 - y^3 + 1 : ℤ) = k * x^2) ↔
  (∃ t : ℕ+, (x = 2 * t ∧ y = 1) ∨
             (x = t ∧ y = 2 * t) ∨
             (x = 8 * t^4 - t ∧ y = 2 * t)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3515_351587


namespace NUMINAMATH_CALUDE_correct_outfit_count_l3515_351547

/-- The number of outfits that can be made with given clothing items, 
    where shirts and hats cannot be the same color. -/
def number_of_outfits (red_shirts green_shirts pants blue_hats red_hats scarves : ℕ) : ℕ :=
  (red_shirts * pants * blue_hats * scarves) + (green_shirts * pants * red_hats * scarves)

/-- Theorem stating the correct number of outfits given specific quantities of clothing items. -/
theorem correct_outfit_count : 
  number_of_outfits 7 8 10 10 10 5 = 7500 := by
  sorry

end NUMINAMATH_CALUDE_correct_outfit_count_l3515_351547


namespace NUMINAMATH_CALUDE_cubic_identity_l3515_351580

theorem cubic_identity : ∀ x : ℝ, (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l3515_351580


namespace NUMINAMATH_CALUDE_li_family_cinema_cost_l3515_351517

def adult_ticket_price : ℝ := 10
def child_discount : ℝ := 0.4
def senior_discount : ℝ := 0.3
def handling_fee : ℝ := 5
def num_adults : ℕ := 2
def num_children : ℕ := 1
def num_seniors : ℕ := 1

def total_cost : ℝ :=
  (num_adults * adult_ticket_price) +
  (num_children * adult_ticket_price * (1 - child_discount)) +
  (num_seniors * adult_ticket_price * (1 - senior_discount)) +
  handling_fee

theorem li_family_cinema_cost : total_cost = 38 := by
  sorry

end NUMINAMATH_CALUDE_li_family_cinema_cost_l3515_351517


namespace NUMINAMATH_CALUDE_final_state_is_twelve_and_fourteen_l3515_351506

/-- Represents the numbers on the blackboard -/
inductive Number
  | eleven
  | twelve
  | thirteen
  | fourteen
  | fifteen

/-- The state of the blackboard -/
structure BoardState where
  counts : Number → Nat
  total : Nat

/-- The initial state of the blackboard -/
def initial_state : BoardState := {
  counts := λ n => match n with
    | Number.eleven => 11
    | Number.twelve => 12
    | Number.thirteen => 13
    | Number.fourteen => 14
    | Number.fifteen => 15
  total := 65
}

/-- Represents an operation on the board -/
def operation (s : BoardState) : BoardState :=
  sorry  -- Implementation of the operation

/-- Predicate to check if a state has exactly two numbers remaining -/
def has_two_remaining (s : BoardState) : Prop :=
  (s.total = 2) ∧ (∃ a b : Number, a ≠ b ∧ s.counts a > 0 ∧ s.counts b > 0 ∧ 
    ∀ c : Number, c ≠ a ∧ c ≠ b → s.counts c = 0)

/-- The main theorem -/
theorem final_state_is_twelve_and_fourteen :
  ∃ (n : Nat), 
    let final_state := (operation^[n] initial_state)
    has_two_remaining final_state ∧ 
    final_state.counts Number.twelve > 0 ∧ 
    final_state.counts Number.fourteen > 0 :=
  sorry


end NUMINAMATH_CALUDE_final_state_is_twelve_and_fourteen_l3515_351506


namespace NUMINAMATH_CALUDE_heptagon_interior_angle_sum_heptagon_interior_angle_sum_proof_l3515_351507

/-- The sum of the interior angles of a heptagon is 900 degrees. -/
theorem heptagon_interior_angle_sum : ℝ :=
  900

/-- A heptagon is a polygon with 7 sides. -/
def heptagon_sides : ℕ := 7

/-- The formula for the sum of interior angles of a polygon with n sides. -/
def polygon_interior_angle_sum (n : ℕ) : ℝ :=
  (n - 2) * 180

theorem heptagon_interior_angle_sum_proof :
  polygon_interior_angle_sum heptagon_sides = heptagon_interior_angle_sum :=
by
  sorry

end NUMINAMATH_CALUDE_heptagon_interior_angle_sum_heptagon_interior_angle_sum_proof_l3515_351507


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3515_351554

/-- The sum of a geometric series with first term 2, common ratio -2, and last term 1024 -/
def geometricSeriesSum : ℤ := -682

/-- The first term of the geometric series -/
def firstTerm : ℤ := 2

/-- The common ratio of the geometric series -/
def commonRatio : ℤ := -2

/-- The last term of the geometric series -/
def lastTerm : ℤ := 1024

theorem geometric_series_sum :
  ∃ (n : ℕ), n > 0 ∧ firstTerm * commonRatio^(n - 1) = lastTerm ∧
  geometricSeriesSum = firstTerm * (commonRatio^n - 1) / (commonRatio - 1) :=
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3515_351554


namespace NUMINAMATH_CALUDE_sequence_existence_theorem_l3515_351543

theorem sequence_existence_theorem :
  (¬ ∃ (a : ℕ → ℕ), ∀ n, (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2))) ∧
  (∃ (a : ℤ → ℝ), (∀ n, Irrational (a n)) ∧ (∀ n, (a (n - 1))^2 ≥ 2 * (a n) * (a (n - 2)))) :=
by sorry

end NUMINAMATH_CALUDE_sequence_existence_theorem_l3515_351543


namespace NUMINAMATH_CALUDE_equation_solutions_l3515_351545

theorem equation_solutions :
  (∀ x, x^2 - 7*x = 0 ↔ x = 0 ∨ x = 7) ∧
  (∀ x, 2*x^2 - 6*x + 1 = 0 ↔ x = (3 + Real.sqrt 7) / 2 ∨ x = (3 - Real.sqrt 7) / 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3515_351545


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l3515_351584

/-- Given sets M and N in the real numbers, prove that the intersection of the complement of M and N is the set of all real numbers less than -2. -/
theorem complement_M_intersect_N (M N : Set ℝ) 
  (hM : M = {x : ℝ | -2 ≤ x ∧ x ≤ 2})
  (hN : N = {x : ℝ | x < 1}) :
  (Mᶜ ∩ N) = {x : ℝ | x < -2} := by
  sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l3515_351584


namespace NUMINAMATH_CALUDE_difference_of_numbers_l3515_351582

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 - y^2 = 50) : x - y = 10 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l3515_351582


namespace NUMINAMATH_CALUDE_chord_bisected_by_point_l3515_351531

/-- The equation of an ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

/-- The midpoint of two points -/
def is_midpoint (x₁ y₁ x₂ y₂ x_mid y_mid : ℝ) : Prop :=
  x_mid = (x₁ + x₂) / 2 ∧ y_mid = (y₁ + y₂) / 2

/-- A point is on a line -/
def is_on_line (x y : ℝ) : Prop := x + 2*y - 8 = 0

/-- The main theorem -/
theorem chord_bisected_by_point (x₁ y₁ x₂ y₂ : ℝ) :
  is_on_ellipse x₁ y₁ →
  is_on_ellipse x₂ y₂ →
  is_midpoint x₁ y₁ x₂ y₂ 4 2 →
  is_on_line x₁ y₁ ∧ is_on_line x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_chord_bisected_by_point_l3515_351531


namespace NUMINAMATH_CALUDE_melanie_initial_dimes_l3515_351581

/-- The number of dimes Melanie had initially -/
def initial_dimes : ℕ := sorry

/-- The number of dimes Melanie received from her dad -/
def dimes_from_dad : ℕ := 8

/-- The number of dimes Melanie received from her mother -/
def dimes_from_mom : ℕ := 4

/-- The total number of dimes Melanie has now -/
def total_dimes_now : ℕ := 19

/-- Theorem stating that Melanie initially had 7 dimes -/
theorem melanie_initial_dimes : 
  initial_dimes = 7 :=
by sorry

end NUMINAMATH_CALUDE_melanie_initial_dimes_l3515_351581


namespace NUMINAMATH_CALUDE_parabola_comparison_l3515_351566

theorem parabola_comparison :
  ∀ x : ℝ, -x^2 + 2*x + 3 > x^2 - 2*x + 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_comparison_l3515_351566


namespace NUMINAMATH_CALUDE_arc_length_of_sector_l3515_351578

/-- Given a circle with radius 4 cm and a sector with an area of 7 square centimeters,
    the length of the arc forming this sector is 3.5 cm. -/
theorem arc_length_of_sector (r : ℝ) (area : ℝ) (arc_length : ℝ) : 
  r = 4 → area = 7 → arc_length = (area * 2) / r → arc_length = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_of_sector_l3515_351578


namespace NUMINAMATH_CALUDE_f_minimized_at_x_min_l3515_351591

/-- The quadratic function we're minimizing -/
def f (x : ℝ) := 2 * x^2 - 8 * x + 6

/-- The value of x that minimizes f -/
def x_min : ℝ := 2

theorem f_minimized_at_x_min :
  ∀ x : ℝ, f x_min ≤ f x :=
sorry

end NUMINAMATH_CALUDE_f_minimized_at_x_min_l3515_351591


namespace NUMINAMATH_CALUDE_grain_production_scientific_notation_l3515_351586

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) (h : x > 0) : ScientificNotation :=
  sorry

theorem grain_production_scientific_notation :
  toScientificNotation 686530000 (by norm_num) =
    ScientificNotation.mk 6.8653 8 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_grain_production_scientific_notation_l3515_351586


namespace NUMINAMATH_CALUDE_report_card_recess_num_of_ds_l3515_351502

/-- Calculates the number of Ds on report cards given the recess rules and grades --/
theorem report_card_recess (normal_recess : ℕ) (a_bonus : ℕ) (b_bonus : ℕ) (d_penalty : ℕ)
  (num_a : ℕ) (num_b : ℕ) (num_c : ℕ) (total_recess : ℕ) : ℕ :=
  let extra_time := num_a * a_bonus + num_b * b_bonus
  let expected_time := normal_recess + extra_time
  let reduced_time := expected_time - total_recess
  reduced_time / d_penalty

/-- Proves that there are 5 Ds on the report cards --/
theorem num_of_ds : report_card_recess 20 2 1 1 10 12 14 47 = 5 := by
  sorry

end NUMINAMATH_CALUDE_report_card_recess_num_of_ds_l3515_351502


namespace NUMINAMATH_CALUDE_sum_of_cubes_difference_l3515_351572

theorem sum_of_cubes_difference (a b c : ℕ+) :
  (a + b + c)^3 - a^3 - b^3 - c^3 = 2700 → a + b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_difference_l3515_351572


namespace NUMINAMATH_CALUDE_smallest_product_of_primes_above_50_l3515_351577

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def first_prime_above_50 : ℕ := 53

def second_prime_above_50 : ℕ := 59

theorem smallest_product_of_primes_above_50 :
  (is_prime first_prime_above_50) ∧
  (is_prime second_prime_above_50) ∧
  (first_prime_above_50 > 50) ∧
  (second_prime_above_50 > 50) ∧
  (first_prime_above_50 < second_prime_above_50) ∧
  (∀ p : ℕ, is_prime p ∧ p > 50 ∧ p ≠ first_prime_above_50 → p ≥ second_prime_above_50) →
  first_prime_above_50 * second_prime_above_50 = 3127 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_of_primes_above_50_l3515_351577


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l3515_351573

theorem gcd_lcm_sum : Nat.gcd 54 72 + Nat.lcm 50 15 = 168 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l3515_351573


namespace NUMINAMATH_CALUDE_sin_2theta_value_l3515_351546

theorem sin_2theta_value (θ : Real) (h : Real.sin (θ + π/4) = 1/3) : 
  Real.sin (2*θ) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l3515_351546


namespace NUMINAMATH_CALUDE_max_ab_value_l3515_351521

/-- Given a > 0, b > 0, and f(x) = 4x^3 - ax^2 - 2bx + 2 has an extremum at x = 1,
    the maximum value of ab is 9. -/
theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let f := fun x => 4 * x^3 - a * x^2 - 2 * b * x + 2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  (∀ c d : ℝ, c > 0 → d > 0 → 
    (let g := fun x => 4 * x^3 - c * x^2 - 2 * d * x + 2
     ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), g x ≤ g 1 ∨ g x ≥ g 1) →
    a * b ≥ c * d) ∧ a * b = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_ab_value_l3515_351521


namespace NUMINAMATH_CALUDE_pta_funds_remaining_l3515_351530

def initial_amount : ℚ := 600

def amount_after_supplies (initial : ℚ) : ℚ :=
  initial - (2 / 5) * initial

def amount_after_food (after_supplies : ℚ) : ℚ :=
  after_supplies - (30 / 100) * after_supplies

def final_amount (after_food : ℚ) : ℚ :=
  after_food - (1 / 3) * after_food

theorem pta_funds_remaining :
  final_amount (amount_after_food (amount_after_supplies initial_amount)) = 168 := by
  sorry

end NUMINAMATH_CALUDE_pta_funds_remaining_l3515_351530


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l3515_351535

/-- A quadratic function f(x) = 4x^2 - kx - 8 has monotonicity on the interval (∞, 5] if and only if k ≥ 40 -/
theorem quadratic_monotonicity (k : ℝ) :
  (∀ x > 5, Monotone (fun x => 4 * x^2 - k * x - 8)) ↔ k ≥ 40 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l3515_351535


namespace NUMINAMATH_CALUDE_brothers_ages_theorem_l3515_351561

/-- Represents the ages of the three brothers -/
structure BrothersAges where
  kolya : ℕ
  vanya : ℕ
  petya : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : BrothersAges) : Prop :=
  ages.petya = 10 ∧
  ages.kolya = ages.petya + 3 ∧
  ages.vanya = ages.petya - 1

/-- The theorem to be proved -/
theorem brothers_ages_theorem (ages : BrothersAges) :
  satisfiesConditions ages → ages.vanya = 9 ∧ ages.kolya = 13 := by
  sorry

#check brothers_ages_theorem

end NUMINAMATH_CALUDE_brothers_ages_theorem_l3515_351561


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l3515_351526

theorem right_triangle_third_side (a b x : ℝ) :
  (a - 3)^2 + |b - 4| = 0 →
  (x^2 = a^2 + b^2 ∨ x^2 + a^2 = b^2 ∨ x^2 + b^2 = a^2) →
  x = 5 ∨ x = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l3515_351526


namespace NUMINAMATH_CALUDE_seungjus_class_size_l3515_351544

theorem seungjus_class_size :
  ∃! n : ℕ, 50 < n ∧ n < 70 ∧ n % 5 = 3 ∧ n % 7 = 2 :=
sorry

end NUMINAMATH_CALUDE_seungjus_class_size_l3515_351544


namespace NUMINAMATH_CALUDE_cone_max_volume_l3515_351501

/-- A cone with slant height 20 cm has maximum volume when its height is (20√3)/3 cm. -/
theorem cone_max_volume (h : ℝ) (h_pos : 0 < h) (h_bound : h < 20) :
  let r := Real.sqrt (400 - h^2)
  let v := (1/3) * Real.pi * h * r^2
  (∀ h' : ℝ, 0 < h' → h' < 20 → 
    (1/3) * Real.pi * h' * (Real.sqrt (400 - h'^2))^2 ≤ v) →
  h = 20 * Real.sqrt 3 / 3 := by
sorry


end NUMINAMATH_CALUDE_cone_max_volume_l3515_351501


namespace NUMINAMATH_CALUDE_fifteenth_term_ratio_l3515_351509

/-- Represents an arithmetic series -/
structure ArithmeticSeries where
  first_term : ℚ
  common_difference : ℚ

/-- Sum of the first n terms of an arithmetic series -/
def sum_n_terms (series : ArithmeticSeries) (n : ℕ) : ℚ :=
  n * (2 * series.first_term + (n - 1) * series.common_difference) / 2

/-- The nth term of an arithmetic series -/
def nth_term (series : ArithmeticSeries) (n : ℕ) : ℚ :=
  series.first_term + (n - 1) * series.common_difference

theorem fifteenth_term_ratio 
  (series1 series2 : ArithmeticSeries)
  (h : ∀ n : ℕ, sum_n_terms series1 n / sum_n_terms series2 n = (5 * n + 3) / (3 * n + 11)) :
  nth_term series1 15 / nth_term series2 15 = 71 / 52 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_ratio_l3515_351509


namespace NUMINAMATH_CALUDE_three_numbers_sum_l3515_351563

theorem three_numbers_sum : ∀ (a b c : ℝ),
  a ≤ b ∧ b ≤ c →  -- Arrange numbers in ascending order
  b = 10 →  -- Median is 10
  (a + b + c) / 3 = a + 8 →  -- Mean is 8 more than least
  (a + b + c) / 3 = c - 20 →  -- Mean is 20 less than greatest
  a + b + c = 66 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l3515_351563


namespace NUMINAMATH_CALUDE_kabulek_numbers_are_correct_l3515_351590

/-- A four-digit number is a Kabulek number if it equals the square of the sum of its first two digits and last two digits. -/
def isKabulek (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧ ∃ a b : ℕ, 
    a ≥ 10 ∧ a < 100 ∧ b ≥ 0 ∧ b < 100 ∧
    n = 100 * a + b ∧ n = (a + b)^2

/-- The set of all four-digit Kabulek numbers. -/
def kabulekNumbers : Set ℕ := {2025, 3025, 9801}

/-- Theorem stating that the set of all four-digit Kabulek numbers is exactly {2025, 3025, 9801}. -/
theorem kabulek_numbers_are_correct : 
  ∀ n : ℕ, isKabulek n ↔ n ∈ kabulekNumbers := by sorry

end NUMINAMATH_CALUDE_kabulek_numbers_are_correct_l3515_351590


namespace NUMINAMATH_CALUDE_smallest_divisor_of_28_l3515_351516

theorem smallest_divisor_of_28 : ∀ d : ℕ, d > 0 → d ∣ 28 → d ≥ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_divisor_of_28_l3515_351516


namespace NUMINAMATH_CALUDE_distribute_subtraction_l3515_351553

theorem distribute_subtraction (a b c : ℝ) : 5*a - (b + 2*c) = 5*a - b - 2*c := by
  sorry

end NUMINAMATH_CALUDE_distribute_subtraction_l3515_351553


namespace NUMINAMATH_CALUDE_inequality_holds_for_even_positive_integers_l3515_351565

theorem inequality_holds_for_even_positive_integers (n : ℕ) (hn : Even n) (hn_pos : 0 < n) :
  ∀ x : ℝ, 3 * x^n + n * (x + 2) - 3 ≥ n * x^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_for_even_positive_integers_l3515_351565


namespace NUMINAMATH_CALUDE_cube_root_simplification_l3515_351536

theorem cube_root_simplification :
  (20^3 + 30^3 + 40^3 : ℝ)^(1/3) = 10 * 99^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l3515_351536


namespace NUMINAMATH_CALUDE_S_intersect_T_eq_T_l3515_351534

-- Define the sets S and T
def S : Set ℝ := {y | ∃ x, y = 3*x + 2}
def T : Set ℝ := {y | ∃ x, y = x^2 - 1}

-- Statement to prove
theorem S_intersect_T_eq_T : S ∩ T = T := by
  sorry

end NUMINAMATH_CALUDE_S_intersect_T_eq_T_l3515_351534


namespace NUMINAMATH_CALUDE_M_intersect_N_l3515_351588

-- Define set M
def M : Set ℝ := {0, 1, 2}

-- Define set N
def N : Set ℝ := {x : ℝ | x^2 - 3*x + 2 ≤ 0}

-- Theorem statement
theorem M_intersect_N : M ∩ N = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_l3515_351588


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3515_351552

/-- A line in the form y = kx + b -/
structure Line where
  k : ℝ
  b : ℝ

/-- Checks if a line has equal intercepts on the coordinate axes -/
def hasEqualIntercepts (l : Line) : Prop :=
  ∃ c : ℝ, c ≠ 0 ∧ l.k * c + l.b = -c ∧ l.b = c

/-- The specific line y = kx + 2k - 1 -/
def specificLine (k : ℝ) : Line :=
  { k := k, b := 2 * k - 1 }

/-- The condition k = -1 is sufficient but not necessary for the line to have equal intercepts -/
theorem sufficient_not_necessary :
  (∀ k : ℝ, k = -1 → hasEqualIntercepts (specificLine k)) ∧
  (∃ k : ℝ, k ≠ -1 ∧ hasEqualIntercepts (specificLine k)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3515_351552


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l3515_351520

theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℚ),
  (3/4 : ℚ) * 16 * banana_value = 12 * orange_value →
  (3/5 : ℚ) * 20 * banana_value = 12 * orange_value := by
sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l3515_351520


namespace NUMINAMATH_CALUDE_no_valid_coloring_l3515_351575

/-- Represents a coloring of a rectangular grid --/
def GridColoring (m n : ℕ) := Fin m → Fin n → Bool

/-- Checks if the number of white cells equals the number of black cells --/
def equalColors (m n : ℕ) (coloring : GridColoring m n) : Prop :=
  (Finset.univ.sum fun i => (Finset.univ.sum fun j => if coloring i j then 1 else 0)) =
  (Finset.univ.sum fun i => (Finset.univ.sum fun j => if coloring i j then 0 else 1))

/-- Checks if more than 3/4 of cells in each row are of the same color --/
def rowColorDominance (m n : ℕ) (coloring : GridColoring m n) : Prop :=
  ∀ i, (4 * (Finset.univ.sum fun j => if coloring i j then 1 else 0) > 3 * n) ∨
       (4 * (Finset.univ.sum fun j => if coloring i j then 0 else 1) > 3 * n)

/-- Checks if more than 3/4 of cells in each column are of the same color --/
def columnColorDominance (m n : ℕ) (coloring : GridColoring m n) : Prop :=
  ∀ j, (4 * (Finset.univ.sum fun i => if coloring i j then 1 else 0) > 3 * m) ∨
       (4 * (Finset.univ.sum fun i => if coloring i j then 0 else 1) > 3 * m)

/-- The main theorem stating that no valid coloring exists --/
theorem no_valid_coloring (m n : ℕ) : ¬∃ (coloring : GridColoring m n),
  equalColors m n coloring ∧ rowColorDominance m n coloring ∧ columnColorDominance m n coloring :=
sorry

end NUMINAMATH_CALUDE_no_valid_coloring_l3515_351575


namespace NUMINAMATH_CALUDE_sum_of_squares_l3515_351537

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_cubes_eq_sum_sevenths : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3515_351537


namespace NUMINAMATH_CALUDE_remainder_of_1394_divided_by_2535_l3515_351514

theorem remainder_of_1394_divided_by_2535 : Int.mod 1394 2535 = 1394 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_1394_divided_by_2535_l3515_351514


namespace NUMINAMATH_CALUDE_intersection_M_N_l3515_351597

open Set Real

def M : Set ℝ := {x | Real.exp (x - 1) > 1}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem intersection_M_N : M ∩ N = Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3515_351597


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_and_product_l3515_351505

theorem consecutive_odd_integers_sum_and_product :
  ∀ x : ℚ,
  (x + 4 = 4 * x) →
  (x + (x + 4) = 20 / 3) ∧
  (x * (x + 4) = 64 / 9) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_and_product_l3515_351505


namespace NUMINAMATH_CALUDE_intersection_on_line_x_eq_4_l3515_351538

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m * y + 1

-- Define points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (2, 0)

-- Define the intersection points M and N
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ line_l m p.1 p.2}

-- Define the line AM
def line_AM (m : ℝ) (x y : ℝ) : Prop :=
  ∃ (M : ℝ × ℝ), M ∈ intersection_points m ∧
  (y - point_A.2) * (M.1 - point_A.1) = (x - point_A.1) * (M.2 - point_A.2)

-- Define the line BN
def line_BN (m : ℝ) (x y : ℝ) : Prop :=
  ∃ (N : ℝ × ℝ), N ∈ intersection_points m ∧
  (y - point_B.2) * (N.1 - point_B.1) = (x - point_B.1) * (N.2 - point_B.2)

-- Theorem statement
theorem intersection_on_line_x_eq_4 (m : ℝ) :
  ∃ (x y : ℝ), line_AM m x y ∧ line_BN m x y → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_on_line_x_eq_4_l3515_351538


namespace NUMINAMATH_CALUDE_matrix_transformation_l3515_351540

theorem matrix_transformation (P Q : Matrix (Fin 3) (Fin 3) ℝ) : 
  P = !![3, 0, 0; 0, 0, 1; 0, 1, 0] → 
  (∀ a b c d e f g h i : ℝ, 
    Q = !![a, b, c; d, e, f; g, h, i] → 
    P * Q = !![3*a, 3*b, 3*c; g, h, i; d, e, f]) :=
by sorry

end NUMINAMATH_CALUDE_matrix_transformation_l3515_351540


namespace NUMINAMATH_CALUDE_regular_20gon_symmetry_sum_l3515_351527

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- Add any necessary fields here

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := sorry

/-- The smallest positive angle of rotational symmetry in degrees -/
def smallestRotationAngle (p : RegularPolygon n) : ℝ := sorry

theorem regular_20gon_symmetry_sum :
  ∀ (p : RegularPolygon 20),
    (linesOfSymmetry p : ℝ) + smallestRotationAngle p = 38 := by sorry

end NUMINAMATH_CALUDE_regular_20gon_symmetry_sum_l3515_351527


namespace NUMINAMATH_CALUDE_extreme_values_of_f_l3515_351515

def f (x : ℝ) : ℝ := 3 * x^5 - 5 * x^3

theorem extreme_values_of_f :
  ∃ (a b : ℝ), (∀ x : ℝ, f x ≤ f a ∨ f x ≥ f b) ∧
               (∀ c : ℝ, (∀ x : ℝ, f x ≤ f c ∨ f x ≥ f c) → c = a ∨ c = b) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_l3515_351515


namespace NUMINAMATH_CALUDE_modular_inverse_100_mod_101_l3515_351519

theorem modular_inverse_100_mod_101 : ∃ x : ℕ, x ≤ 100 ∧ (100 * x) % 101 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_100_mod_101_l3515_351519


namespace NUMINAMATH_CALUDE_profit_percent_when_cost_is_quarter_of_selling_price_l3515_351579

/-- If the cost price is 25% of the selling price, then the profit percent is 300%. -/
theorem profit_percent_when_cost_is_quarter_of_selling_price :
  ∀ (selling_price : ℝ) (cost_price : ℝ),
    selling_price > 0 →
    cost_price = 0.25 * selling_price →
    (selling_price - cost_price) / cost_price * 100 = 300 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_when_cost_is_quarter_of_selling_price_l3515_351579


namespace NUMINAMATH_CALUDE_min_dials_for_same_remainder_l3515_351595

/-- A dial is a regular 12-sided polygon with numbers from 1 to 12 -/
def Dial := Fin 12 → Fin 12

/-- A stack of dials -/
def Stack := ℕ → Dial

/-- The sum of numbers in a column of the stack -/
def columnSum (s : Stack) (col : Fin 12) (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => (s i col).val + 1)

/-- Whether all column sums have the same remainder modulo 12 -/
def allColumnsSameRemainder (s : Stack) (n : ℕ) : Prop :=
  ∀ (c₁ c₂ : Fin 12), columnSum s c₁ n % 12 = columnSum s c₂ n % 12

/-- The theorem stating that 12 is the minimum number of dials required -/
theorem min_dials_for_same_remainder :
  ∀ (s : Stack), (∃ (n : ℕ), allColumnsSameRemainder s n) →
  (∃ (m : ℕ), m = 12 ∧ allColumnsSameRemainder s m ∧
    ∀ (k : ℕ), k < m → ¬allColumnsSameRemainder s k) :=
sorry

end NUMINAMATH_CALUDE_min_dials_for_same_remainder_l3515_351595


namespace NUMINAMATH_CALUDE_exam_score_theorem_l3515_351557

theorem exam_score_theorem (total_students : ℕ) 
                            (assigned_day_percentage : ℚ) 
                            (makeup_day_percentage : ℚ) 
                            (makeup_day_average : ℚ) 
                            (class_average : ℚ) :
  total_students = 100 →
  assigned_day_percentage = 70 / 100 →
  makeup_day_percentage = 30 / 100 →
  makeup_day_average = 80 / 100 →
  class_average = 66 / 100 →
  ∃ (assigned_day_average : ℚ),
    assigned_day_average = 60 / 100 ∧
    class_average * total_students = 
      (assigned_day_percentage * total_students * assigned_day_average) +
      (makeup_day_percentage * total_students * makeup_day_average) :=
by sorry

end NUMINAMATH_CALUDE_exam_score_theorem_l3515_351557


namespace NUMINAMATH_CALUDE_fencing_cost_per_meter_l3515_351599

/-- Given a rectangular plot with specified dimensions and total fencing cost,
    calculate the cost per meter of fencing. -/
theorem fencing_cost_per_meter
  (length : ℝ) (breadth : ℝ) (total_cost : ℝ)
  (h1 : length = 60)
  (h2 : breadth = 40)
  (h3 : total_cost = 5300) :
  total_cost / (2 * (length + breadth)) = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_per_meter_l3515_351599


namespace NUMINAMATH_CALUDE_sum_after_removal_l3515_351500

theorem sum_after_removal (numbers : List ℝ) (avg : ℝ) (removed : ℝ) :
  numbers.length = 8 →
  numbers.sum / numbers.length = avg →
  avg = 5.2 →
  removed = 4.6 →
  removed ∈ numbers →
  (numbers.erase removed).sum = 37 := by
  sorry

end NUMINAMATH_CALUDE_sum_after_removal_l3515_351500


namespace NUMINAMATH_CALUDE_skittles_distribution_l3515_351598

theorem skittles_distribution (total_skittles : ℕ) (skittles_per_person : ℕ) (people : ℕ) :
  total_skittles = 20 →
  skittles_per_person = 2 →
  people * skittles_per_person = total_skittles →
  people = 10 := by
  sorry

end NUMINAMATH_CALUDE_skittles_distribution_l3515_351598
