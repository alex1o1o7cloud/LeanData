import Mathlib

namespace NUMINAMATH_CALUDE_sum_ages_theorem_l3474_347401

/-- The sum of Josiah's and Hans' ages after 3 years, given their current ages -/
def sum_ages_after_3_years (hans_current_age : ℕ) (josiah_current_age : ℕ) : ℕ :=
  (hans_current_age + 3) + (josiah_current_age + 3)

/-- Theorem stating the sum of Josiah's and Hans' ages after 3 years -/
theorem sum_ages_theorem (hans_current_age : ℕ) (josiah_current_age : ℕ) 
  (h1 : hans_current_age = 15)
  (h2 : josiah_current_age = 3 * hans_current_age) :
  sum_ages_after_3_years hans_current_age josiah_current_age = 66 := by
sorry

end NUMINAMATH_CALUDE_sum_ages_theorem_l3474_347401


namespace NUMINAMATH_CALUDE_inequality_holds_iff_theta_in_range_l3474_347460

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem inequality_holds_iff_theta_in_range :
  ∀ x θ : ℝ,
  x ≥ 3/2 →
  0 < θ →
  θ < π →
  (f (x / Real.sin θ) - (4 * (Real.sin θ)^2 * f x) ≤ f (x - 1) + 4 * f (Real.sin θ))
  ↔
  π/3 ≤ θ ∧ θ ≤ 2*π/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_theta_in_range_l3474_347460


namespace NUMINAMATH_CALUDE_at_least_thirty_percent_have_all_colors_l3474_347474

/-- Represents the distribution of flags among children -/
structure FlagDistribution where
  total_children : ℕ
  blue_percentage : ℚ
  red_percentage : ℚ
  green_percentage : ℚ

/-- Conditions for the flag distribution problem -/
def valid_distribution (d : FlagDistribution) : Prop :=
  d.blue_percentage = 55 / 100 ∧
  d.red_percentage = 45 / 100 ∧
  d.green_percentage = 30 / 100 ∧
  (d.total_children * 3) % 2 = 0 ∧
  d.blue_percentage + d.red_percentage + d.green_percentage ≥ 1

/-- The main theorem stating that at least 30% of children have all three colors -/
theorem at_least_thirty_percent_have_all_colors (d : FlagDistribution) 
  (h : valid_distribution d) : 
  ∃ (all_colors_percentage : ℚ), 
    all_colors_percentage ≥ 30 / 100 ∧ 
    all_colors_percentage ≤ d.blue_percentage ∧
    all_colors_percentage ≤ d.red_percentage ∧
    all_colors_percentage ≤ d.green_percentage :=
sorry

end NUMINAMATH_CALUDE_at_least_thirty_percent_have_all_colors_l3474_347474


namespace NUMINAMATH_CALUDE_negation_equivalence_l3474_347426

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3474_347426


namespace NUMINAMATH_CALUDE_football_yards_gained_l3474_347419

/-- Represents the yards gained by a football team after an initial loss -/
def yards_gained (initial_loss : ℤ) (final_progress : ℤ) : ℤ :=
  final_progress - initial_loss

/-- Theorem: If a team loses 5 yards and ends with 6 yards of progress, they gained 11 yards -/
theorem football_yards_gained :
  yards_gained (-5) 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_football_yards_gained_l3474_347419


namespace NUMINAMATH_CALUDE_round_trip_speed_ratio_l3474_347485

/-- Proves that given a round trip with specific conditions, the ratio of return speed to outward speed is 2 --/
theorem round_trip_speed_ratio
  (distance : ℝ)
  (total_time : ℝ)
  (return_speed : ℝ)
  (h_distance : distance = 35)
  (h_total_time : total_time = 6)
  (h_return_speed : return_speed = 17.5)
  : (return_speed / ((2 * distance) / total_time - return_speed)) = 2 := by
  sorry

#check round_trip_speed_ratio

end NUMINAMATH_CALUDE_round_trip_speed_ratio_l3474_347485


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_perpendicular_planes_parallel_l3474_347456

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_plane_plane : Plane → Plane → Prop)
variable (parallel_line : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Theorem 1: If two lines are perpendicular to the same plane, they are parallel
theorem perpendicular_lines_parallel (a b : Line) (α : Plane) :
  perp_line_plane a α → perp_line_plane b α → parallel_line a b :=
sorry

-- Theorem 2: If a line is perpendicular to a plane, and that plane is perpendicular to another plane, then the two planes are parallel
theorem perpendicular_planes_parallel (a : Line) (α β : Plane) :
  perp_line_plane a α → perp_plane_plane α β → parallel_plane α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_perpendicular_planes_parallel_l3474_347456


namespace NUMINAMATH_CALUDE_planes_count_theorem_l3474_347457

/-- A straight line in 3D space -/
structure Line3D where
  -- Define necessary properties for a line

/-- A point in 3D space -/
structure Point3D where
  -- Define necessary properties for a point

/-- A plane in 3D space -/
structure Plane3D where
  -- Define necessary properties for a plane

/-- Predicate to check if a point is outside a line -/
def is_outside (p : Point3D) (l : Line3D) : Prop :=
  sorry

/-- Predicate to check if three points are collinear -/
def are_collinear (p1 p2 p3 : Point3D) : Prop :=
  sorry

/-- Function to count the number of unique planes determined by a line and three points -/
def count_planes (l : Line3D) (p1 p2 p3 : Point3D) : Nat :=
  sorry

/-- Theorem stating the possible number of planes -/
theorem planes_count_theorem (l : Line3D) (A B C : Point3D) 
  (h1 : is_outside A l)
  (h2 : is_outside B l)
  (h3 : is_outside C l) :
  (count_planes l A B C = 1) ∨ (count_planes l A B C = 3) ∨ (count_planes l A B C = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_planes_count_theorem_l3474_347457


namespace NUMINAMATH_CALUDE_parabolas_intersection_l3474_347486

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℚ := {-5/3, 0}

/-- First parabola function -/
def f (x : ℚ) : ℚ := 3 * x^2 - 4 * x + 2

/-- Second parabola function -/
def g (x : ℚ) : ℚ := 9 * x^2 + 6 * x + 2

/-- Theorem stating that the two parabolas intersect at the given points -/
theorem parabolas_intersection :
  ∀ x ∈ intersection_x, f x = g x ∧ 
  (x = -5/3 → f x = 17) ∧ 
  (x = 0 → f x = 2) :=
sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l3474_347486


namespace NUMINAMATH_CALUDE_product_of_sum_of_logs_l3474_347422

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem product_of_sum_of_logs (a b : ℝ) (h : log10 a + log10 b = 1) : a * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_of_logs_l3474_347422


namespace NUMINAMATH_CALUDE_bankers_calculation_l3474_347480

/-- Proves that given specific banker's gain, banker's discount, and interest rate, the time period is 3 years -/
theorem bankers_calculation (bankers_gain : ℝ) (bankers_discount : ℝ) (interest_rate : ℝ) :
  bankers_gain = 270 →
  bankers_discount = 1020 →
  interest_rate = 0.12 →
  ∃ (time : ℝ), time = 3 ∧ bankers_discount = (bankers_discount - bankers_gain) * (1 + interest_rate * time) :=
by sorry

end NUMINAMATH_CALUDE_bankers_calculation_l3474_347480


namespace NUMINAMATH_CALUDE_power_function_even_l3474_347447

-- Define the power function f
def f (x : ℝ) : ℝ := x^(2/3)

-- Theorem statement
theorem power_function_even : 
  (f 8 = 4) → (∀ x : ℝ, f (-x) = f x) :=
by
  sorry

end NUMINAMATH_CALUDE_power_function_even_l3474_347447


namespace NUMINAMATH_CALUDE_complete_square_factorization_l3474_347484

theorem complete_square_factorization (a : ℝ) :
  a^2 - 6*a + 8 = (a - 4) * (a - 2) := by sorry

end NUMINAMATH_CALUDE_complete_square_factorization_l3474_347484


namespace NUMINAMATH_CALUDE_incorrect_assignment_l3474_347455

-- Define valid assignment statements
def valid_assignment (stmt : String) : Prop :=
  stmt = "N = N + 1" ∨ stmt = "K = K * K" ∨ stmt = "C = A / B"

-- Define the statement in question
def questionable_statement : String := "C = A(B + D)"

-- Theorem to prove
theorem incorrect_assignment :
  (∀ stmt, valid_assignment stmt → stmt ≠ questionable_statement) →
  ¬(valid_assignment questionable_statement) :=
by
  sorry

end NUMINAMATH_CALUDE_incorrect_assignment_l3474_347455


namespace NUMINAMATH_CALUDE_gumball_multiple_proof_l3474_347496

theorem gumball_multiple_proof :
  ∀ (joanna_initial jacques_initial total_final multiple : ℕ),
    joanna_initial = 40 →
    jacques_initial = 60 →
    total_final = 500 →
    (joanna_initial + joanna_initial * multiple) +
    (jacques_initial + jacques_initial * multiple) = total_final →
    multiple = 4 := by
  sorry

end NUMINAMATH_CALUDE_gumball_multiple_proof_l3474_347496


namespace NUMINAMATH_CALUDE_parabola_intersection_slope_l3474_347498

/-- Parabola defined by y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Point M -/
def point_M : ℝ × ℝ := (-1, 2)

/-- Line passing through focus with slope k -/
def line (k x : ℝ) : ℝ := k * (x - focus.1)

/-- Intersection points of the line and parabola -/
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola p.1 p.2 ∧ p.2 = line k p.1}

/-- Angle AMB is 90 degrees -/
def right_angle (A B : ℝ × ℝ) : Prop :=
  (A.2 - point_M.2) * (B.2 - point_M.2) = -(A.1 - point_M.1) * (B.1 - point_M.1)

theorem parabola_intersection_slope :
  ∀ k : ℝ, ∃ A B : ℝ × ℝ,
    A ∈ intersection_points k ∧
    B ∈ intersection_points k ∧
    A ≠ B ∧
    right_angle A B →
    k = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_slope_l3474_347498


namespace NUMINAMATH_CALUDE_cricket_solution_l3474_347477

def cricket_problem (initial_average : ℝ) (runs_10th_innings : ℕ) : Prop :=
  let total_runs_9_innings := 9 * initial_average
  let total_runs_10_innings := total_runs_9_innings + runs_10th_innings
  let new_average := total_runs_10_innings / 10
  (new_average = initial_average + 8) ∧ (new_average = 128)

theorem cricket_solution :
  ∀ initial_average : ℝ,
  ∃ runs_10th_innings : ℕ,
  cricket_problem initial_average runs_10th_innings ∧
  runs_10th_innings = 200 :=
by sorry

end NUMINAMATH_CALUDE_cricket_solution_l3474_347477


namespace NUMINAMATH_CALUDE_polynomial_differential_equation_l3474_347421

/-- A polynomial of the form a(x + b)^n satisfies (p'(x))^2 = c * p(x) * p''(x) for some constant c -/
theorem polynomial_differential_equation (a b : ℝ) (n : ℕ) (hn : n > 1) (ha : a ≠ 0) :
  ∃ c : ℝ, ∀ x : ℝ,
    let p := fun x => a * (x + b) ^ n
    let p' := fun x => n * a * (x + b) ^ (n - 1)
    let p'' := fun x => n * (n - 1) * a * (x + b) ^ (n - 2)
    (p' x) ^ 2 = c * (p x) * (p'' x) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_differential_equation_l3474_347421


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3474_347406

theorem quadratic_roots_property (a b : ℝ) : 
  (3 * a^2 + 9 * a - 21 = 0) → 
  (3 * b^2 + 9 * b - 21 = 0) → 
  (3 * a - 4) * (2 * b - 2) = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3474_347406


namespace NUMINAMATH_CALUDE_total_fish_bought_l3474_347428

theorem total_fish_bought (goldfish : ℕ) (blue_fish : ℕ) (angelfish : ℕ) (neon_tetras : ℕ)
  (h1 : goldfish = 23)
  (h2 : blue_fish = 15)
  (h3 : angelfish = 8)
  (h4 : neon_tetras = 12) :
  goldfish + blue_fish + angelfish + neon_tetras = 58 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_bought_l3474_347428


namespace NUMINAMATH_CALUDE_negation_equivalence_l3474_347476

theorem negation_equivalence (a b : ℝ) :
  ¬(((a - 2) * (b - 3) = 0) → (a = 2 ∨ b = 3)) ↔
  (((a - 2) * (b - 3) ≠ 0) → (a ≠ 2 ∧ b ≠ 3)) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3474_347476


namespace NUMINAMATH_CALUDE_annas_earnings_is_96_l3474_347478

/-- Calculates Anna's earnings from selling cupcakes given the number of trays, cupcakes per tray, price per cupcake, and fraction sold. -/
def annas_earnings (num_trays : ℕ) (cupcakes_per_tray : ℕ) (price_per_cupcake : ℚ) (fraction_sold : ℚ) : ℚ :=
  (num_trays * cupcakes_per_tray : ℚ) * fraction_sold * price_per_cupcake

/-- Theorem stating that Anna's earnings are $96 given the specific conditions. -/
theorem annas_earnings_is_96 :
  annas_earnings 4 20 2 (3/5) = 96 := by
  sorry

end NUMINAMATH_CALUDE_annas_earnings_is_96_l3474_347478


namespace NUMINAMATH_CALUDE_cubic_identities_l3474_347429

/-- Prove algebraic identities for cubic expressions -/
theorem cubic_identities (x y : ℝ) : 
  ((x + y) * (x^2 - x*y + y^2) = x^3 + y^3) ∧
  ((x + 3) * (x^2 - 3*x + 9) = x^3 + 27) ∧
  ((x - 1) * (x^2 + x + 1) = x^3 - 1) ∧
  ((2*x - 3) * (4*x^2 + 6*x + 9) = 8*x^3 - 27) := by
  sorry


end NUMINAMATH_CALUDE_cubic_identities_l3474_347429


namespace NUMINAMATH_CALUDE_mark_weekly_reading_pages_l3474_347469

-- Define the initial reading time in hours
def initial_reading_time : ℝ := 2

-- Define the percentage increase in reading time
def reading_time_increase : ℝ := 150

-- Define the initial pages read per day
def initial_pages_per_day : ℝ := 100

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Theorem to prove
theorem mark_weekly_reading_pages :
  let new_reading_time := initial_reading_time * (1 + reading_time_increase / 100)
  let new_pages_per_day := initial_pages_per_day * (new_reading_time / initial_reading_time)
  let weekly_pages := new_pages_per_day * days_in_week
  weekly_pages = 1750 := by sorry

end NUMINAMATH_CALUDE_mark_weekly_reading_pages_l3474_347469


namespace NUMINAMATH_CALUDE_triangle_right_angle_l3474_347488

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_right_angle (t : Triangle) 
  (h : t.b - t.a * Real.cos t.B = t.a * Real.cos t.C - t.c) : 
  t.A = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_right_angle_l3474_347488


namespace NUMINAMATH_CALUDE_students_above_eight_l3474_347450

theorem students_above_eight (total : ℕ) (below_eight : ℕ) (eight : ℕ) (above_eight : ℕ) : 
  total = 80 →
  below_eight = total / 4 →
  eight = 36 →
  above_eight = 2 * eight / 3 →
  above_eight = 24 := by
sorry

end NUMINAMATH_CALUDE_students_above_eight_l3474_347450


namespace NUMINAMATH_CALUDE_volleyball_score_ratio_l3474_347445

theorem volleyball_score_ratio :
  let lizzie_score : ℕ := 4
  let nathalie_score : ℕ := lizzie_score + 3
  let combined_score : ℕ := lizzie_score + nathalie_score
  let team_total : ℕ := 50
  let teammates_score : ℕ := 17
  ∃ (m : ℕ), 
    m * combined_score = team_total - lizzie_score - nathalie_score - teammates_score ∧
    m * combined_score = 2 * combined_score :=
by
  sorry

#check volleyball_score_ratio

end NUMINAMATH_CALUDE_volleyball_score_ratio_l3474_347445


namespace NUMINAMATH_CALUDE_interest_rate_calculation_interest_rate_proof_l3474_347415

theorem interest_rate_calculation (initial_investment : ℝ) 
  (first_rate : ℝ) (first_duration : ℝ) (second_duration : ℝ) 
  (final_value : ℝ) : ℝ :=
  let first_growth := initial_investment * (1 + first_rate * first_duration / 12)
  let second_rate := ((final_value / first_growth - 1) * 12 / second_duration) * 100
  second_rate

theorem interest_rate_proof (initial_investment : ℝ) 
  (first_rate : ℝ) (first_duration : ℝ) (second_duration : ℝ) 
  (final_value : ℝ) :
  initial_investment = 12000 ∧ 
  first_rate = 0.08 ∧ 
  first_duration = 3 ∧ 
  second_duration = 3 ∧ 
  final_value = 12980 →
  interest_rate_calculation initial_investment first_rate first_duration second_duration final_value = 24 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_interest_rate_proof_l3474_347415


namespace NUMINAMATH_CALUDE_bridge_problem_l3474_347475

/-- A graph representing the bridge system. -/
structure BridgeGraph where
  /-- The set of nodes (islands) in the graph. -/
  nodes : Finset (Fin 4)
  /-- The set of edges (bridges) in the graph. -/
  edges : Finset (Fin 4 × Fin 4)
  /-- The degree of each node. -/
  degree : Fin 4 → Nat
  /-- Condition that node 0 (A) has degree 3. -/
  degree_A : degree 0 = 3
  /-- Condition that node 1 (B) has degree 5. -/
  degree_B : degree 1 = 5
  /-- Condition that node 2 (C) has degree 3. -/
  degree_C : degree 2 = 3
  /-- Condition that node 3 (D) has degree 3. -/
  degree_D : degree 3 = 3
  /-- The total number of edges is 9. -/
  edge_count : edges.card = 9

/-- The number of Eulerian paths in the bridge graph. -/
def countEulerianPaths (g : BridgeGraph) : Nat :=
  sorry

/-- Theorem stating that the number of Eulerian paths is 132. -/
theorem bridge_problem (g : BridgeGraph) : countEulerianPaths g = 132 :=
  sorry

end NUMINAMATH_CALUDE_bridge_problem_l3474_347475


namespace NUMINAMATH_CALUDE_find_number_l3474_347452

theorem find_number : ∃ n : ℝ, 7 * n - 15 = 2 * n + 10 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3474_347452


namespace NUMINAMATH_CALUDE_simple_interest_time_l3474_347432

/-- Simple interest calculation -/
theorem simple_interest_time (principal rate interest : ℝ) :
  principal > 0 →
  rate > 0 →
  interest > 0 →
  (interest * 100) / (principal * rate) = 2 →
  principal = 400 →
  rate = 12.5 →
  interest = 100 →
  (interest * 100) / (principal * rate) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_time_l3474_347432


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_plus_ten_l3474_347427

theorem gcf_lcm_sum_plus_ten (a b : ℕ) (h1 : a = 8) (h2 : b = 12) :
  Nat.gcd a b + Nat.lcm a b + 10 = 38 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_plus_ten_l3474_347427


namespace NUMINAMATH_CALUDE_not_square_sum_of_square_and_divisor_l3474_347489

theorem not_square_sum_of_square_and_divisor (A B : ℕ) (hA : A ≠ 0) (hAsq : ∃ n : ℕ, A = n^2) (hB : B ∣ A) :
  ¬ ∃ m : ℕ, A + B = m^2 := by
sorry

end NUMINAMATH_CALUDE_not_square_sum_of_square_and_divisor_l3474_347489


namespace NUMINAMATH_CALUDE_ellipse_equation_l3474_347446

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line with slope m passing through point p -/
structure Line where
  m : ℝ
  p : Point

/-- The theorem statement -/
theorem ellipse_equation (E : Ellipse) (F : Point) (l : Line) (M : Point) :
  F.x = 3 ∧ F.y = 0 ∧  -- Right focus at (3,0)
  l.m = 1/2 ∧ l.p = F ∧  -- Line with slope 1/2 passing through F
  M.x = 1 ∧ M.y = -1 ∧  -- Midpoint at (1,-1)
  (∃ A B : Point, A ≠ B ∧
    (A.x^2 / E.a^2 + A.y^2 / E.b^2 = 1) ∧
    (B.x^2 / E.a^2 + B.y^2 / E.b^2 = 1) ∧
    (A.y - F.y = l.m * (A.x - F.x)) ∧
    (B.y - F.y = l.m * (B.x - F.x)) ∧
    M.x = (A.x + B.x) / 2 ∧
    M.y = (A.y + B.y) / 2) →
  E.a^2 = 18 ∧ E.b^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3474_347446


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3474_347434

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = 1 / x}
def B : Set ℝ := {x | ∃ y, y = Real.log x}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | x ≠ 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3474_347434


namespace NUMINAMATH_CALUDE_problem_statement_l3474_347423

theorem problem_statement (x y z : ℚ) : 
  x = 1/3 → y = 2/3 → z = x * y → 3 * x^2 * y^5 * z^3 = 768/1594323 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3474_347423


namespace NUMINAMATH_CALUDE_range_of_a_l3474_347412

open Set Real

def A : Set ℝ := {x | 1 ≤ x ∧ x < 3}

def B (a : ℝ) : Set ℝ := {x | x^2 - a*x ≤ x - a}

theorem range_of_a :
  ∀ a : ℝ, (B a ⊆ A) ↔ (1 ≤ a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3474_347412


namespace NUMINAMATH_CALUDE_equation_solution_l3474_347400

theorem equation_solution :
  ∃ n : ℚ, (22 + Real.sqrt (-4 + 18 * n) = 24) ∧ n = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3474_347400


namespace NUMINAMATH_CALUDE_condition_implies_linear_l3474_347443

/-- A function satisfying the given inequality condition -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ (a b p : ℝ), f (p * a + (1 - p) * b) ≤ p * f a + (1 - p) * f b

/-- A linear function -/
def IsLinear (f : ℝ → ℝ) : Prop :=
  ∃ (A B : ℝ), ∀ x, f x = A * x + B

/-- Theorem: If a function satisfies the condition, then it is linear -/
theorem condition_implies_linear (f : ℝ → ℝ) :
  SatisfiesCondition f → IsLinear f := by
  sorry

end NUMINAMATH_CALUDE_condition_implies_linear_l3474_347443


namespace NUMINAMATH_CALUDE_circle_and_line_problem_l3474_347494

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 3)^2 + (y + 2)^2 = 25

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the line m
def line_m (x y : ℝ) : Prop := x = 1 ∨ (5/12)*x - y + 43/12 = 0

-- Theorem statement
theorem circle_and_line_problem :
  -- Given conditions
  (circle_C 0 2) ∧ 
  (circle_C 2 (-2)) ∧ 
  (∃ (x y : ℝ), circle_C x y ∧ line_l x y) ∧
  (line_m 1 4) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    circle_C x₁ y₁ ∧ 
    circle_C x₂ y₂ ∧ 
    line_m x₁ y₁ ∧ 
    line_m x₂ y₂ ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 36) →
  -- Conclusion
  (∀ (x y : ℝ), circle_C x y ↔ (x + 3)^2 + (y + 2)^2 = 25) ∧
  (∀ (x y : ℝ), line_m x y ↔ (x = 1 ∨ (5/12)*x - y + 43/12 = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_and_line_problem_l3474_347494


namespace NUMINAMATH_CALUDE_soccer_expansion_l3474_347461

/-- The total number of kids playing soccer after expansion -/
def total_kids (initial : ℕ) (friends_per_kid : ℕ) : ℕ :=
  initial + initial * friends_per_kid

/-- Theorem stating that with 14 initial kids and 3 friends per kid, the total is 56 -/
theorem soccer_expansion : total_kids 14 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_soccer_expansion_l3474_347461


namespace NUMINAMATH_CALUDE_fold_square_crease_l3474_347403

/-- Given a square ABCD with side length 18 cm, if point B is folded to point E on AD
    such that DE = 6 cm, and the resulting crease intersects AB at point F,
    then the length of FB is 13 cm. -/
theorem fold_square_crease (A B C D E F : ℝ × ℝ) : 
  -- Square ABCD with side length 18
  (A = (0, 0) ∧ B = (18, 0) ∧ C = (18, 18) ∧ D = (0, 18)) →
  -- E is on AD and DE = 6
  (E.1 = 0 ∧ E.2 = 12) →
  -- F is on AB
  (F.2 = 0) →
  -- F is on the perpendicular bisector of BE
  (F.2 - 6 = (3/2) * (F.1 - 9)) →
  -- The length of FB is 13
  Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2) = 13 :=
by sorry

end NUMINAMATH_CALUDE_fold_square_crease_l3474_347403


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3474_347411

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 6)
  (h_a3 : a 3 = 2) :
  a 5 = -2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3474_347411


namespace NUMINAMATH_CALUDE_bruce_goals_l3474_347410

theorem bruce_goals (bruce_goals : ℕ) 
  (michael_goals : ℕ)
  (h1 : michael_goals = 3 * bruce_goals)
  (h2 : bruce_goals + michael_goals = 16) : 
  bruce_goals = 4 := by
sorry

end NUMINAMATH_CALUDE_bruce_goals_l3474_347410


namespace NUMINAMATH_CALUDE_expand_product_l3474_347436

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12*x + 27 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3474_347436


namespace NUMINAMATH_CALUDE_cos_12_18_minus_sin_12_18_l3474_347448

theorem cos_12_18_minus_sin_12_18 :
  Real.cos (12 * π / 180) * Real.cos (18 * π / 180) - 
  Real.sin (12 * π / 180) * Real.sin (18 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_12_18_minus_sin_12_18_l3474_347448


namespace NUMINAMATH_CALUDE_chair_cost_l3474_347430

def total_spent : ℕ := 56
def table_cost : ℕ := 34
def num_chairs : ℕ := 2

theorem chair_cost (chair_cost : ℕ) 
  (h1 : chair_cost * num_chairs + table_cost = total_spent) 
  (h2 : chair_cost > 0) : chair_cost = 11 := by
  sorry

end NUMINAMATH_CALUDE_chair_cost_l3474_347430


namespace NUMINAMATH_CALUDE_board_number_remainder_l3474_347417

theorem board_number_remainder (n a b c d : ℕ) : 
  n = 102 * a + b ∧ 
  n = 103 * c + d ∧ 
  a + d = 20 ∧ 
  b < 102 →
  b = 20 := by sorry

end NUMINAMATH_CALUDE_board_number_remainder_l3474_347417


namespace NUMINAMATH_CALUDE_village_households_l3474_347437

/-- The number of households in a village where:
    - Each household uses 150 litres of water per month
    - 6000 litres of water lasts for 4 months for all households
-/
def number_of_households : ℕ := 10

/-- Water usage per household per month in litres -/
def water_per_household_per_month : ℕ := 150

/-- Total water available in litres -/
def total_water : ℕ := 6000

/-- Number of months the water lasts -/
def months : ℕ := 4

theorem village_households : 
  number_of_households * water_per_household_per_month * months = total_water :=
sorry

end NUMINAMATH_CALUDE_village_households_l3474_347437


namespace NUMINAMATH_CALUDE_school_ratio_problem_l3474_347407

/-- Given a school with 300 students, where the ratio of boys to girls is x : y,
    prove that if the number of boys is increased by z such that the number of girls
    becomes x% of the total, then z = 300 - 3x - 300x / (x + y). -/
theorem school_ratio_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  ∃ z : ℝ, z = 300 - 3*x - 300*x / (x + y) := by
  sorry


end NUMINAMATH_CALUDE_school_ratio_problem_l3474_347407


namespace NUMINAMATH_CALUDE_inequality_proof_l3474_347464

/-- Proves that given a = 0.1e^0.1, b = 1/9, and c = -ln 0.9, the inequality c < a < b holds -/
theorem inequality_proof (a b c : ℝ) 
  (ha : a = 0.1 * Real.exp 0.1) 
  (hb : b = 1 / 9) 
  (hc : c = -Real.log 0.9) : 
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3474_347464


namespace NUMINAMATH_CALUDE_count_numbers_satisfying_condition_l3474_347463

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Define the property we're looking for
def satisfiesCondition (n : ℕ) : Prop :=
  n > 0 ∧ n < 1000 ∧ n = 7 * sumOfDigits n

-- State the theorem
theorem count_numbers_satisfying_condition :
  ∃ (S : Finset ℕ), S.card = 3 ∧ ∀ n, n ∈ S ↔ satisfiesCondition n :=
sorry

end NUMINAMATH_CALUDE_count_numbers_satisfying_condition_l3474_347463


namespace NUMINAMATH_CALUDE_special_function_value_l3474_347467

/-- A function satisfying certain properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 1) = f (1 - x)) ∧ 
  (∀ x, f (x + 2) = f (x + 1) - f x) ∧
  (f 1 = 1/2)

/-- Theorem stating that for any function satisfying the special properties, f(2024) = 1/4 -/
theorem special_function_value (f : ℝ → ℝ) (h : special_function f) : f 2024 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l3474_347467


namespace NUMINAMATH_CALUDE_sum_frequencies_equals_total_data_l3474_347453

/-- Represents a frequency distribution table -/
structure FrequencyDistributionTable where
  groups : List ℕ  -- List of frequencies for each group
  total_data : ℕ   -- Total number of data points

/-- 
Theorem: In a frequency distribution table, the sum of the frequencies 
of all groups is equal to the total number of data points.
-/
theorem sum_frequencies_equals_total_data (table : FrequencyDistributionTable) : 
  table.groups.sum = table.total_data := by
  sorry


end NUMINAMATH_CALUDE_sum_frequencies_equals_total_data_l3474_347453


namespace NUMINAMATH_CALUDE_bertha_family_women_without_daughters_l3474_347490

/-- Represents a woman in Bertha's family tree -/
structure Woman where
  has_daughters : Bool

/-- Bertha's family tree -/
structure Family where
  daughters : Finset Woman
  granddaughters : Finset Woman

/-- The number of women who have no daughters in Bertha's family -/
def num_women_without_daughters (f : Family) : Nat :=
  (f.daughters.filter (fun w => !w.has_daughters)).card +
  (f.granddaughters.filter (fun w => !w.has_daughters)).card

theorem bertha_family_women_without_daughters :
  ∃ f : Family,
    f.daughters.card = 8 ∧
    (∀ d ∈ f.daughters, d.has_daughters) ∧
    (∀ d ∈ f.daughters, (f.granddaughters.filter (fun g => g.has_daughters.not)).card = 4) ∧
    (f.daughters.card + f.granddaughters.card = 40) ∧
    num_women_without_daughters f = 32 := by
  sorry

end NUMINAMATH_CALUDE_bertha_family_women_without_daughters_l3474_347490


namespace NUMINAMATH_CALUDE_matrix_inverse_from_eigenvectors_l3474_347479

theorem matrix_inverse_from_eigenvectors :
  ∀ (a b c d : ℝ),
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A.mulVec ![1, 1] = (6 : ℝ) • ![1, 1]) →
  (A.mulVec ![3, -2] = (1 : ℝ) • ![3, -2]) →
  A⁻¹ = !![2/3, -1/2; -1/3, 1/2] :=
by sorry

end NUMINAMATH_CALUDE_matrix_inverse_from_eigenvectors_l3474_347479


namespace NUMINAMATH_CALUDE_derivative_of_fraction_l3474_347431

open Real

theorem derivative_of_fraction (x : ℝ) (h : x > 0) :
  deriv (λ x => (1 - log x) / (1 + log x)) x = -2 / (x * (1 + log x)^2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_fraction_l3474_347431


namespace NUMINAMATH_CALUDE_square_of_9_divided_by_cube_root_of_125_remainder_l3474_347459

theorem square_of_9_divided_by_cube_root_of_125_remainder (n m q r : ℕ) : 
  n = 9^2 → 
  m = 5 → 
  n = m * q + r → 
  r < m → 
  r = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_square_of_9_divided_by_cube_root_of_125_remainder_l3474_347459


namespace NUMINAMATH_CALUDE_min_sum_given_product_l3474_347405

theorem min_sum_given_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y - (x + y) = 1) :
  x + y ≥ 2 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_product_l3474_347405


namespace NUMINAMATH_CALUDE_min_value_of_sum_l3474_347416

theorem min_value_of_sum (x₁ x₂ x₃ x₄ x₅ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) :
  let f := (x₁ + x₃) / (x₅ + 2*x₂ + 3*x₄) + (x₂ + x₄) / (x₁ + 2*x₃ + 3*x₅) + 
           (x₃ + x₅) / (x₂ + 2*x₄ + 3*x₁) + (x₄ + x₁) / (x₃ + 2*x₅ + 3*x₂) + 
           (x₅ + x₂) / (x₄ + 2*x₁ + 3*x₃)
  f ≥ 5/3 ∧ (f = 5/3 ↔ x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l3474_347416


namespace NUMINAMATH_CALUDE_pyramid_volume_l3474_347440

theorem pyramid_volume (base_length : Real) (base_width : Real) (height : Real) :
  base_length = 1 → base_width = 1/4 → height = 1 →
  (1/3) * (base_length * base_width) * height = 1/12 := by
sorry

end NUMINAMATH_CALUDE_pyramid_volume_l3474_347440


namespace NUMINAMATH_CALUDE_fifteen_fishers_tomorrow_l3474_347473

/-- Represents the fishing schedule in a coastal village -/
structure FishingSchedule where
  daily : ℕ              -- Number of people fishing daily
  everyOtherDay : ℕ      -- Number of people fishing every other day
  everyThreeDay : ℕ      -- Number of people fishing every three days
  yesterday : ℕ          -- Number of people who fished yesterday
  today : ℕ              -- Number of people fishing today

/-- Calculates the number of people who will fish tomorrow given a FishingSchedule -/
def tomorrowFishers (schedule : FishingSchedule) : ℕ :=
  schedule.daily +
  schedule.everyThreeDay +
  (schedule.everyOtherDay - (schedule.yesterday - schedule.daily))

/-- Theorem stating that given the specific fishing schedule, 15 people will fish tomorrow -/
theorem fifteen_fishers_tomorrow 
  (schedule : FishingSchedule)
  (h1 : schedule.daily = 7)
  (h2 : schedule.everyOtherDay = 8)
  (h3 : schedule.everyThreeDay = 3)
  (h4 : schedule.yesterday = 12)
  (h5 : schedule.today = 10) :
  tomorrowFishers schedule = 15 := by
  sorry

#eval tomorrowFishers { daily := 7, everyOtherDay := 8, everyThreeDay := 3, yesterday := 12, today := 10 }

end NUMINAMATH_CALUDE_fifteen_fishers_tomorrow_l3474_347473


namespace NUMINAMATH_CALUDE_circle_line_intersection_l3474_347497

/-- A circle C with center (a, 0) and radius r -/
structure Circle where
  a : ℝ
  r : ℝ
  r_pos : r > 0

/-- A line with slope k passing through (-1, 0) -/
structure Line where
  k : ℝ

/-- Theorem: Given a circle C and a line l satisfying certain conditions, 
    the dot product of OA and OB is -(26 + 9√2) / 5 -/
theorem circle_line_intersection 
  (C : Circle) 
  (l : Line) 
  (h1 : C.r = |C.a - 2 * Real.sqrt 2| / Real.sqrt 2)  -- C is tangent to x + y - 2√2 = 0
  (h2 : 4 * Real.sqrt 2 = 2 * Real.sqrt (C.r^2 - (|C.a| / Real.sqrt 2)^2))  -- chord length on y = x is 4√2
  (h3 : ∃ (m : ℝ), m / l.k^2 = -3 - Real.sqrt 2)  -- condition on slopes product
  : ∃ (A B : ℝ × ℝ), 
    (A.1 - C.a)^2 + A.2^2 = C.r^2 ∧   -- A is on circle C
    (B.1 - C.a)^2 + B.2^2 = C.r^2 ∧   -- B is on circle C
    A.2 = l.k * (A.1 + 1) ∧           -- A is on line l
    B.2 = l.k * (B.1 + 1) ∧           -- B is on line l
    (A.1 - C.a) * (B.1 - C.a) + A.2 * B.2 = -(26 + 9 * Real.sqrt 2) / 5  -- OA · OB
    := by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l3474_347497


namespace NUMINAMATH_CALUDE_inequality_chain_l3474_347420

theorem inequality_chain (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (2 * a * b) / (a + b) < Real.sqrt (a * b) ∧
  Real.sqrt (a * b) < (a + b) / 2 ∧
  (a + b) / 2 < Real.sqrt ((a^2 + b^2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_chain_l3474_347420


namespace NUMINAMATH_CALUDE_fence_cost_square_plot_l3474_347433

theorem fence_cost_square_plot (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 56) :
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_cost := perimeter * price_per_foot
  total_cost = 3808 := by
sorry

end NUMINAMATH_CALUDE_fence_cost_square_plot_l3474_347433


namespace NUMINAMATH_CALUDE_log_expression_equals_negative_one_l3474_347483

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_negative_one :
  log10 (5/2) + 2 * log10 2 - (1/2)⁻¹ = -1 := by sorry

end NUMINAMATH_CALUDE_log_expression_equals_negative_one_l3474_347483


namespace NUMINAMATH_CALUDE_apple_mango_equivalence_l3474_347465

theorem apple_mango_equivalence (apple_value mango_value : ℝ) :
  (5 / 4 * 16 * apple_value = 10 * mango_value) →
  (3 / 4 * 12 * apple_value = 4.5 * mango_value) := by
  sorry

end NUMINAMATH_CALUDE_apple_mango_equivalence_l3474_347465


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l3474_347439

theorem unique_solution_quadratic_inequality (a : ℝ) :
  (∃! x : ℝ, 0 ≤ x^2 + a*x + 5 ∧ x^2 + a*x + 5 ≤ 4) ↔ (a = 2 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l3474_347439


namespace NUMINAMATH_CALUDE_puzzle_time_relationship_l3474_347409

/-- Represents the time needed to complete a puzzle given the gluing rate -/
def puzzle_completion_time (initial_pieces : ℕ) (pieces_per_minute : ℕ) : ℕ :=
  (initial_pieces - 1) / (pieces_per_minute - 1)

/-- Theorem stating the relationship between puzzle completion times
    with different gluing rates -/
theorem puzzle_time_relationship :
  ∀ (initial_pieces : ℕ),
    initial_pieces > 1 →
    puzzle_completion_time initial_pieces 2 = 120 →
    puzzle_completion_time initial_pieces 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_time_relationship_l3474_347409


namespace NUMINAMATH_CALUDE_circle_parabola_tangency_l3474_347471

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a parabola with equation y = x^2 + 1 -/
def Parabola : Point → Prop :=
  fun p => p.y = p.x^2 + 1

/-- Check if a circle is tangent to the parabola at two points -/
def IsTangent (c : Circle) (p1 p2 : Point) : Prop :=
  Parabola p1 ∧ Parabola p2 ∧
  (c.center.x - p1.x)^2 + (c.center.y - p1.y)^2 = c.radius^2 ∧
  (c.center.x - p2.x)^2 + (c.center.y - p2.y)^2 = c.radius^2

/-- The main theorem -/
theorem circle_parabola_tangency 
  (c : Circle) (p1 p2 : Point) (h : IsTangent c p1 p2) :
  c.center.y - p1.y = p1.x^2 - 1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_parabola_tangency_l3474_347471


namespace NUMINAMATH_CALUDE_election_votes_l3474_347468

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) (winner_percent : ℚ) :
  total_votes = 7500 →
  invalid_percent = 1/5 →
  winner_percent = 11/20 →
  ∃ (other_candidate_votes : ℕ), other_candidate_votes = 2700 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l3474_347468


namespace NUMINAMATH_CALUDE_parabola_properties_l3474_347493

-- Define the parabola function
def f (x : ℝ) : ℝ := (x - 1)^2 - 3

theorem parabola_properties :
  -- 1. The parabola opens upwards
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f ((x₁ + x₂) / 2) < (f x₁ + f x₂) / 2) ∧
  -- 2. The parabola intersects the x-axis at two distinct points
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  -- 3. The minimum value of y is -3 and occurs when x = 1
  (∀ x : ℝ, f x ≥ -3) ∧ (f 1 = -3) ∧
  -- 4. There exists an x > 1 such that y ≤ 0
  (∃ x : ℝ, x > 1 ∧ f x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l3474_347493


namespace NUMINAMATH_CALUDE_ellipse_max_value_l3474_347487

theorem ellipse_max_value (x y : ℝ) :
  x^2 / 9 + y^2 = 1 → x + 3 * y ≤ 3 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_max_value_l3474_347487


namespace NUMINAMATH_CALUDE_basketball_shot_probability_l3474_347470

theorem basketball_shot_probability :
  let p1 : ℚ := 2/3  -- Probability of making the first shot
  let p2_success : ℚ := 2/3  -- Probability of making the second shot if the first shot was successful
  let p2_fail : ℚ := 1/3  -- Probability of making the second shot if the first shot failed
  let p3_success : ℚ := 2/3  -- Probability of making the third shot after making the second
  let p3_fail : ℚ := 1/3  -- Probability of making the third shot after missing the second
  
  (p1 * p2_success * p3_success) +  -- Case 1: Make all three shots
  (p1 * (1 - p2_success) * p3_fail) +  -- Case 2: Make first, miss second, make third
  ((1 - p1) * p2_fail * p3_success) +  -- Case 3: Miss first, make second and third
  ((1 - p1) * (1 - p2_fail) * p3_fail) = 14/27  -- Case 4: Miss first and second, make third
  := by sorry

end NUMINAMATH_CALUDE_basketball_shot_probability_l3474_347470


namespace NUMINAMATH_CALUDE_congruence_solution_l3474_347442

theorem congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 31 ∧ -250 ≡ n [ZMOD 31] ∧ n = 29 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l3474_347442


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3474_347495

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 10

/-- The point where the minimum occurs -/
def min_point : ℝ := -4

theorem quadratic_minimum :
  ∀ x : ℝ, f x ≥ f min_point :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3474_347495


namespace NUMINAMATH_CALUDE_quadratic_inequality_sets_l3474_347491

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 > 0}
def B (m : ℝ) : Set ℝ := {x | m*x^2 - (m+2)*x + 2 < 0}

-- State the theorem
theorem quadratic_inequality_sets :
  (∀ m : ℝ, B m ⊆ (Set.univ \ A) ↔ m ∈ Set.Icc 1 2) ∧
  (∀ m : ℝ, (A ∩ B m).Nonempty ↔ m ∈ Set.Ioi 2 ∪ Set.Iio 1) ∧
  (∀ m : ℝ, A ∪ B m = A ↔ m ∈ Set.Ici 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_sets_l3474_347491


namespace NUMINAMATH_CALUDE_sequence_has_repeating_pair_l3474_347414

def is_valid_sequence (a : Fin 99 → Fin 10) : Prop :=
  ∀ n : Fin 98, (a n = 1 → a (n + 1) ≠ 2) ∧ (a n = 3 → a (n + 1) ≠ 4)

theorem sequence_has_repeating_pair (a : Fin 99 → Fin 10) (h : is_valid_sequence a) :
  ∃ k l : Fin 98, k ≠ l ∧ a k = a l ∧ a (k + 1) = a (l + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_has_repeating_pair_l3474_347414


namespace NUMINAMATH_CALUDE_sqrt_point_zero_nine_equals_point_three_l3474_347444

theorem sqrt_point_zero_nine_equals_point_three :
  Real.sqrt 0.09 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_point_zero_nine_equals_point_three_l3474_347444


namespace NUMINAMATH_CALUDE_product_nonpositive_implies_factor_nonpositive_l3474_347454

theorem product_nonpositive_implies_factor_nonpositive (a b : ℝ) : 
  a * b ≤ 0 → a ≤ 0 ∨ b ≤ 0 := by sorry

end NUMINAMATH_CALUDE_product_nonpositive_implies_factor_nonpositive_l3474_347454


namespace NUMINAMATH_CALUDE_floor_sqrt_20_squared_l3474_347492

theorem floor_sqrt_20_squared : ⌊Real.sqrt 20⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_20_squared_l3474_347492


namespace NUMINAMATH_CALUDE_concentric_circles_theorem_l3474_347472

/-- Two concentric circles with radii R and r, where R > r -/
structure ConcentricCircles (R r : ℝ) :=
  (h : R > r)

/-- Points on the circles -/
structure Points (R r : ℝ) extends ConcentricCircles R r :=
  (P : ℝ × ℝ)
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)
  (hP : P.1^2 + P.2^2 = r^2)
  (hA : A.1^2 + A.2^2 = r^2)
  (hB : B.1^2 + B.2^2 = R^2)
  (hC : C.1^2 + C.2^2 = R^2)
  (hPerp : (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = 0)

/-- The theorem to be proved -/
theorem concentric_circles_theorem (R r : ℝ) (pts : Points R r) :
  let BC := (pts.B.1 - pts.C.1)^2 + (pts.B.2 - pts.C.2)^2
  let CA := (pts.C.1 - pts.A.1)^2 + (pts.C.2 - pts.A.2)^2
  let AB := (pts.A.1 - pts.B.1)^2 + (pts.A.2 - pts.B.2)^2
  let midpoint := ((pts.A.1 + pts.B.1) / 2, (pts.A.2 + pts.B.2) / 2)
  (BC + CA + AB = 6 * R^2 + 2 * r^2) ∧
  ((midpoint.1 + r/2)^2 + midpoint.2^2 = (R/2)^2) :=
sorry

end NUMINAMATH_CALUDE_concentric_circles_theorem_l3474_347472


namespace NUMINAMATH_CALUDE_one_of_each_color_probability_l3474_347499

def total_marbles : ℕ := 9
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 3
def green_marbles : ℕ := 3
def selected_marbles : ℕ := 3

def probability_one_of_each_color : ℚ := 9 / 28

theorem one_of_each_color_probability :
  probability_one_of_each_color = 
    (red_marbles * blue_marbles * green_marbles : ℚ) / 
    (Nat.choose total_marbles selected_marbles) :=
by sorry

end NUMINAMATH_CALUDE_one_of_each_color_probability_l3474_347499


namespace NUMINAMATH_CALUDE_sine_amplitude_l3474_347404

theorem sine_amplitude (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x, a * Real.sin (b * x) ≤ 3) ∧ (∃ x, a * Real.sin (b * x) = 3) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_sine_amplitude_l3474_347404


namespace NUMINAMATH_CALUDE_unique_counterexample_l3474_347402

-- Define geometric figures
inductive GeometricFigure
| Line
| Plane

-- Define spatial relationships
def perpendicular (a b : GeometricFigure) : Prop := sorry
def parallel (a b : GeometricFigure) : Prop := sorry

-- Define the proposition
def proposition (x y z : GeometricFigure) : Prop :=
  (perpendicular x y ∧ parallel y z) → perpendicular x z

-- Theorem statement
theorem unique_counterexample :
  ∀ x y z : GeometricFigure,
    ¬proposition x y z ↔ 
      x = GeometricFigure.Line ∧ 
      y = GeometricFigure.Line ∧ 
      z = GeometricFigure.Plane :=
sorry

end NUMINAMATH_CALUDE_unique_counterexample_l3474_347402


namespace NUMINAMATH_CALUDE_converse_proposition_l3474_347413

theorem converse_proposition : 
  (∀ x : ℝ, x > 0 → x^2 - 1 > 0) ↔ 
  (∀ x : ℝ, x^2 - 1 > 0 → x > 0) :=
by sorry

end NUMINAMATH_CALUDE_converse_proposition_l3474_347413


namespace NUMINAMATH_CALUDE_max_value_of_a_l3474_347435

theorem max_value_of_a (a b c : ℝ) : 
  a^2 - b*c - 8*a + 7 = 0 → 
  b^2 + c^2 + b*c - 6*a + 6 = 0 → 
  a ≤ 9 ∧ ∃ b c : ℝ, a^2 - b*c - 8*a + 7 = 0 ∧ b^2 + c^2 + b*c - 6*a + 6 = 0 ∧ a = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3474_347435


namespace NUMINAMATH_CALUDE_expression_value_l3474_347438

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 4) : 
  (x^4 + 3*x^2 - 2*y + 2*y^2) / 6 = 22 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3474_347438


namespace NUMINAMATH_CALUDE_denise_crayon_sharing_l3474_347462

/-- The number of crayons Denise has -/
def total_crayons : ℕ := 210

/-- The number of crayons each friend gets -/
def crayons_per_friend : ℕ := 7

/-- The number of friends Denise shares crayons with -/
def number_of_friends : ℕ := total_crayons / crayons_per_friend

theorem denise_crayon_sharing :
  number_of_friends = 30 := by sorry

end NUMINAMATH_CALUDE_denise_crayon_sharing_l3474_347462


namespace NUMINAMATH_CALUDE_not_all_zero_deriv_is_critical_point_l3474_347441

open Set
open Function
open Filter

/-- A point x₀ is a critical point of a differentiable function f if f'(x₀) = 0 
    and f'(x) changes sign in any neighborhood of x₀. -/
def IsCriticalPoint (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  DifferentiableAt ℝ f x₀ ∧ 
  (deriv f) x₀ = 0 ∧
  ∀ ε > 0, ∃ x₁ x₂, x₁ < x₀ ∧ x₀ < x₂ ∧ 
    abs (x₁ - x₀) < ε ∧ abs (x₂ - x₀) < ε ∧
    (deriv f) x₁ * (deriv f) x₂ < 0

/-- The statement "For all differentiable functions f, if f'(x₀) = 0, 
    then x₀ is a critical point of f" is false. -/
theorem not_all_zero_deriv_is_critical_point :
  ¬ (∀ (f : ℝ → ℝ) (x₀ : ℝ), DifferentiableAt ℝ f x₀ → (deriv f) x₀ = 0 → IsCriticalPoint f x₀) :=
by sorry

end NUMINAMATH_CALUDE_not_all_zero_deriv_is_critical_point_l3474_347441


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_negation_l3474_347466

theorem sufficient_not_necessary_negation 
  (p q : Prop) 
  (h_suff : p → q) 
  (h_not_nec : ¬(q → p)) : 
  (¬q → ¬p) ∧ ¬(¬p → ¬q) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_negation_l3474_347466


namespace NUMINAMATH_CALUDE_complex_square_theorem_l3474_347425

theorem complex_square_theorem (z : ℂ) (h : z * Complex.I = 2 + Complex.I) : z^2 = -3 - 4 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_theorem_l3474_347425


namespace NUMINAMATH_CALUDE_letter_A_value_l3474_347424

-- Define the letter values as variables
variable (M A T E : ℤ)

-- Define the known value of H
def H : ℤ := 12

-- Define the word values as given in the problem
def MATH : ℤ := M + A + T + H
def TEAM : ℤ := T + E + A + M
def MEET : ℤ := M + E + E + T

-- State the theorem
theorem letter_A_value 
  (h1 : MATH = 40)
  (h2 : TEAM = 50)
  (h3 : MEET = 44) :
  A = 28 := by sorry

end NUMINAMATH_CALUDE_letter_A_value_l3474_347424


namespace NUMINAMATH_CALUDE_reflection_sum_l3474_347458

theorem reflection_sum (x : ℝ) : 
  let C : ℝ × ℝ := (x, -3)
  let D : ℝ × ℝ := (-x, -3)
  (C.1 + C.2 + D.1 + D.2) = -6 := by sorry

end NUMINAMATH_CALUDE_reflection_sum_l3474_347458


namespace NUMINAMATH_CALUDE_least_valid_integer_l3474_347408

def is_valid (a : ℕ) : Prop :=
  a % 2 = 0 ∧ a % 3 = 1 ∧ a % 4 = 2

theorem least_valid_integer : ∃ (a : ℕ), is_valid a ∧ ∀ (b : ℕ), b < a → ¬(is_valid b) :=
by
  use 10
  sorry

end NUMINAMATH_CALUDE_least_valid_integer_l3474_347408


namespace NUMINAMATH_CALUDE_factorial_simplification_l3474_347481

theorem factorial_simplification : (13 * 12 * 11 * 10 * 9 * Nat.factorial 8) / (10 * 9 * Nat.factorial 8 + 3 * 9 * Nat.factorial 8) = 1320 := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l3474_347481


namespace NUMINAMATH_CALUDE_vip_price_is_60_l3474_347482

/-- Represents the ticket sales and pricing for a snooker tournament --/
structure SnookerTickets where
  totalTickets : ℕ
  totalRevenue : ℕ
  generalPrice : ℕ
  vipDifference : ℕ

/-- The specific ticket sales scenario for the tournament --/
def tournamentSales : SnookerTickets :=
  { totalTickets := 320
  , totalRevenue := 7500
  , generalPrice := 10
  , vipDifference := 148
  }

/-- Calculates the price of a VIP ticket --/
def vipPrice (s : SnookerTickets) : ℕ :=
  let generalTickets := (s.totalTickets + s.vipDifference) / 2
  let vipTickets := s.totalTickets - generalTickets
  (s.totalRevenue - s.generalPrice * generalTickets) / vipTickets

/-- Theorem stating that the VIP ticket price for the given scenario is $60 --/
theorem vip_price_is_60 : vipPrice tournamentSales = 60 := by
  sorry

end NUMINAMATH_CALUDE_vip_price_is_60_l3474_347482


namespace NUMINAMATH_CALUDE_well_digging_time_l3474_347451

theorem well_digging_time 
  (combined_time : ℝ) 
  (paul_time : ℝ) 
  (hari_time : ℝ) 
  (h1 : combined_time = 8)
  (h2 : paul_time = 24)
  (h3 : hari_time = 48) : 
  ∃ jake_time : ℝ, 
    jake_time = 16 ∧ 
    1 / combined_time = 1 / jake_time + 1 / paul_time + 1 / hari_time :=
by sorry

end NUMINAMATH_CALUDE_well_digging_time_l3474_347451


namespace NUMINAMATH_CALUDE_max_product_l3474_347449

def digits : Finset Nat := {1, 3, 5, 8, 9}

def valid_combination (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def three_digit (a b c : Nat) : Nat := 100 * a + 10 * b + c

def two_digit (d e : Nat) : Nat := 10 * d + e

theorem max_product :
  ∀ a b c d e,
    valid_combination a b c d e →
    (three_digit a b c) * (two_digit d e) ≤ (three_digit 9 3 1) * (two_digit 8 5) :=
by sorry

end NUMINAMATH_CALUDE_max_product_l3474_347449


namespace NUMINAMATH_CALUDE_trouser_price_decrease_trouser_price_decrease_result_l3474_347418

/-- Calculates the final percent decrease in price for a trouser purchase with given conditions. -/
theorem trouser_price_decrease (original_price : ℝ) (clearance_discount : ℝ) 
  (german_vat : ℝ) (us_vat : ℝ) (exchange_rate : ℝ) : ℝ :=
  let discounted_price := original_price * (1 - clearance_discount)
  let price_with_german_vat := discounted_price * (1 + german_vat)
  let price_in_usd := price_with_german_vat * exchange_rate
  let final_price := price_in_usd * (1 + us_vat)
  let original_price_usd := original_price * exchange_rate
  let percent_decrease := (original_price_usd - final_price) / original_price_usd * 100
  percent_decrease

/-- The final percent decrease in price is approximately 10.0359322%. -/
theorem trouser_price_decrease_result : 
  abs (trouser_price_decrease 100 0.3 0.19 0.08 1.18 - 10.0359322) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_trouser_price_decrease_trouser_price_decrease_result_l3474_347418
