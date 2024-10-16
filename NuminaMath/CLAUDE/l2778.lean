import Mathlib

namespace NUMINAMATH_CALUDE_rebus_solution_l2778_277838

theorem rebus_solution : 
  ∃! (A P M I R : ℕ), 
    A ∈ Finset.range 10 ∧ 
    P ∈ Finset.range 10 ∧ 
    M ∈ Finset.range 10 ∧ 
    I ∈ Finset.range 10 ∧ 
    R ∈ Finset.range 10 ∧ 
    A ≠ P ∧ A ≠ M ∧ A ≠ I ∧ A ≠ R ∧ 
    P ≠ M ∧ P ≠ I ∧ P ≠ R ∧ 
    M ≠ I ∧ M ≠ R ∧ 
    I ≠ R ∧ 
    (10 * A + P) ^ M = 100 * M + 10 * I + R ∧ 
    A = 1 ∧ P = 6 ∧ M = 2 ∧ I = 5 ∧ R = 6 := by
  sorry

end NUMINAMATH_CALUDE_rebus_solution_l2778_277838


namespace NUMINAMATH_CALUDE_peach_count_correct_l2778_277812

/-- The number of baskets -/
def total_baskets : ℕ := 150

/-- The number of peaches in each odd-numbered basket -/
def peaches_odd : ℕ := 14

/-- The number of peaches in each even-numbered basket -/
def peaches_even : ℕ := 12

/-- The total number of peaches -/
def total_peaches : ℕ := 1950

theorem peach_count_correct : 
  (total_baskets / 2) * peaches_odd + (total_baskets / 2) * peaches_even = total_peaches := by
  sorry

end NUMINAMATH_CALUDE_peach_count_correct_l2778_277812


namespace NUMINAMATH_CALUDE_cyclist_speed_north_cyclist_speed_north_proof_l2778_277830

/-- The speed of the cyclist going north, given two cyclists starting from the same place
    in opposite directions, with one going south at 25 km/h, and they take 1.4285714285714286 hours
    to be 50 km apart. -/
theorem cyclist_speed_north : ℝ → Prop :=
  fun v : ℝ =>
    let south_speed : ℝ := 25
    let time : ℝ := 1.4285714285714286
    let distance : ℝ := 50
    v > 0 ∧ distance = (v + south_speed) * time → v = 10

/-- Proof of the cyclist_speed_north theorem -/
theorem cyclist_speed_north_proof : cyclist_speed_north 10 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_north_cyclist_speed_north_proof_l2778_277830


namespace NUMINAMATH_CALUDE_travel_equations_correct_l2778_277862

/-- Represents the travel scenario with bike riding and walking -/
structure TravelScenario where
  total_time : ℝ
  total_distance : ℝ
  bike_speed : ℝ
  walk_speed : ℝ
  bike_time : ℝ
  walk_time : ℝ

/-- The given travel scenario matches the system of equations -/
def scenario_matches_equations (s : TravelScenario) : Prop :=
  s.total_time = 1.5 ∧
  s.total_distance = 20 ∧
  s.bike_speed = 15 ∧
  s.walk_speed = 5 ∧
  s.bike_time + s.walk_time = s.total_time ∧
  s.bike_speed * s.bike_time + s.walk_speed * s.walk_time = s.total_distance

/-- The system of equations correctly represents the travel scenario -/
theorem travel_equations_correct (s : TravelScenario) :
  scenario_matches_equations s →
  s.bike_time + s.walk_time = 1.5 ∧
  15 * s.bike_time + 5 * s.walk_time = 20 :=
by sorry

end NUMINAMATH_CALUDE_travel_equations_correct_l2778_277862


namespace NUMINAMATH_CALUDE_sparklers_burn_time_l2778_277806

/-- The number of sparklers -/
def num_sparklers : ℕ := 10

/-- The time it takes for one sparkler to burn down completely (in minutes) -/
def burn_time : ℚ := 2

/-- The fraction of time left when the next sparkler is lit -/
def fraction_left : ℚ := 1/10

/-- The time each sparkler burns before the next one is lit -/
def individual_burn_time : ℚ := burn_time * (1 - fraction_left)

/-- The total time for all sparklers to burn down (in minutes) -/
def total_burn_time : ℚ := (num_sparklers - 1) * individual_burn_time + burn_time

/-- Conversion function from minutes to minutes and seconds -/
def to_minutes_and_seconds (time : ℚ) : ℕ × ℕ :=
  let minutes := time.floor
  let seconds := ((time - minutes) * 60).floor
  (minutes.toNat, seconds.toNat)

theorem sparklers_burn_time :
  to_minutes_and_seconds total_burn_time = (18, 12) :=
sorry

end NUMINAMATH_CALUDE_sparklers_burn_time_l2778_277806


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l2778_277824

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + a + 3 = 0}

-- State the theorem
theorem subset_implies_a_range (a : ℝ) (h : B a ⊆ A) : -2 ≤ a ∧ a < 6 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l2778_277824


namespace NUMINAMATH_CALUDE_barge_unloading_time_l2778_277889

/-- Represents the unloading scenario of a barge with different crane configurations -/
structure BargeUnloading where
  /-- Time (in hours) for one crane of greater capacity to unload the barge alone -/
  x : ℝ
  /-- Time (in hours) for one crane of lesser capacity to unload the barge alone -/
  y : ℝ
  /-- Time (in hours) for one crane of greater capacity and one of lesser capacity to unload together -/
  z : ℝ

/-- The main theorem about the barge unloading scenario -/
theorem barge_unloading_time (b : BargeUnloading) : b.z = 14.4 :=
  sorry

end NUMINAMATH_CALUDE_barge_unloading_time_l2778_277889


namespace NUMINAMATH_CALUDE_distance_AK_equals_sqrt2_plus_1_l2778_277854

/-- Given a quadrilateral ABCD with vertices A(0, 0), B(0, -1), C(1, 0), D(√2/2, √2/2),
    and K is the intersection point of lines AB and CD,
    prove that the distance AK = √2 + 1 -/
theorem distance_AK_equals_sqrt2_plus_1 :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, -1)
  let C : ℝ × ℝ := (1, 0)
  let D : ℝ × ℝ := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)
  let K : ℝ × ℝ := (0, -(Real.sqrt 2 + 1))  -- Intersection point of AB and CD
  -- Distance formula
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  distance A K = Real.sqrt 2 + 1 := by sorry

end NUMINAMATH_CALUDE_distance_AK_equals_sqrt2_plus_1_l2778_277854


namespace NUMINAMATH_CALUDE_calculate_expression_l2778_277890

theorem calculate_expression : 101 * 102^2 - 101 * 98^2 = 80800 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2778_277890


namespace NUMINAMATH_CALUDE_number_difference_proof_l2778_277813

theorem number_difference_proof (x : ℚ) : x - (3/5) * x = 58 → x = 145 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_proof_l2778_277813


namespace NUMINAMATH_CALUDE_intersection_points_concyclic_l2778_277818

/-- A circle in which quadrilateral ABCD is inscribed -/
structure CircumCircle where
  center : ℝ × ℝ
  radius : ℝ

/-- A convex quadrilateral ABCD inscribed in a circle -/
structure InscribedQuadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  circle : CircumCircle

/-- Circles drawn with each side of ABCD as a chord -/
structure SideCircles where
  AB : CircumCircle
  BC : CircumCircle
  CD : CircumCircle
  DA : CircumCircle

/-- Intersection points of circles drawn over adjacent sides -/
structure IntersectionPoints where
  A1 : ℝ × ℝ
  B1 : ℝ × ℝ
  C1 : ℝ × ℝ
  D1 : ℝ × ℝ

/-- Function to check if four points are concyclic -/
def areConcyclic (p1 p2 p3 p4 : ℝ × ℝ) : Prop := sorry

theorem intersection_points_concyclic 
  (quad : InscribedQuadrilateral) 
  (sides : SideCircles) 
  (points : IntersectionPoints) : 
  areConcyclic points.A1 points.B1 points.C1 points.D1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_concyclic_l2778_277818


namespace NUMINAMATH_CALUDE_common_factor_implies_a_values_l2778_277821

theorem common_factor_implies_a_values (a : ℝ) :
  (∃ (p : ℝ) (A B : ℝ → ℝ), p ≠ 0 ∧
    (∀ x, x^3 - x - a = A x * (x + p)) ∧
    (∀ x, x^2 + x - a = B x * (x + p))) →
  (a = 0 ∨ a = 10 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_common_factor_implies_a_values_l2778_277821


namespace NUMINAMATH_CALUDE_x_minus_y_positive_l2778_277880

theorem x_minus_y_positive (x y a : ℝ) 
  (h1 : x + y > 0) 
  (h2 : a < 0) 
  (h3 : a * y > 0) : 
  x - y > 0 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_positive_l2778_277880


namespace NUMINAMATH_CALUDE_solve_equation_l2778_277814

theorem solve_equation (k l x : ℝ) : 
  (2 : ℝ) / 3 = k / 54 ∧ 
  (2 : ℝ) / 3 = (k + l) / 90 ∧ 
  (2 : ℝ) / 3 = (x - l) / 150 → 
  x = 106 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l2778_277814


namespace NUMINAMATH_CALUDE_final_price_after_discounts_l2778_277836

/-- Given an original price p and two consecutive 10% discounts,
    the final selling price is 0.81p -/
theorem final_price_after_discounts (p : ℝ) : 
  let discount := 0.1
  let first_discount := p * (1 - discount)
  let second_discount := first_discount * (1 - discount)
  second_discount = 0.81 * p := by
sorry

end NUMINAMATH_CALUDE_final_price_after_discounts_l2778_277836


namespace NUMINAMATH_CALUDE_cube_of_negative_l2778_277898

theorem cube_of_negative (x : ℝ) : (-x)^3 = -x^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_l2778_277898


namespace NUMINAMATH_CALUDE_no_real_roots_l2778_277846

theorem no_real_roots :
  ∀ x : ℝ, ¬(Real.sqrt (x + 9) - Real.sqrt (x - 5) + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_l2778_277846


namespace NUMINAMATH_CALUDE_anns_total_blocks_l2778_277883

/-- Ann's initial number of blocks -/
def initial_blocks : ℕ := 9

/-- Number of blocks Ann finds -/
def found_blocks : ℕ := 44

/-- Theorem: Ann's total number of blocks after finding more -/
theorem anns_total_blocks : initial_blocks + found_blocks = 53 := by
  sorry

end NUMINAMATH_CALUDE_anns_total_blocks_l2778_277883


namespace NUMINAMATH_CALUDE_cube_as_difference_of_squares_l2778_277816

theorem cube_as_difference_of_squares (n : ℤ) (h : n > 1) :
  ∃ (a b : ℤ), a > 0 ∧ b > 0 ∧ n^3 = a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_as_difference_of_squares_l2778_277816


namespace NUMINAMATH_CALUDE_trig_simplification_l2778_277869

theorem trig_simplification (x : Real) :
  (2 * Real.cos (55 * π / 180) - Real.sqrt 3 * Real.sin (5 * π / 180)) / Real.cos (5 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l2778_277869


namespace NUMINAMATH_CALUDE_odd_function_sum_l2778_277847

def f (x : ℝ) (b : ℝ) : ℝ := 2016 * x^3 - 5 * x + b + 2

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (a b : ℝ) :
  (∃ f : ℝ → ℝ, is_odd f ∧ (∀ x, f x = 2016 * x^3 - 5 * x + b + 2) ∧
   (∃ c d : ℝ, c = a - 4 ∧ d = 2 * a - 2 ∧ Set.Icc c d = Set.range f)) →
  f a + f b = 0 :=
sorry

end NUMINAMATH_CALUDE_odd_function_sum_l2778_277847


namespace NUMINAMATH_CALUDE_marble_distribution_l2778_277888

def jasmine_initial : ℕ := 120
def lola_initial : ℕ := 15
def marbles_given : ℕ := 19

theorem marble_distribution :
  (jasmine_initial - marbles_given) = 3 * (lola_initial + marbles_given) := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l2778_277888


namespace NUMINAMATH_CALUDE_line_through_P_with_opposite_sign_intercepts_l2778_277856

-- Define the point P
def P : ℝ × ℝ := (3, -2)

-- Define the line equation types
inductive LineEquation
| Standard (a b c : ℝ) : LineEquation  -- ax + by + c = 0
| SlopeIntercept (m b : ℝ) : LineEquation  -- y = mx + b

-- Define a predicate for a line passing through a point
def passesThrough (eq : LineEquation) (p : ℝ × ℝ) : Prop :=
  match eq with
  | LineEquation.Standard a b c => a * p.1 + b * p.2 + c = 0
  | LineEquation.SlopeIntercept m b => p.2 = m * p.1 + b

-- Define a predicate for a line having intercepts of opposite signs
def hasOppositeSignIntercepts (eq : LineEquation) : Prop :=
  match eq with
  | LineEquation.Standard a b c =>
    (c / a) * (c / b) < 0 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0
  | LineEquation.SlopeIntercept m b =>
    b * (b / m) < 0 ∧ m ≠ 0 ∧ b ≠ 0

-- The main theorem
theorem line_through_P_with_opposite_sign_intercepts :
  ∃ (eq : LineEquation),
    (eq = LineEquation.Standard 1 (-1) (-5) ∨ eq = LineEquation.SlopeIntercept (-2/3) 0) ∧
    passesThrough eq P ∧
    hasOppositeSignIntercepts eq :=
  sorry

end NUMINAMATH_CALUDE_line_through_P_with_opposite_sign_intercepts_l2778_277856


namespace NUMINAMATH_CALUDE_forgot_homework_percentage_l2778_277819

/-- Represents the percentage of students who forgot their homework in group B -/
def percentage_forgot_B : ℝ := 15

theorem forgot_homework_percentage :
  let total_students : ℕ := 100
  let group_A_students : ℕ := 20
  let group_B_students : ℕ := 80
  let percentage_forgot_A : ℝ := 20
  let percentage_forgot_total : ℝ := 16
  percentage_forgot_B = ((percentage_forgot_total * total_students) - 
                         (percentage_forgot_A * group_A_students)) / group_B_students * 100 :=
by sorry

end NUMINAMATH_CALUDE_forgot_homework_percentage_l2778_277819


namespace NUMINAMATH_CALUDE_gabled_cuboid_theorem_l2778_277858

/-- Represents a cuboid with gable-shaped figures on each face -/
structure GabledCuboid where
  a : ℝ
  b : ℝ
  c : ℝ
  h_ab : a > b
  h_bc : b > c

/-- Properties of the gabled cuboid -/
def GabledCuboidProperties (g : GabledCuboid) : Prop :=
  ∃ (num_faces num_edges num_vertices : ℕ) (volume : ℝ),
    num_faces = 12 ∧
    num_edges = 30 ∧
    num_vertices = 20 ∧
    volume = g.a * g.b * g.c + (1/2) * (g.a * g.b^2 + g.a * g.c^2 + g.b * g.c^2) - g.b^3/6 - g.c^3/3

theorem gabled_cuboid_theorem (g : GabledCuboid) : GabledCuboidProperties g := by
  sorry

end NUMINAMATH_CALUDE_gabled_cuboid_theorem_l2778_277858


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2778_277829

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_chord : 4 * a + 2 * b = 2) : 
  (1 / a + 2 / b) ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 4 * a₀ + 2 * b₀ = 2 ∧ 1 / a₀ + 2 / b₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2778_277829


namespace NUMINAMATH_CALUDE_relay_race_distance_l2778_277811

theorem relay_race_distance (siwon_fraction dawon_fraction : ℚ) 
  (combined_distance : ℝ) (total_distance : ℝ) : 
  siwon_fraction = 3 / 10 →
  dawon_fraction = 4 / 10 →
  combined_distance = 140 →
  (siwon_fraction + dawon_fraction : ℝ) * total_distance = combined_distance →
  total_distance = 200 :=
by sorry

end NUMINAMATH_CALUDE_relay_race_distance_l2778_277811


namespace NUMINAMATH_CALUDE_numbers_with_2019_divisors_l2778_277865

def has_2019_divisors (n : ℕ) : Prop :=
  (Finset.card (Nat.divisors n) = 2019)

theorem numbers_with_2019_divisors :
  {n : ℕ | n < 128^97 ∧ has_2019_divisors n} =
  {2^672 * 3^2, 2^672 * 5^2, 2^672 * 7^2, 2^672 * 11^2} :=
by sorry

end NUMINAMATH_CALUDE_numbers_with_2019_divisors_l2778_277865


namespace NUMINAMATH_CALUDE_circle_radius_in_triangle_l2778_277884

/-- Represents a triangle with side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Determines if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop := sorry

/-- Determines if a circle is tangent to two sides of a triangle -/
def is_tangent_to_sides (c : Circle) (t : Triangle) : Prop := sorry

/-- Determines if a circle lies entirely within a triangle -/
def lies_within_triangle (c : Circle) (t : Triangle) : Prop := sorry

/-- Main theorem statement -/
theorem circle_radius_in_triangle (t : Triangle) (r s : Circle) : 
  t.a = 120 → t.b = 120 → t.c = 70 →
  r.radius = 20 →
  is_tangent_to_sides r t →
  are_externally_tangent r s →
  is_tangent_to_sides s t →
  lies_within_triangle s t →
  s.radius = 54 - 8 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_in_triangle_l2778_277884


namespace NUMINAMATH_CALUDE_expression_evaluation_l2778_277895

theorem expression_evaluation :
  let x : ℚ := -2
  (3 + x * (3 + x) - 3^2) / (x - 3 + x^2) = 8 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2778_277895


namespace NUMINAMATH_CALUDE_triangle_inequality_range_l2778_277841

theorem triangle_inequality_range (A B C : ℝ) (t : ℝ) : 
  0 < B → B ≤ π/3 → 
  (∀ x : ℝ, (x + 2 + Real.sin (2*B))^2 + (Real.sqrt 2 * t * Real.sin (B + π/4))^2 ≥ 1) →
  t ∈ Set.Ici 1 ∪ Set.Iic (-1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_range_l2778_277841


namespace NUMINAMATH_CALUDE_landscape_breadth_l2778_277859

/-- Represents a rectangular landscape with specific features -/
structure Landscape where
  length : ℝ
  breadth : ℝ
  playground_area : ℝ
  walking_path_ratio : ℝ
  water_body_ratio : ℝ

/-- Theorem stating the breadth of the landscape given specific conditions -/
theorem landscape_breadth (l : Landscape) 
  (h1 : l.breadth = 8 * l.length)
  (h2 : l.playground_area = 3200)
  (h3 : l.playground_area = (l.length * l.breadth) / 9)
  (h4 : l.walking_path_ratio = 1 / 18)
  (h5 : l.water_body_ratio = 1 / 6)
  : l.breadth = 480 := by
  sorry

end NUMINAMATH_CALUDE_landscape_breadth_l2778_277859


namespace NUMINAMATH_CALUDE_consecutive_integers_divisibility_l2778_277875

theorem consecutive_integers_divisibility (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a < b) (h5 : b < c) :
  ∀ n : ℕ, ∃ x y z : ℕ,
    x ∈ Finset.range (2 * c) ∧
    y ∈ Finset.range (2 * c) ∧
    z ∈ Finset.range (2 * c) ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (a * b * c) ∣ (x * y * z) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_divisibility_l2778_277875


namespace NUMINAMATH_CALUDE_square_is_self_product_l2778_277851

theorem square_is_self_product (b : ℚ) : b^2 = b * b := by
  sorry

end NUMINAMATH_CALUDE_square_is_self_product_l2778_277851


namespace NUMINAMATH_CALUDE_abc_inequality_l2778_277874

theorem abc_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2778_277874


namespace NUMINAMATH_CALUDE_quidditch_tournament_equal_wins_l2778_277876

/-- Represents a team in the Quidditch tournament -/
structure Team :=
  (id : Nat)

/-- Represents the tournament setup -/
structure Tournament :=
  (teams : Finset Team)
  (num_teams : Nat)
  (wins : Team → Nat)
  (h_num_teams : teams.card = num_teams)
  (h_wins_bound : ∀ t ∈ teams, wins t < num_teams)
  (h_total_wins : (teams.sum wins) = num_teams * (num_teams - 1) / 2)

/-- Main theorem statement -/
theorem quidditch_tournament_equal_wins (tournament : Tournament) 
  (h_eight_teams : tournament.num_teams = 8) :
  ∃ (A B C D : Team), A ∈ tournament.teams ∧ B ∈ tournament.teams ∧ 
    C ∈ tournament.teams ∧ D ∈ tournament.teams ∧ A ≠ B ∧ C ≠ D ∧ 
    tournament.wins A + tournament.wins B = tournament.wins C + tournament.wins D :=
by sorry

end NUMINAMATH_CALUDE_quidditch_tournament_equal_wins_l2778_277876


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l2778_277850

theorem initial_mean_calculation (n : ℕ) (incorrect_value correct_value : ℝ) (correct_mean : ℝ) :
  n = 30 ∧ 
  incorrect_value = 135 ∧ 
  correct_value = 165 ∧ 
  correct_mean = 151 →
  ∃ (initial_mean : ℝ),
    n * initial_mean + (correct_value - incorrect_value) = n * correct_mean ∧
    initial_mean = 150 :=
by sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l2778_277850


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2778_277866

theorem inequality_solution_range (a : ℝ) : 
  (∃ x ∈ Set.Ioo 1 4, x^2 - 4*x - 2 - a > 0) → a ∈ Set.Ioi (-2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2778_277866


namespace NUMINAMATH_CALUDE_lg_sum_equals_three_l2778_277848

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_equals_three : lg 8 + 3 * lg 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lg_sum_equals_three_l2778_277848


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_vector_expression_equality_l2778_277817

-- Part 1
theorem trigonometric_expression_equality :
  Real.cos (25 * Real.pi / 3) + Real.tan (-15 * Real.pi / 4) = 3/2 := by sorry

-- Part 2
theorem vector_expression_equality {n : Type*} [NormedAddCommGroup n] :
  ∀ (a b : n), 2 • (a - b) - (2 • a + b) + 3 • b = 0 := by sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_vector_expression_equality_l2778_277817


namespace NUMINAMATH_CALUDE_zero_decomposition_l2778_277896

/-- Represents a base-10 arithmetic system -/
structure Base10Arithmetic where
  /-- Multiplication operation in base-10 arithmetic -/
  mul : ℤ → ℤ → ℤ
  /-- Axiom: Multiplication by zero always results in zero -/
  mul_zero : ∀ a : ℤ, mul 0 a = 0

/-- 
Theorem: In base-10 arithmetic, the only way to decompose 0 into a product 
of two integers is 0 * a = 0, where a is any integer.
-/
theorem zero_decomposition (B : Base10Arithmetic) : 
  ∀ x y : ℤ, B.mul x y = 0 → x = 0 ∨ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_decomposition_l2778_277896


namespace NUMINAMATH_CALUDE_subset_sum_divisible_by_2n_l2778_277894

theorem subset_sum_divisible_by_2n
  (n : ℕ)
  (h_n : n ≥ 4)
  (a : Fin n → ℕ)
  (h_distinct : Function.Injective a)
  (h_bounds : ∀ i : Fin n, 0 < a i ∧ a i < 2*n) :
  ∃ (S : Finset (Fin n)), (S.sum (λ i => a i)) % (2*n) = 0 :=
sorry

end NUMINAMATH_CALUDE_subset_sum_divisible_by_2n_l2778_277894


namespace NUMINAMATH_CALUDE_four_fours_exist_l2778_277810

/-- A datatype representing arithmetic expressions using only the digit 4 --/
inductive Expr4
  | four : Expr4
  | add : Expr4 → Expr4 → Expr4
  | sub : Expr4 → Expr4 → Expr4
  | mul : Expr4 → Expr4 → Expr4
  | div : Expr4 → Expr4 → Expr4

/-- Evaluate an Expr4 to a rational number --/
def eval : Expr4 → ℚ
  | Expr4.four => 4
  | Expr4.add e1 e2 => eval e1 + eval e2
  | Expr4.sub e1 e2 => eval e1 - eval e2
  | Expr4.mul e1 e2 => eval e1 * eval e2
  | Expr4.div e1 e2 => eval e1 / eval e2

/-- Count the number of 4's used in an Expr4 --/
def count_fours : Expr4 → ℕ
  | Expr4.four => 1
  | Expr4.add e1 e2 => count_fours e1 + count_fours e2
  | Expr4.sub e1 e2 => count_fours e1 + count_fours e2
  | Expr4.mul e1 e2 => count_fours e1 + count_fours e2
  | Expr4.div e1 e2 => count_fours e1 + count_fours e2

/-- Theorem stating that expressions for 2, 3, 4, 5, and 6 exist using four 4's --/
theorem four_fours_exist : 
  ∃ (e2 e3 e4 e5 e6 : Expr4), 
    (count_fours e2 = 4 ∧ eval e2 = 2) ∧
    (count_fours e3 = 4 ∧ eval e3 = 3) ∧
    (count_fours e4 = 4 ∧ eval e4 = 4) ∧
    (count_fours e5 = 4 ∧ eval e5 = 5) ∧
    (count_fours e6 = 4 ∧ eval e6 = 6) := by
  sorry

end NUMINAMATH_CALUDE_four_fours_exist_l2778_277810


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2778_277803

theorem sqrt_inequality (x : ℝ) : 
  Real.sqrt (3 - x) - Real.sqrt (x + 1) > (1 : ℝ) / 2 ↔ 
  -1 ≤ x ∧ x < 1 - Real.sqrt 31 / 8 :=
sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2778_277803


namespace NUMINAMATH_CALUDE_a_sequence_property_l2778_277842

def a : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | 2 => 1
  | (n + 3) => a (n + 1) + 1998 * a n

theorem a_sequence_property (n : ℕ) (h : n > 0) :
  a (2 * n - 1) = 2 * a n * a (n + 1) + 1998 * a (n - 1) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_a_sequence_property_l2778_277842


namespace NUMINAMATH_CALUDE_valid_number_count_l2778_277891

/-- Represents a valid seven-digit number configuration --/
structure ValidNumber :=
  (digits : Fin 7 → Fin 7)
  (injective : Function.Injective digits)
  (no_6_7_at_ends : digits 0 ≠ 5 ∧ digits 0 ≠ 6 ∧ digits 6 ≠ 5 ∧ digits 6 ≠ 6)
  (one_adjacent_six : ∃ i, (digits i = 0 ∧ digits (i+1) = 5) ∨ (digits i = 5 ∧ digits (i+1) = 0))

/-- The number of valid seven-digit numbers --/
def count_valid_numbers : ℕ := sorry

/-- Theorem stating the count of valid numbers --/
theorem valid_number_count : count_valid_numbers = 768 := by sorry

end NUMINAMATH_CALUDE_valid_number_count_l2778_277891


namespace NUMINAMATH_CALUDE_largest_number_l2778_277887

theorem largest_number (S : Set ℝ) (hS : S = {1/2, 0, 1, -9}) : 
  ∃ m ∈ S, ∀ x ∈ S, x ≤ m ∧ m = 1 :=
sorry

end NUMINAMATH_CALUDE_largest_number_l2778_277887


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_341_l2778_277864

theorem greatest_prime_factor_of_341 : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 341 → q ≤ p :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_341_l2778_277864


namespace NUMINAMATH_CALUDE_unit_digit_of_7_to_500_l2778_277807

theorem unit_digit_of_7_to_500 : 7^500 % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_7_to_500_l2778_277807


namespace NUMINAMATH_CALUDE_rectangle_ratio_l2778_277849

/-- Given a rectangular plot with area 363 sq m and breadth 11 m, 
    prove that the ratio of length to breadth is 3:1 -/
theorem rectangle_ratio : ∀ (length breadth : ℝ),
  breadth = 11 →
  length * breadth = 363 →
  length / breadth = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l2778_277849


namespace NUMINAMATH_CALUDE_square_circle_difference_l2778_277823

-- Define the square and circle
def square_diagonal : ℝ := 8
def circle_diameter : ℝ := 8

-- Theorem statement
theorem square_circle_difference :
  let square_side := (square_diagonal ^ 2 / 2).sqrt
  let square_area := square_side ^ 2
  let square_perimeter := 4 * square_side
  let circle_radius := circle_diameter / 2
  let circle_area := π * circle_radius ^ 2
  let circle_perimeter := 2 * π * circle_radius
  (circle_area - square_area = 16 * π - 32) ∧
  (circle_perimeter - square_perimeter = 8 * π - 16 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_square_circle_difference_l2778_277823


namespace NUMINAMATH_CALUDE_solution_form_and_sum_l2778_277832

theorem solution_form_and_sum (x y : ℝ) : 
  (x + y = 7 ∧ 4 * x * y = 7) →
  ∃ (a b c d : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d ∧
    a = 7 ∧ b = 1 ∧ c = 42 ∧ d = 2 ∧
    a + b + c + d = 52 :=
by sorry

end NUMINAMATH_CALUDE_solution_form_and_sum_l2778_277832


namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l2778_277886

def polynomial (x : ℝ) : ℝ := 4 * (x^4 + 3*x^2 + 1)

theorem sum_of_squares_of_coefficients :
  (4^2) + (12^2) + (4^2) = 176 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l2778_277886


namespace NUMINAMATH_CALUDE_tommy_books_l2778_277879

/-- The number of books Tommy wants to buy -/
def num_books (book_cost savings_needed current_money : ℕ) : ℕ :=
  (savings_needed + current_money) / book_cost

/-- Proof that Tommy wants to buy 8 books -/
theorem tommy_books : num_books 5 27 13 = 8 := by
  sorry

end NUMINAMATH_CALUDE_tommy_books_l2778_277879


namespace NUMINAMATH_CALUDE_number_puzzle_l2778_277840

theorem number_puzzle :
  ∃ x : ℝ, 3 * (2 * x + 9) = 75 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2778_277840


namespace NUMINAMATH_CALUDE_opinion_change_difference_is_twenty_percent_l2778_277881

/-- Represents the percentage of students who like science -/
structure ScienceOpinion where
  initial_like : ℚ
  final_like : ℚ

/-- Calculate the difference between maximum and minimum percentage of students who changed their opinion -/
def opinion_change_difference (opinion : ScienceOpinion) : ℚ :=
  let initial_dislike := 1 - opinion.initial_like
  let final_dislike := 1 - opinion.final_like
  let min_change := |opinion.final_like - opinion.initial_like|
  let max_change := min opinion.initial_like final_dislike + min initial_dislike opinion.final_like
  max_change - min_change

/-- Theorem statement for the specific problem -/
theorem opinion_change_difference_is_twenty_percent :
  ∃ (opinion : ScienceOpinion),
    opinion.initial_like = 2/5 ∧
    opinion.final_like = 4/5 ∧
    opinion_change_difference opinion = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_opinion_change_difference_is_twenty_percent_l2778_277881


namespace NUMINAMATH_CALUDE_arccos_cos_three_l2778_277872

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 := by sorry

end NUMINAMATH_CALUDE_arccos_cos_three_l2778_277872


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2778_277860

/-- The quadratic equation (a-5)x^2 - 4x - 1 = 0 has real roots if and only if a ≥ 1 and a ≠ 5. -/
theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, (a - 5) * x^2 - 4*x - 1 = 0) ↔ (a ≥ 1 ∧ a ≠ 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2778_277860


namespace NUMINAMATH_CALUDE_no_inscribed_triangle_with_sine_roots_l2778_277834

theorem no_inscribed_triangle_with_sine_roots :
  ¬ ∃ (a b c : ℝ) (A B C : ℝ),
    0 < a ∧ 0 < b ∧ 0 < c ∧
    0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
    A + B + C = π ∧
    a = 2 * Real.sin (A / 2) ∧
    b = 2 * Real.sin (B / 2) ∧
    c = 2 * Real.sin (C / 2) ∧
    ∃ (p : ℝ),
      (Real.sin A)^3 - 2 * a * (Real.sin A)^2 + b * c * Real.sin A = p ∧
      (Real.sin B)^3 - 2 * a * (Real.sin B)^2 + b * c * Real.sin B = p ∧
      (Real.sin C)^3 - 2 * a * (Real.sin C)^2 + b * c * Real.sin C = p :=
by sorry

end NUMINAMATH_CALUDE_no_inscribed_triangle_with_sine_roots_l2778_277834


namespace NUMINAMATH_CALUDE_betty_orange_boxes_l2778_277855

/-- The minimum number of boxes needed to store oranges given specific conditions -/
def min_boxes (total_oranges : ℕ) (first_box : ℕ) (second_box : ℕ) (max_per_box : ℕ) : ℕ :=
  2 + (total_oranges - first_box - second_box + max_per_box - 1) / max_per_box

/-- Proof that Betty needs 5 boxes to store her oranges -/
theorem betty_orange_boxes : 
  min_boxes 120 30 25 30 = 5 :=
by sorry

end NUMINAMATH_CALUDE_betty_orange_boxes_l2778_277855


namespace NUMINAMATH_CALUDE_quadratic_function_condition_l2778_277839

/-- Given a quadratic function f(x) = x^2 + bx + c, if there exists an x₀ such that
    f(f(x₀)) = 0 and f(x₀) ≠ 0, then b < 0 or b ≥ 4 -/
theorem quadratic_function_condition (b c : ℝ) : 
  (∃ x₀ : ℝ, (x₀^2 + b*x₀ + c)^2 + b*(x₀^2 + b*x₀ + c) + c = 0 ∧ 
              x₀^2 + b*x₀ + c ≠ 0) →
  b < 0 ∨ b ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_condition_l2778_277839


namespace NUMINAMATH_CALUDE_gcd_143_117_l2778_277885

theorem gcd_143_117 : Nat.gcd 143 117 = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_143_117_l2778_277885


namespace NUMINAMATH_CALUDE_ashton_pencils_l2778_277827

theorem ashton_pencils (initial_pencils_per_box : ℕ) : 
  (2 * initial_pencils_per_box) - 6 = 22 → initial_pencils_per_box = 14 :=
by sorry

end NUMINAMATH_CALUDE_ashton_pencils_l2778_277827


namespace NUMINAMATH_CALUDE_child_ticket_cost_l2778_277873

theorem child_ticket_cost
  (adult_price : ℕ)
  (total_tickets : ℕ)
  (total_receipts : ℕ)
  (adult_tickets : ℕ)
  (h1 : adult_price = 12)
  (h2 : total_tickets = 130)
  (h3 : total_receipts = 840)
  (h4 : adult_tickets = 40)
  : ∃ (child_price : ℕ),
    child_price = 4 ∧
    adult_price * adult_tickets + child_price * (total_tickets - adult_tickets) = total_receipts :=
by
  sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l2778_277873


namespace NUMINAMATH_CALUDE_computer_profit_pricing_l2778_277804

theorem computer_profit_pricing (cost selling_price_40 selling_price_60 : ℝ) :
  selling_price_40 = 2240 ∧
  selling_price_40 = cost * 1.4 →
  selling_price_60 = cost * 1.6 →
  selling_price_60 = 2560 := by
sorry

end NUMINAMATH_CALUDE_computer_profit_pricing_l2778_277804


namespace NUMINAMATH_CALUDE_median_of_consecutive_integers_l2778_277845

theorem median_of_consecutive_integers (n : ℕ) (sum : ℕ) (h1 : n = 36) (h2 : sum = 1296) :
  sum / n = 36 := by
  sorry

end NUMINAMATH_CALUDE_median_of_consecutive_integers_l2778_277845


namespace NUMINAMATH_CALUDE_board_block_system_l2778_277837

/-- A proof problem about forces and acceleration on a board and block system. -/
theorem board_block_system 
  (M : Real) (m : Real) (μ : Real) (g : Real) (a : Real)
  (hM : M = 4)
  (hm : m = 1)
  (hμ : μ = 0.2)
  (hg : g = 10)
  (ha : a = g / 5) :
  let T := m * (a + μ * g)
  let F := μ * g * (M + 2 * m) + M * a + T
  T = 4 ∧ F = 24 := by
  sorry


end NUMINAMATH_CALUDE_board_block_system_l2778_277837


namespace NUMINAMATH_CALUDE_man_mass_from_boat_displacement_l2778_277857

/-- Calculates the mass of a man based on the displacement of a boat -/
theorem man_mass_from_boat_displacement (boat_length boat_breadth boat_sink_height water_density : Real) 
  (h1 : boat_length = 3)
  (h2 : boat_breadth = 2)
  (h3 : boat_sink_height = 0.01)
  (h4 : water_density = 1000) : 
  boat_length * boat_breadth * boat_sink_height * water_density = 60 := by
  sorry

#check man_mass_from_boat_displacement

end NUMINAMATH_CALUDE_man_mass_from_boat_displacement_l2778_277857


namespace NUMINAMATH_CALUDE_race_result_l2778_277852

/-- Represents the difference in meters between two runners at the end of a 1000-meter race. -/
def finish_difference (runner1 runner2 : ℕ) : ℝ := sorry

theorem race_result (A B C : ℕ) :
  finish_difference A C = 200 →
  finish_difference B C = 120.87912087912093 →
  finish_difference A B = 79.12087912087907 :=
by sorry

end NUMINAMATH_CALUDE_race_result_l2778_277852


namespace NUMINAMATH_CALUDE_two_solutions_iff_a_gt_neg_one_l2778_277893

/-- The equation has exactly two solutions if and only if a > -1 -/
theorem two_solutions_iff_a_gt_neg_one (a : ℝ) :
  (∃! x y, x ≠ y ∧ x^2 + 2*x + 2*|x+1| = a ∧ y^2 + 2*y + 2*|y+1| = a) ↔ a > -1 :=
sorry

end NUMINAMATH_CALUDE_two_solutions_iff_a_gt_neg_one_l2778_277893


namespace NUMINAMATH_CALUDE_cafeteria_pies_l2778_277800

def number_of_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  ((initial_apples - handed_out) / apples_per_pie : ℕ)

theorem cafeteria_pies :
  number_of_pies 150 24 15 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l2778_277800


namespace NUMINAMATH_CALUDE_bankers_discount_equation_l2778_277833

/-- The banker's discount (BD) for a certain sum of money. -/
def BD : ℚ := 80

/-- The true discount (TD) for the same sum of money. -/
def TD : ℚ := 70

/-- The present value (PV) of the sum due. -/
def PV : ℚ := 490

/-- Theorem stating that the given BD, TD, and PV satisfy the banker's discount equation. -/
theorem bankers_discount_equation : BD = TD + TD^2 / PV := by sorry

end NUMINAMATH_CALUDE_bankers_discount_equation_l2778_277833


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2778_277808

theorem repeating_decimal_sum (b c : ℕ) : 
  b < 10 → c < 10 →
  (10 * b + c : ℚ) / 99 + (100 * c + 10 * b + c : ℚ) / 999 = 83 / 222 →
  b = 1 ∧ c = 1 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2778_277808


namespace NUMINAMATH_CALUDE_weight_difference_l2778_277843

/-- Antonio's weight in kilograms -/
def antonio_weight : ℕ := 50

/-- Total weight of Antonio and his sister in kilograms -/
def total_weight : ℕ := 88

/-- Antonio's sister's weight in kilograms -/
def sister_weight : ℕ := total_weight - antonio_weight

theorem weight_difference :
  antonio_weight > sister_weight ∧
  antonio_weight - sister_weight = 12 := by
  sorry

#check weight_difference

end NUMINAMATH_CALUDE_weight_difference_l2778_277843


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l2778_277831

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (λ acc (i, b) => acc + if b then 2^i else 0) 0

-- Define a function to convert ternary to decimal
def ternary_to_decimal (ternary : List ℕ) : ℕ :=
  ternary.enum.foldl (λ acc (i, d) => acc + d * 3^i) 0

theorem product_of_binary_and_ternary :
  let binary := [false, true, false, true]  -- 1010 in binary
  let ternary := [2, 0, 1]  -- 102 in ternary
  (binary_to_decimal binary) * (ternary_to_decimal ternary) = 110 := by
  sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l2778_277831


namespace NUMINAMATH_CALUDE_no_extreme_points_iff_l2778_277867

/-- The function f(x) defined as ax³ + ax² + 7x -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + a * x^2 + 7 * x

/-- A function has no extreme points if its derivative is always non-negative or always non-positive -/
def has_no_extreme_points (g : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, (deriv g) x ≥ 0) ∨ (∀ x : ℝ, (deriv g) x ≤ 0)

/-- The main theorem: f(x) has no extreme points if and only if 0 ≤ a ≤ 21 -/
theorem no_extreme_points_iff (a : ℝ) :
  has_no_extreme_points (f a) ↔ 0 ≤ a ∧ a ≤ 21 := by sorry

end NUMINAMATH_CALUDE_no_extreme_points_iff_l2778_277867


namespace NUMINAMATH_CALUDE_adam_apples_l2778_277828

theorem adam_apples (jackie_apples : ℕ) (adam_apples : ℕ) 
  (h1 : jackie_apples = 10) 
  (h2 : jackie_apples = adam_apples + 1) : 
  adam_apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_adam_apples_l2778_277828


namespace NUMINAMATH_CALUDE_muffin_cost_calculation_l2778_277815

/-- Given a purchase of 3 items of equal cost and one item of known cost,
    with a discount applied, prove the original cost of each equal-cost item. -/
theorem muffin_cost_calculation (M : ℝ) : 
  (∃ (M : ℝ), 
    (0.85 * (3 * M + 1.45) = 3.70) ∧ 
    (abs (M - 0.97) < 0.01)) := by
  sorry

end NUMINAMATH_CALUDE_muffin_cost_calculation_l2778_277815


namespace NUMINAMATH_CALUDE_max_sum_with_gcf_six_l2778_277835

theorem max_sum_with_gcf_six (a b : ℕ) : 
  10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 →  -- a and b are two-digit positive integers
  Nat.gcd a b = 6 →                    -- greatest common factor of a and b is 6
  a + b ≤ 186 ∧                        -- upper bound
  ∃ (a' b' : ℕ), 10 ≤ a' ∧ a' ≤ 99 ∧ 10 ≤ b' ∧ b' ≤ 99 ∧ 
    Nat.gcd a' b' = 6 ∧ a' + b' = 186  -- existence of a pair that achieves the maximum
  := by sorry

end NUMINAMATH_CALUDE_max_sum_with_gcf_six_l2778_277835


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2778_277820

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

theorem condition_sufficient_not_necessary (a : ℕ → ℝ) :
  (∀ n, a (n + 1) > |a n|) → is_increasing a ∧
  ¬(is_increasing a → ∀ n, a (n + 1) > |a n|) :=
by
  sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2778_277820


namespace NUMINAMATH_CALUDE_complex_sum_equals_seven_plus_three_i_l2778_277871

theorem complex_sum_equals_seven_plus_three_i :
  let B : ℂ := 3 + 2*I
  let Q : ℂ := -3
  let R : ℂ := -2*I
  let T : ℂ := 1 + 3*I
  B - Q + R + T = 7 + 3*I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_equals_seven_plus_three_i_l2778_277871


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2778_277892

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + 2*y' = 1 → 1/x' + 1/y' ≥ 3 + 2*Real.sqrt 2) ∧
  (∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 1 ∧ 1/x₀ + 1/y₀ = 3 + 2*Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2778_277892


namespace NUMINAMATH_CALUDE_union_A_complement_B_l2778_277899

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2 + 1}

-- Theorem statement
theorem union_A_complement_B : A ∪ (U \ B) = {x | x < 2} := by sorry

end NUMINAMATH_CALUDE_union_A_complement_B_l2778_277899


namespace NUMINAMATH_CALUDE_nedy_crackers_l2778_277809

/-- The number of cracker packs Nedy ate from Monday to Thursday -/
def monday_to_thursday : ℕ := 8

/-- The number of cracker packs Nedy ate on Friday -/
def friday : ℕ := 2 * monday_to_thursday

/-- The total number of cracker packs Nedy ate from Monday to Friday -/
def total : ℕ := monday_to_thursday + friday

theorem nedy_crackers : total = 24 := by sorry

end NUMINAMATH_CALUDE_nedy_crackers_l2778_277809


namespace NUMINAMATH_CALUDE_range_of_a_l2778_277802

def A : Set ℝ := {x | x^2 - 5*x + 4 > 0}

def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + (a+2) = 0}

theorem range_of_a (a : ℝ) : 
  (A ∩ B a).Nonempty → a ∈ {x | x < -1 ∨ x > 18/7} := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2778_277802


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2778_277805

theorem complex_equation_solution (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (a - 2 * i) * i = b - i) : a + b * i = -1 + 2 * i :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2778_277805


namespace NUMINAMATH_CALUDE_right_triangle_special_case_l2778_277882

/-- In a right triangle ABC with leg lengths a and b, and hypotenuse length c,
    if c = 2a + 1, then b^2 = 3a^2 + 4a + 1 -/
theorem right_triangle_special_case (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2)  -- Pythagorean theorem
  (h5 : c = 2*a + 1)      -- Given condition
  : b^2 = 3*a^2 + 4*a + 1 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_special_case_l2778_277882


namespace NUMINAMATH_CALUDE_circle_radius_proof_l2778_277870

theorem circle_radius_proof (a : ℝ) : 
  a > 0 ∧ 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y = 2*x^2 - 27 ∧ (x - a)^2 + (y - (2*a^2 - 27))^2 = a^2) ∧
  a^2 = (4*a - 3*(2*a^2 - 27))^2 / (4^2 + 3^2) →
  a = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l2778_277870


namespace NUMINAMATH_CALUDE_complex_power_simplification_l2778_277822

theorem complex_power_simplification :
  ((1 + 2 * Complex.I) / (1 - 2 * Complex.I)) ^ 1012 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_simplification_l2778_277822


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_five_l2778_277844

/-- A geometric sequence with common ratio not equal to 1 -/
structure GeometricSequence where
  a : ℕ → ℚ
  q : ℚ
  q_ne_one : q ≠ 1
  geom_prop : ∀ n : ℕ, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
def sum_n (g : GeometricSequence) (n : ℕ) : ℚ :=
  (g.a 1) * (1 - g.q^n) / (1 - g.q)

theorem geometric_sequence_sum_five (g : GeometricSequence) 
  (h1 : g.a 1 * g.a 2 * g.a 3 * g.a 4 * g.a 5 = 1 / 1024)
  (h2 : 2 * g.a 4 = g.a 2 + g.a 3) : 
  sum_n g 5 = 11 / 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_five_l2778_277844


namespace NUMINAMATH_CALUDE_intersection_M_N_l2778_277897

def M : Set ℝ := {x | 1 - 2/x < 0}
def N : Set ℝ := {x | -1 ≤ x}

theorem intersection_M_N : ∀ x : ℝ, x ∈ M ∩ N ↔ 0 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2778_277897


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_2017_l2778_277825

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_arithmetic_sequence (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n : ℤ) * a 1 + (n * (n - 1) : ℤ) * (a 2 - a 1) / 2

theorem arithmetic_sequence_sum_2017 
  (a : ℕ → ℤ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_first_term : a 1 = -2015)
  (h_sum_condition : sum_arithmetic_sequence a 6 - 2 * sum_arithmetic_sequence a 3 = 18) :
  sum_arithmetic_sequence a 2017 = 2017 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_2017_l2778_277825


namespace NUMINAMATH_CALUDE_valid_ten_digit_numbers_l2778_277868

/-- Sequence representing the number of n-digit numbers satisfying the conditions -/
def a : ℕ → ℕ
  | 0 => 0  -- Additional base case for completeness
  | 1 => 2  -- Base case: a_1 = 2
  | 2 => 3  -- Base case: a_2 = 3
  | (n + 3) => a (n + 2) + a (n + 1)  -- Recurrence relation

/-- Theorem stating that the number of valid 10-digit numbers is 144 -/
theorem valid_ten_digit_numbers : a 10 = 144 := by
  sorry

end NUMINAMATH_CALUDE_valid_ten_digit_numbers_l2778_277868


namespace NUMINAMATH_CALUDE_perpendicular_unit_vectors_l2778_277877

def a : ℝ × ℝ := (4, 2)

theorem perpendicular_unit_vectors :
  let v₁ : ℝ × ℝ := (Real.sqrt 5 / 5, -2 * Real.sqrt 5 / 5)
  let v₂ : ℝ × ℝ := (-Real.sqrt 5 / 5, 2 * Real.sqrt 5 / 5)
  (v₁.1 * a.1 + v₁.2 * a.2 = 0 ∧ v₁.1^2 + v₁.2^2 = 1) ∧
  (v₂.1 * a.1 + v₂.2 * a.2 = 0 ∧ v₂.1^2 + v₂.2^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_unit_vectors_l2778_277877


namespace NUMINAMATH_CALUDE_expression_evaluation_l2778_277861

theorem expression_evaluation : (1/3)⁻¹ - 2 * Real.cos (30 * π / 180) - |2 - Real.sqrt 3| - (4 - Real.pi)^0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2778_277861


namespace NUMINAMATH_CALUDE_hannah_strawberries_l2778_277801

theorem hannah_strawberries (daily_harvest : ℕ) (days : ℕ) (stolen : ℕ) (remaining : ℕ) :
  daily_harvest = 5 →
  days = 30 →
  stolen = 30 →
  remaining = 100 →
  daily_harvest * days - stolen - remaining = 20 :=
by sorry

end NUMINAMATH_CALUDE_hannah_strawberries_l2778_277801


namespace NUMINAMATH_CALUDE_expression_evaluation_l2778_277863

theorem expression_evaluation : 
  let x : ℕ := 3
  x + x^2 * (x^(x^2)) = 177150 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2778_277863


namespace NUMINAMATH_CALUDE_pens_probability_theorem_l2778_277878

def total_pens : ℕ := 8
def red_pens : ℕ := 4
def blue_pens : ℕ := 4
def pens_to_pick : ℕ := 4

def probability_leftmost_blue_not_picked_rightmost_red_picked : ℚ :=
  4 / 49

theorem pens_probability_theorem :
  let total_arrangements := Nat.choose total_pens red_pens
  let total_pick_ways := Nat.choose total_pens pens_to_pick
  let favorable_red_arrangements := Nat.choose (total_pens - 2) (red_pens - 1)
  let favorable_pick_ways := Nat.choose (total_pens - 2) (pens_to_pick - 1)
  (favorable_red_arrangements * favorable_pick_ways : ℚ) / (total_arrangements * total_pick_ways) =
    probability_leftmost_blue_not_picked_rightmost_red_picked :=
by
  sorry

#check pens_probability_theorem

end NUMINAMATH_CALUDE_pens_probability_theorem_l2778_277878


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2778_277826

theorem constant_term_binomial_expansion :
  ∃ (c : ℝ), c = 15 ∧ 
  ∀ x : ℝ, (fun y : ℝ => (4^y - 2^(-y))^6) x = c + (fun z : ℝ => z - c) ((4^x - 2^(-x))^6) :=
sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2778_277826


namespace NUMINAMATH_CALUDE_fraction_equality_l2778_277853

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 10)
  (h2 : p / n = 2)
  (h3 : p / q = 1 / 5) :
  m / q = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2778_277853
