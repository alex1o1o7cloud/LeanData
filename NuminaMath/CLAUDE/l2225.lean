import Mathlib

namespace NUMINAMATH_CALUDE_max_nonmanagers_for_nine_managers_l2225_222528

/-- Represents a department in the corporation -/
structure Department where
  managers : ℕ
  nonManagers : ℕ

/-- The conditions for a valid department -/
def isValidDepartment (d : Department) : Prop :=
  d.managers > 0 ∧
  d.managers * 37 > 7 * d.nonManagers ∧
  d.managers ≥ 5 ∧
  d.managers + d.nonManagers ≤ 300 ∧
  d.managers = (d.managers + d.nonManagers) * 12 / 100

theorem max_nonmanagers_for_nine_managers :
  ∀ d : Department,
    isValidDepartment d →
    d.managers = 9 →
    d.nonManagers ≤ 66 :=
by sorry

end NUMINAMATH_CALUDE_max_nonmanagers_for_nine_managers_l2225_222528


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l2225_222577

theorem necessary_not_sufficient (a b h : ℝ) (h_pos : h > 0) :
  (∀ a b : ℝ, |a - 1| < h ∧ |b - 1| < h → |a - b| < 2 * h) ∧
  (∃ a b : ℝ, |a - b| < 2 * h ∧ ¬(|a - 1| < h ∧ |b - 1| < h)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l2225_222577


namespace NUMINAMATH_CALUDE_gym_equipment_cost_l2225_222516

/-- The cost of replacing all cardio machines in a global gym chain --/
theorem gym_equipment_cost (num_gyms : ℕ) (num_bikes : ℕ) (num_treadmills : ℕ) (num_ellipticals : ℕ)
  (treadmill_cost_factor : ℚ) (elliptical_cost_factor : ℚ) (total_cost : ℚ) :
  num_gyms = 20 →
  num_bikes = 10 →
  num_treadmills = 5 →
  num_ellipticals = 5 →
  treadmill_cost_factor = 3/2 →
  elliptical_cost_factor = 2 →
  total_cost = 455000 →
  ∃ (bike_cost : ℚ),
    bike_cost = 700 ∧
    total_cost = num_gyms * (num_bikes * bike_cost +
                             num_treadmills * treadmill_cost_factor * bike_cost +
                             num_ellipticals * elliptical_cost_factor * treadmill_cost_factor * bike_cost) :=
by sorry

end NUMINAMATH_CALUDE_gym_equipment_cost_l2225_222516


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2225_222525

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 4 * m * x - 4 < 0) ↔ -1 < m ∧ m ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2225_222525


namespace NUMINAMATH_CALUDE_iggy_running_time_l2225_222551

/-- Represents the daily running distances in miles -/
def daily_miles : List Nat := [3, 4, 6, 8, 3]

/-- Represents the pace in minutes per mile -/
def pace : Nat := 10

/-- Calculates the total running time in hours -/
def total_running_hours (miles : List Nat) (pace : Nat) : Nat :=
  (miles.sum * pace) / 60

/-- Theorem: Iggy's total running time from Monday to Friday is 4 hours -/
theorem iggy_running_time :
  total_running_hours daily_miles pace = 4 := by
  sorry

#eval total_running_hours daily_miles pace

end NUMINAMATH_CALUDE_iggy_running_time_l2225_222551


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l2225_222513

/-- Calculates the total surface area of a cube with holes cut through each face --/
def total_surface_area (cube_edge : ℝ) (hole_side : ℝ) : ℝ :=
  let original_surface_area := 6 * cube_edge^2
  let hole_area := 6 * hole_side^2
  let exposed_area := 6 * 4 * hole_side^2
  original_surface_area - hole_area + exposed_area

/-- Theorem stating that the total surface area of the given cube with holes is 222 square meters --/
theorem cube_with_holes_surface_area :
  total_surface_area 5 2 = 222 := by
  sorry

#eval total_surface_area 5 2

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l2225_222513


namespace NUMINAMATH_CALUDE_expression_value_l2225_222580

theorem expression_value : 3^2 * 7 + 5 * 4^2 - 45 / 3 = 128 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2225_222580


namespace NUMINAMATH_CALUDE_intersection_determines_B_l2225_222583

def A : Set ℝ := {0, 1, 2, 3}

def B (m : ℝ) : Set ℝ := {x | x^2 - 5*x + m = 0}

theorem intersection_determines_B :
  ∃ m : ℝ, (A ∩ B m = {1}) → (B m = {1, 4}) := by sorry

end NUMINAMATH_CALUDE_intersection_determines_B_l2225_222583


namespace NUMINAMATH_CALUDE_pencil_distribution_l2225_222509

theorem pencil_distribution (initial_pencils : ℕ) (containers : ℕ) (additional_pencils : ℕ)
  (h1 : initial_pencils = 150)
  (h2 : containers = 5)
  (h3 : additional_pencils = 30) :
  (initial_pencils + additional_pencils) / containers = 36 :=
by sorry

end NUMINAMATH_CALUDE_pencil_distribution_l2225_222509


namespace NUMINAMATH_CALUDE_sum_11_is_negative_11_l2225_222542

/-- An arithmetic sequence with its sum of terms -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- Sum of first n terms
  first_term : a 1 = -11
  sum_condition : S 10 / 10 - S 8 / 8 = 2

/-- The sum of the first 11 terms in the given arithmetic sequence is -11 -/
theorem sum_11_is_negative_11 (seq : ArithmeticSequence) : seq.S 11 = -11 := by
  sorry

end NUMINAMATH_CALUDE_sum_11_is_negative_11_l2225_222542


namespace NUMINAMATH_CALUDE_train_speed_conversion_l2225_222599

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

/-- Conversion factor from hours to seconds -/
def h_to_s : ℝ := 3600

/-- Speed of the train in km/h -/
def train_speed_kmh : ℝ := 162

/-- Theorem stating that 162 km/h is equal to 45 m/s -/
theorem train_speed_conversion :
  (train_speed_kmh * km_to_m) / h_to_s = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_conversion_l2225_222599


namespace NUMINAMATH_CALUDE_x_one_minus_f_equals_four_power_500_l2225_222576

/-- Given x = (3 + √5)^500, n = ⌊x⌋, and f = x - n, prove that x(1 - f) = 4^500 -/
theorem x_one_minus_f_equals_four_power_500 :
  let x : ℝ := (3 + Real.sqrt 5) ^ 500
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 4 ^ 500 := by
  sorry

end NUMINAMATH_CALUDE_x_one_minus_f_equals_four_power_500_l2225_222576


namespace NUMINAMATH_CALUDE_inequality_solution_range_of_a_l2225_222515

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 1|
def g (x a : ℝ) : ℝ := 2 * |x| + a

-- Part 1: Inequality solution
theorem inequality_solution :
  {x : ℝ | f x ≤ g x (-1)} = {x : ℝ | x ≤ -2/3 ∨ x ≥ 2} :=
sorry

-- Part 2: Range of a
theorem range_of_a (h : ∃ x₀ : ℝ, f x₀ ≥ (1/2) * g x₀ a) :
  a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_of_a_l2225_222515


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2225_222548

theorem trigonometric_identity (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi) (h3 : -Real.sin α = 2 * Real.cos α) :
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2225_222548


namespace NUMINAMATH_CALUDE_least_non_lucky_multiple_of_11_l2225_222519

def sumOfDigits (n : ℕ) : ℕ := sorry

def isLuckyInteger (n : ℕ) : Prop :=
  n > 0 ∧ n % (sumOfDigits n) = 0

def isMultipleOf11 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 11 * k

theorem least_non_lucky_multiple_of_11 :
  (isMultipleOf11 132) ∧
  ¬(isLuckyInteger 132) ∧
  ∀ n : ℕ, n > 0 ∧ n < 132 ∧ (isMultipleOf11 n) → (isLuckyInteger n) := by
  sorry

end NUMINAMATH_CALUDE_least_non_lucky_multiple_of_11_l2225_222519


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_value_l2225_222501

theorem no_linear_term_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x + m) * (x + 3) = a * x^2 + b) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_value_l2225_222501


namespace NUMINAMATH_CALUDE_statement_is_false_l2225_222579

-- Define the AM-GM inequality for positive real numbers
axiom am_gm_inequality {a b : ℝ} (ha : 0 < a) (hb : 0 < b) : a + b ≥ 2 * Real.sqrt (a * b)

-- Define the statement we want to prove false
def statement_to_disprove : Prop := ∀ x : ℝ, x + 1 / x ≥ 2

-- Theorem stating that the above statement is false
theorem statement_is_false : ¬statement_to_disprove := by
  sorry

end NUMINAMATH_CALUDE_statement_is_false_l2225_222579


namespace NUMINAMATH_CALUDE_min_speed_for_race_l2225_222586

/-- Proves that the minimum speed required to travel 5 kilometers in 20 minutes is 15 km/h -/
theorem min_speed_for_race (distance : ℝ) (time_minutes : ℝ) (speed : ℝ) : 
  distance = 5 → 
  time_minutes = 20 → 
  speed = distance / (time_minutes / 60) → 
  speed = 15 := by sorry

end NUMINAMATH_CALUDE_min_speed_for_race_l2225_222586


namespace NUMINAMATH_CALUDE_power_of_three_mod_eight_l2225_222560

theorem power_of_three_mod_eight : 3^2023 % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_eight_l2225_222560


namespace NUMINAMATH_CALUDE_integer_solutions_equation_l2225_222503

theorem integer_solutions_equation :
  ∀ (a b c : ℤ),
    c ≤ 94 →
    (a + Real.sqrt c)^2 + (b + Real.sqrt c)^2 = 60 + 20 * Real.sqrt c →
    ((a = 3 ∧ b = 7 ∧ c = 41) ∨
     (a = 4 ∧ b = 6 ∧ c = 44) ∨
     (a = 5 ∧ b = 5 ∧ c = 45) ∨
     (a = 6 ∧ b = 4 ∧ c = 44) ∨
     (a = 7 ∧ b = 3 ∧ c = 41)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_equation_l2225_222503


namespace NUMINAMATH_CALUDE_larger_number_proof_l2225_222521

theorem larger_number_proof (x y : ℕ) 
  (h1 : y - x = 1365) 
  (h2 : y = 6 * x + 15) : 
  y = 1635 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2225_222521


namespace NUMINAMATH_CALUDE_lcm_75_120_l2225_222569

theorem lcm_75_120 : Nat.lcm 75 120 = 600 := by
  sorry

end NUMINAMATH_CALUDE_lcm_75_120_l2225_222569


namespace NUMINAMATH_CALUDE_crayons_distribution_l2225_222514

/-- Given a total number of crayons and a number of boxes, 
    calculate the number of crayons per box -/
def crayons_per_box (total_crayons : ℕ) (num_boxes : ℕ) : ℕ :=
  total_crayons / num_boxes

/-- Theorem stating that given 80 crayons and 10 boxes, 
    the number of crayons per box is 8 -/
theorem crayons_distribution :
  crayons_per_box 80 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_crayons_distribution_l2225_222514


namespace NUMINAMATH_CALUDE_abs_neg_gt_neg_implies_positive_l2225_222566

theorem abs_neg_gt_neg_implies_positive (a : ℝ) : |(-a)| > -a → a > 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_gt_neg_implies_positive_l2225_222566


namespace NUMINAMATH_CALUDE_smallest_product_l2225_222536

def S : Finset Int := {-7, -5, -1, 1, 3}

theorem smallest_product (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int) (hx : x ∈ S) (hy : y ∈ S), x * y = -21 ∧ ∀ (c d : Int), c ∈ S → d ∈ S → x * y ≤ c * d :=
sorry

end NUMINAMATH_CALUDE_smallest_product_l2225_222536


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2225_222500

/-- The minimum value of 1/m + 1/n given the conditions -/
theorem min_value_sum_reciprocals (a m n : ℝ) (ha : a > 0) (ha' : a ≠ 1)
  (hmn : m * n > 0) (h_line : -2 * m - n + 1 = 0) :
  (1 / m + 1 / n) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2225_222500


namespace NUMINAMATH_CALUDE_max_value_of_f_l2225_222552

-- Define the function f on [1, 4]
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- State the theorem
theorem max_value_of_f :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x ∈ Set.Icc 1 4, f x = x^2 - 4*x + 5) →  -- f(x) = x^2 - 4x + 5 for x ∈ [1, 4]
  (∃ c ∈ Set.Icc (-4) (-1), ∀ x ∈ Set.Icc (-4) (-1), f x ≤ f c) →  -- maximum exists on [-4, -1]
  (∀ x ∈ Set.Icc (-4) (-1), f x ≤ -1) ∧  -- maximum value is at most -1
  (∃ x ∈ Set.Icc (-4) (-1), f x = -1)  -- maximum value -1 is achieved
  := by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2225_222552


namespace NUMINAMATH_CALUDE_puppies_sold_l2225_222575

theorem puppies_sold (initial_puppies : ℕ) (puppies_per_cage : ℕ) (cages_used : ℕ) : 
  initial_puppies = 13 → puppies_per_cage = 2 → cages_used = 3 →
  initial_puppies - (puppies_per_cage * cages_used) = 7 := by
  sorry

end NUMINAMATH_CALUDE_puppies_sold_l2225_222575


namespace NUMINAMATH_CALUDE_equation_system_solution_l2225_222541

theorem equation_system_solution (x z : ℝ) 
  (eq1 : 3 * x^2 + 9 * x + 7 * z + 2 = 0)
  (eq2 : 3 * x + z + 4 = 0) :
  z^2 + 20 * z - 14 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l2225_222541


namespace NUMINAMATH_CALUDE_bananas_per_box_l2225_222589

/-- Given 40 bananas and 8 boxes, prove that 5 bananas must go in each box. -/
theorem bananas_per_box (total_bananas : ℕ) (num_boxes : ℕ) 
  (h1 : total_bananas = 40) (h2 : num_boxes = 8) : 
  total_bananas / num_boxes = 5 := by
  sorry

#check bananas_per_box

end NUMINAMATH_CALUDE_bananas_per_box_l2225_222589


namespace NUMINAMATH_CALUDE_rectangle_to_square_l2225_222526

/-- A rectangle can be cut into three parts to form a square --/
theorem rectangle_to_square :
  ∃ (a b c : ℕ × ℕ),
    -- The original rectangle is 25 × 4
    25 * 4 = (a.1 * a.2) + (b.1 * b.2) + (c.1 * c.2) ∧
    -- The three parts can form a square
    ∃ (s : ℕ), s * s = (a.1 * a.2) + (b.1 * b.2) + (c.1 * c.2) ∧
    -- There are exactly three parts
    a ≠ b ∧ b ≠ c ∧ a ≠ c :=
by sorry


end NUMINAMATH_CALUDE_rectangle_to_square_l2225_222526


namespace NUMINAMATH_CALUDE_problem_statement_problem_statement_2_l2225_222592

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

theorem problem_statement (a : ℝ) (h1 : a > 1) :
  (∀ x ∈ Set.Icc 1 a, f a x ∈ Set.Icc 1 a ∧ ∀ y ∈ Set.Icc 1 a, ∃ x ∈ Set.Icc 1 a, f a x = y) →
  a = 2 :=
sorry

theorem problem_statement_2 (a : ℝ) (h1 : a > 1) :
  (∀ x y : ℝ, x < y ∧ y ≤ 2 → f a x > f a y) ∧
  (∀ x ∈ Set.Icc 1 2, f a x ≤ 0) →
  a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_problem_statement_problem_statement_2_l2225_222592


namespace NUMINAMATH_CALUDE_constant_grid_values_l2225_222563

theorem constant_grid_values (f : ℤ × ℤ → ℕ) 
  (h : ∀ (x y : ℤ), f (x, y) = (f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1)) / 4) : 
  ∃ (c : ℕ), ∀ (x y : ℤ), f (x, y) = c :=
sorry

end NUMINAMATH_CALUDE_constant_grid_values_l2225_222563


namespace NUMINAMATH_CALUDE_circle_circumference_increase_l2225_222558

theorem circle_circumference_increase (d : ℝ) : 
  let original_circumference := π * d
  let new_circumference := π * (d + 2 * π)
  new_circumference - original_circumference = 2 * π^2 := by
sorry

end NUMINAMATH_CALUDE_circle_circumference_increase_l2225_222558


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l2225_222568

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define our function f
noncomputable def f (x : ℝ) : ℝ :=
  (floor x : ℝ) + Real.sqrt (x - floor x)

-- State the theorem
theorem f_strictly_increasing :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f x₁ < f x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l2225_222568


namespace NUMINAMATH_CALUDE_highest_numbered_street_l2225_222540

/-- Represents the length of Apple Street in meters -/
def street_length : ℕ := 15000

/-- Represents the distance between intersections in meters -/
def intersection_distance : ℕ := 500

/-- Calculates the number of numbered intersecting streets -/
def numbered_intersections : ℕ :=
  (street_length / intersection_distance) - 2

/-- Proves that the highest-numbered street is the 28th Street -/
theorem highest_numbered_street :
  numbered_intersections = 28 := by
  sorry

end NUMINAMATH_CALUDE_highest_numbered_street_l2225_222540


namespace NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l2225_222505

/-- Given three numbers in ratio 5:7:9 with LCM 6300, their sum is 14700 -/
theorem sum_of_numbers_in_ratio (a b c : ℕ) : 
  (a : ℚ) / 5 = (b : ℚ) / 7 ∧ (b : ℚ) / 7 = (c : ℚ) / 9 →
  Nat.lcm a (Nat.lcm b c) = 6300 →
  a + b + c = 14700 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l2225_222505


namespace NUMINAMATH_CALUDE_ceiling_product_equation_l2225_222574

theorem ceiling_product_equation : ∃ x : ℝ, ⌈x⌉ * x = 210 ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_product_equation_l2225_222574


namespace NUMINAMATH_CALUDE_tournament_probability_l2225_222561

/-- Represents a soccer tournament with the given conditions -/
structure SoccerTournament where
  num_teams : ℕ
  games_per_team : ℕ
  win_probability : ℝ

/-- Calculates the probability of team A finishing with more points than team B -/
def probability_A_beats_B (tournament : SoccerTournament) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem tournament_probability : 
  let tournament := SoccerTournament.mk 7 6 (1/2)
  probability_A_beats_B tournament = 319/512 := by sorry

end NUMINAMATH_CALUDE_tournament_probability_l2225_222561


namespace NUMINAMATH_CALUDE_pentagonal_pyramid_edges_l2225_222578

-- Define a pentagonal pyramid
structure PentagonalPyramid where
  base : Pentagon
  triangular_faces : Fin 5 → Triangle
  common_vertex : Point

-- Define the number of edges in a pentagonal pyramid
def num_edges_pentagonal_pyramid (pp : PentagonalPyramid) : ℕ := 10

-- Theorem statement
theorem pentagonal_pyramid_edges (pp : PentagonalPyramid) :
  num_edges_pentagonal_pyramid pp = 10 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_pyramid_edges_l2225_222578


namespace NUMINAMATH_CALUDE_max_2012_gons_sharing_vertices_not_sides_target_2012_gons_impossible_l2225_222590

/-- The number of vertices in each polygon -/
def n : ℕ := 2012

/-- The maximum number of different polygons we want to prove is impossible -/
def target_polygons : ℕ := 1006

/-- The actual maximum number of polygons possible -/
def max_polygons : ℕ := 1005

theorem max_2012_gons_sharing_vertices_not_sides :
  ∀ (num_polygons : ℕ),
    (∀ (v : Fin n), num_polygons * 2 ≤ n - 1) →
    num_polygons ≤ max_polygons :=
by sorry

theorem target_2012_gons_impossible :
  ¬(∀ (v : Fin n), target_polygons * 2 ≤ n - 1) :=
by sorry

end NUMINAMATH_CALUDE_max_2012_gons_sharing_vertices_not_sides_target_2012_gons_impossible_l2225_222590


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l2225_222506

theorem closest_integer_to_cube_root (x : ℝ) : 
  x = (7^3 + 9^3 + 10 : ℝ)^(1/3) → 
  ∃ (n : ℤ), n = 10 ∧ ∀ (m : ℤ), |x - n| ≤ |x - m| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l2225_222506


namespace NUMINAMATH_CALUDE_total_contribution_l2225_222565

def contribution_problem (niraj brittany angela : ℕ) : Prop :=
  brittany = 3 * niraj ∧
  angela = 3 * brittany ∧
  niraj = 80

theorem total_contribution :
  ∀ niraj brittany angela : ℕ,
  contribution_problem niraj brittany angela →
  niraj + brittany + angela = 1040 :=
by
  sorry

end NUMINAMATH_CALUDE_total_contribution_l2225_222565


namespace NUMINAMATH_CALUDE_probability_consecutive_cards_l2225_222562

/-- A type representing the cards labeled A, B, C, D, E -/
inductive Card : Type
  | A | B | C | D | E

/-- A function to check if two cards are consecutive -/
def consecutive (c1 c2 : Card) : Bool :=
  match c1, c2 with
  | Card.A, Card.B | Card.B, Card.A => true
  | Card.B, Card.C | Card.C, Card.B => true
  | Card.C, Card.D | Card.D, Card.C => true
  | Card.D, Card.E | Card.E, Card.D => true
  | _, _ => false

/-- The total number of ways to choose 2 cards from 5 -/
def totalChoices : Nat := 10

/-- The number of ways to choose 2 consecutive cards -/
def consecutiveChoices : Nat := 4

/-- Theorem stating the probability of drawing two consecutive cards -/
theorem probability_consecutive_cards :
  (consecutiveChoices : ℚ) / totalChoices = 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_probability_consecutive_cards_l2225_222562


namespace NUMINAMATH_CALUDE_average_marks_all_candidates_l2225_222582

/-- Proves that the average marks of all candidates is 35 given the specified conditions -/
theorem average_marks_all_candidates
  (total_candidates : ℕ)
  (passed_candidates : ℕ)
  (failed_candidates : ℕ)
  (avg_marks_passed : ℚ)
  (avg_marks_failed : ℚ)
  (h1 : total_candidates = 120)
  (h2 : passed_candidates = 100)
  (h3 : failed_candidates = total_candidates - passed_candidates)
  (h4 : avg_marks_passed = 39)
  (h5 : avg_marks_failed = 15) :
  (passed_candidates * avg_marks_passed + failed_candidates * avg_marks_failed) / total_candidates = 35 :=
by
  sorry

#check average_marks_all_candidates

end NUMINAMATH_CALUDE_average_marks_all_candidates_l2225_222582


namespace NUMINAMATH_CALUDE_c_work_time_l2225_222556

-- Define the work rates of a, b, and c
variable (A B C : ℝ)

-- Define the conditions
def condition1 : Prop := A + B = 1 / 6
def condition2 : Prop := B + C = 1 / 8
def condition3 : Prop := C + A = 1 / 12

-- Theorem statement
theorem c_work_time (h1 : condition1 A B) (h2 : condition2 B C) (h3 : condition3 C A) :
  1 / C = 48 := by sorry

end NUMINAMATH_CALUDE_c_work_time_l2225_222556


namespace NUMINAMATH_CALUDE_max_ab_value_l2225_222594

theorem max_ab_value (a b c : ℝ) (h1 : a + b + c = 4) (h2 : 3*a + 2*b - c = 0) :
  ∀ x y : ℝ, x + y + c = 4 → 3*x + 2*y - c = 0 → x*y ≤ a*b ∧ a*b = 1/3 :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l2225_222594


namespace NUMINAMATH_CALUDE_linear_function_proof_l2225_222598

def is_linear (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x, f x = a * x + b

theorem linear_function_proof (f : ℝ → ℝ) 
  (h_linear : is_linear f) 
  (h_composite : ∀ x, f (f x) = 4 * x - 1) 
  (h_specific : f 3 = -5) : 
  ∀ x, f x = -2 * x + 1 := by
sorry

end NUMINAMATH_CALUDE_linear_function_proof_l2225_222598


namespace NUMINAMATH_CALUDE_strawberry_jelly_sales_l2225_222520

/-- Represents the number of jars sold for each type of jelly -/
structure JellySales where
  grape : ℕ
  strawberry : ℕ
  raspberry : ℕ
  plum : ℕ

/-- Defines the relationships between jelly sales and proves the number of strawberry jelly jars sold -/
theorem strawberry_jelly_sales (sales : JellySales) : 
  sales.grape = 2 * sales.strawberry ∧ 
  sales.raspberry = 2 * sales.plum ∧
  sales.raspberry = sales.grape / 3 ∧
  sales.plum = 6 →
  sales.strawberry = 18 := by
sorry

end NUMINAMATH_CALUDE_strawberry_jelly_sales_l2225_222520


namespace NUMINAMATH_CALUDE_max_area_rectangle_l2225_222584

/-- The maximum area of a rectangle with integer side lengths and perimeter 150 feet is 1406 square feet. -/
theorem max_area_rectangle (w h : ℕ) : 
  w + h = 75 → w * h ≤ 1406 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l2225_222584


namespace NUMINAMATH_CALUDE_inequality_holds_l2225_222524

theorem inequality_holds (x : ℝ) : (1 : ℝ) / (x^2 + 1) > (1 : ℝ) / (x^2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l2225_222524


namespace NUMINAMATH_CALUDE_product_of_digits_of_largest_valid_number_l2225_222537

/-- A function that returns true if the digits of a natural number are in strictly increasing order --/
def strictly_increasing_digits (n : ℕ) : Prop := sorry

/-- A function that returns the sum of the squares of the digits of a natural number --/
def sum_of_squared_digits (n : ℕ) : ℕ := sorry

/-- A function that returns the product of the digits of a natural number --/
def product_of_digits (n : ℕ) : ℕ := sorry

/-- The largest natural number whose digits are in strictly increasing order and whose digits' squares sum to 50 --/
def largest_valid_number : ℕ := sorry

theorem product_of_digits_of_largest_valid_number : 
  strictly_increasing_digits largest_valid_number ∧ 
  sum_of_squared_digits largest_valid_number = 50 ∧
  product_of_digits largest_valid_number = 36 ∧
  ∀ m : ℕ, 
    strictly_increasing_digits m ∧ 
    sum_of_squared_digits m = 50 → 
    m ≤ largest_valid_number :=
sorry

end NUMINAMATH_CALUDE_product_of_digits_of_largest_valid_number_l2225_222537


namespace NUMINAMATH_CALUDE_paco_cookies_l2225_222588

/-- Calculates the number of cookies Paco bought given the initial, eaten, and final cookie counts. -/
def cookies_bought (initial : ℕ) (eaten : ℕ) (final : ℕ) : ℕ :=
  final - (initial - eaten)

/-- Proves that Paco bought 37 cookies given the problem conditions. -/
theorem paco_cookies : cookies_bought 40 2 75 = 37 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_l2225_222588


namespace NUMINAMATH_CALUDE_land_plots_area_l2225_222573

theorem land_plots_area (x y z : ℝ) (h1 : x = (2/5) * (x + y + z))
  (h2 : y / z = (3/2) / (4/3)) (h3 : z = x - 16) :
  x + y + z = 96 := by
  sorry

end NUMINAMATH_CALUDE_land_plots_area_l2225_222573


namespace NUMINAMATH_CALUDE_zoo_revenue_calculation_l2225_222508

/-- Calculates the total revenue for a zoo over two days with given attendance and pricing information --/
def zoo_revenue (
  monday_children monday_adults monday_seniors : ℕ)
  (tuesday_children tuesday_adults tuesday_seniors : ℕ)
  (monday_child_price monday_adult_price monday_senior_price : ℚ)
  (tuesday_child_price tuesday_adult_price tuesday_senior_price : ℚ)
  (tuesday_discount : ℚ) : ℚ :=
  let monday_total := 
    monday_children * monday_child_price + 
    monday_adults * monday_adult_price + 
    monday_seniors * monday_senior_price
  let tuesday_total := 
    tuesday_children * tuesday_child_price + 
    tuesday_adults * tuesday_adult_price + 
    tuesday_seniors * tuesday_senior_price
  let tuesday_discounted := tuesday_total * (1 - tuesday_discount)
  monday_total + tuesday_discounted

theorem zoo_revenue_calculation : 
  zoo_revenue 7 5 3 9 6 2 3 4 3 4 5 3 (1/10) = 114.8 := by
  sorry

end NUMINAMATH_CALUDE_zoo_revenue_calculation_l2225_222508


namespace NUMINAMATH_CALUDE_point_on_line_l2225_222535

/-- A line in the xy-plane with slope m and y-intercept b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

/-- Theorem: For a line with slope 4 and y-intercept 4, 
    the point (199, 800) lies on this line -/
theorem point_on_line : 
  let l : Line := { m := 4, b := 4 }
  let p : Point := { x := 199, y := 800 }
  p.onLine l := by sorry

end NUMINAMATH_CALUDE_point_on_line_l2225_222535


namespace NUMINAMATH_CALUDE_min_blocks_for_wall_l2225_222591

/-- Represents a block in the wall -/
inductive Block
| OneFootBlock
| TwoFootBlock

/-- Represents a row of blocks in the wall -/
def Row := List Block

/-- The wall specification -/
structure WallSpec where
  length : Nat
  height : Nat
  blockHeight : Nat
  evenEnds : Bool
  staggeredJoins : Bool

/-- Checks if a row of blocks is valid according to the wall specification -/
def isValidRow (spec : WallSpec) (row : Row) : Prop := sorry

/-- Checks if a list of rows forms a valid wall according to the specification -/
def isValidWall (spec : WallSpec) (rows : List Row) : Prop := sorry

/-- Counts the total number of blocks in a list of rows -/
def countBlocks (rows : List Row) : Nat := sorry

/-- The main theorem to be proved -/
theorem min_blocks_for_wall (spec : WallSpec) : 
  spec.length = 102 ∧ 
  spec.height = 8 ∧ 
  spec.blockHeight = 1 ∧ 
  spec.evenEnds = true ∧ 
  spec.staggeredJoins = true → 
  ∃ (rows : List Row), 
    isValidWall spec rows ∧ 
    countBlocks rows = 416 ∧ 
    ∀ (otherRows : List Row), 
      isValidWall spec otherRows → 
      countBlocks otherRows ≥ 416 := by sorry

end NUMINAMATH_CALUDE_min_blocks_for_wall_l2225_222591


namespace NUMINAMATH_CALUDE_root_sum_product_l2225_222567

theorem root_sum_product (p q : ℝ) : 
  (Complex.I * 2 - 1)^2 + p * (Complex.I * 2 - 1) + q = 0 → p + q = 7 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_product_l2225_222567


namespace NUMINAMATH_CALUDE_clown_balloons_l2225_222518

/-- The number of balloons a clown has after blowing up two sets of balloons -/
def total_balloons (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that the clown has 60 balloons after blowing up 47 and then 13 more -/
theorem clown_balloons :
  total_balloons 47 13 = 60 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloons_l2225_222518


namespace NUMINAMATH_CALUDE_exists_solution_a4_eq_b3_plus_c2_l2225_222544

theorem exists_solution_a4_eq_b3_plus_c2 : 
  ∃ (a b c : ℕ+), (a : ℝ)^4 = (b : ℝ)^3 + (c : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_solution_a4_eq_b3_plus_c2_l2225_222544


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2225_222517

/-- Represents an ellipse with semi-major axis 2 and semi-minor axis b -/
structure Ellipse (b : ℝ) :=
  (equation : ℝ → ℝ → Prop)
  (b_pos : b > 0)

/-- Represents a point on the ellipse -/
structure EllipsePoint (E : Ellipse b) :=
  (x y : ℝ)
  (on_ellipse : E.equation x y)

/-- The left focus of the ellipse -/
def left_focus (E : Ellipse b) : ℝ × ℝ := sorry

/-- The right focus of the ellipse -/
def right_focus (E : Ellipse b) : ℝ × ℝ := sorry

/-- A line passing through the left focus -/
structure FocalLine (E : Ellipse b) :=
  (passes_through_left_focus : Prop)

/-- Intersection points of a focal line with the ellipse -/
def intersection_points (E : Ellipse b) (l : FocalLine E) : EllipsePoint E × EllipsePoint E := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Maximum sum of distances from intersection points to the right focus -/
def max_sum_distances (E : Ellipse b) : ℝ := sorry

/-- Eccentricity of the ellipse -/
def eccentricity (E : Ellipse b) : ℝ := sorry

/-- Main theorem -/
theorem ellipse_eccentricity (b : ℝ) (E : Ellipse b) :
  max_sum_distances E = 5 → eccentricity E = 1/2 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2225_222517


namespace NUMINAMATH_CALUDE_jonathan_took_45_oranges_l2225_222595

/-- The number of oranges Jonathan took -/
def oranges_taken (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that Jonathan took 45 oranges -/
theorem jonathan_took_45_oranges :
  oranges_taken 96 51 = 45 := by
  sorry

end NUMINAMATH_CALUDE_jonathan_took_45_oranges_l2225_222595


namespace NUMINAMATH_CALUDE_die_roll_probability_l2225_222547

theorem die_roll_probability : 
  let n : ℕ := 8  -- number of rolls
  let p_even : ℚ := 1/2  -- probability of rolling an even number
  let p_odd : ℚ := 1 - p_even  -- probability of rolling an odd number
  1 - p_odd^n = 255/256 :=
by sorry

end NUMINAMATH_CALUDE_die_roll_probability_l2225_222547


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2225_222554

theorem simple_interest_rate_calculation (total_amount interest_difference amount_B : ℚ)
  (h1 : total_amount = 10000)
  (h2 : interest_difference = 360)
  (h3 : amount_B = 4000) :
  let amount_A := total_amount - amount_B
  let rate_A := 15 / 100
  let time := 2
  let interest_A := amount_A * rate_A * time
  let interest_B := interest_A - interest_difference
  let rate_B := interest_B / (amount_B * time)
  rate_B = 18 / 100 := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2225_222554


namespace NUMINAMATH_CALUDE_circle_diameter_l2225_222564

theorem circle_diameter (r : ℝ) (h : r > 0) : 
  3 * (2 * π * r) = 2 * (π * r^2) → 2 * r = 6 := by
sorry

end NUMINAMATH_CALUDE_circle_diameter_l2225_222564


namespace NUMINAMATH_CALUDE_complement_union_problem_l2225_222531

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-1, 2}
def B : Set Int := {-1, 0, 1}

theorem complement_union_problem : (U \ B) ∪ A = {-2, -1, 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_problem_l2225_222531


namespace NUMINAMATH_CALUDE_apple_price_reduction_l2225_222593

-- Define the given conditions
def reduced_price_per_dozen : ℚ := 3
def total_money : ℚ := 40
def additional_apples : ℕ := 64

-- Define the function to calculate the percentage reduction
def calculate_percentage_reduction (original_price reduced_price : ℚ) : ℚ :=
  ((original_price - reduced_price) / original_price) * 100

-- State the theorem
theorem apple_price_reduction :
  let dozens_at_reduced_price := total_money / reduced_price_per_dozen
  let additional_dozens := additional_apples / 12
  let dozens_at_original_price := dozens_at_reduced_price - additional_dozens
  let original_price_per_dozen := total_money / dozens_at_original_price
  calculate_percentage_reduction original_price_per_dozen reduced_price_per_dozen = 40 := by
  sorry

end NUMINAMATH_CALUDE_apple_price_reduction_l2225_222593


namespace NUMINAMATH_CALUDE_infinite_primes_dividing_S_l2225_222545

-- Define the set of positive integers
def PositiveIntegers : Set ℕ := {n : ℕ | n > 0}

-- Define an infinite subset of positive integers
def S : Set ℕ := sorry

-- Define the set T
def T (S : Set ℕ) : Set ℕ := 
  {z : ℕ | ∃ x y : ℕ, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ z = x + y}

-- Define the set of primes congruent to 1 modulo 4
def Primes1Mod4 : Set ℕ := {p : ℕ | Nat.Prime p ∧ p % 4 = 1}

-- Define the set of primes dividing some element of T
def PrimesDividingT (S : Set ℕ) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∃ t ∈ T S, p ∣ t}

-- Define the set of primes dividing some element of S
def PrimesDividingS (S : Set ℕ) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∃ s ∈ S, p ∣ s}

-- The main theorem
theorem infinite_primes_dividing_S :
  S.Infinite ∧ S ⊆ PositiveIntegers →
  (Primes1Mod4 ∩ PrimesDividingT S).Finite →
  (PrimesDividingS S).Infinite :=
sorry

end NUMINAMATH_CALUDE_infinite_primes_dividing_S_l2225_222545


namespace NUMINAMATH_CALUDE_quadratic_function_y_order_l2225_222527

/-- Given a quadratic function f(x) = -x² - 4x + m, where m is a constant,
    and three points A, B, C on its graph, prove that the y-coordinate of B
    is greater than that of A, which is greater than that of C. -/
theorem quadratic_function_y_order (m : ℝ) (y₁ y₂ y₃ : ℝ) : 
  ((-3)^2 + 4*(-3) + m = y₁) →
  ((-2)^2 + 4*(-2) + m = y₂) →
  (1^2 + 4*1 + m = y₃) →
  y₂ > y₁ ∧ y₁ > y₃ := by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_y_order_l2225_222527


namespace NUMINAMATH_CALUDE_fairview_soccer_contest_l2225_222539

/-- Calculates the number of penalty kicks in a soccer team contest --/
def penalty_kicks (total_players : ℕ) (initial_goalies : ℕ) (absent_players : ℕ) (absent_goalies : ℕ) : ℕ :=
  let remaining_players := total_players - absent_players
  let remaining_goalies := initial_goalies - absent_goalies
  remaining_goalies * (remaining_players - 1)

/-- Theorem stating the number of penalty kicks for the Fairview College Soccer Team contest --/
theorem fairview_soccer_contest : 
  penalty_kicks 25 4 2 1 = 66 := by
  sorry

end NUMINAMATH_CALUDE_fairview_soccer_contest_l2225_222539


namespace NUMINAMATH_CALUDE_candy_sampling_theorem_l2225_222512

theorem candy_sampling_theorem (caught_percentage : Real) (total_sampling_percentage : Real)
  (h1 : caught_percentage = 22)
  (h2 : total_sampling_percentage = 24.444444444444443) :
  total_sampling_percentage - caught_percentage = 2.444444444444443 := by
  sorry

end NUMINAMATH_CALUDE_candy_sampling_theorem_l2225_222512


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2225_222543

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (1 + m * Complex.I) / (1 - Complex.I)
  (∃ (y : ℝ), z = Complex.I * y) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2225_222543


namespace NUMINAMATH_CALUDE_problem_solution_l2225_222533

theorem problem_solution : (((3^1 : ℝ) + 2 + 6^2 + 3)⁻¹ * 6) = 3/22 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2225_222533


namespace NUMINAMATH_CALUDE_x_over_y_equals_four_l2225_222532

theorem x_over_y_equals_four (x y : ℝ) (h1 : y ≠ 0) (h2 : 2 * x - y = 1.75 * x) : x / y = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_over_y_equals_four_l2225_222532


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2225_222504

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -2) : 
  a^3 + b^3 = 45 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2225_222504


namespace NUMINAMATH_CALUDE_birds_on_fence_l2225_222572

theorem birds_on_fence (initial_birds final_birds joined_birds : ℕ) : 
  final_birds = initial_birds + joined_birds →
  joined_birds = 4 →
  final_birds = 5 →
  initial_birds = 1 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l2225_222572


namespace NUMINAMATH_CALUDE_smallest_sum_with_conditions_l2225_222571

def is_relatively_prime (x y : ℕ) : Prop := Nat.gcd x y = 1

theorem smallest_sum_with_conditions :
  ∃ (a b c d e : ℕ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    ¬(is_relatively_prime a b) ∧
    ¬(is_relatively_prime b c) ∧
    ¬(is_relatively_prime c d) ∧
    ¬(is_relatively_prime d e) ∧
    is_relatively_prime a c ∧
    is_relatively_prime a d ∧
    is_relatively_prime a e ∧
    is_relatively_prime b d ∧
    is_relatively_prime b e ∧
    is_relatively_prime c e ∧
    a + b + c + d + e = 75 ∧
    (∀ (a' b' c' d' e' : ℕ),
      a' > 0 → b' > 0 → c' > 0 → d' > 0 → e' > 0 →
      ¬(is_relatively_prime a' b') →
      ¬(is_relatively_prime b' c') →
      ¬(is_relatively_prime c' d') →
      ¬(is_relatively_prime d' e') →
      is_relatively_prime a' c' →
      is_relatively_prime a' d' →
      is_relatively_prime a' e' →
      is_relatively_prime b' d' →
      is_relatively_prime b' e' →
      is_relatively_prime c' e' →
      a' + b' + c' + d' + e' ≥ 75) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_with_conditions_l2225_222571


namespace NUMINAMATH_CALUDE_fraction_calculation_l2225_222538

theorem fraction_calculation : 
  (8 / 4 * 9 / 3 * 20 / 5) / (10 / 5 * 12 / 4 * 15 / 3) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2225_222538


namespace NUMINAMATH_CALUDE_truck_loading_time_l2225_222570

theorem truck_loading_time (rate1 rate2 : ℚ) (h1 : rate1 = 1 / 6) (h2 : rate2 = 1 / 5) :
  1 / (rate1 + rate2) = 30 / 11 := by
  sorry

end NUMINAMATH_CALUDE_truck_loading_time_l2225_222570


namespace NUMINAMATH_CALUDE_expand_binomials_l2225_222597

theorem expand_binomials (x y : ℝ) : (x + 5) * (3 * y + 15) = 3 * x * y + 15 * x + 15 * y + 75 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomials_l2225_222597


namespace NUMINAMATH_CALUDE_race_time_problem_l2225_222529

/-- Given two racers A and B, where their speeds are in the ratio 3:4 and A takes 30 minutes more
    than B to reach the destination, prove that A takes 120 minutes to reach the destination. -/
theorem race_time_problem (v_A v_B : ℝ) (t_A t_B : ℝ) (D : ℝ) :
  v_A / v_B = 3 / 4 →  -- speeds are in ratio 3:4
  t_A = t_B + 30 →     -- A takes 30 minutes more than B
  D = v_A * t_A →      -- distance = speed * time for A
  D = v_B * t_B →      -- distance = speed * time for B
  t_A = 120 :=         -- A takes 120 minutes
by sorry

end NUMINAMATH_CALUDE_race_time_problem_l2225_222529


namespace NUMINAMATH_CALUDE_x_intercept_of_line_x_intercept_specific_line_l2225_222549

/-- Given two points on a line, calculate its x-intercept -/
theorem x_intercept_of_line (x₁ y₁ x₂ y₂ : ℝ) (h : x₁ ≠ x₂) :
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (0 - b) / m = (m * x₁ - y₁) / m :=
by sorry

/-- The x-intercept of a line passing through (10, 3) and (-12, -8) is 4 -/
theorem x_intercept_specific_line :
  let x₁ : ℝ := 10
  let y₁ : ℝ := 3
  let x₂ : ℝ := -12
  let y₂ : ℝ := -8
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (0 - b) / m = 4 :=
by sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_x_intercept_specific_line_l2225_222549


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l2225_222596

/-- Rectangle with known side length and area -/
structure Rectangle1 where
  side : ℝ
  area : ℝ

/-- Rectangle similar to Rectangle1 with known diagonal -/
structure Rectangle2 where
  diagonal : ℝ

/-- The area of Rectangle2 given the properties of Rectangle1 -/
def area_rectangle2 (r1 : Rectangle1) (r2 : Rectangle2) : ℝ :=
  sorry

theorem rectangle_area_theorem (r1 : Rectangle1) (r2 : Rectangle2) :
  r1.side = 3 ∧ r1.area = 18 ∧ r2.diagonal = 20 → area_rectangle2 r1 r2 = 160 :=
sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l2225_222596


namespace NUMINAMATH_CALUDE_root_sum_of_coefficients_l2225_222555

theorem root_sum_of_coefficients (a b : ℝ) : 
  (Complex.I * 2 + 1) ^ 2 + a * (Complex.I * 2 + 1) + b = 0 → a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_of_coefficients_l2225_222555


namespace NUMINAMATH_CALUDE_square_of_8y_minus_2_l2225_222559

theorem square_of_8y_minus_2 (y : ℝ) (h : 4 * y^2 + 7 = 6 * y + 12) :
  (8 * y - 2)^2 = 248 := by
  sorry

end NUMINAMATH_CALUDE_square_of_8y_minus_2_l2225_222559


namespace NUMINAMATH_CALUDE_initial_garlic_cloves_l2225_222522

/-- 
Given that Maria used 86 cloves of garlic for roast chicken and has 7 cloves left,
prove that she initially stored 93 cloves of garlic.
-/
theorem initial_garlic_cloves (used : ℕ) (left : ℕ) (h1 : used = 86) (h2 : left = 7) :
  used + left = 93 := by
  sorry

end NUMINAMATH_CALUDE_initial_garlic_cloves_l2225_222522


namespace NUMINAMATH_CALUDE_miss_both_mutually_exclusive_not_contradictory_l2225_222523

-- Define the sample space for two shots
inductive ShotOutcome
| HitBoth
| HitFirst
| HitSecond
| MissBoth

-- Define the events
def hit_exactly_once (outcome : ShotOutcome) : Prop :=
  outcome = ShotOutcome.HitFirst ∨ outcome = ShotOutcome.HitSecond

def miss_both (outcome : ShotOutcome) : Prop :=
  outcome = ShotOutcome.MissBoth

-- Theorem stating that "Miss both times" is mutually exclusive but not contradictory to "hit exactly once"
theorem miss_both_mutually_exclusive_not_contradictory :
  (∀ outcome : ShotOutcome, ¬(hit_exactly_once outcome ∧ miss_both outcome)) ∧
  (∃ outcome : ShotOutcome, hit_exactly_once outcome ∨ miss_both outcome) :=
sorry

end NUMINAMATH_CALUDE_miss_both_mutually_exclusive_not_contradictory_l2225_222523


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_6_l2225_222502

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  s : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, s n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: For an arithmetic sequence with given conditions, S_6 = 6 -/
theorem arithmetic_sequence_sum_6 (seq : ArithmeticSequence) 
    (h1 : seq.a 1 = 6)
    (h2 : seq.a 3 + seq.a 5 = 0) : 
  seq.s 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_6_l2225_222502


namespace NUMINAMATH_CALUDE_regular_polygon_with_140_degree_interior_angles_is_nonagon_l2225_222507

theorem regular_polygon_with_140_degree_interior_angles_is_nonagon :
  ∀ (n : ℕ) (interior_angle : ℝ),
    n ≥ 3 →
    interior_angle = 140 →
    (n - 2) * 180 = n * interior_angle →
    n = 9 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_with_140_degree_interior_angles_is_nonagon_l2225_222507


namespace NUMINAMATH_CALUDE_perimeter_of_specific_quadrilateral_l2225_222587

structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  h_positive : 0 < AB ∧ 0 < BC ∧ 0 < CD ∧ 0 < DA

def perimeter (q : Quadrilateral) : ℝ :=
  q.AB + q.BC + q.CD + q.DA

theorem perimeter_of_specific_quadrilateral :
  ∃ (q : Quadrilateral), 
    q.DA < q.BC ∧
    q.DA = 4 ∧
    q.AB = 5 ∧
    q.BC = 10 ∧
    q.CD = 7 ∧
    perimeter q = 26 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_specific_quadrilateral_l2225_222587


namespace NUMINAMATH_CALUDE_crayons_given_to_friends_l2225_222581

theorem crayons_given_to_friends (crayons_lost : ℕ) (total_crayons_lost_or_given : ℕ) 
  (h1 : crayons_lost = 535)
  (h2 : total_crayons_lost_or_given = 587) :
  total_crayons_lost_or_given - crayons_lost = 52 := by
  sorry

end NUMINAMATH_CALUDE_crayons_given_to_friends_l2225_222581


namespace NUMINAMATH_CALUDE_least_n_for_indefinite_play_l2225_222546

/-- A box in the game, containing a certain number of coins. -/
structure Box where
  label : ℕ
  coins : ℕ

/-- The game state, consisting of a list of boxes. -/
structure GameState where
  boxes : List Box

/-- Perform a single step of the game. -/
def gameStep (state : GameState) (k : ℕ) : Option GameState :=
  sorry

/-- Check if the game can be played indefinitely from a given state. -/
def canPlayIndefinitely (initialState : GameState) (k : ℕ) : Prop :=
  sorry

/-- The main theorem: the least n ≥ k+1 for which the game can be played indefinitely. -/
theorem least_n_for_indefinite_play (k : ℕ) :
  let n := 2^k + k - 1
  let initialState : GameState :=
    { boxes := List.range n |>.map (λ i => { label := i + 1, coins := i + 1 }) }
  (∀ m : ℕ, m ≥ k + 1 → m < n →
    ¬ canPlayIndefinitely
      { boxes := List.range m |>.map (λ i => { label := i + 1, coins := i + 1 }) }
      k) ∧
  canPlayIndefinitely initialState k :=
by
  sorry

end NUMINAMATH_CALUDE_least_n_for_indefinite_play_l2225_222546


namespace NUMINAMATH_CALUDE_log_equation_l2225_222534

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation : (log10 5)^2 + log10 2 * log10 50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_l2225_222534


namespace NUMINAMATH_CALUDE_triangle_property_l2225_222585

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.c * Real.cos t.C = (t.a * Real.cos t.B + t.b * Real.cos t.A) / 2)
  (h2 : t.c = 2) :
  t.C = π / 3 ∧ 
  (∀ (t' : Triangle), t'.c = 2 → t.a + t.b + t.c ≥ t'.a + t'.b + t'.c) ∧
  t.a + t.b + t.c = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l2225_222585


namespace NUMINAMATH_CALUDE_area_of_LMNOPQ_l2225_222511

/-- Represents a rectangle with side lengths a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.a * r.b

/-- Represents the polygon LMNOPQ formed by two overlapping rectangles -/
structure PolygonLMNOPQ where
  lmno : Rectangle
  opqr : Rectangle
  lm : ℝ
  mn : ℝ
  no : ℝ
  -- Conditions
  h1 : lmno.a = lm
  h2 : lmno.b = mn
  h3 : opqr.a = mn
  h4 : opqr.b = lm
  h5 : lm = 8
  h6 : mn = 10
  h7 : no = 3

theorem area_of_LMNOPQ (p : PolygonLMNOPQ) : p.lmno.area = 80 := by
  sorry

#check area_of_LMNOPQ

end NUMINAMATH_CALUDE_area_of_LMNOPQ_l2225_222511


namespace NUMINAMATH_CALUDE_arrangements_with_space_theorem_l2225_222530

/-- The number of arrangements of 6 people in a row where person A and person B
    have at least one person between them. -/
def arrangements_with_space_between (total_arrangements : ℕ) 
                                    (adjacent_arrangements : ℕ) : ℕ :=
  total_arrangements - adjacent_arrangements

/-- Theorem stating that the number of arrangements of 6 people in a row
    where person A and person B have at least one person between them is 480. -/
theorem arrangements_with_space_theorem :
  arrangements_with_space_between 720 240 = 480 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_space_theorem_l2225_222530


namespace NUMINAMATH_CALUDE_domain_exclusion_sum_l2225_222550

theorem domain_exclusion_sum (A B : ℝ) : 
  (∀ x : ℝ, 3 * x^2 - 9 * x + 6 = 0 ↔ x = A ∨ x = B) → A + B = 3 := by
sorry

end NUMINAMATH_CALUDE_domain_exclusion_sum_l2225_222550


namespace NUMINAMATH_CALUDE_slices_left_over_l2225_222557

-- Define the number of slices for each pizza size
def small_pizza_slices : ℕ := 4
def large_pizza_slices : ℕ := 8

-- Define the number of pizzas purchased
def small_pizzas_bought : ℕ := 3
def large_pizzas_bought : ℕ := 2

-- Define the number of slices each person eats
def george_slices : ℕ := 3
def bob_slices : ℕ := george_slices + 1
def susie_slices : ℕ := bob_slices / 2
def bill_slices : ℕ := 3
def fred_slices : ℕ := 3
def mark_slices : ℕ := 3

-- Calculate total slices and slices eaten
def total_slices : ℕ := small_pizza_slices * small_pizzas_bought + large_pizza_slices * large_pizzas_bought
def total_slices_eaten : ℕ := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices

-- Theorem to prove
theorem slices_left_over : total_slices - total_slices_eaten = 10 := by
  sorry

end NUMINAMATH_CALUDE_slices_left_over_l2225_222557


namespace NUMINAMATH_CALUDE_arithmetic_mean_4_16_l2225_222553

theorem arithmetic_mean_4_16 (x : ℝ) : x = (4 + 16) / 2 → x = 10 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_4_16_l2225_222553


namespace NUMINAMATH_CALUDE_a_range_proof_l2225_222510

-- Define the propositions p and q
def p (x a : ℝ) : Prop := -4 < x - a ∧ x - a < 4

def q (x : ℝ) : Prop := (x - 1) * (x - 3) < 0

-- Define the range of a
def a_range (a : ℝ) : Prop := -1 ≤ a ∧ a ≤ 5

-- State the theorem
theorem a_range_proof :
  (∀ x a : ℝ, q x → p x a) ∧  -- q is sufficient for p
  (∃ x a : ℝ, p x a ∧ ¬(q x)) ∧  -- q is not necessary for p
  (∀ a : ℝ, a_range a ↔ ∀ x : ℝ, q x → p x a) :=
by sorry

end NUMINAMATH_CALUDE_a_range_proof_l2225_222510
