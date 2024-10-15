import Mathlib

namespace NUMINAMATH_CALUDE_door_height_calculation_l1948_194887

/-- Calculates the height of a door in a room given the room dimensions, 
    whitewashing cost, window dimensions, and total cost. -/
theorem door_height_calculation 
  (room_length room_width room_height : ℝ)
  (door_width : ℝ)
  (window_width window_height : ℝ)
  (num_windows : ℕ)
  (whitewash_cost_per_sqft : ℝ)
  (total_cost : ℝ) :
  room_length = 25 ∧ 
  room_width = 15 ∧ 
  room_height = 12 ∧
  door_width = 6 ∧
  window_width = 4 ∧
  window_height = 3 ∧
  num_windows = 3 ∧
  whitewash_cost_per_sqft = 2 ∧
  total_cost = 1812 →
  ∃ (door_height : ℝ),
    door_height = 3 ∧
    total_cost = whitewash_cost_per_sqft * 
      (2 * (room_length + room_width) * room_height - 
       (door_width * door_height + num_windows * window_width * window_height)) :=
by sorry

end NUMINAMATH_CALUDE_door_height_calculation_l1948_194887


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_18_l1948_194803

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_9_with_digit_sum_18 :
  ∀ n : ℕ, is_three_digit n → n % 9 = 0 → digit_sum n = 18 → n ≤ 990 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_18_l1948_194803


namespace NUMINAMATH_CALUDE_smallest_abcd_l1948_194806

/-- Represents a four-digit number ABCD -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_range : a ∈ Finset.range 10
  b_range : b ∈ Finset.range 10
  c_range : c ∈ Finset.range 10
  d_range : d ∈ Finset.range 10
  all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Converts a FourDigitNumber to its numerical value -/
def FourDigitNumber.toNat (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- Represents the conditions of the problem -/
structure ProblemConditions where
  ab : Nat
  a : Nat
  b : Nat
  ab_two_digit : ab ∈ Finset.range 100
  ab_eq : ab = 10 * a + b
  a_not_eq_b : a ≠ b
  result : FourDigitNumber
  multiplication_condition : ab * a = result.toNat

/-- The main theorem stating that the smallest ABCD satisfying the conditions is 2046 -/
theorem smallest_abcd (conditions : ProblemConditions) :
  ∀ other : FourDigitNumber,
    (∃ other_conditions : ProblemConditions, other_conditions.result = other) →
    conditions.result.toNat ≤ other.toNat ∧ conditions.result.toNat = 2046 := by
  sorry


end NUMINAMATH_CALUDE_smallest_abcd_l1948_194806


namespace NUMINAMATH_CALUDE_element_in_set_l1948_194832

theorem element_in_set : ∀ (M : Set ℕ), M = {0, 1, 2} → 0 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l1948_194832


namespace NUMINAMATH_CALUDE_symmetrical_line_passes_through_point_l1948_194884

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The line y = x -/
def lineYEqX : Line := ⟨1, -1, 0⟩

/-- Get the symmetrical line with respect to y = x -/
def symmetricalLine (l : Line) : Line :=
  ⟨l.b, l.a, l.c⟩

theorem symmetrical_line_passes_through_point :
  let l₁ : Line := ⟨2, 1, -1⟩  -- y = -2x + 1 rewritten as 2x + y - 1 = 0
  let l₂ := symmetricalLine l₁
  let p : Point := ⟨3, -1⟩
  pointOnLine l₂ p := by sorry

end NUMINAMATH_CALUDE_symmetrical_line_passes_through_point_l1948_194884


namespace NUMINAMATH_CALUDE_total_campers_rowing_hiking_l1948_194870

/-- The total number of campers who went rowing and hiking -/
theorem total_campers_rowing_hiking 
  (morning_rowing : ℕ) 
  (morning_hiking : ℕ) 
  (afternoon_rowing : ℕ) 
  (h1 : morning_rowing = 41)
  (h2 : morning_hiking = 4)
  (h3 : afternoon_rowing = 26) :
  morning_rowing + morning_hiking + afternoon_rowing = 71 := by
  sorry

end NUMINAMATH_CALUDE_total_campers_rowing_hiking_l1948_194870


namespace NUMINAMATH_CALUDE_ratio_expression_value_l1948_194801

theorem ratio_expression_value (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ratio_expression_value_l1948_194801


namespace NUMINAMATH_CALUDE_intersection_sum_l1948_194834

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 5*x < 0}
def N (p : ℝ) : Set ℝ := {x | p < x ∧ x < 6}

-- Define the intersection of M and N
def M_intersect_N (p q : ℝ) : Set ℝ := {x | 2 < x ∧ x < q}

-- Theorem statement
theorem intersection_sum (p q : ℝ) : 
  M ∩ N p = M_intersect_N p q → p + q = 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l1948_194834


namespace NUMINAMATH_CALUDE_oliver_new_socks_l1948_194874

theorem oliver_new_socks (initial_socks : ℕ) (thrown_away : ℕ) (final_socks : ℕ)
  (h1 : initial_socks = 11)
  (h2 : thrown_away = 4)
  (h3 : final_socks = 33) :
  final_socks - (initial_socks - thrown_away) = 26 := by
  sorry

end NUMINAMATH_CALUDE_oliver_new_socks_l1948_194874


namespace NUMINAMATH_CALUDE_contrapositive_square_sum_zero_l1948_194813

theorem contrapositive_square_sum_zero (m n : ℝ) :
  (¬(mn = 0) → ¬(m^2 + n^2 = 0)) ↔ (m^2 + n^2 = 0 → mn = 0) := by sorry

end NUMINAMATH_CALUDE_contrapositive_square_sum_zero_l1948_194813


namespace NUMINAMATH_CALUDE_julia_internet_speed_l1948_194829

-- Define the given conditions
def songs_downloaded : ℕ := 7200
def download_time_minutes : ℕ := 30
def song_size_mb : ℕ := 5

-- Define the internet speed calculation function
def calculate_internet_speed (songs : ℕ) (time_minutes : ℕ) (size_mb : ℕ) : ℚ :=
  (songs * size_mb : ℚ) / (time_minutes * 60 : ℚ)

-- Theorem statement
theorem julia_internet_speed :
  calculate_internet_speed songs_downloaded download_time_minutes song_size_mb = 20 := by
  sorry


end NUMINAMATH_CALUDE_julia_internet_speed_l1948_194829


namespace NUMINAMATH_CALUDE_children_education_expense_l1948_194825

def monthly_salary (saved_amount : ℚ) (savings_rate : ℚ) : ℚ :=
  saved_amount / savings_rate

def total_expenses (rent milk groceries petrol misc education : ℚ) : ℚ :=
  rent + milk + groceries + petrol + misc + education

theorem children_education_expense 
  (rent milk groceries petrol misc : ℚ)
  (savings_rate saved_amount : ℚ)
  (h1 : rent = 5000)
  (h2 : milk = 1500)
  (h3 : groceries = 4500)
  (h4 : petrol = 2000)
  (h5 : misc = 5200)
  (h6 : savings_rate = 1/10)
  (h7 : saved_amount = 2300)
  : ∃ (education : ℚ), 
    education = 2500 ∧ 
    total_expenses rent milk groceries petrol misc education = 
      monthly_salary saved_amount savings_rate := by
  sorry

end NUMINAMATH_CALUDE_children_education_expense_l1948_194825


namespace NUMINAMATH_CALUDE_no_parallel_axes_in_bounded_figures_parallel_axes_in_unbounded_figures_intersecting_axes_in_all_figures_l1948_194861

-- Define a spatial geometric figure
structure SpatialFigure where
  isBounded : Bool

-- Define an axis of symmetry
structure SymmetryAxis where
  figure : SpatialFigure

-- Define a relation for parallel axes
def areParallel (a1 a2 : SymmetryAxis) : Prop :=
  sorry

-- Define a relation for intersecting axes
def areIntersecting (a1 a2 : SymmetryAxis) : Prop :=
  sorry

-- Theorem 1: Bounded figures cannot have parallel axes of symmetry
theorem no_parallel_axes_in_bounded_figures (f : SpatialFigure) (h : f.isBounded) :
  ¬∃ (a1 a2 : SymmetryAxis), a1.figure = f ∧ a2.figure = f ∧ areParallel a1 a2 :=
sorry

-- Theorem 2: Unbounded figures can have parallel axes of symmetry
theorem parallel_axes_in_unbounded_figures :
  ∃ (f : SpatialFigure) (a1 a2 : SymmetryAxis), 
    ¬f.isBounded ∧ a1.figure = f ∧ a2.figure = f ∧ areParallel a1 a2 :=
sorry

-- Theorem 3: Both bounded and unbounded figures can have intersecting axes of symmetry
theorem intersecting_axes_in_all_figures :
  (∃ (f : SpatialFigure) (a1 a2 : SymmetryAxis), 
    f.isBounded ∧ a1.figure = f ∧ a2.figure = f ∧ areIntersecting a1 a2) ∧
  (∃ (f : SpatialFigure) (a1 a2 : SymmetryAxis), 
    ¬f.isBounded ∧ a1.figure = f ∧ a2.figure = f ∧ areIntersecting a1 a2) :=
sorry

end NUMINAMATH_CALUDE_no_parallel_axes_in_bounded_figures_parallel_axes_in_unbounded_figures_intersecting_axes_in_all_figures_l1948_194861


namespace NUMINAMATH_CALUDE_roots_of_quadratic_sum_of_eighth_powers_l1948_194862

theorem roots_of_quadratic_sum_of_eighth_powers (a b : ℂ) : 
  (a^2 - 2*a + 5 = 0) → (b^2 - 2*b + 5 = 0) → Complex.abs (a^8 + b^8) = 1054 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_sum_of_eighth_powers_l1948_194862


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1948_194820

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 = 198 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1948_194820


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l1948_194871

/-- A regular nonagon is a 9-sided regular polygon -/
def RegularNonagon : Type := Unit

/-- The number of vertices in a regular nonagon -/
def num_vertices : ℕ := 9

/-- The number of diagonals in a regular nonagon -/
def num_diagonals (n : RegularNonagon) : ℕ := (num_vertices.choose 2) - num_vertices

/-- The number of ways to choose 2 diagonals from all diagonals -/
def num_diagonal_pairs (n : RegularNonagon) : ℕ := (num_diagonals n).choose 2

/-- The number of ways to choose 4 vertices that form a convex quadrilateral -/
def num_intersecting_pairs (n : RegularNonagon) : ℕ := num_vertices.choose 4

/-- The probability that two randomly chosen diagonals intersect -/
def intersection_probability (n : RegularNonagon) : ℚ :=
  (num_intersecting_pairs n : ℚ) / (num_diagonal_pairs n : ℚ)

theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  intersection_probability n = 14 / 39 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l1948_194871


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l1948_194897

theorem difference_of_squares_factorization (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l1948_194897


namespace NUMINAMATH_CALUDE_boys_without_calculators_l1948_194802

theorem boys_without_calculators (total_students : ℕ) (total_boys : ℕ) (students_with_calculators : ℕ) (girls_with_calculators : ℕ) 
  (h1 : total_students = 40)
  (h2 : total_boys = 20)
  (h3 : students_with_calculators = 30)
  (h4 : girls_with_calculators = 18) :
  total_boys - (students_with_calculators - girls_with_calculators) = 8 := by
  sorry

end NUMINAMATH_CALUDE_boys_without_calculators_l1948_194802


namespace NUMINAMATH_CALUDE_sqrt_15_bounds_l1948_194875

theorem sqrt_15_bounds : 3 < Real.sqrt 15 ∧ Real.sqrt 15 < 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_15_bounds_l1948_194875


namespace NUMINAMATH_CALUDE_gum_sharing_l1948_194872

/-- The number of pieces of gum each person will receive when shared equally --/
def gum_per_person (john cole aubrey maria : ℕ) : ℕ :=
  (john + cole + aubrey + maria) / 6

/-- Theorem stating that given the initial gum distribution, each person will receive 34 pieces --/
theorem gum_sharing (john cole aubrey maria : ℕ) 
  (h_john : john = 54)
  (h_cole : cole = 45)
  (h_aubrey : aubrey = 37)
  (h_maria : maria = 70) :
  gum_per_person john cole aubrey maria = 34 := by
  sorry

end NUMINAMATH_CALUDE_gum_sharing_l1948_194872


namespace NUMINAMATH_CALUDE_ceiling_sum_equality_l1948_194827

theorem ceiling_sum_equality : 
  ⌈Real.sqrt (16/9 : ℝ)⌉ + ⌈(16/9 : ℝ)⌉ + ⌈((16/9 : ℝ)^2)⌉ = 8 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_equality_l1948_194827


namespace NUMINAMATH_CALUDE_same_heads_probability_l1948_194819

/-- The number of possible outcomes when tossing two pennies -/
def keiko_outcomes : ℕ := 4

/-- The number of possible outcomes when tossing three pennies -/
def ephraim_outcomes : ℕ := 8

/-- The number of ways Keiko and Ephraim can get the same number of heads -/
def matching_outcomes : ℕ := 7

/-- The total number of possible outcomes when Keiko tosses two pennies and Ephraim tosses three pennies -/
def total_outcomes : ℕ := keiko_outcomes * ephraim_outcomes

/-- The probability that Ephraim gets the same number of heads as Keiko -/
def probability : ℚ := matching_outcomes / total_outcomes

theorem same_heads_probability : probability = 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_same_heads_probability_l1948_194819


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1948_194808

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The sum of specific terms in the sequence equals 120 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sum_condition a) : 
  2 * a 10 - a 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1948_194808


namespace NUMINAMATH_CALUDE_orange_cost_l1948_194894

/-- Given that three dozen oranges cost $18.00, prove that four dozen oranges at the same rate cost $24.00 -/
theorem orange_cost (cost_three_dozen : ℝ) (h1 : cost_three_dozen = 18) :
  let cost_per_dozen := cost_three_dozen / 3
  cost_per_dozen * 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_l1948_194894


namespace NUMINAMATH_CALUDE_inequality_proof_l1948_194869

theorem inequality_proof (a b c d : ℝ) :
  (a^2 - a + 1) * (b^2 - b + 1) * (c^2 - c + 1) * (d^2 - d + 1) ≥ 
  9/16 * (a - b) * (b - c) * (c - d) * (d - a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1948_194869


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1948_194856

/-- The parabola is defined by the equation y^2 = 8x -/
def is_parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (2, 0)

/-- The equation of the directrix -/
def directrix_x : ℝ := -2

/-- The distance from a point to a vertical line -/
def distance_to_line (p : ℝ × ℝ) (line_x : ℝ) : ℝ :=
  |p.1 - line_x|

theorem parabola_focus_directrix_distance :
  distance_to_line focus directrix_x = 4 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1948_194856


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1948_194844

theorem polynomial_coefficient_sum (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) : 
  (∀ x : ℝ, x^6 - 2*x^5 + 3*x^4 - 3*x^3 + 3*x^2 - 2*x + 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃)) →
  b₁*c₁ + b₂*c₂ + b₃*c₃ = -2 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1948_194844


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1948_194860

theorem sphere_surface_area (V : ℝ) (h : V = 72 * Real.pi) :
  ∃ (r : ℝ), V = (4 / 3) * Real.pi * r^3 ∧
              4 * Real.pi * r^2 = 36 * Real.pi * 2^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1948_194860


namespace NUMINAMATH_CALUDE_longest_chord_line_eq_l1948_194809

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- Represents a line in 2D space -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Given a circle and a point inside it, returns the line containing the longest chord passing through the point -/
def longestChordLine (c : Circle) (m : Point) : Line :=
  sorry

/-- The theorem stating that the longest chord line passing through M(3, -1) in the given circle has the equation x + 2y - 2 = 0 -/
theorem longest_chord_line_eq (c : Circle) (m : Point) :
  c.equation = (fun x y => x^2 + y^2 - 4*x + y - 2 = 0) →
  m = ⟨3, -1⟩ →
  (longestChordLine c m).equation = (fun x y => x + 2*y - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_longest_chord_line_eq_l1948_194809


namespace NUMINAMATH_CALUDE_min_sum_squares_on_line_l1948_194836

/-- Given points M(1, 0) and N(-1, 0), and P(x, y) on the line 2x - y - 1 = 0,
    the minimum value of PM^2 + PN^2 is 2/5, achieved when P is at (1/5, -3/5) -/
theorem min_sum_squares_on_line :
  let M : ℝ × ℝ := (1, 0)
  let N : ℝ × ℝ := (-1, 0)
  let P : ℝ → ℝ × ℝ := fun x => (x, 2*x - 1)
  let dist_squared (a b : ℝ × ℝ) : ℝ := (a.1 - b.1)^2 + (a.2 - b.2)^2
  let sum_dist_squared (x : ℝ) : ℝ := dist_squared (P x) M + dist_squared (P x) N
  ∀ x : ℝ, sum_dist_squared x ≥ 2/5 ∧ 
    (sum_dist_squared (1/5) = 2/5 ∧ P (1/5) = (1/5, -3/5)) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_on_line_l1948_194836


namespace NUMINAMATH_CALUDE_floor_plus_one_l1948_194804

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the ceiling function
noncomputable def ceil (x : ℝ) : ℤ :=
  -Int.floor (-x)

-- Statement to prove
theorem floor_plus_one (x : ℝ) : floor (x + 1) = floor x + 1 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_one_l1948_194804


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l1948_194854

/-- For a hyperbola x²/a² - y²/b² = 1 with a > b, if the angle between asymptotes is 45°, then a/b = 1 -/
theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : a > 0) (h3 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → 
  (angle_between_asymptotes = Real.pi / 4) → 
  a / b = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l1948_194854


namespace NUMINAMATH_CALUDE_problem_statement_l1948_194805

theorem problem_statement (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -1) :
  2 * a + 2 * b - 3 * a * b = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1948_194805


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_coinciding_foci_l1948_194851

/-- The value of b^2 for an ellipse and hyperbola with coinciding foci -/
theorem ellipse_hyperbola_coinciding_foci (b : ℝ) : 
  (∃ (x y : ℝ), x^2/25 + y^2/b^2 = 1) ∧ 
  (∃ (x y : ℝ), x^2/169 - y^2/144 = 1/36) ∧
  (∀ (x y : ℝ), x^2/25 + y^2/b^2 = 1 ↔ x^2/169 - y^2/144 = 1/36) →
  b^2 = 587/36 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_coinciding_foci_l1948_194851


namespace NUMINAMATH_CALUDE_hotel_room_charge_difference_l1948_194873

theorem hotel_room_charge_difference (G R P : ℝ) : 
  P = G * 0.9 →
  R = G * 1.19999999999999986 →
  (R - P) / R * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_hotel_room_charge_difference_l1948_194873


namespace NUMINAMATH_CALUDE_joan_found_six_shells_l1948_194885

/-- The number of seashells Jessica found -/
def jessica_shells : ℕ := 8

/-- The total number of seashells Joan and Jessica found together -/
def total_shells : ℕ := 14

/-- The number of seashells Joan found -/
def joan_shells : ℕ := total_shells - jessica_shells

theorem joan_found_six_shells : joan_shells = 6 := by
  sorry

end NUMINAMATH_CALUDE_joan_found_six_shells_l1948_194885


namespace NUMINAMATH_CALUDE_student_pairing_fraction_l1948_194817

theorem student_pairing_fraction (t s : ℕ) (ht : t > 0) (hs : s > 0) :
  (t / 4 : ℚ) = (3 * s / 7 : ℚ) →
  (3 * s / 7 : ℚ) / ((t : ℚ) + (s : ℚ)) = 3 / 19 := by
sorry

end NUMINAMATH_CALUDE_student_pairing_fraction_l1948_194817


namespace NUMINAMATH_CALUDE_expression_value_l1948_194839

theorem expression_value : ∀ x y : ℝ, x = 2 ∧ y = 3 → x^3 + y^2 * (x^2 * y) = 116 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1948_194839


namespace NUMINAMATH_CALUDE_triangle_properties_l1948_194852

/-- Theorem about a specific triangle ABC -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists with sides a, b, c opposite to angles A, B, C
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  0 < a ∧ 0 < b ∧ 0 < c →
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Law of sines
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  -- Given conditions
  (2 * c - a) * Real.cos B = b * Real.cos A →
  3 * a + b = 2 * c →
  b = 2 →
  1 / Real.sin A + 1 / Real.sin C = 4 * Real.sqrt 3 / 3 →
  -- Conclusions
  Real.cos C = -1/7 ∧ 
  (1/2 * a * c * Real.sin B : ℝ) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1948_194852


namespace NUMINAMATH_CALUDE_no_grades_four_or_five_l1948_194843

theorem no_grades_four_or_five (n : ℕ) (x : ℕ) : 
  (5 : ℕ) ≠ 0 → -- There are 5 problems
  n * x + (x + 1) = 25 → -- Total problems solved
  9 ≤ n + 1 → -- At least 9 students (including Peter)
  x + 1 ≤ 5 → -- Maximum grade is 5
  ¬(x = 3 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_no_grades_four_or_five_l1948_194843


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1948_194814

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a > 0)
  (x₁ x₂ : ℝ)
  (hroots : ∀ x, f a b c x - x = 0 ↔ x = x₁ ∨ x = x₂)
  (horder : 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1/a) :
  (∀ x, 0 < x ∧ x < x₁ → x < f a b c x ∧ f a b c x < x₁) ∧
  (-b / (2*a) < x₁ / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1948_194814


namespace NUMINAMATH_CALUDE_max_comic_books_l1948_194828

theorem max_comic_books (cost : ℚ) (budget : ℚ) (h1 : cost = 25/20) (h2 : budget = 10) :
  ⌊budget / cost⌋ = 8 := by
sorry

end NUMINAMATH_CALUDE_max_comic_books_l1948_194828


namespace NUMINAMATH_CALUDE_jean_calories_consumption_l1948_194866

/-- Calculates the total calories consumed by Jean while writing her paper. -/
def total_calories (pages_written : ℕ) (pages_per_donut : ℕ) (calories_per_donut : ℕ) : ℕ :=
  (pages_written / pages_per_donut) * calories_per_donut

/-- Proves that Jean consumes 900 calories while writing her paper. -/
theorem jean_calories_consumption :
  total_calories 12 2 150 = 900 := by
  sorry

#eval total_calories 12 2 150

end NUMINAMATH_CALUDE_jean_calories_consumption_l1948_194866


namespace NUMINAMATH_CALUDE_fixed_point_power_function_l1948_194857

theorem fixed_point_power_function (f : ℝ → ℝ) :
  (∃ α : ℝ, ∀ x : ℝ, f x = x ^ α) →
  f 2 = Real.sqrt 2 / 2 →
  f 9 = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_power_function_l1948_194857


namespace NUMINAMATH_CALUDE_prime_even_intersection_l1948_194800

-- Define the set of prime numbers
def P : Set ℕ := {n : ℕ | Nat.Prime n}

-- Define the set of even numbers
def Q : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 2 * k}

-- Theorem statement
theorem prime_even_intersection :
  P ∩ Q = {2} :=
sorry

end NUMINAMATH_CALUDE_prime_even_intersection_l1948_194800


namespace NUMINAMATH_CALUDE_function_properties_l1948_194895

noncomputable section

variables (a b : ℝ)

def f (x : ℝ) : ℝ := x + a * x^2 + b * Real.log x

theorem function_properties :
  (f a b 1 = 0 ∧ (deriv (f a b)) 1 = 2) →
  (a = -1 ∧ b = 3 ∧ ∀ x > 0, f a b x ≤ 2 * x - 2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1948_194895


namespace NUMINAMATH_CALUDE_square_sum_theorem_l1948_194855

theorem square_sum_theorem (r s : ℝ) (h1 : r * s = 16) (h2 : r + s = 10) : r^2 + s^2 = 68 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l1948_194855


namespace NUMINAMATH_CALUDE_min_points_to_win_correct_l1948_194867

/-- Represents a chess tournament with 6 players where each player plays 2 games against every other player. -/
structure ChessTournament where
  num_players : ℕ
  games_per_pair : ℕ
  win_points : ℚ
  draw_points : ℚ
  loss_points : ℚ

/-- The minimum number of points needed to guarantee a player has more points than any other player -/
def min_points_to_win (t : ChessTournament) : ℚ := 9.5

/-- Theorem stating that 9.5 points is the minimum required to guarantee winning the tournament -/
theorem min_points_to_win_correct (t : ChessTournament) 
  (h1 : t.num_players = 6)
  (h2 : t.games_per_pair = 2)
  (h3 : t.win_points = 1)
  (h4 : t.draw_points = 0.5)
  (h5 : t.loss_points = 0) :
  ∀ (p : ℚ), p < min_points_to_win t → 
  ∃ (other_player_points : ℚ), other_player_points ≥ p ∧ other_player_points ≤ (t.num_players - 1) * t.games_per_pair * t.win_points :=
sorry

end NUMINAMATH_CALUDE_min_points_to_win_correct_l1948_194867


namespace NUMINAMATH_CALUDE_max_power_sum_l1948_194840

theorem max_power_sum (c d : ℕ) : 
  d > 1 → 
  c^d < 630 → 
  (∀ (x y : ℕ), y > 1 → x^y < 630 → c^d ≥ x^y) → 
  c + d = 27 := by
sorry

end NUMINAMATH_CALUDE_max_power_sum_l1948_194840


namespace NUMINAMATH_CALUDE_problem_solution_l1948_194807

theorem problem_solution (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 - 3*x^2 + 1
  let g : ℝ → ℝ := λ x ↦ -x^3 + 3*x^2 + x - 7
  (f x + g x = x - 6) → (g x = -x^3 + 3*x^2 + x - 7) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1948_194807


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1948_194847

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 21| + |x - 17| = |2*x - 38| :=
by
  -- The unique solution is x = 19
  use 19
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1948_194847


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1948_194898

def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + (y - 3)^2) + Real.sqrt ((x + 7)^2 + (y + 2)^2) = 24

def focus1 : ℝ × ℝ := (1, 3)
def focus2 : ℝ × ℝ := (-7, -2)

theorem ellipse_foci_distance :
  let d := Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2)
  d = Real.sqrt 89 := by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1948_194898


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1948_194845

-- Define the ellipse D
def ellipse_D (x y : ℝ) : Prop := x^2 / 50 + y^2 / 25 = 1

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 9

-- Define the hyperbola G
def hyperbola_G (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the foci of ellipse D
def foci_D : Set (ℝ × ℝ) := {(-5, 0), (5, 0)}

-- Theorem statement
theorem hyperbola_equation :
  ∀ x y : ℝ,
  (∀ x' y' : ℝ, ellipse_D x' y' → foci_D = {(-5, 0), (5, 0)}) →
  (∀ x' y' : ℝ, hyperbola_G x' y' → foci_D = {(-5, 0), (5, 0)}) →
  (∃ a b : ℝ, ∀ x' y' : ℝ, (b * x' = a * y' ∨ b * x' = -a * y') →
    ∃ t : ℝ, circle_M (x' + t) (y' + t)) →
  hyperbola_G x y := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1948_194845


namespace NUMINAMATH_CALUDE_same_distance_different_speeds_l1948_194879

/-- Proves that given Joann's average speed and time, Fran needs to ride at a specific speed to cover the same distance in her given time -/
theorem same_distance_different_speeds (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) 
  (h1 : joann_speed = 15) 
  (h2 : joann_time = 4)
  (h3 : fran_time = 5) :
  joann_speed * joann_time = (60 / fran_time) * fran_time :=
by sorry

end NUMINAMATH_CALUDE_same_distance_different_speeds_l1948_194879


namespace NUMINAMATH_CALUDE_cheburashka_count_l1948_194830

/-- Represents the number of characters in a row -/
def n : ℕ := 16

/-- Represents the total number of Krakozyabras -/
def total_krakozyabras : ℕ := 29

/-- Represents the number of Cheburashkas -/
def num_cheburashkas : ℕ := 11

theorem cheburashka_count :
  (2 * (n - 1) = total_krakozyabras) ∧
  (num_cheburashkas * 2 + (num_cheburashkas - 1) * 2 + num_cheburashkas = n) :=
by sorry

#check cheburashka_count

end NUMINAMATH_CALUDE_cheburashka_count_l1948_194830


namespace NUMINAMATH_CALUDE_complex_real_condition_l1948_194892

theorem complex_real_condition (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / Complex.I
  z.im = 0 → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l1948_194892


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1948_194888

def set_A : Set ℝ := {x | |x| < 3}
def set_B : Set ℝ := {x | x^2 - 4*x + 3 < 0}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x | 1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1948_194888


namespace NUMINAMATH_CALUDE_room_area_ratio_problem_l1948_194841

/-- Proof of room area ratio problem -/
theorem room_area_ratio_problem (original_length original_width increase : ℕ) 
  (total_area : ℕ) (num_equal_rooms : ℕ) :
  let new_length : ℕ := original_length + increase
  let new_width : ℕ := original_width + increase
  let equal_room_area : ℕ := new_length * new_width
  let total_equal_rooms_area : ℕ := num_equal_rooms * equal_room_area
  let largest_room_area : ℕ := total_area - total_equal_rooms_area
  original_length = 13 ∧ 
  original_width = 18 ∧ 
  increase = 2 ∧
  total_area = 1800 ∧
  num_equal_rooms = 4 →
  largest_room_area / equal_room_area = 2 := by
sorry

end NUMINAMATH_CALUDE_room_area_ratio_problem_l1948_194841


namespace NUMINAMATH_CALUDE_count_possible_D_values_l1948_194811

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Checks if a list of digits are all distinct -/
def all_distinct (digits : List Digit) : Prop :=
  ∀ i j, i ≠ j → digits.get i ≠ digits.get j

/-- Converts a list of digits to a natural number -/
def to_nat (digits : List Digit) : ℕ :=
  digits.foldl (λ acc d => 10 * acc + d.val) 0

/-- The main theorem -/
theorem count_possible_D_values :
  ∃ (possible_D_values : Finset Digit),
    (∀ A B C E D : Digit,
      all_distinct [A, B, C, E, D] →
      to_nat [A, B, C, E, B] + to_nat [B, C, E, D, A] = to_nat [D, B, D, D, D] →
      D ∈ possible_D_values) ∧
    possible_D_values.card = 7 := by
  sorry

end NUMINAMATH_CALUDE_count_possible_D_values_l1948_194811


namespace NUMINAMATH_CALUDE_prob_not_green_correct_l1948_194846

/-- Given odds for pulling a green marble from a bag -/
def green_marble_odds : ℚ := 5 / 6

/-- The probability of not pulling a green marble -/
def prob_not_green : ℚ := 6 / 11

/-- Theorem stating that given the odds for pulling a green marble,
    the probability of not pulling a green marble is correct -/
theorem prob_not_green_correct :
  green_marble_odds = 5 / 6 →
  prob_not_green = 6 / 11 := by
sorry

end NUMINAMATH_CALUDE_prob_not_green_correct_l1948_194846


namespace NUMINAMATH_CALUDE_infinitely_many_squarefree_n_squared_plus_one_l1948_194877

/-- A natural number is squarefree if it has no repeated prime factors -/
def IsSquarefree (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ^ 2 ∣ n → p = 2)

/-- The set of positive integers n for which n^2 + 1 is squarefree -/
def SquarefreeSet : Set ℕ := {n : ℕ | n > 0 ∧ IsSquarefree (n^2 + 1)}

theorem infinitely_many_squarefree_n_squared_plus_one : Set.Infinite SquarefreeSet := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_squarefree_n_squared_plus_one_l1948_194877


namespace NUMINAMATH_CALUDE_union_of_sets_l1948_194865

theorem union_of_sets (a : ℝ) : 
  let A : Set ℝ := {x | x ≥ 0}
  let B : Set ℝ := {x | x ≤ a}
  (A ∪ B = Set.univ) → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l1948_194865


namespace NUMINAMATH_CALUDE_equation_solution_l1948_194858

theorem equation_solution : 
  ∃ x : ℝ, x = (8 * Real.sqrt 2) / 3 ∧ 
  Real.sqrt (9 + Real.sqrt (16 + 3*x)) + Real.sqrt (3 + Real.sqrt (4 + x)) = 3 + 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1948_194858


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1948_194893

theorem partial_fraction_decomposition (A B : ℚ) :
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 →
    (B * x - 19) / (x^2 - 8*x + 15) = A / (x - 3) + 5 / (x - 5)) →
  A + B = 33/5 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1948_194893


namespace NUMINAMATH_CALUDE_f_2x_equals_x_plus_1_over_x_minus_1_l1948_194821

theorem f_2x_equals_x_plus_1_over_x_minus_1 
  (x : ℝ) 
  (h : x^2 ≠ 4) : 
  let f := fun (y : ℝ) => (y + 2) / (y - 2)
  f (2 * x) = (x + 1) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_f_2x_equals_x_plus_1_over_x_minus_1_l1948_194821


namespace NUMINAMATH_CALUDE_quadrilateral_area_theorem_l1948_194822

-- Define the triangle and its division
structure DividedTriangle where
  total_area : ℝ
  triangle1_area : ℝ
  triangle2_area : ℝ
  triangle3_area : ℝ
  quadrilateral_area : ℝ
  division_valid : 
    total_area = triangle1_area + triangle2_area + triangle3_area + quadrilateral_area

-- State the theorem
theorem quadrilateral_area_theorem (t : DividedTriangle) 
  (h1 : t.triangle1_area = 4)
  (h2 : t.triangle2_area = 9)
  (h3 : t.triangle3_area = 9) :
  t.quadrilateral_area = 36 := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_area_theorem_l1948_194822


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1948_194850

theorem system_solution_ratio (x y a b : ℝ) 
  (eq1 : 8 * x - 6 * y = a)
  (eq2 : 9 * y - 12 * x = b)
  (x_nonzero : x ≠ 0)
  (y_nonzero : y ≠ 0)
  (b_nonzero : b ≠ 0) :
  a / b = -2 / 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1948_194850


namespace NUMINAMATH_CALUDE_min_y_value_l1948_194886

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 20*x + 26*y) : 
  ∃ (y_min : ℝ), y_min = 13 - Real.sqrt 269 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 = 20*x' + 26*y' → y' ≥ y_min := by
sorry

end NUMINAMATH_CALUDE_min_y_value_l1948_194886


namespace NUMINAMATH_CALUDE_marta_average_earnings_l1948_194810

/-- Represents Marta's work and earnings on her grandparent's farm --/
structure FarmWork where
  total_collected : ℕ
  task_a_rate : ℕ
  task_b_rate : ℕ
  task_c_rate : ℕ
  tips : ℕ
  task_a_hours : ℕ
  task_b_hours : ℕ
  task_c_hours : ℕ

/-- Calculates the average hourly earnings including tips --/
def average_hourly_earnings (work : FarmWork) : ℚ :=
  work.total_collected / (work.task_a_hours + work.task_b_hours + work.task_c_hours)

/-- Theorem stating that Marta's average hourly earnings, including tips, is $16 per hour --/
theorem marta_average_earnings :
  let work := FarmWork.mk 240 12 10 8 50 3 5 7
  average_hourly_earnings work = 16 := by
  sorry


end NUMINAMATH_CALUDE_marta_average_earnings_l1948_194810


namespace NUMINAMATH_CALUDE_number_of_hens_l1948_194815

theorem number_of_hens (total_animals : ℕ) (total_feet : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) 
  (h1 : total_animals = 44)
  (h2 : total_feet = 140)
  (h3 : hen_feet = 2)
  (h4 : cow_feet = 4) :
  ∃ (hens cows : ℕ), 
    hens + cows = total_animals ∧ 
    hen_feet * hens + cow_feet * cows = total_feet ∧
    hens = 18 := by
  sorry

end NUMINAMATH_CALUDE_number_of_hens_l1948_194815


namespace NUMINAMATH_CALUDE_ellipse_properties_l1948_194853

-- Define the ellipse C
def ellipse_C (x y : ℝ) (a : ℝ) : Prop :=
  x^2 / a^2 + y^2 / (7 - a^2) = 1 ∧ a > 0

-- Define the focal distance
def focal_distance (a : ℝ) : ℝ := 2

-- Define the standard form of the ellipse
def standard_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define a line passing through (4,0)
def line_through_R (x y : ℝ) (k : ℝ) : Prop :=
  y = k * (x - 4)

-- Define a point on the ellipse
def point_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define the right focus
def right_focus : (ℝ × ℝ) := (1, 0)

-- Theorem statement
theorem ellipse_properties (a : ℝ) (k : ℝ) (x1 y1 x2 y2 : ℝ) :
  (∀ x y, ellipse_C x y a ↔ standard_ellipse x y) ∧
  (∀ x y, line_through_R x y k ∧ point_on_ellipse x y →
    ∃ xn yn xq yq,
      point_on_ellipse xn yn ∧
      point_on_ellipse xq yq ∧
      xn = x1 ∧ yn = -y1 ∧
      (yn - y2) * (xq - right_focus.1) = (yq - right_focus.2) * (xn - right_focus.1)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1948_194853


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1948_194826

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x - 3| + |x + 1| > a) → a < 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1948_194826


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1948_194835

theorem average_speed_calculation (total_distance : ℝ) (first_half_distance : ℝ) (second_half_distance : ℝ) 
  (first_half_speed : ℝ) (second_half_speed : ℝ) : 
  total_distance = 50 →
  first_half_distance = 25 →
  second_half_distance = 25 →
  first_half_speed = 66 →
  second_half_speed = 33 →
  (total_distance / ((first_half_distance / first_half_speed) + (second_half_distance / second_half_speed))) = 44 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1948_194835


namespace NUMINAMATH_CALUDE_great_eighteen_games_l1948_194816

/-- Great Eighteen Soccer League -/
structure SoccerLeague where
  divisions : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculate the total number of games in the league -/
def total_games (league : SoccerLeague) : Nat :=
  let total_teams := league.divisions * league.teams_per_division
  let intra_games := (league.divisions * league.teams_per_division * (league.teams_per_division - 1) * league.intra_division_games) / 2
  let inter_games := (total_teams * (total_teams - league.teams_per_division) * league.inter_division_games) / 2
  intra_games + inter_games

/-- Theorem: The Great Eighteen Soccer League has 351 scheduled games -/
theorem great_eighteen_games :
  let league := SoccerLeague.mk 3 6 3 2
  total_games league = 351 := by
  sorry

end NUMINAMATH_CALUDE_great_eighteen_games_l1948_194816


namespace NUMINAMATH_CALUDE_x_value_l1948_194881

theorem x_value : ∃ x : ℝ, 3 * x - 48.2 = 0.25 * (4 * x + 56.8) ∧ x = 31.2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1948_194881


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1948_194878

/-- Given a quadratic function f(x) = x^2 + ax + b, if f(1) = 0 and f(2) = 0, then f(-1) = 6 -/
theorem quadratic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + a*x + b
  (f 1 = 0) → (f 2 = 0) → (f (-1) = 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1948_194878


namespace NUMINAMATH_CALUDE_rosie_pies_from_27_apples_l1948_194876

/-- Represents the number of pies Rosie can make given a certain number of apples -/
def pies_from_apples (apples : ℕ) : ℚ :=
  (2 : ℚ) * apples / 9

theorem rosie_pies_from_27_apples :
  pies_from_apples 27 = 6 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pies_from_27_apples_l1948_194876


namespace NUMINAMATH_CALUDE_derivative_inequality_l1948_194833

theorem derivative_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) (h1 : ∀ x, HasDerivAt f (f' x) x)
  (h2 : ∀ x, f' x + f x > 0) (x₁ x₂ : ℝ) (h3 : x₁ < x₂) :
  Real.exp x₁ * f x₁ < Real.exp x₂ * f x₂ := by
  sorry

end NUMINAMATH_CALUDE_derivative_inequality_l1948_194833


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1948_194899

/-- Trajectory M in the Cartesian plane -/
def trajectory_M (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1 ∧ x ≠ 2 ∧ x ≠ -2

/-- Line l in the Cartesian plane -/
def line_l (k m x y : ℝ) : Prop :=
  y = k * x + m

/-- Intersection points of line l and trajectory M -/
def intersection_points (k m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    trajectory_M x₁ y₁ ∧ trajectory_M x₂ y₂ ∧
    line_l k m x₁ y₁ ∧ line_l k m x₂ y₂ ∧
    x₁ ≠ x₂

/-- Angle condition for F₂P and F₂Q -/
def angle_condition (k m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    trajectory_M x₁ y₁ ∧ trajectory_M x₂ y₂ ∧
    line_l k m x₁ y₁ ∧ line_l k m x₂ y₂ ∧
    (y₁ - 0) / (x₁ - 1) + (y₂ - 0) / (x₂ - 1) = 0

theorem line_passes_through_fixed_point :
  ∀ k m : ℝ,
    k ≠ 0 →
    intersection_points k m →
    angle_condition k m →
    ∃ x y : ℝ, x = 4 ∧ y = 0 ∧ line_l k m x y :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1948_194899


namespace NUMINAMATH_CALUDE_power_function_sum_l1948_194891

/-- Given a power function f(x) = kx^α that passes through the point (1/2, √2),
    prove that k + α = 1/2 -/
theorem power_function_sum (k α : ℝ) (h : k * (1/2)^α = Real.sqrt 2) : k + α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_sum_l1948_194891


namespace NUMINAMATH_CALUDE_pen_price_relationship_l1948_194859

/-- Represents the relationship between the number of pens and their selling price. -/
theorem pen_price_relationship (x y : ℝ) : 
  (∀ (box_pens : ℝ) (box_price : ℝ), box_pens = 10 ∧ box_price = 16 → 
    y = (box_price / box_pens) * x) → 
  y = 1.6 * x := by
  sorry

end NUMINAMATH_CALUDE_pen_price_relationship_l1948_194859


namespace NUMINAMATH_CALUDE_johnsonville_marching_band_max_size_l1948_194880

theorem johnsonville_marching_band_max_size :
  ∀ m : ℕ,
  (∃ k : ℕ, 30 * m = 34 * k + 2) →
  30 * m < 1500 →
  (∀ n : ℕ, (∃ j : ℕ, 30 * n = 34 * j + 2) → 30 * n < 1500 → 30 * n ≤ 30 * m) →
  30 * m = 1260 :=
by sorry

end NUMINAMATH_CALUDE_johnsonville_marching_band_max_size_l1948_194880


namespace NUMINAMATH_CALUDE_range_of_expression_l1948_194883

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem range_of_expression (a b : ℕ) 
  (ha : isPrime a ∧ 49 < a ∧ a < 61) 
  (hb : isPrime b ∧ 59 < b ∧ b < 71) : 
  -297954 ≤ (a^2 : ℤ) - b^3 ∧ (a^2 : ℤ) - b^3 ≤ -223500 :=
sorry

end NUMINAMATH_CALUDE_range_of_expression_l1948_194883


namespace NUMINAMATH_CALUDE_extended_triangle_similarity_l1948_194812

/-- Triangle ABC with given side lengths --/
structure Triangle :=
  (AB : ℝ)
  (BC : ℝ)
  (CA : ℝ)

/-- Extended triangle PABC --/
structure ExtendedTriangle extends Triangle :=
  (PC : ℝ)

/-- Similarity of triangles PAB and PCA --/
def is_similar (t : ExtendedTriangle) : Prop :=
  t.PC / t.AB = t.CA / t.PC

theorem extended_triangle_similarity (t : ExtendedTriangle) 
  (h1 : t.AB = 8)
  (h2 : t.BC = 7)
  (h3 : t.CA = 6)
  (h4 : is_similar t) :
  t.PC = 9 := by
  sorry

end NUMINAMATH_CALUDE_extended_triangle_similarity_l1948_194812


namespace NUMINAMATH_CALUDE_coffee_shop_sales_teas_sold_l1948_194868

/-- The number of teas sold at a coffee shop -/
def num_teas : ℕ := 6

/-- The number of lattes sold at a coffee shop -/
def num_lattes : ℕ := 32

/-- Theorem stating the relationship between lattes and teas sold -/
theorem coffee_shop_sales : num_lattes = 4 * num_teas + 8 := by
  sorry

/-- Theorem proving the number of teas sold -/
theorem teas_sold : num_teas = 6 := by
  sorry

end NUMINAMATH_CALUDE_coffee_shop_sales_teas_sold_l1948_194868


namespace NUMINAMATH_CALUDE_fred_cantelopes_count_l1948_194863

/-- The number of cantelopes grown by Fred and Tim together -/
def total_cantelopes : ℕ := 82

/-- The number of cantelopes grown by Tim -/
def tim_cantelopes : ℕ := 44

/-- The number of cantelopes grown by Fred -/
def fred_cantelopes : ℕ := total_cantelopes - tim_cantelopes

theorem fred_cantelopes_count : fred_cantelopes = 38 := by
  sorry

end NUMINAMATH_CALUDE_fred_cantelopes_count_l1948_194863


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l1948_194842

/-- A normal distribution with mean 54 and 3 standard deviations below the mean greater than 47 has a standard deviation less than 2.33 -/
theorem normal_distribution_std_dev (σ : ℝ) 
  (h1 : 54 - 3 * σ > 47) : 
  σ < 2.33 := by
sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l1948_194842


namespace NUMINAMATH_CALUDE_star_value_l1948_194837

theorem star_value (x : ℤ) : 45 - (28 - (37 - (15 - x))) = 59 → x = -154 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l1948_194837


namespace NUMINAMATH_CALUDE_josiah_hans_age_ratio_l1948_194864

theorem josiah_hans_age_ratio :
  ∀ (hans_age josiah_age : ℕ),
    hans_age = 15 →
    josiah_age + 3 + hans_age + 3 = 66 →
    josiah_age / hans_age = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_josiah_hans_age_ratio_l1948_194864


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l1948_194849

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^3 - x + 1 > 0) ↔ (∀ x : ℝ, x^3 - x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l1948_194849


namespace NUMINAMATH_CALUDE_consecutive_sum_product_l1948_194824

theorem consecutive_sum_product (a b c : ℕ) : 
  (a + 1 = b) → (b + 1 = c) → (a + b + c = 48) → (a * c = 255) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_product_l1948_194824


namespace NUMINAMATH_CALUDE_smallest_absolute_value_is_zero_l1948_194823

theorem smallest_absolute_value_is_zero : 
  ∀ q : ℚ, |0| ≤ |q| :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_is_zero_l1948_194823


namespace NUMINAMATH_CALUDE_range_of_a_l1948_194831

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2 * |x - 1| + |x - a| ≥ 2) ↔ a ∈ Set.Iic (-1) ∪ Set.Ici 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1948_194831


namespace NUMINAMATH_CALUDE_multiply_32519_9999_l1948_194818

theorem multiply_32519_9999 : 32519 * 9999 = 324857481 := by
  sorry

end NUMINAMATH_CALUDE_multiply_32519_9999_l1948_194818


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l1948_194882

-- Problem 1
theorem problem_1 : -|-2/3 - 3/2| - |-1/5 + (-2/5)| = -83/30 := by sorry

-- Problem 2
theorem problem_2 : (-7.33) * 42.07 + (-2.07) * (-7.33) = -293.2 := by sorry

-- Problem 3
theorem problem_3 : -4 - 28 - (-19) + (-24) = -37 := by sorry

-- Problem 4
theorem problem_4 : -|-2023| - (-2023) + 2023 = 2023 := by sorry

-- Problem 5
theorem problem_5 : 19 * (31/32) * (-4) = -79 - 7/8 := by sorry

-- Problem 6
theorem problem_6 : (1/2 + 5/6 - 7/12) * (-36) = -27 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l1948_194882


namespace NUMINAMATH_CALUDE_arithmetic_sequence_and_max_lambda_l1948_194889

def sequence_a : ℕ → ℚ
  | 0 => 2
  | n + 1 => 2 - 1 / sequence_a n

def sequence_b : ℕ → ℚ
  | 0 => 20 * sequence_a 3
  | n + 1 => sequence_a n * sequence_b n

def T (n : ℕ) : ℚ := (List.range n).map sequence_b |>.sum

theorem arithmetic_sequence_and_max_lambda :
  (∀ n : ℕ, 1 / (sequence_a n - 1) = n + 1) ∧
  (∀ n : ℕ+, 2 * T n + 400 ≥ 225 * n) ∧
  (∀ ε > 0, ∃ n : ℕ+, 2 * T n + 400 < (225 + ε) * n) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_and_max_lambda_l1948_194889


namespace NUMINAMATH_CALUDE_distinct_products_count_l1948_194848

def S : Finset ℕ := {2, 3, 5, 7, 13}

def products (s : Finset ℕ) : Finset ℕ :=
  (Finset.powerset s).filter (λ t => t.card ≥ 2) |>.image (λ t => t.prod id)

theorem distinct_products_count : (products S).card = 26 := by
  sorry

end NUMINAMATH_CALUDE_distinct_products_count_l1948_194848


namespace NUMINAMATH_CALUDE_expression_value_l1948_194838

theorem expression_value (x y : ℝ) (h : x / (2 * y) = 3 / 2) :
  (7 * x + 5 * y) / (x - 2 * y) = 26 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1948_194838


namespace NUMINAMATH_CALUDE_unique_monic_quadratic_l1948_194890

-- Define a monic polynomial of degree 2
def MonicQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ b c : ℝ, ∀ x, f x = x^2 + b*x + c

theorem unique_monic_quadratic (f : ℝ → ℝ) 
  (monic : MonicQuadratic f) 
  (eval_zero : f 0 = 4)
  (eval_one : f 1 = 10) :
  ∀ x, f x = x^2 + 5*x + 4 := by
sorry

end NUMINAMATH_CALUDE_unique_monic_quadratic_l1948_194890


namespace NUMINAMATH_CALUDE_total_weight_lifted_l1948_194896

def weight_per_hand : ℕ := 8
def number_of_hands : ℕ := 2

theorem total_weight_lifted : weight_per_hand * number_of_hands = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_lifted_l1948_194896
