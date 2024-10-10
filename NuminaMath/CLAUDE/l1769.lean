import Mathlib

namespace prob_800_to_1000_l1769_176952

/-- Probability that a light bulb works after 800 hours -/
def prob_800 : ℝ := 0.8

/-- Probability that a light bulb works after 1000 hours -/
def prob_1000 : ℝ := 0.5

/-- Theorem stating the probability of a light bulb continuing to work from 800 to 1000 hours -/
theorem prob_800_to_1000 : (prob_1000 / prob_800 : ℝ) = 5/8 := by sorry

end prob_800_to_1000_l1769_176952


namespace mouse_jump_difference_l1769_176983

/-- Proves that the mouse jumped 12 inches less than the frog in the jumping contest. -/
theorem mouse_jump_difference (grasshopper_jump : ℕ) (grasshopper_frog_diff : ℕ) (mouse_jump : ℕ)
  (h1 : grasshopper_jump = 39)
  (h2 : grasshopper_frog_diff = 19)
  (h3 : mouse_jump = 8) :
  grasshopper_jump - grasshopper_frog_diff - mouse_jump = 12 := by
  sorry

end mouse_jump_difference_l1769_176983


namespace books_sold_l1769_176900

theorem books_sold (initial_books : ℕ) (added_books : ℕ) (final_books : ℕ) : 
  initial_books = 4 → added_books = 10 → final_books = 11 → 
  ∃ (sold_books : ℕ), initial_books - sold_books + added_books = final_books ∧ sold_books = 3 :=
by sorry

end books_sold_l1769_176900


namespace square_root_equation_l1769_176975

theorem square_root_equation (x : ℝ) : 
  Real.sqrt (x - 3) = 5 → x = 28 := by sorry

end square_root_equation_l1769_176975


namespace investment_plans_count_l1769_176982

/-- The number of ways to distribute projects among cities --/
def distribute_projects (n_projects : ℕ) (n_cities : ℕ) (max_per_city : ℕ) : ℕ := sorry

/-- Theorem statement --/
theorem investment_plans_count :
  distribute_projects 3 4 2 = 60 := by sorry

end investment_plans_count_l1769_176982


namespace b_income_percentage_over_c_l1769_176957

/-- Prove that B's monthly income is 12% more than C's monthly income given the specified conditions --/
theorem b_income_percentage_over_c (a_annual_income b_monthly_income c_monthly_income : ℚ) : 
  c_monthly_income = 16000 →
  a_annual_income = 537600 →
  a_annual_income / 12 / b_monthly_income = 5 / 2 →
  (b_monthly_income - c_monthly_income) / c_monthly_income = 12 / 100 := by
  sorry

end b_income_percentage_over_c_l1769_176957


namespace min_problems_for_45_points_l1769_176903

/-- Represents the possible point values for each problem -/
inductive PointValue
  | Three
  | Eight
  | Ten

/-- Represents a solution to the olympiad problem -/
structure Solution :=
  (threes : Nat)
  (eights : Nat)
  (tens : Nat)

/-- Calculates the total points for a given solution -/
def totalPoints (s : Solution) : Nat :=
  3 * s.threes + 8 * s.eights + 10 * s.tens

/-- Calculates the total number of problems solved for a given solution -/
def totalProblems (s : Solution) : Nat :=
  s.threes + s.eights + s.tens

/-- Defines a valid solution that achieves exactly 45 points -/
def isValidSolution (s : Solution) : Prop :=
  totalPoints s = 45

/-- Theorem stating that the minimum number of problems to achieve 45 points is 6 -/
theorem min_problems_for_45_points :
  ∃ (s : Solution), isValidSolution s ∧
  (∀ (s' : Solution), isValidSolution s' → totalProblems s ≤ totalProblems s') ∧
  totalProblems s = 6 :=
sorry

end min_problems_for_45_points_l1769_176903


namespace sphere_surface_area_doubling_l1769_176990

/-- Given a sphere whose surface area doubles when its radius is doubled,
    prove that if the new surface area is 9856 cm², 
    then the original surface area is 2464 cm². -/
theorem sphere_surface_area_doubling (r : ℝ) :
  (4 * Real.pi * (2 * r)^2 = 9856) → (4 * Real.pi * r^2 = 2464) := by
  sorry

end sphere_surface_area_doubling_l1769_176990


namespace expression_undefined_at_ten_expression_undefined_when_denominator_zero_l1769_176942

/-- The expression is not defined when x = 10 -/
theorem expression_undefined_at_ten : 
  ∀ x : ℝ, x = 10 → (x^3 - 30*x^2 + 300*x - 1000 = 0) := by
  sorry

/-- The denominator of the expression -/
def denominator (x : ℝ) : ℝ := x^3 - 30*x^2 + 300*x - 1000

/-- The expression is undefined when the denominator is zero -/
theorem expression_undefined_when_denominator_zero (x : ℝ) : 
  denominator x = 0 → ¬∃y : ℝ, y = (3*x^4 + 2*x + 6) / (x^3 - 30*x^2 + 300*x - 1000) := by
  sorry

end expression_undefined_at_ten_expression_undefined_when_denominator_zero_l1769_176942


namespace range_of_2a_plus_3b_l1769_176904

theorem range_of_2a_plus_3b (a b : ℝ) 
  (h1 : -1 < a + b) (h2 : a + b < 3) 
  (h3 : 2 < a - b) (h4 : a - b < 4) : 
  ∀ x, (2*a + 3*b = x) → (-9/2 < x ∧ x < 13/2) :=
sorry

end range_of_2a_plus_3b_l1769_176904


namespace tiger_escape_distance_l1769_176921

/-- Represents the speed and duration of each phase of the tiger's escape --/
structure EscapePhase where
  speed : ℝ
  duration : ℝ

/-- Calculates the total distance traveled by the tiger --/
def totalDistance (phases : List EscapePhase) : ℝ :=
  phases.foldl (fun acc phase => acc + phase.speed * phase.duration) 0

/-- The escape phases of the tiger --/
def tigerEscapePhases : List EscapePhase := [
  { speed := 25, duration := 1 },
  { speed := 35, duration := 2 },
  { speed := 20, duration := 1.5 },
  { speed := 10, duration := 1 },
  { speed := 50, duration := 0.5 }
]

theorem tiger_escape_distance :
  totalDistance tigerEscapePhases = 160 := by
  sorry

end tiger_escape_distance_l1769_176921


namespace walts_age_inconsistency_l1769_176916

theorem walts_age_inconsistency :
  ¬ ∃ (w : ℕ), 
    (3 * w + 12 = 2 * (w + 12)) ∧ 
    (4 * w + 15 = 3 * (w + 15)) := by
  sorry

end walts_age_inconsistency_l1769_176916


namespace composition_value_l1769_176944

/-- Given two functions g and h, prove that their composition at x = 2 equals 3890 -/
theorem composition_value :
  let g (x : ℝ) := 3 * x^2 + 2
  let h (x : ℝ) := -5 * x^3 + 4
  g (h 2) = 3890 := by
  sorry

end composition_value_l1769_176944


namespace larger_cube_volume_l1769_176966

-- Define the number of smaller cubes
def num_small_cubes : ℕ := 343

-- Define the volume of each smaller cube
def small_cube_volume : ℝ := 1

-- Define the surface area difference
def surface_area_difference : ℝ := 1764

-- Theorem statement
theorem larger_cube_volume :
  let large_cube_side : ℝ := (num_small_cubes : ℝ) ^ (1/3)
  let small_cube_side : ℝ := small_cube_volume ^ (1/3)
  let large_cube_volume : ℝ := large_cube_side ^ 3
  (num_small_cubes : ℝ) * (6 * small_cube_side ^ 2) - (6 * large_cube_side ^ 2) = surface_area_difference →
  large_cube_volume = num_small_cubes * small_cube_volume :=
by
  sorry

end larger_cube_volume_l1769_176966


namespace sufficient_but_not_necessary_l1769_176923

theorem sufficient_but_not_necessary (a : ℝ) :
  (∀ x : ℝ, x > 1 → x ≠ 1) ∧ (∃ y : ℝ, y ≠ 1 ∧ ¬(y > 1)) :=
by sorry

end sufficient_but_not_necessary_l1769_176923


namespace min_value_expression_l1769_176933

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * (a + b + c) + b * c = 4 - 2 * Real.sqrt 3) :
  2 * a + b + c ≥ 2 * Real.sqrt 3 - 2 :=
by sorry

end min_value_expression_l1769_176933


namespace function_composition_equality_l1769_176947

/-- Given f(x) = x/3 + 4 and g(x) = 7 - x, if f(g(a)) = 6, then a = 1 -/
theorem function_composition_equality (f g : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = x / 3 + 4)
  (hg : ∀ x, g x = 7 - x)
  (h : f (g a) = 6) : 
  a = 1 := by sorry

end function_composition_equality_l1769_176947


namespace ali_flower_sales_l1769_176955

def flower_problem (monday_sales : ℕ) (friday_multiplier : ℕ) (total_sales : ℕ) : Prop :=
  let friday_sales := friday_multiplier * monday_sales
  let tuesday_sales := total_sales - monday_sales - friday_sales
  tuesday_sales = 8

theorem ali_flower_sales : flower_problem 4 2 20 := by
  sorry

end ali_flower_sales_l1769_176955


namespace problem_solution_l1769_176943

def p (a : ℝ) : Prop := ∀ x ≥ 1, x - x^2 ≤ a

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - a*x + 1 = 0

theorem problem_solution (a : ℝ) :
  (¬(¬(p a)) → a ≥ 0) ∧
  ((¬(p a ∧ q a) ∧ (p a ∨ q a)) → (a ≤ -2 ∨ (0 ≤ a ∧ a < 2))) :=
by sorry

end problem_solution_l1769_176943


namespace fifteenth_term_is_negative_one_l1769_176936

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  first_term : ℤ
  common_diff : ℤ

/-- The nth term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.first_term + (n - 1 : ℤ) * seq.common_diff

theorem fifteenth_term_is_negative_one
  (seq : ArithmeticSequence)
  (h21 : nth_term seq 21 = 17)
  (h22 : nth_term seq 22 = 20) :
  nth_term seq 15 = -1 := by
  sorry

end fifteenth_term_is_negative_one_l1769_176936


namespace sequence_repeat_value_l1769_176911

theorem sequence_repeat_value (p q n : ℕ+) (x : Fin (n + 1) → ℤ)
  (h1 : p + q < n)
  (h2 : x 0 = 0 ∧ x n = 0)
  (h3 : ∀ i : Fin n, x (i + 1) - x i = p ∨ x (i + 1) - x i = -q) :
  ∃ i j : Fin (n + 1), i < j ∧ (i, j) ≠ (0, n) ∧ x i = x j := by
  sorry

end sequence_repeat_value_l1769_176911


namespace passengers_on_time_l1769_176934

theorem passengers_on_time (total : ℕ) (late : ℕ) (h1 : total = 14720) (h2 : late = 213) :
  total - late = 14507 := by
  sorry

end passengers_on_time_l1769_176934


namespace square_area_problem_l1769_176963

theorem square_area_problem (s : ℝ) : 
  (0.8 * s) * (5 * s) = s^2 + 15.18 → s^2 = 5.06 := by sorry

end square_area_problem_l1769_176963


namespace line_intersection_bound_l1769_176951

/-- Given points A(2,7) and B(9,6) in the Cartesian plane, and a line y = kx (k ≠ 0) that
    intersects the line segment AB, prove that k is bounded by 2/3 ≤ k ≤ 7/2. -/
theorem line_intersection_bound (k : ℝ) : k ≠ 0 → 
  (∃ x y : ℝ, x ∈ Set.Icc 2 9 ∧ y ∈ Set.Icc 6 7 ∧ y = k * x ∧ y - 7 = (6 - 7) / (9 - 2) * (x - 2)) →
  2/3 ≤ k ∧ k ≤ 7/2 := by
  sorry

end line_intersection_bound_l1769_176951


namespace symmetry_point_l1769_176912

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- A line in the form y = mx + b -/
structure Line where
  m : ℚ
  b : ℚ

/-- Check if two points are symmetric with respect to a line -/
def areSymmetric (P Q : Point) (l : Line) : Prop :=
  -- The product of the slopes of PQ and l is -1
  ((Q.y - P.y) / (Q.x - P.x)) * l.m = -1 ∧
  -- The midpoint of PQ lies on l
  ((Q.y + P.y) / 2) = l.m * ((Q.x + P.x) / 2) + l.b

theorem symmetry_point :
  let P : Point := ⟨-1, 2⟩
  let Q : Point := ⟨7/5, 4/5⟩
  let l : Line := ⟨2, 1⟩  -- y = 2x + 1
  areSymmetric P Q l := by sorry

end symmetry_point_l1769_176912


namespace absolute_value_equation_simplification_l1769_176941

theorem absolute_value_equation_simplification
  (a b c : ℝ)
  (h1 : ∀ x : ℝ, |5*x - 4| + a ≠ 0)
  (h2 : ∃ x y : ℝ, x ≠ y ∧ |4*x - 3| + b = 0 ∧ |4*y - 3| + b = 0)
  (h3 : ∃! x : ℝ, |3*x - 2| + c = 0) :
  |a - c| + |c - b| - |a - b| = 0 := by
sorry

end absolute_value_equation_simplification_l1769_176941


namespace arrangement_count_l1769_176967

/-- The number of ways to choose 2 items from a set of 4 items -/
def choose_2_from_4 : ℕ := 6

/-- The number of ways to arrange 3 items -/
def arrange_3 : ℕ := 6

/-- The total number of arrangements -/
def total_arrangements : ℕ := choose_2_from_4 * arrange_3

/-- Theorem: The number of ways to arrange 4 letters from the set {a, b, c, d, e, f},
    where a and b must be selected and adjacent (with a in front of b), is equal to 36 -/
theorem arrangement_count : total_arrangements = 36 := by
  sorry

end arrangement_count_l1769_176967


namespace cone_vertex_angle_l1769_176964

noncomputable def vertex_angle_third_cone : ℝ := 2 * Real.arcsin (1/4)

theorem cone_vertex_angle 
  (first_two_cones_identical : Bool)
  (fourth_cone_internal : Bool)
  (first_two_cones_half_fourth : Bool) :
  ∃ (α : ℝ), 
    α = π/6 + Real.arcsin (1/4) ∧ 
    α > 0 ∧ 
    α < π/2 ∧
    2 * α = vertex_angle_third_cone ∧
    first_two_cones_identical = true ∧
    fourth_cone_internal = true ∧
    first_two_cones_half_fourth = true :=
by sorry

end cone_vertex_angle_l1769_176964


namespace winning_candidate_percentage_l1769_176920

def election_votes : List Nat := [1000, 2000, 4000]

theorem winning_candidate_percentage :
  let total_votes := election_votes.sum
  let winning_votes := election_votes.maximum?
  winning_votes.map (λ w => (w : ℚ) / total_votes * 100) = some (4000 / 7000 * 100) := by
  sorry

end winning_candidate_percentage_l1769_176920


namespace camp_cedar_counselors_l1769_176959

/-- Calculates the number of counselors needed for a camp --/
def counselors_needed (num_boys : ℕ) (girls_to_boys_ratio : ℕ) (children_per_counselor : ℕ) : ℕ :=
  let num_girls := num_boys * girls_to_boys_ratio
  let total_children := num_boys + num_girls
  total_children / children_per_counselor

/-- Proves that Camp Cedar needs 20 counselors --/
theorem camp_cedar_counselors :
  counselors_needed 40 3 8 = 20 := by
  sorry

end camp_cedar_counselors_l1769_176959


namespace complement_A_union_B_l1769_176992

def A : Set Int := {x | ∃ k : Int, x = 3 * k + 1}
def B : Set Int := {x | ∃ k : Int, x = 3 * k + 2}
def U : Set Int := Set.univ

theorem complement_A_union_B :
  (A ∪ B)ᶜ = {x : Int | ∃ k : Int, x = 3 * k} :=
sorry

end complement_A_union_B_l1769_176992


namespace arithmetic_to_geometric_sequence_l1769_176919

/-- 
Given an arithmetic sequence {a_n} with a₁ = -8 and a₂ = -6,
if x is added to a₁, a₄, and a₅ to form a geometric sequence,
then x = -1.
-/
theorem arithmetic_to_geometric_sequence (a : ℕ → ℤ) (x : ℤ) : 
  a 1 = -8 →
  a 2 = -6 →
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →
  ((-8 + x) * x = (-2 + x)^2) →
  ((-2 + x)^2 = x * x) →
  x = -1 := by sorry

end arithmetic_to_geometric_sequence_l1769_176919


namespace fraction_irreducible_l1769_176932

theorem fraction_irreducible (n : ℤ) : Int.gcd (21*n + 4) (14*n + 3) = 1 := by
  sorry

end fraction_irreducible_l1769_176932


namespace common_root_theorem_l1769_176999

theorem common_root_theorem (a b c d : ℝ) (h1 : a + d = 2017) (h2 : b + c = 2017) :
  ∃ x : ℝ, x = 2017 / 2 ∧ (x - a) * (x - b) = (x - c) * (x - d) :=
by sorry

end common_root_theorem_l1769_176999


namespace expand_polynomial_l1769_176935

theorem expand_polynomial (x : ℝ) : 
  (x - 3) * (x + 3) * (x^2 + 9) * (x - 1) = x^5 - x^4 - 81*x + 81 := by
sorry

end expand_polynomial_l1769_176935


namespace concert_audience_fraction_l1769_176949

/-- The fraction of the audience for the second band at a concert -/
def fraction_second_band (total_audience : ℕ) (under_30_percent : ℚ) 
  (women_percent : ℚ) (men_under_30 : ℕ) : ℚ :=
  2 / 3

theorem concert_audience_fraction 
  (total_audience : ℕ) 
  (under_30_percent : ℚ) 
  (women_percent : ℚ) 
  (men_under_30 : ℕ) : 
  fraction_second_band total_audience under_30_percent women_percent men_under_30 = 2 / 3 :=
by
  sorry

#check concert_audience_fraction 150 (1/2) (3/5) 20

end concert_audience_fraction_l1769_176949


namespace ellipse_properties_l1769_176925

-- Define the ellipse
def ellipse (x y m : ℝ) : Prop := x^2 / 25 + y^2 / m^2 = 1

-- Define the focus
def left_focus (x y : ℝ) : Prop := x = -4 ∧ y = 0

-- Define eccentricity
def eccentricity (e : ℝ) (a c : ℝ) : Prop := e = c / a

theorem ellipse_properties (m : ℝ) (h : m > 0) :
  (∃ x y, ellipse x y m ∧ left_focus x y) →
  m = 3 ∧ ∃ e, eccentricity e 5 4 ∧ e = 4/5 :=
by sorry

end ellipse_properties_l1769_176925


namespace parallelogram_reflection_l1769_176906

/-- Reflect a point across the x-axis -/
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Reflect a point across the line y = -x -/
def reflect_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, -p.1)

/-- The final position of point C after two reflections -/
def final_position (C : ℝ × ℝ) : ℝ × ℝ :=
  reflect_y_neg_x (reflect_x_axis C)

theorem parallelogram_reflection :
  let A : ℝ × ℝ := (2, 5)
  let B : ℝ × ℝ := (4, 9)
  let C : ℝ × ℝ := (6, 5)
  let D : ℝ × ℝ := (4, 1)
  final_position C = (5, -6) := by sorry

end parallelogram_reflection_l1769_176906


namespace minimize_distance_sum_l1769_176979

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculate the angle at vertex B of a triangle -/
def angle_at_vertex (t : Triangle) (v : Point) : ℝ := sorry

/-- Check if a point is inside a triangle -/
def is_inside (p : Point) (t : Triangle) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- The main theorem about the point that minimizes the sum of distances -/
theorem minimize_distance_sum (t : Triangle) : 
  (∀ v, angle_at_vertex t v < 120) → 
    ∃ O, is_inside O t ∧ 
      ∀ P, is_inside P t → 
        distance O t.A + distance O t.B + distance O t.C ≤ 
        distance P t.A + distance P t.B + distance P t.C 
  ∧ 
  (∃ v, angle_at_vertex t v ≥ 120) → 
    ∃ v, angle_at_vertex t v ≥ 120 ∧ 
      ∀ P, is_inside P t → 
        distance v t.A + distance v t.B + distance v t.C ≤ 
        distance P t.A + distance P t.B + distance P t.C :=
sorry

end minimize_distance_sum_l1769_176979


namespace hamburger_combinations_l1769_176913

/-- The number of available condiments -/
def num_condiments : ℕ := 8

/-- The number of choices for meat patties -/
def num_patty_choices : ℕ := 4

/-- The total number of hamburger combinations -/
def total_combinations : ℕ := 2^num_condiments * num_patty_choices

theorem hamburger_combinations :
  total_combinations = 1024 :=
by sorry

end hamburger_combinations_l1769_176913


namespace units_digit_of_17_to_1995_l1769_176931

theorem units_digit_of_17_to_1995 : (17 ^ 1995 : ℕ) % 10 = 3 := by
  sorry

end units_digit_of_17_to_1995_l1769_176931


namespace arithmetic_calculation_l1769_176997

theorem arithmetic_calculation : 8 / 2 - 3 - 9 + 3 * 9 - 3^2 = 10 := by
  sorry

end arithmetic_calculation_l1769_176997


namespace cotton_planting_rate_l1769_176961

/-- Calculates the required acres per tractor per day to plant cotton --/
theorem cotton_planting_rate (total_acres : ℕ) (total_days : ℕ) 
  (tractors_first_period : ℕ) (days_first_period : ℕ)
  (tractors_second_period : ℕ) (days_second_period : ℕ) :
  total_acres = 1700 →
  total_days = 5 →
  tractors_first_period = 2 →
  days_first_period = 2 →
  tractors_second_period = 7 →
  days_second_period = 3 →
  (total_acres : ℚ) / ((tractors_first_period * days_first_period + 
    tractors_second_period * days_second_period) : ℚ) = 68 := by
  sorry

#eval (1700 : ℚ) / 25  -- Should output 68

end cotton_planting_rate_l1769_176961


namespace consecutive_numbers_divisible_by_2014_l1769_176910

theorem consecutive_numbers_divisible_by_2014 :
  ∃ (n : ℕ), n < 96 ∧ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 2014 = 0 := by
  sorry

end consecutive_numbers_divisible_by_2014_l1769_176910


namespace isosceles_triangles_count_l1769_176940

/-- A triangle represented by its three vertices in 2D space -/
structure Triangle where
  v1 : (Int × Int)
  v2 : (Int × Int)
  v3 : (Int × Int)

/-- Check if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Bool :=
  let d12 := (t.v1.1 - t.v2.1)^2 + (t.v1.2 - t.v2.2)^2
  let d23 := (t.v2.1 - t.v3.1)^2 + (t.v2.2 - t.v3.2)^2
  let d31 := (t.v3.1 - t.v1.1)^2 + (t.v3.2 - t.v1.2)^2
  d12 = d23 || d23 = d31 || d31 = d12

/-- The list of triangles from the problem -/
def triangles : List Triangle := [
  { v1 := (0, 8), v2 := (2, 8), v3 := (1, 6) },
  { v1 := (3, 5), v2 := (3, 8), v3 := (6, 5) },
  { v1 := (0, 2), v2 := (4, 3), v3 := (8, 2) },
  { v1 := (7, 5), v2 := (6, 8), v3 := (10, 5) },
  { v1 := (7, 2), v2 := (8, 4), v3 := (10, 1) },
  { v1 := (3, 1), v2 := (5, 1), v3 := (4, 3) }
]

theorem isosceles_triangles_count : 
  (triangles.filter isIsosceles).length = 5 := by sorry

end isosceles_triangles_count_l1769_176940


namespace trigonometric_identity_30_degrees_l1769_176994

theorem trigonometric_identity_30_degrees : 
  (Real.tan (30 * π / 180))^2 - (Real.sin (30 * π / 180))^2 = 
  (Real.tan (30 * π / 180))^2 * (Real.sin (30 * π / 180))^2 := by
  sorry

end trigonometric_identity_30_degrees_l1769_176994


namespace limit_cubic_difference_quotient_l1769_176908

theorem limit_cubic_difference_quotient :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → |(x^3 - 1) / (x - 1) - 3| < ε :=
by sorry

end limit_cubic_difference_quotient_l1769_176908


namespace sufficient_not_necessary_condition_l1769_176991

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x > y ∧ y > 0 → x^2 > y^2) ∧
  ∃ x y : ℝ, x^2 > y^2 ∧ ¬(x > y ∧ y > 0) :=
by sorry

end sufficient_not_necessary_condition_l1769_176991


namespace divisible_by_1968_l1769_176998

theorem divisible_by_1968 (n : ℕ) : ∃ k : ℤ, 
  (-1)^(2*n) + 9^(4*n) - 6^(8*n) + 8^(16*n) = 1968 * k := by
  sorry

end divisible_by_1968_l1769_176998


namespace sum_of_fractions_inequality_l1769_176987

theorem sum_of_fractions_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + c) / (a + b) + (b + d) / (b + c) + (c + a) / (c + d) + (d + b) / (d + a) ≥ 4 := by
  sorry

end sum_of_fractions_inequality_l1769_176987


namespace justin_jersey_cost_l1769_176924

/-- The total cost of jerseys bought by Justin -/
def total_cost (long_sleeve_count : ℕ) (long_sleeve_price : ℕ) (striped_count : ℕ) (striped_price : ℕ) : ℕ :=
  long_sleeve_count * long_sleeve_price + striped_count * striped_price

/-- Theorem stating that Justin's total cost for jerseys is $80 -/
theorem justin_jersey_cost :
  total_cost 4 15 2 10 = 80 := by
  sorry

end justin_jersey_cost_l1769_176924


namespace largest_number_proof_l1769_176974

theorem largest_number_proof (w x y z : ℕ) : 
  w + x + y = 190 ∧ 
  w + x + z = 210 ∧ 
  w + y + z = 220 ∧ 
  x + y + z = 235 → 
  max w (max x (max y z)) = 95 := by
sorry

end largest_number_proof_l1769_176974


namespace units_digit_of_product_division_l1769_176950

theorem units_digit_of_product_division : 
  (12 * 13 * 14 * 15 * 16 * 17) / 2000 % 10 = 6 :=
by sorry

end units_digit_of_product_division_l1769_176950


namespace total_face_masks_produced_l1769_176922

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Represents the duration of Manolo's shift in hours -/
def shift_duration : ℕ := 4

/-- Represents the time to make one face-mask in the first hour (in minutes) -/
def first_hour_rate : ℕ := 4

/-- Represents the time to make one face-mask after the first hour (in minutes) -/
def subsequent_rate : ℕ := 6

/-- Calculates the number of face-masks made in the first hour -/
def first_hour_production : ℕ := minutes_per_hour / first_hour_rate

/-- Calculates the number of face-masks made in the subsequent hours -/
def subsequent_hours_production : ℕ := (shift_duration - 1) * minutes_per_hour / subsequent_rate

/-- Theorem: The total number of face-masks produced in a four-hour shift is 45 -/
theorem total_face_masks_produced :
  first_hour_production + subsequent_hours_production = 45 := by
  sorry

end total_face_masks_produced_l1769_176922


namespace geometric_sequence_general_term_l1769_176915

/-- Given a geometric sequence {a_n} where the first three terms are x, x-1, and 2x-2 respectively,
    prove that the general term is a_n = -2^(n-1) -/
theorem geometric_sequence_general_term (a : ℕ → ℝ) (x : ℝ) (h1 : a 1 = x) (h2 : a 2 = x - 1) (h3 : a 3 = 2*x - 2) :
  ∀ n : ℕ, a n = -2^(n-1) := by
sorry

end geometric_sequence_general_term_l1769_176915


namespace G_equals_4F_l1769_176971

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := F ((4 * x - x^3) / (1 + 4 * x^2))

theorem G_equals_4F (x : ℝ) : G x = 4 * F x :=
  sorry

end G_equals_4F_l1769_176971


namespace invalid_external_diagonals_l1769_176985

/-- Represents a right regular prism with external diagonal lengths a, b, and c --/
structure RightRegularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- Theorem stating that {3, 4, 6} cannot be the lengths of external diagonals of a right regular prism --/
theorem invalid_external_diagonals (p : RightRegularPrism) :
  p.a = 3 ∧ p.b = 4 ∧ p.c = 6 → False := by
  sorry

#check invalid_external_diagonals

end invalid_external_diagonals_l1769_176985


namespace parallel_perpendicular_transitivity_l1769_176938

/-- A structure representing a 3D space with lines and planes -/
structure Space3D where
  Line : Type
  Plane : Type
  parallel_lines : Line → Line → Prop
  perpendicular_line_plane : Line → Plane → Prop

/-- The theorem stating that if a line is parallel to another line that is perpendicular to a plane, 
    then the first line is also perpendicular to that plane -/
theorem parallel_perpendicular_transitivity 
  {S : Space3D} {m n : S.Line} {α : S.Plane} :
  S.parallel_lines m n → S.perpendicular_line_plane m α → S.perpendicular_line_plane n α :=
sorry

end parallel_perpendicular_transitivity_l1769_176938


namespace students_not_in_biology_l1769_176945

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) 
  (h1 : total_students = 880)
  (h2 : biology_percentage = 40 / 100) :
  ↑total_students * (1 - biology_percentage) = 528 := by
  sorry

end students_not_in_biology_l1769_176945


namespace alex_age_difference_l1769_176914

/-- Proves the number of years ago when Alex was one-third as old as his father -/
theorem alex_age_difference (alex_current_age : ℝ) (alex_father_age : ℝ) (years_ago : ℝ) : 
  alex_current_age = 16.9996700066 →
  alex_father_age = 2 * alex_current_age + 5 →
  alex_current_age - years_ago = (1 / 3) * (alex_father_age - years_ago) →
  years_ago = 6.4998350033 := by
sorry

end alex_age_difference_l1769_176914


namespace cube_shadow_problem_l1769_176996

theorem cube_shadow_problem (x : ℝ) : 
  let cube_edge : ℝ := 2
  let shadow_area : ℝ := 300
  let total_shadow_area : ℝ := shadow_area + cube_edge^2
  let shadow_side : ℝ := Real.sqrt total_shadow_area
  x = (cube_edge / (shadow_side - cube_edge)) →
  ⌊1000 * x⌋ = 706 := by
sorry

end cube_shadow_problem_l1769_176996


namespace acceptable_outfits_l1769_176980

/-- The number of shirts, pants, and hats available -/
def num_items : ℕ := 8

/-- The number of colors available for each item -/
def num_colors : ℕ := 8

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_items^3

/-- The number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- The number of outfits where shirt and pants are the same color but hat is different -/
def shirt_pants_same : ℕ := num_colors * (num_colors - 1)

/-- Theorem stating the number of acceptable outfit combinations -/
theorem acceptable_outfits : 
  total_combinations - same_color_outfits - shirt_pants_same = 448 := by
  sorry

end acceptable_outfits_l1769_176980


namespace range_of_a_l1769_176993

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Define the statement that ¬p is a necessary but not sufficient condition for ¬q
def not_p_necessary_not_sufficient_for_not_q (a : ℝ) : Prop :=
  (∀ x, q x → p x a) ∧ (∃ x, ¬q x ∧ p x a)

-- Main theorem
theorem range_of_a :
  ∀ a : ℝ, a > 0 → not_p_necessary_not_sufficient_for_not_q a → 1 < a ∧ a ≤ 2 :=
sorry

end range_of_a_l1769_176993


namespace investment_rate_proof_l1769_176946

theorem investment_rate_proof (total_investment : ℝ) (first_investment : ℝ) (second_investment : ℝ)
  (first_rate : ℝ) (second_rate : ℝ) (desired_income : ℝ) :
  total_investment = 12000 ∧
  first_investment = 5000 ∧
  second_investment = 4000 ∧
  first_rate = 0.05 ∧
  second_rate = 0.035 ∧
  desired_income = 600 →
  ∃ (remaining_rate : ℝ),
    remaining_rate = 0.07 ∧
    (total_investment - first_investment - second_investment) * remaining_rate +
    first_investment * first_rate + second_investment * second_rate = desired_income :=
by sorry

end investment_rate_proof_l1769_176946


namespace actual_distance_traveled_l1769_176988

/-- Given two speeds and an additional distance, proves that the actual distance traveled is 50 km -/
theorem actual_distance_traveled (speed1 speed2 additional_distance : ℝ) 
  (h1 : speed1 = 10)
  (h2 : speed2 = 14)
  (h3 : additional_distance = 20)
  (h4 : ∀ D : ℝ, D / speed1 = (D + additional_distance) / speed2) :
  ∃ D : ℝ, D = 50 := by
  sorry

end actual_distance_traveled_l1769_176988


namespace equation_solution_l1769_176930

def solution_set : Set (ℕ × ℕ) :=
  {(0, 1), (1, 1), (3, 25), (4, 31), (5, 41), (8, 85)}

theorem equation_solution :
  {(a, b) : ℕ × ℕ | a * b + 2 = a ^ 3 + 2 * b} = solution_set := by
  sorry

end equation_solution_l1769_176930


namespace solve_equation_l1769_176917

theorem solve_equation (x : ℚ) : (3 * x - 2) / 4 = 14 → x = 58 / 3 := by
  sorry

end solve_equation_l1769_176917


namespace checkers_draw_fraction_l1769_176909

theorem checkers_draw_fraction (dan_wins eve_wins : ℚ) (h1 : dan_wins = 4/9) (h2 : eve_wins = 1/3) :
  1 - (dan_wins + eve_wins) = 2/9 := by
sorry

end checkers_draw_fraction_l1769_176909


namespace planes_parallel_or_intersect_l1769_176956

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- Parallel relation between lines -/
def Line.parallel (l1 l2 : Line) : Prop := sorry

/-- A line is contained in a plane -/
def Line.contained_in (l : Line) (p : Plane) : Prop := sorry

/-- Two planes are parallel -/
def Plane.parallel (p1 p2 : Plane) : Prop := sorry

/-- Two planes intersect -/
def Plane.intersect (p1 p2 : Plane) : Prop := sorry

/-- Main theorem: Given the conditions, planes α and β are either parallel or intersecting -/
theorem planes_parallel_or_intersect (α β : Plane) (a b c : Line) 
  (h1 : a.parallel b) (h2 : b.parallel c)
  (h3 : a.contained_in α) (h4 : b.contained_in β) (h5 : c.contained_in β) :
  Plane.parallel α β ∨ Plane.intersect α β := by sorry

end planes_parallel_or_intersect_l1769_176956


namespace log_identity_l1769_176962

-- Define the logarithm base 5
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- State the theorem
theorem log_identity : (log5 (3 * log5 25))^2 = (1 + log5 1.2)^2 := by sorry

end log_identity_l1769_176962


namespace price_reduction_for_same_profit_no_solution_for_460_profit_l1769_176939

/-- Represents the fruit sales scenario at Huimin Fresh Supermarket -/
structure FruitSales where
  cost_price : ℝ
  initial_selling_price : ℝ
  initial_daily_sales : ℝ
  sales_increase_rate : ℝ

/-- Calculates the daily profit given a price reduction -/
def daily_profit (fs : FruitSales) (price_reduction : ℝ) : ℝ :=
  (fs.initial_selling_price - price_reduction - fs.cost_price) *
  (fs.initial_daily_sales + fs.sales_increase_rate * price_reduction)

/-- The scenario described in the problem -/
def huimin_scenario : FruitSales := {
  cost_price := 20
  initial_selling_price := 40
  initial_daily_sales := 20
  sales_increase_rate := 2
}

theorem price_reduction_for_same_profit :
  daily_profit huimin_scenario 10 = daily_profit huimin_scenario 0 := by sorry

theorem no_solution_for_460_profit :
  ∀ x : ℝ, daily_profit huimin_scenario x ≠ 460 := by sorry

end price_reduction_for_same_profit_no_solution_for_460_profit_l1769_176939


namespace prob_at_least_three_out_of_five_is_half_l1769_176954

def probability_at_least_three_out_of_five : ℚ :=
  let n : ℕ := 5  -- total number of games
  let p : ℚ := 1/2  -- probability of winning a single game
  let winning_prob : ℕ → ℚ := λ k => Nat.choose n k * p^k * (1-p)^(n-k)
  (winning_prob 3) + (winning_prob 4) + (winning_prob 5)

theorem prob_at_least_three_out_of_five_is_half :
  probability_at_least_three_out_of_five = 1/2 := by
  sorry

end prob_at_least_three_out_of_five_is_half_l1769_176954


namespace debby_candy_l1769_176901

def initial_candy : ℕ → ℕ → ℕ
  | remaining, eaten => remaining + eaten

theorem debby_candy (remaining eaten : ℕ) 
  (h1 : remaining = 3) 
  (h2 : eaten = 9) : 
  initial_candy remaining eaten = 12 := by
  sorry

end debby_candy_l1769_176901


namespace quotient_problem_l1769_176978

theorem quotient_problem (k : ℤ) (h : k = 4) : 16 / k = 4 := by
  sorry

end quotient_problem_l1769_176978


namespace sum_even_digits_1_to_200_l1769_176902

/-- E(n) represents the sum of even digits in the number n -/
def E (n : ℕ) : ℕ := sorry

/-- The sum of E(n) for n from 1 to 200 -/
def sumE : ℕ := (Finset.range 200).sum E + E 200

theorem sum_even_digits_1_to_200 : sumE = 800 := by sorry

end sum_even_digits_1_to_200_l1769_176902


namespace function_inequality_l1769_176927

theorem function_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x y, x < y ∧ y ≤ 2 → f x < f y) 
  (h2 : ∀ x, f (-x + 2) = f (x + 2)) : 
  f (-1) < f 3 := by
sorry

end function_inequality_l1769_176927


namespace recurrence_sequence_uniqueness_l1769_176953

/-- A sequence defined by a recurrence relation -/
def RecurrenceSequence (p q : ℝ) (a₀ a₁ : ℝ) : ℕ → ℝ
| 0 => a₀
| 1 => a₁
| (n + 2) => p * RecurrenceSequence p q a₀ a₁ (n + 1) + q * RecurrenceSequence p q a₀ a₁ n

/-- Theorem: All terms in the sequence are uniquely determined -/
theorem recurrence_sequence_uniqueness (p q : ℝ) (a₀ a₁ : ℝ) :
  ∀ n : ℕ, ∃! x : ℝ, x = RecurrenceSequence p q a₀ a₁ n :=
by sorry

end recurrence_sequence_uniqueness_l1769_176953


namespace fraction_equality_l1769_176984

theorem fraction_equality (p q r : ℕ+) 
  (h : (p : ℚ) + 1 / ((q : ℚ) + 1 / (r : ℚ)) = 25 / 19) : 
  q = 3 := by
  sorry

end fraction_equality_l1769_176984


namespace right_triangle_properties_l1769_176972

/-- A right triangle with sides 9 cm and 12 cm -/
structure RightTriangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  area : ℝ
  side1_eq : side1 = 9
  side2_eq : side2 = 12
  pythagorean : side1^2 + side2^2 = hypotenuse^2
  area_formula : area = (1/2) * side1 * side2

/-- The hypotenuse of the right triangle is 15 cm and its area is 54 cm² -/
theorem right_triangle_properties (t : RightTriangle) : t.hypotenuse = 15 ∧ t.area = 54 := by
  sorry

#check right_triangle_properties

end right_triangle_properties_l1769_176972


namespace problem_statement_l1769_176907

theorem problem_statement : |Real.sqrt 3 - 2| + 2 * Real.sin (60 * π / 180) - 2023^0 = 1 := by
  sorry

end problem_statement_l1769_176907


namespace symmetry_implies_t_zero_l1769_176977

/-- Line l in the Cartesian coordinate system -/
def line_l (x y : ℝ) : Prop :=
  8 * x + 6 * y + 1 = 0

/-- Circle C₁ in the Cartesian coordinate system -/
def circle_C1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 8*x - 2*y + 13 = 0

/-- Circle C₂ in the Cartesian coordinate system -/
def circle_C2 (t x y : ℝ) : Prop :=
  x^2 + y^2 + 8*t*x - 8*y + 16*t + 12 = 0

/-- The center of circle C₁ -/
def center_C1 : ℝ × ℝ :=
  (-4, 1)

/-- The center of circle C₂ -/
def center_C2 (t : ℝ) : ℝ × ℝ :=
  (-4*t, 4)

/-- Theorem: When circle C₁ and circle C₂ are symmetric about line l, t = 0 -/
theorem symmetry_implies_t_zero :
  ∀ t : ℝ, (∃ x y : ℝ, line_l x y ∧ 
    ((x - (-4))^2 + (y - 1)^2 = (x - (-4*t))^2 + (y - 4)^2) ∧
    ((8*x + 6*y + 1 = 0) → 
      ((-4 + (-4*t))/2 = x ∧ (1 + 4)/2 = y))) →
  t = 0 := by
  sorry

end symmetry_implies_t_zero_l1769_176977


namespace intersection_area_is_zero_l1769_176970

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle defined by three vertices -/
structure Triangle :=
  (v1 : Point)
  (v2 : Point)
  (v3 : Point)

/-- Calculate the area of intersection between two triangles -/
def areaOfIntersection (t1 t2 : Triangle) : ℝ := sorry

/-- The main theorem stating that the area of intersection is zero -/
theorem intersection_area_is_zero :
  let t1 := Triangle.mk (Point.mk 0 2) (Point.mk 2 1) (Point.mk 0 0)
  let t2 := Triangle.mk (Point.mk 2 2) (Point.mk 0 1) (Point.mk 2 0)
  areaOfIntersection t1 t2 = 0 := by sorry

end intersection_area_is_zero_l1769_176970


namespace sum_of_a_and_b_is_one_l1769_176918

theorem sum_of_a_and_b_is_one :
  ∀ (a b : ℝ),
  (∃ (x : ℝ), x = a + Real.sqrt b) →
  (a + Real.sqrt b + (a - Real.sqrt b) = -4) →
  ((a + Real.sqrt b) * (a - Real.sqrt b) = 1) →
  a + b = 1 :=
by sorry

end sum_of_a_and_b_is_one_l1769_176918


namespace vector_equality_implies_x_value_l1769_176928

/-- Given that the vector (x+3, x^2-3x-4) is equal to (2, 0), prove that x = -1 -/
theorem vector_equality_implies_x_value : 
  ∀ x : ℝ, (x + 3 = 2 ∧ x^2 - 3*x - 4 = 0) → x = -1 := by
  sorry

end vector_equality_implies_x_value_l1769_176928


namespace final_price_percentage_l1769_176981

/-- Calculates the final price after applying multiple discounts -/
def final_price (original_price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (λ price discount => price * (1 - discount)) original_price

/-- Theorem stating that after applying the given discounts, the final price is 58.14% of the original -/
theorem final_price_percentage (original_price : ℝ) (original_price_pos : 0 < original_price) :
  let discounts := [0.2, 0.1, 0.05, 0.15]
  final_price original_price discounts / original_price = 0.5814 := by
sorry

#eval (final_price 100 [0.2, 0.1, 0.05, 0.15])

end final_price_percentage_l1769_176981


namespace quadrilateral_area_quadrilateral_area_with_given_diagonal_and_offsets_l1769_176995

/-- The area of a quadrilateral with a diagonal of length d and offsets h₁ and h₂ is (d * h₁ + d * h₂) / 2 -/
theorem quadrilateral_area (d h₁ h₂ : ℝ) (h_d_pos : d > 0) (h_h₁_pos : h₁ > 0) (h_h₂_pos : h₂ > 0) :
  (d * h₁ + d * h₂) / 2 = d * (h₁ + h₂) / 2 :=
by sorry

theorem quadrilateral_area_with_given_diagonal_and_offsets :
  let diagonal : ℝ := 20
  let offset1 : ℝ := 5
  let offset2 : ℝ := 4
  (diagonal * offset1 + diagonal * offset2) / 2 = 90 :=
by sorry

end quadrilateral_area_quadrilateral_area_with_given_diagonal_and_offsets_l1769_176995


namespace current_rate_calculation_l1769_176976

/-- Given a boat with speed in still water and its downstream travel distance and time,
    calculate the rate of the current. -/
theorem current_rate_calculation (boat_speed : ℝ) (downstream_distance : ℝ) (travel_time : ℝ) :
  boat_speed = 20 →
  downstream_distance = 6.25 →
  travel_time = 0.25 →
  ∃ (current_rate : ℝ),
    current_rate = 5 ∧
    downstream_distance = (boat_speed + current_rate) * travel_time :=
by
  sorry


end current_rate_calculation_l1769_176976


namespace remaining_sticker_sheets_l1769_176958

theorem remaining_sticker_sheets 
  (initial_stickers : ℕ) 
  (shared_stickers : ℕ) 
  (stickers_per_sheet : ℕ) 
  (h1 : initial_stickers = 150) 
  (h2 : shared_stickers = 100) 
  (h3 : stickers_per_sheet = 10) 
  (h4 : stickers_per_sheet > 0) : 
  (initial_stickers - shared_stickers) / stickers_per_sheet = 5 := by
sorry

end remaining_sticker_sheets_l1769_176958


namespace simplify_expressions_l1769_176968

/-- Prove the simplification of two algebraic expressions -/
theorem simplify_expressions (x y : ℝ) :
  (7 * x + 3 * (x^2 - 2) - 3 * (1/2 * x^2 - x + 3) = 3/2 * x^2 + 10 * x - 15) ∧
  (3 * (2 * x^2 * y - x * y^2) - 4 * (-x * y^2 + 3 * x^2 * y) = -6 * x^2 * y + x * y^2) :=
by sorry

end simplify_expressions_l1769_176968


namespace expression_bounds_l1769_176969

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) : 
  2 * Real.sqrt 2 ≤ Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
    Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-a)^2) ∧
  Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
    Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-a)^2) ≤ 4 := by
  sorry

end expression_bounds_l1769_176969


namespace trapezoid_perimeter_l1769_176948

-- Define the trapezoid EFGH
structure Trapezoid :=
  (EF : ℝ) (GH : ℝ) (height : ℝ)

-- Define the properties of the trapezoid
def isIsoscelesTrapezoid (t : Trapezoid) : Prop :=
  t.EF = t.GH

-- Theorem statement
theorem trapezoid_perimeter 
  (t : Trapezoid) 
  (h1 : isIsoscelesTrapezoid t) 
  (h2 : t.height = 5) 
  (h3 : t.GH = 10) 
  (h4 : t.EF = 4) : 
  ∃ (perimeter : ℝ), perimeter = 14 + 2 * Real.sqrt 34 :=
by
  sorry

end trapezoid_perimeter_l1769_176948


namespace folded_paper_triangle_perimeter_l1769_176929

/-- A square piece of paper with side length 2 is folded such that vertex C meets edge AB at point C',
    making C'B = 2/3. Edge BC intersects edge AD at point E. -/
theorem folded_paper_triangle_perimeter :
  ∀ (A B C D C' E : ℝ × ℝ),
    -- Square conditions
    A = (0, 2) ∧ B = (0, 0) ∧ C = (2, 0) ∧ D = (2, 2) →
    -- Folding conditions
    C' = (0, 4/3) →
    -- Intersection condition
    E = (2, 0) →
    -- Perimeter calculation
    dist A E + dist E C' + dist C' A = 4 :=
by sorry

end folded_paper_triangle_perimeter_l1769_176929


namespace power_sum_constant_implies_zero_or_one_l1769_176926

/-- Given a natural number n > 1 and a list of real numbers x,
    if the sum of the k-th powers of these numbers is constant for k from 1 to n+1,
    then each number in the list is either 0 or 1. -/
theorem power_sum_constant_implies_zero_or_one (n : ℕ) (x : List ℝ) :
  n > 1 →
  x.length = n →
  (∀ k : ℕ, k ≥ 1 → k ≤ n + 1 →
    (List.sum (List.map (fun xi => xi ^ k) x)) = (List.sum (List.map (fun xi => xi ^ 1) x))) →
  ∀ xi ∈ x, xi = 0 ∨ xi = 1 := by
  sorry

end power_sum_constant_implies_zero_or_one_l1769_176926


namespace parabola_line_intersection_ratio_l1769_176986

/-- Parabola type -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Line passing through a point with given slope angle -/
structure Line where
  slope_angle : ℝ
  point : ℝ × ℝ

/-- Intersection points of a line and a parabola -/
structure Intersection where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Theorem stating the ratio of distances from intersection points to focus -/
theorem parabola_line_intersection_ratio
  (C : Parabola)
  (l : Line)
  (i : Intersection)
  (h1 : l.slope_angle = π / 3) -- 60 degrees in radians
  (h2 : l.point = (C.p / 2, 0)) -- Focus of the parabola
  (h3 : i.A.1 > 0 ∧ i.A.2 > 0) -- A in first quadrant
  (h4 : i.B.1 > 0 ∧ i.B.2 < 0) -- B in fourth quadrant
  (h5 : i.A.2^2 = 2 * C.p * i.A.1) -- A satisfies parabola equation
  (h6 : i.B.2^2 = 2 * C.p * i.B.1) -- B satisfies parabola equation
  (h7 : i.A.2 - 0 = Real.sqrt 3 * (i.A.1 - C.p / 2)) -- A satisfies line equation
  (h8 : i.B.2 - 0 = Real.sqrt 3 * (i.B.1 - C.p / 2)) -- B satisfies line equation
  : (Real.sqrt ((i.A.1 - C.p / 2)^2 + i.A.2^2)) / (Real.sqrt ((i.B.1 - C.p / 2)^2 + i.B.2^2)) = 4 :=
sorry

end parabola_line_intersection_ratio_l1769_176986


namespace chicken_wings_distribution_l1769_176989

theorem chicken_wings_distribution (total_friends : ℕ) (initial_wings : ℕ) (cooked_wings : ℕ) (non_eating_friends : ℕ) :
  total_friends = 15 →
  initial_wings = 7 →
  cooked_wings = 45 →
  non_eating_friends = 2 →
  (initial_wings + cooked_wings) / (total_friends - non_eating_friends) = 4 :=
by sorry

end chicken_wings_distribution_l1769_176989


namespace even_periodic_function_monotonicity_l1769_176965

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x > f y

theorem even_periodic_function_monotonicity (f : ℝ → ℝ)
  (h_even : is_even f) (h_period : has_period f 2) :
  increasing_on f 0 1 ↔ decreasing_on f 3 4 := by sorry

end even_periodic_function_monotonicity_l1769_176965


namespace max_faces_convex_polyhedron_l1769_176960

/-- A convex polyhedron with n congruent triangular faces, each having angles 36°, 72°, and 72° -/
structure ConvexPolyhedron where
  n : ℕ  -- number of faces
  convex : Bool
  congruentFaces : Bool
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

/-- The maximum number of faces for the given polyhedron is 36 -/
theorem max_faces_convex_polyhedron (p : ConvexPolyhedron) 
  (h1 : p.convex = true)
  (h2 : p.congruentFaces = true)
  (h3 : p.angleA = 36)
  (h4 : p.angleB = 72)
  (h5 : p.angleC = 72) :
  p.n ≤ 36 :=
sorry

end max_faces_convex_polyhedron_l1769_176960


namespace class_average_theorem_l1769_176973

theorem class_average_theorem (total_students : ℝ) (h_total : total_students > 0) :
  let group1_percent : ℝ := 25
  let group1_average : ℝ := 80
  let group2_percent : ℝ := 50
  let group2_average : ℝ := 65
  let group3_percent : ℝ := 100 - group1_percent - group2_percent
  let group3_average : ℝ := 90
  let overall_average : ℝ := (group1_percent * group1_average + group2_percent * group2_average + group3_percent * group3_average) / 100
  overall_average = 75 := by
  sorry


end class_average_theorem_l1769_176973


namespace september_to_august_ratio_l1769_176937

def july_earnings : ℕ := 150
def august_earnings : ℕ := 3 * july_earnings
def total_earnings : ℕ := 1500

def september_earnings_ratio (x : ℚ) : Prop :=
  july_earnings + august_earnings + x * august_earnings = total_earnings

theorem september_to_august_ratio :
  ∃ x : ℚ, september_earnings_ratio x ∧ x = 2 := by
  sorry

end september_to_august_ratio_l1769_176937


namespace continued_fraction_value_l1769_176905

theorem continued_fraction_value : 
  ∃ y : ℝ, y = 3 + 5 / (1 + 5 / y) → y = 5 := by
  sorry

end continued_fraction_value_l1769_176905
